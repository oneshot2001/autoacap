/**
 * AutoACAP — C Person Detection Pipeline
 *
 * Uses VDO for video capture and Larod for ML inference on ARTPEC-9 DLPU.
 * Detects persons using SSD MobileNet V2 (INT8 quantized).
 * Uses Larod's built-in cpu-proc preprocessing for NV12→RGB conversion.
 *
 * This file is modified by the autonomous research agent.
 * The agent optimizes for maximum FPS while maintaining mAP >= 0.40.
 *
 * Based on Axis ACAP Native SDK object-detection example.
 */

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <syslog.h>
#include <time.h>
#include <unistd.h>

#include <glib.h>
#include <vdo-buffer.h>
#include <vdo-channel.h>
#include <vdo-error.h>
#include <vdo-frame.h>
#include <vdo-map.h>
#include <vdo-stream.h>
#include <vdo-types.h>

#include <larod.h>

/* ─── Configuration ─────────────────────────────────────────────────────── */

#define MODEL_PATH       "/usr/local/packages/autoacap/model/ssd_mobilenet_v2_int8.tflite"
#define METRICS_PATH     "/tmp/autoacap_metrics.json"
#define DETECTIONS_PATH  "/tmp/autoacap_detections.json"

#define STREAM_WIDTH     640
#define STREAM_HEIGHT    480
#define STREAM_FPS       15.0

#define CONFIDENCE_THRESHOLD 0.5f
#define PERSON_CLASS_ID      1
#define MAX_DETECTIONS       100
#define MAX_LATENCIES        2000

/* DLPU device name for ARTPEC-8/9 */
#define LAROD_DEVICE_NAME "axis-a8-dlpu"
/* Preprocessing device */
#define PP_DEVICE_NAME    "cpu-proc"

#define MAX_POWER_RETRIES 50

/* ─── Types ─────────────────────────────────────────────────────────────── */

typedef struct {
    float y_min, x_min, y_max, x_max;
    float confidence;
    int class_id;
} Detection;

typedef struct {
    int fd;
    void *data;
    size_t size;
    larodTensorDataType datatype;
} TensorOutput;

/* ─── Globals ───────────────────────────────────────────────────────────── */

static volatile sig_atomic_t running = 1;

static void signal_handler(int sig) {
    (void)sig;
    running = 0;
}

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ─── VDO Stream Setup ──────────────────────────────────────────────────── */

typedef struct {
    VdoStream *stream;
    int poll_fd;
    unsigned int width;
    unsigned int height;
    unsigned int pitch;
    VdoFormat format;
} VdoProvider;

static VdoProvider *vdo_provider_new(void) {
    GError *error = NULL;
    VdoProvider *vp = calloc(1, sizeof(VdoProvider));
    if (!vp) return NULL;

    VdoMap *settings = vdo_map_new();
    vdo_map_set_uint32(settings, "format", VDO_FORMAT_YUV);
    vdo_map_set_uint32(settings, "width", STREAM_WIDTH);
    vdo_map_set_uint32(settings, "height", STREAM_HEIGHT);
    vdo_map_set_double(settings, "framerate", STREAM_FPS);
    vdo_map_set_uint32(settings, "buffer.count", 2);
    vdo_map_set_boolean(settings, "socket.blocking", FALSE);

    vp->stream = vdo_stream_new(settings, NULL, &error);
    g_object_unref(settings);

    if (!vp->stream) {
        syslog(LOG_ERR, "Failed to create VDO stream: %s",
               error ? error->message : "unknown");
        if (error) g_error_free(error);
        free(vp);
        return NULL;
    }

    /* Read actual stream info */
    VdoMap *info = vdo_stream_get_info(vp->stream, &error);
    if (info) {
        vp->width = vdo_map_get_uint32(info, "width", STREAM_WIDTH);
        vp->height = vdo_map_get_uint32(info, "height", STREAM_HEIGHT);
        vp->pitch = vdo_map_get_uint32(info, "pitch", vp->width);
        vp->format = vdo_map_get_uint32(info, "format", VDO_FORMAT_YUV);
        g_object_unref(info);
    } else {
        vp->width = STREAM_WIDTH;
        vp->height = STREAM_HEIGHT;
        vp->pitch = STREAM_WIDTH;
        vp->format = VDO_FORMAT_YUV;
    }

    if (!vdo_stream_start(vp->stream, &error)) {
        syslog(LOG_ERR, "Failed to start VDO stream: %s",
               error ? error->message : "unknown");
        if (error) g_error_free(error);
        g_object_unref(vp->stream);
        free(vp);
        return NULL;
    }

    vp->poll_fd = vdo_stream_get_fd(vp->stream, &error);
    if (vp->poll_fd < 0) {
        syslog(LOG_ERR, "Failed to get stream fd");
        g_object_unref(vp->stream);
        free(vp);
        return NULL;
    }

    syslog(LOG_INFO, "VDO stream: %ux%u pitch=%u format=%u",
           vp->width, vp->height, vp->pitch, vp->format);

    return vp;
}

static VdoBuffer *vdo_provider_get_frame(VdoProvider *vp) {
    GError *error = NULL;
    struct pollfd pfd = { .fd = vp->poll_fd, .events = POLLIN };

    while (running) {
        int ret;
        do {
            ret = poll(&pfd, 1, 1000); /* 1s timeout */
        } while (ret == -1 && errno == EINTR);

        if (ret <= 0) continue;

        VdoBuffer *buf = vdo_stream_get_buffer(vp->stream, &error);
        if (buf) return buf;

        if (error && g_error_matches(error, VDO_ERROR, VDO_ERROR_NO_DATA)) {
            g_clear_error(&error);
            continue;
        }
        if (error) {
            syslog(LOG_ERR, "VDO get_buffer error: %s", error->message);
            g_clear_error(&error);
        }
        return NULL;
    }
    return NULL;
}

static void vdo_provider_destroy(VdoProvider *vp) {
    if (!vp) return;
    if (vp->stream) {
        vdo_stream_stop(vp->stream);
        g_object_unref(vp->stream);
    }
    free(vp);
}

/* ─── Larod Model Setup ─────────────────────────────────────────────────── */

typedef struct {
    larodConnection *conn;
    larodModel *model;
    larodModel *pp_model;
    int model_fd;

    /* Inference tensors */
    larodTensor **input_tensors;
    size_t num_inputs;
    larodTensor **output_tensors;
    size_t num_outputs;

    /* Preprocessing tensors */
    larodTensor **pp_input_tensors;
    size_t pp_num_inputs;
    larodTensor **pp_output_tensors;
    size_t pp_num_outputs;

    /* Job requests */
    larodJobRequest *pp_req;
    larodJobRequest *inf_req;

    /* Memory-mapped input buffer (for copying VDO frame data) */
    int image_input_fd;
    void *image_input_addr;
    size_t image_buffer_size;

    /* Memory-mapped output tensors */
    TensorOutput *outputs;

    /* Model input dimensions (from tensor metadata) */
    unsigned int model_width;
    unsigned int model_height;
} LarodProvider;

static LarodProvider *larod_provider_new(unsigned int vdo_width,
                                          unsigned int vdo_height,
                                          unsigned int vdo_pitch) {
    larodError *error = NULL;
    LarodProvider *lp = calloc(1, sizeof(LarodProvider));
    if (!lp) return NULL;
    lp->image_input_fd = -1;
    lp->model_fd = -1;

    /* Connect to larod */
    if (!larodConnect(&lp->conn, &error)) {
        syslog(LOG_ERR, "larodConnect failed: %s", error ? error->msg : "unknown");
        larodClearError(&error);
        free(lp);
        return NULL;
    }

    /* Load inference model onto DLPU */
    const larodDevice *device = larodGetDevice(lp->conn, LAROD_DEVICE_NAME, 0, &error);
    if (!device) {
        syslog(LOG_ERR, "larodGetDevice(%s) failed: %s",
               LAROD_DEVICE_NAME, error ? error->msg : "unknown");
        larodClearError(&error);
        larodDisconnect(&lp->conn, NULL);
        free(lp);
        return NULL;
    }

    lp->model_fd = open(MODEL_PATH, O_RDONLY);
    if (lp->model_fd < 0) {
        syslog(LOG_ERR, "Failed to open model: %s", MODEL_PATH);
        larodDisconnect(&lp->conn, NULL);
        free(lp);
        return NULL;
    }

    syslog(LOG_INFO, "Loading model onto DLPU (may take a moment)...");

    /* Retry model loading if power not available */
    int retries = 0;
    do {
        larodClearError(&error);
        lp->model = larodLoadModel(lp->conn, lp->model_fd, device,
                                    LAROD_ACCESS_PRIVATE,
                                    "autoacap-detection", NULL, &error);
        if (lp->model) break;
        if (error && error->code == LAROD_ERROR_POWER_NOT_AVAILABLE) {
            retries++;
            usleep(250 * 1000 * retries);
        } else {
            break;
        }
    } while (retries < MAX_POWER_RETRIES);

    if (!lp->model) {
        syslog(LOG_ERR, "larodLoadModel failed: %s", error ? error->msg : "unknown");
        larodClearError(&error);
        close(lp->model_fd);
        larodDisconnect(&lp->conn, NULL);
        free(lp);
        return NULL;
    }
    syslog(LOG_INFO, "Model loaded successfully");

    /* Allocate inference tensors */
    lp->input_tensors = larodAllocModelInputs(lp->conn, lp->model, 0,
                                               &lp->num_inputs, NULL, &error);
    if (!lp->input_tensors) {
        syslog(LOG_ERR, "Failed to alloc input tensors: %s", error ? error->msg : "unknown");
        goto fail;
    }
    lp->output_tensors = larodAllocModelOutputs(lp->conn, lp->model, 0,
                                                 &lp->num_outputs, NULL, &error);
    if (!lp->output_tensors) {
        syslog(LOG_ERR, "Failed to alloc output tensors: %s", error ? error->msg : "unknown");
        goto fail;
    }

    syslog(LOG_INFO, "Model has %zu inputs, %zu outputs", lp->num_inputs, lp->num_outputs);

    /* Get model input dimensions */
    const larodTensorDims *dims = larodGetTensorDims(lp->input_tensors[0], &error);
    if (!dims) {
        syslog(LOG_ERR, "Failed to get input dims: %s", error ? error->msg : "unknown");
        goto fail;
    }
    /* NHWC: dims[0]=batch, dims[1]=height, dims[2]=width, dims[3]=channels */
    lp->model_height = dims->dims[1];
    lp->model_width = dims->dims[2];
    syslog(LOG_INFO, "Model input: %ux%u", lp->model_width, lp->model_height);

    /* Setup preprocessing model (NV12 → RGB, resize) using cpu-proc */
    const larodDevice *pp_device = larodGetDevice(lp->conn, PP_DEVICE_NAME, 0, &error);
    if (!pp_device) {
        syslog(LOG_ERR, "larodGetDevice(%s) failed: %s",
               PP_DEVICE_NAME, error ? error->msg : "unknown");
        goto fail;
    }

    larodMap *pp_map = larodCreateMap(&error);
    if (!pp_map) goto fail;
    larodMapSetStr(pp_map, "image.input.format", "nv12", &error);
    larodMapSetIntArr2(pp_map, "image.input.size", vdo_width, vdo_height, &error);
    larodMapSetInt(pp_map, "image.input.row-pitch", vdo_pitch, &error);
    larodMapSetStr(pp_map, "image.output.format", "rgb-interleaved", &error);
    larodMapSetIntArr2(pp_map, "image.output.size", lp->model_width, lp->model_height, &error);

    lp->pp_model = larodLoadModel(lp->conn, -1, pp_device,
                                   LAROD_ACCESS_PRIVATE, "", pp_map, &error);
    larodDestroyMap(&pp_map);
    if (!lp->pp_model) {
        syslog(LOG_ERR, "Failed to load preprocessing model: %s",
               error ? error->msg : "unknown");
        goto fail;
    }

    /* Allocate preprocessing tensors */
    lp->pp_input_tensors = larodAllocModelInputs(lp->conn, lp->pp_model, 0,
                                                  &lp->pp_num_inputs, NULL, &error);
    lp->pp_output_tensors = larodAllocModelOutputs(lp->conn, lp->pp_model, 0,
                                                    &lp->pp_num_outputs, NULL, &error);
    if (!lp->pp_input_tensors || !lp->pp_output_tensors) {
        syslog(LOG_ERR, "Failed to alloc preprocessing tensors");
        goto fail;
    }

    /* Map preprocessing input tensor for VDO frame data */
    lp->image_input_fd = larodGetTensorFd(lp->pp_input_tensors[0], &error);
    if (lp->image_input_fd == LAROD_INVALID_FD) {
        syslog(LOG_ERR, "Failed to get pp input fd");
        goto fail;
    }
    if (!larodGetTensorFdSize(lp->pp_input_tensors[0], &lp->image_buffer_size, &error)) {
        syslog(LOG_ERR, "Failed to get pp input size");
        goto fail;
    }
    lp->image_input_addr = mmap(NULL, lp->image_buffer_size,
                                 PROT_READ | PROT_WRITE, MAP_SHARED,
                                 lp->image_input_fd, 0);
    if (lp->image_input_addr == MAP_FAILED) {
        syslog(LOG_ERR, "mmap pp input failed: %s", strerror(errno));
        goto fail;
    }

    syslog(LOG_INFO, "Preprocessing: NV12 %ux%u → RGB %ux%u (buffer %zu bytes)",
           vdo_width, vdo_height, lp->model_width, lp->model_height,
           lp->image_buffer_size);

    /* Create job requests */
    /* Preprocessing: pp_input → pp_output */
    lp->pp_req = larodCreateJobRequest(lp->pp_model,
                                        lp->pp_input_tensors, lp->pp_num_inputs,
                                        lp->pp_output_tensors, lp->pp_num_outputs,
                                        NULL, &error);
    if (!lp->pp_req) {
        syslog(LOG_ERR, "Failed to create pp job request: %s", error ? error->msg : "unknown");
        goto fail;
    }

    /* Inference: pp_output → model output (chaining preprocessing output to inference input) */
    lp->inf_req = larodCreateJobRequest(lp->model,
                                         lp->pp_output_tensors, lp->pp_num_outputs,
                                         lp->output_tensors, lp->num_outputs,
                                         NULL, &error);
    if (!lp->inf_req) {
        syslog(LOG_ERR, "Failed to create inf job request: %s", error ? error->msg : "unknown");
        goto fail;
    }

    /* Memory-map output tensors for reading results */
    lp->outputs = calloc(lp->num_outputs, sizeof(TensorOutput));
    for (size_t i = 0; i < lp->num_outputs; i++) {
        int fd = larodGetTensorFd(lp->output_tensors[i], &error);
        if (fd == LAROD_INVALID_FD) {
            syslog(LOG_ERR, "Failed to get output tensor %zu fd", i);
            goto fail;
        }
        lp->outputs[i].fd = fd;

        size_t sz = 0;
        if (!larodGetTensorFdSize(lp->output_tensors[i], &sz, &error)) {
            syslog(LOG_ERR, "Failed to get output tensor %zu size", i);
            goto fail;
        }
        lp->outputs[i].size = sz;

        void *addr = mmap(NULL, sz, PROT_READ, MAP_SHARED, fd, 0);
        if (addr == MAP_FAILED) {
            syslog(LOG_ERR, "mmap output tensor %zu failed", i);
            goto fail;
        }
        lp->outputs[i].data = addr;
        lp->outputs[i].datatype = larodGetTensorDataType(lp->output_tensors[i], &error);

        syslog(LOG_INFO, "Output tensor %zu: %zu bytes", i, sz);
    }

    syslog(LOG_INFO, "Larod fully initialized");
    return lp;

fail:
    larodClearError(&error);
    /* Partial cleanup — good enough for init failure */
    if (lp->model_fd >= 0) close(lp->model_fd);
    larodDisconnect(&lp->conn, NULL);
    free(lp->outputs);
    free(lp);
    return NULL;
}

static int larod_run_inference(LarodProvider *lp, const uint8_t *frame_data) {
    larodError *error = NULL;
    static int power_retries = 0;

    /* Copy VDO frame into preprocessing input buffer */
    memcpy(lp->image_input_addr, frame_data, lp->image_buffer_size);

    /* Run preprocessing (NV12 → RGB, resize) */
    if (!larodRunJob(lp->conn, lp->pp_req, &error)) {
        if (error && error->code == LAROD_ERROR_POWER_NOT_AVAILABLE) {
            larodClearError(&error);
            power_retries++;
            usleep(250 * 1000 * power_retries);
            return -1; /* Retry */
        }
        syslog(LOG_ERR, "Preprocessing failed: %s", error ? error->msg : "unknown");
        larodClearError(&error);
        return -1;
    }
    power_retries = 0;

    /* Run inference on DLPU */
    if (!larodRunJob(lp->conn, lp->inf_req, &error)) {
        if (error && error->code == LAROD_ERROR_POWER_NOT_AVAILABLE) {
            larodClearError(&error);
            power_retries++;
            usleep(250 * 1000 * power_retries);
            return -1;
        }
        syslog(LOG_ERR, "Inference failed: %s", error ? error->msg : "unknown");
        larodClearError(&error);
        return -1;
    }
    power_retries = 0;

    return 0;
}

static int larod_get_detections(LarodProvider *lp, Detection *dets, int max_dets) {
    /*
     * SSD MobileNet V2 outputs 4 tensors:
     *   [0] locations: float[N][4] — [ymin, xmin, ymax, xmax] normalized
     *   [1] classes:   float[N]    — class indices
     *   [2] scores:    float[N]    — confidence scores
     *   [3] num_dets:  float[1]    — number of valid detections
     */
    if (lp->num_outputs < 4) {
        syslog(LOG_ERR, "Expected 4 output tensors, got %zu", lp->num_outputs);
        return 0;
    }

    const float *locations = (const float *)lp->outputs[0].data;
    const float *classes   = (const float *)lp->outputs[1].data;
    const float *scores    = (const float *)lp->outputs[2].data;
    const float *num_raw   = (const float *)lp->outputs[3].data;

    int n = (int)num_raw[0];
    int count = 0;

    for (int i = 0; i < n && count < max_dets; i++) {
        int cls = (int)classes[i];
        float conf = scores[i];

        if (cls != PERSON_CLASS_ID || conf < CONFIDENCE_THRESHOLD)
            continue;

        dets[count].y_min = locations[4 * i + 0];
        dets[count].x_min = locations[4 * i + 1];
        dets[count].y_max = locations[4 * i + 2];
        dets[count].x_max = locations[4 * i + 3];
        dets[count].confidence = conf;
        dets[count].class_id = cls;
        count++;
    }

    return count;
}

static void larod_provider_destroy(LarodProvider *lp) {
    if (!lp) return;

    larodDestroyJobRequest(&lp->pp_req);
    larodDestroyJobRequest(&lp->inf_req);

    if (lp->image_input_addr && lp->image_input_addr != MAP_FAILED)
        munmap(lp->image_input_addr, lp->image_buffer_size);

    if (lp->outputs) {
        for (size_t i = 0; i < lp->num_outputs; i++) {
            if (lp->outputs[i].data && lp->outputs[i].data != MAP_FAILED)
                munmap(lp->outputs[i].data, lp->outputs[i].size);
        }
        free(lp->outputs);
    }

    larodError *error = NULL;
    larodDestroyTensors(lp->conn, &lp->pp_input_tensors, lp->pp_num_inputs, &error);
    larodDestroyTensors(lp->conn, &lp->pp_output_tensors, lp->pp_num_outputs, &error);
    larodDestroyTensors(lp->conn, &lp->input_tensors, lp->num_inputs, &error);
    larodDestroyTensors(lp->conn, &lp->output_tensors, lp->num_outputs, &error);

    larodDestroyModel(&lp->model);
    larodDestroyModel(&lp->pp_model);

    if (lp->model_fd >= 0) close(lp->model_fd);
    larodDisconnect(&lp->conn, NULL);
    free(lp);
}

/* ─── Metrics & Detection Output ────────────────────────────────────────── */

static void write_metrics(double fps, const double *latencies, int num_latencies) {
    FILE *f = fopen(METRICS_PATH, "w");
    if (!f) return;

    fprintf(f, "{\n  \"fps\": %.1f,\n  \"latencies_ms\": [", fps);
    for (int i = 0; i < num_latencies; i++) {
        if (i > 0) fprintf(f, ", ");
        fprintf(f, "%.1f", latencies[i]);
    }
    fprintf(f, "]\n}\n");
    fclose(f);
}

static void write_detection_frame(FILE *f, const Detection *dets, int num_dets,
                                   int frame_id, int is_first) {
    if (is_first) {
        fprintf(f, "{\"frames\": [\n");
    } else {
        fprintf(f, ",\n");
    }
    fprintf(f, "  {\"frame_id\": %d, \"detections\": [", frame_id);
    for (int i = 0; i < num_dets; i++) {
        if (i > 0) fprintf(f, ", ");
        float x = dets[i].x_min;
        float y = dets[i].y_min;
        float w = dets[i].x_max - dets[i].x_min;
        float h = dets[i].y_max - dets[i].y_min;
        fprintf(f, "{\"bbox\": [%.4f, %.4f, %.4f, %.4f], "
                    "\"confidence\": %.4f, \"class\": \"person\"}",
                x, y, w, h, dets[i].confidence);
    }
    fprintf(f, "]}");
}

/* ─── Main ──────────────────────────────────────────────────────────────── */

int main(void) {
    openlog("autoacap", LOG_PID, LOG_LOCAL0);
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    syslog(LOG_INFO, "AutoACAP C variant starting");
    printf("AutoACAP C variant starting...\n");

    /* Initialize VDO */
    VdoProvider *vdo = vdo_provider_new();
    if (!vdo) {
        syslog(LOG_ERR, "Failed to init VDO");
        return 1;
    }

    /* Initialize Larod with preprocessing */
    LarodProvider *larod = larod_provider_new(vdo->width, vdo->height, vdo->pitch);
    if (!larod) {
        syslog(LOG_ERR, "Failed to init Larod");
        vdo_provider_destroy(vdo);
        return 1;
    }

    /* Metrics tracking */
    double latencies[MAX_LATENCIES];
    int num_latencies = 0;
    int frame_count = 0;
    double fps = 0.0;
    double start_time = get_time_ms();

    Detection detections[MAX_DETECTIONS];
    FILE *det_file = fopen(DETECTIONS_PATH, "w");

    syslog(LOG_INFO, "Entering detection loop");
    printf("Running detection loop...\n");

    /* ─── Main detection loop ─── */
    while (running) {
        double frame_start = get_time_ms();

        /* Get frame from VDO */
        VdoBuffer *vdo_buf = vdo_provider_get_frame(vdo);
        if (!vdo_buf) continue;

        uint8_t *frame_data = (uint8_t *)vdo_buffer_get_data(vdo_buf);

        /* Run preprocessing + inference */
        int ret = larod_run_inference(larod, frame_data);

        /* Release VDO buffer immediately after copy */
        GError *vdo_err = NULL;
        vdo_stream_buffer_unref(vdo->stream, &vdo_buf, &vdo_err);
        if (vdo_err) g_error_free(vdo_err);

        if (ret < 0) continue; /* Inference failed, skip frame */

        /* Extract detections */
        int num_dets = larod_get_detections(larod, detections, MAX_DETECTIONS);

        /* Write detections for mAP */
        if (det_file) {
            write_detection_frame(det_file, detections, num_dets,
                                  frame_count, frame_count == 0);
        }

        /* Track latency */
        double latency = get_time_ms() - frame_start;
        if (num_latencies < MAX_LATENCIES) {
            latencies[num_latencies++] = latency;
        }

        frame_count++;

        /* Update FPS */
        double elapsed_s = (get_time_ms() - start_time) / 1000.0;
        if (elapsed_s > 0) fps = frame_count / elapsed_s;

        /* Log periodically */
        if (frame_count % 10 == 0) {
            write_metrics(fps, latencies, num_latencies);
            printf("\rFrames: %d | FPS: %.1f | Latency: %.1fms | Dets: %d   ",
                   frame_count, fps, latency, num_dets);
            fflush(stdout);
            syslog(LOG_INFO, "Frames: %d FPS: %.1f Latency: %.1fms Dets: %d",
                   frame_count, fps, latency, num_dets);
        }
    }

    printf("\nShutting down. Total frames: %d, FPS: %.1f\n", frame_count, fps);
    syslog(LOG_INFO, "Shutting down. Frames: %d FPS: %.1f", frame_count, fps);

    /* Close detections file */
    if (det_file) {
        fprintf(det_file, "\n]}\n");
        fclose(det_file);
    }

    /* Final metrics write */
    write_metrics(fps, latencies, num_latencies);

    /* Cleanup */
    larod_provider_destroy(larod);
    vdo_provider_destroy(vdo);
    closelog();

    return 0;
}
