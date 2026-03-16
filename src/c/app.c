/**
 * AutoACAP — C Person Detection Pipeline
 *
 * Uses VDO for video capture and Larod for ML inference on ARTPEC-9 DLPU.
 * Detects persons using SSD MobileNet V2 (INT8 quantized).
 *
 * This file is modified by the autonomous research agent.
 * The agent optimizes for maximum FPS while maintaining mAP >= 0.40.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

/* Axis ACAP APIs */
#include <vdo-stream.h>
#include <vdo-frame.h>
#include <vdo-buffer.h>
#include <larod.h>

/* Detection output format */
#define MAX_DETECTIONS 100
#define CONFIDENCE_THRESHOLD 0.5f
#define NMS_THRESHOLD 0.45f
#define INPUT_WIDTH 300
#define INPUT_HEIGHT 300
#define PERSON_CLASS_ID 1

/* Model path on camera */
#define MODEL_PATH "/usr/local/packages/autoacap/model/ssd_mobilenet_v2_int8.tflite"

/* Metrics output path */
#define METRICS_PATH "/tmp/autoacap_metrics.json"
#define DETECTIONS_PATH "/tmp/autoacap_detections.json"

typedef struct {
    float x, y, w, h;
    float confidence;
    int class_id;
} Detection;

typedef struct {
    double fps;
    double *latencies_ms;
    int num_latencies;
    int capacity;
} Metrics;

static volatile int running = 1;

static void signal_handler(int sig) {
    (void)sig;
    running = 0;
}

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/**
 * Initialize VDO stream for video capture.
 * Returns VdoStream pointer or NULL on failure.
 */
static VdoStream *init_vdo_stream(void) {
    GError *error = NULL;
    VdoMap *settings = vdo_map_new();

    vdo_map_set_uint32(settings, "format", VDO_FORMAT_YUV_NV12);
    vdo_map_set_uint32(settings, "width", 640);
    vdo_map_set_uint32(settings, "height", 480);
    vdo_map_set_uint32(settings, "framerate", 15);

    VdoStream *stream = vdo_stream_new(settings, NULL, &error);
    g_object_unref(settings);

    if (!stream) {
        fprintf(stderr, "Failed to create VDO stream: %s\n",
                error ? error->message : "unknown error");
        if (error) g_error_free(error);
        return NULL;
    }

    if (!vdo_stream_start(stream, &error)) {
        fprintf(stderr, "Failed to start VDO stream: %s\n",
                error ? error->message : "unknown error");
        if (error) g_error_free(error);
        g_object_unref(stream);
        return NULL;
    }

    return stream;
}

/**
 * Initialize Larod inference session.
 * Loads the TFLite model onto the DLPU.
 */
static larodConnection *init_larod(larodModel **model_out,
                                    larodTensor **input_tensor_out,
                                    larodTensor **output_tensors_out,
                                    int *num_outputs) {
    larodError *error = NULL;

    larodConnection *conn = larodConnect(&error);
    if (!conn) {
        fprintf(stderr, "Failed to connect to larod: %s\n",
                error ? error->msg : "unknown");
        larodClearError(&error);
        return NULL;
    }

    /* Load model onto DLPU (chip 12 for ARTPEC-8/9) */
    const larodDevice *device = larodGetDevice(conn, "axis-a8-dlpu", 0, &error);
    if (!device) {
        fprintf(stderr, "Failed to get DLPU device: %s\n",
                error ? error->msg : "unknown");
        larodClearError(&error);
        larodDisconnect(&conn, NULL);
        return NULL;
    }

    int model_fd = open(MODEL_PATH, O_RDONLY);
    if (model_fd < 0) {
        fprintf(stderr, "Failed to open model file: %s\n", MODEL_PATH);
        larodDisconnect(&conn, NULL);
        return NULL;
    }

    larodModel *model = larodLoadModel(conn, model_fd, device,
                                        LAROD_ACCESS_PRIVATE, "autoacap", NULL, &error);
    close(model_fd);

    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n",
                error ? error->msg : "unknown");
        larodClearError(&error);
        larodDisconnect(&conn, NULL);
        return NULL;
    }

    /* Get input/output tensor info */
    size_t num_inputs = 0;
    larodTensor **inputs = larodGetModelInputs(model, &num_inputs, &error);
    if (!inputs || num_inputs == 0) {
        fprintf(stderr, "Failed to get model inputs\n");
        larodClearError(&error);
        larodDisconnect(&conn, NULL);
        return NULL;
    }

    size_t n_outputs = 0;
    larodTensor **outputs = larodGetModelOutputs(model, &n_outputs, &error);
    if (!outputs || n_outputs == 0) {
        fprintf(stderr, "Failed to get model outputs\n");
        larodClearError(&error);
        larodDisconnect(&conn, NULL);
        return NULL;
    }

    *model_out = model;
    *input_tensor_out = inputs[0];
    *output_tensors_out = outputs[0];
    *num_outputs = (int)n_outputs;

    return conn;
}

/**
 * Pre-process: resize and convert NV12 frame to model input format.
 * This is a hot path — optimization target.
 */
static void preprocess_frame(const uint8_t *nv12_data, int src_w, int src_h,
                              uint8_t *rgb_output, int dst_w, int dst_h) {
    /* Simple bilinear resize + NV12 to RGB conversion */
    /* TODO: Agent can optimize this with NEON intrinsics, lookup tables, etc. */

    float x_ratio = (float)src_w / dst_w;
    float y_ratio = (float)src_h / dst_h;

    for (int y = 0; y < dst_h; y++) {
        for (int x = 0; x < dst_w; x++) {
            int src_x = (int)(x * x_ratio);
            int src_y = (int)(y * y_ratio);

            /* NV12: Y plane followed by interleaved UV plane */
            uint8_t Y = nv12_data[src_y * src_w + src_x];
            int uv_offset = src_w * src_h + (src_y / 2) * src_w + (src_x & ~1);
            uint8_t U = nv12_data[uv_offset];
            uint8_t V = nv12_data[uv_offset + 1];

            /* YUV to RGB */
            int C = Y - 16;
            int D = U - 128;
            int E = V - 128;

            int R = (298 * C + 409 * E + 128) >> 8;
            int G = (298 * C - 100 * D - 208 * E + 128) >> 8;
            int B = (298 * C + 516 * D + 128) >> 8;

            int idx = (y * dst_w + x) * 3;
            rgb_output[idx + 0] = (uint8_t)(R < 0 ? 0 : (R > 255 ? 255 : R));
            rgb_output[idx + 1] = (uint8_t)(G < 0 ? 0 : (G > 255 ? 255 : G));
            rgb_output[idx + 2] = (uint8_t)(B < 0 ? 0 : (B > 255 ? 255 : B));
        }
    }
}

/**
 * Post-process: extract detections from model output tensors.
 * Applies confidence threshold and NMS.
 */
static int postprocess_detections(const float *boxes, const float *scores,
                                   const float *classes, int num_raw,
                                   Detection *out, int max_out) {
    int count = 0;

    for (int i = 0; i < num_raw && count < max_out; i++) {
        int class_id = (int)classes[i];
        float confidence = scores[i];

        if (class_id != PERSON_CLASS_ID || confidence < CONFIDENCE_THRESHOLD) {
            continue;
        }

        /* SSD output: [ymin, xmin, ymax, xmax] normalized */
        float ymin = boxes[i * 4 + 0];
        float xmin = boxes[i * 4 + 1];
        float ymax = boxes[i * 4 + 2];
        float xmax = boxes[i * 4 + 3];

        out[count].x = xmin;
        out[count].y = ymin;
        out[count].w = xmax - xmin;
        out[count].h = ymax - ymin;
        out[count].confidence = confidence;
        out[count].class_id = class_id;
        count++;
    }

    /* TODO: NMS — agent can optimize this */

    return count;
}

/**
 * Write metrics to JSON file for the benchmark harness to read.
 */
static void write_metrics(const Metrics *metrics) {
    FILE *f = fopen(METRICS_PATH, "w");
    if (!f) return;

    fprintf(f, "{\n");
    fprintf(f, "  \"fps\": %.1f,\n", metrics->fps);
    fprintf(f, "  \"latencies_ms\": [");
    for (int i = 0; i < metrics->num_latencies; i++) {
        if (i > 0) fprintf(f, ", ");
        fprintf(f, "%.1f", metrics->latencies_ms[i]);
    }
    fprintf(f, "]\n");
    fprintf(f, "}\n");
    fclose(f);
}

/**
 * Write detections to JSON for mAP computation.
 */
static void write_detections(Detection *dets, int num_dets, int frame_id, FILE *f) {
    if (frame_id == 0) {
        fprintf(f, "{\"frames\": [\n");
    } else {
        fprintf(f, ",\n");
    }

    fprintf(f, "  {\"frame_id\": %d, \"detections\": [", frame_id);
    for (int i = 0; i < num_dets; i++) {
        if (i > 0) fprintf(f, ", ");
        fprintf(f, "{\"bbox\": [%.4f, %.4f, %.4f, %.4f], \"confidence\": %.4f, \"class\": \"person\"}",
                dets[i].x, dets[i].y, dets[i].w, dets[i].h, dets[i].confidence);
    }
    fprintf(f, "]}");
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    printf("AutoACAP C variant starting...\n");

    /* Initialize VDO stream */
    VdoStream *stream = init_vdo_stream();
    if (!stream) {
        fprintf(stderr, "Failed to initialize VDO\n");
        return 1;
    }

    /* Initialize Larod */
    larodModel *model = NULL;
    larodTensor *input_tensor = NULL;
    larodTensor *output_tensors = NULL;
    int num_outputs = 0;

    larodConnection *larod = init_larod(&model, &input_tensor, &output_tensors, &num_outputs);
    if (!larod) {
        fprintf(stderr, "Failed to initialize Larod\n");
        g_object_unref(stream);
        return 1;
    }

    /* Allocate pre-processing buffer */
    uint8_t *rgb_buffer = (uint8_t *)malloc(INPUT_WIDTH * INPUT_HEIGHT * 3);
    if (!rgb_buffer) {
        fprintf(stderr, "Failed to allocate RGB buffer\n");
        return 1;
    }

    /* Metrics tracking */
    Metrics metrics = {0};
    metrics.capacity = 1000;
    metrics.latencies_ms = (double *)malloc(sizeof(double) * metrics.capacity);
    metrics.num_latencies = 0;

    Detection detections[MAX_DETECTIONS];
    int frame_count = 0;
    double start_time = get_time_ms();

    /* Open detections output file */
    FILE *det_file = fopen(DETECTIONS_PATH, "w");

    printf("Running detection loop...\n");

    /* Main detection loop */
    while (running) {
        GError *error = NULL;
        double frame_start = get_time_ms();

        /* Capture frame */
        VdoFrame *frame = vdo_stream_get_frame(stream, &error);
        if (!frame) {
            if (error) g_error_free(error);
            continue;
        }

        VdoBuffer *buffer = vdo_frame_get_buffer(frame);
        uint8_t *frame_data = (uint8_t *)vdo_buffer_get_data(buffer);

        /* Pre-process: NV12 -> RGB, resize to model input */
        preprocess_frame(frame_data, 640, 480, rgb_buffer, INPUT_WIDTH, INPUT_HEIGHT);

        /* Run inference via Larod */
        larodError *larod_err = NULL;
        /* TODO: Copy rgb_buffer to input tensor, run inference, read outputs */
        /* This is where the actual larod inference call goes */
        /* For now this is a skeleton — the agent will fill in the proper */
        /* larod job creation and execution flow */

        /* Post-process detections */
        /* TODO: Read output tensors and extract detections */
        int num_detections = 0;
        /* num_detections = postprocess_detections(...) */

        /* Write detections for mAP */
        if (det_file) {
            write_detections(detections, num_detections, frame_count, det_file);
        }

        /* Track latency */
        double frame_end = get_time_ms();
        double latency = frame_end - frame_start;

        if (metrics.num_latencies < metrics.capacity) {
            metrics.latencies_ms[metrics.num_latencies++] = latency;
        }

        g_object_unref(frame);
        frame_count++;

        /* Update FPS periodically */
        double elapsed = (frame_end - start_time) / 1000.0;
        if (elapsed > 0) {
            metrics.fps = frame_count / elapsed;
        }

        /* Write metrics every 10 frames */
        if (frame_count % 10 == 0) {
            write_metrics(&metrics);
            printf("\rFrames: %d | FPS: %.1f | Latency: %.1fms",
                   frame_count, metrics.fps, latency);
            fflush(stdout);
        }
    }

    printf("\nShutting down. Total frames: %d, FPS: %.1f\n",
           frame_count, metrics.fps);

    /* Close detections file */
    if (det_file) {
        fprintf(det_file, "\n]}\n");
        fclose(det_file);
    }

    /* Final metrics write */
    write_metrics(&metrics);

    /* Cleanup */
    free(rgb_buffer);
    free(metrics.latencies_ms);
    larodDisconnect(&larod, NULL);
    vdo_stream_stop(stream);
    g_object_unref(stream);

    return 0;
}
