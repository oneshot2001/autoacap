//! AutoACAP — Rust Person Detection Pipeline
//!
//! Uses VDO for video capture and Larod for ML inference on ARTPEC-9 DLPU.
//! Detects persons using SSD MobileNet V2 (INT8 quantized).
//! Uses Larod's built-in cpu-proc preprocessing for NV12→RGB conversion.
//!
//! This file is modified by the autonomous research agent.
//! The agent optimizes for maximum FPS while maintaining mAP >= 0.40.
//!
//! Based on Axis ACAP acap-rs FFI bindings (larod-sys, vdo-sys).

use std::ffi::CString;
use std::fs::File;
use std::io::Write;
use std::os::unix::io::AsRawFd;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use log::{error, info, warn};
use serde::Serialize;

// ─── Configuration ──────────────────────────────────────────────────────

const MODEL_PATH: &str = "/usr/local/packages/autoacap/model/ssd_mobilenet_v2_int8.tflite";
const METRICS_PATH: &str = "/tmp/autoacap_metrics.json";
const DETECTIONS_PATH: &str = "/tmp/autoacap_detections.json";

const STREAM_WIDTH: u32 = 640;
const STREAM_HEIGHT: u32 = 480;
const STREAM_FPS: f64 = 15.0;

const CONFIDENCE_THRESHOLD: f32 = 0.5;
const PERSON_CLASS_ID: i32 = 1;
const MAX_LATENCIES: usize = 2000;
const MAX_POWER_RETRIES: u32 = 50;

// Discovered at runtime via larodListDevices
const LAROD_DEVICE_FALLBACKS: &[&str] = &["a9-dlpu-tflite", "axis-a9-dlpu", "axis-a8-dlpu"];
const PP_DEVICE_NAME: &str = "cpu-proc";

// ─── Types ──────────────────────────────────────────────────────────────

const MAX_DETECTIONS: usize = 100;

#[derive(Debug, Clone, Copy, Serialize)]
struct Detection {
    bbox: [f32; 4],
    confidence: f32,
    class_id: i32,
}

#[derive(Debug, Serialize)]
struct Metrics {
    fps: f64,
    latencies_ms: Vec<f64>,
}

// ─── Output tensor info ─────────────────────────────────────────────────

struct TensorOutput {
    data: *mut u8,
    size: usize,
}

// ─── Larod Provider ─────────────────────────────────────────────────────

struct LarodProvider {
    conn: *mut larod_sys::larodConnection,
    _model_fd: i32,

    pp_req: *mut larod_sys::larodJobRequest,
    inf_req: *mut larod_sys::larodJobRequest,

    // Preprocessing input (mmap'd for VDO frame copy)
    image_input_addr: *mut u8,
    image_buffer_size: usize,

    // Output tensors (mmap'd for reading results)
    outputs: Vec<TensorOutput>,
    num_outputs: usize,

    // Model input dimensions
    model_width: u32,
    model_height: u32,

    // Raw tensor pointers for cleanup
    pp_input_tensors: *mut *mut larod_sys::larodTensor,
    pp_num_inputs: usize,
    pp_output_tensors: *mut *mut larod_sys::larodTensor,
    pp_num_outputs: usize,
    input_tensors: *mut *mut larod_sys::larodTensor,
    num_inputs: usize,
    output_tensors: *mut *mut larod_sys::larodTensor,
}

impl LarodProvider {
    unsafe fn new(vdo_width: u32, vdo_height: u32, vdo_pitch: u32) -> Option<Self> {
        let mut error: *mut larod_sys::larodError = ptr::null_mut();
        let mut conn: *mut larod_sys::larodConnection = ptr::null_mut();

        // Connect to larod
        if !larod_sys::larodConnect(&mut conn, &mut error) {
            error!("larodConnect failed");
            larod_sys::larodClearError(&mut error);
            return None;
        }

        // Discover DLPU device
        let mut num_devices: usize = 0;
        let devices = larod_sys::larodListDevices(conn, &mut num_devices, &mut error);
        if devices.is_null() || num_devices == 0 {
            error!("larodListDevices failed");
            larod_sys::larodClearError(&mut error);
            larod_sys::larodDisconnect(&mut conn, ptr::null_mut());
            return None;
        }

        let mut device: *const larod_sys::larodDevice = ptr::null();
        for i in 0..num_devices {
            let dev = *devices.add(i);
            let name_ptr = larod_sys::larodGetDeviceName(dev, &mut error);
            if !name_ptr.is_null() {
                let name = std::ffi::CStr::from_ptr(name_ptr).to_string_lossy();
                info!("Larod device [{}]: {}", i, name);
                // Prefer dlpu-tflite for .tflite models
                if name.contains("dlpu-tflite") {
                    device = dev;
                    info!("Selected DLPU device: {}", name);
                    break;
                } else if name.contains("dlpu") && device.is_null() {
                    device = dev;
                    info!("Selected DLPU device (fallback): {}", name);
                }
            }
        }

        // Try fallback names if no dlpu found via listing
        if device.is_null() {
            for fallback in LAROD_DEVICE_FALLBACKS {
                let name = CString::new(*fallback).unwrap();
                larod_sys::larodClearError(&mut error);
                let d = larod_sys::larodGetDevice(conn, name.as_ptr(), 0, &mut error);
                if !d.is_null() {
                    device = d;
                    info!("Found DLPU via fallback: {}", fallback);
                    break;
                }
            }
        }

        if device.is_null() {
            error!("No DLPU device found");
            larod_sys::larodDisconnect(&mut conn, ptr::null_mut());
            return None;
        }

        // Open and load model
        let model_path = CString::new(MODEL_PATH).unwrap();
        let model_fd = libc::open(model_path.as_ptr(), libc::O_RDONLY);
        if model_fd < 0 {
            error!("Failed to open model: {}", MODEL_PATH);
            larod_sys::larodDisconnect(&mut conn, ptr::null_mut());
            return None;
        }

        info!("Loading model onto DLPU...");
        let model_desc = CString::new("autoacap-detection").unwrap();
        let mut model: *mut larod_sys::larodModel = ptr::null_mut();
        let mut retries = 0u32;

        loop {
            larod_sys::larodClearError(&mut error);
            model = larod_sys::larodLoadModel(
                conn, model_fd, device,
                larod_sys::larodAccess::LAROD_ACCESS_PRIVATE,
                model_desc.as_ptr(),
                ptr::null_mut(),
                &mut error,
            );
            if !model.is_null() { break; }
            if !error.is_null()
                && (*error).code == larod_sys::larodErrorCode::LAROD_ERROR_POWER_NOT_AVAILABLE
            {
                retries += 1;
                if retries >= MAX_POWER_RETRIES {
                    error!("Max power retries exceeded");
                    libc::close(model_fd);
                    larod_sys::larodDisconnect(&mut conn, ptr::null_mut());
                    return None;
                }
                libc::usleep(250_000 * retries);
            } else {
                error!("larodLoadModel failed");
                libc::close(model_fd);
                larod_sys::larodClearError(&mut error);
                larod_sys::larodDisconnect(&mut conn, ptr::null_mut());
                return None;
            }
        }
        info!("Model loaded successfully");

        // Allocate inference tensors
        let mut num_inputs: usize = 0;
        let input_tensors = larod_sys::larodAllocModelInputs(
            conn, model, 0, &mut num_inputs, ptr::null_mut(), &mut error,
        );
        if input_tensors.is_null() {
            error!("Failed to alloc input tensors");
            larod_sys::larodClearError(&mut error);
            libc::close(model_fd);
            larod_sys::larodDisconnect(&mut conn, ptr::null_mut());
            return None;
        }

        let mut num_outputs: usize = 0;
        let output_tensors = larod_sys::larodAllocModelOutputs(
            conn, model, 0, &mut num_outputs, ptr::null_mut(), &mut error,
        );
        if output_tensors.is_null() {
            error!("Failed to alloc output tensors");
            larod_sys::larodClearError(&mut error);
            libc::close(model_fd);
            larod_sys::larodDisconnect(&mut conn, ptr::null_mut());
            return None;
        }
        info!("Model: {} inputs, {} outputs", num_inputs, num_outputs);

        // Get model input dimensions
        let dims = larod_sys::larodGetTensorDims(*input_tensors, &mut error);
        if dims.is_null() {
            error!("Failed to get input tensor dims");
            larod_sys::larodClearError(&mut error);
            libc::close(model_fd);
            larod_sys::larodDisconnect(&mut conn, ptr::null_mut());
            return None;
        }
        let model_height = (*dims).dims[1] as u32;
        let model_width = (*dims).dims[2] as u32;
        info!("Model input: {}x{}", model_width, model_height);

        // Setup preprocessing model (NV12 → RGB, resize)
        let pp_device_name = CString::new(PP_DEVICE_NAME).unwrap();
        let pp_device = larod_sys::larodGetDevice(conn, pp_device_name.as_ptr(), 0, &mut error);
        if pp_device.is_null() {
            error!("Failed to get preprocessing device");
            larod_sys::larodClearError(&mut error);
            libc::close(model_fd);
            larod_sys::larodDisconnect(&mut conn, ptr::null_mut());
            return None;
        }

        let pp_map = larod_sys::larodCreateMap(&mut error);
        if pp_map.is_null() {
            error!("Failed to create preprocessing map");
            larod_sys::larodClearError(&mut error);
            libc::close(model_fd);
            larod_sys::larodDisconnect(&mut conn, ptr::null_mut());
            return None;
        }

        let key_input_format = CString::new("image.input.format").unwrap();
        let val_nv12 = CString::new("nv12").unwrap();
        larod_sys::larodMapSetStr(pp_map, key_input_format.as_ptr(), val_nv12.as_ptr(), &mut error);

        let key_input_size = CString::new("image.input.size").unwrap();
        larod_sys::larodMapSetIntArr2(
            pp_map, key_input_size.as_ptr(),
            vdo_width as i32, vdo_height as i32, &mut error,
        );

        let key_input_pitch = CString::new("image.input.row-pitch").unwrap();
        larod_sys::larodMapSetInt(
            pp_map, key_input_pitch.as_ptr(), vdo_pitch as i32, &mut error,
        );

        let key_output_format = CString::new("image.output.format").unwrap();
        let val_rgb = CString::new("rgb-interleaved").unwrap();
        larod_sys::larodMapSetStr(pp_map, key_output_format.as_ptr(), val_rgb.as_ptr(), &mut error);

        let key_output_size = CString::new("image.output.size").unwrap();
        larod_sys::larodMapSetIntArr2(
            pp_map, key_output_size.as_ptr(),
            model_width as i32, model_height as i32, &mut error,
        );

        let empty_desc = CString::new("").unwrap();
        let pp_model = larod_sys::larodLoadModel(
            conn, -1, pp_device,
            larod_sys::larodAccess::LAROD_ACCESS_PRIVATE,
            empty_desc.as_ptr(), pp_map, &mut error,
        );
        larod_sys::larodDestroyMap(&mut (pp_map as *mut _));
        if pp_model.is_null() {
            error!("Failed to load preprocessing model");
            larod_sys::larodClearError(&mut error);
            libc::close(model_fd);
            larod_sys::larodDisconnect(&mut conn, ptr::null_mut());
            return None;
        }

        // Allocate preprocessing tensors
        let mut pp_num_inputs: usize = 0;
        let pp_input_tensors = larod_sys::larodAllocModelInputs(
            conn, pp_model, 0, &mut pp_num_inputs, ptr::null_mut(), &mut error,
        );
        let mut pp_num_outputs: usize = 0;
        let pp_output_tensors = larod_sys::larodAllocModelOutputs(
            conn, pp_model, 0, &mut pp_num_outputs, ptr::null_mut(), &mut error,
        );
        if pp_input_tensors.is_null() || pp_output_tensors.is_null() {
            error!("Failed to alloc preprocessing tensors");
            larod_sys::larodClearError(&mut error);
            libc::close(model_fd);
            larod_sys::larodDisconnect(&mut conn, ptr::null_mut());
            return None;
        }

        // Map preprocessing input for VDO frame data
        let image_input_fd = larod_sys::larodGetTensorFd(*pp_input_tensors, &mut error);
        let mut image_buffer_size: usize = 0;
        larod_sys::larodGetTensorFdSize(*pp_input_tensors, &mut image_buffer_size, &mut error);

        let image_input_addr = libc::mmap(
            ptr::null_mut(),
            image_buffer_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            image_input_fd,
            0,
        ) as *mut u8;
        if image_input_addr == libc::MAP_FAILED as *mut u8 {
            error!("mmap preprocessing input failed");
            libc::close(model_fd);
            larod_sys::larodDisconnect(&mut conn, ptr::null_mut());
            return None;
        }

        info!(
            "Preprocessing: NV12 {}x{} → RGB {}x{} (buffer {} bytes)",
            vdo_width, vdo_height, model_width, model_height, image_buffer_size
        );

        // Create job requests
        let pp_req = larod_sys::larodCreateJobRequest(
            pp_model,
            pp_input_tensors, pp_num_inputs,
            pp_output_tensors, pp_num_outputs,
            ptr::null_mut(), &mut error,
        );
        if pp_req.is_null() {
            error!("Failed to create preprocessing job request");
            larod_sys::larodClearError(&mut error);
            libc::close(model_fd);
            larod_sys::larodDisconnect(&mut conn, ptr::null_mut());
            return None;
        }

        // Chain: preprocessing output → inference input
        let inf_req = larod_sys::larodCreateJobRequest(
            model,
            pp_output_tensors, pp_num_outputs,
            output_tensors, num_outputs,
            ptr::null_mut(), &mut error,
        );
        if inf_req.is_null() {
            error!("Failed to create inference job request");
            larod_sys::larodClearError(&mut error);
            libc::close(model_fd);
            larod_sys::larodDisconnect(&mut conn, ptr::null_mut());
            return None;
        }

        // Memory-map output tensors
        let mut outputs = Vec::with_capacity(num_outputs);
        for i in 0..num_outputs {
            let tensor = *output_tensors.add(i);
            let fd = larod_sys::larodGetTensorFd(tensor, &mut error);
            let mut sz: usize = 0;
            larod_sys::larodGetTensorFdSize(tensor, &mut sz, &mut error);

            let addr = libc::mmap(
                ptr::null_mut(), sz, libc::PROT_READ, libc::MAP_SHARED, fd, 0,
            ) as *mut u8;
            if addr == libc::MAP_FAILED as *mut u8 {
                error!("mmap output tensor {} failed", i);
                libc::close(model_fd);
                larod_sys::larodDisconnect(&mut conn, ptr::null_mut());
                return None;
            }

            info!("Output tensor {}: {} bytes", i, sz);
            outputs.push(TensorOutput { data: addr, size: sz });
        }

        info!("Larod fully initialized");

        Some(LarodProvider {
            conn,
            _model_fd: model_fd,
            pp_req,
            inf_req,
            image_input_addr,
            image_buffer_size,
            outputs,
            num_outputs,
            model_width,
            model_height,
            pp_input_tensors,
            pp_num_inputs,
            pp_output_tensors,
            pp_num_outputs,
            input_tensors,
            num_inputs,
            output_tensors,
        })
    }

    unsafe fn run_inference(&self, frame_data: *const u8) -> bool {
        let mut error: *mut larod_sys::larodError = ptr::null_mut();

        // Copy VDO frame into preprocessing input buffer
        ptr::copy_nonoverlapping(frame_data, self.image_input_addr, self.image_buffer_size);

        // Run preprocessing
        if !larod_sys::larodRunJob(self.conn, self.pp_req, &mut error) {
            if !error.is_null()
                && (*error).code == larod_sys::larodErrorCode::LAROD_ERROR_POWER_NOT_AVAILABLE
            {
                warn!("No power for preprocessing, retrying...");
            } else {
                error!("Preprocessing failed");
            }
            larod_sys::larodClearError(&mut error);
            return false;
        }

        // Run inference
        if !larod_sys::larodRunJob(self.conn, self.inf_req, &mut error) {
            if !error.is_null()
                && (*error).code == larod_sys::larodErrorCode::LAROD_ERROR_POWER_NOT_AVAILABLE
            {
                warn!("No power for inference, retrying...");
            } else {
                error!("Inference failed");
            }
            larod_sys::larodClearError(&mut error);
            return false;
        }

        true
    }

    /// Write detections into a pre-allocated buffer. Returns count. Zero allocations.
    unsafe fn get_detections_into(&self, buf: &mut [Detection; MAX_DETECTIONS]) -> usize {
        if self.num_outputs < 4 {
            error!("Expected 4 output tensors, got {}", self.num_outputs);
            return 0;
        }

        let locations = self.outputs[0].data as *const f32;
        let classes = self.outputs[1].data as *const f32;
        let scores = self.outputs[2].data as *const f32;
        let num_raw = self.outputs[3].data as *const f32;

        let n = *num_raw as i32;
        let mut count = 0usize;

        for i in 0..n {
            if count >= MAX_DETECTIONS { break; }
            let idx = i as usize;
            let cls = *classes.add(idx) as i32;
            let conf = *scores.add(idx);

            if cls != PERSON_CLASS_ID || conf < CONFIDENCE_THRESHOLD {
                continue;
            }

            let y_min = *locations.add(idx * 4);
            let x_min = *locations.add(idx * 4 + 1);
            let y_max = *locations.add(idx * 4 + 2);
            let x_max = *locations.add(idx * 4 + 3);

            buf[count] = Detection {
                bbox: [x_min, y_min, x_max - x_min, y_max - y_min],
                confidence: conf,
                class_id: cls,
            };
            count += 1;
        }

        count
    }
}

impl Drop for LarodProvider {
    fn drop(&mut self) {
        unsafe {
            larod_sys::larodDestroyJobRequest(&mut self.pp_req);
            larod_sys::larodDestroyJobRequest(&mut self.inf_req);

            if !self.image_input_addr.is_null() {
                libc::munmap(self.image_input_addr as *mut libc::c_void, self.image_buffer_size);
            }
            for out in &self.outputs {
                if !out.data.is_null() {
                    libc::munmap(out.data as *mut libc::c_void, out.size);
                }
            }

            let mut error: *mut larod_sys::larodError = ptr::null_mut();
            larod_sys::larodDestroyTensors(
                self.conn, &mut self.pp_input_tensors, self.pp_num_inputs, &mut error,
            );
            larod_sys::larodDestroyTensors(
                self.conn, &mut self.pp_output_tensors, self.pp_num_outputs, &mut error,
            );
            larod_sys::larodDestroyTensors(
                self.conn, &mut self.input_tensors, self.num_inputs, &mut error,
            );
            larod_sys::larodDestroyTensors(
                self.conn, &mut self.output_tensors, self.num_outputs, &mut error,
            );

            libc::close(self._model_fd);
            larod_sys::larodDisconnect(&mut self.conn, ptr::null_mut());
        }
    }
}

// ─── VDO Provider ───────────────────────────────────────────────────────

struct VdoProvider {
    stream: *mut vdo_sys::VdoStream,
    poll_fd: i32,
    width: u32,
    height: u32,
    pitch: u32,
}

impl VdoProvider {
    unsafe fn new() -> Option<Self> {
        let mut error: *mut vdo_sys::GError = ptr::null_mut();

        let settings = vdo_sys::vdo_map_new();
        vdo_sys::vdo_map_set_uint32(settings, c"format".as_ptr(), vdo_sys::VdoFormat::VDO_FORMAT_YUV.0 as u32);
        vdo_sys::vdo_map_set_uint32(settings, c"width".as_ptr(), STREAM_WIDTH);
        vdo_sys::vdo_map_set_uint32(settings, c"height".as_ptr(), STREAM_HEIGHT);
        vdo_sys::vdo_map_set_double(settings, c"framerate".as_ptr(), STREAM_FPS);
        vdo_sys::vdo_map_set_uint32(settings, c"buffer.count".as_ptr(), 2);
        vdo_sys::vdo_map_set_boolean(settings, c"socket.blocking".as_ptr(), 0);

        let stream = vdo_sys::vdo_stream_new(settings, ptr::null_mut(), &mut error);
        vdo_sys::g_object_unref(settings as *mut _);

        if stream.is_null() {
            error!("Failed to create VDO stream");
            return None;
        }

        // Get actual stream info
        let info = vdo_sys::vdo_stream_get_info(stream, &mut error);
        let (width, height, pitch) = if !info.is_null() {
            let w = vdo_sys::vdo_map_get_uint32(info, c"width".as_ptr(), STREAM_WIDTH);
            let h = vdo_sys::vdo_map_get_uint32(info, c"height".as_ptr(), STREAM_HEIGHT);
            let p = vdo_sys::vdo_map_get_uint32(info, c"pitch".as_ptr(), w);
            vdo_sys::g_object_unref(info as *mut _);
            (w, h, p)
        } else {
            (STREAM_WIDTH, STREAM_HEIGHT, STREAM_WIDTH)
        };

        if vdo_sys::vdo_stream_start(stream, &mut error) == 0 {
            error!("Failed to start VDO stream");
            vdo_sys::g_object_unref(stream as *mut _);
            return None;
        }

        let poll_fd = vdo_sys::vdo_stream_get_fd(stream, &mut error);
        if poll_fd < 0 {
            error!("Failed to get VDO stream fd");
            vdo_sys::g_object_unref(stream as *mut _);
            return None;
        }

        info!("VDO stream: {}x{} pitch={}", width, height, pitch);

        Some(VdoProvider { stream, poll_fd, width, height, pitch })
    }

    unsafe fn get_frame(&self) -> *mut vdo_sys::VdoBuffer {
        let mut error: *mut vdo_sys::GError = ptr::null_mut();

        let mut pfd = libc::pollfd {
            fd: self.poll_fd,
            events: libc::POLLIN,
            revents: 0,
        };

        loop {
            let ret = libc::poll(&mut pfd, 1, 1000);
            if ret <= 0 { continue; }

            let buf = vdo_sys::vdo_stream_get_buffer(self.stream, &mut error);
            if !buf.is_null() {
                return buf;
            }

            if !error.is_null() {
                vdo_sys::g_error_free(error);
                error = ptr::null_mut();
            }
        }
    }
}

impl Drop for VdoProvider {
    fn drop(&mut self) {
        unsafe {
            if !self.stream.is_null() {
                vdo_sys::vdo_stream_stop(self.stream);
                vdo_sys::g_object_unref(self.stream as *mut _);
            }
        }
    }
}

// ─── Metrics Output ─────────────────────────────────────────────────────

fn write_metrics(fps: f64, latencies: &[f64]) {
    let metrics = Metrics {
        fps,
        latencies_ms: latencies.to_vec(),
    };
    if let Ok(json) = serde_json::to_string_pretty(&metrics) {
        let _ = std::fs::write(METRICS_PATH, json);
    }
}

// ─── Signal handling ────────────────────────────────────────────────────

static RUNNING: AtomicBool = AtomicBool::new(true);

extern "C" fn handle_signal(_sig: libc::c_int) {
    RUNNING.store(false, Ordering::SeqCst);
}

// ─── Main ───────────────────────────────────────────────────────────────

fn main() {
    acap_logging::init_logger();

    // Install signal handlers
    unsafe {
        libc::signal(libc::SIGINT, handle_signal as libc::sighandler_t);
        libc::signal(libc::SIGTERM, handle_signal as libc::sighandler_t);
    }

    info!("AutoACAP Rust variant starting");
    println!("AutoACAP Rust variant starting...");

    // Initialize VDO
    let vdo = unsafe { VdoProvider::new() };
    let vdo = match vdo {
        Some(v) => v,
        None => {
            error!("Failed to init VDO");
            return;
        }
    };

    // Initialize Larod with preprocessing
    let larod = unsafe { LarodProvider::new(vdo.width, vdo.height, vdo.pitch) };
    let larod = match larod {
        Some(l) => l,
        None => {
            error!("Failed to init Larod");
            return;
        }
    };

    // Metrics tracking — pre-allocated, no growth in hot loop
    let mut latencies: Vec<f64> = Vec::with_capacity(MAX_LATENCIES);
    let mut frame_count: u32 = 0;
    let mut fps: f64 = 0.0;
    let start_time = Instant::now();

    // Pre-allocated detection buffer — zero allocs per frame
    let mut det_buf = [Detection { bbox: [0.0; 4], confidence: 0.0, class_id: 0 }; MAX_DETECTIONS];

    // Stream detections to file instead of accumulating in memory
    let mut det_file = File::create(DETECTIONS_PATH).ok();
    if let Some(ref mut f) = det_file {
        let _ = write!(f, "{{\"frames\": [\n");
    }

    info!("Entering detection loop");
    println!("Running detection loop...");

    // ─── Main detection loop (zero-allocation) ───
    while RUNNING.load(Ordering::SeqCst) {
        let frame_start = Instant::now();

        // Get frame from VDO
        let vdo_buf = unsafe { vdo.get_frame() };
        if vdo_buf.is_null() { continue; }

        let frame_data = unsafe { vdo_sys::vdo_buffer_get_data(vdo_buf) as *const u8 };

        // Run preprocessing + inference
        let ok = unsafe { larod.run_inference(frame_data) };

        // Release VDO buffer
        unsafe {
            let mut vdo_err: *mut vdo_sys::GError = ptr::null_mut();
            vdo_sys::vdo_stream_buffer_unref(vdo.stream, &mut (vdo_buf as *mut _), &mut vdo_err);
            if !vdo_err.is_null() { vdo_sys::g_error_free(vdo_err); }
        }

        if !ok { continue; }

        // Extract detections into pre-allocated buffer — zero allocations
        let num_dets = unsafe { larod.get_detections_into(&mut det_buf) };

        // Stream detections to file (no Vec accumulation)
        if let Some(ref mut f) = det_file {
            if frame_count > 0 { let _ = write!(f, ",\n"); }
            let _ = write!(f, "  {{\"frame_id\": {}, \"detections\": [", frame_count);
            for i in 0..num_dets {
                if i > 0 { let _ = write!(f, ", "); }
                let d = &det_buf[i];
                let _ = write!(f,
                    "{{\"bbox\": [{:.4}, {:.4}, {:.4}, {:.4}], \"confidence\": {:.4}, \"class\": \"person\"}}",
                    d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence
                );
            }
            let _ = write!(f, "]}}");
        }

        // Track latency
        let latency_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
        if latencies.len() < MAX_LATENCIES {
            latencies.push(latency_ms);
        }

        frame_count += 1;

        // Update FPS
        let elapsed_s = start_time.elapsed().as_secs_f64();
        if elapsed_s > 0.0 { fps = frame_count as f64 / elapsed_s; }

        // Log periodically
        if frame_count % 10 == 0 {
            write_metrics(fps, &latencies);
            print!(
                "\rFrames: {} | FPS: {:.1} | Latency: {:.1}ms | Dets: {}   ",
                frame_count, fps, latency_ms, num_dets
            );
            let _ = std::io::stdout().flush();
            info!(
                "Frames: {} FPS: {:.1} Latency: {:.1}ms Dets: {}",
                frame_count, fps, latency_ms, num_dets
            );
        }
    }

    println!("\nShutting down. Total frames: {}, FPS: {:.1}", frame_count, fps);
    info!("Shutting down. Frames: {} FPS: {:.1}", frame_count, fps);

    // Final metrics
    write_metrics(fps, &latencies);

    // Close detections file
    if let Some(ref mut f) = det_file {
        let _ = write!(f, "\n]}}\n");
    }
}
