//! AutoACAP — Rust Person Detection Pipeline
//!
//! Uses VDO for video capture and Larod for ML inference on ARTPEC-9 DLPU.
//! Detects persons using SSD MobileNet V2 (INT8 quantized).
//!
//! This file is modified by the autonomous research agent.
//! The agent optimizes for maximum FPS while maintaining mAP >= 0.40.

use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use serde::Serialize;

/// Detection output
#[derive(Debug, Clone, Serialize)]
struct Detection {
    bbox: [f32; 4], // [x, y, w, h] normalized
    confidence: f32,
    class: String,
}

/// Per-frame detection results
#[derive(Debug, Serialize)]
struct FrameResult {
    frame_id: u32,
    detections: Vec<Detection>,
}

/// Metrics for the benchmark harness
#[derive(Debug, Serialize)]
struct Metrics {
    fps: f64,
    latencies_ms: Vec<f64>,
}

/// Detection results container
#[derive(Debug, Serialize)]
struct DetectionResults {
    frames: Vec<FrameResult>,
}

// Configuration constants
const INPUT_WIDTH: u32 = 300;
const INPUT_HEIGHT: u32 = 300;
const CONFIDENCE_THRESHOLD: f32 = 0.5;
const NMS_THRESHOLD: f32 = 0.45;
const PERSON_CLASS_ID: i32 = 1;
const MODEL_PATH: &str = "/usr/local/packages/autoacap/model/ssd_mobilenet_v2_int8.tflite";
const METRICS_PATH: &str = "/tmp/autoacap_metrics.json";
const DETECTIONS_PATH: &str = "/tmp/autoacap_detections.json";

/// Pre-process: resize and convert NV12 frame to RGB for model input.
/// This is a hot path — optimization target.
fn preprocess_frame(
    nv12_data: &[u8],
    src_w: u32,
    src_h: u32,
    rgb_output: &mut [u8],
    dst_w: u32,
    dst_h: u32,
) {
    let x_ratio = src_w as f32 / dst_w as f32;
    let y_ratio = src_h as f32 / dst_h as f32;

    for y in 0..dst_h {
        for x in 0..dst_w {
            let src_x = (x as f32 * x_ratio) as u32;
            let src_y = (y as f32 * y_ratio) as u32;

            // NV12: Y plane followed by interleaved UV plane
            let y_val = nv12_data[(src_y * src_w + src_x) as usize] as i32;
            let uv_offset = (src_w * src_h + (src_y / 2) * src_w + (src_x & !1)) as usize;
            let u_val = nv12_data[uv_offset] as i32;
            let v_val = nv12_data[uv_offset + 1] as i32;

            // YUV to RGB
            let c = y_val - 16;
            let d = u_val - 128;
            let e = v_val - 128;

            let r = ((298 * c + 409 * e + 128) >> 8).clamp(0, 255) as u8;
            let g = ((298 * c - 100 * d - 208 * e + 128) >> 8).clamp(0, 255) as u8;
            let b = ((298 * c + 516 * d + 128) >> 8).clamp(0, 255) as u8;

            let idx = ((y * dst_w + x) * 3) as usize;
            rgb_output[idx] = r;
            rgb_output[idx + 1] = g;
            rgb_output[idx + 2] = b;
        }
    }
}

/// Post-process: extract person detections from model output.
/// Applies confidence threshold and NMS.
fn postprocess_detections(
    boxes: &[f32],
    scores: &[f32],
    classes: &[f32],
    num_raw: usize,
) -> Vec<Detection> {
    let mut detections = Vec::new();

    for i in 0..num_raw {
        let class_id = classes[i] as i32;
        let confidence = scores[i];

        if class_id != PERSON_CLASS_ID || confidence < CONFIDENCE_THRESHOLD {
            continue;
        }

        // SSD output: [ymin, xmin, ymax, xmax] normalized
        let ymin = boxes[i * 4];
        let xmin = boxes[i * 4 + 1];
        let ymax = boxes[i * 4 + 2];
        let xmax = boxes[i * 4 + 3];

        detections.push(Detection {
            bbox: [xmin, ymin, xmax - xmin, ymax - ymin],
            confidence,
            class: "person".to_string(),
        });
    }

    // TODO: NMS — agent can optimize this

    detections
}

/// Write metrics to JSON file for the benchmark harness.
fn write_metrics(metrics: &Metrics) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(metrics)?;
    std::fs::write(METRICS_PATH, json)?;
    Ok(())
}

fn main() {
    println!("AutoACAP Rust variant starting...");

    // Signal handling for clean shutdown
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting signal handler");

    // TODO: Initialize VDO stream via FFI to libvdo
    // TODO: Initialize Larod via FFI to liblarod
    // These will use unsafe blocks to call the Axis C APIs.
    // The agent will fill in the proper FFI bindings using acap-rs crates.

    // Allocate pre-processing buffer
    let mut rgb_buffer = vec![0u8; (INPUT_WIDTH * INPUT_HEIGHT * 3) as usize];

    // Metrics tracking
    let mut metrics = Metrics {
        fps: 0.0,
        latencies_ms: Vec::with_capacity(1000),
    };

    let mut detection_results = DetectionResults {
        frames: Vec::new(),
    };

    let mut frame_count: u32 = 0;
    let start_time = Instant::now();

    println!("Running detection loop...");

    // Main detection loop
    while running.load(Ordering::SeqCst) {
        let frame_start = Instant::now();

        // TODO: Capture frame from VDO stream
        // let frame_data = vdo_stream_get_frame(...);

        // TODO: Pre-process frame
        // preprocess_frame(&frame_data, 640, 480, &mut rgb_buffer, INPUT_WIDTH, INPUT_HEIGHT);

        // TODO: Run inference via Larod
        // Copy rgb_buffer to input tensor, run inference, read outputs

        // TODO: Post-process detections
        let detections: Vec<Detection> = Vec::new();
        // let detections = postprocess_detections(&boxes, &scores, &classes, num_raw);

        // Record detections for mAP
        detection_results.frames.push(FrameResult {
            frame_id: frame_count,
            detections: detections.clone(),
        });

        // Track latency
        let latency_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
        metrics.latencies_ms.push(latency_ms);

        frame_count += 1;

        // Update FPS
        let elapsed_secs = start_time.elapsed().as_secs_f64();
        if elapsed_secs > 0.0 {
            metrics.fps = frame_count as f64 / elapsed_secs;
        }

        // Write metrics every 10 frames
        if frame_count % 10 == 0 {
            let _ = write_metrics(&metrics);
            print!(
                "\rFrames: {} | FPS: {:.1} | Latency: {:.1}ms",
                frame_count, metrics.fps, latency_ms
            );
            let _ = std::io::stdout().flush();
        }
    }

    println!(
        "\nShutting down. Total frames: {}, FPS: {:.1}",
        frame_count, metrics.fps
    );

    // Write final metrics
    let _ = write_metrics(&metrics);

    // Write detection results for mAP computation
    if let Ok(json) = serde_json::to_string_pretty(&detection_results) {
        let _ = std::fs::write(DETECTIONS_PATH, json);
    }

    // TODO: Cleanup VDO stream and Larod connection
}
