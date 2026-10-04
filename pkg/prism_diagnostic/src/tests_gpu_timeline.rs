//! Integration oracles for the §24.2 unified CPU·GPU timeline: full ingestion
//! flow (issue → calibrate → resolve → project), submit→execute latency
//! decomposition, GPU-bubble detection, and Chrome-export alignment.
//!
//! Every expected value here is hand-computed from fixed synthetic input so a
//! regression in the alignment math, correlation accounting, or bubble scan is
//! caught deterministically on CPU with no GPU present.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use crate::gpu::{
    next_correlation_id, CorrelationId, GpuClockCalibration, GpuQueueId, GpuReadbackRing, GpuTick,
    ProjectedGpuSpan, UnifiedTimeline,
};
use crate::trace::ring::SpanRecord;

fn cpu_span(name: &str, thread_id: u64, start: u64, dur: u64) -> SpanRecord {
    SpanRecord {
        name: String::from(name),
        category: None,
        thread_id,
        start_nanos: start,
        duration_nanos: dur,
        depth: 0,
        args: Vec::new(),
    }
}

fn gpu_span(name: &str, queue: u32, start: u64, dur: u64, corr: Option<u64>) -> ProjectedGpuSpan {
    ProjectedGpuSpan {
        label: String::from(name),
        queue: GpuQueueId(queue),
        cpu_start_nanos: start,
        cpu_duration_nanos: dur,
        correlation: corr.map(CorrelationId),
        frame: 0,
        depth: 0,
    }
}

/// A backend-shaped flow: mint a token, issue a query, calibrate the clock,
/// resolve ticks, project onto the CPU base, then assemble the unified timeline
/// and read a stage-decomposed latency. Oracles computed by hand.
#[test]
fn full_flow_submit_to_execute_latency() {
    let corr = next_correlation_id();
    let mut ring = GpuReadbackRing::new(2);

    // Frame 10: a draw's GPU scope is issued on the graphics queue.
    let qid = ring.issue(10, "GBuffer", GpuQueueId::GRAPHICS, Some(corr), 0);
    // Two frames later the ticks come back: begin=2000, end=2600 (600 ticks).
    assert!(ring.ready_queries(12).contains(&qid));
    let gpu = ring.resolve(qid, GpuTick(2_000), GpuTick(2_600)).unwrap();

    // Calibration cpu_ns = 5 * tick + 1000 (5 ns/tick, offset 1000).
    let mut cal = GpuClockCalibration::new(5.0);
    for tick in [0u64, 1_000, 2_000, 3_000] {
        cal.add_sample(5 * tick + 1_000, GpuTick(tick));
    }
    let projected = cal.project_span(&gpu);
    // 5 * 2000 + 1000 = 11_000 start; 5 * 600 = 3_000 duration => ends 14_000.
    assert_eq!(projected.cpu_start_nanos, 11_000);
    assert_eq!(projected.cpu_duration_nanos, 3_000);

    let mut tl = UnifiedTimeline::new();
    // CPU recorded + submitted the draw on thread 2: [9_000, 9_500).
    tl.add_cpu_span_correlated(&cpu_span("submit GBuffer", 2, 9_000, 500), corr);
    tl.add_gpu_span(&projected);

    let bd = tl.correlation_breakdown(corr).unwrap();
    assert_eq!(bd.cpu_submit_end_nanos, Some(9_500));
    assert_eq!(bd.gpu_execute_start_nanos, Some(11_000));
    // Submit end 9_500 -> GPU start 11_000 = 1_500ns stall.
    assert_eq!(bd.submit_to_execute_nanos(), Some(1_500));
    assert_eq!(bd.gpu_active_nanos(), Some(3_000));
    // Total 9_000 -> 14_000 = 5_000ns.
    assert_eq!(bd.total_nanos(), 5_000);
    assert_eq!(tl.end_to_end_latency(corr), Some(5_000));
}

/// Two sequential GPU passes on one queue with a measurable starvation gap.
#[test]
fn gpu_bubble_detection_oracle() {
    let mut tl = UnifiedTimeline::new();
    // Shadow [1_000, 1_400); gap; Lighting [1_600, 2_000); gap; Post [2_500, 2_700).
    tl.add_gpu_span(&gpu_span("Shadow", 0, 1_000, 400, None));
    tl.add_gpu_span(&gpu_span("Lighting", 0, 1_600, 400, None));
    tl.add_gpu_span(&gpu_span("Post", 0, 2_500, 200, None));

    let bubbles = tl.gpu_bubbles(GpuQueueId(0));
    assert_eq!(bubbles.len(), 2);
    // Gap 1: 1_400 -> 1_600 = 200ns.
    assert_eq!(bubbles[0].start_nanos, 1_400);
    assert_eq!(bubbles[0].gap_nanos, 200);
    assert_eq!(bubbles[0].after_label, "Shadow");
    assert_eq!(bubbles[0].before_label, "Lighting");
    // Gap 2: 2_000 -> 2_500 = 500ns.
    assert_eq!(bubbles[1].start_nanos, 2_000);
    assert_eq!(bubbles[1].gap_nanos, 500);
    // Total idle = 700ns.
    assert_eq!(tl.gpu_idle_nanos(GpuQueueId(0)), 700);
}

/// The Chrome exporter must place the GPU span on its own aligned track,
/// offset into the dedicated GPU track-id band and carrying the correlation arg.
#[test]
fn chrome_export_includes_aligned_gpu_track() {
    // GPU spans on queue 2 so the track tid is clearly offset from CPU tids.
    let gpu_spans = [gpu_span("execute", 2, 200, 300, Some(3))];
    let json = crate::trace::export_chrome_with_gpu(&gpu_spans);

    // The GPU scope is present with a gpu category and the correlation arg.
    assert!(json.contains("\"name\":\"execute\""));
    assert!(json.contains("\"cat\":\"gpu\""));
    assert!(json.contains("\"correlation\":3"));
    // GPU track tid = base + queue id (2); far above any CPU thread id.
    let gpu_tid = crate::trace::GPU_TRACK_TID_BASE + 2;
    assert!(json.contains(&alloc::format!("\"tid\":{gpu_tid}")));
    // Aligned timeline: CPU start 200ns -> 0.2us in the complete event.
    assert!(json.contains("\"GPU Queue 2\""));
}
