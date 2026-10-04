//! Submit→execute latency decomposition and GPU-bubble (idle-gap) analysis.
//!
//! The [`UnifiedTimeline`](crate::gpu::UnifiedTimeline) already merges CPU spans
//! and projected GPU spans onto one nanosecond axis and reconstructs *total*
//! end-to-end latency for a [`CorrelationId`]. But a total number hides the two
//! questions a graphics engineer actually asks (design §24.2):
//!
//! - **"The CPU submitted early — why did the GPU start late?"** That stall is
//!   the *submit→execute latency*: the gap between when the CPU finished
//!   recording/submitting an operation and when the GPU actually began
//!   executing it. A large value means the GPU was blocked on a dependency or
//!   simply backed up, not that the CPU was slow.
//! - **"Is the GPU queue starved?"** A *bubble* is an idle gap on a GPU queue
//!   track between the end of one execution interval and the start of the next.
//!   Bubbles are wasted GPU time: the queue had nothing to chew on.
//!
//! Both analyses are pure functions over the already-projected timeline. They
//! add no new ingestion surface and talk to no GPU — a backend still feeds the
//! raw ticks through [`crate::gpu`]; this module only reads the merged result.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use super::scope::{CorrelationId, GpuQueueId};
use super::timeline::{TimelineTrack, UnifiedTimeline};

/// A stage-decomposed view of one correlated GPU operation.
///
/// Built from every timeline entry sharing a [`CorrelationId`]. The CPU-track
/// entries are treated as the record/submit side and the GPU-track entries as
/// the execute/present side, so the latency between the two can be read
/// directly instead of being folded into a single end-to-end figure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CorrelationBreakdown {
    /// The correlation token this breakdown describes.
    pub correlation: CorrelationId,
    /// Earliest start across every stage, in nanoseconds.
    pub first_start_nanos: u64,
    /// Latest end across every stage, in nanoseconds.
    pub last_end_nanos: u64,
    /// Latest end among the CPU (record/submit) stages, if any CPU stage
    /// participates. This is "the moment the CPU was done handing work off".
    pub cpu_submit_end_nanos: Option<u64>,
    /// Earliest start among the GPU (execute/present) stages, if any GPU stage
    /// participates. This is "the moment the GPU actually began the work".
    pub gpu_execute_start_nanos: Option<u64>,
    /// Latest end among the GPU stages, if any.
    pub gpu_execute_end_nanos: Option<u64>,
    /// Number of CPU-track stages in the chain.
    pub cpu_stage_count: u32,
    /// Number of GPU-track stages in the chain.
    pub gpu_stage_count: u32,
}

impl CorrelationBreakdown {
    /// Total end-to-end latency (`last_end - first_start`, saturating).
    ///
    /// Matches [`UnifiedTimeline::end_to_end_latency`] for the same token.
    #[must_use]
    pub fn total_nanos(&self) -> u64 {
        self.last_end_nanos.saturating_sub(self.first_start_nanos)
    }

    /// Submit→execute latency: the gap between the CPU finishing submission and
    /// the GPU beginning execution.
    ///
    /// Returns `None` unless the chain has both a CPU and a GPU stage. The value
    /// saturates to `0` when the GPU began before the CPU submit stage ended
    /// (overlap), which is normal for pipelined work and simply means there was
    /// no observable submit stall.
    #[must_use]
    pub fn submit_to_execute_nanos(&self) -> Option<u64> {
        match (self.cpu_submit_end_nanos, self.gpu_execute_start_nanos) {
            (Some(cpu_end), Some(gpu_start)) => Some(gpu_start.saturating_sub(cpu_end)),
            _ => None,
        }
    }

    /// Span of GPU-side execution (`gpu_execute_end - gpu_execute_start`,
    /// saturating). `None` when the chain has no GPU stage.
    #[must_use]
    pub fn gpu_active_nanos(&self) -> Option<u64> {
        match (self.gpu_execute_start_nanos, self.gpu_execute_end_nanos) {
            (Some(start), Some(end)) => Some(end.saturating_sub(start)),
            _ => None,
        }
    }
}

/// An idle gap on a single GPU queue track between two consecutive execution
/// intervals.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuBubble {
    /// The queue the bubble was observed on.
    pub queue: GpuQueueId,
    /// Label of the span that ended just before the gap.
    pub after_label: String,
    /// Label of the span that started just after the gap.
    pub before_label: String,
    /// Start of the idle gap (end of the previous coverage), in nanoseconds.
    pub start_nanos: u64,
    /// Length of the idle gap, in nanoseconds.
    pub gap_nanos: u64,
}

impl GpuBubble {
    /// End of the idle gap (`start + gap`), in nanoseconds.
    #[must_use]
    pub fn end_nanos(&self) -> u64 {
        self.start_nanos.saturating_add(self.gap_nanos)
    }
}

impl UnifiedTimeline {
    /// Decompose the correlation chain for `id` into CPU-submit and GPU-execute
    /// stages so submit→execute latency and GPU-active span can be read out.
    ///
    /// Returns `None` if no entry carries the token.
    #[must_use]
    pub fn correlation_breakdown(&self, id: CorrelationId) -> Option<CorrelationBreakdown> {
        let mut first_start = u64::MAX;
        let mut last_end = 0u64;
        let mut cpu_submit_end: Option<u64> = None;
        let mut gpu_start: Option<u64> = None;
        let mut gpu_end: Option<u64> = None;
        let mut cpu_stage_count = 0u32;
        let mut gpu_stage_count = 0u32;
        let mut found = false;

        for e in self.entries().iter().filter(|e| e.correlation == Some(id)) {
            found = true;
            let start = e.start_nanos;
            let end = e.end_nanos();
            first_start = first_start.min(start);
            last_end = last_end.max(end);
            match e.track {
                TimelineTrack::Cpu { .. } => {
                    cpu_stage_count += 1;
                    cpu_submit_end = Some(cpu_submit_end.map_or(end, |cur| cur.max(end)));
                }
                TimelineTrack::Gpu { .. } => {
                    gpu_stage_count += 1;
                    gpu_start = Some(gpu_start.map_or(start, |cur| cur.min(start)));
                    gpu_end = Some(gpu_end.map_or(end, |cur| cur.max(end)));
                }
            }
        }

        found.then_some(CorrelationBreakdown {
            correlation: id,
            first_start_nanos: first_start,
            last_end_nanos: last_end,
            cpu_submit_end_nanos: cpu_submit_end,
            gpu_execute_start_nanos: gpu_start,
            gpu_execute_end_nanos: gpu_end,
            cpu_stage_count,
            gpu_stage_count,
        })
    }

    /// Idle gaps (bubbles) on a GPU `queue` track, in timeline order.
    ///
    /// Walks the queue's execution intervals in start order, tracking a running
    /// coverage end so that overlapping or nested spans do not report spurious
    /// gaps. A bubble is emitted whenever the next interval starts strictly
    /// after the current coverage ends. Returns an empty vector if the queue has
    /// zero or one span (no gap is possible).
    #[must_use]
    pub fn gpu_bubbles(&self, queue: GpuQueueId) -> Vec<GpuBubble> {
        let mut bubbles = Vec::new();
        // Entries are already globally sorted by (start, track-kind, track-id,
        // label), so filtering preserves per-queue start order.
        let mut iter = self
            .entries()
            .iter()
            .filter(|e| e.track == TimelineTrack::Gpu { queue });

        let Some(first) = iter.next() else {
            return bubbles;
        };
        let mut coverage_end = first.end_nanos();
        let mut prev_label = first.label.clone();

        for entry in iter {
            if entry.start_nanos > coverage_end {
                bubbles.push(GpuBubble {
                    queue,
                    after_label: prev_label.clone(),
                    before_label: entry.label.clone(),
                    start_nanos: coverage_end,
                    gap_nanos: entry.start_nanos - coverage_end,
                });
            }
            if entry.end_nanos() >= coverage_end {
                coverage_end = entry.end_nanos();
                prev_label = entry.label.clone();
            }
        }

        bubbles
    }

    /// Total idle time (sum of all bubble gaps) on a GPU `queue` track.
    #[must_use]
    pub fn gpu_idle_nanos(&self, queue: GpuQueueId) -> u64 {
        self.gpu_bubbles(queue)
            .iter()
            .fold(0u64, |acc, b| acc.saturating_add(b.gap_nanos))
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::string::String;
    use alloc::vec::Vec;

    use super::*;
    use crate::gpu::timeline::ProjectedGpuSpan;
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

    fn gpu_span(
        name: &str,
        queue: u32,
        start: u64,
        dur: u64,
        corr: Option<u64>,
    ) -> ProjectedGpuSpan {
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

    #[test]
    fn breakdown_splits_submit_and_execute() {
        let mut tl = UnifiedTimeline::new();
        let corr = CorrelationId(7);
        // CPU submit on thread 1: [100, 120).
        tl.add_cpu_span_correlated(&cpu_span("submit", 1, 100, 20), corr);
        // GPU execute on queue 0: [200, 500). GPU started 80ns after submit end.
        tl.add_gpu_span(&gpu_span("execute", 0, 200, 300, Some(7)));
        // Present on queue 0: [520, 530).
        tl.add_gpu_span(&gpu_span("present", 0, 520, 10, Some(7)));

        let bd = tl.correlation_breakdown(corr).unwrap();
        assert_eq!(bd.cpu_stage_count, 1);
        assert_eq!(bd.gpu_stage_count, 2);
        assert_eq!(bd.cpu_submit_end_nanos, Some(120));
        assert_eq!(bd.gpu_execute_start_nanos, Some(200));
        assert_eq!(bd.gpu_execute_end_nanos, Some(530));
        // submit end 120 -> gpu start 200 = 80ns stall.
        assert_eq!(bd.submit_to_execute_nanos(), Some(80));
        // GPU active span 200 -> 530 = 330ns.
        assert_eq!(bd.gpu_active_nanos(), Some(330));
        // Total 100 -> 530 = 430ns, matching end_to_end_latency.
        assert_eq!(bd.total_nanos(), 430);
        assert_eq!(tl.end_to_end_latency(corr), Some(430));
    }

    #[test]
    fn submit_to_execute_saturates_when_gpu_overlaps_submit() {
        let mut tl = UnifiedTimeline::new();
        let corr = CorrelationId(1);
        // CPU submit [100, 300); GPU already running at 150 (pipelined).
        tl.add_cpu_span_correlated(&cpu_span("submit", 1, 100, 200), corr);
        tl.add_gpu_span(&gpu_span("execute", 0, 150, 100, Some(1)));
        let bd = tl.correlation_breakdown(corr).unwrap();
        // gpu start 150 < cpu submit end 300 => saturating 0 (no observable stall).
        assert_eq!(bd.submit_to_execute_nanos(), Some(0));
    }

    #[test]
    fn breakdown_without_gpu_stage_has_no_submit_latency() {
        let mut tl = UnifiedTimeline::new();
        let corr = CorrelationId(2);
        tl.add_cpu_span_correlated(&cpu_span("submit", 1, 10, 5), corr);
        let bd = tl.correlation_breakdown(corr).unwrap();
        assert_eq!(bd.gpu_stage_count, 0);
        assert_eq!(bd.submit_to_execute_nanos(), None);
        assert_eq!(bd.gpu_active_nanos(), None);
    }

    #[test]
    fn missing_token_yields_no_breakdown() {
        let tl = UnifiedTimeline::new();
        assert_eq!(tl.correlation_breakdown(CorrelationId(99)), None);
    }

    #[test]
    fn bubbles_detect_queue_idle_gaps() {
        let mut tl = UnifiedTimeline::new();
        // Queue 0: [0,100), [150,200), [200,260) — one 50ns gap at 100..150.
        tl.add_gpu_span(&gpu_span("a", 0, 0, 100, None));
        tl.add_gpu_span(&gpu_span("b", 0, 150, 50, None));
        tl.add_gpu_span(&gpu_span("c", 0, 200, 60, None));
        let bubbles = tl.gpu_bubbles(GpuQueueId(0));
        assert_eq!(bubbles.len(), 1);
        assert_eq!(bubbles[0].start_nanos, 100);
        assert_eq!(bubbles[0].gap_nanos, 50);
        assert_eq!(bubbles[0].end_nanos(), 150);
        assert_eq!(bubbles[0].after_label, "a");
        assert_eq!(bubbles[0].before_label, "b");
        assert_eq!(tl.gpu_idle_nanos(GpuQueueId(0)), 50);
    }

    #[test]
    fn overlapping_spans_report_no_bubble() {
        let mut tl = UnifiedTimeline::new();
        // Nested/overlapping: [0,200) contains [50,100); [100,150) still inside.
        tl.add_gpu_span(&gpu_span("outer", 0, 0, 200, None));
        tl.add_gpu_span(&gpu_span("inner", 0, 50, 50, None));
        tl.add_gpu_span(&gpu_span("inner2", 0, 100, 50, None));
        assert!(tl.gpu_bubbles(GpuQueueId(0)).is_empty());
        assert_eq!(tl.gpu_idle_nanos(GpuQueueId(0)), 0);
    }

    #[test]
    fn bubbles_are_per_queue() {
        let mut tl = UnifiedTimeline::new();
        // Queue 0 is contiguous; queue 1 has a gap. Interleaving must not leak.
        tl.add_gpu_span(&gpu_span("q0a", 0, 0, 100, None));
        tl.add_gpu_span(&gpu_span("q1a", 1, 10, 20, None));
        tl.add_gpu_span(&gpu_span("q0b", 0, 100, 100, None));
        tl.add_gpu_span(&gpu_span("q1b", 1, 200, 20, None));
        assert!(tl.gpu_bubbles(GpuQueueId(0)).is_empty());
        let q1 = tl.gpu_bubbles(GpuQueueId(1));
        assert_eq!(q1.len(), 1);
        assert_eq!(q1[0].start_nanos, 30);
        assert_eq!(q1[0].gap_nanos, 170);
    }

    #[test]
    fn single_or_empty_queue_has_no_bubbles() {
        let mut tl = UnifiedTimeline::new();
        assert!(tl.gpu_bubbles(GpuQueueId(0)).is_empty());
        tl.add_gpu_span(&gpu_span("only", 0, 10, 5, None));
        assert!(tl.gpu_bubbles(GpuQueueId(0)).is_empty());
    }
}
