//! GPU timing: timestamp-query scopes + CPU/GPU timeline alignment (`gpu`
//! feature, M4).
//!
//! This module adds GPU-side timing to the diagnostic kernel and projects it
//! onto the same timeline as CPU spans, so you can see "CPU submitted early but
//! the GPU stalled waiting on a dependency" at a glance (design §8, §24.2).
//!
//! ## Backend-neutral by design (the ingestion seam)
//! Prism has no central RHI / render-driver crate yet, and this crate must not
//! depend on any GPU backend. So GPU timing is implemented as a pure
//! data-structures-and-math *ingestion API*: a future RHI backend feeds raw GPU
//! data in; this module stores, aligns, and projects it. Nothing here talks to
//! a GPU, allocates queries, or blocks a frame.
//!
//! The seam has three inputs a backend provides:
//! - raw timestamp ticks per query pair ([`scope::GpuTick`] /
//!   [`scope::GpuSpan`]),
//! - the queue's `timestamp_period` plus periodic `(cpu_now, gpu_now)`
//!   calibration samples ([`calibration::GpuClockCalibration`]),
//! - issue/resolve calls that correlate queries to the frame they were issued
//!   on ([`ring::GpuReadbackRing`]).
//!
//! From those it produces a [`timeline::UnifiedTimeline`] and
//! [`timeline::ProjectedGpuSpan`]s the Chrome exporter can emit as a GPU track
//! aligned to the CPU tracks (see
//! [`export_chrome_with_gpu`](crate::trace::export_chrome_with_gpu)).
//!
//! ## Dependency boundary (honest scope)
//! Real GPU-timestamp *accuracy* and multi-driver validation (D3D12 / Vulkan /
//! Metal query semantics, `timestamp_period` quirks, disjoint-query handling,
//! queue clock domains) are **deferred until an RHI backend wires into this
//! seam**. That is not blocking: the ingestion API is the contract, and the
//! alignment math, readback correlation, and timeline projection here are fully
//! unit-tested on CPU with synthetic ticks. Span recording at runtime is driven
//! by a backend feeding this API — this crate never fabricates GPU behavior.
//!
//! ## Layout
//! - [`scope`] — the raw ingestion vocabulary (ticks, queues, correlation
//!   tokens, resolved GPU spans).
//! - [`calibration`] — the CPU↔GPU affine fit and span projection.
//! - [`ring`] — the N-frame readback correlation ring.
//! - [`timeline`] — the unified CPU+GPU timeline and cross-queue latency
//!   reconstruction.

pub mod calibration;
pub mod ring;
pub mod scope;
pub mod timeline;

use core::sync::atomic::{AtomicU64, Ordering};

pub use calibration::{AffineFit, CalibrationSample, GpuClockCalibration, DEFAULT_CALIBRATION_WINDOW};
pub use ring::{GpuQueryId, GpuReadbackRing, PendingQuery, DEFAULT_RESOLVED_CAPACITY};
pub use scope::{CorrelationId, GpuQueueId, GpuSpan, GpuTick};
pub use timeline::{ProjectedGpuSpan, TimelineEntry, TimelineTrack, UnifiedTimeline};

/// Allocate the next process-unique [`CorrelationId`].
///
/// Mint one at the CPU record site and thread the same token through submit →
/// GPU execute → present so [`UnifiedTimeline::end_to_end_latency`] can
/// reconstruct the operation's end-to-end latency across tracks.
pub fn next_correlation_id() -> CorrelationId {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    CorrelationId(COUNTER.fetch_add(1, Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::string::String;
    use alloc::vec::Vec;

    use super::*;

    #[test]
    fn correlation_ids_are_unique_and_monotonic() {
        let a = next_correlation_id();
        let b = next_correlation_id();
        let c = next_correlation_id();
        assert!(a.0 < b.0 && b.0 < c.0);
    }

    #[test]
    fn end_to_end_ingestion_flow() {
        // Simulate a backend: issue queries, calibrate, resolve ticks, project,
        // and reconstruct end-to-end latency on the unified timeline.
        let corr = next_correlation_id();
        let mut ring = GpuReadbackRing::new(2);

        // Frame 7: issue a GPU scope on the graphics queue.
        let qid = ring.issue(7, "ShadowPass", GpuQueueId::GRAPHICS, Some(corr), 0);
        assert_eq!(ring.pending_len(), 1);

        // Two frames later the backend reads the ticks back and resolves.
        assert!(ring.ready_queries(9).contains(&qid));
        let gpu_span = ring.resolve(qid, GpuTick(1_000), GpuTick(1_500)).unwrap();

        // Calibration: 2 ns/tick, offset 100 ns (cpu_ns = 2*tick + 100).
        let mut cal = GpuClockCalibration::new(2.0);
        for tick in [0u64, 500, 1000, 1500] {
            cal.add_sample(2 * tick + 100, GpuTick(tick));
        }
        let projected = cal.project_span(&gpu_span);
        assert_eq!(projected.cpu_start_nanos, 2_100); // 2*1000 + 100
        assert_eq!(projected.cpu_duration_nanos, 1_000); // 2 * 500

        // Build a unified timeline with a correlated CPU submit span.
        let mut tl = UnifiedTimeline::new();
        let submit = crate::trace::SpanRecord {
            name: String::from("submit"),
            category: None,
            thread_id: 1,
            start_nanos: 2_000,
            duration_nanos: 50,
            depth: 0,
            args: Vec::new(),
        };
        tl.add_cpu_span_correlated(&submit, corr);
        tl.add_gpu_span(&projected);

        // Earliest start = CPU submit at 2000; latest end = GPU end at 3100.
        assert_eq!(tl.end_to_end_latency(corr), Some(1_100));
    }
}
