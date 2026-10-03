//! Unified CPU + GPU timeline projection and cross-queue correlation.
//!
//! Once GPU spans are projected onto the CPU nanosecond base (see
//! [`GpuClockCalibration`](crate::gpu::calibration::GpuClockCalibration)), CPU
//! spans and GPU spans live on the *same* axis. [`UnifiedTimeline`] merges both
//! into one deterministically ordered list of [`TimelineEntry`]s, each tagged
//! with the [`TimelineTrack`] it belongs to (a CPU thread or a GPU queue).
//!
//! Because every stage of one logical operation can share a
//! [`CorrelationId`], the timeline can reconstruct end-to-end latency —
//! CPU submit → GPU execute → present — with
//! [`end_to_end_latency`](UnifiedTimeline::end_to_end_latency), even when the
//! stages were recorded independently on different tracks.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use super::scope::{CorrelationId, GpuQueueId};
use crate::trace::ring::SpanRecord;

/// A GPU span projected onto the CPU nanosecond timeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectedGpuSpan {
    /// Scope label.
    pub label: String,
    /// Queue the span was measured on.
    pub queue: GpuQueueId,
    /// Start on the CPU monotonic timeline, in nanoseconds.
    pub cpu_start_nanos: u64,
    /// Duration on the CPU timeline, in nanoseconds.
    pub cpu_duration_nanos: u64,
    /// Optional cross-queue correlation token.
    pub correlation: Option<CorrelationId>,
    /// Frame the span was issued on.
    pub frame: u64,
    /// Nesting depth within the frame's GPU scopes.
    pub depth: u32,
}

/// Which track a [`TimelineEntry`] belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TimelineTrack {
    /// A CPU thread track, identified by its Prism-assigned thread id.
    Cpu {
        /// Prism-assigned thread id.
        thread_id: u64,
    },
    /// A GPU queue track.
    Gpu {
        /// The GPU queue.
        queue: GpuQueueId,
    },
}

impl TimelineTrack {
    /// A small ordinal placing CPU tracks before GPU tracks in deterministic
    /// ordering ties.
    const fn kind_ord(self) -> u8 {
        match self {
            TimelineTrack::Cpu { .. } => 0,
            TimelineTrack::Gpu { .. } => 1,
        }
    }
}

/// One span on the unified timeline, source-track-tagged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimelineEntry {
    /// The track this entry belongs to.
    pub track: TimelineTrack,
    /// Scope label.
    pub label: String,
    /// Start on the CPU timeline, in nanoseconds.
    pub start_nanos: u64,
    /// Duration on the CPU timeline, in nanoseconds.
    pub duration_nanos: u64,
    /// Optional cross-queue correlation token.
    pub correlation: Option<CorrelationId>,
}

impl TimelineEntry {
    /// End of this entry on the timeline (`start + duration`, saturating).
    #[must_use]
    pub fn end_nanos(&self) -> u64 {
        self.start_nanos.saturating_add(self.duration_nanos)
    }
}

/// A deterministically ordered merge of CPU and GPU spans on one timeline.
///
/// Build with [`add_cpu_span`](Self::add_cpu_span) /
/// [`add_gpu_span`](Self::add_gpu_span), then read [`entries`](Self::entries),
/// which are kept sorted by `(start_nanos, track-kind, track-id, label)` so the
/// ordering is stable and reproducible.
#[derive(Clone, Debug, Default)]
pub struct UnifiedTimeline {
    entries: Vec<TimelineEntry>,
}

impl UnifiedTimeline {
    /// Create an empty timeline.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add a CPU span (an existing [`SpanRecord`]) to the timeline.
    pub fn add_cpu_span(&mut self, span: &SpanRecord) {
        self.push(TimelineEntry {
            track: TimelineTrack::Cpu {
                thread_id: span.thread_id,
            },
            label: span.name.clone(),
            start_nanos: span.start_nanos,
            duration_nanos: span.duration_nanos,
            // CPU spans carry no GPU correlation token.
            correlation: None,
        });
    }

    /// Add a CPU span that participates in a cross-queue correlation (e.g. the
    /// CPU-side submit of a draw whose GPU execution shares the same token).
    pub fn add_cpu_span_correlated(&mut self, span: &SpanRecord, correlation: CorrelationId) {
        self.push(TimelineEntry {
            track: TimelineTrack::Cpu {
                thread_id: span.thread_id,
            },
            label: span.name.clone(),
            start_nanos: span.start_nanos,
            duration_nanos: span.duration_nanos,
            correlation: Some(correlation),
        });
    }

    /// Add a projected GPU span to the timeline.
    pub fn add_gpu_span(&mut self, span: &ProjectedGpuSpan) {
        self.push(TimelineEntry {
            track: TimelineTrack::Gpu { queue: span.queue },
            label: span.label.clone(),
            start_nanos: span.cpu_start_nanos,
            duration_nanos: span.cpu_duration_nanos,
            correlation: span.correlation,
        });
    }

    /// Insert `entry`, keeping [`entries`](Self::entries) sorted.
    fn push(&mut self, entry: TimelineEntry) {
        let key = Self::sort_key(&entry);
        let pos = self
            .entries
            .partition_point(|e| Self::sort_key(e) <= key);
        self.entries.insert(pos, entry);
    }

    /// Deterministic ordering key.
    fn sort_key(entry: &TimelineEntry) -> (u64, u8, u64, &str) {
        let track_id = match entry.track {
            TimelineTrack::Cpu { thread_id } => thread_id,
            TimelineTrack::Gpu { queue } => queue.0 as u64,
        };
        (
            entry.start_nanos,
            entry.track.kind_ord(),
            track_id,
            entry.label.as_str(),
        )
    }

    /// The timeline entries in deterministic order.
    #[must_use]
    pub fn entries(&self) -> &[TimelineEntry] {
        &self.entries
    }

    /// Number of entries on the timeline.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the timeline has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// All entries that share correlation token `id`, in timeline order.
    #[must_use]
    pub fn correlated(&self, id: CorrelationId) -> Vec<&TimelineEntry> {
        self.entries
            .iter()
            .filter(|e| e.correlation == Some(id))
            .collect()
    }

    /// End-to-end latency for a correlation token: the span from the earliest
    /// start to the latest end across every stage sharing `id`.
    ///
    /// Returns `None` if no entry carries the token.
    #[must_use]
    pub fn end_to_end_latency(&self, id: CorrelationId) -> Option<u64> {
        let mut min_start = u64::MAX;
        let mut max_end = 0u64;
        let mut found = false;
        for e in self.entries.iter().filter(|e| e.correlation == Some(id)) {
            found = true;
            min_start = min_start.min(e.start_nanos);
            max_end = max_end.max(e.end_nanos());
        }
        found.then(|| max_end.saturating_sub(min_start))
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::string::String;
    use alloc::vec::Vec;

    use super::*;

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

    #[test]
    fn entries_sorted_by_start_then_track() {
        let mut tl = UnifiedTimeline::new();
        tl.add_gpu_span(&gpu_span("gpu_late", 0, 300, 10, None));
        tl.add_cpu_span(&cpu_span("cpu_mid", 1, 100, 10));
        // Same start as the CPU span: CPU track sorts before GPU track.
        tl.add_gpu_span(&gpu_span("gpu_tie", 0, 100, 10, None));
        tl.add_cpu_span(&cpu_span("cpu_early", 2, 50, 10));

        let order: Vec<_> = tl.entries().iter().map(|e| e.label.as_str()).collect();
        assert_eq!(order, ["cpu_early", "cpu_mid", "gpu_tie", "gpu_late"]);
    }

    #[test]
    fn ordering_is_insertion_independent() {
        let mut a = UnifiedTimeline::new();
        let mut b = UnifiedTimeline::new();

        let c1 = cpu_span("c1", 1, 10, 5);
        let c2 = cpu_span("c2", 1, 20, 5);
        let g1 = gpu_span("g1", 0, 15, 5, None);

        a.add_cpu_span(&c1);
        a.add_gpu_span(&g1);
        a.add_cpu_span(&c2);

        b.add_gpu_span(&g1);
        b.add_cpu_span(&c2);
        b.add_cpu_span(&c1);

        let la: Vec<_> = a.entries().iter().map(|e| e.label.as_str()).collect();
        let lb: Vec<_> = b.entries().iter().map(|e| e.label.as_str()).collect();
        assert_eq!(la, lb);
        assert_eq!(la, ["c1", "g1", "c2"]);
    }

    #[test]
    fn end_to_end_latency_across_tracks() {
        let mut tl = UnifiedTimeline::new();
        let corr = CorrelationId(42);
        // CPU submit on thread 1 at t=100 for 20ns.
        tl.add_cpu_span_correlated(&cpu_span("submit", 1, 100, 20), corr);
        // GPU execute on queue 0 at t=200 for 300ns (ends at 500).
        tl.add_gpu_span(&gpu_span("execute", 0, 200, 300, Some(42)));
        // Present on queue 0 at t=520 for 10ns (ends at 530).
        tl.add_gpu_span(&gpu_span("present", 0, 520, 10, Some(42)));
        // An unrelated span not in the correlation.
        tl.add_cpu_span(&cpu_span("other", 1, 50, 5));

        let chain = tl.correlated(corr);
        assert_eq!(chain.len(), 3);
        // earliest start 100, latest end 530 => 430ns end-to-end.
        assert_eq!(tl.end_to_end_latency(corr), Some(430));
        assert_eq!(tl.end_to_end_latency(CorrelationId(999)), None);
    }

    #[test]
    fn entry_end_saturates() {
        let e = TimelineEntry {
            track: TimelineTrack::Cpu { thread_id: 1 },
            label: String::from("x"),
            start_nanos: u64::MAX,
            duration_nanos: 10,
            correlation: None,
        };
        assert_eq!(e.end_nanos(), u64::MAX);
    }
}
