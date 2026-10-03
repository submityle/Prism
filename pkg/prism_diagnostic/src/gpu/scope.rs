//! GPU scope / timestamp-query data model (the ingestion vocabulary).
//!
//! A GPU scope is the moral equivalent of a CPU [`Scope`](crate::span::Scope),
//! but it is measured on the GPU timeline rather than the CPU one. A backend
//! inserts a *timestamp-query pair* around a range of GPU work (a `begin` query
//! and an `end` query); when those queries resolve, the backend reads back two
//! raw [`GpuTick`]s. This module defines the plain data the backend feeds in;
//! it performs no GPU work itself.
//!
//! ## Why ticks, not nanoseconds
//! GPU timestamp queries count in device-specific *ticks*, not nanoseconds. The
//! conversion factor (`timestamp_period`, nanoseconds-per-tick) is a property of
//! the device/queue and is applied by [`GpuClockCalibration`] during timeline
//! projection — see [`crate::gpu::calibration`]. Keeping raw ticks here means
//! the ingestion seam carries exactly what hardware reports, with no lossy
//! early conversion.

extern crate alloc;

use alloc::string::String;

/// A raw GPU timestamp counter value, in device ticks.
///
/// The value is meaningful only relative to other ticks from the *same* queue
/// (clocks are not comparable across queues without correlation) and only after
/// scaling by the queue's `timestamp_period`. See
/// [`GpuClockCalibration`](crate::gpu::calibration::GpuClockCalibration).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuTick(pub u64);

impl GpuTick {
    /// Ticks elapsed since `earlier` on the same queue (saturating).
    #[must_use]
    pub fn saturating_since(self, earlier: GpuTick) -> u64 {
        self.0.saturating_sub(earlier.0)
    }
}

/// Identifies a GPU queue (graphics, compute, copy, ...). Timestamps are only
/// directly comparable within a single queue; cross-queue reasoning goes
/// through a [`CorrelationId`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuQueueId(pub u32);

impl GpuQueueId {
    /// The conventional primary graphics queue.
    pub const GRAPHICS: GpuQueueId = GpuQueueId(0);
}

/// A cross-queue / cross-stage correlation token.
///
/// The same token is threaded through the lifetime of one logical GPU operation
/// — CPU record → queue submit → GPU execute → present — so that end-to-end
/// latency can be reconstructed on the unified timeline even though each stage
/// is recorded independently (and possibly on different queues/threads). Mint
/// one with [`next_correlation_id`](crate::gpu::next_correlation_id).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CorrelationId(pub u64);

/// A resolved GPU span: one timestamp-query pair whose `begin`/`end` ticks have
/// been read back from the device.
///
/// This is what the readback ring produces once a backend reports results for a
/// previously issued query (see [`crate::gpu::ring`]). It still lives on the GPU
/// tick timeline; converting it to the CPU timeline is
/// [`GpuClockCalibration::project_span`](crate::gpu::calibration::GpuClockCalibration::project_span).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuSpan {
    /// Human-readable scope label (e.g. `"ShadowPass"`).
    pub label: String,
    /// The queue the timestamps were taken on.
    pub queue: GpuQueueId,
    /// Raw tick recorded by the `begin` timestamp query.
    pub begin_tick: GpuTick,
    /// Raw tick recorded by the `end` timestamp query.
    pub end_tick: GpuTick,
    /// Optional cross-queue correlation token.
    pub correlation: Option<CorrelationId>,
    /// Frame number the query pair was issued on.
    pub frame: u64,
    /// Nesting depth within the frame's GPU scopes (0 is top level).
    pub depth: u32,
}

impl GpuSpan {
    /// Duration in raw GPU ticks (`end - begin`, saturating).
    ///
    /// Returns `0` if the end tick precedes the begin tick, which can happen
    /// when a backend reports an out-of-order or dropped query; callers that
    /// care should treat a zero duration as suspect.
    #[must_use]
    pub fn duration_ticks(&self) -> u64 {
        self.end_tick.saturating_since(self.begin_tick)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_delta_saturates() {
        assert_eq!(GpuTick(100).saturating_since(GpuTick(40)), 60);
        assert_eq!(GpuTick(40).saturating_since(GpuTick(100)), 0);
    }

    #[test]
    fn span_duration_in_ticks() {
        let span = GpuSpan {
            label: String::from("ShadowPass"),
            queue: GpuQueueId::GRAPHICS,
            begin_tick: GpuTick(1_000),
            end_tick: GpuTick(1_750),
            correlation: Some(CorrelationId(7)),
            frame: 3,
            depth: 0,
        };
        assert_eq!(span.duration_ticks(), 750);
    }

    #[test]
    fn reversed_ticks_yield_zero_duration() {
        let span = GpuSpan {
            label: String::from("weird"),
            queue: GpuQueueId::GRAPHICS,
            begin_tick: GpuTick(2_000),
            end_tick: GpuTick(1_000),
            correlation: None,
            frame: 0,
            depth: 0,
        };
        assert_eq!(span.duration_ticks(), 0);
    }
}
