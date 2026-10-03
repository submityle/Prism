//! Profiling event vocabulary: zones, frame marks, plots, and GPU zones.
//!
//! These are the backend-neutral records a [`Profiler`](super::Profiler)
//! backend consumes. A *zone* is one completed scope (name + wall interval +
//! nesting depth), the real-time analogue of the timeline's
//! [`SpanRecord`](crate::trace::SpanRecord); a *frame mark* delimits a frame; a
//! *plot* samples a named numeric series; a *GPU zone* carries a resolved
//! GPU-side interval. Backends (Tracy, custom panels, ...) translate these into
//! their own representation.

extern crate alloc;

use alloc::string::String;

use crate::trace::SpanRecord;

/// A completed profiling zone: one closed scope on one thread.
///
/// Unlike a streaming begin/end pair this carries the full interval, which maps
/// cleanly onto the timeline's completed-span model and keeps backends simple.
/// A backend that wants begin/end semantics (such as Tracy) can split this into
/// two wire messages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Zone {
    /// Human-readable zone name.
    pub name: String,
    /// Optional category/track label.
    pub category: Option<String>,
    /// Owning thread id (Prism-assigned, dense from 1).
    pub thread_id: u64,
    /// Start timestamp in nanoseconds (monotonic clock).
    pub start_nanos: u64,
    /// Measured wall duration in nanoseconds.
    pub duration_nanos: u64,
    /// Nesting depth at entry (0 is top level).
    pub depth: u32,
}

impl Zone {
    /// Inclusive end timestamp (`start + duration`), saturating on overflow.
    pub fn end_nanos(&self) -> u64 {
        self.start_nanos.saturating_add(self.duration_nanos)
    }
}

impl From<&SpanRecord> for Zone {
    fn from(span: &SpanRecord) -> Self {
        Self {
            name: span.name.clone(),
            category: span.category.clone(),
            thread_id: span.thread_id,
            start_nanos: span.start_nanos,
            duration_nanos: span.duration_nanos,
            depth: span.depth,
        }
    }
}

/// A frame boundary marker.
///
/// Tracy distinguishes the primary continuous frame (the main loop) from named
/// secondary frames (for example a render-thread frame); this mirrors that.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FrameMark {
    /// The primary, continuous frame boundary.
    Continuous,
    /// A named secondary frame boundary.
    Named(String),
}

/// A numeric sample for a named plot/counter series.
///
/// Kept as a typed enum (rather than always `f64`) so integer counters survive
/// without precision loss across the wire.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlotValue {
    /// Signed integer sample.
    I64(i64),
    /// Unsigned integer sample.
    U64(u64),
    /// Floating-point sample.
    F64(f64),
}

impl PlotValue {
    /// Lossy conversion to `f64` for backends that only plot floats.
    pub fn as_f64(self) -> f64 {
        match self {
            PlotValue::I64(v) => v as f64,
            PlotValue::U64(v) => v as f64,
            PlotValue::F64(v) => v,
        }
    }
}

/// A resolved GPU-side zone, correlated to a queue and (optionally) a frame.
///
/// This is the profiler-facing projection of the `gpu` module's resolved spans;
/// it carries already-aligned nanosecond timestamps so a backend never needs to
/// know about `timestamp_period` or clock calibration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuZone {
    /// Human-readable zone name.
    pub name: String,
    /// Backend-assigned queue id this zone executed on.
    pub queue_id: u32,
    /// Start timestamp in nanoseconds, projected onto the CPU clock.
    pub start_nanos: u64,
    /// Measured GPU duration in nanoseconds.
    pub duration_nanos: u64,
    /// Optional CPU-side correlation token threaded submit → execute → present.
    pub correlation: Option<u64>,
}

impl GpuZone {
    /// Inclusive end timestamp (`start + duration`), saturating on overflow.
    pub fn end_nanos(&self) -> u64 {
        self.start_nanos.saturating_add(self.duration_nanos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn zone_from_span_record_preserves_fields() {
        let span = SpanRecord {
            name: String::from("update"),
            category: Some(String::from("ecs")),
            thread_id: 3,
            start_nanos: 100,
            duration_nanos: 50,
            depth: 2,
            args: Vec::new(),
        };
        let zone = Zone::from(&span);
        assert_eq!(zone.name, "update");
        assert_eq!(zone.category.as_deref(), Some("ecs"));
        assert_eq!(zone.thread_id, 3);
        assert_eq!(zone.end_nanos(), 150);
        assert_eq!(zone.depth, 2);
    }

    #[test]
    fn plot_value_as_f64() {
        assert_eq!(PlotValue::I64(-2).as_f64(), -2.0);
        assert_eq!(PlotValue::U64(5).as_f64(), 5.0);
        assert_eq!(PlotValue::F64(1.5).as_f64(), 1.5);
    }

    #[test]
    fn gpu_zone_end_saturates() {
        let z = GpuZone {
            name: String::from("pass"),
            queue_id: 0,
            start_nanos: u64::MAX - 1,
            duration_nanos: 10,
            correlation: Some(7),
        };
        assert_eq!(z.end_nanos(), u64::MAX);
    }
}
