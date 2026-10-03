//! # prism_diagnostic
//!
//! Prism's diagnostic kernel: a cross-cutting logging/observability surface
//! that every subsystem depends on by "instrumenting" itself.
//!
//! ## M0 scope (this build) — the logging skeleton
//! - [`Level`] + [`Event`]/[`Field`] structured model.
//! - Logging macros ([`info!`], [`warn!`], [`error!`], [`debug!`], [`trace!`],
//!   [`event!`]) with compile-time (`max_level_*` features) and runtime level
//!   filtering.
//! - Pluggable [`Sink`](sink::Sink)s: [`ConsoleSink`](sink::ConsoleSink),
//!   [`FileSink`](sink::FileSink), and an in-memory
//!   [`CaptureSink`](sink::CaptureSink).
//!
//! ## M2 scope (this build) — counters + frame statistics
//! - Metric instruments in [`metrics`]: [`Gauge`](metrics::Gauge) (last value),
//!   [`Counter`](metrics::Counter)/[`Sum`](metrics::Sum) (monotonic add), and
//!   [`Histogram`](metrics::Histogram) (configurable buckets with
//!   percentile/mean/min/max), all behind a name-keyed
//!   [`MetricRegistry`](metrics::MetricRegistry).
//! - Built-in [`FrameTimer`](metrics::FrameTimer) frame statistics (delta, FPS,
//!   sliding-window min/avg/max) plus per-frame named counters (drawcalls,
//!   triangles, ...).
//! - A [`Hud`](metrics::Hud) data provider that formats metrics + frame stats
//!   into overlay text lines; this crate performs no rendering.
//!
//! Later milestones add GPU timing scopes, lock-free Tracy/Perfetto sinks, and
//! crash/minidump capture.
//!
//! Depends on `prism_utils` (containers) and `prism_platform` (clock). Contains
//! no Unreal Engine source or derived code and depends on no `bevy_*` crate.

#![forbid(unsafe_code)]

pub mod filter;
pub mod fmt;
pub mod macros;
pub mod metrics;
pub mod model;
pub mod prelude;
pub mod sink;
pub mod span;
pub mod trace;

pub use filter::{max_level, set_max_level};
pub use metrics::{
    hud_lines, Counter, FrameStatsSnapshot, FrameTimer, Gauge, Histogram, HistogramSnapshot, Hud,
    HudSnapshot, MetricRegistry, RegistrySnapshot, Sum,
};
pub use model::{Event, Field, FieldValue, Level};
pub use sink::{clear_sink, set_sink, CaptureSink, ConsoleSink, FileSink, Sink};
pub use span::Scope;
pub use trace::{
    export_chrome_string, export_chrome_to_file, RingBuffer, SpanRecord, ThreadTrace,
};

/// Macro support: build and dispatch an event from `format_args!` output.
/// Not part of the stable surface; call the logging macros instead.
#[doc(hidden)]
pub fn __dispatch_message(level: Level, target: &'static str, args: core::fmt::Arguments<'_>) {
    use alloc::string::ToString as _;
    let event = Event::new(level, target, args.to_string());
    sink::dispatch(&event);
}

extern crate alloc;

#[cfg(test)]
mod tests;
