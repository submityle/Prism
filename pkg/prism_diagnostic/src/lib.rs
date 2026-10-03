//! # `prism_diagnostic`
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
//! ## M3 scope (this build) — ECS/tasks integration
//! - Automatic instrumentation in [`instrument`]:
//!   [`SystemScope`](instrument::SystemScope)/[`instrument_system`] time an ECS
//!   system's per-frame work, and [`JobFlow`](instrument::JobFlow)/
//!   [`JobScope`](instrument::JobScope) time a task enqueued on one thread and
//!   executed on another. The opt-in [`Instrumentable`](instrument::Instrumentable)
//!   trait lets `prism_ecs`/`prism_tasks` adopt this without a dependency edge.
//! - Cross-thread flow connection: [`trace::flow`] records Chrome flow
//!   (`s`/`t`/`f`) and async (`b`/`e`) events, serialized by the Chrome exporter
//!   with `id`/`cat`/`tid`, so a job that hops threads is visually linked.
//! - Load visualization: [`LoadProfile`](metrics::LoadProfile) rolls recorded
//!   spans up into per-thread utilization and per-system totals.
//!
//! ## M4 scope (this build, `gpu` feature) — GPU timing + unified timeline
//! - Backend-neutral GPU timing in [`gpu`]: a timestamp-query model
//!   ([`GpuSpan`](gpu::GpuSpan)/[`GpuTick`](gpu::GpuTick)), CPU↔GPU affine
//!   calibration ([`GpuClockCalibration`](gpu::GpuClockCalibration)), an
//!   N-frame readback correlation ring ([`GpuReadbackRing`](gpu::GpuReadbackRing)),
//!   cross-queue correlation tokens ([`CorrelationId`](gpu::CorrelationId)), and a
//!   [`UnifiedTimeline`](gpu::UnifiedTimeline) that projects CPU and GPU spans
//!   onto one axis. The Chrome exporter can emit an aligned GPU track via
//!   [`export_chrome_with_gpu`](trace::export_chrome_with_gpu).
//! - This is implemented as a pure ingestion API with no GPU-backend
//!   dependency; a future RHI backend feeds raw ticks in. Live multi-driver GPU
//!   validation is deferred to that backend (the ingestion seam makes it
//!   non-blocking).
//!
//! Later milestones add GPU timing scopes, lock-free Tracy/Perfetto sinks, and
//! crash/minidump capture.
//!
//! Depends on `prism_utils` (containers) and `prism_platform` (clock). Contains
//! no Unreal Engine source or derived code and depends on no `bevy_*` crate.

#![forbid(unsafe_code)]

pub mod filter;
pub mod fmt;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod instrument;
pub mod macros;
pub mod metrics;
pub mod model;
pub mod prelude;
pub mod sink;
pub mod span;
pub mod trace;

pub use filter::{max_level, set_max_level};
pub use instrument::{
    async_begin, async_end, flow_finish, flow_start, flow_step, instrument, instrument_system,
    next_flow_id, FlowId, Instrumentable, JobFlow, JobScope, SystemScope,
};
pub use metrics::{
    hud_lines, Counter, FrameStatsSnapshot, FrameTimer, Gauge, Histogram, HistogramSnapshot, Hud,
    HudSnapshot, LoadProfile, MetricRegistry, RegistrySnapshot, Sum, SystemLoad, ThreadLoad,
};
pub use model::{Event, Field, FieldValue, Level};
#[cfg(feature = "gpu")]
pub use gpu::{
    next_correlation_id, AffineFit, CalibrationSample, CorrelationId, GpuClockCalibration,
    GpuQueryId, GpuQueueId, GpuReadbackRing, GpuSpan, GpuTick, PendingQuery, ProjectedGpuSpan,
    TimelineEntry, TimelineTrack, UnifiedTimeline,
};
pub use sink::{clear_sink, set_sink, CaptureSink, ConsoleSink, FileSink, Sink};
pub use span::Scope;
pub use trace::{
    export_chrome_string, export_chrome_to_file, FlowPhase, FlowRecord, RingBuffer, SpanRecord,
    ThreadTrace,
};
#[cfg(feature = "gpu")]
pub use trace::export_chrome_with_gpu;

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
