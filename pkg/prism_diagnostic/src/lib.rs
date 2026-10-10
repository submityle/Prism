//! # `prism_diagnostic`
//!
//! Prism's diagnostic kernel: a cross-cutting logging/observability surface
//! that every subsystem depends on by "instrumenting" itself.
//!
//! ## Milestone status
//! - M0 logging skeleton — done.
//! - M1 scope timing / Chrome Trace — done.
//! - M2 counters + frame statistics — done.
//! - M3 ECS/tasks integration — done.
//! - M4 `GPU` timing + unified timeline (`gpu`) — done.
//! - M5 Tracy + remote observability (`tracy`/`remote`) — done.
//! - M6 crash + memory profiling + hitch — done (this build).
//!
//! ## M0 scope — the logging skeleton
//! - [`Level`] + [`Event`]/[`Field`] structured model.
//! - Logging macros ([`info!`], [`warn!`], [`error!`], [`debug!`], [`trace!`],
//!   [`event!`]) with compile-time (`max_level_*` features) and runtime level
//!   filtering.
//! - Pluggable [`Sink`](sink::Sink)s: [`ConsoleSink`](sink::ConsoleSink),
//!   [`FileSink`](sink::FileSink), and an in-memory
//!   [`CaptureSink`](sink::CaptureSink).
//!
//! ## M2 scope — counters + frame statistics
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
//! ## M3 scope — ECS/tasks integration
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
//! ## M4 scope (`gpu` feature) — `GPU` timing + unified timeline
//! - Backend-neutral `GPU` timing in [`gpu`]: a timestamp-query model
//!   ([`GpuSpan`](gpu::GpuSpan)/[`GpuTick`](gpu::GpuTick)), CPU↔GPU affine
//!   calibration ([`GpuClockCalibration`](gpu::GpuClockCalibration)), an
//!   N-frame readback correlation ring ([`GpuReadbackRing`](gpu::GpuReadbackRing)),
//!   cross-queue correlation tokens ([`CorrelationId`](gpu::CorrelationId)), and a
//!   [`UnifiedTimeline`](gpu::UnifiedTimeline) that projects CPU and GPU spans
//!   onto one axis. The Chrome exporter can emit an aligned `GPU` track via
//!   [`export_chrome_with_gpu`](trace::export_chrome_with_gpu).
//!
//! ## M5 scope — Tracy + remote observability
//! - A backend-neutral real-time profiling surface in [`profiler`]: the
//!   [`Profiler`](profiler::Profiler) backend trait, a global slot, the
//!   [`ProfiledZone`](profiler::ProfiledZone) `RAII` guard
//!   ([`profiled_zone!`]), and free functions that feed
//!   zones/frames/plots/messages/GPU zones to the installed backend. The default
//!   backend is the no-op [`NoopProfiler`](profiler::NoopProfiler); a
//!   self-contained Tracy-compatible emitter ([`profiler::tracy`]) lives behind
//!   the `tracy` feature (no external `tracy-client` dependency).
//! - A remote observability transport in [`remote`] (behind the `remote`
//!   feature): a serializable protocol with a hand-rolled binary codec
//!   ([`wire`]), shared lock-free
//!   [`RuntimeControls`](remote::RuntimeControls), and a loopback
//!   [`RemoteServer`](remote::RemoteServer)/[`RemoteClient`](remote::RemoteClient)
//!   over `std::net`.
//!
//! ## M6 scope — crash + memory profiling + hitch
//! - Crash reporting in [`crash`] (behind the `crash` feature): a self-contained,
//!   serializable [`CrashContext`](crash::CrashContext) model (crashing-thread
//!   backtrace + [`RegisterSnapshot`](crash::RegisterSnapshot), module list,
//!   reason/signal, timestamp, build id) and a deterministic binary
//!   [`CrashReport`](crash::CrashReport) writer built on [`wire`]. `OS`
//!   signal/exception capture is intentionally left to `prism_platform`; this
//!   crate owns the report *format* and accepts a synthetic or real context.
//! - Allocation tracking in [`alloc_track`] (behind the `alloc-track` feature):
//!   a [`TrackingAllocator`](alloc_track::TrackingAllocator) `GlobalAlloc`
//!   wrapper that accounts live/peak bytes, alloc/free counts, and optional
//!   per-callsite tags, with enable/disable and a
//!   [`snapshot`](alloc_track::snapshot)/[`tag_report`](alloc_track::tag_report)
//!   API. This is the one module that uses `unsafe`.
//! - Hitch detection in [`hitch`]: a [`HitchDetector`](hitch::HitchDetector)
//!   frame-time monitor flagging frames over a budget or a rolling-percentile
//!   spike threshold as [`HitchEvent`](hitch::HitchEvent)s.
//! - Deterministic replay markers in [`replay`]: a
//!   [`ReplayTimeline`](replay::ReplayTimeline) of tagged
//!   [`ReplayMarker`](replay::ReplayMarker)s that
//!   [`compare_timelines`](replay::compare_timelines) matches across runs to
//!   localize the first divergence.
//!
//! Depends on `prism_utils` (containers) and `prism_platform` (clock). Contains
//! no Unreal Engine source or derived code and depends on no `bevy_*` crate.

// The crate is `unsafe`-free except for the `alloc-track` `GlobalAlloc` wrapper,
// which inherently requires `unsafe`. `forbid` cannot be relaxed locally, so it
// is downgraded to the workspace `deny` only when that feature is enabled; the
// single unsafe module then carries an `#[expect(unsafe_code, reason = ...)]`.
#![cfg_attr(not(feature = "alloc-track"), forbid(unsafe_code))]

pub mod aggregate;
#[cfg(feature = "alloc-track")]
pub mod alloc_track;
pub mod budget;
#[cfg(feature = "crash")]
pub mod crash;
pub mod determinism;
pub mod filter;
pub mod fmt;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod hitch;
pub mod instrument;
pub mod macros;
pub mod mem;
pub mod metrics;
pub mod model;
pub mod prelude;
pub mod profiler;
#[cfg(feature = "remote")]
pub mod remote;
pub mod replay;
pub mod sampling;
pub mod sink;
pub mod span;
pub mod telemetry;
pub mod trace;
#[cfg(any(feature = "tracy", feature = "remote", feature = "crash"))]
pub mod wire;

pub use aggregate::{
    estimate_fleet_reports, AnomalyConfig, AssembledTrace, ClusterAggregator,
    ClusterFrametimeReport, CriticalPath, DistributedSpan, InstanceAnomaly, InstanceFrameReport,
    InstanceSummary, SampleRate, SampleReason, SamplingController, SamplingDecision,
    SamplingPolicy, ServiceLatency, SpanId, SpanKind, TraceAssembler, TraceId, TraceNode,
};
#[cfg(feature = "alloc-track")]
pub use alloc_track::{
    register_tag, reset_all, reset_peak, set_enabled, snapshot as alloc_snapshot, tag_report,
    tag_scope, AllocSnapshot, LiveTrackingAllocator, TagId, TagScope, TagStat, TrackingAllocator,
};
pub use budget::{
    hotspot_diff, Baseline, BudgetRegistry, BudgetStatus, FrameBudget, FrameBudgetReport, Hotspot,
    HotspotDelta, RegressionAlert, RegressionConfig, RegressionTracker,
};
#[cfg(feature = "crash")]
pub use crash::{
    CrashContext, CrashReason, CrashReport, ModuleEntry, RegisterSnapshot, StackFrame,
    ThreadContext, CRASH_REPORT_MAGIC, CRASH_REPORT_VERSION,
};
pub use determinism::{
    compare as compare_traces, fnv1a_64 as determinism_fnv1a_64, DeterminismTrace, FrameHash,
    FrameInput, InputRecorder, InputReplay, StateHasher, TraceDiff,
};
pub use filter::{max_level, set_max_level};
#[cfg(feature = "gpu")]
pub use gpu::{
    next_correlation_id, AffineFit, CalibrationSample, CorrelationId, GpuClockCalibration,
    GpuQueryId, GpuQueueId, GpuReadbackRing, GpuSpan, GpuTick, PendingQuery, ProjectedGpuSpan,
    TimelineEntry, TimelineTrack, UnifiedTimeline,
};
pub use hitch::{HitchConfig, HitchDetector, HitchEvent};
pub use instrument::{
    async_begin, async_end, flow_finish, flow_start, flow_step, instrument, instrument_system,
    next_flow_id, FlowId, Instrumentable, JobFlow, JobScope, SystemScope,
};
pub use mem::{
    analyze_fragmentation, occupancy_map, FragmentationReport, LeakCheckpoint, LeakReport,
    MemBudget, MemBudgetRegistry, MemBudgetReport, MemBudgetStatus, Span,
};
#[cfg(feature = "alloc-track")]
pub use mem::{TagLeakCheckpoint, TagLeakReport, TagLeakResidual};
pub use metrics::{
    hud_lines, Counter, FrameStatsSnapshot, FrameTimer, Gauge, Histogram, HistogramSnapshot, Hud,
    HudSnapshot, LoadProfile, MetricRegistry, RegistrySnapshot, Sum, SystemLoad, ThreadLoad,
};
pub use model::{Event, Field, FieldValue, Level};
pub use profiler::{
    clear_profiler, emit_span, frame_mark, gpu_zone, is_active as profiler_active, message, plot,
    set_profiler, zone as profile_zone, FrameMark, GpuZone, NoopProfiler, PlotValue, ProfiledZone,
    Profiler, Zone,
};
#[cfg(feature = "remote")]
pub use remote::{
    CommandOutcome, FrameSummary, RemoteClient, RemoteCommand, RemoteEvent, RemoteServer,
    RemoteServerHandle, RuntimeControls,
};
pub use replay::{compare_timelines, fnv1a_64, ReplayDivergence, ReplayMarker, ReplayTimeline};
pub use sampling::{
    call_tree, collapsed_stacks, facet_by_lane, facet_by_thread, flat_profile, fuse, CallNode,
    CallTree, CollapsedStack, FlatProfile, FoldDirection, FrameId, FrameStat, FusedEntry,
    FusedProfile, FusionSource, InstrumentedSpan, LaneKind, LaneProfile, SamplingProfiler,
    StackSample, SymbolTable, ThreadProfile,
};
pub use sink::{clear_sink, set_sink, CaptureSink, ConsoleSink, FileSink, Sink};
pub use span::Scope;
pub use telemetry::{
    build_event as build_telemetry_event, hash_identifier, redact_user_path, truncate_str,
    AggregatedEvent, EventAggregator, EventSchema, FieldDisposition, FieldTier, RedactedEvent,
    RedactionPolicy, SampleOutcome, SampleRatio, TelemetrySampler, DEFAULT_MAX_STRING_CHARS,
    REDACTED_SEGMENT, TRUNCATION_MARKER,
};
#[cfg(feature = "gpu")]
pub use trace::export_chrome_with_gpu;
pub use trace::{
    export_chrome_string, export_chrome_to_file, FlowPhase, FlowRecord, RingBuffer, SpanRecord,
    ThreadTrace,
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
#[cfg(test)]
mod tests_aggregate;
#[cfg(test)]
mod tests_budget;
#[cfg(test)]
mod tests_determinism;
#[cfg(all(test, feature = "gpu"))]
mod tests_gpu_timeline;
#[cfg(all(test, feature = "alloc-track"))]
mod tests_m6_alloc;
#[cfg(all(test, feature = "crash"))]
mod tests_m6_crash;
#[cfg(test)]
mod tests_m6_hitch;
#[cfg(test)]
mod tests_m6_replay;
#[cfg(test)]
mod tests_mem;
#[cfg(test)]
mod tests_sampling;
#[cfg(test)]
mod tests_telemetry;
