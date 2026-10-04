//! Common imports: `use prism_diagnostic::prelude::*;`.

pub use crate::filter::{max_level, set_max_level};
pub use crate::aggregate::{
    AnomalyConfig, AssembledTrace, ClusterAggregator, ClusterFrametimeReport, CriticalPath,
    DistributedSpan, InstanceFrameReport, InstanceSummary, SampleRate, SamplingController,
    SamplingDecision, SpanId, SpanKind, TraceAssembler, TraceId,
};
pub use crate::sampling::{
    call_tree, collapsed_stacks, facet_by_lane, facet_by_thread, flat_profile, fuse, CallTree,
    CollapsedStack, FlatProfile, FoldDirection, FrameId, FusedProfile, FusionSource,
    InstrumentedSpan, LaneKind, SamplingProfiler, StackSample, SymbolTable,
};
pub use crate::determinism::{
    compare, DeterminismTrace, FrameHash, FrameInput, InputRecorder, InputReplay, StateHasher,
    TraceDiff,
};
pub use crate::instrument::{
    instrument, next_flow_id, FlowId, Instrumentable, JobFlow, JobScope, SystemScope,
};
pub use crate::metrics::{
    hud_lines, Counter, FrameStatsSnapshot, FrameTimer, Gauge, Histogram, HistogramSnapshot, Hud,
    HudSnapshot, LoadProfile, MetricRegistry, RegistrySnapshot, Sum, SystemLoad, ThreadLoad,
};
pub use crate::model::{Event, Field, FieldValue, Level};
pub use crate::profiler::{
    frame_mark, gpu_zone, message, plot, set_profiler, FrameMark, GpuZone, NoopProfiler, PlotValue,
    ProfiledZone, Profiler, Zone,
};
pub use crate::sink::{set_sink, CaptureSink, ConsoleSink, FileSink, Sink};
pub use crate::span::Scope;
pub use crate::telemetry::{
    build_event, hash_identifier, redact_user_path, truncate_str, AggregatedEvent, EventAggregator,
    EventSchema, FieldDisposition, FieldTier, RedactedEvent, RedactionPolicy, SampleOutcome,
    SampleRatio, TelemetrySampler,
};
pub use crate::trace::{
    export_chrome_string, export_chrome_to_file, FlowPhase, FlowRecord, SpanRecord,
};
pub use crate::{debug, error, event, info, instrument_system, profiled_zone, span, trace, warn};

#[cfg(feature = "gpu")]
pub use crate::gpu::{
    next_correlation_id, CorrelationBreakdown, CorrelationId, GpuBubble, GpuClockCalibration,
    GpuQueueId, GpuReadbackRing, GpuSpan, GpuTick, ProjectedGpuSpan, TimelineEntry,
    TimelineTrack, UnifiedTimeline,
};

#[cfg(feature = "remote")]
pub use crate::remote::{
    CommandOutcome, RemoteClient, RemoteCommand, RemoteEvent, RemoteServer, RemoteServerHandle,
    RuntimeControls,
};
