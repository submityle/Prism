//! Common imports: `use prism_diagnostic::prelude::*;`.

pub use crate::filter::{max_level, set_max_level};
pub use crate::instrument::{
    instrument, next_flow_id, FlowId, Instrumentable, JobFlow, JobScope, SystemScope,
};
pub use crate::metrics::{
    hud_lines, Counter, FrameStatsSnapshot, FrameTimer, Gauge, Histogram, HistogramSnapshot, Hud,
    HudSnapshot, LoadProfile, MetricRegistry, RegistrySnapshot, Sum, SystemLoad, ThreadLoad,
};
pub use crate::model::{Event, Field, FieldValue, Level};
pub use crate::sink::{set_sink, CaptureSink, ConsoleSink, FileSink, Sink};
pub use crate::span::Scope;
pub use crate::trace::{
    export_chrome_string, export_chrome_to_file, FlowPhase, FlowRecord, SpanRecord,
};
pub use crate::{debug, error, event, info, instrument_system, span, trace, warn};
