//! Bounded data models and real capture producers for the Prism editor profiler.
//!
//! This crate is intentionally transport- and backend-neutral. A producer may
//! adapt Tracy spans, render-frame graphs, or diagnostic counters into these
//! types. [`capture`] records real CPU time; optional `gpu` records and reads
//! back hardware timestamps using an application-owned wgpu device/queue.
//! Every model collection is bounded by [`ProfilerLimits`]
//! and every externally visible ordering has an explicit stable tie-breaker.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod capture;
pub mod frame;
pub mod frame_graph;
#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(feature = "gpu")]
pub mod gpu_preview;
pub mod graph_capture;
pub mod memory;
pub mod visual_log;

pub use frame::{Frame, FrameId, FrameStore, Span, SpanId, SpanNode, SpanTree};
pub use frame_graph::{CriticalPath, FrameGraph, GraphNode, GraphNodeId};
#[cfg(feature = "render-graph")]
pub use graph_capture::architecture::{
    capture_feature_graph, capture_runtime_graph, GraphCaptureInputs,
};
pub use graph_capture::{
    CapturedDependencyKind, CapturedGraphAccess, CapturedGraphBudget, CapturedGraphDependency,
    CapturedGraphPass, CapturedGraphResource, CapturedResourceLayout, GraphCaptureLimits,
    GraphCaptureReport, GraphCaptureSource, GraphPassTiming, GraphTimingSource,
};
pub use memory::{MemoryAlert, MemoryBudget, MemoryProfile, MemorySample, MemorySeverity};
pub use visual_log::{
    TrackProjection, VisualEvent, VisualEventId, VisualLog, VisualPrimitive, VisualTrack,
};

/// Limits applied to all profiler data entering the editor model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProfilerLimits {
    /// Maximum number of frames retained by one [`FrameStore`].
    pub max_frames: usize,
    /// Maximum number of spans in one frame.
    pub max_spans_per_frame: usize,
    /// Maximum nesting depth of a span tree, including the root span.
    pub max_span_depth: usize,
    /// Maximum number of nodes in one [`FrameGraph`].
    pub max_graph_nodes: usize,
    /// Maximum number of directed dependencies in one [`FrameGraph`].
    pub max_graph_dependencies: usize,
    /// Maximum number of memory categories in one [`MemoryProfile`].
    pub max_memory_budgets: usize,
    /// Maximum number of memory samples in one [`MemoryProfile`].
    pub max_memory_samples: usize,
    /// Maximum number of visual events in one [`VisualLog`].
    pub max_visual_events: usize,
    /// Maximum number of visual tracks in one [`VisualLog`].
    pub max_visual_tracks: usize,
    /// Maximum number of projected visual events returned by one query.
    pub max_projection_events: usize,
    /// Maximum UTF-8 byte length of a name, category, track, or label.
    pub max_text_bytes: usize,
}

impl Default for ProfilerLimits {
    fn default() -> Self {
        Self {
            max_frames: 4_096,
            max_spans_per_frame: 65_536,
            max_span_depth: 256,
            max_graph_nodes: 4_096,
            max_graph_dependencies: 16_384,
            max_memory_budgets: 256,
            max_memory_samples: 65_536,
            max_visual_events: 65_536,
            max_visual_tracks: 1_024,
            max_projection_events: 16_384,
            max_text_bytes: 4_096,
        }
    }
}

/// Errors raised when a normalized profiler input violates a data contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProfilerError {
    /// A bounded collection would exceed its configured limit.
    LimitExceeded {
        /// Stable name of the bounded resource.
        resource: &'static str,
        /// Configured maximum.
        limit: usize,
        /// Observed or requested size.
        actual: usize,
    },
    /// A required text field was empty.
    EmptyText {
        /// Stable name of the field.
        field: &'static str,
    },
    /// A text field exceeded the configured byte limit.
    TextTooLong {
        /// Stable name of the field.
        field: &'static str,
        /// Configured maximum.
        limit: usize,
        /// Observed UTF-8 byte length.
        actual: usize,
    },
    /// An identifier was already present.
    DuplicateId {
        /// Stable name of the identifier domain.
        resource: &'static str,
        /// Conflicting identifier.
        id: u64,
    },
    /// A referenced identifier was not present.
    MissingId {
        /// Stable name of the identifier domain.
        resource: &'static str,
        /// Missing identifier.
        id: u64,
    },
    /// A range or coordinate was invalid.
    InvalidRange {
        /// Stable description of the invalid field.
        field: &'static str,
    },
    /// A graph dependency would introduce a self-edge or duplicate edge.
    InvalidDependency,
    /// A graph contains a directed cycle.
    Cycle,
    /// An arithmetic operation would overflow.
    ArithmeticOverflow,
    /// A numeric value cannot be represented in the data model.
    NonFiniteValue {
        /// Stable name of the invalid value.
        field: &'static str,
    },
    /// A budget category was referenced before it was defined.
    UnknownCategory(String),
}

impl std::fmt::Display for ProfilerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LimitExceeded {
                resource,
                limit,
                actual,
            } => write!(formatter, "{resource} exceeds limit {limit} with {actual}"),
            Self::EmptyText { field } => write!(formatter, "{field} cannot be empty"),
            Self::TextTooLong {
                field,
                limit,
                actual,
            } => write!(formatter, "{field} has {actual} bytes; limit is {limit}"),
            Self::DuplicateId { resource, id } => {
                write!(formatter, "duplicate {resource} id {id}")
            }
            Self::MissingId { resource, id } => write!(formatter, "missing {resource} id {id}"),
            Self::InvalidRange { field } => write!(formatter, "invalid range for {field}"),
            Self::InvalidDependency => formatter.write_str("invalid graph dependency"),
            Self::Cycle => formatter.write_str("graph contains a cycle"),
            Self::ArithmeticOverflow => formatter.write_str("profiler arithmetic overflow"),
            Self::NonFiniteValue { field } => write!(formatter, "{field} must be finite"),
            Self::UnknownCategory(category) => {
                write!(formatter, "unknown memory category {category}")
            }
        }
    }
}

impl std::error::Error for ProfilerError {}

pub(crate) fn validate_text(
    value: &str,
    field: &'static str,
    limits: &ProfilerLimits,
) -> Result<(), ProfilerError> {
    if value.is_empty() {
        return Err(ProfilerError::EmptyText { field });
    }
    if value.len() > limits.max_text_bytes {
        return Err(ProfilerError::TextTooLong {
            field,
            limit: limits.max_text_bytes,
            actual: value.len(),
        });
    }
    Ok(())
}

pub(crate) fn validate_limit(
    actual: usize,
    limit: usize,
    resource: &'static str,
) -> Result<(), ProfilerError> {
    if actual > limit {
        Err(ProfilerError::LimitExceeded {
            resource,
            limit,
            actual,
        })
    } else {
        Ok(())
    }
}
