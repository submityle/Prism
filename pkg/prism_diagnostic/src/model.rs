//! Core diagnostic data model: levels, fields, and events.

extern crate alloc;

use alloc::string::String;
use prism_utils::SmallVec;

/// Severity level, ordered from most verbose ([`Level::Trace`]) to most severe
/// ([`Level::Error`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Level {
    /// Extremely verbose tracing.
    Trace = 0,
    /// Debugging detail.
    Debug = 1,
    /// Informational milestones.
    Info = 2,
    /// Recoverable problems.
    Warn = 3,
    /// Errors needing attention.
    Error = 4,
}

impl Level {
    /// Short uppercase label.
    pub const fn label(self) -> &'static str {
        match self {
            Level::Trace => "TRACE",
            Level::Debug => "DEBUG",
            Level::Info => "INFO",
            Level::Warn => "WARN",
            Level::Error => "ERROR",
        }
    }
}

/// A typed structured value attached to an event.
#[derive(Clone, Debug, PartialEq)]
pub enum FieldValue {
    /// Text.
    Str(String),
    /// Signed integer.
    I64(i64),
    /// Unsigned integer.
    U64(u64),
    /// Floating point.
    F64(f64),
    /// Boolean.
    Bool(bool),
}

/// A key/value pair attached to an event.
#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    /// Static field key.
    pub key: &'static str,
    /// Field value.
    pub value: FieldValue,
}

/// A single diagnostic event.
#[derive(Clone, Debug)]
pub struct Event {
    /// Severity.
    pub level: Level,
    /// Source module path / subsystem.
    pub target: &'static str,
    /// Human-readable message.
    pub message: String,
    /// Structured fields (inline up to 4 before spilling).
    pub fields: SmallVec<Field, 4>,
    /// Monotonic timestamp in nanoseconds.
    pub timestamp_nanos: u64,
}

impl Event {
    /// Start building an event at `level` for `target` with `message`.
    pub fn new(level: Level, target: &'static str, message: impl Into<String>) -> Self {
        Self {
            level,
            target,
            message: message.into(),
            fields: SmallVec::new(),
            timestamp_nanos: prism_platform::now().0,
        }
    }

    /// Attach a structured field (builder style).
    pub fn with_field(mut self, key: &'static str, value: FieldValue) -> Self {
        self.fields.push(Field { key, value });
        self
    }
}
