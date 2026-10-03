//! Crash reporting (`crash` feature, design §12, M6).
//!
//! This module owns the crash report *format* and *writer*: a self-contained,
//! serializable [`CrashContext`] model (crashing thread backtrace + register
//! snapshot, module list, reason/signal, timestamp, build id) and a
//! deterministic binary serializer ([`CrashReport`]) built on the crate's
//! hand-rolled [`wire`](crate::wire) codec.
//!
//! It deliberately does **not** install `OS` signal/exception handlers. Capturing
//! a live fault (`SIGSEGV`/`SIGABRT`/`SEH`), walking the stack, and reading
//! registers is async-signal-safety-critical, platform-specific work that
//! belongs in `prism_platform` (design §12 / §23 risk 3); that layer will
//! populate a [`CrashContext`] and hand it here to serialize. Keeping the format
//! in this crate makes it fully self-contained and testable with a synthetic
//! context — no platform hooks required.
//!
//! ## Layout
//! - [`context`] — the [`CrashContext`] data model and its parts.
//! - [`report`] — the versioned [`CrashReport`] container + binary writer/reader.

pub mod context;
pub mod report;

pub use context::{
    CrashContext, CrashReason, ModuleEntry, RegisterSnapshot, StackFrame, ThreadContext,
};
pub use report::{CrashReport, CRASH_REPORT_MAGIC, CRASH_REPORT_VERSION};
