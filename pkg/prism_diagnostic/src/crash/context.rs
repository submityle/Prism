//! The crash-context data model (`crash` feature, design §12, M6).
//!
//! These structs describe *what* a crash report captures: the crashing thread's
//! backtrace frames and register snapshot, the loaded module list, the crash
//! reason/signal, a timestamp, and a build id, plus free-form redacted notes
//! (engine version, scene, `GPU`/driver). This crate owns the *format*; the
//! actual `OS` signal/exception capture that fills these fields lives in
//! `prism_platform` and is a documented follow-up. Everything here is plain
//! owned data so it can be built synthetically, serialized, and round-tripped
//! without any platform hooks.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

/// Why the process crashed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CrashReason {
    /// A `POSIX` signal (e.g. `SIGSEGV`), carrying the signal number.
    Signal(i32),
    /// A Windows structured exception (`SEH`), carrying the exception code.
    Exception(u32),
    /// `abort()` / `std::process::abort`.
    Abort,
    /// A failed engine assertion/contract check.
    Assertion,
    /// Allocation failure / out of memory.
    OutOfMemory,
    /// An otherwise-uncategorized fault, described by text.
    Other(String),
}

/// A snapshot of a thread's `CPU` registers at crash time.
///
/// The three named pointers are broken out because they drive unwinding; any
/// remaining general-purpose registers travel as `(name, value)` pairs so the
/// model stays architecture-neutral.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RegisterSnapshot {
    /// Instruction pointer (`PC`/`RIP`).
    pub instruction_pointer: u64,
    /// Stack pointer (`SP`/`RSP`).
    pub stack_pointer: u64,
    /// Frame/base pointer (`FP`/`RBP`).
    pub frame_pointer: u64,
    /// Remaining architecture-specific registers as `(name, value)`.
    pub general: Vec<(String, u64)>,
}

/// One unwound stack frame.
///
/// Symbol/module names are optional: a release capture typically records only
/// addresses and resolves symbols offline against a symbol file.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StackFrame {
    /// Zero-based depth from the innermost (crashing) frame.
    pub index: u32,
    /// Absolute instruction pointer for this frame.
    pub instruction_pointer: u64,
    /// Offset of the instruction pointer from the owning module base.
    pub module_offset: u64,
    /// Resolved symbol name, if symbolized in-process.
    pub symbol: Option<String>,
    /// Owning module name, if known.
    pub module: Option<String>,
}

/// A loaded module (executable or shared library) in the crashed process.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModuleEntry {
    /// Module file name or path (callers redact absolute paths per policy).
    pub name: String,
    /// Load base address.
    pub base_address: u64,
    /// Mapped size in bytes.
    pub size: u64,
    /// Build id / debug id used for offline symbolization, if known.
    pub build_id: Option<String>,
}

/// Per-thread capture: identity, registers, and backtrace.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ThreadContext {
    /// Platform thread id.
    pub thread_id: u64,
    /// Thread name, if named.
    pub thread_name: Option<String>,
    /// Register snapshot at capture time.
    pub registers: RegisterSnapshot,
    /// Unwound backtrace, innermost frame first.
    pub frames: Vec<StackFrame>,
}

/// The full crash context: everything a report captures about one crash.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CrashContext {
    /// Why the process crashed.
    pub reason: Option<CrashReason>,
    /// The crashing thread (registers + backtrace).
    pub crashing_thread: ThreadContext,
    /// Other captured threads, if any.
    pub other_threads: Vec<ThreadContext>,
    /// Loaded module list for offline symbolization.
    pub modules: Vec<ModuleEntry>,
    /// Monotonic timestamp (nanoseconds) sampled at capture.
    pub timestamp_nanos: u64,
    /// Build id of the crashing binary.
    pub build_id: String,
    /// Redacted `(key, value)` notes (engine version, scene, `GPU`/driver, ...).
    pub notes: Vec<(String, String)>,
}

impl CrashContext {
    /// Create an empty context stamped with the given build id and the current
    /// monotonic timestamp.
    pub fn new(build_id: impl Into<String>) -> Self {
        Self {
            build_id: build_id.into(),
            timestamp_nanos: prism_platform::now().0,
            ..Self::default()
        }
    }

    /// Set the crash reason (builder style).
    pub fn with_reason(mut self, reason: CrashReason) -> Self {
        self.reason = Some(reason);
        self
    }

    /// Set the crashing thread (builder style).
    pub fn with_crashing_thread(mut self, thread: ThreadContext) -> Self {
        self.crashing_thread = thread;
        self
    }

    /// Add a loaded module (builder style).
    pub fn with_module(mut self, module: ModuleEntry) -> Self {
        self.modules.push(module);
        self
    }

    /// Add a redacted note (builder style).
    pub fn with_note(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.notes.push((key.into(), value.into()));
        self
    }
}
