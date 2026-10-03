//! Crash capture (design doc §16 崩溃捕获, §22 M6, §23 risk #2).
//!
//! This is the engine's single entry point for turning a hard process fault
//! into a structured, retrievable [`CrashContext`] that `prism_diagnostic` can
//! later symbolicate and report. The crate only *captures* the fault; it owns
//! no UI or upload policy.
//!
//! ## Backends
//! - **Real `POSIX` host** (desktop Linux + macOS): [`install`] arms
//!   async-signal-safe `sigaction` handlers for `SIGSEGV`/`SIGABRT`/`SIGBUS`/
//!   `SIGILL`/`SIGFPE` on an alternate signal stack. On a fault the handler
//!   fills a pre-allocated [`CrashContext`] (signal, faulting address,
//!   frame-pointer backtrace, pre-registered build metadata), hands it to the
//!   registered handler, then restores the default disposition and re-raises so
//!   the process still dies with the correct signal. No allocation and no
//!   locking happen inside the handler (design doc §23 risk #2).
//! - **`mock`** (feature `mock`, every target): [`mock`] drives the exact same
//!   capture → handler → retrieve pipeline from a synthesized fault, so the
//!   crash plumbing is testable without crashing the process.
//! - **Unsupported hosts** (Web / Android / iOS and other non-desktop targets):
//!   [`install`]/[`uninstall`] honestly return [`CrashError::Unsupported`] and
//!   the [`crate::PlatformCaps`] crash bit is cleared — graceful degradation,
//!   not a panic.
//!
//! ## Capability
//! [`SUPPORTED`] (mirrored by [`crate::PlatformCaps::has_crash_capture`]) is
//! `true` only where a real signal backend is compiled in.
//!
//! ## Contract
//! [`install`] is expected to be called once during process startup, before any
//! fault and before other threads can fault; the pre-registered metadata and
//! handler are published before the kernel can deliver a captured signal.
#![expect(
    unsafe_code,
    reason = "the lock-free store writes the pre-allocated CrashContext slot through an UnsafeCell and transmutes the stored handler function pointer back from an AtomicUsize, both inside the async-signal-safe dispatch path"
)]

use core::cell::UnsafeCell;
use core::fmt;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

pub mod context;

#[cfg(feature = "mock")]
pub mod mock;

pub use context::{Backtrace, BuildMetadata, CrashContext, Signal, MAX_FRAMES};

#[cfg(all(unix, not(any(target_os = "android", target_os = "ios"))))]
#[path = "posix.rs"]
mod backend;

#[cfg(not(all(unix, not(any(target_os = "android", target_os = "ios")))))]
#[path = "unsupported.rs"]
mod backend;

/// Whether a real crash-capture backend is compiled in for this target.
///
/// `true` on desktop `POSIX` hosts (Linux and macOS), `false` on Web, Android,
/// iOS, and any other target without the real signal backend. Mirrored by
/// [`crate::PlatformCaps::has_crash_capture`].
pub const SUPPORTED: bool = cfg!(all(unix, not(any(target_os = "android", target_os = "ios"))));

/// The signature of a user crash handler.
///
/// It receives the captured [`CrashContext`] by reference. On the real host it
/// runs inside the signal handler, so it **must itself be async-signal-safe**:
/// no allocation, no locking, no non-reentrant library calls (design doc §23
/// risk #2).
pub type CrashHandlerFn = fn(&CrashContext);

/// Error type for crash-capture operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrashError {
    /// This target has no real crash-capture backend compiled in; the caller
    /// should degrade gracefully (design doc §23 risk #6).
    Unsupported,
    /// Installing or removing the `OS` signal handlers failed; carries the
    /// `errno` reported by the failing `sigaction` call.
    Os(i32),
}

impl fmt::Display for CrashError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported => f.write_str("crash capture is unsupported on this platform"),
            Self::Os(code) => write!(f, "crash handler install/uninstall failed (errno {code})"),
        }
    }
}

impl core::error::Error for CrashError {}

/// Result alias for crash-capture operations.
pub type Result<T> = core::result::Result<T, CrashError>;

// ---------------------------------------------------------------------------
// Lock-free, pre-allocated store shared by every backend.
//
// All fields live in statics so the async-signal-safe dispatch path never
// allocates. Writers of the `UnsafeCell` slots run only outside signal context
// (at `install` time, or on the single test thread for `mock`) and publish via
// the atomics; the handler only reads them.
// ---------------------------------------------------------------------------

/// Registered handler function pointer stored as a `usize` (`0` means none).
static HANDLER: AtomicUsize = AtomicUsize::new(0);

/// Whether a [`CrashContext`] has been captured and is readable.
static HAS_CONTEXT: AtomicBool = AtomicBool::new(false);

/// Pre-allocated slot for the most recently captured context.
struct ContextSlot(UnsafeCell<CrashContext>);
// SAFETY: the slot is only written outside signal context or from the single
// async-signal-safe dispatch call, and reads are published/observed through
// `HAS_CONTEXT` with release/acquire ordering; concurrent access is excluded by
// the one-time `install` contract.
unsafe impl Sync for ContextSlot {}
static CONTEXT: ContextSlot = ContextSlot(UnsafeCell::new(CrashContext::placeholder()));

/// Pre-registered build/module metadata, published before handlers are armed.
struct MetaSlot(UnsafeCell<BuildMetadata>);
// SAFETY: written only outside signal context (during `install`/`mock` setup)
// before any captured signal can be delivered, then only read at fault time.
unsafe impl Sync for MetaSlot {}
static METADATA: MetaSlot = MetaSlot(UnsafeCell::new(BuildMetadata::empty()));

/// Store the registered handler (`None` clears it). Not called from signal
/// context.
pub(crate) fn set_handler(handler: Option<CrashHandlerFn>) {
    let value = match handler {
        Some(f) => f as usize,
        None => 0,
    };
    HANDLER.store(value, Ordering::Release);
}

/// Store the pre-registered metadata. Not called from signal context; must run
/// before handlers are armed so the fault-time read sees it.
pub(crate) fn set_metadata(metadata: BuildMetadata) {
    // SAFETY: executed outside signal context before any captured signal can be
    // delivered (the one-time `install` contract), so there is no concurrent
    // reader of the slot.
    unsafe {
        *METADATA.0.get() = metadata;
    }
}

/// Read the pre-registered metadata. Async-signal-safe (a plain `Copy` read of
/// a slot published before any fault).
#[must_use]
pub(crate) fn metadata() -> BuildMetadata {
    // SAFETY: the slot was initialized before handlers were armed and is only
    // read here; `BuildMetadata` is `Copy` and holds only `'static` references.
    unsafe { *METADATA.0.get() }
}

/// Async-signal-safe: record the captured context and invoke the registered
/// handler. Performs no allocation and takes no lock.
pub(crate) fn dispatch(ctx: CrashContext) {
    // SAFETY: the only writer of the slot in signal context; readers observe it
    // through the `HAS_CONTEXT` release/acquire edge below. The one-time
    // `install` contract excludes a concurrent writer.
    unsafe {
        *CONTEXT.0.get() = ctx;
    }
    HAS_CONTEXT.store(true, Ordering::Release);

    let raw = HANDLER.load(Ordering::Acquire);
    if raw != 0 {
        // SAFETY: `raw` is non-zero, so it is a `CrashHandlerFn` previously
        // stored by `set_handler` via `f as usize`; transmuting it back yields
        // the same function pointer.
        let handler: CrashHandlerFn = unsafe { core::mem::transmute::<usize, CrashHandlerFn>(raw) };
        handler(&ctx);
    }
}

/// The most recently captured [`CrashContext`], if any.
///
/// Safe to call after a (test-triggered or `mock`) capture to inspect what the
/// handler saw.
#[must_use]
pub fn last_context() -> Option<CrashContext> {
    if HAS_CONTEXT.load(Ordering::Acquire) {
        // SAFETY: `HAS_CONTEXT` was set with release ordering after the slot was
        // fully written, establishing a happens-before edge; `CrashContext` is
        // `Copy` so we take an owned snapshot.
        Some(unsafe { *CONTEXT.0.get() })
    } else {
        None
    }
}

/// Take and clear the most recently captured [`CrashContext`].
pub fn take_last_context() -> Option<CrashContext> {
    let ctx = last_context();
    if ctx.is_some() {
        HAS_CONTEXT.store(false, Ordering::Release);
    }
    ctx
}

/// Clear any captured context without reading it.
pub fn clear_last_context() {
    HAS_CONTEXT.store(false, Ordering::Release);
}

/// Whether a real crash-capture backend is compiled in for this target.
#[must_use]
pub fn supported() -> bool {
    SUPPORTED
}

/// Whether the real signal handlers are currently installed.
#[must_use]
pub fn is_installed() -> bool {
    backend::is_installed()
}

/// Install the real crash-capture signal handlers with empty build metadata.
///
/// Register metadata with [`install_with`] instead when build/module info is
/// available. Idempotent: a second call while already installed is a no-op that
/// returns `Ok(())` and does not disturb the saved previous handlers.
///
/// # Errors
/// Returns [`CrashError::Unsupported`] on targets without a real backend, or
/// [`CrashError::Os`] if the underlying `sigaction` calls fail.
pub fn install(handler: CrashHandlerFn) -> Result<()> {
    install_with(handler, BuildMetadata::empty())
}

/// Install the real crash-capture signal handlers and pre-register `metadata`.
///
/// The handler and metadata are published before the `OS` handlers are armed,
/// so a fault delivered immediately afterward sees fully initialized state.
///
/// # Errors
/// Returns [`CrashError::Unsupported`] on targets without a real backend, or
/// [`CrashError::Os`] if the underlying `sigaction` calls fail. On failure the
/// handler registration is rolled back.
pub fn install_with(handler: CrashHandlerFn, metadata: BuildMetadata) -> Result<()> {
    set_metadata(metadata);
    set_handler(Some(handler));
    match backend::install() {
        Ok(()) => Ok(()),
        Err(err) => {
            set_handler(None);
            Err(err)
        }
    }
}

/// Uninstall the crash-capture signal handlers, restoring the previously
/// installed dispositions, and clear the registered handler.
///
/// Idempotent: calling it when nothing is installed returns `Ok(())`.
///
/// # Errors
/// Returns [`CrashError::Unsupported`] on targets without a real backend, or
/// [`CrashError::Os`] if restoring a disposition fails.
pub fn uninstall() -> Result<()> {
    let outcome = backend::uninstall();
    set_handler(None);
    outcome
}

/// Current `OS` disposition pointer for `sig` (test-only, real backend only).
#[cfg(all(test, unix, not(any(target_os = "android", target_os = "ios")), not(feature = "mock")))]
pub(crate) fn disposition(sig: i32) -> usize {
    backend::disposition(sig)
}

/// Address of the real signal trampoline (test-only, real backend only).
#[cfg(all(test, unix, not(any(target_os = "android", target_os = "ios")), not(feature = "mock")))]
pub(crate) fn trampoline_addr() -> usize {
    backend::trampoline_addr()
}
