//! In-process `mock` crash backend (feature `mock`).
//!
//! A legitimate test aid — not a stand-in for the real backend — that lets the
//! whole crash pipeline be exercised without actually crashing the process
//! (design doc §19 "mock 后端做单测"). It registers a handler in the shared
//! store and synthesizes a [`CrashContext`] through the *exact same*
//! [`super::dispatch`] and [`super::context::build`] code the real `POSIX`
//! handler runs, so the capture → handler → retrieve round-trip is genuinely
//! covered on every platform, including Web where there is no real backend.
//!
//! Unlike [`super::install`], this never arms `OS` signal handlers, so it is
//! available and total on all targets.

use super::{context, BuildMetadata, CrashContext, CrashHandlerFn, Signal};

/// Register a crash handler for the mock pipeline (no `OS` handlers armed).
pub fn install(handler: CrashHandlerFn) {
    install_with(handler, BuildMetadata::empty());
}

/// Register a crash handler and pre-register `metadata` for the mock pipeline.
pub fn install_with(handler: CrashHandlerFn, metadata: BuildMetadata) {
    super::set_metadata(metadata);
    super::set_handler(Some(handler));
}

/// Clear the mock handler registration.
pub fn uninstall() {
    super::set_handler(None);
}

/// Synthesize a fault and drive the full capture → handler → retrieve pipeline
/// in-process.
///
/// Builds a [`CrashContext`] (including a real frame-pointer backtrace of the
/// current stack) for `signal` at `fault_address`, stores it, and invokes the
/// registered handler — returning the same context for convenience. The stored
/// context is afterwards retrievable via [`super::last_context`].
pub fn simulate(signal: Signal, fault_address: usize) -> CrashContext {
    // The mock backend is not inside a real signal handler, so there is no
    // interrupted `ucontext`: pass `(0, 0)` and let the backtrace fall back to
    // the calling thread's frame walk.
    let ctx = context::build(signal, signal.raw(), fault_address, super::metadata(), 0, 0);
    super::dispatch(ctx);
    ctx
}
