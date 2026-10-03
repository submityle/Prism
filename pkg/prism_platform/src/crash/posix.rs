//! Real async-signal-safe `POSIX` crash backend (desktop Linux + macOS).
//!
//! Arms `sigaction` handlers with `SA_SIGINFO` on an alternate signal stack
//! (`sigaltstack`, so a stack-overflow `SIGSEGV` can still be handled) for
//! `SIGSEGV`/`SIGABRT`/`SIGBUS`/`SIGILL`/`SIGFPE`. On a fault the handler reads
//! the faulting address from `siginfo_t::si_addr`, builds a [`CrashContext`]
//! into the pre-allocated store (no allocation, no locking — design doc §23
//! risk #2), hands it to the registered handler, then restores the default
//! disposition and re-raises so the process still terminates with the correct
//! signal and a usable core dump.
//!
//! Every `OS` call is a dependency-free `extern` declaration resolved against
//! the C runtime `std` already links — there is no `libc` crate — mirroring
//! [`crate::vm`] and [`crate::thread::affinity`]. The previous dispositions are
//! saved on install and restored on uninstall.
#![expect(
    unsafe_code,
    reason = "crash capture requires sigaction/sigaltstack/signal/raise plus a raw siginfo_t::si_addr read via dependency-free C FFI, all exercised inside the async-signal-safe signal handler"
)]

use core::cell::UnsafeCell;
use core::ffi::c_void;
use core::sync::atomic::{AtomicBool, Ordering};

use super::context::{self, Signal, SIGABRT, SIGBUS, SIGFPE, SIGILL, SIGSEGV};
use super::CrashError;

/// The fatal signals this backend captures, in a fixed order.
const SIGNALS: [i32; 5] = [SIGSEGV, SIGABRT, SIGBUS, SIGILL, SIGFPE];

/// Default signal disposition (`SIG_DFL`).
const SIG_DFL: usize = 0;

// --- Platform ABI: struct layouts, flag values, and errno access. ----------

#[cfg(target_vendor = "apple")]
mod abi {
    use core::ffi::c_void;

    /// macOS `struct sigaction`: `{ void* __sigaction_u; sigset_t(u32) sa_mask;
    /// int sa_flags; }`.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub(super) struct SigAction {
        pub(super) sa_sigaction: usize,
        pub(super) sa_mask: u32,
        pub(super) sa_flags: i32,
    }

    impl SigAction {
        pub(super) const fn zeroed() -> Self {
            Self {
                sa_sigaction: 0,
                sa_mask: 0,
                sa_flags: 0,
            }
        }
    }

    /// macOS `stack_t`: `{ void* ss_sp; size_t ss_size; int ss_flags; }`.
    #[repr(C)]
    pub(super) struct StackT {
        pub(super) ss_sp: *mut c_void,
        pub(super) ss_size: usize,
        pub(super) ss_flags: i32,
    }

    pub(super) const SA_SIGINFO: i32 = 0x0040;
    pub(super) const SA_ONSTACK: i32 = 0x0001;
    pub(super) const SA_RESTART: i32 = 0x0002;

    /// Offset of `void* si_addr` within macOS `siginfo_t`.
    pub(super) const SI_ADDR_OFFSET: usize = 24;

    pub(super) fn errno() -> i32 {
        unsafe extern "C" {
            fn __error() -> *mut i32;
        }
        // SAFETY: `__error` returns a valid pointer to this thread's `errno`,
        // which we only read.
        unsafe { *__error() }
    }
}

#[cfg(not(target_vendor = "apple"))]
mod abi {
    use core::ffi::c_void;

    /// Linux/glibc `struct sigaction`: `{ handler; sigset_t(u64[16]) sa_mask;
    /// int sa_flags; void* sa_restorer; }`.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub(super) struct SigAction {
        pub(super) sa_sigaction: usize,
        pub(super) sa_mask: [u64; 16],
        pub(super) sa_flags: i32,
        pub(super) sa_restorer: usize,
    }

    impl SigAction {
        pub(super) const fn zeroed() -> Self {
            Self {
                sa_sigaction: 0,
                sa_mask: [0; 16],
                sa_flags: 0,
                sa_restorer: 0,
            }
        }
    }

    /// Linux `stack_t`: `{ void* ss_sp; int ss_flags; size_t ss_size; }`.
    #[repr(C)]
    pub(super) struct StackT {
        pub(super) ss_sp: *mut c_void,
        pub(super) ss_flags: i32,
        pub(super) ss_size: usize,
    }

    pub(super) const SA_SIGINFO: i32 = 0x0000_0004;
    pub(super) const SA_ONSTACK: i32 = 0x0800_0000;
    pub(super) const SA_RESTART: i32 = 0x1000_0000;

    /// Offset of `void* si_addr` within 64-bit Linux `siginfo_t` (three leading
    /// `int`s plus 4 bytes of alignment padding before the fault union).
    pub(super) const SI_ADDR_OFFSET: usize = 16;

    pub(super) fn errno() -> i32 {
        unsafe extern "C" {
            fn __errno_location() -> *mut i32;
        }
        // SAFETY: `__errno_location` returns a valid pointer to this thread's
        // `errno`, which we only read.
        unsafe { *__errno_location() }
    }
}

use abi::{SigAction, StackT, SA_ONSTACK, SA_RESTART, SA_SIGINFO, SI_ADDR_OFFSET};

unsafe extern "C" {
    fn sigaction(signum: i32, act: *const SigAction, oldact: *mut SigAction) -> i32;
    fn sigaltstack(ss: *const StackT, old_ss: *mut StackT) -> i32;
    fn signal(signum: i32, handler: usize) -> usize;
    fn raise(signum: i32) -> i32;
}

// --- Pre-allocated static state. -------------------------------------------

/// Size of the alternate signal stack (64 KiB, comfortably above every
/// platform's `MINSIGSTKSZ`).
const ALT_STACK_SIZE: usize = 64 * 1024;

/// Pre-allocated alternate signal stack buffer.
struct AltStack(UnsafeCell<[u8; ALT_STACK_SIZE]>);
// SAFETY: the buffer is handed to the kernel once via `sigaltstack` under the
// install lock and thereafter only the kernel writes it while delivering a
// signal; Rust code never touches its bytes concurrently.
unsafe impl Sync for AltStack {}
static ALT_STACK: AltStack = AltStack(UnsafeCell::new([0; ALT_STACK_SIZE]));

/// Saved previous dispositions, one per entry in [`SIGNALS`], restored on
/// uninstall.
struct SavedActions(UnsafeCell<[SigAction; SIGNALS.len()]>);
// SAFETY: only read/written under `LOCK` during install/uninstall, never from
// signal context.
unsafe impl Sync for SavedActions {}
static SAVED: SavedActions = SavedActions(UnsafeCell::new([SigAction::zeroed(); SIGNALS.len()]));

/// Spin lock serializing install/uninstall (both are startup/shutdown-time,
/// not fault-time, operations).
static LOCK: AtomicBool = AtomicBool::new(false);

/// Whether handlers are currently installed.
static INSTALLED: AtomicBool = AtomicBool::new(false);

/// Whether the alternate signal stack has been registered (once per process).
static ALT_STACK_READY: AtomicBool = AtomicBool::new(false);

struct LockGuard;

impl LockGuard {
    fn acquire() -> Self {
        while LOCK
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            core::hint::spin_loop();
        }
        Self
    }
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        LOCK.store(false, Ordering::Release);
    }
}

// --- The signal handler. ----------------------------------------------------

/// Async-signal-safe `SA_SIGINFO` trampoline.
///
/// Reads the faulting address, builds the pre-allocated [`CrashContext`], hands
/// it to the registered handler, then restores the default disposition and
/// re-raises so the process dies with the correct signal.
extern "C" fn trampoline(sig: i32, info: *mut c_void, _ucontext: *mut c_void) {
    let fault_address = if info.is_null() {
        0
    } else {
        // SAFETY: for a kernel-delivered `SA_SIGINFO` signal, `info` points at a
        // valid `siginfo_t`; `si_addr` is a pointer-sized field at
        // `SI_ADDR_OFFSET` within it. We read exactly that one word.
        unsafe { *info.cast::<u8>().add(SI_ADDR_OFFSET).cast::<usize>() }
    };

    let ctx = context::build(
        Signal::from_raw(sig),
        sig,
        fault_address,
        super::metadata(),
    );
    super::dispatch(ctx);

    // Restore the default disposition and re-raise. `signal` and `raise` are
    // both on the `POSIX` async-signal-safe list; this guarantees the process
    // still terminates with the faulting signal after we have captured it.
    // SAFETY: both calls are async-signal-safe and operate only on the current
    // signal/thread.
    unsafe {
        signal(sig, SIG_DFL);
        raise(sig);
    }
}

// --- Install / uninstall. ---------------------------------------------------

/// Register the alternate signal stack once.
fn ensure_alt_stack() -> Result<(), CrashError> {
    if ALT_STACK_READY.load(Ordering::Acquire) {
        return Ok(());
    }
    let ss = StackT {
        ss_sp: ALT_STACK.0.get().cast::<c_void>(),
        ss_size: ALT_STACK_SIZE,
        ss_flags: 0,
    };
    // SAFETY: `ss` describes a valid, process-lifetime static buffer; passing a
    // null `old_ss` is allowed and simply discards the previous stack.
    let rc = unsafe { sigaltstack(&ss, core::ptr::null_mut()) };
    if rc != 0 {
        return Err(CrashError::Os(abi::errno()));
    }
    ALT_STACK_READY.store(true, Ordering::Release);
    Ok(())
}

/// Install the crash handlers. Idempotent and thread-safe.
pub(super) fn install() -> Result<(), CrashError> {
    let _guard = LockGuard::acquire();
    if INSTALLED.load(Ordering::Acquire) {
        return Ok(());
    }
    ensure_alt_stack()?;

    let mut act = SigAction::zeroed();
    act.sa_sigaction = trampoline as *const () as usize;
    act.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_RESTART;

    for (slot, &sig) in SIGNALS.iter().enumerate() {
        let mut old = SigAction::zeroed();
        // SAFETY: `act` and `old` are valid, correctly sized `sigaction`
        // structures for this platform's ABI; the call only reads `act` and
        // writes `old`.
        let rc = unsafe { sigaction(sig, &act, &mut old) };
        if rc != 0 {
            let err = CrashError::Os(abi::errno());
            // Roll back the handlers already installed in this loop.
            for &done in &SIGNALS[..slot] {
                // SAFETY: restoring with a null `oldact` using our own
                // just-installed `act` is a well-formed `sigaction` call; a best
                // effort during error rollback.
                unsafe {
                    let _ = sigaction(done, &act, core::ptr::null_mut());
                }
            }
            return Err(err);
        }
        // SAFETY: holding `LOCK`, so we are the sole accessor of `SAVED`; `slot`
        // is in bounds by construction.
        unsafe {
            (*SAVED.0.get())[slot] = old;
        }
    }

    INSTALLED.store(true, Ordering::Release);
    Ok(())
}

/// Uninstall the crash handlers, restoring saved dispositions. Idempotent.
pub(super) fn uninstall() -> Result<(), CrashError> {
    let _guard = LockGuard::acquire();
    if !INSTALLED.load(Ordering::Acquire) {
        return Ok(());
    }

    let mut result = Ok(());
    for (slot, &sig) in SIGNALS.iter().enumerate() {
        // SAFETY: holding `LOCK`, so we are the sole accessor of `SAVED`.
        let saved = unsafe { (*SAVED.0.get())[slot] };
        // SAFETY: `saved` is a valid `sigaction` structure captured on install;
        // restoring it with a null `oldact` is well-formed.
        let rc = unsafe { sigaction(sig, &saved, core::ptr::null_mut()) };
        if rc != 0 && result.is_ok() {
            result = Err(CrashError::Os(abi::errno()));
        }
    }

    INSTALLED.store(false, Ordering::Release);
    result
}

/// Whether handlers are currently installed.
pub(super) fn is_installed() -> bool {
    INSTALLED.load(Ordering::Acquire)
}

/// Read the current `sa_sigaction` pointer for `sig` (test-only inspection of
/// real `OS` disposition state).
#[cfg(all(test, not(feature = "mock")))]
pub(super) fn disposition(sig: i32) -> usize {
    let mut old = SigAction::zeroed();
    // SAFETY: a null `act` makes `sigaction` only read the current disposition
    // into `old` without changing it.
    let rc = unsafe { sigaction(sig, core::ptr::null(), &mut old) };
    assert_eq!(rc, 0, "sigaction query failed");
    old.sa_sigaction
}

/// The address of the real trampoline (test-only), to assert it is installed.
#[cfg(all(test, not(feature = "mock")))]
pub(super) fn trampoline_addr() -> usize {
    trampoline as *const () as usize
}
