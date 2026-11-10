//! Process termination signals and graceful-exit notification.
//!
//! The OS callback only records a small atomic bit; it never allocates, locks,
//! or calls into the application. Runners poll the bits on their normal frame
//! thread and turn the result into an ordinary [`AppExitRequest`](crate).
//! Hosts that already own the OS callback can use [`notify`] without calling
//! [`install`].

#![expect(
    unsafe_code,
    reason = "process signal handling requires dependency-free POSIX signal and Win32 console-control FFI"
)]

use core::fmt;
use core::num::NonZeroU8;
use core::sync::atomic::{AtomicBool, AtomicU8, Ordering};

const INTERRUPT: u8 = 1 << 0;
const TERMINATE: u8 = 1 << 1;
const HANGUP: u8 = 1 << 2;

static PENDING: AtomicU8 = AtomicU8::new(0);
static INSTALLED: AtomicBool = AtomicBool::new(false);

/// A process-level request to terminate gracefully.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProcessSignal {
    /// Interactive interrupt (`SIGINT` / Ctrl+C).
    Interrupt,
    /// Termination request (`SIGTERM` / console close or shutdown).
    Terminate,
    /// Hangup (`SIGHUP`), commonly used by a supervisor to request restart.
    Hangup,
}

impl ProcessSignal {
    /// Conventional Unix-style exit status for this request.
    #[must_use]
    pub const fn exit_code(self) -> NonZeroU8 {
        match self {
            Self::Interrupt => NonZeroU8::new(130).expect("130 is non-zero"),
            Self::Terminate => NonZeroU8::new(143).expect("143 is non-zero"),
            Self::Hangup => NonZeroU8::new(129).expect("129 is non-zero"),
        }
    }

    const fn bit(self) -> u8 {
        match self {
            Self::Interrupt => INTERRUPT,
            Self::Terminate => TERMINATE,
            Self::Hangup => HANGUP,
        }
    }
}

/// Why installing the process signal bridge failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignalError {
    /// This target has no supported process-control callback API.
    Unsupported,
    /// Another owner already installed the bridge in this process.
    AlreadyInstalled,
    /// The OS rejected installation; the value is its native error code.
    System(i32),
}

impl fmt::Display for SignalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported => f.write_str("process signal handling is unsupported"),
            Self::AlreadyInstalled => f.write_str("process signal bridge is already installed"),
            Self::System(code) => write!(f, "process signal installation failed (code {code})"),
        }
    }
}

impl std::error::Error for SignalError {}

/// Whether this target has a native callback path for process termination.
#[must_use]
pub const fn supported() -> bool {
    cfg!(any(unix, windows))
}

/// Record a termination request from an OS adapter or an embedding host.
///
/// This is intentionally a separate API from [`install`]: a GUI/event-loop
/// host may already receive the platform callback and only needs to publish it
/// to the engine's frame thread. Repeated notifications coalesce by signal
/// kind, while different kinds remain observable in deterministic priority
/// order through [`poll`].
pub fn notify(signal: ProcessSignal) {
    PENDING.fetch_or(signal.bit(), Ordering::Release);
}

/// Remove and return the oldest pending signal in deterministic priority order.
pub fn poll() -> Option<ProcessSignal> {
    loop {
        let bits = PENDING.load(Ordering::Acquire);
        let (bit, signal) = if bits & INTERRUPT != 0 {
            (INTERRUPT, ProcessSignal::Interrupt)
        } else if bits & TERMINATE != 0 {
            (TERMINATE, ProcessSignal::Terminate)
        } else if bits & HANGUP != 0 {
            (HANGUP, ProcessSignal::Hangup)
        } else {
            return None;
        };
        let remaining = bits & !bit;
        if PENDING
            .compare_exchange_weak(bits, remaining, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            return Some(signal);
        }
    }
}

/// Clear all pending requests. Intended for a host that has consumed or
/// deliberately reset its process-lifecycle state before starting a run.
pub fn clear() {
    PENDING.store(0, Ordering::Release);
}

/// Whether the native callback bridge is currently installed.
#[must_use]
pub fn installed() -> bool {
    INSTALLED.load(Ordering::Acquire)
}

/// An owned installation of the native process callback.
pub struct SignalSubscription {
    _inner: backend::Subscription,
}

/// Install the native process callback bridge.
pub fn install() -> Result<SignalSubscription, SignalError> {
    if !supported() {
        return Err(SignalError::Unsupported);
    }
    if INSTALLED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Err(SignalError::AlreadyInstalled);
    }
    match backend::install() {
        Ok(inner) => Ok(SignalSubscription { _inner: inner }),
        Err(error) => {
            INSTALLED.store(false, Ordering::Release);
            Err(error)
        }
    }
}

#[cfg(unix)]
mod backend {
    use super::{record_raw, SignalError};

    const SIGINT: i32 = 2;
    const SIGHUP: i32 = 1;
    const SIGTERM: i32 = 15;
    const SIG_ERR: usize = usize::MAX;

    unsafe extern "C" {
        fn signal(signum: i32, handler: usize) -> usize;
    }

    extern "C" fn handler(signal: i32) {
        record_raw(signal);
    }

    pub(super) struct Subscription {
        previous: [usize; 3],
    }

    pub(super) fn install() -> Result<Subscription, SignalError> {
        let signals = [SIGINT, SIGTERM, SIGHUP];
        let mut previous = [0usize; 3];
        for (index, &signal_number) in signals.iter().enumerate() {
            // SAFETY: the callback is an extern "C" function with static
            // lifetime; it only performs an atomic operation.
            let old = unsafe { signal(signal_number, handler as *const () as usize) };
            if old == SIG_ERR {
                for (&restore_signal, &restore_handler) in
                    signals[..index].iter().zip(&previous[..index])
                {
                    // SAFETY: restore the disposition returned by `signal`.
                    unsafe { signal(restore_signal, restore_handler) };
                }
                let code = std::io::Error::last_os_error().raw_os_error().unwrap_or(-1);
                return Err(SignalError::System(code));
            }
            previous[index] = old;
        }
        Ok(Subscription { previous })
    }

    impl Drop for Subscription {
        fn drop(&mut self) {
            for (&signal_number, &old) in [SIGINT, SIGTERM, SIGHUP].iter().zip(&self.previous) {
                // SAFETY: these are the exact dispositions returned at install.
                unsafe { signal(signal_number, old) };
            }
            super::mark_uninstalled();
        }
    }
}

#[cfg(windows)]
mod backend {
    use super::{record_raw, SignalError};

    const CTRL_C_EVENT: u32 = 0;
    const CTRL_BREAK_EVENT: u32 = 1;
    const CTRL_CLOSE_EVENT: u32 = 2;
    const CTRL_LOGOFF_EVENT: u32 = 5;
    const CTRL_SHUTDOWN_EVENT: u32 = 6;

    type Handler = unsafe extern "system" fn(u32) -> i32;

    unsafe extern "system" {
        fn SetConsoleCtrlHandler(handler: Option<Handler>, add: i32) -> i32;
        fn GetLastError() -> u32;
    }

    unsafe extern "system" fn handler(event: u32) -> i32 {
        match event {
            CTRL_C_EVENT | CTRL_BREAK_EVENT => record_raw(2),
            CTRL_CLOSE_EVENT | CTRL_LOGOFF_EVENT | CTRL_SHUTDOWN_EVENT => record_raw(15),
            _ => return 0,
        }
        1
    }

    pub(super) struct Subscription;

    pub(super) fn install() -> Result<Subscription, SignalError> {
        // SAFETY: the handler is static and only records an atomic bit.
        let ok = unsafe { SetConsoleCtrlHandler(Some(handler), 1) };
        if ok == 0 {
            Err(SignalError::System(unsafe { GetLastError() } as i32))
        } else {
            Ok(Subscription)
        }
    }

    impl Drop for Subscription {
        fn drop(&mut self) {
            // SAFETY: remove the exact static callback installed above.
            unsafe { SetConsoleCtrlHandler(Some(handler), 0) };
            super::mark_uninstalled();
        }
    }
}

#[cfg(not(any(unix, windows)))]
mod backend {
    use super::SignalError;

    pub(super) struct Subscription;

    pub(super) fn install() -> Result<Subscription, SignalError> {
        Err(SignalError::Unsupported)
    }
}

fn record_raw(raw: i32) {
    let signal = match raw {
        2 => ProcessSignal::Interrupt,
        15 => ProcessSignal::Terminate,
        1 => ProcessSignal::Hangup,
        _ => return,
    };
    notify(signal);
}

fn mark_uninstalled() {
    INSTALLED.store(false, Ordering::Release);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn notification_queue_is_coalesced_and_prioritized() {
        let _guard = TEST_LOCK.lock().expect("signal test lock");
        clear();
        notify(ProcessSignal::Hangup);
        notify(ProcessSignal::Terminate);
        notify(ProcessSignal::Interrupt);
        assert_eq!(poll(), Some(ProcessSignal::Interrupt));
        assert_eq!(poll(), Some(ProcessSignal::Terminate));
        assert_eq!(poll(), Some(ProcessSignal::Hangup));
        assert_eq!(poll(), None);
    }

    #[test]
    fn exit_codes_are_stable() {
        let _guard = TEST_LOCK.lock().expect("signal test lock");
        assert_eq!(ProcessSignal::Interrupt.exit_code().get(), 130);
        assert_eq!(ProcessSignal::Terminate.exit_code().get(), 143);
        assert_eq!(ProcessSignal::Hangup.exit_code().get(), 129);
    }

    #[test]
    fn installation_matches_target_capability() {
        let _guard = TEST_LOCK.lock().expect("signal test lock");
        clear();
        if supported() {
            let subscription = install().expect("native signal bridge installs");
            assert!(installed());
            drop(subscription);
            assert!(!installed());
        } else {
            assert!(matches!(install(), Err(SignalError::Unsupported)));
        }
    }
}
