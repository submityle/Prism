//! M6 tests for crash capture: value model, capability mirroring, the `mock`
//! capture pipeline, and real `POSIX` handler install/uninstall.
//!
//! Crash capture keeps its registered handler and last-context in process-wide
//! statics. The host test binary's other tests (`mod tests`) never touch those
//! statics, so the single global-mutating test here owns them exclusively:
//! exactly one test installs a handler / simulates / installs real handlers, so
//! there is no cross-test race on the shared store.

use crate::crash::{self, Backtrace, BuildMetadata, CrashError, Signal};

#[test]
fn signal_classification_round_trips() {
    for sig in [
        Signal::Segv,
        Signal::Abort,
        Signal::Bus,
        Signal::Ill,
        Signal::Fpe,
    ] {
        assert_eq!(Signal::from_raw(sig.raw()), sig, "raw round-trip");
        assert!(sig.name().starts_with("SIG"));
    }
    // An unknown signal number is preserved verbatim.
    let other = Signal::from_raw(1234);
    assert_eq!(other, Signal::Other(1234));
    assert_eq!(other.raw(), 1234);
    assert_eq!(other.name(), "SIGOTHER");
}

#[test]
fn backtrace_empty_is_consistent() {
    let bt = Backtrace::empty();
    assert!(bt.is_empty());
    assert_eq!(bt.len(), 0);
    assert_eq!(bt.as_slice(), &[] as &[usize]);
}

#[test]
fn build_metadata_empty_and_default_agree() {
    let empty = BuildMetadata::empty();
    assert_eq!(empty.module_name, "");
    assert_eq!(empty.version, "");
    assert_eq!(empty.build_id, "");
    assert_eq!(empty.image_base, 0);
    assert_eq!(BuildMetadata::default(), empty);
}

#[test]
fn supported_mirrors_the_compiled_backend_and_caps_bit() {
    let expected = cfg!(all(
        unix,
        not(any(target_os = "android", target_os = "ios"))
    ));
    assert_eq!(crash::SUPPORTED, expected);
    assert_eq!(crash::supported(), expected);

    // The platform capability bit must mirror the crash backend exactly.
    let caps = crate::Platform::current().caps;
    assert_eq!(
        caps.has_crash_capture,
        crash::SUPPORTED,
        "PlatformCaps crash bit must mirror the compiled crash backend"
    );
}

#[test]
fn crash_error_displays_are_distinct_and_informative() {
    let unsupported = CrashError::Unsupported.to_string();
    let os = CrashError::Os(13).to_string();
    assert!(unsupported.contains("unsupported"));
    assert!(os.contains("13"));
    assert_ne!(unsupported, os);
}

// The single global-store-mutating test when the `mock` backend is compiled in.
#[cfg(feature = "mock")]
#[test]
fn mock_pipeline_captures_dispatches_and_retrieves() {
    use crate::crash::mock;
    use core::sync::atomic::{AtomicI32, AtomicUsize, Ordering};

    static SEEN_SIGNUM: AtomicI32 = AtomicI32::new(-1);
    static SEEN_ADDR: AtomicUsize = AtomicUsize::new(0);

    fn handler(ctx: &crash::CrashContext) {
        SEEN_SIGNUM.store(ctx.signal_number, Ordering::SeqCst);
        SEEN_ADDR.store(ctx.fault_address, Ordering::SeqCst);
    }

    const META: BuildMetadata = BuildMetadata {
        module_name: "prism_test",
        version: "1.2.3",
        build_id: "deadbeef",
        image_base: 0x4000,
    };

    crash::clear_last_context();
    mock::install_with(handler, META);

    let fault = 0xdead_0000_usize;
    let ctx = mock::simulate(Signal::Segv, fault);

    // The returned context is fully populated and deterministic.
    assert_eq!(ctx.signal, Signal::Segv);
    assert_eq!(ctx.signal_number, Signal::Segv.raw());
    assert_eq!(ctx.fault_address, fault);
    assert_eq!(ctx.metadata, META, "pre-registered metadata is snapshotted");
    assert!(ctx.backtrace.len() <= crash::MAX_FRAMES);
    assert_eq!(ctx.backtrace.len(), ctx.backtrace.as_slice().len());

    // The registered handler ran synchronously with the same context.
    assert_eq!(SEEN_SIGNUM.load(Ordering::SeqCst), Signal::Segv.raw());
    assert_eq!(SEEN_ADDR.load(Ordering::SeqCst), fault);

    // The context is retrievable, then consumed exactly once.
    let last = crash::last_context().expect("a context was captured");
    assert_eq!(last.fault_address, fault);
    let taken = crash::take_last_context().expect("take returns the captured context");
    assert_eq!(taken.fault_address, fault);
    assert!(
        crash::take_last_context().is_none(),
        "the captured context is consumed exactly once"
    );

    mock::uninstall();
}

// When `mock` is NOT compiled in, the real backend is the sole global-store
// user, so the install/uninstall round-trip test owns the statics alone.
#[cfg(all(
    unix,
    not(any(target_os = "android", target_os = "ios")),
    not(feature = "mock")
))]
#[test]
fn real_posix_handlers_install_and_restore() {
    fn noop(_ctx: &crash::CrashContext) {}

    assert!(crash::supported());
    assert!(!crash::is_installed(), "handlers start uninstalled");

    crash::install(noop).expect("install arms POSIX handlers on this host");
    assert!(crash::is_installed());

    // Every captured signal's disposition now points at our trampoline.
    let trampoline = crash::trampoline_addr();
    for sig in [
        Signal::Segv,
        Signal::Abort,
        Signal::Bus,
        Signal::Ill,
        Signal::Fpe,
    ] {
        assert_eq!(
            crash::disposition(sig.raw()),
            trampoline,
            "{} disposition must point at the installed trampoline",
            sig.name()
        );
    }

    // Idempotent re-install is a no-op that still reports installed.
    crash::install(noop).expect("second install is a no-op");
    assert!(crash::is_installed());

    crash::uninstall().expect("uninstall restores previous dispositions");
    assert!(!crash::is_installed());
}

// On targets with no real backend and no mock, install honestly degrades.
#[cfg(all(
    not(all(unix, not(any(target_os = "android", target_os = "ios")))),
    not(feature = "mock")
))]
#[test]
fn unsupported_targets_degrade_gracefully() {
    fn noop(_ctx: &crash::CrashContext) {}
    assert!(!crash::supported());
    assert_eq!(crash::install(noop), Err(CrashError::Unsupported));
}
