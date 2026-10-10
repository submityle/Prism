use crate::prelude::*;

#[test]
fn detects_nonzero_cores() {
    let info = CpuInfo::detect();
    assert!(info.logical_cores >= 1);
    assert!(info.preferred_f32_lanes() >= 1);
}

#[test]
fn arch_matches_cfg() {
    let info = CpuInfo::detect();
    #[cfg(target_arch = "x86_64")]
    assert_eq!(info.arch, "x86_64");
    #[cfg(target_arch = "aarch64")]
    assert_eq!(info.arch, "aarch64");
    let _ = info;
}

#[test]
fn monotonic_clock_does_not_go_backwards() {
    let a = clock_now();
    let b = clock_now();
    assert!(b >= a);
    assert_eq!(b.saturating_since(a), b.0 - a.0);
}

#[test]
fn platform_current_is_consistent() {
    let p = Platform::current();
    assert_eq!(p.caps.has_std, cfg!(feature = "std"));
    assert_ne!(p.os, Os::Unknown, "desktop test target should be known");
    // M4 capability bits track feature + target cfg honestly.
    assert_eq!(
        p.caps.has_mmap,
        cfg!(feature = "std") && cfg!(any(unix, windows))
    );
    assert_eq!(
        p.caps.has_dynlib_unload,
        cfg!(feature = "dynlib") && cfg!(any(unix, windows))
    );
    // M5 capability bits.
    assert_eq!(
        p.caps.has_subprocess,
        cfg!(feature = "std") && cfg!(not(target_family = "wasm"))
    );
    assert_eq!(p.caps.has_wall_clock, cfg!(feature = "std"));
    let _ = (spin_hint(), full_fence());
}

#[cfg(feature = "std")]
#[test]
fn wall_clock_tracks_unix_epoch_ordering() {
    let a = wall_now();
    let b = wall_now();
    // Wall clock can step, but two back-to-back reads on a sane host are
    // ordered; assert via the signed delta API that tolerates either way.
    assert!(b.signed_nanos_since(a) >= 0 || a.signed_nanos_since(b) >= 0);
    // We are well past the Unix epoch.
    assert!(a.unix_seconds() > 1_600_000_000);
    assert!(a.unix_nanos() > 0);
    // epoch helpers are consistent.
    assert_eq!(WallTime::unix_epoch().unix_nanos(), 0);
    assert_eq!(WallTime::unix_epoch().unix_seconds(), 0);
}

#[cfg(feature = "std")]
#[test]
fn wall_clock_sample_pairs_wall_and_monotonic() {
    let s0 = WallClock::new().sample();
    let s1 = wallclock::sample();
    // Monotonic half never goes backwards.
    assert!(s1.monotonic >= s0.monotonic);
    // Wall half is a sane calendar time.
    assert!(s0.wall.unix_seconds() > 1_600_000_000);
}

#[cfg(feature = "std")]
#[test]
fn wall_time_duration_since_is_none_on_backward() {
    let later =
        WallTime::from_system_time(std::time::UNIX_EPOCH + core::time::Duration::from_secs(100));
    let earlier = WallTime::unix_epoch();
    assert_eq!(
        later.duration_since(earlier),
        Some(core::time::Duration::from_secs(100))
    );
    assert_eq!(earlier.duration_since(later), None);
    assert_eq!(earlier.signed_nanos_since(later), -100_000_000_000);
}

#[cfg(feature = "std")]
#[test]
fn stdio_handles_and_terminal_probes_are_available() {
    use crate::stdio::{self, Stream};
    // Constructing the handles must not panic and must expose their stream.
    let _out = stdio::stdout();
    let _err = stdio::stderr();
    let _in = stdio::stdin();
    // Probing is_terminal is pure and total; under `cargo test` these are
    // typically pipes, but we only assert the call is well-defined.
    let _ = (
        stdio::is_terminal(Stream::Stdin),
        stdio::is_terminal(Stream::Stdout),
        stdio::is_terminal(Stream::Stderr),
    );
}
