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
    let _ = (spin_hint(), full_fence());
}
