//! Tests for the §24.6 capability database, degradation API, and report.

use crate::capability::{
    Capability, CapabilityDatabase, Category, Selection, SupportLevel,
};
use crate::platform::{Os, PlatformCaps};

/// A capability-flag snapshot for a fully-featured desktop build.
fn desktop_caps() -> PlatformCaps {
    PlatformCaps {
        has_std: true,
        has_monotonic_clock: true,
        has_mmap: true,
        has_native_file_watch: true,
        has_dynlib_unload: true,
        has_subprocess: true,
        has_wall_clock: true,
        has_crash_capture: true,
    }
}

/// A snapshot for a `no_std`, no-OS build (bare metal / minimal wasm).
fn bare_caps() -> PlatformCaps {
    PlatformCaps {
        has_std: false,
        has_monotonic_clock: false,
        has_mmap: false,
        has_native_file_watch: false,
        has_dynlib_unload: false,
        has_subprocess: false,
        has_wall_clock: false,
        has_crash_capture: false,
    }
}

/// A `std` build with no native mmap / watch / dynlib / crash backend — e.g.
/// a conservative portable target that still has threads and a wall clock.
fn portable_std_caps() -> PlatformCaps {
    PlatformCaps {
        has_std: true,
        has_monotonic_clock: true,
        has_mmap: false,
        has_native_file_watch: false,
        has_dynlib_unload: false,
        has_subprocess: true,
        has_wall_clock: true,
        has_crash_capture: false,
    }
}

#[test]
fn catalog_is_closed_and_unique() {
    // Every variant appears exactly once in ALL and keys are unique.
    assert_eq!(Capability::ALL.len(), 10);
    for (i, &a) in Capability::ALL.iter().enumerate() {
        for &b in &Capability::ALL[i + 1..] {
            assert_ne!(a, b, "duplicate capability variant");
            assert_ne!(a.key(), b.key(), "duplicate capability key");
            assert_ne!(a.summary(), "", "missing summary");
        }
    }
    // Keys are stable, lowercase, snake_case-ish (no spaces).
    for &c in Capability::ALL {
        assert!(!c.key().is_empty());
        assert!(!c.key().contains(' '));
        assert!(!c.summary().is_empty());
    }
}

#[test]
fn entries_match_catalog_order_and_lookup() {
    let db = CapabilityDatabase::from_platform(Os::Linux, desktop_caps());
    assert_eq!(db.entries().len(), Capability::ALL.len());
    // Stored order equals the catalog order.
    for (entry, &cap) in db.entries().iter().zip(Capability::ALL) {
        assert_eq!(entry.capability, cap);
        // `support()` lookup returns the same record as the positional entry.
        assert_eq!(db.support(cap), *entry);
    }
}

#[test]
fn desktop_linux_truth_table() {
    // Hand-verified oracle for a full desktop Linux build.
    let db = CapabilityDatabase::from_platform(Os::Linux, desktop_caps());
    let native = [
        Capability::Std,
        Capability::MonotonicClock,
        Capability::WallClock,
        Capability::Mmap,
        Capability::NativeFileWatch,
        Capability::DynlibUnload,
        Capability::Subprocess,
        Capability::CrashCapture,
        Capability::HugePages,
        Capability::ThreadAffinity,
    ];
    for cap in native {
        assert_eq!(db.level(cap), SupportLevel::Native, "{}", cap.key());
        assert!(db.is_native(cap));
        assert!(db.support(cap).detail.is_empty());
    }
    assert_eq!(db.count_at(SupportLevel::Native), 10);
    assert_eq!(db.count_at(SupportLevel::Degraded), 0);
    assert_eq!(db.count_at(SupportLevel::Unsupported), 0);
    assert_eq!(db.degraded().count(), 0);
}

#[test]
fn desktop_macos_degrades_huge_pages_and_drops_affinity() {
    let db = CapabilityDatabase::from_platform(Os::Apple, desktop_caps());
    // Huge pages degrade (correct, no TLB win); affinity is unsupported.
    assert_eq!(db.level(Capability::HugePages), SupportLevel::Degraded);
    assert!(db.is_usable(Capability::HugePages));
    assert!(!db.is_native(Capability::HugePages));
    assert!(!db.support(Capability::HugePages).detail.is_empty());

    assert_eq!(db.level(Capability::ThreadAffinity), SupportLevel::Unsupported);
    assert!(!db.is_usable(Capability::ThreadAffinity));

    // Everything else on a full macOS build is native.
    assert_eq!(db.count_at(SupportLevel::Native), 8);
    assert_eq!(db.count_at(SupportLevel::Degraded), 1);
    assert_eq!(db.count_at(SupportLevel::Unsupported), 1);
    // The degradation matrix lists exactly huge pages + affinity.
    let mut flagged: Vec<Capability> =
        db.degraded().map(|s| s.capability).collect();
    flagged.sort();
    let mut expected = alloc::vec![Capability::HugePages, Capability::ThreadAffinity];
    expected.sort();
    assert_eq!(flagged, expected);
}

#[test]
fn portable_std_degrades_mmap_and_watch() {
    let db = CapabilityDatabase::from_platform(Os::Linux, portable_std_caps());
    // mmap and file watch degrade to the buffered / polling fallbacks.
    assert_eq!(db.level(Capability::Mmap), SupportLevel::Degraded);
    assert_eq!(db.level(Capability::NativeFileWatch), SupportLevel::Degraded);
    assert!(db.is_usable(Capability::Mmap));
    // dynlib + crash have no fallback here.
    assert_eq!(db.level(Capability::DynlibUnload), SupportLevel::Unsupported);
    assert_eq!(db.level(Capability::CrashCapture), SupportLevel::Unsupported);
    // Linux + std ⇒ huge pages and affinity are native.
    assert_eq!(db.level(Capability::HugePages), SupportLevel::Native);
    assert_eq!(db.level(Capability::ThreadAffinity), SupportLevel::Native);
}

#[test]
fn bare_no_std_is_all_unsupported() {
    let db = CapabilityDatabase::from_platform(Os::Unknown, bare_caps());
    for &cap in Capability::ALL {
        assert_eq!(db.level(cap), SupportLevel::Unsupported, "{}", cap.key());
        assert!(!db.is_usable(cap));
        assert!(!db.support(cap).detail.is_empty(), "{} missing reason", cap.key());
    }
    assert_eq!(db.count_at(SupportLevel::Unsupported), 10);
}

#[test]
fn web_build_truth_table() {
    // A wasm build: std-ish services off at the OS layer, no subprocess/mmap/
    // dynlib/crash; wall + monotonic clock present; huge pages degrade.
    let caps = PlatformCaps {
        has_std: true,
        has_monotonic_clock: true,
        has_mmap: false,
        has_native_file_watch: false,
        has_dynlib_unload: false,
        has_subprocess: false,
        has_wall_clock: true,
        has_crash_capture: false,
    };
    let db = CapabilityDatabase::from_platform(Os::Web, caps);
    assert_eq!(db.level(Capability::Subprocess), SupportLevel::Unsupported);
    assert_eq!(db.level(Capability::ThreadAffinity), SupportLevel::Unsupported);
    assert_eq!(db.level(Capability::HugePages), SupportLevel::Degraded);
    assert_eq!(db.level(Capability::Mmap), SupportLevel::Degraded);
    assert_eq!(db.level(Capability::MonotonicClock), SupportLevel::Native);
}

#[test]
fn require_select_takes_native_branch_only() {
    let db = CapabilityDatabase::from_platform(Os::Linux, desktop_caps());
    let mut fallback_ran = false;
    let sel: Selection<&str> = db
        .require(Capability::HugePages)
        .select(|| "huge", || {
            fallback_ran = true;
            "normal"
        });
    assert_eq!(sel.value, "huge");
    assert!(!sel.degraded);
    assert!(!fallback_ran, "fallback must not run when native is available");
    assert_eq!(sel.support.level, SupportLevel::Native);
}

#[test]
fn require_select_takes_fallback_branch_only() {
    let db = CapabilityDatabase::from_platform(Os::Apple, desktop_caps());
    let mut native_ran = false;
    let sel = db.require(Capability::ThreadAffinity).select(
        || {
            native_ran = true;
            1u32
        },
        || 2u32,
    );
    assert_eq!(sel.value, 2);
    assert!(sel.degraded);
    assert!(!native_ran, "native branch must not run when unavailable");
    assert_eq!(sel.support.level, SupportLevel::Unsupported);
}

#[test]
fn require_value_and_native_only_helpers() {
    let db = CapabilityDatabase::from_platform(Os::Apple, desktop_caps());

    // Degraded huge pages ⇒ or_fallback picks the fallback value.
    let sel = db.require(Capability::HugePages).or_fallback(64, 4);
    assert_eq!(sel.value, 4);
    assert!(sel.degraded);

    // Native wall clock ⇒ or_fallback picks the native value.
    let sel = db.require(Capability::WallClock).or_fallback("real", "stub");
    assert_eq!(sel.value, "real");
    assert!(!sel.degraded);

    // native_only yields Some only for native capabilities.
    assert_eq!(db.require(Capability::WallClock).native_only(|| 7), Some(7));
    assert_eq!(
        db.require(Capability::ThreadAffinity).native_only(|| 7),
        None
    );
}

#[test]
fn report_is_deterministic_and_well_formed() {
    let db = CapabilityDatabase::from_platform(Os::Apple, desktop_caps());
    let a = db.report();
    let b = db.report();
    assert_eq!(a, b, "report must be byte-stable");

    // Header + every category header present + a summary line.
    assert!(a.starts_with("platform capability report\n"));
    for cat in [
        Category::Timing,
        Category::Memory,
        Category::Io,
        Category::Concurrency,
        Category::Diagnostics,
        Category::Process,
    ] {
        let header = alloc::format!("[{}]", cat.key());
        assert!(a.contains(&header), "missing category header {}", cat.key());
    }
    // Every capability key appears exactly once.
    for &cap in Capability::ALL {
        let needle = alloc::format!("  {}: ", cap.key());
        assert_eq!(
            a.matches(&needle).count(),
            1,
            "capability {} should appear exactly once",
            cap.key()
        );
    }
    assert!(a.contains("summary: 8 native, 1 degraded, 1 unsupported"));
    // Degraded/unsupported lines carry a reason; native lines do not have " — ".
    assert!(a.contains("huge_pages: degraded — "));
    assert!(a.contains("thread_affinity: unsupported — "));
    assert!(a.contains("wall_clock: native\n"));
}

#[test]
fn current_build_is_self_consistent() {
    let db = CapabilityDatabase::current();
    for entry in db.entries() {
        // Native entries never carry a detail; non-native entries always do.
        if entry.is_native() {
            assert!(entry.detail.is_empty(), "{} native w/ detail", entry.capability.key());
        } else {
            assert!(!entry.detail.is_empty(), "{} non-native w/o reason", entry.capability.key());
        }
    }
    // Counts partition the catalog.
    let total = db.count_at(SupportLevel::Native)
        + db.count_at(SupportLevel::Degraded)
        + db.count_at(SupportLevel::Unsupported);
    assert_eq!(total, Capability::ALL.len());
    // The report mentions the host's actual std status consistently.
    let report = db.report();
    assert!(report.contains("std: "));
}
