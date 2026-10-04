//! Oracle tests for the §24.5 security-posture data model. Every expected value
//! is hand-derived from the documented platform defaults, not from the code
//! under test.

use crate::platform::Os;
use crate::security::mitigation::{Mitigation, MitigationStatus};
use crate::security::sandbox::{Access, SandboxModel};
use crate::security::signing::{
    IntegrityPolicy, LoadDecision, RejectReason, SigningStatus, SigningStrictness,
};
use crate::security::{SecurityPosture, MITIGATION_COUNT};

#[test]
fn mitigation_all_is_complete_and_unique() {
    assert_eq!(Mitigation::ALL.len(), MITIGATION_COUNT);
    assert_eq!(MITIGATION_COUNT, 10);
    for (i, a) in Mitigation::ALL.iter().enumerate() {
        for b in &Mitigation::ALL[i + 1..] {
            assert_ne!(a, b, "duplicate mitigation in ALL");
            assert_ne!(a.id(), b.id(), "duplicate mitigation id");
        }
    }
}

#[test]
fn linux_baseline_matches_hand_table() {
    // Hand-derived from modern Linux distro release defaults.
    let expect = [
        (Mitigation::Aslr, MitigationStatus::Enforced),
        (Mitigation::DepNx, MitigationStatus::Enforced),
        (Mitigation::Pie, MitigationStatus::Enforced),
        (Mitigation::StackCanary, MitigationStatus::Enforced),
        (Mitigation::ControlFlowGuard, MitigationStatus::Unknown),
        (Mitigation::ShadowStack, MitigationStatus::Unknown),
        (Mitigation::PointerAuth, MitigationStatus::Unknown),
        (Mitigation::Relro, MitigationStatus::Enforced),
        (Mitigation::FortifySource, MitigationStatus::Enforced),
        (Mitigation::CodeSigningEnforced, MitigationStatus::Unknown),
    ];
    for (m, want) in expect {
        assert_eq!(m.platform_default(Os::Linux), want, "{}", m.id());
    }
}

#[test]
fn apple_baseline_matches_hand_table() {
    assert_eq!(
        Mitigation::Aslr.platform_default(Os::Apple),
        MitigationStatus::Enforced
    );
    // Apple uses image ASLR, not ELF PIE/RELRO/FORTIFY.
    assert_eq!(
        Mitigation::Pie.platform_default(Os::Apple),
        MitigationStatus::Enforced
    );
    assert_eq!(
        Mitigation::Relro.platform_default(Os::Apple),
        MitigationStatus::NotApplicable
    );
    assert_eq!(
        Mitigation::FortifySource.platform_default(Os::Apple),
        MitigationStatus::NotApplicable
    );
    // Mandatory code signing on Apple silicon / Gatekeeper.
    assert_eq!(
        Mitigation::CodeSigningEnforced.platform_default(Os::Apple),
        MitigationStatus::Enforced
    );
}

#[test]
fn windows_pie_is_not_applicable_but_aslr_enforced() {
    assert_eq!(
        Mitigation::Aslr.platform_default(Os::Windows),
        MitigationStatus::Enforced
    );
    assert_eq!(
        Mitigation::Pie.platform_default(Os::Windows),
        MitigationStatus::NotApplicable
    );
    // CFG is opt-in (`/guard:cf`), so it is not assumed.
    assert_eq!(
        Mitigation::ControlFlowGuard.platform_default(Os::Windows),
        MitigationStatus::Unknown
    );
}

#[test]
fn web_mitigations_are_not_applicable() {
    for m in Mitigation::ALL {
        assert_eq!(
            m.platform_default(Os::Web),
            MitigationStatus::NotApplicable,
            "{} should be N/A on Web",
            m.id()
        );
    }
}

#[test]
fn baseline_posture_is_never_marked_probed() {
    for os in [Os::Windows, Os::Apple, Os::Linux, Os::Android, Os::Web, Os::Unknown] {
        let p = SecurityPosture::baseline(os);
        assert!(!p.is_probed(), "baseline must not claim to be probed");
        assert_eq!(p.os(), os);
    }
}

#[test]
fn baseline_has_no_weakness_because_nothing_is_notenforced() {
    // The baseline only ever uses Enforced / NotApplicable / Unknown — never a
    // definitive NotEnforced — so it must not report a weakness.
    for os in [Os::Windows, Os::Apple, Os::Linux, Os::Android] {
        assert!(!SecurityPosture::baseline(os).has_weakness());
    }
}

#[test]
fn probed_notenforced_is_a_weakness() {
    let p = SecurityPosture::builder(Os::Linux)
        .mitigation(Mitigation::Aslr, MitigationStatus::NotEnforced)
        .mark_probed()
        .build();
    assert!(p.is_probed());
    assert!(p.has_weakness());
    assert_eq!(p.mitigation(Mitigation::Aslr), MitigationStatus::NotEnforced);
    // Unchanged entries keep the Linux baseline.
    assert_eq!(p.mitigation(Mitigation::DepNx), MitigationStatus::Enforced);
}

#[test]
fn mitigations_iter_is_stable_and_full() {
    let p = SecurityPosture::baseline(Os::Linux);
    let collected: Vec<_> = p.mitigations().map(|(m, _)| m).collect();
    assert_eq!(&collected[..], &Mitigation::ALL[..]);
}

#[test]
fn sandbox_defaults_are_conservative() {
    assert_eq!(SandboxModel::default_for(Os::Android), SandboxModel::AndroidAppSandbox);
    assert_eq!(SandboxModel::default_for(Os::Web), SandboxModel::WebBrowser);
    // Desktop is unsandboxed unless a real probe says otherwise.
    assert_eq!(SandboxModel::default_for(Os::Windows), SandboxModel::None);
    assert_eq!(SandboxModel::default_for(Os::Apple), SandboxModel::None);
    assert_eq!(SandboxModel::default_for(Os::Linux), SandboxModel::None);
    assert_eq!(SandboxModel::default_for(Os::Unknown), SandboxModel::Unknown);
}

#[test]
fn sandbox_capabilities_match_models() {
    let none = SandboxModel::None.capabilities();
    assert_eq!(none.filesystem, Access::Full);
    assert_eq!(none.network, Access::Full);
    assert!(none.can_spawn_process);

    let web = SandboxModel::WebBrowser.capabilities();
    assert_eq!(web.filesystem, Access::None);
    assert!(!web.filesystem.is_available());
    assert!(!web.can_spawn_process);

    let apple = SandboxModel::AppleAppSandbox.capabilities();
    assert_eq!(apple.filesystem, Access::Container);
    assert!(apple.filesystem.is_available());
    assert_eq!(apple.network, Access::Full);
    assert!(!apple.can_spawn_process);

    assert!(SandboxModel::WebBrowser.is_sandboxed());
    assert!(!SandboxModel::None.is_sandboxed());
    assert!(!SandboxModel::Unknown.is_sandboxed());
}

#[test]
fn integrity_observe_allows_everything() {
    let p = IntegrityPolicy::observe_only();
    for s in [
        SigningStatus::Trusted,
        SigningStatus::SignedUntrusted,
        SigningStatus::Invalid,
        SigningStatus::Unsigned,
        SigningStatus::Unknown,
    ] {
        assert_eq!(p.decision(s), LoadDecision::Allow, "{s:?}");
        assert!(p.decision(s).is_allowed());
    }
}

#[test]
fn integrity_reject_bad_default() {
    let p = IntegrityPolicy::default();
    assert_eq!(p.strictness, SigningStrictness::RejectBad);
    assert!(p.allow_unknown);
    assert_eq!(p.decision(SigningStatus::Trusted), LoadDecision::Allow);
    assert_eq!(p.decision(SigningStatus::SignedUntrusted), LoadDecision::Allow);
    assert_eq!(p.decision(SigningStatus::Unknown), LoadDecision::Allow);
    assert_eq!(
        p.decision(SigningStatus::Invalid),
        LoadDecision::Reject(RejectReason::InvalidSignature)
    );
    assert_eq!(
        p.decision(SigningStatus::Unsigned),
        LoadDecision::Reject(RejectReason::Unsigned)
    );
}

#[test]
fn integrity_reject_bad_without_allow_unknown() {
    let p = IntegrityPolicy {
        strictness: SigningStrictness::RejectBad,
        allow_unknown: false,
    };
    assert_eq!(
        p.decision(SigningStatus::Unknown),
        LoadDecision::Reject(RejectReason::Unverifiable)
    );
}

#[test]
fn integrity_locked_down_requires_trusted() {
    let p = IntegrityPolicy::locked_down();
    assert_eq!(p.strictness, SigningStrictness::RequireTrusted);
    assert!(!p.allow_unknown);
    assert_eq!(p.decision(SigningStatus::Trusted), LoadDecision::Allow);
    assert_eq!(
        p.decision(SigningStatus::SignedUntrusted),
        LoadDecision::Reject(RejectReason::Untrusted)
    );
    assert_eq!(
        p.decision(SigningStatus::Unknown),
        LoadDecision::Reject(RejectReason::Unverifiable)
    );
    assert_eq!(
        p.decision(SigningStatus::Unsigned),
        LoadDecision::Reject(RejectReason::Unsigned)
    );
    assert_eq!(
        p.decision(SigningStatus::Invalid),
        LoadDecision::Reject(RejectReason::InvalidSignature)
    );
}

#[test]
fn signing_status_predicates() {
    assert!(SigningStatus::Trusted.is_trusted());
    assert!(!SigningStatus::SignedUntrusted.is_trusted());
    assert!(SigningStatus::Unsigned.is_definitely_bad());
    assert!(SigningStatus::Invalid.is_definitely_bad());
    assert!(!SigningStatus::Unknown.is_definitely_bad());
    assert!(!SigningStatus::SignedUntrusted.is_definitely_bad());
}

#[test]
fn posture_integrates_sandbox_and_integrity() {
    let p = SecurityPosture::builder(Os::Apple)
        .sandbox(SandboxModel::AppleAppSandbox)
        .integrity_policy(IntegrityPolicy::locked_down())
        .mark_probed()
        .build();
    assert_eq!(p.sandbox(), SandboxModel::AppleAppSandbox);
    assert_eq!(p.sandbox_capabilities().filesystem, Access::Container);
    assert_eq!(
        p.module_load_decision(SigningStatus::Unsigned),
        LoadDecision::Reject(RejectReason::Unsigned)
    );
    assert_eq!(p.module_load_decision(SigningStatus::Trusted), LoadDecision::Allow);
}

#[test]
fn detect_is_portable_and_unprobed() {
    let p = SecurityPosture::detect();
    assert!(!p.is_probed());
    // On the test host (desktop) the mitigation table should be populated.
    let _ = p.mitigation(Mitigation::Aslr);
}
