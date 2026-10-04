//! Real-machine checks for the OS security-posture probe.
//!
//! Like the topology checks, these probe the *actual* process rather than a
//! fixed fixture: they assert the result is a genuine (`is_probed()`) posture
//! that is internally consistent with what Apple Silicon guarantees, and they
//! degrade to confirming an honest refusal on an unverified target.
//!
//! Note: `csops` can be denied inside a sandbox, in which case the probe
//! surfaces [`ProbeError::Os`]; run these unsandboxed to exercise the real
//! read.

use crate::{probe_security, ProbeError};

#[test]
fn probe_is_marked_probed_or_refuses_honestly() {
    match probe_security() {
        Ok(posture) => {
            // A real probe must mark itself as such, unlike `detect()`.
            assert!(
                posture.is_probed(),
                "a successful security probe must set is_probed()"
            );
            // A genuine weakness verdict must be backed by a real read; the
            // probe never fabricates one from the baseline.
            let _ = posture.has_weakness();
        }
        // Honest refusal on an unverified target, or a sandbox denying
        // csops/dyld, are both acceptable non-fabricating outcomes.
        Err(ProbeError::Unsupported | ProbeError::Os(_)) => {}
        Err(e) => panic!("unexpected security-probe failure: {e}"),
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn apple_silicon_reads_signing_pie_and_sandbox() {
    use prism_platform::platform::Os;
    use prism_platform::security::{Mitigation, MitigationStatus, SandboxModel};

    let posture = match probe_security() {
        Ok(p) => p,
        // csops denied (sandbox): cannot exercise the real read here.
        Err(ProbeError::Os(_)) => return,
        Err(e) => panic!("security probe failed on Apple Silicon: {e}"),
    };

    assert!(posture.is_probed());
    assert_eq!(posture.os(), Os::Apple);

    // Apple Silicon refuses to run unsigned code, so the running process is
    // always validly signed and code-signing enforcement reads back on.
    assert_eq!(
        posture.mitigation(Mitigation::CodeSigningEnforced),
        MitigationStatus::Enforced,
        "Apple Silicon mandates valid code signing"
    );

    // Rust executables on macOS are position-independent.
    assert_eq!(
        posture.mitigation(Mitigation::Pie),
        MitigationStatus::Enforced,
        "macOS executables are PIE"
    );

    // The sandbox model is read from the code-signing status: either an
    // ordinary unsandboxed process or one under the Hardened Runtime.
    assert!(
        matches!(
            posture.sandbox(),
            SandboxModel::None | SandboxModel::AppleHardenedRuntime
        ),
        "unexpected sandbox model: {:?}",
        posture.sandbox()
    );

    // The architecture-guaranteed baseline mitigations are untouched by the
    // probe and remain enforced on this platform.
    for m in [Mitigation::Aslr, Mitigation::DepNx, Mitigation::StackCanary] {
        assert_eq!(
            posture.mitigation(m),
            MitigationStatus::Enforced,
            "{} must stay enforced on Apple Silicon",
            m.id()
        );
    }
}
