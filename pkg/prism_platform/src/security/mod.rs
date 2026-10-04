//! Security-hardening posture: exploit-mitigation state, code-signing /
//! integrity policy, and sandbox capabilities as portable data (design §24.5).
//!
//! This layer does **defensive** hardening and probing only: it describes which
//! mitigations are expected, decides whether a module may load given its
//! signing status, and reports what the platform sandbox allows. It contains
//! no bypass, no attack tooling, and no anti-debug adversarial code.
//!
//! # Honesty boundary
//!
//! Everything here is a portable **data model** plus deterministic policy. The
//! aggregate [`SecurityPosture`] built by [`SecurityPosture::detect`] fills in
//! the parts that are knowable without a syscall — the OS family, the
//! per-platform mitigation *baseline* ([`Mitigation::platform_default`]), and
//! the conservative sandbox model ([`SandboxModel::default_for`]) — and marks
//! [`SecurityPosture::is_probed`] `false`. A real live probe (reading the
//! running image's load configuration, verifying signatures through the OS, or
//! inspecting the actual sandbox profile) needs per-OS APIs that are not wired
//! yet; until then callers must treat the posture as the platform's *expected*
//! policy, never as a verified read of this exact process.

use crate::platform::Os;

pub mod mitigation;
pub mod sandbox;
pub mod signing;

pub use mitigation::{Mitigation, MitigationStatus};
pub use sandbox::{Access, SandboxCapabilities, SandboxModel};
pub use signing::{
    IntegrityPolicy, LoadDecision, RejectReason, SigningStatus, SigningStrictness,
};

/// The number of modeled mitigations (see [`Mitigation::ALL`]).
pub const MITIGATION_COUNT: usize = Mitigation::ALL.len();

/// An aggregate, portable description of the process's security posture.
///
/// It carries the per-mitigation status table (indexed by [`Mitigation::ALL`]
/// order), the sandbox model and its capabilities, and the integrity policy to
/// apply to dynamic modules. Build it with [`SecurityPosture::detect`] for a
/// best-effort snapshot or with a [`PostureBuilder`] when a backend (or a test)
/// already knows the real values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SecurityPosture {
    os: Os,
    mitigations: [MitigationStatus; MITIGATION_COUNT],
    sandbox: SandboxModel,
    integrity: IntegrityPolicy,
    probed: bool,
}

impl SecurityPosture {
    /// A best-effort posture for the current target.
    ///
    /// Portable and allocation-free: it uses the compile-time target OS, the
    /// per-platform mitigation baseline, and the conservative sandbox model.
    /// [`SecurityPosture::is_probed`] is `false` because no live verification
    /// has occurred (see the module honesty boundary).
    #[must_use]
    pub fn detect() -> Self {
        let os = current_target_os();
        Self::baseline(os)
    }

    /// The policy-baseline posture for an explicit `os` (useful for tests and
    /// for reasoning about a target other than the host).
    #[must_use]
    pub fn baseline(os: Os) -> Self {
        let mut mitigations = [MitigationStatus::Unknown; MITIGATION_COUNT];
        let mut i = 0;
        while i < MITIGATION_COUNT {
            mitigations[i] = Mitigation::ALL[i].platform_default(os);
            i += 1;
        }
        Self {
            os,
            mitigations,
            sandbox: SandboxModel::default_for(os),
            integrity: IntegrityPolicy::default(),
            probed: false,
        }
    }

    /// Start building a posture from the `os` baseline, to be refined by a
    /// backend that has real probe results.
    #[must_use]
    pub fn builder(os: Os) -> PostureBuilder {
        PostureBuilder {
            posture: Self::baseline(os),
        }
    }

    /// The OS family this posture describes.
    #[must_use]
    pub const fn os(&self) -> Os {
        self.os
    }

    /// Whether the richer fields reflect a real live probe (`true`) or the
    /// platform policy baseline (`false`). See the module honesty boundary.
    #[must_use]
    pub const fn is_probed(&self) -> bool {
        self.probed
    }

    /// The status of a single mitigation.
    #[must_use]
    pub fn mitigation(&self, m: Mitigation) -> MitigationStatus {
        let idx = Mitigation::ALL
            .iter()
            .position(|&c| c == m)
            .expect("every Mitigation is in Mitigation::ALL");
        self.mitigations[idx]
    }

    /// Iterate every `(mitigation, status)` pair in stable order.
    pub fn mitigations(&self) -> impl Iterator<Item = (Mitigation, MitigationStatus)> + '_ {
        Mitigation::ALL
            .iter()
            .copied()
            .zip(self.mitigations.iter().copied())
    }

    /// The sandbox model.
    #[must_use]
    pub const fn sandbox(&self) -> SandboxModel {
        self.sandbox
    }

    /// The capabilities implied by the sandbox model.
    #[must_use]
    pub const fn sandbox_capabilities(&self) -> SandboxCapabilities {
        self.sandbox.capabilities()
    }

    /// The module-integrity policy.
    #[must_use]
    pub const fn integrity_policy(&self) -> IntegrityPolicy {
        self.integrity
    }

    /// Whether any modeled mitigation is a definite *weakness* (known off on a
    /// platform where it applies). `Unknown`/`NotApplicable` do not count.
    #[must_use]
    pub fn has_weakness(&self) -> bool {
        self.mitigations.iter().any(|s| s.is_weakness())
    }

    /// Decide whether a dynamic module with the given signing `status` may load
    /// under this posture's integrity policy.
    #[must_use]
    pub const fn module_load_decision(&self, status: SigningStatus) -> LoadDecision {
        self.integrity.decision(status)
    }
}

/// A builder that refines a baseline [`SecurityPosture`] with real probe
/// results. Once a backend sets any field from a live read it should call
/// [`PostureBuilder::mark_probed`] so [`SecurityPosture::is_probed`] is honest.
#[derive(Clone, Copy, Debug)]
pub struct PostureBuilder {
    posture: SecurityPosture,
}

impl PostureBuilder {
    /// Override one mitigation's status with a probed value.
    #[must_use]
    pub fn mitigation(mut self, m: Mitigation, status: MitigationStatus) -> Self {
        let idx = Mitigation::ALL
            .iter()
            .position(|&c| c == m)
            .expect("every Mitigation is in Mitigation::ALL");
        self.posture.mitigations[idx] = status;
        self
    }

    /// Set the (probed) sandbox model.
    #[must_use]
    pub const fn sandbox(mut self, model: SandboxModel) -> Self {
        self.posture.sandbox = model;
        self
    }

    /// Set the integrity policy to apply.
    #[must_use]
    pub const fn integrity_policy(mut self, policy: IntegrityPolicy) -> Self {
        self.posture.integrity = policy;
        self
    }

    /// Mark the posture as reflecting a real live probe.
    #[must_use]
    pub const fn mark_probed(mut self) -> Self {
        self.posture.probed = true;
        self
    }

    /// Finish and return the posture.
    #[must_use]
    pub const fn build(self) -> SecurityPosture {
        self.posture
    }
}

/// The OS family of the current compile target, mirroring the mapping used by
/// [`crate::platform::Platform`].
#[must_use]
const fn current_target_os() -> Os {
    #[cfg(target_os = "windows")]
    {
        Os::Windows
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        Os::Apple
    }
    #[cfg(target_os = "android")]
    {
        Os::Android
    }
    #[cfg(all(target_os = "linux", not(target_os = "android")))]
    {
        Os::Linux
    }
    #[cfg(target_arch = "wasm32")]
    {
        Os::Web
    }
    #[cfg(not(any(
        target_os = "windows",
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "linux",
        target_arch = "wasm32"
    )))]
    {
        Os::Unknown
    }
}
