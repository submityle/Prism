//! Code-signing status and the load-time integrity policy derived from it
//! (design §24.5, interacts with the dynamic-library layer §10).
//!
//! Rejecting unsigned or wrongly-signed modules is the foundation of an
//! anti-injection / anti-tamper story (it is defensive infrastructure, not a
//! cheat and not an attack tool). This module models a module's signing
//! [`SigningStatus`] as portable data and an [`IntegrityPolicy`] whose
//! [`IntegrityPolicy::decision`] is pure, deterministic, and fully testable
//! without touching any real signature database.
//!
//! Honesty: computing a real [`SigningStatus`] for a file requires per-OS
//! signature verification (Authenticode on Windows, `codesign`/`SecCode` on
//! Apple, distro package signatures / IMA on Linux). That live verification is
//! not wired yet; callers construct [`SigningStatus`] from a backend that does
//! the OS call, and the policy layer below is what consumes it.

/// The outcome of verifying a module's code signature.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum SigningStatus {
    /// Signed and the signature verified against a trusted authority.
    Trusted,
    /// Signed, and the signature is cryptographically valid, but the signer is
    /// not in the trust set (self-signed, unknown CA, ad-hoc).
    SignedUntrusted,
    /// A signature is present but failed verification (tampered / corrupt).
    Invalid,
    /// No signature at all.
    Unsigned,
    /// Not yet verified.
    #[default]
    Unknown,
}

impl SigningStatus {
    /// Whether the module is signed *and* trusted.
    #[must_use]
    pub const fn is_trusted(self) -> bool {
        matches!(self, SigningStatus::Trusted)
    }

    /// Whether verification reached a definitive bad verdict (tampered or
    /// outright unsigned) as opposed to merely untrusted or unknown.
    #[must_use]
    pub const fn is_definitely_bad(self) -> bool {
        matches!(self, SigningStatus::Invalid | SigningStatus::Unsigned)
    }
}

/// How strictly the loader treats module signatures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum SigningStrictness {
    /// Load anything; only record the status for telemetry.
    Observe,
    /// Reject modules that are definitively bad (tampered or unsigned) but
    /// allow signed-but-untrusted (useful for dev / third-party plugins).
    #[default]
    RejectBad,
    /// Reject anything that is not signed *and* trusted.
    RequireTrusted,
}

/// The load-time integrity policy applied to dynamic modules.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IntegrityPolicy {
    /// How strict signature checking is.
    pub strictness: SigningStrictness,
    /// Whether an [`SigningStatus::Unknown`] (unverifiable) module is allowed.
    /// Defaults to `false` under the stricter policies: if we cannot verify,
    /// we do not load.
    pub allow_unknown: bool,
}

impl Default for IntegrityPolicy {
    fn default() -> Self {
        Self {
            strictness: SigningStrictness::RejectBad,
            // Under the default `RejectBad` policy, an unknown status is still
            // conservatively allowed (dev convenience); the stricter builder
            // flips this off.
            allow_unknown: true,
        }
    }
}

/// The decision the loader should take for a module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LoadDecision {
    /// Load the module.
    Allow,
    /// Refuse to load the module, with the reason.
    Reject(RejectReason),
}

impl LoadDecision {
    /// Whether the module may be loaded.
    #[must_use]
    pub const fn is_allowed(self) -> bool {
        matches!(self, LoadDecision::Allow)
    }
}

/// Why a module load was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RejectReason {
    /// The signature was present but invalid (tampered / corrupt).
    InvalidSignature,
    /// The module was unsigned under a policy that forbids unsigned modules.
    Unsigned,
    /// The module was signed but not by a trusted authority.
    Untrusted,
    /// The signing status could not be determined under a policy that forbids
    /// unknown modules.
    Unverifiable,
}

impl IntegrityPolicy {
    /// The permissive dev policy: observe only, never reject.
    #[must_use]
    pub const fn observe_only() -> Self {
        Self {
            strictness: SigningStrictness::Observe,
            allow_unknown: true,
        }
    }

    /// The locked-down shipping policy: require signed-and-trusted modules and
    /// refuse anything unverifiable.
    #[must_use]
    pub const fn locked_down() -> Self {
        Self {
            strictness: SigningStrictness::RequireTrusted,
            allow_unknown: false,
        }
    }

    /// Decide whether a module with the given signing `status` may load.
    ///
    /// Pure and deterministic; this is the whole point of modeling signing as
    /// data — the decision can be exercised exhaustively in tests.
    #[must_use]
    pub const fn decision(&self, status: SigningStatus) -> LoadDecision {
        match self.strictness {
            SigningStrictness::Observe => LoadDecision::Allow,
            SigningStrictness::RejectBad => match status {
                SigningStatus::Invalid => LoadDecision::Reject(RejectReason::InvalidSignature),
                SigningStatus::Unsigned => LoadDecision::Reject(RejectReason::Unsigned),
                SigningStatus::Unknown if !self.allow_unknown => {
                    LoadDecision::Reject(RejectReason::Unverifiable)
                }
                SigningStatus::Trusted
                | SigningStatus::SignedUntrusted
                | SigningStatus::Unknown => LoadDecision::Allow,
            },
            SigningStrictness::RequireTrusted => match status {
                SigningStatus::Trusted => LoadDecision::Allow,
                SigningStatus::Invalid => LoadDecision::Reject(RejectReason::InvalidSignature),
                SigningStatus::Unsigned => LoadDecision::Reject(RejectReason::Unsigned),
                SigningStatus::SignedUntrusted => LoadDecision::Reject(RejectReason::Untrusted),
                SigningStatus::Unknown if self.allow_unknown => LoadDecision::Allow,
                SigningStatus::Unknown => LoadDecision::Reject(RejectReason::Unverifiable),
            },
        }
    }
}
