//! Manifest validation against an expected `ABI` binding.
//!
//! A [`ShaderPackageManifest`] is only usable if the `ABI` it was compiled
//! against matches what the runtime expects. Validation compares, in order:
//!
//! 1. `AbiVersion` compatibility (semantic-version rule),
//! 2. exact `AbiHash` equality (byte-identical `ABI` layout),
//! 3. the entry-point name when the expectation pins one, and
//! 4. that the package declares at least one permutation.
//!
//! The first failing check produces a specific [`RejectionReason`]; otherwise
//! the manifest is [`ValidationOutcome::Accepted`].

use alloc::string::String;

use crate::abi::{AbiHash, AbiVersion};

use super::ShaderPackageManifest;

/// The `ABI` binding and constraints a manifest must satisfy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ManifestExpectation {
    /// Minimum compatible render `ABI` version.
    pub abi_version: AbiVersion,
    /// Exact expected `ABI` content hash.
    pub abi_hash: AbiHash,
    /// Optional required entry-point name; `None` accepts any.
    pub required_entry_point: Option<String>,
}

impl ManifestExpectation {
    /// Builds an expectation that pins only the `ABI` version and hash.
    #[must_use]
    pub const fn new(abi_version: AbiVersion, abi_hash: AbiHash) -> Self {
        Self {
            abi_version,
            abi_hash,
            required_entry_point: None,
        }
    }

    /// Returns a copy that additionally pins the entry-point name.
    #[must_use]
    pub fn with_entry_point(mut self, entry_point: impl Into<String>) -> Self {
        self.required_entry_point = Some(entry_point.into());
        self
    }
}

/// A specific reason a manifest failed validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RejectionReason {
    /// The manifest's `ABI` version is not compatible with the expectation.
    AbiVersionIncompatible {
        /// Version required by the expectation.
        expected: AbiVersion,
        /// Version declared by the manifest.
        found: AbiVersion,
    },
    /// The manifest's `ABI` hash differs from the expectation.
    AbiHashMismatch {
        /// Hash required by the expectation.
        expected: AbiHash,
        /// Hash declared by the manifest.
        found: AbiHash,
    },
    /// A required entry point was pinned but the manifest declares another.
    EntryPointMismatch {
        /// Entry point required by the expectation.
        expected: String,
        /// Entry point declared by the manifest.
        found: String,
    },
    /// The manifest declares zero permutations, so nothing can be selected.
    NoPermutations,
}

/// Result of validating a manifest against a [`ManifestExpectation`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ValidationOutcome {
    /// The manifest satisfied every check.
    Accepted,
    /// The manifest was rejected for the given reason.
    Rejected(RejectionReason),
}

impl ValidationOutcome {
    /// Returns `true` when the manifest passed validation.
    #[must_use]
    pub const fn is_accepted(&self) -> bool {
        matches!(self, Self::Accepted)
    }

    /// Borrows the rejection reason, if any.
    #[must_use]
    pub const fn rejection_reason(&self) -> Option<&RejectionReason> {
        match self {
            Self::Accepted => None,
            Self::Rejected(reason) => Some(reason),
        }
    }
}

impl ShaderPackageManifest {
    /// Validates this manifest against `expectation`.
    ///
    /// Checks run in a fixed order and short-circuit on the first failure so
    /// the returned [`RejectionReason`] is deterministic.
    #[must_use]
    pub fn validate(&self, expectation: &ManifestExpectation) -> ValidationOutcome {
        if !self.abi_version.is_compatible_with(expectation.abi_version) {
            return ValidationOutcome::Rejected(RejectionReason::AbiVersionIncompatible {
                expected: expectation.abi_version,
                found: self.abi_version,
            });
        }
        if self.abi_hash != expectation.abi_hash {
            return ValidationOutcome::Rejected(RejectionReason::AbiHashMismatch {
                expected: expectation.abi_hash,
                found: self.abi_hash,
            });
        }
        if let Some(required) = &expectation.required_entry_point
            && required != &self.entry_point
        {
            return ValidationOutcome::Rejected(RejectionReason::EntryPointMismatch {
                expected: required.clone(),
                found: self.entry_point.clone(),
            });
        }
        if self.permutation_count == 0 {
            return ValidationOutcome::Rejected(RejectionReason::NoPermutations);
        }
        ValidationOutcome::Accepted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shader_package::ShaderPackageId;
    use alloc::string::ToString;

    fn manifest(version: AbiVersion, hash: AbiHash, permutations: u32) -> ShaderPackageManifest {
        ShaderPackageManifest {
            id: ShaderPackageId::new("gbuffer"),
            source_revision: "rev-1".to_string(),
            entry_point: "main".to_string(),
            abi_version: version,
            abi_hash: hash,
            permutation_count: permutations,
        }
    }

    #[test]
    fn accepts_compatible_manifest() {
        let hash = AbiHash::from_bytes(b"layout");
        let m = manifest(AbiVersion::from_parts(2, 5), hash, 4);
        let expect = ManifestExpectation::new(AbiVersion::from_parts(2, 3), hash);
        assert_eq!(m.validate(&expect), ValidationOutcome::Accepted);
    }

    #[test]
    fn rejects_incompatible_version() {
        let hash = AbiHash::from_bytes(b"layout");
        let m = manifest(AbiVersion::from_parts(1, 9), hash, 4);
        let expect = ManifestExpectation::new(AbiVersion::from_parts(2, 0), hash);
        let outcome = m.validate(&expect);
        assert_eq!(
            outcome.rejection_reason(),
            Some(&RejectionReason::AbiVersionIncompatible {
                expected: AbiVersion::from_parts(2, 0),
                found: AbiVersion::from_parts(1, 9),
            })
        );
    }

    #[test]
    fn rejects_hash_mismatch() {
        let m = manifest(AbiVersion::from_parts(2, 3), AbiHash::from_bytes(b"a"), 4);
        let expect =
            ManifestExpectation::new(AbiVersion::from_parts(2, 0), AbiHash::from_bytes(b"b"));
        assert!(matches!(
            m.validate(&expect),
            ValidationOutcome::Rejected(RejectionReason::AbiHashMismatch { .. })
        ));
    }

    #[test]
    fn rejects_entry_point_mismatch() {
        let hash = AbiHash::from_bytes(b"layout");
        let m = manifest(AbiVersion::from_parts(2, 3), hash, 4);
        let expect = ManifestExpectation::new(AbiVersion::from_parts(2, 0), hash)
            .with_entry_point("vs_main");
        assert_eq!(
            m.validate(&expect),
            ValidationOutcome::Rejected(RejectionReason::EntryPointMismatch {
                expected: "vs_main".to_string(),
                found: "main".to_string(),
            })
        );
    }

    #[test]
    fn rejects_empty_permutations() {
        let hash = AbiHash::from_bytes(b"layout");
        let m = manifest(AbiVersion::from_parts(2, 3), hash, 0);
        let expect = ManifestExpectation::new(AbiVersion::from_parts(2, 0), hash);
        assert_eq!(
            m.validate(&expect),
            ValidationOutcome::Rejected(RejectionReason::NoPermutations)
        );
    }

    #[test]
    fn version_check_precedes_hash_check() {
        // Both version and hash are wrong; version must be reported first.
        let m = manifest(AbiVersion::from_parts(1, 0), AbiHash::from_bytes(b"a"), 4);
        let expect =
            ManifestExpectation::new(AbiVersion::from_parts(2, 0), AbiHash::from_bytes(b"b"));
        assert!(matches!(
            m.validate(&expect),
            ValidationOutcome::Rejected(RejectionReason::AbiVersionIncompatible { .. })
        ));
    }
}
