//! Complete, deterministic manifest of every compiled shader variant.

use alloc::{collections::BTreeSet, vec::Vec};

use super::{ArtifactDigest, PermutationError, PermutationIndex, PermutationKey, PermutationSpace};

/// One compiled variant in a shader package.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShaderPackageVariant {
    /// Dense index owned by the package's permutation space.
    pub index: PermutationIndex,
    /// Full mixed-radix key; retained to catch index/key drift.
    pub key: PermutationKey,
    /// Digest of the compiled variant code/IR.
    pub code_digest: ArtifactDigest,
}

/// The complete ordered variant table for one shader package.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ShaderPackageVariantManifest {
    variants: Vec<ShaderPackageVariant>,
}

/// Failure in the complete variant-table contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VariantManifestError {
    Permutation(PermutationError),
    CountMismatch {
        expected: u64,
        actual: usize,
    },
    IndexMismatch {
        expected: u64,
        actual: PermutationIndex,
    },
    KeyMismatch {
        index: PermutationIndex,
    },
    DuplicateKey {
        index: PermutationIndex,
    },
    ZeroCodeDigest {
        index: PermutationIndex,
    },
}

impl core::fmt::Display for VariantManifestError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Permutation(error) => write!(f, "permutation space is invalid: {error:?}"),
            Self::CountMismatch { expected, actual } => {
                write!(
                    f,
                    "variant manifest contains {actual} entries, expected {expected}"
                )
            }
            Self::IndexMismatch { expected, actual } => {
                write!(
                    f,
                    "variant manifest index is {actual:?}, expected {expected}"
                )
            }
            Self::KeyMismatch { index } => {
                write!(f, "variant manifest key does not match index {index:?}")
            }
            Self::DuplicateKey { index } => {
                write!(f, "variant manifest repeats the key at index {index:?}")
            }
            Self::ZeroCodeDigest { index } => {
                write!(f, "variant {index:?} has a zero code digest")
            }
        }
    }
}

impl std::error::Error for VariantManifestError {}

impl ShaderPackageVariantManifest {
    /// Constructs the canonical full list from the owning permutation space.
    pub fn from_digests(
        space: &PermutationSpace,
        digests: impl IntoIterator<Item = ArtifactDigest>,
    ) -> Result<Self, VariantManifestError> {
        let expected = space
            .checked_total()
            .map_err(VariantManifestError::Permutation)?;
        let digests = digests.into_iter().collect::<Vec<_>>();
        if expected != digests.len() as u64 {
            return Err(VariantManifestError::CountMismatch {
                expected,
                actual: digests.len(),
            });
        }
        let mut variants = Vec::with_capacity(digests.len());
        for (raw, digest) in digests.into_iter().enumerate() {
            let index = space
                .check_index(raw as u64)
                .map_err(VariantManifestError::Permutation)?;
            variants.push(ShaderPackageVariant {
                index,
                key: space
                    .key_from_index(raw as u64)
                    .expect("checked permutation index must decode"),
                code_digest: digest,
            });
        }
        let manifest = Self { variants };
        manifest.validate_against(space)?;
        Ok(manifest)
    }

    /// Validates that every dense variant exists exactly once and retains the
    /// key that the owning space decodes for that index.
    pub fn validate_against(&self, space: &PermutationSpace) -> Result<(), VariantManifestError> {
        let expected = space
            .checked_total()
            .map_err(VariantManifestError::Permutation)?;
        if expected != self.variants.len() as u64 {
            return Err(VariantManifestError::CountMismatch {
                expected,
                actual: self.variants.len(),
            });
        }
        let mut keys = BTreeSet::new();
        for (raw, variant) in self.variants.iter().enumerate() {
            let expected_index = raw as u64;
            if variant.index.get() != expected_index {
                return Err(VariantManifestError::IndexMismatch {
                    expected: expected_index,
                    actual: variant.index,
                });
            }
            if variant.code_digest.is_zero() {
                return Err(VariantManifestError::ZeroCodeDigest {
                    index: variant.index,
                });
            }
            if space.key_from_index(expected_index).as_ref() != Some(&variant.key) {
                return Err(VariantManifestError::KeyMismatch {
                    index: variant.index,
                });
            }
            if !keys.insert(variant.key.clone()) {
                return Err(VariantManifestError::DuplicateKey {
                    index: variant.index,
                });
            }
        }
        Ok(())
    }

    /// Validates the parts of the manifest that do not require the owning
    /// permutation axes. The package boundary calls this before an artifact
    /// can be registered; the execution table additionally calls
    /// [`Self::validate_against`] with the concrete permutation space.
    pub fn validate_structure(&self) -> Result<(), VariantManifestError> {
        let mut keys = BTreeSet::new();
        for (raw, variant) in self.variants.iter().enumerate() {
            let expected_index = raw as u64;
            if variant.index.get() != expected_index {
                return Err(VariantManifestError::IndexMismatch {
                    expected: expected_index,
                    actual: variant.index,
                });
            }
            if !keys.insert(variant.key.clone()) {
                return Err(VariantManifestError::DuplicateKey {
                    index: variant.index,
                });
            }
            if variant.code_digest.is_zero() {
                return Err(VariantManifestError::ZeroCodeDigest {
                    index: variant.index,
                });
            }
        }
        Ok(())
    }

    /// Borrows the complete ordered variant list.
    #[must_use]
    pub fn variants(&self) -> &[ShaderPackageVariant] {
        &self.variants
    }

    /// Number of entries in the complete variant list.
    #[must_use]
    pub fn len(&self) -> usize {
        self.variants.len()
    }

    /// Reports whether the manifest has no variants.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.variants.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> (PermutationSpace, ShaderPackageVariantManifest) {
        let space = PermutationSpace::new([2, 2]);
        let manifest = ShaderPackageVariantManifest::from_digests(
            &space,
            [
                ArtifactDigest([1; 32]),
                ArtifactDigest([2; 32]),
                ArtifactDigest([3; 32]),
                ArtifactDigest([4; 32]),
            ],
        )
        .unwrap();
        (space, manifest)
    }

    #[test]
    fn lists_every_variant_in_dense_order() {
        let (space, manifest) = manifest();
        assert_eq!(manifest.len(), 4);
        assert_eq!(manifest.variants()[2].index.get(), 2);
        assert_eq!(manifest.variants()[2].key.coords(), &[1, 0]);
        assert!(manifest.validate_against(&space).is_ok());
    }

    #[test]
    fn rejects_missing_duplicate_or_reordered_entries() {
        let (space, mut value) = manifest();
        value.variants.remove(1);
        assert!(matches!(
            value.validate_against(&space),
            Err(VariantManifestError::CountMismatch { .. })
        ));

        let (space, mut value) = manifest();
        value.variants.swap(0, 1);
        assert!(matches!(
            value.validate_against(&space),
            Err(VariantManifestError::IndexMismatch { .. })
        ));
    }

    #[test]
    fn rejects_key_and_code_digest_drift() {
        let (space, mut value) = manifest();
        value.variants[1].key = PermutationKey::new([1, 1]);
        assert!(matches!(
            value.validate_against(&space),
            Err(VariantManifestError::KeyMismatch { .. })
        ));

        let (space, mut value) = manifest();
        value.variants[1].code_digest = ArtifactDigest([0; 32]);
        assert!(matches!(
            value.validate_against(&space),
            Err(VariantManifestError::ZeroCodeDigest { .. })
        ));
    }
}
