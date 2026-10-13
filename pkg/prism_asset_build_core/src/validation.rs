//! Release-quality validation for immutable artifact manifests and CAS blobs.
//!
//! This module is deliberately independent from build scheduling. It can be
//! used by a build worker, a CI job, or a package/release step after an
//! artifact set has been produced. The gate validates both semantic invariants
//! (descriptor identity and dependency topology) and physical invariants
//! (manifest persistence and CAS byte integrity).

use crate::cas::{CasError, ContentAddressedStore};
use crate::transaction::{PublishError, PublishedManifest, RevisionStore};
use crate::types::{ArtifactDescriptor, ArtifactKey, BuildRevision, ContentHash, Manifest};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

/// Resource and topology limits applied by [`ArtifactQualityGate`].
///
/// Limits are part of validation rather than an implementation detail: a
/// malformed or hostile manifest must not make a CI or packaging process
/// allocate unbounded memory or spend unbounded time traversing dependencies.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidationPolicy {
    /// Maximum number of logical assets in one manifest.
    pub max_assets: usize,
    /// Maximum number of direct dependencies on one artifact.
    pub max_dependencies_per_artifact: usize,
    /// Maximum number of dependency edges in one manifest.
    pub max_dependency_edges: usize,
    /// Optional maximum size of an individual CAS blob.
    pub max_blob_bytes: Option<u64>,
    /// Optional maximum declared size of one artifact.
    ///
    /// This limit is checked during manifest preflight, before any CAS read.
    pub max_artifact_bytes: Option<u64>,
    /// Optional maximum total size of unique artifact bytes in the manifest.
    ///
    /// Repeated references to the same content hash count once. The declared
    /// sizes are checked before CAS reads and the verified CAS sizes are
    /// checked again after each unique read.
    pub max_total_manifest_bytes: Option<u64>,
    /// Require every descriptor dependency to be present in this manifest.
    pub require_dependency_closure: bool,
}

impl Default for ValidationPolicy {
    fn default() -> Self {
        Self {
            max_assets: 1_000_000,
            max_dependencies_per_artifact: 1_000_000,
            max_dependency_edges: 10_000_000,
            max_blob_bytes: None,
            max_artifact_bytes: None,
            max_total_manifest_bytes: None,
            require_dependency_closure: true,
        }
    }
}

/// Summary emitted after a complete quality-gate pass.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidationReport {
    /// Revision validated from the manifest.
    pub revision: BuildRevision,
    /// Number of logical manifest entries checked.
    pub checked_assets: usize,
    /// Number of unique artifact descriptors checked.
    pub checked_artifacts: usize,
    /// Number of unique CAS blobs read and verified.
    pub checked_blobs: usize,
    /// Total bytes represented by unique checked blobs.
    pub checked_bytes: u64,
    /// Number of dependency edges checked.
    pub checked_dependency_edges: usize,
}

/// A failed artifact quality gate with enough context for CI diagnostics.
#[derive(Debug)]
pub enum ValidationError {
    /// The manifest exceeds a configured entry limit.
    ManifestTooLarge {
        /// Number of entries observed.
        actual: usize,
        /// Maximum number of entries allowed.
        max: usize,
    },
    /// The manifest has more dependency edges than the configured limit.
    DependencyEdgesTooLarge {
        /// Number of edges observed.
        actual: usize,
        /// Maximum number of edges allowed.
        max: usize,
    },
    /// A descriptor failed a structural invariant.
    InvalidDescriptor {
        /// Logical asset containing the invalid descriptor.
        asset: u128,
        /// Invalid artifact key.
        key: ArtifactKey,
        /// Stable reason suitable for diagnostics.
        reason: &'static str,
    },
    /// Two logical assets use one semantic key with different descriptors.
    ConflictingDescriptor {
        /// Conflicting semantic key.
        key: ArtifactKey,
    },
    /// A dependency is missing from a closure-required manifest.
    MissingDependency {
        /// Logical asset containing the missing edge.
        asset: u128,
        /// Artifact declaring the missing edge.
        key: ArtifactKey,
        /// Referenced artifact that is absent.
        dependency: ArtifactKey,
    },
    /// An artifact directly depends on itself.
    SelfDependency {
        /// Logical asset containing the invalid descriptor.
        asset: u128,
        /// Self-referencing artifact key.
        key: ArtifactKey,
    },
    /// The same dependency appears more than once on one descriptor.
    DuplicateDependency {
        /// Logical asset containing the invalid descriptor.
        asset: u128,
        /// Artifact declaring the duplicate edge.
        key: ArtifactKey,
        /// Repeated dependency key.
        dependency: ArtifactKey,
    },
    /// The known artifact dependency graph contains a cycle.
    DependencyCycle {
        /// Keys remaining in the cyclic portion of the graph.
        artifacts: Vec<ArtifactKey>,
    },
    /// A blob is larger than the configured policy permits.
    BlobTooLarge {
        /// Logical asset containing the oversized blob.
        asset: u128,
        /// Artifact key associated with the blob.
        key: ArtifactKey,
        /// Declared blob size.
        actual: u64,
        /// Maximum permitted size.
        max: u64,
    },
    /// An artifact exceeds the configured per-artifact byte budget.
    ArtifactBytesTooLarge {
        /// Logical asset containing the oversized artifact.
        asset: u128,
        /// Artifact key associated with the oversized artifact.
        key: ArtifactKey,
        /// Declared artifact size.
        actual: u64,
        /// Maximum permitted artifact size.
        max: u64,
    },
    /// The unique artifact bytes in a manifest exceed the configured budget.
    ManifestBytesTooLarge {
        /// Total unique artifact bytes observed or declared.
        actual: u64,
        /// Maximum permitted total size.
        max: u64,
    },
    /// Unique-byte accounting overflowed the representable counter.
    ManifestBytesOverflow,
    /// CAS could not read or verify an artifact blob.
    Cas {
        /// Logical asset containing the unreadable blob.
        asset: u128,
        /// Artifact key associated with the blob.
        key: ArtifactKey,
        /// Underlying CAS failure.
        source: CasError,
    },
    /// The bytes in CAS have a different length from their descriptor.
    BlobLengthMismatch {
        /// Logical asset containing the mismatched blob.
        asset: u128,
        /// Artifact key associated with the blob.
        key: ArtifactKey,
        /// Length recorded by the descriptor.
        expected: u64,
        /// Length observed in CAS.
        actual: u64,
    },
    /// The persisted publication hash differs from the supplied publication.
    ManifestHashMismatch {
        /// Hash stored in the publication object.
        expected: ContentHash,
        /// Hash recomputed from canonical manifest bytes.
        actual: ContentHash,
    },
    /// The revision file is not the canonical encoding of the supplied manifest.
    PersistedManifestMismatch {
        /// Revision whose persisted bytes differ.
        revision: BuildRevision,
    },
    /// `CURRENT` does not point at the publication being validated.
    CurrentRevisionMismatch {
        /// Revision expected by the publication.
        expected: BuildRevision,
        /// Revision currently visible through `CURRENT`.
        actual: Option<BuildRevision>,
    },
    /// The revision store could not be read.
    Store(PublishError),
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ManifestTooLarge { actual, max } => {
                write!(f, "manifest has {actual} assets, maximum is {max}")
            }
            Self::DependencyEdgesTooLarge { actual, max } => {
                write!(
                    f,
                    "manifest has {actual} dependency edges, maximum is {max}"
                )
            }
            Self::InvalidDescriptor { asset, key, reason } => {
                write!(
                    f,
                    "asset {asset} descriptor {} is invalid: {reason}",
                    key.to_hex()
                )
            }
            Self::ConflictingDescriptor { key } => write!(
                f,
                "artifact key {} maps to conflicting descriptors",
                key.to_hex()
            ),
            Self::MissingDependency {
                asset,
                key,
                dependency,
            } => write!(
                f,
                "asset {asset} artifact {} is missing dependency {}",
                key.to_hex(),
                dependency.to_hex()
            ),
            Self::SelfDependency { asset, key } => write!(
                f,
                "asset {asset} artifact {} depends on itself",
                key.to_hex()
            ),
            Self::DuplicateDependency {
                asset,
                key,
                dependency,
            } => write!(
                f,
                "asset {asset} artifact {} repeats dependency {}",
                key.to_hex(),
                dependency.to_hex()
            ),
            Self::DependencyCycle { artifacts } => {
                write!(f, "artifact dependency graph contains a cycle involving ")?;
                for (index, key) in artifacts.iter().enumerate() {
                    if index != 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}", key.to_hex())?;
                }
                Ok(())
            }
            Self::BlobTooLarge {
                asset,
                key,
                actual,
                max,
            } => write!(
                f,
                "asset {asset} artifact {} is {actual} bytes, maximum is {max}",
                key.to_hex()
            ),
            Self::ArtifactBytesTooLarge {
                asset,
                key,
                actual,
                max,
            } => write!(
                f,
                "asset {asset} artifact {} is {actual} bytes, artifact maximum is {max}",
                key.to_hex()
            ),
            Self::ManifestBytesTooLarge { actual, max } => write!(
                f,
                "manifest contains {actual} unique artifact bytes, maximum is {max}"
            ),
            Self::ManifestBytesOverflow => {
                f.write_str("manifest unique artifact byte accounting overflowed")
            }
            Self::Cas { asset, key, source } => write!(
                f,
                "asset {asset} artifact {} failed CAS validation: {source}",
                key.to_hex()
            ),
            Self::BlobLengthMismatch {
                asset,
                key,
                expected,
                actual,
            } => write!(
                f,
                "asset {asset} artifact {} declares {expected} bytes, CAS has {actual}",
                key.to_hex()
            ),
            Self::ManifestHashMismatch { expected, actual } => write!(
                f,
                "published manifest hash mismatch: expected {}, got {}",
                expected.to_hex(),
                actual.to_hex()
            ),
            Self::PersistedManifestMismatch { revision } => write!(
                f,
                "persisted manifest for revision {} is not canonical",
                revision.0
            ),
            Self::CurrentRevisionMismatch { expected, actual } => write!(
                f,
                "CURRENT points to {:?}, expected revision {}",
                actual, expected.0
            ),
            Self::Store(error) => write!(f, "revision store validation failed: {error}"),
        }
    }
}

impl std::error::Error for ValidationError {}

/// Independent release gate for manifest and CAS integrity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactQualityGate {
    policy: ValidationPolicy,
}

impl Default for ArtifactQualityGate {
    fn default() -> Self {
        Self::new(ValidationPolicy::default())
    }
}

impl ArtifactQualityGate {
    /// Creates a gate with explicit resource and closure policy.
    pub fn new(policy: ValidationPolicy) -> Self {
        Self { policy }
    }

    /// Returns the policy used by this gate.
    pub fn policy(&self) -> &ValidationPolicy {
        &self.policy
    }

    /// Validates a manifest against the actual immutable bytes in `cas`.
    pub fn validate_manifest(
        &self,
        manifest: &Manifest,
        cas: &ContentAddressedStore,
    ) -> Result<ValidationReport, ValidationError> {
        if manifest.entries.len() > self.policy.max_assets {
            return Err(ValidationError::ManifestTooLarge {
                actual: manifest.entries.len(),
                max: self.policy.max_assets,
            });
        }

        let mut descriptors = BTreeMap::<ArtifactKey, ArtifactDescriptor>::new();
        let mut dependency_edges = 0usize;
        for (asset, descriptor) in &manifest.entries {
            let key = descriptor.key;
            descriptor
                .validate()
                .map_err(|reason| ValidationError::InvalidDescriptor {
                    asset: asset.0,
                    key,
                    reason,
                })?;
            if descriptor.dependencies.len() > self.policy.max_dependencies_per_artifact {
                return Err(ValidationError::InvalidDescriptor {
                    asset: asset.0,
                    key,
                    reason: "dependency count exceeds policy",
                });
            }
            dependency_edges = dependency_edges
                .checked_add(descriptor.dependencies.len())
                .ok_or(ValidationError::DependencyEdgesTooLarge {
                    actual: usize::MAX,
                    max: self.policy.max_dependency_edges,
                })?;
            if dependency_edges > self.policy.max_dependency_edges {
                return Err(ValidationError::DependencyEdgesTooLarge {
                    actual: dependency_edges,
                    max: self.policy.max_dependency_edges,
                });
            }

            let mut dependencies = BTreeSet::new();
            for dependency in &descriptor.dependencies {
                if *dependency == key {
                    return Err(ValidationError::SelfDependency {
                        asset: asset.0,
                        key,
                    });
                }
                if !dependencies.insert(*dependency) {
                    return Err(ValidationError::DuplicateDependency {
                        asset: asset.0,
                        key,
                        dependency: *dependency,
                    });
                }
            }

            if let Some(existing) = descriptors.insert(key, descriptor.clone())
                && existing != *descriptor
            {
                return Err(ValidationError::ConflictingDescriptor { key });
            }
        }

        let known_keys: BTreeSet<_> = descriptors.keys().copied().collect();
        for (asset, descriptor) in &manifest.entries {
            for dependency in &descriptor.dependencies {
                if self.policy.require_dependency_closure && !known_keys.contains(dependency) {
                    return Err(ValidationError::MissingDependency {
                        asset: asset.0,
                        key: descriptor.key,
                        dependency: *dependency,
                    });
                }
            }
        }
        validate_acyclic(&descriptors)?;

        // Preflight all declared byte budgets before touching CAS. The
        // manifest is BTreeMap-backed, so the first diagnostic and the
        // aggregate accounting order are stable across runs.
        let mut declared_unique_bytes = BTreeMap::<ContentHash, u64>::new();
        for (asset, descriptor) in &manifest.entries {
            if let Some(max) = self.policy.max_artifact_bytes
                && descriptor.byte_len > max
            {
                return Err(ValidationError::ArtifactBytesTooLarge {
                    asset: asset.0,
                    key: descriptor.key,
                    actual: descriptor.byte_len,
                    max,
                });
            }
            if let Some(max) = self.policy.max_blob_bytes
                && descriptor.byte_len > max
            {
                return Err(ValidationError::BlobTooLarge {
                    asset: asset.0,
                    key: descriptor.key,
                    actual: descriptor.byte_len,
                    max,
                });
            }

            // A malformed manifest may use one content hash with conflicting
            // declared lengths. Keep the largest declaration for conservative
            // preflight accounting; CAS verification still reports the exact
            // descriptor mismatch after the first read.
            let entry = declared_unique_bytes
                .entry(descriptor.content_hash)
                .or_insert(0);
            *entry = (*entry).max(descriptor.byte_len);
        }
        let declared_total = declared_unique_bytes
            .values()
            .try_fold(0u64, |total, length| total.checked_add(*length));
        let declared_total = declared_total.ok_or(ValidationError::ManifestBytesOverflow)?;
        if let Some(max) = self.policy.max_total_manifest_bytes
            && declared_total > max
        {
            return Err(ValidationError::ManifestBytesTooLarge {
                actual: declared_total,
                max,
            });
        }

        let mut checked_bytes = 0u64;
        let mut checked_blobs = BTreeSet::new();
        let mut blob_lengths = BTreeMap::<ContentHash, u64>::new();
        for (asset, descriptor) in &manifest.entries {
            let actual = if let Some(actual) = blob_lengths.get(&descriptor.content_hash) {
                *actual
            } else {
                let bytes =
                    cas.read(descriptor.content_hash)
                        .map_err(|source| ValidationError::Cas {
                            asset: asset.0,
                            key: descriptor.key,
                            source,
                        })?;
                let actual = bytes.len() as u64;
                blob_lengths.insert(descriptor.content_hash, actual);
                checked_blobs.insert(descriptor.content_hash);
                if let Some(max) = self.policy.max_artifact_bytes
                    && actual > max
                {
                    return Err(ValidationError::ArtifactBytesTooLarge {
                        asset: asset.0,
                        key: descriptor.key,
                        actual,
                        max,
                    });
                }
                if let Some(max) = self.policy.max_blob_bytes
                    && actual > max
                {
                    return Err(ValidationError::BlobTooLarge {
                        asset: asset.0,
                        key: descriptor.key,
                        actual,
                        max,
                    });
                }
                checked_bytes = checked_bytes
                    .checked_add(actual)
                    .ok_or(ValidationError::ManifestBytesOverflow)?;
                if let Some(max) = self.policy.max_total_manifest_bytes
                    && checked_bytes > max
                {
                    return Err(ValidationError::ManifestBytesTooLarge {
                        actual: checked_bytes,
                        max,
                    });
                }
                actual
            };
            if actual != descriptor.byte_len {
                return Err(ValidationError::BlobLengthMismatch {
                    asset: asset.0,
                    key: descriptor.key,
                    expected: descriptor.byte_len,
                    actual,
                });
            }
        }

        Ok(ValidationReport {
            revision: manifest.revision,
            checked_assets: manifest.entries.len(),
            checked_artifacts: descriptors.len(),
            checked_blobs: checked_blobs.len(),
            checked_bytes,
            checked_dependency_edges: dependency_edges,
        })
    }

    /// Validates a publication's self-reported manifest hash and its artifacts.
    pub fn validate_published(
        &self,
        published: &PublishedManifest,
        cas: &ContentAddressedStore,
    ) -> Result<ValidationReport, ValidationError> {
        let actual = ContentHash::digest(&published.manifest.canonical_bytes());
        if actual != published.manifest_hash {
            return Err(ValidationError::ManifestHashMismatch {
                expected: published.manifest_hash,
                actual,
            });
        }
        self.validate_manifest(&published.manifest, cas)
    }

    /// Validates a publication and proves that its persisted revision is the
    /// revision currently visible through the store's atomic `CURRENT` pointer.
    pub fn validate_published_revision(
        &self,
        store: &RevisionStore,
        published: &PublishedManifest,
    ) -> Result<ValidationReport, ValidationError> {
        let expected = published.manifest.revision;
        let actual = store.current_revision().map_err(ValidationError::Store)?;
        if actual != Some(expected) {
            return Err(ValidationError::CurrentRevisionMismatch { expected, actual });
        }
        let persisted = store
            .manifest_bytes(expected)
            .map_err(ValidationError::Store)?;
        let canonical = published.manifest.canonical_bytes();
        if persisted != canonical {
            return Err(ValidationError::PersistedManifestMismatch { revision: expected });
        }
        self.validate_published(published, store.cas())
    }
}

fn validate_acyclic(
    descriptors: &BTreeMap<ArtifactKey, ArtifactDescriptor>,
) -> Result<(), ValidationError> {
    let mut remaining = BTreeMap::<ArtifactKey, usize>::new();
    let mut dependents = BTreeMap::<ArtifactKey, Vec<ArtifactKey>>::new();
    for (key, descriptor) in descriptors {
        let known_dependencies = descriptor
            .dependencies
            .iter()
            .filter(|dependency| descriptors.contains_key(dependency))
            .copied()
            .collect::<Vec<_>>();
        remaining.insert(*key, known_dependencies.len());
        for dependency in known_dependencies {
            dependents.entry(dependency).or_default().push(*key);
        }
    }

    let mut ready = remaining
        .iter()
        .filter_map(|(key, count)| (*count == 0).then_some(*key))
        .collect::<VecDeque<_>>();
    let mut processed = 0usize;
    while let Some(key) = ready.pop_front() {
        processed += 1;
        if let Some(users) = dependents.get(&key) {
            for user in users {
                let count = remaining
                    .get_mut(user)
                    .expect("dependent must be in the descriptor map");
                *count -= 1;
                if *count == 0 {
                    ready.push_back(*user);
                }
            }
        }
    }

    if processed != descriptors.len() {
        let artifacts = remaining
            .into_iter()
            .filter_map(|(key, count)| (count != 0).then_some(key))
            .collect();
        return Err(ValidationError::DependencyCycle { artifacts });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::RevisionStore;
    use crate::types::{AssetId, ManifestEntry};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "prism-artifact-validation-{label}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock must be after epoch")
                .as_nanos()
        ))
    }

    fn descriptor(key: u8, bytes: &[u8], dependencies: Vec<ArtifactKey>) -> ArtifactDescriptor {
        ArtifactDescriptor::for_bytes(ArtifactKey([key; 32]), 7, 1, 16, dependencies, bytes)
    }

    fn manifest(entries: Vec<(u128, ArtifactDescriptor)>) -> Manifest {
        Manifest::new(
            BuildRevision(3),
            entries
                .into_iter()
                .map(|(asset, descriptor)| ManifestEntry {
                    asset: AssetId(asset),
                    descriptor,
                }),
        )
        .unwrap()
    }

    #[test]
    fn validates_closure_and_deduplicates_shared_blob_reads() {
        let root = temp_dir("valid");
        let cas = ContentAddressedStore::open(root.join("cas")).unwrap();
        let shared = descriptor(1, b"shared", vec![]);
        let output = descriptor(2, b"output", vec![shared.key]);
        cas.put(b"shared").unwrap();
        cas.put(b"output").unwrap();

        let report = ArtifactQualityGate::default()
            .validate_manifest(&manifest(vec![(1, shared), (2, output)]), &cas)
            .unwrap();
        assert_eq!(report.checked_assets, 2);
        assert_eq!(report.checked_artifacts, 2);
        assert_eq!(report.checked_blobs, 2);
        assert_eq!(report.checked_bytes, 12);
        assert_eq!(report.checked_dependency_edges, 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_missing_dependency_before_reading_cas() {
        let root = temp_dir("missing-dependency");
        let cas = ContentAddressedStore::open(root.join("cas")).unwrap();
        let output = descriptor(2, b"output", vec![ArtifactKey([1; 32])]);
        let error = ArtifactQualityGate::default()
            .validate_manifest(&manifest(vec![(2, output)]), &cas)
            .unwrap_err();
        assert!(matches!(error, ValidationError::MissingDependency { .. }));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn default_policy_keeps_byte_budgets_unbounded() {
        let policy = ValidationPolicy::default();
        assert_eq!(policy.max_artifact_bytes, None);
        assert_eq!(policy.max_total_manifest_bytes, None);
    }

    #[test]
    fn rejects_artifact_budget_before_reading_cas() {
        let root = temp_dir("artifact-budget");
        let cas = ContentAddressedStore::open(root.join("cas")).unwrap();
        let artifact = descriptor(1, b"payload", vec![]);
        let mut policy = ValidationPolicy::default();
        policy.max_artifact_bytes = Some(3);

        let error = ArtifactQualityGate::new(policy)
            .validate_manifest(&manifest(vec![(1, artifact)]), &cas)
            .unwrap_err();
        assert!(matches!(
            error,
            ValidationError::ArtifactBytesTooLarge {
                actual: 7,
                max: 3,
                ..
            }
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_total_manifest_budget_before_reading_cas() {
        let root = temp_dir("manifest-budget");
        let cas = ContentAddressedStore::open(root.join("cas")).unwrap();
        let first = descriptor(1, b"first", vec![]);
        let second = descriptor(2, b"second", vec![]);
        let mut policy = ValidationPolicy::default();
        policy.max_total_manifest_bytes = Some(10);

        let error = ArtifactQualityGate::new(policy)
            .validate_manifest(&manifest(vec![(1, first), (2, second)]), &cas)
            .unwrap_err();
        assert!(matches!(
            error,
            ValidationError::ManifestBytesTooLarge {
                actual: 11,
                max: 10
            }
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn total_manifest_budget_counts_shared_content_once() {
        let root = temp_dir("shared-manifest-budget");
        let cas = ContentAddressedStore::open(root.join("cas")).unwrap();
        let shared = descriptor(1, b"shared", vec![]);
        let mut second = shared.clone();
        second.key = ArtifactKey([2; 32]);
        let mut policy = ValidationPolicy::default();
        policy.max_total_manifest_bytes = Some(6);
        cas.put(b"shared").unwrap();

        let report = ArtifactQualityGate::new(policy)
            .validate_manifest(&manifest(vec![(1, shared), (2, second)]), &cas)
            .unwrap();
        assert_eq!(report.checked_blobs, 1);
        assert_eq!(report.checked_bytes, 6);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn total_manifest_budget_accounting_is_deterministic() {
        let first = descriptor(1, b"first", vec![]);
        let second = descriptor(2, b"second", vec![]);
        let mut policy = ValidationPolicy::default();
        policy.max_total_manifest_bytes = Some(10);
        let gate = ArtifactQualityGate::new(policy);

        let root = temp_dir("deterministic-manifest-budget");
        let cas = ContentAddressedStore::open(root.join("cas")).unwrap();
        let forward = gate
            .validate_manifest(
                &manifest(vec![(1, first.clone()), (2, second.clone())]),
                &cas,
            )
            .unwrap_err();
        let reverse = gate
            .validate_manifest(&manifest(vec![(2, second), (1, first)]), &cas)
            .unwrap_err();
        assert!(matches!(
            forward,
            ValidationError::ManifestBytesTooLarge {
                actual: 11,
                max: 10
            }
        ));
        assert!(matches!(
            reverse,
            ValidationError::ManifestBytesTooLarge {
                actual: 11,
                max: 10
            }
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn actual_artifact_bytes_budget_rejects_after_cas_read() {
        let root = temp_dir("actual-artifact-budget");
        let cas = ContentAddressedStore::open(root.join("cas")).unwrap();
        let bytes = b"payload";
        let mut artifact = descriptor(1, bytes, vec![]);
        artifact.byte_len = 1;
        cas.put(bytes).unwrap();
        let mut policy = ValidationPolicy::default();
        policy.max_artifact_bytes = Some(3);

        let error = ArtifactQualityGate::new(policy)
            .validate_manifest(&manifest(vec![(1, artifact)]), &cas)
            .unwrap_err();
        assert!(matches!(
            error,
            ValidationError::ArtifactBytesTooLarge {
                actual: 7,
                max: 3,
                ..
            }
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn actual_unique_bytes_budget_rejects_underdeclared_blob_after_read() {
        let root = temp_dir("actual-manifest-budget");
        let cas = ContentAddressedStore::open(root.join("cas")).unwrap();
        let bytes = b"payload";
        let mut artifact = descriptor(1, bytes, vec![]);
        artifact.byte_len = 1;
        cas.put(bytes).unwrap();
        let mut policy = ValidationPolicy::default();
        policy.max_total_manifest_bytes = Some(3);

        let error = ArtifactQualityGate::new(policy)
            .validate_manifest(&manifest(vec![(1, artifact)]), &cas)
            .unwrap_err();
        assert!(matches!(
            error,
            ValidationError::ManifestBytesTooLarge { actual: 7, max: 3 }
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_conflicting_descriptors_for_one_artifact_key() {
        let root = temp_dir("conflict");
        let cas = ContentAddressedStore::open(root.join("cas")).unwrap();
        let first = descriptor(1, b"one", vec![]);
        let mut second = descriptor(1, b"two", vec![]);
        second.kind = 8;
        cas.put(b"one").unwrap();
        cas.put(b"two").unwrap();
        let error = ArtifactQualityGate::default()
            .validate_manifest(&manifest(vec![(1, first), (2, second)]), &cas)
            .unwrap_err();
        assert!(matches!(
            error,
            ValidationError::ConflictingDescriptor { .. }
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_cycles_without_recursion_or_stack_growth() {
        let root = temp_dir("cycle");
        let cas = ContentAddressedStore::open(root.join("cas")).unwrap();
        let first = descriptor(1, b"one", vec![ArtifactKey([2; 32])]);
        let second = descriptor(2, b"two", vec![ArtifactKey([1; 32])]);
        cas.put(b"one").unwrap();
        cas.put(b"two").unwrap();
        let result = std::panic::catch_unwind(|| {
            ArtifactQualityGate::default()
                .validate_manifest(&manifest(vec![(1, first), (2, second)]), &cas)
        });
        assert!(result.is_ok(), "cycle detection must be a validation error");
        assert!(matches!(
            result.unwrap().unwrap_err(),
            ValidationError::DependencyCycle { .. }
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn validates_persisted_publication_and_detects_current_mismatch() {
        let root = temp_dir("publication");
        let store = RevisionStore::open(&root).unwrap();
        let bytes = b"payload".to_vec();
        let descriptor = descriptor(1, &bytes, vec![]);
        let mut transaction = store.begin(BuildRevision(0), None);
        transaction.stage(AssetId(9), descriptor, bytes).unwrap();
        let published = transaction.commit().unwrap();
        let report = ArtifactQualityGate::default()
            .validate_published_revision(&store, &published)
            .unwrap();
        assert_eq!(report.revision, BuildRevision(0));

        let mut tampered = published.clone();
        tampered.manifest.revision = BuildRevision(1);
        assert!(matches!(
            ArtifactQualityGate::default().validate_published_revision(&store, &tampered),
            Err(ValidationError::CurrentRevisionMismatch { .. })
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_corrupt_cas_blob() {
        let root = temp_dir("corrupt");
        let cas = ContentAddressedStore::open(root.join("cas")).unwrap();
        let artifact = descriptor(1, b"payload", vec![]);
        fs::create_dir_all(cas.blob_path(artifact.content_hash).parent().unwrap()).unwrap();
        fs::write(cas.blob_path(artifact.content_hash), b"tampered").unwrap();
        let error = ArtifactQualityGate::default()
            .validate_manifest(&manifest(vec![(1, artifact)]), &cas)
            .unwrap_err();
        assert!(matches!(error, ValidationError::Cas { .. }));
        fs::remove_dir_all(root).unwrap();
    }
}
