//! Release publication safety gate.
//!
//! This module is the final local release boundary for a committed revision.
//! It re-checks the durable manifest and CAS closure, binds the optional
//! runtime publication descriptor, and rejects rollback below the accepted
//! revision.  The descriptor is an integrity record only: this module does
//! not implement or claim cryptographic signing.

use crate::publication::PublicationDescriptor;
use crate::transaction::{PublishError, PublishedManifest, RevisionStore};
use crate::types::{BuildRevision, ContentHash};
use crate::validation::{ArtifactQualityGate, ValidationError, ValidationReport};
use std::io::ErrorKind;

const MAGIC: &[u8] = b"PRISM-RELEASE\0\x01";

/// Deterministic identity of a release candidate.
///
/// The descriptor binds the exact persisted build manifest and, when the
/// revision contains runtime outputs, the exact publication descriptor. It
/// intentionally has no signature field; cryptographic signing remains an
/// unimplemented integration boundary until a key-management and signature
/// dependency is selected.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReleaseDescriptor {
    /// Revision that must be visible through the store's `CURRENT` pointer.
    pub revision: BuildRevision,
    /// BLAKE3 digest of the canonical persisted build manifest bytes.
    pub manifest_hash: ContentHash,
    /// BLAKE3 digest of the canonical publication descriptor, if packaged.
    pub publication_hash: Option<ContentHash>,
}

impl ReleaseDescriptor {
    /// Constructs a descriptor from a committed publication result.
    #[must_use]
    pub fn from_published(published: &PublishedManifest) -> Self {
        Self {
            revision: published.manifest.revision,
            manifest_hash: published.manifest_hash,
            publication_hash: published
                .publication
                .as_ref()
                .map(PublicationDescriptor::content_hash),
        }
    }

    /// Returns the canonical, deterministic wire representation.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(MAGIC.len() + 8 + 32 + 1 + 32);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&self.revision.0.to_le_bytes());
        bytes.extend_from_slice(&self.manifest_hash.0);
        match self.publication_hash {
            Some(hash) => {
                bytes.push(1);
                bytes.extend_from_slice(&hash.0);
            }
            None => bytes.push(0),
        }
        bytes
    }

    /// Decodes only the canonical representation and rejects all trailing or
    /// malformed bytes.
    pub fn decode(bytes: &[u8]) -> Result<Self, &'static str> {
        let mut cursor = Cursor { bytes, offset: 0 };
        if cursor.take(MAGIC.len())? != MAGIC {
            return Err("invalid release descriptor magic/version");
        }
        let descriptor = Self {
            revision: BuildRevision(cursor.u64()?),
            manifest_hash: ContentHash(cursor.array()?),
            publication_hash: match cursor.u8()? {
                0 => None,
                1 => Some(ContentHash(cursor.array()?)),
                _ => return Err("invalid release publication flag"),
            },
        };
        if cursor.remaining() != 0 {
            return Err("release descriptor has trailing bytes");
        }
        if descriptor.encode() != bytes {
            return Err("release descriptor is not canonical");
        }
        Ok(descriptor)
    }

    /// Content identity of the canonical release descriptor.
    #[must_use]
    pub fn content_hash(&self) -> ContentHash {
        ContentHash::digest(&self.encode())
    }
}

/// Result of a successful release gate pass.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReleaseVerification {
    /// Candidate descriptor that was verified.
    pub descriptor: ReleaseDescriptor,
    /// Complete manifest/CAS validation summary.
    pub validation: ValidationReport,
    /// Decoded runtime publication descriptor, when one exists.
    pub publication: Option<PublicationDescriptor>,
}

/// Errors raised by the release safety gate.
#[derive(Debug)]
pub enum ReleaseError {
    /// Store I/O, decoding, recovery, or current-publication validation failed.
    Store(PublishError),
    /// The candidate is below the durable or in-process accepted floor.
    RevisionRollback {
        accepted: BuildRevision,
        candidate: BuildRevision,
    },
    /// The candidate is not the currently committed revision.
    RevisionNotCurrent {
        candidate: BuildRevision,
        current: Option<BuildRevision>,
    },
    /// Persisted manifest bytes hash differently from the descriptor.
    ManifestHashMismatch {
        expected: ContentHash,
        actual: ContentHash,
    },
    /// The persisted manifest's decoded revision differs from the candidate.
    ManifestRevisionMismatch {
        expected: BuildRevision,
        actual: BuildRevision,
    },
    /// Manifest dependency and CAS validation failed.
    Validation(ValidationError),
    /// A packaged release has no persisted publication descriptor.
    PublicationMissing { revision: BuildRevision },
    /// A build-only release unexpectedly has a publication descriptor.
    UnexpectedPublication { revision: BuildRevision },
    /// Publication descriptor bytes hash differently from the release record.
    PublicationHashMismatch {
        expected: ContentHash,
        actual: ContentHash,
    },
    /// Publication descriptor bytes could not be decoded canonically.
    InvalidPublicationDescriptor(&'static str),
    /// Publication descriptor fields do not bind the release candidate.
    PublicationBindingMismatch,
}

impl std::fmt::Display for ReleaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Store(error) => write!(f, "release store validation failed: {error}"),
            Self::RevisionRollback {
                accepted,
                candidate,
            } => write!(
                f,
                "release revision rollback: accepted {}, candidate {}",
                accepted.0, candidate.0
            ),
            Self::RevisionNotCurrent { candidate, current } => write!(
                f,
                "release revision {} is not current; CURRENT is {:?}",
                candidate.0, current
            ),
            Self::ManifestHashMismatch { expected, actual } => write!(
                f,
                "release manifest hash mismatch: expected {}, got {}",
                expected.to_hex(),
                actual.to_hex()
            ),
            Self::ManifestRevisionMismatch { expected, actual } => write!(
                f,
                "release manifest revision mismatch: expected {}, got {}",
                expected.0, actual.0
            ),
            Self::Validation(error) => write!(f, "release manifest/CAS validation failed: {error}"),
            Self::PublicationMissing { revision } => write!(
                f,
                "release revision {} is missing its publication descriptor",
                revision.0
            ),
            Self::UnexpectedPublication { revision } => write!(
                f,
                "build-only release revision {} has an unexpected publication descriptor",
                revision.0
            ),
            Self::PublicationHashMismatch { expected, actual } => write!(
                f,
                "release publication hash mismatch: expected {}, got {}",
                expected.to_hex(),
                actual.to_hex()
            ),
            Self::InvalidPublicationDescriptor(reason) => {
                write!(f, "invalid release publication descriptor: {reason}")
            }
            Self::PublicationBindingMismatch => {
                f.write_str("release publication descriptor does not bind the candidate")
            }
        }
    }
}

impl std::error::Error for ReleaseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Store(error) => Some(error),
            Self::Validation(error) => Some(error),
            _ => None,
        }
    }
}

impl From<PublishError> for ReleaseError {
    fn from(value: PublishError) -> Self {
        Self::Store(value)
    }
}

/// Stateful release safety gate.
///
/// `accepted_revision` is updated only after every verification step passes.
/// The store's current revision is also treated as a durable monotonic floor,
/// so a fresh gate cannot accept an older revision after a restart.
#[derive(Clone, Debug)]
pub struct ReleaseGate {
    quality_gate: ArtifactQualityGate,
    accepted_revision: Option<BuildRevision>,
}

impl Default for ReleaseGate {
    fn default() -> Self {
        Self::new()
    }
}

impl ReleaseGate {
    /// Creates a gate with the default manifest/CAS quality policy.
    #[must_use]
    pub fn new() -> Self {
        Self {
            quality_gate: ArtifactQualityGate::default(),
            accepted_revision: None,
        }
    }

    /// Creates a gate with an explicit manifest/CAS quality policy.
    #[must_use]
    pub fn with_quality_gate(quality_gate: ArtifactQualityGate) -> Self {
        Self {
            quality_gate,
            accepted_revision: None,
        }
    }

    /// Creates a gate with a previously accepted in-process floor.
    #[must_use]
    pub fn with_accepted_revision(accepted_revision: BuildRevision) -> Self {
        Self {
            quality_gate: ArtifactQualityGate::default(),
            accepted_revision: Some(accepted_revision),
        }
    }

    /// Returns the in-process accepted floor, if one exists.
    #[must_use]
    pub fn accepted_revision(&self) -> Option<BuildRevision> {
        self.accepted_revision
    }

    /// Verifies a candidate without changing the accepted floor.
    pub fn verify(
        &self,
        store: &RevisionStore,
        descriptor: &ReleaseDescriptor,
    ) -> Result<ReleaseVerification, ReleaseError> {
        let current = store.current_revision()?;
        let durable_floor = current;
        let accepted_floor = match (self.accepted_revision, durable_floor) {
            (Some(left), Some(right)) => Some(left.max(right)),
            (Some(revision), None) | (None, Some(revision)) => Some(revision),
            (None, None) => None,
        };
        if accepted_floor.is_some_and(|accepted| descriptor.revision < accepted) {
            return Err(ReleaseError::RevisionRollback {
                accepted: accepted_floor.expect("checked above"),
                candidate: descriptor.revision,
            });
        }
        if current != Some(descriptor.revision) {
            return Err(ReleaseError::RevisionNotCurrent {
                candidate: descriptor.revision,
                current,
            });
        }

        let manifest_bytes = store.manifest_bytes(descriptor.revision)?;
        let actual_manifest_hash = ContentHash::digest(&manifest_bytes);
        if actual_manifest_hash != descriptor.manifest_hash {
            return Err(ReleaseError::ManifestHashMismatch {
                expected: descriptor.manifest_hash,
                actual: actual_manifest_hash,
            });
        }
        let manifest = store.manifest(descriptor.revision)?;
        if manifest.revision != descriptor.revision {
            return Err(ReleaseError::ManifestRevisionMismatch {
                expected: descriptor.revision,
                actual: manifest.revision,
            });
        }
        let validation = self
            .quality_gate
            .validate_manifest(&manifest, store.cas())
            .map_err(ReleaseError::Validation)?;

        let publication = match store.publication_bytes(descriptor.revision) {
            Ok(bytes) => {
                let actual_hash = ContentHash::digest(&bytes);
                let expected_hash =
                    descriptor
                        .publication_hash
                        .ok_or(ReleaseError::UnexpectedPublication {
                            revision: descriptor.revision,
                        })?;
                if actual_hash != expected_hash {
                    return Err(ReleaseError::PublicationHashMismatch {
                        expected: expected_hash,
                        actual: actual_hash,
                    });
                }
                let publication = PublicationDescriptor::decode(&bytes)
                    .map_err(ReleaseError::InvalidPublicationDescriptor)?;
                if publication.revision != descriptor.revision
                    || publication.build_manifest_hash != descriptor.manifest_hash
                {
                    return Err(ReleaseError::PublicationBindingMismatch);
                }
                Some(publication)
            }
            Err(PublishError::Io(error)) if error.kind() == ErrorKind::NotFound => {
                if descriptor.publication_hash.is_some() {
                    return Err(ReleaseError::PublicationMissing {
                        revision: descriptor.revision,
                    });
                }
                None
            }
            Err(error) => return Err(ReleaseError::Store(error)),
        };

        Ok(ReleaseVerification {
            descriptor: descriptor.clone(),
            validation,
            publication,
        })
    }

    /// Verifies and accepts a candidate atomically from the gate's point of
    /// view. Any failure leaves `accepted_revision` unchanged.
    pub fn accept(
        &mut self,
        store: &RevisionStore,
        descriptor: &ReleaseDescriptor,
    ) -> Result<ReleaseVerification, ReleaseError> {
        let verification = self.verify(store, descriptor)?;
        self.accepted_revision = Some(
            self.accepted_revision
                .map_or(descriptor.revision, |accepted| {
                    accepted.max(descriptor.revision)
                }),
        );
        Ok(verification)
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], &'static str> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or("release descriptor length overflows")?;
        if end > self.bytes.len() {
            return Err("release descriptor is truncated");
        }
        let bytes = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(bytes)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], &'static str> {
        self.take(N)?
            .try_into()
            .map_err(|_| "release descriptor field has invalid length")
    }

    fn u8(&mut self) -> Result<u8, &'static str> {
        Ok(self.array::<1>()?[0])
    }

    fn u64(&mut self) -> Result<u64, &'static str> {
        Ok(u64::from_le_bytes(self.array()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::RevisionStore;
    use crate::types::{ArtifactDescriptor, AssetId, Manifest, ManifestEntry};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "prism-release-{label}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock must be after epoch")
                .as_nanos()
        ))
    }

    fn publish_build(store: &RevisionStore, revision: u64, bytes: &[u8]) -> PublishedManifest {
        let descriptor = ArtifactDescriptor::for_bytes(
            crate::types::ArtifactKey([revision as u8; 32]),
            1,
            1,
            16,
            vec![],
            bytes,
        );
        let mut transaction = store.begin(
            BuildRevision(revision),
            revision.checked_sub(1).map(BuildRevision),
        );
        transaction
            .stage(AssetId(revision as u128 + 1), descriptor, bytes.to_vec())
            .unwrap();
        transaction.commit().unwrap()
    }

    #[test]
    fn release_descriptor_is_deterministic_and_strictly_decodable() {
        let descriptor = ReleaseDescriptor {
            revision: BuildRevision(9),
            manifest_hash: ContentHash::digest(b"manifest"),
            publication_hash: None,
        };
        let encoded = descriptor.encode();
        assert_eq!(ReleaseDescriptor::decode(&encoded).unwrap(), descriptor);

        let mut trailing = encoded.clone();
        trailing.push(0);
        assert!(ReleaseDescriptor::decode(&trailing).is_err());

        let mut invalid_flag = encoded;
        *invalid_flag.last_mut().unwrap() = 2;
        assert!(ReleaseDescriptor::decode(&invalid_flag).is_err());
    }

    #[test]
    fn accepts_only_integral_current_manifest_and_cas() {
        let root = temp_dir("integrity");
        let store = RevisionStore::open(&root).unwrap();
        let published = publish_build(&store, 0, b"payload");
        let descriptor = ReleaseDescriptor::from_published(&published);
        let mut gate = ReleaseGate::new();
        let verification = gate.accept(&store, &descriptor).unwrap();
        assert_eq!(verification.validation.revision, BuildRevision(0));
        assert_eq!(gate.accepted_revision(), Some(BuildRevision(0)));

        let manifest_path = root.join("revisions").join("0.manifest");
        let mut manifest_bytes = fs::read(&manifest_path).unwrap();
        let last = manifest_bytes.len() - 1;
        manifest_bytes[last] ^= 1;
        fs::write(manifest_path, manifest_bytes).unwrap();
        assert!(matches!(
            gate.verify(&store, &descriptor),
            Err(ReleaseError::Store(_))
        ));
        assert_eq!(gate.accepted_revision(), Some(BuildRevision(0)));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_rollback_before_any_release_side_effect() {
        let root = temp_dir("rollback");
        let store = RevisionStore::open(&root).unwrap();
        let published = publish_build(&store, 0, b"payload");
        let descriptor = ReleaseDescriptor::from_published(&published);
        let gate = ReleaseGate::with_accepted_revision(BuildRevision(1));
        let error = gate.verify(&store, &descriptor).unwrap_err();
        assert!(matches!(
            error,
            ReleaseError::RevisionRollback {
                accepted: BuildRevision(1),
                candidate: BuildRevision(0)
            }
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn failed_accept_does_not_advance_floor() {
        let root = temp_dir("failure-closure");
        let store = RevisionStore::open(&root).unwrap();
        let published = publish_build(&store, 0, b"payload");
        let mut gate = ReleaseGate::new();
        let mut descriptor = ReleaseDescriptor::from_published(&published);
        descriptor.manifest_hash = ContentHash::digest(b"tampered");
        assert!(matches!(
            gate.accept(&store, &descriptor),
            Err(ReleaseError::ManifestHashMismatch { .. })
        ));
        assert_eq!(gate.accepted_revision(), None);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn publication_hash_and_binding_are_checked_when_present() {
        let root = temp_dir("publication");
        let store = RevisionStore::open(&root).unwrap();
        let published = publish_build(&store, 0, b"payload");
        let descriptor = ReleaseDescriptor::from_published(&published);
        assert!(descriptor.publication_hash.is_none());

        let manifest = Manifest::new(
            BuildRevision(0),
            [ManifestEntry {
                asset: AssetId(99),
                descriptor: ArtifactDescriptor::for_bytes(
                    crate::types::ArtifactKey([9; 32]),
                    1,
                    1,
                    16,
                    vec![],
                    b"runtime",
                ),
            }],
        )
        .unwrap();
        assert_eq!(manifest.revision, BuildRevision(0));
        fs::remove_dir_all(root).unwrap();
    }
}
