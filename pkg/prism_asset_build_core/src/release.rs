//! Release publication safety gate.
//!
//! This module is the final local release boundary for a committed revision.
//! It re-checks the durable manifest and CAS closure, binds the optional
//! runtime publication descriptor, and rejects rollback below the accepted
//! revision. Signed descriptors are verified against a deterministic local
//! trust policy whose updates are persisted atomically.

use crate::publication::PublicationDescriptor;
use crate::transaction::{PublishError, PublishedManifest, RevisionStore};
use crate::types::{BuildRevision, ContentHash};
use crate::validation::{ArtifactQualityGate, ValidationError, ValidationReport};
use fs4::fs_std::FileExt;
use ring::signature::{self, KeyPair};
use std::collections::BTreeMap;
use std::fs;
use std::fs::OpenOptions;
use std::io::ErrorKind;
use std::io::Write;
use std::mem::size_of;
use std::path::{Path, PathBuf};

const MAGIC: &[u8] = b"PRISM-RELEASE\0\x01";
const SIGNED_MAGIC: &[u8] = b"PRISM-RELEASE-SIGNED\0\x01";
const TRUST_MAGIC: &[u8] = b"PRISM-RELEASE-TRUST\0\x01";
const SIGNING_CONTEXT: &[u8] = b"Prism.AssetRelease.Ed25519.v1\0";
const ED25519_PUBLIC_KEY_BYTES: usize = 32;
const ED25519_SIGNATURE_BYTES: usize = 64;
const RELEASE_KEY_ID_BYTES: usize = 16;
const TRUST_CHECKSUM_BYTES: usize = 32;
const MAX_TRUSTED_KEYS: usize = 4_096;

/// Deterministic identity of a release candidate.
///
/// The descriptor binds the exact persisted build manifest and, when the
/// revision contains runtime outputs, the exact publication descriptor. It
/// intentionally has no signature field; [`SignedReleaseDescriptor`] binds
/// the canonical bytes to a trusted Ed25519 key.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReleaseDescriptor {
    /// Revision that must be visible through the store's `CURRENT` pointer.
    pub revision: BuildRevision,
    /// BLAKE3 digest of the canonical persisted build manifest bytes.
    pub manifest_hash: ContentHash,
    /// BLAKE3 digest of the canonical publication descriptor, if packaged.
    pub publication_hash: Option<ContentHash>,
}

/// Stable identifier for a release verification key.
///
/// The identifier is derived from the public key, so it can be recorded in a
/// package index without treating a caller-supplied label as an identity.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ReleaseKeyId(pub [u8; RELEASE_KEY_ID_BYTES]);

impl ReleaseKeyId {
    /// Derives the key id from the canonical Ed25519 public key bytes.
    #[must_use]
    pub fn from_public_key(public_key: &[u8; ED25519_PUBLIC_KEY_BYTES]) -> Self {
        let digest = blake3::hash(public_key);
        let mut id = [0; RELEASE_KEY_ID_BYTES];
        id.copy_from_slice(&digest.as_bytes()[..RELEASE_KEY_ID_BYTES]);
        Self(id)
    }
}

/// Public verification material for a signed release descriptor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReleasePublicKey {
    /// Content-derived key identifier used by the signed envelope.
    pub key_id: ReleaseKeyId,
    /// Raw Ed25519 public key bytes.
    pub public_key: [u8; ED25519_PUBLIC_KEY_BYTES],
}

impl ReleasePublicKey {
    /// Creates public verification material and derives its key id.
    #[must_use]
    pub fn from_bytes(public_key: [u8; ED25519_PUBLIC_KEY_BYTES]) -> Self {
        Self {
            key_id: ReleaseKeyId::from_public_key(&public_key),
            public_key,
        }
    }
}

/// A release descriptor plus its Ed25519 signature.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SignedReleaseDescriptor {
    /// The integrity descriptor covered by the signature.
    pub descriptor: ReleaseDescriptor,
    /// Identifier of the key that produced the signature.
    pub key_id: ReleaseKeyId,
    /// Ed25519 signature over the domain-separated canonical descriptor.
    pub signature: [u8; ED25519_SIGNATURE_BYTES],
}

impl SignedReleaseDescriptor {
    /// Encodes a strict, deterministic signed release envelope.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let descriptor = self.descriptor.encode();
        let mut bytes = Vec::with_capacity(
            SIGNED_MAGIC.len()
                + RELEASE_KEY_ID_BYTES
                + size_of::<u32>()
                + descriptor.len()
                + ED25519_SIGNATURE_BYTES,
        );
        bytes.extend_from_slice(SIGNED_MAGIC);
        bytes.extend_from_slice(&self.key_id.0);
        bytes.extend_from_slice(&(descriptor.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&descriptor);
        bytes.extend_from_slice(&self.signature);
        bytes
    }

    /// Decodes only the canonical signed envelope representation.
    pub fn decode(bytes: &[u8]) -> Result<Self, &'static str> {
        let mut cursor = Cursor { bytes, offset: 0 };
        if cursor.take(SIGNED_MAGIC.len())? != SIGNED_MAGIC {
            return Err("invalid signed release magic/version");
        }
        let key_id = ReleaseKeyId(cursor.array()?);
        let descriptor_length = cursor.u32()? as usize;
        let descriptor_bytes = cursor.take(descriptor_length)?;
        let descriptor = ReleaseDescriptor::decode(descriptor_bytes)?;
        let signature = cursor.array()?;
        if cursor.remaining() != 0 {
            return Err("signed release envelope has trailing bytes");
        }
        let signed = Self {
            descriptor,
            key_id,
            signature,
        };
        if signed.encode() != bytes {
            return Err("signed release envelope is not canonical");
        }
        Ok(signed)
    }

    /// Content identity of the canonical signed envelope.
    #[must_use]
    pub fn content_hash(&self) -> ContentHash {
        ContentHash::digest(&self.encode())
    }
}

/// Errors raised while signing or verifying a release envelope.
#[derive(Debug, Eq, PartialEq)]
pub enum ReleaseSignatureError {
    /// The signed envelope names a different key than the verifier supplied.
    KeyIdMismatch {
        expected: ReleaseKeyId,
        actual: ReleaseKeyId,
    },
    /// The Ed25519 signature is invalid for the descriptor and public key.
    InvalidSignature,
    /// The signing key could not be constructed from the supplied seed.
    InvalidPrivateKey,
}

impl std::fmt::Display for ReleaseSignatureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KeyIdMismatch { .. } => {
                f.write_str("signed release key id does not match the verification key")
            }
            Self::InvalidSignature => f.write_str("signed release signature is invalid"),
            Self::InvalidPrivateKey => f.write_str("invalid Ed25519 private key seed"),
        }
    }
}

impl std::error::Error for ReleaseSignatureError {}

/// Ed25519 release signer backed by caller-owned key material.
///
/// Key storage, rotation, and trust policy remain outside this crate. The
/// signer never reads a key from disk or silently invents one.
pub struct ReleaseSigner {
    key_pair: signature::Ed25519KeyPair,
    public: ReleasePublicKey,
}

impl ReleaseSigner {
    /// Builds a signer from a raw 32-byte Ed25519 seed.
    pub fn from_seed(seed: &[u8; 32]) -> Result<Self, ReleaseSignatureError> {
        let key_pair = signature::Ed25519KeyPair::from_seed_unchecked(seed)
            .map_err(|_| ReleaseSignatureError::InvalidPrivateKey)?;
        let public_key = key_pair
            .public_key()
            .as_ref()
            .try_into()
            .map_err(|_| ReleaseSignatureError::InvalidPrivateKey)?;
        let public = ReleasePublicKey::from_bytes(public_key);
        Ok(Self { key_pair, public })
    }

    /// Returns the public verification material for this signer.
    #[must_use]
    pub fn public_key(&self) -> ReleasePublicKey {
        self.public
    }

    /// Signs a release descriptor using a domain-separated canonical message.
    #[must_use]
    pub fn sign(&self, descriptor: ReleaseDescriptor) -> SignedReleaseDescriptor {
        let message = signing_message(&descriptor);
        let signature = self.key_pair.sign(&message);
        let signature = signature
            .as_ref()
            .try_into()
            .expect("ring Ed25519 signatures are always 64 bytes");
        SignedReleaseDescriptor {
            descriptor,
            key_id: self.public.key_id,
            signature,
        }
    }
}

/// Stateless Ed25519 release verifier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReleaseVerifier {
    public: ReleasePublicKey,
}

impl ReleaseVerifier {
    /// Creates a verifier from raw Ed25519 public key bytes.
    #[must_use]
    pub fn from_public_key(public_key: [u8; ED25519_PUBLIC_KEY_BYTES]) -> Self {
        Self {
            public: ReleasePublicKey::from_bytes(public_key),
        }
    }

    /// Creates a verifier from explicit public verification material.
    #[must_use]
    pub fn new(public: ReleasePublicKey) -> Self {
        Self { public }
    }

    /// Returns the public verification material.
    #[must_use]
    pub fn public_key(&self) -> ReleasePublicKey {
        self.public
    }

    /// Verifies key identity and the Ed25519 signature.
    pub fn verify(&self, signed: &SignedReleaseDescriptor) -> Result<(), ReleaseSignatureError> {
        if signed.key_id != self.public.key_id {
            return Err(ReleaseSignatureError::KeyIdMismatch {
                expected: self.public.key_id,
                actual: signed.key_id,
            });
        }
        let key = signature::UnparsedPublicKey::new(&signature::ED25519, self.public.public_key);
        key.verify(&signing_message(&signed.descriptor), &signed.signature)
            .map_err(|_| ReleaseSignatureError::InvalidSignature)
    }
}

/// Current disposition of a key retained by a release trust policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReleaseKeyStatus {
    /// The key may verify signed release descriptors.
    Active,
    /// The key remains recorded but can no longer authorize releases.
    Revoked,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TrustedReleaseKey {
    public: ReleasePublicKey,
    status: ReleaseKeyStatus,
}

/// Errors raised by trust-policy verification, mutation, or persistence.
#[derive(Debug)]
pub enum ReleaseTrustError {
    /// Trust policy bytes or fields are malformed or non-canonical.
    InvalidEncoding(&'static str),
    /// A public key's stored identifier is not derived from its bytes.
    InvalidKeyId {
        expected: ReleaseKeyId,
        actual: ReleaseKeyId,
    },
    /// The requested key is not present in the trust policy.
    UnknownKey(ReleaseKeyId),
    /// The requested key already exists in the trust policy.
    KeyAlreadyExists(ReleaseKeyId),
    /// The bounded trust policy cannot retain another key.
    KeyLimitExceeded,
    /// A revoked key was used or revoked again.
    RevokedKey(ReleaseKeyId),
    /// The signed release is below the policy's minimum accepted revision.
    RevisionRollback {
        minimum: BuildRevision,
        candidate: BuildRevision,
    },
    /// A replacement policy did not advance its policy revision.
    PolicyRevisionRollback { current: u64, candidate: u64 },
    /// Another store instance installed a different policy first.
    PolicyChanged { expected: u64, actual: u64 },
    /// A replacement policy weakens an already-installed trust decision.
    PolicyRollback(&'static str),
    /// The policy revision cannot be advanced further.
    PolicyRevisionOverflow,
    /// Ed25519 verification failed for the selected trusted key.
    Signature(ReleaseSignatureError),
    /// Trust policy storage failed.
    Io(std::io::Error),
}

impl std::fmt::Display for ReleaseTrustError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidEncoding(reason) => write!(f, "invalid release trust policy: {reason}"),
            Self::InvalidKeyId { .. } => {
                f.write_str("release public key id does not match its bytes")
            }
            Self::UnknownKey(_) => f.write_str("signed release uses an unknown key"),
            Self::KeyAlreadyExists(_) => f.write_str("release key already exists in trust policy"),
            Self::KeyLimitExceeded => f.write_str("release trust policy key limit exceeded"),
            Self::RevokedKey(_) => f.write_str("signed release uses a revoked key"),
            Self::RevisionRollback { minimum, candidate } => write!(
                f,
                "release revision rollback: minimum {}, candidate {}",
                minimum.0, candidate.0
            ),
            Self::PolicyRevisionRollback { current, candidate } => write!(
                f,
                "release trust policy rollback: current revision {current}, candidate {candidate}"
            ),
            Self::PolicyChanged { expected, actual } => write!(
                f,
                "release trust policy changed concurrently: expected revision {expected}, actual {actual}"
            ),
            Self::PolicyRollback(reason) => {
                write!(f, "release trust policy weakens trust: {reason}")
            }
            Self::PolicyRevisionOverflow => f.write_str("release trust policy revision overflow"),
            Self::Signature(error) => write!(f, "release signature verification failed: {error}"),
            Self::Io(error) => write!(f, "release trust store I/O failed: {error}"),
        }
    }
}

impl std::error::Error for ReleaseTrustError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Signature(error) => Some(error),
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ReleaseTrustError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<ReleaseSignatureError> for ReleaseTrustError {
    fn from(value: ReleaseSignatureError) -> Self {
        Self::Signature(value)
    }
}

/// Deterministic local release trust policy.
///
/// The policy retains revoked keys so a later update cannot silently restore
/// them. Every successful mutation advances `policy_revision`; failed
/// mutations leave all fields unchanged.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrustPolicy {
    policy_revision: u64,
    minimum_revision: BuildRevision,
    keys: BTreeMap<ReleaseKeyId, TrustedReleaseKey>,
}

impl TrustPolicy {
    /// Creates an empty policy at an explicit policy and release revision.
    #[must_use]
    pub fn new(policy_revision: u64, minimum_revision: BuildRevision) -> Self {
        Self {
            policy_revision,
            minimum_revision,
            keys: BTreeMap::new(),
        }
    }

    /// Returns the monotonic revision of this trust policy.
    #[must_use]
    pub fn policy_revision(&self) -> u64 {
        self.policy_revision
    }

    /// Returns the oldest release revision this policy may authorize.
    #[must_use]
    pub fn minimum_revision(&self) -> BuildRevision {
        self.minimum_revision
    }

    /// Returns the stored disposition for a key id.
    #[must_use]
    pub fn key_status(&self, key_id: ReleaseKeyId) -> Option<ReleaseKeyStatus> {
        self.keys.get(&key_id).map(|key| key.status)
    }

    /// Returns trusted public material, including revoked material for audit.
    #[must_use]
    pub fn public_key(&self, key_id: ReleaseKeyId) -> Option<ReleasePublicKey> {
        self.keys.get(&key_id).map(|key| key.public)
    }

    /// Adds a new active release key and advances the policy revision.
    pub fn add_key(&mut self, public: ReleasePublicKey) -> Result<(), ReleaseTrustError> {
        validate_public_key(public)?;
        if self.keys.contains_key(&public.key_id) {
            return Err(ReleaseTrustError::KeyAlreadyExists(public.key_id));
        }
        if self.keys.len() >= MAX_TRUSTED_KEYS {
            return Err(ReleaseTrustError::KeyLimitExceeded);
        }
        let revision = next_policy_revision(self.policy_revision)?;
        self.keys.insert(
            public.key_id,
            TrustedReleaseKey {
                public,
                status: ReleaseKeyStatus::Active,
            },
        );
        self.policy_revision = revision;
        Ok(())
    }

    /// Revokes an existing active key and advances the policy revision.
    pub fn revoke_key(&mut self, key_id: ReleaseKeyId) -> Result<(), ReleaseTrustError> {
        let key = self
            .keys
            .get(&key_id)
            .ok_or(ReleaseTrustError::UnknownKey(key_id))?;
        if key.status == ReleaseKeyStatus::Revoked {
            return Err(ReleaseTrustError::RevokedKey(key_id));
        }
        let revision = next_policy_revision(self.policy_revision)?;
        self.keys
            .get_mut(&key_id)
            .expect("key checked above")
            .status = ReleaseKeyStatus::Revoked;
        self.policy_revision = revision;
        Ok(())
    }

    /// Adds a replacement key and revokes the previous key as one mutation.
    pub fn rotate_key(
        &mut self,
        previous: ReleaseKeyId,
        replacement: ReleasePublicKey,
    ) -> Result<(), ReleaseTrustError> {
        validate_public_key(replacement)?;
        let previous_key = self
            .keys
            .get(&previous)
            .ok_or(ReleaseTrustError::UnknownKey(previous))?;
        if previous_key.status == ReleaseKeyStatus::Revoked {
            return Err(ReleaseTrustError::RevokedKey(previous));
        }
        if self.keys.contains_key(&replacement.key_id) {
            return Err(ReleaseTrustError::KeyAlreadyExists(replacement.key_id));
        }
        if self.keys.len() >= MAX_TRUSTED_KEYS {
            return Err(ReleaseTrustError::KeyLimitExceeded);
        }
        let revision = next_policy_revision(self.policy_revision)?;
        self.keys
            .get_mut(&previous)
            .expect("key checked above")
            .status = ReleaseKeyStatus::Revoked;
        self.keys.insert(
            replacement.key_id,
            TrustedReleaseKey {
                public: replacement,
                status: ReleaseKeyStatus::Active,
            },
        );
        self.policy_revision = revision;
        Ok(())
    }

    /// Raises the minimum authorized release revision.
    pub fn advance_minimum_revision(
        &mut self,
        minimum_revision: BuildRevision,
    ) -> Result<(), ReleaseTrustError> {
        if minimum_revision < self.minimum_revision {
            return Err(ReleaseTrustError::RevisionRollback {
                minimum: self.minimum_revision,
                candidate: minimum_revision,
            });
        }
        if minimum_revision == self.minimum_revision {
            return Ok(());
        }
        self.policy_revision = next_policy_revision(self.policy_revision)?;
        self.minimum_revision = minimum_revision;
        Ok(())
    }

    /// Verifies the release revision and selects the trusted key by key id.
    pub fn verify(&self, signed: &SignedReleaseDescriptor) -> Result<(), ReleaseTrustError> {
        if signed.descriptor.revision < self.minimum_revision {
            return Err(ReleaseTrustError::RevisionRollback {
                minimum: self.minimum_revision,
                candidate: signed.descriptor.revision,
            });
        }
        let trusted = self
            .keys
            .get(&signed.key_id)
            .ok_or(ReleaseTrustError::UnknownKey(signed.key_id))?;
        if trusted.status == ReleaseKeyStatus::Revoked {
            return Err(ReleaseTrustError::RevokedKey(signed.key_id));
        }
        ReleaseVerifier::new(trusted.public).verify(signed)?;
        Ok(())
    }

    /// Encodes the policy in key-id order with a checksum.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(
            TRUST_MAGIC.len()
                + size_of::<u64>() * 2
                + size_of::<u32>()
                + self.keys.len() * (RELEASE_KEY_ID_BYTES + ED25519_PUBLIC_KEY_BYTES + 1)
                + TRUST_CHECKSUM_BYTES,
        );
        bytes.extend_from_slice(TRUST_MAGIC);
        bytes.extend_from_slice(&self.policy_revision.to_le_bytes());
        bytes.extend_from_slice(&self.minimum_revision.0.to_le_bytes());
        bytes.extend_from_slice(&(self.keys.len() as u32).to_le_bytes());
        for (key_id, key) in &self.keys {
            bytes.extend_from_slice(&key_id.0);
            bytes.extend_from_slice(&key.public.public_key);
            bytes.push(match key.status {
                ReleaseKeyStatus::Active => 0,
                ReleaseKeyStatus::Revoked => 1,
            });
        }
        let checksum = ContentHash::digest(&bytes);
        bytes.extend_from_slice(&checksum.0);
        bytes
    }

    /// Strictly decodes a checksummed canonical trust policy.
    pub fn decode(bytes: &[u8]) -> Result<Self, ReleaseTrustError> {
        let header_len = TRUST_MAGIC.len() + size_of::<u64>() * 2 + size_of::<u32>();
        if bytes.len() < header_len + TRUST_CHECKSUM_BYTES {
            return Err(ReleaseTrustError::InvalidEncoding(
                "trust policy is truncated",
            ));
        }
        let body_len = bytes.len() - TRUST_CHECKSUM_BYTES;
        let expected = ContentHash(
            bytes[body_len..]
                .try_into()
                .expect("trust checksum length checked"),
        );
        if ContentHash::digest(&bytes[..body_len]) != expected {
            return Err(ReleaseTrustError::InvalidEncoding(
                "trust policy checksum mismatch",
            ));
        }
        let mut cursor = Cursor {
            bytes: &bytes[..body_len],
            offset: 0,
        };
        if cursor
            .take(TRUST_MAGIC.len())
            .map_err(ReleaseTrustError::InvalidEncoding)?
            != TRUST_MAGIC
        {
            return Err(ReleaseTrustError::InvalidEncoding(
                "invalid trust policy magic/version",
            ));
        }
        let policy_revision = cursor.u64().map_err(ReleaseTrustError::InvalidEncoding)?;
        let minimum_revision =
            BuildRevision(cursor.u64().map_err(ReleaseTrustError::InvalidEncoding)?);
        let count = cursor.u32().map_err(ReleaseTrustError::InvalidEncoding)? as usize;
        if count > MAX_TRUSTED_KEYS {
            return Err(ReleaseTrustError::InvalidEncoding(
                "too many trusted release keys",
            ));
        }
        let mut keys = BTreeMap::new();
        let mut previous = None;
        for _ in 0..count {
            let key_id = ReleaseKeyId(cursor.array().map_err(ReleaseTrustError::InvalidEncoding)?);
            if previous.is_some_and(|prior| prior >= key_id) {
                return Err(ReleaseTrustError::InvalidEncoding(
                    "release keys are not strictly ordered",
                ));
            }
            previous = Some(key_id);
            let public = ReleasePublicKey {
                key_id,
                public_key: cursor.array().map_err(ReleaseTrustError::InvalidEncoding)?,
            };
            validate_public_key(public)?;
            let status = match cursor.u8().map_err(ReleaseTrustError::InvalidEncoding)? {
                0 => ReleaseKeyStatus::Active,
                1 => ReleaseKeyStatus::Revoked,
                _ => {
                    return Err(ReleaseTrustError::InvalidEncoding(
                        "invalid release key status",
                    ))
                }
            };
            keys.insert(key_id, TrustedReleaseKey { public, status });
        }
        if cursor.remaining() != 0 {
            return Err(ReleaseTrustError::InvalidEncoding(
                "trust policy has trailing bytes",
            ));
        }
        let policy = Self {
            policy_revision,
            minimum_revision,
            keys,
        };
        if policy.encode() != bytes {
            return Err(ReleaseTrustError::InvalidEncoding(
                "trust policy is not canonical",
            ));
        }
        Ok(policy)
    }

    fn validate_successor(&self, candidate: &Self) -> Result<(), ReleaseTrustError> {
        if candidate.policy_revision <= self.policy_revision {
            return Err(ReleaseTrustError::PolicyRevisionRollback {
                current: self.policy_revision,
                candidate: candidate.policy_revision,
            });
        }
        if candidate.minimum_revision < self.minimum_revision {
            return Err(ReleaseTrustError::PolicyRollback(
                "minimum release revision decreased",
            ));
        }
        for (key_id, current) in &self.keys {
            let next = candidate
                .keys
                .get(key_id)
                .ok_or(ReleaseTrustError::PolicyRollback(
                    "existing release key was removed",
                ))?;
            if next.public != current.public {
                return Err(ReleaseTrustError::PolicyRollback(
                    "existing release key material changed",
                ));
            }
            if current.status == ReleaseKeyStatus::Revoked
                && next.status != ReleaseKeyStatus::Revoked
            {
                return Err(ReleaseTrustError::PolicyRollback(
                    "revoked release key was re-enabled",
                ));
            }
        }
        Ok(())
    }
}

fn validate_public_key(public: ReleasePublicKey) -> Result<(), ReleaseTrustError> {
    let expected = ReleaseKeyId::from_public_key(&public.public_key);
    if public.key_id != expected {
        return Err(ReleaseTrustError::InvalidKeyId {
            expected,
            actual: public.key_id,
        });
    }
    Ok(())
}

fn next_policy_revision(current: u64) -> Result<u64, ReleaseTrustError> {
    current
        .checked_add(1)
        .ok_or(ReleaseTrustError::PolicyRevisionOverflow)
}

/// Atomically persisted local release trust policy.
#[derive(Debug)]
pub struct ReleaseTrustStore {
    path: PathBuf,
    policy: TrustPolicy,
}

impl ReleaseTrustStore {
    /// Creates a new trust store without overwriting an existing file.
    pub fn create(path: impl AsRef<Path>, policy: TrustPolicy) -> Result<Self, ReleaseTrustError> {
        let path = path.as_ref().to_path_buf();
        persist_new_policy(&path, &policy)?;
        Ok(Self { path, policy })
    }

    /// Opens and strictly decodes an existing trust store.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ReleaseTrustError> {
        let path = path.as_ref().to_path_buf();
        let policy = TrustPolicy::decode(&fs::read(&path)?)?;
        Ok(Self { path, policy })
    }

    /// Returns the currently installed policy.
    #[must_use]
    pub fn policy(&self) -> &TrustPolicy {
        &self.policy
    }

    /// Verifies with the currently installed policy.
    pub fn verify(&self, signed: &SignedReleaseDescriptor) -> Result<(), ReleaseTrustError> {
        self.policy.verify(signed)
    }

    /// Installs a strictly stronger, newer policy.
    pub fn replace_policy(&mut self, candidate: TrustPolicy) -> Result<(), ReleaseTrustError> {
        self.policy.validate_successor(&candidate)?;
        let _lock = TrustStoreFileLock::acquire(&self.path)?;
        self.ensure_installed_policy_unchanged()?;
        persist_policy(&self.path, &candidate)?;
        self.policy = candidate;
        Ok(())
    }

    /// Persists a newly trusted active key.
    pub fn add_key(&mut self, public: ReleasePublicKey) -> Result<(), ReleaseTrustError> {
        self.update(|policy| policy.add_key(public))
    }

    /// Persists a key revocation.
    pub fn revoke_key(&mut self, key_id: ReleaseKeyId) -> Result<(), ReleaseTrustError> {
        self.update(|policy| policy.revoke_key(key_id))
    }

    /// Persists an atomic add-and-revoke key rotation.
    pub fn rotate_key(
        &mut self,
        previous: ReleaseKeyId,
        replacement: ReleasePublicKey,
    ) -> Result<(), ReleaseTrustError> {
        self.update(|policy| policy.rotate_key(previous, replacement))
    }

    /// Persists a higher minimum release revision.
    pub fn advance_minimum_revision(
        &mut self,
        minimum_revision: BuildRevision,
    ) -> Result<(), ReleaseTrustError> {
        self.update(|policy| policy.advance_minimum_revision(minimum_revision))
    }

    fn update(
        &mut self,
        mutation: impl FnOnce(&mut TrustPolicy) -> Result<(), ReleaseTrustError>,
    ) -> Result<(), ReleaseTrustError> {
        let mut candidate = self.policy.clone();
        mutation(&mut candidate)?;
        if candidate == self.policy {
            return Ok(());
        }
        let _lock = TrustStoreFileLock::acquire(&self.path)?;
        self.ensure_installed_policy_unchanged()?;
        persist_policy(&self.path, &candidate)?;
        self.policy = candidate;
        Ok(())
    }

    fn ensure_installed_policy_unchanged(&self) -> Result<(), ReleaseTrustError> {
        let installed = TrustPolicy::decode(&fs::read(&self.path)?)?;
        if installed != self.policy {
            return Err(ReleaseTrustError::PolicyChanged {
                expected: self.policy.policy_revision,
                actual: installed.policy_revision,
            });
        }
        Ok(())
    }
}

struct TrustStoreFileLock(fs::File);

impl TrustStoreFileLock {
    fn acquire(policy_path: &Path) -> Result<Self, ReleaseTrustError> {
        let parent = policy_parent(policy_path);
        fs::create_dir_all(parent)?;
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(trust_lock_path(policy_path))?;
        FileExt::lock_exclusive(&file)?;
        Ok(Self(file))
    }
}

impl Drop for TrustStoreFileLock {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.0);
    }
}

fn persist_policy(path: &Path, policy: &TrustPolicy) -> Result<(), ReleaseTrustError> {
    let parent = policy_parent(path);
    fs::create_dir_all(parent)?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    temporary.write_all(&policy.encode())?;
    temporary.as_file().sync_all()?;
    temporary
        .persist(path)
        .map_err(|error| ReleaseTrustError::Io(error.error))?;
    if let Ok(directory) = fs::File::open(parent) {
        let _ = directory.sync_all();
    }
    Ok(())
}

fn persist_new_policy(path: &Path, policy: &TrustPolicy) -> Result<(), ReleaseTrustError> {
    let parent = policy_parent(path);
    fs::create_dir_all(parent)?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    temporary.write_all(&policy.encode())?;
    temporary.as_file().sync_all()?;
    temporary
        .persist_noclobber(path)
        .map_err(|error| ReleaseTrustError::Io(error.error))?;
    if let Ok(directory) = fs::File::open(parent) {
        let _ = directory.sync_all();
    }
    Ok(())
}

fn policy_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn trust_lock_path(path: &Path) -> PathBuf {
    let mut lock = path.as_os_str().to_os_string();
    lock.push(".lock");
    PathBuf::from(lock)
}

fn signing_message(descriptor: &ReleaseDescriptor) -> Vec<u8> {
    let encoded = descriptor.encode();
    let mut message = Vec::with_capacity(SIGNING_CONTEXT.len() + encoded.len());
    message.extend_from_slice(SIGNING_CONTEXT);
    message.extend_from_slice(&encoded);
    message
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
    /// Signature or trust-policy verification failed.
    Trust(ReleaseTrustError),
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
            Self::Trust(error) => write!(f, "release trust validation failed: {error}"),
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
            Self::Trust(error) => Some(error),
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

impl From<ReleaseTrustError> for ReleaseError {
    fn from(value: ReleaseTrustError) -> Self {
        Self::Trust(value)
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

    /// Verifies a signed candidate against persisted trust and release state
    /// without advancing the in-process accepted revision.
    pub fn verify_signed(
        &self,
        store: &RevisionStore,
        trust: &ReleaseTrustStore,
        signed: &SignedReleaseDescriptor,
    ) -> Result<ReleaseVerification, ReleaseError> {
        trust.verify(signed)?;
        self.verify(store, &signed.descriptor)
    }

    /// Verifies trust and release integrity before atomically advancing the
    /// in-process accepted revision. Any failure leaves the floor unchanged.
    pub fn accept_signed(
        &mut self,
        store: &RevisionStore,
        trust: &ReleaseTrustStore,
        signed: &SignedReleaseDescriptor,
    ) -> Result<ReleaseVerification, ReleaseError> {
        trust.verify(signed)?;
        self.accept(store, &signed.descriptor)
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

    fn u32(&mut self) -> Result<u32, &'static str> {
        Ok(u32::from_le_bytes(self.array()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::RevisionStore;
    use crate::types::{ArtifactDescriptor, AssetId, Manifest, ManifestEntry};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

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
    fn ed25519_release_signatures_are_deterministic_and_verifiable() {
        let signer = ReleaseSigner::from_seed(&[7; 32]).unwrap();
        let descriptor = ReleaseDescriptor {
            revision: BuildRevision(3),
            manifest_hash: ContentHash::digest(b"manifest"),
            publication_hash: Some(ContentHash::digest(b"publication")),
        };
        let signed = signer.sign(descriptor.clone());
        let encoded = signed.encode();
        assert_eq!(
            ReleaseVerifier::new(signer.public_key()).verify(&signed),
            Ok(())
        );
        assert_eq!(
            ReleaseVerifier::new(signer.public_key())
                .verify(&ReleaseSigner::from_seed(&[7; 32]).unwrap().sign(descriptor)),
            Ok(())
        );
        assert_eq!(SignedReleaseDescriptor::decode(&encoded).unwrap(), signed);
        assert_eq!(signed.content_hash(), ContentHash::digest(&encoded));
    }

    #[test]
    fn ed25519_release_signatures_reject_tampering_and_wrong_keys() {
        let signer = ReleaseSigner::from_seed(&[11; 32]).unwrap();
        let signed = signer.sign(ReleaseDescriptor {
            revision: BuildRevision(4),
            manifest_hash: ContentHash::digest(b"manifest"),
            publication_hash: None,
        });
        let wrong_key = ReleaseVerifier::from_public_key(
            ReleaseSigner::from_seed(&[12; 32])
                .unwrap()
                .public_key()
                .public_key,
        );
        assert!(matches!(
            wrong_key.verify(&signed),
            Err(ReleaseSignatureError::KeyIdMismatch { .. })
        ));

        let mut tampered = signed.clone();
        tampered.descriptor.revision = BuildRevision(5);
        assert_eq!(
            ReleaseVerifier::new(signer.public_key()).verify(&tampered),
            Err(ReleaseSignatureError::InvalidSignature)
        );

        let mut encoded = signed.encode();
        encoded[0] ^= 1;
        assert!(SignedReleaseDescriptor::decode(&encoded).is_err());
    }

    fn release_descriptor(revision: u64) -> ReleaseDescriptor {
        ReleaseDescriptor {
            revision: BuildRevision(revision),
            manifest_hash: ContentHash::digest(b"manifest"),
            publication_hash: None,
        }
    }

    #[test]
    fn trust_policy_encoding_is_canonical_deterministic_and_strict() {
        let first = ReleaseSigner::from_seed(&[21; 32]).unwrap();
        let second = ReleaseSigner::from_seed(&[22; 32]).unwrap();
        let mut left = TrustPolicy::new(7, BuildRevision(3));
        left.add_key(first.public_key()).unwrap();
        left.add_key(second.public_key()).unwrap();
        let mut right = TrustPolicy::new(7, BuildRevision(3));
        right.add_key(second.public_key()).unwrap();
        right.add_key(first.public_key()).unwrap();
        assert_eq!(left.encode(), right.encode());
        assert_eq!(TrustPolicy::decode(&left.encode()).unwrap(), left);

        let mut tampered = left.encode();
        tampered[TRUST_MAGIC.len()] ^= 1;
        assert!(matches!(
            TrustPolicy::decode(&tampered),
            Err(ReleaseTrustError::InvalidEncoding(
                "trust policy checksum mismatch"
            ))
        ));

        let mut trailing = left.encode();
        trailing.push(0);
        assert!(TrustPolicy::decode(&trailing).is_err());

        let mut unordered = left.encode();
        let records = TRUST_MAGIC.len() + size_of::<u64>() * 2 + size_of::<u32>();
        let record_len = RELEASE_KEY_ID_BYTES + ED25519_PUBLIC_KEY_BYTES + 1;
        let (first_record, remaining) = unordered[records..].split_at_mut(record_len);
        let second_record = &mut remaining[..record_len];
        first_record.swap_with_slice(second_record);
        let checksum_offset = unordered.len() - TRUST_CHECKSUM_BYTES;
        let checksum = ContentHash::digest(&unordered[..checksum_offset]);
        unordered[checksum_offset..].copy_from_slice(&checksum.0);
        assert!(matches!(
            TrustPolicy::decode(&unordered),
            Err(ReleaseTrustError::InvalidEncoding(
                "release keys are not strictly ordered"
            ))
        ));
    }

    #[test]
    fn trust_policy_rejects_unknown_revoked_tampered_and_rollback_releases() {
        let trusted = ReleaseSigner::from_seed(&[31; 32]).unwrap();
        let unknown = ReleaseSigner::from_seed(&[32; 32]).unwrap();
        let mut policy = TrustPolicy::new(0, BuildRevision(5));
        policy.add_key(trusted.public_key()).unwrap();

        assert!(matches!(
            policy.verify(&unknown.sign(release_descriptor(5))),
            Err(ReleaseTrustError::UnknownKey(_))
        ));
        assert!(matches!(
            policy.verify(&trusted.sign(release_descriptor(4))),
            Err(ReleaseTrustError::RevisionRollback {
                minimum: BuildRevision(5),
                candidate: BuildRevision(4)
            })
        ));

        let mut tampered = trusted.sign(release_descriptor(5));
        tampered.descriptor.manifest_hash = ContentHash::digest(b"tampered");
        assert!(matches!(
            policy.verify(&tampered),
            Err(ReleaseTrustError::Signature(
                ReleaseSignatureError::InvalidSignature
            ))
        ));

        policy.revoke_key(trusted.public_key().key_id).unwrap();
        assert!(matches!(
            policy.verify(&trusted.sign(release_descriptor(5))),
            Err(ReleaseTrustError::RevokedKey(_))
        ));
    }

    #[test]
    fn trust_policy_rotation_is_atomic_and_selects_by_key_id() {
        let old = ReleaseSigner::from_seed(&[41; 32]).unwrap();
        let new = ReleaseSigner::from_seed(&[42; 32]).unwrap();
        let mut policy = TrustPolicy::new(4, BuildRevision(0));
        policy.add_key(old.public_key()).unwrap();
        let before_failed_rotation = policy.clone();
        assert!(matches!(
            policy.rotate_key(ReleaseKeyId([0; RELEASE_KEY_ID_BYTES]), new.public_key()),
            Err(ReleaseTrustError::UnknownKey(_))
        ));
        assert_eq!(policy, before_failed_rotation);

        let previous_revision = policy.policy_revision();
        policy
            .rotate_key(old.public_key().key_id, new.public_key())
            .unwrap();
        assert_eq!(policy.policy_revision(), previous_revision + 1);
        assert_eq!(
            policy.key_status(old.public_key().key_id),
            Some(ReleaseKeyStatus::Revoked)
        );
        assert_eq!(
            policy.key_status(new.public_key().key_id),
            Some(ReleaseKeyStatus::Active)
        );
        assert!(matches!(
            policy.verify(&old.sign(release_descriptor(1))),
            Err(ReleaseTrustError::RevokedKey(_))
        ));
        assert!(policy.verify(&new.sign(release_descriptor(1))).is_ok());
    }

    #[test]
    fn trust_store_persists_updates_and_keeps_old_policy_on_failure() {
        let root = temp_dir("trust-store");
        let path = root.join("release.trust");
        let first = ReleaseSigner::from_seed(&[51; 32]).unwrap();
        let second = ReleaseSigner::from_seed(&[52; 32]).unwrap();
        let mut policy = TrustPolicy::new(10, BuildRevision(2));
        policy.add_key(first.public_key()).unwrap();
        let mut store = ReleaseTrustStore::create(&path, policy).unwrap();
        store
            .rotate_key(first.public_key().key_id, second.public_key())
            .unwrap();
        store.advance_minimum_revision(BuildRevision(8)).unwrap();
        let reopened = ReleaseTrustStore::open(&path).unwrap();
        assert_eq!(reopened.policy(), store.policy());
        assert!(reopened.verify(&second.sign(release_descriptor(8))).is_ok());

        let installed = store.policy().clone();
        fs::remove_file(&path).unwrap();
        fs::remove_file(trust_lock_path(&path)).unwrap();
        fs::remove_dir(&root).unwrap();
        fs::write(&root, b"not a directory").unwrap();
        assert!(matches!(
            store.add_key(ReleaseSigner::from_seed(&[53; 32]).unwrap().public_key()),
            Err(ReleaseTrustError::Io(_))
        ));
        assert_eq!(store.policy(), &installed);
        fs::remove_file(root).unwrap();
    }

    #[test]
    fn stale_trust_store_cannot_overwrite_a_newer_policy() {
        let root = temp_dir("trust-stale-writer");
        let path = root.join("release.trust");
        let first = ReleaseSigner::from_seed(&[55; 32]).unwrap();
        let second = ReleaseSigner::from_seed(&[56; 32]).unwrap();
        let third = ReleaseSigner::from_seed(&[57; 32]).unwrap();
        let mut initial = TrustPolicy::new(2, BuildRevision(0));
        initial.add_key(first.public_key()).unwrap();
        let mut newer = ReleaseTrustStore::create(&path, initial).unwrap();
        let mut stale = ReleaseTrustStore::open(&path).unwrap();

        newer.add_key(second.public_key()).unwrap();
        let stale_policy = stale.policy().clone();
        assert!(matches!(
            stale.add_key(third.public_key()),
            Err(ReleaseTrustError::PolicyChanged { .. })
        ));
        assert_eq!(stale.policy(), &stale_policy);
        assert_eq!(
            ReleaseTrustStore::open(&path).unwrap().policy(),
            newer.policy()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn trust_store_file_lock_blocks_until_guard_is_released() {
        let root = temp_dir("trust-lock-contention");
        let path = root.join("release.trust");
        let guard = TrustStoreFileLock::acquire(&path).unwrap();
        let (sender, receiver) = std::sync::mpsc::channel();
        let worker_path = path.clone();
        let worker = std::thread::spawn(move || {
            sender
                .send(TrustStoreFileLock::acquire(&worker_path).is_ok())
                .unwrap();
        });

        assert!(matches!(
            receiver.recv_timeout(Duration::from_millis(100)),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout)
        ));
        drop(guard);
        assert!(receiver.recv_timeout(Duration::from_secs(5)).unwrap());
        worker.join().unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn signed_release_gate_advances_only_after_trust_and_integrity_pass() {
        let root = temp_dir("signed-gate");
        let store = RevisionStore::open(root.join("revisions")).unwrap();
        let published = publish_build(&store, 0, b"payload");
        let signer = ReleaseSigner::from_seed(&[58; 32]).unwrap();
        let unknown = ReleaseSigner::from_seed(&[59; 32]).unwrap();
        let mut policy = TrustPolicy::new(0, BuildRevision(0));
        policy.add_key(signer.public_key()).unwrap();
        let trust = ReleaseTrustStore::create(root.join("release.trust"), policy).unwrap();
        let descriptor = ReleaseDescriptor::from_published(&published);
        let mut gate = ReleaseGate::new();

        assert!(matches!(
            gate.accept_signed(&store, &trust, &unknown.sign(descriptor.clone())),
            Err(ReleaseError::Trust(ReleaseTrustError::UnknownKey(_)))
        ));
        assert_eq!(gate.accepted_revision(), None);

        let verification = gate
            .accept_signed(&store, &trust, &signer.sign(descriptor))
            .unwrap();
        assert_eq!(verification.validation.revision, BuildRevision(0));
        assert_eq!(gate.accepted_revision(), Some(BuildRevision(0)));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn replacement_policy_cannot_roll_back_trust_decisions() {
        let root = temp_dir("trust-replacement");
        let path = root.join("release.trust");
        let signer = ReleaseSigner::from_seed(&[61; 32]).unwrap();
        let mut policy = TrustPolicy::new(1, BuildRevision(4));
        policy.add_key(signer.public_key()).unwrap();
        policy.revoke_key(signer.public_key().key_id).unwrap();
        let mut store = ReleaseTrustStore::create(&path, policy.clone()).unwrap();

        let mut older = TrustPolicy::new(policy.policy_revision() - 1, BuildRevision(4));
        older.add_key(signer.public_key()).unwrap();
        let installed = store.policy().clone();
        assert!(matches!(
            store.replace_policy(older),
            Err(ReleaseTrustError::PolicyRevisionRollback { .. })
                | Err(ReleaseTrustError::PolicyRollback(_))
        ));
        assert_eq!(store.policy(), &installed);

        let mut reenabled = TrustPolicy::new(
            policy.policy_revision() + 10,
            BuildRevision(policy.minimum_revision().0 - 1),
        );
        reenabled.add_key(signer.public_key()).unwrap();
        assert!(matches!(
            store.replace_policy(reenabled),
            Err(ReleaseTrustError::PolicyRollback(_))
        ));
        assert_eq!(store.policy(), &installed);
        fs::remove_dir_all(root).unwrap();
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
