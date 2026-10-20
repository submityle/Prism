//! Runtime patch descriptors shared by the build and runtime sides.
//!
//! A descriptor is deliberately separate from a runtime manifest.  The
//! manifest describes the routes introduced by a patch, while this record
//! binds that delta to the base and target revisions and lists logical routes
//! that must be hidden from the base overlay.

use alloc::vec::Vec;
use core::fmt;

use prism_asset::StableGuid;

/// Eight-byte discriminator for runtime patch descriptors.
pub const PATCH_MAGIC: [u8; 8] = *b"PRSPATCH";
/// Current runtime patch descriptor wire version.
pub const PATCH_VERSION: u16 = 1;
const HEADER_LEN: usize = 8 + 2 + 2 + 8 + 8 + (32 * 5) + 4;
const HASH_LEN: usize = 32;

/// Immutable metadata needed to apply one runtime patch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimePatchDescriptor {
    /// Revision from which the patch may be applied.
    pub base_revision: u64,
    /// Revision produced after applying the patch.
    pub target_revision: u64,
    /// Canonical build manifest hash of the base revision.
    pub base_build_manifest_hash: [u8; 32],
    /// Canonical build manifest hash of the target revision.
    pub target_build_manifest_hash: [u8; 32],
    /// Canonical runtime manifest hash of the base revision.
    pub base_runtime_manifest_hash: [u8; 32],
    /// Canonical runtime manifest hash after applying the patch.
    pub target_runtime_manifest_hash: [u8; 32],
    /// Canonical runtime manifest hash carried by the patch payload itself.
    pub patch_runtime_manifest_hash: [u8; 32],
    /// Logical routes removed from the base revision, in canonical order.
    pub removed_guids: Vec<StableGuid>,
}

/// Errors raised by patch descriptor construction or decoding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PatchDescriptorError {
    /// The target revision must be newer than the base revision.
    TargetNotNewer {
        /// Currently installed revision.
        base: u64,
        /// Requested target revision, which must exceed `base`.
        target: u64,
    },
    /// A nil GUID cannot be a tombstone identity.
    NilGuid,
    /// Tombstone identities must be strictly ascending and unique.
    NonCanonicalGuids,
    /// The descriptor bytes ended before a field was complete.
    Truncated,
    /// The descriptor discriminator was invalid.
    BadMagic,
    /// The descriptor wire version is unsupported.
    UnsupportedVersion(u16),
    /// Reserved bits, count, or trailing bytes were invalid.
    InvalidFormat,
    /// The body digest does not match the encoded body.
    HashMismatch,
}

impl fmt::Display for PatchDescriptorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TargetNotNewer { base, target } => {
                write!(
                    f,
                    "patch target revision {target} is not newer than base {base}"
                )
            }
            Self::NilGuid => f.write_str("patch descriptor contains a nil tombstone GUID"),
            Self::NonCanonicalGuids => {
                f.write_str("patch tombstone GUIDs are not strictly canonical")
            }
            Self::Truncated => f.write_str("runtime patch descriptor is truncated"),
            Self::BadMagic => f.write_str("invalid runtime patch descriptor magic"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported runtime patch descriptor version {version}")
            }
            Self::InvalidFormat => f.write_str("invalid runtime patch descriptor format"),
            Self::HashMismatch => f.write_str("runtime patch descriptor hash mismatch"),
        }
    }
}

impl std::error::Error for PatchDescriptorError {}

impl RuntimePatchDescriptor {
    /// Validates and constructs a descriptor.
    pub fn new(
        base_revision: u64,
        target_revision: u64,
        base_build_manifest_hash: [u8; 32],
        target_build_manifest_hash: [u8; 32],
        base_runtime_manifest_hash: [u8; 32],
        target_runtime_manifest_hash: [u8; 32],
        patch_runtime_manifest_hash: [u8; 32],
        removed_guids: Vec<StableGuid>,
    ) -> Result<Self, PatchDescriptorError> {
        let descriptor = Self {
            base_revision,
            target_revision,
            base_build_manifest_hash,
            target_build_manifest_hash,
            base_runtime_manifest_hash,
            target_runtime_manifest_hash,
            patch_runtime_manifest_hash,
            removed_guids,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    /// Validates semantic and canonical invariants without encoding.
    pub fn validate(&self) -> Result<(), PatchDescriptorError> {
        if self.target_revision <= self.base_revision {
            return Err(PatchDescriptorError::TargetNotNewer {
                base: self.base_revision,
                target: self.target_revision,
            });
        }
        let mut previous = None;
        for guid in &self.removed_guids {
            if guid.is_nil() {
                return Err(PatchDescriptorError::NilGuid);
            }
            if previous.is_some_and(|previous| previous >= *guid) {
                return Err(PatchDescriptorError::NonCanonicalGuids);
            }
            previous = Some(*guid);
        }
        Ok(())
    }

    /// Encodes the canonical descriptor and appends a BLAKE3 body digest.
    pub fn encode(&self) -> Result<Vec<u8>, PatchDescriptorError> {
        self.validate()?;
        let count = u32::try_from(self.removed_guids.len())
            .map_err(|_| PatchDescriptorError::InvalidFormat)?;
        let mut bytes = Vec::with_capacity(
            HEADER_LEN
                .checked_add(self.removed_guids.len().saturating_mul(16))
                .and_then(|length| length.checked_add(HASH_LEN))
                .ok_or(PatchDescriptorError::InvalidFormat)?,
        );
        bytes.extend_from_slice(&PATCH_MAGIC);
        bytes.extend_from_slice(&PATCH_VERSION.to_le_bytes());
        bytes.extend_from_slice(&0_u16.to_le_bytes());
        bytes.extend_from_slice(&self.base_revision.to_le_bytes());
        bytes.extend_from_slice(&self.target_revision.to_le_bytes());
        bytes.extend_from_slice(&self.base_build_manifest_hash);
        bytes.extend_from_slice(&self.target_build_manifest_hash);
        bytes.extend_from_slice(&self.base_runtime_manifest_hash);
        bytes.extend_from_slice(&self.target_runtime_manifest_hash);
        bytes.extend_from_slice(&self.patch_runtime_manifest_hash);
        bytes.extend_from_slice(&count.to_le_bytes());
        for guid in &self.removed_guids {
            bytes.extend_from_slice(&guid.to_u128().to_le_bytes());
        }
        let digest = blake3::hash(&bytes);
        bytes.extend_from_slice(digest.as_bytes());
        Ok(bytes)
    }

    /// Decodes and strictly validates one complete descriptor.
    pub fn decode(bytes: &[u8]) -> Result<Self, PatchDescriptorError> {
        if bytes.len() < HEADER_LEN + HASH_LEN {
            return Err(PatchDescriptorError::Truncated);
        }
        if bytes[..8] != PATCH_MAGIC {
            return Err(PatchDescriptorError::BadMagic);
        }
        let version = u16::from_le_bytes(bytes[8..10].try_into().unwrap());
        if version != PATCH_VERSION {
            return Err(PatchDescriptorError::UnsupportedVersion(version));
        }
        if bytes[10..12] != [0, 0] {
            return Err(PatchDescriptorError::InvalidFormat);
        }
        let body_end = bytes
            .len()
            .checked_sub(HASH_LEN)
            .ok_or(PatchDescriptorError::Truncated)?;
        if blake3::hash(&bytes[..body_end]).as_bytes() != &bytes[body_end..] {
            return Err(PatchDescriptorError::HashMismatch);
        }
        let count =
            u32::from_le_bytes(bytes[HEADER_LEN - 4..HEADER_LEN].try_into().unwrap()) as usize;
        let guid_bytes = count
            .checked_mul(16)
            .ok_or(PatchDescriptorError::InvalidFormat)?;
        let expected_len = HEADER_LEN
            .checked_add(guid_bytes)
            .and_then(|length| length.checked_add(HASH_LEN))
            .ok_or(PatchDescriptorError::InvalidFormat)?;
        if expected_len != bytes.len() {
            return Err(if expected_len > bytes.len() {
                PatchDescriptorError::Truncated
            } else {
                PatchDescriptorError::InvalidFormat
            });
        }
        let mut removed_guids = Vec::with_capacity(count);
        let mut cursor = HEADER_LEN;
        for _ in 0..count {
            let end = cursor
                .checked_add(16)
                .ok_or(PatchDescriptorError::InvalidFormat)?;
            removed_guids.push(StableGuid::from_u128(u128::from_le_bytes(
                bytes[cursor..end].try_into().unwrap(),
            )));
            cursor = end;
        }
        let descriptor = Self {
            base_revision: u64::from_le_bytes(bytes[12..20].try_into().unwrap()),
            target_revision: u64::from_le_bytes(bytes[20..28].try_into().unwrap()),
            base_build_manifest_hash: bytes[28..60].try_into().unwrap(),
            target_build_manifest_hash: bytes[60..92].try_into().unwrap(),
            base_runtime_manifest_hash: bytes[92..124].try_into().unwrap(),
            target_runtime_manifest_hash: bytes[124..156].try_into().unwrap(),
            patch_runtime_manifest_hash: bytes[156..188].try_into().unwrap(),
            removed_guids,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_round_trips_and_is_canonical() {
        let descriptor = RuntimePatchDescriptor::new(
            4,
            5,
            [1; 32],
            [2; 32],
            [3; 32],
            [4; 32],
            [5; 32],
            vec![StableGuid::from_u128(3), StableGuid::from_u128(9)],
        )
        .unwrap();
        let encoded = descriptor.encode().unwrap();
        assert_eq!(
            RuntimePatchDescriptor::decode(&encoded).unwrap(),
            descriptor
        );
    }

    #[test]
    fn descriptor_rejects_rollback_duplicate_and_tampering() {
        assert_eq!(
            RuntimePatchDescriptor::new(5, 5, [0; 32], [0; 32], [0; 32], [0; 32], [0; 32], vec![]),
            Err(PatchDescriptorError::TargetNotNewer { base: 5, target: 5 })
        );
        assert_eq!(
            RuntimePatchDescriptor::new(
                1,
                2,
                [0; 32],
                [0; 32],
                [0; 32],
                [0; 32],
                [0; 32],
                vec![StableGuid::from_u128(2), StableGuid::from_u128(2)],
            ),
            Err(PatchDescriptorError::NonCanonicalGuids)
        );
        let descriptor =
            RuntimePatchDescriptor::new(1, 2, [0; 32], [0; 32], [0; 32], [0; 32], [0; 32], vec![])
                .unwrap();
        let mut encoded = descriptor.encode().unwrap();
        encoded[20] ^= 1;
        assert_eq!(
            RuntimePatchDescriptor::decode(&encoded),
            Err(PatchDescriptorError::HashMismatch)
        );
    }
}
