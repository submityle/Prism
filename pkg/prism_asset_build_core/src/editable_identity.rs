//! Deterministic identities for importer-owned `.prism` source objects.
//!
//! A persistent key must identify the same logical source object across edits
//! and importer upgrades. Use a source UUID, persistent object handle, or a
//! caller-maintained identity mapping. Array positions, display names, paths
//! containing mutable names, and content hashes do not provide that guarantee.
//! If the source loses its object identities, the caller must preserve a mapping
//! or explicitly reconcile new identities; this module cannot infer continuity.
//!
//! The importer protocol ID is stable across implementation versions. Changing
//! that protocol ID, the asset namespace, or the persistent key changes the
//! identity input. Names, kinds, source content, and directory order are excluded.
//! A kind change can still be rejected separately by editable reimport validation.

use crate::editable::EditableSubAsset;
use crate::types::AssetId;
use std::fmt;

/// Maximum directory size, matching the editable package's subasset limit.
pub const MAX_IDENTITY_SUBASSETS: usize = 65_536;
/// Maximum UTF-8 name length in bytes, matching the editable package limit.
pub const MAX_IDENTITY_NAME_BYTES: usize = 4096;
/// Maximum persistent source-object key length in bytes.
pub const MAX_PERSISTENT_SUBASSET_KEY_BYTES: usize = 4096;

// This domain and framing are the identity protocol, independent of importer
// implementation versions and editable package schema versions.
const IDENTITY_DOMAIN: &[u8] = b"PRISM-EDITABLE-SUBASSET-ID\0\x01";

/// Borrowed input for constructing one stable editable subasset directory entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EditableSubAssetInput<'a> {
    /// Nonempty opaque bytes with persistent source-object identity semantics.
    /// Binary keys, including embedded NUL bytes, are accepted without normalization.
    pub persistent_key: &'a [u8],
    /// Current display name; must be nonempty, NUL-free, and unique in the directory.
    pub name: &'a str,
    /// Current semantic kind; excluded from identity derivation.
    pub kind: u128,
}

/// Bounded failures while deriving or constructing stable subasset identities.
///
/// Indices refer to the caller's input slice. Errors never clone caller strings
/// or keys, and a collision is rejected rather than resolved using input order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EditableSubAssetIdentityError {
    /// A source-object key has no identity bytes.
    EmptyPersistentKey,
    /// A source-object key exceeds the supported byte limit.
    PersistentKeyTooLong {
        /// Observed byte length.
        actual: usize,
        /// Maximum byte length.
        limit: usize,
    },
    /// The directory exceeds the editable package's entry limit.
    TooManySubassets {
        /// Observed entry count.
        actual: usize,
        /// Maximum entry count.
        limit: usize,
    },
    /// A name is empty or contains NUL.
    InvalidName {
        /// Input entry index.
        index: usize,
    },
    /// A name exceeds the editable package's UTF-8 byte limit.
    NameTooLong {
        /// Input entry index.
        index: usize,
        /// Observed byte length.
        actual: usize,
        /// Maximum byte length.
        limit: usize,
    },
    /// Two entries claim the same persistent object, even if their metadata differs.
    DuplicateKey {
        /// Earlier input entry index.
        first: usize,
        /// Later input entry index.
        second: usize,
    },
    /// Two entries have the same name, which editable packages cannot represent.
    DuplicateName {
        /// Earlier input entry index.
        first: usize,
        /// Later input entry index.
        second: usize,
    },
    /// Distinct persistent keys derived the same nonzero ID.
    IdCollision {
        /// Colliding derived ID.
        id: u64,
        /// Earlier input entry index.
        first: usize,
        /// Later input entry index.
        second: usize,
    },
}

impl fmt::Display for EditableSubAssetIdentityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPersistentKey => f.write_str("persistent subasset key is empty"),
            Self::PersistentKeyTooLong { actual, limit } => {
                write!(f, "persistent subasset key exceeds {limit} bytes: {actual}")
            }
            Self::TooManySubassets { actual, limit } => {
                write!(f, "subasset directory exceeds {limit} entries: {actual}")
            }
            Self::InvalidName { index } => {
                write!(f, "subasset name at index {index} is empty or contains NUL")
            }
            Self::NameTooLong {
                index,
                actual,
                limit,
            } => {
                write!(
                    f,
                    "subasset name at index {index} exceeds {limit} bytes: {actual}"
                )
            }
            Self::DuplicateKey { first, second } => {
                write!(
                    f,
                    "duplicate persistent subasset key at indices {first} and {second}"
                )
            }
            Self::DuplicateName { first, second } => {
                write!(f, "duplicate subasset name at indices {first} and {second}")
            }
            Self::IdCollision { id, first, second } => {
                write!(
                    f,
                    "subasset ID {id} collides at indices {first} and {second}"
                )
            }
        }
    }
}

impl std::error::Error for EditableSubAssetIdentityError {}

/// Derives a nonzero ID from an asset namespace, importer protocol, and persistent key.
///
/// The BLAKE3 transcript is the fixed identity domain, little-endian asset u128,
/// little-endian importer protocol u128, little-endian key byte length u64, and
/// exact key bytes. The first eight digest bytes are interpreted little-endian;
/// zero maps to one. This mapping is fixed and requires no retry loop.
///
/// Pass only the importer's stable protocol ID, never its implementation version.
/// This single-key helper cannot detect directory collisions; use
/// [`build_editable_subasset_directory`] to validate an entire directory.
pub fn derive_editable_subasset_id(
    asset: AssetId,
    importer_protocol_id: u128,
    persistent_key: &[u8],
) -> Result<u64, EditableSubAssetIdentityError> {
    validate_key(persistent_key)?;
    Ok(hash_identity(asset, importer_protocol_id, persistent_key))
}

/// Maps persistent source-object keys and current metadata to a canonical directory.
///
/// All counts and byte lengths are checked before hashing, cloning, or allocating
/// directory scratch space. Duplicate keys and names are rejected before hashing.
/// IDs are checked for collisions before any output names are cloned. Entries are
/// returned in strictly ascending ID order, independent of caller iteration order.
///
/// Work is bounded by the published count and byte limits, with O(n log n) sorting
/// and O(n) scratch space borrowing the caller's keys and names. No collision retry
/// or automatic key repair occurs. The result is a directory, not a guarantee that
/// the rest of an editable package fits its total encoded byte limit.
pub fn build_editable_subasset_directory(
    asset: AssetId,
    importer_protocol_id: u128,
    inputs: &[EditableSubAssetInput<'_>],
) -> Result<Vec<EditableSubAsset>, EditableSubAssetIdentityError> {
    build_directory_with(inputs, |key| {
        hash_identity(asset, importer_protocol_id, key)
    })
}

fn validate_key(key: &[u8]) -> Result<(), EditableSubAssetIdentityError> {
    if key.is_empty() {
        return Err(EditableSubAssetIdentityError::EmptyPersistentKey);
    }
    if key.len() > MAX_PERSISTENT_SUBASSET_KEY_BYTES {
        return Err(EditableSubAssetIdentityError::PersistentKeyTooLong {
            actual: key.len(),
            limit: MAX_PERSISTENT_SUBASSET_KEY_BYTES,
        });
    }
    Ok(())
}

fn hash_identity(asset: AssetId, importer_protocol_id: u128, key: &[u8]) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(IDENTITY_DOMAIN);
    hasher.update(&asset.0.to_le_bytes());
    hasher.update(&importer_protocol_id.to_le_bytes());
    hasher.update(&(key.len() as u64).to_le_bytes());
    hasher.update(key);
    nonzero_digest_id(hasher.finalize().as_bytes())
}

fn nonzero_digest_id(digest: &[u8; 32]) -> u64 {
    let mut bytes = [0; 8];
    bytes.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(bytes).max(1)
}

// The private hash seam allows deterministic collision and preflight tests
// without weakening the public derivation protocol or searching for collisions.
fn build_directory_with(
    inputs: &[EditableSubAssetInput<'_>],
    mut derive: impl FnMut(&[u8]) -> u64,
) -> Result<Vec<EditableSubAsset>, EditableSubAssetIdentityError> {
    if inputs.len() > MAX_IDENTITY_SUBASSETS {
        return Err(EditableSubAssetIdentityError::TooManySubassets {
            actual: inputs.len(),
            limit: MAX_IDENTITY_SUBASSETS,
        });
    }
    for (index, input) in inputs.iter().enumerate() {
        validate_key(input.persistent_key)?;
        if input.name.len() > MAX_IDENTITY_NAME_BYTES {
            return Err(EditableSubAssetIdentityError::NameTooLong {
                index,
                actual: input.name.len(),
                limit: MAX_IDENTITY_NAME_BYTES,
            });
        }
    }
    for (index, input) in inputs.iter().enumerate() {
        if input.name.is_empty() || input.name.as_bytes().contains(&0) {
            return Err(EditableSubAssetIdentityError::InvalidName { index });
        }
    }

    let mut order: Vec<_> = (0..inputs.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        inputs[a]
            .persistent_key
            .cmp(inputs[b].persistent_key)
            .then(a.cmp(&b))
    });
    for pair in order.windows(2) {
        if inputs[pair[0]].persistent_key == inputs[pair[1]].persistent_key {
            return Err(EditableSubAssetIdentityError::DuplicateKey {
                first: pair[0],
                second: pair[1],
            });
        }
    }
    order.sort_unstable_by(|&a, &b| inputs[a].name.cmp(inputs[b].name).then(a.cmp(&b)));
    for pair in order.windows(2) {
        if inputs[pair[0]].name == inputs[pair[1]].name {
            return Err(EditableSubAssetIdentityError::DuplicateName {
                first: pair[0],
                second: pair[1],
            });
        }
    }
    let mut ids: Vec<_> = order
        .into_iter()
        .map(|index| (derive(inputs[index].persistent_key), index))
        .collect();
    ids.sort_unstable();
    for pair in ids.windows(2) {
        if pair[0].0 == pair[1].0 {
            return Err(EditableSubAssetIdentityError::IdCollision {
                id: pair[0].0,
                first: pair[0].1,
                second: pair[1].1,
            });
        }
    }
    Ok(ids
        .into_iter()
        .map(|(id, index)| EditableSubAsset {
            id,
            name: inputs[index].name.to_owned(),
            kind: inputs[index].kind,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recipe::ProcessorIdentity;
    use crate::types::ContentHash;

    const ASSET: AssetId = AssetId(0x1234);
    const IMPORTER: u128 = 0x5678;

    fn input<'a>(key: &'a [u8], name: &'a str) -> EditableSubAssetInput<'a> {
        EditableSubAssetInput {
            persistent_key: key,
            name,
            kind: 17,
        }
    }

    fn directory(inputs: &[EditableSubAssetInput<'_>]) -> Vec<EditableSubAsset> {
        build_editable_subasset_directory(ASSET, IMPORTER, inputs).unwrap()
    }

    #[test]
    fn derivation_uses_the_fixed_framed_protocol() {
        let mut transcript = b"PRISM-EDITABLE-SUBASSET-ID\0\x01".to_vec();
        transcript.extend_from_slice(&ASSET.0.to_le_bytes());
        transcript.extend_from_slice(&IMPORTER.to_le_bytes());
        transcript.extend_from_slice(&3_u64.to_le_bytes());
        transcript.extend_from_slice(b"a\0b");
        let digest = ContentHash::digest(&transcript);
        let mut prefix = [0; 8];
        prefix.copy_from_slice(&digest.0[..8]);
        let expected = u64::from_le_bytes(prefix).max(1);
        assert_eq!(
            derive_editable_subasset_id(ASSET, IMPORTER, b"a\0b"),
            Ok(expected)
        );
        assert_eq!(
            derive_editable_subasset_id(ASSET, IMPORTER, b"a\0b"),
            Ok(expected)
        );
    }

    #[test]
    fn zero_digest_prefix_has_a_bounded_nonzero_mapping() {
        assert_eq!(nonzero_digest_id(&[0; 32]), 1);
        let mut digest = [0; 32];
        digest[0] = 1;
        assert_eq!(nonzero_digest_id(&digest), 1);
        digest[7] = 0x80;
        assert_eq!(nonzero_digest_id(&digest), (1_u64 << 63) | 1);
    }

    #[test]
    fn rename_reorder_and_kind_changes_preserve_identity() {
        let original = [input(b"uuid-a", "mesh"), input(b"uuid-b", "material")];
        let before = directory(&original);
        assert_eq!(before, directory(&[original[1], original[0]]));
        let renamed = [
            input(b"uuid-b", "new material"),
            EditableSubAssetInput {
                kind: 99,
                ..input(b"uuid-a", "new mesh")
            },
        ];
        let after = directory(&renamed);
        assert_eq!(
            before.iter().map(|s| s.id).collect::<Vec<_>>(),
            after.iter().map(|s| s.id).collect::<Vec<_>>()
        );
        assert!(after.windows(2).all(|pair| pair[0].id < pair[1].id));
        for entry in after {
            let old = before.iter().find(|old| old.id == entry.id).unwrap();
            match old.name.as_str() {
                "mesh" => {
                    assert_eq!(entry.name, "new mesh");
                    assert_eq!(entry.kind, 99);
                }
                "material" => assert_eq!(entry.name, "new material"),
                _ => panic!("unexpected entry"),
            }
        }
    }

    #[test]
    fn content_and_implementation_versions_are_not_identity_inputs() {
        let first = (ProcessorIdentity::new(IMPORTER, 1), b"source v1".as_slice());
        let next = (
            ProcessorIdentity::new(IMPORTER, u32::MAX),
            b"edited source v2".as_slice(),
        );
        assert_ne!(ContentHash::digest(first.1), ContentHash::digest(next.1));
        assert_ne!(first.0.version, next.0.version);
        let inputs = [input(b"persistent source UUID", "object")];
        assert_eq!(
            build_editable_subasset_directory(ASSET, first.0.id, &inputs),
            build_editable_subasset_directory(ASSET, next.0.id, &inputs),
        );
    }

    #[test]
    fn asset_importer_and_key_namespaces_are_independent() {
        let base = derive_editable_subasset_id(ASSET, IMPORTER, b"object").unwrap();
        let ids = [
            base,
            derive_editable_subasset_id(AssetId(ASSET.0 ^ (1 << 127)), IMPORTER, b"object")
                .unwrap(),
            derive_editable_subasset_id(ASSET, IMPORTER ^ (1 << 127), b"object").unwrap(),
            derive_editable_subasset_id(ASSET, IMPORTER, b"other object").unwrap(),
        ];
        assert!(ids.iter().all(|&id| id != 0));
        for (index, id) in ids.iter().enumerate() {
            assert!(!ids[..index].contains(id));
        }
        assert!(
            build_editable_subasset_directory(AssetId(0), 0, &[input(b"key", "name")]).unwrap()[0]
                .id
                != 0
        );
    }

    #[test]
    fn adding_an_unrelated_object_does_not_renumber_existing_entries() {
        let before = directory(&[input(b"uuid-a", "a")]);
        let after = directory(&[input(b"uuid-b", "b"), input(b"uuid-a", "a")]);
        assert!(after.contains(&before[0]));
        assert_eq!(directory(&[]), Vec::new());
    }

    #[test]
    fn duplicate_keys_are_rejected_before_hashing_even_with_different_metadata() {
        let inputs = [
            input(b"same", "a"),
            EditableSubAssetInput {
                kind: 19,
                ..input(b"same", "b")
            },
        ];
        assert_eq!(
            build_directory_with(&inputs, |_| panic!("must not hash duplicates")),
            Err(EditableSubAssetIdentityError::DuplicateKey {
                first: 0,
                second: 1
            })
        );
    }

    #[test]
    fn duplicate_names_are_rejected_before_hashing() {
        assert_eq!(
            build_directory_with(&[input(b"b", "same"), input(b"a", "same")], |_| panic!(
                "must not hash duplicates"
            )),
            Err(EditableSubAssetIdentityError::DuplicateName {
                first: 0,
                second: 1
            })
        );
    }

    #[test]
    fn collisions_are_reported_without_retries() {
        let mut calls = 0;
        let result = build_directory_with(&[input(b"b", "b"), input(b"a", "a")], |_| {
            calls += 1;
            7
        });
        assert_eq!(calls, 2);
        assert_eq!(
            result,
            Err(EditableSubAssetIdentityError::IdCollision {
                id: 7,
                first: 0,
                second: 1
            })
        );
        // The fixed zero-to-one mapping can itself create a collision, which
        // must follow the same rejection path as any truncated digest collision.
        assert_eq!(
            build_directory_with(&[input(b"a", "a"), input(b"b", "b")], |key| {
                let mut digest = [0; 32];
                digest[0] = u8::from(key == b"b");
                nonzero_digest_id(&digest)
            }),
            Err(EditableSubAssetIdentityError::IdCollision {
                id: 1,
                first: 0,
                second: 1
            })
        );
    }

    #[test]
    fn key_limits_apply_to_single_ids_and_entire_directories() {
        let long = vec![0; MAX_PERSISTENT_SUBASSET_KEY_BYTES + 1];
        for (key, error) in [
            (
                b"".as_slice(),
                EditableSubAssetIdentityError::EmptyPersistentKey,
            ),
            (
                long.as_slice(),
                EditableSubAssetIdentityError::PersistentKeyTooLong {
                    actual: long.len(),
                    limit: MAX_PERSISTENT_SUBASSET_KEY_BYTES,
                },
            ),
        ] {
            assert_eq!(
                derive_editable_subasset_id(ASSET, IMPORTER, key),
                Err(error.clone())
            );
            assert_eq!(
                build_directory_with(
                    &[input(b"valid", "first"), input(key, "second")],
                    |_| panic!("must preflight every key")
                ),
                Err(error)
            );
        }
        let boundary = vec![0; MAX_PERSISTENT_SUBASSET_KEY_BYTES];
        assert!(derive_editable_subasset_id(ASSET, IMPORTER, &boundary).unwrap() != 0);
        assert_eq!(directory(&[input(&boundary, "binary key")]).len(), 1);
    }

    #[test]
    fn name_validation_is_bounded_and_precedes_hashing() {
        for name in ["", "bad\0name"] {
            assert_eq!(
                build_directory_with(&[input(b"a", "first"), input(b"b", name)], |_| panic!(
                    "must preflight every name"
                )),
                Err(EditableSubAssetIdentityError::InvalidName { index: 1 })
            );
        }
        let long = "é".repeat(MAX_IDENTITY_NAME_BYTES / 2 + 1);
        assert_eq!(
            build_directory_with(&[input(b"a", "first"), input(b"b", &long)], |_| panic!(
                "must preflight name byte lengths"
            )),
            Err(EditableSubAssetIdentityError::NameTooLong {
                index: 1,
                actual: long.len(),
                limit: MAX_IDENTITY_NAME_BYTES
            })
        );
        let boundary = "é".repeat(MAX_IDENTITY_NAME_BYTES / 2);
        assert_eq!(directory(&[input(b"key", &boundary)])[0].name, boundary);
    }

    #[test]
    fn count_limit_is_checked_before_fields_or_hashing() {
        let inputs = vec![input(b"", ""); MAX_IDENTITY_SUBASSETS + 1];
        assert_eq!(
            build_directory_with(&inputs, |_| panic!("must preflight count")),
            Err(EditableSubAssetIdentityError::TooManySubassets {
                actual: inputs.len(),
                limit: MAX_IDENTITY_SUBASSETS
            })
        );
        // Exactly the limit passes count validation and reaches field validation.
        assert_eq!(
            build_editable_subasset_directory(ASSET, IMPORTER, &inputs[..MAX_IDENTITY_SUBASSETS]),
            Err(EditableSubAssetIdentityError::EmptyPersistentKey)
        );
    }
}
