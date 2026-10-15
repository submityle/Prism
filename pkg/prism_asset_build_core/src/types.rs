use std::collections::BTreeMap;

/// Stable logical identity of an authored asset.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AssetId(pub u128);

/// Semantic build key. This is not the hash of the output bytes.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ArtifactKey(pub [u8; 32]);

/// Digest of the exact stored artifact bytes (BLAKE3-256).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ContentHash(pub [u8; 32]);

/// Monotonically increasing committed manifest revision.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BuildRevision(pub u64);

/// Stable identity of one publication transaction.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TransactionId(pub [u8; 16]);

/// Storage residency of an artifact descriptor.
///
/// Only [`ArtifactStorageClass::Shared`] artifacts may enter a build
/// `Manifest` or the shared DDC. `DriverLocal` is reserved for PSO/pipeline
/// and driver shader caches whose identity includes device/backend state.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ArtifactStorageClass {
    Shared,
    DriverLocal,
}

impl ArtifactStorageClass {
    #[must_use]
    pub const fn is_shared(self) -> bool {
        matches!(self, Self::Shared)
    }
}

impl TransactionId {
    /// Returns the canonical lowercase hexadecimal representation.
    #[must_use]
    pub fn to_hex(self) -> String {
        hex(&self.0)
    }
}

/// A committed artifact descriptor. `kind`/`schema` are explicit protocol IDs;
/// dependencies preserve caller-declared order. Blob integrity is checked by CAS.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactDescriptor {
    pub key: ArtifactKey,
    pub kind: u128,
    pub schema: u32,
    pub storage_class: ArtifactStorageClass,
    pub content_hash: ContentHash,
    pub byte_len: u64,
    pub alignment: u32,
    pub dependencies: Vec<ArtifactKey>,
}

impl ArtifactDescriptor {
    pub fn for_bytes(
        key: ArtifactKey,
        kind: u128,
        schema: u32,
        alignment: u32,
        dependencies: Vec<ArtifactKey>,
        bytes: &[u8],
    ) -> Self {
        Self {
            key,
            kind,
            schema,
            storage_class: ArtifactStorageClass::Shared,
            content_hash: ContentHash::digest(bytes),
            byte_len: bytes.len() as u64,
            alignment,
            dependencies,
        }
    }

    /// Creates a descriptor for a driver-local result. Such a descriptor is
    /// intentionally rejected by DDC, build Manifest, and runtime packaging.
    pub fn for_driver_local_bytes(
        key: ArtifactKey,
        kind: u128,
        schema: u32,
        alignment: u32,
        dependencies: Vec<ArtifactKey>,
        bytes: &[u8],
    ) -> Self {
        let mut descriptor = Self::for_bytes(key, kind, schema, alignment, dependencies, bytes);
        descriptor.storage_class = ArtifactStorageClass::DriverLocal;
        descriptor
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.alignment == 0 || !self.alignment.is_power_of_two() {
            return Err("alignment must be a non-zero power of two");
        }
        Ok(())
    }
}

impl ContentHash {
    pub fn digest(bytes: &[u8]) -> Self {
        Self(*blake3::hash(bytes).as_bytes())
    }

    pub fn to_hex(self) -> String {
        hex(&self.0)
    }
}

impl ArtifactKey {
    pub fn to_hex(self) -> String {
        hex(&self.0)
    }
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

/// One logical asset's selected artifact in a manifest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ManifestEntry {
    pub asset: AssetId,
    pub descriptor: ArtifactDescriptor,
}

/// Immutable, deterministically ordered snapshot published at a revision.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Manifest {
    pub revision: BuildRevision,
    pub entries: BTreeMap<AssetId, ArtifactDescriptor>,
}

impl Manifest {
    pub fn new(
        revision: BuildRevision,
        entries: impl IntoIterator<Item = ManifestEntry>,
    ) -> Result<Self, &'static str> {
        let mut map = BTreeMap::new();
        for entry in entries {
            entry.descriptor.validate()?;
            if !entry.descriptor.storage_class.is_shared() {
                return Err("manifest cannot contain driver-local artifacts");
            }
            if map.insert(entry.asset, entry.descriptor).is_some() {
                return Err("manifest contains duplicate AssetId");
            }
        }
        Ok(Self {
            revision,
            entries: map,
        })
    }

    /// Stable wire representation for local publication. Integer fields are
    /// little-endian; entries follow ascending AssetId and dependencies retain order.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"PRISM-MANIFEST\0\x02");
        out.extend_from_slice(&self.revision.0.to_le_bytes());
        out.extend_from_slice(&(self.entries.len() as u64).to_le_bytes());
        for (asset, d) in &self.entries {
            out.extend_from_slice(&asset.0.to_le_bytes());
            out.extend_from_slice(&d.key.0);
            out.extend_from_slice(&d.kind.to_le_bytes());
            out.extend_from_slice(&d.schema.to_le_bytes());
            out.push(match d.storage_class {
                ArtifactStorageClass::Shared => 0,
                ArtifactStorageClass::DriverLocal => 1,
            });
            out.extend_from_slice(&d.content_hash.0);
            out.extend_from_slice(&d.byte_len.to_le_bytes());
            out.extend_from_slice(&d.alignment.to_le_bytes());
            out.extend_from_slice(&(d.dependencies.len() as u64).to_le_bytes());
            for dep in &d.dependencies {
                out.extend_from_slice(&dep.0);
            }
        }
        out
    }
}
