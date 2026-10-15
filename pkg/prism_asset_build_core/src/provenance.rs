//! Canonical local release provenance.
//!
//! This records facts available at the package boundary: the committed build
//! manifest, selected artifacts, runtime publication and emitted containers.
//! It is intentionally not a remote attestation service or a source-control
//! record; higher layers can bind those facts to this descriptor.

use crate::publication::PublicationDescriptor;
use crate::types::{ArtifactKey, BuildRevision, ContentHash, Manifest};

const MAGIC: &[u8] = b"PRISM-PROVENANCE\0\x01";
const PRODUCER: &[u8] = b"prism_asset_build_core";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProvenanceArtifact {
    pub asset: u128,
    pub key: ArtifactKey,
    pub kind: u128,
    pub schema: u32,
    pub content_hash: ContentHash,
    pub byte_len: u64,
    pub dependencies: Vec<ArtifactKey>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProvenanceContainer {
    pub name: String,
    pub byte_len: u64,
    pub content_hash: ContentHash,
}

/// Deterministic local provenance for one immutable runtime package.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReleaseProvenance {
    pub revision: BuildRevision,
    pub build_manifest_hash: ContentHash,
    pub publication_hash: ContentHash,
    pub runtime_manifest_hash: ContentHash,
    pub producer: String,
    pub artifacts: Vec<ProvenanceArtifact>,
    pub containers: Vec<ProvenanceContainer>,
}

impl ReleaseProvenance {
    pub fn from_manifest(
        manifest: &Manifest,
        build_manifest_hash: ContentHash,
        publication: &PublicationDescriptor,
    ) -> Result<Self, &'static str> {
        if manifest.revision != publication.revision
            || publication.build_manifest_hash != build_manifest_hash
        {
            return Err("provenance inputs do not bind the same build revision");
        }
        Ok(Self {
            revision: manifest.revision,
            build_manifest_hash,
            publication_hash: publication.content_hash(),
            runtime_manifest_hash: publication.runtime_manifest_hash,
            producer: String::from_utf8(PRODUCER.to_vec()).expect("producer is UTF-8"),
            artifacts: manifest
                .entries
                .iter()
                .map(|(asset, descriptor)| ProvenanceArtifact {
                    asset: asset.0,
                    key: descriptor.key,
                    kind: descriptor.kind,
                    schema: descriptor.schema,
                    content_hash: descriptor.content_hash,
                    byte_len: descriptor.byte_len,
                    dependencies: descriptor.dependencies.clone(),
                })
                .collect(),
            containers: publication
                .containers
                .iter()
                .map(|container| ProvenanceContainer {
                    name: container.name.clone(),
                    byte_len: container.byte_len,
                    content_hash: container.content_hash,
                })
                .collect(),
        })
    }

    pub fn encode(&self) -> Result<Vec<u8>, &'static str> {
        if self.producer.as_bytes() != PRODUCER {
            return Err("unsupported provenance producer");
        }
        let artifact_count =
            u32::try_from(self.artifacts.len()).map_err(|_| "too many provenance artifacts")?;
        let container_count =
            u32::try_from(self.containers.len()).map_err(|_| "too many provenance containers")?;
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&self.revision.0.to_le_bytes());
        out.extend_from_slice(&self.build_manifest_hash.0);
        out.extend_from_slice(&self.publication_hash.0);
        out.extend_from_slice(&self.runtime_manifest_hash.0);
        out.extend_from_slice(&(PRODUCER.len() as u16).to_le_bytes());
        out.extend_from_slice(PRODUCER);
        out.extend_from_slice(&artifact_count.to_le_bytes());
        let mut previous_asset = None;
        for artifact in &self.artifacts {
            if previous_asset.is_some_and(|previous| previous >= artifact.asset) {
                return Err("provenance artifacts are not in canonical order");
            }
            previous_asset = Some(artifact.asset);
            let dependency_count = u32::try_from(artifact.dependencies.len())
                .map_err(|_| "too many artifact dependencies")?;
            out.extend_from_slice(&artifact.asset.to_le_bytes());
            out.extend_from_slice(&artifact.key.0);
            out.extend_from_slice(&artifact.kind.to_le_bytes());
            out.extend_from_slice(&artifact.schema.to_le_bytes());
            out.extend_from_slice(&artifact.content_hash.0);
            out.extend_from_slice(&artifact.byte_len.to_le_bytes());
            out.extend_from_slice(&dependency_count.to_le_bytes());
            for dependency in &artifact.dependencies {
                out.extend_from_slice(&dependency.0);
            }
        }
        out.extend_from_slice(&container_count.to_le_bytes());
        let mut previous_name = None;
        for container in &self.containers {
            if container.name.is_empty()
                || container.name.contains('/')
                || container.name.contains('\\')
                || container.name.contains('\0')
            {
                return Err("invalid provenance container name");
            }
            if previous_name.is_some_and(|previous| previous >= container.name.as_str()) {
                return Err("provenance containers are not in canonical order");
            }
            previous_name = Some(container.name.as_str());
            let name_len = u16::try_from(container.name.len())
                .map_err(|_| "provenance container name is too long")?;
            out.extend_from_slice(&name_len.to_le_bytes());
            out.extend_from_slice(container.name.as_bytes());
            out.extend_from_slice(&container.byte_len.to_le_bytes());
            out.extend_from_slice(&container.content_hash.0);
        }
        Ok(out)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, &'static str> {
        let mut cursor = Cursor { bytes, offset: 0 };
        if cursor.take(MAGIC.len())? != MAGIC {
            return Err("invalid provenance magic/version");
        }
        let revision = BuildRevision(cursor.u64()?);
        let build_manifest_hash = ContentHash(cursor.array()?);
        let publication_hash = ContentHash(cursor.array()?);
        let runtime_manifest_hash = ContentHash(cursor.array()?);
        let producer_len = cursor.u16()? as usize;
        let producer = std::str::from_utf8(cursor.take(producer_len)?)
            .map_err(|_| "provenance producer is not UTF-8")?
            .to_owned();
        if producer.as_bytes() != PRODUCER {
            return Err("unsupported provenance producer");
        }
        let artifact_count = cursor.u32()? as usize;
        let mut artifacts = Vec::with_capacity(artifact_count);
        for _ in 0..artifact_count {
            let asset = cursor.u128()?;
            let key = ArtifactKey(cursor.array()?);
            let kind = cursor.u128()?;
            let schema = cursor.u32()?;
            let content_hash = ContentHash(cursor.array()?);
            let byte_len = cursor.u64()?;
            let dependency_count = cursor.u32()? as usize;
            let mut dependencies = Vec::with_capacity(dependency_count);
            for _ in 0..dependency_count {
                dependencies.push(ArtifactKey(cursor.array()?));
            }
            artifacts.push(ProvenanceArtifact {
                asset,
                key,
                kind,
                schema,
                content_hash,
                byte_len,
                dependencies,
            });
        }
        let container_count = cursor.u32()? as usize;
        let mut containers = Vec::with_capacity(container_count);
        for _ in 0..container_count {
            let name_len = cursor.u16()? as usize;
            let name = std::str::from_utf8(cursor.take(name_len)?)
                .map_err(|_| "provenance container name is not UTF-8")?
                .to_owned();
            let byte_len = cursor.u64()?;
            let content_hash = ContentHash(cursor.array()?);
            containers.push(ProvenanceContainer {
                name,
                byte_len,
                content_hash,
            });
        }
        if cursor.remaining() != 0 {
            return Err("provenance has trailing bytes");
        }
        let provenance = Self {
            revision,
            build_manifest_hash,
            publication_hash,
            runtime_manifest_hash,
            producer,
            artifacts,
            containers,
        };
        if provenance.encode()?.as_slice() != bytes {
            return Err("provenance is not canonical");
        }
        Ok(provenance)
    }

    #[must_use]
    pub fn content_hash(&self) -> ContentHash {
        ContentHash::digest(&self.encode().expect("validated provenance must encode"))
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
            .ok_or("provenance length overflows")?;
        if end > self.bytes.len() {
            return Err("provenance is truncated");
        }
        let bytes = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(bytes)
    }
    fn array<const N: usize>(&mut self) -> Result<[u8; N], &'static str> {
        self.take(N)?
            .try_into()
            .map_err(|_| "provenance field has invalid length")
    }
    fn u16(&mut self) -> Result<u16, &'static str> {
        Ok(u16::from_le_bytes(self.array()?))
    }
    fn u32(&mut self) -> Result<u32, &'static str> {
        Ok(u32::from_le_bytes(self.array()?))
    }
    fn u64(&mut self) -> Result<u64, &'static str> {
        Ok(u64::from_le_bytes(self.array()?))
    }
    fn u128(&mut self) -> Result<u128, &'static str> {
        Ok(u128::from_le_bytes(self.array()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::publication::PublishedContainer;
    use crate::types::{ArtifactDescriptor, ArtifactStorageClass, AssetId, ManifestEntry};

    #[test]
    fn provenance_round_trips_and_binds_artifacts_and_containers() {
        let descriptor = ArtifactDescriptor {
            key: ArtifactKey([1; 32]),
            kind: 7,
            schema: 2,
            storage_class: ArtifactStorageClass::Shared,
            content_hash: ContentHash::digest(b"artifact"),
            byte_len: 8,
            alignment: 16,
            dependencies: vec![ArtifactKey([2; 32])],
        };
        let manifest = Manifest::new(
            BuildRevision(3),
            [ManifestEntry {
                asset: AssetId(11),
                descriptor,
            }],
        )
        .unwrap();
        let publication = PublicationDescriptor {
            revision: BuildRevision(3),
            build_manifest_hash: ContentHash::digest(&manifest.canonical_bytes()),
            runtime_manifest_hash: ContentHash::digest(b"runtime"),
            containers: vec![PublishedContainer {
                name: "core".into(),
                byte_len: 9,
                content_hash: ContentHash::digest(b"container"),
            }],
        };
        let provenance = ReleaseProvenance::from_manifest(
            &manifest,
            publication.build_manifest_hash,
            &publication,
        )
        .unwrap();
        let encoded = provenance.encode().unwrap();
        assert_eq!(ReleaseProvenance::decode(&encoded).unwrap(), provenance);
    }

    #[test]
    fn provenance_rejects_noncanonical_order_and_trailing_bytes() {
        let mut provenance = ReleaseProvenance {
            revision: BuildRevision(0),
            build_manifest_hash: ContentHash::digest(b"manifest"),
            publication_hash: ContentHash::digest(b"publication"),
            runtime_manifest_hash: ContentHash::digest(b"runtime"),
            producer: String::from_utf8(PRODUCER.to_vec()).unwrap(),
            artifacts: Vec::new(),
            containers: vec![
                ProvenanceContainer {
                    name: "z".into(),
                    byte_len: 0,
                    content_hash: ContentHash::digest(b""),
                },
                ProvenanceContainer {
                    name: "a".into(),
                    byte_len: 0,
                    content_hash: ContentHash::digest(b""),
                },
            ],
        };
        assert!(provenance.encode().is_err());
        provenance.containers.swap(0, 1);
        let mut encoded = provenance.encode().unwrap();
        encoded.push(0);
        assert!(ReleaseProvenance::decode(&encoded).is_err());
    }
}
