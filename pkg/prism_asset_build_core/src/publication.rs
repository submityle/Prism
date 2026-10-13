//! Canonical descriptor for one immutable runtime publication.
//!
//! Runtime containers and their routing manifest are persisted as immutable
//! revision outputs. The descriptor binds their names, lengths and digests so
//! a revision can be validated as one package before it becomes visible.

use std::collections::BTreeMap;

use crate::types::{BuildRevision, ContentHash};

const MAGIC: &[u8] = b"PRISM-PUBLICATION\0\x01";

/// A persisted runtime container's canonical identity and content digest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublishedContainer {
    /// Name used by runtime manifest routes.
    pub name: String,
    /// Exact stored byte length.
    pub byte_len: u64,
    /// BLAKE3 digest of the complete container bytes.
    pub content_hash: ContentHash,
}

/// Immutable identity of the complete runtime output associated with a build revision.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublicationDescriptor {
    /// Build revision whose artifacts were packaged.
    pub revision: BuildRevision,
    /// Digest of the canonical build manifest.
    pub build_manifest_hash: ContentHash,
    /// Digest of the canonical runtime routing manifest.
    pub runtime_manifest_hash: ContentHash,
    /// Containers sorted by their canonical name.
    pub containers: Vec<PublishedContainer>,
}

impl PublicationDescriptor {
    /// Creates a descriptor from package bytes and enforces canonical names/order.
    pub fn new(
        revision: BuildRevision,
        build_manifest_hash: ContentHash,
        runtime_manifest: &[u8],
        containers: &BTreeMap<String, Vec<u8>>,
    ) -> Result<Self, &'static str> {
        let mut records = Vec::with_capacity(containers.len());
        for (name, bytes) in containers {
            validate_container_name(name)?;
            records.push(PublishedContainer {
                name: name.clone(),
                byte_len: u64::try_from(bytes.len()).map_err(|_| "container length overflows")?,
                content_hash: ContentHash::digest(bytes),
            });
        }
        Ok(Self {
            revision,
            build_manifest_hash,
            runtime_manifest_hash: ContentHash::digest(runtime_manifest),
            containers: records,
        })
    }

    /// Deterministic wire encoding; records are in ascending UTF-8 name order.
    pub fn encode(&self) -> Result<Vec<u8>, &'static str> {
        let count = u32::try_from(self.containers.len()).map_err(|_| "too many containers")?;
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&self.revision.0.to_le_bytes());
        out.extend_from_slice(&self.build_manifest_hash.0);
        out.extend_from_slice(&self.runtime_manifest_hash.0);
        out.extend_from_slice(&count.to_le_bytes());
        let mut previous: Option<&str> = None;
        for container in &self.containers {
            validate_container_name(&container.name)?;
            if previous.is_some_and(|name| name >= container.name.as_str()) {
                return Err("container names are not in unique canonical order");
            }
            previous = Some(&container.name);
            let name_len =
                u16::try_from(container.name.len()).map_err(|_| "container name is too long")?;
            out.extend_from_slice(&name_len.to_le_bytes());
            out.extend_from_slice(container.name.as_bytes());
            out.extend_from_slice(&container.byte_len.to_le_bytes());
            out.extend_from_slice(&container.content_hash.0);
        }
        Ok(out)
    }

    /// Decodes only canonical encodings and rejects trailing or malformed data.
    pub fn decode(bytes: &[u8]) -> Result<Self, &'static str> {
        let mut cursor = Cursor { bytes, offset: 0 };
        if cursor.take(MAGIC.len())? != MAGIC {
            return Err("invalid publication descriptor magic/version");
        }
        let revision = BuildRevision(cursor.u64()?);
        let build_manifest_hash = ContentHash(cursor.array()?);
        let runtime_manifest_hash = ContentHash(cursor.array()?);
        let count = cursor.u32()? as usize;
        if count > cursor.remaining() / 42 {
            return Err("publication container count exceeds descriptor size");
        }
        let mut containers = Vec::with_capacity(count);
        for _ in 0..count {
            let name_len = cursor.u16()? as usize;
            let name = std::str::from_utf8(cursor.take(name_len)?)
                .map_err(|_| "container name is not UTF-8")?
                .to_owned();
            validate_container_name(&name)?;
            let byte_len = cursor.u64()?;
            let content_hash = ContentHash(cursor.array()?);
            containers.push(PublishedContainer {
                name,
                byte_len,
                content_hash,
            });
        }
        if cursor.remaining() != 0 {
            return Err("publication descriptor has trailing bytes");
        }
        let descriptor = Self {
            revision,
            build_manifest_hash,
            runtime_manifest_hash,
            containers,
        };
        if descriptor.encode()?.as_slice() != bytes {
            return Err("publication descriptor is not canonical");
        }
        Ok(descriptor)
    }

    /// Content identity of this complete publication descriptor.
    #[must_use]
    pub fn content_hash(&self) -> ContentHash {
        ContentHash::digest(&self.encode().expect("validated descriptor must encode"))
    }
}

fn validate_container_name(name: &str) -> Result<(), &'static str> {
    if name.is_empty()
        || name.contains('\0')
        || name == "."
        || name == ".."
        || name.contains('/')
        || name.contains('\\')
    {
        return Err("invalid runtime container name");
    }
    Ok(())
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
            .ok_or("publication length overflows")?;
        if end > self.bytes.len() {
            return Err("publication descriptor is truncated");
        }
        let bytes = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(bytes)
    }
    fn array<const N: usize>(&mut self) -> Result<[u8; N], &'static str> {
        self.take(N)?
            .try_into()
            .map_err(|_| "publication field has invalid length")
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_encoding_round_trips_and_binds_all_outputs() {
        let mut containers = BTreeMap::new();
        containers.insert("base.container".to_owned(), b"base".to_vec());
        containers.insert("level.container".to_owned(), b"level".to_vec());
        let descriptor = PublicationDescriptor::new(
            BuildRevision(7),
            ContentHash::digest(b"build"),
            b"runtime",
            &containers,
        )
        .unwrap();
        let encoded = descriptor.encode().unwrap();
        assert_eq!(PublicationDescriptor::decode(&encoded).unwrap(), descriptor);
        assert_eq!(
            descriptor.containers[0].content_hash,
            ContentHash::digest(b"base")
        );
    }

    #[test]
    fn descriptor_rejects_noncanonical_names_and_trailing_bytes() {
        let mut containers = BTreeMap::new();
        containers.insert("../escape".to_owned(), vec![]);
        assert!(PublicationDescriptor::new(
            BuildRevision(0),
            ContentHash::digest(b"build"),
            b"runtime",
            &containers,
        )
        .is_err());

        let descriptor = PublicationDescriptor::new(
            BuildRevision(0),
            ContentHash::digest(b"build"),
            b"runtime",
            &BTreeMap::new(),
        )
        .unwrap();
        let mut encoded = descriptor.encode().unwrap();
        encoded.push(0);
        assert!(PublicationDescriptor::decode(&encoded).is_err());
    }
}
