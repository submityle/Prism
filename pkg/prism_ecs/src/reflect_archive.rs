//! Reflect-driven binary archives (design §16.5 / §24.3, `serialize` feature).
//!
//! Project a set of entities' **reflected** components into one binary blob and
//! read it back — a save file, a prefab payload, or a network snapshot — without
//! a line of per-component glue. The payload uses `prism_reflect`'s binary
//! encoder for each value (the single source of truth for the format, including
//! its [`DeserializeLimits`](prism_reflect::DeserializeLimits) guard against
//! hostile input); this module only adds the *envelope* that groups values by
//! entity.
//!
//! # Envelope
//!
//! ```text
//! magic "PECA" | version u8 | entity_count u32le
//! per entity:  entity_bits u64le | component_count u32le
//! per comp:    stable_id u64le | blob_len u32le | blob (prism_reflect to_binary)
//! ```
//!
//! Components are written in ascending [`StableTypeId`] order, so the same world
//! always produces byte-identical archives (they can be hashed, diffed, or
//! cached by content). Entities with no reflected component are skipped.
//!
//! # Load semantics
//!
//! For each archived entity: if the world already contains that exact id the
//! values are applied to it (overwriting); otherwise a **fresh** entity is
//! spawned. The returned `Vec<Entity>` is in archive order and names what each
//! record landed on, so a caller that needs the mapping (relations, scene
//! rebinding) can build one. Ids are therefore preserved when loading back into
//! the world they came from, and new ids are handed out when loading into an
//! empty one.

use alloc::vec::Vec;

use crate::entity::Entity;
use crate::world::World;
use prism_reflect::{DeserializeError, SerializeError, StableTypeId};

/// The archive envelope's magic bytes.
const MAGIC: [u8; 4] = *b"PECA";
/// The envelope format version.
const VERSION: u8 = 1;

/// Why an archive could not be written or read.
#[derive(Debug)]
pub enum ReflectArchiveError {
    /// The blob does not start with the archive magic.
    BadMagic,
    /// The blob was written by an unsupported envelope version.
    UnsupportedVersion(u8),
    /// The blob ended in the middle of a record.
    Truncated,
    /// An archived entity id is not a valid [`Entity`] handle.
    InvalidEntity(u64),
    /// The blob names a component type this world has not registered for
    /// reflection, so it cannot be reconstructed.
    UnknownType(StableTypeId),
    /// A component value could not be encoded.
    Encode(SerializeError),
    /// A component value could not be decoded.
    Decode(DeserializeError),
    /// The archive cannot be represented: more entities or a larger component
    /// blob than the `u32` envelope fields can express.
    TooLarge,
    /// A decoded value could not be materialised into its component type (the
    /// decoder and the bridge disagreed about the value shape).
    Unrepresentable(StableTypeId),
}

impl core::fmt::Display for ReflectArchiveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadMagic => f.write_str("not a Prism reflect archive (bad magic)"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported archive version {v}"),
            Self::Truncated => f.write_str("archive ended mid-record"),
            Self::InvalidEntity(bits) => write!(f, "archive entity bits {bits:#x} are not a valid handle"),
            Self::UnknownType(id) => {
                write!(f, "archive names unregistered reflected type {:#x}", id.value())
            }
            Self::Encode(e) => write!(f, "component encode failed: {e}"),
            Self::Decode(e) => write!(f, "component decode failed: {e}"),
            Self::TooLarge => f.write_str("archive would exceed the envelope's u32 size fields"),
            Self::Unrepresentable(id) => {
                write!(f, "decoded value is not representable as {:#x}", id.value())
            }
        }
    }
}

impl core::error::Error for ReflectArchiveError {}

impl From<SerializeError> for ReflectArchiveError {
    fn from(error: SerializeError) -> Self {
        Self::Encode(error)
    }
}

impl From<DeserializeError> for ReflectArchiveError {
    fn from(error: DeserializeError) -> Self {
        Self::Decode(error)
    }
}

/// Little-endian prefix helpers, kept local so the envelope stays explicit.
fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// A cursor over the blob that turns short reads into [`Truncated`](ReflectArchiveError::Truncated).
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], ReflectArchiveError> {
        let end = self
            .at
            .checked_add(len)
            .filter(|end| *end <= self.bytes.len())
            .ok_or(ReflectArchiveError::Truncated)?;
        let slice = &self.bytes[self.at..end];
        self.at = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, ReflectArchiveError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, ReflectArchiveError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u64(&mut self) -> Result<u64, ReflectArchiveError> {
        let bytes = self.take(8)?;
        let mut raw = [0u8; 8];
        raw.copy_from_slice(bytes);
        Ok(u64::from_le_bytes(raw))
    }
}

impl World {
    /// Encode the reflected components of `entities` into a binary archive.
    ///
    /// Entities carrying no registered reflected component are skipped. The
    /// output is deterministic (components in ascending `StableTypeId` order).
    ///
    /// # Errors
    /// [`ReflectArchiveError::Encode`] when a component value cannot be encoded
    /// (e.g. a leaf type outside the binary encoder's supported set).
    pub fn save_reflect_archive(
        &self,
        entities: impl IntoIterator<Item = Entity>,
    ) -> Result<Vec<u8>, ReflectArchiveError> {
        // Collect first so the writer can emit the entity count up front.
        let mut records: Vec<(Entity, Vec<(StableTypeId, &dyn prism_reflect::Reflect)>)> =
            Vec::new();
        for entity in entities {
            let components = self.reflect_entity(entity);
            if !components.is_empty() {
                records.push((entity, components));
            }
        }

        // The envelope stores counts and blob lengths as `u32`; refuse rather
        // than silently truncating a length and writing a corrupt archive.
        if records.len() > u32::MAX as usize {
            return Err(ReflectArchiveError::TooLarge);
        }

        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC);
        out.push(VERSION);
        push_u32(&mut out, records.len() as u32);
        for (entity, components) in records {
            push_u64(&mut out, entity.to_bits());
            push_u32(&mut out, components.len() as u32);
            for (type_id, value) in components {
                let blob = prism_reflect::to_binary(value)?;
                if blob.len() > u32::MAX as usize {
                    return Err(ReflectArchiveError::TooLarge);
                }
                push_u64(&mut out, type_id.value());
                push_u32(&mut out, blob.len() as u32);
                out.extend_from_slice(&blob);
            }
        }
        Ok(out)
    }

    /// Decode an archive produced by
    /// [`save_reflect_archive`](World::save_reflect_archive) and apply it.
    ///
    /// See the module docs for the load semantics (same id reused if live,
    /// otherwise a fresh entity). Returns the entities the records landed on, in
    /// archive order.
    ///
    /// # Errors
    /// [`ReflectArchiveError`] for a malformed archive, a type this world has no
    /// reflection registration for, or a value the decoder rejects.
    pub fn load_reflect_archive(&mut self, bytes: &[u8]) -> Result<Vec<Entity>, ReflectArchiveError> {
        let mut reader = Reader::new(bytes);
        if reader.take(4)? != MAGIC {
            return Err(ReflectArchiveError::BadMagic);
        }
        let version = reader.u8()?;
        if version != VERSION {
            return Err(ReflectArchiveError::UnsupportedVersion(version));
        }
        let entity_count = reader.u32()? as usize;

        // The count comes from the blob, so bound the pre-allocation by what the
        // remaining bytes could actually hold (each record is >= 12 bytes);
        // otherwise a tiny hostile blob could request gigabytes up front.
        let mut landed = Vec::with_capacity(entity_count.min(bytes.len() / 12 + 1));
        for _ in 0..entity_count {
            let bits = reader.u64()?;
            let archived = Entity::from_bits(bits).ok_or(ReflectArchiveError::InvalidEntity(bits))?;
            let component_count = reader.u32()? as usize;

            // Resolve the target *before* decoding, so a malformed component does
            // not leave a half-spawned entity behind for this record.
            let target = if self.contains(archived) {
                archived
            } else {
                self.spawn(())
            };

            for _ in 0..component_count {
                let raw_id = reader.u64()?;
                let blob_len = reader.u32()? as usize;
                let blob = reader.take(blob_len)?;
                let type_id = StableTypeId::from_raw(raw_id);
                let decoded = self
                    .reflect
                    .decode(type_id, blob)
                    .ok_or(ReflectArchiveError::UnknownType(type_id))??;
                // The bridge accepts `Dynamic*` trees, which is exactly what the
                // decoder produces, so no separate downcast path is needed here.
                if !self.insert_reflect(target, &*decoded) {
                    return Err(ReflectArchiveError::Unrepresentable(type_id));
                }
            }
            landed.push(target);
        }
        Ok(landed)
    }
}
