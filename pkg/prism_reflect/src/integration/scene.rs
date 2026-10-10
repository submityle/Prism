//! Scene snapshots: an ECS-agnostic, ordered collection of reflected values
//! that can be serialized to a compact binary blob or a framed, human-readable
//! text envelope and reconstructed through a [`TypeRegistry`].
//!
//! A [`DynamicScene`] is the shared primitive behind ECS component snapshots,
//! prefab payloads, and on-disk scene files (design §22 "M6 集成"). It stores
//! each value together with its type name so the loader can look the concrete
//! type up in the registry and drive deserialization against the right
//! [`TypeInfo`](crate::TypeInfo).

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

use crate::ser::{
    from_binary, from_ron, to_binary, to_ron, DeserializeError, SerializeError, StableTypeId,
};
use crate::{Reflect, TypeRegistry};

/// A single reflected entry inside a [`DynamicScene`].
///
/// The entry owns the value plus the identity needed to reload it: the
/// `type_name` used to resolve a registration and the [`StableTypeId`] derived
/// from that name for cross-build verification.
pub struct SceneEntry {
    type_name: String,
    stable_id: StableTypeId,
    value: Box<dyn Reflect>,
}

impl SceneEntry {
    /// The registered type name of the stored value.
    #[must_use]
    pub fn type_name(&self) -> &str {
        &self.type_name
    }

    /// The deterministic [`StableTypeId`] of the stored value's type.
    #[must_use]
    pub fn stable_id(&self) -> StableTypeId {
        self.stable_id
    }

    /// A shared reference to the stored reflected value.
    #[must_use]
    pub fn value(&self) -> &dyn Reflect {
        &*self.value
    }

    /// A mutable reference to the stored reflected value.
    #[must_use]
    pub fn value_mut(&mut self) -> &mut dyn Reflect {
        &mut *self.value
    }

    /// Consume the entry and recover the owned reflected value.
    #[must_use]
    pub fn into_value(self) -> Box<dyn Reflect> {
        self.value
    }
}

impl fmt::Debug for SceneEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SceneEntry")
            .field("type_name", &self.type_name)
            .field("stable_id", &self.stable_id.value())
            .finish()
    }
}

/// An ordered, type-tagged collection of reflected values.
///
/// Build one by [`push`](DynamicScene::push)ing reflected values, then
/// [`to_binary`](DynamicScene::to_binary) /
/// [`to_text`](DynamicScene::to_text) it for storage and reload it with the
/// matching [`from_binary`](DynamicScene::from_binary) /
/// [`from_text`](DynamicScene::from_text) against a [`TypeRegistry`].
#[derive(Debug, Default)]
pub struct DynamicScene {
    entries: Vec<SceneEntry>,
}

const BIN_MAGIC: [u8; 4] = *b"PSCN";
const BIN_VERSION: u8 = 1;
const TEXT_HEADER: &str = "PSCN-RON v1";

impl DynamicScene {
    /// Create an empty scene.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Create an empty scene with room for `capacity` entries.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }

    /// Append a reflected value, tagging it with its type name and id.
    pub fn push(&mut self, value: Box<dyn Reflect>) {
        let type_name = value.type_name().to_string();
        let stable_id = StableTypeId::of_path(value.type_name());
        self.entries.push(SceneEntry {
            type_name,
            stable_id,
            value,
        });
    }

    /// Append a concrete reflected value by value.
    pub fn push_value<T: Reflect>(&mut self, value: T) {
        self.push(Box::new(value));
    }

    /// The number of entries in the scene.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the scene has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// All entries in insertion order.
    #[must_use]
    pub fn entries(&self) -> &[SceneEntry] {
        &self.entries
    }

    /// Iterate the stored reflected values in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = &dyn Reflect> {
        self.entries.iter().map(SceneEntry::value)
    }

    /// Remove every entry, keeping the allocated capacity.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Serialize the whole scene to the compact binary scene format.
    ///
    /// Layout: a 4-byte magic, a 1-byte version, a little-endian `u32` entry
    /// count, then for each entry a length-prefixed type name followed by the
    /// length-prefixed per-value payload produced by [`to_binary`].
    ///
    /// # Errors
    /// Returns a [`SceneError::Serialize`] if any entry's value fails to
    /// serialize.
    pub fn to_binary(&self) -> Result<Vec<u8>, SceneError> {
        let mut out = Vec::new();
        out.extend_from_slice(&BIN_MAGIC);
        out.push(BIN_VERSION);
        let count = u32::try_from(self.entries.len()).map_err(|_| SceneError::TooManyEntries)?;
        out.extend_from_slice(&count.to_le_bytes());
        for entry in &self.entries {
            let name = entry.type_name.as_bytes();
            let name_len = u32::try_from(name.len()).map_err(|_| SceneError::TooManyEntries)?;
            out.extend_from_slice(&name_len.to_le_bytes());
            out.extend_from_slice(name);
            let payload = to_binary(&*entry.value).map_err(SceneError::Serialize)?;
            let payload_len =
                u32::try_from(payload.len()).map_err(|_| SceneError::TooManyEntries)?;
            out.extend_from_slice(&payload_len.to_le_bytes());
            out.extend_from_slice(&payload);
        }
        Ok(out)
    }

    /// Reconstruct a scene from the binary scene format using `registry` to
    /// resolve each entry's concrete type.
    ///
    /// # Errors
    /// Returns a [`SceneError`] for a bad header, truncated input, an entry
    /// whose type name is not registered, or a per-value deserialization
    /// failure.
    pub fn from_binary(bytes: &[u8], registry: &TypeRegistry) -> Result<Self, SceneError> {
        let mut reader = SliceReader::new(bytes);
        if reader.take(4)? != BIN_MAGIC {
            return Err(SceneError::BadHeader);
        }
        if reader.take_u8()? != BIN_VERSION {
            return Err(SceneError::BadHeader);
        }
        let count = reader.take_u32()? as usize;
        let mut scene = Self::with_capacity(count);
        for _ in 0..count {
            let name_len = reader.take_u32()? as usize;
            let name_bytes = reader.take(name_len)?;
            let type_name = core::str::from_utf8(name_bytes)
                .map_err(|_| SceneError::BadHeader)?
                .to_string();
            let payload_len = reader.take_u32()? as usize;
            let payload = reader.take(payload_len)?;
            let value = decode_entry(&type_name, payload, registry, Payload::Binary)?;
            let stable_id = StableTypeId::of_path(value.type_name());
            scene.entries.push(SceneEntry {
                type_name,
                stable_id,
                value,
            });
        }
        if !reader.is_empty() {
            return Err(SceneError::TrailingData);
        }
        Ok(scene)
    }

    /// Serialize the scene to a framed, human-readable text envelope.
    ///
    /// The envelope is a header block (`PSCN-RON v1`, the entry count, then one
    /// `type_name<TAB>byte_len` line per entry), a `--` separator, and the
    /// concatenated per-entry RON payloads. Byte lengths let the loader slice
    /// each payload back out without re-parsing boundaries.
    ///
    /// # Errors
    /// Returns a [`SceneError::Serialize`] if any entry fails to serialize, or
    /// [`SceneError::BadHeader`] if a type name contains a newline or tab.
    pub fn to_text(&self) -> Result<String, SceneError> {
        let mut payloads: Vec<String> = Vec::with_capacity(self.entries.len());
        let mut header = String::new();
        header.push_str(TEXT_HEADER);
        header.push('\n');
        header.push_str(&self.entries.len().to_string());
        header.push('\n');
        for entry in &self.entries {
            if entry.type_name.contains('\n') || entry.type_name.contains('\t') {
                return Err(SceneError::BadHeader);
            }
            let payload = to_ron(&*entry.value).map_err(SceneError::Serialize)?;
            header.push_str(&format!("{}\t{}\n", entry.type_name, payload.len()));
            payloads.push(payload);
        }
        header.push_str("--\n");
        for payload in payloads {
            header.push_str(&payload);
        }
        Ok(header)
    }

    /// Reconstruct a scene from the [`to_text`](DynamicScene::to_text) envelope.
    ///
    /// # Errors
    /// Returns a [`SceneError`] for a malformed header, a length that runs off
    /// the end of the payload block, an unregistered type name, or a per-value
    /// deserialization failure.
    pub fn from_text(text: &str, registry: &TypeRegistry) -> Result<Self, SceneError> {
        let mut lines = text.lines();
        if lines.next() != Some(TEXT_HEADER) {
            return Err(SceneError::BadHeader);
        }
        let count: usize = lines
            .next()
            .ok_or(SceneError::BadHeader)?
            .parse()
            .map_err(|_| SceneError::BadHeader)?;
        let mut specs: Vec<(String, usize)> = Vec::with_capacity(count);
        for _ in 0..count {
            let line = lines.next().ok_or(SceneError::BadHeader)?;
            let (name, len) = line.split_once('\t').ok_or(SceneError::BadHeader)?;
            let len: usize = len.parse().map_err(|_| SceneError::BadHeader)?;
            specs.push((name.to_string(), len));
        }
        if lines.next() != Some("--") {
            return Err(SceneError::BadHeader);
        }
        // Everything after the separator line is the concatenated payload block.
        let sep = "--\n";
        let body_start = text
            .find(sep)
            .map(|i| i + sep.len())
            .ok_or(SceneError::BadHeader)?;
        let body = &text[body_start..];
        let mut scene = Self::with_capacity(count);
        let mut cursor = 0usize;
        for (type_name, len) in specs {
            let end = cursor.checked_add(len).ok_or(SceneError::Truncated)?;
            if end > body.len() {
                return Err(SceneError::Truncated);
            }
            let payload = body.get(cursor..end).ok_or(SceneError::Truncated)?;
            let value = decode_entry(&type_name, payload.as_bytes(), registry, Payload::Text)?;
            let stable_id = StableTypeId::of_path(value.type_name());
            scene.entries.push(SceneEntry {
                type_name,
                stable_id,
                value,
            });
            cursor = end;
        }
        Ok(scene)
    }
}

enum Payload {
    Binary,
    Text,
}

fn decode_entry(
    type_name: &str,
    payload: &[u8],
    registry: &TypeRegistry,
    kind: Payload,
) -> Result<Box<dyn Reflect>, SceneError> {
    let registration = registry
        .get_with_name(type_name)
        .ok_or_else(|| SceneError::UnknownType(type_name.to_string()))?;
    let target = registration.type_info();
    match kind {
        Payload::Binary => from_binary(payload, registry, target).map_err(SceneError::Deserialize),
        Payload::Text => {
            let text = core::str::from_utf8(payload).map_err(|_| SceneError::BadHeader)?;
            from_ron(text, registry, target).map_err(SceneError::Deserialize)
        }
    }
}

/// A minimal forward-only reader over a byte slice for the binary scene format.
struct SliceReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> SliceReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn is_empty(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], SceneError> {
        let end = self.pos.checked_add(len).ok_or(SceneError::Truncated)?;
        let slice = self.bytes.get(self.pos..end).ok_or(SceneError::Truncated)?;
        self.pos = end;
        Ok(slice)
    }

    fn take_u8(&mut self) -> Result<u8, SceneError> {
        Ok(self.take(1)?[0])
    }

    fn take_u32(&mut self) -> Result<u32, SceneError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }
}

/// An error produced while serializing or reconstructing a [`DynamicScene`].
#[derive(Debug)]
#[non_exhaustive]
pub enum SceneError {
    /// The binary magic/version or text header did not match.
    BadHeader,
    /// The input ended before a declared length was satisfied.
    Truncated,
    /// Extra bytes remained after the declared entries were read.
    TrailingData,
    /// The scene holds more entries or longer names than the format allows.
    TooManyEntries,
    /// A stored type name is not present in the [`TypeRegistry`].
    UnknownType(String),
    /// A per-value payload failed to serialize.
    Serialize(SerializeError),
    /// A per-value payload failed to deserialize.
    Deserialize(DeserializeError),
}

impl fmt::Display for SceneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SceneError::BadHeader => f.write_str("malformed scene header"),
            SceneError::Truncated => f.write_str("scene data ended unexpectedly"),
            SceneError::TrailingData => f.write_str("trailing bytes after scene entries"),
            SceneError::TooManyEntries => f.write_str("scene exceeds the format's size limits"),
            SceneError::UnknownType(name) => write!(f, "type `{name}` is not registered"),
            SceneError::Serialize(err) => write!(f, "scene entry serialization failed: {err}"),
            SceneError::Deserialize(err) => write!(f, "scene entry deserialization failed: {err}"),
        }
    }
}

impl core::error::Error for SceneError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            SceneError::Serialize(err) => Some(err),
            SceneError::Deserialize(err) => Some(err),
            _ => None,
        }
    }
}
