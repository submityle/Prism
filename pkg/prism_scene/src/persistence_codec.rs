//! Deterministic, bounded binary codec for scene-instance persistence.
//!
//! This wire format is separate from the world transaction log: it stores one
//! scene snapshot, while `prism_world_persist` stores ordered world deltas.

use crate::identity::{LocalEntityId, PersistentId};
use crate::overrides::{ArchetypeSignature, ComponentTypeId, OverrideValue, StructuralOp};
use crate::path::{FieldPath, FieldSegment};
use crate::persistence::{PersistedPropertyOverride, SceneInstancePersistence};
use alloc::string::String;
use alloc::vec::Vec;
use prism_asset::StableGuid;

const MAGIC: &[u8; 4] = b"PRSI";
const VERSION: u16 = 1;

/// Bounds applied while decoding an untrusted scene snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PersistenceDecodeLimits {
    /// Maximum complete encoded snapshot size.
    pub max_bytes: usize,
    /// Maximum property records.
    pub max_properties: usize,
    /// Maximum structural records.
    pub max_structural: usize,
    /// Maximum path segments per property.
    pub max_path_segments: usize,
    /// Maximum UTF-8 bytes in one path name or string.
    pub max_name_bytes: usize,
    /// Maximum component ids in one added-entity archetype.
    pub max_archetype_components: usize,
    /// Maximum opaque bytes in one value or structural payload.
    pub max_value_bytes: usize,
}

impl Default for PersistenceDecodeLimits {
    fn default() -> Self {
        Self {
            max_bytes: 64 * 1024 * 1024,
            max_properties: 1_000_000,
            max_structural: 1_000_000,
            max_path_segments: 1024,
            max_name_bytes: 16 * 1024,
            max_archetype_components: 4096,
            max_value_bytes: 64 * 1024 * 1024,
        }
    }
}

/// Failure while encoding or decoding a scene persistence snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PersistenceCodecError {
    /// The four-byte wire magic did not match.
    BadMagic,
    /// The wire version is not supported.
    UnsupportedVersion(u16),
    /// The input ended before one complete value was available.
    Truncated,
    /// Bytes remained after one complete snapshot.
    TrailingBytes,
    /// A count or payload exceeded a caller-controlled bound.
    ResourceLimit(&'static str),
    /// A value violated the canonical wire contract.
    InvalidValue(&'static str),
}

/// Encodes one scene persistence snapshot as PRSI v1 bytes.
pub fn encode_persistence(
    snapshot: &SceneInstancePersistence,
) -> Result<Vec<u8>, PersistenceCodecError> {
    let mut writer = Writer(Vec::new());
    writer.bytes(MAGIC);
    writer.u16(VERSION);
    writer.guid(snapshot.template);
    write_optional_id(&mut writer, snapshot.instance);
    writer.count(snapshot.properties.len(), "properties")?;
    for property in &snapshot.properties {
        writer.u32(property.local.index());
        write_optional_id(&mut writer, property.persistent);
        writer.u64(property.component.raw());
        write_path(&mut writer, &property.path)?;
        write_value(&mut writer, &property.value)?;
    }
    writer.count(snapshot.structural.len(), "structural")?;
    for operation in &snapshot.structural {
        write_structural(&mut writer, operation)?;
    }
    Ok(writer.0)
}

/// Decodes and validates a PRSI v1 scene persistence snapshot.
pub fn decode_persistence(
    bytes: &[u8],
    limits: PersistenceDecodeLimits,
) -> Result<SceneInstancePersistence, PersistenceCodecError> {
    if bytes.len() > limits.max_bytes {
        return Err(PersistenceCodecError::ResourceLimit("snapshot bytes"));
    }
    let mut reader = Reader::new(bytes);
    if reader.take(4)? != MAGIC {
        return Err(PersistenceCodecError::BadMagic);
    }
    let version = reader.u16()?;
    if version != VERSION {
        return Err(PersistenceCodecError::UnsupportedVersion(version));
    }
    let template = reader.guid()?;
    let instance = read_optional_id(&mut reader)?;
    let property_count = reader.count("properties", limits.max_properties)?;
    let mut properties = Vec::with_capacity(property_count);
    for _ in 0..property_count {
        let local = LocalEntityId::new(reader.u32()?);
        let persistent = read_optional_id(&mut reader)?;
        let component = ComponentTypeId::from_raw(reader.u64()?);
        let path = read_path(&mut reader, limits)?;
        let value = read_value(&mut reader, limits)?;
        properties.push(PersistedPropertyOverride {
            local,
            persistent,
            component,
            path,
            value,
        });
    }
    let structural_count = reader.count("structural", limits.max_structural)?;
    let mut structural = Vec::with_capacity(structural_count);
    for _ in 0..structural_count {
        structural.push(read_structural(&mut reader, limits)?);
    }
    if !reader.is_empty() {
        return Err(PersistenceCodecError::TrailingBytes);
    }
    Ok(SceneInstancePersistence {
        template,
        instance,
        properties,
        structural,
    })
}

fn write_optional_id(writer: &mut Writer, value: Option<PersistentId>) {
    match value {
        Some(id) => {
            writer.u8(1);
            writer.u128(id.to_u128());
        }
        None => writer.u8(0),
    }
}
fn read_optional_id(
    reader: &mut Reader<'_>,
) -> Result<Option<PersistentId>, PersistenceCodecError> {
    match reader.u8()? {
        0 => Ok(None),
        1 => Ok(Some(PersistentId::from_u128(reader.u128()?))),
        _ => Err(PersistenceCodecError::InvalidValue("optional id")),
    }
}

fn write_path(writer: &mut Writer, path: &FieldPath) -> Result<(), PersistenceCodecError> {
    writer.count(path.len(), "path segments")?;
    for segment in path.segments() {
        match segment {
            FieldSegment::Name(name) => {
                writer.u8(0);
                writer.string(name)?;
            }
            FieldSegment::Index(index) => {
                writer.u8(1);
                writer.u32(*index);
            }
        }
    }
    Ok(())
}
fn read_path(
    reader: &mut Reader<'_>,
    limits: PersistenceDecodeLimits,
) -> Result<FieldPath, PersistenceCodecError> {
    let count = reader.count("path segments", limits.max_path_segments)?;
    let mut segments = Vec::with_capacity(count);
    for _ in 0..count {
        segments.push(match reader.u8()? {
            0 => FieldSegment::Name(reader.string(limits.max_name_bytes)?.into_boxed_str()),
            1 => FieldSegment::Index(reader.u32()?),
            _ => return Err(PersistenceCodecError::InvalidValue("field segment")),
        });
    }
    Ok(FieldPath::from_segments(segments))
}

fn write_value(writer: &mut Writer, value: &OverrideValue) -> Result<(), PersistenceCodecError> {
    match value {
        OverrideValue::Bool(value) => {
            writer.u8(0);
            writer.u8(u8::from(*value));
        }
        OverrideValue::I64(value) => {
            writer.u8(1);
            writer.bytes(&value.to_le_bytes());
        }
        OverrideValue::U64(value) => {
            writer.u8(2);
            writer.u64(*value);
        }
        OverrideValue::F32(value) => {
            if !value.is_finite() {
                return Err(PersistenceCodecError::InvalidValue("non-finite f32"));
            }
            writer.u8(3);
            writer.bytes(&value.to_bits().to_le_bytes());
        }
        OverrideValue::F64(value) => {
            if !value.is_finite() {
                return Err(PersistenceCodecError::InvalidValue("non-finite f64"));
            }
            writer.u8(4);
            writer.bytes(&value.to_bits().to_le_bytes());
        }
        OverrideValue::Str(value) => {
            writer.u8(5);
            writer.string(value)?;
        }
        OverrideValue::Bytes(value) => {
            writer.u8(6);
            writer.payload(value)?;
        }
    }
    Ok(())
}
fn read_value(
    reader: &mut Reader<'_>,
    limits: PersistenceDecodeLimits,
) -> Result<OverrideValue, PersistenceCodecError> {
    Ok(match reader.u8()? {
        0 => match reader.u8()? {
            0 => OverrideValue::Bool(false),
            1 => OverrideValue::Bool(true),
            _ => return Err(PersistenceCodecError::InvalidValue("bool")),
        },
        1 => OverrideValue::I64(i64::from_le_bytes(reader.array()?)),
        2 => OverrideValue::U64(reader.u64()?),
        3 => {
            let value = f32::from_bits(u32::from_le_bytes(reader.array()?));
            if !value.is_finite() {
                return Err(PersistenceCodecError::InvalidValue("non-finite f32"));
            }
            OverrideValue::F32(value)
        }
        4 => {
            let value = f64::from_bits(u64::from_le_bytes(reader.array()?));
            if !value.is_finite() {
                return Err(PersistenceCodecError::InvalidValue("non-finite f64"));
            }
            OverrideValue::F64(value)
        }
        5 => OverrideValue::Str(reader.string(limits.max_value_bytes)?),
        6 => OverrideValue::Bytes(reader.payload(limits.max_value_bytes)?),
        _ => return Err(PersistenceCodecError::InvalidValue("override value")),
    })
}

fn write_structural(
    writer: &mut Writer,
    operation: &StructuralOp,
) -> Result<(), PersistenceCodecError> {
    match operation {
        StructuralOp::AddEntity {
            local,
            parent,
            archetype,
            data,
        } => {
            writer.u8(0);
            writer.u32(local.index());
            writer.u32(parent.index());
            writer.count(archetype.len(), "archetype components")?;
            for component in archetype.components() {
                writer.u64(component.raw());
            }
            writer.payload(data)?;
        }
        StructuralOp::RemoveEntity { local } => {
            writer.u8(1);
            writer.u32(local.index());
        }
        StructuralOp::AddComponent {
            target,
            component,
            data,
        } => {
            writer.u8(2);
            writer.u32(target.index());
            writer.u64(component.raw());
            writer.payload(data)?;
        }
        StructuralOp::RemoveComponent { target, component } => {
            writer.u8(3);
            writer.u32(target.index());
            writer.u64(component.raw());
        }
    }
    Ok(())
}
fn read_structural(
    reader: &mut Reader<'_>,
    limits: PersistenceDecodeLimits,
) -> Result<StructuralOp, PersistenceCodecError> {
    Ok(match reader.u8()? {
        0 => {
            let local = LocalEntityId::new(reader.u32()?);
            let parent = LocalEntityId::new(reader.u32()?);
            let count = reader.count("archetype components", limits.max_archetype_components)?;
            let mut components = Vec::with_capacity(count);
            for _ in 0..count {
                components.push(ComponentTypeId::from_raw(reader.u64()?));
            }
            StructuralOp::AddEntity {
                local,
                parent,
                archetype: ArchetypeSignature::new(components),
                data: reader.payload(limits.max_value_bytes)?,
            }
        }
        1 => StructuralOp::RemoveEntity {
            local: LocalEntityId::new(reader.u32()?),
        },
        2 => StructuralOp::AddComponent {
            target: LocalEntityId::new(reader.u32()?),
            component: ComponentTypeId::from_raw(reader.u64()?),
            data: reader.payload(limits.max_value_bytes)?,
        },
        3 => StructuralOp::RemoveComponent {
            target: LocalEntityId::new(reader.u32()?),
            component: ComponentTypeId::from_raw(reader.u64()?),
        },
        _ => return Err(PersistenceCodecError::InvalidValue("structural operation")),
    })
}

struct Writer(Vec<u8>);
impl Writer {
    fn bytes(&mut self, value: &[u8]) {
        self.0.extend_from_slice(value);
    }
    fn u8(&mut self, value: u8) {
        self.0.push(value);
    }
    fn u16(&mut self, value: u16) {
        self.bytes(&value.to_le_bytes());
    }
    fn u32(&mut self, value: u32) {
        self.bytes(&value.to_le_bytes());
    }
    fn u64(&mut self, value: u64) {
        self.bytes(&value.to_le_bytes());
    }
    fn u128(&mut self, value: u128) {
        self.bytes(&value.to_le_bytes());
    }
    fn guid(&mut self, value: StableGuid) {
        self.u128(value.to_u128());
    }
    fn count(&mut self, value: usize, kind: &'static str) -> Result<(), PersistenceCodecError> {
        self.u32(u32::try_from(value).map_err(|_| PersistenceCodecError::ResourceLimit(kind))?);
        Ok(())
    }
    fn string(&mut self, value: &str) -> Result<(), PersistenceCodecError> {
        self.count(value.len(), "string bytes")?;
        self.bytes(value.as_bytes());
        Ok(())
    }
    fn payload(&mut self, value: &[u8]) -> Result<(), PersistenceCodecError> {
        self.count(value.len(), "payload bytes")?;
        self.bytes(value);
        Ok(())
    }
}
struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}
impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }
    fn take(&mut self, len: usize) -> Result<&'a [u8], PersistenceCodecError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(PersistenceCodecError::Truncated)?;
        if end > self.bytes.len() {
            return Err(PersistenceCodecError::Truncated);
        }
        let value = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(value)
    }
    fn array<const N: usize>(&mut self) -> Result<[u8; N], PersistenceCodecError> {
        self.take(N)?
            .try_into()
            .map_err(|_| PersistenceCodecError::Truncated)
    }
    fn u8(&mut self) -> Result<u8, PersistenceCodecError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, PersistenceCodecError> {
        Ok(u16::from_le_bytes(self.array()?))
    }
    fn u32(&mut self) -> Result<u32, PersistenceCodecError> {
        Ok(u32::from_le_bytes(self.array()?))
    }
    fn u64(&mut self) -> Result<u64, PersistenceCodecError> {
        Ok(u64::from_le_bytes(self.array()?))
    }
    fn u128(&mut self) -> Result<u128, PersistenceCodecError> {
        Ok(u128::from_le_bytes(self.array()?))
    }
    fn guid(&mut self) -> Result<StableGuid, PersistenceCodecError> {
        Ok(StableGuid::from_u128(self.u128()?))
    }
    fn count(&mut self, kind: &'static str, limit: usize) -> Result<usize, PersistenceCodecError> {
        let value =
            usize::try_from(self.u32()?).map_err(|_| PersistenceCodecError::ResourceLimit(kind))?;
        if value > limit {
            return Err(PersistenceCodecError::ResourceLimit(kind));
        }
        Ok(value)
    }
    fn string(&mut self, limit: usize) -> Result<String, PersistenceCodecError> {
        let bytes = self.payload(limit)?;
        String::from_utf8(bytes).map_err(|_| PersistenceCodecError::InvalidValue("utf-8"))
    }
    fn payload(&mut self, limit: usize) -> Result<Vec<u8>, PersistenceCodecError> {
        let len = self.count("payload bytes", limit)?;
        Ok(self.take(len)?.to_vec())
    }
    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn persistence_snapshot_round_trips_all_record_kinds() {
        let mut path = FieldPath::new();
        path.push(FieldSegment::Name("transform".into()));
        path.push(FieldSegment::Index(2));
        let snapshot = SceneInstancePersistence {
            template: StableGuid::from_u128(7),
            instance: Some(PersistentId::from_u128(8)),
            properties: vec![PersistedPropertyOverride {
                local: LocalEntityId::new(1),
                persistent: None,
                component: ComponentTypeId::from_raw(9),
                path,
                value: OverrideValue::Str("hello".into()),
            }],
            structural: vec![
                StructuralOp::AddEntity {
                    local: LocalEntityId::reserved(0),
                    parent: LocalEntityId::new(0),
                    archetype: ArchetypeSignature::new(vec![ComponentTypeId::from_raw(1)]),
                    data: vec![1, 2, 3],
                },
                StructuralOp::AddComponent {
                    target: LocalEntityId::new(0),
                    component: ComponentTypeId::from_raw(2),
                    data: vec![4],
                },
                StructuralOp::RemoveComponent {
                    target: LocalEntityId::new(0),
                    component: ComponentTypeId::from_raw(3),
                },
                StructuralOp::RemoveEntity {
                    local: LocalEntityId::new(4),
                },
            ],
        };
        let encoded = encode_persistence(&snapshot).unwrap();
        assert_eq!(
            decode_persistence(&encoded, PersistenceDecodeLimits::default()).unwrap(),
            snapshot
        );
    }

    #[test]
    fn empty_snapshot_matches_canonical_little_endian_wire_vector() {
        let snapshot = SceneInstancePersistence {
            template: StableGuid::from_u128(0x1716_1514_1312_1110_0f0e_0d0c_0b0a_0908),
            instance: None,
            properties: Vec::new(),
            structural: Vec::new(),
        };
        let expected = [
            b'P', b'R', b'S', b'I', // magic
            1, 0, // wire version
            8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, // GUID LE
            0,  // no instance id
            0, 0, 0, 0, // no property rows
            0, 0, 0, 0, // no structural operations
        ];

        let encoded = encode_persistence(&snapshot).unwrap();
        assert_eq!(encoded, expected);
        assert_eq!(
            decode_persistence(&expected, PersistenceDecodeLimits::default()).unwrap(),
            snapshot
        );
    }

    #[test]
    fn persistence_codec_rejects_truncation_and_budget_overrun() {
        let snapshot = SceneInstancePersistence {
            template: StableGuid::from_u128(1),
            instance: None,
            properties: Vec::new(),
            structural: Vec::new(),
        };
        let encoded = encode_persistence(&snapshot).unwrap();
        assert_eq!(
            decode_persistence(
                &encoded[..encoded.len() - 1],
                PersistenceDecodeLimits::default()
            ),
            Err(PersistenceCodecError::Truncated)
        );
        let limits = PersistenceDecodeLimits {
            max_bytes: encoded.len() - 1,
            ..PersistenceDecodeLimits::default()
        };
        assert_eq!(
            decode_persistence(&encoded, limits),
            Err(PersistenceCodecError::ResourceLimit("snapshot bytes"))
        );
    }
}
