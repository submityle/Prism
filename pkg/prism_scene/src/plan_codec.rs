//! Deterministic binary codec for [`InstantiationPlan`](crate::InstantiationPlan).
//!
//! This is the runtime-facing wire contract for a baked plan.  It stores only
//! validated structural data and opaque component column bytes; decoding never
//! reinterprets those bytes as Rust values.  A materializer remains responsible
//! for component-specific decoding at the ECS boundary.

use alloc::{sync::Arc, vec::Vec};

use crate::{
    ArchetypeGroup, ArchetypeSignature, Column, ComponentTypeId, FieldSlot, InstantiationPlan,
    LocalEntityId, PlanError,
};

const MAGIC: &[u8; 4] = b"PRSP";
const VERSION: u16 = 1;

/// Bounds applied while decoding an untrusted baked plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanDecodeLimits {
    /// Maximum encoded input size in bytes.
    pub max_bytes: usize,
    /// Maximum number of archetype groups.
    pub max_groups: usize,
    /// Maximum total entities listed across all groups.
    pub max_entities: usize,
    /// Maximum components in one signature or column list.
    pub max_components_per_group: usize,
    /// Maximum bytes in one component column.
    pub max_column_bytes: usize,
    /// Maximum entity-valued field slots.
    pub max_field_slots: usize,
    /// Maximum hierarchy edges.
    pub max_hierarchy_edges: usize,
}

impl Default for PlanDecodeLimits {
    fn default() -> Self {
        Self {
            max_bytes: 256 * 1024 * 1024,
            max_groups: 1_048_576,
            max_entities: 16_777_216,
            max_components_per_group: 4096,
            max_column_bytes: 256 * 1024 * 1024,
            max_field_slots: 16_777_216,
            max_hierarchy_edges: 16_777_216,
        }
    }
}

/// A malformed or unsafe plan wire representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanCodecError {
    /// The input does not contain the plan magic.
    BadMagic,
    /// The encoded version is not supported.
    UnsupportedVersion(u16),
    /// The input ended before a complete value was available.
    Truncated,
    /// Bytes remained after the plan was decoded.
    TrailingBytes,
    /// A count or byte length exceeded a configured limit.
    ResourceLimit(&'static str),
    /// A count could not be represented as a platform `usize`.
    LengthOverflow,
    /// The wire value is structurally invalid before plan validation.
    InvalidValue(&'static str),
    /// The decoded plan violates a runtime invariant.
    InvalidPlan(PlanError),
}

/// Encodes a validated plan in the stable version-1 binary format.
pub fn encode_plan(plan: &InstantiationPlan) -> Result<Vec<u8>, PlanCodecError> {
    plan.validate().map_err(PlanCodecError::InvalidPlan)?;
    let mut writer = Writer(Vec::new());
    writer.bytes(MAGIC);
    writer.u16(VERSION);
    writer.u32(plan.entity_count);
    writer.u32(plan.reserved_local_base);
    writer.u32(checked_len(plan.groups.len(), "groups")?);
    for group in &plan.groups {
        writer.u32(checked_len(group.signature.len(), "components")?);
        for component in group.signature.components() {
            writer.u64(component.raw());
        }
        writer.u32(checked_len(group.locals.len(), "entities")?);
        for local in &group.locals {
            writer.u32(local.index());
        }
        writer.u32(checked_len(group.columns.len(), "components")?);
        for column in &group.columns {
            writer.u64(column.component.raw());
            writer.u32(checked_len(column.blob.len(), "column bytes")?);
            writer.bytes(&column.blob);
        }
    }
    writer.u32(checked_len(plan.entity_valued_fields.len(), "field slots")?);
    for slot in &plan.entity_valued_fields {
        writer.u32(slot.local.index());
        writer.u32(slot.group);
        writer.u32(slot.column);
        writer.u32(slot.byte_offset);
    }
    writer.u32(checked_len(plan.hierarchy.len(), "hierarchy")?);
    for &(child, parent) in &plan.hierarchy {
        writer.u32(child.index());
        writer.u32(parent.index());
    }
    Ok(writer.0)
}

fn checked_len(len: usize, kind: &'static str) -> Result<u32, PlanCodecError> {
    u32::try_from(len).map_err(|_| PlanCodecError::ResourceLimit(kind))
}

/// Decodes and validates a plan from the stable version-1 binary format.
pub fn decode_plan(
    bytes: &[u8],
    limits: PlanDecodeLimits,
) -> Result<InstantiationPlan, PlanCodecError> {
    if bytes.len() > limits.max_bytes {
        return Err(PlanCodecError::ResourceLimit("bytes"));
    }
    let mut reader = Reader { bytes, offset: 0 };
    if reader.take(4)? != MAGIC {
        return Err(PlanCodecError::BadMagic);
    }
    let version = reader.u16()?;
    if version != VERSION {
        return Err(PlanCodecError::UnsupportedVersion(version));
    }
    let entity_count = reader.u32()?;
    let entity_count_usize =
        usize::try_from(entity_count).map_err(|_| PlanCodecError::LengthOverflow)?;
    if entity_count_usize > limits.max_entities {
        return Err(PlanCodecError::ResourceLimit("entities"));
    }
    let reserved_local_base = reader.u32()?;
    let groups_len = reader.count("groups", limits.max_groups)?;
    let mut groups = Vec::with_capacity(groups_len);
    let mut total_entities = 0usize;
    for _ in 0..groups_len {
        let signature_len = reader.count("components", limits.max_components_per_group)?;
        let mut signature_components = Vec::with_capacity(signature_len);
        for _ in 0..signature_len {
            signature_components.push(ComponentTypeId::from_raw(reader.u64()?));
        }
        let signature = ArchetypeSignature::new(signature_components.clone());
        if signature.components() != signature_components.as_slice() {
            return Err(PlanCodecError::InvalidValue(
                "unsorted or duplicate signature",
            ));
        }

        let locals_len = reader.count("entities", limits.max_entities)?;
        total_entities = total_entities
            .checked_add(locals_len)
            .ok_or(PlanCodecError::LengthOverflow)?;
        if total_entities > limits.max_entities {
            return Err(PlanCodecError::ResourceLimit("entities"));
        }
        let mut locals = Vec::with_capacity(locals_len);
        for _ in 0..locals_len {
            locals.push(LocalEntityId::new(reader.u32()?));
        }

        let columns_len = reader.count("components", limits.max_components_per_group)?;
        let mut columns = Vec::with_capacity(columns_len);
        for _ in 0..columns_len {
            let component = ComponentTypeId::from_raw(reader.u64()?);
            let byte_len = reader.count("column bytes", limits.max_column_bytes)?;
            let blob: Arc<[u8]> = reader.take(byte_len)?.to_vec().into_boxed_slice().into();
            columns.push(Column::new(component, blob));
        }
        groups.push(ArchetypeGroup::new(
            signature,
            locals.into_boxed_slice(),
            columns.into_boxed_slice(),
        ));
    }

    let slots_len = reader.count("field slots", limits.max_field_slots)?;
    let mut slots = Vec::with_capacity(slots_len);
    for _ in 0..slots_len {
        slots.push(FieldSlot {
            local: LocalEntityId::new(reader.u32()?),
            group: reader.u32()?,
            column: reader.u32()?,
            byte_offset: reader.u32()?,
        });
    }

    let hierarchy_len = reader.count("hierarchy", limits.max_hierarchy_edges)?;
    let mut hierarchy = Vec::with_capacity(hierarchy_len);
    for _ in 0..hierarchy_len {
        hierarchy.push((
            LocalEntityId::new(reader.u32()?),
            LocalEntityId::new(reader.u32()?),
        ));
    }
    if reader.offset != bytes.len() {
        return Err(PlanCodecError::TrailingBytes);
    }

    let plan = InstantiationPlan {
        entity_count,
        groups: groups.into_boxed_slice(),
        entity_valued_fields: slots.into_boxed_slice(),
        hierarchy: hierarchy.into_boxed_slice(),
        reserved_local_base,
    };
    plan.validate().map_err(PlanCodecError::InvalidPlan)?;
    Ok(plan)
}

struct Writer(Vec<u8>);

impl Writer {
    fn bytes(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
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
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], PlanCodecError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(PlanCodecError::LengthOverflow)?;
        if end > self.bytes.len() {
            return Err(PlanCodecError::Truncated);
        }
        let value = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(value)
    }

    fn count(&mut self, kind: &'static str, limit: usize) -> Result<usize, PlanCodecError> {
        let raw = self.u32()?;
        let value = usize::try_from(raw).map_err(|_| PlanCodecError::LengthOverflow)?;
        if value > limit {
            return Err(PlanCodecError::ResourceLimit(kind));
        }
        Ok(value)
    }

    fn u16(&mut self) -> Result<u16, PlanCodecError> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().expect("checked width"),
        ))
    }

    fn u32(&mut self) -> Result<u32, PlanCodecError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("checked width"),
        ))
    }

    fn u64(&mut self) -> Result<u64, PlanCodecError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("checked width"),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{decode_plan, encode_plan, PlanCodecError, PlanDecodeLimits};
    use crate::{ArchetypeGroup, ArchetypeSignature, Column, FieldSlot, InstantiationPlan};
    use crate::{ComponentTypeId, LocalEntityId};
    use alloc::{boxed::Box, vec};

    fn plan() -> InstantiationPlan {
        let component = ComponentTypeId::from_raw(7);
        InstantiationPlan {
            entity_count: 2,
            groups: Box::new([ArchetypeGroup::new(
                ArchetypeSignature::new(vec![component]),
                Box::new([LocalEntityId::new(0), LocalEntityId::new(1)]),
                Box::new([Column::new(component, vec![1, 2, 3, 4].into_boxed_slice())]),
            )]),
            entity_valued_fields: Box::new([FieldSlot {
                local: LocalEntityId::new(0),
                group: 0,
                column: 0,
                byte_offset: 0,
            }]),
            hierarchy: Box::new([(LocalEntityId::new(1), LocalEntityId::new(0))]),
            reserved_local_base: LocalEntityId::RESERVED_BASE,
        }
    }

    #[test]
    fn round_trip_preserves_plan_and_opaque_column_bytes() {
        let value = plan();
        let bytes = encode_plan(&value).unwrap();
        assert_eq!(
            decode_plan(&bytes, PlanDecodeLimits::default()).unwrap(),
            value
        );
    }

    #[test]
    fn rejects_truncation_trailing_bytes_and_bad_version() {
        let bytes = encode_plan(&plan()).unwrap();
        assert_eq!(
            decode_plan(&bytes[..bytes.len() - 1], PlanDecodeLimits::default()),
            Err(PlanCodecError::Truncated)
        );
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert_eq!(
            decode_plan(&trailing, PlanDecodeLimits::default()),
            Err(PlanCodecError::TrailingBytes)
        );
        let mut version = bytes;
        version[4..6].copy_from_slice(&9u16.to_le_bytes());
        assert_eq!(
            decode_plan(&version, PlanDecodeLimits::default()),
            Err(PlanCodecError::UnsupportedVersion(9))
        );
    }

    #[test]
    fn limits_apply_before_allocation() {
        let bytes = encode_plan(&plan()).unwrap();
        let limits = PlanDecodeLimits {
            max_bytes: bytes.len() - 1,
            ..PlanDecodeLimits::default()
        };
        assert_eq!(
            decode_plan(&bytes, limits),
            Err(PlanCodecError::ResourceLimit("bytes"))
        );
        let limits = PlanDecodeLimits {
            max_column_bytes: 3,
            ..PlanDecodeLimits::default()
        };
        assert_eq!(
            decode_plan(&bytes, limits),
            Err(PlanCodecError::ResourceLimit("column bytes"))
        );
    }

    #[test]
    fn invalid_plan_is_rejected_after_wire_decode() {
        let bytes = encode_plan(&plan()).unwrap();
        let mut malformed = bytes;
        let last = malformed.len() - 8;
        malformed[last..last + 4].copy_from_slice(&0u32.to_le_bytes());
        assert!(matches!(
            decode_plan(&malformed, PlanDecodeLimits::default()),
            Err(PlanCodecError::InvalidPlan(_))
        ));
    }
}
