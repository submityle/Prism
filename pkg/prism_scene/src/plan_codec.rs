//! Deterministic binary codec for [`InstantiationPlan`](crate::InstantiationPlan).
//!
//! This is the runtime-facing wire contract for a baked plan.  It stores only
//! validated structural data and opaque component column bytes; decoding never
//! reinterprets those bytes as Rust values.  A materializer remains responsible
//! for component-specific decoding at the ECS boundary.

use alloc::vec::Vec;

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
    /// The host could not reserve memory for the decoded representation.
    AllocationFailed,
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
    let mut groups = checked_vec(groups_len, &reader, 12)?;
    let mut total_entities = 0usize;
    for _ in 0..groups_len {
        let signature_len = reader.count("components", limits.max_components_per_group)?;
        let mut signature_components = checked_vec(signature_len, &reader, 8)?;
        for _ in 0..signature_len {
            push_checked(
                &mut signature_components,
                ComponentTypeId::from_raw(reader.u64()?),
            )?;
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
        let mut locals = checked_vec(locals_len, &reader, 4)?;
        for _ in 0..locals_len {
            push_checked(&mut locals, LocalEntityId::new(reader.u32()?))?;
        }

        let columns_len = reader.count("components", limits.max_components_per_group)?;
        let mut columns = checked_vec(columns_len, &reader, 12)?;
        for _ in 0..columns_len {
            let component = ComponentTypeId::from_raw(reader.u64()?);
            let byte_len = reader.count("column bytes", limits.max_column_bytes)?;
            let source = reader.take(byte_len)?;
            let mut blob = Vec::new();
            blob.try_reserve_exact(byte_len)
                .map_err(|_| PlanCodecError::AllocationFailed)?;
            blob.extend_from_slice(source);
            push_checked(
                &mut columns,
                Column::new(component, blob.into_boxed_slice()),
            )?;
        }
        push_checked(
            &mut groups,
            ArchetypeGroup::new(
                signature,
                locals.into_boxed_slice(),
                columns.into_boxed_slice(),
            ),
        )?;
    }

    let slots_len = reader.count("field slots", limits.max_field_slots)?;
    let mut slots = checked_vec(slots_len, &reader, 16)?;
    for _ in 0..slots_len {
        push_checked(
            &mut slots,
            FieldSlot {
                local: LocalEntityId::new(reader.u32()?),
                group: reader.u32()?,
                column: reader.u32()?,
                byte_offset: reader.u32()?,
            },
        )?;
    }

    let hierarchy_len = reader.count("hierarchy", limits.max_hierarchy_edges)?;
    let mut hierarchy = checked_vec(hierarchy_len, &reader, 8)?;
    for _ in 0..hierarchy_len {
        push_checked(
            &mut hierarchy,
            (
                LocalEntityId::new(reader.u32()?),
                LocalEntityId::new(reader.u32()?),
            ),
        )?;
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

    fn require_min_bytes(&self, count: usize, width: usize) -> Result<(), PlanCodecError> {
        let required = count
            .checked_mul(width)
            .ok_or(PlanCodecError::LengthOverflow)?;
        let remaining = self.bytes.len().saturating_sub(self.offset);
        if required > remaining {
            return Err(PlanCodecError::Truncated);
        }
        Ok(())
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

fn checked_vec<T>(
    capacity: usize,
    reader: &Reader<'_>,
    minimum_wire_width: usize,
) -> Result<Vec<T>, PlanCodecError> {
    reader.require_min_bytes(capacity, minimum_wire_width)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| PlanCodecError::AllocationFailed)?;
    Ok(values)
}

fn push_checked<T>(values: &mut Vec<T>, value: T) -> Result<(), PlanCodecError> {
    values
        .try_reserve(1)
        .map_err(|_| PlanCodecError::AllocationFailed)?;
    values.push(value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{decode_plan, encode_plan, PlanCodecError, PlanDecodeLimits};
    use crate::{ArchetypeGroup, ArchetypeSignature, Column, FieldSlot, InstantiationPlan};
    use crate::{ComponentTypeId, LocalEntityId};
    use alloc::{boxed::Box, vec};

    const ENTITY_COUNT_OFFSET: usize = 6;
    const GROUPS_LEN_OFFSET: usize = 14;
    const FIRST_GROUP_SIGNATURE_LEN_OFFSET: usize = 18;
    const FIRST_GROUP_SIGNATURE_OFFSET: usize = 22;
    const FIRST_GROUP_LOCALS_LEN_OFFSET: usize = 30;
    const FIRST_GROUP_COLUMNS_LEN_OFFSET: usize = 42;
    const FIRST_GROUP_COLUMN_COMPONENT_OFFSET: usize = 46;
    const FIRST_GROUP_COLUMN_BYTES_LEN_OFFSET: usize = 54;
    const SLOTS_LEN_OFFSET: usize = 62;
    const HIERARCHY_LEN_OFFSET: usize = 82;

    fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
        bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

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

    fn plan_with_two_components() -> InstantiationPlan {
        let first = ComponentTypeId::from_raw(7);
        let second = ComponentTypeId::from_raw(11);
        InstantiationPlan {
            entity_count: 2,
            groups: Box::new([ArchetypeGroup::new(
                ArchetypeSignature::new(vec![second, first]),
                Box::new([LocalEntityId::new(0), LocalEntityId::new(1)]),
                Box::new([
                    Column::new(first, vec![1, 2].into_boxed_slice()),
                    Column::new(second, vec![3, 4].into_boxed_slice()),
                ]),
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
    fn rejects_every_strict_wire_prefix_as_truncated() {
        let bytes = encode_plan(&plan()).unwrap();
        for length in 0..bytes.len() {
            assert_eq!(
                decode_plan(&bytes[..length], PlanDecodeLimits::default()),
                Err(PlanCodecError::Truncated),
                "strict prefix of length {length} must be truncated"
            );
        }
    }

    #[test]
    fn rejects_trailing_bytes() {
        let bytes = encode_plan(&plan()).unwrap();
        let mut trailing = bytes.clone();
        trailing.extend_from_slice(&[0, 0xff, 0x7f]);
        assert_eq!(
            decode_plan(&trailing, PlanDecodeLimits::default()),
            Err(PlanCodecError::TrailingBytes)
        );
    }

    #[test]
    fn rejects_bad_magic_and_unsupported_version_before_payload_decode() {
        let bytes = encode_plan(&plan()).unwrap();
        let mut bad_magic = bytes.clone();
        bad_magic[0] ^= 0xff;
        assert_eq!(
            decode_plan(&bad_magic, PlanDecodeLimits::default()),
            Err(PlanCodecError::BadMagic)
        );

        let mut version = bytes;
        put_u16(&mut version, 4, u16::MAX);
        assert_eq!(
            decode_plan(&version, PlanDecodeLimits::default()),
            Err(PlanCodecError::UnsupportedVersion(u16::MAX))
        );
    }

    #[test]
    fn byte_and_column_length_limits_accept_exact_boundary_and_reject_overflow() {
        let bytes = encode_plan(&plan()).unwrap();

        assert!(decode_plan(
            &bytes,
            PlanDecodeLimits {
                max_bytes: bytes.len(),
                max_column_bytes: 4,
                ..PlanDecodeLimits::default()
            }
        )
        .is_ok());

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

        let mut overlong_column = bytes;
        put_u32(
            &mut overlong_column,
            FIRST_GROUP_COLUMN_BYTES_LEN_OFFSET,
            u32::MAX,
        );
        assert_eq!(
            decode_plan(
                &overlong_column,
                PlanDecodeLimits {
                    max_column_bytes: 4,
                    ..PlanDecodeLimits::default()
                }
            ),
            Err(PlanCodecError::ResourceLimit("column bytes"))
        );
    }

    #[test]
    fn every_wire_count_limit_accepts_exact_boundary_and_rejects_one_more() {
        let bytes = encode_plan(&plan()).unwrap();

        assert!(decode_plan(
            &bytes,
            PlanDecodeLimits {
                max_groups: 1,
                max_components_per_group: 1,
                max_field_slots: 1,
                max_hierarchy_edges: 1,
                ..PlanDecodeLimits::default()
            }
        )
        .is_ok());

        let mut groups = bytes.clone();
        put_u32(&mut groups, GROUPS_LEN_OFFSET, 2);
        assert_eq!(
            decode_plan(
                &groups,
                PlanDecodeLimits {
                    max_groups: 1,
                    ..PlanDecodeLimits::default()
                }
            ),
            Err(PlanCodecError::ResourceLimit("groups"))
        );

        let mut columns = bytes.clone();
        put_u32(&mut columns, FIRST_GROUP_COLUMNS_LEN_OFFSET, 2);
        assert_eq!(
            decode_plan(
                &columns,
                PlanDecodeLimits {
                    max_components_per_group: 1,
                    ..PlanDecodeLimits::default()
                }
            ),
            Err(PlanCodecError::ResourceLimit("components"))
        );

        let mut signature = bytes.clone();
        put_u32(&mut signature, FIRST_GROUP_SIGNATURE_LEN_OFFSET, 2);
        assert_eq!(
            decode_plan(
                &signature,
                PlanDecodeLimits {
                    max_components_per_group: 1,
                    ..PlanDecodeLimits::default()
                }
            ),
            Err(PlanCodecError::ResourceLimit("components"))
        );

        let mut slots = bytes.clone();
        put_u32(&mut slots, SLOTS_LEN_OFFSET, 2);
        assert_eq!(
            decode_plan(
                &slots,
                PlanDecodeLimits {
                    max_field_slots: 1,
                    ..PlanDecodeLimits::default()
                }
            ),
            Err(PlanCodecError::ResourceLimit("field slots"))
        );

        let mut hierarchy = bytes;
        put_u32(&mut hierarchy, HIERARCHY_LEN_OFFSET, 2);
        assert_eq!(
            decode_plan(
                &hierarchy,
                PlanDecodeLimits {
                    max_hierarchy_edges: 1,
                    ..PlanDecodeLimits::default()
                }
            ),
            Err(PlanCodecError::ResourceLimit("hierarchy"))
        );
    }

    #[test]
    fn rejects_sparse_large_counts_before_allocating_the_declared_vectors() {
        let mut bytes = encode_plan(&plan()).unwrap();
        put_u32(&mut bytes, GROUPS_LEN_OFFSET, 1_000_000);
        assert_eq!(
            decode_plan(&bytes, PlanDecodeLimits::default()),
            Err(PlanCodecError::Truncated)
        );
    }

    #[test]
    fn entity_count_and_entity_list_limits_are_checked_at_their_boundaries() {
        let bytes = encode_plan(&plan()).unwrap();
        let exact = PlanDecodeLimits {
            max_entities: 2,
            ..PlanDecodeLimits::default()
        };
        assert!(decode_plan(&bytes, exact).is_ok());

        let mut over_budget_entity_count = bytes.clone();
        put_u32(&mut over_budget_entity_count, ENTITY_COUNT_OFFSET, 3);
        assert_eq!(
            decode_plan(&over_budget_entity_count, exact),
            Err(PlanCodecError::ResourceLimit("entities"))
        );

        let mut over_budget_locals = bytes;
        put_u32(&mut over_budget_locals, FIRST_GROUP_LOCALS_LEN_OFFSET, 3);
        assert_eq!(
            decode_plan(&over_budget_locals, exact),
            Err(PlanCodecError::ResourceLimit("entities"))
        );
    }

    #[test]
    fn rejects_unsorted_and_duplicate_signature_wire_values() {
        let encoded = encode_plan(&plan_with_two_components()).unwrap();

        let mut unsorted = encoded.clone();
        put_u64(&mut unsorted, FIRST_GROUP_SIGNATURE_OFFSET, 11);
        put_u64(&mut unsorted, FIRST_GROUP_SIGNATURE_OFFSET + 8, 7);
        assert_eq!(
            decode_plan(&unsorted, PlanDecodeLimits::default()),
            Err(PlanCodecError::InvalidValue(
                "unsorted or duplicate signature"
            ))
        );

        let mut duplicate = encoded;
        put_u64(&mut duplicate, FIRST_GROUP_SIGNATURE_OFFSET + 8, 7);
        assert_eq!(
            decode_plan(&duplicate, PlanDecodeLimits::default()),
            Err(PlanCodecError::InvalidValue(
                "unsorted or duplicate signature"
            ))
        );
    }

    #[test]
    fn rejects_column_component_that_is_absent_from_signature() {
        let bytes = encode_plan(&plan()).unwrap();
        let mut malformed = bytes;
        put_u64(&mut malformed, FIRST_GROUP_COLUMN_COMPONENT_OFFSET, 99);
        assert!(matches!(
            decode_plan(&malformed, PlanDecodeLimits::default()),
            Err(PlanCodecError::InvalidPlan(
                crate::PlanError::ColumnNotInSignature {
                    group: 0,
                    component
                }
            )) if component == ComponentTypeId::from_raw(99)
        ));
    }

    #[test]
    fn rejects_wire_entity_count_mismatch_after_budget_checks() {
        let mut malformed = encode_plan(&plan()).unwrap();
        put_u32(&mut malformed, ENTITY_COUNT_OFFSET, 3);
        assert!(matches!(
            decode_plan(&malformed, PlanDecodeLimits::default()),
            Err(PlanCodecError::InvalidPlan(
                crate::PlanError::LocalCountMismatch {
                    expected: 3,
                    found: 2
                }
            ))
        ));
    }
}
