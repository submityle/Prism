//! Safe, deterministic materialization of opaque scene columns (M1).
//!
//! A column is never reinterpreted as a Rust allocation. Each registered
//! component supplies a fixed row layout plus a safe decoder/remapper adapter.
//! The whole plan is validated and decoded before the first world write.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::mem::{align_of, size_of};

use crate::identity::LocalEntityId;
use crate::instantiate::SceneAttachError;
use crate::instantiate::SceneSpawn;
use crate::overrides::ComponentTypeId;
use crate::plan::{FieldSlot, InstantiationPlan, PlanError};
use crate::remap::RemapTable;
use crate::residency::ResidencyTier;
use prism_asset::StableGuid;
use prism_ecs::entity::Entity;
use prism_ecs::hierarchy::ChildOf;
use prism_ecs::world::World;
use prism_reflect::{FromReflect, GetTypeRegistration, Reflect, StableTypeId, TypeRegistry, Typed};

/// Fixed row ABI for one materialized component column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowLayout {
    /// Bytes consumed by the safe decoder for one row.
    pub size: usize,
    /// Logical row alignment in the baked column format. No pointer casts are
    /// performed; this is checked as an ABI contract, not relied on for safety.
    pub alignment: usize,
}

/// Checked resource budgets for opaque-column materialization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaterializeLimits {
    /// Maximum reserved entities in one materialization transaction.
    pub max_entities: usize,
    /// Maximum component columns in the plan.
    pub max_columns: usize,
    /// Maximum total row visits across all columns.
    pub max_rows: usize,
    /// Maximum encoded column bytes inspected by one transaction.
    pub max_blob_bytes: usize,
}

impl Default for MaterializeLimits {
    fn default() -> Self {
        Self {
            max_entities: 1_000_000,
            max_columns: 1_000_000,
            max_rows: 4_000_000,
            max_blob_bytes: 256 * 1024 * 1024,
        }
    }
}

impl RowLayout {
    /// Describes a fixed-size row format.
    #[must_use]
    pub const fn new(size: usize, alignment: usize) -> Self {
        Self { size, alignment }
    }
}

/// A validated entity-reference slot and its resolved instance entity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RemappedEntityField {
    /// Offset in the source row, containing a little-endian `LocalEntityId`.
    pub byte_offset: usize,
    /// Decoded local id before remapping.
    pub local: LocalEntityId,
    /// Runtime entity in this instance.
    pub entity: Entity,
}

/// User-defined safe adapter error code; stable within the application's
/// materializer contract and preserved in [`MaterializeError`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaterializeFailure(pub u32);

type DecodeFn = Box<
    dyn Fn(&[u8], &[RemappedEntityField]) -> Result<Box<dyn Reflect>, MaterializeFailure>
        + Send
        + Sync,
>;

/// Registered safe decoder for one component type.
pub struct ComponentMaterializer {
    layout: RowLayout,
    reflected_type: StableTypeId,
    decode: DecodeFn,
}

impl ComponentMaterializer {
    /// Creates an adapter for `T`.
    ///
    /// `decode` must decode exactly one row into the reflected component value;
    /// `remap` must replace the fields identified by the validated entity slots
    /// using the supplied full-generation [`Entity`] handles. Both callbacks
    /// operate on typed values/slices and require no unsafe code.
    pub fn new<T>(
        layout: RowLayout,
        decode: impl Fn(&[u8]) -> Result<T, MaterializeFailure> + Send + Sync + 'static,
        remap: impl Fn(&mut T, &[RemappedEntityField]) -> Result<(), MaterializeFailure>
            + Send
            + Sync
            + 'static,
    ) -> Self
    where
        T: Reflect + 'static,
    {
        Self {
            layout,
            reflected_type: StableTypeId::of_type::<T>(),
            decode: Box::new(move |row, fields| {
                let mut value = decode(row)?;
                remap(&mut value, fields)?;
                Ok(Box::new(value))
            }),
        }
    }

    /// Builds the common no-entity-reference adapter from a safe row decoder.
    pub fn without_entity_fields<T>(
        layout: RowLayout,
        decode: impl Fn(&[u8]) -> Result<T, MaterializeFailure> + Send + Sync + 'static,
    ) -> Self
    where
        T: Reflect + 'static,
    {
        Self::new(layout, decode, |_, fields| {
            if fields.is_empty() {
                Ok(())
            } else {
                Err(MaterializeFailure(u32::MAX))
            }
        })
    }
}

/// Deterministic registry from scene component ids to safe row adapters.
#[derive(Default)]
pub struct MaterializerRegistry {
    entries: Vec<(ComponentTypeId, ComponentMaterializer)>,
}

impl MaterializerRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a component adapter; duplicate ids are rejected without
    /// replacing the existing decoder.
    pub fn register(
        &mut self,
        component: ComponentTypeId,
        materializer: ComponentMaterializer,
    ) -> Result<(), MaterializeError> {
        match self.entries.binary_search_by_key(&component, |(id, _)| *id) {
            Ok(_) => Err(MaterializeError::DuplicateRegistration { component }),
            Err(index) => {
                self.entries.insert(index, (component, materializer));
                Ok(())
            }
        }
    }

    /// Registers a reflected `T` using Prism's canonical little-endian binary
    /// reflection codec. This path is for components without `Entity` fields;
    /// entity-valued components use [`ComponentMaterializer::new`].
    pub fn register_reflect_binary<T>(
        &mut self,
        component: ComponentTypeId,
        layout: RowLayout,
    ) -> Result<(), MaterializeError>
    where
        T: prism_ecs::component::Component
            + Reflect
            + FromReflect
            + GetTypeRegistration
            + Clone
            + Typed,
    {
        let mut types = TypeRegistry::new();
        types.register::<T>();
        let types = alloc::sync::Arc::new(types);
        self.register(
            component,
            ComponentMaterializer {
                layout,
                reflected_type: StableTypeId::of_type::<T>(),
                decode: Box::new(move |bytes, fields| {
                    if !fields.is_empty() {
                        return Err(MaterializeFailure(u32::MAX));
                    }
                    let decoded =
                        prism_reflect::from_binary(bytes, &types, <T as Typed>::type_info())
                            .map_err(|_| MaterializeFailure(u32::MAX))?;
                    let value = T::from_reflect(&*decoded).ok_or(MaterializeFailure(u32::MAX))?;
                    Ok(Box::new(value))
                }),
            },
        )
    }

    fn get(&self, id: ComponentTypeId) -> Option<&ComponentMaterializer> {
        self.entries
            .binary_search_by_key(&id, |(key, _)| *key)
            .ok()
            .map(|i| &self.entries[i].1)
    }
}

/// Why a complete scene plan could not be materialized.
#[expect(
    missing_docs,
    reason = "Each structured error is documented by its variant"
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterializeError {
    /// The plan exceeded a caller-controlled preflight budget.
    ResourceLimit {
        /// Budget family that was exceeded.
        kind: &'static str,
        /// Configured maximum.
        limit: usize,
        /// Value declared by the plan.
        found: usize,
    },
    /// Structural plan validation failed.
    InvalidPlan(PlanError),
    /// The target world has pending reservations owned by another caller.
    OutstandingReservations,
    /// No adapter is registered for a column's component.
    MissingRegistration {
        component: ComponentTypeId,
        group: u32,
    },
    /// Registry key and reflected concrete type do not agree with the world.
    WorldTypeNotRegistered { component: ComponentTypeId },
    /// Duplicate materializer id was registered.
    DuplicateRegistration { component: ComponentTypeId },
    /// Layout uses zero size, non-power-of-two alignment, or an incompatible stride.
    InvalidRowLayout {
        component: ComponentTypeId,
        group: u32,
    },
    /// Column length is not exactly rows multiplied by its row size.
    ColumnByteLength {
        component: ComponentTypeId,
        group: u32,
        expected: usize,
        found: usize,
    },
    /// A `FieldSlot` is outside its row or not aligned for `LocalEntityId` bytes.
    FieldSlotBounds { slot: FieldSlot, row_size: usize },
    /// Multiple slots target the same byte range in the same column.
    DuplicateFieldSlot { slot: FieldSlot },
    /// Slot bytes do not encode an existing local entity mapping.
    UnknownFieldTarget {
        slot: FieldSlot,
        local: LocalEntityId,
    },
    /// Adapter rejected the row. No world writes have occurred for decode errors.
    Decode {
        component: ComponentTypeId,
        group: u32,
        row: u32,
        code: u32,
    },
    /// A decoded value's reflected type did not match the registered world component.
    ReflectedTypeMismatch {
        component: ComponentTypeId,
        group: u32,
        row: u32,
    },
    /// An ECS insertion rejected an already validated reflected value.
    WorldWriteRejected {
        component: ComponentTypeId,
        local: LocalEntityId,
    },
    /// Dynamic hierarchy was cyclic.
    HierarchyCycle {
        child: LocalEntityId,
        parent: LocalEntityId,
    },
}

/// Fully instantiated scene ownership result.
#[derive(Clone, Debug, PartialEq)]
pub struct MaterializedScene {
    /// Root entity, when the plan is non-empty.
    pub root: Option<Entity>,
    /// The instance-local local-to-runtime entity map.
    pub spawn: SceneSpawn,
}

impl MaterializedScene {
    /// Creates the root [`SceneInstance`] anchor for this materialized plan.
    #[must_use]
    pub fn scene_instance(
        &self,
        template: StableGuid,
        tier: ResidencyTier,
    ) -> crate::SceneInstance {
        self.spawn.scene_instance(template, tier)
    }

    /// Attaches the live root anchor through the materialized-scene boundary.
    pub fn attach_instance(
        &self,
        world: &mut World,
        template: StableGuid,
        tier: ResidencyTier,
    ) -> Result<Entity, SceneAttachError> {
        self.spawn.attach_instance(world, template, tier)
    }
}

/// Materializes all registered columns and hierarchy for a plan.
///
/// Preflight validates every column, every slot and decodes every row before
/// reserving entities or mutating the World. Materialization order is canonical:
/// sorted registry ids, then plan group/column/row order; entities are reserved
/// in local-id order. A returned failure after writes rolls back owned entities.
pub fn materialize_plan(
    world: &mut World,
    plan: &InstantiationPlan,
    registry: &MaterializerRegistry,
) -> Result<MaterializedScene, MaterializeError> {
    materialize_plan_with_limits(world, plan, registry, MaterializeLimits::default())
}

/// Materializes an opaque-column plan after checked budget preflight.
pub fn materialize_plan_with_limits(
    world: &mut World,
    plan: &InstantiationPlan,
    registry: &MaterializerRegistry,
    limits: MaterializeLimits,
) -> Result<MaterializedScene, MaterializeError> {
    plan.validate().map_err(MaterializeError::InvalidPlan)?;
    preflight_limits(plan, limits)?;
    if world.entities().has_outstanding_reservations() {
        return Err(MaterializeError::OutstandingReservations);
    }

    struct Pending {
        component: ComponentTypeId,
        local: LocalEntityId,
        value: Box<dyn Reflect>,
    }
    let mut slots_by_column = Vec::new();

    for (group_index, group) in plan.groups.iter().enumerate() {
        for column_index in 0..group.columns.len() {
            let column = &group.columns[column_index];
            let adapter =
                registry
                    .get(column.component)
                    .ok_or(MaterializeError::MissingRegistration {
                        component: column.component,
                        group: group_index as u32,
                    })?;
            let layout = adapter.layout;
            if layout.size == 0
                || layout.alignment == 0
                || !layout.alignment.is_power_of_two()
                || !layout.size.is_multiple_of(layout.alignment)
            {
                return Err(MaterializeError::InvalidRowLayout {
                    component: column.component,
                    group: group_index as u32,
                });
            }
            let expected =
                group
                    .len()
                    .checked_mul(layout.size)
                    .ok_or(MaterializeError::InvalidRowLayout {
                        component: column.component,
                        group: group_index as u32,
                    })?;
            if column.blob.len() != expected {
                return Err(MaterializeError::ColumnByteLength {
                    component: column.component,
                    group: group_index as u32,
                    expected,
                    found: column.blob.len(),
                });
            }
            let reflected = world.reflect_component_id(adapter.reflected_type).ok_or(
                MaterializeError::WorldTypeNotRegistered {
                    component: column.component,
                },
            )?;
            let _ = reflected;
            let mut slots: Vec<FieldSlot> = plan
                .entity_valued_fields
                .iter()
                .copied()
                .filter(|slot| {
                    slot.group as usize == group_index && slot.column as usize == column_index
                })
                .collect();
            slots.sort_unstable_by_key(|slot| (slot.byte_offset, slot.local));
            for (i, slot) in slots.iter().enumerate() {
                let offset = slot.byte_offset as usize;
                if !offset.is_multiple_of(align_of::<u32>())
                    || offset
                        .checked_add(size_of::<u32>())
                        .is_none_or(|end| end > layout.size)
                {
                    return Err(MaterializeError::FieldSlotBounds {
                        slot: *slot,
                        row_size: layout.size,
                    });
                }
                if i > 0
                    && slots[i - 1].local == slot.local
                    && slots[i - 1].byte_offset == slot.byte_offset
                {
                    return Err(MaterializeError::DuplicateFieldSlot { slot: *slot });
                }
            }
            for slot in &slots {
                let row_index = group
                    .locals
                    .iter()
                    .position(|local| *local == slot.local)
                    .expect("plan validation checked the slot owner");
                let start = row_index * layout.size;
                let raw = u32::from_le_bytes(
                    column.blob
                        [start + slot.byte_offset as usize..start + slot.byte_offset as usize + 4]
                        .try_into()
                        .expect("slot range was validated"),
                );
                let target = LocalEntityId::new(raw);
                if plan_local_entity(plan, target).is_none() {
                    return Err(MaterializeError::UnknownFieldTarget {
                        slot: *slot,
                        local: target,
                    });
                }
            }
            slots_by_column.push(slots);
        }
    }

    // Reserve deterministically only after all structural, layout, registration,
    // and slot-target checks have completed. Any later decode failure rolls back
    // these empty reservations before returning.
    let mut locals: Vec<_> = plan
        .groups
        .iter()
        .flat_map(|g| g.locals.iter().copied())
        .collect();
    locals.sort_unstable();
    let mut remap = RemapTable::with_base_capacity(locals.len());
    for _ in &locals {
        remap.push_base(world.entities().reserve_entity());
    }
    world.flush_reserved();

    let decode_result = (|| {
        let mut pending = Vec::new();
        let mut slot_index = 0usize;
        for (group_index, group) in plan.groups.iter().enumerate() {
            for column in group.columns.iter() {
                let adapter = registry
                    .get(column.component)
                    .expect("preflight checked registration");
                let slots = &slots_by_column[slot_index];
                slot_index += 1;
                for (row_index, local) in group.locals.iter().copied().enumerate() {
                    let start = row_index * adapter.layout.size;
                    let row = &column.blob[start..start + adapter.layout.size];
                    let mut remapped = Vec::new();
                    for slot in slots.iter().filter(|slot| slot.local == local) {
                        let offset = slot.byte_offset as usize;
                        let raw = u32::from_le_bytes(
                            row[offset..offset + 4]
                                .try_into()
                                .expect("validated slot width"),
                        );
                        let target = LocalEntityId::new(raw);
                        let entity =
                            remap
                                .get(target)
                                .ok_or(MaterializeError::UnknownFieldTarget {
                                    slot: *slot,
                                    local: target,
                                })?;
                        remapped.push(RemappedEntityField {
                            byte_offset: offset,
                            local: target,
                            entity,
                        });
                    }
                    let value = (adapter.decode)(row, &remapped).map_err(|error| {
                        MaterializeError::Decode {
                            component: column.component,
                            group: group_index as u32,
                            row: row_index as u32,
                            code: error.0,
                        }
                    })?;
                    if StableTypeId::of_path(value.type_name()) != adapter.reflected_type {
                        return Err(MaterializeError::ReflectedTypeMismatch {
                            component: column.component,
                            group: group_index as u32,
                            row: row_index as u32,
                        });
                    }
                    pending.push(Pending {
                        component: column.component,
                        local,
                        value,
                    });
                }
            }
        }
        Ok::<_, MaterializeError>(pending)
    })();
    let pending = match decode_result {
        Ok(pending) => pending,
        Err(error) => {
            rollback(world, &remap);
            return Err(error);
        }
    };

    for &local in &locals {
        let entity = remap.get(local).expect("locals and remap are aligned");
        world.spawn_at(entity, ());
    }
    for item in pending {
        let entity = remap.get(item.local).expect("validated local has mapping");
        if !world.insert_reflect(entity, &*item.value) {
            rollback(world, &remap);
            return Err(MaterializeError::WorldWriteRejected {
                component: item.component,
                local: item.local,
            });
        }
    }
    world.register_child_of();
    for &(child, parent) in plan.hierarchy.iter() {
        let child_entity = remap.get(child).expect("validated child");
        let parent_entity = remap.get(parent).expect("validated parent");
        world.add_relation::<ChildOf>(child_entity, parent_entity);
    }
    let root = remap.get(LocalEntityId::new(0));
    Ok(MaterializedScene {
        root,
        spawn: SceneSpawn::from_remap(remap),
    })
}

fn preflight_limits(
    plan: &InstantiationPlan,
    limits: MaterializeLimits,
) -> Result<(), MaterializeError> {
    let entities = plan.entity_count as usize;
    if entities > limits.max_entities {
        return Err(MaterializeError::ResourceLimit {
            kind: "entities",
            limit: limits.max_entities,
            found: entities,
        });
    }

    let mut columns = 0usize;
    let mut rows = 0usize;
    let mut blob_bytes = 0usize;
    for group in plan.groups.iter() {
        columns =
            columns
                .checked_add(group.columns.len())
                .ok_or(MaterializeError::ResourceLimit {
                    kind: "columns",
                    limit: limits.max_columns,
                    found: usize::MAX,
                })?;
        rows = rows
            .checked_add(group.len().saturating_mul(group.columns.len()))
            .ok_or(MaterializeError::ResourceLimit {
                kind: "rows",
                limit: limits.max_rows,
                found: usize::MAX,
            })?;
        let group_blob_bytes = group
            .columns
            .iter()
            .try_fold(0usize, |total, column| total.checked_add(column.blob.len()))
            .unwrap_or(usize::MAX);
        blob_bytes =
            blob_bytes
                .checked_add(group_blob_bytes)
                .ok_or(MaterializeError::ResourceLimit {
                    kind: "blob_bytes",
                    limit: limits.max_blob_bytes,
                    found: usize::MAX,
                })?;
    }
    if columns > limits.max_columns {
        return Err(MaterializeError::ResourceLimit {
            kind: "columns",
            limit: limits.max_columns,
            found: columns,
        });
    }
    if rows > limits.max_rows {
        return Err(MaterializeError::ResourceLimit {
            kind: "rows",
            limit: limits.max_rows,
            found: rows,
        });
    }
    if blob_bytes > limits.max_blob_bytes {
        return Err(MaterializeError::ResourceLimit {
            kind: "blob_bytes",
            limit: limits.max_blob_bytes,
            found: blob_bytes,
        });
    }
    Ok(())
}

fn plan_local_entity(plan: &InstantiationPlan, local: LocalEntityId) -> Option<LocalEntityId> {
    if local.is_reserved() {
        return None;
    }
    plan.groups
        .iter()
        .any(|group| group.locals.contains(&local))
        .then_some(local)
}

fn rollback(world: &mut World, remap: &RemapTable) {
    for (_, entity) in remap.entities() {
        world.despawn(entity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overrides::ArchetypeSignature;
    use crate::plan::{ArchetypeGroup, Column};
    use alloc::boxed::Box;
    use alloc::sync::Arc;
    use prism_ecs::component::Component;
    use prism_reflect::Reflect;

    #[derive(Reflect, Clone, Debug, PartialEq, Eq)]
    struct Payload {
        value: u32,
        target_bits: u64,
    }
    impl Component for Payload {}

    fn row(value: u32, target: u32) -> [u8; 8] {
        let mut bytes = [0; 8];
        bytes[..4].copy_from_slice(&value.to_le_bytes());
        bytes[4..].copy_from_slice(&target.to_le_bytes());
        bytes
    }

    fn plan(blob: Vec<u8>, count: u32, slots: Vec<FieldSlot>) -> InstantiationPlan {
        let component = ComponentTypeId::of::<Payload>();
        InstantiationPlan {
            entity_count: count,
            groups: Box::new([ArchetypeGroup::new(
                ArchetypeSignature::new(alloc::vec![component]),
                (0..count).map(LocalEntityId::new).collect(),
                Box::new([Column::new(component, Arc::<[u8]>::from(blob))]),
            )]),
            entity_valued_fields: slots.into_boxed_slice(),
            hierarchy: (1..count)
                .map(|local| (LocalEntityId::new(local), LocalEntityId::new(0)))
                .collect(),
            reserved_local_base: LocalEntityId::RESERVED_BASE,
        }
    }

    fn registry(world: &mut World, fail_on: Option<u32>) -> MaterializerRegistry {
        world.register_reflect_component::<Payload>();
        let mut registry = MaterializerRegistry::new();
        registry
            .register(
                ComponentTypeId::of::<Payload>(),
                ComponentMaterializer::new(
                    RowLayout::new(8, 4),
                    move |bytes| {
                        let value = u32::from_le_bytes(bytes[..4].try_into().unwrap());
                        if fail_on == Some(value) {
                            return Err(MaterializeFailure(9));
                        }
                        Ok(Payload {
                            value,
                            target_bits: 0,
                        })
                    },
                    |value, fields| {
                        if let Some(field) = fields.first() {
                            value.target_bits = field.entity.to_bits();
                        }
                        Ok(())
                    },
                ),
            )
            .unwrap();
        registry
    }

    #[test]
    fn materializes_rows_and_remaps_slots_to_runtime_entities() {
        let mut world = World::new();
        let registry = registry(&mut world, None);
        let slots = (0..2)
            .map(|local| FieldSlot {
                local: LocalEntityId::new(local),
                group: 0,
                column: 0,
                byte_offset: 4,
            })
            .collect();
        let scene = materialize_plan(
            &mut world,
            &plan([row(11, 1), row(22, 0)].concat(), 2, slots),
            &registry,
        )
        .unwrap();
        let first = scene.spawn.resolve(LocalEntityId::new(0)).unwrap();
        let second = scene.spawn.resolve(LocalEntityId::new(1)).unwrap();
        assert_eq!(world.get::<Payload>(first).unwrap().value, 11);
        assert_eq!(
            world.get::<Payload>(first).unwrap().target_bits,
            second.to_bits()
        );
        assert_eq!(world.get::<Payload>(second).unwrap().value, 22);
        assert_eq!(
            world.get::<Payload>(second).unwrap().target_bits,
            first.to_bits()
        );
        let root = scene
            .attach_instance(
                &mut world,
                StableGuid::from_u128(0xabc),
                ResidencyTier::Interactive,
            )
            .unwrap();
        assert_eq!(
            world.get::<crate::SceneInstance>(root).unwrap().template,
            StableGuid::from_u128(0xabc)
        );
    }

    #[test]
    fn decode_error_rolls_back_reserved_entities_before_component_writes() {
        let mut world = World::new();
        let registry = registry(&mut world, Some(22));
        let before = world.entities().len();
        let error = materialize_plan(
            &mut world,
            &plan([row(11, 0), row(22, 0)].concat(), 2, Vec::new()),
            &registry,
        )
        .unwrap_err();
        assert!(matches!(error, MaterializeError::Decode { code: 9, .. }));
        assert_eq!(world.entities().len(), before);
    }

    #[test]
    fn missing_registration_is_reported_before_reservation() {
        let mut world = World::new();
        let before = world.entities().len();
        let error = materialize_plan(
            &mut world,
            &plan(row(7, 0).to_vec(), 1, Vec::new()),
            &MaterializerRegistry::new(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            MaterializeError::MissingRegistration { .. }
        ));
        assert_eq!(world.entities().len(), before);
    }

    #[test]
    fn budget_preflight_rejects_before_reserving_world_entities() {
        let mut world = World::new();
        let registry = registry(&mut world, None);
        let before = world.entities().len();
        let error = materialize_plan_with_limits(
            &mut world,
            &plan([row(7, 0), row(8, 0)].concat(), 2, Vec::new()),
            &registry,
            MaterializeLimits {
                max_entities: 1,
                ..MaterializeLimits::default()
            },
        )
        .unwrap_err();
        assert_eq!(
            error,
            MaterializeError::ResourceLimit {
                kind: "entities",
                limit: 1,
                found: 2,
            }
        );
        assert_eq!(world.entities().len(), before);
    }

    #[test]
    fn row_and_blob_budgets_are_checked_before_decode() {
        let mut world = World::new();
        let registry = registry(&mut world, None);
        let scene_plan = plan([row(7, 0), row(8, 0)].concat(), 2, Vec::new());

        let row_error = materialize_plan_with_limits(
            &mut world,
            &scene_plan,
            &registry,
            MaterializeLimits {
                max_rows: 1,
                ..MaterializeLimits::default()
            },
        )
        .unwrap_err();
        assert!(matches!(
            row_error,
            MaterializeError::ResourceLimit {
                kind: "rows",
                limit: 1,
                found: 2
            }
        ));

        let blob_error = materialize_plan_with_limits(
            &mut world,
            &scene_plan,
            &registry,
            MaterializeLimits {
                max_blob_bytes: 1,
                ..MaterializeLimits::default()
            },
        )
        .unwrap_err();
        assert!(matches!(
            blob_error,
            MaterializeError::ResourceLimit {
                kind: "blob_bytes",
                limit: 1,
                found: 16
            }
        ));
        assert_eq!(world.entities().len(), 0);
    }
}
