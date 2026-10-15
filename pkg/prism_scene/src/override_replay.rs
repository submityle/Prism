//! Transactional replay of live scene overrides.
//!
//! Structural payloads are decoded through an explicit typed registry. The
//! decoder boundary never casts bytes, and all structural payloads are decoded
//! and checked before the first world mutation. `AddEntity` uses a canonical
//! row format: components are listed in `ArchetypeSignature` order and each
//! payload is prefixed by a little-endian `u32` byte length. `AddComponent`
//! payloads are passed directly to the registered decoder.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::instantiate::SceneSpawn;
use crate::overrides::{
    ArchetypeSignature, ComponentTypeId, OverrideSet, PropertyOverride, StructuralOp,
};
use prism_ecs::component::Component;
use prism_ecs::entity::Entity;
use prism_ecs::world::World;
use prism_reflect::{FromReflect, GetTypeRegistration, Reflect, StableTypeId};

/// Failure while replaying a live override set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OverrideReplayError {
    /// The compatibility entry point was asked to replay structural operations.
    /// Use [`replay_overrides_with_registry`] with a mutable [`SceneSpawn`].
    StructuralOperationsUnsupported,
    /// No decoder was registered for a serialized component.
    MissingDecoder {
        /// The serialized component id.
        component: ComponentTypeId,
    },
    /// The decoder rejected a serialized component payload.
    Decode {
        /// The serialized component id.
        component: ComponentTypeId,
        /// Application-defined stable decoder code.
        code: u32,
    },
    /// A decoder returned a reflected value for a different concrete type.
    DecodedTypeMismatch {
        /// The serialized component id.
        component: ComponentTypeId,
    },
    /// An `AddEntity` row was not framed exactly as required by this module.
    InvalidEntityData {
        /// The local id of the entity being added.
        local: crate::identity::LocalEntityId,
        /// Stable reason code (`1` means truncated frame, `2` means trailing bytes).
        code: u32,
    },
    /// A structural target is not part of the scene spawn or was already removed.
    StructuralTargetNotOwned(crate::identity::LocalEntityId),
    /// The operation addressed a structurally-added entity before it had a
    /// live ECS handle and therefore cannot report a runtime entity handle.
    PendingEntity(crate::identity::LocalEntityId),
    /// A structural add used a local id that is already mapped.
    StructuralEntityAlreadyExists(crate::identity::LocalEntityId),
    /// A structural operation found a component state different from its preflight.
    StructuralStateMismatch {
        /// The live target entity.
        entity: Entity,
        /// The component involved in the mismatch.
        component: ComponentTypeId,
    },
    /// The ECS rejected a preflighted structural write.
    StructuralWriteRejected {
        /// The live target entity.
        entity: Entity,
        /// The component involved in the rejected write.
        component: ComponentTypeId,
    },
    /// A relation or entity operation could not be committed.
    StructuralEntityWriteRejected(crate::identity::LocalEntityId),
    /// A committed structural transaction could not be restored.
    StructuralRollbackFailed,
    /// A property target is not owned by the scene spawn.
    TargetNotOwned(crate::identity::LocalEntityId),
    /// The target does not carry the reflected component.
    MissingComponent {
        /// The live target entity.
        entity: Entity,
        /// The component requested by the override.
        component: ComponentTypeId,
    },
    /// The property value/path could not be applied to its staged component.
    Property(crate::property_apply::PropertyApplyError),
    /// The reflected component rejected the staged value during commit.
    WriteRejected {
        /// The live target entity.
        entity: Entity,
        /// The component whose write was rejected.
        component: ComponentTypeId,
    },
    /// Restoring a previously committed component failed.
    RollbackFailed {
        /// The live target entity whose original value could not be restored.
        entity: Entity,
        /// The component whose original value could not be restored.
        component: ComponentTypeId,
    },
}

/// Stable application-defined error returned by a structural decoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StructuralDecodeFailure(pub u32);

type DecodeFn =
    Box<dyn Fn(&[u8]) -> Result<Box<dyn Reflect>, StructuralDecodeFailure> + Send + Sync>;

/// One typed decoder entry in a [`StructuralDecoderRegistry`].
pub struct StructuralDecoder {
    reflected_type: StableTypeId,
    decode: DecodeFn,
}

impl StructuralDecoder {
    /// Creates a decoder for a concrete reflected component type.
    pub fn typed<T>(
        decode: impl Fn(&[u8]) -> Result<T, StructuralDecodeFailure> + Send + Sync + 'static,
    ) -> Self
    where
        T: Component + Reflect + FromReflect + GetTypeRegistration + Clone,
    {
        Self {
            reflected_type: StableTypeId::of_type::<T>(),
            decode: Box::new(move |bytes| Ok(Box::new(decode(bytes)?))),
        }
    }

    fn decode(&self, bytes: &[u8]) -> Result<Box<dyn Reflect>, StructuralDecodeFailure> {
        (self.decode)(bytes)
    }
}

/// Deterministic registry for safe typed structural component decoders.
#[derive(Default)]
pub struct StructuralDecoderRegistry {
    entries: Vec<(ComponentTypeId, StructuralDecoder)>,
}

impl StructuralDecoderRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a typed decoder. Duplicate component ids are rejected without
    /// replacing the original entry.
    pub fn register<T>(
        &mut self,
        component: ComponentTypeId,
        decode: impl Fn(&[u8]) -> Result<T, StructuralDecodeFailure> + Send + Sync + 'static,
    ) -> Result<(), StructuralDecoderRegistrationError>
    where
        T: Component + Reflect + FromReflect + GetTypeRegistration + Clone,
    {
        let decoder = StructuralDecoder::typed(decode);
        match self.entries.binary_search_by_key(&component, |(id, _)| *id) {
            Ok(_) => Err(StructuralDecoderRegistrationError::Duplicate { component }),
            Err(index) => {
                self.entries.insert(index, (component, decoder));
                Ok(())
            }
        }
    }

    fn get(&self, component: ComponentTypeId) -> Option<&StructuralDecoder> {
        self.entries
            .binary_search_by_key(&component, |(id, _)| *id)
            .ok()
            .map(|index| &self.entries[index].1)
    }
}

/// Failure while building a [`StructuralDecoderRegistry`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructuralDecoderRegistrationError {
    /// The component already has a decoder.
    Duplicate {
        /// The duplicated component id.
        component: ComponentTypeId,
    },
}

struct StagedComponent {
    entity: Entity,
    component: ComponentTypeId,
    original: Box<dyn Reflect>,
    properties: Vec<PropertyOverride>,
    staged: Option<Box<dyn Reflect>>,
}

/// Replays property overrides through the compatibility API.
///
/// Structural operations require [`replay_overrides_with_registry`] because a
/// successful structural replay must update the scene's local-to-live mapping.
/// This function keeps the existing signature and behavior for property-only
/// callers.
pub fn replay_overrides(
    world: &mut World,
    spawn: &SceneSpawn,
    overrides: &OverrideSet,
) -> Result<(), OverrideReplayError> {
    if !overrides.structural().is_empty() {
        return Err(OverrideReplayError::StructuralOperationsUnsupported);
    }

    replay_property_overrides(world, spawn, overrides.properties())
}

/// Replays property and structural overrides as one transaction.
///
/// The transaction has three phases: decode and validate every structural row,
/// preflight every property against an isolated reflected state, then commit.
/// A failed reflected write is undone from cloned values; newly created
/// entities, added components and relations are removed in reverse order. Entity
/// removals are committed last, after all fallible writes have been preflighted.
/// The caller's [`SceneSpawn`] is changed only after the world commit succeeds.
pub fn replay_overrides_with_registry(
    world: &mut World,
    spawn: &mut SceneSpawn,
    overrides: &OverrideSet,
    registry: &StructuralDecoderRegistry,
) -> Result<(), OverrideReplayError> {
    if overrides.is_empty() {
        return Ok(());
    }

    if world.entities().has_outstanding_reservations() {
        return Err(OverrideReplayError::StructuralEntityWriteRejected(
            crate::identity::LocalEntityId::new(u32::MAX),
        ));
    }

    let mut next_spawn = spawn.clone();
    let mut state = build_virtual_state(world, spawn)?;
    let initial_reserved = spawn.remap().reserved_len() as u32;
    let mut prepared = Vec::with_capacity(overrides.structural().len());
    let mut added_count = 0u32;

    for op in overrides.structural() {
        match op {
            StructuralOp::AddEntity {
                local,
                parent,
                archetype,
                data,
            } => {
                if !local.is_reserved()
                    || local.index()
                        != crate::identity::LocalEntityId::RESERVED_BASE
                            .saturating_add(initial_reserved)
                            .saturating_add(added_count)
                {
                    return Err(OverrideReplayError::StructuralEntityAlreadyExists(*local));
                }
                if state.contains_key(local) {
                    return Err(OverrideReplayError::StructuralEntityAlreadyExists(*local));
                }
                if !is_live_state(&state, *parent) {
                    return Err(OverrideReplayError::StructuralTargetNotOwned(*parent));
                }
                let components = decode_entity_row(registry, *local, archetype, data, world)?;
                let mut values = BTreeMap::new();
                for (component, value) in &components {
                    values.insert(*component, value.reflect_clone());
                }
                state.insert(
                    *local,
                    VirtualEntity {
                        entity: None,
                        components: values,
                    },
                );
                prepared.push(PreparedStructural::AddEntity {
                    local: *local,
                    parent: *parent,
                    components,
                });
                added_count = added_count.saturating_add(1);
            }
            StructuralOp::RemoveEntity { local } => {
                if *local == crate::identity::LocalEntityId::new(0) {
                    return Err(OverrideReplayError::StructuralEntityWriteRejected(*local));
                }
                let Some(entity) = state.remove(local) else {
                    return Err(OverrideReplayError::StructuralTargetNotOwned(*local));
                };
                if let Some(live_entity) = entity.entity {
                    let live_children = world
                        .relation_sources::<prism_ecs::hierarchy::ChildOf>(live_entity)
                        .iter()
                        .copied()
                        .find(|child| {
                            state
                                .values()
                                .any(|candidate| candidate.entity == Some(*child))
                        });
                    if live_children.is_some() {
                        return Err(OverrideReplayError::StructuralEntityWriteRejected(*local));
                    }
                }
                prepared.push(PreparedStructural::RemoveEntity { local: *local });
            }
            StructuralOp::AddComponent {
                target,
                component,
                data,
            } => {
                let Some(target_state) = state.get_mut(target) else {
                    return Err(OverrideReplayError::StructuralTargetNotOwned(*target));
                };
                if target_state.components.contains_key(component) {
                    let Some(entity) = target_state.entity else {
                        return Err(OverrideReplayError::PendingEntity(*target));
                    };
                    return Err(OverrideReplayError::StructuralStateMismatch {
                        entity,
                        component: *component,
                    });
                }
                let value = decode_component(registry, world, *component, data)?;
                target_state
                    .components
                    .insert(*component, value.reflect_clone());
                prepared.push(PreparedStructural::AddComponent {
                    target: *target,
                    component: *component,
                    value,
                });
            }
            StructuralOp::RemoveComponent { target, component } => {
                let Some(target_state) = state.get_mut(target) else {
                    return Err(OverrideReplayError::StructuralTargetNotOwned(*target));
                };
                let Some(original) = target_state.components.remove(component) else {
                    let Some(entity) = target_state.entity else {
                        return Err(OverrideReplayError::PendingEntity(*target));
                    };
                    return Err(OverrideReplayError::StructuralStateMismatch {
                        entity,
                        component: *component,
                    });
                };
                prepared.push(PreparedStructural::RemoveComponent {
                    target: *target,
                    component: *component,
                    original,
                });
            }
        }
    }

    let mut properties = Vec::<PreparedProperty>::new();
    for property in overrides.properties() {
        let Some(target) = state.get_mut(&property.target) else {
            return Err(OverrideReplayError::TargetNotOwned(property.target));
        };
        let Some(value) = target.components.get_mut(&property.component) else {
            let Some(entity) = target.entity else {
                return Err(OverrideReplayError::PendingEntity(property.target));
            };
            return Err(OverrideReplayError::MissingComponent {
                entity,
                component: property.component,
            });
        };
        crate::property_apply::apply_property_to_reflect_for_replay(value.as_mut(), property)
            .map_err(OverrideReplayError::Property)?;
        if let Some(existing) = properties
            .iter_mut()
            .find(|entry| entry.target == property.target && entry.component == property.component)
        {
            existing.value = value.reflect_clone();
        } else {
            properties.push(PreparedProperty {
                target: property.target,
                component: property.component,
                value: value.reflect_clone(),
            });
        }
    }

    let mut added_entities = Vec::<(crate::identity::LocalEntityId, Entity)>::new();
    for operation in &prepared {
        if let PreparedStructural::AddEntity { local, .. } = operation {
            added_entities.push((*local, world.entities().reserve_entity()));
        }
    }
    world.flush_reserved();

    let mut entity_map = BTreeMap::<crate::identity::LocalEntityId, Entity>::new();
    for (local, entity) in spawn.remap().entities() {
        if !spawn.is_removed(local) {
            entity_map.insert(local, entity);
        }
    }
    for &(local, entity) in &added_entities {
        entity_map.insert(local, entity);
        if !next_spawn.add_reserved_mapping(local, entity) {
            return Err(OverrideReplayError::StructuralRollbackFailed);
        }
    }
    for operation in &prepared {
        if let PreparedStructural::RemoveEntity { local } = operation
            && !next_spawn.mark_removed(*local)
        {
            return Err(OverrideReplayError::StructuralRollbackFailed);
        }
    }

    let mut transaction = Transaction::default();
    for operation in &prepared {
        let result = match operation {
            PreparedStructural::AddEntity {
                local,
                parent,
                components,
            } => {
                let entity = entity_map[local];
                world.spawn_at(entity, ());
                for (component, value) in components {
                    if !world.insert_reflect(entity, value.as_ref()) {
                        return rollback_or_error(
                            world,
                            &mut transaction,
                            OverrideReplayError::StructuralWriteRejected {
                                entity,
                                component: *component,
                            },
                        );
                    }
                }
                let parent_entity = entity_map[parent];
                world.register_child_of();
                world.add_relation::<prism_ecs::hierarchy::ChildOf>(entity, parent_entity);
                transaction.relations.push((entity, parent_entity));
                transaction.created.push(entity);
                Ok(())
            }
            PreparedStructural::RemoveEntity { .. } => Ok(()),
            PreparedStructural::AddComponent {
                target,
                component,
                value,
            } => {
                let entity = entity_map[target];
                if world.insert_reflect(entity, value.as_ref()) {
                    transaction.added_components.push((entity, *component));
                    Ok(())
                } else {
                    Err(OverrideReplayError::StructuralWriteRejected {
                        entity,
                        component: *component,
                    })
                }
            }
            PreparedStructural::RemoveComponent {
                target,
                component,
                original,
            } => {
                let entity = entity_map[target];
                if world.remove_reflect(entity, StableTypeId::from_raw(component.raw())) {
                    transaction.removed_components.push((
                        entity,
                        *component,
                        original.reflect_clone(),
                    ));
                    Ok(())
                } else {
                    Err(OverrideReplayError::StructuralWriteRejected {
                        entity,
                        component: *component,
                    })
                }
            }
        };
        if let Err(error) = result {
            return rollback_or_error(world, &mut transaction, error);
        }
    }

    for property in properties {
        let entity = entity_map[&property.target];
        let Some(original) = world
            .get_reflect(entity, StableTypeId::from_raw(property.component.raw()))
            .map(Reflect::reflect_clone)
        else {
            return rollback_or_error(
                world,
                &mut transaction,
                OverrideReplayError::MissingComponent {
                    entity,
                    component: property.component,
                },
            );
        };
        if !world.insert_reflect(entity, property.value.as_ref()) {
            return rollback_or_error(
                world,
                &mut transaction,
                OverrideReplayError::StructuralWriteRejected {
                    entity,
                    component: property.component,
                },
            );
        }
        transaction
            .property_writes
            .push((entity, property.component, original));
    }

    // Removal is deliberately last: all typed decode, component writes and
    // property writes have already been proven possible at this point.
    for operation in &prepared {
        let PreparedStructural::RemoveEntity { local } = operation else {
            continue;
        };
        let entity = entity_map[local];
        if !world.despawn(entity) {
            return Err(OverrideReplayError::StructuralRollbackFailed);
        }
    }

    *spawn = next_spawn;
    Ok(())
}

fn replay_property_overrides(
    world: &mut World,
    spawn: &SceneSpawn,
    properties: &[PropertyOverride],
) -> Result<(), OverrideReplayError> {
    let mut staged = Vec::<StagedComponent>::new();
    for property in properties {
        let entity = spawn
            .resolve(property.target)
            .ok_or(OverrideReplayError::TargetNotOwned(property.target))?;
        let Some(current) =
            world.get_reflect(entity, StableTypeId::from_raw(property.component.raw()))
        else {
            return Err(OverrideReplayError::MissingComponent {
                entity,
                component: property.component,
            });
        };
        let Some(group) = staged
            .iter_mut()
            .find(|group| group.entity == entity && group.component == property.component)
        else {
            staged.push(StagedComponent {
                entity,
                component: property.component,
                original: current.reflect_clone(),
                properties: vec![property.clone()],
                staged: None,
            });
            continue;
        };
        group.properties.push(property.clone());
    }

    // Validate every group against an isolated clone before touching the world.
    for group in &mut staged {
        let Some(current) =
            world.get_reflect(group.entity, StableTypeId::from_raw(group.component.raw()))
        else {
            return Err(OverrideReplayError::MissingComponent {
                entity: group.entity,
                component: group.component,
            });
        };
        let mut probe = current.reflect_clone();
        for property in &group.properties {
            crate::property_apply::apply_property_to_reflect_for_replay(probe.as_mut(), property)
                .map_err(OverrideReplayError::Property)?;
        }
        group.staged = Some(probe);
    }

    for index in 0..staged.len() {
        let (entity, component, staged_value) = {
            let group = &mut staged[index];
            let Some(staged_value) = group.staged.take() else {
                return Err(OverrideReplayError::RollbackFailed {
                    entity: group.entity,
                    component: group.component,
                });
            };
            (group.entity, group.component, staged_value)
        };
        if !world.insert_reflect(entity, staged_value.as_ref()) {
            for previous in staged.iter().take(index) {
                if !world.insert_reflect(previous.entity, previous.original.as_ref()) {
                    return Err(OverrideReplayError::RollbackFailed {
                        entity: previous.entity,
                        component: previous.component,
                    });
                }
            }
            return Err(OverrideReplayError::WriteRejected { entity, component });
        }
    }
    Ok(())
}

struct VirtualEntity {
    entity: Option<Entity>,
    components: BTreeMap<ComponentTypeId, Box<dyn Reflect>>,
}

enum PreparedStructural {
    AddEntity {
        local: crate::identity::LocalEntityId,
        parent: crate::identity::LocalEntityId,
        components: Vec<(ComponentTypeId, Box<dyn Reflect>)>,
    },
    RemoveEntity {
        local: crate::identity::LocalEntityId,
    },
    AddComponent {
        target: crate::identity::LocalEntityId,
        component: ComponentTypeId,
        value: Box<dyn Reflect>,
    },
    RemoveComponent {
        target: crate::identity::LocalEntityId,
        component: ComponentTypeId,
        original: Box<dyn Reflect>,
    },
}

struct PreparedProperty {
    target: crate::identity::LocalEntityId,
    component: ComponentTypeId,
    value: Box<dyn Reflect>,
}

#[derive(Default)]
struct Transaction {
    created: Vec<Entity>,
    relations: Vec<(Entity, Entity)>,
    added_components: Vec<(Entity, ComponentTypeId)>,
    removed_components: Vec<(Entity, ComponentTypeId, Box<dyn Reflect>)>,
    property_writes: Vec<(Entity, ComponentTypeId, Box<dyn Reflect>)>,
}

fn build_virtual_state(
    world: &World,
    spawn: &SceneSpawn,
) -> Result<BTreeMap<crate::identity::LocalEntityId, VirtualEntity>, OverrideReplayError> {
    let mut state = BTreeMap::new();
    for (local, entity) in spawn.remap().entities() {
        if spawn.is_removed(local) {
            continue;
        }
        if !world.contains(entity)
            || world
                .entities()
                .location(entity)
                .is_none_or(prism_ecs::entity::EntityLocation::is_empty)
        {
            return Err(OverrideReplayError::StructuralTargetNotOwned(local));
        }
        let components = world
            .reflect_entity(entity)
            .into_iter()
            .map(|(type_id, value)| {
                (
                    ComponentTypeId::from_raw(type_id.value()),
                    value.reflect_clone(),
                )
            })
            .collect();
        state.insert(
            local,
            VirtualEntity {
                entity: Some(entity),
                components,
            },
        );
    }
    Ok(state)
}

fn is_live_state(
    state: &BTreeMap<crate::identity::LocalEntityId, VirtualEntity>,
    local: crate::identity::LocalEntityId,
) -> bool {
    state.contains_key(&local)
}

fn decode_component(
    registry: &StructuralDecoderRegistry,
    world: &World,
    component: ComponentTypeId,
    bytes: &[u8],
) -> Result<Box<dyn Reflect>, OverrideReplayError> {
    let Some(decoder) = registry.get(component) else {
        return Err(OverrideReplayError::MissingDecoder { component });
    };
    if world.reflect_component_id(decoder.reflected_type).is_none() {
        return Err(OverrideReplayError::DecodedTypeMismatch { component });
    }
    let value = decoder
        .decode(bytes)
        .map_err(|failure| OverrideReplayError::Decode {
            component,
            code: failure.0,
        })?;
    if StableTypeId::of_path(value.type_name()) != decoder.reflected_type {
        return Err(OverrideReplayError::DecodedTypeMismatch { component });
    }
    Ok(value)
}

fn decode_entity_row(
    registry: &StructuralDecoderRegistry,
    local: crate::identity::LocalEntityId,
    archetype: &ArchetypeSignature,
    data: &[u8],
    world: &World,
) -> Result<Vec<(ComponentTypeId, Box<dyn Reflect>)>, OverrideReplayError> {
    let mut offset = 0usize;
    let mut values = Vec::with_capacity(archetype.len());
    for &component in archetype.components() {
        let Some(length_bytes) = data.get(offset..offset.saturating_add(4)) else {
            return Err(OverrideReplayError::InvalidEntityData { local, code: 1 });
        };
        let length = u32::from_le_bytes([
            length_bytes[0],
            length_bytes[1],
            length_bytes[2],
            length_bytes[3],
        ]) as usize;
        offset = offset.saturating_add(4);
        let Some(end) = offset.checked_add(length) else {
            return Err(OverrideReplayError::InvalidEntityData { local, code: 1 });
        };
        let Some(payload) = data.get(offset..end) else {
            return Err(OverrideReplayError::InvalidEntityData { local, code: 1 });
        };
        values.push((
            component,
            decode_component(registry, world, component, payload)?,
        ));
        offset = end;
    }
    if offset != data.len() {
        return Err(OverrideReplayError::InvalidEntityData { local, code: 2 });
    }
    Ok(values)
}

fn rollback_or_error(
    world: &mut World,
    transaction: &mut Transaction,
    error: OverrideReplayError,
) -> Result<(), OverrideReplayError> {
    let mut failed = false;
    for (entity, component, original) in transaction.property_writes.drain(..).rev() {
        if !world.insert_reflect(entity, original.as_ref()) {
            failed = true;
        }
        let _ = component;
    }
    for (entity, component) in transaction.added_components.drain(..).rev() {
        if !world.remove_reflect(entity, StableTypeId::from_raw(component.raw())) {
            failed = true;
        }
    }
    for (entity, _component, original) in transaction.removed_components.drain(..).rev() {
        if !world.insert_reflect(entity, original.as_ref()) {
            failed = true;
        }
    }
    for (child, parent) in transaction.relations.drain(..).rev() {
        if !world.remove_relation::<prism_ecs::hierarchy::ChildOf>(child, parent) {
            failed = true;
        }
    }
    for entity in transaction.created.drain(..).rev() {
        if !world.despawn(entity) {
            failed = true;
        }
    }
    if failed {
        Err(OverrideReplayError::StructuralRollbackFailed)
    } else {
        Err(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::LocalEntityId;
    use crate::instantiate::{spawn_typed_plan, TypedSpawnPlan};
    use crate::overrides::OverrideValue;
    use crate::path::FieldPath;
    use crate::plan::{ArchetypeGroup, InstantiationPlan};
    use prism_ecs::component::Component;
    use prism_reflect::Reflect;

    #[derive(Clone, Debug, Reflect, PartialEq)]
    struct Settings {
        speed: u64,
        label: String,
    }
    impl Component for Settings {}

    fn plan(count: u32) -> InstantiationPlan {
        InstantiationPlan {
            entity_count: count,
            groups: if count == 0 {
                Box::new([])
            } else {
                Box::new([ArchetypeGroup::new(
                    Default::default(),
                    (0..count).map(LocalEntityId::new).collect(),
                    Box::new([]),
                )])
            },
            entity_valued_fields: Box::new([]),
            hierarchy: (1..count)
                .map(|local| (LocalEntityId::new(local), LocalEntityId::new(0)))
                .collect(),
            reserved_local_base: LocalEntityId::RESERVED_BASE,
        }
    }

    fn property(target: u32, path: &str, value: OverrideValue) -> PropertyOverride {
        PropertyOverride::new(
            LocalEntityId::new(target),
            ComponentTypeId::of::<Settings>(),
            FieldPath::parse(path).unwrap(),
            value,
        )
    }

    #[test]
    fn replays_properties_through_instance_remap() {
        let mut world = World::new();
        world.register_reflect_component::<Settings>();
        let spawn = spawn_typed_plan(
            &mut world,
            &plan(1),
            TypedSpawnPlan::new().with_bundle(
                LocalEntityId::new(0),
                Settings {
                    speed: 2,
                    label: "old".into(),
                },
            ),
        )
        .unwrap();
        let mut overrides = OverrideSet::new();
        overrides.set_property(property(0, "speed", OverrideValue::U64(9)));
        overrides.set_property(property(0, "label", OverrideValue::Str("new".into())));

        replay_overrides(&mut world, &spawn, &overrides).unwrap();
        assert_eq!(
            world
                .get::<Settings>(spawn.resolve(LocalEntityId::new(0)).unwrap())
                .unwrap(),
            &Settings {
                speed: 9,
                label: "new".into()
            }
        );
    }

    #[test]
    fn invalid_later_group_leaves_earlier_group_untouched() {
        let mut world = World::new();
        world.register_reflect_component::<Settings>();
        let spawn = spawn_typed_plan(
            &mut world,
            &plan(2),
            TypedSpawnPlan::new()
                .with_bundle(
                    LocalEntityId::new(0),
                    Settings {
                        speed: 2,
                        label: "a".into(),
                    },
                )
                .with_bundle(
                    LocalEntityId::new(1),
                    Settings {
                        speed: 3,
                        label: "b".into(),
                    },
                ),
        )
        .unwrap();
        let mut overrides = OverrideSet::new();
        overrides.set_property(property(0, "speed", OverrideValue::U64(9)));
        overrides.set_property(property(1, "speed", OverrideValue::I64(9)));

        assert!(matches!(
            replay_overrides(&mut world, &spawn, &overrides),
            Err(OverrideReplayError::Property(_))
        ));
        assert_eq!(
            world
                .get::<Settings>(spawn.resolve(LocalEntityId::new(0)).unwrap())
                .unwrap()
                .speed,
            2
        );
    }

    fn settings_decoder(bytes: &[u8]) -> Result<Settings, StructuralDecodeFailure> {
        if bytes.len() < 16 {
            return Err(StructuralDecodeFailure(1));
        }
        let speed = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let label_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let end = 16usize.saturating_add(label_len);
        let Some(label) = bytes.get(16..end) else {
            return Err(StructuralDecodeFailure(2));
        };
        let label = String::from_utf8(label.to_vec()).map_err(|_| StructuralDecodeFailure(3))?;
        Ok(Settings { speed, label })
    }

    #[test]
    fn structural_component_replay_uses_registered_decoder_and_updates_state() {
        let mut world = World::new();
        world.register_reflect_component::<Settings>();
        let spawn = spawn_typed_plan(
            &mut world,
            &plan(1),
            TypedSpawnPlan::new().with_bundle(LocalEntityId::new(0), ()),
        )
        .unwrap();
        let mut registry = StructuralDecoderRegistry::new();
        registry
            .register(ComponentTypeId::of::<Settings>(), settings_decoder)
            .unwrap();
        let mut overrides = OverrideSet::new();
        overrides.push_structural(StructuralOp::AddComponent {
            target: LocalEntityId::new(0),
            component: ComponentTypeId::of::<Settings>(),
            data: [
                9u64.to_le_bytes().as_slice(),
                3u64.to_le_bytes().as_slice(),
                b"new".as_slice(),
            ]
            .concat(),
        });

        let mut live = spawn;
        replay_overrides_with_registry(&mut world, &mut live, &overrides, &registry).unwrap();
        assert_eq!(
            world.get::<Settings>(live.resolve(LocalEntityId::new(0)).unwrap()),
            Some(&Settings {
                speed: 9,
                label: "new".into()
            })
        );
    }

    #[test]
    fn add_and_remove_entity_replay_preserves_local_mapping_invariants() {
        let mut world = World::new();
        let mut live = spawn_typed_plan(
            &mut world,
            &plan(1),
            TypedSpawnPlan::new().with_bundle(LocalEntityId::new(0), ()),
        )
        .unwrap();
        let local = LocalEntityId::reserved(0);
        let mut overrides = OverrideSet::new();
        overrides.push_structural(StructuralOp::AddEntity {
            local,
            parent: LocalEntityId::new(0),
            archetype: Default::default(),
            data: vec![],
        });
        replay_overrides_with_registry(
            &mut world,
            &mut live,
            &overrides,
            &StructuralDecoderRegistry::new(),
        )
        .unwrap();
        let added = live.resolve(local).unwrap();
        assert!(world.contains(added));
        assert!(world.has_relation::<prism_ecs::hierarchy::ChildOf>(
            added,
            live.resolve(LocalEntityId::new(0)).unwrap()
        ));

        let mut remove = OverrideSet::new();
        remove.push_structural(StructuralOp::RemoveEntity { local });
        replay_overrides_with_registry(
            &mut world,
            &mut live,
            &remove,
            &StructuralDecoderRegistry::new(),
        )
        .unwrap();
        assert!(!world.contains(added));
        assert!(live.resolve(local).is_none());
    }
}
