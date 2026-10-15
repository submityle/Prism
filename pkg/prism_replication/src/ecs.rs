//! A bounded, real `prism_ecs` replication provider.
//!
//! The transport/session layers in this crate intentionally do not own an ECS
//! world. This module is the narrow provider seam that turns stable
//! entity/component records into the actual `prism_ecs::World` operations:
//! `spawn`, reflected component insertion, reflected field updates, and
//! `despawn`. Component type identity is supplied by a manifest key and is
//! never inferred from a process-local `ComponentId`.

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::fmt;

use prism_ecs::component::Component;
use prism_ecs::entity::Entity;
use prism_ecs::world::World;
use prism_reflect::{
    decode_delta, encode_delta, from_binary_with_limits, to_binary, DeltaError, DeserializeError,
    DeserializeLimits, FieldDelta, FromReflect, GetTypeRegistration, Reflect, ReflectMut,
    SerializeError, StableTypeId, TypeInfo, TypeRegistry,
};

use crate::mapping::{
    AuthorityId, ComponentKey, EntityId, LocalEntityId, MappingEntry, MappingError,
    ReplicationMapping, Revision, SchemaDescriptor,
};

/// Bounds applied to one ECS replication provider.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EcsReplicationLimits {
    /// Maximum number of components in one lifecycle record.
    pub max_components_per_entity: usize,
    /// Maximum encoded bytes for one component payload.
    pub max_component_payload: usize,
    /// Maximum top-level fields in one reflected component.
    pub max_fields_per_component: usize,
    /// Bounds passed to the reflection binary decoder.
    pub decode: DeserializeLimits,
}

impl Default for EcsReplicationLimits {
    fn default() -> Self {
        Self {
            max_components_per_entity: 64,
            max_component_payload: 64 * 1024,
            max_fields_per_component: 128,
            decode: DeserializeLimits::new(32, 4096),
        }
    }
}

/// A component payload carried by a spawn or full component update.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentRecord {
    /// Stable manifest key for the component.
    pub component: ComponentKey,
    /// Monotonic revision for this entity/component value.
    pub revision: Revision,
    /// Reflection binary payload for the complete component value.
    pub payload: Vec<u8>,
}

/// A stable entity snapshot used to create one live ECS entity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EntitySnapshot {
    /// Stable scene/entity identity.
    pub entity: EntityId,
    /// Current authority for the records.
    pub authority: AuthorityId,
    /// Complete component values to install.
    pub components: Vec<ComponentRecord>,
}

/// A stable entity update. Missing component records are left untouched.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EntityUpdate {
    /// Stable scene/entity identity.
    pub entity: EntityId,
    /// Current authority for the records.
    pub authority: AuthorityId,
    /// Complete component values to overwrite or insert.
    pub components: Vec<ComponentRecord>,
}

/// Failure returned by the real ECS provider.
#[derive(Debug)]
pub enum EcsReplicationError {
    /// A stable entity has no mapping in this provider.
    UnknownEntity(EntityId),
    /// A spawn attempted to reuse a stable identity that is still live.
    EntityAlreadyMapped(EntityId),
    /// A lifecycle record contained no component identity with which to track
    /// the local ECS handle.
    EmptyEntitySnapshot,
    /// A lifecycle record exceeded the configured component bound.
    TooManyComponents {
        /// Number of component records supplied by the caller.
        count: usize,
        /// Configured maximum component count.
        max: usize,
    },
    /// The payload exceeded the configured byte bound.
    PayloadTooLarge {
        /// Encoded payload length supplied by the caller.
        len: usize,
        /// Configured maximum payload length.
        max: usize,
    },
    /// The reflected component shape has more top-level fields than allowed.
    TooManyFields {
        /// Number of reflected top-level fields.
        count: usize,
        /// Configured maximum field count.
        max: usize,
    },
    /// A live ECS entity carries a reflected type absent from the manifest.
    UnknownType(StableTypeId),
    /// The component key is not registered in the provider manifest.
    UnknownComponent(ComponentKey),
    /// The local ECS handle was recycled or despawned before the operation.
    LocalEntityUnavailable(LocalEntityId),
    /// The ECS world rejected an insertion or despawn.
    WorldRejected(Entity),
    /// Reflection binary encoding failed.
    Serialize(SerializeError),
    /// Reflection binary decoding failed.
    Deserialize(DeserializeError),
    /// A field delta could not be decoded or applied.
    FieldDelta(DeltaError),
    /// A reflected value could not be inserted into the target component slot.
    ReflectInsertRejected(ComponentKey),
    /// The mapping invariants rejected a lifecycle revision or identity.
    Mapping(MappingError),
}

impl fmt::Display for EcsReplicationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownEntity(id) => write!(f, "unknown replicated entity {}", id.as_u128()),
            Self::EntityAlreadyMapped(id) => {
                write!(f, "replicated entity {} is already mapped", id.as_u128())
            }
            Self::EmptyEntitySnapshot => f.write_str("replicated entity has no component record"),
            Self::TooManyComponents { count, max } => {
                write!(f, "component count {count} exceeds bound {max}")
            }
            Self::PayloadTooLarge { len, max } => {
                write!(f, "component payload {len} exceeds bound {max}")
            }
            Self::TooManyFields { count, max } => {
                write!(f, "field count {count} exceeds bound {max}")
            }
            Self::UnknownType(type_id) => {
                write!(f, "unregistered reflected type {}", type_id.value())
            }
            Self::UnknownComponent(key) => write!(
                f,
                "unknown component schema={} id={}",
                key.schema().as_u128(),
                key.component()
            ),
            Self::LocalEntityUnavailable(local) => {
                write!(
                    f,
                    "local entity handle {} is unavailable",
                    local.entity_bits()
                )
            }
            Self::WorldRejected(entity) => {
                write!(f, "world rejected entity operation on {entity:?}")
            }
            Self::Serialize(error) => write!(f, "component serialization failed: {error}"),
            Self::Deserialize(error) => write!(f, "component deserialization failed: {error}"),
            Self::FieldDelta(error) => write!(f, "component field delta failed: {error}"),
            Self::ReflectInsertRejected(key) => write!(
                f,
                "world rejected reflected component schema={} id={}",
                key.schema().as_u128(),
                key.component()
            ),
            Self::Mapping(error) => write!(f, "replication mapping rejected operation: {error:?}"),
        }
    }
}

impl core::error::Error for EcsReplicationError {}

/// Static manifest information for one reflected component.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ComponentDescriptor {
    /// Stable manifest key.
    pub key: ComponentKey,
    /// Stable reflected type id.
    pub type_id: StableTypeId,
    /// Schema version registered for this component family.
    pub schema_version: u32,
    /// Number of top-level fields accepted by this provider.
    pub field_count: usize,
}

#[derive(Clone, Copy)]
struct RegisteredComponent {
    descriptor: ComponentDescriptor,
    type_info: &'static TypeInfo,
}

/// A concrete provider over one `prism_ecs::World`'s reflected components.
pub struct EcsReplication {
    limits: EcsReplicationLimits,
    mapping: ReplicationMapping,
    types: TypeRegistry,
    components: BTreeMap<ComponentKey, RegisteredComponent>,
    by_type: BTreeMap<StableTypeId, ComponentKey>,
}

impl EcsReplication {
    /// Create an empty provider with explicit resource and decode bounds.
    #[must_use]
    pub fn new(
        limits: EcsReplicationLimits,
        mapping_limits: crate::mapping::MappingLimits,
    ) -> Self {
        Self {
            limits,
            mapping: ReplicationMapping::new(mapping_limits),
            types: TypeRegistry::new(),
            components: BTreeMap::new(),
            by_type: BTreeMap::new(),
        }
    }

    /// Return the provider's live local-to-stable mapping.
    #[must_use]
    pub fn mapping(&self) -> &ReplicationMapping {
        &self.mapping
    }

    /// Return the configured bounds.
    #[must_use]
    pub const fn limits(&self) -> EcsReplicationLimits {
        self.limits
    }

    /// Register a project-defined reflected component under a stable manifest
    /// key. Registration is idempotent only when both the key and type agree.
    pub fn register_component<T>(
        &mut self,
        world: &mut World,
        key: ComponentKey,
        schema_version: u32,
    ) -> Result<ComponentDescriptor, EcsReplicationError>
    where
        T: Component + Reflect + FromReflect + GetTypeRegistration + Clone,
    {
        let type_id = world.register_reflect_component::<T>();
        self.types.register::<T>();
        let registration = T::get_type_registration();
        let field_count = reflected_field_count(registration.type_info());
        if field_count > self.limits.max_fields_per_component {
            return Err(EcsReplicationError::TooManyFields {
                count: field_count,
                max: self.limits.max_fields_per_component,
            });
        }
        let descriptor = ComponentDescriptor {
            key,
            type_id,
            schema_version,
            field_count,
        };
        if let Some(existing) = self.components.get(&key) {
            if existing.descriptor.type_id != type_id
                || existing.descriptor.schema_version != schema_version
            {
                return Err(EcsReplicationError::UnknownComponent(key));
            }
            return Ok(existing.descriptor);
        }
        self.mapping
            .register_schema(SchemaDescriptor {
                id: key.schema(),
                version: schema_version,
            })
            .map_err(EcsReplicationError::Mapping)?;
        if let Some(existing_key) = self.by_type.get(&type_id).copied()
            && existing_key != key
        {
            return Err(EcsReplicationError::UnknownComponent(key));
        }
        self.by_type.insert(type_id, key);
        self.components.insert(
            key,
            RegisteredComponent {
                descriptor,
                type_info: registration.type_info(),
            },
        );
        Ok(descriptor)
    }

    /// Capture all registered reflected components on a live ECS entity.
    pub fn snapshot_entity(
        &self,
        world: &World,
        entity: Entity,
        stable: EntityId,
        authority: AuthorityId,
        revision: Revision,
    ) -> Result<EntitySnapshot, EcsReplicationError> {
        if !world.contains(entity) {
            return Err(EcsReplicationError::WorldRejected(entity));
        }
        let mut components = Vec::new();
        for (type_id, value) in world.reflect_entity(entity) {
            let key = self
                .by_type
                .get(&type_id)
                .copied()
                .ok_or(EcsReplicationError::UnknownType(type_id))?;
            let payload = self.encode_value(key, value)?;
            components.push(ComponentRecord {
                component: key,
                revision,
                payload,
            });
        }
        if components.is_empty() {
            return Err(EcsReplicationError::EmptyEntitySnapshot);
        }
        self.check_component_count(components.len())?;
        Ok(EntitySnapshot {
            entity: stable,
            authority,
            components,
        })
    }

    /// Spawn one real ECS entity and install its reflected component values.
    pub fn spawn_entity(
        &mut self,
        world: &mut World,
        snapshot: &EntitySnapshot,
    ) -> Result<Entity, EcsReplicationError> {
        self.check_component_count(snapshot.components.len())?;
        if snapshot.components.is_empty() {
            return Err(EcsReplicationError::EmptyEntitySnapshot);
        }
        if self.mapping.resolve_entity(snapshot.entity).is_some() {
            return Err(EcsReplicationError::EntityAlreadyMapped(snapshot.entity));
        }

        let decoded = self.decode_records(&snapshot.components)?;
        let entity = world.spawn(());
        let local = LocalEntityId::from_entity_bits(entity.to_bits())
            .ok_or(EcsReplicationError::WorldRejected(entity))?;
        let mut next_mapping = self.mapping.clone();
        for record in &snapshot.components {
            next_mapping
                .upsert(MappingEntry {
                    local,
                    entity: snapshot.entity,
                    component: record.component,
                    authority: snapshot.authority,
                    revision: record.revision,
                })
                .map_err(EcsReplicationError::Mapping)?;
        }
        for (key, value) in decoded {
            if !world.insert_reflect(entity, &*value) {
                world.despawn(entity);
                return Err(EcsReplicationError::ReflectInsertRejected(key));
            }
        }
        self.mapping = next_mapping;
        Ok(entity)
    }

    /// Apply a stable update to one real ECS entity.
    pub fn update_entity(
        &mut self,
        world: &mut World,
        update: &EntityUpdate,
    ) -> Result<(), EcsReplicationError> {
        self.check_component_count(update.components.len())?;
        let local = self
            .mapping
            .resolve_entity(update.entity)
            .ok_or(EcsReplicationError::UnknownEntity(update.entity))?;
        let entity = Entity::from_bits(local.entity_bits())
            .filter(|entity| world.contains(*entity))
            .ok_or(EcsReplicationError::LocalEntityUnavailable(local))?;
        let decoded = self.decode_records(&update.components)?;
        let mut next_mapping = self.mapping.clone();
        for record in &update.components {
            next_mapping
                .upsert(MappingEntry {
                    local,
                    entity: update.entity,
                    component: record.component,
                    authority: update.authority,
                    revision: record.revision,
                })
                .map_err(EcsReplicationError::Mapping)?;
        }
        for (key, value) in decoded {
            if !world.insert_reflect(entity, &*value) {
                return Err(EcsReplicationError::ReflectInsertRejected(key));
            }
        }
        self.mapping = next_mapping;
        Ok(())
    }

    /// Despawn one real ECS entity and remove all of its stable mappings.
    pub fn despawn_entity(
        &mut self,
        world: &mut World,
        stable: EntityId,
    ) -> Result<bool, EcsReplicationError> {
        let local = self
            .mapping
            .resolve_entity(stable)
            .ok_or(EcsReplicationError::UnknownEntity(stable))?;
        let entity = Entity::from_bits(local.entity_bits())
            .filter(|entity| world.contains(*entity))
            .ok_or(EcsReplicationError::LocalEntityUnavailable(local))?;
        if !world.despawn(entity) {
            return Err(EcsReplicationError::WorldRejected(entity));
        }
        self.mapping.remove_entity(stable);
        Ok(true)
    }

    /// Encode a complete reflected component value with the configured bounds.
    pub fn encode_component(
        &self,
        world: &World,
        entity: Entity,
        key: ComponentKey,
    ) -> Result<Vec<u8>, EcsReplicationError> {
        let type_id = self.spec(key)?.descriptor.type_id;
        let value = world.get_reflect(entity, type_id).ok_or(
            EcsReplicationError::LocalEntityUnavailable(
                LocalEntityId::from_entity_bits(entity.to_bits())
                    .ok_or(EcsReplicationError::WorldRejected(entity))?,
            ),
        )?;
        self.encode_value(key, value)
    }

    /// Encode only selected top-level struct fields of a reflected component.
    /// The resulting bytes are the existing bounded-by-provider field-delta
    /// contract from `prism_reflect::net_delta`.
    pub fn encode_component_fields(
        &self,
        world: &World,
        entity: Entity,
        key: ComponentKey,
        dirty: &prism_reflect::DirtyMask,
    ) -> Result<Vec<u8>, EcsReplicationError> {
        let spec = self.spec(key)?;
        if dirty.count() > self.limits.max_fields_per_component {
            return Err(EcsReplicationError::TooManyFields {
                count: dirty.count(),
                max: self.limits.max_fields_per_component,
            });
        }
        let value = world
            .get_reflect(entity, spec.descriptor.type_id)
            .ok_or(EcsReplicationError::UnknownComponent(key))?;
        let bytes = encode_delta(value, dirty).map_err(EcsReplicationError::FieldDelta)?;
        self.check_payload(&bytes)?;
        Ok(bytes)
    }

    /// Apply a bounded field delta to the mapped live component.
    pub fn apply_component_fields(
        &mut self,
        world: &mut World,
        entity: EntityId,
        key: ComponentKey,
        bytes: &[u8],
    ) -> Result<(), EcsReplicationError> {
        self.check_payload(bytes)?;
        let local = self
            .mapping
            .resolve_entity(entity)
            .ok_or(EcsReplicationError::UnknownEntity(entity))?;
        let local_entity = Entity::from_bits(local.entity_bits())
            .filter(|candidate| world.contains(*candidate))
            .ok_or(EcsReplicationError::LocalEntityUnavailable(local))?;
        let spec = self.spec(key)?;
        let delta = decode_delta(bytes).map_err(EcsReplicationError::FieldDelta)?;
        if delta.len() > self.limits.max_fields_per_component {
            return Err(EcsReplicationError::TooManyFields {
                count: delta.len(),
                max: self.limits.max_fields_per_component,
            });
        }
        let target = world
            .get_reflect_mut(local_entity, spec.descriptor.type_id)
            .ok_or(EcsReplicationError::UnknownComponent(key))?;
        apply_bounded_delta(target, &delta, &self.types, self.limits.decode)
            .map_err(EcsReplicationError::FieldDelta)
    }

    /// Adopt an entity that was created by the scene subsystem without
    /// spawning a duplicate ECS entity. The scene remains the owner of the
    /// entity's lifetime; this provider only records its stable mapping.
    pub fn adopt_entity(
        &mut self,
        world: &World,
        entity: Entity,
        stable: EntityId,
        authority: AuthorityId,
        revision: Revision,
    ) -> Result<(), EcsReplicationError> {
        let snapshot = self.snapshot_entity(world, entity, stable, authority, revision)?;
        let local = LocalEntityId::from_entity_bits(entity.to_bits())
            .ok_or(EcsReplicationError::WorldRejected(entity))?;
        let mut next_mapping = self.mapping.clone();
        for record in snapshot.components {
            next_mapping
                .upsert(MappingEntry {
                    local,
                    entity: stable,
                    component: record.component,
                    authority,
                    revision: record.revision,
                })
                .map_err(EcsReplicationError::Mapping)?;
        }
        self.mapping = next_mapping;
        Ok(())
    }

    /// Forget an entity mapping after the scene subsystem has despawned it.
    /// This deliberately does not touch the ECS world.
    pub fn forget_entity(&mut self, stable: EntityId) -> usize {
        self.mapping.remove_entity(stable)
    }

    fn spec(&self, key: ComponentKey) -> Result<RegisteredComponent, EcsReplicationError> {
        self.components
            .get(&key)
            .copied()
            .ok_or(EcsReplicationError::UnknownComponent(key))
    }

    fn check_component_count(&self, count: usize) -> Result<(), EcsReplicationError> {
        if count > self.limits.max_components_per_entity {
            return Err(EcsReplicationError::TooManyComponents {
                count,
                max: self.limits.max_components_per_entity,
            });
        }
        Ok(())
    }

    fn check_payload(&self, bytes: &[u8]) -> Result<(), EcsReplicationError> {
        if bytes.len() > self.limits.max_component_payload {
            return Err(EcsReplicationError::PayloadTooLarge {
                len: bytes.len(),
                max: self.limits.max_component_payload,
            });
        }
        Ok(())
    }

    fn encode_value(
        &self,
        key: ComponentKey,
        value: &dyn Reflect,
    ) -> Result<Vec<u8>, EcsReplicationError> {
        let spec = self.spec(key)?;
        let bytes = to_binary(value).map_err(EcsReplicationError::Serialize)?;
        self.check_payload(&bytes)?;
        if reflected_field_count(value.type_info()) != spec.descriptor.field_count {
            return Err(EcsReplicationError::TooManyFields {
                count: reflected_field_count(value.type_info()),
                max: self.limits.max_fields_per_component,
            });
        }
        Ok(bytes)
    }

    fn decode_records(
        &self,
        records: &[ComponentRecord],
    ) -> Result<Vec<(ComponentKey, Box<dyn Reflect>)>, EcsReplicationError> {
        let mut seen = BTreeSet::new();
        let mut decoded = Vec::with_capacity(records.len());
        for record in records {
            if !seen.insert(record.component) {
                return Err(EcsReplicationError::UnknownComponent(record.component));
            }
            self.check_payload(&record.payload)?;
            let spec = self.spec(record.component)?;
            let value = from_binary_with_limits(
                &record.payload,
                &self.types,
                spec.type_info,
                &self.limits.decode,
            )
            .map_err(EcsReplicationError::Deserialize)?;
            if StableTypeId::of_path(value.type_name()) != spec.descriptor.type_id {
                return Err(EcsReplicationError::UnknownComponent(record.component));
            }
            decoded.push((record.component, value));
        }
        Ok(decoded)
    }
}

fn reflected_field_count(info: &TypeInfo) -> usize {
    match info {
        TypeInfo::Struct(info) => info.field_count(),
        TypeInfo::TupleStruct(info) => info.field_count(),
        _ => 1,
    }
}

fn apply_bounded_delta(
    target: &mut dyn Reflect,
    delta: &FieldDelta,
    registry: &TypeRegistry,
    limits: DeserializeLimits,
) -> Result<(), DeltaError> {
    if !matches!(target.reflect_mut(), ReflectMut::Struct(_)) {
        return Err(DeltaError::NotAStruct);
    }
    for (index, bytes) in delta.entries() {
        let ReflectMut::Struct(structure) = target.reflect_mut() else {
            return Err(DeltaError::NotAStruct);
        };
        let field_count = structure.field_count();
        let field = structure
            .field_at_mut(*index)
            .ok_or(DeltaError::FieldIndexOutOfRange {
                index: *index,
                field_count,
            })?;
        let decoded = from_binary_with_limits(bytes, registry, field.type_info(), &limits)
            .map_err(DeltaError::Deserialize)?;
        field.apply(&*decoded).map_err(DeltaError::Apply)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use prism_ecs::component::Component;
    use prism_reflect::{DirtyMask, Reflect};

    #[derive(Clone, Debug, PartialEq, Reflect)]
    struct Transform {
        x: i32,
        y: i32,
    }
    impl Component for Transform {}

    fn provider() -> EcsReplication {
        EcsReplication::new(
            EcsReplicationLimits {
                max_components_per_entity: 4,
                max_component_payload: 1024,
                max_fields_per_component: 4,
                decode: DeserializeLimits::new(8, 32),
            },
            crate::mapping::MappingLimits::new(2, 8),
        )
    }

    #[test]
    fn real_world_spawn_update_field_delta_and_despawn_round_trip() {
        let key = ComponentKey::new(crate::mapping::SchemaId::from_u128(7), 11);
        let stable = EntityId::from_u128(99);
        let mut source = World::new();
        let source_entity = source.spawn(Transform { x: 1, y: 2 });
        let mut source_provider = provider();
        source_provider
            .register_component::<Transform>(&mut source, key, 1)
            .unwrap();
        let snapshot = source_provider
            .snapshot_entity(
                &source,
                source_entity,
                stable,
                AuthorityId::from_u128(3),
                Revision::new(1),
            )
            .unwrap();

        let mut target = World::new();
        let mut target_provider = provider();
        target_provider
            .register_component::<Transform>(&mut target, key, 1)
            .unwrap();
        let target_entity = target_provider
            .spawn_entity(&mut target, &snapshot)
            .unwrap();
        assert_eq!(
            target.get::<Transform>(target_entity),
            Some(&Transform { x: 1, y: 2 })
        );

        let mut dirty = DirtyMask::new();
        dirty.mark(1);
        source.get_mut::<Transform>(source_entity).unwrap().y = 9;
        let delta = source_provider
            .encode_component_fields(&source, source_entity, key, &dirty)
            .unwrap();
        target_provider
            .apply_component_fields(&mut target, stable, key, &delta)
            .unwrap();
        assert_eq!(
            target.get::<Transform>(target_entity),
            Some(&Transform { x: 1, y: 9 })
        );

        assert!(target_provider.despawn_entity(&mut target, stable).unwrap());
        assert!(!target.contains(target_entity));
        assert_eq!(target_provider.mapping().entry_len(), 0);
    }

    #[test]
    fn hostile_payload_and_stale_revision_are_rejected_before_world_mutation() {
        let key = ComponentKey::new(crate::mapping::SchemaId::from_u128(7), 11);
        let stable = EntityId::from_u128(99);
        let mut world = World::new();
        let mut provider = provider();
        provider
            .register_component::<Transform>(&mut world, key, 1)
            .unwrap();
        let entity = world.spawn(Transform { x: 1, y: 2 });
        let snapshot = provider
            .snapshot_entity(
                &world,
                entity,
                stable,
                AuthorityId::from_u128(3),
                Revision::new(2),
            )
            .unwrap();
        provider
            .adopt_entity(
                &world,
                entity,
                stable,
                AuthorityId::from_u128(3),
                Revision::new(2),
            )
            .unwrap();
        let local = provider.spawn_entity(&mut world, &snapshot).err();
        assert!(matches!(
            local,
            Some(EcsReplicationError::EntityAlreadyMapped(_))
        ));
        let before = world.entity_count();
        let error = provider.update_entity(
            &mut world,
            &EntityUpdate {
                entity: stable,
                authority: AuthorityId::from_u128(3),
                components: vec![ComponentRecord {
                    component: key,
                    revision: Revision::new(1),
                    payload: vec![0; 2048],
                }],
            },
        );
        assert!(matches!(
            error,
            Err(EcsReplicationError::PayloadTooLarge { .. })
        ));
        assert_eq!(world.entity_count(), before);
    }
}
