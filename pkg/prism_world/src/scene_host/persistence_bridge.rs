//! Explicit mapping between scene property snapshots and world field deltas.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::convert::TryFrom;
use prism_scene::{
    decode_persistence, encode_persistence, ComponentTypeId, FieldPath,
    OverrideValue as SceneValue, PersistenceDecodeLimits,
    PersistedPropertyOverride, PersistentEntityMap, SceneInstancePersistence,
};
use prism_world_persist::{
    ActorId, FieldId, FieldOverride, LogicalClock, OverrideDelta, OverrideValue as WorldValue,
    PersistentId as WorldPersistentId, Transaction, TransactionId, WorldState,
};

/// A schema-owned mapping from a scene component field path to a world field id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SceneFieldBinding {
    /// Scene component type identity.
    pub component: ComponentTypeId,
    /// Scene field path identity.
    pub path: FieldPath,
    /// Field identity in `prism_world_persist`.
    pub field: FieldId,
    /// The exact value interpretation used in both wire domains.
    pub value_kind: SceneFieldValueKind,
}

/// Explicit identity mapping between a scene persistent entity and world object.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SceneObjectBinding {
    /// Scene-runtime persistent entity identity.
    pub scene: prism_scene::PersistentId,
    /// World-persistence object identity.
    pub world: WorldPersistentId,
}

/// Validated bidirectional entity identity mapping.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SceneEntityMap {
    bindings: Vec<SceneObjectBinding>,
    by_world: Vec<SceneObjectBinding>,
}

impl SceneEntityMap {
    /// Creates a canonical one-to-one mapping, rejecting nil and duplicate ids.
    pub fn new(mut bindings: Vec<SceneObjectBinding>) -> Result<Self, ScenePersistenceBridgeError> {
        if bindings.iter().any(|binding| binding.scene.is_nil()) {
            return Err(ScenePersistenceBridgeError::MissingEntityIdentity);
        }
        bindings.sort_unstable_by_key(|binding| binding.scene);
        if bindings
            .windows(2)
            .any(|pair| pair[0].scene == pair[1].scene)
        {
            return Err(ScenePersistenceBridgeError::DuplicateSceneEntity);
        }
        let mut by_world = bindings.clone();
        by_world.sort_unstable_by_key(|binding| binding.world);
        if by_world
            .windows(2)
            .any(|pair| pair[0].world == pair[1].world)
        {
            return Err(ScenePersistenceBridgeError::DuplicateWorldEntity);
        }
        Ok(Self { bindings, by_world })
    }

    fn world_for_scene(&self, scene: prism_scene::PersistentId) -> Option<WorldPersistentId> {
        self.bindings
            .binary_search_by_key(&scene, |binding| binding.scene)
            .ok()
            .and_then(|index| self.bindings.get(index))
            .map(|binding| binding.world)
    }

    fn scene_for_world(&self, world: WorldPersistentId) -> Option<prism_scene::PersistentId> {
        self.by_world
            .binary_search_by_key(&world, |binding| binding.world)
            .ok()
            .and_then(|index| self.by_world.get(index))
            .map(|binding| binding.scene)
    }
}

/// Explicit scene value interpretation for the narrower world value domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneFieldValueKind {
    /// Boolean scalar.
    Bool,
    /// Signed 64-bit integer.
    I64,
    /// Unsigned 64-bit integer.
    U64,
    /// 32-bit float stored losslessly as an f64 world scalar.
    F32,
    /// 64-bit float.
    F64,
    /// UTF-8 string stored as bytes.
    String,
    /// Opaque byte array, including byte arrays that happen to be UTF-8.
    Bytes,
}

/// Bidirectional, explicitly registered schema mapping for scene persistence.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ScenePersistenceSchema {
    bindings: Vec<SceneFieldBinding>,
    by_world_field: Vec<SceneFieldBinding>,
}

/// Property transactions plus the scene-only snapshot data they do not encode.
#[derive(Clone, Debug, PartialEq)]
pub struct ScenePersistenceBatch {
    /// Per-world-object sparse property deltas.
    pub deltas: Vec<(WorldPersistentId, OverrideDelta)>,
    /// PRSI data retaining template/instance identity and ordered structural ops.
    pub scene_snapshot: SceneInstancePersistence,
}

/// Invalid or incomplete scene/world schema mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScenePersistenceBridgeError {
    /// Two scene keys were registered more than once.
    DuplicateSceneField,
    /// Two scene keys map to one world field, making reverse mapping ambiguous.
    DuplicateWorldField,
    /// A snapshot property has no explicit world field mapping.
    UnmappedSceneField,
    /// A replayed world field has no explicit scene field mapping.
    UnmappedWorldField,
    /// A property has no stable entity identity.
    MissingEntityIdentity,
    /// Scene entity identity was registered more than once.
    DuplicateSceneEntity,
    /// World object identity was registered more than once.
    DuplicateWorldEntity,
    /// Scene and world value domains cannot represent the same value.
    IncompatibleValue,
    /// Two snapshot properties resolve to the same world field for one object.
    DuplicateObjectField,
    /// One transaction identity is required for each object delta.
    TransactionIdCountMismatch,
    /// Transaction identities in one scene batch must be unique.
    DuplicateTransactionId,
    /// Transaction sequence arithmetic exceeded the wire-domain limit.
    TransactionSequenceExhausted,
    /// The companion snapshot could not be encoded as canonical PRSI bytes.
    InvalidCompanionSnapshot,
    /// No companion value exists at the explicitly registered object/field.
    MissingCompanionSnapshot,
    /// The companion field exists but is not stored as opaque bytes.
    InvalidCompanionValueKind,
    /// The companion binding aliases a sparse property field in the same batch.
    CompanionFieldCollision,
}

impl ScenePersistenceSchema {
    /// Creates a canonical bidirectional mapping, rejecting ambiguous keys.
    pub fn new(mut bindings: Vec<SceneFieldBinding>) -> Result<Self, ScenePersistenceBridgeError> {
        bindings.sort_unstable_by(|left, right| {
            (left.component, &left.path, left.field).cmp(&(
                right.component,
                &right.path,
                right.field,
            ))
        });
        for pair in bindings.windows(2) {
            if pair[0].component == pair[1].component && pair[0].path == pair[1].path {
                return Err(ScenePersistenceBridgeError::DuplicateSceneField);
            }
        }
        let mut by_world_field = bindings.clone();
        by_world_field.sort_unstable_by_key(|binding| binding.field);
        if by_world_field
            .windows(2)
            .any(|pair| pair[0].field == pair[1].field)
        {
            return Err(ScenePersistenceBridgeError::DuplicateWorldField);
        }
        Ok(Self {
            bindings,
            by_world_field,
        })
    }

    /// Returns the canonical schema bindings.
    pub fn bindings(&self) -> &[SceneFieldBinding] {
        &self.bindings
    }

    /// Converts sparse property overrides into per-object sparse world deltas.
    /// Template/instance identity and ordered structural operations are retained
    /// in `scene_snapshot`; they are never collapsed into unordered field deltas.
    pub fn snapshot_to_deltas(
        &self,
        snapshot: &SceneInstancePersistence,
        identities: &PersistentEntityMap,
        entities: &SceneEntityMap,
        prior: &WorldState,
    ) -> Result<ScenePersistenceBatch, ScenePersistenceBridgeError> {
        let mut desired_by_object: BTreeMap<WorldPersistentId, BTreeMap<FieldId, WorldValue>> =
            BTreeMap::new();
        for property in &snapshot.properties {
            let scene_id = property
                .persistent
                .ok_or(ScenePersistenceBridgeError::MissingEntityIdentity)?;
            let object = entities
                .world_for_scene(scene_id)
                .ok_or(ScenePersistenceBridgeError::MissingEntityIdentity)?;
            if identities.local_for_persistent(scene_id) != Some(property.local) {
                return Err(ScenePersistenceBridgeError::MissingEntityIdentity);
            }
            let binding = self
                .find_scene_field(property.component, &property.path)
                .ok_or(ScenePersistenceBridgeError::UnmappedSceneField)?;
            let value = to_world_value(&property.value, binding.value_kind)?;
            if desired_by_object
                .entry(object)
                .or_default()
                .insert(binding.field, value)
                .is_some()
            {
                return Err(ScenePersistenceBridgeError::DuplicateObjectField);
            }
        }
        let mut objects: BTreeMap<WorldPersistentId, ()> = entities
            .bindings
            .iter()
            .map(|binding| (binding.world, ()))
            .collect();
        for object in desired_by_object.keys() {
            objects.insert(*object, ());
        }
        let mut deltas = Vec::new();
        for object in objects.keys().copied() {
            let desired = desired_by_object.get(&object);
            let mut fields = BTreeMap::new();
            if let Some(previous) = prior.object(object) {
                for (field, _) in previous.iter() {
                    if self
                        .by_world_field
                        .binary_search_by_key(field, |binding| binding.field)
                        .is_err()
                    {
                        return Err(ScenePersistenceBridgeError::UnmappedWorldField);
                    }
                    fields.insert(*field, ());
                }
            }
            if let Some(desired) = desired {
                fields.extend(desired.keys().map(|field| (*field, ())));
            }
            let changes = fields
                .keys()
                .filter_map(|&field| {
                    let old = prior.get(object, field);
                    let new = desired.and_then(|values| values.get(&field));
                    if old == new {
                        None
                    } else {
                        Some(FieldOverride {
                            field,
                            value: new.cloned().unwrap_or(WorldValue::Delete),
                        })
                    }
                })
                .collect();
            let delta = OverrideDelta::new(changes)
                .map_err(|_| ScenePersistenceBridgeError::IncompatibleValue)?;
            if !delta.is_empty() {
                deltas.push((object, delta));
            }
        }
        let mut scene_snapshot = snapshot.clone();
        scene_snapshot.properties.clear();
        Ok(ScenePersistenceBatch {
            deltas,
            scene_snapshot,
        })
    }

    /// Builds ordered transactions with exact inverses from the prior replayed state.
    ///
    /// Ids correspond to canonical `batch.deltas` ordering. This pure conversion
    /// returns no partial transaction list; the caller persists the returned
    /// transactions as one log batch and stores the companion PRSI snapshot.
    pub fn transactions_for_batch(
        &self,
        batch: &ScenePersistenceBatch,
        prior: &WorldState,
        actor: ActorId,
        clock: LogicalClock,
        next_sequence: u64,
        transaction_ids: &[TransactionId],
    ) -> Result<Vec<Transaction>, ScenePersistenceBridgeError> {
        if transaction_ids.len() != batch.deltas.len() {
            return Err(ScenePersistenceBridgeError::TransactionIdCountMismatch);
        }
        let unique_ids: BTreeSet<_> = transaction_ids.iter().copied().collect();
        if unique_ids.len() != transaction_ids.len() {
            return Err(ScenePersistenceBridgeError::DuplicateTransactionId);
        }
        let mut transactions = Vec::with_capacity(batch.deltas.len());
        for ((object, delta), &id) in batch.deltas.iter().zip(transaction_ids) {
            let offset = u64::try_from(transactions.len())
                .map_err(|_| ScenePersistenceBridgeError::TransactionSequenceExhausted)?;
            let sequence = next_sequence
                .checked_add(offset)
                .ok_or(ScenePersistenceBridgeError::TransactionSequenceExhausted)?;
            if sequence == u64::MAX {
                return Err(ScenePersistenceBridgeError::TransactionSequenceExhausted);
            }
            let inverse = OverrideDelta::new(
                delta
                    .iter()
                    .map(|change| FieldOverride {
                        field: change.field,
                        value: prior
                            .get(*object, change.field)
                            .cloned()
                            .unwrap_or(WorldValue::Delete),
                    })
                    .collect(),
            )
            .map_err(|_| ScenePersistenceBridgeError::IncompatibleValue)?;
            transactions.push(
                Transaction::new(id, actor, clock, sequence, *object, delta.clone(), inverse)
                    .map_err(|_| ScenePersistenceBridgeError::IncompatibleValue)?,
            );
        }
        Ok(transactions)
    }

    /// Adds the encoded scene-only companion state as one explicitly bound
    /// opaque world field, so properties and ordered structural metadata can
    /// be committed by the same durable journal batch. The companion is kept
    /// separate from sparse property fields and is never interpreted as LWW
    /// structural operations.
    pub fn transactions_for_batch_with_companion(
        &self,
        batch: &ScenePersistenceBatch,
        prior: &WorldState,
        actor: ActorId,
        clock: LogicalClock,
        next_sequence: u64,
        transaction_ids: &[TransactionId],
        companion_transaction_id: TransactionId,
        companion_object: WorldPersistentId,
        companion_field: FieldId,
    ) -> Result<Vec<Transaction>, ScenePersistenceBridgeError> {
        let mut filtered = batch.clone();
        let mut filtered_ids = Vec::with_capacity(transaction_ids.len());
        for ((object, delta), id) in filtered
            .deltas
            .iter_mut()
            .zip(transaction_ids.iter().copied())
        {
            let changes: Vec<_> = delta
                .iter()
                .filter(|change| prior.get(*object, change.field) != Some(&change.value))
                .cloned()
                .collect();
            *delta = OverrideDelta::new(changes)
                .map_err(|_| ScenePersistenceBridgeError::IncompatibleValue)?;
            if !delta.is_empty() {
                filtered_ids.push(id);
            }
        }
        filtered.deltas.retain(|(_, delta)| !delta.is_empty());
        let mut transactions = self.transactions_for_batch(
            &filtered,
            prior,
            actor,
            clock,
            next_sequence,
            &filtered_ids,
        )?;
        if filtered.deltas.iter().any(|(object, delta)| {
            *object == companion_object && delta.get(companion_field).is_some()
        }) {
            return Err(ScenePersistenceBridgeError::CompanionFieldCollision);
        }
        let sequence = next_sequence
            .checked_add(
                u64::try_from(transactions.len())
                    .map_err(|_| ScenePersistenceBridgeError::TransactionSequenceExhausted)?,
            )
            .filter(|sequence| *sequence < u64::MAX)
            .ok_or(ScenePersistenceBridgeError::TransactionSequenceExhausted)?;
        let bytes = encode_persistence(&batch.scene_snapshot)
            .map_err(|_| ScenePersistenceBridgeError::InvalidCompanionSnapshot)?;
        if prior.get(companion_object, companion_field) == Some(&WorldValue::Bytes(bytes.clone())) {
            return Ok(transactions);
        }
        if filtered_ids.contains(&companion_transaction_id) {
            return Err(ScenePersistenceBridgeError::DuplicateTransactionId);
        }
        let delta = OverrideDelta::single(companion_field, WorldValue::Bytes(bytes))
            .map_err(|_| ScenePersistenceBridgeError::IncompatibleValue)?;
        let inverse = OverrideDelta::single(
            companion_field,
            prior
                .get(companion_object, companion_field)
                .cloned()
                .unwrap_or(WorldValue::Delete),
        )
        .map_err(|_| ScenePersistenceBridgeError::IncompatibleValue)?;
        transactions.push(
            Transaction::new(
                companion_transaction_id,
                actor,
                clock,
                sequence,
                companion_object,
                delta,
                inverse,
            )
            .map_err(|_| ScenePersistenceBridgeError::IncompatibleValue)?,
        );
        Ok(transactions)
    }

    /// Reconstructs property rows from the world state for entities in `identities`.
    /// Template metadata and ordered structural operations are copied from the
    /// supplied scene snapshot; only registered world fields are consumed.
    pub fn deltas_to_snapshot(
        &self,
        world: &WorldState,
        template: &SceneInstancePersistence,
        identities: &PersistentEntityMap,
        entities: &SceneEntityMap,
    ) -> Result<SceneInstancePersistence, ScenePersistenceBridgeError> {
        let mut properties = Vec::new();
        for (object, object_state) in world.iter() {
            let Some(scene_id) = entities.scene_for_world(*object) else {
                continue;
            };
            let local = identities
                .local_for_persistent(scene_id)
                .ok_or(ScenePersistenceBridgeError::MissingEntityIdentity)?;
            if identities.persistent_for_local(local) != Some(scene_id) {
                return Err(ScenePersistenceBridgeError::MissingEntityIdentity);
            }
            if identities.persistent_for_local(local) != Some(scene_id) {
                return Err(ScenePersistenceBridgeError::MissingEntityIdentity);
            }
            for (field, value) in object_state.iter() {
                let binding = self
                    .by_world_field
                    .binary_search_by_key(field, |binding| binding.field)
                    .ok()
                    .and_then(|index| self.by_world_field.get(index))
                    .ok_or(ScenePersistenceBridgeError::UnmappedWorldField)?;
                properties.push(PersistedPropertyOverride {
                    local,
                    persistent: Some(scene_id),
                    component: binding.component,
                    path: binding.path.clone(),
                    value: from_world_value(value, binding.value_kind)?,
                });
            }
        }
        properties.sort_unstable_by(|left, right| {
            (left.persistent, left.component, &left.path).cmp(&(
                right.persistent,
                right.component,
                &right.path,
            ))
        });
        Ok(SceneInstancePersistence {
            template: template.template,
            instance: template.instance,
            properties,
            structural: template.structural.clone(),
        })
    }

    /// Restores a complete scene snapshot from the authoritative replayed world
    /// state. The PRSI companion supplies template/instance identity and ordered
    /// structural operations; registered sparse world fields supply properties.
    /// The caller must provide the explicit companion binding and identity maps
    /// belonging to this scene.
    pub fn snapshot_from_world_state(
        &self,
        world: &WorldState,
        companion_object: WorldPersistentId,
        companion_field: FieldId,
        identities: &PersistentEntityMap,
        entities: &SceneEntityMap,
        limits: PersistenceDecodeLimits,
    ) -> Result<SceneInstancePersistence, ScenePersistenceBridgeError> {
        let bytes = match world.get(companion_object, companion_field) {
            Some(WorldValue::Bytes(bytes)) => bytes,
            Some(_) => return Err(ScenePersistenceBridgeError::InvalidCompanionValueKind),
            None => return Err(ScenePersistenceBridgeError::MissingCompanionSnapshot),
        };
        let companion = decode_persistence(bytes, limits)
            .map_err(|_| ScenePersistenceBridgeError::InvalidCompanionSnapshot)?;
        self.deltas_to_snapshot(world, &companion, identities, entities)
    }

    /// Restores a snapshot when this scene has persisted state, returning `None`
    /// only when the companion field has never been written. A present but
    /// malformed or non-byte value remains an error; absence is the valid
    /// first-load case for a scene with no saved overrides yet.
    pub fn snapshot_from_world_state_if_present(
        &self,
        world: &WorldState,
        companion_object: WorldPersistentId,
        companion_field: FieldId,
        identities: &PersistentEntityMap,
        entities: &SceneEntityMap,
        limits: PersistenceDecodeLimits,
    ) -> Result<Option<SceneInstancePersistence>, ScenePersistenceBridgeError> {
        if world.get(companion_object, companion_field).is_none() {
            return Ok(None);
        }
        self.snapshot_from_world_state(
            world,
            companion_object,
            companion_field,
            identities,
            entities,
            limits,
        )
        .map(Some)
    }

    fn find_scene_field(
        &self,
        component: ComponentTypeId,
        path: &FieldPath,
    ) -> Option<&SceneFieldBinding> {
        self.bindings
            .binary_search_by(|binding| (binding.component, &binding.path).cmp(&(component, path)))
            .ok()
            .and_then(|index| self.bindings.get(index))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        encode_persistence, SceneEntityMap, SceneFieldBinding, SceneFieldValueKind,
        SceneObjectBinding, ScenePersistenceBatch, ScenePersistenceBridgeError,
        ScenePersistenceSchema,
    };
    use alloc::format;
    use alloc::vec;
    use alloc::vec::Vec;
    use prism_asset::StableGuid;
    use prism_scene::{
        ComponentTypeId, EntityBinding, FieldPath, LocalEntityId, OverrideValue,
        PersistedPropertyOverride, PersistentEntityMap, PersistentId, SceneInstancePersistence,
    };
    use prism_world_persist::{
        ActorId, FieldId, FileJournal, LogicalClock, OverrideDelta, OverrideValue as WorldValue,
        PersistentId as WorldPersistentId, Snapshot, Timeline, TransactionId, WorldState,
    };

    fn schema() -> ScenePersistenceSchema {
        ScenePersistenceSchema::new(vec![SceneFieldBinding {
            component: ComponentTypeId::from_raw(11),
            path: FieldPath::parse("health.current").unwrap(),
            field: FieldId(101),
            value_kind: SceneFieldValueKind::I64,
        }])
        .unwrap()
    }

    fn scene_snapshot() -> SceneInstancePersistence {
        SceneInstancePersistence {
            template: StableGuid::from_u128(5),
            instance: Some(PersistentId::from_u128(9)),
            properties: vec![PersistedPropertyOverride {
                local: LocalEntityId::new(3),
                persistent: Some(PersistentId::from_u128(77)),
                component: ComponentTypeId::from_raw(11),
                path: FieldPath::parse("health.current").unwrap(),
                value: OverrideValue::I64(42),
            }],
            structural: Vec::new(),
        }
    }

    fn entity_map() -> SceneEntityMap {
        SceneEntityMap::new(vec![SceneObjectBinding {
            scene: PersistentId::from_u128(77),
            world: WorldPersistentId(7700),
        }])
        .unwrap()
    }

    #[test]
    fn scene_property_snapshot_round_trips_through_timeline_replay() {
        let schema = schema();
        let snapshot = scene_snapshot();
        let entities = entity_map();
        let prior = WorldState::new();
        let identities = PersistentEntityMap::new(vec![EntityBinding::new(
            LocalEntityId::new(3),
            PersistentId::from_u128(77),
        )])
        .unwrap();
        let batch = schema
            .snapshot_to_deltas(&snapshot, &identities, &entities, &prior)
            .unwrap();
        assert_eq!(batch.deltas.len(), 1);
        assert_eq!(batch.deltas[0].0, WorldPersistentId(7700));
        assert_eq!(
            batch.deltas[0].1.get(FieldId(101)),
            Some(&WorldValue::Integer(42))
        );
        assert!(batch.scene_snapshot.properties.is_empty());

        let transactions = schema
            .transactions_for_batch(
                &batch,
                &prior,
                ActorId(2),
                LogicalClock(1),
                0,
                &[TransactionId(1)],
            )
            .unwrap();
        assert_eq!(transactions[0].sequence(), 0);
        assert!(matches!(
            transactions[0].inverse().get(FieldId(101)),
            Some(WorldValue::Delete)
        ));
        let mut timeline = Timeline::new();
        timeline.append(transactions[0].clone()).unwrap();
        let log = timeline.log();
        let replayed = Timeline::replay(&Snapshot::new(0, WorldState::new()), log).unwrap();
        assert_eq!(
            schema
                .deltas_to_snapshot(&replayed, &batch.scene_snapshot, &identities, &entities)
                .unwrap(),
            snapshot
        );
        let undone = timeline
            .undo(TransactionId(2), ActorId(2), LogicalClock(2))
            .unwrap();
        assert!(matches!(
            undone.delta().get(FieldId(101)),
            Some(WorldValue::Delete)
        ));
        let redone = timeline
            .redo(TransactionId(3), ActorId(2), LogicalClock(3))
            .unwrap();
        assert_eq!(
            redone.delta().get(FieldId(101)),
            Some(&WorldValue::Integer(42))
        );
        assert_eq!(
            schema.transactions_for_batch(&batch, &prior, ActorId(2), LogicalClock(1), 0, &[],),
            Err(ScenePersistenceBridgeError::TransactionIdCountMismatch)
        );
    }

    #[test]
    fn transaction_batch_captures_existing_values_as_exact_inverses() {
        let schema = schema();
        let snapshot = scene_snapshot();
        let identities = PersistentEntityMap::new(vec![EntityBinding::new(
            LocalEntityId::new(3),
            PersistentId::from_u128(77),
        )])
        .unwrap();
        let mut initial = Timeline::new();
        initial
            .commit(
                TransactionId(9),
                ActorId(2),
                LogicalClock(1),
                WorldPersistentId(7700),
                OverrideDelta::single(FieldId(101), WorldValue::Integer(7)).unwrap(),
            )
            .unwrap();
        let prior = initial.state().clone();
        let batch = schema
            .snapshot_to_deltas(&snapshot, &identities, &entity_map(), &prior)
            .unwrap();
        let transactions = schema
            .transactions_for_batch(
                &batch,
                &prior,
                ActorId(2),
                LogicalClock(2),
                1,
                &[TransactionId(10)],
            )
            .unwrap();
        assert_eq!(transactions[0].sequence(), 1);
        assert_eq!(
            transactions[0].inverse().get(FieldId(101)),
            Some(&WorldValue::Integer(7))
        );
        let mut timeline = Timeline::new();
        timeline
            .append(initial.log().get(0).unwrap().clone())
            .unwrap();
        timeline.append(transactions[0].clone()).unwrap();
        assert_eq!(
            timeline
                .undo(TransactionId(11), ActorId(2), LogicalClock(3))
                .unwrap()
                .delta()
                .get(FieldId(101)),
            Some(&WorldValue::Integer(7))
        );
    }

    #[test]
    fn durable_companion_and_sparse_property_commit_reopen_as_one_batch() {
        let schema = schema();
        let snapshot = SceneInstancePersistence {
            structural: vec![prism_scene::StructuralOp::RemoveEntity {
                local: LocalEntityId::new(8),
            }],
            ..scene_snapshot()
        };
        let identities = PersistentEntityMap::new(vec![EntityBinding::new(
            LocalEntityId::new(3),
            PersistentId::from_u128(77),
        )])
        .unwrap();
        let prior = WorldState::new();
        let batch = schema
            .snapshot_to_deltas(&snapshot, &identities, &entity_map(), &prior)
            .unwrap();
        let transactions = schema
            .transactions_for_batch_with_companion(
                &batch,
                &prior,
                ActorId(2),
                LogicalClock(1),
                0,
                &[TransactionId(1)],
                TransactionId(2),
                WorldPersistentId(9900),
                FieldId(900),
            )
            .unwrap();
        assert_eq!(transactions.len(), 2);
        assert_eq!(transactions[0].sequence(), 0);
        assert_eq!(transactions[1].sequence(), 1);
        assert!(matches!(
            transactions[1].delta().get(FieldId(900)),
            Some(WorldValue::Bytes(_))
        ));

        let path = std::env::temp_dir().join(format!(
            "prism-scene-companion-{}-{}.pwrj",
            std::process::id(),
            TransactionId(2).0
        ));
        {
            let mut journal = FileJournal::open(&path).unwrap();
            journal.append_batch(&transactions).unwrap();
        }
        let reopened = FileJournal::open(&path).unwrap();
        let replayed =
            Timeline::replay(&Snapshot::new(0, WorldState::new()), reopened.log()).unwrap();
        let bytes = match replayed.get(WorldPersistentId(9900), FieldId(900)).unwrap() {
            WorldValue::Bytes(bytes) => bytes,
            other => panic!("unexpected companion value: {other:?}"),
        };
        assert_eq!(
            prism_scene::decode_persistence(bytes, Default::default()).unwrap(),
            batch.scene_snapshot
        );
        assert_eq!(
            replayed.get(WorldPersistentId(7700), FieldId(101)),
            Some(&WorldValue::Integer(42))
        );
        let restored = schema
            .snapshot_from_world_state(
                &replayed,
                WorldPersistentId(9900),
                FieldId(900),
                &identities,
                &entity_map(),
                Default::default(),
            )
            .unwrap();
        assert_eq!(restored.template, batch.scene_snapshot.template);
        assert_eq!(restored.instance, batch.scene_snapshot.instance);
        assert_eq!(restored.structural, batch.scene_snapshot.structural);
        assert_eq!(restored.properties.len(), 1);
        assert_eq!(restored.properties[0].local, LocalEntityId::new(3));
        assert_eq!(restored.properties[0].persistent, Some(PersistentId::from_u128(77)));
        assert_eq!(
            restored.properties[0].value,
            OverrideValue::I64(42)
        );
        drop(reopened);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn companion_restore_fails_closed_on_missing_or_wrong_kind_value() {
        let schema = schema();
        let identities = PersistentEntityMap::new(vec![EntityBinding::new(
            LocalEntityId::new(3),
            PersistentId::from_u128(77),
        )])
        .unwrap();
        let entities = entity_map();
        let empty = WorldState::new();
        assert_eq!(
            schema
                .snapshot_from_world_state_if_present(
                    &empty,
                    WorldPersistentId(9900),
                    FieldId(900),
                    &identities,
                    &entities,
                    Default::default(),
                )
                .unwrap(),
            None
        );
        assert_eq!(
            schema.snapshot_from_world_state(
                &empty,
                WorldPersistentId(9900),
                FieldId(900),
                &identities,
                &entities,
                Default::default(),
            ),
            Err(ScenePersistenceBridgeError::MissingCompanionSnapshot)
        );
        let mut timeline = Timeline::new();
        timeline
            .commit(
                TransactionId(10),
                ActorId(1),
                LogicalClock(1),
                WorldPersistentId(9900),
                OverrideDelta::single(FieldId(900), WorldValue::Integer(8)).unwrap(),
            )
            .unwrap();
        assert_eq!(
            schema.snapshot_from_world_state(
                timeline.state(),
                WorldPersistentId(9900),
                FieldId(900),
                &identities,
                &entities,
                Default::default(),
            ),
            Err(ScenePersistenceBridgeError::InvalidCompanionValueKind)
        );
        timeline
            .commit(
                TransactionId(11),
                ActorId(1),
                LogicalClock(2),
                WorldPersistentId(9900),
                OverrideDelta::single(FieldId(900), WorldValue::Bytes(vec![0, 1, 2])).unwrap(),
            )
            .unwrap();
        assert_eq!(
            schema.snapshot_from_world_state_if_present(
                timeline.state(),
                WorldPersistentId(9900),
                FieldId(900),
                &identities,
                &entities,
                Default::default(),
            ),
            Err(ScenePersistenceBridgeError::InvalidCompanionSnapshot)
        );
    }

    #[test]
    fn companion_builder_elides_unchanged_bytes_and_rejects_duplicate_ids() {
        let schema = schema();
        let snapshot = SceneInstancePersistence {
            properties: Vec::new(),
            structural: Vec::new(),
            ..scene_snapshot()
        };
        let entities = entity_map();
        let identities = PersistentEntityMap::new(vec![EntityBinding::new(
            LocalEntityId::new(3),
            PersistentId::from_u128(77),
        )])
        .unwrap();
        let prior = WorldState::new();
        let batch = schema
            .snapshot_to_deltas(&snapshot, &identities, &entities, &prior)
            .unwrap();
        let encoded = encode_persistence(&batch.scene_snapshot).unwrap();
        let mut prior_timeline = Timeline::new();
        prior_timeline
            .commit(
                TransactionId(2),
                ActorId(1),
                LogicalClock(1),
                WorldPersistentId(9900),
                OverrideDelta::single(FieldId(900), WorldValue::Bytes(encoded)).unwrap(),
            )
            .unwrap();
        let prior = prior_timeline.state().clone();
        assert!(schema
            .transactions_for_batch_with_companion(
                &batch,
                &prior,
                ActorId(1),
                LogicalClock(2),
                0,
                &[],
                TransactionId(3),
                WorldPersistentId(9900),
                FieldId(900),
            )
            .unwrap()
            .is_empty());
        assert_eq!(
            schema.transactions_for_batch(
                &ScenePersistenceBatch {
                    deltas: alloc::vec![
                        (
                            WorldPersistentId(8),
                            OverrideDelta::single(FieldId(1), WorldValue::Integer(1)).unwrap()
                        ),
                        (
                            WorldPersistentId(9),
                            OverrideDelta::single(FieldId(2), WorldValue::Integer(2)).unwrap()
                        ),
                    ],
                    scene_snapshot: batch.scene_snapshot,
                },
                &WorldState::new(),
                ActorId(1),
                LogicalClock(1),
                0,
                &[TransactionId(4), TransactionId(4)],
            ),
            Err(ScenePersistenceBridgeError::DuplicateTransactionId)
        );
    }

    #[test]
    fn snapshot_diff_emits_delete_for_removed_override_and_skips_unchanged_values() {
        let schema = schema();
        let identities = PersistentEntityMap::new(vec![EntityBinding::new(
            LocalEntityId::new(3),
            PersistentId::from_u128(77),
        )])
        .unwrap();
        let entities = entity_map();
        let mut prior_timeline = Timeline::new();
        prior_timeline
            .commit(
                TransactionId(20),
                ActorId(2),
                LogicalClock(1),
                WorldPersistentId(7700),
                OverrideDelta::single(FieldId(101), WorldValue::Integer(42)).unwrap(),
            )
            .unwrap();
        let prior = prior_timeline.state().clone();
        let removed = SceneInstancePersistence {
            properties: Vec::new(),
            ..scene_snapshot()
        };
        let removal = schema
            .snapshot_to_deltas(&removed, &identities, &entities, &prior)
            .unwrap();
        assert_eq!(removal.deltas.len(), 1);
        assert_eq!(
            removal.deltas[0].1.get(FieldId(101)),
            Some(&WorldValue::Delete)
        );
        let unchanged = schema
            .snapshot_to_deltas(&scene_snapshot(), &identities, &entities, &prior)
            .unwrap();
        assert!(unchanged.deltas.is_empty());
    }

    #[test]
    fn bridge_rejects_ambiguous_schema_and_unsupported_snapshot_content() {
        let binding = SceneFieldBinding {
            component: ComponentTypeId::from_raw(11),
            path: FieldPath::parse("health.current").unwrap(),
            field: FieldId(101),
            value_kind: SceneFieldValueKind::I64,
        };
        assert!(matches!(
            ScenePersistenceSchema::new(vec![binding.clone(), binding]),
            Err(ScenePersistenceBridgeError::DuplicateSceneField)
        ));
        let identities = PersistentEntityMap::new(vec![EntityBinding::new(
            LocalEntityId::new(3),
            PersistentId::from_u128(77),
        )])
        .unwrap();
        let prior = WorldState::new();
        let mut snapshot = scene_snapshot();
        let batch = schema()
            .snapshot_to_deltas(&snapshot, &identities, &entity_map(), &prior)
            .unwrap();
        assert_eq!(batch.scene_snapshot.structural, snapshot.structural);
        assert_eq!(
            batch.deltas[0].1.get(FieldId(101)),
            Some(&WorldValue::Integer(42))
        );
        snapshot
            .structural
            .push(prism_scene::StructuralOp::RemoveEntity {
                local: LocalEntityId::new(4),
            });
        let batch = schema()
            .snapshot_to_deltas(&snapshot, &identities, &entity_map(), &prior)
            .unwrap();
        assert_eq!(batch.scene_snapshot.structural, snapshot.structural);
        assert!(batch.scene_snapshot.properties.is_empty());
        snapshot.structural.clear();
        snapshot.properties[0].persistent = None;
        assert_eq!(
            schema().snapshot_to_deltas(&snapshot, &identities, &entity_map(), &prior),
            Err(ScenePersistenceBridgeError::MissingEntityIdentity)
        );

        let mut incompatible = scene_snapshot();
        incompatible.properties[0].value = OverrideValue::Str("42".into());
        assert_eq!(
            schema().snapshot_to_deltas(&incompatible, &identities, &entity_map(), &prior),
            Err(ScenePersistenceBridgeError::IncompatibleValue)
        );

        assert!(matches!(
            SceneEntityMap::new(vec![
                SceneObjectBinding {
                    scene: PersistentId::from_u128(77),
                    world: WorldPersistentId(7700),
                },
                SceneObjectBinding {
                    scene: PersistentId::from_u128(78),
                    world: WorldPersistentId(7700),
                },
            ]),
            Err(ScenePersistenceBridgeError::DuplicateWorldEntity)
        ));
        assert!(SceneEntityMap::new(vec![SceneObjectBinding {
            scene: PersistentId::from_u128(79),
            world: WorldPersistentId(0),
        }])
        .is_ok());
    }
}

fn to_world_value(
    value: &SceneValue,
    kind: SceneFieldValueKind,
) -> Result<WorldValue, ScenePersistenceBridgeError> {
    match (value, kind) {
        (SceneValue::Bool(value), SceneFieldValueKind::Bool) => Ok(WorldValue::Boolean(*value)),
        (SceneValue::I64(value), SceneFieldValueKind::I64) => Ok(WorldValue::Integer(*value)),
        (SceneValue::U64(value), SceneFieldValueKind::U64) => Ok(WorldValue::Unsigned(*value)),
        (SceneValue::F32(value), SceneFieldValueKind::F32) => WorldValue::float(f64::from(*value))
            .map_err(|_| ScenePersistenceBridgeError::IncompatibleValue),
        (SceneValue::F64(value), SceneFieldValueKind::F64) if value.is_finite() => {
            WorldValue::float(*value).map_err(|_| ScenePersistenceBridgeError::IncompatibleValue)
        }
        (SceneValue::Str(value), SceneFieldValueKind::String) => {
            Ok(WorldValue::Bytes(value.as_bytes().to_vec()))
        }
        (SceneValue::Bytes(value), SceneFieldValueKind::Bytes) => {
            Ok(WorldValue::Bytes(value.clone()))
        }
        _ => Err(ScenePersistenceBridgeError::IncompatibleValue),
    }
}

fn from_world_value(
    value: &WorldValue,
    kind: SceneFieldValueKind,
) -> Result<SceneValue, ScenePersistenceBridgeError> {
    match (value, kind) {
        (WorldValue::Boolean(value), SceneFieldValueKind::Bool) => Ok(SceneValue::Bool(*value)),
        (WorldValue::Integer(value), SceneFieldValueKind::I64) => Ok(SceneValue::I64(*value)),
        (WorldValue::Unsigned(value), SceneFieldValueKind::U64) => Ok(SceneValue::U64(*value)),
        (WorldValue::Float(value), SceneFieldValueKind::F32) if value.is_finite() => {
            let converted = *value as f32;
            if f64::from(converted) == *value {
                Ok(SceneValue::F32(converted))
            } else {
                Err(ScenePersistenceBridgeError::IncompatibleValue)
            }
        }
        (WorldValue::Float(value), SceneFieldValueKind::F64) if value.is_finite() => {
            Ok(SceneValue::F64(*value))
        }
        (WorldValue::Bytes(value), SceneFieldValueKind::String) => {
            let text = core::str::from_utf8(value)
                .map_err(|_| ScenePersistenceBridgeError::IncompatibleValue)?;
            Ok(SceneValue::Str(text.into()))
        }
        (WorldValue::Bytes(value), SceneFieldValueKind::Bytes) => {
            Ok(SceneValue::Bytes(value.clone()))
        }
        _ => Err(ScenePersistenceBridgeError::IncompatibleValue),
    }
}
