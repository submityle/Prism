use std::collections::{BTreeMap, BTreeSet};

use prism_ecs::{
    component::{Component, ComponentId, StorageType},
    entity::Entity,
    partition::{
        cell::{CellCoord, CellState, StreamingDelta},
        data_layer::DataLayers,
        floating_origin::WorldPos,
        persistence::{entity_component_ids, has_lifecycle_callbacks},
        world_partition::WorldPartition,
    },
    prefab::{reflect::PrefabOverrides, IsA},
    reflect_archive::{ReflectArchiveError, ReflectArchiveLoadOptions, ReflectArchiveRemapFn},
    relation::{CleanupPolicy, RelationKind},
    world::{World, WorldId},
};
use prism_reflect::{Reflect, TypeRegistry};

use super::{
    codec::{CellArchive, Edge, Member},
    CellStore,
};
use crate::PersistentId;

/// Serializable cross-cell reference. No runtime Entity is cached in the
/// component: resolution always checks the owner's World identity, restore
/// generation and the current entity generation.
#[derive(Reflect, Clone, Copy, Debug, PartialEq, Eq)]
pub struct PersistentEntityRef {
    id: u128,
}

impl PersistentEntityRef {
    /// Construct a reference; NIL is an explicitly unresolved reference.
    pub const fn new(id: PersistentId) -> Self {
        Self { id: id.to_u128() }
    }

    /// Stable identity independent of cell load order or handle recycling.
    pub const fn id(self) -> PersistentId {
        PersistentId::from_u128(self.id)
    }

    /// Resolve only while this identity is resident in the owner's World.
    pub fn resolve<S: CellStore>(
        self,
        owner: &PersistentPartition<S>,
        world: &World,
    ) -> Option<Entity> {
        owner.resolve(world, self.id())
    }
}

/// Persistence failure; all ordinary save failures precede world/index edits.
#[derive(Debug)]
pub enum PartitionError {
    /// Filesystem/store error (including an absent archive).
    Io(std::io::Error),
    /// Reflection/schema/archive validation error.
    Archive(ReflectArchiveError),
    /// Malformed or internally inconsistent envelope.
    InvalidArchive(&'static str),
    /// Owner was used after World replacement or wholesale snapshot restore.
    WorldChanged,
    /// An entity was despawned outside the owner's lifecycle bookkeeping.
    MissingEntity(Entity),
    /// Caller attempted to reuse a live stable identity.
    DuplicateId(PersistentId),
    /// Bad registration, pending relation edit or cell lifecycle transition.
    InvalidState(&'static str),
    /// The current archive cannot losslessly preserve this concrete component.
    UnsupportedComponent(String),
    /// Arbitrary hooks/observers prevent a transactional persistence boundary.
    LifecycleCallbacks,
    /// Destination must register a relation kind with this stable type name.
    UnknownRelation(String),
    /// Persisted and registered relation policies differ.
    RelationPolicy(String),
    /// An exclusive source would point at multiple persistent targets.
    ExclusiveConflict(String),
}

impl std::fmt::Display for PartitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "partition persistence: {self:?}")
    }
}

impl std::error::Error for PartitionError {}

impl From<std::io::Error> for PartitionError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<ReflectArchiveError> for PartitionError {
    fn from(error: ReflectArchiveError) -> Self {
        Self::Archive(error)
    }
}

/// Failed side of a streaming request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellOperation {
    /// Save-before-unload failed; the cell remains resident and indexed.
    Unload,
    /// Read/restore failed; no entities were settled and it can retry.
    Load,
}

/// One failure does not suppress independent successful cell requests.
#[derive(Debug)]
pub struct CellFailure {
    /// Cell whose request failed.
    pub cell: CellCoord,
    /// Load or unload.
    pub operation: CellOperation,
    /// Exact validation or storage failure.
    pub error: PartitionError,
}

/// Actual completed work from a production streaming frame.
#[derive(Debug, Default)]
pub struct PartitionTick {
    /// Successfully restored and settled cells with their fresh handles.
    pub loaded: Vec<(CellCoord, Vec<Entity>)>,
    /// Cells saved durably and evicted through the real driver.
    pub unloaded: Vec<CellCoord>,
    /// Entities actually despawned, grouped in eviction order.
    pub despawned: Vec<Entity>,
    /// Failed requests; subsequent frames retry while still desired.
    pub failed: Vec<CellFailure>,
}

/// Scene-owned synchronous durable WorldPartition consumer. The driver is
/// intentionally exposed read-only: bypassing save-before-evict would void the
/// persistence contract. I/O is synchronous, so no stale asynchronous settle
/// callback can attach entities to a cancelled request.
pub struct PersistentPartition<S: CellStore> {
    partition: WorldPartition,
    store: S,
    world_id: WorldId,
    generation: u64,
    residents: BTreeMap<PersistentId, Entity>,
    identities: BTreeMap<Entity, PersistentId>,
    // Only resident sources are retained. Unloaded source edges live on disk,
    // not forever in a second in-memory copy of the visited world.
    edges: BTreeSet<Edge>,
    types: TypeRegistry,
    remap: Option<ReflectArchiveRemapFn>,
}

impl<S: CellStore> PersistentPartition<S> {
    /// Bind one owner to one current World; cell size and radii follow the
    /// ordinary WorldPartition contract. All component/relation registrations
    /// remain application-owned and must exist before loading an archive.
    pub fn new(
        world: &World,
        store: S,
        cell_size: f64,
        load_radius: u32,
        unload_radius: u32,
    ) -> Self {
        let mut types = TypeRegistry::new();
        types.register::<PersistentEntityRef>();
        Self {
            partition: WorldPartition::new(cell_size, load_radius, unload_radius),
            store,
            world_id: world.id(),
            generation: world.archetype_generation(),
            residents: BTreeMap::new(),
            identities: BTreeMap::new(),
            edges: BTreeSet::new(),
            types,
            remap: None,
        }
    }

    /// Current real kernel driver, index, origin and cell states, read-only.
    pub fn partition(&self) -> &WorldPartition {
        &self.partition
    }

    /// Scene-owned archive backend.
    pub fn store(&self) -> &S {
        &self.store
    }

    /// Change backend policy or repair a failed store before explicit retry.
    pub fn store_mut(&mut self) -> &mut S {
        &mut self.store
    }

    /// Register nested reflected types used by application component fields.
    pub fn types_mut(&mut self) -> &mut TypeRegistry {
        &mut self.types
    }

    /// Optional application remapping of raw *intra-cell* Entity fields.
    /// Cross-cell fields must use persistent IDs: archived runtime handles do
    /// not identify entities loaded from another cell in another session.
    /// The native archive's map excludes empty standalone entities; references
    /// to those entities must also use [`PersistentEntityRef`].
    pub fn set_remap(&mut self, remap: Option<ReflectArchiveRemapFn>) {
        self.remap = remap;
    }

    /// Actual WorldPartition data-layer gate used by [`Self::advance`].
    pub fn data_layers_mut(&mut self) -> &mut DataLayers {
        self.partition.data_layers_mut()
    }

    fn check_world(&self, world: &World) -> Result<(), PartitionError> {
        if world.id() != self.world_id || world.archetype_generation() != self.generation {
            return Err(PartitionError::WorldChanged);
        }
        Ok(())
    }

    fn check_transaction(&self, world: &World) -> Result<(), PartitionError> {
        self.check_world(world)?;
        if has_lifecycle_callbacks(world) {
            return Err(PartitionError::LifecycleCallbacks);
        }
        Ok(())
    }

    /// Adopt a spawned entity and assign its stable identity and resident cell.
    /// Re-registering the same identity/handle can move it between cells; save
    /// both cells to persist that authoring change (not a multi-file commit).
    pub fn register(
        &mut self,
        world: &World,
        cell: CellCoord,
        entity: Entity,
        id: PersistentId,
    ) -> Result<(), PartitionError> {
        self.check_world(world)?;
        if id.is_nil() {
            return Err(PartitionError::InvalidState("NIL resident identity"));
        }
        if entity_component_ids(world, entity).is_none() {
            return Err(PartitionError::MissingEntity(entity));
        }
        if self
            .residents
            .get(&id)
            .is_some_and(|&current| current != entity)
        {
            return Err(PartitionError::DuplicateId(id));
        }
        if self
            .identities
            .get(&entity)
            .is_some_and(|&current| current != id)
        {
            return Err(PartitionError::InvalidState(
                "entity already has a different stable identity",
            ));
        }
        self.residents.insert(id, entity);
        self.identities.insert(entity, id);
        self.partition.driver_mut().settle_loaded(cell, &[entity]);
        Ok(())
    }

    /// Explicitly author an empty resident cell. Missing archive loads never
    /// take this path implicitly. Save it to establish an empty disk cell.
    pub fn create_empty_cell(
        &mut self,
        world: &World,
        cell: CellCoord,
    ) -> Result<(), PartitionError> {
        self.check_world(world)?;
        if !self.partition.index().entities_in(cell).is_empty() {
            return Err(PartitionError::InvalidState("cell is not empty"));
        }
        self.partition.driver_mut().settle_loaded(cell, &[]);
        Ok(())
    }

    /// Resolve a live identity, never stale handles after unload/replacement.
    pub fn resolve(&self, world: &World, id: PersistentId) -> Option<Entity> {
        self.check_world(world).ok()?;
        let entity = *self.residents.get(&id)?;
        world.contains(entity).then_some(entity)
    }

    /// Add a persistent ordinary relation, allowing an unloaded target. The
    /// source must be resident. Exclusive edits replace both live and pending
    /// targets; use this API to edit relations while their target is absent.
    pub fn link<R: Component>(
        &mut self,
        world: &mut World,
        source: PersistentId,
        target: PersistentId,
    ) -> Result<(), PartitionError> {
        self.check_world(world)?;
        if self.resolve(world, source).is_none() || target.is_nil() {
            return Err(PartitionError::InvalidState(
                "relation needs a resident source and non-NIL target",
            ));
        }
        let relation = world
            .components()
            .id_of::<R>()
            .ok_or_else(|| PartitionError::UnknownRelation(core::any::type_name::<R>().into()))?;
        let edge = describe_edge(world, relation, source, target)?;
        let mut edges = self.current_edges(world)?;
        if edge.policy & 4 != 0 {
            edges.retain(|old| old.source != source || old.name != edge.name);
        }
        edges.insert(edge);
        validate_edges(world, &edges)?;
        apply_edges(world, &edges, &self.residents);
        self.edges = edges;
        Ok(())
    }

    /// Remove a live or pending source-owned relation. Load its source cell
    /// before editing it; this API never silently changes an on-disk archive.
    pub fn unlink<R: Component>(
        &mut self,
        world: &mut World,
        source: PersistentId,
        target: PersistentId,
    ) -> Result<bool, PartitionError> {
        self.check_world(world)?;
        if self.resolve(world, source).is_none() {
            return Err(PartitionError::InvalidState(
                "relation source is not resident",
            ));
        }
        let mut edges = self.current_edges(world)?;
        let old_len = edges.len();
        edges.retain(|edge| {
            edge.name != core::any::type_name::<R>()
                || edge.source != source
                || edge.target != target
        });
        validate_edges(world, &edges)?;
        apply_edges(world, &edges, &self.residents);
        self.edges = edges;
        Ok(self.edges.len() != old_len)
    }

    // Pending edges survive only for resident sources and absent targets.
    // Live ECS edges are authoritative, including direct removal/reparenting.
    fn current_edges(&self, world: &World) -> Result<BTreeSet<Edge>, PartitionError> {
        for (&id, &entity) in &self.residents {
            if self.resolve(world, id) != Some(entity) {
                return Err(PartitionError::MissingEntity(entity));
            }
        }
        let mut pending: BTreeSet<_> = self
            .edges
            .iter()
            .filter(|edge| {
                self.residents.contains_key(&edge.source)
                    && self.resolve(world, edge.target).is_none()
            })
            .cloned()
            .collect();
        let isa = world.components().id_of::<IsA>();
        let mut actual = BTreeSet::new();
        for (relation, source, target) in world.relations().index().iter_edges() {
            let source_id = self.identities.get(&source);
            let target_id = self.identities.get(&target);
            if source_id.is_none() && target_id.is_none() {
                continue;
            }
            let (&source_id, &target_id) =
                source_id
                    .zip(target_id)
                    .ok_or(PartitionError::InvalidState(
                        "relation crosses the persistent owner's entity boundary",
                    ))?;
            if Some(relation) == isa {
                if self.partition.index().cell_of(source) != self.partition.index().cell_of(target)
                {
                    return Err(PartitionError::InvalidState(
                        "prefab ancestors must be in the same persistent cell",
                    ));
                }
                continue; // Reflection archive owns internal prefab ancestry.
            }
            actual.insert(describe_edge(world, relation, source_id, target_id)?);
        }
        // A live exclusive reparent supersedes an older pending target.
        let exclusive: BTreeSet<_> = actual
            .iter()
            .filter(|edge| edge.policy & 4 != 0)
            .map(|edge| (edge.name.as_str(), edge.source))
            .collect();
        pending.retain(|edge| !exclusive.contains(&(edge.name.as_str(), edge.source)));
        pending.extend(actual);
        validate_edges(world, &pending)?;
        Ok(pending)
    }

    fn validate_entity(world: &World, entity: Entity) -> Result<(), PartitionError> {
        let overrides = world.components().id_of::<PrefabOverrides>();
        for id in
            entity_component_ids(world, entity).ok_or(PartitionError::MissingEntity(entity))?
        {
            let info = world
                .components()
                .info(id)
                .ok_or(PartitionError::InvalidState("unknown component"))?;
            if info.storage() == StorageType::Shared
                || (Some(id) != overrides && world.reflect_stable_id(id).is_none())
            {
                return Err(PartitionError::UnsupportedComponent(info.name().into()));
            }
        }
        Ok(())
    }

    /// Persist a resident cell without unloading it. No mutation of World or
    /// cell index happens even when serialization, write, rename or fsync fails.
    pub fn save_cell(&mut self, world: &World, cell: CellCoord) -> Result<(), PartitionError> {
        self.check_transaction(world)?;
        if !matches!(
            self.partition.driver().streamer().state(cell),
            CellState::Loaded | CellState::Unloading
        ) {
            return Err(PartitionError::InvalidState(
                "cannot save a nonresident cell",
            ));
        }
        let edges = self.current_edges(world)?;
        let entities = self.partition.index().entities_in(cell);
        let ancestors: BTreeSet<_> = entities
            .iter()
            .flat_map(|&entity| world.prefabs_of(entity).iter().copied())
            .collect();
        let mut members = Vec::with_capacity(entities.len());
        for &entity in &entities {
            Self::validate_entity(world, entity)?;
            let id = *self
                .identities
                .get(&entity)
                .ok_or(PartitionError::InvalidState(
                    "indexed entity has no persistent identity",
                ))?;
            let archived = !world.reflect_entity(entity).is_empty()
                || !world.prefabs_of(entity).is_empty()
                || world
                    .get::<PrefabOverrides>(entity)
                    .is_some_and(|overrides| overrides.components().next().is_some());
            members.push(Member {
                source: entity,
                id,
                archived,
            });
        }
        let ids: BTreeSet<_> = members.iter().map(|member| member.id).collect();
        // Empty ancestors are emitted by the archive even if they would be
        // omitted as standalone roots. The entire closure is cell-local.
        for member in &mut members {
            if ancestors.contains(&member.source) {
                member.archived = true;
            }
        }
        let payload = world.save_reflect_archive_with_types(entities, &self.types)?;
        let archive = CellArchive {
            cell,
            members,
            edges: edges
                .iter()
                .filter(|edge| ids.contains(&edge.source))
                .cloned()
                .collect(),
            payload,
        };
        let bytes = archive.encode()?;
        self.store.write_atomic(cell, &bytes)?;
        self.edges = edges;
        Ok(())
    }

    /// Save then detach incident edges and evict through the real ECS driver.
    /// Relation detachment prevents cross-cell cleanup policies from deleting
    /// still-resident neighbours. Failed saves leave both World and index intact.
    pub fn unload_cell(
        &mut self,
        world: &mut World,
        cell: CellCoord,
    ) -> Result<Vec<Entity>, PartitionError> {
        self.check_world(world)?;
        if let Err(error) = self.save_cell(world, cell) {
            self.partition.driver_mut().cancel_unload(cell);
            return Err(error);
        }
        let entities = self.partition.index().entities_in(cell);
        detach(world, &entities);
        let despawned = self.partition.driver_mut().evict_cell(world, cell);
        for entity in entities {
            if let Some(id) = self.identities.remove(&entity) {
                self.residents.remove(&id);
            }
        }
        self.edges
            .retain(|edge| self.residents.contains_key(&edge.source));
        Ok(despawned)
    }

    /// Read and reconstruct fresh entities, resolve persistent relations, then
    /// settle Loaded. Ordinary failures settle nothing; failed Loading state
    /// is cancelled so either this method or the next frame can retry.
    pub fn load_cell(
        &mut self,
        world: &mut World,
        cell: CellCoord,
    ) -> Result<Vec<Entity>, PartitionError> {
        self.check_world(world)?;
        let result = self.load_inner(world, cell);
        if result.is_err() {
            self.partition.driver_mut().cancel_load(cell);
        }
        result
    }

    fn load_inner(
        &mut self,
        world: &mut World,
        cell: CellCoord,
    ) -> Result<Vec<Entity>, PartitionError> {
        self.check_transaction(world)?;
        if matches!(
            self.partition.driver().streamer().state(cell),
            CellState::Loaded | CellState::Unloading
        ) || !self.partition.index().entities_in(cell).is_empty()
        {
            return Err(PartitionError::InvalidState("cell is already resident"));
        }
        let bytes = self.store.read(cell)?;
        let archive = CellArchive::decode(&bytes, cell)?;
        for member in &archive.members {
            if self.residents.contains_key(&member.id) {
                return Err(PartitionError::DuplicateId(member.id));
            }
        }
        let mut edges = self.current_edges(world)?;
        edges.extend(archive.edges);
        validate_edges(world, &edges)?;
        let landed = world.load_reflect_archive_with_options(
            &archive.payload,
            ReflectArchiveLoadOptions {
                spawn_fresh: true,
                types: Some(&self.types),
                remap: self.remap,
                ..Default::default()
            },
        )?;
        let mut allocated: Vec<_> = landed.iter().map(|&(_, target)| target).collect();
        let result = (|| {
            let mapping: BTreeMap<_, _> = landed.into_iter().collect();
            let expected: BTreeSet<_> = archive
                .members
                .iter()
                .filter(|member| member.archived)
                .map(|member| member.source)
                .collect();
            if mapping.keys().copied().collect::<BTreeSet<_>>() != expected {
                return Err(PartitionError::InvalidArchive(
                    "payload/member set mismatch",
                ));
            }
            let mut bindings = Vec::with_capacity(archive.members.len());
            for member in archive.members {
                let entity = if member.archived {
                    mapping[&member.source]
                } else {
                    let entity = world.spawn(());
                    allocated.push(entity);
                    entity
                };
                Self::validate_entity(world, entity)?;
                bindings.push((member.id, entity));
            }
            Ok(bindings)
        })();
        let bindings = match result {
            Ok(bindings) => bindings,
            Err(error) => {
                detach(world, &allocated);
                for entity in allocated {
                    world.despawn(entity);
                }
                return Err(error);
            }
        };
        for &(id, entity) in &bindings {
            self.residents.insert(id, entity);
            self.identities.insert(entity, id);
        }
        apply_edges(world, &edges, &self.residents);
        self.edges = edges;
        let entities: Vec<_> = bindings.into_iter().map(|(_, entity)| entity).collect();
        self.partition.driver_mut().settle_loaded(cell, &entities);
        Ok(entities)
    }

    /// Production camera/data-layer consumer. Save/unload precedes load; each
    /// cell reports success or failure independently. Failures are retried on
    /// the next desired plan; no explicit queue can resurrect obsolete demand.
    pub fn advance(
        &mut self,
        world: &mut World,
        camera: WorldPos,
        extra: &[WorldPos],
    ) -> Result<PartitionTick, PartitionError> {
        self.check_world(world)?;
        let StreamingDelta { to_load, to_unload } = self.partition.plan(camera, extra);
        let mut tick = PartitionTick::default();
        for cell in to_unload {
            match self.unload_cell(world, cell) {
                Ok(entities) => {
                    tick.unloaded.push(cell);
                    tick.despawned.extend(entities);
                }
                Err(error) => tick.failed.push(CellFailure {
                    cell,
                    operation: CellOperation::Unload,
                    error,
                }),
            }
        }
        for cell in to_load {
            match self.load_cell(world, cell) {
                Ok(entities) => tick.loaded.push((cell, entities)),
                Err(error) => tick.failed.push(CellFailure {
                    cell,
                    operation: CellOperation::Load,
                    error,
                }),
            }
        }
        Ok(tick)
    }
}

fn policy_code(kind: &RelationKind) -> u8 {
    fn cleanup(policy: CleanupPolicy) -> u8 {
        match policy {
            CleanupPolicy::Remove => 0,
            CleanupPolicy::Delete => 1,
            CleanupPolicy::Panic => 2,
        }
    }
    u8::from(kind.fragmenting)
        | (u8::from(kind.transitive) << 1)
        | (u8::from(kind.exclusive) << 2)
        | (cleanup(kind.on_delete) << 3)
        | (cleanup(kind.on_delete_target) << 5)
}

fn describe_edge(
    world: &World,
    relation: ComponentId,
    source: PersistentId,
    target: PersistentId,
) -> Result<Edge, PartitionError> {
    if world.components().id_of::<IsA>() == Some(relation) {
        return Err(PartitionError::InvalidState(
            "IsA is owned by the reflection archive",
        ));
    }
    let info = world
        .components()
        .info(relation)
        .ok_or(PartitionError::InvalidState("missing relation component"))?;
    let kind = world
        .relations()
        .kind(relation)
        .ok_or_else(|| PartitionError::UnknownRelation(info.name().into()))?;
    Ok(Edge {
        name: info.name().into(),
        policy: policy_code(kind),
        source,
        target,
    })
}

fn edge_component(world: &World, edge: &Edge) -> Result<ComponentId, PartitionError> {
    for (id, kind) in world.relations().iter_kinds() {
        if world
            .components()
            .info(id)
            .is_some_and(|info| info.name() == edge.name)
        {
            if policy_code(kind) != edge.policy {
                return Err(PartitionError::RelationPolicy(edge.name.clone()));
            }
            if world.components().id_of::<IsA>() == Some(id) {
                return Err(PartitionError::InvalidArchive(
                    "IsA cannot be an ordinary persisted relation",
                ));
            }
            return Ok(id);
        }
    }
    Err(PartitionError::UnknownRelation(edge.name.clone()))
}

fn validate_edges(world: &World, edges: &BTreeSet<Edge>) -> Result<(), PartitionError> {
    let mut exclusive = BTreeSet::new();
    for edge in edges {
        edge_component(world, edge)?;
        if edge.policy & 4 != 0 && !exclusive.insert((&edge.name, edge.source)) {
            return Err(PartitionError::ExclusiveConflict(edge.name.clone()));
        }
    }
    Ok(())
}

fn apply_edges(
    world: &mut World,
    edges: &BTreeSet<Edge>,
    residents: &BTreeMap<PersistentId, Entity>,
) {
    // Resolve all metadata before opening the bulk mutation guard. Callers
    // validate first, and no application code executes between these steps.
    let resolved: Vec<_> = edges
        .iter()
        .filter_map(|edge| {
            Some((
                edge_component(world, edge).expect("validated relation metadata"),
                *residents.get(&edge.source)?,
                *residents.get(&edge.target)?,
            ))
        })
        .collect();
    let owned: BTreeSet<_> = residents.values().copied().collect();
    let isa = world.components().id_of::<IsA>();
    let remove: Vec<_> = world
        .relations()
        .index()
        .iter_edges()
        .filter(|&(relation, source, _)| Some(relation) != isa && owned.contains(&source))
        .collect();
    let mut relations = world.relations_mut();
    for (relation, source, target) in remove {
        relations.remove(relation, source, target);
    }
    for (relation, source, target) in resolved {
        relations.add(relation, source, target);
    }
}

fn detach(world: &mut World, entities: &[Entity]) {
    let entities: BTreeSet<_> = entities.iter().copied().collect();
    let incident: Vec<_> = world
        .relations()
        .index()
        .iter_edges()
        .filter(|&(_, source, target)| entities.contains(&source) || entities.contains(&target))
        .collect();
    let mut relations = world.relations_mut();
    for (relation, source, target) in incident {
        relations.remove(relation, source, target);
    }
}
