use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use prism_ecs::change::Tick;
use prism_ecs::component::Component;
use prism_ecs::entity::Entity;
use prism_ecs::hierarchy::ChildOf;
use prism_ecs::query::{QueryState, With};
use prism_ecs::resource::Resource;
use prism_ecs::world::{World, WorldId};

use crate::change::ChangeTicks;
use crate::dirty::DirtyPropagator;
use crate::hierarchy::{Hierarchy, HierarchyError, NodeId};
use crate::{GlobalTransform, Transform};

type Inputs = QueryState<Entity, With<Transform>>;

// The current relation API exposes edges, but no mutation revision. Comparing
// canonical ChildOf edges also detects raw index edits and registry replacement.
fn hierarchy_edges(world: &World) -> Vec<(Entity, Entity)> {
    let Some(child_of) = world.components().id_of::<ChildOf>() else {
        return Vec::new();
    };
    let mut edges = world
        .relations()
        .index()
        .iter_edges()
        .filter_map(|(relation, source, target)| (relation == child_of).then_some((source, target)))
        .collect::<Vec<_>>();
    edges.sort_unstable();
    edges
}

// Chunk versions are upper bounds on row ticks, so static chunks never need
// their individual rows inspected. No component value is borrowed here.
fn changed_entities<T: Component>(world: &World, last: Tick, now: Tick) -> BTreeSet<Entity> {
    let mut changed = BTreeSet::new();
    let Some(id) = world.components().id_of::<T>() else {
        return changed;
    };
    for archetype in world.archetypes().iter() {
        let table = archetype.table();
        let Some(column) = table.column(id) else {
            continue;
        };
        for chunk in 0..column.chunk_count() {
            if !column.chunk_version(chunk).is_newer_than(last, now) {
                continue;
            }
            let start = chunk * column.rows_per_chunk();
            let end = (start + column.rows_per_chunk()).min(table.entities().len());
            for row in start..end {
                if column.changed_tick(row).is_newer_than(last, now) {
                    changed.insert(table.entities()[row]);
                }
            }
        }
    }
    changed
}

pub(super) struct PropagationState {
    query: Inputs,
    world_id: WorldId,
    generation: u64,
    relation_edges: Vec<(Entity, Entity)>,
    entities: Vec<Entity>,
    indices: BTreeMap<Entity, NodeId>,
    parents: Vec<Option<NodeId>>,
    hierarchy: Hierarchy,
    locals: Vec<Transform>,
    globals: Vec<GlobalTransform>,
    ticks: ChangeTicks,
    propagator: DirtyPropagator,
    last_run: Tick,
}

impl Resource for PropagationState {}

impl PropagationState {
    pub(super) fn new(world: &mut World) -> Self {
        Self {
            query: world.query_filtered(),
            world_id: world.id(),
            generation: world.archetype_generation(),
            relation_edges: hierarchy_edges(world),
            entities: Vec::new(),
            indices: BTreeMap::new(),
            parents: Vec::new(),
            hierarchy: Hierarchy::new(),
            locals: Vec::new(),
            globals: Vec::new(),
            ticks: ChangeTicks::new(),
            propagator: DirtyPropagator::new(),
            last_run: Tick::ZERO,
        }
    }

    fn propagate(&mut self, world: &mut World) -> Result<EcsPropagationStats, TransformEcsError> {
        let now = world.change_tick();
        let generation = world.archetype_generation();
        let relation_edges = hierarchy_edges(world);
        let restored = self.generation != generation || self.world_id != world.id();
        if self.world_id != world.id() {
            *self = Self::new(world);
        }
        let signals = world.resource::<super::PropagationSignals>();
        let membership_dirty = signals.membership_dirty || restored;
        let mut stats = EcsPropagationStats::default();
        let new_entities = if membership_dirty {
            stats.membership_scans = 1;
            let mut entities = self.query.iter(world).collect::<Vec<_>>();
            entities.sort_unstable();
            Some(entities)
        } else {
            None
        };
        let entities = new_entities.as_ref().unwrap_or(&self.entities);
        let new_indices = new_entities.as_ref().map(|entities| {
            entities
                .iter()
                .enumerate()
                .map(|(i, &entity)| (entity, NodeId::new(i as u32)))
                .collect::<BTreeMap<_, _>>()
        });
        let indices = new_indices.as_ref().unwrap_or(&self.indices);
        let new_parents = if membership_dirty || self.relation_edges != relation_edges {
            let mut parents = Vec::with_capacity(entities.len());
            for &entity in entities {
                stats.relation_checks += 1;
                let targets = world.relation_targets::<ChildOf>(entity);
                if targets.len() > 1 {
                    return Err(TransformEcsError::MultipleParents(entity));
                }
                parents.push(
                    targets
                        .first()
                        .and_then(|parent| indices.get(parent).copied()),
                );
            }
            Some(parents)
        } else {
            None
        };
        let parents = new_parents.as_ref().unwrap_or(&self.parents);
        let topology_changed = new_entities
            .as_ref()
            .is_some_and(|entities| entities != &self.entities)
            || new_parents
                .as_ref()
                .is_some_and(|parents| parents != &self.parents);
        // A failed topology check must retain the previous cache, edges and
        // lifecycle signals so fixing the relation and retrying is sufficient.
        let hierarchy = if topology_changed {
            Some(Hierarchy::from_parents(parents).map_err(TransformEcsError::Hierarchy)?)
        } else {
            None
        };
        let locals_changed = changed_entities::<Transform>(world, self.last_run, now);
        let mut globals_changed = changed_entities::<GlobalTransform>(world, self.last_run, now);
        globals_changed.extend(signals.globals.iter().copied());
        let signals = world.resource_mut::<super::PropagationSignals>();
        signals.membership_dirty = false;
        signals.globals.clear();
        self.last_run = now;
        self.relation_edges = relation_edges;
        self.generation = generation;
        let mut removed = Vec::new();
        let mut dirty = false;
        if topology_changed || restored {
            let mut ticks = ChangeTicks::new();
            let mut locals = Vec::with_capacity(entities.len());
            let mut globals = Vec::with_capacity(entities.len());
            let mut marked = Vec::new();
            for (i, &entity) in entities.iter().enumerate() {
                ticks.push();
                let old = self.indices.get(&entity).map(|node| node.index());
                let local = if restored || old.is_none() || locals_changed.contains(&entity) {
                    stats.local_reads += 1;
                    *world.get::<Transform>(entity).expect("queried local")
                } else {
                    self.locals[old.expect("cached")]
                };
                let cached = old.map(|j| self.globals[j]);
                let current = world.get::<GlobalTransform>(entity).copied();
                let old_parent = old
                    .and_then(|j| self.parents[j])
                    .map(|parent| self.entities[parent.index()]);
                let new_parent = parents[i].map(|parent| entities[parent.index()]);
                if restored
                    || old.is_none_or(|j| self.locals[j] != local)
                    || old_parent != new_parent
                    || current != cached
                {
                    marked.push(NodeId::new(i as u32));
                }
                locals.push(local);
                globals.push(cached.unwrap_or(GlobalTransform::IDENTITY));
            }
            ticks.end_pass();
            dirty = !marked.is_empty();
            for node in marked {
                ticks.mark(node);
            }
            // A restored standalone global is authoritative snapshot data,
            // even if the abandoned timeline temporarily gave it a local.
            // Only ordinary lifecycle removals discard managed globals.
            if !restored {
                removed.extend(
                    self.entities
                        .iter()
                        .filter(|e| !indices.contains_key(e))
                        .copied(),
                );
            }
            self.locals = locals;
            self.globals = globals;
            self.ticks = ticks;
        } else {
            // Stable topology: touch only signalled rows. In particular our own
            // GlobalTransform publications and net-noop writes do not trigger
            // a full dense-buffer rebuild on the following call.
            self.ticks.end_pass();
            for entity in locals_changed {
                if let Some(&node) = self.indices.get(&entity) {
                    stats.local_reads += 1;
                    let local = *world.get::<Transform>(entity).expect("cached local");
                    if self.locals[node.index()] != local {
                        self.locals[node.index()] = local;
                        self.ticks.mark(node);
                        dirty = true;
                    }
                }
            }
            for entity in globals_changed {
                if let Some(&node) = self.indices.get(&entity) {
                    if world.get::<GlobalTransform>(entity) != Some(&self.globals[node.index()]) {
                        self.ticks.mark(node);
                        dirty = true;
                    }
                }
            }
        }
        if let Some(entities) = new_entities {
            self.entities = entities;
        }
        if let Some(indices) = new_indices {
            self.indices = indices;
        }
        if let Some(parents) = new_parents {
            self.parents = parents;
        }
        if let Some(hierarchy) = hierarchy {
            self.hierarchy = hierarchy;
        }
        if !dirty && removed.is_empty() {
            world.increment_change_tick();
            return Ok(stats);
        }
        let mut publish = if dirty {
            stats.recomputed = self
                .propagator
                .propagate(
                    &self.hierarchy,
                    &self.ticks,
                    &self.locals,
                    &mut self.globals,
                )
                .map_err(TransformEcsError::Hierarchy)?
                .recomputed;
            self.propagator.recomputed_nodes().to_vec()
        } else {
            Vec::new()
        };
        publish.sort_unstable_by_key(|node| node.index());
        // Every publication must be newer than consumers' last flush, including
        // two manual propagation calls with no intervening scheduled system.
        world.increment_change_tick();
        for entity in removed {
            if world.has::<GlobalTransform>(entity) {
                world.remove::<GlobalTransform>(entity);
                stats.removed += 1;
            }
            if world.id() != self.world_id || world.archetype_generation() != generation {
                return Ok(stats);
            }
        }
        for node in publish {
            let i = node.index();
            let entity = self.entities[i];
            if world.get::<GlobalTransform>(entity) != Some(&self.globals[i]) {
                // Equality is checked before acquiring Mut, so unchanged caches
                // never acquire a changed tick or notify downstream observers.
                if let Some(mut global) = world.get_mut::<GlobalTransform>(entity) {
                    *global = self.globals[i];
                } else {
                    world.insert(entity, self.globals[i]);
                }
                stats.written += 1;
            }
            if world.id() != self.world_id || world.archetype_generation() != generation {
                return Ok(stats);
            }
        }
        // Publication lifecycle callbacks can write locals at the publication
        // tick. Keep that tick newer than last_run, and arm subsequent callers.
        world.increment_change_tick();
        Ok(stats)
    }
}

/// Invalid ECS hierarchy input; no globals are published on error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransformEcsError {
    /// The transform-bearing subgraph is cyclic or otherwise invalid.
    Hierarchy(HierarchyError),
    /// A malformed/non-exclusive `ChildOf` relation has several targets.
    MultipleParents(Entity),
}

/// Actual work performed by one ECS propagation step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EcsPropagationStats {
    /// Full membership queries (zero unless lifecycle or restore invalidated it).
    pub membership_scans: usize,
    /// Entities whose ChildOf targets were inspected after topology invalidation.
    /// Excludes the relation-index scan used to detect changed ChildOf edges.
    pub relation_checks: usize,
    /// Authoritative local values read from ECS after chunk/row tick filtering.
    pub local_reads: usize,
    /// World matrices composed, including descendants of dirty locals.
    pub recomputed: usize,
    /// Globals added or changed; these produce ECS changed notifications.
    pub written: usize,
    /// Globals removed after their authoritative local disappeared.
    pub removed: usize,
}

/// Reconcile the ECS hierarchy and propagate only dirty transform subtrees.
///
/// Installs the integration on first use. Existing entities and globals are
/// supported; a removed global is recreated while its local exists. Removing a
/// previously managed local removes its global on this step. A standalone
/// global without a local is untouched and can still drive spatial observers.
/// Flush the ECS change observers afterwards, or let `Schedule::run` do so.
///
/// # Errors
/// Returns invalid hierarchy input without publishing global changes. Fix the
/// relations and retry; the previous valid propagation baseline is retained.
pub fn propagate_transforms(world: &mut World) -> Result<EcsPropagationStats, TransformEcsError> {
    super::install(world);
    let mut state = world
        .remove_resource::<PropagationState>()
        .expect("installed");
    let result = state.propagate(world);
    if state.world_id == world.id() && world.contains_resource::<super::Registration>() {
        world.insert_resource(state);
    }
    result
}

/// Exclusive schedule system; order after all local/hierarchy writers.
///
/// # Panics
/// Panics on invalid hierarchy input. Use [`propagate_transforms`] to handle
/// malformed input explicitly instead.
pub fn transform_system(world: &mut World) {
    propagate_transforms(world).expect("invalid transform hierarchy");
}
