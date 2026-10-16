use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use prism_ecs::change::Tick;
use prism_ecs::component::Component;
use prism_ecs::entity::Entity;
use prism_ecs::hierarchy::ChildOf;
use prism_ecs::observer::ObserverId;
use prism_ecs::query::QueryState;
use prism_ecs::relation::RelationChangeCursor;
use prism_ecs::resource::Resource;
use prism_ecs::world::{World, WorldId};

use crate::hierarchy::HierarchyError;
use crate::{GlobalTransform, Transform};

type Inputs = QueryState<(Entity, &'static Transform)>;
type Globals = QueryState<(Entity, &'static GlobalTransform)>;
type Membership = QueryState<Entity, prism_ecs::query::With<Transform>>;

// The column-owned interval index skips cold chunk subtrees before examining
// row ticks. No component value is borrowed here. The row check remains
// necessary because a chunk summary is deliberately conservative.
fn changed_entities<T: Component>(
    query: &QueryState<(Entity, &'static T)>,
    world: &World,
    last: Tick,
    now: Tick,
    stats: &mut EcsPropagationStats,
) -> BTreeSet<Entity> {
    let mut changed = BTreeSet::new();
    let id = query
        .access()
        .reads()
        .first()
        .copied()
        .expect("transform change query reads its component");
    let report = query.dirty_chunks_with_stats(world, last, now);
    stats.change_chunk_summaries += report.stats.summaries_tested;
    stats.change_chunks += report.stats.reported_chunks;
    for dirty in report.chunks {
        let archetype = world
            .archetypes()
            .get(dirty.archetype)
            .expect("dirty chunk archetype disappeared during read");
        let table = archetype.table();
        let column = table
            .column(id)
            .expect("dirty query reported a chunk without its component");
        for row in dirty.start..dirty.end {
            stats.change_rows_scanned += 1;
            if column.changed_tick(row).is_newer_than(last, now) {
                changed.insert(table.entities()[row]);
            }
        }
    }
    changed
}

struct CachedNode {
    parent: Option<Entity>,
    children: BTreeSet<Entity>,
    local: Transform,
    global: GlobalTransform,
}

pub(super) struct PropagationState {
    query: Membership,
    transform_changes: Inputs,
    globals_query: Globals,
    world_id: WorldId,
    generation: u64,
    relation_cursor: Option<RelationChangeCursor>,
    nodes: BTreeMap<Entity, CachedNode>,
    last_run: Tick,
    needs_rebuild: bool,
    pending_removals: BTreeSet<Entity>,
    pending_publications: BTreeSet<Entity>,
}

impl Resource for PropagationState {}

impl PropagationState {
    pub(super) fn new(world: &mut World) -> Self {
        Self {
            query: world.query_filtered(),
            transform_changes: world.query(),
            globals_query: world.query(),
            world_id: world.id(),
            generation: world.archetype_generation(),
            relation_cursor: None,
            nodes: BTreeMap::new(),
            last_run: Tick::ZERO,
            needs_rebuild: true,
            pending_removals: BTreeSet::new(),
            pending_publications: BTreeSet::new(),
        }
    }

    fn propagate(
        &mut self,
        world: &mut World,
        registration: ObserverId,
    ) -> Result<EcsPropagationStats, TransformEcsError> {
        let now = world.change_tick();
        let generation = world.archetype_generation();
        let restored = self.generation != generation || self.world_id != world.id();
        if self.world_id != world.id() {
            *self = Self::new(world);
        }
        if restored {
            self.pending_removals.clear();
        }
        let rebuild = restored || self.needs_rebuild;
        let signals = world.resource::<super::PropagationSignals>();
        let membership_dirty = signals.membership_dirty || rebuild;
        let mut stats = EcsPropagationStats::default();
        let relation_version = world.relations().index().version();
        let relation_dirty = self.relation_version != Some(relation_version);
        // The relation index exposes a version but not a mutation journal.
        // Refresh the parent cache through the authoritative per-source query
        // on invalidation. This is deterministic and avoids traversing and
        // sorting unrelated relation kinds. The committed cache is replaced
        // only after hierarchy validation succeeds, so errors remain retryable.
        if relation_dirty {
            stats.relation_scans = 1;
        }
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
        let new_parents = if membership_dirty || relation_dirty {
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
        let locals_changed = changed_entities(
            &self.transform_changes,
            world,
            self.last_run,
            now,
            &mut stats,
        );
        let mut globals_changed =
            changed_entities(&self.globals_query, world, self.last_run, now, &mut stats);
        globals_changed.extend(signals.globals.iter().copied());
        let signals = world.resource_mut::<super::PropagationSignals>();
        signals.membership_dirty = false;
        signals.globals.clear();
        self.last_run = now;
        self.relation_version = Some(relation_version);
        self.generation = generation;
        self.needs_rebuild = false;
        let mut removed = Vec::new();
        let mut dirty = false;
        if topology_changed || rebuild {
            let mut ticks = ChangeTicks::new();
            let mut locals = Vec::with_capacity(entities.len());
            let mut globals = Vec::with_capacity(entities.len());
            let mut marked = Vec::new();
            for (i, &entity) in entities.iter().enumerate() {
                ticks.push();
                let old = self.indices.get(&entity).map(|node| node.index());
                let local = if rebuild || old.is_none() || locals_changed.contains(&entity) {
                    stats.local_reads += 1;
                    *world.get::<Transform>(entity).expect("queried local")
                } else {
                    match old {
                        Some(index) => self.locals[index],
                        None => unreachable!("cached transform index missing"),
                    }
                };
                let cached = old.map(|j| self.globals[j]);
                let current = world.get::<GlobalTransform>(entity).copied();
                let old_parent = old
                    .and_then(|j| self.parents[j])
                    .map(|parent| self.entities[parent.index()]);
                let new_parent = parents[i].map(|parent| entities[parent.index()]);
                if rebuild
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
                if let Some(&node) = self.indices.get(&entity)
                    && world.get::<GlobalTransform>(entity) != Some(&self.globals[node.index()])
                {
                    self.ticks.mark(node);
                    dirty = true;
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
        self.pending_removals.extend(removed);
        if !dirty && self.pending_removals.is_empty() {
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
        while let Some(&entity) = self.pending_removals.first() {
            if world.has::<GlobalTransform>(entity) {
                world.remove::<GlobalTransform>(entity);
                stats.removed += 1;
            }
            // Remove observers run before deletion and may unwind. Keep this
            // entity queued until the removal returns so retry cannot miss it.
            self.pending_removals.remove(&entity);
            if world.id() != self.world_id
                || world.archetype_generation() != generation
                || super::registration_id(world) != Some(registration)
            {
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
            if world.id() != self.world_id
                || world.archetype_generation() != generation
                || super::registration_id(world) != Some(registration)
            {
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
    /// Relation-cache refreshes after the global relation index version changes.
    /// Each refresh checks the current transform sources (`O(V)`); stable
    /// frames are zero. The cache never traverses unrelated relation edges.
    pub relation_scans: usize,
    /// Global relation-index edges visited by propagation. Kept as an explicit
    /// zero-valued guard so a regression to an all-edge walk is observable.
    pub relation_edges_scanned: usize,
    /// Global canonical edge sorts performed by propagation. The source cache
    /// is already deterministic and therefore requires none.
    pub relation_sorts: usize,
    /// Entities whose `ChildOf` targets were inspected after topology invalidation.
    /// Separate from relation-index traversals, counted by `relation_scans`.
    pub relation_checks: usize,
    /// Hierarchical dirty-chunk summary nodes tested by transform change scans.
    pub change_chunk_summaries: usize,
    /// Conservative dirty chunks returned by transform change scans.
    pub change_chunks: usize,
    /// Rows inspected after dirty chunks were reported.
    pub change_rows_scanned: usize,
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
    let registration = super::registration_id(world).expect("installed subscribers");
    let state = world
        .remove_resource::<PropagationState>()
        .expect("installed");
    let mut guard = PropagationGuard {
        world,
        state: Some(state),
        registration,
        completed: false,
    };
    let result = guard
        .state
        .as_mut()
        .expect("owned cache")
        .propagate(guard.world, registration);
    guard.completed = true;
    result
}

// Lifecycle observers execute during publication. Preserve both the cache and
// outstanding removals if one unwinds, and re-read authoritative values on retry.
struct PropagationGuard<'w> {
    world: &'w mut World,
    state: Option<PropagationState>,
    registration: ObserverId,
    completed: bool,
}

impl Drop for PropagationGuard<'_> {
    fn drop(&mut self) {
        let Some(mut state) = self.state.take() else {
            return;
        };
        if state.world_id == self.world.id()
            && super::registration_id(self.world) == Some(self.registration)
        {
            state.needs_rebuild |= !self.completed;
            self.world.insert_resource(state);
        }
    }
}

/// Exclusive schedule system; order after all local/hierarchy writers.
///
/// # Panics
/// Panics on invalid hierarchy input. Use [`propagate_transforms`] to handle
/// malformed input explicitly instead.
pub fn transform_system(world: &mut World) {
    propagate_transforms(world).expect("invalid transform hierarchy");
}
