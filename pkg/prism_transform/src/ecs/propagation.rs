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
            // Never apply abandoned-timeline removals/publications to a snapshot.
            self.pending_removals.clear();
            self.pending_publications.clear();
        }
        let rebuild = restored || self.needs_rebuild;
        let mut stats = EcsPropagationStats::default();
        let signals = world.resource_mut::<super::PropagationSignals>();
        signals.rebase(generation);
        let members = signals.membership.clone();
        let global_signals = signals.globals.clone();
        let index = world.relations().index();
        let cursor = index.change_cursor();
        let child_of = world
            .components()
            .id_of::<ChildOf>()
            .expect("installed ChildOf");
        let delta = if rebuild {
            None
        } else {
            self.relation_cursor
                .and_then(|last| index.changed_sources_since(child_of, last))
        };
        let fallback = delta.is_none();
        stats.relation_fallbacks = usize::from(fallback && !rebuild);

        // Only initial discovery/restore/panic recovery queries full membership.
        let baseline = if rebuild {
            stats.membership_scans = 1;
            let current = self.query.iter(world).collect::<BTreeSet<_>>();
            stats.membership_baseline_entities = current.len();
            Some(current)
        } else {
            None
        };
        let mut added = BTreeSet::new();
        let mut removed = BTreeSet::new();
        if let Some(current) = &baseline {
            added.extend(current.iter().copied());
            removed.extend(self.nodes.keys().filter(|e| !current.contains(e)).copied());
        } else {
            for &entity in &members {
                stats.membership_entities += 1;
                match (
                    world.has::<Transform>(entity),
                    self.nodes.contains_key(&entity),
                ) {
                    (true, false) => {
                        added.insert(entity);
                    }
                    (false, true) => {
                        removed.insert(entity);
                    }
                    _ => {}
                }
            }
        }
        // A restored member can lose its local before this cache has adopted
        // it. Only current-generation lifecycle events establish ownership;
        // neither old cache nodes nor old-timeline events may delete standalone
        // globals from a snapshot. Handle add/remove within one pass too.
        let lifecycle_removals: BTreeSet<_> = members
            .iter()
            .copied()
            .filter(|entity| !world.has::<Transform>(*entity))
            .collect();
        let is_member = |entity: &Entity| {
            if let Some(current) = &baseline {
                current.contains(entity)
            } else {
                added.contains(entity)
                    || (self.nodes.contains_key(entity) && !removed.contains(entity))
            }
        };
        let mut sources = BTreeSet::new();
        if rebuild || fallback {
            sources.extend(self.nodes.keys().copied().filter(&is_member));
            sources.extend(added.iter().copied());
        } else if let Some(delta) = delta {
            stats.relation_journal_entries = delta.entries_examined;
            stats.relation_journal_lookups = delta.lookup_steps;
            sources.extend(delta.sources);
        }
        if !rebuild {
            for &entity in &members {
                sources.insert(entity);
                // Membership of the parent changes effective edges without
                // changing ChildOf. Read both the live reverse index and old
                // children (despawn may already have removed the live edges).
                let dependents = world.relation_sources::<ChildOf>(entity);
                stats.relation_dependents_examined += dependents.len();
                sources.extend(dependents.iter().copied());
                if let Some(node) = self.nodes.get(&entity) {
                    stats.relation_dependents_examined += node.children.len();
                    sources.extend(node.children.iter().copied());
                }
            }
        }
        stats.relation_source_candidates = sources.len();
        let mut parents = BTreeMap::new();
        for entity in sources {
            if !is_member(&entity) {
                continue;
            }
            stats.relation_checks += 1;
            let targets = world.relation_targets::<ChildOf>(entity);
            if targets.len() > 1 {
                return Err(TransformEcsError::MultipleParents(entity));
            }
            let parent = targets.first().copied().filter(&is_member);
            if rebuild || added.contains(&entity) || self.nodes[&entity].parent != parent {
                parents.insert(entity, parent);
            }
        }
        stats.relation_scans = usize::from(rebuild || fallback || stats.relation_checks != 0);

        // Validate the FINAL batch overlay, not sequential reparent operations
        // (a valid edge reversal can transiently look cyclic). Existing cached
        // edges are already proven acyclic; only changed chains need walking.
        let mut validated = BTreeSet::new();
        for &entity in parents.keys() {
            let mut path = BTreeSet::new();
            let mut next = Some(entity);
            while let Some(node) = next {
                if validated.contains(&node) {
                    break;
                }
                stats.topology_checks += 1;
                if !path.insert(node) {
                    return Err(TransformEcsError::Hierarchy(HierarchyError::Cycle));
                }
                next = parents
                    .get(&node)
                    .copied()
                    .unwrap_or_else(|| self.nodes.get(&node).and_then(|n| n.parent));
            }
            validated.extend(path);
        }

        let mut locals_changed = changed_entities(
            &self.transform_changes,
            world,
            self.last_run,
            now,
            &mut stats,
        );
        locals_changed.extend(members.iter().copied());
        locals_changed.extend(added.iter().copied());
        let mut globals_changed =
            changed_entities(&self.globals_query, world, self.last_run, now, &mut stats);
        globals_changed.extend(global_signals);

        // No fallible hierarchy operation or user callback before this commit.
        let signals = world.resource_mut::<super::PropagationSignals>();
        signals.membership.clear();
        signals.globals.clear();
        self.last_run = now;
        self.relation_cursor = Some(cursor);
        self.generation = generation;
        self.needs_rebuild = false;
        self.pending_removals.extend(lifecycle_removals);
        let mut dirty = BTreeSet::new();
        if rebuild {
            if !restored {
                self.pending_removals.extend(removed.iter().copied());
            }
            self.nodes.clear();
        } else {
            // Detach the complete batch before attaching new edges.
            for &entity in parents.keys().chain(removed.iter()) {
                if let Some(parent) = self.nodes.get(&entity).and_then(|n| n.parent)
                    && let Some(node) = self.nodes.get_mut(&parent)
                {
                    node.children.remove(&entity);
                }
            }
            for &entity in &removed {
                self.nodes.remove(&entity);
                self.pending_publications.remove(&entity);
                self.pending_removals.insert(entity);
            }
        }
        for &entity in &added {
            stats.local_reads += 1;
            self.nodes.insert(
                entity,
                CachedNode {
                    parent: None,
                    children: BTreeSet::new(),
                    local: *world.get::<Transform>(entity).expect("current member"),
                    global: GlobalTransform::IDENTITY,
                },
            );
            self.pending_removals.remove(&entity);
            dirty.insert(entity);
        }
        for (&entity, &parent) in &parents {
            self.nodes.get_mut(&entity).expect("current child").parent = parent;
            if let Some(parent) = parent {
                self.nodes
                    .get_mut(&parent)
                    .expect("current parent")
                    .children
                    .insert(entity);
            }
            dirty.insert(entity);
        }
        for entity in locals_changed {
            if added.contains(&entity) {
                continue;
            }
            if let Some(node) = self.nodes.get_mut(&entity) {
                stats.local_reads += 1;
                let local = *world.get::<Transform>(entity).expect("current local");
                if node.local != local {
                    node.local = local;
                    dirty.insert(entity);
                }
            }
        }
        for entity in globals_changed {
            if let Some(node) = self.nodes.get(&entity)
                && world.get::<GlobalTransform>(entity) != Some(&node.global)
            {
                dirty.insert(entity);
            }
        }

        // No full forest scan/validation or dense reindexing on local edits.
        // Preparation depends on the dirty set and its ancestor paths.
        stats.dirty_nodes_examined = dirty.len();
        let mut roots = Vec::new();
        for &entity in &dirty {
            let mut parent = self.nodes[&entity].parent;
            let mut covered = false;
            while let Some(ancestor) = parent {
                stats.ancestor_checks += 1;
                if dirty.contains(&ancestor) {
                    covered = true;
                    break;
                }
                parent = self.nodes[&ancestor].parent;
            }
            if !covered {
                roots.push(entity);
            }
        }
        let mut stack = Vec::new();
        for root in roots {
            stack.push(root);
            while let Some(entity) = stack.pop() {
                let node = &self.nodes[&entity];
                let global = match node.parent {
                    Some(parent) => self.nodes[&parent].global.mul_transform(&node.local),
                    None => GlobalTransform::from_transform(&node.local),
                };
                let node = self.nodes.get_mut(&entity).expect("dirty subtree");
                node.global = global;
                stack.extend(node.children.iter().rev().copied());
                self.pending_publications.insert(entity);
                stats.recomputed += 1;
            }
        }

        // Advancing before and after callbacks preserves the observer window.
        world.increment_change_tick();
        while let Some(&entity) = self.pending_removals.first() {
            stats.publications_examined += 1;
            if !world.has::<Transform>(entity) && world.has::<GlobalTransform>(entity) {
                world.remove::<GlobalTransform>(entity);
                stats.removed += 1;
            }
            // Removal observers run before deletion and may unwind.
            self.pending_removals.remove(&entity);
            if self.interrupted(world, registration, generation, cursor) {
                return Ok(stats);
            }
        }
        while let Some(&entity) = self.pending_publications.first() {
            stats.publications_examined += 1;
            // A preceding callback may have removed/despawned another member.
            if world.has::<Transform>(entity) {
                let global = self.nodes[&entity].global;
                if world.get::<GlobalTransform>(entity) != Some(&global) {
                    if let Some(mut current) = world.get_mut::<GlobalTransform>(entity) {
                        *current = global;
                    } else {
                        world.insert(entity, global);
                    }
                    stats.written += 1;
                }
            }
            // Retain unfinished work across normal callback interruption/panic.
            self.pending_publications.remove(&entity);
            if self.interrupted(world, registration, generation, cursor) {
                return Ok(stats);
            }
        }
        world.increment_change_tick();
        Ok(stats)
    }

    fn interrupted(
        &self,
        world: &World,
        registration: ObserverId,
        generation: u64,
        cursor: RelationChangeCursor,
    ) -> bool {
        world.id() != self.world_id
            || world.archetype_generation() != generation
            || super::registration_id(world) != Some(registration)
            || world.relations().index().change_cursor() != cursor
            || !world
                .resource::<super::PropagationSignals>()
                .membership
                .is_empty()
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
    /// Full membership queries (initialization, restore or panic recovery only).
    pub membership_scans: usize,
    /// Entities returned by full membership discovery, zero on delta passes.
    pub membership_baseline_entities: usize,
    /// Specific lifecycle-signalled entities checked for membership.
    pub membership_entities: usize,
    /// Passes that inspect parent targets. Ordinary passes visit only sources
    /// in the ChildOf journal or affected by parent membership changes.
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
    /// Relevant source journal entries consumed (before deduplication).
    pub relation_journal_entries: usize,
    /// Binary-search comparisons used to locate relevant journal entries.
    pub relation_journal_lookups: usize,
    /// Unique journal/membership candidates, including non-transform sources.
    pub relation_source_candidates: usize,
    /// Live reverse-index and cached-child entries read for membership deltas.
    /// Entries seen by both indexes are counted twice, before deduplication.
    pub relation_dependents_examined: usize,
    /// Full parent baseline refreshes caused by a gap/identity mismatch.
    /// Initial discovery/restore/panic rebuilds are counted by membership_scans.
    pub relation_fallbacks: usize,
    /// Ancestor nodes visited when validating changed topology chains.
    pub topology_checks: usize,
    /// Dirty entries examined when preparing subtree roots.
    pub dirty_nodes_examined: usize,
    /// Ancestor links visited when suppressing redundant dirty roots.
    pub ancestor_checks: usize,
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
    /// Pending removal/publication items visited, including equality no-ops.
    pub publications_examined: usize,
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
