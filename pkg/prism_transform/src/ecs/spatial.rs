use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use prism_ecs::component::Component;
use prism_ecs::entity::Entity;
use prism_ecs::resource::Resource;
use prism_ecs::world::{World, WorldId};
use prism_math::Aabb3;

use crate::hierarchy::NodeId;
use crate::spatial_sync::{SpatialCommand, SpatialSync};
use crate::GlobalTransform;

/// Object-space bounds of an opt-in spatial proxy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalBounds(pub Aabb3);
impl Component for LocalBounds {}

/// An incremental spatial edit carrying a full generational ECS identity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpatialUpdate {
    /// Insert a new proxy with these world bounds.
    Insert {
        /// Entity owning the proxy.
        entity: Entity,
        /// World bounds to insert.
        bounds: Aabb3,
    },
    /// Refit an existing proxy.
    Update {
        /// Entity owning the proxy.
        entity: Entity,
        /// Previously stored bounds.
        old: Aabb3,
        /// New world bounds.
        new: Aabb3,
    },
    /// Remove the proxy with its last stored bounds.
    Remove {
        /// Entity owning the proxy, possibly already despawned.
        entity: Entity,
        /// Previously stored bounds.
        old: Aabb3,
    },
}

impl SpatialUpdate {
    /// The generational owner of this edit.
    pub fn entity(&self) -> Entity {
        match *self {
            Self::Insert { entity, .. }
            | Self::Update { entity, .. }
            | Self::Remove { entity, .. } => entity,
        }
    }
}

pub(super) struct SpatialState {
    world_id: WorldId,
    generation: u64,
    pub(super) pending: BTreeSet<Entity>,
    sync: SpatialSync,
    nodes: BTreeMap<Entity, NodeId>,
    owners: Vec<Option<Entity>>,
    free: Vec<NodeId>,
}
impl Resource for SpatialState {}

impl SpatialState {
    pub(super) fn new(world: &mut World) -> Self {
        let query = world.query::<(Entity, &GlobalTransform, &LocalBounds)>();
        Self {
            world_id: world.id(),
            generation: world.archetype_generation(),
            pending: query.iter(world).map(|(entity, ..)| entity).collect(),
            sync: SpatialSync::new(),
            nodes: BTreeMap::new(),
            owners: Vec::new(),
            free: Vec::new(),
        }
    }

    fn reconcile_restore(&mut self, world: &mut World) {
        if self.world_id == world.id() && self.generation == world.archetype_generation() {
            return;
        }
        // Restore bypasses lifecycle/changed observers and can rewind ticks or
        // resurrect an entity with the same full handle. Keep the drained
        // baseline to emit minimal corrections, but discard pre-restore signals.
        self.pending.clear();
        self.pending.extend(self.nodes.keys().copied());
        let query = world.query::<(Entity, &GlobalTransform, &LocalBounds)>();
        self.pending
            .extend(query.iter(world).map(|(entity, ..)| entity));
        self.world_id = world.id();
        self.generation = world.archetype_generation();
    }

    pub(super) fn removals(&self) -> Vec<SpatialUpdate> {
        self.nodes
            .iter()
            .filter_map(|(&entity, &node)| {
                self.sync
                    .stored_bounds(node)
                    .map(|old| SpatialUpdate::Remove { entity, old })
            })
            .collect()
    }
}

/// Snapshot the last drained spatial baseline, in generational entity order.
///
/// This reads only cached bounds and neither flushes nor consumes pending edits.
/// A late-joining index can seed from this baseline, then consume future drains.
/// An uninstalled integration has an empty baseline.
pub fn spatial_snapshot(world: &World) -> Vec<(Entity, Aabb3)> {
    world
        .get_resource::<SpatialState>()
        .map_or_else(Vec::new, |state| {
            state
                .nodes
                .iter()
                .filter_map(|(&entity, &node)| {
                    state
                        .sync
                        .stored_bounds(node)
                        .map(|bounds| (entity, bounds))
                })
                .collect()
        })
}

/// Set the spatial refit margin while no proxies are tracked.
///
/// # Panics
/// Panics if proxies are still tracked. Reconfiguration after all proxies have
/// been removed and drained is allowed, as is configuration before first use.
pub fn set_spatial_margin(world: &mut World, margin: f32) {
    super::install(world);
    let state = world.resource_mut::<SpatialState>();
    assert!(
        state.nodes.is_empty(),
        "configure margin while no proxies are tracked"
    );
    state.sync = SpatialSync::with_margin(margin);
}

/// Drain spatial edits queued by the ECS observer bus, ordered by entity index
/// and then generation. Repeated signals coalesce to one final proxy edit.
///
/// Call after `world.flush_changed_observers()` or a schedule run. This does
/// not flush observers itself. Already-signalled entities use their final world
/// values at drain time, including subsequent callback writes; previously
/// unsignalled in-place writes require a bus flush. Bounds-only edits, removals, despawns and
/// recycled entity indices all flow through the same pending set. No pending
/// signals means no world-bound calculations. The commands use the standalone
/// [`SpatialSync`] exact/fat refit policy; consume them in a BVH/grid backend.
/// After snapshot restore a single reconciliation includes the old proxy owners
/// and all current global/bounds pairs, even when restored ticks are unchanged.
pub fn drain_spatial_updates(world: &mut World) -> Vec<SpatialUpdate> {
    super::install(world);
    let mut state = world.remove_resource::<SpatialState>().expect("installed");
    state.reconcile_restore(world);
    for entity in core::mem::take(&mut state.pending) {
        match (
            world.get::<GlobalTransform>(entity),
            world.get::<LocalBounds>(entity),
        ) {
            (Some(global), Some(bounds)) => {
                let node = *state.nodes.entry(entity).or_insert_with(|| {
                    let node = state.free.pop().unwrap_or_else(|| {
                        let node = NodeId::new(state.owners.len() as u32);
                        state.owners.push(None);
                        node
                    });
                    state.owners[node.index()] = Some(entity);
                    node
                });
                state.sync.register(node, bounds.0);
                state.sync.observe_one(node, global);
            }
            _ => {
                if let Some(&node) = state.nodes.get(&entity) {
                    state.sync.remove(node);
                }
            }
        }
    }
    let (commands, _) = state.sync.drain();
    let mut updates = Vec::with_capacity(commands.len());
    for command in commands {
        let node = command.node();
        let entity = state.owners[node.index()].expect("live proxy owner");
        updates.push(match command {
            SpatialCommand::Insert { bounds, .. } => SpatialUpdate::Insert { entity, bounds },
            SpatialCommand::Update { old, new, .. } => SpatialUpdate::Update { entity, old, new },
            SpatialCommand::Remove { old, .. } => {
                state.nodes.remove(&entity);
                state.owners[node.index()] = None;
                state.free.push(node);
                SpatialUpdate::Remove { entity, old }
            }
        });
    }
    updates.sort_unstable_by_key(SpatialUpdate::entity);
    world.insert_resource(state);
    // Keep subsequent manual in-place writes newer than the consumed window.
    world.increment_change_tick();
    updates
}
