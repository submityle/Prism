//! Project-configured ECS to authority-AOI projection.

use alloc::collections::{BTreeMap, BTreeSet, VecDeque};
use alloc::vec::Vec;
use core::marker::PhantomData;

use prism_app::{FixedPostUpdate, Plugin};
use prism_ecs::change_observers::ChangeContext;
use prism_ecs::component::Component;
use prism_ecs::entity::Entity;
use prism_ecs::observer::{LifecycleEvent, ObserverContext};
use prism_ecs::resource::Resource;
use prism_ecs::world::World;

use super::{AuthorityEntityProjection, MultiPeerAuthority, ServerStrategy};

/// Project-owned identity, spatial-coordinate, and serialization policy.
pub trait EcsAoiProjectionSource: Send + Sync + 'static {
    /// Stable network identity for a live ECS entity.
    fn stable_id(&self, world: &World, entity: Entity) -> Option<u32>;
    /// Two-dimensional AOI position chosen by the application.
    fn position(&self, world: &World, entity: Entity) -> Option<(f32, f32)>;
    /// Complete canonical snapshot for a newly visible or updated entity.
    fn snapshot(&self, world: &World, entity: Entity) -> Result<Vec<u8>, EcsAoiSourceError>;
}

/// Failure to serialize project-owned replicated state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EcsAoiSourceError {
    /// The entity is temporarily not serializable; its dirty mark is retained.
    TemporarilyUnavailable,
    /// Project serialization failed; its dirty mark is retained for repair/retry.
    SerializationFailed,
}

/// Queue and fixed-tick processing bounds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EcsAoiProjectionConfig {
    /// Maximum distinct entity handles retained for projection.
    pub max_pending_entities: usize,
    /// Maximum handles sampled per fixed tick.
    pub max_entities_per_tick: usize,
}

impl EcsAoiProjectionConfig {
    /// Construct valid non-zero bounds.
    pub const fn new(max_pending_entities: usize, max_entities_per_tick: usize) -> Option<Self> {
        if max_pending_entities == 0 || max_entities_per_tick == 0 {
            None
        } else {
            Some(Self {
                max_pending_entities,
                max_entities_per_tick,
            })
        }
    }
}

/// Observable projection queue health.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EcsAoiProjectionStatus {
    /// Distinct entity handles still waiting for commit.
    pub pending_entities: usize,
    /// Successful projection batches.
    pub committed_batches: u64,
    /// Successfully committed entity operations.
    pub committed_entities: u64,
    /// Failed enqueue, sampling, or authority commits.
    pub failures: u64,
    /// Queue overflow occurred. The adapter halts rather than silently dropping changes.
    pub overflowed: bool,
}

struct DirtyQueue {
    items: VecDeque<Entity>,
    membership: BTreeSet<Entity>,
    capacity: usize,
    overflowed: bool,
}
impl Resource for DirtyQueue {}
impl Resource for EcsAoiProjectionStatus {}

/// Mark an entity whose identity, AOI position, or replicated state changed.
/// Call from the authority command handler or the system that owns the change.
/// Duplicate marks coalesce. Overflow is sticky and halts projection so the
/// application can rebuild/reseed explicitly instead of losing updates.
pub fn mark_aoi_entity_dirty(world: &mut World, entity: Entity) -> bool {
    let Some(queue) = world.get_resource_mut::<DirtyQueue>() else {
        return false;
    };
    if queue.membership.contains(&entity) {
        return true;
    }
    if queue.items.len() >= queue.capacity {
        if !queue.overflowed {
            queue.overflowed = true;
            if let Some(status) = world.get_resource_mut::<EcsAoiProjectionStatus>() {
                status.overflowed = true;
                status.failures = status.failures.saturating_add(1);
            }
        }
        return false;
    }
    queue.membership.insert(entity);
    queue.items.push_back(entity);
    true
}

/// Rebuild the dirty queue from an authoritative live-entity snapshot after
/// overflow or whenever the host requests a full reconciliation. The caller
/// must pass every live entity eligible for this adapter; the projection stage
/// samples those entities and also visits every previously published entity so
/// stale network ids become removals. If the unique union exceeds the configured
/// queue capacity, nothing changes and the queue remains halted.
pub fn rebuild_aoi_projection_queue<S, A>(world: &mut World, live_entities: &[Entity]) -> bool
where
    S: EcsAoiProjectionSource,
    A: ServerStrategy,
{
    let Some(mut queue) = world.remove_resource::<DirtyQueue>() else {
        return false;
    };
    let Some(state) = world.get_resource::<ProjectionState<S, A>>() else {
        world.insert_resource(queue);
        return false;
    };
    let mut entities = BTreeSet::new();
    entities.extend(
        live_entities
            .iter()
            .copied()
            .filter(|entity| world.contains(*entity)),
    );
    entities.extend(state.published.keys().copied());
    if entities.len() > queue.capacity {
        world.insert_resource(queue);
        return false;
    }
    queue.items = entities.iter().copied().collect();
    queue.membership = entities;
    queue.overflowed = false;
    if let Some(status) = world.get_resource_mut::<EcsAoiProjectionStatus>() {
        status.overflowed = false;
        status.pending_entities = queue.items.len();
    }
    world.insert_resource(queue);
    true
}

fn mark_lifecycle_dirty(context: &mut ObserverContext<'_>) {
    let _ = mark_aoi_entity_dirty(context.world, context.entity);
}

type RegisterTracker = fn(&mut World);

fn register_tracker<T: Component>(world: &mut World) {
    world.observe::<T, _>(LifecycleEvent::Remove, mark_lifecycle_dirty);
    world.observe_changed::<T, _>(|context: &mut ChangeContext<'_>| {
        let changed = context.changed().to_vec();
        for entity in changed {
            let _ = mark_aoi_entity_dirty(context.world, entity);
        }
    });
}

/// Optional ECS-to-AOI bridge. All project semantics are injected through S.
pub struct EcsAoiProjectionPlugin<S, A>
where
    S: EcsAoiProjectionSource,
    A: ServerStrategy,
{
    source: std::sync::Mutex<Option<S>>,
    trackers: Vec<RegisterTracker>,
    config: EcsAoiProjectionConfig,
    marker: PhantomData<fn() -> A>,
}

impl<S, A> EcsAoiProjectionPlugin<S, A>
where
    S: EcsAoiProjectionSource,
    A: ServerStrategy,
{
    /// Configure project-owned stable identity, AOI position, and snapshot semantics.
    #[must_use]
    pub fn new(source: S, config: EcsAoiProjectionConfig) -> Self {
        Self {
            source: std::sync::Mutex::new(Some(source)),
            trackers: Vec::new(),
            config,
            marker: PhantomData,
        }
    }

    /// Observe insert, overwrite and removal of a project-selected component.
    #[must_use]
    pub fn track_component<T: Component>(mut self) -> Self {
        self.trackers.push(register_tracker::<T>);
        self
    }
}

#[derive(Clone, Copy)]
struct Published {
    network_id: u32,
    position: (f32, f32),
}

struct ProjectionState<S, A> {
    source: S,
    published: BTreeMap<Entity, Published>,
    by_network_id: BTreeMap<u32, Entity>,
    max_entities_per_tick: usize,
    marker: PhantomData<fn() -> A>,
}
impl<S, A> Resource for ProjectionState<S, A>
where
    S: EcsAoiProjectionSource,
    A: ServerStrategy,
{
}

impl<S, A> Plugin for EcsAoiProjectionPlugin<S, A>
where
    S: EcsAoiProjectionSource,
    A: ServerStrategy,
{
    fn build(&self, app: &mut prism_app::App) {
        assert!(
            self.config.max_pending_entities > 0 && self.config.max_entities_per_tick > 0,
            "EcsAoiProjectionConfig requires non-zero queue and tick bounds"
        );
        let source = self
            .source
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            .expect("EcsAoiProjectionPlugin::build ran twice");
        for register in &self.trackers {
            register(app.world_mut());
        }
        app.insert_resource(DirtyQueue {
            items: VecDeque::new(),
            membership: BTreeSet::new(),
            capacity: self.config.max_pending_entities,
            overflowed: false,
        });
        app.insert_resource(EcsAoiProjectionStatus::default());
        app.insert_resource(ProjectionState::<S, A> {
            source,
            published: BTreeMap::new(),
            by_network_id: BTreeMap::new(),
            max_entities_per_tick: self.config.max_entities_per_tick,
            marker: PhantomData,
        });
        app.add_systems(FixedPostUpdate, project_dirty::<S, A>);
    }
}

fn project_dirty<S, A>(world: &mut World)
where
    S: EcsAoiProjectionSource,
    A: ServerStrategy,
{
    // Changed<T> observers enqueue dirty entities. Flush while the queue is
    // still installed as a resource; flushing after taking it would silently
    // reject every observer mark from this projection pass.
    world.flush_changed_observers();
    let Some(mut queue) = world.remove_resource::<DirtyQueue>() else {
        return;
    };
    let Some(mut state) = world.remove_resource::<ProjectionState<S, A>>() else {
        world.insert_resource(queue);
        return;
    };
    let Some(mut authority) = world.remove_resource::<MultiPeerAuthority<A>>() else {
        world.insert_resource(state);
        world.insert_resource(queue);
        return;
    };
    let mut status = world
        .remove_resource::<EcsAoiProjectionStatus>()
        .unwrap_or_default();
    if queue.overflowed {
        status.overflowed = true;
        status.pending_entities = queue.items.len();
        world.insert_resource(authority);
        world.insert_resource(state);
        world.insert_resource(queue);
        world.insert_resource(status);
        return;
    }

    let count = queue.items.len().min(state.max_entities_per_tick());
    let entities: Vec<_> = queue.items.iter().take(count).copied().collect();
    let mut operations = Vec::with_capacity(count);
    let mut removals = Vec::with_capacity(count);
    let mut updates = Vec::with_capacity(count);
    // Overlay only the identities touched by this bounded batch instead of
    // cloning the full published-id index on every fixed tick.
    let mut staged_ids = BTreeMap::<u32, Option<Entity>>::new();
    // Release every prior id owned by a dirty entity before claiming new ids.
    // This makes swaps/rebindings independent of FIFO order within the batch.
    for entity in &entities {
        if let Some(previous) = state.published.get(entity) {
            staged_ids.insert(previous.network_id, None);
        }
    }
    let mut failed = false;
    for entity in &entities {
        let previous = state.published.get(entity).copied();
        if !world.contains(*entity) {
            if let Some(previous) = previous {
                staged_ids.insert(previous.network_id, None);
                removals.push(AuthorityEntityProjection::Remove {
                    entity_id: previous.network_id,
                });
            }
            updates.push((*entity, None));
            continue;
        }
        let identity = state.source.stable_id(world, *entity);
        let position = state.source.position(world, *entity);
        let (Some(network_id), Some(position)) = (identity, position) else {
            if let Some(previous) = previous {
                staged_ids.insert(previous.network_id, None);
                removals.push(AuthorityEntityProjection::Remove {
                    entity_id: previous.network_id,
                });
            }
            updates.push((*entity, None));
            continue;
        };
        if let Some(previous) = previous
            && previous.network_id != network_id
        {
            staged_ids.insert(previous.network_id, None);
            removals.push(AuthorityEntityProjection::Remove {
                entity_id: previous.network_id,
            });
        }
        let current_owner = staged_ids
            .get(&network_id)
            .copied()
            .unwrap_or_else(|| state.by_network_id.get(&network_id).copied());
        if current_owner.is_some_and(|owner| owner != *entity) {
            failed = true;
            break;
        }
        staged_ids.insert(network_id, Some(*entity));
        let snapshot = match state.source.snapshot(world, *entity) {
            Ok(snapshot) => snapshot,
            Err(_) => {
                failed = true;
                break;
            }
        };
        let operation = if previous
            .is_none_or(|old| old.network_id != network_id || old.position != position)
        {
            AuthorityEntityProjection::Upsert {
                entity_id: network_id,
                position,
                snapshot,
            }
        } else {
            AuthorityEntityProjection::Update {
                entity_id: network_id,
                snapshot,
            }
        };
        operations.push(operation);
        updates.push((
            *entity,
            Some(Published {
                network_id,
                position,
            }),
        ));
    }

    removals.append(&mut operations);
    let operations = removals;
    let operation_count = operations.len();
    if !failed && (operations.is_empty() || authority.project_entities_atomic(&operations).is_ok())
    {
        for (entity, published) in updates {
            if let Some(published) = published {
                state.published.insert(entity, published);
            } else {
                state.published.remove(&entity);
            }
        }
        for (network_id, owner) in staged_ids {
            if let Some(owner) = owner {
                state.by_network_id.insert(network_id, owner);
            } else {
                state.by_network_id.remove(&network_id);
            }
        }
        for _ in 0..entities.len() {
            if let Some(entity) = queue.items.pop_front() {
                queue.membership.remove(&entity);
            }
        }
        if !operations.is_empty() {
            status.committed_batches = status.committed_batches.saturating_add(1);
            status.committed_entities = status
                .committed_entities
                .saturating_add(operation_count as u64);
        }
    } else {
        status.failures = status.failures.saturating_add(1);
    }
    status.pending_entities = queue.items.len();
    status.overflowed = queue.overflowed;
    world.insert_resource(authority);
    world.insert_resource(state);
    world.insert_resource(queue);
    world.insert_resource(status);
}

impl<S, A> ProjectionState<S, A> {
    fn max_entities_per_tick(&self) -> usize {
        self.max_entities_per_tick
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::AuthorityReplicationProvider;
    use crate::plugin::AuthorityReplicationProviderError;
    use crate::plugin::PeerId;
    use prism_ecs::component::Component;

    #[derive(Clone)]
    struct ReplicatedMarker(u32, u32);
    impl Component for ReplicatedMarker {}

    struct Source;
    impl EcsAoiProjectionSource for Source {
        fn stable_id(&self, world: &World, entity: Entity) -> Option<u32> {
            world.get::<ReplicatedMarker>(entity).map(|marker| marker.0)
        }
        fn position(&self, world: &World, entity: Entity) -> Option<(f32, f32)> {
            world
                .get::<ReplicatedMarker>(entity)
                .map(|marker| (marker.0 as f32, 0.0))
        }
        fn snapshot(&self, world: &World, entity: Entity) -> Result<Vec<u8>, EcsAoiSourceError> {
            world
                .get::<ReplicatedMarker>(entity)
                .map(|marker| {
                    let mut snapshot = marker.0.to_le_bytes().to_vec();
                    snapshot.extend_from_slice(&marker.1.to_le_bytes());
                    snapshot
                })
                .ok_or(EcsAoiSourceError::TemporarilyUnavailable)
        }
    }

    struct Strategy;
    impl ServerStrategy for Strategy {
        type Command = ();
        fn decode(&self, _: &[u8]) -> Option<Self::Command> {
            Some(())
        }
        fn apply(&self, _: &mut crate::delta::Record, _: Self::Command) {}
    }

    struct Provider;
    impl AuthorityReplicationProvider for Provider {
        fn snapshot_entity(
            &self,
            _: PeerId,
            _: u32,
            _: u64,
        ) -> Result<Vec<u8>, AuthorityReplicationProviderError> {
            Err(AuthorityReplicationProviderError::EntityUnavailable)
        }
    }

    fn authority() -> MultiPeerAuthority<Strategy> {
        MultiPeerAuthority::new(super::super::MultiPeerAuthorityConfig::default(), Provider)
            .unwrap()
    }

    #[test]
    fn lifecycle_tracking_projects_and_removes_without_a_world_scan() {
        let mut app = prism_app::App::new();
        app.insert_resource(authority());
        app.add_plugins(
            EcsAoiProjectionPlugin::<Source, Strategy>::new(
                Source,
                EcsAoiProjectionConfig::new(8, 4).unwrap(),
            )
            .track_component::<ReplicatedMarker>(),
        );
        let entity = app.world_mut().spawn(ReplicatedMarker(77, 1));
        app.world_mut().flush_changed_observers();
        app.world_mut().run_schedule(FixedPostUpdate);
        assert_eq!(
            app.world()
                .resource::<MultiPeerAuthority<Strategy>>()
                .aoi()
                .entity_count(),
            1
        );
        assert_eq!(
            app.world()
                .resource::<EcsAoiProjectionStatus>()
                .committed_entities,
            1
        );
        assert_eq!(
            app.world()
                .resource::<EcsAoiProjectionStatus>()
                .pending_entities,
            0
        );
        app.world_mut()
            .get_mut::<ReplicatedMarker>(entity)
            .unwrap()
            .1 = 2;
        app.world_mut().run_schedule(FixedPostUpdate);
        assert_eq!(
            app.world()
                .resource::<MultiPeerAuthority<Strategy>>()
                .aoi()
                .entity_count(),
            1
        );
        assert_eq!(
            app.world()
                .resource::<EcsAoiProjectionStatus>()
                .committed_entities,
            2
        );
        assert!(app.world_mut().remove::<ReplicatedMarker>(entity));
        app.world_mut().run_schedule(FixedPostUpdate);
        assert_eq!(
            app.world()
                .resource::<MultiPeerAuthority<Strategy>>()
                .aoi()
                .entity_count(),
            0
        );
        assert_eq!(
            app.world()
                .resource::<EcsAoiProjectionStatus>()
                .committed_entities,
            3
        );
    }

    #[test]
    fn dirty_queue_coalesces_handles_and_halts_on_capacity_overflow() {
        let mut world = World::new();
        world.insert_resource(DirtyQueue {
            items: VecDeque::new(),
            membership: BTreeSet::new(),
            capacity: 1,
            overflowed: false,
        });
        world.insert_resource(EcsAoiProjectionStatus::default());
        let first = world.spawn(());
        let second = world.spawn(());
        assert!(mark_aoi_entity_dirty(&mut world, first));
        assert!(mark_aoi_entity_dirty(&mut world, first));
        assert!(!mark_aoi_entity_dirty(&mut world, second));
        assert_eq!(world.resource::<DirtyQueue>().items.len(), 1);
        assert!(world.resource::<DirtyQueue>().overflowed);
        assert!(world.resource::<EcsAoiProjectionStatus>().overflowed);
    }

    #[test]
    fn full_resync_rebuild_clears_overflow_only_after_capacity_check() {
        let mut app = prism_app::App::new();
        app.insert_resource(authority());
        app.add_plugins(EcsAoiProjectionPlugin::<Source, Strategy>::new(
            Source,
            EcsAoiProjectionConfig::new(1, 1).unwrap(),
        ));
        let first = app.world_mut().spawn(ReplicatedMarker(31, 0));
        let second = app.world_mut().spawn(ReplicatedMarker(32, 0));
        assert!(mark_aoi_entity_dirty(app.world_mut(), first));
        assert!(!mark_aoi_entity_dirty(app.world_mut(), second));
        assert!(!rebuild_aoi_projection_queue::<Source, Strategy>(
            app.world_mut(),
            &[first, second],
        ));
        assert!(app.world().resource::<EcsAoiProjectionStatus>().overflowed);

        app.world_mut().despawn(second);
        assert!(rebuild_aoi_projection_queue::<Source, Strategy>(
            app.world_mut(),
            &[first],
        ));
        assert!(!app.world().resource::<EcsAoiProjectionStatus>().overflowed);
        assert_eq!(
            app.world()
                .resource::<EcsAoiProjectionStatus>()
                .pending_entities,
            1
        );
        app.world_mut().run_schedule(FixedPostUpdate);
        assert_eq!(
            app.world()
                .resource::<MultiPeerAuthority<Strategy>>()
                .aoi()
                .entity_count(),
            1
        );
    }

    #[test]
    fn stable_identity_swap_in_one_batch_is_not_order_dependent() {
        let mut app = prism_app::App::new();
        app.insert_resource(authority());
        app.add_plugins(EcsAoiProjectionPlugin::<Source, Strategy>::new(
            Source,
            EcsAoiProjectionConfig::new(4, 4).unwrap(),
        ));
        let first = app.world_mut().spawn(ReplicatedMarker(10, 0));
        let second = app.world_mut().spawn(ReplicatedMarker(20, 0));
        assert!(mark_aoi_entity_dirty(app.world_mut(), first));
        assert!(mark_aoi_entity_dirty(app.world_mut(), second));
        app.world_mut().run_schedule(FixedPostUpdate);
        app.world_mut()
            .get_mut::<ReplicatedMarker>(first)
            .unwrap()
            .0 = 20;
        app.world_mut()
            .get_mut::<ReplicatedMarker>(second)
            .unwrap()
            .0 = 10;
        assert!(mark_aoi_entity_dirty(app.world_mut(), first));
        assert!(mark_aoi_entity_dirty(app.world_mut(), second));
        app.world_mut().run_schedule(FixedPostUpdate);
        assert_eq!(
            app.world()
                .resource::<EcsAoiProjectionStatus>()
                .pending_entities,
            0
        );
        assert_eq!(app.world().resource::<EcsAoiProjectionStatus>().failures, 0);
        assert_eq!(
            app.world()
                .resource::<MultiPeerAuthority<Strategy>>()
                .aoi()
                .entity_count(),
            2
        );
    }

    struct ConstantIdentitySource;
    impl EcsAoiProjectionSource for ConstantIdentitySource {
        fn stable_id(&self, _: &World, _: Entity) -> Option<u32> {
            Some(1)
        }
        fn position(&self, _: &World, entity: Entity) -> Option<(f32, f32)> {
            Some((entity.index() as f32, 0.0))
        }
        fn snapshot(&self, _: &World, entity: Entity) -> Result<Vec<u8>, EcsAoiSourceError> {
            Ok(entity.to_bits().to_le_bytes().to_vec())
        }
    }

    #[test]
    fn duplicate_stable_identity_fails_batch_without_publishing_either_entity() {
        let mut app = prism_app::App::new();
        app.insert_resource(authority());
        app.add_plugins(
            EcsAoiProjectionPlugin::<ConstantIdentitySource, Strategy>::new(
                ConstantIdentitySource,
                EcsAoiProjectionConfig::new(4, 4).unwrap(),
            ),
        );
        let first = app.world_mut().spawn(());
        let second = app.world_mut().spawn(());
        assert!(mark_aoi_entity_dirty(app.world_mut(), first));
        assert!(mark_aoi_entity_dirty(app.world_mut(), second));
        app.world_mut().run_schedule(FixedPostUpdate);
        assert_eq!(
            app.world()
                .resource::<MultiPeerAuthority<Strategy>>()
                .aoi()
                .entity_count(),
            0
        );
        assert_eq!(
            app.world()
                .resource::<EcsAoiProjectionStatus>()
                .pending_entities,
            2
        );
        assert_eq!(app.world().resource::<EcsAoiProjectionStatus>().failures, 1);
    }
}
