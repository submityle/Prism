//! Project-configured ECS to authority-AOI projection.

use alloc::collections::{BTreeMap, BTreeSet, VecDeque};
use alloc::vec::Vec;
use core::marker::PhantomData;

use prism_app::{FixedPostUpdate, Plugin};
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
        queue.overflowed = true;
        if let Some(status) = world.get_resource_mut::<EcsAoiProjectionStatus>() {
            status.overflowed = true;
            status.failures = status.failures.saturating_add(1);
        }
        return false;
    }
    queue.membership.insert(entity);
    queue.items.push_back(entity);
    true
}

fn mark_lifecycle_dirty(context: &mut ObserverContext<'_>) {
    let _ = mark_aoi_entity_dirty(context.world, context.entity);
}

type RegisterTracker = fn(&mut World);

fn register_tracker<T: Component>(world: &mut World) {
    world.observe::<T, _>(LifecycleEvent::Insert, mark_lifecycle_dirty);
    world.observe::<T, _>(LifecycleEvent::Remove, mark_lifecycle_dirty);
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
    let mut updates = Vec::with_capacity(count);
    let mut failed = false;
    for entity in &entities {
        let previous = state.published.get(entity).copied();
        if !world.contains(*entity) {
            if let Some(previous) = previous {
                operations.push(AuthorityEntityProjection::Remove {
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
                operations.push(AuthorityEntityProjection::Remove {
                    entity_id: previous.network_id,
                });
            }
            updates.push((*entity, None));
            continue;
        };
        if let Some(previous) = previous
            && previous.network_id != network_id
        {
            operations.push(AuthorityEntityProjection::Remove {
                entity_id: previous.network_id,
            });
        }
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

    if !failed && (operations.is_empty() || authority.project_entities_atomic(&operations).is_ok())
    {
        for (entity, published) in updates {
            if let Some(published) = published {
                state.published.insert(entity, published);
            } else {
                state.published.remove(&entity);
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
                .saturating_add(entities.len() as u64);
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
    struct ReplicatedMarker(u32);
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
                .map(|marker| marker.0.to_le_bytes().to_vec())
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
        let entity = app.world_mut().spawn(ReplicatedMarker(77));
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
                .pending_entities,
            0
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
            2
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
}
