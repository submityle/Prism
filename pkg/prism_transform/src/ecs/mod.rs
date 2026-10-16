//! Optional `prism_ecs` integration (ECS design §24.2).
//!
//! Enable `ecs`, call [`install`], and run [`transform_system`] after local
//! writes and deferred hierarchy commands. The schedule's final observer flush
//! delivers actual [`GlobalTransform`] changes to `World::observe_changed`.
//! Call [`drain_spatial_updates`] after that flush to consume spatial edits.
//! For manual execution, call [`propagate_transforms`] followed by
//! `world.flush_changed_observers()` before draining.
//!
//! Locals remain authoritative. Lifecycle observers track membership and column
//! chunk summaries limit local-value reads to changed rows. A bounded,
//! per-kind relation journal identifies sources whose `ChildOf` targets need
//! rereading. History gaps and restore explicitly rebuild the parent baseline.
//! Static frames check chunk summaries and seek the bounded relation journal without querying
//! membership or per-entity parents, rebuilding dense buffers, composing matrices
//! or writing globals. Snapshot restore invalidates both the
//! propagation cache and spatial baseline reconciliation through the world's
//! archetype generation; ordinary spatial edits visit only bus-signalled entities.
//!
//! A direct parent without a local transform forms a transform boundary: its
//! child is a transform root. Cycles among transform-bearing entities and
//! multiple parents are rejected before any global writes. Floating-point
//! arithmetic uses the standalone CPU propagation order; deterministic delivery
//! here does not imply cross-platform floating-point bit identity.

mod propagation;
mod spatial;

use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use prism_ecs::component::Component;
use prism_ecs::observer::{LifecycleEvent, ObserverId};
use prism_ecs::resource::Resource;
use prism_ecs::world::World;

use crate::{GlobalTransform, Transform};
pub use propagation::{
    propagate_transforms, transform_system, EcsPropagationStats, TransformEcsError,
};
pub use spatial::{
    drain_spatial_updates, set_spatial_margin, spatial_snapshot, LocalBounds, SpatialUpdate,
};

impl Component for Transform {}
impl Component for GlobalTransform {}

struct Registration(Vec<ObserverId>);
impl Resource for Registration {}

// The first subscriber's id identifies this installation within its World.
// Paired with WorldId, it distinguishes uninstall/reinstall during a callback.
fn registration_id(world: &World) -> Option<ObserverId> {
    world
        .get_resource::<Registration>()
        .and_then(|registration| registration.0.first().copied())
}

pub(super) struct PropagationSignals {
    generation: u64,
    pub(super) membership: BTreeSet<prism_ecs::entity::Entity>,
    pub(super) globals: BTreeSet<prism_ecs::entity::Entity>,
}
impl Resource for PropagationSignals {}

impl PropagationSignals {
    // Restore does not fire lifecycle observers or restore this infrastructure
    // resource. Separate old-timeline events from edits made after restore,
    // even when several restores occur before the next propagation pass.
    fn rebase(&mut self, generation: u64) {
        if self.generation != generation {
            self.membership.clear();
            self.globals.clear();
            self.generation = generation;
        }
    }
}

/// Install the propagation cache and spatial bus subscribers once per world.
///
/// Existing global/bounds pairs are seeded once, including identity poses.
/// Subsequent additions/removals use lifecycle observers; in-place writes use
/// change observers. No required-component rule is installed: propagation adds
/// missing globals and removes the globals of locals it previously managed.
pub fn install(world: &mut World) {
    if world.contains_resource::<Registration>() {
        return;
    }
    if !world
        .components()
        .id_of::<prism_ecs::hierarchy::ChildOf>()
        .is_some_and(|id| world.relations().is_registered(id))
    {
        world.register_child_of();
    }
    let state = propagation::PropagationState::new(world);
    let spatial = spatial::SpatialState::new(world);
    world.insert_resource(state);
    world.insert_resource(spatial);
    world.insert_resource(PropagationSignals {
        generation: world.archetype_generation(),
        membership: BTreeSet::new(),
        globals: BTreeSet::new(),
    });

    let mut observers = Vec::<ObserverId>::new();
    observers.push(world.observe_changed::<GlobalTransform, _>(|ctx| {
        let entities = ctx.changed().to_vec();
        ctx.world
            .resource_mut::<spatial::SpatialState>()
            .pending
            .extend(entities);
    }));
    observers.push(world.observe_changed::<LocalBounds, _>(|ctx| {
        let entities = ctx.changed().to_vec();
        ctx.world
            .resource_mut::<spatial::SpatialState>()
            .pending
            .extend(entities);
    }));
    for event in [LifecycleEvent::Add, LifecycleEvent::Remove] {
        observers.push(world.observe::<Transform, _>(event, |ctx| {
            let generation = ctx.world.archetype_generation();
            let signals = ctx.world.resource_mut::<PropagationSignals>();
            signals.rebase(generation);
            signals.membership.insert(ctx.entity);
        }));
    }
    for event in [LifecycleEvent::Insert, LifecycleEvent::Remove] {
        observers.push(world.observe::<GlobalTransform, _>(event, |ctx| {
            let generation = ctx.world.archetype_generation();
            let signals = ctx.world.resource_mut::<PropagationSignals>();
            signals.rebase(generation);
            signals.globals.insert(ctx.entity);
            ctx.world
                .resource_mut::<spatial::SpatialState>()
                .pending
                .insert(ctx.entity);
        }));
        observers.push(world.observe::<LocalBounds, _>(event, |ctx| {
            ctx.world
                .resource_mut::<spatial::SpatialState>()
                .pending
                .insert(ctx.entity);
        }));
    }
    world.insert_resource(Registration(observers));
    // Arm the first manual writes, even without a schedule advancing ticks.
    world.increment_change_tick();
}

/// Remove this integration's observers and caches, retaining ECS components.
///
/// Returns deterministic removals for every previously drained live proxy,
/// including proxies with pending moves/removals. Consume these in external
/// indices before discarding them. Never-drained insertions need no removal.
/// Reinstalling seeds current pairs as fresh insertions. The optional
/// `prism_transform_ecs` adapter consumes these removals in its owned indices.
pub fn uninstall(world: &mut World) -> Vec<SpatialUpdate> {
    if let Some(registration) = world.remove_resource::<Registration>() {
        for observer in registration.0 {
            world.remove_observer(observer);
        }
    }
    world.remove_resource::<propagation::PropagationState>();
    world.remove_resource::<PropagationSignals>();
    world
        .remove_resource::<spatial::SpatialState>()
        .map_or_else(Vec::new, |state| state.removals())
}
