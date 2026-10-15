use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicUsize, Ordering};

use super::{CommandQueue, ParallelCommandBuffers};
#[cfg(feature = "std")]
use crate::bundle::Bundle;
use crate::change::ComponentTicks;
use crate::component::{Component, ComponentId, Components, StorageType};
use crate::component_hooks::{ComponentHooks, HookContext};
use crate::entity::Entity;
use crate::observer::LifecycleEvent;
use crate::resource::Resource;
use crate::world::World;

macro_rules! components {
    ($($name:ident),*) => {$(
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        struct $name(i32);
        impl Component for $name {}
    )*};
}
components!(A, B, C, D);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Sparse(i32);
impl Component for Sparse {
    const STORAGE: StorageType = StorageType::SparseSet;
}

#[repr(align(256))]
struct Aligned(u64);
impl Component for Aligned {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Shared(u32);
impl Component for Shared {
    const STORAGE: StorageType = StorageType::Shared;
    fn install_storage_glue(components: &mut Components, id: ComponentId) {
        components.set_shared_box(id, crate::component::shared_box_of::<Self>());
    }
}

struct Owned(Arc<AtomicUsize>);
impl Component for Owned {}
impl Drop for Owned {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn mixed_edits_do_one_real_table_migration() {
    let mut immediate = World::new();
    let reference = immediate.spawn((A(1), B(2)));
    immediate.spawn((A(3), B(4)));
    let before = immediate.archetypes().len();
    immediate.insert(reference, C(5));
    immediate.remove::<A>(reference);
    immediate.insert(reference, D(6));
    immediate.remove::<B>(reference);
    assert_eq!(immediate.archetypes().len(), before + 4);

    let mut world = World::new();
    let entity = world.spawn((A(1), B(2)));
    let neighbor = world.spawn((A(3), B(4)));
    let source = world.entities().location(entity).unwrap().archetype_id;
    let epoch = world
        .archetypes()
        .get(source)
        .unwrap()
        .table()
        .chunk_meta(0)
        .structural_epoch();
    let archetypes = world.archetypes().len();
    let mut queue = CommandQueue::new();
    queue
        .commands(world.entities())
        .entity(entity)
        .insert(C(5))
        .remove::<A>()
        .insert(D(6))
        .remove::<B>();
    assert_eq!(queue.len(), 4);
    queue.apply(&mut world);
    assert!(queue.is_empty());
    assert_eq!(world.archetypes().len(), archetypes + 1);
    assert_eq!(
        world
            .archetypes()
            .get(source)
            .unwrap()
            .table()
            .chunk_meta(0)
            .structural_epoch(),
        epoch + 1
    );
    let dest = world.entities().location(entity).unwrap().archetype_id;
    assert_ne!(source, dest);
    assert_eq!(
        world
            .archetypes()
            .get(dest)
            .unwrap()
            .table()
            .chunk_meta(0)
            .structural_epoch(),
        1
    );
    assert_eq!(world.get::<C>(entity), Some(&C(5)));
    assert_eq!(world.get::<D>(entity), Some(&D(6)));
    assert_eq!(world.get::<A>(entity), None);
    assert_eq!(world.get::<B>(entity), None);
    assert_eq!(world.get::<A>(neighbor), Some(&A(3)));
    assert_eq!(world.entities().location(neighbor).unwrap().row, 0);
}

#[test]
fn consecutive_removes_create_only_the_final_archetype() {
    let mut world = World::new();
    let entity = world.spawn((A(1), B(2), C(3), D(4)));
    let before = world.archetypes().len();
    let mut queue = CommandQueue::new();
    queue
        .commands(world.entities())
        .entity(entity)
        .remove::<A>()
        .remove::<B>()
        .remove::<C>();
    queue.apply(&mut world);
    assert_eq!(world.archetypes().len(), before + 1);
    assert_eq!(world.get::<D>(entity), Some(&D(4)));
    assert_eq!(world.query::<&D>().iter(&world).count(), 1);
}

#[test]
fn canceled_edits_and_reinsert_preserve_component_lifetimes() {
    let mut world = World::new();
    let entity = world.spawn((A(1), B(2), Sparse(3)));
    let untouched = world.get_ticks::<B>(entity).unwrap();
    let before = world.archetypes().len();
    let now = world.increment_change_tick();
    let mut queue = CommandQueue::new();
    queue
        .commands(world.entities())
        .entity(entity)
        .insert(A(4))
        .remove::<A>()
        .insert(A(5))
        .insert(C(6))
        .remove::<C>()
        .remove::<Sparse>()
        .insert(Sparse(7))
        .insert(Sparse(8));
    queue.apply(&mut world);
    assert_eq!(world.archetypes().len(), before);
    assert_eq!(world.get::<A>(entity), Some(&A(5)));
    assert_eq!(world.get::<Sparse>(entity), Some(&Sparse(8)));
    assert_eq!(world.get::<C>(entity), None);
    assert_eq!(world.get_ticks::<B>(entity), Some(untouched));
    assert_eq!(
        world.get_ticks::<A>(entity),
        Some(ComponentTicks {
            added: now,
            changed: now
        })
    );
    assert_eq!(
        world.get_ticks::<Sparse>(entity),
        Some(ComponentTicks {
            added: now,
            changed: now
        })
    );
}

#[test]
fn aligned_values_and_untouched_owned_values_move_without_extra_drop() {
    let drops = Arc::new(AtomicUsize::new(0));
    let mut world = World::new();
    let entity = world.spawn((A(1), Owned(drops.clone())));
    let added = world.get_ticks::<A>(entity).unwrap().added;
    let now = world.increment_change_tick();
    let before = world.archetypes().len();
    let mut queue = CommandQueue::new();
    queue
        .commands(world.entities())
        .entity(entity)
        .insert(A(2))
        .insert(Aligned(123))
        .insert(B(3));
    queue.apply(&mut world);
    assert_eq!(world.archetypes().len(), before + 1);
    assert_eq!(world.get::<Aligned>(entity).unwrap().0, 123);
    assert_eq!(
        (world.get::<Aligned>(entity).unwrap() as *const Aligned as usize) % 256,
        0
    );
    assert_eq!(
        world.get_ticks::<A>(entity),
        Some(ComponentTicks {
            added,
            changed: now
        })
    );
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    world.despawn(entity);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn same_shape_reinsert_keeps_neighbors_and_owned_values_valid() {
    let drops = Arc::new(AtomicUsize::new(0));
    let mut world = World::new();
    let entity = world.spawn((A(1), Owned(drops.clone())));
    let neighbor = world.spawn((A(2), Owned(drops.clone())));
    let now = world.increment_change_tick();
    let mut queue = CommandQueue::new();
    queue
        .commands(world.entities())
        .entity(entity)
        .remove::<A>()
        .insert(A(3));
    queue.apply(&mut world);
    assert_eq!(world.get::<A>(entity), Some(&A(3)));
    assert_eq!(world.get_ticks::<A>(entity).unwrap().added, now);
    assert_eq!(world.get::<A>(neighbor), Some(&A(2)));
    assert_eq!(world.query::<&Owned>().iter(&world).count(), 2);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(world);
    assert_eq!(drops.load(Ordering::SeqCst), 2);
}

#[test]
fn closures_and_entity_switches_are_ordered_barriers() {
    let mut world = World::new();
    let entity = world.spawn(A(1));
    let other = world.spawn(A(2));
    let mut queue = CommandQueue::new();
    let mut commands = queue.commands(world.entities());
    commands.entity(entity).insert(B(3));
    commands.queue(move |world| {
        assert_eq!(world.get::<B>(entity), Some(&B(3)));
        assert_eq!(world.get::<C>(entity), None);
        world.insert(entity, D(4));
    });
    commands.entity(entity).insert(C(5));
    commands.entity(other).remove::<A>().insert(B(6));
    commands.queue(move |world| {
        assert_eq!(world.get::<C>(entity), Some(&C(5)));
        assert_eq!(world.get::<D>(entity), Some(&D(4)));
        assert_eq!(world.get::<A>(other), None);
        assert_eq!(world.get::<B>(other), Some(&B(6)));
        world.despawn(entity);
    });
    commands.entity(entity).insert(C(7)).remove::<D>();
    queue.apply(&mut world);
    assert!(!world.contains(entity));
}

#[derive(Default)]
struct Trace(Vec<(&'static str, i32, bool)>);
impl Resource for Trace {}

fn trace(world: &mut World, entity: Entity, event: &'static str) {
    let value = world.get::<A>(entity).unwrap().0;
    let b = world.get::<B>(entity).is_some();
    world.resource_mut::<Trace>().0.push((event, value, b));
}
fn on_add(ctx: HookContext<'_>) {
    trace(ctx.world, ctx.entity, "hook.add");
}
fn on_insert(ctx: HookContext<'_>) {
    trace(ctx.world, ctx.entity, "hook.insert");
}
fn on_replace(ctx: HookContext<'_>) {
    trace(ctx.world, ctx.entity, "hook.replace");
}
fn on_remove(ctx: HookContext<'_>) {
    trace(ctx.world, ctx.entity, "hook.remove");
}

fn lifecycle_trace(deferred: bool) -> Vec<(&'static str, i32, bool)> {
    let mut world = World::new();
    let entity = world.spawn(A(1));
    world.insert_resource(Trace::default());
    world.register_component_hooks::<A>(
        ComponentHooks::new()
            .with_on_add(on_add)
            .with_on_insert(on_insert)
            .with_on_replace(on_replace)
            .with_on_remove(on_remove),
    );
    for (event, name) in [
        (LifecycleEvent::Add, "observer.add"),
        (LifecycleEvent::Insert, "observer.insert"),
        (LifecycleEvent::Replace, "observer.replace"),
        (LifecycleEvent::Remove, "observer.remove"),
    ] {
        world.observe::<A, _>(event, move |ctx| trace(ctx.world, ctx.entity, name));
    }
    if deferred {
        let mut queue = CommandQueue::new();
        queue
            .commands(world.entities())
            .entity(entity)
            .insert(A(2))
            .insert(B(3))
            .remove::<A>()
            .insert(A(4));
        queue.apply(&mut world);
    } else {
        world.insert(entity, A(2));
        world.insert(entity, B(3));
        world.remove::<A>(entity);
        world.insert(entity, A(4));
    }
    world.resource::<Trace>().0.clone()
}

#[test]
fn hooks_and_observers_see_the_exact_immediate_trace() {
    let expected = lifecycle_trace(false);
    assert_eq!(expected.len(), 12);
    assert_eq!(lifecycle_trace(true), expected);
}

#[test]
fn required_components_and_reentrant_observer_writes_are_preserved() {
    let mut world = World::new();
    world.register_required_component::<A, B>();
    world.observe::<A, _>(LifecycleEvent::Insert, |ctx| {
        assert_eq!(ctx.world.get::<B>(ctx.entity), Some(&B(0)));
        ctx.world.insert(ctx.entity, C(7));
    });
    let entity = world.spawn(());
    let mut queue = CommandQueue::new();
    queue
        .commands(world.entities())
        .entity(entity)
        .insert(A(1))
        .remove::<B>()
        .insert(D(8));
    queue.apply(&mut world);
    assert_eq!(world.get::<B>(entity), None);
    assert_eq!(world.get::<C>(entity), Some(&C(7)));
    assert_eq!(world.get::<D>(entity), Some(&D(8)));
}

#[test]
fn carried_shared_bindings_keep_their_pool_reference() {
    let mut world = World::new();
    let entity = world.spawn((A(1), Shared(7)));
    let before = world.archetypes().len();
    let mut queue = CommandQueue::new();
    queue
        .commands(world.entities())
        .entity(entity)
        .insert(B(2))
        .remove::<A>()
        .insert(C(3));
    queue.apply(&mut world);
    assert_eq!(world.archetypes().len(), before + 1);
    assert_eq!(world.get::<Shared>(entity), Some(&Shared(7)));
    world.despawn(entity);
    let new = world.spawn((B(4), C(5), Shared(8)));
    assert_eq!(world.get::<Shared>(new), Some(&Shared(8)));
}

#[test]
fn drop_and_shared_edits_replay_before_the_next_command() {
    let drops = Arc::new(AtomicUsize::new(0));
    let mut world = World::new();
    let entity = world.spawn((A(1), Owned(drops.clone()), Shared(7)));
    let mut queue = CommandQueue::new();
    let mut commands = queue.commands(world.entities());
    commands.entity(entity).insert(B(2)).remove::<Owned>();
    let observed = drops.clone();
    commands.queue(move |world| {
        assert_eq!(observed.load(Ordering::SeqCst), 1);
        assert_eq!(world.get::<B>(entity), Some(&B(2)));
    });
    commands
        .entity(entity)
        .insert(Shared(8))
        .remove::<Shared>()
        .insert(Shared(9));
    queue.apply(&mut world);
    assert_eq!(world.get::<Shared>(entity), Some(&Shared(9)));
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn parallel_same_entity_replay_is_independent_of_recording_order() {
    fn run(order: &[usize]) -> (Vec<super::CommandKey>, Vec<(i32, i32)>, usize) {
        #[derive(Default)]
        struct Log(Vec<(i32, i32)>);
        impl Resource for Log {}
        let mut world = World::new();
        world.insert_resource(Log::default());
        let entity = world.spawn(A(-1));
        let mut buffers = ParallelCommandBuffers::with_producers(3);
        for &p in order {
            let mut commands = buffers.producer(p, world.entities());
            commands
                .entity(entity)
                .insert(A(p as i32))
                .insert(B(p as i32 + 10));
            commands.queue(move |world| {
                let values = (
                    world.get::<A>(entity).unwrap().0,
                    world.get::<B>(entity).unwrap().0,
                );
                world.resource_mut::<Log>().0.push(values);
            });
        }
        let plan = buffers.plan();
        assert_eq!(plan.len(), 9);
        buffers.apply(&mut world);
        assert!(buffers.is_empty());
        assert_eq!(world.get::<A>(entity), Some(&A(2)));
        (
            plan,
            world.resource::<Log>().0.clone(),
            world.archetypes().len(),
        )
    }
    let expected = run(&[0, 1, 2]);
    assert_eq!(expected.1, vec![(0, 10), (1, 11), (2, 12)]);
    assert_eq!(run(&[2, 0, 1]), expected);
    assert_eq!(run(&[1, 2, 0]), expected);
}

#[test]
fn spawn_despawn_and_stale_generations_are_barriers() {
    let mut world = World::new();
    let mut queue = CommandQueue::new();
    let mut commands = queue.commands(world.entities());
    let entity = commands.spawn(A(1));
    commands.entity(entity).insert(B(2)).insert(C(3)).despawn();
    commands.entity(entity).insert(D(4));
    queue.apply(&mut world);
    assert!(!world.contains(entity));
    let new = world.spawn(A(5));
    assert_eq!(entity.index(), new.index());
    assert_ne!(entity, new);
    queue
        .commands(world.entities())
        .entity(entity)
        .insert(B(6))
        .remove::<A>();
    queue.apply(&mut world);
    assert_eq!(world.get::<A>(new), Some(&A(5)));
    assert_eq!(world.get::<B>(new), None);
}

#[test]
fn owning_group_remove_reinsert_retains_sequential_member_order() {
    fn run(deferred: bool) -> Vec<Entity> {
        let mut world = World::new();
        let a = world.register_component::<A>();
        let b = world.register_component::<B>();
        let group = world.register_owning_group(&[a, b]).unwrap();
        let first = world.spawn((A(1), B(2)));
        world.spawn((A(3), B(4)));
        if deferred {
            let mut queue = CommandQueue::new();
            queue
                .commands(world.entities())
                .entity(first)
                .remove::<A>()
                .insert(A(5));
            queue.apply(&mut world);
        } else {
            world.remove::<A>(first);
            world.insert(first, A(5));
        }
        world.owning_group(group).unwrap().packed().to_vec()
    }
    assert_eq!(run(true), run(false));
}

#[test]
fn changed_observers_and_world_reactions_see_coalesced_ticks() {
    let mut world = World::new();
    let entity = world.spawn(A(1));
    let source = world.reaction_source::<A>();
    #[derive(Default)]
    struct Seen(Vec<Vec<Entity>>);
    impl Resource for Seen {}
    world.insert_resource(Seen::default());
    world.add_reaction(&[source], move |world| {
        let changed = world.reaction_changes(source).to_vec();
        world.resource_mut::<Seen>().0.push(changed);
    });
    world.flush_changed_observers();
    world.flush_reactions();
    world.resource_mut::<Seen>().0.clear();
    world.increment_change_tick();
    let archetypes = world.archetypes().len();
    let mut queue = CommandQueue::new();
    queue
        .commands(world.entities())
        .entity(entity)
        .insert(A(2))
        .insert(B(3))
        .insert(A(4));
    queue.apply(&mut world);
    assert_eq!(world.archetypes().len(), archetypes + 1);
    // The changed-observer flush also settles the World reaction graph.
    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(world.flush_reactions(), 0);
    assert_eq!(world.resource::<Seen>().0, vec![vec![entity]]);
    world.flush_changed_observers();
    assert_eq!(world.flush_reactions(), 0);
}

#[cfg(feature = "std")]
#[test]
fn duplicate_bundle_panics_after_prior_structural_edits_are_committed() {
    let mut world = World::new();
    let entity = world.spawn(A(1));
    let mut queue = CommandQueue::new();
    queue
        .commands(world.entities())
        .entity(entity)
        .insert(B(2))
        .insert((C(3), C(4)));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| queue.apply(&mut world)));
    assert!(result.is_err());
    assert_eq!(world.get::<B>(entity), Some(&B(2)));
    assert_eq!(world.get::<C>(entity), None);
}

#[cfg(feature = "std")]
#[test]
fn registration_panic_commits_earlier_commands() {
    struct PanicsOnRegistration;
    impl Component for PanicsOnRegistration {
        fn install_storage_glue(_: &mut Components, _: ComponentId) {
            panic!("registration failure");
        }
    }
    let mut world = World::new();
    let entity = world.spawn(A(1));
    let mut queue = CommandQueue::new();
    queue
        .commands(world.entities())
        .entity(entity)
        .insert(B(2))
        .insert(PanicsOnRegistration);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| queue.apply(&mut world)));
    assert!(result.is_err());
    assert_eq!(world.get::<B>(entity), Some(&B(2)));
}

#[cfg(feature = "std")]
struct PanicAfterYield<B>(B);

#[cfg(feature = "std")]
// SAFETY: delegate both orderings to the same Bundle. Every value is yielded
// exactly once before panicking; no transferred value is dropped by this wrapper.
unsafe impl<B: Bundle> Bundle for PanicAfterYield<B> {
    fn component_ids(components: &mut Components, out: &mut Vec<ComponentId>) {
        B::component_ids(components, out);
    }

    unsafe fn get_components(self, func: &mut dyn FnMut(*mut u8)) {
        // SAFETY: the caller consumes the values under B's original contract.
        unsafe { self.0.get_components(func) };
        panic!("bundle extraction failure after all values were yielded");
    }
}

#[cfg(feature = "std")]
#[test]
fn bundle_take_panic_preserves_sequential_prefix_values_ticks_and_ownership() {
    // Cover in-place replacement, same-shape remove/reinsert, table growth and
    // table shrinkage. A failed insert must not resurrect an accepted removal.
    for mode in 0..4 {
        let expected_drops = Arc::new(AtomicUsize::new(0));
        let actual_drops = Arc::new(AtomicUsize::new(0));
        let mut expected = World::new();
        let mut actual = World::new();
        let entity = expected.spawn((A(1), B(2), Sparse(3), Owned(expected_drops.clone())));
        assert_eq!(
            actual.spawn((A(1), B(2), Sparse(3), Owned(actual_drops.clone()))),
            entity
        );
        let neighbor = expected.spawn((A(4), B(5), Sparse(6), Owned(expected_drops.clone())));
        assert_eq!(
            actual.spawn((A(4), B(5), Sparse(6), Owned(actual_drops.clone()))),
            neighbor
        );
        expected.increment_change_tick();
        actual.increment_change_tick();

        // The reference is the successful command prefix replayed immediately.
        // The failing command is atomic here, even though immediate insert may
        // already have published its yielded values when get_components panics.
        expected.insert(entity, B(20));
        expected.remove::<Sparse>(entity);
        expected.insert(entity, Sparse(30));
        if mode == 1 {
            expected.remove::<A>(entity);
            expected.insert(entity, A(10));
        } else if mode == 2 {
            expected.insert(entity, D(40));
        } else if mode == 3 {
            expected.remove::<A>(entity);
        }

        let mut queue = CommandQueue::new();
        let mut commands = queue.commands(actual.entities());
        commands
            .entity(entity)
            .insert(B(20))
            .remove::<Sparse>()
            .insert(Sparse(30));
        if mode == 1 {
            commands.entity(entity).remove::<A>().insert(A(10));
        } else if mode == 2 {
            commands.entity(entity).insert(D(40));
        } else if mode == 3 {
            commands.entity(entity).remove::<A>();
        }
        commands
            .entity(entity)
            .insert(PanicAfterYield((A(99), B(99), C(99), Sparse(99))))
            .insert(D(99));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            queue.apply(&mut actual);
        }));
        assert!(result.is_err());
        assert!(queue.is_empty());
        for e in [entity, neighbor] {
            assert_same_component::<A>(&expected, &actual, e);
            assert_same_component::<B>(&expected, &actual, e);
            assert_same_component::<C>(&expected, &actual, e);
            assert_same_component::<D>(&expected, &actual, e);
            assert_same_component::<Sparse>(&expected, &actual, e);
        }
        assert_eq!(actual.query::<&Owned>().iter(&actual).count(), 2);
        assert_eq!(expected_drops.load(Ordering::SeqCst), 0);
        assert_eq!(actual_drops.load(Ordering::SeqCst), 0);
        // Reuse the world after unwind, including another structural migration.
        queue
            .commands(actual.entities())
            .entity(entity)
            .insert(C(50));
        queue.apply(&mut actual);
        assert_eq!(actual.get::<C>(entity), Some(&C(50)));
        assert_eq!(actual.get::<A>(neighbor), Some(&A(4)));
        drop(expected);
        drop(actual);
        assert_eq!(expected_drops.load(Ordering::SeqCst), 2);
        assert_eq!(actual_drops.load(Ordering::SeqCst), 2);
    }
}

#[cfg(feature = "std")]
#[test]
fn bundle_take_panic_without_a_prefix_does_not_publish_or_migrate() {
    let mut world = World::new();
    let entity = world.spawn((A(1), Sparse(2)));
    let location = world.entities().location(entity);
    let ticks = world.get_ticks::<A>(entity);
    let sparse_ticks = world.get_ticks::<Sparse>(entity);
    let archetypes = world.archetypes().len();
    world.increment_change_tick();
    let later = Arc::new(AtomicUsize::new(0));
    let seen = later.clone();
    let mut queue = CommandQueue::new();
    let mut commands = queue.commands(world.entities());
    commands
        .entity(entity)
        .insert(PanicAfterYield((A(9), C(9), Sparse(9))));
    commands.queue(move |_| {
        seen.fetch_add(1, Ordering::SeqCst);
    });
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| queue.apply(&mut world)));
    assert!(result.is_err());
    assert_eq!(world.get::<A>(entity), Some(&A(1)));
    assert_eq!(world.get::<Sparse>(entity), Some(&Sparse(2)));
    assert_eq!(world.get::<C>(entity), None);
    assert_eq!(world.get_ticks::<A>(entity), ticks);
    assert_eq!(world.get_ticks::<Sparse>(entity), sparse_ticks);
    assert_eq!(world.entities().location(entity), location);
    assert_eq!(world.archetypes().len(), archetypes);
    assert_eq!(later.load(Ordering::SeqCst), 0);
    assert!(queue.is_empty());
}

#[cfg(feature = "std")]
#[test]
fn empty_bundle_take_panic_matches_immediate_replay() {
    let mut expected = World::new();
    let mut actual = World::new();
    let entity = expected.spawn(A(1));
    assert_eq!(actual.spawn(A(1)), entity);
    expected.increment_change_tick();
    actual.increment_change_tick();
    let immediate = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        expected.insert(entity, B(2));
        expected.insert(entity, PanicAfterYield(()));
    }));
    let mut queue = CommandQueue::new();
    queue
        .commands(actual.entities())
        .entity(entity)
        .insert(B(2))
        .insert(PanicAfterYield(()));
    let deferred = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        queue.apply(&mut actual);
    }));
    assert!(immediate.is_err());
    assert!(deferred.is_err());
    assert_same_component::<A>(&expected, &actual, entity);
    assert_same_component::<B>(&expected, &actual, entity);
}

#[cfg(feature = "std")]
#[test]
fn bundle_take_panic_commits_prefix_across_parallel_producers() {
    let mut expected = World::new();
    let mut actual = World::new();
    let entity = expected.spawn((A(1), B(2), Sparse(3)));
    assert_eq!(actual.spawn((A(1), B(2), Sparse(3))), entity);
    expected.increment_change_tick();
    actual.increment_change_tick();
    expected.remove::<A>(entity);
    expected.insert(entity, A(10));
    expected.insert(entity, B(20));
    expected.insert(entity, D(40));

    let mut buffers = ParallelCommandBuffers::with_producers(3);
    // Record out of order; acceptance is by deterministic producer/sequence key.
    buffers
        .producer(2, actual.entities())
        .entity(entity)
        .insert(PanicAfterYield((A(99), B(99), C(99), Sparse(99))))
        .remove::<D>();
    buffers
        .producer(1, actual.entities())
        .entity(entity)
        .insert(B(20))
        .insert(D(40));
    buffers
        .producer(0, actual.entities())
        .entity(entity)
        .remove::<A>()
        .insert(A(10));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        buffers.apply(&mut actual);
    }));
    assert!(result.is_err());
    assert!(buffers.is_empty());
    assert_same_component::<A>(&expected, &actual, entity);
    assert_same_component::<B>(&expected, &actual, entity);
    assert_same_component::<C>(&expected, &actual, entity);
    assert_same_component::<D>(&expected, &actual, entity);
    assert_same_component::<Sparse>(&expected, &actual, entity);
}

#[cfg(feature = "std")]
#[test]
fn bundle_destructor_panic_discards_current_values_and_drops_once() {
    struct PanicOnDrop {
        values: Option<(A, C)>,
        drops: Arc<AtomicUsize>,
    }
    impl Drop for PanicOnDrop {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
            panic!("bundle destructor failure");
        }
    }
    // SAFETY: exactly the two registered values are taken out of self and
    // transferred in order; the destructor cannot drop them again.
    unsafe impl Bundle for PanicOnDrop {
        fn component_ids(components: &mut Components, out: &mut Vec<ComponentId>) {
            <(A, C)>::component_ids(components, out);
        }

        unsafe fn get_components(mut self, func: &mut dyn FnMut(*mut u8)) {
            // SAFETY: the callback takes ownership under the tuple's contract.
            unsafe { self.values.take().unwrap().get_components(func) };
        }
    }
    let drops = Arc::new(AtomicUsize::new(0));
    let owned_drops = Arc::new(AtomicUsize::new(0));
    let mut world = World::new();
    let entity = world.spawn((A(1), Owned(owned_drops.clone())));
    let mut queue = CommandQueue::new();
    queue
        .commands(world.entities())
        .entity(entity)
        .insert(B(2))
        .insert(PanicOnDrop {
            values: Some((A(99), C(99))),
            drops: drops.clone(),
        });
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| queue.apply(&mut world)));
    assert!(result.is_err());
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(world.get::<A>(entity), Some(&A(1)));
    assert_eq!(world.get::<B>(entity), Some(&B(2)));
    assert_eq!(world.get::<C>(entity), None);
    assert_eq!(owned_drops.load(Ordering::SeqCst), 0);
    drop(world);
    assert_eq!(owned_drops.load(Ordering::SeqCst), 1);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn drop_components_flush_prefix_and_finish_before_later_commands() {
    let drops = Arc::new(AtomicUsize::new(0));
    let mut world = World::new();
    let entity = world.spawn((A(1), Owned(drops.clone())));
    let mut queue = CommandQueue::new();
    let mut commands = queue.commands(world.entities());
    commands
        .entity(entity)
        .insert(B(2))
        .insert(Owned(drops.clone()));
    let seen = drops.clone();
    commands.queue(move |world| {
        assert_eq!(world.get::<B>(entity), Some(&B(2)));
        assert_eq!(seen.load(Ordering::SeqCst), 1);
    });
    commands.entity(entity).insert(C(3)).remove::<Owned>();
    let seen = drops.clone();
    commands.queue(move |world| {
        assert_eq!(world.get::<C>(entity), Some(&C(3)));
        assert!(world.get::<Owned>(entity).is_none());
        assert_eq!(seen.load(Ordering::SeqCst), 2);
    });
    commands.entity(entity).insert(Owned(drops.clone()));
    queue.apply(&mut world);
    assert_eq!(drops.load(Ordering::SeqCst), 2);
    drop(world);
    assert_eq!(drops.load(Ordering::SeqCst), 3);
}

#[test]
fn hooks_installed_by_bundle_registration_flush_prior_changes() {
    struct InstallsHook;
    impl Component for InstallsHook {
        fn install_storage_glue(components: &mut Components, id: ComponentId) {
            components.set_hooks(
                id,
                ComponentHooks::new().with_on_add(|ctx| {
                    assert_eq!(ctx.world.get::<B>(ctx.entity), Some(&B(2)));
                    ctx.world.insert(ctx.entity, C(9));
                }),
            );
        }
    }
    let mut world = World::new();
    let entity = world.spawn(A(1));
    let mut queue = CommandQueue::new();
    queue
        .commands(world.entities())
        .entity(entity)
        .insert(B(2))
        .insert(InstallsHook)
        .insert(D(3));
    queue.apply(&mut world);
    assert_eq!(world.get::<C>(entity), Some(&C(9)));
    assert_eq!(world.get::<D>(entity), Some(&D(3)));
}

#[derive(Clone, Copy)]
enum Edit {
    PutA(i32),
    PutB(i32),
    PutC(i32),
    PutSparse(i32),
    RemoveA,
    RemoveB,
    RemoveC,
    RemoveSparse,
    RemoveAll,
}

impl Edit {
    fn immediate(self, world: &mut World, entity: Entity) {
        match self {
            Self::PutA(v) => {
                world.insert(entity, A(v));
            }
            Self::PutB(v) => {
                world.insert(entity, B(v));
            }
            Self::PutC(v) => {
                world.insert(entity, C(v));
            }
            Self::PutSparse(v) => {
                world.insert(entity, Sparse(v));
            }
            Self::RemoveA => {
                world.remove::<A>(entity);
            }
            Self::RemoveB => {
                world.remove::<B>(entity);
            }
            Self::RemoveC => {
                world.remove::<C>(entity);
            }
            Self::RemoveSparse => {
                world.remove::<Sparse>(entity);
            }
            Self::RemoveAll => {
                world.remove::<(A, B, C, Sparse)>(entity);
            }
        }
    }

    fn deferred(self, commands: &mut super::Commands<'_, '_>, entity: Entity) {
        let mut commands = commands.entity(entity);
        match self {
            Self::PutA(v) => {
                commands.insert(A(v));
            }
            Self::PutB(v) => {
                commands.insert(B(v));
            }
            Self::PutC(v) => {
                commands.insert(C(v));
            }
            Self::PutSparse(v) => {
                commands.insert(Sparse(v));
            }
            Self::RemoveA => {
                commands.remove::<A>();
            }
            Self::RemoveB => {
                commands.remove::<B>();
            }
            Self::RemoveC => {
                commands.remove::<C>();
            }
            Self::RemoveSparse => {
                commands.remove::<Sparse>();
            }
            Self::RemoveAll => {
                commands.remove::<(A, B, C, Sparse)>();
            }
        }
    }
}

fn assert_same_component<T: Component + core::fmt::Debug + PartialEq>(
    expected: &World,
    actual: &World,
    entity: Entity,
) {
    assert_eq!(actual.get::<T>(entity), expected.get::<T>(entity));
    assert_eq!(
        actual.get_ticks::<T>(entity),
        expected.get_ticks::<T>(entity)
    );
}

#[test]
fn seeded_mixed_structural_replay_matches_immediate_values_and_ticks() {
    let mut expected = World::new();
    let mut actual = World::new();
    let mut entities = Vec::new();
    for i in 0..8 {
        let e = expected.spawn((A(i), B(i), Sparse(i)));
        assert_eq!(actual.spawn((A(i), B(i), Sparse(i))), e);
        entities.push(e);
    }
    let mut seed = 0x9e37_79b9_u32;
    let mut queue = CommandQueue::new();
    // 6,144 edits, varying adjacency, empty final shapes, sparse-only entities,
    // overwrite/remove/reinsert lifetimes and absent-component removals.
    for frame in 0..64 {
        expected.increment_change_tick();
        actual.increment_change_tick();
        {
            let mut commands = queue.commands(actual.entities());
            for i in 0..96 {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                let entity = entities[(i / 12 + frame) % entities.len()];
                let v = seed as i32;
                let edit = match seed % 9 {
                    0 => Edit::PutA(v),
                    1 => Edit::PutB(v),
                    2 => Edit::PutC(v),
                    3 => Edit::PutSparse(v),
                    4 => Edit::RemoveA,
                    5 => Edit::RemoveB,
                    6 => Edit::RemoveC,
                    7 => Edit::RemoveSparse,
                    _ => Edit::RemoveAll,
                };
                edit.immediate(&mut expected, entity);
                edit.deferred(&mut commands, entity);
            }
        }
        queue.apply(&mut actual);
        for &entity in &entities {
            assert_same_component::<A>(&expected, &actual, entity);
            assert_same_component::<B>(&expected, &actual, entity);
            assert_same_component::<C>(&expected, &actual, entity);
            assert_same_component::<Sparse>(&expected, &actual, entity);
        }
        assert_eq!(
            actual.query::<&A>().iter(&actual).count(),
            expected.query::<&A>().iter(&expected).count()
        );
    }
}

#[test]
fn same_shape_reinsert_crosses_a_physical_chunk_boundary() {
    let drops = Arc::new(AtomicUsize::new(0));
    let mut world = World::new();
    let first = world.spawn((A(1), Aligned(7), Owned(drops.clone())));
    let location = world.entities().location(first).unwrap();
    let capacity = world
        .archetypes()
        .get(location.archetype_id)
        .unwrap()
        .table()
        .rows_per_chunk();
    let mut entities = vec![first];
    for i in 1..capacity {
        entities.push(world.spawn((A(i as i32), Aligned(i as u64), Owned(drops.clone()))));
    }
    let now = world.increment_change_tick();
    let mut queue = CommandQueue::new();
    queue
        .commands(world.entities())
        .entity(first)
        .remove::<A>()
        .insert(A(99));
    queue.apply(&mut world);
    assert_eq!(world.get::<A>(first), Some(&A(99)));
    assert_eq!(world.get_ticks::<A>(first).unwrap().added, now);
    assert_eq!(world.get::<Aligned>(first).unwrap().0, 7);
    for (i, &entity) in entities.iter().enumerate().skip(1) {
        assert_eq!(world.get::<A>(entity), Some(&A(i as i32)));
        assert_eq!(world.get::<Aligned>(entity).unwrap().0, i as u64);
    }
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(world);
    assert_eq!(drops.load(Ordering::SeqCst), capacity);
}

#[test]
fn zero_sized_components_can_be_canceled_and_reinserted() {
    #[repr(align(128))]
    struct Marker;
    impl Component for Marker {}
    let mut world = World::new();
    let entity = world.spawn((A(1), Marker));
    let now = world.increment_change_tick();
    let mut queue = CommandQueue::new();
    queue
        .commands(world.entities())
        .entity(entity)
        .remove::<Marker>()
        .insert(Marker)
        .insert(B(2))
        .remove::<A>();
    queue.apply(&mut world);
    assert!(world.get::<Marker>(entity).is_some());
    assert_eq!(world.get_ticks::<Marker>(entity).unwrap().added, now);
    assert_eq!(world.get::<B>(entity), Some(&B(2)));
}

#[test]
fn tuple_bundles_mix_table_and_sparse_edits_in_one_migration() {
    let mut world = World::new();
    let entity = world.spawn((A(1), B(2)));
    let before = world.archetypes().len();
    let mut queue = CommandQueue::new();
    queue
        .commands(world.entities())
        .entity(entity)
        .insert((C(3), (D(4), Sparse(5))))
        .remove::<(A, B)>();
    queue.apply(&mut world);
    assert_eq!(world.archetypes().len(), before + 1);
    assert_eq!(world.get::<C>(entity), Some(&C(3)));
    assert_eq!(world.get::<D>(entity), Some(&D(4)));
    assert_eq!(world.get::<Sparse>(entity), Some(&Sparse(5)));
    assert_eq!(world.get::<A>(entity), None);
    assert_eq!(world.get::<B>(entity), None);
}

#[test]
fn sparse_only_edits_leave_table_location_and_epoch_unchanged() {
    let mut world = World::new();
    let entity = world.spawn(A(1));
    let loc = world.entities().location(entity).unwrap();
    let epoch = world
        .archetypes()
        .get(loc.archetype_id)
        .unwrap()
        .table()
        .chunk_meta(0)
        .structural_epoch();
    let before = world.archetypes().len();
    let mut queue = CommandQueue::new();
    queue
        .commands(world.entities())
        .entity(entity)
        .insert(Sparse(2))
        .remove::<Sparse>()
        .insert(Sparse(3));
    queue.apply(&mut world);
    assert_eq!(world.entities().location(entity), Some(loc));
    assert_eq!(world.archetypes().len(), before);
    assert_eq!(
        world
            .archetypes()
            .get(loc.archetype_id)
            .unwrap()
            .table()
            .chunk_meta(0)
            .structural_epoch(),
        epoch
    );
    assert_eq!(world.get::<Sparse>(entity), Some(&Sparse(3)));
}

#[test]
fn parallel_runs_coalesce_across_adjacent_producer_boundaries() {
    let mut world = World::new();
    let entity = world.spawn(A(1));
    let mut buffers = ParallelCommandBuffers::with_producers(3);
    buffers
        .producer(2, world.entities())
        .entity(entity)
        .remove::<A>()
        .insert(D(4));
    buffers
        .producer(0, world.entities())
        .entity(entity)
        .insert(B(2));
    buffers
        .producer(1, world.entities())
        .entity(entity)
        .insert(C(3));
    let before = world.archetypes().len();
    buffers.apply(&mut world);
    assert_eq!(world.archetypes().len(), before + 1);
    assert_eq!(world.get::<A>(entity), None);
    assert_eq!(world.get::<B>(entity), Some(&B(2)));
    assert_eq!(world.get::<C>(entity), Some(&C(3)));
    assert_eq!(world.get::<D>(entity), Some(&D(4)));
}
