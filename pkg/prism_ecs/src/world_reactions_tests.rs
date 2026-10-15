use alloc::vec;
use alloc::vec::Vec;

use crate::prelude::*;
use crate::world_reactions::ReactionId;

#[derive(PartialEq)]
struct Input(i32);
impl Component for Input {}

struct SparseInput(i32);
impl Component for SparseInput {
    const STORAGE: crate::component::StorageType = crate::component::StorageType::SparseSet;
}

#[derive(Default)]
struct Output {
    left: i32,
    right: i32,
    total: i32,
    runs: usize,
    batches: Vec<Vec<Entity>>,
}
impl Resource for Output {}

#[test]
fn component_changes_settle_a_diamond_once_with_consistent_inputs() {
    let mut world = World::new();
    let entity = world.spawn(Input(1));
    world.insert_resource(Output::default());
    let source = world.reaction_source::<Input>();
    assert_eq!(source, world.reaction_source::<Input>());
    let left = world.add_reaction(&[source], move |world| {
        let value = world.get::<Input>(entity).unwrap().0;
        let batch = world.reaction_changes(source).to_vec();
        let output = world.resource_mut::<Output>();
        output.left = value + 1;
        output.batches.push(batch);
    });
    let right = world.add_reaction(&[source], move |world| {
        world.resource_mut::<Output>().right = world.get::<Input>(entity).unwrap().0 + 2;
    });
    world.add_reaction(&[left, right], |world| {
        let output = world.resource_mut::<Output>();
        output.total = output.left + output.right;
        output.runs += 1;
    });
    world.flush_changed_observers();
    assert_eq!(world.resource::<Output>().runs, 0);
    world.get_mut::<Input>(entity).unwrap().0 = 10;
    world.get_mut::<Input>(entity).unwrap().0 = 20;
    world.flush_changed_observers();
    let output = world.resource::<Output>();
    assert_eq!(output.total, 43);
    assert_eq!(output.runs, 1);
    assert_eq!(output.batches, vec![vec![entity]]);
    world.flush_changed_observers();
    assert_eq!(world.resource::<Output>().runs, 1);
    assert!(world.reaction_changes(source).is_empty());
}

#[test]
fn reaction_batch_is_the_exact_entity_boundary_after_coarse_query_discovery() {
    let mut world = World::new();
    let first = world.spawn(Input(1));
    let second = world.spawn(Input(2));
    let source = world.reaction_source::<Input>();
    let delivered = alloc::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let delivered_copy = delivered.clone();
    world.add_reaction(&[source], move |world| {
        delivered_copy
            .lock()
            .unwrap()
            .push(world.reaction_changes(source).to_vec());
    });

    world.get_mut::<Input>(second).unwrap().0 += 1;
    world.get_mut::<Input>(first).unwrap().0 += 1;
    world.get_mut::<Input>(first).unwrap().0 += 1;
    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(*delivered.lock().unwrap(), vec![vec![first, second]]);
}

#[test]
fn unrelated_sources_and_equal_writes_do_not_execute_effects() {
    let mut world = World::new();
    let entity = world.spawn((Input(1), SparseInput(5)));
    world.insert_resource(Output::default());
    let source = world.reaction_source::<Input>();
    let sparse = world.reaction_source::<SparseInput>();
    world.add_reaction(&[source], |world| {
        world.resource_mut::<Output>().runs += 100
    });
    world.add_reaction(&[sparse], |world| world.resource_mut::<Output>().runs += 1);
    world.get_mut::<Input>(entity).unwrap().set_if_neq(Input(1));
    world.get_mut::<SparseInput>(entity).unwrap().0 = 6;
    world.flush_changed_observers();
    assert_eq!(world.resource::<Output>().runs, 1);
}

#[test]
fn effects_feed_components_on_the_next_flush_and_recursive_flush_is_safe() {
    let mut world = World::new();
    let entity = world.spawn(Input(1));
    world.insert_resource(Output::default());
    let source = world.reaction_source::<Input>();
    world.add_reaction(&[source], move |world| {
        world.resource_mut::<Output>().runs += 1;
        world.get_mut::<Input>(entity).unwrap().0 += 1;
        assert_eq!(world.flush_changed_observers(), 0);
    });
    world.get_mut::<Input>(entity).unwrap().0 = 10;
    world.flush_changed_observers();
    assert_eq!(world.resource::<Output>().runs, 1);
    assert_eq!(world.get::<Input>(entity).unwrap().0, 11);
    world.flush_changed_observers();
    assert_eq!(world.resource::<Output>().runs, 2);
}

#[test]
fn a_schedule_automatically_delivers_component_changes_to_effects() {
    let mut world = World::new();
    world.spawn(Input(1));
    world.insert_resource(Output::default());
    let source = world.reaction_source::<Input>();
    world.add_reaction(&[source], |world| world.resource_mut::<Output>().runs += 1);
    let mut schedule = Schedule::new();
    schedule.add_systems(|mut query: Query<&mut Input>| {
        for mut value in &mut query {
            value.0 += 1;
        }
    });
    schedule.run(&mut world);
    assert_eq!(world.resource::<Output>().runs, 1);
}

#[test]
#[should_panic(expected = "different World")]
fn reaction_handles_cannot_cross_worlds() {
    let mut first = World::new();
    let source = first.reaction_source::<Input>();
    World::new().add_reaction(&[source], |_| {});
}

#[cfg(feature = "std")]
#[test]
fn a_panicking_effect_keeps_its_callback_and_pending_work_for_retry() {
    use alloc::sync::Arc;
    use core::sync::atomic::{AtomicBool, Ordering};
    let mut world = World::new();
    let entity = world.spawn(Input(1));
    world.insert_resource(Output::default());
    let source = world.reaction_source::<Input>();
    let fail = Arc::new(AtomicBool::new(true));
    let fail_once = fail.clone();
    let derived = world.add_reaction(&[source], move |world| {
        assert!(!fail_once.swap(false, Ordering::Relaxed), "effect failed");
        assert_eq!(world.reaction_changes(source), &[entity]);
        world.resource_mut::<Output>().runs += 1;
    });
    world.add_reaction(&[derived], |world| {
        world.resource_mut::<Output>().runs += 10
    });
    world.get_mut::<Input>(entity).unwrap().0 = 2;
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        world.flush_changed_observers();
    }))
    .is_err());
    assert_eq!(world.flush_reactions(), 2);
    assert_eq!(world.resource::<Output>().runs, 11);
    assert_eq!(world.flush_reactions(), 0);
}

#[cfg(feature = "std")]
#[test]
fn retry_preserves_the_original_batch_and_defers_new_source_changes() {
    use alloc::sync::Arc;
    use core::sync::atomic::{AtomicBool, Ordering};
    let mut world = World::new();
    let first = world.spawn(Input(1));
    let second = world.spawn(Input(2));
    world.insert_resource(Output::default());
    let source = world.reaction_source::<Input>();
    let fail_once = Arc::new(AtomicBool::new(true));
    world.add_reaction(&[source], move |world| {
        assert!(!fail_once.swap(false, Ordering::Relaxed), "effect failed");
        let batch = world.reaction_changes(source).to_vec();
        world.resource_mut::<Output>().batches.push(batch);
    });
    world.get_mut::<Input>(first).unwrap().0 += 1;
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        world.flush_changed_observers();
    }))
    .is_err());
    world.get_mut::<Input>(second).unwrap().0 += 1;
    world.flush_changed_observers();
    assert_eq!(world.resource::<Output>().batches, vec![vec![first]]);
    world.flush_changed_observers();
    assert_eq!(
        world.resource::<Output>().batches,
        vec![vec![first], vec![second]]
    );
    assert_eq!(world.flush_changed_observers(), 0);
}

#[cfg(feature = "std")]
#[test]
fn repeated_effect_failures_freeze_all_sources_and_do_not_replay_successful_siblings() {
    #[derive(Default)]
    struct Attempts(Vec<(Vec<Entity>, Vec<Entity>)>);
    impl Resource for Attempts {}

    let mut world = World::new();
    let first = world.spawn(Input(0));
    let second = world.spawn(Input(0));
    let sparse_entity = world.spawn(SparseInput(0));
    world.insert_resource(Output::default());
    world.insert_resource(Attempts::default());
    let source = world.reaction_source::<Input>();
    let sparse = world.reaction_source::<SparseInput>();
    let succeeded = world.add_reaction(&[source], move |world| {
        let batch = world.reaction_changes(source).to_vec();
        world.resource_mut::<Output>().batches.push(batch);
    });
    let mut failures = 2;
    let failed = world.add_reaction(&[source, sparse], move |world| {
        let table_batch = world.reaction_changes(source).to_vec();
        let sparse_batch = world.reaction_changes(sparse).to_vec();
        world
            .resource_mut::<Attempts>()
            .0
            .push((table_batch, sparse_batch));
        if failures > 0 {
            failures -= 1;
            panic!("effect failed");
        }
    });
    world.add_reaction(&[succeeded, failed], |world| {
        world.resource_mut::<Output>().runs += 1;
    });
    world.get_mut::<Input>(first).unwrap().0 += 1;
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        world.flush_changed_observers();
    }))
    .is_err());

    world.get_mut::<Input>(second).unwrap().0 += 1;
    world.get_mut::<SparseInput>(sparse_entity).unwrap().0 += 1;
    // Explicit marks and new deliveries must not replay the successful sibling
    // or contaminate the failed effect's table/sparse source slices on retry.
    world.mark_reaction_dirty(succeeded);
    world.mark_reaction_dirty(source);
    world.mark_reaction_dirty(source);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        world.flush_changed_observers();
    }))
    .is_err());
    assert_eq!(world.resource::<Output>().batches, vec![vec![first]]);
    assert_eq!(world.resource::<Output>().runs, 0);
    assert_eq!(world.flush_reactions(), 2);
    assert_eq!(world.resource::<Output>().runs, 1);
    assert_eq!(
        world.resource::<Attempts>().0,
        vec![(vec![first], vec![]); 3]
    );

    assert_eq!(world.flush_reactions(), 3);
    assert_eq!(
        world.resource::<Output>().batches,
        vec![vec![first], vec![second]]
    );
    assert_eq!(
        world.resource::<Attempts>().0.last(),
        Some(&(vec![second], vec![sparse_entity]))
    );
    assert_eq!(world.resource::<Output>().runs, 2);
    assert_eq!(world.flush_reactions(), 0);
    assert_eq!(world.flush_changed_observers(), 0);
    assert_eq!(world.resource::<Attempts>().0.len(), 4);
}

fn populate_reaction_replacement(world: &mut World) -> ReactionId {
    world.insert_resource(Output::default());
    let source = world.reaction_source::<Input>();
    world.add_reaction(&[source], |world| world.resource_mut::<Output>().runs += 1);
    world.mark_reaction_dirty(source);
    source
}

fn reaction_replacement_case(populated: bool, panic_after: bool, flush: impl FnOnce(&mut World)) {
    let mut replacement = World::new();
    let mut replacement_source = populated.then(|| populate_reaction_replacement(&mut replacement));
    let replacement_id = replacement.id();
    let replacement_tick = replacement.change_tick();
    let mut replacement = Some(replacement);
    let mut world = World::new();
    let source = world.reaction_source::<Input>();
    let replace = world.add_reaction(&[source], move |world| {
        *world = replacement.take().expect("old effect must not be restored");
        assert!(!panic_after, "effect replaced world");
    });
    world.add_reaction(&[replace], |_| panic!("old downstream effect ran"));
    world.mark_reaction_dirty(source);
    flush(&mut world);
    assert_eq!(world.id(), replacement_id);
    assert_eq!(world.change_tick(), replacement_tick);
    if populated {
        assert_eq!(world.resource::<Output>().runs, 0);
    } else {
        assert_eq!(world.flush_reactions(), 0);
        assert_eq!(world.flush_changed_observers(), 0);
        replacement_source = Some(populate_reaction_replacement(&mut world));
    }
    let replacement_source = replacement_source.unwrap();
    assert_eq!(world.flush_reactions(), 1);
    assert_eq!(world.resource::<Output>().runs, 1);
    assert_eq!(world.flush_reactions(), 0);
    world.mark_reaction_dirty(replacement_source);
    assert_eq!(world.flush_reactions(), 1);
    assert_eq!(world.resource::<Output>().runs, 2);
    // The replacement's component source observer must remain live too.
    world.spawn(Input(1));
    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(world.resource::<Output>().runs, 3);
}

#[test]
fn replacing_world_stops_old_effects_and_preserves_the_replacement_graph() {
    for populated in [false, true] {
        reaction_replacement_case(populated, false, |world| {
            assert_eq!(world.flush_reactions(), 1);
        });
    }
}

#[cfg(feature = "std")]
#[test]
fn replacing_world_then_panicking_does_not_restore_the_old_reaction_graph() {
    for populated in [false, true] {
        reaction_replacement_case(populated, true, |world| {
            let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                world.flush_reactions();
            }))
            .expect_err("replacement effect must panic");
            assert_eq!(error.downcast_ref::<&str>(), Some(&"effect replaced world"));
        });
    }
}

#[test]
fn an_observer_cannot_settle_effects_before_all_sources_are_delivered() {
    let mut world = World::new();
    let entity = world.spawn((Input(1), SparseInput(2)));
    world.insert_resource(Output::default());
    let first = world.reaction_source::<Input>();
    world.observe_changed::<Input, _>(|ctx| {
        assert_eq!(ctx.world.flush_reactions(), 0);
    });
    let second = world.reaction_source::<SparseInput>();
    world.add_reaction(&[first, second], move |world| {
        assert_eq!(world.reaction_changes(first), &[entity]);
        assert_eq!(world.reaction_changes(second), &[entity]);
        world.resource_mut::<Output>().runs += 1;
    });
    world.get_mut::<Input>(entity).unwrap().0 += 1;
    world.get_mut::<SparseInput>(entity).unwrap().0 += 1;
    world.flush_changed_observers();
    assert_eq!(world.resource::<Output>().runs, 1);
}
