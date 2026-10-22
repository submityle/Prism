//! Ordered multi-source candidate merging and overlap regressions.

use super::*;
use alloc::vec;

macro_rules! components {
    ($($name:ident),+ $(,)?) => {$ (
        struct $name;
        impl Component for $name {}
    )+ };
}

components!(A, B, C, D, E, F);

fn register<T: Component>(world: &mut World) -> usize {
    world.observe_changed::<T, _>(|_| {});
    world
        .change_observers
        .active_indices
        .last()
        .copied()
        .expect("registration appends an active slot")
}

fn assert_candidate_merge(dirty_source_count: usize, include_pending: bool) {
    let mut world = World::new();
    let registrations = [
        (
            register::<A>(&mut world),
            world.components().id_of::<A>().unwrap(),
        ),
        (
            register::<B>(&mut world),
            world.components().id_of::<B>().unwrap(),
        ),
        (
            register::<A>(&mut world),
            world.components().id_of::<A>().unwrap(),
        ),
        (
            register::<C>(&mut world),
            world.components().id_of::<C>().unwrap(),
        ),
        (
            register::<D>(&mut world),
            world.components().id_of::<D>().unwrap(),
        ),
        (
            register::<E>(&mut world),
            world.components().id_of::<E>().unwrap(),
        ),
        (
            register::<B>(&mut world),
            world.components().id_of::<B>().unwrap(),
        ),
        (
            register::<F>(&mut world),
            world.components().id_of::<F>().unwrap(),
        ),
        (
            register::<C>(&mut world),
            world.components().id_of::<C>().unwrap(),
        ),
        (
            register::<D>(&mut world),
            world.components().id_of::<D>().unwrap(),
        ),
        (
            register::<E>(&mut world),
            world.components().id_of::<E>().unwrap(),
        ),
        (
            register::<A>(&mut world),
            world.components().id_of::<A>().unwrap(),
        ),
    ];
    let pending = registrations
        .iter()
        .find_map(|&(index, component)| {
            (component == world.components().id_of::<F>().unwrap()).then_some(index)
        })
        .expect("pending component has a registered observer");
    if include_pending {
        world.change_observers.mark_pending(pending);
    }
    // Also overlap pending with an active dirty source. The merge must emit
    // the slot once, while retaining its global registration-order position.
    if include_pending {
        world.change_observers.mark_pending(registrations[0].0);
    }

    let mut dirty = vec![
        world.components().id_of::<A>().unwrap(),
        world.components().id_of::<B>().unwrap(),
        world.components().id_of::<C>().unwrap(),
        world.components().id_of::<D>().unwrap(),
        world.components().id_of::<E>().unwrap(),
    ];
    dirty.truncate(dirty_source_count);
    let actual = world.change_observers.active_candidates(&dirty);
    let expected: Vec<_> = registrations
        .iter()
        .filter_map(|&(index, component)| {
            (dirty.contains(&component)
                || (include_pending && (index == pending || index == registrations[0].0)))
                .then_some(index)
        })
        .collect();
    assert_eq!(actual, expected);
    assert!(actual.windows(2).all(|pair| {
        world.change_observers.slots[pair[0]].registration_order
            < world.change_observers.slots[pair[1]].registration_order
    }));
}

#[test]
fn small_source_merge_preserves_order_and_deduplicates_pending_overlap() {
    // Pending contributes one source, so three dirty sources exercises the
    // four-source stack maximum with overlap. Four dirty sources also exercise
    // that boundary without pending work.
    for dirty_source_count in 1..=3 {
        assert_candidate_merge(dirty_source_count, true);
    }
    assert_candidate_merge(4, false);
}

#[test]
fn heap_merge_preserves_order_and_deduplicates_pending_overlap() {
    assert_candidate_merge(4, true);
}
