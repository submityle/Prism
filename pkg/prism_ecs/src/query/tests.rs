//! Behavioural tests for the query system: read-only iteration, mutable
//! iteration, archetype filters (`With`/`Without`/`Or`), optional terms and
//! tuple/`Entity` data terms. These exercise the full
//! `World::query*` → `QueryState::iter*` → `QueryIter` path end to end.

use alloc::vec::Vec;

use crate::component::Component;
use crate::entity::Entity;
use crate::query::{Or, With, Without};
use crate::world::World;

#[derive(Debug, PartialEq, Clone, Copy)]
struct Position(i32, i32);
impl Component for Position {}

#[derive(Debug, PartialEq, Clone, Copy)]
struct Velocity(i32, i32);
impl Component for Velocity {}

#[derive(Debug, PartialEq, Clone, Copy)]
struct Tag;
impl Component for Tag {}

#[test]
fn iter_reads_every_matching_row() {
    let mut w = World::new();
    w.spawn(Position(1, 1));
    w.spawn((Position(2, 2), Velocity(0, 0)));
    w.spawn(Velocity(9, 9)); // no Position -> excluded

    let state = w.query::<&Position>();
    let mut seen: Vec<Position> = state.iter(&w).copied().collect();
    seen.sort_by_key(|p| p.0);
    assert_eq!(seen, [Position(1, 1), Position(2, 2)]);
}

#[test]
fn iter_mut_mutates_in_place() {
    let mut w = World::new();
    let a = w.spawn(Position(1, 2));
    let b = w.spawn((Position(3, 4), Velocity(10, 20)));

    let state = w.query::<&mut Position>();
    for mut p in state.iter_mut(&mut w) {
        p.0 += 100;
        p.1 += 1;
    }

    assert_eq!(w.get::<Position>(a), Some(&Position(101, 3)));
    assert_eq!(w.get::<Position>(b), Some(&Position(103, 5)));
}

#[test]
fn with_filter_only_matches_archetypes_having_component() {
    let mut w = World::new();
    w.spawn(Position(1, 0));
    w.spawn((Position(2, 0), Velocity(0, 0)));
    w.spawn((Position(3, 0), Velocity(0, 0)));

    let state = w.query_filtered::<&Position, With<Velocity>>();
    let mut xs: Vec<i32> = state.iter(&w).map(|p| p.0).collect();
    xs.sort_unstable();
    assert_eq!(xs, [2, 3]);
}

#[test]
fn without_filter_excludes_archetypes_having_component() {
    let mut w = World::new();
    w.spawn(Position(1, 0));
    w.spawn((Position(2, 0), Velocity(0, 0)));
    w.spawn(Position(3, 0));

    let state = w.query_filtered::<&Position, Without<Velocity>>();
    let mut xs: Vec<i32> = state.iter(&w).map(|p| p.0).collect();
    xs.sort_unstable();
    assert_eq!(xs, [1, 3]);
}

#[test]
fn or_filter_matches_either_branch() {
    let mut w = World::new();
    w.spawn((Position(1, 0), Velocity(0, 0))); // matches With<Velocity>
    w.spawn((Position(2, 0), Tag)); // matches With<Tag>
    w.spawn(Position(3, 0)); // matches neither -> excluded

    let state = w.query_filtered::<&Position, Or<(With<Velocity>, With<Tag>)>>();
    let mut xs: Vec<i32> = state.iter(&w).map(|p| p.0).collect();
    xs.sort_unstable();
    assert_eq!(xs, [1, 2]);
}

#[test]
fn optional_term_yields_none_and_some() {
    let mut w = World::new();
    w.spawn(Position(1, 0));
    w.spawn((Position(2, 0), Velocity(7, 7)));

    let state = w.query::<(&Position, Option<&Velocity>)>();
    let mut rows: Vec<(i32, Option<Velocity>)> =
        state.iter(&w).map(|(p, v)| (p.0, v.copied())).collect();
    rows.sort_by_key(|(x, _)| *x);
    assert_eq!(rows, [(1, None), (2, Some(Velocity(7, 7)))]);
}

#[test]
fn tuple_with_entity_term_pairs_handle_and_data() {
    let mut w = World::new();
    let a = w.spawn(Position(10, 0));
    let b = w.spawn(Position(20, 0));

    let state = w.query::<(Entity, &Position)>();
    let mut rows: Vec<(Entity, i32)> = state.iter(&w).map(|(e, p)| (e, p.0)).collect();
    rows.sort_by_key(|(_, x)| *x);
    assert_eq!(rows, [(a, 10), (b, 20)]);
}

#[test]
fn for_each_convenience_reads_all() {
    let mut w = World::new();
    w.spawn(Position(5, 0));
    w.spawn(Position(6, 0));

    let mut sum = 0;
    w.for_each::<&Position>(|p| sum += p.0);
    assert_eq!(sum, 11);
}

#[test]
fn fold_fast_path_matches_next_order_and_visits() {
    let mut w = World::new();
    w.spawn(Position(1, 0));
    w.spawn(Velocity(9, 9)); // non-matching archetype is skipped
    w.spawn(Position(2, 0));
    w.spawn((Position(3, 0), Velocity(1, 1)));

    let state = w.query::<&Position>();

    // The `next()`-based for-loop fixes the reference order (archetype order,
    // then row order).
    let mut via_next = Vec::new();
    for p in state.iter(&w) {
        via_next.push(p.0);
    }

    // The batched `fold` path reached three ways: an explicit fold, the
    // iterator's `for_each`, and the `QueryState` convenience.
    let via_fold_sum = state.iter(&w).fold(0, |acc, p| acc + p.0);
    let mut via_foreach = Vec::new();
    state.iter(&w).for_each(|p| via_foreach.push(p.0));
    let mut via_state = Vec::new();
    state.for_each(&w, |p| via_state.push(p.0));
    let via_count = state.iter(&w).count();

    assert_eq!(via_next, [1, 2, 3]);
    assert_eq!(via_foreach, via_next);
    assert_eq!(via_state, via_next);
    assert_eq!(via_fold_sum, 6);
    assert_eq!(via_count, 3);
}

#[test]
fn fold_fast_path_mutates_every_row() {
    let mut w = World::new();
    let a = w.spawn(Position(1, 0));
    let b = w.spawn((Position(2, 0), Velocity(0, 0)));
    w.spawn(Velocity(0, 0)); // no Position -> excluded from the mutation

    let state = w.query::<&mut Position>();
    state.iter_mut(&mut w).for_each(|mut p| p.0 += 10);

    // A `next()`-based pass must observe exactly the fold's writes.
    assert_eq!(w.get::<Position>(a), Some(&Position(11, 0)));
    assert_eq!(w.get::<Position>(b), Some(&Position(12, 0)));
}

#[test]
fn for_each_mut_convenience_mutates_all() {
    let mut w = World::new();
    let a = w.spawn(Position(1, 0));
    let b = w.spawn((Position(2, 0), Velocity(0, 0)));

    let state = w.query::<&mut Position>();
    state.for_each_mut(&mut w, |mut p| p.0 += 5);

    assert_eq!(w.get::<Position>(a), Some(&Position(6, 0)));
    assert_eq!(w.get::<Position>(b), Some(&Position(7, 0)));
}

#[test]
fn iter_is_empty_when_no_archetype_matches() {
    let mut w = World::new();
    w.spawn(Velocity(1, 1));
    let state = w.query::<&Position>();
    assert_eq!(state.iter(&w).count(), 0);
}

#[test]
fn state_is_reusable_across_iterations() {
    let mut w = World::new();
    w.spawn(Position(1, 0));
    let state = w.query::<&Position>();
    assert_eq!(state.iter(&w).count(), 1);
    // Reuse the same state after mutating the world structurally.
    w.spawn(Position(2, 0));
    assert_eq!(state.iter(&w).count(), 2);
}

#[test]
#[should_panic(expected = "borrowed mutably more than once")]
fn self_conflicting_double_mut_panics() {
    let mut w = World::new();
    w.spawn(Position(1, 0));
    // Two `&mut Position` terms alias the same column mutably; the access set
    // must reject this at state construction.
    let _state = w.query::<(&mut Position, &mut Position)>();
}

#[test]
#[should_panic(expected = "borrowed both mutably and immutably")]
fn self_conflicting_read_and_write_panics() {
    let mut w = World::new();
    w.spawn(Position(1, 0));
    // `&Position` aliases the `&mut Position` write of the same query.
    let _state = w.query::<(&mut Position, &Position)>();
}

/// The incremental archetype-match cache (design §7) must observe archetypes
/// created *after* a query state's first iteration: reusing one state, a spawn
/// into a brand-new matching archetype shows up on the next iteration without
/// the state being rebuilt.
#[test]
fn incremental_cache_sees_archetypes_created_after_first_iter() {
    let mut w = World::new();
    w.spawn(Position(1, 0));

    let state = w.query::<&Position>();
    assert_eq!(state.iter(&w).count(), 1);

    // New archetype {Position, Velocity} did not exist at the first refresh.
    w.spawn((Position(2, 0), Velocity(0, 0)));
    w.spawn((Position(3, 0), Velocity(0, 0)));
    assert_eq!(
        state.iter(&w).count(),
        3,
        "cache picks up the new archetype"
    );

    // Another new archetype {Position, Tag}.
    w.spawn((Position(4, 0), Tag));
    let mut xs: Vec<i32> = state.iter(&w).map(|p| p.0).collect();
    xs.sort_unstable();
    assert_eq!(xs, [1, 2, 3, 4]);
}

/// A filtered query's cache must *exclude* newly-created archetypes that do not
/// match, proving the per-id match decision is applied to new archetypes too.
#[test]
fn incremental_cache_excludes_new_non_matching_archetypes() {
    let mut w = World::new();
    w.spawn((Position(1, 0), Velocity(0, 0)));

    let state = w.query_filtered::<&Position, Without<Velocity>>();
    assert_eq!(state.iter(&w).count(), 0, "only match has Velocity");

    // New archetype {Position} matches `Without<Velocity>`...
    w.spawn(Position(2, 0));
    // ...new archetype {Position, Velocity, Tag} does not.
    w.spawn((Position(3, 0), Velocity(0, 0), Tag));
    let xs: Vec<i32> = state.iter(&w).map(|p| p.0).collect();
    assert_eq!(xs, [2], "only the Velocity-free archetype is cached");
}

/// Repeated iterations with no structural change return identical results (the
/// cache short-circuits once every archetype has been tested).
#[test]
fn incremental_cache_is_stable_across_repeated_iters() {
    let mut w = World::new();
    w.spawn(Position(1, 0));
    w.spawn((Position(2, 0), Velocity(0, 0)));

    let state = w.query::<&Position>();
    let first = state.iter(&w).count();
    for _ in 0..8 {
        assert_eq!(state.iter(&w).count(), first);
    }
    assert_eq!(first, 2);
}

#[test]
fn state_cardinality_helpers_agree_with_iteration() {
    let mut w = World::new();
    let a = w.spawn(Position(1, 1));
    let b = w.spawn((Position(2, 2), Velocity(0, 0)));

    let state = w.query::<&Position>();
    assert_eq!(state.count(&w), 2);
    assert!(!state.is_empty(&w));
    assert!(state.contains(&w, a));
    assert!(state.contains(&w, b));

    // A live entity that misses the data term fails `contains`.
    let other = w.spawn(Velocity(1, 1));
    assert!(!state.contains(&w, other));
    // A despawned entity fails it too.
    assert!(w.despawn(a));
    assert!(!state.contains(&w, a));
    assert_eq!(state.count(&w), 1);

    let mut heard = 0usize;
    state.for_each(&w, |_p| heard += 1);
    assert_eq!(heard, 1);

    // An empty world matches nothing.
    let mut empty = World::new();
    let empty_state = empty.query::<&Position>();
    assert_eq!(empty_state.count(&empty), 0);
    assert!(empty_state.is_empty(&empty));
}

#[test]
fn query_iter_size_hint_bounds_the_remaining_rows() {
    let mut w = World::new();
    w.spawn(Position(1, 1));
    w.spawn((Position(2, 2), Velocity(0, 0)));
    w.spawn(Velocity(9, 9)); // excluded archetype

    let state = w.query::<&Position>();
    let mut iter = state.iter(&w);
    assert_eq!(iter.size_hint(), (0, Some(2)));
    assert!(iter.next().is_some());
    // One row consumed: the upper bound drops to the remaining one.
    assert_eq!(iter.size_hint(), (0, Some(1)));
    assert!(iter.next().is_some());
    assert_eq!(iter.size_hint(), (0, Some(0)));
    assert!(iter.next().is_none());
}
