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
    for p in state.iter_mut(&mut w) {
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
