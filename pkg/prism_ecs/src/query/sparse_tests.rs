//! Behavioural tests for **SparseSet-backed** components in the query system
//! (design §6, §7).
//!
//! Unlike table components, a sparse component is routed out of the archetype
//! and keyed by [`Entity`] in the world's `SparseSets` registry. Membership is
//! therefore *per entity within an archetype*, not encoded in the archetype's
//! component set. These tests pin that behaviour across every query term:
//! required reads/writes, mixed table+sparse tuples, `Option<&T>`, the
//! `With`/`Without` membership filters, and the `Added`/`Changed` change
//! filters — including two entities that share one archetype but differ in
//! whether they carry the sparse component, and the degenerate case where no
//! sparse set has been allocated yet.

use alloc::vec::Vec;

use crate::component::{Component, StorageType};
use crate::entity::Entity;
use crate::query::{Added, Changed, With, Without};
use crate::world::World;

/// Table-backed (default) component used as the archetype anchor.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Position(i32);
impl Component for Position {}

/// Sparse-backed component: routed out of the archetype, keyed by entity.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Charge(i32);
impl Component for Charge {
    const STORAGE: StorageType = StorageType::SparseSet;
}

/// A second sparse component, so a sparse set can exist without the component
/// under test ever having been inserted.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Flagged;
impl Component for Flagged {
    const STORAGE: StorageType = StorageType::SparseSet;
}

/// Advance the world as if a system had just finished a run: the next window
/// starts where this one ended, and the frame tick advances by one.
fn advance_frame(w: &mut World) {
    w.set_last_change_tick(w.change_tick());
    w.increment_change_tick();
}

#[test]
fn required_sparse_term_yields_only_entities_that_have_it() {
    let mut w = World::new();
    // Both entities land in the same archetype {Position}; `Charge` is stored
    // out of band, so only `a` carries it.
    let a = w.spawn((Position(1), Charge(10)));
    let _b = w.spawn(Position(2));

    let state = w.query::<(Entity, &Charge)>();
    let seen: Vec<(Entity, Charge)> = state.iter(&w).map(|(e, c)| (e, *c)).collect();
    assert_eq!(seen, [(a, Charge(10))]);
}

#[test]
fn mut_sparse_mutates_in_place_and_stamps_changed() {
    let mut w = World::new();
    let a = w.spawn((Position(1), Charge(5)));
    let _b = w.spawn(Position(2)); // no Charge

    // Move past the spawn frame so "added" no longer dominates the window.
    advance_frame(&mut w);

    let state = w.query::<&mut Charge>();
    let mut count = 0;
    for mut c in state.iter_mut(&mut w) {
        c.0 += 100;
        count += 1;
    }
    assert_eq!(
        count, 1,
        "only the entity with the sparse component is visited"
    );
    assert_eq!(w.get::<Charge>(a), Some(&Charge(105)));

    // The DerefMut stamped changed = this_run, so `Changed<Charge>` sees it.
    let state = w.query_filtered::<Entity, Changed<Charge>>();
    assert_eq!(state.iter(&w).collect::<Vec<_>>(), [a]);
}

#[test]
fn mixed_tuple_requires_both_table_and_sparse() {
    let mut w = World::new();
    let both = w.spawn((Position(1), Charge(9)));
    let _table_only = w.spawn(Position(2));
    let _sparse_only = w.spawn(Charge(3)); // archetype {} + sparse Charge

    let state = w.query::<(Entity, &Position, &Charge)>();
    let seen: Vec<Entity> = state.iter(&w).map(|(e, _, _)| e).collect();
    assert_eq!(seen, [both], "only entities with BOTH terms are yielded");
}

#[test]
fn option_sparse_reports_presence_per_entity() {
    let mut w = World::new();
    let a = w.spawn((Position(1), Charge(7)));
    let b = w.spawn(Position(2));

    let state = w.query::<(Entity, Option<&Charge>)>();
    let mut seen: Vec<(Entity, Option<i32>)> =
        state.iter(&w).map(|(e, c)| (e, c.map(|c| c.0))).collect();
    seen.sort_by_key(|(_, c)| c.is_some());
    assert_eq!(seen, [(b, None), (a, Some(7))]);
}

#[test]
fn with_and_without_resolve_sparse_membership_per_entity() {
    let mut w = World::new();
    let a = w.spawn((Position(1), Charge(1)));
    let b = w.spawn(Position(2));

    let with = w.query_filtered::<Entity, With<Charge>>();
    assert_eq!(with.iter(&w).collect::<Vec<_>>(), [a]);

    let without = w.query_filtered::<Entity, Without<Charge>>();
    assert_eq!(without.iter(&w).collect::<Vec<_>>(), [b]);
}

#[test]
fn toggling_sparse_in_a_shared_archetype_updates_membership() {
    let mut w = World::new();
    let a = w.spawn((Position(1), Charge(1)));
    let b = w.spawn((Position(2), Charge(2)));

    // Remove the sparse component from `a` only — no archetype move happens, so
    // `a` and `b` still share archetype {Position}.
    assert!(w.remove::<Charge>(a));

    let state = w.query::<(Entity, &Charge)>();
    let seen: Vec<Entity> = state.iter(&w).map(|(e, _)| e).collect();
    assert_eq!(seen, [b], "the still-charged entity is the only match");

    // Re-add to `a`: it reappears.
    assert!(w.insert(a, Charge(42)));
    let state = w.query::<(Entity, &Charge)>();
    let mut seen: Vec<Entity> = state.iter(&w).map(|(e, _)| e).collect();
    seen.sort();
    let mut expected = alloc::vec![a, b];
    expected.sort();
    assert_eq!(seen, expected);
}

#[test]
fn added_sparse_matches_spawn_frame_then_goes_stale() {
    let mut w = World::new();
    let a = w.spawn((Position(1), Charge(1)));

    let state = w.query_filtered::<Entity, Added<Charge>>();
    assert_eq!(
        state.iter(&w).collect::<Vec<_>>(),
        [a],
        "freshly inserted sparse component is Added relative to last_run = ZERO"
    );

    advance_frame(&mut w);
    let state = w.query_filtered::<Entity, Added<Charge>>();
    assert_eq!(
        state.iter(&w).count(),
        0,
        "a sparse component added in a previous frame is no longer Added"
    );
}

#[test]
fn changed_sparse_fires_on_mut_then_goes_stale() {
    let mut w = World::new();
    let a = w.spawn((Position(1), Charge(1)));

    // Move past the spawn/added frame.
    advance_frame(&mut w);

    let state = w.query::<&mut Charge>();
    for mut c in state.iter_mut(&mut w) {
        c.0 += 1;
    }

    let state = w.query_filtered::<Entity, Changed<Charge>>();
    assert_eq!(
        state.iter(&w).collect::<Vec<_>>(),
        [a],
        "the written row is Changed"
    );

    advance_frame(&mut w);
    let state = w.query_filtered::<Entity, Changed<Charge>>();
    assert_eq!(
        state.iter(&w).count(),
        0,
        "no new write this frame, so the row is stale"
    );
}

#[test]
fn missing_sparse_set_yields_nothing_for_required_and_all_none_for_option() {
    let mut w = World::new();
    // Entities exist and a *different* sparse set is allocated, but `Charge`'s
    // set is never created (no `Charge` is ever inserted).
    let a = w.spawn((Position(1), Flagged));
    let b = w.spawn(Position(2));

    let required = w.query::<(Entity, &Charge)>();
    assert_eq!(
        required.iter(&w).count(),
        0,
        "required sparse term matches nothing when its set was never allocated"
    );

    let state = w.query::<(Entity, Option<&Charge>)>();
    let mut seen: Vec<(Entity, Option<i32>)> =
        state.iter(&w).map(|(e, c)| (e, c.map(|c| c.0))).collect();
    seen.sort_by_key(|(e, _)| *e);
    let mut expected = alloc::vec![(a, None), (b, None)];
    expected.sort_by_key(|(e, _)| *e);
    assert_eq!(seen, expected, "optional sparse term is None for every row");
}
