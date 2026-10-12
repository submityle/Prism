//! Behavioural tests for batched random access over a caller-provided entity
//! list ([`QueryState::get_many`] / [`QueryState::get_many_mut`] /
//! [`QueryState::iter_many`], design §7).
//!
//! `get_many`/`get_many_mut` resolve a fixed array of ids in order and
//! short-circuit on the first failure; the mutable form additionally rejects
//! duplicate ids (which would alias). `iter_many` is the streaming dual and
//! surfaces one `Result` per id, with a `matched()` adapter to skip failures.

use alloc::vec::Vec;

use crate::component::{Component, StorageType};
use crate::query::{Changed, QueryEntityError, With};
use crate::world::World;

/// Table-backed position.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Pos(i32);
impl Component for Pos {}

/// Table-backed velocity.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Vel(i32);
impl Component for Vel {}

/// Sparse-backed buff, routed out of the archetype (design §6).
#[derive(Debug, PartialEq, Clone, Copy)]
struct Buff(i32);
impl Component for Buff {
    const STORAGE: StorageType = StorageType::SparseSet;
}

/// Advance the world one change-detection frame.
fn next_frame(w: &mut World) {
    w.set_last_change_tick(w.change_tick());
    w.increment_change_tick();
}

#[test]
fn get_many_returns_items_in_list_order() {
    let mut w = World::new();
    let a = w.spawn(Pos(1));
    let b = w.spawn(Pos(2));
    let c = w.spawn(Pos(3));

    let state = w.query::<&Pos>();
    let got = state.get_many(&w, [c, a, b]).expect("all match");
    assert_eq!([got[0].0, got[1].0, got[2].0], [3, 1, 2]);
}

#[test]
fn get_many_short_circuits_on_the_first_failure() {
    let mut w = World::new();
    let a = w.spawn(Pos(1));
    let dead = w.spawn(Pos(2));
    assert!(w.despawn(dead));
    let no_pos = w.spawn(Vel(9));

    let state = w.query::<&Pos>();
    assert_eq!(
        state.get_many(&w, [a, dead, no_pos]),
        Err(QueryEntityError::NoSuchEntity(dead))
    );
    assert_eq!(
        state.get_many(&w, [a, no_pos]),
        Err(QueryEntityError::QueryDoesNotMatch(no_pos))
    );
    // The earlier id's failure wins over the later id's.
    assert_eq!(
        state.get_many(&w, [dead, no_pos]),
        Err(QueryEntityError::NoSuchEntity(dead))
    );
}

#[test]
fn get_many_applies_per_row_gates_per_entity() {
    let mut w = World::new();
    let a = w.spawn(Pos(1));
    let buffed = w.spawn((Pos(2), Buff(5)));
    let state = w.query::<&Buff>();

    assert_eq!(state.get_many(&w, [buffed]).map(|bs| bs[0].0), Ok(5));
    // `a` passes the archetype gate (sparse terms match any archetype) but
    // fails the per-row sparse-presence gate.
    assert_eq!(
        state.get_many(&w, [buffed, a]),
        Err(QueryEntityError::QueryDoesNotMatch(a))
    );

    // The per-row `Changed` gate is likewise evaluated per entity.
    let b = w.spawn(Pos(3));
    next_frame(&mut w);
    w.get_mut::<Pos>(a).expect("live").0 += 1;
    let changed = w.query_filtered::<&Pos, Changed<Pos>>();
    assert_eq!(changed.get_many(&w, [a]).map(|ps| ps[0].0), Ok(2));
    assert_eq!(
        changed.get_many(&w, [a, b]),
        Err(QueryEntityError::QueryDoesNotMatch(b))
    );
}

#[test]
fn get_many_mut_writes_every_entity() {
    let mut w = World::new();
    let a = w.spawn(Pos(1));
    let b = w.spawn(Pos(2));

    let state = w.query::<&mut Pos>();
    {
        let mut items = state
            .get_many_mut(&mut w, [a, b])
            .expect("distinct live matches");
        items[0].0 += 10;
        items[1].0 += 20;
    }
    assert_eq!(w.get::<Pos>(a).map(|p| p.0), Some(11));
    assert_eq!(w.get::<Pos>(b).map(|p| p.0), Some(22));
}

#[test]
fn get_many_mut_rejects_duplicate_ids_before_writing() {
    let mut w = World::new();
    let a = w.spawn(Pos(1));
    let b = w.spawn(Pos(2));

    let state = w.query::<&mut Pos>();
    assert!(matches!(
        state.get_many_mut(&mut w, [a, b, a]),
        Err(QueryEntityError::AliasedMutability(e)) if e == a
    ));
    // The guard fires up front: no value was touched.
    assert_eq!(w.get::<Pos>(a).map(|p| p.0), Some(1));
    assert_eq!(w.get::<Pos>(b).map(|p| p.0), Some(2));
}

#[test]
fn iter_many_yields_in_list_order_and_reports_errors() {
    let mut w = World::new();
    let a = w.spawn(Pos(1));
    let dead = w.spawn(Pos(2));
    assert!(w.despawn(dead));
    let c = w.spawn(Pos(3));
    let no_pos = w.spawn(Vel(9));

    let state = w.query::<&Pos>();
    let results: Vec<_> = state.iter_many(&w, [a, dead, c, no_pos]).collect();
    assert_eq!(results.len(), 4);
    assert_eq!(results[0].map(|p| p.0), Ok(1));
    assert_eq!(results[1], Err(QueryEntityError::NoSuchEntity(dead)));
    assert_eq!(results[2].map(|p| p.0), Ok(3));
    assert_eq!(
        results[3],
        Err(QueryEntityError::QueryDoesNotMatch(no_pos))
    );
}

#[test]
fn iter_many_matched_skips_non_matching_entities() {
    let mut w = World::new();
    let a = w.spawn(Pos(1));
    let dead = w.spawn(Pos(2));
    assert!(w.despawn(dead));
    let c = w.spawn(Pos(3));
    let no_pos = w.spawn(Vel(9));

    let state = w.query::<&Pos>();
    let kept: Vec<i32> = state
        .iter_many(&w, [a, dead, c, no_pos])
        .matched()
        .map(|p| p.0)
        .collect();
    assert_eq!(kept, [1, 3]);
}

#[test]
fn iter_many_honours_archetype_filters() {
    let mut w = World::new();
    let moving = w.spawn((Pos(1), Vel(2)));
    let still = w.spawn(Pos(3));

    let state = w.query_filtered::<&Pos, With<Vel>>();
    let results: Vec<_> = state.iter_many(&w, [still, moving]).collect();
    assert_eq!(
        results[0],
        Err(QueryEntityError::QueryDoesNotMatch(still))
    );
    assert_eq!(results[1].map(|p| p.0), Ok(1));

    let kept: Vec<i32> = state
        .iter_many(&w, [still, moving])
        .matched()
        .map(|p| p.0)
        .collect();
    assert_eq!(kept, [1]);
}

#[test]
fn iter_many_size_hint_is_exact() {
    let mut w = World::new();
    let a = w.spawn(Pos(1));
    let b = w.spawn(Vel(2)); // fails the query, but still yields one `Err`

    let state = w.query::<&Pos>();
    let iter = state.iter_many(&w, [a, b]);
    assert_eq!(iter.size_hint(), (2, Some(2)));
    assert_eq!(iter.count(), 2);
}
