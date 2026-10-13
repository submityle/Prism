//! Behavioural tests for the cardinality-constrained single-entity query
//! accessors ([`QueryState::single`] / [`QueryState::single_mut`], design §7).
//!
//! `single`/`single_mut` assert "exactly one row matches": they take the first
//! row and reject a second, returning [`QuerySingleError`] otherwise. These
//! tests pin the success path, both error cases, filter interaction
//! (`With`/`Without`/`Changed`), sparse per-row resolution, write-through, and
//! parity with the cardinality an explicit iteration would observe.

use alloc::format;

use crate::component::{Component, StorageType};
use crate::query::{Changed, QuerySingleError, With, Without};
use crate::world::World;

/// Table-backed position.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Pos(i32);
impl Component for Pos {}

/// Table-backed velocity.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Vel(i32);
impl Component for Vel {}

/// Table-backed marker used to force a distinct archetype.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Frozen;
impl Component for Frozen {}

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
fn single_returns_the_only_match() {
    let mut w = World::new();
    w.spawn(Pos(7));
    let state = w.query::<&Pos>();
    assert_eq!(state.single(&w).map(|p| p.0), Ok(7));
}

#[test]
fn single_errors_with_no_entities() {
    let mut w = World::new();
    // The world is non-empty, but nothing carries `Pos`.
    w.spawn(Vel(1));
    let state = w.query::<&Pos>();
    assert!(matches!(
        state.single(&w),
        Err(QuerySingleError::NoEntities(_))
    ));
}

#[test]
fn single_errors_with_multiple_entities() {
    let mut w = World::new();
    w.spawn(Pos(1));
    w.spawn(Pos(2));
    let state = w.query::<&Pos>();
    assert!(matches!(
        state.single(&w),
        Err(QuerySingleError::MultipleEntities(_))
    ));
}

#[test]
fn single_ignores_non_matching_archetypes() {
    let mut w = World::new();
    for i in 0..8 {
        w.spawn(Vel(i)); // no `Pos`: a different archetype
    }
    w.spawn(Pos(42));
    let state = w.query::<&Pos>();
    assert_eq!(state.single(&w).map(|p| p.0), Ok(42));
}

#[test]
fn single_honours_with_and_without_filters() {
    let mut w = World::new();
    w.spawn((Pos(1), Vel(2))); // matches `With<Vel>`, not `Without<Frozen>`
    w.spawn((Pos(9), Frozen)); // excluded by `With<Vel>`

    let with_vel = w.query_filtered::<&Pos, With<Vel>>();
    assert_eq!(with_vel.single(&w).map(|p| p.0), Ok(1));

    let without_frozen = w.query_filtered::<&Pos, Without<Frozen>>();
    assert_eq!(without_frozen.single(&w).map(|p| p.0), Ok(1));

    // A second, frozen `Vel` participant makes `with_vel` ambiguous while
    // `Without<Frozen>` stays unique.
    w.spawn((Pos(5), Vel(6), Frozen));
    assert!(matches!(
        with_vel.single(&w),
        Err(QuerySingleError::MultipleEntities(_))
    ));
    assert_eq!(without_frozen.single(&w).map(|p| p.0), Ok(1));
}

#[test]
fn single_mut_writes_back_through_the_fetched_reference() {
    let mut w = World::new();
    let e = w.spawn(Pos(1));

    let state = w.query::<&mut Pos>();
    {
        let mut p = state.single_mut(&mut w).expect("exactly one match");
        p.0 += 41;
    }
    assert_eq!(w.get::<Pos>(e).map(|p| p.0), Some(42));
}

#[test]
fn single_applies_the_per_row_changed_gate() {
    let mut w = World::new();
    let a = w.spawn(Pos(1));
    let b = w.spawn(Pos(2));
    // Advance a frame so both spawn ticks are in the past.
    next_frame(&mut w);

    // Touch only `a` this frame: the per-row gate admits exactly one row.
    w.get_mut::<Pos>(a).expect("live").0 += 1;
    let changed = w.query_filtered::<&Pos, Changed<Pos>>();
    assert_eq!(changed.single(&w).map(|p| p.0), Ok(2));

    // Touch `b` as well: two rows now satisfy `Changed<Pos>`.
    w.get_mut::<Pos>(b).expect("live").0 += 1;
    assert!(matches!(
        changed.single(&w),
        Err(QuerySingleError::MultipleEntities(_))
    ));
}

#[test]
fn single_resolves_sparse_backed_terms_per_entity() {
    let mut w = World::new();
    w.spawn(Pos(1)); // same archetype as the buffed rows, no `Buff`
    w.spawn((Pos(2), Buff(7)));
    w.spawn(Pos(3)); // no `Buff`

    let state = w.query::<&Buff>();
    // Only the buffed row passes the sparse per-row gate, so the query is
    // unique even though three rows share its archetype.
    assert_eq!(state.single(&w).map(|b| b.0), Ok(7));

    // A second sparse participant makes the query ambiguous.
    w.spawn((Pos(4), Buff(9)));
    assert!(matches!(
        state.single(&w),
        Err(QuerySingleError::MultipleEntities(_))
    ));
}

#[test]
fn single_matches_iteration_cardinality() {
    let mut w = World::new();
    let state = w.query::<&Pos>();

    // 0 matches.
    assert!(matches!(
        state.single(&w),
        Err(QuerySingleError::NoEntities(_))
    ));
    assert_eq!(state.iter(&w).count(), 0);

    // 1 match.
    w.spawn(Pos(1));
    assert!(state.single(&w).is_ok());
    assert_eq!(state.iter(&w).count(), 1);

    // 3 matches.
    w.spawn(Pos(2));
    w.spawn(Pos(3));
    assert!(matches!(
        state.single(&w),
        Err(QuerySingleError::MultipleEntities(_))
    ));
    assert_eq!(state.iter(&w).count(), 3);
}

#[test]
fn single_error_reports_the_query_name() {
    let mut w = World::new();
    let state = w.query::<&Pos>();
    let err = state.single(&w).expect_err("no entities yet");
    let QuerySingleError::NoEntities(name) = err else {
        panic!("expected NoEntities, got {err:?}");
    };
    assert!(
        name.contains("Pos"),
        "query name should mention the data term: {name}"
    );
    assert_eq!(
        format!("{err}"),
        format!("no entities fit the query {name}")
    );
}
