//! Soundness regression tests for query-state identity and cache validity
//! (design §7 / §10 / §14).
//!
//! A [`QueryState`](crate::query::QueryState) resolves components/archetypes to
//! dense ids *within one world's registries*. These tests pin the two guards
//! that keep that assumption sound:
//!
//! - **World identity.** Reusing a state against a different world must panic,
//!   never reinterpret the other world's columns.
//! - **Structural generation.** A snapshot `restore` wholesale-replaces the
//!   archetype registry; a state whose match cache was populated before the
//!   restore must rebuild it rather than trust stale archetype ids.

use alloc::vec::Vec;

use crate::component::Component;
use crate::world::World;

/// Table-backed position.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Pos(i32);
impl Component for Pos {}

/// Table-backed velocity.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Vel(i32);
impl Component for Vel {}

#[test]
#[should_panic(expected = "built for one World but used against another")]
fn query_state_iter_rejects_a_different_world() {
    let mut a = World::new();
    let mut b = World::new();
    a.spawn(Pos(1));
    b.spawn(Pos(2));

    let state = a.query::<&Pos>();
    // Iterating `b` with `a`'s state must be rejected up front.
    let _ = state.iter(&b).count();
}

#[test]
#[should_panic(expected = "built for one World but used against another")]
fn query_state_get_rejects_a_different_world() {
    let mut a = World::new();
    let mut b = World::new();
    a.spawn(Pos(1));
    let e = b.spawn(Pos(2));

    let state = a.query::<&Pos>();
    let _ = state.get(&b, e);
}

#[test]
fn query_state_cache_is_invalidated_by_snapshot_restore() {
    let mut w = World::new();
    w.register_snapshot_component::<Pos>();
    w.register_snapshot_component::<Vel>();

    let _a = w.spawn(Pos(1));
    let state = w.query::<&Pos>();
    // Populate the match cache against the pre-snapshot archetype list.
    assert_eq!(state.iter(&w).count(), 1);

    let snap = w.snapshot();

    // Grow the registry past the snapshot's size so a stale `checked` count
    // would exceed the restored registry length and short-circuit a refresh.
    w.spawn((Pos(9), Vel(1)));
    w.spawn(Vel(2));
    w.spawn(Pos(3));
    assert_eq!(state.iter(&w).count(), 3);

    w.restore(&snap);

    // The registry was rebuilt (fewer archetypes) and the entity set is back to
    // just the original `Pos(1)`; the state must observe exactly that, not a
    // stale archetype-id list.
    let values: Vec<i32> = state.iter(&w).map(|p| p.0).collect();
    assert_eq!(values, [1]);
}

/// The parallel entry points must run the same world-identity guard as the
/// serial ones instead of dereferencing another world's columns.
#[cfg(feature = "multi_thread")]
#[test]
#[should_panic(expected = "built for one World but used against another")]
fn parallel_entry_rejects_a_different_world() {
    let mut a = World::new();
    let mut b = World::new();
    a.spawn(Pos(1));
    b.spawn(Pos(2));

    let state = a.query::<&Pos>();
    let pool = prism_tasks::TaskPool::new();
    state.par_for_each(&b, &pool, 4, |_p| {});
}

/// The mutable parallel entry shares the guard.
#[cfg(feature = "multi_thread")]
#[test]
#[should_panic(expected = "built for one World but used against another")]
fn parallel_mut_entry_rejects_a_different_world() {
    let mut a = World::new();
    let mut b = World::new();
    a.spawn(Pos(1));
    b.spawn(Pos(2));

    let state = a.query::<&mut Pos>();
    let pool = prism_tasks::TaskPool::new();
    state.par_for_each_mut(&mut b, &pool, 4, |_p| {});
}
