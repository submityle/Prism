//! Behavioural tests for random-access single-entity query lookup
//! ([`QueryState::get`] / [`QueryState::get_mut`], design §7).
//!
//! `get`/`get_mut` are the `O(1)` dual of iteration: they jump straight to an
//! entity's storage location, re-check its archetype against the query's
//! data/filter terms, and apply the same per-row gate iteration uses. These
//! tests pin the success path (table/sparse/shared, read and write), the two
//! error cases ([`QueryEntityError::NoSuchEntity`] /
//! [`QueryEntityError::QueryDoesNotMatch`]), filter interaction
//! (`With`/`Without`/`Changed`), and parity with iteration.

use alloc::vec::Vec;

use crate::component::{Component, ComponentId, Components, StorageType};
use crate::entity::Entity;
use crate::query::{Changed, QueryEntityError, With, Without};
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

/// Sparse-backed buff, routed out of the archetype.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Buff(i32);
impl Component for Buff {
    const STORAGE: StorageType = StorageType::SparseSet;
}

/// Shared-backed batch key (design §6).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Faction(u32);
impl Component for Faction {
    const STORAGE: StorageType = StorageType::Shared;
    fn install_storage_glue(components: &mut Components, id: ComponentId) {
        components.set_shared_box(id, crate::component::shared_box_of::<Self>());
    }
}

/// Advance the world one change-detection frame (the sparse/change tests rely
/// on this to separate "spawned last frame" from "touched this frame").
fn next_frame(w: &mut World) {
    w.set_last_change_tick(w.change_tick());
    w.increment_change_tick();
}

#[test]
fn get_returns_the_matching_rows_value() {
    let mut w = World::new();
    let a = w.spawn(Pos(10));
    let b = w.spawn(Pos(20));

    let state = w.query::<&Pos>();
    assert_eq!(state.get(&w, a).map(|p| p.0), Ok(10));
    assert_eq!(state.get(&w, b).map(|p| p.0), Ok(20));
}

#[test]
fn get_errors_no_such_entity_for_despawned_and_never_spawned() {
    let mut w = World::new();
    let a = w.spawn(Pos(1));
    assert!(w.despawn(a));

    let state = w.query::<&Pos>();
    // Stale handle to a freed slot.
    assert_eq!(state.get(&w, a), Err(QueryEntityError::NoSuchEntity(a)));
    // A plausible-looking handle that was never allocated.
    let bogus = Entity::from_bits(0xDEAD_0000_0000_0001).expect("valid bit pattern");
    assert_eq!(
        state.get(&w, bogus),
        Err(QueryEntityError::NoSuchEntity(bogus))
    );
}

#[test]
fn get_errors_query_does_not_match_when_component_absent() {
    let mut w = World::new();
    let has = w.spawn((Pos(5), Vel(1)));
    let lacks = w.spawn(Pos(9));

    let state = w.query::<&Vel>();
    assert_eq!(state.get(&w, has).map(|v| v.0), Ok(1));
    assert_eq!(
        state.get(&w, lacks),
        Err(QueryEntityError::QueryDoesNotMatch(lacks))
    );
}

#[test]
fn get_honours_with_and_without_filters() {
    let mut w = World::new();
    let moving = w.spawn((Pos(1), Vel(2)));
    let still = w.spawn(Pos(3));
    let frozen = w.spawn((Pos(4), Vel(5), Frozen));

    let with_vel = w.query_filtered::<&Pos, With<Vel>>();
    assert_eq!(with_vel.get(&w, moving).map(|p| p.0), Ok(1));
    assert_eq!(
        with_vel.get(&w, still),
        Err(QueryEntityError::QueryDoesNotMatch(still))
    );

    let without_frozen = w.query_filtered::<&Pos, (With<Vel>, Without<Frozen>)>();
    assert_eq!(without_frozen.get(&w, moving).map(|p| p.0), Ok(1));
    assert_eq!(
        without_frozen.get(&w, frozen),
        Err(QueryEntityError::QueryDoesNotMatch(frozen))
    );
}

#[test]
fn get_mut_writes_back_through_the_fetched_reference() {
    let mut w = World::new();
    let e = w.spawn(Pos(1));

    let state = w.query::<&mut Pos>();
    {
        let mut p = state.get_mut(&mut w, e).expect("entity matches");
        p.0 += 41;
    }
    assert_eq!(w.get::<Pos>(e).map(|p| p.0), Some(42));
}

#[test]
fn get_resolves_sparse_backed_terms_per_entity() {
    let mut w = World::new();
    let buffed = w.spawn((Pos(1), Buff(7)));
    let plain = w.spawn(Pos(2));

    let state = w.query::<&Buff>();
    assert_eq!(state.get(&w, buffed).map(|b| b.0), Ok(7));
    // `plain` shares no archetype carrying `Buff`, so the archetype gate rejects
    // it as a non-match rather than a missing entity.
    assert_eq!(
        state.get(&w, plain),
        Err(QueryEntityError::QueryDoesNotMatch(plain))
    );
}

#[test]
fn get_resolves_shared_backed_terms() {
    let mut w = World::new();
    let a = w.spawn((Pos(1), Faction(3)));
    let b = w.spawn((Pos(2), Faction(3)));
    let none = w.spawn(Pos(9));

    let state = w.query::<&Faction>();
    assert_eq!(state.get(&w, a).map(|f| f.0), Ok(3));
    assert_eq!(state.get(&w, b).map(|f| f.0), Ok(3));
    assert_eq!(
        state.get(&w, none),
        Err(QueryEntityError::QueryDoesNotMatch(none))
    );
}

#[test]
fn get_applies_the_per_row_changed_gate() {
    let mut w = World::new();
    let e = w.spawn(Pos(1));
    // Advance a frame so the spawn tick is in the past.
    next_frame(&mut w);

    // No write this frame: `Changed<Pos>` must reject the row.
    let changed = w.query_filtered::<Entity, Changed<Pos>>();
    assert_eq!(
        changed.get(&w, e),
        Err(QueryEntityError::QueryDoesNotMatch(e))
    );

    // Touch `Pos` this frame, then the same filter must accept it.
    {
        let mut p = w.get_mut::<Pos>(e).expect("live");
        p.0 += 1;
    }
    let changed = w.query_filtered::<Entity, Changed<Pos>>();
    assert_eq!(changed.get(&w, e), Ok(e));
}

#[test]
fn get_agrees_with_iteration_over_every_entity() {
    let mut w = World::new();
    let mut entities: Vec<Entity> = Vec::new();
    for i in 0..16 {
        if i % 2 == 0 {
            entities.push(w.spawn((Pos(i), Vel(i * 10))));
        } else {
            entities.push(w.spawn(Pos(i)));
        }
    }

    // Build the expected (entity -> pos) map by iterating the Vel-filtered query.
    let state = w.query_filtered::<(Entity, &Pos), With<Vel>>();
    let mut iterated: Vec<(Entity, i32)> = state.iter(&w).map(|(e, p)| (e, p.0)).collect();
    iterated.sort_by_key(|&(e, _)| e.index());

    // `get` must return exactly those, and reject the rest.
    for &e in &entities {
        match state.get(&w, e) {
            Ok((ge, p)) => {
                assert_eq!(ge, e);
                assert!(iterated.contains(&(e, p.0)));
            }
            Err(QueryEntityError::QueryDoesNotMatch(ne)) => assert_eq!(ne, e),
            Err(other) => panic!("unexpected error for a live entity: {other:?}"),
        }
    }
    let matched = entities
        .iter()
        .filter(|&&e| state.get(&w, e).is_ok())
        .count();
    assert_eq!(matched, iterated.len());
}
