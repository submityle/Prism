//! Behavioural tests for the [`Has<T>`](crate::query::Has) presence-probe query
//! term (design §6, §7).
//!
//! `Has<T>` yields `bool` per row — whether the entity carries `T` — without
//! excluding any archetype and without reading the component value. These tests
//! pin that contract across all three storage states (table, sparse, shared),
//! the degenerate "sparse set never allocated" case, composition in tuples, and
//! the key soundness property that `Has<T>` registers no access and therefore
//! coexists with a `&mut T` term on the *same* component.

use alloc::vec::Vec;

use crate::component::{Component, ComponentId, Components, StorageType};
use crate::entity::Entity;
use crate::query::Has;
use crate::world::World;

/// Table-backed (default) anchor component.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Position(i32);
impl Component for Position {}

/// Table-backed component probed by `Has`.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Shield(u32);
impl Component for Shield {}

/// Sparse-backed component: routed out of the archetype, keyed by entity.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Charge(i32);
impl Component for Charge {
    const STORAGE: StorageType = StorageType::SparseSet;
}

/// A second sparse component, so a sparse set can exist in the world without
/// `Charge` ever having been inserted.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Flagged;
impl Component for Flagged {
    const STORAGE: StorageType = StorageType::SparseSet;
}

/// Shared-backed batch key (design §6).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BatchKey(u32);
impl Component for BatchKey {
    const STORAGE: StorageType = StorageType::Shared;
    fn install_storage_glue(components: &mut Components, id: ComponentId) {
        components.set_shared_box(id, crate::component::shared_box_of::<Self>());
    }
}

#[test]
fn has_table_component_reports_presence_without_excluding_rows() {
    let mut w = World::new();
    let with = w.spawn((Position(1), Shield(50)));
    let without = w.spawn(Position(2));

    let state = w.query::<(Entity, Has<Shield>)>();
    let mut seen: Vec<(Entity, bool)> = state.iter(&w).collect();
    seen.sort_by_key(|&(e, _)| e.index());

    // Both entities are visited (Has never filters); only `with` reports true.
    assert_eq!(seen.len(), 2);
    assert!(seen.iter().any(|&(e, h)| e == with && h));
    assert!(seen.iter().any(|&(e, h)| e == without && !h));
}

#[test]
fn has_sparse_component_is_resolved_per_entity_in_one_archetype() {
    let mut w = World::new();
    // Both land in archetype {Position}; `Charge` is stored out of band, so
    // presence differs per entity within the same archetype.
    let has = w.spawn((Position(1), Charge(10)));
    let hasnt = w.spawn(Position(2));

    let state = w.query::<(Entity, Has<Charge>)>();
    let seen: Vec<(Entity, bool)> = state.iter(&w).collect();

    assert_eq!(seen.len(), 2);
    assert!(seen.iter().any(|&(e, h)| e == has && h));
    assert!(seen.iter().any(|&(e, h)| e == hasnt && !h));
}

#[test]
fn has_sparse_component_is_false_when_set_never_allocated() {
    let mut w = World::new();
    // A *different* sparse component is inserted, so a sparse registry exists,
    // but `Charge`'s set was never allocated. `Has<Charge>` must still answer
    // `false` rather than panic.
    w.spawn((Position(1), Flagged));
    w.spawn(Position(2));

    let state = w.query::<Has<Charge>>();
    let any_true = state.iter(&w).any(|h| h);
    assert!(!any_true);
}

#[test]
fn has_shared_component_reports_archetype_wide_binding() {
    let mut w = World::new();
    let keyed = w.spawn((Position(1), BatchKey(7)));
    let plain = w.spawn(Position(2));

    let state = w.query::<(Entity, Has<BatchKey>)>();
    let seen: Vec<(Entity, bool)> = state.iter(&w).collect();

    assert_eq!(seen.len(), 2);
    assert!(seen.iter().any(|&(e, h)| e == keyed && h));
    assert!(seen.iter().any(|&(e, h)| e == plain && !h));
}

#[test]
fn has_coexists_with_mutable_access_to_the_same_component() {
    // `Has<T>` registers no access, so `(&mut T, Has<T>)` must not be rejected
    // as an aliasing conflict and the mutable write must still land.
    let mut w = World::new();
    let a = w.spawn(Shield(1));
    let b = w.spawn(Shield(2));

    let state = w.query::<(&mut Shield, Has<Shield>)>();
    for (mut shield, has) in state.iter_mut(&mut w) {
        // Every yielded row has the component, since `&mut Shield` requires it.
        assert!(has);
        shield.0 += 100;
    }

    assert_eq!(w.get::<Shield>(a), Some(&Shield(101)));
    assert_eq!(w.get::<Shield>(b), Some(&Shield(102)));
}

#[test]
fn has_distinguishes_two_table_archetypes() {
    let mut w = World::new();
    w.spawn((Position(1), Shield(10)));
    w.spawn((Position(2), Shield(20)));
    w.spawn(Position(3));

    let state = w.query::<Has<Shield>>();
    let trues = state.iter(&w).filter(|&h| h).count();
    let falses = state.iter(&w).filter(|&h| !h).count();
    assert_eq!(trues, 2);
    assert_eq!(falses, 1);
}
