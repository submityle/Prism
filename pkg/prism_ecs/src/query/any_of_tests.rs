//! Behavioural tests for the [`AnyOf<(..)>`](crate::query::AnyOf) query term
//! (design §6, §7).
//!
//! `AnyOf<(A, B, ..)>` is the `||` dual of a tuple's `&&`: it matches an
//! archetype when *at least one* element is present and yields each element as
//! an `Option`, visiting a row only when some element is actually present for
//! it. These tests pin that contract across table, sparse, and shared storage,
//! the "neither present" exclusion, mutable elements, and tuple composition.

use alloc::vec;
use alloc::vec::Vec;

use crate::component::{Component, ComponentId, Components, StorageType};
use crate::entity::Entity;
use crate::query::AnyOf;
use crate::world::World;

/// Table-backed anchor so entities can share an archetype that carries neither
/// probed component.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Tag(u32);
impl Component for Tag {}

/// Table-backed role A.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Melee(u32);
impl Component for Melee {}

/// Table-backed role B.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Ranged(u32);
impl Component for Ranged {}

/// Sparse-backed role, routed out of the archetype.
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

#[test]
fn any_of_matches_union_and_yields_per_element_options() {
    let mut w = World::new();
    let only_a = w.spawn(Melee(10));
    let only_b = w.spawn(Ranged(7));
    let both = w.spawn((Melee(3), Ranged(5)));

    let state = w.query::<(Entity, AnyOf<(&Melee, &Ranged)>)>();
    let mut seen: Vec<(Entity, Option<u32>, Option<u32>)> = state
        .iter(&w)
        .map(|(e, (m, r))| (e, m.map(|m| m.0), r.map(|r| r.0)))
        .collect();
    seen.sort_by_key(|&(e, _, _)| e.index());

    assert_eq!(seen.len(), 3);
    assert!(seen.contains(&(only_a, Some(10), None)));
    assert!(seen.contains(&(only_b, None, Some(7))));
    assert!(seen.contains(&(both, Some(3), Some(5))));
}

#[test]
fn any_of_never_visits_rows_where_every_element_is_absent() {
    let mut w = World::new();
    w.spawn(Melee(1));
    w.spawn(Ranged(2));
    // Entities carrying neither probed component must never be yielded, even
    // though they share no archetype with the matched ones.
    w.spawn(Tag(99));
    w.spawn(Tag(100));

    let state = w.query::<AnyOf<(&Melee, &Ranged)>>();
    let rows: Vec<(Option<u32>, Option<u32>)> =
        state.iter(&w).map(|(m, r)| (m.map(|m| m.0), r.map(|r| r.0))).collect();

    assert_eq!(rows.len(), 2);
    assert!(rows.iter().all(|&(m, r)| m.is_some() || r.is_some()));
}

#[test]
fn any_of_resolves_sparse_elements_per_entity() {
    let mut w = World::new();
    // All three share archetype {Tag}; Melee/Buff presence differs per entity.
    let melee_only = w.spawn((Tag(1), Melee(4)));
    let buff_only = w.spawn((Tag(2), Buff(9)));
    let neither = w.spawn(Tag(3));

    let state = w.query::<(Entity, AnyOf<(&Melee, &Buff)>)>();
    let seen: Vec<(Entity, Option<u32>, Option<i32>)> = state
        .iter(&w)
        .map(|(e, (m, b))| (e, m.map(|m| m.0), b.map(|b| b.0)))
        .collect();

    assert!(seen.contains(&(melee_only, Some(4), None)));
    assert!(seen.contains(&(buff_only, None, Some(9))));
    // `neither` has only Tag: not visited despite sharing the archetype.
    assert!(!seen.iter().any(|&(e, _, _)| e == neither));
    assert_eq!(seen.len(), 2);
}

#[test]
fn any_of_handles_shared_elements() {
    let mut w = World::new();
    let keyed = w.spawn((Tag(1), Faction(7)));
    let melee = w.spawn((Tag(2), Melee(5)));
    let plain = w.spawn(Tag(3));

    let state = w.query::<(Entity, AnyOf<(&Faction, &Melee)>)>();
    let seen: Vec<(Entity, Option<u32>, Option<u32>)> = state
        .iter(&w)
        .map(|(e, (f, m))| (e, f.map(|f| f.0), m.map(|m| m.0)))
        .collect();

    assert!(seen.contains(&(keyed, Some(7), None)));
    assert!(seen.contains(&(melee, None, Some(5))));
    assert!(!seen.iter().any(|&(e, _, _)| e == plain));
    assert_eq!(seen.len(), 2);
}

#[test]
fn any_of_mutable_elements_write_back_where_present() {
    let mut w = World::new();
    let a = w.spawn(Melee(1));
    let b = w.spawn(Ranged(2));
    let both = w.spawn((Melee(3), Ranged(4)));

    let state = w.query::<AnyOf<(&mut Melee, &mut Ranged)>>();
    for (melee, ranged) in state.iter_mut(&mut w) {
        if let Some(mut m) = melee {
            m.0 += 100;
        }
        if let Some(mut r) = ranged {
            r.0 += 1000;
        }
    }

    assert_eq!(w.get::<Melee>(a), Some(&Melee(101)));
    assert_eq!(w.get::<Ranged>(b), Some(&Ranged(1002)));
    assert_eq!(w.get::<Melee>(both), Some(&Melee(103)));
    assert_eq!(w.get::<Ranged>(both), Some(&Ranged(1004)));
}

#[test]
fn any_of_single_element_behaves_like_a_gated_option() {
    let mut w = World::new();
    let with = w.spawn(Melee(8));
    w.spawn(Tag(1)); // lacks Melee entirely

    let state = w.query::<(Entity, AnyOf<(&Melee,)>)>();
    let seen: Vec<(Entity, Option<u32>)> =
        state.iter(&w).map(|(e, (m,))| (e, m.map(|m| m.0))).collect();

    // Only the archetype carrying Melee is matched; its one row is Some.
    assert_eq!(seen, vec![(with, Some(8))]);
}

#[test]
fn any_of_composes_inside_a_tuple_with_other_terms() {
    let mut w = World::new();
    let x = w.spawn((Tag(5), Melee(1)));
    let y = w.spawn((Tag(6), Ranged(2)));
    w.spawn(Melee(9)); // no Tag: excluded by the `&Tag` term

    let state = w.query::<(&Tag, AnyOf<(&Melee, &Ranged)>)>();
    let mut seen: Vec<(u32, Option<u32>, Option<u32>)> = state
        .iter(&w)
        .map(|(tag, (m, r))| (tag.0, m.map(|m| m.0), r.map(|r| r.0)))
        .collect();
    seen.sort_by_key(|&(t, _, _)| t);

    assert_eq!(seen, vec![(5, Some(1), None), (6, None, Some(2))]);
    let _ = (x, y);
}
