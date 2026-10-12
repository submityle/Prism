//! Behavioural tests for `QueryState::iter_combinations` (design §7).

use alloc::vec::Vec;

use crate::component::Component;
use crate::entity::Entity;
use crate::query::{With, Without};
use crate::world::World;

/// Table-backed position, used as the combination payload.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Pos(i32);
impl Component for Pos {}

/// Table-backed marker for filtering.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Tag;
impl Component for Tag {}

fn spawned(n: i32) -> (World, Vec<Entity>) {
    let mut world = World::new();
    let entities = (0..n).map(|i| world.spawn(Pos(i))).collect();
    (world, entities)
}

#[test]
fn pairs_are_every_unordered_combination_exactly_once() {
    let (mut world, entities) = spawned(5);
    let state = world.query::<(Entity, &Pos)>();

    let mut pairs: Vec<(Entity, Entity)> = state
        .iter_combinations::<2>(&world)
        .map(|[(a, _), (b, _)]| (a, b))
        .collect();

    // C(5, 2) = 10, each unordered pair once, always ascending in row order.
    assert_eq!(pairs.len(), 10);
    for (a, b) in &pairs {
        assert_ne!(a, b, "a combination never repeats an entity");
        let (ia, ib) = (
            entities.iter().position(|e| e == a).expect("known entity"),
            entities.iter().position(|e| e == b).expect("known entity"),
        );
        assert!(ia < ib, "combinations come out in ascending row order");
    }

    pairs.sort_unstable();
    let mut expected: Vec<(Entity, Entity)> = Vec::new();
    for i in 0..entities.len() {
        for j in (i + 1)..entities.len() {
            expected.push((entities[i], entities[j]));
        }
    }
    expected.sort_unstable();
    assert_eq!(pairs, expected);
}

#[test]
fn triples_are_generated_and_ordered() {
    let (mut world, entities) = spawned(4);
    let state = world.query::<(Entity, &Pos)>();

    let triples: Vec<[Entity; 3]> = state
        .iter_combinations::<3>(&world)
        .map(|[(a, _), (b, _), (c, _)]| [a, b, c])
        .collect();

    // C(4, 3) = 4, lexicographic over the row order.
    assert_eq!(
        triples,
        [
            [entities[0], entities[1], entities[2]],
            [entities[0], entities[1], entities[3]],
            [entities[0], entities[2], entities[3]],
            [entities[1], entities[2], entities[3]],
        ]
    );
}

#[test]
fn impossible_arity_yields_nothing() {
    let (mut world, _entities) = spawned(2);
    let state = world.query::<&Pos>();

    assert_eq!(state.iter_combinations::<0>(&world).count(), 0);
    assert_eq!(state.iter_combinations::<3>(&world).count(), 0);
    // K == 1 degenerates to ordinary iteration.
    assert_eq!(state.iter_combinations::<1>(&world).count(), 2);
}

#[test]
fn combinations_only_use_rows_the_query_matches() {
    let mut world = World::new();
    let tagged = world.spawn((Pos(1), Tag));
    let other = world.spawn((Pos(2), Tag));
    let untagged = world.spawn(Pos(3));

    let with_tag = world.query_filtered::<Entity, With<Tag>>();
    let pairs: Vec<[Entity; 2]> = with_tag.iter_combinations::<2>(&world).collect();
    assert_eq!(pairs, [[tagged, other]]);

    let without_tag = world.query_filtered::<Entity, Without<Tag>>();
    // One match cannot form a pair.
    assert_eq!(without_tag.iter_combinations::<2>(&world).count(), 0);
    assert_eq!(
        without_tag.iter_combinations::<1>(&world).next(),
        Some([untagged])
    );
}

#[test]
fn combinations_are_repeatable() {
    let (mut world, _entities) = spawned(6);
    let state = world.query::<(Entity, &Pos)>();

    let first: Vec<[i32; 2]> = state
        .iter_combinations::<2>(&world)
        .map(|[(_, a), (_, b)]| [a.0, b.0])
        .collect();
    let second: Vec<[i32; 2]> = state
        .iter_combinations::<2>(&world)
        .map(|[(_, a), (_, b)]| [a.0, b.0])
        .collect();
    assert_eq!(first, second, "the same world must yield the same sequence");
    assert_eq!(first.len(), 15, "C(6, 2)");
}
