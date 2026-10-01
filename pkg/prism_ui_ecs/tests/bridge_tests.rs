//! Integration tests for the ECS <-> reactive bridge.

use bevy_ecs::prelude::{Component, World};
use prism_ui_ecs::{EcsBridge, EntityBinding, FieldBinding};
use prism_ui_reactive::Runtime;

#[derive(Component)]
struct Health {
    current: i32,
    max: i32,
}

#[derive(Component)]
struct Name {
    text: u64,
}

#[test]
fn one_way_pull_propagates_component_to_signal() {
    let mut world = World::new();
    let entity = world
        .spawn(Health {
            current: 30,
            max: 100,
        })
        .id();
    let rt = Runtime::new();
    let current = rt.signal(0i32);

    let mut bridge = EcsBridge::new();
    bridge.bind::<Health, i32>(entity, current.clone(), |h| h.current);

    assert_eq!(bridge.pull_all(&world), 1);
    assert_eq!(current.get_untracked(), 30);
}

#[test]
fn unchanged_component_does_not_repropagate() {
    let mut world = World::new();
    let entity = world
        .spawn(Health {
            current: 50,
            max: 100,
        })
        .id();
    let rt = Runtime::new();
    let current = rt.signal(0i32);

    let mut bridge = EcsBridge::new();
    bridge.bind::<Health, i32>(entity, current.clone(), |h| h.current);

    assert_eq!(bridge.pull_all(&world), 1);
    // The tick guard suppresses re-reading an unchanged component.
    assert_eq!(bridge.pull_all(&world), 0);
    assert_eq!(bridge.pull_all(&world), 0);
    assert_eq!(current.get_untracked(), 50);
}

#[test]
fn two_way_push_writes_back_to_component() {
    let mut world = World::new();
    let entity = world
        .spawn(Health {
            current: 0,
            max: 100,
        })
        .id();
    let rt = Runtime::new();
    let current = rt.signal(80i32);

    let mut bridge = EcsBridge::new();
    bridge.bind_two_way::<Health, i32>(
        entity,
        current.clone(),
        |h| h.current,
        |h, v| h.current = *v,
    );

    assert_eq!(bridge.push_all(&mut world), 1);
    assert_eq!(world.get::<Health>(entity).unwrap().current, 80);
}

#[test]
fn round_trip_does_not_oscillate() {
    let mut world = World::new();
    let entity = world
        .spawn(Health {
            current: 10,
            max: 100,
        })
        .id();
    let rt = Runtime::new();
    let current = rt.signal(0i32);

    let mut bridge = EcsBridge::new();
    bridge.bind_two_way::<Health, i32>(
        entity,
        current.clone(),
        |h| h.current,
        |h, v| h.current = *v,
    );

    // ECS -> signal, then an immediate push finds no difference.
    assert_eq!(bridge.pull_all(&world), 1);
    assert_eq!(current.get_untracked(), 10);
    assert_eq!(bridge.push_all(&mut world), 0);

    // signal -> ECS, then an immediate pull reports no net change.
    current.set(25);
    assert_eq!(bridge.push_all(&mut world), 1);
    assert_eq!(world.get::<Health>(entity).unwrap().current, 25);
    assert_eq!(bridge.pull_all(&world), 0);
    assert_eq!(current.get_untracked(), 25);
}

#[test]
fn multiple_bindings_over_multiple_entities() {
    let mut world = World::new();
    let hero = world.spawn(Health { current: 7, max: 7 }).id();
    let foe = world.spawn(Health { current: 3, max: 9 }).id();
    let rt = Runtime::new();
    let hero_cur = rt.signal(0i32);
    let hero_max = rt.signal(0i32);
    let foe_cur = rt.signal(0i32);

    let mut bridge = EcsBridge::new();
    bridge.bind::<Health, i32>(hero, hero_cur.clone(), |h| h.current);
    bridge.bind::<Health, i32>(hero, hero_max.clone(), |h| h.max);
    bridge.bind::<Health, i32>(foe, foe_cur.clone(), |h| h.current);
    assert_eq!(bridge.len(), 3);

    assert_eq!(bridge.pull_all(&world), 3);
    assert_eq!(hero_cur.get_untracked(), 7);
    assert_eq!(hero_max.get_untracked(), 7);
    assert_eq!(foe_cur.get_untracked(), 3);
}

#[test]
fn missing_component_is_safe() {
    let mut world = World::new();
    // Entity without a `Name` component bound below.
    let entity = world.spawn(Health { current: 1, max: 1 }).id();
    let rt = Runtime::new();
    let name = rt.signal(0u64);

    let mut bridge = EcsBridge::new();
    bridge.bind_two_way::<Name, u64>(entity, name.clone(), |n| n.text, |n, v| n.text = *v);

    assert_eq!(bridge.pull_all(&world), 0);
    assert_eq!(bridge.push_all(&mut world), 0);
    assert_eq!(name.get_untracked(), 0);
}

#[test]
fn direct_entity_binding_without_bridge() {
    let mut world = World::new();
    let entity = world
        .spawn(Health {
            current: 42,
            max: 50,
        })
        .id();
    let rt = Runtime::new();
    let current = rt.signal(0i32);

    let binding = FieldBinding::<Health, i32>::read_write(
        current.clone(),
        |h| h.current,
        |h, v| h.current = *v,
    );
    let mut entity_binding = EntityBinding::new(entity, binding);
    use prism_ui_ecs::SyncBinding;

    assert!(entity_binding.pull(&world));
    assert_eq!(current.get_untracked(), 42);

    current.set(13);
    assert!(entity_binding.push(&mut world));
    assert_eq!(world.get::<Health>(entity).unwrap().current, 13);
}
