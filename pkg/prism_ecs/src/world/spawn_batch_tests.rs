//! Behavioural tests for [`World::spawn_batch`] (design §5 / §17 mass spawning).

use alloc::vec::Vec;

use crate::component::{Component, ComponentId, Components, StorageType};
use crate::component_hooks::{ComponentHooks, HookContext};
use crate::entity::Entity;
use crate::resource::Resource;
use crate::world::World;

/// Table-backed position.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Pos(i32);
impl Component for Pos {}

/// Table-backed velocity.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Vel(i32);
impl Component for Vel {}

/// Auto-added companion for [`Pos`].
#[derive(Debug, Default, PartialEq, Clone, Copy)]
struct Required;
impl Component for Required {}

/// Shared-backed batch key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Faction(u32);
impl Component for Faction {
    const STORAGE: StorageType = StorageType::Shared;
    fn install_storage_glue(components: &mut Components, id: ComponentId) {
        components.set_shared_box(id, crate::component::shared_box_of::<Self>());
    }
}

/// Hook firing counts, routed through a resource.
#[derive(Default, PartialEq, Debug)]
struct HookLog {
    adds: usize,
    inserts: usize,
}
impl Resource for HookLog {}

fn count_add(ctx: HookContext<'_>) {
    ctx.world.resource_mut::<HookLog>().adds += 1;
}

fn count_insert(ctx: HookContext<'_>) {
    ctx.world.resource_mut::<HookLog>().inserts += 1;
}

#[test]
fn spawn_batch_matches_individual_spawns() {
    let mut batched = World::new();
    let batched_entities = batched.spawn_batch((0..8).map(|i| (Pos(i), Vel(-i))));

    let mut one_by_one = World::new();
    let expected: Vec<Entity> = (0..8)
        .map(|i| one_by_one.spawn((Pos(i), Vel(-i))))
        .collect();

    assert_eq!(batched_entities.len(), 8);
    assert_eq!(
        batched_entities, expected,
        "entities come out in input order"
    );
    assert_eq!(batched.entity_count(), one_by_one.entity_count());
    for (batched_entity, expected_entity) in batched_entities.iter().zip(&expected) {
        assert_eq!(
            batched.get::<Pos>(*batched_entity),
            one_by_one.get::<Pos>(*expected_entity)
        );
        assert_eq!(
            batched.get::<Vel>(*batched_entity),
            one_by_one.get::<Vel>(*expected_entity)
        );
        // The whole batch landed in one archetype, exactly where one-by-one
        // spawning put them.
        assert_eq!(
            batched
                .entities()
                .location(*batched_entity)
                .expect("live")
                .archetype_id,
            one_by_one
                .entities()
                .location(*expected_entity)
                .expect("live")
                .archetype_id,
        );
    }
}

#[test]
fn spawn_batch_runs_the_per_entity_hooks() {
    let mut world = World::new();
    world.insert_resource(HookLog::default());
    world.register_component_hooks::<Pos>(
        ComponentHooks::new()
            .with_on_add(count_add)
            .with_on_insert(count_insert),
    );

    let _ = world.spawn_batch((0..5).map(Pos));

    // Hoisting the archetype resolution must not hoist the lifecycle work: every
    // spawned entity fires its own add + insert.
    assert_eq!(
        world.resource::<HookLog>(),
        &HookLog {
            adds: 5,
            inserts: 5
        }
    );
}

#[test]
fn spawn_batch_adds_required_components_per_entity() {
    let mut world = World::new();
    world.register_required_component::<Pos, Required>();

    let entities = world.spawn_batch((0..4).map(Pos));

    for entity in &entities {
        assert!(
            world.has::<Required>(*entity),
            "each entity gets its own default"
        );
    }
    assert_eq!(world.entity_count(), 4);
}

#[test]
fn spawn_batch_falls_back_for_shared_bundles() {
    let mut world = World::new();
    // Shared values decide the archetype, so these two items cannot share one
    // resolved target — the per-item path handles it and both values survive.
    let entities = world.spawn_batch([
        (Pos(1), Faction(7)),
        (Pos(2), Faction(9)),
        (Pos(3), Faction(7)),
    ]);

    assert_eq!(world.get::<Faction>(entities[0]).map(|f| f.0), Some(7));
    assert_eq!(world.get::<Faction>(entities[1]).map(|f| f.0), Some(9));
    assert_eq!(world.get::<Faction>(entities[2]).map(|f| f.0), Some(7));
    let first = world
        .entities()
        .location(entities[0])
        .expect("live")
        .archetype_id;
    let second = world
        .entities()
        .location(entities[1])
        .expect("live")
        .archetype_id;
    let third = world
        .entities()
        .location(entities[2])
        .expect("live")
        .archetype_id;
    assert_ne!(first, second, "different shared values split the archetype");
    assert_eq!(first, third, "equal shared values share one archetype");
}

#[test]
fn an_empty_batch_spawns_nothing() {
    let mut world = World::new();
    let entities: Vec<Entity> = world.spawn_batch(core::iter::empty::<(Pos, Vel)>());
    assert!(entities.is_empty());
    assert_eq!(world.entity_count(), 0);
}
