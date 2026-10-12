//! Behavioural tests for archetype prewarm (design §17 原型预热).

use crate::component::{Component, StorageType};
use crate::world::World;

/// Table-backed position.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Pos(i32);
impl Component for Pos {}

/// Table-backed velocity.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Vel(i32);
impl Component for Vel {}

/// A component attached to `Pos` by the required-components rule.
#[derive(Debug, Default, PartialEq, Clone, Copy)]
struct Required;
impl Component for Required {}

/// Shared-backed batch key, which no prewarm can resolve without a value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Faction(u32);
impl Component for Faction {
    const STORAGE: StorageType = StorageType::Shared;
    fn install_storage_glue(components: &mut crate::component::Components, id: crate::component::ComponentId) {
        components.set_shared_box(id, crate::component::shared_box_of::<Self>());
    }
}

/// Sparse-backed component, routed out of the archetype.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Buff(i32);
impl Component for Buff {
    const STORAGE: StorageType = StorageType::SparseSet;
}

#[test]
fn prewarm_creates_the_archetype_and_reserves_capacity_without_spawning() {
    let mut world = World::new();
    assert_eq!(world.entity_count(), 0);

    let archetype = world.prewarm::<(Pos, Vel)>(1024);

    assert_eq!(world.entity_count(), 0, "prewarm spawns nobody");
    let table = world
        .archetypes()
        .get(archetype)
        .expect("the archetype was created")
        .table();
    assert!(
        table.capacity() >= 1024,
        "every column must be able to take 1024 rows, got {}",
        table.capacity()
    );

    // A later spawn of the same bundle lands in the warmed archetype, so the
    // reservation actually pays off.
    let e = world.spawn((Pos(1), Vel(2)));
    let landed = world
        .entities()
        .location(e)
        .expect("spawned")
        .archetype_id;
    assert_eq!(landed, archetype);
    assert!(world.archetypes().get(landed).expect("live").table().capacity() >= 1024);
}

#[test]
fn prewarm_includes_required_components() {
    let mut world = World::new();
    world.register_required_component::<Pos, Required>();

    // The warmed archetype is the one a spawn of `Pos` targets, required
    // component included — otherwise the reservation would be wasted.
    let warmed = world.prewarm::<Pos>(64);
    let e = world.spawn(Pos(3));
    assert_eq!(
        world.entities().location(e).expect("spawned").archetype_id,
        warmed
    );
    assert!(world.has::<Required>(e), "the required component was added");
    assert!(world.archetypes().get(warmed).expect("live").table().capacity() >= 64);
}

#[test]
fn prewarming_again_only_grows() {
    let mut world = World::new();
    let first = world.prewarm::<Pos>(16);
    let second = world.prewarm::<Pos>(4096);
    assert_eq!(first, second, "the same bundle resolves to one archetype");
    assert!(
        world
            .archetypes()
            .get(second)
            .expect("live")
            .table()
            .capacity()
            >= 4096
    );
    assert_eq!(world.entity_count(), 0);
}

#[test]
fn prewarm_covers_sparse_bundles_without_allocating_archetype_rows() {
    let mut world = World::new();
    // A sparse-only bundle has no table column, so prewarm resolves to the
    // empty archetype and must still create the out-of-band set.
    let warmed = world.prewarm::<Buff>(32);
    assert!(world.archetypes().get(warmed).is_some());
    let buff_id = world.components().id_of::<Buff>().expect("registered");
    assert!(
        world.sparse_sets().get(buff_id).is_some(),
        "the sparse set is pre-created so the first insert does not allocate it"
    );
    assert_eq!(world.entity_count(), 0);

    // And a real spawn of the sparse bundle lands in that same archetype.
    let e = world.spawn(Buff(1));
    assert_eq!(
        world.entities().location(e).expect("spawned").archetype_id,
        warmed
    );
}

#[test]
#[should_panic(expected = "shared component")]
fn prewarm_rejects_a_shared_bundle() {
    let mut world = World::new();
    // Which archetype a shared component lands in depends on its interned value,
    // so the bundle type alone cannot name it.
    let _ = world.prewarm::<(Pos, Faction)>(8);
}
