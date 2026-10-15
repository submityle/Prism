//! The core inheritance API works without the reflection feature.

use prism_ecs::component::Component;
use prism_ecs::prefab::PrefabError;
use prism_ecs::world::World;

#[derive(Debug, PartialEq)]
struct Value(u32);
impl Component for Value {}

#[test]
fn breadth_first_precedence_beats_an_earlier_but_deeper_branch() {
    let mut world = World::new();
    let deep = world.spawn(Value(1));
    let left = world.spawn_instance_of(deep);
    let right = world.spawn(Value(2));
    let instance = world.spawn_instance_of(left);
    world.try_add_prefab(instance, right).unwrap();
    assert_eq!(world.prefab_chain(instance), vec![left, right, deep]);
    assert_eq!(world.get_inherited::<Value>(instance), Some(&Value(2)));
    assert_eq!(world.inherited_source::<Value>(instance), Some(right));
    world.remove::<Value>(right);
    assert_eq!(world.get_inherited::<Value>(instance), Some(&Value(1)));
}

#[test]
fn checked_edges_reject_self_cycles_indirect_cycles_and_stale_entities_atomically() {
    let mut world = World::new();
    let base = world.spawn(Value(1));
    let mid = world.spawn_instance_of(base);
    let leaf = world.spawn_instance_of(mid);
    assert_eq!(
        world.try_add_prefab(base, base),
        Err(PrefabError::Cycle(base))
    );
    assert_eq!(
        world.try_add_prefab(base, leaf),
        Err(PrefabError::Cycle(base))
    );
    assert!(world.prefabs_of(base).is_empty());
    let stale = world.spawn(());
    world.despawn(stale);
    assert_eq!(
        world.try_add_prefab(leaf, stale),
        Err(PrefabError::MissingEntity(stale))
    );
    assert_eq!(world.prefabs_of(leaf), &[mid]);
    world.add_prefab(base, leaf);
    assert!(matches!(
        world.validate_prefab_ancestry(leaf),
        Err(PrefabError::Cycle(_))
    ));
    assert_eq!(world.prefab_chain(leaf), vec![mid, base]);
    assert_eq!(world.get_inherited::<Value>(leaf), Some(&Value(1)));
}

#[test]
fn diamonds_are_valid_and_despawn_cleans_incoming_inheritance() {
    let mut world = World::new();
    let base = world.spawn(Value(1));
    let left = world.spawn_instance_of(base);
    let right = world.spawn_instance_of(base);
    let leaf = world.spawn_instance_of(left);
    world.try_add_prefab(leaf, right).unwrap();
    assert_eq!(world.validate_prefab_ancestry(leaf), Ok(()));
    assert_eq!(world.prefab_chain(leaf), vec![left, right, base]);
    world.despawn(base);
    assert_eq!(world.validate_prefab_ancestry(leaf), Ok(()));
    assert_eq!(world.get_inherited::<Value>(leaf), None);
    world.add_prefab(right, base);
    assert_eq!(
        world.validate_prefab_ancestry(leaf),
        Err(PrefabError::MissingEntity(base))
    );
    assert!(!world.is_instance_of(leaf, base));
}

#[test]
fn deep_ancestry_validation_and_resolution_do_not_recurse() {
    let mut world = World::new();
    let base = world.spawn(Value(42));
    let mut leaf = base;
    for _ in 0..8192 {
        leaf = world.spawn_instance_of(leaf);
    }
    assert_eq!(world.validate_prefab_ancestry(leaf), Ok(()));
    assert_eq!(world.prefab_chain(leaf).len(), 8192);
    assert_eq!(world.get_inherited::<Value>(leaf), Some(&Value(42)));
    assert_eq!(
        world.try_add_prefab(base, leaf),
        Err(PrefabError::Cycle(base))
    );
}
