//! Field overrides exercise the public prefab/reflection boundary (§16.3).
#![cfg(feature = "reflect")]

use std::collections::BTreeMap;

use prism_ecs::component::Component;
use prism_ecs::entity::Entity;
use prism_ecs::prefab::reflect::{PrefabComponentOverride, PrefabOverrides, PrefabReflectError};
use prism_ecs::prefab::{IsA, PrefabError};
use prism_ecs::query::Changed;
use prism_ecs::world::World;
use prism_reflect::{FromReflect, Patch, Reflect, StableTypeId};

#[derive(Reflect, Clone, Debug, PartialEq, Eq)]
struct Stats {
    hp: i32,
    mana: i32,
}
impl Component for Stats {}

fn stats(world: &World, entity: Entity, id: StableTypeId) -> Option<Stats> {
    world
        .get_inherited_reflect(entity, id)
        .unwrap()
        .map(|value| Stats::from_reflect(&*value).expect("resolved stats retain their schema"))
}

#[test]
fn fields_track_explicit_ownership_while_live_parent_defaults_change() {
    let mut world = World::new();
    let id = world.register_reflect_component::<Stats>();
    let base = world.spawn(Stats { hp: 10, mana: 20 });
    let instance = world.spawn_instance_of(base);
    let sibling = world.spawn_instance_of(base);

    world
        .override_prefab_field(instance, id, "hp", &99_i32)
        .unwrap();
    assert!(
        world.get::<Stats>(instance).is_none(),
        "no component clone stored"
    );
    let metadata = world.get::<PrefabOverrides>(instance).unwrap();
    let PrefabComponentOverride::Fields(fields) = metadata.component(id).unwrap() else {
        panic!()
    };
    assert_eq!(fields.len(), 1);
    assert_eq!(fields[0].path().segments().len(), 1);
    let Patch::Replace(value) = fields[0].patch() else {
        panic!("field-only patch")
    };
    assert_eq!(value.downcast_ref::<i32>(), Some(&99));
    assert_eq!(
        stats(&world, instance, id),
        Some(Stats { hp: 99, mana: 20 })
    );

    world.insert(base, Stats { hp: 30, mana: 40 });
    assert_eq!(
        stats(&world, instance, id),
        Some(Stats { hp: 99, mana: 40 })
    );
    assert_eq!(stats(&world, sibling, id), Some(Stats { hp: 30, mana: 40 }));
    assert_eq!(world.get::<Stats>(base), Some(&Stats { hp: 30, mana: 40 }));

    assert!(world
        .clear_prefab_field_override(instance, id, ".hp")
        .unwrap());
    assert!(!world
        .clear_prefab_field_override(instance, id, "hp")
        .unwrap());
    assert!(
        !world.has::<PrefabOverrides>(instance),
        "empty metadata is pruned"
    );
    assert_eq!(
        stats(&world, instance, id),
        Some(Stats { hp: 30, mana: 40 })
    );
}

#[test]
fn equal_value_is_an_override_but_unchanged_patch_is_a_true_noop() {
    let mut world = World::new();
    let id = world.register_reflect_component::<Stats>();
    let base = world.spawn(Stats { hp: 10, mana: 20 });
    let instance = world.spawn_instance_of(base);

    world
        .apply_prefab_field_patch(instance, id, "mana", &Patch::Unchanged)
        .unwrap();
    assert!(!world.has::<PrefabOverrides>(instance));
    world
        .override_prefab_field(instance, id, "hp", &10_i32)
        .unwrap();
    world.insert(base, Stats { hp: 100, mana: 200 });
    assert_eq!(
        stats(&world, instance, id),
        Some(Stats { hp: 10, mana: 200 })
    );

    let before = world.change_tick();
    world.set_last_change_tick(before);
    world.increment_change_tick();
    world
        .apply_prefab_field_patch(instance, id, "hp", &Patch::Unchanged)
        .unwrap();
    let changed = world.query_filtered::<Entity, Changed<PrefabOverrides>>();
    assert!(changed.get(&world, instance).is_err());
}

#[test]
fn chained_field_overrides_compose_and_reset_independently() {
    let mut world = World::new();
    let id = world.register_reflect_component::<Stats>();
    let base = world.spawn(Stats { hp: 10, mana: 20 });
    let mid = world.spawn_instance_of(base);
    let leaf = world.spawn_instance_of(mid);
    world.override_prefab_field(mid, id, "hp", &30_i32).unwrap();
    world
        .override_prefab_field(leaf, id, "mana", &40_i32)
        .unwrap();
    world.insert(base, Stats { hp: 50, mana: 60 });
    assert_eq!(stats(&world, mid, id), Some(Stats { hp: 30, mana: 60 }));
    assert_eq!(stats(&world, leaf, id), Some(Stats { hp: 30, mana: 40 }));

    world
        .override_prefab_field(leaf, id, ".hp", &70_i32)
        .unwrap();
    world
        .override_prefab_field(leaf, id, "hp", &80_i32)
        .unwrap();
    assert_eq!(stats(&world, leaf, id), Some(Stats { hp: 80, mana: 40 }));
    assert!(world.clear_prefab_field_override(leaf, id, "hp").unwrap());
    assert_eq!(stats(&world, leaf, id), Some(Stats { hp: 30, mana: 40 }));
    assert!(world.reset_prefab_component(mid, id).unwrap());
    assert_eq!(stats(&world, leaf, id), Some(Stats { hp: 50, mana: 40 }));
    assert!(world.reset_prefab_component(leaf, id).unwrap());
    assert_eq!(stats(&world, leaf, id), Some(Stats { hp: 50, mana: 60 }));
    assert!(!world.reset_prefab_component(leaf, id).unwrap());
}

#[derive(Reflect, Clone, Debug, PartialEq, Eq)]
struct Pair(i32, i32);

#[derive(Reflect, Clone, Debug, PartialEq, Eq)]
enum Mode {
    Named { x: i32, y: i32 },
    Pair(i32, i32),
    Off,
}

#[derive(Reflect, Clone, Debug, PartialEq, Eq)]
struct Config {
    stats: Stats,
    pair: Pair,
    samples: Vec<i32>,
    weights: [i32; 2],
    labels: BTreeMap<String, i32>,
    mode: Mode,
}
impl Component for Config {}

fn config() -> Config {
    Config {
        stats: Stats { hp: 1, mana: 2 },
        pair: Pair(3, 4),
        samples: vec![5, 6, 7],
        weights: [8, 9],
        labels: BTreeMap::from([("a".into(), 10), ("b".into(), 11)]),
        mode: Mode::Named { x: 12, y: 13 },
    }
}

fn resolved_config(world: &World, entity: Entity, id: StableTypeId) -> Config {
    Config::from_reflect(&*world.get_inherited_reflect(entity, id).unwrap().unwrap()).unwrap()
}

#[test]
fn nested_tuple_array_list_map_and_enum_fields_preserve_unselected_defaults() {
    let mut world = World::new();
    let id = world.register_reflect_component::<Config>();
    let base = world.spawn(config());
    let instance = world.spawn_instance_of(base);
    for path in [
        "stats.hp",
        "pair#0",
        "samples[1]",
        "weights[0]",
        "labels['a']",
        "mode.x",
    ] {
        world
            .override_prefab_field(instance, id, path, &100_i32)
            .unwrap();
    }
    let mut parent = config();
    parent.stats = Stats { hp: 21, mana: 22 };
    parent.pair = Pair(23, 24);
    parent.samples = vec![25, 26, 27, 28];
    parent.weights = [29, 30];
    parent.labels = BTreeMap::from([("a".into(), 31), ("b".into(), 32)]);
    parent.mode = Mode::Named { x: 33, y: 34 };
    world.insert(base, parent.clone());
    let mut expected = parent;
    expected.stats.hp = 100;
    expected.pair.0 = 100;
    expected.samples[1] = 100;
    expected.weights[0] = 100;
    expected.labels.insert("a".into(), 100);
    expected.mode = Mode::Named { x: 100, y: 34 };
    assert_eq!(resolved_config(&world, instance, id), expected);
    assert!(!world.has::<Config>(instance));
}

#[test]
fn selecting_a_subtree_replaces_collections_exactly_and_can_switch_enum_variants() {
    let mut world = World::new();
    let id = world.register_reflect_component::<Config>();
    let base = world.spawn(config());
    let instance = world.spawn_instance_of(base);
    world
        .override_prefab_field(instance, id, "samples", &vec![90_i32])
        .unwrap();
    let labels = BTreeMap::from([("c".to_string(), 91_i32)]);
    world
        .override_prefab_field(instance, id, "labels", &labels)
        .unwrap();
    world
        .override_prefab_field(instance, id, "mode", &Mode::Pair(92, 93))
        .unwrap();
    let resolved = resolved_config(&world, instance, id);
    assert_eq!(resolved.samples, vec![90]);
    assert_eq!(resolved.labels, labels);
    assert_eq!(resolved.mode, Mode::Pair(92, 93));
    world.get_mut::<Config>(base).unwrap().samples.push(94);
    assert_eq!(resolved_config(&world, instance, id).samples, vec![90]);

    world
        .clear_prefab_field_override(instance, id, "mode")
        .unwrap();
    world
        .override_prefab_field(instance, id, "mode", &Mode::Off)
        .unwrap();
    assert_eq!(resolved_config(&world, instance, id).mode, Mode::Off);
}

#[test]
fn a_structural_patch_is_applied_to_the_selected_field_before_it_is_pinned() {
    let mut world = World::new();
    let id = world.register_reflect_component::<Config>();
    let base = world.spawn(config());
    let instance = world.spawn_instance_of(base);
    let patch = Patch::Struct(vec![("hp".into(), Patch::Replace(Box::new(50_i32)))]);
    world
        .apply_prefab_field_patch(instance, id, "stats", &patch)
        .unwrap();
    world.get_mut::<Config>(base).unwrap().stats = Stats { hp: 60, mana: 70 };
    assert_eq!(
        resolved_config(&world, instance, id).stats,
        Stats { hp: 50, mana: 2 }
    );
}

#[test]
fn rejected_edits_are_atomic_and_overlapping_paths_are_explicit() {
    let mut world = World::new();
    let id = world.register_reflect_component::<Config>();
    let base = world.spawn(config());
    let instance = world.spawn_instance_of(base);
    for path in ["", "samples[", "missing", "samples[99]"] {
        assert!(world
            .override_prefab_field(instance, id, path, &1_i32)
            .is_err());
    }
    assert!(matches!(
        world.override_prefab_field(instance, id, "stats.hp", &true),
        Err(PrefabReflectError::FieldTypeMismatch { .. })
    ));
    let bad = Patch::Struct(vec![
        ("hp".into(), Patch::Replace(Box::new(90_i32))),
        ("missing".into(), Patch::Replace(Box::new(91_i32))),
    ]);
    assert!(matches!(
        world.apply_prefab_field_patch(instance, id, "stats", &bad),
        Err(PrefabReflectError::Patch(_))
    ));
    assert!(!world.has::<PrefabOverrides>(instance));
    assert_eq!(resolved_config(&world, instance, id), config());
    assert_eq!(world.get::<Config>(base), Some(&config()));

    world
        .override_prefab_field(instance, id, "stats.hp", &80_i32)
        .unwrap();
    assert!(matches!(
        world.override_prefab_field(instance, id, "stats", &Stats { hp: 3, mana: 4 }),
        Err(PrefabReflectError::OverlappingField(_))
    ));
    assert_eq!(
        resolved_config(&world, instance, id).stats,
        Stats { hp: 80, mana: 2 }
    );
    world
        .clear_prefab_field_override(instance, id, "stats.hp")
        .unwrap();
    world
        .override_prefab_field(instance, id, "stats", &Stats { hp: 3, mana: 4 })
        .unwrap();
    assert!(matches!(
        world.override_prefab_field(instance, id, "stats.mana", &5_i32),
        Err(PrefabReflectError::OverlappingField(_))
    ));
}

#[test]
fn absence_is_dormant_and_explicit_removal_is_inherited_until_reset() {
    let mut world = World::new();
    let id = world.register_reflect_component::<Stats>();
    let base = world.spawn(Stats { hp: 10, mana: 20 });
    let mid = world.spawn_instance_of(base);
    let leaf = world.spawn_instance_of(mid);
    world
        .override_prefab_field(leaf, id, "hp", &50_i32)
        .unwrap();
    world.remove::<Stats>(base);
    assert_eq!(stats(&world, leaf, id), None);
    assert!(world.has::<PrefabOverrides>(leaf));
    assert!(
        matches!(world.override_prefab_field(leaf, id, "mana", &1_i32),
        Err(PrefabReflectError::MissingComponent(found)) if found == id)
    );
    world.insert(base, Stats { hp: 30, mana: 40 });
    assert_eq!(stats(&world, leaf, id), Some(Stats { hp: 50, mana: 40 }));

    world.remove_prefab_component(mid, id).unwrap();
    assert_eq!(stats(&world, leaf, id), None);
    world.insert(base, Stats { hp: 60, mana: 70 });
    assert_eq!(stats(&world, mid, id), None);
    assert_eq!(stats(&world, leaf, id), None);
    world.reset_prefab_component(mid, id).unwrap();
    assert_eq!(stats(&world, leaf, id), Some(Stats { hp: 50, mana: 70 }));
    world.remove_prefab_component(leaf, id).unwrap();
    let PrefabComponentOverride::Removed = world
        .get::<PrefabOverrides>(leaf)
        .unwrap()
        .component(id)
        .unwrap()
    else {
        panic!()
    };
    world.reset_prefab_component(leaf, id).unwrap();
    assert_eq!(stats(&world, leaf, id), Some(Stats { hp: 60, mana: 70 }));
}

#[test]
fn concrete_component_overrides_and_reset_keep_component_level_semantics() {
    let mut world = World::new();
    let id = world.register_reflect_component::<Stats>();
    let base = world.spawn(Stats { hp: 1, mana: 2 });
    let mid = world.spawn_instance_of(base);
    world.override_prefab_field(mid, id, "hp", &3_i32).unwrap();
    let leaf = world.spawn_instance_of(mid);
    world.insert(leaf, Stats { hp: 4, mana: 5 });
    assert_eq!(stats(&world, leaf, id), Some(Stats { hp: 4, mana: 5 }));
    assert_eq!(
        world.get_inherited::<Stats>(leaf),
        Some(&Stats { hp: 4, mana: 5 })
    );
    world
        .override_prefab_field(leaf, id, "mana", &6_i32)
        .unwrap();
    assert_eq!(stats(&world, leaf, id), Some(Stats { hp: 4, mana: 6 }));
    world.reset_prefab_component(leaf, id).unwrap();
    assert!(!world.has::<Stats>(leaf));
    assert_eq!(stats(&world, leaf, id), Some(Stats { hp: 3, mana: 2 }));
    world.insert(leaf, Stats { hp: 7, mana: 8 });
    world.remove_prefab_component(leaf, id).unwrap();
    assert!(!world.has::<Stats>(leaf));
    assert_eq!(stats(&world, leaf, id), None);
}

#[test]
fn invalidated_container_paths_report_errors_and_can_be_cleared() {
    let mut world = World::new();
    let id = world.register_reflect_component::<Config>();
    let base = world.spawn(config());
    let mid = world.spawn_instance_of(base);
    world
        .override_prefab_field(mid, id, "samples[2]", &100_i32)
        .unwrap();
    world.get_mut::<Config>(base).unwrap().samples.truncate(1);
    assert!(matches!(
        world.get_inherited_reflect(mid, id),
        Err(PrefabReflectError::MissingField(_))
    ));
    assert!(world
        .clear_prefab_field_override(mid, id, "samples[2]")
        .unwrap());
    assert_eq!(resolved_config(&world, mid, id).samples, vec![5]);

    world
        .override_prefab_field(mid, id, "mode.x", &100_i32)
        .unwrap();
    world.get_mut::<Config>(base).unwrap().mode = Mode::Off;
    assert!(matches!(
        world.get_inherited_reflect(mid, id),
        Err(PrefabReflectError::MissingField(_))
    ));
    world.reset_prefab_component(mid, id).unwrap();
    assert_eq!(resolved_config(&world, mid, id).mode, Mode::Off);
}

#[test]
fn nearest_first_resolution_handles_multiple_parents_and_diamonds() {
    let mut world = World::new();
    let id = world.register_reflect_component::<Stats>();
    let deep = world.spawn(Stats { hp: 1, mana: 2 });
    let left = world.spawn_instance_of(deep);
    let right = world.spawn_instance_of(deep);
    let root = world.spawn_instance_of(left);
    world.try_add_prefab(root, right).unwrap();
    world.override_prefab_field(left, id, "hp", &3_i32).unwrap();
    world
        .override_prefab_field(right, id, "hp", &4_i32)
        .unwrap();
    world
        .override_prefab_field(right, id, "mana", &5_i32)
        .unwrap();
    assert!(world.validate_prefab_ancestry(root).is_ok());
    assert_eq!(world.prefab_chain(root), vec![left, right, deep]);
    assert_eq!(stats(&world, root, id), Some(Stats { hp: 3, mana: 5 }));

    world.insert(right, Stats { hp: 6, mana: 7 });
    assert_eq!(
        world.get_inherited::<Stats>(root),
        Some(&Stats { hp: 6, mana: 7 })
    );
    assert_eq!(stats(&world, root, id), Some(Stats { hp: 3, mana: 5 }));
    world.reset_prefab_component(left, id).unwrap();
    assert_eq!(stats(&world, root, id), Some(Stats { hp: 4, mana: 5 }));
    world.remove_prefab_component(left, id).unwrap();
    assert_eq!(
        stats(&world, root, id),
        None,
        "nearest first parent removal wins"
    );
}

#[test]
fn cycles_and_dangling_ancestors_fail_without_mutation() {
    let mut world = World::new();
    let id = world.register_reflect_component::<Stats>();
    let base = world.spawn(Stats { hp: 1, mana: 2 });
    let instance = world.spawn_instance_of(base);
    assert_eq!(
        world.try_add_prefab(base, instance),
        Err(PrefabError::Cycle(base))
    );
    assert_eq!(
        world.try_add_prefab(base, base),
        Err(PrefabError::Cycle(base))
    );
    assert!(world.prefabs_of(base).is_empty());

    world.add_prefab(base, instance); // Low-level escape hatch deliberately bypasses checks.
    assert!(matches!(
        world.get_inherited_reflect(instance, id),
        Err(PrefabReflectError::Ancestry(PrefabError::Cycle(_)))
    ));
    assert!(world
        .override_prefab_field(instance, id, "hp", &3_i32)
        .is_err());
    assert!(!world.has::<PrefabOverrides>(instance));
    world.remove_relation::<IsA>(base, instance);
    assert_eq!(stats(&world, instance, id), Some(Stats { hp: 1, mana: 2 }));

    let stale = world.spawn(());
    world.despawn(stale);
    assert_eq!(
        world.try_add_prefab(instance, stale),
        Err(PrefabError::MissingEntity(stale))
    );
    world.add_prefab(base, stale);
    assert!(
        matches!(world.get_inherited_reflect(instance, id), Err(PrefabReflectError::Ancestry(PrefabError::MissingEntity(e))) if e == stale)
    );
}

#[test]
fn unregistered_ids_and_dead_entities_cannot_create_override_metadata() {
    let mut world = World::new();
    let entity = world.spawn(());
    let id = StableTypeId::of_type::<Stats>();
    assert!(
        matches!(world.override_prefab_field(entity, id, "hp", &1_i32), Err(PrefabReflectError::UnregisteredType(found)) if found == id)
    );
    assert!(!world.has::<PrefabOverrides>(entity));
    world.register_reflect_component::<Stats>();
    assert_eq!(stats(&world, entity, id), None);
    world.despawn(entity);
    assert!(
        matches!(world.override_prefab_field(entity, id, "hp", &1_i32), Err(PrefabReflectError::Ancestry(PrefabError::MissingEntity(e))) if e == entity)
    );
    assert!(world.reset_prefab_component(entity, id).is_err());
    assert!(world.clear_prefab_field_override(entity, id, "hp").is_err());
    assert!(world.remove_prefab_component(entity, id).is_err());
    assert!(!world.has::<PrefabOverrides>(entity));
}

#[test]
fn reparenting_and_despawn_use_current_ancestry_and_generation() {
    let mut world = World::new();
    let id = world.register_reflect_component::<Stats>();
    let first = world.spawn(Stats { hp: 1, mana: 2 });
    let second = world.spawn(Stats { hp: 3, mana: 4 });
    let instance = world.spawn_instance_of(first);
    world
        .override_prefab_field(instance, id, "hp", &5_i32)
        .unwrap();
    world.remove_relation::<IsA>(instance, first);
    assert_eq!(stats(&world, instance, id), None);
    world.try_add_prefab(instance, second).unwrap();
    assert_eq!(stats(&world, instance, id), Some(Stats { hp: 5, mana: 4 }));
    world.despawn(second);
    assert_eq!(stats(&world, instance, id), None);
    world.despawn(instance);
    let replacement = world.spawn_instance_of(first);
    assert_ne!(replacement, instance);
    assert!(!world.has::<PrefabOverrides>(replacement));
    assert_eq!(
        stats(&world, replacement, id),
        Some(Stats { hp: 1, mana: 2 })
    );
    assert!(world.get_inherited::<Stats>(instance).is_none());
}

#[test]
fn stable_schema_ids_are_independent_of_local_registration_order_and_records_replay() {
    let mut first = World::new();
    let id = first.register_reflect_component::<Stats>();
    first.register_reflect_component::<Config>();
    let base = first.spawn(Stats { hp: 1, mana: 2 });
    let instance = first.spawn_instance_of(base);
    first
        .override_prefab_field(instance, id, "hp", &3_i32)
        .unwrap();

    let mut second = World::new();
    second.register_reflect_component::<Config>();
    assert_eq!(second.register_reflect_component::<Stats>(), id);
    assert_ne!(
        first.reflect_component_id(id),
        second.reflect_component_id(id)
    );
    let other_base = second.spawn(Stats { hp: 4, mana: 5 });
    let other = second.spawn_instance_of(other_base);
    for (type_id, policy) in first.get::<PrefabOverrides>(instance).unwrap().components() {
        let PrefabComponentOverride::Fields(fields) = policy else {
            panic!()
        };
        for field in fields {
            assert_eq!(
                field.path(),
                &prism_reflect::ParsedPath::parse("hp").unwrap()
            );
            second
                .apply_prefab_field_patch(other, type_id, field.path_str(), field.patch())
                .unwrap();
        }
    }
    assert_eq!(stats(&second, other, id), Some(Stats { hp: 3, mana: 5 }));
}

#[test]
fn resolved_reads_do_not_dirty_parent_and_override_edits_use_metadata_change_detection() {
    let mut world = World::new();
    let id = world.register_reflect_component::<Stats>();
    let base = world.spawn(Stats { hp: 1, mana: 2 });
    let instance = world.spawn_instance_of(base);
    world
        .override_prefab_field(instance, id, "hp", &3_i32)
        .unwrap();
    let before = world.change_tick();
    world.set_last_change_tick(before);
    world.increment_change_tick();
    stats(&world, instance, id);
    let changed = world.query_filtered::<Entity, Changed<Stats>>();
    assert!(changed.get(&world, base).is_err());
    let metadata_changed = world.query_filtered::<Entity, Changed<PrefabOverrides>>();
    assert!(metadata_changed.get(&world, instance).is_err());
    world
        .override_prefab_field(instance, id, "hp", &4_i32)
        .unwrap();
    assert_eq!(metadata_changed.get(&world, instance), Ok(instance));
    assert!(changed.get(&world, base).is_err());
}

#[test]
fn sparse_override_metadata_is_snapshot_cloneable() {
    let mut world = World::new();
    let id = world.register_reflect_component::<Stats>();
    world.register_snapshot_component::<Stats>();
    world.register_snapshot_component::<PrefabOverrides>();
    let base = world.spawn(Stats { hp: 1, mana: 2 });
    let instance = world.spawn_instance_of(base);
    world
        .override_prefab_field(instance, id, "hp", &3_i32)
        .unwrap();
    let snapshot = world.snapshot();
    world
        .override_prefab_field(instance, id, "hp", &4_i32)
        .unwrap();
    world.insert(base, Stats { hp: 5, mana: 6 });
    world.restore(&snapshot);
    assert_eq!(stats(&world, instance, id), Some(Stats { hp: 3, mana: 2 }));
}

#[test]
fn named_enum_aliases_share_one_override_and_clear_even_after_invalidation() {
    let mut world = World::new();
    let id = world.register_reflect_component::<Config>();
    let base = world.spawn(config());
    let instance = world.spawn_instance_of(base);
    world
        .override_prefab_field(instance, id, "mode.x", &40_i32)
        .unwrap();
    world
        .override_prefab_field(instance, id, "mode#0", &50_i32)
        .unwrap();
    let PrefabComponentOverride::Fields(fields) = world
        .get::<PrefabOverrides>(instance)
        .unwrap()
        .component(id)
        .unwrap()
    else {
        panic!()
    };
    assert_eq!(fields.len(), 1);
    assert_eq!(
        resolved_config(&world, instance, id).mode,
        Mode::Named { x: 50, y: 13 }
    );
    assert!(world
        .clear_prefab_field_override(instance, id, "mode.x")
        .unwrap());
    world
        .override_prefab_field(instance, id, "mode#0", &60_i32)
        .unwrap();
    world.get_mut::<Config>(base).unwrap().mode = Mode::Off;
    assert!(world.get_inherited_reflect(instance, id).is_err());
    assert!(world
        .clear_prefab_field_override(instance, id, "mode#0")
        .unwrap());
    assert_eq!(resolved_config(&world, instance, id).mode, Mode::Off);
}

#[test]
fn nearer_subtree_masks_stale_ancestor_element_paths() {
    let mut world = World::new();
    let id = world.register_reflect_component::<Config>();
    let base = world.spawn(config());
    let mid = world.spawn_instance_of(base);
    let leaf = world.spawn_instance_of(mid);
    world
        .override_prefab_field(mid, id, "samples[2]", &90_i32)
        .unwrap();
    world
        .override_prefab_field(leaf, id, "samples", &vec![80_i32])
        .unwrap();
    world.get_mut::<Config>(base).unwrap().samples.clear();
    assert!(world.get_inherited_reflect(mid, id).is_err());
    assert_eq!(resolved_config(&world, leaf, id).samples, vec![80]);
    world
        .clear_prefab_field_override(leaf, id, "samples")
        .unwrap();
    assert!(world.get_inherited_reflect(leaf, id).is_err());
    world
        .clear_prefab_field_override(mid, id, "samples[2]")
        .unwrap();
    assert!(resolved_config(&world, leaf, id).samples.is_empty());
}

#[test]
fn malformed_dynamic_fields_and_wrong_element_patches_are_rejected_atomically() {
    let mut world = World::new();
    let id = world.register_reflect_component::<Config>();
    let base = world.spawn(config());
    let instance = world.spawn_instance_of(base);
    let mut bad = prism_reflect::DynamicStruct::new();
    bad.set_represented_type_name(core::any::type_name::<Stats>());
    bad.insert("hp", 90_i32);
    assert!(matches!(
        world.override_prefab_field(instance, id, "stats", &bad),
        Err(PrefabReflectError::InvalidFieldShape(_))
    ));
    let bad_list = Patch::List {
        modified: vec![],
        appended: vec![Box::new(true)],
    };
    assert!(matches!(
        world.apply_prefab_field_patch(instance, id, "samples", &bad_list),
        Err(PrefabReflectError::FieldTypeMismatch { .. })
    ));
    world.get_mut::<Config>(base).unwrap().samples.clear();
    assert!(world
        .apply_prefab_field_patch(instance, id, "samples", &bad_list)
        .is_err());
    assert!(!world.has::<PrefabOverrides>(instance));
    assert!(!world.has::<Config>(instance));
    assert!(resolved_config(&world, instance, id).samples.is_empty());
}
