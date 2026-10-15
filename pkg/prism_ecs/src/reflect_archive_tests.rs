//! Behavioural tests for the reflect-driven binary archive (design §16.5 /
//! §24.3, `serialize` feature).

use alloc::vec::Vec;

use prism_reflect::Reflect;

use crate::component::Component;
use crate::world::World;

// An independent V2 envelope fixture. Using today's writer here would only
// exercise V3 and could silently lose compatibility with existing save files.
fn legacy_v2(entity: crate::entity::Entity, value: &dyn Reflect) -> Vec<u8> {
    let name = value.type_name().as_bytes();
    let blob = prism_reflect::to_binary(value).unwrap();
    let mut bytes = b"PECA".to_vec();
    bytes.push(2);
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&entity.to_bits().to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&(name.len() as u16).to_le_bytes());
    bytes.extend_from_slice(name);
    bytes.extend_from_slice(
        &prism_reflect::StableTypeId::of_path(value.type_name())
            .value()
            .to_le_bytes(),
    );
    bytes.extend_from_slice(&(blob.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&blob);
    bytes
}

/// A reflected component with two integer fields.
#[derive(Reflect, Clone, Copy, Debug, PartialEq, Eq)]
struct Health {
    hp: i32,
    shield: i32,
}
impl Component for Health {}

/// A second reflected component, so records can carry more than one type.
#[derive(Reflect, Clone, Copy, Debug, PartialEq, Eq)]
struct Speed(i32);
impl Component for Speed {}

/// Register the pair on `world` and return their stable ids.
fn register(world: &mut World) -> (prism_reflect::StableTypeId, prism_reflect::StableTypeId) {
    (
        world.register_reflect_component::<Health>(),
        world.register_reflect_component::<Speed>(),
    )
}

/// The **old build's** shape of the health component (no `shield` yet).
#[derive(Reflect, Clone, Copy, Debug, PartialEq, Eq)]
struct HealthV1 {
    hp: i32,
}
impl Component for HealthV1 {}

/// The **new build's** shape: `shield` was added.
#[derive(Reflect, Clone, Copy, Debug, PartialEq, Eq)]
struct HealthV2 {
    hp: i32,
    shield: i32,
}
impl Component for HealthV2 {}

fn register_health_migration(world: &mut World) {
    use prism_reflect::{DynamicStruct, Migration, TypeSchema, Typed};
    world.register_reflect_component::<HealthV2>();
    let schemas = world.reflect_schemas_mut();
    schemas.register_version(
        "Health",
        TypeSchema::new(
            core::any::type_name::<HealthV1>(),
            1,
            <HealthV1 as Typed>::type_info(),
        ),
    );
    schemas.register_version(
        "Health",
        TypeSchema::new(
            core::any::type_name::<HealthV2>(),
            2,
            <HealthV2 as Typed>::type_info(),
        ),
    );
    schemas
        .register_migration(
            "Health",
            Migration::new(1, 2, |value: &mut DynamicStruct| {
                value.insert("shield", 0i32);
                Ok(())
            }),
        )
        .unwrap();
}

#[test]
fn save_then_load_into_an_empty_world_round_trips_every_component() {
    let mut source = World::new();
    register(&mut source);
    let a = source.spawn(Health { hp: 10, shield: 1 });
    let b = source.spawn((Health { hp: 20, shield: 2 }, Speed(7)));
    let c = source.spawn(Speed(9));

    let bytes = source
        .save_reflect_archive([a, b, c])
        .expect("archive encodes");

    // A fresh world with the same registrations reconstructs the records.
    let mut target = World::new();
    register(&mut target);
    let landed = target
        .load_reflect_archive(&bytes)
        .expect("archive decodes");

    assert_eq!(landed.len(), 3, "one record per encoded entity");
    assert_eq!(
        target.get::<Health>(landed[0]),
        Some(&Health { hp: 10, shield: 1 })
    );
    assert_eq!(
        target.get::<Health>(landed[1]),
        Some(&Health { hp: 20, shield: 2 })
    );
    assert_eq!(target.get::<Speed>(landed[1]), Some(&Speed(7)));
    assert_eq!(target.get::<Speed>(landed[2]), Some(&Speed(9)));
    assert!(target.get::<Health>(landed[2]).is_none());
}

#[test]
fn loading_reuses_a_live_id_and_spawns_for_a_missing_one() {
    let mut source = World::new();
    register(&mut source);
    let keep = source.spawn(Health { hp: 1, shield: 1 });
    let missing = source.spawn(Health { hp: 2, shield: 2 });
    let bytes = source
        .save_reflect_archive([keep, missing])
        .expect("encodes");

    let mut target = World::new();
    register(&mut target);
    // Only `keep` exists in the target; `missing` must be re-created.
    let keep_here = target.spawn(Health { hp: 99, shield: 99 });
    assert_eq!(keep_here, keep, "ids are deterministic in a fresh world");

    let landed = target.load_reflect_archive(&bytes).expect("decodes");
    assert_eq!(landed[0], keep, "an existing id is reused");
    assert_eq!(
        target.get::<Health>(keep),
        Some(&Health { hp: 1, shield: 1 })
    );
    // `missing` was not live in the target, so the loader had to create an
    // entity; the allocator may hand back that very id or a different one.
    assert!(
        target.contains(landed[1]),
        "the record landed on a live entity"
    );
    assert_ne!(landed[0], landed[1], "the two records stay distinct");
    assert_eq!(
        target.get::<Health>(landed[1]),
        Some(&Health { hp: 2, shield: 2 })
    );
    let _ = missing;
}

#[test]
fn archives_are_byte_deterministic_and_skip_empty_records() {
    let mut world = World::new();
    register(&mut world);
    let a = world.spawn((Health { hp: 1, shield: 2 }, Speed(3)));
    let empty = world.spawn(());

    let first = world.save_reflect_archive([a, empty]).expect("encodes");
    let second = world.save_reflect_archive([a, empty]).expect("encodes");
    assert_eq!(first, second, "the same world must produce the same bytes");

    // Only the entity with reflected components is recorded.
    let mut target = World::new();
    register(&mut target);
    let landed = target.load_reflect_archive(&first).expect("decodes");
    assert_eq!(landed.len(), 1);
}

#[test]
fn an_unregistered_type_in_the_target_world_is_rejected() {
    let mut source = World::new();
    register(&mut source);
    let e = source.spawn((Health { hp: 1, shield: 1 }, Speed(2)));
    let bytes = source.save_reflect_archive([e]).expect("encodes");

    // The target knows `Health` but never registered `Speed`.
    let mut target = World::new();
    let _health = target.register_reflect_component::<Health>();
    let err = target
        .load_reflect_archive(&bytes)
        .expect_err("unknown component type must fail the load");
    assert!(
        matches!(
            err,
            crate::reflect_archive::ReflectArchiveError::UnknownType(_)
        ),
        "expected UnknownType, got {err:?}"
    );
}

#[test]
fn malformed_archives_are_rejected() {
    let mut world = World::new();
    register(&mut world);

    assert!(matches!(
        world.load_reflect_archive(b"NOPE"),
        Err(crate::reflect_archive::ReflectArchiveError::BadMagic)
    ));

    let good = {
        let e = world.spawn(Health { hp: 1, shield: 2 });
        world.save_reflect_archive([e]).expect("encodes")
    };
    let truncated: Vec<u8> = good[..good.len() - 1].to_vec();
    assert!(matches!(
        world.load_reflect_archive(&truncated),
        Err(crate::reflect_archive::ReflectArchiveError::Truncated)
    ));

    // A version bump must be refused rather than misread.
    let mut wrong_version = good.clone();
    wrong_version[4] = 0xFF;
    assert!(matches!(
        world.load_reflect_archive(&wrong_version),
        Err(crate::reflect_archive::ReflectArchiveError::UnsupportedVersion(0xFF))
    ));
}

/// A tiny hostile blob must not be able to make the loader reserve gigabytes:
/// the entity count is taken from the blob, so the pre-allocation is bounded by
/// what the remaining bytes could actually hold.
#[test]
fn an_absurd_entity_count_is_rejected_without_huge_allocation() {
    let mut world = World::new();
    register(&mut world);

    let mut blob = Vec::new();
    blob.extend_from_slice(b"PECA");
    blob.push(2); // legacy envelope version
    blob.extend_from_slice(&u32::MAX.to_le_bytes()); // entity_count
                                                     // No records at all: the first read runs out of bytes immediately.
    assert!(matches!(
        world.load_reflect_archive(&blob),
        Err(crate::reflect_archive::ReflectArchiveError::Truncated)
    ));
}

/// A save written before a component gained a field must still load: the archive
/// names the concrete type that wrote it, the schema catalogue resolves that to
/// a version, the blob is decoded against *that* shape, and the registered
/// migration chain upgrades it before it meets today's type.
#[test]
fn an_older_schema_version_is_migrated_on_load() {
    // ---- the build that wrote the save: only V1 exists ----
    let mut old = World::new();
    old.register_reflect_component::<HealthV1>();
    let e = old.spawn(HealthV1 { hp: 7 });
    let bytes = old.save_reflect_archive([e]).expect("encodes");

    // ---- today's build: V2 is live, and the catalogue knows both shapes ----
    let mut new = World::new();
    register_health_migration(&mut new);

    for archive in [legacy_v2(e, &HealthV1 { hp: 7 }), bytes] {
        let landed = new
            .load_reflect_archive(&archive)
            .expect("migrates on load");
        assert_eq!(landed.len(), 1);
        assert_eq!(
            new.get::<HealthV2>(landed[0]),
            Some(&HealthV2 { hp: 7, shield: 0 }),
            "both V2 and V3 preserve the concrete-component migration chain"
        );
        assert!(new.get::<HealthV1>(landed[0]).is_none());
    }
}

#[test]
fn migrated_components_can_supply_bases_alongside_other_prefab_policies() {
    let mut old = World::new();
    old.register_reflect_component::<HealthV1>();
    let speed_id = old.register_reflect_component::<Speed>();
    let base = old.spawn((HealthV1 { hp: 7 }, Speed(2)));
    let child = old.spawn_instance_of(base);
    old.override_prefab_field(child, speed_id, "#0", &9i32)
        .unwrap();
    let bytes = old.save_reflect_archive([child]).unwrap();
    let mut new = World::new();
    register_health_migration(&mut new);
    new.register_reflect_component::<Speed>();
    let landed = new.load_reflect_archive(&bytes).unwrap();
    assert_eq!(
        new.get::<HealthV2>(landed[1]),
        Some(&HealthV2 { hp: 7, shield: 0 })
    );
    let speed = new
        .get_inherited_reflect(landed[0], speed_id)
        .unwrap()
        .unwrap();
    assert_eq!(
        <Speed as prism_reflect::FromReflect>::from_reflect(&*speed),
        Some(Speed(9))
    );
    // Migrated represented names also have to work with callbacks and re-save.
    let no_op: super::reflect_archive::ReflectArchiveRemapFn = |_, _, _, _| Ok(());
    let landed = new
        .load_reflect_archive_with_options(
            &bytes,
            super::reflect_archive::ReflectArchiveLoadOptions {
                remap: Some(no_op),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(new.save_reflect_archive([landed[0].1]).is_ok());
}

#[test]
fn field_ownership_is_not_silently_migrated_with_a_different_component_schema() {
    let mut old = World::new();
    let id = old.register_reflect_component::<HealthV1>();
    let base = old.spawn(HealthV1 { hp: 7 });
    let child = old.spawn_instance_of(base);
    old.override_prefab_field(child, id, "hp", &9i32).unwrap();
    let bytes = old.save_reflect_archive([child]).unwrap();
    let mut new = World::new();
    register_health_migration(&mut new);
    assert!(matches!(new.load_reflect_archive(&bytes),
        Err(super::reflect_archive::ReflectArchiveError::PrefabSchemaMismatch(got)) if got == id));
    assert_eq!(new.entity_count(), 0);
}

/// Without a registered catalogue, the same old-shape save cannot be matched to
/// today's component — the loader must say so rather than guess.
#[test]
fn an_unknown_old_shape_is_rejected() {
    let mut old = World::new();
    old.register_reflect_component::<HealthV1>();
    let e = old.spawn(HealthV1 { hp: 1 });
    let bytes = old.save_reflect_archive([e]).expect("encodes");

    // Today's world knows V2 but nothing maps V1's name to it.
    let mut new = World::new();
    new.register_reflect_component::<HealthV2>();
    assert!(matches!(
        new.load_reflect_archive(&bytes),
        Err(crate::reflect_archive::ReflectArchiveError::UnknownType(_))
    ));
}
