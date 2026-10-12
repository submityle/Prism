//! Behavioural tests for the reflect-driven binary archive (design §16.5 /
//! §24.3, `serialize` feature).

use alloc::vec::Vec;

use prism_reflect::Reflect;

use crate::component::Component;
use crate::world::World;

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
    let landed = target.load_reflect_archive(&bytes).expect("archive decodes");

    assert_eq!(landed.len(), 3, "one record per encoded entity");
    assert_eq!(target.get::<Health>(landed[0]), Some(&Health { hp: 10, shield: 1 }));
    assert_eq!(target.get::<Health>(landed[1]), Some(&Health { hp: 20, shield: 2 }));
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
    let bytes = source.save_reflect_archive([keep, missing]).expect("encodes");

    let mut target = World::new();
    register(&mut target);
    // Only `keep` exists in the target; `missing` must be re-created.
    let keep_here = target.spawn(Health { hp: 99, shield: 99 });
    assert_eq!(keep_here, keep, "ids are deterministic in a fresh world");

    let landed = target.load_reflect_archive(&bytes).expect("decodes");
    assert_eq!(landed[0], keep, "an existing id is reused");
    assert_eq!(target.get::<Health>(keep), Some(&Health { hp: 1, shield: 1 }));
    // `missing` was not live in the target, so the loader had to create an
    // entity; the allocator may hand back that very id or a different one.
    assert!(target.contains(landed[1]), "the record landed on a live entity");
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
        matches!(err, crate::reflect_archive::ReflectArchiveError::UnknownType(_)),
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
    blob.push(2); // current envelope version
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
    use prism_reflect::{DynamicStruct, Migration, Typed, TypeSchema};

    // ---- the build that wrote the save: only V1 exists ----
    let mut old = World::new();
    old.register_reflect_component::<HealthV1>();
    let e = old.spawn(HealthV1 { hp: 7 });
    let bytes = old.save_reflect_archive([e]).expect("encodes");

    // ---- today's build: V2 is live, and the catalogue knows both shapes ----
    let mut new = World::new();
    new.register_reflect_component::<HealthV2>();
    {
        let schemas = new.reflect_schemas_mut();
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
            .expect("a single-version step");
    }

    let landed = new.load_reflect_archive(&bytes).expect("migrates on load");
    assert_eq!(landed.len(), 1);
    assert_eq!(
        new.get::<HealthV2>(landed[0]),
        Some(&HealthV2 { hp: 7, shield: 0 }),
        "the old value must arrive as today's type, with the new field defaulted"
    );
    // The old type is not a component in this world at all.
    assert!(new.get::<HealthV1>(landed[0]).is_none());
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
