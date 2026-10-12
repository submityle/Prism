//! Cross-module integration tests for the scene model (design §5-§7).

use alloc::vec;
use alloc::vec::Vec;

use prism_asset::StableGuid;
use prism_ecs::entity::Entity;

use crate::identity::{LocalEntityId, PersistentId, PersistentIdBuilder};
use crate::overrides::{
    ArchetypeSignature, ComponentTypeId, OverrideSet, OverrideValue, PropertyOverride, StructuralOp,
};
use crate::path::FieldPath;
use crate::plan::{ArchetypeGroup, InstantiationPlan};
use crate::remap::RemapTable;
use crate::template::{NestedSlot, SceneTemplate};

fn entity(index: u32, generation: u32) -> Entity {
    Entity::from_bits((u64::from(generation) << 32) | u64::from(index)).unwrap()
}

fn guid(seed: u128) -> StableGuid {
    StableGuid::from_u128(seed)
}

/// A valid plan with `n` entities in one empty-signature group.
fn plan_of(n: u32) -> InstantiationPlan {
    use alloc::boxed::Box;
    let locals: Box<[LocalEntityId]> = (0..n).map(LocalEntityId::new).collect();
    InstantiationPlan {
        entity_count: n,
        groups: Box::new([ArchetypeGroup::new(
            ArchetypeSignature::default(),
            locals,
            Box::new([]),
        )]),
        entity_valued_fields: Box::new([]),
        hierarchy: Box::new([]),
        reserved_local_base: LocalEntityId::RESERVED_BASE,
    }
}

/// The invariant from design §6.1: instantiating the same template twice yields
/// two independent remap tables; a local id in one instance never resolves to an
/// entity of the other.
#[test]
fn two_instances_of_one_template_have_independent_remaps() {
    let mut a = RemapTable::new();
    let mut b = RemapTable::new();
    for i in 0..4 {
        a.push_base(entity(100 + i, 1));
        b.push_base(entity(200 + i, 1));
    }

    let child = LocalEntityId::new(2);
    assert_eq!(a.get(child), Some(entity(102, 1)));
    assert_eq!(b.get(child), Some(entity(202, 1)));
    assert_ne!(a.get(child), b.get(child), "instances never share entities");
}

/// A nested variant replays override layers bottom-up; an upper layer's
/// structural removal orphans a lower layer's property override, which the
/// replay drops (design §7.3).
#[test]
fn nested_variant_replay_drops_orphans_across_layers() {
    let transform = ComponentTypeId::of::<Vec<u8>>();
    let x = FieldPath::parse("translation.x").unwrap();

    // Base variant layer: overrides target entity 0.
    let mut base_layer = OverrideSet::new();
    base_layer.set_property(PropertyOverride::new(
        LocalEntityId::new(0),
        transform,
        x.clone(),
        OverrideValue::F32(1.0),
    ));

    // Instance layer: removes entity 0 and overrides entity 1.
    let mut instance_layer = OverrideSet::new();
    instance_layer.set_property(PropertyOverride::new(
        LocalEntityId::new(0),
        transform,
        x.clone(),
        OverrideValue::F32(9.0),
    ));
    instance_layer.set_property(PropertyOverride::new(
        LocalEntityId::new(1),
        transform,
        x.clone(),
        OverrideValue::F32(2.0),
    ));
    instance_layer.push_structural(StructuralOp::RemoveEntity {
        local: LocalEntityId::new(0),
    });

    // The upper layer's removal orphans the base layer's override for entity 0;
    // mutating the base layer drops it and reports the key.
    let orphans = base_layer.drop_orphans();
    // No removal in the base layer itself, so nothing drops here...
    assert!(orphans.is_empty());
    // ...but applying the instance layer's removal invalidates the base key.
    let mut replay = base_layer.clone();
    replay.set_property(PropertyOverride::new(
        LocalEntityId::new(0),
        transform,
        x.clone(),
        OverrideValue::F32(9.0),
    ));
    replay.push_structural(StructuralOp::RemoveEntity {
        local: LocalEntityId::new(0),
    });
    let dropped = replay.drop_orphans();
    assert_eq!(dropped.len(), 1, "the removed target's override is orphaned");
    assert_eq!(dropped[0].0, LocalEntityId::new(0));
    assert_eq!(replay.property_count(), 0);

    // Entity 1's override in the instance layer is unaffected.
    assert_eq!(instance_layer.property_count(), 2);
    assert_eq!(instance_layer.removed_entities(), vec![LocalEntityId::new(0)]);
}

/// `PersistentId` is stable regardless of where the entity moves within the
/// template's name space, but distinct per template GUID and per local path
/// (design §6.2).
#[test]
fn persistent_id_is_template_and_path_scoped() {
    let scene = guid(0xAAAA);
    let child_guid = guid(0xBBBB);

    let at_root = PersistentIdBuilder::new()
        .template(scene)
        .local(LocalEntityId::new(1))
        .finish();
    let nested = PersistentIdBuilder::new()
        .template(scene)
        .template(child_guid)
        .local(LocalEntityId::new(1))
        .finish();
    assert_ne!(at_root, nested, "the nested chain changes the id");

    // Same chain + same path is stable across calls.
    let again = PersistentId::derive(&[scene, child_guid], &[LocalEntityId::new(1)]);
    assert_eq!(nested, again);
}

/// End-to-end: a template with a nested slot and a name table, an instance
/// override set, and a remap table compose into a coherent instance description.
#[test]
fn template_instance_overrides_compose() {
    let mut template = SceneTemplate::new(guid(1), plan_of(3));
    template.names.insert("root", LocalEntityId::new(0));
    template.names.insert("camera", LocalEntityId::new(2));
    template.nested = vec![NestedSlot::new(LocalEntityId::new(1), guid(2))].into_boxed_slice();

    let root = template.local_of("root").unwrap();
    assert!(template.contains_local(root));
    assert_eq!(template.nested_at(LocalEntityId::new(1)).unwrap().template, guid(2));

    let mut remap = RemapTable::new();
    for i in 0..template.entity_count() {
        remap.push_base(entity(i, 1));
    }
    assert_eq!(remap.get(root), Some(entity(0, 1)));

    let mut overrides = OverrideSet::new();
    overrides.set_property(PropertyOverride::new(
        root,
        ComponentTypeId::from_raw(7),
        FieldPath::parse("scale.x").unwrap(),
        OverrideValue::F32(2.0),
    ));
    assert_eq!(overrides.property_count(), 1);
    assert!(!overrides.is_empty());
}
