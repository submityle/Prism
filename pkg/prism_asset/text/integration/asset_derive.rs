//! End-to-end derive contracts, compiled with only core + alloc in the consumer.
#![no_std]

extern crate alloc;

use alloc::{boxed::Box, sync::Arc, vec, vec::Vec};
use prism_asset::prelude::*;

#[derive(Asset)]
#[asset(type_name = "derive_test::Image")]
struct Image;

#[derive(Asset)]
struct Material {
    direct: Handle<Image>,
    nested: Option<Vec<Option<Vec<Handle<Image>>>>>,
    soft: Vec<Option<SoftHandle<Image>>>,
    mixed: (
        u32,
        [Option<SoftHandle<Image>>; 2],
        Vec<Option<Handle<Image>>>,
    ),
    owned: Box<Option<Handle<Image>>>,
    shared: Arc<Option<SoftHandle<Image>>>,
    borrowed: &'static [SoftHandle<Image>],
    erased: UntypedHandle,
    weak: WeakHandle<Image>,
    erased_weak: UntypedWeakHandle,
    #[asset(skip)]
    ignored: Handle<Image>,
    tint: [f32; 4],
}

#[derive(Asset)]
struct TupleAsset(Option<Handle<Image>>, Vec<Option<SoftHandle<Image>>>, u8);

#[derive(Asset)]
enum Choice {
    Unit,
    Tuple(u8, Option<Handle<Image>>, Vec<SoftHandle<Image>>),
    Named {
        layers: Option<Vec<Handle<Image>>>,
        label: &'static str,
    },
}

// An uninhabited asset is legal: its visitor is an exhaustive empty match.
#[derive(Asset)]
enum Never {}

type ImageAlias = Option<Vec<Handle<Image>>>;

#[derive(VisitAssetDependencies)]
struct NestedGroup {
    image: SoftHandle<Image>,
}

#[derive(Asset)]
struct Composite {
    #[dependency]
    alias: ImageAlias,
    #[dependency]
    records: Vec<Option<NestedGroup>>,
    #[dependency]
    next: Option<Box<Composite>>,
}

mod first {
    #[derive(prism_asset::Asset)]
    pub(super) struct SameName;
}

mod second {
    #[derive(prism_asset::Asset)]
    pub(super) struct SameName;
}

/// Every visible handle is visited in declaration/element order without cloning.
#[test]
fn automatic_nested_traversal_preserves_identity_order_and_reference_counts() {
    let mut images = Assets::new();
    let first = images.insert(Image);
    let second = images.reserve();
    let soft = SoftHandle::<Image>::from_path("textures/layer");
    let null = SoftHandle::null(Image::asset_type());
    let borrowed = Box::leak(Box::new([soft, null]));
    let material = Material {
        direct: first.clone(),
        nested: Some(vec![None, Some(vec![second.clone(), first.clone()])]),
        soft: vec![None, Some(soft), Some(null)],
        mixed: (7, [Some(soft), None], vec![Some(second.clone()), None]),
        owned: Box::new(Some(first.clone())),
        shared: Arc::new(Some(soft)),
        borrowed,
        erased: second.untyped(),
        weak: first.downgrade(),
        erased_weak: second.untyped().downgrade(),
        ignored: second.clone(),
        tint: [1.0; 4],
    };
    let first_count = first.strong_count();
    let second_count = second.strong_count();
    let strong_first = AssetDependency::Strong(first.untyped_id());
    let strong_second = AssetDependency::Strong(second.untyped_id());
    let soft_reference = AssetDependency::Soft {
        guid: soft.guid(),
        type_id: soft.type_id(),
    };
    assert_eq!(
        direct_asset_dependencies(&material),
        vec![
            strong_first,
            strong_second,
            strong_first,
            soft_reference,
            soft_reference,
            strong_second,
            strong_first,
            soft_reference,
            soft_reference,
            strong_second,
        ]
    );
    assert_eq!(
        direct_dependencies(&material),
        vec![
            first.untyped_id(),
            second.untyped_id(),
            first.untyped_id(),
            second.untyped_id(),
            first.untyped_id(),
            second.untyped_id(),
        ]
    );
    material.visit_references(&mut |_| {
        assert_eq!(first.strong_count(), first_count);
        assert_eq!(second.strong_count(), second_count);
    });
    assert_eq!(first.strong_count(), first_count);
    assert_eq!(second.strong_count(), second_count);
    assert!(material.weak.upgrade().is_some());
    assert!(material.erased_weak.upgrade().is_some());
    assert_eq!(material.ignored.id(), second.id());
    assert_eq!(material.tint, [1.0; 4]);
}

/// Enum visitors must read only the active variant and handle tuple indices.
#[test]
fn tuple_and_enum_shapes_visit_only_active_dependencies() {
    let mut images = Assets::new();
    let image = images.insert(Image);
    let soft = SoftHandle::<Image>::from_path("enum/image");
    let tuple = TupleAsset(Some(image.clone()), vec![None, Some(soft)], 9);
    assert_eq!(direct_asset_dependencies(&tuple).len(), 2);
    assert_eq!(tuple.2, 9);
    assert!(direct_asset_dependencies(&Choice::Unit).is_empty());
    let choice = Choice::Tuple(5, Some(image.clone()), vec![soft]);
    assert_eq!(
        direct_asset_dependencies(&choice),
        direct_asset_dependencies(&tuple)
    );
    if let Choice::Tuple(number, ..) = choice {
        assert_eq!(number, 5);
    }
    let named = Choice::Named {
        layers: Some(vec![image.clone()]),
        label: "not a dependency",
    };
    assert_eq!(direct_dependencies(&named), vec![image.untyped_id()]);
    if let Choice::Named { label, .. } = named {
        assert_eq!(label, "not a dependency");
    }
    assert!(direct_dependencies(&Image).is_empty());
    assert_ne!(Never::asset_type(), Image::asset_type());
}

/// Aliases/custom records opt into trait traversal; recursive concrete data works.
#[test]
fn explicit_dependency_fields_remove_handwritten_visitor_lists() {
    let mut images = Assets::new();
    let image = images.insert(Image);
    let soft = SoftHandle::<Image>::from_path("group/image");
    let asset = Composite {
        alias: Some(vec![image.clone()]),
        records: vec![None, Some(NestedGroup { image: soft })],
        next: Some(Box::new(Composite {
            alias: Some(vec![image.clone()]),
            records: Vec::new(),
            next: None,
        })),
    };
    assert_eq!(
        direct_asset_dependencies(&asset),
        vec![
            AssetDependency::Strong(image.untyped_id()),
            AssetDependency::Soft {
                guid: soft.guid(),
                type_id: soft.type_id()
            },
            AssetDependency::Strong(image.untyped_id()),
        ]
    );
}

/// Identity uses the existing frozen name hash, not process-local Rust TypeId.
#[test]
fn explicit_and_default_stable_type_names_use_the_kernel_contract() {
    assert_eq!(
        Image::asset_type(),
        AssetTypeId::of_name("derive_test::Image")
    );
    assert_eq!(Material::TYPE_NAME, concat!(module_path!(), "::Material"));
    assert_eq!(
        Material::asset_type(),
        AssetTypeId::of_name(Material::TYPE_NAME)
    );
    assert_ne!(
        first::SameName::asset_type(),
        second::SameName::asset_type()
    );
}
