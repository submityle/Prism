//! Runtime visitor combinators must preserve soft identities without slot creation.
#![no_std]

extern crate alloc;

use alloc::{boxed::Box, sync::Arc, vec, vec::Vec};
use prism_asset::prelude::*;

struct Image;

impl Asset for Image {
    const TYPE_NAME: &'static str = "visitor_test::Image";
}

/// A handwritten strong visitor still exposes a complete Strong reference view.
#[test]
fn manual_strong_visitor_has_a_rich_reference_projection() {
    struct Manual(Handle<Image>);
    impl Asset for Manual {
        const TYPE_NAME: &'static str = "visitor_test::Manual";
        fn visit_dependencies(&self, visit: &mut dyn FnMut(UntypedAssetId)) {
            visit(self.0.untyped_id());
        }
    }
    let mut images = Assets::new();
    let image = images.insert(Image);
    assert_eq!(
        direct_asset_dependencies(&Manual(image.clone())),
        vec![AssetDependency::Strong(image.untyped_id())]
    );
}

/// Trait combinators use borrowing through slices, boxes, arcs and references.
#[test]
fn container_visitors_preserve_order_duplicates_and_borrows() {
    let soft = SoftHandle::<Image>::from_path("visitor/image");
    let mut values = [Some(soft), None, Some(soft)];
    let slice: &mut [Option<SoftHandle<Image>>] = &mut values;
    let containers = (Box::new(Some(soft)), Arc::new(vec![soft]), slice);
    let mut dependencies = Vec::new();
    containers.visit_asset_dependencies(&mut |dependency| dependencies.push(dependency));
    assert_eq!(
        dependencies,
        vec![
            AssetDependency::Soft {
                guid: soft.guid(),
                type_id: soft.type_id()
            };
            4
        ]
    );
    let by_reference = &containers;
    let mut count = 0;
    VisitAssetDependencies::visit_asset_dependencies(&by_reference, &mut |_| count += 1);
    assert_eq!(count, 4);
}

/// A null GUID is absent, while a non-null GUID retains its declared type tag.
#[test]
fn soft_handles_do_not_invent_slots_or_erase_expected_types() {
    let custom_type = AssetTypeId::from_u64(0xf00d);
    let soft = SoftHandle::<Image>::new(StableGuid::from_path("not_loaded"), custom_type);
    let null = SoftHandle::<Image>::null(custom_type);
    let mut dependencies = Vec::new();
    [null, soft].visit_asset_dependencies(&mut |dependency| dependencies.push(dependency));
    assert_eq!(dependencies.len(), 1);
    assert_eq!(dependencies[0].type_id(), custom_type);
    assert_eq!(dependencies[0].strong_id(), None);
    assert_eq!(
        dependencies[0],
        AssetDependency::Soft {
            guid: soft.guid(),
            type_id: custom_type
        }
    );
}

/// Typed and erased strong handles name the same allocation without extra pins.
#[test]
fn strong_and_erased_handle_visitors_do_not_clone() {
    let mut images = Assets::<Image>::new();
    let handle = images.reserve();
    let erased = handle.untyped();
    let before = handle.strong_count();
    let mut dependencies = Vec::new();
    (&handle, &erased).visit_asset_dependencies(&mut |dependency| {
        assert_eq!(handle.strong_count(), before);
        dependencies.push(dependency);
    });
    assert_eq!(
        dependencies,
        vec![AssetDependency::Strong(handle.untyped_id()); 2]
    );
    assert_eq!(dependencies[0].type_id(), Image::asset_type());
    assert_eq!(dependencies[0].strong_id(), Some(handle.untyped_id()));
    assert_eq!(handle.strong_count(), before);
}
