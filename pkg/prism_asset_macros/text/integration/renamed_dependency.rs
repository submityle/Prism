//! Cargo dependency renames and facade paths must work without a canonical import.
#![no_std]

extern crate alloc;

use alloc::{boxed::Box, vec, vec::Vec};
use core::marker::PhantomData;
use prism_asset_runtime::prelude::*;

#[derive(Asset)]
#[asset(type_name = "macro_test::Image")]
struct Image;

#[derive(VisitAssetDependencies)]
struct Borrowed<'a, T: ?Sized, const N: usize>
where
    T: VisitAssetDependencies,
{
    #[dependency]
    value: &'a T,
    handles: [Option<SoftHandle<Image>>; N],
    unused: PhantomData<&'a T>,
}

// Use a facade re-export to exercise the explicit crate-path escape hatch.
mod facade {
    pub(crate) use prism_asset_runtime::*;
}

#[derive(prism_asset_runtime::Asset)]
#[asset(crate = "crate::facade", type_name = "macro_test::FacadeAsset")]
struct FacadeAsset {
    layers: Option<Box<Vec<SoftHandle<Image>>>>,
}

/// The Cargo rename must be discovered automatically by both derive macros.
#[test]
fn renamed_runtime_and_generic_helper_compile_without_std() {
    let soft = SoftHandle::<Image>::from_path("renamed/image");
    let borrowed = Borrowed {
        value: &soft,
        handles: [None, Some(soft)],
        unused: PhantomData,
    };
    let mut dependencies = Vec::new();
    borrowed.visit_asset_dependencies(&mut |dependency| dependencies.push(dependency));
    assert_eq!(dependencies.len(), 2);
    assert!(dependencies
        .iter()
        .all(|reference| reference.strong_id().is_none()));
    assert_eq!(
        Image::asset_type(),
        AssetTypeId::of_name("macro_test::Image")
    );
}

/// Facades do not require a direct canonical prism_asset crate alias.
#[test]
fn explicit_facade_override_uses_the_public_runtime_contract() {
    let soft = SoftHandle::<Image>::from_path("facade/image");
    let asset = FacadeAsset {
        layers: Some(Box::new(vec![soft])),
    };
    assert_eq!(
        direct_asset_dependencies(&asset),
        vec![AssetDependency::Soft {
            guid: soft.guid(),
            type_id: soft.type_id(),
        }]
    );
    assert!(direct_dependencies(&asset).is_empty());
}
