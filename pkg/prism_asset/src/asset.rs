//! The [`Asset`] trait: what it means to be a loadable, dependency-bearing
//! content type.
//!
//! Every concrete content type (mesh, texture, material, scene, audio clip,
//! animation clip, …) implements [`Asset`]. The trait does two jobs:
//!
//! 1. **Stable type identity** via [`Asset::TYPE_NAME`], hashed into an
//!    [`AssetTypeId`] so erased ids and handles can be routed and type-checked.
//! 2. **Dependency disclosure** via [`Asset::visit_dependencies`], so the
//!    loader/dependency graph can discover an asset's references without the
//!    asset author hand-maintaining a separate list. A `Material` that holds
//!    `Handle<Image>` fields reports them here; forgetting to is the classic
//!    "dependency reclaimed too early" bug, which the `#[derive(Asset)]` macro
//!    (a later milestone) eliminates by generating this method.

use crate::id::UntypedAssetId;
use crate::type_id::AssetTypeId;
use alloc::vec::Vec;

/// A content type that can live in an [`Assets`](crate::Assets) arena, be
/// referenced by [`Handle`](crate::Handle), and participate in the dependency
/// graph.
///
/// Implementors must be `Send + Sync + 'static`: assets are decoded on worker
/// threads and shared across the render/main-world split.
pub trait Asset: Send + Sync + 'static {
    /// A stable, globally-unique type name used to derive this type's
    /// [`AssetTypeId`]. Convention: `"crate_name::TypeName"`. Treat it as a
    /// persisted identity — renaming it is a migration, not a refactor.
    const TYPE_NAME: &'static str;

    /// This type's stable [`AssetTypeId`].
    #[must_use]
    fn asset_type() -> AssetTypeId
    where
        Self: Sized,
    {
        AssetTypeId::of::<Self>()
    }

    /// Reports every asset this value directly references by invoking `visit`
    /// once per dependency id.
    ///
    /// The default reports no dependencies, which is correct for leaf assets
    /// (raw images, audio samples). Composite assets must override it to visit
    /// each embedded handle, including those nested in `Option`, `Vec`, and
    /// other fields.
    fn visit_dependencies(&self, visit: &mut dyn FnMut(UntypedAssetId)) {
        let _ = visit;
    }
}

/// Collects an asset's direct dependencies into a `Vec`, a convenience over
/// [`Asset::visit_dependencies`] for callers that want an owned list.
#[must_use]
pub fn direct_dependencies<A: Asset + ?Sized>(asset: &A) -> Vec<UntypedAssetId> {
    let mut deps = Vec::new();
    asset.visit_dependencies(&mut |id| deps.push(id));
    deps
}
