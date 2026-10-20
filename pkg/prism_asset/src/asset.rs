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
//!    "dependency reclaimed too early" bug, which [`derive@crate::Asset`]
//!    prevents for supported fields by generating the traversal.

use crate::guid::StableGuid;
use crate::id::UntypedAssetId;
use crate::type_id::AssetTypeId;
use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;

/// A direct asset reference, without owning or resolving its target.
///
/// Runtime slots and persistent GUIDs are deliberately different variants:
/// reporting a soft reference must neither fabricate a slot nor load or pin it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AssetDependency {
    /// A resident or reserved runtime allocation held by a strong handle.
    Strong(UntypedAssetId),
    /// A persistent, non-owning reference that may have no runtime allocation.
    Soft {
        /// The persistent identity of the target asset.
        guid: StableGuid,
        /// The expected type, retained even when the target is not loaded.
        type_id: AssetTypeId,
    },
}

impl AssetDependency {
    /// The target's asset type, independent of residency.
    #[must_use]
    pub const fn type_id(self) -> AssetTypeId {
        match self {
            Self::Strong(id) => id.type_id(),
            Self::Soft { type_id, .. } => type_id,
        }
    }

    /// Returns the runtime slot only for a strong reference.
    #[must_use]
    pub const fn strong_id(self) -> Option<UntypedAssetId> {
        match self {
            Self::Strong(id) => Some(id),
            Self::Soft { .. } => None,
        }
    }
}

/// Traverses references embedded in a field or a reusable group of fields.
///
/// Implemented for handles, `Option`, `Vec`, arrays, slices, `Box`, `Arc`,
/// references, and tuples of up to eight dependency-bearing elements. Derive
/// it for nested records; mark an opaque/aliased field with `#[dependency]`
/// so [`derive@crate::Asset`] uses this contract rather than syntactic scanning.
/// Traversal is ordered, does not deduplicate, and never clones handles.
pub trait VisitAssetDependencies {
    /// Calls `visit` for each direct strong or non-null soft reference.
    fn visit_asset_dependencies(&self, visit: &mut dyn FnMut(AssetDependency));
}

impl<T: VisitAssetDependencies> VisitAssetDependencies for Option<T> {
    fn visit_asset_dependencies(&self, visit: &mut dyn FnMut(AssetDependency)) {
        if let Some(value) = self {
            value.visit_asset_dependencies(visit);
        }
    }
}

impl<T: VisitAssetDependencies> VisitAssetDependencies for Vec<T> {
    fn visit_asset_dependencies(&self, visit: &mut dyn FnMut(AssetDependency)) {
        self.as_slice().visit_asset_dependencies(visit);
    }
}

impl<T: VisitAssetDependencies> VisitAssetDependencies for [T] {
    fn visit_asset_dependencies(&self, visit: &mut dyn FnMut(AssetDependency)) {
        for value in self {
            value.visit_asset_dependencies(visit);
        }
    }
}

impl<T: VisitAssetDependencies, const N: usize> VisitAssetDependencies for [T; N] {
    fn visit_asset_dependencies(&self, visit: &mut dyn FnMut(AssetDependency)) {
        self.as_slice().visit_asset_dependencies(visit);
    }
}

impl<T: VisitAssetDependencies + ?Sized> VisitAssetDependencies for Box<T> {
    fn visit_asset_dependencies(&self, visit: &mut dyn FnMut(AssetDependency)) {
        T::visit_asset_dependencies(self, visit);
    }
}

impl<T: VisitAssetDependencies + ?Sized> VisitAssetDependencies for Arc<T> {
    fn visit_asset_dependencies(&self, visit: &mut dyn FnMut(AssetDependency)) {
        T::visit_asset_dependencies(self, visit);
    }
}

impl<T: VisitAssetDependencies + ?Sized> VisitAssetDependencies for &T {
    fn visit_asset_dependencies(&self, visit: &mut dyn FnMut(AssetDependency)) {
        T::visit_asset_dependencies(self, visit);
    }
}

impl<T: VisitAssetDependencies + ?Sized> VisitAssetDependencies for &mut T {
    fn visit_asset_dependencies(&self, visit: &mut dyn FnMut(AssetDependency)) {
        T::visit_asset_dependencies(self, visit);
    }
}

macro_rules! dependency_tuple {
    ($($name:ident : $index:tt),+) => {
        impl<$($name: VisitAssetDependencies),+> VisitAssetDependencies for ($($name,)+) {
            fn visit_asset_dependencies(&self, visit: &mut dyn FnMut(AssetDependency)) {
                $(self.$index.visit_asset_dependencies(visit);)+
            }
        }
    };
}

dependency_tuple!(A: 0);
dependency_tuple!(A: 0, B: 1);
dependency_tuple!(A: 0, B: 1, C: 2);
dependency_tuple!(A: 0, B: 1, C: 2, D: 3);
dependency_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4);
dependency_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5);
dependency_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6);
dependency_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7);

/// A content type that can live in an [`Assets`](crate::Assets) arena, be
/// referenced by [`Handle`](crate::Handle), and participate in the dependency
/// graph.
///
/// Implementors must be `Send + Sync + 'static`: assets are decoded on worker
/// threads and shared across the render/main-world split.
///
/// Prefer the derive to handwritten dependency lists:
///
/// ```rust
/// use prism_asset::prelude::*;
/// #[derive(Asset)]
/// #[asset(type_name = "example::Image")]
/// struct Image;
/// #[derive(Asset)]
/// struct Material {
///     texture: Option<Handle<Image>>,
///     layers: Vec<Option<SoftHandle<Image>>>,
///     tint: [f32; 4],
/// }
/// let material = Material {
///     texture: None,
///     layers: vec![Some(SoftHandle::from_path("texture.png"))],
///     tint: [1.0; 4],
/// };
/// assert!(direct_dependencies(&material).is_empty());
/// assert!(matches!(direct_asset_dependencies(&material)[0], AssetDependency::Soft { .. }));
/// ```
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

    /// Reports strong runtime dependencies, excluding soft references.
    ///
    /// The default reports no dependencies, which is correct for leaf assets
    /// (raw images, audio samples). Composite assets must override it to visit
    /// each embedded handle, including those nested in `Option`, `Vec`, and
    /// other fields.
    fn visit_dependencies(&self, visit: &mut dyn FnMut(UntypedAssetId)) {
        let _ = visit;
    }

    /// Reports all direct references, preserving strong/soft semantics.
    ///
    /// The derive emits this traversal and the strong-only projection above.
    /// A handwritten implementation that only overrides `visit_dependencies`
    /// reports those strong references here as well. No traversal performs
    /// source I/O, resolves GUIDs, deduplicates, or changes reference counts.
    fn visit_references(&self, visit: &mut dyn FnMut(AssetDependency)) {
        self.visit_dependencies(&mut |id| visit(AssetDependency::Strong(id)));
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

/// Collects strong and soft references in field/element traversal order.
///
/// Unlike [`Asset::visit_references`], this convenience function allocates.
#[must_use]
pub fn direct_asset_dependencies<A: Asset + ?Sized>(asset: &A) -> Vec<AssetDependency> {
    let mut dependencies = Vec::new();
    asset.visit_references(&mut |dependency| dependencies.push(dependency));
    dependencies
}
