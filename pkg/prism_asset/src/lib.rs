//! # `prism_asset`
//!
//! Prism's asset kernel: the in-house replacement for `bevy_asset`'s identity,
//! storage, and dependency core. It owns *who* an asset is, *where* it lives in
//! storage, *how many* live references point at it, and *what it depends on* —
//! the deterministic heart that higher layers (loaders, async IO, hot-reload,
//! the bake pipeline) build on in later milestones.
//!
//! ## Pieces
//! - [`AssetIndex`]: a generational `(index, generation)` slot id. Reusing a
//!   freed slot bumps its generation so stale ids never alias a new asset.
//! - [`AssetId`]: a typed id (`AssetId<A>`) wrapping an [`AssetIndex`], plus the
//!   type-erased [`UntypedAssetId`].
//! - [`AssetPath`]: an owned source path with an optional sub-asset `#label`
//!   (for example `scene.gltf#Mesh0`).
//! - [`Handle`] / [`WeakHandle`]: reference-counted handles. A strong [`Handle`]
//!   keeps an asset alive; [`Assets::remove_unused`] reclaims slots whose last
//!   strong handle has dropped. Counting uses [`alloc::sync::Arc`], so it is
//!   `no_std + alloc` and lock-free.
//! - [`Assets`]: a generational dense arena mapping [`AssetId`] to values. It
//!   supports eager [`Assets::insert`] and ahead-of-time [`Assets::reserve`]
//!   (a stable id/handle before an async load resolves via
//!   [`Assets::fulfill`]/[`Assets::fail`]), emitting [`AssetEvent`]s on
//!   add/modify/remove/fail.
//! - [`DependencyGraph`]: directed asset→dependency edges with Kahn
//!   topological ordering and cycle detection, used to order recursive loads.
//! - [`LoadState`] / [`RecursiveDependencyLoadState`]: per-asset load progress.
//!
//! ## Design notes
//! Identity is intentionally value-typed and `Copy` so it is cheap to store in
//! components and pass across the render/main-world split. Lifetime is handled
//! by [`Handle`] reference counting rather than a tracing GC, mirroring the
//! form of mature engine asset systems without copying their code.
//!
//! ## Milestone status
//! - **M0 (this crate, done):** identity ([`AssetIndex`]/[`AssetId`]/
//!   [`UntypedAssetId`]), [`AssetPath`], ref-counted [`Handle`]/[`WeakHandle`],
//!   generational [`Assets`] storage with [`AssetEvent`]s and
//!   [`Assets::remove_unused`], [`DependencyGraph`] (topo order + cycle
//!   detection), and [`LoadState`]. Fully unit-tested, `no_std + alloc`, no
//!   unsafe.
//! - **M1 (kernel recast, done):** three-tier identity ([`StableGuid`]/
//!   [`AssetTypeId`]), [`SoftHandle`], the [`Asset`] trait with
//!   `visit_dependencies`, `reserve`/`fulfill`/`fail` with deferred reclaim,
//!   incremental recursive readiness + invalidation on [`DependencyGraph`],
//!   [`ErrorRegistry`], and the deterministic [`LoaderRegistry`] selection
//!   policy (design §9.1). Fully unit-tested, `no_std + alloc`, no unsafe.
//! - **M2 (planned):** `AssetServer` + `AssetLoader` execution and load
//!   dedup/path→id interning (needs `std`, lives in `prism_asset_import`).
//! - **M2 (planned):** async IO backends and the dependency-aware load
//!   scheduler driving [`DependencyGraph`].
//! - **M3 (planned):** filesystem watch + hot-reload re-emitting
//!   [`AssetEvent::Modified`].
//! - **M4 (planned):** the offline bake/process pipeline and content-hash
//!   caching.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

extern crate alloc;

mod asset;
mod dependency;
mod error;
mod event;
mod guid;
mod handle;
mod hash;
mod id;
mod load_state;
mod loader;
#[cfg(feature = "std")]
mod loader_exec;
mod path;
#[cfg(feature = "std")]
mod source;
mod storage;
mod stores;
mod type_id;

#[cfg(test)]
mod tests;

pub use asset::{direct_dependencies, Asset};
pub use dependency::{DependencyError, DependencyGraph};
pub use error::{AssetError, AssetErrorId, ErrorRegistry};
pub use event::AssetEvent;
pub use guid::{normalize_path, StableGuid};
pub use handle::{Handle, HandleId, SoftHandle, UntypedHandle, WeakHandle};
pub use id::{AssetId, AssetIndex, UntypedAssetId};
pub use load_state::{LoadState, RecursiveDependencyLoadState};
pub use loader::{LoaderId, LoaderRegistry, SuffixConflict};
#[cfg(feature = "std")]
pub use loader_exec::{
    AssetLoader, AssetLoaders, DepRequest, ErasedLoadedAsset, FullLoadOutput, LoadContext,
    LoadError, LoadedAsset,
};
pub use path::AssetPath;
#[cfg(feature = "std")]
pub use source::{AssetMeta, AssetReader, AssetSources, FsSource, MemSource, ReadError};
pub use storage::Assets;
pub use stores::{AssetStores, ErasedAssetStore, StoreError};
pub use type_id::AssetTypeId;

/// Convenient re-exports for downstream crates.
pub mod prelude {
    pub use crate::{
        Asset, AssetError, AssetErrorId, AssetEvent, AssetId, AssetIndex, AssetPath, AssetStores,
        AssetTypeId, Assets, DependencyGraph, ErrorRegistry, Handle, LoadState, LoaderId,
        LoaderRegistry, SoftHandle, StableGuid, SuffixConflict, UntypedAssetId, UntypedHandle,
        WeakHandle,
    };
}
