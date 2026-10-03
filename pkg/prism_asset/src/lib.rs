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
//! - [`Assets`]: a generational dense arena mapping [`AssetId`] to values,
//!   emitting [`AssetEvent`]s on insert/modify/remove.
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
//! - **M1 (planned):** `AssetServer`, `AssetLoader` trait, load handle hand-out
//!   and path→id interning.
//! - **M2 (planned):** async IO backends and the dependency-aware load
//!   scheduler driving [`DependencyGraph`].
//! - **M3 (planned):** filesystem watch + hot-reload re-emitting
//!   [`AssetEvent::Modified`].
//! - **M4 (planned):** the offline bake/process pipeline and content-hash
//!   caching.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

extern crate alloc;

mod dependency;
mod event;
mod handle;
mod id;
mod load_state;
mod path;
mod storage;

#[cfg(test)]
mod tests;

pub use dependency::{DependencyError, DependencyGraph};
pub use event::AssetEvent;
pub use handle::{Handle, HandleId, UntypedHandle, WeakHandle};
pub use id::{AssetId, AssetIndex, UntypedAssetId};
pub use load_state::{LoadState, RecursiveDependencyLoadState};
pub use path::AssetPath;
pub use storage::Assets;

/// Convenient re-exports for downstream crates.
pub mod prelude {
    pub use crate::{
        AssetEvent, AssetId, AssetIndex, AssetPath, Assets, DependencyGraph, Handle, LoadState,
        UntypedAssetId, UntypedHandle, WeakHandle,
    };
}
