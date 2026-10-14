//! # `prism_asset`
//!
//! Prism's asset runtime: the in-house replacement for `bevy_asset`'s identity,
//! storage, loading, and dependency core. It owns *who* an asset is, *where* it
//! lives in storage, *how many* live references point at it, and *what it
//! depends on*. With `std`, it also provides synchronous source/loader execution,
//! an `AssetServer`, a bounded scheduler facade, and hot-reload primitives. The
//! facade currently serializes actual server work; full event-driven parallel
//! loading and runtime streaming remain future work.
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
//! ## Capability status
//! - **Core (implemented):** generational identity, stable GUID/type identity,
//!   ref-counted handles, `Assets` storage and events, dependency graph,
//!   recursive readiness/invalidation, error registry, and deterministic loader
//!   selection. This layer remains `no_std + alloc` and contains no `unsafe`.
//! - **Standard I/O (implemented and covered by crate tests, `std`):** `AssetServer`, `AssetLoader`,
//!   `LoadContext`, path/load deduplication, VFS sources, synchronous dependency
//!   assembly, and a bounded scheduler with a limited lock-outside read/decode/
//!   dependency-commit path, cancellation and backpressure.
//! - **Hot reload (implemented in part):** watcher/event batching, reload plans,
//!   dependency-aware reassembly, and last-known-good retention are available;
//!   lock-free asynchronous reassembly, full labeled-revision lifecycle, and
//!   Windows-equivalent secure path opening remain incomplete.
//! - **Not implemented here:** offline cooking, DDC/CAS ownership, codecs,
//!   labeled-asset async assembly, reload epochs, range-backed assembly,
//!   independent I/O/CPU pools, streaming residency, DirectStorage/GPU decode,
//!   and release packaging. Those belong to Asset Build or future runtime
//!   features. The scheduler slice is not a complete EDL pipeline.

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
mod reload;
mod reload_plan;
#[cfg(feature = "std")]
mod scheduler;
#[cfg(feature = "std")]
mod server;
#[cfg(feature = "std")]
mod source;
mod storage;
mod stores;
#[cfg(feature = "std")]
mod streaming;
mod type_id;
#[cfg(feature = "std")]
mod watch;

#[cfg(test)]
mod tests;

pub use asset::{direct_dependencies, Asset};
pub use dependency::{DependencyError, DependencyGraph};
pub use error::{AssetError, AssetErrorId, ErrorRegistry};
pub use event::AssetEvent;
pub use guid::{normalize_path, StableGuid};
pub use handle::{Handle, HandleId, SoftHandle, UntypedHandle, UntypedWeakHandle, WeakHandle};
pub use id::{AssetId, AssetIndex, UntypedAssetId};
pub use load_state::{LoadState, RecursiveDependencyLoadState};
pub use loader::{LoaderId, LoaderRegistry, SuffixConflict};
#[cfg(feature = "std")]
pub use loader_exec::{
    AssetLoader, AssetLoaders, DepRequest, ErasedLoadedAsset, FullLoadOutput, LoadContext,
    LoadError, LoadedAsset,
};
pub use path::AssetPath;
pub use reload::{ReloadCommit, ReloadCommitGuard, ReloadEpoch};
pub use reload_plan::{plan_reload, ReloadPlan, ReloadPlanError};
#[cfg(feature = "std")]
pub use scheduler::{LoadFailure, LoadScheduler, LoadTicket, ScheduleError, ScheduleOutcome};
#[cfg(feature = "std")]
pub use server::AssetServer;

#[cfg(all(feature = "std", feature = "ecs"))]
impl prism_ecs::resource::Resource for AssetServer {}
#[cfg(feature = "std")]
pub use source::{AssetMeta, AssetReader, AssetSources, FsSource, MemSource, ReadError};
pub use storage::Assets;
pub use stores::{AssetStores, ErasedAssetStore, StoreError};
#[cfg(feature = "std")]
pub use streaming::{
    AsyncStreamIoExecutor, AsyncStreamTask, BudgetKind, Hysteresis, Residency, ResidencyLimits,
    StreamError, StreamFailure, StreamIoError, StreamIoExecutor, StreamRequest, StreamRequestId,
    StreamState, StreamingController, SubmitAndExecuteError, SubmitError,
};
pub use type_id::AssetTypeId;
#[cfg(feature = "std")]
pub use watch::{AssetChange, AssetChangeKind, ChangeBatch, ChangeBatchBuilder};
#[cfg(feature = "file_watcher")]
pub use watch::{AssetWatchHandle, FileWatcher, WatchStartError};

/// Convenient re-exports for downstream crates.
pub mod prelude {
    #[cfg(feature = "std")]
    pub use crate::AssetServer;
    pub use crate::{
        plan_reload, Asset, AssetError, AssetErrorId, AssetEvent, AssetId, AssetIndex, AssetPath,
        AssetStores, AssetTypeId, Assets, DependencyGraph, ErrorRegistry, Handle, LoadState,
        LoaderId, LoaderRegistry, ReloadCommit, ReloadCommitGuard, ReloadEpoch, ReloadPlan,
        ReloadPlanError, SoftHandle, StableGuid, SuffixConflict, UntypedAssetId, UntypedHandle,
        UntypedWeakHandle, WeakHandle,
    };
}

#[cfg(test)]
mod reload_plan_tests;
