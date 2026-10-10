//! The synchronous [`AssetServer`]: the front door that turns an
//! [`AssetPath`] into a resident, reference-counted [`Handle`] (design §9.2).
//!
//! # What the server marries together
//! The kernel is built from independent, individually tested pieces — a loader
//! selection + execution table ([`AssetLoaders`]), a byte-source overlay
//! ([`AssetSources`]), a set of type-erased arenas ([`AssetStores`]), a
//! dependency graph ([`DependencyGraph`]), and an error registry
//! ([`ErrorRegistry`]). The `AssetServer` is the one object that drives all of
//! them through a single load pipeline:
//!
//! 1. resolve the path to a produced [`AssetTypeId`] (either requested by the
//!    caller via [`AssetServer::load`] or inferred from loader selection),
//! 2. reserve a slot in the matching arena *before* reading bytes so a
//!    dependency that cycles back finds an in-flight handle instead of
//!    recursing forever,
//! 3. read bytes through the source overlay, decode them through the selected
//!    loader, intern every labeled sub-asset, recursively load every declared
//!    dependency, wire the dependency graph, and finally fulfill the reserved
//!    slot.
//!
//! # Why this first cut is synchronous
//! The async [`LoadScheduler`](crate) of design §9.3/§9.4 (I/O vs CPU worker
//! pools, priority inheritance, backpressure, epoch-based reload supersede)
//! layers *on top of* this pipeline; it does not replace it. Keeping a fully
//! synchronous, deterministic path first means the entire load/dedup/dependency
//! /reload/GC contract is exercisable without a runtime and is the oracle the
//! async scheduler's loom and golden tests check against (§25). The server
//! takes `&self` and guards all mutable state behind a `std::sync::Mutex`, so it
//! is already `Send + Sync` and shareable as an `Arc<AssetServer>` exactly as
//! the ECS resource will be.
//!
//! # Liveness and reclamation
//! A freshly loaded asset's *only* strong handle is the one [`AssetServer::load`]
//! returns to the caller; the server itself holds only `Weak` references in its
//! intern table so deduplication never pins memory. The transitive sub-assets
//! and dependencies of an asset are kept alive by a `retained` table keyed by
//! the owning asset's id. When the caller drops the top-level handle, the arena
//! reclaims that slot, the server prunes the dead `retained` entry, and the
//! sub-assets and dependencies it was keeping alive become reclaimable in turn —
//! a fixpoint that [`AssetServer::collect_releases`] runs to quiescence. A
//! dependency edge that would close a cycle is rejected by the graph and its
//! handle is deliberately *not* retained, so a cyclic reference can never form a
//! reference-counting leak.

#![cfg(feature = "std")]

use crate::dependency::{DependencyError, DependencyGraph};
use crate::error::{AssetError, ErrorRegistry};
use crate::guid::StableGuid;
use crate::handle::{Handle, SoftHandle, UntypedHandle, UntypedWeakHandle};
use crate::id::UntypedAssetId;
use crate::load_state::{LoadState, RecursiveDependencyLoadState};
use crate::loader_exec::{AssetLoader, AssetLoaders, ErasedLoadedAsset, FullLoadOutput};
use crate::path::AssetPath;
use crate::source::{AssetReader, AssetSources};
use crate::stores::AssetStores;
use crate::type_id::AssetTypeId;
use crate::{Asset, LoaderId, SuffixConflict};
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;
use std::sync::Mutex;

/// The runtime asset front door (design §9.2).
///
/// Register loaders and mount byte sources, then call [`AssetServer::load`] to
/// turn an [`AssetPath`] into a [`Handle`]. All mutable state lives behind an
/// internal `std::sync::Mutex`, so the server takes `&self` throughout and is
/// freely shareable as an `Arc<AssetServer>`.
pub struct AssetServer {
    inner: Mutex<AssetServerInner>,
}

/// The lock-guarded interior of an [`AssetServer`]. Every public method locks
/// the mutex and delegates to a `&mut self` method here, so the invariants
/// (intern/catalog consistency, graph wiring, retained liveness) are maintained
/// under a single critical section.
struct AssetServerInner {
    loaders: AssetLoaders,
    sources: AssetSources,
    stores: AssetStores,
    graph: DependencyGraph,
    errors: ErrorRegistry,
    /// `path -> weak handle` dedup table. Weak so interning never pins an asset
    /// in memory; a dead entry is pruned by [`AssetServerInner::collect_releases`].
    interned: BTreeMap<AssetPath, UntypedWeakHandle>,
    /// `guid -> path` catalog, populated as paths are loaded, so a
    /// [`SoftHandle`] (which stores only a [`StableGuid`]) can be resolved back
    /// to the concrete path to load (design §9.2 soft references).
    catalog: BTreeMap<StableGuid, AssetPath>,
    /// `owner -> strong handles it keeps alive` (its labeled sub-assets and its
    /// successfully wired dependencies, transitively). Replacing an entry on
    /// reload drops the previous closure's strong refs.
    retained: BTreeMap<UntypedAssetId, Vec<UntypedHandle>>,
}

impl AssetServer {
    /// Creates an empty server with no loaders registered and no sources
    /// mounted.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(AssetServerInner {
                loaders: AssetLoaders::new(),
                sources: AssetSources::new(),
                stores: AssetStores::new(),
                graph: DependencyGraph::new(),
                errors: ErrorRegistry::new(),
                interned: BTreeMap::new(),
                catalog: BTreeMap::new(),
                retained: BTreeMap::new(),
            }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, AssetServerInner> {
        self.inner.lock().expect("asset server mutex poisoned")
    }

    /// Registers a typed [`AssetLoader`] and the arena for the asset type it
    /// produces, so a later [`AssetServer::load`] of a matching suffix can both
    /// select the loader and route the decoded value into storage. Returns the
    /// assigned [`LoaderId`] and any advisory suffix [`SuffixConflict`]s.
    pub fn register_loader<L: AssetLoader>(&self, loader: L) -> (LoaderId, Vec<SuffixConflict>) {
        let mut inner = self.lock();
        inner.stores.register::<L::Asset>();
        inner.loaders.register::<L>(loader)
    }

    /// Mounts a byte [`AssetReader`] under `name` at the given overlay
    /// `priority` (higher wins when the same path exists in several sources).
    pub fn mount(&self, name: &str, reader: Arc<dyn AssetReader>, priority: i32) {
        self.lock().sources.mount(name, reader, priority);
    }

    /// Unmounts the source previously mounted under `name`, returning whether a
    /// source was removed.
    pub fn unmount(&self, name: &str) -> bool {
        self.lock().sources.unmount(name)
    }

    /// Loads the asset at `path` as type `A`, returning a strong [`Handle`]
    /// immediately. The returned handle is live right away; its
    /// [`LoadState`](crate::LoadState) progresses to
    /// [`Loaded`](crate::LoadState::Loaded) once decoding and dependency
    /// resolution finish, or [`Failed`](crate::LoadState::Failed) if anything in
    /// the pipeline errors. Repeated loads of the same path return handles to
    /// the same slot (deduplicated through the intern table).
    #[must_use]
    pub fn load<A: Asset>(&self, path: impl Into<AssetPath>) -> Handle<A> {
        let path = path.into();
        let mut inner = self.lock();
        inner.stores.register::<A>();
        match inner.load_path(path, Some(AssetTypeId::of::<A>())) {
            Some(handle) => handle
                .typed::<A>()
                .expect("load_path honored the requested asset type"),
            // Only reachable if the arena vanished between registration and
            // reservation, which cannot happen under the held lock; fail
            // defensively rather than panic.
            None => inner.reserve_failed::<A>("asset arena unavailable"),
        }
    }

    /// Loads the asset at `path` without a statically known type, inferring the
    /// produced type from loader selection. Returns `None` if no registered
    /// loader claims the path (so no arena can be chosen).
    #[must_use]
    pub fn load_untyped(&self, path: impl Into<AssetPath>) -> Option<UntypedHandle> {
        let path = path.into();
        self.lock().load_path(path, None)
    }

    /// Resolves a [`SoftHandle`] to a concrete load. A null soft handle, a guid
    /// absent from the catalog, or a type that disagrees with `A` yields a
    /// [`Failed`](crate::LoadState::Failed) handle carrying the reason rather
    /// than panicking, so persisted references survive missing or retyped
    /// targets gracefully (design §9.2).
    #[must_use]
    pub fn load_soft<A: Asset>(&self, soft: &SoftHandle<A>) -> Handle<A> {
        let mut inner = self.lock();
        if soft.is_null() {
            return inner.reserve_failed::<A>("soft handle is null");
        }
        if soft.type_id() != AssetTypeId::of::<A>() {
            return inner.reserve_failed::<A>("soft handle type does not match requested type");
        }
        let Some(path) = inner.catalog.get(&soft.guid()).cloned() else {
            return inner.reserve_failed::<A>("soft handle guid is not in the catalog");
        };
        inner.stores.register::<A>();
        match inner.load_path(path, Some(AssetTypeId::of::<A>())) {
            Some(handle) => handle
                .typed::<A>()
                .expect("load_path honored the requested asset type"),
            None => inner.reserve_failed::<A>("asset arena unavailable"),
        }
    }

    /// Forces the asset at `path` to be re-read and re-decoded in place,
    /// overwriting the existing slot and emitting a modification event, if it is
    /// currently resident; otherwise behaves like [`AssetServer::load_untyped`].
    /// This is the manual counterpart to the §12 hot-reload watcher. Returns the
    /// handle to the (re)loaded asset, or `None` if it was neither resident nor
    /// loadable.
    pub fn reload(&self, path: impl Into<AssetPath>) -> Option<UntypedHandle> {
        let path = path.into();
        let mut inner = self.lock();
        if let Some(handle) = inner
            .interned
            .get(&path)
            .and_then(UntypedWeakHandle::upgrade)
        {
            let id = handle.id();
            let _ = inner.graph.invalidate(id);
            inner.graph.clear_stale(id);
            let _ = inner.graph.set_self_state(id, LoadState::Loading);
            inner.finalize(path, id, id.type_id());
            Some(handle)
        } else {
            inner.load_path(path, None)
        }
    }

    /// The [`LoadState`] of `id` (its own state, not its dependency closure).
    #[must_use]
    pub fn load_state(&self, id: UntypedAssetId) -> LoadState {
        self.lock().stores.load_state(id)
    }

    /// The recursive load state of `id`'s full dependency closure (design §8).
    #[must_use]
    pub fn recursive_state(&self, id: UntypedAssetId) -> RecursiveDependencyLoadState {
        self.lock().graph.recursive_state(id)
    }

    /// Whether `id` has finished loading successfully.
    #[must_use]
    pub fn is_loaded(&self, id: UntypedAssetId) -> bool {
        matches!(self.lock().stores.load_state(id), LoadState::Loaded)
    }

    /// The number of resident asset slots across all arenas.
    #[must_use]
    pub fn total_assets(&self) -> usize {
        self.lock().stores.total_assets()
    }

    /// The number of registered loaders.
    #[must_use]
    pub fn loader_count(&self) -> usize {
        self.lock().loaders.len()
    }

    /// Reads an asset by value through a closure while the lock is held. Returns
    /// `None` if the asset is not resident as type `A`. The closure form avoids
    /// handing out a reference whose lifetime would outlive the lock.
    #[must_use]
    pub fn with_asset<A: Asset, R>(
        &self,
        id: crate::id::AssetId<A>,
        f: impl FnOnce(&A) -> R,
    ) -> Option<R> {
        let inner = self.lock();
        inner.stores.get::<A>(id).map(f)
    }

    /// Runs the reclamation fixpoint: advances every arena's grace frame,
    /// prunes dead intern entries, drops the retained closures of reclaimed
    /// owners, and repeats until nothing more can be freed. Returns the total
    /// number of asset slots reclaimed. See the module docs for the liveness
    /// model.
    pub fn collect_releases(&self) -> usize {
        self.lock().collect_releases()
    }

    /// Reclaims abandoned slots once without advancing grace frames, then prunes
    /// the server's dead bookkeeping. Prefer [`AssetServer::collect_releases`]
    /// for the full deferred-reclaim semantics (design §6.2).
    pub fn remove_unused(&self) -> usize {
        let mut inner = self.lock();
        let reclaimed = inner.stores.remove_unused();
        inner.prune_dead();
        reclaimed
    }
}

impl Default for AssetServer {
    fn default() -> Self {
        Self::new()
    }
}

impl AssetServerInner {
    /// The core load pipeline. Returns a strong handle to the (reserved and
    /// then fulfilled) asset, or `None` if no type could be resolved for an
    /// untyped load or no arena exists for the resolved type.
    fn load_path(
        &mut self,
        path: AssetPath,
        requested: Option<AssetTypeId>,
    ) -> Option<UntypedHandle> {
        // 1. Dedup / cycle guard: an already-interned path (resident or
        //    in-flight) short-circuits to the existing handle.
        if let Some(handle) = self
            .interned
            .get(&path)
            .and_then(UntypedWeakHandle::upgrade)
        {
            return Some(handle);
        }

        // 2. Resolve the produced type: the caller's request wins; otherwise ask
        //    loader selection what this path would decode to.
        let type_id = match requested {
            Some(ty) => ty,
            None => {
                let lid = self.loaders.policy().resolve_untyped(&path)?;
                self.loaders.policy().produced_type(lid)?
            }
        };

        // 3. Reserve the slot before any I/O so a dependency cycling back to
        //    this path finds the in-flight handle in step 1.
        let handle = self.stores.reserve_by_type(type_id)?;
        let id = handle.id();

        // 4. Intern (weak) and catalog before decoding.
        self.interned.insert(path.clone(), handle.downgrade());
        self.catalog
            .insert(StableGuid::from_path(&path.to_string()), path.clone());

        // 5. Register the graph node and mark it loading.
        self.graph.add_asset(id);
        let _ = self.graph.set_self_state(id, LoadState::Loading);

        // 6. Read, decode, wire dependencies and sub-assets, fulfill.
        self.finalize(path, id, type_id);
        Some(handle)
    }

    /// Reads bytes for `path`, decodes them as `type_id`, interns labeled
    /// sub-assets, recursively loads dependencies and wires the graph, then
    /// fulfills the reserved slot `id`. Any failure transitions `id` to
    /// [`LoadState::Failed`] with a recorded [`AssetError`].
    fn finalize(&mut self, path: AssetPath, id: UntypedAssetId, type_id: AssetTypeId) {
        let bytes = match self.sources.read(&path) {
            Ok(bytes) => bytes,
            Err(err) => {
                self.fail(id, &path, err.to_string());
                return;
            }
        };

        // Always decode via the requested type so the produced value lands in
        // exactly the arena we reserved against.
        let output = match self.loaders.load_for_type(&path, type_id, &bytes) {
            Ok(output) => output,
            Err(err) => {
                self.fail(id, &path, err.to_string());
                return;
            }
        };
        let FullLoadOutput { primary, labeled } = output;
        if primary.type_id != type_id {
            self.fail(id, &path, "loader produced an unexpected asset type");
            return;
        }
        let ErasedLoadedAsset {
            value,
            dependencies,
            ..
        } = primary;

        // Everything this asset must keep alive: its sub-assets and its wired
        // dependencies (and, transitively, their closures via their own
        // retained entries).
        let mut retained: Vec<UntypedHandle> = Vec::new();

        // Intern labeled sub-assets first; they are fully decoded leaves of
        // this load and are owned by this asset.
        for sub in labeled {
            if let Some(sub_handle) = self.intern_labeled(&path, sub, &mut retained) {
                retained.push(sub_handle);
            }
        }

        // Recursively resolve declared dependencies and wire the graph.
        for dep in dependencies {
            let Some(dep_handle) = self.load_path(dep.path.clone(), dep.expected_type) else {
                self.fail(
                    id,
                    &path,
                    alloc::format!("dependency could not be resolved: {}", dep.path),
                );
                return;
            };
            match self.graph.try_add_dependency(id, dep_handle.id()) {
                Ok(()) => retained.push(dep_handle),
                Err(DependencyError::Cycle { participants }) => {
                    // Record a diagnostic but drop the handle: retaining a
                    // cycle-closing edge would form a reference-counting leak.
                    let eid = self.errors.record(AssetError::new(
                        path.path(),
                        alloc::format!(
                            "dependency cycle rejected ({} asset(s)): {}",
                            participants.len(),
                            dep.path
                        ),
                    ));
                    let _ = eid;
                }
            }
        }

        match self.stores.fulfill_erased(id, value) {
            Ok(_) => {
                let _ = self.graph.set_self_state(id, LoadState::Loaded);
                // Replace (not merge): on reload this drops the previous
                // closure's strong references deterministically.
                self.retained.insert(id, retained);
            }
            Err(err) => self.fail(id, &path, err.to_string()),
        }
    }

    /// Interns one labeled sub-asset under `parent#label`, registering its arena
    /// lazily (the sub-asset's concrete type may have no standalone loader),
    /// inserting the decoded value, wiring the graph, and resolving the
    /// sub-asset's own declared dependencies into `retained`. Returns the strong
    /// handle for the parent to retain, or `None` if routing the value failed
    /// (a diagnostic is recorded in that case).
    fn intern_labeled(
        &mut self,
        parent: &AssetPath,
        sub: ErasedLoadedAsset,
        retained: &mut Vec<UntypedHandle>,
    ) -> Option<UntypedHandle> {
        let ErasedLoadedAsset {
            type_id,
            value,
            dependencies,
            label,
            make_store,
        } = sub;
        let sub_path = parent.clone().with_label(label);

        // On reload the sub-path may already be interned: overwrite in place.
        let handle = if let Some(existing) = self
            .interned
            .get(&sub_path)
            .and_then(UntypedWeakHandle::upgrade)
        {
            let sid = existing.id();
            match self.stores.fulfill_erased(sid, value) {
                Ok(_) => existing,
                Err(err) => {
                    self.fail(sid, &sub_path, err.to_string());
                    return None;
                }
            }
        } else {
            self.stores.register_erased(type_id, make_store);
            let handle = match self.stores.insert_erased(type_id, value) {
                Ok(handle) => handle,
                Err(err) => {
                    let eid = self
                        .errors
                        .record(AssetError::new(sub_path.path(), err.to_string()));
                    let _ = eid;
                    return None;
                }
            };
            self.interned.insert(sub_path.clone(), handle.downgrade());
            self.catalog.insert(
                StableGuid::from_path(&sub_path.to_string()),
                sub_path.clone(),
            );
            handle
        };

        let sid = handle.id();
        self.graph.add_asset(sid);
        let _ = self.graph.set_self_state(sid, LoadState::Loaded);

        // Wire the sub-asset's own dependencies; the parent retains them.
        for dep in dependencies {
            if let Some(dep_handle) = self.load_path(dep.path.clone(), dep.expected_type) {
                match self.graph.try_add_dependency(sid, dep_handle.id()) {
                    Ok(()) => retained.push(dep_handle),
                    Err(DependencyError::Cycle { participants }) => {
                        let eid = self.errors.record(AssetError::new(
                            sub_path.path(),
                            alloc::format!(
                                "sub-asset dependency cycle rejected ({} asset(s)): {}",
                                participants.len(),
                                dep.path
                            ),
                        ));
                        let _ = eid;
                    }
                }
            }
        }

        Some(handle)
    }

    /// Records an [`AssetError`] and transitions `id` to
    /// [`LoadState::Failed`].
    fn fail(&mut self, id: UntypedAssetId, path: &AssetPath, reason: impl Into<String>) {
        let eid = self.errors.record(AssetError::new(path.path(), reason));
        self.stores.fail_erased(id, eid);
        let _ = self.graph.set_self_state(id, LoadState::Failed(eid));
    }

    /// Reserves a slot for type `A`, immediately fails it with `reason`, and
    /// returns the typed handle. Used by the public API to surface a load that
    /// could not even begin (null/unknown soft handle, missing arena) as a
    /// first-class [`Failed`](crate::LoadState::Failed) handle rather than a
    /// panic or a silent `None`.
    fn reserve_failed<A: Asset>(&mut self, reason: impl Into<String>) -> Handle<A> {
        self.stores.register::<A>();
        let handle = self.stores.reserve::<A>();
        let id = handle.id();
        self.graph.add_asset(id);
        let eid = self.errors.record(AssetError::new("<unresolved>", reason));
        self.stores.fail_erased(id, eid);
        let _ = self.graph.set_self_state(id, LoadState::Failed(eid));
        handle
            .typed::<A>()
            .expect("reserved slot carries the requested type")
    }

    /// Prunes dead intern entries and the retained closures of reclaimed
    /// owners. Returns whether any bookkeeping entry was removed (used to drive
    /// the [`AssetServerInner::collect_releases`] fixpoint).
    fn prune_dead(&mut self) -> bool {
        let dead_paths: Vec<AssetPath> = self
            .interned
            .iter()
            .filter(|(_, weak)| weak.strong_count() == 0)
            .map(|(path, _)| path.clone())
            .collect();
        for path in &dead_paths {
            self.interned.remove(path);
        }

        let dead_owners: Vec<UntypedAssetId> = self
            .retained
            .keys()
            .copied()
            .filter(|id| !self.stores.contains(*id))
            .collect();
        for id in &dead_owners {
            self.retained.remove(id);
            self.graph.remove_asset(*id);
        }

        !dead_paths.is_empty() || !dead_owners.is_empty()
    }

    /// Advances arena grace frames and prunes dead bookkeeping to a fixpoint,
    /// returning the total number of slots reclaimed.
    fn collect_releases(&mut self) -> usize {
        let mut total = 0;
        loop {
            let reclaimed = self.stores.collect_releases();
            let pruned = self.prune_dead();
            total += reclaimed;
            if reclaimed == 0 && !pruned {
                break;
            }
        }
        total
    }
}
