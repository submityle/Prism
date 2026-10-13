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
use crate::reload::{ReloadCommit, ReloadEpoch};
use crate::reload_plan::ReloadPlan;
use crate::source::{AssetReader, AssetSources};
use crate::stores::AssetStores;
use crate::type_id::AssetTypeId;
use crate::watch::{AssetChangeKind, ChangeBatch};
#[cfg(feature = "file_watcher")]
use crate::watch::{AssetWatchHandle, WatchStartError};
use crate::{Asset, LoaderId, SuffixConflict};
use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;
use std::panic::{catch_unwind, AssertUnwindSafe};
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
    /// Labels whose lifetime has been promoted from their source revision to
    /// the label itself. This records promotions that happen recursively while
    /// the source is still assembling, before its retained closure is
    /// published.
    promoted_labels: BTreeMap<UntypedAssetId, UntypedAssetId>,
    /// Epoch arbitration for reload results. It is also useful to async
    /// callers that decode outside this server's state mutex.
    reload_epoch: ReloadEpoch,
}

/// A labeled value held outside storage until the complete reload assembly has
/// passed dependency, cycle, identity, and type validation. Keeping this
/// staging record separate from [`AssetServerInner::graph`] prevents one bad
/// label from committing a partially revised sub-asset set.
struct StagedLabeledAsset {
    path: AssetPath,
    id: UntypedAssetId,
    value: Box<dyn core::any::Any + Send>,
    dependencies: Vec<UntypedHandle>,
    handle: UntypedHandle,
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
                promoted_labels: BTreeMap::new(),
                reload_epoch: ReloadEpoch::new(),
            }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, AssetServerInner> {
        match self.inner.lock() {
            Ok(guard) => guard,
            // A loader/source is third-party code and must not permanently
            // brick the server if it panics at the boundary. The load path
            // catches such panics and records a failure; this recovery also
            // makes diagnostics and subsequent requests usable if a panic
            // escaped an older/custom entry point.
            Err(poisoned) => poisoned.into_inner(),
        }
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

    /// Starts a managed platform watcher for a filesystem source.
    ///
    /// The returned handle owns the watcher consumer and must be retained for
    /// as long as hot reload is desired. Dropping it synchronously stops the
    /// consumer and the underlying debouncer. Event collection and batch
    /// application happen off the caller thread; unknown paths remain ignored
    /// by [`Self::apply_change_batch`].
    #[cfg(feature = "file_watcher")]
    pub fn watch(
        self: &Arc<Self>,
        root: impl Into<std::path::PathBuf>,
        source: impl Into<String>,
        window: std::time::Duration,
    ) -> Result<AssetWatchHandle, WatchStartError> {
        AssetWatchHandle::spawn(Arc::clone(self), root, source, window)
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
            Some(handle) => handle.typed::<A>().unwrap_or_else(|| {
                inner.reserve_failed::<A>("asset path is already interned with another type")
            }),
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
            Some(handle) => handle.typed::<A>().unwrap_or_else(|| {
                inner.reserve_failed::<A>("soft asset path is already interned with another type")
            }),
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
        let requested_path = path.into();
        let mut inner = self.lock();
        // A label is an output of its source loader, not an independently
        // reloadable file. Reloading `scene.gltf#Mesh0` must therefore use the
        // primary source as the transaction root; otherwise the old code
        // marked the label stale, skipped it in the reload loop, and left the
        // source loader's complete labeled revision untouched.
        let path = requested_path.clone().without_label();
        let guard = inner.reload_epoch.begin();
        let result = match guard.commit(|| {
            if let Some(handle) = inner
                .interned
                .get(&path)
                .and_then(UntypedWeakHandle::upgrade)
            {
                let id = handle.id();
                let affected = inner.graph.invalidate(id);
                let affected: BTreeSet<_> = affected.into_iter().collect();
                let order = ReloadPlan::build(&inner.graph, affected.iter().copied())
                    .expect("the live dependency graph is acyclic");
                for affected_id in order {
                    if !affected.contains(&affected_id) {
                        continue;
                    }
                    // Labeled sub-assets are committed by reloading their
                    // owning source asset; decoding a label directly would
                    // route the source loader's primary value into the wrong
                    // typed slot. The owning source is included in the graph
                    // closure when it depends on this sub-asset.
                    let Some((affected_path, affected_handle)) =
                        inner.interned.iter().find_map(|(candidate, weak)| {
                            weak.upgrade()
                                .filter(|candidate_handle| candidate_handle.id() == affected_id)
                                .map(|candidate_handle| (candidate.clone(), candidate_handle))
                        })
                    else {
                        continue;
                    };
                    if affected_path.has_label() {
                        inner.graph.clear_stale(affected_id);
                        continue;
                    }
                    inner.graph.clear_stale(affected_id);
                    let _ = inner.graph.set_self_state(affected_id, LoadState::Loading);
                    inner.finalize_safe(affected_path, affected_id, affected_id.type_id());
                    drop(affected_handle);
                }
                Some(handle)
            } else {
                // Preserve the public contract for an unknown label: loading
                // the source is allowed to intern the label, but the return
                // value is the exact requested handle when it exists.
                let source_handle = inner.load_path(path.clone(), None)?;
                if requested_path.has_label() {
                    inner
                        .interned
                        .get(&requested_path)
                        .and_then(UntypedWeakHandle::upgrade)
                } else {
                    Some(source_handle)
                }
            }
        }) {
            ReloadCommit::Committed(handle) => handle.and_then(|handle| {
                if requested_path.has_label() {
                    inner
                        .interned
                        .get(&requested_path)
                        .and_then(UntypedWeakHandle::upgrade)
                } else {
                    Some(handle)
                }
            }),
            ReloadCommit::Superseded => None,
        };
        result
    }

    /// Applies one debounced watcher batch to resident assets.
    ///
    /// The watcher deliberately does not own an [`AssetServer`], which keeps
    /// platform event collection independent from loader execution. This
    /// method is the concrete hand-off between the two: every change is
    /// processed in stable batch order and each affected source is reloaded
    /// through the same dependency-aware, atomic assembly path as manual
    /// [`AssetServer::reload`]. Unknown paths are ignored because a filesystem
    /// watcher must not implicitly load every file in a project. A rename
    /// reloads both the old resident path (so a removed source is diagnosed)
    /// and the new resident path; it does not silently change stable identity.
    pub fn apply_change_batch(&self, batch: &ChangeBatch) -> Vec<UntypedHandle> {
        let mut handles = Vec::new();
        for change in batch.changes() {
            let path = qualify_change_path(&change.source, &change.path);
            match &change.kind {
                AssetChangeKind::Created | AssetChangeKind::Modified | AssetChangeKind::Removed => {
                    if self.is_resident_source(&path)
                        && let Some(handle) = self.reload(path)
                    {
                        handles.push(handle);
                    }
                }
                AssetChangeKind::Renamed { from, to } => {
                    let from = qualify_change_path(&change.source, from);
                    let to = qualify_change_path(&change.source, to);
                    if self.is_resident_source(&from)
                        && let Some(handle) = self.reload(from)
                    {
                        handles.push(handle);
                    }
                    if self.is_resident_source(&to)
                        && let Some(handle) = self.reload(to)
                    {
                        handles.push(handle);
                    }
                }
            }
        }
        handles
    }

    fn is_resident_source(&self, path: &AssetPath) -> bool {
        let source_path = path.clone().without_label();
        let inner = self.lock();
        inner.interned.iter().any(|(candidate, weak)| {
            candidate.clone().without_label() == source_path && weak.strong_count() > 0
        })
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

fn qualify_change_path(source: &str, path: &AssetPath) -> AssetPath {
    if source.is_empty() || path.has_scheme() {
        path.clone()
    } else {
        path.clone().with_scheme(source)
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
        // A label names an output produced by its source asset's loader, not a
        // file with an independently selectable suffix. Resolve an existing
        // label before suffix dispatch, or load the source first so its labels
        // are registered, then return the exact interned sub-asset handle.
        if let Some(handle) = self
            .interned
            .get(&path)
            .and_then(UntypedWeakHandle::upgrade)
        {
            // A label is a revision output, not an independently persistent
            // asset.  Once a source revision removed it, the old slot is
            // deliberately failed so existing consumers can observe the
            // invalidation.  Do not let a later `load_soft`/`load_untyped`
            // request keep returning that failed historical slot: remove the
            // weak catalog entry and let the source revision decide whether a
            // replacement label exists.
            if path.has_label() && self.stores.load_state(handle.id()).is_failed() {
                self.interned.remove(&path);
                let guid = StableGuid::from_path(&path.to_string());
                if self.catalog.get(&guid) == Some(&path) {
                    self.catalog.remove(&guid);
                }
            } else {
                let handle = requested
                    .is_none_or(|type_id| handle.id().type_id() == type_id)
                    .then_some(handle)?;
                if path.has_label() {
                    if let Some(source) = self
                        .interned
                        .get(&path.clone().without_label())
                        .and_then(UntypedWeakHandle::upgrade)
                    {
                        self.promote_label_ownership(&handle, &source);
                    }
                }
                return Some(handle);
            }
        }
        if path.has_label() {
            let source_path = path.clone().without_label();
            // Keep the source handle alive while transferring ownership to the
            // label. The source loader may have been reached only recursively;
            // dropping this temporary before promotion would make the weak
            // intern entry look dead and lose the source->label hand-off.
            let source_handle = self.load_path(source_path, None)?;
            let handle = self
                .interned
                .get(&path)
                .and_then(UntypedWeakHandle::upgrade)?;
            let handle = requested
                .is_none_or(|type_id| handle.id().type_id() == type_id)
                .then_some(handle)?;
            self.promote_label_ownership(&handle, &source_handle);
            return Some(handle);
        }

        // Resolve the requested type before deduplication. A path may be
        // requested through two typed APIs; returning the first path's handle
        // and letting a later typed downcast panic would turn a content error
        // into process failure.
        let type_id = match requested {
            Some(ty) => ty,
            None => {
                let lid = self.loaders.policy().resolve_untyped(&path)?;
                self.loaders.policy().produced_type(lid)?
            }
        };

        // 1. Dedup / cycle guard: an already-interned path (resident or
        //    in-flight) short-circuits to the existing handle.
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
        self.finalize_safe(path, id, type_id);
        Some(handle)
    }

    /// Transfers the temporary source-owned reference for a labeled output to
    /// the label once a caller actually requests that output. The source keeps
    /// labels alive until they are first resolved so nested loads can observe
    /// them; after transfer the label keeps its source revision alive instead.
    /// This avoids both a weak-label race during initial assembly and a
    /// permanent source <-> label strong-reference cycle.
    fn promote_label_ownership(&mut self, label: &UntypedHandle, source: &UntypedHandle) {
        let source_id = source.id();
        let label_id = label.id();
        if let Some(retained) = self.retained.get_mut(&source_id) {
            retained.retain(|handle| handle.id() != label_id);
        }
        self.promoted_labels.insert(label_id, source_id);
        let label_retained = self.retained.entry(label_id).or_default();
        if !label_retained.iter().any(|handle| handle.id() == source_id) {
            label_retained.push(source.clone());
        }
    }

    /// Executes format/plugin code behind a panic boundary. A panic is a load
    /// failure, never a reason to poison the server mutex or leave a reserved
    /// slot looking permanently in-flight.
    fn finalize_safe(&mut self, path: AssetPath, id: UntypedAssetId, type_id: AssetTypeId) {
        let failure_path = path.clone();
        if catch_unwind(AssertUnwindSafe(|| self.finalize(path, id, type_id))).is_err() {
            self.fail(id, &failure_path, "asset source or loader panicked");
        }
    }

    /// Reads bytes for `path`, decodes them as `type_id`, interns labeled
    /// sub-assets, recursively loads dependencies and wires the graph, then
    /// fulfills the reserved slot `id`. A first-load failure transitions the
    /// slot to [`LoadState::Failed`] with a recorded [`AssetError`]; a reload
    /// failure keeps the existing ready payload and restores `Loaded` state.
    fn finalize(&mut self, path: AssetPath, id: UntypedAssetId, type_id: AssetTypeId) {
        let had_ready_value = matches!(self.stores.load_state(id), LoadState::Loaded);
        let previous_labels = self.labels_for_source(&path);
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

        // Resolve all dependencies first, but do not mutate graph edges one by
        // one. Reloads may remove dependencies, so the edge set is committed
        // as one replacement after every dependency has been resolved.
        let mut dependency_handles: Vec<UntypedHandle> = Vec::new();
        let mut dependency_ids: Vec<UntypedAssetId> = Vec::new();
        for dep in dependencies {
            let Some(dep_handle) = self.load_path(dep.path.clone(), dep.expected_type) else {
                self.fail(
                    id,
                    &path,
                    alloc::format!("dependency could not be resolved: {}", dep.path),
                );
                return;
            };
            if self.stores.load_state(dep_handle.id()).is_failed() {
                self.fail(
                    id,
                    &path,
                    alloc::format!("dependency failed to load: {}", dep.path),
                );
                return;
            }
            dependency_ids.push(dep_handle.id());
            dependency_handles.push(dep_handle);
        }
        // Stage every labeled value and its dependency set before changing the
        // real graph or replacing any existing labeled payload. In
        // particular, a type change at `path#label` must reject the complete
        // parent revision instead of silently retaining the old typed handle.
        let mut staged_labels = Vec::with_capacity(labeled.len());
        let mut provisional_labels: Vec<(AssetPath, UntypedAssetId)> = Vec::new();
        let mut label_paths = BTreeSet::new();
        for sub in labeled {
            let sub_path = path.clone().with_label(sub.label.clone());
            if !label_paths.insert(sub_path.clone()) {
                self.rollback_staged_labels(&provisional_labels);
                self.fail(id, &path, "loader emitted a duplicate labeled asset");
                return;
            }
            let handle = if let Some(existing) = self
                .interned
                .get(&sub_path)
                .and_then(UntypedWeakHandle::upgrade)
            {
                if existing.id().type_id() != sub.type_id {
                    self.rollback_staged_labels(&provisional_labels);
                    self.fail(
                        id,
                        &path,
                        alloc::format!("labeled asset type changed at {} during reload", sub_path),
                    );
                    return;
                }
                existing
            } else {
                self.stores.register_erased(sub.type_id, sub.make_store);
                let Some(handle) = self.stores.reserve_by_type(sub.type_id) else {
                    self.rollback_staged_labels(&provisional_labels);
                    self.fail(id, &path, "labeled asset arena unavailable");
                    return;
                };
                self.interned.insert(sub_path.clone(), handle.downgrade());
                self.catalog.insert(
                    StableGuid::from_path(&sub_path.to_string()),
                    sub_path.clone(),
                );
                provisional_labels.push((sub_path.clone(), handle.id()));
                handle
            };

            let mut label_dependencies = Vec::new();
            let mut label_dependency_ids = Vec::new();
            for dep in sub.dependencies {
                let Some(dep_handle) = self.load_path(dep.path.clone(), dep.expected_type) else {
                    self.rollback_staged_labels(&provisional_labels);
                    self.fail(
                        id,
                        &path,
                        alloc::format!("labeled dependency could not be resolved: {}", dep.path),
                    );
                    return;
                };
                if self.stores.load_state(dep_handle.id()).is_failed() {
                    self.rollback_staged_labels(&provisional_labels);
                    self.fail(
                        id,
                        &path,
                        alloc::format!("labeled dependency failed to load: {}", dep.path),
                    );
                    return;
                }
                label_dependency_ids.push(dep_handle.id());
                label_dependencies.push(dep_handle);
            }
            staged_labels.push(StagedLabeledAsset {
                path: sub_path,
                id: handle.id(),
                value: sub.value,
                dependencies: label_dependencies,
                handle,
            });
        }

        // Nested dependency loads may have extended the live graph while the
        // values above were being decoded. Clone only now, then apply the
        // complete candidate revision to that latest graph snapshot.
        let mut candidate_graph = self.graph.clone();
        match candidate_graph.replace_dependencies(id, dependency_ids.clone()) {
            Ok(()) => retained.extend(dependency_handles),
            Err(DependencyError::Cycle { participants }) if had_ready_value => {
                self.rollback_staged_labels(&provisional_labels);
                self.fail(
                    id,
                    &path,
                    alloc::format!(
                        "dependency cycle rejected during reload assembly ({} asset(s))",
                        participants.len()
                    ),
                );
                return;
            }
            Err(DependencyError::Cycle { .. }) => {
                // Initial loads tolerate a cycle by dropping offending edges;
                // reloads above reject the new revision transactionally.
                for (dependency_id, dependency_handle) in
                    dependency_ids.iter().copied().zip(dependency_handles)
                {
                    if candidate_graph
                        .try_add_dependency(id, dependency_id)
                        .is_ok()
                    {
                        retained.push(dependency_handle);
                    }
                }
            }
        }
        for staged in &staged_labels {
            // A labeled output is owned by this source revision. Represent
            // that ownership in the dependency graph so invalidating the
            // source also reaches consumers of the label, and the reload
            // plan orders the source before the label's consumers.
            let label_dependencies = staged
                .dependencies
                .iter()
                .map(UntypedHandle::id)
                .chain(core::iter::once(id));
            if let Err(DependencyError::Cycle { participants }) =
                candidate_graph.replace_dependencies(staged.id, label_dependencies)
            {
                self.rollback_staged_labels(&provisional_labels);
                self.fail(
                    id,
                    &path,
                    alloc::format!(
                        "labeled asset dependency cycle rejected ({} asset(s))",
                        participants.len()
                    ),
                );
                return;
            }
        }

        // All graph changes are now known to be valid. Publish the graph as a
        // single revision, then fulfill the staged values. The type checks
        // above make fulfillment infallible for well-formed loaders.
        self.graph = candidate_graph;
        for staged in staged_labels {
            if self.stores.fulfill_erased(staged.id, staged.value).is_err() {
                self.fail(
                    staged.id,
                    &staged.path,
                    "staged labeled asset type mismatch",
                );
                self.fail(id, &path, "staged labeled asset could not be committed");
                return;
            }
            let _ = self.graph.set_self_state(staged.id, LoadState::Loaded);
            retained.extend(staged.dependencies);
            // Keep the label alive through its source until a real label
            // consumer takes ownership via `promote_label_ownership`. This is
            // required because the intern table is intentionally weak.
            if self.promoted_labels.get(&staged.id) != Some(&id) {
                retained.push(staged.handle);
            }
        }

        match self.stores.fulfill_erased(id, value) {
            Ok(_) => {
                let _ = self.graph.set_self_state(id, LoadState::Loaded);
                // Replace (not merge): on reload this drops the previous
                // closure's strong references deterministically.
                self.retained.insert(id, retained);
                self.retire_removed_labels(&path, &previous_labels, &label_paths);
            }
            Err(err) => self.fail(id, &path, err.to_string()),
        }
    }

    /// Discards labels created while assembling a source revision that never
    /// reached the graph/storage commit point.  A failed revision must not
    /// publish ghost paths through either the weak interning table or the soft
    /// reference catalog.  The failed slots remain observable through any
    /// temporary handles until the normal release collector reclaims them.
    fn rollback_staged_labels(&mut self, labels: &[(AssetPath, UntypedAssetId)]) {
        for (path, id) in labels {
            let same_id = self
                .interned
                .get(path)
                .and_then(UntypedWeakHandle::upgrade)
                .is_some_and(|handle| handle.id() == *id);
            if same_id {
                self.interned.remove(path);
            }
            let guid = StableGuid::from_path(&path.to_string());
            if self.catalog.get(&guid) == Some(path) {
                self.catalog.remove(&guid);
            }
            let error = self.errors.record(AssetError::new(
                path.path(),
                "labeled asset staging rolled back",
            ));
            let _ = self.stores.fail_erased(*id, error);
            self.promoted_labels.remove(id);
        }
    }

    /// Returns the currently interned labeled outputs of `source`, including
    /// their stable slot ids. The weak intern table is the source of truth for
    /// labels that belong to the previous source revision.
    fn labels_for_source(&self, source: &AssetPath) -> Vec<(AssetPath, UntypedAssetId)> {
        self.interned
            .iter()
            .filter(|(candidate, _)| {
                candidate.has_label() && (*candidate).clone().without_label() == *source
            })
            .filter_map(|(candidate, weak)| {
                weak.upgrade()
                    .map(|handle| (candidate.clone(), handle.id()))
            })
            .collect()
    }

    /// Invalidates labels that disappeared from a successfully committed
    /// source revision. Existing external handles keep their identity, but
    /// they must no longer expose the old ready value or remain resolvable via
    /// the soft-reference catalog.
    fn retire_removed_labels(
        &mut self,
        source: &AssetPath,
        previous: &[(AssetPath, UntypedAssetId)],
        current: &BTreeSet<AssetPath>,
    ) {
        let Some(source_handle) = self
            .interned
            .get(source)
            .and_then(UntypedWeakHandle::upgrade)
        else {
            return;
        };
        let source_id = source_handle.id();
        for (label_path, label_id) in previous {
            if current.contains(label_path) {
                continue;
            }
            if let Some(retained) = self.retained.get_mut(&source_id) {
                retained.retain(|handle| handle.id() != *label_id);
            }
            self.retained.remove(label_id);
            self.promoted_labels.remove(label_id);
            let guid = StableGuid::from_path(&label_path.to_string());
            if self.catalog.get(&guid) == Some(label_path) {
                self.catalog.remove(&guid);
            }
            self.interned.remove(label_path);
            let error = self.errors.record(AssetError::new(
                label_path.path(),
                "labeled asset was removed from the source revision",
            ));
            let _ = self.stores.fail_erased(*label_id, error);
            let _ = self
                .graph
                .set_self_state(*label_id, LoadState::Failed(error));
        }
    }

    /// Records an [`AssetError`] and transitions `id` to
    /// [`LoadState::Failed`].
    fn fail(&mut self, id: UntypedAssetId, path: &AssetPath, reason: impl Into<String>) {
        let eid = self.errors.record(AssetError::new(path.path(), reason));
        // A failed reload must not destroy the last known-good payload. The
        // graph was marked Loading before reassembly, but the storage slot is
        // still Ready until a successful fulfill replaces it.
        if matches!(self.stores.load_state(id), LoadState::Loaded) {
            let _ = self.graph.set_self_state(id, LoadState::Loaded);
            return;
        }
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

        self.promoted_labels
            .retain(|label, source| self.stores.contains(*label) && self.stores.contains(*source));

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
