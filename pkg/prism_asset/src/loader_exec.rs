//! Loader execution protocol: the typed [`AssetLoader`] trait, the
//! [`LoadContext`] a loader fills in, and the type-erased plumbing that lets an
//! [`AssetServer`](crate::AssetServer) drive loaders it only knows by
//! [`AssetTypeId`] (design §9.1, execution half).
//!
//! ## Split from the selection policy
//! [`LoaderRegistry`](crate::LoaderRegistry) answers *which* loader claims a
//! path. This module is *how* that loader turns bytes into a value plus its
//! declared dependencies and labeled sub-assets. [`AssetLoaders`] below marries
//! the two: it owns the policy table and the parallel table of erased loader
//! objects keyed by the same [`LoaderId`](crate::LoaderId).
//!
//! ## Why the typed loader is synchronous
//! A loader's job is CPU work: parse bytes, build a value. The engine already
//! owns the concurrency — the scheduler reads bytes on an I/O worker and runs
//! `load` on a CPU worker — so the loader itself stays a plain, deterministic,
//! trivially-testable `fn(&[u8], &mut LoadContext) -> Result<Asset, LoadError>`.
//! Keeping async out of the loader trait means a format plugin is a pure
//! function of its input, which is exactly what the determinism and golden
//! tests (§25) require.
//!
//! ## Dependencies and sub-assets
//! A loader does not resolve dependencies itself. It *declares* them on the
//! [`LoadContext`] by path (and optionally a typed [`DepRequest`]); the server
//! interns those paths into handles and schedules their loads, wiring the
//! dependency graph. Likewise a multi-asset source (a glTF with meshes,
//! materials, animations) emits labeled sub-assets via
//! [`LoadContext::add_labeled_asset`], which the server stores under
//! `path#label` identities.

#![cfg(feature = "std")]

use crate::path::AssetPath;
use crate::source::{AssetMeta, AssetSources, ResolvedAssetRead};
use crate::stores::ErasedAssetStore;
use crate::streaming::{
    assemble_ranges_resolved, assemble_ranges_resolved_async, collect_range_parts,
    RangeAssemblyCancellation, RangeAssemblyError, RangeAssemblyLimits, RangeAssemblyPlan,
    RangeAssemblyPlanError, RangePart,
};
use crate::type_id::AssetTypeId;
use crate::{Asset, Assets, LoaderId, LoaderRegistry, SuffixConflict};
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::any::Any;
use core::fmt;

/// Why a loader failed to decode its input. Loaders map their format-specific
/// errors into one of these so the scheduler can treat failures uniformly while
/// still surfacing a human-readable detail.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum LoadError {
    /// The bytes are malformed for this format (truncated, bad magic, bad
    /// checksum). The string is a human-readable detail.
    Malformed(String),
    /// The format is recognized but uses a feature this loader does not
    /// implement (an unsupported codec, version, or extension).
    Unsupported(String),
    /// A declared dependency could not be resolved or itself failed to load.
    DependencyFailed(String),
    /// A security limit was exceeded while decoding untrusted bytes (decompress
    /// bomb, oversized allocation, recursion depth). Enforced by loaders over
    /// untrusted sources (§23.7).
    LimitExceeded(String),
    /// A catch-all for loader-internal errors that do not fit the above.
    Other(String),
}

/// The physical bytes a loader needs before it can decode an asset.
///
/// A loader still receives one contiguous byte slice. `Ranges` therefore
/// concatenates the declared physical ranges in declaration order before
/// calling [`AssetLoader::load`]. This is useful for formats whose manifest or
/// route already identifies independent sections; it is not a claim that an
/// arbitrary format can decode an incomplete file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AssetReadPlan {
    /// Read the complete source through the range API.
    Whole,
    /// Read and concatenate explicitly declared physical ranges.
    Ranges {
        /// Physical ranges, in the order exposed to the decoder.
        parts: Vec<RangePart>,
        /// Limits applied before I/O starts.
        limits: RangeAssemblyLimits,
    },
}

impl AssetReadPlan {
    /// Creates a bounded explicit range plan using the standard limits.
    pub fn ranges(
        parts: impl IntoIterator<Item = RangePart>,
    ) -> Result<Self, RangeAssemblyPlanError> {
        Self::ranges_with_limits(parts, RangeAssemblyLimits::default())
    }

    /// Creates an explicit range plan with caller-selected limits.
    pub fn ranges_with_limits(
        parts: impl IntoIterator<Item = RangePart>,
        limits: RangeAssemblyLimits,
    ) -> Result<Self, RangeAssemblyPlanError> {
        if limits.max_parts == 0 {
            return Err(RangeAssemblyPlanError::ZeroPartLimit);
        }
        if limits.max_parallel == 0 {
            return Err(RangeAssemblyPlanError::ZeroParallelLimit);
        }
        Ok(Self::Ranges {
            parts: collect_range_parts(parts, limits.max_parts)?,
            limits,
        })
    }

    fn validate(self, path: &AssetPath, source_len: u64) -> Result<RangeAssemblyPlan, LoadError> {
        match self {
            Self::Whole => RangeAssemblyPlan::whole(
                path.clone(),
                source_len,
                RangeAssemblyLimits {
                    max_parts: 1,
                    max_bytes: source_len,
                    max_parallel: 1,
                },
            ),
            Self::Ranges { parts, limits } => {
                RangeAssemblyPlan::new(path.clone(), source_len, parts, limits)
            }
        }
        .map_err(|error| LoadError::LimitExceeded(error.to_string()))
    }
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Malformed(d) => write!(f, "malformed asset: {d}"),
            LoadError::Unsupported(d) => write!(f, "unsupported feature: {d}"),
            LoadError::DependencyFailed(d) => write!(f, "dependency failed: {d}"),
            LoadError::LimitExceeded(d) => write!(f, "decode limit exceeded: {d}"),
            LoadError::Other(d) => write!(f, "load error: {d}"),
        }
    }
}

impl std::error::Error for LoadError {}

/// A dependency a loader declares while decoding. The path is required; an
/// optional expected type lets the server type-check the dependency's loader
/// selection up front (a `Material` declaring its `base_color` is an `Image`).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DepRequest {
    /// Where the dependency lives (may carry a scheme and `#label`).
    pub path: AssetPath,
    /// The asset type the loader expects, if known. `None` lets the server pick
    /// by untyped loader priority.
    pub expected_type: Option<AssetTypeId>,
}

impl DepRequest {
    /// A dependency declared only by path.
    #[must_use]
    pub fn new(path: AssetPath) -> Self {
        Self {
            path,
            expected_type: None,
        }
    }

    /// A dependency declared with the expected produced type.
    #[must_use]
    pub fn typed(path: AssetPath, expected_type: AssetTypeId) -> Self {
        Self {
            path,
            expected_type: Some(expected_type),
        }
    }
}

/// A fully type-erased loaded sub-asset: the decoded value boxed behind `Any`,
/// tagged with its [`AssetTypeId`], plus its own declared dependencies. The
/// server downcasts `value` into the right `Assets<A>` arena by matching
/// `type_id`.
pub struct ErasedLoadedAsset {
    /// The produced asset's type.
    pub type_id: AssetTypeId,
    /// The decoded value, downcast via [`Any`] into `Assets<A>` by the server.
    pub value: Box<dyn Any + Send>,
    /// Dependencies this (sub-)asset declared.
    pub dependencies: Vec<DepRequest>,
    /// A stable, loader-chosen name for this value when it is a labeled
    /// sub-asset (the `#label` of its identity). Empty for the primary asset.
    pub label: String,
    /// A constructor for an empty arena of this asset's concrete type.
    ///
    /// A labeled sub-asset may be of a type that has no standalone loader (a
    /// `Mesh` extracted from a glTF scene, a mip level of a texture), so its
    /// [`Assets`] arena may not have been registered up front. The erased layer
    /// captures the concrete type here as a non-capturing constructor so the
    /// server can lazily register the arena before routing the value into it,
    /// without the server ever needing to name the type (design §7/§9.2).
    pub make_store: fn() -> Box<dyn ErasedAssetStore>,
}

impl fmt::Debug for ErasedLoadedAsset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The boxed value is `dyn Any` and cannot be formatted; show its shape.
        f.debug_struct("ErasedLoadedAsset")
            .field("type_id", &self.type_id)
            .field("label", &self.label)
            .field("dependencies", &self.dependencies)
            .field("value", &"<boxed>")
            .finish()
    }
}

/// The output of a successful typed load: the primary value plus everything the
/// loader declared about it. The server consumes this to populate storage, wire
/// the dependency graph, and register sub-assets.
pub struct LoadedAsset<A: Asset> {
    /// The decoded primary value.
    pub value: A,
    /// Dependencies the primary value declared, in declaration order.
    pub dependencies: Vec<DepRequest>,
    /// Labeled sub-assets produced alongside the primary (meshes/materials of a
    /// scene, mip levels, …), each erased and tagged.
    pub labeled_assets: Vec<ErasedLoadedAsset>,
}

/// The builder a loader fills while decoding. It accumulates declared
/// dependencies and labeled sub-assets, and exposes the identity of the asset
/// being loaded so a loader can resolve relative dependency paths against it.
pub struct LoadContext<'a> {
    path: &'a AssetPath,
    dependencies: Vec<DepRequest>,
    labeled_assets: Vec<ErasedLoadedAsset>,
}

impl<'a> LoadContext<'a> {
    /// Creates a context for loading `path`.
    #[must_use]
    pub fn new(path: &'a AssetPath) -> Self {
        Self {
            path,
            dependencies: Vec::new(),
            labeled_assets: Vec::new(),
        }
    }

    /// The identity of the asset currently being loaded. Loaders resolve
    /// relative references against `path().path()`.
    #[must_use]
    pub fn path(&self) -> &AssetPath {
        self.path
    }

    /// Declares a dependency by path (untyped). Returns a handle-less
    /// [`AssetPath`] clone the loader can store if it wants to re-reference it.
    pub fn add_dependency(&mut self, path: AssetPath) {
        self.dependencies.push(DepRequest::new(path));
    }

    /// Declares a dependency with its expected produced type.
    pub fn add_typed_dependency(&mut self, path: AssetPath, expected_type: AssetTypeId) {
        self.dependencies
            .push(DepRequest::typed(path, expected_type));
    }

    /// Emits a labeled sub-asset. `label` becomes the `#label` of its identity
    /// under the primary asset's path; `value` is any [`Asset`] type.
    pub fn add_labeled_asset<B: Asset>(&mut self, label: impl Into<String>, value: B) {
        self.labeled_assets.push(ErasedLoadedAsset {
            type_id: AssetTypeId::of::<B>(),
            value: Box::new(value),
            dependencies: Vec::new(),
            label: label.into(),
            make_store: || -> Box<dyn ErasedAssetStore> { Box::new(Assets::<B>::new()) },
        });
    }

    /// Emits a labeled sub-asset that itself declares dependencies.
    pub fn add_labeled_asset_with_deps<B: Asset>(
        &mut self,
        label: impl Into<String>,
        value: B,
        dependencies: Vec<DepRequest>,
    ) {
        self.labeled_assets.push(ErasedLoadedAsset {
            type_id: AssetTypeId::of::<B>(),
            value: Box::new(value),
            dependencies,
            label: label.into(),
            make_store: || -> Box<dyn ErasedAssetStore> { Box::new(Assets::<B>::new()) },
        });
    }

    /// The dependencies declared so far.
    #[must_use]
    pub fn dependencies(&self) -> &[DepRequest] {
        &self.dependencies
    }

    /// Finalizes the context into a [`LoadedAsset`] around `value`.
    #[must_use]
    pub fn finish<A: Asset>(self, value: A) -> LoadedAsset<A> {
        LoadedAsset {
            value,
            dependencies: self.dependencies,
            labeled_assets: self.labeled_assets,
        }
    }

    /// Splits the accumulated declarations out of the context without a primary
    /// value, used by the erased layer.
    fn into_parts(self) -> (Vec<DepRequest>, Vec<ErasedLoadedAsset>) {
        (self.dependencies, self.labeled_assets)
    }
}

/// A typed asset loader: turns bytes into one [`Asset`] value plus declared
/// dependencies and sub-assets. Concrete format decoders (PNG, glTF, WAV, …)
/// implement this in sibling plugin crates; the engine only ever sees the
/// erased form.
pub trait AssetLoader: Send + Sync + 'static {
    /// The primary asset type this loader produces.
    type Asset: Asset;

    /// The file suffixes this loader claims (no leading dot), used to register
    /// it with the selection policy. Compound suffixes like `tar.zst` are
    /// allowed and preferred over shorter tails.
    fn extensions(&self) -> &'static [&'static str];

    /// The registration priority (higher wins ambiguous suffix matches).
    /// Defaults to `0`.
    fn priority(&self) -> i32 {
        0
    }

    /// Declares the source bytes that must be assembled before [`Self::load`]
    /// runs. The default is a single full-source range, so every normal load
    /// uses the same bounded range-backed I/O path as streaming rather than a
    /// separate whole-file read path.
    ///
    /// Implementations may return [`AssetReadPlan::Ranges`] only when their
    /// decoder accepts the concatenation of those ranges in exactly the
    /// declared order. The runtime validates metadata bounds, part and byte
    /// limits, and every returned range length before publishing a revision.
    fn read_plan(
        &self,
        _path: &AssetPath,
        _metadata: AssetMeta,
    ) -> Result<AssetReadPlan, LoadError> {
        Ok(AssetReadPlan::Whole)
    }

    /// Decodes `bytes` into a value, declaring dependencies/sub-assets on `ctx`.
    ///
    /// # Errors
    /// Any [`LoadError`] describing why the bytes could not be decoded.
    fn load(&self, bytes: &[u8], ctx: &mut LoadContext) -> Result<Self::Asset, LoadError>;
}

/// The object-safe erased view of an [`AssetLoader`], stored by the server.
pub trait ErasedAssetLoader: Send + Sync {
    /// The produced asset type.
    fn produced_type(&self) -> AssetTypeId;

    /// Declares the physical input required by this loader.
    fn read_plan(&self, path: &AssetPath, metadata: AssetMeta) -> Result<AssetReadPlan, LoadError>;

    /// Decodes `bytes` for `path`, returning the primary asset together with
    /// every labeled sub-asset the loader emitted on its [`LoadContext`].
    ///
    /// # Errors
    /// Any [`LoadError`] from the underlying typed loader.
    fn load_erased(
        &self,
        path: &AssetPath,
        bytes: &[u8],
        retired: Option<&mut Vec<Box<dyn Any + Send>>>,
    ) -> Result<FullLoadOutput, LoadError>;
}

impl<L: AssetLoader> ErasedAssetLoader for L {
    fn produced_type(&self) -> AssetTypeId {
        AssetTypeId::of::<L::Asset>()
    }

    fn read_plan(&self, path: &AssetPath, metadata: AssetMeta) -> Result<AssetReadPlan, LoadError> {
        <L as AssetLoader>::read_plan(self, path, metadata)
    }

    fn load_erased(
        &self,
        path: &AssetPath,
        bytes: &[u8],
        retired: Option<&mut Vec<Box<dyn Any + Send>>>,
    ) -> Result<FullLoadOutput, LoadError> {
        let mut ctx = LoadContext::new(path);
        let value = if let Some(retired) = retired {
            // The context outlives the unwind boundary. Framework-owned labeled
            // values must survive decoder errors and panics until the caller
            // releases its publication lock. Plugin-owned locals remain the
            // plugin's responsibility.
            let decoded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.load(bytes, &mut ctx)
            }));
            match decoded {
                Ok(Ok(value)) => value,
                failure => {
                    let (_, labeled) = ctx.into_parts();
                    retired.extend(labeled.into_iter().map(|asset| asset.value));
                    match failure {
                        Ok(Err(error)) => return Err(error),
                        Err(panic) => std::panic::resume_unwind(panic),
                        Ok(Ok(_)) => unreachable!("successful decode handled above"),
                    }
                }
            }
        } else {
            self.load(bytes, &mut ctx)?
        };
        let type_id = AssetTypeId::of::<L::Asset>();
        // Drain the context's accumulated declarations. The primary's declared
        // dependencies ride on the primary erased asset; labeled sub-assets are
        // returned alongside so the server can intern every one of them under
        // `path#label`. Nothing is discarded.
        let (dependencies, labeled) = ctx.into_parts();
        let primary = ErasedLoadedAsset {
            type_id,
            value: Box::new(value),
            dependencies,
            label: String::new(),
            make_store: || -> Box<dyn ErasedAssetStore> { Box::new(Assets::<L::Asset>::new()) },
        };
        Ok(FullLoadOutput { primary, labeled })
    }
}

/// The combined selection + execution table: the policy
/// [`LoaderRegistry`](crate::LoaderRegistry) plus the parallel vector of erased
/// loader objects keyed by [`LoaderId`](crate::LoaderId). This is what the
/// [`AssetServer`](crate::AssetServer) holds to go from an
/// [`AssetPath`] to decoded bytes.
#[derive(Default, Clone)]
pub struct AssetLoaders {
    policy: LoaderRegistry,
    loaders: Vec<Arc<dyn ErasedAssetLoader>>,
}

impl AssetLoaders {
    /// Creates an empty table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a typed loader, wiring both its suffix policy and its erased
    /// object. Returns the assigned [`LoaderId`] and any suffix conflicts
    /// (advisory, as with [`LoaderRegistry::register`]).
    pub fn register<L: AssetLoader>(&mut self, loader: L) -> (LoaderId, Vec<SuffixConflict>) {
        let produced = AssetTypeId::of::<L::Asset>();
        let exts = loader.extensions();
        let priority = loader.priority();
        let (id, conflicts) = self
            .policy
            .register(produced, exts.iter().copied(), priority);
        debug_assert_eq!(id.index() as usize, self.loaders.len());
        self.loaders.push(Arc::new(loader));
        (id, conflicts)
    }

    // Typed registration cannot produce a dishonest tag or store factory.
    // Server regressions replace only the erased implementation, retaining
    // normal selection, reading, reservation, and publication paths.
    #[cfg(all(test, panic = "unwind"))]
    pub(crate) fn replace_erased_for_test(
        &mut self,
        id: LoaderId,
        loader: Arc<dyn ErasedAssetLoader>,
    ) {
        self.loaders[id.index() as usize] = loader;
    }

    /// Registers an alias suffix (`jpeg → jpg`) on the policy.
    pub fn add_alias(&mut self, alias: &str, canonical: &str) {
        self.policy.add_alias(alias, canonical);
    }

    /// The selection policy, for untyped/typed resolution queries.
    #[must_use]
    pub fn policy(&self) -> &LoaderRegistry {
        &self.policy
    }

    /// Resolves and runs the loader for `path` (untyped selection), returning
    /// the full decoded output including labeled sub-assets.
    ///
    /// # Errors
    /// [`LoadError::Unsupported`] if no loader claims the path, otherwise the
    /// decoding error from the selected loader.
    pub fn load_untyped(
        &self,
        path: &AssetPath,
        bytes: &[u8],
    ) -> Result<FullLoadOutput, LoadError> {
        let id = self
            .policy
            .resolve_untyped(path)
            .ok_or_else(|| LoadError::Unsupported(alloc::format!("no loader for {path}")))?;
        self.run(id, path, bytes, None)
    }

    /// Resolves and runs the loader for `path` restricted to loaders producing
    /// `requested`.
    ///
    /// # Errors
    /// As [`AssetLoaders::load_untyped`].
    pub fn load_for_type(
        &self,
        path: &AssetPath,
        requested: AssetTypeId,
        bytes: &[u8],
    ) -> Result<FullLoadOutput, LoadError> {
        let id = self
            .policy
            .resolve_for_type(path, requested)
            .ok_or_else(|| {
                LoadError::Unsupported(alloc::format!(
                    "no loader for {path} producing {requested:?}"
                ))
            })?;
        self.run(id, path, bytes, None)
    }

    /// Synchronous server decoding, retaining context outputs on failure until
    /// the server's outer guard has released its mutex.
    pub(crate) fn load_for_type_deferred(
        &self,
        path: &AssetPath,
        requested: AssetTypeId,
        bytes: &[u8],
        retired: &mut Vec<Box<dyn Any + Send>>,
    ) -> Result<FullLoadOutput, LoadError> {
        let id = self
            .policy
            .resolve_for_type(path, requested)
            .ok_or_else(|| {
                LoadError::Unsupported(alloc::format!(
                    "no loader for {path} producing {requested:?}"
                ))
            })?;
        self.run(id, path, bytes, Some(retired))
    }

    /// Reads the validated loader input for a typed request through
    /// [`AssetSources::read_range`]. Metadata is obtained first, so a malformed
    /// range declaration is rejected before any payload I/O begins.
    pub(crate) fn read_for_type(
        &self,
        sources: &AssetSources,
        path: &AssetPath,
        requested: AssetTypeId,
    ) -> Result<Vec<u8>, String> {
        let (source, plan, whole) = self.prepare_read(sources, path, requested)?;
        let assembled = assemble_ranges_resolved(&source, &plan);
        Self::finish_read(sources, path, whole, assembled)
    }

    /// Called on the scheduler's I/O pool. Preparation and legacy whole-read
    /// fallback remain on I/O workers; only range joins yield asynchronously.
    /// Cancellation is checked at each synchronous boundary; an already-running
    /// preparation or fallback read finishes, but its result is discarded.
    pub(crate) async fn read_for_type_async(
        &self,
        sources: &AssetSources,
        path: &AssetPath,
        requested: AssetTypeId,
        pool: &prism_tasks::TaskPool,
        priority: Arc<prism_tasks::PriorityCell>,
        cancellation: &RangeAssemblyCancellation,
    ) -> Result<Vec<u8>, String> {
        if cancellation.is_cancelled() {
            return Err(RangeAssemblyError::Cancelled.to_string());
        }
        let prepared = self.prepare_read(sources, path, requested);
        if cancellation.is_cancelled() {
            return Err(RangeAssemblyError::Cancelled.to_string());
        }
        let (source, plan, whole) = prepared?;
        let assembled =
            assemble_ranges_resolved_async(&source, &plan, pool, priority, cancellation).await;
        // Both NotFound and empty whole-file results may trigger blocking VFS
        // fallback in finish_read. Do not enter it after cancellation wins.
        if cancellation.is_cancelled() {
            return Err(RangeAssemblyError::Cancelled.to_string());
        }
        let finished = Self::finish_read(sources, path, whole, assembled);
        if cancellation.is_cancelled() {
            return Err(RangeAssemblyError::Cancelled.to_string());
        }
        finished
    }

    fn prepare_read(
        &self,
        sources: &AssetSources,
        path: &AssetPath,
        requested: AssetTypeId,
    ) -> Result<(ResolvedAssetRead, RangeAssemblyPlan, bool), String> {
        let source = sources
            .resolve_read(path)
            .map_err(|error| error.to_string())?;
        let metadata = source.metadata;
        if metadata.is_dir {
            return Err(alloc::format!("asset path is a directory: {path}"));
        }
        let id = self
            .policy
            .resolve_for_type(path, requested)
            .ok_or_else(|| alloc::format!("no loader for {path} producing {requested:?}"))?;
        let loader = self
            .loaders
            .get(id.index() as usize)
            .ok_or_else(|| "loader id out of range".to_string())?;
        let declared_plan = loader
            .read_plan(path, metadata)
            .map_err(|error| error.to_string())?;
        let whole = matches!(declared_plan, AssetReadPlan::Whole);
        let plan = declared_plan
            .validate(path, metadata.size)
            .map_err(|error| error.to_string())?;
        Ok((source, plan, whole))
    }

    fn finish_read(
        sources: &AssetSources,
        path: &AssetPath,
        whole: bool,
        assembled: Result<Vec<u8>, RangeAssemblyError>,
    ) -> Result<Vec<u8>, String> {
        // A source is allowed to report metadata optimistically and later say
        // the path is absent (for example an overlay index that has gone
        // stale). Preserve the VFS's established NotFound-only fallback rule
        // for the default whole-file plan. Explicit range plans intentionally
        // stay pinned to their declared physical layout and never reinterpret
        // a missing part as another source's payload.
        if matches!(
            (whole, &assembled),
            (
                true,
                Err(RangeAssemblyError::Read {
                    error: crate::ReadError::NotFound,
                    ..
                })
            )
        ) {
            return sources.read(path).map_err(|error| error.to_string());
        }
        // A zero-byte range cannot distinguish a genuine empty source from an
        // optimistic overlay whose metadata says zero but whose actual read is
        // absent. Confirm that ambiguous case with the legacy full-read
        // dispatch so a stale overlay cannot turn a lower-priority asset into
        // an empty loader input. Non-empty whole reads, and all explicit range
        // plans, remain range-backed.
        if matches!(
            (whole, &assembled),
            (true, Ok(bytes)) if bytes.is_empty()
        ) {
            return sources.read(path).map_err(|error| error.to_string());
        }
        assembled.map_err(|error| match error {
            RangeAssemblyError::SourcePanicked { .. } => {
                "asset source or loader panicked".to_string()
            }
            error => error.to_string(),
        })
    }

    fn run(
        &self,
        id: LoaderId,
        path: &AssetPath,
        bytes: &[u8],
        mut retired: Option<&mut Vec<Box<dyn Any + Send>>>,
    ) -> Result<FullLoadOutput, LoadError> {
        let loader = self
            .loaders
            .get(id.index() as usize)
            .ok_or_else(|| LoadError::Other("loader id out of range".to_string()))?;
        let output = loader.load_erased(path, bytes, retired.as_deref_mut())?;
        // Invariant: a loader's declared product type must match the type it
        // actually boxed. A mismatch means the typed/erased layers disagree and
        // the server would mis-route the value into the wrong arena.
        #[cfg(debug_assertions)]
        {
            // The output outlives the assertion's unwind scope. In particular,
            // an erased implementation's produced_type callback can panic as
            // well as return a dishonest tag. The synchronous caller still
            // owns its mutex until its outer guard drains retired values.
            let checked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                debug_assert_eq!(
                    output.primary.type_id,
                    loader.produced_type(),
                    "loader produced a value of a different type than it declared"
                );
            }));
            if let Err(payload) = checked {
                if let Some(retired) = retired {
                    retired.push(Box::new(output));
                }
                std::panic::resume_unwind(payload);
            }
        }
        Ok(output)
    }

    /// The number of registered loaders.
    #[must_use]
    pub fn len(&self) -> usize {
        self.loaders.len()
    }

    /// Whether no loaders are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.loaders.is_empty()
    }
}

/// The complete result of a load: the primary erased asset plus its labeled
/// sub-assets, each ready to be interned by the server.
pub struct FullLoadOutput {
    /// The primary decoded (sub-)asset (empty label).
    pub primary: ErasedLoadedAsset,
    /// Labeled sub-assets produced alongside the primary.
    pub labeled: Vec<ErasedLoadedAsset>,
}

impl fmt::Debug for FullLoadOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FullLoadOutput")
            .field("primary", &self.primary)
            .field("labeled", &self.labeled)
            .finish()
    }
}
