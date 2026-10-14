//! The render world's pipeline cache.
//!
//! Bevy's render world carries a `PipelineCache` (and `SpecializedMeshPipelines`
//! on top of it) so a mesh/variant that was compiled once is reused. Prism's
//! equivalent is a thin renderer-side cache over the RHI's
//! [`ContentCache`]: the key is the pipeline's *content* (the material
//! permutation class plus the fixed-function state that changes the PSO), and
//! the value is the RHI pipeline id. Identical keys return the same id, so a PSO
//! is built once and shared; the cache is reference-counted so a variant that
//! stops being used can be evicted (and its GPU object destroyed) without
//! disturbing a variant that is still live.
//!
//! Fixed-function identity is the complete descriptor-derived [`PipelineState`],
//! not a hash or an implicit part of the material class. Every descriptor-taking
//! lookup validates that state before touching the cache, including cache hits.
//! The caller still owns the shader/program and pipeline-layout identity.

#[path = "pipeline_state.rs"]
pub mod pipeline_state;
pub use pipeline_state::PipelineState;

use prism_ecs::prelude::Resource;
use prism_render_architecture::shader_package::{
    AdmitOutcome, DrawDecision, LruPsoCache, PipelineReadiness, PsoCacheKey, ShaderPackageId,
    WarmCounters, WarmRuntime,
};
pub use prism_render_architecture::shader_package::{ExecutionFamily, ModuleId, PermutationIndex};
use prism_render_driver::{
    CacheStats, ContentCache, RenderDevice, RenderPipelineDescriptor, RenderPipelineId,
    VertexBufferLayout,
};
use std::collections::{BTreeSet, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;

/// The content key of one render pipeline.
///
/// A semantic four-part query, never a container for runtime writeback.
/// No string, packed bin class, or hash-only state participates in lookup.
pub type PipelineKey = prism_render_architecture::shader_package::PipelineIdentity<PipelineState>;

/// Converts the typed app identity into the architecture's cold-cache key.
///
/// The conversion is intentionally one-way at the data level: the cold key
/// carries a package handle and a state hash, while the app key retains the
/// complete typed state. [`PipelineCache`] records the observed typed binding
/// so a reverse lookup can be offered without pretending that a 64-bit hash is
/// reversible.
#[must_use]
pub fn cold_key_from_pipeline_key(key: &PipelineKey, package: ShaderPackageId) -> PsoCacheKey {
    PsoCacheKey::new(
        package,
        key.permutation_index.get(),
        key.typed_pipeline_state.to_hash(),
    )
}

/// Why an architecture cold key cannot be mapped to an app pipeline key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PipelineKeyMappingError {
    /// The module was used before its package identity was registered.
    UnregisteredModule(ModuleId),
    /// A module identity was registered with two different packages.
    ConflictingModulePackage {
        /// The module whose registration drifted.
        module_id: ModuleId,
        /// The package already bound to the module.
        existing: ShaderPackageId,
        /// The conflicting package requested by the caller.
        requested: ShaderPackageId,
    },
    /// A package was already assigned to another driver-local module.
    ///
    /// `PsoCacheKey` intentionally contains the package, permutation, and
    /// typed-state encoding, but not `ModuleId`. Sharing one package between
    /// modules would therefore make reverse lookup ambiguous.
    PackageMappedToMultipleModules {
        /// The package already bound to another module.
        package: ShaderPackageId,
        /// The module that owns the package already.
        existing: ModuleId,
        /// The second module attempting to reuse the package.
        requested: ModuleId,
    },
    /// No typed key has been observed for this cold key in this cache.
    UnknownColdKey,
    /// The optional byte-budgeted residency model was not enabled.
    LruDisabled,
    /// Two distinct typed keys mapped to one cold key; do not guess.
    AmbiguousColdKey,
}

/// The requested-specialization lookup result, separate from the pipeline that
/// may actually be bound for this draw. A cold request can therefore report
/// [`Self::Miss`] while [`PipelineResolution::Fallback`] reports the safe bound
/// PSO.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PipelineLookupStatus {
    /// The requested specialization was already resident.
    Ready,
    /// The requested specialization was not ready and a fallback was bound.
    Fallback,
    /// The requested specialization was absent from the app cache.
    Miss,
}

/// RHI pipeline-binary persistence is not available at this app/RHI seam.
///
/// The architecture crate only supplies a validated wire model. Until the RHI
/// exposes driver-owned import/export, these operations must fail explicitly;
/// no `RenderPipelineId`, shader source, or descriptor hash is a binary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PipelineBinaryCacheError {
    /// The selected RHI has no driver-binary import/export API.
    Unsupported,
}

impl From<&RenderPipelineDescriptor> for PipelineState {
    fn from(descriptor: &RenderPipelineDescriptor) -> Self {
        Self::from_descriptor(descriptor)
    }
}

trait ValidatePipelineKey {
    fn assert_matches(&self, descriptor: &RenderPipelineDescriptor);
}

impl ValidatePipelineKey for PipelineKey {
    fn assert_matches(&self, descriptor: &RenderPipelineDescriptor) {
        assert_eq!(
            self.typed_pipeline_state,
            PipelineState::from_descriptor(descriptor),
            "pipeline key fixed-function state differs from descriptor"
        );
    }
}

/// A deterministic 64-bit hash of a vertex layout set (FNV-1a).
///
/// Encodes the buffer count, then each buffer's stride, step mode, and every
/// attribute's format, offset, and shader location, so two layout sets hash
/// differently for distinct tested layouts. This is a compact diagnostic
/// encoding; [`PipelineKey`] compares full layouts rather than this lossy hash.
#[must_use]
pub fn hash_vertex_layouts(layouts: &[VertexBufferLayout]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    fn write(hash: &mut u64, bytes: u64) {
        for byte in bytes.to_le_bytes() {
            *hash ^= u64::from(byte);
            *hash = hash.wrapping_mul(FNV_PRIME);
        }
    }

    let mut hash = FNV_OFFSET;
    write(&mut hash, layouts.len() as u64);
    for layout in layouts {
        write(&mut hash, layout.array_stride);
        write(
            &mut hash,
            match layout.step_mode {
                prism_render_driver::VertexStepMode::Vertex => 0,
                prism_render_driver::VertexStepMode::Instance => 1,
            },
        );
        write(&mut hash, layout.attributes.len() as u64);
        for attribute in &layout.attributes {
            write(&mut hash, attribute.format as u64);
            write(&mut hash, attribute.offset);
            write(&mut hash, u64::from(attribute.shader_location));
        }
    }
    hash
}

/// Compiles a render pipeline from a descriptor.
///
/// The RHI's `RenderDevice::create_render_pipeline` is synchronous; this trait
/// is the seam that lets the cache move that call onto a worker thread without
/// making the whole RHI trait `Send + Sync`. The blanket implementation below
/// covers every backend whose device is shareable (`WgpuDevice` is).
pub trait PipelineCompiler: Send + Sync {
    /// Compiles `descriptor` and returns its pipeline id.
    fn compile(&self, key: PipelineKey, descriptor: &RenderPipelineDescriptor) -> RenderPipelineId;
}

impl<D> PipelineCompiler for D
where
    D: RenderDevice + Send + Sync + 'static,
{
    fn compile(
        &self,
        _key: PipelineKey,
        descriptor: &RenderPipelineDescriptor,
    ) -> RenderPipelineId {
        self.create_render_pipeline(descriptor)
    }
}

/// Where a pipeline lookup resolved.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PipelineResolution {
    /// The requested specialized pipeline is resident.
    Ready(RenderPipelineId),
    /// The requested pipeline is compiling in the background; this is the
    /// full-lobe über fallback to draw with until it becomes ready.
    Fallback(RenderPipelineId),
}

impl PipelineResolution {
    /// The pipeline id to bind now, regardless of which state produced it.
    #[must_use]
    pub const fn id(self) -> RenderPipelineId {
        match self {
            Self::Ready(id) | Self::Fallback(id) => id,
        }
    }

    /// Whether this resolution is the requested specialization.
    #[must_use]
    pub const fn is_ready(self) -> bool {
        matches!(self, Self::Ready(_))
    }
}

/// One background compile request.
struct PipelineCompileJob {
    key: PipelineKey,
    descriptor: RenderPipelineDescriptor,
}

/// Hard scheduling bounds for a background pipeline compiler.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PipelineCompileConfig {
    /// Maximum concurrent compiler calls. Must be nonzero.
    pub worker_count: usize,
    /// Maximum additional descriptors waiting for a worker slot. May be zero.
    /// Excess requests keep drawing the fallback and retry on a later lookup.
    pub queue_capacity: usize,
}

impl Default for PipelineCompileConfig {
    fn default() -> Self {
        Self {
            worker_count: 4,
            queue_capacity: 64,
        }
    }
}

type PipelineCompileResult = (PipelineKey, Result<RenderPipelineId, ()>);

/// The worker queue behind an asynchronous [`PipelineCache`].
struct PipelineCompileQueue {
    config: PipelineCompileConfig,
    jobs: Option<mpsc::SyncSender<PipelineCompileJob>>,
    results: Option<Mutex<mpsc::Receiver<PipelineCompileResult>>>,
    compiling: Arc<AtomicUsize>,
    workers: Vec<JoinHandle<()>>,
}

impl PipelineCompileQueue {
    fn new(compiler: Arc<dyn PipelineCompiler>, config: PipelineCompileConfig) -> Self {
        assert!(
            config.worker_count > 0,
            "pipeline worker count must be nonzero"
        );
        config
            .worker_count
            .checked_add(config.queue_capacity)
            .expect("pipeline compile capacity overflow");
        // Only worker_count jobs may be dispatched without result collection.
        // Both channels are bounded; the cache owns the separate bounded FIFO.
        let (jobs, job_rx) = mpsc::sync_channel::<PipelineCompileJob>(config.worker_count);
        let job_rx = Arc::new(Mutex::new(job_rx));
        let (results_tx, results_rx) = mpsc::sync_channel(config.worker_count);
        let compiling = Arc::new(AtomicUsize::new(0));
        let mut workers = Vec::with_capacity(config.worker_count);
        for index in 0..config.worker_count {
            let compiler = compiler.clone();
            let job_rx = job_rx.clone();
            let results_tx = results_tx.clone();
            let compiling = compiling.clone();
            workers.push(
                std::thread::Builder::new()
                    .name(format!("prism-pipeline-compile-{index}"))
                    .spawn(move || {
                        loop {
                            // std's Receiver is not Sync. Serialize receive only,
                            // never the actual compiler call.
                            let job = job_rx
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner)
                                .recv();
                            let Ok(job) = job else {
                                break;
                            };
                            compiling.fetch_add(1, Ordering::SeqCst);
                            // RHI compilation has no Result API. An unwinding
                            // backend panic fails this key without killing a slot.
                            let result =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    compiler.compile(job.key.clone(), &job.descriptor)
                                }))
                                .map_err(|_| ());
                            compiling.fetch_sub(1, Ordering::SeqCst);
                            if results_tx.send((job.key, result)).is_err() {
                                break;
                            }
                        }
                    })
                    .expect("spawn the pipeline compile worker"),
            );
        }
        Self {
            config,
            jobs: Some(jobs),
            results: Some(Mutex::new(results_rx)),
            compiling,
            workers,
        }
    }

    fn request(
        &self,
        job: PipelineCompileJob,
    ) -> Result<(), mpsc::TrySendError<PipelineCompileJob>> {
        match &self.jobs {
            Some(jobs) => jobs.try_send(job),
            None => Err(mpsc::TrySendError::Disconnected(job)),
        }
    }

    fn poll(&self) -> Vec<PipelineCompileResult> {
        let results = self
            .results
            .as_ref()
            .expect("live result receiver")
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut ready = Vec::new();
        while let Ok(item) = results.try_recv() {
            ready.push(item);
        }
        ready
    }

    fn shutdown(mut self) -> Vec<PipelineCompileResult> {
        self.jobs.take();
        // Drain while workers finish, BEFORE joining: a full bounded result
        // channel must never make shutdown wait for a blocked sender.
        let results = self
            .results
            .take()
            .expect("live result receiver")
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let ready = results.into_iter().collect();
        self.join();
        ready
    }

    fn join(&mut self) {
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

impl Drop for PipelineCompileQueue {
    fn drop(&mut self) {
        self.jobs.take();
        // No device is available in Drop. Disconnect results before joining so
        // blocked sends wake up. Use PipelineCache::shutdown for GPU cleanup.
        self.results.take();
        self.join();
    }
}

/// A content-addressed, reference-counted cache of compiled render pipelines.
#[derive(Resource)]
pub struct PipelineCache {
    cache: ContentCache<PipelineKey, RenderPipelineId>,
    /// Keys whose specialization is being compiled in the background.
    pending: BTreeSet<PipelineKey>,
    /// Cold jobs waiting for an in-flight slot. Bounding this queue keeps a
    /// frame that discovers thousands of new permutations from turning into an
    /// unbounded descriptor backlog.
    waiting: VecDeque<(PipelineKey, RenderPipelineDescriptor)>,
    /// Keys currently owned by the compile worker.
    in_flight: BTreeSet<PipelineKey>,
    /// Pending keys whose callers released all references before the compile
    /// finished. The late pipeline is destroyed on arrival.
    canceled: BTreeSet<PipelineKey>,
    /// Keys whose background compile failed. They keep using the fallback until
    /// a later synchronous request retries them.
    failed: BTreeSet<PipelineKey>,
    compile_queue: Option<PipelineCompileQueue>,
    /// Cold package identity for each driver-local module.
    module_packages: std::collections::BTreeMap<ModuleId, ShaderPackageId>,
    /// Observed typed/cold identity bindings. The reverse table is only
    /// populated after a collision-free typed key has been seen.
    cold_by_typed: std::collections::BTreeMap<PipelineKey, PsoCacheKey>,
    typed_by_cold: std::collections::BTreeMap<PsoCacheKey, PipelineKey>,
    /// Architecture lifecycle model driven by the real queue transitions.
    warm_runtime: WarmRuntime,
    /// Optional byte-budgeted cold residency model. It is opt-in because the
    /// current RHI does not expose driver PSO binary sizes.
    lru: Option<LruPsoCache>,
    /// Last requested-specialization status, distinct from the bound fallback.
    last_lookup: std::collections::BTreeMap<PipelineKey, PipelineLookupStatus>,
    /// The first typed/cold mapping error observed by the current lookup.
    ///
    /// Legacy callers may use the cache without a response-module registry;
    /// those callers still get a usable fallback, but the missing lifecycle
    /// mapping is never silently discarded.
    last_mapping_error: Option<PipelineKeyMappingError>,
}

impl Default for PipelineCache {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineCache {
    /// An empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: ContentCache::new(),
            pending: BTreeSet::new(),
            waiting: VecDeque::new(),
            in_flight: BTreeSet::new(),
            canceled: BTreeSet::new(),
            failed: BTreeSet::new(),
            compile_queue: None,
            module_packages: std::collections::BTreeMap::new(),
            cold_by_typed: std::collections::BTreeMap::new(),
            typed_by_cold: std::collections::BTreeMap::new(),
            warm_runtime: WarmRuntime::new(),
            lru: None,
            last_lookup: std::collections::BTreeMap::new(),
            last_mapping_error: None,
        }
    }

    /// An empty cache with a background compiler.
    ///
    /// Uses the default scheduling bounds. Prefer [`Self::shutdown`] before
    /// teardown to reclaim late GPU results; Drop still safely joins workers.
    #[must_use]
    pub fn with_worker<D>(device: Arc<D>) -> Self
    where
        D: RenderDevice + Send + Sync + 'static,
    {
        Self::with_compiler(device)
    }

    /// An empty cache with explicit worker and waiting-queue bounds.
    #[must_use]
    pub fn with_worker_config<D>(device: Arc<D>, config: PipelineCompileConfig) -> Self
    where
        D: RenderDevice + Send + Sync + 'static,
    {
        Self::with_compiler_config(device, config)
    }

    /// An empty cache with an injected compiler.
    ///
    /// This is the test seam for deterministic scheduling and the extension
    /// point for a backend that wants to compile through a dedicated queue
    /// rather than `RenderDevice::create_render_pipeline`.
    #[must_use]
    pub fn with_compiler(compiler: Arc<dyn PipelineCompiler>) -> Self {
        Self::with_compiler_config(compiler, PipelineCompileConfig::default())
    }

    /// An injected compiler with explicit scheduling bounds.
    ///
    /// Panics if worker_count is zero or the combined capacity overflows.
    #[must_use]
    pub fn with_compiler_config(
        compiler: Arc<dyn PipelineCompiler>,
        config: PipelineCompileConfig,
    ) -> Self {
        Self {
            cache: ContentCache::new(),
            pending: BTreeSet::new(),
            waiting: VecDeque::new(),
            in_flight: BTreeSet::new(),
            canceled: BTreeSet::new(),
            failed: BTreeSet::new(),
            compile_queue: Some(PipelineCompileQueue::new(compiler, config)),
            module_packages: std::collections::BTreeMap::new(),
            cold_by_typed: std::collections::BTreeMap::new(),
            typed_by_cold: std::collections::BTreeMap::new(),
            warm_runtime: WarmRuntime::new(),
            lru: None,
            last_lookup: std::collections::BTreeMap::new(),
            last_mapping_error: None,
        }
    }

    /// Creates a cache with the architecture's byte-budgeted cold residency
    /// model enabled. The budget tracks caller-provided residency telemetry;
    /// it never guesses a driver binary size from an app descriptor.
    #[must_use]
    pub fn with_lru_budget(budget_bytes: u64) -> Self {
        let mut cache = Self::new();
        cache.lru = Some(LruPsoCache::with_budget(budget_bytes));
        cache
    }

    /// Registers the cold package identity for a driver-local module.
    ///
    /// `ModuleId` is deliberately not converted into a package name: it is a
    /// process-local handle and is not a portable cold identity. Registration
    /// is idempotent for the same package and rejects drift loudly.
    pub fn register_module_package(
        &mut self,
        module_id: ModuleId,
        package: ShaderPackageId,
    ) -> Result<(), PipelineKeyMappingError> {
        if let Some(existing) = self.module_packages.get(&module_id).copied() {
            if existing != package {
                return Err(PipelineKeyMappingError::ConflictingModulePackage {
                    module_id,
                    existing,
                    requested: package,
                });
            }
            return Ok(());
        }
        if let Some(existing) =
            self.module_packages
                .iter()
                .find_map(|(registered_module, registered_package)| {
                    (*registered_package == package).then_some(*registered_module)
                })
        {
            return Err(PipelineKeyMappingError::PackageMappedToMultipleModules {
                package,
                existing,
                requested: module_id,
            });
        }
        self.module_packages.insert(module_id, package);
        Ok(())
    }

    /// Returns the cold key for a typed key without mutating the mapping table.
    pub fn cold_key_for(&self, key: &PipelineKey) -> Result<PsoCacheKey, PipelineKeyMappingError> {
        let package = self
            .module_packages
            .get(&key.response_module_id)
            .copied()
            .ok_or(PipelineKeyMappingError::UnregisteredModule(
                key.response_module_id,
            ))?;
        Ok(cold_key_from_pipeline_key(key, package))
    }

    /// Records and returns a typed/cold identity binding.
    pub fn remember_cold_key(
        &mut self,
        key: &PipelineKey,
    ) -> Result<PsoCacheKey, PipelineKeyMappingError> {
        let cold = self.cold_key_for(key)?;
        if let Some(existing) = self.typed_by_cold.get(&cold) {
            if existing != key {
                return Err(PipelineKeyMappingError::AmbiguousColdKey);
            }
        }
        self.cold_by_typed.insert(key.clone(), cold.clone());
        self.typed_by_cold.insert(cold.clone(), key.clone());
        Ok(cold)
    }

    /// Resolves a previously observed cold key back to its complete typed key.
    ///
    /// A cold key seen outside this cache is not reverse-decoded from its hash;
    /// callers must re-submit the typed key so collisions remain impossible.
    pub fn typed_key_for(
        &self,
        cold: &PsoCacheKey,
    ) -> Result<PipelineKey, PipelineKeyMappingError> {
        self.typed_by_cold
            .get(cold)
            .cloned()
            .ok_or(PipelineKeyMappingError::UnknownColdKey)
    }

    /// The last requested-specialization status observed for `key`.
    #[must_use]
    pub fn lookup_status(&self, key: &PipelineKey) -> Option<PipelineLookupStatus> {
        self.last_lookup.get(key).copied()
    }

    /// The first mapping error observed by the current
    /// [`Self::resolve_or_request`] call.
    #[must_use]
    pub fn last_mapping_error(&self) -> Option<PipelineKeyMappingError> {
        self.last_mapping_error
    }

    /// Current architecture readiness for a mapped typed key.
    pub fn readiness(
        &self,
        key: &PipelineKey,
    ) -> Result<PipelineReadiness, PipelineKeyMappingError> {
        let cold = self
            .cold_by_typed
            .get(key)
            .ok_or(PipelineKeyMappingError::UnknownColdKey)?;
        Ok(self.warm_runtime.readiness(cold))
    }

    /// Warm-runtime counters driven by this real pipeline cache.
    #[must_use]
    pub fn warm_counters(&self) -> WarmCounters {
        self.warm_runtime.counters()
    }

    /// Gates a draw using the architecture runtime state without blocking.
    pub fn request_draw(
        &mut self,
        key: &PipelineKey,
        allow_placeholder: bool,
    ) -> Result<DrawDecision, PipelineKeyMappingError> {
        let cold = self.remember_cold_key(key)?;
        Ok(self.warm_runtime.request_draw(&cold, allow_placeholder))
    }

    /// Records a real backend residency-size observation for a compiled PSO.
    ///
    /// The current RHI has no query for this value, so this API accepts only a
    /// caller-supplied measurement. The returned evictions are cold keys from
    /// the architecture model; no native object is destroyed here because
    /// `ContentCache` still owns reference counts for the app's live users.
    pub fn record_resident_size(
        &mut self,
        key: &PipelineKey,
        size_bytes: u64,
    ) -> Result<AdmitOutcome, PipelineKeyMappingError> {
        let cold = self.remember_cold_key(key)?;
        let lru = self
            .lru
            .as_mut()
            .ok_or(PipelineKeyMappingError::LruDisabled)?;
        Ok(lru.admit(cold, size_bytes))
    }

    /// Cold residency model, when enabled with [`Self::with_lru_budget`].
    #[must_use]
    pub fn lru(&self) -> Option<&LruPsoCache> {
        self.lru.as_ref()
    }

    /// Driver-binary import is intentionally unavailable at this RHI seam.
    pub fn import_driver_binary(
        &mut self,
        _key: &PsoCacheKey,
        _binary: &[u8],
    ) -> Result<(), PipelineBinaryCacheError> {
        Err(PipelineBinaryCacheError::Unsupported)
    }

    /// Driver-binary export is intentionally unavailable at this RHI seam.
    pub fn export_driver_binary(
        &self,
        _key: &PsoCacheKey,
    ) -> Result<Vec<u8>, PipelineBinaryCacheError> {
        Err(PipelineBinaryCacheError::Unsupported)
    }

    /// The number of distinct cached pipelines.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache holds no pipelines.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// The hit/miss counters and live entry count.
    #[must_use]
    pub fn stats(&mut self) -> CacheStats {
        self.cache.stats()
    }

    /// The number of specializations currently compiling in the background.
    #[must_use]
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// The number of cold jobs waiting for a bounded in-flight slot.
    #[must_use]
    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    /// Number of compiler calls executing right now (not completed results).
    #[must_use]
    pub fn compiling_len(&self) -> usize {
        self.compile_queue
            .as_ref()
            .map_or(0, |queue| queue.compiling.load(Ordering::SeqCst))
    }

    /// Dispatched requests, including executing calls and unpumped results.
    #[must_use]
    pub fn in_flight_len(&self) -> usize {
        self.in_flight.len()
    }

    /// Scheduling bounds when a background compiler is attached.
    #[must_use]
    pub fn compile_config(&self) -> Option<PipelineCompileConfig> {
        self.compile_queue.as_ref().map(|queue| queue.config)
    }

    /// The number of specializations whose last background compile failed.
    #[must_use]
    pub fn failed_len(&self) -> usize {
        self.failed.len()
    }

    /// Whether this cache has a background compiler attached.
    #[must_use]
    pub fn has_worker(&self) -> bool {
        self.compile_queue.is_some()
    }

    /// Advances a mapped key to `Ready` after a synchronous or completed cache
    /// insertion. This keeps the architecture model in lockstep even when a
    /// caller bypasses the worker path through `get_or_create`.
    fn mark_ready(&mut self, key: &PipelineKey) {
        let Ok(cold) = self.remember_cold_key(key) else {
            return;
        };
        match self.warm_runtime.readiness(&cold) {
            PipelineReadiness::Absent | PipelineReadiness::Failed => {
                let _ = self.warm_runtime.enqueue(cold.clone());
                let _ = self.warm_runtime.begin_compile(&cold);
                let _ = self.warm_runtime.finish_compile(&cold);
            }
            PipelineReadiness::Queued => {
                let _ = self.warm_runtime.begin_compile(&cold);
                let _ = self.warm_runtime.finish_compile(&cold);
            }
            PipelineReadiness::Compiling => {
                let _ = self.warm_runtime.finish_compile(&cold);
            }
            PipelineReadiness::Ready => {}
        }
        if let Some(lru) = self.lru.as_mut() {
            let _ = lru.touch(&cold);
        }
    }

    fn remember_cold_key_observable(&mut self, key: &PipelineKey) -> Option<PsoCacheKey> {
        match self.remember_cold_key(key) {
            Ok(cold) => Some(cold),
            Err(error) => {
                if self.last_mapping_error.is_none() {
                    self.last_mapping_error = Some(error);
                }
                None
            }
        }
    }

    /// Mirrors a successful worker dispatch into the cold lifecycle model.
    fn mark_compiling(&mut self, key: &PipelineKey) {
        let Some(cold) = self.cold_by_typed.get(key).cloned() else {
            return;
        };
        let _ = self.warm_runtime.begin_compile(&cold);
    }

    /// Mirrors a failed/canceled worker result into the cold lifecycle model.
    fn mark_failed(&mut self, key: &PipelineKey) {
        let Some(cold) = self.cold_by_typed.get(key).cloned() else {
            return;
        };
        match self.warm_runtime.readiness(&cold) {
            PipelineReadiness::Queued => {
                let _ = self.warm_runtime.begin_compile(&cold);
                let _ = self.warm_runtime.fail_compile(&cold);
            }
            PipelineReadiness::Compiling => {
                let _ = self.warm_runtime.fail_compile(&cold);
            }
            PipelineReadiness::Absent => {
                let _ = self.warm_runtime.enqueue(cold.clone());
                let _ = self.warm_runtime.begin_compile(&cold);
                let _ = self.warm_runtime.fail_compile(&cold);
            }
            PipelineReadiness::Failed | PipelineReadiness::Ready => {}
        }
    }

    /// Resolves `key`, requesting a background compile when needed.
    ///
    /// The fallback pipeline is compiled synchronously on first use and pinned
    /// for the cache's lifetime, so a cold specialization always has something
    /// correct to draw with. When no worker is attached this degrades to the
    /// synchronous path. Saturated requests retry on a later lookup.
    pub fn resolve_or_request<D: RenderDevice>(
        &mut self,
        device: &D,
        key: PipelineKey,
        descriptor: RenderPipelineDescriptor,
        fallback_key: PipelineKey,
        fallback_descriptor: RenderPipelineDescriptor,
    ) -> PipelineResolution {
        key.assert_matches(&descriptor);
        fallback_key.assert_matches(&fallback_descriptor);
        self.last_mapping_error = None;
        let key_cold = self.remember_cold_key_observable(&key);
        let _fallback_cold = self.remember_cold_key_observable(&fallback_key);
        if let Some(id) = self.cache.acquire(&key) {
            self.mark_ready(&key);
            self.last_lookup.insert(key, PipelineLookupStatus::Ready);
            return PipelineResolution::Ready(id);
        }

        let fallback = self.cache.get_or_create(fallback_key.clone(), || {
            device.create_render_pipeline(&fallback_descriptor)
        });
        self.mark_ready(&fallback_key);
        if fallback_key == key {
            self.last_lookup.insert(key, PipelineLookupStatus::Ready);
            return PipelineResolution::Ready(fallback);
        }
        let was_known = self.pending.contains(&key)
            || self.in_flight.contains(&key)
            || self.waiting.iter().any(|(queued, _)| queued == &key)
            || self.failed.contains(&key);
        self.last_lookup.insert(
            key.clone(),
            if was_known {
                PipelineLookupStatus::Fallback
            } else {
                PipelineLookupStatus::Miss
            },
        );
        if !self.failed.contains(&key) {
            if let Some(cold) = key_cold.or_else(|| self.cold_by_typed.get(&key).cloned()) {
                let _ = self.warm_runtime.request_draw(&cold, true);
            }
        }
        if self.failed.contains(&key) {
            return PipelineResolution::Fallback(fallback);
        }
        if self.pending.contains(&key) {
            return PipelineResolution::Fallback(fallback);
        }

        if self.in_flight.contains(&key) {
            // A released or cleared job is still in the worker pool: revive it
            // instead of enqueuing a duplicate.
            self.canceled.remove(&key);
            self.pending.insert(key);
            return PipelineResolution::Fallback(fallback);
        }

        if self.compile_queue.is_some() {
            self.dispatch_waiting();
            let config = self.compile_queue.as_ref().unwrap().config;
            if self.in_flight.len() >= config.worker_count
                && self.waiting.len() >= config.queue_capacity
            {
                // Keep the render thread nonblocking and memory strictly
                // bounded. The next lookup may try to admit this key again.
                return PipelineResolution::Fallback(fallback);
            }
            self.pending.insert(key.clone());
            self.waiting.push_back((key, descriptor));
            self.dispatch_waiting();
            return PipelineResolution::Fallback(fallback);
        }

        // No worker (or a closed worker): compile synchronously rather than
        // dropping the request. This is also the `PipelineCache::new()` path.
        let id = device.create_render_pipeline(&descriptor);
        self.cache.get_or_create(key.clone(), || id);
        self.mark_ready(&key);
        self.last_lookup.insert(key, PipelineLookupStatus::Ready);
        PipelineResolution::Ready(id)
    }

    /// Queues a batch of cold specializations without a foreground lookup.
    ///
    /// This is the engine-side seam for archetype/CI warmup: the caller owns
    /// the list of hot `(key, descriptor, fallback)` tuples, and the cache
    /// compiles them through the same bounded worker path as an on-demand
    /// request. The fallback is prepared eagerly so the first visible use is
    /// already correct even if the exact PSO is still compiling.
    pub fn prewarm<D: RenderDevice>(
        &mut self,
        device: &D,
        requests: impl IntoIterator<
            Item = (
                PipelineKey,
                RenderPipelineDescriptor,
                PipelineKey,
                RenderPipelineDescriptor,
            ),
        >,
    ) {
        for (key, descriptor, fallback_key, fallback_descriptor) in requests {
            let _ =
                self.resolve_or_request(device, key, descriptor, fallback_key, fallback_descriptor);
        }
    }

    /// Drains completed background compiles.
    ///
    /// A result whose request was canceled is destroyed immediately; a result
    /// whose request is no longer pending is stale for the same reason. The
    /// caller supplies the device so the cache can own this cleanup.
    pub fn pump<D: RenderDevice>(&mut self, device: &D) {
        let ready = self
            .compile_queue
            .as_ref()
            .map(PipelineCompileQueue::poll)
            .unwrap_or_default();
        for (key, result) in ready {
            self.in_flight.remove(&key);
            match result {
                Ok(id) => {
                    if !self.complete(key, id) {
                        device.destroy_render_pipeline(id);
                    }
                }
                Err(()) => {
                    self.canceled.remove(&key);
                    if self.pending.remove(&key) {
                        self.failed.insert(key.clone());
                        self.mark_failed(&key);
                    }
                }
            }
        }
        self.dispatch_waiting();
    }

    /// Sends waiting jobs while worker slots have room. A disconnected
    /// worker fails the key and keeps serving the fallback rather than dropping
    /// the material.
    fn dispatch_waiting(&mut self) {
        let Some(worker_count) = self
            .compile_queue
            .as_ref()
            .map(|queue| queue.config.worker_count)
        else {
            return;
        };
        while self.in_flight.len() < worker_count {
            let Some((key, descriptor)) = self.waiting.pop_front() else {
                break;
            };
            let result = {
                let Some(queue) = self.compile_queue.as_ref() else {
                    self.waiting.push_front((key, descriptor));
                    break;
                };
                queue.request(PipelineCompileJob {
                    key: key.clone(),
                    descriptor,
                })
            };
            match result {
                Ok(()) => {
                    self.in_flight.insert(key.clone());
                    self.mark_compiling(&key);
                }
                Err(mpsc::TrySendError::Full(job)) => {
                    self.waiting.push_front((job.key, job.descriptor));
                    break;
                }
                Err(mpsc::TrySendError::Disconnected(_)) => {
                    self.pending.remove(&key);
                    self.failed.insert(key.clone());
                    self.mark_failed(&key);
                }
            }
        }
    }

    /// Inserts a completed background compile. Returns `false` when the result
    /// was canceled or had no pending request, in which case the caller must
    /// destroy it.
    fn complete(&mut self, key: PipelineKey, id: RenderPipelineId) -> bool {
        if self.canceled.remove(&key) {
            self.pending.remove(&key);
            self.mark_failed(&key);
            return false;
        }
        if !self.pending.remove(&key) || self.cache.refcount(&key) != 0 {
            self.mark_failed(&key);
            return false;
        }
        self.cache.get_or_create(key.clone(), || id);
        self.mark_ready(&key);
        self.last_lookup.insert(key, PipelineLookupStatus::Ready);
        true
    }

    /// Returns the id for `key`, compiling `descriptor` on first use.
    ///
    /// A later call returns the cached id only after validating that its fixed
    /// state agrees. Shader/program identity is still the caller's contract.
    pub fn get_or_create<D: RenderDevice>(
        &mut self,
        device: &D,
        key: PipelineKey,
        descriptor: &RenderPipelineDescriptor,
    ) -> RenderPipelineId {
        key.assert_matches(descriptor);
        self.pending.remove(&key);
        self.waiting.retain(|(queued, _)| queued != &key);
        if self.in_flight.contains(&key) {
            // A foreground compile overtook a dispatched job: its late result
            // must be destroyed, not inserted over the synchronous pipeline.
            self.canceled.insert(key.clone());
        }
        self.failed.remove(&key);
        let id = self
            .cache
            .get_or_create(key.clone(), || device.create_render_pipeline(descriptor));
        self.mark_ready(&key);
        self.last_lookup.insert(key, PipelineLookupStatus::Ready);
        id
    }

    /// Drops one reference to `key`, destroying the pipeline when it reaches
    /// zero. Returns whether the pipeline was evicted.
    pub fn release<D: RenderDevice>(&mut self, device: &D, key: &PipelineKey) -> bool {
        if self.pending.remove(key) {
            self.mark_failed(key);
            self.waiting.retain(|(queued, _)| queued != key);
            if self.in_flight.contains(key) {
                self.canceled.insert(key.clone());
            }
            return false;
        }
        self.failed.remove(key);
        match self.cache.release(key) {
            Some(id) => {
                device.destroy_render_pipeline(id);
                true
            }
            None => false,
        }
    }

    /// Destroys every cached pipeline regardless of reference count, for device
    /// loss or teardown.
    pub fn clear<D: RenderDevice>(&mut self, device: &D) {
        let pending = self
            .pending
            .iter()
            .chain(self.in_flight.iter())
            .cloned()
            .collect::<Vec<_>>();
        for key in &pending {
            self.mark_failed(key);
        }
        self.pending.clear();
        self.waiting.clear();
        // Keep dispatched slots occupied until pump destroys stale results.
        self.canceled = self.in_flight.clone();
        self.failed.clear();
        self.cache
            .drain_all(|id| device.destroy_render_pipeline(id));
        self.warm_runtime = WarmRuntime::new();
        self.last_lookup.clear();
        self.last_mapping_error = None;
        if let Some(budget) = self.lru.as_ref().map(LruPsoCache::budget_bytes) {
            self.lru = Some(LruPsoCache::with_budget(budget));
        }
    }

    /// Stops workers, drains every result, and releases all cache-owned GPU
    /// objects. An executing compiler must return before shutdown finishes.
    /// Draining concurrently with worker exit prevents bounded-channel
    /// deadlock even when the result queue is full.
    pub fn shutdown<D: RenderDevice>(&mut self, device: &D) {
        if let Some(queue) = self.compile_queue.take() {
            for (_, result) in queue.shutdown() {
                if let Ok(id) = result {
                    device.destroy_render_pipeline(id);
                }
            }
        }
        self.pending.clear();
        self.waiting.clear();
        self.in_flight.clear();
        self.canceled.clear();
        self.failed.clear();
        self.cache
            .drain_all(|id| device.destroy_render_pipeline(id));
        self.warm_runtime = WarmRuntime::new();
        self.last_lookup.clear();
        self.last_mapping_error = None;
        if let Some(budget) = self.lru.as_ref().map(LruPsoCache::budget_bytes) {
            self.lru = Some(LruPsoCache::with_budget(budget));
        }
    }
}

#[cfg(test)]
#[path = "pipeline_worker_tests.rs"]
mod worker_tests;
