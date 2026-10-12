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
//! The key must fully determine the pipeline: the caller owns that contract.
//! `get_or_create` deliberately ignores a later descriptor for an already-cached
//! key, which is what makes dedup stable.

use prism_ecs::prelude::Resource;
use prism_render_driver::{
    CacheStats, ContentCache, RenderDevice, RenderPipelineDescriptor, RenderPipelineId,
    PrimitiveTopology, TextureFormat, VertexBufferLayout,
};
use std::collections::{BTreeSet, VecDeque};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;

/// The content key of one render pipeline.
///
/// `class` is the material permutation / bin class (the material layer's
/// `SpecializationId::bin_class()`); the rest is the fixed-function state that
/// changes the compiled pipeline. The fields are plain integers so the key is
/// `Ord` and the cache is a deterministic `BTreeMap`.
///
/// # Coverage
///
/// Covered: the material permutation class (which folds every material-driven
/// state — blend, depth test/write, culling — that the material layer already
/// keys), the color/depth attachment formats, the sample count, the full vertex
/// layout (a hash of the strides, step modes, and attributes), the primitive
/// topology, and the shader identity.
///
/// Not covered: per-pipeline fixed-function overrides that no material class
/// keys (e.g. an explicit depth-bias). A pipeline that needs those must fold
/// them into `class` until the key grows a field for them — `get_or_create`
/// trusts the key, so an under-specified key would silently alias two
/// pipelines.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct PipelineKey {
    /// The material permutation / bin class.
    pub class: u32,
    /// The single color-target format encoded as `format + 1`, or `0` for a
    /// depth-only pipeline (the RHI format enum is not `Ord`, so the key keeps
    /// an integer).
    color_format: u32,
    /// The depth/stencil format encoded as `format + 1`, or `0` for no depth.
    depth_format: u32,
    /// The MSAA sample count (`1` for no multisampling).
    pub sample_count: u32,
    /// A deterministic hash of the **full** vertex layout set the PSO bakes in.
    ///
    /// A single stride is not enough: two pipelines whose buffers have the same
    /// total stride but different attribute formats, offsets, step modes, or
    /// buffer counts compile to different vertex stages. Use
    /// [`hash_vertex_layouts`] to derive this.
    vertex_layout_hash: u64,
    /// The primitive topology, encoded as the RHI enum's discriminant.
    topology: u32,
    /// A stable identity for the shader module **and its entry points**.
    ///
    /// The RHI's `ShaderModuleId` is backend-assigned, so it is not a portable
    /// identity; the caller supplies a content id (e.g. from `prism_shader`'s
    /// permutation) that is equal exactly when two pipelines compile the same
    /// shader program.
    shader_id: u64,
}

impl PipelineKey {
    /// A key for a single-color-target pipeline.
    #[must_use]
    pub const fn new(
        class: u32,
        color_format: Option<TextureFormat>,
        depth_format: Option<TextureFormat>,
        sample_count: u32,
        vertex_layout_hash: u64,
        topology: PrimitiveTopology,
        shader_id: u64,
    ) -> Self {
        Self {
            class,
            color_format: encode_format(color_format),
            depth_format: encode_format(depth_format),
            sample_count,
            vertex_layout_hash,
            topology: topology as u32,
            shader_id,
        }
    }

}

/// A deterministic 64-bit hash of a vertex layout set (FNV-1a).
///
/// Encodes the buffer count, then each buffer's stride, step mode, and every
/// attribute's format, offset, and shader location, so two layout sets hash
/// equal exactly when they describe the same vertex stage. Used to fold the
/// layout into a [`PipelineKey`].
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


/// Encodes an optional format as `format + 1` (`0` is the "none" sentinel), so
/// the key is an `Ord` integer even though `TextureFormat` is not.
const fn encode_format(format: Option<TextureFormat>) -> u32 {
    match format {
        None => 0,
        Some(format) => format as u32 + 1,
    }
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

/// The worker queue behind an asynchronous [`PipelineCache`].
struct PipelineCompileQueue {
    jobs: Option<mpsc::Sender<PipelineCompileJob>>,
    results: Mutex<mpsc::Receiver<(PipelineKey, RenderPipelineId)>>,
    worker: Option<JoinHandle<()>>,
}

impl PipelineCompileQueue {
    fn new(compiler: Arc<dyn PipelineCompiler>) -> Self {
        let (jobs, job_rx) = mpsc::channel::<PipelineCompileJob>();
        let (results_tx, results_rx) = mpsc::channel();
        let worker = std::thread::Builder::new()
            .name("prism-pipeline-compile".into())
            .spawn(move || {
                while let Ok(job) = job_rx.recv() {
                    let id = compiler.compile(job.key, &job.descriptor);
                    if results_tx.send((job.key, id)).is_err() {
                        break;
                    }
                }
            })
            .expect("spawn the pipeline compile worker");
        Self {
            jobs: Some(jobs),
            results: Mutex::new(results_rx),
            worker: Some(worker),
        }
    }

    fn request(&self, key: PipelineKey, descriptor: RenderPipelineDescriptor) -> bool {
        let Some(jobs) = &self.jobs else {
            return false;
        };
        jobs.send(PipelineCompileJob { key, descriptor }).is_ok()
    }

    fn poll(&self) -> Vec<(PipelineKey, RenderPipelineId)> {
        let results = self
            .results
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut ready = Vec::new();
        while let Ok(item) = results.try_recv() {
            ready.push(item);
        }
        ready
    }
}

impl Drop for PipelineCompileQueue {
    fn drop(&mut self) {
        // Close the job channel first, then wait for the in-flight compile. The
        // worker owns an `Arc` to the device, so it remains valid while it
        // finishes.
        self.jobs.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// A content-addressed, reference-counted cache of compiled render pipelines.
#[derive(Resource, Default)]
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
}

/// Maximum background compiles owned by the worker before new cold requests
/// wait in the cache's bounded queue. This is a scheduling bound, not a
/// permutation cap: every request still compiles eventually.
const MAX_IN_FLIGHT_COMPILES: usize = 4;

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
        }
    }

    /// An empty cache with a background compiler.
    ///
    /// The worker owns `device`; dropping the cache closes the job channel and
    /// joins the worker after its in-flight compile finishes.
    #[must_use]
    pub fn with_worker<D>(device: Arc<D>) -> Self
    where
        D: RenderDevice + Send + Sync + 'static,
    {
        Self::with_compiler(device)
    }

    /// An empty cache with an injected compiler.
    ///
    /// This is the test seam for deterministic scheduling and the extension
    /// point for a backend that wants to compile through a dedicated queue
    /// rather than `RenderDevice::create_render_pipeline`.
    #[must_use]
    pub fn with_compiler(compiler: Arc<dyn PipelineCompiler>) -> Self {
        Self {
            cache: ContentCache::new(),
            pending: BTreeSet::new(),
            waiting: VecDeque::new(),
            in_flight: BTreeSet::new(),
            canceled: BTreeSet::new(),
            failed: BTreeSet::new(),
            compile_queue: Some(PipelineCompileQueue::new(compiler)),
        }
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

    /// Resolves `key`, requesting a background compile when needed.
    ///
    /// The fallback pipeline is compiled synchronously on first use and pinned
    /// for the cache's lifetime, so a cold specialization always has something
    /// correct to draw with. When no worker is attached (or the worker channel
    /// is closed), this degrades to the synchronous path.
    pub fn resolve_or_request<D: RenderDevice>(
        &mut self,
        device: &D,
        key: PipelineKey,
        descriptor: RenderPipelineDescriptor,
        fallback_key: PipelineKey,
        fallback_descriptor: RenderPipelineDescriptor,
    ) -> PipelineResolution {
        if let Some(id) = self.cache.acquire(&key) {
            return PipelineResolution::Ready(id);
        }

        let fallback = self
            .cache
            .get_or_create(fallback_key, || {
                device.create_render_pipeline(&fallback_descriptor)
            });
        if fallback_key == key {
            return PipelineResolution::Ready(fallback);
        }
        if self.failed.contains(&key) {
            return PipelineResolution::Fallback(fallback);
        }
        if self.pending.contains(&key) {
            return PipelineResolution::Fallback(fallback);
        }

        if self.compile_queue.is_some() {
            self.pending.insert(key);
            self.waiting.push_back((key, descriptor));
            self.dispatch_waiting();
            return PipelineResolution::Fallback(fallback);
        }

        // No worker (or a closed worker): compile synchronously rather than
        // dropping the request. This is also the `PipelineCache::new()` path.
        let id = device.create_render_pipeline(&descriptor);
        self.cache.get_or_create(key, || id);
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
            let _ = self.resolve_or_request(
                device,
                key,
                descriptor,
                fallback_key,
                fallback_descriptor,
            );
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
        for (key, id) in ready {
            self.in_flight.remove(&key);
            if !self.complete(key, id) {
                device.destroy_render_pipeline(id);
            }
        }
        self.dispatch_waiting();
    }

    /// Sends waiting jobs while the in-flight bound has room. A disconnected
    /// worker fails the key and keeps serving the fallback rather than dropping
    /// the material.
    fn dispatch_waiting(&mut self) {
        while self.in_flight.len() < MAX_IN_FLIGHT_COMPILES {
            let Some((key, descriptor)) = self.waiting.pop_front() else {
                break;
            };
            let sent = self
                .compile_queue
                .as_ref()
                .is_some_and(|queue| queue.request(key, descriptor));
            if sent {
                self.in_flight.insert(key);
            } else {
                self.pending.remove(&key);
                self.failed.insert(key);
            }
        }
    }

    /// Inserts a completed background compile. Returns `false` when the result
    /// was canceled or had no pending request, in which case the caller must
    /// destroy it.
    fn complete(&mut self, key: PipelineKey, id: RenderPipelineId) -> bool {
        if self.canceled.remove(&key) {
            self.pending.remove(&key);
            return false;
        }
        if !self.pending.remove(&key) || self.cache.refcount(&key) != 0 {
            return false;
        }
        self.cache.get_or_create(key, || id);
        true
    }

    /// Returns the id for `key`, compiling `descriptor` on first use.
    ///
    /// A later call with the same key returns the cached id and ignores the new
    /// descriptor: the key is the pipeline's identity, so it must determine the
    /// descriptor exactly.
    pub fn get_or_create<D: RenderDevice>(
        &mut self,
        device: &D,
        key: PipelineKey,
        descriptor: &RenderPipelineDescriptor,
    ) -> RenderPipelineId {
        self.pending.remove(&key);
        self.canceled.remove(&key);
        self.failed.remove(&key);
        self.cache
            .get_or_create(key, || device.create_render_pipeline(descriptor))
    }

    /// Drops one reference to `key`, destroying the pipeline when it reaches
    /// zero. Returns whether the pipeline was evicted.
    pub fn release<D: RenderDevice>(&mut self, device: &D, key: &PipelineKey) -> bool {
        if self.pending.remove(key) {
            self.waiting.retain(|(queued, _)| queued != key);
            if self.in_flight.contains(key) {
                self.canceled.insert(*key);
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
        self.pending.clear();
        self.waiting.clear();
        self.in_flight.clear();
        self.canceled.clear();
        self.failed.clear();
        self.cache
            .drain_all(|id| device.destroy_render_pipeline(id));
    }
}
