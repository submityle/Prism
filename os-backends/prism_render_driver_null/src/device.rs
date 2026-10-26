//! The null [`RenderDevice`]: RHI resource creation and destruction with no
//! GPU behind it.
//!
//! Every `create_*` mints a real, generation-checked id in a
//! [`rhi::GenerationalSlotMap`] and stores a [`Placeholder`]; every `destroy_*`
//! releases the slot, bumping its generation so the id goes stale. That is the
//! same lifecycle a hardware backend exposes, minus the native object — which
//! is exactly what lets CI and GPU-less hosts exercise stale-id handling and
//! slot recycling against the frozen traits.

use alloc::sync::Arc;
use std::sync::Mutex;

use prism_render_driver as rhi;
use rhi::RenderDevice;

/// Locks `mutex`, treating poisoning (a panic in another thread while the lock
/// was held) as recoverable.
///
/// The frozen RHI traits take `&self`, so the resource tables must sit behind a
/// lock to be mutated and to keep the device `Send + Sync`. A poisoned table is
/// not left unsound — the operations touching it are short and panic-free — so
/// recovering the guard beats propagating the panic into an unrelated caller.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Private marker for buffer slots.
enum BufferMarker {}
/// Private marker for texture slots.
enum TextureMarker {}
/// Private marker for texture-view slots.
enum TextureViewMarker {}
/// Private marker for sampler slots.
enum SamplerMarker {}
/// Private marker for shader-module slots.
enum ShaderMarker {}
/// Private marker for bind-group-layout slots.
enum BindGroupLayoutMarker {}
/// Private marker for bind-group slots.
enum BindGroupMarker {}
/// Private marker for pipeline-layout slots.
enum PipelineLayoutMarker {}
/// Private marker for render-pipeline slots.
enum RenderPipelineMarker {}
/// Private marker for compute-pipeline slots.
enum ComputePipelineMarker {}

/// The record the null device stores for each live resource.
///
/// A slot map must own a value per slot so it can generation-guard stale ids.
/// A hardware backend stores the native handle there; the null backend has no
/// native object to own, so each slot holds this zero-sized placeholder.
struct Placeholder;

/// A short alias for the backend's per-kind slot maps.
type Map<K, V> = Mutex<rhi::GenerationalSlotMap<K, V>>;

/// Reinterprets a generational id from one resource kind to another.
///
/// The RHI's `*Kind` marker types are private to `prism_render_driver`, so the
/// backend keys its slot maps by its own markers. Ids are structurally
/// identical `(index, generation)` pairs, so this rebrands an id across tags.
fn convert_id<A: ?Sized, B: ?Sized>(id: rhi::ResourceId<A>) -> rhi::ResourceId<B> {
    rhi::ResourceId::from_parts(id.index(), id.generation())
}

/// The shared state behind a [`NullDevice`] and the surfaces it backs.
///
/// Holds the reported capabilities and one locked slot map per resource kind.
/// Shared through [`Arc`] (and [`NullDevice`]'s cheap `Clone`) so a
/// [`crate::surface::NullSurface`] mints swapchain images against the very
/// tables the device owns.
pub(crate) struct DeviceInner {
    /// The capabilities reported through [`rhi::RenderDevice::capabilities`].
    caps: rhi::DeviceCapabilities,
    /// Live buffers.
    buffers: Map<BufferMarker, Placeholder>,
    /// Live textures.
    textures: Map<TextureMarker, Placeholder>,
    /// Live texture views.
    texture_views: Map<TextureViewMarker, Placeholder>,
    /// Live samplers.
    samplers: Map<SamplerMarker, Placeholder>,
    /// Live shader modules.
    shaders: Map<ShaderMarker, Placeholder>,
    /// Live bind group layouts.
    bind_group_layouts: Map<BindGroupLayoutMarker, Placeholder>,
    /// Live bind groups.
    bind_groups: Map<BindGroupMarker, Placeholder>,
    /// Live pipeline layouts.
    pipeline_layouts: Map<PipelineLayoutMarker, Placeholder>,
    /// Live render pipelines.
    render_pipelines: Map<RenderPipelineMarker, Placeholder>,
    /// Live compute pipelines.
    compute_pipelines: Map<ComputePipelineMarker, Placeholder>,
}

impl DeviceInner {
    /// Builds the shared state with empty resource tables.
    fn new(caps: rhi::DeviceCapabilities) -> Self {
        Self {
            caps,
            buffers: Mutex::new(rhi::GenerationalSlotMap::new()),
            textures: Mutex::new(rhi::GenerationalSlotMap::new()),
            texture_views: Mutex::new(rhi::GenerationalSlotMap::new()),
            samplers: Mutex::new(rhi::GenerationalSlotMap::new()),
            shaders: Mutex::new(rhi::GenerationalSlotMap::new()),
            bind_group_layouts: Mutex::new(rhi::GenerationalSlotMap::new()),
            bind_groups: Mutex::new(rhi::GenerationalSlotMap::new()),
            pipeline_layouts: Mutex::new(rhi::GenerationalSlotMap::new()),
            render_pipelines: Mutex::new(rhi::GenerationalSlotMap::new()),
            compute_pipelines: Mutex::new(rhi::GenerationalSlotMap::new()),
        }
    }
}

/// A null [`rhi::RenderDevice`]: mints and tracks RHI resource ids without a
/// GPU.
///
/// `Clone` is cheap — clones share the same resource tables through an [`Arc`] —
/// which lets a [`crate::surface::NullSurface`] co-own the device it mints
/// swapchain ids against.
#[derive(Clone)]
pub struct NullDevice {
    /// The shared device state.
    inner: Arc<DeviceInner>,
}

impl NullDevice {
    /// Creates a null device reporting `caps`.
    #[must_use]
    pub(crate) fn new(caps: rhi::DeviceCapabilities) -> Self {
        Self::new_with_capabilities(caps)
    }

    /// Creates an isolated resource-tracking device with explicit capabilities.
    ///
    /// Pair with [`crate::NullQueue::new`] for graph and capability-negotiation
    /// tests. Unlike [`crate::NullInstance::new`], this does not select the
    /// portable baseline: the caller chooses the simulated logical-device
    /// limits. It does not execute shaders or certify real hardware support.
    /// The default instance continues to report the eight-storage-buffer floor.
    #[must_use]
    pub fn new_with_capabilities(caps: rhi::DeviceCapabilities) -> Self {
        Self {
            inner: Arc::new(DeviceInner::new(caps)),
        }
    }

    /// The total number of live resources across every kind.
    ///
    /// A diagnostic and test hook: it lets callers observe that `create_*`
    /// occupies a slot and `destroy_*` releases it, which is the observable
    /// half of the resource lifecycle the null backend implements.
    #[must_use]
    pub fn live_resource_count(&self) -> usize {
        lock(&self.inner.buffers).len()
            + lock(&self.inner.textures).len()
            + lock(&self.inner.texture_views).len()
            + lock(&self.inner.samplers).len()
            + lock(&self.inner.shaders).len()
            + lock(&self.inner.bind_group_layouts).len()
            + lock(&self.inner.bind_groups).len()
            + lock(&self.inner.pipeline_layouts).len()
            + lock(&self.inner.render_pipelines).len()
            + lock(&self.inner.compute_pipelines).len()
    }
}

impl RenderDevice for NullDevice {
    fn capabilities(&self) -> &rhi::DeviceCapabilities {
        &self.inner.caps
    }

    fn create_buffer(&self, _descriptor: &rhi::BufferDescriptor) -> rhi::BufferId {
        convert_id(lock(&self.inner.buffers).insert(Placeholder))
    }

    fn create_texture(&self, _descriptor: &rhi::TextureDescriptor) -> rhi::TextureId {
        convert_id(lock(&self.inner.textures).insert(Placeholder))
    }

    fn create_texture_view(
        &self,
        _texture: rhi::TextureId,
        _descriptor: &rhi::TextureViewDescriptor,
    ) -> rhi::TextureViewId {
        convert_id(lock(&self.inner.texture_views).insert(Placeholder))
    }

    fn create_sampler(&self, _descriptor: &rhi::SamplerDescriptor) -> rhi::SamplerId {
        convert_id(lock(&self.inner.samplers).insert(Placeholder))
    }

    fn create_shader_module(
        &self,
        _descriptor: &rhi::ShaderModuleDescriptor,
    ) -> rhi::ShaderModuleId {
        convert_id(lock(&self.inner.shaders).insert(Placeholder))
    }

    fn create_bind_group_layout(
        &self,
        _descriptor: &rhi::BindGroupLayoutDescriptor,
    ) -> rhi::BindGroupLayoutId {
        convert_id(lock(&self.inner.bind_group_layouts).insert(Placeholder))
    }

    fn create_bind_group(&self, _descriptor: &rhi::BindGroupDescriptor) -> rhi::BindGroupId {
        convert_id(lock(&self.inner.bind_groups).insert(Placeholder))
    }

    fn create_pipeline_layout(
        &self,
        _descriptor: &rhi::PipelineLayoutDescriptor,
    ) -> rhi::PipelineLayoutId {
        convert_id(lock(&self.inner.pipeline_layouts).insert(Placeholder))
    }

    fn create_render_pipeline(
        &self,
        _descriptor: &rhi::RenderPipelineDescriptor,
    ) -> rhi::RenderPipelineId {
        convert_id(lock(&self.inner.render_pipelines).insert(Placeholder))
    }

    fn create_compute_pipeline(
        &self,
        _descriptor: &rhi::ComputePipelineDescriptor,
    ) -> rhi::ComputePipelineId {
        convert_id(lock(&self.inner.compute_pipelines).insert(Placeholder))
    }

    fn destroy_buffer(&self, id: rhi::BufferId) {
        lock(&self.inner.buffers).remove(convert_id(id));
    }

    fn destroy_texture(&self, id: rhi::TextureId) {
        lock(&self.inner.textures).remove(convert_id(id));
    }

    fn destroy_texture_view(&self, id: rhi::TextureViewId) {
        lock(&self.inner.texture_views).remove(convert_id(id));
    }

    fn destroy_sampler(&self, id: rhi::SamplerId) {
        lock(&self.inner.samplers).remove(convert_id(id));
    }

    fn destroy_shader_module(&self, id: rhi::ShaderModuleId) {
        lock(&self.inner.shaders).remove(convert_id(id));
    }

    fn destroy_bind_group_layout(&self, id: rhi::BindGroupLayoutId) {
        lock(&self.inner.bind_group_layouts).remove(convert_id(id));
    }

    fn destroy_bind_group(&self, id: rhi::BindGroupId) {
        lock(&self.inner.bind_groups).remove(convert_id(id));
    }

    fn destroy_pipeline_layout(&self, id: rhi::PipelineLayoutId) {
        lock(&self.inner.pipeline_layouts).remove(convert_id(id));
    }

    fn destroy_render_pipeline(&self, id: rhi::RenderPipelineId) {
        lock(&self.inner.render_pipelines).remove(convert_id(id));
    }

    fn destroy_compute_pipeline(&self, id: rhi::ComputePipelineId) {
        lock(&self.inner.compute_pipelines).remove(convert_id(id));
    }
}
