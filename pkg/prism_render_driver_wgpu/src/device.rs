//! The [`WgpuDevice`] resource factory: a `wgpu::Device` behind the frozen
//! [`rhi::RenderDevice`] trait.
//!
//! Every `create_*` call translates an RHI descriptor into its wgpu equivalent
//! (via [`crate::convert`]), creates the real GPU object, and stores it in a
//! generational slot map keyed by the backend's private marker types. The RHI
//! `*Id` handed back is structurally the same `(index, generation)` pair, so
//! [`convert::convert_id`] rebrands the slot-map id into the public RHI id and
//! back on resolution.
//!
//! All slot maps live behind [`Mutex`] so the device is `Send + Sync`, matching
//! the trait's `&self` methods. wgpu resource handles are cheap `Arc`-backed
//! clones, so command replay (in [`crate::queue`]) resolves an id by briefly
//! locking a map, cloning the handle, and releasing the lock.

use alloc::borrow::Cow;
use alloc::sync::Arc;
use core::num::NonZeroU64;
use std::sync::Mutex;

use prism_render_driver as rhi;
use rhi::RenderDevice;

use crate::convert;

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

/// A stored buffer plus the metadata the backend needs after creation.
struct StoredBuffer {
    /// The live wgpu buffer handle.
    buffer: wgpu::Buffer,
}

/// A stored texture plus the metadata views and uploads need.
struct StoredTexture {
    /// The live wgpu texture handle.
    texture: wgpu::Texture,
}

/// A short alias for the backend's per-kind slot maps.
type Map<K, V> = Mutex<rhi::GenerationalSlotMap<K, V>>;

/// The shared, reference-counted state behind a [`WgpuDevice`] and its queue.
///
/// Holds the wgpu device, the reported capabilities, and one locked slot map
/// per resource kind. Shared via [`Arc`] so a [`crate::queue::WgpuQueue`] can
/// resolve the same ids the device minted.
pub(crate) struct DeviceInner {
    /// The adapter the device was created from, retained for surface
    /// configuration (which needs the adapter's surface capabilities).
    pub(crate) adapter: wgpu::Adapter,
    /// The underlying wgpu logical device.
    pub(crate) device: wgpu::Device,
    /// The capabilities reported through [`rhi::RenderDevice::capabilities`].
    caps: rhi::DeviceCapabilities,
    /// Live buffers.
    buffers: Map<BufferMarker, StoredBuffer>,
    /// Live textures.
    textures: Map<TextureMarker, StoredTexture>,
    /// Live texture views.
    texture_views: Map<TextureViewMarker, wgpu::TextureView>,
    /// Live samplers.
    samplers: Map<SamplerMarker, wgpu::Sampler>,
    /// Live shader modules.
    shaders: Map<ShaderMarker, wgpu::ShaderModule>,
    /// Live bind group layouts.
    bind_group_layouts: Map<BindGroupLayoutMarker, wgpu::BindGroupLayout>,
    /// Live bind groups.
    bind_groups: Map<BindGroupMarker, wgpu::BindGroup>,
    /// Live pipeline layouts.
    pipeline_layouts: Map<PipelineLayoutMarker, wgpu::PipelineLayout>,
    /// Live render pipelines.
    render_pipelines: Map<RenderPipelineMarker, wgpu::RenderPipeline>,
    /// Live compute pipelines.
    compute_pipelines: Map<ComputePipelineMarker, wgpu::ComputePipeline>,
}

/// Locks a mutex, treating poisoning (a panic in another thread while the lock
/// was held) as unrecoverable — the slot map would be in an unknown state.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl DeviceInner {
    /// Builds the shared state from a ready wgpu device and its capabilities.
    fn new(adapter: wgpu::Adapter, device: wgpu::Device, caps: rhi::DeviceCapabilities) -> Self {
        Self {
            adapter,
            device,
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

    /// Resolves a buffer id to a cloned wgpu handle, panicking on a stale id.
    pub(crate) fn resolve_buffer(&self, id: rhi::BufferId) -> wgpu::Buffer {
        lock(&self.buffers)
            .get(convert::convert_id(id))
            .expect("stale BufferId in command stream")
            .buffer
            .clone()
    }

    /// Resolves a texture id to a cloned wgpu handle, panicking on a stale id.
    pub(crate) fn resolve_texture(&self, id: rhi::TextureId) -> wgpu::Texture {
        lock(&self.textures)
            .get(convert::convert_id(id))
            .expect("stale TextureId in upload")
            .texture
            .clone()
    }

    /// Resolves a texture-view id to a cloned wgpu handle.
    pub(crate) fn resolve_texture_view(&self, id: rhi::TextureViewId) -> wgpu::TextureView {
        lock(&self.texture_views)
            .get(convert::convert_id(id))
            .expect("stale TextureViewId in command stream")
            .clone()
    }

    /// Resolves a bind-group id to a cloned wgpu handle.
    pub(crate) fn resolve_bind_group(&self, id: rhi::BindGroupId) -> wgpu::BindGroup {
        lock(&self.bind_groups)
            .get(convert::convert_id(id))
            .expect("stale BindGroupId in command stream")
            .clone()
    }

    /// Resolves a render-pipeline id to a cloned wgpu handle.
    pub(crate) fn resolve_render_pipeline(
        &self,
        id: rhi::RenderPipelineId,
    ) -> wgpu::RenderPipeline {
        lock(&self.render_pipelines)
            .get(convert::convert_id(id))
            .expect("stale RenderPipelineId in command stream")
            .clone()
    }

    /// Resolves a compute-pipeline id to a cloned wgpu handle.
    pub(crate) fn resolve_compute_pipeline(
        &self,
        id: rhi::ComputePipelineId,
    ) -> wgpu::ComputePipeline {
        lock(&self.compute_pipelines)
            .get(convert::convert_id(id))
            .expect("stale ComputePipelineId in command stream")
            .clone()
    }
}

/// A wgpu-backed [`rhi::RenderDevice`]: the factory for every GPU resource.
///
/// Clone is cheap — it shares the same underlying device and resource tables —
/// which lets the device and its [`crate::queue::WgpuQueue`] co-own the state.
#[derive(Clone)]
pub struct WgpuDevice {
    /// The shared device state.
    inner: Arc<DeviceInner>,
}

impl WgpuDevice {
    /// Wraps a ready wgpu device and its reported capabilities.
    #[must_use]
    pub(crate) fn from_parts(
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        caps: rhi::DeviceCapabilities,
    ) -> Self {
        Self {
            inner: Arc::new(DeviceInner::new(adapter, device, caps)),
        }
    }

    /// Returns a handle to the shared inner state, used to build the queue.
    #[must_use]
    pub(crate) fn inner(&self) -> Arc<DeviceInner> {
        Arc::clone(&self.inner)
    }

    /// Borrows the underlying wgpu device, e.g. to configure a surface.
    #[must_use]
    pub fn wgpu_device(&self) -> &wgpu::Device {
        &self.inner.device
    }

    /// Borrows the adapter the device was created from, e.g. to query a
    /// surface's supported formats before configuring it.
    #[must_use]
    pub fn wgpu_adapter(&self) -> &wgpu::Adapter {
        &self.inner.adapter
    }

    /// Blocks the calling thread until every previously submitted operation on
    /// this device has completed.
    ///
    /// The frozen RHI queue is fire-and-forget, so this is the backend-specific
    /// hook callers use to force a sync point (e.g. before reading back a
    /// buffer or asserting that submitted work validated). It wraps
    /// `wgpu::Device::poll(PollType::Wait)` so callers need not name a wgpu
    /// type; a device loss during the wait is reported as `false`.
    pub fn poll_wait(&self) -> bool {
        self.inner
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .is_ok()
    }
}

impl RenderDevice for WgpuDevice {
    fn capabilities(&self) -> &rhi::DeviceCapabilities {
        &self.inner.caps
    }

    fn create_buffer(&self, descriptor: &rhi::BufferDescriptor) -> rhi::BufferId {
        let buffer = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: descriptor.label.as_deref(),
            size: descriptor.size,
            usage: convert::buffer_usages(descriptor.usage),
            mapped_at_creation: descriptor.mapped_at_creation,
        });
        let id = lock(&self.inner.buffers).insert(StoredBuffer { buffer });
        convert::convert_id(id)
    }

    fn create_texture(&self, descriptor: &rhi::TextureDescriptor) -> rhi::TextureId {
        let texture = self.inner.device.create_texture(&wgpu::TextureDescriptor {
            label: descriptor.label.as_deref(),
            size: convert::extent3d(descriptor.size),
            mip_level_count: descriptor.mip_level_count,
            sample_count: descriptor.sample_count,
            dimension: convert::texture_dimension(descriptor.dimension),
            format: convert::texture_format(descriptor.format),
            usage: convert::texture_usages(descriptor.usage),
            view_formats: &[],
        });
        let id = lock(&self.inner.textures).insert(StoredTexture { texture });
        convert::convert_id(id)
    }

    fn create_texture_view(
        &self,
        texture: rhi::TextureId,
        descriptor: &rhi::TextureViewDescriptor,
    ) -> rhi::TextureViewId {
        let handle = self.inner.resolve_texture(texture);
        let view = handle.create_view(&wgpu::TextureViewDescriptor {
            label: descriptor.label.as_deref(),
            format: descriptor.format.map(convert::texture_format),
            dimension: Some(convert::texture_view_dimension(descriptor.dimension)),
            usage: None,
            aspect: convert::texture_aspect(descriptor.aspect),
            base_mip_level: descriptor.base_mip_level,
            mip_level_count: descriptor.mip_level_count,
            base_array_layer: descriptor.base_array_layer,
            array_layer_count: descriptor.array_layer_count,
        });
        let id = lock(&self.inner.texture_views).insert(view);
        convert::convert_id(id)
    }

    fn create_sampler(&self, descriptor: &rhi::SamplerDescriptor) -> rhi::SamplerId {
        let sampler = self.inner.device.create_sampler(&wgpu::SamplerDescriptor {
            label: descriptor.label.as_deref(),
            address_mode_u: convert::address_mode(descriptor.address_mode_u),
            address_mode_v: convert::address_mode(descriptor.address_mode_v),
            address_mode_w: convert::address_mode(descriptor.address_mode_w),
            mag_filter: convert::filter_mode(descriptor.mag_filter),
            min_filter: convert::filter_mode(descriptor.min_filter),
            mipmap_filter: convert::mipmap_filter_mode(descriptor.mipmap_filter),
            lod_min_clamp: descriptor.lod_min_clamp,
            lod_max_clamp: descriptor.lod_max_clamp,
            compare: descriptor.compare.map(convert::compare_function),
            anisotropy_clamp: descriptor.anisotropy_clamp,
            border_color: descriptor.border_color.map(convert::sampler_border_color),
        });
        let id = lock(&self.inner.samplers).insert(sampler);
        convert::convert_id(id)
    }

    fn create_shader_module(
        &self,
        descriptor: &rhi::ShaderModuleDescriptor,
    ) -> rhi::ShaderModuleId {
        let text = descriptor.source.as_text().unwrap_or_else(|| {
            panic!("the wgpu backend requires WGSL/WESL text; SpirV source is unsupported")
        });
        let module = self
            .inner
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: descriptor.label.as_deref(),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(text.to_owned())),
            });
        let id = lock(&self.inner.shaders).insert(module);
        convert::convert_id(id)
    }

    fn create_bind_group_layout(
        &self,
        descriptor: &rhi::BindGroupLayoutDescriptor,
    ) -> rhi::BindGroupLayoutId {
        let entries: Vec<wgpu::BindGroupLayoutEntry> = descriptor
            .entries
            .iter()
            .map(convert::bind_group_layout_entry)
            .collect();
        let layout = self
            .inner
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: descriptor.label.as_deref(),
                entries: &entries,
            });
        let id = lock(&self.inner.bind_group_layouts).insert(layout);
        convert::convert_id(id)
    }

    fn create_bind_group(&self, descriptor: &rhi::BindGroupDescriptor) -> rhi::BindGroupId {
        let layout_id = descriptor
            .layout
            .expect("BindGroupDescriptor requires an explicit layout for the wgpu backend");

        // Resolve every referenced resource into an owned handle first, so the
        // borrowed `BindGroupEntry`s below outlive the slot-map locks.
        let owned: Vec<(u32, OwnedBinding)> = {
            let buffers = lock(&self.inner.buffers);
            let samplers = lock(&self.inner.samplers);
            let views = lock(&self.inner.texture_views);
            descriptor
                .entries
                .iter()
                .map(|entry| {
                    let binding = match entry.resource {
                        rhi::BindingResource::Buffer {
                            buffer,
                            offset,
                            size,
                        } => OwnedBinding::Buffer {
                            buffer: buffers
                                .get(convert::convert_id(buffer))
                                .expect("stale BufferId in bind group")
                                .buffer
                                .clone(),
                            offset,
                            size: size.and_then(NonZeroU64::new),
                        },
                        rhi::BindingResource::Sampler(sampler) => OwnedBinding::Sampler(
                            samplers
                                .get(convert::convert_id(sampler))
                                .expect("stale SamplerId in bind group")
                                .clone(),
                        ),
                        rhi::BindingResource::TextureView(view) => OwnedBinding::TextureView(
                            views
                                .get(convert::convert_id(view))
                                .expect("stale TextureViewId in bind group")
                                .clone(),
                        ),
                    };
                    (entry.binding, binding)
                })
                .collect()
        };

        let wgpu_entries: Vec<wgpu::BindGroupEntry> = owned
            .iter()
            .map(|(binding, resource)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: resource.as_wgpu(),
            })
            .collect();

        let layout = lock(&self.inner.bind_group_layouts)
            .get(convert::convert_id(layout_id))
            .expect("stale BindGroupLayoutId in bind group")
            .clone();

        let bind_group = self
            .inner
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: descriptor.label.as_deref(),
                layout: &layout,
                entries: &wgpu_entries,
            });
        let id = lock(&self.inner.bind_groups).insert(bind_group);
        convert::convert_id(id)
    }

    fn create_pipeline_layout(
        &self,
        descriptor: &rhi::PipelineLayoutDescriptor,
    ) -> rhi::PipelineLayoutId {
        let layouts: Vec<wgpu::BindGroupLayout> = {
            let bgls = lock(&self.inner.bind_group_layouts);
            descriptor
                .bind_group_layouts
                .iter()
                .map(|id| {
                    bgls.get(convert::convert_id(*id))
                        .expect("stale BindGroupLayoutId in pipeline layout")
                        .clone()
                })
                .collect()
        };
        let refs: Vec<Option<&wgpu::BindGroupLayout>> = layouts.iter().map(Some).collect();
        let layout = self
            .inner
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: descriptor.label.as_deref(),
                bind_group_layouts: &refs,
                immediate_size: convert::immediate_size(&descriptor.push_constant_ranges),
            });
        let id = lock(&self.inner.pipeline_layouts).insert(layout);
        convert::convert_id(id)
    }

    fn create_render_pipeline(
        &self,
        descriptor: &rhi::RenderPipelineDescriptor,
    ) -> rhi::RenderPipelineId {
        // Clone the shader modules and optional layout out of their maps up
        // front so the borrowed pipeline descriptor outlives the locks.
        let vertex_module = lock(&self.inner.shaders)
            .get(convert::convert_id(descriptor.vertex.module))
            .expect("stale vertex ShaderModuleId")
            .clone();
        let fragment_module = descriptor.fragment.as_ref().map(|frag| {
            lock(&self.inner.shaders)
                .get(convert::convert_id(frag.module))
                .expect("stale fragment ShaderModuleId")
                .clone()
        });
        let layout = descriptor.layout.map(|id| {
            lock(&self.inner.pipeline_layouts)
                .get(convert::convert_id(id))
                .expect("stale PipelineLayoutId")
                .clone()
        });

        // Vertex buffer attributes need stable owned storage referenced by the
        // `VertexBufferLayout`s.
        let attribute_sets: Vec<Vec<wgpu::VertexAttribute>> = descriptor
            .vertex
            .buffers
            .iter()
            .map(|layout| {
                layout
                    .attributes
                    .iter()
                    .map(|attr| wgpu::VertexAttribute {
                        format: convert::vertex_format(attr.format),
                        offset: attr.offset,
                        shader_location: attr.shader_location,
                    })
                    .collect()
            })
            .collect();
        let vertex_buffers: Vec<Option<wgpu::VertexBufferLayout>> = descriptor
            .vertex
            .buffers
            .iter()
            .zip(attribute_sets.iter())
            .map(|(layout, attributes)| {
                Some(wgpu::VertexBufferLayout {
                    array_stride: layout.array_stride,
                    step_mode: convert::vertex_step_mode(layout.step_mode),
                    attributes,
                })
            })
            .collect();

        let color_targets: Vec<Option<wgpu::ColorTargetState>> = descriptor
            .fragment
            .as_ref()
            .map(|frag| {
                frag.targets
                    .iter()
                    .map(|target| {
                        target.map(|t| wgpu::ColorTargetState {
                            format: convert::texture_format(t.format),
                            blend: t.blend.map(convert::blend_state),
                            write_mask: convert::color_writes(t.write_mask),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let fragment_state = fragment_module
            .as_ref()
            .zip(descriptor.fragment.as_ref())
            .map(|(module, frag)| wgpu::FragmentState {
                module,
                entry_point: Some(frag.entry_point.as_ref()),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &color_targets,
            });

        let pipeline = self
            .inner
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: descriptor.label.as_deref(),
                layout: layout.as_ref(),
                vertex: wgpu::VertexState {
                    module: &vertex_module,
                    entry_point: Some(descriptor.vertex.entry_point.as_ref()),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &vertex_buffers,
                },
                primitive: convert::primitive_state(descriptor.primitive),
                depth_stencil: descriptor.depth_stencil.map(convert::depth_stencil_state),
                multisample: convert::multisample_state(descriptor.multisample),
                fragment: fragment_state,
                multiview_mask: None,
                cache: None,
            });
        let id = lock(&self.inner.render_pipelines).insert(pipeline);
        convert::convert_id(id)
    }

    fn create_compute_pipeline(
        &self,
        descriptor: &rhi::ComputePipelineDescriptor,
    ) -> rhi::ComputePipelineId {
        let module = lock(&self.inner.shaders)
            .get(convert::convert_id(descriptor.module))
            .expect("stale compute ShaderModuleId")
            .clone();
        let layout = descriptor.layout.map(|id| {
            lock(&self.inner.pipeline_layouts)
                .get(convert::convert_id(id))
                .expect("stale PipelineLayoutId")
                .clone()
        });
        let pipeline =
            self.inner
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: descriptor.label.as_deref(),
                    layout: layout.as_ref(),
                    module: &module,
                    entry_point: Some(descriptor.entry_point.as_ref()),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });
        let id = lock(&self.inner.compute_pipelines).insert(pipeline);
        convert::convert_id(id)
    }

    fn destroy_buffer(&self, id: rhi::BufferId) {
        lock(&self.inner.buffers).remove(convert::convert_id(id));
    }

    fn destroy_texture(&self, id: rhi::TextureId) {
        lock(&self.inner.textures).remove(convert::convert_id(id));
    }

    fn destroy_texture_view(&self, id: rhi::TextureViewId) {
        lock(&self.inner.texture_views).remove(convert::convert_id(id));
    }

    fn destroy_sampler(&self, id: rhi::SamplerId) {
        lock(&self.inner.samplers).remove(convert::convert_id(id));
    }

    fn destroy_shader_module(&self, id: rhi::ShaderModuleId) {
        lock(&self.inner.shaders).remove(convert::convert_id(id));
    }

    fn destroy_bind_group_layout(&self, id: rhi::BindGroupLayoutId) {
        lock(&self.inner.bind_group_layouts).remove(convert::convert_id(id));
    }

    fn destroy_bind_group(&self, id: rhi::BindGroupId) {
        lock(&self.inner.bind_groups).remove(convert::convert_id(id));
    }

    fn destroy_pipeline_layout(&self, id: rhi::PipelineLayoutId) {
        lock(&self.inner.pipeline_layouts).remove(convert::convert_id(id));
    }

    fn destroy_render_pipeline(&self, id: rhi::RenderPipelineId) {
        lock(&self.inner.render_pipelines).remove(convert::convert_id(id));
    }

    fn destroy_compute_pipeline(&self, id: rhi::ComputePipelineId) {
        lock(&self.inner.compute_pipelines).remove(convert::convert_id(id));
    }
}

/// An owned binding resource, holding wgpu handles alive while the borrowed
/// `wgpu::BindGroupEntry`s that reference them are assembled.
enum OwnedBinding {
    /// A buffer range binding.
    Buffer {
        /// The bound buffer.
        buffer: wgpu::Buffer,
        /// The byte offset into the buffer.
        offset: u64,
        /// The bound size, or `None` to bind to the end of the buffer.
        size: Option<NonZeroU64>,
    },
    /// A sampler binding.
    Sampler(wgpu::Sampler),
    /// A texture-view binding.
    TextureView(wgpu::TextureView),
}

impl OwnedBinding {
    /// Borrows the owned handle as a `wgpu::BindingResource`.
    fn as_wgpu(&self) -> wgpu::BindingResource<'_> {
        match self {
            Self::Buffer {
                buffer,
                offset,
                size,
            } => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: *offset,
                size: *size,
            }),
            Self::Sampler(sampler) => wgpu::BindingResource::Sampler(sampler),
            Self::TextureView(view) => wgpu::BindingResource::TextureView(view),
        }
    }
}
