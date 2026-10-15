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

use alloc::sync::Arc;
use core::fmt;
use core::num::NonZeroU64;
use std::sync::Mutex;

use prism_render_driver as rhi;
use prism_render_driver::DeviceLostReason;
use rhi::RenderDevice;

use crate::convert;
use crate::loss::map_device_lost;

/// The source representation understood by a shader conversion boundary.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShaderSourceKind {
    /// WGSL source text.
    Wgsl,
    /// Prism WESL source text, which is passed to wgpu as WGSL text.
    Wesl,
    /// Pre-compiled SPIR-V words.
    SpirV,
}

/// A structured error raised by the fallible wgpu backend creation interface.
///
/// `Unsupported*` means the RHI value is valid but this backend cannot
/// represent it. `Invalid*` means the descriptor itself violates an RHI
/// invariant. `CapabilityMismatch` means the descriptor is valid but the
/// selected device does not expose the capability required to honor it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BackendError {
    /// The RHI texture format has no representation in this wgpu backend.
    UnsupportedTextureFormat(rhi::TextureFormat),
    /// The RHI vertex format has no representation in this wgpu backend.
    UnsupportedVertexFormat(rhi::VertexFormat),
    /// The shader source kind is not enabled by this backend build.
    UnsupportedShaderSource(ShaderSourceKind),
    /// A bind-group layout entry violates the RHI array/count invariant.
    InvalidBindGroupLayoutEntry {
        /// The malformed binding number.
        binding: u32,
        /// The invariant that was violated.
        reason: rhi::BindGroupLayoutEntryError,
    },
    /// Two layout entries use the same binding number.
    DuplicateBinding(u32),
    /// A bind group was created without a layout.
    MissingBindGroupLayout,
    /// The device lacks a capability required by a valid descriptor.
    CapabilityMismatch {
        /// Features the descriptor requires.
        required: rhi::Features,
        /// Features the device actually exposes.
        available: rhi::Features,
    },
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedTextureFormat(format) => {
                write!(f, "unsupported RHI texture format for wgpu backend: {format:?}")
            }
            Self::UnsupportedVertexFormat(format) => {
                write!(f, "unsupported RHI vertex format for wgpu backend: {format:?}")
            }
            Self::UnsupportedShaderSource(kind) => {
                write!(f, "unsupported shader source for wgpu backend: {kind:?}")
            }
            Self::InvalidBindGroupLayoutEntry { binding, reason } => {
                write!(f, "invalid bind-group layout entry @binding({binding}): {reason}")
            }
            Self::DuplicateBinding(binding) => {
                write!(f, "invalid bind-group layout: duplicate binding {binding}")
            }
            Self::MissingBindGroupLayout => {
                f.write_str("invalid bind-group configuration: an explicit layout is required")
            }
            Self::CapabilityMismatch {
                required,
                available,
            } => write!(
                f,
                "wgpu device capability mismatch: required features {required:?}, available {available:?}"
            ),
        }
    }
}

impl std::error::Error for BackendError {}

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
    /// The most recent device loss reported by wgpu, recorded by the callback
    /// installed in [`DeviceInner::install_device_lost_callback`] and drained
    /// by [`WgpuDevice::take_device_lost`]. `None` until the device is lost.
    device_lost: Mutex<Option<DeviceLostReason>>,
}

/// Locks a mutex, treating poisoning (a panic in another thread while the lock
/// was held) as unrecoverable — the slot map would be in an unknown state.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Returns the RHI feature bit required to create a binding type on wgpu.
fn required_features_for_binding(ty: rhi::BindingType) -> rhi::Features {
    match ty {
        rhi::BindingType::BufferArray { .. } => rhi::Features::BINDLESS_BUFFER_ARRAY,
        rhi::BindingType::SamplerArray(_) => rhi::Features::BINDLESS_SAMPLER_ARRAY,
        rhi::BindingType::TextureArray { .. } => rhi::Features::BINDLESS_TEXTURE_ARRAY,
        rhi::BindingType::StorageTextureArray { .. } => {
            rhi::Features::BINDLESS_STORAGE_TEXTURE_ARRAY
        }
        rhi::BindingType::Buffer { .. }
        | rhi::BindingType::Sampler(_)
        | rhi::BindingType::Texture { .. }
        | rhi::BindingType::StorageTexture { .. } => rhi::Features::NONE,
    }
}

/// Validates that a device can honor a descriptor's optional feature needs.
pub(crate) fn validate_capabilities(
    caps: &rhi::DeviceCapabilities,
    required: rhi::Features,
) -> Result<(), BackendError> {
    if caps.features.contains(required) {
        Ok(())
    } else {
        Err(BackendError::CapabilityMismatch {
            required,
            available: caps.features,
        })
    }
}

/// Converts an error from a new fallible method for the legacy infallible RHI
/// trait surface. The old trait cannot return `Result`; callers that need
/// recoverable configuration errors must migrate to the `try_create_*`
/// methods instead of treating a failed resource as a valid id.
fn legacy<T>(result: Result<T, BackendError>) -> T {
    result.unwrap_or_else(|error| {
        panic!("{error}; use WgpuDevice::try_create_* for recoverable errors")
    })
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
            device_lost: Mutex::new(None),
        }
    }

    /// Installs the wgpu device-lost callback that records why the device went
    /// away, so [`WgpuDevice::take_device_lost`] can report it.
    ///
    /// wgpu signals a lost device through a one-shot callback rather than an
    /// error return — a driver reset or crash routes through the "lose the
    /// device" steps with reason [`DeviceLostReason::Unknown`], and an explicit
    /// `wgpu::Device::destroy` delivers [`DeviceLostReason::Destroyed`] from a
    /// later poll — so the callback is registered at bring-up to capture every
    /// loss. Only the device-lost callback is installed; the default
    /// uncaptured-error handler is left alone so genuine validation errors keep
    /// surfacing as they did before.
    ///
    /// The callback observes the loss through a weak reference: the closure is
    /// owned by the wgpu device stored in `self`, so capturing a strong
    /// reference would keep this state — and the device — alive forever. If no
    /// [`WgpuDevice`] handle is still alive when the loss fires, nobody can
    /// observe it and the reason is dropped.
    fn install_device_lost_callback(self: &Arc<Self>) {
        let inner = Arc::downgrade(self);
        self.device
            .set_device_lost_callback(move |reason, _message| {
                if let Some(inner) = inner.upgrade() {
                    *lock(&inner.device_lost) = Some(map_device_lost(reason));
                }
            });
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

    /// Registers an externally-created texture (e.g. a swapchain image) into the
    /// slot map and returns its RHI id.
    ///
    /// The caller owns the lifetime: an id registered here must be released with
    /// [`release_texture`](Self::release_texture) once it is no longer needed,
    /// or its slot leaks for the device's lifetime.
    pub(crate) fn register_texture(&self, texture: wgpu::Texture) -> rhi::TextureId {
        let id = lock(&self.textures).insert(StoredTexture { texture });
        convert::convert_id(id)
    }

    /// Registers an externally-created texture view into the slot map and
    /// returns its RHI id. See [`register_texture`](Self::register_texture) for
    /// the lifetime contract.
    pub(crate) fn register_texture_view(&self, view: wgpu::TextureView) -> rhi::TextureViewId {
        let id = lock(&self.texture_views).insert(view);
        convert::convert_id(id)
    }

    /// Releases a texture registered with [`register_texture`](Self::register_texture).
    pub(crate) fn release_texture(&self, id: rhi::TextureId) {
        lock(&self.textures).remove(convert::convert_id(id));
    }

    /// Releases a texture view registered with
    /// [`register_texture_view`](Self::register_texture_view).
    pub(crate) fn release_texture_view(&self, id: rhi::TextureViewId) {
        lock(&self.texture_views).remove(convert::convert_id(id));
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

    /// Blocks until the submission named by `index` has completed.
    ///
    /// Shared by [`WgpuDevice::wait_for_submission`] and the queue's
    /// completion-token hook so both go through one `wgpu` poll call. Returns
    /// `false` on device loss or if `wgpu` rejects the index.
    pub(crate) fn wait_for_submission(&self, index: &wgpu::SubmissionIndex) -> bool {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(index.clone()),
                timeout: None,
            })
            .is_ok()
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
        let inner = Arc::new(DeviceInner::new(adapter, device, caps));
        inner.install_device_lost_callback();
        Self { inner }
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

    /// Borrows a live buffer by its RHI id for backend-specific readback or
    /// interop tests.
    #[must_use]
    pub fn wgpu_buffer(&self, id: rhi::BufferId) -> wgpu::Buffer {
        self.inner.resolve_buffer(id)
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

    /// Blocks the calling thread until the submission identified by `index` has
    /// completed, without waiting for later submissions.
    ///
    /// This is the precise-completion hook frames-in-flight pacing is built on:
    /// a caller holds the `wgpu::SubmissionIndex` returned when it submitted a
    /// frame and polls for exactly that frame. Returns `true` once the
    /// submission is done, or `false` if the device was lost or `wgpu` rejected
    /// the index.
    pub fn wait_for_submission(&self, index: &wgpu::SubmissionIndex) -> bool {
        self.inner.wait_for_submission(index)
    }

    /// Creates a texture while returning unsupported-format errors instead of
    /// panicking.
    pub fn try_create_texture(
        &self,
        descriptor: &rhi::TextureDescriptor,
    ) -> Result<rhi::TextureId, BackendError> {
        let texture = self.inner.device.create_texture(&wgpu::TextureDescriptor {
            label: descriptor.label.as_deref(),
            size: convert::extent3d(descriptor.size),
            mip_level_count: descriptor.mip_level_count,
            sample_count: descriptor.sample_count,
            dimension: convert::texture_dimension(descriptor.dimension),
            format: convert::try_texture_format(descriptor.format)?,
            usage: convert::texture_usages(descriptor.usage),
            view_formats: &[],
        });
        let id = lock(&self.inner.textures).insert(StoredTexture { texture });
        Ok(convert::convert_id(id))
    }

    /// Creates a texture view while returning unsupported-format errors
    /// instead of panicking.
    pub fn try_create_texture_view(
        &self,
        texture: rhi::TextureId,
        descriptor: &rhi::TextureViewDescriptor,
    ) -> Result<rhi::TextureViewId, BackendError> {
        let handle = self.inner.resolve_texture(texture);
        let view = handle.create_view(&wgpu::TextureViewDescriptor {
            label: descriptor.label.as_deref(),
            format: descriptor
                .format
                .map(convert::try_texture_format)
                .transpose()?,
            dimension: Some(convert::texture_view_dimension(descriptor.dimension)),
            usage: None,
            aspect: convert::texture_aspect(descriptor.aspect),
            base_mip_level: descriptor.base_mip_level,
            mip_level_count: descriptor.mip_level_count,
            base_array_layer: descriptor.base_array_layer,
            array_layer_count: descriptor.array_layer_count,
        });
        let id = lock(&self.inner.texture_views).insert(view);
        Ok(convert::convert_id(id))
    }

    /// Creates a WGSL/WESL shader module while reporting unsupported source
    /// kinds as a structured error.
    pub fn try_create_shader_module(
        &self,
        descriptor: &rhi::ShaderModuleDescriptor,
    ) -> Result<rhi::ShaderModuleId, BackendError> {
        let module = self
            .inner
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: descriptor.label.as_deref(),
                source: convert::shader_source(&descriptor.source)?,
            });
        let id = lock(&self.inner.shaders).insert(module);
        Ok(convert::convert_id(id))
    }

    /// Creates a bind-group layout while distinguishing malformed descriptors
    /// from devices that lack the requested descriptor-array capability.
    pub fn try_create_bind_group_layout(
        &self,
        descriptor: &rhi::BindGroupLayoutDescriptor,
    ) -> Result<rhi::BindGroupLayoutId, BackendError> {
        if !descriptor.has_unique_bindings() {
            let duplicate = descriptor
                .entries
                .iter()
                .enumerate()
                .find_map(|(index, entry)| {
                    descriptor.entries[..index]
                        .iter()
                        .any(|previous| previous.binding == entry.binding)
                        .then_some(entry.binding)
                })
                .expect("duplicate binding was checked above");
            return Err(BackendError::DuplicateBinding(duplicate));
        }

        let mut required = rhi::Features::NONE;
        for entry in &descriptor.entries {
            if let Err(reason) = entry.validate() {
                return Err(BackendError::InvalidBindGroupLayoutEntry {
                    binding: entry.binding,
                    reason,
                });
            }
            required |= required_features_for_binding(entry.ty);
        }
        validate_capabilities(&self.inner.caps, required)?;

        let entries: Vec<wgpu::BindGroupLayoutEntry> = descriptor
            .entries
            .iter()
            .map(convert::bind_group_layout_entry)
            .collect::<Result<_, _>>()?;
        let layout = self
            .inner
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: descriptor.label.as_deref(),
                entries: &entries,
            });
        let id = lock(&self.inner.bind_group_layouts).insert(layout);
        Ok(convert::convert_id(id))
    }

    /// Creates a render pipeline while returning unsupported format errors.
    pub fn try_create_render_pipeline(
        &self,
        descriptor: &rhi::RenderPipelineDescriptor,
    ) -> Result<rhi::RenderPipelineId, BackendError> {
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

        let attribute_sets: Vec<Vec<wgpu::VertexAttribute>> = descriptor
            .vertex
            .buffers
            .iter()
            .map(|layout| {
                layout
                    .attributes
                    .iter()
                    .map(|attr| {
                        Ok(wgpu::VertexAttribute {
                            format: convert::try_vertex_format(attr.format)?,
                            offset: attr.offset,
                            shader_location: attr.shader_location,
                        })
                    })
                    .collect::<Result<_, BackendError>>()
            })
            .collect::<Result<_, _>>()?;
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
                        target
                            .map(|t| {
                                Ok(wgpu::ColorTargetState {
                                    format: convert::try_texture_format(t.format)?,
                                    blend: t.blend.map(convert::blend_state),
                                    write_mask: convert::color_writes(t.write_mask),
                                })
                            })
                            .transpose()
                    })
                    .collect::<Result<_, BackendError>>()
            })
            .transpose()?
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
                depth_stencil: descriptor
                    .depth_stencil
                    .map(convert::depth_stencil_state)
                    .transpose()?,
                multisample: convert::multisample_state(descriptor.multisample),
                fragment: fragment_state,
                multiview_mask: None,
                cache: None,
            });
        let id = lock(&self.inner.render_pipelines).insert(pipeline);
        Ok(convert::convert_id(id))
    }
}

impl RenderDevice for WgpuDevice {
    fn capabilities(&self) -> &rhi::DeviceCapabilities {
        &self.inner.caps
    }

    fn take_device_lost(&self) -> Option<DeviceLostReason> {
        lock(&self.inner.device_lost).take()
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
        legacy(self.try_create_texture(descriptor))
    }

    fn create_texture_view(
        &self,
        texture: rhi::TextureId,
        descriptor: &rhi::TextureViewDescriptor,
    ) -> rhi::TextureViewId {
        legacy(self.try_create_texture_view(texture, descriptor))
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
        legacy(self.try_create_shader_module(descriptor))
    }

    fn create_bind_group_layout(
        &self,
        descriptor: &rhi::BindGroupLayoutDescriptor,
    ) -> rhi::BindGroupLayoutId {
        legacy(self.try_create_bind_group_layout(descriptor))
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
            let resolve_buffer = |id: rhi::BufferId| {
                buffers
                    .get(convert::convert_id(id))
                    .expect("stale BufferId in bind group")
                    .buffer
                    .clone()
            };
            let resolve_sampler = |id: rhi::SamplerId| {
                samplers
                    .get(convert::convert_id(id))
                    .expect("stale SamplerId in bind group")
                    .clone()
            };
            let resolve_view = |id: rhi::TextureViewId| {
                views
                    .get(convert::convert_id(id))
                    .expect("stale TextureViewId in bind group")
                    .clone()
            };
            descriptor
                .entries
                .iter()
                .map(|entry| {
                    let binding = match &entry.resource {
                        rhi::BindingResource::Buffer {
                            buffer,
                            offset,
                            size,
                        } => OwnedBinding::Buffer {
                            buffer: resolve_buffer(*buffer),
                            offset: *offset,
                            size: size.and_then(NonZeroU64::new),
                        },
                        rhi::BindingResource::BufferArray(elements) => OwnedBinding::BufferArray(
                            elements
                                .iter()
                                .map(|element| OwnedBufferRange {
                                    buffer: resolve_buffer(element.buffer),
                                    offset: element.offset,
                                    size: element.size.and_then(NonZeroU64::new),
                                })
                                .collect(),
                        ),
                        rhi::BindingResource::Sampler(sampler) => {
                            OwnedBinding::Sampler(resolve_sampler(*sampler))
                        }
                        rhi::BindingResource::SamplerArray(samplers) => OwnedBinding::SamplerArray(
                            samplers.iter().map(|id| resolve_sampler(*id)).collect(),
                        ),
                        rhi::BindingResource::TextureView(view) => {
                            OwnedBinding::TextureView(resolve_view(*view))
                        }
                        rhi::BindingResource::TextureViewArray(views) => {
                            OwnedBinding::TextureViewArray(
                                views.iter().map(|id| resolve_view(*id)).collect(),
                            )
                        }
                    };
                    (entry.binding, binding)
                })
                .collect()
        };

        // wgpu borrows array elements as slices of handles, so build the
        // per-binding scratch storage before the borrowed entry list. Both
        // locals outlive `create_bind_group` below.
        let scratch: Vec<BindingScratch<'_>> = owned
            .iter()
            .map(|(_, resource)| resource.scratch())
            .collect();

        let wgpu_entries: Vec<wgpu::BindGroupEntry<'_>> = owned
            .iter()
            .zip(&scratch)
            .map(|((binding, resource), scratch)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: resource.as_wgpu(scratch),
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
        legacy(self.try_create_render_pipeline(descriptor))
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

/// One owned buffer range inside an owned buffer array.
struct OwnedBufferRange {
    /// The bound buffer.
    buffer: wgpu::Buffer,
    /// The byte offset into the buffer.
    offset: u64,
    /// The bound size, or `None` to bind to the end of the buffer.
    size: Option<NonZeroU64>,
}

/// An owned binding resource, holding wgpu handles alive while the borrowed
/// `wgpu::BindGroupEntry`s that reference them are assembled.
///
/// The array variants own a variable-length element list; their borrowed wgpu
/// form is produced through [`BindingScratch`], because
/// `wgpu::BindingResource::BufferArray`/`SamplerArray`/`TextureViewArray` are
/// slices of borrowed handles rather than owned vectors.
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
    /// A buffer-array binding (one element per `binding_array` slot).
    BufferArray(Vec<OwnedBufferRange>),
    /// A sampler binding.
    Sampler(wgpu::Sampler),
    /// A sampler-array binding.
    SamplerArray(Vec<wgpu::Sampler>),
    /// A texture-view binding.
    TextureView(wgpu::TextureView),
    /// A texture-view-array binding.
    TextureViewArray(Vec<wgpu::TextureView>),
}

/// Borrowed element storage for one bind-group entry.
///
/// Built by [`OwnedBinding::scratch`] once the owned handles exist, and handed
/// back to [`OwnedBinding::as_wgpu`] so the borrowed `wgpu::BindingResource`
/// stays sound. The variant always matches the resource kind it was built from.
enum BindingScratch<'a> {
    /// The resource needs no borrowed element list.
    None,
    /// `wgpu::BufferBinding`s backing a buffer array.
    Buffers(Vec<wgpu::BufferBinding<'a>>),
    /// Borrowed samplers backing a sampler array.
    Samplers(Vec<&'a wgpu::Sampler>),
    /// Borrowed views backing a texture-view array.
    Views(Vec<&'a wgpu::TextureView>),
}

impl OwnedBinding {
    /// Builds the borrowed element storage this binding needs.
    fn scratch(&self) -> BindingScratch<'_> {
        match self {
            Self::Buffer { .. } | Self::Sampler(_) | Self::TextureView(_) => BindingScratch::None,
            Self::BufferArray(elements) => BindingScratch::Buffers(
                elements
                    .iter()
                    .map(|element| wgpu::BufferBinding {
                        buffer: &element.buffer,
                        offset: element.offset,
                        size: element.size,
                    })
                    .collect(),
            ),
            Self::SamplerArray(samplers) => BindingScratch::Samplers(samplers.iter().collect()),
            Self::TextureViewArray(views) => BindingScratch::Views(views.iter().collect()),
        }
    }

    /// Borrows the owned handle(s) as a `wgpu::BindingResource`, using the
    /// element storage produced by [`Self::scratch`].
    fn as_wgpu<'a>(&'a self, scratch: &'a BindingScratch<'a>) -> wgpu::BindingResource<'a> {
        match (self, scratch) {
            (
                Self::Buffer {
                    buffer,
                    offset,
                    size,
                },
                _,
            ) => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: *offset,
                size: *size,
            }),
            (Self::BufferArray(_), BindingScratch::Buffers(elements)) => {
                wgpu::BindingResource::BufferArray(elements)
            }
            (Self::Sampler(sampler), _) => wgpu::BindingResource::Sampler(sampler),
            (Self::SamplerArray(_), BindingScratch::Samplers(samplers)) => {
                wgpu::BindingResource::SamplerArray(samplers)
            }
            (Self::TextureView(view), _) => wgpu::BindingResource::TextureView(view),
            (Self::TextureViewArray(_), BindingScratch::Views(views)) => {
                wgpu::BindingResource::TextureViewArray(views)
            }
            (resource, _) => unreachable!(
                "binding scratch does not match its resource kind: {}",
                resource.kind_name()
            ),
        }
    }

    /// The resource kind, for assertion messages.
    fn kind_name(&self) -> &'static str {
        match self {
            Self::Buffer { .. } => "buffer",
            Self::BufferArray(_) => "buffer array",
            Self::Sampler(_) => "sampler",
            Self::SamplerArray(_) => "sampler array",
            Self::TextureView(_) => "texture view",
            Self::TextureViewArray(_) => "texture view array",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bindless_layout_capability_mismatch_is_structured() {
        let caps = rhi::DeviceCapabilities {
            backend: rhi::Backend::Metal,
            features: rhi::Features::NONE,
            limits: rhi::Limits::baseline(),
        };
        let required = rhi::Features::BINDLESS_TEXTURE_ARRAY;
        assert_eq!(
            validate_capabilities(&caps, required),
            Err(BackendError::CapabilityMismatch {
                required,
                available: rhi::Features::NONE,
            })
        );
    }

    #[test]
    fn capability_validation_accepts_the_exact_supported_set() {
        let caps = rhi::DeviceCapabilities {
            backend: rhi::Backend::Vulkan,
            features: rhi::Features::BINDLESS_TEXTURE_ARRAY,
            limits: rhi::Limits::baseline(),
        };
        assert_eq!(
            validate_capabilities(&caps, rhi::Features::BINDLESS_TEXTURE_ARRAY),
            Ok(())
        );
    }

    #[test]
    fn invalid_bind_group_array_configuration_is_not_a_capability_fallback() {
        let entry = rhi::BindGroupLayoutEntry {
            binding: 7,
            visibility: rhi::ShaderStages::COMPUTE,
            ty: rhi::BindingType::TextureArray {
                sample_type: rhi::TextureSampleType::Float,
                view_dimension: rhi::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let error = entry.validate().unwrap_err();
        assert_eq!(
            BackendError::InvalidBindGroupLayoutEntry {
                binding: 7,
                reason: error,
            },
            BackendError::InvalidBindGroupLayoutEntry {
                binding: 7,
                reason: rhi::BindGroupLayoutEntryError::ArrayWithoutCount,
            }
        );
    }
}
