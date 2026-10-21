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
use std::sync::atomic::{AtomicBool, Ordering};
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

/// CPU-only, generation-local loss state. A host notifier must not retain the
/// native device (including when stored inside that device's loss callback).
#[derive(Default)]
struct DeviceLossState {
    lost: AtomicBool,
    reason: Mutex<Option<DeviceLostReason>>,
}

impl DeviceLossState {
    fn notify(&self, reason: DeviceLostReason) {
        let mut recorded = lock(&self.reason);
        // Serialize competing reporters and publish only the first reason.
        // Draining it must not let a duplicate notification publish it again.
        if !self.lost.swap(true, Ordering::AcqRel) {
            *recorded = Some(reason);
        }
    }

    fn notifier(
        self: &Arc<Self>,
    ) -> impl Fn(DeviceLostReason) + Clone + Send + Sync + 'static + use<> {
        let state = Arc::clone(self);
        move |reason| state.notify(reason)
    }

    fn is_lost(&self) -> bool {
        self.lost.load(Ordering::Acquire)
    }

    fn take_reason(&self) -> Option<DeviceLostReason> {
        lock(&self.reason).take()
    }
}

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
    /// Shared by the owned callback and explicit host notifications. Draining
    /// the first reason never revives this device generation's readbacks.
    loss: Arc<DeviceLossState>,
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
            loss: Arc::new(DeviceLossState::default()),
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
    /// The callback retains only CPU loss state, not `DeviceInner` or a native
    /// handle, so it cannot keep the device alive through a reference cycle.
    fn install_device_lost_callback(self: &Arc<Self>) {
        let notify = self.loss.notifier();
        self.device
            .set_device_lost_callback(move |reason, _message| {
                notify(map_device_lost(reason));
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

    /// Readback admission must reject retired IDs instead of using the
    /// command-replay resolver, whose contract deliberately panics on stale IDs.
    pub(crate) fn try_resolve_buffer(&self, id: rhi::BufferId) -> Option<wgpu::Buffer> {
        lock(&self.buffers)
            .get(convert::convert_id(id))
            .map(|stored| stored.buffer.clone())
    }

    pub(crate) fn is_lost(&self) -> bool {
        self.loss.is_lost()
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
///
/// The native resource getters clone the exact handles stored in these tables;
/// they do not create/import resources, change callbacks, or submit work. Native
/// `register_*` methods only retain handles supplied by the host; see
/// [`Self::register_buffer`] for the import contract. Pass
/// IDs minted by this device (or a clone sharing its tables). IDs are table-local:
/// an ID from another adoption can alias a live slot and is not an ownership
/// certificate. Removed/stale IDs panic instead of resolving a recycled slot.
/// A returned handle owns a native reference independently of its RHI slot;
/// dropping it does not release that slot. Conversely, RHI `destroy_*` removes
/// the slot without explicitly destroying retained native clones. Callers must
/// not explicitly destroy an exported buffer/texture while RHI or host work
/// still uses it, and must retain frame allocations until host completion.
#[derive(Clone)]
pub struct WgpuDevice {
    /// The shared device state.
    inner: Arc<DeviceInner>,
}

impl WgpuDevice {
    /// Takes ownership of an original wgpu adapter/device/queue tuple and
    /// installs Prism's device-lost callback via the owned-device path.
    ///
    /// This is not host adoption: it replaces any existing device-lost
    /// callback so [`RenderDevice::take_device_lost`] observes future loss.
    /// The uncaptured-error callback is left unchanged. Use [`Self::adopt`]
    /// when the host must retain loss handling and recovery ownership.
    /// Capabilities reflect the device's enabled features and limits, not
    /// everything the adapter could support. No device is requested and no
    /// work is submitted.
    ///
    /// # Caller-supplied ancestry and resource boundary
    ///
    /// The caller must supply the adapter, device, and queue from the same
    /// original device request. Their ancestry is caller-supplied, not
    /// authenticated here: this method cannot certify the pairing, and native
    /// device equality is not a cross-instance ownership proof. Retained
    /// native clones still share this device and its callback; do not install
    /// another loss callback if Prism must continue observing loss.
    ///
    /// Each call creates fresh RHI resource tables; no existing resources are
    /// imported. IDs cannot cross tables, even for the same native device,
    /// and may collide. Register native resources separately under the
    /// same-device contract in [`Self::register_buffer`]. Dropping the returned
    /// pair releases handles without explicitly destroying the native device;
    /// device replacement requires fresh tables and resource registrations.
    #[must_use]
    pub fn own_original(
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> (Self, crate::WgpuQueue) {
        let caps = rhi::DeviceCapabilities {
            backend: convert::backend_from_wgpu(device.adapter_info().backend),
            features: convert::features_from_wgpu(device.features()),
            limits: convert::limits_from_wgpu(&device.limits()),
        };
        let device = Self::from_parts(adapter, device, caps);
        let queue = crate::WgpuQueue::from_parts(queue, device.inner());
        (device, queue)
    }

    /// Adopts an existing host-owned wgpu device/queue without requesting a
    /// device or replacing either of the host's error/loss callbacks.
    ///
    /// Pass clones of the adapter, device, and queue from the host's original
    /// device request. Capabilities come from the device's *enabled* features
    /// and limits, not the adapter's potentially larger feature set. The host
    /// keeps submission ownership by using [`crate::WgpuQueue::encode`] and
    /// submitting its returned native command buffer itself.
    ///
    /// # Identity and resource boundary
    ///
    /// wgpu 30 exposes no safe device-owner query for a queue, buffer, or
    /// texture. This constructor therefore does not certify the supplied
    /// adapter/queue pairing and does not import any native resources. In
    /// particular, native `Device` equality alone is not a cross-instance
    /// proof: the native backend compares IDs local to its wgpu Global.
    /// Callers must supply the original host tuple. Native registration is a
    /// separate, caller-certified same-device operation, not authentication of
    /// arbitrary resources; see [`Self::register_buffer`].
    ///
    /// Each adoption creates a fresh RHI resource table. IDs cannot cross
    /// adoptions, including two adoptions of the same native device. Dropping
    /// the returned pair releases its handles; it does not call native
    /// `Device::destroy` or submit work. The host owns device-loss handling and
    /// recovery: retire old work and create a fresh bridge/resource table for
    /// the replacement device. This path installs no callback: the host must
    /// forward loss through [`Self::device_loss_notifier`] so readbacks and
    /// [`RenderDevice::take_device_lost`] observe it. Forward to each adoption
    /// separately; their loss states, like their resource tables, are distinct.
    #[must_use]
    pub fn adopt(
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> (Self, crate::WgpuQueue) {
        let caps = rhi::DeviceCapabilities {
            backend: convert::backend_from_wgpu(device.adapter_info().backend),
            features: convert::features_from_wgpu(device.features()),
            limits: convert::limits_from_wgpu(&device.limits()),
        };
        // Do not use from_parts: that owned-device path installs Prism's
        // device-lost callback and would replace the host's callback.
        let inner = Arc::new(DeviceInner::new(adapter, device, caps));
        let queue = crate::WgpuQueue::from_parts(queue, Arc::clone(&inner));
        (Self { inner }, queue)
    }

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

    /// Returns a cloneable, thread-safe notification function for this device
    /// generation. Call it with an RHI loss reason from the host's existing
    /// loss callback, or before a host-controlled device teardown. Available
    /// on both owned and adopted devices; obtaining it does not change any
    /// native callback. It retains only CPU state and may outlive the bridge.
    ///
    /// The first notification permanently rejects new/pending readbacks with
    /// `DeviceLost` and publishes one reason to
    /// [`RenderDevice::take_device_lost`]. Repeated notifications, including
    /// after the reason is drained, do not republish or reset that state.
    /// Already-consumed tickets remain `Consumed`. Recovery needs a fresh
    /// bridge/device generation, not a reset of this notification function.
    ///
    /// This is a notification contract, not an exclusive native-device lease:
    /// it does not cancel a native call already in flight or synchronize an
    /// arbitrary external `destroy`. The host must stop/drain concurrent work
    /// before teardown. Readback checks before/after native operations narrow
    /// but cannot eliminate check-to-call races. Forwarding only from a native
    /// loss callback also leaves the interval before callback delivery ungated.
    #[must_use]
    pub fn device_loss_notifier(
        &self,
    ) -> impl Fn(DeviceLostReason) + Clone + Send + Sync + 'static + use<> {
        self.inner.loss.notifier()
    }

    /// Borrows the underlying wgpu device, e.g. to configure a surface.
    #[must_use]
    pub fn wgpu_device(&self) -> &wgpu::Device {
        &self.inner.device
    }

    /// Registers an exact native buffer clone without owning its allocation.
    /// No GPU allocation, copy, submission, callback change, or device request
    /// occurs. Each call creates an independent table entry, even for the same
    /// buffer; release each entry with [`Self::release_buffer`].
    ///
    /// # Device identity and lifetime
    ///
    /// The caller must supply a live buffer created on this exact logical
    /// device, from the original host tuple. There is no safe native owner query
    /// here: importing does not validate device identity, usage, mapping state,
    /// or explicit destruction. A foreign resource may fail later wgpu
    /// validation; import success is not proof that it belongs to this device.
    ///
    /// IDs belong only to this table (or a `WgpuDevice` clone sharing it).
    /// Separate adoptions, even of the same device, may mint identical IDs.
    /// Never resolve/release an ID through another table: a colliding foreign
    /// ID can resolve or remove an unrelated live entry and cannot be detected.
    ///
    /// Keep a device/table owner and the registration alive through every ID
    /// lookup, including deferred command encoding. A device clone retains the
    /// table, not an entry removed by another caller. A registered handle keeps
    /// a native reference but cannot prevent the host calling `Buffer::destroy`;
    /// the host must not explicitly destroy it while work still uses it, and
    /// must retain reusable graph allocations until actual GPU completion.
    #[must_use = "release the registration when its table-local ID is no longer used"]
    pub fn register_buffer(&self, buffer: &wgpu::Buffer) -> rhi::BufferId {
        let id = lock(&self.inner.buffers).insert(StoredBuffer {
            buffer: buffer.clone(),
        });
        convert::convert_id(id)
    }

    /// Removes only the table entry, never calls native `Buffer::destroy`.
    ///
    /// All ID lookups must have finished. The ID becomes stale immediately;
    /// stale/repeated release on this same table is a no-op, including after
    /// slot reuse. Native clones and already-encoded commands retain their own
    /// references. Releasing the last native reference may naturally free the
    /// allocation; this is not a GPU-completion or storage-retirement signal.
    /// The table-local restriction in [`Self::register_buffer`] also applies
    /// here: IDs do not authenticate the table they came from.
    pub fn release_buffer(&self, id: rhi::BufferId) {
        lock(&self.inner.buffers).remove(convert::convert_id(id));
    }

    /// Registers an exact native texture clone, without owning its allocation.
    /// The same-device, table-local ID and lifetime obligations of
    /// [`Self::register_buffer`] apply. This does not acquire/present a surface
    /// image or extend a surface frame's validity. Release the independent
    /// entry with [`Self::release_texture`].
    #[must_use = "release the registration when its table-local ID is no longer used"]
    pub fn register_texture(&self, texture: &wgpu::Texture) -> rhi::TextureId {
        self.inner.register_texture(texture.clone())
    }

    /// Removes only a texture table entry, never calls native `Texture::destroy`.
    /// The release/identity/completion contract of [`Self::release_buffer`]
    /// applies. Independently registered views remain live in their own slots.
    pub fn release_texture(&self, id: rhi::TextureId) {
        self.inner.release_texture(id);
    }

    /// Registers an exact native view clone, retaining its native texture
    /// reference without allocating a texture or view. No texture table entry
    /// is required or created. The same-device, table-local ID and lifetime
    /// obligations of [`Self::register_buffer`] apply. Release with
    /// [`Self::release_texture_view`]; the host owns surface-frame validity.
    #[must_use = "release the registration when its table-local ID is no longer used"]
    pub fn register_texture_view(&self, view: &wgpu::TextureView) -> rhi::TextureViewId {
        self.inner.register_texture_view(view.clone())
    }

    /// Removes only a view table entry; no native texture is destroyed and no
    /// texture table entry is removed. The stale-ID, same-table and completion
    /// contract of [`Self::release_buffer`] applies.
    pub fn release_texture_view(&self, id: rhi::TextureViewId) {
        self.inner.release_texture_view(id);
    }

    /// Registers the exact host render pipeline without creating a shader,
    /// layout or replacement pipeline. The provenance, table-local ID and
    /// lifetime obligations of [`Self::register_buffer`] apply. Each import
    /// owns an independent entry; native validation still checks compatibility
    /// with the render attachments, vertex bindings and pipeline layout.
    #[must_use = "release the registration when its table-local ID is no longer used"]
    pub fn register_render_pipeline(
        &self,
        pipeline: &wgpu::RenderPipeline,
    ) -> rhi::RenderPipelineId {
        convert::convert_id(lock(&self.inner.render_pipelines).insert(pipeline.clone()))
    }

    /// Releases only this render-pipeline registration, not host clones or
    /// already encoded work. The stale-ID and completion contract of
    /// [`Self::release_buffer`] applies; release is not GPU retirement.
    pub fn release_render_pipeline(&self, id: rhi::RenderPipelineId) {
        lock(&self.inner.render_pipelines).remove(convert::convert_id(id));
    }

    /// Registers the exact host compute pipeline, without creating a shader,
    /// pipeline layout or replacement pipeline. The same-device, table-local
    /// ID and lifetime obligations of [`Self::register_buffer`] apply. Pipeline
    /// and layout compatibility remain subject to native validation at use.
    /// Each import has its own entry; release with [`Self::release_compute_pipeline`].
    #[must_use = "release the registration when its table-local ID is no longer used"]
    pub fn register_compute_pipeline(
        &self,
        pipeline: &wgpu::ComputePipeline,
    ) -> rhi::ComputePipelineId {
        convert::convert_id(lock(&self.inner.compute_pipelines).insert(pipeline.clone()))
    }

    /// Removes only this pipeline table entry, dropping its native reference.
    /// Does not invalidate host clones or encoded commands, submit or wait.
    /// The stale-ID, same-table and completion contract of
    /// [`Self::release_buffer`] applies; this is not a retirement signal.
    pub fn release_compute_pipeline(&self, id: rhi::ComputePipelineId) {
        lock(&self.inner.compute_pipelines).remove(convert::convert_id(id));
    }

    /// Registers the exact host bind group without rebuilding its bindings or
    /// allocating resources. The same-device, table-local ID and lifetime
    /// obligations of [`Self::register_buffer`] apply to the group and all its
    /// bound resources. Native layout/dynamic-offset validation still applies.
    ///
    /// No RHI entries are created for the group's layout or bound resources;
    /// its native references do not keep their separate RHI entries alive.
    /// Import does not infer graph accesses or prove resource freshness or
    /// completion. Each import has its own entry; release with [`Self::release_bind_group`].
    #[must_use = "release the registration when its table-local ID is no longer used"]
    pub fn register_bind_group(&self, bind_group: &wgpu::BindGroup) -> rhi::BindGroupId {
        convert::convert_id(lock(&self.inner.bind_groups).insert(bind_group.clone()))
    }

    /// Removes only this bind-group entry, not any bound resource's entry.
    /// Drops a native reference without explicitly destroying allocations,
    /// submitting or waiting. The stale-ID, same-table and completion contract
    /// of [`Self::release_buffer`] applies; reusable bound storage still needs
    /// actual host-confirmed completion and no future readers before reuse.
    pub fn release_bind_group(&self, id: rhi::BindGroupId) {
        lock(&self.inner.bind_groups).remove(convert::convert_id(id));
    }

    /// Clones the native buffer stored for a live, table-local RHI id.
    /// Panics for a removed/stale id; see [`Self`]'s native handle contract.
    #[must_use]
    pub fn wgpu_buffer(&self, id: rhi::BufferId) -> wgpu::Buffer {
        self.inner.resolve_buffer(id)
    }

    /// Clones the native texture stored for a live, table-local RHI id.
    /// Panics for a removed/stale id; see [`Self`]'s native handle contract.
    #[must_use]
    pub fn wgpu_texture(&self, id: rhi::TextureId) -> wgpu::Texture {
        self.inner.resolve_texture(id)
    }

    /// Clones the native view stored for a live, table-local RHI id.
    /// Panics for a removed/stale id; see [`Self`]'s native handle contract.
    #[must_use]
    pub fn wgpu_texture_view(&self, id: rhi::TextureViewId) -> wgpu::TextureView {
        self.inner.resolve_texture_view(id)
    }

    /// Clones the native sampler stored for a live, table-local RHI id.
    /// Panics for a removed/stale id; see [`Self`]'s native handle contract.
    #[must_use]
    pub fn wgpu_sampler(&self, id: rhi::SamplerId) -> wgpu::Sampler {
        lock(&self.inner.samplers)
            .get(convert::convert_id(id))
            .expect("stale SamplerId in native getter")
            .clone()
    }

    /// Clones the native shader module stored for a live, table-local RHI id.
    /// Panics for a removed/stale id; see [`Self`]'s native handle contract.
    #[must_use]
    pub fn wgpu_shader_module(&self, id: rhi::ShaderModuleId) -> wgpu::ShaderModule {
        lock(&self.inner.shaders)
            .get(convert::convert_id(id))
            .expect("stale ShaderModuleId in native getter")
            .clone()
    }

    /// Clones the native bind-group layout stored for a live, table-local RHI id.
    /// Panics for a removed/stale id; see [`Self`]'s native handle contract.
    #[must_use]
    pub fn wgpu_bind_group_layout(&self, id: rhi::BindGroupLayoutId) -> wgpu::BindGroupLayout {
        lock(&self.inner.bind_group_layouts)
            .get(convert::convert_id(id))
            .expect("stale BindGroupLayoutId in native getter")
            .clone()
    }

    /// Clones the native bind group stored for a live, table-local RHI id.
    /// Panics for a removed/stale id; see [`Self`]'s native handle contract.
    #[must_use]
    pub fn wgpu_bind_group(&self, id: rhi::BindGroupId) -> wgpu::BindGroup {
        self.inner.resolve_bind_group(id)
    }

    /// Clones the native pipeline layout stored for a live, table-local RHI id.
    /// Panics for a removed/stale id; see [`Self`]'s native handle contract.
    #[must_use]
    pub fn wgpu_pipeline_layout(&self, id: rhi::PipelineLayoutId) -> wgpu::PipelineLayout {
        lock(&self.inner.pipeline_layouts)
            .get(convert::convert_id(id))
            .expect("stale PipelineLayoutId in native getter")
            .clone()
    }

    /// Clones the native render pipeline stored for a live, table-local RHI id.
    /// Panics for a removed/stale id; see [`Self`]'s native handle contract.
    #[must_use]
    pub fn wgpu_render_pipeline(&self, id: rhi::RenderPipelineId) -> wgpu::RenderPipeline {
        self.inner.resolve_render_pipeline(id)
    }

    /// Clones the native compute pipeline stored for a live, table-local RHI id.
    /// Panics for a removed/stale id; see [`Self`]'s native handle contract.
    #[must_use]
    pub fn wgpu_compute_pipeline(&self, id: rhi::ComputePipelineId) -> wgpu::ComputePipeline {
        self.inner.resolve_compute_pipeline(id)
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
        self.inner.loss.take_reason()
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
    use std::panic::{catch_unwind, AssertUnwindSafe};

    #[test]
    fn async_readback_loss_notifier_is_cpu_only_sticky_and_cloneable() {
        let state = Arc::new(DeviceLossState::default());
        let weak = Arc::downgrade(&state);
        let notify = state.notifier();
        let duplicate = notify.clone();
        assert!(!state.is_lost());
        assert_eq!(state.take_reason(), None);
        notify(DeviceLostReason::Destroyed);
        assert!(state.is_lost());
        assert_eq!(state.take_reason(), Some(DeviceLostReason::Destroyed));
        duplicate(DeviceLostReason::Unknown);
        assert!(state.is_lost());
        assert_eq!(state.take_reason(), None, "duplicates must not republish");
        // The notifier owns only this CPU state; no adapter/device is needed,
        // and it remains callable after its original owner goes away.
        drop(state);
        notify(DeviceLostReason::Unknown);
        assert!(weak.upgrade().is_some());
        drop(notify);
        drop(duplicate);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn async_readback_loss_notifier_concurrent_reporters_publish_once() {
        let state = Arc::new(DeviceLossState::default());
        let notify = state.notifier();
        let duplicate = notify.clone();
        std::thread::scope(|scope| {
            scope.spawn(move || notify(DeviceLostReason::Unknown));
            scope.spawn(move || duplicate(DeviceLostReason::Destroyed));
        });
        assert!(state.is_lost());
        assert!(matches!(
            state.take_reason(),
            Some(DeviceLostReason::Unknown | DeviceLostReason::Destroyed)
        ));
        state.notify(DeviceLostReason::Unknown);
        assert_eq!(state.take_reason(), None);
        assert!(state.is_lost());
        assert!(
            !DeviceLossState::default().is_lost(),
            "new generations start healthy"
        );
    }

    fn import_device_or_skip() -> Option<(WgpuDevice, crate::WgpuQueue)> {
        match crate::WgpuInstance::new().request_default_hardware_device() {
            Ok(pair) => Some(pair),
            Err(crate::HardwareDeviceError::Device(error)) => {
                panic!("native import test device creation failed: {error}")
            }
            Err(error) => {
                assert!(
                    !crate::require_hardware_gpu(),
                    "PRISM_REQUIRE_GPU=1: {error}"
                );
                eprintln!("SKIPPED native import GPU test: {error}");
                None
            }
        }
    }

    #[test]
    fn original_tuple_apis_are_public_with_descriptor_only_label_lifetime() {
        let _: fn(wgpu::Adapter, wgpu::Device, wgpu::Queue) -> (WgpuDevice, crate::WgpuQueue) =
            WgpuDevice::own_original;
        let _: for<'adapter, 'descriptor> fn(
            &'adapter wgpu::Adapter,
            &'descriptor rhi::DeviceDescriptor,
        ) -> wgpu::DeviceDescriptor<'descriptor> = crate::WgpuInstance::device_descriptor;
    }

    #[test]
    fn original_tuple_preserves_enabled_caps_and_takes_over_loss_callback() {
        let instance = crate::WgpuInstance::new();
        let Some(adapter) = instance.enumerate_adapters().into_iter().find(|adapter| {
            matches!(
                adapter.get_info().device_type,
                wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::DiscreteGpu
            )
        }) else {
            assert!(
                !crate::require_hardware_gpu(),
                "PRISM_REQUIRE_GPU=1: no physical adapter for original tuple test"
            );
            eprintln!("SKIPPED original tuple GPU test: no physical adapter");
            return;
        };
        let mut descriptor = rhi::DeviceDescriptor {
            label: Some("original-tuple-test".into()),
            required_features: rhi::Features::all(),
            required_limits: convert::limits_from_wgpu(&wgpu::Limits::default()),
        };
        descriptor.required_limits.max_texture_dimension_2d = 1024;
        {
            // The adapter borrow must not constrain the returned label.
            let native = {
                let adapter_clone = adapter.clone();
                crate::WgpuInstance::device_descriptor(&adapter_clone, &descriptor)
            };
            assert_eq!(native.label, descriptor.label.as_deref());
            assert_eq!(
                native.required_features,
                convert::features_to_wgpu(descriptor.required_features) & adapter.features()
            );
            assert_eq!(
                native.required_limits,
                convert::limits_to_wgpu(descriptor.required_limits)
            );
        }
        descriptor.label = Some("".into());
        assert_eq!(
            crate::WgpuInstance::device_descriptor(&adapter, &descriptor).label,
            Some("")
        );
        descriptor.label = None;
        assert_eq!(
            crate::WgpuInstance::device_descriptor(&adapter, &descriptor).label,
            Some("prism_render_driver_wgpu device")
        );
        // Request no optional features so capability checks cannot merely
        // echo the adapter's larger supported feature set.
        descriptor.required_features = rhi::Features::NONE;
        let (native_device, native_queue) = futures_lite::future::block_on(adapter.request_device(
            &crate::WgpuInstance::device_descriptor(&adapter, &descriptor),
        ))
        .expect("physical adapter must create the original tuple");
        let (adopted, _adopted_queue) =
            WgpuDevice::adopt(adapter.clone(), native_device.clone(), native_queue.clone());
        let existing =
            adopted.create_buffer(&rhi::BufferDescriptor::new(4, rhi::BufferUsages::COPY_DST));
        let host_loss = Arc::new(Mutex::new(false));
        let callback_loss = Arc::clone(&host_loss);
        native_device.set_device_lost_callback(move |_, _| *lock(&callback_loss) = true);
        let (device, queue) =
            WgpuDevice::own_original(adapter.clone(), native_device.clone(), native_queue.clone());
        // Equality here compares retained handles from one original request,
        // not arbitrary devices or resources from independent instances.
        assert_eq!(device.wgpu_device(), &native_device);
        assert_eq!(device.wgpu_adapter(), &adapter);
        assert_eq!(queue.wgpu_queue(), &native_queue);
        assert_eq!(
            device.capabilities(),
            &rhi::DeviceCapabilities {
                backend: convert::backend_from_wgpu(native_device.adapter_info().backend),
                features: convert::features_from_wgpu(native_device.features()),
                limits: convert::limits_from_wgpu(&native_device.limits()),
            }
        );
        assert!(!Arc::ptr_eq(&device.inner, &adopted.inner));
        assert!(lock(&device.inner.buffers).is_empty());
        assert_eq!(adopted.wgpu_buffer(existing).size(), 4);
        adopted.destroy_buffer(existing);
        assert!(device.take_device_lost().is_none());
        native_device.destroy();
        let mut observed = None;
        for _ in 0..200 {
            let _ = native_device.poll(wgpu::PollType::Poll);
            observed = device.take_device_lost();
            if observed.is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(observed, Some(DeviceLostReason::Destroyed));
        assert!(
            !*lock(&host_loss),
            "Prism must replace the old loss callback"
        );
        assert!(device.take_device_lost().is_none(), "loss is drained once");
        assert!(adopted.take_device_lost().is_none());
    }

    #[test]
    fn native_import_api_is_available_on_public_device() {
        // Compile-time regression for methods accidentally placed only on
        // DeviceInner: downstream callers only have this public device type.
        let _: fn(&WgpuDevice, &wgpu::Buffer) -> rhi::BufferId = WgpuDevice::register_buffer;
        let _: fn(&WgpuDevice, rhi::BufferId) = WgpuDevice::release_buffer;
        let _: fn(&WgpuDevice, &wgpu::Texture) -> rhi::TextureId = WgpuDevice::register_texture;
        let _: fn(&WgpuDevice, rhi::TextureId) = WgpuDevice::release_texture;
        let _: fn(&WgpuDevice, &wgpu::TextureView) -> rhi::TextureViewId =
            WgpuDevice::register_texture_view;
        let _: fn(&WgpuDevice, rhi::TextureViewId) = WgpuDevice::release_texture_view;
        let _: fn(&WgpuDevice, &wgpu::RenderPipeline) -> rhi::RenderPipelineId =
            WgpuDevice::register_render_pipeline;
        let _: fn(&WgpuDevice, rhi::RenderPipelineId) = WgpuDevice::release_render_pipeline;
        let _: fn(&WgpuDevice, &wgpu::ComputePipeline) -> rhi::ComputePipelineId =
            WgpuDevice::register_compute_pipeline;
        let _: fn(&WgpuDevice, rhi::ComputePipelineId) = WgpuDevice::release_compute_pipeline;
        let _: fn(&WgpuDevice, &wgpu::BindGroup) -> rhi::BindGroupId =
            WgpuDevice::register_bind_group;
        let _: fn(&WgpuDevice, rhi::BindGroupId) = WgpuDevice::release_bind_group;
    }

    #[test]
    fn native_compute_imports_release_only_independent_generation_checked_entries() {
        let Some((device, queue)) = import_device_or_skip() else {
            return;
        };
        let native = device.wgpu_device();
        let output = native.create_buffer(&wgpu::BufferDescriptor {
            label: Some("native-compute-import-output"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let shader = native.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("native-compute-import-shader"),
            source: wgpu::ShaderSource::Wgsl(
                "@group(0) @binding(0) var<storage, read_write> output: u32;
                 @compute @workgroup_size(1) fn main() { output = 314159u; }"
                    .into(),
            ),
        });
        let pipeline = native.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("native-compute-import-pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let group = native.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("native-compute-import-group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: output.as_entire_binding(),
            }],
        });
        let pipeline_id = device.register_compute_pipeline(&pipeline);
        let pipeline_duplicate = device.register_compute_pipeline(&pipeline);
        let group_id = device.register_bind_group(&group);
        let group_duplicate = device.register_bind_group(&group);
        assert_ne!(pipeline_id, pipeline_duplicate);
        assert_ne!(group_id, group_duplicate);
        assert_eq!(device.wgpu_compute_pipeline(pipeline_id), pipeline);
        assert_eq!(device.wgpu_bind_group(group_id), group);
        // Imports do not synthesize any dependencies in other resource tables.
        assert!(lock(&device.inner.shaders).is_empty());
        assert!(lock(&device.inner.pipeline_layouts).is_empty());
        assert!(lock(&device.inner.bind_group_layouts).is_empty());
        assert!(lock(&device.inner.buffers).is_empty());
        device.release_compute_pipeline(pipeline_id);
        device.release_bind_group(group_id);
        assert!(catch_unwind(AssertUnwindSafe(
            || device.wgpu_compute_pipeline(pipeline_id)
        ))
        .is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| device.wgpu_bind_group(group_id))).is_err());
        assert_eq!(device.wgpu_compute_pipeline(pipeline_duplicate), pipeline);
        assert_eq!(device.wgpu_bind_group(group_duplicate), group);
        let pipeline_replacement = device.register_compute_pipeline(&pipeline);
        let group_replacement = device.register_bind_group(&group);
        assert_eq!(pipeline_replacement.index(), pipeline_id.index());
        assert_eq!(group_replacement.index(), group_id.index());
        assert_ne!(pipeline_replacement, pipeline_id);
        assert_ne!(group_replacement, group_id);
        device.clone().release_compute_pipeline(pipeline_id);
        device.clone().release_bind_group(group_id);
        assert_eq!(device.wgpu_compute_pipeline(pipeline_replacement), pipeline);
        assert_eq!(device.wgpu_bind_group(group_replacement), group);
        device.release_compute_pipeline(pipeline_duplicate);
        device.release_bind_group(group_duplicate);
        device.release_compute_pipeline(pipeline_replacement);
        device.release_bind_group(group_replacement);
        device.release_compute_pipeline(pipeline_replacement);
        device.release_bind_group(group_replacement);
        assert!(lock(&device.inner.compute_pipelines).is_empty());
        assert!(lock(&device.inner.bind_groups).is_empty());

        // The original native objects remain usable after every registration
        // is removed, including the group's otherwise unregistered allocation.
        let mut encoder = native.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&group), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.wgpu_queue().submit([encoder.finish()]);
        let output_id = device.register_buffer(&output);
        assert_eq!(queue.read_buffer(output_id, 4), 314159_u32.to_le_bytes());
        device.release_buffer(output_id);
    }

    #[test]
    fn native_buffer_imports_are_independent_and_stale_release_preserves_reuse() {
        let Some((device, queue)) = import_device_or_skip() else {
            return;
        };
        let native = device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("native-import-buffer"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let first = device.register_buffer(&native);
        let second = device.register_buffer(&native);
        assert_ne!(first, second);
        assert_eq!(device.wgpu_buffer(first), native);
        assert_eq!(device.wgpu_buffer(second), native);
        device.release_buffer(first);
        assert!(catch_unwind(AssertUnwindSafe(|| device.wgpu_buffer(first))).is_err());
        assert_eq!(device.wgpu_buffer(second), native);
        let replacement = device.register_buffer(&native);
        assert_eq!(
            replacement.index(),
            first.index(),
            "exercise actual slot reuse"
        );
        assert_ne!(replacement, first);
        device.clone().release_buffer(first);
        assert_eq!(device.wgpu_buffer(replacement), native);
        device.release_buffer(second);
        device.release_buffer(replacement);
        device.release_buffer(replacement);
        assert!(lock(&device.inner.buffers).is_empty());

        // Native use after all registrations were released catches accidental
        // Buffer::destroy, not merely retention of the public handle's metadata.
        queue.wgpu_queue().write_buffer(&native, 0, &[3, 1, 4, 1]);
        let again = device.register_buffer(&native);
        assert_eq!(queue.read_buffer(again, 4), [3, 1, 4, 1]);
        device.release_buffer(again);
    }

    #[test]
    fn native_texture_and_view_imports_have_independent_table_lifetimes() {
        let Some((device, _queue)) = import_device_or_skip() else {
            return;
        };
        let native = device
            .wgpu_device()
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("native-import-texture"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
        let view = native.create_view(&Default::default());
        let view_id = device.register_texture_view(&view);
        assert!(
            lock(&device.inner.textures).is_empty(),
            "view import needs no texture ID"
        );
        let first = device.register_texture(&native);
        let second = device.register_texture(&native);
        assert_ne!(first, second);
        assert_eq!(device.wgpu_texture(first), native);
        device.release_texture(first);
        let replacement = device.register_texture(&native);
        assert_eq!(replacement.index(), first.index());
        assert_ne!(replacement, first);
        device.release_texture(first);
        assert_eq!(device.wgpu_texture(replacement), native);
        assert_eq!(device.wgpu_texture(second), native);
        device.release_texture(second);
        device.release_texture(replacement);
        assert!(catch_unwind(AssertUnwindSafe(|| device.wgpu_texture(first))).is_err());
        assert_eq!(device.wgpu_texture_view(view_id), view);

        device.release_texture_view(view_id);
        assert!(catch_unwind(AssertUnwindSafe(|| device.wgpu_texture_view(view_id))).is_err());
        let new_view_id = device.register_texture_view(&view);
        assert_eq!(new_view_id.index(), view_id.index());
        assert_ne!(new_view_id, view_id);
        device.release_texture_view(view_id);
        assert_eq!(device.wgpu_texture_view(new_view_id), view);
        device.release_texture_view(new_view_id);
        device.release_texture_view(new_view_id);
        assert!(lock(&device.inner.textures).is_empty());
        assert!(lock(&device.inner.texture_views).is_empty());
    }

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
