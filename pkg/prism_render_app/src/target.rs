//! The offscreen frame and opaque-surface targets the render sub-app owns.

use prism_ecs::prelude::Resource;
use prism_render_driver::{
    Extent3d, FrameSync, Instance, RenderDevice, TextureDescriptor, TextureFormat, TextureId,
    TextureUsages, TextureViewDescriptor, TextureViewId,
};
use prism_render_runtime::RenderRuntime;
use std::collections::BTreeMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use crate::intent::RenderViewport;

/// The usages the frame target is created with.
///
/// It is a color attachment (the passes render into it), a storage texture
/// (lighting composition writes the scene-linear HDR output), a copy source
/// (its contents can be read back for golden-image and diagnostic verification),
/// and a copy destination (it can be pre-filled or uploaded from the CPU, which
/// is what lets a test distinguish "the frame's passes wrote this" from "the
/// texture was born zeroed").
pub(crate) const RENDER_TARGET_USAGES: TextureUsages = TextureUsages::RENDER_ATTACHMENT
    .union(TextureUsages::STORAGE_BINDING)
    .union(TextureUsages::COPY_SRC)
    .union(TextureUsages::COPY_DST)
    .union(TextureUsages::TEXTURE_BINDING);

/// The format of the opaque normal attachment.
pub const GBUFFER_FORMAT: TextureFormat = crate::lighting_gpu::COMPOSITE_NORMAL_FORMAT;

/// The compact sRGB-encoded opaque albedo attachment format.
pub(crate) const ALBEDO_GBUFFER_FORMAT: TextureFormat =
    crate::lighting_gpu::COMPOSITE_ALBEDO_FORMAT;

/// The world-position component formats required by the lighting composite contract.
pub(crate) const POSITION_X_GBUFFER_FORMAT: TextureFormat =
    crate::lighting_gpu::COMPOSITE_POSITION_X_FORMAT;
pub(crate) const POSITION_Y_GBUFFER_FORMAT: TextureFormat =
    crate::lighting_gpu::COMPOSITE_POSITION_Y_FORMAT;
pub(crate) const POSITION_Z_GBUFFER_FORMAT: TextureFormat =
    crate::lighting_gpu::COMPOSITE_POSITION_Z_FORMAT;

/// The stable surface identifier format required by the lighting composite.
pub(crate) const SURFACE_ID_GBUFFER_FORMAT: TextureFormat =
    crate::lighting_gpu::COMPOSITE_SURFACE_ID_FORMAT;

/// The usages required by every persistent GBuffer plane.
pub(crate) const GBUFFER_USAGES: TextureUsages = TextureUsages::RENDER_ATTACHMENT
    .union(TextureUsages::COPY_SRC)
    .union(TextureUsages::COPY_DST)
    .union(TextureUsages::TEXTURE_BINDING);

/// The render sub-app's offscreen frame target and opaque GBuffer.
///
/// All planes are ordinary RHI textures imported into each frame graph, owned
/// here so they survive across frames (the graph never allocates or frees them).
/// They live in one resource because every attachment must agree on size — a
/// mismatched MRT plane or depth plane is an invalid framebuffer — so they are
/// created, resized, and destroyed together.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Resource)]
pub struct RenderTarget {
    /// The final scene-linear HDR output consumed by display transform.
    texture: TextureId,
    view: TextureViewId,
    format: TextureFormat,
    /// World-space surface position components written by opaque mesh raster.
    position_x_texture: TextureId,
    position_x_view: TextureViewId,
    position_x_format: TextureFormat,
    position_y_texture: TextureId,
    position_y_view: TextureViewId,
    position_y_format: TextureFormat,
    position_z_texture: TextureId,
    position_z_view: TextureViewId,
    position_z_format: TextureFormat,
    /// Normalized world-space surface normal written by opaque mesh raster.
    normal_texture: TextureId,
    normal_view: TextureViewId,
    normal_format: TextureFormat,
    /// Scene-linear material albedo written by opaque mesh raster.
    albedo_texture: TextureId,
    albedo_view: TextureViewId,
    albedo_format: TextureFormat,
    /// Stable opaque surface identifier written by opaque mesh raster.
    surface_id_texture: TextureId,
    surface_id_view: TextureViewId,
    surface_id_format: TextureFormat,
    depth_texture: TextureId,
    depth_view: TextureViewId,
    depth_format: TextureFormat,
    viewport: RenderViewport,
    owns_color: bool,
    color_entry_state: prism_render_driver::TextureState,
    color_exit_state: prism_render_driver::TextureState,
}

impl RenderTarget {
    /// Creates a target of `extent` in [`RENDER_TARGET_FORMAT`].
    ///
    /// [`RENDER_TARGET_FORMAT`]: crate::RENDER_TARGET_FORMAT
    pub(crate) fn new<D: RenderDevice>(device: &D, extent: Extent3d) -> Self {
        Self::create(device, RenderViewport::from_extent(extent))
    }

    pub(crate) fn with_external_color<D: RenderDevice>(
        device: &D,
        texture: TextureId,
        view: TextureViewId,
        descriptor: &TextureDescriptor,
        entry_state: prism_render_driver::TextureState,
        exit_state: prism_render_driver::TextureState,
    ) -> Self {
        let mut target = Self::create(device, RenderViewport::from_extent(descriptor.size));
        device.destroy_texture_view(target.view);
        device.destroy_texture(target.texture);
        target.texture = texture;
        target.view = view;
        target.format = descriptor.format;
        target.owns_color = false;
        target.color_entry_state = entry_state;
        target.color_exit_state = exit_state;
        target
    }

    /// Creates a target of `viewport` in the crate's frame-target formats.
    fn create<D: RenderDevice>(device: &D, viewport: RenderViewport) -> Self {
        let mut pending = TargetAllocation {
            device,
            texture: None,
            view: None,
            position_x_texture: None,
            position_x_view: None,
            position_y_texture: None,
            position_y_view: None,
            position_z_texture: None,
            position_z_view: None,
            normal_texture: None,
            normal_view: None,
            albedo_texture: None,
            albedo_view: None,
            surface_id_texture: None,
            surface_id_view: None,
            depth_texture: None,
            depth_view: None,
        };
        let format = crate::RENDER_TARGET_FORMAT;
        let texture = device.create_texture(&TextureDescriptor::new_2d(
            viewport.extent(),
            format,
            RENDER_TARGET_USAGES,
        ));
        pending.texture = Some(texture);
        let view = device.create_texture_view(texture, &TextureViewDescriptor::default());
        pending.view = Some(view);

        let position_x_texture = device.create_texture(&TextureDescriptor::new_2d(
            viewport.extent(),
            POSITION_X_GBUFFER_FORMAT,
            GBUFFER_USAGES,
        ));
        pending.position_x_texture = Some(position_x_texture);
        let position_x_view =
            device.create_texture_view(position_x_texture, &TextureViewDescriptor::default());
        pending.position_x_view = Some(position_x_view);
        let position_y_texture = device.create_texture(&TextureDescriptor::new_2d(
            viewport.extent(),
            POSITION_Y_GBUFFER_FORMAT,
            GBUFFER_USAGES,
        ));
        pending.position_y_texture = Some(position_y_texture);
        let position_y_view =
            device.create_texture_view(position_y_texture, &TextureViewDescriptor::default());
        pending.position_y_view = Some(position_y_view);
        let position_z_texture = device.create_texture(&TextureDescriptor::new_2d(
            viewport.extent(),
            POSITION_Z_GBUFFER_FORMAT,
            GBUFFER_USAGES,
        ));
        pending.position_z_texture = Some(position_z_texture);
        let position_z_view =
            device.create_texture_view(position_z_texture, &TextureViewDescriptor::default());
        pending.position_z_view = Some(position_z_view);

        let normal_texture = device.create_texture(&TextureDescriptor::new_2d(
            viewport.extent(),
            GBUFFER_FORMAT,
            GBUFFER_USAGES,
        ));
        pending.normal_texture = Some(normal_texture);
        let normal_view =
            device.create_texture_view(normal_texture, &TextureViewDescriptor::default());
        pending.normal_view = Some(normal_view);

        let albedo_texture = device.create_texture(&TextureDescriptor::new_2d(
            viewport.extent(),
            ALBEDO_GBUFFER_FORMAT,
            GBUFFER_USAGES,
        ));
        pending.albedo_texture = Some(albedo_texture);
        let albedo_view =
            device.create_texture_view(albedo_texture, &TextureViewDescriptor::default());
        pending.albedo_view = Some(albedo_view);

        let surface_id_texture = device.create_texture(&TextureDescriptor::new_2d(
            viewport.extent(),
            SURFACE_ID_GBUFFER_FORMAT,
            GBUFFER_USAGES,
        ));
        pending.surface_id_texture = Some(surface_id_texture);
        let surface_id_view =
            device.create_texture_view(surface_id_texture, &TextureViewDescriptor::default());
        pending.surface_id_view = Some(surface_id_view);

        let depth_format = crate::RENDER_DEPTH_FORMAT;
        let depth_texture = device.create_texture(&TextureDescriptor::new_2d(
            viewport.extent(),
            depth_format,
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        ));
        pending.depth_texture = Some(depth_texture);
        let depth_view =
            device.create_texture_view(depth_texture, &TextureViewDescriptor::default());
        pending.depth_view = Some(depth_view);
        // Transfer ownership only after the complete MRT and depth set exists.
        pending.texture = None;
        pending.view = None;
        pending.position_x_texture = None;
        pending.position_x_view = None;
        pending.position_y_texture = None;
        pending.position_y_view = None;
        pending.position_z_texture = None;
        pending.position_z_view = None;
        pending.normal_texture = None;
        pending.normal_view = None;
        pending.albedo_texture = None;
        pending.albedo_view = None;
        pending.surface_id_texture = None;
        pending.surface_id_view = None;
        pending.depth_texture = None;
        pending.depth_view = None;
        Self {
            texture,
            view,
            format,
            position_x_texture,
            position_x_view,
            position_x_format: POSITION_X_GBUFFER_FORMAT,
            position_y_texture,
            position_y_view,
            position_y_format: POSITION_Y_GBUFFER_FORMAT,
            position_z_texture,
            position_z_view,
            position_z_format: POSITION_Z_GBUFFER_FORMAT,
            normal_texture,
            normal_view,
            normal_format: GBUFFER_FORMAT,
            albedo_texture,
            albedo_view,
            albedo_format: ALBEDO_GBUFFER_FORMAT,
            surface_id_texture,
            surface_id_view,
            surface_id_format: SURFACE_ID_GBUFFER_FORMAT,
            depth_texture,
            depth_view,
            depth_format,
            viewport,
            owns_color: true,
            color_entry_state: prism_render_driver::TextureState::initial(),
            color_exit_state: prism_render_driver::TextureState::copy_src(),
        }
    }

    /// Resizes the target to `viewport`, reallocating the texture when the size
    /// actually changed.
    ///
    /// The caller must have drained in-flight frames first (see
    /// [`RenderRuntime::wait_idle`](prism_render_runtime::RenderRuntime::wait_idle)):
    /// an outstanding frame may still reference the old texture, so reallocating
    /// out from under it would be a use-after-free on the GPU.
    pub(crate) fn resize<D: RenderDevice>(&mut self, device: &D, viewport: RenderViewport) {
        if self.viewport == viewport {
            return;
        }
        let replacement = Self::create(device, viewport);
        let previous = core::mem::replace(self, replacement);
        previous.destroy(device);
    }

    /// Releases the target's textures and views.
    pub(crate) fn destroy<D: RenderDevice>(&self, device: &D) {
        if self.owns_color {
            device.destroy_texture_view(self.view);
            device.destroy_texture(self.texture);
        }
        device.destroy_texture_view(self.position_x_view);
        device.destroy_texture(self.position_x_texture);
        device.destroy_texture_view(self.position_y_view);
        device.destroy_texture(self.position_y_texture);
        device.destroy_texture_view(self.position_z_view);
        device.destroy_texture(self.position_z_texture);
        device.destroy_texture_view(self.normal_view);
        device.destroy_texture(self.normal_texture);
        device.destroy_texture_view(self.albedo_view);
        device.destroy_texture(self.albedo_texture);
        device.destroy_texture_view(self.surface_id_view);
        device.destroy_texture(self.surface_id_texture);
        device.destroy_texture_view(self.depth_view);
        device.destroy_texture(self.depth_texture);
    }

    /// The RHI texture the frame is rendered into.
    #[must_use]
    pub fn texture(&self) -> TextureId {
        self.texture
    }

    /// The default view of [`texture`](Self::texture).
    #[must_use]
    pub fn view(&self) -> TextureViewId {
        self.view
    }

    /// The current pixel size of the target.
    #[must_use]
    pub fn viewport(&self) -> RenderViewport {
        self.viewport
    }

    /// The texel format of the target.
    #[must_use]
    pub fn format(&self) -> TextureFormat {
        self.format
    }

    pub(crate) const fn color_entry_state(&self) -> prism_render_driver::TextureState {
        self.color_entry_state
    }
    pub(crate) const fn color_exit_state(&self) -> prism_render_driver::TextureState {
        self.color_exit_state
    }

    /// The RHI texture holding the world-space X position component.
    #[must_use]
    pub fn position_x_texture(&self) -> TextureId {
        self.position_x_texture
    }

    /// The default view of the world-space X position texture.
    #[must_use]
    pub fn position_x_view(&self) -> TextureViewId {
        self.position_x_view
    }

    /// The position X GBuffer plane's texel format.
    #[must_use]
    pub fn position_x_format(&self) -> TextureFormat {
        self.position_x_format
    }

    /// The RHI texture holding the world-space Y position component.
    #[must_use]
    pub fn position_y_texture(&self) -> TextureId {
        self.position_y_texture
    }
    /// The default view of the world-space Y position texture.
    #[must_use]
    pub fn position_y_view(&self) -> TextureViewId {
        self.position_y_view
    }
    /// The position Y GBuffer plane's texel format.
    #[must_use]
    pub fn position_y_format(&self) -> TextureFormat {
        self.position_y_format
    }
    /// The RHI texture holding the world-space Z position component.
    #[must_use]
    pub fn position_z_texture(&self) -> TextureId {
        self.position_z_texture
    }
    /// The default view of the world-space Z position texture.
    #[must_use]
    pub fn position_z_view(&self) -> TextureViewId {
        self.position_z_view
    }
    /// The position Z GBuffer plane's texel format.
    #[must_use]
    pub fn position_z_format(&self) -> TextureFormat {
        self.position_z_format
    }

    /// The RHI texture holding normalized world-space opaque surface normals.
    #[must_use]
    pub fn normal_texture(&self) -> TextureId {
        self.normal_texture
    }

    /// The default view of [`normal_texture`](Self::normal_texture).
    #[must_use]
    pub fn normal_view(&self) -> TextureViewId {
        self.normal_view
    }

    /// The normal GBuffer plane's texel format.
    #[must_use]
    pub fn normal_format(&self) -> TextureFormat {
        self.normal_format
    }

    /// The RHI texture holding scene-linear opaque material albedo.
    #[must_use]
    pub fn albedo_texture(&self) -> TextureId {
        self.albedo_texture
    }

    /// The default view of [`albedo_texture`](Self::albedo_texture).
    #[must_use]
    pub fn albedo_view(&self) -> TextureViewId {
        self.albedo_view
    }

    /// The albedo GBuffer plane's texel format.
    #[must_use]
    pub fn albedo_format(&self) -> TextureFormat {
        self.albedo_format
    }

    /// The RHI texture holding stable opaque surface identifiers.
    #[must_use]
    pub fn surface_id_texture(&self) -> TextureId {
        self.surface_id_texture
    }

    /// The default view of [`surface_id_texture`](Self::surface_id_texture).
    #[must_use]
    pub fn surface_id_view(&self) -> TextureViewId {
        self.surface_id_view
    }

    /// The surface-id GBuffer plane's texel format.
    #[must_use]
    pub fn surface_id_format(&self) -> TextureFormat {
        self.surface_id_format
    }

    /// The RHI texture holding the frame's depth.
    #[must_use]
    pub fn depth_texture(&self) -> TextureId {
        self.depth_texture
    }

    /// The default view of [`depth_texture`](Self::depth_texture).
    #[must_use]
    pub fn depth_view(&self) -> TextureViewId {
        self.depth_view
    }

    /// The depth plane's texel format.
    #[must_use]
    pub fn depth_format(&self) -> TextureFormat {
        self.depth_format
    }
}

struct TargetAllocation<'a, D: RenderDevice> {
    device: &'a D,
    texture: Option<TextureId>,
    view: Option<TextureViewId>,
    position_x_texture: Option<TextureId>,
    position_x_view: Option<TextureViewId>,
    position_y_texture: Option<TextureId>,
    position_y_view: Option<TextureViewId>,
    position_z_texture: Option<TextureId>,
    position_z_view: Option<TextureViewId>,
    normal_texture: Option<TextureId>,
    normal_view: Option<TextureViewId>,
    albedo_texture: Option<TextureId>,
    albedo_view: Option<TextureViewId>,
    surface_id_texture: Option<TextureId>,
    surface_id_view: Option<TextureViewId>,
    depth_texture: Option<TextureId>,
    depth_view: Option<TextureViewId>,
}

impl<D: RenderDevice> Drop for TargetAllocation<'_, D> {
    fn drop(&mut self) {
        if let Some(id) = self.depth_view {
            self.device.destroy_texture_view(id);
        }
        if let Some(id) = self.depth_texture {
            self.device.destroy_texture(id);
        }
        if let Some(id) = self.view {
            self.device.destroy_texture_view(id);
        }
        if let Some(id) = self.texture {
            self.device.destroy_texture(id);
        }
        if let Some(id) = self.albedo_view {
            self.device.destroy_texture_view(id);
        }
        if let Some(id) = self.albedo_texture {
            self.device.destroy_texture(id);
        }
        if let Some(id) = self.surface_id_view {
            self.device.destroy_texture_view(id);
        }
        if let Some(id) = self.surface_id_texture {
            self.device.destroy_texture(id);
        }
        if let Some(id) = self.normal_view {
            self.device.destroy_texture_view(id);
        }
        if let Some(id) = self.normal_texture {
            self.device.destroy_texture(id);
        }
        if let Some(id) = self.position_x_view {
            self.device.destroy_texture_view(id);
        }
        if let Some(id) = self.position_x_texture {
            self.device.destroy_texture(id);
        }
        if let Some(id) = self.position_y_view {
            self.device.destroy_texture_view(id);
        }
        if let Some(id) = self.position_y_texture {
            self.device.destroy_texture(id);
        }
        if let Some(id) = self.position_z_view {
            self.device.destroy_texture_view(id);
        }
        if let Some(id) = self.position_z_texture {
            self.device.destroy_texture(id);
        }
    }
}

/// Stable logical texture target identity. The camera keeps the stable id,
/// while the renderer keeps the generation private to owner validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextureTargetHandle {
    /// Caller-assigned stable slot.
    pub id: u64,
    /// Slot generation.
    pub generation: u64,
    /// Device generation at registration/recovery.
    pub device_generation: u64,
}

impl TextureTargetHandle {
    /// A camera route carrying both the stable slot and its exact generation.
    /// IDs and generations are bounded to 32 bits so stale camera routes fail closed.
    #[must_use]
    pub const fn route_id(self) -> u64 {
        (self.generation << 32) | (self.id & (u32::MAX as u64))
    }
}

/// Result of the latest graph attempt for one registered texture target.
///
/// A successful receipt records queue submission, not GPU completion. The
/// completion callback remains an internal lifetime signal and is deliberately
/// not exposed as a presentation/readback guarantee.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureTargetReceipt {
    /// Exact stable target handle used by the graph.
    pub handle: TextureTargetHandle,
    /// Render record that produced the graph.
    pub frame_index: u64,
    /// Runtime completion value, present only after successful submission.
    pub submission: Option<u64>,
    /// Number of admitted passes in this target graph.
    pub pass_count: u32,
    /// Whether graph admission/recording/submission succeeded.
    pub submitted: bool,
    /// Physical output texture and view actually imported by this graph.
    pub texture: TextureId,
    pub view: TextureViewId,
    /// Output contract observed at graph admission.
    pub extent: Extent3d,
    pub format: TextureFormat,
}

/// Caller-owned external texture registration contract.
#[derive(Clone)]
pub struct TextureTargetRegistration<D: RenderDevice> {
    /// Exact device instance which created the supplied texture and view.
    pub device: Arc<D>,
    /// Device generation at which the supplied identifiers were allocated.
    pub device_generation: u64,
    /// Physical texture to write.
    pub texture: TextureId,
    /// Whole-resource view used as the graph color attachment.
    pub view: TextureViewId,
    /// Exact descriptor used to create the texture.
    pub descriptor: TextureDescriptor,
    /// Descriptor contract used to create the view.
    pub view_descriptor: TextureViewDescriptor,
    /// State in which the owner leaves the texture before rendering.
    pub entry_state: prism_render_driver::TextureState,
    /// State required after rendering for the owner/readback consumer.
    pub exit_state: prism_render_driver::TextureState,
    /// Keeps the native owner and host-side synchronization state alive.
    pub owner: Arc<dyn std::any::Any + Send + Sync>,
}

/// Failure while changing a logical texture target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureTargetError {
    /// A graph attachment cannot have a zero-sized extent.
    EmptyExtent,
    /// The creating queue did not prove that the previous attachment is idle.
    CompletionUnconfirmed,
    /// The supplied runtime is not the device that owns the current slot.
    DeviceMismatch,
    /// The slot generation cannot be advanced without wrapping.
    GenerationExhausted,
    /// Camera route ids and generations must fit the encoded 32-bit fields.
    InvalidRouteIdentity,
    /// The requested extent exceeds the creating device's 2D texture limit.
    ExtentExceedsDeviceLimit,
    /// The external texture descriptor is not valid for a camera color target.
    InvalidExternalDescriptor,
    /// The supplied view is not a complete 2D view of the registered texture.
    InvalidExternalView,
}

struct TextureTargetOwner<D: RenderDevice> {
    target: RenderTarget,
    device: Arc<D>,
    device_generation: u64,
    completion: Option<Arc<AtomicBool>>,
    receipt: Option<TextureTargetReceipt>,
    external_owner: Option<Arc<dyn std::any::Any + Send + Sync>>,
    /// Lost-device resources are dropped without invoking destruction through the dead device.
    destroy_on_retire: bool,
}

/// Render-world registry for app-owned camera output attachments.
///
/// Registration and graph consumption are serialized by the render schedule.
/// Replacement/unregistration invalidate the old route immediately, while the
/// actual GPU objects remain quarantined until the creating queue confirms
/// completion. Recovery recreates active logical targets on the new device and
/// keeps old-generation objects attached to their original device owner.
pub struct TextureTargetRegistry<D: RenderDevice> {
    active: BTreeMap<u64, TextureTargetOwner<D>>,
    retired: Vec<TextureTargetOwner<D>>,
    generations: BTreeMap<u64, u64>,
}

impl<D: RenderDevice> Default for TextureTargetRegistry<D> {
    fn default() -> Self {
        Self {
            active: BTreeMap::new(),
            retired: Vec::new(),
            generations: BTreeMap::new(),
        }
    }
}

impl<D: RenderDevice + Send + Sync + 'static> Resource for TextureTargetRegistry<D> {}

impl<D: RenderDevice> TextureTargetRegistry<D> {
    /// Active registered texture targets.
    pub fn len(&self) -> usize {
        self.active.len()
    }

    /// Whether no logical texture outputs are registered.
    pub fn is_empty(&self) -> bool {
        self.active.is_empty()
    }

    /// Register an internally-owned target for a stable camera route.
    /// Re-registering replaces the current generation after proving queue
    /// completion; unchanged dimensions reuse the current generation.
    pub fn register<I: Instance<Device = D> + 'static>(
        &mut self,
        runtime: &RenderRuntime<I>,
        id: u64,
        extent: Extent3d,
    ) -> Result<TextureTargetHandle, TextureTargetError>
    where
        I::Queue: FrameSync,
    {
        if id > u64::from(u32::MAX) {
            return Err(TextureTargetError::InvalidRouteIdentity);
        }
        if extent.width == 0 || extent.height == 0 {
            return Err(TextureTargetError::EmptyExtent);
        }
        if extent.width
            > runtime
                .device()
                .capabilities()
                .limits
                .max_texture_dimension_2d
            || extent.height
                > runtime
                    .device()
                    .capabilities()
                    .limits
                    .max_texture_dimension_2d
        {
            return Err(TextureTargetError::ExtentExceedsDeviceLimit);
        }
        if let Some(owner) = self.active.get(&id) {
            if owner.device_generation != runtime.device_generation()
                || !Arc::ptr_eq(&owner.device, &runtime.device_handle())
            {
                return Err(TextureTargetError::DeviceMismatch);
            }
            if owner.target.viewport() == RenderViewport::from_extent(extent)
                && owner.device_generation == runtime.device_generation()
                && Arc::ptr_eq(&owner.device, &runtime.device_handle())
            {
                return Ok(TextureTargetHandle {
                    id,
                    generation: self.generations[&id],
                    device_generation: owner.device_generation,
                });
            }
            if !runtime.wait_idle() {
                return Err(TextureTargetError::CompletionUnconfirmed);
            }
        }
        let device = runtime.device_handle();
        let generation = self
            .generations
            .get(&id)
            .copied()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or(TextureTargetError::GenerationExhausted)?;
        if generation > u64::from(u32::MAX) {
            return Err(TextureTargetError::GenerationExhausted);
        }
        self.generations.insert(id, generation);
        if let Some(mut previous) = self.active.remove(&id) {
            previous.completion = Some(Arc::new(AtomicBool::new(true)));
            self.retired.push(previous);
        }
        self.active.insert(
            id,
            TextureTargetOwner {
                target: RenderTarget::new(&*device, extent),
                device,
                device_generation: runtime.device_generation(),
                completion: Some(Arc::new(AtomicBool::new(true))),
                receipt: None,
                external_owner: None,
                destroy_on_retire: true,
            },
        );
        Ok(TextureTargetHandle {
            id,
            generation,
            device_generation: runtime.device_generation(),
        })
    }

    /// Registers an application-owned color texture as a camera output.
    /// The registry creates no replacement color resource and never destroys
    /// the supplied texture/view; it retains owner until submitted use completes.
    pub fn register_external<I: Instance<Device = D> + 'static>(
        &mut self,
        runtime: &RenderRuntime<I>,
        id: u64,
        registration: TextureTargetRegistration<D>,
    ) -> Result<TextureTargetHandle, TextureTargetError>
    where
        I::Queue: FrameSync,
    {
        if id > u64::from(u32::MAX) {
            return Err(TextureTargetError::InvalidRouteIdentity);
        }
        let descriptor = &registration.descriptor;
        let size = descriptor.size;
        if !Arc::ptr_eq(&registration.device, &runtime.device_handle())
            || registration.device_generation != runtime.device_generation()
        {
            return Err(TextureTargetError::DeviceMismatch);
        }
        if size.width == 0
            || size.height == 0
            || size.depth_or_array_layers != 1
            || descriptor.dimension != prism_render_driver::TextureDimension::D2
            || descriptor.sample_count != 1
            || descriptor.mip_level_count != 1
            || descriptor.format != crate::RENDER_TARGET_FORMAT
            || !descriptor.usage.contains(
                TextureUsages::RENDER_ATTACHMENT
                    .union(TextureUsages::STORAGE_BINDING)
                    .union(TextureUsages::COPY_SRC)
                    .union(TextureUsages::TEXTURE_BINDING),
            )
        {
            return Err(TextureTargetError::InvalidExternalDescriptor);
        }
        let view = &registration.view_descriptor;
        if view.dimension != prism_render_driver::TextureViewDimension::D2
            || view.aspect != prism_render_driver::TextureAspect::All
            || view.base_mip_level != 0
            || !matches!(view.mip_level_count, None | Some(1))
            || view.base_array_layer != 0
            || !matches!(view.array_layer_count, None | Some(1))
            || view
                .format
                .is_some_and(|format| format != descriptor.format)
        {
            return Err(TextureTargetError::InvalidExternalView);
        }
        if size.width
            > runtime
                .device()
                .capabilities()
                .limits
                .max_texture_dimension_2d
            || size.height
                > runtime
                    .device()
                    .capabilities()
                    .limits
                    .max_texture_dimension_2d
        {
            return Err(TextureTargetError::ExtentExceedsDeviceLimit);
        }
        if self.active.get(&id).is_some_and(|owner| {
            owner.device_generation != runtime.device_generation()
                || !Arc::ptr_eq(&owner.device, &runtime.device_handle())
        }) {
            return Err(TextureTargetError::DeviceMismatch);
        }
        if self.active.contains_key(&id) && !runtime.wait_idle() {
            return Err(TextureTargetError::CompletionUnconfirmed);
        }
        let device = runtime.device_handle();
        let generation = self
            .generations
            .get(&id)
            .copied()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or(TextureTargetError::GenerationExhausted)?;
        if generation > u64::from(u32::MAX) {
            return Err(TextureTargetError::GenerationExhausted);
        }
        self.generations.insert(id, generation);
        if let Some(mut previous) = self.active.remove(&id) {
            previous.completion = Some(Arc::new(AtomicBool::new(true)));
            self.retired.push(previous);
        }
        let target = RenderTarget::with_external_color(
            &*device,
            registration.texture,
            registration.view,
            descriptor,
            registration.entry_state,
            registration.exit_state,
        );
        self.active.insert(
            id,
            TextureTargetOwner {
                target,
                device,
                device_generation: runtime.device_generation(),
                completion: Some(Arc::new(AtomicBool::new(true))),
                receipt: None,
                external_owner: Some(registration.owner),
                destroy_on_retire: true,
            },
        );
        Ok(TextureTargetHandle {
            id,
            generation,
            device_generation: runtime.device_generation(),
        })
    }

    /// Unregister the exact active generation. A stale handle has no effect.
    pub fn unregister<I: Instance<Device = D> + 'static>(
        &mut self,
        runtime: &RenderRuntime<I>,
        handle: TextureTargetHandle,
    ) -> Result<bool, TextureTargetError>
    where
        I::Queue: FrameSync,
    {
        let Some(owner) = self.active.get(&handle.id) else {
            return Ok(false);
        };
        if self.current_generation(handle.id) != Some(handle.generation)
            || owner.device_generation != handle.device_generation
            || !Arc::ptr_eq(&owner.device, &runtime.device_handle())
        {
            return Ok(false);
        }
        if !runtime.wait_idle() {
            return Err(TextureTargetError::CompletionUnconfirmed);
        }
        let Some(mut owner) = self.active.remove(&handle.id) else {
            return Ok(false);
        };
        owner.completion = Some(Arc::new(AtomicBool::new(true)));
        self.retired.push(owner);
        Ok(true)
    }

    /// Current generation for a stable slot.
    pub fn current_generation(&self, id: u64) -> Option<u64> {
        self.active
            .contains_key(&id)
            .then(|| self.generations.get(&id).copied())
            .flatten()
    }

    /// Resolve only an exact id+generation+device-generation match.
    pub fn get(&self, handle: TextureTargetHandle) -> Option<&RenderTarget> {
        let owner = self.active.get(&handle.id)?;
        (self.current_generation(handle.id) == Some(handle.generation)
            && owner.device_generation == handle.device_generation)
            .then_some(&owner.target)
    }

    /// Resolve a stable camera route only when its active owner is on the
    /// current runtime device. The returned handle carries the private
    /// generation used for receipts and retirement validation.
    pub fn resolve<I: Instance<Device = D> + 'static>(
        &self,
        runtime: &RenderRuntime<I>,
        route_id: u64,
    ) -> Option<(TextureTargetHandle, &RenderTarget)> {
        let id = route_id & u64::from(u32::MAX);
        let route_generation = route_id >> 32;
        let generation = self.generations.get(&id).copied()?;
        let owner = self.active.get(&id)?;
        if route_generation == 0
            || generation != route_generation
            || generation != self.current_generation(id)?
            || owner.device_generation != runtime.device_generation()
            || !Arc::ptr_eq(&owner.device, &runtime.device_handle())
        {
            return None;
        }
        Some((
            TextureTargetHandle {
                id,
                generation,
                device_generation: owner.device_generation,
            },
            &owner.target,
        ))
    }

    /// Iterate active targets in stable id order, restricted to this runtime.
    pub fn iter<I: Instance<Device = D> + 'static>(
        &self,
        runtime: &RenderRuntime<I>,
    ) -> impl Iterator<Item = (TextureTargetHandle, &RenderTarget)> {
        let device = runtime.device_handle();
        let device_generation = runtime.device_generation();
        self.active.iter().filter_map(move |(&id, owner)| {
            (owner.device_generation == device_generation && Arc::ptr_eq(&owner.device, &device))
                .then(|| {
                    (
                        TextureTargetHandle {
                            id,
                            generation: self.generations[&id],
                            device_generation,
                        },
                        &owner.target,
                    )
                })
        })
    }

    /// Associate actual target-specific execution evidence with a route.
    pub(crate) fn publish_receipt(&mut self, receipt: TextureTargetReceipt) -> bool {
        let Some(owner) = self.active.get_mut(&receipt.handle.id) else {
            return false;
        };
        if self.generations.get(&receipt.handle.id) != Some(&receipt.handle.generation)
            || owner.device_generation != receipt.handle.device_generation
            || receipt.submitted != receipt.submission.is_some()
            || (receipt.submitted && (receipt.pass_count == 0 || receipt.submission == Some(0)))
            || owner
                .receipt
                .is_some_and(|previous| previous.frame_index >= receipt.frame_index)
            || owner.target.texture() != receipt.texture
            || owner.target.view() != receipt.view
            || owner.target.viewport().extent() != receipt.extent
            || owner.target.format() != receipt.format
        {
            return false;
        }
        owner.receipt = Some(receipt);
        true
    }

    /// Latest receipt for the exact current handle.
    pub fn receipt(&self, handle: TextureTargetHandle) -> Option<TextureTargetReceipt> {
        self.get(handle)?;
        self.active.get(&handle.id)?.receipt
    }

    /// Returns the owner lease for graph pinning of an active generation.
    pub(crate) fn owner(
        &self,
        handle: TextureTargetHandle,
    ) -> Option<Arc<dyn std::any::Any + Send + Sync>> {
        let owner = self.active.get(&handle.id)?;
        (self.current_generation(handle.id) == Some(handle.generation))
            .then(|| owner.external_owner.as_ref().map(Arc::clone))
            .flatten()
    }

    /// Resolve an owner lease for pinning the actual output graph through GPU completion.
    pub(crate) fn external_owner(
        &self,
        handle: TextureTargetHandle,
    ) -> Option<Arc<dyn std::any::Any + Send + Sync>> {
        let owner = self.active.get(&handle.id)?;
        (self.current_generation(handle.id) == Some(handle.generation)
            && owner.device_generation == handle.device_generation)
            .then(|| owner.external_owner.as_ref().map(Arc::clone))
            .flatten()
    }

    /// Pin target attachments through the queue's actual final-use completion.
    pub fn seal<I: Instance<Device = D> + 'static>(
        &mut self,
        runtime: &RenderRuntime<I>,
        handle: TextureTargetHandle,
    ) -> bool
    where
        I::Queue: FrameSync,
    {
        let Some(owner) = self.active.get_mut(&handle.id) else {
            return false;
        };
        if self.generations.get(&handle.id) != Some(&handle.generation)
            || owner.device_generation != runtime.device_generation()
            || !Arc::ptr_eq(&owner.device, &runtime.device_handle())
        {
            return false;
        }
        let complete = Arc::new(AtomicBool::new(false));
        owner.completion = Some(Arc::clone(&complete));
        runtime
            .queue()
            .on_submitted_work_done(Box::new(move || complete.store(true, Ordering::Release)));
        true
    }

    /// Reclaim only callback-confirmed retired targets using their creating devices.
    pub fn collect_completed(&mut self) -> usize {
        let mut destroyed = 0;
        self.retired.retain(|owner| {
            if !owner
                .completion
                .as_ref()
                .is_some_and(|done| done.load(Ordering::Acquire))
            {
                return true;
            }
            if owner.destroy_on_retire {
                owner.target.destroy(&*owner.device);
            }
            destroyed += 1;
            false
        });
        destroyed
    }

    /// Release all resources after the plugin has confirmed the current runtime idle.
    /// Old/lost-device entries are dropped without calling into their device.
    pub fn shutdown_after_idle(&mut self, device: Arc<D>, generation: u64) {
        for (_, owner) in core::mem::take(&mut self.active) {
            if owner.destroy_on_retire
                && owner.device_generation == generation
                && Arc::ptr_eq(&owner.device, &device)
            {
                owner.target.destroy(&*owner.device);
            }
        }
        for owner in core::mem::take(&mut self.retired) {
            if owner.destroy_on_retire
                && owner.device_generation == generation
                && Arc::ptr_eq(&owner.device, &device)
            {
                owner.target.destroy(&*owner.device);
            }
        }
    }

    /// Recreate active internal targets after device recovery. Old owners stay
    /// quarantined until the old runtime has drained its queue; logical route
    /// IDs stay stable while their registration generations advance. External
    /// targets are unregistered because their handles belong to the old device.
    pub fn recover<I: Instance<Device = D> + 'static>(&mut self, runtime: &RenderRuntime<I>) {
        let device = runtime.device_handle();
        let ids: Vec<u64> = self.active.keys().copied().collect();
        for id in ids {
            let Some(owner) = self.active.get_mut(&id) else {
                continue;
            };
            if owner.device_generation == runtime.device_generation()
                && Arc::ptr_eq(&owner.device, &device)
            {
                continue;
            }
            if owner.external_owner.is_some() {
                let mut old = self.active.remove(&id).expect("active owner just checked");
                old.completion = Some(Arc::new(AtomicBool::new(true)));
                old.destroy_on_retire = false;
                self.retired.push(old);
                if let Some(generation) = self.generations.get_mut(&id) {
                    if let Some(next) = generation.checked_add(1) {
                        *generation = next;
                    }
                }
                continue;
            }
            let Some(generation) = self.generations.get(&id).copied() else {
                continue;
            };
            let Some(next_generation) = generation.checked_add(1) else {
                // Preserve the old allocation rather than publish a replacement
                // whose logical handle cannot be represented.
                continue;
            };
            let extent = owner.target.viewport().extent();
            let replacement = TextureTargetOwner {
                target: RenderTarget::new(&*device, extent),
                device: Arc::clone(&device),
                device_generation: runtime.device_generation(),
                completion: None,
                receipt: None,
                external_owner: owner.external_owner.clone(),
                destroy_on_retire: true,
            };
            let mut old = core::mem::replace(owner, replacement);
            self.generations.insert(id, next_generation);
            // RenderRuntime recovery has already waited on the old queue's
            // submission barrier before publishing the new device generation.
            old.completion = Some(Arc::new(AtomicBool::new(true)));
            old.destroy_on_retire = false;
            self.retired.push(old);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_driver::{DeviceDescriptor, Instance, RequestAdapterOptions};
    use prism_render_driver_null::NullInstance;

    fn null_device() -> prism_render_driver_null::NullDevice {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .expect("the null backend always has an adapter");
        instance
            .create_device(&adapter, &DeviceDescriptor::default())
            .expect("the null backend always creates a device")
            .0
    }

    #[test]
    fn composite_gbuffer_planes_have_contract_formats_and_unique_resources() {
        let device = null_device();
        let target = RenderTarget::new(&device, Extent3d::new_2d(32, 16));

        assert!(RENDER_TARGET_USAGES.contains(TextureUsages::STORAGE_BINDING));
        assert_eq!(target.position_x_format(), POSITION_X_GBUFFER_FORMAT);
        assert_eq!(target.position_y_format(), POSITION_Y_GBUFFER_FORMAT);
        assert_eq!(target.position_z_format(), POSITION_Z_GBUFFER_FORMAT);
        assert_eq!(target.normal_format(), GBUFFER_FORMAT);
        assert_eq!(target.albedo_format(), ALBEDO_GBUFFER_FORMAT);
        assert_eq!(target.surface_id_format(), SURFACE_ID_GBUFFER_FORMAT);

        let textures = [
            target.position_x_texture(),
            target.position_y_texture(),
            target.position_z_texture(),
            target.normal_texture(),
            target.albedo_texture(),
            target.surface_id_texture(),
        ];
        for (index, texture) in textures.iter().copied().enumerate() {
            assert!(!textures[..index].contains(&texture));
        }

        let views = [
            target.position_x_view(),
            target.position_y_view(),
            target.position_z_view(),
            target.normal_view(),
            target.albedo_view(),
            target.surface_id_view(),
        ];
        for (index, view) in views.iter().copied().enumerate() {
            assert!(!views[..index].contains(&view));
        }

        target.destroy(&device);
    }

    #[test]
    fn resize_replaces_surface_id_plane_with_the_target() {
        let device = null_device();
        let mut target = RenderTarget::new(&device, Extent3d::new_2d(32, 16));
        let old_surface_id = target.surface_id_texture();

        target.resize(&device, RenderViewport::new(64, 32));

        assert_eq!(target.viewport(), RenderViewport::new(64, 32));
        assert_ne!(target.surface_id_texture(), old_surface_id);
        assert_eq!(target.surface_id_format(), SURFACE_ID_GBUFFER_FORMAT);

        target.destroy(&device);
    }

    #[test]
    fn texture_target_registry_rejects_stale_routes_and_retires_replacements() {
        let extent = Extent3d::new_2d(16, 8);
        let runtime = RenderRuntime::<NullInstance>::headless(NullInstance::new(), extent)
            .expect("the null runtime always creates a device");
        let mut registry = TextureTargetRegistry::<prism_render_driver_null::NullDevice>::default();
        let first = registry
            .register(&runtime, 7, extent)
            .expect("valid target registration");
        assert_eq!(first.route_id(), (first.generation << 32) | 7);
        assert!(registry.resolve(&runtime, first.route_id()).is_some());

        let second = registry
            .register(&runtime, 7, Extent3d::new_2d(32, 16))
            .expect("replacement waits for the null queue");
        assert_eq!(second.generation, first.generation + 1);
        assert!(registry.resolve(&runtime, first.route_id()).is_none());
        assert!(registry.resolve(&runtime, second.route_id()).is_some());
        assert_eq!(registry.collect_completed(), 1);
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn texture_target_registry_requires_exact_unregister_handle() {
        let extent = Extent3d::new_2d(8, 8);
        let runtime = RenderRuntime::<NullInstance>::headless(NullInstance::new(), extent)
            .expect("the null runtime always creates a device");
        let mut registry = TextureTargetRegistry::<prism_render_driver_null::NullDevice>::default();
        let handle = registry
            .register(&runtime, 3, extent)
            .expect("valid target registration");
        let stale = TextureTargetHandle {
            generation: handle.generation - 1,
            ..handle
        };
        assert!(registry.get(stale).is_none());
        assert_eq!(registry.unregister(&runtime, stale), Ok(false));
        assert_eq!(registry.unregister(&runtime, handle), Ok(true));
        assert!(registry.resolve(&runtime, handle.route_id()).is_none());
        assert_eq!(registry.collect_completed(), 1);
        assert!(registry.is_empty());
    }

    #[test]
    fn external_texture_target_lifecycle_preserves_owner_and_receipts() {
        let extent = Extent3d::new_2d(16, 8);
        let runtime = RenderRuntime::<NullInstance>::headless(NullInstance::new(), extent)
            .expect("the null runtime always creates a device");
        let device = runtime.device_handle();
        let descriptor =
            TextureDescriptor::new_2d(extent, crate::RENDER_TARGET_FORMAT, RENDER_TARGET_USAGES);
        let texture = device.create_texture(&descriptor);
        let view_descriptor = TextureViewDescriptor::default();
        let view = device.create_texture_view(texture, &view_descriptor);
        let resources_with_external = device.live_resource_count();
        let owner = Arc::new(());
        let mut registry = TextureTargetRegistry::<prism_render_driver_null::NullDevice>::default();
        let handle = registry
            .register_external(
                &runtime,
                17,
                TextureTargetRegistration {
                    device: Arc::clone(&device),
                    device_generation: runtime.device_generation(),
                    texture,
                    view,
                    descriptor,
                    view_descriptor,
                    entry_state: prism_render_driver::TextureState::initial(),
                    exit_state: prism_render_driver::TextureState::copy_src(),
                    owner: owner.clone(),
                },
            )
            .expect("valid caller-owned output registration");
        assert_eq!(
            registry.resolve(&runtime, handle.route_id()).unwrap().0,
            handle
        );
        assert_eq!(registry.get(handle).unwrap().texture(), texture);
        let receipt = TextureTargetReceipt {
            handle,
            frame_index: 9,
            submission: Some(1),
            pass_count: 2,
            submitted: true,
            texture,
            view,
            extent,
            format: crate::RENDER_TARGET_FORMAT,
        };
        assert!(registry.publish_receipt(receipt));
        assert_eq!(registry.receipt(handle), Some(receipt));
        assert!(!registry.publish_receipt(TextureTargetReceipt {
            submission: Some(0),
            ..receipt
        }));
        assert!(!registry.publish_receipt(TextureTargetReceipt {
            pass_count: 0,
            ..receipt
        }));
        assert!(registry.unregister(&runtime, handle).unwrap());
        assert!(registry.resolve(&runtime, handle.route_id()).is_none());
        assert_eq!(registry.collect_completed(), 1);
        assert_eq!(device.live_resource_count(), resources_with_external);
        device.destroy_texture_view(view);
        device.destroy_texture(texture);
    }

    #[test]
    fn texture_target_receipts_are_generation_and_frame_monotonic() {
        let extent = Extent3d::new_2d(8, 8);
        let runtime = RenderRuntime::<NullInstance>::headless(NullInstance::new(), extent)
            .expect("the null runtime always creates a device");
        let mut registry = TextureTargetRegistry::<prism_render_driver_null::NullDevice>::default();
        let handle = registry
            .register(&runtime, 11, extent)
            .expect("valid target registration");
        let target = registry.get(handle).expect("current target");
        let receipt = TextureTargetReceipt {
            handle,
            frame_index: 4,
            submission: Some(1),
            pass_count: 3,
            submitted: true,
            texture: target.texture(),
            view: target.view(),
            extent,
            format: target.format(),
        };
        assert!(registry.publish_receipt(receipt));
        assert_eq!(registry.receipt(handle), Some(receipt));
        assert!(!registry.publish_receipt(TextureTargetReceipt {
            frame_index: 3,
            ..receipt
        }));
        assert!(!registry.publish_receipt(TextureTargetReceipt {
            submitted: false,
            ..receipt
        }));
        let replacement = registry
            .register(&runtime, 11, Extent3d::new_2d(16, 8))
            .expect("replacement invalidates the old generation");
        assert_ne!(replacement.generation, receipt.handle.generation);
        assert_eq!(registry.receipt(replacement), None);
        assert!(!registry.publish_receipt(receipt));
    }
}
