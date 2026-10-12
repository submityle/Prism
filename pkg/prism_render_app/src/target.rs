//! The offscreen frame target the render sub-app draws into.

use prism_ecs::prelude::Resource;
use prism_render_driver::{
    Extent3d, RenderDevice, TextureDescriptor, TextureFormat, TextureId, TextureUsages,
    TextureViewDescriptor, TextureViewId,
};

use crate::intent::RenderViewport;

/// The usages the frame target is created with.
///
/// It is a color attachment (the passes render into it), a copy source (its
/// contents can be read back for golden-image and diagnostic verification), and
/// a copy destination (it can be pre-filled or uploaded from the CPU, which is
/// what lets a test distinguish "the frame's passes wrote this" from "the
/// texture was born zeroed").
pub(crate) const RENDER_TARGET_USAGES: TextureUsages = TextureUsages::RENDER_ATTACHMENT
    .union(TextureUsages::COPY_SRC)
    .union(TextureUsages::COPY_DST);

/// The render sub-app's offscreen frame target: its color plane and its depth
/// plane.
///
/// Both are ordinary RHI textures imported into each frame graph, owned here so
/// they survive across frames (the graph never allocates or frees them). They
/// live in one resource because they must agree on size — a depth plane that
/// disagrees with the color plane is an invalid framebuffer — so they are
/// created, resized, and destroyed together.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Resource)]
pub struct RenderTarget {
    texture: TextureId,
    view: TextureViewId,
    format: TextureFormat,
    depth_texture: TextureId,
    depth_view: TextureViewId,
    depth_format: TextureFormat,
    viewport: RenderViewport,
}

impl RenderTarget {
    /// Creates a target of `extent` in [`RENDER_TARGET_FORMAT`].
    ///
    /// [`RENDER_TARGET_FORMAT`]: crate::RENDER_TARGET_FORMAT
    pub(crate) fn new<D: RenderDevice>(device: &D, extent: Extent3d) -> Self {
        Self::create(device, RenderViewport::from_extent(extent))
    }

    /// Creates a target of `viewport` in the crate's frame-target formats.
    fn create<D: RenderDevice>(device: &D, viewport: RenderViewport) -> Self {
        let format = crate::RENDER_TARGET_FORMAT;
        let texture = device.create_texture(&TextureDescriptor::new_2d(
            viewport.extent(),
            format,
            RENDER_TARGET_USAGES,
        ));
        let view = device.create_texture_view(texture, &TextureViewDescriptor::default());
        let depth_format = crate::RENDER_DEPTH_FORMAT;
        let depth_texture = device.create_texture(&TextureDescriptor::new_2d(
            viewport.extent(),
            depth_format,
            TextureUsages::RENDER_ATTACHMENT,
        ));
        let depth_view =
            device.create_texture_view(depth_texture, &TextureViewDescriptor::default());
        Self {
            texture,
            view,
            format,
            depth_texture,
            depth_view,
            depth_format,
            viewport,
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
        self.destroy(device);
        *self = Self::create(device, viewport);
    }

    /// Releases the target's textures and views.
    pub(crate) fn destroy<D: RenderDevice>(&self, device: &D) {
        device.destroy_texture_view(self.view);
        device.destroy_texture(self.texture);
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
