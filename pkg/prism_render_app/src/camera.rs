//! The main world's render intent: a minimal camera/view placeholder.

use prism_ecs::prelude::Resource;
use prism_render_driver::Extent3d;

/// The pixel size of the frame the render sub-app should produce.
///
/// A tiny value type so the camera placeholder and the render target can share
/// one notion of "how big is this frame". Zero in either axis means the frame
/// is not worth rendering (a minimized window), which the render system skips.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RenderViewport {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl RenderViewport {
    /// A viewport of `width`×`height` pixels.
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// The viewport matching an [`Extent3d`] (its depth/layers are ignored).
    #[must_use]
    pub const fn from_extent(extent: Extent3d) -> Self {
        Self {
            width: extent.width,
            height: extent.height,
        }
    }

    /// The viewport as a 2D [`Extent3d`] with a single layer.
    #[must_use]
    pub const fn extent(self) -> Extent3d {
        Extent3d::new_2d(self.width, self.height)
    }

    /// Whether either axis is zero, i.e. there is nothing to render.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }
}

impl Default for RenderViewport {
    fn default() -> Self {
        Self::from_extent(crate::config::RenderAppConfig::DEFAULT.extent)
    }
}

/// The main world's render intent.
///
/// This is the authoritative, simulation-side description of the frame: a
/// camera placeholder carrying the viewport the renderer should fill. The
/// [`PrismRenderPlugin`](crate::PrismRenderPlugin) installs a default when it
/// builds and reads it every frame through the extract step, so a gameplay
/// system can resize the frame by writing this resource and nothing in the main
/// world ever touches the GPU.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Resource)]
pub struct RenderCamera {
    /// The pixel size of the frame to render.
    pub viewport: RenderViewport,
}

impl RenderCamera {
    /// A camera rendering a `viewport`-sized frame.
    #[must_use]
    pub const fn new(viewport: RenderViewport) -> Self {
        Self { viewport }
    }

    /// A camera rendering an `extent`-sized frame.
    #[must_use]
    pub const fn from_extent(extent: Extent3d) -> Self {
        Self::new(RenderViewport::from_extent(extent))
    }

    /// The frame size this camera asks the renderer to produce.
    #[must_use]
    pub const fn viewport(self) -> RenderViewport {
        self.viewport
    }
}

impl Default for RenderCamera {
    fn default() -> Self {
        Self::from_extent(crate::config::RenderAppConfig::DEFAULT.extent)
    }
}
