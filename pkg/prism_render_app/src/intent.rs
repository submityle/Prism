//! The main world's render intent: the frame the renderer should produce.
//!
//! The renderer deliberately owns no camera (see `views.rs`: a camera is
//! scene/gameplay state, and the scene side turns it into the views the render
//! world consumes). What the main world publishes across the boundary is only
//! the *size* of the frame to render, which the extract step copies into the
//! render world's [`RenderRecord`](crate::RenderRecord). Per-view specs (stereo
//! / shadow / reflection / capture) arrive with scene-side view extraction and
//! land in the render world's view family, not here.

use prism_ecs::prelude::Resource;
use prism_render_driver::Extent3d;

/// The pixel size of the frame the render sub-app should produce.
///
/// A tiny value type so the main world's frame intent and the render target can
/// share one notion of "how big is this frame". Zero in either axis means the
/// frame is not worth rendering (a minimized window), which the render system
/// skips.
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
/// The authoritative, simulation-side description of the frame the renderer
/// should produce — today just its size. The
/// [`PrismRenderPlugin`](crate::PrismRenderPlugin) installs a default when it
/// builds and reads it every frame through the extract step, so a gameplay
/// system can resize the frame by writing this resource and nothing in the main
/// world ever touches the GPU.
///
/// This is not a camera: the renderer consumes views, and the scene side owns
/// cameras.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Resource)]
pub struct RenderIntent {
    /// The pixel size of the frame to render.
    pub viewport: RenderViewport,
}

impl RenderIntent {
    /// An intent to render a `viewport`-sized frame.
    #[must_use]
    pub const fn new(viewport: RenderViewport) -> Self {
        Self { viewport }
    }

    /// An intent to render an `extent`-sized frame.
    #[must_use]
    pub const fn from_extent(extent: Extent3d) -> Self {
        Self::new(RenderViewport::from_extent(extent))
    }

    /// The frame size this intent asks the renderer to produce.
    #[must_use]
    pub const fn viewport(self) -> RenderViewport {
        self.viewport
    }
}

impl Default for RenderIntent {
    fn default() -> Self {
        Self::from_extent(crate::config::RenderAppConfig::DEFAULT.extent)
    }
}
