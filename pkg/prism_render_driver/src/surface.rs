//! Surface: an on-screen presentable target and its acquired frames.
//!
//! A [`Surface`] owns a swapchain and hands out frames the renderer draws into
//! and then presents. It is deliberately backend-owned: the acquired frame is
//! an opaque associated type, and the frame's image is exposed to the render
//! graph as ordinary RHI [`TextureId`]/[`TextureViewId`] handles (registered by
//! the backend at acquisition), so the graph imports a swapchain image exactly
//! like any other external texture.

use alloc::vec::Vec;
use core::fmt;

use crate::format::TextureFormat;
use crate::resource::{TextureId, TextureViewId};
use crate::texture::Extent3d;

/// How a completed frame is presented to the display.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum PresentMode {
    /// Vertical-sync-locked, first-in-first-out: no tearing, capped to refresh.
    #[default]
    Fifo,
    /// Like [`Fifo`](Self::Fifo) but allows late frames to drop v-sync.
    FifoRelaxed,
    /// Present immediately: lowest latency, may tear.
    Immediate,
    /// Replace the queued frame: low latency without tearing.
    Mailbox,
}

/// How the swapchain composites against the window's alpha.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum CompositeAlphaMode {
    /// Let the platform pick the most appropriate mode.
    #[default]
    Auto,
    /// The surface's alpha is ignored (opaque window).
    Opaque,
    /// Color channels are premultiplied by alpha.
    Premultiplied,
    /// Color channels are not premultiplied.
    Postmultiplied,
    /// Inherit the platform's native alpha behavior.
    Inherit,
}

/// A swapchain configuration.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SurfaceConfig {
    /// The texel format the swapchain images are created in.
    pub format: TextureFormat,
    /// How completed frames are presented.
    pub present_mode: PresentMode,
    /// How the swapchain composites against window alpha.
    pub alpha_mode: CompositeAlphaMode,
    /// The current surface width in physical pixels.
    pub width: u32,
    /// The current surface height in physical pixels.
    pub height: u32,
    /// The maximum number of frames the backend may queue ahead (typically `2`).
    pub desired_maximum_frame_latency: u32,
}

impl SurfaceConfig {
    /// A configuration for a `width`×`height` swapchain in `format`, using
    /// v-synced presentation with double buffering.
    #[must_use]
    pub const fn new(format: TextureFormat, width: u32, height: u32) -> Self {
        Self {
            format,
            present_mode: PresentMode::Fifo,
            alpha_mode: CompositeAlphaMode::Auto,
            width,
            height,
            desired_maximum_frame_latency: 2,
        }
    }
}

/// What a surface supports on the adapter it was created against.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct SurfaceCapabilities {
    /// The swapchain formats the surface accepts, best-first.
    pub formats: Vec<TextureFormat>,
    /// The present modes the surface accepts.
    pub present_modes: Vec<PresentMode>,
    /// The composite-alpha modes the surface accepts.
    pub alpha_modes: Vec<CompositeAlphaMode>,
}

/// The RHI image backing an acquired surface frame.
///
/// The ids are valid until the frame is presented; the render graph imports
/// them like any other external texture.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SurfaceImage {
    /// The swapchain image texture.
    pub texture: TextureId,
    /// A whole-image view over [`texture`](Self::texture).
    pub view: TextureViewId,
    /// The image's texel format.
    pub format: TextureFormat,
    /// The image's extent.
    pub size: Extent3d,
}

/// A non-successful surface operation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SurfaceError {
    /// The surface could not be created or configured for this adapter
    /// (no compatible format/present-mode), or creation failed at the platform.
    Unsupported,
    /// Acquiring the next frame timed out; retry later.
    Timeout,
    /// The window is occluded (e.g. minimized); skip this frame.
    Occluded,
    /// The configuration is outdated (e.g. after a resize); reconfigure.
    Outdated,
    /// The surface was lost and must be recreated.
    Lost,
    /// A validation error was raised while acquiring the frame.
    Validation,
}

impl fmt::Display for SurfaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let reason = match self {
            Self::Unsupported => "surface is not supported by the adapter or platform",
            Self::Timeout => "timed out",
            Self::Occluded => "window is occluded",
            Self::Outdated => "surface configuration is outdated",
            Self::Lost => "surface was lost",
            Self::Validation => "surface operation raised a validation error",
        };
        write!(f, "surface error: {reason}")
    }
}

/// A presentable surface and its swapchain.
pub trait Surface {
    /// The backend-owned acquired frame, held from [`acquire`](Surface::acquire)
    /// until [`present`](Surface::present).
    type Frame;

    /// The formats, present modes, and alpha modes this surface supports.
    fn capabilities(&self) -> SurfaceCapabilities;

    /// (Re)configures the swapchain, e.g. at startup or after a resize.
    ///
    /// # Errors
    ///
    /// Returns [`SurfaceError::Unsupported`] when the configuration cannot be
    /// satisfied.
    fn configure(&self, config: &SurfaceConfig) -> Result<(), SurfaceError>;

    /// Acquires the next frame to render into.
    ///
    /// # Errors
    ///
    /// Returns a [`SurfaceError`] describing why no frame was available.
    fn acquire(&self) -> Result<Self::Frame, SurfaceError>;

    /// The RHI image backing `frame`, registered into the surface's device.
    fn image(&self, frame: &Self::Frame) -> SurfaceImage;

    /// Presents `frame`, consuming it.
    ///
    /// # Errors
    ///
    /// Returns a [`SurfaceError`] when presentation fails (e.g. the surface was
    /// lost).
    fn present(&self, frame: Self::Frame) -> Result<(), SurfaceError>;

    /// The current surface extent in physical pixels.
    fn size(&self) -> Extent3d;

    /// A debug label surfaced in GPU tooling, when set.
    fn label(&self) -> Option<&str> {
        None
    }
}
