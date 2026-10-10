//! Optional window-surface/swapchain support (behind the `surface` feature).
//!
//! Wraps `wgpu::Surface` with a complete configure → acquire → present cycle.
//! Window handles are accepted through `raw-window-handle` (wgpu's safe
//! `create_surface` path), so no `unsafe` is required.

use core::fmt;

use crate::device::WgpuDevice;
use crate::instance::WgpuInstance;
use crate::queue::WgpuQueue;

/// A presentable window surface and its swapchain.
pub struct WgpuSurface<'window> {
    /// The underlying wgpu surface.
    surface: wgpu::Surface<'window>,
}

/// An error raised while configuring a [`WgpuSurface`].
#[derive(Debug)]
pub enum SurfaceConfigureError {
    /// The adapter exposes no format/present-mode combination compatible with
    /// the surface, so a default configuration could not be derived.
    Unsupported,
}

impl fmt::Display for SurfaceConfigureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported => f.write_str("surface is not compatible with the adapter"),
        }
    }
}

impl std::error::Error for SurfaceConfigureError {}

/// A non-success outcome of [`WgpuSurface::acquire`].
///
/// Each variant mirrors a `wgpu::CurrentSurfaceTexture` status that did not
/// yield a usable frame; callers typically skip the frame or reconfigure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceAcquireError {
    /// Acquiring the next frame timed out; try again later.
    Timeout,
    /// The window is occluded (e.g. minimized); skip this frame.
    Occluded,
    /// The surface configuration is outdated; reconfigure and retry.
    Outdated,
    /// The surface was lost and must be recreated.
    Lost,
    /// A validation error was raised while acquiring the frame.
    Validation,
}

impl fmt::Display for SurfaceAcquireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let reason = match self {
            Self::Timeout => "timed out",
            Self::Occluded => "window is occluded",
            Self::Outdated => "surface configuration is outdated",
            Self::Lost => "surface was lost",
            Self::Validation => "surface acquisition raised a validation error",
        };
        write!(f, "failed to acquire surface frame: {reason}")
    }
}

impl std::error::Error for SurfaceAcquireError {}

/// An acquired swapchain frame.
///
/// Hold it while recording and submitting the frame's commands, then call
/// [`WgpuFrame::present`] (or drop it to discard the frame).
pub struct WgpuFrame {
    /// The acquired surface texture, consumed on present.
    texture: wgpu::SurfaceTexture,
    /// Whether the surface reported the frame as suboptimal.
    suboptimal: bool,
}

impl WgpuFrame {
    /// Borrows the frame's backing texture, e.g. to create a render target
    /// view.
    #[must_use]
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture.texture
    }

    /// Creates a default view over the frame's texture for use as a color
    /// attachment.
    #[must_use]
    pub fn create_view(&self) -> wgpu::TextureView {
        self.texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Whether the frame was acquired but no longer matches the surface's
    /// properties; the surface should be reconfigured for optimal performance.
    #[must_use]
    pub fn is_suboptimal(&self) -> bool {
        self.suboptimal
    }

    /// Presents the frame on `queue`, consuming it.
    pub fn present(self, queue: &WgpuQueue) {
        queue.wgpu_queue().present(self.texture);
    }
}

impl<'window> WgpuSurface<'window> {
    /// Creates a surface targeting `window` (any `raw-window-handle` target).
    ///
    /// # Errors
    ///
    /// Returns the wgpu error if the platform surface could not be created.
    pub fn create(
        instance: &WgpuInstance,
        window: impl Into<wgpu::SurfaceTarget<'window>>,
    ) -> Result<Self, wgpu::CreateSurfaceError> {
        let surface = instance.wgpu_instance().create_surface(window)?;
        Ok(Self { surface })
    }

    /// Borrows the underlying wgpu surface.
    #[must_use]
    pub fn wgpu_surface(&self) -> &wgpu::Surface<'window> {
        &self.surface
    }

    /// Configures (or reconfigures) the swapchain for `width`×`height` using
    /// the adapter's preferred format and present mode.
    ///
    /// # Errors
    ///
    /// Returns [`SurfaceConfigureError::Unsupported`] if the adapter exposes no
    /// compatible configuration for this surface.
    pub fn configure(
        &self,
        device: &WgpuDevice,
        width: u32,
        height: u32,
    ) -> Result<(), SurfaceConfigureError> {
        let config = self
            .surface
            .get_default_config(device.wgpu_adapter(), width, height)
            .ok_or(SurfaceConfigureError::Unsupported)?;
        self.surface.configure(device.wgpu_device(), &config);
        Ok(())
    }

    /// Acquires the next swapchain frame to render into.
    ///
    /// # Errors
    ///
    /// Returns a [`SurfaceAcquireError`] if no usable frame was available
    /// (timeout, occlusion, an outdated/lost surface, or a validation error).
    pub fn acquire(&self) -> Result<WgpuFrame, SurfaceAcquireError> {
        match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => Ok(WgpuFrame {
                texture,
                suboptimal: false,
            }),
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => Ok(WgpuFrame {
                texture,
                suboptimal: true,
            }),
            wgpu::CurrentSurfaceTexture::Timeout => Err(SurfaceAcquireError::Timeout),
            wgpu::CurrentSurfaceTexture::Occluded => Err(SurfaceAcquireError::Occluded),
            wgpu::CurrentSurfaceTexture::Outdated => Err(SurfaceAcquireError::Outdated),
            wgpu::CurrentSurfaceTexture::Lost => Err(SurfaceAcquireError::Lost),
            wgpu::CurrentSurfaceTexture::Validation => Err(SurfaceAcquireError::Validation),
        }
    }
}
