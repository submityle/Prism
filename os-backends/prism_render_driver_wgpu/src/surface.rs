//! wgpu window-surface support (behind the `surface` feature).
//!
//! [`WgpuSurface`] owns a `wgpu::Surface` for the lifetime of its window
//! (the window is held as an `Arc`, so the surface is `'static`) and implements
//! the frozen [`rhi::Surface`] contract: configure → acquire → image →
//! present. Acquired swapchain images are registered into the shared device
//! slot maps and exposed as ordinary RHI texture/view ids, so the render graph
//! imports a swapchain image exactly like any other external texture.
//!
//! Window handles arrive through `raw-window-handle` (wgpu's safe
//! `create_surface` path), so no `unsafe` is required.

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;

use prism_render_driver as rhi;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use rhi::Surface;

use crate::convert;
use crate::device::DeviceInner;
use crate::instance::WgpuInstance;

/// A presentable window surface and its swapchain.
pub struct WgpuSurface {
    /// The underlying wgpu surface, owning its window for `'static`.
    surface: wgpu::Surface<'static>,
    /// The device whose slot maps external swapchain images are registered in.
    inner: Arc<DeviceInner>,
    /// The queue frames are presented on.
    queue: wgpu::Queue,
    /// A debug label surfaced in GPU tooling.
    label: Option<String>,
    /// The current swapchain extent (updated on every `configure`).
    size: rhi::Extent3d,
    /// The swapchain format frames are acquired in, once configured.
    format: Option<wgpu::TextureFormat>,
}

/// An error raised while creating a [`WgpuSurface`].
#[derive(Debug)]
pub struct SurfaceCreateError(pub wgpu::CreateSurfaceError);

impl fmt::Display for SurfaceCreateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "failed to create wgpu surface: {}", self.0)
    }
}

impl std::error::Error for SurfaceCreateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

/// An acquired swapchain frame.
///
/// Hold it while recording and submitting the frame's commands, then pass it to
/// [`Surface::present`] (or drop it to discard the frame). Dropping without
/// presenting leaks the registered image ids for the device's lifetime, so
/// callers should always present or discard explicitly.
pub struct WgpuFrame {
    /// The acquired surface texture, consumed on present.
    texture: wgpu::SurfaceTexture,
    /// The RHI id the frame's texture was registered under.
    texture_id: rhi::TextureId,
    /// The RHI id the frame's default view was registered under.
    view_id: rhi::TextureViewId,
    /// The frame's texel format.
    format: rhi::TextureFormat,
    /// The frame's extent.
    size: rhi::Extent3d,
    /// Whether the surface reported the frame as suboptimal.
    suboptimal: bool,
}

impl WgpuFrame {
    /// Whether the frame was acquired but no longer matches the surface's
    /// properties; the surface should be reconfigured for optimal performance.
    #[must_use]
    pub fn is_suboptimal(&self) -> bool {
        self.suboptimal
    }
}

impl WgpuSurface {
    /// Creates a surface targeting `window`, registered against `inner`/`queue`.
    pub(crate) fn create<W>(
        instance: &WgpuInstance,
        window: Arc<W>,
        inner: Arc<DeviceInner>,
        queue: wgpu::Queue,
        label: Option<String>,
    ) -> Result<Self, SurfaceCreateError>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        let surface = instance
            .wgpu_instance()
            .create_surface(window)
            .map_err(SurfaceCreateError)?;
        Ok(Self {
            surface,
            inner,
            queue,
            label,
            size: rhi::Extent3d::new_2d(1, 1),
            format: None,
        })
    }

    /// Borrows the underlying wgpu surface.
    #[must_use]
    pub fn wgpu_surface(&self) -> &wgpu::Surface<'static> {
        &self.surface
    }

    /// Releases the RHI ids registered for `frame`.
    ///
    /// Shared by [`Surface::present`] and [`Surface::discard`], which differ
    /// only in whether the platform frame is presented or dropped.
    fn release_frame(&self, frame: &WgpuFrame) {
        self.inner.release_texture_view(frame.view_id);
        self.inner.release_texture(frame.texture_id);
    }
}

impl Surface for WgpuSurface {
    type Frame = WgpuFrame;

    fn capabilities(&self) -> rhi::SurfaceCapabilities {
        let caps = self.surface.get_capabilities(&self.inner.adapter);
        rhi::SurfaceCapabilities {
            formats: caps
                .formats
                .iter()
                .filter_map(|f| convert::surface_format_from_wgpu(*f))
                .collect(),
            present_modes: caps
                .present_modes
                .iter()
                .filter_map(|m| convert::present_mode_from_wgpu(*m))
                .collect(),
            alpha_modes: caps
                .alpha_modes
                .iter()
                .map(|m| convert::alpha_mode_from_wgpu(*m))
                .collect(),
        }
    }

    fn configure(&mut self, config: &rhi::SurfaceConfig) -> Result<(), rhi::SurfaceError> {
        let wgpu_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: convert::texture_format(config.format),
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: config.width.max(1),
            height: config.height.max(1),
            present_mode: convert::present_mode_to_wgpu(config.present_mode),
            desired_maximum_frame_latency: config.desired_maximum_frame_latency.max(1),
            alpha_mode: convert::alpha_mode_to_wgpu(config.alpha_mode),
            view_formats: Vec::new(),
        };
        // wgpu panics if the chosen present mode is unsupported; validate it
        // against the surface's advertised set first and report cleanly.
        let caps = self.surface.get_capabilities(&self.inner.adapter);
        if !caps.present_modes.contains(&wgpu_config.present_mode) {
            return Err(rhi::SurfaceError::Unsupported);
        }
        self.surface.configure(&self.inner.device, &wgpu_config);
        self.format = Some(wgpu_config.format);
        self.size = rhi::Extent3d::new_2d(wgpu_config.width, wgpu_config.height);
        Ok(())
    }

    fn acquire(&self) -> Result<Self::Frame, rhi::SurfaceError> {
        let format = self.format.ok_or(rhi::SurfaceError::Unsupported)?;
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => (texture, false),
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => (texture, true),
            wgpu::CurrentSurfaceTexture::Timeout => return Err(rhi::SurfaceError::Timeout),
            wgpu::CurrentSurfaceTexture::Occluded => return Err(rhi::SurfaceError::Occluded),
            wgpu::CurrentSurfaceTexture::Outdated => return Err(rhi::SurfaceError::Outdated),
            wgpu::CurrentSurfaceTexture::Lost => return Err(rhi::SurfaceError::Lost),
            wgpu::CurrentSurfaceTexture::Validation => return Err(rhi::SurfaceError::Validation),
        };
        let (texture, suboptimal) = surface_texture;
        let view = texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let texture_id = self.inner.register_texture(texture.texture.clone());
        let view_id = self.inner.register_texture_view(view);
        Ok(WgpuFrame {
            texture,
            texture_id,
            view_id,
            format: convert::surface_format_from_wgpu(format)
                .ok_or(rhi::SurfaceError::Unsupported)?,
            size: self.size,
            suboptimal,
        })
    }

    fn image(&self, frame: &Self::Frame) -> rhi::SurfaceImage {
        rhi::SurfaceImage {
            texture: frame.texture_id,
            view: frame.view_id,
            format: frame.format,
            size: frame.size,
        }
    }

    fn discard(&self, frame: Self::Frame) {
        // Release the registered image ids; dropping the `SurfaceTexture`
        // without presenting discards the frame at the platform level.
        self.release_frame(&frame);
    }

    fn present(&self, frame: Self::Frame) -> Result<(), rhi::SurfaceError> {
        self.release_frame(&frame);
        self.queue.present(frame.texture);
        Ok(())
    }

    fn size(&self) -> rhi::Extent3d {
        self.size
    }

    fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}
