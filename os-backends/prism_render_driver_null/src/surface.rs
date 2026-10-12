//! The null [`Surface`]: a CPU-side swapchain that mints and releases RHI image
//! ids without ever presenting to a display.
//!
//! The configure → acquire → present cycle is fully implemented against the
//! frozen surface contract: `acquire` mints a texture and a whole-image view on
//! the owning device and hands them out as an ordinary [`rhi::SurfaceImage`],
//! exactly as a real swapchain would; `present` releases them. No window or OS
//! handle is touched, so the cycle runs on any host.

use prism_render_driver as rhi;
use rhi::{
    CompositeAlphaMode, Extent3d, PresentMode, RenderDevice, Surface, SurfaceCapabilities,
    SurfaceConfig, SurfaceError, SurfaceImage, TextureDescriptor, TextureDimension,
    TextureFormat, TextureUsages, TextureViewDescriptor,
};

use crate::device::NullDevice;

/// A null [`rhi::Surface`]: a headless swapchain with no window behind it.
pub struct NullSurface {
    /// The device that owns the swapchain images' ids.
    device: NullDevice,
    /// The formats/present modes/alpha modes this surface accepts.
    caps: SurfaceCapabilities,
    /// The active configuration, or `None` before the first
    /// [`configure`](Surface::configure).
    config: Option<SurfaceConfig>,
}

/// A frame acquired from a [`NullSurface`].
///
/// It carries the RHI image the render graph would import; that image's ids
/// stay live until the frame is presented.
pub struct NullFrame {
    /// The RHI image backing the acquired frame.
    image: SurfaceImage,
}

impl NullSurface {
    /// Creates a null surface minting its images on `device`.
    ///
    /// The reported capabilities are fixed: a surface the null backend can
    /// always satisfy. They include the formats and present mode the design
    /// calls for ([`TextureFormat::Rgba8Unorm`] and [`PresentMode::Fifo`]).
    #[must_use]
    pub(crate) fn new(device: NullDevice) -> Self {
        let caps = SurfaceCapabilities {
            formats: alloc::vec![TextureFormat::Rgba8Unorm, TextureFormat::Bgra8Unorm],
            present_modes: alloc::vec![PresentMode::Fifo, PresentMode::Immediate],
            alpha_modes: alloc::vec![CompositeAlphaMode::Opaque, CompositeAlphaMode::Auto],
        };
        Self {
            device,
            caps,
            config: None,
        }
    }
}

impl Surface for NullSurface {
    type Frame = NullFrame;

    fn capabilities(&self) -> SurfaceCapabilities {
        self.caps.clone()
    }

    fn configure(&mut self, config: &SurfaceConfig) -> Result<(), SurfaceError> {
        if !self.caps.formats.contains(&config.format)
            || !self.caps.present_modes.contains(&config.present_mode)
        {
            return Err(SurfaceError::Unsupported);
        }
        self.config = Some(*config);
        Ok(())
    }

    fn acquire(&self) -> Result<NullFrame, SurfaceError> {
        let config = self.config.ok_or(SurfaceError::Outdated)?;
        let size = Extent3d::new_2d(config.width, config.height);
        let texture = self.device.create_texture(&TextureDescriptor {
            label: Some("null swapchain image".into()),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: config.format,
            usage: TextureUsages::RENDER_ATTACHMENT,
        });
        let view = self
            .device
            .create_texture_view(texture, &TextureViewDescriptor::default());
        Ok(NullFrame {
            image: SurfaceImage {
                texture,
                view,
                format: config.format,
                size,
            },
        })
    }

    fn image(&self, frame: &NullFrame) -> SurfaceImage {
        frame.image
    }

    fn present(&self, frame: NullFrame) -> Result<(), SurfaceError> {
        self.device.destroy_texture_view(frame.image.view);
        self.device.destroy_texture(frame.image.texture);
        Ok(())
    }

    fn size(&self) -> Extent3d {
        match self.config {
            Some(config) => Extent3d::new_2d(config.width, config.height),
            None => Extent3d::new_2d(0, 0),
        }
    }
}
