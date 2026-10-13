//! Owned storage texture contract for the shade-cache atlas.

use bevy_render::{
    render_resource::{
        Extent3d, Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
        TextureView, TextureViewDescriptor, TextureViewDimension,
    },
    renderer::RenderDevice,
};

/// Format written by `gpu/shade.wgsl`.
pub const SHADE_ATLAS_FORMAT: TextureFormat = TextureFormat::Rgba32Float;

/// Failure to construct a valid shade atlas.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShadeAtlasError {
    /// Storage textures cannot have a zero width or height.
    ZeroExtent,
}

/// A 2D storage atlas that owns its texture and default view.
#[derive(Debug)]
pub struct ShadeAtlas {
    texture: Texture,
    view: TextureView,
    extent: [u32; 2],
}

impl ShadeAtlas {
    /// Allocates an atlas suitable for the shade writer.
    pub fn new(device: &RenderDevice, extent: [u32; 2]) -> Result<Self, ShadeAtlasError> {
        if extent.contains(&0) {
            return Err(ShadeAtlasError::ZeroExtent);
        }
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("prism shade-cache atlas"),
            size: Extent3d {
                width: extent[0],
                height: extent[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: SHADE_ATLAS_FORMAT,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::COPY_SRC
                | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor {
            dimension: Some(TextureViewDimension::D2),
            ..Default::default()
        });
        Ok(Self {
            texture,
            view,
            extent,
        })
    }

    /// Atlas extent in texels.
    #[must_use]
    pub const fn extent(&self) -> [u32; 2] {
        self.extent
    }

    /// Storage view bound at group 0 binding 0.
    #[must_use]
    pub fn view(&self) -> &TextureView {
        &self.view
    }

    /// Underlying texture, useful for clear/copy operations owned by a graph.
    #[must_use]
    pub fn texture(&self) -> &Texture {
        &self.texture
    }
}
