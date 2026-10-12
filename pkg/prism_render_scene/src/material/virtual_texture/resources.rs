//! Device-owned material-VT atlas, page-table, config, and sampler resources.

use bevy_render::{
    render_resource::{
        AddressMode, Buffer, BufferDescriptor, BufferUsages, Extent3d, Sampler, SamplerDescriptor,
        Texture, TextureDescriptor, TextureDimension, TextureUsages, TextureView,
        TextureViewDescriptor, TextureViewDimension,
    },
    renderer::RenderDevice,
};

use super::{
    abi::{page_table_buffer_size, MaterialVirtualTextureGpuConfig},
    atlas::{MaterialVirtualTextureAtlasConfig, MaterialVirtualTextureAtlasError},
};

/// Byte sizes of the independent material-VT GPU allocations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaterialVirtualTextureResourceSizes {
    /// Dedicated storage page-table buffer size.
    pub page_table_bytes: u64,
    /// Number of entries that fit in the dedicated page-table buffer.
    pub page_table_entry_capacity: u32,
    /// Config uniform/storage buffer size.
    pub config_bytes: u64,
    /// One atlas layer's width/height in texels.
    pub atlas_extent_px: u32,
    /// Number of atlas array layers.
    pub atlas_layers: u32,
}

/// The real Bevy/wgpu resources owned by one material-VT slice.
///
/// No shade-cache buffer, texture, view, or sampler is accepted here. The
/// page table is a dedicated `STORAGE | COPY_DST` allocation and the atlas is
/// a dedicated `D2Array | TEXTURE_BINDING | COPY_DST` allocation.
pub struct MaterialVirtualTextureGpuBundle {
    atlas_texture: Texture,
    atlas_view: TextureView,
    page_table_buffer: Buffer,
    config_buffer: Buffer,
    sampler: Sampler,
    atlas_config: MaterialVirtualTextureAtlasConfig,
    sizes: MaterialVirtualTextureResourceSizes,
}

impl MaterialVirtualTextureGpuBundle {
    /// Creates all material-VT GPU resources from a validated atlas config.
    pub fn new(
        device: &RenderDevice,
        atlas_config: MaterialVirtualTextureAtlasConfig,
        page_table_entry_capacity: u32,
    ) -> Result<Self, MaterialVirtualTextureAtlasError> {
        let atlas_extent_px = atlas_config.atlas_extent_px()?;
        let atlas_layers = atlas_config.atlas_layers()?;
        let sizes = MaterialVirtualTextureResourceSizes {
            page_table_bytes: page_table_buffer_size(page_table_entry_capacity),
            page_table_entry_capacity,
            config_bytes: MaterialVirtualTextureGpuConfig::byte_size(),
            atlas_extent_px,
            atlas_layers,
        };
        let atlas_texture = device.create_texture(&TextureDescriptor {
            label: Some("prism material VT atlas"),
            size: Extent3d {
                width: atlas_extent_px,
                height: atlas_extent_px,
                depth_or_array_layers: atlas_layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: atlas_config.texture_format(),
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let atlas_view = atlas_texture.create_view(&TextureViewDescriptor {
            label: Some("prism material VT atlas view"),
            format: Some(atlas_config.texture_format()),
            dimension: Some(TextureViewDimension::D2Array),
            aspect: bevy_render::render_resource::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(atlas_layers),
            ..Default::default()
        });
        let page_table_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("prism material VT page table"),
            size: sizes.page_table_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let config_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("prism material VT config"),
            size: sizes.config_bytes,
            usage: BufferUsages::UNIFORM | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (mag_filter, min_filter, mipmap_filter) = atlas_config.sampler_filter();
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("prism material VT sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter,
            min_filter,
            mipmap_filter,
            ..Default::default()
        });
        Ok(Self {
            atlas_texture,
            atlas_view,
            page_table_buffer,
            config_buffer,
            sampler,
            atlas_config,
            sizes,
        })
    }

    /// Atlas texture owner.
    #[must_use]
    pub fn atlas_texture(&self) -> &Texture {
        &self.atlas_texture
    }

    /// D2-array sampling view for the atlas.
    #[must_use]
    pub fn atlas_view(&self) -> &TextureView {
        &self.atlas_view
    }

    /// Dedicated material-VT page-table storage buffer.
    #[must_use]
    pub fn page_table_buffer(&self) -> &Buffer {
        &self.page_table_buffer
    }

    /// Config uniform/storage buffer.
    #[must_use]
    pub fn config_buffer(&self) -> &Buffer {
        &self.config_buffer
    }

    /// Atlas sampler.
    #[must_use]
    pub fn sampler(&self) -> &Sampler {
        &self.sampler
    }

    /// Immutable atlas configuration.
    #[must_use]
    pub const fn atlas_config(&self) -> MaterialVirtualTextureAtlasConfig {
        self.atlas_config
    }

    /// Allocation sizes and geometry.
    #[must_use]
    pub const fn sizes(&self) -> MaterialVirtualTextureResourceSizes {
        self.sizes
    }
}
