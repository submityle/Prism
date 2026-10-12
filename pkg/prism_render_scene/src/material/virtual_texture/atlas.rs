//! Material-VT atlas geometry and format validation.

use prism_render_architecture::texture_streaming::{
    AtlasGeometry, AtlasPageLayout, AtlasTileFormat, TexturePageKey,
};
use prism_render_material::{
    MaterialVtAtlasConfig, MaterialVtAtlasTile, MaterialVtFormat, MaterialVtSourceDescriptor,
};
use wgpu_types::{FilterMode, MipmapFilterMode, TextureFormat};

/// Failure while constructing a material-VT GPU atlas contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaterialVirtualTextureAtlasError {
    /// A required dimension or capacity was zero.
    ZeroDimension,
    /// The requested geometry overflowed or was not copy-valid.
    InvalidGeometry,
    /// A source tile does not match this atlas's layout.
    TileMismatch,
    /// A tile's physical slot is outside the configured capacity.
    SlotOutOfRange,
}

/// Device-independent configuration shared by one material-VT atlas.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaterialVirtualTextureAtlasConfig {
    page_size: u32,
    gutter_px: u32,
    tiles_per_row: u32,
    capacity: u32,
    format: MaterialVtFormat,
}

impl MaterialVirtualTextureAtlasConfig {
    /// Creates a validated square tile atlas configuration.
    pub fn new(
        page_size: u32,
        gutter_px: u32,
        tiles_per_row: u32,
        capacity: u32,
        format: MaterialVtFormat,
    ) -> Result<Self, MaterialVirtualTextureAtlasError> {
        if page_size == 0 || tiles_per_row == 0 || capacity == 0 {
            return Err(MaterialVirtualTextureAtlasError::ZeroDimension);
        }
        let config = Self {
            page_size,
            gutter_px,
            tiles_per_row,
            capacity,
            format,
        };
        config.geometry()?;
        Ok(config)
    }

    /// Creates atlas configuration from an immutable source descriptor.
    pub fn from_source(
        source: MaterialVtSourceDescriptor,
        gutter_px: u32,
        tiles_per_row: u32,
        capacity: u32,
    ) -> Result<Self, MaterialVirtualTextureAtlasError> {
        Self::new(
            source.page_size(),
            gutter_px,
            tiles_per_row,
            capacity,
            source.format(),
        )
    }

    /// Logical page edge.
    #[must_use]
    pub const fn page_size(self) -> u32 {
        self.page_size
    }

    /// Per-edge gutter.
    #[must_use]
    pub const fn gutter_px(self) -> u32 {
        self.gutter_px
    }

    /// Stored tiles per atlas-layer edge.
    #[must_use]
    pub const fn tiles_per_row(self) -> u32 {
        self.tiles_per_row
    }

    /// Maximum physical slot accepted by this atlas plus one.
    #[must_use]
    pub const fn capacity(self) -> u32 {
        self.capacity
    }

    /// Source/sample format.
    #[must_use]
    pub const fn format(self) -> MaterialVtFormat {
        self.format
    }

    /// Validated page layout used by the source tile builder.
    pub fn page_layout(self) -> Result<AtlasPageLayout, MaterialVirtualTextureAtlasError> {
        AtlasPageLayout::new(self.page_size, self.gutter_px, self.tile_format())
            .ok_or(MaterialVirtualTextureAtlasError::InvalidGeometry)
    }

    /// Physical atlas geometry used to map slots to array layers and origins.
    pub fn geometry(self) -> Result<AtlasGeometry, MaterialVirtualTextureAtlasError> {
        let layout = self.page_layout()?;
        AtlasGeometry::for_page_layout(self.tiles_per_row, layout)
            .ok_or(MaterialVirtualTextureAtlasError::InvalidGeometry)
    }

    /// Number of texture array layers required by the configured capacity.
    pub fn atlas_layers(self) -> Result<u32, MaterialVirtualTextureAtlasError> {
        Ok(self.geometry()?.layers_for_capacity(self.capacity).max(1))
    }

    /// Full width/height of one atlas layer in texels.
    pub fn atlas_extent_px(self) -> Result<u32, MaterialVirtualTextureAtlasError> {
        self.geometry()?
            .tile_extent_px()
            .checked_mul(self.tiles_per_row)
            .ok_or(MaterialVirtualTextureAtlasError::InvalidGeometry)
    }

    /// Converts the source format to the corresponding wgpu texture format.
    #[must_use]
    pub const fn texture_format(self) -> TextureFormat {
        match self.format {
            MaterialVtFormat::R8Unorm => TextureFormat::R8Unorm,
            MaterialVtFormat::Rg8Unorm => TextureFormat::Rg8Unorm,
            MaterialVtFormat::Rgba8Unorm => TextureFormat::Rgba8Unorm,
            MaterialVtFormat::Rgba8Srgb => TextureFormat::Rgba8UnormSrgb,
            MaterialVtFormat::Rgba16Float => TextureFormat::Rgba16Float,
            MaterialVtFormat::Rgba32Float => TextureFormat::Rgba32Float,
        }
    }

    /// Chooses a sampler mode valid for all of the supported source formats.
    ///
    /// Float formats are sampled nearest because many adapters expose them as
    /// non-filterable; normalized byte formats retain linear filtering for
    /// gutter-aware page seams.
    #[must_use]
    pub const fn sampler_filter(self) -> (FilterMode, FilterMode, MipmapFilterMode) {
        match self.format {
            MaterialVtFormat::Rgba16Float | MaterialVtFormat::Rgba32Float => (
                FilterMode::Nearest,
                FilterMode::Nearest,
                MipmapFilterMode::Nearest,
            ),
            _ => (
                FilterMode::Linear,
                FilterMode::Linear,
                MipmapFilterMode::Nearest,
            ),
        }
    }

    /// Validates a tile before it is sent to the GPU queue.
    pub fn validate_tile(
        self,
        tile: &MaterialVtAtlasTile,
    ) -> Result<(), MaterialVirtualTextureAtlasError> {
        if tile.slot >= self.capacity {
            return Err(MaterialVirtualTextureAtlasError::SlotOutOfRange);
        }
        let geometry = self.geometry()?;
        let layout = self.page_layout()?;
        if tile.layout != layout
            || tile.bytes_per_row != geometry.bytes_per_row()
            || tile.staging_bytes.len()
                != usize::try_from(geometry.tile_staging_bytes()).unwrap_or(usize::MAX)
        {
            return Err(MaterialVirtualTextureAtlasError::TileMismatch);
        }
        Ok(())
    }

    /// Builds the source-side atlas config used by the existing tile builder.
    pub fn source_tile_config(
        self,
    ) -> Result<MaterialVtAtlasConfig, prism_render_material::MaterialVtAtlasUploadError> {
        MaterialVtAtlasConfig::new(self.gutter_px, self.tiles_per_row)
    }

    /// Returns the physical destination for a tile key/slot pair.
    pub fn slot_placement(
        self,
        key: TexturePageKey,
        slot: u32,
    ) -> Result<
        (
            TexturePageKey,
            prism_render_architecture::texture_streaming::SlotPlacement,
        ),
        MaterialVirtualTextureAtlasError,
    > {
        if slot >= self.capacity {
            return Err(MaterialVirtualTextureAtlasError::SlotOutOfRange);
        }
        Ok((key, self.geometry()?.slot_placement(slot)))
    }

    const fn tile_format(self) -> AtlasTileFormat {
        AtlasTileFormat {
            block_extent_px: 1,
            bytes_per_block: self.format.bytes_per_texel(),
        }
    }
}
