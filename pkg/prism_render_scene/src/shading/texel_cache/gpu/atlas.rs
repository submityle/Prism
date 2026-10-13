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

/// Failure while deriving a deterministic physical slot layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShadeAtlasLayoutError {
    /// The atlas or one tile axis is zero.
    ZeroExtent,
    /// A tile does not fit in the atlas.
    TileDoesNotFit,
    /// The slot count or tile-origin arithmetic overflowed.
    ArithmeticOverflow,
}

/// CPU-side slot-to-origin contract shared by shade allocation and page-table
/// construction. Slots are row-major, never wrap across an atlas row, and do
/// not overlap. This type owns no GPU resource.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShadeAtlasLayout {
    atlas_extent: [u32; 2],
    tile_extent: [u32; 2],
    columns: u32,
    rows: u32,
    capacity: u32,
}

impl ShadeAtlasLayout {
    /// Creates a fixed-tile row-major layout for a shade atlas.
    pub fn new(
        atlas_extent: [u32; 2],
        tile_extent: [u32; 2],
    ) -> Result<Self, ShadeAtlasLayoutError> {
        if atlas_extent.contains(&0) || tile_extent.contains(&0) {
            return Err(ShadeAtlasLayoutError::ZeroExtent);
        }
        if tile_extent[0] > atlas_extent[0] || tile_extent[1] > atlas_extent[1] {
            return Err(ShadeAtlasLayoutError::TileDoesNotFit);
        }
        let columns = atlas_extent[0] / tile_extent[0];
        let rows = atlas_extent[1] / tile_extent[1];
        let capacity = columns
            .checked_mul(rows)
            .ok_or(ShadeAtlasLayoutError::ArithmeticOverflow)?;
        if capacity == 0 {
            return Err(ShadeAtlasLayoutError::TileDoesNotFit);
        }
        Ok(Self {
            atlas_extent,
            tile_extent,
            columns,
            rows,
            capacity,
        })
    }

    /// Atlas extent in texels.
    #[must_use]
    pub const fn atlas_extent(self) -> [u32; 2] {
        self.atlas_extent
    }

    /// Fixed page tile extent in texels.
    #[must_use]
    pub const fn tile_extent(self) -> [u32; 2] {
        self.tile_extent
    }

    /// Number of physical slots addressable by this layout.
    #[must_use]
    pub const fn capacity(self) -> u32 {
        self.capacity
    }

    /// Returns the non-overlapping atlas origin for a physical slot.
    #[must_use]
    pub fn origin(self, slot: u32) -> Option<[u32; 2]> {
        if slot >= self.capacity {
            return None;
        }
        let x = (slot % self.columns).checked_mul(self.tile_extent[0])?;
        let y = (slot / self.columns).checked_mul(self.tile_extent[1])?;
        Some([x, y])
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_major_layout_maps_slots_without_overlap() {
        let layout = ShadeAtlasLayout::new([16, 8], [4, 4]).unwrap();
        assert_eq!(layout.capacity(), 8);
        assert_eq!(layout.origin(0), Some([0, 0]));
        assert_eq!(layout.origin(3), Some([12, 0]));
        assert_eq!(layout.origin(4), Some([0, 4]));
        assert_eq!(layout.origin(7), Some([12, 4]));
        assert_eq!(layout.origin(8), None);
    }

    #[test]
    fn layout_rejects_zero_and_non_fitting_tiles() {
        assert_eq!(
            ShadeAtlasLayout::new([0, 8], [4, 4]),
            Err(ShadeAtlasLayoutError::ZeroExtent)
        );
        assert_eq!(
            ShadeAtlasLayout::new([8, 8], [16, 4]),
            Err(ShadeAtlasLayoutError::TileDoesNotFit)
        );
    }
}
