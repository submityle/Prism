//! CPU mirror of the material virtual-texture resource ABI.

use bytemuck::{Pod, Zeroable};
use core::mem::size_of;

/// Number of `u32` words occupied by one material-VT page-table entry.
pub const MATERIAL_VIRTUAL_TEXTURE_ENTRY_WORDS: usize = 4;
/// Byte stride of one material-VT page-table entry.
pub const MATERIAL_VIRTUAL_TEXTURE_ENTRY_STRIDE: u64 =
    (MATERIAL_VIRTUAL_TEXTURE_ENTRY_WORDS * size_of::<u32>()) as u64;

/// Configuration consumed by `material_virtual_texture.wesl`.
///
/// This is intentionally a plain 32-byte block so it can be bound either as a
/// uniform buffer or as a read-only storage buffer. The field order mirrors
/// the shader struct exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Pod, Zeroable)]
pub struct MaterialVirtualTextureGpuConfig {
    /// Nonzero while material virtual texturing is enabled.
    pub enabled: u32,
    /// Number of valid entries in the dedicated page-table buffer.
    pub entry_count: u32,
    /// Logical virtual-page edge in texels.
    pub page_size: u32,
    /// Stored atlas tile edge, including both gutters.
    pub tile_extent_px: u32,
    /// Number of stored tiles along one atlas layer edge.
    pub tiles_per_row: u32,
    /// Number of array layers allocated for the atlas.
    pub atlas_layers: u32,
    /// Per-edge gutter in texels.
    pub gutter_px: u32,
    /// Last mip the resolver may search.
    pub coarsest_mip: u32,
}

impl MaterialVirtualTextureGpuConfig {
    /// Builds a config from validated atlas geometry and page-table count.
    #[must_use]
    pub const fn new(
        enabled: bool,
        entry_count: u32,
        page_size: u32,
        tile_extent_px: u32,
        tiles_per_row: u32,
        atlas_layers: u32,
        gutter_px: u32,
        coarsest_mip: u32,
    ) -> Self {
        Self {
            enabled: enabled as u32,
            entry_count,
            page_size,
            tile_extent_px,
            tiles_per_row,
            atlas_layers,
            gutter_px,
            coarsest_mip,
        }
    }

    /// Returns the stable byte layout expected by a bind-group declaration.
    #[must_use]
    pub const fn byte_size() -> u64 {
        size_of::<Self>() as u64
    }
}

/// Exact page-table storage-buffer size for `entry_count` entries.
///
/// WebGPU disallows zero-sized buffers, so an empty logical table still gets
/// one 16-byte sentinel-sized allocation. The shader receives `entry_count=0`
/// and therefore never reads that padding.
#[must_use]
pub const fn page_table_buffer_size(entry_count: u32) -> u64 {
    let bytes = (entry_count as u64) * MATERIAL_VIRTUAL_TEXTURE_ENTRY_STRIDE;
    if bytes < MATERIAL_VIRTUAL_TEXTURE_ENTRY_STRIDE {
        MATERIAL_VIRTUAL_TEXTURE_ENTRY_STRIDE
    } else {
        bytes
    }
}
