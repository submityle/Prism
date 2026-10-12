//! Material-VT source pages to physical-atlas staging tiles.
//!
//! [`super::MaterialVtSourceSnapshot`] exposes tight source rows. A physical
//! atlas needs a fixed logical page, a filtering gutter, and a copy-aligned
//! row pitch. This module is the device-free producer at that boundary. It
//! does not allocate page-table slots or publish residency.

use alloc::{collections::BTreeSet, vec::Vec};
use core::fmt;

use prism_render_architecture::texture_streaming::{
    AtlasGeometry, AtlasPageLayout, AtlasTileFormat, TexturePageKey,
};

use super::MaterialVtSourceSnapshot;

/// Parameters shared by all tiles in one material-VT atlas.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaterialVtAtlasConfig {
    gutter_px: u32,
    tiles_per_row: u32,
}

impl MaterialVtAtlasConfig {
    /// Creates atlas parameters. `gutter_px` is per edge and may be zero.
    pub const fn new(
        gutter_px: u32,
        tiles_per_row: u32,
    ) -> Result<Self, MaterialVtAtlasUploadError> {
        if tiles_per_row == 0 {
            return Err(MaterialVtAtlasUploadError::ZeroTilesPerRow);
        }
        Ok(Self {
            gutter_px,
            tiles_per_row,
        })
    }

    /// Per-edge filtering gutter in texels.
    #[must_use]
    pub const fn gutter_px(self) -> u32 {
        self.gutter_px
    }

    /// Number of physical tiles along one atlas layer edge.
    #[must_use]
    pub const fn tiles_per_row(self) -> u32 {
        self.tiles_per_row
    }
}

/// One complete, fixed-size material-VT tile ready for staging.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialVtAtlasTile {
    /// Virtual page represented by this tile.
    pub key: TexturePageKey,
    /// Physical atlas slot assigned by the streaming pool.
    pub slot: u32,
    /// Logical/stored tile geometry used to produce `staging_bytes`.
    pub layout: AtlasPageLayout,
    /// Padded source row pitch required by the copy API.
    pub bytes_per_row: u32,
    /// Stored tile bytes, row-major, including gutter and row padding.
    pub staging_bytes: Vec<u8>,
}

impl MaterialVtAtlasTile {
    /// Number of bytes in the stored tile, including row padding.
    #[must_use]
    pub fn len(&self) -> usize {
        self.staging_bytes.len()
    }

    /// Whether the tile has no staging bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.staging_bytes.is_empty()
    }
}

/// Failure while converting a source page into an atlas staging tile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaterialVtAtlasUploadError {
    /// The atlas cannot have zero tiles along an edge.
    ZeroTilesPerRow,
    /// The source page dimensions or format cannot form a valid atlas layout.
    InvalidLayout,
    /// A source page could not be addressed or extracted.
    SourcePage,
    /// A derived byte count does not fit the host address space.
    SizeOverflow,
    /// The host could not reserve the requested staging memory.
    AllocationFailed,
}

impl fmt::Display for MaterialVtAtlasUploadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroTilesPerRow => {
                formatter.write_str("material VT atlas tiles_per_row must be nonzero")
            }
            Self::InvalidLayout => formatter.write_str("material VT atlas layout is invalid"),
            Self::SourcePage => formatter.write_str("material VT source page cannot be extracted"),
            Self::SizeOverflow => formatter.write_str("material VT atlas staging size overflows"),
            Self::AllocationFailed => {
                formatter.write_str("material VT atlas staging allocation failed")
            }
        }
    }
}

impl std::error::Error for MaterialVtAtlasUploadError {}

/// Builds one material-VT atlas tile from an immutable source snapshot.
///
/// `resident_neighbors` controls which adjacent source pages may contribute
/// actual texels to the gutter. Missing neighbors are extended from the
/// center page and must be rebuilt when they become resident.
pub fn build_material_vt_atlas_tile(
    source: &MaterialVtSourceSnapshot,
    key: TexturePageKey,
    slot: u32,
    resident_neighbors: &BTreeSet<TexturePageKey>,
    config: MaterialVtAtlasConfig,
) -> Result<MaterialVtAtlasTile, MaterialVtAtlasUploadError> {
    let bpp = usize::try_from(source.descriptor().format().bytes_per_texel())
        .map_err(|_| MaterialVtAtlasUploadError::SizeOverflow)?;
    let logical_extent = usize::try_from(source.descriptor().page_size())
        .map_err(|_| MaterialVtAtlasUploadError::SizeOverflow)?;
    let logical_texels = logical_extent
        .checked_mul(logical_extent)
        .and_then(|value| value.checked_mul(bpp))
        .ok_or(MaterialVtAtlasUploadError::SizeOverflow)?;
    let format = AtlasTileFormat {
        block_extent_px: 1,
        bytes_per_block: u32::try_from(bpp)
            .map_err(|_| MaterialVtAtlasUploadError::SizeOverflow)?,
    };
    let layout = AtlasPageLayout::new(source.descriptor().page_size(), config.gutter_px, format)
        .ok_or(MaterialVtAtlasUploadError::InvalidLayout)?;
    let geometry = AtlasGeometry::for_page_layout(config.tiles_per_row, layout)
        .ok_or(MaterialVtAtlasUploadError::InvalidLayout)?;

    let center_page = materialize_fixed_page(source, key, bpp, logical_extent, logical_texels)?;
    let mut neighbors: [Option<Vec<u8>>; 8] = Default::default();
    for (index, (dx, dy)) in NEIGHBOR_OFFSETS.iter().enumerate() {
        let Some(neighbor_key) = offset_key(source, key, *dx, *dy) else {
            continue;
        };
        if resident_neighbors.contains(&neighbor_key) {
            neighbors[index] = Some(materialize_fixed_page(
                source,
                neighbor_key,
                bpp,
                logical_extent,
                logical_texels,
            )?);
        }
    }

    let stored_extent = usize::try_from(layout.stored_tile_extent_px())
        .map_err(|_| MaterialVtAtlasUploadError::SizeOverflow)?;
    let bytes_per_row = usize::try_from(geometry.bytes_per_row())
        .map_err(|_| MaterialVtAtlasUploadError::SizeOverflow)?;
    let staging_len = stored_extent
        .checked_mul(bytes_per_row)
        .ok_or(MaterialVtAtlasUploadError::SizeOverflow)?;
    let mut staging_bytes = Vec::new();
    staging_bytes
        .try_reserve_exact(staging_len)
        .map_err(|_| MaterialVtAtlasUploadError::AllocationFailed)?;
    staging_bytes.resize(staging_len, 0);

    let gutter = usize::try_from(layout.gutter_px())
        .map_err(|_| MaterialVtAtlasUploadError::SizeOverflow)?;
    for output_y in 0..stored_extent {
        let centered_y = isize::try_from(output_y)
            .ok()
            .and_then(|value| value.checked_sub(isize::try_from(gutter).ok()?))
            .ok_or(MaterialVtAtlasUploadError::SizeOverflow)?;
        for output_x in 0..stored_extent {
            let centered_x = isize::try_from(output_x)
                .ok()
                .and_then(|value| value.checked_sub(isize::try_from(gutter).ok()?))
                .ok_or(MaterialVtAtlasUploadError::SizeOverflow)?;
            let (page_offset_x, page_offset_y) =
                page_offset(centered_x, centered_y, logical_extent);
            let source_page = match (page_offset_x, page_offset_y) {
                (-1, -1) => neighbors[0].as_deref(),
                (0, -1) => neighbors[1].as_deref(),
                (1, -1) => neighbors[2].as_deref(),
                (-1, 0) => neighbors[3].as_deref(),
                (1, 0) => neighbors[4].as_deref(),
                (-1, 1) => neighbors[5].as_deref(),
                (0, 1) => neighbors[6].as_deref(),
                (1, 1) => neighbors[7].as_deref(),
                (0, 0) => None,
                _ => unreachable!("page offsets are constrained to a 3x3 neighborhood"),
            };
            let use_neighbor = source_page.is_some();
            let source_page = source_page.unwrap_or(&center_page);
            let source_x = texel_coordinate(centered_x, logical_extent, use_neighbor);
            let source_y = texel_coordinate(centered_y, logical_extent, use_neighbor);
            let source_offset = source_y
                .checked_mul(logical_extent)
                .and_then(|value| value.checked_add(source_x))
                .and_then(|value| value.checked_mul(bpp))
                .ok_or(MaterialVtAtlasUploadError::SizeOverflow)?;
            let output_offset = output_y
                .checked_mul(bytes_per_row)
                .and_then(|value| value.checked_add(output_x.checked_mul(bpp)?))
                .ok_or(MaterialVtAtlasUploadError::SizeOverflow)?;
            let output_end = output_offset
                .checked_add(bpp)
                .ok_or(MaterialVtAtlasUploadError::SizeOverflow)?;
            staging_bytes[output_offset..output_end]
                .copy_from_slice(&source_page[source_offset..source_offset + bpp]);
        }
    }

    Ok(MaterialVtAtlasTile {
        key,
        slot,
        layout,
        bytes_per_row: geometry.bytes_per_row(),
        staging_bytes,
    })
}

const NEIGHBOR_OFFSETS: [(i8, i8); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

fn offset_key(
    source: &MaterialVtSourceSnapshot,
    key: TexturePageKey,
    dx: i8,
    dy: i8,
) -> Option<TexturePageKey> {
    let x = i32::from(key.x) + i32::from(dx);
    let y = i32::from(key.y) + i32::from(dy);
    if x < 0 || y < 0 || x > i32::from(u16::MAX) || y > i32::from(u16::MAX) {
        return None;
    }
    let candidate = TexturePageKey {
        x: u16::try_from(x).ok()?,
        y: u16::try_from(y).ok()?,
        ..key
    };
    source.page(candidate).ok().map(|_| candidate)
}

fn materialize_fixed_page(
    source: &MaterialVtSourceSnapshot,
    key: TexturePageKey,
    bpp: usize,
    logical_extent: usize,
    output_len: usize,
) -> Result<Vec<u8>, MaterialVtAtlasUploadError> {
    let page = source
        .page(key)
        .map_err(|_| MaterialVtAtlasUploadError::SourcePage)?;
    let metadata = page.metadata();
    let width = usize::try_from(metadata.extent_width)
        .map_err(|_| MaterialVtAtlasUploadError::SizeOverflow)?;
    let height = usize::try_from(metadata.extent_height)
        .map_err(|_| MaterialVtAtlasUploadError::SizeOverflow)?;
    if width == 0 || height == 0 {
        return Err(MaterialVtAtlasUploadError::SourcePage);
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_len)
        .map_err(|_| MaterialVtAtlasUploadError::AllocationFailed)?;
    output.resize(output_len, 0);
    for y in 0..logical_extent {
        let source_y = y.min(height - 1);
        let row = page
            .row(u32::try_from(source_y).map_err(|_| MaterialVtAtlasUploadError::SizeOverflow)?)
            .ok_or(MaterialVtAtlasUploadError::SourcePage)?;
        for x in 0..logical_extent {
            let source_x = x.min(width - 1);
            let source_offset = source_x
                .checked_mul(bpp)
                .ok_or(MaterialVtAtlasUploadError::SizeOverflow)?;
            let output_offset = y
                .checked_mul(logical_extent)
                .and_then(|value| value.checked_add(x))
                .and_then(|value| value.checked_mul(bpp))
                .ok_or(MaterialVtAtlasUploadError::SizeOverflow)?;
            output[output_offset..output_offset + bpp]
                .copy_from_slice(&row.bytes[source_offset..source_offset + bpp]);
        }
    }
    Ok(output)
}

fn page_offset(x: isize, y: isize, extent: usize) -> (i8, i8) {
    let extent = isize::try_from(extent).expect("validated page extent fits isize");
    (
        if x < 0 {
            -1
        } else if x >= extent {
            1
        } else {
            0
        },
        if y < 0 {
            -1
        } else if y >= extent {
            1
        } else {
            0
        },
    )
}

fn texel_coordinate(coordinate: isize, extent: usize, neighbor: bool) -> usize {
    let extent = isize::try_from(extent).expect("validated page extent fits isize");
    if neighbor {
        coordinate.rem_euclid(extent) as usize
    } else {
        coordinate.clamp(0, extent - 1) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MaterialVtFormat, MaterialVtSourceDescriptor};

    fn source() -> MaterialVtSourceSnapshot {
        let descriptor =
            MaterialVtSourceDescriptor::new(7, 4, 4, 1, 2, 3, MaterialVtFormat::R8Unorm).unwrap();
        let mut bytes = Vec::new();
        for mip in 0..3_u32 {
            let width = (4_u32 >> mip).max(1);
            let height = (4_u32 >> mip).max(1);
            bytes.extend((0..width * height).map(|value| value as u8));
        }
        MaterialVtSourceSnapshot::new(descriptor, bytes).unwrap()
    }

    fn key(x: u16, y: u16) -> TexturePageKey {
        TexturePageKey {
            texture: 7,
            mip: 0,
            layer: 0,
            x,
            y,
        }
    }

    #[test]
    fn builds_neighbor_gutter_and_aligned_rows() {
        let source = source();
        let config = MaterialVtAtlasConfig::new(1, 4).unwrap();
        let resident = [key(0, 0), key(1, 0), key(0, 1), key(1, 1)]
            .into_iter()
            .collect();
        let tile = build_material_vt_atlas_tile(&source, key(0, 0), 9, &resident, config).unwrap();
        assert_eq!(tile.layout.logical_page_extent_px(), 2);
        assert_eq!(tile.layout.stored_tile_extent_px(), 4);
        assert_eq!(tile.bytes_per_row, 256);
        assert_eq!(tile.staging_bytes.len(), 4 * 256);
        assert_eq!(&tile.staging_bytes[0..4], &[0, 0, 1, 1]);
        assert_eq!(&tile.staging_bytes[256..260], &[0, 0, 1, 2]);
        assert_eq!(&tile.staging_bytes[512..516], &[4, 4, 5, 6]);
    }

    #[test]
    fn missing_neighbors_use_center_edge_extension() {
        let source = source();
        let config = MaterialVtAtlasConfig::new(1, 1).unwrap();
        let tile =
            build_material_vt_atlas_tile(&source, key(0, 0), 0, &BTreeSet::new(), config).unwrap();
        assert_eq!(&tile.staging_bytes[0..4], &[0, 0, 1, 1]);
        assert_eq!(&tile.staging_bytes[256..260], &[0, 0, 1, 1]);
    }
}
