//! Atlas copy-layout planner: the device-free contract for staging page tiles
//! into a physical atlas texture.
//!
//! [`pool::PhysicalPagePool`](super::pool::PhysicalPagePool) assigns every
//! resident page a `u32` slot and emits [`PageUpload`](super::pool::PageUpload)
//! records for the tiles a frame must copy. This module turns those abstract
//! slots into the concrete geometry a `GPU` copy needs, holding no device
//! handle:
//!
//! * the destination subresource — which array layer and texel origin inside
//!   the atlas a slot addresses, and
//! * the source staging layout — the byte offset, `bytes_per_row`, and row
//!   count a `copy_buffer_to_texture` reads from, with `bytes_per_row` rounded
//!   up to [`COPY_BYTES_PER_ROW_ALIGNMENT`] so the layout is valid for `wgpu`.
//!
//! The atlas is modelled as a texture array where each layer holds a square
//! `tiles_per_row * tiles_per_row` grid of equal tiles. A slot maps to
//! `(layer, column, row)` by integer division, so the mapping is total,
//! deterministic, and independent of upload order. Block-compressed formats
//! (`BCn`, `ASTC`) are addressed in blocks: a tile is a whole number of blocks
//! on each axis and the staging rows are block rows.
//!
//! [`plan_atlas_copies`] packs each upload's staging bytes back-to-back; because
//! every tile's padded size is a multiple of [`COPY_BYTES_PER_ROW_ALIGNMENT`],
//! all per-tile offsets stay aligned without extra padding between tiles.

use super::pool::PageUpload;
use super::TexturePageKey;

/// Row-pitch alignment (in bytes) `wgpu` requires for `copy_buffer_to_texture`.
/// Mirrors `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT`; duplicated here so the crate
/// stays device-free.
pub const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;

/// Texel-block description of the atlas texture format.
///
/// For an uncompressed format `block_extent_px == 1` and `bytes_per_block` is
/// the per-texel byte size. For a block-compressed format (`BCn`, `ASTC`)
/// `block_extent_px` is the block edge in texels (for example `4`) and
/// `bytes_per_block` is the compressed block size (for example `8` for `BC1`,
/// `16` for `BC7`).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AtlasTileFormat {
    /// Edge of one texel block, in texels. `1` for uncompressed formats.
    pub block_extent_px: u32,
    /// Bytes per block (per texel when `block_extent_px == 1`).
    pub bytes_per_block: u32,
}

/// Geometry of the physical atlas the pool's slots index into.
///
/// The atlas is a texture array; each layer is a `tiles_per_row * tiles_per_row`
/// grid of `tile_extent_px`-edged square tiles. Construct through
/// [`AtlasGeometry::new`] so the invariants are checked once.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AtlasGeometry {
    tile_extent_px: u32,
    tiles_per_row: u32,
    format: AtlasTileFormat,
}

/// A single staging-to-atlas tile copy, fully resolved against the geometry.
///
/// Fields map one-to-one onto a `wgpu` `copy_buffer_to_texture`: the source is
/// the staging buffer at `staging_offset` with pitch `bytes_per_row` over `rows`
/// block rows; the destination is the atlas array layer `dst_layer` at texel
/// origin `(dst_origin_x, dst_origin_y)` covering an `extent_px`-edged square.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AtlasCopy {
    /// Virtual page this copy fills, carried through from the upload.
    pub key: TexturePageKey,
    /// Physical slot the tile occupies, carried through from the upload.
    pub slot: u32,
    /// Destination array layer in the atlas texture.
    pub dst_layer: u32,
    /// Destination texel X origin inside the layer.
    pub dst_origin_x: u32,
    /// Destination texel Y origin inside the layer.
    pub dst_origin_y: u32,
    /// Tile edge in texels (square).
    pub extent_px: u32,
    /// Byte offset of this tile's first row in the shared staging buffer.
    pub staging_offset: u64,
    /// Padded source row pitch in bytes; a multiple of
    /// [`COPY_BYTES_PER_ROW_ALIGNMENT`].
    pub bytes_per_row: u32,
    /// Number of block rows copied (tile height measured in blocks).
    pub rows: u32,
}

/// Rounds `value` up to the next multiple of `align` (a power of two).
const fn align_up(value: u32, align: u32) -> u32 {
    (value + align - 1) & !(align - 1)
}

impl AtlasGeometry {
    /// Builds a geometry, returning `None` when the parameters cannot describe a
    /// valid block-aligned tile grid.
    ///
    /// Rejects any zero dimension and any tile whose edge is not a whole number
    /// of blocks, since a partial trailing block has no valid copy footprint.
    #[must_use]
    pub fn new(tile_extent_px: u32, tiles_per_row: u32, format: AtlasTileFormat) -> Option<Self> {
        if tile_extent_px == 0
            || tiles_per_row == 0
            || format.block_extent_px == 0
            || format.bytes_per_block == 0
            || !tile_extent_px.is_multiple_of(format.block_extent_px)
        {
            return None;
        }
        Some(Self {
            tile_extent_px,
            tiles_per_row,
            format,
        })
    }

    /// Tile edge in texels.
    #[must_use]
    pub fn tile_extent_px(&self) -> u32 {
        self.tile_extent_px
    }

    /// Tiles along each axis of one array layer.
    #[must_use]
    pub fn tiles_per_row(&self) -> u32 {
        self.tiles_per_row
    }

    /// Format of the atlas texels.
    #[must_use]
    pub fn format(&self) -> AtlasTileFormat {
        self.format
    }

    /// Tiles stored per array layer (`tiles_per_row` squared).
    #[must_use]
    pub fn tiles_per_layer(&self) -> u32 {
        self.tiles_per_row * self.tiles_per_row
    }

    /// Number of texel blocks along one tile edge.
    #[must_use]
    pub fn blocks_per_tile_edge(&self) -> u32 {
        self.tile_extent_px / self.format.block_extent_px
    }

    /// Padded source row pitch for one tile, aligned for `wgpu`.
    #[must_use]
    pub fn bytes_per_row(&self) -> u32 {
        let unpadded = self.blocks_per_tile_edge() * self.format.bytes_per_block;
        align_up(unpadded, COPY_BYTES_PER_ROW_ALIGNMENT)
    }

    /// Padded staging bytes a single tile occupies (pitch times block rows).
    #[must_use]
    pub fn tile_staging_bytes(&self) -> u64 {
        u64::from(self.bytes_per_row()) * u64::from(self.blocks_per_tile_edge())
    }

    /// Resolves a slot to its `(layer, origin_x_px, origin_y_px)` placement.
    #[must_use]
    pub fn slot_placement(&self, slot: u32) -> SlotPlacement {
        let per_layer = self.tiles_per_layer();
        let layer = slot / per_layer;
        let within = slot % per_layer;
        let column = within % self.tiles_per_row;
        let row = within / self.tiles_per_row;
        SlotPlacement {
            layer,
            origin_x_px: column * self.tile_extent_px,
            origin_y_px: row * self.tile_extent_px,
        }
    }

    /// Minimum array-layer count an atlas needs to host `capacity` slots.
    #[must_use]
    pub fn layers_for_capacity(&self, capacity: u32) -> u32 {
        capacity.div_ceil(self.tiles_per_layer())
    }
}

/// Where a slot lands inside the atlas: array layer and texel origin.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SlotPlacement {
    /// Destination array layer.
    pub layer: u32,
    /// Texel X origin of the tile within the layer.
    pub origin_x_px: u32,
    /// Texel Y origin of the tile within the layer.
    pub origin_y_px: u32,
}

/// Resolves every upload into an [`AtlasCopy`] against `geometry`, packing the
/// staging bytes back-to-back in input order.
///
/// Returns the copies and the total staging-buffer size in bytes. Because each
/// tile's padded footprint is a multiple of [`COPY_BYTES_PER_ROW_ALIGNMENT`],
/// every per-tile offset is aligned with no inter-tile padding. The result
/// order matches `uploads`; the caller controls ordering (the pool already
/// emits evict-then-load, key-sorted uploads).
#[must_use]
pub fn plan_atlas_copies(geometry: &AtlasGeometry, uploads: &[PageUpload]) -> AtlasCopyPlan {
    let bytes_per_row = geometry.bytes_per_row();
    let rows = geometry.blocks_per_tile_edge();
    let tile_bytes = geometry.tile_staging_bytes();

    let mut copies = Vec::with_capacity(uploads.len());
    let mut offset: u64 = 0;
    for upload in uploads {
        let placement = geometry.slot_placement(upload.slot);
        copies.push(AtlasCopy {
            key: upload.key,
            slot: upload.slot,
            dst_layer: placement.layer,
            dst_origin_x: placement.origin_x_px,
            dst_origin_y: placement.origin_y_px,
            extent_px: geometry.tile_extent_px(),
            staging_offset: offset,
            bytes_per_row,
            rows,
        });
        offset += tile_bytes;
    }

    AtlasCopyPlan {
        copies,
        staging_bytes: offset,
    }
}

/// Output of [`plan_atlas_copies`]: the per-tile copies plus the staging size.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AtlasCopyPlan {
    /// Resolved copies, one per input upload, in input order.
    pub copies: Vec<AtlasCopy>,
    /// Total staging-buffer bytes the copies read from.
    pub staging_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    const RGBA8: AtlasTileFormat = AtlasTileFormat {
        block_extent_px: 1,
        bytes_per_block: 4,
    };
    const BC7: AtlasTileFormat = AtlasTileFormat {
        block_extent_px: 4,
        bytes_per_block: 16,
    };

    fn upload(texture: u32, slot: u32) -> PageUpload {
        PageUpload {
            key: TexturePageKey {
                texture,
                mip: 0,
                layer: 0,
                x: 0,
                y: 0,
            },
            slot,
        }
    }

    #[test]
    fn rejects_degenerate_geometry() {
        assert!(AtlasGeometry::new(0, 4, RGBA8).is_none());
        assert!(AtlasGeometry::new(128, 0, RGBA8).is_none());
        assert!(AtlasGeometry::new(
            128,
            4,
            AtlasTileFormat {
                block_extent_px: 0,
                bytes_per_block: 4
            }
        )
        .is_none());
        assert!(AtlasGeometry::new(
            128,
            4,
            AtlasTileFormat {
                block_extent_px: 4,
                bytes_per_block: 0
            }
        )
        .is_none());
        // Tile edge not a whole number of blocks.
        assert!(AtlasGeometry::new(130, 4, BC7).is_none());
    }

    #[test]
    fn uncompressed_row_pitch_is_padded_to_alignment() {
        // 64 texels * 4 bytes = 256, already aligned.
        let g = AtlasGeometry::new(64, 8, RGBA8).unwrap();
        assert_eq!(g.bytes_per_row(), 256);
        // 40 texels * 4 bytes = 160 -> padded up to 256.
        let g = AtlasGeometry::new(40, 8, RGBA8).unwrap();
        assert_eq!(g.bytes_per_row(), 256);
        assert_eq!(g.tile_staging_bytes(), 256 * 40);
    }

    #[test]
    fn block_compressed_row_pitch_counts_blocks() {
        // 128 texels / 4 = 32 blocks * 16 bytes = 512 (already aligned).
        let g = AtlasGeometry::new(128, 4, BC7).unwrap();
        assert_eq!(g.blocks_per_tile_edge(), 32);
        assert_eq!(g.bytes_per_row(), 512);
        assert_eq!(g.tile_staging_bytes(), 512 * 32);
    }

    #[test]
    fn slot_placement_tiles_row_major_then_layers() {
        // 2x2 tiles per layer, 64px tiles.
        let g = AtlasGeometry::new(64, 2, RGBA8).unwrap();
        assert_eq!(g.tiles_per_layer(), 4);
        assert_eq!(
            g.slot_placement(0),
            SlotPlacement {
                layer: 0,
                origin_x_px: 0,
                origin_y_px: 0
            }
        );
        assert_eq!(
            g.slot_placement(1),
            SlotPlacement {
                layer: 0,
                origin_x_px: 64,
                origin_y_px: 0
            }
        );
        assert_eq!(
            g.slot_placement(2),
            SlotPlacement {
                layer: 0,
                origin_x_px: 0,
                origin_y_px: 64
            }
        );
        assert_eq!(
            g.slot_placement(3),
            SlotPlacement {
                layer: 0,
                origin_x_px: 64,
                origin_y_px: 64
            }
        );
        // Slot 4 overflows into the next layer.
        assert_eq!(
            g.slot_placement(4),
            SlotPlacement {
                layer: 1,
                origin_x_px: 0,
                origin_y_px: 0
            }
        );
    }

    #[test]
    fn layers_for_capacity_rounds_up() {
        let g = AtlasGeometry::new(64, 2, RGBA8).unwrap();
        assert_eq!(g.layers_for_capacity(0), 0);
        assert_eq!(g.layers_for_capacity(1), 1);
        assert_eq!(g.layers_for_capacity(4), 1);
        assert_eq!(g.layers_for_capacity(5), 2);
    }

    #[test]
    fn plan_packs_offsets_back_to_back_and_resolves_destinations() {
        let g = AtlasGeometry::new(64, 2, RGBA8).unwrap();
        let tile = g.tile_staging_bytes();
        let plan = plan_atlas_copies(&g, &[upload(1, 0), upload(2, 3), upload(3, 4)]);
        assert_eq!(plan.copies.len(), 3);
        assert_eq!(plan.staging_bytes, tile * 3);

        assert_eq!(plan.copies[0].staging_offset, 0);
        assert_eq!(plan.copies[0].dst_layer, 0);
        assert_eq!(plan.copies[0].dst_origin_x, 0);
        assert_eq!(plan.copies[0].dst_origin_y, 0);

        assert_eq!(plan.copies[1].staging_offset, tile);
        assert_eq!(plan.copies[1].dst_layer, 0);
        assert_eq!(plan.copies[1].dst_origin_x, 64);
        assert_eq!(plan.copies[1].dst_origin_y, 64);

        assert_eq!(plan.copies[2].staging_offset, tile * 2);
        assert_eq!(plan.copies[2].dst_layer, 1);
        assert_eq!(plan.copies[2].dst_origin_x, 0);
        assert_eq!(plan.copies[2].dst_origin_y, 0);
    }

    #[test]
    fn every_offset_is_alignment_multiple() {
        let g = AtlasGeometry::new(40, 4, RGBA8).unwrap();
        let uploads: Vec<_> = (0..7).map(|s| upload(s, s)).collect();
        let plan = plan_atlas_copies(&g, &uploads);
        for copy in &plan.copies {
            assert_eq!(copy.bytes_per_row % COPY_BYTES_PER_ROW_ALIGNMENT, 0);
            assert_eq!(
                copy.staging_offset % u64::from(COPY_BYTES_PER_ROW_ALIGNMENT),
                0
            );
        }
    }

    #[test]
    fn empty_uploads_plan_is_empty() {
        let g = AtlasGeometry::new(64, 2, RGBA8).unwrap();
        let plan = plan_atlas_copies(&g, &[]);
        assert!(plan.copies.is_empty());
        assert_eq!(plan.staging_bytes, 0);
    }
}
