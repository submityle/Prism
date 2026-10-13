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
//! [`AtlasPageLayout`] is the explicit virtual-page contract for tiles that
//! carry a filtering gutter. It distinguishes the logical page extent from
//! the larger stored tile extent and requires both the logical extent and
//! gutter to respect the format's compression-block boundaries. The CPU
//! clamp-edge builder is a per-page fallback only: it repeats this page's edge
//! and does not fetch adjacent virtual pages, so it cannot promise seamless
//! filtering across page boundaries.
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

/// Logical virtual-page dimensions and the physical extent stored per atlas
/// slot.
///
/// `stored_tile_extent_px = logical_page_extent_px + 2 * gutter_px`. For
/// block-compressed formats, the logical extent and gutter must each be a
/// multiple of `block_extent_px`, making both the inner-page origin and the
/// outer tile extent block aligned. A zero gutter is valid and is an identity
/// layout for legacy page data.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AtlasPageLayout {
    logical_page_extent_px: u32,
    gutter_px: u32,
    stored_tile_extent_px: u32,
    format: AtlasTileFormat,
}

/// CPU-built tile with its explicit logical/stored dimensions.
///
/// `texels` is row-major decoded texel data, one `T` per texel, including the
/// gutter. For compressed uploads, encode this result using the layout's
/// format before staging it; this helper does not synthesize compressed block
/// bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AtlasGutteredPage<T> {
    /// Layout used to build this tile.
    pub layout: AtlasPageLayout,
    /// Row-major texels covering `layout.stored_tile_extent_px` square.
    pub texels: Vec<T>,
}

/// The resident 3×3 page neighborhood used to build a seam-aware gutter.
///
/// `center` is required and every provided neighbor must have the same square
/// extent. Missing neighbors are extended from the nearest center-page edge;
/// this lets pages be produced before the entire neighborhood is resident,
/// but callers must rebuild affected gutters when a missing neighbor arrives.
#[derive(Clone, Copy, Debug)]
pub struct AtlasPageNeighborhood<'a, T> {
    /// Page immediately left of the center page.
    pub left: Option<&'a [T]>,
    /// Page immediately right of the center page.
    pub right: Option<&'a [T]>,
    /// Page immediately above the center page.
    pub top: Option<&'a [T]>,
    /// Page immediately below the center page.
    pub bottom: Option<&'a [T]>,
    /// Diagonal page above and left of the center page.
    pub top_left: Option<&'a [T]>,
    /// Diagonal page above and right of the center page.
    pub top_right: Option<&'a [T]>,
    /// Diagonal page below and left of the center page.
    pub bottom_left: Option<&'a [T]>,
    /// Diagonal page below and right of the center page.
    pub bottom_right: Option<&'a [T]>,
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

/// Rounds `value` up to the next multiple of `align` (a power of two),
/// returning `None` when the aligned value is not representable as a `u32`.
const fn checked_align_up(value: u32, align: u32) -> Option<u32> {
    match value.checked_add(align - 1) {
        Some(value) => Some(value & !(align - 1)),
        None => None,
    }
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

        // Validate every derived u32 quantity used by the infallible geometry
        // accessors below. This also guarantees that slot origins fit in u32.
        tiles_per_row.checked_mul(tiles_per_row)?;
        tiles_per_row.checked_mul(tile_extent_px)?;

        let blocks_per_tile_edge = tile_extent_px / format.block_extent_px;
        let unpadded_bytes_per_row = blocks_per_tile_edge.checked_mul(format.bytes_per_block)?;
        checked_align_up(unpadded_bytes_per_row, COPY_BYTES_PER_ROW_ALIGNMENT)?;

        Some(Self {
            tile_extent_px,
            tiles_per_row,
            format,
        })
    }

    /// Builds physical atlas geometry whose slot tile matches `layout`.
    ///
    /// This is the preferred constructor for guttered pages: atlas placement
    /// and row-pitch calculations use the stored extent, never the logical
    /// page extent. Existing [`Self::new`] calls retain their original
    /// no-gutter behavior.
    #[must_use]
    pub fn for_page_layout(tiles_per_row: u32, layout: AtlasPageLayout) -> Option<Self> {
        Self::new(layout.stored_tile_extent_px, tiles_per_row, layout.format)
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
        self.tiles_per_row
            .checked_mul(self.tiles_per_row)
            .expect("AtlasGeometry::new validates tiles per layer")
    }

    /// Number of texel blocks along one tile edge.
    #[must_use]
    pub fn blocks_per_tile_edge(&self) -> u32 {
        self.tile_extent_px / self.format.block_extent_px
    }

    /// Padded source row pitch for one tile, aligned for `wgpu`.
    #[must_use]
    pub fn bytes_per_row(&self) -> u32 {
        let unpadded = self
            .blocks_per_tile_edge()
            .checked_mul(self.format.bytes_per_block)
            .expect("AtlasGeometry::new validates unpadded row pitch");
        checked_align_up(unpadded, COPY_BYTES_PER_ROW_ALIGNMENT)
            .expect("AtlasGeometry::new validates aligned row pitch")
    }

    /// Padded staging bytes a single tile occupies (pitch times block rows).
    #[must_use]
    pub fn tile_staging_bytes(&self) -> u64 {
        u64::from(self.bytes_per_row())
            .checked_mul(u64::from(self.blocks_per_tile_edge()))
            .expect("u32 row pitch times u32 block rows fits in u64")
    }

    /// Resolves a slot to its `(layer, origin_x_px, origin_y_px)` placement.
    #[must_use]
    pub fn slot_placement(&self, slot: u32) -> SlotPlacement {
        let per_layer = self.tiles_per_layer();
        let layer = slot
            .checked_div(per_layer)
            .expect("AtlasGeometry::new guarantees nonzero tiles per layer");
        let within = slot
            .checked_rem(per_layer)
            .expect("AtlasGeometry::new guarantees nonzero tiles per layer");
        let column = within % self.tiles_per_row;
        let row = within / self.tiles_per_row;
        SlotPlacement {
            layer,
            origin_x_px: column
                .checked_mul(self.tile_extent_px)
                .expect("AtlasGeometry::new validates atlas width"),
            origin_y_px: row
                .checked_mul(self.tile_extent_px)
                .expect("AtlasGeometry::new validates atlas height"),
        }
    }

    /// Minimum array-layer count an atlas needs to host `capacity` slots.
    #[must_use]
    pub fn layers_for_capacity(&self, capacity: u32) -> u32 {
        let per_layer = self.tiles_per_layer();
        let whole_layers = capacity / per_layer;
        whole_layers
            .checked_add(u32::from(capacity % per_layer != 0))
            .expect("ceil(capacity / nonzero tiles_per_layer) fits in u32")
    }
}

impl AtlasPageLayout {
    /// Creates a validated logical-page / gutter / stored-tile layout.
    ///
    /// Returns `None` for zero logical extent, invalid format data, arithmetic
    /// overflow, a logical extent or gutter that is not compression-block
    /// aligned, or a row pitch that cannot be represented by the copy API.
    #[must_use]
    pub fn new(
        logical_page_extent_px: u32,
        gutter_px: u32,
        format: AtlasTileFormat,
    ) -> Option<Self> {
        if logical_page_extent_px == 0
            || format.block_extent_px == 0
            || format.bytes_per_block == 0
            || !logical_page_extent_px.is_multiple_of(format.block_extent_px)
            || !gutter_px.is_multiple_of(format.block_extent_px)
        {
            return None;
        }

        let stored_tile_extent_px = gutter_px
            .checked_mul(2)?
            .checked_add(logical_page_extent_px)?;
        let blocks_per_edge = stored_tile_extent_px / format.block_extent_px;
        let unpadded_bytes_per_row = blocks_per_edge.checked_mul(format.bytes_per_block)?;
        checked_align_up(unpadded_bytes_per_row, COPY_BYTES_PER_ROW_ALIGNMENT)?;

        Some(Self {
            logical_page_extent_px,
            gutter_px,
            stored_tile_extent_px,
            format,
        })
    }

    /// Extent of the useful virtual page, excluding its gutter.
    #[must_use]
    pub fn logical_page_extent_px(&self) -> u32 {
        self.logical_page_extent_px
    }

    /// Per-edge gutter extent in texels.
    #[must_use]
    pub fn gutter_px(&self) -> u32 {
        self.gutter_px
    }

    /// Extent copied to and stored in one physical atlas slot.
    #[must_use]
    pub fn stored_tile_extent_px(&self) -> u32 {
        self.stored_tile_extent_px
    }

    /// Texel-block description used to encode and copy this page.
    #[must_use]
    pub fn format(&self) -> AtlasTileFormat {
        self.format
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
    try_plan_atlas_copies(geometry, uploads).expect("atlas staging plan size must fit in u64")
}

/// Plans staging copies for pages laid out by [`AtlasPageLayout`].
///
/// The atlas geometry must use the layout's `stored_tile_extent_px` and exact
/// format. Each returned [`AtlasCopy`] therefore copies the complete stored
/// tile (logical page plus all four gutters). Its destination origin is the
/// outer tile origin; the logical page begins `layout.gutter_px()` texels
/// inward on both axes. The input staging bytes must already contain the
/// guttered/encoded tile; this planner only describes copies.
///
/// Returns `None` if geometry and layout disagree or the aggregate staging
/// size overflows. The legacy [`plan_atlas_copies`] remains available for
/// existing no-gutter callers.
#[must_use]
pub fn plan_atlas_page_copies(
    geometry: &AtlasGeometry,
    layout: AtlasPageLayout,
    uploads: &[PageUpload],
) -> Option<AtlasPageCopyPlan> {
    if geometry.tile_extent_px() != layout.stored_tile_extent_px
        || geometry.format() != layout.format
    {
        return None;
    }
    let copy_plan = try_plan_atlas_copies(geometry, uploads)?;
    Some(AtlasPageCopyPlan { layout, copy_plan })
}

/// Builds a square page's clamp-edge gutter in row-major texel space.
///
/// Each output texel maps to the nearest texel inside the input logical page.
/// This is a local edge-extension policy, not an adjacency-aware virtual
/// texture filter: neighboring page content is not consulted, so bilinear or
/// trilinear filtering can still show seams at page boundaries. Callers that
/// require seamless cross-page filtering must source gutters from the
/// appropriate neighboring pages (including corners) or use a sampling path
/// that combines cross-page taps.
///
/// The input must contain exactly `logical_page_extent_px²` decoded texels.
/// For a block-compressed layout, encode the returned expanded texels after
/// building; this function operates on texels, not compressed blocks. Returns
/// `None` for a mismatched input length or unrepresentable output allocation.
#[must_use]
pub fn build_clamp_edge_gutter<T: Copy>(
    logical_page: &[T],
    layout: AtlasPageLayout,
) -> Option<AtlasGutteredPage<T>> {
    let logical_extent = usize::try_from(layout.logical_page_extent_px).ok()?;
    let stored_extent = usize::try_from(layout.stored_tile_extent_px).ok()?;
    let gutter = usize::try_from(layout.gutter_px).ok()?;
    let expected_input_len = logical_extent.checked_mul(logical_extent)?;
    if logical_page.len() != expected_input_len {
        return None;
    }
    let output_len = stored_extent.checked_mul(stored_extent)?;
    let mut texels = Vec::new();
    texels.try_reserve_exact(output_len).ok()?;

    for y in 0..stored_extent {
        let source_y = y.saturating_sub(gutter).min(logical_extent - 1);
        for x in 0..stored_extent {
            let source_x = x.saturating_sub(gutter).min(logical_extent - 1);
            texels.push(logical_page[source_y * logical_extent + source_x]);
        }
    }

    Some(AtlasGutteredPage { layout, texels })
}

/// Builds a page tile using resident neighbors for its filtering gutter.
///
/// Unlike [`build_clamp_edge_gutter`], this copies actual neighboring page
/// texels into each available edge and corner, preventing filtering seams
/// where the neighborhood is resident and gutters are current. Missing
/// neighbors use center-edge extension and therefore remain a temporary seam
/// fallback. Callers must invalidate/rebuild this page when any missing
/// neighbor becomes resident or changes content. The gutter must not exceed
/// one logical page extent because the supplied neighborhood is exactly 3×3.
///
/// Returns `None` when the center or a supplied neighbor has the wrong texel
/// count, the gutter exceeds one page, or output allocation cannot be reserved.
#[must_use]
pub fn build_neighbor_guttered_page<T: Copy>(
    center: &[T],
    neighborhood: AtlasPageNeighborhood<'_, T>,
    layout: AtlasPageLayout,
) -> Option<AtlasGutteredPage<T>> {
    let extent = usize::try_from(layout.logical_page_extent_px).ok()?;
    let gutter = usize::try_from(layout.gutter_px).ok()?;
    if gutter > extent {
        return None;
    }
    let input_len = extent.checked_mul(extent)?;
    if center.len() != input_len {
        return None;
    }
    let neighbors = [
        neighborhood.left,
        neighborhood.right,
        neighborhood.top,
        neighborhood.bottom,
        neighborhood.top_left,
        neighborhood.top_right,
        neighborhood.bottom_left,
        neighborhood.bottom_right,
    ];
    if neighbors
        .into_iter()
        .flatten()
        .any(|page| page.len() != input_len)
    {
        return None;
    }

    let stored_extent = usize::try_from(layout.stored_tile_extent_px).ok()?;
    let signed_extent = isize::try_from(extent).ok()?;
    let signed_gutter = isize::try_from(gutter).ok()?;
    let signed_last_texel = isize::try_from(extent - 1).ok()?;
    let output_len = stored_extent.checked_mul(stored_extent)?;
    let mut texels = Vec::new();
    texels.try_reserve_exact(output_len).ok()?;

    for output_y in 0..stored_extent {
        let centered_y = isize::try_from(output_y).ok()? - signed_gutter;
        for output_x in 0..stored_extent {
            let centered_x = isize::try_from(output_x).ok()? - signed_gutter;
            let page_x = if centered_x < 0 {
                -1
            } else if centered_x >= signed_extent {
                1
            } else {
                0
            };
            let page_y = if centered_y < 0 {
                -1
            } else if centered_y >= signed_extent {
                1
            } else {
                0
            };
            let neighbor = match (page_x, page_y) {
                (-1, -1) => neighborhood.top_left,
                (0, -1) => neighborhood.top,
                (1, -1) => neighborhood.top_right,
                (-1, 0) => neighborhood.left,
                (0, 0) => None,
                (1, 0) => neighborhood.right,
                (-1, 1) => neighborhood.bottom_left,
                (0, 1) => neighborhood.bottom,
                (1, 1) => neighborhood.bottom_right,
                _ => unreachable!("page offsets are constrained to the 3x3 neighborhood"),
            };

            let source_page = neighbor.unwrap_or(center);
            let source_x = if neighbor.is_some() {
                centered_x.rem_euclid(signed_extent) as usize
            } else {
                centered_x.clamp(0, signed_last_texel) as usize
            };
            let source_y = if neighbor.is_some() {
                centered_y.rem_euclid(signed_extent) as usize
            } else {
                centered_y.clamp(0, signed_last_texel) as usize
            };
            texels.push(source_page[source_y * extent + source_x]);
        }
    }

    Some(AtlasGutteredPage { layout, texels })
}

fn try_plan_atlas_copies(
    geometry: &AtlasGeometry,
    uploads: &[PageUpload],
) -> Option<AtlasCopyPlan> {
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
        offset = offset.checked_add(tile_bytes)?;
    }

    Some(AtlasCopyPlan {
        copies,
        staging_bytes: offset,
    })
}

/// Output of [`plan_atlas_page_copies`]: validated page layout and resolved
/// physical copy plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AtlasPageCopyPlan {
    /// Logical extent, gutter, stored extent, and format shared by each copy.
    pub layout: AtlasPageLayout,
    /// Copies of complete stored tiles, with their staging-buffer size.
    pub copy_plan: AtlasCopyPlan,
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
    fn rejects_geometry_whose_derived_u32_values_overflow() {
        // tiles_per_layer cannot represent 65_536 squared.
        assert!(AtlasGeometry::new(1, 65_536, RGBA8).is_none());

        // Atlas layer width cannot represent three 2^31-texel tiles. The
        // compressed row pitch itself remains representable in this case.
        let wide_tile = AtlasTileFormat {
            block_extent_px: 4,
            bytes_per_block: 1,
        };
        assert!(AtlasGeometry::new(1 << 31, 3, wide_tile).is_none());

        // The row pitch rounds beyond u32::MAX when aligned to 256 bytes.
        assert!(AtlasGeometry::new(
            u32::MAX - 254,
            1,
            AtlasTileFormat {
                block_extent_px: 1,
                bytes_per_block: 1,
            }
        )
        .is_none());
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
    fn large_valid_geometry_keeps_slot_and_capacity_arithmetic_in_range() {
        let g = AtlasGeometry::new(1, 65_535, RGBA8).unwrap();
        assert_eq!(g.tiles_per_layer(), 65_535 * 65_535);
        assert_eq!(g.layers_for_capacity(u32::MAX), 2);

        let placement = g.slot_placement(u32::MAX);
        assert_eq!(placement.layer, 1);
        assert!(placement.origin_x_px < g.tiles_per_row());
        assert!(placement.origin_y_px < g.tiles_per_row());
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

    #[test]
    fn one_pixel_clamp_gutter_repeats_edges_and_corners() {
        let layout = AtlasPageLayout::new(2, 1, RGBA8).unwrap();
        let page = [1_u8, 2, 3, 4];
        let tile = build_clamp_edge_gutter(&page, layout).unwrap();

        assert_eq!(layout.logical_page_extent_px(), 2);
        assert_eq!(layout.gutter_px(), 1);
        assert_eq!(layout.stored_tile_extent_px(), 4);
        assert_eq!(
            tile.texels,
            [1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]
        );
    }

    #[test]
    fn multi_pixel_clamp_gutter_extends_each_edge_without_neighbor_data() {
        let layout = AtlasPageLayout::new(2, 2, RGBA8).unwrap();
        let page = [10_u8, 20, 30, 40];
        let tile = build_clamp_edge_gutter(&page, layout).unwrap();

        assert_eq!(layout.stored_tile_extent_px(), 6);
        assert_eq!(
            tile.texels,
            [
                10, 10, 10, 20, 20, 20, 10, 10, 10, 20, 20, 20, 10, 10, 10, 20, 20, 20, 30, 30, 30,
                40, 40, 40, 30, 30, 30, 40, 40, 40, 30, 30, 30, 40, 40, 40,
            ]
        );
    }

    #[test]
    fn neighbor_gutter_uses_adjacent_edges_and_diagonal_corners() {
        let layout = AtlasPageLayout::new(2, 1, RGBA8).unwrap();
        let left = [10_u8; 4];
        let right = [20_u8; 4];
        let top = [30_u8; 4];
        let bottom = [40_u8; 4];
        let top_left = [50_u8; 4];
        let top_right = [60_u8; 4];
        let bottom_left = [70_u8; 4];
        let bottom_right = [80_u8; 4];
        let center = [1_u8, 2, 3, 4];

        let tile = build_neighbor_guttered_page(
            &center,
            AtlasPageNeighborhood {
                left: Some(&left),
                right: Some(&right),
                top: Some(&top),
                bottom: Some(&bottom),
                top_left: Some(&top_left),
                top_right: Some(&top_right),
                bottom_left: Some(&bottom_left),
                bottom_right: Some(&bottom_right),
            },
            layout,
        )
        .unwrap();

        assert_eq!(
            tile.texels,
            [
                50, 30, 30, 60, // top row: corner pages + top edge
                10, 1, 2, 20, // left edge + center page + right edge
                10, 3, 4, 20, // second logical row
                70, 40, 40, 80, // bottom row
            ]
        );
    }

    #[test]
    fn missing_neighbors_extend_center_edges_and_bad_neighbors_are_rejected() {
        let layout = AtlasPageLayout::new(2, 1, RGBA8).unwrap();
        let center = [1_u8, 2, 3, 4];
        let fallback = build_neighbor_guttered_page(
            &center,
            AtlasPageNeighborhood {
                left: None,
                right: None,
                top: None,
                bottom: None,
                top_left: None,
                top_right: None,
                bottom_left: None,
                bottom_right: None,
            },
            layout,
        )
        .unwrap();
        assert_eq!(
            fallback.texels,
            [1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]
        );

        let bad_neighbor = [9_u8; 3];
        assert!(build_neighbor_guttered_page(
            &center,
            AtlasPageNeighborhood {
                left: Some(&bad_neighbor),
                right: None,
                top: None,
                bottom: None,
                top_left: None,
                top_right: None,
                bottom_left: None,
                bottom_right: None,
            },
            layout,
        )
        .is_none());

        let too_wide_gutter = AtlasPageLayout::new(2, 4, RGBA8).unwrap();
        assert!(build_neighbor_guttered_page(
            &center,
            AtlasPageNeighborhood {
                left: Some(&center),
                right: Some(&center),
                top: Some(&center),
                bottom: Some(&center),
                top_left: Some(&center),
                top_right: Some(&center),
                bottom_left: Some(&center),
                bottom_right: Some(&center),
            },
            too_wide_gutter,
        )
        .is_none());
    }

    #[test]
    fn zero_gutter_is_legacy_identity() {
        let page = [1_u32, 2, 3, 4];
        let layout = AtlasPageLayout::new(2, 0, RGBA8).unwrap();
        let tile = build_clamp_edge_gutter(&page, layout).unwrap();
        assert_eq!(
            layout.stored_tile_extent_px(),
            layout.logical_page_extent_px()
        );
        assert_eq!(tile.texels, page);

        let legacy_geometry = AtlasGeometry::new(2, 4, RGBA8).unwrap();
        let layout_geometry = AtlasGeometry::for_page_layout(4, layout).unwrap();
        assert_eq!(legacy_geometry, layout_geometry);
    }

    #[test]
    fn compressed_page_layout_requires_logical_and_gutter_block_alignment() {
        assert!(AtlasPageLayout::new(6, 4, BC7).is_none());
        assert!(AtlasPageLayout::new(8, 2, BC7).is_none());

        let layout = AtlasPageLayout::new(8, 4, BC7).unwrap();
        assert_eq!(layout.stored_tile_extent_px(), 16);
        let geometry = AtlasGeometry::for_page_layout(2, layout).unwrap();
        assert_eq!(geometry.tile_extent_px(), 16);
        assert_eq!(geometry.blocks_per_tile_edge(), 4);
        assert_eq!(geometry.bytes_per_row(), 256);

        let plan = plan_atlas_page_copies(&geometry, layout, &[upload(7, 2)]).unwrap();
        assert_eq!(plan.copy_plan.copies[0].extent_px, 16);
        assert_eq!(plan.copy_plan.copies[0].rows, 4);
        assert_eq!(plan.copy_plan.copies[0].dst_origin_x, 0);
        assert_eq!(plan.copy_plan.copies[0].dst_origin_y, 16);
    }

    #[test]
    fn guttered_copy_planner_rejects_geometry_layout_mismatch() {
        let layout = AtlasPageLayout::new(64, 4, RGBA8).unwrap();
        let wrong_extent = AtlasGeometry::new(64, 4, RGBA8).unwrap();
        let wrong_format = AtlasGeometry::new(72, 4, BC7).unwrap();
        assert!(plan_atlas_page_copies(&wrong_extent, layout, &[]).is_none());
        assert!(plan_atlas_page_copies(&wrong_format, layout, &[]).is_none());
    }

    #[test]
    fn page_layout_rejects_bad_input_and_overflowing_extents() {
        let layout = AtlasPageLayout::new(2, 1, RGBA8).unwrap();
        assert!(build_clamp_edge_gutter(&[1_u8, 2, 3], layout).is_none());
        assert!(AtlasPageLayout::new(u32::MAX, 1, RGBA8).is_none());
        assert!(AtlasPageLayout::new(1, u32::MAX, RGBA8).is_none());

        let near_limit = AtlasPageLayout::new(
            u32::MAX - 1,
            0,
            AtlasTileFormat {
                block_extent_px: 1,
                bytes_per_block: 1,
            },
        );
        assert!(near_limit.is_none());
    }

    #[test]
    fn guttered_copy_plan_rejects_aggregate_staging_size_overflow() {
        let format = AtlasTileFormat {
            block_extent_px: 1,
            bytes_per_block: 1,
        };
        // Each tile footprint fits in u64, but two such footprints do not.
        // The test plans descriptors only and never allocates the staging data.
        let extent = u32::MAX - (COPY_BYTES_PER_ROW_ALIGNMENT - 1);
        let layout = AtlasPageLayout::new(extent, 0, format).unwrap();
        let geometry = AtlasGeometry::for_page_layout(1, layout).unwrap();
        assert!(plan_atlas_page_copies(&geometry, layout, &[upload(1, 0), upload(2, 0)]).is_none());
    }
}
