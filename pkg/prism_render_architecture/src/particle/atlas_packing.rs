//! Sprite-atlas rectangle bin-packing contract for flipbook and sprite
//! renderers (design §16, §22).
//!
//! Flipbook animation and static sprite rendering both consume normalized
//! `UV` rectangles carved out of a shared texture `atlas`. Rather than authoring
//! those rectangles by hand, production engines (Unreal's sprite atlas, Unity's
//! sprite packer, and countless texture-atlas tools) run a deterministic
//! rectangle packer at bake time and publish the resulting placements. This
//! module owns the `CPU`-verifiable contract for that step: a *shelf packer*
//! that places axis-aligned rectangles without overlap, plus the small helpers
//! the renderer needs to turn integer placements into `GPU`-ready normalized
//! `UV` rectangles and to reason about packing quality.
//!
//! The shelf strategy (place tallest-first, break rows when the current row
//! overflows the `atlas` width) is chosen over the more space-efficient skyline
//! packer precisely because its non-overlap invariant is trivial to verify: `x`
//! strictly increases within a row and `y` strictly increases between rows, so
//! the placements can never collide. The packer never divides by zero, never
//! panics, and never emits a `NaN`: an `atlas` with a zero dimension yields an
//! all-zero `UV` rectangle instead.

use alloc::vec::Vec;

use crate::particle::gpu_layout::VEC4_STRIDE;

/// Comparison epsilon for normalized `UV` coordinates: two `f32` `UV` values
/// closer than this are treated as equal, so the contract never relies on exact
/// floating-point equality.
pub const CMP_EPS: f32 = 1e-6;

/// The unpadded pixel size of a rectangle that still needs a home in the
/// `atlas`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RectSize {
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
}

impl RectSize {
    /// Creates a new size from a width and height in texels.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// The rectangle's area in texels, saturating instead of overflowing so a
    /// degenerate size can never wrap to a small area.
    #[must_use]
    pub fn area(&self) -> u64 {
        u64::from(self.width).saturating_mul(u64::from(self.height))
    }
}

/// A rectangle that has been assigned a concrete origin inside the `atlas`.
///
/// The `id` is the caller's original slice index, so a packed placement is
/// always traceable back to the input rectangle it came from even though the
/// packer reorders rectangles internally for a tighter fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PackedRect {
    /// Original input index this placement corresponds to.
    pub id: u32,
    /// Left edge in texels.
    pub x: u32,
    /// Top edge in texels.
    pub y: u32,
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
}

impl PackedRect {
    /// The exclusive right edge (`x + width`), saturating on overflow.
    #[must_use]
    pub fn right(&self) -> u32 {
        self.x.saturating_add(self.width)
    }

    /// The exclusive bottom edge (`y + height`), saturating on overflow.
    #[must_use]
    pub fn bottom(&self) -> u32 {
        self.y.saturating_add(self.height)
    }

    /// The normalized `UV` rectangle `[u0, v0, u1, v1]` for an `atlas` of the
    /// given texel dimensions.
    ///
    /// Returns all zeros when either `atlas` dimension is zero, so a degenerate
    /// `atlas` never produces a division by zero or a `NaN`.
    #[must_use]
    pub fn uv_rect(&self, atlas_w: u32, atlas_h: u32) -> [f32; 4] {
        if atlas_w == 0 || atlas_h == 0 {
            return [0.0; 4];
        }
        let aw = atlas_w as f32;
        let ah = atlas_h as f32;
        [
            self.x as f32 / aw,
            self.y as f32 / ah,
            self.right() as f32 / aw,
            self.bottom() as f32 / ah,
        ]
    }
}

/// Why a pack attempt failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PackError {
    /// At least one rectangle (with padding) does not fit in the `atlas`, or
    /// the rows overflow the `atlas` height.
    AtlasTooSmall,
    /// The caller passed no rectangles to pack.
    EmptyInput,
}

/// A deterministic shelf-based rectangle packer.
///
/// Rectangles are placed tallest-first into horizontal shelves: each shelf is
/// as tall as its tallest member, and a new shelf opens below the previous one
/// once the current shelf runs out of width. Every rectangle reserves an extra
/// `padding` texels on its right and bottom so neighbours are separated by at
/// least `padding` texels of gutter, which prevents bilinear `UV` bleeding at
/// sample time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShelfPacker {
    atlas_width: u32,
    atlas_height: u32,
    padding: u32,
}

impl ShelfPacker {
    /// Creates a packer for an `atlas` of the given texel dimensions and inter-
    /// rectangle padding. The width and height are clamped up to at least one so
    /// the `atlas` always has usable area.
    #[must_use]
    pub fn new(atlas_width: u32, atlas_height: u32, padding: u32) -> Self {
        Self {
            atlas_width: atlas_width.max(1),
            atlas_height: atlas_height.max(1),
            padding,
        }
    }

    /// The clamped `atlas` width in texels.
    #[must_use]
    pub fn atlas_width(&self) -> u32 {
        self.atlas_width
    }

    /// The clamped `atlas` height in texels.
    #[must_use]
    pub fn atlas_height(&self) -> u32 {
        self.atlas_height
    }

    /// The inter-rectangle padding in texels.
    #[must_use]
    pub fn padding(&self) -> u32 {
        self.padding
    }

    /// Packs `sizes` into non-overlapping placements.
    ///
    /// Rectangles are placed in order of decreasing height, then decreasing
    /// width, then original index (a stable total order), each reserving
    /// `padding` texels of gutter on its right and bottom. Returns
    /// [`PackError::EmptyInput`] for an empty slice and
    /// [`PackError::AtlasTooSmall`] when a rectangle (plus padding) is wider
    /// than the `atlas` or the accumulated shelves overflow the `atlas` height.
    /// Each returned [`PackedRect`] carries the original input index in its
    /// `id`, and the placements are guaranteed pairwise non-overlapping.
    #[must_use = "the packing result reports whether every rectangle fit"]
    pub fn pack(&self, sizes: &[RectSize]) -> Result<Vec<PackedRect>, PackError> {
        if sizes.is_empty() {
            return Err(PackError::EmptyInput);
        }

        // Build the placement order: tallest-first, then widest, then original
        // index. The sort is stable, so the trailing index keeps ties in their
        // original relative order and makes the ordering fully deterministic.
        let mut order: Vec<usize> = (0..sizes.len()).collect();
        order.sort_by_key(|&i| {
            (
                core::cmp::Reverse(sizes[i].height),
                core::cmp::Reverse(sizes[i].width),
                i,
            )
        });

        let mut placed: Vec<PackedRect> = Vec::with_capacity(sizes.len());
        let mut cursor_x: u32 = 0;
        let mut cursor_y: u32 = 0;
        let mut shelf_height: u32 = 0;

        for &i in &order {
            let size = sizes[i];
            let padded_w = size.width.saturating_add(self.padding);
            let padded_h = size.height.saturating_add(self.padding);

            // Open a new shelf when the current one cannot hold this rectangle.
            if cursor_x.saturating_add(padded_w) > self.atlas_width {
                cursor_y = cursor_y.saturating_add(shelf_height);
                cursor_x = 0;
                shelf_height = 0;
            }

            // Even on a fresh shelf the rectangle can be too wide or too tall.
            if cursor_x.saturating_add(padded_w) > self.atlas_width
                || cursor_y.saturating_add(padded_h) > self.atlas_height
            {
                return Err(PackError::AtlasTooSmall);
            }

            let id = u32::try_from(i).unwrap_or(u32::MAX);
            placed.push(PackedRect {
                id,
                x: cursor_x,
                y: cursor_y,
                width: size.width,
                height: size.height,
            });

            cursor_x = cursor_x.saturating_add(padded_w);
            if padded_h > shelf_height {
                shelf_height = padded_h;
            }
        }

        Ok(placed)
    }
}

/// The fraction of the `atlas` covered by the packed rectangles.
///
/// Returns `0.0` when the `atlas` has zero area, and clamps the result to
/// `[0.0, 1.0]` so contrived inputs (placements larger than the `atlas`) never
/// report more than full coverage.
#[must_use]
pub fn occupancy(packed: &[PackedRect], atlas_w: u32, atlas_h: u32) -> f32 {
    let total = u64::from(atlas_w).saturating_mul(u64::from(atlas_h));
    if total == 0 {
        return 0.0;
    }
    let mut used: u64 = 0;
    for rect in packed {
        used = used.saturating_add(u64::from(rect.width).saturating_mul(u64::from(rect.height)));
    }
    let ratio = used as f32 / total as f32;
    ratio.clamp(0.0, 1.0)
}

/// The byte size of a `UV`-rectangle storage buffer holding one `vec4<f32>`
/// (`[u0, v0, u1, v1]`) per packed rectangle.
///
/// Uses the shared [`VEC4_STRIDE`] and saturates instead of overflowing.
#[must_use]
pub fn uv_buffer_bytes(rect_count: u32) -> u64 {
    let stride = u64::try_from(VEC4_STRIDE).unwrap_or(u64::MAX);
    stride.saturating_mul(u64::from(rect_count))
}

/// Whether two packed rectangles overlap.
///
/// Rectangles are treated as half-open intervals `[x, x + width)` and
/// `[y, y + height)`, so edge-to-edge neighbours do not count as overlapping.
/// The packer guarantees its own output never overlaps; this helper exists for
/// tests and run-time assertions.
#[must_use]
pub fn rects_overlap(a: &PackedRect, b: &PackedRect) -> bool {
    a.x < b.right() && b.x < a.right() && a.y < b.bottom() && b.y < a.bottom()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_reports_empty_error() {
        let packer = ShelfPacker::new(64, 64, 1);
        assert_eq!(packer.pack(&[]), Err(PackError::EmptyInput));
    }

    #[test]
    fn single_rect_lands_at_origin() {
        let packer = ShelfPacker::new(64, 64, 0);
        let packed = packer.pack(&[RectSize::new(10, 12)]).expect("fits");
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0].x, 0);
        assert_eq!(packed[0].y, 0);
        assert_eq!(packed[0].width, 10);
        assert_eq!(packed[0].height, 12);
        assert_eq!(packed[0].id, 0);
    }

    #[test]
    fn many_rects_all_placed() {
        let packer = ShelfPacker::new(64, 64, 1);
        let sizes = [
            RectSize::new(10, 20),
            RectSize::new(15, 5),
            RectSize::new(8, 8),
            RectSize::new(20, 20),
            RectSize::new(6, 30),
        ];
        let packed = packer.pack(&sizes).expect("fits");
        assert_eq!(packed.len(), sizes.len());
    }

    #[test]
    fn packed_rects_never_overlap() {
        let packer = ShelfPacker::new(48, 96, 2);
        let sizes = [
            RectSize::new(10, 20),
            RectSize::new(15, 5),
            RectSize::new(8, 8),
            RectSize::new(20, 22),
            RectSize::new(6, 30),
            RectSize::new(12, 12),
            RectSize::new(30, 4),
        ];
        let packed = packer.pack(&sizes).expect("fits");
        for i in 0..packed.len() {
            for j in (i + 1)..packed.len() {
                assert!(
                    !rects_overlap(&packed[i], &packed[j]),
                    "rects {i} and {j} overlap"
                );
            }
        }
    }

    #[test]
    fn rect_wider_than_atlas_is_too_small() {
        let packer = ShelfPacker::new(16, 64, 0);
        assert_eq!(
            packer.pack(&[RectSize::new(100, 4)]),
            Err(PackError::AtlasTooSmall)
        );
    }

    #[test]
    fn rect_taller_than_atlas_is_too_small() {
        let packer = ShelfPacker::new(64, 16, 0);
        assert_eq!(
            packer.pack(&[RectSize::new(4, 100)]),
            Err(PackError::AtlasTooSmall)
        );
    }

    #[test]
    fn padding_separates_neighbours() {
        let padding = 4;
        let packer = ShelfPacker::new(128, 64, padding);
        // Two equal-height rectangles share the first shelf in input order.
        let packed = packer
            .pack(&[RectSize::new(10, 10), RectSize::new(12, 10)])
            .expect("fits");
        assert_eq!(packed[0].y, packed[1].y, "same shelf");
        let gap = i64::from(packed[1].x) - i64::from(packed[0].right());
        assert!(gap >= i64::from(padding), "gap {gap} < padding");
    }

    #[test]
    fn ids_trace_back_to_original_indices() {
        let packer = ShelfPacker::new(64, 128, 1);
        let sizes = [
            RectSize::new(4, 30),
            RectSize::new(4, 10),
            RectSize::new(4, 20),
        ];
        let packed = packer.pack(&sizes).expect("fits");
        let mut ids: Vec<u32> = packed.iter().map(|r| r.id).collect();
        ids.sort_unstable();
        assert_eq!(ids, alloc::vec![0, 1, 2]);
        // Each id maps to a placement whose size matches the original input.
        for rect in &packed {
            let original = sizes[rect.id as usize];
            assert_eq!(rect.width, original.width);
            assert_eq!(rect.height, original.height);
        }
    }

    #[test]
    fn uv_rect_is_normalized() {
        let rect = PackedRect {
            id: 0,
            x: 16,
            y: 32,
            width: 16,
            height: 32,
        };
        let uv = rect.uv_rect(64, 128);
        assert!((uv[0] - 0.25).abs() < CMP_EPS);
        assert!((uv[1] - 0.25).abs() < CMP_EPS);
        assert!((uv[2] - 0.5).abs() < CMP_EPS);
        assert!((uv[3] - 0.5).abs() < CMP_EPS);
    }

    #[test]
    fn uv_rect_zero_atlas_is_all_zero() {
        let rect = PackedRect {
            id: 0,
            x: 4,
            y: 8,
            width: 2,
            height: 2,
        };
        for uv in rect.uv_rect(0, 128) {
            assert!(uv.abs() < CMP_EPS);
        }
        for uv in rect.uv_rect(128, 0) {
            assert!(uv.abs() < CMP_EPS);
        }
    }

    #[test]
    fn occupancy_of_empty_is_zero() {
        assert!(occupancy(&[], 64, 64).abs() < CMP_EPS);
    }

    #[test]
    fn occupancy_of_full_atlas_is_near_one() {
        let packer = ShelfPacker::new(10, 10, 0);
        let packed = packer.pack(&[RectSize::new(10, 10)]).expect("fits");
        let value = occupancy(&packed, 10, 10);
        assert!((value - 1.0).abs() < CMP_EPS);
    }

    #[test]
    fn occupancy_is_clamped_to_unit_range() {
        // A placement larger than the atlas must not report over full coverage.
        let oversized = PackedRect {
            id: 0,
            x: 0,
            y: 0,
            width: 20,
            height: 20,
        };
        let value = occupancy(&[oversized], 10, 10);
        assert!(value <= 1.0);
        assert!((value - 1.0).abs() < CMP_EPS);
    }

    #[test]
    fn occupancy_of_zero_area_atlas_is_zero() {
        let rect = PackedRect {
            id: 0,
            x: 0,
            y: 0,
            width: 4,
            height: 4,
        };
        assert!(occupancy(&[rect], 0, 0).abs() < CMP_EPS);
    }

    #[test]
    fn tallest_rect_is_placed_first() {
        let packer = ShelfPacker::new(64, 128, 1);
        let sizes = [
            RectSize::new(4, 10),
            RectSize::new(4, 30),
            RectSize::new(4, 20),
        ];
        let packed = packer.pack(&sizes).expect("fits");
        // Placement order follows decreasing height, so the first placement is
        // the tallest input (index 1).
        assert_eq!(packed[0].id, 1);
        assert_eq!(packed[0].height, 30);
    }

    #[test]
    fn equal_rects_keep_stable_index_order() {
        let packer = ShelfPacker::new(128, 64, 0);
        let sizes = [
            RectSize::new(5, 10),
            RectSize::new(5, 10),
            RectSize::new(5, 10),
        ];
        let packed = packer.pack(&sizes).expect("fits");
        assert_eq!(packed[0].id, 0);
        assert_eq!(packed[1].id, 1);
        assert_eq!(packed[2].id, 2);
    }

    #[test]
    fn rect_size_area_saturates() {
        assert_eq!(RectSize::new(3, 4).area(), 12);
        assert_eq!(
            RectSize::new(u32::MAX, u32::MAX).area(),
            u64::from(u32::MAX) * u64::from(u32::MAX)
        );
    }

    #[test]
    fn packed_rect_edges_saturate() {
        let normal = PackedRect {
            id: 0,
            x: 5,
            y: 7,
            width: 10,
            height: 20,
        };
        assert_eq!(normal.right(), 15);
        assert_eq!(normal.bottom(), 27);

        let saturating = PackedRect {
            id: 0,
            x: u32::MAX,
            y: u32::MAX,
            width: 10,
            height: 10,
        };
        assert_eq!(saturating.right(), u32::MAX);
        assert_eq!(saturating.bottom(), u32::MAX);
    }

    #[test]
    fn rects_overlap_detects_both_cases() {
        let a = PackedRect {
            id: 0,
            x: 0,
            y: 0,
            width: 10,
            height: 10,
        };
        let overlapping = PackedRect {
            id: 1,
            x: 5,
            y: 5,
            width: 10,
            height: 10,
        };
        let touching = PackedRect {
            id: 2,
            x: 10,
            y: 0,
            width: 10,
            height: 10,
        };
        assert!(rects_overlap(&a, &overlapping));
        // Edge-to-edge neighbours are half-open and thus do not overlap.
        assert!(!rects_overlap(&a, &touching));
    }

    #[test]
    fn uv_buffer_bytes_uses_vec4_stride() {
        assert_eq!(uv_buffer_bytes(0), 0);
        assert_eq!(uv_buffer_bytes(4), 64);
        assert_eq!(
            uv_buffer_bytes(1),
            u64::try_from(VEC4_STRIDE).expect("stride fits u64")
        );
    }

    #[test]
    fn new_clamps_dimensions_to_at_least_one() {
        let packer = ShelfPacker::new(0, 0, 3);
        assert_eq!(packer.atlas_width(), 1);
        assert_eq!(packer.atlas_height(), 1);
        assert_eq!(packer.padding(), 3);
    }

    #[test]
    fn shelf_wraps_to_a_new_row() {
        // Two rectangles that cannot share a row must stack vertically.
        let packer = ShelfPacker::new(12, 64, 0);
        let packed = packer
            .pack(&[RectSize::new(10, 10), RectSize::new(10, 10)])
            .expect("fits");
        assert_eq!(packed[0].y, 0);
        assert!(packed[1].y >= packed[0].bottom());
        assert!(!rects_overlap(&packed[0], &packed[1]));
    }
}
