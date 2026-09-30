//! Mip-safe `atlas` gutter / inner-padding contract for texture-`atlas` blocks
//! (design §16, §22).
//!
//! When many sprite or flipbook sub-images share one texture `atlas`, a
//! `mipmap` chain averages neighbouring `texel`s together as it downsamples. At
//! a block boundary that averaging pulls in `texel`s belonging to the *adjacent*
//! block, so a coarse mip level bleeds one block's colour into its neighbour —
//! the classic `atlas` colour-seam artifact. Production bakers (Unreal's
//! `atlas` groups, Unity's sprite packer, and every `texture`-`atlas` tool that
//! ships mips) fix this by reserving a *gutter* of padding `texel`s around each
//! block and filling that gutter by copying the block's own edge `texel`s
//! outward, so a coarse mip only ever averages same-block colour.
//!
//! This module owns the `CPU`-verifiable contract for that step:
//!
//! - how wide the gutter must be for a given `mipmap` level count
//!   ([`required_padding`]) and how many mip levels a block can safely carry
//!   ([`max_safe_mip_levels`]);
//! - the block-local rectangle grown by that gutter ([`Rect::pad_rect`]);
//! - the edge-extension rule that maps a padded coordinate back to a source
//!   `texel` inside the block ([`gutter_source_index`]) under the three border
//!   [`PadMode`]s;
//! - the concrete destination/source `texel` copy list a baker replays to fill
//!   the gutter ([`Rect::border_copy_map`]);
//! - the padded-frame and content `UV` rectangles a renderer samples, complete
//!   with a half-`texel` inset ([`Rect::build_padded_uv`]).
//!
//! # Boundary with siblings
//!
//! [`super::atlas_packing`] owns *where* blocks are placed (the shelf packer,
//! occupancy, and its own `UV` rectangle). [`super::billboard_atlas`] owns
//! flipbook *frame animation*. This module owns neither: it only computes the
//! mip gutter, the edge-copy map, and the padded `UV` for a block whose origin
//! and size a caller already decided.
//!
//! # Transcendental-free contract
//!
//! Every result is pure integer bit arithmetic plus, for `UV`s, a single divide
//! guarded against a zero `atlas` dimension. Log-base-two comes from
//! [`u32::leading_zeros`], powers of two from a bounded left shift, and no
//! `sin`/`cos`/`exp`/`log`/`powf`/`ceil`/`round` is ever called, so the
//! `CPU`-baked gutter matches a future `GPU` kernel bit for bit and nothing
//! panics, divides by zero, or produces a `NaN`.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Comparison epsilon for normalized `UV` coordinates: two `f32` `UV` values
/// closer than this are treated as equal, so the contract and its tests never
/// rely on exact floating-point equality.
pub const CMP_EPS: f32 = 1e-6;

/// How the gutter is filled outside a block's content region.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PadMode {
    /// Repeat the nearest edge `texel` outward (the standard `atlas` gutter).
    ClampEdge,
    /// Reflect the content across each edge so the gutter mirrors the block.
    Mirror,
    /// Leave the gutter empty; padded coordinates resolve to the
    /// [`u32::MAX`] sentinel and are skipped by the copy map.
    Transparent,
}

/// A pixel-space rectangle inside a texture `atlas`, owned by this module so the
/// gutter contract never depends on the packer's [`super::atlas_packing`]
/// placement type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Rect {
    /// Left edge in `texel`s.
    pub x: u32,
    /// Top edge in `texel`s.
    pub y: u32,
    /// Width in `texel`s.
    pub w: u32,
    /// Height in `texel`s.
    pub h: u32,
}

impl Rect {
    /// Creates a rectangle from an origin and size in `texel`s.
    #[must_use]
    pub fn new(x: u32, y: u32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }

    /// The exclusive right edge (`x + w`), saturating on overflow.
    #[must_use]
    pub fn right(&self) -> u32 {
        self.x.saturating_add(self.w)
    }

    /// The exclusive bottom edge (`y + h`), saturating on overflow.
    #[must_use]
    pub fn bottom(&self) -> u32 {
        self.y.saturating_add(self.h)
    }

    /// Grows this rectangle outward by `pad` `texel`s on every side.
    ///
    /// The origin is clamped at the `atlas` origin `(0, 0)`, so a block hugging
    /// the top-left corner never produces a negative coordinate; whatever left
    /// or top padding is clipped by that clamp is folded into the width and
    /// height so the exclusive right and bottom edges stay at
    /// `x + w + pad` and `y + h + pad`. All arithmetic saturates.
    #[must_use]
    pub fn pad_rect(&self, pad: u32) -> Rect {
        let nx = self.x.saturating_sub(pad);
        let ny = self.y.saturating_sub(pad);
        let left = self.x - nx;
        let top = self.y - ny;
        let nw = self.w.saturating_add(left).saturating_add(pad);
        let nh = self.h.saturating_add(top).saturating_add(pad);
        Rect {
            x: nx,
            y: ny,
            w: nw,
            h: nh,
        }
    }

    /// Returns the padded-frame `UV` rectangle and the content `UV` rectangle
    /// for this block inside an `atlas` of the given `texel` dimensions.
    ///
    /// Both rectangles are `[u0, v0, u1, v1]`. The first spans the whole padded
    /// frame (`pad_rect`); the second is the block content inset by a
    /// half-`texel` on every side so a sampler lands on `texel` centres and
    /// never on the seam between content and gutter. Returns all zeros when
    /// either `atlas` dimension is zero, so a degenerate `atlas` never divides
    /// by zero or yields a `NaN`; a degenerate content size collapses the inner
    /// rectangle to its inset origin rather than inverting it.
    #[must_use]
    pub fn build_padded_uv(&self, atlas_w: u32, atlas_h: u32, pad: u32) -> ([f32; 4], [f32; 4]) {
        if atlas_w == 0 || atlas_h == 0 {
            return ([0.0; 4], [0.0; 4]);
        }
        let aw = atlas_w as f32;
        let ah = atlas_h as f32;

        let outer = self.pad_rect(pad);
        let outer_uv = [
            outer.x as f32 / aw,
            outer.y as f32 / ah,
            outer.right() as f32 / aw,
            outer.bottom() as f32 / ah,
        ];

        let half_u = 0.5 / aw;
        let half_v = 0.5 / ah;
        let u0 = self.x as f32 / aw + half_u;
        let v0 = self.y as f32 / ah + half_v;
        let u1 = (self.right() as f32 / aw - half_u).max(u0);
        let v1 = (self.bottom() as f32 / ah - half_v).max(v0);

        (outer_uv, [u0, v0, u1, v1])
    }

    /// Builds the gutter copy list for this block: one
    /// `(dst_x, dst_y, src_x, src_y)` entry per padding `texel`, in row-major
    /// order, all in absolute `atlas` `texel` coordinates.
    ///
    /// Each destination is a `texel` inside `pad_rect(pad)` but outside the
    /// content region; its source is found by mapping the block-local
    /// coordinate through [`gutter_source_index`] on each axis. Under
    /// [`PadMode::Transparent`] every padding `texel` resolves to the sentinel
    /// on at least one axis and is skipped, yielding an empty list. Returns an
    /// empty list for a zero-area block or `pad == 0`.
    #[must_use]
    pub fn border_copy_map(&self, pad: u32, mode: PadMode) -> Vec<(u32, u32, u32, u32)> {
        let mut out = Vec::new();
        if self.w == 0 || self.h == 0 || pad == 0 {
            return out;
        }

        let padded = self.pad_rect(pad);
        let content_x = self.x..self.right();
        let content_y = self.y..self.bottom();

        for dy in padded.y..padded.bottom() {
            let inside_y = content_y.contains(&dy);
            for dx in padded.x..padded.right() {
                if inside_y && content_x.contains(&dx) {
                    continue;
                }
                let lx = i64::from(dx) - i64::from(self.x);
                let ly = i64::from(dy) - i64::from(self.y);
                let lxi = i32::try_from(lx).unwrap_or(i32::MAX);
                let lyi = i32::try_from(ly).unwrap_or(i32::MAX);
                let sx = gutter_source_index(lxi, self.w, mode);
                let sy = gutter_source_index(lyi, self.h, mode);
                if sx == u32::MAX || sy == u32::MAX {
                    continue;
                }
                out.push((dx, dy, self.x.saturating_add(sx), self.y.saturating_add(sy)));
            }
        }
        out
    }
}

/// The gutter width, in `texel`s, needed so a `mipmap` chain of `mip_levels`
/// levels never averages across a block boundary.
///
/// The coarsest level averages a `2^(mip_levels - 1)`-`texel` footprint, so the
/// gutter is that half-footprint: `1 << (mip_levels - 1)`. `mip_levels == 0`
/// (no chain) needs no gutter and returns `0`; `mip_levels == 1` (base level
/// only) still returns `1` for the one-`texel` bilinear seam. The shift is
/// clamped to 31 bits so an absurd level count saturates instead of panicking.
#[must_use]
pub fn required_padding(mip_levels: u32) -> u32 {
    if mip_levels == 0 {
        return 0;
    }
    let shift = (mip_levels - 1).min(31);
    1u32 << shift
}

/// The largest `mipmap` level count a block of `w` x `h` `texel`s can carry
/// before its smaller dimension shrinks below one `texel`.
///
/// This is `floor(log2(min(w, h))) + 1`, computed from
/// [`u32::leading_zeros`] rather than a logarithm: for a smaller dimension
/// `m >= 1` the level count is `32 - m.leading_zeros()`. A zero dimension is
/// degenerate and carries `0` levels.
#[must_use]
pub fn max_safe_mip_levels(w: u32, h: u32) -> u32 {
    let m = w.min(h);
    if m == 0 {
        return 0;
    }
    32 - m.leading_zeros()
}

/// Maps a block-local coordinate in `[-pad, size + pad)` back to a source
/// `texel` index in `[0, size)` under the given border [`PadMode`].
///
/// [`PadMode::ClampEdge`] clamps to the nearest edge; [`PadMode::Mirror`]
/// reflects across the edges without repeating them; [`PadMode::Transparent`]
/// returns the [`u32::MAX`] sentinel for any coordinate outside `[0, size)`. A
/// zero `size` is degenerate and always returns the sentinel.
#[must_use]
pub fn gutter_source_index(coord: i32, size: u32, mode: PadMode) -> u32 {
    if size == 0 {
        return u32::MAX;
    }
    let s = i32::try_from(size).unwrap_or(i32::MAX);
    match mode {
        PadMode::ClampEdge => u32::try_from(coord.clamp(0, s - 1)).unwrap_or(0),
        PadMode::Transparent => {
            if (0..s).contains(&coord) {
                u32::try_from(coord).unwrap_or(u32::MAX)
            } else {
                u32::MAX
            }
        }
        PadMode::Mirror => {
            let period = s.saturating_mul(2);
            let mut c = coord % period;
            if c < 0 {
                c += period;
            }
            if c >= s {
                c = period - 1 - c;
            }
            u32::try_from(c).unwrap_or(0)
        }
    }
}

/// Total byte size of the `GPU` `UV` buffer holding one padded-frame `UV` and
/// one content `UV` (`vec4<f32>` each) per block, clamped up to a single
/// element and saturating so a degenerate count never wraps.
#[must_use]
pub fn uv_buffer_bytes(rect_count: u32) -> u64 {
    let pair = u64::try_from(VEC4_STRIDE).unwrap_or(16).saturating_mul(2);
    pair.saturating_mul(u64::from(rect_count).max(1))
}

/// Total `std430` storage-buffer byte size for `count` blocks, reusing
/// [`storage_bytes`] with the two-`vec4` (padded-frame + content `UV`) stride.
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(VEC4_STRIDE.saturating_mul(2), count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= CMP_EPS
    }

    #[test]
    fn rect_new_stores_fields() {
        let r = Rect::new(3, 5, 7, 9);
        assert_eq!(
            r,
            Rect {
                x: 3,
                y: 5,
                w: 7,
                h: 9
            }
        );
    }

    #[test]
    fn rect_right_bottom_basic() {
        let r = Rect::new(10, 20, 30, 40);
        assert_eq!(r.right(), 40);
        assert_eq!(r.bottom(), 60);
    }

    #[test]
    fn rect_right_bottom_saturate() {
        let r = Rect::new(u32::MAX, u32::MAX, 5, 5);
        assert_eq!(r.right(), u32::MAX);
        assert_eq!(r.bottom(), u32::MAX);
    }

    #[test]
    fn required_padding_zero_needs_no_gutter() {
        assert_eq!(required_padding(0), 0);
    }

    #[test]
    fn required_padding_one_is_bilinear_seam() {
        assert_eq!(required_padding(1), 1);
    }

    #[test]
    fn required_padding_follows_half_footprint() {
        assert_eq!(required_padding(2), 2);
        assert_eq!(required_padding(3), 4);
        assert_eq!(required_padding(4), 8);
        assert_eq!(required_padding(5), 16);
        assert_eq!(required_padding(11), 1024);
    }

    #[test]
    fn required_padding_saturates_without_panic() {
        assert_eq!(required_padding(40), 1u32 << 31);
        assert_eq!(required_padding(u32::MAX), 1u32 << 31);
    }

    #[test]
    fn max_safe_mip_levels_powers_of_two() {
        assert_eq!(max_safe_mip_levels(1, 1), 1);
        assert_eq!(max_safe_mip_levels(2, 2), 2);
        assert_eq!(max_safe_mip_levels(4, 4), 3);
        assert_eq!(max_safe_mip_levels(8, 8), 4);
        assert_eq!(max_safe_mip_levels(1024, 1024), 11);
    }

    #[test]
    fn max_safe_mip_levels_non_powers_of_two() {
        assert_eq!(max_safe_mip_levels(3, 3), 2);
        assert_eq!(max_safe_mip_levels(5, 5), 3);
        assert_eq!(max_safe_mip_levels(7, 7), 3);
        assert_eq!(max_safe_mip_levels(1023, 1023), 10);
    }

    #[test]
    fn max_safe_mip_levels_uses_smaller_dimension() {
        assert_eq!(max_safe_mip_levels(1024, 4), 3);
        assert_eq!(max_safe_mip_levels(4, 1024), 3);
    }

    #[test]
    fn max_safe_mip_levels_zero_dimension() {
        assert_eq!(max_safe_mip_levels(0, 256), 0);
        assert_eq!(max_safe_mip_levels(256, 0), 0);
    }

    #[test]
    fn pad_rect_interior_grows_symmetrically() {
        let r = Rect::new(10, 20, 30, 40);
        let p = r.pad_rect(2);
        assert_eq!(p, Rect::new(8, 18, 34, 44));
        assert_eq!(p.right(), r.right() + 2);
        assert_eq!(p.bottom(), r.bottom() + 2);
    }

    #[test]
    fn pad_rect_clamps_origin_at_zero() {
        let r = Rect::new(1, 0, 4, 4);
        let p = r.pad_rect(2);
        assert_eq!(p.x, 0);
        assert_eq!(p.y, 0);
        // Right/bottom edges stay at x + w + pad and y + h + pad.
        assert_eq!(p.right(), r.right() + 2);
        assert_eq!(p.bottom(), r.bottom() + 2);
    }

    #[test]
    fn pad_rect_zero_pad_is_identity() {
        let r = Rect::new(7, 9, 11, 13);
        assert_eq!(r.pad_rect(0), r);
    }

    #[test]
    fn gutter_clamp_edge_inside_is_identity() {
        assert_eq!(gutter_source_index(0, 4, PadMode::ClampEdge), 0);
        assert_eq!(gutter_source_index(3, 4, PadMode::ClampEdge), 3);
    }

    #[test]
    fn gutter_clamp_edge_beyond_bounds() {
        assert_eq!(gutter_source_index(-1, 4, PadMode::ClampEdge), 0);
        assert_eq!(gutter_source_index(-9, 4, PadMode::ClampEdge), 0);
        assert_eq!(gutter_source_index(4, 4, PadMode::ClampEdge), 3);
        assert_eq!(gutter_source_index(99, 4, PadMode::ClampEdge), 3);
    }

    #[test]
    fn gutter_mirror_reflects_left_edge() {
        // size 4: -1 -> 0, -2 -> 1, -3 -> 2, -4 -> 3.
        assert_eq!(gutter_source_index(-1, 4, PadMode::Mirror), 0);
        assert_eq!(gutter_source_index(-2, 4, PadMode::Mirror), 1);
        assert_eq!(gutter_source_index(-3, 4, PadMode::Mirror), 2);
        assert_eq!(gutter_source_index(-4, 4, PadMode::Mirror), 3);
    }

    #[test]
    fn gutter_mirror_reflects_right_edge() {
        // size 4: 4 -> 3, 5 -> 2, 6 -> 1, 7 -> 0.
        assert_eq!(gutter_source_index(4, 4, PadMode::Mirror), 3);
        assert_eq!(gutter_source_index(5, 4, PadMode::Mirror), 2);
        assert_eq!(gutter_source_index(6, 4, PadMode::Mirror), 1);
        assert_eq!(gutter_source_index(7, 4, PadMode::Mirror), 0);
        // Inside stays put.
        assert_eq!(gutter_source_index(2, 4, PadMode::Mirror), 2);
    }

    #[test]
    fn gutter_transparent_sentinel_outside() {
        assert_eq!(gutter_source_index(2, 4, PadMode::Transparent), 2);
        assert_eq!(gutter_source_index(-1, 4, PadMode::Transparent), u32::MAX);
        assert_eq!(gutter_source_index(4, 4, PadMode::Transparent), u32::MAX);
    }

    #[test]
    fn gutter_zero_size_is_sentinel_for_all_modes() {
        assert_eq!(gutter_source_index(0, 0, PadMode::ClampEdge), u32::MAX);
        assert_eq!(gutter_source_index(0, 0, PadMode::Mirror), u32::MAX);
        assert_eq!(gutter_source_index(0, 0, PadMode::Transparent), u32::MAX);
    }

    #[test]
    fn build_padded_uv_zero_atlas_is_all_zero() {
        let r = Rect::new(10, 20, 30, 40);
        assert_eq!(r.build_padded_uv(0, 100, 2), ([0.0; 4], [0.0; 4]));
        assert_eq!(r.build_padded_uv(100, 0, 2), ([0.0; 4], [0.0; 4]));
    }

    #[test]
    fn build_padded_uv_frame_and_half_texel_inset() {
        let r = Rect::new(10, 20, 30, 40);
        let (outer, inner) = r.build_padded_uv(100, 100, 2);

        assert!(approx(outer[0], 0.08));
        assert!(approx(outer[1], 0.18));
        assert!(approx(outer[2], 0.42));
        assert!(approx(outer[3], 0.62));

        assert!(approx(inner[0], 0.105));
        assert!(approx(inner[1], 0.205));
        assert!(approx(inner[2], 0.395));
        assert!(approx(inner[3], 0.595));

        // Inner content sits strictly inside the padded frame.
        assert!(inner[0] > outer[0] && inner[1] > outer[1]);
        assert!(inner[2] < outer[2] && inner[3] < outer[3]);
    }

    #[test]
    fn build_padded_uv_degenerate_content_does_not_invert() {
        let r = Rect::new(10, 20, 0, 0);
        let (_outer, inner) = r.build_padded_uv(100, 100, 2);
        assert!(inner[2] >= inner[0]);
        assert!(inner[3] >= inner[1]);
    }

    #[test]
    fn border_copy_map_empty_for_zero_pad_or_empty_rect() {
        let r = Rect::new(5, 5, 4, 4);
        assert!(r.border_copy_map(0, PadMode::ClampEdge).is_empty());
        assert!(Rect::new(5, 5, 0, 4)
            .border_copy_map(1, PadMode::ClampEdge)
            .is_empty());
        assert!(Rect::new(5, 5, 4, 0)
            .border_copy_map(1, PadMode::ClampEdge)
            .is_empty());
    }

    #[test]
    fn border_copy_map_clamp_ring_count_and_corner_source() {
        let r = Rect::new(5, 5, 2, 2);
        let map = r.border_copy_map(1, PadMode::ClampEdge);
        // Padded region 4x4 minus 2x2 content = 12 ring texels.
        assert_eq!(map.len(), 12);
        // Top-left corner (4,4) clamps to the block's own (5,5).
        assert!(map.contains(&(4, 4, 5, 5)));
        // No destination falls inside the content region.
        for (dx, dy, sx, sy) in map {
            let in_content = (5..7).contains(&dx) && (5..7).contains(&dy);
            assert!(!in_content);
            // Every source is a real content texel.
            assert!((5..7).contains(&sx) && (5..7).contains(&sy));
        }
    }

    #[test]
    fn border_copy_map_transparent_is_empty_ring() {
        let r = Rect::new(5, 5, 4, 4);
        assert!(r.border_copy_map(2, PadMode::Transparent).is_empty());
    }

    #[test]
    fn border_copy_map_mirror_left_column_sources() {
        let r = Rect::new(10, 10, 4, 4);
        let map = r.border_copy_map(2, PadMode::Mirror);
        // Along the content's first row (dy = 10), the two left gutter columns
        // mirror the block's own left edge: dx=9 -> src x 10, dx=8 -> src x 11.
        assert!(map.contains(&(9, 10, 10, 10)));
        assert!(map.contains(&(8, 10, 11, 10)));
    }

    #[test]
    fn border_copy_map_clamps_origin_without_negative_coords() {
        let r = Rect::new(1, 1, 3, 3);
        let map = r.border_copy_map(2, PadMode::ClampEdge);
        // Origin clamps to 0; every coordinate is a valid u32 with a real source.
        assert!(!map.is_empty());
        for (_dx, _dy, sx, sy) in map {
            assert!((1..4).contains(&sx) && (1..4).contains(&sy));
        }
    }

    #[test]
    fn uv_buffer_bytes_reserves_one_and_scales() {
        assert_eq!(uv_buffer_bytes(0), 32);
        assert_eq!(uv_buffer_bytes(1), 32);
        assert_eq!(uv_buffer_bytes(10), 320);
    }

    #[test]
    fn uv_buffer_bytes_saturates() {
        assert_eq!(
            uv_buffer_bytes(u32::MAX),
            u64::from(u32::MAX).saturating_mul(32)
        );
    }

    #[test]
    fn gpu_storage_bytes_reuses_storage_bytes() {
        assert_eq!(gpu_storage_bytes(0), storage_bytes(VEC4_STRIDE * 2, 0));
        assert_eq!(gpu_storage_bytes(4), storage_bytes(VEC4_STRIDE * 2, 4));
        assert_eq!(gpu_storage_bytes(4), 128);
    }

    #[test]
    fn pad_mode_equates_and_hashes() {
        assert_eq!(PadMode::ClampEdge, PadMode::ClampEdge);
        assert_ne!(PadMode::ClampEdge, PadMode::Mirror);
    }
}
