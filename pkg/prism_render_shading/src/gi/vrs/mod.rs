//! Variable-rate shading (VRS) classification CPU golden references.
//!
//! Deterministic, GPU-free tile classifiers that pick a coarse shading rate
//! (1x1 … 4x4) per tile from image-space signals, mirroring console VRS tier-2
//! pipelines:
//!
//! * [`luma`] — luminance-variance / just-noticeable-difference classifier.
//! * [`edge`] — gradient (Sobel) contrast classifier that forbids coarsening
//!   across strong edges.
//! * [`motion`] — motion-magnitude classifier that coarsens fast-moving tiles
//!   (shading error is masked by motion blur).
//!
//! The common currency of all three is [`ShadingRate`], the tier-2 shading-rate
//! lattice (`1x1`, `1x2`, `2x1`, `2x2`, `2x4`, `4x2`, `4x4`).  Each classifier
//! independently proposes a rate from its own signal; the high-level
//! [`classify_tile`] combines the proposals by taking the **finest** (most
//! conservative) of them via [`motion::combine_rates`], so a tile is only
//! coarsened when *every* signal agrees it is safe to do so.
//!
//! # Conventions
//! * `no_std`-friendly: only `core`/`bevy_math` types are used; the tile
//!   samples are borrowed as `&[f32]`, so no allocation ever happens here.
//! * Pure, deterministic functions: no RNG, IO, GPU, or `unsafe`.
//! * Transcendental functions, when needed, route through [`bevy_math::ops`];
//!   plain `x.sqrt()` is used directly where that suffices.
//! * `-W unused-qualifications`-clean: types are referenced by their short
//!   names, never fully path-qualified.
//! * Defensive clamping is pervasive.  Degenerate input — an empty tile, a
//!   mis-sized tile, a non-finite sample, or a divide-by-zero — always falls
//!   back to the **finest** rate [`ShadingRate::X1x1`], because shading at full
//!   rate is never *wrong*, only slower; coarsening a tile that actually needed
//!   detail is the only visible failure, so the safe default is "shade
//!   everything".

use bevy_math::Vec2;

pub mod luma;
pub mod edge;
pub mod motion;

/// Tier-2 variable-rate-shading lattice.
///
/// A `ShadingRate` describes how many framebuffer pixels share one shaded
/// sample, as a `WIDTH x HEIGHT` footprint.  `X2x4`, for instance, means one
/// shaded sample covers a 2-wide, 4-tall block of pixels.  The seven variants
/// are exactly the set guaranteed by Direct3D 12 VRS *tier 2* / console
/// equivalents: both axes are drawn from `{1, 2, 4}`, excluding the two
/// forbidden `1x4` / `4x1` combinations (whose 4:1 aspect ratio is not
/// representable in tier-2 hardware).
///
/// Variants are named `X<width>x<height>`; the leading `X` keeps them valid
/// Rust identifiers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShadingRate {
    /// Full rate: one shaded sample per pixel (`1x1`). The finest, safest rate.
    X1x1,
    /// `1x2`: full horizontal rate, half vertical rate (coarsen vertically).
    X1x2,
    /// `2x1`: half horizontal rate, full vertical rate (coarsen horizontally).
    X2x1,
    /// `2x2`: half rate on both axes.
    X2x2,
    /// `2x4`: half horizontal, quarter vertical.
    X2x4,
    /// `4x2`: quarter horizontal, half vertical.
    X4x2,
    /// `4x4`: quarter rate on both axes. The coarsest tier-2 rate.
    X4x4,
}

impl ShadingRate {
    /// The finest (full-rate) shading rate, [`ShadingRate::X1x1`].
    pub const FINEST: Self = ShadingRate::X1x1;

    /// The coarsest tier-2 shading rate, [`ShadingRate::X4x4`].
    pub const COARSEST: Self = ShadingRate::X4x4;

    /// Horizontal shading-rate factor (how many pixels wide one sample covers).
    ///
    /// One of `1`, `2`, or `4`.
    #[inline]
    pub const fn x_rate(self) -> u8 {
        match self {
            ShadingRate::X1x1 | ShadingRate::X1x2 => 1,
            ShadingRate::X2x1 | ShadingRate::X2x2 | ShadingRate::X2x4 => 2,
            ShadingRate::X4x2 | ShadingRate::X4x4 => 4,
        }
    }

    /// Vertical shading-rate factor (how many pixels tall one sample covers).
    ///
    /// One of `1`, `2`, or `4`.
    #[inline]
    pub const fn y_rate(self) -> u8 {
        match self {
            ShadingRate::X1x1 | ShadingRate::X2x1 => 1,
            ShadingRate::X1x2 | ShadingRate::X2x2 | ShadingRate::X4x2 => 2,
            ShadingRate::X2x4 | ShadingRate::X4x4 => 4,
        }
    }

    /// `(x_rate, y_rate)` footprint of this rate.
    #[inline]
    pub const fn footprint(self) -> (u8, u8) {
        (self.x_rate(), self.y_rate())
    }

    /// Number of pixels one shaded sample covers (`x_rate * y_rate`).
    ///
    /// Ranges from `1` (`X1x1`) to `16` (`X4x4`); the shading-work reduction
    /// factor relative to full rate.
    #[inline]
    pub const fn area(self) -> u32 {
        (self.x_rate() as u32) * (self.y_rate() as u32)
    }

    /// Total-order *coarseness rank*, `0` (finest) … `6` (coarsest).
    ///
    /// Unlike [`area`](Self::area) — under which `X2x4` and `X4x2` tie at `8` —
    /// the rank is a strict total order, so it gives a deterministic tie-break
    /// when combining rates.  The order is
    /// `X1x1 < X1x2 < X2x1 < X2x2 < X2x4 < X4x2 < X4x4`.
    #[inline]
    pub const fn rank(self) -> u8 {
        match self {
            ShadingRate::X1x1 => 0,
            ShadingRate::X1x2 => 1,
            ShadingRate::X2x1 => 2,
            ShadingRate::X2x2 => 3,
            ShadingRate::X2x4 => 4,
            ShadingRate::X4x2 => 5,
            ShadingRate::X4x4 => 6,
        }
    }

    /// Returns the **finer** (lower-rank) of `self` and `other`.
    ///
    /// This is the conservative combine used across signals: whichever
    /// classifier demands the most detail wins.  Ties resolve to `self`.
    #[inline]
    pub const fn finer_of(self, other: Self) -> Self {
        if self.rank() <= other.rank() { self } else { other }
    }

    /// Returns the **coarser** (higher-rank) of `self` and `other`.
    #[inline]
    pub const fn coarser_of(self, other: Self) -> Self {
        if self.rank() >= other.rank() { self } else { other }
    }

    /// Snaps an arbitrary per-axis coarsening factor to the nearest legal
    /// tier-2 axis factor `{1, 2, 4}`.
    ///
    /// `0` and `1` map to `1`; `2` and `3` map to `2`; everything `>= 4` maps
    /// to `4`.  (The `3 -> 2` tie-break favours the *finer* factor, matching
    /// the module's "when in doubt, shade more" policy.)
    #[inline]
    const fn snap_axis(factor: u32) -> u8 {
        if factor <= 1 {
            1
        } else if factor <= 3 {
            2
        } else {
            4
        }
    }

    /// Builds a tier-2 rate from desired per-axis coarsening factors.
    ///
    /// Each factor is first snapped to `{1, 2, 4}` via [`snap_axis`].  The two
    /// combinations unrepresentable in tier-2, `1x4` and `4x1`, are nudged to
    /// their nearest legal neighbours `2x4` and `4x2` respectively (the coarse
    /// axis is preserved and the fine axis is softened from `1` to `2`, since
    /// the 4:1 anisotropy is what the hardware disallows).
    ///
    /// This is the anisotropy-aware constructor the directional [`edge`] and
    /// [`motion`] classifiers use to turn per-axis signal strength into a rate.
    #[inline]
    pub const fn from_factors(x_factor: u32, y_factor: u32) -> Self {
        let x = Self::snap_axis(x_factor);
        let y = Self::snap_axis(y_factor);
        // Repair the two tier-2-forbidden combinations.
        let (x, y) = match (x, y) {
            (1, 4) => (2, 4),
            (4, 1) => (4, 2),
            other => other,
        };
        match (x, y) {
            (1, 1) => ShadingRate::X1x1,
            (1, 2) => ShadingRate::X1x2,
            (2, 1) => ShadingRate::X2x1,
            (2, 2) => ShadingRate::X2x2,
            (2, 4) => ShadingRate::X2x4,
            (4, 2) => ShadingRate::X4x2,
            (4, 4) => ShadingRate::X4x4,
            // Unreachable after snapping+repair; stay safe at full rate.
            _ => ShadingRate::X1x1,
        }
    }
}

impl Default for ShadingRate {
    /// The safe default is the finest rate: shade everything.
    #[inline]
    fn default() -> Self {
        ShadingRate::X1x1
    }
}

/// Aggregated thresholds for the three VRS classifiers.
///
/// [`Default`] yields a balanced, perceptually motivated configuration suitable
/// for an SDR 1080p/1440p target; tune per-field for other displays or quality
/// bars.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct VrsConfig {
    /// Luminance / JND classifier thresholds.
    pub luma: luma::LumaThresholds,
    /// Sobel edge classifier thresholds.
    pub edge: edge::EdgeThresholds,
    /// Motion-magnitude classifier thresholds.
    pub motion: motion::MotionThresholds,
}

impl Default for VrsConfig {
    #[inline]
    fn default() -> Self {
        Self {
            luma: luma::LumaThresholds::default(),
            edge: edge::EdgeThresholds::default(),
            motion: motion::MotionThresholds::default(),
        }
    }
}

/// Classifies a tile by running all three signal classifiers and combining
/// their proposals into the single safe rate.
///
/// * `luma` is the row-major tile of per-pixel luminances, `width * height`
///   long.
/// * `width` / `height` are the tile dimensions in pixels.
/// * `velocity` is the tile's average screen-space motion vector in
///   pixels/frame.
/// * `cfg` carries the per-classifier thresholds.
///
/// The three proposals are merged with [`motion::combine_rates`], i.e. the
/// **finest** proposal wins.  Any degenerate input makes an individual
/// classifier fall back to [`ShadingRate::X1x1`], which — being the finest —
/// then forces the combined result to full rate as well.
pub fn classify_tile(
    luma: &[f32],
    width: usize,
    height: usize,
    velocity: Vec2,
    cfg: &VrsConfig,
) -> ShadingRate {
    let luma_rate = luma::classify_luma(luma, &cfg.luma);
    let edge_rate = edge::classify_edge(luma, width, height, &cfg.edge);
    let motion_rate = motion::classify_motion_vec(velocity, &cfg.motion);
    motion::combine_rates(luma_rate, motion::combine_rates(edge_rate, motion_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn footprints_and_areas_are_consistent() {
        assert_eq!(ShadingRate::X1x1.footprint(), (1, 1));
        assert_eq!(ShadingRate::X2x4.footprint(), (2, 4));
        assert_eq!(ShadingRate::X4x2.footprint(), (4, 2));
        assert_eq!(ShadingRate::X1x1.area(), 1);
        assert_eq!(ShadingRate::X2x2.area(), 4);
        assert_eq!(ShadingRate::X2x4.area(), 8);
        assert_eq!(ShadingRate::X4x2.area(), 8);
        assert_eq!(ShadingRate::X4x4.area(), 16);
    }

    #[test]
    fn rank_is_a_strict_total_order() {
        let all = [
            ShadingRate::X1x1,
            ShadingRate::X1x2,
            ShadingRate::X2x1,
            ShadingRate::X2x2,
            ShadingRate::X2x4,
            ShadingRate::X4x2,
            ShadingRate::X4x4,
        ];
        // Ranks are exactly 0..7, each used once, and area is non-decreasing.
        for (i, r) in all.iter().enumerate() {
            assert_eq!(r.rank() as usize, i);
        }
        for pair in all.windows(2) {
            assert!(pair[0].rank() < pair[1].rank());
            assert!(pair[0].area() <= pair[1].area());
        }
    }

    #[test]
    fn finer_and_coarser_pick_the_right_side() {
        assert_eq!(ShadingRate::X1x1.finer_of(ShadingRate::X4x4), ShadingRate::X1x1);
        assert_eq!(ShadingRate::X4x4.finer_of(ShadingRate::X2x2), ShadingRate::X2x2);
        assert_eq!(ShadingRate::X1x1.coarser_of(ShadingRate::X4x4), ShadingRate::X4x4);
        // Equal rank resolves to `self`.
        assert_eq!(ShadingRate::X2x4.finer_of(ShadingRate::X2x4), ShadingRate::X2x4);
    }

    #[test]
    fn from_factors_snaps_and_repairs() {
        assert_eq!(ShadingRate::from_factors(1, 1), ShadingRate::X1x1);
        assert_eq!(ShadingRate::from_factors(3, 2), ShadingRate::X2x2);
        assert_eq!(ShadingRate::from_factors(9, 4), ShadingRate::X4x4);
        // Forbidden 1x4 / 4x1 are nudged to the nearest legal rate.
        assert_eq!(ShadingRate::from_factors(1, 4), ShadingRate::X2x4);
        assert_eq!(ShadingRate::from_factors(4, 1), ShadingRate::X4x2);
        // Zero factor behaves like full rate on that axis.
        assert_eq!(ShadingRate::from_factors(0, 2), ShadingRate::X1x2);
    }

    #[test]
    fn classify_tile_takes_the_finest_signal() {
        let cfg = VrsConfig::default();
        // Finest-combine means motion is a *permission* to coarsen: a still
        // tile demands full detail (no motion blur to hide error), so even a
        // perfectly flat still tile stays at full rate.
        let flat = [0.5_f32; 16];
        let still_rate = classify_tile(&flat, 4, 4, Vec2::ZERO, &cfg);
        assert_eq!(still_rate, ShadingRate::X1x1, "still tile shades fully");

        // The same flat tile, now moving fast, may coarsen because luma and
        // edge agree it is featureless and motion blur hides the error.
        let fast = Vec2::new(64.0, 64.0);
        let moving_rate = classify_tile(&flat, 4, 4, fast, &cfg);
        assert!(moving_rate.area() > 1, "flat fast tile should coarsen");

        // A strong internal edge must keep the tile fine regardless of motion:
        // edge detail is not masked across the silhouette.
        let edged = [
            0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 1.0, 1.0_f32,
        ];
        let edged_rate = classify_tile(&edged, 4, 4, fast, &cfg);
        assert_eq!(edged_rate, ShadingRate::X1x1);
    }

    #[test]
    fn classify_tile_degenerate_input_is_full_rate() {
        let cfg = VrsConfig::default();
        assert_eq!(classify_tile(&[], 0, 0, Vec2::ZERO, &cfg), ShadingRate::X1x1);
        let nan = [f32::NAN; 16];
        assert_eq!(
            classify_tile(&nan, 4, 4, Vec2::ZERO, &cfg),
            ShadingRate::X1x1
        );
        // Mis-sized tile (len != w*h) is also degenerate.
        let small = [0.5_f32; 9];
        assert_eq!(
            classify_tile(&small, 4, 4, Vec2::ZERO, &cfg),
            ShadingRate::X1x1
        );
    }
}
