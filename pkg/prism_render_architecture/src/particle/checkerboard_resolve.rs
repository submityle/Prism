//! Checkerboard-rendering *resolve*: the pure-`CPU` gold standard that weaves a
//! half-density checkerboard sample pattern back into a full-resolution image
//! (design §21).
//!
//! Checkerboard rendering shades only half of the pixels each frame — the cells
//! of one color of a checkerboard — and reconstructs the other half from the
//! previous frame plus the current frame's spatial neighborhood. A pixel is
//! *shaded this frame* when its parity `(x + y + frame) & 1` selects the color
//! being rendered; the complementary cells are *missing* and must be rebuilt.
//! Because every orthogonal neighbor of a missing cell has the opposite parity,
//! all four of its up / down / left / right neighbors are freshly shaded, which
//! is exactly what makes the spatial fill well posed.
//!
//! This module owns four independent pieces the resolve pass combines:
//!
//! 1. **Parity weave** — [`pixel_parity`] / [`is_current_sample`] decide which
//!    cells are shaded this frame, and [`CheckerboardResolution`] counts full,
//!    shaded, and missing pixels and reports the `std430` byte sizes of the
//!    color and coverage-mask storage buffers.
//! 2. **Spatial fill** — for a missing cell, [`resolve_spatial`] averages its
//!    in-bounds orthogonal neighbors with an *edge-aware* choice of axis: it
//!    fills along the direction of the smaller luma gradient (using a
//!    `smoothstep` blend) so a hard edge is interpolated along it, not across
//!    it. Border cells clamp to whichever neighbors exist.
//! 3. **Neighborhood box clamp** — a supplied history color (already
//!    reprojected upstream, if at all) is clamped component-wise into the
//!    min/max box of the missing cell's known neighbors. This is the
//!    anti-ghosting guard: stale history that drifts outside the local range is
//!    pulled back before it is trusted.
//! 4. **Resolve** — [`resolve`] weaves shaded cells through verbatim and, for
//!    missing cells, blends the edge-aware spatial fill against the
//!    box-clamped history by a configurable weight.
//!
//! **Scope.** This is strictly *spatial* checkerboard reconstruction plus a
//! local box clamp. It deliberately does **not** reproject history along motion
//! vectors, nor does it do the sharper `YCoCg` line clip — that belongs to
//! [`crate::particle::temporal_reprojection`], whose already-reprojected output
//! this module happily consumes as its history input. It is likewise unrelated
//! to the ordered dither of [`crate::particle::temporal_dither`] and to motion
//! blur. Every value is produced with integer arithmetic, `f32` `min` / `max` /
//! `clamp`, a single `sqrt`-free `smoothstep`, and no transcendental functions,
//! so this reference is portable to a future `GPU` kernel bit for bit. `RGBA`
//! colors are laid out as `std430` `vec4<f32>` and the coverage mask as `u32`.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, U32_STRIDE, VEC4_STRIDE};

/// Gradient magnitude below which the two fill axes are treated as equally
/// smooth, so the spatial fill falls back to their plain mean instead of
/// dividing by a near-zero denominator.
const GRAD_EPS: f32 = 1.0e-6;

/// Rec.601 luma weight for the red channel (used only for the edge-aware axis
/// decision, never for output color).
const LUMA_R: f32 = 0.299;
/// Rec.601 luma weight for the green channel.
const LUMA_G: f32 = 0.587;
/// Rec.601 luma weight for the blue channel.
const LUMA_B: f32 = 0.114;

/// Widens a `u32` extent or index to `usize` for buffer arithmetic.
///
/// Every caller passes an image width, height, or coordinate, all of which fit
/// in a `usize` on the 32-bit and 64-bit targets this crate supports.
fn to_usize(v: u32) -> usize {
    usize::try_from(v).expect("u32 image extent fits in usize on supported targets")
}

/// A straight-alpha `RGBA` color stored as four `f32` channels (`std430`
/// `vec4<f32>` layout).
#[derive(Clone, Copy, Debug)]
pub struct Rgba {
    /// Red channel.
    pub r: f32,
    /// Green channel.
    pub g: f32,
    /// Blue channel.
    pub b: f32,
    /// Alpha channel.
    pub a: f32,
}

impl Rgba {
    /// Builds a color from its four channels.
    #[must_use]
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Builds a color whose four channels all equal `v`.
    #[must_use]
    pub const fn splat(v: f32) -> Self {
        Self {
            r: v,
            g: v,
            b: v,
            a: v,
        }
    }

    /// Component-wise sum (kept private and named to avoid shadowing
    /// `core::ops::Add`).
    fn combined(self, other: Self) -> Self {
        Self {
            r: self.r + other.r,
            g: self.g + other.g,
            b: self.b + other.b,
            a: self.a + other.a,
        }
    }

    /// Component-wise scale by a scalar.
    fn scaled(self, s: f32) -> Self {
        Self {
            r: self.r * s,
            g: self.g * s,
            b: self.b * s,
            a: self.a * s,
        }
    }

    /// Linear interpolation toward `other` by `t` (component-wise `lerp`).
    fn mix(self, other: Self, t: f32) -> Self {
        let inv = 1.0 - t;
        Self {
            r: self.r * inv + other.r * t,
            g: self.g * inv + other.g * t,
            b: self.b * inv + other.b * t,
            a: self.a * inv + other.a * t,
        }
    }

    /// Component-wise minimum.
    fn min_with(self, other: Self) -> Self {
        Self {
            r: self.r.min(other.r),
            g: self.g.min(other.g),
            b: self.b.min(other.b),
            a: self.a.min(other.a),
        }
    }

    /// Component-wise maximum.
    fn max_with(self, other: Self) -> Self {
        Self {
            r: self.r.max(other.r),
            g: self.g.max(other.g),
            b: self.b.max(other.b),
            a: self.a.max(other.a),
        }
    }

    /// Clamps each channel into the inclusive box `[lo, hi]` (component-wise).
    ///
    /// Callers always pass `lo` and `hi` derived from a component-wise
    /// min/max, so `lo <= hi` holds on every channel.
    fn clamp_box(self, lo: Self, hi: Self) -> Self {
        Self {
            r: self.r.clamp(lo.r, hi.r),
            g: self.g.clamp(lo.g, hi.g),
            b: self.b.clamp(lo.b, hi.b),
            a: self.a.clamp(lo.a, hi.a),
        }
    }

    /// Rec.601 luma, used only to compare edge strength between the two fill
    /// axes.
    fn luma(self) -> f32 {
        LUMA_R * self.r + LUMA_G * self.g + LUMA_B * self.b
    }
}

/// The checkerboard cell parity `(x + y + frame) & 1`.
///
/// Cells whose parity selects the color rendered this frame are *shaded*; the
/// complementary cells are *missing* and rebuilt by the resolve.
#[must_use]
pub const fn pixel_parity(x: u32, y: u32, frame: u32) -> u32 {
    x.wrapping_add(y).wrapping_add(frame) & 1
}

/// Whether pixel `(x, y)` is shaded this `frame` (parity `0`) rather than
/// rebuilt.
#[must_use]
pub const fn is_current_sample(x: u32, y: u32, frame: u32) -> bool {
    pixel_parity(x, y, frame) == 0
}

/// A `smoothstep` in `[0, 1]`: `t * t * (3 - 2t)` after clamping `t`.
///
/// This is the only easing used by the edge-aware fill; it needs no
/// transcendental functions.
fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The full-resolution dimensions a checkerboard pass reconstructs.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CheckerboardResolution {
    /// Full-resolution width in pixels.
    pub full_width: u32,
    /// Full-resolution height in pixels.
    pub full_height: u32,
}

impl CheckerboardResolution {
    /// Builds a full-resolution descriptor.
    #[must_use]
    pub const fn new(full_width: u32, full_height: u32) -> Self {
        Self {
            full_width,
            full_height,
        }
    }

    /// The number of pixels in the full-resolution image (`width * height`).
    #[must_use]
    pub fn full_pixel_count(self) -> usize {
        to_usize(self.full_width) * to_usize(self.full_height)
    }

    /// The width of the packed *shaded* half-density buffer, `ceil(width / 2)`.
    ///
    /// A checkerboard pass shades one cell per `2x1` span of each row, so the
    /// shaded samples pack into a buffer this wide by the full height.
    #[must_use]
    pub fn shaded_width(self) -> u32 {
        self.full_width.div_ceil(2)
    }

    /// The number of shaded samples stored per frame
    /// (`shaded_width * height`).
    #[must_use]
    pub fn shaded_pixel_count(self) -> usize {
        to_usize(self.shaded_width()) * to_usize(self.full_height)
    }

    /// The count of cells actually shaded on `frame` (parity `0`).
    ///
    /// For an even-area image this is exactly half the pixels for every frame;
    /// for an odd area the two parities differ by one and swap as the frame
    /// parity flips.
    #[must_use]
    pub fn samples_this_frame(self, frame: u32) -> usize {
        let even_x = to_usize(self.full_width.div_ceil(2));
        let odd_x = to_usize(self.full_width / 2);
        // Rows where `(y + frame)` is even need an even `x` to reach parity 0.
        let rows_same = if frame & 1 == 0 {
            self.full_height.div_ceil(2)
        } else {
            self.full_height / 2
        };
        let rows_diff = self.full_height - rows_same;
        to_usize(rows_same) * even_x + to_usize(rows_diff) * odd_x
    }

    /// The count of cells missing on `frame` (rebuilt by the resolve).
    #[must_use]
    pub fn missing_this_frame(self, frame: u32) -> usize {
        self.full_pixel_count() - self.samples_this_frame(frame)
    }

    /// The `std430` byte size of a full-resolution `RGBA` (`vec4<f32>`) buffer.
    #[must_use]
    pub fn resolved_bytes(self) -> usize {
        storage_bytes(VEC4_STRIDE, self.full_pixel_count())
    }

    /// The `std430` byte size of the packed shaded `RGBA` (`vec4<f32>`) buffer.
    #[must_use]
    pub fn shaded_bytes(self) -> usize {
        storage_bytes(VEC4_STRIDE, self.shaded_pixel_count())
    }

    /// The `std430` byte size of a full-resolution `u32` coverage mask.
    #[must_use]
    pub fn coverage_mask_bytes(self) -> usize {
        storage_bytes(U32_STRIDE, self.full_pixel_count())
    }
}

/// How the resolve blends spatial fill against box-clamped history for missing
/// cells.
#[derive(Clone, Copy, Debug)]
pub struct CheckerboardConfig {
    /// Trust placed in the box-clamped history versus the spatial fill, in
    /// `[0, 1]`. `0` is a pure spatial rebuild; `1` uses the clamped history.
    pub history_weight: f32,
}

impl CheckerboardConfig {
    /// Builds a configuration, clamping `history_weight` into `[0, 1]`.
    #[must_use]
    pub fn new(history_weight: f32) -> Self {
        Self {
            history_weight: history_weight.clamp(0.0, 1.0),
        }
    }
}

impl Default for CheckerboardConfig {
    fn default() -> Self {
        Self::new(1.0)
    }
}

/// Flat row-major index of pixel `(x, y)`.
fn index(res: CheckerboardResolution, x: u32, y: u32) -> usize {
    to_usize(y) * to_usize(res.full_width) + to_usize(x)
}

/// Reads the color at `(x, y)` from a full-resolution buffer.
fn sample(buf: &[Rgba], res: CheckerboardResolution, x: u32, y: u32) -> Rgba {
    buf[index(res, x, y)]
}

/// The four in-bounds orthogonal neighbors `[left, right, up, down]`.
///
/// For a missing cell every present neighbor is a shaded sample, so callers may
/// treat each `Some` as known-good data.
fn orthogonal_neighbors(
    buf: &[Rgba],
    res: CheckerboardResolution,
    x: u32,
    y: u32,
) -> [Option<Rgba>; 4] {
    let left = if x > 0 {
        Some(sample(buf, res, x - 1, y))
    } else {
        None
    };
    let right = if x + 1 < res.full_width {
        Some(sample(buf, res, x + 1, y))
    } else {
        None
    };
    let up = if y > 0 {
        Some(sample(buf, res, x, y - 1))
    } else {
        None
    };
    let down = if y + 1 < res.full_height {
        Some(sample(buf, res, x, y + 1))
    } else {
        None
    };
    [left, right, up, down]
}

/// The mean of an opposed neighbor pair and the luma gradient across it, when
/// both members are present.
fn pair_axis(a: Option<Rgba>, b: Option<Rgba>) -> Option<(Rgba, f32)> {
    match (a, b) {
        (Some(a), Some(b)) => {
            let mean = a.mix(b, 0.5);
            let grad = (a.luma() - b.luma()).abs();
            Some((mean, grad))
        }
        _ => None,
    }
}

/// Chooses between the horizontal and vertical fill by leaning toward the axis
/// of smaller luma gradient.
///
/// A large horizontal gradient (a vertical edge) drives the blend toward the
/// vertical mean, so the fill interpolates along the edge instead of smearing
/// across it.
fn edge_aware(h_mean: Rgba, h_grad: f32, v_mean: Rgba, v_grad: f32) -> Rgba {
    let denom = h_grad + v_grad;
    if denom < GRAD_EPS {
        return h_mean.mix(v_mean, 0.5);
    }
    let lean_vertical = smoothstep(h_grad / denom);
    h_mean.mix(v_mean, lean_vertical)
}

/// Averages whatever single neighbors exist when no opposed pair is complete
/// (a border or corner cell); falls back to the cell's own stored value when it
/// is fully isolated.
fn corner_fill(neighbors: [Option<Rgba>; 4], own: Rgba) -> Rgba {
    let mut acc = Rgba::splat(0.0);
    let mut count = 0u32;
    for c in neighbors.into_iter().flatten() {
        acc = acc.combined(c);
        count += 1;
    }
    match count {
        0 => own,
        1 => acc,
        2 => acc.scaled(0.5),
        _ => acc.scaled(1.0 / 3.0),
    }
}

/// The edge-aware spatial fill for a missing cell at `(x, y)`.
fn spatial_fill(buf: &[Rgba], res: CheckerboardResolution, x: u32, y: u32) -> Rgba {
    let neighbors = orthogonal_neighbors(buf, res, x, y);
    let [left, right, up, down] = neighbors;
    let horizontal = pair_axis(left, right);
    let vertical = pair_axis(up, down);
    match (horizontal, vertical) {
        (Some((h_mean, h_grad)), Some((v_mean, v_grad))) => {
            edge_aware(h_mean, h_grad, v_mean, v_grad)
        }
        (Some((h_mean, _)), None) => h_mean,
        (None, Some((v_mean, _))) => v_mean,
        (None, None) => corner_fill(neighbors, sample(buf, res, x, y)),
    }
}

/// The min/max box of a missing cell's known neighbors, or `(fallback,
/// fallback)` when it is fully isolated.
fn neighbor_box(
    buf: &[Rgba],
    res: CheckerboardResolution,
    x: u32,
    y: u32,
    fallback: Rgba,
) -> (Rgba, Rgba) {
    let mut lo: Option<Rgba> = None;
    let mut hi: Option<Rgba> = None;
    for c in orthogonal_neighbors(buf, res, x, y).into_iter().flatten() {
        lo = Some(match lo {
            Some(l) => l.min_with(c),
            None => c,
        });
        hi = Some(match hi {
            Some(h) => h.max_with(c),
            None => c,
        });
    }
    match (lo, hi) {
        (Some(l), Some(h)) => (l, h),
        _ => (fallback, fallback),
    }
}

/// Reconstructs the full-resolution image from this frame's shaded cells and a
/// history buffer, weaving shaded cells through and rebuilding missing cells.
///
/// `current` holds the shaded samples (only cells where [`is_current_sample`]
/// is true are read); `history` is the previous resolved frame, already
/// reprojected upstream if the pipeline reprojects at all. Both must contain
/// exactly [`CheckerboardResolution::full_pixel_count`] pixels in row-major
/// order. Each missing cell blends its edge-aware spatial fill against the
/// neighbor-box-clamped history by `config.history_weight`.
#[must_use]
pub fn resolve(
    res: CheckerboardResolution,
    current: &[Rgba],
    history: &[Rgba],
    frame: u32,
    config: CheckerboardConfig,
) -> Vec<Rgba> {
    let weight = config.history_weight.clamp(0.0, 1.0);
    let mut out = Vec::with_capacity(res.full_pixel_count());
    for y in 0..res.full_height {
        for x in 0..res.full_width {
            if is_current_sample(x, y, frame) {
                out.push(sample(current, res, x, y));
            } else {
                let fill = spatial_fill(current, res, x, y);
                let (lo, hi) = neighbor_box(current, res, x, y, fill);
                let clamped = sample(history, res, x, y).clamp_box(lo, hi);
                out.push(fill.mix(clamped, weight));
            }
        }
    }
    out
}

/// Reconstructs the full-resolution image using only this frame's shaded cells
/// and the spatial fill, with no history input.
///
/// `current` must contain exactly [`CheckerboardResolution::full_pixel_count`]
/// pixels in row-major order.
#[must_use]
pub fn resolve_spatial(res: CheckerboardResolution, current: &[Rgba], frame: u32) -> Vec<Rgba> {
    let mut out = Vec::with_capacity(res.full_pixel_count());
    for y in 0..res.full_height {
        for x in 0..res.full_width {
            if is_current_sample(x, y, frame) {
                out.push(sample(current, res, x, y));
            } else {
                out.push(spatial_fill(current, res, x, y));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for `f32` equality assertions in this module's tests.
    #[cfg(test)]
    const CMP_EPS: f32 = 1.0e-6;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    fn approx_rgba(a: Rgba, b: Rgba) -> bool {
        approx(a.r, b.r) && approx(a.g, b.g) && approx(a.b, b.b) && approx(a.a, b.a)
    }

    fn gray(v: f32) -> Rgba {
        Rgba::splat(v)
    }

    #[test]
    fn pixel_parity_matches_the_formula() {
        assert_eq!(pixel_parity(0, 0, 0), 0);
        assert_eq!(pixel_parity(1, 0, 0), 1);
        assert_eq!(pixel_parity(0, 1, 0), 1);
        assert_eq!(pixel_parity(1, 1, 0), 0);
        assert_eq!(pixel_parity(1, 1, 1), 1);
        assert_eq!(pixel_parity(2, 3, 4), (2 + 3 + 4) & 1);
    }

    #[test]
    fn is_current_sample_flips_with_frame_parity() {
        for x in 0..5u32 {
            for y in 0..5u32 {
                assert_ne!(
                    is_current_sample(x, y, 0),
                    is_current_sample(x, y, 1),
                    "cell ({x},{y}) must flip between consecutive frames"
                );
            }
        }
    }

    #[test]
    fn single_frame_coverage_is_exactly_half_on_even_area() {
        let res = CheckerboardResolution::new(4, 4);
        for frame in 0..4u32 {
            assert_eq!(res.samples_this_frame(frame), 8);
        }
        assert_eq!(res.full_pixel_count(), 16);
    }

    #[test]
    fn odd_area_coverage_differs_by_one_and_swaps_with_frame() {
        let res = CheckerboardResolution::new(3, 3);
        assert_eq!(res.full_pixel_count(), 9);
        assert_eq!(res.samples_this_frame(0), 5);
        assert_eq!(res.samples_this_frame(1), 4);
        // The two parities always partition the whole image.
        assert_eq!(res.samples_this_frame(0) + res.samples_this_frame(1), 9);
    }

    #[test]
    fn samples_this_frame_matches_a_brute_force_count() {
        for w in 1..7u32 {
            for h in 1..7u32 {
                let res = CheckerboardResolution::new(w, h);
                for frame in 0..3u32 {
                    let mut brute = 0usize;
                    for y in 0..h {
                        for x in 0..w {
                            if is_current_sample(x, y, frame) {
                                brute += 1;
                            }
                        }
                    }
                    assert_eq!(res.samples_this_frame(frame), brute, "{w}x{h}@{frame}");
                }
            }
        }
    }

    #[test]
    fn missing_count_complements_samples() {
        let res = CheckerboardResolution::new(5, 3);
        for frame in 0..4u32 {
            assert_eq!(
                res.samples_this_frame(frame) + res.missing_this_frame(frame),
                res.full_pixel_count()
            );
        }
    }

    #[test]
    fn two_consecutive_frames_partition_every_pixel() {
        let res = CheckerboardResolution::new(5, 3);
        for y in 0..res.full_height {
            for x in 0..res.full_width {
                let a = is_current_sample(x, y, 7);
                let b = is_current_sample(x, y, 8);
                // Exactly one of two consecutive frames shades each cell.
                assert!(a ^ b, "cell ({x},{y}) not covered exactly once");
            }
        }
    }

    #[test]
    fn uniform_neighbors_fill_to_their_shared_value() {
        // 3x3, frame 1 => center (1,1) is missing (parity 1).
        let res = CheckerboardResolution::new(3, 3);
        assert!(!is_current_sample(1, 1, 1));
        let c = gray(0.4);
        let mut buf = alloc::vec![gray(0.0); res.full_pixel_count()];
        buf[index(res, 0, 1)] = c; // left
        buf[index(res, 2, 1)] = c; // right
        buf[index(res, 1, 0)] = c; // up
        buf[index(res, 1, 2)] = c; // down
        let filled = spatial_fill(&buf, res, 1, 1);
        assert!(approx_rgba(filled, c));
    }

    #[test]
    fn edge_aware_fill_follows_the_low_gradient_axis() {
        // Vertical edge across the horizontal pair: left/right differ strongly,
        // up/down agree, so the fill must land on the vertical mean.
        let res = CheckerboardResolution::new(3, 3);
        let mut buf = alloc::vec![gray(0.0); res.full_pixel_count()];
        buf[index(res, 0, 1)] = gray(0.0); // left
        buf[index(res, 2, 1)] = gray(1.0); // right (big horizontal gradient)
        buf[index(res, 1, 0)] = gray(0.5); // up
        buf[index(res, 1, 2)] = gray(0.5); // down (zero vertical gradient)
        let filled = spatial_fill(&buf, res, 1, 1);
        assert!(approx_rgba(filled, gray(0.5)));
        // And it must not be the naive four-neighbor mean (which is also 0.5
        // here), so flip the vertical mean to prove the axis choice.
        let mut buf2 = buf.clone();
        buf2[index(res, 1, 0)] = gray(0.2);
        buf2[index(res, 1, 2)] = gray(0.2);
        let filled2 = spatial_fill(&buf2, res, 1, 1);
        assert!(approx_rgba(filled2, gray(0.2)));
    }

    #[test]
    fn border_corner_fill_uses_only_available_neighbors() {
        // 2x2, frame 1 => (0,0) is missing; only right (1,0) and down (0,1)
        // exist. Fill is their mean, and nothing panics.
        let res = CheckerboardResolution::new(2, 2);
        assert!(!is_current_sample(0, 0, 1));
        let mut buf = alloc::vec![gray(0.0); res.full_pixel_count()];
        buf[index(res, 1, 0)] = gray(0.2);
        buf[index(res, 0, 1)] = gray(0.8);
        let filled = spatial_fill(&buf, res, 0, 0);
        assert!(approx_rgba(filled, gray(0.5)));
    }

    #[test]
    fn neighbor_box_clamp_suppresses_a_ghost_history() {
        // Center missing, all neighbors 0.5, history is a wild outlier.
        let res = CheckerboardResolution::new(3, 3);
        let mut current = alloc::vec![gray(0.0); res.full_pixel_count()];
        for &(x, y) in &[(0u32, 1u32), (2, 1), (1, 0), (1, 2)] {
            current[index(res, x, y)] = gray(0.5);
        }
        let mut history = alloc::vec![gray(0.0); res.full_pixel_count()];
        history[index(res, 1, 1)] = gray(5.0); // stale ghost
        let cfg = CheckerboardConfig::new(1.0);
        let out = resolve(res, &current, &history, 1, cfg);
        // The ghost is clamped to the neighbor box [0.5, 0.5].
        assert!(approx_rgba(out[index(res, 1, 1)], gray(0.5)));
    }

    #[test]
    fn history_inside_the_box_passes_through_unchanged() {
        let res = CheckerboardResolution::new(3, 3);
        let mut current = alloc::vec![gray(0.0); res.full_pixel_count()];
        current[index(res, 0, 1)] = gray(0.2);
        current[index(res, 2, 1)] = gray(0.8);
        current[index(res, 1, 0)] = gray(0.4);
        current[index(res, 1, 2)] = gray(0.6);
        let mut history = alloc::vec![gray(0.0); res.full_pixel_count()];
        history[index(res, 1, 1)] = gray(0.5); // inside [0.2, 0.8]
        let out = resolve(res, &current, &history, 1, CheckerboardConfig::new(1.0));
        assert!(approx_rgba(out[index(res, 1, 1)], gray(0.5)));
    }

    #[test]
    fn history_weight_zero_matches_pure_spatial_resolve() {
        let res = CheckerboardResolution::new(4, 4);
        let mut current = alloc::vec![gray(0.0); res.full_pixel_count()];
        for y in 0..4u32 {
            for x in 0..4u32 {
                current[index(res, x, y)] = gray(0.1 * (x + y) as f32);
            }
        }
        let history = alloc::vec![gray(9.0); res.full_pixel_count()];
        let blended = resolve(res, &current, &history, 0, CheckerboardConfig::new(0.0));
        let spatial = resolve_spatial(res, &current, 0);
        for i in 0..res.full_pixel_count() {
            assert!(approx_rgba(blended[i], spatial[i]), "pixel {i}");
        }
    }

    #[test]
    fn history_weight_one_yields_clamped_history_on_missing_cells() {
        let res = CheckerboardResolution::new(3, 3);
        let mut current = alloc::vec![gray(0.0); res.full_pixel_count()];
        for &(x, y) in &[(0u32, 1u32), (2, 1), (1, 0), (1, 2)] {
            current[index(res, x, y)] = gray(0.3);
        }
        let mut history = alloc::vec![gray(0.0); res.full_pixel_count()];
        history[index(res, 1, 1)] = gray(0.3); // inside box -> untouched
        let out = resolve(res, &current, &history, 1, CheckerboardConfig::new(1.0));
        assert!(approx_rgba(out[index(res, 1, 1)], gray(0.3)));
    }

    #[test]
    fn shaded_cells_are_woven_through_verbatim() {
        let res = CheckerboardResolution::new(4, 3);
        let mut current = alloc::vec![gray(0.0); res.full_pixel_count()];
        for y in 0..3u32 {
            for x in 0..4u32 {
                current[index(res, x, y)] =
                    Rgba::new(0.05 * x as f32, 0.07 * y as f32, 0.11 * (x + y) as f32, 1.0);
            }
        }
        let history = alloc::vec![gray(0.0); res.full_pixel_count()];
        let out = resolve(res, &current, &history, 2, CheckerboardConfig::default());
        for y in 0..3u32 {
            for x in 0..4u32 {
                if is_current_sample(x, y, 2) {
                    let i = index(res, x, y);
                    assert!(approx_rgba(out[i], current[i]), "shaded ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn resolved_and_shaded_bytes_follow_std430() {
        let res = CheckerboardResolution::new(8, 4);
        // 32 pixels * 16 bytes/vec4 = 512.
        assert_eq!(res.resolved_bytes(), 512);
        // shaded_width = ceil(8/2) = 4 => 16 samples * 16 = 256.
        assert_eq!(res.shaded_bytes(), 256);
    }

    #[test]
    fn coverage_mask_bytes_follow_std430() {
        let res = CheckerboardResolution::new(8, 4);
        // 32 pixels * 4 bytes/u32 = 128.
        assert_eq!(res.coverage_mask_bytes(), 128);
    }

    #[test]
    fn shaded_width_rounds_up_odd_widths() {
        assert_eq!(CheckerboardResolution::new(1, 4).shaded_width(), 1);
        assert_eq!(CheckerboardResolution::new(5, 4).shaded_width(), 3);
        assert_eq!(CheckerboardResolution::new(8, 4).shaded_width(), 4);
    }

    #[test]
    fn degenerate_zero_resolution_is_empty_but_nonzero_std430() {
        let res = CheckerboardResolution::new(0, 0);
        assert_eq!(res.full_pixel_count(), 0);
        assert_eq!(res.samples_this_frame(0), 0);
        assert_eq!(res.missing_this_frame(0), 0);
        // std430 clamps an empty buffer up to one element.
        assert_eq!(res.resolved_bytes(), VEC4_STRIDE);
        assert_eq!(res.coverage_mask_bytes(), U32_STRIDE);
        let out = resolve_spatial(res, &[], 0);
        assert!(out.is_empty());
    }

    #[test]
    fn degenerate_one_by_one_resolve_does_not_panic() {
        let res = CheckerboardResolution::new(1, 1);
        let current = alloc::vec![gray(0.25); 1];
        let history = alloc::vec![gray(0.75); 1];
        // frame 0: the single cell is shaded and copied through.
        let a = resolve(res, &current, &history, 0, CheckerboardConfig::default());
        assert!(approx_rgba(a[0], gray(0.25)));
        // frame 1: the single cell is missing and isolated; the fill falls back
        // to its own stored value, and the (fallback, fallback) box leaves the
        // clamped history equal to that fallback.
        let b = resolve(res, &current, &history, 1, CheckerboardConfig::new(1.0));
        assert!(approx_rgba(b[0], gray(0.25)));
    }

    #[test]
    fn smoothstep_hits_its_endpoints_and_clamps() {
        assert!(approx(smoothstep(0.0), 0.0));
        assert!(approx(smoothstep(1.0), 1.0));
        assert!(approx(smoothstep(0.5), 0.5));
        // Out-of-range inputs clamp rather than overshoot.
        assert!(approx(smoothstep(-1.0), 0.0));
        assert!(approx(smoothstep(2.0), 1.0));
    }

    #[test]
    fn clamp_box_is_component_wise() {
        let lo = Rgba::new(0.1, 0.2, 0.3, 0.4);
        let hi = Rgba::new(0.6, 0.7, 0.8, 0.9);
        let clamped = Rgba::new(-1.0, 0.5, 9.0, 0.4).clamp_box(lo, hi);
        assert!(approx_rgba(clamped, Rgba::new(0.1, 0.5, 0.8, 0.4)));
    }

    #[test]
    fn resolve_spatial_rebuilds_missing_cells_without_history() {
        let res = CheckerboardResolution::new(3, 3);
        let mut current = alloc::vec![gray(0.0); res.full_pixel_count()];
        for &(x, y) in &[(0u32, 1u32), (2, 1), (1, 0), (1, 2)] {
            current[index(res, x, y)] = gray(0.6);
        }
        let out = resolve_spatial(res, &current, 1);
        assert!(approx_rgba(out[index(res, 1, 1)], gray(0.6)));
    }
}
