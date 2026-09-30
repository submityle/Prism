//! `FXAA` 3.11-style luma-adaptive antialiasing — the `CPU` gold-standard
//! contract for the particle stylization stack (design §16-§21).
//!
//! `FXAA` (Fast Approximate Anti-Aliasing) is the classic single-pass,
//! shader-only edge smoother: instead of super-sampling geometry it inspects a
//! `3x3` window of perceived brightness (`luma`) around each output texel,
//! decides whether a contrasty edge runs through it, works out which way that
//! edge is oriented, and returns a blend weight that a later resolve pass uses
//! to pull the texel toward its neighbors along the edge. It is entirely a
//! post-process on an already-shaded image, so it composites naturally on top
//! of the particle color buffer this subsystem produces.
//!
//! # The algorithm
//!
//! 1. **Luma** — collapse an `RGB` sample to a single brightness scalar with a
//!    green-weighted `Rec.601` combination (green dominates human luminance
//!    perception, which is why `FXAA` keys off it); see [`luma`].
//! 2. **Edge test** — form the local contrast `max - min` over the center and
//!    its four cross neighbors and compare it against an absolute floor and a
//!    brightness-relative threshold; see [`detect_edge`].
//! 3. **Direction** — compare a horizontal `Sobel`-style gradient magnitude
//!    against the vertical one to classify the edge line as
//!    [`EdgeDir::Vertical`] or [`EdgeDir::Horizontal`]; see [`edge_direction`].
//! 4. **Subpixel** — measure how far the `3x3` low-pass mean departs from the
//!    center, smooth it, and scale by a subpixel-quality knob; see
//!    [`subpixel_blend_factor`].
//! 5. **Blend** — combine the gated edge strength with the subpixel term into
//!    the final along-edge sampling offset weight in `0..=1`; see
//!    [`blend_amount`].
//!
//! [`resolve_luma_grid`] batches step 2-5 over a clamped-border `luma` image,
//! and [`to_std430`] packs [`FxaaParams`] into the 16-byte `std430` block a
//! future `GPU` resolve kernel binds.
//!
//! # Strict scope
//!
//! This file is *only* the `FXAA` luma-edge antialiasing contract. It is **not**
//! the `Sobel`/`Roberts` edge-magnitude contract (that is
//! [`super::edge_detect`]), it is **not** the `Toksvig`/`Frostbite` normal-map
//! specular antialiasing contract (that is [`super::specular_aa`]), and it is
//! **not** the contrast-adaptive sharpen kernel (that is
//! [`super::sharpen_cas`]). It defines its own [`Rgb`] and [`Neighborhood`]
//! types and reuses none of theirs; input and output share one resolution.
//!
//! # Determinism
//!
//! The determinism-locked contract layer forbids transcendental functions. This
//! module uses only `+ - * /`, `f32::min`/`max`/`abs`/`clamp`, a hand-rolled
//! `smoothstep`, and integer `div_ceil`, every division guarded against a
//! near-zero denominator, so a future `GPU` kernel reproduces the `CPU` result
//! bit for bit. Only [`super::gpu_layout`] is imported.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Number of scalar fields packed into the [`FxaaParams`] `std430` block:
/// `edge_threshold`, `edge_threshold_min`, and `subpix_quality`.
const FXAA_FIELD_COUNT: usize = 3;

/// Byte size of the `std430` packing of [`FxaaParams`]: the three scalars
/// rounded up to a whole `vec4` slot so the block honors the 16-byte `std430`
/// base alignment.
pub const FXAA_STD430_SIZE: usize = FXAA_FIELD_COUNT.div_ceil(4) * VEC4_STRIDE;

/// `Rec.601` red weight of the green-dominated `luma` combination.
const LUMA_R: f32 = 0.299;

/// `Rec.601` green weight of the green-dominated `luma` combination.
const LUMA_G: f32 = 0.587;

/// `Rec.601` blue weight of the green-dominated `luma` combination.
const LUMA_B: f32 = 0.114;

/// Denominators with magnitude at or below this are treated as (near) zero so
/// evaluation falls back to a defined result instead of dividing by zero or
/// propagating `NaN`. It is the single production `f32` tolerance; direct `==`
/// / `!=` comparison of floating point is intentionally avoided.
const CMP_EPS: f32 = 1e-6;

/// A linear `RGB` color sample, the input to [`luma`].
///
/// This is deliberately distinct from the color types in the sibling
/// antialiasing modules: `FXAA` only ever needs the three channels it collapses
/// into brightness.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rgb {
    /// Red channel, linear.
    pub r: f32,
    /// Green channel, linear.
    pub g: f32,
    /// Blue channel, linear.
    pub b: f32,
}

impl Rgb {
    /// Builds an [`Rgb`] sample from its three linear channels.
    #[must_use]
    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }
}

/// The perceived brightness (`luma`) of a `3x3` window: the center texel `m`
/// plus its eight neighbors, named by compass direction.
///
/// All fields are already-computed `luma` scalars (see [`luma`]); the `FXAA`
/// maths never re-reads color once the neighborhood is built.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Neighborhood {
    /// Center (middle) texel `luma`.
    pub m: f32,
    /// North (up) neighbor `luma`.
    pub n: f32,
    /// South (down) neighbor `luma`.
    pub s: f32,
    /// East (right) neighbor `luma`.
    pub e: f32,
    /// West (left) neighbor `luma`.
    pub w: f32,
    /// North-east (up-right) diagonal neighbor `luma`.
    pub ne: f32,
    /// North-west (up-left) diagonal neighbor `luma`.
    pub nw: f32,
    /// South-east (down-right) diagonal neighbor `luma`.
    pub se: f32,
    /// South-west (down-left) diagonal neighbor `luma`.
    pub sw: f32,
}

/// The orientation of the edge line running through a [`Neighborhood`].
///
/// The classification compares the horizontal gradient magnitude (contrast
/// across columns) against the vertical one (contrast across rows). A dominant
/// horizontal gradient means the brightness steps left-to-right, so the edge
/// *line* itself is [`EdgeDir::Vertical`]; the symmetric case is
/// [`EdgeDir::Horizontal`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeDir {
    /// The edge line runs horizontally (top-to-bottom brightness step).
    Horizontal,
    /// The edge line runs vertically (left-to-right brightness step).
    Vertical,
}

/// The three tuning scalars of the `FXAA` edge/subpixel decision.
///
/// * `edge_threshold` — brightness-relative contrast fraction; the local
///   contrast must reach `luma_max * edge_threshold` to count as an edge.
/// * `edge_threshold_min` — absolute contrast floor that suppresses edges in
///   near-black regions where the relative threshold would be tiny.
/// * `subpix_quality` — `0..=1` scale on the subpixel-aliasing blend term.
///
/// [`FxaaParams::default_quality`] returns the canonical `FXAA` 3.11 "quality"
/// values `0.166` / `0.0833` / `0.75`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FxaaParams {
    /// Brightness-relative contrast fraction required to flag an edge.
    pub edge_threshold: f32,
    /// Absolute contrast floor for near-black regions.
    pub edge_threshold_min: f32,
    /// `0..=1` scale applied to the subpixel-aliasing blend term.
    pub subpix_quality: f32,
}

impl FxaaParams {
    /// The canonical `FXAA` 3.11 "quality" preset.
    ///
    /// `edge_threshold = 0.166`, `edge_threshold_min = 0.0833`,
    /// `subpix_quality = 0.75`.
    #[must_use]
    pub const fn default_quality() -> Self {
        Self {
            edge_threshold: 0.166,
            edge_threshold_min: 0.0833,
            subpix_quality: 0.75,
        }
    }
}

/// Collapses a linear [`Rgb`] sample to its green-dominated `Rec.601` `luma`.
///
/// The weights sum to `1.0`, so a gray sample (`r == g == b`) returns its own
/// value unchanged.
#[must_use]
pub fn luma(color: &Rgb) -> f32 {
    color.r * LUMA_R + color.g * LUMA_G + color.b * LUMA_B
}

/// The `luma` `(min, max)` over the center and the four cross neighbors, the
/// window the `FXAA` early edge test keys off.
#[must_use]
fn cross_extent(nb: &Neighborhood) -> (f32, f32) {
    let lo = nb.m.min(nb.n).min(nb.s).min(nb.e).min(nb.w);
    let hi = nb.m.max(nb.n).max(nb.s).max(nb.e).max(nb.w);
    (lo, hi)
}

/// The `luma` `(min, max)` over the full `3x3` window, used by the subpixel
/// term.
#[must_use]
fn full_extent(nb: &Neighborhood) -> (f32, f32) {
    let lo =
        nb.m.min(nb.n)
            .min(nb.s)
            .min(nb.e)
            .min(nb.w)
            .min(nb.ne)
            .min(nb.nw)
            .min(nb.se)
            .min(nb.sw);
    let hi =
        nb.m.max(nb.n)
            .max(nb.s)
            .max(nb.e)
            .max(nb.w)
            .max(nb.ne)
            .max(nb.nw)
            .max(nb.se)
            .max(nb.sw);
    (lo, hi)
}

/// Hermite `smoothstep` on an already-unit input: `t*t*(3 - 2t)`.
///
/// The argument is clamped to `0..=1` first so out-of-range callers stay on the
/// defined `0..=1` curve. No transcendental is used.
#[must_use]
fn smoothstep_unit(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Returns `true` when the neighborhood's local contrast is high enough to be
/// treated as an `FXAA` edge.
///
/// The local contrast is `max - min` over the cross window; it must reach
/// `max(edge_threshold_min, luma_max * edge_threshold)`. A flat neighborhood
/// (zero contrast) never passes, because the floor is strictly positive for the
/// canonical preset.
#[must_use]
pub fn detect_edge(nb: &Neighborhood, params: &FxaaParams) -> bool {
    let (lo, hi) = cross_extent(nb);
    let range = hi - lo;
    let threshold = params.edge_threshold_min.max(hi * params.edge_threshold);
    range >= threshold
}

/// Classifies the orientation of the edge line through the neighborhood.
///
/// A `Sobel`-style horizontal gradient magnitude (contrast across columns) is
/// compared against the vertical one (contrast across rows). Ties, including a
/// flat neighborhood, resolve to [`EdgeDir::Vertical`].
#[must_use]
pub fn edge_direction(nb: &Neighborhood) -> EdgeDir {
    let grad_x = ((nb.ne + 2.0 * nb.e + nb.se) - (nb.nw + 2.0 * nb.w + nb.sw)).abs();
    let grad_y = ((nb.nw + 2.0 * nb.n + nb.ne) - (nb.sw + 2.0 * nb.s + nb.se)).abs();
    if grad_x >= grad_y {
        EdgeDir::Vertical
    } else {
        EdgeDir::Horizontal
    }
}

/// The subpixel-aliasing blend term in `0..=1`.
///
/// It measures how far the `3x3` low-pass mean departs from the center relative
/// to the full-window contrast, smooths that ratio, and scales it by
/// `subpix_quality`. A flat neighborhood yields `0`.
#[must_use]
pub fn subpixel_blend_factor(nb: &Neighborhood, params: &FxaaParams) -> f32 {
    let sum = nb.m + nb.n + nb.s + nb.e + nb.w + nb.ne + nb.nw + nb.se + nb.sw;
    let avg = sum / 9.0;
    let (lo, hi) = full_extent(nb);
    let range = hi - lo;
    let contrast = (avg - nb.m).abs();
    let ratio = if range > CMP_EPS {
        (contrast / range).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (smoothstep_unit(ratio) * params.subpix_quality).clamp(0.0, 1.0)
}

/// The final along-edge sampling offset weight in `0..=1`.
///
/// Returns `0` when [`detect_edge`] rejects the neighborhood. Otherwise it takes
/// the larger of the brightness-normalized edge strength (`contrast / luma_max`)
/// and the [`subpixel_blend_factor`], clamped to `0..=1`.
#[must_use]
pub fn blend_amount(nb: &Neighborhood, params: &FxaaParams) -> f32 {
    if !detect_edge(nb, params) {
        return 0.0;
    }
    let (lo, hi) = cross_extent(nb);
    let range = hi - lo;
    let edge_blend = if hi > CMP_EPS {
        (range / hi).clamp(0.0, 1.0)
    } else {
        0.0
    };
    edge_blend
        .max(subpixel_blend_factor(nb, params))
        .clamp(0.0, 1.0)
}

/// Builds the clamped-border [`Neighborhood`] centered at `(x, y)` from a
/// row-major `luma` image.
///
/// Out-of-bounds coordinates replicate the edge texel. The caller guarantees
/// `width >= 1`, `height >= 1`, `x < width`, `y < height`, and
/// `luma.len() >= width * height`.
#[must_use]
fn sample_neighborhood(
    luma: &[f32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
) -> Neighborhood {
    let xm = x.saturating_sub(1);
    let xp = (x + 1).min(width - 1);
    let ym = y.saturating_sub(1);
    let yp = (y + 1).min(height - 1);
    Neighborhood {
        m: luma[y * width + x],
        n: luma[ym * width + x],
        s: luma[yp * width + x],
        e: luma[y * width + xp],
        w: luma[y * width + xm],
        ne: luma[ym * width + xp],
        nw: luma[ym * width + xm],
        se: luma[yp * width + xp],
        sw: luma[yp * width + xm],
    }
}

/// Evaluates [`blend_amount`] for every texel of a row-major `luma` image with
/// clamped-border sampling.
///
/// Returns one blend weight per input texel in row-major order. Returns an empty
/// [`Vec`] when either dimension is zero or the buffer is shorter than
/// `width * height`.
#[must_use]
pub fn resolve_luma_grid(
    luma: &[f32],
    width: usize,
    height: usize,
    params: &FxaaParams,
) -> Vec<f32> {
    let count = width.saturating_mul(height);
    if width == 0 || height == 0 || luma.len() < count {
        return Vec::new();
    }
    (0..height)
        .flat_map(|y| {
            (0..width).map(move |x| {
                let nb = sample_neighborhood(luma, width, height, x, y);
                blend_amount(&nb, params)
            })
        })
        .collect()
}

/// Packs [`FxaaParams`] into its `std430` byte block.
///
/// The three `f32` scalars are written little-endian in field order into the
/// leading 12 bytes; the trailing 4 bytes are the `vec4`-alignment pad and stay
/// zero. A future `GPU` resolve kernel binds this block directly.
#[must_use]
pub fn to_std430(params: &FxaaParams) -> [u8; FXAA_STD430_SIZE] {
    let fields = [
        params.edge_threshold,
        params.edge_threshold_min,
        params.subpix_quality,
    ];
    let mut bytes = [0u8; FXAA_STD430_SIZE];
    for (slot, value) in bytes.chunks_exact_mut(4).zip(fields.iter()) {
        slot.copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Total `std430` storage bytes reserved for `count` [`FxaaParams`] blocks.
///
/// Reuses [`storage_bytes`], so an empty request still reserves one block (a
/// `WebGPU` storage binding may not be zero-sized).
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(FXAA_STD430_SIZE, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only absolute tolerance; production comparisons use [`CMP_EPS`].
    const TEST_EPS: f32 = 1e-5;

    /// `true` when two scalars agree within [`TEST_EPS`], avoiding `f32` `==`.
    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= TEST_EPS
    }

    /// Builds a [`Neighborhood`] from a row-major `[nw, n, ne, w, m, e, sw, s, se]`.
    fn nb(cells: [f32; 9]) -> Neighborhood {
        Neighborhood {
            nw: cells[0],
            n: cells[1],
            ne: cells[2],
            w: cells[3],
            m: cells[4],
            e: cells[5],
            sw: cells[6],
            s: cells[7],
            se: cells[8],
        }
    }

    #[test]
    fn luma_uses_green_dominant_weights() {
        assert!(approx(luma(&Rgb::new(1.0, 0.0, 0.0)), LUMA_R));
        assert!(approx(luma(&Rgb::new(0.0, 1.0, 0.0)), LUMA_G));
        assert!(approx(luma(&Rgb::new(0.0, 0.0, 1.0)), LUMA_B));
    }

    #[test]
    fn luma_of_gray_is_identity() {
        assert!(approx(luma(&Rgb::new(0.5, 0.5, 0.5)), 0.5));
        assert!(approx(luma(&Rgb::new(0.25, 0.25, 0.25)), 0.25));
    }

    #[test]
    fn luma_weights_sum_to_one() {
        assert!(approx(LUMA_R + LUMA_G + LUMA_B, 1.0));
    }

    #[test]
    fn flat_region_has_no_edge() {
        let params = FxaaParams::default_quality();
        let flat = nb([0.5; 9]);
        assert!(!detect_edge(&flat, &params));
    }

    #[test]
    fn flat_region_has_zero_blend() {
        let params = FxaaParams::default_quality();
        let flat = nb([0.5; 9]);
        assert!(approx(blend_amount(&flat, &params), 0.0));
    }

    #[test]
    fn strong_edge_is_detected() {
        let params = FxaaParams::default_quality();
        let edge = nb([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
        assert!(detect_edge(&edge, &params));
    }

    #[test]
    fn strong_edge_has_positive_blend() {
        let params = FxaaParams::default_quality();
        let edge = nb([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
        assert!(blend_amount(&edge, &params) > 0.0);
    }

    #[test]
    fn faint_edge_below_floor_is_rejected() {
        // Cross contrast 0.05 sits below the 0.0833 absolute floor.
        let params = FxaaParams::default_quality();
        let faint = nb([0.5, 0.5, 0.5, 0.5, 0.5, 0.55, 0.5, 0.5, 0.5]);
        assert!(!detect_edge(&faint, &params));
    }

    #[test]
    fn vertical_edge_direction_is_vertical() {
        let edge = nb([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
        assert_eq!(edge_direction(&edge), EdgeDir::Vertical);
    }

    #[test]
    fn horizontal_edge_direction_is_horizontal() {
        let edge = nb([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        assert_eq!(edge_direction(&edge), EdgeDir::Horizontal);
    }

    #[test]
    fn flat_direction_ties_to_vertical() {
        let flat = nb([0.3; 9]);
        assert_eq!(edge_direction(&flat), EdgeDir::Vertical);
    }

    #[test]
    fn subpixel_factor_is_within_unit_range() {
        let params = FxaaParams::default_quality();
        let checker = nb([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]);
        let factor = subpixel_blend_factor(&checker, &params);
        assert!((0.0..=1.0).contains(&factor));
    }

    #[test]
    fn subpixel_factor_zero_on_flat() {
        let params = FxaaParams::default_quality();
        let flat = nb([0.4; 9]);
        assert!(approx(subpixel_blend_factor(&flat, &params), 0.0));
    }

    #[test]
    fn subpixel_factor_grows_with_quality() {
        let cells = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let low = FxaaParams {
            edge_threshold: 0.166,
            edge_threshold_min: 0.0833,
            subpix_quality: 0.25,
        };
        let high = FxaaParams {
            edge_threshold: 0.166,
            edge_threshold_min: 0.0833,
            subpix_quality: 0.75,
        };
        let lo = subpixel_blend_factor(&nb(cells), &low);
        let hi = subpixel_blend_factor(&nb(cells), &high);
        assert!(hi >= lo);
    }

    #[test]
    fn blend_amount_is_within_unit_range() {
        let params = FxaaParams::default_quality();
        let edge = nb([0.0, 0.1, 0.9, 0.0, 0.2, 1.0, 0.1, 0.0, 0.8]);
        let amount = blend_amount(&edge, &params);
        assert!((0.0..=1.0).contains(&amount));
    }

    #[test]
    fn blend_amount_zero_when_no_edge() {
        let params = FxaaParams::default_quality();
        let flat = nb([0.7; 9]);
        assert!(approx(blend_amount(&flat, &params), 0.0));
    }

    #[test]
    fn detect_edge_monotone_in_threshold() {
        // If a higher relative threshold still flags the edge, the lower one
        // must flag it too.
        let edge = nb([0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5]);
        let low = FxaaParams {
            edge_threshold: 0.1,
            edge_threshold_min: 0.01,
            subpix_quality: 0.75,
        };
        let high = FxaaParams {
            edge_threshold: 0.9,
            edge_threshold_min: 0.01,
            subpix_quality: 0.75,
        };
        if detect_edge(&edge, &high) {
            assert!(detect_edge(&edge, &low));
        }
    }

    #[test]
    fn blend_amount_monotone_non_increasing_in_threshold() {
        let edge = nb([0.0, 0.0, 0.3, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3]);
        let low = FxaaParams {
            edge_threshold: 0.1,
            edge_threshold_min: 0.01,
            subpix_quality: 0.75,
        };
        let high = FxaaParams {
            edge_threshold: 0.95,
            edge_threshold_min: 0.5,
            subpix_quality: 0.75,
        };
        let lo = blend_amount(&edge, &low);
        let hi = blend_amount(&edge, &high);
        assert!(lo >= hi);
    }

    #[test]
    fn default_quality_matches_canonical_preset() {
        let params = FxaaParams::default_quality();
        assert!(approx(params.edge_threshold, 0.166));
        assert!(approx(params.edge_threshold_min, 0.0833));
        assert!(approx(params.subpix_quality, 0.75));
    }

    #[test]
    fn resolve_grid_length_matches_dimensions() {
        let params = FxaaParams::default_quality();
        let img = [0.5_f32; 12];
        let out = resolve_luma_grid(&img, 4, 3, &params);
        assert_eq!(out.len(), 12);
    }

    #[test]
    fn resolve_grid_zero_width_is_empty() {
        let params = FxaaParams::default_quality();
        let img = [0.5_f32; 4];
        assert!(resolve_luma_grid(&img, 0, 4, &params).is_empty());
    }

    #[test]
    fn resolve_grid_zero_height_is_empty() {
        let params = FxaaParams::default_quality();
        let img = [0.5_f32; 4];
        assert!(resolve_luma_grid(&img, 4, 0, &params).is_empty());
    }

    #[test]
    fn resolve_grid_short_buffer_is_empty() {
        let params = FxaaParams::default_quality();
        let img = [0.5_f32; 3];
        assert!(resolve_luma_grid(&img, 4, 4, &params).is_empty());
    }

    #[test]
    fn resolve_grid_flat_is_all_zero() {
        let params = FxaaParams::default_quality();
        let img = [0.5_f32; 9];
        let out = resolve_luma_grid(&img, 3, 3, &params);
        assert!(out.iter().all(|v| approx(*v, 0.0)));
    }

    #[test]
    fn resolve_grid_boundary_safe_and_bounded() {
        // A vertical edge across a 4x2 image: borders are sampled with clamp, so
        // no index goes out of range and every weight stays in 0..=1.
        let params = FxaaParams::default_quality();
        let img = [0.0_f32, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let out = resolve_luma_grid(&img, 4, 2, &params);
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|v| (0.0..=1.0).contains(v)));
        assert!(out.iter().any(|v| *v > 0.0));
    }

    #[test]
    fn single_pixel_grid_is_zero() {
        let params = FxaaParams::default_quality();
        let img = [0.5_f32];
        let out = resolve_luma_grid(&img, 1, 1, &params);
        assert_eq!(out.len(), 1);
        assert!(approx(out[0], 0.0));
    }

    #[test]
    fn to_std430_has_vec4_block_size() {
        let params = FxaaParams::default_quality();
        let bytes = to_std430(&params);
        assert_eq!(bytes.len(), FXAA_STD430_SIZE);
        assert_eq!(FXAA_STD430_SIZE, VEC4_STRIDE);
        assert_eq!(FXAA_STD430_SIZE % VEC4_STRIDE, 0);
    }

    #[test]
    fn to_std430_encodes_fields_in_order() {
        let params = FxaaParams {
            edge_threshold: 0.2,
            edge_threshold_min: 0.05,
            subpix_quality: 0.6,
        };
        let bytes = to_std430(&params);
        let mut a = [0u8; 4];
        let mut b = [0u8; 4];
        let mut c = [0u8; 4];
        a.copy_from_slice(&bytes[0..4]);
        b.copy_from_slice(&bytes[4..8]);
        c.copy_from_slice(&bytes[8..12]);
        assert!(approx(f32::from_le_bytes(a), 0.2));
        assert!(approx(f32::from_le_bytes(b), 0.05));
        assert!(approx(f32::from_le_bytes(c), 0.6));
    }

    #[test]
    fn to_std430_pads_trailing_slot_with_zero() {
        let params = FxaaParams::default_quality();
        let bytes = to_std430(&params);
        assert!(bytes[12..16].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn gpu_storage_bytes_reserves_blocks() {
        assert_eq!(gpu_storage_bytes(0), FXAA_STD430_SIZE);
        assert_eq!(gpu_storage_bytes(1), FXAA_STD430_SIZE);
        assert_eq!(gpu_storage_bytes(4), 4 * FXAA_STD430_SIZE);
    }
}
