//! Bicubic Catmull-Rom history resampling — CPU golden.
//!
//! When a TAA / temporal-GI pass reprojects history by a sub-texel motion
//! vector it must *resample* the history buffer at a fractional coordinate.
//! Plain bilinear filtering is cheap but blurs the image a little every frame,
//! and over a long temporal feedback loop that blur compounds into a soft,
//! smeared result.  A **Catmull-Rom** bicubic reconstruction has sharpening
//! negative lobes that counteract this, keeping reprojected history crisp — at
//! the cost of possible overshoot ("ringing") near high-contrast edges.
//!
//! This module is the backend-neutral reference for that filter:
//!
//! * [`catmull_rom_weights`] — the four cubic weights for a 1-D fractional
//!   position (sum to one, interpolating).
//! * [`clamp_negative_lobes`] — optional ringing suppression that zeroes the
//!   sharpening lobes and renormalizes.
//! * [`CatmullRom5Tap`] + [`catmull_rom_5tap_axis`] — the Jimenez 5-tap
//!   optimization that folds the two inner cubic taps into one bilinear fetch
//!   per axis, so a separable 2-D sample costs 5 bilinear reads instead of 16.
//! * [`sample_catmull_rom_9tap`] / [`sample_catmull_rom_5tap`] — full 2-D
//!   resamplers driven by a caller-supplied bilinear reader.
//!
//! # Conventions
//! * The continuous texel coordinate for a UV `u` is `u * size - 0.5` (texel
//!   centers at half-integers), matching [`super::reproject`] and the GPU twin.
//! * Catmull-Rom here uses the standard tension `τ = 0.5` cardinal spline.  Its
//!   weights sum to one for any fractional position, so a flat input region is
//!   reproduced exactly (energy preserving).
//! * All helpers are deterministic and allocation-free (no RNG/IO/GPU/unsafe),
//!   floor sizes to one texel, sanitize non-finite inputs, and never emit NaN.

use bevy_math::{ops, IVec2, Vec2, Vec3};

/// Smallest texture extent (per axis) the samplers will address.
const MIN_TEXTURE_SIZE: f32 = 1.0;

/// Replaces a non-finite scalar with `fallback`.
#[inline]
fn finite_or(x: f32, fallback: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        fallback
    }
}

/// Clamps a texture size to be at least one texel per axis and finite.
#[inline]
fn sanitize_size(size: Vec2) -> Vec2 {
    Vec2::new(
        finite_or(size.x, MIN_TEXTURE_SIZE).max(MIN_TEXTURE_SIZE),
        finite_or(size.y, MIN_TEXTURE_SIZE).max(MIN_TEXTURE_SIZE),
    )
}

/// Clamp-to-edge integer texel addressing into `[0, size-1]`.
#[inline]
fn clamp_texel(coord: IVec2, size: Vec2) -> IVec2 {
    let max_x = (size.x as i32 - 1).max(0);
    let max_y = (size.y as i32 - 1).max(0);
    IVec2::new(coord.x.clamp(0, max_x), coord.y.clamp(0, max_y))
}

/// Returns the four Catmull-Rom cubic weights `[w0, w1, w2, w3]` for a
/// fractional offset `f ∈ [0, 1]` between the two inner control points.
///
/// `w0` and `w3` weight the outer samples (the sharpening lobes, usually
/// negative); `w1` and `w2` weight the two inner samples straddling the sample
/// position.  The weights sum to exactly one for any `f`, and at `f = 0` /
/// `f = 1` they collapse to the pure inner samples, so the spline interpolates
/// its control points.  `f` is clamped to `[0, 1]` to stay well-defined.
#[inline]
pub fn catmull_rom_weights(f: f32) -> [f32; 4] {
    let f = finite_or(f, 0.0).clamp(0.0, 1.0);
    let f2 = f * f;
    let f3 = f2 * f;
    // Cardinal spline (tension 0.5) basis, pre-expanded.
    let w0 = -0.5 * f3 + f2 - 0.5 * f;
    let w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
    let w2 = -1.5 * f3 + 2.0 * f2 + 0.5 * f;
    let w3 = 0.5 * f3 - 0.5 * f2;
    [w0, w1, w2, w3]
}

/// Suppresses ringing by clamping the sharpening lobes to be non-positive-free:
/// any negative weight is set to zero and the four weights are renormalized so
/// they again sum to one.
///
/// This trades Catmull-Rom's crispness for a guaranteed overshoot-free result
/// (the output becomes a convex combination of the taps, so it can never leave
/// their min/max range).  If every weight is non-positive after clamping (a
/// degenerate input), it falls back to picking the nominal inner tap `w1`.
#[inline]
pub fn clamp_negative_lobes(weights: [f32; 4]) -> [f32; 4] {
    let clamped = [
        weights[0].max(0.0),
        weights[1].max(0.0),
        weights[2].max(0.0),
        weights[3].max(0.0),
    ];
    let sum = clamped[0] + clamped[1] + clamped[2] + clamped[3];
    if sum > 1.0e-8 {
        [
            clamped[0] / sum,
            clamped[1] / sum,
            clamped[2] / sum,
            clamped[3] / sum,
        ]
    } else {
        [0.0, 1.0, 0.0, 0.0]
    }
}

/// Renormalizes four weights to sum to one, falling back to the inner tap when
/// the total magnitude is degenerate.  Useful after dropping taps (e.g. the
/// 5-tap cross pattern that discards footprint corners).
#[inline]
pub fn normalize_weights(weights: [f32; 4]) -> [f32; 4] {
    let sum = weights[0] + weights[1] + weights[2] + weights[3];
    if sum.abs() > 1.0e-8 {
        [
            weights[0] / sum,
            weights[1] / sum,
            weights[2] / sum,
            weights[3] / sum,
        ]
    } else {
        [0.0, 1.0, 0.0, 0.0]
    }
}

/// The three combined taps for one axis of the Jimenez 5-tap Catmull-Rom
/// optimization.
///
/// Instead of four discrete samples, the two inner cubic weights `w1 + w2` are
/// serviced by a *single* bilinear fetch placed between texels `1` and `2`.
/// Each tap stores its texel-center coordinate (in continuous texel units) and
/// its total weight; the three weights sum to one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CatmullRom5Tap {
    /// Texel-center coordinates of the three taps (outer-low, combined, outer-high).
    pub coords: [f32; 3],
    /// Weights of the three taps; sum to one.
    pub weights: [f32; 3],
}

/// Computes the 5-tap combined taps for one axis of a bicubic Catmull-Rom
/// resample at continuous coordinate `sample_coord = uv * size`.
///
/// Returns the three bilinear tap centers and weights (see [`CatmullRom5Tap`]).
/// `texture_size` is floored to one texel to avoid division by zero; a
/// degenerate inner weight falls back to the geometric midpoint so the tap
/// coordinate stays finite.
#[inline]
pub fn catmull_rom_5tap_axis(sample_coord: f32, texture_size: f32) -> CatmullRom5Tap {
    let size = finite_or(texture_size, MIN_TEXTURE_SIZE).max(MIN_TEXTURE_SIZE);
    let sample_coord = finite_or(sample_coord, 0.5 * size);
    // Center of the texel just below the sample position.
    let tex_pos1 = ops::floor(sample_coord - 0.5) + 0.5;
    let f = (sample_coord - tex_pos1).clamp(0.0, 1.0);
    let w = catmull_rom_weights(f);
    let (w0, w1, w2, w3) = (w[0], w[1], w[2], w[3]);
    let w12 = w1 + w2;
    // Fractional placement of the combined inner tap between texels 1 and 2.
    let offset12 = if w12.abs() > 1.0e-8 { w2 / w12 } else { 0.5 };

    let coord0 = tex_pos1 - 1.0;
    let coord12 = tex_pos1 + offset12;
    let coord3 = tex_pos1 + 2.0;

    CatmullRom5Tap {
        coords: [coord0, coord12, coord3],
        weights: [w0, w12, w3],
    }
}

/// Separable 2-D Catmull-Rom resample using the exact 4x4 footprint (16 texels
/// via point reads), driven by a caller-supplied *point* reader.
///
/// This is the reference the optimized paths are checked against: it applies
/// [`catmull_rom_weights`] independently per axis and sums the outer product.
/// `fetch` returns the stored value at an integer texel; clamp-to-edge
/// addressing is applied here.  When `clamp_ringing` is set, the sharpening
/// lobes are removed via [`clamp_negative_lobes`] for an overshoot-free result.
#[inline]
pub fn sample_catmull_rom_9tap<F>(uv: Vec2, size: Vec2, clamp_ringing: bool, mut fetch: F) -> Vec3
where
    F: FnMut(IVec2) -> Vec3,
{
    let size = sanitize_size(size);
    let uv = Vec2::new(finite_or(uv.x, 0.5), finite_or(uv.y, 0.5));
    let coord = uv * size - Vec2::splat(0.5);
    let fx = ops::floor(coord.x);
    let fy = ops::floor(coord.y);
    let tx = (coord.x - fx).clamp(0.0, 1.0);
    let ty = (coord.y - fy).clamp(0.0, 1.0);

    let mut wx = catmull_rom_weights(tx);
    let mut wy = catmull_rom_weights(ty);
    if clamp_ringing {
        wx = clamp_negative_lobes(wx);
        wy = clamp_negative_lobes(wy);
    }

    let base = IVec2::new(fx as i32 - 1, fy as i32 - 1);
    let mut acc = Vec3::ZERO;
    for (j, wyj) in wy.iter().enumerate() {
        let mut row = Vec3::ZERO;
        for (i, wxi) in wx.iter().enumerate() {
            let texel = clamp_texel(base + IVec2::new(i as i32, j as i32), size);
            row += fetch(texel) * *wxi;
        }
        acc += row * *wyj;
    }
    acc
}

/// Separable 2-D Catmull-Rom resample via the 5-tap cross pattern, driven by a
/// caller-supplied *bilinear* reader.
///
/// Uses [`catmull_rom_5tap_axis`] per axis and reads the cross of taps
/// `(center-row x 3) + (center-column outer x 2)` — the four footprint corners
/// are dropped and the remaining weights renormalized, matching the common AAA
/// TAA optimization.  `fetch` samples the texture with hardware-style bilinear
/// filtering at a UV; this helper converts tap texel-centers back to UVs.
#[inline]
pub fn sample_catmull_rom_5tap<F>(uv: Vec2, size: Vec2, mut fetch: F) -> Vec3
where
    F: FnMut(Vec2) -> Vec3,
{
    let size = sanitize_size(size);
    let uv = Vec2::new(finite_or(uv.x, 0.5), finite_or(uv.y, 0.5));
    let sample_coord = uv * size;
    let ax = catmull_rom_5tap_axis(sample_coord.x, size.x);
    let ay = catmull_rom_5tap_axis(sample_coord.y, size.y);

    let inv = Vec2::new(1.0 / size.x, 1.0 / size.y);
    // Convert a texel-center coordinate pair to a UV.
    let to_uv = |cx: f32, cy: f32| Vec2::new(cx * inv.x, cy * inv.y);

    // Cross-pattern taps and their separable weights (corners dropped):
    //   top    center : (coord12, coord0_y)  weight w12x * w0y
    //   middle left   : (coord0,  coord12_y) weight w0x  * w12y
    //   middle center : (coord12, coord12_y) weight w12x * w12y
    //   middle right  : (coord3,  coord12_y) weight w3x  * w12y
    //   bottom center : (coord12, coord3_y)  weight w12x * w3y
    let samples = [
        (to_uv(ax.coords[1], ay.coords[0]), ax.weights[1] * ay.weights[0]),
        (to_uv(ax.coords[0], ay.coords[1]), ax.weights[0] * ay.weights[1]),
        (to_uv(ax.coords[1], ay.coords[1]), ax.weights[1] * ay.weights[1]),
        (to_uv(ax.coords[2], ay.coords[1]), ax.weights[2] * ay.weights[1]),
        (to_uv(ax.coords[1], ay.coords[2]), ax.weights[1] * ay.weights[2]),
    ];

    let total: f32 = samples.iter().map(|(_, w)| *w).sum();
    let inv_total = if total.abs() > 1.0e-8 { 1.0 / total } else { 0.0 };

    let mut acc = Vec3::ZERO;
    for (tap_uv, w) in samples {
        acc += fetch(tap_uv) * (w * inv_total);
    }
    if inv_total == 0.0 {
        // Degenerate weights: fall back to a single bilinear read at `uv`.
        fetch(uv)
    } else {
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-5;

    #[test]
    fn weights_sum_to_one() {
        for i in 0..=10 {
            let f = i as f32 / 10.0;
            let w = catmull_rom_weights(f);
            let s: f32 = w.iter().sum();
            assert!((s - 1.0).abs() < EPS, "f={f} sum={s}");
        }
    }

    #[test]
    fn weights_interpolate_endpoints() {
        let w0 = catmull_rom_weights(0.0);
        assert!((w0[0]).abs() < EPS);
        assert!((w0[1] - 1.0).abs() < EPS);
        assert!((w0[2]).abs() < EPS);
        assert!((w0[3]).abs() < EPS);

        let w1 = catmull_rom_weights(1.0);
        assert!((w1[0]).abs() < EPS);
        assert!((w1[1]).abs() < EPS);
        assert!((w1[2] - 1.0).abs() < EPS);
        assert!((w1[3]).abs() < EPS);
    }

    #[test]
    fn outer_lobes_are_sharpening() {
        // At the midpoint the outer lobes must be negative (sharpening).
        let w = catmull_rom_weights(0.5);
        assert!(w[0] < 0.0);
        assert!(w[3] < 0.0);
        assert!(w[1] > 0.0 && w[2] > 0.0);
    }

    #[test]
    fn clamp_lobes_removes_negatives_and_renormalizes() {
        let w = catmull_rom_weights(0.5);
        let c = clamp_negative_lobes(w);
        assert!(c.iter().all(|&x| x >= 0.0));
        let s: f32 = c.iter().sum();
        assert!((s - 1.0).abs() < EPS);
    }

    #[test]
    fn normalize_handles_degenerate() {
        let n = normalize_weights([0.0, 0.0, 0.0, 0.0]);
        assert_eq!(n, [0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn five_tap_weights_sum_to_one() {
        let a = catmull_rom_5tap_axis(10.37, 32.0);
        let s: f32 = a.weights.iter().sum();
        assert!((s - 1.0).abs() < EPS);
    }

    #[test]
    fn five_tap_combined_coord_between_inner_texels() {
        let a = catmull_rom_5tap_axis(10.37, 32.0);
        // coord12 must lie between texels 1 and 2 of the footprint.
        assert!(a.coords[1] > a.coords[0]);
        assert!(a.coords[1] < a.coords[2]);
    }

    #[test]
    fn nine_tap_reproduces_constant() {
        let size = Vec2::new(16.0, 16.0);
        let got = sample_catmull_rom_9tap(Vec2::new(0.4, 0.6), size, false, |_| Vec3::splat(3.0));
        assert!((got - Vec3::splat(3.0)).length() < EPS);
    }

    #[test]
    fn nine_tap_interpolates_control_point() {
        // Sampling exactly at a texel center returns that texel's value even
        // with sharpening lobes active (interpolation property).
        let size = Vec2::new(8.0, 8.0);
        let fetch = |t: IVec2| Vec3::splat((t.x * 10 + t.y) as f32);
        // Texel (3,5) center UV: ((3+0.5)/8, (5+0.5)/8).
        let uv = Vec2::new(3.5 / 8.0, 5.5 / 8.0);
        let got = sample_catmull_rom_9tap(uv, size, false, fetch);
        assert!((got.x - 35.0).abs() < 1.0e-3, "got {}", got.x);
    }

    #[test]
    fn five_tap_reproduces_constant() {
        let size = Vec2::new(16.0, 16.0);
        let got = sample_catmull_rom_5tap(Vec2::new(0.4, 0.6), size, |_| Vec3::splat(7.0));
        assert!((got - Vec3::splat(7.0)).length() < EPS);
    }

    #[test]
    fn non_finite_inputs_safe() {
        let size = Vec2::new(8.0, 8.0);
        let got = sample_catmull_rom_9tap(
            Vec2::new(f32::NAN, 0.5),
            size,
            true,
            |_| Vec3::splat(1.0),
        );
        assert!(got.x.is_finite() && (got - Vec3::splat(1.0)).length() < EPS);
        let w = catmull_rom_weights(f32::INFINITY);
        assert!(w.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn deterministic() {
        let size = Vec2::new(12.0, 9.0);
        let a = sample_catmull_rom_9tap(Vec2::new(0.3, 0.7), size, false, |t| {
            Vec3::splat((t.x + t.y) as f32)
        });
        let b = sample_catmull_rom_9tap(Vec2::new(0.3, 0.7), size, false, |t| {
            Vec3::splat((t.x + t.y) as f32)
        });
        assert_eq!(a, b);
    }
}
