//! R2 low-discrepancy ("blue-noise-like") sequence with temporal animation.
//!
//! The R2 sequence (Roberts 2018, *The Unreasonable Effectiveness of
//! Quasirandom Sequences*) is the 2-D additive recurrence driven by the plastic
//! constant.  Its point sets are highly uniform and visually blue-noise-like,
//! which — after temporal accumulation and the spatio-temporal denoiser — reads
//! far cleaner at 1–2 spp than white noise, while remaining a fully analytic,
//! deterministic pure function (no precomputed texture).
//!
//! Pixels are decorrelated with a per-pixel additive (Cranley–Patterson)
//! rotation, and successive frames advance the sequence index by a per-pixel
//! golden-ratio stride so the animated noise is itself low-discrepancy over
//! time (Wolfe/Heitz style animation, texture-free).
//!
//! This is intentionally labelled *blue-noise-like*: it is a genuine
//! low-discrepancy sampler, not a void-and-cluster STBN texture.

use super::sobol::hash_combine;

/// Reciprocal of the plastic constant \`p\` (root of \`x^3 = x + 1\`): the first R2
/// irrational stride \`1/p\`.
const R2_A1: f64 = 0.754_877_666_246_692_8;
/// Second R2 stride \`1/p^2\`.
const R2_A2: f64 = 0.569_840_290_998_040_3;
/// Golden-ratio fractional stride used to advance the sample index per frame.
const GOLDEN_FRACT: f64 = 0.618_033_988_749_894_8;

#[inline]
fn fract(x: f64) -> f64 {
    x - x.floor()
}

/// Converts a 32-bit integer to a unit \`f64\` in \`[0, 1)\` for use as an additive
/// rotation offset.
#[inline]
fn to_unit_f64(x: u32) -> f64 {
    x as f64 * (1.0 / 4_294_967_296.0)
}

/// Raw R2 point for sequence position \`index\`, \`(x, y)\` in \`[0, 1)^2\`.
#[inline]
pub fn r2_sample_2d(index: u32) -> (f32, f32) {
    let n = index as f64;
    let x = fract(0.5 + R2_A1 * n);
    let y = fract(0.5 + R2_A2 * n);
    (x as f32, y as f32)
}

/// Per-pixel, per-frame animated blue-noise-like 2-D sample.
///
/// * \`pixel\` — integer framebuffer coordinate (decorrelates neighbours).
/// * \`frame\` — temporal index; advances the sequence by a golden-ratio stride.
/// * \`sample\` — intra-frame sample index (for multi-ray budgets).
///
/// Returns \`(u, v)\` in \`[0, 1)^2\`.
#[inline]
pub fn animated_sample_2d(pixel: (u32, u32), frame: u32, sample: u32) -> (f32, f32) {
    // Per-pixel decorrelation seed → additive Cranley–Patterson rotation.
    let seed = hash_combine(pixel.0.wrapping_mul(0x9e37_79b9) ^ pixel.1, 0x68e3_1da4);
    let rot_x = to_unit_f64(seed);
    let rot_y = to_unit_f64(seed.rotate_left(16));

    // Advance the R2 index per frame by a per-pixel golden-ratio stride so the
    // temporal sequence is itself low-discrepancy (and never repeats a pixel's
    // offset in lock-step with its neighbours).
    let frame_advance = fract((frame as f64) * GOLDEN_FRACT * (1.0 + to_unit_f64(seed)));
    let (bx, by) = r2_sample_2d(sample);

    let u = fract(bx as f64 + rot_x + frame_advance);
    let v = fract(by as f64 + rot_y + frame_advance * R2_A2);
    (u as f32, v as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r2_values_are_in_range() {
        for i in 0..8192u32 {
            let (x, y) = r2_sample_2d(i);
            assert!((0.0..1.0).contains(&x), "x = {x}");
            assert!((0.0..1.0).contains(&y), "y = {y}");
        }
    }

    #[test]
    fn animated_values_are_in_range() {
        for frame in 0..64u32 {
            for s in 0..16u32 {
                let (u, v) = animated_sample_2d((37, 101), frame, s);
                assert!((0.0..1.0).contains(&u), "u = {u}");
                assert!((0.0..1.0).contains(&v), "v = {v}");
            }
        }
    }

    #[test]
    fn is_deterministic() {
        assert_eq!(animated_sample_2d((5, 9), 3, 1), animated_sample_2d((5, 9), 3, 1));
    }

    #[test]
    fn neighbouring_pixels_are_decorrelated() {
        let a = animated_sample_2d((10, 10), 0, 0);
        let b = animated_sample_2d((11, 10), 0, 0);
        assert_ne!(a, b, "adjacent pixels must receive different offsets");
    }

    #[test]
    fn successive_frames_differ() {
        let a = animated_sample_2d((4, 4), 0, 0);
        let b = animated_sample_2d((4, 4), 1, 0);
        assert_ne!(a, b, "the sequence must advance between frames");
    }

    #[test]
    fn r2_mean_approaches_one_half() {
        let n = 8192u32;
        let (mut sx, mut sy) = (0.0f64, 0.0f64);
        for i in 0..n {
            let (x, y) = r2_sample_2d(i);
            sx += x as f64;
            sy += y as f64;
        }
        assert!((sx / n as f64 - 0.5).abs() < 0.005);
        assert!((sy / n as f64 - 0.5).abs() < 0.005);
    }

    #[test]
    fn r2_point_set_is_well_spread_minimum_distance() {
        // A blue-noise-like set keeps points apart: the closest pair among the
        // first N points stays well above the white-noise expectation.  For R2
        // the min pairwise distance is ~0.7/sqrt(N).
        let n = 256usize;
        let pts: Vec<(f32, f32)> = (0..n as u32).map(r2_sample_2d).collect();
        let mut min_d2 = f32::INFINITY;
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = pts[i].0 - pts[j].0;
                let dy = pts[i].1 - pts[j].1;
                min_d2 = min_d2.min(dx * dx + dy * dy);
            }
        }
        let min_d = min_d2.sqrt();
        let bound = 0.5 / (n as f32).sqrt();
        assert!(min_d > bound, "min distance {min_d} below blue-noise bound {bound}");
    }
}
