//! Backend-neutral CPU golden for an NPR ordered-dithering (Bayer threshold)
//! posterisation pass — the classic retro / limited-palette look where a
//! per-pixel Bayer matrix biases a hard colour quantisation so that flat
//! gradients resolve into the ordered cross-hatch of an old indexed display.
//!
//! This is *ordered dithering*, not the `halftone` pass: `halftone` renders
//! tonal value as the area of round ink dots, while this pass thresholds each
//! channel against a fixed 4x4 Bayer matrix and then snaps the result to a small
//! number of discrete `levels` (a bit-depth / palette fit). The Bayer bias is
//! what turns the banding of a naive posterise into the familiar stable,
//! non-flickering dither pattern.
//!
//! Ordered dithering is a shared post-processing base, not a peer of the
//! PBR/NPR shading fronts: it runs on the resolved, pre-exposed radiance (see
//! [`crate::exposure`]) after the stylized fronts have written the same buffer,
//! so one implementation serves them all.
//!
//! The pipeline, applied per pixel by [`apply_ordered_dither`], is:
//!
//! * **Bayer threshold.** [`bayer4x4_threshold`] indexes the fixed 4x4 Bayer
//!   matrix by `(x mod 4, y mod 4)`, normalises the cell to `[0, 1)` by `/16`
//!   and recentres by `-0.5` to an approximately zero-mean bias in
//!   `[-0.5, 0.4375]`.
//! * **Dither offset + quantise.** Each channel is nudged by
//!   `threshold / levels` before [`quantize`] snaps it to the nearest of
//!   `levels` evenly spaced steps. The sub-step bias means neighbouring pixels
//!   round to different steps, so the average over a 4x4 tile tracks the input.
//! * **Strength blend.** The dithered result is mixed back over the original by
//!   an artist `strength` (`a + (b - a) * t`), so the effect fades continuously
//!   to the untouched scene.
//!
//! Everything is pure `f32` maths — no transcendental calls, only `floor` /
//! `clamp` / `min` / `max` — mirrored arm-for-arm by
//! `shaders/ordered_dither.wesl` (same Bayer constants element-for-element,
//! same quantiser, componentwise vector ops matching the per-channel Rust) so
//! the CPU golden and the GPU twin agree.

/// The fixed 4x4 Bayer (recursive dispersed-dot) matrix in row-major order,
/// values `0..16`. `M[y][x] / 16 - 0.5` is the ordered-dither threshold; the
/// 16 distinct values give the maximally spread-out dither of the classic
/// `Bayer` construction.
pub const ORDERED_DITHER_BAYER_4X4: [f32; 16] = [
    0.0, 8.0, 2.0, 10.0, //
    12.0, 4.0, 14.0, 6.0, //
    3.0, 11.0, 1.0, 9.0, //
    15.0, 7.0, 13.0, 5.0, //
];

/// Ordered-dither threshold for pixel `(x, y)`: index the fixed 4x4 Bayer
/// matrix by `(x mod 4, y mod 4)` (via `& 3`), normalise the cell to `[0, 1)`
/// by `/16` and recentre by `-0.5`. The result is a stable per-pixel bias in
/// `[-0.5, 0.4375]` that tiles every 4 pixels.
#[must_use]
pub fn bayer4x4_threshold(x: u32, y: u32) -> f32 {
    let ix = (x & 3) as usize;
    let iy = (y & 3) as usize;
    ORDERED_DITHER_BAYER_4X4[iy * 4 + ix] / 16.0 - 0.5
}

/// Snap `scalar` to the nearest of `levels` evenly spaced steps:
/// `floor(scalar * (levels - 1) + 0.5) / (levels - 1)`. With `levels` steps the
/// output lands on a multiple of `1 / (levels - 1)` (so `levels == 2` is a pure
/// black/white threshold at `0.5`). `levels` is clamped up to `2` so the
/// denominator is never zero.
#[must_use]
pub fn quantize(scalar: f32, levels: u32) -> f32 {
    let denom = (levels.max(2) - 1) as f32;
    (scalar * denom + 0.5).floor() / denom
}

/// Apply ordered dithering to one pixel: bias each channel by the Bayer
/// `threshold / levels`, [`quantize`] to `levels` steps, then blend the
/// posterised colour back over the input by `strength` (`a + (b - a) * t`).
///
/// A disabled pass ([`OrderedDitherParams::enabled`] `false`) is the identity,
/// as is `strength == 0`. `px` / `py` are the integer pixel coordinates that
/// index the Bayer matrix.
#[must_use]
pub fn apply_ordered_dither(
    scene: [f32; 3],
    px: u32,
    py: u32,
    params: &OrderedDitherParams,
) -> [f32; 3] {
    if !params.enabled {
        return scene;
    }
    let threshold = bayer4x4_threshold(px, py);
    let offset = threshold / params.levels as f32;
    let mut out = scene;
    for c in 0..3 {
        let dithered = quantize(scene[c] + offset, params.levels);
        out[c] = scene[c] + (dithered - scene[c]) * params.strength;
    }
    out
}

/// Artist controls for the ordered-dither pass. `Default` is a disabled,
/// identity pass: `enabled == false` leaves the scene untouched regardless of
/// the other fields.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrderedDitherParams {
    /// Number of discrete quantisation steps per channel (the palette / bit
    /// depth). Clamped up to `2` inside [`quantize`].
    pub levels: u32,
    /// Blend weight of the dithered result over the input (`a + (b - a) * t`).
    /// `0` is the identity.
    pub strength: f32,
    /// Master enable; `false` short-circuits the pass to the identity.
    pub enabled: bool,
}

impl Default for OrderedDitherParams {
    fn default() -> Self {
        // Disabled retro default: a 4-level palette at full strength, but off,
        // so the pass is the identity until an artist enables it.
        Self {
            levels: 4,
            strength: 1.0,
            enabled: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1.0e-5, "expected {b}, got {a}");
    }

    fn approx3(a: [f32; 3], b: [f32; 3]) {
        approx(a[0], b[0]);
        approx(a[1], b[1]);
        approx(a[2], b[2]);
    }

    // --- bayer4x4_threshold ---

    #[test]
    fn bayer4x4_threshold_known_values() {
        // (0,0) = 0/16 - 0.5, (1,0) = 8/16 - 0.5 = 0, (0,1) = 12/16 - 0.5.
        approx(bayer4x4_threshold(0, 0), -0.5);
        approx(bayer4x4_threshold(1, 0), 0.0);
        approx(bayer4x4_threshold(0, 1), 0.25);
        approx(bayer4x4_threshold(3, 3), 5.0 / 16.0 - 0.5);
    }

    #[test]
    fn bayer4x4_threshold_wraps_mod4() {
        // Coordinates fold every 4 pixels via the `& 3` index.
        approx(bayer4x4_threshold(4, 4), bayer4x4_threshold(0, 0));
        approx(bayer4x4_threshold(5, 0), bayer4x4_threshold(1, 0));
        approx(bayer4x4_threshold(0, 9), bayer4x4_threshold(0, 1));
        approx(bayer4x4_threshold(103, 402), bayer4x4_threshold(3, 2));
    }

    #[test]
    fn bayer4x4_threshold_in_range() {
        // Every cell recenters into [-0.5, 0.5).
        for y in 0..4 {
            for x in 0..4 {
                let t = bayer4x4_threshold(x, y);
                assert!((-0.5..0.5).contains(&t), "threshold {t} out of range");
            }
        }
    }

    #[test]
    fn bayer4x4_threshold_all_distinct() {
        // The 16 Bayer cells are all different (maximal dispersion).
        let mut seen: Vec<f32> = Vec::new();
        for y in 0u32..4 {
            for x in 0u32..4 {
                let t = bayer4x4_threshold(x, y);
                for &s in &seen {
                    assert!((t - s).abs() > 1.0e-4, "duplicate threshold {t}");
                }
                seen.push(t);
            }
        }
        assert_eq!(seen.len(), 16);
    }

    // --- quantize ---

    #[test]
    fn quantize_two_levels_is_threshold_at_half() {
        // levels = 2 -> denom 1 -> snap to 0 or 1 about 0.5.
        approx(quantize(0.2, 2), 0.0);
        approx(quantize(0.49, 2), 0.0);
        approx(quantize(0.5, 2), 1.0);
        approx(quantize(0.8, 2), 1.0);
    }

    #[test]
    fn quantize_preserves_endpoints() {
        for &levels in &[2, 3, 4, 8] {
            approx(quantize(0.0, levels), 0.0);
            approx(quantize(1.0, levels), 1.0);
        }
    }

    #[test]
    fn quantize_lands_on_discrete_steps() {
        // Output is always a multiple of 1 / (levels - 1).
        let levels = 4;
        let denom = (levels - 1) as f32;
        for i in 0..=20 {
            let q = quantize(i as f32 / 20.0, levels);
            let step = q * denom;
            approx(step, step.floor());
        }
    }

    #[test]
    fn quantize_three_levels_midpoints() {
        // denom = 2: 0.5 rounds to the middle step 0.5; 0.24 -> 0.0; 0.26 -> 0.5.
        approx(quantize(0.5, 3), 0.5);
        approx(quantize(0.24, 3), 0.0);
        approx(quantize(0.26, 3), 0.5);
    }

    #[test]
    fn quantize_levels_below_two_is_guarded() {
        // levels 0 and 1 clamp up to 2 (no divide-by-zero, same as levels 2).
        approx(quantize(0.7, 0), quantize(0.7, 2));
        approx(quantize(0.7, 1), quantize(0.7, 2));
    }

    // --- apply_ordered_dither ---

    #[test]
    fn apply_disabled_is_identity() {
        let params = OrderedDitherParams::default();
        let scene = [0.15, 0.4, 0.85];
        approx3(apply_ordered_dither(scene, 1, 2, &params), scene);
    }

    #[test]
    fn apply_strength_zero_is_identity() {
        let params = OrderedDitherParams {
            levels: 4,
            strength: 0.0,
            enabled: true,
        };
        let scene = [0.15, 0.4, 0.85];
        approx3(apply_ordered_dither(scene, 1, 2, &params), scene);
    }

    #[test]
    fn apply_strength_one_is_full_quantize() {
        let params = OrderedDitherParams {
            levels: 4,
            strength: 1.0,
            enabled: true,
        };
        let scene = [0.15, 0.4, 0.85];
        let offset = bayer4x4_threshold(3, 5) / params.levels as f32;
        let expected = [
            quantize(scene[0] + offset, params.levels),
            quantize(scene[1] + offset, params.levels),
            quantize(scene[2] + offset, params.levels),
        ];
        approx3(apply_ordered_dither(scene, 3, 5, &params), expected);
    }

    #[test]
    fn apply_strength_half_blends() {
        let params = OrderedDitherParams {
            levels: 3,
            strength: 0.5,
            enabled: true,
        };
        let scene = [0.3, 0.6, 0.9];
        let full = {
            let mut p = params;
            p.strength = 1.0;
            apply_ordered_dither(scene, 2, 2, &p)
        };
        let half = apply_ordered_dither(scene, 2, 2, &params);
        for c in 0..3 {
            approx(half[c], scene[c] + (full[c] - scene[c]) * 0.5);
        }
    }

    #[test]
    fn apply_is_per_channel_independent() {
        // The same Bayer offset applies to every channel; quantisation is
        // otherwise per-channel, so a grey stays grey.
        let params = OrderedDitherParams {
            levels: 4,
            strength: 1.0,
            enabled: true,
        };
        let out = apply_ordered_dither([0.5, 0.5, 0.5], 2, 3, &params);
        approx(out[0], out[1]);
        approx(out[1], out[2]);
    }

    #[test]
    fn apply_pixel_wraps_mod4() {
        let params = OrderedDitherParams {
            levels: 4,
            strength: 1.0,
            enabled: true,
        };
        let scene = [0.2, 0.55, 0.8];
        approx3(
            apply_ordered_dither(scene, 1, 1, &params),
            apply_ordered_dither(scene, 5, 5, &params),
        );
    }

    #[test]
    fn apply_dither_offset_biases_quantization() {
        // Two Bayer cells with different thresholds can round a flat mid-grey to
        // different steps, which is the whole point of ordered dithering.
        let params = OrderedDitherParams {
            levels: 2,
            strength: 1.0,
            enabled: true,
        };
        let grey = [0.5, 0.5, 0.5];
        // Cell (0,0): threshold -0.5 -> offset -0.25 -> quantize(0.25, 2) = 0.
        let dark = apply_ordered_dither(grey, 0, 0, &params);
        // Cell (3,0): threshold 15/16 - 0.5 = 0.4375 -> offset ~0.219 ->
        // quantize(0.719, 2) = 1.
        let bright = apply_ordered_dither(grey, 3, 0, &params);
        approx3(dark, [0.0, 0.0, 0.0]);
        approx3(bright, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn quantize_is_monotonic_nondecreasing() {
        let mut prev = -1.0_f32;
        for i in 0..=100 {
            let q = quantize(i as f32 / 100.0, 5);
            assert!(q >= prev - 1.0e-6, "quantize decreased at {i}");
            prev = q;
        }
    }

    #[test]
    fn quantize_eight_levels_step_is_one_seventh() {
        // levels = 8 -> denom 7 -> the smallest non-zero step is 1/7.
        approx(quantize(1.0 / 7.0, 8), 1.0 / 7.0);
        approx(quantize(1.0 / 7.0 + 0.01, 8), 1.0 / 7.0);
    }

    #[test]
    fn bayer4x4_threshold_matches_const_table() {
        // The helper is exactly the shared const table normalised and recentred.
        for y in 0..4u32 {
            for x in 0..4u32 {
                let raw = ORDERED_DITHER_BAYER_4X4[(y * 4 + x) as usize];
                approx(bayer4x4_threshold(x, y), raw / 16.0 - 0.5);
            }
        }
    }

    #[test]
    fn bayer4x4_threshold_mean_is_near_zero() {
        // v/16 - 0.5 has a mean of 7.5/16 - 0.5 = -0.03125 (approximately zero).
        let mut sum = 0.0;
        for y in 0..4 {
            for x in 0..4 {
                sum += bayer4x4_threshold(x, y);
            }
        }
        approx(sum / 16.0, -0.03125);
    }

    #[test]
    fn apply_black_and_white_are_stable() {
        // Pure 0 and 1 quantise to themselves for any Bayer cell (offset aside,
        // the endpoints are fixed points of the two-level threshold here).
        let params = OrderedDitherParams { levels: 4, strength: 1.0, enabled: true };
        approx3(apply_ordered_dither([0.0, 0.0, 0.0], 2, 1, &params), [0.0, 0.0, 0.0]);
        approx3(apply_ordered_dither([1.0, 1.0, 1.0], 2, 1, &params), [1.0, 1.0, 1.0]);
    }

    // --- params ---

    #[test]
    fn ordered_dither_params_default_is_disabled_identity() {
        let p = OrderedDitherParams::default();
        assert!(!p.enabled);
        assert_eq!(p.levels, 4);
        approx(p.strength, 1.0);
        // Disabled leaves the scene untouched regardless of levels/strength.
        approx3(apply_ordered_dither([0.3, 0.3, 0.3], 2, 2, &p), [0.3, 0.3, 0.3]);
    }
}
