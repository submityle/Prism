//! **RCAS** — Robust Contrast-Adaptive Sharpening — the FSR sharpening pass
//! temporal upscaling runs on its reconstructed output.
//!
//! After the temporal accumulation resolves a display-resolution image the
//! result is slightly soft (the reconstruction kernel and history blend both
//! low-pass the signal), so upscalers finish with a sharpening pass. RCAS is the
//! FSR variant tuned for *upscaled* input: unlike plain CAS (see [`crate::cas`])
//! it uses only the **five-tap cross** (no corners), derives a single shared
//! sharpening *lobe* from the per-channel contrast limits, and clamps that lobe
//! to [`RCAS_LIMIT`] so it lifts detail without the ringing haloes a naive
//! unsharp mask leaves on a reconstructed edge.
//!
//! The kernel, per FSR's `FsrRcasF`:
//!
//! ```text
//!     b
//!   d e f      (centre e)
//!     h
//! ```
//!
//! 1. Take the per-channel min/max over the ring `{b, d, f, h}`.
//! 2. Form the limiters `hitMin = mn / (4·mx)` and
//!    `hitMax = (1 - mx) / (4·mn - 4)`, and `lobeRGB = max(-hitMin, hitMax)`.
//! 3. Reduce to one lobe `clamp(max_channel(lobeRGB), -RCAS_LIMIT, 0)·sharp`,
//!    optionally attenuated by the noise estimate [`rcas_noise`].
//! 4. Resolve `((b + d + f + h)·lobe + e) / (4·lobe + 1)`.
//!
//! `sharp` is [`TemporalUpscaleSettings::sharpness`](crate::TemporalUpscaleSettings)
//! taken directly as a `[0, 1]` lobe scale, so `sharpness == 0` is an exact
//! identity (no sharpening) and `1` is full RCAS. A flat neighbourhood is a
//! fixed point at every sharpness. Everything is plain arithmetic, mirrored
//! bit-for-bit by the GPU twin.

use bevy_math::Vec3;

/// The FSR RCAS lobe clamp `0.25 - 1/16`. Capping the sharpening lobe here is
/// what keeps RCAS "robust": it bounds the overshoot so a high-contrast edge
/// cannot ring.
pub const RCAS_LIMIT: f32 = 0.25 - 1.0 / 16.0;

/// The five-tap cross RCAS operates on, in the order `[up, left, centre, right,
/// down]` (`[b, d, e, f, h]`). The centre `e` is the pixel being sharpened.
pub type CrossTaps = [Vec3; 5];

/// Tunables for the RCAS pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RcasParams {
    /// Lobe scale in `[0, 1]`, taken from
    /// [`TemporalUpscaleSettings::sharpness`](crate::TemporalUpscaleSettings).
    /// `0` disables sharpening (exact identity); `1` is full-strength RCAS.
    pub sharpness: f32,
    /// Whether to attenuate the lobe by the [`rcas_noise`] estimate, which pulls
    /// sharpening back in noisy regions so RCAS does not amplify grain.
    pub denoise: bool,
}

impl Default for RcasParams {
    fn default() -> Self {
        Self {
            sharpness: 0.0,
            denoise: false,
        }
    }
}

/// FSR's cheap luma proxy `g + 0.5·(r + b)` (luma times two), used only for the
/// noise estimate so it needs no exact weighting.
#[must_use]
pub fn rcas_luma(rgb: Vec3) -> f32 {
    rgb.y + 0.5 * (rgb.x + rgb.z)
}

/// The RCAS noise attenuation factor in `[0.5, 1]` for the cross.
///
/// It measures how much the centre luma departs from the ring average relative
/// to the ring's luma spread: a lone bright/dark sample (noise) yields a large
/// normalised departure and a factor near `0.5` (halve the sharpening), while a
/// coherent edge yields a factor near `1`. A flat (zero-spread) ring returns `1`
/// (nothing to denoise).
#[must_use]
pub fn rcas_noise(taps: &CrossTaps) -> f32 {
    let (b, d, e, f, h) = (
        rcas_luma(taps[0]),
        rcas_luma(taps[1]),
        rcas_luma(taps[2]),
        rcas_luma(taps[3]),
        rcas_luma(taps[4]),
    );
    let spread = b.max(d).max(e).max(f).max(h) - b.min(d).min(e).min(f).min(h);
    if spread < 1.0e-6 {
        return 1.0;
    }
    let departure = 0.25 * (b + d + f + h) - e;
    let nz = (departure.abs() / spread).clamp(0.0, 1.0);
    -0.5 * nz + 1.0
}

/// The full RCAS kernel over a five-tap cross at the given [`RcasParams`].
///
/// Returns the centre unchanged when `sharpness == 0` or the neighbourhood is
/// flat. All limiter denominators are guarded so a black or white flat ring is
/// an exact identity rather than a NaN.
#[must_use]
pub fn rcas(taps: &CrossTaps, params: &RcasParams) -> Vec3 {
    let sharp = params.sharpness.clamp(0.0, 1.0);
    if sharp <= 0.0 {
        return taps[2];
    }
    let (b, d, e, f, h) = (taps[0], taps[1], taps[2], taps[3], taps[4]);

    // Per-channel min/max over the ring {b, d, f, h}.
    let mn = b.min(d).min(f).min(h);
    let mx = b.max(d).max(f).max(h);

    // Limiters. Guard the denominators away from zero (preserving their sign:
    // `4·mn - 4` is non-positive for a display-referred `[0, 1]` signal).
    let eps = 1.0e-6;
    let four_mx = (4.0 * mx).max(Vec3::splat(eps));
    let four_mn_minus_four = (4.0 * mn - Vec3::splat(4.0)).min(Vec3::splat(-eps));
    let hit_min = mn / four_mx;
    let hit_max = (Vec3::ONE - mx) / four_mn_minus_four;
    let lobe_rgb = (-hit_min).max(hit_max);

    // One shared lobe: the sharpest (most negative overshoot) channel, clamped
    // into `[-RCAS_LIMIT, 0]`, scaled by the sharpness knob.
    let lobe_channel = lobe_rgb.x.max(lobe_rgb.y).max(lobe_rgb.z);
    let mut lobe = lobe_channel.clamp(-RCAS_LIMIT, 0.0) * sharp;
    if params.denoise {
        lobe *= rcas_noise(taps);
    }

    // Resolve. `lobe` is non-positive, so `4·lobe + 1 >= 1 - 4·RCAS_LIMIT = 0.25`
    // stays safely positive.
    let rcp = 1.0 / (4.0 * lobe + 1.0);
    ((b + d + f + h) * lobe + e) * rcp
}

/// Convenience wrapper over [`rcas`] taking the taps by value.
#[must_use]
pub fn apply_rcas(taps: CrossTaps, params: RcasParams) -> Vec3 {
    rcas(&taps, &params)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: Vec3, b: Vec3) -> bool {
        (a - b).abs().max_element() < 1.0e-5
    }

    fn cross(up: f32, left: f32, centre: f32, right: f32, down: f32) -> CrossTaps {
        [
            Vec3::splat(up),
            Vec3::splat(left),
            Vec3::splat(centre),
            Vec3::splat(right),
            Vec3::splat(down),
        ]
    }

    #[test]
    fn zero_sharpness_is_identity() {
        let taps = cross(0.2, 0.3, 0.7, 0.1, 0.4);
        let out = rcas(&taps, &RcasParams::default());
        assert!(approx(out, taps[2]), "sharpness 0 must pass the centre through: {out:?}");
    }

    #[test]
    fn flat_ring_is_a_fixed_point_at_every_sharpness() {
        let taps = cross(0.5, 0.5, 0.5, 0.5, 0.5);
        for &s in &[0.0, 0.25, 0.5, 1.0] {
            let out = rcas(&taps, &RcasParams { sharpness: s, denoise: false });
            assert!(approx(out, Vec3::splat(0.5)), "flat must be fixed at s={s}: {out:?}");
        }
    }

    #[test]
    fn black_flat_ring_is_identity_not_nan() {
        let taps = cross(0.0, 0.0, 0.0, 0.0, 0.0);
        let out = rcas(&taps, &RcasParams { sharpness: 1.0, denoise: false });
        assert!(out.x.is_finite() && approx(out, Vec3::ZERO), "{out:?}");
    }

    #[test]
    fn white_flat_ring_is_identity_not_nan() {
        let taps = cross(1.0, 1.0, 1.0, 1.0, 1.0);
        let out = rcas(&taps, &RcasParams { sharpness: 1.0, denoise: false });
        assert!(out.x.is_finite() && approx(out, Vec3::ONE), "{out:?}");
    }

    #[test]
    fn bright_centre_is_lifted() {
        // Centre brighter than the ring => sharpening should raise it further.
        let taps = cross(0.4, 0.4, 0.6, 0.4, 0.4);
        let out = rcas(&taps, &RcasParams { sharpness: 1.0, denoise: false });
        assert!(out.x > 0.6, "a bright centre must be lifted, got {out:?}");
    }

    #[test]
    fn dim_centre_is_pushed_down() {
        let taps = cross(0.6, 0.6, 0.4, 0.6, 0.6);
        let out = rcas(&taps, &RcasParams { sharpness: 1.0, denoise: false });
        assert!(out.x < 0.4, "a dim centre must be pushed darker, got {out:?}");
    }

    #[test]
    fn stronger_sharpness_lifts_more() {
        let taps = cross(0.4, 0.4, 0.6, 0.4, 0.4);
        let gentle = rcas(&taps, &RcasParams { sharpness: 0.25, denoise: false }).x;
        let strong = rcas(&taps, &RcasParams { sharpness: 1.0, denoise: false }).x;
        assert!(strong > gentle, "stronger knob must sharpen more: {strong} vs {gentle}");
        assert!(gentle > 0.6, "even a gentle knob lifts a bright centre: {gentle}");
    }

    #[test]
    fn output_stays_finite_on_a_hard_edge() {
        let taps = cross(1.0, 0.0, 0.5, 1.0, 0.0);
        let out = rcas(&taps, &RcasParams { sharpness: 1.0, denoise: false });
        assert!(out.x.is_finite() && out.y.is_finite() && out.z.is_finite(), "{out:?}");
    }

    #[test]
    fn noise_factor_is_unity_on_a_flat_ring() {
        let taps = cross(0.3, 0.3, 0.3, 0.3, 0.3);
        assert!((rcas_noise(&taps) - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn noise_factor_halves_on_a_lone_spike() {
        // Centre spikes far above a flat ring => maximal departure => 0.5.
        let taps = cross(0.0, 0.0, 1.0, 0.0, 0.0);
        let nz = rcas_noise(&taps);
        assert!((nz - 0.5).abs() < 1.0e-5, "a lone spike should halve sharpening, got {nz}");
    }

    #[test]
    fn denoise_reduces_the_applied_sharpening() {
        // A noisy centre spike: denoise should pull the sharpened result back
        // toward the centre relative to the un-denoised pass.
        let taps = cross(0.2, 0.2, 0.8, 0.2, 0.2);
        let plain = rcas(&taps, &RcasParams { sharpness: 1.0, denoise: false }).x;
        let denoised = rcas(&taps, &RcasParams { sharpness: 1.0, denoise: true }).x;
        assert!(
            (denoised - 0.8).abs() < (plain - 0.8).abs(),
            "denoise should temper sharpening: plain {plain}, denoised {denoised}"
        );
    }

    #[test]
    fn per_channel_sharpening_is_independent() {
        let taps = [
            Vec3::new(0.4, 0.5, 0.6),
            Vec3::new(0.4, 0.5, 0.6),
            Vec3::new(0.6, 0.5, 0.4),
            Vec3::new(0.4, 0.5, 0.6),
            Vec3::new(0.4, 0.5, 0.6),
        ];
        let out = rcas(&taps, &RcasParams { sharpness: 1.0, denoise: false });
        assert!(out.x > 0.6, "red centre above its ring is lifted: {out:?}");
        assert!((out.y - 0.5).abs() < 1.0e-4, "flat green channel is identity: {out:?}");
        assert!(out.z < 0.4, "blue centre below its ring is pushed down: {out:?}");
    }

    #[test]
    fn apply_matches_the_borrowed_kernel() {
        let taps = cross(0.4, 0.4, 0.6, 0.4, 0.4);
        let params = RcasParams { sharpness: 0.8, denoise: true };
        assert!(approx(apply_rcas(taps, params), rcas(&taps, &params)));
    }
}
