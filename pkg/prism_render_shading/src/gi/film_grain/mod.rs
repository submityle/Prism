//! Film-grain and sensor-noise CPU golden references.
//!
//! Deterministic, GPU-free photographic grain synthesis, distinct from the
//! ordered/blue-noise dithering in [`crate::gi::sharpen::deband`]:
//!
//! * [`grain`] — Newson-style stochastic film-grain intensity with luminance
//!   response (shadows/mids grain more than highlights).
//! * [`sensor`] — Poisson-Gaussian photon-shot + read-noise sensor model.
//! * [`response`] — grain blending curves (overlay / signal-dependent gain)
//!   and ISO-driven strength mapping.
//!
//! [`apply_film_grain`] is the high-level composite that ties the three
//! together: it derives the grain `strength` from the ISO, builds a
//! luminance-weighted zero-mean grain field, overlay-composites it into the
//! colour, then adds the Poisson-Gaussian sensor noise on top — all from a
//! single deterministic seed so the WESL/GPU twin can reproduce it exactly.
//!
//! # Conventions
//! * Deterministic pure functions: no RNG, IO, GPU, global state, or `unsafe`.
//! * Colours and signals are display-referred and sanitized to `[0, 1]`.
//! * `grain_scale` and all derived strengths are clamped to `[0, 1]`.
//! * Every output channel is finite and within `[0, 1]`.

pub mod grain;
pub mod response;
pub mod sensor;

use bevy_math::{UVec2, Vec3};

pub use grain::{film_grain, luminance};
pub use response::{apply_grain, iso_to_strength};
pub use sensor::{apply_sensor_noise, noise_std, noise_variance};

/// Per-channel seed offsets so the three channels get decorrelated sensor
/// noise while sharing one luminance-coupled grain field.
const CHANNEL_OFFSETS: [u32; 3] = [0x0000_0000, 0x51ED_270B, 0xA341_316C];

/// Replaces a non-finite scalar with `fallback`.
#[inline]
#[must_use]
fn finite_or(x: f32, fallback: f32) -> f32 {
    if x.is_finite() { x } else { fallback }
}

/// Composites film grain and sensor noise onto a display-referred colour.
///
/// Pipeline, all driven by the single `seed` and pixel coordinate `pixel`:
///
/// 1. Sanitize `color` to `[0, 1]` and take its Rec.709 [`luminance`].
/// 2. Derive the grain `strength` as `iso_to_strength(iso) · grain_scale`.
/// 3. Build a luminance-weighted, zero-mean grain sample (strong in
///    shadows/mids, zero at white) and overlay-composite it into the colour
///    with [`apply_grain`]. Overlay fixes the `0` / `1` endpoints, so this step
///    cannot leave `[0, 1]`.
/// 4. Add Poisson-Gaussian [`sensor`] noise per channel (shot variance ∝ the
///    channel signal, plus the `read_noise` floor, both ISO-scaled), with a
///    decorrelated seed per channel, and clamp back to `[0, 1]`.
///
/// `grain_scale` is clamped to `[0, 1]`. The result is deterministic in all
/// inputs and every channel is finite and within `[0, 1]`.
///
/// A pure-black pixel with `read_noise = 0` is preserved exactly: overlay fixes
/// the `0` endpoint and the shot-noise variance vanishes at zero signal.
#[must_use]
pub fn apply_film_grain(
    color: Vec3,
    pixel: UVec2,
    iso: f32,
    read_noise: f32,
    seed: u32,
    grain_scale: f32,
) -> Vec3 {
    let base = Vec3::new(
        finite_or(color.x, 0.0).clamp(0.0, 1.0),
        finite_or(color.y, 0.0).clamp(0.0, 1.0),
        finite_or(color.z, 0.0).clamp(0.0, 1.0),
    );

    let luma = luminance(base);
    let scale = finite_or(grain_scale, 0.0).clamp(0.0, 1.0);
    let strength = (iso_to_strength(iso) * scale).clamp(0.0, 1.0);

    // Unit-strength, luminance-weighted zero-mean grain; `strength` is applied
    // by the overlay composite so there is a single strength knob.
    let grain = film_grain(pixel, luma, seed, 1.0);
    let grained = apply_grain(base, grain, strength);

    // Per-channel Poisson-Gaussian sensor noise with decorrelated seeds.
    Vec3::new(
        apply_sensor_noise(grained.x, iso, read_noise, pixel, seed ^ CHANNEL_OFFSETS[0]),
        apply_sensor_noise(grained.y, iso, read_noise, pixel, seed ^ CHANNEL_OFFSETS[1]),
        apply_sensor_noise(grained.z, iso, read_noise, pixel, seed ^ CHANNEL_OFFSETS[2]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The composite is a deterministic pure function of all its inputs.
    #[test]
    fn apply_film_grain_is_deterministic() {
        let c = Vec3::new(0.3, 0.5, 0.7);
        let a = apply_film_grain(c, UVec2::new(12, 34), 800.0, 0.02, 7, 1.0);
        let b = apply_film_grain(c, UVec2::new(12, 34), 800.0, 0.02, 7, 1.0);
        assert_eq!(a, b);
    }

    /// Every output channel stays finite and within `[0, 1]`, even for
    /// out-of-range colours and extreme ISO / read-noise.
    #[test]
    fn apply_film_grain_stays_in_range() {
        let colors = [
            Vec3::new(0.0, 0.5, 1.0),
            Vec3::new(-1.0, 2.0, f32::NAN),
            Vec3::splat(f32::INFINITY),
        ];
        for &c in &colors {
            for &iso in &[100.0_f32, 1600.0, 12_800.0] {
                for y in 0..16u32 {
                    for x in 0..16u32 {
                        let out = apply_film_grain(c, UVec2::new(x, y), iso, 0.1, 3, 1.0);
                        assert!(
                            (0.0..=1.0).contains(&out.x)
                                && (0.0..=1.0).contains(&out.y)
                                && (0.0..=1.0).contains(&out.z),
                            "out={out:?} c={c:?} iso={iso}"
                        );
                    }
                }
            }
        }
    }

    /// Pure black with zero read-noise is preserved exactly: overlay fixes the
    /// `0` endpoint and shot-noise variance vanishes at zero signal.
    #[test]
    fn black_is_preserved_without_read_noise() {
        for &iso in &[100.0_f32, 800.0, 6400.0] {
            for y in 0..8u32 {
                for x in 0..8u32 {
                    let out = apply_film_grain(Vec3::ZERO, UVec2::new(x, y), iso, 0.0, 5, 1.0);
                    assert_eq!(out, Vec3::ZERO, "black moved at iso={iso} ({x},{y})");
                }
            }
        }
    }

    /// With no grain scale and no read noise, a mid-grey still receives some
    /// shot noise (sensor noise is always present), but the mean over a tile
    /// stays close to the input — the noise is zero-mean.
    #[test]
    fn mean_is_approximately_preserved() {
        let c = Vec3::splat(0.5);
        let mut acc = Vec3::ZERO;
        let mut n = 0.0_f32;
        for y in 0..48u32 {
            for x in 0..48u32 {
                acc += apply_film_grain(c, UVec2::new(x, y), 400.0, 0.0, 9, 1.0);
                n += 1.0;
            }
        }
        let mean = acc / n;
        assert!((mean.x - 0.5).abs() < 0.02, "mean={mean:?}");
        assert!((mean.y - 0.5).abs() < 0.02, "mean={mean:?}");
        assert!((mean.z - 0.5).abs() < 0.02, "mean={mean:?}");
    }

    /// Higher ISO produces a larger average per-pixel deviation from the input
    /// (more grain + more sensor noise).
    #[test]
    fn higher_iso_adds_more_noise() {
        let c = Vec3::splat(0.5);
        let avg_dev = |iso: f32| {
            let mut acc = 0.0_f64;
            let mut n = 0.0_f64;
            for y in 0..48u32 {
                for x in 0..48u32 {
                    let out = apply_film_grain(c, UVec2::new(x, y), iso, 0.02, 1, 1.0);
                    acc += (out - c).abs().element_sum() as f64;
                    n += 1.0;
                }
            }
            acc / n
        };
        let low = avg_dev(200.0);
        let high = avg_dev(6400.0);
        assert!(high > low, "expected more noise at high ISO: low={low} high={high}");
    }
}
