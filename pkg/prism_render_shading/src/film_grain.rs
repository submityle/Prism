//! Backend-neutral CPU golden for film grain (sensor/emulsion noise).
//!
//! Film grain is a shared post-processing base, not a peer of the PBR/NPR
//! shading fronts: it runs on the resolved, pre-exposed HDR radiance (see
//! [`crate::exposure`]) and adds a per-pixel, per-frame noise that emulates the
//! stochastic silver-halide grains of photographic film (or sensor noise),
//! reintroducing the high-frequency texture a clean render lacks. Every
//! illumination model — physically based or stylized — writes into the same
//! HDR buffer this pass reads, so one implementation serves them all.
//!
//! The model is the modern AAA "hash noise + luminance response + intensity
//! mix" (as used by UE's `FilmGrain` and countless post stacks):
//!
//! * **Deterministic hash noise.** A screen/time coordinate is hashed to a
//!   uniform value in `[0, 1)` by Dave Hoskins' *Hash without Sine* `hash12`
//!   (constants `0.1031` / `33.33`). It is pure multiply/add/`fract` — no
//!   transcendental `sin`, which both keeps it clippy-clean and gives a
//!   `libm`-independent, deterministic result identical on CPU and GPU.
//! * **Signed grain.** The hash is remapped to `[-1, 1)` so it can add or
//!   subtract light around a pixel without a DC bias.
//! * **Luminance response.** Real emulsion shows grain most in the shadows and
//!   least in the highlights, so a self-written `smoothstep` biases the grain
//!   weight toward darker luma (`Rec. 709`), with a `response` knob blending
//!   from a flat weight (`0`) to the full shadow-biased falloff (`1`).
//! * **Apply.** The weighted grain is added to the radiance, scaled by an
//!   artist `intensity`; a monochrome grain shares one value across channels
//!   while a coloured grain hashes each channel with an offset seed.
//!
//! Everything is pure `f32` maths — no transcendental calls — mirrored
//! arm-for-arm by `shaders/film_grain.wesl`, so the CPU golden and the GPU twin
//! agree. `fract` is written `x - floor(x)` so it matches WGSL's `fract`
//! (which floors) for negative inputs too, keeping the twins bit-identical.

/// `Rec. 709` luma weights (linear `sRGB` primaries).
pub const FILM_GRAIN_LUMA_WEIGHTS: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Fractional part `x - floor(x)`, matching WGSL's `fract` (which floors, so
/// `fract(-0.25) == 0.75`) rather than Rust's truncating `f32::fract`. Kept as
/// a shared helper so the CPU golden and GPU twin agree for negative inputs.
#[must_use]
pub fn grain_fract(x: f32) -> f32 {
    x - x.floor()
}

/// `Rec. 709` relative luma of a linear RGB radiance sample.
#[must_use]
pub fn luminance(rgb: [f32; 3]) -> f32 {
    rgb[0] * FILM_GRAIN_LUMA_WEIGHTS[0]
        + rgb[1] * FILM_GRAIN_LUMA_WEIGHTS[1]
        + rgb[2] * FILM_GRAIN_LUMA_WEIGHTS[2]
}

/// Dave Hoskins' *Hash without Sine* `hash12`: hashes a 2D coordinate to a
/// uniform pseudo-random value in `[0, 1)`. Pure multiply/add/`fract` (no `sin`)
/// with the canonical constants `0.1031` and `33.33`, so it is deterministic
/// and `libm`-independent and mirrors the GPU twin exactly.
#[must_use]
pub fn hash12(p: [f32; 2]) -> f32 {
    // p3 = fract(vec3(p.x, p.y, p.x) * 0.1031)
    let mut px = grain_fract(p[0] * 0.1031);
    let mut py = grain_fract(p[1] * 0.1031);
    let mut pz = grain_fract(p[0] * 0.1031);
    // p3 += dot(p3, p3.yzx + 33.33)
    let d = px * (py + 33.33) + py * (pz + 33.33) + pz * (px + 33.33);
    px += d;
    py += d;
    pz += d;
    // return fract((p3.x + p3.y) * p3.z)
    grain_fract((px + py) * pz)
}

/// Signed grain value in `[-1, 1)` for a screen coordinate. The UV is scaled by
/// `resolution` (so grain is per-pixel, not stretched with the image) and
/// offset by `time_seed` (so it animates per frame), hashed, then remapped from
/// `[0, 1)` to `[-1, 1)`.
#[must_use]
pub fn grain(uv: [f32; 2], resolution: [f32; 2], time_seed: f32) -> f32 {
    let p = [
        uv[0] * resolution[0] + time_seed,
        uv[1] * resolution[1] + time_seed,
    ];
    hash12(p) * 2.0 - 1.0
}

/// Self-written `smoothstep`: `0` at/below `edge0`, `1` at/above `edge1`, and
/// the Hermite `3t^2 - 2t^3` in between. Written out (not the intrinsic) so the
/// CPU golden and GPU twin round identically. Degenerate `edge0 == edge1`
/// returns a hard step at the edge rather than dividing by zero.
#[must_use]
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    if span == 0.0 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / span).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Linear interpolation `a + (b - a) * t`, written out to match the GPU twin.
#[must_use]
pub fn grain_lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Luminance-dependent grain weight. Real film shows grain most in the shadows,
/// so the shadow-biased weight is `1 - smoothstep(0, 1, luma)` (weight `1` at
/// black, `0` at white). `response` in `[0, 1]` blends from a flat weight of
/// `1` (no luma dependence) toward that full falloff, so `response == 0` grains
/// uniformly and `response == 1` grains only the shadows.
#[must_use]
pub fn grain_luminance_weight(luma: f32, response: f32) -> f32 {
    let shadow = 1.0 - smoothstep(0.0, 1.0, luma);
    grain_lerp(1.0, shadow, response.clamp(0.0, 1.0))
}

/// Adds a scalar grain value to a radiance sample: `rgb + grain * intensity *
/// luma_weight` per channel (additive, so an `intensity` of `0` is the
/// identity). The same value is added to every channel for monochrome grain.
#[must_use]
pub fn apply_grain(rgb: [f32; 3], grain_value: f32, intensity: f32, luma_weight: f32) -> [f32; 3] {
    let g = grain_value * intensity * luma_weight;
    [rgb[0] + g, rgb[1] + g, rgb[2] + g]
}

/// Artist controls for the film-grain pass. Defaults to *disabled*
/// (`intensity == 0`), so [`FilmGrainParams::apply`] is the identity until grain
/// is explicitly turned on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilmGrainParams {
    /// Grain strength added to the radiance. `0` disables the effect.
    pub intensity: f32,
    /// Luminance response in `[0, 1]`: `0` grains uniformly, `1` biases the
    /// grain fully toward the shadows.
    pub response: f32,
    /// Grain cell size; larger values sample a lower frequency (coarser grain).
    /// Scales the sampling resolution by `1 / size`.
    pub size: f32,
    /// When `true`, each channel is hashed with an offset seed for coloured
    /// grain; when `false`, one value is shared across channels (monochrome).
    pub colored: bool,
}

impl Default for FilmGrainParams {
    fn default() -> Self {
        Self {
            intensity: 0.0,
            response: 0.8,
            size: 1.0,
            colored: false,
        }
    }
}

impl FilmGrainParams {
    /// Effective sampling resolution after the grain `size` scale
    /// (`resolution / size`, guarding a non-positive size).
    #[must_use]
    fn scaled_resolution(&self, resolution: [f32; 2]) -> [f32; 2] {
        let inv_size = if self.size > 0.0 { 1.0 / self.size } else { 1.0 };
        [resolution[0] * inv_size, resolution[1] * inv_size]
    }

    /// Applies film grain to a radiance sample at screen coordinate `uv`,
    /// animated by `time_seed`. Uses the `Rec. 709` luma for the response
    /// weight; a coloured grain hashes each channel with an offset seed. With
    /// the default (`intensity == 0`) this returns `rgb` unchanged.
    #[must_use]
    pub fn apply(
        &self,
        rgb: [f32; 3],
        uv: [f32; 2],
        resolution: [f32; 2],
        time_seed: f32,
    ) -> [f32; 3] {
        let luma = luminance(rgb);
        let weight = grain_luminance_weight(luma, self.response);
        let res = self.scaled_resolution(resolution);
        if self.colored {
            let gr = grain(uv, res, time_seed);
            let gg = grain(uv, res, time_seed + 1.0);
            let gb = grain(uv, res, time_seed + 2.0);
            [
                rgb[0] + gr * self.intensity * weight,
                rgb[1] + gg * self.intensity * weight,
                rgb[2] + gb * self.intensity * weight,
            ]
        } else {
            let g = grain(uv, res, time_seed);
            apply_grain(rgb, g, self.intensity, weight)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1.0e-4, "{a} !~= {b}");
    }

    #[test]
    fn grain_fract_matches_floor_definition() {
        // Positive fract like the truncating one.
        approx(grain_fract(0.25), 0.25);
        approx(grain_fract(3.75), 0.75);
        // Negative fract floors (unlike Rust's truncating f32::fract).
        approx(grain_fract(-0.25), 0.75);
        approx(grain_fract(-3.1), 0.9);
    }

    #[test]
    fn hash12_is_in_unit_range() {
        for i in 0..64 {
            let f = i as f32;
            let h = hash12([f * 1.7, f * 0.3 - 5.0]);
            assert!((0.0..1.0).contains(&h), "hash12 out of range: {h}");
        }
    }

    #[test]
    fn hash12_is_deterministic() {
        approx(hash12([12.0, 34.0]), hash12([12.0, 34.0]));
    }

    #[test]
    fn hash12_varies_with_input() {
        // Neighbouring coordinates should hash to different values.
        let a = hash12([10.0, 10.0]);
        let b = hash12([11.0, 10.0]);
        let c = hash12([10.0, 11.0]);
        assert!((a - b).abs() > 1.0e-4, "hash did not vary in x");
        assert!((a - c).abs() > 1.0e-4, "hash did not vary in y");
    }

    #[test]
    fn grain_is_in_signed_unit_range() {
        for i in 0..64 {
            let f = i as f32;
            let g = grain([f * 0.01, f * 0.02], [1920.0, 1080.0], f * 0.5);
            assert!((-1.0..1.0).contains(&g), "grain out of range: {g}");
        }
    }

    #[test]
    fn grain_is_deterministic() {
        let a = grain([0.3, 0.7], [1920.0, 1080.0], 2.0);
        let b = grain([0.3, 0.7], [1920.0, 1080.0], 2.0);
        approx(a, b);
    }

    #[test]
    fn grain_animates_with_time_seed() {
        let a = grain([0.3, 0.7], [1920.0, 1080.0], 1.0);
        let b = grain([0.3, 0.7], [1920.0, 1080.0], 2.0);
        assert!((a - b).abs() > 1.0e-5, "grain did not animate with time");
    }

    #[test]
    fn luminance_rec709() {
        approx(luminance([1.0, 1.0, 1.0]), 1.0);
        approx(luminance([1.0, 0.0, 0.0]), 0.2126);
        approx(luminance([0.0, 1.0, 0.0]), 0.7152);
        approx(luminance([0.0, 0.0, 1.0]), 0.0722);
    }

    #[test]
    fn smoothstep_endpoints_and_midpoint() {
        approx(smoothstep(0.0, 1.0, -1.0), 0.0);
        approx(smoothstep(0.0, 1.0, 0.0), 0.0);
        approx(smoothstep(0.0, 1.0, 1.0), 1.0);
        approx(smoothstep(0.0, 1.0, 2.0), 1.0);
        approx(smoothstep(0.0, 1.0, 0.5), 0.5);
    }

    #[test]
    fn smoothstep_is_monotonic() {
        let mut prev = -1.0_f32;
        for i in 0..=32 {
            let x = i as f32 / 32.0;
            let s = smoothstep(0.0, 1.0, x);
            assert!(s >= prev, "smoothstep not monotonic at {x}");
            prev = s;
        }
    }

    #[test]
    fn smoothstep_degenerate_edges_are_a_hard_step() {
        approx(smoothstep(0.5, 0.5, 0.4), 0.0);
        approx(smoothstep(0.5, 0.5, 0.6), 1.0);
    }

    #[test]
    fn grain_lerp_basic() {
        approx(grain_lerp(0.0, 10.0, 0.0), 0.0);
        approx(grain_lerp(0.0, 10.0, 1.0), 10.0);
        approx(grain_lerp(0.0, 10.0, 0.5), 5.0);
    }

    #[test]
    fn luminance_weight_response_zero_is_uniform() {
        // No luma dependence: weight is 1 everywhere.
        approx(grain_luminance_weight(0.0, 0.0), 1.0);
        approx(grain_luminance_weight(0.5, 0.0), 1.0);
        approx(grain_luminance_weight(1.0, 0.0), 1.0);
    }

    #[test]
    fn luminance_weight_response_one_biases_shadows() {
        // Full response: 1 at black, ~0 at white, 0.5 at mid.
        approx(grain_luminance_weight(0.0, 1.0), 1.0);
        approx(grain_luminance_weight(1.0, 1.0), 0.0);
        approx(grain_luminance_weight(0.5, 1.0), 0.5);
    }

    #[test]
    fn luminance_weight_is_monotonic_decreasing() {
        let mut prev = f32::INFINITY;
        for i in 0..=32 {
            let luma = i as f32 / 32.0;
            let w = grain_luminance_weight(luma, 1.0);
            assert!(w <= prev, "weight not decreasing at luma {luma}");
            prev = w;
        }
    }

    #[test]
    fn apply_grain_intensity_zero_is_identity() {
        let rgb = [0.3, 0.5, 0.7];
        approx3(apply_grain(rgb, 0.9, 0.0, 1.0), rgb);
    }

    fn approx3(a: [f32; 3], b: [f32; 3]) {
        approx(a[0], b[0]);
        approx(a[1], b[1]);
        approx(a[2], b[2]);
    }

    #[test]
    fn apply_grain_is_additive() {
        let rgb = [0.3, 0.5, 0.7];
        // g = 0.5 * 0.2 * 1.0 = 0.1 added to each channel.
        approx3(apply_grain(rgb, 0.5, 0.2, 1.0), [0.4, 0.6, 0.8]);
    }

    #[test]
    fn params_default_is_disabled_identity() {
        let p = FilmGrainParams::default();
        approx(p.intensity, 0.0);
        let rgb = [0.2, 0.4, 0.6];
        approx3(p.apply(rgb, [0.3, 0.7], [1920.0, 1080.0], 3.0), rgb);
    }

    #[test]
    fn params_enabled_changes_output() {
        let p = FilmGrainParams {
            intensity: 0.1,
            ..FilmGrainParams::default()
        };
        let rgb = [0.2, 0.2, 0.2];
        let out = p.apply(rgb, [0.31, 0.73], [1920.0, 1080.0], 3.0);
        assert!(
            (out[0] - rgb[0]).abs() > 1.0e-6,
            "enabled grain should change the sample, got {out:?}"
        );
    }

    #[test]
    fn params_colored_grain_differs_per_channel() {
        let p = FilmGrainParams {
            intensity: 0.2,
            response: 0.0, // uniform weight so only the per-channel hash differs
            colored: true,
            ..FilmGrainParams::default()
        };
        // Start from equal channels; coloured grain should split them.
        let rgb = [0.5, 0.5, 0.5];
        let out = p.apply(rgb, [0.42, 0.18], [1920.0, 1080.0], 7.0);
        assert!(
            (out[0] - out[1]).abs() > 1.0e-6 || (out[1] - out[2]).abs() > 1.0e-6,
            "coloured grain should differ per channel, got {out:?}"
        );
    }

    #[test]
    fn params_size_zero_is_safe() {
        let p = FilmGrainParams {
            intensity: 0.1,
            size: 0.0,
            ..FilmGrainParams::default()
        };
        // Must not divide by zero / produce NaN.
        let out = p.apply([0.3, 0.3, 0.3], [0.3, 0.7], [1920.0, 1080.0], 1.0);
        assert!(out.iter().all(|c| c.is_finite()), "size 0 produced NaN: {out:?}");
    }
}
