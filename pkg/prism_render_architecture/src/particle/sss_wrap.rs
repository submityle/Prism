//! `subsurface`-scattering (`SSS`) `wrap` lighting for translucent particles
//! (design section 16-21).
//!
//! Thin translucent media — smoke lit from behind, wax, skin-like sprites, wax
//! candles, foliage cards — do not obey a hard Lambert terminator: light
//! *wraps* around the shaded point and re-emerges on the shadowed side, tinted
//! by the wavelength-dependent scattering distance (red penetrates deepest, so
//! the shadow terminator glows warm). Production `GPU` engines approximate this
//! with a pre-integrated `subsurface` profile or a diffusion `dipole`; this
//! module owns a device-free, `CPU`-verifiable reference built from only
//! `f32::sqrt`, `f32::floor`, rational polynomials, and integer exponentiation,
//! so a future `GPU` kernel can match it bit for bit.
//!
//! The building blocks are: (1) a scalar `wrap` diffuse that bends the Lambert
//! terminator past zero ([`wrap_ndotl`]); (2) a per-channel `RGB` scatter
//! `染色` term that tints the wrapped-in shadow light using a wider effective
//! `wrap` per channel ([`scatter_color`]); (3) a thickness-driven transmission
//! term that glows where the medium is thin and the viewer looks toward the
//! light through it ([`thickness_transmission`]); and (4) [`SssParams`] gathers
//! the tunables and drives [`SssParams::evaluate`]. `GPU` packing follows the
//! shared `std430` `vec4` alignment from [`super::gpu_layout`].
//!
//! This is deliberately independent of [`super::shading`] (which carries no
//! `subsurface` model): nothing here imports or re-derives that router. No
//! transcendental function is used — no `exp`-based `dipole`, no trigonometry —
//! so every result is deterministic and platform independent.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Generic denominator / soft-edge guard below which a division collapses to a
/// hard step, so no divide-by-zero can produce a `NaN`.
const MIN_EDGE: f32 = 1e-6;

/// Byte stride of one [`SssParams`] record in a `std430` storage buffer.
///
/// The seven scalars pack into two `vec4` slots: `vec4(wrap, scatter_r,
/// scatter_g, scatter_b)` followed by `vec4(thickness_scale,
/// transmission_power, ambient, pad)`.
pub const SSS_PARAMS_STRIDE: usize = 2 * VEC4_STRIDE;

/// Clamps a scalar into the `0..=1` range without branching on equality.
#[must_use]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Hermite `smoothstep` from `edge0` to `edge1` evaluated at `x`.
///
/// Returns `0.0` at or below `edge0`, `1.0` at or above `edge1`, and the
/// `t * t * (3 - 2 * t)` interpolation in between. A degenerate (near-equal)
/// interval collapses to a hard step at `edge1` rather than dividing by zero.
#[must_use]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    if span < MIN_EDGE {
        return if x < edge1 { 0.0 } else { 1.0 };
    }
    let t = clamp01((x - edge0) / span);
    t * t * (3.0 - 2.0 * t)
}

/// Dot product of two hand-rolled `vec3` values.
#[must_use]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Normalizes a `vec3`, returning the zero vector for a (near-)zero input so no
/// `NaN` can leak from dividing by a vanishing length.
#[must_use]
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len_sq = dot3(v, v);
    if len_sq < MIN_EDGE {
        return [0.0, 0.0, 0.0];
    }
    let inv_len = 1.0 / len_sq.sqrt();
    [v[0] * inv_len, v[1] * inv_len, v[2] * inv_len]
}

/// Raises `base` to the integer power `exp` by repeated multiplication.
///
/// Avoids `f32::powf`: `exp == 0` yields `1.0` by the usual `x^0` convention.
/// For a fixed `base` in `0..=1` the result is non-increasing in `exp`.
#[must_use]
fn ipow(base: f32, exp: u32) -> f32 {
    let mut acc = 1.0;
    for _ in 0..exp {
        acc *= base;
    }
    acc
}

/// `wrap`-diffuse `NdotL`: bends the Lambert terminator past zero so light
/// wraps onto the shadowed side.
///
/// Computes `clamp((n_dot_l + wrap) / (1 + wrap), 0, 1)`. With `wrap == 0` this
/// is exactly the clamped Lambert term `clamp(n_dot_l, 0, 1)`; a positive
/// `wrap` lifts the whole curve so a back-facing point (`n_dot_l < 0`) still
/// receives light. The denominator is guarded so a degenerate `wrap <= -1`
/// cannot divide by zero.
#[must_use]
pub fn wrap_ndotl(n_dot_l: f32, wrap: f32) -> f32 {
    let denom = (1.0 + wrap).max(MIN_EDGE);
    clamp01((n_dot_l + wrap) / denom)
}

/// Per-channel `RGB` scatter `染色` for the wrapped-in shadow light.
///
/// Each channel uses a wider effective `wrap` (`wrap + scatter_rgb[c]`), so a
/// larger per-channel scatter radius bends more light onto the shadowed side —
/// with a red-dominant `scatter_rgb` the terminator glows warm (red > green >
/// blue) exactly where the un-wrapped Lambert term has gone dark. The returned
/// triple is the *extra* colored light beyond the hard terminator: on the fully
/// lit side (`n_dot_l >= 0` large) the channel wrap matches the core term and
/// the contribution fades to zero, while in the backlit region it is the
/// tinted, per-channel wrapped diffuse. Each channel is tinted by
/// `scatter_rgb[c]`, reinforcing the wavelength-dependent penetration.
#[must_use]
pub fn scatter_color(n_dot_l: f32, wrap: f32, scatter_rgb: [f32; 3]) -> [f32; 3] {
    let core = clamp01(n_dot_l);
    core::array::from_fn(|c| {
        let radius = scatter_rgb[c].max(0.0);
        let wrapped_c = wrap_ndotl(n_dot_l, wrap + radius);
        let extra = (wrapped_c - core).max(0.0);
        radius * extra
    })
}

/// Thickness-driven back transmission for a thin translucent medium.
///
/// Thin regions glow brightest, so the depth term is the rational attenuation
/// `1 / (1 + thickness_scale * thickness)` (an `exp`-free stand-in for
/// Beer-Lambert), which decreases monotonically as `thickness` grows. The
/// viewer only sees the transmitted glow when looking *toward* the light
/// through the medium, so the view lobe is the back-facing alignment
/// `clamp(-dot(view, light), 0, 1)` shaped by a `smoothstep` soft edge and
/// sharpened by the integer `transmission_power` lobe. Both direction inputs
/// are normalized internally; `thickness` and `thickness_scale` are floored at
/// zero so the term stays in `0..=1`.
#[must_use]
pub fn thickness_transmission(
    thickness: f32,
    light_dir: [f32; 3],
    view_dir: [f32; 3],
    params: SssParams,
) -> f32 {
    let l = normalize3(light_dir);
    let v = normalize3(view_dir);
    // Looking toward the light through the medium: view opposes the light dir.
    let back = clamp01(-dot3(v, l));
    let lobe = ipow(smoothstep(0.0, 1.0, back), params.transmission_power);
    let t = thickness.max(0.0);
    let k = params.thickness_scale.max(0.0);
    let attenuation = 1.0 / (1.0 + k * t);
    attenuation * lobe
}

/// Tunables for the `subsurface` `wrap` lighting model (design section 16-21).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SssParams {
    /// How far light wraps past the Lambert terminator (`0` = hard Lambert).
    pub wrap: f32,
    /// Per-channel `RGB` scatter radii; a red-dominant triple warms the
    /// terminator as red light penetrates deepest.
    pub scatter_color: [f32; 3],
    /// Rational thickness-attenuation rate for the transmission term.
    pub thickness_scale: f32,
    /// Integer exponent sharpening the back-transmission view lobe.
    pub transmission_power: u32,
    /// Uniform ambient floor added to every diffuse channel.
    pub ambient: f32,
}

/// One evaluated `subsurface` shading sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SssSample {
    /// Per-channel `RGB` wrapped diffuse plus scatter `染色` and ambient.
    pub diffuse: [f32; 3],
    /// Scalar back-transmission glow in `0..=1`.
    pub transmission: f32,
}

impl SssParams {
    /// Creates `subsurface` `wrap` parameters from all fields.
    #[must_use]
    pub const fn new(
        wrap: f32,
        scatter_color: [f32; 3],
        thickness_scale: f32,
        transmission_power: u32,
        ambient: f32,
    ) -> Self {
        Self {
            wrap,
            scatter_color,
            thickness_scale,
            transmission_power,
            ambient,
        }
    }

    /// Evaluates the `subsurface` `wrap` diffuse and back transmission for one
    /// shaded point.
    ///
    /// The `normal`, `light_dir`, and `view_dir` are normalized internally.
    /// The diffuse is the scalar [`wrap_ndotl`] term broadened per channel by
    /// [`scatter_color`] and lifted by [`ambient`](Self::ambient); each channel
    /// is floored at zero. The transmission comes from
    /// [`thickness_transmission`] over the given `thickness`.
    #[must_use]
    pub fn evaluate(
        &self,
        normal: [f32; 3],
        light_dir: [f32; 3],
        view_dir: [f32; 3],
        thickness: f32,
    ) -> SssSample {
        let n = normalize3(normal);
        let l = normalize3(light_dir);
        let n_dot_l = dot3(n, l);
        let wrapped = wrap_ndotl(n_dot_l, self.wrap);
        let scatter = scatter_color(n_dot_l, self.wrap, self.scatter_color);
        let diffuse = core::array::from_fn(|c| (wrapped + scatter[c] + self.ambient).max(0.0));
        let transmission = thickness_transmission(thickness, light_dir, view_dir, *self);
        SssSample {
            diffuse,
            transmission,
        }
    }

    /// Packs the parameters into their `std430` `vec4`-aligned word layout.
    ///
    /// Layout: `[wrap, scatter_r, scatter_g, scatter_b, thickness_scale,
    /// transmission_power, ambient, pad]` as raw `u32` words (the six `f32`
    /// fields via `f32::to_bits`) — two `vec4` slots, matching
    /// [`SSS_PARAMS_STRIDE`]. The trailing word is padding.
    #[must_use]
    pub fn to_std430(&self) -> [u32; 8] {
        [
            self.wrap.to_bits(),
            self.scatter_color[0].to_bits(),
            self.scatter_color[1].to_bits(),
            self.scatter_color[2].to_bits(),
            self.thickness_scale.to_bits(),
            self.transmission_power,
            self.ambient.to_bits(),
            0,
        ]
    }
}

/// Packs a slice of [`SssParams`] into a flat `std430` `u32` word buffer.
///
/// Each record contributes the eight words of [`SssParams::to_std430`] in
/// order, so the result length is `8 * params.len()`.
#[must_use]
pub fn pack_params(params: &[SssParams]) -> Vec<u32> {
    let mut out = Vec::with_capacity(params.len() * 8);
    for p in params {
        out.extend_from_slice(&p.to_std430());
    }
    out
}

/// Total byte size of a `std430` storage buffer holding `count` packed
/// [`SssParams`] records.
///
/// Uses [`SSS_PARAMS_STRIDE`] and the shared clamp-to-one-element rule from
/// [`storage_bytes`], so an empty set still yields a valid `GPU` binding.
#[must_use]
pub fn sss_params_buffer_bytes(count: usize) -> usize {
    storage_bytes(SSS_PARAMS_STRIDE, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for the float assertions in this module's tests.
    const CMP_EPS: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    #[test]
    fn wrap_zero_is_clamped_lambert() {
        // wrap == 0 reproduces clamp(n_dot_l, 0, 1) exactly.
        assert!(approx_eq(wrap_ndotl(0.7, 0.0), 0.7));
        assert!(approx_eq(wrap_ndotl(-0.3, 0.0), 0.0));
        assert!(approx_eq(wrap_ndotl(1.0, 0.0), 1.0));
    }

    #[test]
    fn wrap_lights_the_backfacing_side() {
        // A back-facing point (n_dot_l < 0) is dark under hard Lambert...
        assert!(approx_eq(wrap_ndotl(-0.2, 0.0), 0.0));
        // ...but a positive wrap lifts it above zero.
        let lit = wrap_ndotl(-0.2, 0.5);
        assert!(lit > 0.0);
        assert!(lit <= 1.0);
        // Wrapping is monotone non-decreasing in `wrap` for a fixed angle.
        let mut prev = -1.0;
        let steps: [f32; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
        for w in steps {
            let v = wrap_ndotl(-0.2, w);
            assert!(v >= prev - CMP_EPS);
            prev = v;
        }
    }

    #[test]
    fn wrap_denominator_is_guarded() {
        // A degenerate wrap <= -1 must not divide by zero / produce NaN.
        let v = wrap_ndotl(0.5, -1.0);
        assert!(v.is_finite());
        assert!(v >= 0.0);
        assert!(v <= 1.0);
    }

    #[test]
    fn scatter_tints_the_shadow_side_warm() {
        // Backlit point with a red-dominant scatter radius: red > green > blue.
        let s = scatter_color(-0.2, 0.3, [0.8, 0.4, 0.2]);
        assert!(s[0] > s[1]);
        assert!(s[1] > s[2]);
        assert!(s[2] >= 0.0);
    }

    #[test]
    fn scatter_fades_on_the_fully_lit_side() {
        // Facing the light head-on, the wrapped channels match the core term,
        // so the extra scatter contribution is (near) zero.
        let s = scatter_color(1.0, 0.3, [0.8, 0.4, 0.2]);
        for c in s {
            assert!(approx_eq(c, 0.0));
        }
    }

    #[test]
    fn transmission_decreases_with_thickness() {
        // View directly opposite the light so the back lobe is fully on.
        let params = SssParams::new(0.5, [0.8, 0.4, 0.2], 2.0, 4, 0.05);
        let light = [0.0, 0.0, 1.0];
        let view = [0.0, 0.0, -1.0];
        let thin = thickness_transmission(0.1, light, view, params);
        let mid = thickness_transmission(1.0, light, view, params);
        let thick = thickness_transmission(4.0, light, view, params);
        assert!(thin > mid);
        assert!(mid > thick);
        assert!(thick >= 0.0);
        assert!(thin <= 1.0);
    }

    #[test]
    fn transmission_power_is_monotone() {
        // A partial back alignment (dot(view, light) == -0.5) keeps the lobe
        // strictly inside (0, 1), so raising the power lowers the transmission.
        let light = [0.0, 0.0, 1.0];
        let view = normalize3([0.0, 0.866_025_4, -0.5]);
        let mut prev = 2.0;
        for power in 0..6u32 {
            let params = SssParams::new(0.4, [0.6, 0.3, 0.15], 1.0, power, 0.0);
            let t = thickness_transmission(0.5, light, view, params);
            assert!(t <= prev + CMP_EPS);
            assert!(t >= 0.0);
            assert!(t <= 1.0);
            prev = t;
        }
    }

    #[test]
    fn transmission_needs_a_back_facing_view() {
        // Looking along the light dir (not through the medium): no glow.
        let params = SssParams::new(0.5, [0.8, 0.4, 0.2], 1.0, 3, 0.0);
        let light = [0.0, 0.0, 1.0];
        let view = [0.0, 0.0, 1.0];
        assert!(approx_eq(
            thickness_transmission(0.2, light, view, params),
            0.0
        ));
    }

    #[test]
    fn evaluate_wraps_diffuse_and_tints_the_shadow() {
        let params = SssParams::new(0.5, [0.8, 0.4, 0.2], 2.0, 4, 0.02);
        // Normal facing +z, light coming slightly from behind (-z bias).
        let normal = [0.0, 0.0, 1.0];
        let light = normalize3([0.0, 0.6, -0.2]);
        let view = [0.0, 0.0, -1.0];
        let sample = params.evaluate(normal, light, view, 0.5);
        // Warm terminator: red channel leads green leads blue.
        assert!(sample.diffuse[0] > sample.diffuse[1]);
        assert!(sample.diffuse[1] > sample.diffuse[2]);
        // Ambient floor keeps every channel above zero.
        for c in sample.diffuse {
            assert!(c >= params.ambient - CMP_EPS);
        }
        assert!(sample.transmission >= 0.0);
        assert!(sample.transmission <= 1.0);
    }

    #[test]
    fn evaluate_is_deterministic() {
        let params = SssParams::new(0.3, [0.7, 0.35, 0.2], 1.5, 3, 0.01);
        let normal = [0.1, 0.2, 0.97];
        let light = [0.3, -0.4, -0.5];
        let view = [0.0, 0.1, -1.0];
        let a = params.evaluate(normal, light, view, 0.8);
        let b = params.evaluate(normal, light, view, 0.8);
        assert_eq!(a, b);
    }

    #[test]
    fn std430_layout_round_trips() {
        let params = SssParams::new(0.5, [0.8, 0.4, 0.2], 2.0, 7, 0.05);
        let words = params.to_std430();
        assert!(approx_eq(f32::from_bits(words[0]), 0.5));
        assert!(approx_eq(f32::from_bits(words[1]), 0.8));
        assert!(approx_eq(f32::from_bits(words[2]), 0.4));
        assert!(approx_eq(f32::from_bits(words[3]), 0.2));
        assert!(approx_eq(f32::from_bits(words[4]), 2.0));
        assert_eq!(words[5], 7);
        assert!(approx_eq(f32::from_bits(words[6]), 0.05));
        assert_eq!(words[7], 0);
    }

    #[test]
    fn std430_stride_is_two_vec4_aligned() {
        assert_eq!(SSS_PARAMS_STRIDE, 32);
        assert_eq!(SSS_PARAMS_STRIDE % VEC4_STRIDE, 0);
    }

    #[test]
    fn buffer_bytes_follow_the_shared_rule() {
        // Empty set still reserves one element for a valid GPU binding.
        assert_eq!(sss_params_buffer_bytes(0), SSS_PARAMS_STRIDE);
        assert_eq!(sss_params_buffer_bytes(3), 3 * SSS_PARAMS_STRIDE);
    }

    #[test]
    fn pack_params_concatenates_records() {
        let a = SssParams::new(0.5, [0.8, 0.4, 0.2], 2.0, 4, 0.05);
        let b = SssParams::new(0.1, [0.3, 0.2, 0.1], 1.0, 2, 0.0);
        let packed = pack_params(&[a, b]);
        assert_eq!(packed.len(), 16);
        assert_eq!(&packed[0..8], &a.to_std430());
        assert_eq!(&packed[8..16], &b.to_std430());
        // An empty slice packs to nothing.
        assert!(pack_params(&[]).is_empty());
    }
}
