//! Split-sum environment `BRDF` (`DFG`) `CPU` gold standard for particle image
//! based lighting (design §17 "`PBR` 粒子与体积着色", §22 "光追碰撞与受光").
//!
//! Image based lighting evaluates the specular reflectance integral
//! `∫ brdf(l, v) * L(l) * (n·l) dl` over the environment. Karis's *split-sum*
//! factors that integral into two independently prefiltered pieces:
//!
//! 1. a **prefiltered environment map** `∫ L(l) dl` sampled from a roughness
//!    dependent `mip` chain (the "pre-filtered color"), and
//! 2. the **environment `BRDF`** `∫ brdf * (n·l) dl`, a view-and-roughness only
//!    term that is independent of the lighting and so can be baked once into a
//!    2D `(NoV, roughness)` table of two channels `(scale, bias)` such that the
//!    specular response reconstructs as `specular = F0 * scale + bias`.
//!
//! That second term is the *`DFG`* term (named for its `D`istribution,
//! `F`resnel and `G`eometry factors). This module owns the `CPU` reference for
//! it so a future `GPU` draw kernel matches it bit for bit. It is distinct from
//! the `Schlick` `Fresnel` rim ([`super::fresnel_rim`]) — that is a direct per
//! light term, whereas this is the *integrated* environment term — and from the
//! specular anti-aliasing roughness widening ([`super::specular_aa`]).
//!
//! Two ways to obtain `(scale, bias)` are provided, matching the two paths real
//! engines ship:
//!
//! - **Analytic approximation** ([`env_brdf_approx`]): the Lazarov / Karis
//!   "`EnvBRDFApprox`" polynomial fit (Dimitar Lazarov, *Physically Based
//!   Lighting in Call of Duty: Black Ops 2*, 2013; popularised by Brian Karis's
//!   mobile `PBR` notes). It needs no table and no texture fetch — ideal for the
//!   cheap particle path. The published fit uses one `exp2(-9.28 * NoV)` grazing
//!   term; because the workspace determinism lint forbids transcendentals, this
//!   module reproduces that term with a transcendental-free surrogate
//!   ([`exp2_grazing_falloff`]) that agrees to better than `3e-3` relative error
//!   over `NoV ∈ 0..=1`. This is **not** a fundamental trigonometry barrier: the
//!   base-two exponential is recovered from a short rational series raised to an
//!   integer power by squaring, so the result stays deterministic.
//! - **Baked lookup table** ([`DfgLut`]): a caller-supplied `N × N` grid of
//!   `(scale, bias)` samples (the caller bakes it, since the true hemispherical
//!   integral is a transcendental Monte-Carlo integration that deliberately
//!   lives outside this transcendental-free module). This module only clamps the
//!   `(NoV, roughness)` coordinate into range and does a bilinear fetch, plus
//!   `std430` packing for the `GPU`.
//!
//! A [`prefilter_mip_lod`] helper maps roughness onto the prefiltered map's
//! `mip` `LOD` for the first split-sum piece.
//!
//! Only `f32` `sqrt`, `floor`, `clamp`/`min`/`max` and rational / integer-power
//! arithmetic are used — no transcendental functions — so the result is
//! deterministic and portable. Vectors use the shared [`super::Vec3`] (an `RGB`
//! `F0` is carried as a `Vec3`); `GPU` packing follows the shared `std430`
//! alignment from [`super::gpu_layout`].

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, VEC2_STRIDE};
use crate::particle::Vec3;

/// Minimum table edge length ( `size` ) a [`DfgLut`] accepts.
///
/// Bilinear interpolation needs at least two taps per axis, so a `1 × 1` (or
/// empty) grid cannot be sampled and is rejected at construction.
const MIN_LUT_SIZE: usize = 2;

/// Clamps a scalar into the `0..=1` range without branching on equality.
#[must_use]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Squares `base` `k` times, i.e. raises it to the `2^k`-th power.
///
/// This recovers an integer power that is itself a power of two using only
/// multiplication, replacing any `powf`/`powi` call so the module stays
/// transcendental-free and bit-reproducible. `k = 0` returns `base` unchanged
/// (the `2^0 = 1`-th power).
#[must_use]
fn squared_k_times(base: f32, k: u32) -> f32 {
    let mut acc = base;
    let mut remaining = k;
    while remaining > 0 {
        acc *= acc;
        remaining -= 1;
    }
    acc
}

/// Evaluates `2^(-9.28 * NoV)` for `n_dot_v ∈ 0..=1` without a transcendental.
///
/// The published `EnvBRDFApprox` grazing term is `exp2(-9.28 * NoV)`. Since the
/// determinism lint forbids `exp2`, this reconstructs it as
/// `(2^(-9.28 * NoV / 32))^32`: the exponent is divided by `32`, so the small
/// argument `s = 9.28 * NoV / 32 ∈ 0..=0.29` is evaluated by the truncated
/// series of `2^(-s)` and then raised back to the 32nd power by five squarings
/// ([`squared_k_times`]). Over `NoV ∈ 0..=1` the surrogate matches the true
/// `exp2(-9.28 * NoV)` to within `~7e-6` absolute (`~3e-3` relative).
///
/// `n_dot_v` is clamped to `0..=1` first (grazing is `NoV = 0`, head-on is
/// `NoV = 1`).
#[must_use]
pub fn exp2_grazing_falloff(n_dot_v: f32) -> f32 {
    // Magic constant: the Lazarov / Karis fit's grazing rate `9.28`.
    let exponent_magnitude = 9.28 * clamp01(n_dot_v);
    // 32 = 2^5: divide the exponent by 32, then undo it with five squarings.
    let s = exponent_magnitude * (1.0 / 32.0);
    // Truncated Maclaurin series of `2^(-s)` = `exp(-s * ln 2)`. The magic
    // coefficients are the `ln(2)` powers: `ln 2`, `ln(2)^2 / 2`,
    // `ln(2)^3 / 6`. On `s ∈ 0..=0.29` the cubic truncation error is `< 5e-5`.
    const LN2: f32 = core::f32::consts::LN_2;
    const LN2_SQ_HALF: f32 = LN2 * LN2 / 2.0; // ln(2)^2 / 2
    const LN2_CUBE_SIXTH: f32 = LN2 * LN2 * LN2 / 6.0; // ln(2)^3 / 6
    let base = 1.0 - LN2 * s + LN2_SQ_HALF * s * s - LN2_CUBE_SIXTH * s * s * s;
    squared_k_times(base, 5)
}

/// The integrated environment `BRDF` term for one `(NoV, roughness)` sample.
///
/// The specular image based lighting response reconstructs from these two
/// channels and the surface reflectance at normal incidence `F0` as
/// `specular = F0 * scale + bias`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnvBrdfTerms {
    /// The multiplicative `scale` applied to `F0` (the `A` of the `DFG` fit).
    pub scale: f32,
    /// The additive `bias` added after scaling `F0` (the `B` of the `DFG` fit).
    pub bias: f32,
}

impl EnvBrdfTerms {
    /// Builds a term pair from its two channels.
    #[must_use]
    pub const fn new(scale: f32, bias: f32) -> Self {
        Self { scale, bias }
    }

    /// Reconstructs the scalar specular response `F0 * scale + bias`.
    ///
    /// `f0` is the grayscale reflectance at normal incidence (`~0.04` for common
    /// dielectrics, higher for metals).
    #[must_use]
    pub fn apply(self, f0: f32) -> f32 {
        f0 * self.scale + self.bias
    }

    /// Reconstructs the `RGB` specular response `F0 * scale + bias` per channel.
    ///
    /// `f0` is the per-channel reflectance at normal incidence carried as a
    /// [`Vec3`] (metals use a tinted `F0`; dielectrics a gray one). The `scale`
    /// and `bias` are shared across channels, so this is `F0 * scale` plus a
    /// uniform `bias` added to every channel.
    #[must_use]
    pub fn apply_rgb(self, f0: Vec3) -> Vec3 {
        f0.scale(self.scale).add(Vec3::splat(self.bias))
    }
}

/// The Lazarov / Karis analytic environment `BRDF` approximation.
///
/// Returns the `(scale, bias)` such that `specular = F0 * scale + bias`, from
/// the view term `n_dot_v` (the cosine between the surface normal and the view
/// direction) and the perceptual `roughness`, both clamped to `0..=1`.
///
/// The fit is purely polynomial in `roughness` and uses the transcendental-free
/// grazing term [`exp2_grazing_falloff`]:
///
/// ```text
/// c0 = (-1,  -0.0275, -0.572,  0.022)
/// c1 = ( 1,   0.0425,  1.04,  -0.04)
/// r  = roughness * c0 + c1
/// a  = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y
/// scale = -1.04 * a + r.z
/// bias  =  1.04 * a + r.w
/// ```
///
/// At `roughness = 0`, `NoV = 1` (smooth mirror, head-on) the result is
/// `scale ≈ 1`, `bias ≈ 0`, so the specular response is `≈ F0`. Toward grazing
/// (`NoV → 0`) the `bias` rises sharply, brightening silhouettes regardless of
/// `F0` (the integrated `Fresnel` edge).
#[must_use]
pub fn env_brdf_approx(n_dot_v: f32, roughness: f32) -> EnvBrdfTerms {
    let n = clamp01(n_dot_v);
    let rough = clamp01(roughness);
    // Lazarov / Karis fit coefficients `c0` and `c1`; the four lanes feed the
    // scalar polynomial below, named `rx`..`rw` after the original `vec4` lanes.
    let rx = 1.0 - rough;
    let ry = rough * -0.0275 + 0.0425;
    let rz = rough * -0.572 + 1.04;
    let rw = rough * 0.022 + -0.04;
    // The grazing term `min(r.x^2, exp2(-9.28 * NoV))`; `r.x^2` is a plain
    // square (no `powf`).
    let grazing = (rx * rx).min(exp2_grazing_falloff(n));
    let a = grazing * rx + ry;
    // Magic constants `-1.04` / `1.04`: the fit's channel mixing weights.
    EnvBrdfTerms::new(-1.04 * a + rz, 1.04 * a + rw)
}

/// The scalar specular image based lighting response via [`env_brdf_approx`].
///
/// Convenience wrapper for `env_brdf_approx(n_dot_v, roughness).apply(f0)` with
/// a grayscale `F0`.
#[must_use]
pub fn env_brdf_specular(n_dot_v: f32, roughness: f32, f0: f32) -> f32 {
    env_brdf_approx(n_dot_v, roughness).apply(f0)
}

/// The `RGB` specular image based lighting response via [`env_brdf_approx`].
///
/// Convenience wrapper for `env_brdf_approx(n_dot_v, roughness).apply_rgb(f0)`
/// with a per-channel `F0` carried as a [`Vec3`]. For a grayscale `F0` (equal
/// channels) every channel equals the scalar [`env_brdf_specular`].
#[must_use]
pub fn env_brdf_specular_rgb(n_dot_v: f32, roughness: f32, f0: Vec3) -> Vec3 {
    env_brdf_approx(n_dot_v, roughness).apply_rgb(f0)
}

/// A baked 2D `DFG` lookup table of `(scale, bias)` samples.
///
/// The grid is `size × size`, row-major, with the fast (column) axis indexing
/// `NoV ∈ 0..=1` and the slow (row) axis indexing `roughness ∈ 0..=1`, both on
/// an inclusive `0..=1` lattice (sample `i` sits at `i / (size - 1)`). The
/// caller bakes the values — the true environment `BRDF` is a hemispherical
/// Monte-Carlo integral that needs transcendentals and therefore lives outside
/// this module — and this type only clamps the query coordinate and does a
/// bilinear fetch, plus `std430` packing.
#[derive(Clone, Debug, PartialEq)]
pub struct DfgLut {
    /// Edge length of the square grid (at least [`MIN_LUT_SIZE`]).
    size: usize,
    /// `size * size` samples in row-major `(roughness, NoV)` order.
    data: Vec<EnvBrdfTerms>,
}

impl DfgLut {
    /// Builds a table from a `size × size` row-major sample grid.
    ///
    /// Returns `None` when `size` is below [`MIN_LUT_SIZE`] (bilinear needs two
    /// taps per axis) or when `data.len()` does not equal `size * size`, so a
    /// malformed table can never be sampled.
    #[must_use]
    pub fn new(size: usize, data: Vec<EnvBrdfTerms>) -> Option<Self> {
        if size < MIN_LUT_SIZE {
            return None;
        }
        if data.len() != size * size {
            return None;
        }
        Some(Self { size, data })
    }

    /// The edge length of the square grid.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Fetches the raw sample at integer grid coordinates `(ix, iy)`.
    ///
    /// `ix` indexes `NoV` and `iy` indexes `roughness`; both are clamped to
    /// `0..=size-1`, so out-of-range indices saturate to the edge rather than
    /// panicking.
    #[must_use]
    pub fn texel(&self, ix: usize, iy: usize) -> EnvBrdfTerms {
        let last = self.size - 1;
        let cx = ix.min(last);
        let cy = iy.min(last);
        self.data[cy * self.size + cx]
    }

    /// Bilinearly samples the table at `(n_dot_v, roughness)`.
    ///
    /// Both coordinates are clamped to `0..=1` (so out-of-range queries saturate
    /// to the table edge), scaled onto the `0..=size-1` lattice, and bilinearly
    /// interpolated from the four surrounding texels. A query that lands exactly
    /// on a lattice point returns that texel's stored value.
    #[must_use]
    pub fn sample(&self, n_dot_v: f32, roughness: f32) -> EnvBrdfTerms {
        let last = self.size - 1;
        // `last` is a small grid index; the cast to `f32` is exact for any
        // realistic table size.
        let span = last as f32;
        let fx = clamp01(n_dot_v) * span;
        let fy = clamp01(roughness) * span;
        let fx0 = fx.floor();
        let fy0 = fy.floor();
        let tx = fx - fx0;
        let ty = fy - fy0;
        // `floor` of a non-negative, in-range coordinate is a valid index; the
        // cast back is exact. `+ 1` saturates at the top edge.
        let ix0 = (fx0 as usize).min(last);
        let iy0 = (fy0 as usize).min(last);
        let ix1 = (ix0 + 1).min(last);
        let iy1 = (iy0 + 1).min(last);

        let s00 = self.texel(ix0, iy0);
        let s10 = self.texel(ix1, iy0);
        let s01 = self.texel(ix0, iy1);
        let s11 = self.texel(ix1, iy1);

        let scale = bilerp(s00.scale, s10.scale, s01.scale, s11.scale, tx, ty);
        let bias = bilerp(s00.bias, s10.bias, s01.bias, s11.bias, tx, ty);
        EnvBrdfTerms::new(scale, bias)
    }

    /// Packs the table into its `std430` scalar layout, row-major.
    ///
    /// Each texel is a `vec2<f32>` `(scale, bias)` emitted as its `f32::to_bits`
    /// patterns, matching a `vec2` stride ([`VEC2_STRIDE`]), so the block round
    /// trips exactly without a lossy cast. The returned length is
    /// `2 * size * size`.
    #[must_use]
    pub fn to_std430(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.data.len() * 2);
        for texel in &self.data {
            out.push(texel.scale.to_bits());
            out.push(texel.bias.to_bits());
        }
        out
    }

    /// Total `std430` byte size of this table's packed buffer.
    ///
    /// Uses [`VEC2_STRIDE`] and the shared clamp-to-one-element rule from
    /// [`storage_bytes`], so an empty query still yields a valid `GPU` binding.
    #[must_use]
    pub fn buffer_bytes(&self) -> usize {
        storage_bytes(VEC2_STRIDE, self.data.len())
    }
}

/// Bilinear interpolation of four corner values on a unit cell.
///
/// `v00`/`v10`/`v01`/`v11` are the `(x0, y0)`, `(x1, y0)`, `(x0, y1)`,
/// `(x1, y1)` corners; `tx` and `ty` are the in-cell fractions in `0..=1`. With
/// `tx = ty = 0` the result is exactly `v00`.
#[must_use]
fn bilerp(v00: f32, v10: f32, v01: f32, v11: f32, tx: f32, ty: f32) -> f32 {
    let bottom = v00 + (v10 - v00) * tx;
    let top = v01 + (v11 - v01) * tx;
    bottom + (top - bottom) * ty
}

/// Maps a perceptual `roughness` to the prefiltered environment map's `mip`
/// `LOD` for the split-sum color lookup.
///
/// A linear mapping `roughness * (mip_count - 1)`: `roughness = 0` (mirror)
/// samples the sharp `mip 0`, `roughness = 1` (fully rough) samples the coarsest
/// `mip`. `roughness` is clamped to `0..=1`. A zero- or one-level chain has no
/// roughness range and always returns `LOD 0`.
#[must_use]
pub fn prefilter_mip_lod(roughness: f32, mip_count: usize) -> f32 {
    if mip_count <= 1 {
        return 0.0;
    }
    // `mip_count - 1` is a small level count; the cast to `f32` is exact.
    let max_lod = (mip_count - 1) as f32;
    clamp01(roughness) * max_lod
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for the float assertions in this module's tests.
    const CMP_EPS: f32 = 1e-6;

    /// Looser tolerance for comparing the analytic fit against hand-computed
    /// reference values (the fit itself is an approximation).
    const FIT_EPS: f32 = 2e-3;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn squared_k_times_is_power_of_two_exponent() {
        assert!(approx(squared_k_times(2.0, 0), 2.0, CMP_EPS));
        assert!(approx(squared_k_times(2.0, 1), 4.0, CMP_EPS));
        assert!(approx(squared_k_times(2.0, 3), 256.0, CMP_EPS));
        // 0.5 squared five times is 0.5^32.
        let manual = {
            let mut v = 0.5_f32;
            for _ in 0..5 {
                v *= v;
            }
            v
        };
        assert!(approx(squared_k_times(0.5, 5), manual, CMP_EPS));
    }

    #[test]
    fn exp2_surrogate_matches_true_exp2_endpoints() {
        // `NoV = 0` -> exp2(0) = 1.
        assert!(approx(exp2_grazing_falloff(0.0), 1.0, FIT_EPS));
        // `NoV = 1` -> exp2(-9.28) ~= 0.0016.
        let reference = {
            // 2^(-9.28) via repeated halving of the integer part and a small
            // residual, independent of the module's own surrogate.
            let whole = 1.0_f32 / 512.0; // 2^-9
            whole * 0.823_50 // 2^-0.28 ~= 0.82350
        };
        assert!(approx(exp2_grazing_falloff(1.0), reference, FIT_EPS));
    }

    #[test]
    fn exp2_surrogate_is_monotone_decreasing() {
        let a = exp2_grazing_falloff(0.1);
        let b = exp2_grazing_falloff(0.5);
        let c = exp2_grazing_falloff(0.9);
        assert!(b < a);
        assert!(c < b);
        // Out-of-range inputs clamp to the endpoints.
        assert!(approx(exp2_grazing_falloff(-1.0), exp2_grazing_falloff(0.0), CMP_EPS));
        assert!(approx(exp2_grazing_falloff(2.0), exp2_grazing_falloff(1.0), CMP_EPS));
    }

    #[test]
    fn approx_smooth_mirror_is_scale_one_bias_zero() {
        // roughness = 0, head-on: specular ~= F0, i.e. scale ~= 1, bias ~= 0.
        let terms = env_brdf_approx(1.0, 0.0);
        assert!(approx(terms.scale, 1.0, 1e-2));
        assert!(approx(terms.bias, 0.0, 1e-2));
        // The reconstructed specular for a dielectric F0 is ~= F0.
        let f0 = 0.04;
        assert!(approx(terms.apply(f0), f0, 1e-2));
    }

    #[test]
    fn approx_matches_reference_fit_values() {
        // Hand-computed from the published fit with the exp2 surrogate.
        let t = env_brdf_approx(1.0, 0.0);
        assert!(approx(t.scale, 0.994_131, FIT_EPS));
        assert!(approx(t.bias, 0.005_869, FIT_EPS));

        let t = env_brdf_approx(0.0, 0.0);
        assert!(approx(t.scale, -0.044, FIT_EPS));
        assert!(approx(t.bias, 1.0442, FIT_EPS));

        let t = env_brdf_approx(1.0, 0.5);
        assert!(approx(t.scale, 0.723_266, FIT_EPS));
        assert!(approx(t.bias, 0.001_734, FIT_EPS));
    }

    #[test]
    fn approx_bias_rises_toward_grazing() {
        // The integrated Fresnel edge: bias grows as NoV shrinks (fixed rough).
        let head_on = env_brdf_approx(1.0, 0.3).bias;
        let mid = env_brdf_approx(0.5, 0.3).bias;
        let grazing = env_brdf_approx(0.1, 0.3).bias;
        assert!(mid > head_on);
        assert!(grazing > mid);
    }

    #[test]
    fn approx_clamps_out_of_range_inputs() {
        // NoV and roughness beyond 0..=1 saturate to the clamped evaluation.
        assert_eq!(env_brdf_approx(2.0, 0.3), env_brdf_approx(1.0, 0.3));
        assert_eq!(env_brdf_approx(-1.0, 0.3), env_brdf_approx(0.0, 0.3));
        assert_eq!(env_brdf_approx(0.5, 2.0), env_brdf_approx(0.5, 1.0));
        assert_eq!(env_brdf_approx(0.5, -1.0), env_brdf_approx(0.5, 0.0));
    }

    #[test]
    fn rgb_and_scalar_agree_for_gray_f0() {
        let f0 = 0.08;
        let scalar = env_brdf_specular(0.7, 0.4, f0);
        let rgb = env_brdf_specular_rgb(0.7, 0.4, Vec3::splat(f0));
        assert!(approx(rgb.x, scalar, CMP_EPS));
        assert!(approx(rgb.y, scalar, CMP_EPS));
        assert!(approx(rgb.z, scalar, CMP_EPS));
    }

    #[test]
    fn rgb_tints_per_channel() {
        // A tinted (metal) F0 scales each channel independently, sharing bias.
        let f0 = Vec3::new(0.9, 0.6, 0.3);
        let terms = env_brdf_approx(0.6, 0.5);
        let rgb = env_brdf_specular_rgb(0.6, 0.5, f0);
        assert!(approx(rgb.x, f0.x * terms.scale + terms.bias, CMP_EPS));
        assert!(approx(rgb.y, f0.y * terms.scale + terms.bias, CMP_EPS));
        assert!(approx(rgb.z, f0.z * terms.scale + terms.bias, CMP_EPS));
    }

    #[test]
    fn lut_rejects_malformed_grids() {
        // Too small for bilinear.
        assert!(DfgLut::new(1, alloc::vec![EnvBrdfTerms::new(0.0, 0.0)]).is_none());
        assert!(DfgLut::new(0, Vec::new()).is_none());
        // Wrong element count.
        assert!(DfgLut::new(2, alloc::vec![EnvBrdfTerms::new(0.0, 0.0); 3]).is_none());
        // Correct count builds.
        assert!(DfgLut::new(2, alloc::vec![EnvBrdfTerms::new(0.0, 0.0); 4]).is_some());
    }

    /// A `3 x 3` grid whose `(scale, bias)` encode their `(ix, iy)` indices, so
    /// a sample's expected interpolation is easy to hand-check.
    fn index_grid() -> DfgLut {
        let size = 3;
        let mut data = Vec::with_capacity(size * size);
        for iy in 0..size {
            for ix in 0..size {
                // scale = ix, bias = iy as floats.
                data.push(EnvBrdfTerms::new(ix as f32, iy as f32));
            }
        }
        DfgLut::new(size, data).expect("valid 3x3 grid")
    }

    #[test]
    fn lut_hits_grid_points_exactly() {
        let lut = index_grid();
        // size = 3 -> lattice at NoV/roughness in {0, 0.5, 1}, exactly
        // representable, so 0.5 * 2 = 1.0 lands on an integer texel.
        let s = lut.sample(0.5, 1.0);
        assert!(approx(s.scale, 1.0, CMP_EPS)); // ix = 1
        assert!(approx(s.bias, 2.0, CMP_EPS)); // iy = 2
        let corner = lut.sample(0.0, 0.0);
        assert!(approx(corner.scale, 0.0, CMP_EPS));
        assert!(approx(corner.bias, 0.0, CMP_EPS));
        let far = lut.sample(1.0, 1.0);
        assert!(approx(far.scale, 2.0, CMP_EPS));
        assert!(approx(far.bias, 2.0, CMP_EPS));
    }

    #[test]
    fn lut_bilinear_interpolates_between_texels() {
        let lut = index_grid();
        // Quarter of the way along NoV (0.25 * 2 = 0.5 cell units) sits halfway
        // between texels ix = 0 and ix = 1, so scale = 0.5.
        let s = lut.sample(0.25, 0.0);
        assert!(approx(s.scale, 0.5, CMP_EPS));
        assert!(approx(s.bias, 0.0, CMP_EPS));
        // Center of the whole grid averages the four index corners.
        let mid = lut.sample(0.25, 0.25);
        assert!(approx(mid.scale, 0.5, CMP_EPS));
        assert!(approx(mid.bias, 0.5, CMP_EPS));
    }

    #[test]
    fn lut_clamps_out_of_range_queries() {
        let lut = index_grid();
        let clamped = lut.sample(5.0, -2.0);
        let edge = lut.sample(1.0, 0.0);
        assert_eq!(clamped, edge);
    }

    #[test]
    fn lut_std430_round_trips_and_sizes() {
        let lut = index_grid();
        let packed = lut.to_std430();
        assert_eq!(packed.len(), 2 * 3 * 3);
        // First texel is (0, 0); second column texel is (1, 0).
        assert!(approx(f32::from_bits(packed[0]), 0.0, CMP_EPS));
        assert!(approx(f32::from_bits(packed[1]), 0.0, CMP_EPS));
        assert!(approx(f32::from_bits(packed[2]), 1.0, CMP_EPS));
        assert!(approx(f32::from_bits(packed[3]), 0.0, CMP_EPS));
        // Byte size uses the vec2 stride.
        assert_eq!(lut.buffer_bytes(), VEC2_STRIDE * 9);
    }

    #[test]
    fn lut_texel_saturates_out_of_range_indices() {
        let lut = index_grid();
        // Indices past the edge clamp to the last row/column.
        assert_eq!(lut.texel(99, 99), lut.texel(2, 2));
    }

    #[test]
    fn prefilter_mip_lod_maps_roughness_linearly() {
        // 5 mips -> max LOD 4.
        assert!(approx(prefilter_mip_lod(0.0, 5), 0.0, CMP_EPS));
        assert!(approx(prefilter_mip_lod(1.0, 5), 4.0, CMP_EPS));
        assert!(approx(prefilter_mip_lod(0.5, 5), 2.0, CMP_EPS));
        // Clamped roughness.
        assert!(approx(prefilter_mip_lod(2.0, 5), 4.0, CMP_EPS));
        assert!(approx(prefilter_mip_lod(-1.0, 5), 0.0, CMP_EPS));
        // Degenerate chains always sample LOD 0.
        assert!(approx(prefilter_mip_lod(0.7, 1), 0.0, CMP_EPS));
        assert!(approx(prefilter_mip_lod(0.7, 0), 0.0, CMP_EPS));
    }
}
