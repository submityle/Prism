//! Specular anti-aliasing (`Toksvig` / normal-variance) `CPU` gold standard for
//! particle shading (design §16, §17).
//!
//! High-gloss surfaces alias badly under minification: as a normal map is
//! minified, one shaded texel covers many microscopic normals whose highlights
//! flicker as the camera or geometry moves. This module owns the `CPU`
//! reference that suppresses that flicker by *widening the effective specular
//! lobe* instead of narrowing it, so a future `GPU` draw kernel can match it bit
//! for bit. It deliberately does **not** evaluate a full `BRDF`: it only derives
//! the extra roughness that the shading model (see [`super::shading`]) then
//! feeds into its own `NDF`. It is likewise distinct from the `Fresnel` rim
//! ([`super::fresnel_rim`]) and depth-to-normal reconstruction
//! ([`super::normal_reconstruct`]).
//!
//! Three classic pieces compose here, all transcendental-free:
//!
//! 1. **`Toksvig` factor** — from the length of the *averaged (unnormalized)*
//!    normal `|Na|` and a gloss power `s`, the factor
//!    `|Na| / (|Na| + s * (1 - |Na|))` widens the effective `NdotH` distribution
//!    (see [`toksvig_factor`]). A unit average length (`|Na| = 1`, no variation)
//!    yields a factor of `1` (no widening); as the footprint's normals diverge,
//!    `|Na|` shrinks and the factor falls.
//! 2. **`Frostbite` geometric specular anti-aliasing** — a normal *variance* is
//!    turned into a roughness increment via
//!    `roughness' = sqrt(roughness^2 + min(2 * variance, kappa))`, with `kappa`
//!    clamping the maximum added roughness (see [`frostbite_specular_aa`]).
//! 3. **`mip` normal-length chain** — averaging normal vectors down a box `mip`
//!    chain and taking each level's average length yields the `|Na|` a shaded
//!    texel sees at that `mip`, which maps monotonically to roughness (see
//!    [`mip_average_normal_lengths`] and [`normal_length_variance`]).
//!
//! Roughness is carried in two conventions: *perceptual* roughness (the
//! artist-facing value) and *linear* roughness (the `NDF` `alpha`). They convert
//! by the squaring relation `linear = perceptual * perceptual` and its inverse
//! `perceptual = sqrt(linear)` — never `powf` (see
//! [`perceptual_to_linear_roughness`] / [`linear_to_perceptual_roughness`]).
//!
//! Only `f32` `sqrt`, `floor`, integer `div_ceil`, and rational arithmetic are
//! used, so the result is deterministic and portable. `GPU` packing follows the
//! shared `std430` `vec4` alignment from [`super::gpu_layout`].

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Minimum average-normal length used to clamp the `Toksvig` / variance
/// denominators away from zero, so a fully degenerate footprint never divides
/// by zero or yields a `NaN`.
const MIN_LEN: f32 = 1e-6;

/// Minimum squared length below which a hand-rolled direction is treated as
/// degenerate and normalizes to the zero vector instead of dividing by zero.
const MIN_LEN_SQ: f32 = 1e-12;

/// Byte stride of one [`SpecularAaParams`] record in a `std430` storage buffer.
///
/// The four scalars pack into a single `vec4` slot
/// `vec4(base_perceptual_roughness, screen_variance, variance_clamp,
/// gloss_power)`.
pub const SPECULAR_AA_PARAMS_STRIDE: usize = VEC4_STRIDE;

/// A hand-rolled 3-component normal vector.
///
/// This module is self-contained and does not borrow the engine's shared
/// `Vec3`, so it defines the minimal algebra it needs. The additive operators
/// are named [`NormalVec3::plus`] / [`NormalVec3::minus`] to avoid colliding
/// with the `core::ops` traits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalVec3 {
    /// The x component.
    pub x: f32,
    /// The y component.
    pub y: f32,
    /// The z component.
    pub z: f32,
}

impl NormalVec3 {
    /// The zero vector.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Creates a vector from its three components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Component-wise sum (named `plus` to avoid the `core::ops::Add` trait).
    #[must_use]
    pub fn plus(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    /// Component-wise difference (named `minus` to avoid `core::ops::Sub`).
    #[must_use]
    pub fn minus(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    /// Uniform scale by a scalar.
    #[must_use]
    pub fn scale(self, k: f32) -> Self {
        Self::new(self.x * k, self.y * k, self.z * k)
    }

    /// Dot product with another vector.
    #[must_use]
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Squared Euclidean length.
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Euclidean length via `sqrt`.
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Normalizes the vector, returning [`NormalVec3::ZERO`] for a degenerate
    /// (near-zero-length) input instead of producing a `NaN`.
    #[must_use]
    pub fn normalize_or_zero(self) -> Self {
        let len_sq = self.length_squared();
        if len_sq < MIN_LEN_SQ {
            return Self::ZERO;
        }
        let inv_len = 1.0 / len_sq.sqrt();
        self.scale(inv_len)
    }
}

/// Clamps a scalar into `0..=1` without branching on equality.
#[must_use]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// The length of the *averaged (unnormalized)* normal over a footprint.
///
/// The normals are summed, divided by the count, and the length of the mean
/// vector is taken and clamped to `0..=1`. For unit-length inputs this is the
/// `Toksvig` `|Na|`: it is `1` when every normal agrees and shrinks toward `0`
/// as they diverge. An empty slice yields `1` (a degenerate footprint is
/// treated as perfectly coherent, i.e. no widening).
#[must_use]
pub fn average_normal_length(normals: &[NormalVec3]) -> f32 {
    if normals.is_empty() {
        return 1.0;
    }
    let mut sum = NormalVec3::ZERO;
    for &n in normals {
        sum = sum.plus(n);
    }
    let inv_count = 1.0 / (normals.len() as f32);
    clamp01(sum.scale(inv_count).length())
}

/// The `Toksvig` factor from an average normal length and a gloss power.
///
/// Returns `|Na| / (|Na| + s * (1 - |Na|))`, clamped to `0..=1`, where `|Na|`
/// (`avg_normal_length`) is clamped to `MIN_LEN..=1` and `s` (`gloss_power`) is
/// clamped to be non-negative. At `|Na| = 1` (no normal variation) the factor is
/// `1` (no widening); as `|Na|` shrinks the factor falls, monotonically widening
/// the effective specular lobe.
#[must_use]
pub fn toksvig_factor(avg_normal_length: f32, gloss_power: f32) -> f32 {
    let r = avg_normal_length.clamp(MIN_LEN, 1.0);
    let s = gloss_power.max(0.0);
    let denom = r + s * (1.0 - r);
    clamp01(r / denom)
}

/// The effective gloss power after `Toksvig` widening: `factor * s`.
///
/// This is the `Toksvig` mapping of a Blinn-style gloss power `s` for the
/// footprint, always less than or equal to the input gloss (equal only when the
/// footprint's normals are perfectly coherent).
#[must_use]
pub fn toksvig_effective_gloss(avg_normal_length: f32, gloss_power: f32) -> f32 {
    toksvig_factor(avg_normal_length, gloss_power) * gloss_power.max(0.0)
}

/// The normal-distribution variance implied by an average normal length.
///
/// Uses the `Toksvig`/von-Mises-Fisher approximation `variance = (1 - r) / r`
/// with `r` (`avg_normal_length`) clamped to `MIN_LEN..=1`. At `r = 1` the
/// variance is `0` (a coherent footprint, the identity case); as `r` shrinks the
/// variance grows monotonically. This variance is what
/// [`frostbite_specular_aa`] consumes.
#[must_use]
pub fn normal_length_variance(avg_normal_length: f32) -> f32 {
    let r = avg_normal_length.clamp(MIN_LEN, 1.0);
    (1.0 - r) / r
}

/// Converts perceptual roughness to linear (`NDF` `alpha`) roughness.
///
/// The linear roughness is the square of the perceptual roughness
/// (`linear = perceptual * perceptual`); the input is clamped to `0..=1` so the
/// result stays in `0..=1`. Squaring replaces any `powf` call.
#[must_use]
pub fn perceptual_to_linear_roughness(perceptual: f32) -> f32 {
    let p = clamp01(perceptual);
    p * p
}

/// Converts linear (`NDF` `alpha`) roughness back to perceptual roughness.
///
/// The perceptual roughness is the square root of the linear roughness
/// (`perceptual = sqrt(linear)`); the input is clamped to `0..=1`. This is the
/// exact inverse of [`perceptual_to_linear_roughness`] on `0..=1`.
#[must_use]
pub fn linear_to_perceptual_roughness(linear: f32) -> f32 {
    clamp01(linear).sqrt()
}

/// `Frostbite` geometric specular anti-aliasing applied in linear-roughness
/// space.
///
/// Computes `roughness' = sqrt(clamp(roughness^2 + min(2 * variance, kappa), 0,
/// 1))`, where `roughness` (`linear_roughness`) and the result are linear (`NDF`
/// `alpha`) roughness. `variance` is clamped to be non-negative and `kappa`
/// (`variance_clamp`) caps the added kernel roughness so a huge variance cannot
/// blow the lobe wide open. The returned roughness is always greater than or
/// equal to the input (the kernel term is non-negative), and it equals the input
/// exactly when `variance` is `0`.
#[must_use]
pub fn frostbite_specular_aa(linear_roughness: f32, variance: f32, kappa: f32) -> f32 {
    let base = clamp01(linear_roughness);
    let base_sq = base * base;
    let kernel = (2.0 * variance.max(0.0)).min(kappa.max(0.0));
    let filtered = (base_sq + kernel).clamp(0.0, 1.0);
    filtered.sqrt()
}

/// The mean over a `mip` level of its per-texel *averaged-normal* lengths.
///
/// Each texel's normal length is clamped to `0..=1` (unit inputs give `1`), and
/// the level's mean is returned. A coarser level, whose texels are box averages
/// of finer neighbours, has shorter per-texel lengths when those neighbours
/// diverge, so this mean falls down the chain.
#[must_use]
fn mean_texel_length(level: &[NormalVec3]) -> f32 {
    if level.is_empty() {
        return 1.0;
    }
    let mut sum = 0.0;
    for &n in level {
        sum += clamp01(n.length());
    }
    sum / (level.len() as f32)
}

/// The averaged normal length at each level of a box `mip` chain.
///
/// Starting from a base level of normals, each successive level averages
/// adjacent pairs of normal *vectors* (a 2:1 box `mip` downsample, using
/// [`usize::div_ceil`] for odd lengths) and records the whole-level average
/// length via [`average_normal_length`]. As divergent normals are averaged
/// together the recorded length shrinks down the chain, which
/// [`normal_length_variance`] maps to increasing roughness. An empty base yields
/// an empty chain.
#[must_use]
pub fn mip_average_normal_lengths(base: &[NormalVec3]) -> Vec<f32> {
    let mut lengths = Vec::new();
    if base.is_empty() {
        return lengths;
    }
    let mut current: Vec<NormalVec3> = base.to_vec();
    loop {
        lengths.push(mean_texel_length(&current));
        if current.len() <= 1 {
            break;
        }
        let next_len = current.len().div_ceil(2);
        let mut next = Vec::with_capacity(next_len);
        let mut i = 0;
        while i < current.len() {
            let a = current[i];
            let b = if i + 1 < current.len() {
                current[i + 1]
            } else {
                a
            };
            next.push(a.plus(b).scale(0.5));
            i += 2;
        }
        current = next;
    }
    lengths
}

/// The result of evaluating specular anti-aliasing for one shaded footprint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpecularAaResult {
    /// The anti-aliased linear (`NDF` `alpha`) roughness.
    pub linear_roughness: f32,
    /// The anti-aliased perceptual roughness (`sqrt` of [`Self::linear_roughness`]).
    pub perceptual_roughness: f32,
    /// The `Toksvig` factor for the footprint (see [`toksvig_factor`]).
    pub toksvig_factor: f32,
    /// The kernel roughness actually added in linear-roughness-squared space,
    /// i.e. `min(2 * total_variance, kappa)`.
    pub added_variance: f32,
}

/// Parameters controlling specular anti-aliasing for a renderer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpecularAaParams {
    /// The artist-facing base perceptual roughness (`0..=1`).
    pub base_perceptual_roughness: f32,
    /// An extra interpolated-normal variance from screen-space derivatives,
    /// added on top of the `mip` normal-length variance (non-negative).
    pub screen_variance: f32,
    /// The `kappa` clamp on the maximum added kernel roughness.
    pub variance_clamp: f32,
    /// The Blinn-style gloss power used for the reported [`toksvig_factor`].
    pub gloss_power: f32,
}

impl SpecularAaParams {
    /// Creates specular-anti-aliasing parameters from all fields.
    #[must_use]
    pub const fn new(
        base_perceptual_roughness: f32,
        screen_variance: f32,
        variance_clamp: f32,
        gloss_power: f32,
    ) -> Self {
        Self {
            base_perceptual_roughness,
            screen_variance,
            variance_clamp,
            gloss_power,
        }
    }

    /// Evaluates the anti-aliased roughness for a footprint's average normal
    /// length.
    ///
    /// The `mip` normal-length variance ([`normal_length_variance`]) is added to
    /// the parameter's [`Self::screen_variance`], fed through
    /// [`frostbite_specular_aa`] in linear-roughness space, and converted back to
    /// perceptual roughness. When `avg_normal_length` is `1` and
    /// [`Self::screen_variance`] is `0`, the perceptual roughness is returned
    /// unchanged (the identity case).
    #[must_use]
    pub fn evaluate(&self, avg_normal_length: f32) -> SpecularAaResult {
        let map_variance = normal_length_variance(avg_normal_length);
        let total_variance = map_variance + self.screen_variance.max(0.0);
        let base_linear = perceptual_to_linear_roughness(self.base_perceptual_roughness);
        let linear_prime = frostbite_specular_aa(base_linear, total_variance, self.variance_clamp);
        let perceptual_prime = linear_to_perceptual_roughness(linear_prime);
        let added = (2.0 * total_variance).min(self.variance_clamp.max(0.0));
        SpecularAaResult {
            linear_roughness: linear_prime,
            perceptual_roughness: perceptual_prime,
            toksvig_factor: toksvig_factor(avg_normal_length, self.gloss_power),
            added_variance: added,
        }
    }

    /// Packs the parameters into their `std430` `vec4`-aligned word layout.
    ///
    /// The four scalars map to one `vec4` in the order
    /// `(base_perceptual_roughness, screen_variance, variance_clamp,
    /// gloss_power)`, matching [`SPECULAR_AA_PARAMS_STRIDE`].
    #[must_use]
    pub fn to_std430(&self) -> [u32; 4] {
        [
            self.base_perceptual_roughness.to_bits(),
            self.screen_variance.to_bits(),
            self.variance_clamp.to_bits(),
            self.gloss_power.to_bits(),
        ]
    }
}

/// The `std430` byte size of a storage buffer holding `count` records of
/// [`SpecularAaParams`], clamped up to a single element (see
/// [`super::gpu_layout::storage_bytes`]).
#[must_use]
pub fn specular_aa_buffer_bytes(count: usize) -> usize {
    storage_bytes(SPECULAR_AA_PARAMS_STRIDE, count)
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
    fn vector_algebra_is_hand_rolled() {
        let a = NormalVec3::new(1.0, 2.0, 3.0);
        let b = NormalVec3::new(4.0, 5.0, 6.0);
        assert_eq!(a.plus(b), NormalVec3::new(5.0, 7.0, 9.0));
        assert_eq!(b.minus(a), NormalVec3::new(3.0, 3.0, 3.0));
        assert_eq!(a.scale(2.0), NormalVec3::new(2.0, 4.0, 6.0));
        assert!(approx_eq(a.dot(b), 32.0));
        assert!(approx_eq(NormalVec3::new(3.0, 4.0, 0.0).length(), 5.0));
    }

    #[test]
    fn normalize_zero_vector_is_zero_not_nan() {
        assert_eq!(NormalVec3::ZERO.normalize_or_zero(), NormalVec3::ZERO);
        let unit = NormalVec3::new(0.0, 5.0, 0.0).normalize_or_zero();
        assert!(approx_eq(unit.dot(unit), 1.0));
    }

    #[test]
    fn coherent_footprint_has_unit_average_length() {
        let n = NormalVec3::new(0.0, 0.0, 1.0);
        let footprint = [n, n, n, n];
        assert!(approx_eq(average_normal_length(&footprint), 1.0));
    }

    #[test]
    fn opposing_normals_average_to_zero_length() {
        let up = NormalVec3::new(0.0, 0.0, 1.0);
        let down = NormalVec3::new(0.0, 0.0, -1.0);
        assert!(approx_eq(average_normal_length(&[up, down]), 0.0));
    }

    #[test]
    fn empty_footprint_is_treated_as_coherent() {
        assert!(approx_eq(average_normal_length(&[]), 1.0));
    }

    #[test]
    fn unit_average_normal_gives_toksvig_factor_one() {
        assert!(approx_eq(toksvig_factor(1.0, 64.0), 1.0));
        assert!(approx_eq(toksvig_factor(1.0, 8.0), 1.0));
    }

    #[test]
    fn diverging_normals_shrink_toksvig_factor() {
        let coherent = toksvig_factor(1.0, 32.0);
        let mild = toksvig_factor(0.9, 32.0);
        let strong = toksvig_factor(0.5, 32.0);
        assert!(mild < coherent);
        assert!(strong < mild);
        assert!((0.0..=1.0).contains(&strong));
    }

    #[test]
    fn effective_gloss_never_exceeds_input_gloss() {
        let s = 48.0;
        assert!(toksvig_effective_gloss(0.7, s) <= s);
        assert!(approx_eq(toksvig_effective_gloss(1.0, s), s));
    }

    #[test]
    fn normal_length_variance_is_zero_at_unit_length() {
        assert!(approx_eq(normal_length_variance(1.0), 0.0));
    }

    #[test]
    fn normal_length_variance_grows_as_length_shrinks() {
        let low = normal_length_variance(0.9);
        let mid = normal_length_variance(0.6);
        let high = normal_length_variance(0.3);
        assert!(mid > low);
        assert!(high > mid);
    }

    #[test]
    fn perceptual_linear_round_trips() {
        for &p in &[0.0_f32, 0.1, 0.25, 0.5, 0.8, 1.0] {
            let linear = perceptual_to_linear_roughness(p);
            let back = linear_to_perceptual_roughness(linear);
            assert!(approx_eq(back, p));
        }
    }

    #[test]
    fn perceptual_to_linear_is_the_square() {
        assert!(approx_eq(perceptual_to_linear_roughness(0.5), 0.25));
        assert!(approx_eq(perceptual_to_linear_roughness(0.3), 0.09));
    }

    #[test]
    fn zero_variance_is_the_identity() {
        let r = 0.4_f32;
        assert!(approx_eq(frostbite_specular_aa(r, 0.0, 1.0), r));
    }

    #[test]
    fn larger_variance_yields_larger_roughness() {
        let base = frostbite_specular_aa(0.3, 0.0, 1.0);
        let more = frostbite_specular_aa(0.3, 0.05, 1.0);
        let most = frostbite_specular_aa(0.3, 0.2, 1.0);
        assert!(more > base);
        assert!(most > more);
    }

    #[test]
    fn output_roughness_is_monotone_at_least_input() {
        for &r in &[0.0_f32, 0.2, 0.5, 0.75, 1.0] {
            let out = frostbite_specular_aa(r, 0.3, 1.0);
            assert!(out >= clamp01(r) - CMP_EPS);
        }
    }

    #[test]
    fn kappa_clamps_the_added_kernel() {
        // A huge variance without a clamp would saturate roughness to 1; a tight
        // kappa caps the kernel term so the result stays well below saturation.
        let clamped = frostbite_specular_aa(0.1, 100.0, 0.01);
        let unclamped = frostbite_specular_aa(0.1, 100.0, 1.0);
        assert!(clamped < unclamped);
        // With kappa = 0.01 and base^2 = 0.01, filtered = 0.02 -> sqrt(0.02).
        assert!(approx_eq(clamped, 0.02_f32.sqrt()));
    }

    #[test]
    fn output_roughness_stays_in_unit_range() {
        let out = frostbite_specular_aa(1.0, 1000.0, 10.0);
        assert!((0.0..=1.0).contains(&out));
        let low = frostbite_specular_aa(-5.0, -5.0, -5.0);
        assert!((0.0..=1.0).contains(&low));
    }

    #[test]
    fn mip_chain_length_shrinks_with_divergence() {
        // A checkerboard of opposing normals: coarser mips average them and the
        // recorded average length falls monotonically.
        let up = NormalVec3::new(0.0, 0.0, 1.0);
        let tilt = NormalVec3::new(0.8, 0.0, 0.6).normalize_or_zero();
        let base = [up, tilt, up, tilt, up, tilt, up, tilt];
        let chain = mip_average_normal_lengths(&base);
        assert_eq!(chain.len(), 4); // 8 -> 4 -> 2 -> 1
        for pair in chain.windows(2) {
            assert!(pair[1] <= pair[0] + CMP_EPS);
        }
        assert!(chain[chain.len() - 1] < chain[0]);
    }

    #[test]
    fn mip_chain_of_coherent_normals_stays_unit() {
        let n = NormalVec3::new(0.0, 1.0, 0.0);
        let base = [n, n, n, n, n];
        let chain = mip_average_normal_lengths(&base);
        // 5 -> 3 -> 2 -> 1
        assert_eq!(chain.len(), 4);
        for &len in &chain {
            assert!(approx_eq(len, 1.0));
        }
    }

    #[test]
    fn empty_base_yields_empty_mip_chain() {
        assert!(mip_average_normal_lengths(&[]).is_empty());
    }

    #[test]
    fn coarser_mip_maps_to_more_roughness() {
        let up = NormalVec3::new(0.0, 0.0, 1.0);
        let tilt = NormalVec3::new(1.0, 0.0, 0.2).normalize_or_zero();
        let base = [up, tilt, up, tilt];
        let chain = mip_average_normal_lengths(&base);
        let fine = frostbite_specular_aa(0.2, normal_length_variance(chain[0]), 1.0);
        let coarse =
            frostbite_specular_aa(0.2, normal_length_variance(chain[chain.len() - 1]), 1.0);
        assert!(coarse >= fine);
    }

    #[test]
    fn evaluate_is_identity_for_coherent_footprint() {
        let params = SpecularAaParams::new(0.5, 0.0, 1.0, 32.0);
        let result = params.evaluate(1.0);
        assert!(approx_eq(result.perceptual_roughness, 0.5));
        assert!(approx_eq(result.toksvig_factor, 1.0));
        assert!(approx_eq(result.added_variance, 0.0));
    }

    #[test]
    fn evaluate_widens_roughness_for_divergent_footprint() {
        let params = SpecularAaParams::new(0.3, 0.1, 1.0, 16.0);
        let coherent = params.evaluate(1.0);
        let divergent = params.evaluate(0.4);
        assert!(divergent.perceptual_roughness > coherent.perceptual_roughness);
        assert!(divergent.toksvig_factor < coherent.toksvig_factor);
    }

    #[test]
    fn std430_stride_and_buffer_size_follow_layout() {
        assert_eq!(SPECULAR_AA_PARAMS_STRIDE, 16);
        assert_eq!(specular_aa_buffer_bytes(0), 16);
        assert_eq!(specular_aa_buffer_bytes(4), 64);
    }

    #[test]
    fn std430_packing_preserves_scalar_bits() {
        let params = SpecularAaParams::new(0.5, 0.125, 0.75, 24.0);
        let words = params.to_std430();
        assert_eq!(words[0], 0.5_f32.to_bits());
        assert_eq!(words[1], 0.125_f32.to_bits());
        assert_eq!(words[2], 0.75_f32.to_bits());
        assert_eq!(words[3], 24.0_f32.to_bits());
    }
}
