//! Screen-space per-particle motion blur: turning one screen-space motion
//! vector into a symmetric multi-`tap` sampling kernel (design §21).
//!
//! This module owns exactly one narrow job in the motion-blur pipeline: given a
//! particle's *already-computed* screen-space motion vector (pixels per frame)
//! and a shutter model, produce the along-vector `tap` offsets, their
//! normalized weights, and the weighted `tap` accumulation that a future `GPU`
//! draw kernel evaluates to smear a particle along its motion. It mirrors the
//! reconstruction-filter blur of production engines (Unreal, Unity `VFX Graph`,
//! `Frostbite`) at the algorithm level without reusing any of their code.
//!
//! **Strict boundaries.** This module never *computes* the screen-space motion
//! vector itself and never reprojects world positions — that is
//! [`super::motion_vectors`]'s `ScreenMotionVector` / `CameraMotionState`
//! territory. It also never stretches a `billboard` quad's geometry — that is
//! [`super::sprite_stretch`]'s `StretchParams` / `stretched_corners`
//! territory. Here the motion vector is a *given input*; the output is a blur
//! `tap` kernel plus its weighted colour accumulation.
//!
//! Everything is built from ordinary arithmetic plus `sqrt` only — no
//! transcendental functions. Randomness for `tap` jitter comes from a
//! self-contained integer bit-mixer (this module never depends on
//! [`super::determinism`] or [`super::noise`]), so a `CPU` reference and a
//! `GPU` kernel reproduce the same jittered `tap` positions bit for bit.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, U32_STRIDE, VEC4_STRIDE};

/// Smallest vector length treated as non-zero when normalizing a direction.
///
/// A shorter vector has no well-defined direction, so it normalizes to the zero
/// vector rather than dividing by a vanishing length.
pub const EPS_LEN: f32 = 1e-12;

/// Falloff constant of the rational `tap`-weight profile.
///
/// The weight is `1 / (1 + FALLOFF * t*t)` for the normalized `tap` coordinate
/// `t` in `[-1, 1]`, so the centre `tap` (`t = 0`) is weighted `1` and the
/// endpoints (`t = ±1`) are weighted `1 / (1 + FALLOFF)`. A rational profile
/// stays strictly positive everywhere, so the weight sum is never zero and the
/// normalization below can never divide by zero — unlike a bare `1 - t*t`
/// polynomial whose endpoints vanish. It also avoids the transcendental `exp`
/// of a Gaussian kernel.
const WEIGHT_FALLOFF: f32 = 3.0;

/// Reciprocal of `2^24`, scaling a 24-bit integer exactly into `[0, 1)`.
///
/// A 24-bit payload fits the `f32` mantissa exactly, so the multiply is exact
/// and the mapping is uniform; multiplying by a constant reciprocal is ordinary
/// arithmetic, never a transcendental call.
const INV_2POW24: f32 = 1.0 / 16_777_216.0;

/// `std430` alignment (bytes) the parameter block rounds up to.
///
/// A four-scalar block occupies one `vec4`-sized slot, so a params array packs
/// at this stride with no padding.
pub const STD430_ALIGN: usize = VEC4_STRIDE;

/// A minimal two-component screen-space (`UV` / pixel) vector.
///
/// Motion blur lives in the two-dimensional screen plane, so this module uses
/// its own small `2D` type rather than the three-dimensional shared vector. It
/// uses only add / subtract / multiply / divide plus `sqrt`, never a
/// transcendental function, and derives only [`PartialEq`] (no `Eq` / `Hash`)
/// because it holds `f32` fields.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    /// Horizontal (`U`) component, in pixels.
    pub x: f32,
    /// Vertical (`V`) component, in pixels.
    pub y: f32,
}

impl Vec2 {
    /// The zero vector.
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    /// Constructs a vector from its components.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Uniformly scales both components.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
        }
    }

    /// Squared Euclidean length (no `sqrt`).
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    /// Euclidean length in pixels.
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Returns the unit-length direction, or [`Vec2::ZERO`] when the vector is
    /// shorter than [`EPS_LEN`] and has no well-defined direction.
    #[must_use]
    pub fn normalize_or_zero(self) -> Self {
        let len = self.length();
        if len <= EPS_LEN {
            Self::ZERO
        } else {
            self.scale(1.0 / len)
        }
    }
}

/// A linear `RGBA` colour sample used by the `tap` accumulator.
///
/// The four channels are premultiplied-agnostic here; the accumulator simply
/// forms a weighted average per channel.
pub type Rgba = [f32; 4];

/// `std430`-friendly motion-blur parameters (design §21).
///
/// The block is four 4-byte scalars — three `f32` and one `u32` — so it packs
/// into a single `vec4`-sized `std430` slot with no interior padding. All
/// fields are validity-clamped by [`MotionBlurParams::new`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionBlurParams {
    /// Maximum blur-streak length in pixels; the computed streak is clamped to
    /// this so a fast particle cannot smear across the whole frame.
    pub max_blur_px: f32,
    /// Per-particle `tap` jitter strength in `[0, 1]`, expressed as a fraction
    /// of one `tap` spacing. Jitter decorrelates the fixed `tap` grid to fight
    /// banding artifacts.
    pub jitter_strength: f32,
    /// Shutter fraction in `[0, 1]`: how much of the frame interval the shutter
    /// is open. A screen speed of `s` pixels/frame smears over `s * shutter`
    /// pixels before clamping.
    pub shutter: f32,
    /// Number of `tap` samples along the streak, at least one.
    pub tap_count: u32,
}

impl MotionBlurParams {
    /// Builds a validity-clamped parameter block.
    ///
    /// `max_blur_px` is floored at zero, `jitter_strength` and `shutter` are
    /// clamped to `[0, 1]`, and `tap_count` is floored at one so the kernel
    /// always has at least a single centre `tap`.
    #[must_use]
    pub fn new(max_blur_px: f32, jitter_strength: f32, shutter: f32, tap_count: u32) -> Self {
        Self {
            max_blur_px: max_blur_px.max(0.0),
            jitter_strength: jitter_strength.clamp(0.0, 1.0),
            shutter: shutter.clamp(0.0, 1.0),
            tap_count: tap_count.max(1),
        }
    }

    /// Byte size of the `std430` parameter block: four 4-byte scalars.
    #[must_use]
    pub fn std430_size() -> usize {
        storage_bytes(U32_STRIDE, 4)
    }

    /// Effective `tap` count as a `usize`, always at least one.
    #[must_use]
    pub fn effective_taps(&self) -> usize {
        usize::try_from(self.tap_count).unwrap_or(1).max(1)
    }

    /// Streak length in pixels for a particle moving `screen_speed_px` pixels
    /// per frame: `screen_speed_px * shutter`, clamped to `[0, max_blur_px]`.
    ///
    /// A negative input speed is treated as zero; the screen motion vector's
    /// magnitude is a non-negative length.
    #[must_use]
    pub fn span_length(&self, screen_speed_px: f32) -> f32 {
        let raw = screen_speed_px.max(0.0) * self.shutter;
        raw.clamp(0.0, self.max_blur_px)
    }

    /// Uniform normalized `tap` spacing in `[-1, 1]` space, or zero for a
    /// single `tap`.
    #[must_use]
    pub fn tap_spacing(&self) -> f32 {
        let n = self.effective_taps();
        if n <= 1 {
            0.0
        } else {
            let denom = to_f32_usize(n - 1);
            2.0 / denom
        }
    }

    /// Per-particle jitter displacement in normalized `[-1, 1]` `tap` space.
    ///
    /// The displacement is a symmetric fraction of one `tap` spacing driven by
    /// the self-contained integer hash, so it is deterministic per
    /// `particle_seed` and reproduces bit for bit on any backend.
    #[must_use]
    pub fn jitter_offset(&self, particle_seed: u32) -> f32 {
        let unit = unit_f32_from_bits(hash_u32(particle_seed));
        (unit - 0.5) * self.jitter_strength * self.tap_spacing()
    }

    /// Builds the along-vector blur `tap` kernel for one particle.
    ///
    /// `motion_px` is the *given* screen-space motion vector (pixels/frame);
    /// this module only consumes it and never recomputes it. The returned
    /// offsets are measured from the particle centre, symmetric about it (up to
    /// the deterministic per-particle jitter), and the weights are normalized to
    /// sum to one. A zero-length motion vector or zero streak collapses to a
    /// single centre `tap`-equivalent: every offset is the zero vector.
    #[must_use]
    pub fn build_taps(&self, motion_px: Vec2, particle_seed: u32) -> TapKernel {
        let n = self.effective_taps();
        let dir = motion_px.normalize_or_zero();
        let half = self.span_length(motion_px.length()) * 0.5;
        let jitter = self.jitter_offset(particle_seed);

        let (offsets, raw_weights): (Vec<Vec2>, Vec<f32>) = (0..n)
            .map(|i| {
                let t = (sample_coord(i, n) + jitter).clamp(-1.0, 1.0);
                (dir.scale(t * half), tap_weight(t))
            })
            .unzip();

        let weights = normalize_weights(&raw_weights);
        TapKernel { offsets, weights }
    }
}

/// A generated set of blur `tap` offsets and their normalized weights.
///
/// `offsets[i]` is the screen-space displacement (pixels) from the particle
/// centre for `tap` `i`; `weights[i]` is its normalized contribution. The two
/// vectors always have equal length, and the weights sum to one (within
/// floating-point rounding) whenever there is at least one `tap`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TapKernel {
    /// Per-`tap` screen-space offsets from the particle centre, in pixels.
    pub offsets: Vec<Vec2>,
    /// Per-`tap` normalized weights, summing to one.
    pub weights: Vec<f32>,
}

impl TapKernel {
    /// Number of `tap` samples in the kernel.
    #[must_use]
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    /// Whether the kernel has no `tap` samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }
}

/// Splits a *given* screen-space motion vector into its unit direction and its
/// length in pixels.
///
/// This is a pure decomposition of an input vector, not a recomputation of the
/// motion vector itself; a zero-length input yields a zero direction and zero
/// speed.
#[must_use]
pub fn dir_and_speed(motion_px: Vec2) -> (Vec2, f32) {
    (motion_px.normalize_or_zero(), motion_px.length())
}

/// Rational `tap` weight for a normalized `tap` coordinate `t` in `[-1, 1]`.
///
/// `1 / (1 + WEIGHT_FALLOFF * t*t)` peaks at the centre and decays toward the
/// endpoints while staying strictly positive, so it never contributes a zero or
/// negative weight and avoids the transcendental `exp` of a Gaussian.
#[must_use]
pub fn tap_weight(t: f32) -> f32 {
    1.0 / (1.0 + WEIGHT_FALLOFF * t * t)
}

/// Cubic `smoothstep` `t*t*(3 - 2t)` clamped to `[0, 1]`, for soft edges and
/// blur falloff.
///
/// Used to fade blur strength in and out without a transcendental curve; inputs
/// outside `[0, 1]` saturate to the endpoints.
#[must_use]
pub fn smoothstep(t: f32) -> f32 {
    let x = t.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Forms the weighted average of the sampled `tap` colours.
///
/// `colors[i]` is the scene colour sampled at `tap` `i`'s offset and
/// `weights[i]` its weight; the result is the per-channel weighted mean. The
/// shorter of the two slices bounds the sum. An empty input or a non-positive
/// weight sum yields a fully transparent black `RGBA` sample.
#[must_use]
pub fn accumulate_taps(colors: &[Rgba], weights: &[f32]) -> Rgba {
    let mut acc: Rgba = [0.0; 4];
    let mut wsum = 0.0f32;
    for (color, &w) in colors.iter().zip(weights.iter()) {
        wsum += w;
        for (channel, &c) in acc.iter_mut().zip(color.iter()) {
            *channel += c * w;
        }
    }
    if wsum <= 0.0 {
        return [0.0; 4];
    }
    let inv = 1.0 / wsum;
    for channel in &mut acc {
        *channel *= inv;
    }
    acc
}

/// Normalized `tap` coordinate in `[-1, 1]` for `tap` `i` of `n`.
///
/// A single `tap` sits at the centre; otherwise the coordinates are evenly
/// spaced endpoint-to-endpoint so the set is symmetric about the centre.
fn sample_coord(i: usize, n: usize) -> f32 {
    if n <= 1 {
        0.0
    } else {
        let denom = to_f32_usize(n - 1);
        -1.0 + 2.0 * to_f32_usize(i) / denom
    }
}

/// Divides each weight by the total so the returned weights sum to one.
///
/// A non-positive total (only possible for an empty input) returns the weights
/// unchanged, since there is nothing to normalize.
fn normalize_weights(raw: &[f32]) -> Vec<f32> {
    let sum: f32 = raw.iter().sum();
    if sum <= 0.0 {
        return raw.to_vec();
    }
    let inv = 1.0 / sum;
    raw.iter().map(|&w| w * inv).collect()
}

/// Converts a `usize` to `f32` for `tap`-coordinate arithmetic.
///
/// `tap` counts are tiny (a handful of samples), far below the `f32` mantissa's
/// exact-integer range, so this conversion is exact in practice.
fn to_f32_usize(v: usize) -> f32 {
    // Small `tap` counts convert exactly; a lossy cast is intended and safe.
    let clamped = u32::try_from(v).unwrap_or(u32::MAX);
    from_u32(clamped)
}

/// Converts a `u32` to `f32`.
fn from_u32(v: u32) -> f32 {
    // Widening a 32-bit integer to `f32` is intentional for arithmetic.
    v as f32
}

/// A self-contained `SplitMix32`-style integer bit-mixer.
///
/// It scrambles every input bit across the output using only shifts, xors, and
/// odd multiplies, so it is a pure, backend-reproducible function. Kept local so
/// this module depends on no shared randomness module.
fn hash_u32(mut x: u32) -> u32 {
    x = x.wrapping_add(0x9e37_79b9);
    x = (x ^ (x >> 16)).wrapping_mul(0x85eb_ca6b);
    x = (x ^ (x >> 13)).wrapping_mul(0xc2b2_ae35);
    x ^ (x >> 16)
}

/// Maps a 32-bit hash word to a uniform `f32` in `[0, 1)`.
///
/// Only the high 24 bits are used so the value lands exactly on a multiple of
/// `2^-24`, keeping the mapping uniform and exactly representable.
fn unit_f32_from_bits(bits: u32) -> f32 {
    // The 24-bit payload is exactly representable, so this cast is intentional.
    let payload = bits >> 8;
    from_u32(payload) * INV_2POW24
}

#[cfg(test)]
mod tests {
    use super::*;

    const CMP_EPS: f32 = 1e-6;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    fn vec_close(a: Vec2, b: Vec2) -> bool {
        close(a.x, b.x) && close(a.y, b.y)
    }

    fn params() -> MotionBlurParams {
        MotionBlurParams::new(20.0, 0.0, 1.0, 5)
    }

    #[test]
    fn span_length_clamps_to_max() {
        let p = MotionBlurParams::new(10.0, 0.0, 1.0, 4);
        assert!(close(p.span_length(100.0), 10.0));
        assert!(close(p.span_length(5.0), 5.0));
    }

    #[test]
    fn span_length_scales_with_shutter_and_speed() {
        let p = MotionBlurParams::new(100.0, 0.0, 0.5, 4);
        assert!(close(p.span_length(8.0), 4.0));
        assert!(close(p.span_length(20.0), 10.0));
    }

    #[test]
    fn span_length_zero_and_negative_speed_is_zero() {
        let p = params();
        assert!(close(p.span_length(0.0), 0.0));
        assert!(close(p.span_length(-7.0), 0.0));
    }

    #[test]
    fn zero_speed_collapses_to_zero_offsets() {
        let p = params();
        let kernel = p.build_taps(Vec2::ZERO, 42);
        assert_eq!(kernel.len(), 5);
        for &off in &kernel.offsets {
            assert!(off.length() < CMP_EPS);
        }
        let sum: f32 = kernel.weights.iter().sum();
        assert!(close(sum, 1.0));
    }

    #[test]
    fn single_tap_kernel_is_centered_full_weight() {
        let p = MotionBlurParams::new(20.0, 0.5, 1.0, 1);
        let kernel = p.build_taps(Vec2::new(30.0, 0.0), 7);
        assert_eq!(kernel.len(), 1);
        assert!(vec_close(kernel.offsets[0], Vec2::ZERO));
        assert!(close(kernel.weights[0], 1.0));
    }

    #[test]
    fn taps_are_symmetric_without_jitter() {
        let p = params();
        let kernel = p.build_taps(Vec2::new(40.0, 0.0), 123);
        let n = kernel.len();
        for i in 0..n {
            let mirror = kernel.offsets[n - 1 - i];
            let negated = kernel.offsets[i].scale(-1.0);
            assert!(vec_close(mirror, negated));
        }
    }

    #[test]
    fn weights_are_normalized() {
        let p = MotionBlurParams::new(20.0, 0.0, 1.0, 7);
        let kernel = p.build_taps(Vec2::new(0.0, 50.0), 9);
        let sum: f32 = kernel.weights.iter().sum();
        assert!(close(sum, 1.0));
    }

    #[test]
    fn center_weight_dominates_edges() {
        let p = params();
        let kernel = p.build_taps(Vec2::new(40.0, 0.0), 0);
        let center = kernel.len() / 2;
        assert!(kernel.weights[center] >= kernel.weights[0]);
        assert!(kernel.weights[center] >= kernel.weights[kernel.len() - 1]);
    }

    #[test]
    fn jitter_is_deterministic() {
        let p = MotionBlurParams::new(20.0, 1.0, 1.0, 6);
        let motion = Vec2::new(25.0, 25.0);
        let a = p.build_taps(motion, 1337);
        let b = p.build_taps(motion, 1337);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.offsets.iter().zip(b.offsets.iter()) {
            assert!(vec_close(*x, *y));
        }
    }

    #[test]
    fn jitter_varies_with_seed() {
        let p = MotionBlurParams::new(20.0, 1.0, 1.0, 6);
        let motion = Vec2::new(25.0, 25.0);
        let a = p.build_taps(motion, 1);
        let b = p.build_taps(motion, 2);
        let differs = a
            .offsets
            .iter()
            .zip(b.offsets.iter())
            .any(|(x, y)| !vec_close(*x, *y));
        assert!(differs);
    }

    #[test]
    fn accumulate_weighted_average_is_correct() {
        let colors = [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]];
        let weights = [1.0, 3.0];
        let out = accumulate_taps(&colors, &weights);
        assert!(close(out[0], 0.25));
        assert!(close(out[1], 0.75));
        assert!(close(out[2], 0.0));
        assert!(close(out[3], 1.0));
    }

    #[test]
    fn accumulate_empty_is_transparent_black() {
        let out = accumulate_taps(&[], &[]);
        for &c in &out {
            assert!(close(c, 0.0));
        }
    }

    #[test]
    fn accumulate_stops_at_shorter_slice() {
        let colors = [[2.0, 2.0, 2.0, 2.0]];
        let weights = [4.0, 1.0, 1.0];
        let out = accumulate_taps(&colors, &weights);
        for &c in &out {
            assert!(close(c, 2.0));
        }
    }

    #[test]
    fn smoothstep_endpoints_and_midpoint() {
        assert!(close(smoothstep(0.0), 0.0));
        assert!(close(smoothstep(1.0), 1.0));
        assert!(close(smoothstep(0.5), 0.5));
        assert!(close(smoothstep(-4.0), 0.0));
        assert!(close(smoothstep(4.0), 1.0));
    }

    #[test]
    fn tap_weight_peaks_at_center() {
        assert!(close(tap_weight(0.0), 1.0));
        assert!(tap_weight(0.0) > tap_weight(1.0));
        assert!(close(tap_weight(1.0), tap_weight(-1.0)));
    }

    #[test]
    fn dir_and_speed_decomposes_vector() {
        let (dir, speed) = dir_and_speed(Vec2::new(3.0, 4.0));
        assert!(close(speed, 5.0));
        assert!(vec_close(dir, Vec2::new(0.6, 0.8)));
        let (zdir, zspeed) = dir_and_speed(Vec2::ZERO);
        assert!(close(zspeed, 0.0));
        assert!(vec_close(zdir, Vec2::ZERO));
    }

    #[test]
    fn std430_size_and_alignment() {
        assert_eq!(MotionBlurParams::std430_size(), 16);
        assert_eq!(MotionBlurParams::std430_size() % STD430_ALIGN, 0);
        assert_eq!(STD430_ALIGN, VEC4_STRIDE);
    }

    #[test]
    fn new_clamps_fields_into_range() {
        let p = MotionBlurParams::new(-5.0, 3.0, 9.0, 0);
        assert!(close(p.max_blur_px, 0.0));
        assert!(close(p.jitter_strength, 1.0));
        assert!(close(p.shutter, 1.0));
        assert_eq!(p.tap_count, 1);
        assert_eq!(p.effective_taps(), 1);
    }

    #[test]
    fn hash_is_deterministic_and_varies() {
        assert_eq!(hash_u32(5), hash_u32(5));
        assert_ne!(hash_u32(5), hash_u32(6));
        let u = unit_f32_from_bits(hash_u32(99));
        assert!((0.0..1.0).contains(&u));
    }

    #[test]
    fn jitter_bounded_by_spacing() {
        let p = MotionBlurParams::new(20.0, 1.0, 1.0, 5);
        let spacing = p.tap_spacing();
        for seed in 0..64u32 {
            let j = p.jitter_offset(seed);
            assert!(j.abs() <= 0.5 * spacing + CMP_EPS);
        }
    }
}
