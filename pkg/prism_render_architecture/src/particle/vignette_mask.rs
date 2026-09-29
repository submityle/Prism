//! Screen-edge `vignette` mask for the particle post/compositing stack
//! (design §16, §21).
//!
//! A `vignette` darkens an image toward its edges: the framing device that
//! guides the eye toward the center and hides the harsh rectangle of the frame.
//! Production post stacks (Unreal's post-process `vignette`, `Frostbite`'s lens
//! model, Unity `HDRP`'s procedural + textured `vignette`) all drive the same
//! quantity — a per-pixel *mask* in `[0, 1]` that multiplies the composited
//! color. This module owns the `CPU`-verifiable *maths* of that contract: given
//! a screen `UV` in `[0, 1]^2` and the mask parameters, it returns the mask
//! weight, applies it to an `RGB` color, and packs the parameters into the
//! `std430` block a `GPU` post pass binds.
//!
//! # Shape
//!
//! The mask is radial about a configurable optical center. The horizontal axis
//! is scaled by an aspect factor so a non-square framebuffer produces circular
//! (rather than stretched) contours. A `roundness` control blends between a
//! chebyshev (box) distance and a euclidean (round) distance, so the same
//! parameter block expresses both a soft round `vignette` and a boxed frame.
//!
//! # Falloff and determinism
//!
//! Inside the inner radius the mask is exactly `1.0` (untouched); beyond the
//! outer radius it saturates to `1 - intensity` (fully darkened). Between the
//! two it follows a `smoothstep` band position fed through a rational-polynomial
//! softening curve, both monotonic with clamped endpoints. Everything is pure
//! algebra plus [`f32::sqrt`] and integer-loop powers: no transcendental
//! function (`sin`/`cos`/`exp`/`ln`/`pow`) and no `f32::round`/`f32::ceil` is
//! ever called, matching the determinism contract of the sibling
//! [`super::depth_of_field`] module so a future `GPU` evaluation reproduces the
//! `CPU` result.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Spans (and denominators) with magnitude below this are treated as zero so
/// evaluation falls back to a defined result instead of dividing by (near) zero
/// or propagating `NaN`.
const MIN_SPAN: f32 = 1e-6;

/// Raises `base` to a non-negative integer power via a multiplicative fold,
/// avoiding the forbidden transcendental [`f32::powf`]. An exponent of zero
/// yields `1.0`.
#[must_use]
fn powi(base: f32, exp: u32) -> f32 {
    (0..exp).fold(1.0, |acc, _| acc * base)
}

/// Evaluates the `smoothstep` interpolation of `x` across `[edge0, edge1]`,
/// returning `0.0` at or below `edge0`, `1.0` at or above `edge1`, and the
/// cubic `3t^2 - 2t^3` in between. A degenerate (near-zero-width) edge interval
/// falls back to a hard step at `edge0`.
#[must_use]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    if span.abs() < MIN_SPAN {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / span).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// A rational-polynomial S-curve softening of the band position `t`:
/// `t^2 / (t^2 + (1 - t)^2)`. Strictly increasing on `[0, 1]` with exact
/// endpoints `soft(0) = 0` and `soft(1) = 1`, so it deepens the mid-band
/// transition without changing the fully-lit or fully-dark limits. The
/// denominator is at least `0.5` for `t` in `[0, 1]`, so the division is safe.
#[must_use]
fn rational_soften(t: f32) -> f32 {
    let a = powi(t, 2);
    let b = powi(1.0 - t, 2);
    let denom = a + b;
    if denom < MIN_SPAN {
        return t;
    }
    a / denom
}

/// The parameters a screen-space `vignette` post pass evaluates against.
///
/// All distances are in the aspect-corrected `UV` metric where the optical
/// center sits at `center` and the horizontal axis is scaled by `aspect`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VignetteParams {
    /// Optical center in screen `UV` (`[0, 1]^2`, typically `[0.5, 0.5]`).
    pub center: [f32; 2],
    /// Radius (aspect-corrected `UV` units) inside which the mask is `1.0`.
    pub inner_radius: f32,
    /// Radius beyond which the mask saturates to `1 - intensity`.
    pub outer_radius: f32,
    /// Darkening strength in `[0, 1]`: the edge mask is `1 - intensity`.
    pub intensity: f32,
    /// Shape blend in `[0, 1]`: `0.0` is a chebyshev box, `1.0` is a euclidean
    /// round `vignette`; values between interpolate the two distances.
    pub roundness: f32,
    /// Horizontal scale applied to `uv.x - center.x` so a non-square
    /// framebuffer yields circular contours (`aspect = width / height`).
    pub aspect: f32,
}

impl VignetteParams {
    /// Byte size of the `std430` packing: seven scalars occupy the first
    /// `vec4` plus three scalars of the second, padded up to two `vec4` slots so
    /// the block honors the 16-byte `std430` base alignment expected by the
    /// `GPU` kernel.
    pub const STD430_SIZE: usize = 2 * VEC4_STRIDE;

    /// Builds a parameter block from its fields.
    #[must_use]
    pub const fn new(
        center: [f32; 2],
        inner_radius: f32,
        outer_radius: f32,
        intensity: f32,
        roundness: f32,
        aspect: f32,
    ) -> Self {
        Self {
            center,
            inner_radius,
            outer_radius,
            intensity,
            roundness,
            aspect,
        }
    }

    /// The aspect-corrected, shape-blended distance from `uv` to the optical
    /// center. `roundness = 1.0` yields the euclidean (round) distance,
    /// `roundness = 0.0` the chebyshev (box) distance, and values between
    /// linearly interpolate the two.
    #[must_use]
    pub fn shape_distance(&self, uv: [f32; 2]) -> f32 {
        let dx = (uv[0] - self.center[0]) * self.aspect;
        let dy = uv[1] - self.center[1];
        let euclid = (dx * dx + dy * dy).sqrt();
        let cheby = dx.abs().max(dy.abs());
        let r = self.roundness.clamp(0.0, 1.0);
        cheby + (euclid - cheby) * r
    }

    /// The `vignette` mask weight at `uv`, in `[1 - intensity, 1]`.
    ///
    /// Exactly `1.0` at or inside [`VignetteParams::inner_radius`], exactly
    /// `1 - intensity` at or beyond [`VignetteParams::outer_radius`], and a
    /// monotonically decreasing `smoothstep`-then-rational transition between.
    #[must_use]
    pub fn evaluate(&self, uv: [f32; 2]) -> f32 {
        let dist = self.shape_distance(uv);
        let band = smoothstep(self.inner_radius, self.outer_radius, dist);
        let shaped = rational_soften(band);
        let intensity = self.intensity.clamp(0.0, 1.0);
        (1.0 - intensity * shaped).clamp(0.0, 1.0)
    }

    /// Applies the mask at `uv` to a linear `RGB` color, multiplying every
    /// channel by [`VignetteParams::evaluate`].
    #[must_use]
    pub fn apply(&self, color_rgb: [f32; 3], uv: [f32; 2]) -> [f32; 3] {
        let mask = self.evaluate(uv);
        [
            color_rgb[0] * mask,
            color_rgb[1] * mask,
            color_rgb[2] * mask,
        ]
    }

    /// Packs the parameters into their `std430` uniform-block bytes.
    ///
    /// The seven scalars are laid out little-endian as `f32`s
    /// (`center.x, center.y, inner_radius, outer_radius, intensity, roundness,
    /// aspect`); the trailing scalar is zero padding so the block spans two
    /// `vec4` slots ([`VignetteParams::STD430_SIZE`] bytes).
    #[must_use]
    pub fn to_std430(&self) -> [u8; Self::STD430_SIZE] {
        let fields = [
            self.center[0],
            self.center[1],
            self.inner_radius,
            self.outer_radius,
            self.intensity,
            self.roundness,
            self.aspect,
        ];
        let mut bytes = [0u8; Self::STD430_SIZE];
        for (slot, value) in bytes.chunks_exact_mut(4).zip(fields.iter()) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    /// Packs a slice of parameter blocks into one contiguous `std430` byte
    /// buffer (element stride [`VignetteParams::STD430_SIZE`]), the layout a
    /// `GPU` storage array of `vignette` params binds.
    #[must_use]
    pub fn pack_slice(params: &[Self]) -> Vec<u8> {
        let mut buffer = Vec::with_capacity(Self::STD430_SIZE * params.len());
        for p in params {
            buffer.extend_from_slice(&p.to_std430());
        }
        buffer
    }

    /// Total `std430` byte size of a storage buffer holding `count` packed
    /// [`VignetteParams`] blocks, clamped up to a single element per the shared
    /// [`storage_bytes`] rule.
    #[must_use]
    pub fn gpu_storage_bytes(count: usize) -> usize {
        storage_bytes(Self::STD430_SIZE, count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for the `f32` equality comparisons used by the tests;
    /// direct `==` on floating point is intentionally avoided.
    const CMP_EPS: f32 = 1e-6;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    /// A representative round `vignette`: centered, soft band, strong edge.
    fn params() -> VignetteParams {
        VignetteParams::new([0.5, 0.5], 0.2, 0.5, 0.8, 1.0, 1.0)
    }

    #[test]
    fn powi_is_integer_power() {
        assert!(approx(powi(2.0, 0), 1.0));
        assert!(approx(powi(2.0, 3), 8.0));
        assert!(approx(powi(0.5, 2), 0.25));
    }

    #[test]
    fn rational_soften_has_clamped_endpoints() {
        assert!(approx(rational_soften(0.0), 0.0));
        assert!(approx(rational_soften(1.0), 1.0));
        assert!(approx(rational_soften(0.5), 0.5));
        // Strictly increasing across the unit interval.
        let mut previous = -1.0;
        for i in 0..=10 {
            #[expect(
                clippy::cast_precision_loss,
                reason = "small loop index maps exactly to f32"
            )]
            let t = i as f32 / 10.0;
            let s = rational_soften(t);
            assert!(s > previous, "soften must increase: {s} <= {previous}");
            previous = s;
        }
    }

    #[test]
    fn center_mask_is_one() {
        let p = params();
        assert!(approx(p.evaluate(p.center), 1.0));
        // Anywhere strictly inside the inner radius is also fully lit.
        assert!(approx(p.evaluate([0.55, 0.55]), 1.0));
    }

    #[test]
    fn corners_are_darkest_edge_value() {
        let p = params();
        let edge = 1.0 - p.intensity;
        // All four corners lie beyond the outer radius, so they saturate.
        for corner in [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]] {
            assert!(approx(p.evaluate(corner), edge));
        }
        // The corner value is the global minimum of the mask.
        let center = p.evaluate(p.center);
        assert!(p.evaluate([0.0, 0.0]) <= center);
    }

    #[test]
    fn smoothstep_band_endpoints() {
        let p = params();
        // Exactly at the inner radius: fully lit. Exactly at the outer radius
        // (and beyond): fully darkened.
        assert!(approx(p.evaluate([0.5 + p.inner_radius, 0.5]), 1.0));
        assert!(approx(
            p.evaluate([0.5 + p.outer_radius, 0.5]),
            1.0 - p.intensity
        ));
    }

    #[test]
    fn mask_decreases_monotonically_across_band() {
        let p = params();
        let mut previous = f32::INFINITY;
        // Walk radially outward from the inner to the outer radius.
        for i in 0..=20 {
            #[expect(
                clippy::cast_precision_loss,
                reason = "small loop index maps exactly to f32"
            )]
            let frac = i as f32 / 20.0;
            let radius = p.inner_radius + (p.outer_radius - p.inner_radius) * frac;
            let mask = p.evaluate([0.5 + radius, 0.5]);
            assert!(
                mask <= previous + CMP_EPS,
                "mask must be non-increasing outward: {mask} > {previous}"
            );
            previous = mask;
        }
    }

    #[test]
    fn aspect_correction_is_symmetric() {
        let p = VignetteParams::new([0.5, 0.5], 0.2, 0.5, 0.8, 1.0, 2.0);
        // Mirror across the horizontal axis about the center.
        assert!(approx(p.evaluate([0.65, 0.5]), p.evaluate([0.35, 0.5])));
        // Mirror across the vertical axis about the center.
        assert!(approx(p.evaluate([0.5, 0.7]), p.evaluate([0.5, 0.3])));
        // The horizontal axis is aspect-scaled, so an x-offset darkens faster
        // than the same y-offset.
        assert!(p.evaluate([0.7, 0.5]) < p.evaluate([0.5, 0.7]));
    }

    #[test]
    fn roundness_blends_box_and_round() {
        let round = VignetteParams::new([0.5, 0.5], 0.2, 0.9, 0.8, 1.0, 1.0);
        let square = VignetteParams::new([0.5, 0.5], 0.2, 0.9, 0.8, 0.0, 1.0);
        let uv = [0.9, 0.9];
        // The diagonal point is farther under the euclidean (round) metric than
        // the chebyshev (box) metric, so the round mask is darker there.
        assert!(round.shape_distance(uv) > square.shape_distance(uv));
        assert!(round.evaluate(uv) <= square.evaluate(uv) + CMP_EPS);
        // On an axis the two metrics coincide.
        assert!(approx(
            round.shape_distance([0.9, 0.5]),
            square.shape_distance([0.9, 0.5])
        ));
    }

    #[test]
    fn apply_scales_color_by_mask() {
        let p = params();
        let color = [0.4, 0.6, 1.0];
        let uv = [0.0, 0.0];
        let mask = p.evaluate(uv);
        let out = p.apply(color, uv);
        assert!(approx(out[0], color[0] * mask));
        assert!(approx(out[1], color[1] * mask));
        assert!(approx(out[2], color[2] * mask));
        // At the center the color is untouched.
        let lit = p.apply(color, p.center);
        assert!(approx(lit[0], color[0]));
        assert!(approx(lit[1], color[1]));
        assert!(approx(lit[2], color[2]));
    }

    #[test]
    fn evaluation_is_deterministic() {
        let p = params();
        let uv = [0.31, 0.72];
        assert!(approx(p.evaluate(uv), p.evaluate(uv)));
        assert_eq!(p.to_std430(), p.to_std430());
    }

    #[test]
    fn std430_layout_matches_fields() {
        let p = params();
        let bytes = p.to_std430();
        assert_eq!(bytes.len(), VignetteParams::STD430_SIZE);
        assert_eq!(VignetteParams::STD430_SIZE, 32);
        // std430 base alignment: the block spans whole vec4 slots.
        assert_eq!(VignetteParams::STD430_SIZE % VEC4_STRIDE, 0);
        let expected = [
            p.center[0],
            p.center[1],
            p.inner_radius,
            p.outer_radius,
            p.intensity,
            p.roundness,
            p.aspect,
        ];
        for (i, value) in expected.iter().enumerate() {
            let mut word = [0u8; 4];
            word.copy_from_slice(&bytes[i * 4..i * 4 + 4]);
            assert!(approx(f32::from_le_bytes(word), *value));
        }
        // The trailing padding scalar is zero.
        assert_eq!(&bytes[28..32], &[0u8; 4]);
    }

    #[test]
    fn pack_slice_concatenates_blocks() {
        let a = params();
        let b = VignetteParams::new([0.4, 0.6], 0.1, 0.7, 0.5, 0.0, 1.5);
        let buffer = VignetteParams::pack_slice(&[a, b]);
        assert_eq!(buffer.len(), 2 * VignetteParams::STD430_SIZE);
        assert_eq!(&buffer[..VignetteParams::STD430_SIZE], &a.to_std430());
        assert_eq!(&buffer[VignetteParams::STD430_SIZE..], &b.to_std430());
    }

    #[test]
    fn gpu_storage_bytes_clamps_and_scales() {
        assert_eq!(
            VignetteParams::gpu_storage_bytes(0),
            VignetteParams::STD430_SIZE
        );
        assert_eq!(
            VignetteParams::gpu_storage_bytes(4),
            4 * VignetteParams::STD430_SIZE
        );
    }
}
