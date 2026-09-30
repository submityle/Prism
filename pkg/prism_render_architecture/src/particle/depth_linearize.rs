//! Perspective depth linearization and reverse-`Z` reconstruction: the
//! `CPU`-verifiable contract that converts a non-linear depth-buffer value into
//! a linear view-space distance and back, matching the depth math every
//! screen-space particle pass silently depends on (design §16-§21).
//!
//! # Why depth is not linear
//!
//! A perspective projection stores `1 / z` shaped depth, not `z`, so the depth
//! buffer packs enormous precision next to the near plane and almost none near
//! the far plane. Any pass that wants a *metric* view-space distance — soft
//! particles fading against the scene, depth-of-field circle-of-confusion,
//! screen-space fog — must first undo that non-linearity. This module owns that
//! single arithmetic primitive and its exact inverse.
//!
//! # Depth conventions
//!
//! Two device conventions are supported without pulling in the projection
//! matrix:
//!
//! * **`D3D` / `wgpu` `[0, 1]` clip depth** — the near plane maps to `0` and the
//!   far plane to `1`. The linear distance is
//!   `near * far / (far - depth * (far - near))`; at `depth = 0` this is exactly
//!   `near` and at `depth = 1` it is exactly `far`.
//! * **`OpenGL` `[-1, 1]` `NDC` depth** — the near plane maps to `-1` and the
//!   far plane to `+1`, giving `2 * near * far / (far + near - ndc * (far - near))`.
//!
//! # Reverse-`Z`
//!
//! Reverse-`Z` flips the mapping so the near plane writes `1` and the far plane
//! writes `0`, which spreads floating-point precision far more evenly across the
//! view frustum. It is expressed here as a single pre-flip (`depth = 1 - depth`)
//! before the shared `[0, 1]` formula, and a matching post-flip on the inverse,
//! so both endpoints agree with the ordinary-`Z` path.
//!
//! # Strict scope — only scalars cross the boundary
//!
//! This file deliberately shares *nothing* with its neighbors beyond bare
//! scalar `near` / `far` / `depth` values. It does not import or reference the
//! [`super::camera`] projection type, the [`super::depth_downsample`] reduction
//! pyramid, the [`super::depth_of_field`] circle-of-confusion model, or the
//! [`super::normal_reconstruct`] position rebuild — each of those *consumes* a
//! linear depth but none of them owns the linearization formula, and this module
//! never reaches back into their types. The only sibling import is the shared
//! [`super::gpu_layout`] `std430` byte-layout helper.
//!
//! # Determinism
//!
//! Every routine is plain `f32` `+ - * /` guarded against division by zero with
//! an epsilon comparison; there are no transcendental functions, no `f32`
//! `==` / `!=`, and no rounding. Results are bit-reproducible against a future
//! `GPU` kernel.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, U32_STRIDE, VEC4_STRIDE};

/// Epsilon below which a denominator (or the `far - near` range) is treated as
/// degenerate so the routine falls back instead of dividing by zero.
const CMP_EPS: f32 = 1e-6;

/// Packed byte size of [`DepthParams::to_std430`]: a single `vec4<f32>` slot.
pub const DEPTH_LINEARIZE_STD430_SIZE: usize = VEC4_STRIDE;

/// The scalar depth-range parameters needed to linearize a perspective depth.
///
/// Only the three scalars a shader would push in a uniform are stored; no
/// projection matrix, camera, or sibling-pass type is referenced.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthParams {
    /// Positive distance to the near clip plane (view space).
    pub near: f32,
    /// Positive distance to the far clip plane (view space), expected `> near`.
    pub far: f32,
    /// Whether the depth buffer uses reverse-`Z` (near writes `1`, far `0`).
    pub reverse_z: bool,
}

impl DepthParams {
    /// Builds a parameter set from raw near / far distances and the reverse-`Z`
    /// flag.
    #[must_use]
    pub const fn new(near: f32, far: f32, reverse_z: bool) -> Self {
        Self {
            near,
            far,
            reverse_z,
        }
    }

    /// The signed depth range `far - near`.
    #[must_use]
    pub fn range(&self) -> f32 {
        self.far - self.near
    }

    /// Whether the range is degenerate (`far` and `near` within [`CMP_EPS`]).
    #[must_use]
    pub fn is_degenerate(&self) -> bool {
        self.range().abs() < CMP_EPS
    }

    /// Serializes `near`, `far`, and the reverse-`Z` flag into one `std430`
    /// `vec4<f32>` slot: `[near, far, reverse_flag, pad]`, little-endian.
    ///
    /// The boolean is packed as `1.0` / `0.0` through a conditional rather than
    /// a cast, matching the `GPU`-side `f32` uniform field.
    #[must_use]
    pub fn to_std430(&self) -> [u8; DEPTH_LINEARIZE_STD430_SIZE] {
        let reverse_flag: f32 = if self.reverse_z { 1.0 } else { 0.0 };
        let mut bytes = [0u8; DEPTH_LINEARIZE_STD430_SIZE];
        bytes[0..U32_STRIDE].copy_from_slice(&self.near.to_le_bytes());
        bytes[U32_STRIDE..2 * U32_STRIDE].copy_from_slice(&self.far.to_le_bytes());
        bytes[2 * U32_STRIDE..3 * U32_STRIDE].copy_from_slice(&reverse_flag.to_le_bytes());
        // The final `U32_STRIDE` bytes stay zero as `vec4` padding.
        bytes
    }
}

/// Total `std430` byte size of a storage buffer holding `count` [`DepthParams`]
/// slots, reusing the shared clamp-to-one rule from [`super::gpu_layout`].
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(DEPTH_LINEARIZE_STD430_SIZE, count)
}

/// Converts a `[0, 1]` (`D3D` / `wgpu`) non-linear depth-buffer value into a
/// positive view-space linear distance.
///
/// Under reverse-`Z` the value is pre-flipped (`depth = 1 - depth`) so the near
/// plane's `1` and the far plane's `0` land on `near` and `far` respectively.
/// A degenerate range or denominator falls back to a clip-plane distance rather
/// than dividing by zero.
#[must_use]
pub fn linearize_01(params: &DepthParams, ndc_depth_01: f32) -> f32 {
    let range = params.range();
    if range.abs() < CMP_EPS {
        return params.near;
    }
    let depth = if params.reverse_z {
        1.0 - ndc_depth_01
    } else {
        ndc_depth_01
    };
    let denom = params.far - depth * range;
    if denom.abs() < CMP_EPS {
        return params.far;
    }
    params.near * params.far / denom
}

/// Inverse of [`linearize_01`]: converts a positive view-space linear distance
/// back into the `[0, 1]` depth-buffer value, applying the reverse-`Z` post-flip
/// when configured.
///
/// A degenerate range or a non-positive linear distance falls back to a
/// clip-plane depth instead of dividing by zero.
#[must_use]
pub fn delinearize_01(params: &DepthParams, linear: f32) -> f32 {
    let range = params.range();
    if range.abs() < CMP_EPS {
        return if params.reverse_z { 1.0 } else { 0.0 };
    }
    if linear.abs() < CMP_EPS {
        return if params.reverse_z { 1.0 } else { 0.0 };
    }
    let depth = params.far * (linear - params.near) / (linear * range);
    if params.reverse_z {
        1.0 - depth
    } else {
        depth
    }
}

/// Normalizes a view-space linear distance to `[0, 1]` by
/// `(linear - near) / (far - near)`, clamped to the unit range.
///
/// Unlike [`linearize_01`] this is a plain linear remap (no `1 / z` shaping); it
/// is the value a debug visualizer or a linear fog factor wants.
#[must_use]
pub fn linear_to_01_normalized(params: &DepthParams, linear: f32) -> f32 {
    let range = params.range();
    if range.abs() < CMP_EPS {
        return 0.0;
    }
    f32::clamp((linear - params.near) / range, 0.0, 1.0)
}

/// Converts an `OpenGL`-style `[-1, 1]` `NDC` depth into a positive view-space
/// linear distance: `2 * near * far / (far + near - ndc * (far - near))`.
///
/// Reverse-`Z` is expressed as an `NDC` sign flip so the near plane's `+1`
/// (post-flip) still resolves to `near`. Degenerate ranges fall back to a
/// clip-plane distance.
#[must_use]
pub fn ndc_to_view_z(params: &DepthParams, ndc_z: f32) -> f32 {
    let range = params.range();
    if range.abs() < CMP_EPS {
        return params.near;
    }
    let ndc = if params.reverse_z { -ndc_z } else { ndc_z };
    let denom = params.far + params.near - ndc * range;
    if denom.abs() < CMP_EPS {
        return params.far;
    }
    2.0 * params.near * params.far / denom
}

/// Perspective-correct interpolation of a vertex attribute across a primitive
/// edge, weighting each endpoint by its `1 / w` (clip-space reciprocal depth).
///
/// A screen-linear parameter `t` does not interpolate attributes correctly under
/// perspective; the rasterizer instead blends `attr / w` and `1 / w` linearly
/// and divides. This mirrors that:
/// `(a * inv_w_a * (1 - t) + b * inv_w_b * t) / (inv_w_a * (1 - t) + inv_w_b * t)`.
/// The endpoints reproduce `a` at `t = 0` and `b` at `t = 1`; a degenerate
/// weight sum falls back to a plain screen-linear `lerp`.
#[must_use]
pub fn perspective_interpolate(a: f32, b: f32, inv_w_a: f32, inv_w_b: f32, t: f32) -> f32 {
    let w0 = inv_w_a * (1.0 - t);
    let w1 = inv_w_b * t;
    let denom = w0 + w1;
    if denom.abs() < CMP_EPS {
        return a + (b - a) * t;
    }
    (a * w0 + b * w1) / denom
}

/// Linearizes a whole slice of `[0, 1]` depth-buffer values, preserving order
/// and length. Equivalent to mapping [`linearize_01`] over the input.
#[must_use]
pub fn linearize_buffer(params: &DepthParams, depths: &[f32]) -> Vec<f32> {
    depths
        .iter()
        .map(|&depth| linearize_01(params, depth))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute-tolerance float comparison used throughout the tests so no raw
    /// `f32` `==` appears.
    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-3 * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn near_plane_depth_linearizes_to_near() {
        let p = DepthParams::new(0.5, 100.0, false);
        assert!(approx(linearize_01(&p, 0.0), 0.5));
    }

    #[test]
    fn far_plane_depth_linearizes_to_far() {
        let p = DepthParams::new(0.5, 100.0, false);
        assert!(approx(linearize_01(&p, 1.0), 100.0));
    }

    #[test]
    fn reverse_z_near_plane_is_one() {
        let p = DepthParams::new(0.5, 100.0, true);
        // Reverse-Z writes 1 at the near plane.
        assert!(approx(linearize_01(&p, 1.0), 0.5));
    }

    #[test]
    fn reverse_z_far_plane_is_zero() {
        let p = DepthParams::new(0.5, 100.0, true);
        // Reverse-Z writes 0 at the far plane.
        assert!(approx(linearize_01(&p, 0.0), 100.0));
    }

    #[test]
    fn reverse_and_standard_agree_at_both_ends() {
        let std = DepthParams::new(0.25, 50.0, false);
        let rev = DepthParams::new(0.25, 50.0, true);
        // Near plane: std depth 0 <-> rev depth 1.
        assert!(approx(linearize_01(&std, 0.0), linearize_01(&rev, 1.0)));
        // Far plane: std depth 1 <-> rev depth 0.
        assert!(approx(linearize_01(&std, 1.0), linearize_01(&rev, 0.0)));
    }

    #[test]
    fn linearize_then_delinearize_roundtrips_standard() {
        let p = DepthParams::new(0.5, 100.0, false);
        for &depth in &[0.0_f32, 0.1, 0.37, 0.5, 0.83, 1.0] {
            let linear = linearize_01(&p, depth);
            let back = delinearize_01(&p, linear);
            assert!(approx(back, depth), "depth {depth} -> {back}");
        }
    }

    #[test]
    fn linearize_then_delinearize_roundtrips_reverse_z() {
        let p = DepthParams::new(0.5, 100.0, true);
        for &depth in &[0.0_f32, 0.2, 0.5, 0.7, 1.0] {
            let linear = linearize_01(&p, depth);
            let back = delinearize_01(&p, linear);
            assert!(approx(back, depth), "depth {depth} -> {back}");
        }
    }

    #[test]
    fn delinearize_of_near_and_far_hits_endpoints() {
        let p = DepthParams::new(0.5, 100.0, false);
        assert!(approx(delinearize_01(&p, 0.5), 0.0));
        assert!(approx(delinearize_01(&p, 100.0), 1.0));
    }

    #[test]
    fn normalized_endpoints_are_zero_and_one() {
        let p = DepthParams::new(2.0, 40.0, false);
        assert!(approx(linear_to_01_normalized(&p, 2.0), 0.0));
        assert!(approx(linear_to_01_normalized(&p, 40.0), 1.0));
    }

    #[test]
    fn normalized_clamps_outside_range() {
        let p = DepthParams::new(2.0, 40.0, false);
        assert!(approx(linear_to_01_normalized(&p, -5.0), 0.0));
        assert!(approx(linear_to_01_normalized(&p, 1000.0), 1.0));
    }

    #[test]
    fn normalized_interior_stays_in_unit_range() {
        let p = DepthParams::new(1.0, 10.0, false);
        let v = linear_to_01_normalized(&p, 5.5);
        assert!((0.0..=1.0).contains(&v));
        // Midpoint of [1, 10] is 5.5 -> exactly 0.5.
        assert!(approx(v, 0.5));
    }

    #[test]
    fn perspective_interpolate_endpoint_t0_is_a() {
        assert!(approx(
            perspective_interpolate(3.0, 9.0, 0.25, 1.0, 0.0),
            3.0
        ));
    }

    #[test]
    fn perspective_interpolate_endpoint_t1_is_b() {
        assert!(approx(
            perspective_interpolate(3.0, 9.0, 0.25, 1.0, 1.0),
            9.0
        ));
    }

    #[test]
    fn perspective_interpolate_midpoint_is_nonlinear() {
        // Distinct 1/w weights bend the midpoint away from the screen-linear
        // average of 6.0.
        let v = perspective_interpolate(3.0, 9.0, 0.25, 1.0, 0.5);
        assert!(!approx(v, 6.0));
        // Weighted result: (3*0.25 + 9*1.0) / (0.25 + 1.0) = 7.8.
        assert!(approx(v, 7.8));
    }

    #[test]
    fn perspective_interpolate_equal_weights_is_linear() {
        let v = perspective_interpolate(2.0, 8.0, 0.5, 0.5, 0.25);
        // Equal 1/w collapses to a plain lerp: 2 + (8-2)*0.25 = 3.5.
        assert!(approx(v, 3.5));
    }

    #[test]
    fn perspective_interpolate_zero_weight_falls_back_to_lerp() {
        // Both reciprocals zero -> degenerate denominator -> screen lerp.
        let v = perspective_interpolate(4.0, 10.0, 0.0, 0.0, 0.5);
        assert!(approx(v, 7.0));
    }

    #[test]
    fn buffer_preserves_length() {
        let p = DepthParams::new(0.5, 100.0, false);
        let out = linearize_buffer(&p, &[0.0, 0.5, 1.0]);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn buffer_matches_scalar() {
        let p = DepthParams::new(0.5, 100.0, false);
        let depths = [0.0_f32, 0.2, 0.6, 1.0];
        let out = linearize_buffer(&p, &depths);
        for (i, &depth) in depths.iter().enumerate() {
            assert!(approx(out[i], linearize_01(&p, depth)));
        }
    }

    #[test]
    fn empty_buffer_is_empty() {
        let p = DepthParams::new(0.5, 100.0, false);
        assert!(linearize_buffer(&p, &[]).is_empty());
    }

    #[test]
    fn degenerate_range_is_safe_not_nan() {
        let p = DepthParams::new(10.0, 10.0, false);
        let l = linearize_01(&p, 0.5);
        assert!(l.is_finite());
        assert!(approx(l, 10.0));
        let d = delinearize_01(&p, 10.0);
        assert!(d.is_finite());
        let n = linear_to_01_normalized(&p, 10.0);
        assert!(n.is_finite());
        assert!(approx(n, 0.0));
        assert!(p.is_degenerate());
    }

    #[test]
    fn delinearize_nonpositive_linear_is_safe() {
        let p = DepthParams::new(0.5, 100.0, false);
        let d = delinearize_01(&p, 0.0);
        assert!(d.is_finite());
        let rev = DepthParams::new(0.5, 100.0, true);
        let dr = delinearize_01(&rev, 0.0);
        assert!(dr.is_finite());
        assert!(approx(dr, 1.0));
    }

    #[test]
    fn ndc_endpoints_map_to_near_and_far() {
        let p = DepthParams::new(0.5, 100.0, false);
        assert!(approx(ndc_to_view_z(&p, -1.0), 0.5));
        assert!(approx(ndc_to_view_z(&p, 1.0), 100.0));
    }

    #[test]
    fn ndc_reverse_z_flips_endpoints() {
        let p = DepthParams::new(0.5, 100.0, true);
        // Reverse-Z near plane sits at NDC +1 (pre-flip), resolving to near.
        assert!(approx(ndc_to_view_z(&p, 1.0), 0.5));
        assert!(approx(ndc_to_view_z(&p, -1.0), 100.0));
    }

    #[test]
    fn ndc_degenerate_range_is_safe() {
        let p = DepthParams::new(7.0, 7.0, false);
        assert!(approx(ndc_to_view_z(&p, 0.0), 7.0));
    }

    #[test]
    fn new_getters_roundtrip() {
        let p = DepthParams::new(0.3, 250.0, true);
        assert!(approx(p.near, 0.3));
        assert!(approx(p.far, 250.0));
        assert!(p.reverse_z);
        assert!(approx(p.range(), 249.7));
    }

    #[test]
    fn std430_size_is_one_vec4() {
        assert_eq!(DEPTH_LINEARIZE_STD430_SIZE, 16);
        assert_eq!(DEPTH_LINEARIZE_STD430_SIZE, VEC4_STRIDE);
    }

    #[test]
    fn std430_layout_packs_near_far_flag() {
        let p = DepthParams::new(0.5, 100.0, true);
        let bytes = p.to_std430();
        assert_eq!(bytes.len(), 16);
        let near = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let far = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let flag = f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let pad = f32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        assert!(approx(near, 0.5));
        assert!(approx(far, 100.0));
        assert!(approx(flag, 1.0));
        assert!(approx(pad, 0.0));
    }

    #[test]
    fn std430_flag_zero_when_not_reverse() {
        let p = DepthParams::new(0.5, 100.0, false);
        let bytes = p.to_std430();
        let flag = f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        assert!(approx(flag, 0.0));
    }

    #[test]
    fn gpu_storage_bytes_scales_and_clamps() {
        assert_eq!(gpu_storage_bytes(0), DEPTH_LINEARIZE_STD430_SIZE);
        assert_eq!(gpu_storage_bytes(1), DEPTH_LINEARIZE_STD430_SIZE);
        assert_eq!(gpu_storage_bytes(4), 64);
    }
}
