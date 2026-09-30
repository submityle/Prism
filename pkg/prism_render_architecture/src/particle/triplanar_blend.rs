//! Triplanar mapping blend: the `CPU`-verifiable contract that turns a world
//! surface normal into three projection-plane weights, hands back the per-plane
//! `UV` each projection samples, and blends the three texture samples into the
//! final shaded value (design §16-§21).
//!
//! # Why triplanar
//!
//! A single planar `UV` set stretches badly on geometry that faces away from its
//! projection axis: a rock textured with a top-down `UV` smears into vertical
//! streaks on its cliffs, and procedural / world-space particles (impact decals,
//! terrain-conforming sprites, voxel debris) have no authored `UV` at all.
//! Triplanar mapping sidesteps both problems by projecting the *same* texture
//! three times — once down each world axis — and cross-fading between the three
//! projections by how strongly the surface faces each axis. The result reads as
//! seamless material on arbitrary orientation with no authored coordinates, the
//! standard trick in `Unreal`, `Unity`, and every terrain / voxel `AAA` stack.
//!
//! Three pieces, mirroring the production shader:
//!
//! 1. [`TriplanarWeights::from_normal`] — the *blend weights*. Take the absolute
//!    value of each world-normal component (the surface faces `+X` and `-X` the
//!    same), sharpen each by a `sharpness_exp` power so the dominant axis wins,
//!    then normalize so the three weights sum to one. The sharpen is an
//!    **integer-exponent repeated multiply**, never `powf`, so the reference
//!    reproduces bit for bit on a future `GPU` kernel.
//! 2. [`plane_uv`] — the *per-plane coordinates*. Each projection drops the axis
//!    it looks down and keeps the other two world components as its `UV` (see the
//!    method's own documented axis convention).
//! 3. [`TriplanarWeights::blend_scalar`] / [`TriplanarWeights::blend_vec3`] — the
//!    *composite*. A plain weighted sum of the three plane samples; because the
//!    weights already sum to one, the blend is an exact convex combination.
//!
//! # What this module deliberately does *not* do
//!
//! It samples nothing itself: the caller feeds in the three already-fetched
//! texture values (scalar or `vec3`), keeping this contract free of any texture
//! backend. It does not scroll, tile, or animate the `UV` (`uv_animation` owns
//! that), and it does not offset the `UV` for relief (`parallax_offset`). It
//! imports no sibling particle module except the shared `std430` layout
//! primitives.
//!
//! # Determinism
//!
//! The only floating-point primitives beyond ordinary arithmetic are integer
//! exponent multiplies and one reciprocal for the weight normalization. There
//! are no `sin` / `cos` / `tan` / `exp` / `ln` / `pow` calls anywhere. Bare
//! `==` / `!=` on `f32` is avoided; the degenerate-normal guard compares a
//! magnitude against [`CMP_EPS`].

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Absolute tolerance for the degenerate-normal guard, standing in for the
/// forbidden bare `==` / `!=` on `f32`.
pub const CMP_EPS: f32 = 1.0e-6;

/// Byte size of the `std430` packing of [`TriplanarWeights`]: three `f32`
/// weights padded up to one 16-byte `vec4` slot (the natural `std430` base
/// alignment for a `vec3`-shaped block, with a four-byte zero tail).
pub const TRIPLANAR_STD430_SIZE: usize = VEC4_STRIDE;

/// Raises `base` to the non-negative integer power `exp` by repeated
/// multiplication.
///
/// This is the transcendental-free stand-in for `powf`: `exp == 0` yields `1.0`
/// for any base (so a zero sharpness collapses every axis weight to one), and
/// larger `exp` sharpens the fall-off toward the dominant axis.
fn int_pow(base: f32, exp: u32) -> f32 {
    let mut acc = 1.0;
    for _ in 0..exp {
        acc *= base;
    }
    acc
}

/// The three normalized triplanar projection-plane weights.
///
/// `x` weights the `X`-axis projection (the plane whose normal is world `X`),
/// `y` the `Y`-axis projection, and `z` the `Z`-axis projection. The three are
/// non-negative and sum to one, so blending with them is an exact convex
/// combination.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriplanarWeights {
    /// Weight of the `X`-axis projection (sampling the world `YZ` plane).
    pub x: f32,
    /// Weight of the `Y`-axis projection (sampling the world `XZ` plane).
    pub y: f32,
    /// Weight of the `Z`-axis projection (sampling the world `XY` plane).
    pub z: f32,
}

impl TriplanarWeights {
    /// Builds the blend weights from a world-space surface `normal`.
    ///
    /// Each component's absolute value is sharpened by the integer power
    /// `sharpness_exp` (repeated multiply, never `powf`) and the three sharpened
    /// magnitudes are normalized to sum to one. `sharpness_exp == 0` makes every
    /// sharpened magnitude `1`, giving the uniform `1/3, 1/3, 1/3`. A degenerate
    /// (zero-length, or fully cancelled) normal whose sharpened sum falls at or
    /// below [`CMP_EPS`] also falls back to the uniform `1/3` split rather than
    /// dividing by zero.
    #[must_use]
    pub fn from_normal(normal: [f32; 3], sharpness_exp: u32) -> Self {
        let ax = int_pow(normal[0].abs(), sharpness_exp);
        let ay = int_pow(normal[1].abs(), sharpness_exp);
        let az = int_pow(normal[2].abs(), sharpness_exp);
        let sum = ax + ay + az;
        if sum <= CMP_EPS {
            let third = 1.0 / 3.0;
            return Self {
                x: third,
                y: third,
                z: third,
            };
        }
        let inv = 1.0 / sum;
        Self {
            x: ax * inv,
            y: ay * inv,
            z: az * inv,
        }
    }

    /// Returns the three weights as `[x, y, z]`.
    #[must_use]
    pub fn as_array(&self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    /// Returns the sum of the three weights (`≈ 1` for any well-formed set).
    #[must_use]
    pub fn sum(&self) -> f32 {
        self.x + self.y + self.z
    }

    /// Blends three scalar plane samples into one value.
    ///
    /// `sx`, `sy`, `sz` are the samples fetched with the `X`-, `Y`-, and
    /// `Z`-plane `UV` respectively (see [`plane_uv`]). The result is their
    /// convex combination under these weights.
    #[must_use]
    pub fn blend_scalar(&self, sx: f32, sy: f32, sz: f32) -> f32 {
        sx * self.x + sy * self.y + sz * self.z
    }

    /// Blends three `vec3` plane samples component-wise into one `vec3`.
    ///
    /// Each output channel is the weighted sum of that channel across the three
    /// plane samples, matching [`Self::blend_scalar`] applied per component.
    #[must_use]
    pub fn blend_vec3(&self, sx: [f32; 3], sy: [f32; 3], sz: [f32; 3]) -> [f32; 3] {
        [
            self.blend_scalar(sx[0], sy[0], sz[0]),
            self.blend_scalar(sx[1], sy[1], sz[1]),
            self.blend_scalar(sx[2], sy[2], sz[2]),
        ]
    }

    /// Packs the three weights into their `std430` byte layout.
    ///
    /// Bytes `0..12` hold `x`, `y`, `z` as little-endian `f32`; bytes `12..16`
    /// are the zero pad that rounds the block up to one `vec4` slot.
    #[must_use]
    pub fn to_std430(&self) -> [u8; TRIPLANAR_STD430_SIZE] {
        let mut out = [0_u8; TRIPLANAR_STD430_SIZE];
        out[0..4].copy_from_slice(&self.x.to_le_bytes());
        out[4..8].copy_from_slice(&self.y.to_le_bytes());
        out[8..12].copy_from_slice(&self.z.to_le_bytes());
        out
    }
}

/// The per-plane `UV` each triplanar projection samples, given a `world`
/// position.
///
/// Axis convention (each plane drops the axis it looks down and keeps the other
/// two world components):
///
/// * `[0]` — `X`-plane `UV = (world.z, world.y)`.
/// * `[1]` — `Y`-plane `UV = (world.x, world.z)`.
/// * `[2]` — `Z`-plane `UV = (world.x, world.y)`.
#[must_use]
pub fn plane_uv(world: [f32; 3]) -> [[f32; 2]; 3] {
    [
        [world[2], world[1]],
        [world[0], world[2]],
        [world[0], world[1]],
    ]
}

/// Total `std430` byte size of a storage buffer holding `count` packed
/// [`TriplanarWeights`] blocks, clamped up to a single element for a valid
/// `WebGPU` binding.
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(TRIPLANAR_STD430_SIZE, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1.0e-5
    }

    #[test]
    fn weights_sum_to_one_for_arbitrary_normal() {
        let w = TriplanarWeights::from_normal([0.3, -0.7, 0.5], 4);
        assert!(approx(w.sum(), 1.0));
    }

    #[test]
    fn weights_sum_to_one_across_many_normals() {
        let normals = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.3, 0.6, -0.2],
            [-0.9, -0.1, 0.4],
        ];
        for n in normals {
            for exp in [1_u32, 2, 3, 8] {
                let w = TriplanarWeights::from_normal(n, exp);
                assert!(approx(w.sum(), 1.0));
            }
        }
    }

    #[test]
    fn positive_x_axis_gives_full_x_weight() {
        let w = TriplanarWeights::from_normal([1.0, 0.0, 0.0], 4);
        assert!(approx(w.x, 1.0));
        assert!(approx(w.y, 0.0));
        assert!(approx(w.z, 0.0));
    }

    #[test]
    fn negative_x_axis_gives_full_x_weight() {
        // |n| makes -X and +X identical.
        let w = TriplanarWeights::from_normal([-1.0, 0.0, 0.0], 4);
        assert!(approx(w.x, 1.0));
        assert!(approx(w.y, 0.0));
        assert!(approx(w.z, 0.0));
    }

    #[test]
    fn positive_y_axis_gives_full_y_weight() {
        let w = TriplanarWeights::from_normal([0.0, 1.0, 0.0], 4);
        assert!(approx(w.y, 1.0));
        assert!(approx(w.x, 0.0));
        assert!(approx(w.z, 0.0));
    }

    #[test]
    fn positive_z_axis_gives_full_z_weight() {
        let w = TriplanarWeights::from_normal([0.0, 0.0, -1.0], 4);
        assert!(approx(w.z, 1.0));
        assert!(approx(w.x, 0.0));
        assert!(approx(w.y, 0.0));
    }

    #[test]
    fn diagonal_45_degrees_in_xy_is_symmetric() {
        let s = core::f32::consts::FRAC_1_SQRT_2;
        let w = TriplanarWeights::from_normal([s, s, 0.0], 3);
        assert!(approx(w.x, w.y));
        assert!(approx(w.z, 0.0));
        assert!(approx(w.x, 0.5));
        assert!(approx(w.y, 0.5));
    }

    #[test]
    fn full_diagonal_splits_evenly() {
        let s = 1.0 / (3.0_f32).sqrt();
        let w = TriplanarWeights::from_normal([s, s, s], 5);
        let third = 1.0 / 3.0;
        assert!(approx(w.x, third));
        assert!(approx(w.y, third));
        assert!(approx(w.z, third));
    }

    #[test]
    fn larger_sharpness_biases_toward_dominant_axis() {
        // X dominates; a higher exponent pushes more weight onto X.
        let n = [0.8, 0.5, 0.3];
        let soft = TriplanarWeights::from_normal(n, 1);
        let hard = TriplanarWeights::from_normal(n, 8);
        assert!(hard.x > soft.x);
        assert!(hard.y < soft.y);
        assert!(hard.z < soft.z);
    }

    #[test]
    fn sharpness_zero_is_uniform() {
        let w = TriplanarWeights::from_normal([0.8, 0.5, 0.3], 0);
        let third = 1.0 / 3.0;
        assert!(approx(w.x, third));
        assert!(approx(w.y, third));
        assert!(approx(w.z, third));
    }

    #[test]
    fn sharpness_zero_uniform_even_with_zero_components() {
        // exp == 0 makes every axis weight 1 regardless of the normal.
        let w = TriplanarWeights::from_normal([0.0, 0.0, 0.0], 0);
        let third = 1.0 / 3.0;
        assert!(approx(w.x, third));
        assert!(approx(w.y, third));
        assert!(approx(w.z, third));
    }

    #[test]
    fn zero_normal_degenerates_to_thirds() {
        let w = TriplanarWeights::from_normal([0.0, 0.0, 0.0], 4);
        let third = 1.0 / 3.0;
        assert!(approx(w.x, third));
        assert!(approx(w.y, third));
        assert!(approx(w.z, third));
    }

    #[test]
    fn weights_are_non_negative() {
        let w = TriplanarWeights::from_normal([-0.4, 0.9, -0.2], 6);
        assert!(w.x >= 0.0);
        assert!(w.y >= 0.0);
        assert!(w.z >= 0.0);
    }

    #[test]
    fn plane_uv_follows_axis_convention() {
        let uv = plane_uv([1.0, 2.0, 3.0]);
        // X plane -> (z, y)
        assert!(approx(uv[0][0], 3.0));
        assert!(approx(uv[0][1], 2.0));
        // Y plane -> (x, z)
        assert!(approx(uv[1][0], 1.0));
        assert!(approx(uv[1][1], 3.0));
        // Z plane -> (x, y)
        assert!(approx(uv[2][0], 1.0));
        assert!(approx(uv[2][1], 2.0));
    }

    #[test]
    fn plane_uv_negative_world_position() {
        let uv = plane_uv([-2.0, -4.0, -6.0]);
        assert!(approx(uv[0][0], -6.0));
        assert!(approx(uv[0][1], -4.0));
        assert!(approx(uv[1][0], -2.0));
        assert!(approx(uv[1][1], -6.0));
        assert!(approx(uv[2][0], -2.0));
        assert!(approx(uv[2][1], -4.0));
    }

    #[test]
    fn blend_scalar_is_weighted_sum() {
        let w = TriplanarWeights::from_normal([1.0, 0.0, 0.0], 4);
        // Full X weight -> picks the X sample.
        assert!(approx(w.blend_scalar(7.0, 2.0, 5.0), 7.0));
        let uniform = TriplanarWeights::from_normal([1.0, 1.0, 1.0], 0);
        // Uniform -> average.
        assert!(approx(uniform.blend_scalar(3.0, 6.0, 9.0), 6.0));
    }

    #[test]
    fn blend_scalar_manual_convex_combination() {
        let w = TriplanarWeights {
            x: 0.2,
            y: 0.3,
            z: 0.5,
        };
        let expected = 10.0 * 0.2 + 20.0 * 0.3 + 40.0 * 0.5;
        assert!(approx(w.blend_scalar(10.0, 20.0, 40.0), expected));
    }

    #[test]
    fn blend_vec3_is_component_wise() {
        let w = TriplanarWeights {
            x: 0.5,
            y: 0.25,
            z: 0.25,
        };
        let out = w.blend_vec3([1.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 8.0]);
        assert!(approx(out[0], 0.5));
        assert!(approx(out[1], 1.0));
        assert!(approx(out[2], 2.0));
    }

    #[test]
    fn blend_vec3_matches_per_channel_scalar() {
        let w = TriplanarWeights::from_normal([0.6, 0.3, 0.7], 3);
        let sx = [1.0, 2.0, 3.0];
        let sy = [4.0, 5.0, 6.0];
        let sz = [7.0, 8.0, 9.0];
        let out = w.blend_vec3(sx, sy, sz);
        for (c, &got) in out.iter().enumerate() {
            assert!(approx(got, w.blend_scalar(sx[c], sy[c], sz[c])));
        }
    }

    #[test]
    fn as_array_matches_fields() {
        let w = TriplanarWeights {
            x: 0.1,
            y: 0.2,
            z: 0.7,
        };
        assert_eq!(w.as_array(), [0.1, 0.2, 0.7]);
    }

    #[test]
    fn std430_size_and_storage_bytes() {
        assert_eq!(TRIPLANAR_STD430_SIZE, 16);
        assert_eq!(gpu_storage_bytes(0), TRIPLANAR_STD430_SIZE);
        assert_eq!(gpu_storage_bytes(1), TRIPLANAR_STD430_SIZE);
        assert_eq!(gpu_storage_bytes(4), 64);
    }

    #[test]
    fn std430_roundtrip_fields_and_zero_pad() {
        let w = TriplanarWeights {
            x: 0.25,
            y: 0.5,
            z: 0.125,
        };
        let bytes = w.to_std430();
        let x = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let y = f32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let z = f32::from_le_bytes(bytes[8..12].try_into().unwrap());
        assert!(approx(x, 0.25));
        assert!(approx(y, 0.5));
        assert!(approx(z, 0.125));
        assert_eq!(&bytes[12..16], &[0, 0, 0, 0]);
    }

    #[test]
    fn sharpness_two_matches_squared_magnitudes() {
        // Independent check of the integer power against hand arithmetic.
        let n = [0.6_f32, 0.8, 0.0];
        let w = TriplanarWeights::from_normal(n, 2);
        let ax = 0.6_f32 * 0.6;
        let ay = 0.8_f32 * 0.8;
        let sum = ax + ay;
        assert!(approx(w.x, ax / sum));
        assert!(approx(w.y, ay / sum));
        assert!(approx(w.z, 0.0));
    }
}
