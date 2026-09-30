//! Unit dual-quaternion algebra for scale-free rigid transforms and skinning
//! (design §16, §24).
//!
//! A rigid transform (rotation plus translation, no scale/shear) is stored as a
//! `DualQuat`: a pair of quaternions `real + dual*ε` where `real` carries the
//! rotation and `dual` encodes the translation. This is the representation
//! production skinning pipelines use for *dual-quaternion linear blending*
//! (`DLB` / `DQS`), because averaging dual quaternions preserves rigidity and
//! avoids the candy-wrapper collapse that plain matrix `LERP` produces.
//!
//! # Strict scope
//! This module is *self-contained* rigid-transform algebra. It deliberately does
//! **not** reuse [`crate::particle::quaternion_rotate`]'s `Quat` type: that
//! module owns pure orientation algebra with its own conventions, while this
//! module needs a private `Quat` paired into a `DualQuat`. The two never share a
//! type so their contracts stay independently verifiable. All vector math here
//! is hand-written.
//!
//! # No transcendental math
//! Every routine is a polynomial plus at most one `sqrt` (the norm used to
//! renormalize a `DualQuat` back onto the unit-rigid manifold). There is no
//! `sin`/`cos`/`acos` anywhere: rotations are supplied as ready-made unit
//! quaternions, so construction is pure `Hamilton`-product arithmetic.
//!
//! Degenerate inputs never produce a `NaN`: renormalizing a near-zero `DualQuat`
//! falls back to the identity rather than dividing by a near-zero magnitude.

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Magnitude below which a quaternion norm is treated as degenerate, and the
/// epsilon used for tolerant float comparisons instead of exact `==`/`!=`.
const CMP_EPS: f32 = 1e-6;

/// A quaternion `x*i + y*j + z*k + w`, private to the dual-quaternion contract.
///
/// This is intentionally distinct from [`crate::particle::quaternion_rotate`]'s
/// `Quat`: it exists only to be paired inside a `DualQuat`, so the two modules
/// keep separate, independently verified algebra contracts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    /// Coefficient of the `i` basis.
    pub x: f32,
    /// Coefficient of the `j` basis.
    pub y: f32,
    /// Coefficient of the `k` basis.
    pub z: f32,
    /// Scalar part.
    pub w: f32,
}

impl Quat {
    /// Builds a quaternion directly from its four components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// The identity rotation `(0, 0, 0, 1)`.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    }

    /// The Euclidean norm `sqrt(x² + y² + z² + w²)`.
    #[must_use]
    pub fn norm(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }

    /// The unit quaternion `self / norm`. A near-zero quaternion has no stable
    /// direction, so the identity is returned as a fallback.
    #[must_use]
    pub fn normalized(&self) -> Quat {
        let n = self.norm();
        if n < CMP_EPS {
            return Quat::identity();
        }
        let inv = 1.0 / n;
        self.scale(inv)
    }

    /// The conjugate `(-x, -y, -z, w)`. For a unit quaternion this is the
    /// inverse rotation.
    #[must_use]
    pub fn conjugate(&self) -> Quat {
        Quat {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: self.w,
        }
    }

    /// The four-component dot product.
    #[must_use]
    pub fn dot(&self, other: &Quat) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    /// Component-wise sum. Named `plus` rather than implementing `Add` because
    /// this is a building block for the blend accumulator, not a public
    /// arithmetic trait surface.
    #[must_use]
    pub fn plus(&self, other: &Quat) -> Quat {
        Quat {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }

    /// Scalar multiply. Named `scale` rather than implementing `Mul` because it
    /// is a component-wise scaling, not quaternion multiplication.
    #[must_use]
    pub fn scale(&self, factor: f32) -> Quat {
        Quat {
            x: self.x * factor,
            y: self.y * factor,
            z: self.z * factor,
            w: self.w * factor,
        }
    }

    /// The `Hamilton` product `self * other` composing two rotations (apply
    /// `other` first, then `self`). Named `hamilton` rather than `mul` because
    /// the product is non-commutative and is not a component-wise `Mul`.
    #[must_use]
    pub fn hamilton(&self, other: &Quat) -> Quat {
        let (x1, y1, z1, w1) = (self.x, self.y, self.z, self.w);
        let (x2, y2, z2, w2) = (other.x, other.y, other.z, other.w);
        Quat {
            x: w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            y: w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            z: w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w: w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        }
    }
}

/// Byte size of a `DualQuat` in `std430`: two `vec4<f32>` slots, one per
/// quaternion, with no padding between them.
pub const DUAL_QUAT_STD430_SIZE: usize = 2 * VEC4_STRIDE;

/// A rigid transform stored as a dual quaternion `real + dual*ε`.
///
/// `real` is the unit rotation quaternion; `dual` encodes the translation via
/// `dual = 0.5 * (t_quat ⊗ real)` where `t_quat = (t, 0)`. A unit `DualQuat`
/// (`real` normalized, `real·dual = 0`) represents exactly one rotation plus
/// one translation with no scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DualQuat {
    /// Rotation part (unit quaternion for a rigid transform).
    pub real: Quat,
    /// Translation-encoding part.
    pub dual: Quat,
}

impl DualQuat {
    /// Builds a `DualQuat` directly from its two quaternion parts.
    #[must_use]
    pub const fn new(real: Quat, dual: Quat) -> Self {
        Self { real, dual }
    }

    /// The identity transform: identity rotation, zero translation.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            real: Quat::identity(),
            dual: Quat::new(0.0, 0.0, 0.0, 0.0),
        }
    }

    /// Builds a `DualQuat` from a unit rotation quaternion and a translation.
    ///
    /// The rotation is renormalized so a slightly off-unit input still yields a
    /// rigid transform, and `dual = 0.5 * (t_quat ⊗ real)`.
    #[must_use]
    pub fn from_rotation_translation(rot: &Quat, t: [f32; 3]) -> DualQuat {
        let real = rot.normalized();
        let t_quat = Quat::new(t[0], t[1], t[2], 0.0);
        let dual = t_quat.hamilton(&real).scale(0.5);
        DualQuat { real, dual }
    }

    /// Recovers the `(rotation, translation)` pair encoded by this `DualQuat`.
    ///
    /// The translation is `t = 2 * (dual ⊗ real*)`, taking the vector part.
    #[must_use]
    pub fn to_rotation_translation(&self) -> (Quat, [f32; 3]) {
        let t_quat = self.dual.hamilton(&self.real.conjugate()).scale(2.0);
        (self.real, [t_quat.x, t_quat.y, t_quat.z])
    }

    /// Renormalizes onto the unit-rigid manifold by dividing both parts by the
    /// `real` norm, preserving the rotation/translation the `DualQuat` encodes.
    ///
    /// A near-zero `real` has no stable rotation, so the identity is returned.
    #[must_use]
    pub fn normalized(&self) -> DualQuat {
        let n = self.real.norm();
        if n < CMP_EPS {
            return DualQuat::identity();
        }
        let inv = 1.0 / n;
        DualQuat {
            real: self.real.scale(inv),
            dual: self.dual.scale(inv),
        }
    }

    /// Applies this rigid transform to a point: rotate, then translate.
    #[must_use]
    pub fn transform_point(&self, p: [f32; 3]) -> [f32; 3] {
        let unit = self.normalized();
        let (rot, t) = unit.to_rotation_translation();
        let pv = Quat::new(p[0], p[1], p[2], 0.0);
        let rotated = rot.hamilton(&pv).hamilton(&rot.conjugate());
        [rotated.x + t[0], rotated.y + t[1], rotated.z + t[2]]
    }

    /// Dual-quaternion linear blend (`DLB` / `DQS`).
    ///
    /// Accumulates `sum(weight_i * dq_i)`, first flipping the sign of any
    /// `DualQuat` whose `real` points away from the first entry's `real`
    /// (negative dot), so antipodal-but-equal rotations reinforce instead of
    /// cancelling. The weighted sum is then renormalized back onto the
    /// unit-rigid manifold. An empty input yields the identity.
    #[must_use]
    pub fn blend(dqs: &[DualQuat], weights: &[f32]) -> DualQuat {
        let Some(first) = dqs.first() else {
            return DualQuat::identity();
        };
        let reference = first.real;
        let mut real_acc = Quat::new(0.0, 0.0, 0.0, 0.0);
        let mut dual_acc = Quat::new(0.0, 0.0, 0.0, 0.0);
        for (dq, &weight) in dqs.iter().zip(weights.iter()) {
            let signed = if dq.real.dot(&reference) < 0.0 {
                -weight
            } else {
                weight
            };
            real_acc = real_acc.plus(&dq.real.scale(signed));
            dual_acc = dual_acc.plus(&dq.dual.scale(signed));
        }
        DualQuat {
            real: real_acc,
            dual: dual_acc,
        }
        .normalized()
    }

    /// Serializes to the `std430` byte image (little-endian): `real` then
    /// `dual`, each four consecutive `f32` scalar slots filling one `vec4` slot.
    #[must_use]
    pub fn to_std430(&self) -> [u8; DUAL_QUAT_STD430_SIZE] {
        let mut bytes = [0u8; DUAL_QUAT_STD430_SIZE];
        let words = [
            self.real.x,
            self.real.y,
            self.real.z,
            self.real.w,
            self.dual.x,
            self.dual.y,
            self.dual.z,
            self.dual.w,
        ];
        for (slot, value) in bytes.chunks_exact_mut(4).zip(words.iter()) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

/// Total byte size of a storage buffer holding `count` dual quaternions.
///
/// Reserved through [`storage_bytes`], so an empty set still yields a valid,
/// non-zero-sized `GPU` storage binding.
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(DUAL_QUAT_STD430_SIZE, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;
    /// `sqrt(1/2)`, the half-angle sine/cosine of a 90° rotation, derived with
    /// the allowed `sqrt` rather than a transcendental call or approximated
    /// literal (which `clippy::approx_constant` rejects).
    fn sqrt_half() -> f32 {
        0.5_f32.sqrt()
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    fn point_close(a: [f32; 3], b: [f32; 3]) -> bool {
        close(a[0], b[0]) && close(a[1], b[1]) && close(a[2], b[2])
    }

    fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Unit rotation of 90° about the `+z` axis.
    fn rot_z90() -> Quat {
        Quat::new(0.0, 0.0, sqrt_half(), sqrt_half())
    }

    #[test]
    fn quat_identity_values() {
        let q = Quat::identity();
        assert!(close(q.x, 0.0) && close(q.y, 0.0) && close(q.z, 0.0) && close(q.w, 1.0));
    }

    #[test]
    fn quat_norm_of_identity_is_one() {
        assert!(close(Quat::identity().norm(), 1.0));
    }

    #[test]
    fn quat_normalized_is_unit() {
        let q = Quat::new(3.0, 0.0, 4.0, 0.0);
        assert!(close(q.normalized().norm(), 1.0));
    }

    #[test]
    fn quat_normalized_zero_falls_back_to_identity() {
        let q = Quat::new(0.0, 0.0, 0.0, 0.0);
        assert_eq!(q.normalized(), Quat::identity());
    }

    #[test]
    fn quat_conjugate_negates_vector_part() {
        let c = Quat::new(1.0, -2.0, 3.0, 4.0).conjugate();
        assert!(close(c.x, -1.0) && close(c.y, 2.0) && close(c.z, -3.0) && close(c.w, 4.0));
    }

    #[test]
    fn quat_hamilton_identity_is_noop() {
        let q = rot_z90();
        let r = q.hamilton(&Quat::identity());
        assert!(close(r.x, q.x) && close(r.y, q.y) && close(r.z, q.z) && close(r.w, q.w));
    }

    #[test]
    fn quat_hamilton_composes_rotations() {
        // 90° + 90° about z equals 180° about z: (0,0,1,0).
        let composed = rot_z90().hamilton(&rot_z90());
        assert!(
            close(composed.x, 0.0)
                && close(composed.y, 0.0)
                && close(composed.z, 1.0)
                && close(composed.w, 0.0)
        );
    }

    #[test]
    fn dual_quat_identity_parts() {
        let dq = DualQuat::identity();
        assert_eq!(dq.real, Quat::identity());
        assert_eq!(dq.dual, Quat::new(0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn identity_leaves_point_unchanged() {
        let dq = DualQuat::identity();
        assert!(point_close(
            dq.transform_point([1.0, 2.0, 3.0]),
            [1.0, 2.0, 3.0]
        ));
    }

    #[test]
    fn pure_rotation_matches_quaternion_rotation() {
        // 90° about z sends (1,0,0) to (0,1,0).
        let dq = DualQuat::from_rotation_translation(&rot_z90(), [0.0, 0.0, 0.0]);
        assert!(point_close(
            dq.transform_point([1.0, 0.0, 0.0]),
            [0.0, 1.0, 0.0]
        ));
    }

    #[test]
    fn pure_rotation_second_axis() {
        // 90° about z sends (0,1,0) to (-1,0,0).
        let dq = DualQuat::from_rotation_translation(&rot_z90(), [0.0, 0.0, 0.0]);
        assert!(point_close(
            dq.transform_point([0.0, 1.0, 0.0]),
            [-1.0, 0.0, 0.0]
        ));
    }

    #[test]
    fn pure_translation_moves_point() {
        let dq = DualQuat::from_rotation_translation(&Quat::identity(), [5.0, -2.0, 7.0]);
        assert!(point_close(
            dq.transform_point([1.0, 1.0, 1.0]),
            [6.0, -1.0, 8.0]
        ));
    }

    #[test]
    fn combined_rotate_then_translate() {
        // Rotate (1,0,0) -> (0,1,0), then translate by (0,0,3).
        let dq = DualQuat::from_rotation_translation(&rot_z90(), [0.0, 0.0, 3.0]);
        assert!(point_close(
            dq.transform_point([1.0, 0.0, 0.0]),
            [0.0, 1.0, 3.0]
        ));
    }

    #[test]
    fn roundtrip_recovers_rotation() {
        let dq = DualQuat::from_rotation_translation(&rot_z90(), [0.0, 0.0, 0.0]);
        let (rot, t) = dq.to_rotation_translation();
        assert!(close(rot.x, 0.0) && close(rot.y, 0.0));
        assert!(close(rot.z, sqrt_half()) && close(rot.w, sqrt_half()));
        assert!(point_close(t, [0.0, 0.0, 0.0]));
    }

    #[test]
    fn roundtrip_recovers_translation() {
        let t_in = [1.5, -3.25, 8.0];
        let dq = DualQuat::from_rotation_translation(&Quat::identity(), t_in);
        let (_rot, t_out) = dq.to_rotation_translation();
        assert!(point_close(t_out, t_in));
    }

    #[test]
    fn roundtrip_recovers_combined() {
        let t_in = [2.0, 4.0, -6.0];
        let dq = DualQuat::from_rotation_translation(&rot_z90(), t_in);
        let (rot, t_out) = dq.to_rotation_translation();
        assert!(close(rot.z, sqrt_half()) && close(rot.w, sqrt_half()));
        assert!(point_close(t_out, t_in));
    }

    #[test]
    fn normalized_keeps_real_unit() {
        let dq = DualQuat::new(rot_z90().scale(2.0), rot_z90().scale(2.0));
        assert!(close(dq.normalized().real.norm(), 1.0));
    }

    #[test]
    fn normalized_zero_real_falls_back_to_identity() {
        let dq = DualQuat::new(Quat::new(0.0, 0.0, 0.0, 0.0), Quat::new(1.0, 1.0, 1.0, 1.0));
        assert_eq!(dq.normalized(), DualQuat::identity());
    }

    #[test]
    fn single_weight_blend_equals_that_dq() {
        let dq = DualQuat::from_rotation_translation(&rot_z90(), [1.0, 2.0, 3.0]);
        let blended = DualQuat::blend(&[dq], &[1.0]);
        assert!(point_close(
            blended.transform_point([1.0, 0.0, 0.0]),
            dq.transform_point([1.0, 0.0, 0.0])
        ));
    }

    #[test]
    fn two_identical_dq_blend_equals_self() {
        let dq = DualQuat::from_rotation_translation(&rot_z90(), [0.5, -1.0, 2.0]);
        let blended = DualQuat::blend(&[dq, dq], &[0.5, 0.5]);
        let p = [2.0, -1.0, 0.5];
        assert!(point_close(
            blended.transform_point(p),
            dq.transform_point(p)
        ));
    }

    #[test]
    fn blend_result_is_unit_rigid() {
        let a = DualQuat::from_rotation_translation(&rot_z90(), [1.0, 0.0, 0.0]);
        let b = DualQuat::from_rotation_translation(&Quat::identity(), [0.0, 1.0, 0.0]);
        let blended = DualQuat::blend(&[a, b], &[0.5, 0.5]);
        assert!(close(blended.real.norm(), 1.0));
    }

    #[test]
    fn sign_alignment_avoids_cancellation() {
        // rot and its antipode encode the same rotation; naive averaging cancels
        // the real part. Sign alignment must recover the shared rotation.
        let dq = DualQuat::from_rotation_translation(&rot_z90(), [0.0, 0.0, 0.0]);
        let antipode = DualQuat::new(dq.real.scale(-1.0), dq.dual.scale(-1.0));
        let blended = DualQuat::blend(&[dq, antipode], &[0.5, 0.5]);
        assert!(close(blended.real.norm(), 1.0));
        assert!(point_close(
            blended.transform_point([1.0, 0.0, 0.0]),
            [0.0, 1.0, 0.0]
        ));
    }

    #[test]
    fn empty_blend_is_identity() {
        assert_eq!(DualQuat::blend(&[], &[]), DualQuat::identity());
    }

    #[test]
    fn rigid_transform_preserves_distance() {
        let dq = DualQuat::from_rotation_translation(&rot_z90(), [3.0, -4.0, 5.0]);
        let a = [1.0, 2.0, 3.0];
        let b = [-2.0, 0.5, 4.0];
        let before = dist(a, b);
        let after = dist(dq.transform_point(a), dq.transform_point(b));
        assert!(close(before, after));
    }

    #[test]
    fn blend_transform_preserves_distance() {
        let x = DualQuat::from_rotation_translation(&rot_z90(), [1.0, 0.0, 0.0]);
        let y = DualQuat::from_rotation_translation(&Quat::identity(), [0.0, 2.0, 0.0]);
        let blended = DualQuat::blend(&[x, y], &[0.3, 0.7]);
        let a = [1.0, 1.0, 1.0];
        let b = [4.0, -1.0, 2.0];
        assert!(close(
            dist(a, b),
            dist(blended.transform_point(a), blended.transform_point(b))
        ));
    }

    #[test]
    fn std430_size_is_two_vec4() {
        assert_eq!(DUAL_QUAT_STD430_SIZE, 32);
        assert_eq!(DUAL_QUAT_STD430_SIZE, 2 * VEC4_STRIDE);
    }

    #[test]
    fn to_std430_layout_matches_parts() {
        let dq = DualQuat::new(Quat::new(1.0, 2.0, 3.0, 4.0), Quat::new(5.0, 6.0, 7.0, 8.0));
        let bytes = dq.to_std430();
        assert_eq!(bytes.len(), DUAL_QUAT_STD430_SIZE);
        let expected = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for (slot, value) in bytes.chunks_exact(4).zip(expected.iter()) {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(slot);
            assert!(close(f32::from_le_bytes(buf), *value));
        }
    }

    #[test]
    fn gpu_storage_bytes_scales_with_count() {
        assert_eq!(gpu_storage_bytes(4), 4 * DUAL_QUAT_STD430_SIZE);
    }

    #[test]
    fn gpu_storage_bytes_empty_reserves_one() {
        assert_eq!(gpu_storage_bytes(0), DUAL_QUAT_STD430_SIZE);
    }
}
