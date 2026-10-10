//! Deterministic fixed-point transform propagation.
//!
//! Enabled by the `determinism` feature. This is milestone **M5** of the
//! design doc (§12): a bit-exact propagation path for rollback netcode,
//! replays, and server-authoritative simulation.
//!
//! `f32` is *not* portable: fused-multiply-add, compiler reassociation, and
//! differing transcendental implementations make the same scene diverge
//! bit-for-bit across CPUs, OSes, and optimisation levels. That divergence is
//! fatal to lockstep/rollback schemes (GGPO/Quantum form) which assume *same
//! input → same state* on every peer.
//!
//! This module removes float entirely. Every value is a Q32.32 [`Fixed`]
//! integer, every operation is integer arithmetic, and the hierarchy sweep
//! runs in a fixed parent-before-child order with no parallel non-determinism.
//! The result is byte-identical on every platform, and — because the math is
//! pure integer with a fixed evaluation order — a second run on the same
//! inputs reproduces the first exactly (the "double-run equivalence" the
//! roadmap calls for). [`hash_globals`] folds a whole world into a 64-bit
//! digest for cross-peer desync detection.

use alloc::vec::Vec;

use prism_math::{Fixed, FxVec3, StateHasher};

use crate::hierarchy::{Hierarchy, HierarchyError, NodeId};

/// A deterministic fixed-point column-major 3×3 matrix.
///
/// Columns are the images of the basis vectors, matching the `f32`
/// [`prism_math::Mat3`] convention so the two paths compose identically.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FxMat3 {
    /// Image of the X basis vector.
    pub x_axis: FxVec3,
    /// Image of the Y basis vector.
    pub y_axis: FxVec3,
    /// Image of the Z basis vector.
    pub z_axis: FxVec3,
}

impl Default for FxMat3 {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl FxMat3 {
    /// The identity matrix.
    pub const IDENTITY: Self = Self {
        x_axis: FxVec3::X,
        y_axis: FxVec3::Y,
        z_axis: FxVec3::Z,
    };

    /// Build from explicit columns.
    #[inline]
    pub const fn from_cols(x_axis: FxVec3, y_axis: FxVec3, z_axis: FxVec3) -> Self {
        Self {
            x_axis,
            y_axis,
            z_axis,
        }
    }

    /// A diagonal (non-uniform) scale matrix.
    #[inline]
    pub fn from_scale(scale: FxVec3) -> Self {
        Self {
            x_axis: FxVec3::new(scale.x, Fixed::ZERO, Fixed::ZERO),
            y_axis: FxVec3::new(Fixed::ZERO, scale.y, Fixed::ZERO),
            z_axis: FxVec3::new(Fixed::ZERO, Fixed::ZERO, scale.z),
        }
    }

    /// Rotation about the X axis by `angle` radians, using the deterministic
    /// fixed-point [`Fixed::sin_cos`].
    #[inline]
    pub fn from_rotation_x(angle: Fixed) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            x_axis: FxVec3::X,
            y_axis: FxVec3::new(Fixed::ZERO, c, s),
            z_axis: FxVec3::new(Fixed::ZERO, -s, c),
        }
    }

    /// Rotation about the Y axis by `angle` radians.
    #[inline]
    pub fn from_rotation_y(angle: Fixed) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            x_axis: FxVec3::new(c, Fixed::ZERO, -s),
            y_axis: FxVec3::Y,
            z_axis: FxVec3::new(s, Fixed::ZERO, c),
        }
    }

    /// Rotation about the Z axis by `angle` radians.
    #[inline]
    pub fn from_rotation_z(angle: Fixed) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            x_axis: FxVec3::new(c, s, Fixed::ZERO),
            y_axis: FxVec3::new(-s, c, Fixed::ZERO),
            z_axis: FxVec3::Z,
        }
    }

    /// Apply this matrix to a vector: `x·x_axis + y·y_axis + z·z_axis`.
    #[inline]
    pub fn mul_vec3(self, v: FxVec3) -> FxVec3 {
        v.x * self.x_axis + v.y * self.y_axis + v.z * self.z_axis
    }

    /// Matrix product. `self * rhs` applies `rhs` first, then `self` — each
    /// column of the result is `self` applied to the corresponding column of
    /// `rhs`, matching the `f32` convention.
    #[inline]
    pub fn mul_mat3(self, rhs: Self) -> Self {
        Self {
            x_axis: self.mul_vec3(rhs.x_axis),
            y_axis: self.mul_vec3(rhs.y_axis),
            z_axis: self.mul_vec3(rhs.z_axis),
        }
    }

    /// The raw Q32.32 bits of all nine entries, column-major, for hashing and
    /// serialization.
    #[inline]
    pub fn to_bits(self) -> [i64; 9] {
        let [ax, ay, az] = self.x_axis.to_bits();
        let [bx, by, bz] = self.y_axis.to_bits();
        let [cx, cy, cz] = self.z_axis.to_bits();
        [ax, ay, az, bx, by, bz, cx, cy, cz]
    }
}

impl core::ops::Mul for FxMat3 {
    type Output = FxMat3;
    #[inline]
    fn mul(self, rhs: FxMat3) -> FxMat3 {
        self.mul_mat3(rhs)
    }
}

impl core::ops::Mul<FxVec3> for FxMat3 {
    type Output = FxVec3;
    #[inline]
    fn mul(self, rhs: FxVec3) -> FxVec3 {
        self.mul_vec3(rhs)
    }
}

/// A deterministic fixed-point affine transform: a [`FxMat3`] linear part plus
/// a [`FxVec3`] translation. This is the fixed-point analogue of
/// [`prism_math::Affine3`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FxAffine3 {
    /// The linear (rotation × scale, possibly sheared) part.
    pub matrix3: FxMat3,
    /// The translation.
    pub translation: FxVec3,
}

impl Default for FxAffine3 {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl FxAffine3 {
    /// The identity transform.
    pub const IDENTITY: Self = Self {
        matrix3: FxMat3::IDENTITY,
        translation: FxVec3::ZERO,
    };

    /// Build from a linear part and a translation.
    #[inline]
    pub const fn from_mat3_translation(matrix3: FxMat3, translation: FxVec3) -> Self {
        Self {
            matrix3,
            translation,
        }
    }

    /// Translation-only transform.
    #[inline]
    pub const fn from_translation(translation: FxVec3) -> Self {
        Self {
            matrix3: FxMat3::IDENTITY,
            translation,
        }
    }

    /// Scale-only transform.
    #[inline]
    pub fn from_scale(scale: FxVec3) -> Self {
        Self {
            matrix3: FxMat3::from_scale(scale),
            translation: FxVec3::ZERO,
        }
    }

    /// Rotation-only transform about the Z axis (the common 2.5D / top-down
    /// case); X and Y variants are available via [`FxMat3`].
    #[inline]
    pub fn from_rotation_z(angle: Fixed) -> Self {
        Self {
            matrix3: FxMat3::from_rotation_z(angle),
            translation: FxVec3::ZERO,
        }
    }

    /// A scale → rotate-Z → translate transform, the deterministic analogue of
    /// a TRS. Scale is applied first, then rotation, then translation, matching
    /// [`prism_math::Affine3::from_scale_rotation_translation`].
    #[inline]
    pub fn from_scale_rotation_z_translation(
        scale: FxVec3,
        angle: Fixed,
        translation: FxVec3,
    ) -> Self {
        let matrix3 = FxMat3::from_rotation_z(angle).mul_mat3(FxMat3::from_scale(scale));
        Self {
            matrix3,
            translation,
        }
    }

    /// Transform a point: `matrix3 · p + translation`.
    #[inline]
    pub fn transform_point3(self, p: FxVec3) -> FxVec3 {
        self.matrix3.mul_vec3(p) + self.translation
    }

    /// Transform a direction (ignores translation).
    #[inline]
    pub fn transform_vector3(self, v: FxVec3) -> FxVec3 {
        self.matrix3.mul_vec3(v)
    }

    /// Compose two transforms. `self * rhs` applies `rhs` first, then `self`,
    /// matching [`prism_math::Affine3`]'s `Mul` so the deterministic path
    /// mirrors the float path exactly in structure.
    #[inline]
    pub fn mul_affine(self, rhs: Self) -> Self {
        Self {
            matrix3: self.matrix3.mul_mat3(rhs.matrix3),
            translation: self.matrix3.mul_vec3(rhs.translation) + self.translation,
        }
    }

    /// The raw Q32.32 bits of the whole transform (nine matrix entries then
    /// three translation entries) for hashing and serialization.
    #[inline]
    pub fn to_bits(self) -> [i64; 12] {
        let m = self.matrix3.to_bits();
        let [tx, ty, tz] = self.translation.to_bits();
        [
            m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], tx, ty, tz,
        ]
    }
}

impl core::ops::Mul for FxAffine3 {
    type Output = FxAffine3;
    #[inline]
    fn mul(self, rhs: FxAffine3) -> FxAffine3 {
        self.mul_affine(rhs)
    }
}

/// Run a full **deterministic** propagation pass over `hierarchy`, writing a
/// fixed-point world transform for every node.
///
/// `locals[i]` / `globals[i]` are node `i`'s local / world transform. The
/// traversal is the fixed parent-before-child order from
/// [`Hierarchy::compute_order`] and the composition is pure integer
/// [`FxAffine3`] algebra, so the output is bit-identical across platforms and
/// reproduces exactly on a re-run.
///
/// # Errors
/// - [`HierarchyError::LengthMismatch`] if either buffer length differs from
///   the node count.
/// - [`HierarchyError::Cycle`] if the hierarchy is not a forest.
pub fn propagate_fixed(
    hierarchy: &Hierarchy,
    locals: &[FxAffine3],
    globals: &mut [FxAffine3],
) -> Result<(), HierarchyError> {
    if locals.len() != hierarchy.len() || globals.len() != hierarchy.len() {
        return Err(HierarchyError::LengthMismatch);
    }
    let order = hierarchy.compute_order()?;
    propagate_fixed_in_order(hierarchy, &order, locals, globals);
    Ok(())
}

/// Propagate fixed-point world transforms for the nodes in `order`
/// (parent-before-child).
///
/// Every referenced parent's world transform must already be valid in
/// `globals`.
///
/// # Panics
/// Panics if any id in `order`, or any parent reachable from it, is out of
/// bounds for `locals`/`globals`.
pub fn propagate_fixed_in_order(
    hierarchy: &Hierarchy,
    order: &[NodeId],
    locals: &[FxAffine3],
    globals: &mut [FxAffine3],
) {
    for &node in order {
        let local = locals[node.index()];
        let world = match hierarchy.parent(node) {
            None => local,
            Some(parent) => globals[parent.index()].mul_affine(local),
        };
        globals[node.index()] = world;
    }
}

/// Allocate an identity fixed-point world buffer sized for `hierarchy`.
#[inline]
pub fn identity_globals_fixed(hierarchy: &Hierarchy) -> Vec<FxAffine3> {
    let mut globals = Vec::with_capacity(hierarchy.len());
    globals.resize(hierarchy.len(), FxAffine3::IDENTITY);
    globals
}

/// Fold a whole set of world transforms into a 64-bit deterministic digest for
/// cross-peer desync detection.
///
/// Two peers that agree on every world transform produce the same digest; a
/// single differing bit anywhere changes it. The order of `globals` is part of
/// the hash, so peers must iterate the same node order (they do: it is the
/// hierarchy's stable creation order).
#[inline]
pub fn hash_globals(globals: &[FxAffine3]) -> u64 {
    let mut hasher = StateHasher::new();
    for g in globals {
        for bits in g.to_bits() {
            hasher.write_i64(bits);
        }
    }
    hasher.finish()
}
