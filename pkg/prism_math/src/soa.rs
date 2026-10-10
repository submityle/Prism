//! Structure-of-arrays (`SoA`) vector batches.
//!
//! An array-of-structures (`AoS`) layout stores `[Vec3, Vec3, ...]` with the
//! `x`, `y`, and `z` of each element interleaved. A structure-of-arrays
//! (`SoA`) layout instead keeps three contiguous component streams
//! (`xs`, `ys`, `zs`). For wide, data-parallel kernels (particles, skinning,
//! culling) `SoA` is the friendlier shape: each component stream is unit-stride
//! and vectorizes without gather/scatter.
//!
//! [`SoaVec3`] owns its three streams and offers real batch operations
//! (component add/scale, dot, length, normalize, and affine/matrix transform).
//! All operations keep the three streams the same length, which is this type's
//! core invariant.

use crate::affine::Affine3;
use crate::float::f32 as mf;
use crate::mat::Mat4;
use crate::vec::Vec3;
use alloc::vec::Vec;

/// A batch of 3D vectors stored structure-of-arrays: three equal-length
/// component streams.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SoaVec3 {
    xs: Vec<f32>,
    ys: Vec<f32>,
    zs: Vec<f32>,
}

impl SoaVec3 {
    /// An empty batch.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            xs: Vec::new(),
            ys: Vec::new(),
            zs: Vec::new(),
        }
    }

    /// An empty batch with capacity reserved for `n` vectors.
    #[inline]
    #[must_use]
    pub fn with_capacity(n: usize) -> Self {
        Self {
            xs: Vec::with_capacity(n),
            ys: Vec::with_capacity(n),
            zs: Vec::with_capacity(n),
        }
    }

    /// Build from an array-of-structures slice.
    #[inline]
    #[must_use]
    pub fn from_aos(src: &[Vec3]) -> Self {
        let mut out = Self::with_capacity(src.len());
        for v in src {
            out.push(*v);
        }
        out
    }

    /// Number of vectors in the batch.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.xs.len()
    }

    /// True if the batch holds no vectors.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.xs.is_empty()
    }

    /// The `x` component stream.
    #[inline]
    #[must_use]
    pub fn xs(&self) -> &[f32] {
        &self.xs
    }
    /// The `y` component stream.
    #[inline]
    #[must_use]
    pub fn ys(&self) -> &[f32] {
        &self.ys
    }
    /// The `z` component stream.
    #[inline]
    #[must_use]
    pub fn zs(&self) -> &[f32] {
        &self.zs
    }

    /// Append one vector.
    #[inline]
    pub fn push(&mut self, v: Vec3) {
        self.xs.push(v.x);
        self.ys.push(v.y);
        self.zs.push(v.z);
    }

    /// Read element `i`.
    ///
    /// # Panics
    /// Panics if `i` is out of bounds.
    #[inline]
    #[must_use]
    pub fn get(&self, i: usize) -> Vec3 {
        Vec3::new(self.xs[i], self.ys[i], self.zs[i])
    }

    /// Write element `i`.
    ///
    /// # Panics
    /// Panics if `i` is out of bounds.
    #[inline]
    pub fn set(&mut self, i: usize, v: Vec3) {
        self.xs[i] = v.x;
        self.ys[i] = v.y;
        self.zs[i] = v.z;
    }

    /// Materialize the batch back into an array-of-structures vector.
    #[inline]
    #[must_use]
    pub fn to_aos(&self) -> Vec<Vec3> {
        let mut out = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            out.push(self.get(i));
        }
        out
    }

    /// Iterate over the batch as [`Vec3`] values.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = Vec3> + '_ {
        (0..self.len()).map(move |i| self.get(i))
    }

    /// Add `rhs` to every element in place.
    ///
    /// # Panics
    /// Panics if the batches differ in length.
    #[inline]
    pub fn add_assign_batch(&mut self, rhs: &Self) {
        assert_eq!(
            self.len(),
            rhs.len(),
            "SoaVec3::add_assign_batch length mismatch"
        );
        for i in 0..self.len() {
            self.xs[i] += rhs.xs[i];
            self.ys[i] += rhs.ys[i];
            self.zs[i] += rhs.zs[i];
        }
    }

    /// Add a single vector to every element in place (broadcast).
    #[inline]
    pub fn add_broadcast(&mut self, v: Vec3) {
        for i in 0..self.len() {
            self.xs[i] += v.x;
            self.ys[i] += v.y;
            self.zs[i] += v.z;
        }
    }

    /// Scale every element by a scalar in place.
    #[inline]
    pub fn scale(&mut self, s: f32) {
        for i in 0..self.len() {
            self.xs[i] *= s;
            self.ys[i] *= s;
            self.zs[i] *= s;
        }
    }

    /// Per-element dot product against `rhs`, written to `out`.
    ///
    /// # Panics
    /// Panics if `self`, `rhs`, and `out` are not all the same length.
    #[inline]
    pub fn dot_batch(&self, rhs: &Self, out: &mut [f32]) {
        assert_eq!(self.len(), rhs.len(), "SoaVec3::dot_batch length mismatch");
        assert_eq!(
            self.len(),
            out.len(),
            "SoaVec3::dot_batch output length mismatch"
        );
        for (i, o) in out.iter_mut().enumerate() {
            *o = self.xs[i] * rhs.xs[i] + self.ys[i] * rhs.ys[i] + self.zs[i] * rhs.zs[i];
        }
    }

    /// Per-element Euclidean length, written to `out`.
    ///
    /// # Panics
    /// Panics if `out` is not the same length as the batch.
    #[inline]
    pub fn length_batch(&self, out: &mut [f32]) {
        assert_eq!(
            self.len(),
            out.len(),
            "SoaVec3::length_batch output length mismatch"
        );
        for (i, o) in out.iter_mut().enumerate() {
            let x = self.xs[i];
            let y = self.ys[i];
            let z = self.zs[i];
            *o = mf::sqrt(x * x + y * y + z * z);
        }
    }

    /// Normalize every element in place. Elements with a length at or below
    /// `1e-20` are left as zero to avoid producing non-finite values.
    #[inline]
    pub fn normalize(&mut self) {
        for i in 0..self.len() {
            let x = self.xs[i];
            let y = self.ys[i];
            let z = self.zs[i];
            let len = mf::sqrt(x * x + y * y + z * z);
            if len > 1.0e-20 {
                let inv = 1.0 / len;
                self.xs[i] = x * inv;
                self.ys[i] = y * inv;
                self.zs[i] = z * inv;
            } else {
                self.xs[i] = 0.0;
                self.ys[i] = 0.0;
                self.zs[i] = 0.0;
            }
        }
    }

    /// Transform every element as a point by an [`Affine3`] in place.
    #[inline]
    pub fn transform_points(&mut self, a: &Affine3) {
        let m = a.matrix3;
        let t = a.translation;
        for i in 0..self.len() {
            let x = self.xs[i];
            let y = self.ys[i];
            let z = self.zs[i];
            let rx = m.x_axis.x * x + m.y_axis.x * y + m.z_axis.x * z + t.x;
            let ry = m.x_axis.y * x + m.y_axis.y * y + m.z_axis.y * z + t.y;
            let rz = m.x_axis.z * x + m.y_axis.z * y + m.z_axis.z * z + t.z;
            self.xs[i] = rx;
            self.ys[i] = ry;
            self.zs[i] = rz;
        }
    }

    /// Transform every element as a direction by an [`Affine3`] in place
    /// (ignores translation).
    #[inline]
    pub fn transform_vectors(&mut self, a: &Affine3) {
        let m = a.matrix3;
        for i in 0..self.len() {
            let x = self.xs[i];
            let y = self.ys[i];
            let z = self.zs[i];
            self.xs[i] = m.x_axis.x * x + m.y_axis.x * y + m.z_axis.x * z;
            self.ys[i] = m.x_axis.y * x + m.y_axis.y * y + m.z_axis.y * z;
            self.zs[i] = m.x_axis.z * x + m.y_axis.z * y + m.z_axis.z * z;
        }
    }

    /// Transform every element as a point by a [`Mat4`] in place (applies the
    /// translation column, assuming an affine matrix with `w = 1`).
    #[inline]
    pub fn transform_points_mat4(&mut self, m: &Mat4) {
        for i in 0..self.len() {
            let x = self.xs[i];
            let y = self.ys[i];
            let z = self.zs[i];
            let r = m.x_axis * x + m.y_axis * y + m.z_axis * z + m.w_axis;
            self.xs[i] = r.x;
            self.ys[i] = r.y;
            self.zs[i] = r.z;
        }
    }
}
