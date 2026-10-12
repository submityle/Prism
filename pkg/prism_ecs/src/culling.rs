//! CPU-side view-frustum culling for chunk windows (design §13 / §15).
//!
//! The kernel stores a world-space AABB per chunk ([`ChunkMeta::bounds`],
//! produced by [`Table::recompute_chunk_bounds`]) and a `visible` bit
//! ([`ChunkMeta::visible`]). This module supplies the missing middle: a
//! [`Frustum`] and the conservative AABB test, so a render pass can write the
//! `visible` bit of every window and then hand the table straight to
//! [`GpuBatchBuilder::build_from_table`](crate::gpu_batch::GpuBatchBuilder::build_from_table)
//! — a fully CPU-side, testable pipeline with no GPU dependency.
//!
//! [`Table::recompute_chunk_bounds`]: crate::storage::Table::recompute_chunk_bounds
//! [`Table::cull_chunks`]: crate::storage::Table::cull_chunks
//! [`ChunkMeta::bounds`]: crate::storage::ChunkMeta::bounds
//! [`ChunkMeta::visible`]: crate::storage::ChunkMeta::visible

/// A clip plane in `a*x + b*y + c*z + d >= 0` (inside) form.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    /// Plane normal `(a, b, c)`.
    pub normal: [f32; 3],
    /// Plane offset `d`.
    pub d: f32,
}

impl Plane {
    /// Build a plane from its normal and offset.
    #[inline]
    pub const fn new(normal: [f32; 3], d: f32) -> Self {
        Self { normal, d }
    }

    /// Signed distance of `p` from the plane, scaled by the normal's length
    /// (the sign is what the frustum test needs; call `normalized` first if you
    /// want a true distance).
    #[inline]
    pub fn signed_distance(&self, p: [f32; 3]) -> f32 {
        self.normal[0] * p[0] + self.normal[1] * p[1] + self.normal[2] * p[2] + self.d
    }

    /// Scale the plane so `signed_distance` is a true distance.
    #[inline]
    pub fn normalized(self) -> Self {
        let [a, b, c] = self.normal;
        let len = libm::sqrtf(a * a + b * b + c * c);
        if len > 0.0 {
            Self::new([a / len, b / len, c / len], self.d / len)
        } else {
            self
        }
    }
}

/// Six clip planes bounding the visible region.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frustum {
    planes: [Plane; 6],
}

impl Frustum {
    /// Build a frustum from its six planes (order is irrelevant to the test).
    #[inline]
    pub const fn from_planes(planes: [Plane; 6]) -> Self {
        Self { planes }
    }

    /// The six clip planes.
    #[inline]
    pub fn planes(&self) -> &[Plane; 6] {
        &self.planes
    }

    /// Extract the six clip planes from a **row-major** view-projection matrix
    /// (Gribb–Hartmann), using the OpenGL clip convention `-w <= z <= w`.
    ///
    /// With `m[i]` the `i`-th row, the planes are `row3 ± rowj` for
    /// `j = 0..3` (left/right, bottom/top, near/far). The result is
    /// conservative: a box the frustum touches exactly is reported visible.
    pub fn from_view_projection(m: &[[f32; 4]; 4]) -> Self {
        let add = |a: [f32; 4], b: [f32; 4]| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
        let sub = |a: [f32; 4], b: [f32; 4]| [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]];
        let row3 = m[3];
        let plane = |v: [f32; 4]| Plane::new([v[0], v[1], v[2]], v[3]);
        Self::from_planes([
            plane(add(row3, m[0])), // left
            plane(sub(row3, m[0])), // right
            plane(add(row3, m[1])), // bottom
            plane(sub(row3, m[1])), // top
            plane(add(row3, m[2])), // near
            plane(sub(row3, m[2])), // far
        ])
    }

    /// Conservative AABB test: `false` only when the box is **fully outside** at
    /// least one plane (so a box touching the frustum is reported visible).
    pub fn intersects_aabb(&self, min: [f32; 3], max: [f32; 3]) -> bool {
        for plane in &self.planes {
            // The AABB's "positive vertex" — the corner furthest along the plane
            // normal. If even that corner is behind the plane the whole box is.
            let positive = [
                if plane.normal[0] >= 0.0 {
                    max[0]
                } else {
                    min[0]
                },
                if plane.normal[1] >= 0.0 {
                    max[1]
                } else {
                    min[1]
                },
                if plane.normal[2] >= 0.0 {
                    max[2]
                } else {
                    min[2]
                },
            ];
            if plane.signed_distance(positive) < 0.0 {
                return false;
            }
        }
        true
    }
}

/// What one call to [`Table::cull_chunks`](crate::storage::Table::cull_chunks)
/// did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChunkCullStats {
    /// Chunk windows examined.
    pub tested: usize,
    /// Windows left visible.
    pub visible: usize,
    /// Windows culled.
    pub culled: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An axis-aligned "box frustum": `min <= p <= max` on every axis, built as
    /// six planes.
    fn box_frustum(min: [f32; 3], max: [f32; 3]) -> Frustum {
        Frustum::from_planes([
            Plane::new([1.0, 0.0, 0.0], -min[0]),
            Plane::new([-1.0, 0.0, 0.0], max[0]),
            Plane::new([0.0, 1.0, 0.0], -min[1]),
            Plane::new([0.0, -1.0, 0.0], max[1]),
            Plane::new([0.0, 0.0, 1.0], -min[2]),
            Plane::new([0.0, 0.0, -1.0], max[2]),
        ])
    }

    #[test]
    fn aabb_test_is_conservative_on_touching_boxes() {
        let f = box_frustum([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
        // Fully inside.
        assert!(f.intersects_aabb([1.0, 1.0, 1.0], [9.0, 9.0, 9.0]));
        // Straddling an edge is visible.
        assert!(f.intersects_aabb([-5.0, 2.0, 2.0], [5.0, 4.0, 4.0]));
        // Exactly touching a face is still visible (conservative).
        assert!(f.intersects_aabb([10.0, 2.0, 2.0], [12.0, 4.0, 4.0]));
        // Fully outside on one axis is culled.
        assert!(!f.intersects_aabb([11.0, 2.0, 2.0], [12.0, 4.0, 4.0]));
        assert!(!f.intersects_aabb([-3.0, -3.0, -3.0], [-0.5, -0.5, -0.5]));
    }

    #[test]
    fn planes_from_an_identity_view_projection_bound_the_unit_cube() {
        // With `M = I` the clip condition `-w <= xyz <= w` is `-1 <= xyz <= 1`,
        // so the extracted frustum is the [-1, 1] cube.
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let f = Frustum::from_view_projection(&identity);
        for plane in f.planes() {
            let n = plane.normal;
            let len = libm::sqrtf(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            // Each plane's unit-normal distance from the origin is 1, and the
            // outward normal points away from the origin.
            assert!((plane.d / len - 1.0).abs() < 1e-5);
        }
        assert!(f.intersects_aabb([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]));
        assert!(!f.intersects_aabb([2.0, -0.5, -0.5], [3.0, 0.5, 0.5]));
        assert!(!f.intersects_aabb([-0.5, -0.5, -3.0], [0.5, 0.5, -2.0]));
    }
}
