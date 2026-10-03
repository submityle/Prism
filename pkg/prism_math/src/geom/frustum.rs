//! View frustum [`Frustum`] extracted from a view-projection matrix.

use crate::geom::plane::Plane;
use crate::mat::Mat4;
use crate::vec::Vec3;

/// Index of the left clip plane in [`Frustum::planes`].
pub const LEFT: usize = 0;
/// Index of the right clip plane in [`Frustum::planes`].
pub const RIGHT: usize = 1;
/// Index of the bottom clip plane in [`Frustum::planes`].
pub const BOTTOM: usize = 2;
/// Index of the top clip plane in [`Frustum::planes`].
pub const TOP: usize = 3;
/// Index of the near clip plane in [`Frustum::planes`].
pub const NEAR: usize = 4;
/// Index of the far clip plane in [`Frustum::planes`].
pub const FAR: usize = 5;

/// A six-plane view frustum whose plane normals point *inward*, so a point is
/// inside the frustum exactly when it has non-negative signed distance to all
/// six planes.
///
/// Build one with [`Frustum::from_view_proj`]. The plane order matches the
/// [`LEFT`], [`RIGHT`], [`BOTTOM`], [`TOP`], [`NEAR`], [`FAR`] indices.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Frustum {
    /// The six bounding planes, each with an inward-facing unit normal.
    pub planes: [Plane; 6],
}

impl Frustum {
    /// Construct a frustum directly from six planes (already inward-facing).
    #[inline]
    pub const fn from_planes(planes: [Plane; 6]) -> Self {
        Self { planes }
    }

    /// Extract the six clip planes from a view-projection matrix using the
    /// Gribb-Hartmann method.
    ///
    /// `view_proj` maps world-space points to clip space via `view_proj * v`.
    /// The resulting planes assume a clip volume of `-w <= x, y, z <= w` (the
    /// `[-1, 1]` depth convention). Normals are normalized so signed distances
    /// are Euclidean.
    pub fn from_view_proj(view_proj: Mat4) -> Self {
        // Rows of the column-major matrix: row `r` gathers component `r` from
        // each column vector.
        let c0 = view_proj.x_axis;
        let c1 = view_proj.y_axis;
        let c2 = view_proj.z_axis;
        let c3 = view_proj.w_axis;
        let row0 = [c0.x, c1.x, c2.x, c3.x];
        let row1 = [c0.y, c1.y, c2.y, c3.y];
        let row2 = [c0.z, c1.z, c2.z, c3.z];
        let row3 = [c0.w, c1.w, c2.w, c3.w];

        let plane = |a: f32, b: f32, c: f32, d: f32| Plane::new(Vec3::new(a, b, c), d).normalized();
        let add = |p: [f32; 4], q: [f32; 4]| [p[0] + q[0], p[1] + q[1], p[2] + q[2], p[3] + q[3]];
        let sub = |p: [f32; 4], q: [f32; 4]| [p[0] - q[0], p[1] - q[1], p[2] - q[2], p[3] - q[3]];

        let l = add(row3, row0);
        let r = sub(row3, row0);
        let b = add(row3, row1);
        let t = sub(row3, row1);
        let n = add(row3, row2);
        let f = sub(row3, row2);

        Self {
            planes: [
                plane(l[0], l[1], l[2], l[3]),
                plane(r[0], r[1], r[2], r[3]),
                plane(b[0], b[1], b[2], b[3]),
                plane(t[0], t[1], t[2], t[3]),
                plane(n[0], n[1], n[2], n[3]),
                plane(f[0], f[1], f[2], f[3]),
            ],
        }
    }

    /// True if `p` is inside (or on) all six planes.
    #[inline]
    pub fn contains_point(self, p: Vec3) -> bool {
        self.planes.iter().all(|plane| plane.signed_distance(p) >= 0.0)
    }
}
