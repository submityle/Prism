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
    /// Gribb-Hartmann method, for Prism's default clip volume
    /// `-w <= x, y <= w`, `0 <= z <= w` (the `[0, 1]` depth convention used by
    /// `wgpu`/D3D/Metal and by [`crate::projection::perspective_rh`]).
    ///
    /// `view_proj` maps world-space points to clip space via `view_proj * v`.
    /// Normals are normalized so signed distances are Euclidean.
    ///
    /// Use [`Frustum::from_view_proj_gl`] for the OpenGL `[-1, 1]` depth
    /// convention ([`crate::projection::perspective_rh_gl`]). Passing a
    /// `[-1, 1]` matrix here mislocates the near plane.
    pub fn from_view_proj(view_proj: Mat4) -> Self {
        let rows = frustum_rows(view_proj);
        let add = |p: [f32; 4], q: [f32; 4]| [p[0] + q[0], p[1] + q[1], p[2] + q[2], p[3] + q[3]];
        let sub = |p: [f32; 4], q: [f32; 4]| [p[0] - q[0], p[1] - q[1], p[2] - q[2], p[3] - q[3]];

        // `x >= -w` / `x <= w` / `y >= -w` / `y <= w`, and `z >= 0` (near is
        // `row2` directly, not `row3 + row2`) / `z <= w`.
        let l = add(rows[3], rows[0]);
        let r = sub(rows[3], rows[0]);
        let b = add(rows[3], rows[1]);
        let t = sub(rows[3], rows[1]);
        let n = rows[2];
        let f = sub(rows[3], rows[2]);
        Self::from_plane_rows([l, r, b, t, n, f])
    }

    /// Extract the six clip planes for the OpenGL `[-1, 1]` depth convention
    /// (`-w <= x, y, z <= w`), as produced by
    /// [`crate::projection::perspective_rh_gl`].
    pub fn from_view_proj_gl(view_proj: Mat4) -> Self {
        let rows = frustum_rows(view_proj);
        let add = |p: [f32; 4], q: [f32; 4]| [p[0] + q[0], p[1] + q[1], p[2] + q[2], p[3] + q[3]];
        let sub = |p: [f32; 4], q: [f32; 4]| [p[0] - q[0], p[1] - q[1], p[2] - q[2], p[3] - q[3]];

        let l = add(rows[3], rows[0]);
        let r = sub(rows[3], rows[0]);
        let b = add(rows[3], rows[1]);
        let t = sub(rows[3], rows[1]);
        let n = add(rows[3], rows[2]);
        let f = sub(rows[3], rows[2]);
        Self::from_plane_rows([l, r, b, t, n, f])
    }

    /// Build a frustum from six `[a, b, c, d]` plane rows, normalizing normals
    /// so signed distances are Euclidean.
    fn from_plane_rows(rows: [[f32; 4]; 6]) -> Self {
        let plane = |p: [f32; 4]| {
            let n = Vec3::new(p[0], p[1], p[2]);
            if n.length_squared() <= 1.0e-30 {
                // A degenerate clip row is an already-satisfied condition (e.g.
                // the far plane of an infinite projection, where `z <= w` holds
                // for every point). Normalizing `(0,0,0)` would yield NaN, so
                // store an always-pass plane `(0,0,0,1)` instead (CR-found).
                Plane::new(Vec3::ZERO, 1.0)
            } else {
                Plane::new(n, p[3]).normalized()
            }
        };
        Self {
            planes: [
                plane(rows[LEFT]),
                plane(rows[RIGHT]),
                plane(rows[BOTTOM]),
                plane(rows[TOP]),
                plane(rows[NEAR]),
                plane(rows[FAR]),
            ],
        }
    }

    /// True if `p` is inside (or on) all six planes.
    #[inline]
    pub fn contains_point(self, p: Vec3) -> bool {
        self.planes
            .iter()
            .all(|plane| plane.signed_distance(p) >= 0.0)
    }
}

/// The four rows `[row0, row1, row2, row3]` of a column-major matrix.
fn frustum_rows(view_proj: Mat4) -> [[f32; 4]; 4] {
    // Rows of the column-major matrix: row `r` gathers component `r` from each
    // column vector.
    let c0 = view_proj.x_axis;
    let c1 = view_proj.y_axis;
    let c2 = view_proj.z_axis;
    let c3 = view_proj.w_axis;
    let row0 = [c0.x, c1.x, c2.x, c3.x];
    let row1 = [c0.y, c1.y, c2.y, c3.y];
    let row2 = [c0.z, c1.z, c2.z, c3.z];
    let row3 = [c0.w, c1.w, c2.w, c3.w];
    [row0, row1, row2, row3]
}
