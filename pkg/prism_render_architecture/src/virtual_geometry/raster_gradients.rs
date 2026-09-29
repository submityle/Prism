//! Screen-space triangle gradients for the vis-buffer software rasterizer.
//!
//! The shipping compute twin does not recompute the edge functions at every
//! pixel; it sets up per-triangle *gradients* once and steps them incrementally
//! across the bounding box (`w += w_x` per column, `w += w_y` per row) with the
//! depth plane carried along the same way (`z += z_x` / `z += z_y`). This module
//! pins that gradient setup as a CPU reference so the twin's incremental
//! traversal is diffed against a known-good affine formulation, complementing
//! the per-pixel-recompute golden path in [`super::software_raster`].
//!
//! # Relationship to the recompute path
//!
//! For a front-facing triangle with signed double area `A = edge(v0, v1, v2)`,
//! the three edge functions
//!
//! ```text
//! w = (edge(v1, v2, p), edge(v2, v0, p), edge(v0, v1, p))
//! ```
//!
//! are each *affine* in the pixel position `p`, so their exact per-pixel value
//! equals a single setup value stepped by a constant gradient:
//!
//! ```text
//! w_x = (v1.y - v2.y, v2.y - v0.y, v0.y - v1.y)   // d(w)/dx
//! w_y = (v2.x - v1.x, v0.x - v2.x, v1.x - v0.x)   // d(w)/dy
//! ```
//!
//! Screen-space-linear depth is likewise affine. With the shipping
//! normalization `vertices_z = (v0.z, v1.z, v2.z) / A`, the interpolated depth
//! is `z(p) = dot(vertices_z, w(p))`, which matches the barycentric depth of
//! the recompute path exactly (`b_i = w_i / A`), and its gradients are
//! `z_x = dot(vertices_z, w_x)`, `z_y = dot(vertices_z, w_y)`.
//!
//! # Numerical note
//!
//! In real arithmetic the incremental walk is exact; in floating point the
//! twin's `w += w_x` accumulation may drift from a per-pixel recompute, which is
//! why [`super::software_raster::rasterize_triangle`] recomputes and remains the
//! authoritative watertight reference. This module documents and validates the
//! *setup* the twin derives its walk from; the equivalence tests use
//! integer-coordinate triangles where every operation is exact.

use super::software_raster::{edge, ScreenVertex};

/// Per-triangle screen-space gradients matching the shipping compute twin.
///
/// Construct with [`TriangleGradients::new`]; the fields expose the exact
/// constants the twin steps its edge functions and depth plane by.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriangleGradients {
    /// Signed double area `edge(v0, v1, v2)`, strictly positive (front-facing).
    pub double_area: f32,
    /// Per-column (`+x`) increment of the three edge functions.
    pub w_x: [f32; 3],
    /// Per-row (`+y`) increment of the three edge functions.
    pub w_y: [f32; 3],
    /// Depth weights `(v0.z, v1.z, v2.z) / double_area` (shipping `vertices_z`).
    pub vertices_z: [f32; 3],
    /// Per-column (`+x`) increment of screen-space-linear depth.
    pub z_x: f32,
    /// Per-row (`+y`) increment of screen-space-linear depth.
    pub z_y: f32,
}

impl TriangleGradients {
    /// Builds the gradient setup for a front-facing triangle.
    ///
    /// Mirrors the shipping twin's precondition: the signed double area
    /// `edge(v0, v1, v2)` must be strictly positive (front-facing under the
    /// y-down, positive-area convention). Degenerate or back-facing triangles
    /// (`double_area <= 0`) return `None`, exactly the cases the twin culls
    /// before setting up gradients.
    #[must_use]
    pub fn new(v0: ScreenVertex, v1: ScreenVertex, v2: ScreenVertex) -> Option<Self> {
        let double_area = edge(v0.pos, v1.pos, v2.pos);
        if double_area <= 0.0 {
            return None;
        }
        let inv_area = 1.0 / double_area;
        let w_x = [
            v1.pos[1] - v2.pos[1],
            v2.pos[1] - v0.pos[1],
            v0.pos[1] - v1.pos[1],
        ];
        let w_y = [
            v2.pos[0] - v1.pos[0],
            v0.pos[0] - v2.pos[0],
            v1.pos[0] - v0.pos[0],
        ];
        let vertices_z = [v0.depth * inv_area, v1.depth * inv_area, v2.depth * inv_area];
        let z_x = vertices_z[0] * w_x[0] + vertices_z[1] * w_x[1] + vertices_z[2] * w_x[2];
        let z_y = vertices_z[0] * w_y[0] + vertices_z[1] * w_y[1] + vertices_z[2] * w_y[2];
        Some(Self {
            double_area,
            w_x,
            w_y,
            vertices_z,
            z_x,
            z_y,
        })
    }

    /// Screen-space-linear depth from a triple of edge-function values.
    ///
    /// `w` is `(edge(v1, v2, p), edge(v2, v0, p), edge(v0, v1, p))` at the
    /// pixel; the result is `dot(vertices_z, w)`, equal to the barycentric depth
    /// of the recompute path.
    #[must_use]
    pub fn depth_from_edges(&self, w: [f32; 3]) -> f32 {
        self.vertices_z[0] * w[0] + self.vertices_z[1] * w[1] + self.vertices_z[2] * w[2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sv(x: f32, y: f32, depth: f32) -> ScreenVertex {
        ScreenVertex::new([x, y], depth)
    }

    #[test]
    fn back_facing_and_degenerate_return_none() {
        // Clockwise winding in y-down space -> non-positive area.
        let back = TriangleGradients::new(sv(0.0, 0.0, 0.0), sv(0.0, 4.0, 0.0), sv(4.0, 0.0, 0.0));
        assert!(back.is_none());
        // Collinear vertices -> zero area.
        let degen = TriangleGradients::new(sv(0.0, 0.0, 0.0), sv(1.0, 1.0, 0.0), sv(2.0, 2.0, 0.0));
        assert!(degen.is_none());
    }

    #[test]
    fn w_gradients_match_edge_finite_differences() {
        // Integer coordinates keep every f32 operation exact.
        let v0 = sv(1.0, 2.0, 0.0);
        let v1 = sv(6.0, 3.0, 0.0);
        let v2 = sv(2.0, 7.0, 0.0);
        let g = TriangleGradients::new(v0, v1, v2).unwrap();
        let p = [3.0, 4.0];
        let px = [p[0] + 1.0, p[1]];
        let py = [p[0], p[1] + 1.0];
        // Edge order matches shipping w_row: (v1v2, v2v0, v0v1).
        let e = [
            edge(v1.pos, v2.pos, p),
            edge(v2.pos, v0.pos, p),
            edge(v0.pos, v1.pos, p),
        ];
        let e_dx = [
            edge(v1.pos, v2.pos, px),
            edge(v2.pos, v0.pos, px),
            edge(v0.pos, v1.pos, px),
        ];
        let e_dy = [
            edge(v1.pos, v2.pos, py),
            edge(v2.pos, v0.pos, py),
            edge(v0.pos, v1.pos, py),
        ];
        for i in 0..3 {
            assert_eq!(e_dx[i] - e[i], g.w_x[i], "w_x[{i}] must be the +x finite difference");
            assert_eq!(e_dy[i] - e[i], g.w_y[i], "w_y[{i}] must be the +y finite difference");
        }
    }

    #[test]
    fn depth_plane_matches_barycentric_recompute() {
        // Distinct integer depths so the plane is non-trivial and exact.
        let v0 = sv(0.0, 0.0, 0.25);
        let v1 = sv(8.0, 0.0, 0.5);
        let v2 = sv(0.0, 8.0, 1.0);
        let g = TriangleGradients::new(v0, v1, v2).unwrap();
        let p = [2.0, 3.0];
        let e = [
            edge(v1.pos, v2.pos, p),
            edge(v2.pos, v0.pos, p),
            edge(v0.pos, v1.pos, p),
        ];
        let inv_area = 1.0 / g.double_area;
        // Barycentric depth exactly as the recompute path computes it.
        let bary = (e[0] * inv_area) * v0.depth
            + (e[1] * inv_area) * v1.depth
            + (e[2] * inv_area) * v2.depth;
        assert_eq!(g.depth_from_edges(e), bary);
    }

    #[test]
    fn depth_gradients_step_the_plane_exactly() {
        let v0 = sv(0.0, 0.0, 0.25);
        let v1 = sv(8.0, 0.0, 0.5);
        let v2 = sv(0.0, 8.0, 1.0);
        let g = TriangleGradients::new(v0, v1, v2).unwrap();
        let p = [2.0, 3.0];
        let e_at = |q: [f32; 2]| {
            [
                edge(v1.pos, v2.pos, q),
                edge(v2.pos, v0.pos, q),
                edge(v0.pos, v1.pos, q),
            ]
        };
        let z0 = g.depth_from_edges(e_at(p));
        let z_dx = g.depth_from_edges(e_at([p[0] + 1.0, p[1]]));
        let z_dy = g.depth_from_edges(e_at([p[0], p[1] + 1.0]));
        // The gradient reproduces the recomputed plane step for integer coords.
        assert_eq!(z_dx - z0, g.z_x);
        assert_eq!(z_dy - z0, g.z_y);
    }
}
