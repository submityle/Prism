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

    /// Per-row coverage interval `[k_lo, k_hi]` in `+x` step units, or `None`.
    ///
    /// The shipping compute twin, when a row is wide enough
    /// (`subgroupAny(max_x - min_x > 4)`), stops testing every pixel and instead
    /// solves the three edge inequalities for the row to get one contiguous
    /// `[x0, x1]` span, then walks only that span. This method pins the closed
    /// form of that span as a CPU reference.
    ///
    /// `w_row` is the edge-function triple
    /// `(edge(v1, v2, p0), edge(v2, v0, p0), edge(v0, v1, p0))` sampled at the
    /// row's first candidate pixel center `p0`; `steps` is the number of `+x`
    /// unit steps to the last candidate (so the row spans `k in 0..=steps`, i.e.
    /// `steps + 1` pixels). Because each edge is affine, its value at step `k` is
    /// `w_row[i] + k * self.w_x[i]`, and the covered set `all(w_i >= 0)` is the
    /// intersection of three half-lines — a single interval. The returned bounds
    /// are clamped to `[0, steps]`; `None` means the row is not covered.
    ///
    /// # Numerical note
    ///
    /// Coverage uses inclusive `w_i >= 0` (not the top-left tie-break of
    /// [`super::software_raster`]); at generic pixel centers no edge value is
    /// exactly zero, so the interval matches a per-pixel `all(w_i >= 0)` scan
    /// bit-for-bit. Exact-on-edge ties are resolved by the twin's per-pixel guard
    /// and are outside this closed form's scope.
    #[must_use]
    pub fn row_span(&self, w_row: [f32; 3], steps: u32) -> Option<(u32, u32)> {
        let steps_f = steps as f32;
        let mut lo = 0.0f32;
        let mut hi = steps_f;
        for (&g, &w0) in self.w_x.iter().zip(w_row.iter()) {
            if g > 0.0 {
                // w0 + k*g >= 0  =>  k >= -w0/g
                lo = lo.max(-w0 / g);
            } else if g < 0.0 {
                // w0 + k*g >= 0  =>  k <= -w0/g
                hi = hi.min(-w0 / g);
            } else if w0 < 0.0 {
                // Edge is constant along the row and already outside.
                return None;
            }
        }
        let k_lo = lo.ceil();
        let k_hi = hi.floor();
        if k_lo > k_hi || k_hi < 0.0 || k_lo > steps_f {
            return None;
        }
        Some((k_lo.max(0.0) as u32, k_hi.min(steps_f) as u32))
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

    #[test]
    fn row_span_matches_per_pixel_scan_and_is_contiguous() {
        // Generic-position integer vertices: no pixel center lands exactly on an
        // edge, so the closed-form interval and a per-pixel `all(w_i >= 0)` scan
        // agree with no float tie-breaking ambiguity.
        let v0 = sv(1.0, 1.0, 0.2);
        let v1 = sv(20.0, 3.0, 0.6);
        let v2 = sv(4.0, 18.0, 0.9);
        let g = TriangleGradients::new(v0, v1, v2).unwrap();
        // Sanity: this winding is front-facing.
        assert!(g.double_area > 0.0);

        const WIDTH: u32 = 24;
        let steps = WIDTH - 1;
        for y in 0..24u32 {
            let cy = y as f32 + 0.5;
            // Edge triple at the first candidate pixel center (x = 0).
            let p0 = [0.5, cy];
            let w_row = [
                edge(v1.pos, v2.pos, p0),
                edge(v2.pos, v0.pos, p0),
                edge(v0.pos, v1.pos, p0),
            ];

            // Brute-force per-pixel coverage over the same row (inclusive w >= 0).
            let mut covered = Vec::new();
            for k in 0..=steps {
                let p = [k as f32 + 0.5, cy];
                let w = [
                    edge(v1.pos, v2.pos, p),
                    edge(v2.pos, v0.pos, p),
                    edge(v0.pos, v1.pos, p),
                ];
                if w[0] >= 0.0 && w[1] >= 0.0 && w[2] >= 0.0 {
                    covered.push(k);
                }
            }

            match g.row_span(w_row, steps) {
                None => {
                    assert!(covered.is_empty(), "row y={y}: closed form empty but scan covered {covered:?}");
                }
                Some((k_lo, k_hi)) => {
                    assert!(!covered.is_empty(), "row y={y}: closed form span but scan empty");
                    assert_eq!(k_lo, *covered.first().unwrap(), "row y={y}: lo bound mismatch");
                    assert_eq!(k_hi, *covered.last().unwrap(), "row y={y}: hi bound mismatch");
                    // The covered set must be exactly the contiguous run [k_lo, k_hi].
                    let expected: Vec<u32> = (k_lo..=k_hi).collect();
                    assert_eq!(covered, expected, "row y={y}: coverage not contiguous");
                }
            }
        }
    }

    #[test]
    fn row_span_clamps_to_the_step_range() {
        // A large triangle so an interior row is fully covered; the span must be
        // clamped to [0, steps] rather than running past the candidate range.
        let v0 = sv(-100.0, -100.0, 0.1);
        let v1 = sv(300.0, -100.0, 0.5);
        let v2 = sv(-100.0, 300.0, 0.9);
        let g = TriangleGradients::new(v0, v1, v2).unwrap();
        let cy = 8.5;
        let p0 = [0.5, cy];
        let w_row = [
            edge(v1.pos, v2.pos, p0),
            edge(v2.pos, v0.pos, p0),
            edge(v0.pos, v1.pos, p0),
        ];
        let steps = 15u32;
        let (k_lo, k_hi) = g.row_span(w_row, steps).unwrap();
        assert_eq!(k_lo, 0, "fully covered row must start at step 0");
        assert_eq!(k_hi, steps, "span must be clamped to the last step");
    }

    #[test]
    fn span_walk_depth_matches_recompute_bit_for_bit() {
        // The compute twin's inner loop walks only the `row_span` interval while
        // stepping the depth plane by `z_x` per column (`z += z_x`) instead of
        // recomputing the barycentric depth at each pixel. This pins that fast
        // path against the per-pixel recompute. Dyadic coordinates keep every
        // operation exact: `double_area = 256 = 2^8`, so `vertices_z` and `z_x`
        // are exactly representable and the walk is bit-for-bit, not approximate.
        let v0 = sv(0.0, 0.0, 0.25);
        let v1 = sv(16.0, 0.0, 0.5);
        let v2 = sv(0.0, 16.0, 0.75);
        let g = TriangleGradients::new(v0, v1, v2).unwrap();
        assert_eq!(g.double_area, 256.0, "chosen so vertices_z/z_x are dyadic-exact");

        const WIDTH: u32 = 18;
        let steps = WIDTH - 1;
        let mut rows_tested = 0u32;
        for y in 0..16u32 {
            let cy = y as f32 + 0.5;
            let p0 = [0.5, cy];
            let w_row = [
                edge(v1.pos, v2.pos, p0),
                edge(v2.pos, v0.pos, p0),
                edge(v0.pos, v1.pos, p0),
            ];
            let Some((k_lo, k_hi)) = g.row_span(w_row, steps) else {
                continue;
            };
            rows_tested += 1;

            // Depth at the row's origin pixel (x = 0); the closed-form plane is
            // defined for every column, covered or not.
            let z_base = g.depth_from_edges(w_row);
            for k in k_lo..=k_hi {
                // Incremental fast path: base plane stepped `k` columns by z_x.
                let z_incremental = z_base + (k as f32) * g.z_x;

                // Recompute path A: edge triple stepped, then dot(vertices_z, .).
                let w_k = [
                    w_row[0] + (k as f32) * g.w_x[0],
                    w_row[1] + (k as f32) * g.w_x[1],
                    w_row[2] + (k as f32) * g.w_x[2],
                ];
                let z_recompute = g.depth_from_edges(w_k);

                // Recompute path B: edge functions sampled directly at the pixel
                // center, exactly as `software_raster` does per pixel.
                let p = [k as f32 + 0.5, cy];
                let w_pixel = [
                    edge(v1.pos, v2.pos, p),
                    edge(v2.pos, v0.pos, p),
                    edge(v0.pos, v1.pos, p),
                ];
                let z_pixel = g.depth_from_edges(w_pixel);

                assert_eq!(
                    z_incremental, z_recompute,
                    "row y={y} step k={k}: z_x walk must equal edge-recompute"
                );
                assert_eq!(
                    z_recompute, z_pixel,
                    "row y={y} step k={k}: stepped edges must equal per-pixel sample"
                );
            }
        }
        assert!(rows_tested >= 4, "expected several covered rows, got {rows_tested}");
    }
}
