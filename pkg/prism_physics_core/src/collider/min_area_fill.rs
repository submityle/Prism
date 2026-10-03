//! Minimum-area triangulation of a closed boundary loop.
//!
//! Filling a hole in a collision mesh means covering a closed ring of vertices
//! with a triangle fan or strip. When the ring is (nearly) planar, projecting
//! it to a best-fit plane and ear-clipping works well. When the ring is
//! *strongly non-planar* — a saddle-shaped gap, a twisted seam, a crater rim —
//! a projected ear-clip can produce long, skinny, or self-overlapping
//! triangles because the projection distorts the true geometry.
//!
//! This module avoids the projection entirely. It computes, over all
//! triangulations that respect the ring's cyclic order, the one whose total
//! triangle area in 3D is minimal. This is the classic minimum-weight polygon
//! triangulation solved by Klincsek's dynamic program in `O(n^3)` time and
//! `O(n^2)` memory, with triangle area as the weight. Minimizing surface area
//! is a good proxy for a taut, well-shaped fill and is robust to how the loop
//! bends through space, so it degrades gracefully on the non-planar rings that
//! defeat the projected ear-clip.
//!
//! The algorithm triangulates the polygon bounded by the ring edges
//! `(0,1), (1,2), ..., (n-2,n-1)` and the closing edge `(n-1, 0)`. It produces
//! exactly `n - 2` triangles for a ring of `n` vertices, referenced by their
//! local index into the input slice, so a caller is free to map them back onto
//! whatever global vertex ids the loop came from. Ties between equal-area
//! split points are broken toward the lowest index, making the output fully
//! deterministic.
//!
//! This is pure polyline geometry with no coupling to the collision pipeline,
//! and nothing here is derived from Unreal Engine source.

use glam::Vec3;

/// An upper bound on ring size, guarding the `O(n^3)` dynamic program against
/// pathological inputs. Boundary loops that need filling are small in practice.
pub const MAX_LOOP_VERTICES: usize = 1024;

/// A minimum-area triangulation of a boundary loop.
#[derive(Clone, Debug, PartialEq)]
pub struct MinAreaFill {
    /// Triangles as triples of local indices into the input loop slice.
    ///
    /// Each triple is wound consistently with the ring's traversal order
    /// (`i < k < j`), so the fill inherits whatever orientation convention the
    /// loop carries.
    pub triangles: Vec<[u32; 3]>,
    /// Total surface area of the triangulation, in the loop's units.
    pub total_area: f32,
}

impl MinAreaFill {
    /// Number of triangles in the fill (`n - 2` for an `n`-vertex loop).
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }
}

/// Area of the triangle `(a, b, c)` evaluated in double precision.
fn triangle_area_f64(a: Vec3, b: Vec3, c: Vec3) -> f64 {
    let (ax, ay, az) = (f64::from(a.x), f64::from(a.y), f64::from(a.z));
    let (bx, by, bz) = (f64::from(b.x), f64::from(b.y), f64::from(b.z));
    let (cx, cy, cz) = (f64::from(c.x), f64::from(c.y), f64::from(c.z));
    let (ux, uy, uz) = (bx - ax, by - ay, bz - az);
    let (vx, vy, vz) = (cx - ax, cy - ay, cz - az);
    let nx = uy * vz - uz * vy;
    let ny = uz * vx - ux * vz;
    let nz = ux * vy - uy * vx;
    0.5 * (nx * nx + ny * ny + nz * nz).sqrt()
}

/// Triangulate a closed boundary loop so that total triangle area is minimal.
///
/// `loop_positions` lists the ring vertices in cyclic order; the closing edge
/// from the last vertex back to the first is implied. Returns [`None`] when the
/// ring has fewer than three vertices or more than [`MAX_LOOP_VERTICES`].
///
/// The returned triangles index into `loop_positions`. The result is
/// deterministic for a given input.
#[must_use]
pub fn triangulate_min_area(loop_positions: &[Vec3]) -> Option<MinAreaFill> {
    let n = loop_positions.len();
    if !(3..=MAX_LOOP_VERTICES).contains(&n) {
        return None;
    }
    if n == 3 {
        let area = triangle_area_f64(loop_positions[0], loop_positions[1], loop_positions[2]);
        return Some(MinAreaFill {
            triangles: vec![[0, 1, 2]],
            total_area: area as f32,
        });
    }

    // `cost[i * n + j]` is the minimum total area of triangulating the
    // sub-polygon spanning ring vertices `i..=j` (via the chord `i-j`), for
    // `j > i`. `split[i * n + j]` records the apex `k` that achieved it so the
    // triangulation can be reconstructed.
    let mut cost = vec![0.0_f64; n * n];
    let mut split = vec![0_u32; n * n];

    // `gap == 1` spans share only an edge, so they contribute no triangle and
    // keep the zero-initialised cost. Build up longer spans from shorter ones.
    for gap in 2..n {
        for i in 0..n - gap {
            let j = i + gap;
            let mut best_cost = f64::INFINITY;
            let mut best_k = i + 1;
            for k in (i + 1)..j {
                let candidate = cost[i * n + k]
                    + cost[k * n + j]
                    + triangle_area_f64(loop_positions[i], loop_positions[k], loop_positions[j]);
                if candidate < best_cost {
                    best_cost = candidate;
                    best_k = k;
                }
            }
            cost[i * n + j] = best_cost;
            split[i * n + j] = best_k as u32;
        }
    }

    // Reconstruct triangles from the span `0..=n-1`, which is closed by the
    // implied edge `(n-1, 0)`. An explicit stack avoids deep recursion on long
    // loops while preserving a deterministic, lowest-index-first ordering.
    let mut triangles = Vec::with_capacity(n - 2);
    let mut stack = vec![(0_usize, n - 1)];
    while let Some((i, j)) = stack.pop() {
        if j <= i + 1 {
            continue;
        }
        let k = split[i * n + j] as usize;
        triangles.push([i as u32, k as u32, j as u32]);
        // Push the right span last so the left span is expanded first, giving
        // a stable, root-to-leaf triangle order.
        stack.push((k, j));
        stack.push((i, k));
    }

    Some(MinAreaFill {
        triangles,
        total_area: cost[n - 1] as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeSet;

    /// Analytic area of a planar polygon via the shoelace formula in its plane.
    fn planar_polygon_area_xz(points: &[Vec3]) -> f64 {
        let mut sum = 0.0_f64;
        for i in 0..points.len() {
            let a = points[i];
            let b = points[(i + 1) % points.len()];
            sum += f64::from(a.x) * f64::from(b.z) - f64::from(b.x) * f64::from(a.z);
        }
        0.5 * sum.abs()
    }

    fn fill_area_from_triangles(points: &[Vec3], fill: &MinAreaFill) -> f64 {
        fill.triangles
            .iter()
            .map(|t| {
                triangle_area_f64(
                    points[t[0] as usize],
                    points[t[1] as usize],
                    points[t[2] as usize],
                )
            })
            .sum()
    }

    #[test]
    fn rejects_degenerate_or_oversized_loops() {
        assert!(triangulate_min_area(&[]).is_none());
        assert!(triangulate_min_area(&[Vec3::ZERO]).is_none());
        assert!(triangulate_min_area(&[Vec3::ZERO, Vec3::X]).is_none());
        let big = vec![Vec3::ZERO; MAX_LOOP_VERTICES + 1];
        assert!(triangulate_min_area(&big).is_none());
    }

    #[test]
    fn single_triangle_is_returned_verbatim() {
        let tri = [
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let fill = triangulate_min_area(&tri).expect("triangle fills");
        assert_eq!(fill.triangle_count(), 1);
        assert_eq!(fill.triangles[0], [0, 1, 2]);
        assert!((f64::from(fill.total_area) - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn produces_n_minus_two_triangles_covering_each_vertex() {
        // Regular octagon in the XZ plane.
        let n = 8usize;
        let ring: Vec<Vec3> = (0..n)
            .map(|i| {
                let a = core::f64::consts::TAU * (i as f64) / (n as f64);
                Vec3::new(a.cos() as f32, 0.0, a.sin() as f32)
            })
            .collect();
        let fill = triangulate_min_area(&ring).expect("octagon fills");
        assert_eq!(fill.triangle_count(), n - 2);

        // Every triangle references in-range, distinct vertices.
        let mut used = BTreeSet::new();
        for t in &fill.triangles {
            assert!(t[0] != t[1] && t[1] != t[2] && t[0] != t[2]);
            for &v in t {
                assert!((v as usize) < n);
                used.insert(v);
            }
        }
        // A valid fan/strip touches every ring vertex at least once.
        assert_eq!(used.len(), n);
    }

    #[test]
    fn convex_planar_polygon_area_equals_polygon_area() {
        // Any triangulation of a convex planar polygon has the same total area,
        // which must equal the polygon's own area.
        let n = 7usize;
        let ring: Vec<Vec3> = (0..n)
            .map(|i| {
                let a = core::f64::consts::TAU * (i as f64) / (n as f64);
                Vec3::new(1.3 * a.cos() as f32, 0.0, 1.3 * a.sin() as f32)
            })
            .collect();
        let fill = triangulate_min_area(&ring).expect("heptagon fills");
        let expected = planar_polygon_area_xz(&ring);
        assert!(
            (f64::from(fill.total_area) - expected).abs() < 1.0e-5,
            "fill area {} vs polygon area {expected}",
            fill.total_area
        );
        // Reported area matches the actual triangle areas.
        assert!((fill_area_from_triangles(&ring, &fill) - expected).abs() < 1.0e-5);
    }

    #[test]
    fn unit_square_fills_with_area_one() {
        let square = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let fill = triangulate_min_area(&square).expect("square fills");
        assert_eq!(fill.triangle_count(), 2);
        assert!((f64::from(fill.total_area) - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn picks_the_cheaper_diagonal_on_a_nonplanar_quad() {
        // A twisted quad where the two diagonals give different total areas.
        let quad = [
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(1.0, 0.3, -1.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(-1.0, 1.5, 1.0),
        ];
        // Diagonal (0,2): triangles (0,1,2) + (0,2,3).
        let d02 = triangle_area_f64(quad[0], quad[1], quad[2])
            + triangle_area_f64(quad[0], quad[2], quad[3]);
        // Diagonal (1,3): triangles (0,1,3) + (1,2,3).
        let d13 = triangle_area_f64(quad[0], quad[1], quad[3])
            + triangle_area_f64(quad[1], quad[2], quad[3]);
        let expected = d02.min(d13);
        assert!(
            (d02 - d13).abs() > 1.0e-3,
            "diagonals must differ for a real test"
        );

        let fill = triangulate_min_area(&quad).expect("quad fills");
        assert_eq!(fill.triangle_count(), 2);
        assert!(
            (f64::from(fill.total_area) - expected).abs() < 1.0e-5,
            "chose area {} but min diagonal is {expected}",
            fill.total_area
        );
    }

    #[test]
    fn result_is_deterministic() {
        let n = 9usize;
        let ring: Vec<Vec3> = (0..n)
            .map(|i| {
                let a = core::f64::consts::TAU * (i as f64) / (n as f64);
                Vec3::new(a.cos() as f32, 0.1 * (3.0 * a).sin() as f32, a.sin() as f32)
            })
            .collect();
        let first = triangulate_min_area(&ring).expect("fills");
        let second = triangulate_min_area(&ring).expect("fills");
        assert_eq!(first, second);
    }
}
