//! Hole filling via oriented boundary-loop triangulation.
//!
//! Open shells and punched meshes expose boundary loops - the ordered rings of
//! open edges recovered by [`extract_boundary_loops`](crate::collider::extract_boundary_loops).
//! A collision cooker usually wants those holes closed so that signed-distance
//! fields, ray-parity inside tests and volume integrals have a watertight
//! surface to work with. AAA cookers (`PhysX`, `Jolt`) offer exactly this "cap
//! holes" pass.
//!
//! Each loop is triangulated independently:
//!
//! - the loop's best-fit plane normal is estimated with Newell's method (robust
//!   for near-planar rings and well defined even when the ring is not perfectly
//!   flat),
//! - the ring is projected onto that plane and triangulated by ear clipping,
//!   which handles convex and concave polygons alike, and
//! - each emitted patch triangle is wound so its boundary edges are the
//!   *reverse* of the loop's open edges. For consistently oriented input this
//!   makes the patched mesh a closed, consistently oriented 2-manifold.
//!
//! The mesh is welded first (through `extract_boundary_loops`) so hairline
//! cracks are not treated as holes, and the returned patch triangles index into
//! that welded vertex list. Pathological loops that cannot be triangulated
//! (fewer than three distinct vertices, a degenerate normal, or ear clipping
//! that stalls on a self-touching ring) are reported as skipped rather than
//! guessed at. This is standard triangle-soup geometry with no coupling to the
//! collision pipeline, and nothing here is derived from Unreal Engine source.

use glam::Vec3;

use crate::collider::boundary_loops::{extract_boundary_loops, BoundaryLoopParams};

/// Loop normals shorter than this (Newell area) are treated as degenerate.
const NORMAL_EPSILON: f32 = 1.0e-12;

/// Signed area below which a projected ring is treated as degenerate.
const AREA_EPSILON: f32 = 1.0e-14;

/// Tuning for [`fill_boundary_loops`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FillHolesParams {
    /// Position tolerance used to weld near-coincident vertices first.
    pub weld_epsilon: f32,
    /// Optional cap on loop length; loops with more edges than this are skipped
    /// instead of filled. `None` fills loops of any size.
    pub max_loop_edges: Option<usize>,
}

impl Default for FillHolesParams {
    /// A `1e-5` weld tolerance with no loop-size cap.
    fn default() -> Self {
        Self {
            weld_epsilon: 1.0e-5,
            max_loop_edges: None,
        }
    }
}

/// Triangulated hole patches over the welded mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct HoleFill {
    /// Welded vertex positions the patch indices refer to. These are the same
    /// vertices [`extract_boundary_loops`](crate::collider::extract_boundary_loops)
    /// reports, so appending `fill_triangles` to the welded mesh's own indices
    /// yields the patched mesh.
    pub vertices: Vec<Vec3>,
    /// Patch triangles, wound so each loop's open edge is reversed. Indices
    /// refer to [`vertices`](HoleFill::vertices).
    pub fill_triangles: Vec<[u32; 3]>,
    /// Number of boundary loops that were successfully triangulated.
    pub filled_loops: usize,
    /// Number of boundary loops that could not be triangulated and were left
    /// open.
    pub skipped_loops: usize,
}

impl HoleFill {
    /// Number of patch triangles generated across all filled loops.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.fill_triangles.len()
    }

    /// Number of boundary loops that were triangulated.
    #[must_use]
    pub fn filled_loop_count(&self) -> usize {
        self.filled_loops
    }

    /// Number of boundary loops that were skipped.
    #[must_use]
    pub fn skipped_loop_count(&self) -> usize {
        self.skipped_loops
    }

    /// Whether every boundary loop was successfully filled (none skipped).
    #[must_use]
    pub fn filled_all(&self) -> bool {
        self.skipped_loops == 0
    }

    /// Whether no patch triangles were produced.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fill_triangles.is_empty()
    }
}

/// Triangulates every boundary loop of the welded mesh to cap its holes.
///
/// Returns `None` when the vertex or index slice is empty, when `weld_epsilon`
/// is not finite and strictly positive, or when welding collapses the mesh to
/// nothing. A closed mesh yields `Some` with no patch triangles and zero filled
/// or skipped loops. The result is deterministic.
#[must_use]
pub fn fill_boundary_loops(
    vertices: &[Vec3],
    indices: &[[u32; 3]],
    params: FillHolesParams,
) -> Option<HoleFill> {
    let loops = extract_boundary_loops(
        vertices,
        indices,
        BoundaryLoopParams {
            weld_epsilon: params.weld_epsilon,
        },
    )?;

    let verts = loops.vertices;
    let mut fill_triangles = Vec::new();
    let mut filled_loops = 0usize;
    let mut skipped_loops = 0usize;

    for ring in &loops.loops {
        match triangulate_ring(&verts, ring, params.max_loop_edges) {
            Some(patch) => {
                fill_triangles.extend(patch);
                filled_loops += 1;
            }
            None => skipped_loops += 1,
        }
    }

    Some(HoleFill {
        vertices: verts,
        fill_triangles,
        filled_loops,
        skipped_loops,
    })
}

/// Triangulates a single boundary ring, returning patch triangles indexed into
/// `verts`, or `None` when the ring cannot be triangulated.
fn triangulate_ring(
    verts: &[Vec3],
    ring: &[u32],
    max_edges: Option<usize>,
) -> Option<Vec<[u32; 3]>> {
    let n = ring.len();
    if n < 3 {
        return None;
    }
    if let Some(limit) = max_edges
        && n > limit
    {
        return None;
    }

    let normal = newell_normal(verts, ring);
    if normal.length() <= NORMAL_EPSILON {
        return None;
    }
    let (axis_u, axis_v) = plane_basis(normal.normalize());

    // Project the ring onto its best-fit plane.
    let projected: Vec<[f32; 2]> = ring
        .iter()
        .map(|&idx| {
            let p = verts[idx as usize];
            [p.dot(axis_u), p.dot(axis_v)]
        })
        .collect();

    let triangles = ear_clip(&projected)?;

    // Map projected-index triangles back to vertex ids, flipping the winding so
    // each loop edge is reversed in the patch.
    let patch = triangles
        .into_iter()
        .map(|[a, b, c]| [ring[a], ring[c], ring[b]])
        .collect();
    Some(patch)
}

/// Estimates a ring's plane normal with Newell's method.
fn newell_normal(verts: &[Vec3], ring: &[u32]) -> Vec3 {
    let n = ring.len();
    let mut normal = Vec3::ZERO;
    for i in 0..n {
        let cur = verts[ring[i] as usize];
        let next = verts[ring[(i + 1) % n] as usize];
        normal.x += (cur.y - next.y) * (cur.z + next.z);
        normal.y += (cur.z - next.z) * (cur.x + next.x);
        normal.z += (cur.x - next.x) * (cur.y + next.y);
    }
    normal
}

/// Builds an orthonormal in-plane basis for a unit `normal`.
fn plane_basis(normal: Vec3) -> (Vec3, Vec3) {
    let reference = if normal.x.abs() <= normal.y.abs() && normal.x.abs() <= normal.z.abs() {
        Vec3::X
    } else if normal.y.abs() <= normal.z.abs() {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let axis_u = normal.cross(reference).normalize();
    let axis_v = normal.cross(axis_u);
    (axis_u, axis_v)
}

/// Ear-clips a simple polygon, returning triangles as index triples into the
/// input, following the polygon's own winding. Returns `None` for a degenerate
/// ring or when clipping stalls on a self-touching polygon.
fn ear_clip(poly: &[[f32; 2]]) -> Option<Vec<[usize; 3]>> {
    let n = poly.len();
    if n < 3 {
        return None;
    }
    let area = signed_area(poly);
    if area.abs() <= AREA_EPSILON {
        return None;
    }
    let ccw = area > 0.0;

    let mut remaining: Vec<usize> = (0..n).collect();
    let mut triangles = Vec::with_capacity(n - 2);

    while remaining.len() > 3 {
        let m = remaining.len();
        let mut clipped = false;
        for k in 0..m {
            let i_prev = remaining[(k + m - 1) % m];
            let i_cur = remaining[k];
            let i_next = remaining[(k + 1) % m];
            let a = poly[i_prev];
            let b = poly[i_cur];
            let c = poly[i_next];

            // Convex corner for this polygon winding?
            let turn = orient(a, b, c);
            let convex = if ccw { turn > 0.0 } else { turn < 0.0 };
            if !convex {
                continue;
            }

            // Reject the ear if any other remaining vertex lies strictly inside.
            let mut blocked = false;
            for &j in &remaining {
                if j == i_prev || j == i_cur || j == i_next {
                    continue;
                }
                if strictly_inside(poly[j], a, b, c) {
                    blocked = true;
                    break;
                }
            }
            if blocked {
                continue;
            }

            triangles.push([i_prev, i_cur, i_next]);
            remaining.remove(k);
            clipped = true;
            break;
        }
        if !clipped {
            return None;
        }
    }

    triangles.push([remaining[0], remaining[1], remaining[2]]);
    Some(triangles)
}

/// Twice the signed area of a 2D polygon (positive for counter-clockwise).
fn signed_area(poly: &[[f32; 2]]) -> f32 {
    let n = poly.len();
    let mut sum = 0.0f32;
    for i in 0..n {
        let p = poly[i];
        let q = poly[(i + 1) % n];
        sum += p[0] * q[1] - q[0] * p[1];
    }
    0.5 * sum
}

/// Orientation of the ordered triple `(a, b, c)`: positive is a left turn.
fn orient(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Whether `p` lies strictly inside triangle `(a, b, c)` (boundary excluded).
fn strictly_inside(p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> bool {
    let d1 = orient(a, b, p);
    let d2 = orient(b, c, p);
    let d3 = orient(c, a, p);
    (d1 > 0.0 && d2 > 0.0 && d3 > 0.0) || (d1 < 0.0 && d2 < 0.0 && d3 < 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::topology::analyze_topology;

    /// A unit cube centred at the origin, outward wound and closed.
    fn unit_cube() -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let v = vec![
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.5, -0.5, -0.5),
            Vec3::new(0.5, 0.5, -0.5),
            Vec3::new(-0.5, 0.5, -0.5),
            Vec3::new(-0.5, -0.5, 0.5),
            Vec3::new(0.5, -0.5, 0.5),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-0.5, 0.5, 0.5),
        ];
        let f = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];
        (v, f)
    }

    /// The unit cube with its `+z` face (triangles `[4,5,6]`,`[4,6,7]`) removed,
    /// leaving a single square boundary loop.
    fn cube_missing_top() -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let (v, mut f) = unit_cube();
        f.retain(|t| *t != [4, 5, 6] && *t != [4, 6, 7]);
        (v, f)
    }

    #[test]
    fn empty_or_invalid_input_is_rejected() {
        let (v, f) = unit_cube();
        let p = FillHolesParams::default();
        assert!(fill_boundary_loops(&[], &f, p).is_none());
        assert!(fill_boundary_loops(&v, &[], p).is_none());
        assert!(fill_boundary_loops(
            &v,
            &f,
            FillHolesParams {
                weld_epsilon: 0.0,
                ..p
            }
        )
        .is_none());
        assert!(fill_boundary_loops(
            &v,
            &f,
            FillHolesParams {
                weld_epsilon: -1.0,
                ..p
            }
        )
        .is_none());
    }

    #[test]
    fn closed_cube_produces_no_patches() {
        let (v, f) = unit_cube();
        let fill = fill_boundary_loops(&v, &f, FillHolesParams::default()).unwrap();
        assert!(fill.is_empty());
        assert_eq!(fill.triangle_count(), 0);
        assert_eq!(fill.filled_loop_count(), 0);
        assert_eq!(fill.skipped_loop_count(), 0);
        assert!(fill.filled_all());
    }

    #[test]
    fn open_quad_fills_with_two_triangles() {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let tris = vec![[0, 1, 2], [0, 2, 3]];
        let fill = fill_boundary_loops(&verts, &tris, FillHolesParams::default()).unwrap();
        assert_eq!(fill.filled_loop_count(), 1);
        assert_eq!(fill.triangle_count(), 2);
        assert!(fill.filled_all());
    }

    #[test]
    fn capped_cube_is_watertight() {
        let (v, f) = cube_missing_top();
        let fill = fill_boundary_loops(&v, &f, FillHolesParams::default()).unwrap();
        assert_eq!(fill.filled_loop_count(), 1);
        assert_eq!(fill.triangle_count(), 2);

        // Patch the mesh and confirm it is a closed, consistently oriented
        // 2-manifold again.
        let mut patched = f.clone();
        patched.extend(fill.fill_triangles.iter().copied());
        let topo = analyze_topology(&fill.vertices, &patched, 1.0e-5).unwrap();
        assert!(topo.is_closed(), "patched cube must be watertight");
        assert!(
            topo.is_watertight_manifold(),
            "patch winding must stay consistent with the shell"
        );
        assert_eq!(topo.euler_characteristic, 2);
    }

    #[test]
    fn concave_quad_picks_the_valid_diagonal() {
        // A concave (one reflex vertex) quad, tiled by two triangles.
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(4.0, 4.0, 0.0),
            Vec3::new(3.0, 1.0, 0.0),
        ];
        let tris = vec![[3, 0, 1], [1, 2, 3]];
        let fill = fill_boundary_loops(&verts, &tris, FillHolesParams::default()).unwrap();
        assert_eq!(fill.filled_loop_count(), 1);
        assert_eq!(fill.triangle_count(), 2);
        assert!(fill.filled_all());

        // A valid, non-overlapping triangulation of a concave polygon must tile
        // it exactly: every patch triangle is non-degenerate and their areas sum
        // to the polygon area (here 4.0). A diagonal that left the polygon would
        // over- or under-count that total.
        let mut total_area = 0.0_f32;
        for t in &fill.fill_triangles {
            let a = fill.vertices[t[0] as usize];
            let b = fill.vertices[t[1] as usize];
            let c = fill.vertices[t[2] as usize];
            let area = 0.5 * (b - a).cross(c - a).length();
            assert!(area > 1.0e-6, "degenerate patch triangle {t:?}");
            total_area += area;
        }
        assert!(
            (total_area - 4.0).abs() < 1.0e-4,
            "patch area {total_area} should tile the quad (4.0)"
        );
    }

    #[test]
    fn convex_hexagon_fan_fills_and_closes() {
        // A flat fan disk: centre vertex 0 plus a convex hexagon rim.
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(-1.0, 2.0, 0.0),
            Vec3::new(-2.0, 0.0, 0.0),
            Vec3::new(-1.0, -2.0, 0.0),
            Vec3::new(1.0, -2.0, 0.0),
        ];
        let tris = vec![
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 5],
            [0, 5, 6],
            [0, 6, 1],
        ];
        let fill = fill_boundary_loops(&verts, &tris, FillHolesParams::default()).unwrap();
        assert_eq!(fill.filled_loop_count(), 1);
        // A hexagonal ring triangulates into four patch triangles.
        assert_eq!(fill.triangle_count(), 4);

        let mut patched = tris.clone();
        patched.extend(fill.fill_triangles.iter().copied());
        let topo = analyze_topology(&fill.vertices, &patched, 1.0e-5).unwrap();
        assert!(topo.is_closed());
    }

    #[test]
    fn non_planar_loop_still_fills() {
        // The open quad, but with one corner lifted out of the plane.
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.3),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let tris = vec![[0, 1, 2], [0, 2, 3]];
        let fill = fill_boundary_loops(&verts, &tris, FillHolesParams::default()).unwrap();
        assert_eq!(fill.triangle_count(), 2);
        assert!(fill.filled_all());
    }

    #[test]
    fn loop_size_cap_skips_large_loops() {
        let (v, f) = cube_missing_top();
        let fill = fill_boundary_loops(
            &v,
            &f,
            FillHolesParams {
                max_loop_edges: Some(3),
                ..FillHolesParams::default()
            },
        )
        .unwrap();
        // The single boundary loop has four edges and exceeds the cap.
        assert_eq!(fill.filled_loop_count(), 0);
        assert_eq!(fill.skipped_loop_count(), 1);
        assert!(fill.is_empty());
        assert!(!fill.filled_all());
    }

    #[test]
    fn result_is_deterministic() {
        let (v, f) = cube_missing_top();
        let p = FillHolesParams::default();
        let a = fill_boundary_loops(&v, &f, p).unwrap();
        let b = fill_boundary_loops(&v, &f, p).unwrap();
        assert_eq!(a, b);
    }
}
