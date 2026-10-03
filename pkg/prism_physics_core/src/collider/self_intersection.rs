//! Self-intersection detection for triangle meshes.
//!
//! A mesh whose faces pierce each other breaks the assumptions that
//! signed-distance cooking, volume integration and convex decomposition rely
//! on, so AAA cookers (`PhysX`, `Jolt`, Chaos) surface self-intersections as a
//! first-class authoring warning. This module reports them as pairs of welded
//! triangle indices.
//!
//! Detection is two-phase:
//!
//! - a uniform spatial-hash broad phase buckets triangle AABBs into grid cells
//!   so only triangles sharing a cell are tested, and
//! - a Moeller-Trumbore segment-triangle narrow phase flags a pair when any
//!   edge of one triangle pierces the interior of the other.
//!
//! Triangles that share a welded vertex (legitimately adjacent faces) are never
//! reported. The edge-pierces-face test is exact for *transversal* crossings,
//! the dominant real defect; perfectly coplanar overlaps are intentionally out
//! of scope and are handled upstream by welding and duplicate-triangle removal.
//!
//! Near-coincident vertices are welded first (reusing
//! [`weld_mesh`](crate::collider::weld_mesh)). This is pure triangle-soup
//! geometry with no coupling to the collision pipeline, and nothing here is
//! derived from Unreal Engine source.

use glam::Vec3;
use std::collections::{HashMap, HashSet};

use crate::collider::weld::{weld_mesh, WeldParams};

/// Shortest cross-product length below which a triangle is treated as
/// degenerate and skipped.
const DEGENERATE_EPSILON: f32 = 1.0e-12;

/// Relative tolerance for the barycentric / segment-parameter inclusion tests.
const HIT_EPSILON: f32 = 1.0e-6;

/// A pair of welded triangle indices whose faces intersect, with `tri_a < tri_b`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct IntersectingPair {
    /// Lower welded triangle index.
    pub tri_a: u32,
    /// Higher welded triangle index.
    pub tri_b: u32,
}

/// Tuning for [`detect_self_intersections`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelfIntersectionParams {
    /// Position tolerance used to weld near-coincident vertices before testing.
    pub weld_epsilon: f32,
    /// Broad-phase grid cell size. `None` picks a size from the mean triangle
    /// AABB extent, which is a robust default.
    pub cell_size: Option<f32>,
}

impl Default for SelfIntersectionParams {
    /// A `1e-5` weld tolerance and an automatically chosen cell size.
    fn default() -> Self {
        Self {
            weld_epsilon: 1.0e-5,
            cell_size: None,
        }
    }
}

/// The self-intersecting triangle pairs of a mesh.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelfIntersectionReport {
    /// Welded triangle count that was tested.
    pub triangles: usize,
    /// Intersecting pairs, sorted and de-duplicated.
    pub pairs: Vec<IntersectingPair>,
}

impl SelfIntersectionReport {
    /// Whether the mesh has any self-intersection.
    #[must_use]
    pub fn intersects(&self) -> bool {
        !self.pairs.is_empty()
    }

    /// Number of intersecting pairs.
    #[must_use]
    pub fn count(&self) -> usize {
        self.pairs.len()
    }

    /// Whether no self-intersections were found.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}

/// Detects transversal self-intersections in a triangle mesh.
///
/// Returns `None` when `vertices` or `indices` is empty, when `weld_epsilon` is
/// not finite and strictly positive, when an explicit `cell_size` is not finite
/// and strictly positive, or when welding leaves no non-degenerate triangle.
#[must_use]
pub fn detect_self_intersections(
    vertices: &[Vec3],
    indices: &[[u32; 3]],
    params: SelfIntersectionParams,
) -> Option<SelfIntersectionReport> {
    if vertices.is_empty()
        || indices.is_empty()
        || !(params.weld_epsilon.is_finite() && params.weld_epsilon > 0.0)
    {
        return None;
    }
    if let Some(cell) = params.cell_size
        && !(cell.is_finite() && cell > 0.0)
    {
        return None;
    }

    let welded = weld_mesh(
        vertices,
        indices,
        WeldParams {
            position_epsilon: params.weld_epsilon,
            drop_duplicate_triangles: true,
        },
    )?;
    if welded.indices.is_empty() {
        return None;
    }

    let verts = &welded.vertices;
    let tris = &welded.indices;

    // Precompute AABBs and skip degenerate triangles.
    let mut aabbs: Vec<Option<(Vec3, Vec3)>> = Vec::with_capacity(tris.len());
    let mut mean_extent = 0.0f32;
    let mut counted = 0usize;
    for tri in tris {
        if let Some(bounds) = triangle_aabb(verts, *tri) {
            let ext = (bounds.1 - bounds.0).max_element();
            mean_extent += ext;
            counted += 1;
            aabbs.push(Some(bounds));
        } else {
            aabbs.push(None);
        }
    }
    if counted == 0 {
        return None;
    }
    mean_extent /= counted as f32;

    let cell_size = params.cell_size.unwrap_or_else(|| mean_extent.max(1.0e-4));
    let inv_cell = 1.0 / cell_size;

    // Broad phase: bucket triangle AABBs into grid cells.
    let mut grid: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
    for (idx, bounds) in aabbs.iter().enumerate() {
        let Some((lo, hi)) = bounds else {
            continue;
        };
        for cell in cell_range(*lo, *hi, inv_cell) {
            grid.entry(cell).or_default().push(idx as u32);
        }
    }

    // Collect candidate pairs that share a cell (de-duplicated).
    let mut candidates: HashSet<(u32, u32)> = HashSet::new();
    for bucket in grid.values() {
        for i in 0..bucket.len() {
            for j in (i + 1)..bucket.len() {
                let a = bucket[i];
                let b = bucket[j];
                candidates.insert(if a < b { (a, b) } else { (b, a) });
            }
        }
    }

    // Narrow phase.
    let mut pairs: Vec<IntersectingPair> = Vec::new();
    for (a, b) in candidates {
        let ta = tris[a as usize];
        let tb = tris[b as usize];
        if shares_vertex(ta, tb) {
            continue;
        }
        // AABB reject.
        let (Some((alo, ahi)), Some((blo, bhi))) = (aabbs[a as usize], aabbs[b as usize]) else {
            continue;
        };
        if !aabbs_overlap(alo, ahi, blo, bhi) {
            continue;
        }
        if triangles_intersect(verts, ta, tb) {
            pairs.push(IntersectingPair { tri_a: a, tri_b: b });
        }
    }

    pairs.sort_unstable();

    Some(SelfIntersectionReport {
        triangles: tris.len(),
        pairs,
    })
}

/// Axis-aligned bounds of one triangle, or `None` when degenerate / out of range.
fn triangle_aabb(verts: &[Vec3], tri: [u32; 3]) -> Option<(Vec3, Vec3)> {
    let n = verts.len();
    let i0 = tri[0] as usize;
    let i1 = tri[1] as usize;
    let i2 = tri[2] as usize;
    if i0 >= n || i1 >= n || i2 >= n {
        return None;
    }
    let a = verts[i0];
    let b = verts[i1];
    let c = verts[i2];
    if (b - a).cross(c - a).length_squared() <= DEGENERATE_EPSILON {
        return None;
    }
    Some((a.min(b).min(c), a.max(b).max(c)))
}

/// Enumerates the integer grid cells overlapped by an AABB.
fn cell_range(lo: Vec3, hi: Vec3, inv_cell: f32) -> Vec<(i32, i32, i32)> {
    let min = [
        (lo.x * inv_cell).floor() as i32,
        (lo.y * inv_cell).floor() as i32,
        (lo.z * inv_cell).floor() as i32,
    ];
    let max = [
        (hi.x * inv_cell).floor() as i32,
        (hi.y * inv_cell).floor() as i32,
        (hi.z * inv_cell).floor() as i32,
    ];
    let mut cells = Vec::new();
    for x in min[0]..=max[0] {
        for y in min[1]..=max[1] {
            for z in min[2]..=max[2] {
                cells.push((x, y, z));
            }
        }
    }
    cells
}

/// Whether two triangles share any vertex index.
fn shares_vertex(a: [u32; 3], b: [u32; 3]) -> bool {
    a.iter().any(|x| b.contains(x))
}

fn aabbs_overlap(alo: Vec3, ahi: Vec3, blo: Vec3, bhi: Vec3) -> bool {
    alo.x <= bhi.x
        && ahi.x >= blo.x
        && alo.y <= bhi.y
        && ahi.y >= blo.y
        && alo.z <= bhi.z
        && ahi.z >= blo.z
}

/// Whether two triangles intersect transversally: an edge of one pierces the
/// other.
fn triangles_intersect(verts: &[Vec3], a: [u32; 3], b: [u32; 3]) -> bool {
    let pa = [
        verts[a[0] as usize],
        verts[a[1] as usize],
        verts[a[2] as usize],
    ];
    let pb = [
        verts[b[0] as usize],
        verts[b[1] as usize],
        verts[b[2] as usize],
    ];
    for (i, j) in [(0, 1), (1, 2), (2, 0)] {
        if segment_triangle_hit(pa[i], pa[j], pb[0], pb[1], pb[2]) {
            return true;
        }
        if segment_triangle_hit(pb[i], pb[j], pa[0], pa[1], pa[2]) {
            return true;
        }
    }
    false
}

/// Moeller-Trumbore segment-vs-triangle intersection: whether the segment
/// `[p0, p1]` crosses triangle `(a, b, c)`.
fn segment_triangle_hit(p0: Vec3, p1: Vec3, a: Vec3, b: Vec3, c: Vec3) -> bool {
    let dir = p1 - p0;
    let e1 = b - a;
    let e2 = c - a;
    let pvec = dir.cross(e2);
    let det = e1.dot(pvec);
    if det.abs() <= DEGENERATE_EPSILON {
        return false; // segment parallel to the triangle plane
    }
    let inv_det = 1.0 / det;
    let tvec = p0 - a;
    let u = tvec.dot(pvec) * inv_det;
    if !(-HIT_EPSILON..=1.0 + HIT_EPSILON).contains(&u) {
        return false;
    }
    let qvec = tvec.cross(e1);
    let v = dir.dot(qvec) * inv_det;
    if v < -HIT_EPSILON || u + v > 1.0 + HIT_EPSILON {
        return false;
    }
    let t = e2.dot(qvec) * inv_det;
    (-HIT_EPSILON..=1.0 + HIT_EPSILON).contains(&t)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit cube centred at the origin, outward wound, with shared vertices.
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

    #[test]
    fn empty_or_invalid_input_is_rejected() {
        assert!(
            detect_self_intersections(&[], &[[0, 1, 2]], SelfIntersectionParams::default())
                .is_none()
        );
        assert!(
            detect_self_intersections(&[Vec3::ZERO], &[], SelfIntersectionParams::default())
                .is_none()
        );
        let bad = SelfIntersectionParams {
            weld_epsilon: 0.0,
            cell_size: None,
        };
        let (v, f) = unit_cube();
        assert!(detect_self_intersections(&v, &f, bad).is_none());
        let bad_cell = SelfIntersectionParams {
            weld_epsilon: 1.0e-5,
            cell_size: Some(-1.0),
        };
        assert!(detect_self_intersections(&v, &f, bad_cell).is_none());
    }

    #[test]
    fn clean_cube_has_no_self_intersections() {
        let (v, f) = unit_cube();
        let report = detect_self_intersections(&v, &f, SelfIntersectionParams::default()).unwrap();
        assert!(!report.intersects());
        assert_eq!(report.count(), 0);
        assert_eq!(report.triangles, 12);
    }

    #[test]
    fn two_crossing_triangles_are_detected() {
        // A horizontal triangle and a vertical triangle that pierce through
        // each other near the origin; they share no vertices.
        let verts = vec![
            // Triangle A: in the z = 0 plane, spanning the origin.
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            // Triangle B: in the y = 0 plane, straddling z = 0.
            Vec3::new(0.0, -0.5, -1.0),
            Vec3::new(0.0, -0.5, 1.0),
            Vec3::new(0.0, 0.5, 0.0),
        ];
        let faces = vec![[0, 1, 2], [3, 4, 5]];
        let report =
            detect_self_intersections(&verts, &faces, SelfIntersectionParams::default()).unwrap();
        assert!(report.intersects());
        assert_eq!(report.count(), 1);
        assert_eq!(report.pairs[0], IntersectingPair { tri_a: 0, tri_b: 1 });
    }

    #[test]
    fn separated_triangles_do_not_intersect() {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(1.0, 0.0, 5.0),
            Vec3::new(0.0, 1.0, 5.0),
        ];
        let faces = vec![[0, 1, 2], [3, 4, 5]];
        let report =
            detect_self_intersections(&verts, &faces, SelfIntersectionParams::default()).unwrap();
        assert!(!report.intersects());
    }

    #[test]
    fn edge_adjacent_triangles_are_not_flagged() {
        // Two triangles sharing edge (0,1): a legitimate fold, not a defect.
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 1.0, 0.0),
            Vec3::new(0.5, -1.0, 0.3),
        ];
        let faces = vec![[0, 1, 2], [1, 0, 3]];
        let report =
            detect_self_intersections(&verts, &faces, SelfIntersectionParams::default()).unwrap();
        assert!(!report.intersects());
    }

    #[test]
    fn explicit_cell_size_matches_auto() {
        let verts = vec![
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -0.5, -1.0),
            Vec3::new(0.0, -0.5, 1.0),
            Vec3::new(0.0, 0.5, 0.0),
        ];
        let faces = vec![[0, 1, 2], [3, 4, 5]];
        let auto =
            detect_self_intersections(&verts, &faces, SelfIntersectionParams::default()).unwrap();
        let fixed = detect_self_intersections(
            &verts,
            &faces,
            SelfIntersectionParams {
                weld_epsilon: 1.0e-5,
                cell_size: Some(0.25),
            },
        )
        .unwrap();
        assert_eq!(auto.pairs, fixed.pairs);
    }

    #[test]
    fn result_is_deterministic_and_sorted() {
        // Three mutually crossing triangles arranged so several pairs intersect.
        let verts = vec![
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -0.5, -1.0),
            Vec3::new(0.0, -0.5, 1.0),
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::new(-1.0, 0.0, -0.5),
            Vec3::new(1.0, 0.0, -0.5),
            Vec3::new(0.0, 0.0, 0.8),
        ];
        let faces = vec![[0, 1, 2], [3, 4, 5], [6, 7, 8]];
        let report =
            detect_self_intersections(&verts, &faces, SelfIntersectionParams::default()).unwrap();
        assert!(report.count() >= 1);
        for pair in report.pairs.windows(2) {
            assert!(pair[0] < pair[1]);
        }
        for p in &report.pairs {
            assert!(p.tri_a < p.tri_b);
        }
    }
}
