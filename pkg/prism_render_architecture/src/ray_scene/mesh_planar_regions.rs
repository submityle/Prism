//! Coplanar face segmentation of a triangle mesh for the `CPU` golden path.
//!
//! Flat-region decomposition is a workhorse of `AAA` content pipelines: planar
//! UV projection, decal footprints, lightmap chart seeding, collision-hull
//! simplification, and "merge flat triangles" passes all begin by grouping the
//! faces that share a plane. This module grows maximal connected clusters of
//! near-coplanar triangles by edge adjacency: starting from a seed face it
//! floods into neighbours whose face normal stays within a caller-supplied
//! cosine of the seed normal *and* whose centroid lies within a distance
//! tolerance of the seed plane. Holding the seed plane fixed for the whole
//! cluster prevents the plane from drifting across a gently curved strip.
//!
//! The caller passes the cosine threshold directly (for example `0.999` for
//! roughly `2.5` degrees), so no inverse-trigonometric function is needed. Face
//! normals come from an edge cross product and are normalized with a square
//! root; all plane arithmetic accumulates in `f64`. Degenerate (zero-area)
//! faces cannot anchor a plane and are emitted as their own singleton regions.
//!
//! [`planar_regions`] returns [`PlanarRegions`], carrying a per-face region id,
//! the region count, and the fixed plane of each region.

use alloc::collections::VecDeque;

use super::triangle_mesh::TriangleMesh;

/// An oriented plane: a unit normal and the signed offset `d` such that points
/// `p` on the plane satisfy `dot(normal, p) = offset`.
///
/// Degenerate seed faces (zero area) yield a zero normal and zero offset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegionPlane {
    /// Unit normal of the region's seed face, or zero for a degenerate seed.
    normal: [f32; 3],
    /// Signed plane offset `dot(normal, seed_centroid)`.
    offset: f32,
}

impl RegionPlane {
    /// Unit normal of the region's plane (zero for a degenerate seed).
    pub fn normal(&self) -> [f32; 3] {
        self.normal
    }

    /// Signed plane offset `dot(normal, seed_centroid)`.
    pub fn offset(&self) -> f32 {
        self.offset
    }
}

/// Result of segmenting a mesh into connected near-coplanar face clusters.
///
/// Produced by [`planar_regions`]. Region ids are dense in `0..region_count`
/// and assigned in ascending seed-face order for determinism.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanarRegions {
    /// Region id for each triangle, indexed by triangle order.
    face_region: Vec<u32>,
    /// Number of distinct regions (`0` only for a mesh with no triangles).
    region_count: u32,
    /// Fixed plane of each region, indexed by region id.
    planes: Vec<RegionPlane>,
}

impl PlanarRegions {
    /// Region id for each triangle, indexed by triangle order.
    pub fn face_region(&self) -> &[u32] {
        &self.face_region
    }

    /// Number of distinct regions.
    pub fn region_count(&self) -> u32 {
        self.region_count
    }

    /// Fixed plane of each region, indexed by region id.
    pub fn planes(&self) -> &[RegionPlane] {
        &self.planes
    }
}

/// Per-face geometry cached for the flood fill: unit normal, centroid, and a
/// degenerate flag set when the triangle has (near) zero area.
struct FaceGeometry {
    /// Unit face normal, or zero when degenerate.
    normal: [f64; 3],
    /// Face centroid (average of the three vertices).
    centroid: [f64; 3],
    /// Whether the triangle area is below the normalization epsilon.
    degenerate: bool,
}

/// Computes the cached geometry (normal, centroid, degeneracy) for one triangle.
fn face_geometry(mesh: &TriangleMesh, tri: [u32; 3]) -> FaceGeometry {
    let positions = mesh.positions();
    let p = |i: u32| -> [f64; 3] {
        let v = positions[i as usize];
        [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])]
    };
    let a = p(tri[0]);
    let b = p(tri[1]);
    let c = p(tri[2]);
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let len = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    let centroid = [
        (a[0] + b[0] + c[0]) / 3.0,
        (a[1] + b[1] + c[1]) / 3.0,
        (a[2] + b[2] + c[2]) / 3.0,
    ];
    if len <= 1e-12 {
        FaceGeometry { normal: [0.0, 0.0, 0.0], centroid, degenerate: true }
    } else {
        FaceGeometry {
            normal: [cross[0] / len, cross[1] / len, cross[2] / len],
            centroid,
            degenerate: false,
        }
    }
}

/// Builds face adjacency keyed by undirected edge: every pair of faces sharing
/// an edge becomes neighbours. Neighbour lists are sorted and de-duplicated for
/// deterministic traversal.
fn build_adjacency(mesh: &TriangleMesh) -> Vec<Vec<usize>> {
    use std::collections::HashMap;
    let indices = mesh.indices();
    let mut edge_faces: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (face, tri) in indices.iter().enumerate() {
        for k in 0..3 {
            let v0 = tri[k];
            let v1 = tri[(k + 1) % 3];
            let key = if v0 <= v1 { (v0, v1) } else { (v1, v0) };
            edge_faces.entry(key).or_default().push(face);
        }
    }
    let mut adjacency = vec![Vec::new(); indices.len()];
    for faces in edge_faces.values() {
        for (i, &fa) in faces.iter().enumerate() {
            for &fb in faces.iter().skip(i + 1) {
                if fa != fb {
                    adjacency[fa].push(fb);
                    adjacency[fb].push(fa);
                }
            }
        }
    }
    for list in &mut adjacency {
        list.sort_unstable();
        list.dedup();
    }
    adjacency
}

/// Dot product of two 3-vectors in `f64`.
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Segments a mesh into connected clusters of near-coplanar faces.
///
/// `cos_threshold` is the minimum dot product (both normals unit) for two faces
/// to count as coplanar; values near `1.0` are stricter. `distance_tolerance`
/// bounds how far a candidate face's centroid may sit from the region's seed
/// plane. Both thresholds are clamped to sane ranges internally.
///
/// Faces are visited in ascending order so region ids are deterministic. A mesh
/// with no triangles yields an empty [`PlanarRegions`] with `region_count` `0`.
pub fn planar_regions(
    mesh: &TriangleMesh,
    cos_threshold: f32,
    distance_tolerance: f32,
) -> PlanarRegions {
    let face_count = mesh.indices().len();
    if face_count == 0 {
        return PlanarRegions { face_region: Vec::new(), region_count: 0, planes: Vec::new() };
    }

    let cos_min = f64::from(cos_threshold).clamp(-1.0, 1.0);
    let dist_tol = f64::from(distance_tolerance).max(0.0);

    let geometry: Vec<FaceGeometry> =
        mesh.indices().iter().map(|&tri| face_geometry(mesh, tri)).collect();
    let adjacency = build_adjacency(mesh);

    let unassigned = u32::MAX;
    let mut face_region = vec![unassigned; face_count];
    let mut planes: Vec<RegionPlane> = Vec::new();
    let mut queue: VecDeque<usize> = VecDeque::new();

    for seed in 0..face_count {
        if face_region[seed] != unassigned {
            continue;
        }
        let region_id = planes.len() as u32;
        let seed_geo = &geometry[seed];
        let seed_normal = seed_geo.normal;
        let seed_offset = dot(seed_normal, seed_geo.centroid);
        planes.push(RegionPlane {
            normal: [seed_normal[0] as f32, seed_normal[1] as f32, seed_normal[2] as f32],
            offset: seed_offset as f32,
        });

        face_region[seed] = region_id;
        // A degenerate seed cannot support coplanarity tests; it stays a
        // singleton region and never recruits neighbours.
        if seed_geo.degenerate {
            continue;
        }

        queue.clear();
        queue.push_back(seed);
        while let Some(current) = queue.pop_front() {
            for &neighbour in &adjacency[current] {
                if face_region[neighbour] != unassigned {
                    continue;
                }
                let geo = &geometry[neighbour];
                if geo.degenerate {
                    continue;
                }
                if dot(seed_normal, geo.normal) < cos_min {
                    continue;
                }
                let plane_distance = (dot(seed_normal, geo.centroid) - seed_offset).abs();
                if plane_distance > dist_tol {
                    continue;
                }
                face_region[neighbour] = region_id;
                queue.push_back(neighbour);
            }
        }
    }

    let region_count = planes.len() as u32;
    PlanarRegions { face_region, region_count, planes }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::triangle_mesh::TriangleMesh;

    /// Builds an index-only triangle mesh from positions (no normals/uvs).
    fn mesh(positions: Vec<[f32; 3]>, indices: Vec<[u32; 3]>) -> TriangleMesh {
        TriangleMesh::new(positions, Vec::new(), Vec::new(), indices).unwrap()
    }

    #[test]
    fn empty_mesh_has_no_regions() {
        let m = mesh(Vec::new(), Vec::new());
        let r = planar_regions(&m, 0.999, 1e-4);
        assert_eq!(r.region_count(), 0);
        assert!(r.face_region().is_empty());
        assert!(r.planes().is_empty());
    }

    #[test]
    fn two_coplanar_triangles_form_one_region() {
        // Unit quad split into two triangles sharing the diagonal edge.
        let m = mesh(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            vec![[0, 1, 2], [0, 2, 3]],
        );
        let r = planar_regions(&m, 0.999, 1e-4);
        assert_eq!(r.region_count(), 1);
        assert_eq!(r.face_region(), &[0, 0]);
        // Plane normal is +Z (both triangles wound CCW in the z=0 plane).
        let n = r.planes()[0].normal();
        assert!((n[2].abs() - 1.0).abs() < 1e-5, "normal {n:?}");
        assert!(r.planes()[0].offset().abs() < 1e-5);
    }

    #[test]
    fn perpendicular_faces_split_into_two_regions() {
        // A floor triangle (z=0) and a wall triangle (y=0) sharing an edge.
        let m = mesh(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            vec![[0, 1, 2], [0, 3, 1]],
        );
        let r = planar_regions(&m, 0.999, 1e-4);
        assert_eq!(r.region_count(), 2);
        assert_ne!(r.face_region()[0], r.face_region()[1]);
    }

    #[test]
    fn loose_threshold_merges_slightly_bent_strip() {
        // Two triangles sharing an edge with a tiny fold; a loose cosine and a
        // generous distance tolerance merge them, a strict one does not.
        let m = mesh(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.02],
            ],
            vec![[0, 1, 2], [0, 2, 3]],
        );
        let loose = planar_regions(&m, 0.99, 0.1);
        assert_eq!(loose.region_count(), 1);
        let strict = planar_regions(&m, 0.99999, 1e-5);
        assert_eq!(strict.region_count(), 2);
    }

    #[test]
    fn parallel_but_offset_faces_do_not_merge() {
        // Two coplanar-normal triangles on different parallel planes (z=0 and
        // z=5) are disconnected and must land in separate regions.
        let m = mesh(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 5.0],
                [1.0, 0.0, 5.0],
                [0.0, 1.0, 5.0],
            ],
            vec![[0, 1, 2], [3, 4, 5]],
        );
        let r = planar_regions(&m, 0.999, 1e-4);
        assert_eq!(r.region_count(), 2);
    }

    #[test]
    fn degenerate_face_is_its_own_region() {
        // One real triangle plus a zero-area triangle (repeated vertex).
        let m = mesh(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![[0, 1, 2], [0, 0, 1]],
        );
        let r = planar_regions(&m, 0.999, 1e-4);
        assert_eq!(r.region_count(), 2);
        assert_ne!(r.face_region()[0], r.face_region()[1]);
        // The degenerate region reports a zero normal.
        let degen_region = r.face_region()[1] as usize;
        assert_eq!(r.planes()[degen_region].normal(), [0.0, 0.0, 0.0]);
    }
}
