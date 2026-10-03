//! Discrete curvature estimation for triangle meshes.
//!
//! Per-vertex mean and Gaussian curvature are the standard inputs to adaptive
//! remeshing, feature-aware simplification, wear/erosion authoring and
//! shading-rate heuristics, so AAA tool-chains expose them alongside the other
//! mesh analysis primitives. This module implements the discrete
//! differential-geometry operators of Meyer, Desbrun, Schroeder & Barr (2003):
//! the Gaussian curvature from the angle deficit over the mixed Voronoi area,
//! and the mean curvature from the cotangent-weighted Laplace-Beltrami
//! operator. Principal curvatures follow from the two invariants.
//!
//! The mesh is welded first (reusing
//! [`weld_mesh`](crate::collider::weld_mesh)) so that split vertices do not
//! break the one-ring adjacency the operators depend on; results are indexed
//! into the welded vertex list. Boundary vertices use the half-angle deficit
//! (`PI - sum`) rather than the interior (`2*PI - sum`). This is pure
//! triangle-soup geometry with no coupling to the collision pipeline, and
//! nothing here is derived from Unreal Engine source.

use core::f64::consts::PI;
use glam::Vec3;
use std::collections::HashMap;

use crate::collider::weld::{weld_mesh, WeldParams};

/// Cross-product length below which a triangle is treated as degenerate.
const DEGENERATE_EPSILON: f32 = 1.0e-12;

/// Per-vertex discrete curvature invariants.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VertexCurvature {
    /// Mean curvature `H = (kappa_1 + kappa_2) / 2`.
    pub mean: f32,
    /// Gaussian curvature `K = kappa_1 * kappa_2`.
    pub gaussian: f32,
    /// Smaller principal curvature `kappa_2`.
    pub min_principal: f32,
    /// Larger principal curvature `kappa_1`.
    pub max_principal: f32,
    /// Mixed Voronoi area associated with the vertex (the operator's local
    /// integration region).
    pub area: f32,
}

impl VertexCurvature {
    /// A flat / undefined vertex (isolated or fully degenerate neighbourhood).
    const FLAT: Self = Self {
        mean: 0.0,
        gaussian: 0.0,
        min_principal: 0.0,
        max_principal: 0.0,
        area: 0.0,
    };
}

/// Tuning for [`estimate_curvature`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurvatureParams {
    /// Position tolerance used to weld near-coincident vertices first.
    pub weld_epsilon: f32,
}

impl Default for CurvatureParams {
    /// A `1e-5` weld tolerance.
    fn default() -> Self {
        Self {
            weld_epsilon: 1.0e-5,
        }
    }
}

/// Per-vertex curvature over the welded mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct CurvatureReport {
    /// Welded vertex positions the curvatures are indexed against.
    pub vertices: Vec<Vec3>,
    /// Curvature invariants, one per entry of [`CurvatureReport::vertices`].
    pub curvatures: Vec<VertexCurvature>,
}

impl CurvatureReport {
    /// Number of vertices covered by the report.
    #[must_use]
    pub fn len(&self) -> usize {
        self.vertices.len()
    }

    /// Whether the report is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// Mean absolute mean-curvature over all vertices, a compact roughness
    /// proxy. Returns `0` for an empty report.
    #[must_use]
    pub fn average_abs_mean(&self) -> f32 {
        if self.curvatures.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .curvatures
            .iter()
            .map(|c| f64::from(c.mean.abs()))
            .sum();
        (sum / self.curvatures.len() as f64) as f32
    }
}

/// Estimates per-vertex mean, Gaussian and principal curvature over the welded
/// mesh.
///
/// Returns `None` when the vertex or index slice is empty, `weld_epsilon` is
/// not finite and positive, or welding collapses the mesh to nothing.
/// Degenerate triangles are skipped; vertices with no finite incident area are
/// reported as flat.
#[must_use]
pub fn estimate_curvature(
    vertices: &[Vec3],
    indices: &[[u32; 3]],
    params: CurvatureParams,
) -> Option<CurvatureReport> {
    if vertices.is_empty()
        || indices.is_empty()
        || !(params.weld_epsilon.is_finite() && params.weld_epsilon > 0.0)
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
    let verts = &welded.vertices;
    let tris = &welded.indices;
    let n = verts.len();
    if n == 0 || tris.is_empty() {
        return None;
    }

    // Per-vertex accumulators.
    let mut angle_sum = vec![0.0_f64; n];
    let mut mixed_area = vec![0.0_f64; n];
    // Cotangent-weighted Laplace-Beltrami vector per vertex.
    let mut laplacian = vec![[0.0_f64; 3]; n];
    // Edge -> number of incident triangles, for boundary detection.
    let mut edge_count: HashMap<(u32, u32), u32> = HashMap::new();

    for tri in tris {
        let (ia, ib, ic) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let (pa, pb, pc) = (verts[ia], verts[ib], verts[ic]);
        let cross_len = (pb - pa).cross(pc - pa).length();
        if cross_len <= DEGENERATE_EPSILON {
            continue;
        }

        // Interior angles via the law of cosines (f64 acos for determinism).
        let ang_a = corner_angle(pa, pb, pc);
        let ang_b = corner_angle(pb, pc, pa);
        let ang_c = corner_angle(pc, pa, pb);
        angle_sum[ia] += ang_a;
        angle_sum[ib] += ang_b;
        angle_sum[ic] += ang_c;

        // Cotangents of the three corners (cot = dot / cross_len).
        let cot_a = cotangent(pa, pb, pc);
        let cot_b = cotangent(pb, pc, pa);
        let cot_c = cotangent(pc, pa, pb);

        // Laplace-Beltrami: edge (i,j) is weighted by the cotangent of the
        // angle opposite it. Edge (b,c) is opposite corner a, etc.
        accumulate_edge(&mut laplacian, verts, ib, ic, cot_a);
        accumulate_edge(&mut laplacian, verts, ic, ia, cot_b);
        accumulate_edge(&mut laplacian, verts, ia, ib, cot_c);

        // Mixed Voronoi area (Meyer et al. 2003, section 3.3).
        let area = 0.5 * f64::from(cross_len);
        mixed_area[ia] += mixed_area_contribution(
            &Corner {
                apex_angle: ang_a,
                other_angles: (ang_b, ang_c),
                edge_lengths: ((pb - pa).length(), (pc - pa).length()),
                opposite_cots: (cot_b, cot_c),
            },
            area,
        );
        mixed_area[ib] += mixed_area_contribution(
            &Corner {
                apex_angle: ang_b,
                other_angles: (ang_c, ang_a),
                edge_lengths: ((pc - pb).length(), (pa - pb).length()),
                opposite_cots: (cot_c, cot_a),
            },
            area,
        );
        mixed_area[ic] += mixed_area_contribution(
            &Corner {
                apex_angle: ang_c,
                other_angles: (ang_a, ang_b),
                edge_lengths: ((pa - pc).length(), (pb - pc).length()),
                opposite_cots: (cot_a, cot_b),
            },
            area,
        );

        for &(u, v) in &[(ia, ib), (ib, ic), (ic, ia)] {
            let key = if u < v {
                (u as u32, v as u32)
            } else {
                (v as u32, u as u32)
            };
            *edge_count.entry(key).or_insert(0) += 1;
        }
    }

    // Mark boundary vertices (touch an edge with a single incident triangle).
    let mut on_boundary = vec![false; n];
    for (&(u, v), &count) in &edge_count {
        if count < 2 {
            on_boundary[u as usize] = true;
            on_boundary[v as usize] = true;
        }
    }

    let mut curvatures = vec![VertexCurvature::FLAT; n];
    for i in 0..n {
        let area = mixed_area[i];
        if area > 0.0 {
            // Gaussian curvature from the angle deficit.
            let full_angle = if on_boundary[i] { PI } else { 2.0 * PI };
            let gaussian = (full_angle - angle_sum[i]) / area;

            // Mean curvature magnitude from the Laplace-Beltrami operator:
            // |K(x)| = 2 H, with K(x) = 1/(2A) * sum cot-weighted edge vectors.
            let lap = laplacian[i];
            let lap_len = (lap[0] * lap[0] + lap[1] * lap[1] + lap[2] * lap[2]).sqrt();
            let mean = 0.5 * (lap_len / (2.0 * area));

            // Principal curvatures from the two invariants.
            let disc = (mean * mean - gaussian).max(0.0).sqrt();
            let k_max = mean + disc;
            let k_min = mean - disc;

            curvatures[i] = VertexCurvature {
                mean: mean as f32,
                gaussian: gaussian as f32,
                min_principal: k_min as f32,
                max_principal: k_max as f32,
                area: area as f32,
            };
        }
    }

    Some(CurvatureReport {
        vertices: welded.vertices,
        curvatures,
    })
}

/// Interior angle at `apex` between edges to `b` and `c`, in radians.
fn corner_angle(apex: Vec3, b: Vec3, c: Vec3) -> f64 {
    let e1 = b - apex;
    let e2 = c - apex;
    let denom = f64::from(e1.length()) * f64::from(e2.length());
    if denom <= 0.0 {
        return 0.0;
    }
    let cos = (f64::from(e1.dot(e2)) / denom).clamp(-1.0, 1.0);
    cos.acos()
}

/// Cotangent of the interior angle at `apex`.
fn cotangent(apex: Vec3, b: Vec3, c: Vec3) -> f64 {
    let e1 = b - apex;
    let e2 = c - apex;
    let cross_len = f64::from(e1.cross(e2).length());
    if cross_len <= f64::from(DEGENERATE_EPSILON) {
        return 0.0;
    }
    f64::from(e1.dot(e2)) / cross_len
}

/// Adds `weight * (x_u - x_v)` to vertex `u` and the opposite to vertex `v`.
fn accumulate_edge(laplacian: &mut [[f64; 3]], verts: &[Vec3], u: usize, v: usize, weight: f64) {
    let d = verts[u] - verts[v];
    let dx = f64::from(d.x) * weight;
    let dy = f64::from(d.y) * weight;
    let dz = f64::from(d.z) * weight;
    laplacian[u][0] += dx;
    laplacian[u][1] += dy;
    laplacian[u][2] += dz;
    laplacian[v][0] -= dx;
    laplacian[v][1] -= dy;
    laplacian[v][2] -= dz;
}

/// The geometric inputs for one triangle corner's mixed-area contribution.
struct Corner {
    /// Interior angle at the corner's vertex.
    apex_angle: f64,
    /// The other two interior angles of the triangle.
    other_angles: (f64, f64),
    /// Lengths of the two edges incident to the vertex.
    edge_lengths: (f32, f32),
    /// Opposite-corner cotangents paired with `edge_lengths`, following Meyer
    /// et al. (2003).
    opposite_cots: (f64, f64),
}

/// Returns the mixed-Voronoi-area contribution of one triangle corner.
fn mixed_area_contribution(corner: &Corner, tri_area: f64) -> f64 {
    let half_pi = PI * 0.5;
    let (b_angle, c_angle) = corner.other_angles;
    let obtuse = corner.apex_angle > half_pi || b_angle > half_pi || c_angle > half_pi;
    let contribution = if obtuse {
        // Barycentric fallback for obtuse triangles.
        if corner.apex_angle > half_pi {
            tri_area * 0.5
        } else {
            tri_area * 0.25
        }
    } else {
        // Voronoi region: (1/8) * (|ab|^2 cot(C) + |ac|^2 cot(B)).
        let (len_ab, len_ac) = corner.edge_lengths;
        let (cot_b, cot_c) = corner.opposite_cots;
        let lab = f64::from(len_ab);
        let lac = f64::from(len_ac);
        0.125 * (lab * lab * cot_c + lac * lac * cot_b)
    };
    contribution.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit cube centred at the origin, outward wound, as a triangle soup.
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

    /// An icosphere approximation via a subdivided octahedron projected onto
    /// the unit sphere.
    fn unit_sphere(subdivisions: u32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let mut verts = vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
        ];
        let mut tris = vec![
            [4u32, 0, 2],
            [4, 2, 1],
            [4, 1, 3],
            [4, 3, 0],
            [5, 2, 0],
            [5, 1, 2],
            [5, 3, 1],
            [5, 0, 3],
        ];
        for _ in 0..subdivisions {
            let mut next = Vec::new();
            for t in &tris {
                let a = verts[t[0] as usize];
                let b = verts[t[1] as usize];
                let c = verts[t[2] as usize];
                let ab = ((a + b) * 0.5).normalize();
                let bc = ((b + c) * 0.5).normalize();
                let ca = ((c + a) * 0.5).normalize();
                let i_ab = verts.len() as u32;
                verts.push(ab);
                let i_bc = verts.len() as u32;
                verts.push(bc);
                let i_ca = verts.len() as u32;
                verts.push(ca);
                next.push([t[0], i_ab, i_ca]);
                next.push([i_ab, t[1], i_bc]);
                next.push([i_ca, i_bc, t[2]]);
                next.push([i_ab, i_bc, i_ca]);
            }
            tris = next;
        }
        (verts, tris)
    }

    #[test]
    fn empty_or_invalid_input_is_rejected() {
        let (v, f) = unit_cube();
        let p = CurvatureParams::default();
        assert!(estimate_curvature(&[], &f, p).is_none());
        assert!(estimate_curvature(&v, &[], p).is_none());
        assert!(estimate_curvature(&v, &f, CurvatureParams { weld_epsilon: 0.0 }).is_none());
        assert!(estimate_curvature(&v, &f, CurvatureParams { weld_epsilon: -1.0 }).is_none());
        assert!(estimate_curvature(
            &v,
            &f,
            CurvatureParams {
                weld_epsilon: f32::NAN
            }
        )
        .is_none());
    }

    #[test]
    fn flat_grid_has_near_zero_curvature() {
        // A 2x2 grid of quads in the z=0 plane (nine vertices, eight triangles).
        let mut verts = Vec::new();
        for y in 0..3 {
            for x in 0..3 {
                verts.push(Vec3::new(x as f32, y as f32, 0.0));
            }
        }
        let idx = |x: u32, y: u32| y * 3 + x;
        let mut tris = Vec::new();
        for y in 0..2u32 {
            for x in 0..2u32 {
                tris.push([idx(x, y), idx(x + 1, y), idx(x + 1, y + 1)]);
                tris.push([idx(x, y), idx(x + 1, y + 1), idx(x, y + 1)]);
            }
        }
        let report = estimate_curvature(&verts, &tris, CurvatureParams::default()).unwrap();
        // The single interior vertex (index 4) should be essentially flat.
        let center = report.curvatures[idx(1, 1) as usize];
        assert!(center.mean.abs() < 1.0e-4, "mean = {}", center.mean);
        assert!(
            center.gaussian.abs() < 1.0e-4,
            "gaussian = {}",
            center.gaussian
        );
    }

    #[test]
    fn sphere_mean_curvature_matches_inverse_radius() {
        // For a unit sphere, H = 1/R = 1 and K = 1/R^2 = 1 everywhere.
        let (verts, tris) = unit_sphere(3);
        let report = estimate_curvature(&verts, &tris, CurvatureParams::default()).unwrap();
        let mut mean_sum = 0.0_f64;
        let mut gauss_sum = 0.0_f64;
        for c in &report.curvatures {
            mean_sum += f64::from(c.mean);
            gauss_sum += f64::from(c.gaussian);
        }
        let count = report.curvatures.len() as f64;
        let mean_avg = mean_sum / count;
        let gauss_avg = gauss_sum / count;
        assert!((mean_avg - 1.0).abs() < 0.1, "mean avg = {mean_avg}");
        assert!((gauss_avg - 1.0).abs() < 0.15, "gauss avg = {gauss_avg}");
    }

    #[test]
    fn sphere_total_gaussian_curvature_obeys_gauss_bonnet() {
        // Gauss-Bonnet: the integral of K over a closed genus-0 surface is 4*PI.
        let (verts, tris) = unit_sphere(3);
        let report = estimate_curvature(&verts, &tris, CurvatureParams::default()).unwrap();
        let integral: f64 = report
            .curvatures
            .iter()
            .map(|c| f64::from(c.gaussian) * f64::from(c.area))
            .sum();
        let expected = 4.0 * PI;
        assert!((integral - expected).abs() < 0.2, "integral = {integral}");
    }

    #[test]
    fn principal_curvatures_bracket_the_mean() {
        let (verts, tris) = unit_sphere(2);
        let report = estimate_curvature(&verts, &tris, CurvatureParams::default()).unwrap();
        for c in &report.curvatures {
            if c.area > 0.0 {
                assert!(c.max_principal >= c.mean - 1.0e-4);
                assert!(c.min_principal <= c.mean + 1.0e-4);
                assert!(c.max_principal >= c.min_principal - 1.0e-4);
            }
        }
    }

    #[test]
    fn report_is_deterministic() {
        let (verts, tris) = unit_sphere(2);
        let p = CurvatureParams::default();
        let a = estimate_curvature(&verts, &tris, p).unwrap();
        let b = estimate_curvature(&verts, &tris, p).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn average_abs_mean_is_positive_for_a_sphere() {
        let (verts, tris) = unit_sphere(2);
        let report = estimate_curvature(&verts, &tris, CurvatureParams::default()).unwrap();
        assert!(report.average_abs_mean() > 0.5);
        assert_eq!(report.len(), report.vertices.len());
        assert!(!report.is_empty());
    }
}
