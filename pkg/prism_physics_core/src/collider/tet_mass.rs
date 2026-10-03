//! Mass, volume and inertia properties of a solid tetrahedral mesh.
//!
//! Given a tetrahedral volume mesh and a uniform density, this module computes
//! the quantities a finite-element or reduced-order soft-body solver needs:
//!
//! * per-vertex **lumped masses** (each tet distributes a quarter of its mass
//!   to each of its four nodes),
//! * the total mass and volume,
//! * the solid **center of mass** (volume-weighted tet centroids), and
//! * the solid **inertia tensor** about the center of mass.
//!
//! The inertia tensor uses the exact per-tet second-moment (covariance)
//! integral, mapped from the canonical reference tetrahedron by the affine
//! Jacobian, so it is exact for the piecewise-linear solid -- not a point-mass
//! approximation. Accumulation is in `f64` for robustness. This is standard
//! rigid-body mathematics; nothing here is derived from Unreal Engine source.

use glam::{Mat3, Vec3};

/// Parameters controlling [`compute_tet_mass_properties`].
#[derive(Clone, Copy, Debug)]
pub struct TetMassParams {
    /// Uniform mass density (mass per unit volume). Must be finite and
    /// positive.
    pub density: f32,
}

impl Default for TetMassParams {
    fn default() -> Self {
        // Density of water in kg/m^3: a sensible soft-body default.
        Self { density: 1000.0 }
    }
}

/// Mass properties of a solid tetrahedral mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct TetMassProperties {
    /// Total mass (`density * total_volume`).
    pub total_mass: f32,
    /// Total enclosed volume (sum of absolute tet volumes).
    pub total_volume: f32,
    /// Solid center of mass in the mesh's coordinate frame.
    pub center_of_mass: Vec3,
    /// Per-input-vertex lumped mass. Length equals `vertices.len()`; vertices
    /// used by no tet receive zero.
    pub nodal_masses: Vec<f32>,
    /// Solid inertia tensor about [`Self::center_of_mass`], axis-aligned to the
    /// mesh frame.
    pub inertia: Mat3,
}

/// Signed `6 x` volume of the tet via the scalar triple product of its edges.
fn tet_six_volume(e1: [f64; 3], e2: [f64; 3], e3: [f64; 3]) -> f64 {
    let cross = [
        e2[1] * e3[2] - e2[2] * e3[1],
        e2[2] * e3[0] - e2[0] * e3[2],
        e2[0] * e3[1] - e2[1] * e3[0],
    ];
    e1[0] * cross[0] + e1[1] * cross[1] + e1[2] * cross[2]
}

/// Upper-triangular entries `(xx, yy, zz, xy, xz, yz)` of the symmetric outer
/// product `u u^T`.
fn self_outer(u: [f64; 3]) -> [f64; 6] {
    [
        u[0] * u[0],
        u[1] * u[1],
        u[2] * u[2],
        u[0] * u[1],
        u[0] * u[2],
        u[1] * u[2],
    ]
}

/// Upper-triangular entries of the symmetric sum `u w^T + w u^T`.
fn cross_outer(u: [f64; 3], w: [f64; 3]) -> [f64; 6] {
    [
        2.0 * u[0] * w[0],
        2.0 * u[1] * w[1],
        2.0 * u[2] * w[2],
        u[0] * w[1] + u[1] * w[0],
        u[0] * w[2] + u[2] * w[0],
        u[1] * w[2] + u[2] * w[1],
    ]
}

/// Computes the mass, volume, center-of-mass and inertia tensor of a solid
/// tetrahedral mesh with uniform density.
///
/// Tet winding is irrelevant: absolute volumes are used, so inverted tets still
/// contribute positive mass.
///
/// Returns `None` when `tets` is empty, `density` is non-finite or
/// non-positive, any tet references a vertex index outside `vertices`, or the
/// total volume is non-positive (fully degenerate input).
#[must_use]
pub fn compute_tet_mass_properties(
    vertices: &[Vec3],
    tets: &[[u32; 4]],
    params: &TetMassParams,
) -> Option<TetMassProperties> {
    if tets.is_empty() || !params.density.is_finite() || params.density <= 0.0 {
        return None;
    }
    let n = vertices.len();
    for t in tets {
        for &id in t {
            if (id as usize) >= n {
                return None;
            }
        }
    }

    let density = f64::from(params.density);
    let p = |id: u32| -> [f64; 3] {
        let v = vertices[id as usize];
        [f64::from(v.x), f64::from(v.y), f64::from(v.z)]
    };

    // Covariance (second-moment) matrix about the origin, as upper-tri entries,
    // accumulated in volume units; scaled by density at the end.
    let mut cov = [0.0f64; 6];
    let mut total_volume = 0.0f64;
    let mut com = [0.0f64; 3];
    let mut nodal_masses = vec![0.0f64; n];

    for t in tets {
        let a = p(t[0]);
        let v1 = p(t[1]);
        let v2 = p(t[2]);
        let v3 = p(t[3]);
        let e1 = [v1[0] - a[0], v1[1] - a[1], v1[2] - a[2]];
        let e2 = [v2[0] - a[0], v2[1] - a[1], v2[2] - a[2]];
        let e3 = [v3[0] - a[0], v3[1] - a[1], v3[2] - a[2]];

        let absdet = tet_six_volume(e1, e2, e3).abs();
        if absdet <= 0.0 {
            continue;
        }
        let vol = absdet / 6.0;
        total_volume += vol;

        let s = [
            e1[0] + e2[0] + e3[0],
            e1[1] + e2[1] + e3[1],
            e1[2] + e2[2] + e3[2],
        ];
        // Centroid = v0 + s/4; accumulate volume-weighted.
        for k in 0..3 {
            com[k] += vol * (a[k] + s[k] * 0.25);
        }

        // Per-tet covariance about origin, canonical reference moments mapped
        // by the affine Jacobian:
        //   M = (1/6) a a^T
        //     + (1/24)(a s^T + s a^T)
        //     + (1/120)(e1 e1^T + e2 e2^T + e3 e3^T + s s^T)
        let aa = self_outer(a);
        let as_sym = cross_outer(a, s);
        let e1e1 = self_outer(e1);
        let e2e2 = self_outer(e2);
        let e3e3 = self_outer(e3);
        let ss = self_outer(s);
        for i in 0..6 {
            let m = aa[i] / 6.0 + as_sym[i] / 24.0 + (e1e1[i] + e2e2[i] + e3e3[i] + ss[i]) / 120.0;
            cov[i] += absdet * m;
        }

        let node_mass = density * vol * 0.25;
        for &id in t {
            nodal_masses[id as usize] += node_mass;
        }
    }

    if total_volume.is_nan() || total_volume <= 0.0 {
        return None;
    }

    let total_mass = density * total_volume;
    let com = [
        com[0] / total_volume,
        com[1] / total_volume,
        com[2] / total_volume,
    ];

    // Mass covariance about origin, then shifted to the center of mass:
    //   C_com = density * C_origin - M (com com^T).
    let com_outer = self_outer(com);
    let mut c_com = [0.0f64; 6];
    for i in 0..6 {
        c_com[i] = density * cov[i] - total_mass * com_outer[i];
    }

    // Inertia = trace(C) * Id - C.
    let trace = c_com[0] + c_com[1] + c_com[2];
    let ixx = (trace - c_com[0]) as f32;
    let iyy = (trace - c_com[1]) as f32;
    let izz = (trace - c_com[2]) as f32;
    let ixy = (-c_com[3]) as f32;
    let ixz = (-c_com[4]) as f32;
    let iyz = (-c_com[5]) as f32;
    let inertia = Mat3::from_cols(
        Vec3::new(ixx, ixy, ixz),
        Vec3::new(ixy, iyy, iyz),
        Vec3::new(ixz, iyz, izz),
    );

    Some(TetMassProperties {
        total_mass: total_mass as f32,
        total_volume: total_volume as f32,
        center_of_mass: Vec3::new(com[0] as f32, com[1] as f32, com[2] as f32),
        nodal_masses: nodal_masses.into_iter().map(|m| m as f32).collect(),
        inertia,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tetrahedralize::{tetrahedralize, TetMeshParams};
    use std::collections::HashMap as StdHashMap;

    fn icosphere(levels: u32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let t = (1.0 + 5.0_f32.sqrt()) * 0.5;
        let mut verts: Vec<Vec3> = vec![
            Vec3::new(-1.0, t, 0.0),
            Vec3::new(1.0, t, 0.0),
            Vec3::new(-1.0, -t, 0.0),
            Vec3::new(1.0, -t, 0.0),
            Vec3::new(0.0, -1.0, t),
            Vec3::new(0.0, 1.0, t),
            Vec3::new(0.0, -1.0, -t),
            Vec3::new(0.0, 1.0, -t),
            Vec3::new(t, 0.0, -1.0),
            Vec3::new(t, 0.0, 1.0),
            Vec3::new(-t, 0.0, -1.0),
            Vec3::new(-t, 0.0, 1.0),
        ];
        for v in &mut verts {
            *v = v.normalize();
        }
        let mut faces: Vec<[u32; 3]> = vec![
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];
        for _ in 0..levels {
            let mut cache: StdHashMap<(u32, u32), u32> = StdHashMap::new();
            let mut next: Vec<[u32; 3]> = Vec::with_capacity(faces.len() * 4);
            let mut midpoint = |a: u32, b: u32, verts: &mut Vec<Vec3>| -> u32 {
                let key = if a < b { (a, b) } else { (b, a) };
                if let Some(&m) = cache.get(&key) {
                    return m;
                }
                let m = verts.len() as u32;
                let pp = ((verts[a as usize] + verts[b as usize]) * 0.5).normalize();
                verts.push(pp);
                cache.insert(key, m);
                m
            };
            for f in &faces {
                let a = midpoint(f[0], f[1], &mut verts);
                let b = midpoint(f[1], f[2], &mut verts);
                let c = midpoint(f[2], f[0], &mut verts);
                next.push([f[0], a, c]);
                next.push([f[1], b, a]);
                next.push([f[2], c, b]);
                next.push([a, b, c]);
            }
            faces = next;
        }
        (verts, faces)
    }

    /// Independent brute-force covariance of a single tet about the origin via
    /// dense grid quadrature over its bounding box (density 1).
    fn numeric_cov(v: [[f64; 3]; 4], grid: usize) -> [f64; 6] {
        // Barycentric inside-test via the inverse Jacobian.
        let a = v[0];
        let e1 = [v[1][0] - a[0], v[1][1] - a[1], v[1][2] - a[2]];
        let e2 = [v[2][0] - a[0], v[2][1] - a[1], v[2][2] - a[2]];
        let e3 = [v[3][0] - a[0], v[3][1] - a[1], v[3][2] - a[2]];
        // Jacobian columns e1,e2,e3; invert the 3x3.
        let det = e1[0] * (e2[1] * e3[2] - e2[2] * e3[1]) - e2[0] * (e1[1] * e3[2] - e1[2] * e3[1])
            + e3[0] * (e1[1] * e2[2] - e1[2] * e2[1]);
        let inv = {
            let m = [
                [e1[0], e2[0], e3[0]],
                [e1[1], e2[1], e3[1]],
                [e1[2], e2[2], e3[2]],
            ];
            let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
            let c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
            let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
            let c10 = m[0][2] * m[2][1] - m[0][1] * m[2][2];
            let c11 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
            let c12 = m[0][1] * m[2][0] - m[0][0] * m[2][1];
            let c20 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
            let c21 = m[0][2] * m[1][0] - m[0][0] * m[1][2];
            let c22 = m[0][0] * m[1][1] - m[0][1] * m[1][0];
            [
                [c00 / det, c10 / det, c20 / det],
                [c01 / det, c11 / det, c21 / det],
                [c02 / det, c12 / det, c22 / det],
            ]
        };
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in &v {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let cell = [
            (hi[0] - lo[0]) / grid as f64,
            (hi[1] - lo[1]) / grid as f64,
            (hi[2] - lo[2]) / grid as f64,
        ];
        let cell_vol = cell[0] * cell[1] * cell[2];
        let mut cov = [0.0f64; 6];
        for i in 0..grid {
            for j in 0..grid {
                for k in 0..grid {
                    let r = [
                        lo[0] + (i as f64 + 0.5) * cell[0],
                        lo[1] + (j as f64 + 0.5) * cell[1],
                        lo[2] + (k as f64 + 0.5) * cell[2],
                    ];
                    let d = [r[0] - a[0], r[1] - a[1], r[2] - a[2]];
                    let b0 = inv[0][0] * d[0] + inv[0][1] * d[1] + inv[0][2] * d[2];
                    let b1 = inv[1][0] * d[0] + inv[1][1] * d[1] + inv[1][2] * d[2];
                    let b2 = inv[2][0] * d[0] + inv[2][1] * d[1] + inv[2][2] * d[2];
                    if b0 >= 0.0 && b1 >= 0.0 && b2 >= 0.0 && (b0 + b1 + b2) <= 1.0 {
                        cov[0] += r[0] * r[0] * cell_vol;
                        cov[1] += r[1] * r[1] * cell_vol;
                        cov[2] += r[2] * r[2] * cell_vol;
                        cov[3] += r[0] * r[1] * cell_vol;
                        cov[4] += r[0] * r[2] * cell_vol;
                        cov[5] += r[1] * r[2] * cell_vol;
                    }
                }
            }
        }
        cov
    }

    #[test]
    fn rejects_bad_input() {
        let v = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
        let tets = vec![[0u32, 1, 2, 3]];
        assert!(compute_tet_mass_properties(&[], &[], &TetMassParams::default()).is_none());
        assert!(
            compute_tet_mass_properties(&v, &[[0u32, 1, 2, 9]], &TetMassParams::default())
                .is_none()
        );
        assert!(compute_tet_mass_properties(&v, &tets, &TetMassParams { density: 0.0 }).is_none());
        assert!(compute_tet_mass_properties(&v, &tets, &TetMassParams { density: -1.0 }).is_none());
        assert!(
            compute_tet_mass_properties(&v, &tets, &TetMassParams { density: f32::NAN }).is_none()
        );
    }

    #[test]
    fn single_tet_covariance_matches_numeric_quadrature() {
        // Asymmetric, off-origin tet to exercise every covariance term.
        let v = [
            [0.3f64, -0.2, 0.1],
            [1.4, 0.1, -0.3],
            [0.2, 1.1, 0.4],
            [-0.1, 0.3, 1.2],
        ];
        let verts = vec![
            Vec3::new(0.3, -0.2, 0.1),
            Vec3::new(1.4, 0.1, -0.3),
            Vec3::new(0.2, 1.1, 0.4),
            Vec3::new(-0.1, 0.3, 1.2),
        ];
        let tets = vec![[0u32, 1, 2, 3]];
        let props =
            compute_tet_mass_properties(&verts, &tets, &TetMassParams { density: 1.0 }).unwrap();

        // Reconstruct the analytic covariance-about-origin from the returned
        // inertia + com + mass: C_com = tr/2 Id - I; then C_origin add back.
        // Simpler: recompute C_origin directly from the closed form here would
        // duplicate code, so instead cross-check the *inertia* via numeric cov.
        let numeric = numeric_cov(v, 90);
        // numeric is C about origin (density 1). Convert to inertia about COM.
        let total_v = {
            let e1 = [v[1][0] - v[0][0], v[1][1] - v[0][1], v[1][2] - v[0][2]];
            let e2 = [v[2][0] - v[0][0], v[2][1] - v[0][1], v[2][2] - v[0][2]];
            let e3 = [v[3][0] - v[0][0], v[3][1] - v[0][1], v[3][2] - v[0][2]];
            tet_six_volume(e1, e2, e3).abs() / 6.0
        };
        let com = [
            (v[0][0] + v[1][0] + v[2][0] + v[3][0]) / 4.0,
            (v[0][1] + v[1][1] + v[2][1] + v[3][1]) / 4.0,
            (v[0][2] + v[1][2] + v[2][2] + v[3][2]) / 4.0,
        ];
        let co = self_outer(com);
        let c_com = [
            numeric[0] - total_v * co[0],
            numeric[1] - total_v * co[1],
            numeric[2] - total_v * co[2],
            numeric[3] - total_v * co[3],
            numeric[4] - total_v * co[4],
            numeric[5] - total_v * co[5],
        ];
        let trace = c_com[0] + c_com[1] + c_com[2];
        let ixx = trace - c_com[0];
        let iyy = trace - c_com[1];
        let izz = trace - c_com[2];
        let ixy = -c_com[3];

        let got = props.inertia;
        let rel = |a: f64, b: f32| (a - f64::from(b)).abs() / (a.abs().max(1e-6));
        assert!(
            rel(ixx, got.col(0).x) < 0.02,
            "Ixx {ixx} vs {}",
            got.col(0).x
        );
        assert!(
            rel(iyy, got.col(1).y) < 0.02,
            "Iyy {iyy} vs {}",
            got.col(1).y
        );
        assert!(
            rel(izz, got.col(2).z) < 0.02,
            "Izz {izz} vs {}",
            got.col(2).z
        );
        assert!(
            rel(ixy, got.col(0).y) < 0.03,
            "Ixy {ixy} vs {}",
            got.col(0).y
        );
    }

    #[test]
    fn nodal_masses_sum_to_total_and_volume_scales() {
        let (v, i) = icosphere(1);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(12)).unwrap();
        let props = compute_tet_mass_properties(
            &mesh.vertices,
            &mesh.tets,
            &TetMassParams { density: 2.0 },
        )
        .unwrap();
        assert_eq!(props.nodal_masses.len(), mesh.vertices.len());
        let sum: f32 = props.nodal_masses.iter().sum();
        assert!((sum - props.total_mass).abs() < 1e-2 * props.total_mass.max(1.0));
        assert!((props.total_mass - 2.0 * props.total_volume).abs() < 1e-3 * props.total_mass);
    }

    #[test]
    fn density_scales_mass_and_inertia_not_geometry() {
        let (v, i) = icosphere(1);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(12)).unwrap();
        let a = compute_tet_mass_properties(
            &mesh.vertices,
            &mesh.tets,
            &TetMassParams { density: 1.0 },
        )
        .unwrap();
        let b = compute_tet_mass_properties(
            &mesh.vertices,
            &mesh.tets,
            &TetMassParams { density: 3.0 },
        )
        .unwrap();
        assert!((b.total_mass - 3.0 * a.total_mass).abs() < 1e-3 * b.total_mass);
        assert!((b.total_volume - a.total_volume).abs() < 1e-4 * a.total_volume.max(1.0));
        assert!((b.center_of_mass - a.center_of_mass).length() < 1e-5);
        assert!(
            (b.inertia.col(0).x - 3.0 * a.inertia.col(0).x).abs()
                < 1e-3 * b.inertia.col(0).x.abs().max(1.0)
        );
    }

    #[test]
    fn sphere_is_centered_and_nearly_isotropic() {
        let (v, i) = icosphere(2);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(20)).unwrap();
        let props =
            compute_tet_mass_properties(&mesh.vertices, &mesh.tets, &TetMassParams::default())
                .unwrap();
        // A sphere centered at the origin: COM near origin.
        assert!(
            props.center_of_mass.length() < 0.03,
            "COM off-center: {:?}",
            props.center_of_mass
        );
        // Near-isotropic: diagonal entries close, off-diagonals small.
        let ixx = props.inertia.col(0).x;
        let iyy = props.inertia.col(1).y;
        let izz = props.inertia.col(2).z;
        let mean = (ixx + iyy + izz) / 3.0;
        for d in [ixx, iyy, izz] {
            assert!(
                (d - mean).abs() < 0.08 * mean,
                "anisotropic diagonal: {d} vs {mean}"
            );
        }
        let ixy = props.inertia.col(0).y.abs();
        let ixz = props.inertia.col(0).z.abs();
        let iyz = props.inertia.col(1).z.abs();
        for o in [ixy, ixz, iyz] {
            assert!(o < 0.06 * mean, "off-diagonal too large: {o} vs {mean}");
        }
        // A solid sphere of the *measured* volume has gyration radius-squared
        // 0.4 R_eff^2 with R_eff = (3V / 4pi)^(1/3). Comparing against the
        // measured volume (not the ideal unit radius) removes the voxelisation
        // volume-shrink artefact, leaving only shape roughness.
        let r_eff2 =
            (3.0 * props.total_volume / (4.0 * core::f32::consts::PI)).powf(2.0 / 3.0);
        let analytic = 0.4 * props.total_mass * r_eff2;
        assert!(
            (mean - analytic).abs() < 0.05 * analytic,
            "mean {mean} vs {analytic}"
        );
    }

    #[test]
    fn is_deterministic() {
        let (v, i) = icosphere(1);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(12)).unwrap();
        let a = compute_tet_mass_properties(&mesh.vertices, &mesh.tets, &TetMassParams::default())
            .unwrap();
        let b = compute_tet_mass_properties(&mesh.vertices, &mesh.tets, &TetMassParams::default())
            .unwrap();
        assert_eq!(a, b);
    }
}
