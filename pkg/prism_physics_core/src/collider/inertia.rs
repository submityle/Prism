//! Full rigid-body inertia for closed triangle meshes.
//!
//! [`ConvexMeshData::unit_density_inertia`](crate::collider::ConvexMeshData::unit_density_inertia)
//! reports only the *diagonal* second moments about the body origin and
//! deliberately discards the off-diagonal product-of-inertia terms (matching
//! the documented capsule approximation). That is enough for symmetric
//! primitives whose principal axes already align with the local frame, but a
//! general convex hull - or any closed watertight mesh - has a genuinely
//! non-diagonal inertia tensor whose principal axes are rotated relative to the
//! mesh frame. AAA solvers need the full tensor to seed the body-space inertia
//! and the principal-axis frame.
//!
//! This module computes, for a closed oriented triangle mesh of uniform
//! density:
//!
//! - the enclosed mass (`density * volume`),
//! - the center of mass, and
//! - the full symmetric `3x3` inertia tensor **about the center of mass**,
//!
//! together with a [`principal`](MeshInertia::principal) decomposition that
//! diagonalises the tensor into principal moments plus a right-handed
//! principal-axis rotation (via a cyclic Jacobi eigensolver).
//!
//! The integration uses the signed-tetrahedron fan `(origin, a, b, c)` over the
//! surface triangles (the same decomposition used by
//! [`ConvexMeshData`](crate::collider::ConvexMeshData) for its diagonal
//! moments), extended to accumulate the full covariance (including the
//! off-diagonal integrals) and the first moments needed for the center of mass.
//! The mesh need not be convex; it only needs to be closed and consistently
//! wound (either outward or inward, since the sign is normalised internally).
//!
//! These are standard closed-form integrals and a standard symmetric
//! eigensolver; nothing here is derived from Unreal Engine source.

use glam::{Mat3, Vec3};

/// Full rigid-body mass properties of a closed triangle mesh at a given
/// uniform density.
///
/// Unlike the diagonal-only
/// [`MassProperties`](crate::state::body::MassProperties) produced by the
/// analytic shapes, [`inertia_com`](MeshInertia::inertia_com) is the complete
/// symmetric tensor and may carry non-zero off-diagonal (product-of-inertia)
/// terms when the mesh frame is not a principal frame.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MeshInertia {
    /// Total mass (`density * enclosed_volume`), in kilograms for metres and
    /// `kg/m^3`.
    pub mass: f32,
    /// Enclosed volume of the solid, in cubic metres (always non-negative).
    pub volume: f32,
    /// Center of mass in the mesh's local frame.
    pub center_of_mass: Vec3,
    /// Full symmetric `3x3` inertia tensor about the center of mass, at the
    /// requested density. Symmetric by construction: `inertia_com == inertia_com.transpose()`
    /// up to floating-point round-off.
    pub inertia_com: Mat3,
}

/// A principal-axis decomposition of a symmetric inertia tensor.
///
/// The three columns of [`axes`](PrincipalInertia::axes) are the orthonormal
/// principal directions (a proper, right-handed rotation with
/// `determinant == +1`), and the matching entries of
/// [`moments`](PrincipalInertia::moments) are the principal moments of inertia,
/// sorted ascending (`moments.x <= moments.y <= moments.z`). The original
/// tensor `I` satisfies `I == axes * diag(moments) * axes.transpose()` up to
/// solver tolerance.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PrincipalInertia {
    /// Principal moments of inertia, sorted ascending.
    pub moments: Vec3,
    /// Right-handed rotation whose columns are the principal axes, ordered to
    /// match [`moments`](PrincipalInertia::moments).
    pub axes: Mat3,
}

/// Volume below which a mesh is treated as degenerate (no inertia).
const MIN_VOLUME: f32 = 1e-9;

/// Computes the full mass properties of a closed, consistently wound triangle
/// mesh at uniform `density`.
///
/// `vertices` are the mesh positions in local space and `triangles` index into
/// them (three vertex indices per face). The winding may be either outward or
/// inward; the enclosed-volume sign is normalised internally so the returned
/// mass and tensor are always physical.
///
/// Returns [`None`] when `density <= 0`, when any triangle index is out of
/// range, or when the enclosed volume is below [`MIN_VOLUME`] (an open, flat,
/// or otherwise degenerate solid), matching the immovable-shape precedent used
/// elsewhere in the collider module.
#[must_use]
pub fn full_inertia_tensor(
    vertices: &[Vec3],
    triangles: &[[u32; 3]],
    density: f32,
) -> Option<MeshInertia> {
    if density <= 0.0 || vertices.is_empty() || triangles.is_empty() {
        return None;
    }

    // Signed accumulators over the tetrahedron fan `(origin, a, b, c)`.
    let mut signed_volume = 0.0_f32;
    // First moments `integral(x_i) dV`, used for the center of mass.
    let mut first = Vec3::ZERO;
    // Covariance `integral(x_i x_j) dV` about the local origin, upper triangle.
    let mut c_xx = 0.0_f32;
    let mut c_yy = 0.0_f32;
    let mut c_zz = 0.0_f32;
    let mut c_xy = 0.0_f32;
    let mut c_xz = 0.0_f32;
    let mut c_yz = 0.0_f32;

    for tri in triangles {
        let a = *vertices.get(tri[0] as usize)?;
        let b = *vertices.get(tri[1] as usize)?;
        let c = *vertices.get(tri[2] as usize)?;

        // Signed volume of the tetrahedron `(origin, a, b, c)`.
        let vol = a.dot(b.cross(c)) / 6.0;
        signed_volume += vol;

        // First moment: the tetrahedron centroid is `(0 + a + b + c) / 4`.
        first += (a + b + c) * (vol * 0.25);

        // Second moments over the tetrahedron (one vertex at the origin):
        //   integral(x_i x_j) = (vol / 20) *
        //       (2 a_i a_j + 2 b_i b_j + 2 c_i c_j
        //        + a_i b_j + a_j b_i + b_i c_j + b_j c_i + c_i a_j + c_j a_i).
        // Setting i == j recovers the diagonal `(vol / 10)` form used by
        // `ConvexMeshData`.
        let w = vol / 20.0;
        c_xx += w * cov_term(a.x, b.x, c.x, a.x, b.x, c.x);
        c_yy += w * cov_term(a.y, b.y, c.y, a.y, b.y, c.y);
        c_zz += w * cov_term(a.z, b.z, c.z, a.z, b.z, c.z);
        c_xy += w * cov_term(a.x, b.x, c.x, a.y, b.y, c.y);
        c_xz += w * cov_term(a.x, b.x, c.x, a.z, b.z, c.z);
        c_yz += w * cov_term(a.y, b.y, c.y, a.z, b.z, c.z);
    }

    if signed_volume.abs() <= MIN_VOLUME {
        return None;
    }

    // Normalise an inward-wound mesh (negative signed volume) so all moments
    // describe the positively oriented solid.
    let sign = if signed_volume < 0.0 { -1.0 } else { 1.0 };
    let volume = signed_volume * sign;
    let first = first * sign;
    let c_xx = c_xx * sign;
    let c_yy = c_yy * sign;
    let c_zz = c_zz * sign;
    let c_xy = c_xy * sign;
    let c_xz = c_xz * sign;
    let c_yz = c_yz * sign;

    let center_of_mass = first / volume;
    let mass = density * volume;

    // Inertia tensor about the local origin at the requested density. For a
    // covariance `C`, `I = density * (trace(C) * E - C)`:
    //   I_xx = density * (C_yy + C_zz),  I_xy = -density * C_xy, ...
    let origin = Mat3::from_cols(
        Vec3::new(density * (c_yy + c_zz), -density * c_xy, -density * c_xz),
        Vec3::new(-density * c_xy, density * (c_xx + c_zz), -density * c_yz),
        Vec3::new(-density * c_xz, -density * c_yz, density * (c_xx + c_yy)),
    );

    // Shift to the center of mass via the parallel-axis theorem:
    //   I_origin = I_com + mass * ((d . d) E - d (x) d),
    // so I_com = I_origin - mass * ((d . d) E - d (x) d).
    let d = center_of_mass;
    let shift = Mat3::IDENTITY * d.length_squared() - outer(d, d);
    let inertia_com = origin - shift * mass;

    Some(MeshInertia {
        mass,
        volume,
        center_of_mass,
        inertia_com: symmetrise(inertia_com),
    })
}

impl MeshInertia {
    /// Diagonalises [`inertia_com`](MeshInertia::inertia_com) into principal
    /// moments and a right-handed principal-axis rotation.
    ///
    /// See [`principal_axes`] for the solver details.
    #[must_use]
    pub fn principal(&self) -> PrincipalInertia {
        principal_axes(self.inertia_com)
    }
}

/// Diagonalises a symmetric `3x3` matrix into eigenvalues and an orthonormal,
/// right-handed eigenvector basis using a cyclic Jacobi sweep.
///
/// The returned [`PrincipalInertia::moments`] are sorted ascending and
/// [`PrincipalInertia::axes`] holds the matching eigenvectors as columns,
/// adjusted (by flipping one axis if needed) to form a proper rotation with
/// `determinant == +1`. Only the symmetric part of `m` is used.
#[must_use]
pub fn principal_axes(m: Mat3) -> PrincipalInertia {
    // Work on a symmetric 3x3 in row-major arrays. `a` is the matrix being
    // diagonalised; `v` accumulates the applied rotations (eigenvectors).
    let s = symmetrise(m);
    let mut a = [
        [s.x_axis.x, s.y_axis.x, s.z_axis.x],
        [s.x_axis.y, s.y_axis.y, s.z_axis.y],
        [s.x_axis.z, s.y_axis.z, s.z_axis.z],
    ];
    let mut v = [[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    // 32 sweeps far exceed the quadratic convergence needed for a 3x3.
    for _ in 0..32 {
        let off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
        if off <= 1e-18 {
            break;
        }
        for (p, q) in [(0_usize, 1_usize), (0, 2), (1, 2)] {
            let apq = a[p][q];
            if apq.abs() <= 1e-20 {
                continue;
            }
            // Jacobi rotation angle, computed via tan to avoid trigonometry.
            let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
            let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
            let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
            let cos = 1.0 / (t * t + 1.0).sqrt();
            let sin = t * cos;

            // Two-sided rotation `a <- R^T a R` and eigenvector update `v <- v R`.
            // Column update: rotate columns `p` and `q` within every row.
            for row in a.iter_mut() {
                let akp = row[p];
                let akq = row[q];
                row[p] = cos * akp - sin * akq;
                row[q] = sin * akp + cos * akq;
            }
            // Row update: rotate rows `p` and `q` across every column. `p < q`
            // always holds, so `split_at_mut(q)` yields disjoint borrows.
            {
                let (lo, hi) = a.split_at_mut(q);
                let rp = &mut lo[p];
                let rq = &mut hi[0];
                for (apk, aqk) in rp.iter_mut().zip(rq.iter_mut()) {
                    let a0 = *apk;
                    let a1 = *aqk;
                    *apk = cos * a0 - sin * a1;
                    *aqk = sin * a0 + cos * a1;
                }
            }
            // Accumulate the rotation into the eigenvector basis.
            for row in v.iter_mut() {
                let vkp = row[p];
                let vkq = row[q];
                row[p] = cos * vkp - sin * vkq;
                row[q] = sin * vkp + cos * vkq;
            }
        }
    }

    // Eigenvalues on the diagonal; eigenvectors are the columns of `v`.
    let mut eig = [
        (a[0][0], Vec3::new(v[0][0], v[1][0], v[2][0])),
        (a[1][1], Vec3::new(v[0][1], v[1][1], v[2][1])),
        (a[2][2], Vec3::new(v[0][2], v[1][2], v[2][2])),
    ];
    eig.sort_by(|l, r| l.0.total_cmp(&r.0));

    let mut axes = Mat3::from_cols(
        eig[0].1.normalize_or_zero(),
        eig[1].1.normalize_or_zero(),
        eig[2].1.normalize_or_zero(),
    );
    // Force a right-handed basis so the result is a proper rotation.
    if axes.determinant() < 0.0 {
        axes.z_axis = -axes.z_axis;
    }

    PrincipalInertia {
        moments: Vec3::new(eig[0].0, eig[1].0, eig[2].0),
        axes,
    }
}

/// Covariance polynomial `2 x1 y1 + 2 x2 y2 + 2 x3 y3 + x1 y2 + x2 y1 + x2 y3 + x3 y2 + x3 y1 + x1 y3`.
///
/// This is the per-tetrahedron second-moment integrand (without the `vol / 20`
/// weight) for the component pair whose first-axis samples are `(x1, x2, x3)`
/// at the triangle corners and whose second-axis samples are `(y1, y2, y3)`.
#[inline]
fn cov_term(x1: f32, x2: f32, x3: f32, y1: f32, y2: f32, y3: f32) -> f32 {
    2.0 * x1 * y1
        + 2.0 * x2 * y2
        + 2.0 * x3 * y3
        + x1 * y2
        + x2 * y1
        + x2 * y3
        + x3 * y2
        + x3 * y1
        + x1 * y3
}

/// Outer product `u (x) w`, the matrix with entry `(i, j) == u_i w_j`.
#[inline]
fn outer(u: Vec3, w: Vec3) -> Mat3 {
    Mat3::from_cols(u * w.x, u * w.y, u * w.z)
}

/// Returns the symmetric part `(m + m^T) / 2`, cancelling accumulated
/// round-off asymmetry.
#[inline]
fn symmetrise(m: Mat3) -> Mat3 {
    (m + m.transpose()) * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A closed, outward-wound axis-aligned box centred at `center` with the
    /// given full `size` along each axis.
    fn box_mesh(center: Vec3, size: Vec3) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let h = size * 0.5;
        let mut verts = Vec::with_capacity(8);
        for sx in [-1.0_f32, 1.0] {
            for sy in [-1.0_f32, 1.0] {
                for sz in [-1.0_f32, 1.0] {
                    verts.push(center + Vec3::new(sx * h.x, sy * h.y, sz * h.z));
                }
            }
        }
        // Vertex index layout: bit0=z, bit1=y, bit2=x (from the loop order).
        // Outward-wound triangles for each of the six faces.
        let tris = vec![
            // -x face (x = -h): verts 0,1,2,3
            [0, 2, 3],
            [0, 3, 1],
            // +x face (x = +h): verts 4,5,6,7
            [4, 5, 7],
            [4, 7, 6],
            // -y face (y = -h): verts 0,1,4,5
            [0, 1, 5],
            [0, 5, 4],
            // +y face (y = +h): verts 2,3,6,7
            [2, 6, 7],
            [2, 7, 3],
            // -z face (z = -h): verts 0,2,4,6
            [0, 4, 6],
            [0, 6, 2],
            // +z face (z = +h): verts 1,3,5,7
            [1, 3, 7],
            [1, 7, 5],
        ];
        (verts, tris)
    }

    #[test]
    fn unit_cube_matches_closed_form() {
        let (v, t) = box_mesh(Vec3::ZERO, Vec3::splat(2.0));
        let mi = full_inertia_tensor(&v, &t, 1.0).expect("non-degenerate");
        // Side 2 => volume 8, mass 8 at unit density.
        assert!((mi.volume - 8.0).abs() < 1e-4);
        assert!((mi.mass - 8.0).abs() < 1e-4);
        assert!(mi.center_of_mass.length() < 1e-5);
        // Box inertia: I = m/12 * (b^2 + c^2); for a cube side s, = m * s^2 / 6.
        let expect = 8.0 * 4.0 / 6.0;
        assert!((mi.inertia_com.x_axis.x - expect).abs() < 1e-3);
        assert!((mi.inertia_com.y_axis.y - expect).abs() < 1e-3);
        assert!((mi.inertia_com.z_axis.z - expect).abs() < 1e-3);
        // Off-diagonals vanish for an axis-aligned box.
        assert!(mi.inertia_com.x_axis.y.abs() < 1e-4);
        assert!(mi.inertia_com.x_axis.z.abs() < 1e-4);
        assert!(mi.inertia_com.y_axis.z.abs() < 1e-4);
    }

    #[test]
    fn density_scales_mass_and_inertia_linearly() {
        let (v, t) = box_mesh(Vec3::ZERO, Vec3::new(1.0, 2.0, 3.0));
        let a = full_inertia_tensor(&v, &t, 1.0).expect("ok");
        let b = full_inertia_tensor(&v, &t, 2.5).expect("ok");
        assert!((b.mass - a.mass * 2.5).abs() < 1e-3);
        for i in 0..3 {
            let ca = a.inertia_com.col(i);
            let cb = b.inertia_com.col(i);
            assert!((cb - ca * 2.5).length() < 1e-3);
        }
    }

    #[test]
    fn offset_box_reports_center_of_mass() {
        let center = Vec3::new(3.0, -2.0, 5.0);
        let (v, t) = box_mesh(center, Vec3::splat(2.0));
        let mi = full_inertia_tensor(&v, &t, 1.0).expect("ok");
        assert!((mi.center_of_mass - center).length() < 1e-4);
        // Inertia about the COM is frame-invariant: same as the centred cube.
        let expect = 8.0 * 4.0 / 6.0;
        assert!((mi.inertia_com.x_axis.x - expect).abs() < 1e-3);
        assert!(mi.inertia_com.x_axis.y.abs() < 1e-4);
    }

    #[test]
    fn inward_winding_is_normalised() {
        let (v, mut t) = box_mesh(Vec3::ZERO, Vec3::splat(2.0));
        for tri in &mut t {
            tri.swap(1, 2);
        }
        let mi = full_inertia_tensor(&v, &t, 1.0).expect("ok");
        assert!(mi.volume > 0.0);
        assert!((mi.mass - 8.0).abs() < 1e-4);
        let expect = 8.0 * 4.0 / 6.0;
        assert!((mi.inertia_com.z_axis.z - expect).abs() < 1e-3);
    }

    #[test]
    fn rotated_box_has_cross_terms_but_matching_principal_moments() {
        // A thin box: distinct principal moments so axes are well-defined.
        let (v0, t) = box_mesh(Vec3::ZERO, Vec3::new(1.0, 2.0, 4.0));
        let axis_aligned = full_inertia_tensor(&v0, &t, 1.0).expect("ok");

        // Rotate every vertex by 30 deg about Z (precomputed cos/sin constants
        // to avoid trigonometry in the collider crate).
        let (c, s) = (0.866_025_4_f32, 0.5_f32);
        let v: Vec<Vec3> = v0
            .iter()
            .map(|p| Vec3::new(c * p.x - s * p.y, s * p.x + c * p.y, p.z))
            .collect();
        let rotated = full_inertia_tensor(&v, &t, 1.0).expect("ok");

        // The rotated tensor is no longer diagonal.
        assert!(rotated.inertia_com.x_axis.y.abs() > 1e-2);

        // But its principal moments equal the axis-aligned diagonal entries.
        let pa = axis_aligned.principal();
        let pr = rotated.principal();
        assert!((pa.moments - pr.moments).length() < 1e-2);

        // The principal axes form a proper rotation.
        assert!((pr.axes.determinant() - 1.0).abs() < 1e-3);

        // Reconstruction: axes * diag(moments) * axes^T recovers the tensor.
        let diag = Mat3::from_cols(
            Vec3::new(pr.moments.x, 0.0, 0.0),
            Vec3::new(0.0, pr.moments.y, 0.0),
            Vec3::new(0.0, 0.0, pr.moments.z),
        );
        let recon = pr.axes * diag * pr.axes.transpose();
        for i in 0..3 {
            assert!((recon.col(i) - rotated.inertia_com.col(i)).length() < 1e-2);
        }
    }

    #[test]
    fn degenerate_inputs_return_none() {
        let (v, t) = box_mesh(Vec3::ZERO, Vec3::splat(2.0));
        assert!(full_inertia_tensor(&v, &t, 0.0).is_none());
        assert!(full_inertia_tensor(&[], &t, 1.0).is_none());
        assert!(full_inertia_tensor(&v, &[], 1.0).is_none());
        // A single flat quad encloses no volume.
        let flat = vec![Vec3::ZERO, Vec3::X, Vec3::new(1.0, 1.0, 0.0), Vec3::Y];
        let flat_t = vec![[0u32, 1, 2], [0, 2, 3]];
        assert!(full_inertia_tensor(&flat, &flat_t, 1.0).is_none());
        // Out-of-range index is rejected.
        let bad = vec![[0u32, 1, 99]];
        assert!(full_inertia_tensor(&v, &bad, 1.0).is_none());
    }

    #[test]
    fn principal_axes_of_diagonal_tensor_are_identity_like() {
        let m = Mat3::from_cols(
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(0.0, 0.0, 9.0),
        );
        let p = principal_axes(m);
        assert!((p.moments - Vec3::new(2.0, 5.0, 9.0)).length() < 1e-5);
        assert!((p.axes.determinant() - 1.0).abs() < 1e-4);
    }
}
