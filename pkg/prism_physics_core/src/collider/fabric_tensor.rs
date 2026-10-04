//! Contact-fabric (texture) tensor and anisotropy diagnostics.
//!
//! The micromechanical state of a granular packing is captured by the
//! distribution of contact-normal orientations. The second-order *fabric
//! tensor* is the orientational average of the dyadic product of unit contact
//! normals,
//!
//! ```text
//! Φ_ij = (1 / N_c) · Σ_c  n_i^c n_j^c
//! ```
//!
//! where `n^c` is the unit branch vector of contact `c` and `N_c` the number
//! of contacts. By construction `Φ` is symmetric positive semi-definite with
//! unit trace. An isotropic packing yields `Φ = I/3`; departures from that
//! baseline quantify directional structure (force chains, bedding planes).
//!
//! The deviatoric part `Φ' = Φ − I/3` measures anisotropy. We report its
//! Frobenius norm and the von Mises scalar `a = √(3/2 · Φ':Φ')`, together with
//! the three principal fabric intensities (eigenvalues of `Φ`, descending)
//! obtained from the closed-form symmetric-3×3 solution.
//!
//! This module performs pure geometric analysis of a sphere packing; it does
//! not couple to the simulation step.

use glam::Vec3;

/// Fabric-tensor diagnostics for a sphere packing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FabricTensor {
    contact_count: usize,
    tensor: [[f32; 3]; 3],
    eigenvalues: [f32; 3],
    deviatoric_norm: f32,
    anisotropy: f32,
}

impl FabricTensor {
    /// Builds the fabric tensor from sphere centres and radii.
    ///
    /// Two spheres are in contact when the centre distance does not exceed the
    /// sum of radii plus `contact_tolerance` (`≥ 0`, an absolute gap margin).
    /// Each contact contributes a unit branch vector `n = (p_j − p_i)/|…|`.
    ///
    /// Returns `None` when `positions`/`radii` lengths disagree, inputs are
    /// empty, any value is non-finite, `contact_tolerance < 0`, or no contacts
    /// are found (the fabric tensor is undefined without contacts).
    pub fn from_contacts(
        positions: &[Vec3],
        radii: &[f32],
        contact_tolerance: f32,
    ) -> Option<Self> {
        if positions.is_empty() || positions.len() != radii.len() {
            return None;
        }
        if !contact_tolerance.is_finite() || contact_tolerance < 0.0 {
            return None;
        }
        let mut max_radius = 0.0_f32;
        for (p, &r) in positions.iter().zip(radii.iter()) {
            if !p.is_finite() || !r.is_finite() || r <= 0.0 {
                return None;
            }
            max_radius = max_radius.max(r);
        }

        // Spatial hash with a cell equal to the largest possible contact
        // distance so that a 3×3×3 neighbourhood captures every contact.
        let cutoff = 2.0 * max_radius + contact_tolerance;
        if !cutoff.is_finite() || cutoff <= 0.0 {
            return None;
        }
        let mut min = positions[0];
        for p in positions.iter() {
            min = min.min(*p);
        }
        let inv_cell = 1.0 / cutoff;
        let cell_of = |p: Vec3| -> (i64, i64, i64) {
            let d = p - min;
            (
                (d.x * inv_cell).floor() as i64,
                (d.y * inv_cell).floor() as i64,
                (d.z * inv_cell).floor() as i64,
            )
        };

        use std::collections::HashMap;
        let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
        for (i, p) in positions.iter().enumerate() {
            grid.entry(cell_of(*p)).or_default().push(i);
        }

        // Accumulate the symmetric fabric tensor in f64 for numerical headroom.
        let mut acc = [[0.0_f64; 3]; 3];
        let mut contact_count: usize = 0;
        for (i, &pi) in positions.iter().enumerate() {
            let (cx, cy, cz) = cell_of(pi);
            let ri = radii[i];
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let Some(bucket) = grid.get(&(cx + dx, cy + dy, cz + dz)) else {
                            continue;
                        };
                        for &j in bucket.iter() {
                            if j <= i {
                                continue; // unordered pairs, once each
                            }
                            let pj = positions[j];
                            let delta = pj - pi;
                            let dist = delta.length();
                            let touch = ri + radii[j] + contact_tolerance;
                            if dist > touch || dist <= 0.0 {
                                continue;
                            }
                            let n = delta / dist;
                            let (nx, ny, nz) = (n.x as f64, n.y as f64, n.z as f64);
                            acc[0][0] += nx * nx;
                            acc[0][1] += nx * ny;
                            acc[0][2] += nx * nz;
                            acc[1][1] += ny * ny;
                            acc[1][2] += ny * nz;
                            acc[2][2] += nz * nz;
                            contact_count += 1;
                        }
                    }
                }
            }
        }

        if contact_count == 0 {
            return None;
        }

        let inv = 1.0 / contact_count as f64;
        let a00 = acc[0][0] * inv;
        let a01 = acc[0][1] * inv;
        let a02 = acc[0][2] * inv;
        let a11 = acc[1][1] * inv;
        let a12 = acc[1][2] * inv;
        let a22 = acc[2][2] * inv;

        let tensor = [
            [a00 as f32, a01 as f32, a02 as f32],
            [a01 as f32, a11 as f32, a12 as f32],
            [a02 as f32, a12 as f32, a22 as f32],
        ];

        let eigenvalues = symmetric_eigenvalues_desc(a00, a11, a22, a01, a02, a12);

        // Deviatoric part Φ' = Φ − I/3; Frobenius norm and von Mises scalar.
        let third = 1.0_f64 / 3.0;
        let d00 = a00 - third;
        let d11 = a11 - third;
        let d22 = a22 - third;
        // ‖Φ'‖_F² = Σ diag² + 2 Σ offdiag².
        let dev_sq = d00 * d00 + d11 * d11 + d22 * d22 + 2.0 * (a01 * a01 + a02 * a02 + a12 * a12);
        let deviatoric_norm = dev_sq.max(0.0).sqrt();
        let anisotropy = (1.5 * dev_sq).max(0.0).sqrt();

        Some(Self {
            contact_count,
            tensor,
            eigenvalues: [
                eigenvalues[0] as f32,
                eigenvalues[1] as f32,
                eigenvalues[2] as f32,
            ],
            deviatoric_norm: deviatoric_norm as f32,
            anisotropy: anisotropy as f32,
        })
    }

    /// Number of contacts that contributed to the tensor.
    pub fn contact_count(&self) -> usize {
        self.contact_count
    }

    /// The symmetric fabric tensor `Φ` (row-major 3×3, unit trace).
    pub fn tensor(&self) -> [[f32; 3]; 3] {
        self.tensor
    }

    /// Trace of `Φ` (nominally `1`).
    pub fn trace(&self) -> f32 {
        self.tensor[0][0] + self.tensor[1][1] + self.tensor[2][2]
    }

    /// Principal fabric intensities (eigenvalues of `Φ`), descending.
    pub fn eigenvalues(&self) -> [f32; 3] {
        self.eigenvalues
    }

    /// Largest principal fabric intensity.
    pub fn major_intensity(&self) -> f32 {
        self.eigenvalues[0]
    }

    /// Smallest principal fabric intensity.
    pub fn minor_intensity(&self) -> f32 {
        self.eigenvalues[2]
    }

    /// Frobenius norm of the deviatoric fabric `Φ' = Φ − I/3`.
    pub fn deviatoric_norm(&self) -> f32 {
        self.deviatoric_norm
    }

    /// Von Mises anisotropy scalar `a = √(3/2 · Φ':Φ')`.
    ///
    /// Zero for a perfectly isotropic fabric; grows with directional
    /// structure.
    pub fn anisotropy(&self) -> f32 {
        self.anisotropy
    }

    /// Eigenvalue spread `λ_max − λ_min`, a simple anisotropy proxy.
    pub fn intensity_spread(&self) -> f32 {
        self.eigenvalues[0] - self.eigenvalues[2]
    }
}

/// Closed-form eigenvalues of a symmetric 3×3 matrix (descending).
///
/// Uses the trigonometric method (Smith, 1961). All intermediate math is in
/// f64 because `acos`/`cos` are disallowed on f32 here.
fn symmetric_eigenvalues_desc(
    a00: f64,
    a11: f64,
    a22: f64,
    a01: f64,
    a02: f64,
    a12: f64,
) -> [f64; 3] {
    let p1 = a01 * a01 + a02 * a02 + a12 * a12;
    if p1 <= 0.0 {
        // Already diagonal.
        let mut e = [a00, a11, a22];
        e.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
        return e;
    }
    let q = (a00 + a11 + a22) / 3.0;
    let d0 = a00 - q;
    let d1 = a11 - q;
    let d2 = a22 - q;
    let p2 = d0 * d0 + d1 * d1 + d2 * d2 + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();
    if p <= 0.0 {
        return [q, q, q];
    }
    let inv_p = 1.0 / p;
    // B = (A - qI)/p; det(B)/2 = r.
    let b00 = d0 * inv_p;
    let b11 = d1 * inv_p;
    let b22 = d2 * inv_p;
    let b01 = a01 * inv_p;
    let b02 = a02 * inv_p;
    let b12 = a12 * inv_p;
    let det_b = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02)
        + b02 * (b01 * b12 - b11 * b02);
    let r = (det_b * 0.5).clamp(-1.0, 1.0);
    let phi = r.acos() / 3.0;
    let two_pi_third = 2.0 * std::f64::consts::PI / 3.0;
    let e1 = q + 2.0 * p * phi.cos();
    let e3 = q + 2.0 * p * (phi + two_pi_third).cos();
    let e2 = 3.0 * q - e1 - e3;
    // e1 is the largest, e3 the smallest by construction.
    [e1, e2, e3]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_input() {
        assert!(FabricTensor::from_contacts(&[], &[], 0.0).is_none());
        let p = [Vec3::ZERO, Vec3::X];
        assert!(FabricTensor::from_contacts(&p, &[1.0], 0.0).is_none());
        assert!(FabricTensor::from_contacts(&p, &[1.0, 1.0], -1.0).is_none());
        assert!(FabricTensor::from_contacts(&p, &[1.0, f32::NAN], 0.0).is_none());
    }

    #[test]
    fn no_contacts_returns_none() {
        // Two spheres far apart never touch.
        let p = [Vec3::ZERO, Vec3::new(100.0, 0.0, 0.0)];
        let r = [1.0, 1.0];
        assert!(FabricTensor::from_contacts(&p, &r, 0.0).is_none());
    }

    #[test]
    fn single_x_contact_is_fully_anisotropic() {
        // One contact along +X → Φ = diag(1,0,0).
        let p = [Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
        let r = [1.0, 1.0];
        let f = FabricTensor::from_contacts(&p, &r, 1e-4).unwrap();
        assert_eq!(f.contact_count(), 1);
        assert!((f.trace() - 1.0).abs() < 1e-5);
        assert!((f.tensor()[0][0] - 1.0).abs() < 1e-5);
        assert!(f.tensor()[1][1].abs() < 1e-5);
        assert!(f.tensor()[2][2].abs() < 1e-5);
        // Eigenvalues {1,0,0}.
        let e = f.eigenvalues();
        assert!((e[0] - 1.0).abs() < 1e-5);
        assert!(e[1].abs() < 1e-5);
        assert!(e[2].abs() < 1e-5);
        // Fully anisotropic: deviatoric norm = √(2/3), von Mises a = 1.
        assert!((f.anisotropy() - 1.0).abs() < 1e-4);
        assert!((f.intensity_spread() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn isotropic_octahedron_has_zero_anisotropy() {
        // A central sphere touched symmetrically along ±X, ±Y, ±Z yields the
        // isotropic fabric Φ = I/3.
        let s = 2.0_f32; // contact distance for unit spheres
        let p = vec![
            Vec3::ZERO,
            Vec3::new(s, 0.0, 0.0),
            Vec3::new(-s, 0.0, 0.0),
            Vec3::new(0.0, s, 0.0),
            Vec3::new(0.0, -s, 0.0),
            Vec3::new(0.0, 0.0, s),
            Vec3::new(0.0, 0.0, -s),
        ];
        let r = vec![1.0_f32; 7];
        let f = FabricTensor::from_contacts(&p, &r, 1e-3).unwrap();
        assert_eq!(f.contact_count(), 6);
        assert!((f.trace() - 1.0).abs() < 1e-5);
        // Diagonal 1/3, off-diagonal 0.
        for i in 0..3 {
            assert!((f.tensor()[i][i] - 1.0 / 3.0).abs() < 1e-5);
        }
        assert!(f.anisotropy() < 1e-4, "isotropic anisotropy ~0");
        assert!(f.deviatoric_norm() < 1e-4);
        for e in f.eigenvalues() {
            assert!((e - 1.0 / 3.0).abs() < 1e-4);
        }
    }

    #[test]
    fn eigenvalues_are_sorted_descending() {
        // A planar square cluster biases the fabric toward the XY plane.
        let s = 2.0_f32;
        let p = vec![
            Vec3::ZERO,
            Vec3::new(s, 0.0, 0.0),
            Vec3::new(0.0, s, 0.0),
            Vec3::new(s, s, 0.0),
        ];
        let r = vec![1.0_f32; 4];
        let f = FabricTensor::from_contacts(&p, &r, 1e-3).unwrap();
        let e = f.eigenvalues();
        assert!(e[0] >= e[1] - 1e-6 && e[1] >= e[2] - 1e-6);
        assert!((e[0] + e[1] + e[2] - 1.0).abs() < 1e-5);
        // Out-of-plane (Z) intensity should be the smallest (zero here).
        assert!(f.minor_intensity() < 1e-4);
    }

    #[test]
    fn tolerance_includes_near_contacts() {
        // Gap of 0.1 beyond touching: excluded at tol 0, included at tol 0.2.
        let p = [Vec3::ZERO, Vec3::new(2.1, 0.0, 0.0)];
        let r = [1.0, 1.0];
        assert!(FabricTensor::from_contacts(&p, &r, 0.0).is_none());
        let f = FabricTensor::from_contacts(&p, &r, 0.2).unwrap();
        assert_eq!(f.contact_count(), 1);
    }
}
