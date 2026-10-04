//! Love–Weber (virial) contact stress tensor for a granular assembly.
//!
//! The macroscopic Cauchy stress of a granular packing is the volume average of
//! the contact forces acting over their branch vectors (the vectors joining the
//! two contacting particle centres). With contact force `f^c` and branch vector
//! `l^c` over a representative volume `V`, the symmetric Love–Weber stress is
//!
//! ```text
//! σ_ij = (1 / 2V) Σ_c (f_i^c l_j^c + f_j^c l_i^c).
//! ```
//!
//! From the symmetric tensor we extract the standard invariants: the mean
//! stress (pressure) `p = tr(σ)/3`, the ordered principal stresses
//! `σ1 ≥ σ2 ≥ σ3`, the von Mises deviatoric stress
//! `q = √(3/2 · s:s)` with `s = σ − p I`, the stress ratio `q/p`, and the
//! maximum shear `(σ1 − σ3)/2`. Principal stresses use a closed-form symmetric
//! `3×3` eigenvalue solver (trigonometric method, evaluated in `f64`). This
//! module is a pure analysis of a supplied contact-force field and does not
//! couple to the simulation step.

use glam::Vec3;

/// Ordered eigenvalues (descending) of a symmetric `3×3` matrix given its
/// upper-triangular entries. Uses the trigonometric method in `f64`.
fn symmetric_eigenvalues(a11: f32, a22: f32, a33: f32, a12: f32, a13: f32, a23: f32) -> [f32; 3] {
    let (a11, a22, a33) = (a11 as f64, a22 as f64, a33 as f64);
    let (a12, a13, a23) = (a12 as f64, a13 as f64, a23 as f64);

    let p1 = a12 * a12 + a13 * a13 + a23 * a23;
    if p1 <= 0.0 {
        // Already diagonal.
        let mut eig = [a11, a22, a33];
        eig.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
        return [eig[0] as f32, eig[1] as f32, eig[2] as f32];
    }

    let q = (a11 + a22 + a33) / 3.0;
    let d11 = a11 - q;
    let d22 = a22 - q;
    let d33 = a33 - q;
    let p2 = d11 * d11 + d22 * d22 + d33 * d33 + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();
    if p <= 0.0 {
        return [q as f32, q as f32, q as f32];
    }

    // r = det((A − qI)/p) / 2.
    let inv_p = 1.0 / p;
    let b11 = d11 * inv_p;
    let b22 = d22 * inv_p;
    let b33 = d33 * inv_p;
    let b12 = a12 * inv_p;
    let b13 = a13 * inv_p;
    let b23 = a23 * inv_p;
    let det = b11 * (b22 * b33 - b23 * b23) - b12 * (b12 * b33 - b23 * b13)
        + b13 * (b12 * b23 - b22 * b13);
    let r = (det / 2.0).clamp(-1.0, 1.0);

    let phi = r.acos() / 3.0;
    let two_pi_third = 2.0 * std::f64::consts::PI / 3.0;
    let eig1 = q + 2.0 * p * phi.cos();
    let eig3 = q + 2.0 * p * (phi + two_pi_third).cos();
    let eig2 = 3.0 * q - eig1 - eig3;
    [eig1 as f32, eig2 as f32, eig3 as f32]
}

/// Love–Weber contact stress tensor and its invariants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContactStress {
    contact_count: usize,
    volume: f32,
    tensor: [[f32; 3]; 3],
    principal: [f32; 3],
    mean_stress: f32,
    von_mises: f32,
    max_shear: f32,
}

impl ContactStress {
    /// Computes the symmetric Love–Weber stress from per-contact force and
    /// branch vectors.
    ///
    /// * `forces` — contact force vectors `f^c`.
    /// * `branches` — branch vectors `l^c` (same length as `forces`).
    /// * `volume` — representative volume `V` (`> 0`).
    ///
    /// Returns `None` for empty or mismatched inputs, non-finite values, or a
    /// non-positive volume.
    pub fn compute(forces: &[Vec3], branches: &[Vec3], volume: f32) -> Option<Self> {
        if forces.is_empty() || forces.len() != branches.len() {
            return None;
        }
        if !volume.is_finite() || volume <= 0.0 {
            return None;
        }

        let mut s = [[0.0f32; 3]; 3];
        for (f, l) in forces.iter().zip(branches.iter()) {
            if !f.is_finite() || !l.is_finite() {
                return None;
            }
            let fv = [f.x, f.y, f.z];
            let lv = [l.x, l.y, l.z];
            for (i, row) in s.iter_mut().enumerate() {
                for (j, slot) in row.iter_mut().enumerate() {
                    // Symmetric form ½(f_i l_j + f_j l_i).
                    *slot += 0.5 * (fv[i] * lv[j] + fv[j] * lv[i]);
                }
            }
        }

        let inv_v = 1.0 / volume;
        for row in s.iter_mut() {
            for slot in row.iter_mut() {
                *slot *= inv_v;
                if !slot.is_finite() {
                    return None;
                }
            }
        }

        let principal = symmetric_eigenvalues(s[0][0], s[1][1], s[2][2], s[0][1], s[0][2], s[1][2]);
        let mean_stress = (s[0][0] + s[1][1] + s[2][2]) / 3.0;

        // von Mises q = √(1/2 [(σ1−σ2)² + (σ2−σ3)² + (σ3−σ1)²]).
        let (p1, p2, p3) = (principal[0], principal[1], principal[2]);
        let a = p1 - p2;
        let b = p2 - p3;
        let c = p3 - p1;
        let von_mises = (0.5 * (a * a + b * b + c * c)).max(0.0).sqrt();
        let max_shear = 0.5 * (p1 - p3);

        Some(Self {
            contact_count: forces.len(),
            volume,
            tensor: s,
            principal,
            mean_stress,
            von_mises,
            max_shear,
        })
    }

    /// Number of contacts aggregated.
    pub fn contact_count(&self) -> usize {
        self.contact_count
    }

    /// Representative volume used for the average.
    pub fn volume(&self) -> f32 {
        self.volume
    }

    /// Symmetric stress tensor `σ` (row-major).
    pub fn tensor(&self) -> [[f32; 3]; 3] {
        self.tensor
    }

    /// Ordered principal stresses `σ1 ≥ σ2 ≥ σ3`.
    pub fn principal_stresses(&self) -> [f32; 3] {
        self.principal
    }

    /// Mean stress (pressure) `p = tr(σ)/3`.
    pub fn mean_stress(&self) -> f32 {
        self.mean_stress
    }

    /// von Mises deviatoric stress `q`.
    pub fn von_mises_stress(&self) -> f32 {
        self.von_mises
    }

    /// Maximum shear stress `(σ1 − σ3)/2`.
    pub fn max_shear_stress(&self) -> f32 {
        self.max_shear
    }

    /// Deviatoric stress ratio `q/p`. `None` when the mean stress is ~zero.
    pub fn stress_ratio(&self) -> Option<f32> {
        if self.mean_stress.abs() <= 1e-12 {
            return None;
        }
        Some(self.von_mises / self.mean_stress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_input() {
        assert!(ContactStress::compute(&[], &[], 1.0).is_none());
        assert!(ContactStress::compute(&[Vec3::X], &[Vec3::X, Vec3::Y], 1.0).is_none());
        assert!(ContactStress::compute(&[Vec3::X], &[Vec3::X], 0.0).is_none());
        assert!(ContactStress::compute(&[Vec3::X], &[Vec3::X], -1.0).is_none());
        let bad = Vec3::new(f32::NAN, 0.0, 0.0);
        assert!(ContactStress::compute(&[bad], &[Vec3::X], 1.0).is_none());
    }

    #[test]
    fn isotropic_compression_has_zero_deviator() {
        // Contacts along ±x,±y,±z with f = c·n, l = n → each axis pair gives
        // 2c(e⊗e); σ = (2c/V) I.
        let c = 3.0f32;
        let axes = [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z];
        let forces: Vec<Vec3> = axes.iter().map(|n| *n * c).collect();
        let branches: Vec<Vec3> = axes.to_vec();
        let v = 2.0f32;
        let s = ContactStress::compute(&forces, &branches, v).unwrap();
        let expected_p = 2.0 * c / v; // 3.0
        assert!((s.mean_stress() - expected_p).abs() < 1e-4);
        assert!(s.von_mises_stress() < 1e-4);
        let [p1, p2, p3] = s.principal_stresses();
        assert!((p1 - expected_p).abs() < 1e-4);
        assert!((p2 - expected_p).abs() < 1e-4);
        assert!((p3 - expected_p).abs() < 1e-4);
        assert!(s.stress_ratio().unwrap() < 1e-4);
    }

    #[test]
    fn uniaxial_stress_principal_values() {
        // Only ±x contacts → σ = diag(2c/V, 0, 0).
        let c = 5.0f32;
        let forces = vec![Vec3::X * c, -Vec3::X * c];
        let branches = vec![Vec3::X, -Vec3::X];
        let v = 1.0f32;
        let s = ContactStress::compute(&forces, &branches, v).unwrap();
        let [p1, p2, p3] = s.principal_stresses();
        assert!((p1 - 2.0 * c).abs() < 1e-3);
        assert!(p2.abs() < 1e-3);
        assert!(p3.abs() < 1e-3);
        // Mean = 2c/3V, max shear = σ1/2.
        assert!((s.mean_stress() - 2.0 * c / 3.0).abs() < 1e-3);
        assert!((s.max_shear_stress() - c).abs() < 1e-3);
        assert!(s.von_mises_stress() > 0.0);
    }

    #[test]
    fn principal_stresses_sum_to_trace() {
        let forces = vec![
            Vec3::new(2.0, 1.0, 0.0),
            Vec3::new(0.0, 3.0, 1.0),
            Vec3::new(1.0, 0.0, 2.0),
        ];
        let branches = vec![
            Vec3::new(1.0, 0.0, 0.5),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.5, 0.5, 1.0),
        ];
        let s = ContactStress::compute(&forces, &branches, 1.5).unwrap();
        let t = &s.tensor();
        let trace = t[0][0] + t[1][1] + t[2][2];
        let [p1, p2, p3] = s.principal_stresses();
        assert!((p1 + p2 + p3 - trace).abs() < 1e-3);
        assert!(p1 >= p2 && p2 >= p3);
    }

    #[test]
    fn tensor_is_symmetric() {
        let forces = vec![Vec3::new(1.0, 2.0, 3.0)];
        let branches = vec![Vec3::new(4.0, 5.0, 6.0)];
        let s = ContactStress::compute(&forces, &branches, 1.0).unwrap();
        let t = s.tensor();
        assert!((t[0][1] - t[1][0]).abs() < 1e-6);
        assert!((t[0][2] - t[2][0]).abs() < 1e-6);
        assert!((t[1][2] - t[2][1]).abs() < 1e-6);
    }

    #[test]
    fn volume_scales_stress_inversely() {
        let forces = vec![Vec3::X * 4.0, -Vec3::X * 4.0];
        let branches = vec![Vec3::X, -Vec3::X];
        let a = ContactStress::compute(&forces, &branches, 1.0).unwrap();
        let b = ContactStress::compute(&forces, &branches, 2.0).unwrap();
        assert!((a.mean_stress() - 2.0 * b.mean_stress()).abs() < 1e-4);
    }

    #[test]
    fn known_symmetric_eigenvalues() {
        // diag(5,3,1) with no coupling → eigenvalues 5,3,1 descending.
        let eig = symmetric_eigenvalues(5.0, 3.0, 1.0, 0.0, 0.0, 0.0);
        assert!((eig[0] - 5.0).abs() < 1e-5);
        assert!((eig[1] - 3.0).abs() < 1e-5);
        assert!((eig[2] - 1.0).abs() < 1e-5);
    }
}
