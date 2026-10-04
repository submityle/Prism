//! Falk–Langer non-affine displacement (`D²min`) and local deformation gradient.
//!
//! Given a reference and a current configuration of the same grains, the local
//! deformation around particle `i` is summarised by the best-fit affine map
//! `F_i` that most closely reproduces the motion of its neighbours. Following
//! Falk and Langer (1998), with neighbour offsets `d0 = r_j − r_i` in the
//! reference state and `dt` in the current state,
//!
//! ```text
//! X = Σ_j dt ⊗ d0,   Y = Σ_j d0 ⊗ d0,   F_i = X · Y⁻¹,
//! D²min_i = Σ_j |dt − F_i · d0|².
//! ```
//!
//! `F_i` is the affine (locally homogeneous) part of the deformation; the
//! residual `D²min_i` measures the non-affine, plastic rearrangement and is the
//! standard detector of shear transformations and strain localisation. A purely
//! affine deformation gives `D²min = 0` and recovers the imposed gradient
//! exactly. This module is a pure kinematic analysis of two position snapshots
//! and does not couple to the simulation step.

use glam::Vec3;

/// Symmetric/general 3×3 matrix stored row-major, used for the local fit.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Mat3 {
    m: [[f32; 3]; 3],
}

impl Mat3 {
    const ZERO: Self = Self { m: [[0.0; 3]; 3] };

    /// Accumulates the outer product `a ⊗ b` (`m[i][j] += a_i b_j`).
    fn add_outer(&mut self, a: Vec3, b: Vec3) {
        let av = [a.x, a.y, a.z];
        let bv = [b.x, b.y, b.z];
        for (row, &ai) in self.m.iter_mut().zip(av.iter()) {
            for (slot, &bj) in row.iter_mut().zip(bv.iter()) {
                *slot += ai * bj;
            }
        }
    }

    fn determinant(&self) -> f32 {
        let m = &self.m;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    /// Inverse via the adjugate; `None` when (near-)singular.
    fn inverse(&self, epsilon: f32) -> Option<Mat3> {
        let det = self.determinant();
        if !det.is_finite() || det.abs() <= epsilon {
            return None;
        }
        let inv_det = 1.0 / det;
        let m = &self.m;
        let c = [
            [
                (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
                (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
                (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
            ],
            [
                (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
                (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
                (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
            ],
            [
                (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
                (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
                (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
            ],
        ];
        Some(Mat3 { m: c })
    }

    /// Matrix product `self · rhs`.
    fn mul(&self, rhs: &Mat3) -> Mat3 {
        let mut out = Mat3::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += self.m[i][k] * rhs.m[k][j];
                }
                out.m[i][j] = s;
            }
        }
        out
    }

    /// Matrix–vector product `self · v`.
    fn mul_vec(&self, v: Vec3) -> Vec3 {
        let vv = [v.x, v.y, v.z];
        let mut out = [0.0f32; 3];
        for (oi, row) in out.iter_mut().zip(self.m.iter()) {
            let mut s = 0.0;
            for (&mij, &vj) in row.iter().zip(vv.iter()) {
                s += mij * vj;
            }
            *oi = s;
        }
        Vec3::new(out[0], out[1], out[2])
    }
}

/// Per-particle non-affine displacement field for two snapshots.
#[derive(Debug, Clone)]
pub struct NonaffineField {
    neighbor_cutoff: f32,
    neighbor_counts: Vec<usize>,
    deformation_gradients: Vec<Option<[[f32; 3]; 3]>>,
    d2min: Vec<Option<f32>>,
}

impl NonaffineField {
    /// Computes the local deformation gradient and `D²min` for every particle.
    ///
    /// * `reference` — positions in the reference configuration.
    /// * `current` — positions in the current configuration (same length).
    /// * `neighbor_cutoff` — radius (`> 0`) selecting neighbours in the
    ///   reference state.
    ///
    /// A particle needs at least three neighbours spanning three dimensions for
    /// `Y` to be invertible; otherwise its gradient and `D²min` are `None`.
    /// Returns `None` for mismatched lengths, an empty system, a non-positive
    /// cutoff, or non-finite coordinates.
    pub fn analyze(reference: &[Vec3], current: &[Vec3], neighbor_cutoff: f32) -> Option<Self> {
        if reference.is_empty() || reference.len() != current.len() {
            return None;
        }
        if !neighbor_cutoff.is_finite() || neighbor_cutoff <= 0.0 {
            return None;
        }
        for (r, c) in reference.iter().zip(current.iter()) {
            if !r.is_finite() || !c.is_finite() {
                return None;
            }
        }

        let cutoff_sq = neighbor_cutoff * neighbor_cutoff;
        // Scale the singularity threshold with the neighbourhood size so it is
        // dimensionally consistent across cutoffs.
        let singular_eps = 1e-6 * cutoff_sq * cutoff_sq * cutoff_sq;

        let n = reference.len();
        let mut neighbor_counts = vec![0usize; n];
        let mut deformation_gradients = vec![None; n];
        let mut d2min = vec![None; n];

        for i in 0..n {
            let ri0 = reference[i];
            let rit = current[i];

            let mut neighbors: Vec<usize> = Vec::new();
            for (j, rj0) in reference.iter().enumerate() {
                if j == i {
                    continue;
                }
                if (*rj0 - ri0).length_squared() <= cutoff_sq {
                    neighbors.push(j);
                }
            }
            neighbor_counts[i] = neighbors.len();
            if neighbors.len() < 3 {
                continue;
            }

            let mut x = Mat3::ZERO;
            let mut y = Mat3::ZERO;
            for &j in &neighbors {
                let d0 = reference[j] - ri0;
                let dt = current[j] - rit;
                x.add_outer(dt, d0);
                y.add_outer(d0, d0);
            }

            let Some(y_inv) = y.inverse(singular_eps) else {
                continue;
            };
            let f = x.mul(&y_inv);

            let mut residual = 0.0f32;
            for &j in &neighbors {
                let d0 = reference[j] - ri0;
                let dt = current[j] - rit;
                let diff = dt - f.mul_vec(d0);
                residual += diff.length_squared();
            }
            if !residual.is_finite() {
                continue;
            }

            deformation_gradients[i] = Some(f.m);
            d2min[i] = Some(residual);
        }

        Some(Self {
            neighbor_cutoff,
            neighbor_counts,
            deformation_gradients,
            d2min,
        })
    }

    /// Neighbour-selection cutoff used for the fit.
    pub fn neighbor_cutoff(&self) -> f32 {
        self.neighbor_cutoff
    }

    /// Number of particles analysed (same as the input length).
    pub fn len(&self) -> usize {
        self.d2min.len()
    }

    /// Whether the field is empty.
    pub fn is_empty(&self) -> bool {
        self.d2min.is_empty()
    }

    /// Neighbour counts within the cutoff, per particle.
    pub fn neighbor_counts(&self) -> &[usize] {
        &self.neighbor_counts
    }

    /// Best-fit affine deformation gradient `F_i` (row-major), per particle.
    /// `None` when the local system was under-determined or singular.
    pub fn deformation_gradient(&self, index: usize) -> Option<[[f32; 3]; 3]> {
        self.deformation_gradients.get(index).copied().flatten()
    }

    /// `D²min_i` per particle (`None` where no gradient could be fit).
    pub fn d2min(&self) -> &[Option<f32>] {
        &self.d2min
    }

    /// Number of particles with a defined `D²min`.
    pub fn defined_count(&self) -> usize {
        self.d2min.iter().filter(|v| v.is_some()).count()
    }

    /// Mean `D²min` over particles with a defined value. `None` when none are
    /// defined.
    pub fn mean_d2min(&self) -> Option<f32> {
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for v in self.d2min.iter().flatten() {
            sum += *v;
            count += 1;
        }
        if count == 0 {
            return None;
        }
        Some(sum / count as f32)
    }

    /// Maximum `D²min` over defined particles (localisation hot spot). `None`
    /// when none are defined.
    pub fn max_d2min(&self) -> Option<f32> {
        self.d2min
            .iter()
            .flatten()
            .copied()
            .fold(None, |acc, v| Some(acc.map_or(v, |a: f32| a.max(v))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a 3×3×3 unit-spacing grid centred near the origin.
    fn grid() -> Vec<Vec3> {
        let mut pts = Vec::new();
        for ix in 0..3 {
            for iy in 0..3 {
                for iz in 0..3 {
                    pts.push(Vec3::new(ix as f32, iy as f32, iz as f32));
                }
            }
        }
        pts
    }

    fn apply_affine(pts: &[Vec3], f: [[f32; 3]; 3], t: Vec3) -> Vec<Vec3> {
        pts.iter()
            .map(|p| {
                Vec3::new(
                    f[0][0] * p.x + f[0][1] * p.y + f[0][2] * p.z + t.x,
                    f[1][0] * p.x + f[1][1] * p.y + f[1][2] * p.z + t.y,
                    f[2][0] * p.x + f[2][1] * p.y + f[2][2] * p.z + t.z,
                )
            })
            .collect()
    }

    #[test]
    fn rejects_bad_input() {
        let r = grid();
        assert!(NonaffineField::analyze(&[], &[], 1.5).is_none());
        assert!(NonaffineField::analyze(&r, &r[..5], 1.5).is_none());
        assert!(NonaffineField::analyze(&r, &r, 0.0).is_none());
        let mut bad = r.clone();
        bad[0] = Vec3::new(f32::NAN, 0.0, 0.0);
        assert!(NonaffineField::analyze(&bad, &r, 1.5).is_none());
    }

    #[test]
    fn pure_translation_is_affine() {
        let r = grid();
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let c = apply_affine(&r, identity, Vec3::new(2.0, -1.0, 0.5));
        let field = NonaffineField::analyze(&r, &c, 1.5).unwrap();
        // Interior particle (1,1,1) is index 13.
        let d2 = field.d2min()[13].unwrap();
        assert!(d2 < 1e-6);
        let f = field.deformation_gradient(13).unwrap();
        assert!((f[0][0] - 1.0).abs() < 1e-5);
        assert!((f[1][1] - 1.0).abs() < 1e-5);
        assert!((f[0][1]).abs() < 1e-5);
    }

    #[test]
    fn pure_shear_recovers_gradient_with_zero_residual() {
        let r = grid();
        let gamma = 0.25;
        let shear = [[1.0, gamma, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let c = apply_affine(&r, shear, Vec3::ZERO);
        let field = NonaffineField::analyze(&r, &c, 1.5).unwrap();
        let f = field.deformation_gradient(13).unwrap();
        assert!((f[0][1] - gamma).abs() < 1e-4);
        assert!((f[0][0] - 1.0).abs() < 1e-4);
        assert!((f[2][2] - 1.0).abs() < 1e-4);
        assert!(field.d2min()[13].unwrap() < 1e-5);
    }

    #[test]
    fn uniform_stretch_recovers_diagonal() {
        let r = grid();
        let stretch = [[1.2, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 1.1]];
        let c = apply_affine(&r, stretch, Vec3::new(0.3, 0.0, -0.2));
        let field = NonaffineField::analyze(&r, &c, 1.5).unwrap();
        let f = field.deformation_gradient(13).unwrap();
        assert!((f[0][0] - 1.2).abs() < 1e-4);
        assert!((f[1][1] - 0.9).abs() < 1e-4);
        assert!((f[2][2] - 1.1).abs() < 1e-4);
        assert!(field.d2min()[13].unwrap() < 1e-5);
    }

    #[test]
    fn nonaffine_rearrangement_has_positive_d2min() {
        let r = grid();
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut c = apply_affine(&r, identity, Vec3::ZERO);
        // Displace the central particle's neighbour non-affinely.
        c[14] += Vec3::new(0.3, -0.2, 0.15);
        let field = NonaffineField::analyze(&r, &c, 1.5).unwrap();
        assert!(field.d2min()[13].unwrap() > 1e-3);
    }

    #[test]
    fn isolated_particles_have_no_gradient() {
        // Two far-apart particles: neither has 3 neighbours.
        let r = vec![Vec3::ZERO, Vec3::new(100.0, 0.0, 0.0)];
        let c = r.clone();
        let field = NonaffineField::analyze(&r, &c, 1.5).unwrap();
        assert_eq!(field.defined_count(), 0);
        assert!(field.d2min()[0].is_none());
        assert!(field.mean_d2min().is_none());
        assert!(field.max_d2min().is_none());
    }

    #[test]
    fn summary_statistics_are_consistent() {
        let r = grid();
        let shear = [[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut c = apply_affine(&r, shear, Vec3::ZERO);
        c[13] += Vec3::new(0.25, 0.0, 0.0);
        let field = NonaffineField::analyze(&r, &c, 1.5).unwrap();
        assert!(field.defined_count() > 0);
        let mean = field.mean_d2min().unwrap();
        let max = field.max_d2min().unwrap();
        assert!(max >= mean);
        assert!(mean >= 0.0);
        assert_eq!(field.len(), r.len());
        assert!(!field.is_empty());
    }
}
