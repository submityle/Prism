//! Per-particle velocity-gradient tensor and its strain-rate / spin split.
//!
//! Given a particle velocity field sampled at known reference positions, the
//! local velocity gradient `L` is the best-fit linear map from relative
//! positions to relative velocities within a neighbourhood:
//!
//! ```text
//! dv_j ≈ L · dr_j      (dr_j = x_j - x_i,  dv_j = v_j - v_i)
//! L = (Σ dv_j ⊗ dr_j) · (Σ dr_j ⊗ dr_j)^{-1}
//! ```
//!
//! `L` decomposes into a symmetric strain-rate tensor `D = ½(L + Lᵀ)` and an
//! antisymmetric spin tensor `W = ½(L − Lᵀ)`. The trace of `L` is the velocity
//! divergence (volumetric strain rate); the axial vector of `W` gives the
//! vorticity `ω = ∇ × v`; and the second invariant of the deviatoric
//! strain rate gives the equivalent shear rate used by inertial-number
//! rheology. This is a pure analytic diagnostic over a supplied field, with no
//! coupling to the simulation pipeline.
//!
//! The convention is Z-up, matching the rest of the granular stack. Each
//! particle needs at least three non-coplanar neighbours within the cutoff for
//! the gradient to be defined; otherwise it is reported as `None`.
//!
//! No Unreal Engine source or derived code.

use glam::Vec3;

/// A plain 3×3 matrix stored row-major, used locally for the least-squares fit.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Mat3 {
    m: [[f32; 3]; 3],
}

impl Mat3 {
    const ZERO: Self = Self { m: [[0.0; 3]; 3] };

    fn add_outer(&mut self, a: Vec3, b: Vec3) {
        let a = [a.x, a.y, a.z];
        let b = [b.x, b.y, b.z];
        for (row, &ai) in self.m.iter_mut().zip(a.iter()) {
            for (slot, &bj) in row.iter_mut().zip(b.iter()) {
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

    fn inverse(&self, eps: f32) -> Option<Self> {
        let det = self.determinant();
        if det.abs() <= eps {
            return None;
        }
        let inv_det = 1.0 / det;
        let m = &self.m;
        let mut out = [[0.0f32; 3]; 3];
        out[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
        out[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
        out[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
        out[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
        out[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
        out[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;
        out[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
        out[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det;
        out[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;
        Some(Self { m: out })
    }

    fn mul(&self, rhs: &Mat3) -> Mat3 {
        let mut out = [[0.0f32; 3]; 3];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, slot) in row.iter_mut().enumerate() {
                let mut acc = 0.0;
                for k in 0..3 {
                    acc += self.m[i][k] * rhs.m[k][j];
                }
                *slot = acc;
            }
        }
        Mat3 { m: out }
    }
}

/// Per-particle velocity-gradient diagnostic over a reference configuration.
///
/// Build with [`VelocityGradientField::analyze`]; query per-particle tensors
/// and invariants with the accessors. Particles with too few non-coplanar
/// neighbours have an undefined gradient and report `None`.
#[derive(Clone, Debug)]
pub struct VelocityGradientField {
    cutoff: f32,
    neighbor_counts: Vec<u32>,
    gradients: Vec<Option<[[f32; 3]; 3]>>,
}

impl VelocityGradientField {
    /// Fits a local velocity gradient at every particle.
    ///
    /// `reference` and `velocities` must have equal, non-zero length; `cutoff`
    /// must be positive. Returns `None` if those invariants fail or any input
    /// component is non-finite.
    #[must_use]
    pub fn analyze(reference: &[Vec3], velocities: &[Vec3], cutoff: f32) -> Option<Self> {
        if reference.is_empty() || reference.len() != velocities.len() {
            return None;
        }
        if !cutoff.is_finite() || cutoff <= 0.0 {
            return None;
        }
        if reference.iter().any(|p| !p.is_finite()) || velocities.iter().any(|v| !v.is_finite()) {
            return None;
        }

        let cutoff_sq = cutoff * cutoff;
        let eps = 1.0e-9;
        let mut neighbor_counts = Vec::with_capacity(reference.len());
        let mut gradients = Vec::with_capacity(reference.len());

        for (i, (&xi, &vi)) in reference.iter().zip(velocities.iter()).enumerate() {
            let mut moment = Mat3::ZERO; // Σ dr ⊗ dr
            let mut mixed = Mat3::ZERO; // Σ dv ⊗ dr
            let mut count = 0u32;
            for (j, (&xj, &vj)) in reference.iter().zip(velocities.iter()).enumerate() {
                if i == j {
                    continue;
                }
                let dr = xj - xi;
                if dr.length_squared() > cutoff_sq {
                    continue;
                }
                let dv = vj - vi;
                moment.add_outer(dr, dr);
                mixed.add_outer(dv, dr);
                count += 1;
            }
            neighbor_counts.push(count);
            let gradient = if count >= 3 {
                moment.inverse(eps).map(|inv| mixed.mul(&inv).m)
            } else {
                None
            };
            gradients.push(gradient);
        }

        Some(Self {
            cutoff,
            neighbor_counts,
            gradients,
        })
    }

    /// Neighbour search cutoff radius used by [`VelocityGradientField::analyze`].
    #[must_use]
    pub fn neighbor_cutoff(&self) -> f32 {
        self.cutoff
    }

    /// Number of particles analysed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.gradients.len()
    }

    /// Whether the field is empty (never true for a value returned by `analyze`).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.gradients.is_empty()
    }

    /// Per-particle neighbour counts within the cutoff.
    #[must_use]
    pub fn neighbor_counts(&self) -> &[u32] {
        &self.neighbor_counts
    }

    /// Number of particles with a defined velocity gradient.
    #[must_use]
    pub fn defined_count(&self) -> usize {
        self.gradients.iter().filter(|g| g.is_some()).count()
    }

    /// The fitted velocity gradient `L` at particle `i`, if defined.
    #[must_use]
    pub fn velocity_gradient(&self, i: usize) -> Option<[[f32; 3]; 3]> {
        self.gradients.get(i).copied().flatten()
    }

    /// The symmetric strain-rate tensor `D = ½(L + Lᵀ)` at particle `i`.
    #[must_use]
    pub fn strain_rate(&self, i: usize) -> Option<[[f32; 3]; 3]> {
        let l = self.velocity_gradient(i)?;
        let mut d = [[0.0f32; 3]; 3];
        for (a, row) in d.iter_mut().enumerate() {
            for (b, slot) in row.iter_mut().enumerate() {
                *slot = 0.5 * (l[a][b] + l[b][a]);
            }
        }
        Some(d)
    }

    /// The antisymmetric spin tensor `W = ½(L − Lᵀ)` at particle `i`.
    #[must_use]
    pub fn spin(&self, i: usize) -> Option<[[f32; 3]; 3]> {
        let l = self.velocity_gradient(i)?;
        let mut w = [[0.0f32; 3]; 3];
        for (a, row) in w.iter_mut().enumerate() {
            for (b, slot) in row.iter_mut().enumerate() {
                *slot = 0.5 * (l[a][b] - l[b][a]);
            }
        }
        Some(w)
    }

    /// The velocity divergence `tr(L)` (volumetric strain rate) at particle `i`.
    #[must_use]
    pub fn divergence(&self, i: usize) -> Option<f32> {
        let l = self.velocity_gradient(i)?;
        Some(l[0][0] + l[1][1] + l[2][2])
    }

    /// The vorticity vector `ω = ∇ × v` at particle `i`, derived from the spin.
    #[must_use]
    pub fn vorticity(&self, i: usize) -> Option<Vec3> {
        let l = self.velocity_gradient(i)?;
        Some(Vec3::new(
            l[2][1] - l[1][2],
            l[0][2] - l[2][0],
            l[1][0] - l[0][1],
        ))
    }

    /// The equivalent shear rate `γ̇ = √(2 D' : D')` where `D'` is the
    /// deviatoric strain rate at particle `i`.
    #[must_use]
    pub fn equivalent_shear_rate(&self, i: usize) -> Option<f32> {
        let d = self.strain_rate(i)?;
        let trace = d[0][0] + d[1][1] + d[2][2];
        let mean = trace / 3.0;
        let mut sum = 0.0f32;
        for (a, row) in d.iter().enumerate() {
            for (b, &value) in row.iter().enumerate() {
                let dev = if a == b { value - mean } else { value };
                sum += dev * dev;
            }
        }
        Some((2.0 * sum).sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> Vec<Vec3> {
        let mut pts = Vec::new();
        for ix in -1..=1 {
            for iy in -1..=1 {
                for iz in -1..=1 {
                    pts.push(Vec3::new(ix as f32, iy as f32, iz as f32));
                }
            }
        }
        pts
    }

    fn center_index(reference: &[Vec3]) -> usize {
        reference
            .iter()
            .position(|p| *p == Vec3::ZERO)
            .expect("grid contains origin")
    }

    #[test]
    fn analyze_rejects_bad_input() {
        let r = grid();
        let v = vec![Vec3::ZERO; r.len()];
        assert!(VelocityGradientField::analyze(&[], &[], 1.5).is_none());
        assert!(VelocityGradientField::analyze(&r, &v[..r.len() - 1], 1.5).is_none());
        assert!(VelocityGradientField::analyze(&r, &v, 0.0).is_none());
        let mut bad = v.clone();
        bad[0] = Vec3::new(f32::NAN, 0.0, 0.0);
        assert!(VelocityGradientField::analyze(&r, &bad, 1.5).is_none());
    }

    #[test]
    fn uniform_field_has_zero_gradient() {
        let r = grid();
        let v = vec![Vec3::new(2.0, -1.0, 0.5); r.len()];
        let field = VelocityGradientField::analyze(&r, &v, 1.5).expect("ok");
        let c = center_index(&r);
        assert!(field.divergence(c).unwrap().abs() < 1.0e-5);
        assert!(field.equivalent_shear_rate(c).unwrap().abs() < 1.0e-5);
        assert!(field.vorticity(c).unwrap().length() < 1.0e-5);
    }

    #[test]
    fn simple_shear_recovers_rate_and_vorticity() {
        // v = (gamma * y, 0, 0): L[0][1] = gamma, vorticity_z = -gamma.
        let gamma = 0.7f32;
        let r = grid();
        let v: Vec<Vec3> = r.iter().map(|p| Vec3::new(gamma * p.y, 0.0, 0.0)).collect();
        let field = VelocityGradientField::analyze(&r, &v, 1.5).expect("ok");
        let c = center_index(&r);
        let l = field.velocity_gradient(c).expect("defined");
        assert!((l[0][1] - gamma).abs() < 1.0e-4);
        assert!(l[0][0].abs() < 1.0e-4 && l[1][0].abs() < 1.0e-4);
        assert!(field.divergence(c).unwrap().abs() < 1.0e-4);
        let w = field.vorticity(c).unwrap();
        assert!((w.z + gamma).abs() < 1.0e-4);
        // D[0][1] = gamma/2, deviatoric equals full D (trace 0): shear rate = gamma.
        assert!((field.equivalent_shear_rate(c).unwrap() - gamma).abs() < 1.0e-4);
    }

    #[test]
    fn isotropic_expansion_is_pure_divergence() {
        let alpha = 0.4f32;
        let r = grid();
        let v: Vec<Vec3> = r.iter().map(|p| *p * alpha).collect();
        let field = VelocityGradientField::analyze(&r, &v, 1.5).expect("ok");
        let c = center_index(&r);
        assert!((field.divergence(c).unwrap() - 3.0 * alpha).abs() < 1.0e-4);
        assert!(field.equivalent_shear_rate(c).unwrap().abs() < 1.0e-4);
        assert!(field.vorticity(c).unwrap().length() < 1.0e-4);
    }

    #[test]
    fn rigid_rotation_is_pure_spin() {
        // v = Omega x r about z: no strain rate, vorticity_z = 2 Omega.
        let omega = 0.5f32;
        let r = grid();
        let v: Vec<Vec3> = r
            .iter()
            .map(|p| Vec3::new(-omega * p.y, omega * p.x, 0.0))
            .collect();
        let field = VelocityGradientField::analyze(&r, &v, 1.5).expect("ok");
        let c = center_index(&r);
        assert!(field.equivalent_shear_rate(c).unwrap().abs() < 1.0e-4);
        assert!(field.divergence(c).unwrap().abs() < 1.0e-4);
        assert!((field.vorticity(c).unwrap().z - 2.0 * omega).abs() < 1.0e-4);
    }

    #[test]
    fn sparse_particle_has_undefined_gradient() {
        let r = vec![Vec3::ZERO, Vec3::new(0.5, 0.0, 0.0)];
        let v = vec![Vec3::ZERO, Vec3::new(0.1, 0.0, 0.0)];
        let field = VelocityGradientField::analyze(&r, &v, 1.0).expect("ok");
        assert_eq!(field.defined_count(), 0);
        assert!(field.velocity_gradient(0).is_none());
        assert_eq!(field.neighbor_counts()[0], 1);
    }
}
