//! Bimodal strong/weak contact-force network split (Radjai decomposition).
//!
//! In a jammed granular packing the contact forces are strongly heterogeneous.
//! Radjai and co-workers showed the network separates into two populations:
//!
//! * the **strong network** of contacts carrying a normal force above the mean
//!   `f̄`, which forms the load-bearing force chains and is highly anisotropic
//!   along the major principal stress; and
//! * the **weak network** (`f ≤ f̄`), which is quasi-isotropic and props the
//!   strong chains laterally.
//!
//! A hallmark is that the minority strong contacts carry the majority of the
//! load. This module splits a set of contacts (unit normals plus normal-force
//! magnitudes) at the mean force and reports the population sizes, the fraction
//! of load each carries, and the fabric anisotropy of each sub-network. The
//! contact fabric tensor is `Φ = (1/N) Σ n ⊗ n` (unit trace), and the scalar
//! anisotropy is the deviatoric invariant `a = √(3/2 · Φ' : Φ')`. This module is
//! a pure analysis of a supplied contact-force field and does not couple to the
//! simulation step.

use glam::Vec3;

/// Fabric-anisotropy accumulator for a set of unit contact normals.
#[derive(Debug, Clone, Copy, Default)]
struct FabricAccumulator {
    count: usize,
    xx: f32,
    yy: f32,
    zz: f32,
    xy: f32,
    xz: f32,
    yz: f32,
}

impl FabricAccumulator {
    fn push(&mut self, n: Vec3) {
        self.count += 1;
        self.xx += n.x * n.x;
        self.yy += n.y * n.y;
        self.zz += n.z * n.z;
        self.xy += n.x * n.y;
        self.xz += n.x * n.z;
        self.yz += n.y * n.z;
    }

    /// Deviatoric fabric anisotropy `√(3/2 Φ':Φ')`; `None` if empty.
    fn anisotropy(&self) -> Option<f32> {
        if self.count == 0 {
            return None;
        }
        let inv = 1.0 / self.count as f32;
        let (pxx, pyy, pzz) = (self.xx * inv, self.yy * inv, self.zz * inv);
        let (pxy, pxz, pyz) = (self.xy * inv, self.xz * inv, self.yz * inv);
        // Deviator: subtract trace/3 from the diagonal (trace = 1 for unit n).
        let trace = pxx + pyy + pzz;
        let third = trace / 3.0;
        let dxx = pxx - third;
        let dyy = pyy - third;
        let dzz = pzz - third;
        let contraction =
            dxx * dxx + dyy * dyy + dzz * dzz + 2.0 * (pxy * pxy + pxz * pxz + pyz * pyz);
        Some((1.5 * contraction).max(0.0).sqrt())
    }
}

/// Strong/weak decomposition of a contact-force network.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForceChainNetwork {
    contact_count: usize,
    strong_count: usize,
    weak_count: usize,
    mean_normal_force: f32,
    total_normal_force: f32,
    strong_force: f32,
    strong_anisotropy: Option<f32>,
    weak_anisotropy: Option<f32>,
    whole_anisotropy: Option<f32>,
}

impl ForceChainNetwork {
    /// Splits a contact-force network at the mean normal force.
    ///
    /// * `normals` — contact normal directions (need not be pre-normalised).
    /// * `normal_forces` — per-contact normal-force magnitudes (`≥ 0`), same
    ///   length as `normals`.
    ///
    /// Returns `None` for empty or mismatched inputs, non-finite values,
    /// negative forces, a degenerate (zero-length) normal, or a non-positive
    /// total force (no load to split).
    pub fn analyze(normals: &[Vec3], normal_forces: &[f32]) -> Option<Self> {
        if normals.is_empty() || normals.len() != normal_forces.len() {
            return None;
        }

        let contact_count = normals.len();
        let mut unit_normals: Vec<Vec3> = Vec::with_capacity(contact_count);
        let mut total_normal_force = 0.0f32;
        for (n, &f) in normals.iter().zip(normal_forces.iter()) {
            if !n.is_finite() || !f.is_finite() || f < 0.0 {
                return None;
            }
            let len_sq = n.length_squared();
            if len_sq <= 1e-20 {
                return None;
            }
            unit_normals.push(*n / len_sq.sqrt());
            total_normal_force += f;
        }
        if !total_normal_force.is_finite() || total_normal_force <= 0.0 {
            return None;
        }

        let mean_normal_force = total_normal_force / contact_count as f32;

        let mut strong = FabricAccumulator::default();
        let mut weak = FabricAccumulator::default();
        let mut whole = FabricAccumulator::default();
        let mut strong_force = 0.0f32;
        for (n, &f) in unit_normals.iter().zip(normal_forces.iter()) {
            whole.push(*n);
            if f > mean_normal_force {
                strong.push(*n);
                strong_force += f;
            } else {
                weak.push(*n);
            }
        }

        Some(Self {
            contact_count,
            strong_count: strong.count,
            weak_count: weak.count,
            mean_normal_force,
            total_normal_force,
            strong_force,
            strong_anisotropy: strong.anisotropy(),
            weak_anisotropy: weak.anisotropy(),
            whole_anisotropy: whole.anisotropy(),
        })
    }

    /// Total number of contacts.
    pub fn contact_count(&self) -> usize {
        self.contact_count
    }

    /// Number of strong contacts (`f > f̄`).
    pub fn strong_count(&self) -> usize {
        self.strong_count
    }

    /// Number of weak contacts (`f ≤ f̄`).
    pub fn weak_count(&self) -> usize {
        self.weak_count
    }

    /// Mean normal force `f̄`.
    pub fn mean_normal_force(&self) -> f32 {
        self.mean_normal_force
    }

    /// Total normal force `Σf`.
    pub fn total_normal_force(&self) -> f32 {
        self.total_normal_force
    }

    /// Fraction of contacts in the strong network (by count).
    pub fn strong_count_fraction(&self) -> f32 {
        self.strong_count as f32 / self.contact_count as f32
    }

    /// Fraction of the total normal force carried by the strong network.
    pub fn strong_force_fraction(&self) -> f32 {
        self.strong_force / self.total_normal_force
    }

    /// Fabric anisotropy of the strong sub-network; `None` if it is empty.
    pub fn strong_anisotropy(&self) -> Option<f32> {
        self.strong_anisotropy
    }

    /// Fabric anisotropy of the weak sub-network; `None` if it is empty.
    pub fn weak_anisotropy(&self) -> Option<f32> {
        self.weak_anisotropy
    }

    /// Fabric anisotropy of the whole contact network.
    pub fn whole_anisotropy(&self) -> Option<f32> {
        self.whole_anisotropy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn isotropic_normals() -> Vec<Vec3> {
        vec![Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z]
    }

    #[test]
    fn rejects_bad_input() {
        assert!(ForceChainNetwork::analyze(&[], &[]).is_none());
        assert!(ForceChainNetwork::analyze(&[Vec3::X], &[1.0, 2.0]).is_none());
        assert!(ForceChainNetwork::analyze(&[Vec3::X], &[-1.0]).is_none());
        assert!(ForceChainNetwork::analyze(&[Vec3::ZERO], &[1.0]).is_none());
        assert!(ForceChainNetwork::analyze(&[Vec3::X], &[0.0]).is_none()); // no load
        assert!(ForceChainNetwork::analyze(&[Vec3::X], &[f32::NAN]).is_none());
    }

    #[test]
    fn uniform_forces_have_empty_strong_network() {
        // All equal → none strictly above the mean.
        let normals = isotropic_normals();
        let forces = vec![2.0f32; 6];
        let net = ForceChainNetwork::analyze(&normals, &forces).unwrap();
        assert_eq!(net.strong_count(), 0);
        assert_eq!(net.weak_count(), 6);
        assert_eq!(net.strong_force_fraction(), 0.0);
        assert!(net.strong_anisotropy().is_none());
        assert!((net.mean_normal_force() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn strong_network_carries_disproportionate_load() {
        // One large force among many small ones: minority carries majority.
        let normals = vec![Vec3::X; 10];
        let mut forces = vec![1.0f32; 9];
        forces.push(50.0);
        let net = ForceChainNetwork::analyze(&normals, &forces).unwrap();
        assert_eq!(net.strong_count(), 1);
        assert!(net.strong_count_fraction() < 0.2);
        assert!(net.strong_force_fraction() > 0.8);
        // Strong carries more than its share.
        assert!(net.strong_force_fraction() > net.strong_count_fraction());
    }

    #[test]
    fn isotropic_network_has_near_zero_anisotropy() {
        let normals = isotropic_normals();
        let forces = vec![1.0f32; 6];
        let net = ForceChainNetwork::analyze(&normals, &forces).unwrap();
        assert!(net.whole_anisotropy().unwrap() < 1e-5);
    }

    #[test]
    fn fully_aligned_network_has_unit_anisotropy() {
        // All normals along x → Φ = diag(1,0,0), a = √(3/2·2/3) = 1.
        let normals = vec![Vec3::X, -Vec3::X, Vec3::X, -Vec3::X];
        let forces = vec![1.0f32, 1.0, 1.0, 1.0];
        let net = ForceChainNetwork::analyze(&normals, &forces).unwrap();
        assert!((net.whole_anisotropy().unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn strong_network_more_anisotropic_than_weak() {
        // Strong contacts aligned along x (force chains), weak isotropic.
        let mut normals = vec![Vec3::X, -Vec3::X];
        let mut forces = vec![10.0f32, 10.0];
        for n in isotropic_normals() {
            normals.push(n);
            forces.push(1.0);
        }
        let net = ForceChainNetwork::analyze(&normals, &forces).unwrap();
        assert_eq!(net.strong_count(), 2);
        let strong = net.strong_anisotropy().unwrap();
        let weak = net.weak_anisotropy().unwrap();
        assert!(strong > weak);
        assert!(strong > 0.9); // aligned
    }

    #[test]
    fn normals_are_normalised_internally() {
        // Scaling the normals must not change the fabric result.
        let normals: Vec<Vec3> = isotropic_normals().iter().map(|n| *n * 7.3).collect();
        let forces = vec![1.0f32; 6];
        let net = ForceChainNetwork::analyze(&normals, &forces).unwrap();
        assert!(net.whole_anisotropy().unwrap() < 1e-5);
    }
}
