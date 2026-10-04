//! Persistent wet-granular cohesion driver.
//!
//! This composes the two wet-granular building blocks into a single per-frame
//! driver without modifying either:
//!
//! * [`WetBridgeHysteresis`](crate::collider::wet_bridge_hysteresis::WetBridgeHysteresis)
//!   decides *which* grain pairs are bonded, forming bonds on contact and
//!   keeping them through a hysteresis band until they rupture.
//! * [`FiniteLiquidCapillary`](crate::collider::finite_liquid_capillary_forces::FiniteLiquidCapillary)
//!   turns that bonded set plus a finite per-grain liquid budget into central
//!   attractive cohesion forces.
//!
//! The driver owns both pieces and threads them together: each frame it updates
//! the bridge topology from the current grain configuration, then evaluates the
//! finite-liquid capillary forces over exactly the bonds that survived. Keeping
//! the two concerns in separate modules means this driver stays a thin, honest
//! orchestrator — it adds the frame bookkeeping and a combined report, nothing
//! more. Because the capillary forces are central (along grain-centre lines),
//! the driver conserves linear momentum and exerts no net torque.

use glam::Vec3;

use crate::collider::finite_liquid_capillary_forces::FiniteLiquidCapillary;
use crate::collider::wet_bridge_hysteresis::WetBridgeHysteresis;

/// Combined outcome of a single [`WetCohesionDriver::step`] call.
#[derive(Clone, Debug, PartialEq)]
pub struct WetCohesionReport {
    /// Per-grain cohesion force (index-aligned with the grain cloud).
    pub forces: Vec<Vec3>,
    /// Number of bonds active after the topology update.
    pub bridge_count: u32,
    /// Number of bonds that newly formed this frame.
    pub formed: u32,
    /// Number of bonds that ruptured this frame.
    pub ruptured: u32,
    /// Largest single-grain force magnitude in [`forces`](Self::forces).
    pub max_force: f32,
    /// Vector sum of all grain forces; near zero for central forces.
    pub net_force: Vec3,
}

/// Drives persistent wet-granular cohesion by combining bridge hysteresis with
/// finite-liquid capillary forces.
///
/// Construct from a pre-validated [`WetBridgeHysteresis`] and
/// [`FiniteLiquidCapillary`], then call [`step`](Self::step) each frame. The
/// grain count is fixed by the capillary model's liquid distribution; every
/// call must pass exactly that many grains.
#[derive(Clone, Debug)]
pub struct WetCohesionDriver {
    hysteresis: WetBridgeHysteresis,
    capillary: FiniteLiquidCapillary,
}

impl WetCohesionDriver {
    /// Create a driver from its two building blocks.
    pub fn new(hysteresis: WetBridgeHysteresis, capillary: FiniteLiquidCapillary) -> Self {
        Self {
            hysteresis,
            capillary,
        }
    }

    /// Number of grains this driver expects, set by the liquid distribution.
    pub fn grain_count(&self) -> usize {
        self.capillary.liquid().grain_count()
    }

    /// Read-only access to the bridge hysteresis tracker.
    pub fn hysteresis(&self) -> &WetBridgeHysteresis {
        &self.hysteresis
    }

    /// Read-only access to the finite-liquid capillary force model.
    pub fn capillary(&self) -> &FiniteLiquidCapillary {
        &self.capillary
    }

    /// Currently bonded pairs, sorted ascending as `(a, b)` with `a < b`.
    pub fn active_bridges(&self) -> &[(u32, u32)] {
        self.hysteresis.active_bridges()
    }

    /// Advance the wet cohesion state by one frame.
    ///
    /// `positions` and `radii` must each contain exactly
    /// [`grain_count`](Self::grain_count) finite entries (radii strictly
    /// positive). Returns `None` on any invalid input, leaving the stored
    /// topology unchanged.
    ///
    /// Internally it first updates the bridge topology (forming and rupturing
    /// bonds with hysteresis) and then evaluates finite-liquid capillary forces
    /// over the surviving bonds.
    pub fn step(&mut self, positions: &[Vec3], radii: &[f32]) -> Option<WetCohesionReport> {
        let n = self.grain_count();
        if positions.len() != n || radii.len() != n {
            return None;
        }

        let topology = self.hysteresis.update(positions, radii)?;
        // Clone the active set so the immutable borrow of `hysteresis` does not
        // overlap the capillary evaluation below.
        let bridges = self.hysteresis.active_bridges().to_vec();
        let resolution = self.capillary.forces(positions, radii, &bridges)?;

        Some(WetCohesionReport {
            forces: resolution.forces().to_vec(),
            bridge_count: resolution.bridge_count() as u32,
            formed: topology.formed,
            ruptured: topology.ruptured,
            max_force: resolution.max_force(),
            net_force: resolution.net_force(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::capillary_liquid_distribution::LiquidDistribution;

    const GAMMA: f32 = 0.072; // water-like surface tension
    const THETA: f32 = 0.0; // perfectly wetting

    fn driver(formation: f32, rupture: f32, grains: usize, liquid_per: f32) -> WetCohesionDriver {
        let hysteresis = WetBridgeHysteresis::new(formation, rupture).unwrap();
        let liquid = LiquidDistribution::uniform(grains, liquid_per).unwrap();
        let capillary = FiniteLiquidCapillary::new(GAMMA, THETA, liquid).unwrap();
        WetCohesionDriver::new(hysteresis, capillary)
    }

    fn two_grains(gap: f32) -> (Vec<Vec3>, Vec<f32>) {
        let r = 1.0_f32;
        let positions = vec![Vec3::ZERO, Vec3::new(2.0 * r + gap, 0.0, 0.0)];
        let radii = vec![r, r];
        (positions, radii)
    }

    #[test]
    fn grain_count_follows_liquid() {
        let d = driver(0.0, 0.3, 5, 1.0);
        assert_eq!(d.grain_count(), 5);
    }

    #[test]
    fn contact_forms_bridge_and_pulls_together() {
        let mut d = driver(0.0, 0.3, 2, 1.0);
        let (positions, radii) = two_grains(-0.001);
        let report = d.step(&positions, &radii).unwrap();
        assert_eq!(report.formed, 1);
        assert_eq!(report.bridge_count, 1);
        assert_eq!(d.active_bridges(), &[(0, 1)]);
        // Grain 0 is pulled toward grain 1 (+x); grain 1 toward grain 0 (-x).
        assert!(report.forces[0].x > 0.0);
        assert!(report.forces[1].x < 0.0);
        assert!(report.max_force > 0.0);
    }

    #[test]
    fn driver_matches_standalone_capillary() {
        let mut d = driver(0.0, 0.3, 2, 1.0);
        let (positions, radii) = two_grains(-0.001);
        let report = d.step(&positions, &radii).unwrap();

        // Reproduce the same evaluation with a freshly built capillary model
        // over the bond the driver formed.
        let liquid = LiquidDistribution::uniform(2, 1.0).unwrap();
        let capillary = FiniteLiquidCapillary::new(GAMMA, THETA, liquid).unwrap();
        let direct = capillary.forces(&positions, &radii, &[(0, 1)]).unwrap();

        assert_eq!(report.forces, direct.forces().to_vec());
        assert_eq!(report.max_force, direct.max_force());
    }

    #[test]
    fn hysteresis_persists_then_ruptures() {
        let mut d = driver(0.0, 0.3, 2, 1.0);
        // Form on contact.
        let (p0, r0) = two_grains(-0.001);
        d.step(&p0, &r0).unwrap();
        // Separate into the hysteresis band: bond persists, force still present.
        let (p1, r1) = two_grains(0.15);
        let mid = d.step(&p1, &r1).unwrap();
        assert_eq!(mid.formed, 0);
        assert_eq!(mid.ruptured, 0);
        assert_eq!(mid.bridge_count, 1);
        assert!(mid.max_force > 0.0);
        // Pull past the rupture gap: bond breaks, no cohesion force remains.
        let (p2, r2) = two_grains(0.5);
        let far = d.step(&p2, &r2).unwrap();
        assert_eq!(far.ruptured, 1);
        assert_eq!(far.bridge_count, 0);
        assert_eq!(far.max_force, 0.0);
        assert_eq!(far.net_force, Vec3::ZERO);
    }

    #[test]
    fn forces_conserve_momentum() {
        let mut d = driver(0.02, 0.3, 3, 1.0);
        // Three grains in a touching chain along x.
        let r = 1.0_f32;
        let step = 2.0 * r;
        let positions = vec![
            Vec3::ZERO,
            Vec3::new(step, 0.0, 0.0),
            Vec3::new(2.0 * step, 0.0, 0.0),
        ];
        let radii = vec![r; 3];
        let report = d.step(&positions, &radii).unwrap();
        assert_eq!(report.formed, 2);
        // Net force vanishes to within tolerance for central forces.
        assert!(report.net_force.length() < 1.0e-5);
    }

    #[test]
    fn dry_grains_produce_no_force() {
        // Zero liquid per grain: bonds may form geometrically but no capillary
        // volume means no force.
        let mut d = driver(0.0, 0.3, 2, 0.0);
        let (positions, radii) = two_grains(-0.001);
        let report = d.step(&positions, &radii).unwrap();
        assert_eq!(report.max_force, 0.0);
        assert_eq!(report.net_force, Vec3::ZERO);
    }

    #[test]
    fn rejects_wrong_grain_count() {
        let mut d = driver(0.0, 0.3, 2, 1.0);
        let positions = vec![Vec3::ZERO, Vec3::X, Vec3::new(2.0, 0.0, 0.0)];
        let radii = vec![1.0, 1.0, 1.0];
        assert!(d.step(&positions, &radii).is_none());
        // Mismatched positions/radii of the right nominal count are rejected too.
        assert!(d.step(&[Vec3::ZERO, Vec3::X], &[1.0]).is_none());
    }

    #[test]
    fn rejects_invalid_input() {
        let mut d = driver(0.0, 0.3, 2, 1.0);
        let bad = vec![Vec3::ZERO, Vec3::new(f32::NAN, 0.0, 0.0)];
        assert!(d.step(&bad, &[1.0, 1.0]).is_none());
        assert!(d.step(&[Vec3::ZERO, Vec3::X], &[1.0, -1.0]).is_none());
        // State remains clean after rejected calls.
        assert_eq!(d.active_bridges().len(), 0);
    }
}
