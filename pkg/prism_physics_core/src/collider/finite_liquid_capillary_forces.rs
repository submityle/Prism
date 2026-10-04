//! Finite-liquid capillary cohesion forces.
//!
//! This composes two existing building blocks into a physically richer wet-
//! granular cohesion model without modifying either:
//!
//! * [`LiquidDistribution`](crate::collider::capillary_liquid_distribution::LiquidDistribution)
//!   — a finite per-grain liquid budget shared across the bridges a grain
//!   participates in.
//! * [`CapillaryBridgeModel`](crate::collider::capillary_bridge::CapillaryBridgeModel)
//!   — the Rabinovich pendular-bridge force for a *given* bridge liquid volume.
//!
//! The baseline [`CapillaryBridgeResolver`](crate::collider::capillary_bridge_resolver::CapillaryBridgeResolver)
//! gives every bridge the same liquid volume. Here, instead, each bridge's
//! volume comes from the finite grain budgets: a grain with many neighbours
//! spreads its liquid thinner, so each of its bridges is smaller and weaker.
//! For every active bridge this evaluator builds a per-bridge
//! [`CapillaryBridgeModel`] sized to that bridge's share of liquid and sums the
//! resulting central attractive forces.
//!
//! Topology (which grain pairs are bridged) is an *input* — produced upstream
//! by a broad phase or a hysteresis resolver — so this module stays focused on
//! turning a finite liquid budget plus a bridge set into forces. The forces are
//! central (along the grain-centre line), so like the baseline resolver they
//! conserve linear momentum and exert no torque.

use glam::Vec3;

use crate::collider::capillary_bridge::CapillaryBridgeModel;
use crate::collider::capillary_liquid_distribution::LiquidDistribution;

/// Per-bridge diagnostic emitted by [`FiniteLiquidCapillary::forces`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FiniteLiquidBridge {
    /// Lower grain index of the bridge.
    pub grain_a: u32,
    /// Upper grain index of the bridge.
    pub grain_b: u32,
    /// Liquid volume assigned to this bridge from the finite grain budgets.
    pub liquid_volume: f32,
    /// Magnitude of the attractive force (non-negative).
    pub magnitude: f32,
    /// Surface separation of the two grains (negative when cores overlap).
    pub gap: f32,
}

/// Result of a finite-liquid capillary force evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct FiniteLiquidResolution {
    forces: Vec<Vec3>,
    bridges: Vec<FiniteLiquidBridge>,
}

impl FiniteLiquidResolution {
    /// Per-grain cohesive force (one entry per grain).
    pub fn forces(&self) -> &[Vec3] {
        &self.forces
    }

    /// The bridges that actually carried liquid and produced a force.
    pub fn bridges(&self) -> &[FiniteLiquidBridge] {
        &self.bridges
    }

    /// Number of force-carrying bridges.
    pub fn bridge_count(&self) -> usize {
        self.bridges.len()
    }

    /// Largest bridge force magnitude (zero when there are no bridges).
    pub fn max_force(&self) -> f32 {
        self.bridges.iter().map(|b| b.magnitude).fold(0.0, f32::max)
    }

    /// Net of all per-grain forces. Should be close to zero because every
    /// bridge contributes an equal and opposite pair; a useful conservation
    /// diagnostic.
    pub fn net_force(&self) -> Vec3 {
        self.forces.iter().copied().sum()
    }
}

/// Finite-liquid capillary cohesion evaluator.
///
/// Holds the liquid surface tension, contact angle, and the finite per-grain
/// liquid budget. Call [`forces`](Self::forces) with the current geometry and
/// bridge set to obtain per-grain cohesive forces.
#[derive(Clone, Debug)]
pub struct FiniteLiquidCapillary {
    surface_tension: f32,
    contact_angle: f32,
    liquid: LiquidDistribution,
}

impl FiniteLiquidCapillary {
    /// Build from surface tension `γ > 0`, contact angle `θ ∈ [0, π/2]`, and a
    /// finite per-grain liquid budget.
    ///
    /// Returns `None` if `γ` or `θ` are out of range or non-finite. (The liquid
    /// budget is already validated by [`LiquidDistribution`].)
    pub fn new(
        surface_tension: f32,
        contact_angle: f32,
        liquid: LiquidDistribution,
    ) -> Option<Self> {
        if !surface_tension.is_finite() || !contact_angle.is_finite() {
            return None;
        }
        if surface_tension <= 0.0 {
            return None;
        }
        let half_pi = core::f32::consts::PI / 2.0;
        if !(0.0..=half_pi).contains(&contact_angle) {
            return None;
        }
        Some(Self {
            surface_tension,
            contact_angle,
            liquid,
        })
    }

    /// Liquid surface tension `γ`.
    pub fn surface_tension(&self) -> f32 {
        self.surface_tension
    }

    /// Liquid-solid contact angle `θ`.
    pub fn contact_angle(&self) -> f32 {
        self.contact_angle
    }

    /// The finite per-grain liquid budget.
    pub fn liquid(&self) -> &LiquidDistribution {
        &self.liquid
    }

    /// Evaluate per-grain cohesive forces for the given geometry and bridge
    /// set.
    ///
    /// `positions` and `radii` must have one entry per grain and match the
    /// liquid budget's grain count; positions must be finite and radii finite
    /// and positive. `bridges` lists the active grain pairs (as `(a, b)`); each
    /// must reference valid, distinct grains. Returns `None` on any invalid
    /// input. Bridges whose shared liquid volume is zero (both grains dry)
    /// produce no force and are omitted from the result.
    pub fn forces(
        &self,
        positions: &[Vec3],
        radii: &[f32],
        bridges: &[(u32, u32)],
    ) -> Option<FiniteLiquidResolution> {
        let n = positions.len();
        if radii.len() != n || self.liquid.grain_count() != n {
            return None;
        }
        for (pos, &r) in positions.iter().zip(radii.iter()) {
            if !pos.is_finite() || !r.is_finite() || r <= 0.0 {
                return None;
            }
        }

        // Per-bridge liquid volume from the finite budgets. This also validates
        // that every bridge references distinct, in-range grains.
        let volumes = self.liquid.bridge_volumes(bridges)?;

        let mut forces = vec![Vec3::ZERO; n];
        let mut out = Vec::new();
        for (idx, &(a, b)) in bridges.iter().enumerate() {
            let volume = volumes[idx];
            if volume <= 0.0 {
                // Both grains dry: no bridge forms.
                continue;
            }
            let Some(model) =
                CapillaryBridgeModel::new(self.surface_tension, self.contact_angle, volume)
            else {
                continue;
            };
            let (au, bu) = (a as usize, b as usize);
            let Some(bridge) = model.bridge(positions[au], radii[au], positions[bu], radii[bu])
            else {
                continue;
            };
            forces[au] += bridge.force;
            forces[bu] -= bridge.force;
            out.push(FiniteLiquidBridge {
                grain_a: a,
                grain_b: b,
                liquid_volume: volume,
                magnitude: bridge.magnitude,
                gap: bridge.gap,
            });
        }

        Some(FiniteLiquidResolution {
            forces,
            bridges: out,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GAMMA: f32 = 0.072; // water surface tension, N/m
    const THETA: f32 = 0.0;

    fn two_grains() -> ([Vec3; 2], [f32; 2]) {
        // Two unit-radius-ish grains with a small positive gap.
        let positions = [Vec3::new(-0.505, 0.0, 0.0), Vec3::new(0.505, 0.0, 0.0)];
        let radii = [0.5, 0.5];
        (positions, radii)
    }

    #[test]
    fn new_validates_parameters() {
        let liquid = LiquidDistribution::uniform(2, 1.0).unwrap();
        assert!(FiniteLiquidCapillary::new(GAMMA, THETA, liquid.clone()).is_some());
        assert!(FiniteLiquidCapillary::new(0.0, THETA, liquid.clone()).is_none());
        assert!(FiniteLiquidCapillary::new(GAMMA, -0.1, liquid.clone()).is_none());
        let too_wide = core::f32::consts::PI; // > pi/2
        assert!(FiniteLiquidCapillary::new(GAMMA, too_wide, liquid).is_none());
    }

    #[test]
    fn isolated_pair_matches_standalone_model() {
        let (positions, radii) = two_grains();
        // Each grain holds 1.0; both have degree 1, so the bridge volume is 2.0.
        let liquid = LiquidDistribution::uniform(2, 1.0).unwrap();
        let fl = FiniteLiquidCapillary::new(GAMMA, THETA, liquid).unwrap();
        let res = fl.forces(&positions, &radii, &[(0, 1)]).unwrap();

        // Standalone reference with the same total bridge volume.
        let model = CapillaryBridgeModel::new(GAMMA, THETA, 2.0).unwrap();
        let reference = model
            .bridge(positions[0], radii[0], positions[1], radii[1])
            .unwrap();

        assert_eq!(res.bridge_count(), 1);
        assert!((res.forces()[0] - reference.force).length() < 1e-6);
        assert!((res.forces()[1] + reference.force).length() < 1e-6);
        assert!((res.bridges()[0].liquid_volume - 2.0).abs() < 1e-6);
    }

    #[test]
    fn shared_grain_weakens_bridge() {
        let (positions, radii) = two_grains();
        // Three grains, chain 0-1-2, but geometry only matters for pair (0,1).
        let mut p3 = [positions[0], positions[1], Vec3::new(1.6, 0.0, 0.0)];
        let r3 = [radii[0], radii[1], 0.5];
        // Keep grain 2 near grain 1 so the chain is geometrically sane.
        p3[2] = Vec3::new(1.515, 0.0, 0.0);
        let liquid = LiquidDistribution::uniform(3, 1.0).unwrap();
        let fl = FiniteLiquidCapillary::new(GAMMA, THETA, liquid).unwrap();

        // Chain: grain 1 has degree 2, so bridge (0,1) volume = 1 + 0.5 = 1.5.
        let chain = fl.forces(&p3, &r3, &[(0, 1), (1, 2)]).unwrap();
        // Isolated: only (0,1), both degree 1, volume = 2.0.
        let isolated = fl.forces(&p3, &r3, &[(0, 1)]).unwrap();

        let chain_01 = chain
            .bridges()
            .iter()
            .find(|b| b.grain_a == 0 && b.grain_b == 1)
            .unwrap();
        let iso_01 = &isolated.bridges()[0];
        assert!(
            chain_01.liquid_volume < iso_01.liquid_volume,
            "shared grain should give (0,1) less liquid"
        );
        assert!(
            chain_01.magnitude < iso_01.magnitude,
            "less liquid should mean a weaker bridge"
        );
    }

    #[test]
    fn forces_conserve_momentum() {
        let (positions, radii) = two_grains();
        let liquid = LiquidDistribution::uniform(2, 1.0).unwrap();
        let fl = FiniteLiquidCapillary::new(GAMMA, THETA, liquid).unwrap();
        let res = fl.forces(&positions, &radii, &[(0, 1)]).unwrap();
        assert!(res.net_force().length() < 1e-6);
    }

    #[test]
    fn force_is_attractive() {
        let (positions, radii) = two_grains();
        let liquid = LiquidDistribution::uniform(2, 1.0).unwrap();
        let fl = FiniteLiquidCapillary::new(GAMMA, THETA, liquid).unwrap();
        let res = fl.forces(&positions, &radii, &[(0, 1)]).unwrap();
        // Force on grain 0 points toward grain 1 (positive x here).
        let toward = positions[1] - positions[0];
        assert!(res.forces()[0].dot(toward) > 0.0);
    }

    #[test]
    fn dry_grains_produce_no_force() {
        let (positions, radii) = two_grains();
        let liquid = LiquidDistribution::new(vec![0.0, 0.0]).unwrap();
        let fl = FiniteLiquidCapillary::new(GAMMA, THETA, liquid).unwrap();
        let res = fl.forces(&positions, &radii, &[(0, 1)]).unwrap();
        assert_eq!(res.bridge_count(), 0);
        assert!(res.net_force().length() < 1e-12);
        assert_eq!(res.max_force(), 0.0);
    }

    #[test]
    fn rejects_invalid_inputs() {
        let (positions, radii) = two_grains();
        let liquid = LiquidDistribution::uniform(2, 1.0).unwrap();
        let fl = FiniteLiquidCapillary::new(GAMMA, THETA, liquid).unwrap();
        // Radii length mismatch.
        assert!(fl.forces(&positions, &[0.5], &[(0, 1)]).is_none());
        // Out-of-range bridge index.
        assert!(fl.forces(&positions, &radii, &[(0, 2)]).is_none());
        // Non-finite position.
        let bad = [Vec3::new(f32::NAN, 0.0, 0.0), positions[1]];
        assert!(fl.forces(&bad, &radii, &[(0, 1)]).is_none());
    }
}
