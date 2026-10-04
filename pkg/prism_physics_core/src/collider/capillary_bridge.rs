//! Capillary (pendular liquid-bridge) cohesion between wet grains.
//!
//! Dry grains only ever *push* on one another: the contact laws in
//! [`rotational_contact`](super::rotational_contact) and friends emit a force
//! the instant two surfaces overlap and nothing when they separate. A *wet*
//! granular medium behaves very differently. A thin film of liquid wicks into
//! the gap between neighbouring grains and forms a pendular bridge whose
//! surface tension *pulls* the grains together. That cohesion is what lets damp
//! sand hold a vertical wall or a sandcastle keep its shape — the angle of
//! repose of a wet heap is markedly steeper than a dry one.
//!
//! Crucially the attraction acts across a *gap*: the bridge survives until the
//! surface separation exceeds a rupture distance, well after the solid cores
//! have parted. A purely contact-based model can never reproduce that, so the
//! capillary force is its own law layered on top of (not replacing) the contact
//! response.
//!
//! # Model
//!
//! This module implements the explicit pendular-bridge force of Rabinovich et
//! al. (2005), the workhorse closed form used in wet-DEM codes. For two spheres
//! of radii `rₐ`, `r_b` the geometry is condensed into the reduced radius
//!
//! ```text
//! R = 2·rₐ·r_b / (rₐ + r_b)
//! ```
//!
//! A bridge of liquid volume `V`, surface tension `γ` and solid–liquid contact
//! angle `θ` spanning a surface separation `H ≥ 0` pulls with magnitude
//!
//! ```text
//! F₀      = 2π·R·γ·cos θ                              (adhesion at contact, H = 0)
//! d_sp(H) = ½·H·(−1 + √(1 + 2V / (π·R·H²)))           (embracing distance)
//! F(H)    = F₀ / (1 + H / (2·d_sp(H)))                (0 < H ≤ H_rupture)
//! ```
//!
//! and ruptures once the grains draw apart beyond (Lian et al. 1993)
//!
//! ```text
//! H_rupture = (1 + θ/2)·V^{1/3}.
//! ```
//!
//! The force is purely attractive and central (directed along the line of
//! centres), so it conserves linear and angular momentum exactly: the reaction
//! on the partner grain is equal and opposite and the shared line of action
//! produces no net torque. Overlapping grains (`H < 0`) are treated as touching
//! (`H = 0`); the squeezed bridge then contributes its maximum pull `F₀` on top
//! of whatever the contact law reports.

use glam::Vec3;
use std::f32::consts::PI;

/// Parameters of the wetting liquid shared by every pendular bridge.
///
/// All quantities are SI: surface tension in N/m, contact angle in radians,
/// liquid volume per bridge in m³. The angle is restricted to the physically
/// meaningful wetting range `[0, π/2]`; a volume and tension must be strictly
/// positive for a bridge to exist at all.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CapillaryBridgeModel {
    surface_tension: f32,
    contact_angle: f32,
    liquid_volume: f32,
}

impl CapillaryBridgeModel {
    /// Builds a bridge model, returning `None` for non-physical parameters:
    /// non-finite inputs, a non-positive surface tension or volume, or a
    /// contact angle outside `[0, π/2]`.
    pub fn new(surface_tension: f32, contact_angle: f32, liquid_volume: f32) -> Option<Self> {
        if !surface_tension.is_finite() || !contact_angle.is_finite() || !liquid_volume.is_finite()
        {
            return None;
        }
        if surface_tension <= 0.0 || liquid_volume <= 0.0 {
            return None;
        }
        let half_pi = PI / 2.0;
        if !(0.0..=half_pi).contains(&contact_angle) {
            return None;
        }
        Some(Self {
            surface_tension,
            contact_angle,
            liquid_volume,
        })
    }

    /// Liquid surface tension `γ` (N/m).
    pub fn surface_tension(&self) -> f32 {
        self.surface_tension
    }

    /// Solid–liquid contact angle `θ` (radians).
    pub fn contact_angle(&self) -> f32 {
        self.contact_angle
    }

    /// Liquid volume `V` bound in a single bridge (m³).
    pub fn liquid_volume(&self) -> f32 {
        self.liquid_volume
    }

    /// Reduced radius `R = 2·rₐ·r_b / (rₐ + r_b)` of a grain pair.
    ///
    /// Returns `None` unless both radii are finite and strictly positive.
    pub fn reduced_radius(&self, radius_a: f32, radius_b: f32) -> Option<f32> {
        if !radius_a.is_finite() || !radius_b.is_finite() || radius_a <= 0.0 || radius_b <= 0.0 {
            return None;
        }
        Some(2.0 * radius_a * radius_b / (radius_a + radius_b))
    }

    /// Surface separation at which the bridge snaps, `H_rupture = (1 + θ/2)·V^{1/3}`.
    pub fn rupture_distance(&self) -> f32 {
        let cube_root = (self.liquid_volume as f64).cbrt() as f32;
        (1.0 + 0.5 * self.contact_angle) * cube_root
    }

    /// Peak adhesion `F₀ = 2π·R·γ·cos θ` reached when the grains touch.
    ///
    /// Returns `None` for a degenerate radius pair.
    pub fn max_force(&self, radius_a: f32, radius_b: f32) -> Option<f32> {
        let reduced = self.reduced_radius(radius_a, radius_b)?;
        let cos_theta = (self.contact_angle as f64).cos() as f32;
        Some(2.0 * PI * reduced * self.surface_tension * cos_theta)
    }

    /// Attractive force magnitude for a bridge spanning surface separation
    /// `gap` between the two grains.
    ///
    /// A negative `gap` (overlapping cores) is clamped to contact (`H = 0`),
    /// yielding the peak pull [`max_force`](Self::max_force). Returns `None`
    /// once `gap` exceeds [`rupture_distance`](Self::rupture_distance) or for a
    /// degenerate radius pair.
    pub fn force_at_gap(&self, radius_a: f32, radius_b: f32, gap: f32) -> Option<f32> {
        if !gap.is_finite() {
            return None;
        }
        if gap > self.rupture_distance() {
            return None;
        }
        let reduced = self.reduced_radius(radius_a, radius_b)?;
        let cos_theta = (self.contact_angle as f64).cos() as f32;
        let f0 = 2.0 * PI * reduced * self.surface_tension * cos_theta;
        let separation = gap.max(0.0);
        if separation <= 0.0 {
            return Some(f0);
        }
        // Rabinovich embracing distance d_sp, always strictly positive for H > 0.
        let inner = 1.0 + 2.0 * self.liquid_volume / (PI * reduced * separation * separation);
        let embracing = 0.5 * separation * (-1.0 + inner.sqrt());
        Some(f0 / (1.0 + separation / (2.0 * embracing)))
    }

    /// Resolves the pendular bridge between two grain centres.
    ///
    /// Returns the force acting on grain `A`, directed toward grain `B` (the
    /// attraction pulls the pair together); the reaction on `B` is its exact
    /// negation. Returns `None` when the centres coincide, any input is
    /// non-finite, a radius is non-positive, or the grains are farther apart
    /// than the rupture distance (no bridge).
    pub fn bridge(
        &self,
        center_a: Vec3,
        radius_a: f32,
        center_b: Vec3,
        radius_b: f32,
    ) -> Option<CapillaryBridge> {
        if !radius_a.is_finite() || !radius_b.is_finite() || radius_a <= 0.0 || radius_b <= 0.0 {
            return None;
        }
        let offset = center_b - center_a;
        let distance = offset.length();
        if !distance.is_finite() || distance <= 0.0 {
            return None;
        }
        let gap = distance - (radius_a + radius_b);
        let rupture = self.rupture_distance();
        if gap > rupture {
            return None;
        }
        let magnitude = self.force_at_gap(radius_a, radius_b, gap)?;
        let direction = offset / distance;
        Some(CapillaryBridge {
            force: direction * magnitude,
            magnitude,
            gap,
            rupture_distance: rupture,
        })
    }
}

/// Resolved pendular bridge acting on grain `A`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CapillaryBridge {
    /// Attractive force on grain `A`, pointing toward grain `B`. The reaction
    /// on `B` is `-force`.
    pub force: Vec3,
    /// Magnitude of the attraction (always non-negative).
    pub magnitude: f32,
    /// Surface separation of the grains (negative when the cores overlap).
    pub gap: f32,
    /// Separation at which this bridge would rupture.
    pub rupture_distance: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> CapillaryBridgeModel {
        // Water-like film: γ = 0.072 N/m, perfectly wetting, 1 mm³ bridge.
        CapillaryBridgeModel::new(0.072, 0.0, 1.0e-9).unwrap()
    }

    #[test]
    fn new_rejects_non_physical_parameters() {
        assert!(CapillaryBridgeModel::new(0.0, 0.0, 1.0e-9).is_none());
        assert!(CapillaryBridgeModel::new(-0.1, 0.0, 1.0e-9).is_none());
        assert!(CapillaryBridgeModel::new(0.072, 0.0, 0.0).is_none());
        assert!(CapillaryBridgeModel::new(0.072, 0.0, -1.0e-9).is_none());
        // Angle outside [0, π/2].
        assert!(CapillaryBridgeModel::new(0.072, -0.1, 1.0e-9).is_none());
        assert!(CapillaryBridgeModel::new(0.072, PI, 1.0e-9).is_none());
        // Non-finite inputs.
        assert!(CapillaryBridgeModel::new(f32::NAN, 0.0, 1.0e-9).is_none());
        assert!(CapillaryBridgeModel::new(0.072, f32::INFINITY, 1.0e-9).is_none());
        assert!(CapillaryBridgeModel::new(0.072, 0.0, f32::NAN).is_none());
        // A valid model is accepted.
        assert!(CapillaryBridgeModel::new(0.072, PI / 3.0, 1.0e-9).is_some());
    }

    #[test]
    fn getters_round_trip() {
        let m = CapillaryBridgeModel::new(0.05, PI / 4.0, 2.0e-9).unwrap();
        assert!((m.surface_tension() - 0.05).abs() < 1.0e-9);
        assert!((m.contact_angle() - PI / 4.0).abs() < 1.0e-6);
        assert!((m.liquid_volume() - 2.0e-9).abs() < 1.0e-18);
    }

    #[test]
    fn reduced_radius_matches_formula() {
        let m = model();
        // Equal spheres: R = r.
        assert!((m.reduced_radius(0.5, 0.5).unwrap() - 0.5).abs() < 1.0e-6);
        // Unequal: R = 2·1·3 / (1 + 3) = 1.5.
        assert!((m.reduced_radius(1.0, 3.0).unwrap() - 1.5).abs() < 1.0e-6);
        // Degenerate radii.
        assert!(m.reduced_radius(0.0, 1.0).is_none());
        assert!(m.reduced_radius(1.0, -1.0).is_none());
        assert!(m.reduced_radius(f32::NAN, 1.0).is_none());
    }

    #[test]
    fn rupture_distance_tracks_volume_and_angle() {
        let v = 1.0e-9_f32;
        let dry = CapillaryBridgeModel::new(0.072, 0.0, v).unwrap();
        let cube_root = (v as f64).cbrt() as f32;
        // θ = 0 → H_rupture = V^{1/3}.
        assert!((dry.rupture_distance() - cube_root).abs() < 1.0e-9);
        // A larger contact angle widens the rupture distance.
        let wet = CapillaryBridgeModel::new(0.072, PI / 2.0, v).unwrap();
        assert!(wet.rupture_distance() > dry.rupture_distance());
        let expected = (1.0 + 0.5 * (PI / 2.0)) * cube_root;
        assert!((wet.rupture_distance() - expected).abs() < 1.0e-9);
    }

    #[test]
    fn max_force_matches_laplace_young() {
        let m = model();
        let reduced = m.reduced_radius(0.5, 0.5).unwrap();
        let expected = 2.0 * PI * reduced * 0.072 * ((0.0_f64).cos() as f32);
        assert!((m.max_force(0.5, 0.5).unwrap() - expected).abs() < 1.0e-6);
        assert!(m.max_force(0.0, 0.5).is_none());
    }

    #[test]
    fn force_peaks_at_contact_and_decays_with_gap() {
        let m = model();
        let rupture = m.rupture_distance();
        let contact = m.force_at_gap(0.5, 0.5, 0.0).unwrap();
        // Contact force equals the closed-form peak.
        assert!((contact - m.max_force(0.5, 0.5).unwrap()).abs() < 1.0e-6);
        // Sample the force on a monotonically increasing separation.
        let mut previous = contact;
        for i in 1..=8 {
            let gap = rupture * (i as f32) / 9.0;
            let f = m.force_at_gap(0.5, 0.5, gap).unwrap();
            // Attraction stays positive and never increases with separation.
            assert!(f > 0.0);
            assert!(f <= previous + 1.0e-9);
            previous = f;
        }
    }

    #[test]
    fn overlap_clamps_to_peak_force() {
        let m = model();
        let contact = m.force_at_gap(0.5, 0.5, 0.0).unwrap();
        // Overlapping cores feel the squeezed bridge at its peak pull.
        let overlapped = m.force_at_gap(0.5, 0.5, -0.01).unwrap();
        assert!((overlapped - contact).abs() < 1.0e-6);
    }

    #[test]
    fn bridge_breaks_beyond_rupture_distance() {
        let m = model();
        let rupture = m.rupture_distance();
        // Just inside the rupture distance: a bridge exists.
        let near = m.force_at_gap(0.5, 0.5, rupture * 0.99);
        assert!(near.is_some());
        // Beyond it: no bridge.
        assert!(m.force_at_gap(0.5, 0.5, rupture * 1.01).is_none());
    }

    #[test]
    fn bridge_pulls_grains_together() {
        let m = model();
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0 + 1.0e-4, 0.0, 0.0);
        let bridge = m.bridge(a, 0.5, b, 0.5).unwrap();
        // The force on A points toward B (attractive).
        assert!(bridge.force.dot(b - a) > 0.0);
        assert!(bridge.magnitude > 0.0);
        // Reported gap is the surface separation.
        assert!((bridge.gap - 1.0e-4).abs() < 1.0e-5);
    }

    #[test]
    fn bridge_obeys_newtons_third_law() {
        let m = model();
        let a = Vec3::new(0.1, 0.2, 0.3);
        let b = Vec3::new(0.1, 0.2, 1.1);
        let on_a = m.bridge(a, 0.5, b, 0.5).unwrap();
        let on_b = m.bridge(b, 0.5, a, 0.5).unwrap();
        // Equal and opposite reaction.
        assert!((on_a.force + on_b.force).length() < 1.0e-5);
        assert!((on_a.magnitude - on_b.magnitude).abs() < 1.0e-6);
    }

    #[test]
    fn bridge_rejects_degenerate_geometry() {
        let m = model();
        let p = Vec3::new(1.0, 2.0, 3.0);
        // Coincident centres have no bridge direction.
        assert!(m.bridge(p, 0.5, p, 0.5).is_none());
        // Non-positive radius.
        assert!(m.bridge(Vec3::ZERO, 0.0, Vec3::X, 0.5).is_none());
        // Grains farther apart than the rupture distance: no bridge.
        let far = Vec3::new(5.0, 0.0, 0.0);
        assert!(m.bridge(Vec3::ZERO, 0.5, far, 0.5).is_none());
    }

    #[test]
    fn wetting_angle_weakens_adhesion() {
        // cos θ shrinks the pull as the liquid wets the solid less.
        let dry = CapillaryBridgeModel::new(0.072, 0.0, 1.0e-9).unwrap();
        let partial = CapillaryBridgeModel::new(0.072, PI / 3.0, 1.0e-9).unwrap();
        assert!(partial.max_force(0.5, 0.5).unwrap() < dry.max_force(0.5, 0.5).unwrap());
    }
}
