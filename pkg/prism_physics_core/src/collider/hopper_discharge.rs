//! Hopper / silo discharge diagnostics: Beverloo flow prediction and an
//! orifice-crossing census for a draining granular packing.
//!
//! When a packed hopper (see
//! [`wedge_hopper`](super::boundary_container::wedge_hopper)) is opened, grains
//! drain through the bottom slot at a rate that — remarkably — is almost
//! independent of the fill height above it and is instead set by the orifice
//! size. The empirical **Beverloo correlation** captures this: for a long slot
//! of clear width `W` and length `L` discharging grains of diameter `d`, the
//! mass flow rate is
//!
//! ```text
//! Q = C · ρ_b · √g · L · (W − k·d)^{3/2}
//! ```
//!
//! where `ρ_b` is the bulk density, `g` gravity, `C` an empirical discharge
//! coefficient (≈ 0.55–0.65) and `k` a shape factor (≈ 1.4–2.9) that shrinks
//! the *effective* aperture because grain centres cannot reach the very edge.
//! When `W ≤ k·d` the arch spans the slot and flow stops — the orifice jams.
//!
//! This module provides:
//!
//! * [`BeverlooSlot`] — the predictive correlation for a slot orifice;
//! * [`DischargeCensus`] — an exact classification of a packing state into
//!   grains retained above the discharge plane versus material that has passed
//!   below it, splitting straddling grains with the spherical-segment volume;
//! * [`mass_flow_between`] — the empirical mass flow rate between two censuses a
//!   time `dt` apart, for cross-checking a DEM run against the prediction.
//!
//! Everything here is a pure, deterministic analysis; nothing is derived from
//! Unreal Engine source.

use glam::Vec3;

/// Grain-material volume of a sphere centred at height `zc` with radius `r`
/// lying *below* the plane `z = plane`. Computed in `f64` via the spherical
/// segment integral `∫ π·(r² − u²) du = π·[r²·u − u³/3]` with `u = z − zc`.
fn volume_below(zc: f64, r: f64, plane: f64) -> f64 {
    let hi = plane.min(zc + r) - zc;
    let lo = -r;
    if hi <= lo {
        return 0.0;
    }
    std::f64::consts::PI * (r * r * (hi - lo) - (hi * hi * hi - lo * lo * lo) / 3.0)
}

/// Beverloo discharge correlation for a long rectangular slot orifice.
///
/// Build one with [`BeverlooSlot::new`]. The slot is characterised by its clear
/// width `W` (the short, flow-limiting dimension) and length `L` (the long
/// dimension, which enters the flow rate linearly).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BeverlooSlot {
    discharge_coeff: f32,
    shape_factor: f32,
    width: f32,
    length: f32,
}

impl BeverlooSlot {
    /// Creates a slot model from the discharge coefficient `C`, shape factor
    /// `k`, clear slot `width`, and slot `length`.
    ///
    /// Returns `None` unless every argument is finite, `C > 0`, `k ≥ 0`,
    /// `width > 0`, and `length > 0`.
    #[must_use]
    pub fn new(discharge_coeff: f32, shape_factor: f32, width: f32, length: f32) -> Option<Self> {
        let finite = discharge_coeff.is_finite()
            && shape_factor.is_finite()
            && width.is_finite()
            && length.is_finite();
        if !finite {
            return None;
        }
        if discharge_coeff <= 0.0 || shape_factor < 0.0 || width <= 0.0 || length <= 0.0 {
            return None;
        }
        Some(Self {
            discharge_coeff,
            shape_factor,
            width,
            length,
        })
    }

    /// Clear slot width `W`.
    #[must_use]
    pub fn width(&self) -> f32 {
        self.width
    }

    /// Slot length `L`.
    #[must_use]
    pub fn length(&self) -> f32 {
        self.length
    }

    /// Effective aperture `W − k·d` for grains of diameter `d`. May be
    /// non-positive, which signals a jammed orifice.
    #[must_use]
    pub fn effective_aperture(&self, grain_diameter: f32) -> f32 {
        self.width - self.shape_factor * grain_diameter
    }

    /// Whether the orifice jams for grains of diameter `d` (`W ≤ k·d`).
    #[must_use]
    pub fn jams(&self, grain_diameter: f32) -> bool {
        self.effective_aperture(grain_diameter) <= 0.0
    }

    /// Predicted mass flow rate `Q = C·ρ_b·√g·L·(W − k·d)^{3/2}`.
    ///
    /// Returns `None` unless `bulk_density`, `gravity`, and `grain_diameter`
    /// are finite with `bulk_density > 0`, `gravity > 0`, and
    /// `grain_diameter > 0`. A jammed orifice yields `Some(0.0)`.
    #[must_use]
    pub fn mass_flow_rate(
        &self,
        bulk_density: f32,
        gravity: f32,
        grain_diameter: f32,
    ) -> Option<f32> {
        if !(bulk_density.is_finite() && gravity.is_finite() && grain_diameter.is_finite()) {
            return None;
        }
        if bulk_density <= 0.0 || gravity <= 0.0 || grain_diameter <= 0.0 {
            return None;
        }
        let aperture = self.effective_aperture(grain_diameter);
        if aperture <= 0.0 {
            return Some(0.0);
        }
        // (W − k·d)^{3/2} = aperture · √aperture, avoiding powf.
        let aperture_three_halves = aperture * aperture.sqrt();
        let q = self.discharge_coeff
            * bulk_density
            * gravity.sqrt()
            * self.length
            * aperture_three_halves;
        Some(q)
    }

    /// Predicted volumetric flow rate `Q / ρ_b`.
    #[must_use]
    pub fn volumetric_flow_rate(
        &self,
        bulk_density: f32,
        gravity: f32,
        grain_diameter: f32,
    ) -> Option<f32> {
        let q = self.mass_flow_rate(bulk_density, gravity, grain_diameter)?;
        Some(q / bulk_density)
    }
}

/// Census of a packing state relative to a horizontal discharge plane.
///
/// Build one with [`DischargeCensus::classify`]. A grain is counted as
/// *discharged* when its centre lies below the plane and *retained* otherwise;
/// the material volumes split each grain exactly at the plane, so a grain
/// straddling the orifice contributes to both volume totals.
#[derive(Clone, Debug, PartialEq)]
pub struct DischargeCensus {
    discharge_z: f32,
    grain_count: usize,
    retained_count: usize,
    discharged_count: usize,
    retained_volume: f64,
    discharged_volume: f64,
}

impl DischargeCensus {
    /// Classifies parallel `positions`/`radii` against the plane `z = discharge_z`.
    ///
    /// Returns `None` unless the arrays share the same length, every value is
    /// finite, every radius is strictly positive, and `discharge_z` is finite.
    /// An empty packing is accepted and yields zeroed counts and volumes.
    #[must_use]
    pub fn classify(positions: &[Vec3], radii: &[f32], discharge_z: f32) -> Option<Self> {
        if positions.len() != radii.len() {
            return None;
        }
        if !discharge_z.is_finite() {
            return None;
        }
        if positions.iter().any(|p| !p.is_finite()) {
            return None;
        }
        if radii.iter().any(|r| !r.is_finite() || *r <= 0.0) {
            return None;
        }

        let plane = discharge_z as f64;
        let four_thirds_pi = 4.0 / 3.0 * std::f64::consts::PI;
        let mut retained_count = 0;
        let mut discharged_count = 0;
        let mut retained_volume = 0.0_f64;
        let mut discharged_volume = 0.0_f64;

        for (&p, &r) in positions.iter().zip(radii.iter()) {
            let zc = p.z as f64;
            let rd = r as f64;
            let total = four_thirds_pi * rd * rd * rd;
            let below = volume_below(zc, rd, plane);
            discharged_volume += below;
            retained_volume += total - below;
            if p.z < discharge_z {
                discharged_count += 1;
            } else {
                retained_count += 1;
            }
        }

        Some(Self {
            discharge_z,
            grain_count: positions.len(),
            retained_count,
            discharged_count,
            retained_volume,
            discharged_volume,
        })
    }

    /// Discharge plane height.
    #[must_use]
    pub fn discharge_z(&self) -> f32 {
        self.discharge_z
    }

    /// Total number of grains censused.
    #[must_use]
    pub fn grain_count(&self) -> usize {
        self.grain_count
    }

    /// Number of grains whose centre is at or above the discharge plane.
    #[must_use]
    pub fn retained_count(&self) -> usize {
        self.retained_count
    }

    /// Number of grains whose centre is below the discharge plane.
    #[must_use]
    pub fn discharged_count(&self) -> usize {
        self.discharged_count
    }

    /// Grain-material volume at or above the discharge plane.
    #[must_use]
    pub fn retained_volume(&self) -> f32 {
        self.retained_volume as f32
    }

    /// Grain-material volume that has passed below the discharge plane.
    #[must_use]
    pub fn discharged_volume(&self) -> f32 {
        self.discharged_volume as f32
    }

    /// Internal f64 discharged volume, used by [`mass_flow_between`].
    fn discharged_volume_f64(&self) -> f64 {
        self.discharged_volume
    }
}

/// Empirical mass flow rate `ρ · ΔV_discharged / dt` between two censuses taken
/// a time `dt` apart (with `after` the later state).
///
/// Returns `None` unless `dt > 0` and `density` is finite and positive. The
/// result is signed: a negative value means material moved back above the
/// plane between the two samples.
#[must_use]
pub fn mass_flow_between(
    before: &DischargeCensus,
    after: &DischargeCensus,
    dt: f32,
    density: f32,
) -> Option<f32> {
    if !(dt.is_finite() && density.is_finite()) {
        return None;
    }
    if dt <= 0.0 || density <= 0.0 {
        return None;
    }
    let delta = after.discharged_volume_f64() - before.discharged_volume_f64();
    Some((density as f64 * delta / dt as f64) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FOUR_THIRDS_PI: f64 = 4.0 / 3.0 * std::f64::consts::PI;

    fn sphere_volume(r: f32) -> f32 {
        let rd = r as f64;
        (FOUR_THIRDS_PI * rd * rd * rd) as f32
    }

    #[test]
    fn beverloo_rejects_invalid() {
        assert!(BeverlooSlot::new(0.0, 1.4, 0.1, 0.5).is_none());
        assert!(BeverlooSlot::new(0.6, -0.1, 0.1, 0.5).is_none());
        assert!(BeverlooSlot::new(0.6, 1.4, 0.0, 0.5).is_none());
        assert!(BeverlooSlot::new(0.6, 1.4, 0.1, 0.0).is_none());
        assert!(BeverlooSlot::new(f32::NAN, 1.4, 0.1, 0.5).is_none());
    }

    #[test]
    fn beverloo_jams_when_aperture_closes() {
        let slot = BeverlooSlot::new(0.6, 2.0, 0.1, 0.5).unwrap();
        // d = 0.06 → k·d = 0.12 > width 0.1 → jam.
        assert!(slot.jams(0.06));
        assert_eq!(slot.mass_flow_rate(1500.0, 9.81, 0.06).unwrap(), 0.0);
        // d = 0.02 → k·d = 0.04 < 0.1 → flows.
        assert!(!slot.jams(0.02));
        assert!(slot.mass_flow_rate(1500.0, 9.81, 0.02).unwrap() > 0.0);
    }

    #[test]
    fn beverloo_rejects_bad_flow_args() {
        let slot = BeverlooSlot::new(0.6, 1.4, 0.1, 0.5).unwrap();
        assert!(slot.mass_flow_rate(0.0, 9.81, 0.01).is_none());
        assert!(slot.mass_flow_rate(1500.0, 0.0, 0.01).is_none());
        assert!(slot.mass_flow_rate(1500.0, 9.81, 0.0).is_none());
        assert!(slot.mass_flow_rate(1500.0, f32::INFINITY, 0.01).is_none());
    }

    #[test]
    fn beverloo_scales_as_three_halves_power_and_linear_length() {
        let slot = BeverlooSlot::new(0.6, 1.4, 0.2, 0.5).unwrap();
        let rho = 1500.0;
        let g = 9.81;
        // Two diameters → two effective apertures.
        let d1 = 0.02;
        let d2 = 0.05;
        let a1 = slot.effective_aperture(d1);
        let a2 = slot.effective_aperture(d2);
        let q1 = slot.mass_flow_rate(rho, g, d1).unwrap();
        let q2 = slot.mass_flow_rate(rho, g, d2).unwrap();
        let expected_ratio = (a2 / a1) * (a2 / a1).sqrt(); // (a2/a1)^{3/2}
        assert!((q2 / q1 - expected_ratio).abs() < 1e-3, "ratio {}", q2 / q1);

        // Doubling the slot length doubles the flow.
        let long = BeverlooSlot::new(0.6, 1.4, 0.2, 1.0).unwrap();
        let ql = long.mass_flow_rate(rho, g, d1).unwrap();
        assert!((ql / q1 - 2.0).abs() < 1e-3);
    }

    #[test]
    fn beverloo_volumetric_is_mass_over_density() {
        let slot = BeverlooSlot::new(0.6, 1.4, 0.2, 0.5).unwrap();
        let rho = 1200.0;
        let m = slot.mass_flow_rate(rho, 9.81, 0.02).unwrap();
        let v = slot.volumetric_flow_rate(rho, 9.81, 0.02).unwrap();
        assert!((v - m / rho).abs() < 1e-6);
    }

    #[test]
    fn census_rejects_invalid() {
        let p = vec![Vec3::new(0.0, 0.0, 1.0)];
        assert!(DischargeCensus::classify(&p, &[], 0.0).is_none());
        assert!(DischargeCensus::classify(&p, &[0.0], 0.0).is_none());
        assert!(DischargeCensus::classify(&p, &[0.2], f32::NAN).is_none());
    }

    #[test]
    fn census_all_retained_and_all_discharged() {
        let r = vec![0.3_f32, 0.3];
        // Both well above the plane.
        let above = vec![Vec3::new(0.0, 0.0, 2.0), Vec3::new(1.0, 0.0, 3.0)];
        let c = DischargeCensus::classify(&above, &r, 0.0).unwrap();
        assert_eq!(c.retained_count(), 2);
        assert_eq!(c.discharged_count(), 0);
        assert!(c.discharged_volume() < 1e-6);
        assert!((c.retained_volume() - 2.0 * sphere_volume(0.3)).abs() < 1e-5);

        // Both well below the plane.
        let below = vec![Vec3::new(0.0, 0.0, -2.0), Vec3::new(1.0, 0.0, -3.0)];
        let c2 = DischargeCensus::classify(&below, &r, 0.0).unwrap();
        assert_eq!(c2.discharged_count(), 2);
        assert!((c2.discharged_volume() - 2.0 * sphere_volume(0.3)).abs() < 1e-5);
        assert!(c2.retained_volume() < 1e-6);
    }

    #[test]
    fn census_splits_straddling_grain() {
        // Sphere centred exactly on the plane → half the volume discharged.
        let p = vec![Vec3::new(0.0, 0.0, 0.0)];
        let r = vec![0.5_f32];
        let c = DischargeCensus::classify(&p, &r, 0.0).unwrap();
        let half = sphere_volume(0.5) / 2.0;
        assert!((c.discharged_volume() - half).abs() < 1e-5);
        assert!((c.retained_volume() - half).abs() < 1e-5);
        // Centre is not strictly below the plane → counted as retained.
        assert_eq!(c.retained_count(), 1);
        assert_eq!(c.discharged_count(), 0);
    }

    #[test]
    fn empirical_flow_rate_between_states() {
        let r = vec![0.3_f32];
        // Before: grain above the plane.
        let before = DischargeCensus::classify(&[Vec3::new(0.0, 0.0, 1.0)], &r, 0.0).unwrap();
        // After: same grain fully below → whole sphere discharged.
        let after = DischargeCensus::classify(&[Vec3::new(0.0, 0.0, -1.0)], &r, 0.0).unwrap();
        let density = 1500.0_f32;
        let dt = 0.5_f32;
        let q = mass_flow_between(&before, &after, dt, density).unwrap();
        let expected = density * sphere_volume(0.3) / dt;
        assert!((q - expected).abs() < 1e-2, "q {q} expected {expected}");

        // Bad dt / density rejected.
        assert!(mass_flow_between(&before, &after, 0.0, density).is_none());
        assert!(mass_flow_between(&before, &after, dt, 0.0).is_none());
    }
}
