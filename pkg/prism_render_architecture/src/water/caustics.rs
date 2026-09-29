//! Caustic intensity projection: light-space Jacobian, ray-traced, and photon.
//!
//! When light refracts through a wavy water surface the refracted rays
//! converge and diverge, concentrating irradiance into the bright dancing
//! patterns called caustics. This module owns the *intensity* math for the
//! three caustic routes the engine supports, as pure, deterministic,
//! non-negative functions:
//!
//! - *Jacobian projection* (cheap, real-time): the caustic gain at a receiver
//!   is the inverse of the area distortion (the Jacobian determinant) of the
//!   refracted-ray mapping. Where the surface focuses light the patch shrinks,
//!   the Jacobian is small, and the gain is large; where it defocuses, the gain
//!   fades. This is projected to a decal or light map.
//! - *Ray traced* (near-field high quality): rays are cast from the light
//!   through the refracting surface to the receiver via the shared `ray_scene`
//!   service; here we only supply the per-hit intensity weighting.
//! - *Photon mapped* (offline / film reference): photons are emitted, refracted,
//!   splatted onto the receiver, and their local density estimates the
//!   irradiance.
//!
//! Only `sqrt` is used, there are no `f32` equality tests, and there is no
//! AI/ML. Every returned intensity is non-negative.

use super::{EPS, PI};

/// The caustic evaluation route chosen for a water body this frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CausticsMethod {
    /// Photon-mapped caustics: emit, refract, splat, density-estimate. Highest
    /// fidelity, reserved for the closest / reference-quality view.
    PhotonMapped,
    /// Ray-traced caustics through the shared `ray_scene` service: sharp
    /// near-field light spots.
    RayTraced,
    /// Light-space Jacobian projection: the cheap real-time default and
    /// far-field fallback.
    JacobianProjection,
}

impl CausticsMethod {
    /// A cost rank used to assert monotonic selection: cheaper routes rank
    /// lower. `JacobianProjection` is `0`, `RayTraced` is `1`, `PhotonMapped`
    /// is `2`.
    #[must_use]
    pub const fn cost_rank(self) -> u32 {
        match self {
            CausticsMethod::JacobianProjection => 0,
            CausticsMethod::RayTraced => 1,
            CausticsMethod::PhotonMapped => 2,
        }
    }
}

/// Distance thresholds selecting a caustics route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CausticsThresholds {
    /// At or below this camera distance photon-mapped caustics are used.
    pub photon_max_distance: f32,
    /// Above `photon_max_distance` and at or below this, ray-traced caustics
    /// are used; farther receivers fall back to Jacobian projection.
    pub ray_max_distance: f32,
}

/// Chooses a caustics route for a receiver at `camera_distance`.
///
/// A higher `quality_bias` in `0..=1` expands the expensive bands out to
/// farther distances (up to 50% at full bias). The choice is monotonic in
/// distance: a receding receiver never upgrades to a costlier route.
#[must_use]
pub fn select_caustics(
    camera_distance: f32,
    quality_bias: f32,
    thresholds: CausticsThresholds,
) -> CausticsMethod {
    let distance = camera_distance.max(0.0);
    let bias = quality_bias.clamp(0.0, 1.0);
    let expand = 1.0 + 0.5 * bias;
    let photon_max = thresholds.photon_max_distance.max(0.0) * expand;
    let ray_max = thresholds.ray_max_distance.max(0.0) * expand;
    if distance <= photon_max {
        CausticsMethod::PhotonMapped
    } else if distance <= ray_max.max(photon_max) {
        CausticsMethod::RayTraced
    } else {
        CausticsMethod::JacobianProjection
    }
}

/// Caustic gain from the refracted-ray Jacobian determinant.
///
/// The gain is `1 / |jacobian|`, the reciprocal area distortion, clamped to
/// `max_gain` so a near-singular focus cannot produce an unbounded spike. A
/// converging patch (`|jacobian| < 1`) brightens, a diverging patch dims, and a
/// degenerate near-zero Jacobian saturates at `max_gain` instead of dividing by
/// zero. The result is always non-negative.
#[must_use]
pub fn jacobian_caustic_gain(jacobian: f32, max_gain: f32) -> f32 {
    let cap = max_gain.max(0.0);
    let mag = jacobian.abs();
    if mag <= EPS {
        return cap;
    }
    (1.0 / mag).min(cap)
}

/// Caustic irradiance at a receiver via Jacobian projection.
///
/// Multiplies the incident irradiance by [`jacobian_caustic_gain`], so it
/// brightens under focus and dims under defocus while staying non-negative and
/// bounded by `incident * max_gain`.
#[must_use]
pub fn project_caustic_intensity(incident: f32, jacobian: f32, max_gain: f32) -> f32 {
    incident.max(0.0) * jacobian_caustic_gain(jacobian, max_gain)
}

/// Photon-map irradiance estimate from a local splat.
///
/// Estimates irradiance as the total photon power falling inside a splat disk
/// divided by the disk area `PI * radius^2`: `photon_count * photon_power /
/// (PI * radius^2)`. More photons or more power per photon brighten the
/// estimate; a wider gather radius spreads the same energy over more area and
/// dims it. A degenerate radius returns `0` rather than dividing by zero. The
/// estimate is non-negative.
#[must_use]
pub fn photon_splat_density(photon_count: u32, photon_power: f32, radius: f32) -> f32 {
    let r = radius.max(0.0);
    if r <= EPS {
        return 0.0;
    }
    let area = PI * r * r;
    (photon_count as f32) * photon_power.max(0.0) / area
}

#[cfg(test)]
mod tests {
    use super::*;

    const THRESHOLDS: CausticsThresholds = CausticsThresholds {
        photon_max_distance: 5.0,
        ray_max_distance: 30.0,
    };

    #[test]
    fn selection_classifies_and_is_monotonic() {
        assert_eq!(
            select_caustics(1.0, 0.0, THRESHOLDS),
            CausticsMethod::PhotonMapped
        );
        assert_eq!(
            select_caustics(20.0, 0.0, THRESHOLDS),
            CausticsMethod::RayTraced
        );
        assert_eq!(
            select_caustics(100.0, 0.0, THRESHOLDS),
            CausticsMethod::JacobianProjection
        );
        let mut prev = select_caustics(0.0, 0.4, THRESHOLDS).cost_rank();
        let mut d = 0.0;
        while d <= 120.0 {
            let rank = select_caustics(d, 0.4, THRESHOLDS).cost_rank();
            assert!(rank <= prev, "route must not get pricier as it recedes");
            prev = rank;
            d += 1.0;
        }
    }

    #[test]
    fn quality_bias_extends_expensive_bands() {
        // Just past the base photon band: ray at zero bias, photon once biased.
        assert_eq!(
            select_caustics(6.0, 0.0, THRESHOLDS),
            CausticsMethod::RayTraced
        );
        assert_eq!(
            select_caustics(6.0, 1.0, THRESHOLDS),
            CausticsMethod::PhotonMapped
        );
    }

    #[test]
    fn jacobian_gain_focuses_brightens_and_clamps() {
        // Converging patch is brighter than a diverging one.
        let focus = jacobian_caustic_gain(0.25, 100.0);
        let defocus = jacobian_caustic_gain(4.0, 100.0);
        assert!(focus > defocus);
        assert!(focus >= 0.0 && defocus >= 0.0);
        // Near-singular Jacobian saturates at the cap rather than exploding.
        assert!((jacobian_caustic_gain(0.0, 50.0) - 50.0).abs() < EPS);
        // Sign of the Jacobian does not matter, only area magnitude.
        assert!(
            (jacobian_caustic_gain(-0.5, 100.0) - jacobian_caustic_gain(0.5, 100.0)).abs() < EPS
        );
    }

    #[test]
    fn projected_intensity_is_non_negative_and_bounded() {
        let i = project_caustic_intensity(2.0, 0.5, 10.0);
        assert!(i >= 0.0);
        assert!(i <= 2.0 * 10.0 + EPS);
        // Zero incident light yields zero caustics.
        assert!(project_caustic_intensity(0.0, 0.1, 10.0).abs() < EPS);
    }

    #[test]
    fn photon_density_grows_with_photons_and_falls_with_radius() {
        let dense = photon_splat_density(1000, 1.0, 0.5);
        let sparse = photon_splat_density(1000, 1.0, 2.0);
        assert!(dense > sparse, "wider gather dims the estimate");
        assert!(photon_splat_density(0, 1.0, 0.5).abs() < EPS);
        // Degenerate radius is inert, not a division by zero.
        assert!(photon_splat_density(1000, 1.0, 0.0).abs() < EPS);
        assert!(dense >= 0.0);
    }
}
