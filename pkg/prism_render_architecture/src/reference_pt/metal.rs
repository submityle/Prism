//! Measured complex-index presets for common conductors.
//!
//! [`crate::reference_pt::conductor::Conductor`] is parameterized by a metal's
//! complex index of refraction `eta + i*k` sampled per colour channel. Looking
//! those numbers up by hand is error-prone, so this module curates the
//! canonical measured values for the metals an artist actually reaches for and
//! exposes them through a small [`Metal`] enum. The triplets are the spectral
//! optical constants sampled at representative red/green/blue wavelengths
//! (roughly 630/532/465 nm), matching the figures published on
//! `refractiveindex.info` and reused by physically based renderers; they are
//! the ground-truth inputs the real-time metal library is validated against.
//!
//! Only data lives here — no arithmetic — so the module introduces no
//! transcendental calls and stays trivially correct.

use super::conductor::Conductor;
use super::Vec3;

/// A named conductor with a curated complex index of refraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metal {
    /// Gold: strongly warm, red/green reflectance far above blue.
    Gold,
    /// Silver: near-neutral and the most reflective common metal.
    Silver,
    /// Copper: warm reddish-orange.
    Copper,
    /// Aluminium: bright and very slightly cool.
    Aluminium,
    /// Iron: dark and desaturated.
    Iron,
    /// Chromium: hard, slightly cool, high reflectance.
    Chromium,
}

impl Metal {
    /// The per-channel real index `eta` and extinction coefficient `k` of this
    /// metal, as the pair `(eta, k)`.
    ///
    /// The values are the measured spectral optical constants sampled at the
    /// red/green/blue primaries; feeding them to
    /// [`crate::reference_pt::conductor::fresnel_conductor`] reproduces the
    /// metal's base colour and its grazing-angle hue drift.
    #[must_use]
    pub fn complex_ior(self) -> (Vec3, Vec3) {
        match self {
            Self::Gold => (
                Vec3::new(0.143, 0.375, 1.442),
                Vec3::new(3.983, 2.386, 1.603),
            ),
            Self::Silver => (
                Vec3::new(0.155, 0.116, 0.138),
                Vec3::new(4.818, 3.122, 2.146),
            ),
            Self::Copper => (
                Vec3::new(0.200, 0.924, 1.102),
                Vec3::new(3.912, 2.448, 2.137),
            ),
            Self::Aluminium => (
                Vec3::new(1.345, 0.965, 0.617),
                Vec3::new(7.474, 6.399, 5.303),
            ),
            Self::Iron => (
                Vec3::new(2.911, 2.950, 2.580),
                Vec3::new(3.089, 2.931, 2.767),
            ),
            Self::Chromium => (
                Vec3::new(3.181, 3.079, 2.392),
                Vec3::new(3.329, 3.340, 3.148),
            ),
        }
    }
}

/// Builds a rough [`Conductor`] for a named `metal` at the given perceptual
/// `roughness` in `[0, 1]`.
#[must_use]
pub fn conductor_for(metal: Metal, roughness: f32) -> Conductor {
    let (eta, k) = metal.complex_ior();
    Conductor::new(eta, k, roughness)
}

#[cfg(test)]
mod tests {
    use super::super::conductor::fresnel_conductor;
    use super::*;

    /// Every preset's normal-incidence `Fresnel` reflectance is a valid
    /// in-gamut colour on all three channels.
    #[test]
    fn presets_reflectance_in_gamut() {
        for metal in [
            Metal::Gold,
            Metal::Silver,
            Metal::Copper,
            Metal::Aluminium,
            Metal::Iron,
            Metal::Chromium,
        ] {
            let (eta, k) = metal.complex_ior();
            let f0 = fresnel_conductor(eta, k, 1.0);
            for c in [f0.x, f0.y, f0.z] {
                assert!((0.0..=1.0).contains(&c), "{metal:?} F0 {c} out of gamut");
            }
        }
    }

    /// Gold and copper are warm (red reflectance clearly above blue), while
    /// silver and aluminium stay close to neutral.
    #[test]
    fn warm_and_neutral_metals_are_classified() {
        let (eta, k) = Metal::Gold.complex_ior();
        let gold = fresnel_conductor(eta, k, 1.0);
        assert!(gold.x > gold.z + 0.2, "gold not warm: {gold:?}");

        let (eta, k) = Metal::Copper.complex_ior();
        let copper = fresnel_conductor(eta, k, 1.0);
        assert!(copper.x > copper.z + 0.2, "copper not warm: {copper:?}");

        let (eta, k) = Metal::Silver.complex_ior();
        let silver = fresnel_conductor(eta, k, 1.0);
        assert!(
            (silver.x - silver.z).abs() < 0.2,
            "silver not neutral: {silver:?}"
        );
    }

    /// Silver is the most reflective common metal at normal incidence.
    #[test]
    fn silver_is_the_brightest_preset() {
        let (eta, k) = Metal::Silver.complex_ior();
        let silver = fresnel_conductor(eta, k, 1.0).max_component();
        let (eta, k) = Metal::Iron.complex_ior();
        let iron = fresnel_conductor(eta, k, 1.0).max_component();
        assert!(silver > iron, "silver {silver} should beat iron {iron}");
    }

    /// The convenience constructor threads the roughness through to a working
    /// [`Conductor`] that produces in-hemisphere samples.
    #[test]
    fn conductor_for_builds_a_usable_lobe() {
        use super::super::sampler::Rng;
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let wo = Vec3::new(0.2, 0.95, 0.1).normalize_or_zero();
        let conductor = conductor_for(Metal::Gold, 0.25);
        let mut rng = Rng::seed(11);
        let mut hits = 0u32;
        for _ in 0..256 {
            if let Some(s) = conductor.sample(wo, normal, &mut rng) {
                assert!(normal.dot(s.direction) > 0.0);
                hits += 1;
            }
        }
        assert!(hits > 0, "no valid samples produced");
    }
}
