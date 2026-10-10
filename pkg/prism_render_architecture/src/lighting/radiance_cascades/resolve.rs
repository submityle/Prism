//! Resolve a merged cascade-0 field into per-probe incident radiance.
//!
//! After the merge fold, cascade 0 holds — for every probe and every angular
//! bin — the full-range radiance arriving from that direction. Shading needs a
//! scalar-per-probe quantity, so this stage integrates over direction:
//!
//! - [`mean_radiance`] averages the bins, giving the mean incident radiance
//!   (for an isotropic field of radiance `L` it returns `L`).
//! - [`fluence`] multiplies the mean by `TAU`, the 2D angular measure, giving
//!   the incident fluence `∫ L(ω) dω` used as a diffuse-GI irradiance proxy.
//!
//! # Provenance
//! Directional integration of the resolved cascade is the standard final step
//! of Alexander Sannikov's *Radiance Cascades* (2023). Clean-room classical
//! quadrature; no neural or data-driven components.
//!
//! No Unreal Engine source is used anywhere in this module.

use alloc::vec::Vec;

use core::f32::consts::TAU;

use glam::Vec3;

use super::cascade::Cascade;

/// Mean incident radiance per cascade-0 probe, row-major (`row * cols + col`).
///
/// Averages the radiance of all angular bins at each probe. For an isotropic
/// field this equals the common radiance value.
#[must_use]
pub fn mean_radiance(cascade0: &Cascade) -> Vec<Vec3> {
    let (cols, rows) = cascade0.dims();
    let angular = cascade0.angular();
    let inv = 1.0 / (angular as f32);
    let mut out = Vec::with_capacity((cols * rows) as usize);
    for row in 0..rows {
        for col in 0..cols {
            let mut sum = Vec3::ZERO;
            for dir in 0..angular {
                sum += cascade0.get(col, row, dir).radiance;
            }
            out.push(sum * inv);
        }
    }
    out
}

/// Incident fluence per cascade-0 probe (`mean_radiance · TAU`), row-major.
#[must_use]
pub fn fluence(cascade0: &Cascade) -> Vec<Vec3> {
    mean_radiance(cascade0)
        .into_iter()
        .map(|r| r * TAU)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::cascade::SceneSampler;
    use super::super::hierarchy::CascadeHierarchy;
    use super::super::interval::RadianceInterval;
    use super::*;
    use glam::Vec2;

    struct Medium {
        emission: Vec3,
        absorption: f32,
    }
    impl SceneSampler for Medium {
        fn sample_interval(&self, _o: Vec2, _d: Vec2, t0: f32, t1: f32) -> RadianceInterval {
            // Rational emissive-absorbing medium with survival S(t)=1/(1+a t)^2.
            // Transmittance over [t0,t1) is S(t1)/S(t0) and premultiplied radiance
            // integrates e*S exactly to a closed rational form, so sub-intervals
            // composite exactly under `over` with no exp/trig (keeping the golden
            // within the workspace libm-determinism lint).
            let a = self.absorption;
            let s0 = 1.0 + a * t0.max(0.0);
            let s1 = 1.0 + a * t1.max(0.0);
            let ratio = s0 / s1;
            let tr = ratio * ratio;
            let rad = self.emission * ((s0 / a) * (1.0 - ratio));
            RadianceInterval::new(rad, tr)
        }
    }

    fn hierarchy() -> CascadeHierarchy {
        CascadeHierarchy {
            origin: Vec2::ZERO,
            base_spacing: 1.0,
            base_cols: 4,
            base_rows: 4,
            base_angular: 4,
            base_interval: 1.0,
            levels: 3,
        }
    }

    #[test]
    fn mean_radiance_recovers_isotropic_value() {
        let h = hierarchy();
        let m = Medium {
            emission: Vec3::new(0.8, 0.4, 0.2),
            absorption: 0.5,
        };
        let merged = super::super::solve(&h, &m);
        let a = m.absorption;
        let s0 = 1.0 + a * h.interval_start(0);
        let s1 = 1.0 + a * h.interval_end(h.levels - 1);
        let ratio = s0 / s1;
        let expected = m.emission * ((s0 / a) * (1.0 - ratio));
        for r in mean_radiance(&merged) {
            assert!(
                (r - expected).abs().max_element() <= 1.0e-3,
                "{r:?} vs {expected:?}"
            );
        }
    }

    #[test]
    fn fluence_is_mean_times_tau() {
        let h = hierarchy();
        let m = Medium {
            emission: Vec3::splat(1.0),
            absorption: 0.3,
        };
        let merged = super::super::solve(&h, &m);
        let mean = mean_radiance(&merged);
        let flux = fluence(&merged);
        for (m, f) in mean.iter().zip(flux.iter()) {
            assert!((*f - *m * TAU).abs().max_element() <= 1.0e-5);
        }
    }
}
