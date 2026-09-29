//! Aggregate per-body simulation/shading profile.
//!
//! [`WaterSimProfile`] gathers the five per-frame planner profiles — the
//! solver-stepping schedule ([`SimProfile`]), the breaking/foam surface
//! effects ([`SurfaceFxProfile`]), the shoreline waterline/wetness transition
//! ([`ShorelineProfile`]), the spectral optics ([`OpticsProfile`]), and the
//! two-way rigid-body coupling ([`CouplingProfile`]) — into one value-typed
//! record carried by every [`super::WaterBody`]. Splitting the tuning this way
//! keeps each planner's knobs owned by the module that consumes them while the
//! routing contract stays a single `Copy` struct.
//!
//! [`WaterSimProfile::physical_water`] supplies a coherent default tuned for
//! clear open water; its numbers match the per-module test fixtures so the
//! aggregate stays self-consistent with the planners it feeds.

use super::breaking::BreakingCriteria;
use super::coupling_frame::CouplingProfile;
use super::foam::FoamConfig;
use super::optics::OpticsProfile;
use super::pbf::PbfParams;
use super::shading::ShadingProfile;
use super::shoreline::ShorelineProfile;
use super::simulation::SimProfile;
use super::spectrum::{SpectrumKind, SpectrumParams};
use super::surface_fx::SurfaceFxProfile;
use super::underwater::RgbExtinction;
use super::waterline::WaterlineParams;
use super::wetness::WetnessParams;
use super::Vec2;

/// The complete per-body tuning fed to the per-frame planners.
///
/// Every field is an independently owned planner profile; the pipeline reads
/// exactly the slice each stage needs. `Copy` so it can live inline inside a
/// [`super::WaterBody`] without heap traffic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterSimProfile {
    /// Solver-stepping schedule (`CFL`, sub-steps, spectrum, `PBF`).
    pub sim: SimProfile,
    /// Breaking-wave classification, crest spray, and foam coverage tuning.
    pub surface_fx: SurfaceFxProfile,
    /// Waterline transition and surface wetness/puddle behavior.
    pub shoreline: ShorelineProfile,
    /// Spectral refraction, `Beer-Lambert` extinction, and scattering optics.
    pub optics: OpticsProfile,
    /// Two-way rigid-body coupling (buoyancy, drag, added mass, sub-steps).
    pub coupling: CouplingProfile,
    /// Four-frontend lighting-response fork tuning (`PBR`/`NPR`/hybrid/custom).
    pub shading: ShadingProfile,
}

impl WaterSimProfile {
    /// A coherent default profile for clear, wind-driven open water.
    ///
    /// The constants mirror the per-module test fixtures so the aggregate
    /// remains consistent with the planners that consume each slice. Callers
    /// override individual slices for stylized water, murky lakes, or calm
    /// pools.
    #[must_use]
    pub fn physical_water() -> Self {
        Self {
            sim: SimProfile {
                spectrum: SpectrumParams {
                    kind: SpectrumKind::Phillips,
                    wind: Vec2::new(12.0, 0.0),
                    amplitude: 4e-4,
                    peak_enhancement: 1.0,
                    min_wavelength: 0.2,
                    directional_exponent: 2,
                },
                pbf: PbfParams {
                    rest_density: 1000.0,
                    particle_mass: 1.0,
                    smoothing_radius: 0.1,
                    relaxation_epsilon: 1e-3,
                    artificial_pressure_k: 0.1,
                    artificial_pressure_n: 4,
                    artificial_pressure_delta_q: 0.2,
                    solver_iterations: 4,
                },
                cfl_number: 0.5,
                fold_threshold: 0.0,
                max_substeps: 8,
            },
            surface_fx: SurfaceFxProfile {
                criteria: BreakingCriteria {
                    steepness_threshold: 0.6,
                    jacobian_fold_threshold: 0.0,
                    curvature_threshold: 1.0,
                    breaking_intensity: 0.5,
                },
                foam: FoamConfig {
                    nx: 64,
                    nz: 64,
                    dx: 0.5,
                    base_decay: 0.2,
                    persistence_floor: 0.02,
                    reference_speed: 3.0,
                },
                max_foam_rate: 5.0,
                jet_speed: 4.0,
                max_spray_count: 256,
            },
            shoreline: ShorelineProfile {
                waterline: WaterlineParams {
                    transition_half_width: 0.25,
                    shoreline_depth: 1.0,
                },
                wetness: WetnessParams {
                    max_capillary_height: 0.5,
                    absorb_rate: 2.0,
                    dry_rate: 0.5,
                    darkening_strength: 0.4,
                    puddle_threshold: 0.02,
                },
                drain_rate: 0.1,
            },
            optics: OpticsProfile {
                cauchy_a: 1.324,
                cauchy_b: 0.0032,
                refraction_strength: 0.05,
                extinction: RgbExtinction {
                    r: 0.45,
                    g: 0.15,
                    b: 0.08,
                },
                scatter_albedo: 0.7,
                asymmetry_g: 0.6,
                visibility_threshold: 0.02,
            },
            coupling: CouplingProfile {
                fluid_density: 1000.0,
                drag_coeff: 1.0,
                added_mass_coeff: 0.5,
                max_substeps: 16,
                max_readback: 32,
            },
            shading: ShadingProfile::physical_water(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_water_is_deterministic() {
        assert_eq!(
            WaterSimProfile::physical_water(),
            WaterSimProfile::physical_water()
        );
    }

    #[test]
    fn physical_water_slices_are_coherent() {
        let p = WaterSimProfile::physical_water();
        // Fresh water reference density is shared by the PBF rest density and
        // the coupling fluid density.
        assert!((p.sim.pbf.rest_density - p.coupling.fluid_density).abs() < 1.0);
        // Spectral IOR baseline sits in the physical water range.
        assert!(p.optics.cauchy_a > 1.3 && p.optics.cauchy_a < 1.34);
        // Sub-step caps are strictly positive so the scheduler always advances.
        assert!(p.sim.max_substeps >= 1);
        assert!(p.coupling.max_substeps >= 1);
        // Red water attenuates faster than blue (clear-water ordering).
        assert!(p.optics.extinction.r > p.optics.extinction.b);
    }

    #[test]
    fn overriding_one_slice_leaves_others_untouched() {
        let base = WaterSimProfile::physical_water();
        let mut murky = base;
        murky.optics.extinction = RgbExtinction {
            r: 1.2,
            g: 0.9,
            b: 0.7,
        };
        assert_ne!(murky, base);
        assert_eq!(murky.sim, base.sim);
        assert_eq!(murky.coupling, base.coupling);
    }
}
