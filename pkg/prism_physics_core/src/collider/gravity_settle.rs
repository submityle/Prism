//! A deterministic gravity-settling preprocessor for loose sphere packings.
//!
//! A freshly generated packing (for example from
//! [`pack_spheres_grid`](crate::collider::grid_sphere_packing::pack_spheres_grid)
//! or
//! [`pack_spheres_from_distribution`](crate::collider::distribution_packing::pack_spheres_from_distribution))
//! is geometrically valid but *mechanically loose*: grains float at their
//! insertion sites with no guarantee that each one rests on the pile below it.
//! Running a discrete-element simulation from that state wastes the first
//! thousands of steps letting the cloud collapse under gravity before any
//! meaningful dynamics occur.
//!
//! [`GravitySettler`] does that collapse once, offline, and hands back a
//! settled [`SpherePacking`] whose grains sit at rest against one another and
//! against the container. It composes the existing contact pipeline rather than
//! reimplementing it:
//!
//! * [`SphereNarrowPhase`] rebuilds the grain–grain contact set each step,
//! * [`SphereCundallStrackDriver`] evaluates the normal plus tangential-history
//!   force on every grain contact (carrying static-friction springs so a grain
//!   can lock onto the pile),
//! * [`SphereBoundaryDriver`] adds the Hertzian wall/floor reaction, and
//! * a semi-implicit (symplectic) Euler update advances the cloud under gravity
//!   with a per-step velocity-retention factor that bleeds off kinetic energy
//!   so the pile converges to rest.
//!
//! The loop stops as soon as the fastest grain drops below a rest speed, or
//! after a bounded iteration count, and the final positions are repackaged with
//! [`SpherePacking::from_parts`]. The whole routine is pure and deterministic:
//! identical inputs always yield identical settled packings.

use glam::Vec3;

use crate::collider::sphere_boundary_driver::SphereBoundaryDriver;
use crate::collider::sphere_cundall_strack_driver::SphereCundallStrackDriver;
use crate::collider::sphere_dem_integrator::SphereDemState;
use crate::collider::sphere_narrow_phase::SphereNarrowPhase;
use crate::collider::sphere_packing::SpherePacking;

/// Tunable parameters for a [`GravitySettler::settle`] run.
///
/// Construct with [`GravitySettleParams::new`], which validates every field, or
/// with [`GravitySettleParams::earth`] for the common downward-gravity default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GravitySettleParams {
    /// Uniform body acceleration applied to every grain (force `mᵢ · gravity`).
    pub gravity: Vec3,
    /// Mass density used to assign per-grain mass from its volume.
    pub density: f32,
    /// Fixed integration time step.
    pub dt: f32,
    /// Hard cap on the number of integration steps.
    pub max_iterations: usize,
    /// Convergence threshold: the run stops once the fastest grain speed is at
    /// or below this value.
    pub rest_speed: f32,
    /// Equilibrium threshold on net acceleration magnitude: convergence also
    /// requires the largest per-grain net acceleration to be at or below this
    /// value, so a grain in free flight (acceleration `≈ |gravity|`) is never
    /// mistaken for one resting on the pile.
    pub rest_acceleration: f32,
    /// Fraction of velocity kept after each step, in `(0, 1]`. Values below one
    /// dissipate kinetic energy so the pile settles; `1.0` is undamped.
    pub velocity_retention: f32,
    /// Non-negative detection margin forwarded to the narrow phase.
    pub contact_margin: f32,
}

impl GravitySettleParams {
    /// Builds a validated parameter set.
    ///
    /// Returns `None` unless `gravity` is finite, `density` is finite and
    /// strictly positive, `dt` is finite and strictly positive,
    /// `max_iterations` is non-zero, `rest_speed` and `rest_acceleration` are
    /// finite and non-negative, `velocity_retention` is finite and lies in
    /// `(0, 1]`, and `contact_margin` is finite and non-negative.
    #[must_use]
    pub fn new(
        gravity: Vec3,
        density: f32,
        dt: f32,
        max_iterations: usize,
        rest_speed: f32,
        rest_acceleration: f32,
        velocity_retention: f32,
        contact_margin: f32,
    ) -> Option<Self> {
        if !gravity.is_finite() {
            return None;
        }
        if !(density.is_finite() && density > 0.0) {
            return None;
        }
        if !(dt.is_finite() && dt > 0.0) {
            return None;
        }
        if max_iterations == 0 {
            return None;
        }
        if !(rest_speed.is_finite() && rest_speed >= 0.0) {
            return None;
        }
        if !(rest_acceleration.is_finite() && rest_acceleration >= 0.0) {
            return None;
        }
        if !(velocity_retention.is_finite()
            && velocity_retention > 0.0
            && velocity_retention <= 1.0)
        {
            return None;
        }
        if !(contact_margin.is_finite() && contact_margin >= 0.0) {
            return None;
        }
        Some(Self {
            gravity,
            density,
            dt,
            max_iterations,
            rest_speed,
            rest_acceleration,
            velocity_retention,
            contact_margin,
        })
    }

    /// Convenience constructor for the usual case of gravity pointing down the
    /// `-Z` axis at `9.81 m/s²`. The remaining fields are validated exactly as
    /// in [`GravitySettleParams::new`].
    #[must_use]
    pub fn earth(
        density: f32,
        dt: f32,
        max_iterations: usize,
        rest_speed: f32,
        rest_acceleration: f32,
        velocity_retention: f32,
    ) -> Option<Self> {
        Self::new(
            Vec3::new(0.0, 0.0, -9.81),
            density,
            dt,
            max_iterations,
            rest_speed,
            rest_acceleration,
            velocity_retention,
            0.0,
        )
    }
}

/// Diagnostics returned alongside a settled packing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GravitySettleReport {
    iterations: usize,
    converged: bool,
    final_max_speed: f32,
    final_kinetic_energy: f32,
    grain_contacts: u32,
    boundary_contacts: u32,
}

impl GravitySettleReport {
    /// Number of integration steps actually executed.
    #[must_use]
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// `true` if the fastest grain reached the rest speed before the iteration
    /// cap, `false` if the cap was hit first.
    #[must_use]
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// Fastest grain speed on the final step.
    #[must_use]
    pub fn final_max_speed(&self) -> f32 {
        self.final_max_speed
    }

    /// Total kinetic energy of the cloud on the final step.
    #[must_use]
    pub fn final_kinetic_energy(&self) -> f32 {
        self.final_kinetic_energy
    }

    /// Grain–grain contacts resolved on the final step.
    #[must_use]
    pub fn grain_contacts(&self) -> u32 {
        self.grain_contacts
    }

    /// Grain–boundary contacts resolved on the final step.
    #[must_use]
    pub fn boundary_contacts(&self) -> u32 {
        self.boundary_contacts
    }
}

/// Settles a loose packing to rest by composing the grain-contact and boundary
/// drivers under gravity.
///
/// The settler borrows the grain-contact model and a fully configured boundary
/// driver (its registered half-spaces define the container). It owns no mutable
/// per-run state, so a single instance can settle many packings.
pub struct GravitySettler {
    grain_driver: SphereCundallStrackDriver,
    boundary_driver: SphereBoundaryDriver,
}

impl GravitySettler {
    /// Builds a settler from a grain-contact driver and a boundary driver whose
    /// half-spaces describe the container.
    #[must_use]
    pub fn new(
        grain_driver: SphereCundallStrackDriver,
        boundary_driver: SphereBoundaryDriver,
    ) -> Self {
        Self {
            grain_driver,
            boundary_driver,
        }
    }

    /// Number of container half-spaces the boundary driver will enforce.
    #[must_use]
    pub fn boundary_count(&self) -> usize {
        self.boundary_driver.boundary_count()
    }

    /// Settles `packing` under `params`, returning the rested packing and a
    /// diagnostics report.
    ///
    /// An empty packing is returned unchanged with a trivially converged report.
    /// Returns `None` if the packing cannot be turned into a valid dynamic state
    /// (for example a non-positive density/volume) or if the contact pipeline
    /// rejects the cloud at any step.
    #[must_use]
    pub fn settle(
        &self,
        packing: &SpherePacking,
        params: &GravitySettleParams,
    ) -> Option<(SpherePacking, GravitySettleReport)> {
        if packing.is_empty() {
            let settled = SpherePacking::from_parts(Vec::new(), Vec::new())?;
            let report = GravitySettleReport {
                iterations: 0,
                converged: true,
                final_max_speed: 0.0,
                final_kinetic_energy: 0.0,
                grain_contacts: 0,
                boundary_contacts: 0,
            };
            return Some((settled, report));
        }

        let positions = packing.positions().to_vec();
        let radii = packing.radii().to_vec();
        let velocities = vec![Vec3::ZERO; positions.len()];
        let mut state = SphereDemState::from_density(positions, velocities, radii, params.density)?;

        // Per-run mutable scratch; a fresh narrow phase and friction driver keep
        // the settler itself immutable and reusable.
        let mut narrow_phase = SphereNarrowPhase::new();
        let mut friction = self.grain_driver.clone();
        friction.clear();

        let mut report = GravitySettleReport {
            iterations: 0,
            converged: false,
            final_max_speed: 0.0,
            final_kinetic_energy: state.kinetic_energy(),
            grain_contacts: 0,
            boundary_contacts: 0,
        };

        for step in 1..=params.max_iterations {
            let grain_resolution = {
                let contacts =
                    narrow_phase.detect(state.positions(), state.radii(), params.contact_margin)?;
                friction.resolve(contacts, state.radii(), state.velocities(), params.dt)?
            };
            let boundary_resolution = self.boundary_driver.resolve(
                state.positions(),
                state.radii(),
                state.velocities(),
            )?;

            let grain_forces = grain_resolution.forces();
            let boundary_forces = boundary_resolution.forces();
            let masses = state.masses().to_vec();
            let mut velocities = state.velocities().to_vec();
            let mut positions = state.positions().to_vec();
            let mut max_speed = 0.0_f32;
            let mut max_accel = 0.0_f32;

            for (i, mass) in masses.iter().enumerate() {
                let total = grain_forces[i] + boundary_forces[i] + *mass * params.gravity;
                let acceleration = total / *mass;
                max_accel = max_accel.max(acceleration.length());
                let mut velocity = velocities[i] + acceleration * params.dt;
                velocity *= params.velocity_retention;
                let position = positions[i] + velocity * params.dt;
                velocities[i] = velocity;
                positions[i] = position;
                max_speed = max_speed.max(velocity.length());
            }

            let radii = state.radii().to_vec();
            let advanced = SphereDemState::new(positions, velocities, radii, masses)?;
            let kinetic_energy = advanced.kinetic_energy();
            state = advanced;

            report = GravitySettleReport {
                iterations: step,
                converged: max_speed <= params.rest_speed && max_accel <= params.rest_acceleration,
                final_max_speed: max_speed,
                final_kinetic_energy: kinetic_energy,
                grain_contacts: grain_resolution.contact_count(),
                boundary_contacts: boundary_resolution.contact_count(),
            };

            if report.converged {
                break;
            }
        }

        let settled =
            SpherePacking::from_parts(state.positions().to_vec(), state.radii().to_vec())?;
        Some((settled, report))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::hertz_contact::HertzModel;
    use crate::collider::rotational_boundary_contact::HalfSpace;
    use crate::collider::tangential_history_contact::CundallStrackModel;

    fn grain_model() -> CundallStrackModel {
        // kₙ, γₙ, k_t, γ_t, μ. Moderate stiffness keeps the explicit step stable
        // for the small grains used in these tests.
        CundallStrackModel::new(2.0e3, 2.0, 2.0e3, 2.0, 0.5).unwrap()
    }

    fn boundary_model() -> HertzModel {
        // E, poisson, γₙ, γₜ, μ.
        HertzModel::new(1.0e6, 0.3, 5.0, 5.0, 0.5).unwrap()
    }

    fn floor() -> HalfSpace {
        HalfSpace::new(Vec3::ZERO, Vec3::Z).unwrap()
    }

    fn settler_with_floor() -> GravitySettler {
        let boundary = SphereBoundaryDriver::with_boundaries(boundary_model(), vec![floor()]);
        GravitySettler::new(SphereCundallStrackDriver::new(grain_model()), boundary)
    }

    fn settle_params() -> GravitySettleParams {
        GravitySettleParams::earth(1.0e3, 1.0e-4, 60_000, 5.0e-3, 1.0, 0.999).unwrap()
    }

    #[test]
    fn rejects_invalid_params() {
        assert!(GravitySettleParams::earth(0.0, 1.0e-4, 100, 1.0e-3, 0.5, 0.9).is_none());
        assert!(GravitySettleParams::earth(1.0e3, 0.0, 100, 1.0e-3, 0.5, 0.9).is_none());
        assert!(GravitySettleParams::earth(1.0e3, 1.0e-4, 0, 1.0e-3, 0.5, 0.9).is_none());
        assert!(GravitySettleParams::earth(1.0e3, 1.0e-4, 100, -1.0, 0.5, 0.9).is_none());
        assert!(GravitySettleParams::earth(1.0e3, 1.0e-4, 100, 1.0e-3, -1.0, 0.9).is_none());
        assert!(GravitySettleParams::earth(1.0e3, 1.0e-4, 100, 1.0e-3, 0.5, 0.0).is_none());
        assert!(GravitySettleParams::earth(1.0e3, 1.0e-4, 100, 1.0e-3, 0.5, 1.5).is_none());
        assert!(GravitySettleParams::new(
            Vec3::new(f32::NAN, 0.0, 0.0),
            1.0e3,
            1.0e-4,
            100,
            1.0e-3,
            0.5,
            0.9,
            0.0,
        )
        .is_none());
        assert!(GravitySettleParams::new(
            Vec3::new(0.0, 0.0, -9.81),
            1.0e3,
            1.0e-4,
            100,
            1.0e-3,
            0.5,
            0.9,
            -1.0,
        )
        .is_none());
    }

    #[test]
    fn empty_packing_returns_converged_zero_iterations() {
        let settler = settler_with_floor();
        let packing = SpherePacking::from_parts(Vec::new(), Vec::new()).unwrap();
        let (settled, report) = settler.settle(&packing, &settle_params()).unwrap();
        assert!(settled.is_empty());
        assert_eq!(report.iterations(), 0);
        assert!(report.converged());
        assert_eq!(report.final_max_speed(), 0.0);
        assert_eq!(report.grain_contacts(), 0);
        assert_eq!(report.boundary_contacts(), 0);
    }

    #[test]
    fn single_grain_settles_onto_floor() {
        let settler = settler_with_floor();
        let radius = 0.05_f32;
        // Released well above the floor with no initial velocity.
        let packing =
            SpherePacking::from_parts(vec![Vec3::new(0.0, 0.0, 0.5)], vec![radius]).unwrap();
        let (settled, report) = settler.settle(&packing, &settle_params()).unwrap();

        assert!(report.converged(), "run should reach rest");
        let z = settled.positions()[0].z;
        // The grain rests with its centre about one radius above the floor.
        assert!(
            (z - radius).abs() < 0.1 * radius,
            "settled z = {z}, expected ~{radius}"
        );
        assert!(report.boundary_contacts() >= 1);
    }

    #[test]
    fn grain_does_not_sink_below_floor() {
        let settler = settler_with_floor();
        let radius = 0.05_f32;
        let packing =
            SpherePacking::from_parts(vec![Vec3::new(0.0, 0.0, 0.3)], vec![radius]).unwrap();
        let (settled, _) = settler.settle(&packing, &settle_params()).unwrap();
        // A tiny Hertzian overlap is physical; a deep sink is not.
        assert!(settled.positions()[0].z > radius - 0.05 * radius);
    }

    #[test]
    fn loose_stack_collapses_and_lowers_centre_of_mass() {
        let settler = settler_with_floor();
        let radius = 0.05_f32;
        // Three grains stacked with wide gaps between them.
        let initial = vec![
            Vec3::new(0.0, 0.0, 0.15),
            Vec3::new(0.0, 0.0, 0.35),
            Vec3::new(0.0, 0.0, 0.55),
        ];
        let mean_z_before = initial.iter().map(|p| p.z).sum::<f32>() / initial.len() as f32;
        let packing = SpherePacking::from_parts(initial, vec![radius; 3]).unwrap();
        let (settled, report) = settler.settle(&packing, &settle_params()).unwrap();

        let mean_z_after =
            settled.positions().iter().map(|p| p.z).sum::<f32>() / settled.len() as f32;
        assert!(
            mean_z_after < mean_z_before,
            "settling should compact the stack: before {mean_z_before}, after {mean_z_after}"
        );
        assert!(report.final_kinetic_energy() <= report_initial_energy_bound());
        // Every grain stays above the floor.
        for p in settled.positions() {
            assert!(p.z > radius - 0.05 * radius);
        }
    }

    fn report_initial_energy_bound() -> f32 {
        // A generous upper bound: a settled pile holds far less kinetic energy
        // than a single grain moving at the rest speed times the grain count.
        1.0e-3
    }

    #[test]
    fn settling_is_deterministic() {
        let settler = settler_with_floor();
        let radius = 0.05_f32;
        let make = || {
            SpherePacking::from_parts(
                vec![Vec3::new(0.0, 0.0, 0.2), Vec3::new(0.01, 0.0, 0.4)],
                vec![radius; 2],
            )
            .unwrap()
        };
        let params = settle_params();
        let (first, first_report) = settler.settle(&make(), &params).unwrap();
        let (second, second_report) = settler.settle(&make(), &params).unwrap();
        assert_eq!(first.positions(), second.positions());
        assert_eq!(first.radii(), second.radii());
        assert_eq!(first_report, second_report);
    }

    #[test]
    fn boundary_count_reports_container_faces() {
        let boundary = SphereBoundaryDriver::with_boundaries(
            boundary_model(),
            vec![
                floor(),
                HalfSpace::new(Vec3::new(0.5, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)).unwrap(),
            ],
        );
        let settler = GravitySettler::new(SphereCundallStrackDriver::new(grain_model()), boundary);
        assert_eq!(settler.boundary_count(), 2);
    }
}
