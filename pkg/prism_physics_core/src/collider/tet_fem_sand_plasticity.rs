//! Drucker–Prager finite-strain elastoplasticity for cohesionless granular
//! media (dry sand) on a tetrahedral `FEM` element.
//!
//! Sand is *frictional*: it resists shear only in proportion to the confining
//! pressure, cannot sustain tension, and takes a permanent set once it flows.
//! The classic animation model (Klár et al., clean-room reimplementation here)
//! captures this with a **Drucker–Prager cone** written in principal
//! logarithmic (Hencky) strain space on top of the crate's multiplicative split
//!
//! ```text
//! F = Fₑ · Fₚ
//! ```
//!
//! Each step strips the stored plastic deformation to form the elastic
//! predictor `Fₑ_trial = F · Fₚ⁻¹`, takes its signed SVD, and maps the singular
//! values to principal strains `ε = ln|Σ|`. Writing `tr ε` for the volumetric
//! part and `ε̂ = ε − (tr ε)/3` for the deviatoric part, the yield surface is
//!
//! ```text
//! y(ε) = ‖ε̂‖ + ((3λ + 2μ) / (2μ)) · (tr ε) · α   ≤ 0   (elastic)
//! ```
//!
//! with `α` the friction coefficient derived from the internal friction angle.
//! Three regimes result:
//!
//! * **Expansion** (`tr ε > 0`): a cohesionless grain assembly cannot pull on
//!   itself, so the predictor is projected to the cone *tip* (`ε → 0`, zero
//!   stress) and the whole strain becomes plastic.
//! * **Inside the cone** (`y ≤ 0`): fully elastic, the predictor is kept.
//! * **Outside the cone** (`y > 0`): radially return the deviatoric strain onto
//!   the cone surface, keeping the volumetric part.
//!
//! Because more confining pressure (more negative `tr ε`) enlarges the elastic
//! region, the model reproduces pressure-dependent friction rather than a
//! fixed shear yield. Optional linear hardening grows the effective friction
//! with accumulated plastic strain (denser packing resists shear more).
//!
//! Clean-room implementation over the crate's own [`svd3`]; logs/exponentials
//! run in `f64` for deterministic, libm-free precision. No Unreal Engine source
//! or derived code.

use crate::collider::tet_fem_constitutive::LameParameters;
use crate::mpm::svd3;
use glam::{Mat3, Vec3};

/// Principal stretches are clamped to this magnitude before taking a log, so an
/// inverted or collapsed predictor never produces a non-finite strain.
const MIN_STRETCH: f32 = 1e-4;

/// Deviatoric strains with Frobenius norm at or below this have no well-defined
/// return direction and are treated as purely volumetric.
const MIN_DEVIATORIC_NORM: f32 = 1e-9;

/// Which branch of the Drucker–Prager return map produced a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SandYield {
    /// Predictor was inside the cone: the response is fully elastic.
    Elastic,
    /// Deviatoric strain was projected radially onto the cone surface.
    ConeSurface,
    /// Expanding predictor was projected to the cohesionless cone tip
    /// (`ε → 0`, zero stress).
    Tip,
}

/// Material parameters for the Drucker–Prager cone in log-strain space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SandModel {
    /// Friction coefficient `α ≥ 0` (dimensionless). Larger == steeper cone ==
    /// more shear resistance per unit confining pressure.
    friction_alpha: f32,
    /// Linear hardening coefficient `≥ 0`: the effective friction grows as
    /// `α + hardening · accumulated_plastic_strain`.
    hardening: f32,
}

impl SandModel {
    /// Builds a model directly from a friction coefficient `α > 0` and a
    /// hardening coefficient `≥ 0`. Returns `None` on non-finite or
    /// out-of-range inputs.
    #[must_use]
    pub fn from_alpha(friction_alpha: f32, hardening: f32) -> Option<Self> {
        if friction_alpha.is_finite()
            && friction_alpha > 0.0
            && hardening.is_finite()
            && hardening >= 0.0
        {
            Some(Self {
                friction_alpha,
                hardening,
            })
        } else {
            None
        }
    }

    /// Builds a model from an internal friction angle in degrees (strictly
    /// between `0` and `90`) and a hardening coefficient `≥ 0`.
    ///
    /// The friction coefficient follows the standard Drucker–Prager/Mohr–
    /// Coulomb match `α = √(2/3) · 2 sinφ / (3 − sinφ)`.
    #[must_use]
    pub fn from_friction_angle(angle_degrees: f32, hardening: f32) -> Option<Self> {
        if !(angle_degrees.is_finite() && angle_degrees > 0.0 && angle_degrees < 90.0) {
            return None;
        }
        let s = f64::from(angle_degrees).to_radians().sin();
        let alpha = ((2.0_f64 / 3.0).sqrt() * 2.0 * s / (3.0 - s)) as f32;
        Self::from_alpha(alpha, hardening)
    }

    /// A non-hardening (perfectly plastic) cohesionless sand with the given
    /// friction angle in degrees.
    #[must_use]
    pub fn cohesionless(angle_degrees: f32) -> Option<Self> {
        Self::from_friction_angle(angle_degrees, 0.0)
    }

    /// Base friction coefficient `α`.
    #[must_use]
    pub fn friction_alpha(&self) -> f32 {
        self.friction_alpha
    }

    /// Linear hardening coefficient.
    #[must_use]
    pub fn hardening(&self) -> f32 {
        self.hardening
    }

    /// Effective friction coefficient after `accumulated` plastic strain.
    #[must_use]
    pub fn effective_alpha(&self, accumulated: f32) -> f32 {
        self.friction_alpha + self.hardening * accumulated.max(0.0)
    }
}

/// Persistent per-element sand plastic state carried across simulation steps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SandState {
    plastic_gradient: Mat3,
    accumulated_strain: f32,
}

impl SandState {
    /// The undeformed rest state: `Fₚ = I`, zero accumulated strain.
    #[must_use]
    pub fn rest() -> Self {
        Self {
            plastic_gradient: Mat3::IDENTITY,
            accumulated_strain: 0.0,
        }
    }

    /// Current plastic deformation gradient `Fₚ`.
    #[must_use]
    pub fn plastic_gradient(&self) -> Mat3 {
        self.plastic_gradient
    }

    /// Accumulated plastic strain so far (drives hardening).
    #[must_use]
    pub fn accumulated_strain(&self) -> f32 {
        self.accumulated_strain
    }
}

impl Default for SandState {
    fn default() -> Self {
        Self::rest()
    }
}

/// Outcome of a single Drucker–Prager return-mapping step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SandStep {
    /// Elastic deformation gradient `Fₑ` to feed the stress routine.
    pub elastic_gradient: Mat3,
    /// Plastic strain consumed this step (`0` when still elastic).
    pub plastic_increment: f32,
    /// Which return-map branch was taken.
    pub mode: SandYield,
}

/// Performs the Drucker–Prager elastic-predictor / plastic-return update for one
/// element.
///
/// `lame` supplies the shear and first Lamé parameters that set the cone slope;
/// the carried [`SandState`] is mutated in place when the material yields. The
/// returned elastic gradient always satisfies `f_total ≈ Fₑ · Fₚ` with the
/// updated `Fₚ`.
#[must_use]
pub fn return_map_sand(
    f_total: Mat3,
    lame: &LameParameters,
    model: &SandModel,
    state: &mut SandState,
) -> SandStep {
    let fp_inv = state.plastic_gradient.inverse();
    let fe_trial = f_total * fp_inv;

    let svd = svd3(fe_trial);
    let sigma = svd.sigma;
    let signs = Vec3::new(sign_of(sigma.x), sign_of(sigma.y), sign_of(sigma.z));
    let eps = Vec3::new(hencky(sigma.x), hencky(sigma.y), hencky(sigma.z));

    let trace = eps.x + eps.y + eps.z;
    let mean = trace / 3.0;
    let dev = eps - Vec3::splat(mean);
    let dev_norm = dev.length();

    let alpha = model.effective_alpha(state.accumulated_strain);
    // Cone slope factor (3λ + 2μ)/(2μ); μ > 0 for any valid elastic material.
    let two_mu = 2.0 * lame.mu;
    let cone = if two_mu > 0.0 {
        (3.0 * lame.lambda + two_mu) / two_mu
    } else {
        0.0
    };

    let (eps_elastic, increment, mode) = if trace > 0.0 {
        // Expansion: cohesionless material snaps to the cone tip (zero stress).
        (Vec3::ZERO, eps.length(), SandYield::Tip)
    } else if dev_norm <= MIN_DEVIATORIC_NORM {
        // Hydrostatic compression lies on the cone axis: always elastic.
        (eps, 0.0, SandYield::Elastic)
    } else {
        let yield_value = dev_norm + cone * trace * alpha;
        if yield_value <= 0.0 {
            (eps, 0.0, SandYield::Elastic)
        } else {
            // Radial return of the deviatoric strain onto the cone surface.
            let returned = eps - dev * (yield_value / dev_norm);
            (returned, yield_value, SandYield::ConeSurface)
        }
    };

    if mode == SandYield::Elastic {
        return SandStep {
            elastic_gradient: fe_trial,
            plastic_increment: 0.0,
            mode,
        };
    }

    let sigma_elastic = Vec3::new(
        signs.x * stretch_from_log(eps_elastic.x),
        signs.y * stretch_from_log(eps_elastic.y),
        signs.z * stretch_from_log(eps_elastic.z),
    );
    let fe_new = svd.u * Mat3::from_diagonal(sigma_elastic) * svd.v.transpose();

    state.plastic_gradient = fe_new.inverse() * f_total;
    state.accumulated_strain += increment;

    SandStep {
        elastic_gradient: fe_new,
        plastic_increment: increment,
        mode,
    }
}

/// Principal Hencky strain `ln|σ|` of a (clamped) signed singular value,
/// evaluated in `f64` for deterministic libm-free precision then narrowed.
#[inline]
fn hencky(sigma: f32) -> f32 {
    (f64::from(sigma.abs().max(MIN_STRETCH)).ln()) as f32
}

/// Inverse of [`hencky`]: the stretch magnitude `exp(ε)`, evaluated in `f64`.
#[inline]
fn stretch_from_log(eps: f32) -> f32 {
    (f64::from(eps).exp()) as f32
}

/// Sign helper that treats exact zero as `+1` so a collapsed axis reconstructs
/// to a positive stretch rather than vanishing.
#[inline]
fn sign_of(x: f32) -> f32 {
    if x < 0.0 {
        -1.0
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tet_fem_stiffness::IsotropicElasticity;

    fn lame() -> LameParameters {
        LameParameters::from_isotropic(&IsotropicElasticity::new(1.0e6, 0.3).unwrap())
    }

    fn principal(f: Mat3) -> (f32, Vec3) {
        let svd = svd3(f);
        let eps = Vec3::new(
            hencky(svd.sigma.x),
            hencky(svd.sigma.y),
            hencky(svd.sigma.z),
        );
        let mean = (eps.x + eps.y + eps.z) / 3.0;
        let trace = eps.x + eps.y + eps.z;
        (trace, eps - Vec3::splat(mean))
    }

    #[test]
    fn params_validate_ranges() {
        assert!(SandModel::from_friction_angle(30.0, 0.0).is_some());
        assert!(SandModel::from_friction_angle(30.0, 2.0).is_some());
        assert!(SandModel::from_friction_angle(0.0, 0.0).is_none());
        assert!(SandModel::from_friction_angle(90.0, 0.0).is_none());
        assert!(SandModel::from_friction_angle(30.0, -1.0).is_none());
        assert!(SandModel::from_alpha(0.0, 0.0).is_none());
        assert!(SandModel::from_alpha(f32::NAN, 0.0).is_none());
    }

    #[test]
    fn friction_angle_matches_closed_form() {
        // φ = 30° → sinφ = 0.5 → α = √(2/3) · 1 / 2.5.
        let model = SandModel::from_friction_angle(30.0, 0.0).unwrap();
        let expected = (2.0_f64 / 3.0).sqrt() as f32 / 2.5;
        assert!((model.friction_alpha() - expected).abs() < 1e-6);
    }

    #[test]
    fn rest_state_is_identity() {
        let s = SandState::rest();
        assert_eq!(s.plastic_gradient(), Mat3::IDENTITY);
        assert_eq!(s.accumulated_strain(), 0.0);
        assert_eq!(SandState::default(), s);
    }

    #[test]
    fn small_shear_under_confinement_stays_elastic() {
        let model = SandModel::cohesionless(30.0).unwrap();
        let mut state = SandState::rest();
        // Mild compression, little shear: well inside the cone.
        let f = Mat3::from_diagonal(Vec3::new(0.96, 0.95, 0.94));
        let step = return_map_sand(f, &lame(), &model, &mut state);
        assert_eq!(step.mode, SandYield::Elastic);
        assert_eq!(state, SandState::rest(), "elastic step leaves Fp = I");
        assert_eq!(step.elastic_gradient, f);
    }

    #[test]
    fn expansion_projects_to_tip() {
        let model = SandModel::cohesionless(35.0).unwrap();
        let mut state = SandState::rest();
        let f = Mat3::from_diagonal(Vec3::new(1.2, 1.1, 1.15));
        let step = return_map_sand(f, &lame(), &model, &mut state);
        assert_eq!(step.mode, SandYield::Tip);
        // Tip ⇒ ε = 0 ⇒ Fe is a pure rotation/reflection ⇒ FeᵀFe ≈ I.
        let fe = step.elastic_gradient;
        let gram = fe.transpose() * fe;
        assert!(
            (0..3).all(|c| (gram.col(c) - Mat3::IDENTITY.col(c)).length() < 1e-3),
            "tip elastic gradient should be orthonormal, FeᵀFe = {gram:?}"
        );
        assert!(step.plastic_increment > 0.0);
    }

    #[test]
    fn shear_dominant_load_projects_onto_cone_surface() {
        let model = SandModel::cohesionless(30.0).unwrap();
        let lame = lame();
        let mut state = SandState::rest();
        // Large shear, near-zero volume change: clearly outside the cone.
        let f = Mat3::from_diagonal(Vec3::new(1.4, 0.95, 0.75));
        let step = return_map_sand(f, &lame, &model, &mut state);
        assert_eq!(step.mode, SandYield::ConeSurface);

        // The projected elastic strain must sit on the cone surface: y ≈ 0.
        let (trace, dev) = principal(step.elastic_gradient);
        let two_mu = 2.0 * lame.mu;
        let cone = (3.0 * lame.lambda + two_mu) / two_mu;
        let y = dev.length() + cone * trace * model.friction_alpha();
        assert!(
            y.abs() < 1e-3,
            "projected point should lie on the cone, y = {y}"
        );

        // F = Fe · Fp reconstruction.
        let recon = step.elastic_gradient * state.plastic_gradient();
        assert!(
            (0..3).all(|c| (recon.col(c) - f.col(c)).length() < 1e-3),
            "F = Fe·Fp reconstruction"
        );
    }

    #[test]
    fn confining_pressure_enlarges_elastic_region() {
        // Pressure-dependent friction: the SAME deviatoric strain yields at low
        // confinement but stays elastic once enough hydrostatic compression is
        // added (scaling F by s < 1 shifts the trace without changing ε̂).
        let model = SandModel::cohesionless(30.0).unwrap();
        let lame = lame();
        let shear = Mat3::from_diagonal(Vec3::new(1.4, 0.95, 0.75));

        let mut low = SandState::rest();
        let step_low = return_map_sand(shear, &lame, &model, &mut low);
        assert_eq!(
            step_low.mode,
            SandYield::ConeSurface,
            "low confinement yields"
        );

        let compressed = Mat3::from_diagonal(Vec3::splat(0.7)) * shear;
        let mut high = SandState::rest();
        let step_high = return_map_sand(compressed, &lame, &model, &mut high);
        assert_eq!(
            step_high.mode,
            SandYield::Elastic,
            "added confining pressure should keep the same shear elastic"
        );
    }

    #[test]
    fn hardening_raises_effective_friction() {
        let model = SandModel::from_friction_angle(30.0, 4.0).unwrap();
        assert_eq!(model.effective_alpha(0.0), model.friction_alpha());
        assert!(model.effective_alpha(0.5) > model.friction_alpha());
    }

    #[test]
    fn repeated_yield_accumulates_monotonically() {
        let model = SandModel::cohesionless(28.0).unwrap();
        let lame = lame();
        let mut state = SandState::rest();
        let a = return_map_sand(
            Mat3::from_diagonal(Vec3::new(1.4, 0.95, 0.75)),
            &lame,
            &model,
            &mut state,
        );
        let after_first = state.accumulated_strain();
        let _ = return_map_sand(
            Mat3::from_diagonal(Vec3::new(1.6, 0.9, 0.7)),
            &lame,
            &model,
            &mut state,
        );
        assert!(a.plastic_increment > 0.0);
        assert!(state.accumulated_strain() >= after_first);
    }

    #[test]
    fn inverted_predictor_stays_finite() {
        let model = SandModel::cohesionless(30.0).unwrap();
        let mut state = SandState::rest();
        let f = Mat3::from_cols(
            Vec3::new(-1.5, 0.0, 0.0),
            Vec3::new(0.0, 1.1, 0.0),
            Vec3::new(0.0, 0.0, 0.9),
        );
        let step = return_map_sand(f, &lame(), &model, &mut state);
        assert!((0..3).all(|c| step.elastic_gradient.col(c).is_finite()));
    }

    #[test]
    fn is_deterministic() {
        let model = SandModel::from_friction_angle(32.0, 1.0).unwrap();
        let lame = lame();
        let f = Mat3::from_diagonal(Vec3::new(1.45, 0.92, 0.78));
        let mut a = SandState::rest();
        let mut b = SandState::rest();
        let sa = return_map_sand(f, &lame, &model, &mut a);
        let sb = return_map_sand(f, &lame, &model, &mut b);
        assert_eq!(sa, sb);
        assert_eq!(a, b);
    }
}
