//! Cohesive Drucker–Prager finite-strain elastoplasticity with a tension cutoff
//! for soils, concrete, and rock on a tetrahedral `FEM` element.
//!
//! Where [`tet_fem_sand_plasticity`](super::tet_fem_sand_plasticity) models
//! *cohesionless* grains (zero shear strength at zero confinement, apex pinned
//! at the origin), many geomaterials are *cohesive*: they carry some shear at
//! zero pressure and even a little tension before cracking. This module adds
//! two ingredients to the same principal-logarithmic (Hencky) strain return map
//! built on the multiplicative split `F = Fₑ · Fₚ`:
//!
//! * **Cohesion** `β ≥ 0` shifts the Drucker–Prager cone up the hydrostatic
//!   axis, so its apex sits at a *tensile* volumetric strain and the material
//!   sustains shear `‖ε̂‖ ≤ β` at zero confinement.
//! * **A tension cutoff** `ε_t ≥ 0` caps how far into tension the elastic cone
//!   may extend, replacing the sharp apex with a flat tensile cap. Beyond it the
//!   predictor returns to the cutoff (a corner/edge return when it is also
//!   outside the shear cone).
//!
//! Writing `tr ε` for the volumetric strain, `ε̂ = ε − (tr ε)/3` for the
//! deviatoric part, and `c = (3λ + 2μ)/(2μ)` for the cone slope factor, the two
//! yield functions are
//!
//! ```text
//! f(ε) = ‖ε̂‖ + c · α · (tr ε) − β      (shear cone; ≤ 0 elastic)
//! g(ε) = (tr ε) − min(ε_t, β/(c·α))    (tension cutoff; ≤ 0 elastic)
//! ```
//!
//! Four regimes result:
//!
//! * **Interior** (`f ≤ 0` and `g ≤ 0`): fully elastic, the predictor is kept.
//! * **Shear** (`f > 0`, `g ≤ 0`): radially return the deviatoric strain onto
//!   the cone surface, keeping the volumetric part.
//! * **Tension cutoff** (`g > 0`): project the volumetric strain back onto the
//!   cutoff plane and, if the predictor is also outside the cone there, shrink
//!   the deviatoric strain to the cone radius at the cutoff (a corner return).
//!
//! Because more confining pressure (more negative `tr ε`) enlarges the elastic
//! region, the model reproduces pressure-dependent friction; cohesion raises
//! the whole envelope and the cutoff bounds tensile strength independently.
//! Optional linear hardening grows the effective friction with accumulated
//! plastic strain.
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

/// Which branch of the cohesive Drucker–Prager return map produced a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DruckerPragerYield {
    /// Predictor was inside both the cone and the tension cap: fully elastic.
    Elastic,
    /// Deviatoric strain was projected radially onto the cone surface, keeping
    /// the volumetric part.
    ConeSurface,
    /// Volumetric strain exceeded the tension cutoff and was returned onto the
    /// cutoff plane (shrinking the deviatoric part to the cone radius there
    /// when the predictor was also outside the cone).
    TensionCutoff,
}

/// Material parameters for the cohesive Drucker–Prager cone plus tension cutoff
/// in log-strain space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DruckerPragerModel {
    /// Friction coefficient `α > 0` (dimensionless). Larger == steeper cone ==
    /// more shear resistance per unit confining pressure.
    friction_alpha: f32,
    /// Cohesion `β ≥ 0` in log-strain units: the shear strain the material can
    /// carry at zero confinement. `0` recovers a cohesionless cone.
    cohesion: f32,
    /// Linear hardening coefficient `≥ 0`: the effective friction grows as
    /// `α + hardening · accumulated_plastic_strain`.
    hardening: f32,
    /// Tension cutoff `ε_t ≥ 0`: the largest volumetric (tensile) log-strain the
    /// elastic cone may reach. The effective limit is the smaller of this and
    /// the cone apex `β/(c·α)`.
    tension_cutoff: f32,
}

impl DruckerPragerModel {
    /// Builds a model from a friction coefficient `α > 0`, cohesion `β ≥ 0`,
    /// hardening `≥ 0`, and tension cutoff `ε_t ≥ 0`. Returns `None` on
    /// non-finite or out-of-range inputs.
    #[must_use]
    pub fn new(
        friction_alpha: f32,
        cohesion: f32,
        hardening: f32,
        tension_cutoff: f32,
    ) -> Option<Self> {
        if friction_alpha.is_finite()
            && friction_alpha > 0.0
            && cohesion.is_finite()
            && cohesion >= 0.0
            && hardening.is_finite()
            && hardening >= 0.0
            && tension_cutoff.is_finite()
            && tension_cutoff >= 0.0
        {
            Some(Self {
                friction_alpha,
                cohesion,
                hardening,
                tension_cutoff,
            })
        } else {
            None
        }
    }

    /// Builds a model from an internal friction angle in degrees (strictly
    /// between `0` and `90`), cohesion `β ≥ 0`, hardening `≥ 0`, and tension
    /// cutoff `ε_t ≥ 0`.
    ///
    /// The friction coefficient follows the standard Drucker–Prager/Mohr–
    /// Coulomb match `α = √(2/3) · 2 sinφ / (3 − sinφ)`.
    #[must_use]
    pub fn from_friction_angle(
        angle_degrees: f32,
        cohesion: f32,
        hardening: f32,
        tension_cutoff: f32,
    ) -> Option<Self> {
        if !(angle_degrees.is_finite() && angle_degrees > 0.0 && angle_degrees < 90.0) {
            return None;
        }
        let s = f64::from(angle_degrees).to_radians().sin();
        let alpha = ((2.0_f64 / 3.0).sqrt() * 2.0 * s / (3.0 - s)) as f32;
        Self::new(alpha, cohesion, hardening, tension_cutoff)
    }

    /// Base friction coefficient `α`.
    #[must_use]
    pub fn friction_alpha(&self) -> f32 {
        self.friction_alpha
    }

    /// Cohesion `β`.
    #[must_use]
    pub fn cohesion(&self) -> f32 {
        self.cohesion
    }

    /// Linear hardening coefficient.
    #[must_use]
    pub fn hardening(&self) -> f32 {
        self.hardening
    }

    /// Tension cutoff `ε_t`.
    #[must_use]
    pub fn tension_cutoff(&self) -> f32 {
        self.tension_cutoff
    }

    /// Effective friction coefficient after `accumulated` plastic strain.
    #[must_use]
    pub fn effective_alpha(&self, accumulated: f32) -> f32 {
        self.friction_alpha + self.hardening * accumulated.max(0.0)
    }
}

/// Persistent per-element cohesive Drucker–Prager plastic state carried across
/// simulation steps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DruckerPragerState {
    plastic_gradient: Mat3,
    accumulated_strain: f32,
}

impl DruckerPragerState {
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

impl Default for DruckerPragerState {
    fn default() -> Self {
        Self::rest()
    }
}

/// Outcome of a single cohesive Drucker–Prager return-mapping step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DruckerPragerStep {
    /// Elastic deformation gradient `Fₑ` to feed the stress routine.
    pub elastic_gradient: Mat3,
    /// Plastic strain consumed this step (`0` when still elastic).
    pub plastic_increment: f32,
    /// Which return-map branch was taken.
    pub mode: DruckerPragerYield,
}

/// Performs the cohesive Drucker–Prager elastic-predictor / plastic-return
/// update (with tension cutoff) for one element.
///
/// `lame` supplies the shear and first Lamé parameters that set the cone slope;
/// the carried [`DruckerPragerState`] is mutated in place when the material
/// yields. The returned elastic gradient always satisfies `f_total ≈ Fₑ · Fₚ`
/// with the updated `Fₚ`.
#[must_use]
pub fn return_map_drucker_prager(
    f_total: Mat3,
    lame: &LameParameters,
    model: &DruckerPragerModel,
    state: &mut DruckerPragerState,
) -> DruckerPragerStep {
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
    let beta = model.cohesion;

    // Volumetric strain where the shear cone meets the hydrostatic axis
    // (`‖ε̂‖ = 0`). The effective tension limit never exceeds this apex.
    let apex = if cone * alpha > 0.0 {
        beta / (cone * alpha)
    } else {
        f32::INFINITY
    };
    let tension_limit = model.tension_cutoff.min(apex);

    // Shear-cone yield value (positive == outside the cone).
    let f_val = dev_norm + cone * alpha * trace - beta;
    // Tension-cutoff yield value (positive == past the cutoff).
    let g_val = trace - tension_limit;

    let (eps_elastic, mode) = if g_val > 0.0 {
        // Tension-cutoff return: pull the volumetric strain back onto the
        // cutoff plane. The deviatoric cone radius available at the cutoff is
        // `β − c·α·ε_t ≥ 0` (guaranteed since `ε_t ≤ apex`); shrink the
        // deviatoric part to it only when the predictor overshoots.
        let cutoff_mean = tension_limit / 3.0;
        let cone_radius = (beta - cone * alpha * tension_limit).max(0.0);
        let dev_elastic = if dev_norm > cone_radius && dev_norm > MIN_DEVIATORIC_NORM {
            dev * (cone_radius / dev_norm)
        } else {
            dev
        };
        (
            Vec3::splat(cutoff_mean) + dev_elastic,
            DruckerPragerYield::TensionCutoff,
        )
    } else if dev_norm <= MIN_DEVIATORIC_NORM {
        // Hydrostatic strain inside the cutoff lies on the cone axis: elastic.
        (eps, DruckerPragerYield::Elastic)
    } else if f_val <= 0.0 {
        (eps, DruckerPragerYield::Elastic)
    } else {
        // Radial return of the deviatoric strain onto the cone surface,
        // keeping the volumetric part.
        let returned = eps - dev * (f_val / dev_norm);
        (returned, DruckerPragerYield::ConeSurface)
    };

    if mode == DruckerPragerYield::Elastic {
        return DruckerPragerStep {
            elastic_gradient: fe_trial,
            plastic_increment: 0.0,
            mode,
        };
    }

    let increment = (eps - eps_elastic).length();
    let sigma_elastic = Vec3::new(
        signs.x * stretch_from_log(eps_elastic.x),
        signs.y * stretch_from_log(eps_elastic.y),
        signs.z * stretch_from_log(eps_elastic.z),
    );
    let fe_new = svd.u * Mat3::from_diagonal(sigma_elastic) * svd.v.transpose();

    state.plastic_gradient = fe_new.inverse() * f_total;
    state.accumulated_strain += increment;

    DruckerPragerStep {
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

    // Cohesive clay-like model: 30° friction, modest cohesion, small tension
    // cutoff, no hardening.
    fn model() -> DruckerPragerModel {
        DruckerPragerModel::from_friction_angle(30.0, 0.05, 0.0, 0.02).unwrap()
    }

    fn principal_strain(f: Mat3) -> Vec3 {
        let svd = svd3(f);
        Vec3::new(
            hencky(svd.sigma.x),
            hencky(svd.sigma.y),
            hencky(svd.sigma.z),
        )
    }

    // Diagonal deformation gradient whose principal log-strains are exactly the
    // supplied vector (singular values exp(e_i)).
    fn from_log_strain(e: Vec3) -> Mat3 {
        Mat3::from_diagonal(Vec3::new(
            stretch_from_log(e.x),
            stretch_from_log(e.y),
            stretch_from_log(e.z),
        ))
    }

    #[test]
    fn rejects_invalid_parameters() {
        assert!(DruckerPragerModel::new(0.0, 0.05, 0.0, 0.02).is_none());
        assert!(DruckerPragerModel::new(0.3, -0.1, 0.0, 0.02).is_none());
        assert!(DruckerPragerModel::new(0.3, 0.05, -1.0, 0.02).is_none());
        assert!(DruckerPragerModel::new(0.3, 0.05, 0.0, -0.02).is_none());
        assert!(DruckerPragerModel::new(0.3, 0.05, 0.0, 0.02).is_some());
        assert!(DruckerPragerModel::from_friction_angle(0.0, 0.0, 0.0, 0.0).is_none());
        assert!(DruckerPragerModel::from_friction_angle(90.0, 0.0, 0.0, 0.0).is_none());
    }

    #[test]
    fn rest_pose_is_elastic() {
        let mut state = DruckerPragerState::rest();
        let step = return_map_drucker_prager(Mat3::IDENTITY, &lame(), &model(), &mut state);
        assert_eq!(step.mode, DruckerPragerYield::Elastic);
        assert_eq!(step.plastic_increment, 0.0);
        assert_eq!(state, DruckerPragerState::rest());
    }

    #[test]
    fn cohesion_keeps_small_shear_elastic() {
        // Volume-preserving deviatoric strain (trace 0) with norm below the
        // cohesion stays elastic — a cohesionless cone would yield instead.
        let delta = 0.01_f32;
        let f = from_log_strain(Vec3::new(2.0 * delta, -delta, -delta));
        let mut state = DruckerPragerState::rest();
        let step = return_map_drucker_prager(f, &lame(), &model(), &mut state);
        assert_eq!(step.mode, DruckerPragerYield::Elastic);
        assert_eq!(state, DruckerPragerState::rest());
    }

    #[test]
    fn large_shear_returns_to_cone() {
        // A large volume-preserving shear overshoots the cohesion and must be
        // projected onto the cone surface.
        let delta = 0.08_f32;
        let f = from_log_strain(Vec3::new(2.0 * delta, -delta, -delta));
        let mut state = DruckerPragerState::rest();
        let step = return_map_drucker_prager(f, &lame(), &model(), &mut state);
        assert_eq!(step.mode, DruckerPragerYield::ConeSurface);
        assert!(step.plastic_increment > 0.0);
        // Reconstruction: Fₑ · Fₚ == F_total.
        let recon = step.elastic_gradient * state.plastic_gradient;
        for i in 0..3 {
            for j in 0..3 {
                assert!((recon.col(i)[j] - f.col(i)[j]).abs() < 1e-4);
            }
        }
        // Returned deviatoric norm equals the cone radius at trace 0 (= β).
        let eps_e = principal_strain(step.elastic_gradient);
        let mean = (eps_e.x + eps_e.y + eps_e.z) / 3.0;
        let dev_norm = (eps_e - Vec3::splat(mean)).length();
        assert!(
            (dev_norm - model().cohesion()).abs() < 2e-3,
            "dev {dev_norm}"
        );
    }

    #[test]
    fn compression_enlarges_elastic_region() {
        // The same deviatoric strain that yields at zero pressure stays elastic
        // under enough confinement (negative trace shifts the cone outward).
        let delta = 0.08_f32;
        let compress = -0.1_f32;
        let f = from_log_strain(Vec3::new(
            2.0 * delta + compress,
            -delta + compress,
            -delta + compress,
        ));
        let mut state = DruckerPragerState::rest();
        let step = return_map_drucker_prager(f, &lame(), &model(), &mut state);
        assert_eq!(step.mode, DruckerPragerYield::Elastic);
    }

    #[test]
    fn tension_beyond_cutoff_returns_to_cutoff() {
        // Uniform expansion drives the volumetric strain well past the tension
        // cutoff; the step must return onto the cutoff plane.
        let f = from_log_strain(Vec3::splat(0.05));
        let mut state = DruckerPragerState::rest();
        let step = return_map_drucker_prager(f, &lame(), &model(), &mut state);
        assert_eq!(step.mode, DruckerPragerYield::TensionCutoff);
        assert!(step.plastic_increment > 0.0);
        let eps_e = principal_strain(step.elastic_gradient);
        let trace = eps_e.x + eps_e.y + eps_e.z;
        assert!(
            (trace - model().tension_cutoff()).abs() < 2e-3,
            "trace {trace} should sit on the cutoff"
        );
        let recon = step.elastic_gradient * state.plastic_gradient;
        for i in 0..3 {
            for j in 0..3 {
                assert!((recon.col(i)[j] - f.col(i)[j]).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn hardening_grows_effective_alpha() {
        let m = DruckerPragerModel::new(0.3, 0.05, 2.0, 0.02).unwrap();
        assert!((m.effective_alpha(0.0) - 0.3).abs() < 1e-6);
        assert!((m.effective_alpha(0.1) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn is_deterministic() {
        let f = from_log_strain(Vec3::new(0.16, -0.08, -0.08));
        let mut a = DruckerPragerState::rest();
        let mut b = DruckerPragerState::rest();
        let sa = return_map_drucker_prager(f, &lame(), &model(), &mut a);
        let sb = return_map_drucker_prager(f, &lame(), &model(), &mut b);
        assert_eq!(sa, sb);
        assert_eq!(a, b);
    }
}
