//! Mohr–Coulomb finite-strain elastoplasticity for soils and rock on a
//! tetrahedral `FEM` element, using the exact multisurface return map in
//! principal stress space.
//!
//! Where [`tet_fem_drucker_prager_plasticity`](super::tet_fem_drucker_prager_plasticity)
//! smooths the yield surface into a circular cone (a single smooth function of
//! the stress invariants), the Mohr–Coulomb criterion is the *faceted*
//! hexagonal pyramid that geotechnical engineering actually calibrates against.
//! Its surface has planes, edges, and a single apex, so a correct return map
//! must decide **which feature** the elastic predictor projects onto rather
//! than radially shrinking a smooth invariant.
//!
//! Working in the principal (Hencky) elastic strains `ε₁ ≥ ε₂ ≥ ε₃` produced by
//! the SVD of the trial elastic gradient (`F = Fₑ · Fₚ`), the principal
//! Kirchhoff stresses are
//!
//! ```text
//! σᵢ = 2μ · εᵢ + λ · (ε₁ + ε₂ + ε₃)
//! ```
//!
//! and, because the SVD orders the stretches descending, `σ₁ ≥ σ₂ ≥ σ₃`. With
//! friction angle `φ`, dilation angle `ψ`, cohesion `c`, and `k = 2c·cosφ`, the
//! primary yield function (tension positive) is
//!
//! ```text
//! f(σ) = (σ₁ − σ₃) + (σ₁ + σ₃) · sinφ − k      (≤ 0 elastic)
//! ```
//!
//! The non-associated plastic flow uses the dilation angle `ψ` in place of `φ`.
//! The return follows Koiter's rule with the standard feature priority
//! (de Souza Neto / Clausen): try the main plane, then the right edge
//! (triaxial compression, `σ₂ → σ₃`), then the left edge (triaxial extension,
//! `σ₁ → σ₂`), and finally the tensile apex. The first candidate whose returned
//! stresses stay ordered (`σ′₁ ≥ σ′₂ ≥ σ′₃`) with non-negative plastic
//! multipliers is accepted.
//!
//! The elastic strain `D` operator in principal space is isotropic,
//! `(D·m)ᵢ = λ·Σⱼ mⱼ + 2μ·mᵢ`, so every projection is a tiny 1×1 or 2×2 solve.
//! Returned stresses map back to elastic log-strains through the inverse
//! relation `εᵢ = (σ′ᵢ − λ·trσ′/(3λ+2μ)) / (2μ)` and are exponentiated to
//! singular values to rebuild `Fₑ`.
//!
//! Clean-room implementation over the crate's own [`svd3`]; logs/exponentials
//! and all trigonometry run in `f64` for deterministic, libm-free precision.
//! No Unreal Engine source or derived code.

use crate::collider::tet_fem_constitutive::LameParameters;
use crate::mpm::svd3;
use glam::{Mat3, Vec3};

/// Principal stretches are clamped to this magnitude before taking a log, so an
/// inverted or collapsed predictor never produces a non-finite strain.
const MIN_STRETCH: f32 = 1e-4;

/// A plastic multiplier is accepted when it is at least this (slightly
/// negative) value, absorbing round-off in the feature solves.
const MIN_MULTIPLIER: f32 = -1e-6;

/// Which feature of the Mohr–Coulomb pyramid the return map projected onto.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MohrCoulombYield {
    /// Predictor was inside the pyramid: fully elastic.
    Elastic,
    /// Returned onto a single yield plane (the generic case).
    MainPlane,
    /// Returned onto an edge where two yield planes meet (triaxial stress).
    Edge,
    /// Returned onto the tensile apex of the pyramid (hydrostatic tension).
    Apex,
}

/// Material parameters for the Mohr–Coulomb pyramid in principal stress space.
///
/// Angles are stored pre-reduced to their sines/cosine so the hot return map is
/// trig-free and deterministic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MohrCoulombModel {
    /// `sinφ` of the internal friction angle `φ ∈ (0°, 90°)`.
    sin_phi: f32,
    /// `cosφ` of the internal friction angle.
    cos_phi: f32,
    /// `sinψ` of the dilation angle `ψ ∈ [0°, φ]` (non-associated flow).
    sin_psi: f32,
    /// Base cohesion `c ≥ 0` (stress units).
    cohesion: f32,
    /// Linear hardening coefficient `≥ 0`: effective cohesion grows as
    /// `c + hardening · accumulated_plastic_strain`.
    hardening: f32,
}

impl MohrCoulombModel {
    /// Builds a model from a friction angle and dilation angle in degrees, a
    /// cohesion `c ≥ 0`, and a hardening coefficient `≥ 0`.
    ///
    /// Returns `None` unless `0° < φ < 90°`, `0° ≤ ψ ≤ φ`, `c ≥ 0`, and
    /// `hardening ≥ 0`, or if any input is non-finite. The dilation angle is
    /// capped at the friction angle so the plastic flow never generates energy.
    #[must_use]
    pub fn from_angles(
        friction_degrees: f32,
        dilation_degrees: f32,
        cohesion: f32,
        hardening: f32,
    ) -> Option<Self> {
        if !(friction_degrees.is_finite() && friction_degrees > 0.0 && friction_degrees < 90.0) {
            return None;
        }
        if !(dilation_degrees.is_finite()
            && dilation_degrees >= 0.0
            && dilation_degrees <= friction_degrees)
        {
            return None;
        }
        if !(cohesion.is_finite() && cohesion >= 0.0 && hardening.is_finite() && hardening >= 0.0) {
            return None;
        }
        let phi = f64::from(friction_degrees).to_radians();
        let psi = f64::from(dilation_degrees).to_radians();
        Some(Self {
            sin_phi: phi.sin() as f32,
            cos_phi: phi.cos() as f32,
            sin_psi: psi.sin() as f32,
            cohesion,
            hardening,
        })
    }

    /// `sinφ` of the friction angle.
    #[must_use]
    pub fn friction_sin(&self) -> f32 {
        self.sin_phi
    }

    /// `cosφ` of the friction angle.
    #[must_use]
    pub fn friction_cos(&self) -> f32 {
        self.cos_phi
    }

    /// `sinψ` of the dilation angle.
    #[must_use]
    pub fn dilation_sin(&self) -> f32 {
        self.sin_psi
    }

    /// Base cohesion `c`.
    #[must_use]
    pub fn cohesion(&self) -> f32 {
        self.cohesion
    }

    /// Linear hardening coefficient.
    #[must_use]
    pub fn hardening(&self) -> f32 {
        self.hardening
    }

    /// Effective cohesion after `accumulated` plastic strain.
    #[must_use]
    pub fn effective_cohesion(&self, accumulated: f32) -> f32 {
        self.cohesion + self.hardening * accumulated.max(0.0)
    }
}

/// Persistent per-element Mohr–Coulomb plastic state carried across steps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MohrCoulombState {
    plastic_gradient: Mat3,
    accumulated_strain: f32,
}

impl MohrCoulombState {
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

impl Default for MohrCoulombState {
    fn default() -> Self {
        Self::rest()
    }
}

/// Outcome of a single Mohr–Coulomb return-mapping step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MohrCoulombStep {
    /// Elastic deformation gradient `Fₑ` to feed the stress routine.
    pub elastic_gradient: Mat3,
    /// Plastic strain consumed this step (`0` when still elastic).
    pub plastic_increment: f32,
    /// Which feature of the pyramid the return projected onto.
    pub mode: MohrCoulombYield,
}

/// Internal helper bundling the derived scalars for the principal return map so
/// the individual feature solves stay small and parameter-light.
struct PrincipalReturn<'a> {
    lame: &'a LameParameters,
    sin_phi: f32,
    sin_psi: f32,
    k: f32,
    order_tol: f32,
}

impl PrincipalReturn<'_> {
    /// Isotropic elastic operator applied to a principal flow direction:
    /// `(D·m)ᵢ = λ·Σⱼ mⱼ + 2μ·mᵢ`.
    #[inline]
    fn d_times_m(&self, m: Vec3) -> Vec3 {
        let sum = m.x + m.y + m.z;
        let bulk = self.lame.lambda * sum;
        let two_mu = 2.0 * self.lame.mu;
        Vec3::new(
            bulk + two_mu * m.x,
            bulk + two_mu * m.y,
            bulk + two_mu * m.z,
        )
    }

    /// Whether the returned principal stresses keep the required descending
    /// order within tolerance.
    #[inline]
    fn ordered(&self, s: Vec3) -> bool {
        s.x - s.y >= -self.order_tol && s.y - s.z >= -self.order_tol
    }

    /// Primary yield value `f(σ) = (σ₁−σ₃) + (σ₁+σ₃)·sinφ − k`.
    #[inline]
    fn yield_main(&self, s: Vec3) -> f32 {
        (s.x - s.z) + (s.x + s.z) * self.sin_phi - self.k
    }

    /// Attempts the single-plane return. The flow direction `m_a` carries the
    /// dilation angle; the gradient `grad_a` carries the friction angle.
    fn main_plane(&self, sigma: Vec3, f_trial: f32) -> Option<(Vec3, MohrCoulombYield)> {
        let m_a = Vec3::new(1.0 + self.sin_psi, 0.0, -1.0 + self.sin_psi);
        let grad_a = Vec3::new(1.0 + self.sin_phi, 0.0, -1.0 + self.sin_phi);
        let dm_a = self.d_times_m(m_a);
        let denom = grad_a.dot(dm_a);
        if denom.abs() < f32::EPSILON {
            return None;
        }
        let dlambda = f_trial / denom;
        if dlambda < MIN_MULTIPLIER {
            return None;
        }
        let returned = sigma - dm_a * dlambda;
        if self.ordered(returned) {
            Some((returned, MohrCoulombYield::MainPlane))
        } else {
            None
        }
    }

    /// Attempts a two-plane edge return. `right == true` returns onto the right
    /// edge (the second plane acts on `σ₂, σ₃`, i.e. triaxial compression);
    /// otherwise the left edge (second plane acts on `σ₁, σ₂`).
    fn edge(&self, sigma: Vec3, right: bool) -> Option<(Vec3, MohrCoulombYield)> {
        let m_a = Vec3::new(1.0 + self.sin_psi, 0.0, -1.0 + self.sin_psi);
        let grad_a = Vec3::new(1.0 + self.sin_phi, 0.0, -1.0 + self.sin_phi);
        let (m_b, grad_b, f_b) = if right {
            (
                Vec3::new(0.0, 1.0 + self.sin_psi, -1.0 + self.sin_psi),
                Vec3::new(0.0, 1.0 + self.sin_phi, -1.0 + self.sin_phi),
                (sigma.y - sigma.z) + (sigma.y + sigma.z) * self.sin_phi - self.k,
            )
        } else {
            (
                Vec3::new(1.0 + self.sin_psi, -1.0 + self.sin_psi, 0.0),
                Vec3::new(1.0 + self.sin_phi, -1.0 + self.sin_phi, 0.0),
                (sigma.x - sigma.y) + (sigma.x + sigma.y) * self.sin_phi - self.k,
            )
        };
        let f_a = self.yield_main(sigma);
        let dm_a = self.d_times_m(m_a);
        let dm_b = self.d_times_m(m_b);
        let a11 = grad_a.dot(dm_a);
        let a12 = grad_a.dot(dm_b);
        let a21 = grad_b.dot(dm_a);
        let a22 = grad_b.dot(dm_b);
        let det = a11 * a22 - a12 * a21;
        if det.abs() < f32::EPSILON {
            return None;
        }
        let dl_a = (f_a * a22 - f_b * a12) / det;
        let dl_b = (a11 * f_b - a21 * f_a) / det;
        if dl_a < MIN_MULTIPLIER || dl_b < MIN_MULTIPLIER {
            return None;
        }
        let returned = sigma - dm_a * dl_a - dm_b * dl_b;
        if self.ordered(returned) {
            Some((returned, MohrCoulombYield::Edge))
        } else {
            None
        }
    }

    /// The tensile apex of the pyramid, `σ_apex = k / (2 sinφ)` on every axis.
    fn apex(&self) -> (Vec3, MohrCoulombYield) {
        let apex = if self.sin_phi > 0.0 {
            self.k / (2.0 * self.sin_phi)
        } else {
            0.0
        };
        (Vec3::splat(apex), MohrCoulombYield::Apex)
    }
}

/// Performs the Mohr–Coulomb elastic-predictor / plastic-return update for one
/// element.
///
/// `lame` supplies the shear and first Lamé parameters; the carried
/// [`MohrCoulombState`] is mutated in place when the material yields. The
/// returned elastic gradient always satisfies `f_total ≈ Fₑ · Fₚ` with the
/// updated `Fₚ`.
#[must_use]
pub fn return_map_mohr_coulomb(
    f_total: Mat3,
    lame: &LameParameters,
    model: &MohrCoulombModel,
    state: &mut MohrCoulombState,
) -> MohrCoulombStep {
    let fp_inv = state.plastic_gradient.inverse();
    let fe_trial = f_total * fp_inv;

    let svd = svd3(fe_trial);
    let sigma_sv = svd.sigma;
    let signs = Vec3::new(
        sign_of(sigma_sv.x),
        sign_of(sigma_sv.y),
        sign_of(sigma_sv.z),
    );
    let eps = Vec3::new(hencky(sigma_sv.x), hencky(sigma_sv.y), hencky(sigma_sv.z));

    let trace = eps.x + eps.y + eps.z;
    let two_mu = 2.0 * lame.mu;
    let sigma_tr = Vec3::new(
        two_mu * eps.x + lame.lambda * trace,
        two_mu * eps.y + lame.lambda * trace,
        two_mu * eps.z + lame.lambda * trace,
    );

    let cohesion = model.effective_cohesion(state.accumulated_strain);
    let k = 2.0 * cohesion * model.cos_phi;
    let ctx = PrincipalReturn {
        lame,
        sin_phi: model.sin_phi,
        sin_psi: model.sin_psi,
        k,
        order_tol: 1e-4 * sigma_tr.abs().max_element().max(1.0),
    };

    let f_trial = ctx.yield_main(sigma_tr);
    if f_trial <= 0.0 {
        return MohrCoulombStep {
            elastic_gradient: fe_trial,
            plastic_increment: 0.0,
            mode: MohrCoulombYield::Elastic,
        };
    }

    let (sigma_ret, mode) = ctx
        .main_plane(sigma_tr, f_trial)
        .or_else(|| ctx.edge(sigma_tr, true))
        .or_else(|| ctx.edge(sigma_tr, false))
        .unwrap_or_else(|| ctx.apex());

    // Invert σ = 2μ ε + λ (trε) I to recover the returned elastic log-strain.
    let bulk = 3.0 * lame.lambda + two_mu;
    let tr_sigma = sigma_ret.x + sigma_ret.y + sigma_ret.z;
    let tr_eps = if bulk > 0.0 { tr_sigma / bulk } else { 0.0 };
    let eps_elastic = Vec3::new(
        (sigma_ret.x - lame.lambda * tr_eps) / two_mu,
        (sigma_ret.y - lame.lambda * tr_eps) / two_mu,
        (sigma_ret.z - lame.lambda * tr_eps) / two_mu,
    );

    let increment = (eps - eps_elastic).length();
    let sigma_elastic = Vec3::new(
        signs.x * stretch_from_log(eps_elastic.x),
        signs.y * stretch_from_log(eps_elastic.y),
        signs.z * stretch_from_log(eps_elastic.z),
    );
    let fe_new = svd.u * Mat3::from_diagonal(sigma_elastic) * svd.v.transpose();

    state.plastic_gradient = fe_new.inverse() * f_total;
    state.accumulated_strain += increment;

    MohrCoulombStep {
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

    // 30° friction, associated flow, cohesion in stress units, no hardening.
    fn model() -> MohrCoulombModel {
        MohrCoulombModel::from_angles(30.0, 30.0, 2000.0, 0.0).unwrap()
    }

    // Diagonal deformation gradient whose principal log-strains are exactly the
    // supplied (descending) vector.
    fn from_log_strain(e: Vec3) -> Mat3 {
        Mat3::from_diagonal(Vec3::new(
            stretch_from_log(e.x),
            stretch_from_log(e.y),
            stretch_from_log(e.z),
        ))
    }

    fn assert_reconstructs(step: &MohrCoulombStep, state: &MohrCoulombState, f: Mat3) {
        let recon = step.elastic_gradient * state.plastic_gradient;
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (recon.col(i)[j] - f.col(i)[j]).abs() < 1e-3,
                    "Fₑ·Fₚ must reconstruct F_total"
                );
            }
        }
    }

    #[test]
    fn rejects_invalid_parameters() {
        assert!(MohrCoulombModel::from_angles(0.0, 0.0, 1.0, 0.0).is_none());
        assert!(MohrCoulombModel::from_angles(90.0, 0.0, 1.0, 0.0).is_none());
        // dilation must not exceed friction.
        assert!(MohrCoulombModel::from_angles(30.0, 40.0, 1.0, 0.0).is_none());
        assert!(MohrCoulombModel::from_angles(30.0, -1.0, 1.0, 0.0).is_none());
        assert!(MohrCoulombModel::from_angles(30.0, 10.0, -1.0, 0.0).is_none());
        assert!(MohrCoulombModel::from_angles(30.0, 10.0, 1.0, -1.0).is_none());
        assert!(MohrCoulombModel::from_angles(30.0, 10.0, 1.0, 0.0).is_some());
    }

    #[test]
    fn rest_pose_is_elastic() {
        let mut state = MohrCoulombState::rest();
        let step = return_map_mohr_coulomb(Mat3::IDENTITY, &lame(), &model(), &mut state);
        assert_eq!(step.mode, MohrCoulombYield::Elastic);
        assert_eq!(step.plastic_increment, 0.0);
        assert_eq!(state, MohrCoulombState::rest());
    }

    #[test]
    fn small_shear_stays_elastic() {
        // Tiny volume-preserving shear below the cohesive strength.
        let delta = 0.001_f32;
        let f = from_log_strain(Vec3::new(delta, 0.0, -delta));
        let mut state = MohrCoulombState::rest();
        let step = return_map_mohr_coulomb(f, &lame(), &model(), &mut state);
        assert_eq!(step.mode, MohrCoulombYield::Elastic);
        assert_eq!(state, MohrCoulombState::rest());
    }

    #[test]
    fn large_shear_returns_to_main_plane() {
        // A large volume-preserving shear overshoots the yield plane.
        let delta = 0.02_f32;
        let f = from_log_strain(Vec3::new(delta, 0.0, -delta));
        let mut state = MohrCoulombState::rest();
        let step = return_map_mohr_coulomb(f, &lame(), &model(), &mut state);
        assert_eq!(step.mode, MohrCoulombYield::MainPlane);
        assert!(step.plastic_increment > 0.0);
        assert_reconstructs(&step, &state, f);
        // The returned stresses sit on the yield surface (f ≈ 0).
        let two_mu = 2.0 * lame().mu;
        let eps_e = {
            let svd = svd3(step.elastic_gradient);
            Vec3::new(
                hencky(svd.sigma.x),
                hencky(svd.sigma.y),
                hencky(svd.sigma.z),
            )
        };
        let trace = eps_e.x + eps_e.y + eps_e.z;
        let s = Vec3::new(
            two_mu * eps_e.x + lame().lambda * trace,
            two_mu * eps_e.y + lame().lambda * trace,
            two_mu * eps_e.z + lame().lambda * trace,
        );
        let k = 2.0 * model().cohesion() * model().friction_cos();
        let residual = (s.x - s.z) + (s.x + s.z) * model().friction_sin() - k;
        assert!(residual.abs() < 2.0, "residual {residual}");
    }

    #[test]
    fn hydrostatic_tension_hits_apex() {
        // Uniform expansion drives hydrostatic tension far past the apex.
        let f = from_log_strain(Vec3::splat(0.05));
        let mut state = MohrCoulombState::rest();
        let step = return_map_mohr_coulomb(f, &lame(), &model(), &mut state);
        assert_eq!(step.mode, MohrCoulombYield::Apex);
        assert!(step.plastic_increment > 0.0);
        assert_reconstructs(&step, &state, f);
        // Returned stress is hydrostatic at σ = k/(2 sinφ) = c·cotφ.
        let two_mu = 2.0 * lame().mu;
        let eps_e = {
            let svd = svd3(step.elastic_gradient);
            Vec3::new(
                hencky(svd.sigma.x),
                hencky(svd.sigma.y),
                hencky(svd.sigma.z),
            )
        };
        let trace = eps_e.x + eps_e.y + eps_e.z;
        let s = Vec3::new(
            two_mu * eps_e.x + lame().lambda * trace,
            two_mu * eps_e.y + lame().lambda * trace,
            two_mu * eps_e.z + lame().lambda * trace,
        );
        let k = 2.0 * model().cohesion() * model().friction_cos();
        let apex = k / (2.0 * model().friction_sin());
        assert!((s.x - apex).abs() < 2.0, "σ₁ {} vs apex {apex}", s.x);
        assert!((s.x - s.z).abs() < 2.0, "apex must be hydrostatic");
    }

    #[test]
    fn compression_enlarges_elastic_region() {
        // The same shear that yields at zero pressure stays elastic under
        // enough confinement (negative mean stress lowers f).
        let delta = 0.02_f32;
        let compress = -0.02_f32;
        let f = from_log_strain(Vec3::new(delta + compress, compress, -delta + compress));
        let mut state = MohrCoulombState::rest();
        let step = return_map_mohr_coulomb(f, &lame(), &model(), &mut state);
        assert_eq!(step.mode, MohrCoulombYield::Elastic);
    }

    #[test]
    fn hardening_grows_effective_cohesion() {
        let m = MohrCoulombModel::from_angles(30.0, 10.0, 1000.0, 500.0).unwrap();
        assert!((m.effective_cohesion(0.0) - 1000.0).abs() < 1e-3);
        assert!((m.effective_cohesion(2.0) - 2000.0).abs() < 1e-3);
    }

    #[test]
    fn is_deterministic() {
        let f = from_log_strain(Vec3::new(0.03, 0.0, -0.03));
        let mut a = MohrCoulombState::rest();
        let mut b = MohrCoulombState::rest();
        let sa = return_map_mohr_coulomb(f, &lame(), &model(), &mut a);
        let sb = return_map_mohr_coulomb(f, &lame(), &model(), &mut b);
        assert_eq!(sa, sb);
        assert_eq!(a, b);
    }
}
