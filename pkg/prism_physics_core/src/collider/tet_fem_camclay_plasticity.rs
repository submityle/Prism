//! Modified Cam-Clay (`MCC`) cap elastoplasticity for cohesive/frictional soils
//! on a tetrahedral `FEM` element.
//!
//! Dry sand (see [`tet_fem_sand_plasticity`]) is captured by an open
//! Drucker–Prager *cone*: it has no cap in compression and cannot compact
//! plastically. Real soils — wet clay, saturated silt, porous rock — do the
//! opposite: beyond a *pre-consolidation pressure* `p_c` they *yield in pure
//! compression*, pack denser, and in doing so grow `p_c` (the soil "remembers"
//! the largest pressure it ever felt). Modified Cam-Clay models this with a
//! closed **ellipse** in the mean/deviatoric stress plane plus **volumetric
//! hardening** of `p_c`.
//!
//! Like the sand and snow models here it is written in principal logarithmic
//! (Hencky) strain space on top of the crate's multiplicative split
//!
//! ```text
//! F = Fₑ · Fₚ
//! ```
//!
//! Each step strips the stored plastic gradient to form the elastic predictor
//! `Fₑ_trial = F · Fₚ⁻¹`, takes its signed SVD, and maps the singular values to
//! principal strains `ε = ln|Σ|`. The isotropic linear (Hencky) law gives the
//! principal Kirchhoff stress
//!
//! ```text
//! τ_i = 2μ ε_i + λ·trε
//! ```
//!
//! from which we read the two Cam-Clay invariants (compression-positive):
//!
//! ```text
//! p = −K·trε            (mean pressure, K = λ + 2μ/3)
//! q = √(3/2)·‖s‖         (von Mises shear, s_i = 2μ·ε̂_i)
//! ```
//!
//! The yield surface is the ellipse
//!
//! ```text
//! y(p, q, p_c) = q²/M² + p·(p − p_c)   ≤ 0   (elastic)
//! ```
//!
//! with critical-state slope `M`. The ellipse passes through the origin and
//! `p = p_c`, so tension (`p < 0`) always violates it (`y > 0`) — a
//! cohesionless Cam-Clay carries no tension, exactly as a soil should.
//!
//! # Return mapping (robust, not a one-shot approximation)
//!
//! The map is a genuine operator split solved to convergence:
//!
//! * **Inner solve (plastic multiplier).** With `p_c` fixed, associative flow
//!   gives closed-form predictor-corrector relations
//!   `p(Δγ) = (p_tr + K·Δγ·p_c)/(1 + 2K·Δγ)` and
//!   `q(Δγ) = q_tr/(1 + 6μ·Δγ/M²)`. The consistency residual
//!   `g(Δγ) = q(Δγ)²/M² + p(Δγ)·(p(Δγ) − p_c)` satisfies `g(0) = y_tr > 0`
//!   and `g(∞) = −p_c²/4 < 0`, so a root is bracketed and found by bisection
//!   — unconditionally convergent, no Jacobian, no divergence.
//! * **Outer solve (hardening).** `p_c` evolves with the volumetric plastic
//!   strain `Δε_vᵖ = Δγ·(2p − p_c)` through `p_c ← p_c·exp(Δε_vᵖ / H)`
//!   (compaction `2p − p_c > 0` grows `p_c`; dilation shrinks it). We iterate
//!   inner-solve → hardening update a few times to a fixed point in `p_c`.
//!
//! The returned deviatoric *direction* is inherited from the trial (radial
//! return in the deviatoric plane, correct for isotropic associative flow), so
//! only the scalar amplitudes `p`, `q` move. The elastic strain is rebuilt as
//! `ε̂ₑ_i = s_i/(2μ)`, `mean(εₑ) = −p/(3K)`, then `Σ_i = sign·exp(εₑ_i)` and
//! `Fₑ = U·diag(Σ)·Vᵀ`, keeping `F ≈ Fₑ·Fₚ` with the updated plastic gradient.
//!
//! Clean-room implementation over the crate's own [`svd3`]; every transcendental
//! (log, exp) and the whole iterative solve run in `f64` for deterministic,
//! libm-free precision then narrow to `f32`. No Unreal Engine source or derived
//! code.

use crate::collider::tet_fem_constitutive::LameParameters;
use crate::mpm::svd3;
use glam::{Mat3, Vec3};

/// Principal stretches are clamped to this magnitude before taking a log, so an
/// inverted or collapsed predictor never produces a non-finite strain.
const MIN_STRETCH: f32 = 1e-4;

/// Deviatoric strains with Frobenius norm at or below this have no well-defined
/// return direction and are treated as purely volumetric.
const MIN_DEVIATORIC_NORM: f64 = 1e-12;

/// Maximum bisection iterations for the inner plastic-multiplier solve. In
/// `f64` this resolves `Δγ` to full precision for any physical scale.
const INNER_ITERS: u32 = 100;

/// Maximum bracket-expansion doublings when searching for an upper bound.
const BRACKET_ITERS: u32 = 96;

/// Maximum bisection iterations for the inner pre-consolidation solve.
const PC_ITERS: u32 = 64;

/// Material parameters for the modified Cam-Clay ellipse plus its volumetric
/// hardening law.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CamClayModel {
    /// Critical-state slope `M > 0` (ratio of shear to mean stress at the
    /// ellipse crown). Larger `M` == taller ellipse == more shear capacity.
    slope_m: f32,
    /// Hardening modulus `H > 0` (units of volumetric plastic strain). Smaller
    /// `H` == faster growth of `p_c` under compaction (stiffer, less
    /// compressible soil). Governs `p_c ← p_c·exp(Δε_vᵖ / H)`.
    hardening_modulus: f32,
    /// Initial pre-consolidation pressure `p_c0 > 0`: the compression cap of a
    /// freshly reset element.
    initial_pre_consolidation: f32,
}

impl CamClayModel {
    /// Builds a model from the critical-state slope `M`, hardening modulus `H`,
    /// and initial pre-consolidation pressure `p_c0`. Returns `None` unless all
    /// three are finite and strictly positive.
    #[must_use]
    pub fn new(
        slope_m: f32,
        hardening_modulus: f32,
        initial_pre_consolidation: f32,
    ) -> Option<Self> {
        if slope_m.is_finite()
            && slope_m > 0.0
            && hardening_modulus.is_finite()
            && hardening_modulus > 0.0
            && initial_pre_consolidation.is_finite()
            && initial_pre_consolidation > 0.0
        {
            Some(Self {
                slope_m,
                hardening_modulus,
                initial_pre_consolidation,
            })
        } else {
            None
        }
    }

    /// Critical-state slope `M`.
    #[must_use]
    pub fn slope_m(&self) -> f32 {
        self.slope_m
    }

    /// Hardening modulus `H`.
    #[must_use]
    pub fn hardening_modulus(&self) -> f32 {
        self.hardening_modulus
    }

    /// Initial pre-consolidation pressure `p_c0`.
    #[must_use]
    pub fn initial_pre_consolidation(&self) -> f32 {
        self.initial_pre_consolidation
    }

    /// A fresh plastic state consistent with this model's `p_c0`.
    #[must_use]
    pub fn rest_state(&self) -> CamClayState {
        CamClayState::new(self.initial_pre_consolidation)
            .expect("validated model p_c0 is finite and positive")
    }
}

/// Persistent per-element Cam-Clay plastic state carried across steps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CamClayState {
    plastic_gradient: Mat3,
    pre_consolidation: f32,
}

impl CamClayState {
    /// A rest state (`Fₚ = I`) with the given pre-consolidation pressure.
    /// Returns `None` unless `pre_consolidation` is finite and `> 0`.
    #[must_use]
    pub fn new(pre_consolidation: f32) -> Option<Self> {
        if pre_consolidation.is_finite() && pre_consolidation > 0.0 {
            Some(Self {
                plastic_gradient: Mat3::IDENTITY,
                pre_consolidation,
            })
        } else {
            None
        }
    }

    /// Current plastic deformation gradient `Fₚ`.
    #[must_use]
    pub fn plastic_gradient(&self) -> Mat3 {
        self.plastic_gradient
    }

    /// Current pre-consolidation pressure `p_c` (the compression cap).
    #[must_use]
    pub fn pre_consolidation(&self) -> f32 {
        self.pre_consolidation
    }
}

/// Outcome of a single Cam-Clay return-mapping step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CamClayStep {
    /// Elastic deformation gradient `Fₑ` to feed the stress routine.
    pub elastic_gradient: Mat3,
    /// Magnitude of the principal plastic strain consumed this step
    /// (`0` when still elastic).
    pub plastic_increment: f32,
    /// Whether the predictor was outside the ellipse and had to be returned.
    pub yielded: bool,
    /// Updated pre-consolidation pressure after hardening (equals the previous
    /// value on an elastic step).
    pub pre_consolidation: f32,
}

/// Performs the modified Cam-Clay elastic-predictor / plastic-return update for
/// one element.
///
/// `lame` supplies the shear and first Lamé parameters (hence `K = λ + 2μ/3`
/// and the ellipse geometry); the carried [`CamClayState`] is mutated in place
/// when the material yields (both `Fₚ` and `p_c`). The returned elastic
/// gradient always satisfies `f_total ≈ Fₑ · Fₚ` with the updated `Fₚ`.
#[must_use]
pub fn return_map_camclay(
    f_total: Mat3,
    lame: &LameParameters,
    model: &CamClayModel,
    state: &mut CamClayState,
) -> CamClayStep {
    let fp_inv = state.plastic_gradient.inverse();
    let fe_trial = f_total * fp_inv;

    let svd = svd3(fe_trial);
    let sigma = svd.sigma;
    let signs = Vec3::new(sign_of(sigma.x), sign_of(sigma.y), sign_of(sigma.z));
    let eps = Vec3::new(hencky(sigma.x), hencky(sigma.y), hencky(sigma.z));

    let trace = f64::from(eps.x) + f64::from(eps.y) + f64::from(eps.z);
    let mean = trace / 3.0;
    let dev = [
        f64::from(eps.x) - mean,
        f64::from(eps.y) - mean,
        f64::from(eps.z) - mean,
    ];
    let dev_norm = (dev[0] * dev[0] + dev[1] * dev[1] + dev[2] * dev[2]).sqrt();

    let mu = f64::from(lame.mu);
    let two_mu = 2.0 * mu;
    let bulk = f64::from(lame.lambda) + two_mu / 3.0;
    let m = f64::from(model.slope_m);
    let m2 = m * m;

    // Trial invariants (compression-positive mean, non-negative shear).
    let p_tr = -bulk * trace;
    let q_tr = two_mu * (1.5_f64).sqrt() * dev_norm;
    let p_c0 = f64::from(state.pre_consolidation);

    let y_tr = q_tr * q_tr / m2 + p_tr * (p_tr - p_c0);
    if y_tr <= 0.0 {
        // Inside (or on) the ellipse: fully elastic, no hardening.
        return CamClayStep {
            elastic_gradient: fe_trial,
            plastic_increment: 0.0,
            yielded: false,
            pre_consolidation: state.pre_consolidation,
        };
    }

    // Fully-coupled implicit return: outer bisection on the plastic multiplier
    // Δγ, with the pre-consolidation pressure slaved to Δγ by an inner
    // bisection (hardening is anchored to the *original* p_c, never compounded).
    let hardening = f64::from(model.hardening_modulus);
    let solve = CamClaySolve {
        p_tr,
        q_tr,
        p_c0,
        bulk,
        mu,
        m2,
        hardening,
    };
    let (p, q, p_c) = solve.run();

    // Rebuild the returned elastic principal strain from (p, q).
    // Deviatoric unit direction is inherited from the trial predictor.
    let inv_two_mu = 1.0 / two_mu;
    let mean_e = -p / (3.0 * bulk);
    let s_scale = if dev_norm > MIN_DEVIATORIC_NORM {
        (2.0_f64 / 3.0).sqrt() * q / dev_norm
    } else {
        0.0
    };
    let eps_e = [
        (dev[0] * s_scale) * inv_two_mu + mean_e,
        (dev[1] * s_scale) * inv_two_mu + mean_e,
        (dev[2] * s_scale) * inv_two_mu + mean_e,
    ];

    let sigma_elastic = Vec3::new(
        signs.x * stretch_from_log_f64(eps_e[0]),
        signs.y * stretch_from_log_f64(eps_e[1]),
        signs.z * stretch_from_log_f64(eps_e[2]),
    );
    let fe_new = svd.u * Mat3::from_diagonal(sigma_elastic) * svd.v.transpose();

    let increment = {
        let dx = f64::from(eps.x) - eps_e[0];
        let dy = f64::from(eps.y) - eps_e[1];
        let dz = f64::from(eps.z) - eps_e[2];
        (dx * dx + dy * dy + dz * dz).sqrt() as f32
    };

    state.plastic_gradient = fe_new.inverse() * f_total;
    state.pre_consolidation = p_c as f32;

    CamClayStep {
        elastic_gradient: fe_new,
        plastic_increment: increment,
        yielded: true,
        pre_consolidation: state.pre_consolidation,
    }
}

/// Parameters of one modified Cam-Clay return solve, bundled so the nested
/// bisections can share them without threading a long argument list.
struct CamClaySolve {
    p_tr: f64,
    q_tr: f64,
    p_c0: f64,
    bulk: f64,
    mu: f64,
    m2: f64,
    hardening: f64,
}

impl CamClaySolve {
    /// Solves the coupled system `{ p(Δγ), q(Δγ), p_c(Δγ), y = 0 }` and returns
    /// the returned invariants `(p, q, p_c)` on the (hardened) ellipse.
    ///
    /// The outer unknown is the plastic multiplier `Δγ`. The yield residual
    /// `Y(Δγ)` satisfies `Y(0) = y_tr > 0` (callers only enter after detecting
    /// yield) and decreases through zero as `Δγ` grows, so a sign-change bracket
    /// `[0, hi]` is found by doubling and refined by bisection — no derivatives,
    /// unconditional convergence.
    fn run(&self) -> (f64, f64, f64) {
        // Expand an upper Δγ bound until the yield residual turns non-positive.
        let mut hi = 1.0 / self.bulk.max(self.mu).max(1.0);
        let mut bracketed = false;
        for _ in 0..BRACKET_ITERS {
            if self.residual(hi) <= 0.0 {
                bracketed = true;
                break;
            }
            hi *= 2.0;
        }
        if !bracketed {
            // Degenerate scaling: fall back to the asymptotic return.
            return self.invariants(hi);
        }

        let mut lo = 0.0_f64;
        for _ in 0..INNER_ITERS {
            let mid = 0.5 * (lo + hi);
            if self.residual(mid) > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        self.invariants(0.5 * (lo + hi))
    }

    /// Yield residual `Y(Δγ) = q²/M² + p·(p − p_c)` with `p_c` slaved to `Δγ`.
    #[inline]
    fn residual(&self, dg: f64) -> f64 {
        let (p, q, p_c) = self.invariants(dg);
        q * q / self.m2 + p * (p - p_c)
    }

    /// Returned `(p, q, p_c)` for a given `Δγ`, with `p_c` solved from the
    /// exponential hardening law anchored to the original `p_c0`.
    #[inline]
    fn invariants(&self, dg: f64) -> (f64, f64, f64) {
        let p_c = self.consistent_pc(dg);
        let (p, q) = self.returned_pq(dg, p_c);
        (p, q, p_c)
    }

    /// Closed-form predictor-corrector amplitudes for a given `Δγ` and `p_c`.
    #[inline]
    fn returned_pq(&self, dg: f64, p_c: f64) -> (f64, f64) {
        let p = (self.p_tr + self.bulk * dg * p_c) / (1.0 + 2.0 * self.bulk * dg);
        let q = self.q_tr / (1.0 + 6.0 * self.mu * dg / self.m2);
        (p, q)
    }

    /// Pre-consolidation pressure consistent with a given `Δγ`, i.e. the root of
    /// `h(p_c) = p_c0·exp(Δγ·(2p(p_c) − p_c)/H) − p_c`.
    ///
    /// `h` is strictly decreasing (`p` grows sub-linearly in `p_c`, so the
    /// exponent falls while `−p_c` falls), with `h(0⁺) > 0` and `h(∞) < 0`, so
    /// the root is bracketed and bisected. Anchoring to `p_c0` (not the previous
    /// iterate) is what prevents the hardening blow-up of a naive fixed point.
    #[inline]
    fn consistent_pc(&self, dg: f64) -> f64 {
        let h = |p_c: f64| -> f64 {
            let (p, _) = self.returned_pq(dg, p_c);
            let vol_plastic = dg * (2.0 * p - p_c);
            self.p_c0 * (vol_plastic / self.hardening).exp() - p_c
        };
        // h(0) > 0; expand an upper bound until h turns negative.
        let mut hi = self.p_c0;
        let mut bracketed = false;
        for _ in 0..BRACKET_ITERS {
            if h(hi) <= 0.0 {
                bracketed = true;
                break;
            }
            hi *= 2.0;
        }
        if !bracketed {
            return hi;
        }
        let mut lo = 0.0_f64;
        for _ in 0..PC_ITERS {
            let mid = 0.5 * (lo + hi);
            if h(mid) > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }
}

/// Principal Hencky strain `ln|σ|` of a (clamped) signed singular value,
/// evaluated in `f64` for deterministic libm-free precision then narrowed.
#[inline]
fn hencky(sigma: f32) -> f32 {
    (f64::from(sigma.abs().max(MIN_STRETCH)).ln()) as f32
}

/// Inverse of [`hencky`] operating directly on an `f64` log strain: the stretch
/// magnitude `exp(ε)`.
#[inline]
fn stretch_from_log_f64(eps: f64) -> f32 {
    eps.exp() as f32
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

    fn model() -> CamClayModel {
        CamClayModel::new(1.2, 0.05, 5.0e4).unwrap()
    }

    /// Recovers the (p, q, y) invariants of an elastic gradient under the linear
    /// Hencky law, for a given pre-consolidation pressure.
    fn invariants(fe: Mat3, lame: &LameParameters, p_c: f64, m: f64) -> (f64, f64, f64) {
        let svd = svd3(fe);
        let eps = [
            f64::from(hencky(svd.sigma.x)),
            f64::from(hencky(svd.sigma.y)),
            f64::from(hencky(svd.sigma.z)),
        ];
        let trace = eps[0] + eps[1] + eps[2];
        let mean = trace / 3.0;
        let dev = [eps[0] - mean, eps[1] - mean, eps[2] - mean];
        let dev_norm = (dev[0] * dev[0] + dev[1] * dev[1] + dev[2] * dev[2]).sqrt();
        let mu = f64::from(lame.mu);
        let bulk = f64::from(lame.lambda) + 2.0 * mu / 3.0;
        let p = -bulk * trace;
        let q = 2.0 * mu * (1.5_f64).sqrt() * dev_norm;
        let y = q * q / (m * m) + p * (p - p_c);
        (p, q, y)
    }

    #[test]
    fn model_validates_ranges() {
        assert!(CamClayModel::new(1.2, 0.05, 5.0e4).is_some());
        assert!(CamClayModel::new(0.0, 0.05, 5.0e4).is_none());
        assert!(CamClayModel::new(1.2, 0.0, 5.0e4).is_none());
        assert!(CamClayModel::new(1.2, 0.05, 0.0).is_none());
        assert!(CamClayModel::new(f32::NAN, 0.05, 5.0e4).is_none());
        assert!(CamClayModel::new(1.2, f32::INFINITY, 5.0e4).is_none());
        assert!(CamClayModel::new(-1.0, 0.05, 5.0e4).is_none());
    }

    #[test]
    fn state_validates_and_rest_is_identity() {
        assert!(CamClayState::new(0.0).is_none());
        assert!(CamClayState::new(-1.0).is_none());
        assert!(CamClayState::new(f32::NAN).is_none());
        let s = CamClayState::new(5.0e4).unwrap();
        assert_eq!(s.plastic_gradient(), Mat3::IDENTITY);
        assert_eq!(s.pre_consolidation(), 5.0e4);
        let r = model().rest_state();
        assert_eq!(r.plastic_gradient(), Mat3::IDENTITY);
        assert_eq!(r.pre_consolidation(), model().initial_pre_consolidation());
    }

    #[test]
    fn inside_ellipse_is_elastic() {
        let model = model();
        let mut state = model.rest_state();
        // Mild compression: p well inside [0, p_c], negligible shear.
        let f = Mat3::from_diagonal(Vec3::splat(0.99));
        let step = return_map_camclay(f, &lame(), &model, &mut state);
        assert!(!step.yielded);
        assert_eq!(step.plastic_increment, 0.0);
        assert_eq!(step.elastic_gradient, f);
        assert_eq!(state.plastic_gradient(), Mat3::IDENTITY);
        assert_eq!(state.pre_consolidation(), model.initial_pre_consolidation());
    }

    #[test]
    fn compression_past_cap_yields_and_hardens() {
        let model = model();
        let mut state = model.rest_state();
        let pc_before = state.pre_consolidation();
        // Strong uniform compression: p_tr >> p_c, past the cap.
        let f = Mat3::from_diagonal(Vec3::splat(0.9));
        let step = return_map_camclay(f, &lame(), &model, &mut state);
        assert!(step.yielded);
        assert!(step.plastic_increment > 0.0);
        // Hardening: compaction grows the pre-consolidation pressure.
        assert!(step.pre_consolidation > pc_before);
        assert_eq!(state.pre_consolidation(), step.pre_consolidation);
        // Returned state lies on (not outside) the updated ellipse.
        let (_, _, y) = invariants(
            step.elastic_gradient,
            &lame(),
            f64::from(step.pre_consolidation),
            f64::from(model.slope_m()),
        );
        let scale = f64::from(step.pre_consolidation).powi(2);
        assert!(y.abs() <= 1e-4 * scale, "post-return y = {y}");
    }

    #[test]
    fn shear_past_ellipse_returns_to_surface() {
        let model = model();
        let mut state = model.rest_state();
        // Volume-preserving simple shear (det = 1) with large amplitude.
        let f = Mat3::from_cols(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.4, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let step = return_map_camclay(f, &lame(), &model, &mut state);
        assert!(step.yielded);
        let (_, q, y) = invariants(
            step.elastic_gradient,
            &lame(),
            f64::from(step.pre_consolidation),
            f64::from(model.slope_m()),
        );
        let scale = f64::from(step.pre_consolidation).powi(2);
        assert!(y.abs() <= 1e-4 * scale, "post-return y = {y}");
        assert!(q > 0.0);
    }

    #[test]
    fn tension_yields_toward_tip() {
        let model = model();
        let mut state = model.rest_state();
        // Uniform expansion: trε > 0 ⇒ p_tr < 0 ⇒ outside ellipse (tension).
        let f = Mat3::from_diagonal(Vec3::splat(1.1));
        let step = return_map_camclay(f, &lame(), &model, &mut state);
        assert!(step.yielded);
        let (p, _, y) = invariants(
            step.elastic_gradient,
            &lame(),
            f64::from(step.pre_consolidation),
            f64::from(model.slope_m()),
        );
        // Returned toward the tension tip p = 0 and back onto the surface.
        assert!(p.abs() <= 1e-2 * f64::from(model.initial_pre_consolidation()));
        let scale = f64::from(model.initial_pre_consolidation()).powi(2);
        assert!(y.abs() <= 1e-4 * scale, "post-return y = {y}");
    }

    #[test]
    fn reconstructs_total_gradient() {
        let model = model();
        let mut state = model.rest_state();
        let f = Mat3::from_cols(
            Vec3::new(0.92, 0.03, 0.0),
            Vec3::new(0.0, 0.9, 0.05),
            Vec3::new(0.02, 0.0, 0.91),
        );
        let step = return_map_camclay(f, &lame(), &model, &mut state);
        assert!(step.yielded);
        let reconstructed = step.elastic_gradient * state.plastic_gradient();
        let diff = reconstructed - f;
        let frob = (0..3)
            .map(|i| diff.col(i).length_squared())
            .sum::<f32>()
            .sqrt();
        assert!(frob < 1e-3, "||Fe·Fp − F|| = {frob}");
    }

    #[test]
    fn is_deterministic() {
        let model = model();
        let f = Mat3::from_diagonal(Vec3::splat(0.9));
        let mut a = model.rest_state();
        let mut b = model.rest_state();
        let sa = return_map_camclay(f, &lame(), &model, &mut a);
        let sb = return_map_camclay(f, &lame(), &model, &mut b);
        assert_eq!(sa, sb);
        assert_eq!(a, b);
    }
}
