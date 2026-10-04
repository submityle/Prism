//! Perzyna rate-dependent (viscoplastic) overstress for tetrahedral FEM.
//!
//! Rate-independent plasticity ([`tet_fem_plasticity`](super::tet_fem_plasticity))
//! snaps the trial stress back onto the yield surface instantaneously: the
//! amount of plastic flow in a step is independent of how long that step took.
//! Real materials — polymers, metals near melt, soils, biological tissue —
//! flow *gradually*. The stress is allowed to sit **above** the yield surface
//! for a while (an *overstress*), and plastic strain accumulates at a rate that
//! grows with that overstress and with the elapsed time.
//!
//! This module implements the classical **Perzyna** overstress model on top of
//! the same finite-strain, Hencky-strain radial-return machinery used by the
//! inviscid J2 kernel. For a deviatoric trial strain of magnitude `ξ` and an
//! effective yield radius `R = yield + hardening·accumulated`, we look for the
//! returned radius `s ∈ [R, ξ]` that balances the plastic strain consumed this
//! step against the Perzyna flow rate:
//!
//! ```text
//!   ξ − s  =  (Δt / η) · ((s − R) / R)^N
//!   └ consumed ┘        └ Perzyna viscoplastic rate ┘
//! ```
//!
//! * `s − R ≥ 0` is the **viscous overstress** (how far the stress sits above
//!   the yield surface at the end of the step),
//! * `ξ − s ≥ 0` is the **plastic strain increment** actually consumed,
//! * `η > 0` is a relaxation time constant (so `Δt / η` is dimensionless), and
//! * `N > 0` is the rate-sensitivity exponent.
//!
//! The right-hand side is zero at `s = R` and strictly increasing in `s`, while
//! the left-hand side `ξ − s` is strictly decreasing; the balance has a unique
//! root in `(R, ξ]`, found with a safeguarded Newton / bisection solve. The
//! known limits all fall out of this one equation:
//!
//! | regime | behaviour |
//! |---|---|
//! | `η → 0` (or `Δt → ∞`) | `s → R`: recovers rate-independent return mapping |
//! | `η → ∞` (or `Δt → 0`) | `s → ξ`: fully elastic, stress stays above yield |
//! | `N = 1` | closed form `s = R·(ξ + Δt/η) / (R + Δt/η)` |
//!
//! As with the inviscid kernel the plastic flow is **isochoric** — the returned
//! strain scales only the deviatoric part, so `det Fₚ` is preserved — matching
//! the J2 assumption that hydrostatic pressure causes no yielding.
//!
//! This is a self-contained, per-element, integrator-agnostic kernel: feed it
//! the total deformation gradient and the time step, get back the elastic
//! gradient for the stress routine, and keep the mutated [`PerzynaState`].
//!
//! # Attribution
//!
//! Clean-room implementation of the textbook Perzyna viscoplastic overstress
//! model (Simo & Hughes, *Computational Inelasticity*) in Hencky-strain space
//! over the crate's own [`svd3`]. No Unreal Engine source or derived code.

use crate::mpm::svd3;
use glam::{Mat3, Vec3};

/// Principal stretches are clamped to this magnitude before taking a log, so an
/// inverted or collapsed predictor never produces a non-finite strain.
const MIN_STRETCH: f32 = 1e-4;

/// Deviatoric strains with Frobenius norm at or below this are treated as
/// purely hydrostatic (no yield direction), skipping plastic flow.
const MIN_DEVIATORIC_NORM: f32 = 1e-9;

/// Material parameters for Perzyna rate-dependent J2 viscoplasticity.
///
/// The yield surface lives in deviatoric Hencky-strain space (same convention
/// as [`PlasticModel`](super::tet_fem_plasticity::PlasticModel)); viscosity and
/// the rate exponent control *how fast* the overstress relaxes onto it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerzynaModel {
    /// Deviatoric Hencky-strain magnitude at which plastic flow begins
    /// (`‖dev ε‖`, dimensionless). Larger == stiffer / later yielding.
    yield_strain: f32,
    /// Linear isotropic hardening coefficient `≥ 0`: the effective yield grows
    /// as `yield_strain + hardening · accumulated_plastic_strain`.
    hardening: f32,
    /// Relaxation time constant `η > 0`. The step uses the dimensionless ratio
    /// `Δt / η`; smaller `η` means faster relaxation toward the yield surface
    /// (more rate-independent), larger `η` means a stiffer, more elastic
    /// response within a step.
    viscosity: f32,
    /// Rate-sensitivity exponent `N > 0`. `N = 1` is the linear (Bingham-like)
    /// closed form; larger `N` makes the flow rate rise more sharply with
    /// overstress.
    rate_exponent: f32,
}

impl PerzynaModel {
    /// Builds a model, validating that `yield_strain > 0`, `hardening ≥ 0`,
    /// `viscosity > 0`, and `rate_exponent > 0`, all finite.
    #[must_use]
    pub fn new(
        yield_strain: f32,
        hardening: f32,
        viscosity: f32,
        rate_exponent: f32,
    ) -> Option<Self> {
        if !yield_strain.is_finite() || yield_strain <= 0.0 {
            return None;
        }
        if !hardening.is_finite() || hardening < 0.0 {
            return None;
        }
        if !viscosity.is_finite() || viscosity <= 0.0 {
            return None;
        }
        if !rate_exponent.is_finite() || rate_exponent <= 0.0 {
            return None;
        }
        Some(Self {
            yield_strain,
            hardening,
            viscosity,
            rate_exponent,
        })
    }

    /// Base yield strain (before hardening).
    #[must_use]
    pub fn yield_strain(&self) -> f32 {
        self.yield_strain
    }

    /// Hardening coefficient.
    #[must_use]
    pub fn hardening(&self) -> f32 {
        self.hardening
    }

    /// Relaxation time constant `η`.
    #[must_use]
    pub fn viscosity(&self) -> f32 {
        self.viscosity
    }

    /// Rate-sensitivity exponent `N`.
    #[must_use]
    pub fn rate_exponent(&self) -> f32 {
        self.rate_exponent
    }

    /// Effective yield strain after `accumulated` plastic strain.
    #[must_use]
    pub fn effective_yield(&self, accumulated: f32) -> f32 {
        self.yield_strain + self.hardening * accumulated.max(0.0)
    }
}

/// Persistent per-element viscoplastic state carried across simulation steps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerzynaState {
    /// Plastic part of the deformation gradient `Fₚ`.
    plastic_gradient: Mat3,
    /// Accumulated equivalent plastic strain (monotonically non-decreasing),
    /// driving isotropic hardening.
    accumulated_strain: f32,
}

impl PerzynaState {
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

    /// Accumulated equivalent plastic strain so far.
    #[must_use]
    pub fn accumulated_strain(&self) -> f32 {
        self.accumulated_strain
    }
}

impl Default for PerzynaState {
    fn default() -> Self {
        Self::rest()
    }
}

/// Outcome of a single Perzyna viscoplastic return-mapping step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerzynaStep {
    /// Elastic deformation gradient `Fₑ` to feed the stress routine.
    pub elastic_gradient: Mat3,
    /// Equivalent plastic strain consumed this step (`ξ − s`, `0` when elastic).
    pub plastic_increment: f32,
    /// Viscous overstress remaining at the end of the step (`s − R ≥ 0`): how
    /// far the stress still sits above the yield surface.
    pub viscous_overstress: f32,
    /// `true` when the predictor exceeded the yield surface and flowed.
    pub yielded: bool,
}

/// Performs the Perzyna rate-dependent elastic-predictor / viscoplastic-return
/// update for one element over a time step `dt`.
///
/// Given the total deformation gradient `f_total`, the material `model`, the
/// step `dt`, and the carried [`PerzynaState`], returns the elastic gradient to
/// use for stress and mutates `state` (updating `Fₚ` and accumulated strain
/// when the material flows).
///
/// A non-positive or non-finite `dt` is treated as an instantaneous (elastic)
/// probe: no plastic flow occurs and the trial elastic gradient is returned.
/// The returned elastic gradient always satisfies `f_total ≈ Fₑ · Fₚ` with the
/// updated `Fₚ`, and plastic flow preserves `det Fₚ`.
#[must_use]
pub fn return_map_perzyna(
    f_total: Mat3,
    model: &PerzynaModel,
    dt: f32,
    state: &mut PerzynaState,
) -> PerzynaStep {
    // Elastic predictor: strip the stored plastic deformation.
    let fp_inv = state.plastic_gradient.inverse();
    let fe_trial = f_total * fp_inv;

    let svd = svd3(fe_trial);
    let sigma = svd.sigma;
    let signs = Vec3::new(sign_of(sigma.x), sign_of(sigma.y), sign_of(sigma.z));
    let eps = Vec3::new(hencky(sigma.x), hencky(sigma.y), hencky(sigma.z));

    let mean = (eps.x + eps.y + eps.z) / 3.0;
    let dev = eps - Vec3::splat(mean);
    let xi = dev.length();

    let yield_eff = model.effective_yield(state.accumulated_strain);

    // Inside the yield surface, no deviatoric direction, or a degenerate step:
    // fully elastic, zero overstress.
    let elastic_step = PerzynaStep {
        elastic_gradient: fe_trial,
        plastic_increment: 0.0,
        viscous_overstress: 0.0,
        yielded: false,
    };
    if xi <= MIN_DEVIATORIC_NORM || xi <= yield_eff || !dt.is_finite() || dt <= 0.0 {
        return elastic_step;
    }

    // Solve the Perzyna balance for the returned radius s ∈ [R, ξ] in f64.
    let relax = f64::from(dt) / f64::from(model.viscosity);
    let s = solve_returned_radius(
        f64::from(xi),
        f64::from(yield_eff),
        relax,
        f64::from(model.rate_exponent),
    );
    let s = s as f32;

    let increment = (xi - s).max(0.0);
    let viscous_overstress = (s - yield_eff).max(0.0);

    // Scale only the deviatoric part back to radius s (isochoric flow).
    let dev_returned = dev * (s / xi);
    let eps_elastic = Vec3::splat(mean) + dev_returned;

    let sigma_elastic = Vec3::new(
        signs.x * stretch_from_log(eps_elastic.x),
        signs.y * stretch_from_log(eps_elastic.y),
        signs.z * stretch_from_log(eps_elastic.z),
    );
    let fe_new = svd.u * Mat3::from_diagonal(sigma_elastic) * svd.v.transpose();

    // Fold the consumed deformation into the plastic gradient: Fₚ = Fₑ⁻¹ · F.
    state.plastic_gradient = fe_new.inverse() * f_total;
    state.accumulated_strain += increment;

    PerzynaStep {
        elastic_gradient: fe_new,
        plastic_increment: increment,
        viscous_overstress,
        yielded: true,
    }
}

/// Solves `g(s) = (ξ − s) − relax·((s − R)/R)^N = 0` for `s ∈ [R, ξ]`.
///
/// `g` is continuous and strictly decreasing on `[R, ξ]` with `g(R) = ξ−R > 0`
/// and `g(ξ) = −relax·((ξ−R)/R)^N < 0`, so the root is unique. A safeguarded
/// Newton iteration (falling back to bisection whenever a step would leave the
/// bracket or stall) converges robustly even for stiff `relax` or `N < 1`,
/// where the derivative blows up near `s = R`. All math is in `f64`.
fn solve_returned_radius(xi: f64, r: f64, relax: f64, n: f64) -> f64 {
    // Degenerate guard: without a positive yield radius the normalized
    // overstress is undefined; treat as rate-independent (snap to yield).
    if r <= 0.0 {
        return r.max(0.0).min(xi);
    }

    let mut lo = r; // g(lo) >= 0
    let mut hi = xi; // g(hi) <= 0
    let mut s = 0.5 * (lo + hi);

    for _ in 0..80 {
        let t = ((s - r) / r).max(0.0);
        let pow = t.powf(n);
        let g = (xi - s) - relax * pow;

        if g > 0.0 {
            lo = s;
        } else {
            hi = s;
        }

        // g'(s) = −1 − relax·N·t^(N−1)/R. Guard the t→0 singularity for N<1.
        let d_pow = if t > 0.0 {
            n * t.powf(n - 1.0) / r
        } else {
            0.0
        };
        let g_prime = -1.0 - relax * d_pow;

        let mut next = if g_prime.is_finite() && g_prime.abs() > f64::EPSILON {
            s - g / g_prime
        } else {
            0.5 * (lo + hi)
        };
        // Safeguard: keep the iterate strictly inside the bracket.
        if !(next > lo && next < hi) {
            next = 0.5 * (lo + hi);
        }

        if (next - s).abs() <= 1e-12 * (1.0 + s.abs()) {
            return next.clamp(r, xi);
        }
        s = next;
    }

    s.clamp(r, xi)
}

/// Principal Hencky strain `ln|σ|` of a (clamped) signed singular value,
/// evaluated in `f64` for deterministic libm-free precision then narrowed.
#[inline]
fn hencky(sigma: f32) -> f32 {
    (f64::from(sigma.abs().max(MIN_STRETCH)).ln()) as f32
}

/// Inverse of [`hencky`]: the stretch magnitude `exp(ε)` for an elastic log
/// strain, evaluated in `f64` for precision.
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
    use crate::collider::tet_fem_plasticity::{return_map, PlasticModel, PlasticState};

    fn approx_mat(a: Mat3, b: Mat3, tol: f32) -> bool {
        (0..3).all(|c| (a.col(c) - b.col(c)).length() < tol)
    }

    /// Diagonal stretch matrix from principal log strains (`σ = exp(ε)`).
    fn from_log_strain(e: Vec3) -> Mat3 {
        Mat3::from_diagonal(Vec3::new(
            (f64::from(e.x).exp()) as f32,
            (f64::from(e.y).exp()) as f32,
            (f64::from(e.z).exp()) as f32,
        ))
    }

    /// Deviatoric Hencky-strain norm `ξ` of a deformation gradient.
    fn deviatoric_norm(f: Mat3) -> f32 {
        let s = svd3(f).sigma;
        let eps = Vec3::new(hencky(s.x), hencky(s.y), hencky(s.z));
        let mean = (eps.x + eps.y + eps.z) / 3.0;
        (eps - Vec3::splat(mean)).length()
    }

    #[test]
    fn rejects_invalid_parameters() {
        assert!(PerzynaModel::new(0.1, 0.0, 1.0, 1.0).is_some());
        assert!(
            PerzynaModel::new(0.0, 0.0, 1.0, 1.0).is_none(),
            "zero yield"
        );
        assert!(PerzynaModel::new(-0.1, 0.0, 1.0, 1.0).is_none());
        assert!(
            PerzynaModel::new(0.1, -1.0, 1.0, 1.0).is_none(),
            "neg hardening"
        );
        assert!(
            PerzynaModel::new(0.1, 0.0, 0.0, 1.0).is_none(),
            "zero viscosity"
        );
        assert!(PerzynaModel::new(0.1, 0.0, -1.0, 1.0).is_none());
        assert!(
            PerzynaModel::new(0.1, 0.0, 1.0, 0.0).is_none(),
            "zero exponent"
        );
        assert!(PerzynaModel::new(0.1, 0.0, 1.0, -2.0).is_none());
        assert!(PerzynaModel::new(f32::NAN, 0.0, 1.0, 1.0).is_none());
        assert!(PerzynaModel::new(0.1, 0.0, f32::INFINITY, 1.0).is_none());
    }

    #[test]
    fn rest_state_is_identity() {
        let s = PerzynaState::rest();
        assert_eq!(s.plastic_gradient(), Mat3::IDENTITY);
        assert_eq!(s.accumulated_strain(), 0.0);
        assert_eq!(PerzynaState::default(), s);
    }

    #[test]
    fn rest_pose_is_elastic() {
        let model = PerzynaModel::new(0.1, 0.0, 1.0, 1.0).unwrap();
        let mut state = PerzynaState::rest();
        let step = return_map_perzyna(Mat3::IDENTITY, &model, 1.0, &mut state);
        assert!(!step.yielded);
        assert_eq!(step.plastic_increment, 0.0);
        assert_eq!(step.viscous_overstress, 0.0);
        assert_eq!(state.plastic_gradient(), Mat3::IDENTITY);
    }

    #[test]
    fn small_strain_stays_elastic() {
        let model = PerzynaModel::new(0.2, 0.0, 1.0, 1.0).unwrap();
        let mut state = PerzynaState::rest();
        let f = from_log_strain(Vec3::new(0.05, 0.0, -0.05)); // ξ ≈ 0.0707 < 0.2
        let step = return_map_perzyna(f, &model, 1.0, &mut state);
        assert!(!step.yielded);
        assert_eq!(step.plastic_increment, 0.0);
        assert!(approx_mat(step.elastic_gradient, f, 1e-6));
        assert_eq!(state.plastic_gradient(), Mat3::IDENTITY, "Fp untouched");
    }

    #[test]
    fn non_positive_timestep_is_elastic() {
        let model = PerzynaModel::new(0.05, 0.0, 1.0, 1.0).unwrap();
        let f = from_log_strain(Vec3::new(0.1, 0.0, -0.1)); // ξ ≈ 0.141 > 0.05
        for dt in [0.0_f32, -1.0, f32::NAN, f32::INFINITY] {
            let mut state = PerzynaState::rest();
            let step = return_map_perzyna(f, &model, dt, &mut state);
            assert!(!step.yielded, "dt={dt} must be elastic");
            assert_eq!(step.plastic_increment, 0.0);
            assert_eq!(state.accumulated_strain(), 0.0);
        }
    }

    #[test]
    fn inviscid_limit_matches_rate_independent() {
        // η → 0 (huge relax) with dt = 1 recovers the rate-independent return.
        let yield_strain = 0.05_f32;
        let perzyna = PerzynaModel::new(yield_strain, 0.0, 1e-8, 1.0).unwrap();
        let inviscid = PlasticModel::new(yield_strain, 0.0).unwrap();

        let f = from_log_strain(Vec3::new(0.18, -0.02, -0.16));

        let mut ps = PerzynaState::rest();
        let p_step = return_map_perzyna(f, &perzyna, 1.0, &mut ps);

        let mut is = PlasticState::rest();
        let i_step = return_map(f, &inviscid, &mut is);

        assert!(p_step.yielded && i_step.yielded);
        assert!(
            (p_step.plastic_increment - i_step.plastic_increment).abs() < 1e-4,
            "perzyna {} vs inviscid {}",
            p_step.plastic_increment,
            i_step.plastic_increment
        );
        assert!(
            approx_mat(p_step.elastic_gradient, i_step.elastic_gradient, 1e-4),
            "elastic gradients should match in the inviscid limit"
        );
        assert!(p_step.viscous_overstress < 1e-4, "overstress vanishes");
    }

    #[test]
    fn high_viscosity_suppresses_flow() {
        let f = from_log_strain(Vec3::new(0.18, -0.02, -0.16));
        let low = PerzynaModel::new(0.05, 0.0, 0.1, 1.0).unwrap();
        let high = PerzynaModel::new(0.05, 0.0, 100.0, 1.0).unwrap();

        let mut sl = PerzynaState::rest();
        let step_low = return_map_perzyna(f, &low, 1.0, &mut sl);
        let mut sh = PerzynaState::rest();
        let step_high = return_map_perzyna(f, &high, 1.0, &mut sh);

        assert!(
            step_high.plastic_increment < step_low.plastic_increment,
            "stiffer (higher η) material flows less: {} !< {}",
            step_high.plastic_increment,
            step_low.plastic_increment
        );
    }

    #[test]
    fn shorter_timestep_reduces_flow() {
        let model = PerzynaModel::new(0.05, 0.0, 1.0, 2.0).unwrap();
        let f = from_log_strain(Vec3::new(0.18, -0.02, -0.16));

        let mut prev = f32::INFINITY;
        for dt in [1.0_f32, 0.5, 0.25, 0.1, 0.01] {
            let mut state = PerzynaState::rest();
            let step = return_map_perzyna(f, &model, dt, &mut state);
            assert!(
                step.plastic_increment < prev,
                "increment must decrease with dt: dt={dt} gave {} !< {}",
                step.plastic_increment,
                prev
            );
            prev = step.plastic_increment;
        }
    }

    #[test]
    fn linear_exponent_matches_closed_form() {
        // N = 1: s = R·(ξ + relax) / (R + relax), relax = dt/η.
        let (yield_strain, eta, dt) = (0.05_f32, 2.0_f32, 0.5_f32);
        let model = PerzynaModel::new(yield_strain, 0.0, eta, 1.0).unwrap();
        let f = from_log_strain(Vec3::new(0.18, -0.02, -0.16));
        let xi = deviatoric_norm(f);

        let mut state = PerzynaState::rest();
        let step = return_map_perzyna(f, &model, dt, &mut state);

        let relax = dt / eta;
        let s_closed = yield_strain * (xi + relax) / (yield_strain + relax);
        let increment_closed = xi - s_closed;

        assert!(
            (step.plastic_increment - increment_closed).abs() < 1e-4,
            "N=1 increment {} vs closed form {}",
            step.plastic_increment,
            increment_closed
        );
        assert!((step.viscous_overstress - (s_closed - yield_strain)).abs() < 1e-4);
    }

    #[test]
    fn overstress_and_increment_partition_the_excess() {
        // s ∈ [R, ξ] ⇒ (ξ − s) + (s − R) = ξ − R, both terms non-negative.
        let model = PerzynaModel::new(0.05, 0.0, 1.5, 1.5).unwrap();
        let f = from_log_strain(Vec3::new(0.2, -0.03, -0.17));
        let xi = deviatoric_norm(f);

        let mut state = PerzynaState::rest();
        let step = return_map_perzyna(f, &model, 0.3, &mut state);

        assert!(step.plastic_increment >= 0.0);
        assert!(step.viscous_overstress >= 0.0);
        let sum = step.plastic_increment + step.viscous_overstress;
        assert!(
            (sum - (xi - 0.05)).abs() < 1e-4,
            "increment + overstress {} should equal ξ − R {}",
            sum,
            xi - 0.05
        );
    }

    #[test]
    fn reconstructs_total_gradient() {
        let model = PerzynaModel::new(0.05, 0.1, 1.0, 1.0).unwrap();
        let f = from_log_strain(Vec3::new(0.22, -0.04, -0.18));

        let mut state = PerzynaState::rest();
        let step = return_map_perzyna(f, &model, 0.4, &mut state);

        assert!(step.yielded);
        let recon = step.elastic_gradient * state.plastic_gradient();
        assert!(approx_mat(recon, f, 1e-4), "F = Fe * Fp must hold");
    }

    #[test]
    fn plastic_flow_is_volume_preserving() {
        let model = PerzynaModel::new(0.05, 0.0, 1.0, 1.0).unwrap();
        // Include a volumetric component to make the isochoric claim non-trivial.
        let f = from_log_strain(Vec3::new(0.25, 0.05, -0.1));

        let mut state = PerzynaState::rest();
        let step = return_map_perzyna(f, &model, 0.5, &mut state);
        assert!(step.yielded);

        let det_fp = state.plastic_gradient().determinant();
        assert!(
            (det_fp - 1.0).abs() < 1e-4,
            "isochoric plastic flow keeps det Fp == 1, got {det_fp}"
        );
    }

    #[test]
    fn is_deterministic() {
        let model = PerzynaModel::new(0.05, 0.05, 1.3, 1.7).unwrap();
        let f = from_log_strain(Vec3::new(0.2, -0.03, -0.17));

        let mut a = PerzynaState::rest();
        let step_a = return_map_perzyna(f, &model, 0.3, &mut a);
        let mut b = PerzynaState::rest();
        let step_b = return_map_perzyna(f, &model, 0.3, &mut b);

        assert_eq!(step_a, step_b);
        assert_eq!(a, b);
    }
}
