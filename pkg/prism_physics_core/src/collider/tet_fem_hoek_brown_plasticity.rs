//! Generalized Hoek–Brown finite-strain elastoplasticity for rock masses on a
//! tetrahedral `FEM` element, using an implicit return map in principal stress
//! space.
//!
//! Where [`tet_fem_mohr_coulomb_plasticity`](super::tet_fem_mohr_coulomb_plasticity)
//! bounds the shear strength by a *straight* line in the `(σ₁, σ₃)` plane, the
//! Hoek–Brown criterion that rock engineering calibrates against is *curved*:
//! confinement strengthens the rock nonlinearly. Working with the major and
//! minor principal Kirchhoff stresses `σ₁ ≥ σ₂ ≥ σ₃` (tension positive) coming
//! from the SVD of the trial elastic gradient (`F = Fₑ · Fₚ`), the generalized
//! criterion written in the rock-mechanics compressive-positive convention
//! `S₁ = −σ₃`, `S₃ = −σ₁` is
//!
//! ```text
//! S₁ = S₃ + σ_ci · (m_b · S₃ / σ_ci + s)^a
//! ```
//!
//! which, moved to our tension-positive stresses, becomes the yield function
//!
//! ```text
//! f(σ) = (σ₁ − σ₃) − σ_ci · B(σ₁)^a ,   B(σ₁) = s − m_b · σ₁ / σ_ci   (≤ 0 elastic)
//! ```
//!
//! The intermediate stress `σ₂` never enters, so — exactly like Mohr–Coulomb —
//! the surface is a faceted-but-curved pyramid with a main sextant, two edges
//! (triaxial compression `σ₂ → σ₃`, triaxial extension `σ₁ → σ₂`), and a single
//! tensile apex at `σ = s · σ_ci / m_b` where `B = 0`.
//!
//! Because `B(σ₁)^a` is nonlinear, each feature return is solved by a small
//! Newton–Raphson iteration in principal space rather than a closed-form 1×1 or
//! 2×2 solve. The unknowns are the returned principal stresses plus one plastic
//! multiplier per active surface; the isotropic elastic operator
//! `(D·m)ᵢ = λ·Σⱼ mⱼ + 2μ·mᵢ` couples them. Non-associated flow uses an
//! independent dilation parameter `m_g ≤ m_b` in the plastic potential's slope,
//! so dilatancy can be reduced below the (over-dilatant) associated value.
//!
//! The feature priority mirrors de Souza Neto / Clausen: attempt the main
//! surface, then the compression edge, the extension edge, and finally the
//! apex; the first candidate whose returned stresses stay ordered with
//! non-negative multipliers is accepted. Returned stresses map back to elastic
//! log-strains through `εᵢ = (σ′ᵢ − λ·trσ′/(3λ+2μ)) / (2μ)` and are
//! exponentiated to singular values to rebuild `Fₑ`.
//!
//! Clean-room implementation over the crate's own [`svd3`]; logs, exponentials,
//! and all power-law evaluations run in `f64` for deterministic, libm-free
//! precision. No Unreal Engine source or derived code.

use crate::collider::tet_fem_constitutive::LameParameters;
use crate::mpm::svd3;
use glam::{Mat3, Vec3};

/// Principal stretches are clamped to this magnitude before taking a log, so an
/// inverted or collapsed predictor never produces a non-finite strain.
const MIN_STRETCH: f32 = 1e-4;

/// A plastic multiplier is accepted when it is at least this (slightly
/// negative) value, absorbing round-off in the feature solves.
const MIN_MULTIPLIER: f32 = -1e-6;

/// The confinement bracket `B(σ₁)` is clamped to this small positive floor
/// before being raised to a (possibly fractional) power, so the curved term
/// and its derivatives stay finite right up to the tensile apex.
const MIN_BRACKET: f64 = 1e-9;

/// Maximum Newton iterations for a single feature return before the solver
/// gives up and the next feature (or the apex) is tried.
const MAX_NEWTON_ITERS: usize = 60;

/// Which feature of the Hoek–Brown surface the return map projected onto.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HoekBrownYield {
    /// Predictor was inside the surface: fully elastic.
    Elastic,
    /// Returned onto the smooth main sextant (the generic case).
    MainSurface,
    /// Returned onto the triaxial-compression edge (`σ₂ → σ₃`).
    CompressionEdge,
    /// Returned onto the triaxial-extension edge (`σ₁ → σ₂`).
    ExtensionEdge,
    /// Returned onto the tensile apex (hydrostatic tension, `B = 0`).
    Apex,
}

/// Material parameters for the generalized Hoek–Brown rock-mass criterion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HoekBrownModel {
    /// Uniaxial compressive strength of the intact rock `σ_ci > 0` (stress
    /// units).
    sigma_ci: f32,
    /// Reduced Hoek–Brown constant `m_b > 0` for the broken rock mass.
    m_b: f32,
    /// Rock-mass constant `s ∈ (0, 1]` (`s = 1` for intact rock).
    s: f32,
    /// Curvature exponent `a ∈ (0, 1]` (`a = 0.5` for the original criterion).
    a: f32,
    /// Dilation constant `m_g ∈ [0, m_b]` used in the plastic potential's slope
    /// for non-associated flow.
    m_g: f32,
    /// Linear hardening coefficient `≥ 0`: the effective `σ_ci` grows as
    /// `σ_ci + hardening · accumulated_plastic_strain`.
    hardening: f32,
}

impl HoekBrownModel {
    /// Builds a model from the intact strength `σ_ci > 0`, broken-mass constant
    /// `m_b > 0`, rock-mass constants `s ∈ (0, 1]` and `a ∈ (0, 1]`, a dilation
    /// constant `m_g ∈ [0, m_b]`, and a hardening coefficient `≥ 0`.
    ///
    /// Returns `None` if any input is non-finite or out of range. The dilation
    /// constant is capped at `m_b` so the plastic flow never over-dilates.
    #[must_use]
    pub fn new(sigma_ci: f32, m_b: f32, s: f32, a: f32, m_g: f32, hardening: f32) -> Option<Self> {
        if !(sigma_ci.is_finite() && sigma_ci > 0.0) {
            return None;
        }
        if !(m_b.is_finite() && m_b > 0.0) {
            return None;
        }
        if !(s.is_finite() && s > 0.0 && s <= 1.0) {
            return None;
        }
        if !(a.is_finite() && a > 0.0 && a <= 1.0) {
            return None;
        }
        if !(m_g.is_finite() && m_g >= 0.0 && m_g <= m_b) {
            return None;
        }
        if !(hardening.is_finite() && hardening >= 0.0) {
            return None;
        }
        Some(Self {
            sigma_ci,
            m_b,
            s,
            a,
            m_g,
            hardening,
        })
    }

    /// Convenience constructor for the *original* Hoek–Brown criterion
    /// (`s = 1`, `a = 0.5`) with associated flow (`m_g = m_b`) and no
    /// hardening.
    #[must_use]
    pub fn intact(sigma_ci: f32, m_b: f32) -> Option<Self> {
        Self::new(sigma_ci, m_b, 1.0, 0.5, m_b, 0.0)
    }

    /// Intact uniaxial compressive strength `σ_ci`.
    #[must_use]
    pub fn sigma_ci(&self) -> f32 {
        self.sigma_ci
    }

    /// Reduced broken-mass constant `m_b`.
    #[must_use]
    pub fn m_b(&self) -> f32 {
        self.m_b
    }

    /// Rock-mass constant `s`.
    #[must_use]
    pub fn s(&self) -> f32 {
        self.s
    }

    /// Curvature exponent `a`.
    #[must_use]
    pub fn a(&self) -> f32 {
        self.a
    }

    /// Dilation constant `m_g`.
    #[must_use]
    pub fn m_g(&self) -> f32 {
        self.m_g
    }

    /// Linear hardening coefficient.
    #[must_use]
    pub fn hardening(&self) -> f32 {
        self.hardening
    }

    /// Effective intact strength after `accumulated` plastic strain.
    #[must_use]
    pub fn effective_sigma_ci(&self, accumulated: f32) -> f32 {
        self.sigma_ci + self.hardening * accumulated.max(0.0)
    }
}

/// Persistent per-element Hoek–Brown plastic state carried across steps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HoekBrownState {
    plastic_gradient: Mat3,
    accumulated_strain: f32,
}

impl HoekBrownState {
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

impl Default for HoekBrownState {
    fn default() -> Self {
        Self::rest()
    }
}

/// Outcome of a single Hoek–Brown return-mapping step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HoekBrownStep {
    /// Elastic deformation gradient `Fₑ` to feed the stress routine.
    pub elastic_gradient: Mat3,
    /// Plastic strain consumed this step (`0` when still elastic).
    pub plastic_increment: f32,
    /// Which feature of the surface the return projected onto.
    pub mode: HoekBrownYield,
}

/// Internal helper bundling the derived scalars for the principal return map so
/// the Newton feature solves stay compact and parameter-light. All power-law
/// math is evaluated in `f64`.
struct PrincipalReturn<'a> {
    lame: &'a LameParameters,
    sigma_ci: f64,
    m_b: f64,
    s: f64,
    a: f64,
    m_g: f64,
    order_tol: f32,
}

impl PrincipalReturn<'_> {
    /// Isotropic elastic operator applied to a principal flow direction:
    /// `(D·m)ᵢ = λ·Σⱼ mⱼ + 2μ·mᵢ`.
    #[inline]
    fn d_times_m(&self, m: [f64; 3]) -> [f64; 3] {
        let sum = m[0] + m[1] + m[2];
        let lambda = f64::from(self.lame.lambda);
        let two_mu = 2.0 * f64::from(self.lame.mu);
        [
            lambda * sum + two_mu * m[0],
            lambda * sum + two_mu * m[1],
            lambda * sum + two_mu * m[2],
        ]
    }

    /// Yield bracket `B(σ₁) = s − m_b · σ₁ / σ_ci`, clamped to a small positive
    /// floor so the fractional power stays finite at and past the apex.
    #[inline]
    fn bracket(&self, sigma1: f64) -> f64 {
        (self.s - self.m_b * sigma1 / self.sigma_ci).max(MIN_BRACKET)
    }

    /// Dilation bracket `B_g(σ₁) = s − m_g · σ₁ / σ_ci` for the plastic
    /// potential, clamped likewise.
    #[inline]
    fn bracket_g(&self, sigma1: f64) -> f64 {
        (self.s - self.m_g * sigma1 / self.sigma_ci).max(MIN_BRACKET)
    }

    /// Unclamped yield bracket. When this is non-positive the predictor sits at
    /// or past the tensile apex (`σ₁ ≥ s·σ_ci/m_b`) and the only admissible
    /// return is the apex, independent of the clamped yield value.
    #[inline]
    fn raw_bracket(&self, sigma1: f64) -> f64 {
        self.s - self.m_b * sigma1 / self.sigma_ci
    }

    /// Yield value `f(σ) = (σ₁ − σ₃) − σ_ci · B(σ₁)^a` (tension positive).
    #[inline]
    fn yield_value(&self, s1: f64, s3: f64) -> f64 {
        (s1 - s3) - self.sigma_ci * self.bracket(s1).powf(self.a)
    }

    /// Yield derivative w.r.t. the major stress, `∂f/∂σ₁ = 1 + a·m_b·B^(a−1)`.
    #[inline]
    fn dyield_dmajor(&self, s1: f64) -> f64 {
        1.0 + self.a * self.m_b * self.bracket(s1).powf(self.a - 1.0)
    }

    /// Plastic-potential slope term on the major axis,
    /// `g'(σ₁) = a·m_g·B_g^(a−1)`, so the flow direction's major component is
    /// `1 + g'(σ₁)`.
    #[inline]
    fn flow_major(&self, s1: f64) -> f64 {
        self.a * self.m_g * self.bracket_g(s1).powf(self.a - 1.0)
    }

    /// Derivative of [`Self::flow_major`] w.r.t. `σ₁`:
    /// `−a·(a−1)·m_g² / σ_ci · B_g^(a−2)`.
    #[inline]
    fn dflow_major(&self, s1: f64) -> f64 {
        -self.a * (self.a - 1.0) * self.m_g * self.m_g / self.sigma_ci
            * self.bracket_g(s1).powf(self.a - 2.0)
    }

    /// Whether the returned principal stresses keep the required descending
    /// order within tolerance.
    #[inline]
    fn ordered(&self, s: [f64; 3]) -> bool {
        let tol = f64::from(self.order_tol);
        s[0] - s[1] >= -tol && s[1] - s[2] >= -tol
    }

    /// Newton return onto the smooth main sextant. Flow couples `σ₁` and `σ₃`
    /// (potential major slope on `σ₁`, `−1` on `σ₃`); `σ₂` only follows the
    /// volumetric part of the flow. Unknowns `(σ′₁, σ′₂, σ′₃, Δλ)`.
    fn main_surface(&self, trial: [f64; 3]) -> Option<([f64; 3], f64)> {
        let mut sig = trial;
        let mut dl = 0.0_f64;
        for _ in 0..MAX_NEWTON_ITERS {
            let g1 = self.flow_major(sig[0]);
            let m = [1.0 + g1, 0.0, -1.0];
            let dm = self.d_times_m(m);
            // Residuals: σ′ − σ_tr + Δλ·D·m = 0, and f(σ′) = 0.
            let r = [
                sig[0] - trial[0] + dl * dm[0],
                sig[1] - trial[1] + dl * dm[1],
                sig[2] - trial[2] + dl * dm[2],
                self.yield_value(sig[0], sig[2]),
            ];
            if max_abs4(r) < 1e-7 * self.residual_scale() {
                return self.accept_single(sig, dl);
            }
            // Jacobian rows (∂R/∂σ′₁, ∂σ′₂, ∂σ′₃, ∂Δλ).
            let lambda = f64::from(self.lame.lambda);
            let two_mu = 2.0 * f64::from(self.lame.mu);
            let dg1 = self.dflow_major(sig[0]);
            // d(D·m)/dσ′₁ with m = (1+g1, 0, -1): only m₀ varies.
            let ddm0 = (lambda + two_mu) * dg1;
            let ddm1 = lambda * dg1;
            let ddm2 = lambda * dg1;
            let df1 = self.dyield_dmajor(sig[0]);
            let jac = [
                [1.0 + dl * ddm0, 0.0, 0.0, dm[0]],
                [dl * ddm1, 1.0, 0.0, dm[1]],
                [dl * ddm2, 0.0, 1.0, dm[2]],
                [df1, 0.0, -1.0, 0.0],
            ];
            let delta = solve_linear::<4>(jac, r)?;
            sig[0] -= delta[0];
            sig[1] -= delta[1];
            sig[2] -= delta[2];
            dl -= delta[3];
        }
        None
    }

    /// Newton return onto a two-surface edge. `compression == true` activates
    /// the triaxial-compression edge where the second surface pairs `σ₁` with
    /// `σ₂`; otherwise the extension edge pairs `σ₂` with `σ₃`. Unknowns
    /// `(σ′₁, σ′₂, σ′₃, Δλₐ, Δλ_b)`.
    fn edge(&self, trial: [f64; 3], compression: bool) -> Option<([f64; 3], f64)> {
        let mut sig = trial;
        let mut la = 0.0_f64;
        let mut lb = 0.0_f64;
        for _ in 0..MAX_NEWTON_ITERS {
            // Surface a: f_a = (σ₁ − σ₃) − σ_ci·B(σ₁)^a, flow (1+g₁, 0, −1).
            let g1 = self.flow_major(sig[0]);
            let m_a = [1.0 + g1, 0.0, -1.0];
            let (m_b_dir, f_b, df_b, major_b_is_1) = if compression {
                // Surface b: f_b = (σ₁ − σ₂) − σ_ci·B(σ₁)^a, flow (1+g₁, −1, 0).
                (
                    [1.0 + g1, -1.0, 0.0],
                    self.yield_value(sig[0], sig[1]),
                    [self.dyield_dmajor(sig[0]), -1.0, 0.0],
                    true,
                )
            } else {
                // Surface b: f_b = (σ₂ − σ₃) − σ_ci·B(σ₂)^a, flow (0, 1+g₂, −1).
                let g2 = self.flow_major(sig[1]);
                (
                    [0.0, 1.0 + g2, -1.0],
                    self.yield_value(sig[1], sig[2]),
                    [0.0, self.dyield_dmajor(sig[1]), -1.0],
                    false,
                )
            };
            let dm_a = self.d_times_m(m_a);
            let dm_b = self.d_times_m(m_b_dir);
            let r = [
                sig[0] - trial[0] + la * dm_a[0] + lb * dm_b[0],
                sig[1] - trial[1] + la * dm_a[1] + lb * dm_b[1],
                sig[2] - trial[2] + la * dm_a[2] + lb * dm_b[2],
                self.yield_value(sig[0], sig[2]),
                f_b,
            ];
            if max_abs5(r) < 1e-7 * self.residual_scale() {
                return self.accept_double(sig, la, lb);
            }
            let lambda = f64::from(self.lame.lambda);
            let two_mu = 2.0 * f64::from(self.lame.mu);
            let dg1 = self.dflow_major(sig[0]);
            // d(D·m_a)/dσ′₁.
            let dma0 = (lambda + two_mu) * dg1;
            let dma1 = lambda * dg1;
            let dma2 = lambda * dg1;
            // d(D·m_b)/dσ′₁ or dσ′₂ depending on which stress drives surface b.
            let (dmb0, dmb1, dmb2) = if major_b_is_1 {
                ((lambda + two_mu) * dg1, lambda * dg1, lambda * dg1)
            } else {
                let dg2 = self.dflow_major(sig[1]);
                (lambda * dg2, (lambda + two_mu) * dg2, lambda * dg2)
            };
            let df_a = [self.dyield_dmajor(sig[0]), 0.0, -1.0];
            // Columns: σ′₁, σ′₂, σ′₃, Δλₐ, Δλ_b.
            let (dmb_col0, dmb_col1) = if major_b_is_1 {
                // surface-b flow varies with σ′₁ only.
                ([dmb0, dmb1, dmb2], [0.0, 0.0, 0.0])
            } else {
                // surface-b flow varies with σ′₂ only.
                ([0.0, 0.0, 0.0], [dmb0, dmb1, dmb2])
            };
            let jac = [
                [
                    1.0 + la * dma0 + lb * dmb_col0[0],
                    lb * dmb_col1[0],
                    0.0,
                    dm_a[0],
                    dm_b[0],
                ],
                [
                    la * dma1 + lb * dmb_col0[1],
                    1.0 + lb * dmb_col1[1],
                    0.0,
                    dm_a[1],
                    dm_b[1],
                ],
                [
                    la * dma2 + lb * dmb_col0[2],
                    lb * dmb_col1[2],
                    1.0,
                    dm_a[2],
                    dm_b[2],
                ],
                [df_a[0], df_a[1], df_a[2], 0.0, 0.0],
                [df_b[0], df_b[1], df_b[2], 0.0, 0.0],
            ];
            let delta = solve_linear::<5>(jac, r)?;
            sig[0] -= delta[0];
            sig[1] -= delta[1];
            sig[2] -= delta[2];
            la -= delta[3];
            lb -= delta[4];
        }
        None
    }

    /// The tensile apex `σ = s · σ_ci / m_b` on every axis (`B = 0`).
    fn apex(&self) -> [f64; 3] {
        let apex = self.s * self.sigma_ci / self.m_b;
        [apex, apex, apex]
    }

    /// A representative stress magnitude used to scale Newton convergence.
    #[inline]
    fn residual_scale(&self) -> f64 {
        self.sigma_ci.max(1.0)
    }

    /// Accepts a single-surface result when its multiplier is non-negative and
    /// the stresses stay ordered.
    fn accept_single(&self, sig: [f64; 3], dl: f64) -> Option<([f64; 3], f64)> {
        if dl < f64::from(MIN_MULTIPLIER) || !self.ordered(sig) {
            return None;
        }
        Some((sig, dl))
    }

    /// Accepts a two-surface edge result when both multipliers are non-negative
    /// and the stresses stay ordered.
    fn accept_double(&self, sig: [f64; 3], la: f64, lb: f64) -> Option<([f64; 3], f64)> {
        if la < f64::from(MIN_MULTIPLIER) || lb < f64::from(MIN_MULTIPLIER) || !self.ordered(sig) {
            return None;
        }
        Some((sig, la + lb))
    }
}

/// Performs the Hoek–Brown elastic-predictor / plastic-return update for one
/// element.
///
/// `lame` supplies the shear and first Lamé parameters; the carried
/// [`HoekBrownState`] is mutated in place when the material yields. The
/// returned elastic gradient always satisfies `F_total ≈ Fₑ · Fₚ` with the
/// updated `Fₚ`.
#[must_use]
pub fn return_map_hoek_brown(
    f_total: Mat3,
    lame: &LameParameters,
    model: &HoekBrownModel,
    state: &mut HoekBrownState,
) -> HoekBrownStep {
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
    let sigma_tr = [
        f64::from(two_mu * eps.x + lame.lambda * trace),
        f64::from(two_mu * eps.y + lame.lambda * trace),
        f64::from(two_mu * eps.z + lame.lambda * trace),
    ];

    let sigma_ci = model.effective_sigma_ci(state.accumulated_strain);
    let ctx = PrincipalReturn {
        lame,
        sigma_ci: f64::from(sigma_ci),
        m_b: f64::from(model.m_b),
        s: f64::from(model.s),
        a: f64::from(model.a),
        m_g: f64::from(model.m_g),
        order_tol: 1e-4 * sigma_tr_scale(sigma_tr),
    };

    // The elastic domain requires both a non-positive yield value and a
    // major stress below the tensile apex (positive raw bracket); a predictor
    // past the apex is always plastic even though its clamped yield value may
    // read slightly negative.
    let raw = ctx.raw_bracket(sigma_tr[0]);
    let f_trial = ctx.yield_value(sigma_tr[0], sigma_tr[2]);
    if raw > 0.0 && f_trial <= 0.0 {
        return HoekBrownStep {
            elastic_gradient: fe_trial,
            plastic_increment: 0.0,
            mode: HoekBrownYield::Elastic,
        };
    }

    let (sigma_ret, mode) = ctx
        .main_surface(sigma_tr)
        .map(|(s, _)| (s, HoekBrownYield::MainSurface))
        .or_else(|| {
            ctx.edge(sigma_tr, true)
                .map(|(s, _)| (s, HoekBrownYield::CompressionEdge))
        })
        .or_else(|| {
            ctx.edge(sigma_tr, false)
                .map(|(s, _)| (s, HoekBrownYield::ExtensionEdge))
        })
        .unwrap_or_else(|| (ctx.apex(), HoekBrownYield::Apex));

    // Invert σ = 2μ ε + λ (trε) I to recover the returned elastic log-strain.
    let bulk = 3.0 * f64::from(lame.lambda) + 2.0 * f64::from(lame.mu);
    let tr_sigma = sigma_ret[0] + sigma_ret[1] + sigma_ret[2];
    let tr_eps = if bulk > 0.0 { tr_sigma / bulk } else { 0.0 };
    let two_mu64 = 2.0 * f64::from(lame.mu);
    let lambda64 = f64::from(lame.lambda);
    let eps_elastic = Vec3::new(
        ((sigma_ret[0] - lambda64 * tr_eps) / two_mu64) as f32,
        ((sigma_ret[1] - lambda64 * tr_eps) / two_mu64) as f32,
        ((sigma_ret[2] - lambda64 * tr_eps) / two_mu64) as f32,
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

    HoekBrownStep {
        elastic_gradient: fe_new,
        plastic_increment: increment,
        mode,
    }
}

/// Solves the `N×N` linear system `J·x = r` by Gaussian elimination with
/// partial pivoting, returning `None` on a singular system. The matrix is given
/// row-major in `jac`.
fn solve_linear<const N: usize>(mut jac: [[f64; N]; N], mut r: [f64; N]) -> Option<[f64; N]> {
    for col in 0..N {
        // Partial pivot.
        let mut pivot = col;
        let mut best = jac[col][col].abs();
        #[expect(
            clippy::needless_range_loop,
            reason = "pivot search reads one column across rows; an index loop is clearest"
        )]
        for row in (col + 1)..N {
            let v = jac[row][col].abs();
            if v > best {
                best = v;
                pivot = row;
            }
        }
        if best < 1e-18 {
            return None;
        }
        if pivot != col {
            jac.swap(col, pivot);
            r.swap(col, pivot);
        }
        let diag = jac[col][col];
        for row in (col + 1)..N {
            let factor = jac[row][col] / diag;
            if factor != 0.0 {
                #[expect(
                    clippy::needless_range_loop,
                    reason = "row-reduce indexes two distinct rows by the same column index"
                )]
                for k in col..N {
                    jac[row][k] -= factor * jac[col][k];
                }
                r[row] -= factor * r[col];
            }
        }
    }
    let mut x = [0.0_f64; N];
    for col in (0..N).rev() {
        let mut sum = r[col];
        for k in (col + 1)..N {
            sum -= jac[col][k] * x[k];
        }
        x[col] = sum / jac[col][col];
    }
    Some(x)
}

/// Maximum absolute component of a 4-vector residual.
#[inline]
fn max_abs4(r: [f64; 4]) -> f64 {
    r.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

/// Maximum absolute component of a 5-vector residual.
#[inline]
fn max_abs5(r: [f64; 5]) -> f64 {
    r.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

/// Scale of the trial principal stresses for the ordering tolerance.
#[inline]
fn sigma_tr_scale(s: [f64; 3]) -> f32 {
    let m = s.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    (m.max(1.0)) as f32
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

    // A soft rock: low intact strength so yielding is reachable at the strain
    // scale used elsewhere in the suite; original criterion (s = 1, a = 0.5)
    // with associated flow and no hardening.
    fn model() -> HoekBrownModel {
        HoekBrownModel::new(5000.0, 1.0, 1.0, 0.5, 1.0, 0.0).unwrap()
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

    // Principal Kirchhoff stresses of a step's returned elastic gradient.
    fn principal_stress(step: &HoekBrownStep, l: &LameParameters) -> Vec3 {
        let svd = svd3(step.elastic_gradient);
        let eps = Vec3::new(
            hencky(svd.sigma.x),
            hencky(svd.sigma.y),
            hencky(svd.sigma.z),
        );
        let trace = eps.x + eps.y + eps.z;
        let two_mu = 2.0 * l.mu;
        Vec3::new(
            two_mu * eps.x + l.lambda * trace,
            two_mu * eps.y + l.lambda * trace,
            two_mu * eps.z + l.lambda * trace,
        )
    }

    fn assert_reconstructs(step: &HoekBrownStep, state: &HoekBrownState, f: Mat3) {
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
        assert!(HoekBrownModel::new(0.0, 1.0, 1.0, 0.5, 1.0, 0.0).is_none());
        assert!(HoekBrownModel::new(5000.0, 0.0, 1.0, 0.5, 0.0, 0.0).is_none());
        assert!(HoekBrownModel::new(5000.0, 1.0, 0.0, 0.5, 1.0, 0.0).is_none());
        assert!(HoekBrownModel::new(5000.0, 1.0, 1.1, 0.5, 1.0, 0.0).is_none());
        assert!(HoekBrownModel::new(5000.0, 1.0, 1.0, 0.0, 1.0, 0.0).is_none());
        assert!(HoekBrownModel::new(5000.0, 1.0, 1.0, 1.1, 1.0, 0.0).is_none());
        // dilation must not exceed m_b.
        assert!(HoekBrownModel::new(5000.0, 1.0, 1.0, 0.5, 2.0, 0.0).is_none());
        assert!(HoekBrownModel::new(5000.0, 1.0, 1.0, 0.5, 1.0, -1.0).is_none());
        assert!(HoekBrownModel::new(5000.0, 1.0, 1.0, 0.5, 1.0, 0.0).is_some());
        assert!(HoekBrownModel::intact(5000.0, 1.0).is_some());
    }

    #[test]
    fn rest_pose_is_elastic() {
        let mut state = HoekBrownState::rest();
        let step = return_map_hoek_brown(Mat3::IDENTITY, &lame(), &model(), &mut state);
        assert_eq!(step.mode, HoekBrownYield::Elastic);
        assert_eq!(step.plastic_increment, 0.0);
        assert_eq!(state, HoekBrownState::rest());
    }

    #[test]
    fn small_shear_stays_elastic() {
        let delta = 0.001_f32;
        let f = from_log_strain(Vec3::new(delta, 0.0, -delta));
        let mut state = HoekBrownState::rest();
        let step = return_map_hoek_brown(f, &lame(), &model(), &mut state);
        assert_eq!(step.mode, HoekBrownYield::Elastic);
        assert_eq!(state, HoekBrownState::rest());
    }

    #[test]
    fn confined_shear_returns_to_main_surface() {
        // Confined (compressive σ₁) triaxial load overshoots the curved surface.
        let f = from_log_strain(Vec3::new(-0.005, -0.02, -0.035));
        let mut state = HoekBrownState::rest();
        let step = return_map_hoek_brown(f, &lame(), &model(), &mut state);
        assert_eq!(step.mode, HoekBrownYield::MainSurface);
        assert!(step.plastic_increment > 0.0);
        assert_reconstructs(&step, &state, f);
        // The returned stresses sit on the curved yield surface (f ≈ 0).
        let s = principal_stress(&step, &lame());
        let m = model();
        let bracket = (f64::from(m.s())
            - f64::from(m.m_b()) * f64::from(s.x) / f64::from(m.sigma_ci()))
        .max(0.0);
        let residual =
            f64::from(s.x - s.z) - f64::from(m.sigma_ci()) * bracket.powf(f64::from(m.a()));
        assert!(residual.abs() < 5.0, "residual {residual}");
        assert!(s.x >= s.y - 1.0 && s.y >= s.z - 1.0, "stays ordered");
    }

    #[test]
    fn hydrostatic_tension_hits_apex() {
        // Uniform expansion drives hydrostatic tension far past the apex.
        let f = from_log_strain(Vec3::splat(0.05));
        let mut state = HoekBrownState::rest();
        let step = return_map_hoek_brown(f, &lame(), &model(), &mut state);
        assert_eq!(step.mode, HoekBrownYield::Apex);
        assert!(step.plastic_increment > 0.0);
        assert_reconstructs(&step, &state, f);
        // Returned stress is hydrostatic at σ = s·σ_ci/m_b.
        let s = principal_stress(&step, &lame());
        let m = model();
        let apex = m.s() * m.sigma_ci() / m.m_b();
        assert!((s.x - apex).abs() < 5.0, "σ₁ {} vs apex {apex}", s.x);
        assert!((s.x - s.z).abs() < 5.0, "apex must be hydrostatic");
    }

    #[test]
    fn confinement_enlarges_elastic_region() {
        // A shear that yields unconfined stays elastic once a strong hydrostatic
        // compression (very negative σ₁) steepens the curved strength envelope.
        let delta = 0.01_f32;
        let unconfined = from_log_strain(Vec3::new(delta, 0.0, -delta));
        let mut s0 = HoekBrownState::rest();
        let yielded = return_map_hoek_brown(unconfined, &lame(), &model(), &mut s0);
        assert_ne!(yielded.mode, HoekBrownYield::Elastic);

        let compress = -0.03_f32;
        let confined = from_log_strain(Vec3::new(delta + compress, compress, -delta + compress));
        let mut s1 = HoekBrownState::rest();
        let step = return_map_hoek_brown(confined, &lame(), &model(), &mut s1);
        assert_eq!(step.mode, HoekBrownYield::Elastic);
    }

    #[test]
    fn hardening_grows_effective_strength() {
        let m = HoekBrownModel::new(5000.0, 1.0, 1.0, 0.5, 1.0, 1000.0).unwrap();
        assert!((m.effective_sigma_ci(0.0) - 5000.0).abs() < 1e-3);
        assert!((m.effective_sigma_ci(2.0) - 7000.0).abs() < 1e-3);
    }

    #[test]
    fn nonassociated_flow_dilates_less() {
        // Associated vs reduced-dilation potential: both yield on the main
        // surface, and the reduced-dilation flow consumes a different (smaller
        // volumetric) plastic step while remaining admissible.
        let f = from_log_strain(Vec3::new(-0.005, -0.02, -0.035));
        let associated = HoekBrownModel::new(5000.0, 1.0, 1.0, 0.5, 1.0, 0.0).unwrap();
        let dilatant = HoekBrownModel::new(5000.0, 1.0, 1.0, 0.5, 0.2, 0.0).unwrap();
        let mut sa = HoekBrownState::rest();
        let mut sb = HoekBrownState::rest();
        let a = return_map_hoek_brown(f, &lame(), &associated, &mut sa);
        let b = return_map_hoek_brown(f, &lame(), &dilatant, &mut sb);
        assert_eq!(a.mode, HoekBrownYield::MainSurface);
        assert_eq!(b.mode, HoekBrownYield::MainSurface);
        assert!(a.plastic_increment > 0.0 && b.plastic_increment > 0.0);
        // Different plastic potentials give measurably different flow.
        assert!((a.plastic_increment - b.plastic_increment).abs() > 1e-6);
    }

    #[test]
    fn returned_stress_is_finite_and_ordered() {
        // A general asymmetric overshoot must always land on an ordered,
        // finite state regardless of which feature is selected.
        let f = from_log_strain(Vec3::new(0.004, -0.01, -0.03));
        let mut state = HoekBrownState::rest();
        let step = return_map_hoek_brown(f, &lame(), &model(), &mut state);
        assert_ne!(step.mode, HoekBrownYield::Elastic);
        assert_reconstructs(&step, &state, f);
        let s = principal_stress(&step, &lame());
        assert!(s.x.is_finite() && s.y.is_finite() && s.z.is_finite());
        assert!(s.x >= s.y - 1.0 && s.y >= s.z - 1.0, "stays ordered: {s:?}");
    }

    #[test]
    fn is_deterministic() {
        let f = from_log_strain(Vec3::new(-0.005, -0.02, -0.035));
        let mut a = HoekBrownState::rest();
        let mut b = HoekBrownState::rest();
        let sa = return_map_hoek_brown(f, &lame(), &model(), &mut a);
        let sb = return_map_hoek_brown(f, &lame(), &model(), &mut b);
        assert_eq!(sa, sb);
        assert_eq!(a, b);
    }
}
