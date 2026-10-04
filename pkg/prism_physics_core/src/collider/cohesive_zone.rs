//! Mixed-mode bilinear cohesive-zone traction–separation law.
//!
//! A cohesive zone models the gradual loss of cohesion across an interface
//! (a crack face, a bonded contact, or a weak internal surface) as the two
//! sides separate. Unlike the bulk continuum-damage model in
//! [`tet_fem_damage`](super::tet_fem_damage) — which degrades a *volumetric*
//! stress — this is a *surface* law: it maps a relative displacement jump
//! `δ` across the interface to a traction `t` (force per unit area) and tracks
//! how much irreversible decohesion has accumulated.
//!
//! # Bilinear law
//!
//! The scalar response along the effective separation `λ` is the classic
//! bilinear (Alfano–Crisfield) envelope:
//!
//! ```text
//!   t(λ) = K · λ                       for 0 ≤ λ ≤ δ₀      (reversible elastic rise)
//!   t(λ) = σ_c · (δ_f − λ)/(δ_f − δ₀)  for δ₀ < λ < δ_f    (linear softening)
//!   t(λ) = 0                           for λ ≥ δ_f         (fully decohered)
//! ```
//!
//! with penalty stiffness `K`, peak traction (strength) `σ_c`, onset
//! separation `δ₀ = σ_c / K`, and final separation `δ_f = 2·G_c / σ_c` chosen
//! so the area under the curve equals the fracture energy `G_c`. A valid model
//! therefore requires `G_c > σ_c² / (2K)` so that `δ_f > δ₀`.
//!
//! # Irreversible damage
//!
//! The secant form of the law introduces a scalar damage `d ∈ [0, 1]` driven
//! by the largest effective separation ever reached, `κ = max λ`:
//!
//! ```text
//!   d(κ) = 0                               κ ≤ δ₀
//!   d(κ) = δ_f·(κ − δ₀) / (κ·(δ_f − δ₀))   δ₀ < κ < δ_f
//!   d(κ) = 1                               κ ≥ δ_f
//! ```
//!
//! Loading past `δ₀` grows `κ` and hence `d`; unloading and reloading below
//! `κ` follow the reduced secant stiffness `(1 − d)·K` back through the origin,
//! so decohesion never heals.
//!
//! # Mixed mode
//!
//! Normal and tangential opening are combined into the effective separation
//! `λ = sqrt(⟨δₙ⟩₊² + β²·δₜ²)`, where `δₙ` is the normal component, `δₜ` the
//! tangential magnitude, `⟨·⟩₊` the positive part (so compression carries no
//! cohesive damage), and `β ≥ 0` a mode-mixity weight on shear. Under normal
//! compression (`δₙ < 0`) the interface develops the full penalty traction
//! `K·δₙ` to resist interpenetration but accrues no damage; the tangential and
//! tensile-normal tractions are scaled by the shared secant factor `(1 − d)`.
//!
//! Everything here is a pure, per-interface kernel: it holds no solver state
//! beyond the caller-owned [`CohesiveState`] and performs no time integration.

use glam::Vec3;

/// Immutable parameters of a bilinear cohesive-zone law.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CohesiveModel {
    stiffness: f32,
    strength: f32,
    fracture_energy: f32,
    shear_weight: f32,
    onset_separation: f32,
    final_separation: f32,
}

impl CohesiveModel {
    /// Builds a cohesive model from the penalty `stiffness` `K`, peak traction
    /// `strength` `σ_c`, `fracture_energy` `G_c`, and shear mode-mixity weight
    /// `shear_weight` `β`.
    ///
    /// Returns `None` unless every value is finite, `K`, `σ_c`, `G_c` are
    /// strictly positive, `β ≥ 0`, and the fracture energy is large enough that
    /// the softening branch exists (`δ_f = 2·G_c/σ_c > δ₀ = σ_c/K`, i.e.
    /// `G_c > σ_c²/(2K)`).
    #[must_use]
    pub fn new(
        stiffness: f32,
        strength: f32,
        fracture_energy: f32,
        shear_weight: f32,
    ) -> Option<Self> {
        if !(stiffness.is_finite()
            && strength.is_finite()
            && fracture_energy.is_finite()
            && shear_weight.is_finite())
        {
            return None;
        }
        if stiffness <= 0.0 || strength <= 0.0 || fracture_energy <= 0.0 || shear_weight < 0.0 {
            return None;
        }
        let onset_separation = strength / stiffness;
        let final_separation = 2.0 * fracture_energy / strength;
        // Both are finite (derived from validated finite, positive inputs), so a
        // direct comparison is safe and the softening branch requires δ_f > δ₀.
        if final_separation <= onset_separation {
            return None;
        }
        Some(Self {
            stiffness,
            strength,
            fracture_energy,
            shear_weight,
            onset_separation,
            final_separation,
        })
    }

    /// Penalty / initial-slope stiffness `K`.
    #[must_use]
    pub fn stiffness(&self) -> f32 {
        self.stiffness
    }

    /// Peak traction (interface strength) `σ_c`.
    #[must_use]
    pub fn strength(&self) -> f32 {
        self.strength
    }

    /// Fracture energy `G_c` (area under the traction–separation curve).
    #[must_use]
    pub fn fracture_energy(&self) -> f32 {
        self.fracture_energy
    }

    /// Shear mode-mixity weight `β` applied to the tangential separation.
    #[must_use]
    pub fn shear_weight(&self) -> f32 {
        self.shear_weight
    }

    /// Separation `δ₀` at which the traction peaks and softening begins.
    #[must_use]
    pub fn onset_separation(&self) -> f32 {
        self.onset_separation
    }

    /// Separation `δ_f` at which cohesion is fully lost (`d = 1`).
    #[must_use]
    pub fn final_separation(&self) -> f32 {
        self.final_separation
    }

    /// Secant damage `d(κ)` for a monotone history variable `κ ≥ 0`.
    #[must_use]
    pub fn damage_at(&self, kappa: f32) -> f32 {
        if kappa <= self.onset_separation {
            return 0.0;
        }
        if kappa >= self.final_separation {
            return 1.0;
        }
        // d = δ_f·(κ − δ₀) / (κ·(δ_f − δ₀)); continuous: 0 at δ₀, 1 at δ_f.
        let num = self.final_separation * (kappa - self.onset_separation);
        let den = kappa * (self.final_separation - self.onset_separation);
        (num / den).clamp(0.0, 1.0)
    }

    /// Monotonic-loading effective traction magnitude on the bilinear envelope
    /// at effective separation `lambda ≥ 0`. This is the response of a pristine
    /// interface loaded straight to `lambda`; a damaged interface uses the
    /// secant form in [`cohesive_traction`].
    #[must_use]
    pub fn envelope_traction(&self, lambda: f32) -> f32 {
        if lambda <= 0.0 {
            return 0.0;
        }
        if lambda <= self.onset_separation {
            return self.stiffness * lambda;
        }
        if lambda >= self.final_separation {
            return 0.0;
        }
        self.strength * (self.final_separation - lambda)
            / (self.final_separation - self.onset_separation)
    }
}

/// Caller-owned irreversible history of one cohesive interface: the largest
/// effective separation `κ` ever reached.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CohesiveState {
    kappa: f32,
}

impl CohesiveState {
    /// A pristine (never-loaded) interface with `κ = 0`.
    #[must_use]
    pub fn rest() -> Self {
        Self { kappa: 0.0 }
    }

    /// The stored maximum effective separation `κ`.
    #[must_use]
    pub fn kappa(&self) -> f32 {
        self.kappa
    }
}

impl Default for CohesiveState {
    fn default() -> Self {
        Self::rest()
    }
}

/// Outcome of a single cohesive-traction evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CohesiveStep {
    /// Effective (mixed-mode) separation `λ` of this evaluation.
    pub effective_separation: f32,
    /// Secant decohesion damage `d ∈ [0, 1]` after updating the history.
    pub damage: f32,
    /// Full traction vector `t` developed across the interface.
    pub traction: Vec3,
    /// Signed normal component of the traction (`t · n`): positive in tension,
    /// negative under the compressive penalty.
    pub normal_traction: f32,
    /// Whether this evaluation advanced the irreversible history `κ`.
    pub advanced: bool,
}

/// Evaluates the mixed-mode bilinear cohesive traction for a relative
/// displacement jump `delta` across an interface with unit outward `normal`,
/// advancing the irreversible `state` once.
///
/// `normal` is normalised defensively; a degenerate (near-zero) normal yields a
/// zero traction and leaves the state untouched. The returned traction is
/// co-directional with the separation it resists: a tensile normal opening
/// produces a positive normal traction pulling the faces back together, while a
/// compressive normal jump produces the full (undamaged) penalty traction so
/// the surfaces cannot interpenetrate.
#[must_use]
pub fn cohesive_traction(
    delta: Vec3,
    normal: Vec3,
    model: &CohesiveModel,
    state: &mut CohesiveState,
) -> CohesiveStep {
    let n_len = normal.length();
    if n_len <= f32::EPSILON {
        return CohesiveStep {
            effective_separation: 0.0,
            damage: model.damage_at(state.kappa),
            traction: Vec3::ZERO,
            normal_traction: 0.0,
            advanced: false,
        };
    }
    let n = normal / n_len;

    let delta_n = delta.dot(n);
    let tangent_vec = delta - delta_n * n;
    let delta_t = tangent_vec.length();

    // Effective separation that drives damage: compression carries none.
    let open_n = delta_n.max(0.0);
    let beta = model.shear_weight();
    let lambda = (open_n * open_n + beta * beta * delta_t * delta_t).sqrt();

    let advanced = lambda > state.kappa && lambda > model.onset_separation();
    if lambda > state.kappa {
        state.kappa = lambda;
    }
    let damage = model.damage_at(state.kappa);
    let secant = 1.0 - damage;
    let k = model.stiffness();

    // Normal traction: degraded secant in tension, full penalty in compression.
    let normal_traction = if delta_n >= 0.0 {
        secant * k * delta_n
    } else {
        k * delta_n
    };

    // Tangential traction: degraded secant along the opening direction.
    let tangent_traction = if delta_t > 0.0 {
        (secant * k) * tangent_vec
    } else {
        Vec3::ZERO
    };

    let traction = normal_traction * n + tangent_traction;
    CohesiveStep {
        effective_separation: lambda,
        damage,
        traction,
        normal_traction,
        advanced,
    }
}

/// Energy per unit area dissipated by decohesion for a history variable `κ`,
/// i.e. the area between the bilinear envelope and the current secant unloading
/// line. Ranges from `0` at `κ ≤ δ₀` to the full fracture energy `G_c` at
/// `κ ≥ δ_f`.
#[must_use]
pub fn dissipated_energy(model: &CohesiveModel, kappa: f32) -> f32 {
    if kappa <= model.onset_separation() {
        return 0.0;
    }
    if kappa >= model.final_separation() {
        return model.fracture_energy();
    }
    // Dissipation of the bilinear law up to κ:
    //   Φ(κ) = ½·σ_c·δ_f·(κ − δ₀)/(δ_f − δ₀).
    0.5 * model.strength() * model.final_separation() * (kappa - model.onset_separation())
        / (model.final_separation() - model.onset_separation())
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: Vec3 = Vec3::new(0.0, 0.0, 1.0);

    // K = 1e6, σ_c = 1e3 → δ₀ = 1e-3. G_c = 1.0 → δ_f = 2e-3 (> δ₀, valid).
    fn model() -> CohesiveModel {
        CohesiveModel::new(1.0e6, 1.0e3, 1.0, 1.0).unwrap()
    }

    #[test]
    fn new_rejects_bad_params() {
        assert!(
            CohesiveModel::new(0.0, 1.0e3, 1.0, 1.0).is_none(),
            "K must be > 0"
        );
        assert!(
            CohesiveModel::new(1.0e6, 0.0, 1.0, 1.0).is_none(),
            "σ_c must be > 0"
        );
        assert!(
            CohesiveModel::new(1.0e6, 1.0e3, 0.0, 1.0).is_none(),
            "G_c must be > 0"
        );
        assert!(
            CohesiveModel::new(1.0e6, 1.0e3, 1.0, -1.0).is_none(),
            "β must be ≥ 0"
        );
        // G_c too small so δ_f ≤ δ₀: σ_c²/(2K) = 1e6/2e6 = 0.5; G_c = 0.4 < 0.5.
        assert!(
            CohesiveModel::new(1.0e6, 1.0e3, 0.4, 1.0).is_none(),
            "insufficient fracture energy must be rejected"
        );
        assert!(CohesiveModel::new(f32::NAN, 1.0e3, 1.0, 1.0).is_none());
    }

    #[test]
    fn new_derives_onset_and_final() {
        let m = model();
        assert!((m.onset_separation() - 1.0e-3).abs() < 1e-9);
        assert!((m.final_separation() - 2.0e-3).abs() < 1e-9);
    }

    #[test]
    fn rest_zero_separation_is_traction_free() {
        let m = model();
        let mut st = CohesiveState::rest();
        let step = cohesive_traction(Vec3::ZERO, N, &m, &mut st);
        assert_eq!(step.traction, Vec3::ZERO);
        assert_eq!(step.damage, 0.0);
        assert!(!step.advanced);
        assert_eq!(st, CohesiveState::rest());
    }

    #[test]
    fn elastic_below_onset_is_linear_and_undamaged() {
        let m = model();
        let mut st = CohesiveState::rest();
        let delta_n = 0.5e-3; // half the onset
        let step = cohesive_traction(delta_n * N, N, &m, &mut st);
        assert!(!step.advanced, "sub-onset load must not advance history");
        assert_eq!(step.damage, 0.0);
        let expected = m.stiffness() * delta_n;
        assert!((step.normal_traction - expected).abs() <= 1e-3 * expected.abs());
        assert!((step.traction - expected * N).length() <= 1e-3 * expected.abs());
    }

    #[test]
    fn traction_peaks_at_onset() {
        let m = model();
        let mut st = CohesiveState::rest();
        let step = cohesive_traction(m.onset_separation() * N, N, &m, &mut st);
        assert!(
            (step.normal_traction - m.strength()).abs() <= 1e-2 * m.strength(),
            "traction at δ₀ must equal σ_c, got {}",
            step.normal_traction
        );
        assert!(step.damage <= 1e-6, "damage onsets exactly at δ₀");
    }

    #[test]
    fn softening_past_onset_reduces_traction() {
        let m = model();
        let mut st = CohesiveState::rest();
        // Midway through softening: λ = 1.5e-3 ∈ (δ₀, δ_f).
        let delta_n = 1.5e-3;
        let step = cohesive_traction(delta_n * N, N, &m, &mut st);
        assert!(
            step.damage > 0.0 && step.damage < 1.0,
            "partial damage, got {}",
            step.damage
        );
        assert!(step.advanced);
        // Secant traction must be below both the elastic prediction and σ_c.
        assert!(step.normal_traction < m.stiffness() * delta_n);
        assert!(step.normal_traction < m.strength());
        // Envelope at λ: σ_c·(δ_f − λ)/(δ_f − δ₀) = 1e3·0.5e-3/1e-3 = 500.
        assert!((step.normal_traction - 500.0).abs() <= 2.0);
    }

    #[test]
    fn full_separation_loses_all_cohesion() {
        let m = model();
        let mut st = CohesiveState::rest();
        let step = cohesive_traction((m.final_separation() * 1.1) * N, N, &m, &mut st);
        assert!(
            (step.damage - 1.0).abs() < 1e-6,
            "beyond δ_f must be fully damaged"
        );
        assert!(
            step.normal_traction.abs() <= 1e-3,
            "no tensile traction once decohered"
        );
    }

    #[test]
    fn compression_uses_full_penalty_without_damage() {
        let m = model();
        let mut st = CohesiveState::rest();
        let delta_n = -0.5e-3;
        let step = cohesive_traction(delta_n * N, N, &m, &mut st);
        assert_eq!(step.damage, 0.0, "compression carries no cohesive damage");
        assert!(!step.advanced);
        let expected = m.stiffness() * delta_n; // negative penalty
        assert!((step.normal_traction - expected).abs() <= 1e-3 * expected.abs());
        assert!(
            step.normal_traction < 0.0,
            "penalty must resist interpenetration"
        );
    }

    #[test]
    fn compression_still_penalises_after_tensile_damage() {
        // Damage the interface in tension, then push it closed: the penalty is
        // the full (undamaged) stiffness even though tension is degraded.
        let m = model();
        let mut st = CohesiveState::rest();
        let _ = cohesive_traction(1.6e-3 * N, N, &m, &mut st);
        let damaged = m.damage_at(st.kappa());
        assert!(damaged > 0.0);
        let push = cohesive_traction(-0.4e-3 * N, N, &m, &mut st);
        let expected = m.stiffness() * -0.4e-3;
        assert!(
            (push.normal_traction - expected).abs() <= 1e-3 * expected.abs(),
            "compression must use full penalty regardless of tensile damage"
        );
    }

    #[test]
    fn pure_shear_drives_damage_and_opposes_sliding() {
        let m = model();
        let mut st = CohesiveState::rest();
        let tdir = Vec3::new(1.0, 0.0, 0.0);
        // Tangential jump past onset (β = 1 ⇒ λ = δₜ).
        let step = cohesive_traction(1.5e-3 * tdir, N, &m, &mut st);
        assert!(step.damage > 0.0, "shear past onset must damage");
        assert!(step.advanced);
        assert!(
            step.normal_traction.abs() <= 1e-3,
            "no normal opening ⇒ no normal traction"
        );
        // Tangential traction opposes the slide (co-directional restoring stress).
        assert!(step.traction.dot(tdir) > 0.0);
    }

    #[test]
    fn unloading_follows_frozen_secant_through_origin() {
        let m = model();
        let mut st = CohesiveState::rest();
        let big = cohesive_traction(1.6e-3 * N, N, &m, &mut st);
        let frozen = big.damage;
        assert!(frozen > 0.0 && frozen < 1.0);

        let small_delta = 0.8e-3;
        let small = cohesive_traction(small_delta * N, N, &m, &mut st);
        assert!(!small.advanced, "reload below κ must not advance history");
        assert_eq!(
            small.damage, frozen,
            "unloading must reuse the frozen damage"
        );
        let expected = (1.0 - frozen) * m.stiffness() * small_delta;
        assert!(
            (small.normal_traction - expected).abs() <= 1e-3 * expected.abs(),
            "unloading must follow the secant (1−d)K line through the origin"
        );
    }

    #[test]
    fn history_is_monotonic_and_irreversible() {
        let m = model();
        let mut st = CohesiveState::rest();
        let _ = cohesive_traction(1.7e-3 * N, N, &m, &mut st);
        let peak = st.kappa();
        let _ = cohesive_traction(0.5e-3 * N, N, &m, &mut st);
        assert_eq!(st.kappa(), peak, "κ must never decrease");
    }

    #[test]
    fn dissipation_ranges_from_zero_to_fracture_energy() {
        let m = model();
        assert_eq!(dissipated_energy(&m, m.onset_separation()), 0.0);
        assert!((dissipated_energy(&m, m.final_separation()) - m.fracture_energy()).abs() < 1e-6);
        let mid = dissipated_energy(&m, 1.5e-3);
        assert!(
            mid > 0.0 && mid < m.fracture_energy(),
            "partial dissipation, got {mid}"
        );
    }

    #[test]
    fn degenerate_normal_is_traction_free() {
        let m = model();
        let mut st = CohesiveState::rest();
        let step = cohesive_traction(1.0e-3 * Vec3::X, Vec3::ZERO, &m, &mut st);
        assert_eq!(step.traction, Vec3::ZERO);
        assert!(!step.advanced);
        assert_eq!(st, CohesiveState::rest());
    }

    #[test]
    fn is_deterministic() {
        let m = model();
        let delta = Vec3::new(0.3e-3, 0.0, 1.2e-3);
        let mut a = CohesiveState::rest();
        let sa = cohesive_traction(delta, N, &m, &mut a);
        let mut b = CohesiveState::rest();
        let sb = cohesive_traction(delta, N, &m, &mut b);
        assert_eq!(sa, sb);
        assert_eq!(a, b);
    }
}
