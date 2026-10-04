//! Parallel-bond (bonded-particle) constitutive law for discrete elements.
//!
//! A *bonded-particle model* (BPM, after Potyondy & Cundall) glues two discrete
//! particles with a finite-size cylindrical cement that transmits force **and**
//! moment until it breaks. Unlike the surface
//! [`cohesive_zone`](super::cohesive_zone) law — which maps a displacement jump
//! to a traction on a facet — a parallel bond is an *incremental beam* between
//! two particle centres: it accumulates an axial force, a shear force, a bending
//! moment, and a twisting moment from relative motion, and snaps when the
//! extreme-fibre stress reaches a strength envelope.
//!
//! # Bond geometry
//!
//! The cement is a disc of radius `R`, giving cross-sectional area
//! `A = π·R²`, second moment of area `I = π·R⁴/4`, and polar moment
//! `J = π·R⁴/2`. These convert the per-area normal / shear stiffnesses into the
//! translational and rotational responses below.
//!
//! # Incremental response
//!
//! Each update advances the bond by a relative translational increment `Δu`
//! (particle *b* relative to particle *a*) and a relative rotation increment
//! `Δθ`, decomposed about the current bond axis `n̂`:
//!
//! ```text
//!   ΔFₙ = kₙ·A·(Δu·n̂)                 axial force   (+ = tension)
//!   ΔFₛ = kₛ·A·(Δu − (Δu·n̂)·n̂)        shear force   (perpendicular)
//!   ΔMₜ = kₛ·J·(Δθ·n̂)                 twisting moment (about n̂)
//!   ΔM_b = kₙ·I·(Δθ − (Δθ·n̂)·n̂)       bending moment (perpendicular)
//! ```
//!
//! Increments accumulate in the caller-owned [`BondState`]; this pure kernel
//! performs no reprojection as the axis rotates, which is the standard
//! small-increment BPM simplification.
//!
//! # Strength envelope
//!
//! The extreme-fibre stresses combine the direct and flexural / torsional parts:
//!
//! ```text
//!   σ_max = Fₙ/A + |M_b|·R/I            (tension positive)
//!   τ_max = |Fₛ|/A + |Mₜ|·R/J
//! ```
//!
//! The bond breaks irreversibly when either limit is reached:
//!
//! ```text
//!   tensile:  σ_max ≥ σ_c
//!   shear:    τ_max ≥ c + μ·σ_comp      with σ_comp = max(−Fₙ/A, 0)
//! ```
//!
//! i.e. a Mohr–Coulomb shear envelope whose cohesion `c` is boosted by friction
//! `μ` acting on the compressive normal stress. Once broken, the bond carries no
//! force or moment and every later update is a no-op.

use glam::Vec3;
use std::f32::consts::PI;

/// Immutable parameters of a parallel bond.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BondModel {
    normal_stiffness: f32,
    shear_stiffness: f32,
    radius: f32,
    tensile_strength: f32,
    cohesion: f32,
    friction_coeff: f32,
    area: f32,
    inertia: f32,
    polar: f32,
}

impl BondModel {
    /// Builds a parallel bond from the per-area `normal_stiffness` `kₙ`,
    /// per-area `shear_stiffness` `kₛ`, cement `radius` `R`, tensile
    /// `tensile_strength` `σ_c`, shear `cohesion` `c`, and dimensionless
    /// `friction_coeff` `μ = tan φ`.
    ///
    /// Returns `None` unless every value is finite, `kₙ`, `kₛ`, `R`, `σ_c`, `c`
    /// are strictly positive, and `μ ≥ 0`.
    #[must_use]
    pub fn new(
        normal_stiffness: f32,
        shear_stiffness: f32,
        radius: f32,
        tensile_strength: f32,
        cohesion: f32,
        friction_coeff: f32,
    ) -> Option<Self> {
        if !(normal_stiffness.is_finite()
            && shear_stiffness.is_finite()
            && radius.is_finite()
            && tensile_strength.is_finite()
            && cohesion.is_finite()
            && friction_coeff.is_finite())
        {
            return None;
        }
        if normal_stiffness <= 0.0
            || shear_stiffness <= 0.0
            || radius <= 0.0
            || tensile_strength <= 0.0
            || cohesion <= 0.0
            || friction_coeff < 0.0
        {
            return None;
        }
        let r2 = radius * radius;
        let area = PI * r2;
        let inertia = 0.25 * PI * r2 * r2;
        let polar = 0.5 * PI * r2 * r2;
        Some(Self {
            normal_stiffness,
            shear_stiffness,
            radius,
            tensile_strength,
            cohesion,
            friction_coeff,
            area,
            inertia,
            polar,
        })
    }

    /// Per-area normal (axial) stiffness `kₙ`.
    #[must_use]
    pub fn normal_stiffness(&self) -> f32 {
        self.normal_stiffness
    }

    /// Per-area shear stiffness `kₛ`.
    #[must_use]
    pub fn shear_stiffness(&self) -> f32 {
        self.shear_stiffness
    }

    /// Cement disc radius `R`.
    #[must_use]
    pub fn radius(&self) -> f32 {
        self.radius
    }

    /// Tensile strength `σ_c`.
    #[must_use]
    pub fn tensile_strength(&self) -> f32 {
        self.tensile_strength
    }

    /// Shear cohesion `c`.
    #[must_use]
    pub fn cohesion(&self) -> f32 {
        self.cohesion
    }

    /// Friction coefficient `μ = tan φ`.
    #[must_use]
    pub fn friction_coeff(&self) -> f32 {
        self.friction_coeff
    }

    /// Cross-sectional area `A = π·R²`.
    #[must_use]
    pub fn area(&self) -> f32 {
        self.area
    }

    /// Second moment of area `I = π·R⁴/4`.
    #[must_use]
    pub fn inertia(&self) -> f32 {
        self.inertia
    }

    /// Polar moment of area `J = π·R⁴/2`.
    #[must_use]
    pub fn polar(&self) -> f32 {
        self.polar
    }
}

/// Caller-owned incremental state of one parallel bond.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BondState {
    normal_force: f32,
    shear_force: Vec3,
    bending_moment: Vec3,
    twisting_moment: f32,
    broken: bool,
}

impl BondState {
    /// A pristine, unloaded, intact bond.
    #[must_use]
    pub fn intact() -> Self {
        Self {
            normal_force: 0.0,
            shear_force: Vec3::ZERO,
            bending_moment: Vec3::ZERO,
            twisting_moment: 0.0,
            broken: false,
        }
    }

    /// Accumulated axial force (`+` tension, `−` compression).
    #[must_use]
    pub fn normal_force(&self) -> f32 {
        self.normal_force
    }

    /// Accumulated shear force vector (perpendicular to the bond axis).
    #[must_use]
    pub fn shear_force(&self) -> Vec3 {
        self.shear_force
    }

    /// Accumulated bending moment vector (perpendicular to the bond axis).
    #[must_use]
    pub fn bending_moment(&self) -> Vec3 {
        self.bending_moment
    }

    /// Accumulated twisting moment about the bond axis.
    #[must_use]
    pub fn twisting_moment(&self) -> f32 {
        self.twisting_moment
    }

    /// Whether the bond has broken (and now carries nothing).
    #[must_use]
    pub fn is_broken(&self) -> bool {
        self.broken
    }
}

impl Default for BondState {
    fn default() -> Self {
        Self::intact()
    }
}

/// Outcome of a single bond update.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BondStep {
    /// Extreme-fibre tensile stress `σ_max` after this increment.
    pub normal_stress: f32,
    /// Extreme-fibre shear stress `τ_max` after this increment.
    pub shear_stress: f32,
    /// Tensile strength limit `σ_c`.
    pub tensile_limit: f32,
    /// Mohr–Coulomb shear limit `c + μ·σ_comp` for this increment.
    pub shear_limit: f32,
    /// Whether the bond broke on this step.
    pub broke: bool,
}

/// Advances a parallel bond by a relative translational increment `delta_disp`
/// (particle `b` relative to `a`) and relative rotation increment `delta_rot`,
/// both decomposed about the unit bond axis `normal_axis`, then tests the
/// strength envelope.
///
/// `normal_axis` is normalised defensively; a degenerate (near-zero) axis or an
/// already-broken bond leaves the state untouched and reports zero stresses.
/// When the bond breaks this step, its force and moment are zeroed and
/// [`BondStep::broke`] is `true`; the stresses reported are the ones that
/// triggered failure.
#[must_use]
pub fn update_bond(
    model: &BondModel,
    normal_axis: Vec3,
    delta_disp: Vec3,
    delta_rot: Vec3,
    state: &mut BondState,
) -> BondStep {
    if state.broken {
        return BondStep {
            normal_stress: 0.0,
            shear_stress: 0.0,
            tensile_limit: model.tensile_strength,
            shear_limit: model.cohesion,
            broke: false,
        };
    }
    let axis_len = normal_axis.length();
    if axis_len <= f32::EPSILON {
        return BondStep {
            normal_stress: 0.0,
            shear_stress: 0.0,
            tensile_limit: model.tensile_strength,
            shear_limit: model.cohesion,
            broke: false,
        };
    }
    let n = normal_axis / axis_len;

    // Translational increments.
    let du_n = delta_disp.dot(n);
    let du_s = delta_disp - du_n * n;
    state.normal_force += model.normal_stiffness * model.area * du_n;
    state.shear_force += (model.shear_stiffness * model.area) * du_s;

    // Rotational increments.
    let dr_t = delta_rot.dot(n);
    let dr_b = delta_rot - dr_t * n;
    state.twisting_moment += model.shear_stiffness * model.polar * dr_t;
    state.bending_moment += (model.normal_stiffness * model.inertia) * dr_b;

    // Extreme-fibre stresses.
    let r_over_i = model.radius / model.inertia;
    let r_over_j = model.radius / model.polar;
    let normal_stress = state.normal_force / model.area + state.bending_moment.length() * r_over_i;
    let shear_stress =
        state.shear_force.length() / model.area + state.twisting_moment.abs() * r_over_j;

    // Mohr–Coulomb shear limit boosted by compressive normal stress.
    let compressive = (-state.normal_force / model.area).max(0.0);
    let shear_limit = model.cohesion + model.friction_coeff * compressive;

    let tensile_fail = normal_stress >= model.tensile_strength;
    let shear_fail = shear_stress >= shear_limit;
    let broke = tensile_fail || shear_fail;
    if broke {
        state.broken = true;
        state.normal_force = 0.0;
        state.shear_force = Vec3::ZERO;
        state.bending_moment = Vec3::ZERO;
        state.twisting_moment = 0.0;
    }

    BondStep {
        normal_stress,
        shear_stress,
        tensile_limit: model.tensile_strength,
        shear_limit,
        broke,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const X: Vec3 = Vec3::new(1.0, 0.0, 0.0);

    // kn = ks = 1e9 (per area), R = 0.01 → A ≈ 3.14e-4.
    // σ_c = 1e6, c = 1e6, μ = 0.5.
    fn model() -> BondModel {
        BondModel::new(1.0e9, 1.0e9, 0.01, 1.0e6, 1.0e6, 0.5).unwrap()
    }

    #[test]
    fn geometry_is_consistent() {
        let m = model();
        let r2 = 0.01f32 * 0.01;
        assert!((m.area() - PI * r2).abs() < 1e-10);
        assert!((m.inertia() - 0.25 * PI * r2 * r2).abs() < 1e-14);
        assert!((m.polar() - 2.0 * m.inertia()).abs() < 1e-14);
    }

    #[test]
    fn new_rejects_bad_params() {
        assert!(BondModel::new(0.0, 1.0e9, 0.01, 1.0e6, 1.0e6, 0.5).is_none());
        assert!(BondModel::new(1.0e9, 1.0e9, 0.0, 1.0e6, 1.0e6, 0.5).is_none());
        assert!(BondModel::new(1.0e9, 1.0e9, 0.01, 1.0e6, 1.0e6, -0.1).is_none());
        assert!(BondModel::new(1.0e9, 1.0e9, 0.01, f32::NAN, 1.0e6, 0.5).is_none());
    }

    #[test]
    fn small_tension_builds_axial_force_without_breaking() {
        let m = model();
        let mut s = BondState::intact();
        // δn small enough that σ = kn·δn stays below σ_c.
        // σ = kn·δn = 1e9·δn ⇒ δn = 1e-4 → σ = 1e5 < 1e6.
        let step = update_bond(&m, X, X * 1.0e-4, Vec3::ZERO, &mut s);
        assert!(!step.broke);
        assert!(!s.is_broken());
        assert!(s.normal_force() > 0.0, "tension positive");
        assert!((step.normal_stress - 1.0e5).abs() < 1.0, "σ = kn·δn");
    }

    #[test]
    fn tension_breaks_at_strength() {
        let m = model();
        let mut s = BondState::intact();
        // δn = 2e-3 ⇒ σ = 1e9·2e-3 = 2e6 ≥ σ_c (1e6) → break.
        let step = update_bond(&m, X, X * 2.0e-3, Vec3::ZERO, &mut s);
        assert!(step.broke);
        assert!(s.is_broken());
        assert_eq!(s.normal_force(), 0.0, "broken bond carries nothing");
    }

    #[test]
    fn broken_bond_is_a_noop() {
        let m = model();
        let mut s = BondState::intact();
        let _ = update_bond(&m, X, X * 2.0e-3, Vec3::ZERO, &mut s);
        assert!(s.is_broken());
        let step = update_bond(&m, X, X * 1.0, Vec3::ZERO, &mut s);
        assert!(!step.broke, "already broken");
        assert_eq!(step.normal_stress, 0.0);
        assert_eq!(s.normal_force(), 0.0);
    }

    #[test]
    fn compression_raises_shear_capacity() {
        let m = model();
        // Pure shear needed to break at zero normal: τ = c = 1e6.
        // τ = ks·δs ⇒ δs = 1e-3. Use δs = 9e-4 (τ = 9e5 < 1e6): intact.
        let mut s = BondState::intact();
        let shear_dir = Vec3::new(0.0, 9.0e-4, 0.0);
        let step = update_bond(&m, X, shear_dir, Vec3::ZERO, &mut s);
        assert!(!step.broke, "below cohesion with no confinement");

        // Now add compression first (δn = -1e-3 ⇒ σ_comp = 1e6), which raises
        // the shear limit to c + μ·σ_comp = 1e6 + 0.5·1e6 = 1.5e6.
        let mut s2 = BondState::intact();
        let _ = update_bond(&m, X, X * -1.0e-3, Vec3::ZERO, &mut s2);
        let step2 = update_bond(&m, X, shear_dir, Vec3::ZERO, &mut s2);
        assert!(step2.shear_limit > 1.4e6, "friction boosts the limit");
        assert!(!step2.broke, "confined shear below the raised limit");
    }

    #[test]
    fn shear_breaks_above_cohesion() {
        let m = model();
        let mut s = BondState::intact();
        // δs = 2e-3 ⇒ τ = 2e6 ≥ c (1e6) → shear break.
        let step = update_bond(&m, X, Vec3::new(0.0, 2.0e-3, 0.0), Vec3::ZERO, &mut s);
        assert!(step.broke);
        assert!(s.is_broken());
    }

    #[test]
    fn bending_moment_contributes_to_tensile_stress() {
        let m = model();
        let mut s = BondState::intact();
        // Pure bending rotation about y: M_b = kn·I·θ, σ = M_b·R/I = kn·R·θ.
        // θ = 2e-4 ⇒ σ = 1e9·0.01·2e-4 = 2e3 (tiny) — stays intact but nonzero.
        let step = update_bond(&m, X, Vec3::ZERO, Vec3::new(0.0, 2.0e-4, 0.0), &mut s);
        assert!(!step.broke);
        assert!(step.normal_stress > 0.0, "bending adds fibre tension");
        assert!(s.bending_moment().length() > 0.0);
    }

    #[test]
    fn twisting_moment_contributes_to_shear_stress() {
        let m = model();
        let mut s = BondState::intact();
        // Pure twist about x: M_t = ks·J·θ, τ = M_t·R/J = ks·R·θ.
        let step = update_bond(&m, X, Vec3::ZERO, X * 1.0e-4, &mut s);
        assert!(!step.broke);
        assert!(step.shear_stress > 0.0, "twist adds fibre shear");
        assert!(s.twisting_moment().abs() > 0.0);
    }

    #[test]
    fn degenerate_axis_is_a_noop() {
        let m = model();
        let mut s = BondState::intact();
        let step = update_bond(&m, Vec3::ZERO, X * 1.0, Vec3::ZERO, &mut s);
        assert!(!step.broke);
        assert_eq!(s.normal_force(), 0.0);
        assert!(!s.is_broken());
    }

    #[test]
    fn increments_accumulate_across_updates() {
        let m = model();
        let mut s = BondState::intact();
        // Two half-steps of δn = 5e-5 reach the same σ as one 1e-4 step.
        let _ = update_bond(&m, X, X * 5.0e-5, Vec3::ZERO, &mut s);
        let step = update_bond(&m, X, X * 5.0e-5, Vec3::ZERO, &mut s);
        assert!(
            (step.normal_stress - 1.0e5).abs() < 1.0,
            "increments add up"
        );
    }
}
