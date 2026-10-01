//! Dynamic Follow-The-Leader (`DFTL`) inextensible strand integration.
//!
//! This is the strand *fast tier*'s length solver: a single-pass, root-to-tip
//! position propagation that makes a guide strand rigidly inextensible in one
//! sweep, plus the velocity correction that keeps that one-sweep projection
//! from pumping fake energy into the hair. It mirrors Müller, Kim and Chentanez
//! 2012 ("Fast Simulation of Inextensible Hair and Fur", the method `TressFX`
//! popularized) at the algorithm level without reusing any of its code.
//!
//! Plain Follow-The-Leader (`FTL`) walks a strand from its pinned root to its
//! tip and snaps every segment back onto its rest length, which is
//! unconditionally stable and strictly length-preserving regardless of time
//! step — unlike an iterative edge-distance solver, which needs many sweeps and
//! can still stretch under fast motion. The catch is that moving a follower
//! particle to chase its leader injects a spurious velocity; left alone this
//! makes a whipping strand gain energy and "explode". `DFTL` ("Dynamic" `FTL`)
//! fixes exactly that: after the position pass it corrects each particle's
//! velocity by the *follower's* `FTL` displacement, so the strand stays
//! inextensible and energy-bounded at the same time.
//!
//! Each guide is a poly-line of [`FtlParticle`]s in root-to-tip order. The root
//! particle is *pinned* (`inverse_mass == 0`) so it rides the skinned scalp
//! instead of falling; every other particle is free. One
//! [`simulate_strand_dftl`] call advances a strand by `dt`, split into
//! `substeps` semi-implicit integration steps, each performing three phases:
//!
//! 1. **Predict** — every free particle integrates gravity and velocity damping
//!    semi-implicitly to a predicted position.
//! 2. **`FTL` propagation** — from the root toward the tip, each segment is
//!    forced back to its rest length `d_i` via
//!    `p_i = p_{i-1} + d_i * normalize(p_i - p_{i-1})`; a (numerically)
//!    zero-length segment falls back to a fixed safe axis so the result is
//!    never `NaN`. A single sweep is already inextensible.
//! 3. **Velocity correction** — the `FTL` move is undone in velocity space with
//!    Müller's term `v_i += correction * (p_{i+1}^{pre} - p_{i+1}) / dt`, where
//!    `p_{i+1}^{pre}` is the follower's predicted (pre-`FTL`) position and
//!    `p_{i+1}` its post-`FTL` position. The `correction` fraction is in
//!    `0..=1` (typically `0.9`); it transfers the follower's correction back
//!    into the leader's velocity so the strand neither gains energy nor goes
//!    limp.
//!
//! Everything here is deterministic array-in/array-out `CPU` math: identical
//! inputs produce bit-identical outputs, so results can be golden-tested by
//! hand. This module is intentionally self-contained — it defines its own
//! value types rather than sharing [`super::dynamics`]'s, so the two length
//! solvers stay disjoint — while matching that module's conventions (named
//! vector ops, pinned roots via zero inverse mass, defensive companion-slice
//! reads, no transcendental math).

use alloc::vec::Vec;

/// Vectors whose squared length is at or below this are treated as zero-length,
/// so normalization falls back to a safe axis instead of producing `NaN`.
const EPS_LEN_SQ: f32 = 1.0e-24;

/// Fallback step used when parameter sanitation rejects a non-finite or
/// non-positive time step; a benign 60 Hz frame.
const DEFAULT_DT: f32 = 1.0 / 60.0;

/// Default velocity-correction fraction substituted for a non-finite authoring
/// value; the canonical `DFTL` damping of `0.9`.
const DEFAULT_CORRECTION: f32 = 0.9;

/// Deterministic unit direction used when an `FTL` segment is (numerically)
/// zero-length and has no well-defined direction of its own.
const SAFE_AXIS: Vec3 = Vec3::new(1.0, 0.0, 0.0);

/// A minimal 3-component vector for strand math.
///
/// The crate carries no linear-algebra dependency, so this solver defines its
/// own value type. All operations are plain `f32` arithmetic evaluated in a
/// fixed order, which is what makes the solver bit-for-bit reproducible. It is
/// deliberately distinct from [`super::dynamics::Vec3`] to keep the two length
/// solvers independent.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
}

impl Vec3 {
    /// Constructs a vector from its components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// The zero vector.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Component-wise sum.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The strand math API is specified with named add/sub methods for call-site uniformity; operator traits are intentionally not part of this internal type."
    )]
    pub fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }

    /// Component-wise difference `self - rhs`.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "See add: the specified API uses named sub for call-site uniformity, not operator traits."
    )]
    pub fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }

    /// Uniform scale by a scalar.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    /// Dot (inner) product.
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Squared Euclidean length; cheaper than [`Vec3::length`] when only
    /// comparisons are needed.
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Euclidean length.
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Returns the unit vector along `self`, or `fallback` when `self` is
    /// (numerically) the zero vector. The `FTL` pass uses this so a collapsed
    /// segment resolves to a fixed axis instead of dividing by ~0 and yielding
    /// `NaN`.
    #[must_use]
    pub fn direction_or(self, fallback: Vec3) -> Self {
        let len_sq = self.length_squared();
        if len_sq > EPS_LEN_SQ {
            self.scale(1.0 / len_sq.sqrt())
        } else {
            fallback
        }
    }

    /// Returns the unit vector along `self`, or [`Vec3::ZERO`] when `self` is
    /// (numerically) the zero vector, so normalization never yields `NaN`.
    #[must_use]
    pub fn normalize_or_zero(self) -> Self {
        self.direction_or(Vec3::ZERO)
    }

    /// Replaces any non-finite component with `0`, so a poisoned external
    /// vector (gravity, an imported position) cannot smuggle `NaN`/`inf` into
    /// the deterministic solve.
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self::new(
            finite_or(self.x, 0.0),
            finite_or(self.y, 0.0),
            finite_or(self.z, 0.0),
        )
    }

    /// Returns `true` when every component is finite.
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

/// Returns `x` when it is finite, otherwise the supplied `default`.
fn finite_or(x: f32, default: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        default
    }
}

/// Clamps a `0..=1` tuning fraction, substituting `default` for a non-finite
/// authoring value.
fn sanitize_unit(x: f32, default: f32) -> f32 {
    if x.is_finite() {
        x.clamp(0.0, 1.0)
    } else {
        default
    }
}

/// One control point of a guide strand under the `DFTL` solver.
///
/// Velocity is stored explicitly (rather than Verlet-implicitly) because the
/// `DFTL` correction edits velocity directly. A particle with
/// `inverse_mass == 0` (equivalently non-positive) is *pinned*: it is treated
/// as infinitely heavy, never integrated, and never moved by the `FTL` pass,
/// which is how the root rides the skinned scalp.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FtlParticle {
    /// Current world-space position.
    pub position: Vec3,
    /// Current world-space velocity (world units/second).
    pub velocity: Vec3,
    /// Reciprocal mass; `0` (or non-positive) pins the particle in place.
    pub inverse_mass: f32,
}

impl FtlParticle {
    /// Convenience constructor for a free particle of unit inverse mass, at
    /// rest.
    #[must_use]
    pub fn free(position: Vec3) -> Self {
        Self {
            position,
            velocity: Vec3::ZERO,
            inverse_mass: 1.0,
        }
    }

    /// Convenience constructor for a pinned (kinematic) particle, at rest.
    #[must_use]
    pub fn pinned(position: Vec3) -> Self {
        Self {
            position,
            velocity: Vec3::ZERO,
            inverse_mass: 0.0,
        }
    }

    /// Returns `true` when this particle is pinned and must not move.
    #[must_use]
    pub fn is_pinned(&self) -> bool {
        self.inverse_mass <= 0.0
    }
}

/// Tunable parameters for one `DFTL` solve.
///
/// Call [`DftlParams::sanitized`] (done internally by the simulate entry
/// points) to coerce out-of-range or non-finite authoring values onto a stable
/// domain: `dt > 0` and finite, `substeps >= 1`, `damping` and `correction` in
/// `0..=1`, and a finite `gravity`.
#[derive(Clone, Copy, Debug)]
pub struct DftlParams {
    /// Frame time step advanced by one simulate call (seconds).
    pub dt: f32,
    /// Number of semi-implicit integration steps `dt` is split into.
    pub substeps: u32,
    /// Uniform acceleration applied to every free particle (world units/s^2).
    pub gravity: Vec3,
    /// Velocity retention is `1 - damping`; `0` keeps all velocity, `1` removes
    /// it. Models aerodynamic drag / numerical damping. In `0..=1`.
    pub damping: f32,
    /// `DFTL` velocity-correction fraction in `0..=1` (typically `0.9`): how
    /// much of the follower's `FTL` displacement is transferred back into the
    /// leader's velocity. `0` degrades to plain `FTL` (position-only); `1`
    /// cancels the full spurious velocity.
    pub correction: f32,
}

impl Default for DftlParams {
    fn default() -> Self {
        Self {
            dt: DEFAULT_DT,
            substeps: 1,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            damping: 0.0,
            correction: DEFAULT_CORRECTION,
        }
    }
}

impl DftlParams {
    /// Returns a copy with every field coerced onto the stable domain
    /// documented on the struct. A non-finite or non-positive `dt` becomes
    /// [`DEFAULT_DT`], zero `substeps` becomes `1`, `damping`/`correction` are
    /// clamped (non-finite `correction` becomes [`DEFAULT_CORRECTION`],
    /// non-finite `damping` becomes `0`), and `gravity` is made finite.
    #[must_use]
    pub fn sanitized(self) -> Self {
        let dt = if self.dt.is_finite() && self.dt > 0.0 {
            self.dt
        } else {
            DEFAULT_DT
        };
        Self {
            dt,
            substeps: self.substeps.max(1),
            gravity: self.gravity.sanitized(),
            damping: sanitize_unit(self.damping, 0.0),
            correction: sanitize_unit(self.correction, DEFAULT_CORRECTION),
        }
    }
}

/// Advances a single guide strand by `params.dt` with the `DFTL` solver.
///
/// `particles` is the strand's poly-line in root-to-tip order and is updated in
/// place. `rest_lengths[i]` is the target length of the segment between
/// particle `i` and particle `i + 1`, so the slice is expected to hold
/// `particles.len() - 1` entries; it is read defensively with `get`, so a short
/// or mismatched slice simply leaves the missing segments unconstrained for
/// that sweep instead of panicking (any trailing entries are ignored). Negative
/// rest lengths are treated as `0`.
///
/// The call is a no-op when there is nothing to advance: a strand with fewer
/// than two particles (no segment to constrain). Parameters are sanitized
/// internally, so a non-finite `dt`, zero `substeps`, or out-of-range tuning
/// values never panic. Pinned particles are left exactly where the caller
/// placed them (their skinned pose), with their velocity untouched.
pub fn simulate_strand_dftl(
    particles: &mut [FtlParticle],
    rest_lengths: &[f32],
    params: DftlParams,
) {
    let count = particles.len();
    if count < 2 {
        return;
    }

    let params = params.sanitized();
    let sub_dt = params.dt / params.substeps as f32;
    // `sub_dt` is strictly positive here because `dt > 0` and `substeps >= 1`
    // are guaranteed by sanitation, so the reciprocal is finite.
    let inv_sub_dt = 1.0 / sub_dt;
    let velocity_retain = 1.0 - params.damping;
    let gravity_step = params.gravity.scale(sub_dt);

    // Per-substep scratch: the position each particle held at the start of the
    // substep, and its predicted (pre-`FTL`) position. The velocity correction
    // needs both, so they are kept rather than recomputed.
    let mut start_positions: Vec<Vec3> = Vec::new();
    let mut predicted_positions: Vec<Vec3> = Vec::new();
    start_positions.resize(count, Vec3::ZERO);
    predicted_positions.resize(count, Vec3::ZERO);

    let mut step = 0;
    while step < params.substeps {
        predict(
            particles,
            &mut start_positions,
            &mut predicted_positions,
            gravity_step,
            velocity_retain,
            sub_dt,
        );
        propagate_ftl(particles, rest_lengths);
        correct_velocities(
            particles,
            &start_positions,
            &predicted_positions,
            params.correction,
            inv_sub_dt,
        );
        step += 1;
    }
}

/// Phase 1: semi-implicit prediction.
///
/// Records each particle's start position, then advances free particles by
/// damping their velocity, adding the gravity impulse `gravity * sub_dt`, and
/// stepping the position forward by `velocity * sub_dt`; the resulting
/// predicted position is also recorded for the later velocity correction.
/// Pinned particles keep their position and velocity, and their predicted
/// position equals their start position.
fn predict(
    particles: &mut [FtlParticle],
    start_positions: &mut [Vec3],
    predicted_positions: &mut [Vec3],
    gravity_step: Vec3,
    velocity_retain: f32,
    sub_dt: f32,
) {
    let count = particles.len();
    let mut i = 0;
    while i < count {
        let particle = &mut particles[i];
        start_positions[i] = particle.position;
        if particle.is_pinned() {
            predicted_positions[i] = particle.position;
        } else {
            let new_velocity = particle.velocity.scale(velocity_retain).add(gravity_step);
            particle.velocity = new_velocity;
            let predicted = particle.position.add(new_velocity.scale(sub_dt));
            particle.position = predicted;
            predicted_positions[i] = predicted;
        }
        i += 1;
    }
}

/// Phase 2: `FTL` position propagation from root to tip.
///
/// Walks each segment from the (already-corrected) parent toward the child and
/// snaps the child onto the segment's rest length along the current direction.
/// A collapsed segment uses [`SAFE_AXIS`]. Pinned particles are never moved;
/// missing rest-length entries leave that segment unconstrained.
fn propagate_ftl(particles: &mut [FtlParticle], rest_lengths: &[f32]) {
    let count = particles.len();
    let mut i = 1;
    while i < count {
        if particles[i].is_pinned() {
            i += 1;
            continue;
        }
        let Some(&rest) = rest_lengths.get(i - 1) else {
            i += 1;
            continue;
        };
        let parent = particles[i - 1].position;
        let child = particles[i].position;
        let direction = child.sub(parent).direction_or(SAFE_AXIS);
        particles[i].position = parent.add(direction.scale(rest.max(0.0)));
        i += 1;
    }
}

/// Phase 3: `DFTL` velocity correction.
///
/// Each free particle's velocity is set from its net position change over the
/// substep, then corrected by its follower's `FTL` displacement
/// `(predicted_{i+1} - position_{i+1})` scaled by `correction / sub_dt`. The
/// tip has no follower and so receives no correction term. Pinned particles are
/// left untouched.
fn correct_velocities(
    particles: &mut [FtlParticle],
    start_positions: &[Vec3],
    predicted_positions: &[Vec3],
    correction: f32,
    inv_sub_dt: f32,
) {
    let count = particles.len();
    let mut i = 0;
    while i < count {
        if particles[i].is_pinned() {
            i += 1;
            continue;
        }
        let base = particles[i]
            .position
            .sub(start_positions[i])
            .scale(inv_sub_dt);
        let follower_term = if i + 1 < count {
            predicted_positions[i + 1]
                .sub(particles[i + 1].position)
                .scale(correction * inv_sub_dt)
        } else {
            Vec3::ZERO
        };
        particles[i].velocity = base.add(follower_term);
        i += 1;
    }
}

/// Advances many concatenated guide strands in a single deterministic pass.
///
/// `particles` and `rest_lengths` are flat arrays shared by all strands;
/// `strand_lengths[k]` is the particle count of strand `k`, and the strands
/// occupy consecutive ranges in that order. Each strand is solved with
/// [`simulate_strand_dftl`] over its own sub-slice. `rest_lengths` is sliced
/// index-for-index with `particles` (entry `i` is the rest length of the
/// segment leaving particle `i`; each strand's last particle has no outgoing
/// segment and its entry is ignored). A `strand_lengths` entry that would run
/// past the end of `particles` stops the walk, so malformed layouts truncate
/// deterministically instead of panicking.
pub fn simulate_guides_dftl(
    particles: &mut [FtlParticle],
    strand_lengths: &[usize],
    rest_lengths: &[f32],
    params: DftlParams,
) {
    let mut offset = 0usize;
    for &length in strand_lengths {
        let Some(end) = offset.checked_add(length) else {
            break;
        };
        if end > particles.len() {
            break;
        }
        let strand = &mut particles[offset..end];
        let rest = rest_lengths.get(offset..end).unwrap_or(&[]);
        simulate_strand_dftl(strand, rest, params);
        offset = end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Tolerance for approximate float comparisons in the golden tests.
    const EPS: f32 = 1.0e-4;

    /// A no-gravity, single-substep, undamped parameter set with the canonical
    /// correction; the clean base for precise golden checks.
    fn still_params() -> DftlParams {
        DftlParams {
            dt: 0.5,
            substeps: 1,
            gravity: Vec3::ZERO,
            damping: 0.0,
            correction: 0.9,
        }
    }

    #[test]
    fn vec3_helpers_are_correct() {
        let a = Vec3::new(3.0, 4.0, 0.0);
        assert!((a.length() - 5.0).abs() < EPS);
        assert!((a.length_squared() - 25.0).abs() < EPS);
        assert_eq!(Vec3::ZERO.normalize_or_zero(), Vec3::ZERO);
        assert_eq!(Vec3::ZERO.direction_or(SAFE_AXIS), SAFE_AXIS);
        let unit = a.direction_or(SAFE_AXIS);
        assert!((unit.length() - 1.0).abs() < EPS);
        let b = Vec3::new(1.0, 0.0, 2.0);
        assert!((a.dot(b) - 3.0).abs() < EPS);
        assert_eq!(a.add(b), Vec3::new(4.0, 4.0, 2.0));
        assert_eq!(a.sub(b), Vec3::new(2.0, 4.0, -2.0));
        assert_eq!(a.scale(2.0), Vec3::new(6.0, 8.0, 0.0));
    }

    #[test]
    fn segment_lengths_are_preserved_each_sweep() {
        // A strand released horizontally falls, but FTL must keep every segment
        // at its exact rest length after the sweep, for any step size.
        let mut particles = vec![
            FtlParticle::pinned(Vec3::ZERO),
            FtlParticle::free(Vec3::new(1.0, 0.0, 0.0)),
            FtlParticle::free(Vec3::new(2.0, 0.0, 0.0)),
            FtlParticle::free(Vec3::new(3.0, 0.0, 0.0)),
        ];
        let rest = [1.0, 1.0, 1.0];
        let params = DftlParams {
            dt: 1.0 / 60.0,
            substeps: 4,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            damping: 0.0,
            correction: 0.9,
        };
        simulate_strand_dftl(&mut particles, &rest, params);
        for i in 0..rest.len() {
            let seg = particles[i + 1]
                .position
                .sub(particles[i].position)
                .length();
            assert!((seg - rest[i]).abs() < EPS, "segment {i} len {seg}");
        }
    }

    #[test]
    fn pinned_root_never_moves_under_gravity() {
        let mut particles = vec![
            FtlParticle::pinned(Vec3::new(5.0, 7.0, -2.0)),
            FtlParticle::free(Vec3::new(6.0, 7.0, -2.0)),
        ];
        let rest = [1.0];
        let params = DftlParams {
            dt: 1.0 / 60.0,
            substeps: 3,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            damping: 0.0,
            correction: 0.9,
        };
        simulate_strand_dftl(&mut particles, &rest, params);
        assert_eq!(particles[0].position, Vec3::new(5.0, 7.0, -2.0));
        assert_eq!(particles[0].velocity, Vec3::ZERO);
    }

    #[test]
    fn gravity_swings_free_particle_downward() {
        // A horizontal segment under gravity cannot stretch, so the free tip
        // swings down: its y must become negative while length is kept.
        let mut particles = vec![
            FtlParticle::pinned(Vec3::ZERO),
            FtlParticle::free(Vec3::new(1.0, 0.0, 0.0)),
        ];
        let rest = [1.0];
        let params = DftlParams {
            dt: 1.0 / 60.0,
            substeps: 2,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            damping: 0.0,
            correction: 0.9,
        };
        simulate_strand_dftl(&mut particles, &rest, params);
        assert!(particles[1].position.y < 0.0);
        let seg = particles[1].position.sub(particles[0].position).length();
        assert!((seg - 1.0).abs() < EPS);
    }

    #[test]
    fn velocity_correction_transfers_follower_displacement() {
        // Precise golden for the DFTL correction. Root pinned at the origin;
        // p1 already satisfies its segment (so it is not moved by FTL); p2 is
        // flung outward so FTL snaps it back by exactly dt along +x.
        //
        // With dt = 0.5, correction = 0.9:
        //   p2 predicted = (2.5,0,0), FTL -> (2,0,0), displacement -c2 = (0.5,0,0)
        //   p1 velocity  = 0 + 0.9 * (0.5,0,0) / 0.5 = (0.9,0,0)
        //   p2 velocity  = (2,0,0)-(2,0,0))/dt + 0   = 0
        let mut particles = vec![
            FtlParticle::pinned(Vec3::ZERO),
            FtlParticle::free(Vec3::new(1.0, 0.0, 0.0)),
            FtlParticle {
                position: Vec3::new(2.0, 0.0, 0.0),
                velocity: Vec3::new(1.0, 0.0, 0.0),
                inverse_mass: 1.0,
            },
        ];
        let rest = [1.0, 1.0];
        simulate_strand_dftl(&mut particles, &rest, still_params());

        assert_eq!(particles[0].position, Vec3::ZERO);
        assert_eq!(particles[1].position, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(particles[2].position, Vec3::new(2.0, 0.0, 0.0));

        assert!((particles[1].velocity.x - 0.9).abs() < EPS);
        assert!(particles[1].velocity.y.abs() < EPS);
        assert!(particles[1].velocity.z.abs() < EPS);
        assert!(particles[2].velocity.length() < EPS);
    }

    #[test]
    fn correction_bounds_energy_over_many_steps() {
        // A whipping strand must stay inextensible and bounded: with the
        // correction active the tip can never leave the reach sphere, and no
        // value is allowed to blow up to NaN/inf over a long run.
        let make = || {
            vec![
                FtlParticle::pinned(Vec3::ZERO),
                FtlParticle::free(Vec3::new(1.0, 0.0, 0.0)),
                FtlParticle::free(Vec3::new(2.0, 0.0, 0.0)),
                FtlParticle::free(Vec3::new(3.0, 0.0, 0.0)),
            ]
        };
        let rest = [1.0, 1.0, 1.0];
        let total_reach: f32 = rest.iter().sum();
        let mut particles = make();
        let params = DftlParams {
            dt: 1.0 / 60.0,
            substeps: 2,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            damping: 0.0,
            correction: 0.9,
        };
        let mut step = 0;
        while step < 400 {
            simulate_strand_dftl(&mut particles, &rest, params);
            step += 1;
        }
        for p in &particles {
            assert!(p.position.is_finite(), "position went non-finite");
            assert!(p.velocity.is_finite(), "velocity went non-finite");
            // Inextensibility caps the distance from the pinned root.
            assert!(p.position.length() <= total_reach + EPS);
        }
    }

    #[test]
    fn solve_is_deterministic() {
        let make = || {
            vec![
                FtlParticle::pinned(Vec3::ZERO),
                FtlParticle::free(Vec3::new(1.0, 0.0, 0.0)),
                FtlParticle::free(Vec3::new(2.0, 0.1, 0.0)),
            ]
        };
        let rest = [1.0, 1.0];
        let params = DftlParams {
            dt: 1.0 / 90.0,
            substeps: 3,
            gravity: Vec3::new(0.3, -9.81, 0.2),
            damping: 0.05,
            correction: 0.8,
        };
        let mut a = make();
        let mut b = make();
        simulate_strand_dftl(&mut a, &rest, params);
        simulate_strand_dftl(&mut b, &rest, params);
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert_eq!(pa.position, pb.position);
            assert_eq!(pa.velocity, pb.velocity);
        }
    }

    #[test]
    fn zero_length_segment_falls_back_to_safe_axis() {
        // Parent and child coincide: FTL has no direction, must use SAFE_AXIS
        // and never produce NaN.
        let mut particles = vec![
            FtlParticle::pinned(Vec3::ZERO),
            FtlParticle::free(Vec3::ZERO),
        ];
        let rest = [2.0];
        simulate_strand_dftl(&mut particles, &rest, still_params());
        assert_eq!(particles[1].position, SAFE_AXIS.scale(2.0));
        assert!(particles[1].position.is_finite());
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        let params = still_params();
        // Empty strand.
        let mut empty: Vec<FtlParticle> = Vec::new();
        simulate_strand_dftl(&mut empty, &[], params);
        assert!(empty.is_empty());

        // Single particle: no segment, left exactly in place.
        let mut single = vec![FtlParticle::free(Vec3::new(1.0, 2.0, 3.0))];
        simulate_strand_dftl(&mut single, &[], params);
        assert_eq!(single[0].position, Vec3::new(1.0, 2.0, 3.0));

        // Mismatched (too-short) rest-length slice: missing segments are simply
        // left unconstrained rather than panicking.
        let mut strand = vec![
            FtlParticle::pinned(Vec3::ZERO),
            FtlParticle::free(Vec3::new(1.0, 0.0, 0.0)),
            FtlParticle::free(Vec3::new(2.0, 0.0, 0.0)),
        ];
        simulate_strand_dftl(&mut strand, &[1.0], params);
        for p in &strand {
            assert!(p.position.is_finite());
        }
    }

    #[test]
    fn non_finite_params_are_sanitized() {
        let mut particles = vec![
            FtlParticle::pinned(Vec3::new(0.0, 1.0, 0.0)),
            FtlParticle::free(Vec3::new(1.0, 1.0, 0.0)),
        ];
        let rest = [1.0];
        let params = DftlParams {
            dt: f32::NAN,
            substeps: 0,
            gravity: Vec3::new(f32::INFINITY, f32::NAN, 0.0),
            damping: f32::NAN,
            correction: 5.0,
        };
        simulate_strand_dftl(&mut particles, &rest, params);
        // Root unmoved; nothing went non-finite despite the poisoned inputs.
        assert_eq!(particles[0].position, Vec3::new(0.0, 1.0, 0.0));
        for p in &particles {
            assert!(p.position.is_finite());
            assert!(p.velocity.is_finite());
        }
    }

    #[test]
    fn params_sanitized_clamps_domain() {
        let s = DftlParams {
            dt: -1.0,
            substeps: 0,
            gravity: Vec3::new(f32::NAN, 1.0, f32::INFINITY),
            damping: 2.0,
            correction: -3.0,
        }
        .sanitized();
        assert!((s.dt - DEFAULT_DT).abs() < EPS);
        assert_eq!(s.substeps, 1);
        assert_eq!(s.gravity, Vec3::new(0.0, 1.0, 0.0));
        assert!((s.damping - 1.0).abs() < EPS);
        assert!(s.correction.abs() < EPS);
    }

    #[test]
    fn simulate_guides_dftl_solves_each_strand_and_truncates_safely() {
        // Two strands packed flat; the second declared length overruns the
        // buffer and must be dropped without panicking.
        let mut particles = vec![
            // strand 0
            FtlParticle::pinned(Vec3::ZERO),
            FtlParticle::free(Vec3::new(1.0, 0.0, 0.0)),
            // strand 1
            FtlParticle::pinned(Vec3::new(0.0, 10.0, 0.0)),
            FtlParticle::free(Vec3::new(1.0, 10.0, 0.0)),
        ];
        let strand_lengths = [2usize, 2usize, 99usize];
        let rest_lengths = [1.0, 0.0, 1.0, 0.0];
        let params = DftlParams {
            dt: 1.0 / 60.0,
            substeps: 2,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            damping: 0.0,
            correction: 0.9,
        };
        simulate_guides_dftl(&mut particles, &strand_lengths, &rest_lengths, params);
        // Both roots stayed pinned.
        assert_eq!(particles[0].position, Vec3::ZERO);
        assert_eq!(particles[2].position, Vec3::new(0.0, 10.0, 0.0));
        // Each free tip kept its segment length and swung down under gravity.
        let seg0 = particles[1].position.sub(particles[0].position).length();
        let seg1 = particles[3].position.sub(particles[2].position).length();
        assert!((seg0 - 1.0).abs() < EPS);
        assert!((seg1 - 1.0).abs() < EPS);
        assert!(particles[1].position.y < 0.0);
        assert!(particles[3].position.y < 10.0);
    }
}
