//! High-fidelity discrete elastic rod (`DER`) / `Cosserat` rod slot for the
//! hair engine (design doc §8.5 item 1, the highest-complexity sim tier).
//!
//! Pure mass-spring strands (see [`super::dynamics`]) cannot represent the
//! torsional stiffness and natural `helix` rest shape of tight curls and
//! braids: a spring network has no notion of a *material frame* and therefore
//! no twist energy at all. This module adds that missing degree of freedom
//! with a position-and-orientation based `Cosserat` rod, following `Bergou`
//! 2008/2010 for the material-frame bending-plus-twisting energy and
//! `Kugelstadt` 2016 for the position-based `Cosserat` constraint projection.
//!
//! Each rod is an edge-vector poly-line of [`RodParticle`]s paired with an
//! array of quaternion material frames ([`Quat`]). The relative rotation
//! between two adjacent frames is a discrete `Darboux` vector whose three
//! components are the two bending curvatures and the twist; aligning that
//! relative rotation to a non-zero *rest* `Darboux` vector is exactly what
//! encodes a natural `helix` rest (constant curvature plus twist) that a
//! spring network cannot express.
//!
//! The solver is deterministic array-in/array-out `CPU` math (design §7): a
//! semi-implicit prediction, a compliant edge-length (stretch) projection, and
//! an `XPBD` / `PB`-`Cosserat` bend-twist projection that drives each adjacent
//! frame pair toward its rest `Darboux` vector. Identical inputs produce
//! bit-identical outputs, so every result below is hand-checkable in a golden
//! test. This module only *solves*; it builds no budget request and reuses no
//! types from the other hair modules (it defines its own [`Vec3`]/[`Quat`]).
//!
//! ## Trig-free orientation math
//!
//! The crate carries no linear-algebra dependency and bans the `f32`
//! transcendental methods for `libm`-style determinism, so the quaternion math
//! here never calls `sin`/`cos`. Orientation changes integrate as a quaternion
//! derivative, `q' = normalize(q + 0.5 * (omega * q) * dt)` with `omega` a pure
//! angular-velocity quaternion ([`Quat::integrate`]); the minimal rotation
//! taking one unit vector onto another is built from the half-vector identity
//! `q = (dot(a, h), cross(a, h))` with `h = normalize(a + b)`
//! ([`Quat::from_min_rotation`]). The near-antiparallel case (`a` approximately
//! equal to `-b`) falls back to a `180`-degree rotation about an arbitrary
//! orthogonal axis. None of these paths evaluate a trigonometric function, and
//! none can produce a `NaN`.
#![forbid(unsafe_code)]

use alloc::vec::Vec;

/// Squared lengths below this treat a vector as numerically zero, so
/// normalization never divides by (almost) zero.
const EPS_LEN_SQ: f32 = 1.0e-24;
/// Segment lengths below this are skipped to avoid dividing by (almost) zero.
const EPS_LEN: f32 = 1.0e-12;
/// Threshold on `dot(a, b)` of two unit vectors below which
/// [`Quat::from_min_rotation`] treats the inputs as antiparallel and uses the
/// orthogonal-axis fallback instead of the half-vector formula.
const ANTIPARALLEL_DOT: f32 = -1.0 + 1.0e-6;
/// Gauss-Seidel constraint sweeps run per substep. Fixed (not caller-tunable)
/// so the solve stays deterministic regardless of configuration.
const SOLVER_ITERATIONS: u32 = 8;

/// Replaces a non-finite scalar with zero so sanitation cannot leak `NaN` /
/// infinity downstream.
#[inline]
#[must_use]
fn finite_or_zero(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

/// Clamps an `XPBD` compliance (inverse stiffness) to the valid `>= 0` range,
/// mapping any non-finite or negative authoring value to the rigid default `0`.
#[inline]
#[must_use]
fn sanitize_compliance(c: f32) -> f32 {
    if c.is_finite() && c >= 0.0 {
        c
    } else {
        0.0
    }
}

/// A minimal 3-component vector for rod math.
///
/// The crate carries no linear-algebra dependency, so this module defines its
/// own value type. All operations are plain `f32` arithmetic evaluated in a
/// fixed order, which is what makes the solver bit-for-bit reproducible.
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
    /// The zero vector.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Constructs a vector from its components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Component-wise sum.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The rod math API is specified with named add/sub methods for call-site uniformity; operator traits are intentionally not part of this internal type."
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

    /// Cross (vector) product `self x rhs`.
    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
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

    /// Returns the unit vector along `self`, or [`Vec3::ZERO`] when `self` is
    /// (numerically) the zero vector, so normalization never yields `NaN`.
    #[must_use]
    pub fn normalize_or_zero(self) -> Self {
        let len_sq = self.length_squared();
        if len_sq.is_finite() && len_sq > EPS_LEN_SQ {
            self.scale(1.0 / len_sq.sqrt())
        } else {
            Self::ZERO
        }
    }

    /// Returns a copy with every non-finite component replaced by zero.
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self::new(
            finite_or_zero(self.x),
            finite_or_zero(self.y),
            finite_or_zero(self.z),
        )
    }
}

/// A unit quaternion material frame, stored as `w + xi + yj + zk`.
///
/// All rotations in this module are expressed as quaternions and advanced only
/// with multiplication, addition, and normalization, never with `sin`/`cos`,
/// which keeps the solve inside the crate's trig-free determinism contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    /// Real (scalar) part.
    pub w: f32,
    /// `i` component.
    pub x: f32,
    /// `j` component.
    pub y: f32,
    /// `k` component.
    pub z: f32,
}

impl Quat {
    /// The identity rotation.
    pub const IDENTITY: Self = Self {
        w: 1.0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Constructs a quaternion from its raw components (not normalized).
    #[must_use]
    pub const fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        Self { w, x, y, z }
    }

    /// Component-wise sum (used by the derivative integrator and the
    /// position-based bend-twist projection; the result is re-normalized by the
    /// caller).
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The rod math API is specified with named add/sub/mul methods for call-site uniformity; operator traits are intentionally not part of this internal type."
    )]
    pub fn add(self, rhs: Self) -> Self {
        Self::new(
            self.w + rhs.w,
            self.x + rhs.x,
            self.y + rhs.y,
            self.z + rhs.z,
        )
    }

    /// Component-wise difference `self - rhs`.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "See add: the specified API uses named sub for call-site uniformity, not operator traits."
    )]
    pub fn sub(self, rhs: Self) -> Self {
        Self::new(
            self.w - rhs.w,
            self.x - rhs.x,
            self.y - rhs.y,
            self.z - rhs.z,
        )
    }

    /// Uniform scale of all four components by a scalar.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.w * s, self.x * s, self.y * s, self.z * s)
    }

    /// Hamilton product `self * rhs` (rotation composition).
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "See add: the specified API uses named mul for the quaternion product, not the std Mul trait."
    )]
    pub fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        )
    }

    /// The conjugate `w - xi - yj - zk`; equals the inverse for a unit
    /// quaternion.
    #[must_use]
    pub fn conjugate(self) -> Self {
        Self::new(self.w, -self.x, -self.y, -self.z)
    }

    /// Squared norm `w^2 + x^2 + y^2 + z^2`.
    #[must_use]
    pub fn norm_squared(self) -> f32 {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Returns the unit quaternion along `self`, falling back to
    /// [`Quat::IDENTITY`] when `self` is degenerate or non-finite, so
    /// normalization never yields `NaN`.
    #[must_use]
    pub fn normalize(self) -> Self {
        let n2 = self.norm_squared();
        if n2.is_finite() && n2 > EPS_LEN_SQ {
            self.scale(1.0 / n2.sqrt())
        } else {
            Self::IDENTITY
        }
    }

    /// Rotates a vector by this (assumed unit) quaternion via `q * v * q^-1`,
    /// expanded through the Hamilton product so no trig is involved.
    #[must_use]
    pub fn rotate_vec(self, v: Vec3) -> Vec3 {
        let p = Self::new(0.0, v.x, v.y, v.z);
        let r = self.mul(p).mul(self.conjugate());
        Vec3::new(r.x, r.y, r.z)
    }

    /// Builds the minimal rotation taking unit-ish `a` onto unit-ish `b`,
    /// without any trigonometric call.
    ///
    /// Uses the half-vector identity: with `h = normalize(a + b)` the shortest
    /// arc is `q = (dot(a, h), cross(a, h))`, since the angle between `a` and
    /// `h` is half the angle between `a` and `b`. When `a` and `b` are nearly
    /// antiparallel (`a + b` collapses to zero) it falls back to a `180`-degree
    /// rotation about an arbitrary axis orthogonal to `a`. Zero-length inputs
    /// return [`Quat::IDENTITY`].
    #[must_use]
    pub fn from_min_rotation(a: Vec3, b: Vec3) -> Self {
        let na = a.normalize_or_zero();
        let nb = b.normalize_or_zero();
        if na.length_squared() < 0.5 || nb.length_squared() < 0.5 {
            return Self::IDENTITY;
        }
        let d = na.dot(nb);
        if d < ANTIPARALLEL_DOT {
            let axis = orthonormal(na);
            return Self::new(0.0, axis.x, axis.y, axis.z);
        }
        let h = na.add(nb).normalize_or_zero();
        if h.length_squared() < 0.5 {
            let axis = orthonormal(na);
            return Self::new(0.0, axis.x, axis.y, axis.z);
        }
        let w = na.dot(h);
        let v = na.cross(h);
        Self::new(w, v.x, v.y, v.z).normalize()
    }

    /// Advances this orientation by angular velocity `omega` over `dt` using
    /// the quaternion derivative `q' = normalize(q + 0.5 * (omega * q) * dt)`,
    /// where `omega` enters as the pure quaternion `(0, omega)`. The result is
    /// re-normalized, so the output is always a unit quaternion.
    #[must_use]
    pub fn integrate(self, omega: Vec3, dt: f32) -> Self {
        let omega_q = Self::new(0.0, omega.x, omega.y, omega.z);
        let deriv = omega_q.mul(self).scale(0.5 * dt);
        self.add(deriv).normalize()
    }

    /// Returns a finite, normalized copy: non-finite components are zeroed and
    /// the result is renormalized (falling back to [`Quat::IDENTITY`]).
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self::new(
            finite_or_zero(self.w),
            finite_or_zero(self.x),
            finite_or_zero(self.y),
            finite_or_zero(self.z),
        )
        .normalize()
    }
}

/// Returns a unit vector orthogonal to `n` (assumed unit). The reference axis
/// is chosen so the cross product can never collapse to zero.
#[must_use]
fn orthonormal(n: Vec3) -> Vec3 {
    let base = if n.x.abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    n.cross(base).normalize_or_zero()
}

/// Discrete `Darboux` vector between two adjacent material frames: the
/// imaginary part of `conj(a) * b`, i.e. the vector part of the relative
/// rotation expressed in `a`'s local frame. Its components are `(bend1, bend2,
/// twist)`.
#[must_use]
fn darboux(a: Quat, b: Quat) -> Vec3 {
    let rel = a.conjugate().mul(b);
    Vec3::new(rel.x, rel.y, rel.z)
}

/// One control point of a `Cosserat` rod.
///
/// A particle with `inverse_mass <= 0` is *pinned*: it is treated as infinitely
/// heavy, never integrated, and never moved by any constraint, which is how the
/// root rides the skinned scalp.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RodParticle {
    /// Current world-space position.
    pub position: Vec3,
    /// Linear velocity (world units per second).
    pub velocity: Vec3,
    /// Reciprocal mass; non-positive pins the particle in place.
    pub inverse_mass: f32,
}

impl RodParticle {
    /// A free particle of unit inverse mass at `position`, initially at rest.
    #[must_use]
    pub fn free(position: Vec3) -> Self {
        Self {
            position,
            velocity: Vec3::ZERO,
            inverse_mass: 1.0,
        }
    }

    /// A pinned (kinematic) particle at `position`.
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
    pub fn is_pinned(self) -> bool {
        self.inverse_mass <= 0.0
    }

    /// Returns a copy with finite position/velocity and a finite, non-negative
    /// inverse mass (any non-finite or negative inverse mass pins the particle,
    /// which is the safe default).
    #[must_use]
    pub fn sanitized(self) -> Self {
        let inverse_mass = if self.inverse_mass.is_finite() && self.inverse_mass > 0.0 {
            self.inverse_mass
        } else {
            0.0
        };
        Self {
            position: self.position.sanitized(),
            velocity: self.velocity.sanitized(),
            inverse_mass,
        }
    }
}

/// Tunable parameters for one `Cosserat` solve.
///
/// The three `*_compliance` values are `XPBD` inverse stiffnesses in `>= 0`
/// (`0` is perfectly rigid) and are clamped to be non-negative; `damping` is a
/// velocity-retention fraction clamped to `0..=1`.
#[derive(Clone, Copy, Debug)]
pub struct CosseratParams {
    /// Frame time step advanced by one solve call (seconds).
    pub dt: f32,
    /// Number of semi-implicit substeps `dt` is split into (at least `1`).
    pub substeps: u32,
    /// Inverse stiffness of the edge-length (stretch) constraint (`0` = rigid).
    pub stretch_compliance: f32,
    /// Inverse stiffness of the two bending curvatures (`0` = rigid).
    pub bend_compliance: f32,
    /// Inverse stiffness of the twist about the tangent (`0` = rigid).
    pub twist_compliance: f32,
    /// Velocity damping fraction in `0..=1` (`0` keeps all velocity, `1`
    /// removes it) applied when writing velocities back.
    pub damping: f32,
}

impl Default for CosseratParams {
    fn default() -> Self {
        Self {
            dt: 1.0 / 60.0,
            substeps: 8,
            stretch_compliance: 0.0,
            bend_compliance: 0.0,
            twist_compliance: 0.0,
            damping: 0.0,
        }
    }
}

impl CosseratParams {
    /// Returns a copy with every field forced into its valid range: a
    /// non-finite or non-positive `dt` resets to `1/60` and is capped at `1`
    /// second, `substeps` is raised to at least `1`, each compliance is made
    /// finite and non-negative, and `damping` is clamped to `0..=1`.
    #[must_use]
    pub fn sanitized(self) -> Self {
        let dt = if self.dt.is_finite() && self.dt > 0.0 {
            self.dt.min(1.0)
        } else {
            1.0 / 60.0
        };
        let damping = if self.damping.is_finite() {
            self.damping.clamp(0.0, 1.0)
        } else {
            0.0
        };
        Self {
            dt,
            substeps: self.substeps.max(1),
            stretch_compliance: sanitize_compliance(self.stretch_compliance),
            bend_compliance: sanitize_compliance(self.bend_compliance),
            twist_compliance: sanitize_compliance(self.twist_compliance),
            damping,
        }
    }
}

/// Parallel-transports a material frame along a poly-line, returning one frame
/// per segment.
///
/// Starting from `initial`, the first segment's frame is rotated so its local
/// `+z` axis (`d3`) points along the first edge, and each subsequent frame is
/// the previous one rotated by the minimal rotation carrying the previous edge
/// direction onto the current one ([`Quat::from_min_rotation`], a pure
/// cross/dot construction). Degenerate (zero-length) segments reuse the running
/// frame. The result has `points.len() - 1` entries (empty when there are fewer
/// than two points).
#[must_use]
pub fn parallel_transport_frames(points: &[Vec3], initial: Quat) -> Vec<Quat> {
    let mut frames = Vec::new();
    if points.len() < 2 {
        return frames;
    }
    let mut frame = initial.normalize();
    let mut prev_dir: Option<Vec3> = None;
    let mut i = 0;
    while i + 1 < points.len() {
        let dir = points[i + 1].sub(points[i]).normalize_or_zero();
        if dir.length_squared() < 0.5 {
            // Degenerate segment: keep the running frame unchanged.
            frames.push(frame);
            i += 1;
            continue;
        }
        if let Some(pd) = prev_dir {
            let rot = Quat::from_min_rotation(pd, dir);
            frame = rot.mul(frame).normalize();
        } else {
            let d3 = frame.rotate_vec(Vec3::new(0.0, 0.0, 1.0));
            let rot = Quat::from_min_rotation(d3, dir);
            frame = rot.mul(frame).normalize();
        }
        frames.push(frame);
        prev_dir = Some(dir);
        i += 1;
    }
    frames
}

/// Computes the rest `Darboux` vector between every adjacent pair of frames,
/// returning `frames.len() - 1` entries (empty when there are fewer than two
/// frames). This is the companion `rest_darboux` input expected by
/// [`simulate_strand_cosserat`].
#[must_use]
pub fn rest_darboux_from_frames(frames: &[Quat]) -> Vec<Vec3> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 1 < frames.len() {
        out.push(darboux(frames[i], frames[i + 1]));
        i += 1;
    }
    out
}

/// Advances a single `Cosserat` rod by `params.dt` with the position-based
/// solver.
///
/// `particles` is the rod's poly-line in root-to-tip order; `orientations` are
/// the per-segment material frames. `rest_lengths[i]` is the target length of
/// the segment between particle `i` and `i + 1`; `rest_darboux[j]` is the rest
/// relative rotation (as a `Darboux` vector) between frame `j` and frame
/// `j + 1`. Both companion slices are read defensively with `get`, so a short
/// slice simply disables the corresponding constraint for the missing indices
/// (a missing `rest_darboux` entry is treated as a straight `0` rest). All four
/// slices are updated in place.
///
/// Each `params.substeps` substep predicts positions semi-implicitly from
/// velocity, runs a fixed number of Gauss-Seidel sweeps projecting the stretch
/// constraint (positions) and the bend-twist constraint (orientations), then
/// writes velocities back from the position change scaled by the damping
/// retention. Pinned particles (`inverse_mass <= 0`) never move and are left
/// with zero velocity. The call early-returns when there are fewer than two
/// particles; empty, single-particle, mismatched-length, and non-finite inputs
/// are sanitized and never panic.
pub fn simulate_strand_cosserat(
    particles: &mut [RodParticle],
    orientations: &mut [Quat],
    rest_lengths: &[f32],
    rest_darboux: &[Vec3],
    params: CosseratParams,
) {
    let count = particles.len();
    if count < 2 {
        return;
    }
    let params = params.sanitized();

    // Clean incoming state so NaN / infinity cannot propagate through the
    // solve, no matter how malformed the caller's buffers are.
    for p in particles.iter_mut() {
        *p = p.sanitized();
    }
    for q in orientations.iter_mut() {
        *q = q.sanitized();
    }

    let substeps = params.substeps;
    let sub_dt = params.dt / substeps as f32;
    if sub_dt <= 0.0 || !sub_dt.is_finite() {
        return;
    }
    let sub_dt_sq = sub_dt * sub_dt;
    let retain = 1.0 - params.damping;
    // XPBD compliances are normalized by the squared substep so stiffness does
    // not change with the substep count.
    let stretch_alpha = params.stretch_compliance / sub_dt_sq;
    let bend_alpha = params.bend_compliance / sub_dt_sq;
    let twist_alpha = params.twist_compliance / sub_dt_sq;

    let mut prev_positions = alloc::vec![Vec3::ZERO; count];

    for _ in 0..substeps {
        predict(particles, &mut prev_positions, sub_dt);
        for _ in 0..SOLVER_ITERATIONS {
            solve_stretch(particles, rest_lengths, stretch_alpha);
            solve_bend_twist(orientations, rest_darboux, bend_alpha, twist_alpha);
        }
        finalize(particles, &prev_positions, sub_dt, retain);
    }
}

/// Semi-implicit prediction: records each particle's start position and
/// advances free particles by `velocity * sub_dt`. Pinned particles keep their
/// position (their recorded previous position is still their current one).
fn predict(particles: &mut [RodParticle], prev: &mut [Vec3], sub_dt: f32) {
    for (i, p) in particles.iter_mut().enumerate() {
        prev[i] = p.position;
        if p.is_pinned() {
            continue;
        }
        p.position = p.position.add(p.velocity.scale(sub_dt));
    }
}

/// Projects the compliant edge-length (stretch) constraint over every segment
/// once. The correction is split between endpoints by inverse mass, so a pinned
/// endpoint absorbs none of it. Missing rest lengths and degenerate segments
/// are skipped.
fn solve_stretch(particles: &mut [RodParticle], rest_lengths: &[f32], alpha: f32) {
    let count = particles.len();
    let mut i = 0;
    while i + 1 < count {
        let Some(&rest) = rest_lengths.get(i) else {
            i += 1;
            continue;
        };
        if !rest.is_finite() {
            i += 1;
            continue;
        }
        let (head, tail) = particles.split_at_mut(i + 1);
        let a = &mut head[i];
        let b = &mut tail[0];
        let w_a = a.inverse_mass.max(0.0);
        let w_b = b.inverse_mass.max(0.0);
        let w_sum = w_a + w_b;
        if w_sum <= 0.0 {
            i += 1;
            continue;
        }
        let delta = b.position.sub(a.position);
        let len = delta.length();
        if len <= EPS_LEN {
            i += 1;
            continue;
        }
        let dir = delta.scale(1.0 / len);
        let constraint = len - rest;
        let lambda = constraint / (w_sum + alpha);
        a.position = a.position.add(dir.scale(w_a * lambda));
        b.position = b.position.sub(dir.scale(w_b * lambda));
        i += 1;
    }
}

/// Projects the bend-twist constraint over every adjacent orientation pair
/// once.
///
/// For each pair the discrete `Darboux` vector `omega = Im(conj(q_i) * q_j)` is
/// compared against the rest value; the component-wise error is weighted by the
/// `XPBD` compliance (bending for the two tangential components, twist for the
/// component about the tangent) and applied as the quaternion corrections
/// `dq_i = +q_j * (0, lambda)` and `dq_j = -q_i * (0, lambda)` with uniform
/// orientation weights. Both frames are renormalized afterward. A missing rest
/// entry is treated as a straight `0` rest.
fn solve_bend_twist(
    orientations: &mut [Quat],
    rest_darboux: &[Vec3],
    bend_alpha: f32,
    twist_alpha: f32,
) {
    let count = orientations.len();
    // Uniform orientation weights: each frame contributes an inverse inertia of
    // one, so the shared denominator is two plus the compliance.
    let w_sum = 2.0;
    let mut i = 0;
    while i + 1 < count {
        let rest = rest_darboux.get(i).copied().unwrap_or(Vec3::ZERO);
        let (head, tail) = orientations.split_at_mut(i + 1);
        let qi = &mut head[i];
        let qj = &mut tail[0];
        let omega = darboux(*qi, *qj);
        let c = omega.sub(rest);
        let lambda = Quat::new(
            0.0,
            c.x / (w_sum + bend_alpha),
            c.y / (w_sum + bend_alpha),
            c.z / (w_sum + twist_alpha),
        );
        let dqi = (*qj).mul(lambda);
        let dqj = (*qi).mul(lambda);
        *qi = (*qi).add(dqi).normalize();
        *qj = (*qj).sub(dqj).normalize();
        i += 1;
    }
}

/// Writes velocities back from the per-substep position change, scaled by the
/// damping retention. Pinned particles are forced to zero velocity so they stay
/// exactly where the caller placed them.
fn finalize(particles: &mut [RodParticle], prev: &[Vec3], sub_dt: f32, retain: f32) {
    let inv_dt = 1.0 / sub_dt;
    for (i, p) in particles.iter_mut().enumerate() {
        if p.is_pinned() {
            p.velocity = Vec3::ZERO;
            continue;
        }
        p.velocity = p.position.sub(prev[i]).scale(inv_dt).scale(retain);
    }
}

/// Advances many concatenated `Cosserat` rods in a single deterministic pass.
///
/// `particles`, `orientations`, `rest_lengths`, and `rest_darboux` are flat
/// arrays shared by all rods; `strand_lengths[k]` is the particle count of rod
/// `k`, and the rods occupy consecutive ranges in that order (the same
/// offset-slicing discipline used by [`super::dynamics::simulate_guides`]). For
/// a rod of `L` particles the solver consumes `L` particles, `L - 1`
/// orientations and rest lengths, and `L - 2` rest `Darboux` vectors. A
/// `strand_lengths` entry that would run past the end of `particles` stops the
/// walk, so malformed layouts truncate deterministically instead of panicking;
/// companion slices that run short simply disable their constraint for the
/// missing indices.
pub fn simulate_guides_cosserat(
    particles: &mut [RodParticle],
    orientations: &mut [Quat],
    strand_lengths: &[usize],
    rest_lengths: &[f32],
    rest_darboux: &[Vec3],
    params: CosseratParams,
) {
    let mut p_off = 0usize;
    let mut o_off = 0usize;
    let mut rl_off = 0usize;
    let mut rd_off = 0usize;
    for &length in strand_lengths {
        let Some(p_end) = p_off.checked_add(length) else {
            break;
        };
        if p_end > particles.len() {
            break;
        }
        let segments = length.saturating_sub(1);
        let o_end = o_off.saturating_add(segments);
        let rl_end = rl_off.saturating_add(segments);
        let rd_end = rd_off.saturating_add(segments.saturating_sub(1));

        let strand = &mut particles[p_off..p_end];
        let ori = orientations.get_mut(o_off..o_end).unwrap_or(&mut []);
        let rest = rest_lengths.get(rl_off..rl_end).unwrap_or(&[]);
        let darb = rest_darboux.get(rd_off..rd_end).unwrap_or(&[]);
        simulate_strand_cosserat(strand, ori, rest, darb, params);

        p_off = p_end;
        o_off = o_end;
        rl_off = rl_end;
        rd_off = rd_end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Tolerance for approximate float comparisons in the golden tests.
    const EPS: f32 = 1.0e-3;

    /// Returns `true` when two scalars are equal within [`EPS`].
    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    /// Builds the rest relative rotation whose imaginary part is `omega`
    /// (requires `|omega| < 1`), used to synthesize helix / twist rest frames
    /// without any trig.
    fn rel_from_omega(omega: Vec3) -> Quat {
        let w = (1.0 - omega.length_squared()).sqrt();
        Quat::new(w, omega.x, omega.y, omega.z)
    }

    #[test]
    fn vec3_algebra_exact_values() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        assert!(close(x.dot(y), 0.0));
        assert!(close(x.dot(x), 1.0));
        let z = x.cross(y);
        assert!(close(z.x, 0.0) && close(z.y, 0.0) && close(z.z, 1.0));
        let s = x.add(y).scale(2.0);
        assert!(close(s.x, 2.0) && close(s.y, 2.0) && close(s.z, 0.0));
        assert!(close(Vec3::new(3.0, 4.0, 0.0).length(), 5.0));
    }

    #[test]
    fn quat_mul_identity_and_inverse() {
        // 90-degree rotation about z.
        let q = Quat::new(0.707_106_77, 0.0, 0.0, 0.707_106_77);
        let same = Quat::IDENTITY.mul(q);
        assert!(close(same.w, q.w) && close(same.z, q.z));
        // q * conj(q) must be the identity for a unit quaternion.
        let id = q.mul(q.conjugate());
        assert!(close(id.w, 1.0) && close(id.x, 0.0) && close(id.y, 0.0) && close(id.z, 0.0));
    }

    #[test]
    fn quat_rotate_vec_identity_and_flip() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let same = Quat::IDENTITY.rotate_vec(v);
        assert!(close(same.x, 1.0) && close(same.y, 2.0) && close(same.z, 3.0));
        // 180 degrees about z maps (1,0,0) to (-1,0,0).
        let flip = Quat::new(0.0, 0.0, 0.0, 1.0);
        let r = flip.rotate_vec(Vec3::new(1.0, 0.0, 0.0));
        assert!(close(r.x, -1.0) && close(r.y, 0.0) && close(r.z, 0.0));
    }

    #[test]
    fn from_min_rotation_maps_a_onto_b() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let q = Quat::from_min_rotation(a, b);
        let r = q.rotate_vec(a);
        assert!(close(r.x, b.x) && close(r.y, b.y) && close(r.z, b.z));
        // The result must be a unit quaternion.
        assert!(close(q.norm_squared(), 1.0));
    }

    #[test]
    fn integrate_preserves_unit_norm() {
        let q = Quat::IDENTITY.integrate(Vec3::new(0.0, 0.0, 1.0), 0.1);
        assert!(close(q.norm_squared(), 1.0));
        // A larger step must still produce a unit quaternion.
        let q2 = Quat::new(0.0, 1.0, 0.0, 0.0).integrate(Vec3::new(2.0, -1.0, 0.5), 0.25);
        assert!(close(q2.norm_squared(), 1.0));
    }

    #[test]
    fn straight_rod_without_bending_is_static() {
        let start = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        let mut particles = vec![
            RodParticle::free(start[0]),
            RodParticle::free(start[1]),
            RodParticle::free(start[2]),
        ];
        let mut orientations = vec![Quat::IDENTITY, Quat::IDENTITY];
        let rest_lengths = [1.0, 1.0];
        let rest_darboux = [Vec3::ZERO];
        let params = CosseratParams::default();
        for _ in 0..10 {
            simulate_strand_cosserat(
                &mut particles,
                &mut orientations,
                &rest_lengths,
                &rest_darboux,
                params,
            );
        }
        for (p, s) in particles.iter().zip(start.iter()) {
            assert!(
                close(p.position.x, s.x) && close(p.position.y, s.y) && close(p.position.z, s.z)
            );
        }
    }

    #[test]
    fn helix_rest_is_a_fixed_point() {
        let omega0 = Vec3::new(0.1, 0.0, 0.05);
        let rel = rel_from_omega(omega0);
        let q0 = Quat::IDENTITY;
        let q1 = q0.mul(rel).normalize();
        let q2 = q1.mul(rel).normalize();
        let q3 = q2.mul(rel).normalize();
        let mut orientations = vec![q0, q1, q2, q3];
        let rest_darboux = [omega0, omega0, omega0];
        let mut particles = vec![
            RodParticle::free(Vec3::ZERO),
            RodParticle::free(Vec3::new(1.0, 0.0, 0.0)),
        ];
        let rest_lengths = [1.0];
        let params = CosseratParams {
            substeps: 4,
            ..Default::default()
        };
        simulate_strand_cosserat(
            &mut particles,
            &mut orientations,
            &rest_lengths,
            &rest_darboux,
            params,
        );
        let d = darboux(orientations[0], orientations[1]);
        assert!(close(d.x, omega0.x) && close(d.y, omega0.y) && close(d.z, omega0.z));
    }

    #[test]
    fn bending_constraint_pulls_frames_to_rest_curvature() {
        // Start straight, request a helix rest; the bend-twist projection must
        // converge the adjacent darboux vector toward the rest value.
        let omega0 = Vec3::new(0.08, -0.04, 0.0);
        let mut orientations = vec![Quat::IDENTITY, Quat::IDENTITY];
        let rest_darboux = [omega0];
        let mut particles = vec![
            RodParticle::free(Vec3::ZERO),
            RodParticle::free(Vec3::new(1.0, 0.0, 0.0)),
        ];
        let rest_lengths = [1.0];
        let params = CosseratParams {
            substeps: 20,
            ..Default::default()
        };
        simulate_strand_cosserat(
            &mut particles,
            &mut orientations,
            &rest_lengths,
            &rest_darboux,
            params,
        );
        let d = darboux(orientations[0], orientations[1]);
        assert!(close(d.x, omega0.x) && close(d.y, omega0.y) && close(d.z, omega0.z));
    }

    #[test]
    fn pure_twist_is_conserved() {
        let omega0 = Vec3::new(0.0, 0.0, 0.2);
        let rel = rel_from_omega(omega0);
        let q0 = Quat::IDENTITY;
        let q1 = q0.mul(rel).normalize();
        let q2 = q1.mul(rel).normalize();
        let mut orientations = vec![q0, q1, q2];
        let rest_darboux = [omega0, omega0];
        let mut particles = vec![
            RodParticle::free(Vec3::ZERO),
            RodParticle::free(Vec3::new(1.0, 0.0, 0.0)),
        ];
        let rest_lengths = [1.0];
        let params = CosseratParams::default();
        for _ in 0..5 {
            simulate_strand_cosserat(
                &mut particles,
                &mut orientations,
                &rest_lengths,
                &rest_darboux,
                params,
            );
        }
        let d = darboux(orientations[1], orientations[2]);
        assert!(close(d.x, 0.0) && close(d.y, 0.0) && close(d.z, omega0.z));
    }

    #[test]
    fn pinned_root_never_moves() {
        let root = Vec3::new(1.0, 2.0, 3.0);
        let mut particles = vec![
            RodParticle::pinned(root),
            RodParticle::free(Vec3::new(2.0, 2.0, 3.0)),
        ];
        particles[1].velocity = Vec3::new(0.0, -5.0, 1.0);
        let mut orientations = vec![Quat::IDENTITY];
        let rest_lengths = [1.0];
        let rest_darboux: [Vec3; 0] = [];
        let params = CosseratParams::default();
        for _ in 0..20 {
            simulate_strand_cosserat(
                &mut particles,
                &mut orientations,
                &rest_lengths,
                &rest_darboux,
                params,
            );
        }
        let p = particles[0].position;
        assert!(close(p.x, root.x) && close(p.y, root.y) && close(p.z, root.z));
    }

    #[test]
    fn edge_lengths_hold_under_external_velocity() {
        let mut particles = vec![
            RodParticle::pinned(Vec3::ZERO),
            RodParticle::free(Vec3::new(1.0, 0.0, 0.0)),
            RodParticle::free(Vec3::new(2.0, 0.0, 0.0)),
        ];
        // Push the free particles hard; a rigid rod must not stretch.
        particles[1].velocity = Vec3::new(0.0, -8.0, 0.0);
        particles[2].velocity = Vec3::new(0.0, -8.0, 0.0);
        let mut orientations = vec![Quat::IDENTITY, Quat::IDENTITY];
        let rest_lengths = [1.0, 1.0];
        let rest_darboux = [Vec3::ZERO];
        let params = CosseratParams {
            substeps: 8,
            ..Default::default()
        };
        for _ in 0..30 {
            simulate_strand_cosserat(
                &mut particles,
                &mut orientations,
                &rest_lengths,
                &rest_darboux,
                params,
            );
        }
        let e0 = particles[1].position.sub(particles[0].position).length();
        let e1 = particles[2].position.sub(particles[1].position).length();
        assert!((e0 - 1.0).abs() < 1.0e-2, "edge0 = {e0}");
        assert!((e1 - 1.0).abs() < 1.0e-2, "edge1 = {e1}");
    }

    #[test]
    fn many_steps_stay_bounded_and_finite() {
        let mut particles = vec![
            RodParticle::pinned(Vec3::ZERO),
            RodParticle::free(Vec3::new(1.0, 0.0, 0.0)),
            RodParticle::free(Vec3::new(2.0, 0.0, 0.0)),
            RodParticle::free(Vec3::new(3.0, 0.0, 0.0)),
        ];
        particles[2].velocity = Vec3::new(0.0, -3.0, 1.0);
        let mut orientations = vec![Quat::IDENTITY, Quat::IDENTITY, Quat::IDENTITY];
        let rest_lengths = [1.0, 1.0, 1.0];
        let rest_darboux = [Vec3::ZERO, Vec3::ZERO];
        let params = CosseratParams {
            damping: 0.1,
            ..Default::default()
        };
        for _ in 0..200 {
            simulate_strand_cosserat(
                &mut particles,
                &mut orientations,
                &rest_lengths,
                &rest_darboux,
                params,
            );
        }
        for p in &particles {
            assert!(
                p.position.x.is_finite() && p.position.y.is_finite() && p.position.z.is_finite()
            );
            assert!(
                p.velocity.x.is_finite() && p.velocity.y.is_finite() && p.velocity.z.is_finite()
            );
            assert!(p.position.length() < 100.0);
        }
        for q in &orientations {
            assert!(close(q.norm_squared(), 1.0));
        }
    }

    #[test]
    fn two_runs_are_bit_identical() {
        fn run() -> (Vec<RodParticle>, Vec<Quat>) {
            let mut particles = vec![
                RodParticle::pinned(Vec3::ZERO),
                RodParticle::free(Vec3::new(1.0, 0.0, 0.0)),
                RodParticle::free(Vec3::new(2.0, 0.0, 0.0)),
            ];
            particles[1].velocity = Vec3::new(0.3, -2.0, 0.7);
            let mut orientations = vec![Quat::IDENTITY, Quat::IDENTITY];
            let rest_lengths = [1.0, 1.0];
            let rest_darboux = [Vec3::new(0.05, 0.0, 0.1)];
            let params = CosseratParams {
                substeps: 6,
                ..Default::default()
            };
            for _ in 0..25 {
                simulate_strand_cosserat(
                    &mut particles,
                    &mut orientations,
                    &rest_lengths,
                    &rest_darboux,
                    params,
                );
            }
            (particles, orientations)
        }
        let (pa, oa) = run();
        let (pb, ob) = run();
        assert_eq!(pa.len(), pb.len());
        assert_eq!(oa.len(), ob.len());
        for (a, b) in pa.iter().zip(pb.iter()) {
            assert_eq!(a.position.x.to_bits(), b.position.x.to_bits());
            assert_eq!(a.position.y.to_bits(), b.position.y.to_bits());
            assert_eq!(a.position.z.to_bits(), b.position.z.to_bits());
            assert_eq!(a.velocity.x.to_bits(), b.velocity.x.to_bits());
            assert_eq!(a.velocity.y.to_bits(), b.velocity.y.to_bits());
            assert_eq!(a.velocity.z.to_bits(), b.velocity.z.to_bits());
        }
        for (a, b) in oa.iter().zip(ob.iter()) {
            assert_eq!(a.w.to_bits(), b.w.to_bits());
            assert_eq!(a.x.to_bits(), b.x.to_bits());
            assert_eq!(a.y.to_bits(), b.y.to_bits());
            assert_eq!(a.z.to_bits(), b.z.to_bits());
        }
    }

    #[test]
    fn empty_single_and_mismatched_inputs_do_not_panic() {
        let params = CosseratParams::default();
        // Empty.
        let mut empty: Vec<RodParticle> = Vec::new();
        let mut no_ori: Vec<Quat> = Vec::new();
        simulate_strand_cosserat(&mut empty, &mut no_ori, &[], &[], params);
        assert_eq!(empty.len(), 0);
        // Single particle: early return.
        let mut one = vec![RodParticle::free(Vec3::ZERO)];
        simulate_strand_cosserat(&mut one, &mut no_ori, &[], &[], params);
        assert_eq!(one.len(), 1);
        // Mismatched companion lengths (short rests, long orientations).
        let mut particles = vec![
            RodParticle::free(Vec3::ZERO),
            RodParticle::free(Vec3::new(1.0, 0.0, 0.0)),
            RodParticle::free(Vec3::new(2.0, 0.0, 0.0)),
        ];
        let mut orientations = vec![Quat::IDENTITY; 5];
        let rest_lengths = [1.0];
        simulate_strand_cosserat(
            &mut particles,
            &mut orientations,
            &rest_lengths,
            &[],
            params,
        );
        assert_eq!(particles.len(), 3);
        assert_eq!(orientations.len(), 5);
    }

    #[test]
    fn non_finite_inputs_are_sanitized_to_finite_output() {
        let mut particles = vec![
            RodParticle {
                position: Vec3::new(f32::NAN, 0.0, 0.0),
                velocity: Vec3::new(f32::INFINITY, 0.0, 0.0),
                inverse_mass: f32::NAN,
            },
            RodParticle {
                position: Vec3::new(1.0, f32::NEG_INFINITY, 0.0),
                velocity: Vec3::ZERO,
                inverse_mass: -1.0,
            },
            RodParticle::free(Vec3::new(2.0, 0.0, 0.0)),
        ];
        let mut orientations = vec![
            Quat::new(f32::NAN, 0.0, 0.0, 0.0),
            Quat::new(0.0, f32::INFINITY, 0.0, 0.0),
        ];
        let rest_lengths = [f32::NAN, 1.0];
        let rest_darboux = [Vec3::new(f32::INFINITY, 0.0, 0.0)];
        let params = CosseratParams {
            dt: f32::NAN,
            substeps: 0,
            stretch_compliance: -1.0,
            bend_compliance: f32::INFINITY,
            twist_compliance: f32::NAN,
            damping: f32::NAN,
        };
        for _ in 0..5 {
            simulate_strand_cosserat(
                &mut particles,
                &mut orientations,
                &rest_lengths,
                &rest_darboux,
                params,
            );
        }
        for p in &particles {
            assert!(
                p.position.x.is_finite() && p.position.y.is_finite() && p.position.z.is_finite()
            );
            assert!(
                p.velocity.x.is_finite() && p.velocity.y.is_finite() && p.velocity.z.is_finite()
            );
        }
        for q in &orientations {
            assert!(close(q.norm_squared(), 1.0));
        }
    }

    #[test]
    fn antiparallel_parallel_transport_stays_finite() {
        // Minimal rotation between antiparallel vectors must flip without NaN.
        let q = Quat::from_min_rotation(Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
        assert!(close(q.norm_squared(), 1.0));
        let r = q.rotate_vec(Vec3::new(1.0, 0.0, 0.0));
        assert!(close(r.x, -1.0) && close(r.y, 0.0) && close(r.z, 0.0));
        // A poly-line that doubles back exercises the degenerate transport path.
        let points = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
        ];
        let frames = parallel_transport_frames(&points, Quat::IDENTITY);
        assert_eq!(frames.len(), 2);
        for f in &frames {
            assert!(close(f.norm_squared(), 1.0));
        }
    }

    #[test]
    fn guides_batch_truncates_without_panic() {
        let mut particles = vec![
            RodParticle::pinned(Vec3::ZERO),
            RodParticle::free(Vec3::new(1.0, 0.0, 0.0)),
            RodParticle::free(Vec3::new(2.0, 0.0, 0.0)),
            RodParticle::free(Vec3::new(3.0, 0.0, 0.0)),
            RodParticle::free(Vec3::new(4.0, 0.0, 0.0)),
        ];
        let mut orientations = vec![Quat::IDENTITY; 3];
        // Second strand (length 10) runs past the end and must stop the walk.
        let strand_lengths = [3usize, 10usize];
        let rest_lengths = [1.0, 1.0];
        let rest_darboux = [Vec3::ZERO];
        let params = CosseratParams::default();
        simulate_guides_cosserat(
            &mut particles,
            &mut orientations,
            &strand_lengths,
            &rest_lengths,
            &rest_darboux,
            params,
        );
        assert_eq!(particles.len(), 5);
        // Root of the first strand stayed pinned.
        let root = particles[0].position;
        assert!(close(root.x, 0.0) && close(root.y, 0.0) && close(root.z, 0.0));
    }
}
