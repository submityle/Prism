//! Built-in force library: the deterministic `CPU` reference for the §8.2 force
//! kernels (attractors, radial blasts, orbits, gravity wells, drags, turbulence,
//! spring anchors) plus a `ForceField` accumulator that composes them.
//!
//! This module is deliberately orthogonal to its two siblings:
//!
//! * [`super::simulation`] owns the *core integrators* (semi-implicit Euler /
//!   Verlet / `RK2`) and the *primitive* forces — [`gravity`], [`linear_drag`],
//!   [`wind`], [`vortex`], and the divergence-free [`curl_noise`] field. This
//!   file never re-derives those primitives; it `use`s them directly and only
//!   *adds* the forces `simulation` does not provide (point / line attractors,
//!   radial blasts, orbital motion, gravity wells, quadratic drag, layered
//!   turbulence, spring-damper anchors) together with a composition layer.
//! * [`super::boids`] owns *steering* behaviors (separation, alignment,
//!   cohesion, goal seeking). Steering is an intentful control law over a
//!   neighborhood; the forces here are pure world-space physics fields with no
//!   perception, no neighbor query, and no arrival/seek logic. Nothing in this
//!   file duplicates boids steering.
//!
//! Determinism rules (design §29) are inherited: only `sqrt` (through [`Vec3`])
//! and `f32::abs` / `f32::floor` are used; no transcendental (`sin` / `cos` /
//! `acos` / `exp` / `pow`) is ever called, so a future `GPU` kernel can
//! reproduce these results bit for bit. Any angular quantity is taken as a
//! numeric input (for example a precomputed `cos_theta`) rather than computed
//! here, and multi-octave frequency / amplitude ladders take their per-layer
//! multipliers as inputs rather than deriving them.

use alloc::vec::Vec;

use super::simulation::{curl_noise, gravity, linear_drag, vortex, wind};
use super::Vec3;

/// Absolute floating-point comparison tolerance for this module.
///
/// Bare `==` / `!=` on `f32` is avoided throughout; scalar magnitudes are
/// compared against `EPS` and squared lengths against `EPS * EPS`, mirroring the
/// `EPS_LEN_SQ` convention used by [`super::simulation`].
pub const EPS: f32 = 1.0e-6;

/// Distance-based shaping curves shared by the radius-limited forces.
///
/// Each variant maps a normalized distance ratio `t = distance / radius` (in
/// `[0, 1]`) to a multiplier in `(0, 1]`, with `t = 0` at the force's center.
/// This is a field-less enum, so it may safely derive `Eq` and `Hash`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Falloff {
    /// Flat response: full strength everywhere inside the radius.
    Constant,
    /// Linear ramp `1 - t`, reaching zero exactly at the radius.
    Linear,
    /// Smoothstep ramp with zero slope at both ends (soft cutoff).
    Smoothstep,
    /// Windowed inverse-square `1 / (1 + t^2)`: strong near the center, `0.5`
    /// at the radius, never singular because the denominator starts at one.
    InverseSquare,
}

impl Falloff {
    /// Evaluates the shaping multiplier for a normalized distance `ratio`.
    ///
    /// The ratio is clamped to `[0, 1]`, so callers may pass an unclamped
    /// `distance / radius`. Returns `1.0` at the center for every variant.
    #[must_use]
    pub fn shape(self, ratio: f32) -> f32 {
        let t = ratio.clamp(0.0, 1.0);
        match self {
            Falloff::Constant => 1.0,
            Falloff::Linear => 1.0 - t,
            Falloff::Smoothstep => {
                let s = 1.0 - t;
                s * s * (3.0 - 2.0 * s)
            }
            Falloff::InverseSquare => 1.0 / (1.0 + t * t),
        }
    }
}

/// A softened unit direction from `origin` toward `target`.
///
/// Returns the direction together with the true (un-softened) distance. The
/// direction magnitude is `distance / sqrt(distance^2 + softening^2)`, which is
/// `~1` far away and smoothly collapses to zero at the center, so a force built
/// on it has no singularity (design §8.2 softening `r^2 + eps^2`). A negative
/// or zero `softening` still gets an `EPS`-scale floor so the divisor is finite.
fn softened_direction(origin: Vec3, target: Vec3, softening: f32) -> (Vec3, f32) {
    let to = target.sub(origin);
    let dist_sq = to.length_squared();
    let soft = softening.max(0.0);
    let denom = dist_sq + soft * soft + EPS * EPS;
    let inv = 1.0 / denom.sqrt();
    (to.scale(inv), dist_sq.sqrt())
}

/// Returns `true` when a hard influence radius rejects the given distance.
///
/// A non-positive `radius` (`<= EPS`) means "unbounded", so the gate never
/// rejects; otherwise points strictly beyond `radius` are cut off.
fn outside_radius(distance: f32, radius: f32) -> bool {
    radius > EPS && distance > radius
}

/// Acceleration pulling a particle toward (or pushing it from) a single point.
///
/// A positive `strength` attracts and a negative `strength` repels. The pull is
/// shaped by `falloff` over the influence `radius` and softened by `softening`
/// so it stays finite at the point itself. A `radius <= EPS` is treated as an
/// unbounded field (`falloff` still shapes it, using the raw distance clamped to
/// the radius, which for an unbounded field is simply full `Constant` strength).
/// Points beyond a finite radius receive zero force.
#[must_use]
pub fn point_attractor(
    position: Vec3,
    point: Vec3,
    strength: f32,
    radius: f32,
    softening: f32,
    falloff: Falloff,
) -> Vec3 {
    let (dir, dist) = softened_direction(position, point, softening);
    if outside_radius(dist, radius) {
        return Vec3::ZERO;
    }
    let ratio = if radius > EPS { dist / radius } else { 0.0 };
    dir.scale(strength * falloff.shape(ratio))
}

/// Acceleration pulling a particle toward the closest point on a line.
///
/// The line passes through `line_point` with (not necessarily unit) direction
/// `line_direction`; the closest point is the orthogonal projection of
/// `position` onto that infinite line. The influence `radius` gates on the
/// *perpendicular* distance to the line. A degenerate (near-zero) direction
/// collapses to a [`point_attractor`] about `line_point`.
#[must_use]
pub fn line_attractor(
    position: Vec3,
    line_point: Vec3,
    line_direction: Vec3,
    strength: f32,
    radius: f32,
    softening: f32,
    falloff: Falloff,
) -> Vec3 {
    let axis = line_direction.normalize_or_zero();
    if axis.length_squared() <= EPS * EPS {
        return point_attractor(position, line_point, strength, radius, softening, falloff);
    }
    let offset = position.sub(line_point);
    let along = axis.scale(offset.dot(axis));
    let closest = line_point.add(along);
    point_attractor(position, closest, strength, radius, softening, falloff)
}

/// Radial acceleration away from (or toward) a center: the explosion primitive.
///
/// A positive `strength` blasts particles outward (explosion); a negative
/// `strength` implodes them toward the center. Shaping and softening match
/// [`point_attractor`]; the direction is simply reversed (center → particle).
#[must_use]
pub fn radial_force(
    position: Vec3,
    center: Vec3,
    strength: f32,
    radius: f32,
    softening: f32,
    falloff: Falloff,
) -> Vec3 {
    // Outward is the negative of the attractor's inward pull.
    point_attractor(position, center, -strength, radius, softening, falloff)
}

/// Convenience wrapper for an outward blast (`strength` is taken as positive).
///
/// Equivalent to [`radial_force`] with `strength.abs()`, documenting intent at
/// call sites that author an explosion.
#[must_use]
pub fn explosion(
    position: Vec3,
    center: Vec3,
    strength: f32,
    radius: f32,
    softening: f32,
    falloff: Falloff,
) -> Vec3 {
    radial_force(position, center, strength.abs(), radius, softening, falloff)
}

/// Convenience wrapper for an inward collapse (`strength` is taken as positive).
///
/// Equivalent to [`radial_force`] with `-strength.abs()`, documenting intent at
/// call sites that author an implosion.
#[must_use]
pub fn implosion(
    position: Vec3,
    center: Vec3,
    strength: f32,
    radius: f32,
    softening: f32,
    falloff: Falloff,
) -> Vec3 {
    radial_force(
        position,
        center,
        -strength.abs(),
        radius,
        softening,
        falloff,
    )
}

/// Orbital acceleration that holds a target orbit radius about an axis.
///
/// Unlike [`vortex`] (whose tangential push falls off with distance and lets
/// particles drift outward), this force adds a radial spring toward
/// `target_radius`, so particles settle into and *maintain* a circular orbit in
/// the plane through `center` perpendicular to `axis`.
///
/// * `tangential_strength` sets the along-orbit acceleration (its sign selects
///   the orbit direction).
/// * `radial_stiffness` sets how firmly the orbit radius is corrected.
///
/// A particle exactly on the axis has no defined orbital plane and receives no
/// force.
#[must_use]
pub fn orbital_force(
    position: Vec3,
    center: Vec3,
    axis: Vec3,
    target_radius: f32,
    tangential_strength: f32,
    radial_stiffness: f32,
) -> Vec3 {
    let unit_axis = axis.normalize_or_zero();
    if unit_axis.length_squared() <= EPS * EPS {
        return Vec3::ZERO;
    }
    let radial = position.sub(center);
    let axial = unit_axis.scale(radial.dot(unit_axis));
    let perp = radial.sub(axial);
    let dist_sq = perp.length_squared();
    if dist_sq <= EPS * EPS {
        // On the axis: no orbital plane is defined.
        return Vec3::ZERO;
    }
    let dist = dist_sq.sqrt();
    let perp_unit = perp.normalize_or_zero();
    let tangent = unit_axis.cross(perp_unit).normalize_or_zero();
    let tangential = tangent.scale(tangential_strength);
    // Radial spring pulling the orbit back to `target_radius`.
    let radial_error = dist - target_radius;
    let correction = perp_unit.scale(-radial_stiffness * radial_error);
    tangential.add(correction)
}

/// Inverse-square gravitational acceleration toward a center, with softening.
///
/// The magnitude is `strength / (distance^2 + softening^2)`, the classic
/// Plummer-softened well that stays finite at the center (design §8.2). A
/// positive `strength` attracts; a negative `strength` produces an anti-gravity
/// push. This is the pure physical well; use [`point_attractor`] when a shaped,
/// radius-limited pull is wanted instead.
#[must_use]
pub fn gravity_well(position: Vec3, center: Vec3, strength: f32, softening: f32) -> Vec3 {
    let to = center.sub(position);
    let dist_sq = to.length_squared();
    let soft = softening.max(0.0);
    let denom = dist_sq + soft * soft + EPS * EPS;
    let dir = to.normalize_or_zero();
    dir.scale(strength / denom)
}

/// Quadratic (form) drag acceleration `-k * |v| * v`.
///
/// Distinct from the linear (Stokes) [`linear_drag`] in [`super::simulation`]:
/// the retarding force grows with the *square* of speed, dominating at high
/// velocity as real aerodynamic drag does. A non-positive `coefficient` yields
/// no drag; a particle at rest feels nothing.
#[must_use]
pub fn quadratic_drag(velocity: Vec3, coefficient: f32) -> Vec3 {
    if coefficient <= 0.0 {
        return Vec3::ZERO;
    }
    let speed = velocity.length();
    if speed <= EPS {
        return Vec3::ZERO;
    }
    velocity.scale(-coefficient * speed)
}

/// Layered (fractal) turbulence built from the divergence-free [`curl_noise`].
///
/// Sums `octaves` samples of the curl-noise field. Layer zero samples at
/// `frequency` with weight `amplitude`; each subsequent layer multiplies the
/// frequency by `frequency_multiplier` and the amplitude by
/// `amplitude_multiplier` (both supplied as numeric inputs, not derived here).
/// Higher frequencies are decorrelated by offsetting the noise `seed` per layer.
/// The result stays divergence-free because a sum of divergence-free fields is
/// itself divergence-free (design §8, §10).
#[must_use]
pub fn turbulence(
    position: Vec3,
    seed: u32,
    frequency: f32,
    amplitude: f32,
    octaves: u32,
    frequency_multiplier: f32,
    amplitude_multiplier: f32,
    epsilon: f32,
) -> Vec3 {
    let mut total = Vec3::ZERO;
    let mut freq = frequency;
    let mut amp = amplitude;
    for octave in 0..octaves {
        let sample_pos = position.scale(freq);
        let layer_seed = seed.wrapping_add(octave.wrapping_mul(0x0100_0193));
        let value = curl_noise(sample_pos, layer_seed, epsilon);
        total = total.add(value.scale(amp));
        freq *= frequency_multiplier;
        amp *= amplitude_multiplier;
    }
    total
}

/// Spring-damper acceleration anchoring a particle to a fixed point.
///
/// Implements `-stiffness * (position - anchor) - damping * velocity`: a Hooke
/// spring pulling toward `anchor` plus a viscous damper opposing velocity. With
/// positive `stiffness` and `damping` this is an unconditionally decaying
/// harmonic oscillator, useful for ribbons, trails, and soft attachment.
#[must_use]
pub fn spring_damper(
    position: Vec3,
    velocity: Vec3,
    anchor: Vec3,
    stiffness: f32,
    damping: f32,
) -> Vec3 {
    let restoring = anchor.sub(position).scale(stiffness);
    let resistance = velocity.scale(-damping);
    restoring.add(resistance)
}

/// A single force term for the [`ForceField`] accumulator.
///
/// Every variant carries its own parameters so a `ForceField` can hold a
/// heterogeneous, `Copy` list of forces without dynamic dispatch. Variants
/// prefixed with a note reuse the [`super::simulation`] primitive of the same
/// name rather than reimplementing it. Because the enum holds `f32` fields it
/// intentionally derives neither `Eq` nor `Hash`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Force {
    /// Constant gravitational acceleration (reuses [`gravity`]).
    Gravity {
        /// Gravitational acceleration vector.
        acceleration: Vec3,
    },
    /// Linear Stokes drag (reuses [`linear_drag`]); needs the step `dt`.
    LinearDrag {
        /// Inverse time-constant drag coefficient.
        coefficient: f32,
        /// Integration step used to clamp the drag rate.
        dt: f32,
    },
    /// Quadratic (form) drag; see [`quadratic_drag`].
    QuadraticDrag {
        /// Quadratic drag coefficient.
        coefficient: f32,
    },
    /// Aerodynamic wind toward an air-mass velocity (reuses [`wind`]).
    Wind {
        /// Velocity of the moving air mass.
        velocity: Vec3,
        /// Coupling coefficient to the relative velocity.
        coefficient: f32,
    },
    /// Swirling vortex about an axis (reuses [`vortex`]).
    Vortex {
        /// A point on the spin axis.
        center: Vec3,
        /// Spin axis direction.
        axis: Vec3,
        /// Swirl strength.
        strength: f32,
    },
    /// Point attractor / repeller; see [`point_attractor`].
    PointAttractor {
        /// Attraction target.
        point: Vec3,
        /// Signed pull strength (positive attracts).
        strength: f32,
        /// Influence radius (`<= EPS` means unbounded).
        radius: f32,
        /// Singularity softening.
        softening: f32,
        /// Distance shaping curve.
        falloff: Falloff,
    },
    /// Line attractor; see [`line_attractor`].
    LineAttractor {
        /// A point on the line.
        point: Vec3,
        /// Line direction.
        direction: Vec3,
        /// Signed pull strength (positive attracts).
        strength: f32,
        /// Influence radius on the perpendicular distance.
        radius: f32,
        /// Singularity softening.
        softening: f32,
        /// Distance shaping curve.
        falloff: Falloff,
    },
    /// Radial blast / implosion; see [`radial_force`].
    Radial {
        /// Blast center.
        center: Vec3,
        /// Signed strength (positive blasts outward).
        strength: f32,
        /// Influence radius (`<= EPS` means unbounded).
        radius: f32,
        /// Singularity softening.
        softening: f32,
        /// Distance shaping curve.
        falloff: Falloff,
    },
    /// Radius-holding orbital motion; see [`orbital_force`].
    Orbital {
        /// Orbit center.
        center: Vec3,
        /// Orbit axis.
        axis: Vec3,
        /// Target orbit radius held by the radial spring.
        target_radius: f32,
        /// Along-orbit acceleration (sign selects direction).
        tangential_strength: f32,
        /// Radial spring stiffness holding the orbit radius.
        radial_stiffness: f32,
    },
    /// Softened inverse-square gravity well; see [`gravity_well`].
    GravityWell {
        /// Well center.
        center: Vec3,
        /// Signed well strength (positive attracts).
        strength: f32,
        /// Plummer softening length.
        softening: f32,
    },
    /// Layered curl-noise turbulence; see [`turbulence`].
    Turbulence {
        /// Base noise seed.
        seed: u32,
        /// Base sampling frequency.
        frequency: f32,
        /// Base layer amplitude.
        amplitude: f32,
        /// Number of octaves to sum.
        octaves: u32,
        /// Per-layer frequency multiplier.
        frequency_multiplier: f32,
        /// Per-layer amplitude multiplier.
        amplitude_multiplier: f32,
        /// Central-difference step for the curl.
        epsilon: f32,
    },
    /// Spring-damper anchor; see [`spring_damper`].
    SpringDamper {
        /// Rest anchor position.
        anchor: Vec3,
        /// Spring stiffness.
        stiffness: f32,
        /// Viscous damping coefficient.
        damping: f32,
    },
}

impl Force {
    /// Evaluates this force's acceleration for a particle's `position` and
    /// `velocity`.
    ///
    /// Velocity-independent forces ignore `velocity`; velocity-dependent forces
    /// (drags, wind, spring damping) consume it. The dispatch is a pure function
    /// of its inputs, matching the stateless determinism contract (design §29).
    #[must_use]
    pub fn evaluate(self, position: Vec3, velocity: Vec3) -> Vec3 {
        match self {
            Force::Gravity { acceleration } => gravity(acceleration),
            Force::LinearDrag { coefficient, dt } => linear_drag(velocity, coefficient, dt),
            Force::QuadraticDrag { coefficient } => quadratic_drag(velocity, coefficient),
            Force::Wind {
                velocity: air,
                coefficient,
            } => wind(air, velocity, coefficient),
            Force::Vortex {
                center,
                axis,
                strength,
            } => vortex(position, center, axis, strength),
            Force::PointAttractor {
                point,
                strength,
                radius,
                softening,
                falloff,
            } => point_attractor(position, point, strength, radius, softening, falloff),
            Force::LineAttractor {
                point,
                direction,
                strength,
                radius,
                softening,
                falloff,
            } => line_attractor(
                position, point, direction, strength, radius, softening, falloff,
            ),
            Force::Radial {
                center,
                strength,
                radius,
                softening,
                falloff,
            } => radial_force(position, center, strength, radius, softening, falloff),
            Force::Orbital {
                center,
                axis,
                target_radius,
                tangential_strength,
                radial_stiffness,
            } => orbital_force(
                position,
                center,
                axis,
                target_radius,
                tangential_strength,
                radial_stiffness,
            ),
            Force::GravityWell {
                center,
                strength,
                softening,
            } => gravity_well(position, center, strength, softening),
            Force::Turbulence {
                seed,
                frequency,
                amplitude,
                octaves,
                frequency_multiplier,
                amplitude_multiplier,
                epsilon,
            } => turbulence(
                position,
                seed,
                frequency,
                amplitude,
                octaves,
                frequency_multiplier,
                amplitude_multiplier,
                epsilon,
            ),
            Force::SpringDamper {
                anchor,
                stiffness,
                damping,
            } => spring_damper(position, velocity, anchor, stiffness, damping),
        }
    }
}

/// One entry in a [`ForceField`]: a [`Force`] plus its enable flag, blend
/// weight, and an optional spatial gate.
///
/// The gate is a hard on/off sphere: when `gate_radius > EPS`, the entry only
/// contributes while the particle is within `gate_radius` of `gate_center`.
/// A `gate_radius <= EPS` disables the gate (the force applies everywhere,
/// subject to its own internal radius, if any).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ForceEntry {
    /// The physics force to evaluate.
    pub force: Force,
    /// Whether this entry participates in accumulation.
    pub enabled: bool,
    /// Linear blend weight applied to the force's acceleration.
    pub weight: f32,
    /// Center of the optional spatial gate sphere.
    pub gate_center: Vec3,
    /// Radius of the spatial gate sphere (`<= EPS` disables the gate).
    pub gate_radius: f32,
}

impl ForceEntry {
    /// Builds an enabled, unit-weight, ungated entry for `force`.
    #[must_use]
    pub fn new(force: Force) -> Self {
        Self {
            force,
            enabled: true,
            weight: 1.0,
            gate_center: Vec3::ZERO,
            gate_radius: 0.0,
        }
    }

    /// Returns a copy with a new blend `weight`.
    #[must_use]
    pub fn with_weight(self, weight: f32) -> Self {
        Self { weight, ..self }
    }

    /// Returns a copy with a spatial gate sphere.
    #[must_use]
    pub fn with_gate(self, center: Vec3, radius: f32) -> Self {
        Self {
            gate_center: center,
            gate_radius: radius,
            ..self
        }
    }

    /// Returns a copy with the enable flag set to `enabled`.
    #[must_use]
    pub fn with_enabled(self, enabled: bool) -> Self {
        Self { enabled, ..self }
    }

    /// Evaluates the (weighted, gated) contribution of this entry.
    ///
    /// Returns [`Vec3::ZERO`] when disabled or when the particle lies outside
    /// the gate sphere.
    #[must_use]
    pub fn contribution(self, position: Vec3, velocity: Vec3) -> Vec3 {
        if !self.enabled {
            return Vec3::ZERO;
        }
        if self.gate_radius > EPS {
            let gate_sq = self.gate_radius * self.gate_radius;
            if position.distance_squared(self.gate_center) > gate_sq {
                return Vec3::ZERO;
            }
        }
        self.force.evaluate(position, velocity).scale(self.weight)
    }
}

/// A deterministic accumulator that composes a list of [`ForceEntry`] terms.
///
/// Forces are summed in insertion order so the total acceleration is
/// reproducible regardless of how the list was built (design §29). Because the
/// entries own `f32` parameters and a heap `Vec`, this type derives `Clone`,
/// `Debug`, and `PartialEq` (not `Copy`, `Eq`, or `Hash`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ForceField {
    /// The ordered force terms.
    entries: Vec<ForceEntry>,
}

impl ForceField {
    /// Builds an empty force field.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Appends a fully specified entry, returning `&mut self` for chaining.
    pub fn push(&mut self, entry: ForceEntry) -> &mut Self {
        self.entries.push(entry);
        self
    }

    /// Appends a bare [`Force`] as an enabled, unit-weight, ungated entry.
    pub fn add(&mut self, force: Force) -> &mut Self {
        self.entries.push(ForceEntry::new(force));
        self
    }

    /// Read-only view of the ordered entries.
    #[must_use]
    pub fn entries(&self) -> &[ForceEntry] {
        &self.entries
    }

    /// Number of entries currently enabled.
    #[must_use]
    pub fn enabled_count(&self) -> usize {
        self.entries.iter().filter(|e| e.enabled).count()
    }

    /// Accumulates the total acceleration at `position` / `velocity`.
    ///
    /// This is the deterministic net force (per unit mass) the integrator's
    /// `accel_at` closure would return: the ordered sum of every enabled,
    /// in-gate entry's weighted contribution.
    #[must_use]
    pub fn accumulate(&self, position: Vec3, velocity: Vec3) -> Vec3 {
        let mut total = Vec3::ZERO;
        for entry in &self.entries {
            total = total.add(entry.contribution(position, velocity));
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1.0e-4
    }

    fn approx_vec(a: Vec3, b: Vec3) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
    }

    fn is_zero(v: Vec3) -> bool {
        v.length_squared() <= EPS * EPS
    }

    #[test]
    fn falloff_endpoints_are_bounded() {
        for f in [
            Falloff::Constant,
            Falloff::Linear,
            Falloff::Smoothstep,
            Falloff::InverseSquare,
        ] {
            assert!(approx(f.shape(0.0), 1.0), "{f:?} center must be one");
            let edge = f.shape(1.0);
            assert!((0.0..=1.0).contains(&edge));
        }
        assert!(approx(Falloff::Linear.shape(1.0), 0.0));
        assert!(approx(Falloff::Smoothstep.shape(1.0), 0.0));
        assert!(approx(Falloff::InverseSquare.shape(1.0), 0.5));
        // Ratio clamps: out-of-range inputs stay bounded.
        assert!(approx(Falloff::Linear.shape(2.0), 0.0));
        assert!(approx(Falloff::Constant.shape(-1.0), 1.0));
    }

    #[test]
    fn point_attractor_pulls_toward_the_point() {
        let p = Vec3::new(0.0, 0.0, 0.0);
        let target = Vec3::new(10.0, 0.0, 0.0);
        let a = point_attractor(p, target, 2.0, 0.0, 0.1, Falloff::Constant);
        // Unbounded constant field: pull points toward +x.
        assert!(a.x > 0.0);
        assert!(approx(a.y, 0.0) && approx(a.z, 0.0));
    }

    #[test]
    fn point_attractor_negative_strength_repels() {
        let p = Vec3::ZERO;
        let target = Vec3::new(5.0, 0.0, 0.0);
        let a = point_attractor(p, target, -1.0, 0.0, 0.1, Falloff::Constant);
        // Repulsion points away from the target (toward -x).
        assert!(a.x < 0.0);
    }

    #[test]
    fn point_attractor_respects_influence_radius() {
        let p = Vec3::new(100.0, 0.0, 0.0);
        let target = Vec3::ZERO;
        let a = point_attractor(p, target, 1.0, 5.0, 0.1, Falloff::Linear);
        assert!(is_zero(a), "outside the radius the force is zero");
    }

    #[test]
    fn point_attractor_is_finite_at_the_center() {
        // Zero distance would be singular without softening.
        let a = point_attractor(
            Vec3::ZERO,
            Vec3::ZERO,
            1000.0,
            10.0,
            0.5,
            Falloff::InverseSquare,
        );
        assert!(
            is_zero(a),
            "softened direction collapses to zero at the point"
        );
    }

    #[test]
    fn line_attractor_pulls_perpendicular_to_the_line() {
        // Line is the y-axis through the origin; particle offset in +x.
        let a = line_attractor(
            Vec3::new(3.0, 7.0, 0.0),
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            1.0,
            0.0,
            0.05,
            Falloff::Constant,
        );
        // Pull toward the closest point (0, 7, 0): purely -x, no y component.
        assert!(a.x < 0.0);
        assert!(approx(a.y, 0.0));
        assert!(approx(a.z, 0.0));
    }

    #[test]
    fn line_attractor_degenerate_direction_falls_back_to_point() {
        let degenerate = line_attractor(
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::ZERO,
            Vec3::ZERO,
            1.0,
            0.0,
            0.05,
            Falloff::Constant,
        );
        let point = point_attractor(
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::ZERO,
            1.0,
            0.0,
            0.05,
            Falloff::Constant,
        );
        assert!(approx_vec(degenerate, point));
    }

    #[test]
    fn radial_force_blasts_outward_and_implodes_inward() {
        let p = Vec3::new(2.0, 0.0, 0.0);
        let out = radial_force(p, Vec3::ZERO, 1.0, 0.0, 0.1, Falloff::Constant);
        assert!(out.x > 0.0, "positive strength pushes away from center");
        let inn = implosion(p, Vec3::ZERO, 1.0, 0.0, 0.1, Falloff::Constant);
        assert!(inn.x < 0.0, "implosion pulls toward center");
        let boom = explosion(p, Vec3::ZERO, 1.0, 0.0, 0.1, Falloff::Constant);
        assert!(approx_vec(boom, out));
    }

    #[test]
    fn orbital_force_is_tangential_and_holds_radius() {
        // Axis +z, particle at radius 2 on +x, target radius 2.
        let a = orbital_force(
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, 1.0),
            2.0,
            1.0,
            4.0,
        );
        // On-radius: no radial correction, tangent is +y (z cross x = y).
        assert!(approx(a.x, 0.0));
        assert!(a.y > 0.0);
        assert!(approx(a.z, 0.0));
    }

    #[test]
    fn orbital_force_corrects_radius_error() {
        // Particle beyond the target radius: expect an inward radial pull.
        let a = orbital_force(
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, 1.0),
            2.0,
            0.0,
            3.0,
        );
        // No tangential term, only the spring: 3 * (5 - 2) inward = -9 on x.
        assert!(approx(a.x, -9.0));
        assert!(approx(a.y, 0.0));
    }

    #[test]
    fn orbital_force_on_axis_is_zero() {
        let a = orbital_force(
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, 1.0),
            2.0,
            1.0,
            1.0,
        );
        assert!(is_zero(a));
    }

    #[test]
    fn gravity_well_follows_inverse_square() {
        // Distance 2 -> magnitude ~ strength / 4 (softening negligible).
        let a = gravity_well(Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO, 8.0, 0.0);
        assert!(a.x < 0.0, "attraction points toward the center");
        assert!(approx(a.x, -2.0));
        // Half the distance -> four times the magnitude.
        let near = gravity_well(Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO, 8.0, 0.0);
        assert!(approx(near.x, -8.0));
    }

    #[test]
    fn gravity_well_is_finite_at_center() {
        let a = gravity_well(Vec3::ZERO, Vec3::ZERO, 1.0e6, 1.0);
        assert!(
            is_zero(a),
            "zero direction yields no force at the exact center"
        );
    }

    #[test]
    fn quadratic_drag_opposes_and_scales_with_speed_squared() {
        let v = Vec3::new(3.0, 0.0, 0.0);
        let a = quadratic_drag(v, 0.5);
        // -k * |v| * v = -0.5 * 3 * 3 = -4.5 on x.
        assert!(approx(a.x, -4.5));
        // Doubling speed quadruples the magnitude.
        let a2 = quadratic_drag(Vec3::new(6.0, 0.0, 0.0), 0.5);
        assert!(approx(a2.x, -18.0));
    }

    #[test]
    fn quadratic_drag_zero_velocity_is_zero() {
        assert!(is_zero(quadratic_drag(Vec3::ZERO, 1.0)));
        assert!(is_zero(quadratic_drag(Vec3::new(1.0, 0.0, 0.0), 0.0)));
    }

    #[test]
    fn turbulence_scales_with_amplitude_and_sums_octaves() {
        let p = Vec3::new(1.3, -2.1, 0.7);
        let one = turbulence(p, 42, 1.0, 1.0, 1, 2.0, 0.5, 1.0e-3);
        let scaled = turbulence(p, 42, 1.0, 2.0, 1, 2.0, 0.5, 1.0e-3);
        // A single octave scales linearly with amplitude.
        assert!(approx_vec(scaled, one.scale(2.0)));
        // Zero octaves contribute nothing.
        assert!(is_zero(turbulence(p, 42, 1.0, 1.0, 0, 2.0, 0.5, 1.0e-3)));
        // More octaves generally change the field (non-degenerate check).
        let three = turbulence(p, 42, 1.0, 1.0, 3, 2.0, 0.5, 1.0e-3);
        assert!(!approx_vec(three, one) || is_zero(one));
    }

    #[test]
    fn spring_damper_restores_and_damps() {
        // Displaced +x with +x velocity: spring pulls -x, damper opposes -x.
        let a = spring_damper(
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::ZERO,
            4.0,
            0.5,
        );
        // -4 * 2 - 0.5 * 1 = -8.5 on x.
        assert!(approx(a.x, -8.5));
    }

    #[test]
    fn spring_damper_at_rest_at_anchor_is_zero() {
        let a = spring_damper(Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, 10.0, 1.0);
        assert!(is_zero(a));
    }

    #[test]
    fn force_field_sums_in_order_with_weight() {
        let mut field = ForceField::new();
        field
            .add(Force::Gravity {
                acceleration: Vec3::new(0.0, -9.8, 0.0),
            })
            .push(
                ForceEntry::new(Force::Gravity {
                    acceleration: Vec3::new(0.0, -1.0, 0.0),
                })
                .with_weight(2.0),
            );
        let total = field.accumulate(Vec3::ZERO, Vec3::ZERO);
        // -9.8 + 2 * (-1.0) = -11.8.
        assert!(approx(total.y, -11.8));
        assert_eq!(field.enabled_count(), 2);
    }

    #[test]
    fn force_field_disabled_and_gated_entries_drop_out() {
        let mut field = ForceField::new();
        field
            .push(
                ForceEntry::new(Force::Gravity {
                    acceleration: Vec3::new(0.0, -9.8, 0.0),
                })
                .with_enabled(false),
            )
            .push(
                // Gated far from the particle: contributes nothing.
                ForceEntry::new(Force::Gravity {
                    acceleration: Vec3::new(5.0, 0.0, 0.0),
                })
                .with_gate(Vec3::new(100.0, 0.0, 0.0), 1.0),
            )
            .push(
                // Gated around the particle: contributes.
                ForceEntry::new(Force::Gravity {
                    acceleration: Vec3::new(0.0, 0.0, 3.0),
                })
                .with_gate(Vec3::ZERO, 1.0),
            );
        let total = field.accumulate(Vec3::ZERO, Vec3::ZERO);
        assert!(approx_vec(total, Vec3::new(0.0, 0.0, 3.0)));
        assert_eq!(field.enabled_count(), 2);
    }

    #[test]
    fn force_field_matches_direct_evaluation() {
        // The accumulator must agree with calling the primitives directly.
        let pos = Vec3::new(1.0, 2.0, 3.0);
        let vel = Vec3::new(0.5, -0.5, 0.0);
        let drag = Force::QuadraticDrag { coefficient: 0.3 };
        let well = Force::GravityWell {
            center: Vec3::ZERO,
            strength: 4.0,
            softening: 0.2,
        };
        let mut field = ForceField::new();
        field.add(drag).add(well);
        let expected = drag.evaluate(pos, vel).add(well.evaluate(pos, vel));
        assert!(approx_vec(field.accumulate(pos, vel), expected));
    }

    #[test]
    fn empty_force_field_is_zero() {
        let field = ForceField::new();
        assert!(is_zero(
            field.accumulate(Vec3::new(1.0, 2.0, 3.0), Vec3::ZERO)
        ));
        assert_eq!(field.enabled_count(), 0);
    }
}
