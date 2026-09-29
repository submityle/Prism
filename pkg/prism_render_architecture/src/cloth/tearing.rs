//! Tearing and plastic deformation for cloth constraints.
//!
//! Two related permanent-damage models act on the distance-constraint graph:
//!
//! * **Tearing** removes an edge once its tensile strain exceeds a break
//!   threshold, so an over-stretched garment rips instead of stretching without
//!   bound. This mirrors the constraint-breaking used in production cloth
//!   (`Chaos` Cloth, Houdini `Vellum`) where a failed edge is simply dropped
//!   from the solve.
//! * **Plasticity** lets an edge that is stretched past a yield strain creep its
//!   rest length toward the current length, capturing permanent wrinkles and
//!   sag (a plastic set) while a residual elastic strain is retained.
//!
//! Both operate purely on the constraint list and the current particle
//! positions; they are `O(edges)`, deterministic (edges visited in order), and
//! use only [`f32::sqrt`] via the shared distance helper. Only two-sided fabric
//! constraints participate: one-sided attachments ([`ConstraintKind::Lra`] and
//! [`ConstraintKind::Tether`]) are skipped so an anchor leash is never torn or
//! plastically lengthened. Tensile strain is `(len - rest) / rest`; a
//! (near) zero rest length is skipped to avoid a divide-by-zero.

use alloc::vec::Vec;

use super::{ClothParticle, Constraint};

/// Numerical floor below which a rest length is treated as degenerate.
const EPS_REST: f32 = 1e-9;

/// Tuning for constraint tearing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TearingParams {
    /// Tensile strain `(len - rest) / rest` above which a two-sided edge breaks
    /// and is removed from the graph. Non-negative; compression never tears.
    pub break_strain: f32,
}

impl Default for TearingParams {
    /// A default break strain of 50% elongation.
    fn default() -> Self {
        Self { break_strain: 0.5 }
    }
}

impl TearingParams {
    /// Returns a copy with `break_strain` forced non-negative and finite, so a
    /// mis-authored `NaN` or negative threshold cannot tear every edge.
    #[must_use]
    pub fn sanitized(self) -> Self {
        let break_strain = if self.break_strain.is_nan() || self.break_strain < 0.0 {
            f32::INFINITY
        } else {
            self.break_strain
        };
        Self { break_strain }
    }
}

/// A summary of the tensile state of the constraint graph.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TearReport {
    /// Number of two-sided edges inspected (edges with a valid rest length and
    /// in-range endpoints).
    pub inspected: usize,
    /// Number of inspected edges whose tensile strain exceeds the threshold.
    pub over_threshold: usize,
    /// The largest tensile strain observed across inspected edges (`0` when
    /// none were inspected).
    pub max_strain: f32,
}

/// Returns the tensile strain `(len - rest) / rest` of `constraint`, or `None`
/// when the edge is one-sided, has a degenerate rest length, or references a
/// particle index outside `particles`.
fn edge_strain(constraint: &Constraint, particles: &[ClothParticle]) -> Option<f32> {
    if constraint.kind.is_one_sided() {
        return None;
    }
    if constraint.rest_length <= EPS_REST {
        return None;
    }
    let a = constraint.a as usize;
    let b = constraint.b as usize;
    if a >= particles.len() || b >= particles.len() {
        return None;
    }
    let len = particles[a].position.distance(particles[b].position);
    Some((len - constraint.rest_length) / constraint.rest_length)
}

/// Removes every two-sided fabric edge whose tensile strain exceeds
/// `params.break_strain`, in place, and returns the number of edges torn.
///
/// One-sided attachments, degenerate edges, and edges with out-of-range
/// endpoints are always kept. Retained edges preserve their original relative
/// order, so the graph coloring downstream stays deterministic.
pub fn apply_tearing(
    constraints: &mut Vec<Constraint>,
    particles: &[ClothParticle],
    params: TearingParams,
) -> usize {
    let params = params.sanitized();
    let before = constraints.len();
    constraints.retain(|c| match edge_strain(c, particles) {
        Some(strain) => strain <= params.break_strain,
        None => true,
    });
    before - constraints.len()
}

/// Inspects the graph without modifying it and returns a [`TearReport`] of the
/// tensile state against `params.break_strain`.
#[must_use]
pub fn tear_report(
    constraints: &[Constraint],
    particles: &[ClothParticle],
    params: TearingParams,
) -> TearReport {
    let params = params.sanitized();
    let mut report = TearReport {
        inspected: 0,
        over_threshold: 0,
        max_strain: 0.0,
    };
    for c in constraints {
        if let Some(strain) = edge_strain(c, particles) {
            report.inspected += 1;
            if strain > report.max_strain {
                report.max_strain = strain;
            }
            if strain > params.break_strain {
                report.over_threshold += 1;
            }
        }
    }
    report
}

/// Tuning for plastic (permanent) rest-length creep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlasticParams {
    /// Strain magnitude beyond which plastic flow begins; below this the edge is
    /// purely elastic and its rest length is untouched. Non-negative.
    pub yield_strain: f32,
    /// Fraction in `[0, 1]` of the beyond-yield strain that is converted to a
    /// permanent rest-length change each call; `0` disables creep and `1` sets
    /// the rest length aggressively toward the current length.
    pub creep: f32,
    /// Cap on the residual elastic strain magnitude left after creep, so an
    /// edge can never be relaxed so far that it holds more than this strain.
    /// Non-negative.
    pub max_strain: f32,
}

impl Default for PlasticParams {
    /// Sensible defaults: yield at 10% strain, slow creep, capped residual.
    fn default() -> Self {
        Self {
            yield_strain: 0.1,
            creep: 0.1,
            max_strain: 0.3,
        }
    }
}

impl PlasticParams {
    /// Returns a copy with `yield_strain` and `max_strain` forced non-negative
    /// and `creep` clamped to `[0, 1]`, replacing any `NaN` with a safe value.
    #[must_use]
    pub fn sanitized(self) -> Self {
        let clamp_nonneg = |v: f32| if v.is_nan() || v < 0.0 { 0.0 } else { v };
        Self {
            yield_strain: clamp_nonneg(self.yield_strain),
            creep: if self.creep.is_nan() {
                0.0
            } else {
                self.creep.clamp(0.0, 1.0)
            },
            max_strain: clamp_nonneg(self.max_strain),
        }
    }
}

/// Applies plastic rest-length creep to every two-sided fabric edge stretched
/// (or compressed) past `params.yield_strain`, in place.
///
/// For an edge with strain `e` and `|e| > yield`, the rest length is moved by
/// `creep` times the beyond-yield excess toward the current length, then
/// clamped so the residual elastic strain magnitude does not exceed
/// `max_strain`. Edges within the yield band, one-sided attachments, degenerate
/// edges, and out-of-range endpoints are left unchanged.
pub fn apply_plasticity(
    constraints: &mut [Constraint],
    particles: &[ClothParticle],
    params: PlasticParams,
) {
    let params = params.sanitized();
    for c in constraints.iter_mut() {
        let Some(strain) = edge_strain(c, particles) else {
            continue;
        };
        let magnitude = strain.abs();
        if magnitude <= params.yield_strain {
            continue;
        }
        let a = c.a as usize;
        let b = c.b as usize;
        let len = particles[a].position.distance(particles[b].position);
        let sign = if strain >= 0.0 { 1.0 } else { -1.0 };
        let excess = strain - sign * params.yield_strain;
        // Move the rest length by `creep` fraction of the excess strain.
        let mut new_rest = c.rest_length * (1.0 + params.creep * excess);
        if new_rest <= EPS_REST {
            new_rest = EPS_REST;
        }
        // Clamp so the residual elastic strain magnitude stays within max.
        let residual = (len - new_rest) / new_rest;
        if residual.abs() > params.max_strain {
            let residual_sign = if residual >= 0.0 { 1.0 } else { -1.0 };
            new_rest = len / (1.0 + residual_sign * params.max_strain);
        }
        if new_rest > EPS_REST {
            c.rest_length = new_rest;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloth::{Compliance, ConstraintKind, Vec3};
    use alloc::vec;

    /// Builds a free particle at `position` with unit inverse mass.
    fn free_particle(position: Vec3) -> ClothParticle {
        ClothParticle {
            position,
            velocity: Vec3::ZERO,
            inverse_mass: 1.0,
        }
    }

    /// Builds a stretch edge between indices `a` and `b` with the given rest.
    fn stretch(a: u32, b: u32, rest: f32) -> Constraint {
        Constraint::new(a, b, rest, Compliance::RIGID, ConstraintKind::Stretch)
    }

    #[test]
    fn tearing_removes_over_strained_edges_only() {
        let particles = [
            free_particle(Vec3::ZERO),
            free_particle(Vec3::new(2.0, 0.0, 0.0)),
            free_particle(Vec3::new(2.0, 1.05, 0.0)),
        ];
        // Edge 0-1 has strain 1.0 (rest 1, len 2); edge 1-2 strain ~0.05.
        let mut constraints = vec![stretch(0, 1, 1.0), stretch(1, 2, 1.0)];
        let torn = apply_tearing(
            &mut constraints,
            &particles,
            TearingParams { break_strain: 0.5 },
        );
        assert_eq!(torn, 1);
        assert_eq!(constraints.len(), 1);
        // The surviving edge is the low-strain 1-2 edge.
        assert_eq!(constraints[0].a, 1);
        assert_eq!(constraints[0].b, 2);
    }

    #[test]
    fn tearing_never_removes_one_sided_or_compressed() {
        let particles = [
            free_particle(Vec3::ZERO),
            free_particle(Vec3::new(3.0, 0.0, 0.0)),
        ];
        // A tether stretched well past threshold, plus a compressed stretch edge.
        let mut constraints = vec![
            Constraint::new(0, 1, 1.0, Compliance::RIGID, ConstraintKind::Tether),
            stretch(0, 1, 10.0),
        ];
        let torn = apply_tearing(
            &mut constraints,
            &particles,
            TearingParams { break_strain: 0.5 },
        );
        assert_eq!(torn, 0);
        assert_eq!(constraints.len(), 2);
    }

    #[test]
    fn tear_report_captures_max_strain() {
        let particles = [
            free_particle(Vec3::ZERO),
            free_particle(Vec3::new(2.0, 0.0, 0.0)),
        ];
        let constraints = vec![stretch(0, 1, 1.0)];
        let report = tear_report(
            &constraints,
            &particles,
            TearingParams { break_strain: 0.5 },
        );
        assert_eq!(report.inspected, 1);
        assert_eq!(report.over_threshold, 1);
        assert!((report.max_strain - 1.0).abs() < 1e-5);
    }

    #[test]
    fn plasticity_lengthens_rest_under_tension() {
        let particles = [
            free_particle(Vec3::ZERO),
            free_particle(Vec3::new(2.0, 0.0, 0.0)),
        ];
        let mut constraints = vec![stretch(0, 1, 1.0)];
        apply_plasticity(
            &mut constraints,
            &particles,
            PlasticParams {
                yield_strain: 0.1,
                creep: 0.5,
                max_strain: 10.0,
            },
        );
        // strain = 1.0, excess = 0.9, new_rest = 1 * (1 + 0.5*0.9) = 1.45.
        assert!((constraints[0].rest_length - 1.45).abs() < 1e-5);
    }

    #[test]
    fn plasticity_ignores_strain_below_yield() {
        let particles = [
            free_particle(Vec3::ZERO),
            free_particle(Vec3::new(1.05, 0.0, 0.0)),
        ];
        let mut constraints = vec![stretch(0, 1, 1.0)];
        apply_plasticity(&mut constraints, &particles, PlasticParams::default());
        assert!((constraints[0].rest_length - 1.0).abs() < 1e-6);
    }

    #[test]
    fn plasticity_clamps_residual_to_max_strain() {
        let particles = [
            free_particle(Vec3::ZERO),
            free_particle(Vec3::new(2.0, 0.0, 0.0)),
        ];
        let mut constraints = vec![stretch(0, 1, 1.0)];
        // Aggressive creep would relax far, but max_strain caps residual at 0.1.
        apply_plasticity(
            &mut constraints,
            &particles,
            PlasticParams {
                yield_strain: 0.1,
                creep: 1.0,
                max_strain: 0.1,
            },
        );
        let len = 2.0;
        let residual = (len - constraints[0].rest_length) / constraints[0].rest_length;
        assert!(residual.abs() <= 0.1 + 1e-5);
    }

    #[test]
    fn out_of_range_indices_are_skipped() {
        let particles = [free_particle(Vec3::ZERO)];
        let mut constraints = vec![stretch(0, 9, 1.0)];
        let torn = apply_tearing(&mut constraints, &particles, TearingParams::default());
        assert_eq!(torn, 0);
        apply_plasticity(&mut constraints, &particles, PlasticParams::default());
        assert!((constraints[0].rest_length - 1.0).abs() < 1e-6);
    }
}
