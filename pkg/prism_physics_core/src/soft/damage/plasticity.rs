//! Plasticity: permanent rest-length creep for over-stretched fabric edges.
//!
//! Plasticity lets a distance edge stretched past a yield strain creep its rest
//! length toward the current length, capturing permanent wrinkles and sag (a
//! plastic set) while a bounded residual elastic strain is retained. It is the
//! counterpart to [`super::tearing`]: tearing removes an edge outright, whereas
//! plasticity permanently lengthens (or shortens) it. Both are expressed from
//! first principles.
//!
//! For an edge with signed strain `e` and `|e| > yield`, the rest length moves
//! by `creep` times the beyond-yield excess toward the current length, then is
//! clamped so the residual elastic strain magnitude never exceeds `max_strain`.
//! Edges within the yield band, degenerate edges, and out-of-range endpoints are
//! left untouched. The pass is `O(edges)` and deterministic.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Rest-length
//! creep past a yield strain is a standard, publicly documented plastic-set
//! model for position-based cloth.

use glam::Vec3;

use crate::math::scalar::Real;
use crate::soft::constraint::DistanceConstraint;

use super::strain::{edge_length, EPS_REST};

/// Tuning for plastic (permanent) rest-length creep.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PlasticParams {
    /// Strain magnitude beyond which plastic flow begins; below this the edge is
    /// purely elastic and its rest length is untouched. Non-negative.
    pub yield_strain: Real,
    /// Fraction in `[0, 1]` of the beyond-yield strain converted to a permanent
    /// rest-length change each call; `0` disables creep and `1` drives the rest
    /// length aggressively toward the current length.
    pub creep: Real,
    /// Cap on the residual elastic strain magnitude left after creep, so an edge
    /// is never relaxed so far that it still holds more than this strain.
    /// Non-negative.
    pub max_strain: Real,
}

impl Default for PlasticParams {
    /// Sensible defaults: yield at 10% strain, slow creep, capped residual.
    fn default() -> Self {
        PlasticParams {
            yield_strain: 0.1,
            creep: 0.1,
            max_strain: 0.3,
        }
    }
}

impl PlasticParams {
    /// Creates plastic parameters from a yield strain, creep fraction, and
    /// residual-strain cap.
    #[must_use]
    pub fn new(yield_strain: Real, creep: Real, max_strain: Real) -> Self {
        PlasticParams {
            yield_strain,
            creep,
            max_strain,
        }
    }

    /// Returns a copy with `yield_strain` and `max_strain` forced non-negative
    /// and `creep` clamped to `[0, 1]`, replacing any `NaN` with a safe value.
    #[must_use]
    pub fn sanitized(self) -> Self {
        let clamp_nonneg = |v: Real| if v.is_nan() || v < 0.0 { 0.0 } else { v };
        PlasticParams {
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

/// Applies plastic rest-length creep to every distance edge stretched (or
/// compressed) past `params.yield_strain`, in place.
///
/// For an edge with signed strain `e` and `|e| > yield`, the rest length is
/// moved by `creep` times the beyond-yield excess toward the current length,
/// then clamped so the residual elastic strain magnitude does not exceed
/// `max_strain`. Edges within the yield band, degenerate edges, and out-of-range
/// endpoints are left unchanged. Returns the number of edges that were
/// plastically modified.
pub fn apply_plasticity(
    constraints: &mut [DistanceConstraint],
    positions: &[Vec3],
    params: PlasticParams,
) -> usize {
    let params = params.sanitized();
    let mut modified = 0;
    for c in constraints.iter_mut() {
        // Out-of-range endpoints make the edge inert; the rest-length guard and
        // the whole creep decision live in [`plastic_rest_length`].
        let Some(len) = edge_length(c, positions) else {
            continue;
        };
        if let Some(new_rest) = plastic_rest_length(c.rest_length, len, params) {
            c.rest_length = new_rest;
            modified += 1;
        }
    }
    modified
}

/// Computes the plastically crept rest length for a single distance edge from
/// its current `rest_length`, its current `length` (the separation of its two
/// endpoints), and the plastic `params`.
///
/// Returns `Some(new_rest)` when the signed strain `(length - rest) / rest` lies
/// beyond the yield band and the edge creeps to a strictly positive new rest
/// length, or `None` when the edge is within the yield band, has a degenerate
/// rest length (`<= EPS_REST`), or would collapse to the numerical floor. The
/// returned rest length is moved by `creep` times the beyond-yield excess toward
/// `length`, then clamped so the residual elastic strain magnitude never exceeds
/// `max_strain`.
///
/// This is the single scalar kernel shared by the sequential [`apply_plasticity`]
/// pass and the `prism_physics_gpu` cloth twin, so both agree on the creep
/// arithmetic up to floating-point rounding. `params` is sanitized internally,
/// so the function is safe to call with raw authored values.
#[must_use]
pub fn plastic_rest_length(rest_length: Real, length: Real, params: PlasticParams) -> Option<Real> {
    let params = params.sanitized();
    if rest_length <= EPS_REST {
        return None;
    }
    let strain = (length - rest_length) / rest_length;
    if strain.abs() <= params.yield_strain {
        return None;
    }
    let sign = if strain >= 0.0 { 1.0 } else { -1.0 };
    let excess = strain - sign * params.yield_strain;
    // Move the rest length by `creep` fraction of the excess strain.
    let mut new_rest = rest_length * (1.0 + params.creep * excess);
    if new_rest <= EPS_REST {
        new_rest = EPS_REST;
    }
    // Clamp so the residual elastic strain magnitude stays within max.
    let residual = (length - new_rest) / new_rest;
    if residual.abs() > params.max_strain {
        let residual_sign = if residual >= 0.0 { 1.0 } else { -1.0 };
        new_rest = length / (1.0 + residual_sign * params.max_strain);
    }
    if new_rest > EPS_REST {
        Some(new_rest)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::soft::particle::ParticleHandle;

    fn h(i: u32) -> ParticleHandle {
        ParticleHandle::from_index(i)
    }

    fn edge(a: u32, b: u32, rest: Real) -> DistanceConstraint {
        DistanceConstraint::new(h(a), h(b), rest, 0.0)
    }

    #[test]
    fn default_params_are_sensible() {
        let p = PlasticParams::default();
        assert_eq!(p.yield_strain, 0.1);
        assert_eq!(p.creep, 0.1);
        assert_eq!(p.max_strain, 0.3);
    }

    #[test]
    fn sanitize_clamps_creep_and_floors_negatives() {
        let p = PlasticParams::new(-1.0, 5.0, Real::NAN).sanitized();
        assert_eq!(p.yield_strain, 0.0);
        assert_eq!(p.creep, 1.0);
        assert_eq!(p.max_strain, 0.0);
    }

    #[test]
    fn within_yield_band_is_untouched() {
        // strain 0.05 < yield 0.1
        let positions = [Vec3::ZERO, Vec3::new(1.05, 0.0, 0.0)];
        let mut constraints = vec![edge(0, 1, 1.0)];
        let modified = apply_plasticity(
            &mut constraints,
            &positions,
            PlasticParams::new(0.1, 0.5, 1.0),
        );
        assert_eq!(modified, 0);
        assert_eq!(constraints[0].rest_length, 1.0);
    }

    #[test]
    fn beyond_yield_creeps_rest_length_up() {
        // strain 1.0 (rest 1, len 2) >> yield 0.1
        let positions = [Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
        let mut constraints = vec![edge(0, 1, 1.0)];
        let modified = apply_plasticity(
            &mut constraints,
            &positions,
            PlasticParams::new(0.1, 0.5, 10.0),
        );
        assert_eq!(modified, 1);
        // excess = 1.0 - 0.1 = 0.9; new_rest = 1 * (1 + 0.5*0.9) = 1.45.
        assert!(
            (constraints[0].rest_length - 1.45).abs() < 1e-5,
            "rest {}",
            constraints[0].rest_length
        );
    }

    #[test]
    fn residual_strain_is_capped() {
        // Full creep toward current length, but capped so residual <= max_strain.
        let positions = [Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
        let mut constraints = vec![edge(0, 1, 1.0)];
        apply_plasticity(
            &mut constraints,
            &positions,
            PlasticParams::new(0.1, 1.0, 0.2),
        );
        let rest = constraints[0].rest_length;
        let residual = (2.0 - rest) / rest;
        assert!(residual.abs() <= 0.2 + 1e-5, "residual {residual}");
    }

    #[test]
    fn degenerate_and_out_of_range_edges_are_untouched() {
        let positions = [Vec3::ZERO, Vec3::new(5.0, 0.0, 0.0)];
        let mut constraints = vec![edge(0, 1, 0.0), edge(0, 9, 1.0)];
        let modified = apply_plasticity(
            &mut constraints,
            &positions,
            PlasticParams::new(0.1, 0.5, 1.0),
        );
        assert_eq!(modified, 0);
        assert_eq!(constraints[0].rest_length, 0.0);
        assert_eq!(constraints[1].rest_length, 1.0);
    }
}
