//! Flocking / `Boids` cluster simulation for the particle subsystem (design §10).
//!
//! This is the `CPU`-verifiable contract layer for herd / flock / school /
//! firefly-swarm behavior, aligned with the flocking operators of Unreal
//! `Niagara` and `Houdini` at the algorithm level (no reuse of their code). It
//! models Reynolds' three classic `Boids` rules — *separation*, *alignment*,
//! and *cohesion* — plus *goal seeking*, each with its own weight, and composes
//! them into a single steering acceleration under explicit speed / force / turn
//! limits.
//!
//! Neighborhoods are gathered through the shared spatial hash
//! ([`NeighborGrid`], design §10) rather than an O(n²) scan, then narrowed by a
//! perception radius and a field-of-view test. The field-of-view cull is a
//! normalized dot-product threshold, and the turn / speed / force clamps use
//! only `length` and `scale`: no transcendental (`sin` / `cos` / `acos` /
//! `exp` / `pow`) is ever called, so the reference stays bit-reproducible
//! against a future `GPU` kernel and draws no randomness from the hash `RNG`.
//!
//! The force-composition order is fixed (separation, then alignment, then
//! cohesion, then goal) so the accumulated acceleration is deterministic
//! regardless of neighbor discovery order (design §29). A quality ladder scales
//! the perception radius and caps the neighbor sample count so lower tiers do
//! strictly less work (design §28).

use alloc::vec::Vec;

use super::stages::{CellCoord, NeighborGrid};
use super::{Vec3, EPS_LEN_SQ};

/// Per-rule blend weights for the steering composition.
///
/// Each weight scales one contribution before the fixed-order sum in
/// [`combine_forces`]. A weight of `0.0` disables its rule without changing the
/// deterministic accumulation order. Holds `f32` fields, so it derives
/// [`PartialEq`] (not [`Eq`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoidsWeights {
    /// Weight of the separation (short-range repulsion) contribution.
    pub separation: f32,
    /// Weight of the alignment (match-neighbor-heading) contribution.
    pub alignment: f32,
    /// Weight of the cohesion (steer-to-centroid) contribution.
    pub cohesion: f32,
    /// Weight of the goal-seeking (steer-to-target) contribution.
    pub goal: f32,
}

impl BoidsWeights {
    /// Builds a weight set from its four components.
    #[must_use]
    pub const fn new(separation: f32, alignment: f32, cohesion: f32, goal: f32) -> Self {
        Self {
            separation,
            alignment,
            cohesion,
            goal,
        }
    }

    /// A balanced preset in the spirit of Reynolds' original `Boids`:
    /// separation dominates, alignment and cohesion are moderate, and goal
    /// seeking is off (`0.0`).
    #[must_use]
    pub const fn classic_flock() -> Self {
        Self::new(1.5, 1.0, 1.0, 0.0)
    }
}

/// Kinematic limits that bound the steering response.
///
/// The field-of-view is expressed as `fov_cos_threshold`, the minimum value of
/// `dot(heading, dir_to_neighbor)` for a neighbor to be *visible* — a
/// transcendental-free stand-in for a maximum view half-angle (`1.0` is a
/// forward pinhole, `0.0` is a 180° hemisphere, `-1.0` sees everything). The
/// turn clamp `max_turn` limits the steering component perpendicular to the
/// current heading (a linear tangential proxy for a maximum turn angle), while
/// `max_force` clamps the total steering magnitude and `max_speed` clamps the
/// integrated velocity. Holds `f32` fields, so it derives [`PartialEq`] (not
/// [`Eq`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoidsLimits {
    /// Radius within which neighbors are perceived (feeds [`NeighborGrid`]).
    pub perception_radius: f32,
    /// Minimum `dot(heading, dir_to_neighbor)` for a neighbor to be visible.
    pub fov_cos_threshold: f32,
    /// Maximum integrated speed (velocity magnitude clamp).
    pub max_speed: f32,
    /// Maximum steering-force magnitude (acceleration clamp).
    pub max_force: f32,
    /// Maximum steering magnitude perpendicular to the heading (turn proxy).
    pub max_turn: f32,
}

impl BoidsLimits {
    /// Builds a limit set from its components.
    #[must_use]
    pub const fn new(
        perception_radius: f32,
        fov_cos_threshold: f32,
        max_speed: f32,
        max_force: f32,
        max_turn: f32,
    ) -> Self {
        Self {
            perception_radius,
            fov_cos_threshold,
            max_speed,
            max_force,
            max_turn,
        }
    }
}

/// A rendering quality tier for flocking work (design §28).
///
/// Mirrors the coarse ladder the `LOD` module uses so a caller can degrade
/// flocking cost in lockstep with the rest of the particle subsystem.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BoidsQuality {
    /// Lowest fidelity: smallest perception radius, tightest neighbor cap.
    Low,
    /// Medium fidelity.
    Medium,
    /// High fidelity.
    High,
    /// Highest fidelity: full perception radius and neighbor cap.
    Ultra,
}

/// The scaled work budget a [`BoidsQuality`] tier grants.
///
/// `perception_scale` multiplies the base perception radius and `max_neighbors`
/// caps how many visible neighbors a boid samples; both shrink monotonically as
/// the tier drops, so lower tiers do strictly less work. Holds an `f32`, so it
/// derives [`PartialEq`] (not [`Eq`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoidsBudget {
    /// Multiplier applied to the base perception radius (`0..=1`).
    pub perception_scale: f32,
    /// Upper bound on the number of neighbors a boid samples per step.
    pub max_neighbors: u32,
}

impl BoidsBudget {
    /// Applies the perception scale to a base radius.
    #[must_use]
    pub fn scaled_radius(self, base_radius: f32) -> f32 {
        base_radius * self.perception_scale
    }
}

/// Maps a [`BoidsQuality`] tier to its [`BoidsBudget`] (design §28).
///
/// The staircase is fixed and total: each lower tier both shrinks the
/// perception radius and lowers the neighbor sample cap, so a scheduler can
/// trade fidelity for cost predictably.
#[must_use]
pub fn boids_quality_budget(quality: BoidsQuality) -> BoidsBudget {
    match quality {
        BoidsQuality::Low => BoidsBudget {
            perception_scale: 0.5,
            max_neighbors: 8,
        },
        BoidsQuality::Medium => BoidsBudget {
            perception_scale: 0.7,
            max_neighbors: 16,
        },
        BoidsQuality::High => BoidsBudget {
            perception_scale: 0.85,
            max_neighbors: 32,
        },
        BoidsQuality::Ultra => BoidsBudget {
            perception_scale: 1.0,
            max_neighbors: 64,
        },
    }
}

/// The spatial-hash cell a boid at `position` occupies for a given `cell_size`.
///
/// A thin, self-documenting wrapper over [`NeighborGrid::cell_of`] used when a
/// caller wants to bucket boids (for example to partition a flock across
/// `GPU` workgroups) with the same lattice the neighbor query uses.
#[must_use]
pub fn boid_cell(position: Vec3, cell_size: f32) -> CellCoord {
    NeighborGrid::cell_of(position, cell_size)
}

/// Clamps a vector's magnitude to `max_len` (no-op when already shorter).
///
/// Uses only `length`-style arithmetic and `scale`; a (near) zero vector or a
/// non-positive `max_len` short-circuits so normalization never divides by
/// zero.
#[must_use]
pub fn clamp_length(v: Vec3, max_len: f32) -> Vec3 {
    let len_sq = v.length_squared();
    let max_sq = max_len * max_len;
    if max_len > 0.0 && len_sq > max_sq && len_sq > EPS_LEN_SQ {
        v.scale(max_len / len_sq.sqrt())
    } else if max_len > 0.0 {
        v
    } else {
        Vec3::ZERO
    }
}

/// Separation: a distance-weighted push away from perceived `neighbors`.
///
/// Each neighbor contributes a unit vector pointing from it toward the boid,
/// scaled by the inverse of its distance so closer crowd-mates repel harder.
/// The accumulated push is normalized to a steering *direction*; an empty
/// neighborhood yields [`Vec3::ZERO`].
#[must_use]
pub fn separation_force(index: usize, positions: &[Vec3], neighbors: &[u32]) -> Vec3 {
    let self_pos = positions[index];
    let mut push = Vec3::ZERO;
    for &n in neighbors {
        let other = positions[n as usize];
        let offset = self_pos.sub(other);
        let dist_sq = offset.length_squared();
        if dist_sq > EPS_LEN_SQ {
            let dist = dist_sq.sqrt();
            push = push.add(offset.scale(1.0 / (dist * dist)));
        }
    }
    push.normalize_or_zero()
}

/// Alignment: steer toward the average heading of the `neighbors`.
///
/// Averages the neighbor velocities and normalizes the result to a steering
/// *direction*; an empty neighborhood (or neighbors that cancel out) yields
/// [`Vec3::ZERO`].
#[must_use]
pub fn alignment_force(velocities: &[Vec3], neighbors: &[u32]) -> Vec3 {
    if neighbors.is_empty() {
        return Vec3::ZERO;
    }
    let mut sum = Vec3::ZERO;
    for &n in neighbors {
        sum = sum.add(velocities[n as usize]);
    }
    let inv = 1.0 / neighbors.len() as f32;
    sum.scale(inv).normalize_or_zero()
}

/// Cohesion: steer toward the centroid of the `neighbors`.
///
/// Computes the neighbor centroid and returns the normalized direction from the
/// boid toward it; an empty neighborhood yields [`Vec3::ZERO`].
#[must_use]
pub fn cohesion_force(index: usize, positions: &[Vec3], neighbors: &[u32]) -> Vec3 {
    if neighbors.is_empty() {
        return Vec3::ZERO;
    }
    let mut sum = Vec3::ZERO;
    for &n in neighbors {
        sum = sum.add(positions[n as usize]);
    }
    let inv = 1.0 / neighbors.len() as f32;
    let centroid = sum.scale(inv);
    centroid.sub(positions[index]).normalize_or_zero()
}

/// Goal seeking: steer from `position` toward a `goal` point.
///
/// Returns the normalized direction to the goal; a goal coincident with the
/// boid yields [`Vec3::ZERO`]. Following a path is expressed by feeding the
/// current path target as `goal`.
#[must_use]
pub fn goal_force(position: Vec3, goal: Vec3) -> Vec3 {
    goal.sub(position).normalize_or_zero()
}

/// Gathers the visible neighbors of boid `index` through the spatial hash.
///
/// Queries [`NeighborGrid::query_ball`] within `limits.perception_radius`,
/// drops the boid itself, and applies the field-of-view cull: a neighbor is
/// kept only when `dot(heading, dir_to_neighbor) >= limits.fov_cos_threshold`.
/// A boid with (near) zero velocity has no heading and therefore perceives
/// every in-radius neighbor. Results keep the grid's stable index order and are
/// truncated to `max_neighbors`, so the sample is deterministic.
#[must_use]
pub fn visible_neighbors(
    index: usize,
    positions: &[Vec3],
    velocities: &[Vec3],
    grid: &NeighborGrid,
    limits: BoidsLimits,
    max_neighbors: u32,
) -> Vec<u32> {
    let self_pos = positions[index];
    let heading = velocities[index].normalize_or_zero();
    let has_heading = heading.length_squared() > EPS_LEN_SQ;
    let candidates = grid.query_ball(positions, self_pos, limits.perception_radius);
    let mut visible = Vec::new();
    for n in candidates {
        if n as usize == index {
            continue;
        }
        if has_heading {
            let dir = positions[n as usize].sub(self_pos).normalize_or_zero();
            if dir.dot(heading) < limits.fov_cos_threshold {
                continue;
            }
        }
        visible.push(n);
        if visible.len() as u32 >= max_neighbors {
            break;
        }
    }
    visible
}

/// Limits the turn implied by `steer` relative to the current `velocity`.
///
/// Splits `steer` into a component parallel to the heading (free to accelerate
/// or brake) and a perpendicular component (the turn), then clamps the
/// perpendicular part to `max_turn`. This is a transcendental-free tangential
/// proxy for a maximum turn angle. A boid with no heading is left free to pick
/// any initial direction and its steering is returned unchanged.
#[must_use]
pub fn limit_turn(steer: Vec3, velocity: Vec3, max_turn: f32) -> Vec3 {
    let heading = velocity.normalize_or_zero();
    if heading.length_squared() <= EPS_LEN_SQ {
        return steer;
    }
    let along = heading.scale(steer.dot(heading));
    let perpendicular = steer.sub(along);
    along.add(clamp_length(perpendicular, max_turn))
}

/// Weighted, fixed-order sum of the four steering contributions.
///
/// The accumulation order is *separation, alignment, cohesion, goal* and never
/// depends on neighbor discovery order, so the composed force is deterministic
/// (design §29). The result is the raw steering force *before* any limit
/// clamp; callers pass it through [`limit_turn`] and [`clamp_length`].
#[must_use]
pub fn combine_forces(
    separation: Vec3,
    alignment: Vec3,
    cohesion: Vec3,
    goal: Vec3,
    weights: BoidsWeights,
) -> Vec3 {
    let mut force = separation.scale(weights.separation);
    force = force.add(alignment.scale(weights.alignment));
    force = force.add(cohesion.scale(weights.cohesion));
    force = force.add(goal.scale(weights.goal));
    force
}

/// Computes the clamped steering acceleration for boid `index`.
///
/// Evaluates the three `Boids` rules over `neighbors` plus optional goal
/// seeking, composes them in the deterministic order of [`combine_forces`],
/// limits the turn against the current velocity, and finally clamps the
/// magnitude to `limits.max_force`. The returned vector is an acceleration to
/// feed [`integrate`].
#[must_use]
pub fn steer(
    index: usize,
    positions: &[Vec3],
    velocities: &[Vec3],
    neighbors: &[u32],
    goal: Option<Vec3>,
    weights: BoidsWeights,
    limits: BoidsLimits,
) -> Vec3 {
    let separation = separation_force(index, positions, neighbors);
    let alignment = alignment_force(velocities, neighbors);
    let cohesion = cohesion_force(index, positions, neighbors);
    let goal_dir = match goal {
        Some(target) => goal_force(positions[index], target),
        None => Vec3::ZERO,
    };
    let raw = combine_forces(separation, alignment, cohesion, goal_dir, weights);
    let turned = limit_turn(raw, velocities[index], limits.max_turn);
    clamp_length(turned, limits.max_force)
}

/// Integrates a boid's velocity under `acceleration` over `dt`, speed-clamped.
///
/// Semi-implicit velocity update `v' = clamp(v + a·dt, max_speed)`; the clamp
/// keeps the flock within its kinematic budget without any transcendental.
#[must_use]
pub fn integrate(velocity: Vec3, acceleration: Vec3, dt: f32, max_speed: f32) -> Vec3 {
    let stepped = velocity.add(acceleration.scale(dt));
    clamp_length(stepped, max_speed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-5
    }

    fn approx_vec(a: Vec3, b: Vec3) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
    }

    #[test]
    fn separation_pushes_away_from_a_close_neighbor() {
        // Self at origin, neighbor to the +x side: push must point toward -x.
        let positions = [Vec3::ZERO, Vec3::new(0.5, 0.0, 0.0)];
        let force = separation_force(0, &positions, &[1]);
        assert!(force.x < 0.0);
        assert!(approx(force.y, 0.0));
        assert!(approx(force.z, 0.0));
        // Normalized direction.
        assert!(approx(force.length(), 1.0));
    }

    #[test]
    fn separation_weights_the_closer_neighbor_more() {
        // A near neighbor on +x and a far one on -x: net push is toward -x
        // because the near one repels harder (inverse-distance weighting).
        let positions = [
            Vec3::ZERO,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(-4.0, 0.0, 0.0),
        ];
        let force = separation_force(0, &positions, &[1, 2]);
        assert!(force.x < 0.0);
    }

    #[test]
    fn alignment_matches_average_neighbor_heading() {
        // Both neighbors head +y, so alignment steers +y regardless of self.
        let velocities = [
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 4.0, 0.0),
        ];
        let force = alignment_force(&velocities, &[1, 2]);
        assert!(approx_vec(force, Vec3::new(0.0, 1.0, 0.0)));
    }

    #[test]
    fn cohesion_steers_toward_the_neighbor_centroid() {
        // Neighbors centroid is at +x, so cohesion from origin points +x.
        let positions = [
            Vec3::ZERO,
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
        ];
        let force = cohesion_force(0, &positions, &[1, 2]);
        assert!(approx_vec(force, Vec3::new(1.0, 0.0, 0.0)));
    }

    #[test]
    fn goal_force_points_at_the_target() {
        let force = goal_force(Vec3::ZERO, Vec3::new(0.0, 0.0, 5.0));
        assert!(approx_vec(force, Vec3::new(0.0, 0.0, 1.0)));
        // A goal at the boid yields no pull.
        assert_eq!(goal_force(Vec3::ZERO, Vec3::ZERO), Vec3::ZERO);
    }

    #[test]
    fn empty_neighborhood_yields_zero_for_every_rule() {
        let positions = [Vec3::ZERO];
        let velocities = [Vec3::ZERO];
        assert_eq!(separation_force(0, &positions, &[]), Vec3::ZERO);
        assert_eq!(alignment_force(&velocities, &[]), Vec3::ZERO);
        assert_eq!(cohesion_force(0, &positions, &[]), Vec3::ZERO);
    }

    #[test]
    fn field_of_view_culls_neighbors_behind_the_boid() {
        // Boid at origin moving +x. One neighbor ahead (+x), one behind (-x).
        let positions = [
            Vec3::ZERO,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(-0.5, 0.0, 0.0),
        ];
        let velocities = [Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO, Vec3::ZERO];
        let grid = NeighborGrid::build(&positions, 1.0, 32);
        // Forward pinhole-ish FOV keeps only the neighbor ahead.
        let limits = BoidsLimits::new(1.0, 0.5, 1.0, 1.0, 1.0);
        let visible = visible_neighbors(0, &positions, &velocities, &grid, limits, 64);
        assert_eq!(visible, vec![1]);
    }

    #[test]
    fn zero_heading_perceives_all_in_radius_neighbors() {
        let positions = [
            Vec3::ZERO,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(-0.5, 0.0, 0.0),
        ];
        let velocities = [Vec3::ZERO, Vec3::ZERO, Vec3::ZERO];
        let grid = NeighborGrid::build(&positions, 1.0, 32);
        let limits = BoidsLimits::new(1.0, 0.9, 1.0, 1.0, 1.0);
        let mut visible = visible_neighbors(0, &positions, &velocities, &grid, limits, 64);
        visible.sort_unstable();
        assert_eq!(visible, vec![1, 2]);
    }

    #[test]
    fn neighbor_sample_is_capped_and_index_ordered() {
        // Five in-radius, no heading: the cap keeps the first three by index.
        let positions = [
            Vec3::ZERO,
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(0.2, 0.0, 0.0),
            Vec3::new(0.3, 0.0, 0.0),
            Vec3::new(0.4, 0.0, 0.0),
        ];
        let velocities = [Vec3::ZERO; 5];
        let grid = NeighborGrid::build(&positions, 1.0, 32);
        let limits = BoidsLimits::new(1.0, -1.0, 1.0, 1.0, 1.0);
        let visible = visible_neighbors(0, &positions, &velocities, &grid, limits, 3);
        assert_eq!(visible, vec![1, 2, 3]);
    }

    #[test]
    fn clamp_length_bounds_only_when_over_the_limit() {
        let short = Vec3::new(0.3, 0.0, 0.0);
        assert!(approx_vec(clamp_length(short, 1.0), short));
        let long = Vec3::new(3.0, 4.0, 0.0); // length 5
        let clamped = clamp_length(long, 1.0);
        assert!(approx(clamped.length(), 1.0));
        assert!(approx_vec(clamped, Vec3::new(0.6, 0.8, 0.0)));
        // Non-positive limit collapses to zero.
        assert_eq!(clamp_length(long, 0.0), Vec3::ZERO);
    }

    #[test]
    fn limit_turn_clamps_the_perpendicular_component() {
        // Heading +x, steer strongly +y: the +y (turn) part is capped, the
        // along-heading part passes through untouched.
        let velocity = Vec3::new(2.0, 0.0, 0.0);
        let steer = Vec3::new(1.0, 5.0, 0.0);
        let turned = limit_turn(steer, velocity, 0.5);
        assert!(approx(turned.x, 1.0));
        assert!(approx(turned.y, 0.5));
        assert!(approx(turned.z, 0.0));
    }

    #[test]
    fn limit_turn_is_identity_without_a_heading() {
        let steer = Vec3::new(1.0, 5.0, 0.0);
        assert!(approx_vec(limit_turn(steer, Vec3::ZERO, 0.5), steer));
    }

    #[test]
    fn combine_forces_uses_the_fixed_weighted_order() {
        let sep = Vec3::new(1.0, 0.0, 0.0);
        let align = Vec3::new(0.0, 1.0, 0.0);
        let coh = Vec3::new(0.0, 0.0, 1.0);
        let goal = Vec3::new(1.0, 1.0, 1.0);
        let w = BoidsWeights::new(2.0, 3.0, 4.0, 5.0);
        let expected = sep
            .scale(2.0)
            .add(align.scale(3.0))
            .add(coh.scale(4.0))
            .add(goal.scale(5.0));
        assert!(approx_vec(
            combine_forces(sep, align, coh, goal, w),
            expected
        ));
    }

    #[test]
    fn combine_forces_is_order_independent_in_inputs() {
        // Determinism: the same contributions always compose to the same force
        // no matter how the caller discovered the neighbors.
        let sep = Vec3::new(0.2, -0.4, 0.1);
        let align = Vec3::new(-0.3, 0.5, 0.7);
        let coh = Vec3::new(0.9, 0.1, -0.2);
        let goal = Vec3::new(-0.1, -0.6, 0.3);
        let w = BoidsWeights::classic_flock();
        let a = combine_forces(sep, align, coh, goal, w);
        let b = combine_forces(sep, align, coh, goal, w);
        assert_eq!(a, b);
    }

    #[test]
    fn steer_is_bounded_by_max_force() {
        let positions = [Vec3::ZERO, Vec3::new(0.2, 0.0, 0.0)];
        let velocities = [Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO];
        let weights = BoidsWeights::new(10.0, 10.0, 10.0, 10.0);
        let limits = BoidsLimits::new(1.0, -1.0, 5.0, 0.5, 10.0);
        let accel = steer(
            0,
            &positions,
            &velocities,
            &[1],
            Some(Vec3::new(0.0, 0.0, 9.0)),
            weights,
            limits,
        );
        assert!(accel.length() <= 0.5 + 1e-5);
    }

    #[test]
    fn steer_is_deterministic_across_repeated_calls() {
        let positions = [
            Vec3::ZERO,
            Vec3::new(0.3, 0.1, 0.0),
            Vec3::new(-0.2, 0.4, 0.1),
        ];
        let velocities = [
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.1, 0.2, 0.0),
            Vec3::new(-0.1, 0.3, 0.0),
        ];
        let weights = BoidsWeights::classic_flock();
        let limits = BoidsLimits::new(2.0, -1.0, 3.0, 2.0, 1.5);
        let a = steer(0, &positions, &velocities, &[1, 2], None, weights, limits);
        let b = steer(0, &positions, &velocities, &[1, 2], None, weights, limits);
        assert_eq!(a, b);
    }

    #[test]
    fn integrate_clamps_to_max_speed() {
        let v = integrate(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            1.0,
            2.0,
        );
        assert!(approx(v.length(), 2.0));
        assert!(approx_vec(v, Vec3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn quality_budget_degrades_monotonically() {
        let low = boids_quality_budget(BoidsQuality::Low);
        let medium = boids_quality_budget(BoidsQuality::Medium);
        let high = boids_quality_budget(BoidsQuality::High);
        let ultra = boids_quality_budget(BoidsQuality::Ultra);
        assert!(low.perception_scale < medium.perception_scale);
        assert!(medium.perception_scale < high.perception_scale);
        assert!(high.perception_scale < ultra.perception_scale);
        assert!(low.max_neighbors < medium.max_neighbors);
        assert!(medium.max_neighbors < high.max_neighbors);
        assert!(high.max_neighbors < ultra.max_neighbors);
        assert!(approx(ultra.scaled_radius(4.0), 4.0));
        assert!(approx(low.scaled_radius(4.0), 2.0));
    }

    #[test]
    fn boid_cell_matches_the_grid_lattice() {
        let p = Vec3::new(1.5, -0.5, 2.5);
        assert_eq!(boid_cell(p, 1.0), NeighborGrid::cell_of(p, 1.0));
    }
}
