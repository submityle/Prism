//! Broad-phase-accelerated closest-point distance query: for a world-space
//! query point, find the single nearest static target and the exact closest
//! point on its (optionally rounded) surface, the outward surface normal there,
//! the surface distance, and whether the point lies inside the target. This is
//! the one-against-many distance query `AAA` engines expose as `PhysX`
//! `PxGeometryQuery::pointDistance`, Jolt's narrow-phase point-distance helper,
//! and Unreal `Chaos` nearest-point queries. It is a distinct capability class
//! from the containment query [`collide_point`](super::collide_point_bvh): that
//! one answers only "which targets is the point inside"; this one additionally
//! returns, for a point *outside* every target, the nearest surface point and
//! how far away it is.
//!
//! # Semantics
//!
//! Each target is a rounded convex: a core [`ConvexHull`] posed in the world and
//! inflated by a convex rounding [`RoundedConvex::radius`]. For the query point:
//!
//! * **Outside** the rounded surface: the [`ClosestPointHit::distance`] is the
//!   positive surface distance, [`ClosestPointHit::point`] is the closest point
//!   *on the rounded surface*, [`ClosestPointHit::normal`] is the outward unit
//!   surface normal there (pointing from the target toward the query point), and
//!   [`ClosestPointHit::inside`] is `false`.
//! * **Inside** the rounded surface (within the core, or within the rounding
//!   skin): the distance is `0`, `inside` is `true`, the point is the query point
//!   itself, and the normal is zero. Recovering a signed penetration depth and
//!   direction for an interior point is the job of the expanding-polytope
//!   algorithm, which this query deliberately does not run — matching `PhysX`
//!   `pointDistance`, whose result is `0` and whose closest point is undefined
//!   for an interior query.
//!
//! The nearest target is the one of least surface distance, ties broken by
//! ascending target index. An interior point has distance `0` against every
//! target that contains it, so the lowest-indexed containing target wins.
//!
//! # Why the `BVH` query equals brute force
//!
//! The core distance between the query point and a target is a `GJK` distance
//! walk (the point is a one-vertex hull). The brute query runs it against every
//! target and keeps the least surface distance; the `BVH` query descends a
//! linear `BVH` over the targets' rounded world bounds and prunes a subtree only
//! when the subtree's bounding box is strictly farther from the query point than
//! the best surface distance found so far. Because each target's box is its core
//! bound grown by the rounding radius, the point-to-box distance is a lower
//! bound on that target's surface distance, so a pruned subtree can contain no
//! target that is nearer than — or tied with a lower index than — the current
//! best. The `BVH` query therefore visits a superset of the targets that could
//! change the answer and applies the identical distance-then-index rule, so it
//! returns the identical hit. The unit tests assert that equivalence scene for
//! scene against [`closest_point`].
//!
//! The `GPU` twin
//! [`GpuSceneClosestPoint`](super::closest_point_bvh_gpu::GpuSceneClosestPoint)
//! runs the identical per-target `GJK` distance on the device and reduces on the
//! host; it is pinned to this brute golden by its own real-device parity suite.
//!
//! # Provenance
//!
//! Closest-point distance via a Gilbert-Johnson-Keerthi distance walk (Gilbert,
//! Johnson, and Keerthi, 1988) with the Voronoi sub-distance of Ericson (2005),
//! gathered through the linear `BVH` of Karras (2012) with a branch-and-bound
//! prune after the classic nearest-neighbour search of Friedman, Bentley, and
//! Finkel (1977). No Unreal Engine source or derived code.

use core::cmp::Ordering;

use glam::{Quat, Vec3};

use crate::bvh::{cpu_build_lbvh, Aabb, Lbvh};

use super::convex_hull::ConvexHull;
use super::convex_pose::ConvexPose;
use super::gjk::{gjk, GjkStatus};
use super::shape_cast::RoundedConvex;

/// Surface distance at or below which the query point is treated as lying on or
/// inside the rounded surface rather than outside it.
const SURFACE_EPS: f32 = 1.0e-6;

/// A world-space query point for a closest-point distance query.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SceneClosestPoint {
    /// World position whose nearest target surface point is sought.
    pub position: Vec3,
}

impl SceneClosestPoint {
    /// Builds a closest-point query at `position`.
    #[must_use]
    pub fn new(position: Vec3) -> SceneClosestPoint {
        SceneClosestPoint { position }
    }
}

/// The nearest target to a query point, with the closest surface point, the
/// outward surface normal there, the surface distance, and whether the query
/// point is inside the target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClosestPointHit {
    /// Index of the nearest target in the `targets` slice handed to the query.
    pub target: u32,
    /// Closest point on the target's rounded surface in world space. For an
    /// interior query point this is the query point itself.
    pub point: Vec3,
    /// Surface distance from the query point to the target: positive when the
    /// point is outside, `0` when it is inside.
    pub distance: f32,
    /// Outward unit surface normal at [`ClosestPointHit::point`], pointing from
    /// the target toward the query point. Zero when the query point is inside.
    pub normal: Vec3,
    /// Whether the query point lies inside the target's rounded surface.
    pub inside: bool,
}

/// The query point modelled as a single-vertex convex hull, so a point-versus
/// target distance is an ordinary two-hull `GJK` distance. The hull is a shared
/// constant; the per-query position rides in the pose translation.
fn point_pose(position: Vec3) -> ConvexPose {
    ConvexPose::new(position, Quat::IDENTITY)
}

/// Evaluates the closest-point hit of a single target against the query point.
///
/// Runs the `GJK` distance between the one-vertex query hull and the target's
/// core hull, then folds in the rounding radius through
/// [`closest_hit_from_core`] — the single surface-and-inside rule the `GPU`
/// twin also applies to its device `GJK` output, so the two queries agree
/// exactly.
fn evaluate(point_hull: &ConvexHull, position: Vec3, target: &RoundedConvex, index: u32) -> ClosestPointHit {
    let pose = point_pose(position);
    match gjk(point_hull, &pose, target.hull, &target.pose) {
        GjkStatus::Separated {
            distance,
            point_b,
            normal,
            ..
        } => closest_hit_from_core(index, target.radius, false, distance, point_b, normal, position),
        GjkStatus::Intersecting(_) => {
            closest_hit_from_core(index, target.radius, true, 0.0, position, Vec3::ZERO, position)
        }
    }
}

/// Folds a core `GJK` distance result and a target's rounding radius into the
/// final [`ClosestPointHit`]. This is the one surface-and-inside rule shared by
/// the brute and `BVH` `CPU` queries and the `GPU` twin: given whether the
/// query point's one-vertex hull intersects the target core, the core
/// `core_distance`, the core closest point `point_b` on the target, the outward
/// unit `normal` (pointing from the target toward the query point), and the
/// query `position`, it returns the rounded-surface hit. A separated point
/// whose surface distance `core_distance - radius` exceeds [`SURFACE_EPS`] is
/// outside: its surface point is `point_b` pushed out along the normal by the
/// radius. Any closer or intersecting point is inside at distance `0`.
pub(crate) fn closest_hit_from_core(
    index: u32,
    radius: f32,
    intersecting: bool,
    core_distance: f32,
    point_b: Vec3,
    normal: Vec3,
    position: Vec3,
) -> ClosestPointHit {
    if intersecting {
        return ClosestPointHit {
            target: index,
            point: position,
            distance: 0.0,
            normal: Vec3::ZERO,
            inside: true,
        };
    }
    let surface = core_distance - radius;
    if surface <= SURFACE_EPS {
        ClosestPointHit {
            target: index,
            point: position,
            distance: 0.0,
            normal: Vec3::ZERO,
            inside: true,
        }
    } else {
        ClosestPointHit {
            target: index,
            point: point_b + normal * radius,
            distance: surface,
            normal,
            inside: false,
        }
    }
}

/// Whether `candidate` is a strictly better closest-point hit than `best`: a
/// smaller surface distance wins, and an exact distance tie is broken by the
/// lower target index. Uses [`f32::total_cmp`] so the ordering is total and
/// deterministic without a floating-point equality comparison.
pub(crate) fn better(candidate: &ClosestPointHit, best: &ClosestPointHit) -> bool {
    match candidate.distance.total_cmp(&best.distance) {
        Ordering::Less => true,
        Ordering::Equal => candidate.target < best.target,
        Ordering::Greater => false,
    }
}

/// Brute-force closest-point query: the `golden` the `BVH` query is checked
/// against. Runs the `GJK` distance against every target and keeps the least
/// surface distance, ties broken by ascending target index. Returns `None` only
/// when there are no targets.
#[must_use]
pub fn closest_point(targets: &[RoundedConvex], query: &SceneClosestPoint) -> Option<ClosestPointHit> {
    let point_hull = ConvexHull::from_point();
    let mut best: Option<ClosestPointHit> = None;
    for (i, target) in targets.iter().enumerate() {
        let hit = evaluate(&point_hull, query.position, target, i as u32);
        best = Some(match best {
            Some(b) if !better(&hit, &b) => b,
            _ => hit,
        });
    }
    best
}

/// The world-space bounding box of a rounded target: the core hull's world
/// vertices' bounds grown by the rounding radius. The point-to-box distance of
/// this box is a lower bound on the target's surface distance, which the `BVH`
/// prune relies on.
fn target_aabb(target: &RoundedConvex) -> Aabb {
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for v in target.hull.vertices() {
        let world = target.pose.rotation * *v + target.pose.translation;
        lo = lo.min(world);
        hi = hi.max(world);
    }
    let r = Vec3::splat(target.radius);
    Aabb::new(lo - r, hi + r)
}

/// The distance from a point to the nearest point of an axis-aligned box, `0`
/// when the point is inside the box. A lower bound on the surface distance of
/// any target whose rounded bound is `aabb`.
fn aabb_point_distance(aabb: &Aabb, point: Vec3) -> f32 {
    let clamped = point.clamp(aabb.min, aabb.max);
    (clamped - point).length()
}

/// The bounding box of an encoded `BVH` node: its leaf box when the id names a
/// leaf, otherwise its internal-node union.
fn node_aabb(tree: &Lbvh, encoded: u32) -> Aabb {
    if tree.is_leaf(encoded) {
        tree.leaf_aabb[(encoded as usize) - tree.num_internal]
    } else {
        tree.internal_aabb[encoded as usize]
    }
}

/// Broad-phase-accelerated form of [`closest_point`]: the nearest target, found
/// by descending a linear `BVH` over the targets' rounded world bounds and
/// pruning any subtree whose box is strictly farther from the query point than
/// the best surface distance so far. The result matches [`closest_point`]
/// exactly, including the lower-index rule on an exact distance tie. Returns
/// `None` only when there are no targets.
#[must_use]
pub fn closest_point_bvh(targets: &[RoundedConvex], query: &SceneClosestPoint) -> Option<ClosestPointHit> {
    if targets.is_empty() {
        return None;
    }
    let point_hull = ConvexHull::from_point();
    let boxes: Vec<Aabb> = targets.iter().map(target_aabb).collect();
    let tree = cpu_build_lbvh(&boxes);

    let mut best: Option<ClosestPointHit> = None;
    let mut stack = vec![tree.root];
    while let Some(node) = stack.pop() {
        let lower_bound = aabb_point_distance(&node_aabb(&tree, node), query.position);
        if let Some(b) = best
            && lower_bound > b.distance
        {
            continue;
        }
        if tree.is_leaf(node) {
            let leaf = (node as usize) - tree.num_internal;
            let prim = tree.sorted_indices[leaf] as usize;
            let hit = evaluate(&point_hull, query.position, &targets[prim], prim as u32);
            best = Some(match best {
                Some(b) if !better(&hit, &b) => b,
                _ => hit,
            });
        } else {
            stack.push(tree.left[node as usize]);
            stack.push(tree.right[node as usize]);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An axis-aligned unit box hull (half-extent `1` on every axis).
    fn unit_box() -> ConvexHull {
        ConvexHull::from_box(Vec3::splat(1.0))
    }

    /// A stationary target hull centred at `center` with the given rounding.
    fn target_at(hull: &ConvexHull, center: Vec3, radius: f32) -> RoundedConvex<'_> {
        RoundedConvex::still(hull, ConvexPose::new(center, Quat::IDENTITY), radius)
    }

    #[test]
    fn closest_point_outside_reports_surface_point_distance_and_normal() {
        // A unit box at the origin; a query 3 along +x. The nearest surface is
        // the +x face at x = 1, so the surface point is (1,0,0), the distance 2,
        // the outward normal +x, and the point is outside.
        let hull = unit_box();
        let targets = [target_at(&hull, Vec3::ZERO, 0.0)];
        let query = SceneClosestPoint::new(Vec3::new(3.0, 0.0, 0.0));
        let hit = closest_point(&targets, &query).expect("one target is always nearest");
        assert_eq!(hit.target, 0);
        assert!(!hit.inside, "the point is outside the box");
        assert!((hit.distance - 2.0).abs() < 1e-4, "distance was {}", hit.distance);
        assert!((hit.point - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-4, "point was {:?}", hit.point);
        assert!((hit.normal - Vec3::X).length() < 1e-4, "normal was {:?}", hit.normal);
        assert_eq!(closest_point_bvh(&targets, &query), Some(hit), "BVH must equal brute force");
    }

    #[test]
    fn closest_point_rounding_pushes_surface_out_and_shortens_distance() {
        // The same box inflated by radius 0.5: the rounded surface sits half a
        // unit nearer the query, so the distance drops by the radius and the
        // surface point moves out to x = 1.5.
        let hull = unit_box();
        let targets = [target_at(&hull, Vec3::ZERO, 0.5)];
        let query = SceneClosestPoint::new(Vec3::new(3.0, 0.0, 0.0));
        let hit = closest_point(&targets, &query).expect("one target is always nearest");
        assert!(!hit.inside, "the point is still outside the rounded box");
        assert!((hit.distance - 1.5).abs() < 1e-4, "distance was {}", hit.distance);
        assert!((hit.point - Vec3::new(1.5, 0.0, 0.0)).length() < 1e-4, "point was {:?}", hit.point);
        assert_eq!(closest_point_bvh(&targets, &query), Some(hit), "BVH must equal brute force");
    }

    #[test]
    fn closest_point_inside_reports_zero_distance_and_inside() {
        // A query at the box centre is interior: distance 0, inside true, and
        // the reported point is the query point itself.
        let hull = unit_box();
        let targets = [target_at(&hull, Vec3::ZERO, 0.0)];
        let query = SceneClosestPoint::new(Vec3::new(0.2, -0.1, 0.3));
        let hit = closest_point(&targets, &query).expect("one target is always nearest");
        assert!(hit.inside, "the point is inside the box");
        assert_eq!(hit.distance, 0.0, "an interior point has zero surface distance");
        assert!((hit.point - query.position).length() < 1e-6, "interior point is the query point");
        assert_eq!(closest_point_bvh(&targets, &query), Some(hit), "BVH must equal brute force");
    }

    #[test]
    fn closest_point_picks_nearest_of_many_like_brute_force() {
        // Three boxes strung along +x; a query near the middle one must pick it.
        let hulls = [unit_box(), unit_box(), unit_box()];
        let targets = [
            target_at(&hulls[0], Vec3::new(0.0, 0.0, 0.0), 0.0),
            target_at(&hulls[1], Vec3::new(8.0, 0.0, 0.0), 0.0),
            target_at(&hulls[2], Vec3::new(16.0, 0.0, 0.0), 0.0),
        ];
        let query = SceneClosestPoint::new(Vec3::new(9.0, 0.0, 0.0));
        let brute = closest_point(&targets, &query).expect("a nearest target exists");
        assert_eq!(brute.target, 1, "the middle box is nearest");
        assert_eq!(closest_point_bvh(&targets, &query), Some(brute), "BVH must equal brute force");
    }

    #[test]
    fn closest_point_bvh_matches_brute_force_over_a_large_scene() {
        // A grid of boxes plus a far-off wall; the BVH prune must still land on
        // the identical nearest target and surface geometry as brute force for a
        // battery of query points.
        let mut hulls = Vec::new();
        let mut centers = Vec::new();
        for gx in 0..5 {
            for gz in 0..5 {
                hulls.push(unit_box());
                centers.push(Vec3::new(4.0 * (gx as f32), 0.0, 4.0 * (gz as f32)));
            }
        }
        for k in 0..10 {
            hulls.push(unit_box());
            centers.push(Vec3::new(2.0 * (k as f32), 80.0, 0.0));
        }
        let targets: Vec<RoundedConvex> = (0..hulls.len())
            .map(|i| target_at(&hulls[i], centers[i], if i % 3 == 0 { 0.3 } else { 0.0 }))
            .collect();
        for q in [
            Vec3::new(5.0, 1.0, 5.0),
            Vec3::new(-3.0, 0.0, 10.0),
            Vec3::new(9.5, 0.2, 2.1),
            Vec3::new(16.0, 2.0, 16.0),
            Vec3::new(7.0, 0.0, 7.0),
        ] {
            let query = SceneClosestPoint::new(q);
            assert_eq!(
                closest_point(&targets, &query),
                closest_point_bvh(&targets, &query),
                "BVH must equal brute force at query {q:?}"
            );
        }
    }

    #[test]
    fn closest_point_no_targets_is_none() {
        let targets: [RoundedConvex; 0] = [];
        let query = SceneClosestPoint::new(Vec3::ZERO);
        assert_eq!(closest_point(&targets, &query), None);
        assert_eq!(closest_point_bvh(&targets, &query), None);
    }
}
