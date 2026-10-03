//! General convex *shape-cast* (sweep) queries against analytic shapes.
//!
//! A shape-cast moves an arbitrary bounded convex collider (a sphere, box, or
//! capsule) from a start pose along a fixed world-space `motion` vector and
//! reports the first body it touches, together with the world-space contact
//! point and the outward contact normal. This generalises
//! [`spherecast`](crate::world::PhysicsWorld::spherecast), which can only sweep
//! a sphere, to the swept box/capsule queries that character controllers,
//! cameras, and projectiles need in UE/Chaos, `PhysX`, and Jolt.
//!
//! # Dispatch
//!
//! - Against a bounded convex target (sphere/box/capsule) the mover and target
//!   are both exposed as geometry [`SupportMap`](prism_physics_geometry::SupportMap)s
//!   via [`CcdSupport`] and handed to the shared
//!   [`conservative_advancement`] time-of-impact solver, so the sweep respects
//!   the target's true orientation rather than a bounding sphere.
//! - Against a [`ColliderShape::Plane`] (an unbounded two-sided half-space that
//!   can never itself be a mover) the sweep is solved analytically: the mover's
//!   extreme support point toward the plane travels a straight line, so the
//!   time of impact is a single ratio.
//!
//! The reported `time_of_impact` is the **distance** the mover's origin travels
//! before contact (`toi_fraction * |motion|`), matching the distance semantics
//! of [`SweepHit`](crate::query::SweepHit). The reported normal is the outward
//! surface normal of the *target*, i.e. it points back toward the mover.
//!
//! # Provenance
//!
//! The conservative-advancement reduction is this workspace's own geometry
//! solver, and the swept-convex-versus-half-space ratio is a standard
//! computational-geometry result (see Ericson, *Real-Time Collision
//! Detection*). This file contains no Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_geometry::{conservative_advancement, Aabb, SupportMap, TimeOfImpact};

use crate::ccd::triangle_support::TriangleSupport;
use crate::ccd::CcdSupport;
use crate::collider::{ColliderShape, ShapeRegistry, TriMeshHandle};
use crate::math::scalar::Real;
use crate::math::transform::Isometry;
use crate::query::ray::RayShapeHit;

/// Separation (world units) at which the swept mover is declared in contact.
///
/// Mirrors the tolerance the CCD path feeds [`conservative_advancement`]: large
/// enough to keep the GJK witness well-conditioned, small enough that the
/// reported time of impact is tight.
const SWEEP_CONTACT_TOLERANCE: Real = 1.0e-4;

/// Below this squared motion magnitude the cast degenerates to a stationary
/// overlap test and reports no swept contact.
const MIN_MOTION_SQ: Real = 1.0e-12;

/// Sweeps `mover` (posed at `mover_pose`) along `motion` against `target`
/// (posed at `target_pose`) and returns the nearest contact, or [`None`] when
/// they never touch over the motion interval.
///
/// Returns [`None`] when `motion` is negligible, when `mover` is not a bounded
/// convex volume (a plane can never be swept), or when the shapes miss.
pub(crate) fn shapecast_shape(
    mover: &ColliderShape,
    mover_pose: &Isometry,
    target: &ColliderShape,
    target_pose: &Isometry,
    motion: Vec3,
    shapes: &ShapeRegistry,
) -> Option<RayShapeHit> {
    if motion.length_squared() <= MIN_MOTION_SQ {
        return None;
    }

    // A plane target is an unbounded half-space with no support map; solve it
    // analytically. (A plane can never be the mover, so this ordering is safe.)
    if let ColliderShape::Plane { normal, offset } = *target {
        return sweep_convex_vs_plane(
            mover,
            mover_pose,
            target_pose,
            normal,
            offset,
            motion,
            shapes,
        );
    }

    // A triangle-mesh target is concave and has no single support map; sweep
    // the mover against each candidate triangle (broad-phase culled by the
    // mover's swept AABB) and keep the earliest contact.
    if let ColliderShape::TriangleMesh { mesh, .. } = *target {
        return sweep_convex_vs_trimesh(mover, mover_pose, target_pose, mesh, motion, shapes);
    }

    let mover_support =
        CcdSupport::from_shape_in(mover, mover_pose.translation, mover_pose.rotation, shapes)?;
    let target_support = CcdSupport::from_shape_in(
        target,
        target_pose.translation,
        target_pose.rotation,
        shapes,
    )?;

    let toi = conservative_advancement(
        &mover_support,
        &target_support,
        motion,
        SWEEP_CONTACT_TOLERANCE,
    )?;

    Some(RayShapeHit {
        // `conservative_advancement` reports the fraction of `motion`; convert
        // it to the distance the mover origin travelled before contact.
        time_of_impact: toi.toi * motion.length(),
        point: toi.point,
        // The solver's normal points from the mover toward the target; the
        // query contract reports the target's *outward* normal (toward mover).
        normal: -toi.normal,
    })
}

/// Analytic sweep of a bounded convex `mover` against a two-sided plane.
///
/// The plane is `n · x = d` in world space. Whichever side the mover starts on,
/// its extreme support point toward the plane is the first point to touch, and
/// under pure translation that extreme vertex travels a straight line, so the
/// contact time is `gap / closing_speed`.
fn sweep_convex_vs_plane(
    mover: &ColliderShape,
    mover_pose: &Isometry,
    plane_pose: &Isometry,
    plane_normal_local: Vec3,
    plane_offset: Real,
    motion: Vec3,
    shapes: &ShapeRegistry,
) -> Option<RayShapeHit> {
    let mover_support =
        CcdSupport::from_shape_in(mover, mover_pose.translation, mover_pose.rotation, shapes)?;

    let n = plane_pose
        .transform_vector(plane_normal_local)
        .normalize_or_zero();
    if n == Vec3::ZERO {
        return None;
    }
    let point_on_plane = plane_pose.transform_point(plane_normal_local * plane_offset);
    let plane_d = n.dot(point_on_plane);

    // Extremes of the mover along +/- n select the side and the touching vertex.
    let support_pos = mover_support.support_point(n);
    let support_neg = mover_support.support_point(-n);
    let s_max = n.dot(support_pos);
    let s_min = n.dot(support_neg);

    let closing = n.dot(motion);

    // `(gap, contact_vertex, outward_normal, approach_speed)` by starting side.
    let (gap, contact_vertex, outward_normal, approach) = if s_min >= plane_d {
        // Mover sits on the +n side; its deepest point toward the plane is the
        // -n extreme, which must move in the -n direction (closing < 0) to hit.
        (s_min - plane_d, support_neg, n, -closing)
    } else if s_max <= plane_d {
        // Mover sits on the -n side; its deepest point toward the plane is the
        // +n extreme, which must move in the +n direction (closing > 0) to hit.
        (plane_d - s_max, support_pos, -n, closing)
    } else {
        // The mover straddles the plane: it already overlaps at t = 0.
        let vertex = if s_max - plane_d >= plane_d - s_min {
            support_pos
        } else {
            support_neg
        };
        let outward = if closing <= 0.0 { n } else { -n };
        return Some(RayShapeHit {
            time_of_impact: 0.0,
            point: vertex,
            normal: outward,
        });
    };

    if approach <= SWEEP_CONTACT_TOLERANCE {
        // Moving away from or tangent to the plane: no contact this interval.
        return None;
    }

    let toi = gap / approach;
    if !(0.0..=1.0).contains(&toi) {
        return None;
    }

    Some(RayShapeHit {
        time_of_impact: toi * motion.length(),
        point: contact_vertex + motion * toi,
        normal: outward_normal,
    })
}

/// Sweeps a bounded convex `mover` against a concave triangle-mesh target.
///
/// The mesh exposes no single support map, so the mover's swept AABB (its local
/// box translated along `motion`, pulled into the target's local frame) selects
/// candidate triangles via the mesh broad phase. Each candidate becomes a
/// [`TriangleSupport`] in world space and is advanced against with the same
/// [`conservative_advancement`] reduction used for convex targets; the earliest
/// contact over all triangles wins.
fn sweep_convex_vs_trimesh(
    mover: &ColliderShape,
    mover_pose: &Isometry,
    target_pose: &Isometry,
    mesh: TriMeshHandle,
    motion: Vec3,
    shapes: &ShapeRegistry,
) -> Option<RayShapeHit> {
    let mover_support =
        CcdSupport::from_shape_in(mover, mover_pose.translation, mover_pose.rotation, shapes)?;
    let tri_data = shapes.tri_mesh(mesh)?;
    let mesh_geom = tri_data.mesh();
    if mesh_geom.is_empty() {
        return None;
    }

    // Build the mover's swept AABB in world space, then pull its corners into
    // the target's local frame for the triangle broad phase.
    let (local_min, local_max) = mover.local_aabb();
    let local_corners = [
        Vec3::new(local_min.x, local_min.y, local_min.z),
        Vec3::new(local_max.x, local_min.y, local_min.z),
        Vec3::new(local_min.x, local_max.y, local_min.z),
        Vec3::new(local_max.x, local_max.y, local_min.z),
        Vec3::new(local_min.x, local_min.y, local_max.z),
        Vec3::new(local_max.x, local_min.y, local_max.z),
        Vec3::new(local_min.x, local_max.y, local_max.z),
        Vec3::new(local_max.x, local_max.y, local_max.z),
    ];
    let inv_target = target_pose.inverse();
    let mut query_points = [Vec3::ZERO; 16];
    for (i, corner) in local_corners.iter().enumerate() {
        let world = mover_pose.transform_point(*corner);
        query_points[i] = inv_target.transform_point(world);
        query_points[i + 8] = inv_target.transform_point(world + motion);
    }
    let query_aabb = Aabb::from_points(&query_points)?;

    let mut best: Option<TimeOfImpact> = None;
    for tri_index in mesh_geom.overlap_aabb(&query_aabb) {
        let Some([a, b, c]) = mesh_geom.triangle(tri_index as usize) else {
            continue;
        };
        let tri = TriangleSupport::new(
            target_pose.transform_point(a),
            target_pose.transform_point(b),
            target_pose.transform_point(c),
        );
        if let Some(toi) =
            conservative_advancement(&mover_support, &tri, motion, SWEEP_CONTACT_TOLERANCE)
            && best.is_none_or(|current| toi.toi < current.toi)
        {
            best = Some(toi);
        }
    }

    let toi = best?;
    Some(RayShapeHit {
        time_of_impact: toi.toi * motion.length(),
        point: toi.point,
        // The solver normal points from the mover toward the triangle; the query
        // contract reports the target's outward normal (toward the mover).
        normal: -toi.normal,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    const EPS: Real = 1.0e-3;

    fn iso(translation: Vec3) -> Isometry {
        Isometry::new(translation, Quat::IDENTITY)
    }

    #[test]
    fn box_sweep_hits_box_head_on() {
        // Unit box centred at origin, swept +X toward a unit box at x = 5.
        let mover = ColliderShape::Cuboid {
            half_extents: Vec3::ONE,
        };
        let target = ColliderShape::Cuboid {
            half_extents: Vec3::ONE,
        };
        let hit = shapecast_shape(
            &mover,
            &iso(Vec3::ZERO),
            &target,
            &iso(Vec3::new(5.0, 0.0, 0.0)),
            Vec3::new(10.0, 0.0, 0.0),
            &ShapeRegistry::new(),
        )
        .expect("a box swept along +X must hit a box ahead of it");

        // Faces touch when the gap of 3 (centres 5 apart minus two half-widths)
        // is closed, i.e. after the origin travels 3 units.
        assert!(
            (hit.time_of_impact - 3.0).abs() < EPS,
            "distance was {}",
            hit.time_of_impact
        );
        assert!(
            hit.normal.dot(Vec3::X) < -0.9,
            "outward normal should face back toward the mover (-X), was {:?}",
            hit.normal
        );
    }

    #[test]
    fn box_sweep_misses_when_offset_sideways() {
        let mover = ColliderShape::Cuboid {
            half_extents: Vec3::splat(0.5),
        };
        let target = ColliderShape::Cuboid {
            half_extents: Vec3::splat(0.5),
        };
        // Target is offset far along +Y, so a +X sweep slides past it.
        let hit = shapecast_shape(
            &mover,
            &iso(Vec3::ZERO),
            &target,
            &iso(Vec3::new(5.0, 10.0, 0.0)),
            Vec3::new(10.0, 0.0, 0.0),
            &ShapeRegistry::new(),
        );
        assert!(hit.is_none(), "a sideways-offset box must be missed");
    }

    #[test]
    fn capsule_sweep_hits_sphere() {
        let mover = ColliderShape::Capsule {
            half_height: 1.0,
            radius: 0.5,
        };
        let target = ColliderShape::Sphere { radius: 1.0 };
        let hit = shapecast_shape(
            &mover,
            &iso(Vec3::ZERO),
            &target,
            &iso(Vec3::new(6.0, 0.0, 0.0)),
            Vec3::new(10.0, 0.0, 0.0),
            &ShapeRegistry::new(),
        )
        .expect("a capsule swept +X must hit a sphere ahead of it");

        // Capsule radius 0.5 + sphere radius 1.0 = 1.5; centres 6 apart, so the
        // surfaces meet after the origin travels 6 - 1.5 = 4.5.
        assert!(
            (hit.time_of_impact - 4.5).abs() < 2.0e-2,
            "distance was {}",
            hit.time_of_impact
        );
    }

    #[test]
    fn box_sweep_hits_ground_plane_from_above() {
        // Plane y = 0 with +Y normal; box of half-height 1 dropped from y = 5.
        let mover = ColliderShape::Cuboid {
            half_extents: Vec3::ONE,
        };
        let target = ColliderShape::Plane {
            normal: Vec3::Y,
            offset: 0.0,
        };
        let hit = shapecast_shape(
            &mover,
            &iso(Vec3::new(0.0, 5.0, 0.0)),
            &target,
            &iso(Vec3::ZERO),
            Vec3::new(0.0, -10.0, 0.0),
            &ShapeRegistry::new(),
        )
        .expect("a box falling onto a ground plane must hit it");

        // Bottom face at y = 4 reaches y = 0 after travelling 4 units.
        assert!(
            (hit.time_of_impact - 4.0).abs() < EPS,
            "distance was {}",
            hit.time_of_impact
        );
        assert!(
            hit.normal.dot(Vec3::Y) > 0.9,
            "ground outward normal should face +Y, was {:?}",
            hit.normal
        );
        assert!(
            (hit.point.y).abs() < EPS,
            "contact should land on the plane, was {:?}",
            hit.point
        );
    }

    #[test]
    fn plane_sweep_moving_away_misses() {
        let mover = ColliderShape::Sphere { radius: 1.0 };
        let target = ColliderShape::Plane {
            normal: Vec3::Y,
            offset: 0.0,
        };
        // Sphere above the plane moving further up: never contacts.
        let hit = shapecast_shape(
            &mover,
            &iso(Vec3::new(0.0, 5.0, 0.0)),
            &target,
            &iso(Vec3::ZERO),
            Vec3::new(0.0, 10.0, 0.0),
            &ShapeRegistry::new(),
        );
        assert!(
            hit.is_none(),
            "a mover receding from the plane cannot hit it"
        );
    }

    #[test]
    fn zero_motion_reports_no_sweep() {
        let mover = ColliderShape::Sphere { radius: 1.0 };
        let target = ColliderShape::Sphere { radius: 1.0 };
        let hit = shapecast_shape(
            &mover,
            &iso(Vec3::ZERO),
            &target,
            &iso(Vec3::new(1.0, 0.0, 0.0)),
            Vec3::ZERO,
            &ShapeRegistry::new(),
        );
        assert!(hit.is_none(), "a zero-length cast is not a sweep");
    }

    #[test]
    fn rotated_box_target_is_respected() {
        // A box target rotated 45 deg about Z presents a corner toward the
        // mover, so a diagonal face is nearer than an axis-aligned one would be.
        let mover = ColliderShape::Sphere { radius: 0.5 };
        let target = ColliderShape::Cuboid {
            half_extents: Vec3::ONE,
        };
        let rot = Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);
        let hit = shapecast_shape(
            &mover,
            &iso(Vec3::ZERO),
            &target,
            &Isometry::new(Vec3::new(5.0, 0.0, 0.0), rot),
            Vec3::new(10.0, 0.0, 0.0),
            &ShapeRegistry::new(),
        )
        .expect("sphere swept +X must hit the rotated box");

        // The rotated unit box reaches sqrt(2) toward -X (corner), so the
        // sphere (radius 0.5) touches after 5 - sqrt(2) - 0.5 travelled.
        let expected = 5.0 - std::f32::consts::SQRT_2 - 0.5;
        assert!(
            (hit.time_of_impact - expected).abs() < 5.0e-2,
            "distance was {} (expected ~{expected})",
            hit.time_of_impact
        );
    }
}
