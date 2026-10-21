//! CCD over immutable collision leaves, never a support map of their union.
//! Both bodies follow their root's interpolated translation and shortest spin.

use glam::{Quat, Vec3};
use prism_physics_geometry::Transformed;
use prism_physics_geometry::{gjk_closest_points, Aabb, SupportMap};

use super::{shape_sweep::bounding_radius, support::CcdSupport, triangle_support::TriangleSupport};
use crate::collider::{ColliderShape, ShapeRegistry};
use crate::math::transform::Isometry;
use crate::state::handle::BodyHandle;
use crate::world::PhysicsWorld;

const BAND: f32 = 1.0e-3;
const ITERATIONS: usize = 64;

#[derive(Clone, Copy)]
struct Motion {
    start: Isometry,
    displacement: Vec3,
    rotation: Quat,
    radius: f32,
}

impl Motion {
    fn new(start: Isometry, end: Isometry, radius: f32) -> Self {
        Self {
            start,
            displacement: end.translation - start.translation,
            rotation: (end.rotation * start.rotation.inverse()).normalize(),
            radius,
        }
    }

    fn spin(self, t: f32) -> Quat {
        // At exactly 180 degrees the two quaternion representatives differ
        // only by their sign, but `Quat::slerp` may choose the opposite arc
        // when the scalar part is a tiny negative rounding residual. Preserve
        // the authored axis for that tie so a compound child follows the
        // intended root-pivot path instead of silently reversing it.
        let rotation = if self.rotation.w.abs() <= 1.0e-6 {
            Quat::from_xyzw(self.rotation.x, self.rotation.y, self.rotation.z, 0.0).normalize()
        } else {
            self.rotation
        };
        Quat::IDENTITY.slerp(rotation, t)
    }

    fn posed<'a, S: SupportMap>(self, shape: &'a S, t: f32) -> Transformed<'a, S> {
        let rotation = self.spin(t) * self.start.rotation;
        Transformed::new(
            shape,
            rotation,
            self.start.translation + self.displacement * t,
        )
    }

    fn angular_speed(self) -> f32 {
        self.rotation.to_axis_angle().1.abs() * self.radius
    }

    fn bounds(self) -> Aabb {
        let end = self.start.translation + self.displacement;
        Aabb::new(
            self.start.translation.min(end) - Vec3::splat(self.radius + BAND),
            self.start.translation.max(end) + Vec3::splat(self.radius + BAND),
        )
    }
}

enum LeafSupport<'a> {
    Convex(CcdSupport<'a>, Option<bool>),
    Triangle(TriangleSupport, Option<bool>),
}

impl LeafSupport<'_> {
    fn self_collision(&self) -> Option<bool> {
        match self {
            Self::Convex(_, flag) | Self::Triangle(_, flag) => *flag,
        }
    }
}

impl SupportMap for LeafSupport<'_> {
    fn support_point(&self, dir: Vec3) -> Vec3 {
        match self {
            Self::Convex(s, _) => s.support_point(dir),
            Self::Triangle(s, _) => s.support_point(dir),
        }
    }
}

/// Registry validation has already flattened compounds and bounded their leaf
/// count. Meshes remain collections of real triangles, on either side.
fn supports<'a>(
    shape: &ColliderShape,
    pose: Isometry,
    shapes: &'a ShapeRegistry,
    out: &mut Vec<LeafSupport<'a>>,
) -> Option<()> {
    supports_with_policy(shape, pose, shapes, out, None)
}

fn supports_with_policy<'a>(
    shape: &ColliderShape,
    pose: Isometry,
    shapes: &'a ShapeRegistry,
    out: &mut Vec<LeafSupport<'a>>,
    self_collision: Option<bool>,
) -> Option<()> {
    match *shape {
        ColliderShape::Compound { compound, .. } => {
            let data = shapes.compound(compound)?;
            for (child, policy) in data.children().iter().zip(data.policies()) {
                supports_with_policy(
                    shapes.get(child.collider)?,
                    pose.mul(&child.local_pose),
                    shapes,
                    out,
                    self_collision.or(policy.self_collision),
                )?;
            }
        }
        ColliderShape::TriangleMesh { mesh, .. } => {
            let mesh = shapes.tri_mesh(mesh)?.mesh();
            for i in 0..mesh.triangle_count() {
                // Match triangle-mesh queries: malformed index triples are
                // absent geometry, not a reason to discard the valid mesh.
                let Some([a, b, c]) = mesh.triangle(i) else {
                    continue;
                };
                out.push(LeafSupport::Triangle(
                    TriangleSupport::new(
                        pose.transform_point(a),
                        pose.transform_point(b),
                        pose.transform_point(c),
                    ),
                    self_collision,
                ));
            }
        }
        _ => out.push(LeafSupport::Convex(
            CcdSupport::from_shape_in(shape, pose.translation, pose.rotation, shapes)?,
            self_collision,
        )),
    }
    Some(())
}

fn pair_toi(a: &LeafSupport<'_>, ma: Motion, b: &LeafSupport<'_>, mb: Motion) -> Option<f32> {
    let angular = ma.angular_speed() + mb.angular_speed();
    let linear = ma.displacement - mb.displacement;
    let mut t = 0.0;
    for _ in 0..ITERATIONS {
        let Some(points) = gjk_closest_points(&ma.posed(a, t), &mb.posed(b, t)) else {
            return Some(t);
        };
        if points.distance <= BAND {
            return Some(t);
        }
        let closing = linear.dot(points.normal) + angular;
        if closing <= 0.0 {
            return None;
        }
        let next = t + (points.distance - BAND) / closing;
        if next > 1.0 {
            return None;
        }
        if next <= t {
            return Some(t);
        }
        t = next;
    }
    // Bounded conservative advancement may stop short on grazing spins. This
    // is a lower-bound time, not a hull/AABB narrow-phase collision result.
    Some(t)
}

fn plane_toi(
    a: &LeafSupport<'_>,
    ma: Motion,
    mb: Motion,
    normal: Vec3,
    offset: f32,
) -> Option<f32> {
    let mut t = 0.0;
    // The normal can rotate as well. Bound its rate against every possible
    // mover point's distance from the plane body's moving origin.
    let relative = ma.start.translation - mb.start.translation;
    let linear = ma.displacement - mb.displacement;
    let target_angle = mb.rotation.to_axis_angle().1.abs();
    let speed = linear.length()
        + ma.angular_speed()
        + target_angle * (relative.length() + linear.length() + ma.radius);
    for _ in 0..ITERATIONS {
        let rotation = mb.spin(t) * mb.start.rotation;
        let n = (rotation * normal).normalize_or_zero();
        if n == Vec3::ZERO {
            return None;
        }
        let origin = mb.start.translation + mb.displacement * t;
        let height = n.dot(ma.posed(a, t).support_point(-n) - origin) - offset;
        if height <= BAND {
            return Some(t);
        }
        if speed <= 0.0 {
            return None;
        }
        let next = t + (height - BAND) / speed;
        if next > 1.0 {
            return None;
        }
        if next <= t {
            return Some(t);
        }
        t = next;
    }
    Some(t)
}

pub(super) fn sweep_toi(
    world: &PhysicsWorld,
    mover: BodyHandle,
    shape: &ColliderShape,
    start: Isometry,
    end: Isometry,
) -> Option<f32> {
    let motion = Motion::new(start, end, bounding_radius(shape));
    if motion.displacement == Vec3::ZERO && motion.angular_speed() == 0.0 {
        return None;
    }
    let mut movers = Vec::new();
    supports(shape, Isometry::IDENTITY, &world.shapes, &mut movers)?;
    let bounds = motion.bounds();
    let mut best: Option<f32> = None;
    // Current-pose broad-phase alone misses moving targets that cross the
    // sweep and end outside it. Scan swept target bounds until the broad-phase
    // exposes motion bounds; no world mutation/integration change is needed.
    for slot in 0..world.bodies.slot_count() {
        let Some(target) = world.bodies.handle_at_slot(slot) else {
            continue;
        };
        if mover == target {
            continue;
        }
        let Some(shape) = world
            .bodies
            .collider(target)
            .and_then(|h| world.shapes.get(h))
        else {
            continue;
        };
        let Some(end_pos) = world.bodies.position(target) else {
            continue;
        };
        let end_rot = world.bodies.orientation(target).unwrap_or(Quat::IDENTITY);
        let start = Isometry::new(
            world.bodies.prev_position(target).unwrap_or(end_pos),
            world.bodies.prev_orientation(target).unwrap_or(end_rot),
        );
        let target_motion = Motion::new(
            start,
            Isometry::new(end_pos, end_rot),
            bounding_radius(shape),
        );
        if !matches!(shape, ColliderShape::Plane { .. })
            && !bounds.intersects(&target_motion.bounds())
        {
            continue;
        }
        if let ColliderShape::Plane { normal, offset } = *shape {
            for leaf in &movers {
                if !world.collision_exclusions().allows_leaf_pair(
                    mover,
                    target,
                    leaf.self_collision(),
                    None,
                ) {
                    continue;
                }
                if let Some(t) = plane_toi(leaf, motion, target_motion, normal, offset) {
                    best = Some(best.map_or(t, |old| old.min(t)));
                }
            }
        } else {
            let mut targets = Vec::new();
            if supports(shape, Isometry::IDENTITY, &world.shapes, &mut targets).is_none() {
                continue;
            }
            for a in &movers {
                for b in &targets {
                    if !world.collision_exclusions().allows_leaf_pair(
                        mover,
                        target,
                        a.self_collision(),
                        b.self_collision(),
                    ) {
                        continue;
                    }
                    if let Some(t) = pair_toi(a, motion, b, target_motion) {
                        best = Some(best.map_or(t, |old| old.min(t)));
                    }
                }
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::{ColliderHandle, CompoundChild, TriMeshData};
    use crate::state::body::BodyDesc;

    fn union(
        world: &mut PhysicsWorld,
        collider: ColliderHandle,
        offsets: &[Vec3],
    ) -> ColliderHandle {
        world
            .shapes
            .insert_compound(
                &offsets
                    .iter()
                    .map(|p| CompoundChild {
                        collider,
                        local_pose: Isometry::from_translation(*p),
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap()
    }

    #[test]
    fn compound_target_gap_is_not_a_ccd_wall_but_real_leaf_is_hit() {
        let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
        let sphere = world.shapes.insert(ColliderShape::Sphere { radius: 0.5 });
        let target = union(&mut world, sphere, &[Vec3::X * -2.0, Vec3::X * 2.0]);
        world.spawn(BodyDesc::static_at(Vec3::ZERO).with_collider(target));
        let shape = ColliderShape::Sphere { radius: 0.1 };
        let mover_shape = world.shapes.insert(shape);
        let mover = world.spawn(BodyDesc::dynamic_at(Vec3::Z * -4.0).with_collider(mover_shape));
        let cast = |x| {
            sweep_toi(
                &world,
                mover,
                &shape,
                Isometry::from_translation(Vec3::new(x, 0.0, -4.0)),
                Isometry::from_translation(Vec3::new(x, 0.0, 4.0)),
            )
        };
        assert_eq!(cast(0.0), None);
        assert!((cast(2.0).unwrap() - 0.425).abs() < 0.005);
        // Outside the child's sphere, while still inside its broad-phase AABB.
        assert_eq!(cast(2.5 + 0.11), None);
    }

    #[test]
    fn starting_root_rotation_is_preserved_for_target_and_mover() {
        let rotation = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        for compound_moves in [false, true] {
            let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
            let sphere = world.shapes.insert(ColliderShape::Sphere { radius: 0.2 });
            let compound = union(&mut world, sphere, &[-Vec3::X * 2.0, Vec3::X * 2.0]);
            let mut target_desc = BodyDesc::static_at(Vec3::ZERO).with_collider(compound);
            target_desc.orientation = rotation;
            let target = world.spawn(target_desc);
            let sphere_shape = *world.shapes.get(sphere).unwrap();
            let mover =
                world.spawn(BodyDesc::dynamic_at(Vec3::new(0.0, 2.0, -4.0)).with_collider(sphere));
            let cast = |position: Vec3| {
                if compound_moves {
                    // Relative motion is identical, but now the rotated
                    // compound is the mover rather than the static target.
                    world.bodies.set_position(mover, position);
                    world.bodies.set_prev_position(mover, position);
                    sweep_toi(
                        &world,
                        target,
                        world.shapes.get(compound).unwrap(),
                        Isometry::new(-Vec3::Z * 4.0, rotation),
                        Isometry::new(Vec3::Z * 4.0, rotation),
                    )
                } else {
                    sweep_toi(
                        &world,
                        mover,
                        &sphere_shape,
                        Isometry::from_translation(position - Vec3::Z * 4.0),
                        Isometry::from_translation(position + Vec3::Z * 4.0),
                    )
                }
            };
            let mut cast = cast;
            let t = cast(Vec3::Y * 2.0).expect("rotated leaf must be hit");
            assert!(
                (t - 0.45).abs() < 0.005,
                "compound_moves={compound_moves}: {t}"
            );
            assert_eq!(
                cast(Vec3::X * 2.0),
                None,
                "unrotated location must remain empty"
            );
            assert_eq!(cast(Vec3::ZERO), None, "gap must remain empty");
        }
    }

    #[test]
    fn root_spin_composes_after_arbitrary_starting_rotation() {
        let start = Isometry::new(
            Vec3::new(1.0, 2.0, 3.0),
            Quat::from_rotation_x(0.7) * Quat::from_rotation_y(-0.4),
        );
        let delta = Quat::from_rotation_z(0.8);
        let end = Isometry::new(start.translation + Vec3::Y, delta * start.rotation);
        let motion = Motion::new(start, end, 2.0);
        let sphere = CcdSupport::from_shape(
            &ColliderShape::Sphere { radius: 0.2 },
            Vec3::new(1.5, 0.3, -0.2),
            Quat::IDENTITY,
        )
        .unwrap();
        for t in [0.0, 0.5, 1.0] {
            let orientation = Quat::IDENTITY.slerp(delta, t) * start.rotation;
            let expected = orientation * sphere.support_point(orientation.inverse() * Vec3::X)
                + start.translation
                + motion.displacement * t;
            let actual = motion.posed(&sphere, t).support_point(Vec3::X);
            assert!(
                (actual - expected).length() < 1.0e-5,
                "t={t}: {actual:?} != {expected:?}"
            );
        }
    }

    #[test]
    fn compound_mover_gap_and_nested_root_spin_are_preserved() {
        let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
        let sphere = world.shapes.insert(ColliderShape::Sphere { radius: 0.2 });
        let child = union(&mut world, sphere, &[Vec3::X * 2.0]);
        let nested = world
            .shapes
            .insert_compound(&[CompoundChild {
                collider: child,
                local_pose: Isometry::from_rotation(Quat::from_rotation_z(
                    std::f32::consts::FRAC_PI_2,
                )),
            }])
            .unwrap();
        let shape = *world.shapes.get(nested).unwrap();
        let mover = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO).with_collider(nested));
        let target = world.spawn(BodyDesc::static_at(Vec3::X * -2.0).with_collider(sphere));
        let t = sweep_toi(
            &world,
            mover,
            &shape,
            Isometry::IDENTITY,
            Isometry::from_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)),
        )
        .unwrap();
        assert!(t > 0.6 && t < 1.0, "root-pivot spin TOI: {t}");
        world.bodies.set_position(target, Vec3::ZERO);
        // Also update target's start via a fresh world; its prior motion above
        // is intentionally irrelevant to the pure two-support gap check.
        let ma = Motion::new(
            Isometry::from_translation(Vec3::Z * -2.0),
            Isometry::from_translation(Vec3::Z * 2.0),
            bounding_radius(&shape),
        );
        let mut leaves = Vec::new();
        supports(&shape, ma.start, &world.shapes, &mut leaves).unwrap();
        let b = LeafSupport::Convex(
            CcdSupport::from_shape(
                &ColliderShape::Sphere { radius: 0.2 },
                Vec3::ZERO,
                Quat::IDENTITY,
            )
            .unwrap(),
            None,
        );
        let mb = Motion::new(Isometry::IDENTITY, Isometry::IDENTITY, 0.2);
        assert!(leaves.iter().all(|a| pair_toi(a, ma, &b, mb).is_none()));
    }

    #[test]
    fn rotating_compound_target_and_crossing_target_are_not_culled_at_end_pose() {
        let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
        let sphere = world.shapes.insert(ColliderShape::Sphere { radius: 0.2 });
        let target_shape = union(&mut world, sphere, &[Vec3::X * 2.0]);
        let target = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO).with_collider(target_shape));
        world
            .bodies
            .set_orientation(target, Quat::from_rotation_z(std::f32::consts::PI));
        let shape = *world.shapes.get(sphere).unwrap();
        let mover = world.spawn(BodyDesc::dynamic_at(Vec3::Y * 2.0).with_collider(sphere));
        let t = sweep_toi(
            &world,
            mover,
            &shape,
            Isometry::from_translation(Vec3::Y * 2.0),
            Isometry::from_translation(Vec3::Y * 2.0 + Vec3::Z * 0.01),
        )
        .unwrap();
        assert!(t > 0.3 && t < 0.6, "spinning target TOI: {t}");

        let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
        let s = world.shapes.insert(shape);
        let target = world.spawn(BodyDesc::dynamic_at(Vec3::X * -4.0).with_collider(s));
        world.bodies.set_position(target, Vec3::X * 4.0);
        let mover = world.spawn(BodyDesc::dynamic_at(Vec3::Z * -1.0).with_collider(s));
        assert!(sweep_toi(
            &world,
            mover,
            &shape,
            Isometry::from_translation(Vec3::Z * -1.0),
            Isometry::from_translation(Vec3::Z)
        )
        .is_some());
    }

    #[test]
    fn compound_mesh_leaf_is_swept_as_real_triangles_on_either_side() {
        let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
        let mesh = world.shapes.insert_tri_mesh(TriMeshData::new(
            vec![
                Vec3::new(0.0, -2.0, -2.0),
                Vec3::new(0.0, 2.0, -2.0),
                Vec3::new(0.0, 0.0, 2.0),
            ],
            vec![[0, 1, 2]],
        ));
        let shape = world.shapes.tri_mesh_shape(mesh).unwrap();
        let mesh = world.shapes.insert(shape);
        let compound = union(&mut world, mesh, &[Vec3::ZERO]);
        let compound_shape = *world.shapes.get(compound).unwrap();
        let sphere_shape = ColliderShape::Sphere { radius: 0.1 };
        let sphere = world.shapes.insert(sphere_shape);
        let target = world.spawn(BodyDesc::static_at(Vec3::ZERO).with_collider(compound));
        let mover = world.spawn(BodyDesc::dynamic_at(Vec3::X * -2.0).with_collider(sphere));
        let t = sweep_toi(
            &world,
            mover,
            &sphere_shape,
            Isometry::from_translation(Vec3::X * -2.0),
            Isometry::from_translation(Vec3::X * 2.0),
        )
        .unwrap();
        assert!((t - 0.475).abs() < 0.01);
        // Reverse moving/target roles without ever support-mapping the mesh union.
        let t = sweep_toi(
            &world,
            target,
            &compound_shape,
            Isometry::IDENTITY,
            Isometry::from_translation(Vec3::X * -4.0),
        )
        .unwrap();
        assert!((t - 0.475).abs() < 0.01);
    }
}
