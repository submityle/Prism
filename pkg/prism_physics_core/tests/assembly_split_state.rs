//! Fixed-boundary assembly splits must preserve current rigid motion and
//! coordinate history, not resurrect velocities or erase winding at merge time.

use glam::{Quat, Vec3};
use prism_physics_core::joint::{DistanceJoint, RevoluteJoint};
use prism_physics_core::solver::xpbd::{gear, joint_constraint};
use prism_physics_core::{
    BodyDesc, GearJoint, JointAnchor, JointDesc, JointKind, MassProperties, PhysicsWorld,
};

fn pair() -> (
    PhysicsWorld,
    prism_physics_core::BodyHandle,
    prism_physics_core::BodyHandle,
) {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    let body = |position| {
        BodyDesc::dynamic_at(position).with_mass_properties(MassProperties {
            inv_mass: 1.0,
            inv_inertia: Vec3::ONE,
        })
    };
    let a = world.spawn(body(-Vec3::X));
    let b = world.spawn(body(Vec3::X));
    (world, a, b)
}

#[test]
fn split_inherits_current_proxy_twist_in_world_space() {
    let (mut world, a, b) = pair();
    world
        .connect_assembly(
            a,
            b,
            None,
            JointKind::Distance(DistanceJoint::rigid(2.0)),
            None,
        )
        .unwrap();
    let proxy = world.merge_assembly(&[a, b]).unwrap();
    let initial_orientation = world.bodies.orientation(proxy).unwrap();
    let turn = Quat::from_rotation_y(0.7) * Quat::from_rotation_z(1.1);
    let center = Vec3::new(3.0, -2.0, 5.0);
    let velocity = Vec3::new(2.0, -1.0, 0.5);
    let omega = Vec3::new(1.4, -0.7, 2.3);
    assert!(world.bodies.set_position(proxy, center));
    assert!(world
        .bodies
        .set_orientation(proxy, turn * initial_orientation));
    assert!(world.bodies.set_linear_velocity(proxy, velocity));
    assert!(world.bodies.set_angular_velocity(proxy, omega));

    let children = world.split_assembly(proxy).unwrap();
    for (child, original_offset) in children.iter().zip([-Vec3::X, Vec3::X]) {
        let offset = turn * original_offset;
        assert!((world.bodies.position(*child).unwrap() - center - offset).length() < 2.0e-5);
        assert!(
            (world.bodies.linear_velocity(*child).unwrap() - velocity - omega.cross(offset))
                .length()
                < 2.0e-5,
            "split must use omega cross world-space radius"
        );
        assert_eq!(world.bodies.angular_velocity(*child), Some(omega));
    }
    let momentum: Vec3 = children
        .iter()
        .map(|h| world.bodies.linear_velocity(*h).unwrap())
        .sum();
    assert!((momentum - velocity * 2.0).length() < 2.0e-5);
}

#[test]
fn split_inherits_proxy_sleep_state_not_premerge_member_state() {
    for proxy_sleeping in [false, true] {
        let (mut world, a, b) = pair();
        {
            let view = world.bodies.solver_view_mut();
            for body in [a, b] {
                view.sleeping[body.index() as usize] = !proxy_sleeping;
                view.sleep_timers[body.index() as usize] = 9.0;
            }
        }
        world
            .connect_assembly(
                a,
                b,
                None,
                JointKind::Distance(DistanceJoint::rigid(2.0)),
                None,
            )
            .unwrap();
        let proxy = world.merge_assembly(&[a, b]).unwrap();
        {
            let view = world.bodies.solver_view_mut();
            view.sleeping[proxy.index() as usize] = proxy_sleeping;
            view.sleep_timers[proxy.index() as usize] = if proxy_sleeping { 2.0 } else { 0.0 };
        }
        for child in world.split_assembly(proxy).unwrap() {
            assert_eq!(world.bodies.is_sleeping(child), Some(proxy_sleeping));
            assert_eq!(
                world.bodies.sleep_timer(child),
                Some(if proxy_sleeping { 2.0 } else { 0.0 })
            );
        }
    }
}

#[test]
fn generational_split_remap_preserves_hinge_winding_after_observation() {
    let (mut world, a, b) = pair();
    let kind = JointKind::Revolute(RevoluteJoint::new(Vec3::Z));
    let joint = world.spawn_joint(JointDesc::new(
        JointAnchor::at_center(a),
        JointAnchor::at_center(b),
        kind,
    ));
    assert!(world
        .joints
        .get_mut(joint)
        .unwrap()
        .set_revolute_coordinate(20.0, [Quat::IDENTITY; 2],));
    let proxy = world.merge_assembly(&[a, b]).unwrap();
    let children = world.split_assembly(proxy).unwrap();
    assert!(!children.contains(&a) && !children.contains(&b));
    let view = world.bodies.solver_view_mut();
    joint_constraint::begin_revolute_substep(&view, &mut world.joints, 1.0 / 60.0);
    joint_constraint::finish_revolute_substep(&view, &mut world.joints);
    assert!(
        (world
            .joints
            .get(joint)
            .unwrap()
            .revolute_state()
            .angle()
            .unwrap()
            - 20.0)
            .abs()
            < 1.0e-6
    );
}

#[test]
fn generational_split_remap_preserves_each_gear_shaft_winding() {
    let (mut world, a, b) = pair();
    let model = GearJoint::new(Vec3::Z, Vec3::Z, -2.0);
    let joint = world.spawn_joint(JointDesc::gear(
        JointAnchor::at_center(a),
        JointAnchor::at_center(b),
        model,
    ));
    world.joints.get_mut(joint).unwrap().enabled = false;
    for step in 0..=100 {
        let theta = step as f32 * 0.2;
        world
            .bodies
            .set_orientation(a, Quat::from_rotation_z(theta));
        world
            .bodies
            .set_orientation(b, Quat::from_rotation_z(-2.0 * theta));
        let view = world.bodies.solver_view_mut();
        gear::begin_substep(&view, &mut world.joints);
        gear::finish_substep(&view, &mut world.joints);
    }
    let before = world
        .joints
        .get(joint)
        .unwrap()
        .gear_state()
        .angles()
        .unwrap();
    assert!((before[0] - 20.0).abs() < 1.0e-5 && (before[1] + 40.0).abs() < 1.0e-5);
    let proxy = world.merge_assembly(&[a, b]).unwrap();
    world.split_assembly(proxy).unwrap();
    let view = world.bodies.solver_view_mut();
    gear::begin_substep(&view, &mut world.joints);
    gear::finish_substep(&view, &mut world.joints);
    let after = world
        .joints
        .get(joint)
        .unwrap()
        .gear_state()
        .angles()
        .unwrap();
    for i in 0..2 {
        assert!(
            (after[i] - before[i]).abs() < 1.0e-5,
            "shaft {i} winding changed"
        );
    }
}
