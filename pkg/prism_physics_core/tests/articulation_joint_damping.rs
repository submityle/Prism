//! Physical joint-space damping regressions for reduced articulations.

use glam::Vec3;
use prism_physics_core::math::transform::Isometry;
use prism_physics_core::reduced::articulation::{
    Articulation, ArticulationBase, ArticulationBuildError, ArticulationBuilder, ArticulationJoint,
    JointMotor, LinkContactImpulse, SpatialInertia, SpatialMotion, SpatialTransform,
};

fn close(actual: f32, expected: f32) {
    let tolerance = 4.0e-5 * (1.0 + expected.abs());
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual {actual}, expected {expected}, tolerance {tolerance}"
    );
}

fn scalar_tree(revolute: bool, inertia: f32, damping: f32) -> Articulation {
    let joint = if revolute {
        ArticulationJoint::revolute(Vec3::Z)
    } else {
        ArticulationJoint::prismatic(Vec3::X)
    }
    .with_damping(damping);
    let body_inertia = SpatialInertia::diagonal(inertia, Vec3::ZERO, Vec3::splat(inertia));
    let mut builder = ArticulationBuilder::new(ArticulationBase::Fixed, body_inertia);
    builder
        .add_link(None, SpatialTransform::IDENTITY, body_inertia, joint)
        .unwrap();
    builder.build().unwrap()
}

#[test]
fn scalar_viscous_damping_depends_on_physical_mass_and_is_stable_when_stiff() {
    for revolute in [false, true] {
        for mass in [2.0f32, 7.0] {
            for damping in [1.25f32, 100_000.0] {
                let dt = 0.04;
                let initial = -3.0;
                let mut tree = scalar_tree(revolute, mass, damping);
                assert!(tree.set_velocity(0, initial));
                let report = tree.step(dt);
                let expected = mass * initial / (mass + dt * damping);
                close(tree.velocities()[0], expected);
                close(report.joint_accelerations[0], (expected - initial) / dt);
                assert!(tree.velocities()[0].is_finite());
                assert!(tree.velocities()[0].abs() <= initial.abs());
            }
        }
    }
}

#[test]
fn motor_effort_and_passive_damping_share_the_implicit_mass_solve() {
    let mass = 7.0;
    let damping = 3.0;
    let dt = 0.05;
    let initial = 1.25;
    let motor = JointMotor::new(
        Vec3::splat(0.4),
        Vec3::splat(-0.5),
        11.0,
        2.0,
        Vec3::splat(100.0),
    );
    let body_inertia = SpatialInertia::diagonal(mass, Vec3::ZERO, Vec3::splat(mass));
    let mut builder = ArticulationBuilder::new(ArticulationBase::Fixed, body_inertia);
    builder
        .add_link(
            None,
            SpatialTransform::IDENTITY,
            body_inertia,
            ArticulationJoint::prismatic(Vec3::X)
                .with_damping(damping)
                .with_motor(motor),
        )
        .unwrap();
    let mut tree = builder.build().unwrap();
    let position = 0.1;
    assert!(tree.set_position(0, position));
    assert!(tree.set_velocity(0, initial));
    let effort = motor.stiffness * (motor.target_position.x - position)
        + motor.damping * (motor.target_velocity.x - initial);
    let expected = (mass * initial + dt * effort) / (mass + dt * damping);
    tree.step(dt);
    close(tree.velocities()[0], expected);
}

fn coupled_tree(revolute: bool, damping: [f32; 2], mass: [f32; 2]) -> Articulation {
    let inertia = |value| SpatialInertia::diagonal(value, Vec3::ZERO, Vec3::splat(value));
    let joint = |coefficient| {
        if revolute {
            ArticulationJoint::revolute(Vec3::Z)
        } else {
            ArticulationJoint::prismatic(Vec3::X)
        }
        .with_damping(coefficient)
    };
    let mut builder = ArticulationBuilder::new(ArticulationBase::Fixed, inertia(1.0));
    builder
        .add_link(
            None,
            SpatialTransform::IDENTITY,
            inertia(mass[0]),
            joint(damping[0]),
        )
        .unwrap();
    builder
        .add_link(
            Some(0),
            SpatialTransform::IDENTITY,
            inertia(mass[1]),
            joint(damping[1]),
        )
        .unwrap();
    builder.build().unwrap()
}

#[test]
fn coupled_chain_matches_dense_backward_euler_reference() {
    let masses = [2.0f32, 5.0];
    let damping = [3.0f32, 7.0];
    let initial = [1.25f32, -0.75];
    let dt = 0.03;
    for revolute in [false, true] {
        let mut tree = coupled_tree(revolute, damping, masses);
        assert!(tree.set_velocity(0, initial[0]));
        assert!(tree.set_velocity(1, initial[1]));

        let m00 = masses[0] + masses[1];
        let m01 = masses[1];
        let m11 = masses[1];
        let a00 = m00 + dt * damping[0];
        let a01 = m01;
        let a11 = m11 + dt * damping[1];
        let rhs0 = m00 * initial[0] + m01 * initial[1];
        let rhs1 = m01 * initial[0] + m11 * initial[1];
        let determinant = a00 * a11 - a01 * a01;
        let expected0 = (a11 * rhs0 - a01 * rhs1) / determinant;
        let expected1 = (-a01 * rhs0 + a00 * rhs1) / determinant;

        tree.step(dt);
        close(tree.velocities()[0], expected0);
        close(tree.velocities()[1], expected1);
    }
}

#[test]
fn floating_base_damping_conserves_total_linear_momentum() {
    let base_mass = 3.0;
    let link_mass = 5.0;
    let inertia = |mass| SpatialInertia::diagonal(mass, Vec3::ZERO, Vec3::splat(mass));
    let mut builder = ArticulationBuilder::new(ArticulationBase::Floating, inertia(base_mass));
    builder
        .add_link(
            None,
            SpatialTransform::IDENTITY,
            inertia(link_mass),
            ArticulationJoint::prismatic(Vec3::X).with_damping(20.0),
        )
        .unwrap();
    let mut tree = builder.build().unwrap();
    assert!(tree.set_base_pose(Isometry::IDENTITY));
    assert!(tree.set_base_velocity(SpatialMotion::new(Vec3::ZERO, Vec3::X * -0.25)));
    assert!(tree.set_velocity(0, 2.0));
    let before_link = tree.link_velocities()[0].linear.x;
    let before = base_mass * tree.base_velocity().linear.x + link_mass * before_link;
    tree.step(0.1);
    let after_link = tree.link_velocities()[0].linear.x;
    let after = base_mass * tree.base_velocity().linear.x + link_mass * after_link;
    close(after, before);
    assert!(tree.velocities()[0].abs() < 2.0);
}

#[test]
fn contact_inverse_mass_excludes_damping_but_includes_armature() {
    let impulse = LinkContactImpulse {
        link: 0,
        point: Vec3::ZERO,
        impulse: Vec3::X,
    };
    let undamped = scalar_tree(false, 2.0, 0.0);
    let damped = scalar_tree(false, 2.0, 1000.0);
    close(
        undamped
            .generalized_contact_impulse_response(&[impulse])
            .generalized_velocity_change[0],
        0.5,
    );
    close(
        damped
            .generalized_contact_impulse_response(&[impulse])
            .generalized_velocity_change[0],
        0.5,
    );

    let body_inertia = SpatialInertia::diagonal(2.0, Vec3::ZERO, Vec3::splat(2.0));
    let mut builder = ArticulationBuilder::new(ArticulationBase::Fixed, body_inertia);
    builder
        .add_link(
            None,
            SpatialTransform::IDENTITY,
            body_inertia,
            ArticulationJoint::prismatic(Vec3::X)
                .with_damping(1000.0)
                .with_armature(3.0),
        )
        .unwrap();
    let tree = builder.build().unwrap();
    close(
        tree.generalized_contact_impulse_response(&[impulse])
            .generalized_velocity_change[0],
        0.2,
    );
}

#[test]
fn fixed_joint_rejects_a_nonzero_damping_coefficient() {
    let inertia = SpatialInertia::diagonal(1.0, Vec3::ZERO, Vec3::ONE);
    let mut builder = ArticulationBuilder::new(ArticulationBase::Fixed, inertia);
    assert!(matches!(
        builder.add_link(
            None,
            SpatialTransform::IDENTITY,
            inertia,
            ArticulationJoint::fixed().with_damping(1.0),
        ),
        Err(ArticulationBuildError::InvalidJointParameters { link: 0 })
    ));
}
