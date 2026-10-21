//! Independent analytical impulse and kinematic regressions using public APIs.
//!
//! Expected values come from scalar mechanics, a two-link mass matrix, or
//! compound rigid-body mechanics, never from the articulation's dynamics step.
use glam::{Mat3, Quat, Vec3};
use prism_physics_core::math::transform::Isometry;
use prism_physics_core::reduced::articulation::{
    Articulation, ArticulationBase, ArticulationBuildError, ArticulationBuilder, ArticulationJoint,
    JointMotor, LinkContactImpulse, SpatialForce, SpatialInertia, SpatialMotion, SpatialTransform,
};

fn close(actual: f32, expected: f32) {
    let tolerance = 2.0e-5 * (1.0 + expected.abs());
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual {actual}, expected {expected}, tolerance {tolerance}"
    );
}

fn close_vec(actual: Vec3, expected: Vec3) {
    let tolerance = 5.0e-5 * (1.0 + expected.length());
    assert!(
        (actual - expected).length() <= tolerance,
        "actual {actual:?}, expected {expected:?}, tolerance {tolerance}"
    );
}

fn close_motion(actual: SpatialMotion, expected: SpatialMotion) {
    close_vec(actual.angular, expected.angular);
    close_vec(actual.linear, expected.linear);
}

fn single_link(joint: ArticulationJoint, gravity: Vec3) -> Articulation {
    let mut builder = ArticulationBuilder::new(
        ArticulationBase::Fixed,
        SpatialInertia::diagonal(1.0, Vec3::ZERO, Vec3::ONE),
    )
    .with_gravity(gravity);
    builder
        .add_link(
            None,
            SpatialTransform::IDENTITY,
            SpatialInertia::diagonal(2.0, Vec3::ZERO, Vec3::splat(0.2)),
            joint,
        )
        .unwrap();
    builder.build().unwrap()
}

fn contact(link: usize, point: Vec3, impulse: Vec3) -> LinkContactImpulse {
    LinkContactImpulse {
        link,
        point,
        impulse,
    }
}

fn check_scalar_impulse(mut tree: Articulation, impulse: LinkContactImpulse, expected: f32) {
    assert!(tree.set_velocity(0, -0.7));
    let positions = tree.positions().to_vec();
    let base_pose = tree.base_pose();
    let base_velocity = tree.base_velocity();
    let response = tree.generalized_contact_impulse_response(&[impulse]);
    assert_eq!(response.generalized_velocity_change.len(), 1);
    close(response.generalized_velocity_change[0], expected);
    close_motion(response.base_velocity_change, SpatialMotion::default());
    assert_eq!(tree.velocities(), &[-0.7]);
    let applied = tree.apply_contact_impulses(&[impulse]);
    close(applied.generalized_velocity_change[0], expected);
    close(tree.velocities()[0], -0.7 + expected);
    assert_eq!(tree.positions(), positions);
    assert_eq!(tree.base_pose(), base_pose);
    assert_eq!(tree.base_velocity(), base_velocity);
}

#[test]
fn prismatic_mass_two_has_half_unit_impulse_response() {
    // m * delta_qd = J; rotational inertia cannot affect axial translation.
    check_scalar_impulse(
        single_link(ArticulationJoint::prismatic(Vec3::X), Vec3::ZERO),
        contact(0, Vec3::ZERO, Vec3::X),
        0.5,
    );
}

#[test]
fn revolute_inertia_point_two_has_five_unit_point_impulse_response() {
    // r = X, J = Y: angular impulse about Z is one; delta_qd = 1 / 0.2.
    check_scalar_impulse(
        single_link(ArticulationJoint::revolute(Vec3::Z), Vec3::ZERO),
        contact(0, Vec3::X, Vec3::Y),
        5.0,
    );
}

#[test]
fn impulse_response_excludes_motor_damping_forces_gravity_and_velocity_bias() {
    for revolute in [false, true] {
        // Exercise each source alone, then all together. Nonzero q/qd also
        // ensure a servo calculation would not vanish.
        for (motor, forces, gravity) in [
            (false, false, false),
            (true, false, false),
            (false, true, false),
            (false, false, true),
            (true, true, true),
        ] {
            let mut joint = if revolute {
                ArticulationJoint::revolute(Vec3::Z)
            } else {
                ArticulationJoint::prismatic(Vec3::X)
            };
            if motor {
                joint = joint
                    .with_motor(JointMotor::new(
                        Vec3::splat(2.0),
                        Vec3::splat(-3.0),
                        17.0,
                        4.0,
                        Vec3::splat(100.0),
                    ))
                    .with_damping(7.0);
            }
            let mut tree = single_link(
                joint,
                if gravity {
                    Vec3::new(4.0, -9.81, 2.0)
                } else {
                    Vec3::ZERO
                },
            );
            assert!(tree.set_position(0, 0.37));
            assert!(tree.set_velocity(0, -1.3));
            if forces {
                assert!(tree.set_link_force(
                    0,
                    SpatialForce::new(Vec3::new(3.0, -2.0, 11.0), Vec3::new(7.0, 5.0, -4.0)),
                ));
                assert!(tree.set_base_force(SpatialForce::new(Vec3::ONE, Vec3::splat(9.0))));
            }
            let (point, impulse, expected) = if revolute {
                (Vec3::X, Vec3::Y, 5.0)
            } else {
                (Vec3::ZERO, Vec3::X, 0.5)
            };
            let positions = tree.positions().to_vec();
            // Both an empty batch and a valid zero impulse must be exactly zero.
            for impulses in [vec![], vec![contact(0, point, Vec3::ZERO)]] {
                let response = tree.generalized_contact_impulse_response(&impulses);
                assert_eq!(response.generalized_velocity_change, vec![0.0]);
                assert_eq!(response.base_velocity_change, SpatialMotion::default());
                let applied = tree.apply_contact_impulses(&impulses);
                assert_eq!(applied, response);
                assert_eq!(tree.velocities(), &[-1.3]);
            }
            for _ in 0..2 {
                // Repeated queries must not consume retained forces or step.
                let response =
                    tree.generalized_contact_impulse_response(&[contact(0, point, impulse)]);
                close(response.generalized_velocity_change[0], expected);
                close_motion(response.base_velocity_change, SpatialMotion::default());
                assert_eq!(tree.velocities(), &[-1.3]);
            }
            let applied = tree.apply_contact_impulses(&[contact(0, point, impulse)]);
            close(applied.generalized_velocity_change[0], expected);
            close(tree.velocities()[0], -1.3 + expected);
            assert_eq!(tree.positions(), positions);
        }
    }
}

#[test]
fn coaxial_prismatic_chain_matches_independent_two_by_two_mass_matrix() {
    let mut builder = ArticulationBuilder::new(
        ArticulationBase::Fixed,
        SpatialInertia::diagonal(1.0, Vec3::ZERO, Vec3::ONE),
    )
    .with_gravity(Vec3::ZERO);
    for (parent, mass) in [(None, 2.0), (Some(0), 3.0)] {
        builder
            .add_link(
                parent,
                SpatialTransform::IDENTITY,
                SpatialInertia::diagonal(mass, Vec3::ZERO, Vec3::ONE),
                ArticulationJoint::prismatic(Vec3::X),
            )
            .unwrap();
    }
    let tree = builder.build().unwrap();
    // T = (2*v0^2 + 3*(v0+v1)^2)/2, hence M = [[5,3],[3,3]].
    // M^-1 = [[1/2,-1/2],[-1/2,5/6]]. A root impulse has
    // generalized impulse [1,0]; an end impulse has [1,1].
    for (link, expected) in [(0, [0.5, -0.5]), (1, [0.0, 1.0 / 3.0])] {
        let impulse = contact(link, Vec3::ZERO, Vec3::X);
        let response = tree.generalized_contact_impulse_response(&[impulse]);
        assert_eq!(response.generalized_velocity_change.len(), 2);
        let mut applied = tree.clone();
        let _ = applied.apply_contact_impulses(&[impulse]);
        for (dof, expected) in expected.into_iter().enumerate() {
            close(response.generalized_velocity_change[dof], expected);
            close(applied.velocities()[dof], expected);
        }
        close_vec(
            applied.contact_point_velocity(0, Vec3::ZERO).unwrap(),
            Vec3::X * expected[0],
        );
        close_vec(
            applied.contact_point_velocity(1, Vec3::ZERO).unwrap(),
            Vec3::X * (expected[0] + expected[1]),
        );
    }
}

#[derive(Clone, Copy)]
struct ReferenceBody {
    pose: Isometry,
    mass: f32,
    com: Vec3,
    principal: Vec3,
}

impl ReferenceBody {
    fn inertia(self) -> SpatialInertia {
        SpatialInertia::diagonal(self.mass, self.com, self.principal)
    }

    fn world_com(self) -> Vec3 {
        self.pose.transform_point(self.com)
    }

    fn world_inertia(self) -> Mat3 {
        let rotation = Mat3::from_quat(self.pose.rotation);
        rotation * Mat3::from_diagonal(self.principal) * rotation.transpose()
    }
}

fn floating_fixed_tree(base_pose: Isometry) -> (Articulation, Vec<ReferenceBody>) {
    let base = ReferenceBody {
        pose: base_pose,
        mass: 2.0,
        com: Vec3::new(0.2, -0.1, 0.3),
        principal: Vec3::new(0.5, 0.7, 0.8),
    };
    let mut builder = ArticulationBuilder::new(ArticulationBase::Floating, base.inertia())
        .with_base_pose(base_pose)
        .with_gravity(Vec3::new(3.0, -9.81, -2.0));
    let mut bodies = vec![base];
    // Reference indices include the base: this is a chain plus a branch.
    for (parent, rotation, translation, mass, com) in [
        (
            0,
            Quat::from_rotation_z(0.6),
            Vec3::new(0.8, -0.3, 0.4),
            3.0,
            Vec3::new(-0.2, 0.3, 0.1),
        ),
        (
            1,
            Quat::from_rotation_y(-0.7),
            Vec3::new(-0.4, 0.9, 0.2),
            1.5,
            Vec3::new(0.1, -0.2, 0.3),
        ),
        (
            1,
            Quat::from_rotation_x(0.4),
            Vec3::new(0.2, -0.5, 0.7),
            0.8,
            Vec3::new(0.3, 0.1, -0.2),
        ),
    ] {
        let parent_pose = bodies[parent].pose;
        // x_tree rotates parent coordinates into child coordinates. Build the
        // reference poses directly, without the articulation's FK output.
        let body = ReferenceBody {
            pose: Isometry::new(
                parent_pose.translation + parent_pose.rotation * translation,
                (parent_pose.rotation * rotation.inverse()).normalize(),
            ),
            mass,
            com,
            principal: Vec3::new(0.3, 0.4, 0.5) * mass,
        };
        builder
            .add_link(
                if parent == 0 { None } else { Some(parent - 1) },
                SpatialTransform::new(rotation, translation),
                body.inertia(),
                ArticulationJoint::fixed(),
            )
            .unwrap();
        bodies.push(body);
    }
    (builder.build().unwrap(), bodies)
}

fn compound_properties(bodies: &[ReferenceBody]) -> (f32, Vec3, Mat3) {
    let mass: f32 = bodies.iter().map(|body| body.mass).sum();
    let center = bodies
        .iter()
        .map(|body| body.world_com() * body.mass)
        .sum::<Vec3>()
        / mass;
    let inertia = bodies.iter().fold(Mat3::ZERO, |sum, body| {
        let r = body.world_com() - center;
        let outer = Mat3::from_cols(r * r.x, r * r.y, r * r.z);
        sum + body.world_inertia() + (Mat3::IDENTITY * r.length_squared() - outer) * body.mass
    });
    assert!(inertia.determinant() > 0.0);
    (mass, center, inertia)
}

fn rigid_response(bodies: &[ReferenceBody], point: Vec3, impulse: Vec3) -> SpatialMotion {
    let (mass, center, inertia) = compound_properties(bodies);
    let angular = inertia.inverse() * (point - center).cross(impulse);
    let linear = impulse / mass + angular.cross(bodies[0].pose.translation - center);
    let world_to_base = bodies[0].pose.rotation.inverse();
    SpatialMotion::new(world_to_base * angular, world_to_base * linear)
}

fn rigid_point_response(bodies: &[ReferenceBody], point: Vec3, impulse: Vec3, probe: Vec3) -> Vec3 {
    let (mass, center, inertia) = compound_properties(bodies);
    impulse / mass + (inertia.inverse() * (point - center).cross(impulse)).cross(probe - center)
}

fn world_contact(
    bodies: &[ReferenceBody],
    link: usize,
    point: Vec3,
    impulse: Vec3,
) -> LinkContactImpulse {
    let inverse = bodies[link + 1].pose.inverse();
    contact(
        link,
        inverse.transform_point(point),
        inverse.transform_vector(impulse),
    )
}

fn momentum(tree: &Articulation, bodies: &[ReferenceBody]) -> (Vec3, Vec3) {
    let link_motions = tree.link_velocities();
    assert_eq!(link_motions.len() + 1, bodies.len());
    let motions = std::iter::once(tree.base_velocity()).chain(link_motions);
    let mut linear = Vec3::ZERO;
    let mut angular = Vec3::ZERO;
    for (body, motion) in bodies.iter().zip(motions) {
        let omega = body.pose.rotation * motion.angular;
        let velocity =
            body.pose.rotation * motion.linear + omega.cross(body.pose.rotation * body.com);
        let p = velocity * body.mass;
        linear += p;
        angular += body.world_inertia() * omega + body.world_com().cross(p);
    }
    (linear, angular)
}

fn base_poses() -> [Isometry; 2] {
    [
        Isometry::IDENTITY,
        Isometry::new(
            Vec3::new(2.3, -1.7, 0.9),
            Quat::from_rotation_y(0.8) * Quat::from_rotation_x(-0.5),
        ),
    ]
}

#[test]
fn floating_fixed_tree_matches_compound_inertia_and_conserves_total_momentum() {
    for base_pose in base_poses() {
        let (tree, bodies) = floating_fixed_tree(base_pose);
        assert_eq!(tree.dof_count(), 0);
        for (pose, body) in tree.forward_kinematics().iter().zip(&bodies[1..]) {
            close_vec(pose.translation, body.pose.translation);
            close(pose.rotation.dot(body.pose.rotation).abs(), 1.0);
        }
        for link in 0..tree.link_count() {
            let point = bodies[link + 1]
                .pose
                .transform_point(Vec3::new(0.4, -0.2, 0.5));
            let impulse = base_pose.rotation * Vec3::new(0.7, -1.1, 0.4);
            let contact = world_contact(&bodies, link, point, impulse);
            let expected = rigid_response(&bodies, point, impulse);
            let response = tree.generalized_contact_impulse_response(&[contact]);
            assert!(response.generalized_velocity_change.is_empty());
            close_motion(response.base_velocity_change, expected);
            for moving in [false, true] {
                let mut applied = tree.clone();
                if moving {
                    assert!(applied.set_base_velocity(SpatialMotion::new(
                        Vec3::new(0.3, -0.2, 0.4),
                        Vec3::new(-0.5, 0.6, 0.2),
                    )));
                }
                let before_velocity = applied.base_velocity();
                let before = momentum(&applied, &bodies);
                let response = applied.apply_contact_impulses(&[contact]);
                close_motion(response.base_velocity_change, expected);
                close_motion(
                    applied.base_velocity(),
                    SpatialMotion::new(
                        before_velocity.angular + expected.angular,
                        before_velocity.linear + expected.linear,
                    ),
                );
                let after = momentum(&applied, &bodies);
                close_vec(after.0 - before.0, impulse);
                close_vec(after.1 - before.1, point.cross(impulse));
                assert_eq!(applied.base_pose(), base_pose);
            }
        }
        let zero = tree.generalized_contact_impulse_response(&[contact(1, Vec3::X, Vec3::ZERO)]);
        assert!(zero.generalized_velocity_change.is_empty());
        assert_eq!(zero.base_velocity_change, SpatialMotion::default());
    }
}

#[test]
fn floating_fixed_tree_point_response_is_reciprocal_and_analytical() {
    for base_pose in base_poses() {
        let (tree, bodies) = floating_fixed_tree(base_pose);
        let a = bodies[2].pose.transform_point(Vec3::new(0.4, -0.3, 0.2));
        let b = bodies[3].pose.transform_point(Vec3::new(-0.2, 0.5, 0.3));
        let ja = base_pose.rotation * Vec3::new(0.3, 0.8, -0.5);
        let jb = base_pose.rotation * Vec3::new(-0.7, 0.2, 0.6);
        let mut due_to_a = tree.clone();
        let mut due_to_b = tree.clone();
        let _ = due_to_a.apply_contact_impulses(&[world_contact(&bodies, 1, a, ja)]);
        let _ = due_to_b.apply_contact_impulses(&[world_contact(&bodies, 2, b, jb)]);
        let va_due_to_b = due_to_b.contact_point_velocity(1, a).unwrap();
        let vb_due_to_a = due_to_a.contact_point_velocity(2, b).unwrap();
        close_vec(va_due_to_b, rigid_point_response(&bodies, b, jb, a));
        close_vec(vb_due_to_a, rigid_point_response(&bodies, a, ja, b));
        close(ja.dot(va_due_to_b), jb.dot(vb_due_to_a));
        let direction = ja.normalize();
        close(
            tree.contact_inverse_mass(1, a, direction).unwrap(),
            direction.dot(rigid_point_response(&bodies, a, direction, a)),
        );
    }
}

#[test]
fn mixed_chain_point_velocity_matches_centered_finite_difference_fk() {
    for base in [ArticulationBase::Fixed, ArticulationBase::Floating] {
        let base_pose = base_poses()[1];
        let mut builder =
            ArticulationBuilder::new(base, SpatialInertia::diagonal(2.0, Vec3::ZERO, Vec3::ONE))
                .with_base_pose(base_pose);
        for (parent, rotation, translation, joint) in [
            (
                None,
                Quat::from_rotation_x(0.4),
                Vec3::new(0.5, -0.2, 0.3),
                ArticulationJoint::revolute(Vec3::Z),
            ),
            (
                Some(0),
                Quat::from_rotation_y(-0.6),
                Vec3::new(-0.3, 0.4, 0.2),
                ArticulationJoint::prismatic(Vec3::X),
            ),
            (
                Some(1),
                Quat::from_rotation_z(0.3),
                Vec3::new(0.2, 0.3, -0.4),
                ArticulationJoint::revolute(Vec3::Y),
            ),
        ] {
            builder
                .add_link(
                    parent,
                    SpatialTransform::new(rotation, translation),
                    SpatialInertia::diagonal(1.0, Vec3::ZERO, Vec3::ONE),
                    joint,
                )
                .unwrap();
        }
        let mut tree = builder.build().unwrap();
        for (dof, (q, qd)) in [(0.4, 0.7), (-0.2, -0.5), (0.3, 0.6)]
            .into_iter()
            .enumerate()
        {
            assert!(tree.set_position(dof, q));
            assert!(tree.set_velocity(dof, qd));
        }
        let base_velocity =
            SpatialMotion::new(Vec3::new(0.2, -0.3, 0.4), Vec3::new(-0.5, 0.4, 0.3));
        if base == ArticulationBase::Floating {
            assert!(tree.set_base_velocity(base_velocity));
        }
        let h = 1.0e-3;
        let mut minus = tree.clone();
        let mut plus = tree.clone();
        for (offset, shifted) in [(-h, &mut minus), (h, &mut plus)] {
            for dof in 0..tree.dof_count() {
                assert!(shifted
                    .set_position(dof, tree.positions()[dof] + offset * tree.velocities()[dof]));
            }
            if base == ArticulationBase::Floating {
                assert!(shifted.set_base_pose(Isometry::new(
                    base_pose.translation + offset * (base_pose.rotation * base_velocity.linear),
                    (base_pose.rotation * Quat::from_scaled_axis(offset * base_velocity.angular))
                        .normalize(),
                )));
            }
        }
        let poses = tree.forward_kinematics();
        let minus_poses = minus.forward_kinematics();
        let plus_poses = plus.forward_kinematics();
        for link in 0..tree.link_count() {
            let local_point = Vec3::new(0.3, -0.4, 0.2);
            let difference = (plus_poses[link].transform_point(local_point)
                - minus_poses[link].transform_point(local_point))
                / (2.0 * h);
            let velocity = tree
                .contact_point_velocity(link, poses[link].transform_point(local_point))
                .unwrap();
            // f32 world-position cancellation dominates the centered difference.
            assert!(
                (velocity - difference).length() < 5.0e-4,
                "base {base:?}, link {link}: velocity {velocity:?}, finite difference {difference:?}"
            );
        }
    }
}

#[test]
fn floating_point_mass_preserves_supported_translation_despite_singular_rotation() {
    for link_mass in [0.0, 3.0] {
        let base_pose = base_poses()[1];
        let mut builder =
            ArticulationBuilder::new(ArticulationBase::Floating, SpatialInertia::point_mass(2.0))
                .with_base_pose(base_pose);
        builder
            .add_link(
                None,
                SpatialTransform::new(Quat::from_rotation_z(0.4), Vec3::ZERO),
                SpatialInertia::point_mass(link_mass),
                ArticulationJoint::fixed(),
            )
            .unwrap();
        let mut tree = builder.build().unwrap();
        // The co-located fixed link exposes the contact API without adding
        // rotational inertia. A massless proxy must still move the mass-2 base.
        let impulse = Vec3::new(1.0, -2.0, 3.0);
        let total_mass = 2.0 + link_mass;
        let point = base_pose.translation;
        let contact = tree.contact_impulse_from_world(0, point, impulse).unwrap();
        let expected = SpatialMotion::new(
            Vec3::ZERO,
            base_pose.rotation.inverse() * (impulse / total_mass),
        );
        let response = tree.generalized_contact_impulse_response(&[contact]);
        assert!(response.generalized_velocity_change.is_empty());
        close_motion(response.base_velocity_change, expected);
        let _ = tree.apply_contact_impulses(&[contact]);
        close_motion(tree.base_velocity(), expected);
        let velocity = tree.contact_point_velocity(0, point).unwrap();
        close_vec(velocity, impulse / total_mass);
        close_vec(velocity * total_mass, impulse);
        close(
            tree.contact_inverse_mass(0, point, impulse.normalize())
                .unwrap(),
            1.0 / total_mass,
        );
        assert_eq!(tree.base_pose(), base_pose);
    }
}

#[test]
fn massless_prismatic_has_zero_inverse_mass_not_identity_response() {
    let mut builder =
        ArticulationBuilder::new(ArticulationBase::Fixed, SpatialInertia::point_mass(0.0));
    builder
        .add_link(
            None,
            SpatialTransform::IDENTITY,
            SpatialInertia::point_mass(0.0),
            ArticulationJoint::prismatic(Vec3::X),
        )
        .unwrap();
    let mut tree = builder.build().unwrap();
    let impulse = contact(0, Vec3::ZERO, Vec3::X);
    let response = tree.generalized_contact_impulse_response(&[impulse]);
    assert_eq!(response.generalized_velocity_change, vec![0.0]);
    assert_eq!(response.base_velocity_change, SpatialMotion::default());
    assert_eq!(tree.apply_contact_impulses(&[impulse]), response);
    assert_eq!(tree.velocities(), &[0.0]);
    assert_eq!(tree.contact_inverse_mass(0, Vec3::ZERO, Vec3::X), Some(0.0));
    assert!(tree.set_link_force(0, SpatialForce::new(Vec3::ZERO, Vec3::X)));
    let report = tree.step(0.1);
    assert!(!report.degraded);
    assert_eq!(report.joint_accelerations, vec![0.0]);
    assert_eq!(tree.positions(), &[0.0]);
    assert_eq!(tree.velocities(), &[0.0]);
}

#[test]
fn nonphysical_or_nonfinite_inertia_is_rejected_without_clamping() {
    let negative_mass = SpatialInertia::diagonal(-2.0, Vec3::ZERO, Vec3::ONE);
    assert_eq!(negative_mass.mass, -2.0);
    for invalid in [
        negative_mass,
        SpatialInertia::diagonal(1.0, Vec3::ZERO, Vec3::new(-0.2, 0.3, 0.4)),
        SpatialInertia::new(
            1.0,
            Vec3::ZERO,
            [[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ),
        // Positive definite, but Izz > Ixx + Iyy cannot describe a mass distribution.
        SpatialInertia::diagonal(1.0, Vec3::ZERO, Vec3::new(1.0, 1.0, 3.0)),
        SpatialInertia::point_mass(f32::NAN),
        SpatialInertia::diagonal(1.0, Vec3::new(f32::NAN, 0.0, 0.0), Vec3::ONE),
        SpatialInertia::diagonal(1.0, Vec3::ZERO, Vec3::new(1.0, f32::NAN, 1.0)),
    ] {
        assert!(
            matches!(
                ArticulationBuilder::new(ArticulationBase::Floating, invalid).build(),
                Err(ArticulationBuildError::InvalidBaseInertia)
            ),
            "invalid base inertia accepted: {invalid:?}"
        );
        let mut builder =
            ArticulationBuilder::new(ArticulationBase::Fixed, SpatialInertia::point_mass(0.0));
        assert_eq!(
            builder.add_link(
                None,
                SpatialTransform::IDENTITY,
                invalid,
                ArticulationJoint::fixed()
            ),
            Err(ArticulationBuildError::InvalidInertia { link: 0 }),
        );
        // Rejection must not append a partial link to the builder.
        assert_eq!(
            builder.add_link(
                None,
                SpatialTransform::IDENTITY,
                SpatialInertia::point_mass(1.0),
                ArticulationJoint::fixed()
            ),
            Ok(0),
        );
        assert_eq!(builder.build().unwrap().link_count(), 1);
    }
}

fn invalid_poses() -> [Isometry; 4] {
    [
        Isometry::new(Vec3::new(f32::NAN, 0.0, 0.0), Quat::IDENTITY),
        Isometry::new(Vec3::ZERO, Quat::from_xyzw(f32::NAN, 0.0, 0.0, 1.0)),
        Isometry::new(Vec3::ZERO, Quat::from_xyzw(0.0, 0.0, 0.0, 2.0)),
        Isometry::new(Vec3::ZERO, Quat::from_xyzw(0.0, 0.0, 0.0, 0.0)),
    ]
}

#[test]
fn nonfinite_or_nonunit_initial_transforms_are_rejected() {
    for pose in invalid_poses() {
        let mut builder =
            ArticulationBuilder::new(ArticulationBase::Floating, SpatialInertia::point_mass(2.0));
        assert!(matches!(
            builder.clone().with_base_pose(pose).build(),
            Err(ArticulationBuildError::InvalidTransform { link: None }),
        ));
        assert_eq!(
            builder.add_link(
                None,
                SpatialTransform::new(pose.rotation, pose.translation),
                SpatialInertia::point_mass(1.0),
                ArticulationJoint::fixed(),
            ),
            Err(ArticulationBuildError::InvalidTransform { link: Some(0) }),
        );
        assert_eq!(builder.build().unwrap().link_count(), 0);
    }
    assert!(matches!(
        ArticulationBuilder::new(ArticulationBase::Floating, SpatialInertia::point_mass(2.0))
            .with_gravity(Vec3::new(0.0, f32::NAN, 0.0))
            .build(),
        Err(ArticulationBuildError::InvalidTransform { link: None }),
    ));
}

fn assert_same_public_state(actual: &Articulation, expected: &Articulation) {
    assert_eq!(actual.positions(), expected.positions());
    assert_eq!(actual.velocities(), expected.velocities());
    assert_eq!(actual.base_pose(), expected.base_pose());
    assert_eq!(actual.base_velocity(), expected.base_velocity());
    assert_eq!(actual.gravity(), expected.gravity());
    assert_eq!(actual.forward_kinematics(), expected.forward_kinematics());
    assert_eq!(actual.link_velocities(), expected.link_velocities());
}

#[test]
fn invalid_state_setters_reject_inputs_without_mutating_state_or_retained_forces() {
    let mut builder = ArticulationBuilder::new(
        ArticulationBase::Floating,
        SpatialInertia::diagonal(2.0, Vec3::ZERO, Vec3::ONE),
    )
    .with_base_pose(base_poses()[1]);
    builder
        .add_link(
            None,
            SpatialTransform::IDENTITY,
            SpatialInertia::diagonal(1.0, Vec3::ZERO, Vec3::splat(0.3)),
            ArticulationJoint::prismatic(Vec3::X),
        )
        .unwrap();
    let mut tree = builder.build().unwrap();
    assert!(tree.set_position(0, 0.4));
    assert!(tree.set_velocity(0, -0.7));
    assert!(tree.set_base_velocity(SpatialMotion::new(
        Vec3::new(0.1, -0.2, 0.3),
        Vec3::new(-0.4, 0.2, 0.5)
    )));
    assert!(tree.set_link_force(
        0,
        SpatialForce::new(Vec3::new(0.2, 0.3, 0.1), Vec3::new(0.5, -0.2, 0.4))
    ));
    assert!(tree.set_base_force(SpatialForce::new(
        Vec3::new(-0.1, 0.2, 0.3),
        Vec3::new(-0.3, 0.4, 0.2)
    )));
    let mut untouched = tree.clone();
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(!tree.set_position(0, value));
        assert_same_public_state(&tree, &untouched);
        assert!(!tree.set_velocity(0, value));
        assert_same_public_state(&tree, &untouched);
        for motion in [
            SpatialMotion::new(Vec3::new(value, 0.0, 0.0), Vec3::ZERO),
            SpatialMotion::new(Vec3::ZERO, Vec3::new(0.0, value, 0.0)),
        ] {
            assert!(!tree.set_base_velocity(motion));
            assert_same_public_state(&tree, &untouched);
        }
        for force in [
            SpatialForce::new(Vec3::new(value, 0.0, 0.0), Vec3::ZERO),
            SpatialForce::new(Vec3::ZERO, Vec3::new(0.0, value, 0.0)),
        ] {
            assert!(!tree.set_link_force(0, force));
            assert!(!tree.set_base_force(force));
            assert_same_public_state(&tree, &untouched);
        }
    }
    for pose in invalid_poses() {
        assert!(!tree.set_base_pose(pose));
        assert_same_public_state(&tree, &untouched);
    }
    // External wrenches have no getter: identical subsequent evolution verifies
    // that rejected writes preserved the previously retained finite forces.
    let actual_report = tree.step(0.01);
    let expected_report = untouched.step(0.01);
    assert!(!actual_report.degraded);
    assert_eq!(
        actual_report.joint_accelerations,
        expected_report.joint_accelerations
    );
    assert_eq!(
        actual_report.base_acceleration,
        expected_report.base_acceleration
    );
    assert_same_public_state(&tree, &untouched);
}

#[test]
fn spherical_noncoaxial_body_angular_velocity_integrates_by_right_exponential() {
    // Isotropic inertia at the joint origin gives zero torque-free angular
    // acceleration, so the exact one-step oracle uses the initial omega.
    let mut tree = single_link(ArticulationJoint::spherical(), Vec3::ZERO);
    let q = Vec3::new(0.6, -0.4, 0.2);
    let omega = Vec3::new(0.3, 0.8, -0.5);
    let dt = 0.2;
    for dof in 0..3 {
        assert!(tree.set_position(dof, q[dof]));
        assert!(tree.set_velocity(dof, omega[dof]));
    }
    let expected = (Quat::from_scaled_axis(q) * Quat::from_scaled_axis(omega * dt)).normalize();
    let report = tree.step(dt);
    assert!(!report.degraded);
    assert_eq!(report.joint_accelerations.len(), 3);
    for dof in 0..3 {
        close(report.joint_accelerations[dof], 0.0);
        close(tree.velocities()[dof], omega[dof]);
    }
    let stored = Quat::from_scaled_axis(Vec3::from_slice(tree.positions()));
    let pose = tree.forward_kinematics()[0];
    for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
        close_vec(stored * axis, expected * axis);
        close_vec(pose.rotation * axis, expected * axis);
    }
    close_vec(pose.translation, Vec3::ZERO);
}
