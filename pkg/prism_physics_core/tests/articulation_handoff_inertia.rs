//! Mobility regressions for COM/principal-frame handoff.

use glam::{Mat3, Quat, Vec3};
use prism_physics_core::math::transform::Isometry;
use prism_physics_core::reduced::{
    ArticulationBase, ArticulationBuilder, ArticulationJoint, SpatialInertia, SpatialTransform,
};
use prism_physics_core::world::PhysicsWorld;

/// Observe mobility, not reconstructed inertia: reciprocating a spurious tiny
/// moment can look accurate as a tensor while producing an enormous spin.
#[test]
fn handoff_angular_impulse_response_is_covariant_across_diagonal_fast_path() {
    for smallest in [0.0_f32, 1.0e-12, 1.0e-8, 0.25] {
        for angle in [0.0_f32, -1.0e-6, 1.0e-6] {
            let principal = Quat::from_rotation_z(angle);
            let axes = Mat3::from_quat(principal);
            // Principal moments must satisfy the physical triangle inequality.
            // In particular a line mass has moments (0, I, I), not (0, I, 2I).
            let moments = Vec3::new(smallest, 1.0, 1.0);
            let tensor = axes * Mat3::from_diagonal(moments) * axes.transpose();
            let inertia = SpatialInertia::new(1.0, Vec3::ZERO, tensor.transpose().to_cols_array_2d());
            let authored = Quat::from_rotation_y(0.35) * Quat::from_rotation_x(-0.2);
            let mut builder =
                ArticulationBuilder::new(ArticulationBase::Fixed, SpatialInertia::default())
                    .with_gravity(Vec3::ZERO)
                    .with_base_pose(Isometry::new(Vec3::new(2.0, 3.0, -1.0), authored));
            builder
                .add_link(None, SpatialTransform::IDENTITY, inertia, ArticulationJoint::fixed())
                .unwrap();
            let mut tree = builder.build().unwrap();
            tree.break_joint(0).unwrap();
            let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
            let handle = world.spawn_articulation(tree);
            let report = world.handoff_articulation(handle).unwrap();
            let body = report.links[0].body;
            let world_axes = authored * principal;
            // Scale the impulse on a retained small mode to keep the response
            // bounded without weakening the zero-mobility assertion.
            let retained = smallest > 2.0e-10;
            // Observe the small mode independently. Mixing a large transverse
            // impulse with a 1e-8 moment measures f32 rotation cancellation,
            // rather than whether handoff kept the intended mobility.
            let impulse = Vec3::new(if retained { smallest } else { 1.0 }, 0.0, 0.0);
            let expected = world_axes * Vec3::new(if retained { 1.0 } else { 0.0 }, 0.0, 0.0);
            assert!(world.apply_torque_impulse(body, world_axes * impulse));
            let actual = world.bodies.angular_velocity(body).unwrap();
            assert!(
                actual.is_finite() && (actual - expected).length() < 2.0e-5,
                "moment={smallest}, angle={angle}, actual={actual:?}, expected={expected:?}"
            );
            let mass = world.bodies.mass_properties(body).unwrap();
            let pose = world.bodies.orientation(body).unwrap();
            let inverse = Mat3::from_quat(pose)
                * Mat3::from_diagonal(mass.inv_inertia)
                * Mat3::from_quat(pose).transpose();
            let response_axis = world_axes * Vec3::X;
            let mobility = response_axis.dot(inverse * response_axis);
            if retained {
                assert!((mobility * smallest - 1.0).abs() < 2.0e-4);
            } else {
                assert!(mobility.abs() < 2.0e-6, "null mobility {mobility}");
            }
        }
    }
}
