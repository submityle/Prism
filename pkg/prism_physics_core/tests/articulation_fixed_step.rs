//! The public XPBD entry point must advance each articulation exactly once per
//! physical substep, independently of optional soft-body storage.
use glam::Vec3;
use prism_physics_core::reduced::{
    Articulation, ArticulationBase, ArticulationBuilder, ArticulationJoint, LinkContactImpulse,
    SpatialInertia, SpatialTransform,
};
use prism_physics_core::soft::SoftBody;
use prism_physics_core::solver::{Solver, XpbdSolver};
use prism_physics_core::{PhysicsWorld, WorldConfig};

fn articulation() -> Articulation {
    let mut builder = ArticulationBuilder::new(ArticulationBase::Fixed, SpatialInertia::default());
    builder
        .add_link(
            None,
            SpatialTransform::IDENTITY,
            SpatialInertia::diagonal(1.0, Vec3::ZERO, Vec3::splat(0.2)),
            ArticulationJoint::revolute(Vec3::Z),
        )
        .unwrap();
    let mut articulation = builder.build().unwrap();
    assert!(articulation.set_velocity(0, 0.7));
    articulation
}

#[test]
fn xpbd_matches_one_aba_step_per_substep_with_or_without_soft_bodies() {
    for has_soft_body in [false, true] {
        for substeps in [1, 2, 7] {
            let mut world = PhysicsWorld::new(WorldConfig::with_gravity(Vec3::ZERO));
            let mut expected = articulation();
            let handle = world.spawn_articulation(expected.clone());
            if has_soft_body {
                let mut soft = SoftBody::with_default_config();
                soft.spawn(Vec3::new(10.0, 10.0, 10.0), 1.0);
                world.spawn_soft_body(soft);
            }
            let impulse = LinkContactImpulse {
                link: 0,
                point: Vec3::X,
                impulse: Vec3::Y * 0.1,
            };
            assert!(world.queue_articulation_contact_impulse(handle, impulse));
            let _ = expected.apply_contact_impulses(&[impulse]);

            let mut solver = XpbdSolver::new();
            let dt = 1.0 / 60.0;
            for frame in 0..4 {
                for _ in 0..substeps {
                    expected.step(dt / substeps as f32);
                }
                solver.step(&mut world, dt, substeps);
                let actual = world.articulations.get(handle).unwrap();
                assert_eq!(
                    actual.positions(),
                    expected.positions(),
                    "pose mismatch: soft={has_soft_body}, substeps={substeps}, frame={frame}"
                );
                assert_eq!(actual.velocities(), expected.velocities());
                assert_eq!(actual.base_pose(), expected.base_pose());
                assert_eq!(actual.base_velocity(), expected.base_velocity());
            }
        }
    }
}
