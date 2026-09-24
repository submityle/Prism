//! End-to-end integration test: a dynamic body dropped from rest under gravity
//! must follow the analytic free-fall solution, while a static body stays put.

use glam::Vec3;
use prism_physics_core::{
    BodyDesc, CpuBackend, DriveMode, DriveOutcome, PhysicsBackend, PhysicsWorld, SimulationDriver,
};

#[test]
fn dynamic_body_free_falls_and_static_body_is_fixed() {
    let g = 9.81_f32;
    let mut world = PhysicsWorld::with_gravity(Vec3::new(0.0, -g, 0.0));

    let y0 = 100.0_f32;
    let dynamic = world.spawn(BodyDesc::dynamic_at(Vec3::new(0.0, y0, 0.0)));
    let fixed = world.spawn(BodyDesc::static_at(Vec3::new(5.0, 5.0, 5.0)));

    // Integrate for t = 1s using a small fixed time step.
    let dt = 1.0 / 240.0;
    let steps = 240;
    let mut backend = CpuBackend::with_defaults();
    for _ in 0..steps {
        backend.step(&mut world, dt);
    }

    let t = dt * steps as f32; // exactly 1.0s
    let analytic = y0 - 0.5 * g * t * t;
    let simulated = world.bodies.position(dynamic).unwrap().y;

    // Semi-implicit Euler slightly overshoots the analytic drop; with this many
    // steps the error is well under 0.1 world units.
    assert!(
        (simulated - analytic).abs() < 0.1,
        "simulated y = {simulated}, analytic y = {analytic}"
    );

    // The static body must not have moved at all.
    assert_eq!(world.bodies.position(fixed), Some(Vec3::new(5.0, 5.0, 5.0)));
}

#[test]
fn realtime_driver_advances_via_backend() {
    let mut world = PhysicsWorld::with_gravity(Vec3::new(0.0, -9.81, 0.0));
    let h = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
    let mut backend = CpuBackend::with_defaults();
    let driver = SimulationDriver::new(DriveMode::Realtime);

    let mut outcome = DriveOutcome::Stepped;
    for _ in 0..60 {
        outcome = driver.drive(&mut backend, &mut world, 1.0 / 60.0);
    }
    assert_eq!(outcome, DriveOutcome::Stepped);
    assert!(world.bodies.position(h).unwrap().y < -4.0);
}
