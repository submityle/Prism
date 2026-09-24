//! Gameplay-level integration tests proving the M2 deliverable.
//!
//! Each test drives a *real* XPBD simulation of a mechanism a game would build
//! from the M2 feature set — an articulated door, a vehicle (spinning wheel and
//! sprung suspension), and a load-bearing linkage — and then verifies the
//! settled poses, plus the spatial-query and trigger-event surfaces. Nothing is
//! asserted from a stubbed value: every number checked below is the outcome of
//! stepping `XpbdSolver` over the world.
//!
//! # Provenance
//!
//! These are original scenarios authored for Prism. They contain **no Unreal
//! Engine source or derived code**.

use glam::{Quat, Vec3};
use prism_physics_core::{
    AngleLimit, BodyDesc, ColliderShape, CpuBackend, DistanceJoint, JointAnchor, JointDesc,
    LinearLimit, Motor, PhysicsBackend, PhysicsEvent, PhysicsWorld, PrismaticJoint, QueryFilter,
    RevoluteJoint, XpbdSolver, PI,
};
use prism_physics_geometry::Ray;

/// Builds a CPU backend running the real XPBD solver with `substeps` sub-steps.
fn xpbd_backend(substeps: u32) -> CpuBackend {
    CpuBackend::new(Box::new(XpbdSolver::new()), substeps)
}

/// A motorised hinge door: a velocity motor swings the leaf open and an angle
/// limit at a quarter turn stops it. The leaf must rotate purely about the
/// vertical hinge (staying level), swing a full quarter turn, and be held there
/// by the limit rather than continuing to the motor's implied travel.
#[test]
fn door_hinge_motor_opens_to_limit_and_holds() {
    let mut world = PhysicsWorld::with_gravity(Vec3::new(0.0, -9.81, 0.0));

    // Static jamb at the origin; the leaf's inner edge is pinned to it.
    let jamb = world.spawn(BodyDesc::static_at(Vec3::ZERO));
    let leaf = world.spawn(BodyDesc::dynamic_at(Vec3::new(0.5, 0.0, 0.0)));

    // Hinge about +Y (vertical), so gravity produces no torque about the axis.
    let hinge = RevoluteJoint::new(Vec3::Y)
        .with_limit(AngleLimit::new(0.0, PI / 2.0))
        .with_motor(Motor::velocity(2.0, 50.0));
    world.spawn_joint(JointDesc::revolute(
        JointAnchor::at_point(jamb, Vec3::ZERO),
        JointAnchor::new(leaf, Vec3::new(-0.5, 0.0, 0.0), Quat::IDENTITY),
        hinge,
    ));

    let mut backend = xpbd_backend(8);
    let dt = 1.0 / 120.0;
    for _ in 0..360 {
        backend.step(&mut world, dt);
    }

    let fwd_open = world.bodies.orientation(leaf).unwrap() * Vec3::X;
    // Rotation stayed in the horizontal plane (hinge lock held against gravity).
    assert!(
        fwd_open.y.abs() < 0.05,
        "leaf tilted out of plane: {fwd_open:?}"
    );
    // The leaf swung roughly a quarter turn (its forward is now along Z).
    assert!(
        fwd_open.dot(Vec3::X).abs() < 0.15,
        "leaf did not reach the quarter-turn limit: dot X = {}",
        fwd_open.dot(Vec3::X)
    );
    // Crucially it was *stopped* at the limit and did not run on toward a half
    // turn (which would put the forward back along -X, i.e. dot Z near 0).
    assert!(
        fwd_open.dot(Vec3::Z).abs() > 0.9,
        "limit did not stop the motor at a quarter turn: dot Z = {}",
        fwd_open.dot(Vec3::Z)
    );

    // Let it run further: the limit must keep the leaf parked, not oscillating.
    for _ in 0..180 {
        backend.step(&mut world, dt);
    }
    let fwd_held = world.bodies.orientation(leaf).unwrap() * Vec3::X;
    assert!(
        (fwd_held - fwd_open).length() < 0.05,
        "leaf drifted after reaching the limit: {fwd_open:?} -> {fwd_held:?}"
    );
}

/// A driven vehicle wheel: a velocity motor on a horizontal axle spins the
/// wheel while a coincident-point constraint keeps it mounted on the chassis.
#[test]
fn car_wheel_spins_on_axle_under_motor() {
    let mut world = PhysicsWorld::with_gravity(Vec3::new(0.0, -9.81, 0.0));

    let chassis = world.spawn(BodyDesc::static_at(Vec3::ZERO));
    let wheel = world.spawn(BodyDesc::dynamic_at(Vec3::new(1.0, 0.0, 0.0)));

    // Axle about +X; motor targets a steady spin rate.
    let axle = RevoluteJoint::new(Vec3::X).with_motor(Motor::velocity(6.0, 80.0));
    world.spawn_joint(JointDesc::revolute(
        JointAnchor::at_point(chassis, Vec3::new(1.0, 0.0, 0.0)),
        JointAnchor::at_center(wheel),
        axle,
    ));

    let mut backend = xpbd_backend(8);
    let dt = 1.0 / 120.0;
    for _ in 0..120 {
        backend.step(&mut world, dt);
    }

    let spin = world.bodies.angular_velocity(wheel).unwrap();
    // The wheel is spinning about its axle...
    assert!(
        spin.x.abs() > 1.0,
        "wheel is not spinning about the axle: {spin:?}"
    );
    // ...and essentially only about the axle (off-axis rate is small).
    assert!(
        spin.x.abs() > 4.0 * (spin.y.abs() + spin.z.abs()) + 1e-3,
        "wheel spin leaked off the axle: {spin:?}"
    );
    // The hub stayed mounted at its mount point despite gravity.
    let pos = world.bodies.position(wheel).unwrap();
    assert!(
        (pos - Vec3::new(1.0, 0.0, 0.0)).length() < 0.05,
        "wheel hub drifted off its mount: {pos:?}"
    );
}

/// A sprung suspension strut: a vertical prismatic joint lets the hub travel
/// only along the strut, and a linear limit catches it at full droop under
/// gravity instead of letting it fall away.
#[test]
fn car_suspension_travel_is_bounded_by_linear_limit() {
    let mut world = PhysicsWorld::with_gravity(Vec3::new(0.0, -9.81, 0.0));

    let chassis = world.spawn(BodyDesc::static_at(Vec3::ZERO));
    let hub = world.spawn(BodyDesc::dynamic_at(Vec3::new(0.0, -1.0, 0.0)));

    // Slide along +Y; the hub sits below the chassis so its slide coordinate
    // (hub - chassis) . Y starts at -1 and gravity drives it more negative.
    let strut = PrismaticJoint::new(Vec3::Y).with_limit(LinearLimit::new(-1.5, -0.5));
    world.spawn_joint(JointDesc::prismatic(
        JointAnchor::at_center(chassis),
        JointAnchor::at_center(hub),
        strut,
    ));

    let mut backend = xpbd_backend(8);
    let dt = 1.0 / 120.0;
    for _ in 0..240 {
        backend.step(&mut world, dt);
    }

    let pos = world.bodies.position(hub).unwrap();
    // Full droop: the limit caught the hub at the lower travel stop.
    assert!(
        (pos.y - (-1.5)).abs() < 0.02,
        "hub did not settle at the droop limit: y = {}",
        pos.y
    );
    // The strut kept the hub on-axis (no lateral wander).
    assert!(
        pos.x.abs() < 1e-3 && pos.z.abs() < 1e-3,
        "hub wandered off the strut axis: {pos:?}"
    );
}

/// A load-bearing linkage: a chain of rigid distance joints hangs from a static
/// pivot. After settling under gravity every link spacing must hold its rest
/// length and the chain must hang below the pivot.
#[test]
fn mechanism_distance_chain_holds_link_lengths() {
    let mut world = PhysicsWorld::with_gravity(Vec3::new(0.0, -9.81, 0.0));

    let pivot = world.spawn(BodyDesc::static_at(Vec3::ZERO));
    let l1 = world.spawn(BodyDesc::dynamic_at(Vec3::new(0.0, -1.0, 0.0)));
    let l2 = world.spawn(BodyDesc::dynamic_at(Vec3::new(0.0, -2.0, 0.0)));
    let l3 = world.spawn(BodyDesc::dynamic_at(Vec3::new(0.0, -3.0, 0.0)));

    for (a, b) in [(pivot, l1), (l1, l2), (l2, l3)] {
        world.spawn_joint(JointDesc::distance(
            a,
            Vec3::ZERO,
            b,
            Vec3::ZERO,
            DistanceJoint::rigid(1.0),
        ));
    }

    let mut backend = xpbd_backend(12);
    let dt = 1.0 / 120.0;
    for _ in 0..600 {
        backend.step(&mut world, dt);
    }

    let p0 = world.bodies.position(pivot).unwrap();
    let p1 = world.bodies.position(l1).unwrap();
    let p2 = world.bodies.position(l2).unwrap();
    let p3 = world.bodies.position(l3).unwrap();

    for (from, to) in [(p0, p1), (p1, p2), (p2, p3)] {
        let span = (to - from).length();
        assert!(
            (span - 1.0).abs() < 1.0e-3,
            "link spacing drifted from its rest length: {span}"
        );
    }
    // The chain hangs downward beneath the pivot.
    assert!(
        p1.y < p0.y && p2.y < p1.y && p3.y < p2.y,
        "chain did not hang down"
    );
    assert!(p3.y < -2.9, "chain did not extend under gravity: {p3:?}");
}

/// A rigid weld: a fixed joint bolts a dynamic bracket to a static frame at an
/// offset. Under gravity the bracket must neither fall nor rotate away.
#[test]
fn mechanism_fixed_weld_carries_load() {
    let mut world = PhysicsWorld::with_gravity(Vec3::new(0.0, -9.81, 0.0));

    let frame = world.spawn(BodyDesc::static_at(Vec3::new(0.0, 5.0, 0.0)));
    let bracket = world.spawn(BodyDesc::dynamic_at(Vec3::new(0.5, 5.0, 0.0)));

    world.spawn_joint(JointDesc::fixed(
        JointAnchor::at_point(frame, Vec3::new(0.5, 0.0, 0.0)),
        JointAnchor::at_center(bracket),
    ));

    let mut backend = xpbd_backend(8);
    let dt = 1.0 / 120.0;
    for _ in 0..300 {
        backend.step(&mut world, dt);
    }

    let pos = world.bodies.position(bracket).unwrap();
    assert!(
        (pos - Vec3::new(0.5, 5.0, 0.0)).length() < 0.02,
        "welded bracket sagged under load: {pos:?}"
    );
    let up = world.bodies.orientation(bracket).unwrap() * Vec3::Y;
    assert!(
        up.dot(Vec3::Y) > 0.999,
        "welded bracket rotated under load: {up:?}"
    );
}

/// A live raycast against a settled world returns the nearest body, its
/// entry point, and an outward normal.
#[test]
fn raycast_probes_the_live_world() {
    let mut world = PhysicsWorld::default();
    let collider = world.shapes.insert(ColliderShape::Sphere { radius: 1.0 });
    let target = world.spawn(BodyDesc::static_at(Vec3::new(0.0, 0.0, 5.0)).with_collider(collider));

    let ray = Ray::new(Vec3::ZERO, Vec3::Z);
    let hit = world
        .raycast(&ray, &QueryFilter::ALL)
        .expect("ray should hit the sphere");

    assert_eq!(hit.body, target);
    assert!(
        (hit.time_of_impact - 4.0).abs() < 1.0e-4,
        "unexpected time of impact: {}",
        hit.time_of_impact
    );
    assert!((hit.point - Vec3::new(0.0, 0.0, 4.0)).length() < 1.0e-4);
    assert!(
        hit.normal.dot(Vec3::NEG_Z) > 0.99,
        "unexpected surface normal: {:?}",
        hit.normal
    );
}

/// A trigger volume: a dynamic probe coasts through a static sensor. Draining
/// events per frame must yield a trigger-enter as it arrives and a
/// trigger-exit as it leaves, with no solid-collision response deflecting it.
#[test]
fn trigger_volume_reports_enter_then_exit() {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);

    let sensor_shape = world.shapes.insert(ColliderShape::Sphere { radius: 0.5 });
    let probe_shape = world.shapes.insert(ColliderShape::Sphere { radius: 0.5 });

    let sensor = world.spawn(
        BodyDesc::static_at(Vec3::ZERO)
            .with_collider(sensor_shape)
            .with_sensor(true),
    );
    let probe = world.spawn(
        BodyDesc::dynamic_at(Vec3::new(-3.0, 0.0, 0.0))
            .with_collider(probe_shape)
            .with_linear_velocity(Vec3::new(2.0, 0.0, 0.0)),
    );

    let mut backend = xpbd_backend(4);
    let dt = 1.0 / 60.0;

    let mut entered = false;
    let mut exited = false;
    let mut entered_before_exited = false;

    for _ in 0..300 {
        backend.step(&mut world, dt);
        for event in world.drain_contact_events() {
            match event {
                PhysicsEvent::TriggerEntered { sensor: s, other } => {
                    assert_eq!(s, sensor, "wrong sensor in enter event");
                    assert_eq!(other, probe, "wrong other body in enter event");
                    entered = true;
                }
                PhysicsEvent::TriggerExited { sensor: s, other } => {
                    assert_eq!(s, sensor, "wrong sensor in exit event");
                    assert_eq!(other, probe, "wrong other body in exit event");
                    exited = true;
                    entered_before_exited = entered;
                }
                // The sensor must never generate a solid collision response.
                PhysicsEvent::CollisionStarted(_) | PhysicsEvent::CollisionEnded(_) => {
                    panic!("sensor produced a solid collision event: {event:?}");
                }
            }
        }
    }

    assert!(entered, "probe never entered the trigger volume");
    assert!(exited, "probe never exited the trigger volume");
    assert!(entered_before_exited, "exit reported before enter");

    // The probe coasted straight through: no deflection from the sensor.
    let pos = world.bodies.position(probe).unwrap();
    assert!(
        pos.x > 0.5,
        "probe did not pass through the sensor: {pos:?}"
    );
    assert!(
        pos.y.abs() < 1e-4 && pos.z.abs() < 1e-4,
        "probe was deflected: {pos:?}"
    );
}
