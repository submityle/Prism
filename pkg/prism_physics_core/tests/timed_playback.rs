//! Time-addressed full-world playback regression tests.

use std::sync::Arc;

use glam::{Quat, Vec3};
use prism_physics_core::{
    BodyDesc, CacheHandle, CpuBackend, DriveMode, DriveOutcome, DriverCache, PhysicsWorld,
    PlaybackError, PlaybackRepeat, SimulationDriver, StateSnapshot, TimedPlayback, XpbdSolver,
};

fn record() -> (DriverCache, prism_physics_core::BodyHandle) {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    let handle = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
    let driver = SimulationDriver::new(DriveMode::Offline {
        substeps: 1,
        iterations: 1,
        target_frames: 3,
    });
    let mut backend = CpuBackend::new(Box::new(XpbdSolver::new()), 1);
    for (index, dt) in [0.125, 0.25, 0.5].into_iter().enumerate() {
        world
            .bodies
            .set_position(handle, Vec3::X * index as f32 * 10.0);
        world
            .bodies
            .set_orientation(handle, Quat::from_rotation_y(index as f32 * 0.4));
        assert!(matches!(
            driver.drive(&mut backend, &mut world, dt),
            DriveOutcome::OfflineFrame { .. }
        ));
    }
    (driver.recorded_cache().unwrap(), handle)
}

#[test]
fn variable_duration_lookup_interpolates_without_inventing_initial_state() {
    let (cache, handle) = record();
    assert_eq!(cache.duration(), 0.875);
    for (index, time) in [0.125, 0.375, 0.875].into_iter().enumerate() {
        assert_eq!(cache.frame(index as u32).unwrap().time(), time);
        let sample = cache.sample_at(time).unwrap();
        assert_eq!(sample.boundary().index(), index as u32);
        assert_eq!(sample.pose(handle).unwrap().position.x, index as f32 * 10.0);
    }
    assert_eq!(
        cache.sample_at(0.0).unwrap().pose(handle).unwrap().position,
        Vec3::ZERO
    );
    assert_eq!(
        cache
            .sample_at(100.0)
            .unwrap()
            .pose(handle)
            .unwrap()
            .position
            .x,
        20.0
    );
    let sample = cache.sample_at(0.25).unwrap();
    assert_eq!(sample.alpha(), 0.5);
    let pose = sample.pose(handle).unwrap();
    assert_eq!(pose.position.x, 5.0);
    assert!(pose
        .orientation
        .abs_diff_eq(Quat::from_rotation_y(0.2), 1e-6));
    let sample = cache.sample_at(0.625).unwrap();
    assert_eq!(sample.alpha(), 0.5);
    assert_eq!(sample.pose(handle).unwrap().position.x, 15.0);
    let mut output = StateSnapshot::new();
    sample.render_into(&mut output);
    assert_eq!(output.pose(handle), sample.pose(handle));
}

#[test]
fn sample_restores_full_boundary_not_fractional_solver_pose() {
    let (cache, handle) = record();
    let sample = cache.sample_at(0.625).unwrap();
    let mut destination = PhysicsWorld::with_gravity(Vec3::ONE);
    destination.spawn(BodyDesc::dynamic_at(Vec3::splat(100.0)));
    sample.restore_boundary(&mut destination);
    assert_eq!(destination.bodies.position(handle).unwrap().x, 10.0);
    assert_eq!(sample.pose(handle).unwrap().position.x, 15.0);
    assert_eq!(
        prism_physics_core::snapshot::world_hash::hash_world(&destination).unwrap(),
        sample.boundary().snapshot().state_hash().unwrap(),
    );
    // Ordinary discrete playback still restores the identical complete frame.
    let driver = SimulationDriver::new(DriveMode::Playback(CacheHandle(1)));
    driver.register_cache(CacheHandle(1), cache).unwrap();
    let mut backend = CpuBackend::new(Box::new(XpbdSolver::new()), 1);
    driver.drive(&mut backend, &mut destination, 1.0);
    assert_eq!(destination.bodies.position(handle).unwrap().x, 0.0);
}

#[test]
fn player_supports_pause_reverse_wrap_seek_and_independent_shared_clocks() {
    let (cache, handle) = record();
    let mut player = TimedPlayback::new(Arc::new(cache)).unwrap();
    let other = player.clone();
    assert!(Arc::ptr_eq(player.cache(), other.cache()));
    assert_eq!(
        player
            .advance(0.25)
            .unwrap()
            .pose(handle)
            .unwrap()
            .position
            .x,
        5.0
    );
    assert_eq!(other.time(), 0.0);
    player.set_speed(0.0).unwrap();
    player.advance(10.0).unwrap();
    assert_eq!(player.time(), 0.25);
    player.set_speed(2.0).unwrap();
    player.advance(0.125).unwrap();
    assert_eq!(player.time(), 0.5);
    player.seek(999.0).unwrap();
    assert_eq!(player.time(), 0.875);
    player.set_repeat(PlaybackRepeat::Loop);
    assert_eq!(player.time(), 0.0);
    player.set_speed(-1.0).unwrap();
    player.advance(0.25).unwrap();
    assert_eq!(player.time(), 0.625);
    player.seek(1.125).unwrap();
    assert_eq!(player.time(), 0.25);
    player.set_repeat(PlaybackRepeat::Once);
    player.advance(100.0).unwrap();
    assert_eq!(player.time(), 0.0);
}

#[test]
fn invalid_time_speed_and_overflow_are_atomic() {
    let (cache, _) = record();
    for invalid in [-1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            cache.sample_at(invalid).unwrap_err(),
            PlaybackError::InvalidTime
        );
    }
    let mut player = TimedPlayback::new(Arc::new(cache)).unwrap();
    player.seek(0.25).unwrap();
    for invalid in [-1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            player.advance(invalid).unwrap_err(),
            PlaybackError::InvalidTime
        );
        assert_eq!(
            player.seek(invalid).unwrap_err(),
            PlaybackError::InvalidTime
        );
        assert_eq!(player.time(), 0.25);
    }
    assert_eq!(player.set_speed(f64::NAN), Err(PlaybackError::InvalidSpeed));
    assert_eq!(player.speed(), 1.0);
    player.set_speed(f64::MAX).unwrap();
    assert_eq!(
        player.advance(2.0).unwrap_err(),
        PlaybackError::TimeOverflow
    );
    assert_eq!(player.time(), 0.25);
    assert_eq!(
        TimedPlayback::new(Arc::new(DriverCache::new(1))).unwrap_err(),
        PlaybackError::EmptyCache,
    );
    assert_eq!(
        TimedPlayback::new(Arc::new(DriverCache::with_version(100, 1))).unwrap_err(),
        PlaybackError::UnsupportedVersion,
    );
}

#[test]
fn births_removals_and_generation_reuse_switch_only_at_boundaries() {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    let old = world.spawn(BodyDesc::dynamic_at(Vec3::X));
    let driver = SimulationDriver::new(DriveMode::Offline {
        substeps: 1,
        iterations: 1,
        target_frames: 2,
    });
    let mut backend = CpuBackend::new(Box::new(XpbdSolver::new()), 1);
    driver.drive(&mut backend, &mut world, 0.25);
    world.bodies.remove(old);
    let new = world.spawn(BodyDesc::dynamic_at(Vec3::splat(100.0)));
    assert_eq!(old.index(), new.index());
    assert_ne!(old.generation(), new.generation());
    driver.drive(&mut backend, &mut world, 0.25);
    let cache = driver.recorded_cache().unwrap();
    let sample = cache.sample_at(0.375).unwrap();
    assert_eq!(sample.pose(old).unwrap().position, Vec3::X);
    assert_eq!(sample.pose(new), None);
    let mut output = StateSnapshot::new();
    sample.render_into(&mut output);
    assert_eq!(output.pose(old), sample.pose(old));
    assert_eq!(output.pose(new), None);
    let sample = cache.sample_at(0.5).unwrap();
    sample.render_into(&mut output);
    assert_eq!(sample.pose(old), None);
    assert_eq!(output.pose(old), None);
    assert_eq!(output.pose(new).unwrap().position, Vec3::splat(100.0));
}

// Use the real offline driver to construct public caches; no private timeline
// mutation or alternate cache-construction semantics are needed here. Static
// bodies keep the authored render endpoints independent of solver integration.
fn record_worlds<const N: usize>(frames: [(PhysicsWorld, f32); N]) -> DriverCache {
    let driver = SimulationDriver::new(DriveMode::Offline {
        substeps: 1,
        iterations: 1,
        target_frames: N as u32,
    });
    let mut backend = CpuBackend::new(Box::new(XpbdSolver::new()), 1);
    for (index, (mut world, dt)) in frames.into_iter().enumerate() {
        assert!(matches!(
            driver.drive(&mut backend, &mut world, dt),
            DriveOutcome::OfflineFrame { frame, .. } if frame == index as u32
        ));
    }
    driver.recorded_cache().unwrap()
}

#[test]
fn empty_cache_and_single_frame_have_explicit_endpoint_behavior() {
    assert_eq!(
        DriverCache::new(0).sample_at(0.0).unwrap_err(),
        PlaybackError::EmptyCache,
    );
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    let body = world.spawn(BodyDesc {
        linear_velocity: Vec3::new(2.0, -4.0, 6.0),
        angular_velocity: Vec3::new(-8.0, 10.0, 12.0),
        ..BodyDesc::static_at(Vec3::new(3.0, 5.0, 7.0))
    });
    let expected = StateSnapshot::capture(&world);
    let cache = record_worlds([(world, 0.25)]);
    let mut output = StateSnapshot::new();
    for time in [0.0, -0.0, 0.125, 0.25, f64::MAX] {
        let sample = cache.sample_at(time).unwrap();
        assert_eq!(sample.time(), time.min(0.25));
        assert!(std::ptr::eq(sample.boundary(), sample.next_boundary()));
        assert_eq!(sample.alpha(), 0.0);
        assert_eq!(sample.pose(body), expected.pose(body));
        sample.render_into(&mut output);
        assert_eq!(output, expected);
    }
    let mut player = TimedPlayback::new(Arc::new(cache)).unwrap();
    player.advance(f64::MAX).unwrap();
    assert_eq!(player.time(), 0.25);
    player.set_repeat(PlaybackRepeat::Loop);
    assert_eq!(player.time(), 0.0);
    player.set_speed(-1.0).unwrap();
    player.advance(0.0625).unwrap();
    assert_eq!(player.time(), 0.1875);
    player.sample().unwrap().render_into(&mut output);
    assert_eq!(output, expected);
}

#[test]
fn empty_world_frames_clear_reused_output_without_stale_slots() {
    let cache = record_worlds([
        (PhysicsWorld::default(), 0.125),
        (PhysicsWorld::default(), 0.5),
    ]);
    let mut stale_world = PhysicsWorld::default();
    let stale = stale_world.spawn(BodyDesc::static_at(Vec3::ONE));
    let mut output = StateSnapshot::capture(&stale_world);
    for time in [0.0, 0.375, 0.625, f64::MAX] {
        let sample = cache.sample_at(time).unwrap();
        assert_eq!(sample.pose(stale), None);
        sample.render_into(&mut output);
        assert_eq!(output, StateSnapshot::new());
        assert_eq!(output.linear_velocity(stale), None);
        assert_eq!(output.angular_velocity_at_slot(0), Vec3::ZERO);
    }
}

#[test]
fn non_binary_durations_use_recorded_f32_time_and_exact_boundary_selection() {
    let durations = [0.1_f32, 0.03, 0.7];
    let cache = record_worlds(durations.map(|dt| (PhysicsWorld::default(), dt)));
    let mut previous = 0.0;
    for (index, dt) in durations.into_iter().enumerate() {
        let time = previous + f64::from(dt);
        assert_eq!(cache.frame(index as u32).unwrap().time(), time);
        assert_eq!(
            cache.sample_at(time).unwrap().boundary().index(),
            index as u32,
        );
        if index > 0 {
            let between = cache.sample_at(previous + f64::from(dt) * 0.5).unwrap();
            assert_eq!(between.boundary().index(), index as u32 - 1);
            assert_eq!(between.next_boundary().index(), index as u32);
            assert_eq!(between.alpha(), 0.5);
            // Alpha may round to 1 in f32, but ownership must not switch early.
            assert_eq!(
                cache
                    .sample_at(time.next_down())
                    .unwrap()
                    .boundary()
                    .index(),
                index as u32 - 1,
            );
        }
        previous = time;
    }
    assert_eq!(cache.duration(), previous);
}

#[test]
fn render_interpolates_both_velocity_columns_without_changing_authority() {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    let body = world.spawn(BodyDesc {
        linear_velocity: Vec3::new(2.0, -4.0, 6.0),
        angular_velocity: Vec3::new(-8.0, 10.0, 12.0),
        ..BodyDesc::static_at(Vec3::ZERO)
    });
    let earlier = world.clone();
    world.bodies.set_position(body, Vec3::new(12.0, 8.0, -4.0));
    world
        .bodies
        .set_orientation(body, Quat::from_rotation_z(0.8));
    world
        .bodies
        .set_linear_velocity(body, Vec3::new(10.0, 4.0, -2.0));
    world
        .bodies
        .set_angular_velocity(body, Vec3::new(8.0, -6.0, 4.0));
    let cache = record_worlds([(earlier, 0.125), (world, 0.5)]);
    let before = [
        cache.frame(0).unwrap().snapshot().state_hash().unwrap(),
        cache.frame(1).unwrap().snapshot().state_hash().unwrap(),
    ];
    let sample = cache.sample_at(0.25).unwrap();
    assert_eq!(sample.alpha(), 0.25);
    let mut output = StateSnapshot::new();
    sample.render_into(&mut output);
    assert_eq!(output.pose(body), sample.pose(body));
    assert_eq!(
        output.pose(body).unwrap().position,
        Vec3::new(3.0, 2.0, -1.0)
    );
    assert!(output
        .pose(body)
        .unwrap()
        .orientation
        .abs_diff_eq(Quat::from_rotation_z(0.2), 1e-6,));
    assert_eq!(
        output.linear_velocity(body),
        Some(Vec3::new(4.0, -2.0, 4.0))
    );
    assert_eq!(
        output.angular_velocity_at_slot(body.index() as usize),
        Vec3::new(-4.0, 6.0, 10.0),
    );
    // Velocities are interpolated recorded values, not a derivative inferred
    // from endpoint displacement, and must never be restored as solver state.
    let mut restored = PhysicsWorld::default();
    sample.restore_boundary(&mut restored);
    assert_eq!(
        restored.bodies.linear_velocity(body),
        Some(Vec3::new(2.0, -4.0, 6.0))
    );
    assert_eq!(
        restored.bodies.angular_velocity(body),
        Some(Vec3::new(-8.0, 10.0, 12.0))
    );
    assert_eq!(restored.bodies.position(body), Some(Vec3::ZERO));
    for (index, hash) in before.into_iter().enumerate() {
        assert_eq!(
            cache
                .frame(index as u32)
                .unwrap()
                .snapshot()
                .state_hash()
                .unwrap(),
            hash
        );
    }
}

#[test]
fn snapshot_projection_keeps_earlier_lifecycle_and_clears_longer_output() {
    let mut world = PhysicsWorld::default();
    let stable = world.spawn(BodyDesc::static_at(Vec3::ZERO));
    let replaced = world.spawn(BodyDesc {
        linear_velocity: Vec3::X * 3.0,
        angular_velocity: Vec3::Y * 5.0,
        ..BodyDesc::static_at(Vec3::ONE)
    });
    let removed = world.spawn(BodyDesc::static_at(Vec3::splat(2.0)));
    let earlier = StateSnapshot::capture(&world);
    let born = world.spawn(BodyDesc::static_at(Vec3::splat(100.0)));
    assert!(world.bodies.remove(replaced));
    let replacement = world.spawn(BodyDesc {
        linear_velocity: Vec3::splat(100.0),
        angular_velocity: Vec3::splat(200.0),
        ..BodyDesc::static_at(Vec3::splat(300.0))
    });
    assert_eq!(replaced.index(), replacement.index());
    assert_ne!(replaced.generation(), replacement.generation());
    assert!(world.bodies.remove(removed));
    world.bodies.set_position(stable, Vec3::X * 8.0);
    let later = StateSnapshot::capture(&world);
    let mut output = later.clone();
    for alpha in [0.0, 0.5, 1.0] {
        earlier.interpolate_into(&later, alpha, &mut output);
        assert_eq!(output.slot_count(), earlier.slot_count());
        assert_eq!(
            output.pose(stable).unwrap().position,
            Vec3::X * (8.0 * alpha)
        );
        for held in [replaced, removed] {
            assert_eq!(output.pose(held), earlier.pose(held));
            assert_eq!(output.linear_velocity(held), earlier.linear_velocity(held));
            assert_eq!(
                output.angular_velocity_at_slot(held.index() as usize),
                earlier.angular_velocity_at_slot(held.index() as usize)
            );
        }
        assert_eq!(output.pose(replacement), None);
        assert_eq!(output.linear_velocity(replacement), None);
        assert_eq!(output.pose(born), None);
    }
    later.interpolate_into(&earlier, 0.0, &mut output);
    assert_eq!(output, later);
    assert_eq!(output.pose(replaced), None);
    assert_eq!(output.pose(removed), None);
    // A shorter/empty target holds all earlier bodies; an empty source must
    // instead clear the output, including its velocity and generation columns.
    earlier.interpolate_into(&StateSnapshot::new(), 0.5, &mut output);
    assert_eq!(output, earlier);
    StateSnapshot::new().interpolate_into(&later, 0.5, &mut output);
    assert_eq!(output, StateSnapshot::new());
}

#[test]
fn snapshot_projection_clamps_alpha_consistently_for_pose_and_velocities() {
    let mut world = PhysicsWorld::default();
    let body = world.spawn(BodyDesc::static_at(Vec3::ZERO));
    let earlier = StateSnapshot::capture(&world);
    world.bodies.set_position(body, Vec3::X * 8.0);
    world.bodies.set_linear_velocity(body, Vec3::Y * 12.0);
    world.bodies.set_angular_velocity(body, Vec3::Z * 16.0);
    let later = StateSnapshot::capture(&world);
    let mut output = StateSnapshot::new();
    for (input, expected) in [
        (-2.0, 0.0),
        (0.0, 0.0),
        (0.25, 0.25),
        (1.0, 1.0),
        (2.0, 1.0),
        (f32::NAN, 0.0),
        (f32::INFINITY, 0.0),
        (f32::NEG_INFINITY, 0.0),
    ] {
        earlier.interpolate_into(&later, input, &mut output);
        assert_eq!(
            output.pose(body).unwrap().position,
            Vec3::X * (8.0 * expected)
        );
        assert_eq!(
            output.linear_velocity(body),
            Some(Vec3::Y * (12.0 * expected))
        );
        assert_eq!(
            output.angular_velocity_at_slot(body.index() as usize),
            Vec3::Z * (16.0 * expected)
        );
    }
}

#[test]
fn finite_extreme_timeline_and_clock_inputs_do_not_require_frame_stepping() {
    for dt in [f32::from_bits(1), f32::MAX] {
        let cache = record_worlds([(PhysicsWorld::default(), dt)]);
        assert_eq!(cache.duration(), f64::from(dt));
        let sample = cache.sample_at(f64::MAX).unwrap();
        assert_eq!(sample.time(), f64::from(dt));
        assert_eq!(sample.boundary().index(), 0);
        let mut player = TimedPlayback::new(Arc::new(cache)).unwrap();
        player.advance(f64::MAX).unwrap();
        assert_eq!(player.time(), f64::from(dt));
        player.set_speed(0.0).unwrap();
        player.advance(f64::MAX).unwrap();
        assert_eq!(player.time(), f64::from(dt));
        player.set_repeat(PlaybackRepeat::Loop);
        player.seek(f64::MAX).unwrap();
        assert!(player.time().is_finite());
        assert!((0.0..f64::from(dt)).contains(&player.time()));
        for invalid in [f64::NEG_INFINITY, f64::INFINITY, f64::NAN] {
            let time = player.time();
            assert_eq!(player.set_speed(invalid), Err(PlaybackError::InvalidSpeed));
            assert_eq!(player.speed(), 0.0);
            assert_eq!(
                player.seek(invalid).unwrap_err(),
                PlaybackError::InvalidTime
            );
            assert_eq!(
                player.advance(invalid).unwrap_err(),
                PlaybackError::InvalidTime
            );
            assert_eq!(player.time(), time);
        }
    }
}

#[test]
fn reverse_loop_roundoff_never_exposes_the_excluded_duration_endpoint() {
    let (cache, _) = record();
    let duration = cache.duration();
    let mut player = TimedPlayback::new(Arc::new(cache)).unwrap();
    player.set_repeat(PlaybackRepeat::Loop);
    player.set_speed(-1.0).unwrap();
    let sample = player.advance(f64::from_bits(1)).unwrap();
    assert!(sample.time().is_finite());
    // Floating-point rem_euclid can round (-epsilon) % duration back up to
    // duration. The player still promises the half-open interval [0, duration).
    assert!(
        (0.0..duration).contains(&sample.time()),
        "reverse wrapping exposed the excluded endpoint: {} / {duration}",
        sample.time(),
    );
    assert!((0.0..duration).contains(&player.time()));
}
