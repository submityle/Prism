//! Retained bounded-PD scratch adds no warmed serial-solver heap activity.
//! Other XPBD stages still allocate; compare identical zero-effort topology
//! against no-policy topology rather than asserting the whole solver is free.

use glam::Vec3;
use prism_physics_core::joint::{
    BoundedPd, JointAnchor, JointDesc, JointKind, PrismaticJoint, RevoluteJoint,
};
use prism_physics_core::snapshot::{world_hash::hash_world, FullStateSnapshot};
use prism_physics_core::solver::{Solver, XpbdConfig, XpbdSolver};
use prism_physics_core::{BodyDesc, PhysicsWorld};

// Keep unsafe forwarding confined to this test crate. Per-thread counters
// exclude allocations made by parallel tests and the test harness.
#[allow(
    unsafe_code,
    reason = "GlobalAlloc requires unsafe forwarding to System"
)]
mod allocation_probe {
    use std::{
        alloc::{GlobalAlloc, Layout, System},
        cell::Cell,
    };

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Activity {
        pub allocations: usize,
        pub deallocations: usize,
        pub reallocations: usize,
    }

    thread_local! {
        static ACTIVE: Cell<Option<Activity>> = const { Cell::new(None) };
    }

    fn count(update: impl FnOnce(&mut Activity)) {
        // TLS may already be unavailable while the thread is shutting down.
        let _ = ACTIVE.try_with(|active| {
            if let Some(mut activity) = active.get() {
                update(&mut activity);
                active.set(Some(activity));
            }
        });
    }

    struct Probe;

    // SAFETY: Each operation forwards the caller's pointer/layout contract
    // unchanged to System; counter bookkeeping never accesses allocated data.
    unsafe impl GlobalAlloc for Probe {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            count(|a| a.allocations = a.allocations.saturating_add(1));
            // SAFETY: Guaranteed by the GlobalAlloc caller.
            unsafe { System.alloc(layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            count(|a| a.allocations = a.allocations.saturating_add(1));
            // SAFETY: Guaranteed by the GlobalAlloc caller.
            unsafe { System.alloc_zeroed(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            count(|a| a.deallocations = a.deallocations.saturating_add(1));
            // SAFETY: Pointer and original layout come from the caller.
            unsafe { System.dealloc(ptr, layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            count(|a| a.reallocations = a.reallocations.saturating_add(1));
            // SAFETY: Pointer, original layout and new size come from caller.
            unsafe { System.realloc(ptr, layout, size) }
        }
    }

    #[global_allocator]
    static ALLOCATOR: Probe = Probe;

    struct Reset;

    impl Drop for Reset {
        fn drop(&mut self) {
            ACTIVE.with(|active| active.set(None));
        }
    }

    pub fn measure<T>(run: impl FnOnce() -> T) -> (T, Activity) {
        ACTIVE.with(|active| {
            assert!(active.get().is_none(), "allocation probes cannot nest");
            active.set(Some(Activity::default()));
        });
        let reset = Reset;
        let result = run();
        let activity = ACTIVE.with(|active| active.get().unwrap());
        drop(reset);
        (result, activity)
    }
}

fn solver(parallel: bool, colored: bool) -> XpbdSolver {
    XpbdSolver::with_config(XpbdConfig {
        parallel_islands: parallel,
        parallel_within_island: colored,
        ..XpbdConfig::default()
    })
}

fn scene(count: usize, hinge: bool, policy: Option<BoundedPd>) -> PhysicsWorld {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    world.config.sleep.enabled = false;
    for _ in 0..count {
        let a = world.spawn(BodyDesc::static_at(Vec3::ZERO));
        let b = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO).with_damping(0.0, 0.0));
        let kind = if hinge {
            JointKind::Revolute(RevoluteJoint::new(Vec3::Z))
        } else {
            JointKind::Prismatic(PrismaticJoint::new(Vec3::X))
        };
        let mut desc = JointDesc::new(JointAnchor::at_center(a), JointAnchor::at_center(b), kind);
        if let Some(policy) = policy {
            desc = desc.with_bounded_pd(policy);
        }
        world.spawn_joint(desc);
    }
    world
}

fn measured_step(solver: &mut XpbdSolver, seed: &PhysicsWorld) -> allocation_probe::Activity {
    // World restoration and destruction must stay outside the measurement.
    let mut world = seed.clone();
    let (_, activity) = allocation_probe::measure(|| {
        solver.step(&mut world, 0.0625, 4);
        std::hint::black_box(&world);
    });
    activity
}

#[test]
fn warmed_pd_scratch_has_no_extra_allocations_after_empty_and_no_drive_steps() {
    for hinge in [false, true] {
        // A zero cap samples/prepares every joint but emits no loads/activity,
        // isolating the three scratch buffers from wake/load pipeline costs.
        let zero = BoundedPd::new(1.0, -2.0, 3.0, 4.0, 0.0).unwrap();
        let driven = scene(32, hinge, Some(zero));
        let plain = scene(32, hinge, None);
        let empty = PhysicsWorld::with_gravity(Vec3::ZERO);
        let mut retained = solver(false, false);
        let cold = measured_step(&mut retained, &driven);
        let warm = measured_step(&mut retained, &driven);
        assert!(
            cold.allocations + cold.reallocations >= warm.allocations + warm.reallocations + 3,
            "positive control: cold {cold:?}, warm {warm:?}",
        );

        let mut no_drive = solver(false, false);
        let baseline = measured_step(&mut no_drive, &plain);
        assert_eq!(measured_step(&mut no_drive, &plain), baseline);
        assert_eq!(warm, baseline, "only retained scratch may differ");
        // Solver clones carry configuration, not disposable buffer storage.
        let mut fresh_clone = retained.clone();
        assert_eq!(measured_step(&mut fresh_clone, &driven), cold);
        for _ in 0..8 {
            assert_eq!(
                measured_step(&mut retained, &empty),
                measured_step(&mut no_drive, &empty)
            );
            assert_eq!(measured_step(&mut retained, &plain), baseline);
            assert_eq!(measured_step(&mut retained, &driven), warm);
        }
    }
}

#[test]
fn retained_scratch_is_not_snapshot_state_and_preserves_replay_in_each_solve_mode() {
    let policy = BoundedPd::new(0.7, -0.3, 2.0, 1.0, 5.0).unwrap();
    for hinge in [false, true] {
        for (parallel, colored) in [(false, false), (true, false), (false, true), (true, true)] {
            let seed = scene(8, hinge, Some(policy));
            let snapshot = FullStateSnapshot::capture(0, &seed);
            let initial_hash = snapshot.state_hash().unwrap();
            let mut retained = solver(parallel, colored);
            // Warm with unrelated handles/slot counts and nonzero activity.
            let mut unrelated = scene(32, !hinge, Some(policy));
            retained.step(&mut unrelated, 0.125, 4);
            let mut warm_world = snapshot.fork_world();
            let mut fresh_world = snapshot.fork_world();
            let mut fresh = solver(parallel, colored);
            for _ in 0..4 {
                retained.step(&mut warm_world, 0.0625, 4);
                fresh.step(&mut fresh_world, 0.0625, 4);
                assert_eq!(
                    hash_world(&warm_world).unwrap(),
                    hash_world(&fresh_world).unwrap()
                );
            }
            assert_eq!(snapshot.state_hash().unwrap(), initial_hash);
            snapshot.restore_into(&mut warm_world);
            snapshot.restore_into(&mut fresh_world);
            retained.step(&mut warm_world, 0.0625, 4);
            fresh.step(&mut fresh_world, 0.0625, 4);
            assert_eq!(
                hash_world(&warm_world).unwrap(),
                hash_world(&fresh_world).unwrap()
            );
        }
    }
}

#[test]
fn shared_solver_contact_callback_can_reenter_with_independent_scratch() {
    let policy = BoundedPd::new(0.5, 0.2, 2.0, 1.0, 3.0).unwrap();
    let shared = solver(false, false);
    let seed = scene(4, false, Some(policy));
    let nested_seed = scene(8, true, Some(policy));
    let mut outer = seed.clone();
    let mut nested = nested_seed.clone();
    let mut callbacks = 0;
    shared.step_with_contact_generator(&mut outer, 0.0625, 1, |_, _| {
        callbacks += 1;
        shared.step_with_contact_generator(&mut nested, 0.0625, 1, |_, _| {});
    });
    assert_eq!(callbacks, 1);
    let mut expected_outer = seed;
    let mut expected_nested = nested_seed;
    let mut fresh = solver(false, false);
    fresh.step(&mut expected_outer, 0.0625, 1);
    fresh.step(&mut expected_nested, 0.0625, 1);
    assert_eq!(
        hash_world(&outer).unwrap(),
        hash_world(&expected_outer).unwrap()
    );
    assert_eq!(
        hash_world(&nested).unwrap(),
        hash_world(&expected_nested).unwrap()
    );
}
