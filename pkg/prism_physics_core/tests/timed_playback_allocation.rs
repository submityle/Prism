//! Allocation contracts for render-only playback, not full-world restoration.

use std::{hint::black_box, sync::Arc};

use glam::Vec3;
use prism_physics_core::{
    BodyDesc, BodyHandle, CpuBackend, DriveMode, DriveOutcome, DriverCache, PhysicsWorld,
    PlaybackRepeat, SimulationDriver, StateSnapshot, TimedPlayback, XpbdSolver,
};

// Keep the unsafe allocator forwarding confined to this test crate. Per-thread
// counters exclude allocations made by parallel tests and the test harness.
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

fn recorded_cache() -> (Arc<DriverCache>, BodyHandle) {
    let mut small = PhysicsWorld::with_gravity(Vec3::ZERO);
    let handle = small.spawn(BodyDesc::static_at(Vec3::ZERO));
    let mut large = small.clone();
    for index in 1..128 {
        large.spawn(BodyDesc::static_at(Vec3::X * index as f32));
    }
    let mut later = large.clone();
    later.bodies.set_position(handle, Vec3::Y * 8.0);
    later.bodies.set_linear_velocity(handle, Vec3::Z * 4.0);
    let driver = SimulationDriver::new(DriveMode::Offline {
        substeps: 1,
        iterations: 1,
        target_frames: 4,
    });
    let mut backend = CpuBackend::new(Box::new(XpbdSolver::new()), 1);
    for mut world in [small, large, later, PhysicsWorld::default()] {
        assert!(matches!(
            driver.drive(&mut backend, &mut world, 0.125),
            DriveOutcome::OfflineFrame { .. }
        ));
    }
    (Arc::new(driver.recorded_cache().unwrap()), handle)
}

#[test]
fn warmed_sampling_and_projection_reuse_capacity_after_shrinking_to_empty() {
    // A positive control prevents a disabled counter from passing the test.
    let (_, activity) = allocation_probe::measure(|| drop(black_box(Box::new(17_u64))));
    assert!(activity.allocations > 0);
    assert!(activity.deallocations > 0);

    let (cache, handle) = recorded_cache();
    let mut player = TimedPlayback::new(Arc::clone(&cache)).unwrap();
    player.set_repeat(PlaybackRepeat::Loop);
    let mut output = StateSnapshot::new();
    cache.sample_at(0.3125).unwrap().render_into(&mut output);
    assert_eq!(output.slot_count(), 128);
    let earlier = cache.frame(1).unwrap().snapshot().render();
    let later = cache.frame(2).unwrap().snapshot().render();
    let empty = StateSnapshot::new();

    let (_, activity) = allocation_probe::measure(|| {
        for _ in 0..128 {
            for time in [0.0, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5] {
                let sample = cache.sample_at(black_box(time)).unwrap();
                let _ = black_box(sample.pose(handle));
                sample.render_into(&mut output);
                let _ = black_box(output.linear_velocity(handle));
                let _ = black_box(output.angular_velocity_at_slot(0));
                player.seek(time).unwrap().render_into(&mut output);
                player.advance(0.03125).unwrap().render_into(&mut output);
                let _ = black_box(player.sample().unwrap().pose(handle));
            }
            empty.interpolate_into(later, 0.5, &mut output);
            earlier.interpolate_into(later, 0.5, &mut output);
            black_box(&output);
        }
    });
    assert_eq!(activity, allocation_probe::Activity::default());
    assert_eq!(output.slot_count(), 128);
    assert_eq!(output.pose(handle).unwrap().position, Vec3::Y * 4.0);
    assert_eq!(output.linear_velocity(handle), Some(Vec3::Z * 2.0));
}

#[test]
fn cloned_players_share_cache_without_heap_activity_and_outlive_the_original() {
    let (cache, handle) = recorded_cache();
    let original = TimedPlayback::new(Arc::clone(&cache)).unwrap();
    let initial_owners = Arc::strong_count(&cache);
    let ((mut left, right), activity) = allocation_probe::measure(|| {
        let left = black_box(&original).clone();
        let right = black_box(&original).clone();
        for _ in 0..128 {
            drop(black_box(&left).clone());
        }
        (left, right)
    });
    assert_eq!(activity, allocation_probe::Activity::default());
    assert_eq!(Arc::strong_count(&cache), initial_owners + 2);
    assert!(Arc::ptr_eq(original.cache(), left.cache()));
    assert!(Arc::ptr_eq(left.cache(), right.cache()));
    left.set_repeat(PlaybackRepeat::Loop);
    left.set_speed(-2.0).unwrap();
    left.seek(0.3125).unwrap();
    assert_eq!(right.time(), 0.0);
    assert_eq!(right.speed(), 1.0);
    assert_eq!(original.time(), 0.0);
    drop(original);
    drop(cache);
    assert_eq!(Arc::strong_count(left.cache()), 2);
    assert_eq!(
        left.sample().unwrap().pose(handle).unwrap().position,
        Vec3::Y * 4.0
    );
    assert_eq!(
        right.sample().unwrap().pose(handle).unwrap().position,
        Vec3::ZERO
    );
    drop(right);
    assert_eq!(Arc::strong_count(left.cache()), 1);
    assert_eq!(left.advance(0.03125).unwrap().time(), 0.25);
}
