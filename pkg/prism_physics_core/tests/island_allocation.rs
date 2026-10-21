//! Packed connectivity must not allocate a buffer for every small island.

use glam::Vec3;
use prism_physics_core::collide::contact::{ContactManifold, ContactPoint};
use prism_physics_core::island::IslandBuilder;
use prism_physics_core::solver::xpbd::{ContactConstraint, SolveIslands};
use prism_physics_core::state::storage::BodyStorage;
use prism_physics_core::BodyDesc;

#[allow(unsafe_code, reason = "GlobalAlloc requires forwarding to System")]
mod probe {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        static COUNT: Cell<Option<usize>> = const { Cell::new(None) };
    }

    fn record() {
        let _ = COUNT.try_with(|count| {
            if let Some(value) = count.get() {
                count.set(Some(value + 1));
            }
        });
    }

    struct Allocator;

    // SAFETY: The caller's pointer/layout contract is forwarded unchanged to
    // System. Counting is thread-local and never accesses allocation contents.
    unsafe impl GlobalAlloc for Allocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            record();
            // SAFETY: The GlobalAlloc caller supplies a valid layout.
            unsafe { System.alloc(layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            record();
            // SAFETY: The GlobalAlloc caller supplies a valid layout.
            unsafe { System.alloc_zeroed(layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            record();
            // SAFETY: The original pointer/layout and new size come from caller.
            unsafe { System.realloc(ptr, layout, size) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // SAFETY: The original allocation pointer/layout come from caller.
            unsafe { System.dealloc(ptr, layout) }
        }
    }

    #[global_allocator]
    static ALLOCATOR: Allocator = Allocator;

    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            COUNT.with(|count| count.set(None));
        }
    }

    pub fn measure<T>(run: impl FnOnce() -> T) -> (T, usize) {
        COUNT.with(|count| {
            assert!(count.get().is_none());
            count.set(Some(0));
        });
        let reset = Reset;
        let result = run();
        let count = COUNT.with(|count| count.get().unwrap());
        drop(reset);
        (result, count)
    }
}

fn scene(count: usize) -> (BodyStorage, Vec<ContactManifold>) {
    let mut bodies = BodyStorage::new();
    let ground = bodies.insert(BodyDesc::static_at(Vec3::ZERO));
    let mut contacts = Vec::with_capacity(count);
    for i in 0..count {
        let position = Vec3::new(i as f32 * 4.0, 0.4, 0.0);
        let body = bodies.insert(BodyDesc::dynamic_at(position));
        let mut manifold = ContactManifold::new(ground, body, Vec3::Y);
        manifold.push(ContactPoint::new(position, position - Vec3::Y * 0.1, 0.1));
        contacts.push(manifold);
    }
    (bodies, contacts)
}

#[test]
fn contact_working_set_allocates_only_its_outer_buffer() {
    let (mut bodies, manifolds) = scene(4096);
    let view = bodies.solver_view_mut();
    let (constraints, allocations) = probe::measure(|| ContactConstraint::build(&view, &manifolds));
    assert_eq!(constraints.len(), manifolds.len());
    assert_eq!(allocations, 1, "contact points must remain inline");
    let (_, clone_allocations) = probe::measure(|| {
        for constraint in &constraints {
            std::hint::black_box(constraint.clone());
        }
    });
    assert_eq!(
        clone_allocations, 0,
        "scratch copies must not allocate points"
    );
}

#[test]
fn independent_components_use_packed_allocations() {
    let mut build_counts = Vec::new();
    let mut solve_counts = Vec::new();
    for count in [64, 8192] {
        let mut builder = IslandBuilder::new(count);
        let (partition, allocations) = probe::measure(|| builder.build());
        assert_eq!(partition.island_count(), count);
        for slot in 0..count {
            assert_eq!(partition.members(partition.island_of(slot)), [slot]);
        }
        assert!(
            allocations < 32,
            "{count} components: {allocations} allocations"
        );
        build_counts.push(allocations);

        let (mut bodies, manifolds) = scene(count);
        let view = bodies.solver_view_mut();
        let constraints = ContactConstraint::build(&view, &manifolds);
        let (islands, allocations) =
            probe::measure(|| SolveIslands::build(&view, &constraints, &[]));
        assert_eq!(islands.island_count(), count);
        for i in 0..count {
            assert_eq!(islands.contacts(i), [i]);
            assert_eq!(islands.members(i), [i + 1]);
            assert!(islands.joints(i).is_empty());
        }
        assert!(
            allocations < 40,
            "{count} solver islands: {allocations} allocations"
        );
        solve_counts.push(allocations);
    }
    // Only geometric growth of the contiguous count buffers may add calls.
    assert!(build_counts[1] <= build_counts[0] + 8, "{build_counts:?}");
    assert!(solve_counts[1] <= solve_counts[0] + 8, "{solve_counts:?}");
    eprintln!("packed partition allocations {build_counts:?}; solver partition {solve_counts:?}");
}

#[cfg(feature = "parallel")]
#[test]
fn tiny_island_scratch_allocations_scale_with_batches() {
    use prism_physics_core::solver::xpbd::parallel_solve::solve_islands_parallel;
    use prism_physics_core::solver::XpbdConfig;

    let count = 4096;
    let (mut bodies, manifolds) = scene(count);
    let mut view = bodies.solver_view_mut();
    let constraints = ContactConstraint::build(&view, &manifolds);
    let islands = SolveIslands::build(&view, &constraints, &[]);
    let active: Vec<_> = (0..islands.island_count()).collect();
    let (loads, allocations) = probe::measure(|| {
        // Serial scratch execution isolates allocations from worker TLS and
        // Rayon initialization. It follows the production batching schedule.
        solve_islands_parallel(
            &mut view,
            &islands,
            &active,
            &constraints,
            &mut [],
            &[],
            &XpbdConfig::default(),
            1.0 / 60.0,
            false,
        )
    });
    assert!(loads.loads().unwrap().is_empty());
    assert!(
        allocations < count / 2,
        "{count} tiny islands: {allocations} allocations"
    );
    for slot in 1..=count {
        assert!(view.positions[slot].is_finite());
        assert!(view.positions[slot].y > 0.4);
    }
    eprintln!("{count} tiny-island scratch allocations: {allocations}");
}
