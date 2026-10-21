//! Assembly edits must not copy unrelated loads, graph edges or break policies.

use glam::Vec3;
use prism_physics_core::{
    dynamics::assembly::{AssemblyError, AssemblyGraph, BreakLimit, BreakableJointSet},
    joint::{DistanceJoint, JointStorage},
    JointDesc,
};
use prism_physics_core::{dynamics::force::ForceAccumulator, BodyDesc, BodyStorage};

#[allow(
    unsafe_code,
    reason = "GlobalAlloc forwards the original allocation contract to System"
)]
mod probe {
    use std::{
        alloc::{GlobalAlloc, Layout, System},
        cell::Cell,
    };
    thread_local! { static BYTES: Cell<Option<usize>> = const { Cell::new(None) }; }
    struct Allocator;
    fn count(bytes: usize) {
        let _ = BYTES.try_with(|value| {
            if let Some(current) = value.get() {
                value.set(Some(current.saturating_add(bytes)));
            }
        });
    }
    // SAFETY: All operations forward the caller's exact pointer/layout contract.
    unsafe impl GlobalAlloc for Allocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            count(layout.size());
            // SAFETY: The caller supplies a valid allocation layout.
            unsafe { System.alloc(layout) }
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            count(layout.size());
            // SAFETY: The caller supplies a valid allocation layout.
            unsafe { System.alloc_zeroed(layout) }
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            count(size);
            // SAFETY: Pointer and layout are forwarded unchanged.
            unsafe { System.realloc(ptr, layout, size) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // SAFETY: Pointer and original layout come from the allocator caller.
            unsafe { System.dealloc(ptr, layout) }
        }
    }
    #[global_allocator]
    static ALLOCATOR: Allocator = Allocator;
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            BYTES.with(|value| value.set(None));
        }
    }
    pub fn measure(run: impl FnOnce()) -> usize {
        BYTES.with(|value| {
            assert!(value.get().is_none());
            value.set(Some(0));
        });
        let reset = Reset;
        run();
        let bytes = BYTES.with(|value| value.get().unwrap());
        drop(reset);
        bytes
    }
}

fn remap_bytes(unrelated: u32) -> (usize, usize) {
    let mut loads = ForceAccumulator::new();
    let mut bodies = BodyStorage::new();
    let a = bodies.insert(BodyDesc::default());
    let b = bodies.insert(BodyDesc::default());
    let proxy = bodies.insert(BodyDesc::default());
    loads.accumulate_force(a, Vec3::Y).unwrap();
    loads.accumulate_force(b, Vec3::Z).unwrap();
    // Warm spare capacity before measuring. No accumulator growth is needed
    // when the original two loads merge into one and split back into two.
    let spare = bodies.insert(BodyDesc::default());
    loads.accumulate_force(spare, Vec3::ZERO).unwrap();
    let unrelated_handles: Vec<_> = (0..unrelated)
        .map(|_| bodies.insert(BodyDesc::default()))
        .collect();
    for &handle in &unrelated_handles {
        loads.accumulate_force(handle, Vec3::ONE).unwrap();
    }
    let merge = probe::measure(|| {
        assert!(loads
            .merge_bodies(&[(b, Vec3::X), (a, -Vec3::X)], proxy, Vec3::ZERO)
            .is_some());
    });
    let split = probe::measure(|| {
        assert!(loads.split_body(proxy, Vec3::ZERO, &[(b, 1.0, Vec3::X), (a, 1.0, -Vec3::X)]));
    });
    assert_eq!(loads.len(), unrelated as usize + 3);
    for handle in unrelated_handles {
        assert_eq!(loads.pending(handle).unwrap().force, Vec3::ONE);
    }
    (merge, split)
}

#[test]
fn remap_allocation_depends_on_affected_members_not_world_load_count() {
    let small = remap_bytes(0);
    let large = remap_bytes(16_384);
    assert_eq!(small, large, "unrelated force queues must not be cloned");
    assert!(
        large.0 < 1024 && large.1 < 1024,
        "sparse edit bytes: {large:?}"
    );
}

#[test]
fn connection_transaction_does_not_clone_existing_graph_or_policies() {
    let mut bodies = BodyStorage::new();
    let mut joints = JointStorage::new();
    let mut graph = AssemblyGraph::new();
    let mut policies = BreakableJointSet::default();
    let limit = BreakLimit::new(10.0, 10.0);
    for _ in 0..2048 {
        let a = bodies.insert(BodyDesc::default());
        let b = bodies.insert(BodyDesc::default());
        let desc = JointDesc::distance(a, Vec3::ZERO, b, Vec3::ZERO, DistanceJoint::rigid(1.0));
        let joint = joints.insert(desc);
        graph
            .connect_with_policy(&mut policies, a, b, Some(joint), desc.kind, Some(limit))
            .unwrap();
    }
    let a = bodies.insert(BodyDesc::default());
    let b = bodies.insert(BodyDesc::default());
    let desc = JointDesc::distance(a, Vec3::ZERO, b, Vec3::ZERO, DistanceJoint::rigid(1.0));
    let joint = joints.insert(desc);
    let accepted = probe::measure(|| {
        graph
            .connect_with_policy(&mut policies, a, b, Some(joint), desc.kind, Some(limit))
            .unwrap();
    });
    assert!(
        accepted < 16_384,
        "single-edge allocation bytes: {accepted}"
    );

    let c = bodies.insert(BodyDesc::default());
    let rejected = probe::measure(|| {
        assert_eq!(
            graph.connect_with_policy(&mut policies, a, c, Some(joint), desc.kind, Some(limit)),
            Err(AssemblyError::DuplicatePolicy)
        );
    });
    assert_eq!(rejected, 0, "rejection must allocate no transaction copies");
    assert_eq!(graph.connections().count(), 2049);
    assert!(policies.contains(joint));
}
