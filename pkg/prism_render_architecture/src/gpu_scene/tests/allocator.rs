use super::super::{GpuCompletionValue, SceneHandle, SceneHandleAllocator, SceneHandleError};

#[test]
fn reuse_waits_for_completion_and_changes_generation() {
    let mut allocator = SceneHandleAllocator::new(3);
    let first = allocator.allocate().unwrap();
    allocator.retire(first, GpuCompletionValue(10)).unwrap();
    let second = allocator.allocate().unwrap();
    assert_ne!(first.index, second.index);
    assert_eq!(allocator.reclaim_completed(GpuCompletionValue(9)), 0);
    assert!(allocator.allocate().is_err());
    assert_eq!(allocator.reclaim_completed(GpuCompletionValue(10)), 1);
    let reused = allocator.allocate().unwrap();
    assert_eq!(reused.index, first.index);
    assert_eq!(reused.generation, first.generation + 1);
    assert!(!allocator.validate(first));
}

#[test]
fn batch_allocation_is_atomic_on_capacity_failure() {
    let mut allocator = SceneHandleAllocator::new(4);
    assert!(allocator.allocate_batch(4).is_err());
    assert_eq!(allocator.stats().live, 0);
    assert_eq!(allocator.allocate_batch(3).unwrap().len(), 3);
}

#[test]
fn invalid_and_duplicate_retire_are_rejected() {
    let mut allocator = SceneHandleAllocator::new(2);
    assert_eq!(
        allocator.retire(SceneHandle::INVALID, GpuCompletionValue(1)),
        Err(SceneHandleError::Invalid)
    );
    let handle = allocator.allocate().unwrap();
    allocator.retire(handle, GpuCompletionValue(1)).unwrap();
    assert_eq!(
        allocator.retire(handle, GpuCompletionValue(1)),
        Err(SceneHandleError::NotLive(handle))
    );
}

#[test]
fn generation_overflow_permanently_exhausts_slot() {
    let mut allocator = SceneHandleAllocator::new(2);
    let handle = allocator.allocate().unwrap();
    allocator.force_generation_for_test(handle, u32::MAX);
    let last = SceneHandle {
        generation: u32::MAX,
        ..handle
    };
    allocator.retire(last, GpuCompletionValue(1)).unwrap();
    assert_eq!(allocator.reclaim_completed(GpuCompletionValue(1)), 0);
    assert_eq!(allocator.stats().exhausted, 1);
    assert!(allocator.allocate().is_err());
}

#[test]
fn unpublished_allocation_can_be_canceled_without_generation_change() {
    let mut allocator = SceneHandleAllocator::new(2);
    let handle = allocator.allocate().unwrap();
    allocator.cancel_allocation(handle).unwrap();
    let reused = allocator.allocate().unwrap();
    assert_eq!(reused, handle);
}

#[test]
fn million_randomized_operations_never_alias_live_handles() {
    let mut allocator = SceneHandleAllocator::new(4096);
    let mut live = Vec::new();
    let mut stale = Vec::new();
    let mut random = 0x9e37_79b9_7f4a_7c15_u64;
    let mut completion = 0_u64;

    for _ in 0..1_000_000 {
        random = random
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        if live.is_empty() || (random & 3 != 0 && live.len() < 4000) {
            if let Ok(handle) = allocator.allocate() {
                assert!(!live.contains(&handle));
                assert!(allocator.validate(handle));
                live.push(handle);
            }
        } else {
            let index = random as usize % live.len();
            let handle = live.swap_remove(index);
            completion += 1;
            allocator
                .retire(handle, GpuCompletionValue(completion))
                .unwrap();
            stale.push(handle);
            allocator.reclaim_completed(GpuCompletionValue(completion.saturating_sub(8)));
        }
    }

    allocator.reclaim_completed(GpuCompletionValue(u64::MAX));
    assert!(live.iter().all(|handle| allocator.validate(*handle)));
    assert!(stale.iter().all(|handle| !allocator.validate(*handle)));
}
