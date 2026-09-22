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
