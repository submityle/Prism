use prism_render_architecture::{
    abi::GenerationalHandle,
    gpu_scene::{GpuCompletionValue, SceneHandleAllocator, SceneHandleError},
};

pub type MaterialHandleError = SceneHandleError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaterialCapacityError {
    pub max_materials: u32,
}

pub struct MaterialHandleAllocator {
    inner: SceneHandleAllocator,
}

impl MaterialHandleAllocator {
    pub fn new(max_materials: u32) -> Self {
        Self {
            inner: SceneHandleAllocator::new(max_materials),
        }
    }

    pub fn allocate(&mut self) -> Result<GenerationalHandle, MaterialCapacityError> {
        self.inner
            .allocate()
            .map_err(|error| MaterialCapacityError {
                max_materials: error.max_slots,
            })
    }

    pub fn retire(
        &mut self,
        handle: GenerationalHandle,
        completion: GpuCompletionValue,
    ) -> Result<(), MaterialHandleError> {
        self.inner.retire(handle, completion)
    }

    pub fn cancel(&mut self, handle: GenerationalHandle) -> Result<(), MaterialHandleError> {
        self.inner.cancel_allocation(handle)
    }

    pub fn reclaim_completed(&mut self, completed: GpuCompletionValue) -> u32 {
        self.inner.reclaim_completed(completed)
    }

    pub fn validate(&self, handle: GenerationalHandle) -> bool {
        self.inner.validate(handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocated_handles_validate() {
        let mut allocator = MaterialHandleAllocator::new(8);
        let handle = allocator.allocate().unwrap();
        assert!(handle.is_valid());
        assert!(allocator.validate(handle));
    }

    #[test]
    fn allocation_stops_at_capacity_and_reports_limit() {
        // Slot zero is a reserved placeholder, so a capacity of two leaves room
        // for exactly one user allocation before the pool is exhausted.
        let mut allocator = MaterialHandleAllocator::new(2);
        let first = allocator.allocate().unwrap();
        assert!(allocator.validate(first));
        let overflow = allocator.allocate();
        assert_eq!(overflow, Err(MaterialCapacityError { max_materials: 2 }));
    }

    #[test]
    fn validate_rejects_stale_generation() {
        let mut allocator = MaterialHandleAllocator::new(8);
        let handle = allocator.allocate().unwrap();
        let stale = GenerationalHandle {
            index: handle.index,
            generation: handle.generation.wrapping_add(1),
        };
        assert!(!allocator.validate(stale));
    }

    #[test]
    fn retire_then_reclaim_frees_the_slot() {
        let mut allocator = MaterialHandleAllocator::new(8);
        let handle = allocator.allocate().unwrap();
        allocator.retire(handle, GpuCompletionValue(10)).unwrap();
        // The GPU has not yet passed the retirement fence.
        assert_eq!(allocator.reclaim_completed(GpuCompletionValue(9)), 0);
        // Once the fence is reached the slot is reclaimed exactly once.
        assert_eq!(allocator.reclaim_completed(GpuCompletionValue(10)), 1);
        assert!(!allocator.validate(handle));
    }

    #[test]
    fn cancel_returns_a_pending_allocation() {
        let mut allocator = MaterialHandleAllocator::new(8);
        let handle = allocator.allocate().unwrap();
        allocator.cancel(handle).unwrap();
        assert!(!allocator.validate(handle));
    }
}
