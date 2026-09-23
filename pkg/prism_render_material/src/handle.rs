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
