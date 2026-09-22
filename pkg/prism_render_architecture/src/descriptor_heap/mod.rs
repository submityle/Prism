//! Global bindless descriptor indirection.

use crate::abi::GenerationalHandle;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ResourceClass {
    SampledImage,
    StorageImage,
    Sampler,
    Buffer,
    AccelerationStructure,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GpuResourceHandle {
    pub handle: GenerationalHandle,
    pub class: ResourceClass,
}

/// Maps stable logical handles onto relocatable physical descriptor slots.
pub trait DescriptorHeap {
    fn resolve(&self, handle: GpuResourceHandle) -> Option<u32>;
    fn retire(&mut self, handle: GpuResourceHandle, frame_epoch: u64);
}
