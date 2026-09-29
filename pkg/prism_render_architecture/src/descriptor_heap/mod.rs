//! Global bindless descriptor indirection.
//!
//! Prism binds resources through a single large bindless descriptor heap: a
//! shader reads a resource by a small integer index rather than a bound slot.
//! This module owns the CPU-side contract that hands out those indices as
//! stable, generational handles and keeps them collision-free across resource
//! kinds, safe against stale reuse, and safe against in-flight GPU reads.
//!
//! * [`allocator`] — generational, free-list slot allocator for one segment.
//! * [`retirement`] — epoch-gated deferred reclamation of retired slots.
//! * [`heap`] — the segmented [`BindlessDescriptorHeap`] and the
//!   [`DescriptorHeap`] contract implementation.
//!
//! The layer is `GPU`-independent and deterministic; the descriptor writes it
//! describes are pending the GPU backend.

use crate::abi::GenerationalHandle;

pub mod allocator;
pub mod heap;
pub mod retirement;

pub use allocator::{AllocError, GenerationalAllocator};
pub use heap::{BindlessDescriptorHeap, ClassStats, HeapConfig};
pub use retirement::RetirementQueue;

/// The kind of GPU resource a descriptor slot binds.
///
/// Each class occupies its own contiguous segment of the bindless heap, so a
/// sampled-image index and a sampler index that share a value still name
/// different bindings.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ResourceClass {
    SampledImage,
    StorageImage,
    Sampler,
    Buffer,
    AccelerationStructure,
}

impl ResourceClass {
    /// Every resource class, in the fixed order segments are laid out.
    pub const ALL: [Self; Self::COUNT] = [
        Self::SampledImage,
        Self::StorageImage,
        Self::Sampler,
        Self::Buffer,
        Self::AccelerationStructure,
    ];

    /// Number of distinct resource classes.
    pub const COUNT: usize = 5;

    /// Dense array index for this class, matching its position in [`Self::ALL`].
    #[must_use]
    pub const fn ordinal(self) -> usize {
        match self {
            Self::SampledImage => 0,
            Self::StorageImage => 1,
            Self::Sampler => 2,
            Self::Buffer => 3,
            Self::AccelerationStructure => 4,
        }
    }
}

/// A stable logical reference to a bound resource: a generational handle tagged
/// with the [`ResourceClass`] whose segment owns it.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinal_matches_all_ordering() {
        for (i, class) in ResourceClass::ALL.iter().enumerate() {
            assert_eq!(class.ordinal(), i);
        }
        assert_eq!(ResourceClass::ALL.len(), ResourceClass::COUNT);
    }

    #[test]
    fn ordinals_are_unique() {
        let mut seen = [false; ResourceClass::COUNT];
        for class in ResourceClass::ALL {
            let o = class.ordinal();
            assert!(!seen[o], "duplicate ordinal {o}");
            seen[o] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }
}
