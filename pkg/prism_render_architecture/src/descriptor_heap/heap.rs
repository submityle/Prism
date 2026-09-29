//! Segmented bindless descriptor heap and the [`DescriptorHeap`] contract impl.
//!
//! A bindless heap exposes one flat array of descriptor slots to shaders, but
//! different resource kinds must not collide: a sampled-image index and a
//! sampler index that happen to be equal name different bindings. This heap
//! carves the flat range into one contiguous *segment* per [`ResourceClass`],
//! each with its own capacity and its own [`GenerationalAllocator`]. A logical
//! handle names a slot inside its class segment; the heap adds the segment base
//! to produce the physical descriptor index the GPU sees.
//!
//! Segments are laid out back to back in [`ResourceClass::ALL`] order, so the
//! physical range of class *k* is `[base_k, base_k + capacity_k)` and the whole
//! heap spans `[0, total_capacity)`. Layout is fixed at construction from a
//! [`HeapConfig`], which keeps physical indices stable across a session — a
//! requirement for descriptor tables the backend uploads once.
//!
//! Retirement is deferred through a per-segment [`RetirementQueue`]. Retiring a
//! handle invalidates it immediately (its generation is bumped) yet keeps the
//! physical slot reserved; [`BindlessDescriptorHeap::reclaim`] later returns the
//! slot to its allocator once the configured frames-in-flight have elapsed. The
//! heap is `GPU`-independent and deterministic: it maps handles to indices and
//! tracks ownership, while the descriptor writes themselves are pending the GPU
//! backend.

use crate::descriptor_heap::allocator::{AllocError, GenerationalAllocator};
use crate::descriptor_heap::retirement::RetirementQueue;
use crate::descriptor_heap::{DescriptorHeap, GpuResourceHandle, ResourceClass};

/// Per-class capacity plan for a [`BindlessDescriptorHeap`].
///
/// Capacities default to zero; set each class you intend to use. A zero-capacity
/// class allocates nothing and always reports [`AllocError::Exhausted`].
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct HeapConfig {
    capacities: [u32; ResourceClass::COUNT],
}

impl HeapConfig {
    /// Creates a config with every class capacity set to zero.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            capacities: [0; ResourceClass::COUNT],
        }
    }

    /// Sets the slot `capacity` for `class`, returning the updated config.
    #[must_use]
    pub const fn with_class(mut self, class: ResourceClass, capacity: u32) -> Self {
        self.capacities[class.ordinal()] = capacity;
        self
    }

    /// Capacity currently configured for `class`.
    #[must_use]
    pub const fn capacity(&self, class: ResourceClass) -> u32 {
        self.capacities[class.ordinal()]
    }

    /// Sum of every class capacity, i.e. the physical heap size.
    #[must_use]
    pub fn total_capacity(&self) -> u32 {
        self.capacities.iter().copied().sum()
    }
}

/// Usage snapshot for one resource-class segment.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ClassStats {
    /// Slots currently owned by a live handle.
    pub used: u32,
    /// Total slots in the segment.
    pub capacity: u32,
    /// Slots retired but not yet reclaimed.
    pub pending_retirement: u32,
}

impl ClassStats {
    /// Slots immediately available for allocation.
    #[must_use]
    pub const fn free(&self) -> u32 {
        self.capacity - self.used - self.pending_retirement
    }
}

/// One class segment: its physical base plus allocation and retirement state.
#[derive(Clone, Debug)]
struct Segment {
    /// Physical index of this segment's first slot in the flat heap.
    base: u32,
    /// Generational allocator over the segment's local slot range.
    allocator: GenerationalAllocator,
    /// Deferred-retirement queue for this segment's slots.
    retirement: RetirementQueue,
}

/// Fixed-layout, segmented bindless descriptor heap.
///
/// Construct one from a [`HeapConfig`], allocate [`GpuResourceHandle`]s per
/// class, resolve them to physical indices through the [`DescriptorHeap`]
/// contract, and each frame call [`reclaim`](Self::reclaim) to return
/// safely-retired slots to their segments.
#[derive(Clone, Debug)]
pub struct BindlessDescriptorHeap {
    /// One segment per [`ResourceClass`], indexed by [`ResourceClass::ordinal`].
    segments: [Segment; ResourceClass::COUNT],
    /// Total physical slot count across all segments.
    total_capacity: u32,
}

impl BindlessDescriptorHeap {
    /// Builds a heap whose segments follow `config`, laid out in
    /// [`ResourceClass::ALL`] order with contiguous, non-overlapping ranges.
    #[must_use]
    pub fn new(config: HeapConfig) -> Self {
        let mut base = 0u32;
        let segments = ResourceClass::ALL.map(|class| {
            let capacity = config.capacity(class);
            let segment = Segment {
                base,
                allocator: GenerationalAllocator::new(capacity),
                retirement: RetirementQueue::new(),
            };
            base += capacity;
            segment
        });
        Self {
            segments,
            total_capacity: base,
        }
    }

    /// Total physical slot count across every class segment.
    #[must_use]
    pub const fn total_capacity(&self) -> u32 {
        self.total_capacity
    }

    /// Physical index of `class`'s first slot in the flat heap.
    #[must_use]
    pub fn segment_base(&self, class: ResourceClass) -> u32 {
        self.segment(class).base
    }

    /// Allocates a slot in `class`, returning a handle tagged with that class.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError::Exhausted`] when the class segment is full (every
    /// slot live or retiring); the heap never panics on overflow.
    pub fn allocate(&mut self, class: ResourceClass) -> Result<GpuResourceHandle, AllocError> {
        let handle = self.segment_mut(class).allocator.allocate()?;
        Ok(GpuResourceHandle { handle, class })
    }

    /// Reclaims every slot, across all classes, whose deferred retirement has
    /// aged past `frames_in_flight` relative to `current_epoch`.
    ///
    /// Returns the number of slots returned to their allocators. Call once per
    /// frame with the GPU's current frame epoch and buffered-frame depth.
    pub fn reclaim(&mut self, current_epoch: u64, frames_in_flight: u32) -> u32 {
        let mut reclaimed = 0u32;
        for segment in &mut self.segments {
            for index in segment.retirement.reclaim(current_epoch, frames_in_flight) {
                if segment.allocator.reclaim_slot(index) {
                    reclaimed += 1;
                }
            }
        }
        reclaimed
    }

    /// Usage snapshot for a single class segment.
    #[must_use]
    pub fn class_stats(&self, class: ResourceClass) -> ClassStats {
        let segment = self.segment(class);
        ClassStats {
            used: segment.allocator.live_count(),
            capacity: segment.allocator.capacity(),
            pending_retirement: segment.retirement.pending_count(),
        }
    }

    /// Convenience accessor returning per-class stats in [`ResourceClass::ALL`]
    /// order.
    #[must_use]
    pub fn stats(&self) -> [ClassStats; ResourceClass::COUNT] {
        ResourceClass::ALL.map(|class| self.class_stats(class))
    }

    /// Shared access to a class segment.
    fn segment(&self, class: ResourceClass) -> &Segment {
        &self.segments[class.ordinal()]
    }

    /// Exclusive access to a class segment.
    fn segment_mut(&mut self, class: ResourceClass) -> &mut Segment {
        &mut self.segments[class.ordinal()]
    }
}

impl DescriptorHeap for BindlessDescriptorHeap {
    fn resolve(&self, handle: GpuResourceHandle) -> Option<u32> {
        let segment = self.segment(handle.class);
        segment
            .allocator
            .resolve(handle.handle)
            .map(|index| segment.base + index)
    }

    fn retire(&mut self, handle: GpuResourceHandle, frame_epoch: u64) {
        let class = handle.class;
        let index = handle.handle.index;
        let segment = self.segment_mut(class);
        if segment.allocator.retire(handle.handle) {
            segment.retirement.retire(index, frame_epoch);
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::abi::GenerationalHandle;

    fn config() -> HeapConfig {
        HeapConfig::new()
            .with_class(ResourceClass::SampledImage, 4)
            .with_class(ResourceClass::Sampler, 2)
            .with_class(ResourceClass::Buffer, 3)
    }

    #[test]
    fn layout_places_segments_contiguously_in_class_order() {
        let heap = BindlessDescriptorHeap::new(config());
        // ALL order: SampledImage(4), StorageImage(0), Sampler(2), Buffer(3),
        // AccelerationStructure(0).
        assert_eq!(heap.segment_base(ResourceClass::SampledImage), 0);
        assert_eq!(heap.segment_base(ResourceClass::StorageImage), 4);
        assert_eq!(heap.segment_base(ResourceClass::Sampler), 4);
        assert_eq!(heap.segment_base(ResourceClass::Buffer), 6);
        assert_eq!(heap.segment_base(ResourceClass::AccelerationStructure), 9);
        assert_eq!(heap.total_capacity(), 9);
    }

    #[test]
    fn resolve_maps_handles_into_the_right_physical_segment() {
        let mut heap = BindlessDescriptorHeap::new(config());
        let img = heap.allocate(ResourceClass::SampledImage).unwrap();
        let sampler = heap.allocate(ResourceClass::Sampler).unwrap();
        let buffer = heap.allocate(ResourceClass::Buffer).unwrap();
        // Each resolves to base + local index; classes never collide.
        assert_eq!(heap.resolve(img), Some(0));
        assert_eq!(heap.resolve(sampler), Some(4));
        assert_eq!(heap.resolve(buffer), Some(6));
    }

    #[test]
    fn classes_are_isolated_and_share_no_slots() {
        let mut heap = BindlessDescriptorHeap::new(config());
        let img0 = heap.allocate(ResourceClass::SampledImage).unwrap();
        let sampler0 = heap.allocate(ResourceClass::Sampler).unwrap();
        // Same local index (0) in different classes, distinct physical slots.
        assert_eq!(img0.handle.index, sampler0.handle.index);
        assert_ne!(heap.resolve(img0), heap.resolve(sampler0));
    }

    #[test]
    fn overflow_returns_error_per_class() {
        let mut heap = BindlessDescriptorHeap::new(config());
        assert!(heap.allocate(ResourceClass::Sampler).is_ok());
        assert!(heap.allocate(ResourceClass::Sampler).is_ok());
        assert_eq!(
            heap.allocate(ResourceClass::Sampler),
            Err(AllocError::Exhausted)
        );
        // A different class is unaffected.
        assert!(heap.allocate(ResourceClass::SampledImage).is_ok());
    }

    #[test]
    fn zero_capacity_class_is_always_exhausted() {
        let mut heap = BindlessDescriptorHeap::new(config());
        assert_eq!(
            heap.allocate(ResourceClass::StorageImage),
            Err(AllocError::Exhausted)
        );
    }

    #[test]
    fn retire_invalidates_immediately_and_defers_slot_reuse() {
        let mut heap = BindlessDescriptorHeap::new(config());
        let img = heap.allocate(ResourceClass::SampledImage).unwrap();
        assert_eq!(heap.resolve(img), Some(0));
        heap.retire(img, 10);
        // Stale handle no longer resolves.
        assert_eq!(heap.resolve(img), None);
        let stats = heap.class_stats(ResourceClass::SampledImage);
        assert_eq!(stats.used, 0);
        assert_eq!(stats.pending_retirement, 1);
        // Too early to reclaim (10 + 2 > 11): slot stays reserved.
        assert_eq!(heap.reclaim(11, 2), 0);
        assert_eq!(
            heap.class_stats(ResourceClass::SampledImage)
                .pending_retirement,
            1
        );
    }

    #[test]
    fn reclaim_after_frames_in_flight_recycles_slot() {
        let mut heap = BindlessDescriptorHeap::new(config());
        let img = heap.allocate(ResourceClass::SampledImage).unwrap();
        heap.retire(img, 10);
        // 10 + 2 <= 12: reclaimable now.
        assert_eq!(heap.reclaim(12, 2), 1);
        let stats = heap.class_stats(ResourceClass::SampledImage);
        assert_eq!(stats.pending_retirement, 0);
        assert_eq!(stats.free(), 4);
        // Reused slot yields a fresh generation; the old handle stays dead.
        let reused = heap.allocate(ResourceClass::SampledImage).unwrap();
        assert_eq!(reused.handle.index, img.handle.index);
        assert_ne!(reused.handle.generation, img.handle.generation);
        assert_eq!(heap.resolve(img), None);
        assert_eq!(heap.resolve(reused), Some(0));
    }

    #[test]
    fn retiring_unknown_handle_is_ignored() {
        let mut heap = BindlessDescriptorHeap::new(config());
        let bogus = GpuResourceHandle {
            handle: GenerationalHandle {
                index: 0,
                generation: 5,
            },
            class: ResourceClass::Buffer,
        };
        // No live slot matches: retire must not enqueue anything.
        heap.retire(bogus, 3);
        assert_eq!(
            heap.class_stats(ResourceClass::Buffer).pending_retirement,
            0
        );
        assert_eq!(heap.reclaim(100, 0), 0);
    }

    #[test]
    fn stats_reflect_live_and_pending_counts() {
        let mut heap = BindlessDescriptorHeap::new(config());
        let a = heap.allocate(ResourceClass::Buffer).unwrap();
        let _b = heap.allocate(ResourceClass::Buffer).unwrap();
        heap.retire(a, 1);
        let stats = heap.class_stats(ResourceClass::Buffer);
        assert_eq!(stats.used, 1);
        assert_eq!(stats.capacity, 3);
        assert_eq!(stats.pending_retirement, 1);
        assert_eq!(stats.free(), 1);
    }

    #[test]
    fn deterministic_alloc_retire_reclaim_cycle() {
        let build = || {
            let mut heap = BindlessDescriptorHeap::new(config());
            let mut resolved = Vec::new();
            for epoch in 0..6u64 {
                let h = heap.allocate(ResourceClass::Buffer).unwrap();
                resolved.push(heap.resolve(h));
                heap.retire(h, epoch);
                heap.reclaim(epoch, 1);
            }
            resolved
        };
        // Two independent runs must produce identical resolution sequences.
        assert_eq!(build(), build());
    }
}
