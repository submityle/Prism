//! Cross-pass bind-group assembly and validation for the §9 GPU particle
//! pipeline.
//!
//! The per-pass byte-layout modules (`emitter_pass_buffers`, `spawn_pass_buffers`,
//! `sim_pass_buffers`, `event_pass_buffers`, `sort_pass_buffers`,
//! `draw_pass_buffers`) each own one pass's `@group(0)` binding order, strides,
//! and byte sizing in isolation. The orchestration layer
//! ([`super::frame_pipeline`]) owns the per-frame pass ordering and barrier
//! analysis. This module stitches the two together: it maps every
//! [`FramePass`] to the buffer contract it dispatches over, derives each pass's
//! sizing extent from a single whole-frame extent, sums a conservative
//! whole-frame `VRAM` upper bound, and validates the producer to consumer
//! handoffs where one pass's output buffer aliases a later pass's input.
//!
//! Like the pass modules, everything here is a `CPU`-verifiable contract; the
//! `GPU` bind-group objects and the `WESL` shaders are pending the `GPU`
//! backend. The layout mirrors production `GPU`-driven `VFX` engines (Unreal
//! `Niagara`, `Frostbite`'s FX stack) where a single per-emitter buffer set is
//! threaded through the compute front and the indirect draw.

use alloc::vec::Vec;

use super::draw_pass_buffers::{FillDrawArgsBuffer, ParticleDrawExtent, RenderDrawBuffer};
use super::emitter_pass_buffers::{EmitterUpdateBuffer, ParticleEmitterExtent};
use super::event_pass_buffers::{EventScatterBuffer, ParticleEventExtent};
use super::frame_pipeline::FramePass;
use super::sim_pass_buffers::{SimPassBuffer, SimPassExtent};
use super::sort_pass_buffers::{
    BoundsBuffer, CompactionBuffer, CullBuffer, SortBuffer, SortCullExtent,
};
use super::spawn_pass_buffers::{SpawnPassBuffer, SpawnPassExtent};

/// A single set of whole-frame scalars from which every pass's sizing extent is
/// derived, so the byte-layout modules and the orchestration layer agree on one
/// source of truth for element counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ParticleFrameExtent {
    /// Number of emitters advanced this frame.
    pub emitter_count: u32,
    /// Pooled particle capacity (the compile-time upper bound and the length of
    /// every per-particle Structure-of-Arrays pool).
    pub particle_capacity: u32,
    /// Particles spawned this frame (the spawn-index / spawn-request domain).
    pub spawn_count: u32,
    /// Live particles entering compaction, bounds, and sort.
    pub alive_count: u32,
    /// Particles surviving cull that become the indirect draw's instance count.
    pub visible_count: u32,
    /// Spatial-hash grid cell count (the cell-offset table length).
    pub grid_cell_count: u32,
    /// `XPBD` constraint count in the batch (constraint and `lambda` lengths).
    pub constraint_count: u32,
    /// Capacity of the raw source-event pool.
    pub event_source_capacity: u32,
    /// Number of event append channels.
    pub event_channel_count: u32,
    /// Capacity of the compacted scattered-event pool.
    pub event_scattered_capacity: u32,
    /// `radix` histogram bucket count for the sort (for example 256).
    pub radix_buckets: u32,
    /// Number of workgroups a reduction / sort pass dispatches.
    pub workgroup_count: u32,
}

impl ParticleFrameExtent {
    /// Derives the emitter-update pass extent.
    #[must_use]
    pub fn emitter_extent(self) -> ParticleEmitterExtent {
        ParticleEmitterExtent {
            emitter_count: self.emitter_count,
            spawn_request_capacity: self.spawn_count,
        }
    }

    /// Derives the spawn pass extent.
    #[must_use]
    pub fn spawn_extent(self) -> SpawnPassExtent {
        SpawnPassExtent {
            capacity: self.particle_capacity,
            spawn_count: self.spawn_count,
            emitter_count: self.emitter_count,
        }
    }

    /// Derives the simulation-stages pass extent.
    #[must_use]
    pub fn sim_extent(self) -> SimPassExtent {
        SimPassExtent {
            particle_capacity: self.particle_capacity,
            grid_cell_count: self.grid_cell_count,
            constraint_count: self.constraint_count,
        }
    }

    /// Derives the event-scatter pass extent.
    #[must_use]
    pub fn event_extent(self) -> ParticleEventExtent {
        ParticleEventExtent {
            source_event_capacity: self.event_source_capacity,
            channel_count: self.event_channel_count,
            scattered_capacity: self.event_scattered_capacity,
        }
    }

    /// Derives the shared compaction / bounds / cull / sort extent.
    #[must_use]
    pub fn sort_cull_extent(self) -> SortCullExtent {
        SortCullExtent {
            particle_count: self.alive_count,
            candidate_count: self.alive_count,
            radix_buckets: self.radix_buckets,
            workgroup_count: self.workgroup_count,
        }
    }

    /// Derives the draw pass extent (fill-draw-args and render-draw).
    #[must_use]
    pub fn draw_extent(self) -> ParticleDrawExtent {
        ParticleDrawExtent {
            capacity: self.particle_capacity,
            live_count: self.visible_count,
        }
    }
}

/// Returns the number of `@group(0)` bindings a pass declares.
#[must_use]
pub fn pass_binding_count(pass: FramePass) -> u32 {
    let count = match pass {
        FramePass::EmitterUpdate => EmitterUpdateBuffer::ALL.len(),
        FramePass::Spawn => SpawnPassBuffer::ALL.len(),
        FramePass::SimulationStages => SimPassBuffer::ALL.len(),
        FramePass::EventScatter => EventScatterBuffer::ALL.len(),
        FramePass::Compaction => CompactionBuffer::ALL.len(),
        FramePass::Bounds => BoundsBuffer::ALL.len(),
        FramePass::Cull => CullBuffer::ALL.len(),
        FramePass::Sort => SortBuffer::ALL.len(),
        FramePass::FillDrawArgs => FillDrawArgsBuffer::ALL.len(),
        FramePass::RenderDraw => RenderDrawBuffer::ALL.len(),
    };
    count as u32
}

/// Returns the total declared byte size of every buffer a pass binds, each
/// buffer clamped up to a valid non-empty `WebGPU` binding.
///
/// This counts each binding a pass declares, so buffers that persist and alias
/// across passes (the particle pool, the sorted-index list, the indirect draw
/// args) are counted once per pass that binds them. See
/// [`frame_upper_bound_bytes`] for the whole-frame implication.
#[must_use]
pub fn pass_declared_bytes(pass: FramePass, extent: ParticleFrameExtent) -> usize {
    match pass {
        FramePass::EmitterUpdate => {
            let e = extent.emitter_extent();
            EmitterUpdateBuffer::ALL
                .into_iter()
                .map(|b| b.byte_size(e))
                .fold(0usize, usize::saturating_add)
        }
        FramePass::Spawn => {
            let e = extent.spawn_extent();
            SpawnPassBuffer::ALL
                .into_iter()
                .map(|b| b.byte_size(e))
                .fold(0usize, usize::saturating_add)
        }
        FramePass::SimulationStages => {
            let e = extent.sim_extent();
            SimPassBuffer::ALL
                .into_iter()
                .map(|b| b.byte_size(e))
                .fold(0usize, usize::saturating_add)
        }
        FramePass::EventScatter => {
            let e = extent.event_extent();
            EventScatterBuffer::ALL
                .into_iter()
                .map(|b| b.byte_size(&e))
                .fold(0usize, usize::saturating_add)
        }
        FramePass::Compaction => {
            let e = extent.sort_cull_extent();
            CompactionBuffer::ALL
                .into_iter()
                .map(|b| b.byte_size(e))
                .fold(0usize, usize::saturating_add)
        }
        FramePass::Bounds => {
            let e = extent.sort_cull_extent();
            BoundsBuffer::ALL
                .into_iter()
                .map(|b| b.byte_size(e))
                .fold(0usize, usize::saturating_add)
        }
        FramePass::Cull => {
            let e = extent.sort_cull_extent();
            CullBuffer::ALL
                .into_iter()
                .map(|b| b.byte_size(e))
                .fold(0usize, usize::saturating_add)
        }
        FramePass::Sort => {
            let e = extent.sort_cull_extent();
            SortBuffer::ALL
                .into_iter()
                .map(|b| b.byte_size(e))
                .fold(0usize, usize::saturating_add)
        }
        FramePass::FillDrawArgs => {
            let e = extent.draw_extent();
            FillDrawArgsBuffer::ALL
                .into_iter()
                .map(|b| b.byte_size(&e))
                .fold(0usize, usize::saturating_add)
        }
        FramePass::RenderDraw => {
            let e = extent.draw_extent();
            RenderDrawBuffer::ALL
                .into_iter()
                .map(|b| b.byte_size(&e))
                .fold(0usize, usize::saturating_add)
        }
    }
}

/// Returns the total number of `@group(0)` bindings declared across the whole
/// per-frame pipeline (every pass counted once).
#[must_use]
pub fn frame_binding_count() -> u32 {
    FramePass::ALL
        .into_iter()
        .map(pass_binding_count)
        .fold(0u32, u32::saturating_add)
}

/// Returns a conservative whole-frame `VRAM` upper bound: the sum of every
/// pass's declared bytes.
///
/// This is an *upper bound*, not the true residency: persistent buffers (the
/// particle pool, the sorted-index list, the indirect draw args) are bound by
/// more than one pass and are therefore counted more than once here. A real
/// allocator aliases those and would report less; this sum is safe to budget
/// against because it never under-counts.
#[must_use]
pub fn frame_upper_bound_bytes(extent: ParticleFrameExtent) -> usize {
    FramePass::ALL
        .into_iter()
        .map(|pass| pass_declared_bytes(pass, extent))
        .fold(0usize, usize::saturating_add)
}

/// A logical buffer that one pass produces and a later pass consumes, so the
/// two passes must agree on its element stride.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HandoffResource {
    /// The per-particle position pool: written by the simulation stages,
    /// reduced by the bounds pass.
    PoolPositions,
    /// The sorted visible-instance index list: written by the fill-draw-args
    /// pass, read by the indirect render draw.
    SortedIndices,
    /// The indirect draw argument buffer: written by the fill-draw-args pass,
    /// read (as the indirect source) by the render draw.
    IndirectDrawArgs,
}

/// A resolved producer to consumer handoff, carrying both endpoints' declared
/// element strides for consistency checking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Handoff {
    /// The logical buffer flowing across the two passes.
    pub resource: HandoffResource,
    /// The pass that writes the buffer.
    pub producer: FramePass,
    /// The pass that reads the buffer.
    pub consumer: FramePass,
    /// The producer's declared element stride, in bytes.
    pub producer_stride: usize,
    /// The consumer's declared element stride, in bytes.
    pub consumer_stride: usize,
}

impl Handoff {
    /// Whether the producer and consumer agree on the element stride. A stride
    /// mismatch means the two passes disagree on the record layout and the
    /// aliased binding would misread the data.
    #[must_use]
    pub fn is_consistent(self) -> bool {
        self.producer_stride == self.consumer_stride
    }
}

/// The number of cross-pass buffer handoffs the pipeline validates.
pub const HANDOFF_COUNT: usize = 3;

/// Resolves every cross-pass handoff, reading each endpoint's stride from the
/// per-pass byte-layout contracts.
#[must_use]
pub fn describe_handoffs() -> Vec<Handoff> {
    let mut out = Vec::with_capacity(HANDOFF_COUNT);
    out.push(Handoff {
        resource: HandoffResource::PoolPositions,
        producer: FramePass::SimulationStages,
        consumer: FramePass::Bounds,
        producer_stride: SimPassBuffer::Positions.stride(),
        consumer_stride: BoundsBuffer::Positions.stride(),
    });
    out.push(Handoff {
        resource: HandoffResource::SortedIndices,
        producer: FramePass::FillDrawArgs,
        consumer: FramePass::RenderDraw,
        producer_stride: FillDrawArgsBuffer::SortedIndices.stride(),
        consumer_stride: RenderDrawBuffer::SortedIndices.stride(),
    });
    out.push(Handoff {
        resource: HandoffResource::IndirectDrawArgs,
        producer: FramePass::FillDrawArgs,
        consumer: FramePass::RenderDraw,
        producer_stride: FillDrawArgsBuffer::DrawArgs.stride(),
        consumer_stride: RenderDrawBuffer::DrawArgs.stride(),
    });
    out
}

/// The first inconsistent handoff, reported when [`validate_handoffs`] fails.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HandoffError {
    /// The buffer whose producer and consumer disagree on stride.
    pub resource: HandoffResource,
    /// The producer's declared stride.
    pub producer_stride: usize,
    /// The consumer's declared stride.
    pub consumer_stride: usize,
}

/// Validates that every cross-pass handoff's producer and consumer agree on the
/// element stride, returning the first mismatch found.
///
/// # Errors
///
/// Returns a [`HandoffError`] describing the first handoff whose producer and
/// consumer strides disagree.
pub fn validate_handoffs() -> Result<(), HandoffError> {
    for handoff in describe_handoffs() {
        if !handoff.is_consistent() {
            return Err(HandoffError {
                resource: handoff.resource,
                producer_stride: handoff.producer_stride,
                consumer_stride: handoff.consumer_stride,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_extent() -> ParticleFrameExtent {
        ParticleFrameExtent {
            emitter_count: 4,
            particle_capacity: 4096,
            spawn_count: 128,
            alive_count: 2048,
            visible_count: 1024,
            grid_cell_count: 512,
            constraint_count: 256,
            event_source_capacity: 512,
            event_channel_count: 8,
            event_scattered_capacity: 1024,
            radix_buckets: 256,
            workgroup_count: 32,
        }
    }

    #[test]
    fn binding_counts_match_each_pass_contract() {
        assert_eq!(
            pass_binding_count(FramePass::EmitterUpdate),
            EmitterUpdateBuffer::ALL.len() as u32
        );
        assert_eq!(
            pass_binding_count(FramePass::Spawn),
            SpawnPassBuffer::ALL.len() as u32
        );
        assert_eq!(
            pass_binding_count(FramePass::SimulationStages),
            SimPassBuffer::ALL.len() as u32
        );
        assert_eq!(
            pass_binding_count(FramePass::EventScatter),
            EventScatterBuffer::ALL.len() as u32
        );
        assert_eq!(
            pass_binding_count(FramePass::Compaction),
            CompactionBuffer::ALL.len() as u32
        );
        assert_eq!(
            pass_binding_count(FramePass::Bounds),
            BoundsBuffer::ALL.len() as u32
        );
        assert_eq!(
            pass_binding_count(FramePass::Cull),
            CullBuffer::ALL.len() as u32
        );
        assert_eq!(
            pass_binding_count(FramePass::Sort),
            SortBuffer::ALL.len() as u32
        );
        assert_eq!(
            pass_binding_count(FramePass::FillDrawArgs),
            FillDrawArgsBuffer::ALL.len() as u32
        );
        assert_eq!(
            pass_binding_count(FramePass::RenderDraw),
            RenderDrawBuffer::ALL.len() as u32
        );
    }

    #[test]
    fn frame_binding_count_is_the_sum_over_passes() {
        let expected: u32 = FramePass::ALL
            .into_iter()
            .map(pass_binding_count)
            .fold(0u32, u32::saturating_add);
        assert_eq!(frame_binding_count(), expected);
        assert!(frame_binding_count() > 0);
    }

    #[test]
    fn every_pass_declares_some_bytes_for_a_nonzero_extent() {
        let extent = sample_extent();
        for pass in FramePass::ALL {
            assert!(
                pass_declared_bytes(pass, extent) > 0,
                "pass {pass:?} declared zero bytes"
            );
        }
    }

    #[test]
    fn frame_upper_bound_is_the_sum_of_pass_bytes() {
        let extent = sample_extent();
        let expected: usize = FramePass::ALL
            .into_iter()
            .map(|pass| pass_declared_bytes(pass, extent))
            .fold(0usize, usize::saturating_add);
        assert_eq!(frame_upper_bound_bytes(extent), expected);
    }

    #[test]
    fn upper_bound_is_monotonic_in_capacity() {
        let small = sample_extent();
        let mut large = small;
        large.particle_capacity = small.particle_capacity.saturating_mul(2);
        assert!(frame_upper_bound_bytes(large) >= frame_upper_bound_bytes(small));
    }

    #[test]
    fn zero_extent_still_clamps_to_nonzero_bindings() {
        let extent = ParticleFrameExtent::default();
        // Every buffer clamps up to one element, so a fully-zero extent still
        // reports a positive whole-frame footprint and never panics.
        assert!(frame_upper_bound_bytes(extent) > 0);
    }

    #[test]
    fn extent_derivers_thread_the_scalars_through() {
        let extent = sample_extent();
        assert_eq!(extent.emitter_extent().emitter_count, 4);
        assert_eq!(extent.emitter_extent().spawn_request_capacity, 128);
        assert_eq!(extent.spawn_extent().capacity, 4096);
        assert_eq!(extent.sim_extent().grid_cell_count, 512);
        assert_eq!(extent.sim_extent().constraint_count, 256);
        assert_eq!(extent.event_extent().channel_count, 8);
        assert_eq!(extent.sort_cull_extent().particle_count, 2048);
        assert_eq!(extent.sort_cull_extent().radix_buckets, 256);
        assert_eq!(extent.draw_extent().live_count, 1024);
    }

    #[test]
    fn real_handoffs_are_stride_consistent() {
        assert_eq!(validate_handoffs(), Ok(()));
        let handoffs = describe_handoffs();
        assert_eq!(handoffs.len(), HANDOFF_COUNT);
        for handoff in handoffs {
            assert!(
                handoff.is_consistent(),
                "handoff {:?} strides disagree: {} vs {}",
                handoff.resource,
                handoff.producer_stride,
                handoff.consumer_stride
            );
        }
    }

    #[test]
    fn handoff_endpoints_are_the_expected_passes() {
        let handoffs = describe_handoffs();
        let positions = handoffs
            .iter()
            .find(|h| h.resource == HandoffResource::PoolPositions)
            .expect("positions handoff present");
        assert_eq!(positions.producer, FramePass::SimulationStages);
        assert_eq!(positions.consumer, FramePass::Bounds);

        let sorted = handoffs
            .iter()
            .find(|h| h.resource == HandoffResource::SortedIndices)
            .expect("sorted-indices handoff present");
        assert_eq!(sorted.producer, FramePass::FillDrawArgs);
        assert_eq!(sorted.consumer, FramePass::RenderDraw);
    }

    #[test]
    fn inconsistent_handoff_is_detected() {
        let bad = Handoff {
            resource: HandoffResource::PoolPositions,
            producer: FramePass::SimulationStages,
            consumer: FramePass::Bounds,
            producer_stride: 16,
            consumer_stride: 8,
        };
        assert!(!bad.is_consistent());

        let good = Handoff {
            consumer_stride: 16,
            ..bad
        };
        assert!(good.is_consistent());
    }
}
