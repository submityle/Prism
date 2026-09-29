//! Device-free byte-layout contract for the particle §9 `Event Scatter` compute
//! pass `@group(0)` bindings.
//!
//! The §9 `GPU` pipeline runs its passes per emitter per frame (`EmitterUpdate`,
//! `Spawn`, `SimulationStages`, `Event Scatter`, `Compaction`, `Bounds`,
//! `Cull`, `Sort`, `Fill Draw Args`, `Render Draw`). This module publishes
//! *what the `Event Scatter` pass binds* — the authoritative element stride,
//! access mode, element count and total byte size of every storage buffer in
//! the `particle_event_scatter.wesl` kernel's `@group(0)`. Exactly like the
//! `hair/` per-pass contracts ([`crate::hair::gpu_buffers`] and siblings) and
//! the sibling [`super::spawn_pass_buffers`], the sizing lives once here in the
//! zero-dependency crate so the render graph binds against a stable `ABI` and
//! never hand-computes strides next to the pipeline.
//!
//! ## What the pass does
//! During the simulation stages each particle can raise lightweight *events*
//! (spawn / death / collision / condition — the terms mirror [`super::events`],
//! which this module deliberately does not import, only aligns naming with).
//! Those raw events land unsorted in one `SourceEvents` record buffer. The
//! `Event Scatter` pass reads each source record, looks up its channel's base
//! offset in the `ChannelOffsets` prefix-sum table, atomically bumps that
//! channel's slot in the `EventCounters` block, and writes the record into the
//! compacted `ScatteredEvents` output at `offset + reserved_slot`. The result is
//! a per-channel contiguous run of events that sub-emitters (the `PerEvent`
//! stage) and `GPU` readback consumers can iterate without a second gather.
//!
//! ## Atomic append & aliasing semantics
//! - `EventCounters` is an `array<atomic<u32>>` with one counter per channel;
//!   the pass appends via `atomicAdd`, so every write reserves a unique
//!   destination slot without a host round-trip and back-pressure is a simple
//!   compare against the channel's capacity.
//! - `ScatteredEvents` is a *distinct* allocation from `SourceEvents`. Although
//!   the two share the same 32-byte record stride, the scatter is a copy from
//!   the source pool into the compacted output; the output never aliases the
//!   input (unlike the in-place hair `Wind`/`SdfCollision` passes), so a source
//!   record and its scattered copy can be read independently in the same frame.
//! - `SourceEvents` and `ChannelOffsets` are read-only inputs for this pass.
//!
//! Everything is pure integer arithmetic: byte sizes clamp up to one element so
//! an empty pool still yields a valid non-empty `WebGPU` binding, the
//! multiplication saturates rather than overflowing, and nothing panics or
//! divides by zero.

use crate::particle::gpu_layout::{storage_bytes, ParticleBufferAccess, U32_STRIDE, VEC4_STRIDE};

/// `std430` array stride of one packed event record shared by `SourceEvents`
/// and `ScatteredEvents`: `payload: vec4<f32>` (16) + `kind: u32` (4) +
/// `particle_index: u32` (4) + two `u32` pads (8) = 32 bytes.
const EVENT_RECORD_STRIDE: usize = VEC4_STRIDE + 4 * U32_STRIDE;

/// The element-count sources the `Event Scatter` pass sizes its buffers against
/// (design §9 `Event Scatter`, §14).
///
/// `source_event_capacity` bounds the raw per-frame event pool, `channel_count`
/// is how many append channels the counter and offset tables cover, and
/// `scattered_capacity` bounds the compacted per-channel output pool.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ParticleEventExtent {
    /// Capacity of the raw `SourceEvents` record pool (events raised per frame).
    pub source_event_capacity: u32,
    /// Number of append channels (`EventCounters` / `ChannelOffsets` length).
    pub channel_count: u32,
    /// Capacity of the compacted `ScatteredEvents` output pool.
    pub scattered_capacity: u32,
}

/// One storage buffer bound by the `Event Scatter` kernel
/// (`particle_event_scatter.wesl` `@group(0)`), in binding order `0..4`
/// (design §9 `Event Scatter`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EventScatterBuffer {
    /// `@binding(0)` per-channel append counters `array<atomic<u32>>`
    /// (`read_write`): each channel's live scattered-event count, bumped with
    /// `atomicAdd` to reserve a unique destination slot.
    EventCounters,
    /// `@binding(1)` raw packed event records `array<EventRecord>` (`read`): the
    /// unsorted events the simulation stages raised this frame (`payload`,
    /// `kind`, `particle_index`).
    SourceEvents,
    /// `@binding(2)` compacted per-channel output `array<EventRecord>`
    /// (`read_write`): the scattered records written at
    /// `ChannelOffsets[channel] + reserved_slot`. A distinct allocation that
    /// never aliases `SourceEvents`.
    ScatteredEvents,
    /// `@binding(3)` per-channel prefix-sum base offsets `array<u32>` (`read`):
    /// where each channel's run begins inside `ScatteredEvents`.
    ChannelOffsets,
}

impl EventScatterBuffer {
    /// Every `Event Scatter` buffer in `@binding` order. Its length matches the
    /// pass's `@group(0)` binding count.
    pub const ALL: [EventScatterBuffer; 4] = [
        Self::EventCounters,
        Self::SourceEvents,
        Self::ScatteredEvents,
        Self::ChannelOffsets,
    ];

    /// The `@group(0)` binding index in `particle_event_scatter.wesl`.
    #[must_use]
    pub fn binding(self) -> u32 {
        match self {
            Self::EventCounters => 0,
            Self::SourceEvents => 1,
            Self::ScatteredEvents => 2,
            Self::ChannelOffsets => 3,
        }
    }

    /// Byte stride of one element, matching the `WESL` struct / scalar layout.
    #[must_use]
    pub fn stride(self) -> usize {
        match self {
            // array<atomic<u32>> counters / array<u32> offsets — 4-byte scalars.
            Self::EventCounters | Self::ChannelOffsets => U32_STRIDE,
            // array<EventRecord> — the shared 32-byte packed record.
            Self::SourceEvents | Self::ScatteredEvents => EVENT_RECORD_STRIDE,
        }
    }

    /// Whether the kernel reads or read-writes this buffer.
    ///
    /// The source events and the channel offset table are read-only inputs; the
    /// atomic counters and the compacted output are mutated in place as records
    /// are appended.
    #[must_use]
    pub fn access(self) -> ParticleBufferAccess {
        match self {
            Self::SourceEvents | Self::ChannelOffsets => ParticleBufferAccess::Read,
            Self::EventCounters | Self::ScatteredEvents => ParticleBufferAccess::ReadWrite,
        }
    }

    /// Number of elements this buffer holds, derived from the pass extent.
    ///
    /// The counter and offset tables have one entry per channel; the source and
    /// scattered pools span their respective capacities.
    #[must_use]
    pub fn element_count(self, extent: &ParticleEventExtent) -> u32 {
        match self {
            Self::EventCounters | Self::ChannelOffsets => extent.channel_count,
            Self::SourceEvents => extent.source_event_capacity,
            Self::ScatteredEvents => extent.scattered_capacity,
        }
    }

    /// Total byte size of this buffer for the given extent, clamped up to one
    /// element (an empty pool still yields a valid non-empty `WebGPU` binding)
    /// and saturating on the multiply.
    #[must_use]
    pub fn byte_size(self, extent: &ParticleEventExtent) -> usize {
        storage_bytes(self.stride(), self.element_count(extent) as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_extent() -> ParticleEventExtent {
        ParticleEventExtent {
            source_event_capacity: 2048,
            channel_count: 8,
            scattered_capacity: 1024,
        }
    }

    #[test]
    fn all_matches_the_binding_count() {
        assert_eq!(EventScatterBuffer::ALL.len(), 4);
    }

    #[test]
    fn bindings_are_dense_and_ordered() {
        for (index, buffer) in EventScatterBuffer::ALL.into_iter().enumerate() {
            assert_eq!(buffer.binding() as usize, index);
        }
    }

    #[test]
    fn strides_match_the_wesl_layout() {
        assert_eq!(EventScatterBuffer::EventCounters.stride(), 4);
        assert_eq!(EventScatterBuffer::SourceEvents.stride(), 32);
        assert_eq!(EventScatterBuffer::ScatteredEvents.stride(), 32);
        assert_eq!(EventScatterBuffer::ChannelOffsets.stride(), 4);
    }

    #[test]
    fn source_and_scattered_share_the_record_stride() {
        assert_eq!(
            EventScatterBuffer::SourceEvents.stride(),
            EventScatterBuffer::ScatteredEvents.stride()
        );
        assert_eq!(EVENT_RECORD_STRIDE, 32);
    }

    #[test]
    fn access_modes_match_the_kernel() {
        assert_eq!(
            EventScatterBuffer::EventCounters.access(),
            ParticleBufferAccess::ReadWrite
        );
        assert_eq!(
            EventScatterBuffer::SourceEvents.access(),
            ParticleBufferAccess::Read
        );
        assert_eq!(
            EventScatterBuffer::ScatteredEvents.access(),
            ParticleBufferAccess::ReadWrite
        );
        assert_eq!(
            EventScatterBuffer::ChannelOffsets.access(),
            ParticleBufferAccess::Read
        );
    }

    #[test]
    fn only_the_inputs_are_non_writable() {
        for buffer in EventScatterBuffer::ALL {
            let writable = buffer.access().is_writable();
            let is_output = matches!(
                buffer,
                EventScatterBuffer::EventCounters | EventScatterBuffer::ScatteredEvents
            );
            assert_eq!(writable, is_output);
        }
    }

    #[test]
    fn element_counts_follow_the_extent() {
        let extent = sample_extent();
        assert_eq!(EventScatterBuffer::EventCounters.element_count(&extent), 8);
        assert_eq!(
            EventScatterBuffer::SourceEvents.element_count(&extent),
            2048
        );
        assert_eq!(
            EventScatterBuffer::ScatteredEvents.element_count(&extent),
            1024
        );
        assert_eq!(EventScatterBuffer::ChannelOffsets.element_count(&extent), 8);
    }

    #[test]
    fn byte_sizes_multiply_count_by_stride() {
        let extent = sample_extent();
        assert_eq!(EventScatterBuffer::EventCounters.byte_size(&extent), 8 * 4);
        assert_eq!(
            EventScatterBuffer::SourceEvents.byte_size(&extent),
            2048 * 32
        );
        assert_eq!(
            EventScatterBuffer::ScatteredEvents.byte_size(&extent),
            1024 * 32
        );
        assert_eq!(EventScatterBuffer::ChannelOffsets.byte_size(&extent), 8 * 4);
    }

    #[test]
    fn byte_size_scales_linearly_with_count() {
        let one = ParticleEventExtent {
            source_event_capacity: 1,
            channel_count: 1,
            scattered_capacity: 1,
        };
        let ten = ParticleEventExtent {
            source_event_capacity: 10,
            channel_count: 10,
            scattered_capacity: 10,
        };
        assert_eq!(
            EventScatterBuffer::SourceEvents.byte_size(&ten),
            10 * EventScatterBuffer::SourceEvents.byte_size(&one)
        );
        assert_eq!(
            EventScatterBuffer::ScatteredEvents.byte_size(&ten),
            10 * EventScatterBuffer::ScatteredEvents.byte_size(&one)
        );
        assert_eq!(
            EventScatterBuffer::EventCounters.byte_size(&ten),
            10 * EventScatterBuffer::EventCounters.byte_size(&one)
        );
    }

    #[test]
    fn empty_extent_clamps_every_buffer_to_one_element() {
        let extent = ParticleEventExtent::default();
        for buffer in EventScatterBuffer::ALL {
            assert_eq!(buffer.byte_size(&extent), buffer.stride());
        }
    }

    #[test]
    fn byte_size_saturates_instead_of_overflowing() {
        let extent = ParticleEventExtent {
            source_event_capacity: u32::MAX,
            channel_count: u32::MAX,
            scattered_capacity: u32::MAX,
        };
        for buffer in EventScatterBuffer::ALL {
            assert!(buffer.byte_size(&extent) >= buffer.stride());
        }
    }
}
