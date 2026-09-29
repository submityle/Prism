//! Device-free byte-layout contract for the particle §9 `EmitterUpdate` compute
//! pass `@group(0)` bindings.
//!
//! The §9 `GPU` pipeline runs ten passes per frame (`EmitterUpdate`, `Spawn`,
//! `SimulationStages`, `Event Scatter`, `Compaction`, `Bounds`, `Cull`, `Sort`,
//! `Fill Draw Args`, `Render Draw`). This module publishes *what the first pass,
//! `EmitterUpdate`, binds* — the authoritative element stride, access mode,
//! element count and total byte size of every buffer in the
//! `particle_emitter_update.wesl` kernel's `@group(0)`. Exactly like the `hair/`
//! per-pass contracts ([`crate::hair::gpu_buffers`] and siblings) and the
//! sibling [`super::spawn_pass_buffers`], the sizing lives once here in the
//! zero-dependency crate so the render graph binds against a stable `ABI` and
//! never hand-computes strides next to the pipeline.
//!
//! The `EmitterUpdate` pass (design §9 step 1) advances each emitter's mutable
//! per-frame state — the fractional spawn accumulator, the emitter age, the
//! deterministic hash-`RNG` seed and an active flag — from its immutable
//! authored descriptor, then emits *spawn requests* describing how many
//! particles each emitter wants this frame. Terminology follows
//! [`super::emitter`], whose `CPU` reference turns an authored spawn rate and
//! bursts into a concrete new-particle count.
//!
//! ## Producer / consumer handshake
//! [`EmitterUpdateBuffer::SpawnRequests`] and
//! [`EmitterUpdateBuffer::SpawnRequestCounter`] are the *producer* side of the
//! handshake with the `Spawn` pass: this pass appends one request record per
//! emitter that wants to spawn and bumps the atomic counter, and the downstream
//! [`super::spawn_pass_buffers`] contract *consumes* those records to pop free
//! slots and initialise particles. The counter is a fixed single `u32` append
//! index — it never grows with the extent — while the request array spans the
//! frame's `spawn_request_capacity`.
//!
//! ## Orthogonality
//! - [`super::emitter`] owns the *semantics* of budgeting, distribution shapes
//!   and slot allocation; this module only names the buffers and their `std430`
//!   byte layout and never re-derives that behaviour.
//! - [`super::gpu_layout`] owns the shared stride constants, the access enum and
//!   the clamp-to-one byte-size rule; this module reuses them rather than
//!   redefining stride arithmetic.
//!
//! Everything is pure integer arithmetic: byte sizes clamp up to one element so
//! an empty frame still yields a valid non-empty `WebGPU` binding, the
//! multiplication saturates rather than overflowing, and nothing panics or
//! divides by zero.

use super::gpu_layout::{storage_bytes, ParticleBufferAccess, U32_STRIDE, VEC4_STRIDE};

/// `std430` array stride of the per-emitter `EmitterParams` descriptor: four
/// 16-byte-aligned `vec4<f32>` rows (authored spawn rate / lifetime / burst
/// schedule / distribution-shape params) = 64 bytes. Immutable this frame.
const EMITTER_PARAMS_STRIDE: usize = 4 * VEC4_STRIDE;

/// `std430` array stride of the per-emitter `EmitterState` accumulator: two
/// 16-byte-aligned `vec4` rows holding the fractional spawn accumulator, the
/// emitter age, the hash-`RNG` seed and an active flag = 32 bytes.
const EMITTER_STATE_STRIDE: usize = 2 * VEC4_STRIDE;

/// `std430` array stride of one appended `SpawnRequest`: a `vec4<u32>` packing
/// `{ emitter_index, count, seed, flags }` = 16 bytes.
const SPAWN_REQUEST_STRIDE: usize = VEC4_STRIDE;

/// Number of `u32` append counters the `EmitterUpdate` pass binds for the spawn
/// request queue: a single atomically-bumped write index.
///
/// This is a fixed layout constant independent of the frame extent.
pub const SPAWN_REQUEST_COUNTER_COUNT: usize = 1;

/// The element-count sources the `EmitterUpdate` pass sizes its buffers against
/// (design §9 step 1).
///
/// `emitter_count` is how many emitters the descriptor and state arrays hold
/// (also the dispatch domain), and `spawn_request_capacity` is the upper bound
/// on `SpawnRequest` records the producer side may append this frame for the
/// downstream `Spawn` pass to consume.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ParticleEmitterExtent {
    /// Number of emitters advanced this frame (the dispatch domain and the
    /// `EmitterParams` / `EmitterState` array length).
    pub emitter_count: u32,
    /// Upper bound on `SpawnRequest` records appended this frame.
    pub spawn_request_capacity: u32,
}

/// One storage buffer the `EmitterUpdate` kernel declares at `@group(0)`
/// (`particle_emitter_update.wesl`), in binding order `0..4` (design §9 step 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EmitterUpdateBuffer {
    /// `@binding(0)` per-emitter authored descriptor `array<EmitterParams>`
    /// (`read`): immutable spawn rate, lifetime, burst schedule and
    /// distribution-shape parameters. Sized by `emitter_count`.
    EmitterParams,
    /// `@binding(1)` per-emitter mutable accumulator `array<EmitterState>`
    /// (`read_write`): the fractional spawn accumulator, emitter age,
    /// hash-`RNG` seed and active flag, advanced in place. Sized by
    /// `emitter_count`.
    EmitterState,
    /// `@binding(2)` appended spawn-request queue `array<SpawnRequest>`
    /// (`read_write`): the producer side of the handshake with the `Spawn`
    /// pass, one `{ emitter_index, count, seed, flags }` record per emitter that
    /// wants to spawn. Sized by `spawn_request_capacity`; consumed downstream by
    /// [`super::spawn_pass_buffers`].
    SpawnRequests,
    /// `@binding(3)` spawn-request append counter `array<atomic<u32>>`
    /// (`read_write`): the single write index the pass bumps as it appends to
    /// [`EmitterUpdateBuffer::SpawnRequests`]. A fixed
    /// [`SPAWN_REQUEST_COUNTER_COUNT`]-element block that never scales with the
    /// extent; the downstream `Spawn` pass reads it to bound its dispatch.
    SpawnRequestCounter,
}

impl EmitterUpdateBuffer {
    /// Every `EmitterUpdate` buffer in `@binding` order. Its length matches the
    /// pass's `@group(0)` binding count.
    pub const ALL: [EmitterUpdateBuffer; 4] = [
        Self::EmitterParams,
        Self::EmitterState,
        Self::SpawnRequests,
        Self::SpawnRequestCounter,
    ];

    /// The `@group(0)` binding index in `particle_emitter_update.wesl`.
    #[must_use]
    pub fn binding(self) -> u32 {
        match self {
            Self::EmitterParams => 0,
            Self::EmitterState => 1,
            Self::SpawnRequests => 2,
            Self::SpawnRequestCounter => 3,
        }
    }

    /// Byte stride of one element, matching the `WESL` struct / scalar layout.
    #[must_use]
    pub fn stride(self) -> usize {
        match self {
            Self::EmitterParams => EMITTER_PARAMS_STRIDE,
            Self::EmitterState => EMITTER_STATE_STRIDE,
            Self::SpawnRequests => SPAWN_REQUEST_STRIDE,
            Self::SpawnRequestCounter => U32_STRIDE,
        }
    }

    /// Whether the kernel reads or read-writes this buffer.
    ///
    /// Only the authored `EmitterParams` descriptor is read-only; the state
    /// accumulator, the spawn-request queue and its counter are all mutated in
    /// place as emitters are advanced and requests appended.
    #[must_use]
    pub fn access(self) -> ParticleBufferAccess {
        match self {
            Self::EmitterParams => ParticleBufferAccess::Read,
            Self::EmitterState | Self::SpawnRequests | Self::SpawnRequestCounter => {
                ParticleBufferAccess::ReadWrite
            }
        }
    }

    /// Number of elements this buffer holds, derived from the frame extent.
    ///
    /// The descriptor and state arrays span `emitter_count`; the spawn-request
    /// queue spans `spawn_request_capacity`; the counter is a fixed
    /// [`SPAWN_REQUEST_COUNTER_COUNT`] regardless of extent.
    #[must_use]
    pub fn element_count(self, extent: ParticleEmitterExtent) -> usize {
        match self {
            Self::EmitterParams | Self::EmitterState => extent.emitter_count as usize,
            Self::SpawnRequests => extent.spawn_request_capacity as usize,
            Self::SpawnRequestCounter => SPAWN_REQUEST_COUNTER_COUNT,
        }
    }

    /// Total byte size of this buffer for the given extent, clamped up to one
    /// element (an empty frame still yields a valid non-empty `WebGPU` binding)
    /// and saturating on the multiply.
    #[must_use]
    pub fn byte_size(self, extent: ParticleEmitterExtent) -> usize {
        storage_bytes(self.stride(), self.element_count(extent))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_extent() -> ParticleEmitterExtent {
        ParticleEmitterExtent {
            emitter_count: 12,
            spawn_request_capacity: 256,
        }
    }

    #[test]
    fn all_covers_every_binding() {
        assert_eq!(
            EmitterUpdateBuffer::ALL.len(),
            EmitterUpdateBuffer::ALL
                .into_iter()
                .map(EmitterUpdateBuffer::binding)
                .count()
        );
        assert_eq!(EmitterUpdateBuffer::ALL.len(), 4);
    }

    #[test]
    fn bindings_are_dense_and_ordered() {
        for (index, buffer) in EmitterUpdateBuffer::ALL.into_iter().enumerate() {
            assert_eq!(buffer.binding() as usize, index);
        }
    }

    #[test]
    fn bindings_are_unique() {
        let mut seen = 0u32;
        for buffer in EmitterUpdateBuffer::ALL {
            let bit = 1u32 << buffer.binding();
            assert_eq!(seen & bit, 0, "duplicate binding");
            seen |= bit;
        }
    }

    #[test]
    fn strides_match_the_wesl_layout() {
        assert_eq!(EmitterUpdateBuffer::EmitterParams.stride(), 64);
        assert_eq!(EmitterUpdateBuffer::EmitterState.stride(), 32);
        assert_eq!(EmitterUpdateBuffer::SpawnRequests.stride(), 16);
        assert_eq!(EmitterUpdateBuffer::SpawnRequestCounter.stride(), 4);
    }

    #[test]
    fn access_modes_match_the_kernel() {
        assert_eq!(
            EmitterUpdateBuffer::EmitterParams.access(),
            ParticleBufferAccess::Read
        );
        assert_eq!(
            EmitterUpdateBuffer::EmitterState.access(),
            ParticleBufferAccess::ReadWrite
        );
        assert_eq!(
            EmitterUpdateBuffer::SpawnRequests.access(),
            ParticleBufferAccess::ReadWrite
        );
        assert_eq!(
            EmitterUpdateBuffer::SpawnRequestCounter.access(),
            ParticleBufferAccess::ReadWrite
        );
    }

    #[test]
    fn only_the_descriptor_is_read_only() {
        for buffer in EmitterUpdateBuffer::ALL {
            let writable = buffer.access().is_writable();
            assert_eq!(writable, buffer != EmitterUpdateBuffer::EmitterParams);
        }
    }

    #[test]
    fn element_counts_follow_the_extent() {
        let extent = sample_extent();
        assert_eq!(EmitterUpdateBuffer::EmitterParams.element_count(extent), 12);
        assert_eq!(EmitterUpdateBuffer::EmitterState.element_count(extent), 12);
        assert_eq!(
            EmitterUpdateBuffer::SpawnRequests.element_count(extent),
            256
        );
        assert_eq!(
            EmitterUpdateBuffer::SpawnRequestCounter.element_count(extent),
            SPAWN_REQUEST_COUNTER_COUNT
        );
    }

    #[test]
    fn counter_count_is_fixed_regardless_of_extent() {
        let big = ParticleEmitterExtent {
            emitter_count: 1_000_000,
            spawn_request_capacity: 999_999,
        };
        assert_eq!(
            EmitterUpdateBuffer::SpawnRequestCounter.element_count(big),
            SPAWN_REQUEST_COUNTER_COUNT
        );
        assert_eq!(
            EmitterUpdateBuffer::SpawnRequestCounter
                .element_count(ParticleEmitterExtent::default()),
            SPAWN_REQUEST_COUNTER_COUNT
        );
        // The fixed counter block is always a single u32.
        assert_eq!(
            EmitterUpdateBuffer::SpawnRequestCounter.byte_size(big),
            SPAWN_REQUEST_COUNTER_COUNT * U32_STRIDE
        );
    }

    #[test]
    fn byte_sizes_multiply_count_by_stride() {
        let extent = sample_extent();
        assert_eq!(
            EmitterUpdateBuffer::EmitterParams.byte_size(extent),
            12 * 64
        );
        assert_eq!(EmitterUpdateBuffer::EmitterState.byte_size(extent), 12 * 32);
        assert_eq!(
            EmitterUpdateBuffer::SpawnRequests.byte_size(extent),
            256 * 16
        );
        assert_eq!(
            EmitterUpdateBuffer::SpawnRequestCounter.byte_size(extent),
            SPAWN_REQUEST_COUNTER_COUNT * 4
        );
    }

    #[test]
    fn byte_size_scales_linearly_with_count() {
        let one = ParticleEmitterExtent {
            emitter_count: 1,
            spawn_request_capacity: 1,
        };
        let ten = ParticleEmitterExtent {
            emitter_count: 10,
            spawn_request_capacity: 10,
        };
        assert_eq!(
            EmitterUpdateBuffer::EmitterParams.byte_size(ten),
            10 * EmitterUpdateBuffer::EmitterParams.byte_size(one)
        );
        assert_eq!(
            EmitterUpdateBuffer::SpawnRequests.byte_size(ten),
            10 * EmitterUpdateBuffer::SpawnRequests.byte_size(one)
        );
    }

    #[test]
    fn empty_extent_clamps_every_buffer_to_one_element() {
        let extent = ParticleEmitterExtent::default();
        assert_eq!(
            EmitterUpdateBuffer::EmitterParams.byte_size(extent),
            EmitterUpdateBuffer::EmitterParams.stride()
        );
        assert_eq!(
            EmitterUpdateBuffer::EmitterState.byte_size(extent),
            EmitterUpdateBuffer::EmitterState.stride()
        );
        assert_eq!(
            EmitterUpdateBuffer::SpawnRequests.byte_size(extent),
            EmitterUpdateBuffer::SpawnRequests.stride()
        );
        // The counter is never empty: a fixed single-element block.
        assert_eq!(
            EmitterUpdateBuffer::SpawnRequestCounter.byte_size(extent),
            SPAWN_REQUEST_COUNTER_COUNT * EmitterUpdateBuffer::SpawnRequestCounter.stride()
        );
        // Every buffer keeps a valid non-empty WebGPU binding.
        for buffer in EmitterUpdateBuffer::ALL {
            assert!(buffer.byte_size(extent) >= buffer.stride());
        }
    }

    #[test]
    fn byte_size_saturates_instead_of_overflowing() {
        let extent = ParticleEmitterExtent {
            emitter_count: u32::MAX,
            spawn_request_capacity: u32::MAX,
        };
        // Never wraps below a single element.
        assert!(
            EmitterUpdateBuffer::SpawnRequests.byte_size(extent)
                >= EmitterUpdateBuffer::SpawnRequests.stride()
        );
        assert!(
            EmitterUpdateBuffer::EmitterParams.byte_size(extent)
                >= EmitterUpdateBuffer::EmitterParams.stride()
        );
    }
}
