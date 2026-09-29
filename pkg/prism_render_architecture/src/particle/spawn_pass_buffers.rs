//! Device-free byte-layout contract for the particle §9 `Spawn`/`Emit` compute
//! pass `@group(0)` bindings.
//!
//! The §9 `GPU` pipeline runs ten passes per emitter per frame (`EmitterUpdate`,
//! `Spawn`, `SimulationStages`, `Event Scatter`, `Compaction`, `Bounds`,
//! `Cull`, `Sort`, `Fill Draw Args`, `Render Draw`). This module publishes
//! *what the `Spawn`/`Emit` pass binds* — the authoritative element stride,
//! access mode, element count and total byte size of every buffer in the
//! `particle_spawn.wesl` kernel's `@group(0)`. Exactly like the `hair/`
//! per-pass contracts ([`crate::hair::gpu_buffers`] and siblings), the sizing
//! lives once here in the zero-dependency crate so the render graph binds
//! against a stable `ABI` and never hand-computes strides next to the pipeline.
//!
//! The `Spawn` pass (design §9 step 2) dispatches `spawn_count` threads that pop
//! free slots from the pool's `free_list` stack, bump the atomic `counters`,
//! and write each new particle's initial `position`/`velocity` into the
//! Structure-of-Arrays attribute pool — the `GPU`-driven emission that keeps the
//! particle count entirely device-side (mirroring `Niagara`/`Frostbite` stacks
//! with zero readback).
//!
//! ## Orthogonality
//! - [`super::attributes`] owns *which* per-particle channels the `SoA` pool
//!   has and how wide each is; this module never re-derives attribute widths.
//! - [`super::pool`] owns the runtime free-list/counter *semantics*; this module
//!   only names the buffers and their `std430`/`std140` byte layout.
//! - [`super::gpu_layout`] owns the shared stride constants, the access enum and
//!   the clamp-to-one byte-size rule; this module reuses them rather than
//!   redefining stride arithmetic.
//!
//! Everything is pure integer arithmetic: byte sizes clamp up to one element so
//! an empty pool still yields a valid non-empty `WebGPU` binding, the
//! multiplication saturates rather than overflowing, and nothing panics or
//! divides by zero.

use super::gpu_layout::{storage_bytes, ParticleBufferAccess, U32_STRIDE, VEC4_STRIDE};

/// `std140` array stride of the per-emitter `SpawnParams` uniform block: two
/// 16-byte-aligned `vec4` rows (`spawn_count`/`first_slot`/`base_seed`/`flags`
/// + reserved animation row), the natural 32-byte stride for a `WebGPU` uniform.
pub const SPAWN_PARAMS_STRIDE: usize = 2 * VEC4_STRIDE;

/// Number of atomic `u32` counters the `Spawn` pass binds: `alive_count`,
/// `spawn_count`, `dead_count` and the `free_list` stack top (design §5.2).
///
/// This is a fixed layout constant independent of the pool capacity.
pub const SPAWN_COUNTER_COUNT: usize = 4;

/// Whether a `@group(0)` binding is a `storage` buffer or a `uniform` block.
///
/// The `Spawn` pass mixes both: the free-list, counters, spawn indices and
/// attribute pools are `std430` storage buffers, while the per-emitter spawn
/// parameters ride a `std140` uniform block that is *not* part of the pool's
/// storage footprint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpawnBindingKind {
    /// `var<storage, ...>` — a `std430` storage buffer sized against the pool.
    Storage,
    /// `var<uniform>` — a `std140` uniform block, excluded from storage totals.
    Uniform,
}

/// The element-count sources the `Spawn` pass sizes its buffers against
/// (design §9 step 2, §11).
///
/// `capacity` is the emitter's compile-time pool upper bound (the free-list and
/// attribute pools span it), `spawn_count` is this dispatch's new-particle count
/// (the spawn-index queue length), and `emitter_count` is how many per-emitter
/// `SpawnParams` blocks the uniform holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SpawnPassExtent {
    /// Emitter pool capacity — the compile-time particle upper bound.
    pub capacity: u32,
    /// Number of particles spawned this dispatch (the spawn-index queue length).
    pub spawn_count: u32,
    /// Number of per-emitter `SpawnParams` blocks in the uniform.
    pub emitter_count: u32,
}

/// One binding the `Spawn`/`Emit` kernel declares at `@group(0)`
/// (`particle_spawn.wesl`), in binding order `0..6` (design §9 step 2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpawnPassBuffer {
    /// `@binding(0)` free-slot stack `array<u32>` (`read_write`): the pass pops
    /// recycled slot indices from this `free_list` stack (design §5.2, §11).
    FreeList,
    /// `@binding(1)` pool counters `array<atomic<u32>>` (`read_write`):
    /// `alive_count` / `spawn_count` / `dead_count` / free-list top, bumped
    /// atomically as slots are claimed (design §5.2).
    Counters,
    /// `@binding(2)` spawn-index queue `array<u32>` (`read_write`): the list of
    /// freshly claimed slot indices this dispatch initialises.
    SpawnIndices,
    /// `@binding(3)` per-emitter `SpawnParams` `uniform` block: spawn count,
    /// base seed and emitter animation params. A `std140` uniform, *not* part of
    /// the pool storage footprint.
    SpawnParams,
    /// `@binding(4)` target `position` pool `array<vec4<f32>>` (`read_write`):
    /// the `SoA` attribute channel new particles' initial positions are written
    /// into (design §5).
    PositionPool,
    /// `@binding(5)` target `velocity` pool `array<vec4<f32>>` (`read_write`):
    /// the `SoA` attribute channel new particles' initial velocities are written
    /// into (design §5).
    VelocityPool,
}

impl SpawnPassBuffer {
    /// Every `Spawn`-pass binding in `@binding` order; `ALL[i].binding() == i`.
    pub const ALL: [SpawnPassBuffer; 6] = [
        Self::FreeList,
        Self::Counters,
        Self::SpawnIndices,
        Self::SpawnParams,
        Self::PositionPool,
        Self::VelocityPool,
    ];

    /// The `@group(0)` binding index this buffer occupies in
    /// `particle_spawn.wesl`.
    #[must_use]
    pub fn binding(self) -> u32 {
        match self {
            Self::FreeList => 0,
            Self::Counters => 1,
            Self::SpawnIndices => 2,
            Self::SpawnParams => 3,
            Self::PositionPool => 4,
            Self::VelocityPool => 5,
        }
    }

    /// Byte stride of one element, matching the `WESL` struct / scalar layout.
    #[must_use]
    pub fn stride(self) -> usize {
        match self {
            // array<u32> / array<atomic<u32>> — 4-byte scalars.
            Self::FreeList | Self::Counters | Self::SpawnIndices => U32_STRIDE,
            // std140 per-emitter uniform block.
            Self::SpawnParams => SPAWN_PARAMS_STRIDE,
            // array<vec4<f32>> attribute pools.
            Self::PositionPool | Self::VelocityPool => VEC4_STRIDE,
        }
    }

    /// Whether this binding is a `std430` storage buffer or a `std140` uniform.
    #[must_use]
    pub fn kind(self) -> SpawnBindingKind {
        match self {
            Self::SpawnParams => SpawnBindingKind::Uniform,
            Self::FreeList
            | Self::Counters
            | Self::SpawnIndices
            | Self::PositionPool
            | Self::VelocityPool => SpawnBindingKind::Storage,
        }
    }

    /// Whether the kernel reads or read-writes this binding.
    ///
    /// The uniform `SpawnParams` is read-only; every storage buffer is mutated
    /// in place (slots popped, counters bumped, initial attributes written).
    #[must_use]
    pub fn access(self) -> ParticleBufferAccess {
        match self {
            Self::SpawnParams => ParticleBufferAccess::Read,
            Self::FreeList
            | Self::Counters
            | Self::SpawnIndices
            | Self::PositionPool
            | Self::VelocityPool => ParticleBufferAccess::ReadWrite,
        }
    }

    /// Number of elements this binding holds, derived from the pass extent.
    ///
    /// The free-list stack and both attribute pools span the whole `capacity`;
    /// the counter block is a fixed [`SPAWN_COUNTER_COUNT`]; the spawn-index
    /// queue holds this dispatch's `spawn_count`; the uniform holds one
    /// `SpawnParams` block per emitter.
    #[must_use]
    pub fn element_count(self, extent: SpawnPassExtent) -> usize {
        match self {
            Self::FreeList | Self::PositionPool | Self::VelocityPool => extent.capacity as usize,
            Self::Counters => SPAWN_COUNTER_COUNT,
            Self::SpawnIndices => extent.spawn_count as usize,
            Self::SpawnParams => extent.emitter_count as usize,
        }
    }

    /// Total byte size of this binding for the given extent, clamped up to one
    /// element (an empty pool still yields a valid non-empty `WebGPU` binding)
    /// and saturating on the multiply.
    #[must_use]
    pub fn byte_size(self, extent: SpawnPassExtent) -> usize {
        storage_bytes(self.stride(), self.element_count(extent))
    }
}

/// Total bytes of the `Spawn` pass's `std430` *storage* buffers for the given
/// extent, excluding the `std140` [`SpawnPassBuffer::SpawnParams`] uniform.
///
/// This is the pool-side allocation the render graph reserves for the pass; the
/// uniform block is owned and sized separately by the host uniform ring.
#[must_use]
pub fn total_storage_bytes(extent: SpawnPassExtent) -> usize {
    let mut total = 0usize;
    for buffer in SpawnPassBuffer::ALL {
        if buffer.kind() == SpawnBindingKind::Storage {
            total = total.saturating_add(buffer.byte_size(extent));
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_extent() -> SpawnPassExtent {
        SpawnPassExtent {
            capacity: 4096,
            spawn_count: 256,
            emitter_count: 3,
        }
    }

    #[test]
    fn bindings_are_dense_and_ordered() {
        for (index, buffer) in SpawnPassBuffer::ALL.into_iter().enumerate() {
            assert_eq!(buffer.binding() as usize, index);
        }
    }

    #[test]
    fn strides_match_the_wesl_layout() {
        assert_eq!(SpawnPassBuffer::FreeList.stride(), 4);
        assert_eq!(SpawnPassBuffer::Counters.stride(), 4);
        assert_eq!(SpawnPassBuffer::SpawnIndices.stride(), 4);
        assert_eq!(SpawnPassBuffer::SpawnParams.stride(), 32);
        assert_eq!(SpawnPassBuffer::PositionPool.stride(), 16);
        assert_eq!(SpawnPassBuffer::VelocityPool.stride(), 16);
    }

    #[test]
    fn kinds_flag_only_the_uniform() {
        assert_eq!(
            SpawnPassBuffer::SpawnParams.kind(),
            SpawnBindingKind::Uniform
        );
        for buffer in SpawnPassBuffer::ALL {
            if buffer != SpawnPassBuffer::SpawnParams {
                assert_eq!(buffer.kind(), SpawnBindingKind::Storage);
            }
        }
    }

    #[test]
    fn access_modes_match_the_kernel() {
        assert_eq!(
            SpawnPassBuffer::FreeList.access(),
            ParticleBufferAccess::ReadWrite
        );
        assert_eq!(
            SpawnPassBuffer::Counters.access(),
            ParticleBufferAccess::ReadWrite
        );
        assert_eq!(
            SpawnPassBuffer::SpawnIndices.access(),
            ParticleBufferAccess::ReadWrite
        );
        assert_eq!(
            SpawnPassBuffer::SpawnParams.access(),
            ParticleBufferAccess::Read
        );
        assert_eq!(
            SpawnPassBuffer::PositionPool.access(),
            ParticleBufferAccess::ReadWrite
        );
        assert_eq!(
            SpawnPassBuffer::VelocityPool.access(),
            ParticleBufferAccess::ReadWrite
        );
    }

    #[test]
    fn only_the_uniform_is_non_writable() {
        for buffer in SpawnPassBuffer::ALL {
            let writable = buffer.access().is_writable();
            assert_eq!(writable, buffer.kind() == SpawnBindingKind::Storage);
        }
    }

    #[test]
    fn element_counts_follow_the_extent() {
        let extent = sample_extent();
        assert_eq!(SpawnPassBuffer::FreeList.element_count(extent), 4096);
        assert_eq!(
            SpawnPassBuffer::Counters.element_count(extent),
            SPAWN_COUNTER_COUNT
        );
        assert_eq!(SpawnPassBuffer::SpawnIndices.element_count(extent), 256);
        assert_eq!(SpawnPassBuffer::SpawnParams.element_count(extent), 3);
        assert_eq!(SpawnPassBuffer::PositionPool.element_count(extent), 4096);
        assert_eq!(SpawnPassBuffer::VelocityPool.element_count(extent), 4096);
    }

    #[test]
    fn counter_count_is_fixed_regardless_of_extent() {
        let big = SpawnPassExtent {
            capacity: 1_000_000,
            spawn_count: 999,
            emitter_count: 42,
        };
        assert_eq!(
            SpawnPassBuffer::Counters.element_count(big),
            SPAWN_COUNTER_COUNT
        );
        assert_eq!(
            SpawnPassBuffer::Counters.element_count(SpawnPassExtent::default()),
            SPAWN_COUNTER_COUNT
        );
    }

    #[test]
    fn byte_sizes_multiply_count_by_stride() {
        let extent = sample_extent();
        assert_eq!(SpawnPassBuffer::FreeList.byte_size(extent), 4096 * 4);
        assert_eq!(
            SpawnPassBuffer::Counters.byte_size(extent),
            SPAWN_COUNTER_COUNT * 4
        );
        assert_eq!(SpawnPassBuffer::SpawnIndices.byte_size(extent), 256 * 4);
        assert_eq!(SpawnPassBuffer::SpawnParams.byte_size(extent), 3 * 32);
        assert_eq!(SpawnPassBuffer::PositionPool.byte_size(extent), 4096 * 16);
        assert_eq!(SpawnPassBuffer::VelocityPool.byte_size(extent), 4096 * 16);
    }

    #[test]
    fn byte_size_scales_linearly_with_count() {
        let one = SpawnPassExtent {
            capacity: 1,
            spawn_count: 1,
            emitter_count: 1,
        };
        let ten = SpawnPassExtent {
            capacity: 10,
            spawn_count: 10,
            emitter_count: 10,
        };
        // Pools span capacity: 10x the count is 10x the bytes.
        assert_eq!(
            SpawnPassBuffer::PositionPool.byte_size(ten),
            10 * SpawnPassBuffer::PositionPool.byte_size(one)
        );
        assert_eq!(
            SpawnPassBuffer::SpawnIndices.byte_size(ten),
            10 * SpawnPassBuffer::SpawnIndices.byte_size(one)
        );
    }

    #[test]
    fn empty_extent_clamps_every_buffer_to_one_element() {
        let extent = SpawnPassExtent::default();
        assert_eq!(
            SpawnPassBuffer::FreeList.byte_size(extent),
            SpawnPassBuffer::FreeList.stride()
        );
        assert_eq!(
            SpawnPassBuffer::SpawnIndices.byte_size(extent),
            SpawnPassBuffer::SpawnIndices.stride()
        );
        assert_eq!(
            SpawnPassBuffer::SpawnParams.byte_size(extent),
            SpawnPassBuffer::SpawnParams.stride()
        );
        assert_eq!(
            SpawnPassBuffer::PositionPool.byte_size(extent),
            SpawnPassBuffer::PositionPool.stride()
        );
        assert_eq!(
            SpawnPassBuffer::VelocityPool.byte_size(extent),
            SpawnPassBuffer::VelocityPool.stride()
        );
        // Counters is never empty: it is a fixed 4-element block.
        assert_eq!(
            SpawnPassBuffer::Counters.byte_size(extent),
            SPAWN_COUNTER_COUNT * SpawnPassBuffer::Counters.stride()
        );
    }

    #[test]
    fn byte_size_saturates_instead_of_overflowing() {
        let extent = SpawnPassExtent {
            capacity: u32::MAX,
            spawn_count: u32::MAX,
            emitter_count: u32::MAX,
        };
        // vec4 stride * ~4 billion elements would overflow a 32-bit usize but is
        // representable on 64-bit; assert it never wraps below one element.
        assert!(
            SpawnPassBuffer::PositionPool.byte_size(extent)
                >= SpawnPassBuffer::PositionPool.stride()
        );
        assert!(total_storage_bytes(extent) >= SpawnPassBuffer::PositionPool.stride());
    }

    #[test]
    fn total_storage_bytes_excludes_the_uniform() {
        let extent = sample_extent();
        let expected = SpawnPassBuffer::FreeList.byte_size(extent)
            + SpawnPassBuffer::Counters.byte_size(extent)
            + SpawnPassBuffer::SpawnIndices.byte_size(extent)
            + SpawnPassBuffer::PositionPool.byte_size(extent)
            + SpawnPassBuffer::VelocityPool.byte_size(extent);
        assert_eq!(total_storage_bytes(extent), expected);
        // The uniform block's bytes are not part of the storage total.
        assert!(SpawnPassBuffer::SpawnParams.byte_size(extent) > 0);
    }
}
