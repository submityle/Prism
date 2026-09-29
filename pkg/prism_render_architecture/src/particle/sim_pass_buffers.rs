//! Device-free `@group(0)` bind-group and `std430` byte-layout contract for the
//! particle §9 `SimulationStages` compute pass (design §7, §8, §5).
//!
//! The §9 `GPU` pipeline routes every emitter through an ordered list of compute
//! passes (`EmitterUpdate`, `Spawn`, `SimulationStages`, `Event Scatter`,
//! `Compaction`, …). This module publishes *what the `SimulationStages` kernel
//! binds* at `@group(0)`: the authoritative element stride, access mode, element
//! count and total byte size of every storage (and the one uniform) buffer the
//! `WESL` kernel declares, so the render graph binds against a stable `ABI`
//! instead of hand-computing strides next to the pipeline. It mirrors the hair
//! subsystem's per-pass `*_buffers.rs` files at the algorithm level (production
//! `VFX` stacks — `Niagara`'s Simulation Stages, `Frostbite`'s FX compute graph —
//! bind the same shape of pools) without reusing any of their code.
//!
//! It is deliberately orthogonal to its neighbours:
//! - [`super::attributes`] owns *which* per-particle Structure-of-Arrays channels
//!   exist and *how wide* each is. This module never re-derives an attribute
//!   width; it only names the pools the pass binds and defers stride constants
//!   to [`super::gpu_layout`].
//! - [`super::stages`] owns the *iteration domain* (per-particle, per-cell,
//!   per-constraint) that decides a stage's dispatch size. This module owns only
//!   the byte layout of the buffers those dispatches read and write.
//! - [`super::gpu_layout`] owns the shared `std430` stride constants
//!   ([`U32_STRIDE`], [`VEC2_STRIDE`], [`VEC4_STRIDE`]), the
//!   [`ParticleBufferAccess`] enum and the clamp-to-one [`storage_bytes`] rule.
//!   This file reuses all of them and re-derives none.
//!
//! Per §5, the per-particle pools are double-buffered *ping-pong* allocations:
//! "read the previous frame, write the current frame". The Simulate pass binds
//! those persistent pools in place (see [`SimPassBuffer::persists_across_frames`]
//! and [`SimPassBuffer::aliases_persistent_pool`]) rather than allocating fresh
//! storage; only the neighbourhood grid, constraint and `lambda` scratch buffers
//! and the per-dispatch uniform block are transient per-frame allocations.
//!
//! Everything is pure integer arithmetic: an empty emitter still yields a valid,
//! non-empty `WebGPU` storage binding (one element via [`storage_bytes`]), and
//! nothing panics or divides by zero.

use super::gpu_layout::{
    storage_bytes, ParticleBufferAccess, U32_STRIDE, VEC2_STRIDE, VEC4_STRIDE,
};

/// `std430` array stride of one `XPBD` constraint descriptor:
/// `particles: vec4<u32>` (up to four incident particle indices, 16 bytes) +
/// `params: vec4<f32>` (rest value, `compliance`, plus two type-specific slots,
/// 16 bytes) = 32 bytes.
const CONSTRAINT_STRIDE: usize = 32;

/// `std430` size of the Simulate pass uniform block: `timing: vec4<f32>`
/// (`dt`, substep `dt`, substep/iteration counts reinterpreted as floats, 16) +
/// `gravity: vec4<f32>` (16) + `wind: vec4<f32>` (16) + `params: vec4<f32>`
/// (drag, damping, `compliance` scale, time, 16) = 64 bytes.
const SIM_UNIFORM_SIZE: usize = 64;

/// The non-domain extents the `SimulationStages` pass sizes its `@group(0)`
/// buffers against.
///
/// These are element counts, not dispatch domains (those live in
/// [`super::stages`]); an empty field is clamped to one element by
/// [`storage_bytes`], so a degenerate extent still binds validly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SimPassExtent {
    /// Pooled particle capacity — the length of every per-particle
    /// Structure-of-Arrays pool (positions, velocities, attributes) and of the
    /// neighbourhood entry list (one entry per particle).
    pub particle_capacity: u32,
    /// Number of spatial-hash grid cells — the length of the cell-offset table
    /// the 27-neighbour query walks.
    pub grid_cell_count: u32,
    /// Number of `XPBD` constraints in the batch — the length of the constraint
    /// descriptor and `lambda` accumulation buffers.
    pub constraint_count: u32,
}

/// One buffer bound by the `SimulationStages` kernel at `@group(0)`, in binding
/// order `0..8` (design §7, §9).
///
/// Binding `0..3` are the per-particle pools and the timing/force uniform block;
/// `4..5` are the neighbourhood spatial-hash grid; `6..7` are the `XPBD` constraint
/// batch. See each variant for its `WESL` type and access mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SimPassBuffer {
    /// `@binding(0)` position pool `array<vec4<f32>>` (`xyz` = position, `w` =
    /// inverse mass or packed age), `var<storage, read_write>`. Ping-pong per §5:
    /// the kernel reads the previous frame's positions and writes the current
    /// frame's in the aliased persistent pool.
    Positions,
    /// `@binding(1)` velocity pool `array<vec4<f32>>` (`xyz` = velocity, `w` =
    /// spare), `var<storage, read_write>`; integrated in place each substep.
    Velocities,
    /// `@binding(2)` auxiliary per-particle attribute pool `array<vec4<f32>>`
    /// (colour / custom channels; exact widths owned by [`super::attributes`]),
    /// `var<storage, read_write>` so over-life modules can mutate it.
    Attributes,
    /// `@binding(3)` timing and force uniform block `var<uniform>`: `dt`, substep
    /// `dt`, substep / iteration counts, gravity, wind and shared force
    /// parameters. Read-only to the kernel; re-uploaded each frame.
    SimParams,
    /// `@binding(4)` spatial-hash cell-offset table `array<u32>`,
    /// `var<storage, read>`: the start index into [`SimPassBuffer::GridEntries`]
    /// for each cell, letting a 27-neighbour query slice its candidates.
    GridCellStart,
    /// `@binding(5)` spatial-hash entry list `array<vec2<u32>>`
    /// (`x` = cell key, `y` = particle index), `var<storage, read>`: the sorted
    /// particle-to-cell entries the 27-neighbour query iterates.
    GridEntries,
    /// `@binding(6)` `XPBD` constraint descriptors `array<XpbdConstraint>`
    /// (incident particles + rest / `compliance` params), `var<storage, read>`:
    /// the graph-coloured batch this dispatch solves.
    Constraints,
    /// `@binding(7)` per-constraint `lambda` accumulators `array<f32>`,
    /// `var<storage, read_write>`: the running Lagrange multipliers `XPBD`
    /// accumulates across substep iterations.
    ConstraintLambdas,
}

impl SimPassBuffer {
    /// Every `SimulationStages` buffer in `@binding` order (`0..8`).
    pub const ALL: [SimPassBuffer; 8] = [
        Self::Positions,
        Self::Velocities,
        Self::Attributes,
        Self::SimParams,
        Self::GridCellStart,
        Self::GridEntries,
        Self::Constraints,
        Self::ConstraintLambdas,
    ];

    /// The `@group(0)` binding index in the `SimulationStages` `WESL` kernel.
    #[must_use]
    pub fn binding(self) -> u32 {
        match self {
            Self::Positions => 0,
            Self::Velocities => 1,
            Self::Attributes => 2,
            Self::SimParams => 3,
            Self::GridCellStart => 4,
            Self::GridEntries => 5,
            Self::Constraints => 6,
            Self::ConstraintLambdas => 7,
        }
    }

    /// Byte stride of one element, matching the `WESL` struct / scalar layout.
    ///
    /// The uniform block reports its whole-block `std430` size, since it binds a
    /// single element.
    #[must_use]
    pub fn stride(self) -> usize {
        match self {
            Self::Positions | Self::Velocities | Self::Attributes => VEC4_STRIDE,
            Self::SimParams => SIM_UNIFORM_SIZE,
            Self::GridCellStart | Self::ConstraintLambdas => U32_STRIDE,
            Self::GridEntries => VEC2_STRIDE,
            Self::Constraints => CONSTRAINT_STRIDE,
        }
    }

    /// Whether the kernel reads or read-writes this buffer.
    ///
    /// The uniform block is read-only, so it reports [`ParticleBufferAccess::Read`].
    #[must_use]
    pub fn access(self) -> ParticleBufferAccess {
        match self {
            Self::Positions | Self::Velocities | Self::Attributes | Self::ConstraintLambdas => {
                ParticleBufferAccess::ReadWrite
            }
            Self::SimParams | Self::GridCellStart | Self::GridEntries | Self::Constraints => {
                ParticleBufferAccess::Read
            }
        }
    }

    /// Whether the pass writes this buffer rather than only reading it.
    #[must_use]
    pub fn is_output(self) -> bool {
        self.access().is_writable()
    }

    /// Whether the buffer's contents survive between frames.
    ///
    /// The per-particle Structure-of-Arrays pools (positions, velocities and the
    /// auxiliary attribute pool) are persistent ping-pong allocations per §5. The
    /// uniform block, neighbourhood grid and constraint / `lambda` buffers are
    /// rebuilt or re-uploaded every frame and do not persist.
    #[must_use]
    pub fn persists_across_frames(self) -> bool {
        matches!(self, Self::Positions | Self::Velocities | Self::Attributes)
    }

    /// Whether this binding aliases an existing persistent pool rather than a
    /// fresh per-pass allocation.
    ///
    /// The in-place-integrated per-particle pools bind the persistent §5
    /// double-buffered storage (a ping-pong half), so they allocate nothing new.
    /// Everything else ([`SimPassBuffer::SimParams`], the grid tables and the
    /// constraint batch) is transient per-frame scratch. Mirrors the hair
    /// `Simulate`-stage aliasing of the persistent guide-position pool.
    #[must_use]
    pub fn aliases_persistent_pool(self) -> bool {
        self.persists_across_frames()
    }

    /// Element count for the given `extent`.
    ///
    /// The per-particle pools and the neighbourhood entry list are `particle`-sized;
    /// the cell-offset table is grid-sized; the constraint and `lambda` buffers are
    /// constraint-sized; the uniform block is always a single element.
    #[must_use]
    pub fn element_count(self, extent: SimPassExtent) -> u32 {
        match self {
            Self::Positions | Self::Velocities | Self::Attributes | Self::GridEntries => {
                extent.particle_capacity
            }
            Self::SimParams => 1,
            Self::GridCellStart => extent.grid_cell_count,
            Self::Constraints | Self::ConstraintLambdas => extent.constraint_count,
        }
    }

    /// Total byte size for `extent`, clamped up to one element so an empty emitter
    /// still yields a valid non-empty `WebGPU` binding (via [`storage_bytes`]).
    #[must_use]
    pub fn byte_size(self, extent: SimPassExtent) -> usize {
        storage_bytes(self.stride(), self.element_count(extent) as usize)
    }
}

/// Bytes the `SimulationStages` pass allocates afresh for a given `extent`: the
/// uniform block, neighbourhood grid tables and constraint batch.
///
/// The per-particle pools are excluded because they alias the persistent §5
/// ping-pong allocation (see [`SimPassBuffer::aliases_persistent_pool`]) and cost
/// no new storage this pass. Every summed buffer is clamped to one element.
#[must_use]
pub fn transient_scratch_bytes(extent: SimPassExtent) -> usize {
    SimPassBuffer::ALL
        .into_iter()
        .filter(|buffer| !buffer.aliases_persistent_pool())
        .map(|buffer| buffer.byte_size(extent))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_extent() -> SimPassExtent {
        SimPassExtent {
            particle_capacity: 1024,
            grid_cell_count: 512,
            constraint_count: 200,
        }
    }

    #[test]
    fn bindings_are_dense_and_ordered() {
        for (index, buffer) in SimPassBuffer::ALL.into_iter().enumerate() {
            assert_eq!(buffer.binding() as usize, index);
        }
        assert_eq!(SimPassBuffer::ALL.len(), 8);
    }

    #[test]
    fn strides_match_the_wesl_layout() {
        assert_eq!(SimPassBuffer::Positions.stride(), 16);
        assert_eq!(SimPassBuffer::Velocities.stride(), 16);
        assert_eq!(SimPassBuffer::Attributes.stride(), 16);
        assert_eq!(SimPassBuffer::SimParams.stride(), 64);
        assert_eq!(SimPassBuffer::GridCellStart.stride(), 4);
        assert_eq!(SimPassBuffer::GridEntries.stride(), 8);
        assert_eq!(SimPassBuffer::Constraints.stride(), 32);
        assert_eq!(SimPassBuffer::ConstraintLambdas.stride(), 4);
    }

    #[test]
    fn access_modes_match_the_kernel() {
        for buffer in [
            SimPassBuffer::Positions,
            SimPassBuffer::Velocities,
            SimPassBuffer::Attributes,
            SimPassBuffer::ConstraintLambdas,
        ] {
            assert_eq!(buffer.access(), ParticleBufferAccess::ReadWrite);
            assert!(buffer.is_output());
        }
        for buffer in [
            SimPassBuffer::SimParams,
            SimPassBuffer::GridCellStart,
            SimPassBuffer::GridEntries,
            SimPassBuffer::Constraints,
        ] {
            assert_eq!(buffer.access(), ParticleBufferAccess::Read);
            assert!(!buffer.is_output());
        }
    }

    #[test]
    fn persistence_covers_only_the_particle_pools() {
        for buffer in [
            SimPassBuffer::Positions,
            SimPassBuffer::Velocities,
            SimPassBuffer::Attributes,
        ] {
            assert!(buffer.persists_across_frames());
            assert!(buffer.aliases_persistent_pool());
        }
        for buffer in [
            SimPassBuffer::SimParams,
            SimPassBuffer::GridCellStart,
            SimPassBuffer::GridEntries,
            SimPassBuffer::Constraints,
            SimPassBuffer::ConstraintLambdas,
        ] {
            assert!(!buffer.persists_across_frames());
            assert!(!buffer.aliases_persistent_pool());
        }
    }

    #[test]
    fn element_counts_follow_the_extent() {
        let extent = sample_extent();
        assert_eq!(SimPassBuffer::Positions.element_count(extent), 1024);
        assert_eq!(SimPassBuffer::Velocities.element_count(extent), 1024);
        assert_eq!(SimPassBuffer::Attributes.element_count(extent), 1024);
        assert_eq!(SimPassBuffer::GridEntries.element_count(extent), 1024);
        assert_eq!(SimPassBuffer::SimParams.element_count(extent), 1);
        assert_eq!(SimPassBuffer::GridCellStart.element_count(extent), 512);
        assert_eq!(SimPassBuffer::Constraints.element_count(extent), 200);
        assert_eq!(SimPassBuffer::ConstraintLambdas.element_count(extent), 200);
    }

    #[test]
    fn byte_sizes_multiply_count_by_stride() {
        let extent = sample_extent();
        assert_eq!(SimPassBuffer::Positions.byte_size(extent), 1024 * 16);
        assert_eq!(SimPassBuffer::SimParams.byte_size(extent), 64);
        assert_eq!(SimPassBuffer::GridCellStart.byte_size(extent), 512 * 4);
        assert_eq!(SimPassBuffer::GridEntries.byte_size(extent), 1024 * 8);
        assert_eq!(SimPassBuffer::Constraints.byte_size(extent), 200 * 32);
        assert_eq!(SimPassBuffer::ConstraintLambdas.byte_size(extent), 200 * 4);
    }

    #[test]
    fn empty_extent_clamps_every_buffer_to_one_element() {
        let extent = SimPassExtent::default();
        for buffer in SimPassBuffer::ALL {
            assert_eq!(buffer.byte_size(extent), buffer.stride());
        }
    }

    #[test]
    fn transient_scratch_excludes_the_persistent_pools() {
        let extent = sample_extent();
        let expected = SimPassBuffer::SimParams.byte_size(extent)
            + SimPassBuffer::GridCellStart.byte_size(extent)
            + SimPassBuffer::GridEntries.byte_size(extent)
            + SimPassBuffer::Constraints.byte_size(extent)
            + SimPassBuffer::ConstraintLambdas.byte_size(extent);
        assert_eq!(transient_scratch_bytes(extent), expected);
        assert_eq!(
            transient_scratch_bytes(extent),
            64 + 2048 + 8192 + 6400 + 800
        );
    }

    #[test]
    fn writable_buffers_are_exactly_the_outputs() {
        let outputs: usize = SimPassBuffer::ALL
            .into_iter()
            .filter(|buffer| buffer.is_output())
            .count();
        assert_eq!(outputs, 4);
    }
}
