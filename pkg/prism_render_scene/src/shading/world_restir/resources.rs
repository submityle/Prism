//! Per-view resident reservoir table backing the world-space `ReSTIR`
//! direct-illumination fill pass.
//!
//! World-space `ReSTIR` keeps a `SHARC`-style spatial-hash cache of streaming
//! `RIS` reservoirs — one open-addressed slot per hashed world cell — resident
//! on the GPU across frames. The fill pass reads last frame's table and writes
//! this frame's, so the subsystem owns a *ping-pong* pair of storage buffers
//! and flips which is the read source each frame:
//!
//! * `fill_main` binds the previous table read-only (`src`) and the next table
//!   read-write (`dst`): it copies empty slots through unchanged and, for every
//!   occupied slot, recomputes the cell key, merges a jittered ring of spatial
//!   neighbours (golden `GRIS` reuse) and finalises the slot's contribution
//!   weight. Next frame the two buffers swap, so the finalised `dst` becomes
//!   the `src` the following fill reuses.
//!
//! Each buffer is `max(capacity, 1)` slots of
//! [`WORLD_RESTIR_RESERVOIR_STRIDE`] bytes. `wgpu` zero-initialises a freshly
//! created storage buffer before its first shader read, so both tables start
//! as all-[`GpuWorldRestirReservoir::EMPTY`] (invalid) slots without an
//! explicit clear. The table dimension is world-space (hash-grid capacity), not
//! viewport-sized, so the only reallocation trigger is a `capacity` change; the
//! per-frame work is the cheap ping-pong flip + `RNG`-seed advance.
//!
//! The subsystem is opt-in: when [`PrismWorldRestirSettings::enabled`] is
//! `false` the table is dropped and nothing is allocated or dispatched, exactly
//! mirroring [`super::super::world_space_gi`]'s `ViewWorldSpaceGi` lifecycle.

use bevy_ecs::prelude::*;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{Buffer, BufferDescriptor, BufferUsages},
    renderer::RenderDevice,
};

use super::abi::{GpuWorldRestirReservoir, WORLD_RESTIR_RESERVOIR_STRIDE};
use super::settings::PrismWorldRestirSettings;

/// Compile-time guard that the resident buffer stride the allocator sizes with
/// is byte-for-byte the frozen reservoir-slot ABI the shader indexes, so the
/// host allocation and the WESL `array<WorldRestirReservoir>` never drift.
const _: () = assert!(size_of::<GpuWorldRestirReservoir>() as u64 == WORLD_RESTIR_RESERVOIR_STRIDE);

/// Ping-pong read-source buffer index for a given monotonic frame: even frames
/// read slot `0`, odd frames read slot `1`.
const fn reservoir_src_index(frame: u32) -> usize {
    (frame & 1) as usize
}

/// Ping-pong write-target buffer index for a given monotonic frame: the
/// complement of [`reservoir_src_index`], so the fill never reads and writes
/// the same buffer.
const fn reservoir_dst_index(frame: u32) -> usize {
    ((frame & 1) ^ 1) as usize
}

/// Per-view resident world-space `ReSTIR` reservoir table, present only while
/// the fill pass is enabled.
#[derive(Component)]
pub(crate) struct ViewWorldRestir {
    /// Ping-pong pair of reservoir storage buffers. Each is `capacity` slots of
    /// [`WORLD_RESTIR_RESERVOIR_STRIDE`] bytes; [`frame`](Self::frame) selects
    /// which is this frame's read source and which is the write target.
    reservoirs: [Buffer; 2],
    /// Resident slot count this pair was allocated for (clamped to at least
    /// `1`); the dispatch extent and the only reallocation trigger.
    pub(crate) capacity: u32,
    /// Monotonic frame counter: drives the ping-pong flip (`frame & 1`) and
    /// seeds the fill shader's per-slot streaming `RNG`. The crate does not link
    /// Bevy's `FrameCount`, so the counter lives here and advances in
    /// [`prepare_world_restir_reservoirs`] (mirroring `volumetric_clouds`).
    frame: u32,
}

impl ViewWorldRestir {
    /// This frame's read-only reservoir table (last frame's finalised `dst`).
    pub(crate) fn src_buffer(&self) -> &Buffer {
        &self.reservoirs[reservoir_src_index(self.frame)]
    }

    /// This frame's read-write reservoir table (the fill pass's output).
    pub(crate) fn dst_buffer(&self) -> &Buffer {
        &self.reservoirs[reservoir_dst_index(self.frame)]
    }

    /// Monotonic frame index seeding the fill shader's per-slot `RNG`.
    pub(crate) fn frame(&self) -> u32 {
        self.frame
    }
}

/// (Re)allocates [`ViewWorldRestir`] for every camera view while the fill pass
/// is enabled, and removes it otherwise.
///
/// Gated solely on [`PrismWorldRestirSettings::enabled`] (the resident table is
/// world-space, not tied to the prepass or viewport). When the table already
/// exists at the live `capacity` the system only advances the ping-pong frame
/// counter; the buffers are recreated only when `capacity` changes.
pub(crate) fn prepare_world_restir_reservoirs(
    mut commands: Commands,
    settings: Res<PrismWorldRestirSettings>,
    device: Res<RenderDevice>,
    mut views: Query<(Entity, Option<&mut ViewWorldRestir>), With<ExtractedCamera>>,
) {
    for (entity, existing) in &mut views {
        if !settings.enabled {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewWorldRestir>();
            }
            continue;
        }

        let capacity = settings.capacity.max(1);
        // Steady state: an existing table already at the live capacity only
        // flips the ping-pong and advances the RNG seed. Every other case (no
        // table yet, or the capacity was retuned) falls through to (re)allocate
        // the ping-pong pair below.
        if let Some(mut table) = existing
            && table.capacity == capacity
        {
            table.frame = table.frame.wrapping_add(1);
            continue;
        }

        let size = settings.reservoir_buffer_size();
        // STORAGE: bound read-only as `src`, read-write as `dst`. COPY_DST so a
        // host-side zero clear can be scheduled if ever needed; `wgpu` already
        // zero-initialises the buffer before its first shader read, so both
        // tables start as all-EMPTY (invalid) slots.
        let make = |label: &'static str| {
            device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let reservoirs = [
            make("prism world-space ReSTIR reservoirs A"),
            make("prism world-space ReSTIR reservoirs B"),
        ];

        commands.entity(entity).insert(ViewWorldRestir {
            reservoirs,
            capacity,
            frame: 0,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservoir_stride_matches_the_slot_abi() {
        // The compile-time guard above already enforces this; assert it at
        // runtime too so the intent is visible in the test report.
        assert_eq!(
            size_of::<GpuWorldRestirReservoir>() as u64,
            WORLD_RESTIR_RESERVOIR_STRIDE
        );
    }

    #[test]
    fn ping_pong_indices_alternate_and_never_alias() {
        for frame in 0u32..8 {
            let src = reservoir_src_index(frame);
            let dst = reservoir_dst_index(frame);
            // src/dst are always the two distinct slots of the pair.
            assert!(src < 2 && dst < 2);
            assert_ne!(src, dst, "fill must never read and write the same buffer");
        }
        // Even frames read slot 0; odd frames read slot 1 (and write the other).
        assert_eq!(reservoir_src_index(0), 0);
        assert_eq!(reservoir_dst_index(0), 1);
        assert_eq!(reservoir_src_index(1), 1);
        assert_eq!(reservoir_dst_index(1), 0);
        // Consecutive frames swap read source and write target (true ping-pong):
        // this frame's dst is next frame's src.
        assert_eq!(reservoir_dst_index(0), reservoir_src_index(1));
        assert_eq!(reservoir_dst_index(1), reservoir_src_index(2));
    }
}
