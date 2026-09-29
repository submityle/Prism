//! Persistent `GPU` buffer bookkeeping and the async double-buffer state
//! machine for the water/fluid solvers.
//!
//! A production `GPU`-driven water engine keeps its simulation state resident on
//! the device across frames rather than re-uploading it every frame: the ocean
//! spectrum amplitude arrays and their transformed displacement/normal fields,
//! the Shallow-Water height/velocity grid, the `PBF` particle pool and its
//! neighbour spatial-hash, the `FLIP`/`APIC` particle pool and `MAC` grid
//! scalars, the semi-Lagrangian foam coverage field and the surface wetness
//! field all live in persistent storage across frames, and the compute
//! dispatches read and write them in place. Only the small per-step uniform
//! blocks are streamed.
//!
//! This module owns the *sizing and lifecycle contract* for those buffers: how
//! many elements and bytes each resident buffer needs (saturating integer
//! bookkeeping, so a pathological count can never overflow the allocation size
//! the backend reads), which buffers are double-buffered so a solve can read
//! the previous frame while writing the next, and the async state machine that
//! sequences record → submit → retire without ever reading a slot the `GPU` is
//! still writing. It holds no `GPU` handles and does no allocation itself; the
//! backend consumes the byte counts to make the real allocations, and the
//! numerical passes live in `WESL` (`water_ocean`, `water_flip`, `water_pbf`,
//! `water_render_fx`, `water_surface`) mirroring these strides byte-for-byte.

use alloc::vec::Vec;

/// Byte stride of one packed complex spectrum amplitude (`h0`), a
/// `vec2<f32>` (real, imaginary) in `std430`. Matches the
/// `array<vec2<f32>>` binding the ocean `Tessendorf` `WESL` reads for the
/// initial spectrum and its conjugate.
pub const SPECTRUM_AMPLITUDE_STRIDE: u32 = 8;

/// Byte stride of one displacement texel, an `rgba32float` (`xyz` horizontal +
/// vertical displacement, `w` foldover/Jacobian). Matches the
/// `texture_storage_2d<rgba32float>` the spectrum/`Gerstner` kernels write.
pub const DISPLACEMENT_TEXEL_STRIDE: u32 = 16;

/// Byte stride of one surface normal texel, an `rgba32float` (`xyz` normal,
/// `w` folding/whitecap weight). Matches the normal storage texture.
pub const NORMAL_TEXEL_STRIDE: u32 = 16;

/// Byte stride of one analytic `Gerstner` wave record: eight `f32` lanes
/// (`dir_x`, `dir_z`, amplitude, wavelength, steepness, speed, phase and a
/// pad lane) packed as two `vec4<f32>` rows, matching the `WESL`
/// `GerstnerWave` `struct` byte-for-byte (2 x 16 = 32 bytes). A single
/// `vec4<f32>` (16 bytes) cannot carry all seven meaningful wave scalars, so
/// sizing the wave-train buffer at 16 bytes under-allocated it by half.
pub const GERSTNER_WAVE_STRIDE: u32 = 32;

/// Byte stride of one Shallow-Water grid cell: a `vec4<f32>` packing the water
/// height, the two horizontal velocity components and a flux/terrain lane, at
/// the 16-byte `std430` stride.
pub const SWE_CELL_STRIDE: u32 = 16;

/// Byte stride of one `PBF` particle position record, a `vec4<f32>` (`xyz`
/// world position, `w` active/lambda lane). Matches the
/// `array<vec4<f32>>` position bindings the `PBF` density solve ping-pongs.
pub const PBF_PARTICLE_STRIDE: u32 = 16;

/// Byte stride of one `FLIP`/`APIC` particle, five `vec4<f32>` lanes:
/// position with active flag, velocity, and the three `APIC` affine-matrix
/// rows. Matches the `WESL` `FlipParticle` struct byte-for-byte (5 × 16 = 80
/// bytes).
pub const FLIP_PARTICLE_STRIDE: u32 = 80;

/// Byte stride of one grid scalar (`FLIP` pressure/divergence, spatial-hash
/// entry), a single `u32`/`f32` at 4 bytes.
pub const GRID_SCALAR_STRIDE: u32 = 4;

/// Byte stride of one foam coverage cell, a single `f32` density at 4 bytes.
/// The foam field is semi-Lagrangian advected, so it is double-buffered.
pub const FOAM_CELL_STRIDE: u32 = 4;

/// Byte stride of one surface wetness cell, a single `f32` moisture level at
/// 4 bytes.
pub const WETNESS_CELL_STRIDE: u32 = 4;

/// Byte stride of one underwater froxel, an `rgba32float` accumulating
/// single-scatter radiance (`rgb`) and transmittance (`a`).
pub const FROXEL_STRIDE: u32 = 16;

/// The resident element counts that size every persistent water buffer for one
/// water body.
///
/// Each field is an element count (spectrum texels, grid cells, particles,
/// froxels), not a byte size; [`WaterPersistentBufferSet`] turns them into byte
/// sizes with the packed strides above. All arithmetic downstream is saturating
/// so an adversarial count clamps to `u32::MAX` bytes rather than wrapping.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct WaterBufferCounts {
    /// Number of ocean spectrum texels summed across cascades (sizes the `h0`
    /// amplitude arrays and the displacement/normal storage textures).
    pub spectrum_texels: u32,
    /// Number of analytic `Gerstner` wave trains.
    pub gerstner_waves: u32,
    /// Number of Shallow-Water grid cells (height/velocity field).
    pub swe_cells: u32,
    /// Number of `PBF` particles (density-constrained fluid).
    pub pbf_particles: u32,
    /// Number of spatial-hash entries for the `PBF` neighbourhood search (one
    /// `u32` particle index per particle).
    pub pbf_hash_entries: u32,
    /// Number of `FLIP`/`APIC` particles.
    pub flip_particles: u32,
    /// Number of `FLIP`/`APIC` `MAC` grid scalar cells (pressure/divergence,
    /// one scalar buffer's worth).
    pub flip_grid_cells: u32,
    /// Number of foam coverage cells (semi-Lagrangian, double-buffered).
    pub foam_cells: u32,
    /// Number of surface wetness cells.
    pub wetness_cells: u32,
    /// Number of underwater froxels (single-scatter volume).
    pub froxels: u32,
}

/// A view over one water body's resident buffer sizing, derived from its
/// [`WaterBufferCounts`]. Every accessor returns saturating byte counts so an
/// adversarial element count can never wrap the allocation size.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct WaterPersistentBufferSet {
    counts: WaterBufferCounts,
}

impl WaterPersistentBufferSet {
    /// Builds the sizing view from element counts.
    #[must_use]
    pub fn new(counts: WaterBufferCounts) -> Self {
        Self { counts }
    }

    /// The element counts backing this set.
    #[must_use]
    pub fn counts(self) -> WaterBufferCounts {
        self.counts
    }

    /// Bytes for one spectrum `h0` amplitude array. The conjugate spectrum
    /// (`h0_neg`) is a second array of the same size.
    #[must_use]
    pub fn spectrum_amplitude_bytes(self) -> u32 {
        self.counts
            .spectrum_texels
            .saturating_mul(SPECTRUM_AMPLITUDE_STRIDE)
    }

    /// Bytes for the transformed displacement storage texture.
    #[must_use]
    pub fn displacement_bytes(self) -> u32 {
        self.counts
            .spectrum_texels
            .saturating_mul(DISPLACEMENT_TEXEL_STRIDE)
    }

    /// Bytes for the surface normal/foldover storage texture.
    #[must_use]
    pub fn normal_bytes(self) -> u32 {
        self.counts
            .spectrum_texels
            .saturating_mul(NORMAL_TEXEL_STRIDE)
    }

    /// Bytes for the analytic `Gerstner` wave-train buffer.
    #[must_use]
    pub fn gerstner_bytes(self) -> u32 {
        self.counts
            .gerstner_waves
            .saturating_mul(GERSTNER_WAVE_STRIDE)
    }

    /// Bytes for the Shallow-Water height/velocity grid.
    #[must_use]
    pub fn swe_bytes(self) -> u32 {
        self.counts.swe_cells.saturating_mul(SWE_CELL_STRIDE)
    }

    /// Bytes for one `PBF` particle-position pool. The `PBF` density solve
    /// ping-pongs positions, so this pool is double-buffered.
    #[must_use]
    pub fn pbf_particle_bytes(self) -> u32 {
        self.counts
            .pbf_particles
            .saturating_mul(PBF_PARTICLE_STRIDE)
    }

    /// Bytes for the `PBF` spatial-hash entry buffer.
    #[must_use]
    pub fn pbf_hash_bytes(self) -> u32 {
        self.counts
            .pbf_hash_entries
            .saturating_mul(GRID_SCALAR_STRIDE)
    }

    /// Bytes for the `FLIP`/`APIC` particle pool.
    #[must_use]
    pub fn flip_particle_bytes(self) -> u32 {
        self.counts
            .flip_particles
            .saturating_mul(FLIP_PARTICLE_STRIDE)
    }

    /// Bytes for one `FLIP` `MAC` grid scalar buffer (pressure/divergence). The
    /// pressure solve reads the previous iterate while writing the next, so
    /// this buffer is double-buffered.
    #[must_use]
    pub fn flip_grid_bytes(self) -> u32 {
        self.counts
            .flip_grid_cells
            .saturating_mul(GRID_SCALAR_STRIDE)
    }

    /// Bytes for one foam coverage field. Semi-Lagrangian advection reads the
    /// previous field while writing the next, so it is double-buffered.
    #[must_use]
    pub fn foam_bytes(self) -> u32 {
        self.counts.foam_cells.saturating_mul(FOAM_CELL_STRIDE)
    }

    /// Bytes for the surface wetness field.
    #[must_use]
    pub fn wetness_bytes(self) -> u32 {
        self.counts
            .wetness_cells
            .saturating_mul(WETNESS_CELL_STRIDE)
    }

    /// Bytes for the underwater single-scatter froxel volume.
    #[must_use]
    pub fn froxel_bytes(self) -> u32 {
        self.counts.froxels.saturating_mul(FROXEL_STRIDE)
    }

    /// Total resident bytes for every persistent water buffer. The
    /// double-buffered pools — the spectrum `h0`/conjugate pair, the `PBF`
    /// position pool, the `FLIP` pressure buffer and the foam field — are
    /// counted twice. Saturating throughout.
    #[must_use]
    pub fn total_bytes(self) -> u32 {
        let spectrum_pair = self.spectrum_amplitude_bytes().saturating_mul(2);
        let pbf_double = self.pbf_particle_bytes().saturating_mul(2);
        let pressure_double = self.flip_grid_bytes().saturating_mul(2);
        let foam_double = self.foam_bytes().saturating_mul(2);
        spectrum_pair
            .saturating_add(self.displacement_bytes())
            .saturating_add(self.normal_bytes())
            .saturating_add(self.gerstner_bytes())
            .saturating_add(self.swe_bytes())
            .saturating_add(pbf_double)
            .saturating_add(self.pbf_hash_bytes())
            .saturating_add(self.flip_particle_bytes())
            .saturating_add(pressure_double)
            .saturating_add(foam_double)
            .saturating_add(self.wetness_bytes())
            .saturating_add(self.froxel_bytes())
    }

    /// Returns `true` when the set has at least one active simulation buffer,
    /// i.e. some solver can actually run. An empty body needs no resident
    /// allocation.
    #[must_use]
    pub fn is_simulatable(self) -> bool {
        self.counts.spectrum_texels > 0
            || self.counts.gerstner_waves > 0
            || self.counts.swe_cells > 0
            || self.counts.pbf_particles > 0
            || self.counts.flip_particles > 0
    }
}

/// Which of the two copies a double-buffered water pool currently treats as the
/// readable front buffer.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BufferParity {
    /// Copy 0 is the front (readable) buffer; copy 1 is the write target.
    Even,
    /// Copy 1 is the front (readable) buffer; copy 0 is the write target.
    Odd,
}

impl BufferParity {
    /// The index (0 or 1) of the readable front buffer.
    #[must_use]
    pub fn front_index(self) -> u32 {
        match self {
            BufferParity::Even => 0,
            BufferParity::Odd => 1,
        }
    }

    /// The index (0 or 1) of the write-target back buffer.
    #[must_use]
    pub fn back_index(self) -> u32 {
        match self {
            BufferParity::Even => 1,
            BufferParity::Odd => 0,
        }
    }

    /// The parity after one swap.
    #[must_use]
    pub fn swapped(self) -> Self {
        match self {
            BufferParity::Even => BufferParity::Odd,
            BufferParity::Odd => BufferParity::Even,
        }
    }
}

/// The lifecycle state of one async frame slot in the double-buffered water
/// pipeline.
///
/// A solve records commands into an `Idle` slot (→ `Recording`), submits it
/// (→ `InFlight`), and once the `GPU` fence signals marks it `Ready` for its
/// results (readbacks, coupling queries) to be read, after which it returns to
/// `Idle`. The transitions are enforced by [`AsyncFrameState`] so a slot is
/// never read while the `GPU` is still writing it.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SlotState {
    /// Free to record into.
    Idle,
    /// Currently recording compute dispatches on the `CPU` timeline.
    Recording,
    /// Submitted to the `GPU`; its fence has not yet signalled.
    InFlight,
    /// The `GPU` fence signalled; results are safe to read, then recycle.
    Ready,
}

/// A single async frame slot: its lifecycle state and the buffer parity it was
/// recorded with, so a slot read back later knows which pool copy holds its
/// result.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FrameSlot {
    /// The lifecycle state of the slot.
    pub state: SlotState,
    /// The buffer parity this slot integrated with.
    pub parity: BufferParity,
}

/// An error returned when an async slot transition is requested out of order.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PipelineError {
    /// A transition was requested that the current [`SlotState`] does not allow
    /// (for example submitting a slot that is not recording).
    IllegalTransition,
    /// A slot index outside the ring was addressed.
    SlotOutOfRange,
    /// Recording was requested but no slot is currently `Idle`, i.e. the `GPU`
    /// is saturated and the `CPU` must wait for a fence.
    NoIdleSlot,
}

/// The async double-buffer state machine for one water body's `GPU` solve.
///
/// It owns a small ring of [`FrameSlot`]s (double- or triple-buffered) and the
/// live [`BufferParity`]. Each frame the `CPU` acquires an `Idle` slot, records
/// into it, submits it, and later retires it when its fence signals — swapping
/// the pool parity exactly once per submitted frame so the next frame reads the
/// freshly written state. All indices are bounded and every transition is
/// validated, so the machine can be exhaustively `CPU`-tested.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AsyncFrameState {
    slots: Vec<FrameSlot>,
    parity: BufferParity,
    submitted: u64,
    retired: u64,
}

impl AsyncFrameState {
    /// Builds a state machine with `slot_count` frame slots (clamped to at
    /// least two so the pipeline is always at least double-buffered).
    #[must_use]
    pub fn new(slot_count: u32) -> Self {
        let count = slot_count.max(2) as usize;
        let mut slots = Vec::with_capacity(count);
        for _ in 0..count {
            slots.push(FrameSlot {
                state: SlotState::Idle,
                parity: BufferParity::Even,
            });
        }
        Self {
            slots,
            parity: BufferParity::Even,
            submitted: 0,
            retired: 0,
        }
    }

    /// The number of frame slots in the ring.
    #[must_use]
    pub fn slot_count(&self) -> u32 {
        self.slots.len() as u32
    }

    /// The live buffer parity the next recording will integrate with.
    #[must_use]
    pub fn parity(&self) -> BufferParity {
        self.parity
    }

    /// The total number of frames submitted to the `GPU` so far.
    #[must_use]
    pub fn submitted(&self) -> u64 {
        self.submitted
    }

    /// The total number of frames retired (fence-signalled and read back).
    #[must_use]
    pub fn retired(&self) -> u64 {
        self.retired
    }

    /// The number of frames currently in flight (submitted but not retired).
    #[must_use]
    pub fn in_flight(&self) -> u64 {
        self.submitted.saturating_sub(self.retired)
    }

    /// The lifecycle state of slot `index`, or `None` when out of range.
    #[must_use]
    pub fn slot(&self, index: u32) -> Option<FrameSlot> {
        self.slots.get(index as usize).copied()
    }

    /// Finds the first `Idle` slot without changing state.
    #[must_use]
    pub fn first_idle(&self) -> Option<u32> {
        self.slots
            .iter()
            .position(|slot| slot.state == SlotState::Idle)
            .map(|index| index as u32)
    }

    /// Acquires the first `Idle` slot, transitions it to `Recording`, and
    /// stamps it with the live parity. Returns the acquired slot index.
    ///
    /// Fails with [`PipelineError::NoIdleSlot`] when every slot is busy, which
    /// is the signal for the caller to wait on the oldest in-flight fence
    /// before recording another frame.
    pub fn begin_record(&mut self) -> Result<u32, PipelineError> {
        let Some(index) = self.first_idle() else {
            return Err(PipelineError::NoIdleSlot);
        };
        let parity = self.parity;
        let slot = &mut self.slots[index as usize];
        slot.state = SlotState::Recording;
        slot.parity = parity;
        Ok(index)
    }

    /// Submits a `Recording` slot to the `GPU` (→ `InFlight`), increments the
    /// submitted counter, and swaps the pool parity so the next recorded frame
    /// reads the state this frame will write.
    pub fn submit(&mut self, index: u32) -> Result<(), PipelineError> {
        let slot = self
            .slots
            .get_mut(index as usize)
            .ok_or(PipelineError::SlotOutOfRange)?;
        if slot.state != SlotState::Recording {
            return Err(PipelineError::IllegalTransition);
        }
        slot.state = SlotState::InFlight;
        self.submitted = self.submitted.saturating_add(1);
        self.parity = self.parity.swapped();
        Ok(())
    }

    /// Marks an `InFlight` slot `Ready` when its `GPU` fence signals. The
    /// results are then safe to read.
    pub fn signal_ready(&mut self, index: u32) -> Result<(), PipelineError> {
        let slot = self
            .slots
            .get_mut(index as usize)
            .ok_or(PipelineError::SlotOutOfRange)?;
        if slot.state != SlotState::InFlight {
            return Err(PipelineError::IllegalTransition);
        }
        slot.state = SlotState::Ready;
        Ok(())
    }

    /// Retires a `Ready` slot back to `Idle` after its results are read,
    /// incrementing the retired counter so the slot can be recorded again.
    pub fn retire(&mut self, index: u32) -> Result<(), PipelineError> {
        let slot = self
            .slots
            .get_mut(index as usize)
            .ok_or(PipelineError::SlotOutOfRange)?;
        if slot.state != SlotState::Ready {
            return Err(PipelineError::IllegalTransition);
        }
        slot.state = SlotState::Idle;
        self.retired = self.retired.saturating_add(1);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AsyncFrameState, BufferParity, PipelineError, SlotState, WaterBufferCounts,
        WaterPersistentBufferSet, FLIP_PARTICLE_STRIDE, GERSTNER_WAVE_STRIDE, PBF_PARTICLE_STRIDE,
        SPECTRUM_AMPLITUDE_STRIDE,
    };

    fn counts() -> WaterBufferCounts {
        WaterBufferCounts {
            spectrum_texels: 256 * 256,
            gerstner_waves: 32,
            swe_cells: 128 * 128,
            pbf_particles: 20_000,
            pbf_hash_entries: 20_000,
            flip_particles: 40_000,
            flip_grid_cells: 64 * 64 * 64,
            foam_cells: 512 * 512,
            wetness_cells: 256 * 256,
            froxels: 160 * 90 * 64,
        }
    }

    #[test]
    fn flip_particle_stride_matches_wesl_golden() {
        // Five vec4<f32> lanes: pos+flag, vel, and three APIC affine rows.
        assert_eq!(FLIP_PARTICLE_STRIDE, 5 * 16);
    }

    #[test]
    fn gerstner_wave_stride_matches_wesl_golden() {
        // Eight f32 lanes (dir_x, dir_z, amplitude, wavelength, steepness,
        // speed, phase, pad) = two vec4<f32> rows, matching the WESL
        // `GerstnerWave` struct. Pins the stride so a drift back to 16 bytes
        // (which would under-allocate the wave-train buffer by half and read
        // past its end on the GPU) fails the build.
        assert_eq!(GERSTNER_WAVE_STRIDE, 8 * 4);
        let set = WaterPersistentBufferSet::new(counts());
        assert_eq!(
            set.gerstner_bytes(),
            counts().gerstner_waves * GERSTNER_WAVE_STRIDE
        );
    }

    #[test]
    fn spectrum_amplitude_is_complex_pair() {
        assert_eq!(SPECTRUM_AMPLITUDE_STRIDE, 8);
        let set = WaterPersistentBufferSet::new(counts());
        assert_eq!(
            set.spectrum_amplitude_bytes(),
            counts().spectrum_texels * SPECTRUM_AMPLITUDE_STRIDE
        );
    }

    #[test]
    fn total_bytes_double_counts_the_ping_pong_pools() {
        let set = WaterPersistentBufferSet::new(counts());
        let expected = set.spectrum_amplitude_bytes() * 2
            + set.displacement_bytes()
            + set.normal_bytes()
            + set.gerstner_bytes()
            + set.swe_bytes()
            + set.pbf_particle_bytes() * 2
            + set.pbf_hash_bytes()
            + set.flip_particle_bytes()
            + set.flip_grid_bytes() * 2
            + set.foam_bytes() * 2
            + set.wetness_bytes()
            + set.froxel_bytes();
        assert_eq!(set.total_bytes(), expected);
    }

    #[test]
    fn pbf_pool_uses_vec4_stride() {
        assert_eq!(PBF_PARTICLE_STRIDE, 16);
        let set = WaterPersistentBufferSet::new(counts());
        assert_eq!(
            set.pbf_particle_bytes(),
            counts().pbf_particles * PBF_PARTICLE_STRIDE
        );
    }

    #[test]
    fn pathological_counts_saturate_rather_than_wrap() {
        let set = WaterPersistentBufferSet::new(WaterBufferCounts {
            spectrum_texels: u32::MAX,
            gerstner_waves: u32::MAX,
            swe_cells: u32::MAX,
            pbf_particles: u32::MAX,
            pbf_hash_entries: u32::MAX,
            flip_particles: u32::MAX,
            flip_grid_cells: u32::MAX,
            foam_cells: u32::MAX,
            wetness_cells: u32::MAX,
            froxels: u32::MAX,
        });
        assert_eq!(set.total_bytes(), u32::MAX);
    }

    #[test]
    fn empty_set_is_not_simulatable() {
        let set = WaterPersistentBufferSet::new(WaterBufferCounts::default());
        assert!(!set.is_simulatable());
        assert_eq!(set.total_bytes(), 0);
    }

    #[test]
    fn ocean_only_body_is_simulatable() {
        let set = WaterPersistentBufferSet::new(WaterBufferCounts {
            spectrum_texels: 256 * 256,
            ..WaterBufferCounts::default()
        });
        assert!(set.is_simulatable());
    }

    #[test]
    fn parity_swaps_and_indices_are_complementary() {
        let even = BufferParity::Even;
        assert_eq!(even.front_index(), 0);
        assert_eq!(even.back_index(), 1);
        let odd = even.swapped();
        assert_eq!(odd.front_index(), 1);
        assert_eq!(odd.back_index(), 0);
        assert_eq!(odd.swapped(), BufferParity::Even);
    }

    #[test]
    fn async_frame_state_clamps_to_double_buffer() {
        assert_eq!(AsyncFrameState::new(0).slot_count(), 2);
        assert_eq!(AsyncFrameState::new(1).slot_count(), 2);
        assert_eq!(AsyncFrameState::new(3).slot_count(), 3);
    }

    #[test]
    fn record_submit_signal_retire_round_trips() {
        let mut state = AsyncFrameState::new(2);
        let a = state.begin_record().expect("idle slot available");
        assert_eq!(state.slot(a).map(|s| s.state), Some(SlotState::Recording));
        state.submit(a).expect("recording slot submits");
        assert_eq!(state.in_flight(), 1);
        assert_eq!(state.parity(), BufferParity::Odd);
        state.signal_ready(a).expect("in-flight slot signals");
        state.retire(a).expect("ready slot retires");
        assert_eq!(state.in_flight(), 0);
        assert_eq!(state.retired(), 1);
    }

    #[test]
    fn saturated_ring_reports_no_idle_slot() {
        let mut state = AsyncFrameState::new(2);
        let a = state.begin_record().expect("first idle");
        state.submit(a).expect("submit first");
        let b = state.begin_record().expect("second idle");
        state.submit(b).expect("submit second");
        assert_eq!(state.begin_record(), Err(PipelineError::NoIdleSlot));
    }

    #[test]
    fn illegal_transitions_are_rejected() {
        let mut state = AsyncFrameState::new(2);
        // Cannot submit a slot that is not recording.
        assert_eq!(state.submit(0), Err(PipelineError::IllegalTransition));
        // Out-of-range slot addressing is bounded.
        assert_eq!(state.signal_ready(99), Err(PipelineError::SlotOutOfRange));
        let a = state.begin_record().expect("idle");
        // Cannot retire before ready.
        assert_eq!(state.retire(a), Err(PipelineError::IllegalTransition));
    }
}
