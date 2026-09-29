//! Persistent `GPU` buffer bookkeeping and the async double-buffer state
//! machine for the cloth solver.
//!
//! A production `GPU`-driven cloth engine keeps its simulation state resident on
//! the device across frames rather than re-uploading it: the particle
//! position/velocity pools, the graph-colored constraint arrays, the
//! self-collision spatial-hash buffers, and the render-mesh embed weights all
//! live in persistent storage buffers, and the solver dispatches read and write
//! them in place. Only the small per-substep uniform block is streamed.
//!
//! This module owns the *sizing and lifecycle contract* for those buffers: how
//! many elements and bytes each resident buffer needs (saturating integer
//! bookkeeping, so a pathological count can never overflow the allocation size
//! the backend reads), which buffers are double-buffered so the solver can read
//! the previous frame while writing the next, and the async state machine that
//! sequences record → submit → retire without ever reading a slot the `GPU` is
//! still writing. It holds no `GPU` handles and does no allocation itself; the
//! backend consumes the byte counts to make the real allocations.

use alloc::vec::Vec;

/// The byte size of one packed `vec3<f32>` padded to 16 bytes, matching the
/// `std430` alignment the `WESL` storage buffers use for a position or velocity
/// element. Padding to 16 keeps the stride aligned for the `GPU`.
pub const PARTICLE_VEC_STRIDE: u32 = 16;

/// The byte size of one packed distance/bending constraint record (two or four
/// `u32` indices plus a packed rest-length and compliance `f32`), rounded up to
/// the 16-byte `std430` stride the constraint storage buffer uses.
pub const CONSTRAINT_STRIDE: u32 = 16;

/// The byte size of one spatial-hash cell header (a start offset plus an
/// occupant count, two `u32`), padded to 8 bytes.
pub const HASH_CELL_STRIDE: u32 = 8;

/// The byte size of one render-mesh embed record, padded to the 32-byte
/// `std430` stride.
///
/// Mirrors the CPU golden `super::super::embed::BarycentricBinding`: three host
/// sim-triangle indices (`tri: [u32; 3]`), three barycentric weights
/// (`bary: (f32, f32, f32)`), and a signed face-normal offset
/// (`normal_offset: f32`) that restores garment thickness — seven 4-byte words
/// (28 bytes) rounded up to the next 16-byte-aligned `std430` stride. The GPU
/// twin `cloth_embed.wesl` binds the record at exactly this stride (a matching
/// `ClothEmbedBinding` with one trailing pad word), so the host allocation and
/// the shader agree.
pub const EMBED_STRIDE: u32 = 32;

/// The byte size of one packed painted-backstop record, padded to the 32-byte
/// `std430` stride.
///
/// Mirrors the CPU golden `super::super::collision::Backstop`: an anchor
/// `origin: Vec3`, an outward plane `normal: Vec3`, and a scalar `distance:
/// f32` — seven 4-byte words. The `WESL` twin `cloth_backstop` packs them as
/// two 16-byte-aligned `vec4` slots (`origin.xyz + distance`, `normal.xyz +
/// pad`), so the host allocation and the shader agree on a 32-byte stride.
pub const BACKSTOP_STRIDE: u32 = 32;

/// The resident element counts that size every persistent cloth buffer for one
/// piece.
///
/// Each field is an element count (particles, constraints, hash cells, render
/// vertices), not a byte size; [`PersistentBufferSet`] turns them into byte
/// sizes with the packed strides above. All arithmetic downstream is saturating
/// so an adversarial count clamps to `u32::MAX` bytes rather than wrapping.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct BufferCounts {
    /// Number of sim-mesh particles (positions, velocities, predicted
    /// positions, inverse masses).
    pub particles: u32,
    /// Total number of constraints across every graph color.
    pub constraints: u32,
    /// Number of spatial-hash cells the self-collision grid covers.
    pub hash_cells: u32,
    /// Number of render-mesh vertices bound to the sim mesh by embedding.
    pub render_vertices: u32,
    /// Number of painted-backstop records (one per constrained particle). A
    /// piece with no painted backstops leaves this zero and skips the pass.
    pub backstops: u32,
}

/// The persistent, device-resident buffer set for one cloth piece.
///
/// Built once from a [`BufferCounts`] and re-used every frame. The position
/// buffer is double-buffered (see [`Self::position_bytes`] returns the size of
/// *one* of the two copies) so the solver can integrate the next frame from the
/// previous frame's positions without a hazard; the other buffers are single
/// copies mutated in place.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct PersistentBufferSet {
    counts: BufferCounts,
}

impl PersistentBufferSet {
    /// Builds the buffer set for the given resident element counts.
    #[must_use]
    pub fn new(counts: BufferCounts) -> Self {
        Self { counts }
    }

    /// The element counts backing this set.
    #[must_use]
    pub fn counts(self) -> BufferCounts {
        self.counts
    }

    /// Bytes for one copy of the particle position buffer. Double-buffered, so
    /// the resident position storage costs twice this (see [`Self::total_bytes`]).
    #[must_use]
    pub fn position_bytes(self) -> u32 {
        self.counts.particles.saturating_mul(PARTICLE_VEC_STRIDE)
    }

    /// Bytes for the particle velocity buffer (single copy).
    #[must_use]
    pub fn velocity_bytes(self) -> u32 {
        self.counts.particles.saturating_mul(PARTICLE_VEC_STRIDE)
    }

    /// Bytes for the predicted-position scratch buffer used within a substep.
    #[must_use]
    pub fn predicted_bytes(self) -> u32 {
        self.counts.particles.saturating_mul(PARTICLE_VEC_STRIDE)
    }

    /// Bytes for the packed constraint buffer (all colors concatenated).
    #[must_use]
    pub fn constraint_bytes(self) -> u32 {
        self.counts.constraints.saturating_mul(CONSTRAINT_STRIDE)
    }

    /// Bytes for the self-collision hash cell-header buffer.
    #[must_use]
    pub fn hash_cell_bytes(self) -> u32 {
        self.counts.hash_cells.saturating_mul(HASH_CELL_STRIDE)
    }

    /// Bytes for the self-collision hash entry buffer (one `u32` particle index
    /// per particle, since every particle lands in exactly one cell).
    #[must_use]
    pub fn hash_entry_bytes(self) -> u32 {
        self.counts.particles.saturating_mul(4)
    }

    /// Bytes for the render-mesh embed-weight buffer.
    #[must_use]
    pub fn embed_bytes(self) -> u32 {
        self.counts.render_vertices.saturating_mul(EMBED_STRIDE)
    }

    /// Bytes for the painted-backstop record buffer.
    #[must_use]
    pub fn backstop_bytes(self) -> u32 {
        self.counts.backstops.saturating_mul(BACKSTOP_STRIDE)
    }

    /// Total resident bytes for every persistent buffer, counting the position
    /// buffer twice for double-buffering. Saturating.
    #[must_use]
    pub fn total_bytes(self) -> u32 {
        let doubled_position = self.position_bytes().saturating_mul(2);
        doubled_position
            .saturating_add(self.velocity_bytes())
            .saturating_add(self.predicted_bytes())
            .saturating_add(self.constraint_bytes())
            .saturating_add(self.hash_cell_bytes())
            .saturating_add(self.hash_entry_bytes())
            .saturating_add(self.embed_bytes())
            .saturating_add(self.backstop_bytes())
    }

    /// Returns `true` when the set has at least one particle and one
    /// constraint, i.e. it can actually simulate. An empty piece needs no
    /// resident allocation.
    #[must_use]
    pub fn is_simulatable(self) -> bool {
        self.counts.particles > 0 && self.counts.constraints > 0
    }
}

/// Which of the two position copies the solver currently treats as the readable
/// front buffer.
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

/// The lifecycle state of one async frame slot in the double-buffered
/// pipeline.
///
/// The solver records commands into an `Idle` slot (→ `Recording`), submits it
/// (→ `InFlight`), and once the `GPU` fence signals marks it `Ready` for its
/// results to be read, after which it returns to `Idle`. The transitions are
/// enforced by [`AsyncFrameState::advance`] so a slot is never read while the
/// `GPU` is still writing it.
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
/// recorded with, so a slot read back later knows which position copy holds its
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

/// The async double-buffer state machine for one cloth piece's `GPU` solve.
///
/// It owns a small ring of [`FrameSlot`]s (double- or triple-buffered) and the
/// live [`BufferParity`]. Each frame the `CPU` acquires an `Idle` slot, records
/// into it, submits it, and later retires it when its fence signals — swapping
/// the position parity exactly once per submitted frame so the next frame reads
/// the freshly written positions. All indices are bounded and every transition
/// is validated, so the machine can be exhaustively `CPU`-tested.
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
    /// submitted counter, and swaps the position parity so the next recorded
    /// frame reads the positions this frame will write.
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
        AsyncFrameState, BufferCounts, BufferParity, PersistentBufferSet, PipelineError, SlotState,
        PARTICLE_VEC_STRIDE,
    };

    fn counts() -> BufferCounts {
        BufferCounts {
            particles: 100,
            constraints: 400,
            hash_cells: 64,
            render_vertices: 500,
            backstops: 100,
        }
    }

    #[test]
    fn total_bytes_double_counts_position_and_saturates() {
        let set = PersistentBufferSet::new(counts());
        assert_eq!(set.position_bytes(), 100 * PARTICLE_VEC_STRIDE);
        // Position appears twice in the total.
        let expected = set.position_bytes() * 2
            + set.velocity_bytes()
            + set.predicted_bytes()
            + set.constraint_bytes()
            + set.hash_cell_bytes()
            + set.hash_entry_bytes()
            + set.embed_bytes()
            + set.backstop_bytes();
        assert_eq!(set.total_bytes(), expected);
    }

    #[test]
    fn pathological_counts_saturate_rather_than_wrap() {
        let set = PersistentBufferSet::new(BufferCounts {
            particles: u32::MAX,
            constraints: u32::MAX,
            hash_cells: u32::MAX,
            render_vertices: u32::MAX,
            backstops: u32::MAX,
        });
        assert_eq!(set.total_bytes(), u32::MAX);
    }

    #[test]
    fn empty_set_is_not_simulatable() {
        let set = PersistentBufferSet::new(BufferCounts::default());
        assert!(!set.is_simulatable());
        assert_eq!(set.total_bytes(), 0);
    }

    #[test]
    fn parity_swaps_and_indices_are_disjoint() {
        let even = BufferParity::Even;
        assert_eq!(even.front_index(), 0);
        assert_eq!(even.back_index(), 1);
        let odd = even.swapped();
        assert_eq!(odd.front_index(), 1);
        assert_eq!(odd.back_index(), 0);
        assert_eq!(odd.swapped(), even);
    }

    #[test]
    fn async_lifecycle_round_trips_through_every_state() {
        let mut state = AsyncFrameState::new(2);
        assert_eq!(state.slot_count(), 2);
        assert_eq!(state.parity(), BufferParity::Even);

        let slot = state.begin_record().unwrap();
        assert_eq!(state.slot(slot).unwrap().state, SlotState::Recording);
        assert_eq!(state.slot(slot).unwrap().parity, BufferParity::Even);

        state.submit(slot).unwrap();
        assert_eq!(state.slot(slot).unwrap().state, SlotState::InFlight);
        assert_eq!(state.submitted(), 1);
        // Parity swapped on submit.
        assert_eq!(state.parity(), BufferParity::Odd);
        assert_eq!(state.in_flight(), 1);

        state.signal_ready(slot).unwrap();
        assert_eq!(state.slot(slot).unwrap().state, SlotState::Ready);

        state.retire(slot).unwrap();
        assert_eq!(state.slot(slot).unwrap().state, SlotState::Idle);
        assert_eq!(state.retired(), 1);
        assert_eq!(state.in_flight(), 0);
    }

    #[test]
    fn double_buffering_keeps_two_frames_in_flight() {
        let mut state = AsyncFrameState::new(2);
        let a = state.begin_record().unwrap();
        state.submit(a).unwrap();
        let b = state.begin_record().unwrap();
        state.submit(b).unwrap();
        assert_ne!(a, b);
        assert_eq!(state.in_flight(), 2);
        // Both slots busy: the third record must wait.
        assert_eq!(state.begin_record(), Err(PipelineError::NoIdleSlot));
    }

    #[test]
    fn illegal_transitions_are_rejected() {
        let mut state = AsyncFrameState::new(2);
        // Submitting a slot that is not recording.
        assert_eq!(state.submit(0), Err(PipelineError::IllegalTransition));
        // Out-of-range slot.
        assert_eq!(state.signal_ready(99), Err(PipelineError::SlotOutOfRange));
        let slot = state.begin_record().unwrap();
        // Retiring before ready.
        assert_eq!(state.retire(slot), Err(PipelineError::IllegalTransition));
    }

    #[test]
    fn slot_count_is_clamped_to_at_least_two() {
        assert_eq!(AsyncFrameState::new(0).slot_count(), 2);
        assert_eq!(AsyncFrameState::new(1).slot_count(), 2);
        assert_eq!(AsyncFrameState::new(3).slot_count(), 3);
    }

    #[test]
    fn parity_alternates_across_submitted_frames() {
        let mut state = AsyncFrameState::new(3);
        let mut seen_even = false;
        let mut seen_odd = false;
        for _ in 0..3 {
            let slot = state.begin_record().unwrap();
            match state.slot(slot).unwrap().parity {
                BufferParity::Even => seen_even = true,
                BufferParity::Odd => seen_odd = true,
            }
            state.submit(slot).unwrap();
            state.signal_ready(slot).unwrap();
            state.retire(slot).unwrap();
        }
        assert!(seen_even && seen_odd);
    }
}
