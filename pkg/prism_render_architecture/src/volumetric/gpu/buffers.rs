//! Persistent `GPU` buffer bookkeeping and the async double-buffer state
//! machine for the volumetric cloud/atmosphere solve (design section 17,
//! milestone M8).
//!
//! A `GPU`-driven volumetric engine keeps its heavy state resident on the
//! device across frames rather than re-uploading it every frame: the baked 3D
//! density cache the ray-march samples, the evolving weather map, the low-res
//! scattering/transmittance target, the reprojection history, the light-space
//! cloud-shadow (`AVSM`) map, and the small pre-integrated multiple-scatter
//! `LUT` all live in persistent storage and the compute passes read and write
//! them in place. Only the tiny per-frame uniform block (camera, sun, wind,
//! step policy) is streamed.
//!
//! This module owns the *sizing and lifecycle contract* for those resources:
//! how many bytes each resident buffer needs (saturating integer bookkeeping,
//! so a pathological element count clamps to `u32::MAX` rather than wrapping
//! the allocation size the backend reads), which buffer is double-buffered so
//! the temporal reprojection can read the previous frame while writing the
//! next, and the async state machine that sequences record -> submit -> retire
//! without ever reading a slot the `GPU` is still writing. It holds no `GPU`
//! handles and does no allocation itself; the backend consumes the byte counts
//! to make the real allocations.
//!
//! **Not machine-verified.** The sandbox has no `GPU`; the strides below are
//! the `std430`-aligned design targets the `WESL` kernels anticipate and must
//! be re-checked against the real backend layout once it lands.

use alloc::vec::Vec;

/// Bytes per baked density-cache voxel: a single scalar density plus a packed
/// lighting/type byte quad, stored as one `f32` (`4` bytes) in the design
/// target. The ray-march samples this cache trilinearly.
pub const DENSITY_VOXEL_STRIDE: u32 = 4;

/// Bytes per weather-map texel: coverage, cloud-type, precipitation and
/// wetness packed as an `RGBA8` quad (`4` bytes).
pub const WEATHER_TEXEL_STRIDE: u32 = 4;

/// Bytes per low-resolution ray-march tile sample: scattering `RGB` plus
/// transmittance packed as an `rgba16f` quad (`8` bytes).
pub const RAYMARCH_TILE_STRIDE: u32 = 8;

/// Bytes per full-resolution reprojection-history pixel: resolved scattering
/// `RGB` plus transmittance as an `rgba16f` quad (`8` bytes).
pub const HISTORY_PIXEL_STRIDE: u32 = 8;

/// Bytes per light-space cloud-shadow texel: the `AVSM` deep-shadow curve, a
/// fixed four-node `(depth, transmittance)` `f32` array (`32` bytes).
pub const SHADOW_TEXEL_STRIDE: u32 = 32;

/// Bytes per multiple-scatter `LUT` cell: pre-integrated scattering `RGB` plus
/// an energy-normalisation term as an `rgba16f` quad (`8` bytes).
pub const MULTISCATTER_CELL_STRIDE: u32 = 8;

/// The resident element counts that size every persistent volumetric buffer for
/// one cloud domain.
///
/// Each field is an element count (voxels, texels, tiles, pixels, cells), not a
/// byte size; [`PersistentBufferSet`] turns them into byte sizes with the
/// packed strides above. All arithmetic downstream is saturating so an
/// adversarial count clamps to `u32::MAX` bytes rather than wrapping.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct BufferCounts {
    /// Total voxels in the 3D density cache (resolution cubed for a cube
    /// domain, or `x * y * z` for a box domain).
    pub density_voxels: u32,
    /// Weather-map texels (the 2D coverage/type/precip field).
    pub weather_texels: u32,
    /// Low-resolution ray-march tiles (quarter/checkerboard view integration).
    pub raymarch_tiles: u32,
    /// Full-resolution reprojection-history pixels (also the upsample target
    /// extent).
    pub history_pixels: u32,
    /// Light-space cloud-shadow (`AVSM`) texels.
    pub shadow_texels: u32,
    /// Multiple-scatter `LUT` cells (cosine x optical-depth x albedo).
    pub multiscatter_cells: u32,
}

/// The persistent, device-resident buffer set for one cloud domain.
///
/// Built once from a [`BufferCounts`] and re-used every frame. The history
/// buffer is double-buffered (see [`Self::history_bytes`] returns the size of
/// *one* of the two copies) so the temporal reprojection can read the previous
/// frame's resolve while writing the next without a hazard; every other buffer
/// is a single copy mutated in place.
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

    /// Bytes for the baked 3D density cache (single copy).
    #[must_use]
    pub fn density_bytes(self) -> u32 {
        self.counts
            .density_voxels
            .saturating_mul(DENSITY_VOXEL_STRIDE)
    }

    /// Bytes for the weather map (single copy).
    #[must_use]
    pub fn weather_bytes(self) -> u32 {
        self.counts
            .weather_texels
            .saturating_mul(WEATHER_TEXEL_STRIDE)
    }

    /// Bytes for the low-resolution ray-march scattering/transmittance target
    /// (single copy).
    #[must_use]
    pub fn raymarch_bytes(self) -> u32 {
        self.counts
            .raymarch_tiles
            .saturating_mul(RAYMARCH_TILE_STRIDE)
    }

    /// Bytes for one copy of the reprojection-history buffer. Double-buffered,
    /// so the resident history storage costs twice this (see
    /// [`Self::total_bytes`]).
    #[must_use]
    pub fn history_bytes(self) -> u32 {
        self.counts
            .history_pixels
            .saturating_mul(HISTORY_PIXEL_STRIDE)
    }

    /// Bytes for the light-space cloud-shadow (`AVSM`) map (single copy).
    #[must_use]
    pub fn shadow_bytes(self) -> u32 {
        self.counts
            .shadow_texels
            .saturating_mul(SHADOW_TEXEL_STRIDE)
    }

    /// Bytes for the pre-integrated multiple-scatter `LUT` (single copy).
    #[must_use]
    pub fn multiscatter_bytes(self) -> u32 {
        self.counts
            .multiscatter_cells
            .saturating_mul(MULTISCATTER_CELL_STRIDE)
    }

    /// Total resident bytes across every buffer, counting the history buffer
    /// twice for double-buffering. Saturating.
    #[must_use]
    pub fn total_bytes(self) -> u32 {
        let doubled_history = self.history_bytes().saturating_mul(2);
        doubled_history
            .saturating_add(self.density_bytes())
            .saturating_add(self.weather_bytes())
            .saturating_add(self.raymarch_bytes())
            .saturating_add(self.shadow_bytes())
            .saturating_add(self.multiscatter_bytes())
    }

    /// Returns `true` when the domain has at least one density voxel and one
    /// ray-march tile, i.e. it can actually render. An empty domain needs no
    /// resident allocation.
    #[must_use]
    pub fn is_renderable(self) -> bool {
        self.counts.density_voxels > 0 && self.counts.raymarch_tiles > 0
    }
}

/// Which of the two reprojection-history copies the resolve currently treats as
/// the readable previous-frame buffer.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BufferParity {
    /// Copy 0 is the readable previous-frame buffer; copy 1 is the write target.
    Even,
    /// Copy 1 is the readable previous-frame buffer; copy 0 is the write target.
    Odd,
}

impl BufferParity {
    /// The index (0 or 1) of the readable previous-frame buffer.
    #[must_use]
    pub fn front_index(self) -> u32 {
        match self {
            BufferParity::Even => 0,
            BufferParity::Odd => 1,
        }
    }

    /// The index (0 or 1) of the write-target buffer for this frame's resolve.
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

/// The lifecycle state of one async frame slot in the double-buffered pipeline.
///
/// The renderer records dispatches into an `Idle` slot (-> `Recording`),
/// submits it (-> `InFlight`), and once the `GPU` fence signals marks it
/// `Ready` for its resolve to be read back, after which it returns to `Idle`.
/// The transitions are enforced so a slot's history is never sampled while the
/// `GPU` is still writing it.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SlotState {
    /// Free to record into.
    Idle,
    /// Currently recording compute dispatches on the `CPU` timeline.
    Recording,
    /// Submitted to the `GPU`; its fence has not yet signalled.
    InFlight,
    /// The `GPU` fence signalled; the resolve is safe to read, then recycle.
    Ready,
}

/// A single async frame slot: its lifecycle state and the buffer parity it was
/// recorded with, so a slot read back later knows which history copy holds its
/// resolve.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FrameSlot {
    /// The lifecycle state of the slot.
    pub state: SlotState,
    /// The buffer parity this slot resolved with.
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

/// The async double-buffer state machine for one cloud domain's `GPU` solve.
///
/// It owns a small ring of [`FrameSlot`]s (double- or triple-buffered) and the
/// live [`BufferParity`]. Each frame the `CPU` acquires an `Idle` slot, records
/// into it, submits it, and later retires it when its fence signals — swapping
/// the history parity exactly once per submitted frame so the next frame reads
/// the freshly resolved history. All indices are bounded and every transition
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

    /// The live buffer parity the next recording will resolve with.
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

    /// Submits a `Recording` slot to the `GPU` (-> `InFlight`), increments the
    /// submitted counter, and swaps the history parity so the next recorded
    /// frame reads the history this frame will resolve into.
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
    /// resolve is then safe to read.
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

    /// Retires a `Ready` slot back to `Idle` after its resolve is read,
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
        DENSITY_VOXEL_STRIDE,
    };

    fn counts() -> BufferCounts {
        BufferCounts {
            density_voxels: 1000,
            weather_texels: 256,
            raymarch_tiles: 480,
            history_pixels: 1920,
            shadow_texels: 512,
            multiscatter_cells: 64,
        }
    }

    #[test]
    fn total_bytes_double_counts_history_and_sums_the_rest() {
        let set = PersistentBufferSet::new(counts());
        assert_eq!(set.density_bytes(), 1000 * DENSITY_VOXEL_STRIDE);
        let expected = set.history_bytes() * 2
            + set.density_bytes()
            + set.weather_bytes()
            + set.raymarch_bytes()
            + set.shadow_bytes()
            + set.multiscatter_bytes();
        assert_eq!(set.total_bytes(), expected);
    }

    #[test]
    fn pathological_counts_saturate_rather_than_wrap() {
        let set = PersistentBufferSet::new(BufferCounts {
            density_voxels: u32::MAX,
            weather_texels: u32::MAX,
            raymarch_tiles: u32::MAX,
            history_pixels: u32::MAX,
            shadow_texels: u32::MAX,
            multiscatter_cells: u32::MAX,
        });
        assert_eq!(set.total_bytes(), u32::MAX);
    }

    #[test]
    fn empty_domain_is_not_renderable() {
        let set = PersistentBufferSet::new(BufferCounts::default());
        assert!(!set.is_renderable());
        assert_eq!(set.total_bytes(), 0);
    }

    #[test]
    fn a_domain_without_tiles_is_not_renderable() {
        let set = PersistentBufferSet::new(BufferCounts {
            density_voxels: 10,
            raymarch_tiles: 0,
            ..BufferCounts::default()
        });
        assert!(!set.is_renderable());
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
        assert_eq!(state.begin_record(), Err(PipelineError::NoIdleSlot));
    }

    #[test]
    fn illegal_transitions_are_rejected() {
        let mut state = AsyncFrameState::new(2);
        assert_eq!(state.submit(0), Err(PipelineError::IllegalTransition));
        assert_eq!(state.signal_ready(99), Err(PipelineError::SlotOutOfRange));
        let slot = state.begin_record().unwrap();
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
