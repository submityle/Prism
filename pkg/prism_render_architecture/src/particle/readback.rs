//! `GPU`→`CPU` statistics readback contract for the particle subsystem
//! (design §9 pipeline stats, §13 culling counters).
//!
//! Production `GPU`-driven VFX engines never trust the `CPU` to know how many
//! particles survived a frame: the counts are produced on the device by the
//! spawn, simulation, event-scatter and cull passes, written into a small
//! `std430` counter record, and copied back to a `CPU`-visible staging buffer.
//! Because the copy cannot stall the pipeline, the readback is *ring-buffered*
//! and therefore *latent*: frame `N`'s statistics only become legible several
//! frames later, exactly like `Niagara`'s `GPU` sim-stats readback and
//! `Frostbite`'s FX statistics path.
//!
//! This module owns the pure, device-free half of that contract:
//!
//! 1. [`ParticleStatField`] — the fixed set of `u32` counters the `GPU` writes,
//!    with their `std430` byte offsets inside one tightly packed record.
//! 2. [`stat_record_bytes`] — the byte size of one counter record, reusing the
//!    shared clamp-to-one [`storage_bytes`] rule so an empty record is still a
//!    valid `WebGPU` binding.
//! 3. [`ReadbackRing`] — the frames-in-flight ring that maps a frame index to
//!    its write slot and, after the latency window, to its readable slot, plus
//!    the total staging byte budget.
//! 4. [`reduce_partials`] — the reference `CPU` reduction that folds the
//!    per-workgroup partial counters the `GPU` scatter produces into one record.
//! 5. [`StatSnapshot`] — a reduced record with the derived accessors the render
//!    graph reports (total culled, alive after cull).
//!
//! Everything is deterministic `u32` counting: every accumulation is
//! `saturating_*`, no index can ever be out of bounds, and nothing panics on a
//! degenerate ring size or empty partial list. The record is a dense `SoA`-free
//! array of scalar counters, so no attribute widths are re-derived here.

use super::gpu_layout::{storage_bytes, ParticleBufferAccess, U32_STRIDE};

/// Upper bound on how many frames of statistics the ring keeps in flight.
///
/// Three to four frames covers every realistic `CPU`/`GPU` pipeline depth; the
/// ceiling keeps the staging allocation bounded even if a caller passes a wild
/// value. The floor of one is enforced separately so the ring is always usable.
pub const MAX_FRAMES_IN_FLIGHT: u32 = 8;

/// A single `u32` counter the `GPU` statistics pass writes for a frame.
///
/// The discriminant order is the `std430` record order: field *i* lives at byte
/// offset `i * U32_STRIDE` inside the tightly packed record, so the enum doubles
/// as the shader-side struct layout. New counters must be appended, never
/// inserted, to keep the `ABI` stable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParticleStatField {
    /// Particles still alive at the end of the frame's simulation.
    AliveCount,
    /// Particles spawned this frame by the rate/burst emission passes.
    SpawnedThisFrame,
    /// Particles killed this frame by age-out, events, or collision.
    KilledThisFrame,
    /// Particles culled because their bounds fell outside the view frustum.
    CulledFrustum,
    /// Particles culled because they exceeded the distance/`LOD` cutoff.
    CulledDistance,
    /// Particles culled by the hierarchical-Z (`HZB`) occlusion test.
    CulledHzb,
    /// Gameplay/spawn events the event-scatter pass recorded this frame.
    EventCount,
}

impl ParticleStatField {
    /// Every counter in `std430` record order.
    ///
    /// The array length is the authoritative field count; [`STAT_FIELD_COUNT`]
    /// is derived from it so the two can never disagree.
    pub const ALL: [Self; 7] = [
        Self::AliveCount,
        Self::SpawnedThisFrame,
        Self::KilledThisFrame,
        Self::CulledFrustum,
        Self::CulledDistance,
        Self::CulledHzb,
        Self::EventCount,
    ];

    /// Zero-based slot of this counter inside a reduced record.
    #[must_use]
    pub fn index(self) -> usize {
        match self {
            Self::AliveCount => 0,
            Self::SpawnedThisFrame => 1,
            Self::KilledThisFrame => 2,
            Self::CulledFrustum => 3,
            Self::CulledDistance => 4,
            Self::CulledHzb => 5,
            Self::EventCount => 6,
        }
    }

    /// `std430` byte offset of this counter inside one packed record.
    #[must_use]
    pub fn byte_offset(self) -> usize {
        self.index() * U32_STRIDE
    }
}

/// Number of `u32` counters in one statistics record.
pub const STAT_FIELD_COUNT: usize = ParticleStatField::ALL.len();

/// Host-side access mode of the readback record.
///
/// The device produces the counters and the `CPU` only ever reads the staged
/// copy, so from the host's perspective the record binds read-only.
#[must_use]
pub const fn readback_access() -> ParticleBufferAccess {
    ParticleBufferAccess::Read
}

/// Byte size of one tightly packed `std430` statistics record.
///
/// Routed through [`storage_bytes`] so it inherits the shared clamp-to-one rule:
/// even a hypothetical zero-field record reserves one element and stays a legal
/// `WebGPU` storage binding.
#[must_use]
pub fn stat_record_bytes() -> usize {
    storage_bytes(U32_STRIDE, STAT_FIELD_COUNT)
}

/// Ring buffer mapping frame indices to their `GPU`→`CPU` readback slots.
///
/// The `GPU` writes frame `N`'s record into slot `N % frames_in_flight`; the
/// `CPU` may only read a slot once the copy for that frame has certainly
/// completed, which is `frames_in_flight` frames later. Early frames therefore
/// have nothing to read yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReadbackRing {
    /// Number of frames the ring keeps in flight, clamped to
    /// `[1, MAX_FRAMES_IN_FLIGHT]` at construction.
    frames_in_flight: u32,
}

impl ReadbackRing {
    /// Build a ring, clamping `frames_in_flight` into
    /// `[1, MAX_FRAMES_IN_FLIGHT]`.
    ///
    /// A requested depth of zero would make the ring unusable (no slots), so it
    /// is clamped up to one; anything above the ceiling is clamped down.
    #[must_use]
    pub fn new(frames_in_flight: u32) -> Self {
        Self {
            frames_in_flight: frames_in_flight.clamp(1, MAX_FRAMES_IN_FLIGHT),
        }
    }

    /// Number of frames this ring keeps in flight (already clamped).
    #[must_use]
    pub fn frames_in_flight(self) -> u32 {
        self.frames_in_flight
    }

    /// Slot the `GPU` writes frame `frame_index`'s record into.
    ///
    /// Never panics: `frames_in_flight` is guaranteed non-zero, so the modulo is
    /// always defined, and the result is a valid slot in `[0, frames)`.
    #[must_use]
    pub fn write_slot(self, frame_index: u64) -> usize {
        (frame_index % u64::from(self.frames_in_flight)) as usize
    }

    /// Slot holding the newest record the `CPU` may safely read this frame, or
    /// `None` while still inside the initial latency window.
    ///
    /// Frame `frame_index`'s readable record was produced `frames_in_flight`
    /// frames ago; before that many frames have elapsed there is no completed
    /// copy yet, so the caller must not read stale garbage.
    #[must_use]
    pub fn read_slot(self, frame_index: u64) -> Option<usize> {
        let latency = u64::from(self.frames_in_flight);
        frame_index
            .checked_sub(latency)
            .map(|readable_frame| self.write_slot(readable_frame))
    }

    /// Total staging byte budget: one record per in-flight frame.
    ///
    /// Uses [`storage_bytes`] so the whole staging region also honours the
    /// clamp-to-one rule and saturates rather than overflowing.
    #[must_use]
    pub fn staging_bytes(self) -> usize {
        storage_bytes(stat_record_bytes(), self.frames_in_flight as usize)
    }
}

/// Fold per-workgroup partial counter records into one reduced record.
///
/// The `GPU` cull/scatter passes emit one partial counter array per workgroup;
/// this is the reference `CPU` reduction the readback validates against. Each
/// field is summed with `saturating_add` so a pathological partial set clamps at
/// `u32::MAX` instead of wrapping. An empty input reduces to all zeroes.
#[must_use]
pub fn reduce_partials(partials: &[[u32; STAT_FIELD_COUNT]]) -> [u32; STAT_FIELD_COUNT] {
    let mut reduced = [0u32; STAT_FIELD_COUNT];
    for partial in partials {
        for (slot, &value) in reduced.iter_mut().zip(partial.iter()) {
            *slot = slot.saturating_add(value);
        }
    }
    reduced
}

/// A reduced statistics record with the derived accessors the render graph
/// reports back to gameplay and profiling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StatSnapshot {
    /// One reduced `u32` per [`ParticleStatField`], in record order.
    counters: [u32; STAT_FIELD_COUNT],
}

impl StatSnapshot {
    /// Wrap an already-reduced counter record.
    #[must_use]
    pub fn new(counters: [u32; STAT_FIELD_COUNT]) -> Self {
        Self { counters }
    }

    /// Reduce a set of per-workgroup partials directly into a snapshot.
    #[must_use]
    pub fn from_partials(partials: &[[u32; STAT_FIELD_COUNT]]) -> Self {
        Self::new(reduce_partials(partials))
    }

    /// Read one counter by field.
    #[must_use]
    pub fn get(&self, field: ParticleStatField) -> u32 {
        self.counters[field.index()]
    }

    /// Sum of all three culling counters, saturating at `u32::MAX`.
    #[must_use]
    pub fn total_culled(&self) -> u32 {
        self.get(ParticleStatField::CulledFrustum)
            .saturating_add(self.get(ParticleStatField::CulledDistance))
            .saturating_add(self.get(ParticleStatField::CulledHzb))
    }

    /// Alive particles remaining after culling, clamped at zero on underflow.
    ///
    /// A frame where the reported cull count exceeds the alive count (possible
    /// only from inconsistent partials) yields zero rather than wrapping.
    #[must_use]
    pub fn alive_after_cull(&self) -> u32 {
        self.get(ParticleStatField::AliveCount)
            .saturating_sub(self.total_culled())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_ordering_is_dense_and_non_overlapping() {
        assert_eq!(ParticleStatField::ALL.len(), STAT_FIELD_COUNT);
        for (expected, field) in ParticleStatField::ALL.iter().enumerate() {
            assert_eq!(field.index(), expected);
            assert_eq!(field.byte_offset(), expected * U32_STRIDE);
        }
        // Offsets are strictly monotonic in exact stride steps, so no two
        // counters share a byte and there is no padding gap.
        let mut previous: Option<usize> = None;
        for field in ParticleStatField::ALL {
            if let Some(prev) = previous {
                assert_eq!(field.byte_offset(), prev + U32_STRIDE);
            }
            previous = Some(field.byte_offset());
        }
    }

    #[test]
    fn record_bytes_is_packed() {
        assert_eq!(stat_record_bytes(), STAT_FIELD_COUNT * U32_STRIDE);
    }

    #[test]
    fn readback_record_is_read_only() {
        assert!(!readback_access().is_writable());
    }

    #[test]
    fn reduce_empty_partials_is_all_zero() {
        assert_eq!(reduce_partials(&[]), [0u32; STAT_FIELD_COUNT]);
    }

    #[test]
    fn reduce_single_workgroup_is_identity() {
        let mut partial = [0u32; STAT_FIELD_COUNT];
        for (i, slot) in partial.iter_mut().enumerate() {
            *slot = i as u32 + 1;
        }
        assert_eq!(reduce_partials(&[partial]), partial);
    }

    #[test]
    fn reduce_multiple_workgroups_sums_per_field() {
        let a = [1, 2, 3, 4, 5, 6, 7];
        let b = [10, 20, 30, 40, 50, 60, 70];
        let c = [100, 200, 300, 400, 500, 600, 700];
        let reduced = reduce_partials(&[a, b, c]);
        assert_eq!(reduced, [111, 222, 333, 444, 555, 666, 777]);
    }

    #[test]
    fn reduce_saturates_near_u32_max() {
        let big = [u32::MAX; STAT_FIELD_COUNT];
        let one = [1u32; STAT_FIELD_COUNT];
        // MAX + MAX + 1 would wrap; saturating keeps it pinned at MAX.
        assert_eq!(
            reduce_partials(&[big, big, one]),
            [u32::MAX; STAT_FIELD_COUNT],
        );
    }

    #[test]
    fn ring_clamps_zero_to_one() {
        assert_eq!(ReadbackRing::new(0).frames_in_flight(), 1);
    }

    #[test]
    fn ring_clamps_above_ceiling() {
        assert_eq!(
            ReadbackRing::new(1_000).frames_in_flight(),
            MAX_FRAMES_IN_FLIGHT,
        );
    }

    #[test]
    fn write_slot_wraps_around_ring() {
        let ring = ReadbackRing::new(3);
        assert_eq!(ring.write_slot(0), 0);
        assert_eq!(ring.write_slot(1), 1);
        assert_eq!(ring.write_slot(2), 2);
        assert_eq!(ring.write_slot(3), 0);
        assert_eq!(ring.write_slot(7), 1);
    }

    #[test]
    fn read_slot_is_none_inside_latency_window() {
        let ring = ReadbackRing::new(3);
        assert_eq!(ring.read_slot(0), None);
        assert_eq!(ring.read_slot(1), None);
        assert_eq!(ring.read_slot(2), None);
    }

    #[test]
    fn read_slot_trails_write_slot_by_latency() {
        let ring = ReadbackRing::new(3);
        // Frame 3 reads frame 0's slot, frame 4 reads frame 1's slot, etc.
        assert_eq!(ring.read_slot(3), Some(ring.write_slot(0)));
        assert_eq!(ring.read_slot(4), Some(ring.write_slot(1)));
        assert_eq!(ring.read_slot(10), Some(ring.write_slot(7)));
    }

    #[test]
    fn single_frame_ring_reads_previous_frame() {
        let ring = ReadbackRing::new(1);
        assert_eq!(ring.write_slot(42), 0);
        assert_eq!(ring.read_slot(0), None);
        assert_eq!(ring.read_slot(1), Some(0));
    }

    #[test]
    fn staging_bytes_is_record_times_frames() {
        let ring = ReadbackRing::new(4);
        assert_eq!(ring.staging_bytes(), stat_record_bytes() * 4);
    }

    #[test]
    fn snapshot_get_returns_each_field() {
        let counters = [11, 22, 33, 44, 55, 66, 77];
        let snap = StatSnapshot::new(counters);
        for field in ParticleStatField::ALL {
            assert_eq!(snap.get(field), counters[field.index()]);
        }
    }

    #[test]
    fn snapshot_total_culled_sums_three_cull_fields() {
        let snap = StatSnapshot::new([100, 0, 0, 4, 5, 6, 0]);
        assert_eq!(snap.total_culled(), 15);
        assert_eq!(snap.alive_after_cull(), 85);
    }

    #[test]
    fn snapshot_total_culled_saturates() {
        let snap = StatSnapshot::new([0, 0, 0, u32::MAX, u32::MAX, 1, 0]);
        assert_eq!(snap.total_culled(), u32::MAX);
    }

    #[test]
    fn snapshot_alive_after_cull_clamps_underflow_to_zero() {
        let snap = StatSnapshot::new([10, 0, 0, 20, 0, 0, 0]);
        assert_eq!(snap.alive_after_cull(), 0);
    }

    #[test]
    fn snapshot_from_partials_matches_manual_reduction() {
        let a = [5, 1, 1, 2, 0, 0, 3];
        let b = [7, 2, 0, 1, 1, 0, 4];
        let snap = StatSnapshot::from_partials(&[a, b]);
        assert_eq!(snap.get(ParticleStatField::AliveCount), 12);
        assert_eq!(snap.total_culled(), 4);
        assert_eq!(snap.alive_after_cull(), 8);
    }
}
