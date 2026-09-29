//! Device-free frame-in-flight residency contract for the hair `GPU` pipeline.
//!
//! [`gpu_buffers`](super::gpu_buffers) and [`interp_buffers`](super::interp_buffers)
//! publish the byte layout of the guide-`XPBD` sim and the guide-to-render
//! resolve; [`frame_pass_layout`](super::frame_pass_layout) sequences the passes
//! of a single frame. This module publishes the missing *temporal* half of the
//! §8 GPU-driven persistence boundary: how those buffers are replicated across
//! frames that are in flight at once so the render graph can hand the previous
//! frame's deformed guides and render points downstream while the next frame's
//! solve and resolve run, without either stomping the other's memory.
//!
//! The scene/render crate owns the actual wgpu `Buffer` ring and the frame
//! fences; sizing that ring (how much `VRAM` a groom costs at a given pipeline
//! depth, and which copy each frame reads and writes) must agree with the ABI
//! declared next door, so — exactly as the per-pass layouts live once here
//! rather than being hand-derived next to the pipeline — the residency math
//! lives here in the zero-dependency crate too.
//!
//! ## Residency classes
//!
//! Every hair-owned storage buffer falls into one of two temporal classes:
//!
//! * **Frame-replicated handoff state** — the read-write buffers a frame
//!   *produces* and a completed frame *hands downstream*: the persistent guide
//!   state (`positions` + `prev_positions`, see
//!   [`gpu_buffers::persistent_state_bytes`](super::gpu_buffers::persistent_state_bytes))
//!   and the resolve output (`out_points`, see
//!   [`interp_buffers::output_bytes`](super::interp_buffers::output_bytes)).
//!   These are allocated once *per in-flight frame* so frame `F` writes its own
//!   copy while the render of frame `F - 1` reads the copy it produced. A depth
//!   of two is the classic double buffer; the solver's cross-frame `positions`
//!   ping-pong is subsumed by this same ring (a depth `>= 2` always gives the
//!   next solve a distinct copy from the one still being read).
//! * **Shared inputs** — the read-only inputs a frame *consumes*: the sim's
//!   `goals`, `rest_lengths`, `strands` and `colliders` (see
//!   [`gpu_buffers::upload_input_bytes`](super::gpu_buffers::upload_input_bytes))
//!   plus the resolve's static import tables `guide_ranges` and `bindings`.
//!   These are a single authoritative copy, refreshed or derived in place each
//!   frame. The resolve's `guide_points` input is deliberately *not* counted
//!   here: it aliases the persistent `positions` allocation already charged to
//!   the replicated class, so counting it again would double-bill the guide
//!   state.
//!
//! Everything is pure integer arithmetic: the pipeline depth clamps up to at
//! least one in-flight frame, byte totals reuse the already-clamped per-buffer
//! sizes so an empty groom still costs a valid non-empty allocation, and nothing
//! panics or divides by zero.

use crate::hair::gpu_buffers;
use crate::hair::gpu_dispatch::HairGpuCounts;
use crate::hair::interp_buffers::{self, HairInterpBuffer};

/// Total bytes of the resolve's static import inputs that need a single shared
/// allocation: `guide_ranges` and `bindings`.
///
/// `guide_points` is excluded on purpose — it aliases the persistent
/// `positions` buffer counted in the frame-replicated class, so folding it in
/// here would double-count the guide state.
#[must_use]
fn interp_static_input_bytes(counts: &HairGpuCounts, render_points: u32) -> usize {
    HairInterpBuffer::ALL
        .into_iter()
        .filter(|buffer| {
            matches!(
                buffer,
                HairInterpBuffer::GuideRanges | HairInterpBuffer::Bindings
            )
        })
        .map(|buffer| buffer.byte_size(counts, render_points))
        .sum()
}

/// How deep the hair `GPU` pipeline is: how many frames may be in flight at
/// once, and therefore how many copies of the frame-replicated handoff state
/// the render graph keeps in its ring.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HairFramePipeline {
    frames_in_flight: u32,
}

impl HairFramePipeline {
    /// A pipeline of the requested depth, clamped up to at least one in-flight
    /// frame (a depth of zero would leave no buffer to write and is treated as
    /// a single-buffered pipeline).
    #[must_use]
    pub fn new(frames_in_flight: u32) -> Self {
        Self {
            frames_in_flight: frames_in_flight.max(1),
        }
    }

    /// A single-buffered pipeline: one copy, solved and read in place with no
    /// cross-frame overlap.
    #[must_use]
    pub fn single_buffered() -> Self {
        Self::new(1)
    }

    /// A double-buffered pipeline: two copies, the classic depth that lets the
    /// render of the previous frame overlap the next frame's solve and resolve.
    #[must_use]
    pub fn double_buffered() -> Self {
        Self::new(2)
    }

    /// The clamped pipeline depth (always at least one).
    #[must_use]
    pub fn frames_in_flight(self) -> u32 {
        self.frames_in_flight
    }

    /// Whether at least two frames may overlap, so a completed frame's handoff
    /// state is read from a different copy than the next frame writes.
    #[must_use]
    pub fn is_double_buffered(self) -> bool {
        self.frames_in_flight >= 2
    }

    /// Which copy of the frame-replicated handoff state the given absolute
    /// `frame` index writes.
    #[must_use]
    pub fn write_index(self, frame: u64) -> u32 {
        (frame % u64::from(self.frames_in_flight)) as u32
    }

    /// Which copy holds the previous frame's completed handoff state that the
    /// render / downstream passes read while `frame` writes its own copy.
    ///
    /// For a single-buffered pipeline this equals [`write_index`](Self::write_index)
    /// (read-in-place, no overlap). For frame `0` the returned slot holds no
    /// valid prior state, so the first frame has nothing to hand downstream.
    #[must_use]
    pub fn read_index(self, frame: u64) -> u32 {
        let depth = u64::from(self.frames_in_flight);
        ((frame + depth - 1) % depth) as u32
    }

    /// Bytes of frame-replicated handoff state for *one* in-flight frame: the
    /// persistent guide sim state plus the resolve output.
    #[must_use]
    pub fn replicated_bytes_per_frame(
        counts: &HairGpuCounts,
        collider_count: u32,
        render_points: u32,
    ) -> usize {
        gpu_buffers::persistent_state_bytes(counts, collider_count)
            + interp_buffers::output_bytes(counts, render_points)
    }

    /// Bytes of the single shared-input allocation: the sim's read-only inputs
    /// plus the resolve's static import tables.
    #[must_use]
    pub fn shared_input_bytes(
        counts: &HairGpuCounts,
        collider_count: u32,
        render_points: u32,
    ) -> usize {
        gpu_buffers::upload_input_bytes(counts, collider_count)
            + interp_static_input_bytes(counts, render_points)
    }

    /// The full `VRAM` residency of a groom at this pipeline depth: the
    /// frame-replicated handoff state times the in-flight depth, plus the single
    /// shared-input allocation.
    #[must_use]
    pub fn plan_residency(
        self,
        counts: &HairGpuCounts,
        collider_count: u32,
        render_points: u32,
    ) -> HairFrameResidency {
        HairFrameResidency {
            frames_in_flight: self.frames_in_flight,
            replicated_bytes_per_frame: Self::replicated_bytes_per_frame(
                counts,
                collider_count,
                render_points,
            ),
            shared_input_bytes: Self::shared_input_bytes(counts, collider_count, render_points),
        }
    }
}

/// The resident `VRAM` cost of one groom at a fixed pipeline depth, split into
/// the class replicated per in-flight frame and the single shared-input class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HairFrameResidency {
    /// The pipeline depth this plan was sized for (always at least one).
    pub frames_in_flight: u32,
    /// Bytes of frame-replicated handoff state for a single in-flight frame
    /// (persistent guide state + resolve output).
    pub replicated_bytes_per_frame: usize,
    /// Bytes of the single shared-input allocation (sim inputs + static import
    /// tables), independent of the pipeline depth.
    pub shared_input_bytes: usize,
}

impl HairFrameResidency {
    /// Total bytes of frame-replicated handoff state across the whole ring
    /// (`replicated_bytes_per_frame * frames_in_flight`).
    #[must_use]
    pub fn replicated_bytes(&self) -> usize {
        self.replicated_bytes_per_frame * self.frames_in_flight as usize
    }

    /// The full resident cost: the whole replicated ring plus the shared inputs.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.replicated_bytes() + self.shared_input_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hair::gpu_dispatch::HairGpuCounts;

    fn sample_counts() -> HairGpuCounts {
        HairGpuCounts {
            roots: 10,
            guide_strands: 100,
            guide_particles: 3200,
            render_strands: 50_000,
            light_texels: 0,
        }
    }

    const COLLIDERS: u32 = 8;
    const RENDER_POINTS: u32 = 600_000;

    #[test]
    fn depth_clamps_up_to_one_frame() {
        assert_eq!(HairFramePipeline::new(0).frames_in_flight(), 1);
        assert_eq!(HairFramePipeline::new(1).frames_in_flight(), 1);
        assert_eq!(HairFramePipeline::new(3).frames_in_flight(), 3);
        assert_eq!(HairFramePipeline::single_buffered().frames_in_flight(), 1);
        assert_eq!(HairFramePipeline::double_buffered().frames_in_flight(), 2);
    }

    #[test]
    fn double_buffered_flag_tracks_depth() {
        assert!(!HairFramePipeline::single_buffered().is_double_buffered());
        assert!(HairFramePipeline::double_buffered().is_double_buffered());
        assert!(HairFramePipeline::new(3).is_double_buffered());
    }

    #[test]
    fn single_buffered_reads_and_writes_in_place() {
        let pipe = HairFramePipeline::single_buffered();
        for frame in 0..5u64 {
            assert_eq!(pipe.write_index(frame), 0);
            assert_eq!(pipe.read_index(frame), 0);
        }
    }

    #[test]
    fn double_buffered_ping_pongs_between_opposite_slots() {
        let pipe = HairFramePipeline::double_buffered();
        // Each frame writes one slot and reads the other.
        for frame in 0..6u64 {
            let write = pipe.write_index(frame);
            let read = pipe.read_index(frame);
            assert!(write < 2 && read < 2);
            assert_ne!(write, read);
            // This frame's read slot is the previous frame's write slot.
            if frame > 0 {
                assert_eq!(read, pipe.write_index(frame - 1));
            }
        }
    }

    #[test]
    fn triple_buffered_cycles_three_slots() {
        let pipe = HairFramePipeline::new(3);
        assert_eq!(
            [
                pipe.write_index(0),
                pipe.write_index(1),
                pipe.write_index(2),
                pipe.write_index(3),
            ],
            [0, 1, 2, 0]
        );
        // Read slot always trails the write slot by one in the ring.
        assert_eq!(pipe.read_index(0), 2);
        assert_eq!(pipe.read_index(1), 0);
        assert_eq!(pipe.read_index(2), 1);
    }

    #[test]
    fn replicated_bytes_are_persistent_state_plus_resolve_output() {
        let counts = sample_counts();
        let expected = gpu_buffers::persistent_state_bytes(&counts, COLLIDERS)
            + interp_buffers::output_bytes(&counts, RENDER_POINTS);
        assert_eq!(
            HairFramePipeline::replicated_bytes_per_frame(&counts, COLLIDERS, RENDER_POINTS),
            expected
        );
        // 2 * 3200 * 16 (positions + prev) + 600_000 * 16 (out_points).
        assert_eq!(expected, 2 * 3200 * 16 + 600_000 * 16);
    }

    #[test]
    fn shared_inputs_exclude_the_aliased_guide_points() {
        let counts = sample_counts();
        let shared = HairFramePipeline::shared_input_bytes(&counts, COLLIDERS, RENDER_POINTS);
        // sim inputs: goals 3200*16 + rest_lengths 3100*4 + strands 100*16 + colliders 8*32.
        let sim_inputs = 3200 * 16 + 3100 * 4 + 100 * 16 + 8 * 32;
        // resolve static tables: guide_ranges 100*8 + bindings 50_000*48.
        let interp_tables = 100 * 8 + 50_000 * 48;
        assert_eq!(shared, sim_inputs + interp_tables);
        // guide_points (3200 * 16) must NOT be part of the shared class.
        let guide_points = HairInterpBuffer::GuidePoints.byte_size(&counts, RENDER_POINTS);
        assert_eq!(guide_points, 3200 * 16);
        assert!(shared < sim_inputs + interp_tables + guide_points);
    }

    #[test]
    fn total_residency_scales_replicated_class_by_depth() {
        let counts = sample_counts();
        let replicated =
            HairFramePipeline::replicated_bytes_per_frame(&counts, COLLIDERS, RENDER_POINTS);
        let shared = HairFramePipeline::shared_input_bytes(&counts, COLLIDERS, RENDER_POINTS);

        let single =
            HairFramePipeline::single_buffered().plan_residency(&counts, COLLIDERS, RENDER_POINTS);
        let double =
            HairFramePipeline::double_buffered().plan_residency(&counts, COLLIDERS, RENDER_POINTS);

        assert_eq!(single.total_bytes(), replicated + shared);
        assert_eq!(double.total_bytes(), 2 * replicated + shared);
        // A deeper pipeline costs exactly one more replicated copy per frame.
        assert_eq!(double.total_bytes() - single.total_bytes(), replicated);
    }

    #[test]
    fn residency_exposes_ring_and_shared_split() {
        let counts = sample_counts();
        let plan = HairFramePipeline::new(3).plan_residency(&counts, COLLIDERS, RENDER_POINTS);
        assert_eq!(plan.frames_in_flight, 3);
        assert_eq!(plan.replicated_bytes(), 3 * plan.replicated_bytes_per_frame);
        assert_eq!(
            plan.total_bytes(),
            plan.replicated_bytes() + plan.shared_input_bytes
        );
    }

    #[test]
    fn empty_groom_still_costs_a_valid_clamped_allocation() {
        let counts = HairGpuCounts::default();
        let plan = HairFramePipeline::double_buffered().plan_residency(&counts, 0, 0);
        // No panic, and every clamped buffer contributes at least its stride.
        assert!(plan.replicated_bytes_per_frame > 0);
        assert!(plan.shared_input_bytes > 0);
        assert!(plan.total_bytes() > 0);
    }
}
