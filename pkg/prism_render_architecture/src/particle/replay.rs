//! Deterministic rewind / replay timeline contract (design §29, §25).
//!
//! `AAA` `GPU`-driven particle engines expose a *replay driver* so authors can
//! scrub a live simulation backwards and forwards without re-simulating from
//! frame zero every time. The technique mirrors Unreal `Niagara`'s determinism
//! rewind and `Frostbite`'s FX replay stack: the simulation periodically
//! captures a full keyframe *snapshot* of the `SoA` particle state, and every
//! intermediate frame stores only a compact *delta*. A seek to an arbitrary
//! target frame then restores the nearest earlier keyframe and re-plays the
//! handful of delta frames in between, which is bounded and cheap.
//!
//! This module is the `CPU`-verifiable contract layer for that driver. It owns
//! no `GPU` state and performs no simulation; it computes the *bookkeeping* a
//! backend needs: which keyframe covers a frame, how many replay steps a seek
//! costs, how much `VRAM` the capture consumes, how many keyframes a byte
//! budget affords, and the ring-buffer eviction of the oldest keyframe. All
//! arithmetic is integer where the quantity is a frame count or a byte count;
//! any `f32` comparison uses the shared [`CMP_EPS`] tolerance rather than `==`.
//!
//! The module is intentionally self-contained: it declares its own small
//! constants and does not import sibling particle modules, so it can be
//! developed in parallel with the rest of the subsystem.

use alloc::vec::Vec;

/// Absolute tolerance for `f32` comparisons.
///
/// `f32` equality (`==` / `!=`) is banned across the crate because it is not
/// robust to rounding; callers compare magnitudes against this epsilon instead.
pub const CMP_EPS: f32 = 1.0e-6;

/// A captured keyframe: a full snapshot of the particle state at one frame.
///
/// A keyframe is the anchor a seek restores from. Unlike a delta frame it
/// stores the entire `SoA` pool, so it is comparatively expensive in `VRAM`;
/// the replay driver spaces keyframes `keyframe_interval` frames apart to trade
/// capture cost against seek cost.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ReplayKeyframe {
    /// The absolute frame index this keyframe snapshots.
    pub frame_index: u32,
    /// The size of the captured snapshot in bytes.
    pub snapshot_bytes: u64,
}

/// The replay timeline configuration and its derived bookkeeping.
///
/// A timeline captures a full keyframe every `keyframe_interval` frames and a
/// delta for every frame in `total_frames`. The type answers the driver's
/// questions about keyframe placement, seek cost, and capture footprint.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ReplayTimeline {
    /// Frames between successive keyframes; at least `1`.
    keyframe_interval: u32,
    /// Total number of simulated frames on the timeline.
    total_frames: u32,
    /// Bytes captured by one keyframe snapshot.
    keyframe_snapshot_bytes: u64,
    /// Bytes captured by one delta frame.
    delta_frame_bytes: u64,
}

impl ReplayTimeline {
    /// Builds a timeline, clamping `keyframe_interval` up to at least `1`.
    ///
    /// A zero interval is nonsensical (it would place keyframes zero frames
    /// apart), so it is clamped to `1`, which makes every frame a keyframe.
    #[must_use]
    pub fn new(
        keyframe_interval: u32,
        total_frames: u32,
        keyframe_snapshot_bytes: u64,
        delta_frame_bytes: u64,
    ) -> Self {
        let keyframe_interval = if keyframe_interval < 1 {
            1
        } else {
            keyframe_interval
        };
        Self {
            keyframe_interval,
            total_frames,
            keyframe_snapshot_bytes,
            delta_frame_bytes,
        }
    }

    /// Returns the (clamped) keyframe interval in frames.
    #[must_use]
    pub fn keyframe_interval(&self) -> u32 {
        self.keyframe_interval
    }

    /// Returns the total simulated frame count.
    #[must_use]
    pub fn total_frames(&self) -> u32 {
        self.total_frames
    }

    /// Returns the nearest keyframe frame index at or before `frame`.
    ///
    /// This is an integer floor to the interval grid: `(frame / interval) *
    /// interval`. Frame `0` is always a keyframe, so the result is `0` for any
    /// frame in the first interval.
    #[must_use]
    pub fn keyframe_for_frame(&self, frame: u32) -> u32 {
        (frame / self.keyframe_interval) * self.keyframe_interval
    }

    /// Returns how many delta frames separate `frame` from its keyframe.
    ///
    /// This is exactly the number of replay steps a seek to `frame` must apply
    /// after restoring the keyframe.
    #[must_use]
    pub fn frames_since_keyframe(&self, frame: u32) -> u32 {
        frame - self.keyframe_for_frame(frame)
    }

    /// Returns the number of keyframes captured over the timeline.
    ///
    /// Keyframe `0` always exists, and one more is captured every
    /// `keyframe_interval` frames, so the count is a ceiling division of
    /// `total_frames` by the interval. A zero-length timeline captures no
    /// keyframes.
    #[must_use]
    pub fn keyframe_count(&self) -> u32 {
        if self.total_frames == 0 {
            return 0;
        }
        self.total_frames.div_ceil(self.keyframe_interval)
    }

    /// Returns the total bytes captured by the timeline.
    ///
    /// This sums the keyframe snapshots and the per-frame deltas, using
    /// saturating arithmetic so a pathological configuration reports
    /// [`u64::MAX`] rather than wrapping.
    #[must_use]
    pub fn captured_bytes(&self) -> u64 {
        let keyframe_bytes =
            u64::from(self.keyframe_count()).saturating_mul(self.keyframe_snapshot_bytes);
        let delta_bytes = u64::from(self.total_frames).saturating_mul(self.delta_frame_bytes);
        keyframe_bytes.saturating_add(delta_bytes)
    }
}

/// A resolved seek plan: which keyframe to restore and how far to replay.
///
/// The replay driver restores `restore_keyframe`'s snapshot, then applies
/// `replay_steps` delta frames to land exactly on the requested target frame.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ReplaySeek {
    /// The keyframe frame index to restore before replaying.
    pub restore_keyframe: u32,
    /// The number of delta frames to replay after restoring.
    pub replay_steps: u32,
}

impl ReplayTimeline {
    /// Resolves the seek plan for `target_frame`.
    ///
    /// Restores the nearest earlier keyframe and replays the frames between it
    /// and the target. Seeking directly to a keyframe frame yields zero replay
    /// steps.
    #[must_use]
    pub fn seek_plan(&self, target_frame: u32) -> ReplaySeek {
        let restore_keyframe = self.keyframe_for_frame(target_frame);
        ReplaySeek {
            restore_keyframe,
            replay_steps: target_frame - restore_keyframe,
        }
    }
}

/// A byte budget for keyframe capture.
///
/// `VRAM` is finite, so the driver caps how much of it the replay ring may
/// consume. This type converts a byte budget into a keyframe capacity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ReplayBudget {
    /// The maximum bytes the keyframe capture may occupy.
    pub max_bytes: u64,
}

impl ReplayBudget {
    /// Returns how many keyframes of `per_keyframe_bytes` fit in the budget.
    ///
    /// A zero-size keyframe carries no information and would divide by zero, so
    /// it yields `0`. The integer quotient is clamped into a `u32`, saturating
    /// to [`u32::MAX`] rather than truncating, to keep the `CPU`/`GPU` seek
    /// bookkeeping honest.
    #[must_use]
    pub fn max_keyframes(&self, per_keyframe_bytes: u64) -> u32 {
        if per_keyframe_bytes == 0 {
            return 0;
        }
        let fit = self.max_bytes / per_keyframe_bytes;
        u32::try_from(fit).unwrap_or(u32::MAX)
    }
}

/// A ring buffer of keyframe slots that evicts the oldest when full.
///
/// The replay driver keeps only the most recent `capacity` keyframes resident;
/// pushing a new keyframe into a full ring overwrites (evicts) the oldest. The
/// ring stores slot bookkeeping only — the backend owns the actual snapshots.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct KeyframeRing {
    /// The fixed number of resident keyframe slots; at least `1`.
    capacity: u32,
    /// The slot index of the oldest resident keyframe.
    head: u32,
    /// The number of resident keyframes, in `0..=capacity`.
    len: u32,
}

impl KeyframeRing {
    /// Builds an empty ring, clamping `capacity` up to at least `1`.
    #[must_use]
    pub fn new(capacity: u32) -> Self {
        let capacity = if capacity < 1 { 1 } else { capacity };
        Self {
            capacity,
            head: 0,
            len: 0,
        }
    }

    /// Returns the ring's fixed slot capacity.
    #[must_use]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Returns the number of resident keyframes.
    #[must_use]
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Returns `true` when the ring holds no keyframes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns `true` when every slot is occupied.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    /// Returns the slot index the oldest resident keyframe occupies.
    ///
    /// When the ring is empty this is the `head`, which is where the first
    /// keyframe will land.
    #[must_use]
    pub fn oldest_frame_slot(&self) -> u32 {
        self.head
    }

    /// Returns the slot index of the most recently pushed keyframe.
    ///
    /// On an empty ring there is no newest keyframe, so this returns the `head`
    /// slot as a benign default.
    #[must_use]
    pub fn newest_slot(&self) -> u32 {
        if self.len == 0 {
            return self.head;
        }
        (self.head + self.len - 1) % self.capacity
    }

    /// Pushes a keyframe, evicting the oldest slot when the ring is full.
    ///
    /// Returns the slot index the new keyframe occupies. When the ring was
    /// full the oldest keyframe is evicted and `head` advances; otherwise the
    /// length grows.
    pub fn push(&mut self) -> u32 {
        if self.is_full() {
            let slot = self.head;
            self.head = (self.head + 1) % self.capacity;
            slot
        } else {
            let slot = (self.head + self.len) % self.capacity;
            self.len += 1;
            slot
        }
    }

    /// Returns the occupied slot indices from oldest to newest.
    ///
    /// This materializes the logical order the driver would iterate when
    /// rebuilding the replay window, resolving the ring wrap into a flat list.
    #[must_use]
    pub fn resident_slots(&self) -> Vec<u32> {
        let mut slots = Vec::new();
        let mut i = 0;
        while i < self.len {
            slots.push((self.head + i) % self.capacity);
            i += 1;
        }
        slots
    }
}

/// Returns whether a seek is bit-exactly deterministic.
///
/// A replay is only reproducible when the stateless hash `RNG` stream is
/// continuous across the seek boundary *and* a keyframe is present to restore
/// from. If either is missing the restored state cannot match the original
/// run, so the seek is treated as non-deterministic.
#[must_use]
pub fn is_seek_deterministic(rng_stream_continuous: bool, keyframe_present: bool) -> bool {
    rng_stream_continuous && keyframe_present
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_clamps_zero_interval_to_one() {
        let timeline = ReplayTimeline::new(0, 100, 4096, 64);
        assert_eq!(timeline.keyframe_interval(), 1);
        // Every frame becomes its own keyframe.
        assert_eq!(timeline.keyframe_for_frame(37), 37);
        assert_eq!(timeline.frames_since_keyframe(37), 0);
    }

    #[test]
    fn keyframe_for_frame_floors_to_interval_grid() {
        let timeline = ReplayTimeline::new(10, 100, 4096, 64);
        assert_eq!(timeline.keyframe_for_frame(0), 0);
        assert_eq!(timeline.keyframe_for_frame(9), 0);
        assert_eq!(timeline.keyframe_for_frame(10), 10);
        assert_eq!(timeline.keyframe_for_frame(23), 20);
    }

    #[test]
    fn frame_zero_anchors_on_keyframe_zero() {
        let timeline = ReplayTimeline::new(8, 64, 1024, 16);
        assert_eq!(timeline.keyframe_for_frame(0), 0);
        assert_eq!(timeline.frames_since_keyframe(0), 0);
    }

    #[test]
    fn frames_since_keyframe_counts_delta_span() {
        let timeline = ReplayTimeline::new(10, 100, 4096, 64);
        assert_eq!(timeline.frames_since_keyframe(23), 3);
        assert_eq!(timeline.frames_since_keyframe(20), 0);
        assert_eq!(timeline.frames_since_keyframe(29), 9);
    }

    #[test]
    fn keyframe_count_uses_div_ceil() {
        // 100 frames, interval 10 -> keyframes at 0,10,..,90 = 10 keyframes.
        assert_eq!(ReplayTimeline::new(10, 100, 1, 1).keyframe_count(), 10);
        // 101 frames rounds up to 11.
        assert_eq!(ReplayTimeline::new(10, 101, 1, 1).keyframe_count(), 11);
        // 95 frames rounds up to 10.
        assert_eq!(ReplayTimeline::new(10, 95, 1, 1).keyframe_count(), 10);
    }

    #[test]
    fn keyframe_count_is_zero_for_empty_timeline() {
        assert_eq!(ReplayTimeline::new(10, 0, 4096, 64).keyframe_count(), 0);
    }

    #[test]
    fn captured_bytes_sums_keyframes_and_deltas() {
        // 100 frames, interval 10 -> 10 keyframes * 4096 + 100 deltas * 64.
        let timeline = ReplayTimeline::new(10, 100, 4096, 64);
        assert_eq!(timeline.captured_bytes(), 10 * 4096 + 100 * 64);
    }

    #[test]
    fn captured_bytes_saturates_on_overflow() {
        let timeline = ReplayTimeline::new(1, u32::MAX, u64::MAX, u64::MAX);
        assert_eq!(timeline.captured_bytes(), u64::MAX);
    }

    #[test]
    fn seek_plan_to_keyframe_has_zero_steps() {
        let timeline = ReplayTimeline::new(10, 100, 4096, 64);
        let plan = timeline.seek_plan(20);
        assert_eq!(plan.restore_keyframe, 20);
        assert_eq!(plan.replay_steps, 0);
    }

    #[test]
    fn seek_plan_replays_from_nearest_keyframe() {
        let timeline = ReplayTimeline::new(10, 100, 4096, 64);
        let plan = timeline.seek_plan(23);
        assert_eq!(plan.restore_keyframe, 20);
        assert_eq!(plan.replay_steps, 3);
    }

    #[test]
    fn seek_plan_frame_zero_is_trivial() {
        let timeline = ReplayTimeline::new(10, 100, 4096, 64);
        let plan = timeline.seek_plan(0);
        assert_eq!(plan.restore_keyframe, 0);
        assert_eq!(plan.replay_steps, 0);
    }

    #[test]
    fn budget_zero_size_keyframe_yields_zero() {
        let budget = ReplayBudget {
            max_bytes: 1_000_000,
        };
        assert_eq!(budget.max_keyframes(0), 0);
    }

    #[test]
    fn budget_zero_bytes_fits_nothing() {
        let budget = ReplayBudget { max_bytes: 0 };
        assert_eq!(budget.max_keyframes(4096), 0);
    }

    #[test]
    fn budget_divides_bytes_into_keyframes() {
        let budget = ReplayBudget { max_bytes: 10_240 };
        assert_eq!(budget.max_keyframes(4096), 2);
    }

    #[test]
    fn budget_saturates_keyframe_count_into_u32() {
        let budget = ReplayBudget {
            max_bytes: u64::MAX,
        };
        assert_eq!(budget.max_keyframes(1), u32::MAX);
    }

    #[test]
    fn ring_new_clamps_zero_capacity() {
        let ring = KeyframeRing::new(0);
        assert_eq!(ring.capacity(), 1);
        assert!(ring.is_empty());
        assert!(!ring.is_full());
    }

    #[test]
    fn ring_fills_before_evicting() {
        let mut ring = KeyframeRing::new(3);
        assert_eq!(ring.push(), 0);
        assert_eq!(ring.push(), 1);
        assert_eq!(ring.push(), 2);
        assert!(ring.is_full());
        assert_eq!(ring.len(), 3);
        assert_eq!(ring.oldest_frame_slot(), 0);
        assert_eq!(ring.newest_slot(), 2);
    }

    #[test]
    fn ring_full_push_evicts_oldest_slot() {
        let mut ring = KeyframeRing::new(3);
        ring.push();
        ring.push();
        ring.push();
        // Full: the next push reuses slot 0 and advances head to 1.
        assert_eq!(ring.push(), 0);
        assert!(ring.is_full());
        assert_eq!(ring.len(), 3);
        assert_eq!(ring.oldest_frame_slot(), 1);
        assert_eq!(ring.newest_slot(), 0);
    }

    #[test]
    fn ring_resident_slots_are_oldest_to_newest_with_wrap() {
        let mut ring = KeyframeRing::new(3);
        ring.push();
        ring.push();
        ring.push();
        ring.push();
        // After one eviction the window is slots 1, 2, 0 in age order.
        assert_eq!(ring.resident_slots(), Vec::from([1, 2, 0]));
    }

    #[test]
    fn ring_empty_slots_default_to_head() {
        let ring = KeyframeRing::new(4);
        assert_eq!(ring.oldest_frame_slot(), 0);
        assert_eq!(ring.newest_slot(), 0);
        assert!(ring.resident_slots().is_empty());
    }

    #[test]
    fn keyframe_struct_records_frame_and_size() {
        let keyframe = ReplayKeyframe {
            frame_index: 40,
            snapshot_bytes: 8192,
        };
        assert_eq!(keyframe.frame_index, 40);
        assert_eq!(keyframe.snapshot_bytes, 8192);
    }

    #[test]
    fn seek_determinism_requires_both_conditions() {
        assert!(is_seek_deterministic(true, true));
        assert!(!is_seek_deterministic(true, false));
        assert!(!is_seek_deterministic(false, true));
        assert!(!is_seek_deterministic(false, false));
    }

    #[test]
    fn cmp_eps_guards_float_comparison() {
        let a = 0.1_f32 + 0.2_f32;
        let b = 0.3_f32;
        assert!((a - b).abs() < CMP_EPS);
    }
}
