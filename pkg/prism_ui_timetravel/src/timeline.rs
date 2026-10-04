//! An ordered, navigable history of recorded [`Frame`]s.
//!
//! A [`Timeline`] behaves like an editor's undo history. It holds the frames
//! in record order and keeps a `cursor` marking the "current" frame.
//! [`Timeline::back`] and [`Timeline::forward`] move the cursor one step,
//! [`Timeline::jump`] moves it to an arbitrary index, and
//! [`Timeline::record`] appends a new frame — truncating any
//! forward ("redo") branch first, exactly as typing after an undo discards the
//! redo stack.
//!
//! Every operation is deterministic and allocation-light, so timelines are
//! trivial to unit test.

use alloc::vec::Vec;

use crate::frame::Frame;

/// An ordered history of [`Frame`]s with a movable current-position cursor.
///
/// The cursor is only meaningful while the timeline is non-empty; on an empty
/// timeline [`Timeline::current`] returns [`None`]. Recording a frame while the
/// cursor is behind the end discards the frames after the cursor before
/// appending, mirroring undo/redo semantics.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Timeline {
    /// Recorded frames in chronological order.
    frames: Vec<Frame>,
    /// Index of the current frame; meaningful only when `frames` is non-empty.
    cursor: usize,
    /// Maximum number of frames to retain, or `0` for an unbounded history.
    ///
    /// When positive, recording past the limit evicts the oldest frames so the
    /// newest `capacity` entries are kept, exactly like a bounded undo history
    /// in a production editor.
    capacity: usize,
}

impl Timeline {
    /// Creates an empty timeline.
    #[must_use]
    pub fn new() -> Self {
        Self {
            frames: Vec::new(),
            cursor: 0,
            capacity: 0,
        }
    }

    /// Creates an empty timeline that retains at most `capacity` frames.
    ///
    /// Once the timeline holds `capacity` frames, each new [`Timeline::record`]
    /// evicts the oldest frame(s) so the newest `capacity` entries are kept,
    /// mirroring a bounded undo history. A `capacity` of `0` means unbounded,
    /// identical to [`Timeline::new`].
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            frames: Vec::new(),
            cursor: 0,
            capacity,
        }
    }

    /// Records `frame` as the newest entry and makes it current.
    ///
    /// If the cursor is not already at the last frame, every frame after the
    /// cursor is discarded first, so recording always extends the branch the
    /// cursor currently sits on.
    pub fn record(&mut self, frame: Frame) {
        if self.frames.is_empty() {
            self.frames.push(frame);
            self.cursor = 0;
        } else {
            self.frames.truncate(self.cursor + 1);
            self.frames.push(frame);
            self.cursor = self.frames.len() - 1;
        }
        self.enforce_capacity();
    }

    /// Returns the number of recorded frames.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Returns `true` when no frames have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Returns the current cursor index.
    ///
    /// On an empty timeline this is `0` but does not point at a frame; use
    /// [`Timeline::current`] to read the frame safely.
    #[must_use]
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Returns the frame the cursor currently points at, if any.
    #[must_use]
    pub fn current(&self) -> Option<&Frame> {
        self.frames.get(self.cursor)
    }

    /// Moves the cursor one frame toward the start and returns the new current
    /// frame, or [`None`] when already at the first frame.
    pub fn back(&mut self) -> Option<&Frame> {
        if self.can_undo() {
            self.cursor -= 1;
            self.frames.get(self.cursor)
        } else {
            None
        }
    }

    /// Moves the cursor one frame toward the end and returns the new current
    /// frame, or [`None`] when already at the last frame.
    pub fn forward(&mut self) -> Option<&Frame> {
        if self.can_redo() {
            self.cursor += 1;
            self.frames.get(self.cursor)
        } else {
            None
        }
    }

    /// Moves the cursor to `index` and returns that frame, or [`None`] when
    /// `index` is out of range (leaving the cursor unchanged).
    pub fn jump(&mut self, index: usize) -> Option<&Frame> {
        if index < self.frames.len() {
            self.cursor = index;
            self.frames.get(self.cursor)
        } else {
            None
        }
    }

    /// Returns the first recorded frame without moving the cursor.
    #[must_use]
    pub fn first(&self) -> Option<&Frame> {
        self.frames.first()
    }

    /// Returns the last recorded frame without moving the cursor.
    #[must_use]
    pub fn last(&self) -> Option<&Frame> {
        self.frames.last()
    }

    /// Returns `true` when [`Timeline::back`] would move the cursor.
    #[must_use]
    pub fn can_undo(&self) -> bool {
        !self.frames.is_empty() && self.cursor > 0
    }

    /// Returns `true` when [`Timeline::forward`] would move the cursor.
    #[must_use]
    pub fn can_redo(&self) -> bool {
        self.cursor + 1 < self.frames.len()
    }

    /// Returns the recorded frames in chronological order.
    #[must_use]
    pub fn frames(&self) -> &[Frame] {
        &self.frames
    }

    /// Returns the retention limit, or [`None`] when the history is unbounded.
    #[must_use]
    pub fn capacity(&self) -> Option<usize> {
        if self.capacity == 0 {
            None
        } else {
            Some(self.capacity)
        }
    }

    /// Sets the retention limit, evicting the oldest frames immediately when the
    /// current length exceeds the new limit.
    ///
    /// A `capacity` of `0` makes the history unbounded. When eviction removes
    /// frames the cursor is shifted to keep pointing at the same frame; if that
    /// frame is itself evicted, the cursor clamps to the new oldest frame.
    pub fn set_capacity(&mut self, capacity: usize) {
        self.capacity = capacity;
        self.enforce_capacity();
    }

    /// Drops the oldest frames until the length fits within `capacity`.
    ///
    /// Does nothing when the history is unbounded (`capacity == 0`) or already
    /// within the limit. Eviction always removes a prefix of the oldest frames,
    /// so the retained frames stay a contiguous suffix of the record order and
    /// the newest frame is never evicted. The cursor is decremented by the
    /// number of evicted frames, saturating at `0`.
    fn enforce_capacity(&mut self) {
        if self.capacity == 0 || self.frames.len() <= self.capacity {
            return;
        }
        let excess = self.frames.len() - self.capacity;
        self.frames.drain(0..excess);
        self.cursor = self.cursor.saturating_sub(excess);
    }

    /// Removes every frame and resets the cursor.
    ///
    /// The retention limit set via [`Timeline::with_capacity`] or
    /// [`Timeline::set_capacity`] is preserved.
    pub fn clear(&mut self) {
        self.frames.clear();
        self.cursor = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::Timeline;
    use crate::frame::Frame;
    use prism_ui::Element;

    fn frame(label: &str, text: &str) -> Frame {
        Frame::capture(label, &Element::box_().child(Element::text(text)))
    }

    #[test]
    fn new_timeline_is_empty() {
        let timeline = Timeline::new();
        assert!(timeline.is_empty());
        assert_eq!(timeline.len(), 0);
        assert!(timeline.current().is_none());
        assert!(!timeline.can_undo());
        assert!(!timeline.can_redo());
    }

    #[test]
    fn record_appends_and_advances_cursor() {
        let mut timeline = Timeline::new();
        timeline.record(frame("a", "1"));
        timeline.record(frame("b", "2"));
        assert_eq!(timeline.len(), 2);
        assert_eq!(timeline.cursor(), 1);
        assert_eq!(timeline.current().unwrap().label(), "b");
    }

    #[test]
    fn back_and_forward_walk_history() {
        let mut timeline = Timeline::new();
        timeline.record(frame("a", "1"));
        timeline.record(frame("b", "2"));
        timeline.record(frame("c", "3"));

        assert_eq!(timeline.back().unwrap().label(), "b");
        assert_eq!(timeline.back().unwrap().label(), "a");
        assert!(timeline.back().is_none());
        assert_eq!(timeline.cursor(), 0);

        assert_eq!(timeline.forward().unwrap().label(), "b");
        assert_eq!(timeline.forward().unwrap().label(), "c");
        assert!(timeline.forward().is_none());
    }

    #[test]
    fn record_truncates_redo_branch() {
        let mut timeline = Timeline::new();
        timeline.record(frame("a", "1"));
        timeline.record(frame("b", "2"));
        timeline.record(frame("c", "3"));
        // Rewind two steps, then record a new branch.
        timeline.back();
        timeline.back();
        assert_eq!(timeline.current().unwrap().label(), "a");
        timeline.record(frame("d", "4"));
        assert_eq!(timeline.len(), 2);
        assert_eq!(timeline.current().unwrap().label(), "d");
        assert!(!timeline.can_redo());
        assert_eq!(timeline.frames()[0].label(), "a");
        assert_eq!(timeline.frames()[1].label(), "d");
    }

    #[test]
    fn jump_moves_to_index() {
        let mut timeline = Timeline::new();
        timeline.record(frame("a", "1"));
        timeline.record(frame("b", "2"));
        timeline.record(frame("c", "3"));
        assert_eq!(timeline.jump(0).unwrap().label(), "a");
        assert_eq!(timeline.cursor(), 0);
        assert_eq!(timeline.jump(2).unwrap().label(), "c");
        assert!(timeline.jump(3).is_none());
        // Out-of-range jump leaves the cursor where it was.
        assert_eq!(timeline.cursor(), 2);
    }

    #[test]
    fn first_and_last_do_not_move_cursor() {
        let mut timeline = Timeline::new();
        timeline.record(frame("a", "1"));
        timeline.record(frame("b", "2"));
        timeline.back();
        assert_eq!(timeline.first().unwrap().label(), "a");
        assert_eq!(timeline.last().unwrap().label(), "b");
        assert_eq!(timeline.cursor(), 0);
    }

    #[test]
    fn can_undo_and_redo_report_edges() {
        let mut timeline = Timeline::new();
        timeline.record(frame("a", "1"));
        assert!(!timeline.can_undo());
        assert!(!timeline.can_redo());
        timeline.record(frame("b", "2"));
        assert!(timeline.can_undo());
        assert!(!timeline.can_redo());
        timeline.back();
        assert!(!timeline.can_undo());
        assert!(timeline.can_redo());
    }

    #[test]
    fn clear_resets_timeline() {
        let mut timeline = Timeline::new();
        timeline.record(frame("a", "1"));
        timeline.record(frame("b", "2"));
        timeline.clear();
        assert!(timeline.is_empty());
        assert_eq!(timeline.cursor(), 0);
        assert!(timeline.current().is_none());
    }

    #[test]
    fn default_matches_new() {
        assert_eq!(Timeline::default(), Timeline::new());
    }

    #[test]
    fn new_timeline_is_unbounded() {
        assert_eq!(Timeline::new().capacity(), None);
        assert_eq!(Timeline::with_capacity(0).capacity(), None);
        assert_eq!(Timeline::with_capacity(4).capacity(), Some(4));
    }

    #[test]
    fn with_capacity_evicts_oldest() {
        let mut timeline = Timeline::with_capacity(2);
        timeline.record(frame("a", "1"));
        timeline.record(frame("b", "2"));
        timeline.record(frame("c", "3")); // evicts "a"
        assert_eq!(timeline.len(), 2);
        assert_eq!(timeline.frames()[0].label(), "b");
        assert_eq!(timeline.frames()[1].label(), "c");
        // The newest frame is never evicted and stays current.
        assert_eq!(timeline.current().unwrap().label(), "c");
        assert_eq!(timeline.cursor(), 1);
    }

    #[test]
    fn eviction_shifts_cursor_to_track_current_frame() {
        let mut timeline = Timeline::with_capacity(3);
        timeline.record(frame("a", "1"));
        timeline.record(frame("b", "2"));
        timeline.record(frame("c", "3"));
        // Rewind to "b" so a later record forks a branch.
        timeline.back();
        assert_eq!(timeline.current().unwrap().label(), "b");
        assert_eq!(timeline.cursor(), 1);
        // Recording truncates "c" then appends "d": [a, b, d], still within cap.
        timeline.record(frame("d", "4"));
        assert_eq!(timeline.len(), 3);
        assert_eq!(timeline.cursor(), 2);
        // One more record overflows and evicts "a": [b, d, e]; cursor shifts to
        // stay on the newest frame.
        timeline.record(frame("e", "5"));
        assert_eq!(timeline.len(), 3);
        assert_eq!(timeline.frames()[0].label(), "b");
        assert_eq!(timeline.current().unwrap().label(), "e");
        assert_eq!(timeline.cursor(), 2);
    }

    #[test]
    fn set_capacity_evicts_and_clamps_cursor() {
        let mut timeline = Timeline::new();
        assert_eq!(timeline.capacity(), None);
        for (label, text) in [
            ("a", "1"),
            ("b", "2"),
            ("c", "3"),
            ("d", "4"),
            ("e", "5"),
        ] {
            timeline.record(frame(label, text));
        }
        // Point the cursor at "b"; it sits inside the prefix we are about to
        // evict.
        timeline.jump(1);
        assert_eq!(timeline.current().unwrap().label(), "b");
        // Shrinking to 2 keeps [d, e]; the cursor clamps to the new oldest.
        timeline.set_capacity(2);
        assert_eq!(timeline.len(), 2);
        assert_eq!(timeline.frames()[0].label(), "d");
        assert_eq!(timeline.frames()[1].label(), "e");
        assert_eq!(timeline.cursor(), 0);
        assert_eq!(timeline.current().unwrap().label(), "d");
        // Lifting the cap retains the current frames without regrowing them.
        timeline.set_capacity(0);
        assert_eq!(timeline.capacity(), None);
        assert_eq!(timeline.len(), 2);
    }

    #[test]
    fn with_capacity_zero_is_unbounded() {
        use alloc::string::ToString;

        let mut timeline = Timeline::with_capacity(0);
        for id in 0u64..10 {
            timeline.record(frame(&id.to_string(), "x"));
        }
        assert_eq!(timeline.len(), 10);
        assert_eq!(timeline.cursor(), 9);
    }

    #[test]
    fn clear_preserves_capacity() {
        let mut timeline = Timeline::with_capacity(2);
        timeline.record(frame("a", "1"));
        timeline.record(frame("b", "2"));
        timeline.clear();
        assert_eq!(timeline.capacity(), Some(2));
        assert!(timeline.is_empty());
        assert_eq!(timeline.cursor(), 0);
    }

    #[test]
    fn bounded_history_matches_reference_model() {
        use alloc::string::ToString;
        use alloc::vec::Vec;

        // An independent reimplementation of the record/undo/redo + eviction
        // spec over integer ids, used as the oracle.
        struct Model {
            frames: Vec<u64>,
            cursor: usize,
            cap: usize,
        }

        impl Model {
            fn record(&mut self, id: u64) {
                if self.frames.is_empty() {
                    self.frames.push(id);
                    self.cursor = 0;
                } else {
                    self.frames.truncate(self.cursor + 1);
                    self.frames.push(id);
                    self.cursor = self.frames.len() - 1;
                }
                if self.cap != 0 && self.frames.len() > self.cap {
                    let excess = self.frames.len() - self.cap;
                    self.frames.drain(0..excess);
                    self.cursor = self.cursor.saturating_sub(excess);
                }
            }

            fn back(&mut self) {
                if !self.frames.is_empty() && self.cursor > 0 {
                    self.cursor -= 1;
                }
            }

            fn forward(&mut self) {
                if self.cursor + 1 < self.frames.len() {
                    self.cursor += 1;
                }
            }
        }

        // SplitMix64 for deterministic pseudo-random operation sequences.
        fn next_rand(state: &mut u64) -> u64 {
            *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        for cap in [1usize, 2, 3, 5] {
            let mut state =
                0x1234_5678_9ABC_DEF0u64.wrapping_add((cap as u64).wrapping_mul(0x9E37_79B9));
            let mut timeline = Timeline::with_capacity(cap);
            let mut model = Model {
                frames: Vec::new(),
                cursor: 0,
                cap,
            };
            let mut next_id = 0u64;

            for _ in 0..400 {
                match next_rand(&mut state) % 4 {
                    0 | 1 => {
                        let id = next_id;
                        next_id += 1;
                        timeline.record(frame(&id.to_string(), "x"));
                        model.record(id);
                    }
                    2 => {
                        timeline.back();
                        model.back();
                    }
                    _ => {
                        timeline.forward();
                        model.forward();
                    }
                }

                // Capacity bound and cursor validity.
                assert!(timeline.len() <= cap);
                assert_eq!(timeline.len(), model.frames.len());
                assert_eq!(timeline.cursor(), model.cursor);
                if timeline.is_empty() {
                    assert!(timeline.current().is_none());
                } else {
                    assert!(timeline.cursor() < timeline.len());
                }

                // Retained frames exactly match the reference suffix.
                let labels: Vec<u64> = timeline
                    .frames()
                    .iter()
                    .map(|f| f.label().parse::<u64>().unwrap())
                    .collect();
                assert_eq!(labels, model.frames);
            }
        }
    }
}
