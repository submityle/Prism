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
}

impl Timeline {
    /// Creates an empty timeline.
    #[must_use]
    pub fn new() -> Self {
        Self {
            frames: Vec::new(),
            cursor: 0,
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

    /// Removes every frame and resets the cursor.
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
}
