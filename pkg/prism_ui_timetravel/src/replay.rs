//! Replay and comparison helpers layered on top of a [`Timeline`].
//!
//! Where [`Timeline`] owns the history and the cursor, this module offers
//! read-only views over it:
//!
//! * [`Replay`] borrows a timeline and answers questions about it — the raw
//!   frames, a textual [`Replay::diff_between`] of any two frames, a
//!   step-by-step [`Replay::change_summary`], and the ordered
//!   [`Replay::replay_labels`].
//! * [`Replayer`] is a forward cursor that walks the frames in order via
//!   [`Replayer::step`], which is handy for feeding a backend one frame at a
//!   time.
//!
//! Both borrow the timeline immutably, so navigation state and replay state
//! stay independent.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui_snapshot::{diff, serialize_tree};

use crate::frame::Frame;
use crate::timeline::Timeline;

/// A single transition between two adjacent frames in a [`Timeline`].
///
/// Produced by [`Replay::change_summary`]. It records the endpoints, the net
/// change in node count, and whether the captured trees differ at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplayStep {
    /// Index of the earlier frame.
    pub from: usize,
    /// Index of the later frame (`from + 1`).
    pub to: usize,
    /// Node-count change from `from` to `to`; negative when nodes were removed.
    pub delta_nodes: isize,
    /// `true` when the two frames' snapshots are not equal.
    pub changed: bool,
}

/// A read-only replay view over a borrowed [`Timeline`].
#[derive(Clone, Copy, Debug)]
pub struct Replay<'a> {
    /// The timeline being inspected.
    timeline: &'a Timeline,
}

impl<'a> Replay<'a> {
    /// Creates a replay view over `timeline`.
    #[must_use]
    pub fn new(timeline: &'a Timeline) -> Self {
        Self { timeline }
    }

    /// Returns the recorded frames in chronological order.
    #[must_use]
    pub fn frames(&self) -> &'a [Frame] {
        self.timeline.frames()
    }

    /// Returns a human-readable, line-level diff of the frames at `a` and `b`.
    ///
    /// The diff is computed from each frame's stable tree serialization via
    /// [`prism_ui_snapshot::serialize_tree`] and
    /// [`prism_ui_snapshot::diff`]. Returns [`None`] when either index is out
    /// of range. Identical frames yield `Some(String::new())`.
    #[must_use]
    pub fn diff_between(&self, a: usize, b: usize) -> Option<String> {
        let frames = self.frames();
        let left = frames.get(a)?;
        let right = frames.get(b)?;
        let left_text = serialize_tree(left.snapshot());
        let right_text = serialize_tree(right.snapshot());
        Some(diff(&left_text, &right_text))
    }

    /// Summarizes every adjacent transition in the timeline.
    ///
    /// The returned vector has `len().saturating_sub(1)` entries; it is empty
    /// for a timeline with fewer than two frames.
    #[must_use]
    pub fn change_summary(&self) -> Vec<ReplayStep> {
        let frames = self.frames();
        let mut steps = Vec::new();
        let mut index = 1;
        while index < frames.len() {
            let from = &frames[index - 1];
            let to = &frames[index];
            let delta_nodes = to.node_count() as isize - from.node_count() as isize;
            let changed = from.snapshot() != to.snapshot();
            steps.push(ReplayStep {
                from: index - 1,
                to: index,
                delta_nodes,
                changed,
            });
            index += 1;
        }
        steps
    }

    /// Returns each frame's label in chronological order.
    #[must_use]
    pub fn replay_labels(&self) -> Vec<&'a str> {
        self.frames().iter().map(Frame::label).collect()
    }
}

/// A forward cursor that walks a timeline's frames one at a time.
///
/// Unlike [`Timeline`]'s own cursor, a replayer never mutates the timeline; it
/// is a lightweight iterator-like position suitable for driving a backend
/// through recorded history.
#[derive(Clone, Copy, Debug)]
pub struct Replayer<'a> {
    /// The timeline being replayed.
    timeline: &'a Timeline,
    /// Index of the next frame [`Replayer::step`] will yield.
    position: usize,
}

impl<'a> Replayer<'a> {
    /// Creates a replayer positioned at the first frame.
    #[must_use]
    pub fn new(timeline: &'a Timeline) -> Self {
        Self {
            timeline,
            position: 0,
        }
    }

    /// Returns the index of the frame the next [`Replayer::step`] will yield.
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Returns `true` when every frame has already been yielded.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.position >= self.timeline.len()
    }

    /// Returns the current frame without advancing, or [`None`] at the end.
    #[must_use]
    pub fn peek(&self) -> Option<&'a Frame> {
        self.timeline.frames().get(self.position)
    }

    /// Yields the current frame and advances the cursor by one.
    ///
    /// Returns [`None`] once the end of the timeline is reached.
    pub fn step(&mut self) -> Option<&'a Frame> {
        let frame = self.timeline.frames().get(self.position)?;
        self.position += 1;
        Some(frame)
    }

    /// Rewinds the replayer to the first frame.
    pub fn reset(&mut self) {
        self.position = 0;
    }
}

impl Timeline {
    /// Returns a read-only [`Replay`] view over this timeline.
    #[must_use]
    pub fn replay(&self) -> Replay<'_> {
        Replay::new(self)
    }

    /// Returns a fresh [`Replayer`] positioned at the first frame.
    #[must_use]
    pub fn replayer(&self) -> Replayer<'_> {
        Replayer::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::{Replay, ReplayStep, Replayer};
    use crate::frame::Frame;
    use crate::timeline::Timeline;
    use alloc::vec::Vec;
    use prism_ui::Element;

    fn timeline() -> Timeline {
        let mut timeline = Timeline::new();
        timeline.record(Frame::capture(
            "one",
            &Element::box_().child(Element::text("a")),
        ));
        timeline.record(Frame::capture(
            "two",
            &Element::box_()
                .child(Element::text("a"))
                .child(Element::text("b")),
        ));
        timeline.record(Frame::capture(
            "three",
            &Element::box_().child(Element::text("a")),
        ));
        timeline
    }

    #[test]
    fn frames_and_labels_are_in_order() {
        let timeline = timeline();
        let replay = Replay::new(&timeline);
        assert_eq!(replay.frames().len(), 3);
        assert_eq!(replay.replay_labels(), ["one", "two", "three"]);
    }

    #[test]
    fn diff_between_reports_changes() {
        let timeline = timeline();
        let replay = timeline.replay();
        let diff = replay.diff_between(0, 1).expect("valid indices");
        assert!(!diff.is_empty());
        assert!(diff.contains("+b"));
        // Identical frames diff to an empty string.
        let same = replay.diff_between(0, 0).expect("valid indices");
        assert!(same.is_empty());
        // Out-of-range indices yield None.
        assert!(replay.diff_between(0, 9).is_none());
    }

    #[test]
    fn change_summary_tracks_node_deltas() {
        let timeline = timeline();
        let summary = timeline.replay().change_summary();
        assert_eq!(
            summary,
            [
                ReplayStep {
                    from: 0,
                    to: 1,
                    delta_nodes: 1,
                    changed: true,
                },
                ReplayStep {
                    from: 1,
                    to: 2,
                    delta_nodes: -1,
                    changed: true,
                },
            ]
        );
    }

    #[test]
    fn change_summary_marks_unchanged_frames() {
        let mut timeline = Timeline::new();
        let view = Element::box_().child(Element::text("same"));
        timeline.record(Frame::capture("a", &view));
        timeline.record(Frame::capture("b", &view));
        let summary = timeline.replay().change_summary();
        assert_eq!(summary.len(), 1);
        assert_eq!(summary[0].delta_nodes, 0);
        assert!(!summary[0].changed);
    }

    #[test]
    fn change_summary_empty_for_short_timeline() {
        let empty = Timeline::new();
        assert!(empty.replay().change_summary().is_empty());
        let mut single = Timeline::new();
        single.record(Frame::capture("only", &Element::box_()));
        assert!(single.replay().change_summary().is_empty());
    }

    #[test]
    fn replayer_walks_frames_then_finishes() {
        let timeline = timeline();
        let mut replayer = Replayer::new(&timeline);
        assert_eq!(replayer.position(), 0);
        assert_eq!(replayer.peek().unwrap().label(), "one");

        let labels: Vec<&str> = core::iter::from_fn(|| replayer.step().map(Frame::label)).collect();
        assert_eq!(labels, ["one", "two", "three"]);
        assert!(replayer.is_finished());
        assert!(replayer.step().is_none());

        replayer.reset();
        assert_eq!(replayer.position(), 0);
        assert!(!replayer.is_finished());
        assert_eq!(replayer.step().unwrap().label(), "one");
    }
}
