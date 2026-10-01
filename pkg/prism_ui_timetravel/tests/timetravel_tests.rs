//! End-to-end integration tests for `prism_ui_timetravel`.
//!
//! These exercise the public API as an external crate would: capturing frames
//! from real `prism_ui` elements, navigating a timeline, diffing frames, and
//! replaying history. One case also attaches a real `OpTrace` produced by a
//! `RecordingBackend` driving the `Ui`.
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::layout::{AvailableSpace, Size};
use prism_ui::{Element, RecordingBackend, Ui};
use prism_ui_devtools::OpTrace;
use prism_ui_timetravel::{Frame, ReplayStep, Timeline};

/// Builds a box containing the given text labels as children.
fn view(texts: &[&str]) -> Element {
    let mut element = Element::box_();
    for text in texts {
        element = element.child(Element::text(*text));
    }
    element
}

#[test]
fn navigation_and_branch_truncation_end_to_end() {
    let mut timeline = Timeline::new();
    timeline.record(Frame::capture("a", &view(&["1"])));
    timeline.record(Frame::capture("b", &view(&["1", "2"])));
    timeline.record(Frame::capture("c", &view(&["1", "2", "3"])));

    assert_eq!(timeline.len(), 3);
    assert_eq!(timeline.current().unwrap().label(), "c");
    assert!(timeline.can_undo());
    assert!(!timeline.can_redo());

    // Walk all the way back.
    assert_eq!(timeline.back().unwrap().label(), "b");
    assert_eq!(timeline.back().unwrap().label(), "a");
    assert!(timeline.back().is_none());

    // Jump directly to the last frame, then forward past the end.
    assert_eq!(timeline.jump(2).unwrap().label(), "c");
    assert!(timeline.forward().is_none());

    // Rewind and record a new branch; the old "c" is discarded.
    timeline.jump(0);
    timeline.record(Frame::capture("b2", &view(&["x"])));
    assert_eq!(timeline.len(), 2);
    assert_eq!(timeline.current().unwrap().label(), "b2");
    assert!(!timeline.can_redo());
    let labels: Vec<&str> = timeline.replay().replay_labels();
    assert_eq!(labels, ["a", "b2"]);
}

#[test]
fn diff_and_change_summary_over_a_sequence() {
    let mut timeline = Timeline::new();
    timeline.record(Frame::capture("one", &view(&["a"])));
    timeline.record(Frame::capture("two", &view(&["a", "b"])));
    timeline.record(Frame::capture("three", &view(&["a"])));

    let replay = timeline.replay();

    // Growing from one child to two shows an inserted line mentioning "b".
    let grow = replay.diff_between(0, 1).expect("valid indices");
    assert!(!grow.is_empty());
    assert!(grow.contains('+'));
    assert!(grow.contains('b'));

    // Identical frames diff to an empty string; bad indices give None.
    assert!(replay.diff_between(0, 0).unwrap().is_empty());
    assert!(replay.diff_between(0, 42).is_none());

    let summary = replay.change_summary();
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
fn replayer_steps_through_every_frame() {
    let mut timeline = Timeline::new();
    timeline.record(Frame::capture("one", &view(&["a"])));
    timeline.record(Frame::capture("two", &view(&["a", "b"])));
    timeline.record(Frame::capture("three", &view(&["a", "b", "c"])));

    let mut replayer = timeline.replayer();
    let mut seen: Vec<String> = Vec::new();
    while let Some(frame) = replayer.step() {
        seen.push(String::from(frame.label()));
    }
    assert_eq!(seen, ["one", "two", "three"]);
    assert!(replayer.is_finished());

    replayer.reset();
    assert_eq!(replayer.position(), 0);
    assert_eq!(replayer.step().unwrap().label(), "one");
}

#[test]
fn frame_carries_a_real_backend_op_trace() {
    let element = view(&["hello", "world"]);

    // Drive a real backend to produce ops, then summarize them as an OpTrace.
    let mut ui = Ui::new(RecordingBackend::new());
    ui.mount(&element);
    ui.compute_layout(Size::new(
        AvailableSpace::Definite(640.0),
        AvailableSpace::Definite(480.0),
    ));
    let trace = OpTrace::from_ops(ui.backend().ops());

    let frame = Frame::capture("mounted", &element).with_trace(trace);
    assert!(frame.has_trace());
    // The box plus its two text children are three created nodes.
    assert_eq!(frame.trace().unwrap().creates(), 3);
    assert_eq!(frame.node_count(), 3);

    let mut timeline = Timeline::new();
    timeline.record(frame);
    assert_eq!(timeline.current().unwrap().trace().unwrap().creates(), 3);
}
