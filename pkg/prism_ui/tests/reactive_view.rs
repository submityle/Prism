//! Integration tests for `ReactiveView`: the signal graph driving the view.

#![allow(
    clippy::std_instead_of_alloc,
    reason = "integration tests run under std"
)]

use prism_ui::backend::BackendOp;
use prism_ui::reactive::Runtime;
use prism_ui::reactive_view::ReactiveView;
use prism_ui::{Element, RecordingBackend, Ui};

/// Counts how many `SetText` ops the backend has recorded.
fn set_text_count(ops: &[BackendOp]) -> usize {
    ops.iter()
        .filter(|op| matches!(op, BackendOp::SetText { .. }))
        .count()
}

#[test]
fn mounts_the_initial_view_once() {
    let rt = Runtime::new();
    let label = rt.signal(String::from("hello"));

    let view = {
        let label = label.clone();
        move || Element::box_().child(Element::text(label.get()))
    };
    let app = ReactiveView::new(&rt, Ui::new(RecordingBackend::new()), view);

    // A box plus its text child were materialised.
    assert_eq!(app.with_ui(Ui::node_count), 2);
}

#[test]
fn signal_change_re_renders_and_emits_set_text() {
    let rt = Runtime::new();
    let label = rt.signal(String::from("hello"));

    let view = {
        let label = label.clone();
        move || Element::box_().child(Element::text(label.get()))
    };
    let app = ReactiveView::new(&rt, Ui::new(RecordingBackend::new()), view);

    let before = app.with_ui(|ui| set_text_count(ui.backend().ops()));
    label.set(String::from("world"));
    let after = app.with_ui(|ui| set_text_count(ui.backend().ops()));

    assert!(
        after > before,
        "changing the signal should emit a new SetText op (before={before}, after={after})"
    );
}

#[test]
fn re_rendering_identical_value_emits_nothing() {
    let rt = Runtime::new();
    let label = rt.signal(String::from("stable"));

    let view = {
        let label = label.clone();
        move || Element::box_().child(Element::text(label.get()))
    };
    let app = ReactiveView::new(&rt, Ui::new(RecordingBackend::new()), view);

    let before = app.with_ui(|ui| ui.backend().len());
    // `set_if_changed` suppresses the write entirely when the value is equal,
    // so the view never re-runs and the backend sees no ops.
    let changed = label.set_if_changed(String::from("stable"));
    let after = app.with_ui(|ui| ui.backend().len());

    assert!(!changed, "setting an equal value should report no change");
    assert_eq!(before, after, "no ops should be emitted for an equal value");
}

#[test]
fn disposing_the_view_stops_updates() {
    let rt = Runtime::new();
    let label = rt.signal(String::from("a"));

    let view = {
        let label = label.clone();
        move || Element::box_().child(Element::text(label.get()))
    };
    let app = ReactiveView::new(&rt, Ui::new(RecordingBackend::new()), view);

    // Grab a shared handle, then drop the ReactiveView to tear down the effect.
    let cell = app.ui_cell();
    let before = cell.borrow().backend().len();
    drop(app);

    label.set(String::from("b"));
    let after = cell.borrow().backend().len();
    assert_eq!(
        before, after,
        "a disposed view must not react to later signal changes"
    );
}
