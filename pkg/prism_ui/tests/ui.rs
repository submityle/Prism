//! Integration tests for the Loom view runtime.
//!
//! These exercise the full pipeline — element build → reconcile → style
//! cascade → flexbox layout → backend ops — against the headless
//! `RecordingBackend`, and assert the central performance contract: that an
//! update only emits ops for what actually changed.

#![allow(
    clippy::std_instead_of_alloc,
    reason = "integration tests run under std"
)]

use prism_ui::backend::{BackendId, BackendOp};
use prism_ui::layout::{AvailableSpace, Size};
use prism_ui::style::{Class, StyleProp, StyleSheet, StyleValue};
use prism_ui::{Element, RecordingBackend, Ui};

fn definite(w: f32, h: f32) -> Size<AvailableSpace> {
    Size::new(AvailableSpace::Definite(w), AvailableSpace::Definite(h))
}

/// A nested tree mounts to the expected creation ops and node hierarchy.
#[test]
fn mount_emits_hierarchy() {
    let view = Element::box_()
        .child(Element::text("a"))
        .child(Element::text("b"));

    let mut ui = Ui::new(RecordingBackend::new());
    ui.mount(&view);

    assert_eq!(ui.node_count(), 3);

    let creates: Vec<_> = ui
        .backend()
        .ops()
        .iter()
        .filter_map(|op| match op {
            BackendOp::Create {
                id, parent, index, ..
            } => Some((*id, *parent, *index)),
            _ => None,
        })
        .collect();

    // Root first, then its two children parented to it, in order.
    assert_eq!(creates.len(), 3);
    assert_eq!(creates[0], (BackendId(0), None, 0));
    assert_eq!(creates[1], (BackendId(1), Some(BackendId(0)), 0));
    assert_eq!(creates[2], (BackendId(2), Some(BackendId(0)), 1));

    // Text nodes get their text set.
    let texts: Vec<_> = ui
        .backend()
        .ops()
        .iter()
        .filter_map(|op| match op {
            BackendOp::SetText { id, text } => Some((*id, text.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(
        texts,
        vec![
            (BackendId(1), "a".to_owned()),
            (BackendId(2), "b".to_owned())
        ]
    );
}

/// Flexbox positions a row of fixed-size boxes left to right.
#[test]
fn compute_layout_positions_children() {
    let child = |w: f32| {
        Element::box_()
            .style(StyleProp::Width, StyleValue::px(w))
            .style(StyleProp::Height, StyleValue::px(50.0))
    };
    let view = Element::box_()
        .style(StyleProp::Width, StyleValue::px(300.0))
        .style(StyleProp::Height, StyleValue::px(100.0))
        .child(child(100.0))
        .child(child(100.0));

    let mut ui = Ui::new(RecordingBackend::new());
    ui.mount(&view);
    ui.backend_mut().clear();
    ui.compute_layout(definite(300.0, 100.0));

    let layouts: Vec<_> = ui
        .backend()
        .ops()
        .iter()
        .filter_map(|op| match op {
            BackendOp::SetLayout { id, location, size } => Some((*id, *location, *size)),
            _ => None,
        })
        .collect();

    // Root (id 0) at origin, 300x100.
    let root = layouts.iter().find(|(id, ..)| *id == BackendId(0)).unwrap();
    assert_eq!(root.1.x, 0.0);
    assert_eq!(root.2.width, 300.0);

    // Two children side by side: second starts where the first ends.
    let c1 = layouts.iter().find(|(id, ..)| *id == BackendId(1)).unwrap();
    let c2 = layouts.iter().find(|(id, ..)| *id == BackendId(2)).unwrap();
    assert_eq!(c1.1.x, 0.0);
    assert_eq!(c1.2.width, 100.0);
    assert_eq!(c2.1.x, 100.0);
    assert_eq!(c2.2.width, 100.0);
}

/// Updating one node's text emits exactly one `SetText` and nothing else.
#[test]
fn update_text_is_minimal() {
    let before = Element::box_().child(Element::text("old"));
    let after = Element::box_().child(Element::text("new"));

    let mut ui = Ui::new(RecordingBackend::new());
    ui.mount(&before);
    ui.backend_mut().clear();

    ui.update(&after);

    let ops = ui.backend().ops();
    assert_eq!(ops.len(), 1, "expected a single op, got {ops:?}");
    match &ops[0] {
        BackendOp::SetText { id, text } => {
            assert_eq!(*id, BackendId(1));
            assert_eq!(text, "new");
        }
        other => panic!("expected SetText, got {other:?}"),
    }
}

/// Re-rendering an identical tree emits no ops at all.
#[test]
fn identical_update_is_noop() {
    let view = Element::box_().class("card").child(Element::text("hi"));

    let sheet = StyleSheet::new().with_class(
        Class::new("card").with(StyleProp::BackgroundColor, StyleValue::token("color.bg")),
    );

    let mut ui = Ui::new(RecordingBackend::new()).with_stylesheet(sheet);
    ui.mount(&view);
    ui.backend_mut().clear();

    ui.update(&view);
    assert!(
        ui.backend().is_empty(),
        "identical update should emit nothing, got {:?}",
        ui.backend().ops()
    );
}

/// A class-driven style change emits a single `SetPaint` for the styled node.
#[test]
fn update_class_changes_paint() {
    let sheet = StyleSheet::new()
        .with_class(Class::new("a").with(
            StyleProp::BackgroundColor,
            StyleValue::rgba8(255, 0, 0, 255),
        ))
        .with_class(Class::new("b").with(
            StyleProp::BackgroundColor,
            StyleValue::rgba8(0, 255, 0, 255),
        ));

    let mut ui = Ui::new(RecordingBackend::new()).with_stylesheet(sheet);
    ui.mount(&Element::box_().class("a"));
    ui.backend_mut().clear();

    ui.update(&Element::box_().class("b"));

    let ops = ui.backend().ops();
    assert_eq!(ops.len(), 1, "expected one SetPaint, got {ops:?}");
    match &ops[0] {
        BackendOp::SetPaint { id, paint } => {
            assert_eq!(*id, BackendId(0));
            assert_eq!(
                paint.background_color,
                Some(prism_ui::style::Color::rgba8(0, 255, 0, 255))
            );
        }
        other => panic!("expected SetPaint, got {other:?}"),
    }
}

/// A keyed list reorder reuses nodes (no creates) and emits a single reorder.
#[test]
fn keyed_reorder_reuses_nodes() {
    let list = |keys: &[i64]| {
        Element::box_().children(
            keys.iter()
                .map(|&k| Element::text(format!("item-{k}")).key_int(k)),
        )
    };

    let mut ui = Ui::new(RecordingBackend::new());
    ui.mount(&list(&[1, 2, 3]));
    assert_eq!(ui.node_count(), 4);
    ui.backend_mut().clear();

    // Reverse the list; keys are stable so nodes should be reused.
    ui.update(&list(&[3, 2, 1]));

    let ops = ui.backend().ops();
    let creates = ops
        .iter()
        .filter(|op| matches!(op, BackendOp::Create { .. }))
        .count();
    let removes = ops
        .iter()
        .filter(|op| matches!(op, BackendOp::Remove { .. }))
        .count();
    let reorders = ops
        .iter()
        .filter(|op| matches!(op, BackendOp::Reorder { .. }))
        .count();

    assert_eq!(creates, 0, "reorder must not create nodes: {ops:?}");
    assert_eq!(removes, 0, "reorder must not remove nodes: {ops:?}");
    assert_eq!(reorders, 1, "exactly one reorder expected: {ops:?}");

    // The reorder carries the fully reversed backend order.
    if let Some(BackendOp::Reorder { order, .. }) = ops
        .iter()
        .find(|op| matches!(op, BackendOp::Reorder { .. }))
    {
        assert_eq!(*order, vec![BackendId(3), BackendId(2), BackendId(1)]);
    }

    // Node count is unchanged.
    assert_eq!(ui.node_count(), 4);
}

/// Adding and removing keyed children emits matching create/remove ops.
#[test]
fn keyed_insert_and_remove() {
    let list =
        |keys: &[i64]| Element::box_().children(keys.iter().map(|&k| Element::box_().key_int(k)));

    let mut ui = Ui::new(RecordingBackend::new());
    ui.mount(&list(&[1, 2]));
    ui.backend_mut().clear();

    // Drop key 1, keep 2, add 3.
    ui.update(&list(&[2, 3]));

    let ops = ui.backend().ops();
    let creates = ops
        .iter()
        .filter(|op| matches!(op, BackendOp::Create { .. }))
        .count();
    let removes = ops
        .iter()
        .filter(|op| matches!(op, BackendOp::Remove { .. }))
        .count();
    assert_eq!(creates, 1, "one node added: {ops:?}");
    assert_eq!(removes, 1, "one node removed: {ops:?}");
}
