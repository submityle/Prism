//! End-to-end integration test exercising the full input pipeline.

use core::cell::RefCell;

use prism_ui_input::geometry::{Point, Rect, Size};
use prism_ui_input::{
    hit_test, Dispatcher, DragAxis, DragRecognizer, FocusRing, GestureArena, HitNode, NodeId,
    Phase, PointerEvent, PointerEvents, PointerId, PointerKind, TapRecognizer,
};

thread_local! {
    static RECORD: RefCell<Vec<(NodeId, Phase)>> = const { RefCell::new(Vec::new()) };
}

fn rect(x: f32, y: f32, w: f32, h: f32) -> Rect {
    Rect::new(Point::new(x, y), Size::new(w, h))
}

fn pointer(kind: PointerKind, x: f32, y: f32, t: u64) -> PointerEvent {
    PointerEvent::new(PointerId::new(1), kind, Point::new(x, y), t)
}

#[test]
fn hit_test_then_dispatch_through_phases() {
    RECORD.with(|r| r.borrow_mut().clear());

    // root(1) -> panel(2) -> button(3); an overlay(4) is transparent to hits.
    let root = HitNode::new(NodeId::new(1), rect(0.0, 0.0, 200.0, 200.0))
        .child(
            HitNode::new(NodeId::new(2), rect(10.0, 10.0, 100.0, 100.0))
                .child(HitNode::new(NodeId::new(3), rect(20.0, 20.0, 40.0, 40.0))),
        )
        .child(
            HitNode::new(NodeId::new(4), rect(0.0, 0.0, 200.0, 200.0))
                .with_z_index(100)
                .with_pointer_events(PointerEvents::None),
        );

    let path = hit_test(&root, Point::new(30.0, 30.0));
    assert_eq!(
        path,
        vec![NodeId::new(1), NodeId::new(2), NodeId::new(3)],
        "transparent overlay must not swallow the hit"
    );

    let mut dispatcher = Dispatcher::new();
    for &node in &path {
        for phase in [Phase::Capture, Phase::Target, Phase::Bubble] {
            dispatcher.on(node, phase, |ctx, _event| {
                RECORD.with(|r| r.borrow_mut().push((ctx.current_target(), ctx.phase())));
            });
        }
    }

    let outcome = dispatcher.dispatch(&path, &pointer(PointerKind::Down, 30.0, 30.0, 0));
    assert!(!outcome.propagation_stopped);

    RECORD.with(|r| {
        let seen = r.borrow().clone();
        assert_eq!(
            seen,
            vec![
                (NodeId::new(1), Phase::Capture),
                (NodeId::new(2), Phase::Capture),
                (NodeId::new(3), Phase::Target),
                (NodeId::new(2), Phase::Bubble),
                (NodeId::new(1), Phase::Bubble),
            ]
        );
    });
}

#[test]
fn arena_resolves_drag_over_tap() {
    let mut arena = GestureArena::new();
    let tap = arena.add(Box::new(TapRecognizer::new(NodeId::new(3), 5.0)));
    let drag = arena.add(Box::new(DragRecognizer::new(
        NodeId::new(3),
        5.0,
        DragAxis::Horizontal,
    )));

    assert_eq!(arena.route(&pointer(PointerKind::Down, 0.0, 0.0, 0)), None);
    // A clearly horizontal move crosses slop and the drag wins eagerly.
    let winner = arena.route(&pointer(PointerKind::Move, 40.0, 2.0, 16));
    assert_eq!(winner, Some(drag));
    assert_ne!(winner, Some(tap));
    assert_eq!(arena.winner(), Some(drag));
}

#[test]
fn focus_ring_tab_cycle() {
    let mut ring = FocusRing::new();
    ring.add(NodeId::new(10), 0);
    ring.add(NodeId::new(11), 0);
    ring.add(NodeId::new(12), 0);

    assert_eq!(ring.focus_next(), Some(NodeId::new(10)));
    assert_eq!(ring.focus_next(), Some(NodeId::new(11)));
    assert_eq!(ring.focus_next(), Some(NodeId::new(12)));
    assert_eq!(ring.focus_next(), Some(NodeId::new(10)));
    assert_eq!(ring.focus_prev(), Some(NodeId::new(12)));
}
