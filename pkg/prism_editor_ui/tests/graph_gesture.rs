//! Pointer capture against real painted graph geometry and authoring commands.
use prism_editor_graph::*;
use prism_editor_ui::{
    BoundGraphPanel, GraphGesturePreview as Preview, GraphPanelAction as A,
    GraphPanelController as Controller, GraphPanelError as Error, GraphPanelLimits,
    GraphPanelViewport, GraphPointerButton as Button, GraphPointerHit as Hit,
    SelectionMode as Mode,
};

fn draft(n: u64, x: i32, y: i32, kind: &str) -> GraphEdit {
    GraphEdit::CreateNode(GraphNodeDraft {
        id: NodeId::from_raw(n),
        name: format!("host node {n}"),
        payload: vec![7; 32],
        position: GraphPoint { x, y },
        ports: [
            PortSpec::input("in", PortType::named(kind)),
            PortSpec::output("out", PortType::named(kind)),
        ]
        .into_iter()
        .enumerate()
        .map(|(i, spec)| GraphPortDraft {
            id: PortId::from_raw(n * 2 - 1 + i as u64),
            spec,
        })
        .collect(),
    })
}
fn fixture() -> Controller {
    let mut c = Controller::new(
        GraphEditor::new(Default::default()).unwrap(),
        vec![],
        Default::default(),
        Default::default(),
    )
    .unwrap();
    c.apply(A::Command(GraphCommand::Edit(vec![
        draft(1, 40, 40, "host.scalar"),
        draft(2, 360, 40, "host.scalar"),
        draft(3, 40, 240, "host.other"),
    ])))
    .unwrap();
    c
}
fn port(v: &BoundGraphPanel, id: u64) -> [f64; 2] {
    let r = v
        .ports()
        .iter()
        .find(|p| p.port == PortId::from_raw(id))
        .unwrap()
        .rect;
    [f64::from(r[0] + r[2] / 2.), f64::from(r[1] + r[3] / 2.)]
}
fn press(c: &mut Controller, p: [f64; 2], button: Button, mode: Mode) -> Hit {
    c.pointer_press(&c.build(), p, button, mode).unwrap()
}
fn release(c: &mut Controller, p: [f64; 2]) -> Result<(), Error> {
    c.pointer_release(&c.build(), p)
}
fn connect(c: &mut Controller, from: u64, to: u64) -> Result<(), Error> {
    let v = c.build();
    let a = port(&v, from);
    let b = port(&v, to);
    c.pointer_press(&v, a, Button::Primary, Mode::Replace)
        .unwrap();
    c.pointer_move(b).unwrap();
    release(c, b)
}
#[test]
fn group_drag_previews_only_and_commits_one_reversible_transaction() {
    let mut c = fixture();
    c.apply(A::Select(NodeId::from_raw(1), Mode::Replace))
        .unwrap();
    c.apply(A::Select(NodeId::from_raw(2), Mode::Add)).unwrap();
    let token = c.editor().snapshot_token();
    let depth = c.editor().undo_depth();
    assert_eq!(
        press(&mut c, [60., 52.], Button::Primary, Mode::Replace),
        Hit::Node(NodeId::from_raw(1))
    );
    for i in 0..400 {
        c.pointer_move([60. + f64::from(i) / 4., 82.]).unwrap();
    }
    assert_eq!(c.editor().snapshot_token(), token);
    assert_eq!(c.editor().undo_depth(), depth);
    assert_eq!(
        c.editor().position(NodeId::from_raw(1)),
        Some(GraphPoint { x: 40, y: 40 })
    );
    assert_eq!(
        c.gesture_preview(),
        Some(Preview::Nodes {
            delta: GraphPoint { x: 100, y: 30 },
            count: 2
        })
    );
    let v = c.build();
    assert_eq!(v.nodes()[0].rect[..2], [140., 70.]);
    assert_eq!(v.nodes()[1].rect[..2], [460., 70.]);
    release(&mut c, [160., 82.]).unwrap();
    assert!(!c.pointer_active());
    assert_eq!(c.editor().undo_depth(), depth + 1);
    assert_eq!(
        c.editor().position(NodeId::from_raw(2)),
        Some(GraphPoint { x: 460, y: 70 })
    );
    c.apply(A::Command(GraphCommand::Undo)).unwrap();
    assert_eq!(
        c.editor().position(NodeId::from_raw(1)),
        Some(GraphPoint { x: 40, y: 40 })
    );
    assert_eq!(
        c.editor().position(NodeId::from_raw(2)),
        Some(GraphPoint { x: 360, y: 40 })
    );
    c.apply(A::Command(GraphCommand::Redo)).unwrap();
    assert_eq!(
        c.editor().position(NodeId::from_raw(1)),
        Some(GraphPoint { x: 140, y: 70 })
    );
}
#[test]
fn rubber_band_and_reverse_input_drag_create_real_typed_edges_and_topology() {
    let mut c = fixture();
    let token = c.editor().snapshot_token();
    let start = port(&c.build(), 3);
    let end = port(&c.build(), 2);
    press(&mut c, start, Button::Primary, Mode::Replace);
    c.pointer_move(end).unwrap();
    let v = c.build();
    assert!(!v.preview_rects().is_empty());
    assert!(v.preview_rects().len() <= 3);
    assert_eq!(c.editor().snapshot_token(), token);
    assert_eq!(c.editor().graph().edge_count(), 0);
    release(&mut c, end).unwrap();
    assert_eq!(c.editor().graph().edge_count(), 1);
    let mut order = Vec::new();
    c.editor()
        .graph()
        .evaluate(|n| {
            order.push(n.node().id());
            Ok::<_, ()>(())
        })
        .unwrap();
    assert!(order.iter().position(|n| n.raw() == 1) < order.iter().position(|n| n.raw() == 2));
    c.apply(A::Command(GraphCommand::Undo)).unwrap();
    assert_eq!(c.editor().graph().edge_count(), 0);
    c.apply(A::Command(GraphCommand::Redo)).unwrap();
    let view = c.build();
    let wire = &view.wires()[0];
    assert!(!wire.segments.is_empty());
}
#[test]
fn connection_type_cycle_cardinality_and_direction_fail_atomically() {
    let mut c = fixture();
    for (from, to) in [(2, 5), (2, 4), (1, 3)] {
        let token = c.editor().snapshot_token();
        let depth = c.editor().undo_depth();
        let error = connect(&mut c, from, to).unwrap_err();
        if to == 5 {
            assert!(matches!(
                error,
                Error::Authoring(GraphAuthoringError::Graph(GraphError::TypeMismatch { .. }))
            ));
        } else {
            assert!(matches!(
                error,
                Error::Authoring(GraphAuthoringError::Graph(
                    GraphError::WrongDirection { .. }
                ))
            ));
        }
        assert!(!c.pointer_active());
        assert_eq!(c.editor().snapshot_token(), token);
        assert_eq!(c.editor().undo_depth(), depth);
    }
    connect(&mut c, 2, 3).unwrap();
    let token = c.editor().snapshot_token();
    // Duplicate input cardinality and back-edge cycle both go through the kernel.
    assert_eq!(
        connect(&mut c, 2, 3),
        Err(Error::Authoring(GraphAuthoringError::Graph(
            GraphError::InputAlreadyConnected(PortId::from_raw(3))
        )))
    );
    assert!(matches!(
        connect(&mut c, 4, 1),
        Err(Error::Authoring(GraphAuthoringError::Graph(
            GraphError::Cycle(_)
        )))
    ));
    assert_eq!(c.editor().snapshot_token(), token);
    assert_eq!(c.editor().graph().edge_count(), 1);
}
#[test]
fn rectangle_selection_is_deferred_and_supports_replace_add_toggle() {
    let mut c = fixture();
    let token = c.editor().snapshot_token();
    press(&mut c, [20., 20.], Button::Primary, Mode::Replace);
    c.pointer_move([580., 140.]).unwrap();
    assert_eq!(c.selected().count(), 0);
    assert_eq!(c.build().preview_rects().len(), 4);
    release(&mut c, [580., 140.]).unwrap();
    assert_eq!(c.selected().count(), 2);
    press(&mut c, [20., 220.], Button::Primary, Mode::Add);
    release(&mut c, [250., 340.]).unwrap();
    assert_eq!(c.selected().count(), 3);
    press(&mut c, [20., 20.], Button::Primary, Mode::Toggle);
    release(&mut c, [250., 140.]).unwrap();
    assert_eq!(
        c.selected().collect::<Vec<_>>(),
        [NodeId::from_raw(2), NodeId::from_raw(3)]
    );
    assert_eq!(c.editor().snapshot_token(), token);
}
#[test]
fn cancellation_zero_delta_and_invalid_release_never_commit() {
    let mut c = fixture();
    let token = c.editor().snapshot_token();
    press(&mut c, [60., 52.], Button::Primary, Mode::Replace);
    release(&mut c, [60., 52.]).unwrap();
    assert_eq!(c.editor().snapshot_token(), token);
    for bad in [
        [f64::NAN, 10.],
        [f64::INFINITY, 10.],
        [-1., 10.],
        [800., 10.],
    ] {
        press(&mut c, [60., 52.], Button::Primary, Mode::Replace);
        c.pointer_move([100., 100.]).unwrap();
        let preview = c.gesture_preview();
        assert_eq!(c.pointer_move([f64::NAN, 0.]), Err(Error::Geometry));
        assert_eq!(c.gesture_preview(), preview);
        assert_eq!(release(&mut c, bad), Err(Error::Geometry));
        assert!(!c.pointer_active());
        assert_eq!(c.editor().snapshot_token(), token);
    }
    press(&mut c, [60., 52.], Button::Primary, Mode::Replace);
    c.pointer_move([900., 100.]).unwrap();
    assert!(c.cancel_pointer());
    assert!(!c.cancel_pointer());
    assert_eq!(c.editor().snapshot_token(), token);
    let p = port(&c.build(), 2);
    press(&mut c, p, Button::Primary, Mode::Replace);
    assert_eq!(release(&mut c, [700., 400.]), Err(Error::Target));
    assert_eq!(c.editor().snapshot_token(), token);
}
#[test]
fn stale_foreign_and_semantically_invalidated_publications_are_fenced() {
    let mut c = fixture();
    let foreign = fixture().build();
    let old = c.build();
    assert_eq!(
        c.pointer_press(&foreign, [60., 52.], Button::Primary, Mode::Replace),
        Err(Error::StaleView)
    );
    press(&mut c, [60., 52.], Button::Primary, Mode::Replace);
    assert_eq!(
        c.pointer_press(&c.build(), [60., 52.], Button::Primary, Mode::Replace),
        Err(Error::GestureState)
    );
    let token = c.editor().snapshot_token();
    c.pointer_move([80., 52.]).unwrap();
    assert_eq!(c.pointer_release(&old, [80., 52.]), Err(Error::StaleView));
    assert!(!c.pointer_active());
    assert_eq!(c.editor().snapshot_token(), token);
    press(&mut c, [60., 52.], Button::Primary, Mode::Replace);
    let captured = c.build();
    c.apply(A::Pan([1., 0.])).unwrap();
    assert!(!c.pointer_active());
    assert_eq!(
        c.pointer_release(&captured, [80., 52.]),
        Err(Error::StaleView)
    );
    press(&mut c, [60., 52.], Button::Primary, Mode::Replace);
    c.apply(A::Command(GraphCommand::Undo)).unwrap();
    c.apply(A::Command(GraphCommand::Redo)).unwrap();
    assert_eq!(c.pointer_move([80., 52.]), Err(Error::GestureState));
}
#[test]
fn pan_preview_cancel_commit_and_anchor_zoom_obey_coordinate_fences() {
    let mut c = fixture();
    let initial = c.viewport();
    let token = c.editor().snapshot_token();
    press(&mut c, [60., 52.], Button::Pan, Mode::Replace);
    c.pointer_move([100., 72.]).unwrap();
    assert_eq!(c.viewport(), initial);
    assert_eq!(c.build().viewport().origin, [-40., -20.]);
    assert_eq!(c.build().nodes()[0].rect[..2], [80., 60.]);
    assert_eq!(
        c.zoom_at(&c.build(), [100., 100.], 2.),
        Err(Error::GestureState)
    );
    c.cancel_pointer();
    assert_eq!(c.build().viewport(), initial);
    press(&mut c, [60., 52.], Button::Pan, Mode::Replace);
    release(&mut c, [100., 72.]).unwrap();
    assert_eq!(c.viewport().origin, [-40., -20.]);
    let anchor = c.viewport().document_point([300., 160.]).unwrap();
    c.zoom_at(&c.build(), [300., 160.], 2.).unwrap();
    assert_eq!(c.viewport().document_point([300., 160.]), Some(anchor));
    let valid = c.viewport();
    for factor in [f64::NAN, f64::INFINITY, -1., 0., 100.] {
        assert_eq!(
            c.zoom_at(&c.build(), [300., 160.], factor),
            Err(Error::Geometry)
        );
        assert_eq!(c.viewport(), valid);
    }
    assert_eq!(c.editor().snapshot_token(), token);
}
#[test]
fn hits_use_clipped_half_open_geometry_and_reverse_painter_order() {
    let mut c = fixture();
    c.apply(A::Command(GraphCommand::Edit(vec![GraphEdit::MoveNode {
        node: NodeId::from_raw(2),
        position: GraphPoint { x: 40, y: 100 },
    }])))
    .unwrap();
    let v = c.build();
    // Higher node box covers the lower node's output port.
    assert_eq!(
        v.hit_test([200., 104.]),
        Some(Hit::Node(NodeId::from_raw(2)))
    );
    assert_eq!(
        v.hit_test([50., 140.]),
        Some(Hit::Port(PortId::from_raw(3)))
    );
    assert_eq!(v.hit_test([240., 52.]), Some(Hit::Canvas));
    assert_eq!(v.hit_test([800., 0.]), None);
    assert_eq!(v.hit_test([f64::NAN, 0.]), None);
    c.apply(A::Viewport(GraphPanelViewport {
        origin: [100., 0.],
        zoom: 0.5,
        ..c.viewport()
    }))
    .unwrap();
    for p in c.build().ports() {
        assert!(p.rect[0] >= 0. && p.rect[1] >= 0.);
    }
}
#[test]
fn selection_and_visible_query_budgets_reject_partial_rectangle_updates() {
    let mut editor = GraphEditor::new(Default::default()).unwrap();
    editor
        .execute(&GraphCommand::Edit(vec![
            draft(1, 40, 40, "t"),
            draft(2, 360, 40, "t"),
        ]))
        .unwrap();
    let limits = GraphPanelLimits {
        selection: 1,
        visible_nodes: 1,
        visible_ports: 1,
        visible_edges: 1,
        query_visits: 16,
        ..Default::default()
    };
    let mut c = Controller::new(editor, vec![], Default::default(), limits).unwrap();
    c.apply(A::Select(NodeId::from_raw(1), Mode::Replace))
        .unwrap();
    press(&mut c, [20., 20.], Button::Primary, Mode::Replace);
    c.pointer_move([600., 150.]).unwrap();
    assert_eq!(release(&mut c, [600., 150.]), Err(Error::Capacity));
    assert_eq!(c.selected().collect::<Vec<_>>(), [NodeId::from_raw(1)]);
    let v = c.build();
    assert!(v.stats().truncated);
    assert!(v.stats().query_visits <= 16);
    assert!(v.nodes().len() <= 1 && v.ports().len() <= 1 && v.wires().len() <= 1);
}
#[test]
fn drag_at_zoom_uses_initial_coordinates_and_checks_final_document_bounds() {
    let mut c = fixture();
    c.apply(A::Viewport(GraphPanelViewport {
        origin: [0., 0.],
        zoom: 2.,
        ..c.viewport()
    }))
    .unwrap();
    press(&mut c, [100., 100.], Button::Primary, Mode::Replace);
    release(&mut c, [120., 140.]).unwrap();
    assert_eq!(
        c.editor().position(NodeId::from_raw(1)),
        Some(GraphPoint { x: 50, y: 60 })
    );
    c.apply(A::Command(GraphCommand::Edit(vec![GraphEdit::MoveNode {
        node: NodeId::from_raw(1),
        position: GraphPoint { x: 999_990, y: 0 },
    }])))
    .unwrap();
    c.apply(A::Viewport(GraphPanelViewport {
        origin: [999_980., 0.],
        zoom: 1.,
        ..c.viewport()
    }))
    .unwrap();
    press(&mut c, [30., 10.], Button::Primary, Mode::Replace);
    assert_eq!(c.pointer_move([50., 10.]), Err(Error::Geometry));
    assert_eq!(release(&mut c, [50., 10.]), Err(Error::Geometry));
    assert_eq!(c.editor().position(NodeId::from_raw(1)).unwrap().x, 999_990);
}
#[test]
fn preview_query_includes_selected_nodes_entering_from_outside_the_viewport() {
    let mut c = fixture();
    c.apply(A::Command(GraphCommand::Edit(vec![
        GraphEdit::MoveNode {
            node: NodeId::from_raw(1),
            position: GraphPoint { x: 500, y: 40 },
        },
        GraphEdit::MoveNode {
            node: NodeId::from_raw(2),
            position: GraphPoint { x: 1000, y: 40 },
        },
    ])))
    .unwrap();
    c.apply(A::Select(NodeId::from_raw(1), Mode::Replace))
        .unwrap();
    c.apply(A::Select(NodeId::from_raw(2), Mode::Add)).unwrap();
    assert!(c.build().nodes().iter().all(|n| n.node.raw() != 2));
    press(&mut c, [520., 52.], Button::Primary, Mode::Replace);
    c.pointer_move([120., 52.]).unwrap();
    let view = c.build();
    let incoming = view.nodes().iter().find(|n| n.node.raw() == 2).unwrap();
    assert_eq!(incoming.rect[..2], [600., 40.]);
    assert!(view.stats().query_visits <= GraphPanelLimits::default().query_visits);
    assert_eq!(c.editor().position(NodeId::from_raw(2)).unwrap().x, 1000);
    release(&mut c, [120., 52.]).unwrap();
    assert_eq!(c.editor().position(NodeId::from_raw(2)).unwrap().x, 600);
}
#[test]
fn old_preview_and_toggle_removal_cannot_commit_a_pointer_capture() {
    let mut c = fixture();
    press(&mut c, [60., 52.], Button::Primary, Mode::Replace);
    c.pointer_move([80., 52.]).unwrap();
    let previous_move = c.build();
    c.pointer_move([100., 52.]).unwrap();
    let token = c.editor().snapshot_token();
    assert_eq!(
        c.pointer_release(&previous_move, [100., 52.]),
        Err(Error::StaleView)
    );
    assert_eq!(c.editor().snapshot_token(), token);
    assert_eq!(
        press(&mut c, [60., 52.], Button::Primary, Mode::Toggle),
        Hit::Node(NodeId::from_raw(1))
    );
    assert!(!c.pointer_active());
    assert_eq!(c.selected().count(), 0);
    assert_eq!(release(&mut c, [80., 52.]), Err(Error::GestureState));
}
#[test]
fn retained_layout_and_cpu_raster_paint_moved_nodes_ports_wires_and_overlay() {
    use prism_ui::{
        layout::{AvailableSpace, Size},
        Element, Key, Ui,
    };
    use prism_ui_render_backend::{rasterize, RetainedScene};
    fn names(e: &Element, keys: &mut Vec<Option<String>>) {
        keys.push(match e.explicit_key() {
            Some(Key::Str(k)) => Some(k.clone()),
            _ => None,
        });
        for child in e.child_elements() {
            names(child, keys);
        }
    }
    let mut c = fixture();
    connect(&mut c, 2, 3).unwrap();
    press(&mut c, [60., 52.], Button::Primary, Mode::Replace);
    c.pointer_move([90., 72.]).unwrap();
    let moved = c.build();
    assert_eq!(moved.nodes()[0].rect[..2], [70., 60.]);
    assert_eq!(moved.ports()[0].rect[..2], [70., 92.]);
    assert_eq!(moved.wires()[0].segments[0][..2], [270., 122.]);
    c.cancel_pointer();
    let start = port(&c.build(), 2);
    press(&mut c, start, Button::Primary, Mode::Replace);
    c.pointer_move([320., 180.]).unwrap();
    let view = c.build();
    let mut keys = Vec::new();
    names(view.element(), &mut keys);
    let mut ui = Ui::new(RetainedScene::new());
    ui.mount(view.element());
    ui.compute_layout(Size::new(
        AvailableSpace::Definite(1000.),
        AvailableSpace::MaxContent,
    ));
    let mut rects = Vec::new();
    ui.backend().visit_layout(|_, r| rects.push(r));
    assert_eq!(keys.len(), rects.len());
    let keyed: std::collections::BTreeMap<_, _> = keys
        .into_iter()
        .zip(rects)
        .filter_map(|(k, r)| k.map(|k| (k, r)))
        .collect();
    let canvas = keyed["graph:canvas"];
    let image = rasterize(&ui.backend().to_draw_list(), 1000, 600);
    for (i, expected) in view.preview_rects().iter().enumerate() {
        let rect = keyed[&format!("graph:gesture:{i}")];
        assert!((rect.left() - canvas.left() - expected[0]).abs() < 0.1);
        assert!((rect.top() - canvas.top() - expected[1]).abs() < 0.1);
        let pixel = image.pixel(
            (rect.left() + expected[2] / 2.) as u32,
            (rect.top() + expected[3] / 2.) as u32,
        );
        assert!(
            pixel[0] > 0.9 && pixel[1] > 0.6 && pixel[2] < 0.25,
            "{pixel:?}"
        );
    }
}
