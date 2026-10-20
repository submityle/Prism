//! Real retained UI, atomic authoring interactions and visible-window geometry.
use prism_editor_graph::*;
use prism_editor_ui::{
    ActionValue, BoundGraphPanel, GraphNodeTemplate, GraphPanelAction as A,
    GraphPanelController as Controller, GraphPanelError as Error, GraphPanelLimits,
    GraphPanelViewport, SelectionMode, UiEvent,
};
use prism_ui::{
    layout::{AvailableSpace, Size},
    BackendOp, Element, Key, RecordingBackend, Ui,
};

fn template() -> GraphNodeTemplate {
    GraphNodeTemplate {
        name: "Project supplied operator".into(),
        payload: vec![7],
        ports: vec![
            PortSpec::input("input", PortType::named("project.scalar")),
            PortSpec::output("output", PortType::named("project.scalar")),
        ],
    }
}
fn controller() -> Controller {
    Controller::new(
        GraphEditor::new(Default::default()).unwrap(),
        vec![template()],
        Default::default(),
        Default::default(),
    )
    .unwrap()
}
fn find<'a>(e: &'a Element, key: &str) -> Option<&'a Element> {
    if matches!(e.explicit_key(),Some(Key::Str(k)) if k==key) {
        return Some(e);
    }
    e.child_elements().iter().find_map(|e| find(e, key))
}
fn send(
    c: &mut Controller,
    ui: &mut Ui<RecordingBackend>,
    view: BoundGraphPanel,
    key: &str,
) -> BoundGraphPanel {
    assert!(find(view.element(), key).is_some(), "painted key {key}");
    c.handle_event(&view, key, &UiEvent::Activate).unwrap();
    let next = c.build();
    ui.update(next.element());
    for (key, _) in next.bindings() {
        assert!(
            find(next.element(), key).is_some(),
            "binding has painted target {key}"
        );
    }
    next
}
fn draft(n: u64, p: u64, x: i32, y: i32) -> GraphEdit {
    GraphEdit::CreateNode(GraphNodeDraft {
        id: NodeId::from_raw(n),
        name: format!("node{n}"),
        payload: vec![n as u8],
        position: GraphPoint { x, y },
        ports: template()
            .ports
            .into_iter()
            .enumerate()
            .map(|(i, s)| GraphPortDraft {
                id: PortId::from_raw(p + i as u64),
                spec: s,
            })
            .collect(),
    })
}
fn model(c: &mut Controller, edits: Vec<GraphEdit>) {
    c.apply(A::Command(GraphCommand::Edit(edits))).unwrap();
}
#[test]
fn mounted_controls_create_connect_move_delete_undo_redo_and_typed_evaluate() {
    let mut c = controller();
    let mut ui = Ui::new(RecordingBackend::new());
    let v = c.build();
    ui.mount(v.element());
    let v = send(&mut c, &mut ui, v, "graph:create:0");
    assert_eq!(c.editor().graph().node_count(), 1);
    let v = send(&mut c, &mut ui, v, "graph:move:left");
    let v = send(&mut c, &mut ui, v, "graph:create:0");
    let v = send(&mut c, &mut ui, v, "graph:port:2");
    assert_eq!(c.pending_output(), Some(PortId::from_raw(2)));
    let v = send(&mut c, &mut ui, v, "graph:port:3");
    assert_eq!(c.editor().graph().edge_count(), 1);
    assert_eq!(c.pending_output(), None);
    assert_eq!(v.ports()[0].type_key, "project.scalar");
    assert_eq!(v.wires().len(), 1);
    let mut values = std::collections::BTreeMap::new();
    c.editor()
        .graph()
        .evaluate(|n| {
            let input = n
                .input_edges()
                .iter()
                .map(|e| values[&c.editor().graph().port(e.output()).unwrap().node()])
                .sum::<u32>();
            values.insert(n.node().id(), input + u32::from(n.node().payload()[0]));
            Ok::<_, ()>(())
        })
        .unwrap();
    assert_eq!(values[&NodeId::from_raw(2)], 14);
    let v = send(&mut c, &mut ui, v, "graph:delete");
    assert_eq!(c.editor().graph().node_count(), 1);
    assert_eq!(c.editor().graph().edge_count(), 0);
    let v = send(&mut c, &mut ui, v, "graph:undo");
    assert_eq!(c.editor().graph().edge_count(), 1);
    let v = send(&mut c, &mut ui, v, "graph:redo");
    assert_eq!(c.editor().graph().node_count(), 1);
    let v = send(&mut c, &mut ui, v, "graph:undo");
    let v = send(&mut c, &mut ui, v, "graph:edge:1");
    assert_eq!(c.editor().graph().edge_count(), 0);
    let _ = send(&mut c, &mut ui, v, "graph:undo");
    c.editor().graph().validate().unwrap();
}
#[test]
fn semantic_selection_rectangle_and_move_form_one_undo_step() {
    let mut c = controller();
    model(
        &mut c,
        vec![
            draft(1, 1, 0, 0),
            draft(2, 3, 250, 0),
            draft(3, 5, 600, 300),
        ],
    );
    let v = c.build();
    c.handle_event(
        &v,
        "graph:node:1",
        &UiEvent::Select {
            mode: SelectionMode::Replace,
        },
    )
    .unwrap();
    let v = c.build();
    c.handle_event(
        &v,
        "graph:node:2",
        &UiEvent::Select {
            mode: SelectionMode::Add,
        },
    )
    .unwrap();
    assert_eq!(c.selected().count(), 2);
    let depth = c.editor().undo_depth();
    c.apply(A::MoveSelection(GraphPoint { x: 10, y: 20 }))
        .unwrap();
    assert_eq!(c.editor().undo_depth(), depth + 1);
    c.apply(A::Command(GraphCommand::Undo)).unwrap();
    assert_eq!(
        c.editor().position(NodeId::from_raw(1)),
        Some(GraphPoint { x: 0, y: 0 })
    );
    c.apply(A::SelectRect([0., 0., 500., 100.], SelectionMode::Replace))
        .unwrap();
    assert_eq!(c.selected().count(), 2);
    c.apply(A::Select(NodeId::from_raw(2), SelectionMode::Toggle))
        .unwrap();
    assert_eq!(c.selected().count(), 1);
    c.apply(A::DeleteSelection).unwrap();
    assert!(c.editor().graph().node(NodeId::from_raw(1)).is_none());
    c.apply(A::Command(GraphCommand::Undo)).unwrap();
    c.editor().graph().validate().unwrap();
}
#[test]
fn wrong_type_cycle_and_cardinality_leave_pending_and_history_unchanged() {
    let mut c = controller();
    let mut text = draft(3, 5, 500, 0);
    if let GraphEdit::CreateNode(ref mut n) = text {
        n.ports[0].spec = PortSpec::input("input", PortType::named("other.type"));
    }
    model(&mut c, vec![draft(1, 1, 0, 0), draft(2, 3, 250, 0), text]);
    c.apply(A::PickPort(PortId::from_raw(2))).unwrap();
    let v = c.build();
    let revision = c.editor().revision();
    assert!(c
        .handle_event(&v, "graph:port:5", &UiEvent::Activate)
        .is_err());
    assert_eq!(revision, c.editor().revision());
    assert_eq!(c.pending_output(), Some(PortId::from_raw(2)));
    c.handle_event(&v, "graph:port:3", &UiEvent::Activate)
        .unwrap();
    c.apply(A::PickPort(PortId::from_raw(2))).unwrap();
    assert!(c.apply(A::PickPort(PortId::from_raw(3))).is_err());
    c.apply(A::PickPort(PortId::from_raw(4))).unwrap();
    assert!(c.apply(A::PickPort(PortId::from_raw(1))).is_err());
    assert_eq!(c.editor().graph().edge_count(), 1);
    c.editor().graph().validate().unwrap();
}
#[test]
fn viewport_zoom_pan_controls_fence_old_views_and_reject_nonfinite() {
    let mut c = controller();
    let mut ui = Ui::new(RecordingBackend::new());
    let old = c.build();
    ui.mount(old.element());
    let v = send(&mut c, &mut ui, old.clone(), "graph:zoom:in");
    assert_eq!(c.viewport().zoom, 2.);
    let v = send(&mut c, &mut ui, v, "graph:pan:right");
    assert_eq!(c.viewport().origin[0], 300.);
    assert_eq!(
        c.handle_event(&old, "graph:create:0", &UiEvent::Activate),
        Err(Error::StaleView)
    );
    let foreign = controller().build();
    assert_eq!(
        c.handle_event(&foreign, "graph:create:0", &UiEvent::Activate),
        Err(Error::StaleView)
    );
    let before = c.viewport();
    for n in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0., -1., 100.] {
        assert!(c.apply(A::Zoom(n)).is_err());
        assert_eq!(c.viewport(), before);
    }
    for n in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(c.apply(A::Pan([n, 0.])).is_err());
        assert!(c
            .apply(A::SelectRect([0., 0., n, 0.], SelectionMode::Replace))
            .is_err());
    }
    assert_eq!(c.viewport(), before);
    assert_eq!(before.document_point([f64::NAN, 0.]), None);
    assert_eq!(before.document_point([before.size[0], 0.]), None);
    assert!(c
        .handle_event(
            &v,
            "graph:create:0",
            &UiEvent::Commit {
                value: ActionValue::float(f64::NAN)
            }
        )
        .is_err());
    assert_eq!(c.editor().graph().node_count(), 0);
}
#[test]
fn offscreen_endpoints_still_draw_clipped_wire_and_real_layout() {
    let mut c = controller();
    model(
        &mut c,
        vec![
            draft(1, 1, -500, 0),
            draft(2, 3, 1000, 0),
            GraphEdit::Connect {
                edge: EdgeId::from_raw(1),
                output: PortId::from_raw(2),
                input: PortId::from_raw(3),
            },
        ],
    );
    let view = c.build();
    assert!(view.nodes().is_empty());
    assert_eq!(view.wires().len(), 1);
    assert!(view.wires()[0].segments.iter().any(|r| r[0] == 0.));
    assert!(view.wires()[0].segments.iter().any(|r| r[0] + r[2] == 800.));
    let mut ui = Ui::new(RecordingBackend::new());
    ui.mount(view.element());
    ui.compute_layout(Size {
        width: AvailableSpace::Definite(1000.),
        height: AvailableSpace::Definite(1400.),
    });
    for rect in &view.wires()[0].segments {
        assert!(
            rect[0] >= 0.
                && rect[1] >= 0.
                && rect[0] + rect[2] <= 800.
                && rect[1] + rect[3] <= 480.
        );
        assert!(ui.backend().ops().iter().any(|op|matches!(op,BackendOp::SetLayout{size,..} if (size.width-rect[2]).abs()<0.01 && (size.height-rect[3]).abs()<0.01)),"actual segment layout {rect:?}");
    }
    for i in 0..view.wires()[0].segments.len() {
        let key = if i == 0 {
            "graph:edge:1".to_owned()
        } else {
            format!("graph:edge:1:segment:{i}")
        };
        assert!(find(view.element(), &key).is_some());
        assert!(matches!(
            view.dispatch(&key, &UiEvent::Activate),
            Ok(A::Disconnect(e)) if e == EdgeId::from_raw(1)
        ));
    }
    c.handle_event(&view, "graph:edge:1:segment:1", &UiEvent::Activate)
        .unwrap();
    assert_eq!(c.editor().graph().edge_count(), 0);
}
#[test]
fn visible_budgets_and_query_visits_are_explicit_not_graph_scans() {
    let mut e = GraphEditor::new(Default::default()).unwrap();
    for batch in 0..8 {
        e.execute(&GraphCommand::Edit(
            (0..100)
                .map(|i| {
                    let n = batch * 100 + i + 1;
                    draft(n, n * 2 - 1, n as i32 * 300, 0)
                })
                .collect(),
        ))
        .unwrap();
    }
    let mut c = Controller::new(
        e,
        vec![template()],
        GraphPanelViewport {
            size: [300., 200.],
            origin: [300., 0.],
            zoom: 1.,
        },
        GraphPanelLimits {
            visible_nodes: 1,
            visible_ports: 1,
            query_visits: 64,
            ..Default::default()
        },
    )
    .unwrap();
    let v = c.build();
    assert!(v.stats().query_visits <= 64);
    assert!(v.stats().query_visits < 800);
    assert_eq!(v.nodes().len(), 1);
    assert_eq!(v.ports().len(), 1);
    assert!(v.stats().truncated);
    c.apply(A::Pan([300., 0.])).unwrap();
    let v = c.build();
    assert_eq!(v.nodes()[0].node, NodeId::from_raw(2));
    assert!(c
        .apply(A::SelectRect(
            [0., 0., 300000., 100.],
            SelectionMode::Replace
        ))
        .is_err());
    assert_eq!(c.selected().count(), 0);
}
#[test]
fn partially_visible_nodes_ports_and_wires_have_bounded_geometry() {
    let mut c = controller();
    model(
        &mut c,
        vec![
            draft(1, 1, -100, -20),
            draft(2, 3, 720, 430),
            GraphEdit::Connect {
                edge: EdgeId::from_raw(1),
                output: PortId::from_raw(2),
                input: PortId::from_raw(3),
            },
        ],
    );
    let v = c.build();
    assert_eq!(v.nodes().len(), 2);
    assert!(!v.ports().is_empty());
    for rect in v
        .nodes()
        .iter()
        .map(|n| n.rect)
        .chain(v.ports().iter().map(|p| p.rect))
        .chain(v.wires().iter().flat_map(|w| w.segments.iter().copied()))
    {
        assert!(rect.iter().all(|v| v.is_finite()));
        assert!(rect[2] > 0. && rect[3] > 0.);
        assert!(
            rect[0] >= 0.
                && rect[1] >= 0.
                && rect[0] + rect[2] <= 800.
                && rect[1] + rect[3] <= 480.
        );
    }
    for (key, _) in v.bindings() {
        assert!(find(v.element(), key).is_some());
    }
    let mut ui = Ui::new(RecordingBackend::new());
    ui.mount(v.element());
    ui.compute_layout(Size {
        width: AvailableSpace::Definite(1000.),
        height: AvailableSpace::Definite(1400.),
    });
    for rect in v
        .nodes()
        .iter()
        .map(|n| n.rect)
        .chain(v.ports().iter().map(|p| p.rect))
    {
        assert!(
            ui.backend()
                .ops()
                .iter()
                .any(|op| matches!(op,BackendOp::SetLayout {location,size,..}
            if (location.x-rect[0]).abs()<0.01 && (location.y-rect[1]).abs()<0.01 &&
            (size.width-rect[2]).abs()<0.01 && (size.height-rect[3]).abs()<0.01)),
            "actual clipped node/port layout {rect:?}"
        );
    }
}
#[test]
fn invalid_contributions_and_selection_capacity_do_not_mutate_model() {
    let mut t = template();
    t.ports[1] = t.ports[0].clone();
    assert!(Controller::new(
        GraphEditor::new(Default::default()).unwrap(),
        vec![t],
        Default::default(),
        Default::default()
    )
    .is_err());
    let mut c = Controller::new(
        GraphEditor::new(Default::default()).unwrap(),
        vec![],
        Default::default(),
        GraphPanelLimits {
            selection: 1,
            ..Default::default()
        },
    )
    .unwrap();
    model(&mut c, vec![draft(1, 1, 0, 0), draft(2, 3, 250, 0)]);
    c.apply(A::Select(NodeId::from_raw(1), SelectionMode::Replace))
        .unwrap();
    assert_eq!(
        c.apply(A::Select(NodeId::from_raw(2), SelectionMode::Add)),
        Err(Error::Capacity)
    );
    assert_eq!(c.selected().collect::<Vec<_>>(), vec![NodeId::from_raw(1)]);
    assert!(c.apply(A::Create(0)).is_err());
}
