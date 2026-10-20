//! Real retained Loom node/port/wire controls with bounded spatial queries.
use crate::{
    graph::{self, GraphPanelAction as A, GraphPanelController, GraphPanelError, Item},
    graph_gesture::{GraphGesturePreview, GraphPointerHit},
    SelectionMode, UiEvent, UiTrigger,
};
use prism_editor_graph::{EdgeId, GraphCommand, GraphPoint, NodeId, PortDirection, PortId};
use prism_ui::{
    style::{Keyword, StyleProp, StyleValue},
    Element,
};
use std::{collections::BTreeMap, sync::Arc};

/// Visible node with its exact viewport-clipped logical pixel rectangle.
#[derive(Clone, Debug)]
pub struct GraphNodeVisual {
    /// Kernel stable identity.
    pub node: NodeId,
    /// Painted selectable box key.
    pub key: String,
    /// Left, top, width and height, all within the canvas.
    pub rect: [f32; 4],
    /// Current controller selection.
    pub selected: bool,
}
/// A real typed port and its clipped connection control.
#[derive(Clone, Debug)]
pub struct GraphPortVisual {
    /// Owning node; preserves the interleaved node/port painter order for hits.
    pub node: NodeId,
    /// Stable kernel identity.
    pub port: PortId,
    /// Painted activation key.
    pub key: String,
    /// Clipped control rectangle in canvas pixels.
    pub rect: [f32; 4],
    /// Actual kernel direction.
    pub direction: PortDirection,
    /// Actual host-defined type key, without domain remapping.
    pub type_key: String,
    /// True when this output is armed for connecting to an input.
    pub pending: bool,
}
/// An actual edge, including visible portions when both nodes are off screen.
#[derive(Clone, Debug)]
pub struct GraphWireVisual {
    /// Stable edge identity.
    pub edge: EdgeId,
    /// Painted disconnect key on the first visible segment.
    pub key: String,
    /// Up to three orthogonal clipped segment rectangles in canvas pixels.
    pub segments: Vec<[f32; 4]>,
}
/// Query work and explicit truncation evidence for one published subtree.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GraphPanelViewStats {
    /// Spatial tree branches examined, at most the configured visit budget.
    pub query_visits: usize,
    /// Painted node boxes.
    pub nodes: usize,
    /// Painted typed port controls.
    pub ports: usize,
    /// Painted connected edges, each at most three segments.
    pub edges: usize,
    /// Query or materialization budget prevented a complete visible result.
    pub truncated: bool,
}
/// Exact Loom subtree and event bindings fenced to one controller publication.
#[derive(Clone, Debug)]
pub struct BoundGraphPanel {
    element: Element,
    bindings: BTreeMap<(String, UiTrigger), A>,
    nodes: Vec<GraphNodeVisual>,
    ports: Vec<GraphPortVisual>,
    wires: Vec<GraphWireVisual>,
    preview_rects: Vec<[f32; 4]>,
    viewport: graph::GraphPanelViewport,
    stats: GraphPanelViewStats,
    pub(crate) stamp: Arc<()>,
}
impl BoundGraphPanel {
    /// Effective viewport of this publication, including an uncommitted pan.
    pub const fn viewport(&self) -> graph::GraphPanelViewport {
        self.viewport
    }
    /// At most four clipped overlay rectangles painted above nodes and ports.
    pub fn preview_rects(&self) -> &[[f32; 4]] {
        &self.preview_rects
    }
    /// Hit exact painted controls in reverse painter order. Coordinates are
    /// canvas-local logical pixels; canvas and control bounds are half-open.
    /// Non-finite or outside positions have no target.
    pub fn hit_test(&self, position: [f64; 2]) -> Option<GraphPointerHit> {
        self.viewport.document_point(position)?;
        let inside = |r: [f32; 4]| {
            position[0] >= f64::from(r[0])
                && position[1] >= f64::from(r[1])
                && position[0] < f64::from(r[0] + r[2])
                && position[1] < f64::from(r[1] + r[3])
        };
        let mut ports = self.ports.iter().rev().peekable();
        for node in self.nodes.iter().rev() {
            while ports.peek().is_some_and(|p| p.node == node.node) {
                let port = ports.next().expect("painted owner");
                if inside(port.rect) {
                    return Some(GraphPointerHit::Port(port.port));
                }
            }
            if inside(node.rect) {
                return Some(GraphPointerHit::Node(node.node));
            }
        }
        Some(GraphPointerHit::Canvas)
    }
    /// Mount or reconcile in a real Loom Ui.
    pub fn element(&self) -> &Element {
        &self.element
    }
    /// Painted node geometry.
    pub fn nodes(&self) -> &[GraphNodeVisual] {
        &self.nodes
    }
    /// Painted port geometry and actual type keys.
    pub fn ports(&self) -> &[GraphPortVisual] {
        &self.ports
    }
    /// Painted connections, including crossing offscreen endpoints.
    pub fn wires(&self) -> &[GraphWireVisual] {
        &self.wires
    }
    /// Bounded materialization evidence.
    pub const fn stats(&self) -> GraphPanelViewStats {
        self.stats
    }
    /// Only enabled controls, each backed by a painted key.
    pub fn bindings(&self) -> impl Iterator<Item = (&str, UiTrigger)> {
        self.bindings.keys().map(|(key, t)| (key.as_str(), *t))
    }
    /// Resolve one semantic event; use controller handle_event for freshness.
    pub fn dispatch(&self, key: &str, event: &UiEvent) -> Result<A, GraphPanelError> {
        let trigger = match event {
            UiEvent::Activate => UiTrigger::Activate,
            UiEvent::Select { .. } => UiTrigger::Select,
            UiEvent::Commit { .. } => return Err(GraphPanelError::UnboundEvent),
        };
        let mut action = self
            .bindings
            .get(&(key.to_owned(), trigger))
            .cloned()
            .ok_or(GraphPanelError::UnboundEvent)?;
        if let (A::Select(_, mode), UiEvent::Select { mode: m }) = (&mut action, event) {
            *mode = *m;
        }
        Ok(action)
    }
}
fn column() -> Element {
    Element::box_().style(
        StyleProp::FlexDirection,
        StyleValue::keyword(Keyword::Column),
    )
}
fn box_at(rect: [f32; 4]) -> Element {
    Element::box_()
        .style(StyleProp::Position, StyleValue::keyword(Keyword::Absolute))
        .style(StyleProp::Left, StyleValue::px(rect[0]))
        .style(StyleProp::Top, StyleValue::px(rect[1]))
        .style(StyleProp::Width, StyleValue::px(rect[2]))
        .style(StyleProp::Height, StyleValue::px(rect[3]))
        .style(StyleProp::FlexShrink, StyleValue::number(0.))
}
fn button(view: &mut BoundGraphPanel, key: &str, label: &str, action: Option<A>) -> Element {
    let enabled = action.is_some();
    if let Some(a) = action {
        view.bindings.insert((key.into(), UiTrigger::Activate), a);
    }
    Element::box_()
        .key_str(key)
        .style(StyleProp::Height, StyleValue::px(24.))
        .style(
            StyleProp::BackgroundColor,
            StyleValue::rgba(0.14, 0.18, 0.22, if enabled { 1. } else { 0.3 }),
        )
        .child(Element::text(label))
}
fn short(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}
fn clipped(rect: [f64; 4], size: [f64; 2]) -> Option<[f32; 4]> {
    let [x0, y0, x1, y1] = rect;
    let x0 = x0.max(0.);
    let y0 = y0.max(0.);
    let x1 = x1.min(size[0]);
    let y1 = y1.min(size[1]);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some([x0 as f32, y0 as f32, (x1 - x0) as f32, (y1 - y0) as f32])
}
fn screen(c: &GraphPanelController, p: [f64; 2]) -> [f64; 2] {
    let viewport = c.display_viewport();
    [
        (p[0] - viewport.origin[0]) * viewport.zoom,
        (p[1] - viewport.origin[1]) * viewport.zoom,
    ]
}
pub(crate) fn build(c: &GraphPanelController) -> BoundGraphPanel {
    let viewport = c.display_viewport();
    let mut view = BoundGraphPanel {
        element: Element::box_(),
        bindings: BTreeMap::new(),
        nodes: Vec::new(),
        ports: Vec::new(),
        wires: Vec::new(),
        preview_rects: Vec::new(),
        viewport,
        stats: Default::default(),
        stamp: c.stamp.clone(),
    };
    let mut tools = column()
        .key_str("graph:tools")
        .style(StyleProp::Width, StyleValue::px(160.))
        .style(StyleProp::FlexShrink, StyleValue::number(0.));
    for (i, t) in c.templates.iter().enumerate() {
        tools = tools.child(button(
            &mut view,
            &format!("graph:create:{i}"),
            &format!("+ {}", short(&t.name, 16)),
            Some(A::Create(i)),
        ));
    }
    for (key, label, action) in [
        (
            "graph:undo",
            "Undo",
            (c.editor.undo_depth() > 0).then_some(A::Command(GraphCommand::Undo)),
        ),
        (
            "graph:redo",
            "Redo",
            (c.editor.redo_depth() > 0).then_some(A::Command(GraphCommand::Redo)),
        ),
        (
            "graph:delete",
            "Delete selection",
            (!c.selected.is_empty()).then_some(A::DeleteSelection),
        ),
        ("graph:clear", "Clear selection", Some(A::Clear)),
    ] {
        tools = tools.child(button(&mut view, key, label, action));
    }
    for (name, dx, dy) in [
        ("left", -20, 0),
        ("right", 20, 0),
        ("up", 0, -20),
        ("down", 0, 20),
    ] {
        tools = tools.child(button(
            &mut view,
            &format!("graph:move:{name}"),
            &format!("Move {name}"),
            (!c.selected.is_empty()).then_some(A::MoveSelection(GraphPoint { x: dx, y: dy })),
        ));
        let mut v = c.viewport;
        v.origin[0] += f64::from(dx) * 5.;
        v.origin[1] += f64::from(dy) * 5.;
        tools = tools.child(button(
            &mut view,
            &format!("graph:pan:{name}"),
            &format!("Pan {name}"),
            v.validate()
                .ok()
                .map(|_| A::Pan([f64::from(dx) * 5., f64::from(dy) * 5.])),
        ));
    }
    for (name, factor) in [("in", 2.), ("out", 0.5)] {
        let mut v = c.viewport;
        v.zoom *= factor;
        for i in 0..2 {
            v.origin[i] += v.size[i] / c.viewport.zoom / 2. - v.size[i] / v.zoom / 2.;
        }
        tools = tools.child(button(
            &mut view,
            &format!("graph:zoom:{name}"),
            &format!("Zoom {name}"),
            v.validate().ok().map(|_| A::Zoom(factor)),
        ));
    }
    let mut canvas = Element::box_()
        .key_str("graph:canvas")
        .style(StyleProp::Position, StyleValue::keyword(Keyword::Relative))
        .style(StyleProp::Width, StyleValue::px(viewport.size[0] as f32))
        .style(StyleProp::Height, StyleValue::px(viewport.size[1] as f32))
        .style(StyleProp::FlexShrink, StyleValue::number(0.))
        .style(
            StyleProp::BackgroundColor,
            StyleValue::rgba(0.04, 0.05, 0.07, 1.),
        );
    let query = c.index.query(
        c.preview_query(),
        c.limits.query_visits,
        c.limits.visible_nodes + c.limits.visible_edges,
    );
    view.stats.query_visits = query.visits;
    view.stats.truncated = query.truncated;
    // Paint wires first, behind node boxes and typed port controls.
    for item in &query.items {
        let Item::Edge(edge) = *item else {
            continue;
        };
        if view.wires.len() == c.limits.visible_edges {
            view.stats.truncated = true;
            continue;
        }
        let mut points = graph::wire_points(&c.editor, edge);
        let e = c.editor.graph().edge(edge).expect("indexed edge");
        for (index, port) in [(0, e.output()), (3, e.input())] {
            let owner = c.editor.graph().port(port).expect("indexed port").node();
            let offset = c.preview_offset(owner);
            for (axis, d) in offset.into_iter().enumerate() {
                points[index][axis] += d;
            }
        }
        let middle = (points[0][0] + points[3][0]) / 2.;
        points[1] = [middle, points[0][1]];
        points[2] = [middle, points[3][1]];
        let mut segments = Vec::new();
        for ends in points.windows(2) {
            let a = screen(c, ends[0]);
            let b = screen(c, ends[1]);
            let rect = if a[1] == b[1] {
                [a[0].min(b[0]), a[1] - 2., a[0].max(b[0]), a[1] + 2.]
            } else {
                [a[0] - 2., a[1].min(b[1]), a[0] + 2., a[1].max(b[1])]
            };
            if let Some(rect) = clipped(rect, viewport.size) {
                segments.push(rect);
            }
        }
        if segments.is_empty() {
            continue;
        }
        let key = format!("graph:edge:{}", edge.raw());
        for (i, rect) in segments.iter().copied().enumerate() {
            let segment_key = if i == 0 {
                key.clone()
            } else {
                format!("{key}:segment:{i}")
            };
            view.bindings.insert(
                (segment_key.clone(), UiTrigger::Activate),
                A::Disconnect(edge),
            );
            canvas = canvas.child(box_at(rect).key_str(segment_key).style(
                StyleProp::BackgroundColor,
                StyleValue::rgba(0.45, 0.70, 0.95, 1.),
            ));
        }
        view.wires.push(GraphWireVisual {
            edge,
            key,
            segments,
        });
    }
    for item in query.items {
        let Item::Node(n) = item else {
            continue;
        };
        if view.nodes.len() == c.limits.visible_nodes {
            view.stats.truncated = true;
            continue;
        }
        let mut r = graph::node_rect(&c.editor, n).0;
        let offset = c.preview_offset(n);
        for i in 0..4 {
            r[i] += offset[i % 2];
        }
        let a = screen(c, [r[0], r[1]]);
        let b = screen(c, [r[2], r[3]]);
        let Some(rect) = clipped([a[0], a[1], b[0], b[1]], viewport.size) else {
            continue;
        };
        let node = c.editor.graph().node(n).expect("indexed node");
        let key = format!("graph:node:{}", n.raw());
        let selected = c.selected.contains(&n);
        let mut element = box_at(rect).key_str(&key).style(
            StyleProp::BackgroundColor,
            if selected {
                StyleValue::rgba(0.18, 0.32, 0.46, 1.)
            } else {
                StyleValue::rgba(0.12, 0.16, 0.20, 1.)
            },
        );
        // Text is emitted only when its entire title row fits the clipped box.
        if a[0] >= 0.
            && a[1] >= 0.
            && b[0] <= viewport.size[0]
            && viewport.zoom >= 0.75
            && rect[3] >= 24.
        {
            element = element.child(Element::text(short(node.name(), 20)));
        }
        canvas = canvas.child(element);
        view.bindings.insert(
            (key.clone(), UiTrigger::Activate),
            A::Select(n, SelectionMode::Replace),
        );
        view.bindings.insert(
            (key.clone(), UiTrigger::Select),
            A::Select(n, SelectionMode::Replace),
        );
        view.nodes.push(GraphNodeVisual {
            node: n,
            key,
            rect,
            selected,
        });
        for id in node.ports() {
            let p = c.editor.graph().port(*id).expect("real port");
            let mut at = graph::port_point(&c.editor, *id);
            for i in 0..2 {
                at[i] += offset[i];
            }
            let (x0, x1) = if p.direction() == PortDirection::Input {
                (at[0], at[0] + 90.)
            } else {
                (at[0] - 90., at[0])
            };
            let a = screen(c, [x0, at[1] - 8.]);
            let b = screen(c, [x1, at[1] + 8.]);
            let Some(rect) = clipped([a[0], a[1], b[0], b[1]], viewport.size) else {
                continue;
            };
            if view.ports.len() == c.limits.visible_ports {
                view.stats.truncated = true;
                continue;
            }
            let key = format!("graph:port:{}", id.raw());
            let pending = c.pending == Some(*id);
            let mut element = box_at(rect).key_str(&key).style(
                StyleProp::BackgroundColor,
                if pending {
                    StyleValue::rgba(0.80, 0.60, 0.12, 1.)
                } else {
                    StyleValue::rgba(0.22, 0.32, 0.38, 1.)
                },
            );
            if rect[2] >= 88. && rect[3] >= 16. {
                element = element.child(Element::text(format!(
                    "{}:{}",
                    short(p.name(), 5),
                    short(p.port_type().key(), 5)
                )));
            }
            canvas = canvas.child(element);
            view.bindings
                .insert((key.clone(), UiTrigger::Activate), A::PickPort(*id));
            view.ports.push(GraphPortVisual {
                node: n,
                port: *id,
                key,
                rect,
                direction: p.direction(),
                type_key: p.port_type().key().into(),
                pending,
            });
        }
    }
    let mut overlay = Vec::with_capacity(4);
    match c.gesture_preview() {
        Some(GraphGesturePreview::Connection { from, to, .. }) => {
            let mid = (from[0] + to[0]) / 2.;
            for ends in [from, [mid, from[1]], [mid, to[1]], to].windows(2) {
                let (a, b) = (ends[0], ends[1]);
                overlay.push(if a[1] == b[1] {
                    [a[0].min(b[0]), a[1] - 1., a[0].max(b[0]), a[1] + 1.]
                } else {
                    [a[0] - 1., a[1].min(b[1]), a[0] + 1., a[1].max(b[1])]
                });
            }
        }
        Some(GraphGesturePreview::Marquee { rect }) => {
            let a = screen(c, [rect[0], rect[1]]);
            let b = screen(c, [rect[2], rect[3]]);
            overlay.extend([
                [a[0], a[1], b[0], a[1] + 1.],
                [a[0], b[1] - 1., b[0], b[1]],
                [a[0], a[1], a[0] + 1., b[1]],
                [b[0] - 1., a[1], b[0], b[1]],
            ]);
        }
        _ => {}
    }
    for rect in overlay {
        if let Some(rect) = clipped(rect, viewport.size) {
            let i = view.preview_rects.len();
            canvas = canvas.child(box_at(rect).key_str(format!("graph:gesture:{i}")).style(
                StyleProp::BackgroundColor,
                StyleValue::rgba(0.95, 0.70, 0.15, 1.),
            ));
            view.preview_rects.push(rect);
        }
    }
    view.stats.nodes = view.nodes.len();
    view.stats.ports = view.ports.len();
    view.stats.edges = view.wires.len();
    let status = format!(
        "{} nodes / {} edges | selected {}{}",
        c.editor.graph().node_count(),
        c.editor.graph().edge_count(),
        c.selected.len(),
        if view.stats.truncated {
            " | visible budget reached"
        } else {
            ""
        }
    );
    let body = Element::box_()
        .style(StyleProp::FlexDirection, StyleValue::keyword(Keyword::Row))
        .child(tools)
        .child(canvas);
    view.element = column()
        .key_str("graph:panel")
        .style(
            StyleProp::Width,
            StyleValue::px(viewport.size[0] as f32 + 160.),
        )
        .child(body)
        .child(Element::text(status));
    view
}
