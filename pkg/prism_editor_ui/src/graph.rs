//! Host-contributed graph authoring controller, independent of engine domains.
use crate::{graph_view::BoundGraphPanel, SelectionMode, UiEvent};
use prism_editor_graph::{
    EdgeId, GraphAuthoringError, GraphCommand, GraphEdit, GraphEditor, GraphNodeDraft, GraphPoint,
    GraphPortDraft, NodeId, PortDirection, PortId, PortSpec,
};
use std::{collections::BTreeSet, fmt, sync::Arc};

/// Host-supplied creation recipe. No domain node kinds are built into this UI.
#[derive(Clone, Debug)]
pub struct GraphNodeTemplate {
    /// Node display name.
    pub name: String,
    /// Opaque host data, copied only when creating a node.
    pub payload: Vec<u8>,
    /// Actual kernel port types, directions and cardinalities.
    pub ports: Vec<PortSpec>,
}
/// Viewport-local pixels mapped to integer document coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GraphPanelViewport {
    /// Width and height, each 64 through 4096 logical pixels.
    pub size: [f64; 2],
    /// Document coordinate at the canvas top left, bounded to +/- one million.
    pub origin: [f64; 2],
    /// Logical pixels per document unit, from 0.25 through 4.
    pub zoom: f64,
}
impl Default for GraphPanelViewport {
    fn default() -> Self {
        Self {
            size: [800., 480.],
            origin: [0., 0.],
            zoom: 1.,
        }
    }
}
impl GraphPanelViewport {
    /// Reject non-finite values and out-of-range geometry before publication.
    pub fn validate(self) -> Result<(), GraphPanelError> {
        if self
            .size
            .iter()
            .any(|n| !n.is_finite() || !(64. ..=4096.).contains(n))
            || self
                .origin
                .iter()
                .any(|n| !n.is_finite() || n.abs() > 1_000_000.)
            || !self.zoom.is_finite()
            || !(0.25..=4.).contains(&self.zoom)
        {
            return Err(GraphPanelError::Geometry);
        }
        Ok(())
    }
    /// Convert a point inside the half-open canvas to document coordinates.
    pub fn document_point(self, pixel: [f64; 2]) -> Option<[f64; 2]> {
        self.validate().ok()?;
        if (0..2).any(|i| !pixel[i].is_finite() || pixel[i] < 0. || pixel[i] >= self.size[i]) {
            return None;
        }
        Some([
            self.origin[0] + pixel[0] / self.zoom,
            self.origin[1] + pixel[1] / self.zoom,
        ])
    }
    pub(crate) fn bounds(self) -> Rect {
        Rect([
            self.origin[0],
            self.origin[1],
            self.origin[0] + self.size[0] / self.zoom,
            self.origin[1] + self.size[1] / self.zoom,
        ])
    }
}
/// Hard ceilings constrain both spatial-query work and retained UI materialization.
#[derive(Clone, Copy, Debug)]
pub struct GraphPanelLimits {
    /// Host templates, at most 16.
    pub templates: usize,
    /// Selected nodes and atomic multi-node edits, at most 1024.
    pub selection: usize,
    /// Visible node boxes, at most 256.
    pub visible_nodes: usize,
    /// Visible port controls, at most 2048.
    pub visible_ports: usize,
    /// Visible edges, at most 512, each with at most three clipped segments.
    pub visible_edges: usize,
    /// BVH branches visited per build/rectangle selection, at most 8192.
    pub query_visits: usize,
}
impl Default for GraphPanelLimits {
    fn default() -> Self {
        Self {
            templates: 16,
            selection: 128,
            visible_nodes: 128,
            visible_ports: 512,
            visible_edges: 256,
            query_visits: 4096,
        }
    }
}
impl GraphPanelLimits {
    fn validate(self) -> Result<(), GraphPanelError> {
        for (n, max) in [
            (self.templates, 16),
            (self.selection, 1024),
            (self.visible_nodes, 256),
            (self.visible_ports, 2048),
            (self.visible_edges, 512),
            (self.query_visits, 8192),
        ] {
            if n == 0 || n > max {
                return Err(GraphPanelError::Capacity);
            }
        }
        Ok(())
    }
}
/// Actual model edits and viewport/selection actions from Loom or pointer adapters.
#[derive(Clone, Debug)]
pub enum GraphPanelAction {
    /// Execute an atomic, serializable kernel command.
    Command(GraphCommand),
    /// Instantiate a contributed template at the current canvas center.
    Create(usize),
    /// Select a stable node identity with the requested selection behavior.
    Select(NodeId, SelectionMode),
    /// Select nodes intersecting a document-space rectangle; incomplete queries fail.
    SelectRect([f64; 4], SelectionMode),
    /// Clear selection and pending connection.
    Clear,
    /// Move all selected nodes in one reversible command by integer document units.
    MoveSelection(GraphPoint),
    /// Remove selected nodes and their incident edges in one reversible command.
    DeleteSelection,
    /// Choose output, then input to create a real typed connection.
    PickPort(PortId),
    /// Remove one visible connection.
    Disconnect(EdgeId),
    /// Pan by document-space units.
    Pan([f64; 2]),
    /// Zoom by a positive factor around the canvas center.
    Zoom(f64),
    /// Replace the validated viewport.
    Viewport(GraphPanelViewport),
}
/// Rejected UI operation; model failures retain graph/history state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GraphPanelError {
    /// Non-finite or out-of-range geometry.
    Geometry,
    /// A configured bound or selection/query capacity was exceeded.
    Capacity,
    /// Unknown template, node, port or missing pending output.
    Target,
    /// View belongs to another controller or an older publication.
    StaleView,
    /// No enabled control matches the semantic event.
    UnboundEvent,
    /// Atomic authoring command rejected by the kernel.
    Authoring(GraphAuthoringError),
    /// A pointer is already captured or there is no active capture.
    GestureState,
}
impl fmt::Display for GraphPanelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for GraphPanelError {}
impl From<GraphAuthoringError> for GraphPanelError {
    fn from(e: GraphAuthoringError) -> Self {
        Self::Authoring(e)
    }
}
/// Controller owns the editor and a spatial index rebuilt after model commands.
/// View-only changes use bounded index queries, without enumerating the graph.
#[derive(Debug)]
pub struct GraphPanelController {
    pub(crate) editor: GraphEditor,
    pub(crate) templates: Vec<GraphNodeTemplate>,
    pub(crate) viewport: GraphPanelViewport,
    pub(crate) limits: GraphPanelLimits,
    pub(crate) selected: BTreeSet<NodeId>,
    pub(crate) pending: Option<PortId>,
    pub(crate) index: SpatialIndex,
    pub(crate) stamp: Arc<()>,
    pub(crate) gesture: Option<crate::graph_gesture::GraphGesture>,
}
impl GraphPanelController {
    /// Accept a real authoring model and a bounded host contribution set.
    pub fn new(
        editor: GraphEditor,
        templates: Vec<GraphNodeTemplate>,
        viewport: GraphPanelViewport,
        limits: GraphPanelLimits,
    ) -> Result<Self, GraphPanelError> {
        limits.validate()?;
        viewport.validate()?;
        if templates.len() > limits.templates || limits.selection > editor.limits().edits {
            return Err(GraphPanelError::Capacity);
        }
        for t in &templates {
            let l = editor.limits();
            if t.name.len() > l.text_bytes
                || t.payload.len() > l.payload_bytes
                || t.ports.len() > l.ports_per_node
                || t.ports.iter().any(|p| {
                    p.name().len() > l.text_bytes || p.port_type().key().len() > l.text_bytes
                })
            {
                return Err(GraphPanelError::Capacity);
            }
            // Validate the complete recipe via the canonical kernel protocol.
            let d = GraphNodeDraft {
                id: NodeId::from_raw(1),
                name: t.name.clone(),
                payload: t.payload.clone(),
                position: GraphPoint::default(),
                ports: t
                    .ports
                    .iter()
                    .enumerate()
                    .map(|(i, s)| GraphPortDraft {
                        id: PortId::from_raw(i as u64 + 1),
                        spec: s.clone(),
                    })
                    .collect(),
            };
            let mut probe = GraphEditor::new(editor.limits())?;
            probe.execute(&GraphCommand::Edit(vec![GraphEdit::CreateNode(d)]))?;
        }
        let index = SpatialIndex::build(&editor);
        Ok(Self {
            editor,
            templates,
            viewport,
            limits,
            selected: BTreeSet::new(),
            pending: None,
            index,
            stamp: Arc::new(()),
            gesture: None,
        })
    }
    /// Read-only real model; all mutation passes through atomic commands.
    pub fn editor(&self) -> &GraphEditor {
        &self.editor
    }
    /// Current validated viewport.
    pub const fn viewport(&self) -> GraphPanelViewport {
        self.viewport
    }
    /// Selected stable identities in ascending order.
    pub fn selected(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.selected.iter().copied()
    }
    /// Currently armed typed output port, if any.
    pub const fn pending_output(&self) -> Option<PortId> {
        self.pending
    }
    /// Produce a bounded real Loom subtree with exact event bindings.
    pub fn build(&self) -> BoundGraphPanel {
        crate::graph_view::build(self)
    }
    /// Apply a binding only to the exact current controller publication.
    pub fn handle_event(
        &mut self,
        view: &BoundGraphPanel,
        key: &str,
        event: &UiEvent,
    ) -> Result<(), GraphPanelError> {
        if !Arc::ptr_eq(&self.stamp, &view.stamp) {
            return Err(GraphPanelError::StaleView);
        }
        self.apply(view.dispatch(key, event)?)
    }
    /// Apply a model, selection or navigation operation. Failed actions do not
    /// invalidate current bindings or clear a pending typed connection.
    pub fn apply(&mut self, action: GraphPanelAction) -> Result<(), GraphPanelError> {
        match action {
            GraphPanelAction::Command(c) => self.command(&c)?,
            GraphPanelAction::Create(i) => {
                let t = self.templates.get(i).ok_or(GraphPanelError::Target)?;
                let (node, port, _) = self.editor.next_ids();
                let center = [
                    self.viewport.origin[0] + self.viewport.size[0] / self.viewport.zoom / 2.,
                    self.viewport.origin[1] + self.viewport.size[1] / self.viewport.zoom / 2.,
                ];
                let d = GraphNodeDraft {
                    id: node,
                    name: t.name.clone(),
                    payload: t.payload.clone(),
                    position: GraphPoint {
                        x: (center[0] - 100.).round() as i32,
                        y: (center[1] - 40.).round() as i32,
                    },
                    ports: t
                        .ports
                        .iter()
                        .enumerate()
                        .map(|(i, s)| GraphPortDraft {
                            id: PortId::from_raw(port.raw().saturating_add(i as u64)),
                            spec: s.clone(),
                        })
                        .collect(),
                };
                self.command(&GraphCommand::Edit(vec![GraphEdit::CreateNode(d)]))?;
                self.selected.clear();
                self.selected.insert(node);
            }
            GraphPanelAction::Select(n, mode) => {
                if self.editor.graph().node(n).is_none() {
                    return Err(GraphPanelError::Target);
                }
                self.select([n], mode)?;
            }
            GraphPanelAction::SelectRect(rect, mode) => {
                let rect = Rect::checked(rect)?;
                let q = self
                    .index
                    .query(rect, self.limits.query_visits, self.limits.selection + 1);
                if q.truncated {
                    return Err(GraphPanelError::Capacity);
                }
                let ids = q.items.into_iter().filter_map(|i| match i {
                    Item::Node(n) => Some(n),
                    Item::Edge(_) => None,
                });
                self.select(ids, mode)?;
            }
            GraphPanelAction::Clear => {
                self.selected.clear();
                self.pending = None;
            }
            GraphPanelAction::MoveSelection(delta) => {
                if self.selected.is_empty() {
                    return Err(GraphPanelError::Target);
                }
                let edits = self
                    .selected
                    .iter()
                    .map(|n| {
                        let p = self.editor.position(*n).expect("selected node exists");
                        Ok(GraphEdit::MoveNode {
                            node: *n,
                            position: GraphPoint {
                                x: p.x.checked_add(delta.x).ok_or(GraphPanelError::Geometry)?,
                                y: p.y.checked_add(delta.y).ok_or(GraphPanelError::Geometry)?,
                            },
                        })
                    })
                    .collect::<Result<Vec<_>, GraphPanelError>>()?;
                self.command(&GraphCommand::Edit(edits))?;
            }
            GraphPanelAction::DeleteSelection => {
                if self.selected.is_empty() {
                    return Err(GraphPanelError::Target);
                }
                self.command(&GraphCommand::Edit(
                    self.selected
                        .iter()
                        .copied()
                        .map(GraphEdit::DeleteNode)
                        .collect(),
                ))?;
            }
            GraphPanelAction::PickPort(id) => {
                let p = self
                    .editor
                    .graph()
                    .port(id)
                    .ok_or(GraphPanelError::Target)?;
                if p.direction() == PortDirection::Output {
                    self.pending = Some(id);
                } else {
                    let output = self.pending.ok_or(GraphPanelError::Target)?;
                    self.command(&GraphCommand::Edit(vec![GraphEdit::Connect {
                        edge: self.editor.next_ids().2,
                        output,
                        input: id,
                    }]))?;
                    self.pending = None;
                }
            }
            GraphPanelAction::Disconnect(e) => {
                self.command(&GraphCommand::Edit(vec![GraphEdit::Disconnect(e)]))?;
            }
            GraphPanelAction::Pan(delta) => {
                let mut v = self.viewport;
                for (p, d) in v.origin.iter_mut().zip(delta) {
                    *p += d;
                }
                v.validate()?;
                self.viewport = v;
            }
            GraphPanelAction::Zoom(factor) => {
                let mut v = self.viewport;
                v.zoom *= factor;
                for i in 0..2 {
                    v.origin[i] += v.size[i] / self.viewport.zoom / 2. - v.size[i] / v.zoom / 2.;
                }
                v.validate()?;
                self.viewport = v;
            }
            GraphPanelAction::Viewport(v) => {
                v.validate()?;
                self.viewport = v;
            }
        }
        self.stamp = Arc::new(());
        self.gesture = None;
        Ok(())
    }
    fn command(&mut self, c: &GraphCommand) -> Result<(), GraphPanelError> {
        self.editor.execute(c)?;
        self.selected
            .retain(|n| self.editor.graph().node(*n).is_some());
        self.pending = self
            .pending
            .filter(|p| self.editor.graph().port(*p).is_some());
        self.index = SpatialIndex::build(&self.editor);
        Ok(())
    }
    fn select(
        &mut self,
        ids: impl IntoIterator<Item = NodeId>,
        mode: SelectionMode,
    ) -> Result<(), GraphPanelError> {
        let mut next = if mode == SelectionMode::Replace {
            BTreeSet::new()
        } else {
            self.selected.clone()
        };
        for n in ids {
            if mode == SelectionMode::Toggle && !next.insert(n) {
                next.remove(&n);
            } else {
                next.insert(n);
            }
            if next.len() > self.limits.selection {
                return Err(GraphPanelError::Capacity);
            }
        }
        self.selected = next;
        Ok(())
    }
}

pub(crate) const NODE_WIDTH: f64 = 200.;
pub(crate) fn node_rect(e: &GraphEditor, n: NodeId) -> Rect {
    let p = e.position(n).expect("index contains real nodes");
    Rect([
        f64::from(p.x),
        f64::from(p.y),
        f64::from(p.x) + NODE_WIDTH,
        f64::from(p.y) + 40. + e.graph().node(n).expect("real node").ports().len() as f64 * 24.,
    ])
}
pub(crate) fn port_point(e: &GraphEditor, id: PortId) -> [f64; 2] {
    let p = e.graph().port(id).expect("edge port exists");
    let r = node_rect(e, p.node()).0;
    let i = e
        .graph()
        .node(p.node())
        .expect("port owner exists")
        .ports()
        .iter()
        .position(|n| *n == id)
        .expect("port order");
    [
        if p.direction() == PortDirection::Input {
            r[0]
        } else {
            r[2]
        },
        r[1] + 40. + i as f64 * 24.,
    ]
}
pub(crate) fn wire_points(e: &GraphEditor, id: EdgeId) -> [[f64; 2]; 4] {
    let edge = e.graph().edge(id).expect("indexed edge exists");
    let a = port_point(e, edge.output());
    let b = port_point(e, edge.input());
    let mid = (a[0] + b[0]) / 2.;
    [a, [mid, a[1]], [mid, b[1]], b]
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Rect(pub [f64; 4]);
impl Rect {
    fn checked(r: [f64; 4]) -> Result<Self, GraphPanelError> {
        if r.iter().any(|x| !x.is_finite() || x.abs() > 1_100_000.) || r[0] > r[2] || r[1] > r[3] {
            return Err(GraphPanelError::Geometry);
        }
        Ok(Self(r))
    }
    pub(crate) fn intersects(self, r: Self) -> bool {
        self.0[0] <= r.0[2] && self.0[2] >= r.0[0] && self.0[1] <= r.0[3] && self.0[3] >= r.0[1]
    }
    fn union(self, r: Self) -> Self {
        Self([
            self.0[0].min(r.0[0]),
            self.0[1].min(r.0[1]),
            self.0[2].max(r.0[2]),
            self.0[3].max(r.0[3]),
        ])
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum Item {
    Node(NodeId),
    Edge(EdgeId),
}
#[derive(Debug)]
enum Branch {
    Leaf(Rect, Item),
    Pair(Rect, usize, usize),
}
impl Branch {
    fn bounds(&self) -> Rect {
        match self {
            Self::Leaf(r, _) | Self::Pair(r, _, _) => *r,
        }
    }
}
#[derive(Debug, Default)]
pub(crate) struct SpatialIndex {
    branches: Vec<Branch>,
    root: Option<usize>,
}
pub(crate) struct Query {
    pub items: Vec<Item>,
    pub visits: usize,
    pub truncated: bool,
}
impl SpatialIndex {
    fn build(e: &GraphEditor) -> Self {
        let mut leaves: Vec<_> = e
            .positions()
            .map(|(n, _)| (node_rect(e, n), Item::Node(n)))
            .collect();
        for edge in e.graph().edges() {
            let points = wire_points(e, edge.id());
            let mut r = Rect([points[0][0], points[0][1], points[0][0], points[0][1]]);
            for p in &points[1..] {
                r = r.union(Rect([p[0], p[1], p[0], p[1]]));
            }
            // Stroke thickness is two screen pixels; at minimum zoom this
            // reaches eight document units beyond the mathematical route.
            leaves.push((
                Rect([r.0[0] - 8., r.0[1] - 8., r.0[2] + 8., r.0[3] + 8.]),
                Item::Edge(edge.id()),
            ));
        }
        fn split(index: &mut SpatialIndex, leaves: &mut [(Rect, Item)], depth: usize) -> usize {
            let b = if leaves.len() == 1 {
                Branch::Leaf(leaves[0].0, leaves[0].1)
            } else {
                let axis = depth % 2;
                leaves.sort_by(|(a, ia), (b, ib)| {
                    (a.0[axis] + a.0[axis + 2])
                        .total_cmp(&(b.0[axis] + b.0[axis + 2]))
                        .then(ia.cmp(ib))
                });
                let mid = leaves.len() / 2;
                let (a, b) = leaves.split_at_mut(mid);
                let left = split(index, a, depth + 1);
                let right = split(index, b, depth + 1);
                Branch::Pair(
                    index.branches[left]
                        .bounds()
                        .union(index.branches[right].bounds()),
                    left,
                    right,
                )
            };
            let id = index.branches.len();
            index.branches.push(b);
            id
        }
        let mut index = Self::default();
        if !leaves.is_empty() {
            index.root = Some(split(&mut index, &mut leaves, 0));
        }
        index
    }
    pub(crate) fn query(&self, r: Rect, max_visits: usize, max_items: usize) -> Query {
        let mut q = Query {
            items: Vec::new(),
            visits: 0,
            truncated: false,
        };
        let mut stack = Vec::new();
        if let Some(root) = self.root {
            stack.push(root);
        }
        while let Some(i) = stack.pop() {
            if q.visits == max_visits {
                q.truncated = true;
                break;
            }
            q.visits += 1;
            let b = &self.branches[i];
            if !b.bounds().intersects(r) {
                continue;
            }
            match *b {
                Branch::Leaf(_, item) => {
                    if q.items.len() == max_items {
                        q.truncated = true;
                        break;
                    }
                    q.items.push(item);
                }
                Branch::Pair(_, a, b) => {
                    stack.push(b);
                    stack.push(a);
                }
            }
        }
        q.items.sort();
        q
    }
}
