//! Painted viewport-clipped Loom world grid and exact semantic bindings.
use crate::{
    world::{
        WorldGridPlane, WorldPanelAction as Action, WorldPanelController, WorldPanelError,
        WorldPanelLease,
    },
    UiEvent, UiTrigger,
};
use prism_ecs::partition::cell::CellCoord;
use prism_editor_world::CellResidency;
use prism_ui::{
    style::{Keyword, StyleProp, StyleValue},
    Element,
};
use std::{collections::BTreeMap, sync::Arc};

/// Exact clipped geometry and real model metadata of a visible grid slot.
#[derive(Clone, Debug, PartialEq)]
pub struct WorldCellVisual {
    /// Stable key of the painted cell box.
    pub key: String,
    /// Actual world coordinate in the chosen plane/slice.
    pub cell: CellCoord,
    /// Left/top/width/height in viewport-local logical pixels.
    pub rect: [f32; 4],
    /// None means this grid coordinate has no registered cell.
    pub residency: Option<CellResidency>,
    /// Real model dirty flag; absent slots report false.
    pub dirty: bool,
    /// Real cell revision, if registered.
    pub revision: Option<u64>,
    /// Controller's actual lease observation.
    pub lease: WorldPanelLease,
    /// Whether the registered slot is selected.
    pub selected: bool,
}
/// Bounded materialization counts for one viewport build.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WorldPanelViewStats {
    /// Coordinate lookups, including missing cells; capped by visible_cells.
    pub cell_lookups: usize,
    /// Registered cells intersecting the viewport.
    pub registered_cells: usize,
    /// Project layer rows materialized, capped by visible_layers.
    pub visible_layers: usize,
    /// Project has additional layer rows after this bounded page.
    pub has_more_layers: bool,
    /// Pending host intent count at publication.
    pub queued_intents: usize,
}
/// Actual Loom subtree with bindings fenced to one controller publication.
#[derive(Clone, Debug)]
pub struct BoundWorldPanel {
    element: Element,
    bindings: BTreeMap<(String, UiTrigger), Action>,
    cells: Vec<WorldCellVisual>,
    stats: WorldPanelViewStats,
    pub(crate) stamp: Arc<()>,
}
impl BoundWorldPanel {
    /// Mount/reconcile this subtree in a real Loom UI.
    pub fn element(&self) -> &Element {
        &self.element
    }
    /// Painted boxes and clipped geometry for native viewport hit adapters.
    pub fn cells(&self) -> &[WorldCellVisual] {
        &self.cells
    }
    /// Materialization evidence without walking the full tree.
    pub const fn stats(&self) -> WorldPanelViewStats {
        self.stats
    }
    /// Only enabled controls appear here; each key exists in element().
    pub fn bindings(&self) -> impl Iterator<Item = (&str, UiTrigger)> {
        self.bindings.keys().map(|(k, t)| (k.as_str(), *t))
    }
    /// Pure binding resolution; controller handle_event enforces freshness.
    pub fn dispatch(&self, key: &str, event: &UiEvent) -> Result<Action, WorldPanelError> {
        let trigger = match event {
            UiEvent::Activate => UiTrigger::Activate,
            UiEvent::Select { .. } => UiTrigger::Select,
            UiEvent::Commit { .. } => return Err(WorldPanelError::UnboundEvent),
        };
        self.bindings
            .get(&(key.to_owned(), trigger))
            .cloned()
            .ok_or(WorldPanelError::UnboundEvent)
    }
}
fn row() -> Element {
    Element::box_().style(StyleProp::FlexDirection, StyleValue::keyword(Keyword::Row))
}
fn column() -> Element {
    Element::box_().style(
        StyleProp::FlexDirection,
        StyleValue::keyword(Keyword::Column),
    )
}
fn button(
    view: &mut BoundWorldPanel,
    key: &str,
    text: impl Into<String>,
    action: Option<Action>,
) -> Element {
    let enabled = action.is_some();
    if let Some(action) = action {
        view.bindings
            .insert((key.into(), UiTrigger::Activate), action);
    }
    Element::box_()
        .key_str(key)
        .class(if enabled {
            "world-control"
        } else {
            "world-control-disabled"
        })
        .style(StyleProp::MinHeight, StyleValue::px(24.))
        .style(StyleProp::MinWidth, StyleValue::px(56.))
        .style(
            StyleProp::BackgroundColor,
            StyleValue::rgba(0.12, 0.16, 0.20, if enabled { 1. } else { 0.4 }),
        )
        .child(Element::text(text.into()))
}
fn lease_text(lease: WorldPanelLease) -> String {
    match lease {
        WorldPanelLease::Unobserved => "lease:unobserved".into(),
        WorldPanelLease::Expired => "lease:expired".into(),
        WorldPanelLease::Owned { expires_at } => format!("lease:owned<{expires_at}"),
    }
}
pub(crate) fn build(c: &WorldPanelController) -> BoundWorldPanel {
    let mut view = BoundWorldPanel {
        element: Element::box_(),
        bindings: BTreeMap::new(),
        cells: Vec::new(),
        stats: WorldPanelViewStats {
            queued_intents: c.pending_intents(),
            ..Default::default()
        },
        stamp: c.stamp.clone(),
    };
    let mut tools = row().key_str("world:toolbar");
    for (key, text, factor) in [("world:zoom-in", "+", 2.), ("world:zoom-out", "−", 0.5)] {
        let mut n = c.viewport;
        n.cell_pixels *= factor;
        for i in 0..2 {
            n.origin[i] +=
                c.viewport.size[i] / c.viewport.cell_pixels / 2. - n.size[i] / n.cell_pixels / 2.;
        }
        tools = tools.child(button(
            &mut view,
            key,
            text,
            n.bounds(c.limits.visible_cells)
                .ok()
                .map(|_| Action::Zoom(factor)),
        ));
    }
    for (key, text, delta) in [
        ("world:left", "←", [-1., 0.]),
        ("world:right", "→", [1., 0.]),
        ("world:up", "↑", [0., -1.]),
        ("world:down", "↓", [0., 1.]),
    ] {
        let mut n = c.viewport;
        for (i, d) in delta.iter().enumerate() {
            n.origin[i] += d;
        }
        tools = tools.child(button(
            &mut view,
            key,
            text,
            n.bounds(c.limits.visible_cells)
                .ok()
                .map(|_| Action::Pan(delta)),
        ));
    }
    for (key, plane) in [
        ("world:xy", WorldGridPlane::Xy),
        ("world:xz", WorldGridPlane::Xz),
        ("world:yz", WorldGridPlane::Yz),
    ] {
        let mut n = c.viewport;
        n.plane = plane;
        tools = tools.child(button(
            &mut view,
            key,
            format!("{plane:?}"),
            Some(Action::Viewport(n)),
        ));
    }
    for (key, text, delta) in [
        ("world:slice-prev", "slice−", -1),
        ("world:slice-next", "slice+", 1),
    ] {
        let action = c.viewport.slice.checked_add(delta).map(|slice| {
            let mut n = c.viewport;
            n.slice = slice;
            Action::Viewport(n)
        });
        tools = tools.child(button(&mut view, key, text, action));
    }
    let mut grid = Element::box_()
        .key_str("world:grid")
        .style(StyleProp::Position, StyleValue::keyword(Keyword::Relative))
        .style(StyleProp::Width, StyleValue::px(c.viewport.size[0] as f32))
        .style(StyleProp::Height, StyleValue::px(c.viewport.size[1] as f32))
        .style(StyleProp::FlexShrink, StyleValue::number(0.));
    let b = c
        .viewport
        .bounds(c.limits.visible_cells)
        .expect("validated controller viewport");
    for v in b[1]..b[3] {
        for u in b[0]..b[2] {
            let cell = c.viewport.plane.coord(u as i32, v as i32, c.viewport.slice);
            let record = c.world().model().cell(cell);
            view.stats.cell_lookups += 1;
            view.stats.registered_cells += usize::from(record.is_some());
            let left = ((u as f64 - c.viewport.origin[0]) * c.viewport.cell_pixels).max(0.);
            let top = ((v as f64 - c.viewport.origin[1]) * c.viewport.cell_pixels).max(0.);
            let right = (((u + 1) as f64 - c.viewport.origin[0]) * c.viewport.cell_pixels)
                .min(c.viewport.size[0]);
            let bottom = (((v + 1) as f64 - c.viewport.origin[1]) * c.viewport.cell_pixels)
                .min(c.viewport.size[1]);
            let rect = [
                left as f32,
                top as f32,
                (right - left) as f32,
                (bottom - top) as f32,
            ];
            let key = format!("world:cell:{}:{}:{}", cell.x, cell.y, cell.z);
            let selected = c.selected == Some(cell);
            let dirty = record.is_some_and(|r| r.is_dirty());
            let residency = record.map(|r| r.residency());
            let lease = c.lease(cell);
            let color = if selected {
                (0.18, 0.36, 0.55)
            } else if dirty {
                (0.45, 0.25, 0.08)
            } else {
                match residency {
                    Some(CellResidency::Loaded) => (0.08, 0.29, 0.18),
                    Some(CellResidency::Loading(_) | CellResidency::Unloading(_)) => {
                        (0.30, 0.24, 0.40)
                    }
                    Some(CellResidency::Unloaded) => (0.12, 0.16, 0.21),
                    None => (0.05, 0.06, 0.08),
                }
            };
            let mut node = Element::box_()
                .key_str(&key)
                .class("world-cell")
                .style(StyleProp::Position, StyleValue::keyword(Keyword::Absolute))
                .style(StyleProp::Left, StyleValue::px(rect[0]))
                .style(StyleProp::Top, StyleValue::px(rect[1]))
                .style(StyleProp::Width, StyleValue::px(rect[2]))
                .style(StyleProp::Height, StyleValue::px(rect[3]))
                .style(
                    StyleProp::BackgroundColor,
                    StyleValue::rgba(color.0, color.1, color.2, 1.),
                )
                .style(StyleProp::BorderWidth, StyleValue::px(1.))
                .style(
                    StyleProp::BorderColor,
                    match lease {
                        WorldPanelLease::Owned { .. } => StyleValue::rgba(0.2, 0.8, 0.95, 1.),
                        WorldPanelLease::Expired => StyleValue::rgba(0.8, 0.3, 0.3, 1.),
                        WorldPanelLease::Unobserved => StyleValue::rgba(0.35, 0.40, 0.45, 1.),
                    },
                );
            if record.is_some() {
                view.bindings
                    .insert((key.clone(), UiTrigger::Activate), Action::Select(cell));
                view.bindings
                    .insert((key.clone(), UiTrigger::Select), Action::Select(cell));
                // Keep labels outside the clipped grid boxes: Loom text currently
                // has no clipping contract and must not escape viewport edges.
                node = node.class("world-cell-registered");
            }
            grid = grid.child(node);
            view.cells.push(WorldCellVisual {
                key,
                cell,
                rect,
                residency,
                dirty,
                revision: record.map(|r| r.revision()),
                lease,
                selected,
            });
        }
    }
    let mut details = column().key_str("world:details");
    if let Some(cell) = c.selected {
        if let Some(record) = c.world().model().cell(cell) {
            let owned = matches!(c.lease(cell), WorldPanelLease::Owned { .. });
            let loaded = record.residency() == CellResidency::Loaded;
            let queue = c.pending_intents() < c.limits.intents && !c.has_retired_world();
            details = details.child(Element::text(format!(
                "({}, {}, {}) {:?} dirty:{} revision:{} {} queued:{}",
                cell.x,
                cell.y,
                cell.z,
                record.residency(),
                record.is_dirty(),
                record.revision(),
                lease_text(c.lease(cell)),
                c.pending_intents()
            )));
            let mut controls = row();
            for (key, text, action) in [
                (
                    "world:load",
                    "Load",
                    (queue && record.residency() == CellResidency::Unloaded)
                        .then_some(Action::Load),
                ),
                (
                    "world:unload",
                    "Unload",
                    (queue && loaded && !record.is_dirty() && !owned).then_some(Action::Unload),
                ),
                (
                    "world:lease",
                    "Acquire / renew",
                    (!c.has_retired_world()
                        && !matches!(record.residency(), CellResidency::Unloading(_)))
                    .then_some(Action::AcquireLease),
                ),
                (
                    "world:release",
                    "Release",
                    (!matches!(c.lease(cell), WorldPanelLease::Unobserved))
                        .then_some(Action::ReleaseLease),
                ),
                (
                    "world:save",
                    "Save",
                    (queue && loaded && record.is_dirty() && owned).then_some(Action::Save),
                ),
                (
                    "world:source",
                    "Cross-cell source",
                    Some(Action::CrossCellSource),
                ),
                (
                    "world:cross-cell",
                    "Cross-cell intent",
                    (queue
                        && loaded
                        && owned
                        && c.source.is_some_and(|s| {
                            s != cell && matches!(c.lease(s), WorldPanelLease::Owned { .. })
                        }))
                    .then_some(Action::CrossCell),
                ),
            ] {
                controls = controls.child(button(&mut view, key, text, action));
            }
            details = details.child(controls).child(Element::text(format!(
                "Cross-cell source: {:?}; host supplies opaque membership edits",
                c.source
            )));
            let mut layers = c.project().layers().skip(c.project_layer_offset);
            for layer in layers.by_ref().take(c.limits.visible_layers) {
                let id = layer.id;
                view.stats.visible_layers += 1;
                let name: String = layer.name.chars().take(96).collect();
                let visible = button(
                    &mut view,
                    &format!("world:layer:{}:visible", id.get()),
                    format!("{name} visible:{}", layer.visible),
                    Some(Action::LayerVisible {
                        id,
                        visible: !layer.visible,
                    })
                    .filter(|_| !c.has_retired_world()),
                );
                let editable = button(
                    &mut view,
                    &format!("world:layer:{}:editable", id.get()),
                    format!("editable:{}", layer.editable),
                    Some(Action::LayerEditable {
                        id,
                        editable: !layer.editable,
                    })
                    .filter(|_| !c.has_retired_world()),
                );
                let member = c
                    .project()
                    .cell_layers(cell)
                    .is_some_and(|s| s.contains(&id));
                let membership = button(
                    &mut view,
                    &format!("world:layer:{}:member", id.get()),
                    format!("member:{member}"),
                    (!c.has_retired_world()).then_some(Action::CellLayer {
                        id,
                        member: !member,
                    }),
                );
                details = details.child(row().child(visible).child(editable).child(membership));
            }
            view.stats.has_more_layers = layers.next().is_some();
            let previous = c
                .project_layer_offset
                .saturating_sub(c.limits.visible_layers);
            let next = c.project_layer_offset + c.limits.visible_layers;
            let prev = button(
                &mut view,
                "world:layers:previous",
                "Previous layers",
                (c.project_layer_offset != 0).then_some(Action::ProjectLayerPage(previous)),
            );
            let next = button(
                &mut view,
                "world:layers:next",
                "Next layers",
                (next < c.project().layers().len()).then_some(Action::ProjectLayerPage(next)),
            );
            details = details.child(row().child(prev).child(next));
        }
    } else {
        details = details.child(Element::text("Select a registered cell in the viewport"));
    }
    let geometry = Element::text(format!("{:?} slice:{} origin:({:.3}, {:.3}) {:.2} px/cell · loaded green / dirty amber / pending purple · lease owned cyan / expired red / unobserved grey",
        c.viewport.plane, c.viewport.slice, c.viewport.origin[0], c.viewport.origin[1], c.viewport.cell_pixels));
    let (undo, redo, bytes) = c.project_history();
    let pending = c.project_io_pending();
    let candidate = c.project_candidate().is_some();
    let current = c.project_candidate_is_current();
    let retired = c.has_retired_world();
    let bound = !matches!(
        c.project_io_state(),
        crate::world::WorldProjectIoState::Unbound
    );
    let mut project_tools = column().key_str("world:project").child(Element::text(format!(
        "Project {} revision:{} unsaved:{} · undo:{undo} redo:{redo} history:{bytes} B · retirement:{retired}",
        c.project().identity(), c.project().revision(), c.project_is_dirty())));
    // IO errors are real worker output; bound presentation independently from
    // their OS-provided message length. Never fabricate a successful receipt.
    project_tools = project_tools.child(Element::text(
        format!("{:?}", c.project_io_state())
            .chars()
            .take(256)
            .collect::<String>(),
    ));
    let mut project_actions = row();
    for (key, label, action) in [
        (
            "world:project:undo",
            "Undo membership",
            (undo > 0 && !retired).then_some(Action::ProjectUndo),
        ),
        (
            "world:project:redo",
            "Redo membership",
            (redo > 0 && !retired).then_some(Action::ProjectRedo),
        ),
        (
            "world:project:save",
            "Save project",
            c.project_can_save().then_some(Action::ProjectSave),
        ),
        (
            "world:project:load",
            "Read candidate",
            (bound && !pending && !candidate && !retired).then_some(Action::ProjectLoad),
        ),
        (
            "world:project:cancel",
            "Cancel IO",
            pending.then_some(Action::ProjectCancel),
        ),
    ] {
        project_actions = project_actions.child(button(&mut view, key, label, action));
    }
    project_tools = project_tools.child(project_actions);
    if let Some(candidate) = c.project_candidate() {
        project_tools = project_tools.child(Element::text(format!(
            "Candidate revision:{} layers:{} cells:{} actors:{} current:{}",
            candidate.revision(),
            candidate.layers().len(),
            candidate.cells().len(),
            candidate.actors().len(),
            current
        )));
        let mut decisions = row();
        for (key, label, action) in [
            (
                "world:project:adopt",
                "Adopt clean",
                (current && !retired && !c.project_is_dirty()).then_some(Action::ProjectAdopt),
            ),
            (
                "world:project:adopt-discard",
                "Discard membership and adopt",
                (current && !retired).then_some(Action::ProjectAdoptDiscard),
            ),
            (
                "world:project:dismiss",
                "Dismiss candidate",
                Some(Action::ProjectDismiss),
            ),
        ] {
            decisions = decisions.child(button(&mut view, key, label, action));
        }
        project_tools = project_tools.child(decisions);
    }
    view.element = column()
        .key_str("world:panel")
        .child(tools)
        .child(geometry)
        .child(grid)
        .child(details)
        .child(project_tools);
    view
}
