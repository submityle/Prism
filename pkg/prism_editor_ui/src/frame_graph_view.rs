//! Retained Loom queue lanes, dependency wires, SSA tables and real budget gauges.

use crate::{
    frame_graph::{
        FrameGraphPage as Page, FrameGraphPanelAction as Action, FrameGraphPanelController,
        FrameGraphPanelError, FrameGraphSelection as Selection,
    },
    UiEvent, UiTrigger,
};
use prism_editor_profiler::{CapturedDependencyKind, CapturedResourceLayout, GraphCaptureReport};
use prism_ui::{
    style::{Keyword, StyleProp, StyleValue},
    Element,
};
use std::{collections::BTreeMap, sync::Arc};

/// One real pass box in queue lanes; geometry is in canvas pixels.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameGraphPassVisual {
    /// Original pass table index.
    pub pass: u32,
    /// Actual activation target.
    pub key: String,
    /// Left, top, width, height on the fixed-width canvas.
    pub rect: [f32; 4],
    /// Membership of a completely measured critical path.
    pub critical: bool,
    /// Current explicit selection.
    pub selected: bool,
}
/// One real dependency page entry; wires only paint when both passes are visible.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameGraphDependencyVisual {
    /// Canonical dependency index in the report.
    pub dependency: usize,
    /// Real pass table endpoints.
    pub endpoints: [u32; 2],
    /// Selectable wire segments (left, top, width, height), possibly empty.
    pub segments: Vec<[f32; 4]>,
}
/// Real access identity and SSA, whether viewed from a pass or resource.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrameGraphAccessVisual {
    /// Original pass index.
    pub pass: u32,
    /// Original access ordinal.
    pub ordinal: usize,
    /// Original resource index.
    pub resource: u32,
    /// Stable captured access vocabulary.
    pub kind: u8,
    /// Input SSA version.
    pub input_version: u32,
    /// Output SSA version for a write.
    pub output_version: Option<u32>,
}
/// Gauge values copied directly from one real ledger/alias budget snapshot.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameGraphBudgetVisual {
    /// Canonical report budget index.
    pub budget: usize,
    /// Key of the actually painted gauge fill.
    pub key: String,
    /// Recorded consumption.
    pub consumed: u64,
    /// Allowed consumption in the same category unit.
    pub limit: u64,
    /// Strict consumed > limit comparison (equality is allowed).
    pub exceeded: bool,
    /// Clipped gauge fraction; zero budget has no division.
    pub fraction: f32,
}
/// Counts in Passes/Resources/Dependencies/Accesses/Budgets order.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FrameGraphPanelStats {
    /// Source counts after the selected relationship filter.
    pub total: [usize; 5],
    /// Materialized rows; hidden entries are reachable through page controls.
    pub visible: [usize; 5],
    /// Current independent page offsets.
    pub offsets: [usize; 5],
    /// Dependency page entries whose endpoints require another pass page.
    pub off_page_dependencies: usize,
}
/// An actual keyed Loom subtree plus exact publication/navigation fences.
#[derive(Clone, Debug)]
pub struct BoundFrameGraphPanel {
    element: Element,
    bindings: BTreeMap<String, Action>,
    passes: Vec<FrameGraphPassVisual>,
    dependencies: Vec<FrameGraphDependencyVisual>,
    accesses: Vec<FrameGraphAccessVisual>,
    budgets: Vec<FrameGraphBudgetVisual>,
    stats: FrameGraphPanelStats,
    pub(crate) report: Option<Arc<GraphCaptureReport>>,
    pub(crate) stamp: Arc<()>,
}
impl BoundFrameGraphPanel {
    /// Mount/reconcile this retained subtree into a native dock body.
    pub fn element(&self) -> &Element {
        &self.element
    }
    /// Associate these painted keys with actual retained layout hit targets.
    pub fn bindings(&self) -> impl Iterator<Item = (&str, UiTrigger)> {
        self.bindings
            .keys()
            .map(|k| (k.as_str(), UiTrigger::Activate))
    }
    /// Exact report Arc displayed by this tree.
    pub fn report(&self) -> Option<&Arc<GraphCaptureReport>> {
        self.report.as_ref()
    }
    /// Queue-lane boxes in this pass page.
    pub fn passes(&self) -> &[FrameGraphPassVisual] {
        &self.passes
    }
    /// Actual dependency page and any painted wires.
    pub fn dependencies(&self) -> &[FrameGraphDependencyVisual] {
        &self.dependencies
    }
    /// Visible real access/SSA relationships.
    pub fn accesses(&self) -> &[FrameGraphAccessVisual] {
        &self.accesses
    }
    /// Visible real budget gauges.
    pub fn budgets(&self) -> &[FrameGraphBudgetVisual] {
        &self.budgets
    }
    /// Independent virtualization counts.
    pub const fn stats(&self) -> FrameGraphPanelStats {
        self.stats
    }
    /// Pure binding lookup; use controller handle_event for both identity fences.
    pub fn dispatch(&self, target: &str, event: &UiEvent) -> Result<Action, FrameGraphPanelError> {
        if *event != UiEvent::Activate {
            return Err(FrameGraphPanelError::UnboundEvent);
        }
        self.bindings
            .get(target)
            .copied()
            .ok_or(FrameGraphPanelError::UnboundEvent)
    }
}

pub(crate) fn build(c: &FrameGraphPanelController) -> BoundFrameGraphPanel {
    let mut v = BoundFrameGraphPanel {
        element: Element::box_(),
        bindings: BTreeMap::new(),
        passes: Vec::new(),
        dependencies: Vec::new(),
        accesses: Vec::new(),
        budgets: Vec::new(),
        stats: FrameGraphPanelStats {
            offsets: c.offsets,
            ..Default::default()
        },
        report: c.report.clone(),
        stamp: c.stamp.clone(),
    };
    let mut root = column()
        .key_str("frame-graph:panel")
        .style(StyleProp::Width, StyleValue::px(800.));
    let Some(r) = c.report.as_deref() else {
        v.element = root
            .child(Element::text("No frame graph capture published").key_str("frame-graph:empty"));
        return v;
    };
    for page in pages() {
        v.stats.total[page.index()] = c.page_info(page).0;
    }
    root = root.child(
        Element::text(format!(
            "Frame {} · {:?} · {:?} clock · logical {} B · alias heap {} B (not resident VRAM)",
            r.frame(),
            r.source(),
            r.timing_source(),
            r.logical_bytes().expect("validated report"),
            r.alias_heap_bytes()
        ))
        .key_str("frame-graph:summary"),
    );
    root = root.child(
        Element::text(match &c.critical {
            Some(path) => format!(
                "Measured {:?} critical path: {} ns · {} passes (excludes queue stalls)",
                r.timing_source(),
                path.total_duration_ns(),
                path.node_ids().len()
            ),
            None => "Critical path unavailable: incomplete measurements or empty graph".into(),
        })
        .key_str("frame-graph:critical"),
    );
    let tools = row()
        .child(button(
            &mut v,
            "frame-graph:clear",
            "Clear selection",
            Some(Action::ClearSelection),
            false,
        ))
        .child(button(
            &mut v,
            "frame-graph:related",
            if c.related_only {
                "Related edges"
            } else {
                "All edges"
            },
            Some(Action::RelatedOnly(!c.related_only)),
            c.related_only,
        ));
    root = root
        .child(tools)
        .child(pager(c, &mut v, Page::Passes, "passes"));
    let count = r
        .order()
        .iter()
        .skip(c.offsets[0])
        .take(c.limits.passes)
        .len();
    let step = 752. / count.max(1) as f32;
    for (column, &pass) in r
        .order()
        .iter()
        .skip(c.offsets[0])
        .take(c.limits.passes)
        .enumerate()
    {
        let p = &r.passes()[pass as usize];
        v.passes.push(FrameGraphPassVisual {
            pass,
            key: format!("frame-graph:pass:{pass}"),
            rect: [
                48. + column as f32 * step,
                24. + f32::from(p.queue) * 60.,
                (step - 6.).clamp(1., 130.),
                32.,
            ],
            critical: c.critical_passes.contains(&pass),
            selected: c.selection == Some(Selection::Pass(pass)),
        });
    }
    let mut canvas = Element::box_()
        .key_str("frame-graph:canvas")
        .style(StyleProp::Position, StyleValue::keyword(Keyword::Relative))
        .style(StyleProp::Width, StyleValue::px(800.))
        .style(StyleProp::Height, StyleValue::px(204.))
        .style(StyleProp::FlexShrink, StyleValue::number(0.))
        .style(StyleProp::BackgroundColor, rgba([0.04, 0.05, 0.07]));
    for (lane, label) in ["Gfx", "Comp", "Copy"].iter().enumerate() {
        canvas =
            canvas.child(at([2., 24. + lane as f32 * 60., 44., 26.]).child(Element::text(*label)));
    }
    for (id, edge) in r
        .dependencies()
        .iter()
        .enumerate()
        .filter(|(_, e)| c.dependency_matches(e))
        .skip(c.offsets[2])
        .take(c.limits.dependencies)
    {
        let a = v.passes.iter().find(|p| p.pass == edge.before);
        let b = v.passes.iter().find(|p| p.pass == edge.after);
        let segments = match (a, b) {
            (Some(a), Some(b)) => {
                let x0 = a.rect[0] + a.rect[2] * 0.5;
                let x1 = b.rect[0] + b.rect[2] * 0.5;
                let y0 = a.rect[1] + a.rect[3];
                let y1 = b.rect[1] + b.rect[3];
                let rail = 190. + (id % 4) as f32 * 3.;
                vec![
                    [x0, y0, 2., rail - y0],
                    [x0.min(x1), rail, (x1 - x0).abs().max(2.), 2.],
                    [x1, y1, 2., rail - y1],
                ]
            }
            _ => {
                v.stats.off_page_dependencies += 1;
                Vec::new()
            }
        };
        for (segment, rect) in segments.iter().enumerate() {
            let key = format!("frame-graph:wire:{id}:{segment}");
            v.bindings.insert(key.clone(), Action::Dependency(id));
            canvas = canvas.child(at(*rect).key_str(key).style(
                StyleProp::BackgroundColor,
                if c.selection == Some(Selection::Dependency(id)) {
                    rgba([1., 0.7, 0.12])
                } else {
                    rgba([0.35, 0.65, 0.9])
                },
            ));
        }
        v.dependencies.push(FrameGraphDependencyVisual {
            dependency: id,
            endpoints: [edge.before, edge.after],
            segments,
        });
    }
    for p in v.passes.clone() {
        v.bindings.insert(p.key.clone(), Action::Pass(p.pass));
        let data = &r.passes()[p.pass as usize];
        let mut box_ = at(p.rect)
            .key_str(p.key)
            .class(if p.critical {
                "is-measured-critical"
            } else {
                "is-topology"
            })
            .style(
                StyleProp::BackgroundColor,
                if p.selected {
                    rgba([1., 0.7, 0.12])
                } else if p.critical {
                    rgba([0.65, 0.3, 0.65])
                } else {
                    rgba([0.15, 0.45, 0.7])
                },
            );
        if p.rect[2] >= 26. {
            let capacity = (p.rect[2] / 8.) as usize;
            let label = format!(
                "P{} {}",
                p.pass,
                short(&data.name, capacity.saturating_sub(7))
            );
            box_ = box_.child(Element::text(label));
        }
        canvas = canvas.child(box_);
    }
    v.stats.visible[0] = v.passes.len();
    root = root.child(canvas).child(Element::text(
        "Horizontal position = compiler order, not time. Lanes = logical queue classes.",
    ));
    root = root.child(pager(c, &mut v, Page::Resources, "resources"));
    for (id, resource) in r
        .resources()
        .iter()
        .enumerate()
        .skip(c.offsets[1])
        .take(c.limits.resources)
    {
        root = root.child(button(
            &mut v,
            &format!("frame-graph:resource:{id}"),
            &format!(
                "R{id} {} · {} B · lifetime {}",
                short(&resource.name, 60),
                resource.size,
                lifetime(resource.lifetime)
            ),
            Some(Action::Resource(id as u32)),
            c.selection == Some(Selection::Resource(id as u32)),
        ));
        v.stats.visible[1] += 1;
    }
    root = root.child(pager(c, &mut v, Page::Dependencies, "dependencies"));
    for visual in v.dependencies.clone() {
        let id = visual.dependency;
        let edge = &r.dependencies()[id];
        let label = match edge.kind {
            CapturedDependencyKind::Explicit => "explicit prerequisite".into(),
            CapturedDependencyKind::Hazard { resource, before_access, after_access, queue_transfer } => format!("hazard R{resource} A{before_access} → A{after_access} · queue transfer {queue_transfer}"),
        };
        root = root.child(button(
            &mut v,
            &format!("frame-graph:dependency:{id}"),
            &format!(
                "D{id} P{} → P{} · {label}{}",
                edge.before,
                edge.after,
                if visual.segments.is_empty() {
                    " · endpoints off page"
                } else {
                    ""
                }
            ),
            Some(Action::Dependency(id)),
            c.selection == Some(Selection::Dependency(id)),
        ));
    }
    v.stats.visible[2] = v.dependencies.len();
    root = root.child(details(c, &mut v, r));
    root = root.child(pager(c, &mut v, Page::Accesses, "accesses"));
    let accesses = r
        .passes()
        .iter()
        .enumerate()
        .flat_map(|(p, pass)| {
            pass.accesses
                .iter()
                .enumerate()
                .map(move |(a, access)| (p as u32, a, access))
        })
        .filter(|(p, _, a)| c.access_matches(*p, a.resource))
        .skip(c.offsets[3])
        .take(c.limits.accesses);
    for (pass, ordinal, a) in accesses {
        let base = format!("frame-graph:access:{pass}:{ordinal}");
        root = root.child(
            row()
                .key_str(&base)
                .child(button(
                    &mut v,
                    &format!("{base}:pass"),
                    &format!("P{pass} A{ordinal}"),
                    Some(Action::Pass(pass)),
                    false,
                ))
                .child(button(
                    &mut v,
                    &format!("{base}:resource"),
                    &format!("R{}", a.resource),
                    Some(Action::Resource(a.resource)),
                    false,
                ))
                .child(Element::text(format!(
                    "{} · input v{} → {}",
                    access_name(a.kind),
                    a.input_version,
                    a.output_version
                        .map(|v| format!("output v{v}"))
                        .unwrap_or_else(|| "read only".into())
                ))),
        );
        v.accesses.push(FrameGraphAccessVisual {
            pass,
            ordinal,
            resource: a.resource,
            kind: a.kind,
            input_version: a.input_version,
            output_version: a.output_version,
        });
    }
    v.stats.visible[3] = v.accesses.len();
    root = root.child(pager(c, &mut v, Page::Budgets, "budgets"));
    for (id, b) in r
        .budgets()
        .iter()
        .enumerate()
        .skip(c.offsets[4])
        .take(c.limits.budgets)
    {
        let fraction = if b.limit == 0 {
            if b.consumed == 0 {
                0.
            } else {
                1.
            }
        } else {
            (b.consumed as f64 / b.limit as f64).min(1.) as f32
        };
        let key = format!("frame-graph:budget:{id}:fill");
        root = root.child(
            column()
                .key_str(format!("frame-graph:budget:{id}"))
                .child(Element::text(format!(
                    "{} · {} / {}{}",
                    short(&b.category, 80),
                    b.consumed,
                    b.limit,
                    if b.exceeded() {
                        " · OVER BUDGET"
                    } else {
                        " · within budget"
                    }
                )))
                .child(
                    Element::box_()
                        .key_str(format!("frame-graph:budget:{id}:gauge"))
                        .style(StyleProp::Width, StyleValue::px(320.))
                        .style(StyleProp::Height, StyleValue::px(10.))
                        .style(StyleProp::BackgroundColor, rgba([0.08, 0.1, 0.14]))
                        .child(
                            Element::box_()
                                .key_str(&key)
                                .style(StyleProp::Width, StyleValue::percent(fraction * 100.))
                                .style(StyleProp::Height, StyleValue::px(10.))
                                .style(
                                    StyleProp::BackgroundColor,
                                    if b.exceeded() {
                                        rgba([0.95, 0.15, 0.12])
                                    } else {
                                        rgba([0.15, 0.75, 0.4])
                                    },
                                ),
                        ),
                ),
        );
        v.budgets.push(FrameGraphBudgetVisual {
            budget: id,
            key,
            consumed: b.consumed,
            limit: b.limit,
            exceeded: b.exceeded(),
            fraction,
        });
    }
    v.stats.visible[4] = v.budgets.len();
    v.element = root;
    v
}

fn details(
    c: &FrameGraphPanelController,
    v: &mut BoundFrameGraphPanel,
    r: &GraphCaptureReport,
) -> Element {
    let mut out = column().key_str("frame-graph:details");
    match c.selection {
        Some(Selection::Pass(id)) => {
            let p = &r.passes()[id as usize];
            out = out.child(Element::text(format!(
                "P{id} {} · key {} · feature {} · variant {} · queue {} · {}",
                short(&p.name, 100),
                p.key,
                p.feature,
                p.variant,
                queue(p.queue),
                p.duration_ns
                    .map(|n| format!("measured {n} ns {:?}", r.timing_source()))
                    .unwrap_or_else(|| "unmeasured".into())
            )));
        }
        Some(Selection::Resource(id)) => {
            let resource = &r.resources()[id as usize];
            let layout = match &resource.layout {
                CapturedResourceLayout::Buffer { bytes, usage } => format!("buffer {bytes} B · usage 0x{usage:x}"),
                CapturedResourceLayout::Texture { extent, dimension, format, mip_levels, samples, usage } => format!("texture {extent:?} D{dimension} {} · mips {mip_levels} · samples {samples} · usage 0x{usage:x}", short(format, 80)),
            };
            out = out.child(Element::text(format!(
                "R{id} {} · owner {:?} slot {} frame {} · {} · alignment {} · {layout}",
                short(&resource.name, 100),
                resource.owner,
                resource.slot,
                resource.frame,
                lifetime(resource.lifetime),
                resource.alignment
            )));
        }
        Some(Selection::Dependency(id)) => {
            let e = &r.dependencies()[id];
            out = out
                .child(Element::text(format!(
                    "D{id} · P{} → P{}",
                    e.before, e.after
                )))
                .child(
                    row()
                        .child(button(
                            v,
                            "frame-graph:edge:before",
                            "Inspect prerequisite pass",
                            Some(Action::Pass(e.before)),
                            false,
                        ))
                        .child(button(
                            v,
                            "frame-graph:edge:after",
                            "Inspect dependent pass",
                            Some(Action::Pass(e.after)),
                            false,
                        )),
                );
            if let CapturedDependencyKind::Hazard {
                resource,
                before_access,
                after_access,
                queue_transfer,
            } = e.kind
            {
                let a = &r.passes()[e.before as usize].accesses[before_access as usize];
                let b = &r.passes()[e.after as usize].accesses[after_access as usize];
                out = out.child(Element::text(format!("R{resource} hazard · P{} A{before_access} input v{} output {:?} → P{} A{after_access} input v{} output {:?} · queue transfer {queue_transfer}", e.before, a.input_version, a.output_version, e.after, b.input_version, b.output_version)))
                    .child(button(v, "frame-graph:edge:resource", "Inspect hazard resource", Some(Action::Resource(resource)), false));
            } else {
                out = out.child(Element::text(
                    "Explicit prerequisite; no resource hazard attached",
                ));
            }
        }
        None => {
            out = out.child(Element::text(
                "Select a pass, resource or dependency to inspect the captured relationships",
            ));
        }
    }
    out
}
fn pages() -> [Page; 5] {
    [
        Page::Passes,
        Page::Resources,
        Page::Dependencies,
        Page::Accesses,
        Page::Budgets,
    ]
}
fn pager(
    c: &FrameGraphPanelController,
    v: &mut BoundFrameGraphPanel,
    page: Page,
    name: &str,
) -> Element {
    let (total, size) = c.page_info(page);
    let offset = c.offsets[page.index()];
    row()
        .child(Element::text(format!(
            "{name}: {offset}–{} / {total}",
            (offset + size).min(total)
        )))
        .child(button(
            v,
            &format!("frame-graph:{name}:previous"),
            "Previous",
            (offset > 0).then_some(Action::Previous(page)),
            false,
        ))
        .child(button(
            v,
            &format!("frame-graph:{name}:next"),
            "Next",
            (offset + size < total).then_some(Action::Next(page)),
            false,
        ))
}
fn button(
    v: &mut BoundFrameGraphPanel,
    key: &str,
    label: &str,
    action: Option<Action>,
    selected: bool,
) -> Element {
    if let Some(a) = action {
        v.bindings.insert(key.into(), a);
    }
    Element::box_()
        .key_str(key)
        .style(StyleProp::PaddingLeft, StyleValue::px(3.))
        .style(StyleProp::PaddingRight, StyleValue::px(3.))
        .style(
            StyleProp::Opacity,
            StyleValue::number(if action.is_some() { 1. } else { 0.4 }),
        )
        .style(
            StyleProp::BackgroundColor,
            if selected {
                rgba([0.45, 0.3, 0.1])
            } else {
                rgba([0.1, 0.13, 0.17])
            },
        )
        .child(Element::text(label))
}
fn column() -> Element {
    Element::box_()
        .style(StyleProp::Display, StyleValue::keyword(Keyword::Flex))
        .style(
            StyleProp::FlexDirection,
            StyleValue::keyword(Keyword::Column),
        )
        .style(StyleProp::Gap, StyleValue::px(2.))
        .style(StyleProp::FlexShrink, StyleValue::number(0.))
}
fn row() -> Element {
    Element::box_()
        .style(StyleProp::Display, StyleValue::keyword(Keyword::Flex))
        .style(StyleProp::FlexDirection, StyleValue::keyword(Keyword::Row))
        .style(StyleProp::Gap, StyleValue::px(3.))
        .style(StyleProp::FlexShrink, StyleValue::number(0.))
}
fn at(r: [f32; 4]) -> Element {
    Element::box_()
        .style(StyleProp::Position, StyleValue::keyword(Keyword::Absolute))
        .style(StyleProp::Left, StyleValue::px(r[0]))
        .style(StyleProp::Top, StyleValue::px(r[1]))
        .style(StyleProp::Width, StyleValue::px(r[2]))
        .style(StyleProp::Height, StyleValue::px(r[3]))
}
fn rgba(c: [f32; 3]) -> StyleValue {
    StyleValue::rgba(c[0], c[1], c[2], 1.)
}
fn short(s: &str, count: usize) -> String {
    let mut text: String = s.chars().take(count).collect();
    if s.chars().count() > count {
        text.push('…');
    }
    text
}
fn queue(q: u8) -> &'static str {
    match q {
        0 => "graphics",
        1 => "compute",
        _ => "transfer",
    }
}
fn lifetime(l: u8) -> &'static str {
    match l {
        0 => "imported",
        1 => "persistent",
        _ => "transient",
    }
}
fn access_name(k: u8) -> &'static str {
    match k {
        0 => "sampled read",
        1 => "storage read",
        2 => "storage write",
        3 => "color attachment",
        4 => "depth attachment",
        5 => "indirect read",
        6 => "transfer read",
        _ => "transfer write",
    }
}
