//! Retained Loom controls for bounded real Prefab authored-override pages.
use crate::{
    prefab::{
        PrefabPanelAction as A, PrefabPanelController, PrefabPanelError, PrefabPanelSnapshot,
    },
    SelectionMode, UiEvent, UiTrigger,
};
use prism_editor_core::prefab::{PrefabOverride, PrefabOwner, PrefabProperty, PrefabValue};
use prism_ui::{
    style::{Keyword, StyleProp, StyleValue},
    Element,
};
use std::{
    collections::BTreeMap,
    fmt::Write,
    ops::Bound::{Excluded, Unbounded},
    sync::Arc,
};

/// One exact authored property; shared value payloads are never deep-cloned.
#[derive(Clone, Debug)]
pub struct PrefabOverrideVisual {
    /// Painted stable key, encoding the full semantic path without ambiguity.
    pub key: String,
    /// Stable owner of this explicit override.
    pub owner: PrefabOwner,
    /// Exact nested slot/entity/component/field address.
    pub property: PrefabProperty,
    /// Actual Set/Remove, preserving Null and all scalar representations.
    pub value: PrefabOverride,
    /// Bounded escaped display text. The address above remains lossless.
    pub label: String,
    /// Bounded value preview; arrays/objects/bytes display type and length.
    pub preview: String,
    /// Current bounded multi-selection state.
    pub selected: bool,
}
/// Work performed for the authored map page, not kernel transaction cost.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PrefabPanelViewStats {
    /// Entries visited by the map range iterator (at most rows + one).
    pub examined: usize,
    /// Materialized authored override rows.
    pub rows: usize,
    /// More authored properties precede the first visible row.
    pub has_previous: bool,
    /// More authored properties follow the last visible row.
    pub has_next: bool,
}
/// Actual Loom subtree and enabled event bindings, fenced to model and view.
#[derive(Clone, Debug)]
pub struct BoundPrefabPanel {
    element: Element,
    bindings: BTreeMap<(String, UiTrigger), A>,
    rows: Vec<PrefabOverrideVisual>,
    stats: PrefabPanelViewStats,
    snapshot: PrefabPanelSnapshot,
    pub(crate) view: Arc<()>,
}
impl BoundPrefabPanel {
    /// Mount or reconcile this subtree in a real Loom Ui.
    pub fn element(&self) -> &Element {
        &self.element
    }
    /// Exact visible authored overrides in semantic key order.
    pub fn rows(&self) -> &[PrefabOverrideVisual] {
        &self.rows
    }
    /// Bounded page traversal evidence and navigation availability.
    pub const fn stats(&self) -> PrefabPanelViewStats {
        self.stats
    }
    /// Exact model capability captured at build time.
    pub fn snapshot(&self) -> &PrefabPanelSnapshot {
        &self.snapshot
    }
    /// Enabled controls only, each backed by a real painted key.
    pub fn bindings(&self) -> impl Iterator<Item = (&str, UiTrigger)> {
        self.bindings.keys().map(|(key, t)| (key.as_str(), *t))
    }
    /// Pure event decoding. Use controller handle_event to enforce freshness.
    pub fn dispatch(&self, key: &str, event: &UiEvent) -> Result<A, PrefabPanelError> {
        let trigger = match event {
            UiEvent::Activate => UiTrigger::Activate,
            UiEvent::Select { .. } => UiTrigger::Select,
            UiEvent::Commit { .. } => return Err(PrefabPanelError::UnboundEvent),
        };
        let mut action = self
            .bindings
            .get(&(key.into(), trigger))
            .cloned()
            .ok_or(PrefabPanelError::UnboundEvent)?;
        if let (A::Select(_, mode), UiEvent::Select { mode: m }) = (&mut action, event) {
            *mode = *m;
        }
        Ok(action)
    }
}
fn column(width: f32) -> Element {
    Element::box_()
        .style(
            StyleProp::FlexDirection,
            StyleValue::keyword(Keyword::Column),
        )
        .style(StyleProp::Width, StyleValue::px(width))
        .style(StyleProp::FlexShrink, StyleValue::number(0.))
}
fn control(v: &mut BoundPrefabPanel, key: &str, label: &str, action: Option<A>) -> Element {
    let enabled = action.is_some();
    if let Some(action) = action {
        v.bindings.insert((key.into(), UiTrigger::Activate), action);
    }
    Element::box_()
        .key_str(key)
        .style(StyleProp::Height, StyleValue::px(28.))
        .style(StyleProp::FlexShrink, StyleValue::number(0.))
        .style(
            StyleProp::BackgroundColor,
            StyleValue::rgba(0.14, 0.18, 0.22, if enabled { 1. } else { 0.3 }),
        )
        .child(Element::text(label))
}
fn escaped(s: &str, budget: usize) -> String {
    let mut out = String::new();
    for c in s.chars().flat_map(char::escape_debug).take(budget) {
        out.push(c);
    }
    out
}
fn address_label(p: &PrefabProperty) -> String {
    let mut label = String::new();
    for slot in p.entity.slots.iter().take(3) {
        let _ = write!(label, "slot#{:x}/", slot.raw());
    }
    if p.entity.slots.len() > 3 {
        label.push_str(".../");
    }
    let _ = write!(
        label,
        "entity#{:x} [{}]",
        p.entity.entity.raw(),
        escaped(&p.component, 48)
    );
    for name in p.fields.iter().take(4) {
        let _ = write!(label, " [{}]", escaped(name, 32));
    }
    if p.fields.len() > 4 {
        label.push_str(" ...");
    }
    label
}
fn key(p: &PrefabProperty) -> String {
    // Length delimiters distinguish literal punctuation, empty names and nesting.
    let mut key = String::from("prefab:field:");
    for slot in &p.entity.slots {
        let _ = write!(key, "{:032x}/", slot.raw());
    }
    let _ = write!(key, "{:032x}:", p.entity.entity.raw());
    for name in std::iter::once(&p.component).chain(&p.fields) {
        let _ = write!(key, "{}:", name.len());
        for byte in name.bytes() {
            let _ = write!(key, "{byte:02x}");
        }
        key.push(':');
    }
    key
}
fn preview(value: &PrefabOverride) -> String {
    let PrefabOverride::Set(value) = value else {
        return "Remove (explicit)".into();
    };
    match value.as_ref() {
        PrefabValue::Null => "Set Null (explicit)".into(),
        PrefabValue::Bool(n) => format!("Set Bool {n}"),
        PrefabValue::Signed(n) => format!("Set i64 {n}"),
        PrefabValue::Unsigned(n) => format!("Set u64 {n}"),
        PrefabValue::FloatBits(n) => format!("Set f64 bits 0x{n:016x}"),
        PrefabValue::String(s) => format!("Set String {} bytes: {}", s.len(), escaped(s, 64)),
        PrefabValue::Bytes(v) => format!("Set Bytes [{}]", v.len()),
        PrefabValue::Array(v) => format!("Set Array [{}] (atomic)", v.len()),
        PrefabValue::Object(v) => format!("Set Object {{{}}}", v.len()),
    }
}
pub(crate) fn build(c: &PrefabPanelController) -> Result<BoundPrefabPanel, PrefabPanelError> {
    let overrides = &c.store().layer(c.owner)?.overrides;
    let mut v = BoundPrefabPanel {
        element: Element::box_(),
        bindings: BTreeMap::new(),
        rows: Vec::new(),
        stats: Default::default(),
        snapshot: c.snapshot(),
        view: c.view.clone(),
    };
    let backward = c.cursor.as_ref().is_some_and(|(_, back)| *back);
    let iterator: Box<dyn Iterator<Item = (&PrefabProperty, &PrefabOverride)> + '_> =
        match &c.cursor {
            Some((p, true)) => Box::new(overrides.range((Unbounded, Excluded(p))).rev()),
            Some((p, false)) => Box::new(overrides.range((Excluded(p), Unbounded))),
            None => Box::new(overrides.iter()),
        };
    for (p, value) in iterator.take(c.limits.rows + 1) {
        v.stats.examined += 1;
        if v.rows.len() == c.limits.rows {
            break;
        }
        c.limits.address(p)?;
        v.rows.push(PrefabOverrideVisual {
            key: key(p),
            owner: c.owner,
            property: p.clone(),
            value: value.clone(),
            label: address_label(p),
            preview: preview(value),
            selected: c.selected.contains(p),
        });
    }
    if backward {
        v.rows.reverse();
    }
    v.stats.rows = v.rows.len();
    v.stats.has_previous = v.rows.first().is_some_and(|row| {
        overrides
            .first_key_value()
            .is_some_and(|(p, _)| p < &row.property)
    });
    v.stats.has_next = v.rows.last().is_some_and(|row| {
        overrides
            .last_key_value()
            .is_some_and(|(p, _)| p > &row.property)
    });
    let mut tools = column(260.).key_str("prefab:tools");
    tools = tools.child(Element::text(format!("Prefab {:?}", c.owner)));
    let (undo, redo) = c.history_depths();
    let has_selection = !c.selected.is_empty();
    for (key, label, action) in [
        ("prefab:undo", "Undo", (undo > 0).then_some(A::Undo)),
        ("prefab:redo", "Redo", (redo > 0).then_some(A::Redo)),
        (
            "prefab:clear",
            "Clear selection",
            has_selection.then_some(A::Clear),
        ),
        (
            "prefab:revert",
            "Revert overrides",
            has_selection.then_some(A::Revert),
        ),
        (
            "prefab:apply-base",
            "Apply to base",
            (has_selection && c.base.is_some()).then_some(A::ApplyBase),
        ),
        (
            "prefab:propagate",
            "Propagate checked",
            (has_selection && matches!(c.owner, PrefabOwner::Asset(_)) && !c.recipients.is_empty())
                .then_some(A::Propagate),
        ),
        (
            "prefab:first",
            "First page",
            v.stats.has_previous.then_some(A::First),
        ),
        (
            "prefab:previous",
            "Previous page",
            v.rows
                .first()
                .filter(|_| v.stats.has_previous)
                .map(|r| A::Before(r.property.clone())),
        ),
        (
            "prefab:next",
            "Next page",
            v.rows
                .last()
                .filter(|_| v.stats.has_next)
                .map(|r| A::After(r.property.clone())),
        ),
    ] {
        tools = tools.child(control(&mut v, key, label, action));
    }
    tools = tools
        .child(Element::text(format!("Apply target: {:?}", c.base)))
        .child(Element::text(format!(
            "Selected fields: {}",
            c.selected.len()
        )))
        .child(Element::text("Propagation recipients (host scope)"));
    for (i, owner) in c.candidates.iter().enumerate() {
        let label = format!(
            "[{}] {owner:?}",
            if c.recipients.contains(owner) {
                "x"
            } else {
                " "
            }
        );
        tools = tools.child(control(
            &mut v,
            &format!("prefab:recipient:{i}"),
            &label,
            Some(A::ToggleRecipient(*owner)),
        ));
    }
    let mut rows = column(780.)
        .key_str("prefab:overrides")
        .child(Element::text(
            "Explicit field overrides (inherited values are not rows)",
        ));
    for row in &v.rows {
        let action = A::Select(row.property.clone(), SelectionMode::Replace);
        v.bindings
            .insert((row.key.clone(), UiTrigger::Activate), action.clone());
        v.bindings
            .insert((row.key.clone(), UiTrigger::Select), action);
        rows = rows.child(
            column(780.)
                .key_str(&row.key)
                .style(StyleProp::Height, StyleValue::px(48.))
                .style(
                    StyleProp::BackgroundColor,
                    if row.selected {
                        StyleValue::rgba(0.18, 0.34, 0.52, 1.)
                    } else {
                        StyleValue::rgba(0.08, 0.10, 0.13, 1.)
                    },
                )
                .child(Element::text(&row.label))
                .child(Element::text(&row.preview)),
        );
    }
    if v.rows.is_empty() {
        rows = rows.child(Element::text("No explicit field overrides on this page"));
    }
    rows = rows.child(Element::text(format!(
        "{} rows; previous={} next={}",
        v.stats.rows, v.stats.has_previous, v.stats.has_next
    )));
    v.element = Element::box_()
        .key_str("prefab:panel")
        .style(StyleProp::Width, StyleValue::px(1040.))
        .style(StyleProp::FlexDirection, StyleValue::keyword(Keyword::Row))
        .child(tools)
        .child(rows);
    Ok(v)
}
