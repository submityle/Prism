//! Concrete keyed Loom sequence-project view and exact semantic recipes.
use crate::{
    sequence::{
        SequenceClipField as Field, SequencePage as Page, SequencePanelAction as Action,
        SequencePanelController, SequencePanelError as Error, SequencePreview,
    },
    UiEvent, UiTrigger,
};
use prism_editor_timeline::{
    SequenceId, SequenceInstance, SequenceProjectEditor, SequenceProjectToken,
};
use prism_ui::{
    style::{Keyword, StyleProp, StyleValue},
    Element,
};
use prism_ui_component::Component;
use prism_ui_component_kit::inputs::{SearchField, SearchFieldProps};
use std::{collections::BTreeMap, sync::Arc};

#[derive(Clone, Debug)]
enum Recipe {
    Action(Action),
    Field(Field),
    Source,
    Frequency,
    PreviewStart,
    PreviewEnd,
    PreviewSeek,
    Elapsed,
}

/// Actual mountable tree, exact bindings and snapshot evidence.
#[derive(Clone, Debug)]
pub struct BoundSequencePanel {
    element: Element,
    bindings: BTreeMap<(String, UiTrigger), Recipe>,
    pub(crate) identity: Arc<()>,
    pub(crate) token: SequenceProjectToken,
    preview: SequencePreview,
}
impl BoundSequencePanel {
    /// Actual tree to mount or reconcile.
    pub fn element(&self) -> &Element {
        &self.element
    }
    /// Exact painted node keys and accepted semantic triggers.
    pub fn bindings(&self) -> impl Iterator<Item = (&str, UiTrigger)> {
        self.bindings.keys().map(|(k, t)| (k.as_str(), *t))
    }
    /// Exact project token represented by this view, for host save fences.
    pub fn snapshot_token(&self) -> SequenceProjectToken {
        self.token.clone()
    }
    /// Resolve input only. Use controller.handle_event to enforce freshness.
    pub fn dispatch(&self, target: &str, event: &UiEvent) -> Result<Action, Error> {
        let trigger = match event {
            UiEvent::Activate => UiTrigger::Activate,
            UiEvent::Select { .. } => UiTrigger::Select,
            UiEvent::Commit { .. } => UiTrigger::Commit,
        };
        let recipe = self
            .bindings
            .get(&(target.to_owned(), trigger))
            .ok_or(Error::UnboundEvent)?;
        match (recipe, event) {
            (Recipe::Action(action), UiEvent::Activate) => Ok(action.clone()),
            (recipe, UiEvent::Commit { value }) => {
                if matches!(recipe, Recipe::Source) {
                    let id = value
                        .as_unsigned()
                        .map(u128::from)
                        .or_else(|| value.as_text()?.trim().parse().ok())
                        .ok_or(Error::InvalidValue)?;
                    return Ok(Action::Source(SequenceId(id)));
                }
                let number = value
                    .as_unsigned()
                    .or_else(|| value.as_text()?.trim().parse().ok())
                    .ok_or(Error::InvalidValue)?;
                match recipe {
                    Recipe::Field(field) => Ok(Action::Field(*field, number)),
                    Recipe::Frequency => Ok(Action::Frequency(number)),
                    Recipe::PreviewStart => Ok(Action::Preview(SequencePreview {
                        start: number,
                        ..self.preview
                    })),
                    Recipe::PreviewEnd => Ok(Action::Preview(SequencePreview {
                        end: number,
                        ..self.preview
                    })),
                    Recipe::PreviewSeek => Ok(Action::Preview(SequencePreview {
                        seek: number,
                        ..self.preview
                    })),
                    Recipe::Elapsed => Ok(Action::Preview(SequencePreview {
                        elapsed_ns: number,
                        ..self.preview
                    })),
                    _ => Err(Error::UnboundEvent),
                }
            }
            _ => Err(Error::UnboundEvent),
        }
    }
    fn button(&mut self, key: &str, label: &str, action: Action, enabled: bool) -> Element {
        if enabled {
            self.bindings
                .insert((key.into(), UiTrigger::Activate), Recipe::Action(action));
        }
        Element::box_()
            .key_str(key)
            .style(StyleProp::BackgroundColor, StyleValue::token("color.bg"))
            .style(
                StyleProp::Opacity,
                StyleValue::number(if enabled { 1.0 } else { 0.45 }),
            )
            .child(Element::text(label))
    }
    fn field(&mut self, key: &str, label: &str, value: impl ToString, recipe: Recipe) -> Element {
        self.bindings
            .insert((key.into(), UiTrigger::Commit), recipe);
        column().child(Element::text(label)).child(
            SearchField
                .render(
                    &SearchFieldProps::new()
                        .value(value.to_string())
                        .placeholder(label),
                )
                .key_str(key),
        )
    }
    fn pager(
        &mut self,
        c: &SequencePanelController,
        e: &SequenceProjectEditor,
        page: Page,
        name: &str,
    ) -> Element {
        let index = c.pages[page.index()];
        let count = c.count(e, page);
        let last = count.saturating_sub(1) / c.limits.rows;
        row()
            .child(self.button(
                &format!("sequence:{name}:previous"),
                "Previous",
                Action::Page(page, index.saturating_sub(1)),
                index > 0,
            ))
            .child(Element::text(format!(
                "{name}: page {}/{} • {count} items • {} per page",
                index + 1,
                last + 1,
                c.limits.rows
            )))
            .child(self.button(
                &format!("sequence:{name}:next"),
                "Next",
                Action::Page(page, index + 1),
                index < last,
            ))
    }
}
fn column() -> Element {
    Element::box_()
        .style(StyleProp::Display, StyleValue::Keyword(Keyword::Flex))
        .style(
            StyleProp::FlexDirection,
            StyleValue::Keyword(Keyword::Column),
        )
        .style(StyleProp::Gap, StyleValue::px(4.0))
}
fn row() -> Element {
    Element::box_()
        .style(StyleProp::Display, StyleValue::Keyword(Keyword::Flex))
        .style(StyleProp::FlexDirection, StyleValue::Keyword(Keyword::Row))
        .style(StyleProp::Gap, StyleValue::px(8.0))
}
fn path_label(path: &[SequenceInstance]) -> String {
    path.iter()
        .map(|i| format!("clip {} @{}", i.clip.0, i.iteration))
        .collect::<Vec<_>>()
        .join(" / ")
}

pub(crate) fn build(
    c: &SequencePanelController,
    e: &SequenceProjectEditor,
) -> Result<BoundSequencePanel, Error> {
    let mut v = BoundSequencePanel {
        element: Element::box_(),
        bindings: BTreeMap::new(),
        identity: c.identity.clone(),
        token: e.snapshot_token(),
        preview: c.preview,
    };
    let mut tree = column()
        .key_str("sequence:panel")
        .child(Element::text("Sequence project"));
    tree = tree.child(
        row()
            .child(v.button(
                "sequence:undo",
                "Undo project command",
                Action::Undo,
                e.undo_len() > 0,
            ))
            .child(v.button(
                "sequence:redo",
                "Redo project command",
                Action::Redo,
                e.redo_len() > 0,
            ))
            .child(v.button(
                "sequence:open-root",
                "Open root",
                Action::OpenRoot,
                e.document().root().is_some(),
            ))
            .child(v.button(
                "sequence:add-document",
                "Add sequence",
                Action::AddDocument,
                true,
            )),
    );
    if let Some(error) = &c.error {
        tree = tree.child(Element::text(error.to_string()).key_str("sequence:error"));
    }
    tree = tree.child(Element::text(format!(
        "Root {:?} • opened {:?} • project revision {}",
        e.document().root().map(|id| id.0),
        c.selected.map(|id| id.0),
        e.revision()
    )));
    let mut documents =
        column()
            .key_str("sequence:documents")
            .child(v.pager(c, e, Page::Documents, "documents"));
    for (id, document) in e
        .document()
        .documents()
        .skip(c.pages[0] * c.limits.rows)
        .take(c.limits.rows)
    {
        documents = documents.child(
            row()
                .key_str(format!("sequence:document-row:{}", id.0))
                .child(v.button(
                    &format!("sequence:open:{}", id.0),
                    &format!(
                        "{}Sequence {} • {} Hz • {} events",
                        if Some(id) == c.selected {
                            "Selected: "
                        } else {
                            ""
                        },
                        id.0,
                        document.ticks_per_second,
                        document.timeline.event_count()
                    ),
                    Action::Open(id),
                    true,
                ))
                .child(v.button(
                    &format!("sequence:source:{}", id.0),
                    "Use as clip source",
                    Action::Source(id),
                    c.selected.is_some(),
                )),
        );
    }
    tree = tree.child(documents);
    if let Some(id) = c.selected {
        let document = e.document().document(id).ok_or(Error::InvalidValue)?;
        tree = tree
            .child(
                row()
                    .child(v.button(
                        "sequence:set-root",
                        "Make root",
                        Action::SetRoot,
                        e.document().root() != Some(id),
                    ))
                    .child(v.button(
                        "sequence:remove-document",
                        "Remove sequence (unlink first)",
                        Action::RemoveDocument,
                        true,
                    )),
            )
            .child(v.field(
                "sequence:frequency",
                "Opened sequence clock (ticks/second)",
                document.ticks_per_second,
                Recipe::Frequency,
            ));
        let mut parents =
            column()
                .key_str("sequence:parents")
                .child(v.pager(c, e, Page::Parents, "parents"));
        for (parent, clip) in e
            .document()
            .clips()
            .filter(|(_, clip)| clip.source == id)
            .skip(c.pages[2] * c.limits.rows)
            .take(c.limits.rows)
        {
            parents = parents.child(v.button(
                &format!("sequence:parent:{}", clip.id.0),
                &format!("Locate parent {} / clip {}", parent.0, clip.id.0),
                Action::Locate(clip.id),
                true,
            ));
        }
        tree = tree.child(parents);
        let mut clips =
            column()
                .key_str("sequence:clips")
                .child(v.pager(c, e, Page::Clips, "clips"));
        for (_, clip) in e
            .document()
            .clips()
            .filter(|(p, _)| *p == id)
            .skip(c.pages[1] * c.limits.rows)
            .take(c.limits.rows)
        {
            let (n, d) = clip.rate.ratio();
            clips=clips.child(row().key_str(format!("sequence:clip-row:{}",clip.id.0))
                .child(v.button(&format!("sequence:clip:{}",clip.id.0),&format!("{}Clip {} → {} • parent [{},{}) source [{},{}) • rate {n}/{d} • repeat {}",if c.clip==Some(clip.id){"Selected: "}else{""},clip.id.0,clip.source.0,clip.parent.start(),clip.parent.end(),clip.source_range.start(),clip.source_range.end(),clip.repetitions),Action::Locate(clip.id),true))
                .child(v.button(&format!("sequence:child:{}",clip.id.0),"Open child",Action::OpenChild(clip.id),true)));
        }
        tree = tree.child(clips).child(v.button(
            "sequence:new-clip",
            "New clip draft",
            Action::NewClip,
            true,
        ));
        let draft = c.draft;
        tree = tree
            .child(Element::text(format!(
                "Inspector: {}",
                c.clip
                    .map_or_else(|| "new clip".into(), |id| format!("clip {}", id.0))
            )))
            .child(v.field(
                "sequence:clip-source",
                "Source sequence ID",
                draft.source.map_or_else(String::new, |id| id.0.to_string()),
                Recipe::Source,
            ));
        for (key, label, value, field) in [
            (
                "parent-start",
                "Parent start (inclusive)",
                draft.parent_start,
                Field::ParentStart,
            ),
            (
                "parent-end",
                "Parent end (exclusive)",
                draft.parent_end,
                Field::ParentEnd,
            ),
            (
                "source-start",
                "Source start (inclusive)",
                draft.source_start,
                Field::SourceStart,
            ),
            (
                "source-end",
                "Source end (exclusive)",
                draft.source_end,
                Field::SourceEnd,
            ),
            (
                "numerator",
                "Rate numerator",
                draft.numerator,
                Field::Numerator,
            ),
            (
                "denominator",
                "Rate denominator",
                draft.denominator,
                Field::Denominator,
            ),
            (
                "repetitions",
                "Source repeat count",
                draft.repetitions,
                Field::Repetitions,
            ),
        ] {
            tree = tree.child(v.field(
                &format!("sequence:{key}"),
                label,
                value,
                Recipe::Field(field),
            ));
        }
        tree = tree.child(
            row()
                .child(v.button(
                    "sequence:add-clip",
                    "Add clip from draft",
                    Action::AddClip,
                    draft.source.is_some(),
                ))
                .child(v.button(
                    "sequence:save-clip",
                    "Apply selected clip",
                    Action::SaveClip,
                    c.clip.is_some(),
                ))
                .child(v.button(
                    "sequence:remove-clip",
                    "Remove selected clip",
                    Action::RemoveClip,
                    c.clip.is_some(),
                )),
        );
    }
    tree = tree.child(Element::text(
        "Evaluate actual root interval (seek emits no crossed events; play advances elapsed time)",
    ));
    for (key, label, value, recipe) in [
        (
            "preview-start",
            "Root start (inclusive)",
            c.preview.start,
            Recipe::PreviewStart,
        ),
        (
            "preview-end",
            "Root end (exclusive)",
            c.preview.end,
            Recipe::PreviewEnd,
        ),
        (
            "preview-seek",
            "Seek before playing",
            c.preview.seek,
            Recipe::PreviewSeek,
        ),
        (
            "elapsed-ns",
            "Elapsed nanoseconds",
            c.preview.elapsed_ns,
            Recipe::Elapsed,
        ),
    ] {
        tree = tree.child(v.field(&format!("sequence:{key}"), label, value, recipe));
    }
    tree = tree.child(
        row()
            .child(v.button(
                "sequence:loop",
                if c.preview.looping {
                    "Root mode: Loop"
                } else {
                    "Root mode: Once"
                },
                Action::Preview(SequencePreview {
                    looping: !c.preview.looping,
                    ..c.preview
                }),
                true,
            ))
            .child(v.button(
                "sequence:evaluate",
                "Evaluate interval",
                Action::Evaluate,
                e.document().root().is_some(),
            )),
    );
    tree=tree.child(Element::text(format!("Actual destination tick {:?} • evaluation caps: {} events, {} samples, {} visits (also limited by project policy)",c.evaluated_tick,c.limits.diagnostics,c.limits.diagnostics,c.limits.visits)));
    let mut events =
        column()
            .key_str("sequence:events")
            .child(v.pager(c, e, Page::Events, "events"));
    for (index, event) in c
        .events
        .iter()
        .enumerate()
        .skip(c.pages[3] * c.limits.rows)
        .take(c.limits.rows)
    {
        let key = format!("sequence:event:{index}");
        let mut node = column()
            .key_str(&key)
            .child(Element::text(format!(
                "Sequence {} event {} '{}' • local {} → root {} @{} • {}",
                event.sequence.0,
                event.event,
                event.name,
                event.tick,
                event.root_tick,
                event.root_iteration,
                path_label(&event.path)
            )))
            .child(v.button(
                &format!("{key}:open"),
                "Open event sequence",
                Action::Open(event.sequence),
                true,
            ));
        if let Some(instance) = event.path.last() {
            node = node.child(v.button(
                &format!("{key}:locate"),
                "Locate nested clip",
                Action::Locate(instance.clip),
                true,
            ));
        }
        events = events.child(node);
    }
    let mut samples =
        column()
            .key_str("sequence:samples")
            .child(v.pager(c, e, Page::Samples, "samples"));
    for (index, sample) in c
        .samples
        .iter()
        .enumerate()
        .skip(c.pages[4] * c.limits.rows)
        .take(c.limits.rows)
    {
        let key = format!("sequence:sample:{index}");
        let mut node = column()
            .key_str(&key)
            .child(Element::text(format!(
                "Sequence {} • sampled local tick {} • {} tracks • {}",
                sample.sequence.0,
                sample.tick,
                sample.tracks,
                path_label(&sample.path)
            )))
            .child(v.button(
                &format!("{key}:open"),
                "Open sample sequence",
                Action::Open(sample.sequence),
                true,
            ));
        if let Some(instance) = sample.path.last() {
            node = node.child(v.button(
                &format!("{key}:locate"),
                "Locate nested clip",
                Action::Locate(instance.clip),
                true,
            ));
        }
        samples = samples.child(node);
    }
    v.element = tree.child(events).child(samples);
    Ok(v)
}
