//! Real project commands, nested clocks/events, mounted semantic targets and reconciliation.
use prism_editor_timeline::*;
use prism_editor_ui::{
    ActionValue, BoundSequencePanel, SequencePage, SequencePanelAction as A,
    SequencePanelController, SequencePanelError as E, SequencePanelLimits, SequencePreview,
    UiEvent, UiTrigger,
};
use prism_ui::{
    backend::{BackendOp, RecordingBackend},
    layout::{AvailableSpace, Size},
    Element, Key, Ui,
};
use std::collections::BTreeSet;

const ROOT: SequenceId = SequenceId(100);
const CHILD: SequenceId = SequenceId(200);
const LEAF: SequenceId = SequenceId(300);
fn document(hz: u64, events: &[u64]) -> TimelineDocument {
    let mut timeline = Timeline::default();
    let mut track = Track::new(TrackId::from_raw(1), "provider-neutral value", 1).unwrap();
    for (id, tick) in [(1, 0), (2, 100)] {
        track
            .insert(Keyframe::new(
                KeyId::from_raw(id),
                tick,
                vec![tick as f64],
                Interpolation::Linear,
            ))
            .unwrap();
    }
    timeline.add_track(track).unwrap();
    for (id, tick) in events.iter().enumerate() {
        timeline
            .add_event(TimelineEvent {
                id: id as u128 + 1,
                tick: *tick,
                name: format!("provider cue {tick}"),
            })
            .unwrap();
    }
    TimelineDocument {
        ticks_per_second: hz,
        timeline,
    }
}
fn clip(
    id: u128,
    source: SequenceId,
    parent: (u64, u64),
    local: (u64, u64),
    n: u64,
    d: u64,
) -> SequenceClip {
    SequenceClip {
        id: SequenceClipId(id),
        source,
        parent: SequenceRange::new(parent.0, parent.1).unwrap(),
        source_range: SequenceRange::new(local.0, local.1).unwrap(),
        rate: PlaybackRate::new(n, d).unwrap(),
        repetitions: 2,
    }
}
fn command(edits: Vec<SequenceProjectEdit>) -> SequenceProjectCommand {
    SequenceProjectCommand {
        label: "fixture transaction".into(),
        edits,
    }
}
fn fixture() -> SequenceProjectEditor {
    let mut e = SequenceProjectEditor::new(
        SequenceProjectDocument::default(),
        SequenceDocumentLimits::default(),
    )
    .unwrap();
    e.execute(command(vec![
        SequenceProjectEdit::PutDocument {
            id: ROOT,
            document: document(2, &[0, 2]),
        },
        SequenceProjectEdit::PutDocument {
            id: CHILD,
            document: document(4, &[1, 3]),
        },
        SequenceProjectEdit::PutDocument {
            id: LEAF,
            document: document(8, &[11, 13, 17]),
        },
        SequenceProjectEdit::PutClip {
            parent: ROOT,
            clip: clip(20, CHILD, (1, 9), (0, 8), 1, 1),
        },
        SequenceProjectEdit::PutClip {
            parent: CHILD,
            clip: clip(30, LEAF, (0, 8), (10, 18), 1, 2),
        },
        SequenceProjectEdit::SetRoot(Some(ROOT)),
    ]))
    .unwrap();
    e
}
fn walk(element: &Element, keys: &mut Vec<String>, count: &mut usize, depth: usize) -> usize {
    *count += 1;
    if let Some(Key::Str(key)) = element.explicit_key() {
        keys.push(key.clone());
    }
    element.child_elements().iter().fold(depth, |max, child| {
        max.max(walk(child, keys, count, depth + 1))
    })
}
fn verify_tree(view: &BoundSequencePanel) -> (usize, usize) {
    let mut keys = Vec::new();
    let mut count = 0;
    let depth = walk(view.element(), &mut keys, &mut count, 0);
    assert_eq!(
        keys.len(),
        keys.iter().collect::<BTreeSet<_>>().len(),
        "duplicate explicit keys"
    );
    for (key, _) in view.bindings() {
        assert!(
            keys.iter().any(|k| k == key),
            "binding has no painted node: {key}"
        );
        assert!(key.len() < 16 * 1024);
    }
    assert!(count <= 16384);
    assert!(depth <= 128);
    assert!(view.bindings().count() <= 4096);
    (count, depth)
}
fn text(value: impl ToString) -> UiEvent {
    UiEvent::Commit {
        value: ActionValue::text(value.to_string()).unwrap(),
    }
}
struct Host {
    e: SequenceProjectEditor,
    c: SequencePanelController,
    ui: Ui<RecordingBackend>,
}
impl Host {
    fn new(e: SequenceProjectEditor, limits: SequencePanelLimits) -> Self {
        let mut c = SequencePanelController::new(limits).unwrap();
        let view = c.build(&e).unwrap();
        let (nodes, _) = verify_tree(&view);
        let mut ui = Ui::new(RecordingBackend::new());
        ui.mount(view.element());
        assert_eq!(
            ui.backend()
                .ops()
                .iter()
                .filter(|op| matches!(op, BackendOp::Create { .. }))
                .count(),
            nodes
        );
        Self { e, c, ui }
    }
    fn send(&mut self, key: &str, event: UiEvent) -> Result<(), E> {
        let view = self.c.build(&self.e)?;
        verify_tree(&view);
        self.ui.update(view.element());
        self.ui.compute_layout(Size::new(
            AvailableSpace::Definite(1800.0),
            AvailableSpace::MaxContent,
        ));
        let trigger = match event {
            UiEvent::Activate => UiTrigger::Activate,
            UiEvent::Commit { .. } => UiTrigger::Commit,
            UiEvent::Select { .. } => UiTrigger::Select,
        };
        assert!(
            view.bindings().any(|(k, t)| k == key && t == trigger),
            "not an actual mounted semantic target: {key}"
        );
        self.ui.backend_mut().clear();
        let result = self.c.handle_event(&mut self.e, &view, key, &event);
        let after = self.c.build(&self.e)?;
        verify_tree(&after);
        self.ui.update(after.element());
        result
    }
    fn activate(&mut self, key: &str) {
        self.send(key, UiEvent::Activate).unwrap();
    }
    fn commit(&mut self, key: &str, value: impl ToString) {
        self.send(key, text(value)).unwrap();
    }
}

#[test]
fn mounted_project_authoring_edits_real_topology_and_inverse_history() {
    let mut h = Host::new(fixture(), SequencePanelLimits::default());
    h.activate("sequence:clip:20");
    let before = h.e.document().clone();
    let history = h.e.undo_len();
    let token = h.e.snapshot_token();
    for (key, value) in [
        ("parent-start", 2),
        ("parent-end", 10),
        ("source-start", 1),
        ("source-end", 7),
        ("numerator", 3),
        ("denominator", 2),
        ("repetitions", 4),
    ] {
        h.commit(&format!("sequence:{key}"), value);
    }
    assert_eq!(h.e.snapshot_token(), token, "draft must not mutate project");
    h.activate("sequence:save-clip");
    assert_eq!(h.e.undo_len(), history + 1);
    let c =
        h.e.document()
            .clips()
            .find(|(_, c)| c.id == SequenceClipId(20))
            .unwrap()
            .1;
    assert_eq!(
        (
            c.parent.start(),
            c.parent.end(),
            c.source_range.start(),
            c.source_range.end(),
            c.rate.ratio(),
            c.repetitions
        ),
        (2, 10, 1, 7, (3, 2), 4)
    );
    assert!(h
        .ui
        .backend()
        .ops()
        .iter()
        .any(|op| matches!(op,BackendOp::SetText{text,..} if text.contains("rate 3/2"))));
    h.activate("sequence:undo");
    assert_eq!(h.e.document(), &before);
    h.activate("sequence:redo");
    assert_eq!(h.c.draft().numerator, 3);
    h.activate("sequence:child:20");
    assert_eq!(h.c.selected(), Some(CHILD));
    h.activate("sequence:parent:20");
    assert_eq!(h.c.selected(), Some(ROOT));
    assert_eq!(h.c.selected_clip(), Some(SequenceClipId(20)));
    h.activate("sequence:open:300");
    h.activate("sequence:set-root");
    assert_eq!(h.e.document().root(), Some(LEAF));
    h.activate("sequence:undo");
    assert_eq!(h.e.document().root(), Some(ROOT));
    h.activate("sequence:open-root");
    assert_eq!(h.c.selected(), Some(ROOT));
}

#[test]
fn empty_project_creation_add_remove_root_and_clips_are_real_commands() {
    let editor = SequenceProjectEditor::new(
        SequenceProjectDocument::default(),
        SequenceDocumentLimits::default(),
    )
    .unwrap();
    let mut h = Host::new(editor, SequencePanelLimits::default());
    assert_eq!(h.c.selected(), None);
    h.activate("sequence:add-document");
    let root = h.c.selected().unwrap();
    assert_eq!(h.e.document().root(), Some(root));
    h.activate("sequence:add-document");
    let child = h.c.selected().unwrap();
    assert_ne!(root, child);
    h.activate("sequence:open-root");
    h.activate(&format!("sequence:source:{}", child.0));
    h.activate("sequence:add-clip");
    let id = h.c.selected_clip().unwrap();
    assert_eq!(h.e.document().clips().next().unwrap().1.source, child);
    h.activate(&format!("sequence:child:{}", id.0));
    let token = h.e.snapshot_token();
    assert!(h
        .send("sequence:remove-document", UiEvent::Activate)
        .is_err());
    assert_eq!(h.e.snapshot_token(), token);
    h.activate(&format!("sequence:parent:{}", id.0));
    h.activate("sequence:remove-clip");
    assert_eq!(h.e.document().clips().count(), 0);
    h.activate("sequence:undo");
    assert_eq!(h.e.document().clips().count(), 1);
    h.activate("sequence:redo");
    h.activate(&format!("sequence:open:{}", child.0));
    h.activate("sequence:remove-document");
    assert!(h.e.document().document(child).is_none());
    h.activate("sequence:open-root");
    h.activate("sequence:remove-document");
    assert_eq!(h.e.document().root(), None);
    h.activate("sequence:undo");
    assert_eq!(h.e.document().root(), Some(root));
}

#[test]
fn real_nested_clocks_and_crossed_events_are_diagnosed_and_locatable() {
    let mut h = Host::new(
        fixture(),
        SequencePanelLimits {
            rows: 2,
            ..SequencePanelLimits::default()
        },
    );
    h.commit("sequence:elapsed-ns", 2_000_000_000u64);
    h.activate("sequence:evaluate");
    assert!(h
        .c
        .samples()
        .iter()
        .any(|s| s.sequence == LEAF && s.tick == 16 && s.tracks == 1));
    let leaf =
        h.c.events()
            .iter()
            .find(|e| e.sequence == LEAF && e.tick == 11)
            .unwrap();
    assert_eq!(leaf.root_tick, 2);
    assert_eq!(
        leaf.path.iter().map(|p| p.clip.0).collect::<Vec<_>>(),
        [20, 30]
    );
    assert!(h
        .c
        .events()
        .iter()
        .any(|e| e.sequence == ROOT && e.tick == 0));
    let total = h.c.events().len();
    assert!(total > 2);
    h.activate("sequence:events:next");
    h.activate("sequence:event:2:locate");
    assert!(h.c.selected_clip().is_some());
    h.commit("sequence:preview-seek", 4);
    h.commit("sequence:elapsed-ns", 0);
    h.activate("sequence:evaluate");
    assert!(
        h.c.events().is_empty(),
        "seek must not fabricate crossed events"
    );
    assert!(h
        .c
        .samples()
        .iter()
        .any(|s| s.sequence == LEAF && s.tick == 16));
    h.commit("sequence:preview-seek", 0);
    h.commit("sequence:preview-end", 8);
    h.activate("sequence:loop");
    assert!(h.c.preview().looping);
    h.commit("sequence:elapsed-ns", 6_000_000_000u64);
    h.activate("sequence:evaluate");
    assert!(h.c.events().iter().any(|e| e.root_iteration > 0));
    assert!(h
        .c
        .events()
        .iter()
        .any(|e| e.path.iter().any(|p| p.iteration > 0)));
    h.activate("sequence:open-root");
    h.commit("sequence:frequency", 3);
    assert!(h.c.events().is_empty());
    assert!(h.c.samples().is_empty());
}

#[test]
fn cycle_bad_ranges_and_zero_rates_reject_without_partial_edits_and_render_error() {
    let mut h = Host::new(fixture(), SequencePanelLimits::default());
    h.activate("sequence:open:300");
    h.activate("sequence:source:100");
    let before = h.e.document().clone();
    let token = h.e.snapshot_token();
    assert_eq!(
        h.send("sequence:add-clip", UiEvent::Activate),
        Err(E::Document(SequenceDocumentError::Sequence(
            SequenceError::Cycle
        )))
    );
    assert_eq!(h.e.document(), &before);
    assert_eq!(h.e.snapshot_token(), token);
    assert!(h
        .ui
        .backend()
        .ops()
        .iter()
        .any(|op| matches!(op,BackendOp::SetText{text,..} if text.contains("Cycle"))));
    h.activate("sequence:open-root");
    h.activate("sequence:clip:20");
    h.commit("sequence:parent-end", 0);
    assert_eq!(
        h.send("sequence:save-clip", UiEvent::Activate),
        Err(E::Sequence(SequenceError::InvalidRange))
    );
    h.commit("sequence:parent-end", 9);
    h.commit("sequence:denominator", 0);
    assert!(h.send("sequence:save-clip", UiEvent::Activate).is_err());
    assert_eq!(h.e.snapshot_token(), token);
}

#[test]
fn exact_view_fences_reject_other_controller_undo_aba_and_external_project_replacement() {
    let mut e = fixture();
    let mut c = SequencePanelController::default();
    let view = c.build(&e).unwrap();
    let mut other = SequencePanelController::default();
    other.build(&e).unwrap();
    assert_eq!(
        other.handle_event(&mut e, &view, "sequence:undo", &UiEvent::Activate),
        Err(E::StaleView)
    );
    c.apply(&mut e, A::Open(CHILD)).unwrap();
    assert_eq!(
        c.handle_event(&mut e, &view, "sequence:undo", &UiEvent::Activate),
        Err(E::StaleView)
    );
    let view = c.build(&e).unwrap();
    let revision = e.revision();
    let mut replacement = SequenceProjectEditor::new(e.document().clone(), e.limits()).unwrap();
    replacement
        .execute(command(vec![SequenceProjectEdit::SetRoot(Some(ROOT))]))
        .unwrap();
    assert_eq!(revision, replacement.revision());
    assert_eq!(
        c.handle_event(
            &mut replacement,
            &view,
            "sequence:add-document",
            &UiEvent::Activate
        ),
        Err(E::StaleView)
    );
    c.apply(&mut e, A::Frequency(9)).unwrap();
    c.apply(&mut e, A::Undo).unwrap();
    assert_eq!(
        c.handle_event(&mut e, &view, "sequence:undo", &UiEvent::Activate),
        Err(E::StaleView)
    );
    let current = c.build(&e).unwrap();
    assert_eq!(
        c.handle_event(&mut e, &current, "sequence:not-painted", &UiEvent::Activate),
        Err(E::UnboundEvent)
    );
    assert_eq!(
        current.dispatch("sequence:undo", &text("1")),
        Err(E::UnboundEvent)
    );
}

#[test]
fn real_evaluation_budget_failure_clears_old_results_and_stays_authorable() {
    let mut h = Host::new(
        fixture(),
        SequencePanelLimits {
            diagnostics: 1,
            ..SequencePanelLimits::default()
        },
    );
    assert!(h.send("sequence:evaluate", UiEvent::Activate).is_err());
    assert!(h.c.events().is_empty());
    assert!(h.c.samples().is_empty());
    assert!(h.c.error().is_some());
    h.activate("sequence:add-document");
    assert!(h.c.error().is_none());
    assert!(SequencePanelController::new(SequencePanelLimits {
        rows: 33,
        ..SequencePanelLimits::default()
    })
    .is_err());
    assert_eq!(
        h.c.apply(
            &mut h.e,
            A::Preview(SequencePreview {
                seek: u64::MAX,
                ..SequencePreview::default()
            })
        ),
        Err(E::InvalidValue)
    );
}

#[test]
fn deterministic_bounded_pages_have_stable_large_ids_and_zero_reconcile_on_identical_view() {
    let mut e = fixture();
    let edits = (0..100u128)
        .map(|i| SequenceProjectEdit::PutDocument {
            id: SequenceId(u128::MAX - 100 + i),
            document: document(1, &[]),
        })
        .collect();
    e.execute(command(edits)).unwrap();
    let mut h = Host::new(
        e,
        SequencePanelLimits {
            rows: 32,
            ..SequencePanelLimits::default()
        },
    );
    let view = h.c.build(&h.e).unwrap();
    let (nodes, _) = verify_tree(&view);
    assert!(nodes < 2048);
    assert_eq!(
        view.bindings()
            .filter(|(k, _)| k.starts_with("sequence:open:"))
            .count(),
        32
    );
    h.ui.backend_mut().clear();
    h.ui.update(h.c.build(&h.e).unwrap().element());
    assert!(h.ui.backend().ops().is_empty());
    h.activate("sequence:documents:next");
    let a = h.c.build(&h.e).unwrap().element().clone();
    h.activate("sequence:documents:previous");
    h.activate("sequence:documents:next");
    assert_eq!(&a, h.c.build(&h.e).unwrap().element());
    let id = u128::MAX - 1;
    h.commit("sequence:clip-source", id);
    assert_eq!(h.c.draft().source, Some(SequenceId(id)));
    let view = h.c.build(&h.e).unwrap();
    assert_eq!(view.snapshot_token(), h.e.snapshot_token());
    assert_eq!(
        view.dispatch("sequence:clip-source", &text(u128::MAX)),
        Ok(A::Source(SequenceId(u128::MAX)))
    );
    assert_eq!(
        view.dispatch("sequence:frequency", &text("18446744073709551616")),
        Err(E::InvalidValue)
    );
}

#[test]
fn parent_and_clip_pages_locate_global_ids_without_page_identity_aliases() {
    let mut e = fixture();
    e.execute(command(
        (100..141)
            .map(|id| SequenceProjectEdit::PutClip {
                parent: ROOT,
                clip: clip(id, CHILD, (0, 4), (0, 4), 1, 1),
            })
            .collect(),
    ))
    .unwrap();
    let mut h = Host::new(
        e,
        SequencePanelLimits {
            rows: 2,
            ..SequencePanelLimits::default()
        },
    );
    h.activate("sequence:clips:next");
    h.activate("sequence:clip:101");
    assert_eq!(h.c.selected_clip(), Some(SequenceClipId(101)));
    h.activate("sequence:child:101");
    assert_eq!(h.c.selected(), Some(CHILD));
    h.activate("sequence:parents:next");
    h.activate("sequence:parent:102");
    assert_eq!(h.c.selected(), Some(ROOT));
    assert_eq!(h.c.selected_clip(), Some(SequenceClipId(102)));
    let view = h.c.build(&h.e).unwrap();
    assert!(view.bindings().any(|(key, _)| key == "sequence:clip:102"));
    let token = h.e.snapshot_token();
    assert_eq!(
        h.c.apply(&mut h.e, A::Page(SequencePage::Clips, usize::MAX)),
        Err(E::InvalidValue)
    );
    assert_eq!(h.e.snapshot_token(), token);
}

#[test]
fn zero_track_instances_cannot_bypass_the_owned_diagnostic_output_budget() {
    let mut e = SequenceProjectEditor::new(
        SequenceProjectDocument::default(),
        SequenceDocumentLimits::default(),
    )
    .unwrap();
    let empty = || TimelineDocument {
        ticks_per_second: 1,
        timeline: Timeline::default(),
    };
    let mut edits = vec![
        SequenceProjectEdit::PutDocument {
            id: ROOT,
            document: empty(),
        },
        SequenceProjectEdit::PutDocument {
            id: CHILD,
            document: empty(),
        },
        SequenceProjectEdit::SetRoot(Some(ROOT)),
    ];
    edits.extend((0..130).map(|id| SequenceProjectEdit::PutClip {
        parent: ROOT,
        clip: clip(id, CHILD, (0, 4), (0, 4), 1, 1),
    }));
    e.execute(command(edits)).unwrap();
    let mut h = Host::new(e, SequencePanelLimits::default());
    assert_eq!(
        h.send("sequence:evaluate", UiEvent::Activate),
        Err(E::Sequence(SequenceError::Limit("diagnostic sample rows")))
    );
    assert!(h.c.samples().is_empty());
    assert!(h.c.events().is_empty());
}

#[test]
fn external_unlink_and_removal_prune_staged_references_and_reconcile_real_controls() {
    let mut h = Host::new(fixture(), SequencePanelLimits::default());
    h.activate("sequence:clip:20");
    h.activate("sequence:evaluate");
    let old = h.c.build(&h.e).unwrap();
    h.e.execute(command(vec![SequenceProjectEdit::RemoveClip(
        SequenceClipId(20),
    )]))
    .unwrap();
    assert_eq!(
        h.c.handle_event(&mut h.e, &old, "sequence:save-clip", &UiEvent::Activate),
        Err(E::StaleView)
    );
    let view = h.c.build(&h.e).unwrap();
    h.ui.update(view.element());
    assert_eq!(h.c.selected_clip(), None);
    assert_eq!(h.c.draft(), prism_editor_ui::SequenceClipDraft::default());
    assert!(h.c.events().is_empty());
    assert!(h.c.samples().is_empty());
    assert!(!view.bindings().any(|(key, _)| key == "sequence:save-clip"));
    h.activate("sequence:source:200");
    h.e.execute(command(vec![
        SequenceProjectEdit::RemoveClip(SequenceClipId(30)),
        SequenceProjectEdit::RemoveDocument(CHILD),
    ]))
    .unwrap();
    let view = h.c.build(&h.e).unwrap();
    h.ui.update(view.element());
    assert_eq!(h.c.draft().source, None);
    assert!(!view.bindings().any(|(key, _)| key == "sequence:add-clip"));
    h.activate("sequence:open:300");
    h.e.execute(command(vec![SequenceProjectEdit::RemoveDocument(LEAF)]))
        .unwrap();
    let view = h.c.build(&h.e).unwrap();
    h.ui.update(view.element());
    assert_eq!(h.c.selected(), Some(ROOT));
    verify_tree(&view);
}

#[test]
fn mounted_controls_have_real_nonzero_layout_and_retained_paint() {
    use prism_ui_render_backend::{rasterize, RetainedScene};
    let e = fixture();
    let mut c = SequencePanelController::default();
    let view = c.build(&e).unwrap();
    fn names(element: &Element, output: &mut Vec<Option<String>>) {
        output.push(match element.explicit_key() {
            Some(Key::Str(key)) => Some(key.clone()),
            _ => None,
        });
        for child in element.child_elements() {
            names(child, output);
        }
    }
    let mut keyed = Vec::new();
    names(view.element(), &mut keyed);
    let mut ui = Ui::new(RetainedScene::new());
    ui.mount(view.element());
    ui.compute_layout(Size::new(
        AvailableSpace::Definite(1800.0),
        AvailableSpace::MaxContent,
    ));
    let mut rects = Vec::new();
    ui.backend().visit_layout(|_, rect| rects.push(rect));
    assert_eq!(keyed.len(), rects.len());
    let rects: std::collections::BTreeMap<_, _> = keyed
        .into_iter()
        .zip(rects)
        .filter_map(|(key, rect)| key.map(|key| (key, rect)))
        .collect();
    for (key, _) in view.bindings() {
        let rect = rects[key];
        assert!(
            rect.right() > rect.left() && rect.bottom() > rect.top(),
            "zero area {key}: {rect:?}"
        );
    }
    let rect = rects["sequence:undo"];
    let image = rasterize(&ui.backend().to_draw_list(), 1800, 160);
    let pixel = image.pixel((rect.left() + 1.0) as u32, (rect.top() + 1.0) as u32);
    assert!(
        pixel[3] > 0.99,
        "keyed button has no actual painted background: {pixel:?}"
    );
}
