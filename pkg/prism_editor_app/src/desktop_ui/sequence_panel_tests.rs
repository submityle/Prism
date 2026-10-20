//! Actual native geometry/input, nested evaluation, and complete project files.
use super::*;
use prism_editor_timeline::*;
use prism_editor_ui::{SequencePanelError, SequencePanelLimits};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

const ROOT: SequenceId = SequenceId(100);
const CHILD: SequenceId = SequenceId(200);
const LEAF: SequenceId = SequenceId(300);
fn document(hz: u64, events: &[u64]) -> TimelineDocument {
    let mut timeline = Timeline::default();
    let mut track = Track::new(TrackId::from_raw(1), "host scalar", 1).unwrap();
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
                name: format!("host cue {tick}"),
            })
            .unwrap();
    }
    TimelineDocument {
        ticks_per_second: hz,
        timeline,
    }
}
fn panel() -> DesktopSequencePanel {
    let mut editor =
        SequenceProjectEditor::new(SequenceProjectDocument::default(), Default::default()).unwrap();
    let mut edits = vec![
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
        SequenceProjectEdit::SetRoot(Some(ROOT)),
    ];
    for (id, parent, source, start, end, local_start, local_end, denominator) in [
        (20, ROOT, CHILD, 1, 9, 0, 8, 1),
        (30, CHILD, LEAF, 0, 8, 10, 18, 2),
    ] {
        edits.push(SequenceProjectEdit::PutClip {
            parent,
            clip: SequenceClip {
                id: SequenceClipId(id),
                source,
                parent: SequenceRange::new(start, end).unwrap(),
                source_range: SequenceRange::new(local_start, local_end).unwrap(),
                rate: PlaybackRate::new(1, denominator).unwrap(),
                repetitions: 2,
            },
        });
    }
    editor
        .execute(SequenceProjectCommand {
            label: "host fixture".into(),
            edits,
        })
        .unwrap();
    DesktopSequencePanel::new(editor, SequencePanelLimits::default()).unwrap()
}
fn ui(panel: DesktopSequencePanel) -> EditorDesktopUi {
    let mut ui = EditorDesktopUi::new(EditorController::new()).unwrap();
    ui.resize(1600., 1000.).unwrap();
    ui.mount_native_panel("sequence", "Sequence", Box::new(panel))
        .unwrap();
    ui
}
fn click(ui: &mut EditorDesktopUi, key: &str) {
    // Exercise native wheel scrolling and hit the actual clipped retained node.
    let x = ui.size.width / 2.;
    let y = ui.size.height - STATUS_HEIGHT - ui.panel_body_height() / 2.;
    ui.scroll_at(x, y, -10_000).unwrap();
    for _ in 0..256 {
        if let Some(rect) = ui
            .hits
            .iter()
            .find(|h| h.target == key && h.rect.size.width > 0. && h.rect.size.height > 0.)
            .map(|h| h.rect)
        {
            ui.click(
                rect.left() + rect.size.width / 2.,
                rect.top() + rect.size.height / 2.,
                SelectionMode::Replace,
            )
            .unwrap();
            return;
        }
        assert!(ui.scroll_at(x, y, 4).unwrap(), "No retained hit for {key}");
    }
    panic!("No visible native control: {key}");
}
fn commit(ui: &mut EditorDesktopUi, key: &str, value: &str) {
    click(ui, key);
    assert!(
        ui.text_input_rect().is_some(),
        "native field geometry: {key}"
    );
    ui.text(value).unwrap();
    ui.key(DesktopKey::Enter).unwrap();
}
fn get(ui: &EditorDesktopUi) -> &DesktopSequencePanel {
    ui.native_panel::<DesktopSequencePanel>("sequence").unwrap()
}
fn set_frequency(p: &mut DesktopSequencePanel, hz: u64) {
    p.editor
        .execute(SequenceProjectCommand {
            label: "host frequency".into(),
            edits: vec![SequenceProjectEdit::SetFrequency {
                id: ROOT,
                ticks_per_second: hz,
            }],
        })
        .unwrap();
}

#[test]
fn sequence_native_fields_apply_one_command_undo_redo_and_real_nested_evaluate() {
    let mut ui = ui(panel());
    let before = get(&ui).editor.snapshot_token();
    let history = get(&ui).editor.undo_len();
    click(&mut ui, "sequence:clip:20");
    commit(&mut ui, "sequence:parent-end", "10");
    assert_eq!(get(&ui).editor.snapshot_token(), before);
    assert!(get(&ui).has_clip_draft());
    assert!(ui.has_uncommitted_edit());
    click(&mut ui, "sequence:save-clip");
    assert_eq!(get(&ui).editor.undo_len(), history + 1);
    assert!(!get(&ui).has_clip_draft());
    assert_eq!(
        get(&ui)
            .editor
            .document()
            .clips()
            .find(|(_, c)| c.id.0 == 20)
            .unwrap()
            .1
            .parent
            .end(),
        10
    );
    ui.key(DesktopKey::Undo).unwrap();
    assert_eq!(
        get(&ui)
            .editor
            .document()
            .clips()
            .find(|(_, c)| c.id.0 == 20)
            .unwrap()
            .1
            .parent
            .end(),
        9
    );
    ui.key(DesktopKey::Redo).unwrap();
    assert_eq!(
        get(&ui)
            .editor
            .document()
            .clips()
            .find(|(_, c)| c.id.0 == 20)
            .unwrap()
            .1
            .parent
            .end(),
        10
    );
    let token = get(&ui).editor.snapshot_token();
    commit(&mut ui, "sequence:elapsed-ns", "2000000000");
    click(&mut ui, "sequence:evaluate");
    let p = get(&ui);
    assert_eq!(p.editor.snapshot_token(), token, "evaluation is read only");
    assert!(p
        .controller
        .samples()
        .iter()
        .any(|s| s.sequence == LEAF && s.tick == 16 && s.tracks == 1));
    let event = p
        .controller
        .events()
        .iter()
        .find(|e| e.sequence == LEAF && e.tick == 11)
        .unwrap();
    assert_eq!(event.root_tick, 2);
    assert_eq!(
        event.path.iter().map(|p| p.clip.0).collect::<Vec<_>>(),
        [20, 30]
    );
    assert!(
        p.next_wake().is_none(),
        "explicit evaluation has no transport wakeup"
    );
}

#[test]
fn sequence_native_clip_draft_and_rejected_input_fence_save_close_and_navigation() {
    let dir = Directory::new();
    let mut p = panel();
    p.mount_storage(
        dir.file(),
        Default::default(),
        TimelineFileRevision::Missing,
        Duration::from_secs(5),
    )
    .unwrap();
    let mut ui = ui(p);
    let token = get(&ui).editor.snapshot_token();
    click(&mut ui, "sequence:clip:20");
    commit(&mut ui, "sequence:numerator", "3");
    assert!(ui.panel_document_command(true).is_err());
    assert!(!dir.file().exists());
    assert!(ui.unmount_native_panel("sequence").is_err());
    assert!(ui.key(DesktopKey::Undo).is_err());
    assert_eq!(get(&ui).editor.snapshot_token(), token);
    click(&mut ui, "sequence:host:discard-draft");
    assert_eq!(get(&ui).controller.draft().numerator, 1);
    assert!(!ui.has_uncommitted_edit());
    // Actual focus/text/Enter carries the native draft stamp to handle_event.
    click(&mut ui, "sequence:numerator");
    ui.text("bad integer").unwrap();
    assert!(ui.key(DesktopKey::Enter).is_err());
    assert_eq!(
        get(&ui).controller.error(),
        Some(&SequencePanelError::InvalidValue)
    );
    assert!(ui.has_uncommitted_edit());
    assert!(ui.panel_document_command(true).is_err());
    ui.key(DesktopKey::Escape).unwrap();
    click(&mut ui, "sequence:host:discard-draft");
    assert!(get(&ui).controller.error().is_none());
    assert_eq!(get(&ui).editor.snapshot_token(), token);
    assert!(!ui.has_uncommitted_edit());
}

#[test]
fn sequence_stale_bound_project_or_controller_cannot_mutate_or_clear_draft() {
    let mut p = panel();
    p.build().unwrap();
    let original = p.editor.snapshot_token();
    p.controller
        .apply(&mut p.editor, prism_editor_ui::SequencePanelAction::NewClip)
        .unwrap();
    assert!(p
        .event("sequence:add-document", &UiEvent::Activate)
        .unwrap_err()
        .contains("StaleView"));
    assert_eq!(p.editor.snapshot_token(), original);
    p.build().unwrap();
    set_frequency(&mut p, 3);
    let changed = p.editor.snapshot_token();
    assert!(p
        .event(
            "sequence:frequency",
            &UiEvent::Commit {
                value: ActionValue::text("invalid").unwrap(),
            }
        )
        .unwrap_err()
        .contains("StaleView"));
    assert_eq!(p.editor.snapshot_token(), changed);
    assert!(!p.acknowledge_saved(&original));
    assert!(p.fenced());
}

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "prism-native-sequence-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).unwrap();
        Self(dir)
    }
    fn file(&self) -> PathBuf {
        self.0.join("sequence.bin")
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn wait(mut poll: impl FnMut() -> bool) {
    let start = Instant::now();
    while !poll() {
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "sequence storage did not finish"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}
fn settle(p: &mut DesktopSequencePanel) -> Result<(), String> {
    let mut error = None;
    wait(|| {
        if let Err(e) = p.poll(Instant::now()) {
            error = Some(e);
        }
        p.next_wake().is_none()
    });
    error.map_or(Ok(()), Err)
}

#[test]
fn sequence_native_file_route_round_trips_full_project_and_requires_explicit_adoption() {
    let dir = Directory::new();
    let mut p = panel();
    p.mount_storage(
        dir.file(),
        Default::default(),
        TimelineFileRevision::Missing,
        Duration::from_secs(5),
    )
    .unwrap();
    let expected = p.editor.document().clone();
    let mut ui = ui(p);
    assert!(ui.panel_document_command(true).unwrap());
    assert!(ui.has_uncommitted_edit());
    wait(|| {
        ui.poll_native_panels(Instant::now()).unwrap();
        get(&ui).next_wake().is_none()
    });
    assert_eq!(get(&ui).storage_status(), Some("Saved durably"));
    assert!(!ui.has_uncommitted_edit());
    let bytes = std::fs::read(dir.file()).unwrap();
    let decoded = SequenceProjectDocument::decode(&bytes, Default::default()).unwrap();
    assert_eq!(
        decoded, expected,
        "includes three clocks, tracks, events and nested clip topology"
    );
    let token = get(&ui).editor.snapshot_token();
    assert!(ui.panel_document_command(false).unwrap());
    wait(|| {
        ui.poll_native_panels(Instant::now()).unwrap();
        get(&ui).next_wake().is_none()
    });
    assert_eq!(get(&ui).editor.snapshot_token(), token, "load is isolated");
    assert!(ui.has_uncommitted_edit());
    click(&mut ui, "sequence:file:adopt");
    assert_eq!(get(&ui).editor.document(), &expected);
    assert_ne!(get(&ui).editor.snapshot_token(), token);
    assert_eq!(get(&ui).editor.undo_len(), 0);
    assert!(get(&ui).controller.samples().is_empty());
    assert!(!ui.has_uncommitted_edit());
    ui.unmount_native_panel("sequence").unwrap();
}

#[test]
fn sequence_storage_stale_receipt_intervening_edit_and_draft_reject_adoption() {
    let dir = Directory::new();
    let mut p = panel();
    p.mount_storage(
        dir.file(),
        Default::default(),
        TimelineFileRevision::Missing,
        Duration::from_secs(5),
    )
    .unwrap();
    p.save_file().unwrap();
    let saved = p.editor.snapshot_token();
    set_frequency(&mut p, 3);
    settle(&mut p).unwrap();
    assert!(p.fenced());
    assert!(!p.acknowledge_saved(&saved));
    p.save_file().unwrap();
    settle(&mut p).unwrap();
    assert!(!p.fenced());
    p.load_file().unwrap();
    settle(&mut p).unwrap();
    let started = p.editor.snapshot_token();
    set_frequency(&mut p, 4);
    let current = p.editor.snapshot_token();
    assert!(p.adopt_loaded(&started).is_err());
    assert!(p.adopt_loaded(&current).is_err());
    assert_eq!(
        p.editor.document().document(ROOT).unwrap().ticks_per_second,
        4
    );
    p.discard_loaded();
    p.load_file().unwrap();
    settle(&mut p).unwrap();
    p.controller
        .apply(
            &mut p.editor,
            prism_editor_ui::SequencePanelAction::Source(CHILD),
        )
        .unwrap();
    assert!(p.has_clip_draft());
    assert!(p.adopt_loaded(&current).is_err());
    p.discard_clip_draft().unwrap();
    p.adopt_loaded(&current).unwrap();
    assert_eq!(
        p.editor.document().document(ROOT).unwrap().ticks_per_second,
        3
    );
    assert_eq!(p.editor.undo_len(), 0);
    assert!(!p.fenced());
}

#[test]
fn sequence_external_file_conflict_is_visible_and_cancel_retains_request_until_terminal() {
    let dir = Directory::new();
    let mut p = panel();
    p.mount_storage(
        dir.file(),
        Default::default(),
        TimelineFileRevision::Missing,
        Duration::from_secs(5),
    )
    .unwrap();
    p.save_file().unwrap();
    settle(&mut p).unwrap();
    std::fs::write(dir.file(), b"external owner").unwrap();
    set_frequency(&mut p, 3);
    let model = p.editor.snapshot_token();
    p.save_file().unwrap();
    assert!(settle(&mut p).is_err());
    assert!(p.fenced());
    assert_eq!(std::fs::read(dir.file()).unwrap(), b"external owner");
    assert_eq!(p.editor.snapshot_token(), model);
    let bytes = p.editor.document().encode(Default::default()).unwrap();
    std::fs::write(dir.file(), &bytes).unwrap();
    p.load_file().unwrap();
    assert!(matches!(
        p.cancel_file(),
        DesktopPanelCancelOutcome::Cancelled | DesktopPanelCancelOutcome::TooLate
    ));
    assert!(p.fenced());
    assert!(
        p.next_wake().is_some(),
        "cancel never drops the owned request"
    );
    settle(&mut p).unwrap();
    p.discard_loaded();
    assert_eq!(p.editor.snapshot_token(), model);
    assert_eq!(std::fs::read(dir.file()).unwrap(), bytes);
}

fn lock(dir: &Directory) -> std::fs::File {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(dir.0.join("sequence.bin.prism-timeline.lock"))
        .unwrap();
    file.lock().unwrap();
    file
}
fn wait_busy(p: &mut DesktopSequencePanel) {
    wait(|| {
        p.poll(Instant::now()).unwrap();
        p.storage_status().unwrap().contains("Busy")
    });
}
#[test]
fn sequence_actual_temporary_file_lock_retries_original_snapshot_without_blocking_ui() {
    let dir = Directory::new();
    let locked = lock(&dir);
    let mut p = panel();
    p.mount_storage(
        dir.file(),
        Default::default(),
        TimelineFileRevision::Missing,
        Duration::from_secs(5),
    )
    .unwrap();
    p.save_file().unwrap();
    wait_busy(&mut p);
    assert!(p.fenced());
    assert!(p.next_wake().is_some());
    assert!(!dir.file().exists());
    set_frequency(&mut p, 3);
    locked.unlock().unwrap();
    settle(&mut p).unwrap();
    let decoded =
        SequenceProjectDocument::decode(&std::fs::read(dir.file()).unwrap(), Default::default())
            .unwrap();
    assert_eq!(
        decoded.document(ROOT).unwrap().ticks_per_second,
        2,
        "retry saves the original immutable bytes"
    );
    assert!(
        p.fenced(),
        "old snapshot receipt cannot confirm a newer frequency"
    );
    p.save_file().unwrap();
    settle(&mut p).unwrap();
    assert!(!p.fenced());
}
#[test]
fn sequence_actual_persistent_file_lock_exhausts_then_requires_explicit_retry_or_cancel() {
    let dir = Directory::new();
    let locked = lock(&dir);
    let mut p = panel();
    p.mount_storage(
        dir.file(),
        Default::default(),
        TimelineFileRevision::Missing,
        Duration::from_secs(5),
    )
    .unwrap();
    p.save_file().unwrap();
    assert!(settle(&mut p).unwrap_err().contains("Busy"));
    assert!(p
        .storage_status()
        .unwrap()
        .contains(dir.file().to_str().unwrap()));
    assert!(p.fenced());
    assert!(p.next_wake().is_none());
    assert!(
        !p.poll(Instant::now()).unwrap(),
        "exhausted operations never auto retry"
    );
    assert!(!dir.file().exists());
    p.save_file().unwrap();
    wait_busy(&mut p);
    assert_eq!(p.cancel_file(), DesktopPanelCancelOutcome::Cancelled);
    assert!(
        p.next_wake().is_none(),
        "waiting retry has no outstanding worker request"
    );
    locked.unlock().unwrap();
    assert!(!p.poll(Instant::now()).unwrap());
    assert!(
        !dir.file().exists(),
        "cancel cannot publish on a later wake"
    );
    p.save_file().unwrap();
    settle(&mut p).unwrap();
    assert_eq!(p.storage_status(), Some("Saved durably"));
    assert!(!p.fenced());
}
