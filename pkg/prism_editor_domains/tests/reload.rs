//! Real source files, executable Loom replacement and exact publication fences.
use prism_editor_domains::{reload::*, reload_loom::*};
use prism_ui::{Element, RecordingBackend, Ui};
use prism_ui_hotreload::{paths_of, HotReloader};
use prism_ui_style::{StyleProp, StyleValue};
use std::{
    fs,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Files(PathBuf);
impl Files {
    fn new() -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "prism-reload-{}-{nonce}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        Self(root)
    }
    fn write(&self, name: &str, value: impl AsRef<[u8]>) {
        fs::write(self.0.join(name), value).unwrap();
    }
}
impl Drop for Files {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn provider() -> LoomReloadProvider {
    LoomReloadProvider::new("panel.loom".into(), vec![])
}
fn pipeline(files: &Files, limits: ReloadLimits) -> ReloadPipeline<LoomReloadProvider> {
    ReloadPipeline::new(
        &files.0,
        ["panel.loom".into()],
        LOOM_LITERAL_SCHEMA.to_vec(),
        provider(),
        limits,
    )
    .unwrap()
}
fn ready<A>(mut request: ReloadRequest<A>) -> Result<PreparedReload<A>, ReloadError> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(prepared) = request.poll()? {
            return Ok(prepared);
        }
        assert!(Instant::now() < deadline, "worker did not finish");
        std::thread::sleep(Duration::from_millis(1));
    }
}
fn start<P: ReloadProvider>(lane: &mut ReloadPipeline<P>) -> ReloadRequest<P::Artifact> {
    lane.start(Instant::now(), true).unwrap().unwrap()
}

#[test]
fn real_loom_reload_preserves_scroll_input_animation_and_reconciles_runtime() {
    struct Transient {
        scroll: f32,
        input: String,
        animation: f32,
    }
    let files = Files::new();
    files.write("panel.loom", r#"box { key: "root"; text("新文案\n保留空格 ") { key: "field"; style: { opacity: 0.5; }; } text("added") { key: "new"; } }"#);
    let mut lane = pipeline(&files, ReloadLimits::default());
    let initial = Element::box_()
        .key_str("root")
        .child(Element::text("old").key_str("field"))
        .child(Element::text("removed").key_str("old"));
    let field = paths_of(&initial)[1].0.clone();
    let removed = paths_of(&initial)[2].0.clone();
    let mut live = HotReloader::new(initial);
    live.state_mut().insert(
        field.clone(),
        Transient {
            scroll: 120.0,
            input: "typing 未保存".into(),
            animation: 0.75,
        },
    );
    live.state_mut().insert(
        removed,
        Transient {
            scroll: 0.0,
            input: String::new(),
            animation: 0.0,
        },
    );
    let mut slot = lane.slot(live);
    let mut ui = Ui::new(RecordingBackend::new());
    ui.mount(slot.get().current());
    let candidate = ready(start(&mut lane)).unwrap();
    let plan = candidate.artifact().plan(slot.get()).unwrap();
    assert_eq!(plan.counts().preserved, 2);
    assert_eq!(plan.counts().added, 1);
    let report = publish_loom(&lane, candidate, &mut slot).unwrap();
    assert_eq!(report.dropped, 1);
    let state = slot.get().state().get(&field).unwrap();
    assert_eq!(
        (state.scroll, state.input.as_str(), state.animation),
        (120.0, "typing 未保存", 0.75)
    );
    assert_eq!(
        slot.get().current().child_elements()[0].text_content(),
        Some("新文案\n保留空格 ")
    );
    ui.backend_mut().clear();
    ui.update(slot.get().current());
    assert!(!ui.backend().ops().is_empty());
    assert_eq!(slot.get().state().len(), 1);
}

#[test]
fn style_file_changes_are_resolved_into_real_tree_before_publication() {
    let files = Files::new();
    files.write(
        "panel.loom",
        r#"box { class: "card"; style: { opacity: 0.8; }; }"#,
    );
    files.write("panel.loom.style", r#"class "card" { width: px(123); opacity: 0.2; background_color: rgba8(12, 30, 44, 255); }"#);
    let mut lane = ReloadPipeline::new(
        &files.0,
        ["panel.loom".into(), "panel.loom.style".into()],
        LOOM_LITERAL_SCHEMA.to_vec(),
        LoomReloadProvider::new("panel.loom".into(), vec!["panel.loom.style".into()]),
        ReloadLimits::default(),
    )
    .unwrap();
    let mut slot = lane.slot(HotReloader::<()>::new(Element::box_()));
    let candidate = ready(start(&mut lane)).unwrap();
    publish_loom(&lane, candidate, &mut slot).unwrap();
    assert!(slot
        .get()
        .current()
        .inline_pairs()
        .contains(&(StyleProp::Width, StyleValue::px(123.0))));
    assert!(slot
        .get()
        .current()
        .inline_pairs()
        .contains(&(StyleProp::Opacity, StyleValue::number(0.8))));
    files.write("panel.loom.style", r#"class "card" { width: px(456); }"#);
    lane.notify("panel.loom.style", Instant::now()).unwrap();
    let candidate = ready(start(&mut lane)).unwrap();
    publish_loom(&lane, candidate, &mut slot).unwrap();
    assert!(slot
        .get()
        .current()
        .inline_pairs()
        .contains(&(StyleProp::Width, StyleValue::px(456.0))));
}

#[test]
fn invalid_syntax_schema_keys_and_budgets_leave_last_good() {
    let files = Files::new();
    files.write("panel.loom", "box {}");
    let mut lane = pipeline(
        &files,
        ReloadLimits {
            nodes: 3,
            depth: 2,
            ..ReloadLimits::default()
        },
    );
    let mut slot = lane.slot(HotReloader::<()>::new(Element::text("last-good")));
    for source in [
        r#"box { text("one") { key: "x"; } box { key: "x"; } }"#,
        "box { box { box {} } }",
        "box { box {} box {} box {} }",
        "box { style: { opacity: 2; }; }",
        "text($dynamic) {}",
        "box { for_each(xs) }",
        "box { style: { color: px(1); }; }",
        "box { class: \"missing\"; }",
    ] {
        files.write("panel.loom", source);
        lane.notify("panel.loom", Instant::now()).unwrap();
        assert!(ready(start(&mut lane)).is_err(), "accepted {source}");
        assert_eq!(slot.get().current().text_content(), Some("last-good"));
    }
    files.write("panel.loom", "box {}");
    lane.set_schema(b"unknown".to_vec(), Instant::now())
        .unwrap();
    assert!(matches!(
        ready(start(&mut lane)),
        Err(ReloadError::Provider(_))
    ));
    lane.set_schema(LOOM_LITERAL_SCHEMA.to_vec(), Instant::now())
        .unwrap();
    let candidate = ready(start(&mut lane)).unwrap();
    publish_loom(&lane, candidate, &mut slot).unwrap();
}

#[test]
fn new_event_before_debounce_and_schema_aba_reject_ready_results() {
    let files = Files::new();
    files.write("panel.loom", "box {}");
    let mut lane = pipeline(&files, ReloadLimits::default());
    let mut slot = lane.slot(HotReloader::<()>::new(Element::text("old")));
    let result = ready(start(&mut lane)).unwrap();
    lane.notify("panel.loom", Instant::now()).unwrap();
    assert!(matches!(
        publish_loom(&lane, result, &mut slot),
        Err(ReloadError::Stale)
    ));
    let result = ready(start(&mut lane)).unwrap();
    lane.set_schema(b"changed".to_vec(), Instant::now())
        .unwrap();
    lane.set_schema(LOOM_LITERAL_SCHEMA.to_vec(), Instant::now())
        .unwrap();
    assert!(matches!(
        publish_loom(&lane, result, &mut slot),
        Err(ReloadError::Stale)
    ));
    assert_eq!(slot.get().current().text_content(), Some("old"));
}

#[test]
fn on_disk_edit_without_event_and_foreign_slot_are_rejected() {
    let files = Files::new();
    files.write("panel.loom", "box {}");
    let mut lane = pipeline(&files, ReloadLimits::default());
    let other = pipeline(&files, ReloadLimits::default());
    let mut slot = lane.slot(HotReloader::<()>::new(Element::box_()));
    let candidate = ready(start(&mut lane)).unwrap();
    files.write("panel.loom", "text(\"changed\") {}");
    assert!(matches!(
        publish_loom(&lane, candidate, &mut slot),
        Err(ReloadError::Stale)
    ));
    lane.notify("panel.loom", Instant::now()).unwrap();
    let candidate = ready(start(&mut lane)).unwrap();
    let mut foreign = other.slot(HotReloader::<()>::new(Element::box_()));
    assert!(matches!(
        publish_loom(&lane, candidate, &mut foreign),
        Err(ReloadError::Stale)
    ));
}

#[test]
fn second_capture_and_host_failure_preserve_last_good() {
    let files = Files::new();
    files.write("panel.loom", "box {}");
    let mut lane = pipeline(&files, ReloadLimits::default());
    let mut slot = lane.slot(7);
    let candidate = ready(start(&mut lane)).unwrap();
    let outcome = lane.publish(
        candidate,
        &mut slot,
        |_| {
            files.write("panel.loom", "text(\"new\") {}");
            Ok(8)
        },
        |live, _, candidate| *live = candidate,
    );
    assert_eq!(outcome, Err(ReloadError::Stale));
    assert_eq!(*slot.get(), 7);
    lane.notify("panel.loom", Instant::now()).unwrap();
    let candidate = ready(start(&mut lane)).unwrap();
    assert_eq!(
        lane.publish(
            candidate,
            &mut slot,
            |_| Err::<(), _>(ReloadError::provider("host failed")),
            |live, _, ()| *live = 10
        ),
        Err(ReloadError::provider("host failed"))
    );
    assert_eq!(*slot.get(), 7);
}

#[test]
fn debounce_is_bounded_and_repeated_events_do_not_grow_queue() {
    let files = Files::new();
    files.write("panel.loom", "box {}");
    let mut lane = pipeline(&files, ReloadLimits::default());
    let t = Instant::now();
    for _ in 0..10_000 {
        lane.notify("panel.loom", t).unwrap();
    }
    assert_eq!(lane.pending_paths(), 1);
    assert!(lane.start(t, false).unwrap().is_none());
    lane.notify("panel.loom", t + Duration::from_millis(249))
        .unwrap();
    let request = lane
        .start(t + Duration::from_millis(250), false)
        .unwrap()
        .unwrap();
    assert!(ready(request).is_ok());
    assert_eq!(lane.pending_paths(), 0);
    assert!(matches!(
        lane.notify("unknown", t),
        Err(ReloadError::Path(_))
    ));
}

#[test]
fn ready_success_and_error_results_hold_capacity() {
    let files = Files::new();
    files.write("panel.loom", "box {}");
    let mut lane = pipeline(
        &files,
        ReloadLimits {
            in_flight: 1,
            ..ReloadLimits::default()
        },
    );
    let candidate = ready(start(&mut lane)).unwrap();
    lane.notify("panel.loom", Instant::now()).unwrap();
    assert!(matches!(
        lane.start(Instant::now(), true),
        Err(ReloadError::Budget(_))
    ));
    drop(candidate);
    files.write("panel.loom", "invalid");
    let request = start(&mut lane);
    std::thread::sleep(Duration::from_millis(20));
    lane.notify("panel.loom", Instant::now()).unwrap();
    assert!(matches!(
        lane.start(Instant::now(), true),
        Err(ReloadError::Budget(_))
    ));
    drop(request);
    wait_until(|| lane.in_flight() == 0);
}

struct GatedProvider {
    entered: Arc<AtomicBool>,
    release: Arc<AtomicBool>,
}
impl ReloadProvider for GatedProvider {
    type Artifact = LoomCandidate;
    fn prepare(
        &self,
        sources: &ReloadSources,
        schema: &[u8],
        cancellation: &ReloadCancellation,
        limits: &ReloadLimits,
    ) -> Result<LoomCandidate, ReloadError> {
        self.entered.store(true, Ordering::Release);
        while !self.release.load(Ordering::Acquire) {
            std::thread::sleep(Duration::from_millis(1));
        }
        provider().prepare(sources, schema, cancellation, limits)
    }
}
fn wait_until(mut condition: impl FnMut() -> bool) {
    let end = Instant::now() + Duration::from_secs(5);
    while !condition() {
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn cancellation_and_timeout_keep_permit_until_worker_exit() {
    for timeout in [false, true] {
        let files = Files::new();
        files.write("panel.loom", "box {}");
        let entered = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let limits = ReloadLimits {
            in_flight: 1,
            timeout: Duration::from_millis(if timeout { 50 } else { 5000 }),
            ..ReloadLimits::default()
        };
        let mut lane = ReloadPipeline::new(
            &files.0,
            ["panel.loom".into()],
            LOOM_LITERAL_SCHEMA.to_vec(),
            GatedProvider {
                entered: entered.clone(),
                release: release.clone(),
            },
            limits,
        )
        .unwrap();
        let mut request = start(&mut lane);
        wait_until(|| entered.load(Ordering::Acquire));
        if timeout {
            std::thread::sleep(Duration::from_millis(60));
        } else {
            request.cancel();
        }
        assert_eq!(
            request.poll().err().unwrap(),
            if timeout {
                ReloadError::TimedOut
            } else {
                ReloadError::Cancelled
            }
        );
        drop(request);
        lane.notify("panel.loom", Instant::now()).unwrap();
        assert_eq!(lane.in_flight(), 1);
        assert!(matches!(
            lane.start(Instant::now(), true),
            Err(ReloadError::Budget(_))
        ));
        release.store(true, Ordering::Release);
        wait_until(|| lane.in_flight() == 0);
        assert!(ready(start(&mut lane)).is_ok());
    }
}

#[test]
fn invalid_utf8_deleted_file_and_byte_limit_are_explicit() {
    let files = Files::new();
    files.write("panel.loom", [255u8]);
    let mut lane = pipeline(
        &files,
        ReloadLimits {
            file_bytes: 32,
            total_bytes: 32,
            ..ReloadLimits::default()
        },
    );
    assert!(matches!(
        ready(start(&mut lane)),
        Err(ReloadError::Provider(_))
    ));
    files.write("panel.loom", vec![b' '; 33]);
    lane.notify("panel.loom", Instant::now()).unwrap();
    assert!(matches!(
        ready(start(&mut lane)),
        Err(ReloadError::Budget(_))
    ));
    fs::remove_file(files.0.join("panel.loom")).unwrap();
    lane.notify("panel.loom", Instant::now()).unwrap();
    assert!(matches!(ready(start(&mut lane)), Err(ReloadError::Io(_))));
}

#[test]
fn invalid_registration_and_duplicate_live_identities_fail_before_plan() {
    let files = Files::new();
    files.write("panel.loom", "box {}");
    assert!(ReloadPipeline::new(
        &files.0,
        ["../bad".into()],
        LOOM_LITERAL_SCHEMA.to_vec(),
        provider(),
        ReloadLimits::default()
    )
    .is_err());
    assert!(ReloadPipeline::new(
        &files.0,
        ["a".into(), "a".into()],
        LOOM_LITERAL_SCHEMA.to_vec(),
        provider(),
        ReloadLimits::default()
    )
    .is_err());
    let mut lane = pipeline(&files, ReloadLimits::default());
    let old = Element::box_()
        .child(Element::text("a").key_str("dup"))
        .child(Element::box_().key_str("dup"));
    let mut slot = lane.slot(HotReloader::<()>::new(old));
    let candidate = ready(start(&mut lane)).unwrap();
    assert!(candidate.artifact().plan(slot.get()).is_err());
    assert!(publish_loom(&lane, candidate, &mut slot).is_err());
    assert_eq!(slot.get().current().child_elements().len(), 2);
}

#[test]
fn successful_and_failed_reload_preserve_actual_command_stack_and_redo() {
    use prism_editor_core::{
        AffectSet, Command, CommandError, CommandStack, EditContext, MergeResult,
    };
    struct Document(i32);
    impl EditContext for Document {}
    struct SetValue {
        from: i32,
        to: i32,
        affected: AffectSet,
    }
    impl Command<Document> for SetValue {
        fn apply(&mut self, doc: &mut Document) -> Result<(), CommandError> {
            if doc.0 != self.from {
                return Err(CommandError::new("edit conflict"));
            }
            doc.0 = self.to;
            Ok(())
        }
        fn undo(&mut self, doc: &mut Document) -> Result<(), CommandError> {
            if doc.0 != self.to {
                return Err(CommandError::new("undo conflict"));
            }
            doc.0 = self.from;
            Ok(())
        }
        fn label(&self) -> &str {
            "document edit"
        }
        fn affected(&self) -> &AffectSet {
            &self.affected
        }
        fn merge_from(&mut self, _: &dyn Command<Document>) -> MergeResult {
            MergeResult::NotMergeable
        }
    }
    let mut doc = Document(1);
    let mut history = CommandStack::default();
    history
        .execute(
            Box::new(SetValue {
                from: 1,
                to: 2,
                affected: AffectSet::new(),
            }),
            &mut doc,
        )
        .unwrap();
    history
        .execute(
            Box::new(SetValue {
                from: 2,
                to: 3,
                affected: AffectSet::new(),
            }),
            &mut doc,
        )
        .unwrap();
    history.undo(&mut doc).unwrap();
    assert_eq!((doc.0, history.undo_len(), history.redo_len()), (2, 1, 1));
    let files = Files::new();
    files.write("panel.loom", "text(\"new view\") {}");
    let mut lane = pipeline(&files, ReloadLimits::default());
    let mut slot = lane.slot(HotReloader::<()>::new(Element::box_()));
    let candidate = ready(start(&mut lane)).unwrap();
    publish_loom(&lane, candidate, &mut slot).unwrap();
    files.write("panel.loom", "bad syntax");
    lane.notify("panel.loom", Instant::now()).unwrap();
    assert!(ready(start(&mut lane)).is_err());
    assert_eq!((doc.0, history.undo_len(), history.redo_len()), (2, 1, 1));
    assert_eq!(slot.get().current().text_content(), Some("new view"));
    history.redo(&mut doc).unwrap();
    assert_eq!(doc.0, 3);
    history.undo(&mut doc).unwrap();
    history.undo(&mut doc).unwrap();
    assert_eq!(doc.0, 1);
}

#[test]
fn explicit_stable_ids_preserve_macro_node_state_after_sibling_reorder() {
    let initial = prism_ui::loom! { box { text("left") {} text("right") {} } };
    let paths = paths_of(&initial);
    let left = paths[1].0.clone();
    let right = paths[2].0.clone();
    let mut live = HotReloader::new(initial);
    live.state_mut()
        .insert(left.clone(), String::from("left editing"));
    live.state_mut()
        .insert(right.clone(), String::from("right editing"));
    let files = Files::new();
    files.write("panel.loom", r#"box { stable_id: ""; text("right new") { stable_id: "1"; } text("left new") { stable_id: "0"; } }"#);
    let mut lane = pipeline(&files, ReloadLimits::default());
    let mut slot = lane.slot(live);
    let candidate = ready(start(&mut lane)).unwrap();
    assert_eq!(
        candidate
            .artifact()
            .plan(slot.get())
            .unwrap()
            .counts()
            .preserved,
        3
    );
    publish_loom(&lane, candidate, &mut slot).unwrap();
    assert_eq!(
        slot.get().current().child_elements()[0].text_content(),
        Some("right new")
    );
    assert_eq!(slot.get().state().get(&left).unwrap(), "left editing");
    assert_eq!(slot.get().state().get(&right).unwrap(), "right editing");
    files.write(
        "panel.loom",
        r#"box { text("a") { stable_id: "dup"; } box { stable_id: "dup"; } }"#,
    );
    lane.notify("panel.loom", Instant::now()).unwrap();
    assert!(ready(start(&mut lane)).is_err());
    assert_eq!(slot.get().state().len(), 2);
}
