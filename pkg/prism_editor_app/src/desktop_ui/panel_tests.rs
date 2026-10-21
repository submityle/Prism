use super::*;
use alloc::{rc::Rc, sync::Arc};
use prism_editor_timeline::*;
use prism_editor_ui::{
    ProfilerController, ProfilerDomain, ProfilerSources, TimelinePanelAction,
    TimelinePanelController, WorldPanelController, WorldPanelLimits, WorldPanelViewport,
};
use std::{
    cell::{Cell, RefCell},
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

fn editor() -> TimelineEditor {
    let mut timeline = Timeline::default();
    let mut track = Track::new(TrackId::from_raw(7), "Host channel", 1).unwrap();
    track
        .insert(Keyframe::new(
            KeyId::from_raw(1),
            0,
            vec![0.0],
            Interpolation::Linear,
        ))
        .unwrap();
    track
        .insert(Keyframe::new(
            KeyId::from_raw(2),
            100,
            vec![100.0],
            Interpolation::Linear,
        ))
        .unwrap();
    timeline.add_track(track).unwrap();
    timeline
        .add_event(TimelineEvent {
            id: 3,
            tick: 2,
            name: "Host event".into(),
        })
        .unwrap();
    TimelineEditor::new(timeline, TimelineLimits::default()).unwrap()
}
fn timeline(
    consumer: impl FnMut(&DesktopTimelineFrame) -> Result<(), String> + 'static,
) -> DesktopTimelinePanel {
    let mut e = editor();
    let mut c = TimelinePanelController::default();
    c.apply(
        &mut e,
        TimelinePanelAction::PlaybackConfig(PlaybackConfig {
            ticks_per_second: 100,
            start_tick: 0,
            end_tick: 100,
            mode: PlaybackMode::Once,
            max_events: 100,
        }),
    )
    .unwrap();
    DesktopTimelinePanel::new(e, c, consumer)
}
fn ui() -> EditorDesktopUi {
    EditorDesktopUi::new(EditorController::new()).unwrap()
}
fn rect(ui: &EditorDesktopUi, key: &str) -> Rect {
    ui.hits
        .iter()
        .find(|h| h.target == key)
        .unwrap_or_else(|| panic!("missing {key}"))
        .rect
}
fn click(ui: &mut EditorDesktopUi, key: &str) {
    let r = rect(ui, key);
    assert!(
        r.size.width > 0.0 && r.size.height > 0.0,
        "clipped {key}: {r:?}"
    );
    ui.click(
        r.left() + r.size.width * 0.5,
        r.top() + r.size.height * 0.5,
        SelectionMode::Replace,
    )
    .unwrap();
}
fn commit(ui: &mut EditorDesktopUi, key: &str, value: &str) {
    click(ui, key);
    ui.text(value).unwrap();
    ui.key(DesktopKey::Enter).unwrap();
}
fn visible(ui: &EditorDesktopUi, text: &str) -> bool {
    ui.draw_list().commands().iter().any(
        |cmd| matches!(cmd,DrawCommand::Glyph(g) if ui.draw_list().text(g.text).contains(text)),
    )
}

#[test]
fn actual_timeline_transport_fields_layout_and_authoring_undo_fences() {
    let mut ui = ui();
    ui.mount_native_panel("timeline", "Timeline", Box::new(timeline(|_| Ok(()))))
        .unwrap();
    assert!(visible(&ui, "Sequencer"));
    assert!(!ui.refresh().unwrap());
    let doc_revision = ui.controller().model_revision();
    commit(&mut ui, "timeline:seek", "12");
    assert_eq!(
        ui.native_panel::<DesktopTimelinePanel>("timeline")
            .unwrap()
            .controller
            .playhead(),
        12
    );
    click(&mut ui, "timeline:play");
    ui.poll_native_panels(Instant::now()).unwrap();
    ui.refresh().unwrap();
    assert!(ui.next_native_panel_wake().is_some());
    click(&mut ui, "timeline:pause");
    ui.poll_native_panels(Instant::now()).unwrap();
    ui.refresh().unwrap();
    assert!(ui.next_native_panel_wake().is_none());
    commit(&mut ui, "timeline:frequency", "120");
    assert_eq!(
        ui.native_panel::<DesktopTimelinePanel>("timeline")
            .unwrap()
            .controller
            .playback()
            .config()
            .ticks_per_second,
        120
    );
    // This control is in the actual creation row, not a synthetic dispatch.
    commit(&mut ui, "timeline:track-name", "New host track");
    click(&mut ui, "timeline:add-track");
    assert_eq!(
        ui.native_panel::<DesktopTimelinePanel>("timeline")
            .unwrap()
            .editor
            .timeline()
            .track_count(),
        2
    );
    assert!(ui.has_uncommitted_edit());
    assert!(ui.unmount_native_panel("timeline").is_err());
    click(&mut ui, "timeline:undo");
    assert_eq!(
        ui.native_panel::<DesktopTimelinePanel>("timeline")
            .unwrap()
            .editor
            .timeline()
            .track_count(),
        1
    );
    click(&mut ui, "timeline:redo");
    let stale = editor().snapshot_token();
    let p = ui
        .native_panel_mut::<DesktopTimelinePanel>("timeline")
        .unwrap();
    assert!(!p.acknowledge_saved(&stale, 120));
    let token = p.editor.snapshot_token();
    assert!(p.acknowledge_saved(&token, 120));
    ui.refresh().unwrap();
    assert!(!ui.has_uncommitted_edit());
    assert_eq!(ui.controller().model_revision(), doc_revision);
}

#[test]
fn monotonic_real_time_playback_delivers_actual_samples_events_hidden_and_no_scene_writes() {
    let frames = Rc::new(RefCell::new(Vec::new()));
    let output = frames.clone();
    let mut ui = ui();
    ui.mount_content_browser(ContentCatalog::new()).unwrap();
    ui.mount_native_panel(
        "timeline",
        "Timeline",
        Box::new(timeline(move |f| {
            output.borrow_mut().push(f.clone());
            Ok(())
        })),
    )
    .unwrap();
    click(&mut ui, "native:tab:timeline");
    click(&mut ui, "timeline:play");
    let start = Instant::now();
    ui.poll_native_panels(start).unwrap();
    ui.refresh().unwrap();
    click(&mut ui, "native:tab:content");
    let revision = ui.controller().model_revision();
    std::thread::sleep(Duration::from_millis(35));
    let now = Instant::now();
    assert!(now.duration_since(start) >= Duration::from_millis(30));
    ui.poll_native_panels(now).unwrap();
    assert!(!ui.refresh().unwrap());
    let output = frames.borrow();
    let latest = output.last().unwrap();
    assert!(latest.tick >= 3);
    let sample = &latest.samples[&TrackId::from_raw(7)];
    assert_eq!(sample.len(), 1);
    assert!((sample[0] - latest.tick as f64).abs() < 1e-9);
    assert_eq!(
        output
            .iter()
            .flat_map(|f| &f.events)
            .filter(|(e, _)| e.id == 3)
            .count(),
        1
    );
    assert_eq!(ui.controller().model_revision(), revision);
}

#[test]
fn rejected_delivery_retains_exact_frame_and_blocks_io_until_explicit_retry() {
    let fail = Rc::new(Cell::new(false));
    let failed = fail.clone();
    let attempts = Rc::new(RefCell::new(Vec::new()));
    let log = attempts.clone();
    let mut ui = ui();
    ui.mount_native_panel(
        "timeline",
        "Timeline",
        Box::new(timeline(move |f| {
            log.borrow_mut().push(f.clone());
            if failed.get() {
                Err("Host rejected whole frame".into())
            } else {
                Ok(())
            }
        })),
    )
    .unwrap();
    click(&mut ui, "timeline:play");
    let start = Instant::now();
    ui.poll_native_panels(start).unwrap();
    ui.refresh().unwrap();
    fail.set(true);
    assert!(ui
        .poll_native_panels(start + Duration::from_millis(30))
        .is_err());
    let retained = ui
        .native_panel::<DesktopTimelinePanel>("timeline")
        .unwrap()
        .pending_frame()
        .unwrap()
        .clone();
    assert!(ui.has_uncommitted_edit());
    assert!(ui.next_native_panel_wake().is_none());
    assert!(!ui
        .poll_native_panels(start + Duration::from_millis(60))
        .unwrap());
    fail.set(false);
    ui.native_panel_mut::<DesktopTimelinePanel>("timeline")
        .unwrap()
        .retry();
    ui.poll_native_panels(start + Duration::from_millis(90))
        .unwrap();
    assert_eq!(attempts.borrow().last(), Some(&retained));
    assert!(!ui.has_uncommitted_edit());
}

#[test]
fn invalid_timeline_text_retains_draft_ime_and_blocks_tab_switch() {
    let mut ui = ui();
    ui.mount_content_browser(ContentCatalog::new()).unwrap();
    ui.mount_native_panel("timeline", "Timeline", Box::new(timeline(|_| Ok(()))))
        .unwrap();
    click(&mut ui, "native:tab:timeline");
    click(&mut ui, "timeline:seek");
    ui.text("bad tick").unwrap();
    assert!(ui.key(DesktopKey::Enter).is_err());
    assert_eq!(ui.focus.as_ref().unwrap().buffer, "bad tick");
    let r = rect(&ui, "native:tab:content");
    assert!(ui
        .click(r.left() + 2., r.top() + 2., SelectionMode::Replace)
        .is_err());
    assert_eq!(ui.active_native_panel(), Some("timeline"));
    assert!(ui.has_uncommitted_edit());
    ui.preedit("组合").unwrap();
    assert!(!ui.key(DesktopKey::Enter).unwrap());
    ui.key(DesktopKey::Escape).unwrap();
    click(&mut ui, "native:tab:content");
    assert_eq!(ui.active_native_panel(), None);
}

fn world_controller(with_actor: bool) -> WorldPanelController {
    use prism_asset::StableGuid;
    use prism_ecs::partition::{cell::CellCoord, data_layer::DataLayerId};
    use prism_editor_world::{DataLayer, EditorWorldData, EditorWorldLimits};
    let mut world = EditorWorldData::new(
        StableGuid::from_path("host/world"),
        EditorWorldLimits::new(4, 4096, 4096, 4, 8192).unwrap(),
    )
    .unwrap();
    world
        .add_layer(DataLayer::new(DataLayerId::new(11), "Host layer").unwrap())
        .unwrap();
    world
        .add_cell(CellCoord::new(0, 0, 0), [DataLayerId::new(11)])
        .unwrap();
    if with_actor {
        world
            .register_actor(StableGuid::from_path("host/actor"), CellCoord::new(0, 0, 0))
            .unwrap();
    }
    WorldPanelController::new(
        world,
        StableGuid::from_path("host/user"),
        WorldPanelViewport {
            size: [128., 64.],
            ..Default::default()
        },
        WorldPanelLimits::default(),
    )
    .unwrap()
}

fn world() -> WorldPanelController {
    world_controller(false)
}

fn world_with_actor() -> WorldPanelController {
    world_controller(true)
}

struct WorldProjectTemp(PathBuf);
impl WorldProjectTemp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "prism-editor-app-world-project-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn file(&self) -> PathBuf {
        self.0.join("world.prismproject")
    }
}
impl Drop for WorldProjectTemp {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn world_project_binding_survives_fresh_app_session_and_fences_retired_runtime() {
    use prism_asset::StableGuid;
    use prism_ecs::partition::cell::CellCoord;
    use prism_editor_world::project::ProjectLimits;

    let dir = WorldProjectTemp::new();
    let path = dir.file();
    let binding = DesktopWorldProjectBinding::new(
        path.clone(),
        Some(TimelineFileRevision::Missing),
        1,
        ProjectLimits::default(),
        Duration::from_secs(5),
    );
    let controller = world_with_actor();

    let provider_polls = Rc::new(Cell::new(0));
    let observed = provider_polls.clone();
    let first_panel = DesktopWorldPanel::with_project(controller, binding, move |_, _| {
        observed.set(observed.get() + 1);
        Ok(DesktopPanelUpdate::default())
    })
    .unwrap();
    assert_eq!(first_panel.project_path(), Some(path.as_path()));
    assert_eq!(
        first_panel.project_file_revision(),
        Some(&TimelineFileRevision::Missing)
    );

    let mut first = ui();
    first
        .mount_native_panel("world", "World", Box::new(first_panel))
        .unwrap();
    first.refresh().unwrap();
    click(&mut first, "world:cell:0:0:0");
    click(&mut first, "world:layer:11:visible");
    assert!(
        !first
            .native_panel::<DesktopWorldPanel>("world")
            .unwrap()
            .controller
            .project()
            .layer(prism_ecs::partition::data_layer::DataLayerId::new(11))
            .unwrap()
            .visible
    );
    assert!(first.has_uncommitted_edit());
    click(&mut first, "world:project:save");
    assert!(first.next_native_panel_wake().is_some());

    let deadline = Instant::now() + Duration::from_secs(10);
    while first
        .native_panel::<DesktopWorldPanel>("world")
        .unwrap()
        .controller
        .project_io_pending()
    {
        first.poll_native_panels(Instant::now()).unwrap();
        assert!(Instant::now() < deadline, "world project save timed out");
        std::thread::yield_now();
    }
    first.refresh().unwrap();

    let panel = first.native_panel::<DesktopWorldPanel>("world").unwrap();
    assert!(provider_polls.get() > 0);
    assert!(!panel.controller.project_is_dirty());
    assert!(matches!(
        panel.project_file_revision(),
        Some(TimelineFileRevision::Present(_))
    ));
    assert!(path.is_file());
    assert!(!first.has_uncommitted_edit());
    assert!(first.next_native_panel_wake().is_none());

    click(&mut first, "world:cell:0:0:0");
    click(&mut first, "world:load");
    let old_intent = first
        .native_panel_mut::<DesktopWorldPanel>("world")
        .unwrap()
        .controller
        .take_intent()
        .unwrap();
    drop(first);

    let reopened = DesktopWorldProjectBinding::new(
        path.clone(),
        None,
        1,
        ProjectLimits::default(),
        Duration::from_secs(5),
    );
    let second_panel = DesktopWorldPanel::with_project(world_with_actor(), reopened, |_, _| {
        Ok(DesktopPanelUpdate::default())
    })
    .unwrap();
    assert_eq!(second_panel.project_path(), Some(path.as_path()));
    assert_eq!(second_panel.project_file_revision(), None);
    assert!(!second_panel.controller.project_can_save());

    let mut second = ui();
    second
        .mount_native_panel("world", "World", Box::new(second_panel))
        .unwrap();
    second.refresh().unwrap();
    click(&mut second, "world:project:load");
    assert!(second.next_native_panel_wake().is_some());
    let deadline = Instant::now() + Duration::from_secs(10);
    while second
        .native_panel::<DesktopWorldPanel>("world")
        .unwrap()
        .controller
        .project_io_pending()
    {
        second.poll_native_panels(Instant::now()).unwrap();
        assert!(Instant::now() < deadline, "world project load timed out");
        std::thread::yield_now();
    }
    second.refresh().unwrap();
    assert!(second
        .native_panel::<DesktopWorldPanel>("world")
        .unwrap()
        .controller
        .project_candidate()
        .is_some());
    assert!(second.has_uncommitted_edit());

    click(&mut second, "world:project:adopt-discard");
    let panel = second.native_panel::<DesktopWorldPanel>("world").unwrap();
    let project = panel.controller.project();
    assert_eq!(project.identity(), StableGuid::from_path("host/world"));
    assert!(project.revision() > 0);
    assert_eq!(project.layers().len(), 1);
    assert!(
        !project
            .layer(prism_ecs::partition::data_layer::DataLayerId::new(11))
            .unwrap()
            .visible
    );
    assert_eq!(project.cells().len(), 1);
    assert_eq!(project.actors().len(), 1);
    assert_eq!(
        project.actor_cell(StableGuid::from_path("host/actor")),
        Some(CellCoord::new(0, 0, 0))
    );
    assert!(matches!(
        panel.project_file_revision(),
        Some(TimelineFileRevision::Present(_))
    ));
    assert!(panel.controller.has_retired_world());
    assert!(second.has_uncommitted_edit());

    assert!(second
        .native_panel_mut::<DesktopWorldPanel>("world")
        .unwrap()
        .controller
        .begin_residency(&old_intent)
        .is_err());
    let retired = second
        .native_panel_mut::<DesktopWorldPanel>("world")
        .unwrap()
        .controller
        .take_retired_world()
        .unwrap();
    assert!(retired
        .model()
        .layer(prism_ecs::partition::data_layer::DataLayerId::new(11))
        .unwrap()
        .is_visible());
    second.refresh().unwrap();
    assert!(!second.has_uncommitted_edit());
    assert!(second.unmount_native_panel("world").is_ok());
}

#[test]
fn real_world_tree_select_load_reaches_host_intent_without_fake_completion() {
    use prism_ecs::partition::cell::CellCoord;
    let intents = Rc::new(RefCell::new(Vec::new()));
    let log = intents.clone();
    let mut ui = ui();
    ui.mount_native_panel(
        "world",
        "World",
        Box::new(DesktopWorldPanel::new(world(), move |c, _| {
            let Some(intent) = c.take_intent() else {
                return Ok(DesktopPanelUpdate::default());
            };
            log.borrow_mut().push(intent.kind());
            // A real host would submit an I/O request. Here retain the exact token
            // in Loading and verify that app never fabricates a completion.
            c.begin_residency(&intent).map_err(|e| e.to_string())?;
            Ok(DesktopPanelUpdate {
                changed: true,
                fenced: true,
                ..Default::default()
            })
        })),
    )
    .unwrap();
    assert!(visible(&ui, "Host layer") || visible(&ui, "World"));
    click(&mut ui, "world:cell:0:0:0");
    assert_eq!(
        ui.native_panel::<DesktopWorldPanel>("world")
            .unwrap()
            .controller
            .selected(),
        Some(CellCoord::new(0, 0, 0))
    );
    click(&mut ui, "world:load");
    assert!(ui.has_uncommitted_edit());
    ui.poll_native_panels(Instant::now()).unwrap();
    ui.refresh().unwrap();
    assert_eq!(intents.borrow().len(), 1);
    assert!(matches!(
        ui.native_panel::<DesktopWorldPanel>("world")
            .unwrap()
            .controller
            .world()
            .model()
            .cell(CellCoord::new(0, 0, 0))
            .unwrap()
            .residency(),
        prism_editor_world::CellResidency::Loading(_)
    ));
    assert!(ui.unmount_native_panel("world").is_err());
}

#[test]
fn real_cpu_capture_provider_renders_selectable_native_profiler_spans_and_domains() {
    use prism_editor_profiler::{capture::CpuProducer, FrameId, FrameStore, ProfilerLimits};
    let limits = ProfilerLimits::default();
    let mut producer = CpuProducer::new(limits, 1).unwrap();
    let f = producer.begin_frame(FrameId::from_raw(22), 22).unwrap();
    let s = producer
        .begin_span(f, "Native measured work", "host main")
        .unwrap();
    std::thread::sleep(Duration::from_millis(2));
    producer.end_span(s).unwrap();
    let frame = producer.finish_frame(f).unwrap();
    assert!(frame.duration_ns() > 0);
    let mut store = FrameStore::new(limits);
    store.insert(frame).unwrap();
    let mut sources = Some(ProfilerSources {
        cpu: Some(Arc::new(store)),
        gpu: None,
    });
    let mut ui = ui();
    ui.mount_native_panel(
        "profiler",
        "Profiler",
        Box::new(DesktopProfilerPanel::new(
            ProfilerController::default(),
            move |_| Ok((sources.take(), DesktopPanelUpdate::default())),
        )),
    )
    .unwrap();
    ui.poll_native_panels(Instant::now()).unwrap();
    ui.refresh().unwrap();
    let view = ui
        .native_panel::<DesktopProfilerPanel>("profiler")
        .unwrap()
        .controller
        .build();
    let span = view.spans()[0].clone();
    assert!(visible(&ui, "host main"));
    click(&mut ui, &span.key);
    assert!(visible(&ui, "Native measured work"));
    assert_eq!(
        ui.native_panel::<DesktopProfilerPanel>("profiler")
            .unwrap()
            .controller
            .selected_span(),
        Some(span.span)
    );
    click(&mut ui, "profiler:gpu");
    assert!(ui
        .native_panel::<DesktopProfilerPanel>("profiler")
        .unwrap()
        .controller
        .selection()
        .is_none());
    click(&mut ui, "profiler:cpu");
    assert_eq!(
        ui.native_panel::<DesktopProfilerPanel>("profiler")
            .unwrap()
            .controller
            .selection()
            .unwrap()
            .domain,
        ProfilerDomain::Cpu
    );
    assert!(!ui.refresh().unwrap());
    assert!(ui.next_native_panel_wake().is_none());
}

struct Contribution {
    events: Rc<Cell<usize>>,
    fenced: bool,
}
impl DesktopPanel for Contribution {
    fn build(&mut self) -> Result<DesktopPanelView, String> {
        Ok(DesktopPanelView {
            element: Element::box_().child(
                Element::box_()
                    .key_str("host:action")
                    .style(StyleProp::Width, StyleValue::px(100.))
                    .style(StyleProp::Height, StyleValue::px(40.))
                    .child(Element::text("Host contribution")),
            ),
            bindings: vec![("host:action".into(), UiTrigger::Activate)],
        })
    }
    fn event(&mut self, _: &str, _: &UiEvent) -> Result<(), String> {
        self.events.set(self.events.get() + 1);
        Ok(())
    }
    fn fenced(&self) -> bool {
        self.fenced
    }
}
#[test]
fn contribution_slot_real_layout_routes_and_unmount_respects_io_fence() {
    let events = Rc::new(Cell::new(0));
    let mut ui = ui();
    ui.mount_native_panel(
        "host",
        "Host",
        Box::new(Contribution {
            events: events.clone(),
            fenced: false,
        }),
    )
    .unwrap();
    click(&mut ui, "host:action");
    assert_eq!(events.get(), 1);
    assert!(visible(&ui, "Host contribution"));
    ui.native_panel_mut::<Contribution>("host").unwrap().fenced = true;
    assert!(ui.unmount_native_panel("host").is_err());
    ui.native_panel_mut::<Contribution>("host").unwrap().fenced = false;
    let host = ui.unmount_native_panel("host").unwrap();
    assert!((host.as_ref() as &dyn std::any::Any).is::<Contribution>());
    assert!(!ui.has_uncommitted_edit());
}

struct FallibleContribution {
    fail: Rc<Cell<bool>>,
}
impl DesktopPanel for FallibleContribution {
    fn build(&mut self) -> Result<DesktopPanelView, String> {
        if self.fail.get() {
            return Err("Host view is temporarily unavailable".into());
        }
        Ok(DesktopPanelView {
            element: Element::text("Remaining host"),
            bindings: Vec::new(),
        })
    }
    fn event(&mut self, _: &str, _: &UiEvent) -> Result<(), String> {
        Err("No interactive controls".into())
    }
}

#[test]
fn failed_unmount_reconciliation_keeps_original_host_and_active_tab() {
    let mut ui = ui();
    let events = Rc::new(Cell::new(0));
    ui.mount_native_panel(
        "host",
        "Host",
        Box::new(Contribution {
            events: events.clone(),
            fenced: false,
        }),
    )
    .unwrap();
    let fail = Rc::new(Cell::new(false));
    ui.mount_native_panel(
        "remaining",
        "Remaining",
        Box::new(FallibleContribution { fail: fail.clone() }),
    )
    .unwrap();
    fail.set(true);
    assert!(ui.unmount_native_panel("host").is_err());
    assert_eq!(ui.active_native_panel(), Some("host"));
    let retained = ui.native_panel::<Contribution>("host").unwrap();
    assert!(Rc::ptr_eq(&retained.events, &events));
    ui.refresh().unwrap();
    click(&mut ui, "host:action");
    assert_eq!(events.get(), 1);
    fail.set(false);
    let removed = ui.unmount_native_panel("host").unwrap();
    assert!((removed.as_ref() as &dyn std::any::Any).is::<Contribution>());
    assert_eq!(ui.active_native_panel(), Some("remaining"));
}

#[test]
fn all_three_tabs_mount_without_model_copies_or_shared_root_changes() {
    let mut ui = ui();
    ui.mount_native_panel("timeline", "Timeline", Box::new(timeline(|_| Ok(()))))
        .unwrap();
    ui.mount_native_panel(
        "profiler",
        "Profiler",
        Box::new(DesktopProfilerPanel::new(
            ProfilerController::default(),
            |_| Ok((None, DesktopPanelUpdate::default())),
        )),
    )
    .unwrap();
    ui.mount_native_panel(
        "world",
        "World",
        Box::new(DesktopWorldPanel::new(world(), |_, _| {
            Ok(DesktopPanelUpdate::default())
        })),
    )
    .unwrap();
    for (tab, key) in [
        ("native:tab:world", "world:cell:0:0:0"),
        ("native:tab:profiler", "profiler:cpu"),
        ("native:tab:timeline", "timeline:play"),
    ] {
        click(&mut ui, tab);
        let r = rect(&ui, key);
        assert!(r.size.height > 0.0);
    }
    ui.resize(640., 480.).unwrap();
    assert!(
        ui.viewport_rect().bottom()
            <= ui.size().height - STATUS_HEIGHT - ui.content_height() + 0.01
    );
    ui.resize(0., 0.).unwrap();
    assert!(ui
        .hits
        .iter()
        .all(|h| h.rect.size.width == 0.0 || h.rect.size.height == 0.0));
}

fn reveal(ui: &mut EditorDesktopUi, key: &str) -> [f32; 2] {
    for _ in 0..100 {
        let (full, clip) = ui.panels.geometry[key];
        if clip.size.width > 0.0 && clip.size.height > 0.0 {
            return [
                clip.left() + clip.size.width / 2.0,
                clip.top() + clip.size.height / 2.0,
            ];
        }
        let top = ui.size.height - STATUS_HEIGHT - ui.content_height() + ui.panel_tabs_height();
        let direction = if full.bottom() <= top { -1 } else { 1 };
        assert!(
            ui.scroll_at(10.0, top + 5.0, direction).unwrap(),
            "cannot reveal {key}: {full:?}"
        );
    }
    panic!("failed to reveal {key}");
}
fn ticks(ui: &EditorDesktopUi) -> Vec<u64> {
    ui.native_panel::<DesktopTimelinePanel>("timeline")
        .unwrap()
        .editor
        .timeline()
        .track(TrackId::from_raw(7))
        .unwrap()
        .keys_in(0..=u64::MAX)
        .map(Keyframe::tick)
        .collect()
}
fn timeline_ui() -> EditorDesktopUi {
    let mut ui = ui();
    ui.resize(1280.0, 1200.0).unwrap();
    ui.mount_native_panel("timeline", "Timeline", Box::new(timeline(|_| Ok(()))))
        .unwrap();
    ui
}

#[test]
fn native_key_capture_uses_layout_group_preview_and_one_release_transaction() {
    let mut ui = timeline_ui();
    let a = reveal(&mut ui, "timeline:key:7:1");
    ui.click(a[0], a[1], SelectionMode::Replace).unwrap();
    let b = reveal(&mut ui, "timeline:key:7:2");
    ui.click(b[0], b[1], SelectionMode::Add).unwrap();
    let a = reveal(&mut ui, "timeline:key:7:1");
    let token = ui
        .native_panel::<DesktopTimelinePanel>("timeline")
        .unwrap()
        .editor
        .snapshot_token();
    let scene = ui.controller().model_revision();
    assert!(ui
        .panel_pointer_press(a[0], a[1], SelectionMode::Replace)
        .unwrap());
    assert!(ui.panel_pointer_active());
    assert!(ui.has_uncommitted_edit());
    assert!(ui.switch_native_panel(Some("timeline")).is_err());
    assert!(!ui.key(DesktopKey::Undo).unwrap());
    ui.panel_pointer_move(a[0] + 40.0, a[1] - 400.0).unwrap();
    assert_eq!(ticks(&ui), [0, 100]);
    assert_eq!(
        ui.native_panel::<DesktopTimelinePanel>("timeline")
            .unwrap()
            .editor
            .snapshot_token(),
        token
    );
    assert!(ui.panels.geometry["timeline:key:7:1"].0.left() > a[0]);
    assert!(!ui.poll_native_panels(Instant::now()).unwrap());
    ui.panel_pointer_release(Some([a[0] + 80.0, a[1] - 400.0]))
        .unwrap();
    assert!(!ui.panel_pointer_active());
    assert_eq!(ticks(&ui), [30, 130]);
    assert!(ui.has_uncommitted_edit());
    ui.key(DesktopKey::Undo).unwrap();
    assert_eq!(ticks(&ui), [0, 100]);
    ui.key(DesktopKey::Redo).unwrap();
    assert_eq!(ticks(&ui), [30, 130]);
    assert_eq!(ui.controller().model_revision(), scene);
}

#[test]
fn native_key_cancel_invalid_release_resize_and_host_replacement_preserve_model() {
    let mut ui = timeline_ui();
    for cancellation in 0..5 {
        let a = reveal(&mut ui, "timeline:key:7:1");
        let token = ui
            .native_panel::<DesktopTimelinePanel>("timeline")
            .unwrap()
            .editor
            .snapshot_token();
        ui.panel_pointer_press(a[0], a[1], SelectionMode::Replace)
            .unwrap();
        ui.panel_pointer_move(a[0] + 40.0, a[1]).unwrap();
        match cancellation {
            0 => {
                ui.key(DesktopKey::Escape).unwrap();
            }
            1 => {
                assert!(ui.panel_pointer_release(Some([f32::NAN, a[1]])).is_err());
            }
            2 => {
                ui.panel_pointer_release(None).unwrap();
            }
            3 => {
                ui.resize(1279.0, 1200.0).unwrap();
            }
            _ => {
                // The host mutator cancels capture before exposing replacement access.
                ui.native_panel_mut::<DesktopTimelinePanel>("timeline")
                    .unwrap();
                ui.refresh().unwrap();
            }
        }
        assert!(!ui.panel_pointer_active());
        assert!(!ui.has_uncommitted_edit());
        assert_eq!(
            ui.native_panel::<DesktopTimelinePanel>("timeline")
                .unwrap()
                .editor
                .snapshot_token(),
            token
        );
        assert_eq!(ticks(&ui), [0, 100]);
    }
}

#[test]
fn native_tangent_uses_curve_origin_preview_undo_and_invalid_final_event_cancels() {
    let mut ui = timeline_ui();
    let b = reveal(&mut ui, "timeline:key:7:2");
    ui.click(b[0], b[1], SelectionMode::Replace).unwrap();
    let stem = reveal(&mut ui, "timeline:tangent:outgoing:stem:4");
    assert!(!ui
        .panel_pointer_press(stem[0], stem[1], SelectionMode::Replace)
        .unwrap());
    assert!(!ui.panel_pointer_active());
    let p = reveal(&mut ui, "timeline:tangent:outgoing");
    let original = ui
        .native_panel::<DesktopTimelinePanel>("timeline")
        .unwrap()
        .editor
        .snapshot_token();
    ui.panel_pointer_press(p[0], p[1], SelectionMode::Replace)
        .unwrap();
    ui.panel_pointer_move(p[0] + 10.0, p[1] - 10.0).unwrap();
    assert_eq!(
        ui.native_panel::<DesktopTimelinePanel>("timeline")
            .unwrap()
            .editor
            .snapshot_token(),
        original
    );
    ui.panel_pointer_release(Some([p[0] + 10.0, p[1] - 10.0]))
        .unwrap();
    let slope = |ui: &EditorDesktopUi| {
        ui.native_panel::<DesktopTimelinePanel>("timeline")
            .unwrap()
            .editor
            .timeline()
            .track(TrackId::from_raw(7))
            .unwrap()
            .keys_in(0..=u64::MAX)
            .last()
            .unwrap()
            .outgoing()[0]
    };
    let changed = slope(&ui);
    assert!(changed > 0.0);
    ui.key(DesktopKey::Undo).unwrap();
    assert_eq!(slope(&ui), 0.0);
    ui.key(DesktopKey::Redo).unwrap();
    assert_eq!(slope(&ui), changed);
    let p = reveal(&mut ui, "timeline:tangent:outgoing");
    let token = ui
        .native_panel::<DesktopTimelinePanel>("timeline")
        .unwrap()
        .editor
        .snapshot_token();
    ui.panel_pointer_press(p[0], p[1], SelectionMode::Replace)
        .unwrap();
    ui.panel_pointer_move(p[0] + 10.0, p[1] - 10.0).unwrap();
    assert!(ui
        .panel_pointer_release(Some([p[0] - 100.0, p[1]]))
        .is_err());
    assert!(!ui.panel_pointer_active());
    assert_eq!(
        ui.native_panel::<DesktopTimelinePanel>("timeline")
            .unwrap()
            .editor
            .snapshot_token(),
        token
    );
}

#[test]
fn backwards_clock_is_visible_fenced_and_stops_wakeup_until_host_retry() {
    let mut panel = timeline(|_| Ok(()));
    panel
        .controller
        .apply(&mut panel.editor, TimelinePanelAction::Play)
        .unwrap();
    let start = Instant::now();
    panel.poll(start).unwrap();
    assert!(panel.poll(start - Duration::from_millis(1)).is_err());
    assert!(panel.fenced());
    assert!(panel.next_wake().is_none());
    assert!(!panel.poll(start + Duration::from_millis(1)).unwrap());
    panel.retry();
    panel.poll(start + Duration::from_millis(20)).unwrap();
    assert!(!panel.fenced());
}
