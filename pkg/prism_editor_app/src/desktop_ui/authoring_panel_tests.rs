//! Native retained geometry, real kernel commands and actual filesystem receipts.
use super::*;
use alloc::{rc::Rc, sync::Arc};
use prism_editor_graph::*;
use prism_editor_timeline::*;
use prism_editor_ui::{
    GraphNodeTemplate, GraphPanelAction, GraphPanelViewport, TimelinePanelAction,
    TimelinePanelController,
};
use std::{
    cell::Cell,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

fn template() -> GraphNodeTemplate {
    GraphNodeTemplate {
        name: "Host operator".into(),
        payload: vec![31],
        ports: vec![
            PortSpec::input("in", PortType::named("host.scalar")),
            PortSpec::output("out", PortType::named("host.scalar")),
        ],
    }
}
fn graph() -> DesktopGraphPanel {
    DesktopGraphPanel::new(
        GraphEditor::new(Default::default()).unwrap(),
        vec![template()],
        GraphPanelViewport {
            size: [600., 140.],
            ..Default::default()
        },
        Default::default(),
    )
    .unwrap()
}
fn timeline() -> DesktopTimelinePanel {
    let mut doc = Timeline::default();
    doc.add_track(Track::new(TrackId::from_raw(1), "Host track", 1).unwrap())
        .unwrap();
    DesktopTimelinePanel::new(
        TimelineEditor::new(doc, Default::default()).unwrap(),
        TimelinePanelController::default(),
        |_| Ok(()),
    )
}
fn ui() -> EditorDesktopUi {
    let mut ui = EditorDesktopUi::new(EditorController::new()).unwrap();
    ui.resize(1400., 900.).unwrap();
    ui
}
fn click(ui: &mut EditorDesktopUi, key: &str) {
    let rect = ui
        .hits
        .iter()
        .find(|h| h.target == key && h.rect.size.width > 0. && h.rect.size.height > 0.)
        .unwrap_or_else(|| panic!("not visible: {key}"))
        .rect;
    ui.click(
        rect.left() + rect.size.width / 2.,
        rect.top() + rect.size.height / 2.,
        SelectionMode::Replace,
    )
    .unwrap();
}
fn canvas(ui: &EditorDesktopUi) -> [f32; 2] {
    let (rect, clip) = ui.panels.geometry.get("graph:canvas").unwrap();
    assert!(
        clip.size.width > 0. && clip.size.height >= 100.,
        "canvas clipped: {rect:?} {clip:?}"
    );
    [rect.left(), rect.top()]
}
fn add_node(panel: &mut DesktopGraphPanel) {
    panel.controller.apply(GraphPanelAction::Create(0)).unwrap();
}
#[test]
fn graph_native_geometry_preview_release_undo_redo_and_cancel() {
    let mut ui = ui();
    ui.mount_native_panel("graph", "Graph", Box::new(graph()))
        .unwrap();
    click(&mut ui, "graph:create:0");
    let panel = ui.native_panel::<DesktopGraphPanel>("graph").unwrap();
    let node = panel.view.as_ref().unwrap().nodes()[0].node;
    let local = panel.view.as_ref().unwrap().nodes()[0].rect;
    let before = panel.controller.editor().position(node).unwrap();
    let depth = panel.controller.editor().undo_depth();
    let origin = canvas(&ui);
    let point = [origin[0] + local[0] + 30., origin[1] + local[1] + 10.];
    assert!(ui
        .panel_pointer_press(point[0], point[1], SelectionMode::Replace)
        .unwrap());
    ui.panel_pointer_move(point[0] + 25., point[1] + 5.)
        .unwrap();
    assert_eq!(
        ui.native_panel::<DesktopGraphPanel>("graph")
            .unwrap()
            .controller
            .editor()
            .position(node),
        Some(before)
    );
    assert_eq!(
        ui.native_panel::<DesktopGraphPanel>("graph")
            .unwrap()
            .controller
            .editor()
            .undo_depth(),
        depth
    );
    ui.panel_pointer_release(Some([point[0] + 35., point[1] + 8.]))
        .unwrap();
    assert_eq!(
        ui.native_panel::<DesktopGraphPanel>("graph")
            .unwrap()
            .controller
            .editor()
            .position(node),
        Some(GraphPoint {
            x: before.x + 35,
            y: before.y + 8
        })
    );
    assert_eq!(
        ui.native_panel::<DesktopGraphPanel>("graph")
            .unwrap()
            .controller
            .editor()
            .undo_depth(),
        depth + 1
    );
    ui.key(DesktopKey::Undo).unwrap();
    assert_eq!(
        ui.native_panel::<DesktopGraphPanel>("graph")
            .unwrap()
            .controller
            .editor()
            .position(node),
        Some(before)
    );
    ui.key(DesktopKey::Redo).unwrap();
    ui.panel_pointer_press(point[0] + 35., point[1] + 8., SelectionMode::Replace)
        .unwrap();
    ui.panel_pointer_move(point[0] + 40., point[1] + 8.)
        .unwrap();
    ui.key(DesktopKey::Escape).unwrap();
    assert!(!ui.panel_pointer_active());
    let stable = ui
        .native_panel::<DesktopGraphPanel>("graph")
        .unwrap()
        .controller
        .editor()
        .snapshot_token();
    ui.panel_pointer_press(point[0] + 35., point[1] + 8., SelectionMode::Replace)
        .unwrap();
    assert!(ui
        .panel_pointer_release(Some([f32::NAN, point[1]]))
        .is_err());
    assert!(!ui.panel_pointer_active());
    assert_eq!(
        ui.native_panel::<DesktopGraphPanel>("graph")
            .unwrap()
            .controller
            .editor()
            .snapshot_token(),
        stable
    );
    assert!(ui.unmount_native_panel("graph").is_err());
}
#[test]
fn graph_native_pan_zoom_and_host_replacement_cancel_preserve_model() {
    let mut ui = ui();
    ui.mount_native_panel("graph", "Graph", Box::new(graph()))
        .unwrap();
    let origin = canvas(&ui);
    let p = [origin[0] + 300., origin[1] + 70.];
    let token = ui
        .native_panel::<DesktopGraphPanel>("graph")
        .unwrap()
        .controller
        .editor()
        .snapshot_token();
    assert!(ui.panel_pan_press(p[0], p[1]).unwrap());
    ui.panel_pointer_move(p[0] + 20., p[1] + 10.).unwrap();
    ui.panel_pointer_release(Some([p[0] + 20., p[1] + 10.]))
        .unwrap();
    assert_eq!(
        ui.native_panel::<DesktopGraphPanel>("graph")
            .unwrap()
            .controller
            .viewport()
            .origin,
        [-20., -10.]
    );
    ui.scroll_at(p[0], p[1], -1).unwrap();
    assert!(
        ui.native_panel::<DesktopGraphPanel>("graph")
            .unwrap()
            .controller
            .viewport()
            .zoom
            > 1.
    );
    ui.panel_pan_press(p[0], p[1]).unwrap();
    ui.native_panel_mut::<DesktopGraphPanel>("graph").unwrap();
    assert!(!ui.panel_pointer_active());
    assert_eq!(
        ui.native_panel::<DesktopGraphPanel>("graph")
            .unwrap()
            .controller
            .editor()
            .snapshot_token(),
        token
    );
    ui.refresh().unwrap();
    assert!(!ui.has_uncommitted_edit());
}
#[test]
fn graph_native_typed_port_connection_uses_final_hit_and_one_inverse() {
    let mut panel = graph();
    add_node(&mut panel);
    add_node(&mut panel);
    let nodes = panel.controller.editor().graph().node_ids();
    panel
        .controller
        .apply(GraphPanelAction::Command(GraphCommand::Edit(vec![
            GraphEdit::MoveNode {
                node: nodes[0],
                position: GraphPoint { x: 20, y: 10 },
            },
            GraphEdit::MoveNode {
                node: nodes[1],
                position: GraphPoint { x: 340, y: 10 },
            },
        ])))
        .unwrap();
    let before = panel.controller.editor().snapshot_token();
    let history = panel.controller.editor().undo_depth();
    let mut ui = ui();
    ui.mount_native_panel("graph", "Graph", Box::new(panel))
        .unwrap();
    let origin = canvas(&ui);
    let view = ui
        .native_panel::<DesktopGraphPanel>("graph")
        .unwrap()
        .view
        .as_ref()
        .unwrap();
    let point = |node, direction| {
        let p = view
            .ports()
            .iter()
            .find(|p| p.node == node && p.direction == direction)
            .unwrap();
        assert_eq!(p.type_key, "host.scalar");
        [
            origin[0] + p.rect[0] + p.rect[2] / 2.,
            origin[1] + p.rect[1] + p.rect[3] / 2.,
        ]
    };
    let from = point(nodes[0], PortDirection::Output);
    let to = point(nodes[1], PortDirection::Input);
    assert!(ui
        .panel_pointer_press(from[0], from[1], SelectionMode::Replace)
        .unwrap());
    ui.panel_pointer_move(to[0], to[1]).unwrap();
    assert_eq!(
        ui.native_panel::<DesktopGraphPanel>("graph")
            .unwrap()
            .controller
            .editor()
            .snapshot_token(),
        before
    );
    assert!(ui.panel_document_command(true).is_err());
    ui.panel_pointer_release(Some(to)).unwrap();
    let editor = ui
        .native_panel::<DesktopGraphPanel>("graph")
        .unwrap()
        .controller
        .editor();
    assert_eq!(editor.graph().edge_count(), 1);
    assert_eq!(editor.undo_depth(), history + 1);
    ui.key(DesktopKey::Undo).unwrap();
    assert_eq!(
        ui.native_panel::<DesktopGraphPanel>("graph")
            .unwrap()
            .controller
            .editor()
            .graph()
            .edge_count(),
        0
    );
    ui.key(DesktopKey::Redo).unwrap();
    assert_eq!(
        ui.native_panel::<DesktopGraphPanel>("graph")
            .unwrap()
            .controller
            .editor()
            .graph()
            .edge_count(),
        1
    );
}
struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "prism-native-authoring-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).unwrap();
        Self(dir)
    }
    fn file(&self) -> PathBuf {
        self.0.join("document.bin")
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
            "real storage did not finish"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}
fn settle_graph(p: &mut DesktopGraphPanel) -> Result<(), String> {
    let mut error = None;
    wait(|| {
        if let Err(e) = p.poll(Instant::now()) {
            error = Some(e);
        }
        p.file.as_ref().unwrap().next_wake().is_none()
    });
    error.map_or(Ok(()), Err)
}
fn settle_timeline(p: &mut DesktopTimelinePanel) -> Result<(), String> {
    let mut error = None;
    wait(|| {
        if let Err(e) = p.poll(Instant::now()) {
            error = Some(e);
        }
        p.file.as_ref().unwrap().next_wake().is_none()
    });
    error.map_or(Ok(()), Err)
}
#[test]
fn graph_native_shortcut_real_save_reload_candidate_and_stale_receipt() {
    let dir = Directory::new();
    let mut p = graph();
    add_node(&mut p);
    p.mount_storage(
        dir.file(),
        Default::default(),
        GraphFileRevision::Missing,
        Duration::from_secs(5),
    )
    .unwrap();
    let mut ui = ui();
    ui.mount_native_panel("graph", "Graph", Box::new(p))
        .unwrap();
    assert!(ui.panel_document_command(true).unwrap());
    assert!(ui.has_uncommitted_edit());
    wait(|| {
        ui.poll_native_panels(Instant::now()).unwrap();
        ui.native_panel::<DesktopGraphPanel>("graph")
            .unwrap()
            .storage_status()
            == Some("Saved durably")
    });
    ui.refresh().unwrap();
    assert!(!ui.has_uncommitted_edit());
    let saved = std::fs::read(dir.file()).unwrap();
    assert_eq!(
        GraphDocument::decode(&saved, Default::default(), 8 * 1024 * 1024)
            .unwrap()
            .graph()
            .node_count(),
        1
    );
    let p = ui.native_panel_mut::<DesktopGraphPanel>("graph").unwrap();
    add_node(p);
    p.save_file().unwrap();
    add_node(p);
    settle_graph(p).unwrap();
    assert!(
        p.fenced(),
        "older durable receipt cannot confirm a newer edit"
    );
    p.load_file().unwrap();
    settle_graph(p).unwrap();
    let started = p.controller.editor().snapshot_token();
    add_node(p);
    let newer = p.controller.editor().snapshot_token();
    assert!(p.adopt_loaded(&started).is_err());
    assert!(p.adopt_loaded(&newer).is_err());
    assert_eq!(p.controller.editor().graph().node_count(), 4);
    p.discard_loaded();
    p.load_file().unwrap();
    settle_graph(p).unwrap();
    let approved = p.controller.editor().snapshot_token();
    p.adopt_loaded(&approved).unwrap();
    assert_eq!(p.controller.editor().graph().node_count(), 2);
    assert_eq!(p.controller.editor().undo_depth(), 0);
    assert!(!p.fenced());
}
#[test]
fn graph_storage_conflict_and_cancel_keep_live_model_and_disk() {
    let dir = Directory::new();
    let mut p = graph();
    add_node(&mut p);
    p.mount_storage(
        dir.file(),
        Default::default(),
        GraphFileRevision::Missing,
        Duration::from_secs(5),
    )
    .unwrap();
    p.save_file().unwrap();
    settle_graph(&mut p).unwrap();
    let original = std::fs::read(dir.file()).unwrap();
    std::fs::write(dir.file(), b"external owner").unwrap();
    p.save_file().unwrap();
    assert!(settle_graph(&mut p).is_err());
    assert!(p.fenced());
    assert_eq!(std::fs::read(dir.file()).unwrap(), b"external owner");
    assert_eq!(p.controller.editor().graph().node_count(), 1);
    std::fs::write(dir.file(), &original).unwrap();
    p.load_file().unwrap();
    let outcome = p.cancel_file();
    assert!(matches!(
        outcome,
        DesktopPanelCancelOutcome::Cancelled | DesktopPanelCancelOutcome::TooLate
    ));
    assert!(p.fenced());
    settle_graph(&mut p).unwrap();
    p.discard_loaded();
    assert_eq!(std::fs::read(dir.file()).unwrap(), original);
}
#[test]
fn timeline_file_rate_is_part_of_exact_snapshot_and_load_replaces_only_on_approval() {
    let dir = Directory::new();
    let mut p = timeline();
    p.mount_storage(
        dir.file(),
        Default::default(),
        TimelineFileRevision::Missing,
        Duration::from_secs(5),
    )
    .unwrap();
    let token = p.editor.snapshot_token();
    let old_rate = p.controller.playback().config().ticks_per_second;
    p.save_file().unwrap();
    let mut config = p.controller.playback().config();
    config.ticks_per_second = 120;
    p.controller
        .apply(&mut p.editor, TimelinePanelAction::PlaybackConfig(config))
        .unwrap();
    assert_eq!(p.editor.snapshot_token(), token);
    settle_timeline(&mut p).unwrap();
    assert!(p.fenced());
    assert!(!p.acknowledge_saved(&token, old_rate));
    p.save_file().unwrap();
    settle_timeline(&mut p).unwrap();
    assert!(!p.fenced());
    let doc = TimelineDocument::decode(
        &std::fs::read(dir.file()).unwrap(),
        Default::default(),
        8 * 1024 * 1024,
    )
    .unwrap();
    assert_eq!(doc.ticks_per_second, 120);
    assert_eq!(doc.timeline.track_count(), 1);
    p.load_file().unwrap();
    settle_timeline(&mut p).unwrap();
    assert!(p.fenced());
    config.ticks_per_second = 90;
    p.controller
        .apply(&mut p.editor, TimelinePanelAction::PlaybackConfig(config))
        .unwrap();
    assert!(
        p.adopt_loaded(&token).is_err(),
        "TPS-only change invalidates load-start evidence"
    );
    p.discard_loaded();
    p.load_file().unwrap();
    settle_timeline(&mut p).unwrap();
    p.adopt_loaded(&token).unwrap();
    assert_eq!(p.controller.playback().config().ticks_per_second, 120);
    assert_eq!(p.controller.playback().status(), PlaybackStatus::Paused);
    assert!(!p.fenced());
}

#[test]
fn prefab_actual_bound_selection_revert_undo_redo_and_exact_save_fence() {
    use prism_editor_core::prefab::*;
    use prism_editor_ui::{PrefabPanelController, PrefabPanelLimits};
    let owner = PrefabOwner::Asset(PrefabId::from_raw(1));
    let property = PrefabProperty {
        entity: PrefabEntityPath::local(PrefabEntityId::from_raw(1)),
        component: "host/schema@1".into(),
        fields: vec!["value".into()],
    };
    let mut layer = PrefabLayer::default();
    layer.entities.insert(
        PrefabEntityId::from_raw(1),
        PrefabEntity {
            parent: None,
            components: BTreeMap::from([(
                "host/schema@1".into(),
                Arc::new(PrefabValue::Object(BTreeMap::from([(
                    "value".into(),
                    PrefabValue::Signed(1),
                )]))),
            )]),
        },
    );
    let mut store = PrefabStore::new(Default::default()).unwrap();
    store
        .transact(
            store.revision(),
            &[
                PrefabEdit::InsertAsset(Arc::new(PrefabAsset {
                    id: PrefabId::from_raw(1),
                    base: None,
                    layer,
                })),
                PrefabEdit::Override {
                    owner,
                    property: property.clone(),
                    value: PrefabOverride::Set(Arc::new(PrefabValue::Signed(9))),
                },
            ],
        )
        .unwrap();
    let controller =
        PrefabPanelController::new(store, owner, PrefabPanelLimits::default()).unwrap();
    let original = controller.snapshot();
    let mut ui = ui();
    ui.mount_native_panel(
        "prefab",
        "Prefab",
        Box::new(DesktopPrefabPanel::new(controller)),
    )
    .unwrap();
    let row = ui
        .native_panel::<DesktopPrefabPanel>("prefab")
        .unwrap()
        .controller
        .build()
        .unwrap()
        .rows()[0]
        .key
        .clone();
    click(&mut ui, &row);
    click(&mut ui, "prefab:revert");
    assert!(ui.has_uncommitted_edit());
    assert_eq!(
        ui.native_panel::<DesktopPrefabPanel>("prefab")
            .unwrap()
            .controller
            .store()
            .compose(owner)
            .unwrap()
            .value(&property),
        Some(&PrefabValue::Signed(1))
    );
    ui.key(DesktopKey::Undo).unwrap();
    assert_eq!(
        ui.native_panel::<DesktopPrefabPanel>("prefab")
            .unwrap()
            .controller
            .store()
            .compose(owner)
            .unwrap()
            .value(&property),
        Some(&PrefabValue::Signed(9))
    );
    ui.key(DesktopKey::Redo).unwrap();
    let p = ui.native_panel_mut::<DesktopPrefabPanel>("prefab").unwrap();
    assert!(!p.acknowledge_saved(&original));
    let exact = p.controller.snapshot();
    assert!(p.acknowledge_saved(&exact));
    ui.refresh().unwrap();
    assert!(!ui.has_uncommitted_edit());
    ui.unmount_native_panel("prefab").unwrap();
}

#[test]
fn prefab_native_review_confirm_and_undo_apply_real_base_and_pin() {
    use prism_editor_core::prefab::*;
    use prism_editor_ui::{PrefabPanelController, PrefabPanelLimits};
    let base = PrefabId::from_raw(1);
    let child = PrefabId::from_raw(2);
    let owner = PrefabOwner::Asset(child);
    let property = PrefabProperty {
        entity: PrefabEntityPath::local(PrefabEntityId::from_raw(1)),
        component: "host/schema@1".into(),
        fields: vec!["value".into()],
    };
    let mut layer = PrefabLayer::default();
    layer.entities.insert(
        PrefabEntityId::from_raw(1),
        PrefabEntity {
            parent: None,
            components: BTreeMap::from([(
                property.component.clone(),
                Arc::new(PrefabValue::Object(BTreeMap::from([(
                    "value".into(),
                    PrefabValue::Signed(1),
                )]))),
            )]),
        },
    );
    let mut store = PrefabStore::new(Default::default()).unwrap();
    store
        .transact(
            store.revision(),
            &[
                PrefabEdit::InsertAsset(Arc::new(PrefabAsset {
                    id: base,
                    base: None,
                    layer,
                })),
                PrefabEdit::InsertAsset(Arc::new(PrefabAsset {
                    id: child,
                    base: Some(base),
                    layer: PrefabLayer::default(),
                })),
                PrefabEdit::Override {
                    owner: PrefabOwner::Asset(base),
                    property: property.clone(),
                    value: PrefabOverride::Set(Arc::new(PrefabValue::Signed(4))),
                },
                PrefabEdit::Override {
                    owner,
                    property: property.clone(),
                    value: PrefabOverride::Set(Arc::new(PrefabValue::Signed(9))),
                },
            ],
        )
        .unwrap();
    let mut controller =
        PrefabPanelController::new(store, owner, PrefabPanelLimits::default()).unwrap();
    controller
        .configure_targets(&controller.snapshot(), Some(base), vec![])
        .unwrap();
    let token = controller.snapshot();
    let bytes = controller.store().encode().unwrap();
    let row = controller.build().unwrap().rows()[0].key.clone();
    let mut ui = ui();
    ui.mount_native_panel(
        "prefab",
        "Prefab",
        Box::new(DesktopPrefabPanel::new(controller)),
    )
    .unwrap();
    click(&mut ui, &row);
    click(&mut ui, "prefab:review-apply");
    let panel = ui.native_panel::<DesktopPrefabPanel>("prefab").unwrap();
    assert_eq!(panel.controller.store().encode().unwrap(), bytes);
    assert_eq!(panel.controller.history_depths(), (0, 0));
    assert_eq!(
        panel.view.as_ref().unwrap().apply_review().unwrap().len(),
        1
    );
    assert!(ui.has_uncommitted_edit());
    assert!(ui.unmount_native_panel("prefab").is_err());
    click(&mut ui, "prefab:confirm-apply");
    let panel = ui.native_panel::<DesktopPrefabPanel>("prefab").unwrap();
    assert_eq!(panel.controller.history_depths(), (1, 0));
    assert!(panel.view.as_ref().unwrap().apply_review().is_none());
    assert_eq!(
        panel
            .controller
            .store()
            .compose(PrefabOwner::Asset(base))
            .unwrap()
            .value(&property),
        Some(&PrefabValue::Signed(9))
    );
    ui.key(DesktopKey::Undo).unwrap();
    let panel = ui.native_panel::<DesktopPrefabPanel>("prefab").unwrap();
    assert_eq!(panel.controller.history_depths(), (0, 1));
    for (restored_owner, expected) in [(PrefabOwner::Asset(base), 4), (owner, 9)] {
        assert_eq!(
            panel
                .controller
                .store()
                .compose(restored_owner)
                .unwrap()
                .value(&property),
            Some(&PrefabValue::Signed(expected))
        );
    }
    // Undo restores values through a new publication revision.
    assert!(!panel.controller.matches_snapshot(&token));
    ui.key(DesktopKey::Redo).unwrap();
    assert_eq!(
        ui.native_panel::<DesktopPrefabPanel>("prefab")
            .unwrap()
            .controller
            .history_depths(),
        (1, 0)
    );
}

fn capture() -> prism_editor_profiler::GraphCaptureReport {
    use prism_render_architecture::{frame_graph::QueueClass, render_fabric::*};
    static PASSES: [PassTemplate; 1] = [PassTemplate {
        id: PassKey(3),
        queue: QueueClass::Compute,
        accesses: &[],
        depends_on: &[],
    }];
    static VARIANTS: [FeatureVariant; 1] = [FeatureVariant {
        id: VariantId(1),
        rank: 0,
        min_quality: RenderQualityTier::Low,
        required_caps: CapabilitySet::EMPTY,
        requires_services_extra: ServiceSet::EMPTY,
        passes: &PASSES,
        shader_permutation: None,
    }];
    static FEATURES: [FeatureDefinition; 1] = [FeatureDefinition::subsystem(
        FeatureId(77),
        RenderDomain::Surface,
        ServiceSet::EMPTY,
        &VARIANTS,
    )];
    static ROOTS: [FeatureId; 1] = [FeatureId(77)];
    let catalog = FeatureCatalog::new(&FEATURES, &[]).unwrap();
    let roots = FrozenRoots::from_build_profile(&catalog, &ROOTS, &[]).unwrap();
    let selection = FeatureSelection::resolve(
        &catalog,
        &roots,
        CapabilitySet::WEBGPU_BASELINE,
        RenderQualityTier::Low,
    )
    .unwrap();
    let compiled = ResourceTemplates::new(&catalog, &[])
        .unwrap()
        .compile(&selection, 13, &[], &[], GraphLimits::default())
        .unwrap();
    prism_editor_profiler::capture_feature_graph(&compiled, Default::default(), Default::default())
        .unwrap()
}

#[test]
fn framegraph_actual_compiler_capture_routing_and_failed_provider_retains_publication() {
    use prism_editor_ui::{FrameGraphPanelController, FrameGraphSelection};
    let report = Arc::new(capture());
    let source = report.clone();
    let phase = Rc::new(Cell::new(0));
    let producer = phase.clone();
    let mut ui = ui();
    ui.mount_native_panel(
        "framegraph",
        "Frame graph",
        Box::new(DesktopFrameGraphPanel::new(
            FrameGraphPanelController::default(),
            move |now| match producer.get() {
                0 => {
                    producer.set(1);
                    Ok((
                        Some(source.clone()),
                        DesktopPanelUpdate {
                            wake_at: Some(now + Duration::from_millis(1)),
                            ..Default::default()
                        },
                    ))
                }
                1 => Err("actual capture provider disconnected".into()),
                _ => Ok((None, Default::default())),
            },
        )),
    )
    .unwrap();
    ui.poll_native_panels(Instant::now()).unwrap();
    ui.refresh().unwrap();
    click(&mut ui, "frame-graph:pass:0");
    assert_eq!(
        ui.native_panel::<DesktopFrameGraphPanel>("framegraph")
            .unwrap()
            .controller
            .selection(),
        Some(FrameGraphSelection::Pass(0))
    );
    assert!(ui.poll_native_panels(Instant::now()).is_err());
    assert!(Arc::ptr_eq(
        ui.native_panel::<DesktopFrameGraphPanel>("framegraph")
            .unwrap()
            .controller
            .report()
            .unwrap(),
        &report
    ));
    assert!(ui.has_uncommitted_edit());
    assert!(ui.next_native_panel_wake().is_none());
    phase.set(2);
    ui.native_panel_mut::<DesktopFrameGraphPanel>("framegraph")
        .unwrap()
        .retry();
    ui.poll_native_panels(Instant::now()).unwrap();
    ui.refresh().unwrap();
    assert!(!ui.has_uncommitted_edit());
}
