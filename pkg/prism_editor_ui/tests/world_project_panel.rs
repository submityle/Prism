//! Real project files, sparse history, Loom bindings and generation fences.
use prism_asset::StableGuid;
use prism_ecs::partition::{cell::CellCoord, data_layer::DataLayerId};
use prism_editor_timeline::{TimelineCancelOutcome, TimelineFileRevision, TimelineSaveDurability};
use prism_editor_ui::{
    BoundWorldPanel, UiEvent, WorldPanelAction as A, WorldPanelController as Controller,
    WorldPanelError, WorldPanelLimits, WorldPanelViewport, WorldProjectIoState,
    WorldProjectPanelLimits,
};
use prism_editor_world::{
    project::{ProjectEdit, ProjectLayer, ProjectLimits, WorldProject, WorldProjectStorageService},
    CellResidency, DataLayer, EditorWorldData, EditorWorldLimits,
};
use prism_ui::{
    layout::{AvailableSpace, Size},
    BackendOp, Element, Key, RecordingBackend, Ui,
};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

fn coord(x: i32) -> CellCoord {
    CellCoord::new(x, 0, 0)
}
fn id(n: u32) -> DataLayerId {
    DataLayerId::new(n)
}
fn world(cells: usize, layers: u32) -> EditorWorldData {
    let mut w = EditorWorldData::new(
        StableGuid::from_path("world/project-panel"),
        EditorWorldLimits::new(cells.max(8), 4096, 32, 4, 8192).unwrap(),
    )
    .unwrap();
    for n in 0..layers {
        w.add_layer(DataLayer::new(id(n), format!("actual layer {n}")).unwrap())
            .unwrap();
    }
    for n in 0..cells {
        w.add_cell(coord(n as i32), [id(0)]).unwrap();
    }
    w
}
fn controller(cells: usize, layers: u32) -> Controller {
    Controller::new(
        world(cells, layers),
        StableGuid::from_path("panel author"),
        WorldPanelViewport {
            size: [128., 64.],
            ..Default::default()
        },
        WorldPanelLimits {
            visible_layers: 2,
            ..Default::default()
        },
    )
    .unwrap()
}
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "prism-world-project-ui-{}-{}",
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
impl Drop for Temp {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
fn bind(c: &mut Controller, dir: &Temp, expected: Option<TimelineFileRevision>) {
    let service = WorldProjectStorageService::new(
        dir.file(),
        c.project().identity(),
        1,
        ProjectLimits::default(),
    )
    .unwrap();
    c.bind_project_storage(service, expected, Duration::from_secs(5))
        .unwrap();
}
fn wait(c: &mut Controller) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while c.project_io_pending() {
        c.poll_project_storage().unwrap();
        assert!(
            Instant::now() < deadline,
            "real project worker did not finish"
        );
        std::thread::yield_now();
    }
}
fn save(c: &mut Controller) {
    c.apply(A::ProjectSave).unwrap();
    wait(c);
}
fn load(c: &mut Controller) {
    c.apply(A::ProjectLoad).unwrap();
    wait(c);
}
fn find<'a>(e: &'a Element, key: &str) -> Option<&'a Element> {
    if matches!(e.explicit_key(), Some(Key::Str(k)) if k == key) {
        return Some(e);
    }
    e.child_elements().iter().find_map(|e| find(e, key))
}
fn click(c: &mut Controller, ui: &mut Ui<RecordingBackend>, key: &str) -> BoundWorldPanel {
    let view = c.build();
    assert!(find(view.element(), key).is_some(), "painted control {key}");
    c.handle_event(&view, key, &UiEvent::Activate).unwrap();
    let next = c.build();
    ui.update(next.element());
    next
}
fn disk_project(dir: &Temp) -> WorldProject {
    WorldProject::decode(
        &std::fs::read(dir.file()).unwrap(),
        ProjectLimits::default(),
    )
    .unwrap()
}

#[test]
fn mounted_membership_policy_history_preserves_real_payload_and_visible_row_budget() {
    let mut c = controller(20_000, 17);
    c.apply(A::Select(coord(0))).unwrap();
    c.apply(A::Load).unwrap();
    let intent = c.take_intent().unwrap();
    let transition = c.begin_residency(&intent).unwrap();
    c.complete_load(transition, b"opaque resident bytes".to_vec())
        .unwrap();
    c.apply(A::AcquireLease).unwrap();
    c.replace_payload(coord(0), b"opaque unsaved bytes".to_vec())
        .unwrap();
    let first = c.build();
    let mut ui = Ui::new(RecordingBackend::new());
    ui.mount(first.element());
    assert_eq!(first.stats().visible_layers, 2);
    assert!(first.stats().has_more_layers);
    assert!(find(first.element(), "world:layer:2:member").is_none());
    let stale = first.clone();
    let view = click(&mut c, &mut ui, "world:layer:1:member");
    assert!(c
        .world()
        .model()
        .cell(coord(0))
        .unwrap()
        .layers()
        .any(|l| l == id(1)));
    assert!(c.project().cell_layers(coord(0)).unwrap().contains(&id(1)));
    assert_eq!(c.project_history().0, 1);
    assert!(
        c.project_history().2 < 2048,
        "sparse inverse independent of 20k cells"
    );
    assert!(matches!(
        c.handle_event(&stale, "world:layer:0:member", &UiEvent::Activate),
        Err(WorldPanelError::StaleView)
    ));
    ui.compute_layout(Size {
        width: AvailableSpace::Definite(1500.),
        height: AvailableSpace::Definite(1400.),
    });
    assert!(ui.backend().ops().iter().any(|op| matches!(op, BackendOp::SetLayout { size, .. } if size.width > 0. && size.height >= 24.)));
    assert!(view
        .bindings()
        .all(|(key, _)| find(view.element(), key).is_some()));
    click(&mut c, &mut ui, "world:project:undo");
    assert!(!c.project().cell_layers(coord(0)).unwrap().contains(&id(1)));
    click(&mut c, &mut ui, "world:project:redo");
    click(&mut c, &mut ui, "world:layer:0:visible");
    assert!(!c.world().model().is_cell_visible(coord(0)));
    click(&mut c, &mut ui, "world:project:undo");
    assert!(c.world().model().is_cell_visible(coord(0)));
    let next = click(&mut c, &mut ui, "world:layers:next");
    assert!(find(next.element(), "world:layer:2:member").is_some());
    assert!(find(next.element(), "world:layer:0:member").is_none());
    assert_eq!(next.stats().visible_layers, 2);
    assert_eq!(
        c.world().payload(coord(0)),
        Some(&b"opaque unsaved bytes"[..])
    );
    assert!(c.world().model().cell(coord(0)).unwrap().is_dirty());
    c.replace_payload(coord(0), b"same retained lease".to_vec())
        .unwrap();
}

#[test]
fn real_save_receipt_cannot_clean_newer_membership_and_conditional_revision_advances() {
    let dir = Temp::new();
    let mut c = controller(2, 2);
    bind(&mut c, &dir, Some(TimelineFileRevision::Missing));
    assert!(c.project_is_dirty());
    c.apply(A::ProjectSave).unwrap();
    assert!(c.project_io_pending());
    assert!(c.apply(A::ProjectSave).is_err());
    c.apply(A::LayerVisible {
        id: id(0),
        visible: false,
    })
    .unwrap();
    wait(&mut c);
    assert!(matches!(
        c.project_io_state(),
        WorldProjectIoState::Saved {
            current: false,
            durability: TimelineSaveDurability::Synced,
            ..
        }
    ));
    assert!(c.project_is_dirty());
    assert!(disk_project(&dir).layer(id(0)).unwrap().visible);
    save(&mut c);
    assert!(!c.project_is_dirty());
    assert!(!disk_project(&dir).layer(id(0)).unwrap().visible);
    assert!(matches!(
        c.project_io_state(),
        WorldProjectIoState::Saved { current: true, .. }
    ));
    // Undo/redo advance the exact snapshot identity even when values return.
    c.apply(A::ProjectUndo).unwrap();
    c.apply(A::ProjectRedo).unwrap();
    assert!(c.project_is_dirty());
    save(&mut c);
    assert!(!c.project_is_dirty());
}

#[test]
fn actual_load_candidate_is_separate_and_edit_then_undo_still_fences_adoption() {
    let dir = Temp::new();
    let mut c = controller(2, 2);
    bind(&mut c, &dir, Some(TimelineFileRevision::Missing));
    save(&mut c);
    c.apply(A::ProjectLoad).unwrap();
    c.apply(A::LayerEditable {
        id: id(1),
        editable: false,
    })
    .unwrap();
    c.apply(A::ProjectUndo).unwrap();
    wait(&mut c);
    assert!(c.project_candidate().is_some());
    assert!(!c.project_candidate_is_current());
    assert!(c.apply(A::ProjectAdoptDiscard).is_err());
    c.apply(A::ProjectDismiss).unwrap();
    load(&mut c);
    assert!(c.project_candidate_is_current());
    assert!(
        c.apply(A::ProjectAdopt).is_err(),
        "unsaved exact snapshot requires explicit discard"
    );
    c.apply(A::ProjectAdoptDiscard).unwrap();
    assert!(c.has_retired_world());
    assert!(!c.project_is_dirty());
    assert_eq!(c.project_history(), (0, 0, 0));
    let old = c.take_retired_world().unwrap();
    assert_eq!(old.model().cell_count(), 2);
    assert!(!c.has_retired_world());
}

#[test]
fn adoption_rejects_pending_residency_and_old_generation_cannot_complete_reused_domain_token() {
    let dir = Temp::new();
    let mut c = controller(2, 1);
    bind(&mut c, &dir, Some(TimelineFileRevision::Missing));
    save(&mut c);
    load(&mut c);
    c.apply(A::Select(coord(0))).unwrap();
    c.apply(A::Load).unwrap();
    assert!(c.apply(A::ProjectAdopt).is_err(), "queued intent");
    let intent = c.take_intent().unwrap();
    let old = c.begin_residency(&intent).unwrap();
    assert!(c.apply(A::ProjectAdopt).is_err(), "real in-flight load");
    c.fail_residency(old.clone()).unwrap();
    let view = c.build();
    c.apply(A::ProjectAdopt).unwrap();
    c.apply(A::Select(coord(0))).unwrap();
    assert!(
        c.apply(A::Load).is_err(),
        "host must acknowledge retired runtime"
    );
    let _retired = c.take_retired_world().unwrap();
    c.apply(A::Load).unwrap();
    let ticket = c.take_intent().unwrap();
    let new = c.begin_residency(&ticket).unwrap();
    assert_eq!(new.token(), old.token(), "fresh domain reuses numeric IDs");
    assert!(matches!(
        c.complete_load(old, b"stale bytes".to_vec()),
        Err(WorldPanelError::StaleIntent)
    ));
    assert!(matches!(
        c.handle_event(&view, "world:project:load", &UiEvent::Activate),
        Err(WorldPanelError::StaleView)
    ));
    c.complete_load(new, b"current bytes".to_vec()).unwrap();
    assert_eq!(c.world().payload(coord(0)), Some(&b"current bytes"[..]));
}

#[test]
fn candidate_adoption_never_discards_resident_dirty_bytes_or_a_live_local_lease() {
    let dir = Temp::new();
    let mut c = controller(2, 1);
    bind(&mut c, &dir, Some(TimelineFileRevision::Missing));
    save(&mut c);
    load(&mut c);
    c.apply(A::Select(coord(0))).unwrap();
    c.apply(A::AcquireLease).unwrap();
    assert!(c.apply(A::ProjectAdoptDiscard).is_err(), "live local lease");
    c.apply(A::ReleaseLease).unwrap();
    c.apply(A::Load).unwrap();
    let t = c.take_intent().unwrap();
    let transition = c.begin_residency(&t).unwrap();
    c.complete_load(transition, b"resident".to_vec()).unwrap();
    assert!(c.apply(A::ProjectAdoptDiscard).is_err(), "loaded consumers");
    c.apply(A::AcquireLease).unwrap();
    c.replace_payload(coord(0), b"dirty bytes".to_vec())
        .unwrap();
    assert!(c.apply(A::ProjectAdoptDiscard).is_err());
    assert_eq!(c.world().payload(coord(0)), Some(&b"dirty bytes"[..]));
    assert!(c.project_candidate().is_some());
}

#[test]
fn cancellation_retains_real_request_until_terminal_and_unknown_revision_blocks_save() {
    let dir = Temp::new();
    let mut c = controller(1, 1);
    bind(&mut c, &dir, Some(TimelineFileRevision::Missing));
    save(&mut c);
    c.apply(A::ProjectLoad).unwrap();
    assert_eq!(
        c.cancel_project_storage().unwrap(),
        TimelineCancelOutcome::Cancelled
    );
    assert!(c.project_io_pending());
    assert_eq!(c.project_io_state(), WorldProjectIoState::Cancelling);
    wait(&mut c);
    assert_eq!(c.project_io_state(), WorldProjectIoState::Cancelled);
    assert!(c.project_candidate().is_none());
    bind(&mut c, &dir, None);
    assert!(!c.project_can_save());
    assert!(c.apply(A::ProjectSave).is_err());
    load(&mut c);
    c.apply(A::ProjectAdopt).unwrap();
    c.take_retired_world().unwrap();
    save(&mut c);
    assert!(!c.project_is_dirty());
}

#[test]
fn external_file_conflict_and_malformed_bytes_never_clean_or_replace_live_membership() {
    let dir = Temp::new();
    let mut c = controller(1, 1);
    bind(&mut c, &dir, Some(TimelineFileRevision::Missing));
    save(&mut c);
    let mut external = disk_project(&dir);
    external
        .apply(
            external.revision(),
            &[ProjectEdit::Cell {
                cell: coord(1),
                layers: Some([id(0)].into()),
            }],
            ProjectLimits::default(),
            4096,
        )
        .unwrap();
    std::fs::write(
        dir.file(),
        external.encode(ProjectLimits::default()).unwrap(),
    )
    .unwrap();
    c.apply(A::LayerVisible {
        id: id(0),
        visible: false,
    })
    .unwrap();
    save(&mut c);
    assert!(matches!(
        c.project_io_state(),
        WorldProjectIoState::Failed(_)
    ));
    assert!(c.project_is_dirty());
    assert_eq!(disk_project(&dir).cells().len(), 2);
    std::fs::write(dir.file(), b"invalid project bytes").unwrap();
    let token = c.project().token();
    load(&mut c);
    assert!(matches!(
        c.project_io_state(),
        WorldProjectIoState::Failed(_)
    ));
    assert!(c.project_candidate().is_none());
    assert!(c.project().matches_token(&token));
}

#[test]
fn invalid_batch_and_history_budget_reject_without_partial_world_or_history_mutation() {
    let mut c = controller(3, 2);
    let token = c.project().token();
    let mut layer = c.project().layer(id(0)).unwrap().clone();
    layer.visible = false;
    assert!(c
        .edit_project(&[
            ProjectEdit::Layer {
                id: id(0),
                value: Some(layer)
            },
            ProjectEdit::Cell {
                cell: coord(0),
                layers: Some([id(999)].into())
            }
        ])
        .is_err());
    assert!(c.project().matches_token(&token));
    assert!(c.world().model().is_cell_visible(coord(0)));
    assert_eq!(c.project_history(), (0, 0, 0));
    let mut small = Controller::with_project_limits(
        world(3, 1),
        StableGuid::from_path("author"),
        WorldPanelViewport::default(),
        WorldPanelLimits::default(),
        WorldProjectPanelLimits {
            history_bytes: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(small
        .apply(A::LayerVisible {
            id: id(0),
            visible: false
        })
        .is_err());
    assert!(small.world().model().is_cell_visible(coord(0)));
    assert_eq!(small.project_history(), (0, 0, 0));
}

#[test]
fn structural_sparse_history_returns_old_world_and_roundtrips_real_new_membership() {
    let mut c = controller(2, 1);
    let new_layer = ProjectLayer {
        id: id(9),
        name: "real structural layer".into(),
        parent: Some(id(0)),
        visible: true,
        editable: true,
    };
    c.edit_project(&[
        ProjectEdit::Layer {
            id: id(9),
            value: Some(new_layer),
        },
        ProjectEdit::Cell {
            cell: coord(2),
            layers: Some([id(9)].into()),
        },
    ])
    .unwrap();
    assert_eq!(c.project_history().0, 1);
    assert!(c.world().model().cell(coord(2)).is_some());
    assert!(c.has_retired_world());
    assert!(c.apply(A::ProjectUndo).is_err());
    let old = c.take_retired_world().unwrap();
    assert!(old.model().cell(coord(2)).is_none());
    c.apply(A::ProjectUndo).unwrap();
    c.take_retired_world().unwrap();
    assert!(c.project().cell_layers(coord(2)).is_none());
    c.apply(A::ProjectRedo).unwrap();
    c.take_retired_world().unwrap();
    let bytes = c.project().encode(ProjectLimits::default()).unwrap();
    let decoded = WorldProject::decode(&bytes, ProjectLimits::default()).unwrap();
    assert_eq!(decoded.cell_layers(coord(2)), Some(&[id(9)].into()));
    assert_eq!(
        WorldProject::capture(c.world(), ProjectLimits::default())
            .unwrap()
            .cell_layers(coord(2)),
        decoded.cell_layers(coord(2))
    );
    assert_eq!(
        c.world().model().cell(coord(2)).unwrap().residency(),
        CellResidency::Unloaded
    );
}
