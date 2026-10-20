//! Actual Loom interactions, bounded geometry and real filesystem receipts.
use futures_lite::future::{block_on, poll_fn};
use prism_asset::StableGuid;
use prism_ecs::partition::{cell::CellCoord, data_layer::DataLayerId};
use prism_editor_ui::{
    BoundWorldPanel, UiEvent, WorldGridPlane, WorldPanelAction as A,
    WorldPanelController as Controller, WorldPanelError as Error, WorldPanelIntentKind,
    WorldPanelLease, WorldPanelLimits, WorldPanelViewport,
};
use prism_editor_world::{
    CellResidency, DataLayer, EditorWorldData, EditorWorldLimits, FileCellIoLimits,
    FileCellIoProvider, FileWorldTransactionProvider, IoBudget, WorldIoDriver,
    WorldTransactionEdit, WorldTransactionLimits,
};
use prism_ui::{
    layout::{AvailableSpace, Size},
    BackendOp, Element, Key, RecordingBackend, Ui,
};
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

fn cell(x: i32, y: i32) -> CellCoord {
    CellCoord::new(x, y, 0)
}
fn model(count: usize) -> EditorWorldData {
    let mut w = EditorWorldData::new(
        StableGuid::from_path("world/panel"),
        EditorWorldLimits::new(count.max(4), 4096, 4096, 4, 8192).unwrap(),
    )
    .unwrap();
    w.add_layer(DataLayer::new(DataLayerId::new(701), "project supplied layer").unwrap())
        .unwrap();
    for x in 0..count {
        w.add_cell(cell(x as i32, 0), [DataLayerId::new(701)])
            .unwrap();
    }
    w
}
fn controller(count: usize) -> Controller {
    Controller::new(
        model(count),
        StableGuid::from_path("user/panel"),
        WorldPanelViewport {
            size: [128., 64.],
            ..Default::default()
        },
        WorldPanelLimits::default(),
    )
    .unwrap()
}
fn find<'a>(e: &'a Element, k: &str) -> Option<&'a Element> {
    if matches!(e.explicit_key(),Some(Key::Str(s)) if s==k) {
        return Some(e);
    }
    e.child_elements().iter().find_map(|e| find(e, k))
}
fn send(
    c: &mut Controller,
    ui: &mut Ui<RecordingBackend>,
    view: BoundWorldPanel,
    key: &str,
) -> BoundWorldPanel {
    assert!(
        find(view.element(), key).is_some(),
        "actual painted key {key}"
    );
    c.handle_event(&view, key, &UiEvent::Activate).unwrap();
    let next = c.build();
    ui.update(next.element());
    for (key, _) in next.bindings() {
        assert!(find(next.element(), key).is_some());
    }
    next
}
fn load(c: &mut Controller, coord: CellCoord, bytes: &[u8]) {
    c.apply(A::Select(coord)).unwrap();
    c.apply(A::Load).unwrap();
    let i = c.take_intent().unwrap();
    let t = c.begin_residency(&i).unwrap();
    assert!(matches!(
        c.world().model().cell(coord).unwrap().residency(),
        CellResidency::Loading(_)
    ));
    c.complete_load(t, bytes.to_vec()).unwrap();
}
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "prism-world-panel-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
fn wait_idle(mut busy: impl FnMut() -> usize) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while busy() != 0 {
        assert!(Instant::now() < deadline, "worker failed to terminate");
        std::thread::yield_now();
    }
}

#[test]
fn mounted_controls_select_load_layers_lease_dirty_and_fence_old_views() {
    let mut c = controller(3);
    let view = c.build();
    let old = view.clone();
    let mut ui = Ui::new(RecordingBackend::new());
    ui.mount(view.element());
    let view = send(&mut c, &mut ui, view, "world:cell:0:0:0");
    assert!(matches!(
        c.handle_event(&old, "world:zoom-in", &UiEvent::Activate),
        Err(Error::StaleView)
    ));
    let _view = send(&mut c, &mut ui, view, "world:load");
    let ticket = c.take_intent().unwrap();
    assert_eq!(ticket.kind(), WorldPanelIntentKind::Load(cell(0, 0)));
    let token = c.begin_residency(&ticket).unwrap();
    c.complete_load(token, b"opaque domain payload".to_vec())
        .unwrap();
    let view = c.build();
    ui.update(view.element());
    let view = send(&mut c, &mut ui, view, "world:lease");
    assert!(matches!(c.lease(cell(0, 0)), WorldPanelLease::Owned { .. }));
    let view = send(&mut c, &mut ui, view, "world:layer:701:visible");
    assert!(!c.world().model().is_cell_visible(cell(0, 0)));
    let view = send(&mut c, &mut ui, view, "world:layer:701:editable");
    assert!(c
        .replace_payload(cell(0, 0), b"forbidden".to_vec())
        .is_err());
    let _view = send(&mut c, &mut ui, view, "world:layer:701:editable");
    c.replace_payload(cell(0, 0), b"updated opaque bytes".to_vec())
        .unwrap();
    let view = c.build();
    ui.update(view.element());
    assert!(view.cells()[0].dirty);
    assert!(c.apply(A::Unload).is_err());
    assert!(view.bindings().any(|(k, _)| k == "world:save"));
    c.set_time(60).unwrap();
    assert_eq!(c.lease(cell(0, 0)), WorldPanelLease::Expired);
    assert!(c.replace_payload(cell(0, 0), vec![]).is_err());
    assert!(c.set_time(59).is_err());
    assert!(ui.node_count() > view.cells().len());
}

#[test]
fn visible_grid_cost_is_independent_of_registered_world_size_and_geometry_is_real() {
    let small = controller(3);
    let mut large = controller(20_000);
    assert_eq!(
        small.build().stats().cell_lookups,
        large.build().stats().cell_lookups
    );
    large
        .apply(A::Viewport(WorldPanelViewport {
            size: [96., 80.],
            origin: [-0.5, -0.25],
            ..Default::default()
        }))
        .unwrap();
    let view = large.build();
    assert_eq!(view.cells().len(), 4);
    assert_eq!(view.cells()[0].cell, cell(-1, -1));
    assert_eq!(view.cells()[0].rect, [0., 0., 32., 16.]);
    let mut ui = Ui::new(RecordingBackend::new());
    ui.mount(view.element());
    ui.compute_layout(Size {
        width: AvailableSpace::Definite(1200.),
        height: AvailableSpace::Definite(900.),
    });
    for visual in view.cells() {
        assert!(
            ui.backend()
                .ops()
                .iter()
                .any(|op| matches!(op,BackendOp::SetLayout {location,size,..}
            if (location.x-visual.rect[0]).abs()<0.01 && (location.y-visual.rect[1]).abs()<0.01
            && (size.width-visual.rect[2]).abs()<0.01 && (size.height-visual.rect[3]).abs()<0.01)),
            "real solved box {visual:?}"
        );
    }
    let old = view.clone();
    let view = send(&mut large, &mut ui, view, "world:zoom-in");
    assert!(view.cells().len() <= old.cells().len());
    let view = send(&mut large, &mut ui, view, "world:right");
    assert_ne!(view.cells()[0].cell, old.cells()[0].cell);
    let view = send(&mut large, &mut ui, view, "world:xz");
    let _ = send(&mut large, &mut ui, view, "world:slice-next");
    assert_eq!(large.viewport().plane, WorldGridPlane::Xz);
    assert_eq!(large.viewport().slice, 1);
}

#[test]
fn nonfinite_overflow_edges_budget_and_foreign_views_fail_without_mutation() {
    let mut c = controller(2);
    let original = c.viewport();
    for v in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(c.apply(A::Zoom(v)).is_err());
        assert!(c.apply(A::Pan([v, 0.])).is_err());
        assert_eq!(c.viewport(), original);
        assert_eq!(original.hit_test([v, 0.]), None);
    }
    assert_eq!(original.hit_test([64., 0.]), Some(cell(1, 0)));
    assert_eq!(original.hit_test([128., 0.]), None);
    assert_eq!(original.hit_test([-0.01, 0.]), None);
    let mut invalid = original;
    invalid.origin = [f64::from(i32::MAX), 0.];
    invalid.size = [128., 64.];
    assert!(c.apply(A::Viewport(invalid)).is_err());
    invalid = original;
    invalid.size = [8192., 8192.];
    invalid.cell_pixels = 8.;
    assert!(matches!(
        c.apply(A::Viewport(invalid)),
        Err(Error::Capacity)
    ));
    let foreign = controller(2).build();
    assert!(matches!(
        c.handle_event(&foreign, "world:zoom-in", &UiEvent::Activate),
        Err(Error::StaleView)
    ));
    let view = c.build();
    assert!(c
        .handle_event(
            &view,
            "world:cell:1:0:0",
            &UiEvent::Commit {
                value: prism_editor_ui::ActionValue::float(f64::NAN)
            }
        )
        .is_err());
    assert!(c.selected().is_none());
    assert_eq!(c.viewport(), original);
}

#[test]
fn cancellation_unload_and_external_locks_use_real_domain_tokens() {
    let mut w = model(3);
    let other = StableGuid::from_path("other user");
    w.acquire_lock(cell(1, 0), other, 0, 100).unwrap();
    let mut c = Controller::new(
        w,
        StableGuid::from_path("owner"),
        WorldPanelViewport::default(),
        WorldPanelLimits::default(),
    )
    .unwrap();
    load(&mut c, cell(0, 0), b"first");
    load(&mut c, cell(1, 0), b"second");
    assert_eq!(
        c.world().model().cell(cell(0, 0)).unwrap().residency(),
        CellResidency::Loaded
    );
    assert_eq!(c.lease(cell(1, 0)), WorldPanelLease::Unobserved);
    assert!(c.apply(A::AcquireLease).is_err());
    c.apply(A::Unload).unwrap();
    let ticket = c.take_intent().unwrap();
    assert!(c.begin_residency(&ticket).is_err());
    assert_eq!(c.world().payload(cell(1, 0)), Some(&b"second"[..]));
    c.apply(A::Select(cell(2, 0))).unwrap();
    c.apply(A::Load).unwrap();
    let t = c.take_intent().unwrap();
    let token = c.begin_residency(&t).unwrap();
    c.fail_residency(token).unwrap();
    assert_eq!(
        c.world().model().cell(cell(2, 0)).unwrap().residency(),
        CellResidency::Unloaded
    );
    c.apply(A::Select(cell(0, 0))).unwrap();
    c.apply(A::Unload).unwrap();
    let t = c.take_intent().unwrap();
    let token = c.begin_residency(&t).unwrap();
    c.fail_residency(token).unwrap();
    c.apply(A::Unload).unwrap();
    let t = c.take_intent().unwrap();
    let token = c.begin_residency(&t).unwrap();
    c.complete_unload(token).unwrap();
    assert!(c.world().payload(cell(0, 0)).is_none());
}

#[test]
fn queue_leases_and_layer_controls_have_explicit_bounds_and_stale_tickets_fail() {
    let mut w = model(3);
    for id in 702..710 {
        w.add_layer(DataLayer::new(DataLayerId::new(id), format!("layer{id}")).unwrap())
            .unwrap();
    }
    w.set_cell_layers(cell(0, 0), (701..710).map(DataLayerId::new))
        .unwrap();
    let mut c = Controller::new(
        w,
        StableGuid::from_path("owner"),
        WorldPanelViewport::default(),
        WorldPanelLimits {
            intents: 1,
            leases: 1,
            visible_layers: 2,
            ..Default::default()
        },
    )
    .unwrap();
    c.apply(A::Select(cell(0, 0))).unwrap();
    let view = c.build();
    assert_eq!(view.stats().visible_layers, 2);
    assert!(view.stats().has_more_layers);
    c.apply(A::Load).unwrap();
    assert!(matches!(c.apply(A::Load), Err(Error::Capacity)));
    let stale = c.take_intent().unwrap();
    c.apply(A::LayerVisible {
        id: DataLayerId::new(701),
        visible: false,
    })
    .unwrap();
    assert!(matches!(c.begin_residency(&stale), Err(Error::StaleIntent)));
    c.apply(A::AcquireLease).unwrap();
    c.apply(A::Select(cell(1, 0))).unwrap();
    assert!(matches!(c.apply(A::AcquireLease), Err(Error::Capacity)));
    let mut other = controller(2);
    assert!(matches!(
        other.begin_residency(&stale),
        Err(Error::StaleIntent)
    ));
}

#[test]
fn painted_save_intent_round_trips_real_file_and_stale_completion_preserves_dirty() {
    let temp = Temp::new();
    let provider =
        FileCellIoProvider::new(&temp.0, FileCellIoLimits::new(4096, 4).unwrap()).unwrap();
    let mut driver = WorldIoDriver::new(provider.clone(), IoBudget::new(2, 8192).unwrap());
    let mut c = controller(2);
    load(&mut c, cell(0, 0), b"initial");
    c.apply(A::AcquireLease).unwrap();
    c.replace_payload(cell(0, 0), b"saved bytes".to_vec())
        .unwrap();
    let view = c.build();
    let mut ui = Ui::new(RecordingBackend::new());
    ui.mount(view.element());
    let _view = send(&mut c, &mut ui, view, "world:save");
    let ticket = c.take_intent().unwrap();
    let pending = c.prepare_save(&ticket).unwrap();
    driver
        .submit_save(pending.request().clone(), pending.payload().to_vec())
        .unwrap();
    let done = block_on(poll_fn(|cx| driver.poll(cx))).unwrap();
    assert!(c.apply_save_completion(&pending, done).unwrap());
    assert!(!c.world().model().cell(cell(0, 0)).unwrap().is_dirty());
    c.replace_payload(cell(0, 0), b"snapshot A".to_vec())
        .unwrap();
    c.apply(A::Save).unwrap();
    let ticket = c.take_intent().unwrap();
    let pending = c.prepare_save(&ticket).unwrap();
    driver
        .submit_save(pending.request().clone(), pending.payload().to_vec())
        .unwrap();
    let done = block_on(poll_fn(|cx| driver.poll(cx))).unwrap();
    c.replace_payload(cell(0, 0), b"newer B".to_vec()).unwrap();
    assert!(c.apply_save_completion(&pending, done).is_err());
    assert!(c.world().model().cell(cell(0, 0)).unwrap().is_dirty());
    assert_eq!(c.world().payload(cell(0, 0)), Some(&b"newer B"[..]));
    // Reload the actual durable A with a new controller and the same file identity.
    let mut read = controller(2);
    read.apply(A::Select(cell(0, 0))).unwrap();
    read.apply(A::Load).unwrap();
    let ticket = read.take_intent().unwrap();
    let token = read.begin_residency(&ticket).unwrap();
    let request = read.world().load_request(token.token()).unwrap();
    driver.submit_load(request).unwrap();
    let done = block_on(poll_fn(|cx| driver.poll(cx))).unwrap();
    assert!(read.apply_load_completion(&token, done).unwrap());
    assert_eq!(read.world().payload(cell(0, 0)), Some(&b"snapshot A"[..]));
    wait_idle(|| provider.outstanding_operations());
}

#[test]
fn actor_move_expired_receipt_retains_dirty_until_explicit_leased_snapshot_restore() {
    let temp = Temp::new();
    let provider = FileWorldTransactionProvider::new(
        &temp.0,
        WorldTransactionLimits::new(16, 4096, 16384, 2, 32768).unwrap(),
    )
    .unwrap();
    let actor = StableGuid::from_path("project-defined opaque actor");
    let mut world = model(2);
    world.register_actor(actor, cell(0, 0)).unwrap();
    let mut c = Controller::new(
        world,
        StableGuid::from_path("user/panel"),
        WorldPanelViewport {
            size: [128., 64.],
            ..Default::default()
        },
        WorldPanelLimits::default(),
    )
    .unwrap();
    load(&mut c, cell(0, 0), b"source original");
    c.apply(A::AcquireLease).unwrap();
    c.apply(A::CrossCellSource).unwrap();
    load(&mut c, cell(1, 0), b"destination original");
    c.apply(A::AcquireLease).unwrap();
    let edits = |owner, bytes: &'static [u8]| {
        [
            WorldTransactionEdit::Cell {
                cell: cell(0, 0),
                payload: Arc::from(&b"opaque source membership"[..]),
            },
            WorldTransactionEdit::Cell {
                cell: cell(1, 0),
                payload: Arc::from(&b"opaque destination membership"[..]),
            },
            WorldTransactionEdit::Actor {
                actor,
                cell: owner,
                payload: Arc::from(bytes),
            },
        ]
    };
    // Persist the registered actor in its current owner before moving it.
    let snapshot = block_on(provider.load_snapshot(Duration::from_secs(5)).unwrap()).unwrap();
    c.apply(A::CrossCell).unwrap();
    let ticket = c.take_intent().unwrap();
    let pending = c
        .prepare_transaction(
            &ticket,
            StableGuid::from_path("initial actor publication"),
            &snapshot,
            &edits(cell(0, 0), b"actor original"),
            &provider,
        )
        .unwrap();
    let receipt = block_on(
        provider
            .commit(pending.request().clone(), Duration::from_secs(5))
            .unwrap(),
    )
    .unwrap();
    c.commit_transaction(&pending, &receipt).unwrap();
    assert_eq!(c.world().actor_payload(actor), Some(&b"actor original"[..]));
    assert_eq!(c.project().actor_cell(actor), Some(cell(0, 0)));
    // Sparse policy history must survive unrelated real OFPA owner changes.
    c.apply(A::LayerVisible {
        id: DataLayerId::new(701),
        visible: false,
    })
    .unwrap();

    let snapshot = block_on(provider.load_snapshot(Duration::from_secs(5)).unwrap()).unwrap();
    c.apply(A::CrossCell).unwrap();
    let ticket = c.take_intent().unwrap();
    let pending = c
        .prepare_transaction(
            &ticket,
            StableGuid::from_path("actor move publication"),
            &snapshot,
            &edits(cell(1, 0), b"actor moved"),
            &provider,
        )
        .unwrap();
    let receipt = block_on(
        provider
            .commit(pending.request().clone(), Duration::from_secs(5))
            .unwrap(),
    )
    .unwrap();
    c.set_time(60).unwrap();
    assert!(c.commit_transaction(&pending, &receipt).is_err());
    assert_eq!(c.project().actor_cell(actor), Some(cell(0, 0)));
    assert!(c.world().model().cell(cell(0, 0)).unwrap().is_dirty());
    assert!(c.world().model().cell(cell(1, 0)).unwrap().is_dirty());
    assert_eq!(
        c.world().actor_record(actor).unwrap().identity().cell(),
        cell(0, 0)
    );
    assert_eq!(c.world().actor_payload(actor), Some(&b"actor original"[..]));
    let snapshot = block_on(provider.load_snapshot(Duration::from_secs(5)).unwrap()).unwrap();
    assert_eq!(snapshot.files().len(), 3); // Two cells and one moved actor, no old actor file.
    assert!(c
        .discard_edits_and_restore_transaction_snapshot(&snapshot)
        .is_err());
    assert!(c.world().model().cell(cell(0, 0)).unwrap().is_dirty());
    c.apply(A::Select(cell(0, 0))).unwrap();
    c.apply(A::AcquireLease).unwrap();
    c.apply(A::Select(cell(1, 0))).unwrap();
    c.apply(A::AcquireLease).unwrap();
    let old_view = c.build();
    c.discard_edits_and_restore_transaction_snapshot(&snapshot)
        .unwrap();
    assert_eq!(
        c.world().actor_record(actor).unwrap().identity().cell(),
        cell(1, 0)
    );
    assert_eq!(c.world().actor_payload(actor), Some(&b"actor moved"[..]));
    assert_eq!(c.project().actor_cell(actor), Some(cell(1, 0)));
    c.apply(A::ProjectUndo).unwrap();
    assert!(c.world().model().is_cell_visible(cell(0, 0)));
    assert_eq!(c.project().actor_cell(actor), Some(cell(1, 0)));
    assert_eq!(c.world().actor_payload(actor), Some(&b"actor moved"[..]));
    assert!(!c.world().model().cell(cell(0, 0)).unwrap().is_dirty());
    assert!(!c.world().model().cell(cell(1, 0)).unwrap().is_dirty());
    assert!(matches!(
        c.handle_event(&old_view, "world:zoom-in", &UiEvent::Activate),
        Err(Error::StaleView)
    ));
    wait_idle(|| provider.outstanding_operations());
}

#[test]
fn actual_cross_cell_transaction_reconciles_both_payloads_only_after_real_receipt() {
    let temp = Temp::new();
    let provider = FileWorldTransactionProvider::new(
        &temp.0,
        WorldTransactionLimits::new(16, 4096, 16384, 2, 32768).unwrap(),
    )
    .unwrap();
    let mut c = controller(2);
    load(&mut c, cell(0, 0), b"source original");
    c.apply(A::AcquireLease).unwrap();
    c.apply(A::CrossCellSource).unwrap();
    load(&mut c, cell(1, 0), b"destination original");
    c.apply(A::AcquireLease).unwrap();
    let view = c.build();
    let mut ui = Ui::new(RecordingBackend::new());
    ui.mount(view.element());
    let _ = send(&mut c, &mut ui, view, "world:cross-cell");
    let ticket = c.take_intent().unwrap();
    let snapshot = block_on(provider.load_snapshot(Duration::from_secs(5)).unwrap()).unwrap();
    let edits = [
        WorldTransactionEdit::Cell {
            cell: cell(0, 0),
            payload: Arc::from(&b"project schema: source after"[..]),
        },
        WorldTransactionEdit::Cell {
            cell: cell(1, 0),
            payload: Arc::from(&b"project schema: destination after"[..]),
        },
    ];
    assert!(c
        .prepare_transaction(
            &ticket,
            StableGuid::from_path("incomplete"),
            &snapshot,
            &edits[..1],
            &provider
        )
        .is_err());
    assert!(!c.world().model().cell(cell(0, 0)).unwrap().is_dirty());
    let pending = c
        .prepare_transaction(
            &ticket,
            StableGuid::from_path("cross cell ui transaction"),
            &snapshot,
            &edits,
            &provider,
        )
        .unwrap();
    assert!(c.world().model().cell(cell(0, 0)).unwrap().is_dirty());
    assert!(c.world().model().cell(cell(1, 0)).unwrap().is_dirty());
    assert_eq!(c.world().payload(cell(0, 0)), Some(&b"source original"[..]));
    let receipt = block_on(
        provider
            .commit(pending.request().clone(), Duration::from_secs(5))
            .unwrap(),
    )
    .unwrap();
    c.commit_transaction(&pending, &receipt).unwrap();
    assert_eq!(
        c.world().payload(cell(0, 0)),
        Some(&b"project schema: source after"[..])
    );
    assert_eq!(
        c.world().payload(cell(1, 0)),
        Some(&b"project schema: destination after"[..])
    );
    assert!(!c.world().model().cell(cell(0, 0)).unwrap().is_dirty());
    assert!(!c.world().model().cell(cell(1, 0)).unwrap().is_dirty());
    let snapshot = block_on(provider.load_snapshot(Duration::from_secs(5)).unwrap()).unwrap();
    assert_eq!(snapshot.files().len(), 2);
    wait_idle(|| provider.outstanding_operations());
}
