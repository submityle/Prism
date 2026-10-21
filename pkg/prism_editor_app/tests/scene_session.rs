//! CPU and real offscreen GPU contracts for the reflected mesh projection.

#![cfg(feature = "desktop")]

use std::fs;
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::thread;
use std::time::{Duration, Instant};

use prism_ecs::prelude::ChildOf;
use prism_editor_app::{
    editor_scene_mesh_catalog, editor_scene_mesh_registry, ControllerFieldEdit, ControllerLimits,
    DesktopDocumentSession, DesktopGpu, DesktopRenderer, DesktopViewportBounds, DocumentCodec,
    DocumentLimits, EditorController, EditorDesktopUi, EditorSceneMesh, EditorSceneSession,
    SceneProject, SceneProjectConfig,
};
use prism_editor_core::SelectionItem;
use prism_editor_panels::{OutlinerId, OutlinerNode};
use prism_editor_reflect::FieldPath;
use prism_editor_ui::{ActionValue, EditorAction, SelectionMode};
use prism_editor_viewport::SCENE_ID_IDENTITY_MATRIX;
use prism_reflect::{Reflect, TypeRegistry};
use prism_ui_layout::{Point, Rect, Size};

const OBJECT: u128 = 0x1234;
const ROW: u128 = 20;

fn triangle() -> EditorSceneMesh {
    EditorSceneMesh {
        positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        indices: vec![0, 1, 2],
        matrix: SCENE_ID_IDENTITY_MATRIX,
        visible: true,
    }
}

fn register_mesh(controller: &mut EditorController, id: u128, mesh: EditorSceneMesh) {
    controller
        .register_object(
            SelectionItem::entity(id),
            Box::new(mesh),
            editor_scene_mesh_registry(),
        )
        .expect("register reflected mesh root");
}

fn register_row(controller: &mut EditorController, id: u128, object: u128, parent: Option<u128>) {
    let mut row = OutlinerNode::new(
        OutlinerId::from_raw(id),
        parent.map(OutlinerId::from_raw),
        "Mesh",
    );
    row.set_selection(SelectionItem::entity(object));
    controller
        .register_outliner_node(row)
        .expect("register explicit mesh Outliner identity");
}

fn document(mesh: EditorSceneMesh) -> EditorController {
    let mut controller = EditorController::new();
    register_mesh(&mut controller, OBJECT, mesh);
    register_row(&mut controller, ROW, OBJECT, None);
    controller
}

// Deliberately omit ControllerObjectValidator to exercise the projection's
// defence against hosts that did not install the production admission policy.
fn unvalidated_document(mesh: EditorSceneMesh) -> EditorController {
    let mut registry = TypeRegistry::new();
    registry.register::<EditorSceneMesh>();
    registry.register::<Vec<[f32; 3]>>();
    registry.register::<[f32; 3]>();
    registry.register::<Vec<u32>>();
    registry.register::<u32>();
    registry.register::<[[f32; 4]; 4]>();
    registry.register::<[f32; 4]>();
    registry.register::<f32>();
    registry.register::<bool>();
    let mut controller = EditorController::new();
    controller
        .register_object(SelectionItem::entity(OBJECT), Box::new(mesh), registry)
        .unwrap();
    register_row(&mut controller, ROW, OBJECT, None);
    controller
}

fn round_trip(controller: &EditorController) -> EditorController {
    let catalog = editor_scene_mesh_catalog().expect("mesh schema catalog");
    let bytes = DocumentCodec::encode(controller, &catalog, DocumentLimits::default())
        .expect("encode real reflected document");
    DocumentCodec::decode(
        &bytes,
        &catalog,
        DocumentLimits::default(),
        ControllerLimits::default(),
    )
    .expect("decode real reflected document")
    .into_controller()
}

fn edit(controller: &mut EditorController, object: u128, path: &str, value: ActionValue) {
    let prepared = controller
        .prepare_edits(vec![ControllerFieldEdit {
            item: SelectionItem::entity(object),
            path: FieldPath::parse(path).expect("valid field path"),
            value,
        }])
        .expect("prepare reflected field edit");
    controller
        .commit_edits(prepared)
        .expect("commit field edit");
}

fn assert_mesh(
    session: &EditorSceneSession,
    controller: &EditorController,
    object: u128,
    expected: &EditorSceneMesh,
) {
    let item = SelectionItem::entity(object);
    let mut authored = triangle();
    authored
        .apply(controller.object_root(&item).expect("authoring root"))
        .expect("read typed or decoded reflected mesh");
    assert_eq!(&authored, expected, "controller is the authoring authority");

    let entity = session
        .bridge()
        .entity_for_object(&item)
        .expect("ECS mapping");
    assert_eq!(
        session.world().get::<EditorSceneMesh>(entity),
        Some(expected)
    );
    let scene = session
        .scene()
        .objects()
        .iter()
        .find(|mesh| mesh.selection == item)
        .expect("scene descriptor with the same authoring identity");
    assert_eq!(scene.positions, expected.positions);
    assert_eq!(scene.indices, expected.indices);
    assert_eq!(scene.visible, expected.visible);
    assert_eq!(scene.model, expected.matrix);
    assert_eq!(
        session.scene().object_model(scene.id),
        Some(expected.matrix)
    );
}

#[test]
fn empty_production_document_stays_empty_and_idle() {
    let controller = EditorController::new();
    let mut session = EditorSceneSession::new(&controller).expect("empty document projection");
    assert_eq!(session.world().entity_count(), 0);
    assert_eq!(session.bridge().entity_count(), 0);
    assert!(session.scene().objects().is_empty());
    let before = session.work();
    assert_eq!(before.document_rebuilds, 1);
    assert_eq!(before.objects_synchronized, 0);
    assert!(!session.synchronize(&controller).unwrap());
    assert_eq!(session.work(), before);
}

#[test]
fn decoded_reflect_document_materializes_matching_ecs_scene_and_hierarchy() {
    let mut mesh = triangle();
    mesh.matrix[3][0] = 0.25;
    let mut authored = EditorController::new();
    authored
        .register_outliner_node(OutlinerNode::new(OutlinerId::from_raw(10), None, "Meshes"))
        .unwrap();
    register_mesh(&mut authored, OBJECT, mesh.clone());
    register_row(&mut authored, ROW, OBJECT, Some(10));
    let loaded = round_trip(&authored);
    assert!(
        loaded
            .object_root(&SelectionItem::entity(OBJECT))
            .unwrap()
            .downcast_ref::<EditorSceneMesh>()
            .is_none(),
        "exercise decoded dynamic reflection, not just typed roots"
    );

    let session = EditorSceneSession::new(&loaded).expect("materialize decoded mesh");
    assert_mesh(&session, &loaded, OBJECT, &mesh);
    let entity = session
        .bridge()
        .entity_for_object(&SelectionItem::entity(OBJECT))
        .unwrap();
    let group = session
        .bridge()
        .entity_for_outliner(OutlinerId::from_raw(10))
        .unwrap();
    assert_eq!(
        session
            .bridge()
            .entity_for_outliner(OutlinerId::from_raw(ROW)),
        Some(entity)
    );
    assert_eq!(
        session.world().relation_targets::<ChildOf>(entity),
        &[group]
    );
    assert_eq!(session.world().entity_count(), 2);
    assert_eq!(session.scene().objects().len(), 1);
    assert_eq!(
        session.scene().objects()[0].outliner,
        Some(OutlinerId::from_raw(ROW))
    );
    assert_eq!(session.work().document_rebuilds, 1);
    assert_eq!(session.work().objects_synchronized, 1);
}

#[test]
fn matrix_edits_undo_redo_and_selection_do_not_replace_geometry_or_entities() {
    let mut controller = round_trip(&document(triangle()));
    let mut session = EditorSceneSession::new(&controller).unwrap();
    let entity = session
        .bridge()
        .entity_for_object(&SelectionItem::entity(OBJECT))
        .unwrap();
    let before = session.work();
    let mut moved = triangle();
    moved.matrix[3][0] = 0.5;
    edit(
        &mut controller,
        OBJECT,
        "matrix[3][0]",
        ActionValue::float(0.5),
    );
    assert!(session.synchronize(&controller).unwrap());
    assert_mesh(&session, &controller, OBJECT, &moved);

    controller.execute(EditorAction::Undo).unwrap();
    assert!(session.synchronize(&controller).unwrap());
    assert_mesh(&session, &controller, OBJECT, &triangle());
    controller.execute(EditorAction::Redo).unwrap();
    assert!(session.synchronize(&controller).unwrap());
    assert_mesh(&session, &controller, OBJECT, &moved);

    let after = session.work();
    assert_eq!(after.document_rebuilds, before.document_rebuilds);
    assert_eq!(after.geometry_replacements, before.geometry_replacements);
    assert_eq!(after.matrix_updates, before.matrix_updates + 3);
    assert_eq!(after.objects_synchronized, before.objects_synchronized + 3);
    assert_eq!(
        session
            .bridge()
            .entity_for_object(&SelectionItem::entity(OBJECT)),
        Some(entity)
    );
    let model_revision = controller.model_revision();
    controller.select(SelectionItem::entity(OBJECT)).unwrap();
    assert_eq!(controller.model_revision(), model_revision);
    for _ in 0..3 {
        assert!(!session.synchronize(&controller).unwrap());
    }
    assert_eq!(
        session.work(),
        after,
        "selection and idle polls do no projection work"
    );
}

#[test]
fn geometry_and_visibility_follow_the_real_document_undo_redo_history() {
    let mut controller = round_trip(&document(triangle()));
    let mut session = EditorSceneSession::new(&controller).unwrap();
    let before = session.work();
    let mut changed = triangle();
    changed.positions[1][0] = 0.75;
    edit(
        &mut controller,
        OBJECT,
        "positions[1][0]",
        ActionValue::float(0.75),
    );
    assert!(session.synchronize(&controller).unwrap());
    assert_mesh(&session, &controller, OBJECT, &changed);

    let mut hidden = changed.clone();
    hidden.visible = false;
    edit(&mut controller, OBJECT, "visible", ActionValue::bool(false));
    assert!(session.synchronize(&controller).unwrap());
    assert_mesh(&session, &controller, OBJECT, &hidden);
    for expected in [&changed, &triangle()] {
        controller.execute(EditorAction::Undo).unwrap();
        assert!(session.synchronize(&controller).unwrap());
        assert_mesh(&session, &controller, OBJECT, expected);
    }
    for expected in [&changed, &hidden] {
        controller.execute(EditorAction::Redo).unwrap();
        assert!(session.synchronize(&controller).unwrap());
        assert_mesh(&session, &controller, OBJECT, expected);
    }
    let after = session.work();
    assert_eq!(after.document_rebuilds, before.document_rebuilds);
    assert_eq!(
        after.geometry_replacements,
        before.geometry_replacements + 3
    );
    assert_eq!(after.visibility_updates, before.visibility_updates + 3);
    assert_eq!(after.matrix_updates, before.matrix_updates);
    assert_eq!(after.objects_synchronized, before.objects_synchronized + 6);
    assert_mesh(
        &EditorSceneSession::new(&round_trip(&controller)).unwrap(),
        &controller,
        OBJECT,
        &hidden,
    );
}

#[test]
fn visibility_and_coalesced_properties_preserve_cpu_geometry_allocations() {
    for decoded in [false, true] {
        let authored = document(triangle());
        let mut controller = if decoded {
            round_trip(&authored)
        } else {
            authored
        };
        let mut session = EditorSceneSession::new(&controller).unwrap();
        let entity = session
            .bridge()
            .entity_for_object(&SelectionItem::entity(OBJECT))
            .unwrap();
        let pointers = |session: &EditorSceneSession| {
            let mesh = session.world().get::<EditorSceneMesh>(entity).unwrap();
            let object = &session.scene().objects()[0];
            (
                mesh.positions.as_ptr(),
                mesh.indices.as_ptr(),
                object.positions.as_ptr(),
                object.indices.as_ptr(),
            )
        };
        let geometry = pointers(&session);
        let before = session.work();
        let generation = session.scene().status().scene_generation;
        edit(&mut controller, OBJECT, "visible", ActionValue::bool(false));
        assert!(session.synchronize(&controller).unwrap());
        let mut expected = triangle();
        expected.visible = false;
        assert_mesh(&session, &controller, OBJECT, &expected);
        assert!(session.scene().status().scene_generation > generation);
        assert_eq!(pointers(&session), geometry);
        for action in [EditorAction::Undo, EditorAction::Redo] {
            controller.execute(action).unwrap();
            assert!(session.synchronize(&controller).unwrap());
            assert_eq!(pointers(&session), geometry);
        }
        // Two commits coalesce in the field journal before one projection sync.
        edit(&mut controller, OBJECT, "visible", ActionValue::bool(true));
        edit(
            &mut controller,
            OBJECT,
            "matrix[3][0]",
            ActionValue::float(0.25),
        );
        assert!(session.synchronize(&controller).unwrap());
        expected.visible = true;
        expected.matrix[3][0] = 0.25;
        assert_mesh(&session, &controller, OBJECT, &expected);
        assert_eq!(pointers(&session), geometry);
        assert_eq!(
            session.work().geometry_replacements,
            before.geometry_replacements
        );
        assert_eq!(
            session.work().visibility_updates,
            before.visibility_updates + 4
        );
        assert_eq!(session.work().matrix_updates, before.matrix_updates + 1);
        assert_eq!(session.work().document_rebuilds, before.document_rebuilds);
        assert!(!session.synchronize(&controller).unwrap());
    }
}

#[test]
fn structural_registration_rebuilds_even_at_model_revision_zero() {
    let mut controller = EditorController::new();
    let mut session = EditorSceneSession::new(&controller).unwrap();
    register_mesh(&mut controller, OBJECT, triangle());
    assert_eq!(controller.model_revision(), 0);
    assert!(session.synchronize(&controller).unwrap());
    assert_mesh(&session, &controller, OBJECT, &triangle());
    assert_eq!(session.scene().objects()[0].outliner, None);
    assert_eq!(session.work().document_rebuilds, 2);

    register_row(&mut controller, ROW, OBJECT, None);
    assert_eq!(controller.model_revision(), 0);
    assert!(session.synchronize(&controller).unwrap());
    let entity = session
        .bridge()
        .entity_for_object(&SelectionItem::entity(OBJECT))
        .unwrap();
    assert_eq!(
        session
            .bridge()
            .entity_for_outliner(OutlinerId::from_raw(ROW)),
        Some(entity)
    );
    assert_eq!(
        session.scene().objects()[0].outliner,
        Some(OutlinerId::from_raw(ROW))
    );
    assert_eq!(session.work().document_rebuilds, 3);
    assert!(!session.synchronize(&controller).unwrap());
}

#[test]
fn foreign_controller_with_identical_revisions_and_shape_rebuilds_by_identity() {
    let original = document(triangle());
    let mut other_mesh = triangle();
    other_mesh.positions[0][1] = 0.25;
    let foreign = document(other_mesh.clone());
    assert_eq!(original.revision(), foreign.revision());
    assert_eq!(original.model_revision(), foreign.model_revision());
    assert_eq!(original.shape_revision(), foreign.shape_revision());
    let mut session = EditorSceneSession::new(&original).unwrap();
    assert!(session.synchronize(&foreign).unwrap());
    assert_mesh(&session, &foreign, OBJECT, &other_mesh);
    assert_eq!(session.work().document_rebuilds, 2);
    assert!(!session.synchronize(&foreign).unwrap());
    assert!(session.synchronize(&original).unwrap());
    assert_mesh(&session, &original, OBJECT, &triangle());
    assert_eq!(session.work().document_rebuilds, 3);
}

#[test]
fn prepare_and_discard_preserve_live_projection_until_explicit_commit() {
    let original = document(triangle());
    let mut mesh = triangle();
    mesh.visible = false;
    mesh.matrix[3][1] = 0.5;
    let candidate = round_trip(&document(mesh.clone()));
    let mut session = EditorSceneSession::new(&original).unwrap();
    let before = session.work();
    let generation = session.scene().status().scene_generation;

    session.prepare_document(&candidate).unwrap();
    assert_mesh(&session, &original, OBJECT, &triangle());
    assert_eq!(session.work(), before);
    assert_eq!(session.scene().status().scene_generation, generation);
    session.discard_document();
    session.commit_document();
    assert_mesh(&session, &original, OBJECT, &triangle());
    assert_eq!(session.work(), before);
    assert!(!session.synchronize(&original).unwrap());

    session.prepare_document(&candidate).unwrap();
    session.commit_document();
    assert_mesh(&session, &candidate, OBJECT, &mesh);
    assert_eq!(
        session.work().document_rebuilds,
        before.document_rebuilds + 1
    );
    assert!(!session.synchronize(&candidate).unwrap());
    let committed = session.work();
    session.commit_document();
    assert_eq!(
        session.work(),
        committed,
        "commit without a candidate is inert"
    );
}

#[test]
fn pending_candidate_fences_idle_changed_and_foreign_synchronization() {
    let mut original = document(triangle());
    let mut mesh = triangle();
    mesh.visible = false;
    let candidate = round_trip(&document(mesh.clone()));
    let foreign = document(triangle());
    let mut session = EditorSceneSession::new(&original).unwrap();
    let before = session.work();
    let generation = session.scene().status().scene_generation;
    session.prepare_document(&candidate).unwrap();
    assert!(
        session.synchronize(&original).is_err(),
        "even the idle path is fenced"
    );
    edit(
        &mut original,
        OBJECT,
        "matrix[3][0]",
        ActionValue::float(0.25),
    );
    assert!(session.synchronize(&original).is_err());
    assert!(
        session.synchronize(&foreign).is_err(),
        "foreign controller must not overwrite pending"
    );
    assert!(session
        .set_camera(
            SCENE_ID_IDENTITY_MATRIX,
            prism_editor_viewport::SceneIdDepth::Forward
        )
        .is_err());
    assert_eq!(session.work(), before);
    assert_eq!(session.scene().status().scene_generation, generation);
    assert_mesh(&session, &foreign, OBJECT, &triangle());
    session.commit_document();
    assert_mesh(&session, &candidate, OBJECT, &mesh);
    assert!(!session.synchronize(&candidate).unwrap());
    assert_eq!(
        session.work().document_rebuilds,
        before.document_rebuilds + 1
    );
    session.prepare_document(&original).unwrap();
    session.discard_document();
    assert!(!session.synchronize(&candidate).unwrap());
    session.commit_document();
    assert_mesh(&session, &candidate, OBJECT, &mesh);
}

#[test]
fn invalid_prepare_drops_stale_candidate_without_publishing_any_new_state() {
    let original = document(triangle());
    let mut candidate_mesh = triangle();
    candidate_mesh.visible = false;
    let candidate = document(candidate_mesh);
    let mut invalid_mesh = triangle();
    invalid_mesh.indices[2] = 99;
    let invalid = unvalidated_document(invalid_mesh);
    assert!(EditorSceneSession::new(&invalid).is_err());
    let mut session = EditorSceneSession::new(&original).unwrap();
    let before = session.work();
    let generation = session.scene().status().scene_generation;
    session.prepare_document(&candidate).unwrap();
    assert!(session.prepare_document(&invalid).is_err());
    session.commit_document();
    session.discard_document();
    assert_mesh(&session, &original, OBJECT, &triangle());
    assert_eq!(session.work(), before);
    assert_eq!(session.scene().status().scene_generation, generation);
    assert!(!session.synchronize(&original).unwrap());
}

#[test]
fn invalid_incremental_geometry_rejects_legal_sibling_batch_atomically() {
    let mut controller = document(triangle());
    register_mesh(&mut controller, OBJECT + 1, triangle());
    let mut session = EditorSceneSession::new(&controller).unwrap();
    let before = session.work();
    let revision = controller.revision();
    let model_revision = controller.model_revision();
    let generation = session.scene().status().scene_generation;
    let history = controller
        .history_lengths(&SelectionItem::entity(OBJECT))
        .unwrap();
    let sibling_history = controller
        .history_lengths(&SelectionItem::entity(OBJECT + 1))
        .unwrap();
    assert!(controller
        .prepare_edits(vec![
            ControllerFieldEdit {
                item: SelectionItem::entity(OBJECT),
                path: FieldPath::parse("matrix[3][0]").unwrap(),
                value: ActionValue::float(0.5)
            },
            ControllerFieldEdit {
                item: SelectionItem::entity(OBJECT + 1),
                path: FieldPath::parse("indices[2]").unwrap(),
                value: ActionValue::unsigned(99)
            },
        ])
        .is_err());
    assert_eq!(controller.revision(), revision);
    assert_eq!(controller.model_revision(), model_revision);
    assert_eq!(
        controller
            .history_lengths(&SelectionItem::entity(OBJECT))
            .unwrap(),
        history
    );
    assert_eq!(
        controller
            .history_lengths(&SelectionItem::entity(OBJECT + 1))
            .unwrap(),
        sibling_history
    );
    assert!(!session.synchronize(&controller).unwrap());
    assert_eq!(session.work(), before);
    assert_eq!(session.scene().status().scene_generation, generation);
    let unchanged = document(triangle());
    assert_mesh(&session, &unchanged, OBJECT, &triangle());
    let mut second = EditorController::new();
    register_mesh(&mut second, OBJECT + 1, triangle());
    assert_mesh(&session, &second, OBJECT + 1, &triangle());

    assert_mesh(&session, &controller, OBJECT, &triangle());
    assert_mesh(&session, &controller, OBJECT + 1, &triangle());
    edit(
        &mut controller,
        OBJECT,
        "matrix[3][0]",
        ActionValue::float(0.5),
    );
    assert!(session.synchronize(&controller).unwrap());
    let mut moved = triangle();
    moved.matrix[3][0] = 0.5;
    assert_mesh(&session, &controller, OBJECT, &moved);
    assert_mesh(&session, &controller, OBJECT + 1, &triangle());
    assert_eq!(session.work().document_rebuilds, before.document_rebuilds);
    assert!(!session.synchronize(&controller).unwrap());
}

struct TempRoot(PathBuf);

impl TempRoot {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "prism-scene-session-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create private test directory {path:?}: {error}"),
            }
        }
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove only this private scene-session test directory");
    }
}

struct ThreadWake(thread::Thread);

impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

fn wait_for_save(
    persistence: &mut DesktopDocumentSession,
    ui: &mut EditorDesktopUi,
    renderer: &mut EditorSceneSession,
) {
    let waker = Waker::from(Arc::new(ThreadWake(thread::current())));
    let mut context = Context::from_waker(&waker);
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match persistence
            .poll(&mut context, ui, renderer)
            .expect("poll real filesystem save")
        {
            Poll::Ready(_) => break,
            Poll::Pending => {
                assert!(Instant::now() < deadline, "scene-session save timed out");
                thread::park_timeout(Duration::from_millis(5));
            }
        }
    }
    assert!(!persistence.is_busy());
    assert!(!persistence.reconciliation_required());
    assert!(
        !ui.controller().is_dirty(),
        "save must be durably acknowledged"
    );
}

#[test]
fn filesystem_scene_project_save_open_restores_authored_mesh_ecs_and_scene() {
    let temp = TempRoot::new();
    let path = temp.0.join("project");
    let project = SceneProject::create_with(
        &path,
        editor_scene_mesh_catalog().unwrap(),
        SceneProjectConfig::default(),
    )
    .expect("create real mesh-profile project");
    let guid = project.project_guid();
    let (mut ui, mut persistence) = project.into_desktop();
    let mut renderer = EditorSceneSession::new(ui.controller()).unwrap();
    assert!(renderer.scene().objects().is_empty());
    register_mesh(ui.controller_mut(), OBJECT, triangle());
    register_row(ui.controller_mut(), ROW, OBJECT, None);
    edit(
        ui.controller_mut(),
        OBJECT,
        "matrix[3][0]",
        ActionValue::float(0.25),
    );
    assert!(renderer.synchronize(ui.controller()).unwrap());
    let mut expected = triangle();
    expected.matrix[3][0] = 0.25;
    assert_mesh(&renderer, ui.controller(), OBJECT, &expected);
    persistence.save(&mut ui).expect("submit real project save");
    wait_for_save(&mut persistence, &mut ui, &mut renderer);
    let durable = persistence.durable_revision();
    assert_eq!(durable, Some(1));
    drop((ui, persistence, renderer));

    let reopened = SceneProject::open_with(
        &path,
        editor_scene_mesh_catalog().unwrap(),
        SceneProjectConfig::default(),
    )
    .expect("reopen durably saved mesh-profile project");
    assert_eq!(reopened.project_guid(), guid);
    assert_eq!(reopened.document_session().durable_revision(), durable);
    let controller = reopened.ui().controller();
    assert!(!controller.is_dirty());
    assert_eq!(controller.document_objects().len(), 1);
    assert_eq!(controller.document_outliner_nodes().len(), 1);
    let restored = EditorSceneSession::new(controller).expect("materialize reopened real scene");
    assert_mesh(&restored, controller, OBJECT, &expected);
    assert_eq!(
        restored.scene().objects()[0].outliner,
        Some(OutlinerId::from_raw(ROW))
    );
}

#[expect(
    clippy::print_stderr,
    reason = "required hardware and explicit skip evidence"
)]
fn gpu() -> Option<DesktopGpu> {
    let required = std::env::var("PRISM_REQUIRE_GPU").as_deref() == Ok("1");
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter =
        match futures_lite::future::block_on(instance.request_adapter(&Default::default())) {
            Ok(adapter) => adapter,
            Err(error) => {
                assert!(!required, "required scene-session GPU unavailable: {error}");
                eprintln!("SKIP scene-session GPU: {error}");
                return None;
            }
        };
    eprintln!("scene-session GPU: {:?}", adapter.get_info());
    let (device, queue) =
        futures_lite::future::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .expect("real scene-session GPU device");
    Some(DesktopGpu {
        adapter,
        device,
        queue,
        format: wgpu::TextureFormat::Rgba8Unorm,
    })
}

fn render_session(
    session: &mut EditorSceneSession,
    gpu: &DesktopGpu,
    ui: &mut EditorDesktopUi,
    output: &wgpu::Texture,
) {
    let view = output.create_view(&Default::default());
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
    }
    session
        .encode(
            gpu,
            ui,
            &mut encoder,
            &view,
            DesktopViewportBounds {
                logical: Rect::new(Point::new(0.0, 0.0), Size::new(32.0, 32.0)),
                physical: Rect::new(Point::new(0.0, 0.0), Size::new(32.0, 32.0)),
                surface_extent: [32, 32],
            },
        )
        .unwrap();
    gpu.queue.submit([encoder.finish()]);
}

fn pick_session(
    session: &mut EditorSceneSession,
    gpu: &DesktopGpu,
    ui: &mut EditorDesktopUi,
    visible: bool,
) {
    let before = session.scene().status();
    assert!(session
        .pointer(gpu, ui, [16.0, 16.0], SelectionMode::Replace)
        .unwrap());
    assert!(session.scene().status().pending);
    let deadline = Instant::now() + Duration::from_secs(10);
    while session.scene().status().pending {
        session.poll(gpu, ui).unwrap(); // Readback uses nonblocking device polling.
        assert!(
            Instant::now() < deadline,
            "scene-session GPU pick timed out"
        );
        if session.scene().status().pending {
            thread::park_timeout(Duration::from_millis(1));
        }
    }
    assert_eq!(
        session.scene().status().accepted_picks,
        before.accepted_picks + 1
    );
    let expected = visible.then(|| SelectionItem::entity(OBJECT));
    assert_eq!(ui.controller().selection().primary(), expected.as_ref());
}

fn upload_counts(session: &EditorSceneSession) -> (u64, u64, u64, u64) {
    let status = session.scene().status();
    (
        status.geometry_uploads,
        status.geometry_uploaded_bytes,
        status.transform_uploads,
        status.geometry_renderer_creations,
    )
}

#[test]
fn gpu_visibility_undo_redo_picking_retains_renderer_without_geometry_or_transform_uploads() {
    let Some(gpu) = gpu() else {
        return;
    };
    let mut mesh = triangle();
    mesh.positions = vec![[-0.75, -0.75, 0.5], [0.75, -0.75, 0.5], [0.0, 0.75, 0.5]];
    let controller = round_trip(&document(mesh));
    let mut session = EditorSceneSession::new(&controller).unwrap();
    let mut ui = EditorDesktopUi::new(controller).unwrap();
    session.initialize(&gpu).unwrap();
    let output = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene-session visibility proof"),
        size: wgpu::Extent3d {
            width: 32,
            height: 32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: gpu.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    render_session(&mut session, &gpu, &mut ui, &output);
    pick_session(&mut session, &gpu, &mut ui, true);
    let baseline = upload_counts(&session);
    assert_eq!(baseline.0, 1, "one actual initial geometry upload");
    assert_eq!(
        baseline.1,
        3 * 12 + 3 * 4,
        "exact vertex/index payload bytes"
    );
    assert_eq!(baseline.2, 1);
    assert!(baseline.3 > 0);
    let before = session.work();

    // An in-flight old pick is fenced immediately, before the next encode.
    assert!(session
        .pointer(&gpu, &mut ui, [16.0, 16.0], SelectionMode::Replace)
        .unwrap());
    let pending_generation = session.scene().status().scene_generation;
    let rejected = session.scene().status().rejected_picks;
    edit(
        ui.controller_mut(),
        OBJECT,
        "visible",
        ActionValue::bool(false),
    );
    assert!(session.synchronize(ui.controller()).unwrap());
    assert!(!session.scene().status().pending);
    assert_eq!(session.scene().status().rejected_picks, rejected + 1);
    assert!(session.scene().status().scene_generation > pending_generation);
    assert!(
        !session
            .pointer(&gpu, &mut ui, [16.0, 16.0], SelectionMode::Replace)
            .unwrap(),
        "dirty visibility cannot pick old metadata"
    );
    render_session(&mut session, &gpu, &mut ui, &output);
    pick_session(&mut session, &gpu, &mut ui, false);
    assert_eq!(upload_counts(&session), baseline);

    for (action, visible) in [(EditorAction::Undo, true), (EditorAction::Redo, false)] {
        ui.controller_mut().execute(action).unwrap();
        render_session(&mut session, &gpu, &mut ui, &output);
        pick_session(&mut session, &gpu, &mut ui, visible);
        assert_eq!(upload_counts(&session), baseline);
    }
    assert_eq!(
        session.work().visibility_updates,
        before.visibility_updates + 3
    );
    assert_eq!(
        session.work().geometry_replacements,
        before.geometry_replacements
    );
    assert_eq!(session.work().document_rebuilds, before.document_rebuilds);

    // Positive controls ensure these counters really observe GPU upload calls.
    edit(
        ui.controller_mut(),
        OBJECT,
        "matrix[3][0]",
        ActionValue::float(0.125),
    );
    render_session(&mut session, &gpu, &mut ui, &output);
    assert_eq!(
        upload_counts(&session),
        (baseline.0, baseline.1, baseline.2 + 1, baseline.3)
    );
    edit(
        ui.controller_mut(),
        OBJECT,
        "positions[0][0]",
        ActionValue::float(-0.625),
    );
    render_session(&mut session, &gpu, &mut ui, &output);
    let after_geometry = upload_counts(&session);
    assert_eq!(
        after_geometry,
        (baseline.0 + 1, baseline.1 * 2, baseline.2 + 2, baseline.3)
    );
    assert_eq!(
        session.work().geometry_replacements,
        before.geometry_replacements + 1
    );
    render_session(&mut session, &gpu, &mut ui, &output);
    assert_eq!(
        upload_counts(&session),
        after_geometry,
        "idle encode uploads nothing"
    );
}
