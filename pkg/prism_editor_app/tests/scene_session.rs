//! CPU and real offscreen GPU contracts for the reflected mesh projection.

#![cfg(feature = "desktop")]

extern crate alloc;

use alloc::{sync::Arc, task::Wake};
use core::task::{Context, Poll, Waker};
use std::any::{type_name, Any};
use std::fs;
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use prism_ecs::prelude::ChildOf;
use prism_editor_app::{
    editor_scene_mesh_catalog, editor_scene_mesh_registry, install_scene_authoring_commands,
    ControllerFieldEdit, ControllerLimits, ControllerObject, DesktopDocumentSession, DesktopGpu,
    DesktopRenderer, DesktopViewportBounds, DocumentCodec, DocumentLimits, EditorController,
    EditorDesktopUi, EditorSceneMesh, EditorSceneSession, SceneProject, SceneProjectConfig,
    SCENE_ADD_TRIANGLE_COMMAND, SCENE_DELETE_SELECTION_COMMAND,
};
use prism_editor_core::{SelectionItem, SelectionKind};
use prism_editor_panels::{OutlinerId, OutlinerNode};
use prism_editor_reflect::FieldPath;
use prism_editor_ui::{ActionValue, EditorAction, SelectionMode};
use prism_editor_viewport::SCENE_ID_IDENTITY_MATRIX;
use prism_reflect::{
    DynamicStruct, List, Reflect, ReflectMut, ReflectRef, TypeInfo, TypeRegistry, Typed,
};
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
    let mut controller = EditorController::new();
    let mut session = EditorSceneSession::new(&mut controller).expect("empty document projection");
    assert_eq!(session.world().entity_count(), 0);
    assert_eq!(session.bridge().entity_count(), 0);
    assert!(session.scene().objects().is_empty());
    let before = session.work();
    assert_eq!(before.document_rebuilds, 1);
    assert_eq!(before.objects_synchronized, 0);
    assert!(!session.synchronize(&controller).unwrap());
    assert_eq!(session.work(), before);
    assert_eq!(
        controller.document_budget_usage("scene.geometry_bytes"),
        Some((0, 64 * 1024 * 1024))
    );
    assert!(!controller.has_command(SCENE_ADD_TRIANGLE_COMMAND));
}

#[test]
fn constructor_rejects_unbudgeted_undo_or_redo_without_installing_policy() {
    let mut controller = document(triangle());
    edit(&mut controller, OBJECT, "visible", ActionValue::bool(false));
    for history in [(1, 0), (0, 1)] {
        let revision = controller.revision();
        assert_eq!(controller.document_history_lengths(), history);
        let error = EditorSceneSession::new(&mut controller).err().unwrap();
        assert!(error.to_string().contains("undo/redo history"));
        assert_eq!(
            controller.document_budget_usage("scene.geometry_bytes"),
            None
        );
        assert_eq!(controller.revision(), revision);
        assert_eq!(controller.document_history_lengths(), history);
        assert!(!controller.has_command(SCENE_ADD_TRIANGLE_COMMAND));
        assert!(install_scene_authoring_commands(&mut controller).is_err());
        assert!(!controller.has_command(SCENE_ADD_TRIANGLE_COMMAND));
        assert!(!controller.has_command(SCENE_DELETE_SELECTION_COMMAND));
        assert_eq!(
            controller.document_budget_usage("scene.geometry_bytes"),
            None
        );
        assert_eq!(controller.revision(), revision);
        if history == (1, 0) {
            controller.execute(EditorAction::Undo).unwrap();
        }
    }
}

#[test]
fn constructor_validation_failure_never_installs_document_policy() {
    let mut invalid_mesh = triangle();
    invalid_mesh.indices[2] = 99;
    let mut invalid = unvalidated_document(invalid_mesh);
    let revision = invalid.revision();
    assert!(EditorSceneSession::new(&mut invalid).is_err());
    assert_eq!(invalid.document_budget_usage("scene.geometry_bytes"), None);
    assert_eq!(invalid.revision(), revision);
    assert!(!invalid.is_dirty());

    let mut valid = document(triangle());
    let revision = valid.revision();
    let mut bad_camera = SCENE_ID_IDENTITY_MATRIX;
    bad_camera[0][0] = f32::NAN;
    assert!(EditorSceneSession::with_camera(
        &mut valid,
        bad_camera,
        prism_editor_viewport::SceneIdDepth::Forward
    )
    .is_err());
    assert_eq!(valid.document_budget_usage("scene.geometry_bytes"), None);
    assert_eq!(valid.revision(), revision);

    let mut exhausted = EditorController::with_limits(ControllerLimits {
        max_document_budgets: 0,
        ..ControllerLimits::default()
    });
    register_mesh(&mut exhausted, OBJECT, triangle());
    let revision = exhausted.revision();
    assert!(EditorSceneSession::new(&mut exhausted).is_err());
    assert_eq!(
        exhausted.document_budget_usage("scene.geometry_bytes"),
        None
    );
    assert_eq!(exhausted.revision(), revision);
    assert!(!exhausted.has_command(SCENE_ADD_TRIANGLE_COMMAND));
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
    let mut loaded = round_trip(&authored);
    assert!(
        loaded
            .object_root(&SelectionItem::entity(OBJECT))
            .unwrap()
            .downcast_ref::<EditorSceneMesh>()
            .is_none(),
        "exercise decoded dynamic reflection, not just typed roots"
    );

    let session = EditorSceneSession::new(&mut loaded).expect("materialize decoded mesh");
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
    let mut session = EditorSceneSession::new(&mut controller).unwrap();
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
    let mut session = EditorSceneSession::new(&mut controller).unwrap();
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
        &EditorSceneSession::new(&mut round_trip(&controller)).unwrap(),
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
        let mut session = EditorSceneSession::new(&mut controller).unwrap();
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

// An adversarial host can change its reflected lengths after controller
// admission. The session must reject the aggregate before cloning or reading
// any of these synthetic elements; no large allocation is needed for this test.
struct BudgetPositions {
    values: Vec<[f32; 3]>,
    reported_len: Arc<AtomicUsize>,
    reads: Arc<AtomicUsize>,
    clones: Arc<AtomicUsize>,
}

impl BudgetPositions {
    fn assert_readable(&self) {
        assert_eq!(
            self.reported_len.load(Ordering::Relaxed),
            self.values.len(),
            "aggregate budget must be checked before geometry access"
        );
    }
}

impl List for BudgetPositions {
    fn get(&self, index: usize) -> Option<&dyn Reflect> {
        self.assert_readable();
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.values.as_slice().get(index).map(|v| v as &dyn Reflect)
    }
    fn get_mut(&mut self, index: usize) -> Option<&mut dyn Reflect> {
        self.assert_readable();
        self.values
            .as_mut_slice()
            .get_mut(index)
            .map(|v| v as &mut dyn Reflect)
    }
    fn len(&self) -> usize {
        self.reported_len.load(Ordering::Relaxed)
    }
    fn push(&mut self, _: Box<dyn Reflect>) -> Result<(), Box<dyn Reflect>> {
        panic!("test geometry must not be resized")
    }
    fn as_list(&self) -> &dyn List {
        self
    }
}

impl Reflect for BudgetPositions {
    fn type_name(&self) -> &'static str {
        type_name::<Vec<[f32; 3]>>()
    }
    fn type_info(&self) -> &'static TypeInfo {
        <Vec<[f32; 3]> as Typed>::type_info()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
    fn as_reflect(&self) -> &dyn Reflect {
        self
    }
    fn as_reflect_mut(&mut self) -> &mut dyn Reflect {
        self
    }
    fn reflect_ref(&self) -> ReflectRef<'_> {
        ReflectRef::List(self)
    }
    fn reflect_mut(&mut self) -> ReflectMut<'_> {
        ReflectMut::List(self)
    }
    fn reflect_clone(&self) -> Box<dyn Reflect> {
        self.assert_readable();
        self.clones.fetch_add(1, Ordering::Relaxed);
        self.values.reflect_clone()
    }
}

#[test]
fn profile_budget_admits_borrowed_lengths_and_blocks_create_and_bootstrap_overflow() {
    const LIMIT: usize = 64 * 1024 * 1024;
    // Deliberately synthetic reflected lengths: exercise the 64 MiB arithmetic
    // without allocating that payload. Production projection must still reject
    // unreadable geometry, as covered by the defence test below.
    for sibling in [false, true] {
        let reported_len = Arc::new(AtomicUsize::new(3));
        let reads = Arc::new(AtomicUsize::new(0));
        let clones = Arc::new(AtomicUsize::new(0));
        let mut root = DynamicStruct::new();
        root.set_represented_type_name(type_name::<EditorSceneMesh>());
        root.insert(
            "positions",
            BudgetPositions {
                values: triangle().positions,
                reported_len: reported_len.clone(),
                reads: reads.clone(),
                clones: clones.clone(),
            },
        );
        root.insert("indices", vec![0_u32, 1, 2]);
        root.insert("matrix", SCENE_ID_IDENTITY_MATRIX);
        root.insert("visible", false);
        let mut controller = EditorController::new();
        controller
            .register_object(
                SelectionItem::entity(OBJECT),
                Box::new(root),
                editor_scene_mesh_registry(),
            )
            .unwrap();
        if sibling {
            register_mesh(&mut controller, OBJECT + 1, triangle());
        }
        // This root contributes LIMIT - 4; a 48-byte sibling cannot fit.
        reported_len.store(LIMIT / 12 - 1, Ordering::Relaxed);
        reads.store(0, Ordering::Relaxed);
        clones.store(0, Ordering::Relaxed);
        let revision = controller.revision();
        let installed = install_scene_authoring_commands(&mut controller);
        assert_eq!(reads.load(Ordering::Relaxed), 0);
        assert_eq!(clones.load(Ordering::Relaxed), 0);
        if sibling {
            assert!(installed.is_err());
            assert_eq!(controller.revision(), revision);
            assert_eq!(
                controller.document_budget_usage("scene.geometry_bytes"),
                None
            );
            assert!(!controller.has_command(SCENE_ADD_TRIANGLE_COMMAND));
            assert!(!controller.has_command(SCENE_DELETE_SELECTION_COMMAND));
            continue;
        }
        installed.unwrap();
        assert_eq!(
            controller.document_budget_usage("scene.geometry_bytes"),
            Some((LIMIT as u64 - 4, LIMIT as u64))
        );
        let revision = controller.revision();
        let identity = controller.next_entity_identity().unwrap();
        assert!(controller
            .execute(EditorAction::ExecuteCommand {
                command_id: SCENE_ADD_TRIANGLE_COMMAND.into(),
            })
            .is_err());
        assert!(controller
            .register_object(
                SelectionItem::entity(OBJECT + 1),
                Box::new(triangle()),
                editor_scene_mesh_registry()
            )
            .is_err());
        assert_eq!(controller.revision(), revision);
        assert_eq!(controller.next_entity_identity().unwrap(), identity);
        assert_eq!(controller.document_history_lengths(), (0, 0));
        assert_eq!(controller.document_objects().len(), 1);
        assert!(!controller.is_dirty());
        assert_eq!(reads.load(Ordering::Relaxed), 0);
        assert_eq!(clones.load(Ordering::Relaxed), 0);
        assert_eq!(
            controller.document_budget_usage("scene.geometry_bytes"),
            Some((LIMIT as u64 - 4, LIMIT as u64))
        );
    }
}

#[test]
fn aggregate_budget_rejection_precedes_geometry_access_and_preserves_all_live_objects() {
    const BUDGET: usize = 64 * 1024 * 1024;
    let reported_len = Arc::new(AtomicUsize::new(3));
    let reads = Arc::new(AtomicUsize::new(0));
    let clones = Arc::new(AtomicUsize::new(0));
    let mut root = DynamicStruct::new();
    root.set_represented_type_name(type_name::<EditorSceneMesh>());
    root.insert(
        "positions",
        BudgetPositions {
            values: triangle().positions,
            reported_len: reported_len.clone(),
            reads: reads.clone(),
            clones: clones.clone(),
        },
    );
    root.insert("indices", vec![0_u32, 1, 2]);
    root.insert("matrix", SCENE_ID_IDENTITY_MATRIX);
    root.insert("visible", true);
    let mut controller = EditorController::new();
    controller
        .register_object(
            SelectionItem::entity(OBJECT),
            Box::new(root),
            editor_scene_mesh_registry(),
        )
        .unwrap();
    register_mesh(&mut controller, OBJECT + 1, triangle());
    let original = round_trip(&controller);
    let mut session = EditorSceneSession::new(&mut controller).unwrap();
    let before = session.work();
    let status = session.scene().status();
    let uploads = upload_counts(&session);
    let entities: Vec<_> = [OBJECT, OBJECT + 1]
        .map(|id| {
            session
                .bridge()
                .entity_for_object(&SelectionItem::entity(id))
                .unwrap()
        })
        .into_iter()
        .collect();
    let pointers = |session: &EditorSceneSession| {
        entities
            .iter()
            .map(|entity| {
                let mesh = session.world().get::<EditorSceneMesh>(*entity).unwrap();
                (mesh.positions.as_ptr(), mesh.indices.as_ptr())
            })
            .chain(
                session
                    .scene()
                    .objects()
                    .iter()
                    .map(|object| (object.positions.as_ptr(), object.indices.as_ptr())),
            )
            .collect::<Vec<_>>()
    };
    let geometry = pointers(&session);
    let prepared = controller
        .prepare_edits(
            [
                (OBJECT, "positions[0][0]", ActionValue::float(0.125)),
                (OBJECT + 1, "matrix[3][0]", ActionValue::float(0.25)),
                (OBJECT + 1, "visible", ActionValue::bool(false)),
            ]
            .into_iter()
            .map(|(id, path, value)| ControllerFieldEdit {
                item: SelectionItem::entity(id),
                path: FieldPath::parse(path).unwrap(),
                value,
            })
            .collect(),
        )
        .unwrap();
    controller.commit_edits(prepared).unwrap();
    // Individually legal (BUDGET - 4 bytes), but together with the untouched
    // 48-byte sibling exceeds the document limit. Make element access fatal.
    reported_len.store((BUDGET - 12) / 12, Ordering::Relaxed);
    reads.store(0, Ordering::Relaxed);
    clones.store(0, Ordering::Relaxed);
    for _ in 0..2 {
        let error = session.synchronize(&controller).unwrap_err();
        assert!(error.to_string().contains("64 MiB"), "{error}");
        assert_eq!(reads.load(Ordering::Relaxed), 0);
        assert_eq!(clones.load(Ordering::Relaxed), 0);
        assert_eq!(session.work(), before);
        let rejected = session.scene().status();
        assert_eq!(rejected.scene_generation, status.scene_generation);
        assert_eq!(rejected.target, status.target);
        assert_eq!(rejected.identity_allocations, status.identity_allocations);
        assert_eq!(upload_counts(&session), uploads);
        assert_eq!(pointers(&session), geometry);
        for (id, entity) in [OBJECT, OBJECT + 1].into_iter().zip(&entities) {
            assert_eq!(
                session
                    .bridge()
                    .entity_for_object(&SelectionItem::entity(id)),
                Some(*entity)
            );
            assert_mesh(&session, &original, id, &triangle());
        }
    }

    // A rejected sync must not consume the journal or publish the sibling's
    // matrix/visibility. Restoring the length permits retry of the entire batch.
    reported_len.store(3, Ordering::Relaxed);
    assert!(session.synchronize(&controller).unwrap());
    let mut changed = triangle();
    changed.positions[0][0] = 0.125;
    let mut sibling = triangle();
    sibling.matrix[3][0] = 0.25;
    sibling.visible = false;
    assert_mesh(&session, &controller, OBJECT, &changed);
    assert_mesh(&session, &controller, OBJECT + 1, &sibling);
    assert_eq!(session.work().document_rebuilds, before.document_rebuilds);
    assert_eq!(
        session.work().geometry_replacements,
        before.geometry_replacements + 1
    );
    controller.execute(EditorAction::Undo).unwrap();
    assert!(session.synchronize(&controller).unwrap());
    for id in [OBJECT, OBJECT + 1] {
        assert_mesh(&session, &controller, id, &triangle());
    }
    assert!(!session.synchronize(&controller).unwrap());
}

#[test]
fn structural_registration_rebuilds_even_at_model_revision_zero() {
    let mut controller = EditorController::new();
    let mut session = EditorSceneSession::new(&mut controller).unwrap();
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
    let mut original = document(triangle());
    let mut other_mesh = triangle();
    other_mesh.positions[0][1] = 0.25;
    let mut foreign = document(other_mesh.clone());
    // Both hosts carry the same production policy/configuration; only their
    // document identities (and geometry values) differ, not their revisions.
    install_scene_authoring_commands(&mut original).unwrap();
    install_scene_authoring_commands(&mut foreign).unwrap();
    assert_eq!(original.revision(), foreign.revision());
    assert_eq!(original.model_revision(), foreign.model_revision());
    assert_eq!(original.shape_revision(), foreign.shape_revision());
    let mut session = EditorSceneSession::new(&mut original).unwrap();
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
    let mut original = document(triangle());
    let mut mesh = triangle();
    mesh.visible = false;
    mesh.matrix[3][1] = 0.5;
    let candidate = round_trip(&document(mesh.clone()));
    let mut session = EditorSceneSession::new(&mut original).unwrap();
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
    let mut session = EditorSceneSession::new(&mut original).unwrap();
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
    let mut original = document(triangle());
    let mut candidate_mesh = triangle();
    candidate_mesh.visible = false;
    let candidate = document(candidate_mesh);
    let mut invalid_mesh = triangle();
    invalid_mesh.indices[2] = 99;
    let mut invalid = unvalidated_document(invalid_mesh);
    assert!(EditorSceneSession::new(&mut invalid).is_err());
    let mut session = EditorSceneSession::new(&mut original).unwrap();
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
    let mut session = EditorSceneSession::new(&mut controller).unwrap();
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
    let mut renderer = EditorSceneSession::new(ui.controller_mut()).unwrap();
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
    let (mut reopened_ui, _persistence) = reopened.into_desktop();
    let restored = EditorSceneSession::new(reopened_ui.controller_mut())
        .expect("materialize reopened real scene");
    let controller = reopened_ui.controller();
    assert!(!controller.is_dirty());
    assert_eq!(controller.document_objects().len(), 1);
    assert_eq!(controller.document_outliner_nodes().len(), 1);
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
    pick_session_at(session, gpu, ui, [16.0, 16.0], visible.then_some(OBJECT));
}

fn pick_session_at(
    session: &mut EditorSceneSession,
    gpu: &DesktopGpu,
    ui: &mut EditorDesktopUi,
    position: [f32; 2],
    expected: Option<u128>,
) {
    let before = session.scene().status();
    assert!(session
        .pointer(gpu, ui, position, SelectionMode::Replace)
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
    let expected = expected.map(SelectionItem::entity);
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

fn box_session(
    session: &mut EditorSceneSession,
    gpu: &DesktopGpu,
    ui: &mut EditorDesktopUi,
    expected: &[u128],
) {
    let before = session.scene().status();
    assert!(session
        .box_select(gpu, ui, [0.0, 0.0], [32.0, 32.0], SelectionMode::Replace)
        .unwrap());
    assert!(session.scene().status().pending_box);
    let deadline = Instant::now() + Duration::from_secs(10);
    while session.scene().status().pending_box {
        session.poll(gpu, ui).unwrap();
        assert!(
            Instant::now() < deadline,
            "scene-session GPU box pick timed out"
        );
        if session.scene().status().pending_box {
            thread::park_timeout(Duration::from_millis(1));
        }
    }
    assert_eq!(
        session.scene().status().accepted_picks,
        before.accepted_picks + 1
    );
    let mut actual = ui.controller().selection().items().to_vec();
    let mut expected: Vec<_> = expected
        .iter()
        .copied()
        .map(SelectionItem::entity)
        .collect();
    actual.sort();
    expected.sort();
    assert_eq!(
        actual, expected,
        "box picks resolve the current authored identities"
    );
}

fn session_output(gpu: &DesktopGpu, label: &'static str) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
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
    })
}

#[test]
fn gpu_authoring_create_delete_undo_redo_fences_point_and_box_picks_across_empty_shapes() {
    let Some(gpu) = gpu() else { return };
    let mut controller = EditorController::new();
    install_scene_authoring_commands(&mut controller).unwrap();
    let (item, _) = controller.next_entity_identity().unwrap();
    let SelectionKind::Entity(id) = *item.kind() else {
        panic!("authoring must allocate an entity identity");
    };
    let mut session = EditorSceneSession::new(&mut controller).unwrap();
    let mut ui = EditorDesktopUi::new(controller).unwrap();
    session.initialize(&gpu).unwrap();
    let output = session_output(&gpu, "scene-session structural authoring proof");
    render_session(&mut session, &gpu, &mut ui, &output);
    assert_eq!(session.scene().status().identity_allocations, 0);
    pick_session_at(&mut session, &gpu, &mut ui, [16.0, 16.0], None);
    box_session(&mut session, &gpu, &mut ui, &[]);

    // These are the installed production UI commands, not fabricated roots or
    // a replacement document. Picks never add a second Undo/Redo history.
    for (step, (action, present)) in [
        (
            EditorAction::ExecuteCommand {
                command_id: SCENE_ADD_TRIANGLE_COMMAND.into(),
            },
            true,
        ),
        (EditorAction::Undo, false),
        (EditorAction::Redo, true),
        (
            EditorAction::ExecuteCommand {
                command_id: SCENE_DELETE_SELECTION_COMMAND.into(),
            },
            false,
        ),
        (EditorAction::Undo, true),
        (EditorAction::Redo, false),
    ]
    .into_iter()
    .enumerate()
    {
        // Leave real submitted GPU work unpolled; shape publication must fence
        // both channels even if the hardware has already finished the copy.
        if step % 2 == 0 {
            assert!(session
                .pointer(&gpu, &mut ui, [16.0, 16.0], SelectionMode::Replace)
                .unwrap());
            assert!(session.scene().status().pending);
        } else {
            assert!(session
                .box_select(
                    &gpu,
                    &mut ui,
                    [0.0, 0.0],
                    [32.0, 32.0],
                    SelectionMode::Replace
                )
                .unwrap());
            assert!(session.scene().status().pending_box);
        }
        let before = session.scene().status();
        let rebuilds = session.work().document_rebuilds;
        let outcome = ui.execute(action).unwrap();
        assert!(outcome.model_changed);
        let selection = ui.controller().selection().items().to_vec();
        assert!(session.poll(&gpu, &mut ui).unwrap().redraw);
        let after = session.scene().status();
        assert!(!after.pending && !after.pending_box);
        assert_eq!(after.accepted_picks, before.accepted_picks);
        assert_eq!(after.rejected_picks, before.rejected_picks + 1);
        assert_eq!(ui.controller().selection().items(), selection.as_slice());
        assert_eq!(session.work().document_rebuilds, rebuilds + 1);
        let count = usize::from(present);
        assert_eq!(ui.controller().document_objects().len(), count);
        assert_eq!(ui.controller().document_outliner_nodes().len(), count);
        assert_eq!(session.bridge().entity_count(), count);
        assert_eq!(session.scene().objects().len(), count);
        assert_eq!(session.bridge().entity_for_object(&item).is_some(), present);
        if present {
            let expected = EditorSceneMesh {
                positions: vec![[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.0, 0.5, 0.5]],
                indices: vec![0, 1, 2],
                matrix: SCENE_ID_IDENTITY_MATRIX,
                visible: true,
            };
            assert_mesh(&session, ui.controller(), id, &expected);
        }
        // The previous targets cannot service either kind of pick until the
        // new shape has been encoded (including the one-to-zero transition).
        assert!(!session
            .pointer(&gpu, &mut ui, [16.0, 16.0], SelectionMode::Replace)
            .unwrap());
        assert!(!session
            .box_select(
                &gpu,
                &mut ui,
                [0.0, 0.0],
                [32.0, 32.0],
                SelectionMode::Replace
            )
            .unwrap());
        assert_eq!(
            upload_counts(&session),
            (
                before.geometry_uploads,
                before.geometry_uploaded_bytes,
                before.transform_uploads,
                before.geometry_renderer_creations,
            )
        );
        render_session(&mut session, &gpu, &mut ui, &output);
        // Shape changes intentionally attach a fresh renderer. An empty shape
        // allocates no IDs; restored entries must populate it even if unchanged.
        assert_eq!(
            upload_counts(&session),
            (
                before.geometry_uploads + count as u64,
                before.geometry_uploaded_bytes + 48 * count as u64,
                before.transform_uploads + count as u64,
                before.geometry_renderer_creations + 1,
            )
        );
        assert_eq!(
            session.scene().status().identity_allocations,
            before.identity_allocations + count as u64
        );
        pick_session_at(
            &mut session,
            &gpu,
            &mut ui,
            [16.0, 16.0],
            present.then_some(id),
        );
        box_session(
            &mut session,
            &gpu,
            &mut ui,
            if present {
                std::slice::from_ref(&id)
            } else {
                &[]
            },
        );
        let stable = upload_counts(&session);
        let identities = session.scene().status().identity_allocations;
        render_session(&mut session, &gpu, &mut ui, &output);
        assert_eq!(upload_counts(&session), stable);
        assert_eq!(session.scene().status().identity_allocations, identities);
        assert_eq!(session.work().document_rebuilds, rebuilds + 1);
    }
    assert_eq!(ui.controller().document_history_lengths(), (2, 0));
}

#[test]
fn gpu_structural_batch_create_and_middle_delete_undo_redo_remap_survivor_draw_keys() {
    let Some(gpu) = gpu() else { return };
    let mut controller = EditorController::new();
    install_scene_authoring_commands(&mut controller).unwrap();
    let mut session = EditorSceneSession::new(&mut controller).unwrap();
    let mut ui = EditorDesktopUi::new(controller).unwrap();
    session.initialize(&gpu).unwrap();
    let output = session_output(&gpu, "scene-session structural survivor proof");
    render_session(&mut session, &gpu, &mut ui, &output);

    let mut objects = Vec::new();
    for (offset, x) in [-0.6, 0.0, 0.6].into_iter().enumerate() {
        let item = SelectionItem::entity(OBJECT + offset as u128);
        let mut row = OutlinerNode::new(OutlinerId::from_raw(ROW + offset as u128), None, "Mesh");
        row.set_selection(item.clone());
        let mut mesh = triangle();
        // Distinct payload lengths AND silhouettes: using the deleted middle
        // mesh for the right survivor must fail actual pixel picks below.
        match offset {
            0 => {
                mesh.positions = vec![[-0.2, -0.6, 0.5], [0.2, -0.6, 0.5], [0.0, 0.6, 0.5]];
            }
            1 => {
                mesh.positions = vec![
                    [-0.2, 0.1, 0.5],
                    [0.2, 0.1, 0.5],
                    [0.2, 0.9, 0.5],
                    [-0.2, 0.9, 0.5],
                ];
                mesh.indices = vec![0, 1, 2, 0, 2, 3];
            }
            _ => {
                mesh.positions = vec![[-0.2, -0.9, 0.5], [0.2, -0.9, 0.5], [0.0, -0.1, 0.5]];
            }
        }
        mesh.matrix[3][0] = x;
        objects.push(ControllerObject {
            item,
            row,
            root: Box::new(mesh),
            registry: editor_scene_mesh_registry(),
        });
    }
    ui.controller_mut().create_objects(objects).unwrap();
    ui.refresh().unwrap();
    // encode itself observes shape_revision; no explicit synchronize is needed.
    render_session(&mut session, &gpu, &mut ui, &output);
    assert_eq!(session.scene().status().identity_allocations, 3);
    box_session(
        &mut session,
        &gpu,
        &mut ui,
        &[OBJECT, OBJECT + 1, OBJECT + 2],
    );
    pick_session_at(&mut session, &gpu, &mut ui, [16.0, 8.0], Some(OBJECT + 1));

    for (action, expected) in [
        (
            EditorAction::ExecuteCommand {
                command_id: SCENE_DELETE_SELECTION_COMMAND.into(),
            },
            vec![OBJECT, OBJECT + 2],
        ),
        (EditorAction::Undo, vec![OBJECT, OBJECT + 1, OBJECT + 2]),
        (EditorAction::Redo, vec![OBJECT, OBJECT + 2]),
        (EditorAction::Undo, vec![OBJECT, OBJECT + 1, OBJECT + 2]),
        (EditorAction::Undo, vec![]), // Undo the original atomic three-object create.
        (EditorAction::Redo, vec![OBJECT, OBJECT + 1, OBJECT + 2]),
    ] {
        let before = session.scene().status();
        let rebuilds = session.work().document_rebuilds;
        assert!(ui.execute(action).unwrap().model_changed);
        render_session(&mut session, &gpu, &mut ui, &output);
        let count = expected.len() as u64;
        let bytes = expected
            .iter()
            .map(|id| if *id == OBJECT + 1 { 72 } else { 48 })
            .sum::<u64>();
        assert_eq!(session.work().document_rebuilds, rebuilds + 1);
        assert_eq!(
            upload_counts(&session),
            (
                before.geometry_uploads + count,
                before.geometry_uploaded_bytes + bytes,
                before.transform_uploads + count,
                before.geometry_renderer_creations + 1,
            )
        );
        assert_eq!(
            session.scene().status().identity_allocations,
            before.identity_allocations + count
        );
        assert_eq!(
            session.scene().status().scene_metadata_upserts,
            before.scene_metadata_upserts + count
        );
        assert_eq!(session.bridge().entity_count(), expected.len());
        assert_eq!(session.scene().objects().len(), expected.len());
        for (offset, position) in [[6.0, 16.0], [16.0, 8.0], [25.0, 24.0]]
            .into_iter()
            .enumerate()
        {
            let id = OBJECT + offset as u128;
            let present = expected.contains(&id);
            assert_eq!(
                session
                    .bridge()
                    .entity_for_object(&SelectionItem::entity(id))
                    .is_some(),
                present
            );
            assert_eq!(
                ui.controller()
                    .object_root(&SelectionItem::entity(id))
                    .is_ok(),
                present
            );
            // Removing the middle descriptor shifts the right survivor from
            // draw/instance key 2 to 1; it must keep its own geometry and ID.
            pick_session_at(&mut session, &gpu, &mut ui, position, present.then_some(id));
        }
        // Negative silhouette controls distinguish vertex/index payloads from
        // correctly remapped transforms and identity uniforms alone.
        pick_session_at(&mut session, &gpu, &mut ui, [25.0, 8.0], None);
        pick_session_at(&mut session, &gpu, &mut ui, [16.0, 24.0], None);
        box_session(&mut session, &gpu, &mut ui, &expected);
        let stable = upload_counts(&session);
        render_session(&mut session, &gpu, &mut ui, &output);
        assert_eq!(upload_counts(&session), stable);
        assert_eq!(
            session.scene().status().identity_allocations,
            before.identity_allocations + count
        );
    }
    assert_eq!(ui.controller().document_history_lengths(), (1, 1));
}

#[test]
fn gpu_visibility_undo_redo_picking_retains_renderer_without_geometry_or_transform_uploads() {
    let Some(gpu) = gpu() else {
        return;
    };
    let mut mesh = triangle();
    mesh.positions = vec![[-0.75, -0.75, 0.5], [0.75, -0.75, 0.5], [0.0, 0.75, 0.5]];
    let mut controller = round_trip(&document(mesh));
    let mut session = EditorSceneSession::new(&mut controller).unwrap();
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
    assert_eq!(session.scene().status().identity_allocations, 1);
    assert_eq!(session.scene().status().scene_metadata_upserts, 1);
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
    assert_eq!(session.scene().status().identity_allocations, 1);

    for (action, visible) in [(EditorAction::Undo, true), (EditorAction::Redo, false)] {
        ui.controller_mut().execute(action).unwrap();
        render_session(&mut session, &gpu, &mut ui, &output);
        pick_session(&mut session, &gpu, &mut ui, visible);
        assert_eq!(upload_counts(&session), baseline);
        assert_eq!(session.scene().status().identity_allocations, 1);
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
    assert_eq!(session.scene().status().scene_metadata_upserts, 4);

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
    assert_eq!(session.scene().status().scene_metadata_upserts, 4);
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
        (baseline.0 + 1, baseline.1 * 2, baseline.2 + 1, baseline.3),
        "pure geometry retains the existing transform binding"
    );
    assert_eq!(session.scene().status().scene_metadata_upserts, 5);
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
    assert_eq!(session.scene().status().identity_allocations, 1);
    assert_eq!(session.scene().status().scene_metadata_upserts, 5);
}

#[test]
fn gpu_multiple_syncs_before_encode_retain_unrelated_identity_resources_and_final_visibility() {
    let Some(gpu) = gpu() else {
        return;
    };
    let mut controller = EditorController::new();
    for (offset, x) in [-0.6, 0.0, 0.6].into_iter().enumerate() {
        let mut mesh = triangle();
        mesh.positions = vec![[-0.2, -0.6, 0.5], [0.2, -0.6, 0.5], [0.0, 0.6, 0.5]];
        mesh.matrix[3][0] = x;
        register_mesh(&mut controller, OBJECT + offset as u128, mesh);
    }
    let mut controller = round_trip(&controller);
    let mut session = EditorSceneSession::new(&mut controller).unwrap();
    let mut ui = EditorDesktopUi::new(controller).unwrap();
    session.initialize(&gpu).unwrap();
    let output = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene-session retained identity proof"),
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
    let baseline = upload_counts(&session);
    assert_eq!((baseline.0, baseline.1, baseline.2), (3, 144, 3));
    assert_eq!(session.scene().status().identity_allocations, 3);
    assert_eq!(session.scene().status().scene_metadata_upserts, 3);
    for (x, id) in [(6.0, OBJECT), (16.0, OBJECT + 1), (25.0, OBJECT + 2)] {
        pick_session_at(&mut session, &gpu, &mut ui, [x, 16.0], Some(id));
    }

    // Publish several CPU deltas without encoding: the first mesh is edited
    // twice and hidden/restored, while a different object remains hidden.
    for (id, path, value) in [
        (OBJECT, "positions[0][0]", ActionValue::float(-0.25)),
        (OBJECT, "visible", ActionValue::bool(false)),
        (OBJECT + 1, "visible", ActionValue::bool(false)),
        (OBJECT, "positions[1][0]", ActionValue::float(0.25)),
        (OBJECT + 2, "matrix[3][0]", ActionValue::float(0.55)),
        (OBJECT, "visible", ActionValue::bool(true)),
    ] {
        edit(ui.controller_mut(), id, path, value);
        assert!(session.synchronize(ui.controller()).unwrap());
        assert_eq!(session.scene().status().identity_allocations, 3);
    }
    // Coalesce geometry AND matrix into one replacement, not the matrix-only
    // path. It must upload the changed transform now, and not again at encode.
    edit(
        ui.controller_mut(),
        OBJECT,
        "positions[2][0]",
        ActionValue::float(0.01),
    );
    edit(
        ui.controller_mut(),
        OBJECT,
        "matrix[3][0]",
        ActionValue::float(-0.55),
    );
    assert!(session.synchronize(ui.controller()).unwrap());
    assert_eq!(
        upload_counts(&session),
        (baseline.0, baseline.1, baseline.2 + 2, baseline.3),
        "only the independent and replacement matrix changes upload before encode"
    );
    assert_eq!(session.scene().status().scene_metadata_upserts, 3);
    assert!(!session
        .pointer(&gpu, &mut ui, [6.0, 16.0], SelectionMode::Replace)
        .unwrap());
    render_session(&mut session, &gpu, &mut ui, &output);
    let first_upload = upload_counts(&session);
    assert_eq!(
        first_upload,
        (baseline.0 + 1, baseline.1 + 48, baseline.2 + 2, baseline.3),
        "coalesced geometry edits upload only one mesh, without another transform or renderer"
    );
    assert_eq!(
        session.scene().status().scene_metadata_upserts,
        5,
        "one geometry and two visibility dirty indices form a two-object union, not three"
    );
    assert_eq!(
        session.scene().status().identity_allocations,
        3,
        "one mesh edit must not rebuild its or unrelated objects' ID bindings"
    );
    for (x, expected) in [(6.0, Some(OBJECT)), (16.0, None), (25.0, Some(OBJECT + 2))] {
        pick_session_at(&mut session, &gpu, &mut ui, [x, 16.0], expected);
    }

    // Two dirty meshes, plus visibility changes both before and after their
    // replacements. The last state must win when the dirty patch is applied.
    for (id, path, value) in [
        (OBJECT + 1, "visible", ActionValue::bool(true)),
        (OBJECT + 1, "positions[0][0]", ActionValue::float(-0.25)),
        (OBJECT + 2, "visible", ActionValue::bool(false)),
        (OBJECT, "positions[2][1]", ActionValue::float(0.7)),
        (OBJECT + 2, "visible", ActionValue::bool(true)),
        (OBJECT + 2, "visible", ActionValue::bool(false)),
    ] {
        edit(ui.controller_mut(), id, path, value);
        assert!(session.synchronize(ui.controller()).unwrap());
    }
    render_session(&mut session, &gpu, &mut ui, &output);
    let second_upload = upload_counts(&session);
    assert_eq!(
        second_upload,
        (
            first_upload.0 + 2,
            first_upload.1 + 96,
            first_upload.2,
            baseline.3
        )
    );
    assert_eq!(session.scene().status().scene_metadata_upserts, 8);
    assert_eq!(session.scene().status().identity_allocations, 3);
    for (x, expected) in [(6.0, Some(OBJECT)), (16.0, Some(OBJECT + 1)), (25.0, None)] {
        pick_session_at(&mut session, &gpu, &mut ui, [x, 16.0], expected);
    }
    render_session(&mut session, &gpu, &mut ui, &output);
    assert_eq!(upload_counts(&session), second_upload);
    assert_eq!(session.scene().status().identity_allocations, 3);
    assert_eq!(session.scene().status().scene_metadata_upserts, 8);

    // A geometry-only update in a three-object scene visits just one metadata
    // item, allocates no transform or identity, and retains unrelated entries.
    edit(
        ui.controller_mut(),
        OBJECT,
        "positions[1][0]",
        ActionValue::float(0.26),
    );
    render_session(&mut session, &gpu, &mut ui, &output);
    let third_upload = upload_counts(&session);
    assert_eq!(
        third_upload,
        (
            second_upload.0 + 1,
            second_upload.1 + 48,
            second_upload.2,
            baseline.3
        )
    );
    assert_eq!(session.scene().status().scene_metadata_upserts, 9);
    assert_eq!(session.scene().status().identity_allocations, 3);
    for (x, expected) in [(6.0, Some(OBJECT)), (16.0, Some(OBJECT + 1)), (25.0, None)] {
        pick_session_at(&mut session, &gpu, &mut ui, [x, 16.0], expected);
    }
    box_session(&mut session, &gpu, &mut ui, &[OBJECT, OBJECT + 1]);

    // Positive control: a fresh document upload really creates three identity
    // uniform/bind-group pairs (including the hidden object's retained entry).
    session.prepare_document(ui.controller()).unwrap();
    session.commit_document();
    render_session(&mut session, &gpu, &mut ui, &output);
    assert_eq!(session.scene().status().identity_allocations, 6);
    assert_eq!(session.scene().status().scene_metadata_upserts, 12);
    assert_eq!(
        upload_counts(&session),
        (
            third_upload.0 + 3,
            third_upload.1 + 144,
            third_upload.2 + 3,
            baseline.3 + 1
        )
    );
    for (x, expected) in [(6.0, Some(OBJECT)), (16.0, Some(OBJECT + 1)), (25.0, None)] {
        pick_session_at(&mut session, &gpu, &mut ui, [x, 16.0], expected);
    }
}

#[test]
fn gpu_deferred_dirty_meshes_attach_and_resume_with_final_transforms_and_visibility() {
    let Some(gpu) = gpu() else { return };
    let mut controller = EditorController::new();
    for (offset, x) in [-0.5, 0.5].into_iter().enumerate() {
        let mut mesh = triangle();
        mesh.positions = vec![[-0.3, -0.6, 0.5], [0.3, -0.6, 0.5], [0.0, 0.6, 0.5]];
        mesh.matrix[3][0] = x;
        register_mesh(&mut controller, OBJECT + offset as u128, mesh);
    }
    let mut controller = round_trip(&controller);
    let mut session = EditorSceneSession::new(&mut controller).unwrap();
    let mut ui = EditorDesktopUi::new(controller).unwrap();

    // Pending geometry/property deltas exist before any native renderer does.
    for (id, path, value) in [
        (OBJECT, "positions[0][0]", ActionValue::float(-0.35)),
        (OBJECT, "matrix[3][0]", ActionValue::float(-0.45)),
        (OBJECT, "visible", ActionValue::bool(false)),
    ] {
        edit(ui.controller_mut(), id, path, value);
    }
    assert!(session.synchronize(ui.controller()).unwrap());
    for (id, visible) in [(OBJECT, true), (OBJECT + 1, false)] {
        edit(
            ui.controller_mut(),
            id,
            "visible",
            ActionValue::bool(visible),
        );
        assert!(session.synchronize(ui.controller()).unwrap());
    }
    assert_eq!(upload_counts(&session), (0, 0, 0, 0));
    assert_eq!(session.scene().status().scene_metadata_upserts, 0);
    assert_eq!(session.scene().status().identity_allocations, 0);

    session.initialize(&gpu).unwrap();
    let output = session_output(&gpu, "scene-session deferred dirty recovery proof");
    render_session(&mut session, &gpu, &mut ui, &output);
    let attached = upload_counts(&session);
    assert_eq!((attached.0, attached.1, attached.2), (2, 96, 2));
    assert_eq!(session.scene().status().scene_metadata_upserts, 2);
    assert_eq!(session.scene().status().identity_allocations, 2);
    pick_session_at(&mut session, &gpu, &mut ui, [9.0, 16.0], Some(OBJECT));
    pick_session_at(&mut session, &gpu, &mut ui, [23.0, 16.0], None);
    box_session(&mut session, &gpu, &mut ui, &[OBJECT]);

    // Recovery must populate all resources, not just the dirty union, including
    // the hidden unchanged mesh. Several suspended syncs leave only final data.
    session.suspend();
    for (id, path, value) in [
        (OBJECT + 1, "positions[0][0]", ActionValue::float(-0.35)),
        (OBJECT + 1, "matrix[3][0]", ActionValue::float(0.45)),
        (OBJECT + 1, "visible", ActionValue::bool(true)),
    ] {
        edit(ui.controller_mut(), id, path, value);
    }
    assert!(session.synchronize(ui.controller()).unwrap());
    for (id, visible) in [(OBJECT, false), (OBJECT + 1, false), (OBJECT + 1, true)] {
        edit(
            ui.controller_mut(),
            id,
            "visible",
            ActionValue::bool(visible),
        );
        assert!(session.synchronize(ui.controller()).unwrap());
    }
    assert_eq!(upload_counts(&session), attached);
    assert_eq!(session.scene().status().scene_metadata_upserts, 2);
    session.initialize(&gpu).unwrap();
    render_session(&mut session, &gpu, &mut ui, &output);
    let resumed = upload_counts(&session);
    assert_eq!(resumed, (4, 192, 4, attached.3 + 2));
    assert_eq!(session.scene().status().scene_metadata_upserts, 4);
    assert_eq!(session.scene().status().identity_allocations, 4);
    pick_session_at(&mut session, &gpu, &mut ui, [9.0, 16.0], None);
    pick_session_at(&mut session, &gpu, &mut ui, [23.0, 16.0], Some(OBJECT + 1));
    box_session(&mut session, &gpu, &mut ui, &[OBJECT + 1]);

    // One replacement carries both matrix and geometry. Actual picks distinguish
    // its new binding from the old transform; encode must not upload it twice.
    for (path, value) in [
        ("positions[1][0]", ActionValue::float(0.35)),
        ("matrix[3][0]", ActionValue::float(0.0)),
        ("visible", ActionValue::bool(false)),
    ] {
        edit(ui.controller_mut(), OBJECT + 1, path, value);
    }
    assert!(session.synchronize(ui.controller()).unwrap());
    assert_eq!(
        upload_counts(&session),
        (resumed.0, resumed.1, resumed.2 + 1, resumed.3)
    );
    edit(
        ui.controller_mut(),
        OBJECT + 1,
        "visible",
        ActionValue::bool(true),
    );
    assert!(session.synchronize(ui.controller()).unwrap());
    render_session(&mut session, &gpu, &mut ui, &output);
    let retained = upload_counts(&session);
    assert_eq!(retained, (5, 240, 5, resumed.3));
    assert_eq!(session.scene().status().scene_metadata_upserts, 5);
    assert_eq!(session.scene().status().identity_allocations, 4);
    pick_session_at(&mut session, &gpu, &mut ui, [16.0, 16.0], Some(OBJECT + 1));
    pick_session_at(&mut session, &gpu, &mut ui, [23.0, 16.0], None);
    box_session(&mut session, &gpu, &mut ui, &[OBJECT + 1]);
    render_session(&mut session, &gpu, &mut ui, &output);
    assert_eq!(upload_counts(&session), retained);
    assert_eq!(session.scene().status().scene_metadata_upserts, 5);
}
