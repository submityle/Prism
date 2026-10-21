//! Generic codec and real production-provider persistence acceptance tests.
//! No build is needed to create fixtures; all filesystem paths are private
//! temporary directories, and production writes go through the world driver.

extern crate alloc;

#[cfg(feature = "desktop")]
#[path = "../examples/native_gizmo/support.rs"]
mod gizmo_support;

use alloc::{sync::Arc, task::Wake};
use std::{
    fs, io,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    task::{Context, Poll, Waker},
    thread,
    time::{Duration, Instant},
};

use prism_asset::StableGuid;
use prism_ecs::partition::cell::CellCoord;
use prism_editor_app::{
    ControllerLimits, DecodedDocument, DesktopKey, DocumentCodec, DocumentError,
    DocumentFileAdapter, DocumentIoEvent, DocumentLimits, DocumentLoadReceipt, DocumentSaveReceipt,
    DocumentSchemaCatalog, EditorController, EditorDesktopUi,
};
use prism_editor_core::SelectionItem;
use prism_editor_panels::{OutlinerId, OutlinerNode};
use prism_editor_reflect::FieldPath;
use prism_editor_ui::{ActionValue, EditorAction, SelectionMode};
use prism_editor_world::{
    CellLoadRequest, EditorWorld, FileCellIoError, FileCellIoLimits, FileCellIoProvider, IoFailure,
    ResidencyDemand,
};
use prism_reflect::{FieldMetadata, Reflect, TypeMetadata, TypeRegistry};
use prism_ui_render_backend::draw::DrawCommand;

#[derive(Clone, Debug, PartialEq, Reflect)]
struct Nested {
    title: String,
    values: Vec<i32>,
}

#[derive(Clone, Debug, PartialEq, Reflect)]
struct Model {
    count: i32,
    nested: Nested,
    enabled: bool,
}

#[derive(Clone, Debug, PartialEq, Reflect)]
struct Other {
    serial: u128,
    label: String,
}

fn model(count: i32) -> Model {
    Model {
        count,
        nested: Nested {
            title: "场景/船".into(),
            values: vec![i32::MIN, 0, i32::MAX],
        },
        enabled: true,
    }
}

fn registry() -> TypeRegistry {
    let mut registry = TypeRegistry::new();
    registry.register::<Model>();
    registry.register::<Nested>();
    registry.register::<Vec<i32>>();
    registry.register_type_data::<Model, _>(
        TypeMetadata::new().with_field(FieldMetadata::new("count").with_range(0.0, 100.0)),
    );
    registry
}

fn other_registry() -> TypeRegistry {
    let mut registry = TypeRegistry::new();
    registry.register::<Other>();
    registry
}

fn catalog() -> DocumentSchemaCatalog {
    let mut catalog = DocumentSchemaCatalog::new();
    catalog
        .register(std::any::type_name::<Model>(), 7, registry)
        .unwrap();
    catalog
        .register(std::any::type_name::<Other>(), 3, other_registry)
        .unwrap();
    catalog
}

fn items() -> Vec<SelectionItem> {
    vec![
        SelectionItem::entity(u128::MAX),
        SelectionItem::asset(u128::MAX),
        SelectionItem::sub_object(u128::MAX, "../领域/part", u32::MAX),
        SelectionItem::sub_object(u128::MAX, "../领域/part", 0),
    ]
}

fn controller() -> EditorController {
    let mut controller = EditorController::new();
    let mut group = OutlinerNode::new(OutlinerId::from_raw(100), None, "组/全体");
    group.set_expanded(false);
    controller.register_outliner_node(group).unwrap();
    for (index, item) in items().into_iter().enumerate() {
        controller
            .register_object(item.clone(), Box::new(model(index as i32)), registry())
            .unwrap();
        let mut row = OutlinerNode::new(
            OutlinerId::from_raw(index as u128 + 101),
            Some(OutlinerId::from_raw(100)),
            format!("物件 {index}"),
        );
        row.set_selection(item);
        row.set_visible(index != 1);
        controller.register_outliner_node(row).unwrap();
    }
    // Objects absent from both selection and Outliner must still be persisted.
    controller
        .register_object(
            SelectionItem::asset(99),
            Box::new(Other {
                serial: u128::MAX - 1,
                label: "orphan".into(),
            }),
            other_registry(),
        )
        .unwrap();
    controller.select(items()[0].clone()).unwrap();
    controller
}

fn snapshot(controller: &EditorController, item: &SelectionItem) -> Model {
    let mut concrete = model(0);
    concrete
        .apply(&*controller.snapshot(item).unwrap())
        .unwrap();
    concrete
}

fn edit(controller: &mut EditorController, count: i64) {
    controller.select(items()[0].clone()).unwrap();
    controller
        .execute(EditorAction::EditInspectorField {
            path: FieldPath::parse("count").unwrap(),
            value: ActionValue::integer(count),
        })
        .unwrap();
}

fn encode(controller: &EditorController) -> Vec<u8> {
    DocumentCodec::encode(controller, &catalog(), DocumentLimits::default()).unwrap()
}

fn decode(bytes: &[u8]) -> Result<DecodedDocument, DocumentError> {
    DocumentCodec::decode(
        bytes,
        &catalog(),
        DocumentLimits::default(),
        ControllerLimits::default(),
    )
}

#[test]
fn full_typed_identities_hidden_hierarchy_metadata_and_unselected_objects_roundtrip() {
    let mut original = controller();
    edit(&mut original, 42);
    let bytes = encode(&original);
    assert_eq!(&bytes[..8], b"PRISMDOC");
    assert_eq!(&bytes[8..12], &1_u32.to_le_bytes());
    assert_eq!(encode(&original), bytes);
    let decoded = decode(&bytes).unwrap();
    assert_eq!(decoded.source_model_revision(), original.model_revision());
    let restored = decoded.controller();
    assert!(!restored.is_dirty());
    assert_eq!(restored.model_revision(), 0);
    assert_eq!(restored.document_objects().len(), 5);
    for item in items() {
        assert_eq!(snapshot(restored, &item), snapshot(&original, &item));
    }
    let mut other = Other {
        serial: 0,
        label: String::new(),
    };
    other
        .apply(&*restored.snapshot(&SelectionItem::asset(99)).unwrap())
        .unwrap();
    assert_eq!(
        other,
        Other {
            serial: u128::MAX - 1,
            label: "orphan".into()
        }
    );
    for (before, after) in original
        .document_outliner_nodes()
        .iter()
        .zip(restored.document_outliner_nodes())
    {
        assert_eq!(before.id(), after.id());
        assert_eq!(before.parent(), after.parent());
        assert_eq!(before.label(), after.label());
        assert_eq!(before.selection(), after.selection());
        assert_eq!(before.is_expanded(), after.is_expanded());
        assert_eq!(before.is_visible(), after.is_visible());
    }
    assert_eq!(
        restored
            .outliner()
            .node(OutlinerId::from_raw(100))
            .unwrap()
            .children()
            .len(),
        4
    );
    let mut restored = decoded.into_controller();
    restored.select(items()[0].clone()).unwrap();
    assert!(
        restored
            .execute(EditorAction::EditInspectorField {
                path: FieldPath::parse("count").unwrap(),
                value: ActionValue::integer(101)
            })
            .is_err(),
        "registry metadata is restored, not just the root type"
    );
}

#[test]
fn truncation_trailing_bytes_bad_tags_utf8_counts_and_version_are_rejected() {
    let bytes = encode(&controller());
    for prefix in 0..bytes.len() {
        assert!(
            decode(&bytes[..prefix]).is_err(),
            "accepted truncated prefix {prefix}"
        );
    }
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(matches!(
        decode(&trailing),
        Err(DocumentError::BadFormat("trailing bytes"))
    ));
    for offset in [0, 8, 28] {
        let mut bad = bytes.clone();
        bad[offset] = 255;
        assert!(decode(&bad).is_err());
    }
    let mut count = bytes.clone();
    count[20..24].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(matches!(
        decode(&count),
        Err(DocumentError::Limit("objects"))
    ));
    let mut utf8 = bytes;
    // First record is an Entity: 17-byte tagged identity, u32 path length.
    utf8[28 + 17 + 4] = 255;
    assert!(matches!(
        decode(&utf8),
        Err(DocumentError::BadFormat("UTF-8"))
    ));
}

fn one_object() -> Vec<u8> {
    let mut controller = EditorController::new();
    controller
        .register_object(SelectionItem::entity(1), Box::new(model(1)), registry())
        .unwrap();
    encode(&controller)
}

#[test]
fn duplicate_object_identity_unknown_schema_and_corrupt_reflection_are_rejected() {
    let single = one_object();
    let mut duplicate = single.clone();
    duplicate[20..24].copy_from_slice(&2_u32.to_le_bytes());
    duplicate.extend_from_slice(&single[28..]);
    assert!(matches!(
        decode(&duplicate),
        Err(DocumentError::DuplicateObject(_))
    ));
    assert!(matches!(
        DocumentCodec::decode(
            &single,
            &DocumentSchemaCatalog::new(),
            DocumentLimits::default(),
            ControllerLimits::default()
        ),
        Err(DocumentError::UnknownSchema { .. })
    ));
    let path_len = u32::from_le_bytes(single[45..49].try_into().unwrap()) as usize;
    let version_at = 49 + path_len;
    let mut version = single.clone();
    version[version_at..version_at + 4].copy_from_slice(&99_u32.to_le_bytes());
    assert!(matches!(
        decode(&version),
        Err(DocumentError::UnknownSchema { version: 99, .. })
    ));
    let mut binary = single;
    binary[version_at + 8] = 255;
    assert!(matches!(decode(&binary), Err(DocumentError::Reflection(_))));
}

#[test]
fn incomplete_factory_and_changed_validation_metadata_cannot_restore_partial_controller() {
    let bytes = one_object();
    let mut incomplete = DocumentSchemaCatalog::new();
    incomplete
        .register(std::any::type_name::<Model>(), 7, || {
            let mut registry = TypeRegistry::new();
            registry.register::<Model>();
            registry
        })
        .unwrap();
    assert!(matches!(
        DocumentCodec::decode(
            &bytes,
            &incomplete,
            DocumentLimits::default(),
            ControllerLimits::default()
        ),
        Err(DocumentError::Reflection(_))
    ));
    let mut stricter = DocumentSchemaCatalog::new();
    stricter
        .register(std::any::type_name::<Model>(), 7, || {
            let mut registry = registry();
            registry.register_type_data::<Model, _>(
                TypeMetadata::new().with_field(FieldMetadata::new("count").with_range(20.0, 30.0)),
            );
            registry
        })
        .unwrap();
    assert!(matches!(
        DocumentCodec::decode(
            &bytes,
            &stricter,
            DocumentLimits::default(),
            ControllerLimits::default()
        ),
        Err(DocumentError::Controller(_))
    ));
    let mut duplicates = catalog();
    assert!(matches!(
        duplicates.register(std::any::type_name::<Model>(), 8, registry),
        Err(DocumentError::DuplicateSchema(_))
    ));
}

#[test]
fn byte_object_node_depth_string_collection_and_controller_limits_are_enforced() {
    let controller = controller();
    let bytes = encode(&controller);
    for limits in [
        DocumentLimits {
            max_document_bytes: 64,
            max_object_bytes: 32,
            max_string_bytes: 32,
            ..DocumentLimits::default()
        },
        DocumentLimits {
            max_objects: 1,
            ..DocumentLimits::default()
        },
        DocumentLimits {
            max_outliner_nodes: 1,
            ..DocumentLimits::default()
        },
        DocumentLimits {
            max_string_bytes: 2,
            ..DocumentLimits::default()
        },
        DocumentLimits {
            max_object_bytes: 32,
            ..DocumentLimits::default()
        },
        DocumentLimits {
            max_reflect_nodes: 1,
            ..DocumentLimits::default()
        },
        DocumentLimits {
            max_reflect_depth: 1,
            ..DocumentLimits::default()
        },
        DocumentLimits {
            max_collection_len: 1,
            ..DocumentLimits::default()
        },
    ] {
        assert!(DocumentCodec::encode(&controller, &catalog(), limits).is_err());
        assert!(
            DocumentCodec::decode(&bytes, &catalog(), limits, ControllerLimits::default()).is_err()
        );
    }
    assert!(matches!(
        DocumentCodec::decode(
            &bytes,
            &catalog(),
            DocumentLimits {
                max_reflect_depth: 129,
                ..DocumentLimits::default()
            },
            ControllerLimits::default()
        ),
        Err(DocumentError::InvalidLimits)
    ));
    assert!(DocumentCodec::decode(
        &bytes,
        &catalog(),
        DocumentLimits::default(),
        ControllerLimits {
            max_objects: 1,
            ..ControllerLimits::default()
        }
    )
    .is_err());
}

fn row_only() -> Vec<u8> {
    let mut controller = EditorController::new();
    controller
        .register_outliner_node(OutlinerNode::new(OutlinerId::from_raw(1), None, "a"))
        .unwrap();
    encode(&controller)
}

#[test]
fn duplicate_rows_missing_parents_cycles_unknown_objects_and_bad_booleans_are_rejected() {
    let single = row_only();
    let mut duplicate = single.clone();
    duplicate[24..28].copy_from_slice(&2_u32.to_le_bytes());
    duplicate.extend_from_slice(&single[28..]);
    assert!(matches!(
        decode(&duplicate),
        Err(DocumentError::DuplicateOutliner(_))
    ));
    for parent in [1_u128, 999] {
        let mut parented = single.clone();
        parented[44] = 1;
        parented.splice(45..45, parent.to_le_bytes());
        assert!(matches!(
            decode(&parented),
            Err(DocumentError::BadFormat("Outliner parent order/cycle"))
        ));
    }
    let mut missing = single.clone();
    // row offset 28: u128 id, bool parent, u32 label length, "a".
    missing[50] = 1;
    let mut identity = vec![0];
    identity.extend_from_slice(&1_u128.to_le_bytes());
    missing.splice(51..51, identity);
    assert!(matches!(
        decode(&missing),
        Err(DocumentError::BadFormat("Outliner object missing"))
    ));
    for offset in [44, 50, 51, 52] {
        let mut bad = single.clone();
        bad[offset] = 2;
        assert!(matches!(
            decode(&bad),
            Err(DocumentError::BadFormat("boolean"))
        ));
    }
}

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "prism-document-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("private test directory: {error}"),
            }
        }
    }
    fn provider(&self) -> FileCellIoProvider {
        FileCellIoProvider::new(
            &self.0,
            FileCellIoLimits::new(DocumentLimits::default().max_document_bytes, 2).unwrap(),
        )
        .unwrap()
    }
    fn adapter(&self) -> DocumentFileAdapter {
        DocumentFileAdapter::new(self.provider(), request(), DocumentLimits::default()).unwrap()
    }
    fn file(&self) -> PathBuf {
        self.0.join(request().file().relative_path())
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove only private test directory");
    }
}

fn request() -> CellLoadRequest {
    let coord = CellCoord::new(-12, 0, i32::MAX);
    let mut world = EditorWorld::new(StableGuid::from_path("document/acceptance")).unwrap();
    world.add_cell(coord, []).unwrap();
    let file = world.cell(coord).unwrap().file().clone();
    let mut demand = ResidencyDemand::new();
    demand.require(coord);
    let plan = world.plan(&demand).unwrap();
    let tokens = world.begin_plan(&plan).unwrap();
    CellLoadRequest::new(
        tokens[0],
        file,
        DocumentLimits::default().max_document_bytes,
    )
}

struct Notify(thread::Thread);
impl Wake for Notify {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

fn event(adapter: &mut DocumentFileAdapter) -> DocumentIoEvent {
    let waker = Waker::from(Arc::new(Notify(thread::current())));
    let mut context = Context::from_waker(&waker);
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match adapter.poll(&mut context) {
            Poll::Ready(Some(event)) => return event,
            Poll::Ready(None) => panic!("expected accepted operation"),
            Poll::Pending => {}
        }
        assert!(Instant::now() < deadline, "document I/O timeout");
        thread::park_timeout(Duration::from_millis(5));
    }
}

fn workers_finished(provider: &FileCellIoProvider) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while provider.outstanding_operations() != 0 {
        assert!(
            Instant::now() < deadline,
            "cancelled worker cleanup timeout"
        );
        thread::park_timeout(Duration::from_millis(1));
    }
}

fn glyphs(ui: &EditorDesktopUi) -> Vec<String> {
    ui.draw_list()
        .commands()
        .iter()
        .filter_map(|command| match command {
            DrawCommand::Glyph(glyph) => Some(ui.draw_list().text(glyph.text).to_owned()),
            _ => None,
        })
        .collect()
}

fn focus_glyph(ui: &mut EditorDesktopUi, value: &str) {
    let rect = ui
        .draw_list()
        .commands()
        .iter()
        .find_map(|command| match command {
            DrawCommand::Glyph(glyph) if ui.draw_list().text(glyph.text) == value => {
                Some(glyph.rect)
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing value glyph {value}"));
    assert!(ui
        .click(
            rect.location.x + rect.size.width * 0.5,
            rect.location.y + rect.size.height * 0.5,
            SelectionMode::Replace,
        )
        .unwrap());
}

fn saved(adapter: &mut DocumentFileAdapter) -> DocumentSaveReceipt {
    match event(adapter) {
        DocumentIoEvent::Saved(receipt) => receipt,
        DocumentIoEvent::Failed { error, .. } => panic!("save failed: {error}"),
        _ => panic!("expected saved"),
    }
}

fn loaded(adapter: &mut DocumentFileAdapter) -> DocumentLoadReceipt {
    match event(adapter) {
        DocumentIoEvent::Loaded(receipt) => receipt,
        DocumentIoEvent::Failed { error, .. } => panic!("load failed: {error}"),
        _ => panic!("expected loaded"),
    }
}

fn seed(temp: &Temp, controller: &mut EditorController) {
    let mut adapter = temp.adapter();
    adapter.submit_save(controller, &catalog()).unwrap();
    saved(&mut adapter)
        .acknowledge(controller, &catalog())
        .unwrap();
}

fn open(adapter: &mut DocumentFileAdapter, controller: &mut EditorController) {
    adapter.submit_load(controller, &catalog()).unwrap();
    loaded(adapter)
        .prepare(&catalog(), ControllerLimits::default())
        .unwrap()
        .install(adapter, controller, &catalog())
        .unwrap();
}

#[test]
fn real_file_save_reopen_load_prepare_and_atomic_controller_publication() {
    let temp = Temp::new();
    let mut original = controller();
    edit(&mut original, 42);
    let revision = original.model_revision();
    let mut writer = temp.adapter();
    let id = writer.submit_save(&original, &catalog()).unwrap();
    assert!(writer.is_busy());
    assert!(matches!(
        writer.submit_save(&original, &catalog()),
        Err(DocumentError::Busy)
    ));
    let receipt = saved(&mut writer);
    assert_eq!(receipt.request_id, id);
    assert_eq!(receipt.model_revision(), revision);
    assert_eq!(receipt.durable_revision, 0);
    assert!(
        original.is_dirty(),
        "poll must not mark clean without acknowledgment"
    );
    receipt.acknowledge(&mut original, &catalog()).unwrap();
    assert!(!original.is_dirty());
    assert_eq!(&fs::read(temp.file()).unwrap()[..8], b"PEWCELL\0");
    let mut reader = temp.adapter();
    let mut current = EditorController::new();
    reader.submit_load(&current, &catalog()).unwrap();
    let prepared = loaded(&mut reader)
        .prepare(&catalog(), ControllerLimits::default())
        .unwrap();
    assert_eq!(
        current.document_objects().len(),
        0,
        "candidate must not leak before publication"
    );
    assert_eq!(prepared.controller().document_objects().len(), 5);
    assert_eq!(prepared.source_model_revision(), revision);
    prepared
        .install(&mut reader, &mut current, &catalog())
        .unwrap();
    assert_eq!(snapshot(&current, &items()[0]).count, 42);
    assert!(!current.is_dirty());
    // Restarted controller revisions are not reused as disk revisions.
    edit(&mut current, 43);
    reader.submit_save(&current, &catalog()).unwrap();
    let receipt = saved(&mut reader);
    assert_eq!(receipt.durable_revision, 1);
    receipt.acknowledge(&mut current, &catalog()).unwrap();
}

#[cfg(feature = "desktop")]
#[test]
fn host_transform_batch_undo_redo_real_file_save_and_reload() {
    use prism_editor_app::SceneTransformAdapter;
    use prism_editor_viewport::TransformDelta;

    let temp = Temp::new();
    let (mut controller, _scene) = gizmo_support::scene().unwrap();
    let mut catalog = DocumentSchemaCatalog::new();
    catalog
        .register(
            std::any::type_name::<gizmo_support::ExampleTransform>(),
            1,
            gizmo_support::registry,
        )
        .unwrap();
    let matrix = |controller: &EditorController, n| {
        gizmo_support::read(controller.object_root(&SelectionItem::entity(n)).unwrap())
            .unwrap()
            .matrix
    };
    let original = [matrix(&controller, 1), matrix(&controller, 2)];
    let mut edits = Vec::new();
    for n in 1..=2 {
        let item = SelectionItem::entity(n);
        edits.extend(
            gizmo_support::ExampleBinding
                .edits(
                    &item,
                    controller.object_root(&item).unwrap(),
                    TransformDelta {
                        translation_milliunits: [200, -100, 0],
                        scale_milliunits: [1000; 3],
                        ..Default::default()
                    },
                )
                .unwrap(),
        );
    }
    let prepared = controller.prepare_edits(edits).unwrap();
    controller.commit_edits(prepared).unwrap();
    assert_eq!(controller.document_history_lengths(), (1, 0));
    let transformed = [matrix(&controller, 1), matrix(&controller, 2)];
    assert_ne!(transformed, original);
    controller.clear_selection().unwrap();
    controller.execute(EditorAction::Undo).unwrap();
    assert_eq!([matrix(&controller, 1), matrix(&controller, 2)], original);
    controller.execute(EditorAction::Redo).unwrap();
    assert_eq!(
        [matrix(&controller, 1), matrix(&controller, 2)],
        transformed
    );
    let mut writer = temp.adapter();
    writer.submit_save(&controller, &catalog).unwrap();
    saved(&mut writer)
        .acknowledge(&mut controller, &catalog)
        .unwrap();
    assert!(!controller.is_dirty());
    let mut restored = EditorController::new();
    let mut reader = temp.adapter();
    reader.submit_load(&restored, &catalog).unwrap();
    loaded(&mut reader)
        .prepare(&catalog, ControllerLimits::default())
        .unwrap()
        .install(&mut reader, &mut restored, &catalog)
        .unwrap();
    assert_eq!([matrix(&restored, 1), matrix(&restored, 2)], transformed);
    assert_eq!(restored.document_history_lengths(), (0, 0));
    assert!(!restored.is_dirty());
}

#[test]
fn old_save_completion_and_same_revision_replacement_cannot_clear_new_content() {
    let temp = Temp::new();
    let mut controller = controller();
    let mut adapter = temp.adapter();
    edit(&mut controller, 40);
    adapter.submit_save(&controller, &catalog()).unwrap();
    edit(&mut controller, 41);
    let receipt = saved(&mut adapter);
    assert!(matches!(
        receipt.acknowledge(&mut controller, &catalog()),
        Err(DocumentError::StaleRevision { .. })
    ));
    assert!(controller.is_dirty());
    assert_eq!(snapshot(&controller, &items()[0]).count, 41);
    adapter.submit_save(&controller, &catalog()).unwrap();
    let receipt = saved(&mut adapter);
    let mut replacement = self::controller();
    edit(&mut replacement, 20);
    edit(&mut replacement, 21);
    assert_eq!(replacement.model_revision(), receipt.model_revision());
    assert!(matches!(
        receipt.acknowledge(&mut replacement, &catalog()),
        Err(DocumentError::SnapshotChanged)
    ));
    assert!(replacement.is_dirty());
    receipt.acknowledge(&mut controller, &catalog()).unwrap();
    assert!(!controller.is_dirty());
}

#[test]
fn load_receipt_checks_intervening_edits_registration_and_adapter_generation() {
    let temp = Temp::new();
    let mut initial = controller();
    seed(&temp, &mut initial);
    let provider = temp.provider();
    let mut adapter =
        DocumentFileAdapter::new(provider.clone(), request(), DocumentLimits::default()).unwrap();
    let mut current = controller();
    adapter.submit_load(&current, &catalog()).unwrap();
    let prepared = loaded(&mut adapter)
        .prepare(&catalog(), ControllerLimits::default())
        .unwrap();
    edit(&mut current, 55);
    let before = encode(&current);
    assert!(matches!(
        prepared.install(&mut adapter, &mut current, &catalog()),
        Err(DocumentError::StaleRevision { .. })
    ));
    assert_eq!(encode(&current), before);
    assert_eq!(adapter.durable_revision(), None);

    adapter.submit_load(&current, &catalog()).unwrap();
    let prepared = loaded(&mut adapter)
        .prepare(&catalog(), ControllerLimits::default())
        .unwrap();
    current
        .register_object(SelectionItem::entity(888), Box::new(model(1)), registry())
        .unwrap();
    assert!(matches!(
        prepared.install(&mut adapter, &mut current, &catalog()),
        Err(DocumentError::SnapshotChanged)
    ));

    adapter.submit_load(&current, &catalog()).unwrap();
    let prepared = loaded(&mut adapter)
        .prepare(&catalog(), ControllerLimits::default())
        .unwrap();
    let mut wrong_adapter = temp.adapter();
    assert!(matches!(
        prepared.install(&mut wrong_adapter, &mut current, &catalog()),
        Err(DocumentError::SupersededReceipt)
    ));

    adapter.submit_load(&current, &catalog()).unwrap();
    let prepared = loaded(&mut adapter)
        .prepare(&catalog(), ControllerLimits::default())
        .unwrap();
    let id = adapter.submit_load(&current, &catalog()).unwrap();
    assert!(matches!(
        prepared.install(&mut adapter, &mut current, &catalog()),
        Err(DocumentError::SupersededReceipt)
    ));
    assert!(adapter.cancel(id));
    assert!(matches!(
        event(&mut adapter),
        DocumentIoEvent::Cancelled { .. }
    ));
    workers_finished(&provider);
}

#[test]
fn bad_schema_corrupt_file_and_missing_file_never_replace_live_controller() {
    let temp = Temp::new();
    let mut initial = controller();
    seed(&temp, &mut initial);
    let mut current = controller();
    edit(&mut current, 66);
    let before = encode(&current);
    let mut adapter = temp.adapter();
    adapter.submit_load(&current, &catalog()).unwrap();
    assert!(matches!(
        loaded(&mut adapter).prepare(&DocumentSchemaCatalog::new(), ControllerLimits::default()),
        Err(DocumentError::UnknownSchema { .. })
    ));
    assert_eq!(encode(&current), before);
    assert!(current.is_dirty());
    let mut bytes = fs::read(temp.file()).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    fs::write(temp.file(), bytes).unwrap();
    adapter.submit_load(&current, &catalog()).unwrap();
    assert!(matches!(
        event(&mut adapter),
        DocumentIoEvent::Failed {
            error: IoFailure::Provider(FileCellIoError::Corrupt(_)),
            ..
        }
    ));
    assert_eq!(encode(&current), before);
    fs::remove_file(temp.file()).unwrap();
    adapter.submit_load(&current, &catalog()).unwrap();
    assert!(matches!(
        event(&mut adapter),
        DocumentIoEvent::Failed {
            error: IoFailure::Provider(FileCellIoError::Io { .. }),
            ..
        }
    ));
    assert_eq!(encode(&current), before);
}

#[test]
fn two_editor_sessions_conflict_on_stale_durable_base_without_losing_winner() {
    let temp = Temp::new();
    let mut initial = controller();
    seed(&temp, &mut initial);
    let mut left = temp.adapter();
    let mut right = temp.adapter();
    let mut left_controller = EditorController::new();
    let mut right_controller = EditorController::new();
    open(&mut left, &mut left_controller);
    open(&mut right, &mut right_controller);
    edit(&mut left_controller, 70);
    edit(&mut right_controller, 80);
    left.submit_save(&left_controller, &catalog()).unwrap();
    saved(&mut left)
        .acknowledge(&mut left_controller, &catalog())
        .unwrap();
    let winner = fs::read(temp.file()).unwrap();
    right.submit_save(&right_controller, &catalog()).unwrap();
    assert!(matches!(
        event(&mut right),
        DocumentIoEvent::Failed {
            error: IoFailure::Provider(FileCellIoError::RevisionConflict {
                current: Some(1),
                ..
            }),
            ..
        }
    ));
    assert!(right_controller.is_dirty());
    assert!(right.reconciliation_required());
    assert!(matches!(
        right.submit_save(&right_controller, &catalog()),
        Err(DocumentError::ReconciliationRequired)
    ));
    assert_eq!(fs::read(temp.file()).unwrap(), winner);
    // Explicit reload is a user/host decision, never automatic conflict loss.
    open(&mut right, &mut right_controller);
    assert_eq!(snapshot(&right_controller, &items()[0]).count, 70);
    assert!(!right.reconciliation_required());
    edit(&mut right_controller, 81);
    right.submit_save(&right_controller, &catalog()).unwrap();
    assert_eq!(saved(&mut right).durable_revision, 2);
}

#[test]
fn uninitialized_adapter_cannot_overwrite_existing_document_or_reset_durable_counter() {
    let temp = Temp::new();
    let mut initial = controller();
    seed(&temp, &mut initial);
    let original = fs::read(temp.file()).unwrap();
    let mut adapter = temp.adapter();
    edit(&mut initial, 99);
    adapter.submit_save(&initial, &catalog()).unwrap();
    assert!(matches!(
        event(&mut adapter),
        DocumentIoEvent::Failed {
            error: IoFailure::Provider(FileCellIoError::RevisionConflict {
                current: Some(0),
                ..
            }),
            ..
        }
    ));
    assert_eq!(fs::read(temp.file()).unwrap(), original);
}

#[test]
fn failed_atomic_stage_preserves_previous_file_and_identical_retry_is_revision_fenced() {
    let temp = Temp::new();
    let mut controller = controller();
    seed(&temp, &mut controller);
    let original = fs::read(temp.file()).unwrap();
    let mut adapter = temp.adapter();
    open(&mut adapter, &mut controller);
    edit(&mut controller, 30);
    let stage = temp.file().with_extension("prismcell.pending");
    fs::create_dir(&stage).unwrap();
    adapter.submit_save(&controller, &catalog()).unwrap();
    assert!(matches!(
        event(&mut adapter),
        DocumentIoEvent::Failed {
            error: IoFailure::Provider(FileCellIoError::UnsafePath(_)),
            ..
        }
    ));
    assert_eq!(fs::read(temp.file()).unwrap(), original);
    assert!(controller.is_dirty());
    edit(&mut controller, 31);
    fs::remove_dir(stage).unwrap();
    adapter.retry_save().unwrap();
    let receipt = saved(&mut adapter);
    assert!(matches!(
        receipt.acknowledge(&mut controller, &catalog()),
        Err(DocumentError::StaleRevision { .. })
    ));
    assert!(controller.is_dirty());
    let mut reopened = temp.adapter();
    let mut loaded = EditorController::new();
    open(&mut reopened, &mut loaded);
    assert_eq!(
        snapshot(&loaded, &items()[0]).count,
        30,
        "retry must retain the original bytes"
    );
    adapter.submit_save(&controller, &catalog()).unwrap();
    saved(&mut adapter)
        .acknowledge(&mut controller, &catalog())
        .unwrap();
    assert!(!controller.is_dirty());
}

#[test]
fn cancelled_save_requires_reconciliation_and_retry_never_marks_clean_implicitly() {
    let temp = Temp::new();
    let mut controller = controller();
    edit(&mut controller, 25);
    let provider = temp.provider();
    let mut adapter =
        DocumentFileAdapter::new(provider.clone(), request(), DocumentLimits::default()).unwrap();
    let id = adapter.submit_save(&controller, &catalog()).unwrap();
    assert!(adapter.cancel(id));
    assert!(
        matches!(event(&mut adapter), DocumentIoEvent::Cancelled { request_id } if request_id == id)
    );
    assert!(controller.is_dirty());
    assert!(adapter.reconciliation_required());
    assert!(matches!(
        adapter.submit_save(&controller, &catalog()),
        Err(DocumentError::ReconciliationRequired)
    ));
    workers_finished(&provider);
    adapter.retry_save().unwrap();
    let receipt = saved(&mut adapter);
    assert!(controller.is_dirty());
    receipt.acknowledge(&mut controller, &catalog()).unwrap();
    assert!(!controller.is_dirty());
    assert!(!adapter.reconciliation_required());
    assert!(matches!(adapter.retry_save(), Err(DocumentError::NoRetry)));
}

#[test]
fn install_ui_preserves_size_and_publishes_only_after_receipt_guard() {
    let temp = Temp::new();
    let mut source = controller();
    seed(&temp, &mut source);

    let mut adapter = temp.adapter();
    let mut ui = EditorDesktopUi::new(EditorController::new()).unwrap();
    ui.resize(901.0, 607.0).unwrap();
    let size = ui.size();
    adapter.submit_load(ui.controller(), &catalog()).unwrap();
    let prepared = loaded(&mut adapter)
        .prepare(&catalog(), ControllerLimits::default())
        .unwrap();
    prepared
        .install_ui(&mut adapter, &mut ui, &catalog())
        .unwrap();
    assert_eq!(ui.size(), size);
    assert_eq!(ui.controller().document_objects().len(), 5);
    assert_eq!(adapter.durable_revision(), Some(0));

    let mut adapter = temp.adapter();
    let mut ui = EditorDesktopUi::new(controller()).unwrap();
    ui.resize(777.0, 555.0).unwrap();
    let before_size = ui.size();
    let before = encode(ui.controller());
    adapter.submit_load(ui.controller(), &catalog()).unwrap();
    let prepared = loaded(&mut adapter)
        .prepare(&catalog(), ControllerLimits::default())
        .unwrap();
    edit(ui.controller_mut(), 55);
    assert!(matches!(
        prepared.install_ui(&mut adapter, &mut ui, &catalog()),
        Err(DocumentError::StaleRevision { .. })
    ));
    assert_eq!(ui.size(), before_size);
    assert_eq!(snapshot(ui.controller(), &items()[0]).count, 55);
    assert_ne!(encode(ui.controller()), before);
    assert_eq!(adapter.durable_revision(), None);
}

#[test]
fn acknowledge_ui_marks_saved_without_dropping_text_or_ime_draft() {
    let temp = Temp::new();
    let mut model = controller();
    edit(&mut model, 42);
    let mut ui = EditorDesktopUi::new(model).unwrap();
    let mut adapter = temp.adapter();
    adapter.submit_save(ui.controller(), &catalog()).unwrap();
    let receipt = saved(&mut adapter);
    assert!(ui.controller().is_dirty());

    focus_glyph(&mut ui, "42");
    assert!(ui.has_uncommitted_edit());
    assert!(ui.text("43").unwrap());
    assert!(ui.preedit("拼音").unwrap());
    assert!(ui.has_uncommitted_edit());
    receipt.acknowledge_ui(&mut ui, &catalog()).unwrap();

    assert!(!ui.controller().is_dirty());
    assert!(ui.has_uncommitted_edit());
    assert!(glyphs(&ui).iter().any(|text| text == "43[拼音]"));
    assert_eq!(snapshot(ui.controller(), &items()[0]).count, 42);
    assert!(ui.key(DesktopKey::Escape).unwrap());
    assert!(!ui.has_uncommitted_edit());
    assert!(!ui.controller().is_dirty());

    edit(ui.controller_mut(), 44);
    assert!(matches!(
        receipt.acknowledge_ui(&mut ui, &catalog()),
        Err(DocumentError::StaleRevision { .. })
    ));
    assert!(ui.controller().is_dirty());
}

#[cfg(feature = "desktop")]
#[test]
fn native_document_load_rejects_unsaved_controller_changes_before_io() {
    use prism_editor_app::DesktopDocumentSession;

    let temp = Temp::new();
    let mut ui = EditorDesktopUi::new(controller()).unwrap();
    edit(ui.controller_mut(), 42);
    let mut session =
        DesktopDocumentSession::new(temp.adapter(), catalog(), ControllerLimits::default());

    assert!(matches!(
        session.load(&ui),
        Err(DocumentError::UnsavedChanges)
    ));
    assert!(!session.is_busy());
    assert!(session.last_request().is_none());
    assert!(!temp.file().exists());
    assert!(ui.controller().is_dirty());
    assert_eq!(snapshot(ui.controller(), &items()[0]).count, 42);
}

#[cfg(feature = "desktop")]
#[test]
fn native_document_cancellation_is_idempotent_until_terminal_poll() {
    use prism_editor_app::{
        DesktopDocumentCancelOutcome, DesktopDocumentSession, DesktopScene, DesktopWork,
    };

    let temp = Temp::new();
    let provider = temp.provider();
    let adapter =
        DocumentFileAdapter::new(provider.clone(), request(), DocumentLimits::default()).unwrap();
    let mut ui = EditorDesktopUi::new(controller()).unwrap();
    edit(ui.controller_mut(), 42);
    let mut session = DesktopDocumentSession::new(adapter, catalog(), ControllerLimits::default());
    let request_id = session.save(&mut ui).unwrap();

    assert_eq!(session.last_request(), Some(request_id));
    assert_eq!(session.cancel(), DesktopDocumentCancelOutcome::Accepted);
    assert!(session.is_busy());
    assert_eq!(
        session.cancel(),
        DesktopDocumentCancelOutcome::AwaitingTerminal
    );

    let mut scene = DesktopScene::new(Vec::new()).unwrap();
    let waker = Waker::from(Arc::new(Notify(thread::current())));
    let mut context = Context::from_waker(&waker);
    assert_eq!(
        session.poll(&mut context, &mut ui, &mut scene).unwrap(),
        Poll::Ready(DesktopWork {
            redraw: true,
            pending: false,
        })
    );
    assert!(!session.is_busy());
    assert!(session.last_request().is_none());
    assert!(session.reconciliation_required());
    assert!(ui.controller().is_dirty());
    assert_eq!(session.cancel(), DesktopDocumentCancelOutcome::Idle);

    let retry_id = session.retry_save().unwrap();
    assert_eq!(session.last_request(), Some(retry_id));
    assert!(session.is_busy());
    let deadline = Instant::now() + Duration::from_secs(10);
    while session.is_busy() {
        assert!(Instant::now() < deadline, "native retry save deadline");
        let _ = session.poll(&mut context, &mut ui, &mut scene).unwrap();
        thread::yield_now();
    }
    assert!(!session.reconciliation_required());
    assert!(!ui.controller().is_dirty());
    assert!(temp.file().exists());
    workers_finished(&provider);
}

#[cfg(feature = "desktop")]
#[test]
fn native_document_save_runs_content_gate_before_real_io_and_revalidates() {
    use prism_editor_app::{DesktopDocumentSession, DesktopScene};
    use prism_editor_content::{
        validation::{
            AssetValidationRule, ValidationContext, ValidationLimits, ValidationPolicy,
            ValidationRegistry, ValidationSink,
        },
        AssetId, AssetKind, AssetRecord, ContentCatalog, DiagnosticSeverity,
    };
    struct Reject;
    impl AssetValidationRule for Reject {
        fn id(&self) -> &str {
            "host.reject"
        }
        fn validate(
            &self,
            _: &ValidationContext<'_>,
            _: &AssetRecord,
            sink: &mut ValidationSink<'_>,
        ) -> Result<(), String> {
            sink.emit(DiagnosticSeverity::Error, "source", "invalid host source")
                .map_err(|error| error.to_string())
        }
    }
    let temp = Temp::new();
    let mut ui = EditorDesktopUi::new(controller()).unwrap();
    edit(ui.controller_mut(), 42);
    let mut content = ContentCatalog::new();
    content
        .insert(
            AssetRecord::new(
                AssetId::from_raw(1),
                "actual.source",
                AssetKind::new("host").unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    ui.mount_content_browser(content).unwrap();
    let mut rules = ValidationRegistry::new(ValidationLimits::default());
    rules.register(Arc::new(Reject)).unwrap();
    ui.mount_content_validation(rules, ValidationPolicy::default());
    let mut session =
        DesktopDocumentSession::new(temp.adapter(), catalog(), ControllerLimits::default());
    assert!(matches!(session.save(&mut ui), Err(DocumentError::Ui(_))));
    assert!(!session.is_busy());
    assert!(session.last_request().is_none());
    assert!(
        !temp.file().exists(),
        "rejected validation must not submit filesystem work"
    );
    assert!(ui.controller().is_dirty());
    assert_eq!(ui.content_validation_receipt().unwrap().error_count(), 1);

    ui.mount_content_validation(
        ValidationRegistry::new(ValidationLimits::default()),
        ValidationPolicy::default(),
    );
    session.save(&mut ui).unwrap();
    let mut scene = DesktopScene::new(Vec::new()).unwrap();
    let waker = Waker::from(Arc::new(Notify(thread::current())));
    let mut cx = Context::from_waker(&waker);
    let deadline = Instant::now() + Duration::from_secs(10);
    while session.is_busy() {
        assert!(Instant::now() < deadline, "native validated save deadline");
        let _ = session.poll(&mut cx, &mut ui, &mut scene).unwrap();
        thread::yield_now();
    }
    assert!(temp.file().exists());
    assert!(!ui.controller().is_dirty());
    let mut reopened = EditorController::new();
    open(&mut temp.adapter(), &mut reopened);
    assert_eq!(snapshot(&reopened, &items()[0]).count, 42);
}
