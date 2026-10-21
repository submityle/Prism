//! Schema admission runs on complete isolated candidates, never partial models.

use std::{
    any::{type_name, Any},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex, OnceLock,
    },
};

use prism_editor_app::{
    ControllerError, ControllerFieldEdit, ControllerLimits, ControllerObjectValidator,
    DocumentCodec, DocumentError, DocumentLimits, DocumentSchemaCatalog, EditorController,
};
use prism_editor_core::SelectionItem;
use prism_editor_reflect::FieldPath;
use prism_editor_ui::{ActionValue, EditorAction};
use prism_reflect::{
    FieldMetadata, FromReflect, Reflect, ReflectMut, ReflectRef, TypeInfo, TypeMetadata,
    TypeRegistry, ValueInfo,
};

#[derive(Clone, Debug, PartialEq, Reflect)]
struct Interval {
    lower: i32,
    upper: i32,
    locked: i32,
    gain: f32,
}

fn interval() -> Interval {
    Interval {
        lower: 0,
        upper: 10,
        locked: 7,
        gain: 0.5,
    }
}

fn plain_registry() -> TypeRegistry {
    let mut registry = TypeRegistry::new();
    registry.register::<Interval>();
    registry.register_type_data::<Interval, _>(
        TypeMetadata::new()
            .with_field(FieldMetadata::new("locked").readonly(true))
            .with_field(FieldMetadata::new("gain").with_range(0.0, 1.0)),
    );
    registry
}

fn validate_interval(root: &dyn Reflect) -> Result<(), String> {
    let model = Interval::from_reflect(root).ok_or("invalid interval schema")?;
    if model.lower > model.upper {
        Err("lower must not exceed upper".into())
    } else {
        Ok(())
    }
}

fn registry() -> TypeRegistry {
    let mut registry = plain_registry();
    registry.register_type_data::<Interval, _>(ControllerObjectValidator::new(|root, _| {
        validate_interval(root)
    }));
    registry
}

fn edit(id: u128, path: &str, value: ActionValue) -> ControllerFieldEdit {
    ControllerFieldEdit {
        item: SelectionItem::entity(id),
        path: FieldPath::parse(path).unwrap(),
        value,
    }
}

fn snapshot(controller: &EditorController, id: u128) -> Interval {
    let root = controller.object_root(&SelectionItem::entity(id)).unwrap();
    Interval::from_reflect(root).unwrap()
}

fn loaded() -> EditorController {
    let mut controller = EditorController::new();
    for id in 1..=2 {
        controller
            .register_object(SelectionItem::entity(id), Box::new(interval()), registry())
            .unwrap();
    }
    controller.select(SelectionItem::entity(1)).unwrap();
    controller
}

#[test]
fn invalid_registration_does_not_publish_identity_or_revisions() {
    let mut controller = EditorController::new();
    let before = (
        controller.revision(),
        controller.model_revision(),
        controller.shape_revision(),
    );
    let invalid = Interval {
        lower: 11,
        ..interval()
    };
    assert!(matches!(
        controller.register_object(SelectionItem::entity(1), Box::new(invalid), registry()),
        Err(ControllerError::ObjectValidation { item, .. }) if item == SelectionItem::entity(1)
    ));
    assert_eq!(controller.document_objects().count(), 0);
    assert_eq!(
        (
            controller.revision(),
            controller.model_revision(),
            controller.shape_revision()
        ),
        before
    );
    assert_eq!(controller.document_history_lengths(), (0, 0));
    assert!(!controller.is_dirty());
    controller
        .register_object(SelectionItem::entity(1), Box::new(interval()), registry())
        .unwrap();
}

#[test]
fn coupled_fields_validate_once_in_final_state_and_undo_as_one_command() {
    let calls = Arc::new(Mutex::new(Vec::<Vec<String>>::new()));
    let recorded = calls.clone();
    let mut schema = registry();
    schema.register_type_data::<Interval, _>(ControllerObjectValidator::new(move |root, paths| {
        recorded
            .lock()
            .unwrap()
            .push(paths.iter().map(|p| p.as_str().to_owned()).collect());
        validate_interval(root)
    }));
    let mut controller = EditorController::new();
    controller
        .register_object(SelectionItem::entity(1), Box::new(interval()), schema)
        .unwrap();
    assert_eq!(*calls.lock().unwrap(), vec![Vec::<String>::new()]);
    calls.lock().unwrap().clear();
    let prepared = controller
        .prepare_edits(vec![
            edit(1, "lower", ActionValue::integer(20)),
            edit(1, "upper", ActionValue::integer(30)),
        ])
        .unwrap();
    assert_eq!(snapshot(&controller, 1), interval());
    assert_eq!(
        *calls.lock().unwrap(),
        vec![vec![".lower".to_owned(), ".upper".to_owned()]]
    );
    controller.commit_edits(prepared).unwrap();
    let expected = Interval {
        lower: 20,
        upper: 30,
        ..interval()
    };
    assert_eq!(snapshot(&controller, 1), expected);
    assert_eq!(controller.document_history_lengths(), (1, 0));
    controller.execute(EditorAction::Undo).unwrap();
    assert_eq!(snapshot(&controller, 1), interval());
    controller.execute(EditorAction::Redo).unwrap();
    assert_eq!(snapshot(&controller, 1), expected);
    assert_eq!(calls.lock().unwrap().len(), 1);
}

#[test]
fn invalid_sibling_batch_preserves_models_redo_selection_and_save_state() {
    let mut controller = loaded();
    let prepared = controller
        .prepare_edits(vec![edit(1, "upper", ActionValue::integer(15))])
        .unwrap();
    controller.commit_edits(prepared).unwrap();
    controller.execute(EditorAction::Undo).unwrap();
    controller.mark_saved(controller.model_revision()).unwrap();
    let before = (
        controller.revision(),
        controller.model_revision(),
        controller.shape_revision(),
    );
    let selection = controller.selection().clone();
    assert!(matches!(controller.prepare_edits(vec![
        edit(1, "upper", ActionValue::integer(50)),
        edit(2, "lower", ActionValue::integer(11)),
    ]), Err(ControllerError::ObjectValidation { item, .. }) if item == SelectionItem::entity(2)));
    assert_eq!(snapshot(&controller, 1), interval());
    assert_eq!(snapshot(&controller, 2), interval());
    assert_eq!(controller.selection(), &selection);
    assert_eq!(controller.document_history_lengths(), (0, 1));
    assert_eq!(
        (
            controller.revision(),
            controller.model_revision(),
            controller.shape_revision()
        ),
        before
    );
    assert!(!controller.is_dirty());
    controller.execute(EditorAction::Redo).unwrap();
    assert_eq!(snapshot(&controller, 1).upper, 15);
}

#[test]
fn schema_admission_does_not_bypass_readonly_range_or_noop_rules() {
    let mut controller = loaded();
    let before = (controller.revision(), controller.model_revision());
    for fields in [
        vec![edit(1, "locked", ActionValue::integer(8))],
        vec![edit(1, "gain", ActionValue::float(2.0))],
    ] {
        assert!(controller.prepare_edits(fields).is_err());
        assert_eq!(snapshot(&controller, 1), interval());
        assert_eq!((controller.revision(), controller.model_revision()), before);
    }
    let prepared = controller
        .prepare_edits(vec![edit(1, "lower", ActionValue::integer(0))])
        .unwrap();
    controller.commit_edits(prepared).unwrap();
    assert_eq!(controller.document_history_lengths(), (0, 0));
    assert_eq!((controller.revision(), controller.model_revision()), before);
    assert!(!controller.is_dirty());
    assert!(controller
        .execute(EditorAction::EditInspectorField {
            path: FieldPath::parse("lower").unwrap(),
            value: ActionValue::integer(11),
        })
        .is_err());
    assert_eq!(snapshot(&controller, 1), interval());
}

fn catalog(factory: fn() -> TypeRegistry) -> DocumentSchemaCatalog {
    let mut catalog = DocumentSchemaCatalog::new();
    catalog
        .register(type_name::<Interval>(), 1, factory)
        .unwrap();
    catalog
}

#[test]
fn registry_clone_and_document_decode_retain_admission_rules() {
    let source = registry();
    let mut copied = TypeRegistry::new();
    for registration in source.iter() {
        copied.add_registration(registration.clone());
    }
    let mut controller = EditorController::new();
    controller
        .register_object(SelectionItem::entity(1), Box::new(interval()), copied)
        .unwrap();
    assert!(controller
        .prepare_edits(vec![edit(1, "lower", ActionValue::integer(11))])
        .is_err());
    let schemas = catalog(registry);
    let bytes = DocumentCodec::encode(&controller, &schemas, DocumentLimits::default()).unwrap();
    let decoded = DocumentCodec::decode(
        &bytes,
        &schemas,
        DocumentLimits::default(),
        ControllerLimits::default(),
    )
    .unwrap()
    .into_controller();
    assert_eq!(snapshot(&decoded, 1), interval());
    assert!(decoded
        .prepare_edits(vec![edit(1, "lower", ActionValue::integer(11))])
        .is_err());

    let mut unvalidated = EditorController::new();
    unvalidated
        .register_object(
            SelectionItem::entity(1),
            Box::new(Interval {
                lower: 11,
                ..interval()
            }),
            plain_registry(),
        )
        .unwrap();
    let bytes = DocumentCodec::encode(
        &unvalidated,
        &catalog(plain_registry),
        DocumentLimits::default(),
    )
    .unwrap();
    assert!(matches!(
        DocumentCodec::decode(
            &bytes,
            &schemas,
            DocumentLimits::default(),
            ControllerLimits::default()
        ),
        Err(DocumentError::Controller(
            ControllerError::ObjectValidation { .. }
        ))
    ));
}

// A leaf that records actual reflection clones, including clones performed by
// derived roots. Rust Clone counters alone would miss DynamicStruct rebuilding.
struct CloneCounter(Arc<AtomicUsize>);

impl Reflect for CloneCounter {
    fn type_name(&self) -> &'static str {
        type_name::<Self>()
    }
    fn type_info(&self) -> &'static TypeInfo {
        static INFO: OnceLock<TypeInfo> = OnceLock::new();
        INFO.get_or_init(|| TypeInfo::Value(ValueInfo::new(type_name::<Self>())))
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
        ReflectRef::Value(self)
    }
    fn reflect_mut(&mut self) -> ReflectMut<'_> {
        ReflectMut::Value(self)
    }
    fn reflect_clone(&self) -> Box<dyn Reflect> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Box::new(Self(self.0.clone()))
    }
}

impl FromReflect for CloneCounter {
    fn from_reflect(value: &dyn Reflect) -> Option<Self> {
        value
            .downcast_ref::<Self>()
            .map(|value| Self(value.0.clone()))
    }
}

#[derive(Reflect)]
struct CountedRoot {
    a: i32,
    b: i32,
    counter: CloneCounter,
}

#[test]
fn multi_field_prepare_clones_each_touched_root_once_and_noop_skips_validator() {
    let clones = Arc::new(AtomicUsize::new(0));
    let validations = Arc::new(AtomicUsize::new(0));
    let mut schema = TypeRegistry::new();
    schema.register::<CountedRoot>();
    let calls = validations.clone();
    schema.register_type_data::<CountedRoot, _>(ControllerObjectValidator::new(move |_, _| {
        calls.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }));
    let mut controller = EditorController::new();
    controller
        .register_object(
            SelectionItem::entity(1),
            Box::new(CountedRoot {
                a: 0,
                b: 0,
                counter: CloneCounter(clones.clone()),
            }),
            schema,
        )
        .unwrap();
    validations.store(0, Ordering::Relaxed);
    let prepared = controller
        .prepare_edits(vec![
            edit(1, "a", ActionValue::integer(1)),
            edit(1, "b", ActionValue::integer(2)),
        ])
        .unwrap();
    assert_eq!(clones.load(Ordering::Relaxed), 1);
    assert_eq!(validations.load(Ordering::Relaxed), 1);
    controller.commit_edits(prepared).unwrap();
    assert_eq!(clones.load(Ordering::Relaxed), 1);
    let prepared = controller
        .prepare_edits(vec![edit(1, "a", ActionValue::integer(1))])
        .unwrap();
    controller.commit_edits(prepared).unwrap();
    assert_eq!(validations.load(Ordering::Relaxed), 1);
    assert_eq!(controller.document_history_lengths(), (1, 0));
}
