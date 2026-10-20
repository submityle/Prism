use prism_asset_build_core::{
    canonical_settings_path, AssetId, BuildResources, BuildRevision, ContentHash, DerivedDataCache,
    EditableAsset, EditableBuildOptions, EditableBuildPlan, EditableCookDependency,
    EditableCookInput, EditableCooker, EditableCookerDescriptor, EditableCookerRegistry,
    EditableSettings, EditableSettingsCodec, EditableSettingsError, EditableWorkspace,
    EditableWorkspaceEdit, ProcessorError, ProcessorIdentity, RevisionStore, SourceRecord,
    MAX_EDITABLE_WORKSPACE_ENCODED_BYTES,
};
use prism_reflect::schema::{MigrateError, Migration, SchemaRegistry, TypeSchema};
use prism_reflect::{
    to_binary, to_canonical_binary_with_schema, CanonicalLimits, DynamicStruct, Reflect,
    TypeRegistry, Typed,
};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[derive(Reflect, Debug, PartialEq)]
struct Detail {
    #[reflect(range(0.0..=1.0))]
    roughness: f32,
}

#[derive(Reflect, Debug, PartialEq)]
struct Settings {
    #[reflect(range(0.0..=10.0))]
    gain: f32,
    path: String,
    literal: String,
    mode: Option<u32>,
    weights: HashMap<String, u32>,
    tags: HashSet<String>,
    detail: Detail,
}

impl EditableSettings for Settings {
    const LOGICAL_TYPE: &'static str = "test.cook.settings";
    fn canonicalize(&mut self) -> Result<(), String> {
        self.path = canonical_settings_path(&self.path).map_err(|e| e.to_string())?;
        self.mode.get_or_insert(3);
        Ok(())
    }
    fn validate(&self) -> Result<(), String> {
        if self.mode == Some(0) {
            return Err("mode zero is reserved".into());
        }
        Ok(())
    }
}

#[derive(Reflect)]
struct Old {
    amount: f32,
    removed: u32,
}
#[derive(Reflect)]
struct Middle {
    gain: f32,
    path: String,
}

fn settings() -> Settings {
    Settings {
        gain: 2.0,
        path: "textures/main.png".into(),
        literal: "keep\\this".into(),
        mode: Some(3),
        weights: HashMap::from([("a".into(), 7), ("b".into(), 11)]),
        tags: HashSet::from(["opaque".into(), "static".into()]),
        detail: Detail { roughness: 0.5 },
    }
}

fn registries() -> (TypeRegistry, SchemaRegistry) {
    let mut types = TypeRegistry::new();
    types.register::<Settings>();
    types.register::<Detail>();
    types.register::<Old>();
    types.register::<Middle>();
    types.register::<Option<u32>>();
    types.register::<HashMap<String, u32>>();
    types.register::<HashSet<String>>();
    let mut schemas = SchemaRegistry::new();
    schemas.register_version(
        Settings::LOGICAL_TYPE,
        TypeSchema::new(std::any::type_name::<Old>(), 1, <Old as Typed>::type_info()),
    );
    schemas.register_version(
        Settings::LOGICAL_TYPE,
        TypeSchema::new(
            std::any::type_name::<Middle>(),
            2,
            <Middle as Typed>::type_info(),
        ),
    );
    schemas.register_version(
        Settings::LOGICAL_TYPE,
        TypeSchema::new(
            std::any::type_name::<Settings>(),
            3,
            <Settings as Typed>::type_info(),
        ),
    );
    (types, schemas)
}

fn migrations(schemas: &mut SchemaRegistry) {
    schemas
        .register_migration(
            Settings::LOGICAL_TYPE,
            Migration::new(1, 2, |value| {
                let gain = value
                    .remove("amount")
                    .ok_or_else(|| MigrateError::MissingField("amount".into()))?;
                value.remove("removed");
                // Deliberately insert path before gain, contrary to target layout.
                value.insert("path", String::from("textures/./temp/../main.png"));
                value.insert_boxed("gain", gain);
                Ok(())
            }),
        )
        .unwrap();
    schemas
        .register_migration(
            Settings::LOGICAL_TYPE,
            Migration::new(2, 3, |value| {
                let defaults = settings();
                value.insert("literal", defaults.literal);
                value.insert("mode", defaults.mode);
                value.insert("weights", defaults.weights);
                value.insert("tags", defaults.tags);
                value.insert("detail", defaults.detail);
                Ok(())
            }),
        )
        .unwrap();
}

fn envelope<T: Reflect + Typed>(value: &T, version: u32, types: &TypeRegistry) -> Vec<u8> {
    let mut bytes = b"PRVB".to_vec();
    bytes.extend_from_slice(&version.to_le_bytes());
    bytes.extend(
        to_canonical_binary_with_schema(
            value,
            <T as Typed>::type_info(),
            types,
            CanonicalLimits::default(),
        )
        .unwrap(),
    );
    bytes
}

fn asset() -> EditableAsset {
    EditableAsset::new(
        AssetId(1),
        42,
        0,
        vec![],
        SourceRecord::embedded("source.raw", "test/raw", b"source".to_vec()).unwrap(),
        vec![],
        vec![],
        vec![],
        vec![4, 8, 15],
        None,
    )
    .unwrap()
}

#[test]
fn semantic_normalization_is_canonical_and_preserves_literal_strings() {
    let (types, schemas) = registries();
    let codec = EditableSettingsCodec::new(&types, &schemas);
    let mut first = settings();
    first.path = "textures\\folder\\..\\main.png".into();
    first.mode = None;
    first.gain = -0.0;
    let mut second = settings();
    second.gain = 0.0;
    second.weights = HashMap::from([("b".into(), 11), ("a".into(), 7)]);
    second.tags = HashSet::from(["static".into(), "opaque".into()]);
    let encoded = codec.encode(first).unwrap();
    assert_eq!(encoded, codec.encode(second).unwrap());
    let decoded = codec.decode::<Settings>(&encoded).unwrap();
    assert_eq!(decoded.value.path, "textures/main.png");
    assert_eq!(decoded.value.literal, "keep\\this");
    assert_eq!(decoded.value.mode, Some(3));
    assert_eq!(decoded.value.gain.to_bits(), 0);
    assert_eq!(decoded.canonical_bytes, encoded);
}

#[test]
fn explicit_two_step_upgrade_reconstructs_current_rust_type_and_saves_current_bytes() {
    let (types, mut schemas) = registries();
    migrations(&mut schemas);
    let codec = EditableSettingsCodec::new(&types, &schemas);
    let old = envelope(
        &Old {
            amount: 2.0,
            removed: 99,
        },
        1,
        &types,
    );
    let decoded = codec.decode::<Settings>(&old).unwrap();
    assert_eq!(decoded.source_version, 1);
    assert_eq!(decoded.current_version, 3);
    assert_eq!(decoded.value, settings());
    assert_eq!(decoded.canonical_bytes, codec.encode(settings()).unwrap());
    let mut asset = asset();
    asset.canonical_settings = old.clone();
    assert_eq!(
        asset.typed_settings::<Settings>(&codec).unwrap().value,
        settings()
    );
    assert_eq!(
        asset.canonical_settings, old,
        "reading does not silently save a migration"
    );
    assert_eq!(
        asset.migrate_typed_settings::<Settings>(&codec).unwrap(),
        settings()
    );
    assert_eq!(asset.canonical_settings, decoded.canonical_bytes);
}

#[test]
fn strict_current_payload_rejects_semantic_aliases_negative_zero_and_trailing_bytes() {
    let (types, schemas) = registries();
    let codec = EditableSettingsCodec::new(&types, &schemas);
    let mut alias = settings();
    alias.path = "textures/./main.png".into();
    assert!(matches!(
        codec.decode::<Settings>(&envelope(&alias, 3, &types)),
        Err(EditableSettingsError::NonCanonical)
    ));
    alias = settings();
    alias.mode = None;
    assert!(matches!(
        codec.decode::<Settings>(&envelope(&alias, 3, &types)),
        Err(EditableSettingsError::NonCanonical)
    ));
    alias = settings();
    alias.gain = -0.0;
    let mut raw = b"PRVB".to_vec();
    raw.extend_from_slice(&3u32.to_le_bytes());
    raw.extend(to_binary(&alias).unwrap());
    assert!(matches!(
        codec.decode::<Settings>(&raw),
        Err(EditableSettingsError::NonCanonical)
    ));
    let mut trailing = codec.encode(settings()).unwrap();
    trailing.push(0);
    assert!(matches!(
        codec.decode::<Settings>(&trailing),
        Err(EditableSettingsError::Decode(_))
    ));
}

#[test]
fn malformed_future_unknown_wrong_type_and_missing_migrations_are_rejected() {
    let (types, schemas) = registries();
    let codec = EditableSettingsCodec::new(&types, &schemas);
    for bad in [vec![], b"PRVB".to_vec(), b"BAD!1234".to_vec()] {
        assert!(matches!(
            codec.decode::<Settings>(&bad),
            Err(EditableSettingsError::BadEnvelope)
        ));
    }
    assert!(matches!(
        codec.decode::<Settings>(&envelope(&settings(), 4, &types)),
        Err(EditableSettingsError::FutureVersion { .. })
    ));
    assert!(matches!(
        codec.decode::<Settings>(&envelope(&settings(), 0, &types)),
        Err(EditableSettingsError::UnknownVersion(0))
    ));
    assert!(matches!(
        codec.decode::<Settings>(&envelope(
            &Old {
                amount: 2.0,
                removed: 1
            },
            3,
            &types
        )),
        Err(EditableSettingsError::Decode(_))
    ));
    assert!(matches!(
        codec.decode::<Settings>(&envelope(
            &Old {
                amount: 2.0,
                removed: 1
            },
            1,
            &types
        )),
        Err(EditableSettingsError::Migration(_))
    ));
    let mut far = SchemaRegistry::new();
    far.register_version(
        Settings::LOGICAL_TYPE,
        TypeSchema::new(std::any::type_name::<Old>(), 1, <Old as Typed>::type_info()),
    );
    far.register_version(
        Settings::LOGICAL_TYPE,
        TypeSchema::new(
            std::any::type_name::<Settings>(),
            100,
            <Settings as Typed>::type_info(),
        ),
    );
    assert!(matches!(
        EditableSettingsCodec::new(&types, &far).decode::<Settings>(&envelope(
            &Old {
                amount: 2.0,
                removed: 1
            },
            1,
            &types
        )),
        Err(EditableSettingsError::MigrationLimit { .. })
    ));
}

#[test]
fn metadata_semantics_nonfinite_and_failures_never_replace_asset_settings() {
    let (types, schemas) = registries();
    let codec = EditableSettingsCodec::new(&types, &schemas);
    let mut asset = asset();
    asset.set_typed_settings(settings(), &codec).unwrap();
    let expected = asset.canonical_settings.clone();
    let pointer = asset.authoring_payload.as_ptr();
    for bad in [11.0, f32::NAN, f32::INFINITY] {
        let mut value = settings();
        value.gain = bad;
        assert!(asset.set_typed_settings(value, &codec).is_err());
        assert_eq!(asset.canonical_settings, expected);
    }
    let mut value = settings();
    value.detail.roughness = 2.0;
    assert!(matches!(
        asset.set_typed_settings(value, &codec),
        Err(EditableSettingsError::Metadata { .. })
    ));
    let mut value = settings();
    value.mode = Some(0);
    assert!(matches!(
        asset.set_typed_settings(value, &codec),
        Err(EditableSettingsError::Policy(_))
    ));
    let mut value = settings();
    value.path = "../escape.png".into();
    assert!(asset.set_typed_settings(value, &codec).is_err());
    assert_eq!(asset.canonical_settings, expected);
    assert_eq!(asset.authoring_payload.as_ptr(), pointer);
    let mut value = settings();
    value.gain = 11.0;
    assert!(matches!(
        codec.decode::<Settings>(&envelope(&value, 3, &types)),
        Err(EditableSettingsError::Metadata { .. })
    ));
}

#[test]
fn bad_migration_shapes_metadata_and_panics_do_not_mutate_saved_bytes() {
    for case in 0..7 {
        let (types, mut schemas) = registries();
        migrations(&mut schemas);
        schemas
            .register_migration(
                Settings::LOGICAL_TYPE,
                Migration::new(2, 3, move |value| {
                    let defaults = settings();
                    value.insert("literal", defaults.literal);
                    value.insert("mode", defaults.mode);
                    value.insert("weights", defaults.weights);
                    value.insert("tags", defaults.tags);
                    value.insert("detail", defaults.detail);
                    match case {
                        0 => value.insert("gain", 2u32),
                        1 => value.insert("extra", 1u32),
                        2 => {
                            value.remove("gain");
                        }
                        3 => {
                            let mut nested = DynamicStruct::new();
                            nested.insert("wrong", 0.5f32);
                            value.insert("detail", nested);
                        }
                        4 => value.insert("detail", Detail { roughness: 5.0 }),
                        5 => panic!("trusted migration failed"),
                        _ => value.insert("gain", f32::NAN),
                    }
                    Ok(())
                }),
            )
            .unwrap();
        let codec = EditableSettingsCodec::new(&types, &schemas);
        let mut asset = asset();
        asset.canonical_settings = envelope(
            &Old {
                amount: 2.0,
                removed: 9,
            },
            1,
            &types,
        );
        let expected = asset.canonical_settings.clone();
        assert!(
            asset.migrate_typed_settings::<Settings>(&codec).is_err(),
            "case {case}"
        );
        assert_eq!(asset.canonical_settings, expected);
    }
}

#[test]
fn bounded_codec_rejects_large_trees_and_invalid_registration() {
    let (types, schemas) = registries();
    let codec = EditableSettingsCodec::new(&types, &schemas);
    for limits in [
        CanonicalLimits {
            max_bytes: 30,
            ..codec.limits()
        },
        CanonicalLimits {
            max_depth: 1,
            ..codec.limits()
        },
        CanonicalLimits {
            max_nodes: 1,
            ..codec.limits()
        },
        CanonicalLimits {
            max_collection_len: 0,
            ..codec.limits()
        },
    ] {
        let bounded = codec.with_limits(limits).unwrap();
        assert!(bounded.encode(settings()).is_err());
        assert!(bounded
            .decode::<Settings>(&codec.encode(settings()).unwrap())
            .is_err());
    }
    for limits in [
        CanonicalLimits {
            max_bytes: 1_048_577,
            ..codec.limits()
        },
        CanonicalLimits {
            max_bytes: 20,
            ..codec.limits()
        },
        CanonicalLimits {
            max_depth: 65,
            ..codec.limits()
        },
        CanonicalLimits {
            max_nodes: 0,
            ..codec.limits()
        },
        CanonicalLimits {
            max_collection_len: 4097,
            ..codec.limits()
        },
    ] {
        assert!(codec.with_limits(limits).is_err());
    }
    let mut huge = settings();
    huge.literal = "x".repeat(1_048_576);
    assert!(codec.encode(huge).is_err());
    let mut huge = settings();
    huge.tags = (0..4097).map(|i| i.to_string()).collect();
    assert!(codec.encode(huge).is_err());
    let missing = TypeRegistry::new();
    assert!(matches!(
        EditableSettingsCodec::new(&missing, &schemas).encode(settings()),
        Err(EditableSettingsError::Registration(_))
    ));
    let mut mismatched = SchemaRegistry::new();
    mismatched.register_version(
        Settings::LOGICAL_TYPE,
        TypeSchema::new(
            std::any::type_name::<Settings>(),
            1,
            <Old as Typed>::type_info(),
        ),
    );
    assert!(matches!(
        EditableSettingsCodec::new(&types, &mismatched).encode(settings()),
        Err(EditableSettingsError::Registration(_))
    ));
    let mut no_children = TypeRegistry::new();
    no_children.register::<Settings>();
    assert!(EditableSettingsCodec::new(&no_children, &schemas)
        .encode(settings())
        .is_err());
}

#[test]
fn explicit_path_policy_rejects_escape_and_absolute_forms() {
    assert_eq!(canonical_settings_path("a\\b/.././c//d").unwrap(), "a/c/d");
    for path in [
        "",
        ".",
        "..",
        "a/../..",
        "/absolute",
        "\\server\\share",
        "C:relative",
        "http://host",
        "nul\0path",
    ] {
        assert!(canonical_settings_path(path).is_err(), "{path:?}");
    }
}

struct SettingsCooker {
    types: TypeRegistry,
    schemas: SchemaRegistry,
    calls: Arc<AtomicUsize>,
}
impl EditableCooker for SettingsCooker {
    fn descriptor(&self) -> EditableCookerDescriptor {
        EditableCookerDescriptor {
            authoring_kind: 42,
            identity: ProcessorIdentity::new(77, 1),
            output_kind: 43,
            output_schema: 1,
            alignment: 1,
            max_output_bytes: 16,
            resources: BuildResources::default(),
        }
    }
    fn cook(
        &self,
        input: &EditableCookInput,
        _: &[EditableCookDependency<'_>],
    ) -> Result<Vec<u8>, ProcessorError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let settings = EditableSettingsCodec::new(&self.types, &self.schemas)
            .decode::<Settings>(&input.canonical_settings)
            .map_err(|e| ProcessorError::new(e.to_string()))?
            .value;
        Ok(input
            .authoring_payload
            .iter()
            .map(|byte| (*byte as f32 * settings.gain) as u8)
            .collect())
    }
}

#[test]
fn typed_settings_survive_save_reopen_and_real_cook_with_semantic_ddc_reuse() {
    let root = tempfile::tempdir().unwrap();
    let content = root.path().join("content");
    std::fs::create_dir(&content).unwrap();
    let (types, schemas) = registries();
    let codec = EditableSettingsCodec::new(&types, &schemas);
    let mut initial = asset();
    initial.set_typed_settings(settings(), &codec).unwrap();
    initial.write_to_path(content.join("asset.prism")).unwrap();
    let workspace = EditableWorkspace::open(&content).unwrap();
    let mut next = asset();
    let mut equivalent = settings();
    equivalent.path = "textures/./main.png".into();
    equivalent.mode = None;
    next.set_typed_settings(equivalent, &codec).unwrap();
    workspace
        .save(
            vec![EditableWorkspaceEdit {
                expected_hash: ContentHash::digest(&initial.encode().unwrap()),
                replacement: next,
            }],
            MAX_EDITABLE_WORKSPACE_ENCODED_BYTES,
        )
        .unwrap();
    let reopened = EditableWorkspace::open(&content).unwrap();
    let snapshot = reopened
        .snapshot(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)
        .unwrap();
    assert_eq!(
        snapshot[&AssetId(1)]
            .typed_settings::<Settings>(&codec)
            .unwrap()
            .value,
        settings()
    );
    let calls = Arc::new(AtomicUsize::new(0));
    let (cook_types, cook_schemas) = registries();
    let mut registry = EditableCookerRegistry::new();
    registry
        .register(SettingsCooker {
            types: cook_types,
            schemas: cook_schemas,
            calls: calls.clone(),
        })
        .unwrap();
    let plan = EditableBuildPlan::from_snapshot(
        snapshot,
        &[AssetId(1)],
        &registry,
        &EditableBuildOptions::default(),
    )
    .unwrap();
    let store = RevisionStore::open(root.path().join("build")).unwrap();
    let mut ddc = DerivedDataCache::open(root.path().join("ddc")).unwrap();
    let first = plan
        .execute(
            &store,
            &mut ddc,
            BuildRevision(0),
            2,
            BuildResources::UNLIMITED,
            false,
        )
        .unwrap();
    let hash = first.published.manifest.entries[&AssetId(1)].content_hash;
    assert_eq!(store.cas().read(hash).unwrap(), [8, 16, 30]);
    let same = EditableBuildPlan::from_snapshot(
        BTreeMap::from([(AssetId(1), initial)]),
        &[AssetId(1)],
        &registry,
        &EditableBuildOptions::default(),
    )
    .unwrap();
    let cached = same
        .execute(
            &store,
            &mut ddc,
            BuildRevision(1),
            2,
            BuildResources::UNLIMITED,
            false,
        )
        .unwrap();
    assert_eq!(
        cached.published.manifest.entries[&AssetId(1)].content_hash,
        hash
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "semantic aliases reuse the same DDC output"
    );
    let mut edited = reopened
        .snapshot(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)
        .unwrap();
    let mut changed = settings();
    changed.gain = 3.0;
    edited
        .get_mut(&AssetId(1))
        .unwrap()
        .set_typed_settings(changed, &codec)
        .unwrap();
    let changed = EditableBuildPlan::from_snapshot(
        edited,
        &[AssetId(1)],
        &registry,
        &EditableBuildOptions::default(),
    )
    .unwrap();
    let third = changed
        .execute(
            &store,
            &mut ddc,
            BuildRevision(2),
            2,
            BuildResources::UNLIMITED,
            true,
        )
        .unwrap();
    assert_eq!(
        store
            .cas()
            .read(third.published.manifest.entries[&AssetId(1)].content_hash)
            .unwrap(),
        [12, 24, 45]
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        3,
        "changed settings are cooked twice for determinism"
    );
}
