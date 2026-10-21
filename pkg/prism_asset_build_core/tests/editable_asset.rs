use prism_asset_build_core::editable::MAX_EDITABLE_WORKSPACE_ENCODED_BYTES;
use prism_asset_build_core::{
    AssetId, ContentHash, EditableAsset, EditableAssetError, EditableDependency,
    EditableDependencyKind, EditablePayloadReferenceRewriter, EditableReimportOutput,
    EditableReimporter, EditableSubAsset, EditableSubAssetChange, EditableSubAssetRedirect,
    EditableSubAssetReference, EditableSubAssetReferenceRewrite, EditableWorkspace,
    LocalFileSourceProvider, SourceRecord, SourceSnapshot, EDITABLE_ASSET_SCHEMA,
    MAX_EDITABLE_ASSET_BYTES,
};
use std::fs;
use std::path::Path;

fn embedded_asset() -> EditableAsset {
    EditableAsset::new(
        AssetId(0xA55E7),
        0x494D_4147_45,
        0b101,
        b"canonical-settings".to_vec(),
        SourceRecord::embedded(
            "textures/hero.png",
            "image/png",
            b"embedded source bytes".to_vec(),
        )
        .unwrap(),
        vec![
            EditableSubAsset {
                id: 4,
                name: "albedo".into(),
                kind: 0x494D_4147_45,
            },
            EditableSubAsset {
                id: 9,
                name: "normal".into(),
                kind: 0x494D_4147_45,
            },
        ],
        vec![
            EditableDependency {
                asset: AssetId(3),
                kind: EditableDependencyKind::Hard,
            },
            EditableDependency {
                asset: AssetId(7),
                kind: EditableDependencyKind::Soft,
            },
        ],
        Vec::new(),
        b"lossless editable payload".to_vec(),
        Some(b"preview".to_vec()),
    )
    .unwrap()
}

fn external_asset(
    asset: AssetId,
    source_hash: ContentHash,
    subassets: Vec<EditableSubAsset>,
    subasset_references: Vec<EditableSubAssetReference>,
    authoring_payload: &[u8],
) -> EditableAsset {
    EditableAsset::new(
        asset,
        0x494D_4147_45,
        0,
        b"settings".to_vec(),
        SourceRecord::external("sources/hero.input", "test/raw", source_hash).unwrap(),
        subassets,
        Vec::new(),
        subasset_references,
        authoring_payload.to_vec(),
        None,
    )
    .unwrap()
}

#[derive(Clone)]
struct TestReimporter {
    output: EditableReimportOutput,
}

impl EditableReimporter for TestReimporter {
    fn reimport(
        &self,
        _previous: &EditableAsset,
        _previous_payload: &[u8],
        _source: &SourceSnapshot,
    ) -> Result<EditableReimportOutput, EditableAssetError> {
        Ok(self.output.clone())
    }
}

struct NoopPayloadRewriter;

impl EditablePayloadReferenceRewriter for NoopPayloadRewriter {
    fn rewrite_authoring_payload(
        &self,
        _asset: &EditableAsset,
        payload: &[u8],
        _rewrites: &[EditableSubAssetReferenceRewrite],
    ) -> Result<Vec<u8>, EditableAssetError> {
        Ok(payload.to_vec())
    }
}

struct RewritingPayloadRewriter;

impl EditablePayloadReferenceRewriter for RewritingPayloadRewriter {
    fn rewrite_authoring_payload(
        &self,
        _asset: &EditableAsset,
        payload: &[u8],
        rewrites: &[EditableSubAssetReferenceRewrite],
    ) -> Result<Vec<u8>, EditableAssetError> {
        assert_eq!(payload, b"ref:9");
        assert_eq!(
            rewrites,
            &[EditableSubAssetReferenceRewrite {
                asset: AssetId(10),
                from: 9,
                to: 12,
            }]
        );
        Ok(b"ref:12".to_vec())
    }
}

struct FailingPayloadRewriter;

impl EditablePayloadReferenceRewriter for FailingPayloadRewriter {
    fn rewrite_authoring_payload(
        &self,
        _asset: &EditableAsset,
        _payload: &[u8],
        _rewrites: &[EditableSubAssetReferenceRewrite],
    ) -> Result<Vec<u8>, EditableAssetError> {
        Err(EditableAssetError::InvalidSourcePath)
    }
}

fn changed_subassets() -> Vec<EditableSubAsset> {
    vec![
        EditableSubAsset {
            id: 4,
            name: "albedo".into(),
            kind: 0x494D_4147_45,
        },
        EditableSubAsset {
            id: 12,
            name: "normal_detail".into(),
            kind: 0x494D_4147_45,
        },
    ]
}

fn test_reimporter() -> TestReimporter {
    TestReimporter {
        output: EditableReimportOutput {
            canonical_settings: b"reimported-settings".to_vec(),
            subassets: changed_subassets(),
            dependencies: Vec::new(),
            subasset_references: Vec::new(),
            authoring_payload: b"reimported-payload".to_vec(),
            preview: Some(b"reimported-preview".to_vec()),
        },
    }
}

#[test]
fn canonical_encoding_is_deterministic_and_round_trips() {
    let asset = embedded_asset();

    let first = asset.encode().unwrap();
    let decoded = EditableAsset::decode(&first).unwrap();
    let second = decoded.encode().unwrap();

    assert_eq!(decoded, asset);
    assert_eq!(second, first);
}

#[test]
fn embedded_source_forms_the_exact_source_snapshot() {
    let snapshot = embedded_asset().embedded_source_snapshot().unwrap();

    assert_eq!(snapshot.logical_path(), "textures/hero.png");
    assert_eq!(snapshot.bytes(), b"embedded source bytes");
    assert_eq!(
        snapshot.content_hash(),
        ContentHash::digest(b"embedded source bytes")
    );
}

#[test]
fn external_source_cannot_be_mistaken_for_a_snapshot() {
    let source = SourceRecord::external(
        "https://assets.example.invalid/hero.png",
        "image/png",
        ContentHash::digest(b"captured elsewhere"),
    )
    .unwrap();
    let asset = EditableAsset::new(
        AssetId(8),
        1,
        0,
        Vec::new(),
        source,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        None,
    )
    .unwrap();

    assert_eq!(
        asset.embedded_source_snapshot(),
        Err(EditableAssetError::ExternalSourceUnavailable)
    );
}

#[test]
fn local_source_provider_captures_and_verifies_external_source() {
    let directory = tempfile::tempdir().unwrap();
    fs::create_dir(directory.path().join("textures")).unwrap();
    fs::write(
        directory.path().join("textures/hero.png"),
        b"external source bytes",
    )
    .unwrap();
    let bytes = b"external source bytes";
    let asset = EditableAsset::new(
        AssetId(8),
        1,
        0,
        Vec::new(),
        SourceRecord::external("textures/hero.png", "image/png", ContentHash::digest(bytes))
            .unwrap(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        None,
    )
    .unwrap();

    let snapshot = asset
        .resolve_source_snapshot(&LocalFileSourceProvider::new(directory.path()).unwrap())
        .unwrap();
    assert_eq!(snapshot.logical_path(), "textures/hero.png");
    assert_eq!(snapshot.bytes(), bytes);
}

#[test]
fn external_source_hash_and_path_mismatches_are_rejected() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(directory.path().join("hero.png"), b"actual").unwrap();
    let provider = LocalFileSourceProvider::new(directory.path()).unwrap();
    let mismatch = EditableAsset::new(
        AssetId(8),
        1,
        0,
        Vec::new(),
        SourceRecord::external("hero.png", "image/png", ContentHash::digest(b"expected")).unwrap(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        None,
    )
    .unwrap();
    assert_eq!(
        mismatch.resolve_source_snapshot(&provider),
        Err(EditableAssetError::ExternalSourceHashMismatch {
            expected: ContentHash::digest(b"expected"),
            actual: ContentHash::digest(b"actual"),
        })
    );

    let invalid_path = EditableAsset::new(
        AssetId(9),
        1,
        0,
        Vec::new(),
        SourceRecord::external(
            "../outside.png",
            "image/png",
            ContentHash::digest(b"actual"),
        )
        .unwrap(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        None,
    )
    .unwrap();
    assert_eq!(
        invalid_path.resolve_source_snapshot(&provider),
        Err(EditableAssetError::InvalidSourcePath)
    );
}

#[test]
fn decode_rejects_tampered_embedded_source_and_invalid_option_tags() {
    let mut tampered = embedded_asset().encode().unwrap();
    let source_start = tampered
        .windows(b"embedded source bytes".len())
        .position(|window| window == b"embedded source bytes")
        .unwrap();
    tampered[source_start] ^= 0x01;
    assert_eq!(
        EditableAsset::decode(&tampered),
        Err(EditableAssetError::EmbeddedSourceHashMismatch)
    );

    let mut no_preview = embedded_asset();
    no_preview.preview = None;
    let mut invalid_option = no_preview.encode().unwrap();
    *invalid_option.last_mut().unwrap() = 2;
    assert_eq!(
        EditableAsset::decode(&invalid_option),
        Err(EditableAssetError::InvalidOptionTag)
    );
}

#[test]
fn snapshot_revalidates_a_mutated_embedded_source() {
    let mut asset = embedded_asset();
    asset.source.embedded_source.as_mut().unwrap()[0] ^= 0x01;

    assert_eq!(
        asset.embedded_source_snapshot(),
        Err(EditableAssetError::EmbeddedSourceHashMismatch)
    );
}

#[test]
fn decode_rejects_invalid_wire_shape_and_size_ceiling() {
    let encoded = embedded_asset().encode().unwrap();

    assert_eq!(
        EditableAsset::decode(&encoded[..encoded.len() - 1]),
        Err(EditableAssetError::Truncated)
    );

    let mut trailing = encoded;
    trailing.push(0);
    assert_eq!(
        EditableAsset::decode(&trailing),
        Err(EditableAssetError::TrailingBytes)
    );

    let oversized = vec![0; MAX_EDITABLE_ASSET_BYTES + 1];
    assert_eq!(
        EditableAsset::decode(&oversized),
        Err(EditableAssetError::PackageTooLarge {
            actual: MAX_EDITABLE_ASSET_BYTES + 1,
        })
    );
}

#[test]
fn construction_rejects_noncanonical_directories_and_dependencies() {
    let base = embedded_asset();

    let mut subassets = base.subassets.clone();
    subassets.swap(0, 1);
    assert_eq!(
        EditableAsset::new(
            base.asset,
            base.kind,
            base.flags,
            base.canonical_settings.clone(),
            base.source.clone(),
            subassets,
            base.dependencies.clone(),
            base.subasset_references.clone(),
            base.authoring_payload.clone(),
            base.preview.clone(),
        ),
        Err(EditableAssetError::NonCanonicalSubassets)
    );

    let mut dependencies = base.dependencies.clone();
    dependencies.push(dependencies[1]);
    assert_eq!(
        EditableAsset::new(
            base.asset,
            base.kind,
            base.flags,
            base.canonical_settings,
            base.source,
            base.subassets,
            dependencies,
            base.subasset_references,
            base.authoring_payload,
            base.preview,
        ),
        Err(EditableAssetError::NonCanonicalDependencies)
    );
}

#[test]
fn decode_rejects_an_unsupported_schema() {
    let mut encoded = embedded_asset().encode().unwrap();
    let schema_offset = b"PRISM-EDITABLE\0".len();
    encoded[schema_offset..schema_offset + 4]
        .copy_from_slice(&(EDITABLE_ASSET_SCHEMA + 1).to_le_bytes());

    assert_eq!(
        EditableAsset::decode(&encoded),
        Err(EditableAssetError::UnsupportedSchema(
            EDITABLE_ASSET_SCHEMA + 1
        ))
    );
}

#[test]
fn atomic_file_persistence_round_trips_and_replaces_the_old_package() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("hero.prism");
    let mut replacement = embedded_asset();
    replacement.authoring_payload = b"updated editable payload".to_vec().into();

    embedded_asset().write_to_path(&path).unwrap();
    let first_bytes = fs::read(&path).unwrap();
    replacement.write_to_path(&path).unwrap();

    assert_eq!(EditableAsset::read_from_path(&path).unwrap(), replacement);
    assert_eq!(fs::read(&path).unwrap(), replacement.encode().unwrap());
    assert_ne!(fs::read(&path).unwrap(), first_bytes);
    assert!(fs::read_dir(directory.path()).unwrap().all(|entry| !entry
        .unwrap()
        .file_name()
        .to_string_lossy()
        .starts_with(".tmp")));
}

#[test]
fn file_persistence_rejects_non_prism_and_non_regular_paths() {
    let directory = tempfile::tempdir().unwrap();
    let asset = embedded_asset();
    let no_extension = directory.path().join("hero.asset");
    assert_eq!(
        asset.write_to_path(&no_extension),
        Err(EditableAssetError::InvalidFilePath)
    );

    let directory_as_file = directory.path().join("directory.prism");
    fs::create_dir(&directory_as_file).unwrap();
    assert_eq!(
        asset.write_to_path(&directory_as_file),
        Err(EditableAssetError::NonRegularFile)
    );
    assert_eq!(
        EditableAsset::read_from_path(&directory_as_file),
        Err(EditableAssetError::NonRegularFile)
    );
}

#[test]
fn subasset_reimport_plan_reports_stable_id_changes_deterministically() {
    let plan = embedded_asset()
        .plan_subasset_reimport(
            vec![
                EditableSubAsset {
                    id: 4,
                    name: "base_color".into(),
                    kind: 0x494D_4147_45,
                },
                EditableSubAsset {
                    id: 12,
                    name: "emissive".into(),
                    kind: 0x494D_4147_45,
                },
            ],
            &[],
            &[],
        )
        .unwrap();

    assert_eq!(
        plan.changes,
        vec![
            EditableSubAssetChange::Renamed {
                id: 4,
                previous_name: "albedo".into(),
                next_name: "base_color".into(),
            },
            EditableSubAssetChange::Removed {
                subasset: EditableSubAsset {
                    id: 9,
                    name: "normal".into(),
                    kind: 0x494D_4147_45,
                },
            },
            EditableSubAssetChange::Added {
                subasset: EditableSubAsset {
                    id: 12,
                    name: "emissive".into(),
                    kind: 0x494D_4147_45,
                },
            },
        ]
    );
}

#[test]
fn subasset_reimport_blocks_referenced_removal_without_an_explicit_redirect() {
    let next = vec![EditableSubAsset {
        id: 4,
        name: "albedo".into(),
        kind: 0x494D_4147_45,
    }];

    assert_eq!(
        embedded_asset().plan_subasset_reimport(next, &[9], &[]),
        Err(EditableAssetError::ReferencedSubAssetRemoved { id: 9 })
    );
}

#[test]
fn subasset_reimport_applies_explicit_redirects_without_guessing_identity() {
    let plan = embedded_asset()
        .plan_subasset_reimport(
            vec![
                EditableSubAsset {
                    id: 4,
                    name: "albedo".into(),
                    kind: 0x494D_4147_45,
                },
                EditableSubAsset {
                    id: 12,
                    name: "normal_detail".into(),
                    kind: 0x494D_4147_45,
                },
            ],
            &[9],
            &[EditableSubAssetRedirect { from: 9, to: 12 }],
        )
        .unwrap();

    assert_eq!(
        plan.changes,
        vec![
            EditableSubAssetChange::Retained { id: 4 },
            EditableSubAssetChange::Redirected { from: 9, to: 12 },
        ]
    );
}

#[test]
fn subasset_reimport_rejects_kind_changes_and_invalid_redirects() {
    assert_eq!(
        embedded_asset().plan_subasset_reimport(
            vec![
                EditableSubAsset {
                    id: 4,
                    name: "albedo".into(),
                    kind: 0x494D_4147_45,
                },
                EditableSubAsset {
                    id: 9,
                    name: "normal".into(),
                    kind: 99,
                },
            ],
            &[],
            &[],
        ),
        Err(EditableAssetError::SubAssetKindChanged { id: 9 })
    );

    assert_eq!(
        embedded_asset().plan_subasset_reimport(
            vec![
                EditableSubAsset {
                    id: 4,
                    name: "albedo".into(),
                    kind: 0x494D_4147_45,
                },
                EditableSubAsset {
                    id: 12,
                    name: "other".into(),
                    kind: 99,
                },
            ],
            &[9],
            &[EditableSubAssetRedirect { from: 9, to: 12 }],
        ),
        Err(EditableAssetError::RedirectKindMismatch { from: 9, to: 12 })
    );
}

#[test]
fn workspace_reimport_captures_the_current_external_source_and_publishes_the_package() {
    let workspace_root = tempfile::tempdir().unwrap();
    fs::create_dir(workspace_root.path().join("sources")).unwrap();
    fs::write(
        workspace_root.path().join("sources/hero.input"),
        b"old source",
    )
    .unwrap();
    let target = external_asset(
        AssetId(10),
        ContentHash::digest(b"old source"),
        embedded_asset().subassets,
        Vec::new(),
        b"original",
    );
    target
        .write_to_path(workspace_root.path().join("hero.prism"))
        .unwrap();
    fs::write(
        workspace_root.path().join("sources/hero.input"),
        b"new source",
    )
    .unwrap();

    let result = EditableWorkspace::open(workspace_root.path())
        .unwrap()
        .reimport(
            Path::new("hero.prism"),
            &LocalFileSourceProvider::new(workspace_root.path()).unwrap(),
            &test_reimporter(),
            &NoopPayloadRewriter,
            &[],
        )
        .unwrap();

    assert_eq!(result.updated_assets, vec![AssetId(10)]);
    let reimported =
        EditableAsset::read_from_path(workspace_root.path().join("hero.prism")).unwrap();
    assert_eq!(
        reimported.source.content_hash,
        ContentHash::digest(b"new source")
    );
    assert_eq!(reimported.canonical_settings, b"reimported-settings");
    assert_eq!(reimported.subassets, changed_subassets());
    assert_eq!(
        reimported.authoring_payload.inline_bytes(),
        Some(b"reimported-payload".as_slice())
    );
}

#[test]
fn workspace_reimport_rewrites_typed_and_payload_references_together() {
    let workspace_root = tempfile::tempdir().unwrap();
    fs::create_dir(workspace_root.path().join("sources")).unwrap();
    fs::write(
        workspace_root.path().join("sources/hero.input"),
        b"old source",
    )
    .unwrap();
    let target = external_asset(
        AssetId(10),
        ContentHash::digest(b"old source"),
        embedded_asset().subassets,
        Vec::new(),
        b"target",
    );
    let consumer = EditableAsset::new(
        AssetId(20),
        1,
        0,
        Vec::new(),
        SourceRecord::embedded("consumer", "test/raw", b"consumer source".to_vec()).unwrap(),
        Vec::new(),
        Vec::new(),
        vec![EditableSubAssetReference {
            asset: AssetId(10),
            subasset: 9,
            kind: EditableDependencyKind::Hard,
        }],
        b"ref:9".to_vec(),
        None,
    )
    .unwrap();
    target
        .write_to_path(workspace_root.path().join("hero.prism"))
        .unwrap();
    consumer
        .write_to_path(workspace_root.path().join("consumer.prism"))
        .unwrap();
    fs::write(
        workspace_root.path().join("sources/hero.input"),
        b"new source",
    )
    .unwrap();

    let result = EditableWorkspace::open(workspace_root.path())
        .unwrap()
        .reimport(
            Path::new("hero.prism"),
            &LocalFileSourceProvider::new(workspace_root.path()).unwrap(),
            &test_reimporter(),
            &RewritingPayloadRewriter,
            &[EditableSubAssetRedirect { from: 9, to: 12 }],
        )
        .unwrap();

    assert_eq!(result.updated_assets, vec![AssetId(10), AssetId(20)]);
    let rewritten =
        EditableAsset::read_from_path(workspace_root.path().join("consumer.prism")).unwrap();
    assert_eq!(
        rewritten.authoring_payload.inline_bytes(),
        Some(b"ref:12".as_slice())
    );
    assert_eq!(rewritten.subasset_references[0].subasset, 12);
}

#[test]
fn workspace_reimport_rejects_referenced_removal_without_publishing_anything() {
    let workspace_root = tempfile::tempdir().unwrap();
    fs::create_dir(workspace_root.path().join("sources")).unwrap();
    fs::write(
        workspace_root.path().join("sources/hero.input"),
        b"old source",
    )
    .unwrap();
    let target = external_asset(
        AssetId(10),
        ContentHash::digest(b"old source"),
        embedded_asset().subassets,
        Vec::new(),
        b"target",
    );
    let consumer = EditableAsset::new(
        AssetId(20),
        1,
        0,
        Vec::new(),
        SourceRecord::embedded("consumer", "test/raw", b"consumer source".to_vec()).unwrap(),
        Vec::new(),
        Vec::new(),
        vec![EditableSubAssetReference {
            asset: AssetId(10),
            subasset: 9,
            kind: EditableDependencyKind::Hard,
        }],
        b"ref:9".to_vec(),
        None,
    )
    .unwrap();
    let target_path = workspace_root.path().join("hero.prism");
    let consumer_path = workspace_root.path().join("consumer.prism");
    target.write_to_path(&target_path).unwrap();
    consumer.write_to_path(&consumer_path).unwrap();
    let target_before = fs::read(&target_path).unwrap();
    let consumer_before = fs::read(&consumer_path).unwrap();
    fs::write(
        workspace_root.path().join("sources/hero.input"),
        b"new source",
    )
    .unwrap();

    assert_eq!(
        EditableWorkspace::open(workspace_root.path())
            .unwrap()
            .reimport(
                Path::new("hero.prism"),
                &LocalFileSourceProvider::new(workspace_root.path()).unwrap(),
                &test_reimporter(),
                &NoopPayloadRewriter,
                &[],
            ),
        Err(EditableAssetError::ReferencedSubAssetRemoved { id: 9 })
    );
    assert_eq!(fs::read(&target_path).unwrap(), target_before);
    assert_eq!(fs::read(&consumer_path).unwrap(), consumer_before);
}

#[test]
fn workspace_reimport_keeps_every_package_unchanged_when_payload_rewrite_fails() {
    let workspace_root = tempfile::tempdir().unwrap();
    fs::create_dir(workspace_root.path().join("sources")).unwrap();
    fs::write(
        workspace_root.path().join("sources/hero.input"),
        b"old source",
    )
    .unwrap();
    let target = external_asset(
        AssetId(10),
        ContentHash::digest(b"old source"),
        embedded_asset().subassets,
        Vec::new(),
        b"target",
    );
    let consumer = EditableAsset::new(
        AssetId(20),
        1,
        0,
        Vec::new(),
        SourceRecord::embedded("consumer", "test/raw", b"consumer source".to_vec()).unwrap(),
        Vec::new(),
        Vec::new(),
        vec![EditableSubAssetReference {
            asset: AssetId(10),
            subasset: 9,
            kind: EditableDependencyKind::Hard,
        }],
        b"ref:9".to_vec(),
        None,
    )
    .unwrap();
    let target_path = workspace_root.path().join("hero.prism");
    let consumer_path = workspace_root.path().join("consumer.prism");
    target.write_to_path(&target_path).unwrap();
    consumer.write_to_path(&consumer_path).unwrap();
    let target_before = fs::read(&target_path).unwrap();
    let consumer_before = fs::read(&consumer_path).unwrap();
    fs::write(
        workspace_root.path().join("sources/hero.input"),
        b"new source",
    )
    .unwrap();

    assert_eq!(
        EditableWorkspace::open(workspace_root.path())
            .unwrap()
            .reimport(
                Path::new("hero.prism"),
                &LocalFileSourceProvider::new(workspace_root.path()).unwrap(),
                &test_reimporter(),
                &FailingPayloadRewriter,
                &[EditableSubAssetRedirect { from: 9, to: 12 }],
            ),
        Err(EditableAssetError::InvalidSourcePath)
    );
    assert_eq!(fs::read(&target_path).unwrap(), target_before);
    assert_eq!(fs::read(&consumer_path).unwrap(), consumer_before);
}

#[test]
fn workspace_reimport_rejects_duplicate_asset_ids_before_calling_the_importer() {
    let workspace_root = tempfile::tempdir().unwrap();
    let asset = embedded_asset();
    asset
        .write_to_path(workspace_root.path().join("one.prism"))
        .unwrap();
    asset
        .write_to_path(workspace_root.path().join("two.prism"))
        .unwrap();

    assert_eq!(
        EditableWorkspace::open(workspace_root.path())
            .unwrap()
            .reimport(
                Path::new("one.prism"),
                &LocalFileSourceProvider::new(workspace_root.path()).unwrap(),
                &test_reimporter(),
                &NoopPayloadRewriter,
                &[],
            ),
        Err(EditableAssetError::DuplicateWorkspaceAsset { asset: asset.asset })
    );
}

#[test]
fn workspace_snapshot_owns_inputs_and_releases_the_workspace_lock() {
    use fs4::fs_std::FileExt;

    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("asset.prism");
    let original = embedded_asset();
    original.write_to_path(&path).unwrap();
    let workspace = EditableWorkspace::open(directory.path()).unwrap();
    let captured = workspace
        .snapshot(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)
        .unwrap();
    assert_eq!(captured.get(&original.asset), Some(&original));

    let lock = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(directory.path().join(".prism-editable.lock"))
        .unwrap();
    assert!(
        lock.try_lock_exclusive()
            .expect("snapshot lock check must succeed"),
        "snapshot must release the lock before callers cook"
    );
    FileExt::unlock(&lock).unwrap();

    let mut replacement = original.clone();
    replacement.authoring_payload = b"replacement authoring bytes".to_vec().into();
    replacement.subassets[0].name = "renamed".into();
    replacement.source =
        SourceRecord::embedded("textures/hero.png", "image/png", b"new source".to_vec()).unwrap();
    replacement.write_to_path(&path).unwrap();
    assert_eq!(captured.get(&original.asset), Some(&original));
    assert_eq!(
        workspace
            .snapshot(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)
            .unwrap()
            .get(&replacement.asset),
        Some(&replacement)
    );
}

#[test]
fn workspace_snapshot_rejects_duplicate_ids_without_changing_packages() {
    let directory = tempfile::tempdir().unwrap();
    let asset = embedded_asset();
    let bytes = asset.encode().unwrap();
    for name in ["one.prism", "two.prism"] {
        fs::write(directory.path().join(name), &bytes).unwrap();
    }
    let workspace = EditableWorkspace::open(directory.path()).unwrap();
    assert_eq!(
        workspace
            .snapshot(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)
            .map(|_| ()),
        Err(EditableAssetError::DuplicateWorkspaceAsset { asset: asset.asset })
    );
    for name in ["one.prism", "two.prism"] {
        assert_eq!(fs::read(directory.path().join(name)).unwrap(), bytes);
    }
}

#[test]
fn workspace_snapshot_checks_aggregate_budget_and_accepts_the_exact_boundary() {
    let directory = tempfile::tempdir().unwrap();
    let first = embedded_asset();
    let mut second = first.clone();
    second.asset = AssetId(0xA55E8);
    fs::create_dir(directory.path().join("nested")).unwrap();
    first
        .write_to_path(directory.path().join("one.prism"))
        .unwrap();
    second
        .write_to_path(directory.path().join("nested/two.prism"))
        .unwrap();
    let total = (first.encode().unwrap().len() + second.encode().unwrap().len()) as u64;
    let workspace = EditableWorkspace::open(directory.path()).unwrap();
    assert_eq!(
        workspace.snapshot(total - 1).map(|_| ()),
        Err(EditableAssetError::WorkspaceEncodedBytesLimitExceeded {
            actual: total,
            limit: total - 1
        })
    );
    let snapshot = workspace.snapshot(total).unwrap();
    assert_eq!(snapshot.len(), 2);
    assert_eq!(snapshot.get(&first.asset), Some(&first));
    assert_eq!(snapshot.get(&second.asset), Some(&second));
}

#[test]
fn workspace_snapshot_finishes_byte_preflight_before_decoding_any_packages() {
    let directory = tempfile::tempdir().unwrap();
    let malformed = b"malformed .prism bytes";
    fs::write(directory.path().join("a.prism"), malformed).unwrap();
    let bytes = embedded_asset().encode().unwrap();
    fs::write(directory.path().join("z.prism"), &bytes).unwrap();
    let total = (malformed.len() + bytes.len()) as u64;
    let workspace = EditableWorkspace::open(directory.path()).unwrap();
    assert_eq!(
        workspace.snapshot(total - 1).map(|_| ()),
        Err(EditableAssetError::WorkspaceEncodedBytesLimitExceeded {
            actual: total,
            limit: total - 1
        })
    );
    assert_eq!(
        fs::read(directory.path().join("a.prism")).unwrap(),
        malformed
    );
    assert_eq!(fs::read(directory.path().join("z.prism")).unwrap(), bytes);
}

#[test]
fn workspace_snapshot_rejects_malformed_packages_without_rewriting_any_files() {
    let directory = tempfile::tempdir().unwrap();
    let valid = embedded_asset().encode().unwrap();
    let mut malformed = valid.clone();
    malformed[0] ^= 1;
    fs::write(directory.path().join("a.prism"), &valid).unwrap();
    fs::write(directory.path().join("z.prism"), &malformed).unwrap();
    let workspace = EditableWorkspace::open(directory.path()).unwrap();
    assert_eq!(
        workspace
            .snapshot(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)
            .map(|_| ()),
        Err(EditableAssetError::InvalidMagic)
    );
    assert_eq!(fs::read(directory.path().join("a.prism")).unwrap(), valid);
    assert_eq!(
        fs::read(directory.path().join("z.prism")).unwrap(),
        malformed
    );
    assert!(!directory
        .path()
        .join(".prism-editable-transactions")
        .exists());
}

#[test]
fn workspace_snapshot_budget_must_be_positive_and_within_the_hard_cap() {
    let directory = tempfile::tempdir().unwrap();
    let workspace = EditableWorkspace::open(directory.path()).unwrap();
    for requested in [0, MAX_EDITABLE_WORKSPACE_ENCODED_BYTES + 1, u64::MAX] {
        assert_eq!(
            workspace.snapshot(requested).map(|_| ()),
            Err(EditableAssetError::InvalidWorkspaceSnapshotBudget {
                requested,
                limit: MAX_EDITABLE_WORKSPACE_ENCODED_BYTES
            })
        );
    }
    assert!(workspace.snapshot(1).unwrap().is_empty());
    assert!(workspace
        .snapshot(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)
        .unwrap()
        .is_empty());
}

#[test]
fn workspace_snapshot_checks_explicit_reference_graph() {
    let directory = tempfile::tempdir().unwrap();
    let target = embedded_asset();
    let target_path = directory.path().join("target.prism");
    target.write_to_path(&target_path).unwrap();
    let mut consumer = embedded_asset();
    consumer.asset = AssetId(0xA55E8);
    consumer.subasset_references = vec![EditableSubAssetReference {
        asset: target.asset,
        subasset: target.subassets[0].id,
        kind: EditableDependencyKind::Hard,
    }];
    let consumer_path = directory.path().join("consumer.prism");
    consumer.write_to_path(&consumer_path).unwrap();
    let workspace = EditableWorkspace::open(directory.path()).unwrap();
    assert_eq!(
        workspace
            .snapshot(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)
            .unwrap()
            .len(),
        2
    );
    consumer.subasset_references[0].subasset = u64::MAX;
    consumer.write_to_path(&consumer_path).unwrap();
    assert_eq!(
        workspace
            .snapshot(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)
            .map(|_| ()),
        Err(EditableAssetError::MissingReferencedSubAsset {
            asset: target.asset,
            subasset: u64::MAX
        })
    );
    consumer.subasset_references[0].asset = AssetId(99);
    consumer.write_to_path(&consumer_path).unwrap();
    assert_eq!(
        workspace
            .snapshot(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)
            .map(|_| ()),
        Err(EditableAssetError::MissingReferencedAsset { asset: AssetId(99) })
    );
    assert_eq!(EditableAsset::read_from_path(target_path).unwrap(), target);
    assert_eq!(
        EditableAsset::read_from_path(consumer_path).unwrap(),
        consumer
    );
}

#[test]
fn workspace_snapshot_allows_absent_soft_targets_but_checks_present_subassets() {
    let directory = tempfile::tempdir().unwrap();
    let target = embedded_asset();
    let mut consumer = embedded_asset();
    consumer.asset = AssetId(0xA55E8);
    consumer.subasset_references = vec![EditableSubAssetReference {
        asset: target.asset,
        subasset: u64::MAX,
        kind: EditableDependencyKind::Soft,
    }];
    let consumer_path = directory.path().join("consumer.prism");
    consumer.write_to_path(&consumer_path).unwrap();
    let workspace = EditableWorkspace::open(directory.path()).unwrap();
    assert_eq!(
        workspace
            .snapshot(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)
            .unwrap()
            .len(),
        1
    );
    for kind in [
        EditableDependencyKind::Hard,
        EditableDependencyKind::EditorOnly,
    ] {
        consumer.subasset_references[0].kind = kind;
        consumer.write_to_path(&consumer_path).unwrap();
        assert_eq!(
            workspace
                .snapshot(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)
                .map(|_| ()),
            Err(EditableAssetError::MissingReferencedAsset {
                asset: target.asset
            })
        );
    }
    consumer.subasset_references[0].kind = EditableDependencyKind::Soft;
    consumer.write_to_path(&consumer_path).unwrap();
    target
        .write_to_path(directory.path().join("target.prism"))
        .unwrap();
    assert_eq!(
        workspace
            .snapshot(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)
            .map(|_| ()),
        Err(EditableAssetError::MissingReferencedSubAsset {
            asset: target.asset,
            subasset: u64::MAX
        })
    );
    consumer.subasset_references[0].subasset = target.subassets[0].id;
    consumer.write_to_path(&consumer_path).unwrap();
    assert_eq!(
        workspace
            .snapshot(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn workspace_reimport_preserves_unrelated_missing_soft_references() {
    let directory = tempfile::tempdir().unwrap();
    let target = embedded_asset();
    target
        .write_to_path(directory.path().join("target.prism"))
        .unwrap();
    let mut consumer = embedded_asset();
    consumer.asset = AssetId(20);
    consumer.subasset_references = vec![EditableSubAssetReference {
        asset: AssetId(99),
        subasset: 1,
        kind: EditableDependencyKind::Soft,
    }];
    let consumer_path = directory.path().join("consumer.prism");
    consumer.write_to_path(&consumer_path).unwrap();
    let before = fs::read(&consumer_path).unwrap();
    let workspace = EditableWorkspace::open(directory.path()).unwrap();
    let result = workspace
        .reimport(
            Path::new("target.prism"),
            &LocalFileSourceProvider::new(directory.path()).unwrap(),
            &test_reimporter(),
            &NoopPayloadRewriter,
            &[],
        )
        .unwrap();
    assert_eq!(result.updated_assets, vec![target.asset]);
    assert_eq!(fs::read(&consumer_path).unwrap(), before);
    let captured = workspace
        .snapshot(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)
        .unwrap();
    assert_eq!(captured.get(&consumer.asset), Some(&consumer));
    assert_eq!(captured[&target.asset].subassets, changed_subassets());
}

#[test]
fn workspace_reimport_obeys_the_hard_aggregate_scan_budget_before_decoding() {
    let directory = tempfile::tempdir().unwrap();
    // Sparse files exercise the real 512 MiB metadata cap without allocating or
    // reading that payload. Their invalid contents must never reach the decoder.
    let count = MAX_EDITABLE_WORKSPACE_ENCODED_BYTES / MAX_EDITABLE_ASSET_BYTES as u64 + 1;
    for index in 0..count {
        fs::File::create(directory.path().join(format!("{index}.prism")))
            .unwrap()
            .set_len(MAX_EDITABLE_ASSET_BYTES as u64)
            .unwrap();
    }
    let workspace = EditableWorkspace::open(directory.path()).unwrap();
    let expected = EditableAssetError::WorkspaceEncodedBytesLimitExceeded {
        actual: count * MAX_EDITABLE_ASSET_BYTES as u64,
        limit: MAX_EDITABLE_WORKSPACE_ENCODED_BYTES,
    };
    assert_eq!(
        workspace
            .snapshot(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)
            .map(|_| ()),
        Err(expected.clone())
    );
    assert_eq!(
        workspace.reimport(
            Path::new("0.prism"),
            &LocalFileSourceProvider::new(directory.path()).unwrap(),
            &test_reimporter(),
            &NoopPayloadRewriter,
            &[],
        ),
        Err(expected)
    );
    for index in 0..count {
        assert_eq!(
            fs::metadata(directory.path().join(format!("{index}.prism")))
                .unwrap()
                .len(),
            MAX_EDITABLE_ASSET_BYTES as u64
        );
    }
}

#[test]
fn workspace_snapshot_has_a_finite_iterative_directory_depth() {
    let directory = tempfile::tempdir().unwrap();
    let workspace = EditableWorkspace::open(directory.path()).unwrap();
    let mut deepest = directory.path().to_path_buf();
    for _ in 0..64 {
        deepest.push("d");
        fs::create_dir(&deepest).unwrap();
    }
    assert!(workspace.snapshot(1).unwrap().is_empty());
    fs::create_dir(deepest.join("excess")).unwrap();
    assert_eq!(
        workspace.snapshot(1).map(|_| ()),
        Err(EditableAssetError::WorkspaceDepthLimitExceeded { limit: 64 })
    );
}

#[cfg(unix)]
#[test]
fn workspace_snapshot_skips_symlink_cycles_and_external_packages() {
    use std::os::unix::fs::symlink;
    let directory = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    fs::write(outside.path().join("bad.prism"), b"malformed").unwrap();
    symlink(directory.path(), directory.path().join("cycle")).unwrap();
    symlink(outside.path(), directory.path().join("external")).unwrap();
    symlink(
        outside.path().join("bad.prism"),
        directory.path().join("linked.prism"),
    )
    .unwrap();
    let asset = embedded_asset();
    asset
        .write_to_path(directory.path().join("asset.prism"))
        .unwrap();
    let snapshot = EditableWorkspace::open(directory.path())
        .unwrap()
        .snapshot(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)
        .unwrap();
    assert_eq!(snapshot.len(), 1);
    assert_eq!(snapshot.get(&asset.asset), Some(&asset));
}

#[test]
fn workspace_reimport_rejects_missing_explicit_reference_targets_without_publishing() {
    let workspace_root = tempfile::tempdir().unwrap();
    fs::create_dir(workspace_root.path().join("sources")).unwrap();
    fs::write(
        workspace_root.path().join("sources/hero.input"),
        b"old source",
    )
    .unwrap();
    let target = external_asset(
        AssetId(10),
        ContentHash::digest(b"old source"),
        embedded_asset().subassets,
        Vec::new(),
        b"target",
    );
    let consumer = EditableAsset::new(
        AssetId(20),
        1,
        0,
        Vec::new(),
        SourceRecord::embedded("consumer", "test/raw", b"consumer source".to_vec()).unwrap(),
        Vec::new(),
        Vec::new(),
        vec![EditableSubAssetReference {
            asset: AssetId(99),
            subasset: 1,
            kind: EditableDependencyKind::Hard,
        }],
        b"ref:missing".to_vec(),
        None,
    )
    .unwrap();
    let target_path = workspace_root.path().join("hero.prism");
    target.write_to_path(&target_path).unwrap();
    consumer
        .write_to_path(workspace_root.path().join("consumer.prism"))
        .unwrap();
    let before = fs::read(&target_path).unwrap();
    fs::write(
        workspace_root.path().join("sources/hero.input"),
        b"new source",
    )
    .unwrap();

    assert_eq!(
        EditableWorkspace::open(workspace_root.path())
            .unwrap()
            .reimport(
                Path::new("hero.prism"),
                &LocalFileSourceProvider::new(workspace_root.path()).unwrap(),
                &test_reimporter(),
                &NoopPayloadRewriter,
                &[],
            ),
        Err(EditableAssetError::MissingReferencedAsset { asset: AssetId(99) })
    );
    assert_eq!(fs::read(&target_path).unwrap(), before);
}

#[test]
fn workspace_reimport_rejects_missing_explicit_subassets_without_publishing() {
    let workspace_root = tempfile::tempdir().unwrap();
    fs::create_dir(workspace_root.path().join("sources")).unwrap();
    fs::write(
        workspace_root.path().join("sources/hero.input"),
        b"old source",
    )
    .unwrap();
    let target = external_asset(
        AssetId(10),
        ContentHash::digest(b"old source"),
        embedded_asset().subassets,
        Vec::new(),
        b"target",
    );
    let consumer = EditableAsset::new(
        AssetId(20),
        1,
        0,
        Vec::new(),
        SourceRecord::embedded("consumer", "test/raw", b"consumer source".to_vec()).unwrap(),
        Vec::new(),
        Vec::new(),
        vec![EditableSubAssetReference {
            asset: AssetId(10),
            subasset: 99,
            kind: EditableDependencyKind::Hard,
        }],
        b"ref:missing".to_vec(),
        None,
    )
    .unwrap();
    let target_path = workspace_root.path().join("hero.prism");
    target.write_to_path(&target_path).unwrap();
    consumer
        .write_to_path(workspace_root.path().join("consumer.prism"))
        .unwrap();
    let before = fs::read(&target_path).unwrap();
    fs::write(
        workspace_root.path().join("sources/hero.input"),
        b"new source",
    )
    .unwrap();

    assert_eq!(
        EditableWorkspace::open(workspace_root.path())
            .unwrap()
            .reimport(
                Path::new("hero.prism"),
                &LocalFileSourceProvider::new(workspace_root.path()).unwrap(),
                &test_reimporter(),
                &NoopPayloadRewriter,
                &[],
            ),
        Err(EditableAssetError::MissingReferencedSubAsset {
            asset: AssetId(10),
            subasset: 99,
        })
    );
    assert_eq!(fs::read(&target_path).unwrap(), before);
}

#[test]
fn subasset_references_must_be_strictly_canonical() {
    let asset = embedded_asset();
    assert_eq!(
        EditableAsset::new(
            asset.asset,
            asset.kind,
            asset.flags,
            asset.canonical_settings,
            asset.source,
            asset.subassets,
            asset.dependencies,
            vec![
                EditableSubAssetReference {
                    asset: AssetId(9),
                    subasset: 2,
                    kind: EditableDependencyKind::Hard,
                },
                EditableSubAssetReference {
                    asset: AssetId(8),
                    subasset: 1,
                    kind: EditableDependencyKind::Hard,
                },
            ],
            asset.authoring_payload,
            asset.preview,
        ),
        Err(EditableAssetError::NonCanonicalSubAssetReferences)
    );
}
