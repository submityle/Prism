use prism_asset::StableGuid;
use prism_asset_build_core::{
    import::IMAGE_ARTIFACT_KIND, AssetId, BuildError, BuildExecutionError, BuildResources,
    BuildRevision, CanonicalEditableImageCooker, DerivedDataCache, EditableAsset,
    EditableBuildError, EditableBuildOptions, EditableBuildPlan, EditableCookDependency,
    EditableCookInput, EditableCooker, EditableCookerDescriptor, EditableCookerRegistry,
    EditableDependency, EditableDependencyKind as Kind, EditableSubAsset,
    EditableSubAssetReference, EditableWorkspace, JobId, PackageOptions, ProcessorError,
    ProcessorIdentity, RevisionStore, SourceRecord,
};
use prism_asset_container::{ContainerReader, OverlaySet, RuntimeManifest};
use prism_image::{
    AlphaCoveragePolicy, ImageArtifactMipLevel, ImageMipArtifact, RuntimeImageLoader,
    TextureSemantic,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

fn asset(id: u128, payload: &[u8]) -> EditableAsset {
    EditableAsset::new(
        AssetId(id),
        42,
        0,
        b"settings".to_vec(),
        SourceRecord::embedded("source.bin", "test/raw", b"old source".to_vec()).unwrap(),
        vec![EditableSubAsset {
            id: 4,
            name: "part".into(),
            kind: 42,
        }],
        Vec::new(),
        Vec::new(),
        payload.to_vec(),
        None,
    )
    .unwrap()
}

fn dependency(id: u128, kind: Kind) -> EditableDependency {
    EditableDependency {
        asset: AssetId(id),
        kind,
    }
}

fn snapshot(assets: Vec<EditableAsset>) -> BTreeMap<AssetId, EditableAsset> {
    assets.into_iter().map(|a| (a.asset, a)).collect()
}

struct Cooker {
    calls: Arc<AtomicUsize>,
    descriptions: Arc<AtomicUsize>,
    payload_pointer: Option<Arc<AtomicUsize>>,
    max_output_bytes: u64,
}
impl EditableCooker for Cooker {
    fn descriptor(&self) -> EditableCookerDescriptor {
        self.descriptions.fetch_add(1, Ordering::SeqCst);
        EditableCookerDescriptor {
            authoring_kind: 42,
            identity: ProcessorIdentity::new(123, 1),
            output_kind: 43,
            output_schema: 1,
            alignment: 4,
            max_output_bytes: self.max_output_bytes,
            resources: BuildResources {
                memory_bytes: 1024,
                ..BuildResources::default()
            },
        }
    }
    fn cook(
        &self,
        input: &EditableCookInput,
        dependencies: &[EditableCookDependency<'_>],
    ) -> Result<Vec<u8>, ProcessorError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if let Some(pointer) = &self.payload_pointer {
            let address = input.authoring_payload.as_ptr() as usize;
            let first = pointer
                .compare_exchange(0, address, Ordering::SeqCst, Ordering::SeqCst)
                .unwrap_or_else(|seen| seen);
            assert!(first == 0 || first == address);
        }
        assert!(input
            .dependencies
            .iter()
            .all(|d| d.kind != Kind::EditorOnly));
        assert!(input
            .subasset_references
            .iter()
            .all(|r| r.kind != Kind::EditorOnly));
        assert!(dependencies.windows(2).all(|d| d[0].asset < d[1].asset));
        if input.authoring_payload.as_ref() == b"fail" {
            return Err(ProcessorError::new("deliberate cook failure"));
        }
        let mut bytes = input.authoring_payload.to_vec();
        for dependency in dependencies {
            bytes.extend_from_slice(dependency.bytes);
        }
        Ok(bytes)
    }
}

fn registry() -> (EditableCookerRegistry, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let descriptions = Arc::new(AtomicUsize::new(0));
    let mut registry = EditableCookerRegistry::new();
    registry
        .register(Cooker {
            calls: calls.clone(),
            descriptions: descriptions.clone(),
            payload_pointer: None,
            max_output_bytes: 4096,
        })
        .unwrap();
    (registry, calls, descriptions)
}

fn plan(
    assets: Vec<EditableAsset>,
    roots: &[u128],
    registry: &EditableCookerRegistry,
) -> Result<EditableBuildPlan, EditableBuildError> {
    EditableBuildPlan::from_snapshot(
        snapshot(assets),
        &roots.iter().map(|id| AssetId(*id)).collect::<Vec<_>>(),
        registry,
        &EditableBuildOptions::default(),
    )
}

#[test]
fn saved_image_payload_survives_workspace_snapshot_cook_package_and_runtime_load() {
    let root = tempfile::tempdir().unwrap();
    let image = ImageMipArtifact {
        width: 2,
        height: 2,
        semantic: TextureSemantic::ColorSrgb,
        alpha: AlphaCoveragePolicy::Preserve,
        levels: vec![
            ImageArtifactMipLevel {
                width: 2,
                height: 2,
                rgba8: [17, 31, 47, 255].repeat(4),
            },
            ImageArtifactMipLevel {
                width: 1,
                height: 1,
                rgba8: vec![19, 33, 49, 255],
            },
        ],
    };
    let mut saved = asset(7, &image.encode().unwrap());
    saved.kind = IMAGE_ARTIFACT_KIND;
    saved.dependencies = vec![
        dependency(999, Kind::Soft),
        dependency(1000, Kind::EditorOnly),
    ];
    saved.subasset_references = vec![EditableSubAssetReference {
        asset: AssetId(999),
        subasset: 4,
        kind: Kind::Soft,
    }];
    saved
        .write_to_path(root.path().join("edited.prism"))
        .unwrap();
    let workspace = EditableWorkspace::open(root.path()).unwrap();
    let plan = EditableBuildPlan::from_workspace(
        &workspace,
        &[AssetId(7)],
        &EditableCookerRegistry::with_builtin(),
        &EditableBuildOptions::default(),
    )
    .unwrap();
    // Replacing the persisted payload after planning cannot change the frozen job.
    saved.authoring_payload = b"bad later edit".to_vec().into();
    saved
        .write_to_path(root.path().join("edited.prism"))
        .unwrap();
    let store = RevisionStore::open(root.path().join("build")).unwrap();
    let mut ddc = DerivedDataCache::open(root.path().join("ddc")).unwrap();
    let execution = plan
        .execute(
            &store,
            &mut ddc,
            BuildRevision(0),
            2,
            BuildResources::UNLIMITED,
            true,
        )
        .unwrap();
    assert_eq!(execution.published.manifest.entries.len(), 1);
    assert_eq!(
        store
            .cas()
            .read(execution.published.manifest.entries[&AssetId(7)].content_hash)
            .unwrap(),
        image.encode().unwrap(),
    );
    let package = plan
        .package(&execution, store.cas(), "images", PackageOptions::default())
        .unwrap();
    assert_eq!(
        package,
        plan.package(&execution, store.cas(), "images", PackageOptions::default())
            .unwrap()
    );
    let reopened = RevisionStore::open(root.path().join("build")).unwrap();
    assert_eq!(
        package,
        plan.package_manifest(
            &reopened.manifest(BuildRevision(0)).unwrap(),
            reopened.cas(),
            "images",
            PackageOptions::default(),
        )
        .unwrap()
    );
    let manifest = RuntimeManifest::decode(&package.runtime_manifest_bytes).unwrap();
    let guid = StableGuid::from_u128(7);
    assert_eq!(
        manifest.metadata(guid).unwrap().optional_dependencies,
        [StableGuid::from_u128(999)]
    );
    let mut overlays = OverlaySet::new();
    overlays.mount(
        "images",
        0,
        ContainerReader::new(package.container_bytes("images").unwrap().to_vec()).unwrap(),
    );
    let loaded = RuntimeImageLoader::new(manifest, overlays)
        .load(guid)
        .unwrap();
    assert_eq!(loaded.data(), image.levels[0].rgba8);
    assert_eq!(loaded.mip_level_count(), 2);
    assert_eq!(loaded.mip_data(1).unwrap(), image.levels[1].rgba8);
}

#[test]
fn builtin_image_cooker_preserves_canonical_bytes_and_rejects_malformed_mips() {
    let mut input = EditableCookInput {
        asset: AssetId(7),
        kind: IMAGE_ARTIFACT_KIND,
        flags: 0,
        canonical_settings: Vec::new(),
        subassets: Vec::new(),
        dependencies: Vec::new(),
        subasset_references: Vec::new(),
        authoring_payload: Vec::new().into(),
    };
    for semantic in [
        TextureSemantic::ColorSrgb,
        TextureSemantic::ColorLinear,
        TextureSemantic::NormalMap,
        TextureSemantic::Mask,
    ] {
        for alpha in [
            AlphaCoveragePolicy::Preserve,
            AlphaCoveragePolicy::Coverage { threshold: 128 },
            AlphaCoveragePolicy::Mask { threshold: 64 },
        ] {
            input.authoring_payload = ImageMipArtifact {
                width: 2,
                height: 2,
                semantic,
                alpha,
                levels: vec![
                    ImageArtifactMipLevel {
                        width: 2,
                        height: 2,
                        rgba8: [17, 31, 47, 255].repeat(4),
                    },
                    ImageArtifactMipLevel {
                        width: 1,
                        height: 1,
                        rgba8: vec![19, 33, 49, 255],
                    },
                ],
            }
            .encode()
            .unwrap()
            .into();
            assert_eq!(
                CanonicalEditableImageCooker.cook(&input, &[]).unwrap(),
                input.authoring_payload.as_ref(),
            );
        }
    }
    let valid = input.authoring_payload.to_vec();
    let mut trailing = valid.clone();
    trailing.push(0);
    let mut wrong_schema = valid.clone();
    wrong_schema[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
    let mut zero_levels = valid.clone();
    zero_levels[24..28].copy_from_slice(&0u32.to_le_bytes());
    let mut invalid_level = valid.clone();
    // The first level header follows the 28-byte versioned image header.
    invalid_level[28..32].copy_from_slice(&0u32.to_le_bytes());
    let mut wrong_width = valid.clone();
    wrong_width[12..16].copy_from_slice(&3u32.to_le_bytes());
    let mut wrong_height = valid.clone();
    wrong_height[16..20].copy_from_slice(&0u32.to_le_bytes());
    for malformed in [
        valid[..valid.len() - 1].to_vec(),
        trailing,
        wrong_schema,
        zero_levels,
        invalid_level,
        wrong_width,
        wrong_height,
    ] {
        input.authoring_payload = malformed.into();
        assert!(CanonicalEditableImageCooker.cook(&input, &[]).is_err());
    }
}

#[test]
fn builtin_image_output_budget_is_checked_before_decoding() {
    let output_limit = CanonicalEditableImageCooker.descriptor().max_output_bytes as usize;
    let input = EditableCookInput {
        asset: AssetId(7),
        kind: IMAGE_ARTIFACT_KIND,
        flags: 0,
        canonical_settings: Vec::new(),
        subassets: Vec::new(),
        dependencies: Vec::new(),
        subasset_references: Vec::new(),
        // Deliberately malformed too: admission must reject its size before
        // reaching format validation or allocating decoded pixel buffers.
        authoring_payload: vec![0; output_limit + 1].into(),
    };
    assert_eq!(
        CanonicalEditableImageCooker
            .cook(&input, &[])
            .unwrap_err()
            .to_string(),
        "editable image payload exceeds declared output byte bound",
    );
}

#[test]
fn stricter_image_validation_cannot_reuse_a_legacy_warm_artifact() {
    struct LegacyImageCooker;
    impl EditableCooker for LegacyImageCooker {
        fn descriptor(&self) -> EditableCookerDescriptor {
            let mut descriptor = CanonicalEditableImageCooker.descriptor();
            descriptor.identity.version = 1;
            descriptor
        }

        fn cook(
            &self,
            input: &EditableCookInput,
            _: &[EditableCookDependency<'_>],
        ) -> Result<Vec<u8>, ProcessorError> {
            ImageMipArtifact::decode(&input.authoring_payload)
                .and_then(|image| image.encode())
                .map_err(|error| ProcessorError::new(error.to_string()))
        }
    }
    let payload = ImageMipArtifact {
        // Version 1 accepted a header inconsistent with the first mip.
        width: 2,
        height: 1,
        semantic: TextureSemantic::ColorLinear,
        alpha: AlphaCoveragePolicy::Preserve,
        levels: vec![ImageArtifactMipLevel {
            width: 1,
            height: 1,
            rgba8: vec![17, 31, 47, 255],
        }],
    }
    .encode()
    .unwrap();
    let mut saved = asset(7, &payload);
    saved.kind = IMAGE_ARTIFACT_KIND;
    let mut legacy_registry = EditableCookerRegistry::new();
    legacy_registry.register(LegacyImageCooker).unwrap();
    let legacy = plan(vec![saved.clone()], &[7], &legacy_registry).unwrap();
    let current = plan(vec![saved], &[7], &EditableCookerRegistry::with_builtin()).unwrap();
    let legacy_spec = &legacy.specs()[&JobId(7)];
    let current_spec = &current.specs()[&JobId(7)];
    assert_eq!(legacy_spec.processor_id, current_spec.processor_id);
    assert_ne!(
        legacy_spec.artifact_key(1, &[]).unwrap(),
        current_spec.artifact_key(1, &[]).unwrap(),
    );
    let root = tempfile::tempdir().unwrap();
    let store = RevisionStore::open(root.path()).unwrap();
    let mut ddc = DerivedDataCache::open(root.path().join("ddc")).unwrap();
    let baseline = legacy
        .execute(
            &store,
            &mut ddc,
            BuildRevision(0),
            1,
            BuildResources::UNLIMITED,
            false,
        )
        .unwrap();
    assert!(matches!(
        current.execute(
            &store,
            &mut ddc,
            BuildRevision(1),
            1,
            BuildResources::UNLIMITED,
            false,
        ),
        Err(EditableBuildError::Execution(BuildExecutionError::Graph(
            BuildError::ProcessorFailed { job: JobId(7), .. }
        )))
    ));
    let old_descriptor = &baseline.published.manifest.entries[&AssetId(7)];
    assert_eq!(store.current_revision().unwrap(), Some(BuildRevision(0)));
    assert_eq!(ddc.get(old_descriptor.key).as_ref(), Some(old_descriptor));
    assert_eq!(
        store.cas().read(old_descriptor.content_hash).unwrap(),
        payload
    );
}

#[test]
fn hard_closure_is_ordered_while_soft_and_editor_only_targets_do_not_cook() {
    let (registry, calls, descriptions) = registry();
    let mut parent = asset(9, b"parent");
    parent.dependencies = vec![
        dependency(1, Kind::Hard),
        dependency(4, Kind::Hard),
        dependency(20, Kind::Soft),
        dependency(30, Kind::EditorOnly),
    ];
    let mut ignored = asset(20, b"ignored");
    ignored.kind = 999;
    let plan = plan(
        vec![parent, asset(4, b"four"), asset(1, b"one"), ignored],
        &[9],
        &registry,
    )
    .unwrap();
    assert_eq!(
        plan.graph().execution_order().unwrap(),
        [JobId(1), JobId(4), JobId(9)]
    );
    let root = tempfile::tempdir().unwrap();
    let store = RevisionStore::open(root.path()).unwrap();
    let mut ddc = DerivedDataCache::open(root.path().join("ddc")).unwrap();
    let execution = plan
        .execute(
            &store,
            &mut ddc,
            BuildRevision(0),
            2,
            BuildResources::UNLIMITED,
            false,
        )
        .unwrap();
    let output = &execution.published.manifest.entries[&AssetId(9)];
    assert_eq!(
        store.cas().read(output.content_hash).unwrap(),
        b"parentonefour"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(descriptions.load(Ordering::SeqCst), 1);
}

#[test]
fn semantic_edit_rebuilds_only_its_successors_and_editor_metadata_reuses_cache() {
    let (registry, calls, _) = registry();
    let root = tempfile::tempdir().unwrap();
    let store = RevisionStore::open(root.path()).unwrap();
    let mut ddc = DerivedDataCache::open(root.path().join("ddc")).unwrap();
    let mut source = asset(1, b"original");
    let mut consumer = asset(2, b"consumer");
    consumer.dependencies = vec![dependency(1, Kind::Hard)];
    let build = |source: EditableAsset| {
        plan(
            vec![source, consumer.clone(), asset(3, b"unrelated")],
            &[2, 3],
            &registry,
        )
        .unwrap()
    };
    build(source.clone())
        .execute(
            &store,
            &mut ddc,
            BuildRevision(0),
            3,
            BuildResources::UNLIMITED,
            false,
        )
        .unwrap();
    source.source =
        SourceRecord::embedded("moved.bin", "new/format", b"different raw source".to_vec())
            .unwrap();
    source.preview = Some(b"different preview".to_vec());
    source.dependencies = vec![dependency(999, Kind::EditorOnly)];
    source.subasset_references = vec![EditableSubAssetReference {
        asset: AssetId(999),
        subasset: 4,
        kind: Kind::EditorOnly,
    }];
    let warm = build(source.clone())
        .execute(
            &store,
            &mut ddc,
            BuildRevision(1),
            3,
            BuildResources::UNLIMITED,
            false,
        )
        .unwrap();
    assert_eq!(
        warm.cache_hits,
        BTreeSet::from([JobId(1), JobId(2), JobId(3)])
    );
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    source.authoring_payload = b"edited".to_vec().into();
    let next = build(source)
        .execute(
            &store,
            &mut ddc,
            BuildRevision(2),
            3,
            BuildResources::UNLIMITED,
            false,
        )
        .unwrap();
    assert_eq!(next.cache_hits, BTreeSet::from([JobId(3)]));
    assert_eq!(calls.load(Ordering::SeqCst), 5);
    let descriptor = &next.published.manifest.entries[&AssetId(2)];
    assert_eq!(
        store.cas().read(descriptor.content_hash).unwrap(),
        b"consumeredited"
    );
}

#[test]
fn all_cooker_visible_metadata_and_build_contracts_participate_in_identity() {
    let (registry, _, _) = registry();
    let original = asset(1, b"payload");
    let key = |value: EditableAsset, options: &EditableBuildOptions| {
        let plan = EditableBuildPlan::from_snapshot(
            snapshot(vec![value]),
            &[AssetId(1)],
            &registry,
            options,
        )
        .unwrap();
        plan.specs()[&JobId(1)].artifact_key(1, &[]).unwrap()
    };
    let options = EditableBuildOptions::default();
    let expected = key(original.clone(), &options);
    let mut variants = Vec::new();
    let mut a = original.clone();
    a.flags = 1;
    variants.push(a);
    let mut a = original.clone();
    a.canonical_settings.push(1);
    variants.push(a);
    let mut a = original.clone();
    a.subassets[0].name = "renamed".into();
    variants.push(a);
    let mut a = original.clone();
    a.subassets[0].kind = 99;
    variants.push(a);
    let mut a = original.clone();
    a.dependencies = vec![dependency(99, Kind::Soft)];
    variants.push(a);
    let mut a = original.clone();
    a.subasset_references = vec![EditableSubAssetReference {
        asset: AssetId(99),
        subasset: 4,
        kind: Kind::Soft,
    }];
    variants.push(a);
    for value in variants {
        assert_ne!(key(value, &options), expected);
    }
    for options in [
        EditableBuildOptions {
            profile_hash: [1; 32],
            ..Default::default()
        },
        EditableBuildOptions {
            toolchain_fingerprint: b"tool".to_vec(),
            ..Default::default()
        },
        EditableBuildOptions {
            environment_fingerprint: b"env".to_vec(),
            ..Default::default()
        },
    ] {
        assert_ne!(key(original.clone(), &options), expected);
    }
}

#[test]
fn missing_hard_invalid_subasset_and_cycles_reject_before_execution() {
    let (registry, calls, _) = registry();
    let mut first = asset(1, b"one");
    first.dependencies = vec![dependency(2, Kind::Hard)];
    assert!(matches!(
        plan(vec![first.clone()], &[1], &registry),
        Err(EditableBuildError::MissingAsset(AssetId(2)))
    ));
    let mut second = asset(2, b"two");
    second.dependencies = vec![dependency(1, Kind::Hard)];
    assert!(matches!(
        plan(vec![first, second], &[1], &registry),
        Err(EditableBuildError::Graph(_))
    ));
    let mut bad = asset(1, b"one");
    bad.subasset_references = vec![EditableSubAssetReference {
        asset: AssetId(2),
        subasset: 999,
        kind: Kind::Soft,
    }];
    assert!(matches!(
        plan(vec![bad, asset(2, b"two")], &[1], &registry),
        Err(EditableBuildError::MissingSubAsset { .. })
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn internal_subasset_references_do_not_form_artifact_cycles() {
    let (registry, _, _) = registry();
    let mut value = asset(1, b"one");
    value.subasset_references = vec![EditableSubAssetReference {
        asset: AssetId(1),
        subasset: 4,
        kind: Kind::Hard,
    }];
    let built = plan(vec![value.clone()], &[1], &registry).unwrap();
    assert_eq!(built.graph().execution_order().unwrap(), [JobId(1)]);
    value.subasset_references[0].subasset = 999;
    assert!(matches!(
        plan(vec![value], &[1], &registry),
        Err(EditableBuildError::MissingSubAsset { .. })
    ));
}

#[test]
fn optional_cycles_roundtrip_without_entering_the_artifact_dag() {
    let (registry, _, _) = registry();
    let mut first = asset(1, b"one");
    first.dependencies = vec![dependency(2, Kind::Soft)];
    let mut second = asset(2, b"two");
    second.dependencies = vec![dependency(1, Kind::Soft)];
    let plan = plan(vec![first, second], &[1, 2], &registry).unwrap();
    let root = tempfile::tempdir().unwrap();
    let store = RevisionStore::open(root.path()).unwrap();
    let mut ddc = DerivedDataCache::open(root.path().join("ddc")).unwrap();
    let execution = plan
        .execute(
            &store,
            &mut ddc,
            BuildRevision(0),
            2,
            BuildResources::UNLIMITED,
            false,
        )
        .unwrap();
    let package = plan
        .package(&execution, store.cas(), "data", PackageOptions::default())
        .unwrap();
    let decoded = RuntimeManifest::decode(&package.runtime_manifest_bytes).unwrap();
    assert_eq!(
        decoded
            .metadata(StableGuid::from_u128(1))
            .unwrap()
            .optional_dependencies,
        [StableGuid::from_u128(2)]
    );
    assert_eq!(
        decoded
            .metadata(StableGuid::from_u128(2))
            .unwrap()
            .optional_dependencies,
        [StableGuid::from_u128(1)]
    );
}

#[test]
fn resource_output_and_cooker_failures_preserve_the_published_revision() {
    let (registry, calls, _) = registry();
    let root = tempfile::tempdir().unwrap();
    let store = RevisionStore::open(root.path()).unwrap();
    let mut ddc = DerivedDataCache::open(root.path().join("ddc")).unwrap();
    let good = plan(vec![asset(1, b"good")], &[1], &registry).unwrap();
    good.execute(
        &store,
        &mut ddc,
        BuildRevision(0),
        1,
        BuildResources::UNLIMITED,
        false,
    )
    .unwrap();
    let changed = plan(vec![asset(1, b"new")], &[1], &registry).unwrap();
    assert!(changed
        .execute(
            &store,
            &mut ddc,
            BuildRevision(1),
            1,
            BuildResources {
                memory_bytes: 1,
                ..BuildResources::UNLIMITED
            },
            false
        )
        .is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let failing = plan(vec![asset(1, b"fail")], &[1], &registry).unwrap();
    assert!(failing
        .execute(
            &store,
            &mut ddc,
            BuildRevision(1),
            1,
            BuildResources::UNLIMITED,
            false
        )
        .is_err());
    let oversized = plan(vec![asset(1, &vec![0; 4097])], &[1], &registry).unwrap();
    assert!(oversized
        .execute(
            &store,
            &mut ddc,
            BuildRevision(1),
            1,
            BuildResources::UNLIMITED,
            false
        )
        .is_err());
    assert_eq!(store.current_revision().unwrap(), Some(BuildRevision(0)));
}

#[test]
fn tightened_output_contract_cannot_reuse_an_oversized_warm_artifact() {
    let (registry, calls, _) = registry();
    let root = tempfile::tempdir().unwrap();
    let store = RevisionStore::open(root.path()).unwrap();
    let mut ddc = DerivedDataCache::open(root.path().join("ddc")).unwrap();
    let original = plan(vec![asset(1, b"oversized")], &[1], &registry).unwrap();
    let first = original
        .execute(
            &store,
            &mut ddc,
            BuildRevision(0),
            1,
            BuildResources::UNLIMITED,
            false,
        )
        .unwrap();
    let warm = original
        .execute(
            &store,
            &mut ddc,
            BuildRevision(1),
            1,
            BuildResources::UNLIMITED,
            false,
        )
        .unwrap();
    assert_eq!(warm.cache_hits, BTreeSet::from([JobId(1)]));
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let mut tightened_registry = EditableCookerRegistry::new();
    tightened_registry
        .register(Cooker {
            calls: calls.clone(),
            descriptions: Arc::new(AtomicUsize::new(0)),
            payload_pointer: None,
            max_output_bytes: 4,
        })
        .unwrap();
    let tightened = plan(vec![asset(1, b"oversized")], &[1], &tightened_registry).unwrap();
    assert_ne!(original.specs()[&JobId(1)], tightened.specs()[&JobId(1)]);
    assert!(tightened
        .execute(
            &store,
            &mut ddc,
            BuildRevision(2),
            1,
            BuildResources::UNLIMITED,
            false,
        )
        .is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(store.current_revision().unwrap(), Some(BuildRevision(1)));
    assert_eq!(
        store
            .cas()
            .read(first.published.manifest.entries[&AssetId(1)].content_hash)
            .unwrap(),
        b"oversized"
    );
}

#[test]
fn oversized_cached_root_and_dependency_reject_before_cooking_or_publication() {
    let (registry, calls, _) = registry();
    let mut parent = asset(1, b"parent");
    parent.dependencies = vec![dependency(2, Kind::Hard)];
    let frozen = plan(vec![parent, asset(2, b"child")], &[1], &registry).unwrap();
    let root = tempfile::tempdir().unwrap();
    let store = RevisionStore::open(root.path()).unwrap();
    let mut ddc = DerivedDataCache::open(root.path().join("ddc")).unwrap();
    let baseline = frozen
        .execute(
            &store,
            &mut ddc,
            BuildRevision(0),
            2,
            BuildResources::UNLIMITED,
            false,
        )
        .unwrap();
    let oversized = vec![0; 4097];
    let oversized_hash = store.cas().put(&oversized).unwrap();
    frozen
        .package_manifest(
            &baseline.published.manifest,
            store.cas(),
            "data",
            PackageOptions::default(),
        )
        .unwrap();
    for id in [AssetId(1), AssetId(2)] {
        let original = baseline.published.manifest.entries[&id].clone();
        let mut poisoned = original.clone();
        poisoned.byte_len = oversized.len() as u64;
        poisoned.content_hash = oversized_hash;
        ddc.invalidate(original.key).unwrap();
        ddc.put_descriptor(poisoned).unwrap();
        assert!(matches!(
            frozen.execute(
                &store,
                &mut ddc,
                BuildRevision(1),
                2,
                BuildResources::UNLIMITED,
                false,
            ),
            Err(EditableBuildError::Execution(
                BuildExecutionError::InvalidOutput(job, _)
            )) if job == JobId(id.0)
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(store.current_revision().unwrap(), Some(BuildRevision(0)));
        assert_eq!(
            store.manifest(BuildRevision(0)).unwrap(),
            baseline.published.manifest
        );
        ddc.invalidate(original.key).unwrap();
        ddc.put_descriptor(original).unwrap();
    }
    // Only this plan's fully derived keys are admitted. An unrelated large
    // cache result must not prevent a valid warm execution.
    let other = plan(vec![asset(3, b"unrelated")], &[3], &registry).unwrap();
    let mut unrelated = baseline.published.manifest.entries[&AssetId(2)].clone();
    unrelated.key = other.specs()[&JobId(3)].artifact_key(1, &[]).unwrap();
    unrelated.byte_len = oversized.len() as u64;
    unrelated.content_hash = oversized_hash;
    ddc.put_descriptor(unrelated).unwrap();
    let warm = frozen
        .execute(
            &store,
            &mut ddc,
            BuildRevision(1),
            2,
            BuildResources::UNLIMITED,
            false,
        )
        .unwrap();
    assert_eq!(warm.cache_hits, BTreeSet::from([JobId(1), JobId(2)]));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn determinism_failure_and_panicking_cooker_preserve_warm_revision_and_cache() {
    struct FaultyCooker {
        calls: Arc<AtomicUsize>,
        panic: bool,
    }
    impl EditableCooker for FaultyCooker {
        fn descriptor(&self) -> EditableCookerDescriptor {
            Cooker {
                calls: self.calls.clone(),
                descriptions: Arc::new(AtomicUsize::new(0)),
                payload_pointer: None,
                max_output_bytes: 4096,
            }
            .descriptor()
        }

        fn cook(
            &self,
            _: &EditableCookInput,
            _: &[EditableCookDependency<'_>],
        ) -> Result<Vec<u8>, ProcessorError> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            assert!(!self.panic, "deliberate cooker panic");
            Ok(vec![call as u8])
        }
    }
    let (registry, _, _) = registry();
    let root = tempfile::tempdir().unwrap();
    let store = RevisionStore::open(root.path()).unwrap();
    let mut ddc = DerivedDataCache::open(root.path().join("ddc")).unwrap();
    let baseline = plan(vec![asset(1, b"good")], &[1], &registry)
        .unwrap()
        .execute(
            &store,
            &mut ddc,
            BuildRevision(0),
            1,
            BuildResources::UNLIMITED,
            false,
        )
        .unwrap();
    let descriptor = &baseline.published.manifest.entries[&AssetId(1)];
    for panic in [false, true] {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut registry = EditableCookerRegistry::new();
        registry
            .register(FaultyCooker {
                calls: calls.clone(),
                panic,
            })
            .unwrap();
        // Keep the same semantic key as the warm result. Verification must
        // actually execute the cooker instead of accepting that cache hit.
        let frozen = plan(vec![asset(1, b"good")], &[1], &registry).unwrap();
        let error = frozen
            .execute(
                &store,
                &mut ddc,
                BuildRevision(1),
                1,
                BuildResources::UNLIMITED,
                true,
            )
            .unwrap_err();
        if panic {
            assert!(matches!(
                error,
                EditableBuildError::Execution(BuildExecutionError::Graph(
                    BuildError::ProcessorFailed { job: JobId(1), .. }
                ))
            ));
            assert_eq!(calls.load(Ordering::SeqCst), 1);
        } else {
            assert!(matches!(
                error,
                EditableBuildError::Execution(BuildExecutionError::NonDeterministic(_))
            ));
            assert_eq!(calls.load(Ordering::SeqCst), 2);
        }
        assert_eq!(store.current_revision().unwrap(), Some(BuildRevision(0)));
        assert_eq!(ddc.get(descriptor.key).as_ref(), Some(descriptor));
        assert_eq!(store.cas().read(descriptor.content_hash).unwrap(), b"good");
    }
}

#[test]
fn large_fingerprints_have_fixed_per_node_storage_and_preserve_input_identity() {
    let (registry, _, _) = registry();
    let mut options = EditableBuildOptions {
        toolchain_fingerprint: vec![17; 1024 * 1024],
        environment_fingerprint: vec![31; 1024 * 1024],
        ..Default::default()
    };
    let build = |options: &EditableBuildOptions| {
        EditableBuildPlan::from_snapshot(
            snapshot((1..=128).map(|id| asset(id, b"small")).collect()),
            &(1..=128).map(AssetId).collect::<Vec<_>>(),
            &registry,
            options,
        )
        .unwrap()
    };
    let original = build(&options);
    assert_eq!(
        original
            .specs()
            .values()
            .map(|spec| spec.toolchain_fingerprint.len() + spec.environment_fingerprint.len())
            .sum::<usize>(),
        128 * 64
    );
    let original_key = original.specs()[&JobId(1)].artifact_key(1, &[]).unwrap();
    options.toolchain_fingerprint[1024 * 1024 - 1] ^= 1;
    assert_ne!(
        build(&options).specs()[&JobId(1)]
            .artifact_key(1, &[])
            .unwrap(),
        original_key
    );
    options.toolchain_fingerprint[1024 * 1024 - 1] ^= 1;
    options.environment_fingerprint[1024 * 1024 - 1] ^= 1;
    assert_ne!(
        build(&options).specs()[&JobId(1)]
            .artifact_key(1, &[])
            .unwrap(),
        original_key
    );
    options.environment_fingerprint.push(0);
    assert!(matches!(
        EditableBuildPlan::from_snapshot(
            snapshot(vec![asset(1, b"small")]),
            &[AssetId(1)],
            &registry,
            &options,
        ),
        Err(EditableBuildError::Limit("fingerprint bytes"))
    ));
}

#[test]
fn plan_shares_immutable_payload_across_cooks_and_freezes_the_descriptor_once() {
    let value = asset(1, b"unique allocation");
    let pointer = Arc::new(AtomicUsize::new(0));
    let calls = Arc::new(AtomicUsize::new(0));
    let descriptions = Arc::new(AtomicUsize::new(0));
    let mut registry = EditableCookerRegistry::new();
    registry
        .register(Cooker {
            calls: calls.clone(),
            descriptions: descriptions.clone(),
            payload_pointer: Some(Arc::clone(&pointer)),
            max_output_bytes: 4096,
        })
        .unwrap();
    let plan = plan(vec![value], &[1], &registry).unwrap();
    let root = tempfile::tempdir().unwrap();
    let store = RevisionStore::open(root.path()).unwrap();
    let mut ddc = DerivedDataCache::open(root.path().join("ddc")).unwrap();
    plan.execute(
        &store,
        &mut ddc,
        BuildRevision(0),
        1,
        BuildResources::UNLIMITED,
        true,
    )
    .unwrap();
    assert_eq!(descriptions.load(Ordering::SeqCst), 1);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_ne!(pointer.load(Ordering::SeqCst), 0);
}

#[test]
fn packaging_rejects_an_execution_from_another_frozen_plan() {
    let (registry, _, _) = registry();
    let first = plan(vec![asset(1, b"one")], &[1], &registry).unwrap();
    let second = plan(vec![asset(1, b"two")], &[1], &registry).unwrap();
    let root = tempfile::tempdir().unwrap();
    let store = RevisionStore::open(root.path()).unwrap();
    let mut ddc = DerivedDataCache::open(root.path().join("ddc")).unwrap();
    let execution = first
        .execute(
            &store,
            &mut ddc,
            BuildRevision(0),
            1,
            BuildResources::UNLIMITED,
            false,
        )
        .unwrap();
    assert!(matches!(
        second.package(&execution, store.cas(), "data", PackageOptions::default()),
        Err(EditableBuildError::ManifestMismatch)
    ));
    assert!(matches!(
        second.package_manifest(
            &execution.published.manifest,
            store.cas(),
            "data",
            PackageOptions::default()
        ),
        Err(EditableBuildError::ManifestMismatch)
    ));
    let mut oversized = execution.published.manifest.clone();
    oversized.entries.get_mut(&AssetId(1)).unwrap().byte_len = 4097;
    assert!(matches!(
        first.package_manifest(&oversized, store.cas(), "data", PackageOptions::default()),
        Err(EditableBuildError::ManifestMismatch)
    ));
    let descriptor = &execution.published.manifest.entries[&AssetId(1)];
    let mut variants = Vec::new();
    let mut changed = descriptor.clone();
    changed.kind += 1;
    variants.push(changed);
    let mut changed = descriptor.clone();
    changed.schema += 1;
    variants.push(changed);
    let mut changed = descriptor.clone();
    changed.alignment *= 2;
    variants.push(changed);
    let mut changed = descriptor.clone();
    changed.dependencies.push(descriptor.key);
    variants.push(changed);
    let mut changed = descriptor.clone();
    changed.key.0[0] ^= 1;
    variants.push(changed);
    for changed in variants {
        let mut manifest = execution.published.manifest.clone();
        manifest.entries.insert(AssetId(1), changed);
        assert!(matches!(
            first.package_manifest(&manifest, store.cas(), "data", PackageOptions::default()),
            Err(EditableBuildError::ManifestMismatch)
        ));
    }
}

#[test]
fn invalid_identity_snapshot_budget_and_empty_roots_are_rejected() {
    let (registry, _, _) = registry();
    assert!(matches!(
        plan(vec![], &[], &registry),
        Err(EditableBuildError::EmptyRoots)
    ));
    assert!(matches!(
        EditableBuildPlan::from_snapshot(
            BTreeMap::from([(AssetId(2), asset(1, b"one"))]),
            &[AssetId(2)],
            &registry,
            &Default::default()
        ),
        Err(EditableBuildError::InvalidIdentity(AssetId(2)))
    ));
    assert!(matches!(
        EditableBuildPlan::from_snapshot(
            snapshot(vec![asset(1, b"one")]),
            &[AssetId(1)],
            &registry,
            &EditableBuildOptions {
                max_snapshot_bytes: 1,
                ..Default::default()
            }
        ),
        Err(EditableBuildError::Limit(_))
    ));
    let mut unknown = asset(1, b"one");
    unknown.kind = 999;
    assert!(matches!(
        plan(vec![unknown], &[1], &registry),
        Err(EditableBuildError::MissingCooker { .. })
    ));
}
