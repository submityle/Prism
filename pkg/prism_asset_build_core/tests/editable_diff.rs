//! Semantic diff impact checked against real editable plans, execution and DDC.

use prism_asset_build_core::{
    diff_editable_assets, ArtifactKey, AssetId, BuildExecution, BuildResources, BuildRevision,
    DerivedDataCache, EditableAsset, EditableBuildOptions, EditableBuildPlan,
    EditableCookDependency, EditableCookInput, EditableCooker, EditableCookerDescriptor,
    EditableCookerRegistry, EditableDependency, EditableDependencyKind as Kind, EditableDiffImpact,
    EditableSubAsset, EditableSubAssetReference, EditableWorkspace, ProcessorError,
    ProcessorIdentity, RevisionStore, SourceRecord,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tempfile::{tempdir, TempDir};

fn asset(id: u128, payload: &[u8]) -> EditableAsset {
    EditableAsset::new(
        AssetId(id),
        42,
        0,
        b"settings".to_vec(),
        SourceRecord::embedded("source.bin", "test/raw", b"source".to_vec()).unwrap(),
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

fn reference(id: u128, kind: Kind) -> EditableSubAssetReference {
    EditableSubAssetReference {
        asset: AssetId(id),
        subasset: 4,
        kind,
    }
}

struct CountingCooker {
    kind: u128,
    calls: Arc<AtomicUsize>,
}

impl EditableCooker for CountingCooker {
    fn descriptor(&self) -> EditableCookerDescriptor {
        EditableCookerDescriptor {
            authoring_kind: self.kind,
            identity: ProcessorIdentity::new(1000 + self.kind, 1),
            output_kind: 100,
            output_schema: 1,
            alignment: 4,
            max_output_bytes: 4096,
            resources: BuildResources::default(),
        }
    }

    fn cook(
        &self,
        input: &EditableCookInput,
        dependencies: &[EditableCookDependency<'_>],
    ) -> Result<Vec<u8>, ProcessorError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        assert!(input
            .dependencies
            .iter()
            .all(|d| d.kind != Kind::EditorOnly));
        assert!(input
            .subasset_references
            .iter()
            .all(|r| r.kind != Kind::EditorOnly));
        // A cooker need not use every visible field. Those fields must still
        // participate in artifact identity even if output bytes happen to match.
        let mut bytes = input.authoring_payload.clone();
        for dependency in dependencies {
            bytes.extend_from_slice(dependency.bytes);
        }
        Ok(bytes)
    }
}

struct Harness {
    directory: TempDir,
    registry: EditableCookerRegistry,
    calls: Arc<AtomicUsize>,
    next_revision: u64,
}

impl Harness {
    fn new() -> Self {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut registry = EditableCookerRegistry::new();
        for kind in [42, 43] {
            registry
                .register(CountingCooker {
                    kind,
                    calls: Arc::clone(&calls),
                })
                .unwrap();
        }
        Self {
            directory: tempdir().unwrap(),
            registry,
            calls,
            next_revision: 0,
        }
    }

    fn plan(&self, assets: Vec<EditableAsset>, roots: &[AssetId]) -> EditableBuildPlan {
        EditableBuildPlan::from_snapshot(
            assets
                .into_iter()
                .map(|a| (a.asset, a))
                .collect::<BTreeMap<_, _>>(),
            roots,
            &self.registry,
            &EditableBuildOptions::default(),
        )
        .unwrap()
    }

    fn single_plan(&self, asset: &EditableAsset) -> EditableBuildPlan {
        self.plan(vec![asset.clone()], &[asset.asset])
    }

    fn run(&mut self, plan: &EditableBuildPlan, case: &str, phase: &str) -> BuildExecution {
        // Reopen both handles for every build, so a hit proves durable DDC/CAS
        // reuse rather than an in-memory result retained by this test.
        let store = RevisionStore::open(self.directory.path().join("build")).unwrap();
        let mut ddc = DerivedDataCache::open(self.directory.path().join("ddc")).unwrap();
        let execution = plan
            .execute(
                &store,
                &mut ddc,
                BuildRevision(self.next_revision),
                1,
                BuildResources::UNLIMITED,
                false,
            )
            .unwrap_or_else(|error| {
                panic!(
                    "{case} [{phase}], revision {}, jobs {:?}: {error:?}",
                    self.next_revision,
                    plan.specs().keys().collect::<Vec<_>>(),
                )
            });
        self.next_revision += 1;
        for (job, key) in &execution.artifact_keys {
            let descriptor = &execution.published.manifest.entries[&AssetId(job.0)];
            assert_eq!(*key, descriptor.key);
            assert_eq!(ddc.get(*key).as_ref(), Some(descriptor));
            assert_eq!(
                store.cas().read(descriptor.content_hash).unwrap().len() as u64,
                descriptor.byte_len
            );
        }
        execution
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

fn local_key(plan: &EditableBuildPlan, asset: AssetId) -> ArtifactKey {
    // The current editable pipeline uses protocol 1. No dependency keys are
    // supplied here: this is the saved local input contract, not a closure key.
    plan.specs()[&prism_asset_build_core::JobId(asset.0)]
        .artifact_key(1, &[])
        .unwrap()
}

fn hits(assets: &[u128]) -> BTreeSet<prism_asset_build_core::JobId> {
    assets
        .iter()
        .map(|id| prism_asset_build_core::JobId(*id))
        .collect()
}

type Mutation = fn(&mut EditableAsset);

#[test]
fn metadata_only_diff_preserves_keys_and_reuses_durable_cache() {
    let mut harness = Harness::new();
    let before = asset(1, b"payload");
    let baseline_plan = harness.single_plan(&before);
    let baseline_key = local_key(&baseline_plan, before.asset);
    let baseline = harness.run(&baseline_plan, "metadata only", "baseline");
    assert!(baseline.cache_hits.is_empty());
    assert_eq!(harness.call_count(), 1);
    let cases: &[(&str, Mutation)] = &[
        ("source", |a| {
            a.source =
                SourceRecord::embedded("moved.bin", "other/format", b"different source".to_vec())
                    .unwrap()
        }),
        ("source storage", |a| a.source.embedded_source = None),
        ("empty preview", |a| a.preview = Some(Vec::new())),
        ("preview", |a| a.preview = Some(b"preview".to_vec())),
        ("editor dependency", |a| {
            a.dependencies.push(dependency(99, Kind::EditorOnly))
        }),
        ("editor reference", |a| {
            a.subasset_references.push(reference(99, Kind::EditorOnly))
        }),
    ];
    for (label, mutate) in cases {
        let mut after = before.clone();
        mutate(&mut after);
        let diff = diff_editable_assets(&before, &after).unwrap();
        assert!(!diff.is_empty(), "{label}");
        assert_eq!(
            diff.impact(),
            EditableDiffImpact {
                cook: false,
                editor_import: true
            },
            "{label}"
        );
        let plan = harness.single_plan(&after);
        assert_eq!(local_key(&plan, after.asset), baseline_key, "{label}");
        let warm = harness.run(&plan, label, "metadata warm");
        assert_eq!(warm.artifact_keys, baseline.artifact_keys, "{label}");
        assert_eq!(warm.cache_hits, hits(&[1]), "{label}");
        assert_eq!(harness.call_count(), 1, "{label}");
        assert_eq!(
            warm.published.manifest.entries, baseline.published.manifest.entries,
            "{label}"
        );
    }
}

#[test]
fn cooker_visible_diff_changes_keys_and_executes_before_becoming_warm() {
    let mut harness = Harness::new();
    let before = asset(1, b"payload");
    let baseline_plan = harness.single_plan(&before);
    let baseline_key = local_key(&baseline_plan, before.asset);
    let baseline = harness.run(&baseline_plan, "cooker visible", "baseline");
    let cases: &[(&str, Mutation)] = &[
        // Replacing the root identity also removes the previous JobId from the
        // current graph. Keep this on the same store to cover graph replacement.
        ("asset identity", |a| a.asset = AssetId(7)),
        ("kind", |a| a.kind = 43),
        ("flags", |a| a.flags = 1),
        ("settings", |a| a.canonical_settings.push(1)),
        ("same length payload", |a| a.authoring_payload[0] = b'P'),
        ("subasset name", |a| a.subassets[0].name = "renamed".into()),
        ("subasset kind", |a| a.subassets[0].kind = 43),
        ("subasset stable ID", |a| a.subassets[0].id = 8),
        ("subasset addition", |a| {
            a.subassets.push(EditableSubAsset {
                id: 8,
                name: "extra".into(),
                kind: 42,
            })
        }),
        ("subasset removal", |a| a.subassets.clear()),
        ("soft dependency", |a| {
            a.dependencies.push(dependency(99, Kind::Soft))
        }),
        ("soft reference", |a| {
            a.subasset_references.push(reference(99, Kind::Soft))
        }),
    ];
    for (label, mutate) in cases {
        let mut after = before.clone();
        mutate(&mut after);
        let diff = diff_editable_assets(&before, &after).unwrap();
        assert_eq!(
            diff.impact(),
            EditableDiffImpact {
                cook: true,
                editor_import: false
            },
            "{label}"
        );
        let plan = harness.single_plan(&after);
        let expected_key = local_key(&plan, after.asset);
        assert_ne!(expected_key, baseline_key, "{label}");
        let previous_calls = harness.call_count();
        let cold = harness.run(&plan, label, "cold");
        assert!(cold.cache_hits.is_empty(), "{label}");
        assert_eq!(harness.call_count(), previous_calls + 1, "{label}");
        assert_eq!(
            cold.artifact_keys[&prism_asset_build_core::JobId(after.asset.0)],
            expected_key,
            "{label}"
        );
        assert_ne!(
            cold.published.manifest.entries[&after.asset].key,
            baseline.published.manifest.entries[&before.asset].key,
            "{label}"
        );
        assert_eq!(
            cold.published.manifest.entries[&after.asset].content_hash
                == baseline.published.manifest.entries[&before.asset].content_hash,
            after.authoring_payload == before.authoring_payload,
            "{label}: cache identity must track inputs even when cooked bytes match",
        );
        let warm_plan = harness.single_plan(&after);
        let warm = harness.run(&warm_plan, label, "warm");
        assert_eq!(warm.artifact_keys, cold.artifact_keys, "{label}");
        assert_eq!(warm.cache_hits, hits(&[after.asset.0]), "{label}");
        assert_eq!(harness.call_count(), previous_calls + 1, "{label}");
    }
}

#[test]
fn saved_reopened_assets_keep_diff_and_build_cache_semantics() {
    let mut harness = Harness::new();
    let workspace_path = harness.directory.path().join("workspace");
    std::fs::create_dir(&workspace_path).unwrap();
    let path = workspace_path.join("root.prism");
    let before = asset(1, b"payload");
    before.write_to_path(&path).unwrap();
    asset(2, b"editor target")
        .write_to_path(workspace_path.join("editor.prism"))
        .unwrap();
    let reopened_before = EditableAsset::read_from_path(&path).unwrap();
    assert_eq!(before, reopened_before);
    let baseline_plan = EditableBuildPlan::from_workspace(
        &EditableWorkspace::open(&workspace_path).unwrap(),
        &[AssetId(1)],
        &harness.registry,
        &EditableBuildOptions::default(),
    )
    .unwrap();
    let baseline = harness.run(&baseline_plan, "save/reopen", "baseline");
    let mut metadata_edit = reopened_before.clone();
    metadata_edit.source =
        SourceRecord::embedded("moved.bin", "new/format", b"new source".to_vec()).unwrap();
    metadata_edit.preview = Some(b"preview".to_vec());
    metadata_edit.dependencies = vec![dependency(2, Kind::EditorOnly)];
    metadata_edit.subasset_references = vec![reference(2, Kind::EditorOnly)];
    metadata_edit.write_to_path(&path).unwrap();
    let reopened_metadata = EditableAsset::read_from_path(&path).unwrap();
    assert_eq!(metadata_edit, reopened_metadata);
    let diff = diff_editable_assets(&reopened_before, &reopened_metadata).unwrap();
    assert_eq!(
        diff.impact(),
        EditableDiffImpact {
            cook: false,
            editor_import: true
        }
    );
    let warm_plan = EditableBuildPlan::from_workspace(
        &EditableWorkspace::open(&workspace_path).unwrap(),
        &[AssetId(1)],
        &harness.registry,
        &EditableBuildOptions::default(),
    )
    .unwrap();
    assert_eq!(
        local_key(&warm_plan, AssetId(1)),
        local_key(&baseline_plan, AssetId(1))
    );
    let warm = harness.run(&warm_plan, "save/reopen", "metadata warm");
    assert_eq!(warm.artifact_keys, baseline.artifact_keys);
    assert_eq!(warm.cache_hits, hits(&[1]));
    assert_eq!(harness.call_count(), 1);
    // The editor-only target was resolved by the workspace but never cooked.
    assert_eq!(warm.published.manifest.entries.len(), 1);
    let mut semantic_edit = reopened_metadata.clone();
    semantic_edit.subassets[0].name = "renamed after reopen".into();
    semantic_edit.write_to_path(&path).unwrap();
    let reopened_semantic = EditableAsset::read_from_path(&path).unwrap();
    assert!(diff_editable_assets(&reopened_metadata, &reopened_semantic)
        .unwrap()
        .affects_cook());
    let changed_plan = EditableBuildPlan::from_workspace(
        &EditableWorkspace::open(&workspace_path).unwrap(),
        &[AssetId(1)],
        &harness.registry,
        &EditableBuildOptions::default(),
    )
    .unwrap();
    let changed = harness.run(&changed_plan, "save/reopen", "semantic cold");
    assert_ne!(changed.artifact_keys, baseline.artifact_keys);
    assert!(changed.cache_hits.is_empty());
    assert_eq!(harness.call_count(), 2);
    let warm_again = harness.run(&changed_plan, "save/reopen", "semantic warm");
    assert_eq!(warm_again.cache_hits, hits(&[1]));
    assert_eq!(harness.call_count(), 2);
}

#[test]
fn external_artifact_changes_are_outside_local_diff_and_follow_edge_strength() {
    for kind in [Kind::Hard, Kind::Soft] {
        let label = format!("external artifact {kind:?}");
        let mut harness = Harness::new();
        let supplier = asset(1, b"old");
        let mut consumer = asset(2, b"consumer");
        consumer.dependencies = vec![dependency(1, kind)];
        let unchanged_consumer = consumer.clone();
        let unrelated = asset(3, b"unrelated");
        let roots = [AssetId(1), AssetId(2), AssetId(3)];
        // Explicit roots build the supplier even for the soft-edge case.
        let baseline_plan = harness.plan(
            vec![supplier.clone(), consumer.clone(), unrelated.clone()],
            &roots,
        );
        let baseline = harness.run(&baseline_plan, &label, "baseline");
        assert!(baseline.cache_hits.is_empty());
        assert_eq!(harness.call_count(), 3);
        let mut changed_supplier = supplier.clone();
        changed_supplier.authoring_payload = b"new".to_vec();
        assert!(diff_editable_assets(&supplier, &changed_supplier)
            .unwrap()
            .affects_cook());
        let local_diff = diff_editable_assets(&consumer, &unchanged_consumer).unwrap();
        assert!(local_diff.is_empty());
        assert!(!local_diff.affects_cook());
        let changed_plan = harness.plan(
            vec![changed_supplier, unchanged_consumer, unrelated],
            &roots,
        );
        // The consumer's own saved settings and local key are unchanged. Only
        // BuildExecutor's hard input keys can invalidate its final artifact.
        assert_eq!(
            local_key(&baseline_plan, AssetId(2)),
            local_key(&changed_plan, AssetId(2))
        );
        let changed = harness.run(&changed_plan, &label, "dependency changed");
        let job = prism_asset_build_core::JobId;
        assert_ne!(
            changed.artifact_keys[&job(1)],
            baseline.artifact_keys[&job(1)]
        );
        assert_eq!(
            changed.artifact_keys[&job(3)],
            baseline.artifact_keys[&job(3)]
        );
        let (expected_hits, expected_calls, expected_bytes): (_, _, &[u8]) = if kind == Kind::Hard {
            assert_ne!(
                changed.artifact_keys[&job(2)],
                baseline.artifact_keys[&job(2)]
            );
            (hits(&[3]), 5, b"consumernew")
        } else {
            assert_eq!(
                changed.artifact_keys[&job(2)],
                baseline.artifact_keys[&job(2)]
            );
            (hits(&[2, 3]), 4, b"consumer")
        };
        assert_eq!(changed.cache_hits, expected_hits);
        assert_eq!(harness.call_count(), expected_calls);
        let store = RevisionStore::open(harness.directory.path().join("build")).unwrap();
        let descriptor = &changed.published.manifest.entries[&AssetId(2)];
        assert_eq!(
            store.cas().read(descriptor.content_hash).unwrap(),
            expected_bytes
        );
        let warm = harness.run(&changed_plan, &label, "warm");
        assert_eq!(warm.artifact_keys, changed.artifact_keys);
        assert_eq!(warm.cache_hits, hits(&[1, 2, 3]));
        assert_eq!(harness.call_count(), expected_calls);
    }
}
