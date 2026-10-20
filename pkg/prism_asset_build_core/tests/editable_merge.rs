//! Three-way merge through real workspace publication and persistent Cook/DDC/CAS.

use prism_asset_build_core::{
    plan_editable_merge, AssetId, BuildExecution, BuildResources, BuildRevision, ContentHash,
    DerivedDataCache, EditableAsset, EditableAssetError, EditableBuildOptions, EditableBuildPlan,
    EditableCookDependency, EditableCookInput, EditableCooker, EditableCookerDescriptor,
    EditableCookerRegistry, EditableDependencyKind, EditableMergeError, EditableMergePath,
    EditableMergeSide, EditableMergeValue, EditableSubAsset, EditableSubAssetReference,
    EditableWorkspace, EditableWorkspaceEdit, JobId, ProcessorError, ProcessorIdentity,
    RevisionStore, SourceRecord, MAX_EDITABLE_ASSET_BYTES, MAX_EDITABLE_WORKSPACE_ENCODED_BYTES,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tempfile::{tempdir, TempDir};

const MERGE_LIMIT: u64 = MAX_EDITABLE_ASSET_BYTES as u64;

fn asset(id: u128) -> EditableAsset {
    EditableAsset::new(
        AssetId(id),
        42,
        0,
        b"base settings".to_vec(),
        SourceRecord::embedded("source.bin", "test/raw", b"base source".to_vec()).unwrap(),
        vec![EditableSubAsset {
            id: 4,
            name: "part".into(),
            kind: 42,
        }],
        Vec::new(),
        Vec::new(),
        b"base payload".to_vec(),
        None,
    )
    .unwrap()
}

fn edit(previous: &EditableAsset, replacement: EditableAsset) -> EditableWorkspaceEdit {
    EditableWorkspaceEdit {
        expected_hash: ContentHash::digest(&previous.encode().unwrap()),
        replacement,
    }
}

// Encode both independent semantic fields in the actual artifact so a dropped
// side cannot be hidden by checking only merge metadata or artifact keys.
fn cooked_bytes(settings: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut bytes = (settings.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(settings);
    bytes.extend_from_slice(payload);
    bytes
}

struct CountingCooker(Arc<AtomicUsize>);

impl EditableCooker for CountingCooker {
    fn descriptor(&self) -> EditableCookerDescriptor {
        EditableCookerDescriptor {
            authoring_kind: 42,
            identity: ProcessorIdentity::new(1042, 1),
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
        self.0.fetch_add(1, Ordering::SeqCst);
        let mut bytes = cooked_bytes(&input.canonical_settings, &input.authoring_payload);
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
    fn new(assets: &[EditableAsset]) -> Self {
        let directory = tempdir().unwrap();
        let workspace_path = directory.path().join("workspace");
        fs::create_dir(&workspace_path).unwrap();
        for asset in assets {
            asset
                .write_to_path(workspace_path.join(format!("{}.prism", asset.asset.0)))
                .unwrap();
        }
        let calls = Arc::new(AtomicUsize::new(0));
        let mut registry = EditableCookerRegistry::new();
        registry
            .register(CountingCooker(Arc::clone(&calls)))
            .unwrap();
        Self {
            directory,
            registry,
            calls,
            next_revision: 0,
        }
    }

    fn workspace_path(&self) -> PathBuf {
        self.directory.path().join("workspace")
    }

    fn workspace(&self) -> EditableWorkspace {
        EditableWorkspace::open(self.workspace_path()).unwrap()
    }

    fn package_bytes(&self, asset: AssetId) -> Vec<u8> {
        fs::read(self.workspace_path().join(format!("{}.prism", asset.0))).unwrap()
    }

    fn saved(&self, asset: AssetId) -> EditableAsset {
        self.workspace()
            .snapshot(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)
            .unwrap()
            .remove(&asset)
            .unwrap()
    }

    fn save(&self, previous: &EditableAsset, replacement: EditableAsset) {
        let expected_hash = ContentHash::digest(&replacement.encode().unwrap());
        let receipt = self
            .workspace()
            .save(
                vec![edit(previous, replacement)],
                MAX_EDITABLE_WORKSPACE_ENCODED_BYTES,
            )
            .unwrap();
        assert_eq!(receipt.updated_assets, [previous.asset]);
        assert_eq!(receipt.content_hashes[&previous.asset], expected_hash);
    }

    fn cook(&mut self, asset: AssetId) -> (BuildExecution, Vec<u8>) {
        let plan = EditableBuildPlan::from_workspace(
            &self.workspace(),
            &[asset],
            &self.registry,
            &EditableBuildOptions::default(),
        )
        .unwrap();
        // Both handles die at the end of each call. Warm builds must reload the
        // descriptor index and CAS from disk, including after a workspace save.
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
            .unwrap();
        self.next_revision += 1;
        let descriptor = &execution.published.manifest.entries[&asset];
        assert_eq!(execution.artifact_keys[&JobId(asset.0)], descriptor.key);
        assert_eq!(ddc.get(descriptor.key).as_ref(), Some(descriptor));
        let bytes = store.cas().read(descriptor.content_hash).unwrap();
        assert_eq!(bytes.len() as u64, descriptor.byte_len);
        assert_eq!(ContentHash::digest(&bytes), descriptor.content_hash);
        (execution, bytes)
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    fn assert_no_transaction(&self) {
        let path = self.workspace_path().join(".prism-editable-transactions");
        if path.exists() {
            assert_eq!(fs::read_dir(path).unwrap().count(), 0);
        }
    }
}

#[test]
fn independent_edits_survive_save_reopen_and_actual_cook() {
    let base = asset(1);
    let mut local = base.clone();
    local.canonical_settings = b"local settings".to_vec();
    let mut incoming = base.clone();
    incoming.authoring_payload = b"incoming payload".to_vec();
    let mut harness = Harness::new(std::slice::from_ref(&local));
    let (baseline, _) = harness.cook(base.asset);
    let plan = plan_editable_merge(&base, &local, &incoming).unwrap();
    assert!(plan.conflicts().is_empty());
    let merged = plan.resolve(&BTreeMap::new(), MERGE_LIMIT).unwrap();
    assert_eq!(merged.canonical_settings, local.canonical_settings);
    assert_eq!(merged.authoring_payload, incoming.authoring_payload);
    harness.save(&local, merged.clone());
    assert_eq!(harness.saved(base.asset), merged);
    let (cold, bytes) = harness.cook(base.asset);
    assert_eq!(
        bytes,
        cooked_bytes(&local.canonical_settings, &incoming.authoring_payload)
    );
    assert!(cold.cache_hits.is_empty());
    assert_ne!(cold.artifact_keys, baseline.artifact_keys);
    assert_eq!(harness.call_count(), 2);
    let (warm, warm_bytes) = harness.cook(base.asset);
    assert_eq!(warm.cache_hits, BTreeSet::from([JobId(base.asset.0)]));
    assert_eq!(warm.artifact_keys, cold.artifact_keys);
    assert_eq!(warm_bytes, bytes);
    assert_eq!(harness.call_count(), 2);
}

#[test]
fn source_and_preview_merge_preserves_persistent_ddc_hit() {
    let base = asset(1);
    let mut harness = Harness::new(std::slice::from_ref(&base));
    let (baseline, baseline_bytes) = harness.cook(base.asset);
    assert!(baseline.cache_hits.is_empty());
    let mut local = base.clone();
    local.source =
        SourceRecord::embedded("moved.bin", "other/raw", b"new source".to_vec()).unwrap();
    let mut incoming = base.clone();
    incoming.preview = Some(b"incoming preview".to_vec());
    let plan = plan_editable_merge(&base, &local, &incoming).unwrap();
    assert!(plan.conflicts().is_empty());
    let merged = plan.resolve(&BTreeMap::new(), MERGE_LIMIT).unwrap();
    assert_eq!(merged.source, local.source);
    assert_eq!(merged.preview, incoming.preview);
    harness.save(&base, merged.clone());
    assert_ne!(harness.package_bytes(base.asset), base.encode().unwrap());
    assert_eq!(harness.saved(base.asset), merged);
    let (warm, bytes) = harness.cook(base.asset);
    assert_eq!(warm.cache_hits, BTreeSet::from([JobId(base.asset.0)]));
    assert_eq!(warm.artifact_keys, baseline.artifact_keys);
    assert_eq!(
        warm.published.manifest.entries,
        baseline.published.manifest.entries
    );
    assert_eq!(bytes, baseline_bytes);
    assert_eq!(harness.call_count(), 1);
}

#[test]
fn payload_conflict_requires_explicit_choice_and_cooks_each_selected_side() {
    for side in [
        EditableMergeSide::Base,
        EditableMergeSide::Local,
        EditableMergeSide::Incoming,
    ] {
        let base = asset(1);
        let mut local = base.clone();
        local.authoring_payload = b"local payload".to_vec();
        let mut incoming = base.clone();
        incoming.authoring_payload = b"incoming payload".to_vec();
        let mut harness = Harness::new(std::slice::from_ref(&local));
        let before = harness.package_bytes(base.asset);
        let plan = plan_editable_merge(&base, &local, &incoming).unwrap();
        assert_eq!(plan.conflicts().len(), 1);
        assert_eq!(
            plan.conflicts()[0].path,
            EditableMergePath::AuthoringPayload
        );
        let conflict = &plan.conflicts()[0];
        assert_eq!(
            conflict.base,
            EditableMergeValue::Bytes(&base.authoring_payload)
        );
        assert_eq!(
            conflict.local,
            EditableMergeValue::Bytes(&local.authoring_payload)
        );
        assert_eq!(
            conflict.incoming,
            EditableMergeValue::Bytes(&incoming.authoring_payload)
        );
        assert!(matches!(
            plan.resolve(&BTreeMap::new(), MERGE_LIMIT),
            Err(EditableMergeError::UnresolvedConflict(
                EditableMergePath::AuthoringPayload
            ))
        ));
        assert_eq!(harness.package_bytes(base.asset), before);
        harness.assert_no_transaction();
        let expected = match &side {
            EditableMergeSide::Base => &base,
            EditableMergeSide::Local => &local,
            EditableMergeSide::Incoming => &incoming,
        };
        let choices = BTreeMap::from([(EditableMergePath::AuthoringPayload, side)]);
        let merged = plan.resolve(&choices, MERGE_LIMIT).unwrap();
        assert_eq!(&merged, expected);
        // The Local choice is a valid no-op publication, with a receipt hash.
        let receipt = harness
            .workspace()
            .save(
                vec![edit(&local, merged.clone())],
                MAX_EDITABLE_WORKSPACE_ENCODED_BYTES,
            )
            .unwrap();
        assert_eq!(
            receipt.content_hashes[&base.asset],
            ContentHash::digest(&merged.encode().unwrap())
        );
        assert_eq!(harness.saved(base.asset), merged);
        let (execution, bytes) = harness.cook(base.asset);
        assert!(execution.cache_hits.is_empty());
        assert_eq!(
            bytes,
            cooked_bytes(&expected.canonical_settings, &expected.authoring_payload)
        );
        assert_eq!(harness.call_count(), 1);
    }
}

#[test]
fn partial_conflict_choices_cannot_publish_a_silent_fallback() {
    let base = asset(1);
    let mut local = base.clone();
    local.canonical_settings = b"local settings".to_vec();
    local.authoring_payload = b"local payload".to_vec();
    let mut incoming = base.clone();
    incoming.canonical_settings = b"incoming settings".to_vec();
    incoming.authoring_payload = b"incoming payload".to_vec();
    let mut harness = Harness::new(std::slice::from_ref(&local));
    let before = harness.package_bytes(base.asset);
    let plan = plan_editable_merge(&base, &local, &incoming).unwrap();
    assert_eq!(plan.conflicts().len(), 2);
    let mut choices = BTreeMap::from([(
        EditableMergePath::AuthoringPayload,
        EditableMergeSide::Local,
    )]);
    assert!(matches!(
        plan.resolve(&choices, MERGE_LIMIT),
        Err(EditableMergeError::UnresolvedConflict(
            EditableMergePath::Settings
        ))
    ));
    assert_eq!(harness.package_bytes(base.asset), before);
    harness.assert_no_transaction();
    choices.insert(EditableMergePath::Settings, EditableMergeSide::Incoming);
    let merged = plan.resolve(&choices, MERGE_LIMIT).unwrap();
    harness.save(&local, merged.clone());
    assert_eq!(harness.saved(base.asset), merged);
    let (_, bytes) = harness.cook(base.asset);
    assert_eq!(
        bytes,
        cooked_bytes(&incoming.canonical_settings, &local.authoring_payload)
    );
}

#[test]
fn merged_subasset_deletion_is_rejected_without_changing_either_package() {
    let base = asset(1);
    let mut consumer = asset(2);
    consumer
        .subasset_references
        .push(EditableSubAssetReference {
            asset: base.asset,
            subasset: 4,
            kind: EditableDependencyKind::Hard,
        });
    let harness = Harness::new(&[base.clone(), consumer.clone()]);
    let before_target = harness.package_bytes(base.asset);
    let before_consumer = harness.package_bytes(consumer.asset);
    let mut local = base.clone();
    local.subassets.clear();
    let mut incoming = base.clone();
    incoming.authoring_payload = b"incoming payload".to_vec();
    let plan = plan_editable_merge(&base, &local, &incoming).unwrap();
    assert!(plan.conflicts().is_empty());
    let merged = plan.resolve(&BTreeMap::new(), MERGE_LIMIT).unwrap();
    assert!(merged.subassets.is_empty());
    assert_eq!(merged.authoring_payload, incoming.authoring_payload);
    assert_eq!(
        harness.workspace().save(
            vec![edit(&base, merged)],
            MAX_EDITABLE_WORKSPACE_ENCODED_BYTES,
        ),
        Err(EditableAssetError::MissingReferencedSubAsset {
            asset: base.asset,
            subasset: 4,
        })
    );
    assert_eq!(harness.package_bytes(base.asset), before_target);
    assert_eq!(harness.package_bytes(consumer.asset), before_consumer);
    assert_eq!(harness.saved(base.asset), base);
    assert_eq!(harness.saved(consumer.asset), consumer);
    harness.assert_no_transaction();
}

#[test]
fn caller_output_budget_counts_the_encoded_package_and_leaves_workspace_unchanged() {
    let base = asset(1);
    let harness = Harness::new(std::slice::from_ref(&base));
    let before = harness.package_bytes(base.asset);
    let mut local = base.clone();
    local.canonical_settings = b"longer local settings".to_vec();
    let mut incoming = base.clone();
    incoming.authoring_payload = b"longer incoming payload".to_vec();
    let mut expected = local.clone();
    expected.authoring_payload = incoming.authoring_payload.clone();
    let actual = expected.encode().unwrap().len() as u64;
    let plan = plan_editable_merge(&base, &local, &incoming).unwrap();
    assert!(plan.conflicts().is_empty());
    for limit in [1, actual - 1] {
        assert!(matches!(
            plan.resolve(&BTreeMap::new(), limit),
            Err(EditableMergeError::OutputBudgetExceeded { actual: got, limit: cap })
                if got == actual && cap == limit
        ));
        assert_eq!(harness.package_bytes(base.asset), before);
        assert_eq!(harness.saved(base.asset), base);
        harness.assert_no_transaction();
    }
    assert_eq!(plan.resolve(&BTreeMap::new(), actual).unwrap(), expected);
    for requested in [0, MERGE_LIMIT + 1] {
        assert!(matches!(
            plan.resolve(&BTreeMap::new(), requested),
            Err(EditableMergeError::InvalidBudget { requested: got, limit })
                if got == requested && limit == MERGE_LIMIT
        ));
    }
    assert_eq!(harness.package_bytes(base.asset), before);
}

#[test]
fn merged_output_cannot_bypass_the_hard_limit_with_the_maximum_caller_budget() {
    let base = asset(1);
    let harness = Harness::new(std::slice::from_ref(&base));
    let before = harness.package_bytes(base.asset);
    let half = MAX_EDITABLE_ASSET_BYTES / 2;
    let mut local = base.clone();
    local.source = SourceRecord::embedded("source.bin", "test/raw", vec![1; half]).unwrap();
    let mut incoming = base.clone();
    incoming.authoring_payload = vec![2; half];
    // Each side is a valid package. Only their combination exceeds 64 MiB;
    // neither field exceeds its own 32 MiB maximum.
    assert!(local.encode().is_ok());
    assert!(incoming.encode().is_ok());
    let actual = before.len()
        - base.source.embedded_source.as_ref().unwrap().len()
        - base.authoring_payload.len()
        + 2 * half;
    assert!(actual > MAX_EDITABLE_ASSET_BYTES);
    let plan = plan_editable_merge(&base, &local, &incoming).unwrap();
    assert!(plan.conflicts().is_empty());
    match plan.resolve(&BTreeMap::new(), MERGE_LIMIT) {
        Err(EditableMergeError::OutputBudgetExceeded { actual: got, limit }) => {
            assert_eq!(got, actual as u64);
            assert_eq!(limit, MAX_EDITABLE_ASSET_BYTES as u64);
        }
        Err(EditableMergeError::InvalidMerged(EditableAssetError::PackageTooLarge {
            actual: got,
        })) => assert_eq!(got, actual),
        other => panic!("expected hard encoded-output limit rejection, got {other:?}"),
    }
    assert_eq!(harness.package_bytes(base.asset), before);
    assert_eq!(harness.saved(base.asset), base);
    harness.assert_no_transaction();
}

#[test]
fn mismatched_input_identity_is_rejected_before_workspace_publication() {
    let base = asset(1);
    let harness = Harness::new(std::slice::from_ref(&base));
    let before = harness.package_bytes(base.asset);
    let other = asset(2);
    assert!(plan_editable_merge(&base, &other, &base).is_err());
    assert!(plan_editable_merge(&base, &base, &other).is_err());
    assert!(plan_editable_merge(&other, &base, &base).is_err());
    assert_eq!(harness.package_bytes(base.asset), before);
    assert_eq!(harness.saved(base.asset), base);
    harness.assert_no_transaction();
}

#[test]
fn invalid_combined_subasset_names_cannot_reach_workspace_publication() {
    let mut base = asset(1);
    base.subassets.push(EditableSubAsset {
        id: 8,
        name: "other".into(),
        kind: 42,
    });
    let harness = Harness::new(std::slice::from_ref(&base));
    let before = harness.package_bytes(base.asset);
    let mut local = base.clone();
    local.subassets[0].name = "collision".into();
    let mut incoming = base.clone();
    incoming.subassets[1].name = "collision".into();
    let plan = plan_editable_merge(&base, &local, &incoming).unwrap();
    assert!(plan.conflicts().is_empty());
    assert!(matches!(
        plan.resolve(&BTreeMap::new(), MERGE_LIMIT),
        Err(EditableMergeError::InvalidMerged(
            EditableAssetError::DuplicateSubAssetName
        ))
    ));
    assert_eq!(harness.package_bytes(base.asset), before);
    assert_eq!(harness.saved(base.asset), base);
    harness.assert_no_transaction();
}

#[test]
fn intervening_workspace_save_rejects_stale_merged_hash_without_overwrite() {
    let base = asset(1);
    let mut harness = Harness::new(std::slice::from_ref(&base));
    let stale_editor = harness.workspace();
    let other_editor = harness.workspace();
    let mut local = base.clone();
    local.canonical_settings = b"local settings".to_vec();
    let mut incoming = base.clone();
    incoming.authoring_payload = b"incoming payload".to_vec();
    let plan = plan_editable_merge(&base, &local, &incoming).unwrap();
    let merged = plan.resolve(&BTreeMap::new(), MERGE_LIMIT).unwrap();
    // Deterministically interleave a second editor's commit between planning
    // and publication; a preview-only change must also invalidate expected_hash.
    let mut concurrent = base.clone();
    concurrent.preview = Some(b"other editor preview".to_vec());
    other_editor
        .save(
            vec![edit(&base, concurrent.clone())],
            MAX_EDITABLE_WORKSPACE_ENCODED_BYTES,
        )
        .unwrap();
    let committed_bytes = harness.package_bytes(base.asset);
    assert_eq!(
        stale_editor.save(
            vec![edit(&base, merged)],
            MAX_EDITABLE_WORKSPACE_ENCODED_BYTES,
        ),
        Err(EditableAssetError::WorkspaceEditConflict {
            asset: base.asset,
            expected: ContentHash::digest(&base.encode().unwrap()),
            actual: ContentHash::digest(&concurrent.encode().unwrap()),
        })
    );
    assert_eq!(harness.package_bytes(base.asset), committed_bytes);
    assert_eq!(harness.saved(base.asset), concurrent);
    harness.assert_no_transaction();
    let (_, bytes) = harness.cook(base.asset);
    assert_eq!(
        bytes,
        cooked_bytes(&base.canonical_settings, &base.authoring_payload)
    );
}
