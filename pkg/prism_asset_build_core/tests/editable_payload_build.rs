//! External editable payloads through workspace leases, real Cook, DDC and CAS.

use prism_asset_build_core::{
    AssetId, BuildExecution, BuildResources, BuildRevision, ContentHash, DerivedDataCache,
    EditableAsset, EditableBuildError, EditableBuildOptions, EditableBuildPlan,
    EditableCookDependency, EditableCookInput, EditableCooker, EditableCookerDescriptor,
    EditableCookerRegistry, EditableDependency, EditableDependencyKind as Kind, EditablePayload,
    EditablePayloadRef, EditableSubAsset, EditableSubAssetReference, EditableWorkspace,
    EditableWorkspaceEdit, JobId, ProcessorError, ProcessorIdentity, RevisionStore, SourceRecord,
    MAX_EDITABLE_WORKSPACE_ENCODED_BYTES,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::Duration;
use tempfile::{tempdir, TempDir};

type SeenPayloads = Arc<Mutex<BTreeMap<AssetId, Arc<[u8]>>>>;

fn asset(id: u128, payload: impl Into<EditablePayload>) -> EditableAsset {
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
        payload,
        None,
    )
    .unwrap()
}

fn missing_payload() -> EditablePayload {
    EditablePayload::External(EditablePayloadRef {
        content_hash: ContentHash::digest(b"not in the payload store"),
        byte_len: 24,
    })
}

struct Cooker {
    calls: Arc<AtomicUsize>,
    seen: SeenPayloads,
}

impl EditableCooker for Cooker {
    fn descriptor(&self) -> EditableCookerDescriptor {
        EditableCookerDescriptor {
            authoring_kind: 42,
            identity: ProcessorIdentity::new(4242, 1),
            output_kind: 43,
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
        self.seen
            .lock()
            .unwrap()
            .insert(input.asset, Arc::clone(&input.authoring_payload));
        let mut bytes = input.authoring_payload.to_vec();
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
    seen: SeenPayloads,
    next_revision: u64,
}

impl Harness {
    fn new() -> Self {
        let directory = tempdir().unwrap();
        fs::create_dir(directory.path().join("workspace")).unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = SeenPayloads::default();
        let mut registry = EditableCookerRegistry::new();
        registry
            .register(Cooker {
                calls: Arc::clone(&calls),
                seen: Arc::clone(&seen),
            })
            .unwrap();
        Self {
            directory,
            registry,
            calls,
            seen,
            next_revision: 0,
        }
    }

    fn workspace(&self) -> EditableWorkspace {
        EditableWorkspace::open(self.directory.path().join("workspace")).unwrap()
    }

    fn write(&self, value: &EditableAsset) {
        value
            .write_to_path(
                self.directory
                    .path()
                    .join("workspace")
                    .join(format!("{}.prism", value.asset.0)),
            )
            .unwrap();
    }

    fn replace(&self, previous: &EditableAsset, replacement: EditableAsset) {
        self.workspace()
            .save(
                vec![EditableWorkspaceEdit {
                    expected_hash: ContentHash::digest(&previous.encode().unwrap()),
                    replacement,
                }],
                MAX_EDITABLE_WORKSPACE_ENCODED_BYTES,
            )
            .unwrap();
    }

    fn plan(
        &self,
        roots: &[AssetId],
        budget: u64,
    ) -> Result<EditableBuildPlan, EditableBuildError> {
        EditableBuildPlan::from_workspace(
            &self.workspace(),
            roots,
            &self.registry,
            &EditableBuildOptions {
                max_materialized_payload_bytes: budget,
                ..Default::default()
            },
        )
    }

    fn run(&mut self, plan: &EditableBuildPlan) -> (BuildExecution, BTreeMap<AssetId, Vec<u8>>) {
        // Reopen both persistent stores for every execution.
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
        let bytes = execution
            .published
            .manifest
            .entries
            .iter()
            .map(|(id, descriptor)| {
                assert_eq!(execution.artifact_keys[&JobId(id.0)], descriptor.key);
                assert_eq!(ddc.get(descriptor.key).as_ref(), Some(descriptor));
                let bytes = store.cas().read(descriptor.content_hash).unwrap();
                assert_eq!(bytes.len() as u64, descriptor.byte_len);
                assert_eq!(ContentHash::digest(&bytes), descriptor.content_hash);
                (*id, bytes)
            })
            .collect();
        (execution, bytes)
    }
}

#[test]
fn inline_to_external_storage_and_editor_metadata_keep_keys_and_durable_ddc_hits() {
    let mut harness = Harness::new();
    let inline = asset(1, b"same saved bytes".to_vec());
    harness.write(&inline);
    let baseline_plan = harness.plan(&[inline.asset], 1024).unwrap();
    let (baseline, baseline_bytes) = harness.run(&baseline_plan);
    assert!(baseline.cache_hits.is_empty());
    let lease = harness
        .workspace()
        .store_payload(&b"same saved bytes"[..], 1024)
        .unwrap();
    let mut external = inline.clone();
    external.authoring_payload = EditablePayload::External(lease.reference());
    external.source =
        SourceRecord::embedded("moved.bin", "other/format", b"different source".to_vec()).unwrap();
    external.preview = Some(b"new preview".to_vec());
    harness.replace(&inline, external.clone());
    let snapshot = harness
        .workspace()
        .snapshot(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)
        .unwrap();
    assert_eq!(snapshot[&inline.asset], external);
    let cloned = snapshot.clone();
    drop(snapshot);
    let plan = EditableBuildPlan::from_workspace_snapshot(
        cloned,
        &[inline.asset],
        &harness.registry,
        &Default::default(),
    )
    .unwrap();
    assert_eq!(plan.specs()[&JobId(1)], baseline_plan.specs()[&JobId(1)]);
    drop(lease);
    let (warm, bytes) = harness.run(&plan);
    assert_eq!(warm.cache_hits, BTreeSet::from([JobId(1)]));
    assert_eq!(warm.artifact_keys, baseline.artifact_keys);
    assert_eq!(bytes, baseline_bytes);
    assert_eq!(bytes[&AssetId(1)], b"same saved bytes");
    assert_eq!(harness.calls.load(Ordering::SeqCst), 1);
    // Returning to inline storage also reuses the same durable artifact.
    harness.replace(&external, inline);
    let inline_plan = harness.plan(&[AssetId(1)], 1024).unwrap();
    let (warm_again, _) = harness.run(&inline_plan);
    assert_eq!(warm_again.cache_hits, BTreeSet::from([JobId(1)]));
    assert_eq!(warm_again.artifact_keys, baseline.artifact_keys);
    assert_eq!(harness.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn repeated_external_hash_shares_one_arc_and_counts_once_beside_inline_bytes() {
    let mut harness = Harness::new();
    let bytes = b"shared payload";
    let lease = harness.workspace().store_payload(&bytes[..], 1024).unwrap();
    let first = asset(1, EditablePayload::External(lease.reference()));
    let second = asset(2, EditablePayload::External(lease.reference()));
    let mut root = asset(3, bytes.to_vec());
    root.dependencies = vec![
        EditableDependency {
            asset: first.asset,
            kind: Kind::Hard,
        },
        EditableDependency {
            asset: second.asset,
            kind: Kind::Hard,
        },
    ];
    for value in [&first, &second, &root] {
        harness.write(value);
    }
    let actual = 2 * bytes.len() as u64;
    assert!(
        matches!(harness.plan(&[root.asset], actual - 1), Err(EditableBuildError::MaterializedPayloadBudgetExceeded { actual: got, limit }) if got == actual && limit == actual - 1)
    );
    assert_eq!(harness.calls.load(Ordering::SeqCst), 0);
    let plan = harness.plan(&[root.asset, root.asset], actual).unwrap();
    let (cold, outputs) = harness.run(&plan);
    assert!(cold.cache_hits.is_empty());
    assert_eq!(outputs[&root.asset], bytes.repeat(3));
    {
        let seen = harness.seen.lock().unwrap();
        assert!(Arc::ptr_eq(&seen[&first.asset], &seen[&second.asset]));
        assert_eq!(seen[&first.asset].as_ref(), bytes);
        assert_eq!(seen[&root.asset].as_ref(), bytes);
    }
    let reopened = harness.plan(&[root.asset], actual).unwrap();
    let (warm, _) = harness.run(&reopened);
    assert_eq!(
        warm.cache_hits,
        BTreeSet::from([JobId(1), JobId(2), JobId(3)])
    );
    assert_eq!(harness.calls.load(Ordering::SeqCst), 3);
}

#[test]
fn only_roots_and_hard_dependencies_or_references_materialize_payloads() {
    let mut harness = Harness::new();
    let lease = harness
        .workspace()
        .store_payload(&b"hard bytes"[..], 1024)
        .unwrap();
    let hard_dependency = asset(1, EditablePayload::External(lease.reference()));
    let soft = asset(2, missing_payload());
    let editor = asset(3, missing_payload());
    let mut root = asset(4, b"root".to_vec());
    let unrelated = asset(5, missing_payload());
    let hard_reference = asset(6, EditablePayload::External(lease.reference()));
    root.dependencies = vec![
        EditableDependency {
            asset: AssetId(1),
            kind: Kind::Hard,
        },
        EditableDependency {
            asset: AssetId(2),
            kind: Kind::Soft,
        },
        EditableDependency {
            asset: AssetId(3),
            kind: Kind::EditorOnly,
        },
    ];
    root.subasset_references = vec![
        EditableSubAssetReference {
            asset: AssetId(2),
            subasset: 4,
            kind: Kind::Soft,
        },
        EditableSubAssetReference {
            asset: AssetId(3),
            subasset: 4,
            kind: Kind::EditorOnly,
        },
        EditableSubAssetReference {
            asset: AssetId(6),
            subasset: 4,
            kind: Kind::Hard,
        },
    ];
    for value in [
        &hard_dependency,
        &soft,
        &editor,
        &root,
        &unrelated,
        &hard_reference,
    ] {
        harness.write(value);
    }
    let plan = harness.plan(&[root.asset], 14).unwrap();
    assert_eq!(
        plan.specs().keys().copied().collect::<Vec<_>>(),
        [JobId(1), JobId(4), JobId(6)]
    );
    // These references really are unavailable when explicitly selected.
    for id in [soft.asset, editor.asset, unrelated.asset] {
        assert!(matches!(
            harness.plan(&[id], 1024),
            Err(EditableBuildError::Editable(_))
        ));
    }
    let (_, outputs) = harness.run(&plan);
    assert_eq!(outputs[&root.asset], b"roothard byteshard bytes");
    assert_eq!(harness.calls.load(Ordering::SeqCst), 3);
}

#[test]
fn aggregate_materialization_preflight_wins_over_the_first_missing_blob() {
    let harness = Harness::new();
    let missing = asset(1, missing_payload());
    let inline = asset(2, vec![7; 8]);
    harness.write(&missing);
    harness.write(&inline);
    // Asset 1 would fail immediately if resolution began before the whole
    // selected closure had been admitted. No cooker is executed in planning.
    assert!(matches!(
        harness.plan(&[missing.asset, inline.asset], 31),
        Err(EditableBuildError::MaterializedPayloadBudgetExceeded {
            actual: 32,
            limit: 31
        })
    ));
    assert!(matches!(
        harness.plan(&[missing.asset, inline.asset], 32),
        Err(EditableBuildError::Editable(_))
    ));
    assert_eq!(harness.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn plain_snapshot_requires_explicit_workspace_for_selected_external_payloads() {
    let harness = Harness::new();
    let lease = harness
        .workspace()
        .store_payload(&b"stored"[..], 1024)
        .unwrap();
    let external = asset(1, EditablePayload::External(lease.reference()));
    let inline = asset(2, b"inline".to_vec());
    let snapshot = BTreeMap::from([(external.asset, external), (inline.asset, inline)]);
    assert!(matches!(
        EditableBuildPlan::from_snapshot(
            snapshot.clone(),
            &[AssetId(1)],
            &harness.registry,
            &Default::default()
        ),
        Err(EditableBuildError::ExternalPayloadRequiresWorkspace { asset: AssetId(1) })
    ));
    // An unselected external record needs no storage access or resolver.
    assert!(EditableBuildPlan::from_snapshot(
        snapshot,
        &[AssetId(2)],
        &harness.registry,
        &Default::default()
    )
    .is_ok());
}

#[test]
fn materialization_budget_validates_bounds_and_counts_equal_inline_payloads_separately() {
    let harness = Harness::new();
    assert_eq!(
        EditableBuildOptions::default().max_materialized_payload_bytes,
        512 * 1024 * 1024
    );
    let snapshot = BTreeMap::from([
        (AssetId(1), asset(1, vec![3; 4])),
        (AssetId(2), asset(2, vec![3; 4])),
    ]);
    let build = |budget| {
        EditableBuildPlan::from_snapshot(
            snapshot.clone(),
            &[AssetId(1), AssetId(2)],
            &harness.registry,
            &EditableBuildOptions {
                max_materialized_payload_bytes: budget,
                ..Default::default()
            },
        )
    };
    let hard_limit = 4 * 1024 * 1024 * 1024u64;
    for requested in [0, hard_limit + 1] {
        assert!(
            matches!(build(requested), Err(EditableBuildError::InvalidMaterializedPayloadBudget { requested: got, limit }) if got == requested && limit == hard_limit)
        );
    }
    assert!(matches!(
        build(7),
        Err(EditableBuildError::MaterializedPayloadBudgetExceeded {
            actual: 8,
            limit: 7
        })
    ));
    assert!(build(8).is_ok());
    assert!(build(hard_limit).is_ok());

    // A legal maximum-size reference plus one inline byte exceeds the hard
    // aggregate cap. Admission must reject the metadata without opening the
    // missing blob or allocating a multi-GiB test fixture.
    harness.write(&asset(
        3,
        EditablePayload::External(EditablePayloadRef {
            content_hash: ContentHash::digest(b"large unavailable payload"),
            byte_len: hard_limit,
        }),
    ));
    harness.write(&asset(4, vec![0]));
    assert!(
        matches!(harness.plan(&[AssetId(3), AssetId(4)], hard_limit), Err(EditableBuildError::MaterializedPayloadBudgetExceeded { actual, limit }) if actual == hard_limit + 1 && limit == hard_limit)
    );
}

#[test]
fn contradictory_lengths_for_one_hash_are_rejected_before_resolution() {
    let harness = Harness::new();
    let hash = ContentHash::digest(b"missing shared blob");
    harness.write(&asset(
        1,
        EditablePayload::External(EditablePayloadRef {
            content_hash: hash,
            byte_len: 8,
        }),
    ));
    harness.write(&asset(
        2,
        EditablePayload::External(EditablePayloadRef {
            content_hash: hash,
            byte_len: 9,
        }),
    ));
    assert!(
        matches!(harness.plan(&[AssetId(1), AssetId(2)], 1024), Err(EditableBuildError::ConflictingPayloadLength { content_hash, expected: 8, actual: 9 }) if content_hash == hash)
    );
}

#[test]
fn captured_snapshot_leases_old_payload_until_the_plan_owns_materialized_bytes() {
    let mut harness = Harness::new();
    let lease = harness
        .workspace()
        .store_payload(&b"original bytes"[..], 1024)
        .unwrap();
    let external = asset(1, EditablePayload::External(lease.reference()));
    harness.write(&external);
    let snapshot = harness
        .workspace()
        .snapshot(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)
        .unwrap();
    let cloned = snapshot.clone();
    drop(snapshot);
    let replacement = asset(1, b"replacement bytes".to_vec());
    harness.replace(&external, replacement);
    drop(lease);
    // The saved package no longer names the blob. The remaining snapshot clone
    // must protect it until the build has finished resolving the old bytes.
    assert!(
        harness
            .workspace()
            .collect_payloads(Duration::ZERO)
            .unwrap()
            .deferred_due_to_active_readers
    );
    let plan = EditableBuildPlan::from_workspace_snapshot(
        cloned,
        &[external.asset],
        &harness.registry,
        &Default::default(),
    )
    .unwrap();
    let collected = harness
        .workspace()
        .collect_payloads(Duration::ZERO)
        .unwrap();
    assert!(!collected.deferred_due_to_active_readers);
    assert_eq!(collected.sweep.deleted, 1);
    // After collection, execution must use frozen bytes without reopening the
    // workspace payload store.
    let (frozen, bytes) = harness.run(&plan);
    assert_eq!(bytes[&AssetId(1)], b"original bytes");
    let changed_plan = harness.plan(&[AssetId(1)], 1024).unwrap();
    let (changed, bytes) = harness.run(&changed_plan);
    assert!(changed.cache_hits.is_empty());
    assert_ne!(changed.artifact_keys, frozen.artifact_keys);
    assert_eq!(bytes[&AssetId(1)], b"replacement bytes");
    assert_eq!(harness.calls.load(Ordering::SeqCst), 2);
}
