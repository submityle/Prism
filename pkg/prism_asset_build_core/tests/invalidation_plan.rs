use prism_asset_build_core::profile::{
    ProfileField, ProfileFieldType, ProfileLayer, ProfileLayerValues, ProfileMergePolicy,
    ProfileSchema, ProfileValue,
};
use prism_asset_build_core::{
    BuildExecutor, BuildGraph, BuildInputSnapshot, BuildJob, BuildRevision, DirtySetError,
    InvalidationReason, JobId, SourceSnapshot,
};

fn graph() -> BuildGraph {
    let mut graph = BuildGraph::new();
    graph.add_job(job(1, &[])).unwrap();
    graph.add_job(job(2, &[1])).unwrap();
    graph.add_job(job(3, &[2])).unwrap();
    graph.add_job(job(4, &[])).unwrap();
    graph
}

fn job(id: u128, dependencies: &[u128]) -> BuildJob {
    BuildJob::new(
        JobId(id),
        dependencies.iter().copied().map(JobId).collect::<Vec<_>>(),
        |_: &[&[u8]]| Ok(Vec::new()),
    )
}

fn profile(quality: u64, memory: u64) -> prism_asset_build_core::profile::ProfileProjection {
    let schema = ProfileSchema::new(
        "test.profile",
        1,
        [
            (
                "quality",
                ProfileField::new(
                    ProfileFieldType::Unsigned { min: 0, max: 10 },
                    ProfileMergePolicy::ScalarOverride,
                ),
            ),
            (
                "memory_mb",
                ProfileField::new(
                    ProfileFieldType::Unsigned {
                        min: 256,
                        max: 65_536,
                    },
                    ProfileMergePolicy::ScalarOverride,
                ),
            ),
        ],
    )
    .unwrap();
    let mut layer = ProfileLayerValues::new(ProfileLayer::EngineBaseline);
    layer
        .set("quality", ProfileValue::Unsigned(quality))
        .unwrap();
    layer
        .set("memory_mb", ProfileValue::Unsigned(memory))
        .unwrap();
    schema
        .resolve(&[layer])
        .unwrap()
        .project(["quality"])
        .unwrap()
}

#[test]
fn source_settings_and_profile_changes_map_to_direct_jobs_and_successors() {
    let source_before = SourceSnapshot::new("textures/hero.png", b"one".to_vec()).unwrap();
    let source_after = SourceSnapshot::new("textures/hero.png", b"two".to_vec()).unwrap();
    let before = BuildInputSnapshot::new()
        .with_source(JobId(1), &source_before)
        .with_settings(JobId(2), b"settings-v1".to_vec())
        .with_profile(JobId(3), profile(3, 1024));
    let current = BuildInputSnapshot::new()
        .with_source(JobId(1), &source_after)
        .with_settings(JobId(2), b"settings-v2".to_vec())
        .with_profile(JobId(3), profile(7, 1024));

    let plan = before.invalidation_plan(&current, &graph()).unwrap();

    assert_eq!(
        plan.dirty_set().direct_jobs().collect::<Vec<_>>(),
        vec![JobId(1), JobId(2), JobId(3)]
    );
    assert_eq!(
        plan.dirty_set().jobs().collect::<Vec<_>>(),
        vec![JobId(1), JobId(2), JobId(3)]
    );
    assert!(!plan.dirty_set().contains(JobId(4)));
    assert!(matches!(
        &plan.changes()[0],
        InvalidationReason::Source { job: JobId(1), .. }
    ));
    assert!(matches!(
        &plan.changes()[1],
        InvalidationReason::Settings { job: JobId(2), .. }
    ));
    match &plan.changes()[2] {
        InvalidationReason::Profile { job, fields, .. } => {
            assert_eq!(*job, JobId(3));
            assert_eq!(
                fields
                    .iter()
                    .map(|field| field.field.as_str())
                    .collect::<Vec<_>>(),
                ["quality"]
            );
        }
        other => panic!("expected profile invalidation, got {other:?}"),
    }
}

#[test]
fn unrelated_profile_fields_do_not_create_a_dirty_change() {
    let before = BuildInputSnapshot::new().with_profile(JobId(3), profile(3, 1024));
    let current = BuildInputSnapshot::new().with_profile(JobId(3), profile(3, 4096));

    let plan = before.invalidation_plan(&current, &graph()).unwrap();

    assert!(plan.is_empty());
    assert!(plan.dirty_set().is_empty());
    assert!(plan.changes().is_empty());
}

#[test]
fn added_and_removed_inputs_are_reported_as_cross_boundary_changes() {
    let source = SourceSnapshot::new("textures/new.png", b"asset".to_vec()).unwrap();
    let before = BuildInputSnapshot::new()
        .with_source(JobId(1), &source)
        .with_settings(JobId(2), b"old-settings".to_vec())
        .with_profile(JobId(3), profile(3, 1024));
    let current = BuildInputSnapshot::new()
        .with_source(JobId(4), &source)
        .with_settings(JobId(1), b"new-settings".to_vec())
        .with_profile(JobId(2), profile(7, 1024));

    let plan = before.invalidation_plan(&current, &graph()).unwrap();

    assert_eq!(
        plan.dirty_set().direct_jobs().collect::<Vec<_>>(),
        [JobId(1), JobId(2), JobId(3), JobId(4)]
    );
    assert_eq!(plan.changes().len(), 6);
    assert!(matches!(
        &plan.changes()[0],
        InvalidationReason::Source {
            job: JobId(1),
            before: Some(_),
            after: None
        }
    ));
    assert!(matches!(
        &plan.changes()[1],
        InvalidationReason::Settings {
            job: JobId(1),
            before: None,
            after: Some(_)
        }
    ));
    assert!(matches!(
        &plan.changes()[2],
        InvalidationReason::Settings {
            job: JobId(2),
            before: Some(_),
            after: None
        }
    ));
    assert!(matches!(
        &plan.changes()[3],
        InvalidationReason::Profile {
            job: JobId(2),
            before: None,
            after: Some(_),
            fields,
        } if fields.iter().map(|field| field.field.as_str()).collect::<Vec<_>>() == ["quality"]
    ));
    assert!(matches!(
        &plan.changes()[4],
        InvalidationReason::Profile {
            job: JobId(3),
            before: Some(_),
            after: None,
            fields,
        } if fields.iter().map(|field| field.field.as_str()).collect::<Vec<_>>() == ["quality"]
    ));
    assert!(matches!(
        &plan.changes()[5],
        InvalidationReason::Source {
            job: JobId(4),
            before: None,
            after: Some(_)
        }
    ));
}

#[test]
fn unchanged_inputs_produce_an_empty_plan_and_identical_canonical_bytes() {
    let source = SourceSnapshot::new("textures/hero.png", b"same".to_vec()).unwrap();
    let snapshot = BuildInputSnapshot::new()
        .with_profile(JobId(3), profile(5, 1024))
        .with_settings(JobId(2), b"canonical-settings".to_vec())
        .with_source(JobId(1), &source);
    let equivalent = BuildInputSnapshot::new()
        .with_source(JobId(1), &source)
        .with_settings(JobId(2), b"canonical-settings".to_vec())
        .with_profile(JobId(3), profile(5, 4096));

    let first = snapshot.invalidation_plan(&equivalent, &graph()).unwrap();
    let second = equivalent.invalidation_plan(&snapshot, &graph()).unwrap();

    assert!(first.is_empty());
    assert!(first.dirty_set().is_empty());
    assert!(first.changes().is_empty());
    assert_eq!(first.canonical_bytes(), second.canonical_bytes());
}

#[test]
fn unknown_job_ids_are_rejected_even_when_the_input_is_unchanged() {
    let graph = graph();
    let source = SourceSnapshot::new("textures/orphan.png", b"same".to_vec()).unwrap();

    let snapshots = [
        BuildInputSnapshot::new().with_source(JobId(99), &source),
        BuildInputSnapshot::new().with_settings(JobId(99), b"settings".to_vec()),
        BuildInputSnapshot::new().with_profile(JobId(99), profile(5, 1024)),
    ];
    for snapshot in snapshots {
        assert_eq!(
            snapshot.invalidation_plan(&snapshot, &graph).unwrap_err(),
            DirtySetError::UnknownJob(JobId(99))
        );
    }

    let empty = BuildInputSnapshot::new();
    let changed_unknown = empty.clone().with_settings(JobId(100), b"new".to_vec());
    assert_eq!(
        empty
            .invalidation_plan(&changed_unknown, &graph)
            .unwrap_err(),
        DirtySetError::UnknownJob(JobId(100))
    );
}

#[test]
fn explanation_order_and_bytes_do_not_depend_on_snapshot_insertion_order() {
    let source_a = SourceSnapshot::new("a.bin", b"a".to_vec()).unwrap();
    let source_b = SourceSnapshot::new("b.bin", b"b".to_vec()).unwrap();
    let before_a = BuildInputSnapshot::new()
        .with_source(JobId(1), &source_a)
        .with_settings(JobId(2), b"a".to_vec())
        .with_profile(JobId(3), profile(2, 1024));
    let after_a = BuildInputSnapshot::new()
        .with_source(JobId(1), &source_b)
        .with_settings(JobId(2), b"b".to_vec())
        .with_profile(JobId(3), profile(8, 1024));
    let before_b = BuildInputSnapshot::new()
        .with_profile(JobId(3), profile(2, 2048))
        .with_settings(JobId(2), b"a".to_vec())
        .with_source(JobId(1), &source_a);
    let after_b = BuildInputSnapshot::new()
        .with_profile(JobId(3), profile(8, 4096))
        .with_source(JobId(1), &source_b)
        .with_settings(JobId(2), b"b".to_vec());

    let first = before_a.invalidation_plan(&after_a, &graph()).unwrap();
    let second = before_b.invalidation_plan(&after_b, &graph()).unwrap();

    assert_eq!(first.changes(), second.changes());
    assert_eq!(first.dirty_set(), second.dirty_set());
    assert_eq!(first.canonical_bytes(), second.canonical_bytes());
    assert_eq!(
        first
            .changes()
            .iter()
            .map(|change| (
                change.job(),
                match change {
                    InvalidationReason::Source { .. } => 0,
                    InvalidationReason::Settings { .. } => 1,
                    InvalidationReason::Profile { .. } => 2,
                }
            ))
            .collect::<Vec<_>>(),
        [(JobId(1), 0), (JobId(2), 1), (JobId(3), 2)]
    );
}

#[test]
fn rebuild_explanation_round_trips_canonical_bytes() {
    let source_before = SourceSnapshot::new("mesh/hero.bin", b"old".to_vec()).unwrap();
    let source_after = SourceSnapshot::new("mesh/hero.bin", b"new".to_vec()).unwrap();
    let before = BuildInputSnapshot::new().with_source(JobId(1), &source_before);
    let current = BuildInputSnapshot::new().with_source(JobId(1), &source_after);
    let plan = before.invalidation_plan(&current, &graph()).unwrap();

    let loaded =
        prism_asset_build_core::InvalidationPlan::from_canonical_bytes(&plan.canonical_bytes())
            .unwrap();
    assert_eq!(loaded, plan);
    assert_eq!(loaded.canonical_bytes(), plan.canonical_bytes());
}

#[test]
fn semantic_snapshot_round_trip_preserves_profile_evidence_and_unknown_fields() {
    let before = BuildInputSnapshot::new()
        .with_source(
            JobId(1),
            &SourceSnapshot::new("mesh/hero.bin", b"old".to_vec()).unwrap(),
        )
        .with_settings(JobId(2), b"canonical-settings".to_vec())
        .with_profile(JobId(3), profile(3, 1024))
        .with_profile_hash(JobId(4), [3; 32]);
    let current = BuildInputSnapshot::new()
        .with_source(
            JobId(1),
            &SourceSnapshot::new("mesh/hero.bin", b"new".to_vec()).unwrap(),
        )
        .with_settings(JobId(2), b"changed-settings".to_vec())
        .with_profile(JobId(3), profile(7, 1024))
        .with_profile(JobId(4), profile(7, 1024));
    let decode = |snapshot: &BuildInputSnapshot| {
        BuildInputSnapshot::from_canonical_bytes(&snapshot.canonical_bytes()).unwrap()
    };
    assert_eq!(decode(&before), before);
    let expected = before.invalidation_plan(&current, &graph()).unwrap();
    let actual = decode(&before)
        .invalidation_plan(&decode(&current), &graph())
        .unwrap();
    assert_eq!(actual, expected);
    assert!(matches!(actual.changes().last().unwrap(),
        InvalidationReason::Profile { fields, .. } if fields.is_empty()));
    assert!(matches!(&actual.changes()[2],
        InvalidationReason::Profile { fields, .. } if fields.len() == 1 && fields[0].field == "quality"));
}

#[test]
fn decoders_reject_truncation_noncanonical_order_counts_and_unchanged_reasons() {
    use prism_asset_build_core::InvalidationPlan;
    let before = BuildInputSnapshot::new().with_settings(JobId(1), b"old".to_vec());
    let after = BuildInputSnapshot::new()
        .with_settings(JobId(1), b"new".to_vec())
        .with_settings(JobId(4), b"other".to_vec());
    let bytes = before
        .invalidation_plan(&after, &graph())
        .unwrap()
        .canonical_bytes();
    for end in 0..bytes.len() {
        assert!(
            InvalidationPlan::from_canonical_bytes(&bytes[..end]).is_err(),
            "prefix {end}"
        );
    }
    let snapshot = after.canonical_bytes();
    for end in 0..snapshot.len() {
        assert!(
            BuildInputSnapshot::from_canonical_bytes(&snapshot[..end]).is_err(),
            "snapshot prefix {end}"
        );
    }
    let header_len = b"PRISM-REBUILD-EXPLANATION\0\x01".len();
    let mut unordered = bytes.clone();
    let offset = header_len + 8;
    let (first, second) = unordered[offset..offset + 32].split_at_mut(16);
    first.swap_with_slice(second);
    assert!(InvalidationPlan::from_canonical_bytes(&unordered).is_err());
    let mut huge = b"PRISM-BUILD-INPUTS\0\x01".to_vec();
    huge.extend_from_slice(&u64::MAX.to_le_bytes());
    assert!(BuildInputSnapshot::from_canonical_bytes(&huge).is_err());
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(InvalidationPlan::from_canonical_bytes(&trailing).is_err());

    // One settings reason, but both fingerprints are equal. Canonical framing
    // alone cannot make this a real semantic change.
    let mut unchanged = b"PRISM-REBUILD-EXPLANATION\0\x01".to_vec();
    for _ in 0..2 {
        unchanged.extend_from_slice(&1u64.to_le_bytes());
        unchanged.extend_from_slice(&1u128.to_le_bytes());
    }
    unchanged.extend_from_slice(&1u64.to_le_bytes());
    unchanged.push(1);
    unchanged.extend_from_slice(&1u128.to_le_bytes());
    for _ in 0..2 {
        unchanged.push(1);
        unchanged.extend_from_slice(&[7; 32]);
    }
    assert!(InvalidationPlan::from_canonical_bytes(&unchanged).is_err());
}

#[test]
fn explained_build_reopens_reuses_cache_propagates_source_and_retains_failure_baseline() {
    use prism_asset_build_core::{AssetId, DerivedDataCache, RevisionStore};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    let root = tempfile::tempdir().unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let independent = Arc::new(AtomicUsize::new(0));
    let executor = BuildExecutor {
        protocol_version: 1,
    };
    let build = |revision: u64, data: &[u8], fail: bool| {
        let source = SourceSnapshot::new("inputs/source.bin", data.to_vec()).unwrap();
        let mut dag = BuildGraph::new();
        dag.add_job(source.build_job(JobId(1))).unwrap();
        let calls = calls.clone();
        dag.add_job(BuildJob::new(
            JobId(2),
            vec![JobId(1)],
            move |inputs: &[&[u8]]| {
                calls.fetch_add(1, Ordering::SeqCst);
                if fail {
                    return Err(prism_asset_build_core::ProcessorError::new(
                        "deliberate failure",
                    ));
                }
                Ok(inputs[0].to_vec())
            },
        ))
        .unwrap();
        let independent = independent.clone();
        dag.add_job(BuildJob::new(JobId(3), vec![], move |_: &[&[u8]]| {
            independent.fetch_add(1, Ordering::SeqCst);
            Ok(b"independent".to_vec())
        }))
        .unwrap();
        let mut specs = std::collections::BTreeMap::new();
        specs.insert(JobId(1), source.artifact_spec(AssetId(10)));
        let mut dependent_spec = source.artifact_spec(AssetId(20));
        dependent_spec.processor_id = 42;
        dependent_spec.canonical_settings = b"fixed dependent contract".to_vec();
        specs.insert(JobId(2), dependent_spec.clone());
        dependent_spec.asset = AssetId(30);
        dependent_spec.processor_id = 43;
        specs.insert(JobId(3), dependent_spec);
        let snapshot = BuildInputSnapshot::new()
            .with_source(JobId(1), &source)
            .with_settings(JobId(2), b"fixed dependent contract".to_vec())
            .with_profile(JobId(2), profile(3, 1024));
        let store = RevisionStore::open(root.path()).unwrap();
        let mut ddc = DerivedDataCache::open(root.path().join("ddc")).unwrap();
        executor.execute_explained_with_limit(
            &dag,
            &specs,
            &snapshot,
            &store,
            &mut ddc,
            BuildRevision(revision),
            2,
            false,
        )
    };
    let cold = build(0, b"old", false).unwrap();
    assert!(cold.cache_hits.is_empty());
    let initial = executor
        .load_committed_build_inputs(root.path(), BuildRevision(0))
        .unwrap();
    assert!(!initial.plan.is_empty());
    assert_eq!(initial.manifest_hash, cold.published.manifest_hash);
    let warm = build(1, b"old", false).unwrap();
    assert_eq!(warm.cache_hits.len(), 3);
    assert!(executor
        .load_committed_build_inputs(root.path(), BuildRevision(1))
        .unwrap()
        .plan
        .is_empty());
    let changed = build(2, b"new", false).unwrap();
    assert_eq!(
        changed.cache_hits.into_iter().collect::<Vec<_>>(),
        [JobId(3)]
    );
    let changed_inputs = executor
        .load_committed_build_inputs(root.path(), BuildRevision(2))
        .unwrap();
    assert_eq!(
        changed_inputs
            .plan
            .dirty_set()
            .direct_jobs()
            .collect::<Vec<_>>(),
        [JobId(1)]
    );
    assert_eq!(
        changed_inputs.plan.dirty_set().jobs().collect::<Vec<_>>(),
        [JobId(1), JobId(2)]
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(independent.load(Ordering::SeqCst), 1);
    assert!(build(3, b"broken", true).is_err());
    let reopened = RevisionStore::open(root.path()).unwrap();
    assert_eq!(reopened.current_revision().unwrap(), Some(BuildRevision(2)));
    assert_eq!(
        executor
            .load_committed_build_inputs(root.path(), BuildRevision(2))
            .unwrap(),
        changed_inputs
    );
    assert!(executor
        .load_committed_build_inputs(root.path(), BuildRevision(3))
        .is_err());
    // The failed source is never used as the next semantic baseline.
    assert_eq!(build(3, b"new", false).unwrap().cache_hits.len(), 3);
    assert!(executor
        .load_committed_build_inputs(root.path(), BuildRevision(3))
        .unwrap()
        .plan
        .is_empty());
}

#[test]
fn cold_explained_build_rejects_unqueryable_history_before_processing() {
    use prism_asset_build_core::{
        AssetId, BuildExecutionError, ContentHash, DerivedDataCache, RevisionStore,
    };
    use std::{
        fs,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
    };
    let root = tempfile::tempdir().unwrap();
    RevisionStore::open(root.path()).unwrap();
    let journal_path = root.path().join("transactions/journal.bin");
    // A legacy Prepare without a published pointer is recoverable by the
    // store, but it carries no supported input-record transaction binding.
    let mut journal = b"PRISM-TXN-JNL\0\x02".to_vec();
    let mut payload = [0; 83];
    payload[0] = 1;
    journal.extend_from_slice(b"TXN1");
    journal.extend_from_slice(&83u32.to_le_bytes());
    journal.extend_from_slice(&payload);
    journal.extend_from_slice(&ContentHash::digest(&payload).0);
    fs::write(&journal_path, journal).unwrap();
    let store = RevisionStore::open(root.path()).unwrap();
    assert_eq!(store.current_revision().unwrap(), None);
    let recovered_journal = fs::read(&journal_path).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let processor_calls = calls.clone();
    let source = SourceSnapshot::new("a.bin", b"content".to_vec()).unwrap();
    let mut dag = BuildGraph::new();
    dag.add_job(BuildJob::new(JobId(1), [], move |_: &[&[u8]]| {
        processor_calls.fetch_add(1, Ordering::SeqCst);
        Ok(b"content".to_vec())
    }))
    .unwrap();
    let specs = std::collections::BTreeMap::from([(JobId(1), source.artifact_spec(AssetId(1)))]);
    let inputs = BuildInputSnapshot::new().with_source(JobId(1), &source);
    let mut ddc = DerivedDataCache::open(root.path().join("ddc")).unwrap();
    let result = BuildExecutor {
        protocol_version: 1,
    }
    .execute_explained_with_limit(
        &dag,
        &specs,
        &inputs,
        &store,
        &mut ddc,
        BuildRevision(0),
        1,
        false,
    );
    assert!(matches!(result, Err(BuildExecutionError::Explanation(_))));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(store.current_revision().unwrap(), None);
    assert_eq!(fs::read(journal_path).unwrap(), recovered_journal);
    assert!(ddc.is_empty());
    assert_eq!(
        fs::read_dir(root.path().join("revisions")).unwrap().count(),
        0
    );
}

#[test]
fn metadata_persistence_failure_cannot_publish_a_successful_build() {
    use prism_asset_build_core::{AssetId, BuildExecutionError, DerivedDataCache, RevisionStore};
    use std::fs;
    let root = tempfile::tempdir().unwrap();
    let store = RevisionStore::open(root.path()).unwrap();
    let mut ddc = DerivedDataCache::open(root.path().join("ddc")).unwrap();
    let journal = root.path().join("transactions/journal.bin");
    let original_journal = fs::read(&journal).unwrap();
    // After the store is open, obstruct the destination without depending on
    // OS permissions (tests may run with elevated privileges).
    fs::rename(
        root.path().join("revisions"),
        root.path().join("held-revisions"),
    )
    .unwrap();
    fs::write(root.path().join("revisions"), b"not a directory").unwrap();
    let source = SourceSnapshot::new("a.bin", b"content".to_vec()).unwrap();
    let mut dag = BuildGraph::new();
    dag.add_job(source.build_job(JobId(1))).unwrap();
    let specs = std::collections::BTreeMap::from([(JobId(1), source.artifact_spec(AssetId(1)))]);
    let inputs = BuildInputSnapshot::new().with_source(JobId(1), &source);
    let executor = BuildExecutor {
        protocol_version: 1,
    };
    assert!(matches!(
        executor.execute_explained_with_limit(
            &dag,
            &specs,
            &inputs,
            &store,
            &mut ddc,
            BuildRevision(0),
            1,
            false
        ),
        Err(BuildExecutionError::Explanation(_))
    ));
    assert_eq!(store.current_revision().unwrap(), None);
    assert_eq!(fs::read(journal).unwrap(), original_journal);
    assert!(executor
        .load_committed_build_inputs(root.path(), BuildRevision(0))
        .is_err());
}

// The parent launches this same integration-test executable twice. Both
// processes capture the same baseline before either may finish its processor.
#[test]
fn explained_publication_process_worker() {
    use prism_asset_build_core::{
        AssetId, BuildExecutionError, DerivedDataCache, PublishError, RevisionStore,
    };
    use std::{
        fs,
        path::PathBuf,
        time::{Duration, Instant},
    };
    let Some(root) = std::env::var_os("PRISM_WHY_RACE_ROOT") else {
        return;
    };
    let root = PathBuf::from(root);
    let index: usize = std::env::var("PRISM_WHY_RACE_INDEX")
        .unwrap()
        .parse()
        .unwrap();
    let store_root = root.join("store");
    let store = RevisionStore::open(&store_root).unwrap();
    // DDCs are deliberately independent: this test races revision publication,
    // not mutable cache index updates. CAS and metadata still share the store.
    let mut ddc = DerivedDataCache::open(root.join(format!("ddc-{index}"))).unwrap();
    let source = SourceSnapshot::new("race.bin", vec![index as u8]).unwrap();
    let spec = source.artifact_spec(AssetId(1));
    let inputs = BuildInputSnapshot::new().with_source(JobId(1), &source);
    let rendezvous = root.clone();
    let mut dag = BuildGraph::new();
    dag.add_job(BuildJob::new(JobId(1), vec![], move |_: &[&[u8]]| {
        fs::write(rendezvous.join(format!("ready-{index}")), b"ready")
            .map_err(|error| prism_asset_build_core::ProcessorError::new(error.to_string()))?;
        let deadline = Instant::now() + Duration::from_secs(15);
        while !rendezvous.join(format!("ready-{}", 1 - index)).exists() {
            if Instant::now() >= deadline {
                return Err(prism_asset_build_core::ProcessorError::new(
                    "competing build did not reach frozen baseline",
                ));
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        Ok(source.bytes().to_vec())
    }))
    .unwrap();
    let result = BuildExecutor {
        protocol_version: 1,
    }
    .execute_explained_with_limit(
        &dag,
        &std::collections::BTreeMap::from([(JobId(1), spec)]),
        &inputs,
        &store,
        &mut ddc,
        BuildRevision(0),
        1,
        false,
    );
    let outcome = match result {
        Ok(_) => "committed",
        Err(BuildExecutionError::Publish(PublishError::Conflict { .. })) => "conflict",
        other => panic!("unexpected competing publication outcome: {other:?}"),
    };
    fs::write(root.join(format!("result-{index}")), outcome).unwrap();
}

#[test]
fn competing_processes_cannot_commit_an_explanation_from_the_losing_build() {
    use prism_asset_build_core::RevisionStore;
    use std::{
        fs,
        process::{Command, Stdio},
    };
    let root = tempfile::tempdir().unwrap();
    let store_root = root.path().join("store");
    RevisionStore::open(&store_root).unwrap();
    let children: Vec<_> = (0..2)
        .map(|index| {
            Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "explained_publication_process_worker",
                    "--nocapture",
                ])
                .env("PRISM_WHY_RACE_ROOT", root.path())
                .env("PRISM_WHY_RACE_INDEX", index.to_string())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap()
        })
        .collect();
    for child in children {
        let output = child.wait_with_output().unwrap();
        assert!(output.status.success(), "{output:?}");
    }
    let outcomes: Vec<_> = (0..2)
        .map(|index| fs::read_to_string(root.path().join(format!("result-{index}"))).unwrap())
        .collect();
    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| outcome.as_str() == "committed")
            .count(),
        1
    );
    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| outcome.as_str() == "conflict")
            .count(),
        1
    );
    let winner = outcomes
        .iter()
        .position(|outcome| outcome == "committed")
        .unwrap();
    let store = RevisionStore::open(&store_root).unwrap();
    assert_eq!(store.current_revision().unwrap(), Some(BuildRevision(0)));
    let committed = BuildExecutor {
        protocol_version: 1,
    }
    .load_committed_build_inputs(&store_root, BuildRevision(0))
    .unwrap();
    let source = SourceSnapshot::new("race.bin", vec![winner as u8]).unwrap();
    assert_eq!(
        committed.snapshot,
        BuildInputSnapshot::new().with_source(JobId(1), &source)
    );
    assert_eq!(
        committed.manifest_hash,
        prism_asset_build_core::ContentHash::digest(
            &store.manifest_bytes(BuildRevision(0)).unwrap()
        )
    );
}
