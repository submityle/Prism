//! Output admission at actual execution/cache boundaries, not snapshot preflight.

use prism_asset_build_core::{
    ArtifactDescriptor, ArtifactSpec, AssetId, BuildError, BuildExecutionError, BuildExecutor,
    BuildGraph, BuildJob, BuildRevision, ContentHash, DerivedDataCache, JobId, RevisionStore,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const EXECUTOR: BuildExecutor = BuildExecutor {
    protocol_version: 1,
};

fn spec(id: u128) -> ArtifactSpec {
    ArtifactSpec {
        asset: AssetId(id),
        processor_id: 123,
        processor_version: 1,
        canonical_settings: id.to_le_bytes().to_vec(),
        profile_hash: [0; 32],
        toolchain_fingerprint: Vec::new(),
        environment_fingerprint: Vec::new(),
        kind: 42,
        schema: 1,
        alignment: 4,
    }
}

fn graph(bytes: &'static [u8], limit: u64, calls: Arc<AtomicUsize>) -> BuildGraph {
    let mut graph = BuildGraph::new();
    graph
        .add_job(
            BuildJob::new(JobId(1), Vec::new(), move |_: &[&[u8]]| {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(bytes.to_vec())
            })
            .with_max_output_bytes(limit),
        )
        .unwrap();
    graph
}

#[track_caller]
fn assert_limit(error: BuildExecutionError, job: JobId) {
    assert!(
        matches!(&error, BuildExecutionError::InvalidOutput(id, "output exceeds declared byte bound") if *id == job),
        "expected output byte limit failure for {job:?}, got {error:?} ({error})",
    );
}

#[test]
fn job_default_zero_and_direct_graph_execution_limits() {
    let unlimited = BuildJob::new(JobId(1), Vec::new(), |_: &[&[u8]]| Ok(vec![1]));
    assert_eq!(unlimited.max_output_bytes(), u64::MAX);
    let mut unlimited_graph = BuildGraph::new();
    unlimited_graph.add_job(unlimited).unwrap();
    assert_eq!(unlimited_graph.execute().unwrap()[&JobId(1)], [1]);
    let empty = graph(b"", 0, Arc::new(AtomicUsize::new(0)));
    assert!(empty.execute().unwrap()[&JobId(1)].is_empty());
    assert!(empty.execute_parallel_with_limit(2).unwrap()[&JobId(1)].is_empty());
    let bounded =
        BuildJob::new(JobId(1), Vec::new(), |_: &[&[u8]]| Ok(vec![1])).with_max_output_bytes(0);
    assert_eq!(bounded.max_output_bytes(), 0);
    let mut nonempty = BuildGraph::new();
    nonempty.add_job(bounded).unwrap();
    for error in [
        nonempty.execute().unwrap_err(),
        nonempty.execute_parallel_with_limit(2).unwrap_err(),
    ] {
        assert!(
            matches!(error, BuildError::ProcessorFailed { job: JobId(1), source }
            if source.message() == "processor output exceeds declared byte bound")
        );
    }
}

#[test]
fn cold_exact_limit_warm_and_tightened_limit_preserve_publication_boundary() {
    let root = tempfile::tempdir().unwrap();
    let store = RevisionStore::open(root.path()).unwrap();
    let mut ddc = DerivedDataCache::open(root.path().join("ddc")).unwrap();
    let specs = BTreeMap::from([(JobId(1), spec(1))]);
    let calls = Arc::new(AtomicUsize::new(0));
    let too_small = graph(b"four", 3, calls.clone());
    assert_limit(
        EXECUTOR
            .execute(&too_small, &specs, &store, &mut ddc, BuildRevision(0))
            .unwrap_err(),
        JobId(1),
    );
    assert_eq!(store.current_revision().unwrap(), None);
    assert!(ddc.is_empty());
    assert!(!store.cas().blob_path(ContentHash::digest(b"four")).exists());
    let exact = graph(b"four", 4, calls.clone());
    let cold = EXECUTOR
        .execute(&exact, &specs, &store, &mut ddc, BuildRevision(0))
        .unwrap();
    assert!(cold.cache_hits.is_empty());
    let warm = EXECUTOR
        .execute(&exact, &specs, &store, &mut ddc, BuildRevision(1))
        .unwrap();
    assert_eq!(warm.cache_hits, BTreeSet::from([JobId(1)]));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_limit(
        EXECUTOR
            .execute(&too_small, &specs, &store, &mut ddc, BuildRevision(2))
            .unwrap_err(),
        JobId(1),
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(store.current_revision().unwrap(), Some(BuildRevision(1)));
    assert_eq!(
        store.manifest(BuildRevision(1)).unwrap(),
        warm.published.manifest
    );
    assert!(store.manifest(BuildRevision(2)).is_err());
}

#[test]
fn zero_output_limit_accepts_empty_cold_warm_and_verified_outputs() {
    let root = tempfile::tempdir().unwrap();
    let store = RevisionStore::open(root.path()).unwrap();
    let mut ddc = DerivedDataCache::open(root.path().join("ddc")).unwrap();
    let specs = BTreeMap::from([(JobId(1), spec(1))]);
    let calls = Arc::new(AtomicUsize::new(0));
    let empty = graph(b"", 0, calls.clone());
    EXECUTOR
        .execute(&empty, &specs, &store, &mut ddc, BuildRevision(0))
        .unwrap();
    let warm = EXECUTOR
        .execute(&empty, &specs, &store, &mut ddc, BuildRevision(1))
        .unwrap();
    assert_eq!(warm.cache_hits, BTreeSet::from([JobId(1)]));
    let verified = EXECUTOR
        .execute_verified_with_limit(&empty, &specs, &store, &mut ddc, BuildRevision(2), 1)
        .unwrap();
    assert!(verified.cache_hits.is_empty());
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(verified.published.manifest.entries[&AssetId(1)].byte_len, 0);
    let nonempty = graph(b"x", 0, calls.clone());
    assert_limit(
        EXECUTOR
            .execute_verified_with_limit(&nonempty, &specs, &store, &mut ddc, BuildRevision(3), 1)
            .unwrap_err(),
        JobId(1),
    );
    assert_eq!(calls.load(Ordering::SeqCst), 4);
    assert_eq!(store.current_revision().unwrap(), Some(BuildRevision(2)));
    assert!(!store.cas().blob_path(ContentHash::digest(b"x")).exists());
}

#[test]
fn cached_declared_and_actual_lengths_are_checked_before_payload_read() {
    for forged_short in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let store = RevisionStore::open(root.path()).unwrap();
        let mut ddc = DerivedDataCache::open(root.path().join("ddc")).unwrap();
        let spec = spec(1);
        let key = spec.artifact_key(1, &[]).unwrap();
        let mut descriptor = ArtifactDescriptor::for_bytes(
            key,
            spec.kind,
            spec.schema,
            spec.alignment,
            Vec::new(),
            b"oversized",
        );
        if forged_short {
            // A real CAS blob, with a lying descriptor that passes admission.
            store.cas().put(b"oversized").unwrap();
            descriptor.byte_len = 1;
        } else {
            // Leave the CAS file absent: declaration must fail before a
            // missing-blob fallback can invoke the processor.
            assert!(!store.cas().blob_path(descriptor.content_hash).exists());
        }
        ddc.put_descriptor(descriptor.clone()).unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let graph = graph(b"ok", 4, calls.clone());
        let specs = BTreeMap::from([(JobId(1), spec)]);
        assert_limit(
            EXECUTOR
                .execute(&graph, &specs, &store, &mut ddc, BuildRevision(0))
                .unwrap_err(),
            JobId(1),
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(ddc.get(key), Some(descriptor));
        assert_eq!(store.current_revision().unwrap(), None);
    }
}

#[test]
fn in_budget_length_mismatch_and_hash_corruption_rebuild_cache() {
    for corrupt_hash in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let store = RevisionStore::open(root.path()).unwrap();
        let mut ddc = DerivedDataCache::open(root.path().join("ddc")).unwrap();
        let spec = spec(1);
        let key = spec.artifact_key(1, &[]).unwrap();
        let hash = store.cas().put(b"four").unwrap();
        let mut descriptor = ArtifactDescriptor::for_bytes(
            key,
            spec.kind,
            spec.schema,
            spec.alignment,
            Vec::new(),
            b"four",
        );
        if corrupt_hash {
            fs::write(store.cas().blob_path(hash), b"fake").unwrap();
        } else {
            descriptor.byte_len = 1;
        }
        ddc.put_descriptor(descriptor).unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let graph = graph(b"four", 4, calls.clone());
        let specs = BTreeMap::from([(JobId(1), spec)]);
        let result = EXECUTOR
            .execute(&graph, &specs, &store, &mut ddc, BuildRevision(0))
            .unwrap();
        assert!(result.cache_hits.is_empty());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(ddc.get(key).unwrap().byte_len, 4);
        assert_eq!(store.cas().read(hash).unwrap(), b"four");
    }
}

#[test]
fn entries_imported_by_a_real_ddc_refresh_obey_the_later_job_limit() {
    for forged_short in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let store = RevisionStore::open(root.path()).unwrap();
        let mut ddc = DerivedDataCache::open(root.path().join("ddc")).unwrap();
        let baseline = graph(b"old", 4, Arc::new(AtomicUsize::new(0)));
        let baseline_spec = spec(99);
        EXECUTOR
            .execute(
                &baseline,
                &BTreeMap::from([(JobId(1), baseline_spec)]),
                &store,
                &mut ddc,
                BuildRevision(0),
            )
            .unwrap();
        let external = Arc::new(Mutex::new(DerivedDataCache::open(ddc.root()).unwrap()));
        let first_spec = spec(1);
        let second_spec = spec(2);
        let first_key = first_spec.artifact_key(1, &[]).unwrap();
        let second_key = second_spec.artifact_key(1, &[first_key]).unwrap();
        let hash = store.cas().put(b"oversized").unwrap();
        let mut late = ArtifactDescriptor::for_bytes(
            second_key,
            second_spec.kind,
            second_spec.schema,
            second_spec.alignment,
            vec![first_key],
            b"oversized",
        );
        if forged_short {
            late.byte_len = 1;
        }
        assert_eq!(late.content_hash, hash);
        assert!(ddc.get(second_key).is_none());
        let mut graph = BuildGraph::new();
        let insert = late.clone();
        graph
            .add_job(
                BuildJob::new(JobId(1), Vec::new(), move |_: &[&[u8]]| {
                    // Another writer adds an entry after execution has started. The
                    // coordinator's ensuing put refreshes its own in-memory index.
                    external
                        .lock()
                        .unwrap()
                        .put_descriptor(insert.clone())
                        .unwrap();
                    Ok(b"root".to_vec())
                })
                .with_max_output_bytes(4),
            )
            .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let second_calls = calls.clone();
        graph
            .add_job(
                BuildJob::new(JobId(2), vec![JobId(1)], move |_: &[&[u8]]| {
                    second_calls.fetch_add(1, Ordering::SeqCst);
                    Ok(b"ok".to_vec())
                })
                .with_max_output_bytes(4),
            )
            .unwrap();
        let specs = BTreeMap::from([(JobId(1), first_spec), (JobId(2), second_spec)]);
        assert_limit(
            EXECUTOR
                .execute(&graph, &specs, &store, &mut ddc, BuildRevision(1))
                .unwrap_err(),
            JobId(2),
        );
        assert_eq!(ddc.get(second_key), Some(late));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(store.current_revision().unwrap(), Some(BuildRevision(0)));
        assert!(store.manifest(BuildRevision(1)).is_err());
    }
}

#[test]
fn either_determinism_run_exceeding_limit_writes_neither_output() {
    for warm in [false, true] {
        for first_oversized in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let store = RevisionStore::open(root.path()).unwrap();
            let mut ddc = DerivedDataCache::open(root.path().join("ddc")).unwrap();
            let specs = BTreeMap::from([(JobId(1), spec(1))]);
            let key = specs[&JobId(1)].artifact_key(1, &[]).unwrap();
            let previous = if warm {
                let old = graph(b"previous", 8, Arc::new(AtomicUsize::new(0)));
                EXECUTOR
                    .execute(&old, &specs, &store, &mut ddc, BuildRevision(0))
                    .unwrap();
                ddc.get(key)
            } else {
                None
            };
            // Cold publication starts at zero; a warm store already committed
            // zero above. Reach execution rather than revision admission.
            let attempted_revision = BuildRevision(if warm { 1 } else { 0 });
            let calls = Arc::new(AtomicUsize::new(0));
            let worker_calls = calls.clone();
            let mut graph = BuildGraph::new();
            graph
                .add_job(
                    BuildJob::new(JobId(1), Vec::new(), move |_: &[&[u8]]| {
                        let call = worker_calls.fetch_add(1, Ordering::SeqCst);
                        Ok(if first_oversized || call == 1 {
                            b"too big".to_vec()
                        } else {
                            b"a".to_vec()
                        })
                    })
                    .with_max_output_bytes(1),
                )
                .unwrap();
            assert_limit(
                EXECUTOR
                    .execute_verified_with_limit(
                        &graph,
                        &specs,
                        &store,
                        &mut ddc,
                        attempted_revision,
                        2,
                    )
                    .unwrap_err(),
                JobId(1),
            );
            assert_eq!(
                calls.load(Ordering::SeqCst),
                if first_oversized { 1 } else { 2 }
            );
            assert_eq!(
                store.current_revision().unwrap(),
                warm.then_some(BuildRevision(0))
            );
            assert_eq!(ddc.get(key), previous);
            for bytes in [b"a".as_slice(), b"too big".as_slice()] {
                assert!(!store.cas().blob_path(ContentHash::digest(bytes)).exists());
            }
            assert!(store.manifest(attempted_revision).is_err());
        }
    }
}
