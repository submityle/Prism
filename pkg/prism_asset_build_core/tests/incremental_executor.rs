//! Public-API acceptance tests for DirtySet-driven incremental execution.

use prism_asset_build_core::{
    ArtifactSpec, AssetId, BuildError, BuildExecution, BuildExecutionError, BuildExecutor,
    BuildGraph, BuildJob, BuildResources, BuildRevision, DerivedDataCache, DirtySetError, JobId,
    ProcessorError, PublishError, RevisionStore,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

const EXECUTOR: BuildExecutor = BuildExecutor {
    protocol_version: 1,
};

type Specs = BTreeMap<JobId, ArtifactSpec>;
type Counts = Arc<BTreeMap<JobId, Arc<AtomicUsize>>>;

struct Fixture {
    _root: tempfile::TempDir,
    store: RevisionStore,
    ddc: DerivedDataCache,
}

impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let store = RevisionStore::open(root.path()).unwrap();
        let ddc = DerivedDataCache::open(root.path().join("ddc")).unwrap();
        Self {
            _root: root,
            store,
            ddc,
        }
    }

    fn full(&mut self, graph: &BuildGraph, specs: &Specs) -> BuildExecution {
        EXECUTOR
            .execute(graph, specs, &self.store, &mut self.ddc, BuildRevision(0))
            .unwrap()
    }

    fn incremental(
        &mut self,
        graph: &BuildGraph,
        specs: &Specs,
        dirty: &prism_asset_build_core::DirtySet,
    ) -> Result<BuildExecution, BuildExecutionError> {
        EXECUTOR.execute_incremental(
            graph,
            specs,
            dirty,
            &self.store,
            &mut self.ddc,
            BuildRevision(1),
        )
    }
}

fn spec(id: u128) -> ArtifactSpec {
    ArtifactSpec {
        asset: AssetId(10_000 + id),
        processor_id: 41,
        processor_version: 1,
        canonical_settings: id.to_le_bytes().to_vec(),
        profile_hash: [5; 32],
        toolchain_fingerprint: b"incremental-toolchain".to_vec(),
        environment_fingerprint: b"incremental-environment".to_vec(),
        kind: 19,
        schema: 1,
        alignment: 16,
    }
}

fn specs() -> Specs {
    [1_u128, 2, 3, 4, 5]
        .into_iter()
        .map(|id| (JobId(id), spec(id)))
        .collect()
}

fn counts() -> Counts {
    Arc::new(
        [1_u128, 2, 3, 4, 5]
            .into_iter()
            .map(|id| (JobId(id), Arc::new(AtomicUsize::new(0))))
            .collect(),
    )
}

fn graph(counts: &Counts, fail_root: Option<Arc<AtomicBool>>) -> BuildGraph {
    let mut graph = BuildGraph::new();

    let root_failure = fail_root.unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
    let root_count = Arc::clone(&counts[&JobId(1)]);
    graph
        .add_job(BuildJob::new(JobId(1), [], move |_: &[&[u8]]| {
            root_count.fetch_add(1, Ordering::SeqCst);
            if root_failure.load(Ordering::SeqCst) {
                return Err(ProcessorError::new("incremental root failure"));
            }
            Ok(b"one".to_vec())
        }))
        .unwrap();

    let second_count = Arc::clone(&counts[&JobId(2)]);
    graph
        .add_job(BuildJob::new(JobId(2), [], move |_: &[&[u8]]| {
            second_count.fetch_add(1, Ordering::SeqCst);
            Ok(b"two".to_vec())
        }))
        .unwrap();

    let join_count = Arc::clone(&counts[&JobId(3)]);
    graph
        .add_job(BuildJob::new(
            JobId(3),
            [JobId(1), JobId(2)],
            move |inputs: &[&[u8]]| {
                join_count.fetch_add(1, Ordering::SeqCst);
                assert_eq!(inputs, &[b"one".as_slice(), b"two".as_slice()]);
                Ok(b"one+two".to_vec())
            },
        ))
        .unwrap();

    let leaf_count = Arc::clone(&counts[&JobId(4)]);
    graph
        .add_job(BuildJob::new(
            JobId(4),
            [JobId(3)],
            move |inputs: &[&[u8]]| {
                leaf_count.fetch_add(1, Ordering::SeqCst);
                assert_eq!(inputs, &[b"one+two".as_slice()]);
                Ok(b"final".to_vec())
            },
        ))
        .unwrap();

    let independent_count = Arc::clone(&counts[&JobId(5)]);
    graph
        .add_job(BuildJob::new(JobId(5), [], move |_: &[&[u8]]| {
            independent_count.fetch_add(1, Ordering::SeqCst);
            Ok(b"five".to_vec())
        }))
        .unwrap();

    graph
}

fn all_jobs() -> BTreeSet<JobId> {
    [1_u128, 2, 3, 4, 5].into_iter().map(JobId).collect()
}

fn assert_counts(counts: &Counts, expected: &[(u128, usize)]) {
    for &(id, count) in expected {
        assert_eq!(counts[&JobId(id)].load(Ordering::SeqCst), count, "job {id}");
    }
}

#[test]
fn dirty_set_has_deterministic_successor_closure_and_rejects_unknown_jobs() {
    let counts = counts();
    let graph = graph(&counts, None);
    let dirty = graph.dirty_set([JobId(1)]).unwrap();

    assert_eq!(dirty.direct_jobs().collect::<Vec<_>>(), [JobId(1)]);
    assert_eq!(
        dirty.jobs().collect::<Vec<_>>(),
        [JobId(1), JobId(3), JobId(4)]
    );
    assert!(matches!(
        graph.dirty_set([JobId(999)]),
        Err(DirtySetError::UnknownJob(JobId(999)))
    ));
}

#[test]
fn incremental_execution_reuses_unaffected_cache_and_forces_successor_closure() {
    let mut fixture = Fixture::new();
    let counts = counts();
    let graph = graph(&counts, None);
    let specs = specs();
    let first = fixture.full(&graph, &specs);
    let dirty = graph.dirty_set([JobId(1)]).unwrap();

    let second = fixture.incremental(&graph, &specs, &dirty).unwrap();

    assert_eq!(
        second.cache_hits,
        [JobId(2), JobId(5)].into_iter().collect()
    );
    assert_eq!(second.artifact_keys, first.artifact_keys);
    assert_eq!(
        second.published.manifest.entries,
        first.published.manifest.entries
    );
    assert_eq!(
        fixture.store.current_revision().unwrap(),
        Some(BuildRevision(1))
    );
    assert_counts(&counts, &[(1, 2), (2, 1), (3, 2), (4, 2), (5, 1)]);
}

#[test]
fn empty_dirty_set_reuses_every_verified_cache_entry_and_publishes_next_revision() {
    let mut fixture = Fixture::new();
    let counts = counts();
    let graph = graph(&counts, None);
    let specs = specs();
    let first = fixture.full(&graph, &specs);
    let dirty = graph.dirty_set([]).unwrap();

    let second = fixture.incremental(&graph, &specs, &dirty).unwrap();

    assert_eq!(second.cache_hits, all_jobs());
    assert_eq!(second.artifact_keys, first.artifact_keys);
    assert_eq!(
        second.published.manifest.entries,
        first.published.manifest.entries
    );
    assert_eq!(second.published.manifest.revision, BuildRevision(1));
    assert_counts(&counts, &[(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]);
}

#[test]
fn incremental_cache_reuse_requires_verified_cas_bytes() {
    let mut fixture = Fixture::new();
    let counts = counts();
    let graph = graph(&counts, None);
    let specs = specs();
    fixture.full(&graph, &specs);
    let descriptor =
        fixture.store.manifest(BuildRevision(0)).unwrap().entries[&specs[&JobId(5)].asset].clone();
    let mut damaged = fixture.store.cas().read(descriptor.content_hash).unwrap();
    damaged[0] ^= 0xff;
    fs::write(
        fixture.store.cas().blob_path(descriptor.content_hash),
        damaged,
    )
    .unwrap();

    let dirty = graph.dirty_set([JobId(1)]).unwrap();
    let second = fixture.incremental(&graph, &specs, &dirty).unwrap();

    assert_eq!(second.cache_hits, [JobId(2)].into_iter().collect());
    assert_counts(&counts, &[(1, 2), (2, 1), (3, 2), (4, 2), (5, 2)]);
    assert_eq!(
        fixture.store.cas().read(descriptor.content_hash).unwrap(),
        b"five"
    );
}

#[test]
fn dirty_plan_from_another_graph_is_rejected_before_execution() {
    let counts = counts();
    let graph = graph(&counts, None);
    let mut other_graph = BuildGraph::new();
    other_graph
        .add_job(BuildJob::new(JobId(99), [], |_: &[&[u8]]| {
            Ok(b"other".to_vec())
        }))
        .unwrap();
    let other_dirty = other_graph.dirty_set([JobId(99)]).unwrap();
    let specs = specs();
    let mut fixture = Fixture::new();

    let error = fixture
        .incremental(&graph, &specs, &other_dirty)
        .unwrap_err();

    assert!(matches!(
        error,
        BuildExecutionError::UnknownDirtyJob(JobId(99))
    ));
    assert_eq!(fixture.store.current_revision().unwrap(), None);
    assert!(fixture.ddc.is_empty());
    assert_counts(&counts, &[(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]);
}

#[test]
fn incremental_failure_keeps_previous_revision_and_does_not_publish() {
    let mut fixture = Fixture::new();
    let counts = counts();
    let failure = Arc::new(AtomicBool::new(false));
    let graph = graph(&counts, Some(Arc::clone(&failure)));
    let specs = specs();
    let first = fixture.full(&graph, &specs);
    failure.store(true, Ordering::SeqCst);
    let dirty = graph.dirty_set([JobId(1)]).unwrap();

    let error = fixture.incremental(&graph, &specs, &dirty).unwrap_err();

    assert!(matches!(
        error,
        BuildExecutionError::Graph(BuildError::ProcessorFailed { job: JobId(1), .. })
    ));
    assert_eq!(
        fixture.store.current_revision().unwrap(),
        Some(BuildRevision(0))
    );
    assert_eq!(
        fixture.store.manifest(BuildRevision(0)).unwrap(),
        first.published.manifest
    );
    assert!(matches!(
        fixture.store.manifest_bytes(BuildRevision(1)),
        Err(PublishError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound
    ));
}

#[test]
fn incremental_serial_and_parallel_have_identical_results() {
    let mut serial = Fixture::new();
    let serial_counts = counts();
    let serial_graph = graph(&serial_counts, None);
    let specs = specs();
    serial.full(&serial_graph, &specs);
    let dirty = serial_graph.dirty_set([JobId(1)]).unwrap();
    let serial_result = serial.incremental(&serial_graph, &specs, &dirty).unwrap();

    let mut parallel = Fixture::new();
    let parallel_counts = counts();
    let parallel_graph = graph(&parallel_counts, None);
    parallel.full(&parallel_graph, &specs);
    let parallel_dirty = parallel_graph.dirty_set([JobId(1)]).unwrap();
    let parallel_result = EXECUTOR
        .execute_incremental_parallel_with_limit(
            &parallel_graph,
            &specs,
            &parallel_dirty,
            &parallel.store,
            &mut parallel.ddc,
            BuildRevision(1),
            2,
        )
        .unwrap();

    assert_eq!(parallel_result.artifact_keys, serial_result.artifact_keys);
    assert_eq!(
        parallel_result.published.manifest.canonical_bytes(),
        serial_result.published.manifest.canonical_bytes()
    );
    assert_eq!(parallel_result.cache_hits, serial_result.cache_hits);
    assert_counts(&serial_counts, &[(1, 2), (2, 1), (3, 2), (4, 2), (5, 1)]);
    assert_counts(&parallel_counts, &[(1, 2), (2, 1), (3, 2), (4, 2), (5, 1)]);
}

#[test]
fn incremental_resource_budget_and_determinism_gate_remain_explicit() {
    let mut fixture = Fixture::new();
    let counts = counts();
    let graph = graph(&counts, None);
    let specs = specs();
    fixture.full(&graph, &specs);
    let dirty = graph.dirty_set([JobId(1)]).unwrap();
    let budget = BuildResources {
        cpu_threads: 2,
        memory_bytes: 1,
        ..BuildResources::UNLIMITED
    };

    let result = EXECUTOR
        .execute_incremental_with_resources(
            &graph,
            &specs,
            &dirty,
            &fixture.store,
            &mut fixture.ddc,
            BuildRevision(1),
            2,
            budget,
            true,
        )
        .unwrap();

    assert!(result.cache_hits.is_empty());
    assert_counts(&counts, &[(1, 3), (2, 3), (3, 3), (4, 3), (5, 3)]);
    assert_eq!(
        fixture.store.current_revision().unwrap(),
        Some(BuildRevision(1))
    );
}
