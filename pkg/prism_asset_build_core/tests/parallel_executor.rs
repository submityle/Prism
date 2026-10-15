//! Public-API acceptance tests for bounded execution, cache repair, and publication.
//!
//! Synchronization timeouts only prevent hangs on a scheduler regression; no
//! assertion depends on how many milliseconds a processor takes to finish.

use prism_asset_build_core::{
    ArtifactDescriptor, ArtifactSpec, AssetId, BuildError, BuildExecution, BuildExecutionError,
    BuildExecutor, BuildGraph, BuildJob, BuildRevision, CasError, ContentHash, DerivedDataCache,
    JobId, ProcessorError, PublishError, RevisionStore,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

const SYNC_TIMEOUT: Duration = Duration::from_secs(10);
const EXECUTOR: BuildExecutor = BuildExecutor {
    protocol_version: 1,
};
type Specs = BTreeMap<JobId, ArtifactSpec>;
type CallCounts = Arc<BTreeMap<JobId, AtomicUsize>>;

struct Fixture {
    root: tempfile::TempDir,
    store: RevisionStore,
    ddc: DerivedDataCache,
}

impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let store = RevisionStore::open(root.path()).unwrap();
        let ddc = DerivedDataCache::open(root.path().join("ddc")).unwrap();
        Self { root, store, ddc }
    }

    fn parallel(
        &mut self,
        graph: &BuildGraph,
        specs: &Specs,
        revision: u64,
        limit: usize,
    ) -> Result<BuildExecution, BuildExecutionError> {
        EXECUTOR.execute_parallel_with_limit(
            graph,
            specs,
            &self.store,
            &mut self.ddc,
            BuildRevision(revision),
            limit,
        )
    }
}

fn spec(id: u128) -> ArtifactSpec {
    ArtifactSpec {
        asset: AssetId(1000 + id),
        processor_id: 17,
        processor_version: 1,
        canonical_settings: id.to_le_bytes().to_vec(),
        profile_hash: [3; 32],
        toolchain_fingerprint: b"parallel-test-toolchain".to_vec(),
        environment_fingerprint: b"parallel-test-environment".to_vec(),
        kind: 9,
        schema: 1,
        alignment: 16,
    }
}

fn specs_for(ids: impl IntoIterator<Item = u128>) -> Specs {
    ids.into_iter().map(|id| (JobId(id), spec(id))).collect()
}

/// Reusable two-party barrier with a bounded wait, including on regressions
/// that accidentally serialize processors or leave a wave with only one peer.
#[derive(Default)]
struct PairRendezvous {
    state: Mutex<(usize, usize)>, // arrivals, generation
    changed: Condvar,
}

impl PairRendezvous {
    fn wait(&self) {
        let mut state = self.state.lock().unwrap();
        let generation = state.1;
        state.0 += 1;
        if state.0 == 2 {
            state.0 = 0;
            state.1 += 1;
            self.changed.notify_all();
            return;
        }
        let (state, _) = self
            .changed
            .wait_timeout_while(state, SYNC_TIMEOUT, |state| state.1 == generation)
            .unwrap();
        assert_ne!(state.1, generation, "processor rendezvous timed out");
    }
}

/// Forces overlap and orders the processors' final observable actions. This
/// controls the test schedule without assuming OS thread wake-up/return order.
struct OrderedPair {
    rendezvous: PairRendezvous,
    order: [JobId; 2],
    observed: Mutex<Vec<JobId>>,
    changed: Condvar,
}

impl OrderedPair {
    fn new(order: [JobId; 2]) -> Self {
        Self {
            rendezvous: PairRendezvous::default(),
            order,
            observed: Mutex::new(Vec::new()),
            changed: Condvar::new(),
        }
    }

    fn release(&self, id: JobId) {
        self.rendezvous.wait();
        let rank = self.order.iter().position(|entry| *entry == id).unwrap();
        let observed = self.observed.lock().unwrap();
        let (mut observed, _) = self
            .changed
            .wait_timeout_while(observed, SYNC_TIMEOUT, |observed| observed.len() < rank)
            .unwrap();
        assert_eq!(observed.len(), rank, "ordered processor release timed out");
        observed.push(id);
        self.changed.notify_all();
    }

    fn observed(&self) -> Vec<JobId> {
        self.observed.lock().unwrap().clone()
    }
}

// Three roots, fan-out, fan-in, a diamond, cross-level edges, and deliberately
// non-sorted input order. Jobs are inserted in reverse order below.
const MIXED_DAG: &[(u128, &[u128])] = &[
    (10, &[]),
    (20, &[]),
    (30, &[]),
    (40, &[20, 10]),
    (50, &[10]),
    (60, &[30, 40]),
    (70, &[50, 40, 20]),
    (80, &[70, 60]),
];

fn mixed_expected() -> BTreeMap<JobId, Vec<u8>> {
    [
        (10, "10()"),
        (20, "20()"),
        (30, "30()"),
        (40, "40(20()|10())"),
        (50, "50(10())"),
        (60, "60(30()|40(20()|10()))"),
        (70, "70(50(10())|40(20()|10())|20())"),
        (
            80,
            "80(70(50(10())|40(20()|10())|20())|60(30()|40(20()|10())))",
        ),
    ]
    .into_iter()
    .map(|(id, bytes)| (JobId(id), bytes.as_bytes().to_vec()))
    .collect()
}

fn mixed_graph(finish: Option<Arc<OrderedPair>>) -> (BuildGraph, CallCounts) {
    let mut graph = BuildGraph::new();
    let calls: CallCounts = Arc::new(
        MIXED_DAG
            .iter()
            .map(|(id, _)| (JobId(*id), AtomicUsize::new(0)))
            .collect(),
    );
    let expected = Arc::new(mixed_expected());
    for &(id, dependencies) in MIXED_DAG.iter().rev() {
        let calls = Arc::clone(&calls);
        let expected = Arc::clone(&expected);
        let finish = finish.clone();
        graph
            .add_job(BuildJob::new(
                JobId(id),
                dependencies.iter().copied().map(JobId).collect::<Vec<_>>(),
                move |inputs: &[&[u8]]| {
                    calls[&JobId(id)].fetch_add(1, Ordering::SeqCst);
                    assert_eq!(inputs.len(), dependencies.len());
                    let mut output = format!("{id}(").into_bytes();
                    for (index, (&dependency, input)) in
                        dependencies.iter().zip(inputs.iter()).enumerate()
                    {
                        assert_eq!(*input, expected[&JobId(dependency)].as_slice());
                        if index != 0 {
                            output.push(b'|');
                        }
                        output.extend_from_slice(input);
                    }
                    output.push(b')');
                    if matches!(id, 10 | 20) {
                        if let Some(finish) = &finish {
                            finish.release(JobId(id));
                        }
                    }
                    Ok(output)
                },
            ))
            .unwrap();
    }
    (graph, calls)
}

fn assert_mixed_execution(fixture: &Fixture, execution: &BuildExecution) {
    let expected = mixed_expected();
    let manifest = &execution.published.manifest;
    assert_eq!(manifest.entries.len(), MIXED_DAG.len());
    assert_eq!(execution.artifact_keys.len(), MIXED_DAG.len());
    assert_eq!(
        fixture.store.manifest_bytes(manifest.revision).unwrap(),
        manifest.canonical_bytes()
    );
    assert_eq!(
        execution.published.manifest_hash,
        ContentHash::digest(&manifest.canonical_bytes())
    );
    for &(id, dependencies) in MIXED_DAG {
        let spec = spec(id);
        let key = execution.artifact_keys[&JobId(id)];
        let input_keys: Vec<_> = dependencies
            .iter()
            .map(|dependency| execution.artifact_keys[&JobId(*dependency)])
            .collect();
        assert_eq!(key, spec.artifact_key(1, &input_keys).unwrap());
        let descriptor = &manifest.entries[&spec.asset];
        assert_eq!(
            *descriptor,
            ArtifactDescriptor::for_bytes(
                key,
                spec.kind,
                spec.schema,
                spec.alignment,
                input_keys,
                &expected[&JobId(id)],
            )
        );
        assert_eq!(fixture.ddc.get(key).as_ref(), Some(descriptor));
        assert_eq!(
            fixture.store.cas().read(descriptor.content_hash).unwrap(),
            expected[&JobId(id)]
        );
    }
}

#[test]
fn mixed_dependency_dag_matches_serial_manifest_and_cas_in_both_release_orders() {
    let specs = specs_for(MIXED_DAG.iter().map(|(id, _)| *id));
    let mut serial = Fixture::new();
    let (graph, serial_calls) = mixed_graph(None);
    let reference = EXECUTOR
        .execute(
            &graph,
            &specs,
            &serial.store,
            &mut serial.ddc,
            BuildRevision(0),
        )
        .unwrap();
    assert!(reference.cache_hits.is_empty());
    assert_mixed_execution(&serial, &reference);
    assert!(serial_calls
        .values()
        .all(|count| count.load(Ordering::SeqCst) == 1));

    for order in [[JobId(10), JobId(20)], [JobId(20), JobId(10)]] {
        let mut parallel = Fixture::new();
        let finish = Arc::new(OrderedPair::new(order));
        let (graph, calls) = mixed_graph(Some(Arc::clone(&finish)));
        let execution = parallel.parallel(&graph, &specs, 0, 2).unwrap();
        assert_eq!(finish.observed(), order);
        assert!(execution.cache_hits.is_empty());
        assert_eq!(execution.artifact_keys, reference.artifact_keys);
        assert_eq!(execution.published, reference.published);
        assert_mixed_execution(&parallel, &execution);
        for descriptor in reference.published.manifest.entries.values() {
            assert_eq!(
                parallel.store.cas().read(descriptor.content_hash).unwrap(),
                serial.store.cas().read(descriptor.content_hash).unwrap()
            );
        }

        // Reopen the persistent DDC and prove every node, including joins,
        // bypasses its processor on a warm parallel build.
        parallel.ddc = DerivedDataCache::open(parallel.ddc.root()).unwrap();
        let warm = parallel.parallel(&graph, &specs, 1, 2).unwrap();
        assert_eq!(warm.cache_hits, specs.keys().copied().collect());
        assert_eq!(warm.artifact_keys, execution.artifact_keys);
        assert_eq!(
            warm.published.manifest.entries,
            execution.published.manifest.entries
        );
        assert!(calls
            .values()
            .all(|count| count.load(Ordering::SeqCst) == 1));
        assert_mixed_execution(&parallel, &warm);
        assert_eq!(
            parallel.store.current_revision().unwrap(),
            Some(BuildRevision(1))
        );
    }
}

struct ActiveProcessor<'a>(&'a AtomicUsize);

impl Drop for ActiveProcessor<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

#[test]
fn concurrency_two_has_exact_peak_two_across_a_wide_ready_set() {
    let mut fixture = Fixture::new();
    let mut graph = BuildGraph::new();
    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let calls = Arc::new(AtomicUsize::new(0));
    let rendezvous = Arc::new(PairRendezvous::default());
    for id in 1_u128..=8 {
        let active = Arc::clone(&active);
        let peak = Arc::clone(&peak);
        let calls = Arc::clone(&calls);
        let rendezvous = Arc::clone(&rendezvous);
        graph
            .add_job(BuildJob::new(JobId(id), [], move |_: &[&[u8]]| {
                let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                let _active = ActiveProcessor(&active);
                peak.fetch_max(current, Ordering::SeqCst);
                assert!(current <= 2, "executor exceeded the concurrency limit");
                calls.fetch_add(1, Ordering::SeqCst);
                // Neither peer can complete before two processors overlap.
                // Reuse the barrier over four waves, not just the first pair.
                rendezvous.wait();
                Ok(id.to_le_bytes().to_vec())
            }))
            .unwrap();
    }
    let specs = specs_for(1..=8);
    let execution = fixture.parallel(&graph, &specs, 0, 2).unwrap();
    assert_eq!(peak.load(Ordering::SeqCst), 2);
    assert_eq!(active.load(Ordering::SeqCst), 0);
    assert_eq!(calls.load(Ordering::SeqCst), 8);
    assert!(execution.cache_hits.is_empty());
    assert_eq!(execution.published.manifest.entries.len(), 8);
    assert_eq!(
        fixture.store.current_revision().unwrap(),
        Some(BuildRevision(0))
    );
}

#[test]
fn parallel_cas_corruption_rebuilds_only_damaged_nodes_then_hits_persistent_cache() {
    let mut fixture = Fixture::new();
    let specs = specs_for(MIXED_DAG.iter().map(|(id, _)| *id));
    let (graph, _) = mixed_graph(None);
    let first = fixture.parallel(&graph, &specs, 0, 2).unwrap();
    let damaged: BTreeSet<_> = [JobId(10), JobId(20), JobId(60)].into_iter().collect();
    for id in &damaged {
        let descriptor = &first.published.manifest.entries[&specs[id].asset];
        let mut bytes = fixture.store.cas().read(descriptor.content_hash).unwrap();
        bytes[0] ^= 0xff; // Same length: the content hash, not length, must catch it.
        fs::write(
            fixture.store.cas().blob_path(descriptor.content_hash),
            bytes,
        )
        .unwrap();
        assert!(matches!(
            fixture.store.cas().read(descriptor.content_hash),
            Err(CasError::Corrupt { .. })
        ));
    }

    // Keep this store open: reopening while CURRENT's CAS is corrupt correctly
    // fails closed before the executor can exercise its cache-repair path.
    fixture.ddc = DerivedDataCache::open(fixture.ddc.root()).unwrap();
    let finish = Arc::new(OrderedPair::new([JobId(20), JobId(10)]));
    let (graph, calls) = mixed_graph(Some(Arc::clone(&finish)));
    let rebuilt = fixture.parallel(&graph, &specs, 1, 2).unwrap();
    assert_eq!(finish.observed(), [JobId(20), JobId(10)]);
    assert_eq!(rebuilt.artifact_keys, first.artifact_keys);
    assert_eq!(
        rebuilt.published.manifest.entries,
        first.published.manifest.entries
    );
    assert_eq!(
        rebuilt.cache_hits,
        specs
            .keys()
            .copied()
            .filter(|id| !damaged.contains(id))
            .collect()
    );
    for (id, count) in calls.iter() {
        assert_eq!(
            count.load(Ordering::SeqCst),
            usize::from(damaged.contains(id))
        );
    }
    assert_mixed_execution(&fixture, &rebuilt);

    // Repair must survive reopening both stores, rather than only fixing the
    // in-memory descriptor index. Also exercise the host-bounded public API.
    fixture.store = RevisionStore::open(fixture.root.path()).unwrap();
    fixture.ddc = DerivedDataCache::open(fixture.ddc.root()).unwrap();
    assert_eq!(
        fixture.store.current_revision().unwrap(),
        Some(BuildRevision(1))
    );
    let warm = EXECUTOR
        .execute_parallel(
            &graph,
            &specs,
            &fixture.store,
            &mut fixture.ddc,
            BuildRevision(2),
        )
        .unwrap();
    assert_eq!(warm.cache_hits, specs.keys().copied().collect());
    assert_eq!(warm.artifact_keys, first.artifact_keys);
    assert_eq!(
        warm.published.manifest.entries,
        first.published.manifest.entries
    );
    for (id, count) in calls.iter() {
        assert_eq!(
            count.load(Ordering::SeqCst),
            usize::from(damaged.contains(id))
        );
    }
    assert_mixed_execution(&fixture, &warm);
    assert_eq!(
        fixture.store.current_revision().unwrap(),
        Some(BuildRevision(2))
    );
}

fn publish_previous(fixture: &mut Fixture) -> BuildExecution {
    let mut graph = BuildGraph::new();
    graph
        .add_job(BuildJob::new(JobId(900), [], |_: &[&[u8]]| {
            Ok(b"previous revision bytes".to_vec())
        }))
        .unwrap();
    EXECUTOR
        .execute(
            &graph,
            &specs_for([900]),
            &fixture.store,
            &mut fixture.ddc,
            BuildRevision(0),
        )
        .unwrap()
}

fn assert_previous_preserved(fixture: &Fixture, previous: &BuildExecution) {
    let manifest = &previous.published.manifest;
    assert_eq!(
        fixture.store.current_revision().unwrap(),
        Some(manifest.revision)
    );
    assert_eq!(
        fs::read(fixture.root.path().join("CURRENT")).unwrap(),
        manifest.revision.0.to_le_bytes()
    );
    assert_eq!(
        fixture.store.manifest_bytes(manifest.revision).unwrap(),
        manifest.canonical_bytes()
    );
    assert!(matches!(
        fixture.store.manifest_bytes(BuildRevision(1)),
        Err(PublishError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound
    ));
    let reopened = RevisionStore::open(fixture.root.path()).unwrap();
    assert_eq!(
        reopened.current_revision().unwrap(),
        Some(manifest.revision)
    );
    assert_eq!(reopened.manifest(manifest.revision).unwrap(), *manifest);
    for descriptor in manifest.entries.values() {
        assert_eq!(
            reopened.cas().read(descriptor.content_hash).unwrap(),
            b"previous revision bytes"
        );
        assert_eq!(fixture.ddc.get(descriptor.key).as_ref(), Some(descriptor));
    }
}

fn later_failure_keeps_previous(panic_worker: bool) {
    let mut fixture = Fixture::new();
    let previous = publish_previous(&mut fixture);
    let mut graph = BuildGraph::new();
    for (id, bytes) in [(1, b"one"), (2, b"two")] {
        graph
            .add_job(BuildJob::new(JobId(id), [], move |_: &[&[u8]]| {
                Ok(bytes.to_vec())
            }))
            .unwrap();
    }
    let finish = Arc::new(OrderedPair::new([JobId(20), JobId(10)]));
    let peer_calls = Arc::new(AtomicUsize::new(0));
    let peer_finish = Arc::clone(&finish);
    let peer_count = Arc::clone(&peer_calls);
    graph
        .add_job(BuildJob::new(
            JobId(10),
            [JobId(2), JobId(1)],
            move |inputs: &[&[u8]]| {
                assert_eq!(inputs, &[b"two".as_slice(), b"one".as_slice()]);
                peer_count.fetch_add(1, Ordering::SeqCst);
                let mut output = inputs[0].to_vec();
                output.push(b'+');
                output.extend_from_slice(inputs[1]);
                peer_finish.release(JobId(10));
                Ok(output)
            },
        ))
        .unwrap();
    let failure_finish = Arc::clone(&finish);
    graph
        .add_job(BuildJob::new(
            JobId(20),
            [JobId(1)],
            move |inputs: &[&[u8]]| {
                assert_eq!(inputs, &[b"one".as_slice()]);
                failure_finish.release(JobId(20));
                if panic_worker {
                    panic!("injected worker panic");
                }
                Err(ProcessorError::new("injected processor failure"))
            },
        ))
        .unwrap();
    let descendant_calls = Arc::new(AtomicUsize::new(0));
    let descendant_count = Arc::clone(&descendant_calls);
    graph
        .add_job(BuildJob::new(
            JobId(30),
            [JobId(10), JobId(20)],
            move |_: &[&[u8]]| {
                descendant_count.fetch_add(1, Ordering::SeqCst);
                Ok(b"must not run".to_vec())
            },
        ))
        .unwrap();
    let specs = specs_for([1, 2, 10, 20, 30]);
    // An uncaught worker panic would fail this test instead of producing the
    // required typed error with the original job ID.
    let error = fixture.parallel(&graph, &specs, 1, 2).unwrap_err();
    match error {
        BuildExecutionError::Graph(BuildError::ProcessorFailed { job, source }) => {
            assert_eq!(job, JobId(20));
            assert_eq!(
                source.message(),
                if panic_worker {
                    "build worker thread panicked"
                } else {
                    "injected processor failure"
                }
            );
        }
        error => panic!("expected processor failure, got {error:?}"),
    }
    assert_eq!(finish.observed(), [JobId(20), JobId(10)]);
    assert_eq!(peer_calls.load(Ordering::SeqCst), 1);
    assert_eq!(descendant_calls.load(Ordering::SeqCst), 0);

    // Earlier successful waves may populate persistent DDC/CAS even though
    // the build fails. Publication atomicity does NOT mean rolling them back.
    let persisted_ddc = DerivedDataCache::open(fixture.ddc.root()).unwrap();
    for (id, bytes) in [(1, b"one"), (2, b"two")] {
        let key = specs[&JobId(id)].artifact_key(1, &[]).unwrap();
        let descriptor = persisted_ddc
            .get(key)
            .expect("completed wave leaves reusable cache");
        assert_eq!(
            fixture.store.cas().read(descriptor.content_hash).unwrap(),
            bytes
        );
    }
    // The successful peer must also be drained before the failure returns;
    // do not require rollback or a DDC entry for every in-flight peer.
    assert_eq!(
        fixture
            .store
            .cas()
            .read(ContentHash::digest(b"two+one"))
            .unwrap(),
        b"two+one"
    );
    assert_previous_preserved(&fixture, &previous);
}

#[test]
fn parallel_processor_failure_allows_cache_side_effects_but_keeps_previous_revision() {
    later_failure_keeps_previous(false);
}

#[test]
fn parallel_worker_panic_is_a_job_failure_and_keeps_previous_revision() {
    later_failure_keeps_previous(true);
}

#[test]
fn parallel_failure_selection_uses_job_id_in_both_controlled_release_orders() {
    for order in [[JobId(2), JobId(5)], [JobId(5), JobId(2)]] {
        let mut fixture = Fixture::new();
        let previous = publish_previous(&mut fixture);
        let finish = Arc::new(OrderedPair::new(order));
        let mut graph = BuildGraph::new();
        for id in [5, 2] {
            let finish = Arc::clone(&finish);
            graph
                .add_job(BuildJob::new(JobId(id), [], move |_: &[&[u8]]| {
                    let error = ProcessorError::new(format!("failure-{id}"));
                    finish.release(JobId(id));
                    Err(error)
                }))
                .unwrap();
        }
        let descendant_calls = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&descendant_calls);
        graph
            .add_job(BuildJob::new(
                JobId(9),
                [JobId(2), JobId(5)],
                move |_: &[&[u8]]| {
                    count.fetch_add(1, Ordering::SeqCst);
                    Ok(Vec::new())
                },
            ))
            .unwrap();
        let error = fixture
            .parallel(&graph, &specs_for([2, 5, 9]), 1, 2)
            .unwrap_err();
        assert!(matches!(
            error,
            BuildExecutionError::Graph(BuildError::ProcessorFailed { job: JobId(2), source })
                if source.message() == "failure-2"
        ));
        assert_eq!(finish.observed(), order);
        assert_eq!(descendant_calls.load(Ordering::SeqCst), 0);
        assert_previous_preserved(&fixture, &previous);
    }
}

#[test]
fn zero_concurrency_rejects_before_processors_cache_or_current_change() {
    let mut fixture = Fixture::new();
    let previous = publish_previous(&mut fixture);
    let ddc_before = fs::read(fixture.ddc.index_path()).unwrap();
    let (graph, calls) = mixed_graph(None);
    let specs = specs_for(MIXED_DAG.iter().map(|(id, _)| *id));
    assert!(matches!(
        fixture.parallel(&graph, &specs, 1, 0),
        Err(BuildExecutionError::Graph(BuildError::InvalidConcurrency))
    ));
    assert!(calls
        .values()
        .all(|count| count.load(Ordering::SeqCst) == 0));
    assert_eq!(fs::read(fixture.ddc.index_path()).unwrap(), ddc_before);
    assert_previous_preserved(&fixture, &previous);
}
