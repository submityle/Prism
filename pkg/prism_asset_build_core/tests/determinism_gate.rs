//! Public-API acceptance tests for the two-run determinism publication gate.
//!
//! All stores are private to a test. Bounded rendezvous exercise actual worker
//! overlap without using sleeps or assertions about processor duration.

use prism_asset_build_core::{
    ArtifactDescriptor, ArtifactKey, ArtifactSpec, AssetId, BuildError, BuildExecution,
    BuildExecutionError, BuildExecutor, BuildGraph, BuildJob, BuildRevision, CasError,
    ContentAddressedStore, ContentHash, DerivedDataCache, JobId, ProcessorError, PublishError,
    RevisionStore,
};
use std::collections::BTreeMap;
use std::fs;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

const EXECUTOR: BuildExecutor = BuildExecutor {
    protocol_version: 1,
};
const SYNC_TIMEOUT: Duration = Duration::from_secs(10);
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

    fn verified(
        &mut self,
        graph: &BuildGraph,
        specs: &Specs,
        revision: u64,
        limit: usize,
    ) -> Result<BuildExecution, BuildExecutionError> {
        EXECUTOR.execute_verified_with_limit(
            graph,
            specs,
            &self.store,
            &mut self.ddc,
            BuildRevision(revision),
            limit,
        )
    }

    fn reopen(&mut self) {
        self.store = RevisionStore::open(self.root.path()).unwrap();
        self.ddc = DerivedDataCache::open(self.ddc.root()).unwrap();
    }
}

fn spec(id: u128) -> ArtifactSpec {
    ArtifactSpec {
        asset: AssetId(1000 + id),
        processor_id: 29,
        processor_version: 1,
        canonical_settings: id.to_le_bytes().to_vec(),
        profile_hash: [7; 32],
        toolchain_fingerprint: b"determinism-test-toolchain".to_vec(),
        environment_fingerprint: b"determinism-test-environment".to_vec(),
        kind: 9,
        schema: 1,
        alignment: 16,
    }
}

fn specs_for(ids: impl IntoIterator<Item = u128>) -> Specs {
    ids.into_iter().map(|id| (JobId(id), spec(id))).collect()
}

fn counters(ids: impl IntoIterator<Item = u128>) -> CallCounts {
    Arc::new(
        ids.into_iter()
            .map(|id| (JobId(id), AtomicUsize::new(0)))
            .collect(),
    )
}

fn assert_calls(calls: &CallCounts, expected: usize) {
    for (job, count) in calls.iter() {
        assert_eq!(count.load(Ordering::SeqCst), expected, "job {job:?}");
    }
}

/// A reusable two-party barrier whose timeout only guards against hangs.
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

// Roots, fan-out, joins, a diamond and cross-level edges. Dependency order is
// deliberately not JobId order; insertion order is reversed in mixed_graph.
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

fn mixed_graph(overlap: bool) -> (BuildGraph, Specs, CallCounts) {
    let mut graph = BuildGraph::new();
    let calls = counters(MIXED_DAG.iter().map(|(id, _)| *id));
    let expected = Arc::new(mixed_expected());
    let rendezvous = Arc::new(PairRendezvous::default());
    for &(id, dependencies) in MIXED_DAG.iter().rev() {
        let calls = Arc::clone(&calls);
        let expected = Arc::clone(&expected);
        let rendezvous = Arc::clone(&rendezvous);
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
                        // Both runs must receive these exact, ordered bytes.
                        assert_eq!(*input, expected[&JobId(dependency)].as_slice());
                        if index != 0 {
                            output.push(b'|');
                        }
                        output.extend_from_slice(input);
                    }
                    output.push(b')');
                    if overlap && matches!(id, 10 | 20) {
                        rendezvous.wait();
                    }
                    Ok(output)
                },
            ))
            .unwrap();
    }
    let specs = specs_for(MIXED_DAG.iter().map(|(id, _)| *id));
    (graph, specs, calls)
}

fn assert_mixed_artifacts(fixture: &Fixture, execution: &BuildExecution, specs: &Specs) {
    let expected = mixed_expected();
    let manifest = &execution.published.manifest;
    assert!(execution.cache_hits.is_empty());
    assert_eq!(manifest.entries.len(), MIXED_DAG.len());
    assert_eq!(execution.artifact_keys.len(), MIXED_DAG.len());
    assert_eq!(
        fixture.store.current_revision().unwrap(),
        Some(manifest.revision)
    );
    assert_eq!(
        fixture.store.manifest_bytes(manifest.revision).unwrap(),
        manifest.canonical_bytes()
    );
    assert_eq!(
        execution.published.manifest_hash,
        ContentHash::digest(&manifest.canonical_bytes())
    );
    let persisted = DerivedDataCache::open(fixture.ddc.root()).unwrap();
    for &(id, dependencies) in MIXED_DAG {
        let spec = &specs[&JobId(id)];
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
        assert_eq!(persisted.get(key).as_ref(), Some(descriptor));
        assert_eq!(
            fixture.store.cas().read(descriptor.content_hash).unwrap(),
            expected[&JobId(id)]
        );
    }
}

#[test]
fn cold_and_persistent_warm_verified_builds_call_every_node_twice_without_cache_hits() {
    for limit in [1, 2] {
        let mut fixture = Fixture::new();
        let (graph, specs, calls) = mixed_graph(false);
        let cold = fixture.verified(&graph, &specs, 0, limit).unwrap();
        assert_calls(&calls, 2);
        assert_mixed_artifacts(&fixture, &cold, &specs);

        fixture.reopen();
        let warm = fixture.verified(&graph, &specs, 1, limit).unwrap();
        assert_calls(&calls, 4);
        assert_mixed_artifacts(&fixture, &warm, &specs);
        assert_eq!(warm.artifact_keys, cold.artifact_keys);
        assert_eq!(
            warm.published.manifest.entries,
            cold.published.manifest.entries
        );
    }
}

#[test]
fn verified_mixed_dag_serial_and_parallel_have_identical_manifest_cas_and_input_order() {
    let mut serial = Fixture::new();
    let (serial_graph, specs, serial_calls) = mixed_graph(false);
    let reference = serial.verified(&serial_graph, &specs, 0, 1).unwrap();
    assert_calls(&serial_calls, 2);
    assert_mixed_artifacts(&serial, &reference, &specs);

    let mut parallel = Fixture::new();
    let (parallel_graph, parallel_specs, parallel_calls) = mixed_graph(true);
    let execution = parallel
        .verified(&parallel_graph, &parallel_specs, 0, 2)
        .unwrap();
    assert_calls(&parallel_calls, 2);
    assert_mixed_artifacts(&parallel, &execution, &parallel_specs);
    assert_eq!(execution, reference);
    for descriptor in reference.published.manifest.entries.values() {
        assert_eq!(
            parallel.store.cas().read(descriptor.content_hash).unwrap(),
            serial.store.cas().read(descriptor.content_hash).unwrap()
        );
    }
}

fn publish_previous(fixture: &mut Fixture) -> BuildExecution {
    let mut graph = BuildGraph::new();
    graph
        .add_job(BuildJob::new(JobId(900), [], |_: &[&[u8]]| {
            Ok(b"previous determinism-gate revision".to_vec())
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
    let persisted = DerivedDataCache::open(fixture.ddc.root()).unwrap();
    for descriptor in manifest.entries.values() {
        assert_eq!(
            reopened.cas().read(descriptor.content_hash).unwrap(),
            b"previous determinism-gate revision"
        );
        assert_eq!(fixture.ddc.get(descriptor.key).as_ref(), Some(descriptor));
        assert_eq!(persisted.get(descriptor.key).as_ref(), Some(descriptor));
    }
}

fn assert_blob_absent(cas: &ContentAddressedStore, bytes: &[u8]) {
    assert!(matches!(
        cas.read(ContentHash::digest(bytes)),
        Err(CasError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound
    ));
}

#[derive(Clone, Copy)]
enum SecondRun {
    Output(&'static [u8]),
    Error,
    Panic,
}

struct FailurePlan {
    graph: BuildGraph,
    specs: Specs,
    calls: CallCounts,
    failing_key: ArtifactKey,
    checked_before_second_outcome: Arc<AtomicBool>,
}

fn failure_plan(fixture: &Fixture, first: &'static [u8], second: SecondRun) -> FailurePlan {
    let specs = specs_for([1, 2, 10, 20, 30]);
    let root_keys: Vec<_> = [2, 1]
        .map(|id| specs[&JobId(id)].artifact_key(1, &[]).unwrap())
        .to_vec();
    let failing_key = specs[&JobId(10)].artifact_key(1, &root_keys).unwrap();
    let calls = counters([1, 2, 10, 20, 30]);
    let mut graph = BuildGraph::new();
    for (id, bytes) in [
        (1, b"verified root one".as_slice()),
        (2, b"verified root two".as_slice()),
    ] {
        let calls = Arc::clone(&calls);
        graph
            .add_job(BuildJob::new(JobId(id), [], move |inputs: &[&[u8]]| {
                assert!(inputs.is_empty());
                calls[&JobId(id)].fetch_add(1, Ordering::SeqCst);
                Ok(bytes.to_vec())
            }))
            .unwrap();
    }
    let rendezvous = Arc::new(PairRendezvous::default());
    let peer_rendezvous = Arc::clone(&rendezvous);
    let peer_calls = Arc::clone(&calls);
    graph
        .add_job(BuildJob::new(
            JobId(20),
            [JobId(1)],
            move |inputs: &[&[u8]]| {
                assert_eq!(inputs, &[b"verified root one".as_slice()]);
                peer_calls[&JobId(20)].fetch_add(1, Ordering::SeqCst);
                peer_rendezvous.wait();
                Ok(b"verified successful peer".to_vec())
            },
        ))
        .unwrap();
    let checked_before_second_outcome = Arc::new(AtomicBool::new(false));
    let checked = Arc::clone(&checked_before_second_outcome);
    let failed_calls = Arc::clone(&calls);
    let cas = fixture.store.cas().clone();
    let ddc_root = fixture.ddc.root().to_path_buf();
    graph
        .add_job(BuildJob::new(
            JobId(10),
            [JobId(2), JobId(1)],
            move |inputs: &[&[u8]]| {
                assert_eq!(
                    inputs,
                    &[
                        b"verified root two".as_slice(),
                        b"verified root one".as_slice(),
                    ]
                );
                let invocation = failed_calls[&JobId(10)].fetch_add(1, Ordering::SeqCst);
                rendezvous.wait();
                if invocation == 0 {
                    return Ok(first.to_vec());
                }
                assert_eq!(invocation, 1, "a node must run exactly twice");
                // Check before returning the second result, not merely after
                // failure: the first candidate must never be persisted alone.
                assert_blob_absent(&cas, first);
                assert!(DerivedDataCache::open(&ddc_root)
                    .unwrap()
                    .get(failing_key)
                    .is_none());
                // Distinguishes the intentional panic from an assertion panic.
                checked.store(true, Ordering::SeqCst);
                match second {
                    SecondRun::Output(bytes) => Ok(bytes.to_vec()),
                    SecondRun::Error => Err(ProcessorError::new("second verification run failed")),
                    SecondRun::Panic => panic!("second verification run panicked"),
                }
            },
        ))
        .unwrap();
    let descendant_calls = Arc::clone(&calls);
    graph
        .add_job(BuildJob::new(
            JobId(30),
            [JobId(20), JobId(10)],
            move |_: &[&[u8]]| {
                descendant_calls[&JobId(30)].fetch_add(1, Ordering::SeqCst);
                Ok(b"descendant must not execute".to_vec())
            },
        ))
        .unwrap();
    FailurePlan {
        graph,
        specs,
        calls,
        failing_key,
        checked_before_second_outcome,
    }
}

fn assert_failure_state(
    fixture: &Fixture,
    previous: &BuildExecution,
    plan: &FailurePlan,
    rejected_outputs: &[&[u8]],
) {
    for id in [1, 2, 10, 20] {
        assert_eq!(plan.calls[&JobId(id)].load(Ordering::SeqCst), 2);
    }
    assert_eq!(plan.calls[&JobId(30)].load(Ordering::SeqCst), 0);
    assert!(plan.checked_before_second_outcome.load(Ordering::SeqCst));
    assert!(fixture.ddc.get(plan.failing_key).is_none());
    let persisted = DerivedDataCache::open(fixture.ddc.root()).unwrap();
    assert!(persisted.get(plan.failing_key).is_none());
    for bytes in rejected_outputs {
        assert_blob_absent(fixture.store.cas(), bytes);
    }
    assert_blob_absent(fixture.store.cas(), b"descendant must not execute");

    // Successful earlier waves/peers may leave cache side effects. Validate
    // any such entries without requiring rollback or caching of every peer.
    for (id, bytes) in [
        (1, b"verified root one".as_slice()),
        (2, b"verified root two".as_slice()),
        (20, b"verified successful peer".as_slice()),
    ] {
        let input_keys = if id == 20 {
            vec![plan.specs[&JobId(1)].artifact_key(1, &[]).unwrap()]
        } else {
            Vec::new()
        };
        let key = plan.specs[&JobId(id)].artifact_key(1, &input_keys).unwrap();
        assert_eq!(fixture.ddc.get(key), persisted.get(key));
        if let Some(descriptor) = persisted.get(key) {
            assert_eq!(descriptor.content_hash, ContentHash::digest(bytes));
            assert_eq!(
                fixture.store.cas().read(descriptor.content_hash).unwrap(),
                bytes
            );
        }
    }
    assert_previous_preserved(fixture, previous);
}

fn assert_nondeterministic(first: &'static [u8], second: &'static [u8], offset: u64) {
    let mut fixture = Fixture::new();
    let previous = publish_previous(&mut fixture);
    let plan = failure_plan(&fixture, first, SecondRun::Output(second));
    let error = fixture
        .verified(&plan.graph, &plan.specs, 1, 2)
        .unwrap_err();
    match error {
        BuildExecutionError::NonDeterministic(difference) => {
            assert_eq!(difference.job, JobId(10));
            assert_eq!(difference.key, plan.failing_key);
            assert_eq!(difference.first_hash, ContentHash::digest(first));
            assert_eq!(difference.second_hash, ContentHash::digest(second));
            assert_eq!(difference.first_len, first.len() as u64);
            assert_eq!(difference.second_len, second.len() as u64);
            assert_eq!(difference.first_difference, offset);
        }
        error => panic!("expected structured determinism difference, got {error:?}"),
    }
    assert_failure_state(&fixture, &previous, &plan, &[first, second]);
}

#[test]
fn changed_content_reports_exact_byte_offset_and_never_caches_or_publishes_candidates() {
    assert_nondeterministic(b"abc", b"xbc", 0);
    assert_nondeterministic(b"ab\0def", b"ab\xffdef", 2);
    assert_nondeterministic(b"abcd", b"abcX", 3);
}

#[test]
fn changed_lengths_report_common_prefix_or_earlier_byte_difference_and_keep_current() {
    assert_nondeterministic(b"abc", b"abc-more", 3);
    assert_nondeterministic(b"abc-more", b"abc", 3);
    assert_nondeterministic(b"", b"x", 0);
    assert_nondeterministic(b"x", b"", 0);
    // A length mismatch must not hide an earlier differing byte.
    assert_nondeterministic(b"abcdef", b"abX", 2);
}

fn assert_second_run_failure(second: SecondRun) {
    let mut fixture = Fixture::new();
    let previous = publish_previous(&mut fixture);
    let first = b"unverified first candidate";
    let plan = failure_plan(&fixture, first, second);
    let error = fixture
        .verified(&plan.graph, &plan.specs, 1, 2)
        .unwrap_err();
    match error {
        BuildExecutionError::Graph(BuildError::ProcessorFailed { job, source }) => {
            assert_eq!(job, JobId(10));
            match second {
                SecondRun::Error => assert_eq!(source.message(), "second verification run failed"),
                SecondRun::Panic => assert!(source.message().contains("panic")),
                SecondRun::Output(_) => unreachable!(),
            }
        }
        error => panic!("expected second-run processor failure, got {error:?}"),
    }
    assert_failure_state(&fixture, &previous, &plan, &[first]);
}

#[test]
fn second_processor_error_rejects_first_candidate_and_preserves_previous_revision() {
    assert_second_run_failure(SecondRun::Error);
}

#[test]
fn second_processor_panic_is_a_job_failure_without_cache_or_publication() {
    assert_second_run_failure(SecondRun::Panic);
}

#[test]
fn zero_limit_rejects_without_processor_calls_cache_writes_or_publication() {
    let mut fixture = Fixture::new();
    let previous = publish_previous(&mut fixture);
    let index_before = fs::read(fixture.ddc.index_path()).unwrap();
    let (graph, specs, calls) = mixed_graph(false);
    assert!(matches!(
        fixture.verified(&graph, &specs, 1, 0),
        Err(BuildExecutionError::Graph(BuildError::InvalidConcurrency))
    ));
    assert_calls(&calls, 0);
    assert_eq!(fs::read(fixture.ddc.index_path()).unwrap(), index_before);
    for bytes in mixed_expected().values() {
        assert_blob_absent(fixture.store.cas(), bytes);
    }
    assert_previous_preserved(&fixture, &previous);
}
