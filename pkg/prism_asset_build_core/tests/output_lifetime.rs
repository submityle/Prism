use prism_asset_build_core::{
    ArtifactSpec, AssetId, BuildExecution, BuildExecutor, BuildGraph, BuildJob, BuildRevision,
    ContentHash, DerivedDataCache, JobId, RevisionStore,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;

const EXECUTOR: BuildExecutor = BuildExecutor {
    protocol_version: 1,
};

type Specs = BTreeMap<JobId, ArtifactSpec>;
type Calls = Arc<BTreeMap<JobId, AtomicUsize>>;

struct Fixture {
    store: RevisionStore,
    ddc: DerivedDataCache,
    root: tempfile::TempDir,
}

impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let store = RevisionStore::open(root.path()).unwrap();
        let ddc = DerivedDataCache::open(root.path().join("ddc")).unwrap();
        Self { store, ddc, root }
    }

    fn execute(
        &mut self,
        graph: &BuildGraph,
        specs: &Specs,
        revision: u64,
        max_concurrency: usize,
    ) -> BuildExecution {
        EXECUTOR
            .execute_parallel_with_limit(
                graph,
                specs,
                &self.store,
                &mut self.ddc,
                BuildRevision(revision),
                max_concurrency,
            )
            .unwrap()
    }
}

#[derive(Clone)]
struct Node {
    id: u128,
    dependencies: Vec<u128>,
    byte_len: usize,
}

impl Node {
    fn new(id: u128, dependencies: impl Into<Vec<u128>>, byte_len: usize) -> Self {
        Self {
            id,
            dependencies: dependencies.into(),
            byte_len,
        }
    }

    fn payload(&self) -> Vec<u8> {
        vec![u8::try_from(self.id).unwrap(); self.byte_len]
    }
}

fn specs_for(nodes: &[Node]) -> Specs {
    nodes
        .iter()
        .map(|node| {
            let mut settings = node.id.to_le_bytes().to_vec();
            settings.extend_from_slice(&(node.byte_len as u64).to_le_bytes());
            (
                JobId(node.id),
                ArtifactSpec {
                    asset: AssetId(10_000 + node.id),
                    processor_id: 73,
                    processor_version: 1,
                    canonical_settings: settings,
                    profile_hash: [7; 32],
                    toolchain_fingerprint: b"output-lifetime-toolchain".to_vec(),
                    environment_fingerprint: b"output-lifetime-environment".to_vec(),
                    kind: 19,
                    schema: 1,
                    alignment: 16,
                },
            )
        })
        .collect()
}

fn calls_for(nodes: &[Node]) -> Calls {
    Arc::new(
        nodes
            .iter()
            .map(|node| (JobId(node.id), AtomicUsize::new(0)))
            .collect(),
    )
}

fn graph_for(nodes: &[Node], calls: &Calls) -> BuildGraph {
    graph_with_hook(nodes, calls, |_| {})
}

fn graph_with_hook(
    nodes: &[Node],
    calls: &Calls,
    on_job: impl Fn(u128) + Send + Sync + 'static,
) -> BuildGraph {
    let on_job = Arc::new(on_job);
    let mut graph = BuildGraph::new();
    for node in nodes.iter().rev().cloned() {
        let input_contracts: Vec<_> = node
            .dependencies
            .iter()
            .map(|id| nodes.iter().find(|input| input.id == *id).unwrap().clone())
            .collect();
        let calls = Arc::clone(calls);
        let on_job = Arc::clone(&on_job);
        graph
            .add_job(BuildJob::new(
                JobId(node.id),
                node.dependencies
                    .iter()
                    .copied()
                    .map(JobId)
                    .collect::<Vec<_>>(),
                move |inputs: &[&[u8]]| {
                    calls[&JobId(node.id)].fetch_add(1, Ordering::SeqCst);
                    assert_eq!(inputs.len(), input_contracts.len());
                    for (input, contract) in inputs.iter().zip(&input_contracts) {
                        assert_eq!(input.len(), contract.byte_len);
                        assert!(input
                            .iter()
                            .all(|byte| *byte == u8::try_from(contract.id).unwrap()));
                    }
                    on_job(node.id);
                    Ok(node.payload())
                },
            ))
            .unwrap();
    }
    graph
}

fn assert_calls(calls: &Calls, expected: usize) {
    for (job, calls) in calls.iter() {
        assert_eq!(calls.load(Ordering::SeqCst), expected, "job {}", job.0);
    }
}

fn assert_published(fixture: &Fixture, execution: &BuildExecution, nodes: &[Node]) {
    let manifest = &execution.published.manifest;
    assert_eq!(manifest.entries.len(), nodes.len());
    assert_eq!(execution.artifact_keys.len(), nodes.len());
    assert_eq!(
        fixture.store.current_revision().unwrap(),
        Some(manifest.revision)
    );
    assert_eq!(
        fixture.store.manifest(manifest.revision).unwrap(),
        *manifest
    );
    assert_eq!(
        execution.published.manifest_hash,
        ContentHash::digest(&manifest.canonical_bytes())
    );
    for node in nodes {
        let descriptor = &manifest.entries[&AssetId(10_000 + node.id)];
        let payload = node.payload();
        assert_eq!(descriptor.key, execution.artifact_keys[&JobId(node.id)]);
        assert_eq!(descriptor.byte_len, node.byte_len as u64);
        assert_eq!(descriptor.content_hash, ContentHash::digest(&payload));
        assert_eq!(
            descriptor.dependencies,
            node.dependencies
                .iter()
                .map(|id| execution.artifact_keys[&JobId(*id)])
                .collect::<Vec<_>>()
        );
        assert_eq!(
            fixture.store.cas().read(descriptor.content_hash).unwrap(),
            payload
        );
    }
}

fn assert_retained(execution: &BuildExecution, bytes: u64, outputs: usize) {
    // These counters sample only the outputs needed by future consumers after
    // each wave. Worker buffers, inputs and allocator overhead are excluded.
    assert_eq!(execution.output_memory.peak_retained_bytes, bytes);
    assert_eq!(execution.output_memory.peak_retained_outputs, outputs);
}

fn fanout_nodes() -> Vec<Node> {
    vec![
        Node::new(1, [], 11),
        Node::new(2, [1], 23),
        Node::new(3, [2], 1009),
        Node::new(4, [2, 1], 31),
        Node::new(5, [], 1019),
        Node::new(6, [4], 1021),
    ]
}

#[test]
fn long_chain_retains_only_the_output_for_the_next_consumer() {
    const JOBS: usize = 24;
    let nodes: Vec<_> = (1..=JOBS)
        .map(|id| {
            let dependencies = if id == 1 {
                vec![]
            } else {
                vec![(id - 1) as u128]
            };
            Node::new(id as u128, dependencies, 64 + id)
        })
        .collect();
    for limit in [1, 4] {
        let mut fixture = Fixture::new();
        let calls = calls_for(&nodes);
        let execution = fixture.execute(&graph_for(&nodes, &calls), &specs_for(&nodes), 0, limit);

        // The final, largest output is a sink and is never retained.
        assert_retained(&execution, (64 + JOBS - 1) as u64, 1);
        assert!(execution.cache_hits.is_empty());
        assert_calls(&calls, 1);
        assert_published(&fixture, &execution, &nodes);
    }
}

#[test]
fn fanout_keeps_shared_inputs_until_the_last_consumer_wave() {
    let nodes = fanout_nodes();
    for limit in [1, 2] {
        let mut fixture = Fixture::new();
        let calls = calls_for(&nodes);
        let execution = fixture.execute(&graph_for(&nodes, &calls), &specs_for(&nodes), 0, limit);

        // With one worker the frontier is {1}, {1,2}, {1,2}, {4}, {4}, {}.
        // With two workers jobs 3 and 4 share the final consumer wave for 1/2.
        // Keeping either old input alongside output 4 exceeds the byte peak.
        assert_retained(&execution, 11 + 23, 2);
        assert_calls(&calls, 1);
        assert_published(&fixture, &execution, &nodes);
    }
}

#[test]
fn independent_sinks_publish_without_retaining_any_outputs() {
    let nodes: Vec<_> = (1..=12)
        .map(|id| Node::new(id, [], 1024 + id as usize * 13))
        .collect();
    let mut fixture = Fixture::new();
    let calls = calls_for(&nodes);
    let execution = fixture.execute(&graph_for(&nodes, &calls), &specs_for(&nodes), 0, 3);

    assert_retained(&execution, 0, 0);
    assert_calls(&calls, 1);
    assert_published(&fixture, &execution, &nodes);
}

#[test]
fn persisted_ddc_hits_preserve_the_frontier_bound_and_publish_every_output() {
    let nodes = fanout_nodes();
    let specs = specs_for(&nodes);
    let calls = calls_for(&nodes);
    let graph = graph_for(&nodes, &calls);
    let mut fixture = Fixture::new();
    let cold = fixture.execute(&graph, &specs, 0, 1);
    assert!(cold.cache_hits.is_empty());
    assert_retained(&cold, 11 + 23, 2);
    assert_published(&fixture, &cold, &nodes);

    fixture.ddc = DerivedDataCache::open(fixture.root.path().join("ddc")).unwrap();
    let warm = fixture.execute(&graph, &specs, 1, 1);
    assert_eq!(
        warm.cache_hits,
        nodes
            .iter()
            .map(|node| JobId(node.id))
            .collect::<BTreeSet<_>>()
    );
    assert_eq!(warm.artifact_keys, cold.artifact_keys);
    assert_eq!(
        warm.published.manifest.entries,
        cold.published.manifest.entries
    );
    assert_retained(&warm, 11 + 23, 2);
    assert_calls(&calls, 1);
    assert_published(&fixture, &warm, &nodes);
}

#[test]
fn processor_sweep_from_an_independent_store_defers_until_build_outputs_are_published() {
    let (completed, completion) = mpsc::sync_channel(1);
    let worker = thread::spawn(move || {
        check_active_build_sweep();
        completed.send(()).unwrap();
    });
    // A blocking exclusive-lock regression must fail instead of hanging the
    // harness. The worker owns the fixture throughout the timeout path.
    completion
        .recv_timeout(Duration::from_secs(10))
        .expect("processor sweep blocked on the active build or the build failed");
    worker.join().unwrap();
}

fn check_active_build_sweep() {
    let nodes = vec![
        Node::new(1, [], 17),
        Node::new(2, [1], 29),
        Node::new(3, [2], 1024),
    ];
    let mut fixture = Fixture::new();
    let independent_store = RevisionStore::open(fixture.root.path()).unwrap();
    let processor_store = independent_store.clone();
    let root_payload = nodes[0].payload();
    let root_hash = ContentHash::digest(&root_payload);
    let orphan_payload = b"unpublished output-lifetime orphan";
    let orphan_hash = fixture.store.cas().put(orphan_payload).unwrap();
    let sweeps = Arc::new(AtomicUsize::new(0));
    let processor_sweeps = Arc::clone(&sweeps);
    let calls = calls_for(&nodes);
    let graph = graph_with_hook(&nodes, &calls, move |id| {
        if id != 2 {
            return;
        }
        // The dependency forced job 1 to finish and persist before this sweep.
        assert_eq!(processor_store.cas().read(root_hash).unwrap(), root_payload);
        let report = processor_store
            .sweep_unreachable_cas(Duration::ZERO)
            .unwrap();
        assert!(report.deferred_due_to_active_build);
        assert_eq!(report.deleted, 0);
        assert_eq!(processor_store.cas().read(root_hash).unwrap(), root_payload);
        assert_eq!(
            processor_store.cas().read(orphan_hash).unwrap(),
            orphan_payload
        );
        processor_sweeps.fetch_add(1, Ordering::SeqCst);
    });

    let execution = fixture.execute(&graph, &specs_for(&nodes), 0, 1);
    assert_eq!(sweeps.load(Ordering::SeqCst), 1);
    assert_retained(&execution, 29, 1);
    assert_calls(&calls, 1);
    assert_published(&fixture, &execution, &nodes);

    // Completion must release the build guard. Only the orphan is collectible;
    // all graph outputs, including released intermediates, are committed roots.
    let report = independent_store
        .sweep_unreachable_cas(Duration::ZERO)
        .unwrap();
    assert!(!report.deferred_due_to_active_build);
    assert!(!report.deferred_due_to_pending_transaction);
    assert_eq!(report.deleted, 1);
    assert!(!independent_store.cas().blob_path(orphan_hash).exists());
    assert_published(&fixture, &execution, &nodes);
}
