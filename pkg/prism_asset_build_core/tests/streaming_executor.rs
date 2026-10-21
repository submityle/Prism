//! Real CAS/DDC execution of native streams and private candidate failures.

use prism_asset_build_core::job::{ProcessorInput, StreamingProcessor};
use prism_asset_build_core::{
    ArtifactKey, ArtifactSpec, AssetId, BuildError, BuildExecution, BuildExecutionError,
    BuildExecutor, BuildGraph, BuildJob, BuildRevision, CasError, ContentAddressedStore,
    ContentHash, DerivedDataCache, JobId, ProcessorError, RevisionStore,
};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

const BLOCK: usize = 64 * 1024;
const EXECUTOR: BuildExecutor = BuildExecutor {
    protocol_version: 1,
};

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

    fn read(&self, id: u128) -> Vec<u8> {
        let revision = self.store.current_revision().unwrap().unwrap();
        let manifest = self.store.manifest(revision).unwrap();
        self.store
            .cas()
            .read(manifest.entries[&AssetId(id)].content_hash)
            .unwrap()
    }

    fn assert_unpublished(&self, candidates: &[&[u8]]) {
        assert_eq!(self.store.current_revision().unwrap(), None);
        assert!(self.store.manifest(BuildRevision(0)).is_err());
        assert!(self.ddc.is_empty());
        assert!(DerivedDataCache::open(self.ddc.root()).unwrap().is_empty());
        for bytes in candidates {
            assert!(!self
                .store
                .cas()
                .blob_path(ContentHash::digest(bytes))
                .exists());
        }
    }
}

fn spec(id: u128) -> ArtifactSpec {
    ArtifactSpec {
        asset: AssetId(id),
        processor_id: 0x5354_5245_414d,
        processor_version: 1,
        canonical_settings: id.to_le_bytes().to_vec(),
        profile_hash: [7; 32],
        toolchain_fingerprint: b"streaming-executor-test".to_vec(),
        environment_fingerprint: b"streaming-executor-test-env".to_vec(),
        kind: 11,
        schema: 1,
        alignment: 1,
    }
}

fn specs(ids: impl IntoIterator<Item = u128>) -> BTreeMap<JobId, ArtifactSpec> {
    ids.into_iter().map(|id| (JobId(id), spec(id))).collect()
}

#[derive(Clone, Copy)]
enum Mode {
    Serial,
    Parallel,
    Verified,
}

fn execute(
    fixture: &mut Fixture,
    graph: &BuildGraph,
    specs: &BTreeMap<JobId, ArtifactSpec>,
    revision: u64,
    mode: Mode,
) -> Result<BuildExecution, BuildExecutionError> {
    match mode {
        Mode::Serial => EXECUTOR.execute(
            graph,
            specs,
            &fixture.store,
            &mut fixture.ddc,
            BuildRevision(revision),
        ),
        Mode::Parallel => EXECUTOR.execute_parallel_with_limit(
            graph,
            specs,
            &fixture.store,
            &mut fixture.ddc,
            BuildRevision(revision),
            2,
        ),
        Mode::Verified => EXECUTOR.execute_verified_with_limit(
            graph,
            specs,
            &fixture.store,
            &mut fixture.ddc,
            BuildRevision(revision),
            2,
        ),
    }
}

fn pattern(index: usize) -> u8 {
    ((index * 17 + index / BLOCK) % 251) as u8
}
fn payload(length: usize) -> Vec<u8> {
    (0..length).map(pattern).collect()
}
fn process_io(error: io::Error) -> ProcessorError {
    ProcessorError::new(error.to_string())
}

struct ChunkProducer {
    length: usize,
    calls: Arc<AtomicUsize>,
}

impl StreamingProcessor for ChunkProducer {
    fn execute(
        &self,
        inputs: &mut [ProcessorInput<'_>],
        output: &mut dyn Write,
    ) -> Result<(), ProcessorError> {
        assert!(inputs.is_empty());
        self.calls.fetch_add(1, Ordering::SeqCst);
        let mut chunk = [0; 4093];
        let mut offset = 0;
        while offset < self.length {
            let count = chunk.len().min(self.length - offset);
            for (index, byte) in chunk[..count].iter_mut().enumerate() {
                *byte = pattern(offset + index);
            }
            output.write_all(&chunk[..count]).map_err(process_io)?;
            offset += count;
        }
        Ok(())
    }
}

struct CopyStream {
    calls: Arc<AtomicUsize>,
}

impl StreamingProcessor for CopyStream {
    fn execute(
        &self,
        inputs: &mut [ProcessorInput<'_>],
        output: &mut dyn Write,
    ) -> Result<(), ProcessorError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        assert_eq!(inputs.len(), 1);
        let mut chunk = [0; 8191];
        let mut copied = 0;
        loop {
            let count = inputs[0].read(&mut chunk).map_err(process_io)?;
            if count == 0 {
                break;
            }
            output.write_all(&chunk[..count]).map_err(process_io)?;
            copied += count as u64;
        }
        assert_eq!(copied, inputs[0].byte_len());
        Ok(())
    }
}

fn add_producer(graph: &mut BuildGraph, id: u128, length: usize, calls: &Arc<AtomicUsize>) {
    graph
        .add_job(
            BuildJob::streaming(
                JobId(id),
                [],
                ChunkProducer {
                    length,
                    calls: Arc::clone(calls),
                },
            )
            .with_max_output_bytes(length as u64)
            .with_max_materialized_input_bytes(0),
        )
        .unwrap();
}

#[test]
fn native_chunked_producer_and_copy_cross_blocks_with_zero_materialization_budget() {
    let mut fixture = Fixture::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let length = 3 * BLOCK + 37;
    let mut graph = BuildGraph::new();
    add_producer(&mut graph, 1, length, &calls);
    graph
        .add_job(
            BuildJob::streaming(
                JobId(2),
                [JobId(1)],
                CopyStream {
                    calls: Arc::clone(&calls),
                },
            )
            .with_max_output_bytes(length as u64)
            .with_max_materialized_input_bytes(0),
        )
        .unwrap();
    let result = execute(&mut fixture, &graph, &specs([1, 2]), 0, Mode::Serial).unwrap();
    let expected = payload(length);
    assert_eq!(fixture.read(1), expected);
    assert_eq!(fixture.read(2), expected);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(result.cache_hits.is_empty());
    // These public counters describe coordinator retention, not processor heaps.
    assert_eq!(result.output_memory.peak_retained_bytes, 0);
    assert_eq!(result.output_memory.peak_retained_outputs, 0);
}

#[test]
fn serial_parallel_and_reopened_warm_streaming_results_match() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut graph = BuildGraph::new();
    for (producer, consumer, length) in [(1, 3, 2 * BLOCK + 9), (2, 4, 3 * BLOCK + 7)] {
        add_producer(&mut graph, producer, length, &calls);
        graph
            .add_job(
                BuildJob::streaming(
                    JobId(consumer),
                    [JobId(producer)],
                    CopyStream {
                        calls: Arc::clone(&calls),
                    },
                )
                .with_max_materialized_input_bytes(0),
            )
            .unwrap();
    }
    let specs = specs([1, 2, 3, 4]);
    let mut serial = Fixture::new();
    let mut parallel = Fixture::new();
    let cold = execute(&mut serial, &graph, &specs, 0, Mode::Serial).unwrap();
    let parallel_cold = execute(&mut parallel, &graph, &specs, 0, Mode::Parallel).unwrap();
    assert!(cold.cache_hits.is_empty());
    assert!(parallel_cold.cache_hits.is_empty());
    assert_eq!(cold.artifact_keys, parallel_cold.artifact_keys);
    assert_eq!(cold.published.manifest, parallel_cold.published.manifest);
    assert_eq!(calls.load(Ordering::SeqCst), 8);
    parallel.store = RevisionStore::open(parallel.root.path()).unwrap();
    parallel.ddc = DerivedDataCache::open(parallel.ddc.root()).unwrap();
    let warm = execute(&mut parallel, &graph, &specs, 1, Mode::Parallel).unwrap();
    assert_eq!(
        warm.cache_hits,
        BTreeSet::from([JobId(1), JobId(2), JobId(3), JobId(4)])
    );
    assert_eq!(warm.artifact_keys, cold.artifact_keys);
    assert_eq!(
        warm.published.manifest.entries,
        cold.published.manifest.entries
    );
    assert_eq!(calls.load(Ordering::SeqCst), 8);
    for id in 1..=4 {
        assert_eq!(serial.read(id), parallel.read(id));
    }
}

struct SlotProbe {
    expected: Vec<&'static [u8]>,
    calls: Arc<AtomicUsize>,
}

impl StreamingProcessor for SlotProbe {
    fn execute(
        &self,
        inputs: &mut [ProcessorInput<'_>],
        output: &mut dyn Write,
    ) -> Result<(), ProcessorError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        assert_eq!(inputs.len(), self.expected.len());
        for (input, expected) in inputs.iter_mut().zip(self.expected.iter().copied()) {
            assert_eq!(input.content_hash(), ContentHash::digest(expected));
            assert_eq!(input.byte_len(), expected.len() as u64);
            assert_eq!(input.stream_position().map_err(process_io)?, 0);
            assert_eq!(
                input.seek(SeekFrom::End(-1)).map_err(process_io)?,
                expected.len() as u64 - 1
            );
            let mut last = [0];
            input.read_exact(&mut last).map_err(process_io)?;
            assert_eq!(last[0], expected[expected.len() - 1]);
            assert_eq!(input.read(&mut last).map_err(process_io)?, 0);
            for invalid in [
                SeekFrom::Start(input.byte_len() + 1),
                SeekFrom::Current(1),
                SeekFrom::End(1),
            ] {
                assert_eq!(
                    input.seek(invalid).unwrap_err().kind(),
                    io::ErrorKind::InvalidInput
                );
                assert_eq!(
                    input.stream_position().map_err(process_io)?,
                    input.byte_len()
                );
            }
            input.rewind().map_err(process_io)?;
            assert_eq!(
                input.seek(SeekFrom::Current(-1)).unwrap_err().kind(),
                io::ErrorKind::InvalidInput
            );
            assert_eq!(input.stream_position().map_err(process_io)?, 0);
            let mut bytes = Vec::new();
            input.read_to_end(&mut bytes).map_err(process_io)?;
            assert_eq!(bytes, expected);
            output.write_all(&bytes).map_err(process_io)?;
            output.write_all(b"|").map_err(process_io)?;
        }
        Ok(())
    }
}

fn slot_graph(order: &[u128], calls: &Arc<AtomicUsize>) -> BuildGraph {
    let mut graph = BuildGraph::new();
    for (id, bytes) in [
        (1, b"one".as_slice()),
        (2, b"two".as_slice()),
        (3, b"three".as_slice()),
    ] {
        graph
            .add_job(BuildJob::new(JobId(id), [], move |_: &[&[u8]]| {
                Ok(bytes.to_vec())
            }))
            .unwrap();
    }
    graph
        .add_job(
            BuildJob::streaming(
                JobId(4),
                order.iter().copied().map(JobId).collect::<Vec<_>>(),
                SlotProbe {
                    expected: order
                        .iter()
                        .map(|id| match id {
                            1 => b"one".as_slice(),
                            2 => b"two".as_slice(),
                            3 => b"three".as_slice(),
                            _ => panic!("unknown input slot"),
                        })
                        .collect(),
                    calls: Arc::clone(calls),
                },
            )
            .with_max_materialized_input_bytes(0),
        )
        .unwrap();
    graph
}

#[test]
fn declared_slots_keep_full_key_order_and_publish_unique_dependencies_across_warm_reopens() {
    let mut fixture = Fixture::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let plan = specs([1, 2, 3, 4]);
    let cases: &[(&[u128], &[u128], &[u8])] = &[
        (&[3, 1, 2, 3], &[3, 1, 2], b"three|one|two|three|"),
        // Same unique descriptor dependencies and slot count; duplicate position changes.
        (&[3, 1, 3, 2], &[3, 1, 2], b"three|one|three|two|"),
        // Same unique descriptor dependencies; duplicate count changes.
        (&[3, 1, 2, 3, 3], &[3, 1, 2], b"three|one|two|three|three|"),
        (&[1, 3, 2, 3], &[1, 3, 2], b"one|three|two|three|"),
    ];
    let mut consumer_keys = BTreeSet::new();
    let mut root_keys = None;
    for (index, &(order, unique, expected)) in cases.iter().enumerate() {
        let graph = slot_graph(order, &calls);
        let revision = 2 * index as u64;
        let result = execute(&mut fixture, &graph, &plan, revision, Mode::Parallel).unwrap();
        assert_eq!(fixture.read(4), expected);
        assert_eq!(calls.load(Ordering::SeqCst), index + 1);
        assert_eq!(
            result.cache_hits,
            if index == 0 {
                BTreeSet::new()
            } else {
                BTreeSet::from([JobId(1), JobId(2), JobId(3)])
            }
        );
        let roots = [1, 2, 3].map(|id| result.artifact_keys[&JobId(id)]);
        if let Some(previous) = root_keys {
            assert_eq!(roots, previous);
        }
        root_keys = Some(roots);
        let ordered: Vec<_> = order
            .iter()
            .map(|id| result.artifact_keys[&JobId(*id)])
            .collect();
        let key = result.artifact_keys[&JobId(4)];
        assert_eq!(key, plan[&JobId(4)].artifact_key(1, &ordered).unwrap());
        assert!(
            consumer_keys.insert(key),
            "changed slot order/count must change the key"
        );
        let descriptor = &result.published.manifest.entries[&AssetId(4)];
        let expected_dependencies: Vec<_> = unique
            .iter()
            .map(|id| result.artifact_keys[&JobId(*id)])
            .collect();
        assert_eq!(descriptor.dependencies, expected_dependencies);
        assert_ne!(
            key,
            plan[&JobId(4)]
                .artifact_key(1, &expected_dependencies)
                .unwrap()
        );
        assert_eq!(fixture.ddc.get(key), Some(descriptor.clone()));

        fixture.store = RevisionStore::open(fixture.root.path()).unwrap();
        fixture.ddc = DerivedDataCache::open(fixture.ddc.root()).unwrap();
        assert_eq!(fixture.ddc.get(key), Some(descriptor.clone()));
        let warm = execute(&mut fixture, &graph, &plan, revision + 1, Mode::Parallel).unwrap();
        assert_eq!(
            warm.cache_hits,
            BTreeSet::from([JobId(1), JobId(2), JobId(3), JobId(4)])
        );
        assert_eq!(warm.artifact_keys, result.artifact_keys);
        assert_eq!(
            warm.published.manifest.entries,
            result.published.manifest.entries
        );
        assert_eq!(fixture.read(4), expected);
        assert_eq!(calls.load(Ordering::SeqCst), index + 1);
    }
}

#[test]
fn buffered_unique_hash_budget_is_exact_and_rejection_precedes_callback() {
    let length = BLOCK + 17;
    for disjoint in [false, true] {
        let required = if disjoint {
            length + length + 1
        } else {
            length
        };
        for exact in [false, true] {
            let mut fixture = Fixture::new();
            let calls = Arc::new(AtomicUsize::new(0));
            let producers = Arc::new(AtomicUsize::new(0));
            let mut graph = BuildGraph::new();
            add_producer(&mut graph, 1, length, &producers);
            add_producer(&mut graph, 2, length, &producers);
            let mut dependencies = vec![JobId(2), JobId(1), JobId(2)];
            let mut ids = vec![1, 2, 4];
            if disjoint {
                add_producer(&mut graph, 3, length + 1, &producers);
                dependencies.push(JobId(3));
                ids.push(3);
            }
            let callback_calls = Arc::clone(&calls);
            graph
                .add_job(
                    BuildJob::new(JobId(4), dependencies, move |inputs: &[&[u8]]| {
                        callback_calls.fetch_add(1, Ordering::SeqCst);
                        assert_eq!(inputs.len(), if disjoint { 4 } else { 3 });
                        for bytes in &inputs[..3] {
                            assert_eq!(*bytes, payload(length));
                        }
                        assert_eq!(inputs[0].as_ptr(), inputs[1].as_ptr());
                        assert_eq!(inputs[1].as_ptr(), inputs[2].as_ptr());
                        if disjoint {
                            assert_eq!(inputs[3], payload(length + 1));
                        }
                        Ok(b"buffered result".to_vec())
                    })
                    .with_max_materialized_input_bytes((required - usize::from(!exact)) as u64),
                )
                .unwrap();
            let plan = specs(ids);
            let result = execute(&mut fixture, &graph, &plan, 0, Mode::Parallel);
            if exact {
                let result = result.unwrap();
                assert_eq!(calls.load(Ordering::SeqCst), 1);
                assert_eq!(fixture.read(4), b"buffered result");
                // Roots 1 and 2 have identical bytes but different artifact keys.
                // Descriptor deduplication is by key, preserving first occurrence.
                assert_ne!(
                    result.artifact_keys[&JobId(1)],
                    result.artifact_keys[&JobId(2)]
                );
                assert_eq!(
                    result.published.manifest.entries[&AssetId(1)].content_hash,
                    result.published.manifest.entries[&AssetId(2)].content_hash,
                );
                let mut expected_dependencies =
                    [2, 1].map(|id| result.artifact_keys[&JobId(id)]).to_vec();
                if disjoint {
                    expected_dependencies.push(result.artifact_keys[&JobId(3)]);
                }
                let descriptor = &result.published.manifest.entries[&AssetId(4)];
                assert_eq!(descriptor.dependencies, expected_dependencies);
                let key = result.artifact_keys[&JobId(4)];
                assert_eq!(fixture.ddc.get(key), Some(descriptor.clone()));

                fixture.store = RevisionStore::open(fixture.root.path()).unwrap();
                fixture.ddc = DerivedDataCache::open(fixture.ddc.root()).unwrap();
                assert_eq!(fixture.ddc.get(key), Some(descriptor.clone()));
                let warm = execute(&mut fixture, &graph, &plan, 1, Mode::Parallel).unwrap();
                assert_eq!(
                    warm.cache_hits,
                    result
                        .artifact_keys
                        .keys()
                        .copied()
                        .collect::<BTreeSet<_>>()
                );
                assert_eq!(warm.artifact_keys, result.artifact_keys);
                assert_eq!(
                    warm.published.manifest.entries,
                    result.published.manifest.entries
                );
                assert_eq!(calls.load(Ordering::SeqCst), 1);
                assert_eq!(
                    producers.load(Ordering::SeqCst),
                    if disjoint { 3 } else { 2 }
                );
                assert_eq!(fixture.read(4), b"buffered result");
            } else {
                assert!(
                    matches!(result, Err(BuildExecutionError::Graph(BuildError::ProcessorFailed { job: JobId(4), source }))
                    if source.message() == "processor inputs exceed materialization byte bound")
                );
                assert_eq!(calls.load(Ordering::SeqCst), 0);
                assert_eq!(fixture.store.current_revision().unwrap(), None);
                let mut keys = [2, 1, 2]
                    .map(|id| spec(id).artifact_key(1, &[]).unwrap())
                    .to_vec();
                if disjoint {
                    keys.push(spec(3).artifact_key(1, &[]).unwrap());
                }
                let key = spec(4).artifact_key(1, &keys).unwrap();
                assert!(DerivedDataCache::open(fixture.ddc.root())
                    .unwrap()
                    .get(key)
                    .is_none());
                assert!(!fixture
                    .store
                    .cas()
                    .blob_path(ContentHash::digest(b"buffered result"))
                    .exists());
            }
        }
    }
}

struct IgnoreWriteError;

impl StreamingProcessor for IgnoreWriteError {
    fn execute(
        &self,
        _: &mut [ProcessorInput<'_>],
        output: &mut dyn Write,
    ) -> Result<(), ProcessorError> {
        output.write_all(b"four").map_err(process_io)?;
        assert!(output.write_all(b"overflow").is_err());
        assert!(output.write(b"").is_err());
        assert!(output.write_all(b"x").is_err());
        assert!(output.flush().is_err());
        Ok(()) // Deliberately ignore the writer failure.
    }
}

#[test]
fn ignored_output_limit_error_prevents_prefix_publication_in_all_modes() {
    for mode in [Mode::Serial, Mode::Parallel, Mode::Verified] {
        let mut fixture = Fixture::new();
        let mut graph = BuildGraph::new();
        graph
            .add_job(BuildJob::streaming(JobId(1), [], IgnoreWriteError).with_max_output_bytes(4))
            .unwrap();
        let error = execute(&mut fixture, &graph, &specs([1]), 0, mode).unwrap_err();
        assert!(matches!(error, BuildExecutionError::Cas(CasError::Io(_))));
        fixture.assert_unpublished(&[b"four", b"fouroverflow", b""]);
    }
}

struct CooperativeCancellation {
    cancelled: Arc<AtomicBool>,
    started: mpsc::SyncSender<()>,
    resume: Mutex<mpsc::Receiver<()>>,
}

impl StreamingProcessor for CooperativeCancellation {
    fn execute(
        &self,
        _: &mut [ProcessorInput<'_>],
        output: &mut dyn Write,
    ) -> Result<(), ProcessorError> {
        output
            .write_all(b"cancelled private prefix")
            .map_err(process_io)?;
        self.started.send(()).unwrap();
        self.resume
            .lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(10))
            .unwrap();
        if self.cancelled.load(Ordering::SeqCst) {
            return Err(ProcessorError::new("processor cancelled"));
        }
        Ok(())
    }
}

#[test]
fn cooperative_processor_cancellation_discards_written_prefix() {
    // StreamingProcessor has no executor cancellation token. Cancellation is
    // cooperative here and crosses the public API as an ordinary processor error.
    for mode in [Mode::Serial, Mode::Parallel, Mode::Verified] {
        let mut fixture = Fixture::new();
        let cancelled = Arc::new(AtomicBool::new(false));
        let (started_tx, started_rx) = mpsc::sync_channel(1);
        let (resume_tx, resume_rx) = mpsc::sync_channel(1);
        let mut graph = BuildGraph::new();
        graph
            .add_job(BuildJob::streaming(
                JobId(1),
                [],
                CooperativeCancellation {
                    cancelled: Arc::clone(&cancelled),
                    started: started_tx,
                    resume: Mutex::new(resume_rx),
                },
            ))
            .unwrap();
        let request = Arc::clone(&cancelled);
        let error = std::thread::scope(|scope| {
            let requester = scope.spawn(move || {
                // Request cancellation only after the callback wrote its prefix.
                started_rx.recv_timeout(Duration::from_secs(10)).unwrap();
                request.store(true, Ordering::SeqCst);
                resume_tx.send(()).unwrap();
            });
            let result = execute(&mut fixture, &graph, &specs([1]), 0, mode);
            requester.join().unwrap();
            result.unwrap_err()
        });
        assert!(
            matches!(error, BuildExecutionError::Graph(BuildError::ProcessorFailed { job: JobId(1), source })
            if source.message() == "processor cancelled")
        );
        assert!(cancelled.load(Ordering::SeqCst));
        fixture.assert_unpublished(&[b"cancelled private prefix"]);
    }
}

enum SecondRun {
    Bytes(Vec<u8>),
    Error,
    Panic,
}

struct CandidateProbe {
    first: Vec<u8>,
    second: SecondRun,
    calls: Arc<AtomicUsize>,
    observed_private: Arc<AtomicBool>,
    cas: ContentAddressedStore,
    ddc_root: PathBuf,
    key: ArtifactKey,
}

fn write_chunks(output: &mut dyn Write, bytes: &[u8]) -> Result<(), ProcessorError> {
    for chunk in bytes.chunks(4093) {
        output.write_all(chunk).map_err(process_io)?;
    }
    Ok(())
}

impl StreamingProcessor for CandidateProbe {
    fn execute(
        &self,
        inputs: &mut [ProcessorInput<'_>],
        output: &mut dyn Write,
    ) -> Result<(), ProcessorError> {
        assert!(inputs.is_empty());
        match self.calls.fetch_add(1, Ordering::SeqCst) {
            0 => write_chunks(output, &self.first),
            1 => {
                // Observe the first output while the second callback is running,
                // rather than accepting cleanup after premature publication.
                assert!(!self
                    .cas
                    .blob_path(ContentHash::digest(&self.first))
                    .exists());
                assert!(DerivedDataCache::open(&self.ddc_root)
                    .unwrap()
                    .get(self.key)
                    .is_none());
                self.observed_private.store(true, Ordering::SeqCst);
                match &self.second {
                    SecondRun::Bytes(bytes) => write_chunks(output, bytes),
                    SecondRun::Error => {
                        write_chunks(output, b"second private prefix")?;
                        Err(ProcessorError::new("second streaming execution failed"))
                    }
                    SecondRun::Panic => {
                        write_chunks(output, b"second private prefix")?;
                        panic!("second streaming execution panicked");
                    }
                }
            }
            _ => panic!("determinism check executed more than twice"),
        }
    }
}

fn candidate_graph(
    fixture: &Fixture,
    first: &[u8],
    second: SecondRun,
) -> (BuildGraph, Arc<AtomicUsize>, Arc<AtomicBool>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let observed_private = Arc::new(AtomicBool::new(false));
    let mut graph = BuildGraph::new();
    graph
        .add_job(BuildJob::streaming(
            JobId(1),
            [],
            CandidateProbe {
                first: first.to_vec(),
                second,
                calls: Arc::clone(&calls),
                observed_private: Arc::clone(&observed_private),
                cas: fixture.store.cas().clone(),
                ddc_root: fixture.ddc.root().to_path_buf(),
                key: spec(1).artifact_key(1, &[]).unwrap(),
            },
        ))
        .unwrap();
    (graph, calls, observed_private)
}

fn assert_stream_difference(first: &[u8], second: &[u8], offset: u64) {
    let mut fixture = Fixture::new();
    let (graph, calls, observed_private) =
        candidate_graph(&fixture, first, SecondRun::Bytes(second.to_vec()));
    let error = execute(&mut fixture, &graph, &specs([1]), 0, Mode::Verified).unwrap_err();
    match error {
        BuildExecutionError::NonDeterministic(difference) => {
            assert_eq!(difference.job, JobId(1));
            assert_eq!(difference.key, spec(1).artifact_key(1, &[]).unwrap());
            assert_eq!(difference.first_hash, ContentHash::digest(first));
            assert_eq!(difference.second_hash, ContentHash::digest(second));
            assert_eq!(difference.first_len, first.len() as u64);
            assert_eq!(difference.second_len, second.len() as u64);
            assert_eq!(difference.first_difference, offset);
        }
        other => panic!("expected streaming determinism difference, got {other:?}"),
    }
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(observed_private.load(Ordering::SeqCst));
    fixture.assert_unpublished(&[first, second]);
}

#[test]
fn streaming_determinism_reports_cross_block_difference_without_publishing_either_candidate() {
    let first = payload(3 * BLOCK + 37);
    let mut second = first.clone();
    let offset = 2 * BLOCK + 17;
    second[offset] ^= 0xff;
    assert_stream_difference(&first, &second, offset as u64);
}

#[test]
fn streaming_determinism_reports_prefix_lengths_in_both_directions_including_empty() {
    let prefix = payload(2 * BLOCK + 19);
    let mut extended = prefix.clone();
    extended.extend_from_slice(b"tail");
    assert_stream_difference(&prefix, &extended, prefix.len() as u64);
    assert_stream_difference(&extended, &prefix, prefix.len() as u64);
    assert_stream_difference(b"", b"tail", 0);
    assert_stream_difference(b"tail", b"", 0);
}

#[test]
fn second_streaming_execution_error_or_panic_discards_both_private_outputs() {
    for second in [SecondRun::Error, SecondRun::Panic] {
        let is_panic = matches!(&second, SecondRun::Panic);
        let mut fixture = Fixture::new();
        let first = payload(2 * BLOCK + 11);
        let (graph, calls, observed_private) = candidate_graph(&fixture, &first, second);
        let error = execute(&mut fixture, &graph, &specs([1]), 0, Mode::Verified).unwrap_err();
        match error {
            BuildExecutionError::Graph(BuildError::ProcessorFailed { job, source }) => {
                assert_eq!(job, JobId(1));
                if is_panic {
                    assert!(source.message().contains("panicked"));
                } else {
                    assert_eq!(source.message(), "second streaming execution failed");
                }
            }
            other => panic!("expected processor failure, got {other:?}"),
        }
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert!(observed_private.load(Ordering::SeqCst));
        fixture.assert_unpublished(&[&first, b"second private prefix"]);
    }
}
