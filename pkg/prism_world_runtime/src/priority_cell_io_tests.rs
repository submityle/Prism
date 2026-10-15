use super::*;
use core::future::poll_fn;
use core::task::Waker;
use prism_tasks::{CancelToken, Priority, PriorityCell, TaskPool};
use prism_world_io::{checksum, ChunkedRead, IoError, ReadChunk, ReadProvider};
use prism_world_stream::{AabbI64, GlobalPos};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct Recording {
    starts: Mutex<Vec<(IoRequestId, ReadRequest)>>,
    cancels: Mutex<Vec<IoRequestId>>,
    ready: AtomicBool,
    waker: Mutex<Option<Waker>>,
}

impl Recording {
    fn release(&self) {
        self.ready.store(true, Ordering::Release);
        let waker = self.waker.lock().unwrap().take();
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    fn started_ids(&self) -> Vec<u64> {
        self.starts
            .lock()
            .unwrap()
            .iter()
            .map(|(id, _)| id.get())
            .collect()
    }
}

// Record the real scheduler/provider boundary, including the exact retained
// Arc, rather than substituting a CellIo mock that can bypass admission.
struct RecordingProvider {
    bytes: Arc<[u8]>,
    recording: Arc<Recording>,
}

struct RecordingRead {
    bytes: Arc<[u8]>,
    recording: Arc<Recording>,
    id: IoRequestId,
    cursor: usize,
    end: usize,
    chunk_size: usize,
}

impl ReadProvider for RecordingProvider {
    type Operation = RecordingRead;

    fn start(&self, request: &ReadRequest, id: IoRequestId) -> Result<RecordingRead, IoError> {
        self.recording
            .starts
            .lock()
            .unwrap()
            .push((id, request.clone()));
        let cursor = usize::try_from(request.offset()).unwrap();
        let end = cursor.checked_add(request.length() as usize).unwrap();
        assert!(
            end <= self.bytes.len(),
            "read must use the registered range"
        );
        Ok(RecordingRead {
            bytes: Arc::clone(&self.bytes),
            recording: Arc::clone(&self.recording),
            id,
            cursor,
            end,
            chunk_size: request.chunk_size(),
        })
    }
}

impl ChunkedRead for RecordingRead {
    fn poll_chunk(&mut self, cx: &mut Context<'_>) -> Poll<Result<Option<ReadChunk>, IoError>> {
        if !self.recording.ready.load(Ordering::Acquire) {
            *self.recording.waker.lock().unwrap() = Some(cx.waker().clone());
            if !self.recording.ready.load(Ordering::Acquire) {
                return Poll::Pending;
            }
        }
        if self.cursor == self.end {
            return Poll::Ready(Ok(None));
        }
        let offset = self.cursor;
        self.cursor = self.end.min(self.cursor + self.chunk_size);
        Poll::Ready(Ok(Some(ReadChunk::new(
            offset as u64,
            self.bytes[offset..self.cursor].to_vec(),
        ))))
    }

    fn cancel(&mut self) {
        self.recording.cancels.lock().unwrap().push(self.id);
    }
}

fn archive(payloads: &[&[u8]]) -> (RecordingProvider, Vec<CellReadRequest>, Vec<Vec<u8>>) {
    let mut archive = b"unrelated archive prefix".to_vec();
    let mut sources = Vec::new();
    let mut streams = Vec::new();
    for (index, payload) in payloads.iter().enumerate() {
        let x = index as i64;
        let manifest = CellManifest {
            sector: SectorId::new(0, x, 0, 0),
            bounds: AabbI64::new(GlobalPos::new(x, 0, 0), GlobalPos::new(x + 1, 1, 1)).unwrap(),
            scene_key: index as u128 + 100,
            content_version: 7,
            derived_hash: index as u128 + 200,
            io_bytes: (CELL_HEADER_SIZE + payload.len()) as u64,
        };
        let mut bytes = ManifestHeader::from_manifest(manifest).encode();
        bytes.extend_from_slice(payload);
        sources.push(
            CellReadRequest::new(manifest, 1000)
                .with_byte_offset(archive.len() as u64)
                .with_expected_checksum(checksum(&bytes)),
        );
        archive.extend_from_slice(&bytes);
        archive.extend_from_slice(b"gap");
        streams.push(bytes);
    }
    (
        RecordingProvider {
            bytes: Arc::from(archive),
            recording: Arc::new(Recording {
                ready: AtomicBool::new(true),
                ..Recording::default()
            }),
        },
        sources,
        streams,
    )
}

fn poll_once(
    io: &mut impl CellIo<Error = IoError>,
) -> Poll<Result<IoCompletion<IoError>, IoError>> {
    io.poll_completion(&mut Context::from_waker(Waker::noop()))
}

// Synchronous in-memory chunks need no wall-clock timeout or worker race.
fn next(io: &mut impl CellIo<Error = IoError>) -> IoCompletion<IoError> {
    for _ in 0..128 {
        match poll_once(io) {
            Poll::Ready(result) => return result.expect("no internal adapter failure"),
            Poll::Pending => {}
        }
    }
    panic!("released memory provider must reach a terminal within bounded polls");
}

fn assert_completed(
    completion: IoCompletion<IoError>,
    id: RequestId,
    request: CellReadRequest,
    expected: &[u8],
) {
    match completion {
        IoCompletion::Completed {
            request_id,
            request: actual,
            bytes,
        } => {
            assert_eq!(request_id, id);
            assert_eq!(actual, request);
            assert_eq!(bytes, expected);
        }
        other => panic!("expected a completed read, got {other:?}"),
    }
}

fn assert_recorded(actual: &ReadRequest, source: CellReadRequest, priority: &Arc<PriorityCell>) {
    let expected = source.io_request();
    assert_eq!(actual.offset(), expected.offset());
    assert_eq!(actual.length(), expected.length());
    assert_eq!(actual.expected_checksum(), expected.expected_checksum());
    assert_eq!(actual.priority(), expected.priority());
    assert_eq!(actual.chunk_size(), 7);
    assert!(Arc::ptr_eq(actual.shared_priority().unwrap(), priority));
}

#[test]
fn routed_shared_priority_rejects_every_stale_manifest_field_before_admission() {
    let (provider, sources, streams) = archive(&[b"routed opaque payload"]);
    let recording = Arc::clone(&provider.recording);
    let source = sources[0];
    let manifest = source.manifest();
    let priority = Arc::new(PriorityCell::new(Priority::Low));
    let mut io = RoutedCellIo::new(ScheduledCellIo::new(provider, 1, 1, 7).unwrap());
    io.register(source).unwrap();

    let mut stale = [manifest; 6];
    stale[0].sector = SectorId::new(0, 99, 0, 0);
    stale[1].bounds = AabbI64::new(GlobalPos::new(0, 0, 0), GlobalPos::new(9, 9, 9)).unwrap();
    stale[2].scene_key += 1;
    stale[3].content_version += 1;
    stale[4].derived_hash += 1;
    stale[5].io_bytes += 1;
    for (field, manifest) in stale.into_iter().enumerate() {
        assert!(
            matches!(
                io.submit_with_priority(CellReadRequest::new(manifest, 29), Arc::clone(&priority)),
                Err(SubmitError::Provider { .. })
            ),
            "stale manifest field {field} must not be admitted"
        );
        assert_eq!(io.in_flight(), 0);
        assert!(recording.starts.lock().unwrap().is_empty());
    }

    // Routing owns the range/checksum; the caller owns the static stream score.
    let caller = CellReadRequest::new(manifest, 29)
        .with_byte_offset(u64::MAX)
        .with_expected_checksum(0);
    let expected = CellReadRequest {
        priority: 29,
        ..source
    };
    let id = io
        .submit_with_priority(caller, Arc::clone(&priority))
        .unwrap();
    assert_eq!(id.get(), 1, "rejections must not consume identities");
    assert!(recording.starts.lock().unwrap().is_empty());
    // Re-registering a new range cannot rewrite an already admitted request.
    io.register(source.with_byte_offset(u64::MAX - source.manifest().io_bytes))
        .unwrap();
    assert_completed(next(&mut io), id, expected, &streams[0]);
    let starts = recording.starts.lock().unwrap();
    assert_eq!(starts.len(), 1);
    assert_recorded(&starts[0].1, expected, &priority);
    assert_eq!(io.in_flight(), 0);
    assert!(poll_once(&mut io).is_pending());
}

#[test]
fn live_boost_controls_next_admission_then_static_score_and_fifo_without_preemption() {
    for boost in [false, true] {
        let (provider, sources, streams) = archive(&[b"a", b"b", b"c", b"d", b"e"]);
        let recording = Arc::clone(&provider.recording);
        recording.ready.store(false, Ordering::Release);
        let priority = Arc::new(PriorityCell::new(Priority::Background));
        let mut io = ScheduledCellIo::new(provider, 1, 5, 7).unwrap();
        let active = io.submit(sources[0]).unwrap();
        assert!(poll_once(&mut io).is_pending());
        let scores = [5, 50, 5];
        let mut requests = vec![sources[0]];
        let mut ids = vec![active];
        for (source, score) in sources[1..4].iter().zip(scores) {
            let request = CellReadRequest {
                priority: score,
                ..*source
            };
            ids.push(
                io.submit_with_priority(request, Arc::clone(&priority))
                    .unwrap(),
            );
            requests.push(request);
        }
        let ordinary = CellReadRequest {
            priority: u32::MAX,
            ..sources[4]
        };
        ids.push(io.submit(ordinary).unwrap());
        requests.push(ordinary);
        if boost {
            priority.boost_to(Priority::Critical);
            priority.boost_to(Priority::Low);
            assert_eq!(priority.get(), Priority::Critical);
        }
        assert!(poll_once(&mut io).is_pending());
        assert_eq!(
            recording.started_ids(),
            [active.get()],
            "active reads are not preempted"
        );
        recording.release();
        let order = if boost {
            [0, 2, 1, 3, 4]
        } else {
            [0, 4, 2, 1, 3]
        };
        for index in order {
            assert_completed(next(&mut io), ids[index], requests[index], &streams[index]);
        }
        assert_eq!(recording.started_ids(), order.map(|index| ids[index].get()));
        let starts = recording.starts.lock().unwrap();
        for (id, request) in starts.iter() {
            let index = ids
                .iter()
                .position(|candidate| candidate.get() == id.get())
                .unwrap();
            if (1..4).contains(&index) {
                assert_recorded(request, requests[index], &priority);
            } else {
                assert!(request.shared_priority().is_none());
            }
        }
        assert_eq!(io.in_flight(), 0);
        assert!(poll_once(&mut io).is_pending());
    }
}

#[test]
fn wrapper_retains_default_cell_explicit_override_and_pending_requests_into_inner() {
    let (provider, sources, streams) = archive(&[b"default", b"override"]);
    let recording = Arc::clone(&provider.recording);
    let default = Arc::new(PriorityCell::new(Priority::Background));
    let explicit = Arc::new(PriorityCell::new(Priority::High));
    let mut io = PrioritizedCellIo::new(
        ScheduledCellIo::new(provider, 1, 2, 7).unwrap(),
        Arc::clone(&default),
    );
    assert!(Arc::ptr_eq(io.priority(), &default));
    let first = io.submit(sources[0]).unwrap();
    let second = io
        .submit_with_priority(sources[1], Arc::clone(&explicit))
        .unwrap();
    assert!(Arc::ptr_eq(io.priority(), &default));
    assert_eq!(io.in_flight(), 2);
    let mut io = io.into_inner();
    assert_eq!(io.in_flight(), 2);
    assert_completed(next(&mut io), second, sources[1], &streams[1]);
    assert_completed(next(&mut io), first, sources[0], &streams[0]);
    let starts = recording.starts.lock().unwrap();
    assert_eq!(starts.len(), 2);
    assert_recorded(&starts[0].1, sources[1], &explicit);
    assert_recorded(&starts[1].1, sources[0], &default);
    assert_eq!(io.in_flight(), 0);
}

#[test]
fn task_shared_priority_cancellation_holds_count_and_byte_quotas_until_terminal() {
    // Capacity two exercises count pressure; capacity three isolates byte pressure.
    for capacity in [2, 3] {
        let (provider, sources, streams) = archive(&[b"a", b"b", b"c"]);
        let recording = Arc::clone(&provider.recording);
        recording.ready.store(false, Ordering::Release);
        let bytes = sources[0].manifest().io_bytes;
        let backend = ScheduledCellIo::with_byte_limits(
            provider,
            1,
            capacity,
            7,
            IoByteLimits {
                max_request_bytes: bytes,
                max_total_bytes: 2 * bytes,
            },
        )
        .unwrap();
        let parent = CancelToken::new();
        let priority = Arc::new(PriorityCell::new(Priority::Normal));
        let mut io = TaskCellIo::new(backend, &parent).ok().unwrap();
        let first = io
            .submit_with_priority(sources[0], Arc::clone(&priority))
            .unwrap();
        let second = io
            .submit_with_priority(sources[1], Arc::clone(&priority))
            .unwrap();
        assert!(poll_once(&mut io).is_pending());
        for cancelled in [false, true] {
            if cancelled {
                assert!(io.cancel(first));
                assert!(!io.cancel(first));
            }
            let result = io.submit_with_priority(sources[2], Arc::clone(&priority));
            if capacity == 2 {
                assert_eq!(result, Err(SubmitError::Backpressure { capacity }));
            } else {
                assert!(matches!(result, Err(SubmitError::Provider { .. })));
            }
            assert_eq!(io.in_flight(), 2);
        }
        let terminal = next(&mut io);
        assert!(
            matches!(terminal, IoCompletion::Cancelled { request_id, request }
            if request_id == first && request == sources[0])
        );
        assert_eq!(io.in_flight(), 1);
        assert!(!io.cancel(first));
        let third = io
            .submit_with_priority(sources[2], Arc::clone(&priority))
            .unwrap();
        assert_eq!(
            third.get(),
            second.get() + 1,
            "rejections must not consume IDs"
        );
        priority.boost_to(Priority::High);
        recording.release();
        assert_completed(next(&mut io), second, sources[1], &streams[1]);
        assert_completed(next(&mut io), third, sources[2], &streams[2]);
        assert_eq!(io.in_flight(), 0);
        assert!(!parent.is_cancelled());
        assert!(!io.cancel_token().is_cancelled());
        assert!(poll_once(&mut io).is_pending());
        assert_eq!(recording.cancels.lock().unwrap().len(), 1);
        let starts = recording.starts.lock().unwrap();
        assert_eq!(starts.len(), 3);
        for ((_, actual), source) in starts.iter().zip(&sources) {
            assert_recorded(actual, *source, &priority);
        }
    }
}

#[test]
fn task_scopes_share_urgency_but_not_child_cancellation_and_retire_parent_cancellation_once() {
    let (provider, sources, _) = archive(&[b"cancelled child"]);
    let (sibling_provider, sibling_sources, sibling_streams) = archive(&[b"live sibling"]);
    let recording = Arc::clone(&provider.recording);
    let sibling_recording = Arc::clone(&sibling_provider.recording);
    recording.ready.store(false, Ordering::Release);
    let parent = CancelToken::new();
    let priority = Arc::new(PriorityCell::new(Priority::Background));
    let backend = ScheduledCellIo::new(provider, 1, 2, 7).unwrap();
    let mut io = TaskCellIo::new(backend, &parent).ok().unwrap();
    let sibling_backend = ScheduledCellIo::new(sibling_provider, 1, 1, 7).unwrap();
    let mut sibling = TaskCellIo::new(sibling_backend, &parent).ok().unwrap();
    let first = io
        .submit_with_priority(sources[0], Arc::clone(&priority))
        .unwrap();
    assert!(poll_once(&mut io).is_pending());
    let queued = io
        .submit_with_priority(sources[0], Arc::clone(&priority))
        .unwrap();
    let live = sibling
        .submit_with_priority(sibling_sources[0], Arc::clone(&priority))
        .unwrap();
    priority.boost_to(Priority::Critical);
    io.cancel_token().cancel();
    assert!(!parent.is_cancelled());
    assert!(!sibling.cancel_token().is_cancelled());
    assert_eq!(io.in_flight(), 2);
    let mut retired = BTreeSet::new();
    for remaining in [1, 0] {
        let terminal = next(&mut io);
        assert!(matches!(terminal, IoCompletion::Cancelled { .. }));
        assert!(
            retired.insert(terminal.request_id()),
            "one terminal per request"
        );
        assert_eq!(io.in_flight(), remaining);
    }
    assert_eq!(retired, BTreeSet::from([first, queued]));
    assert_eq!(io.submit(sources[0]), Err(SubmitError::Cancelled));
    assert_eq!(
        io.submit_with_priority(sources[0], Arc::clone(&priority)),
        Err(SubmitError::Cancelled)
    );
    assert!(poll_once(&mut io).is_pending());
    assert_eq!(
        recording.started_ids(),
        [first.get()],
        "queued cancellation must not start IO"
    );
    assert_recorded(
        &recording.starts.lock().unwrap()[0].1,
        sources[0],
        &priority,
    );

    assert_completed(
        next(&mut sibling),
        live,
        sibling_sources[0],
        &sibling_streams[0],
    );
    assert_recorded(
        &sibling_recording.starts.lock().unwrap()[0].1,
        sibling_sources[0],
        &priority,
    );
    let last = sibling
        .submit_with_priority(sibling_sources[0], Arc::clone(&priority))
        .unwrap();
    parent.cancel();
    assert_eq!(sibling.in_flight(), 1);
    assert!(
        matches!(next(&mut sibling), IoCompletion::Cancelled { request_id, .. } if request_id == last)
    );
    assert_eq!(sibling.in_flight(), 0);
    assert!(!sibling.cancel(last));
    assert_eq!(
        sibling.submit(sibling_sources[0]),
        Err(SubmitError::Cancelled)
    );
    assert!(poll_once(&mut sibling).is_pending());
    assert_eq!(sibling_recording.started_ids(), [live.get()]);
}

#[derive(Default)]
struct Host {
    loaded: Vec<(CellManifest, Vec<u8>)>,
    evicted: Vec<SectorId>,
}

impl SceneHost for Host {
    type Error = ();

    fn load_cell(&mut self, manifest: &CellManifest, payload: &[u8]) -> Result<(), ()> {
        self.loaded.push((*manifest, payload.to_vec()));
        Ok(())
    }

    fn evict_cell(&mut self, sector: SectorId) -> Result<(), ()> {
        self.evicted.push(sector);
        Ok(())
    }
}

#[test]
fn prioritized_task_future_drives_wrapper_through_frame_coordinator_to_scene_host() {
    let payload = vec![37; 200];
    let (provider, sources, _) = archive(&[&payload]);
    let recording = Arc::clone(&provider.recording);
    let source = sources[0];
    let manifest = source.manifest();
    let priority = Arc::new(PriorityCell::new(Priority::Background));
    let parent = CancelToken::new();
    let mut routed = RoutedCellIo::new(ScheduledCellIo::new(provider, 1, 1, 7).unwrap());
    routed.register(source).unwrap();
    let scoped = TaskCellIo::new(routed, &parent).ok().unwrap();
    let io = PrioritizedCellIo::new(scoped, Arc::clone(&priority));
    let task_priority = Arc::clone(io.priority());
    assert!(Arc::ptr_eq(&task_priority, &priority));
    let pool = TaskPool::with_threads(0);
    let task = pool.spawn_future_with_priority(task_priority, async move {
        let mut coordinator = CellFrameCoordinator::with_io(io);
        let mut host = Host::default();
        let report = coordinator.plan_frame(
            &[DemandSource::new(manifest.sector, 0, 1, 23).unwrap()],
            &[manifest],
            &[],
            StreamBudget {
                io_bytes: manifest.io_bytes,
                max_actions: 1,
            },
            &mut host,
        );
        assert_eq!(report.submitted, 1);
        assert_eq!(report.failed, 0);
        assert_eq!(coordinator.in_flight(), 1);
        assert!(host.loaded.is_empty());
        let mut polls = 0;
        poll_fn(|cx| {
            polls += 1;
            assert!(polls <= 128, "bounded memory read must retire");
            coordinator.poll_io_with_context(cx, &mut host)
        })
        .await
        .unwrap();
        assert!(
            polls > 1,
            "small chunks exercise cooperative future continuation"
        );
        assert_eq!(coordinator.in_flight(), 0);
        assert_eq!(host.loaded, [(manifest, payload)]);
        assert_eq!(coordinator.trace().replay(), Ok(1));

        let report = coordinator.plan_frame(
            &[],
            &[manifest],
            &[LoadedCell::new(manifest)],
            StreamBudget {
                io_bytes: manifest.io_bytes,
                max_actions: 1,
            },
            &mut host,
        );
        assert_eq!(report.actions, 1);
        assert_eq!(host.evicted, [manifest.sector]);
        assert_eq!(coordinator.trace().replay(), Ok(2));
    });
    priority.boost_to(Priority::Critical);
    pool.block_on(task);
    assert!(!parent.is_cancelled());
    let starts = recording.starts.lock().unwrap();
    assert_eq!(starts.len(), 1);
    assert_recorded(
        &starts[0].1,
        CellReadRequest {
            priority: 23,
            ..source
        },
        &priority,
    );
    assert_eq!(starts[0].1.effective_lane(), Priority::Critical);
}
