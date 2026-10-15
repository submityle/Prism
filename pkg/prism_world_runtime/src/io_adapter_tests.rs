use super::tests::{context, manifest, MemoryProvider};
use super::*;
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Duration;

const TIMEOUT: Duration = Duration::from_secs(5);

struct Gate {
    released: AtomicBool,
    // Deliberately retain only the most recent scheduler waker, just as
    // providers that support one polling consumer commonly do.
    waker: Mutex<Option<Waker>>,
    entered: Sender<IoRequestId>,
}

impl Gate {
    fn new() -> (Arc<Self>, Receiver<IoRequestId>) {
        let (entered, receiver) = mpsc::channel();
        (
            Arc::new(Self {
                released: AtomicBool::new(false),
                waker: Mutex::new(None),
                entered,
            }),
            receiver,
        )
    }

    fn release(&self) {
        self.released.store(true, Ordering::Release);
        let waker = self.waker.lock().expect("gate lock").take();
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

struct GateOperation {
    gate: Arc<Gate>,
    id: IoRequestId,
    offset: u64,
    announced: bool,
    delivered: bool,
}

impl ChunkedRead for GateOperation {
    fn poll_chunk(&mut self, cx: &mut Context<'_>) -> Poll<Result<Option<ReadChunk>, IoError>> {
        if !self.gate.released.load(Ordering::Acquire) {
            *self.gate.waker.lock().expect("gate lock") = Some(cx.waker().clone());
            if !self.announced {
                self.announced = true;
                self.gate.entered.send(self.id).expect("announce poll");
            }
            // Close the release/register race without blocking under adapter lock.
            if !self.gate.released.load(Ordering::Acquire) {
                return Poll::Pending;
            }
        }
        if self.delivered {
            Poll::Ready(Ok(None))
        } else {
            self.delivered = true;
            Poll::Ready(Ok(Some(ReadChunk::new(
                self.offset,
                vec![self.offset as u8],
            ))))
        }
    }

    fn cancel(&mut self) {}
}

struct GateProvider(Arc<Gate>);

impl ReadProvider for GateProvider {
    type Operation = GateOperation;

    fn start(&self, request: &ReadRequest, id: IoRequestId) -> Result<Self::Operation, IoError> {
        Ok(GateOperation {
            gate: Arc::clone(&self.0),
            id,
            offset: request.offset(),
            announced: false,
            delivered: false,
        })
    }
}

#[test]
fn simultaneous_blocking_readers_receive_only_their_own_pending_results() {
    let (gate, entered) = Gate::new();
    let adapter =
        Arc::new(IoSchedulerAdapter::new(GateProvider(Arc::clone(&gate)), 2, 2).expect("adapter"));
    let (sender, results) = mpsc::channel();
    let threads: Vec<_> = [17, 29]
        .into_iter()
        .map(|offset| {
            let adapter = Arc::clone(&adapter);
            let sender = sender.clone();
            thread::spawn(move || {
                let result = adapter.read_to_completion(
                    &CellReadRequest::new(manifest(1), 1).with_byte_offset(offset),
                    CancellationToken::new(false),
                );
                sender.send((offset, result)).expect("result channel");
            })
        })
        .collect();
    let first = entered.recv_timeout(TIMEOUT).expect("first reader pending");
    let second = entered
        .recv_timeout(TIMEOUT)
        .expect("second reader pending");
    assert_ne!(first, second);
    // Public polling may drive reserved operations, but cannot steal events.
    assert!(adapter.poll(&mut context()).is_pending());
    gate.release();
    for _ in 0..2 {
        let (offset, result) = results.recv_timeout(TIMEOUT).expect("readers must wake");
        assert_eq!(result, Ok(vec![offset as u8]));
    }
    for thread in threads {
        thread.join().expect("reader thread");
    }
    assert!(adapter.is_empty());
}

#[test]
fn cancelling_parked_blocking_reader_wakes_and_retires_it() {
    let (gate, entered) = Gate::new();
    let adapter = Arc::new(IoSchedulerAdapter::new(GateProvider(gate), 1, 1).expect("adapter"));
    let (sender, results) = mpsc::channel();
    let reader = Arc::clone(&adapter);
    let thread = thread::spawn(move || {
        sender
            .send(reader.read_to_completion(
                &CellReadRequest::new(manifest(1), 1),
                CancellationToken::new(false),
            ))
            .expect("result channel");
    });
    let id = entered.recv_timeout(TIMEOUT).expect("reader parked");
    assert!(adapter.cancel(id));
    assert_eq!(
        results
            .recv_timeout(TIMEOUT)
            .expect("cancel must wake reader"),
        Err(IoError::Cancelled { request_id: id })
    );
    thread.join().expect("reader thread");
    assert!(adapter.is_empty());
}

#[test]
fn active_cancellation_keeps_identity_when_poll_starts_another_request() {
    let (gate, receiver) = Gate::new();
    let adapter = IoSchedulerAdapter::new(GateProvider(Arc::clone(&gate)), 2, 2).expect("adapter");
    let first = adapter
        .submit(CellReadRequest::new(manifest(1), 1))
        .expect("first");
    assert!(adapter.poll(&mut context()).is_pending());
    assert_eq!(receiver.recv_timeout(TIMEOUT).expect("active"), first);
    let second = adapter
        .submit(CellReadRequest::new(manifest(1), 1).with_byte_offset(1))
        .expect("second");
    assert!(adapter.cancel(first));
    assert!(
        matches!(adapter.poll(&mut context()), Poll::Ready(Ok(IoAdapterEvent::Failed {
        request_id, error: IoError::Cancelled { request_id: cancelled }, ..
    })) if request_id == first && cancelled == first)
    );
    gate.release();
    assert!(
        matches!(adapter.poll(&mut context()), Poll::Ready(Ok(IoAdapterEvent::Chunk {
        request_id, ..
    })) if request_id == second)
    );
    assert!(
        matches!(adapter.poll(&mut context()), Poll::Ready(Ok(IoAdapterEvent::Completed {
        request_id, ..
    })) if request_id == second)
    );
    assert!(adapter.is_empty());
}

struct WrongCancellation {
    foreign_id: IoRequestId,
    during_start: bool,
}

struct WrongCancellationOperation(IoRequestId);

impl ChunkedRead for WrongCancellationOperation {
    fn poll_chunk(&mut self, _: &mut Context<'_>) -> Poll<Result<Option<ReadChunk>, IoError>> {
        Poll::Ready(Err(IoError::Cancelled { request_id: self.0 }))
    }

    fn cancel(&mut self) {}
}

impl ReadProvider for WrongCancellation {
    type Operation = WrongCancellationOperation;

    fn start(&self, _: &ReadRequest, _: IoRequestId) -> Result<Self::Operation, IoError> {
        if self.during_start {
            Err(IoError::Cancelled {
                request_id: self.foreign_id,
            })
        } else {
            Ok(WrongCancellationOperation(self.foreign_id))
        }
    }
}

#[test]
fn provider_cancellation_cannot_spoof_another_request_at_start_or_poll() {
    let mut ids = IoScheduler::new(
        MemoryProvider {
            bytes: Arc::new(vec![1]),
        },
        1,
        2,
    )
    .expect("id source");
    ids.submit(ReadRequest::new(0, 1)).expect("id one");
    let foreign_id = ids.submit(ReadRequest::new(0, 1)).expect("id two");
    for during_start in [true, false] {
        let adapter = IoSchedulerAdapter::new(
            WrongCancellation {
                foreign_id,
                during_start,
            },
            2,
            2,
        )
        .expect("adapter");
        let own_id = adapter
            .submit(CellReadRequest::new(manifest(1), 1))
            .expect("own request");
        assert_ne!(own_id, foreign_id);
        assert!(
            matches!(adapter.poll(&mut context()), Poll::Ready(Ok(IoAdapterEvent::Failed {
            request_id, error: IoError::Cancelled { request_id: cancelled }, ..
        })) if request_id == own_id && cancelled == own_id)
        );
        assert!(adapter.is_empty());
    }
}

struct InvalidProvider;

struct InvalidOperation(u64);

impl ReadProvider for InvalidProvider {
    type Operation = InvalidOperation;

    fn start(&self, request: &ReadRequest, _: IoRequestId) -> Result<Self::Operation, IoError> {
        if request.offset() == 0 {
            Err(internal_error("start failed"))
        } else {
            Ok(InvalidOperation(request.offset()))
        }
    }
}

impl ChunkedRead for InvalidOperation {
    fn poll_chunk(&mut self, _: &mut Context<'_>) -> Poll<Result<Option<ReadChunk>, IoError>> {
        match self.0 {
            1 => Poll::Ready(Ok(Some(ReadChunk::new(99, vec![1])))),
            2 => Poll::Ready(Ok(Some(ReadChunk::new(2, vec![1, 2])))),
            3 => Poll::Ready(Ok(None)),
            _ => Poll::Ready(Err(internal_error("poll failed"))),
        }
    }

    fn cancel(&mut self) {}
}

#[test]
fn mixed_start_poll_and_scheduler_validation_failures_correlate_and_retire_once() {
    let adapter = IoSchedulerAdapter::new(InvalidProvider, 3, 5).expect("adapter");
    let ids: Vec<_> = (0..5)
        .map(|offset| {
            adapter
                .submit(CellReadRequest::new(manifest(1), 1).with_byte_offset(offset))
                .expect("submit")
        })
        .collect();
    let mut errors = BTreeMap::new();
    for _ in 0..5 {
        match adapter.poll(&mut context()) {
            Poll::Ready(Ok(IoAdapterEvent::Failed {
                request_id,
                request,
                error,
            })) => {
                assert_eq!(request.io_request().offset(), request_id.get() - 1);
                assert!(errors.insert(request_id, error).is_none());
            }
            other => panic!("expected identified failure: {other:?}"),
        }
    }
    assert!(matches!(errors[&ids[0]], IoError::Provider { .. }));
    assert!(matches!(
        errors[&ids[1]],
        IoError::InvalidChunkOffset { .. }
    ));
    assert!(matches!(
        errors[&ids[2]],
        IoError::InvalidChunkLength { .. }
    ));
    assert!(matches!(errors[&ids[3]], IoError::ShortRead { .. }));
    assert!(matches!(errors[&ids[4]], IoError::Provider { .. }));
    assert!(adapter.is_empty());
    assert!(adapter.poll(&mut context()).is_pending());
}
