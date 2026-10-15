use super::*;
use prism_tasks::{Priority, PriorityCell};
use std::sync::{Arc, Mutex};
use std::task::Waker;

#[derive(Clone, Default)]
struct Provider(Arc<Mutex<Vec<(RequestId, ReadRequest)>>>);

struct Operation {
    offset: u64,
    remaining: u64,
}

impl ReadProvider for Provider {
    type Operation = Operation;
    fn start(&self, request: &ReadRequest, id: RequestId) -> Result<Operation, IoError> {
        self.0.lock().unwrap().push((id, request.clone()));
        Ok(Operation {
            offset: request.offset(),
            remaining: request.length(),
        })
    }
}

impl ChunkedRead for Operation {
    fn poll_chunk(&mut self, _: &mut Context<'_>) -> Poll<Result<Option<ReadChunk>, IoError>> {
        if self.remaining == 0 {
            return Poll::Ready(Ok(None));
        }
        let offset = self.offset;
        self.offset += 1;
        self.remaining -= 1;
        Poll::Ready(Ok(Some(ReadChunk::new(offset, vec![0]))))
    }
    fn cancel(&mut self) {}
}

fn completions(scheduler: &mut IoScheduler<Provider>) -> Vec<RequestId> {
    let mut ids = Vec::new();
    let mut cx = Context::from_waker(Waker::noop());
    // Bounded test driver: an unexpected permanent Pending must fail, not hang.
    for _ in 0..1024 {
        if scheduler.is_empty() {
            return ids;
        }
        match scheduler.poll(&mut cx) {
            Poll::Ready(Ok(IoEvent::Completed(summary))) => ids.push(summary.request_id()),
            Poll::Ready(Ok(IoEvent::Chunk { .. })) => {}
            other => panic!("unexpected event: {other:?}"),
        }
    }
    panic!("scheduler failed to drain");
}

#[test]
fn shared_request_equality_is_identity_not_live_value() {
    let priority = Arc::new(PriorityCell::new(Priority::Low));
    let request = ReadRequest::new(4, 3).with_shared_priority(Arc::clone(&priority));
    let same = request.clone();
    assert_eq!(request, same);
    priority.boost_to(Priority::Critical);
    assert_eq!(request.effective_lane(), Priority::Critical);
    assert_eq!(request, same);
    assert!(Arc::ptr_eq(request.shared_priority().unwrap(), &priority));
    assert_ne!(
        request,
        ReadRequest::new(4, 3)
            .with_shared_priority(Arc::new(PriorityCell::new(Priority::Critical)))
    );
    assert_ne!(request, ReadRequest::new(4, 3));
    assert!(request.clone().with_chunk_size(0).is_err());
}

#[test]
fn queued_late_boost_outranks_static_score_and_preserves_fifo_ties() {
    let provider = Provider::default();
    let probe = provider.clone();
    let mut scheduler = IoScheduler::new(provider, 1, 8).unwrap();
    let cell = Arc::new(PriorityCell::new(Priority::Background));
    let ordinary = scheduler
        .submit(ReadRequest::new(0, 0).with_priority(u32::MAX))
        .unwrap();
    let first = scheduler
        .submit(
            ReadRequest::new(0, 0)
                .with_priority(7)
                .with_shared_priority(Arc::clone(&cell)),
        )
        .unwrap();
    let low_score = scheduler
        .submit(
            ReadRequest::new(0, 0)
                .with_priority(1)
                .with_shared_priority(Arc::clone(&cell)),
        )
        .unwrap();
    let tie = scheduler
        .submit(
            ReadRequest::new(0, 0)
                .with_priority(7)
                .with_shared_priority(Arc::clone(&cell)),
        )
        .unwrap();
    // Boost after all submissions; numeric stream scores cannot mask the lane.
    cell.boost_to(Priority::High);
    assert_eq!(
        completions(&mut scheduler),
        vec![first, tie, low_score, ordinary]
    );
    assert!(probe
        .0
        .lock()
        .unwrap()
        .iter()
        .filter(|(id, _)| *id != ordinary)
        .all(|(_, request)| Arc::ptr_eq(request.shared_priority().unwrap(), &cell)));
}

#[test]
fn normal_lane_mixes_static_and_shared_requests_by_score_then_fifo() {
    let mut scheduler = IoScheduler::new(Provider::default(), 1, 8).unwrap();
    let cell = Arc::new(PriorityCell::new(Priority::Normal));
    let shared_first = scheduler
        .submit(
            ReadRequest::new(0, 0)
                .with_priority(9)
                .with_shared_priority(Arc::clone(&cell)),
        )
        .unwrap();
    let static_tie = scheduler
        .submit(ReadRequest::new(0, 0).with_priority(9))
        .unwrap();
    let shared_tie = scheduler
        .submit(
            ReadRequest::new(0, 0)
                .with_priority(9)
                .with_shared_priority(cell),
        )
        .unwrap();
    let background = scheduler
        .submit(
            ReadRequest::new(0, 0)
                .with_priority(u32::MAX)
                .with_shared_priority(Arc::new(PriorityCell::new(Priority::Background))),
        )
        .unwrap();
    let normal_zero = scheduler.submit(ReadRequest::new(0, 0)).unwrap();
    assert_eq!(
        completions(&mut scheduler),
        vec![
            shared_first,
            static_tie,
            shared_tie,
            normal_zero,
            background
        ]
    );
}

#[test]
fn cancellation_retains_capacity_until_terminal_and_does_not_reuse_ids() {
    let provider = Provider::default();
    let probe = provider.clone();
    let mut scheduler = IoScheduler::new(provider, 1, 1).unwrap();
    let cell = Arc::new(PriorityCell::new(Priority::Low));
    let request = ReadRequest::new(0, 1).with_shared_priority(Arc::clone(&cell));
    let cancelled = scheduler.submit(request.clone()).unwrap();
    assert!(scheduler.cancel(cancelled));
    assert!(!scheduler.cancel(cancelled));
    cell.boost_to(Priority::Critical);
    assert_eq!(scheduler.len(), 1);
    assert!(matches!(
        scheduler.submit(request.clone()),
        Err(IoError::Backpressure { capacity: 1 })
    ));
    assert_eq!(
        scheduler.poll(&mut Context::from_waker(Waker::noop())),
        Poll::Ready(Err(IoError::Cancelled {
            request_id: cancelled
        }))
    );
    assert!(probe.0.lock().unwrap().is_empty());
    let replacement = scheduler.submit(request).unwrap();
    assert!(replacement > cancelled);
    assert_eq!(completions(&mut scheduler), vec![replacement]);
}

#[test]
fn boost_does_not_preempt_active_reads_or_their_round_robin() {
    let mut scheduler = IoScheduler::new(Provider::default(), 2, 3).unwrap();
    let cell = Arc::new(PriorityCell::new(Priority::Low));
    let first = scheduler.submit(ReadRequest::new(0, 2)).unwrap();
    let second = scheduler
        .submit(ReadRequest::new(0, 2).with_shared_priority(Arc::clone(&cell)))
        .unwrap();
    let mut cx = Context::from_waker(Waker::noop());
    assert!(
        matches!(scheduler.poll(&mut cx), Poll::Ready(Ok(IoEvent::Chunk { request_id, .. })) if request_id == first)
    );
    let queued_cell = Arc::new(PriorityCell::new(Priority::Background));
    let queued = scheduler
        .submit(ReadRequest::new(0, 0).with_shared_priority(Arc::clone(&queued_cell)))
        .unwrap();
    cell.boost_to(Priority::Critical);
    queued_cell.boost_to(Priority::Critical);
    for expected in [second, first, second] {
        assert!(
            matches!(scheduler.poll(&mut cx), Poll::Ready(Ok(IoEvent::Chunk { request_id, .. })) if request_id == expected)
        );
    }
    assert_eq!(completions(&mut scheduler), vec![first, second, queued]);
}

#[test]
fn concurrent_boosts_are_monotonic_and_seen_on_next_admission() {
    let cell = Arc::new(PriorityCell::new(Priority::Background));
    std::thread::scope(|scope| {
        for priority in [
            Priority::Low,
            Priority::Normal,
            Priority::Critical,
            Priority::High,
        ] {
            let cell = Arc::clone(&cell);
            scope.spawn(move || cell.boost_to(priority));
        }
    });
    assert_eq!(cell.get(), Priority::Critical);
    let mut scheduler = IoScheduler::new(Provider::default(), 1, 2).unwrap();
    let ordinary = scheduler
        .submit(ReadRequest::new(0, 0).with_priority(u32::MAX))
        .unwrap();
    let shared = scheduler
        .submit(ReadRequest::new(0, 0).with_shared_priority(cell))
        .unwrap();
    assert_eq!(completions(&mut scheduler), vec![shared, ordinary]);
}
