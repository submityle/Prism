//! First admission at a fixed queue depth, not disk throughput or task latency.
use std::hint::black_box;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};
use std::time::Instant;

use prism_tasks::{Priority, PriorityCell};
use prism_world_io::{
    ChunkedRead, IoError, IoEvent, IoScheduler, ReadChunk, ReadProvider, ReadRequest, RequestId,
};

struct EmptyProvider;
struct EmptyRead;

impl ReadProvider for EmptyProvider {
    type Operation = EmptyRead;

    fn start(&self, _: &ReadRequest, _: RequestId) -> Result<EmptyRead, IoError> {
        Ok(EmptyRead)
    }
}

impl ChunkedRead for EmptyRead {
    fn poll_chunk(&mut self, _: &mut Context<'_>) -> Poll<Result<Option<ReadChunk>, IoError>> {
        Poll::Ready(Ok(None))
    }

    fn cancel(&mut self) {}
}

fn measure(depth: usize, shared: bool) -> [u128; 3] {
    let mut samples = Vec::with_capacity(128);
    for round in 0..144 {
        let mut scheduler = IoScheduler::new(EmptyProvider, 1, depth).unwrap();
        let boost = Arc::new(PriorityCell::new(Priority::Background));
        let normal = Arc::new(PriorityCell::new(Priority::Normal));
        let mut expected = None;
        for index in 0..depth {
            let last = index + 1 == depth;
            let request =
                ReadRequest::new(0, 0).with_priority(if last { u32::MAX } else { index as u32 });
            let request = if shared {
                request.with_shared_priority(Arc::clone(if last { &boost } else { &normal }))
            } else {
                request
            };
            expected = Some(scheduler.submit(request).unwrap());
        }
        // Post-enqueue promotion is outside timing; its observation is checked.
        boost.boost_to(Priority::Critical);
        let mut cx = Context::from_waker(Waker::noop());
        let start = Instant::now();
        let result = black_box(scheduler.poll(&mut cx));
        let elapsed = start.elapsed().as_nanos();
        match result {
            Poll::Ready(Ok(IoEvent::Completed(summary))) => {
                assert_eq!(Some(summary.request_id()), expected);
            }
            other => panic!("unexpected first admission: {other:?}"),
        }
        assert_eq!(scheduler.len(), depth - 1);
        if round >= 16 {
            samples.push(elapsed);
        }
        // Remaining queue teardown is deliberately outside the sample.
        black_box(scheduler);
    }
    samples.sort_unstable();
    [samples[63], samples[121], samples[126]]
}

fn main() {
    println!("first admission ns; 16 warmup, 128 samples, nearest-rank P50/P95/P99");
    println!("zero-byte provider; setup/boost/teardown excluded; no files, tasks or GPU");
    for depth in [64, 256, 1024, 4096] {
        for shared in [false, true] {
            let [p50, p95, p99] = measure(depth, shared);
            println!("depth={depth} shared={shared} ns={p50}/{p95}/{p99}");
        }
    }
}
