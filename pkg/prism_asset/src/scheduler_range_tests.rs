//! Physical-range regressions through the real load/reload scheduler.

use super::*;
use crate::{
    AssetLoader, AssetMeta, AssetReadPlan, AssetReader, LoadContext, LoadError,
    RangeAssemblyLimits, RangePart, ReadError,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Condvar;
use std::time::{Duration, Instant};

#[derive(Debug)]
struct Bytes(Vec<u8>);

impl Asset for Bytes {
    const TYPE_NAME: &'static str = "scheduler_range_tests::Bytes";
}

struct Loader {
    limits: RangeAssemblyLimits,
    decodes: Arc<AtomicUsize>,
}

impl AssetLoader for Loader {
    type Asset = Bytes;

    fn extensions(&self) -> &'static [&'static str] {
        &["parallel"]
    }

    fn read_plan(&self, _: &AssetPath, _: AssetMeta) -> Result<AssetReadPlan, LoadError> {
        AssetReadPlan::ranges_with_limits(
            [4, 0, 2, 6].map(|offset| RangePart::new(offset, 2)),
            self.limits,
        )
        .map_err(|error| LoadError::LimitExceeded(error.to_string()))
    }

    fn load(&self, bytes: &[u8], _: &mut LoadContext) -> Result<Bytes, LoadError> {
        self.decodes.fetch_add(1, Ordering::AcqRel);
        Ok(Bytes(bytes.to_vec()))
    }
}

#[derive(Clone, Copy)]
enum Failure {
    None,
    Short,
    Oversized,
    Missing,
    Panic,
}

#[derive(Default)]
struct Reads {
    started: Vec<u64>,
    completed: Vec<u64>,
    active: usize,
    peak: usize,
    release: BTreeSet<u64>,
    release_all: bool,
}

#[derive(Default)]
struct Control {
    reads: Mutex<Reads>,
    changed: Condvar,
    metadata: AtomicUsize,
    full_reads: AtomicUsize,
}

impl Control {
    fn wait_for(&self, predicate: impl Fn(&Reads) -> bool) -> bool {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut state = self.reads.lock().unwrap();
        while !predicate(&state) {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return false;
            }
            state = self.changed.wait_timeout(state, remaining).unwrap().0;
        }
        true
    }

    fn release(&self, offset: u64) {
        self.reads.lock().unwrap().release.insert(offset);
        self.changed.notify_all();
    }

    fn release_all(&self) {
        self.reads.lock().unwrap().release_all = true;
        self.changed.notify_all();
    }
}

struct Source {
    control: Arc<Control>,
    bias: u8,
    failure: Failure,
}

impl AssetReader for Source {
    fn metadata(&self, _: &str) -> Result<AssetMeta, ReadError> {
        self.control.metadata.fetch_add(1, Ordering::AcqRel);
        Ok(AssetMeta {
            size: 8,
            modified_ms: None,
            is_dir: false,
        })
    }

    fn read(&self, _: &str) -> Result<Vec<u8>, ReadError> {
        self.control.full_reads.fetch_add(1, Ordering::AcqRel);
        Err(ReadError::Unsupported)
    }

    fn read_range(&self, _: &str, offset: u64, len: u64) -> Result<Vec<u8>, ReadError> {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut state = self.control.reads.lock().unwrap();
        state.started.push(offset);
        state.active += 1;
        state.peak = state.peak.max(state.active);
        self.control.changed.notify_all();
        while !state.release_all && !state.release.contains(&offset) {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                state.active -= 1;
                self.control.changed.notify_all();
                return Err(ReadError::Io("range gate timed out".into()));
            }
            state = self
                .control
                .changed
                .wait_timeout(state, remaining)
                .unwrap()
                .0;
        }
        state.active -= 1;
        state.completed.push(offset);
        self.control.changed.notify_all();
        drop(state);
        if offset == 4 {
            match self.failure {
                Failure::Short => return Ok(vec![0]),
                Failure::Oversized => return Ok(vec![0; 3]),
                Failure::Missing => return Err(ReadError::NotFound),
                Failure::Panic => panic!("range source panic"),
                Failure::None => {}
            }
        }
        Ok((offset..offset + len)
            .map(|n| n as u8 + self.bias)
            .collect())
    }
}

fn mount(server: &AssetServer, name: &str, bias: u8, failure: Failure) -> Arc<Control> {
    let control = Arc::new(Control::default());
    server.mount(
        name,
        Arc::new(Source {
            control: Arc::clone(&control),
            bias,
            failure,
        }),
        i32::from(bias),
    );
    control
}

fn setup(parallel: usize, max_bytes: u64) -> (Arc<AssetServer>, Arc<AtomicUsize>) {
    let server = Arc::new(AssetServer::new());
    let decodes = Arc::new(AtomicUsize::new(0));
    server.register_loader(Loader {
        limits: RangeAssemblyLimits {
            max_parallel: parallel,
            max_parts: 4,
            max_bytes,
        },
        decodes: Arc::clone(&decodes),
    });
    (server, decodes)
}

#[test]
fn bounded_load_and_reload_overlap_preserve_order_and_operation_pin() {
    let (server, decodes) = setup(2, 8);
    let base = mount(&server, "base", 0, Failure::None);
    let scheduler = LoadScheduler::with_io_cpu_pools(
        Arc::clone(&server),
        TaskPool::with_threads(4),
        TaskPool::with_threads(1),
        1,
    );
    let ticket = scheduler
        .schedule("hero.parallel", Priority::Normal)
        .unwrap();
    let overlap = base.wait_for(|reads| reads.active == 2);
    let patch = mount(&server, "patch", 10, Failure::None);
    // Complete part 1 before part 0, while remounting the source table. The
    // later batch must still use the base reader captured for this operation.
    base.release(0);
    let reversed = base.wait_for(|reads| reads.completed == [0]);
    let no_early_batch = base.reads.lock().unwrap().started.len() == 2;
    base.release_all();
    let resident = scheduler.cpu_pool.block_on(ticket).unwrap();
    let id = resident.id().typed::<Bytes>().unwrap();
    assert!(overlap && reversed && no_early_batch);
    assert_eq!(
        server.with_asset(id, |asset| asset.0.clone()),
        Some(vec![4, 5, 0, 1, 2, 3, 6, 7])
    );
    assert_eq!(base.reads.lock().unwrap().peak, 2);
    assert!(patch.reads.lock().unwrap().started.is_empty());

    let reload = scheduler
        .schedule_reload("hero.parallel", Priority::Normal)
        .unwrap();
    let overlap = patch.wait_for(|reads| reads.active == 2);
    let next = mount(&server, "next", 20, Failure::None);
    patch.release(0);
    let reversed = patch.wait_for(|reads| reads.completed == [0]);
    patch.release_all();
    let result = scheduler.cpu_pool.block_on(reload);
    assert!(overlap && reversed);
    // Reload publication has an additional existing overlay-revision guard:
    // the pinned read may finish, but a remount supersedes its staged bytes.
    assert!(
        matches!(result, Err(LoadFailure::AssemblyFailed(error)) if error == "async reload source overlay changed")
    );
    assert_eq!(
        server.with_asset(id, |asset| asset.0.clone()),
        Some(vec![4, 5, 0, 1, 2, 3, 6, 7])
    );
    assert_eq!(patch.reads.lock().unwrap().peak, 2);
    assert!(next.reads.lock().unwrap().started.is_empty());
    assert_eq!(decodes.load(Ordering::Acquire), 2);
    for control in [&base, &patch] {
        assert_eq!(control.metadata.load(Ordering::Acquire), 1);
        assert_eq!(control.full_reads.load(Ordering::Acquire), 0);
    }
    next.release_all();
    let reload = scheduler
        .schedule_reload("hero.parallel", Priority::Normal)
        .unwrap();
    assert_eq!(
        scheduler.cpu_pool.block_on(reload).unwrap().id(),
        resident.id()
    );
    assert_eq!(
        server.with_asset(id, |asset| asset.0.clone()),
        Some(vec![24, 25, 20, 21, 22, 23, 26, 27])
    );
}

#[test]
fn range_failures_drain_batch_stop_admission_and_keep_resident_revision() {
    for (failure, parallel) in [
        Failure::Short,
        Failure::Oversized,
        Failure::Missing,
        Failure::Panic,
    ]
    .into_iter()
    .flat_map(|failure| [(failure, 2), (failure, usize::MAX)])
    {
        let (server, decodes) = setup(parallel, 8);
        let base = mount(&server, "base", 0, Failure::None);
        base.release_all();
        let resident = server.load::<Bytes>("hero.parallel");
        let before = server
            .with_asset(resident.id(), |asset| asset.0.clone())
            .unwrap();
        let patch = mount(&server, "patch", 10, failure);
        let scheduler = LoadScheduler::with_io_cpu_pools(
            Arc::clone(&server),
            TaskPool::with_threads(2),
            TaskPool::with_threads(1),
            1,
        );
        let ticket = scheduler
            .schedule_reload("hero.parallel", Priority::Normal)
            .unwrap();
        let overlap = patch.wait_for(|reads| reads.active == 2);
        patch.release(4);
        let first_done = patch.wait_for(|reads| reads.completed == [4]);
        // One failed read cannot release the flight while its sibling is
        // still running. Nor may the next batch have been submitted.
        let still_running = scheduler.in_flight() == 1;
        patch.release_all();
        let result = scheduler.cpu_pool.block_on(ticket);
        assert!(overlap && first_done && still_running);
        let error = format!("{:?}", result.unwrap_err());
        assert!(
            match failure {
                Failure::Short | Failure::Oversized => error.contains("range part 0 short read"),
                Failure::Missing => error.contains("range part 0 failed"),
                Failure::Panic => error.contains("asset source or loader panicked"),
                Failure::None => unreachable!(),
            },
            "{error}"
        );
        let reads = patch.reads.lock().unwrap();
        assert_eq!(reads.started.len(), 2);
        assert_eq!(reads.active, 0);
        assert_eq!(scheduler.in_flight(), 0);
        assert_eq!(decodes.load(Ordering::Acquire), 1);
        assert_eq!(
            server.with_asset(resident.id(), |asset| asset.0.clone()),
            Some(before)
        );
        assert_eq!(patch.full_reads.load(Ordering::Acquire), 0);
    }
}

#[test]
fn initial_range_failure_is_terminal_without_decode_or_detached_reads() {
    for failure in [
        Failure::Short,
        Failure::Oversized,
        Failure::Missing,
        Failure::Panic,
    ] {
        let (server, decodes) = setup(2, 8);
        let source = mount(&server, "base", 0, failure);
        let scheduler = LoadScheduler::new(Arc::clone(&server), TaskPool::with_threads(2), 1);
        let ticket = scheduler
            .schedule("hero.parallel", Priority::Normal)
            .unwrap();
        let overlap = source.wait_for(|reads| reads.active == 2);
        source.release_all();
        assert!(scheduler.cpu_pool.block_on(ticket).is_err());
        assert!(overlap);
        assert_eq!(source.reads.lock().unwrap().started.len(), 2);
        assert_eq!(source.reads.lock().unwrap().active, 0);
        assert_eq!(source.full_reads.load(Ordering::Acquire), 0);
        assert_eq!(scheduler.in_flight(), 0);
        assert_eq!(decodes.load(Ordering::Acquire), 0);
        // A terminal failed flight must not retain the admission slot.
        let replacement = mount(&server, "patch", 10, Failure::None);
        replacement.release_all();
        let retry = scheduler
            .schedule("hero.parallel", Priority::Normal)
            .unwrap();
        assert!(scheduler.cpu_pool.block_on(retry).is_ok());
    }
}

#[test]
fn invalid_range_budget_rejects_load_and_reload_before_payload_io() {
    let (server, decodes) = setup(2, 7);
    let source = mount(&server, "base", 0, Failure::None);
    let scheduler = LoadScheduler::new(Arc::clone(&server), TaskPool::with_threads(1), 1);
    let load = scheduler
        .schedule("hero.parallel", Priority::Normal)
        .unwrap();
    assert!(scheduler.cpu_pool.block_on(load).is_err());
    let reload = scheduler
        .schedule_reload("hero.parallel", Priority::Normal)
        .unwrap();
    assert!(scheduler.cpu_pool.block_on(reload).is_err());
    assert!(source.reads.lock().unwrap().started.is_empty());
    assert_eq!(source.full_reads.load(Ordering::Acquire), 0);
    assert_eq!(decodes.load(Ordering::Acquire), 0);
    assert_eq!(scheduler.in_flight(), 0);
}

#[test]
fn cancelled_consumer_does_not_cancel_shared_parallel_load() {
    let (server, decodes) = setup(2, 8);
    let source = mount(&server, "base", 0, Failure::None);
    let scheduler = LoadScheduler::new(Arc::clone(&server), TaskPool::with_threads(2), 1);
    let cancelled = scheduler
        .schedule("hero.parallel", Priority::Normal)
        .unwrap();
    let joined = scheduler
        .schedule("hero.parallel", Priority::Normal)
        .unwrap();
    let overlap = source.wait_for(|reads| reads.active == 2);
    cancelled.cancel();
    assert!(matches!(
        scheduler.cpu_pool.block_on(cancelled),
        Err(LoadFailure::Cancelled)
    ));
    source.release_all();
    let resident = scheduler.cpu_pool.block_on(joined).unwrap();
    assert!(overlap);
    assert!(server.is_loaded(resident.id()));
    assert_eq!(source.reads.lock().unwrap().started.len(), 4);
    assert_eq!(decodes.load(Ordering::Acquire), 1);
}

#[test]
fn single_worker_and_zero_worker_pools_finish_without_nested_blocking() {
    for workers in [0, 1] {
        let (server, _) = setup(usize::MAX, 8);
        let source = mount(&server, "base", 0, Failure::None);
        source.release_all();
        let scheduler = LoadScheduler::new(Arc::clone(&server), TaskPool::with_threads(workers), 1);
        let load = scheduler
            .schedule("hero.parallel", Priority::Normal)
            .unwrap();
        let resident = scheduler.cpu_pool.block_on(load).unwrap();
        let reload = scheduler
            .schedule_reload("hero.parallel", Priority::Normal)
            .unwrap();
        assert_eq!(
            scheduler.cpu_pool.block_on(reload).unwrap().id(),
            resident.id()
        );
        assert_eq!(source.reads.lock().unwrap().peak, 1);
        assert_eq!(source.reads.lock().unwrap().started.len(), 8);
    }
}
