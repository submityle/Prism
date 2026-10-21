//! Same-path retry and rejected-output destruction through the real server.

use super::*;
use crate::{LoadContext, LoadError, LoadFailure, LoadScheduler, MemSource};
use core::future::Future;
use core::task::{Context, Poll, Waker};
use prism_tasks::{Priority, TaskPool};
use std::sync::{Barrier, Condvar, Weak};
use std::task::Wake;
use std::time::{Duration, Instant};

const PATH: &str = "asset.retry";
const WAIT: Duration = Duration::from_secs(5);
const WORKER_WAIT: Duration = Duration::from_secs(30);

#[derive(Debug, PartialEq, Eq)]
struct Value(Vec<u8>);

impl Asset for Value {
    const TYPE_NAME: &'static str = "server_retry_tests::Value";
}

struct Other;

impl Asset for Other {
    const TYPE_NAME: &'static str = "server_retry_tests::Other";
}

#[derive(Default)]
struct DecodeGates {
    // Each invocation rendezvous is independent of scheduling/timing.
    state: Mutex<(usize, usize)>, // (entered, released through)
    changed: Condvar,
}

impl DecodeGates {
    fn enter(&self) {
        let deadline = Instant::now() + WORKER_WAIT;
        let mut state = self.state.lock().unwrap();
        state.0 += 1;
        let invocation = state.0;
        self.changed.notify_all();
        while state.1 < invocation {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero(), "decode gate was not released");
            state = self.changed.wait_timeout(state, remaining).unwrap().0;
        }
    }

    fn wait_entered(&self, count: usize) {
        let deadline = Instant::now() + WAIT;
        let mut state = self.state.lock().unwrap();
        while state.0 < count {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero(), "decoder did not reach rendezvous");
            state = self.changed.wait_timeout(state, remaining).unwrap().0;
        }
    }

    fn release(&self, count: usize) {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).1 = count;
        self.changed.notify_all();
    }
}

struct Loader(Arc<DecodeGates>);

impl AssetLoader for Loader {
    type Asset = Value;

    fn extensions(&self) -> &'static [&'static str] {
        &["retry"]
    }

    fn load(&self, bytes: &[u8], _ctx: &mut LoadContext) -> Result<Value, LoadError> {
        self.0.enter();
        Ok(Value(bytes.to_vec()))
    }
}

struct Scenario {
    server: Arc<AssetServer>,
    scheduler: LoadScheduler,
    gates: Arc<DecodeGates>,
}

impl Scenario {
    fn new() -> Self {
        let server = Arc::new(AssetServer::new());
        let gates = Arc::new(DecodeGates::default());
        server.register_loader(Loader(Arc::clone(&gates)));
        let scheduler = LoadScheduler::with_io_cpu_pools(
            Arc::clone(&server),
            TaskPool::with_threads(1),
            TaskPool::with_threads(1),
            2,
        );
        Self {
            server,
            scheduler,
            gates,
        }
    }

    fn mount(&self, bytes: &[u8]) {
        let mut source = MemSource::new();
        source.insert(PATH, bytes.to_vec());
        self.server.mount("retry", Arc::new(source), 0);
    }

    fn wait_drained(&self) {
        let deadline = Instant::now() + WAIT;
        while self.scheduler.in_flight() != 0 {
            assert!(Instant::now() < deadline, "cancelled worker did not drain");
            std::thread::yield_now();
        }
    }
}

impl Drop for Scenario {
    fn drop(&mut self) {
        // Release before task-pool destruction, including assertion unwind.
        self.gates.release(usize::MAX);
    }
}

struct ThreadWake(std::thread::Thread);

impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

fn finish<F: Future>(future: F) -> F::Output {
    let mut future = Box::pin(future);
    let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    let deadline = Instant::now() + WAIT;
    loop {
        if let Poll::Ready(output) = future.as_mut().poll(&mut cx) {
            return output;
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(!remaining.is_zero(), "ticket was not completed/woken");
        std::thread::park_timeout(remaining);
    }
}

#[test]
fn cancelled_failed_strong_handle_allows_same_path_scheduler_retry() {
    let scenario = Scenario::new();
    scenario.mount(b"old revision");
    let ticket = scenario.scheduler.schedule(PATH, Priority::Normal).unwrap();
    scenario.gates.wait_entered(1);
    let old = scenario.server.load::<Value>(PATH);
    assert_eq!(
        scenario.server.load_state(old.untyped_id()),
        LoadState::Loading
    );
    ticket.cancel();
    assert!(matches!(finish(ticket), Err(LoadFailure::Cancelled)));
    scenario.gates.release(1);
    scenario.wait_drained();

    let failed = scenario.server.load_state(old.untyped_id());
    assert!(failed.is_failed());
    assert_eq!(
        scenario.server.recursive_state(old.untyped_id()),
        RecursiveDependencyLoadState::from(failed)
    );
    // Preserve synchronous primary-load semantics: Failed remains interned
    // until an explicitly scheduled asynchronous retry replaces the path.
    assert_eq!(scenario.server.load::<Value>(PATH).id(), old.id());
    assert_eq!(scenario.server.load_state(old.untyped_id()), failed);
    assert!(scenario
        .server
        .prepare_async_load(PATH.into(), Some(AssetTypeId::of::<Other>()))
        .is_err());
    assert_eq!(scenario.server.load::<Value>(PATH).id(), old.id());

    let revision = scenario.server.lock().source_revision;
    scenario.mount(b"new revision");
    assert_ne!(scenario.server.lock().source_revision, revision);
    let first = scenario.scheduler.schedule(PATH, Priority::Normal).unwrap();
    let second = scenario.scheduler.schedule(PATH, Priority::Normal).unwrap();
    scenario.gates.wait_entered(2);
    assert_eq!(scenario.scheduler.in_flight(), 1);
    let retry = match scenario
        .server
        .prepare_async_load(PATH.into(), None)
        .unwrap()
    {
        AsyncLoadPreparation::Existing(handle) => handle,
        AsyncLoadPreparation::New(_) => panic!("retry was not deduplicated"),
    };
    assert_ne!(retry.id(), old.untyped_id());
    assert_eq!(scenario.server.load_state(retry.id()), LoadState::Loading);
    assert_eq!(scenario.server.load_state(old.untyped_id()), failed);
    assert!(scenario
        .server
        .prepare_async_load(PATH.into(), Some(AssetTypeId::of::<Other>()))
        .is_err());
    scenario.gates.release(2);
    let first = finish(first).unwrap();
    let second = finish(second).unwrap();
    assert_eq!(first.id(), retry.id());
    assert_eq!(second.id(), retry.id());
    assert_eq!(
        scenario
            .server
            .with_asset(first.id().typed::<Value>().unwrap(), |v| v.0.clone()),
        Some(b"new revision".to_vec())
    );
    assert_eq!(scenario.server.load_state(old.untyped_id()), failed);
    assert_eq!(scenario.gates.state.lock().unwrap().0, 2);
    scenario.server.collect_releases();
    assert_eq!(scenario.server.load_state(old.untyped_id()), failed);
}

fn prepare<A: Asset>(server: &AssetServer) -> AsyncLoadPlan {
    server.lock().stores.register::<A>();
    match server
        .prepare_async_load(PATH.into(), Some(AssetTypeId::of::<A>()))
        .unwrap()
    {
        AsyncLoadPreparation::New(plan) => *plan,
        AsyncLoadPreparation::Existing(_) => panic!("expected a fresh preparation"),
    }
}

fn erased<A: Asset>(value: A, label: &str) -> ErasedLoadedAsset {
    ErasedLoadedAsset {
        type_id: AssetTypeId::of::<A>(),
        value: Box::new(value),
        dependencies: Vec::new(),
        label: label.into(),
        make_store: || Box::new(Assets::<A>::new()),
    }
}

#[test]
fn failed_retry_preparation_is_atomic_and_unwind_cleanup_is_id_scoped() {
    let server = Arc::new(AssetServer::new());
    let original = prepare::<Value>(&server);
    let old = original.handle.clone();
    server.fail_async_load(&original, "cancelled");
    let failed = server.load_state(old.id());
    let barrier = Arc::new(Barrier::new(3));
    let workers: Vec<_> = (0..2)
        .map(|_| {
            let server = Arc::clone(&server);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                server
                    .prepare_async_load(PATH.into(), Some(AssetTypeId::of::<Value>()))
                    .unwrap()
            })
        })
        .collect();
    barrier.wait();
    let mut plan = None;
    let mut existing = None;
    for worker in workers {
        match worker.join().unwrap() {
            AsyncLoadPreparation::New(prepared) => assert!(plan.replace(*prepared).is_none()),
            AsyncLoadPreparation::Existing(handle) => assert!(existing.replace(handle).is_none()),
        }
    }
    let plan = plan.unwrap();
    let retry = existing.unwrap();
    assert_eq!(retry.id(), plan.id);
    assert_ne!(retry.id(), old.id());
    server.fail_async_load_if_loading(original.id, &original.path, "late old unwind");
    assert_eq!(server.load_state(old.id()), failed);
    assert_eq!(server.load_state(retry.id()), LoadState::Loading);
    let id = plan.id;
    let path = plan.path.clone();
    server
        .commit_async_load(
            plan,
            FullLoadOutput {
                primary: erased(Value(vec![7]), ""),
                labeled: vec![],
            },
            vec![],
            vec![],
        )
        .unwrap();
    server.fail_async_load_if_loading(id, &path, "postcommit destructor panic");
    assert_eq!(server.load_state(id), LoadState::Loaded);
    assert_eq!(
        server.recursive_state(id),
        RecursiveDependencyLoadState::Loaded
    );

    let server = AssetServer::new();
    let plan = prepare::<Value>(&server);
    let retained = plan.handle.clone();
    server.fail_async_load_if_loading(plan.id, &plan.path, "worker unwind");
    let failed = server.load_state(retained.id());
    assert!(failed.is_failed());
    server.fail_async_load_if_loading(plan.id, &plan.path, "duplicate unwind");
    assert_eq!(server.load_state(retained.id()), failed);
}

struct DropLog {
    server: Weak<AssetServer>,
    id: UntypedAssetId,
    drops: Mutex<Vec<(u8, bool)>>,
    panic_at: Option<u8>,
}

struct Probe {
    number: u8,
    log: Arc<DropLog>,
}

impl Asset for Probe {
    const TYPE_NAME: &'static str = "server_retry_tests::Probe";
}

impl Drop for Probe {
    fn drop(&mut self) {
        let server = self.log.server.upgrade().unwrap();
        let unlocked = match server.inner.try_lock() {
            Ok(guard) => {
                drop(guard);
                true
            }
            Err(_) => false,
        };
        if unlocked {
            // Actual reentry, guarded by a nonblocking probe so a regression
            // fails the assertion instead of deadlocking the test process.
            let _ = server.load_state(self.log.id);
        }
        self.log.drops.lock().unwrap().push((self.number, unlocked));
        if self.log.panic_at == Some(self.number) {
            panic!("expected rejected-payload destructor panic");
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Rejection {
    Duplicate,
    LabelTypeChanged,
    PrimaryCycle,
    LabelCycle,
    NoLongerLoading,
    Incomplete,
    WrongPrimaryType,
}

fn rejected_output_drops_unlocked(rejection: Rejection, panic_at: Option<u8>) {
    let server = Arc::new(AssetServer::new());
    let plan = prepare::<Probe>(&server);
    let handle = plan.handle.clone();
    let log = Arc::new(DropLog {
        server: Arc::downgrade(&server),
        id: plan.id,
        drops: Mutex::new(Vec::new()),
        panic_at,
    });
    let probe = |number| Probe {
        number,
        log: Arc::clone(&log),
    };
    let mut output = FullLoadOutput {
        primary: erased(probe(0), ""),
        labeled: vec![
            erased(probe(1), "part"),
            erased(probe(2), "next"),
            erased(probe(3), "tail"),
        ],
    };
    let mut dependencies = Vec::new();
    let mut labeled_dependencies = vec![vec![], vec![], vec![]];
    let expected = match rejection {
        Rejection::Duplicate => {
            output.labeled[1].label = "part".into();
            "duplicate labeled asset"
        }
        Rejection::LabelTypeChanged | Rejection::LabelCycle => {
            // A live preexisting label is legal; its original slot must not
            // be rolled back along with provisional labels on rejection.
            let mut inner = server.lock();
            let label = if matches!(rejection, Rejection::LabelTypeChanged) {
                inner.stores.reserve::<Other>()
            } else {
                inner.stores.reserve::<Probe>()
            };
            inner.graph.add_asset(label.id());
            inner
                .interned
                .insert(plan.path.clone().with_label("part"), label.downgrade());
            dependencies.push(label);
            if matches!(rejection, Rejection::LabelTypeChanged) {
                "type changed"
            } else {
                "async labeled dependency cycle"
            }
        }
        Rejection::PrimaryCycle => {
            let dependent = server
                .insert_runtime(Value(vec![]), [plan.handle.clone()])
                .unwrap();
            dependencies.push(dependent.untyped());
            "dependency cycle"
        }
        Rejection::NoLongerLoading => {
            server.fail_async_load(&plan, "cancelled before commit");
            "no longer loading"
        }
        Rejection::Incomplete => {
            labeled_dependencies.pop();
            "staging is incomplete"
        }
        Rejection::WrongPrimaryType => {
            output.primary.type_id = AssetTypeId::of::<Other>();
            "unexpected asset type"
        }
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        server.commit_async_load(plan, output, dependencies, labeled_dependencies)
    }));
    if panic_at.is_some() {
        assert!(result.is_err(), "expected Drop panic for {rejection:?}");
    } else {
        assert!(
            result.unwrap().unwrap_err().contains(expected),
            "wrong rejection for {rejection:?}"
        );
    }
    let mut drops = log.drops.lock().unwrap().clone();
    drops.sort_unstable();
    assert_eq!(
        drops,
        vec![(0, true), (1, true), (2, true), (3, true)],
        "payload dropped with server locked: {rejection:?}"
    );
    assert!(
        !server.inner.is_poisoned(),
        "Drop poisoned server: {rejection:?}"
    );
    assert!(server.load_state(handle.id()).is_failed());
    for label in ["next", "tail"] {
        let inner = server.lock();
        let path = AssetPath::from(PATH).with_label(label);
        assert!(!inner.interned.contains_key(&path));
        assert!(!inner
            .catalog
            .contains_key(&StableGuid::from_path(&path.to_string())));
    }
}

#[test]
fn async_commit_rejections_drop_primary_staged_current_and_remaining_payloads_unlocked() {
    for rejection in [
        Rejection::Duplicate,
        Rejection::LabelTypeChanged,
        Rejection::PrimaryCycle,
        Rejection::LabelCycle,
        Rejection::NoLongerLoading,
        Rejection::Incomplete,
        Rejection::WrongPrimaryType,
    ] {
        rejected_output_drops_unlocked(rejection, None);
    }
}

#[test]
fn rejected_payload_drop_panic_does_not_poison_server_or_skip_remaining_drops() {
    // Primary, staged label, currently rejected label, unvisited iterator tail.
    for number in 0..4 {
        rejected_output_drops_unlocked(Rejection::Duplicate, Some(number));
    }
}
