//! Cancellation regressions through real scheduler flights. Gates prove the
//! interleaving; observing a flight never attaches an extra consumer.

use super::*;
use crate::{
    AssetEvent, AssetLoader, AssetMeta, AssetReadPlan, AssetReader, DepRequest, LoadContext,
    LoadError, RangeAssemblyLimits, RangePart, ReadError,
};
use std::sync::Condvar;
use std::task::Wake;
use std::time::{Duration, Instant};

// A worker's gate outlives an observer's deadline. A stuck cancellation must
// fail the observer, not look successful because the blocked reader timed out.
const WAIT: Duration = Duration::from_secs(5);
const GATE_WAIT: Duration = Duration::from_secs(30);
const ROOT: &str = "root.cancel";
const CHILD: &str = "child.cancel";
const OTHER: &str = "other.cancel";
const FRESH: &str = "fresh.cancel";
const CHILD_DEPENDENCIES: &[&str] = &[CHILD];
const INITIAL_BYTES: &[u8] = &[4, 5, 0, 1, 2, 3, 6, 7];

#[derive(Clone, Debug, PartialEq, Eq)]
struct Value(Vec<u8>);

impl Asset for Value {
    const TYPE_NAME: &'static str = "scheduler_cancellation_tests::Value";
}

#[derive(Clone, Copy)]
struct Node {
    path: &'static str,
    dependencies: &'static [&'static str],
    labeled: bool,
    block_reads: bool,
    block_decode: bool,
}

impl Node {
    const fn leaf(path: &'static str) -> Self {
        Self {
            path,
            dependencies: &[],
            labeled: false,
            block_reads: false,
            block_decode: false,
        }
    }

    const fn reading(mut self) -> Self {
        self.block_reads = true;
        self
    }

    const fn decoding(mut self) -> Self {
        self.block_decode = true;
        self
    }

    const fn parent(mut self, labeled: bool) -> Self {
        self.dependencies = CHILD_DEPENDENCIES;
        self.labeled = labeled;
        self
    }
}

#[derive(Debug)]
struct Activity {
    metadata: usize,
    full_reads: usize,
    started: Vec<u64>,
    completed: Vec<u64>,
    active: usize,
    decodes: usize,
    reads_open: bool,
    decode_open: bool,
    released: BTreeSet<u64>,
    byte_bias: u8,
}

struct Control {
    activity: Mutex<BTreeMap<&'static str, Activity>>,
    changed: Condvar,
}

impl Control {
    fn new(nodes: &[Node]) -> Self {
        Self {
            activity: Mutex::new(
                nodes
                    .iter()
                    .map(|node| {
                        (
                            node.path,
                            Activity {
                                metadata: 0,
                                full_reads: 0,
                                started: Vec::new(),
                                completed: Vec::new(),
                                active: 0,
                                decodes: 0,
                                reads_open: !node.block_reads,
                                decode_open: !node.block_decode,
                                released: BTreeSet::new(),
                                byte_bias: 0,
                            },
                        )
                    })
                    .collect(),
            ),
            changed: Condvar::new(),
        }
    }

    fn wait_for(&self, path: &str, predicate: impl Fn(&Activity) -> bool) {
        let deadline = Instant::now() + WAIT;
        let mut state = self.activity.lock().unwrap();
        while !predicate(&state[path]) {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(
                !remaining.is_zero(),
                "activity rendezvous timed out for {path}: {:?}",
                state[path]
            );
            state = self.changed.wait_timeout(state, remaining).unwrap().0;
        }
    }

    fn release_part(&self, path: &str, offset: u64) {
        self.activity
            .lock()
            .unwrap()
            .get_mut(path)
            .unwrap()
            .released
            .insert(offset);
        self.changed.notify_all();
    }

    fn release_reads(&self, path: &str) {
        self.activity
            .lock()
            .unwrap()
            .get_mut(path)
            .unwrap()
            .reads_open = true;
        self.changed.notify_all();
    }

    fn release_decode(&self, path: &str) {
        self.activity
            .lock()
            .unwrap()
            .get_mut(path)
            .unwrap()
            .decode_open = true;
        self.changed.notify_all();
    }

    fn release_all(&self) {
        let mut state = self
            .activity
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        for activity in state.values_mut() {
            activity.reads_open = true;
            activity.decode_open = true;
        }
        self.changed.notify_all();
    }

    fn reset(&self, path: &str, reads_open: bool, decode_open: bool, byte_bias: u8) {
        let mut state = self.activity.lock().unwrap();
        let activity = state.get_mut(path).unwrap();
        assert_eq!(activity.active, 0, "cannot reset active reads for {path}");
        activity.metadata = 0;
        activity.full_reads = 0;
        activity.started.clear();
        activity.completed.clear();
        activity.decodes = 0;
        activity.reads_open = reads_open;
        activity.decode_open = decode_open;
        activity.released.clear();
        activity.byte_bias = byte_bias;
        self.changed.notify_all();
    }

    fn assert_unread(&self, path: &str) {
        let state = self.activity.lock().unwrap();
        let activity = &state[path];
        assert_eq!(activity.metadata, 0, "unexpected metadata for {path}");
        assert_eq!(activity.full_reads, 0);
        assert!(activity.started.is_empty(), "unexpected reads for {path}");
        assert_eq!(activity.decodes, 0, "unexpected decode for {path}");
    }
}

struct Source(Arc<Control>);

impl AssetReader for Source {
    fn metadata(&self, path: &str) -> Result<AssetMeta, ReadError> {
        let mut state = self.0.activity.lock().unwrap();
        let activity = state.get_mut(path).ok_or(ReadError::NotFound)?;
        activity.metadata += 1;
        self.0.changed.notify_all();
        Ok(AssetMeta {
            size: 8,
            modified_ms: None,
            is_dir: false,
        })
    }

    fn read(&self, path: &str) -> Result<Vec<u8>, ReadError> {
        if let Some(activity) = self.0.activity.lock().unwrap().get_mut(path) {
            activity.full_reads += 1;
        }
        Err(ReadError::Unsupported)
    }

    fn read_range(&self, path: &str, offset: u64, len: u64) -> Result<Vec<u8>, ReadError> {
        let deadline = Instant::now() + GATE_WAIT;
        let mut state = self.0.activity.lock().unwrap();
        let activity = state.get_mut(path).ok_or(ReadError::NotFound)?;
        activity.started.push(offset);
        activity.active += 1;
        self.0.changed.notify_all();
        while !state[path].reads_open && !state[path].released.contains(&offset) {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                state.get_mut(path).unwrap().active -= 1;
                self.0.changed.notify_all();
                return Err(ReadError::Io(format!("range gate timed out: {path}")));
            }
            state = self.0.changed.wait_timeout(state, remaining).unwrap().0;
        }
        let activity = state.get_mut(path).unwrap();
        activity.active -= 1;
        activity.completed.push(offset);
        self.0.changed.notify_all();
        let byte_bias = activity.byte_bias;
        Ok((offset..offset + len)
            .map(|value| (value as u8).wrapping_add(byte_bias))
            .collect())
    }
}

struct Loader {
    nodes: BTreeMap<&'static str, Node>,
    control: Arc<Control>,
}

impl AssetLoader for Loader {
    type Asset = Value;

    fn extensions(&self) -> &'static [&'static str] {
        &["cancel"]
    }

    fn read_plan(&self, _: &AssetPath, _: AssetMeta) -> Result<AssetReadPlan, LoadError> {
        AssetReadPlan::ranges_with_limits(
            [4, 0, 2, 6].map(|offset| RangePart::new(offset, 2)),
            RangeAssemblyLimits {
                max_parallel: 2,
                max_parts: 4,
                max_bytes: 8,
            },
        )
        .map_err(|error| LoadError::LimitExceeded(error.to_string()))
    }

    fn load(&self, bytes: &[u8], ctx: &mut LoadContext) -> Result<Value, LoadError> {
        let path = ctx.path().path();
        let node = self
            .nodes
            .get(path)
            .ok_or_else(|| LoadError::Other(format!("no cancellation-test node for {path}")))?;
        let deadline = Instant::now() + GATE_WAIT;
        let mut state = self.control.activity.lock().unwrap();
        state.get_mut(path).unwrap().decodes += 1;
        self.control.changed.notify_all();
        while !state[path].decode_open {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(LoadError::Other(format!("decode gate timed out: {path}")));
            }
            state = self
                .control
                .changed
                .wait_timeout(state, remaining)
                .unwrap()
                .0;
        }
        drop(state);
        let value = Value(bytes.to_vec());
        if node.labeled {
            ctx.add_labeled_asset_with_deps(
                "part",
                value.clone(),
                node.dependencies
                    .iter()
                    .map(|path| DepRequest::typed((*path).into(), AssetTypeId::of::<Value>()))
                    .collect(),
            );
        } else {
            for dependency in node.dependencies {
                ctx.add_typed_dependency((*dependency).into(), AssetTypeId::of::<Value>());
            }
        }
        Ok(value)
    }
}

struct Scenario {
    control: Arc<Control>,
    scheduler: LoadScheduler,
    server: Arc<AssetServer>,
}

impl Scenario {
    fn new(nodes: &[Node], capacity: usize) -> Self {
        let control = Arc::new(Control::new(nodes));
        let server = Arc::new(AssetServer::new());
        server.register_loader(Loader {
            nodes: nodes.iter().map(|node| (node.path, *node)).collect(),
            control: Arc::clone(&control),
        });
        server.mount("cancel-test", Arc::new(Source(Arc::clone(&control))), 0);
        let scheduler = LoadScheduler::with_io_cpu_pools(
            Arc::clone(&server),
            TaskPool::with_threads(4),
            // Once a child starts on this pool, its parent has returned
            // Pending from the dependency wait, including edge registration.
            TaskPool::with_threads(1),
            capacity,
        );
        Self {
            control,
            scheduler,
            server,
        }
    }

    fn schedule(&self, path: &str) -> LoadTicket {
        self.scheduler.schedule(path, Priority::Normal).unwrap()
    }

    fn reserved(&self, path: &str) -> UntypedHandle {
        // Only call after a read/decode rendezvous proves preparation ran.
        // Existing reservations are queried without synchronous source I/O.
        match self
            .server
            .prepare_async_load(path.into(), Some(AssetTypeId::of::<Value>()))
            .unwrap()
        {
            AsyncLoadPreparation::Existing(handle) => handle,
            AsyncLoadPreparation::New(_) => panic!("expected an existing reservation: {path}"),
        }
    }

    fn make_resident(&self, path: &str) -> UntypedHandle {
        self.control.release_all();
        let handle = finish(self.schedule(path)).expect("initial revision must become resident");
        let (events, snapshots) = self.server.drain_events_with_snapshots::<Value>().unwrap();
        assert_eq!(events.iter().filter(|event| event.is_added()).count(), 1);
        assert_eq!(snapshots.len(), 1);
        handle
    }

    fn assert_value(&self, handle: &UntypedHandle, expected: &[u8]) {
        assert_eq!(
            self.server.with_asset(
                handle.id().typed::<Value>().expect("resident Value type"),
                |value| value.0.clone(),
            ),
            Some(expected.to_vec())
        );
    }

    fn assert_no_reload_publication(&self) {
        let (events, snapshots) = self.server.drain_events_with_snapshots::<Value>().unwrap();
        assert!(
            snapshots.is_empty(),
            "cancelled reload published a snapshot"
        );
        assert!(
            events.iter().all(|event| !event.is_modified()),
            "cancelled reload emitted Modified: {events:?}"
        );
    }

    fn assert_clean(&self) {
        let state = self.scheduler.state.lock().unwrap();
        assert!(state.in_flight.is_empty());
        assert!(state.running.is_empty());
        assert!(state.queued.is_empty());
        assert!(state.admission_waiters.is_empty());
        assert!(state.dependency_waits.is_empty());
        assert!(state.dependency_parents.is_empty());
    }

    fn assert_unpublished(&self) {
        let (events, snapshots) = self.server.drain_events_with_snapshots::<Value>().unwrap();
        assert!(snapshots.is_empty(), "cancelled output became resident");
        assert!(
            events.iter().all(|event| !matches!(
                event,
                AssetEvent::Added { .. }
                    | AssetEvent::Modified { .. }
                    | AssetEvent::LoadedWithDependencies { .. }
            )),
            "cancelled output emitted a publication event: {events:?}"
        );
    }
}

impl Drop for Scenario {
    fn drop(&mut self) {
        // Assertion failures must not leave workers stuck in test fixtures.
        self.control.release_all();
    }
}

#[derive(Default)]
struct WakeSignal {
    generation: Mutex<u64>,
    changed: Condvar,
}

impl Wake for WakeSignal {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        *self.generation.lock().unwrap() += 1;
        self.changed.notify_all();
    }
}

impl WakeSignal {
    fn generation(&self) -> u64 {
        *self.generation.lock().unwrap()
    }

    fn wait_after(&self, generation: u64, deadline: Instant) {
        let mut current = self.generation.lock().unwrap();
        while *current == generation {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(
                !remaining.is_zero(),
                "future was not woken before its deadline"
            );
            current = self.changed.wait_timeout(current, remaining).unwrap().0;
        }
    }
}

fn finish<F: Future>(future: F) -> F::Output {
    let mut future = Box::pin(future);
    let signal = Arc::new(WakeSignal::default());
    let waker = Waker::from(Arc::clone(&signal));
    let mut cx = Context::from_waker(&waker);
    let deadline = Instant::now() + WAIT;
    loop {
        let generation = signal.generation();
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(output) => return output,
            Poll::Pending => signal.wait_after(generation, deadline),
        }
        assert!(
            Instant::now() < deadline,
            "future did not finish before its deadline"
        );
    }
}

struct FlightObserver<'a> {
    flight: &'a Flight,
    token: Arc<()>,
}

impl Future for FlightObserver<'_> {
    type Output = ScheduleOutcome;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.flight.poll(&self.token, cx)
    }
}

impl Drop for FlightObserver<'_> {
    fn drop(&mut self) {
        self.flight.unregister(&self.token);
    }
}

fn observe(flight: &Flight) -> FlightObserver<'_> {
    FlightObserver {
        flight,
        token: Arc::new(()),
    }
}

fn assert_pending<F: Future>(future: F) {
    let mut future = Box::pin(future);
    let waker = Waker::from(Arc::new(WakeSignal::default()));
    assert!(future
        .as_mut()
        .poll(&mut Context::from_waker(&waker))
        .is_pending());
}

fn protocol_flight() -> Arc<Flight> {
    Arc::new(Flight {
        notification: Mutex::new(FlightNotification {
            result: None,
            waiters: Vec::new(),
        }),
        priority: FlightPriority::new(Priority::Normal),
        requested_type: Some(AssetTypeId::of::<Value>()),
        cancellation: RangeAssemblyCancellation::new(),
        lifecycle: Mutex::new(FlightLifecycle::default()),
    })
}

// Start only while the caller owns lifecycle. A diagnostic try_lock proves
// the mutex is unavailable to this thread before its actual operation. It
// does not instrument lock entry inside attach/detach. Both rendezvous and
// completion are bounded, including before join.
struct LifecycleContender<T> {
    blocked: Arc<WakeSignal>,
    done: Arc<WakeSignal>,
    thread: std::thread::JoinHandle<T>,
}

impl<T: Send + 'static> LifecycleContender<T> {
    fn start(flight: Arc<Flight>, action: impl FnOnce(&Flight) -> T + Send + 'static) -> Self {
        let blocked = Arc::new(WakeSignal::default());
        let done = Arc::new(WakeSignal::default());
        let thread_blocked = Arc::clone(&blocked);
        let thread_done = Arc::clone(&done);
        let thread = std::thread::spawn(move || {
            assert!(matches!(
                flight.lifecycle.try_lock(),
                Err(std::sync::TryLockError::WouldBlock)
            ));
            thread_blocked.wake_by_ref();
            let result = action(&flight);
            thread_done.wake_by_ref();
            result
        });
        Self {
            blocked,
            done,
            thread,
        }
    }

    fn wait_blocked(&self) {
        self.blocked.wait_after(0, Instant::now() + WAIT);
    }

    fn finish(self) -> T {
        self.done.wait_after(0, Instant::now() + WAIT);
        self.thread.join().expect("lifecycle contender panicked")
    }
}

// Protocol fixture, NOT execute_async_load: stage a real primary and label,
// then call the production lifecycle gate with real AssetServer publication.
// This verifies phase/consumer behavior and actual server events, but cannot
// detect removal of the production worker's final lifecycle lock/check.
fn staged_protocol_commit() -> (
    AssetServer,
    Box<crate::server::AsyncLoadPlan>,
    FullLoadOutput,
) {
    let node = Node {
        labeled: true,
        ..Node::leaf(ROOT)
    };
    let control = Arc::new(Control::new(&[node]));
    let server = AssetServer::new();
    server.register_loader(Loader {
        nodes: [(ROOT, node)].into_iter().collect(),
        control,
    });
    let plan = match server
        .prepare_async_load(ROOT.into(), Some(AssetTypeId::of::<Value>()))
        .unwrap()
    {
        AsyncLoadPreparation::New(plan) => plan,
        AsyncLoadPreparation::Existing(_) => panic!("expected a fresh protocol reservation"),
    };
    let output = plan
        .loaders
        .load_for_type(&plan.path, plan.type_id, &[4, 5, 0, 1, 2, 3, 6, 7])
        .unwrap();
    (server, plan, output)
}

fn commit_protocol(
    flight: &Flight,
    server: &AssetServer,
    plan: Box<crate::server::AsyncLoadPlan>,
    output: FullLoadOutput,
    before_publication: impl FnOnce(),
) -> ScheduleOutcome {
    if !flight.begin_commit() {
        server.fail_async_load(&plan, "asset load cancelled");
        return Err(LoadFailure::Cancelled);
    }
    // A fixture-only rendezvous after the real gate returns, not a hook in
    // execute_async_load. It must be possible to detach before publication.
    before_publication();
    server
        .commit_async_load(*plan, output, Vec::new(), vec![Vec::new()])
        .map_err(LoadFailure::AssemblyFailed)
}

#[derive(Default)]
struct CommitDropState {
    ticket: Option<Weak<LoadTicket>>,
    entered: bool,
    open: bool,
    finished: bool,
    lifecycle_available: bool,
    phase: Option<FlightPhase>,
    cancel_returned: bool,
    panics: usize,
    timed_out: bool,
}

struct CommitDropProbe {
    state: Mutex<CommitDropState>,
    changed: Condvar,
    fired: AtomicBool,
    panic_after_cancel: bool,
}

impl CommitDropProbe {
    fn new(panic_after_cancel: bool) -> Self {
        Self {
            state: Mutex::new(CommitDropState::default()),
            changed: Condvar::new(),
            fired: AtomicBool::new(false),
            panic_after_cancel,
        }
    }

    fn arm(&self, ticket: &Arc<LoadTicket>) {
        self.state.lock().unwrap().ticket = Some(Arc::downgrade(ticket));
    }

    fn wait_for(&self, predicate: impl Fn(&CommitDropState) -> bool) {
        let deadline = Instant::now() + WAIT;
        let mut state = self.state.lock().unwrap();
        while !predicate(&state) {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero(), "commit Drop rendezvous timed out");
            state = self.changed.wait_timeout(state, remaining).unwrap().0;
        }
    }

    fn release(&self) {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .open = true;
        self.changed.notify_all();
    }

    fn on_drop(&self) {
        if self.fired.swap(true, Ordering::AcqRel) {
            return;
        }
        let ticket = self
            .state
            .lock()
            .unwrap()
            .ticket
            .as_ref()
            .and_then(Weak::upgrade)
            .expect("Drop probe must retain access to its real ticket");
        let (available, phase) = match ticket.flight.lifecycle.try_lock() {
            Ok(lifecycle) => (true, Some(lifecycle.phase)),
            Err(_) => (false, None),
        };
        {
            let deadline = Instant::now() + GATE_WAIT;
            let mut state = self.state.lock().unwrap();
            state.lifecycle_available = available;
            state.phase = phase;
            state.entered = true;
            self.changed.notify_all();
            while !state.open {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    state.timed_out = true;
                    break;
                }
                state = self.changed.wait_timeout(state, remaining).unwrap().0;
            }
        }
        // An old worker holds lifecycle across commit. Record that regression
        // instead of deadlocking its worker in cancel(). The test MUST assert
        // availability, Committing, and that the real cancel returned.
        if available {
            // Reenter only the ticket, never AssetServer: rejection can drop
            // staged labels while the server's own mutex is still held.
            ticket.cancel();
        }
        {
            let mut state = self.state.lock().unwrap();
            state.cancel_returned = available;
            state.finished = true;
            if available && self.panic_after_cancel {
                state.panics += 1;
            }
            self.changed.notify_all();
        }
        // Panic after releasing the probe mutex, and only once even if an
        // incidental payload clone also shares this probe during cleanup.
        if available && self.panic_after_cancel {
            panic!("single cancellation-test asset Drop panic");
        }
    }
}

#[derive(Clone)]
struct CommitDropValue {
    bytes: Vec<u8>,
    probe: Option<Arc<CommitDropProbe>>,
}

impl Asset for CommitDropValue {
    const TYPE_NAME: &'static str = "scheduler_cancellation_tests::CommitDropValue";
}

impl Drop for CommitDropValue {
    fn drop(&mut self) {
        if let Some(probe) = &self.probe {
            probe.on_drop();
        }
    }
}

struct CommitDropLoader {
    base: Loader,
    probe: Arc<CommitDropProbe>,
    reject_once: Arc<AtomicBool>,
}

impl AssetLoader for CommitDropLoader {
    type Asset = CommitDropValue;

    fn extensions(&self) -> &'static [&'static str] {
        &["cancel"]
    }

    fn read_plan(&self, path: &AssetPath, meta: AssetMeta) -> Result<AssetReadPlan, LoadError> {
        self.base.read_plan(path, meta)
    }

    fn load(&self, bytes: &[u8], ctx: &mut LoadContext) -> Result<Self::Asset, LoadError> {
        let decoded = self.base.load(bytes, ctx)?;
        if ctx.path().path() == ROOT && self.reject_once.swap(false, Ordering::AcqRel) {
            // Both declarations are legal loader output. Only the real server
            // commit rejects their duplicate identity, after staging the first
            // payload; its Drop therefore runs within the production commit.
            ctx.add_labeled_asset(
                "duplicate",
                CommitDropValue {
                    bytes: Vec::new(),
                    probe: Some(Arc::clone(&self.probe)),
                },
            );
            ctx.add_labeled_asset(
                "duplicate",
                CommitDropValue {
                    bytes: Vec::new(),
                    probe: None,
                },
            );
        }
        Ok(CommitDropValue {
            bytes: decoded.0,
            probe: None,
        })
    }
}

struct CommitDropScenario {
    scenario: Scenario,
    probe: Arc<CommitDropProbe>,
    reject_once: Arc<AtomicBool>,
}

impl CommitDropScenario {
    fn new(panic_after_cancel: bool) -> Self {
        Self::with_initial_rejection(panic_after_cancel, true)
    }

    fn for_reload() -> Self {
        Self::with_initial_rejection(false, false)
    }

    fn with_initial_rejection(panic_after_cancel: bool, reject_initial: bool) -> Self {
        let nodes = [Node::leaf(ROOT).decoding(), Node::leaf(OTHER)];
        let control = Arc::new(Control::new(&nodes));
        let probe = Arc::new(CommitDropProbe::new(panic_after_cancel));
        let reject_once = Arc::new(AtomicBool::new(reject_initial));
        let server = Arc::new(AssetServer::new());
        server.register_loader(CommitDropLoader {
            base: Loader {
                nodes: nodes.iter().map(|node| (node.path, *node)).collect(),
                control: Arc::clone(&control),
            },
            probe: Arc::clone(&probe),
            reject_once: Arc::clone(&reject_once),
        });
        server.mount(
            "commit-drop-test",
            Arc::new(Source(Arc::clone(&control))),
            0,
        );
        let scheduler = LoadScheduler::with_io_cpu_pools(
            Arc::clone(&server),
            TaskPool::with_threads(4),
            TaskPool::with_threads(1),
            1,
        );
        Self {
            scenario: Scenario {
                control,
                scheduler,
                server,
            },
            probe,
            reject_once,
        }
    }

    fn arm_rejection(&self) {
        self.reject_once.store(true, Ordering::Release);
    }
}

impl Drop for CommitDropScenario {
    fn drop(&mut self) {
        self.probe.release();
    }
}

struct CommitDropGateRelease<'a>(&'a CommitDropProbe);

impl Drop for CommitDropGateRelease<'_> {
    fn drop(&mut self) {
        self.0.release();
    }
}

fn assert_worker_commit_drop_recovery(panic_after_cancel: bool) {
    let fixture = CommitDropScenario::new(panic_after_cancel);
    let scenario = &fixture.scenario;
    let ticket = Arc::new(scenario.schedule(ROOT));
    // Declared after ticket: on assertion failure release the destructor
    // before dropping the ticket (which itself may need lifecycle).
    let _release_on_unwind = CommitDropGateRelease(&fixture.probe);
    scenario
        .control
        .wait_for(ROOT, |activity| activity.decodes == 1);
    fixture.probe.arm(&ticket);
    let reserved = match scenario
        .server
        .prepare_async_load(ROOT.into(), Some(AssetTypeId::of::<CommitDropValue>()))
        .unwrap()
    {
        AsyncLoadPreparation::Existing(handle) => handle,
        AsyncLoadPreparation::New(_) => panic!("worker did not reserve its asset"),
    };
    let flight = Arc::clone(&ticket.flight);
    scenario.control.release_decode(ROOT);
    fixture.probe.wait_for(|state| state.entered);
    let (lifecycle_available, phase, finished) = {
        let state = fixture.probe.state.lock().unwrap();
        (state.lifecycle_available, state.phase, state.finished)
    };
    // Do not poison the probe mutex if a regression trips these assertions.
    assert!(
        lifecycle_available,
        "worker held lifecycle inside asset Drop"
    );
    assert!(phase == Some(FlightPhase::Committing));
    assert!(!finished);
    // The destructor is now gated inside server.commit_async_load. Do not
    // call server APIs here: the rejection path may still own server.inner.
    assert_pending(observe(&flight));
    fixture.probe.release();
    fixture.probe.wait_for(|state| state.finished);
    let outcome = finish(observe(&flight));
    if panic_after_cancel {
        assert_eq!(outcome, Err(LoadFailure::WorkerPanicked));
    } else {
        assert!(matches!(
            outcome,
            Err(LoadFailure::AssemblyFailed(ref reason))
                if reason.contains("duplicate labeled asset")
        ));
    }
    {
        let state = fixture.probe.state.lock().unwrap();
        assert!(state.cancel_returned);
        assert!(!state.timed_out);
        assert_eq!(state.panics, usize::from(panic_after_cancel));
    }
    assert!(ticket.is_cancelled());
    assert!(
        !flight.is_cancelled(),
        "Committing must win over last detach"
    );
    {
        let lifecycle = flight.lifecycle.lock().expect("Drop poisoned lifecycle");
        assert_eq!(lifecycle.consumers, 0);
        assert!(lifecycle.phase == FlightPhase::Finished);
    }
    assert!(scenario.server.load_state(reserved.id()).is_failed());
    let (events, snapshots) = scenario
        .server
        .drain_events_with_snapshots::<CommitDropValue>()
        .unwrap();
    assert!(snapshots.is_empty());
    assert!(events.iter().all(|event| !event.is_added()
        && !event.is_modified()
        && !event.is_loaded_with_dependencies()));
    scenario.assert_clean();
    drop(reserved);
    drop(ticket);

    // The same scheduler, same one-worker CPU pool and capacity-one admission
    // must support both a retry of the rejected path and an unrelated load.
    for path in [ROOT, OTHER] {
        let next = scenario.schedule(path);
        assert!(!Arc::ptr_eq(&next.flight, &flight));
        let handle = finish(next).unwrap();
        assert!(scenario.server.is_loaded(handle.id()));
        assert_eq!(
            scenario
                .server
                .with_asset(handle.id().typed::<CommitDropValue>().unwrap(), |value| {
                    value.bytes.clone()
                }),
            Some(vec![4, 5, 0, 1, 2, 3, 6, 7])
        );
        scenario.assert_clean();
    }
    assert_eq!(
        fixture.probe.state.lock().unwrap().panics,
        usize::from(panic_after_cancel)
    );
    assert_eq!(scenario.control.activity.lock().unwrap()[ROOT].decodes, 2);
    assert_eq!(scenario.control.activity.lock().unwrap()[OTHER].decodes, 1);
}

#[test]
fn worker_commit_asset_drop_can_reenter_last_ticket_cancel_without_deadlock() {
    assert_worker_commit_drop_recovery(false);
}

#[test]
fn worker_commit_single_asset_drop_panic_does_not_poison_scheduler_or_prevent_retry() {
    assert_worker_commit_drop_recovery(true);
}

#[test]
fn lifecycle_attach_before_last_detach_keeps_the_new_consumer_active() {
    let flight = protocol_flight();
    assert!(flight.attach_consumer());
    let worker = Arc::new(WakeSignal::default());
    let waker = Waker::from(Arc::clone(&worker));
    assert!(!flight.poll_cancelled(&Context::from_waker(&waker)));
    // The join linearizes first. The former last consumer then contends on
    // the lifecycle mutex; it must observe the already-attached survivor.
    assert!(flight.attach_consumer());
    let lifecycle = flight.lifecycle.lock().unwrap();
    assert_eq!(lifecycle.consumers, 2);
    let detach = LifecycleContender::start(Arc::clone(&flight), Flight::detach_consumer);
    detach.wait_blocked();
    assert_eq!(worker.generation(), 0);
    drop(lifecycle);
    detach.finish();
    {
        let lifecycle = flight.lifecycle.lock().unwrap();
        assert_eq!(lifecycle.consumers, 1);
        assert!(lifecycle.phase == FlightPhase::Active);
        assert!(lifecycle.worker_waker.is_some());
    }
    assert!(!flight.is_cancelled());
    assert_eq!(worker.generation(), 0);
    flight.detach_consumer();
    worker.wait_after(0, Instant::now() + WAIT);
    assert_eq!(worker.generation(), 1);
    assert!(flight.is_cancelled());
    assert!(!flight.attach_consumer());
    assert_eq!(flight.lifecycle.lock().unwrap().consumers, 0);
}

#[test]
fn lifecycle_last_detach_before_attach_refuses_without_resurrecting_the_flight() {
    let flight = protocol_flight();
    assert!(flight.attach_consumer());
    let worker = Arc::new(WakeSignal::default());
    let waker = Waker::from(Arc::clone(&worker));
    assert!(!flight.poll_cancelled(&Context::from_waker(&waker)));
    flight.detach_consumer();
    worker.wait_after(0, Instant::now() + WAIT);
    let lifecycle = flight.lifecycle.lock().unwrap();
    assert!(lifecycle.phase == FlightPhase::Cancelling);
    assert_eq!(lifecycle.consumers, 0);
    assert!(lifecycle.worker_waker.is_none());
    let attach = LifecycleContender::start(Arc::clone(&flight), Flight::attach_consumer);
    attach.wait_blocked();
    drop(lifecycle);
    assert!(!attach.finish());
    assert!(flight.is_cancelled());
    assert!(flight.poll_cancelled(&Context::from_waker(&waker)));
    assert_eq!(worker.generation(), 1);
    {
        let lifecycle = flight.lifecycle.lock().unwrap();
        assert!(lifecycle.phase == FlightPhase::Cancelling);
        assert_eq!(lifecycle.consumers, 0);
        assert!(lifecycle.worker_waker.is_none());
    }
    Flight::wake_all(flight.complete(Err(LoadFailure::Cancelled)));
    assert!(flight.lifecycle.lock().unwrap().phase == FlightPhase::Finished);
    assert!(flight.is_cancelled());
    assert_eq!(finish(observe(&flight)), Err(LoadFailure::Cancelled));
}

#[test]
fn lifecycle_cancel_before_commit_refuses_real_primary_and_label_publication() {
    let (server, plan, output) = staged_protocol_commit();
    let reserved = plan.handle.clone();
    let flight = protocol_flight();
    assert!(flight.attach_consumer());
    let lifecycle = flight.lifecycle.lock().unwrap();
    let detach = LifecycleContender::start(Arc::clone(&flight), Flight::detach_consumer);
    detach.wait_blocked();
    drop(lifecycle);
    // Last detach must finish before the commit gate is allowed to run.
    detach.finish();
    {
        let lifecycle = flight.lifecycle.lock().unwrap();
        assert!(lifecycle.phase == FlightPhase::Cancelling);
        assert_eq!(lifecycle.consumers, 0);
    }
    let outcome = commit_protocol(&flight, &server, plan, output, || {
        panic!("cancel-first must refuse before reaching server publication");
    });
    assert_eq!(outcome, Err(LoadFailure::Cancelled));
    assert!(!flight.attach_consumer());
    assert!(flight.is_cancelled());
    assert!(server.load_state(reserved.id()).is_failed());
    let (events, snapshots) = server.drain_events_with_snapshots::<Value>().unwrap();
    assert!(snapshots.is_empty());
    assert!(events.iter().all(|event| !event.is_added()
        && !event.is_modified()
        && !event.is_loaded_with_dependencies()));
    Flight::wake_all(flight.complete(outcome));
    assert!(flight.lifecycle.lock().unwrap().phase == FlightPhase::Finished);
    assert_eq!(finish(observe(&flight)), Err(LoadFailure::Cancelled));
}

#[test]
fn lifecycle_commit_before_last_detach_preserves_real_publication_and_outcome() {
    let (server, plan, output) = staged_protocol_commit();
    let flight = protocol_flight();
    assert!(flight.attach_consumer());
    let worker = Arc::new(WakeSignal::default());
    let waker = Waker::from(Arc::clone(&worker));
    assert!(!flight.poll_cancelled(&Context::from_waker(&waker)));
    let reserved = plan.handle.clone();
    let outcome = commit_protocol(&flight, &server, plan, output, || {
        // begin_commit has returned with the phase committed but the mutex
        // released. Last detach must finish BEFORE the real server call.
        let lifecycle = flight
            .lifecycle
            .try_lock()
            .expect("commit gate kept its lock");
        assert!(lifecycle.phase == FlightPhase::Committing);
        assert_eq!(lifecycle.consumers, 1);
        let detach = LifecycleContender::start(Arc::clone(&flight), Flight::detach_consumer);
        detach.wait_blocked();
        drop(lifecycle);
        detach.finish();
        assert!(!flight.is_cancelled());
        assert!(server.load_state(reserved.id()).is_loading());
        assert_pending(observe(&flight));
        let lifecycle = flight.lifecycle.lock().unwrap();
        assert_eq!(lifecycle.consumers, 0);
        assert!(lifecycle.phase == FlightPhase::Committing);
    });
    let published = outcome.as_ref().unwrap().clone();
    assert!(!flight.is_cancelled());
    assert_eq!(worker.generation(), 0);
    {
        let lifecycle = flight.lifecycle.lock().unwrap();
        assert_eq!(lifecycle.consumers, 0);
        assert!(lifecycle.phase == FlightPhase::Committing);
    }
    // Last detach occurred before the server call, not merely before the
    // observer saw success. Completion still publishes exactly this outcome.
    assert_pending(observe(&flight));
    Flight::wake_all(flight.complete(outcome));
    assert_eq!(finish(observe(&flight)).unwrap().id(), published.id());
    assert!(flight.lifecycle.lock().unwrap().worker_waker.is_none());
    assert!(server.is_loaded(published.id()));
    let (events, snapshots) = server.drain_events_with_snapshots::<Value>().unwrap();
    assert_eq!(snapshots.len(), 2, "primary and label must both publish");
    assert_eq!(events.iter().filter(|event| event.is_added()).count(), 2);
    assert_eq!(events.iter().filter(|event| event.is_modified()).count(), 0);
    assert!(snapshots
        .iter()
        .any(|(id, _)| id.untyped() == published.id()));
}

#[test]
fn cancelling_or_dropping_one_joiner_does_not_cancel_the_other_consumer() {
    for explicit_cancel in [false, true] {
        let scenario = Scenario::new(&[Node::leaf(ROOT).reading()], 1);
        let first = scenario.schedule(ROOT);
        scenario
            .control
            .wait_for(ROOT, |activity| activity.active == 2);
        let survivor = scenario.schedule(ROOT);
        let flight = Arc::clone(&first.flight);
        assert!(Arc::ptr_eq(&flight, &survivor.flight));
        if explicit_cancel {
            first.cancel();
            first.cancel();
            assert_eq!(finish(first), Err(LoadFailure::Cancelled));
        } else {
            drop(first);
        }
        assert!(!flight.is_cancelled());
        scenario.control.release_reads(ROOT);
        let handle = finish(survivor).unwrap();
        assert!(scenario.server.is_loaded(handle.id()));
        assert_eq!(
            scenario
                .server
                .with_asset(handle.id().typed::<Value>().unwrap(), |value| value
                    .0
                    .clone()),
            Some(vec![4, 5, 0, 1, 2, 3, 6, 7])
        );
        let state = scenario.control.activity.lock().unwrap();
        assert_eq!(state[ROOT].metadata, 1);
        assert_eq!(state[ROOT].started.len(), 4);
        assert_eq!(state[ROOT].decodes, 1);
        drop(state);
        scenario.assert_clean();
    }
}

#[test]
fn last_consumer_read_cancellation_drains_active_batch_without_decode_and_allows_retry() {
    for explicit_cancel in [false, true] {
        let scenario = Scenario::new(&[Node::leaf(ROOT).reading()], 1);
        let ticket = scenario.schedule(ROOT);
        scenario
            .control
            .wait_for(ROOT, |activity| activity.active == 2);
        let reservation = scenario.reserved(ROOT);
        let flight = Arc::clone(&ticket.flight);
        if explicit_cancel {
            ticket.cancel();
            ticket.cancel();
            assert_eq!(finish(ticket), Err(LoadFailure::Cancelled));
        } else {
            drop(ticket);
        }
        assert!(flight.is_cancelled());
        assert_eq!(
            scenario.scheduler.schedule(ROOT, Priority::Critical).err(),
            Some(ScheduleError::Cancelling { path: ROOT.into() })
        );
        // Finish one part only: the flight must retain its slot while the
        // other blocking read is live, and must not admit a second batch.
        scenario.control.release_part(ROOT, 4);
        scenario
            .control
            .wait_for(ROOT, |activity| activity.completed.len() == 1);
        {
            let state = scenario.control.activity.lock().unwrap();
            assert_eq!(state[ROOT].active, 1);
            assert_eq!(state[ROOT].started.len(), 2);
        }
        assert_pending(observe(&flight));
        assert_eq!(scenario.scheduler.in_flight(), 1);
        scenario.control.release_reads(ROOT);
        assert_eq!(finish(observe(&flight)), Err(LoadFailure::Cancelled));
        {
            let state = scenario.control.activity.lock().unwrap();
            assert_eq!(state[ROOT].active, 0);
            assert_eq!(state[ROOT].completed.len(), 2);
            assert_eq!(state[ROOT].started.len(), 2);
            assert_eq!(state[ROOT].decodes, 0);
        }
        assert!(scenario.server.load_state(reservation.id()).is_failed());
        assert!(scenario
            .server
            .recursive_state(reservation.id())
            .is_failed());
        scenario.assert_unpublished();
        scenario.assert_clean();
        // Release the diagnostic handle too; it must not pin a failed path's
        // old reservation while verifying fresh post-drain admission.
        drop(reservation);
        let retry = scenario.schedule(ROOT);
        assert!(!Arc::ptr_eq(&flight, &retry.flight));
        let handle = finish(retry).unwrap();
        assert!(scenario.server.is_loaded(handle.id()));
        {
            let state = scenario.control.activity.lock().unwrap();
            assert_eq!(state[ROOT].metadata, 2);
            assert_eq!(state[ROOT].started.len(), 6);
            assert_eq!(state[ROOT].decodes, 1);
        }
        scenario.assert_clean();
    }
}

#[test]
fn last_consumer_leaving_before_worker_dispatch_performs_no_io_or_decode() {
    let scenario = Scenario::new(&[Node::leaf(ROOT).decoding(), Node::leaf(OTHER)], 2);
    let blocker = scenario.schedule(ROOT);
    scenario
        .control
        .wait_for(ROOT, |activity| activity.decodes == 1);
    let ticket = scenario.schedule(OTHER);
    let flight = Arc::clone(&ticket.flight);
    drop(ticket);
    scenario.control.assert_unread(OTHER);
    scenario.control.release_decode(ROOT);
    finish(blocker).unwrap();
    assert_eq!(finish(observe(&flight)), Err(LoadFailure::Cancelled));
    scenario.control.assert_unread(OTHER);
    scenario.assert_clean();
}

#[test]
fn cancelling_during_decode_never_schedules_ordinary_or_labeled_dependencies_or_publishes() {
    for labeled in [false, true] {
        let scenario = Scenario::new(
            &[
                Node::leaf(ROOT).decoding().parent(labeled),
                Node::leaf(CHILD),
            ],
            1,
        );
        let ticket = scenario.schedule(ROOT);
        scenario
            .control
            .wait_for(ROOT, |activity| activity.decodes == 1);
        let reservation = scenario.reserved(ROOT);
        let flight = Arc::clone(&ticket.flight);
        ticket.cancel();
        assert_eq!(finish(ticket), Err(LoadFailure::Cancelled));
        assert_pending(observe(&flight));
        scenario.control.assert_unread(CHILD);
        scenario.control.release_decode(ROOT);
        assert_eq!(finish(observe(&flight)), Err(LoadFailure::Cancelled));
        scenario.control.assert_unread(CHILD);
        assert!(scenario.server.load_state(reservation.id()).is_failed());
        assert!(scenario
            .server
            .recursive_state(reservation.id())
            .is_failed());
        scenario.assert_unpublished();
        scenario.assert_clean();
    }
}

#[test]
fn cancelled_parent_exits_dependency_wait_before_unshared_child_read_drains() {
    for labeled in [false, true] {
        let scenario = Scenario::new(
            &[
                Node::leaf(ROOT).parent(labeled),
                Node::leaf(CHILD).reading(),
            ],
            1,
        );
        let parent = scenario.schedule(ROOT);
        scenario
            .control
            .wait_for(CHILD, |activity| activity.active == 2);
        let reservation = scenario.reserved(ROOT);
        let parent_flight = Arc::clone(&parent.flight);
        let parent_key = FlightKey::Load(ROOT.into());
        let child_key = FlightKey::Load(CHILD.into());
        let child_flight = {
            let state = scenario.scheduler.state.lock().unwrap();
            assert!(state.dependency_waits[&parent_key].contains(&child_key));
            assert!(state.dependency_parents[&child_key].contains(&parent_key));
            Arc::clone(&state.in_flight[&child_key])
        };
        assert_eq!(child_flight.waiter_count(), 1);
        parent.cancel();
        assert_eq!(finish(parent), Err(LoadFailure::Cancelled));
        // Do not release the child's gate until parent completion is proven.
        assert_eq!(finish(observe(&parent_flight)), Err(LoadFailure::Cancelled));
        assert!(scenario.server.load_state(reservation.id()).is_failed());
        assert!(child_flight.is_cancelled());
        assert_eq!(child_flight.waiter_count(), 0);
        {
            let state = scenario.scheduler.state.lock().unwrap();
            assert!(!state.in_flight.contains_key(&parent_key));
            assert!(!state.running.contains_key(&parent_key));
            assert!(state.dependency_waits.is_empty());
            assert!(state.dependency_parents.is_empty());
            assert_eq!(state.in_flight.len(), 1);
            assert!(Arc::ptr_eq(&state.in_flight[&child_key], &child_flight));
        }
        assert_eq!(scenario.control.activity.lock().unwrap()[CHILD].active, 2);
        assert_pending(observe(&child_flight));
        scenario.control.release_reads(CHILD);
        assert_eq!(finish(observe(&child_flight)), Err(LoadFailure::Cancelled));
        assert_eq!(scenario.control.activity.lock().unwrap()[CHILD].decodes, 0);
        scenario.assert_unpublished();
        scenario.assert_clean();
    }
}

#[test]
fn cancelled_parent_removes_wait_but_shared_child_survives() {
    for labeled in [false, true] {
        let scenario = Scenario::new(
            &[
                Node::leaf(ROOT).parent(labeled),
                Node::leaf(CHILD).reading(),
            ],
            1,
        );
        let parent = scenario.schedule(ROOT);
        scenario
            .control
            .wait_for(CHILD, |activity| activity.active == 2);
        let reservation = scenario.reserved(ROOT);
        let parent_flight = Arc::clone(&parent.flight);
        let child = scenario.schedule(CHILD);
        let child_flight = Arc::clone(&child.flight);
        assert_eq!(child_flight.waiter_count(), 1);
        parent.cancel();
        assert_eq!(finish(parent), Err(LoadFailure::Cancelled));
        assert_eq!(finish(observe(&parent_flight)), Err(LoadFailure::Cancelled));
        assert!(!child_flight.is_cancelled());
        assert_eq!(child_flight.waiter_count(), 0);
        {
            let state = scenario.scheduler.state.lock().unwrap();
            assert!(state.dependency_waits.is_empty());
            assert!(state.dependency_parents.is_empty());
            assert_eq!(state.in_flight.len(), 1);
        }
        assert_eq!(scenario.control.activity.lock().unwrap()[CHILD].active, 2);
        scenario.control.release_reads(CHILD);
        let handle = finish(child).unwrap();
        assert!(scenario.server.is_loaded(handle.id()));
        assert!(scenario.server.load_state(reservation.id()).is_failed());
        let (events, snapshots) = scenario
            .server
            .drain_events_with_snapshots::<Value>()
            .unwrap();
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].0.untyped(), handle.id());
        assert!(events
            .iter()
            .filter(|event| event.id().untyped() == reservation.id())
            .all(|event| !event.is_added()
                && !event.is_modified()
                && !event.is_loaded_with_dependencies()));
        assert_eq!(scenario.control.activity.lock().unwrap()[CHILD].decodes, 1);
        scenario.assert_clean();
    }
}

#[test]
fn cancelled_execution_slot_wait_wakes_and_unregisters_before_capacity_is_available() {
    let scenario = Scenario::new(
        &[Node::leaf(ROOT).reading(), Node::leaf(OTHER).reading()],
        1,
    );
    let ticket = scenario.schedule(ROOT);
    scenario
        .control
        .wait_for(ROOT, |activity| activity.active == 2);
    let flight = Arc::clone(&ticket.flight);
    let key = FlightKey::Load(ROOT.into());
    // Model the slot handoff performed by a dependency-bearing assembly. The
    // blocked I/O keeps both real flights alive while the private slot future
    // is driven directly, without needing an extra scheduler test hook.
    scenario.scheduler.release_execution_slot(&key, &flight);
    let blocker = scenario.schedule(OTHER);
    scenario
        .control
        .wait_for(OTHER, |activity| activity.active == 2);
    let mut slot = Box::pin(scenario.scheduler.acquire_execution_slot(
        key.clone(),
        Arc::clone(&flight),
        Priority::Normal,
    ));
    let signal = Arc::new(WakeSignal::default());
    let waker = Waker::from(Arc::clone(&signal));
    assert!(slot
        .as_mut()
        .poll(&mut Context::from_waker(&waker))
        .is_pending());
    {
        let state = scenario.scheduler.state.lock().unwrap();
        assert_eq!(state.admission_waiters.len(), 1);
        assert_eq!(state.running.len(), 1);
        assert!(!state.running.contains_key(&key));
    }
    let generation = signal.generation();
    ticket.cancel();
    // Prove cancellation wakes an already-registered waiter; polling again
    // alone could hide a missing cancellation notification.
    signal.wait_after(generation, Instant::now() + WAIT);
    assert_eq!(finish(slot), Err(LoadFailure::Cancelled));
    {
        let state = scenario.scheduler.state.lock().unwrap();
        assert!(state.admission_waiters.is_empty());
        assert_eq!(state.running.len(), 1);
        assert!(!state.running.contains_key(&key));
        assert!(state.running.contains_key(&FlightKey::Load(OTHER.into())));
    }
    assert_eq!(scenario.control.activity.lock().unwrap()[ROOT].active, 2);
    assert_eq!(scenario.control.activity.lock().unwrap()[OTHER].active, 2);
    assert_pending(observe(&flight));
    scenario.control.release_reads(ROOT);
    assert_eq!(finish(observe(&flight)), Err(LoadFailure::Cancelled));
    assert_eq!(scenario.control.activity.lock().unwrap()[ROOT].decodes, 0);
    scenario.control.release_reads(OTHER);
    finish(blocker).unwrap();
    scenario.assert_clean();
}

#[test]
fn cancellation_after_commit_cannot_change_the_published_flight_outcome() {
    let scenario = Scenario::new(&[Node::leaf(ROOT).decoding()], 1);
    let ticket = scenario.schedule(ROOT);
    scenario
        .control
        .wait_for(ROOT, |activity| activity.decodes == 1);
    let flight = Arc::clone(&ticket.flight);
    scenario.control.release_decode(ROOT);
    // Observe completion without consuming the ticket, so its subsequent
    // cancellation exercises the committed lifecycle, not a detached ticket.
    let published = finish(observe(&flight)).unwrap();
    ticket.cancel();
    ticket.cancel();
    drop(ticket);
    assert_eq!(finish(observe(&flight)).unwrap().id(), published.id());
    assert!(scenario.server.is_loaded(published.id()));
    assert_eq!(scenario.control.activity.lock().unwrap()[ROOT].decodes, 1);
    let (events, snapshots) = scenario
        .server
        .drain_events_with_snapshots::<Value>()
        .unwrap();
    assert_eq!(snapshots.len(), 1);
    assert_eq!(events.iter().filter(|event| event.is_added()).count(), 1);
    scenario.assert_clean();
}

#[test]
fn cancelled_reload_drains_shared_read_without_decoding_or_replacing_resident_revision() {
    for explicit_cancel in [false, true] {
        let scenario = Scenario::new(&[Node::leaf(ROOT).reading()], 1);
        let resident = scenario.make_resident(ROOT);
        scenario.assert_value(&resident, INITIAL_BYTES);
        scenario.control.reset(ROOT, false, true, 32);

        let ticket = scenario
            .scheduler
            .schedule_reload(ROOT, Priority::Normal)
            .unwrap();
        scenario
            .control
            .wait_for(ROOT, |activity| activity.active == 2);
        let flight = Arc::clone(&ticket.flight);
        if explicit_cancel {
            ticket.cancel();
            assert_eq!(finish(ticket), Err(LoadFailure::Cancelled));
        } else {
            drop(ticket);
        }
        assert!(flight.is_cancelled());
        assert_pending(observe(&flight));

        scenario.control.release_reads(ROOT);
        assert_eq!(finish(observe(&flight)), Err(LoadFailure::Cancelled));
        {
            let activity = scenario.control.activity.lock().unwrap();
            assert_eq!(activity[ROOT].metadata, 1);
            assert_eq!(activity[ROOT].started.len(), 4);
            assert_eq!(activity[ROOT].completed.len(), 4);
            assert_eq!(activity[ROOT].decodes, 0);
        }
        scenario.assert_value(&resident, INITIAL_BYTES);
        scenario.assert_no_reload_publication();
        scenario.assert_clean();
    }
}

#[test]
fn cancelled_reload_finishes_synchronous_decode_but_discards_the_candidate() {
    for explicit_cancel in [false, true] {
        let scenario = Scenario::new(&[Node::leaf(ROOT).decoding()], 1);
        let resident = scenario.make_resident(ROOT);
        scenario.assert_value(&resident, INITIAL_BYTES);
        scenario.control.reset(ROOT, true, false, 32);

        let ticket = scenario
            .scheduler
            .schedule_reload(ROOT, Priority::Normal)
            .unwrap();
        scenario
            .control
            .wait_for(ROOT, |activity| activity.decodes == 1);
        let flight = Arc::clone(&ticket.flight);
        if explicit_cancel {
            ticket.cancel();
            assert_eq!(finish(ticket), Err(LoadFailure::Cancelled));
        } else {
            drop(ticket);
        }
        assert!(flight.is_cancelled());
        assert_pending(observe(&flight));

        scenario.control.release_decode(ROOT);
        assert_eq!(finish(observe(&flight)), Err(LoadFailure::Cancelled));
        assert_eq!(scenario.control.activity.lock().unwrap()[ROOT].decodes, 1);
        scenario.assert_value(&resident, INITIAL_BYTES);
        scenario.assert_no_reload_publication();
        scenario.assert_clean();
    }
}

#[test]
fn queued_reload_cancelled_before_dispatch_performs_no_source_or_loader_work() {
    for explicit_cancel in [false, true] {
        let scenario = Scenario::new(&[Node::leaf(ROOT), Node::leaf(OTHER).decoding()], 1);
        let resident = scenario.make_resident(ROOT);
        scenario.control.reset(ROOT, false, false, 32);
        scenario.control.reset(OTHER, true, false, 0);

        let blocker = scenario.schedule(OTHER);
        scenario
            .control
            .wait_for(OTHER, |activity| activity.decodes == 1);
        let ticket = scenario
            .scheduler
            .schedule_reload_typed_with_ancestors(
                ROOT.into(),
                Priority::Normal,
                None,
                [AssetPath::parse("queued-parent.cancel")]
                    .into_iter()
                    .collect(),
            )
            .unwrap();
        let flight = Arc::clone(&ticket.flight);
        {
            let state = scenario.scheduler.state.lock().unwrap();
            assert_eq!(state.queued.len(), 1);
            assert_eq!(state.running.len(), 1);
        }
        if explicit_cancel {
            ticket.cancel();
            assert_eq!(finish(ticket), Err(LoadFailure::Cancelled));
        } else {
            drop(ticket);
        }
        scenario.control.assert_unread(ROOT);

        scenario.control.release_decode(OTHER);
        finish(blocker).unwrap();
        assert_eq!(finish(observe(&flight)), Err(LoadFailure::Cancelled));
        scenario.control.assert_unread(ROOT);
        scenario.assert_value(&resident, INITIAL_BYTES);
        let (events, snapshots) = scenario
            .server
            .drain_events_with_snapshots::<Value>()
            .unwrap();
        assert!(events.iter().all(|event| !event.is_modified()));
        assert!(
            snapshots
                .iter()
                .all(|(id, _)| id.untyped() != resident.id()),
            "queued reload replaced the resident revision"
        );
        scenario.assert_clean();
    }
}

#[test]
fn reload_commit_candidate_drop_can_reenter_ticket_cancel_without_deadlock() {
    let fixture = CommitDropScenario::for_reload();
    let scenario = &fixture.scenario;
    scenario.control.release_all();
    let resident = finish(scenario.schedule(ROOT)).expect("initial revision must load");
    let (events, snapshots) = scenario
        .server
        .drain_events_with_snapshots::<CommitDropValue>()
        .unwrap();
    assert_eq!(events.iter().filter(|event| event.is_added()).count(), 1);
    assert_eq!(snapshots.len(), 1);
    scenario.control.reset(ROOT, true, false, 32);
    fixture.arm_rejection();

    let ticket = Arc::new(
        scenario
            .scheduler
            .schedule_reload(ROOT, Priority::Normal)
            .unwrap(),
    );
    let _release_on_unwind = CommitDropGateRelease(&fixture.probe);
    scenario
        .control
        .wait_for(ROOT, |activity| activity.decodes == 1);
    fixture.probe.arm(&ticket);
    let flight = Arc::clone(&ticket.flight);
    scenario.control.release_decode(ROOT);
    fixture.probe.wait_for(|state| state.entered);
    {
        let state = fixture.probe.state.lock().unwrap();
        assert!(
            state.lifecycle_available,
            "reload commit held lifecycle in Drop"
        );
        assert!(state.phase == Some(FlightPhase::Committing));
        assert!(!state.finished);
    }
    assert_pending(observe(&flight));
    fixture.probe.release();
    fixture.probe.wait_for(|state| state.finished);
    let outcome = finish(observe(&flight));
    assert!(matches!(
        outcome,
        Err(LoadFailure::AssemblyFailed(ref reason))
            if reason.contains("duplicate labeled asset")
    ));
    {
        let state = fixture.probe.state.lock().unwrap();
        assert!(state.cancel_returned);
        assert!(!state.timed_out);
    }
    assert!(ticket.is_cancelled());
    assert!(
        !flight.is_cancelled(),
        "commit gate must win over reentrant cancel"
    );
    {
        let lifecycle = flight.lifecycle.lock().unwrap();
        assert_eq!(lifecycle.consumers, 0);
        assert!(lifecycle.phase == FlightPhase::Finished);
    }
    assert_eq!(
        scenario.server.with_asset(
            resident
                .id()
                .typed::<CommitDropValue>()
                .expect("resident CommitDropValue type"),
            |value| value.bytes.clone(),
        ),
        Some(INITIAL_BYTES.to_vec())
    );
    let (events, snapshots) = scenario
        .server
        .drain_events_with_snapshots::<CommitDropValue>()
        .unwrap();
    assert!(snapshots.is_empty());
    assert!(events.iter().all(|event| !event.is_modified()));
    scenario.assert_clean();

    let unrelated = finish(scenario.schedule(OTHER)).expect("scheduler must remain reusable");
    assert!(scenario.server.is_loaded(unrelated.id()));
    scenario.assert_clean();
}

#[test]
fn soft_batch_cancelling_refusal_is_atomic_before_admission_or_priority_promotion() {
    let scenario = Scenario::new(
        &[
            Node::leaf(ROOT).reading(),
            Node::leaf(OTHER).reading(),
            Node::leaf(FRESH),
        ],
        3,
    );
    let fresh = scenario.server.register_soft_path::<Value>(FRESH).unwrap();
    let live_ref = scenario.server.register_soft_path::<Value>(OTHER).unwrap();
    let cancelling_ref = scenario.server.register_soft_path::<Value>(ROOT).unwrap();
    let cancelling = scenario
        .scheduler
        .schedule(ROOT, Priority::Background)
        .unwrap();
    let live = scenario
        .scheduler
        .schedule(OTHER, Priority::Background)
        .unwrap();
    scenario
        .control
        .wait_for(ROOT, |activity| activity.active == 2);
    scenario
        .control
        .wait_for(OTHER, |activity| activity.active == 2);
    let cancelling_flight = Arc::clone(&cancelling.flight);
    cancelling.cancel();
    assert!(cancelling_flight.is_cancelled());
    let slots_before = scenario.server.total_assets();
    for (references, index) in [
        (vec![fresh, live_ref, cancelling_ref, fresh], 2),
        (vec![fresh, cancelling_ref, cancelling_ref, live_ref], 1),
    ] {
        let error = scenario
            .scheduler
            .schedule_soft_batch(
                &references,
                Priority::Critical,
                SoftLoadBudget {
                    max_requests: 4,
                    max_unique_assets: 3,
                },
            )
            .err()
            .unwrap();
        assert_eq!(
            error,
            SoftBatchError::Cancelling {
                index,
                path: ROOT.into()
            }
        );
        assert_eq!(live.effective_priority(), Priority::Background);
        assert_eq!(
            cancelling_flight.priority.effective_priority(),
            Priority::Background
        );
        assert!(cancelling_flight.is_cancelled());
        assert!(!live.flight.is_cancelled());
        assert_eq!(scenario.server.total_assets(), slots_before);
        scenario.control.assert_unread(FRESH);
        let state = scenario.scheduler.state.lock().unwrap();
        assert_eq!(state.in_flight.len(), 2);
        assert_eq!(state.running.len(), 2);
        assert!(state.queued.is_empty());
        assert!(state.admission_waiters.is_empty());
        assert!(Arc::ptr_eq(
            &state.in_flight[&FlightKey::Load(ROOT.into())],
            &cancelling_flight
        ));
    }
    scenario.control.release_reads(ROOT);
    assert_eq!(finish(cancelling), Err(LoadFailure::Cancelled));
    assert_eq!(
        finish(observe(&cancelling_flight)),
        Err(LoadFailure::Cancelled)
    );
    scenario.control.release_reads(OTHER);
    finish(live).unwrap();
    scenario.control.assert_unread(FRESH);
    assert_eq!(scenario.control.activity.lock().unwrap()[ROOT].decodes, 0);
    scenario.assert_clean();
}
