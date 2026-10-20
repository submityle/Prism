//! Shared-flight wait cycles must terminate even when roots have disjoint ancestors.
#![cfg(feature = "std")]

use std::{
    collections::BTreeMap,
    future::Future,
    pin::Pin,
    sync::{
        mpsc::{self, SyncSender},
        Arc, Condvar, Mutex,
    },
    task::{Context, Poll, Wake, Waker},
    time::{Duration, Instant},
};

use prism_asset::{
    Asset, AssetLoader, AssetPath, AssetServer, DepRequest, LoadContext, LoadError, LoadFailure,
    LoadScheduler, LoadTicket, MemSource, ScheduleOutcome, UntypedHandle,
};
use prism_tasks::{Priority, TaskPool};

const TIMEOUT: Duration = Duration::from_secs(10);
const HEALTHY: &str = "healthy.wait-graph";

#[derive(Debug)]
struct GraphAsset(String);

impl Asset for GraphAsset {
    const TYPE_NAME: &'static str = "scheduler_wait_cycles::GraphAsset";
}

#[derive(Clone, Copy)]
struct Node {
    path: &'static str,
    dependencies: &'static [&'static str],
    label_dependencies: Option<&'static [&'static str]>,
}

impl Node {
    const fn new(path: &'static str, dependencies: &'static [&'static str]) -> Self {
        Self {
            path,
            dependencies,
            label_dependencies: None,
        }
    }
}

#[derive(Default)]
struct GateState {
    entered: usize,
    open: bool,
}

struct DecodeGates {
    states: Mutex<BTreeMap<&'static str, GateState>>,
    changed: Condvar,
}

impl DecodeGates {
    fn enter(&self, path: &str) -> Result<(), LoadError> {
        let mut states = self
            .states
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let Some(state) = states.get_mut(path) else {
            return Ok(());
        };
        state.entered += 1;
        self.changed.notify_all();
        let deadline = Instant::now() + TIMEOUT;
        while !states[path].open {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(LoadError::Other(format!("decoder gate timed out: {path}")));
            }
            let (next, _) = self
                .changed
                .wait_timeout(states, remaining)
                .unwrap_or_else(|error| error.into_inner());
            states = next;
        }
        Ok(())
    }

    fn wait_for_all_decoders(&self) {
        let deadline = Instant::now() + TIMEOUT;
        let mut states = self
            .states
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        while states.values().any(|state| state.entered == 0) {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(
                !remaining.is_zero(),
                "not all independent roots entered decoding: {:?}",
                states
                    .iter()
                    .map(|(path, state)| (*path, state.entered))
                    .collect::<Vec<_>>()
            );
            let (next, _) = self
                .changed
                .wait_timeout(states, remaining)
                .unwrap_or_else(|error| error.into_inner());
            states = next;
        }
        assert!(states.values().all(|state| !state.open));
    }

    fn open_all(&self) {
        let mut states = self
            .states
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        for state in states.values_mut() {
            state.open = true;
        }
        self.changed.notify_all();
    }

    fn assert_decoded_once(&self) {
        let states = self
            .states
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        for (path, state) in states.iter() {
            assert_eq!(state.entered, 1, "duplicate source decode: {path}");
        }
    }
}

struct GraphLoader {
    nodes: BTreeMap<&'static str, Node>,
    gates: Arc<DecodeGates>,
}

impl AssetLoader for GraphLoader {
    type Asset = GraphAsset;

    fn extensions(&self) -> &'static [&'static str] {
        &["wait-graph"]
    }

    fn load(&self, _: &[u8], ctx: &mut LoadContext<'_>) -> Result<GraphAsset, LoadError> {
        let path = ctx.path().path().to_owned();
        self.gates.enter(&path)?;
        if path == HEALTHY {
            return Ok(GraphAsset(path));
        }
        let node = self
            .nodes
            .get(path.as_str())
            .ok_or_else(|| LoadError::Other(format!("unknown graph source: {path}")))?;
        for dependency in node.dependencies {
            ctx.add_dependency(AssetPath::parse(*dependency));
        }
        if let Some(dependencies) = node.label_dependencies {
            ctx.add_labeled_asset_with_deps(
                "part",
                GraphAsset(format!("{path}#part")),
                dependencies
                    .iter()
                    .map(|dependency| DepRequest::new(AssetPath::parse(*dependency)))
                    .collect(),
            );
        }
        Ok(GraphAsset(path))
    }
}

struct Scenario {
    gates: Arc<DecodeGates>,
    scheduler: LoadScheduler,
    server: Arc<AssetServer>,
}

impl Scenario {
    fn new(nodes: &[Node]) -> Self {
        let gates = Arc::new(DecodeGates {
            states: Mutex::new(
                nodes
                    .iter()
                    .map(|node| (node.path, GateState::default()))
                    .collect(),
            ),
            changed: Condvar::new(),
        });
        let server = Arc::new(AssetServer::new());
        server.register_loader(GraphLoader {
            nodes: nodes.iter().map(|node| (node.path, *node)).collect(),
            gates: Arc::clone(&gates),
        });
        let mut source = MemSource::new();
        source.insert(HEALTHY, [0]);
        for node in nodes {
            source.insert(node.path, [0]);
        }
        server.mount("mem", Arc::new(source), 0);
        // Separate pools and one CPU worker per gated decoder make entry
        // independent of whether another root is blocked inside its loader.
        let scheduler = LoadScheduler::with_io_cpu_pools(
            Arc::clone(&server),
            TaskPool::with_threads(2),
            TaskPool::with_threads(nodes.len().max(2)),
            nodes.len() + 2,
        );
        Self {
            gates,
            scheduler,
            server,
        }
    }

    fn schedule(&self, path: &str) -> LoadTicket {
        self.scheduler
            .schedule(AssetPath::parse(path), Priority::Normal)
            .unwrap_or_else(|error| panic!("root {path} was not admitted: {error}"))
    }

    fn assert_recovered(&self) {
        assert_eq!(self.scheduler.in_flight(), 0, "cycle leaked shared flights");
        let handle = wait_tickets(vec![self.schedule(HEALTHY)])
            .pop()
            .unwrap()
            .expect("unrelated load must succeed after a failed cycle");
        self.assert_value(&handle, HEALTHY);
        assert_eq!(self.scheduler.in_flight(), 0);
    }

    fn assert_value(&self, handle: &UntypedHandle, expected: &str) {
        assert!(self.server.is_loaded(handle.id()));
        let typed = handle.typed::<GraphAsset>().expect("graph asset type");
        assert_eq!(
            self.server.with_asset(typed.id(), |asset| asset.0.clone()),
            Some(expected.to_owned())
        );
    }
}

impl Drop for Scenario {
    fn drop(&mut self) {
        // Also runs if scheduling, decoder admission, or a ticket assertion fails.
        self.gates.open_all();
    }
}

struct TicketWake(SyncSender<()>);

impl Wake for TicketWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        // Capacity one coalesces notifications without blocking a scheduler worker.
        let _ = self.0.try_send(());
    }
}

struct TicketBatch(Vec<Option<LoadTicket>>);

impl Drop for TicketBatch {
    fn drop(&mut self) {
        for ticket in self.0.iter().flatten() {
            ticket.cancel();
        }
    }
}

fn wait_tickets(tickets: Vec<LoadTicket>) -> Vec<ScheduleOutcome> {
    let (sender, receiver) = mpsc::sync_channel(1);
    let waker = Waker::from(Arc::new(TicketWake(sender)));
    let mut context = Context::from_waker(&waker);
    let mut tickets = TicketBatch(tickets.into_iter().map(Some).collect());
    let mut outcomes: Vec<Option<ScheduleOutcome>> = (0..tickets.0.len()).map(|_| None).collect();
    let deadline = Instant::now() + TIMEOUT;
    loop {
        for (ticket, outcome) in tickets.0.iter_mut().zip(&mut outcomes) {
            let Some(pending) = ticket.as_mut() else {
                continue;
            };
            if let Poll::Ready(result) = Pin::new(pending).poll(&mut context) {
                *outcome = Some(result);
                *ticket = None;
            }
        }
        if outcomes.iter().all(Option::is_some) {
            return outcomes.into_iter().map(Option::unwrap).collect();
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(
            !remaining.is_zero(),
            "shared-flight tickets did not terminate: {outcomes:?}"
        );
        receiver.recv_timeout(remaining).unwrap_or_else(|error| {
            panic!("shared-flight ticket wait timed out ({error}): {outcomes:?}")
        });
    }
}

fn assert_cycles(outcomes: Vec<ScheduleOutcome>) {
    for (root, outcome) in outcomes.into_iter().enumerate() {
        assert!(
            matches!(&outcome, Err(LoadFailure::AssemblyFailed(detail)) if detail.contains("cycle")),
            "root {root} must terminate with a cycle diagnostic: {outcome:?}"
        );
    }
}

#[test]
fn independent_two_root_shared_flight_cycle_fails_and_scheduler_recovers() {
    let scenario = Scenario::new(&[
        Node::new("a.wait-graph", &["b.wait-graph"]),
        Node::new("b.wait-graph", &["a.wait-graph"]),
    ]);
    let tickets = vec![
        scenario.schedule("a.wait-graph"),
        scenario.schedule("b.wait-graph"),
    ];
    // Both flights originated as roots with empty, disjoint ancestor sets.
    scenario.gates.wait_for_all_decoders();
    assert_eq!(scenario.scheduler.in_flight(), 2);
    scenario.gates.open_all();
    assert_cycles(wait_tickets(tickets));
    scenario.gates.assert_decoded_once();
    scenario.assert_recovered();
}

#[test]
fn independent_three_root_shared_flight_cycle_fails_and_scheduler_recovers() {
    let scenario = Scenario::new(&[
        Node::new("a.wait-graph", &["b.wait-graph"]),
        Node::new("b.wait-graph", &["c.wait-graph"]),
        Node::new("c.wait-graph", &["a.wait-graph"]),
    ]);
    let tickets =
        ["a.wait-graph", "b.wait-graph", "c.wait-graph"].map(|path| scenario.schedule(path));
    scenario.gates.wait_for_all_decoders();
    assert_eq!(scenario.scheduler.in_flight(), 3);
    scenario.gates.open_all();
    assert_cycles(wait_tickets(tickets.into()));
    scenario.gates.assert_decoded_once();
    scenario.assert_recovered();
}

#[test]
fn acyclic_shared_diamond_commits_every_root_without_false_cycle_or_duplicate_decode() {
    let paths = [
        "root.wait-graph",
        "left.wait-graph",
        "right.wait-graph",
        "leaf.wait-graph",
    ];
    let scenario = Scenario::new(&[
        Node::new(paths[0], &["left.wait-graph", "right.wait-graph"]),
        Node::new(paths[1], &["leaf.wait-graph"]),
        Node::new(paths[2], &["leaf.wait-graph"]),
        Node::new(paths[3], &[]),
    ]);
    let tickets = paths.map(|path| scenario.schedule(path));
    scenario.gates.wait_for_all_decoders();
    assert_eq!(scenario.scheduler.in_flight(), 4);
    scenario.gates.open_all();
    let handles: Vec<_> = wait_tickets(tickets.into())
        .into_iter()
        .map(|outcome| outcome.expect("acyclic diamond must load successfully"))
        .collect();
    for (handle, path) in handles.iter().zip(paths) {
        scenario.assert_value(handle, path);
    }
    scenario.gates.assert_decoded_once();
    scenario.assert_recovered();
}

#[test]
fn direct_label_source_wait_closes_cross_root_cycle() {
    let scenario = Scenario::new(&[
        Node {
            label_dependencies: Some(&[]),
            ..Node::new("a.wait-graph", &["b.wait-graph"])
        },
        Node::new("b.wait-graph", &["a.wait-graph#part"]),
    ]);
    let mut tickets = vec![
        scenario.schedule("a.wait-graph"),
        scenario.schedule("b.wait-graph"),
    ];
    scenario.gates.wait_for_all_decoders();
    // Admit the direct label before either source decoder can finish. The
    // implicit label -> source wait must join a -> b -> label in the graph.
    tickets.push(scenario.schedule("a.wait-graph#part"));
    assert_eq!(scenario.scheduler.in_flight(), 3);
    scenario.gates.open_all();
    assert_cycles(wait_tickets(tickets));
    scenario.gates.assert_decoded_once();
    scenario.assert_recovered();
}

#[test]
fn labeled_output_dependency_participates_in_shared_flight_cycle() {
    let scenario = Scenario::new(&[
        Node {
            label_dependencies: Some(&["b.wait-graph"]),
            ..Node::new("a.wait-graph", &[])
        },
        Node::new("b.wait-graph", &["a.wait-graph#part"]),
    ]);
    let mut tickets = vec![
        scenario.schedule("a.wait-graph"),
        scenario.schedule("b.wait-graph"),
    ];
    scenario.gates.wait_for_all_decoders();
    tickets.push(scenario.schedule("a.wait-graph#part"));
    scenario.gates.open_all();
    assert_cycles(wait_tickets(tickets));
    scenario.gates.assert_decoded_once();
    scenario.assert_recovered();
}
