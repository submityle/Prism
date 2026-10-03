//! The [`Schedule`]: a collection of systems plus their ordering constraints,
//! and the deterministic topological order the executor runs them in
//! (design §8.2).
//!
//! # Ordering model
//!
//! Every system is a member of its [`Phase`] set (default [`Phase::Update`])
//! and of any sets it was added to. Ordering edges come from several sources,
//! unified into one directed graph over systems:
//!
//! * **phases** — the six fixed phases are chained `First → … → Last`, so a
//!   system in an earlier phase always precedes one in a later phase;
//! * **set / system edges** — `before`/`after` against a [`SystemSet`] expand
//!   to edges against every current member of that set;
//! * **chains** — a chained group (see
//!   [`SystemConfigs::chain`](crate::schedule::SystemConfigs::chain)) adds edges
//!   so every system of one child precedes every system of the next.
//!
//! The graph is topologically sorted with insertion order as the deterministic
//! tie-break. A contradictory set of constraints (a cycle) is a hard error and
//! panics — the deterministic analogue of the ambiguity hard-gate in §23.4.
//!
//! # Execution
//!
//! Execution for M1 is single-threaded: each system is run via
//! [`System::run`](crate::system::System::run), which fetches its params, runs
//! the body, and immediately applies its deferred
//! [`Commands`](crate::command::Commands) — a sync point after every system.
//! The parallel conflict-graph executor (§8.2) and fiber job graph (§8.3) layer
//! on top later without changing this API; the per-system
//! [`Access`](crate::query::Access) is already recorded for them.

use alloc::string::String;
use alloc::vec::Vec;

use hashbrown::HashMap;

use crate::query::Access;
use crate::schedule::ambiguity::{self, Ambiguities};
use crate::schedule::condition::BoxedCondition;
use crate::schedule::config::{IntoSystemConfigs, SetConfig, SystemConfig, SystemConfigs};
use crate::schedule::phase::Phase;
use crate::schedule::set::{SystemSet, SystemSetId};
use crate::system::function::BoxedSystem;
use crate::world::World;

/// One scheduled system plus the resolved metadata the executor needs.
pub(crate) struct Node {
    pub(crate) system: BoxedSystem,
    pub(crate) sets: Vec<SystemSetId>,
    pub(crate) before: Vec<SystemSetId>,
    pub(crate) after: Vec<SystemSetId>,
    pub(crate) conditions: Vec<BoxedCondition>,
    pub(crate) chain_after: Vec<usize>,
}

/// A set of systems with ordering constraints, run as one unit against a
/// [`World`].
///
/// Add work with [`add_systems`](Schedule::add_systems) (a system, a tuple, or
/// any [`SystemConfigs`] produced by the fluent combinators). Configure a
/// [`SystemSet`] with [`configure_set`](Schedule::configure_set). Run it with
/// [`run`](Schedule::run); initialization is lazy and idempotent.
#[derive(Default)]
pub struct Schedule {
    pub(crate) nodes: Vec<Node>,
    pub(crate) set_configs: HashMap<SystemSetId, SetConfig>,
    pub(crate) order: Vec<usize>,
    initialized: bool,
}

impl Schedule {
    /// A new, empty schedule with the six fixed phases pre-chained
    /// `First → PreUpdate → Update → FixedUpdate → PostUpdate → Last`.
    #[must_use]
    pub fn new() -> Self {
        let mut schedule = Self {
            nodes: Vec::new(),
            set_configs: HashMap::new(),
            order: Vec::new(),
            initialized: false,
        };
        // Order the fixed phases. Each phase runs after *every* earlier phase
        // (not just the immediately preceding one) so ordering still holds when
        // intermediate phases happen to be empty: set-ordering edges are
        // expanded against concrete members, and an empty phase contributes
        // none, which would otherwise break an adjacent-only chain.
        for (i, &later) in Phase::ORDER.iter().enumerate() {
            let after = schedule.set_configs.entry(later.set_id()).or_default();
            for &earlier in &Phase::ORDER[..i] {
                after.after.push(earlier.set_id());
            }
        }
        schedule
    }

    /// Add systems to the schedule.
    ///
    /// Accepts a bare system, a tuple of systems, or any [`SystemConfigs`] (so
    /// `.chain()`, `.run_if(..)`, `.in_set(..)`, `.before(..)`/`.after(..)`, and
    /// `.in_phase(..)` results all work directly). Re-arms lazy initialization.
    pub fn add_systems<M>(&mut self, systems: impl IntoSystemConfigs<M>) -> &mut Self {
        self.flatten(systems.into_configs());
        self.initialized = false;
        self
    }

    /// Attach shared ordering/condition configuration to `set`.
    pub fn configure_set(&mut self, set: impl SystemSet, config: SetConfig) -> &mut Self {
        let entry = self.set_configs.entry(set.set_id()).or_default();
        entry.before.extend(config.before);
        entry.after.extend(config.after);
        entry.conditions.extend(config.conditions);
        self.initialized = false;
        self
    }

    /// How many systems the schedule holds.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the schedule holds no systems.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Flatten a [`SystemConfigs`] tree into schedule nodes, returning the leaf
    /// node indices it contributed (in order) so a parent chain can wire edges.
    fn flatten(&mut self, configs: SystemConfigs) -> Vec<usize> {
        match configs {
            SystemConfigs::Node(config) => {
                let idx = self.push_config(config);
                alloc::vec![idx]
            }
            SystemConfigs::Group {
                configs,
                chained,
                collective_conditions,
            } => {
                // A group with shared conditions anchors them on a fresh
                // anonymous set, so they are evaluated once per run and cached.
                let group_set = if collective_conditions.is_empty() {
                    None
                } else {
                    let id = SystemSetId::anonymous();
                    self.set_configs
                        .entry(id)
                        .or_default()
                        .conditions
                        .extend(collective_conditions);
                    Some(id)
                };

                let mut all_leaves: Vec<usize> = Vec::new();
                let mut prev_child: Option<Vec<usize>> = None;
                for child in configs {
                    let leaves = self.flatten(child);
                    if chained && let Some(prev) = &prev_child {
                        for &predecessor in prev {
                            for &successor in &leaves {
                                self.nodes[successor].chain_after.push(predecessor);
                            }
                        }
                    }
                    prev_child = Some(leaves.clone());
                    all_leaves.extend(leaves);
                }

                if let Some(id) = group_set {
                    for &leaf in &all_leaves {
                        self.nodes[leaf].sets.push(id);
                    }
                }
                all_leaves
            }
        }
    }

    fn push_config(&mut self, config: SystemConfig) -> usize {
        let SystemConfig {
            system,
            mut sets,
            before,
            after,
            conditions,
            phase,
        } = config;
        // Every system is a member of its phase set.
        sets.push(phase.set_id());
        let idx = self.nodes.len();
        self.nodes.push(Node {
            system,
            sets,
            before,
            after,
            conditions,
            chain_after: Vec::new(),
        });
        idx
    }

    /// Resolve every system's and condition's param state and compute the run
    /// order. Idempotent: a no-op once initialised until systems/sets change.
    pub fn initialize(&mut self, world: &mut World) {
        if self.initialized {
            return;
        }
        for node in &mut self.nodes {
            node.system.initialize(world);
            for condition in &mut node.conditions {
                condition.initialize(world);
            }
        }
        for config in self.set_configs.values_mut() {
            for condition in &mut config.conditions {
                condition.initialize(world);
            }
        }
        self.order = self.compute_order();
        self.initialized = true;
    }

    /// Initialise if needed, then run every system in dependency order via the
    /// single-threaded executor, honoring run-conditions.
    pub fn run(&mut self, world: &mut World) {
        self.initialize(world);
        crate::schedule::executor::SingleThreadedExecutor::run(self, world);
    }

    /// Evaluate (AND) all conditions configured on `set`. A set with no
    /// configured conditions is always `true`.
    pub(crate) fn eval_set_conditions(&mut self, set: SystemSetId, world: &mut World) -> bool {
        let Some(config) = self.set_configs.get_mut(&set) else {
            return true;
        };
        let mut result = true;
        for condition in &mut config.conditions {
            // Evaluate every condition (no short-circuit) so any internal state
            // advances deterministically.
            let value = condition.run(world);
            result &= value;
        }
        result
    }

    /// Build the full set of ordering edges `(from, to)` implied by phase,
    /// set-membership `before`/`after`, and chain constraints. Deduplicated;
    /// self-edges are dropped. Shared by [`compute_order`](Self::compute_order)
    /// and [`ambiguities`](Self::ambiguities) so both see exactly the same
    /// ordering relation.
    fn build_edges(&self) -> hashbrown::HashSet<(usize, usize)> {
        use hashbrown::HashSet;

        // Membership: set id -> member node indices.
        let mut members: HashMap<SystemSetId, Vec<usize>> = HashMap::new();
        for (idx, node) in self.nodes.iter().enumerate() {
            for &set in &node.sets {
                members.entry(set).or_default().push(idx);
            }
        }

        let mut edges: HashSet<(usize, usize)> = HashSet::new();
        fn add_edge(from: usize, to: usize, edges: &mut HashSet<(usize, usize)>) {
            if from != to {
                edges.insert((from, to));
            }
        }

        // Node-level before/after against set membership, plus chain edges.
        for (idx, node) in self.nodes.iter().enumerate() {
            for set in &node.after {
                if let Some(ms) = members.get(set) {
                    for &m in ms {
                        add_edge(m, idx, &mut edges);
                    }
                }
            }
            for set in &node.before {
                if let Some(ms) = members.get(set) {
                    for &m in ms {
                        add_edge(idx, m, &mut edges);
                    }
                }
            }
            for &pred in &node.chain_after {
                add_edge(pred, idx, &mut edges);
            }
        }

        // Set-level before/after: expand to all member pairs.
        for (set, config) in &self.set_configs {
            let Some(set_members) = members.get(set) else {
                continue;
            };
            for after in &config.after {
                if let Some(ms) = members.get(after) {
                    for &a in ms {
                        for &b in set_members {
                            add_edge(a, b, &mut edges);
                        }
                    }
                }
            }
            for before in &config.before {
                if let Some(ms) = members.get(before) {
                    for &b in set_members {
                        for &m in ms {
                            add_edge(b, m, &mut edges);
                        }
                    }
                }
            }
        }

        edges
    }

    /// Topologically sort the systems, honoring phase/set/chain edges with
    /// insertion order as the deterministic tie-break. Panics on a cycle.
    fn compute_order(&self) -> Vec<usize> {
        use alloc::collections::BinaryHeap;
        use core::cmp::Reverse;

        let n = self.nodes.len();
        let edges = self.build_edges();

        // Kahn's algorithm with a min-heap on node index for determinism.
        let mut indegree = alloc::vec![0usize; n];
        let mut adj: Vec<Vec<usize>> = alloc::vec![Vec::new(); n];
        for &(from, to) in &edges {
            adj[from].push(to);
            indegree[to] += 1;
        }

        let mut ready: BinaryHeap<Reverse<usize>> = BinaryHeap::new();
        for (idx, &deg) in indegree.iter().enumerate() {
            if deg == 0 {
                ready.push(Reverse(idx));
            }
        }

        let mut order = Vec::with_capacity(n);
        while let Some(Reverse(idx)) = ready.pop() {
            order.push(idx);
            let mut succ = core::mem::take(&mut adj[idx]);
            succ.sort_unstable();
            for to in succ {
                indegree[to] -= 1;
                if indegree[to] == 0 {
                    ready.push(Reverse(to));
                }
            }
        }

        assert!(
            order.len() == n,
            "prism_ecs schedule: dependency cycle detected among {} systems ({} ordered); \
             check contradictory before/after/chain or phase constraints",
            n,
            order.len()
        );
        order
    }

    /// Analyse the schedule for *ambiguities*: pairs of systems whose
    /// [`Access`](crate::query::Access) conflicts yet have no ordering edge
    /// (directly or transitively) fixing their relative order (design §23.4).
    ///
    /// Initialises the schedule first, because a system's access set is only
    /// meaningful after [`System::initialize`](crate::system::System::initialize)
    /// has run (function systems compute their access there). The result is
    /// deterministic: pairs are reported in ascending node-index order.
    pub fn ambiguities(&mut self, world: &mut World) -> Ambiguities {
        self.initialize(world);
        let accesses: Vec<&Access> = self.nodes.iter().map(|n| n.system.access()).collect();
        let names: Vec<String> = self.nodes.iter().map(|n| n.system.name().into()).collect();
        let edges: Vec<(usize, usize)> = self.build_edges().into_iter().collect();
        ambiguity::detect(&accesses, &names, &edges)
    }

    /// Initialise the schedule and panic if it contains any ambiguity, printing
    /// the full [`Ambiguities::report`]. The deterministic analogue of the
    /// cycle panic in [`compute_order`](Self::compute_order); suitable as a CI
    /// hard gate (design §23.4).
    pub fn assert_no_ambiguities(&mut self, world: &mut World) {
        let ambiguities = self.ambiguities(world);
        assert!(ambiguities.is_empty(), "{}", ambiguities.report());
    }
}
