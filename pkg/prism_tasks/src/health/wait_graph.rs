//! Deterministic, `no_std`-friendly core for deadlock prevention
//! (design §24.8 死锁预防).
//!
//! This module owns the *decision* half of §24.8's two deadlock guards and
//! holds no threads, no clock, and no allocation beyond the graph arena:
//!
//! - **Cycle detection**: a [`WaitGraph`] is a *wait-for* graph. An edge
//!   `waiter -> holder` means the task (or resource) `waiter` is blocked
//!   waiting on `holder`. A directed cycle in that graph is a deadlock, so
//!   [`WaitGraph::try_add_dependency`] admits an edge only if it keeps the
//!   graph acyclic — the "作业图在提交时静态检测环依赖" guard.
//! - **Wait-chain depth limiting**: unbounded chains of blocked fibers can
//!   exhaust the fiber-stack pool (§24.8 "fiber 等待链深度限幅，防栈耗尽").
//!   A [`WaitGraph`] built with [`WaitGraph::with_chain_limit`] additionally
//!   rejects any edge whose longest resulting wait chain would exceed the
//!   limit.
//!
//! Both checks are pure functions of the recorded edges, evaluated in a fixed
//! node-index order, so they are directly unit-testable against a brute-force
//! reachability / path-enumeration oracle. There is no execution façade: the
//! graph is a decision structure a scheduler consults before parking a task.

use alloc::vec::Vec;

/// Stable identifier of a node (a task or a resource) in a [`WaitGraph`]. The
/// value is the node's index in the arena.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct WaitNodeId(pub usize);

impl WaitNodeId {
    /// The raw arena index backing this id.
    #[must_use]
    #[inline]
    pub const fn index(self) -> usize {
        self.0
    }
}

/// Why [`WaitGraph::try_add_dependency`] refused an edge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeadlockError {
    /// The edge would close a directed cycle in the wait-for graph (a
    /// deadlock). The field holds the offending cycle in traversal order:
    /// `holder -> ... -> waiter`, which the rejected edge `waiter -> holder`
    /// would complete.
    Cycle {
        /// The nodes forming the cycle, in traversal order.
        cycle: Vec<WaitNodeId>,
    },
    /// The edge would be acyclic but the longest wait chain passing through it
    /// (counted in nodes) would exceed the configured limit.
    ChainTooDeep {
        /// Length, in nodes, of the longest wait chain the edge would create.
        depth: usize,
        /// The configured maximum wait-chain length, in nodes.
        limit: usize,
    },
}

/// A wait-for graph used for deadlock prevention (design §24.8).
///
/// Nodes model tasks or resources; a directed edge `waiter -> holder` records
/// that `waiter` is blocked on `holder`. Add nodes with [`WaitGraph::add_node`],
/// then gate each new dependency through [`WaitGraph::try_add_dependency`],
/// which keeps the graph acyclic (and, with [`WaitGraph::with_chain_limit`],
/// within a bounded wait-chain depth). Release a dependency with
/// [`WaitGraph::remove_dependency`] when the wait resolves.
#[derive(Clone, Debug)]
pub struct WaitGraph {
    /// Out-edges per node: `edges[w]` lists the holders that node `w` waits on.
    edges: Vec<Vec<WaitNodeId>>,
    /// Optional cap on the longest wait chain, in nodes.
    chain_limit: Option<usize>,
}

impl Default for WaitGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl WaitGraph {
    /// Create an empty graph with no wait-chain limit (cycle detection only).
    #[must_use]
    pub fn new() -> Self {
        Self {
            edges: Vec::new(),
            chain_limit: None,
        }
    }

    /// Create an empty graph that also rejects any edge whose longest resulting
    /// wait chain (in nodes) would exceed `limit`. `limit` is raised to at
    /// least `1`.
    #[must_use]
    pub fn with_chain_limit(limit: usize) -> Self {
        Self {
            edges: Vec::new(),
            chain_limit: Some(limit.max(1)),
        }
    }

    /// Add a fresh node and return its id.
    pub fn add_node(&mut self) -> WaitNodeId {
        let id = WaitNodeId(self.edges.len());
        self.edges.push(Vec::new());
        id
    }

    /// Number of nodes in the graph.
    #[must_use]
    #[inline]
    pub fn node_count(&self) -> usize {
        self.edges.len()
    }

    /// Whether the graph has no nodes.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// The configured wait-chain node limit, if any.
    #[must_use]
    #[inline]
    pub fn chain_limit(&self) -> Option<usize> {
        self.chain_limit
    }

    /// Whether the edge `waiter -> holder` is already present.
    ///
    /// # Panics
    /// Panics if either id is out of range for this graph.
    #[must_use]
    pub fn has_dependency(&self, waiter: WaitNodeId, holder: WaitNodeId) -> bool {
        self.edges[waiter.index()].contains(&holder)
    }

    /// Whether adding `waiter -> holder` would close a directed cycle, i.e.
    /// whether `holder` can already reach `waiter` along existing edges.
    ///
    /// A self-dependency (`waiter == holder`) always reports `true`. This is a
    /// pure query and does not mutate the graph.
    ///
    /// # Panics
    /// Panics if either id is out of range for this graph.
    #[must_use]
    pub fn creates_cycle(&self, waiter: WaitNodeId, holder: WaitNodeId) -> bool {
        if waiter == holder {
            return true;
        }
        self.reaches(holder, waiter)
    }

    /// The length, in nodes, of the longest wait chain that adding
    /// `waiter -> holder` would create through that edge, assuming it is
    /// acyclic. Equal to the longest existing chain ending at `waiter` plus the
    /// longest existing chain starting at `holder`.
    ///
    /// # Panics
    /// Panics if either id is out of range for this graph.
    #[must_use]
    pub fn resulting_chain_depth(&self, waiter: WaitNodeId, holder: WaitNodeId) -> usize {
        // Nodes on the longest path ending at `waiter` (inclusive) plus nodes
        // on the longest path starting at `holder` (inclusive). The new edge
        // joins the two, so the counts simply add.
        self.longest_incoming_nodes(waiter) + self.longest_outgoing_nodes(holder)
    }

    /// Try to add the dependency `waiter -> holder`.
    ///
    /// Succeeds (recording the edge) only if it keeps the graph acyclic and,
    /// when a chain limit is configured, within that limit. A duplicate edge is
    /// a no-op success.
    ///
    /// # Errors
    /// Returns [`DeadlockError::Cycle`] if the edge would close a cycle, or
    /// [`DeadlockError::ChainTooDeep`] if it would exceed the configured
    /// wait-chain limit.
    ///
    /// # Panics
    /// Panics if either id is out of range for this graph.
    pub fn try_add_dependency(
        &mut self,
        waiter: WaitNodeId,
        holder: WaitNodeId,
    ) -> Result<(), DeadlockError> {
        if self.has_dependency(waiter, holder) {
            return Ok(());
        }
        if self.creates_cycle(waiter, holder) {
            // Reconstruct the offending `holder -> ... -> waiter` path for the
            // diagnostic. `waiter == holder` yields the single-node cycle.
            let cycle = self
                .path(holder, waiter)
                .unwrap_or_else(|| alloc::vec![waiter]);
            return Err(DeadlockError::Cycle { cycle });
        }
        if let Some(limit) = self.chain_limit {
            let depth = self.resulting_chain_depth(waiter, holder);
            if depth > limit {
                return Err(DeadlockError::ChainTooDeep { depth, limit });
            }
        }
        self.edges[waiter.index()].push(holder);
        Ok(())
    }

    /// Remove the dependency `waiter -> holder` if present. Returns whether an
    /// edge was removed.
    ///
    /// # Panics
    /// Panics if either id is out of range for this graph.
    pub fn remove_dependency(&mut self, waiter: WaitNodeId, holder: WaitNodeId) -> bool {
        let out = &mut self.edges[waiter.index()];
        if let Some(pos) = out.iter().position(|&h| h == holder) {
            out.remove(pos);
            true
        } else {
            false
        }
    }

    /// The longest wait chain currently in the graph, in nodes (`0` for an
    /// empty graph). On an acyclic graph this is the longest directed path.
    #[must_use]
    pub fn longest_chain(&self) -> usize {
        let mut best = 0;
        for node in 0..self.edges.len() {
            best = best.max(self.longest_outgoing_nodes(WaitNodeId(node)));
        }
        best
    }

    /// Find any directed cycle currently in the graph, returned in traversal
    /// order, or `None` if the graph is acyclic. Uses a three-colour depth-first
    /// search in ascending node-index order, so the result is deterministic.
    #[must_use]
    pub fn find_cycle(&self) -> Option<Vec<WaitNodeId>> {
        /// Depth-first search colours.
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Colour {
            White,
            Gray,
            Black,
        }
        let n = self.edges.len();
        let mut colour = alloc::vec![Colour::White; n];
        let mut parent: Vec<Option<usize>> = alloc::vec![None; n];
        // Iterative DFS with an explicit frame stack: (node, next child index).
        for start in 0..n {
            if colour[start] != Colour::White {
                continue;
            }
            let mut stack: Vec<(usize, usize)> = alloc::vec![(start, 0)];
            colour[start] = Colour::Gray;
            while let Some(&(node, child_ix)) = stack.last() {
                if child_ix < self.edges[node].len() {
                    stack.last_mut().unwrap().1 += 1;
                    let next = self.edges[node][child_ix].index();
                    match colour[next] {
                        Colour::White => {
                            parent[next] = Some(node);
                            colour[next] = Colour::Gray;
                            stack.push((next, 0));
                        }
                        Colour::Gray => {
                            // Back edge `node -> next`: walk parents from `node`
                            // back to `next` to recover the cycle.
                            let mut cycle = alloc::vec![WaitNodeId(next)];
                            let mut cur = node;
                            while cur != next {
                                cycle.push(WaitNodeId(cur));
                                cur = parent[cur].expect("gray node has a parent");
                            }
                            cycle.reverse();
                            return Some(cycle);
                        }
                        Colour::Black => {}
                    }
                } else {
                    colour[node] = Colour::Black;
                    stack.pop();
                }
            }
        }
        None
    }

    /// Whether `from` can reach `to` along existing edges (`from == to` is
    /// `true` by reflexivity). Deterministic depth-first search.
    fn reaches(&self, from: WaitNodeId, to: WaitNodeId) -> bool {
        if from == to {
            return true;
        }
        let mut visited = alloc::vec![false; self.edges.len()];
        let mut stack = alloc::vec![from.index()];
        visited[from.index()] = true;
        while let Some(node) = stack.pop() {
            for &next in &self.edges[node] {
                if next == to {
                    return true;
                }
                if !visited[next.index()] {
                    visited[next.index()] = true;
                    stack.push(next.index());
                }
            }
        }
        false
    }

    /// Recover one directed path `from -> ... -> to` along existing edges, in
    /// order, or `None` if `to` is unreachable from `from`. Deterministic
    /// (ascending child order, breadth-first for a shortest witness).
    fn path(&self, from: WaitNodeId, to: WaitNodeId) -> Option<Vec<WaitNodeId>> {
        use alloc::collections::VecDeque;
        if from == to {
            return Some(alloc::vec![from]);
        }
        let mut parent: Vec<Option<usize>> = alloc::vec![None; self.edges.len()];
        let mut visited = alloc::vec![false; self.edges.len()];
        let mut queue = VecDeque::new();
        queue.push_back(from.index());
        visited[from.index()] = true;
        while let Some(node) = queue.pop_front() {
            for &next in &self.edges[node] {
                if visited[next.index()] {
                    continue;
                }
                visited[next.index()] = true;
                parent[next.index()] = Some(node);
                if next == to {
                    let mut chain = alloc::vec![to];
                    let mut cur = node;
                    loop {
                        chain.push(WaitNodeId(cur));
                        match parent[cur] {
                            Some(p) => cur = p,
                            None => break,
                        }
                    }
                    chain.reverse();
                    return Some(chain);
                }
                queue.push_back(next.index());
            }
        }
        None
    }

    /// Nodes on the longest directed path *starting* at `node` (inclusive).
    /// Assumes the graph is acyclic; recursion is bounded by the node count.
    fn longest_outgoing_nodes(&self, node: WaitNodeId) -> usize {
        let mut best = 0;
        for &next in &self.edges[node.index()] {
            best = best.max(self.longest_outgoing_nodes(next));
        }
        best + 1
    }

    /// Nodes on the longest directed path *ending* at `node` (inclusive).
    /// Computed over the reverse graph; assumes acyclicity.
    fn longest_incoming_nodes(&self, node: WaitNodeId) -> usize {
        let mut best = 0;
        for w in 0..self.edges.len() {
            if self.edges[w].contains(&node) {
                best = best.max(self.longest_incoming_nodes(WaitNodeId(w)));
            }
        }
        best + 1
    }
}
