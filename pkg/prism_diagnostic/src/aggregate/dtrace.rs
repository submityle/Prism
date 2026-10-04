//! Distributed trace correlation: cross-process span trees (§24.6).
//!
//! A single logical request fans out across processes — client → server → DB —
//! and each hop records its own span. Linking them by a shared `trace_id` and
//! parent/child `span_id`s reconstructs one distributed span tree
//! (OpenTelemetry's shape), so an operator can see where a cross-process
//! request spent its time and which service (instance) was slow.
//!
//! [`TraceAssembler`] ingests [`DistributedSpan`]s arriving in any order from
//! any instance, groups them by [`TraceId`], and builds an [`AssembledTrace`]
//! per trace: a parent/child forest plus a [`CriticalPath`] (the longest
//! root-to-leaf chain by duration) and per-service latency attribution. Spans
//! whose declared parent never arrives become additional roots (orphans) rather
//! than being dropped. Network transport is upper-layer; this module owns the
//! deterministic correlation + tree-assembly data structures. Pure
//! `core`/`alloc`, no `unsafe`.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// A 128-bit distributed trace identifier shared by every span of one request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TraceId(pub u128);

/// A 64-bit span identifier, unique within a trace.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SpanId(pub u64);

/// The OpenTelemetry-style kind of a span (where it sits in a request path).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpanKind {
    /// Internal work within one service.
    Internal,
    /// An outbound call awaiting a response (the client side of an RPC).
    Client,
    /// Handling an inbound request (the server side of an RPC).
    Server,
    /// A message producer (enqueues work consumed elsewhere).
    Producer,
    /// A message consumer (processes produced work).
    Consumer,
    /// A database / storage call.
    Db,
}

impl SpanKind {
    /// A short, stable label.
    #[inline]
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Internal => "internal",
            Self::Client => "client",
            Self::Server => "server",
            Self::Producer => "producer",
            Self::Consumer => "consumer",
            Self::Db => "db",
        }
    }
}

/// One span of a distributed trace, recorded by one instance on one hop.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DistributedSpan {
    /// The trace this span belongs to.
    pub trace_id: TraceId,
    /// This span's id, unique within its trace.
    pub span_id: SpanId,
    /// The parent span's id, or `None` for a trace root.
    pub parent_span_id: Option<SpanId>,
    /// The instance (service / shard) that recorded this span.
    pub instance: String,
    /// The operation name (e.g. `"GET /world"`, `"db.query"`).
    pub operation: String,
    /// The span kind.
    pub kind: SpanKind,
    /// Start time on the recording instance's clock, in nanoseconds.
    pub start_nanos: u64,
    /// Span duration, in nanoseconds.
    pub duration_nanos: u64,
}

impl DistributedSpan {
    /// Construct a distributed span.
    #[expect(
        clippy::too_many_arguments,
        reason = "a span mirrors the OpenTelemetry wire shape; grouping its fields into a sub-struct would only move the arity"
    )]
    #[must_use]
    pub fn new(
        trace_id: TraceId,
        span_id: SpanId,
        parent_span_id: Option<SpanId>,
        instance: impl Into<String>,
        operation: impl Into<String>,
        kind: SpanKind,
        start_nanos: u64,
        duration_nanos: u64,
    ) -> Self {
        Self {
            trace_id,
            span_id,
            parent_span_id,
            instance: instance.into(),
            operation: operation.into(),
            kind,
            start_nanos,
            duration_nanos,
        }
    }

    /// End time, `start_nanos + duration_nanos` (saturating).
    #[inline]
    #[must_use]
    pub fn end_nanos(&self) -> u64 {
        self.start_nanos.saturating_add(self.duration_nanos)
    }
}

/// One node in an [`AssembledTrace`]'s forest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceNode {
    /// The span at this node.
    pub span: DistributedSpan,
    /// Child node indices into [`AssembledTrace::nodes`], sorted by child start
    /// time (ties by span id) for determinism.
    pub children: Vec<usize>,
    /// Whether this node is an orphan root: it declared a parent that never
    /// arrived, so it was promoted to a root rather than dropped.
    pub orphan: bool,
}

/// The longest root-to-leaf chain of a trace, by summed duration.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CriticalPath {
    /// Node indices from root to leaf along the critical path.
    pub nodes: Vec<usize>,
    /// Summed duration along the path, in nanoseconds.
    pub total_nanos: u64,
}

/// Per-instance latency attribution within one trace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServiceLatency {
    /// The instance (service) label.
    pub instance: String,
    /// Number of spans this instance contributed to the trace.
    pub span_count: u64,
    /// Summed span duration for this instance, in nanoseconds.
    pub total_nanos: u64,
}

/// A fully assembled distributed trace: a parent/child forest plus derived
/// critical-path and per-service attribution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssembledTrace {
    /// The trace id.
    pub trace_id: TraceId,
    /// All nodes, in the assembler's deterministic insertion order.
    pub nodes: Vec<TraceNode>,
    /// Root node indices (true roots then orphans), sorted by start time.
    pub roots: Vec<usize>,
}

impl AssembledTrace {
    /// Total span count in this trace.
    #[inline]
    #[must_use]
    pub fn span_count(&self) -> usize {
        self.nodes.len()
    }

    /// Borrow a node by index, if in range.
    #[inline]
    #[must_use]
    pub fn node(&self, index: usize) -> Option<&TraceNode> {
        self.nodes.get(index)
    }

    /// The wall-clock span of the whole trace: `max(end) - min(start)` across
    /// every span, in nanoseconds (`0` for an empty trace).
    #[must_use]
    pub fn wall_nanos(&self) -> u64 {
        if self.nodes.is_empty() {
            return 0;
        }
        let min_start = self
            .nodes
            .iter()
            .map(|n| n.span.start_nanos)
            .min()
            .unwrap_or(0);
        let max_end = self
            .nodes
            .iter()
            .map(|n| n.span.end_nanos())
            .max()
            .unwrap_or(0);
        max_end.saturating_sub(min_start)
    }

    /// The number of orphan roots (declared a missing parent).
    #[must_use]
    pub fn orphan_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.orphan).count()
    }

    /// Compute the critical path: the root-to-leaf chain with the greatest
    /// summed span duration. Ties pick the chain whose first node starts
    /// earliest, then the lowest span id, for determinism.
    #[must_use]
    pub fn critical_path(&self) -> CriticalPath {
        let mut best = CriticalPath::default();
        for &root in &self.roots {
            let mut path = Vec::new();
            let candidate = self.deepest_from(root, &mut path);
            if candidate.total_nanos > best.total_nanos
                || (candidate.total_nanos == best.total_nanos
                    && best.nodes.is_empty()
                    && !candidate.nodes.is_empty())
            {
                best = candidate;
            }
        }
        best
    }

    /// Longest-duration chain descending from `index`.
    fn deepest_from(&self, index: usize, stack: &mut Vec<usize>) -> CriticalPath {
        stack.push(index);
        let node = &self.nodes[index];
        let here = node.span.duration_nanos;

        let mut best_child = CriticalPath::default();
        for &child in &node.children {
            let candidate = self.deepest_from(child, stack);
            if candidate.total_nanos > best_child.total_nanos {
                best_child = candidate;
            }
        }
        stack.pop();

        let mut nodes = Vec::with_capacity(1 + best_child.nodes.len());
        nodes.push(index);
        nodes.extend_from_slice(&best_child.nodes);
        CriticalPath {
            nodes,
            total_nanos: here.saturating_add(best_child.total_nanos),
        }
    }

    /// Per-instance latency attribution, sorted by summed duration descending
    /// (ties by instance label ascending).
    #[must_use]
    pub fn service_latencies(&self) -> Vec<ServiceLatency> {
        let mut by_instance: BTreeMap<String, (u64, u64)> = BTreeMap::new();
        for node in &self.nodes {
            let entry = by_instance
                .entry(node.span.instance.clone())
                .or_insert((0, 0));
            entry.0 += 1;
            entry.1 = entry.1.saturating_add(node.span.duration_nanos);
        }
        let mut out: Vec<ServiceLatency> = by_instance
            .into_iter()
            .map(|(instance, (span_count, total_nanos))| ServiceLatency {
                instance,
                span_count,
                total_nanos,
            })
            .collect();
        out.sort_by(|a, b| {
            b.total_nanos
                .cmp(&a.total_nanos)
                .then_with(|| a.instance.cmp(&b.instance))
        });
        out
    }
}

/// Assembles [`DistributedSpan`]s into per-trace [`AssembledTrace`]s.
///
/// Spans may arrive in any order and from any instance; the assembler buffers
/// them and links parents to children on [`assemble`](Self::assemble). A span
/// whose `parent_span_id` is `None`, or names a parent not present in the same
/// trace, becomes a root (the latter flagged [`TraceNode::orphan`]). Duplicate
/// `span_id`s within a trace keep the first-ingested span.
#[derive(Clone, Debug, Default)]
pub struct TraceAssembler {
    /// Buffered spans in ingestion order.
    spans: Vec<DistributedSpan>,
}

impl TraceAssembler {
    /// A new, empty assembler.
    #[must_use]
    pub fn new() -> Self {
        Self { spans: Vec::new() }
    }

    /// Ingest one span (any trace, any order).
    pub fn ingest(&mut self, span: DistributedSpan) {
        self.spans.push(span);
    }

    /// Number of buffered spans.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Whether no spans have been ingested.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// The distinct trace ids seen so far, ascending.
    #[must_use]
    pub fn trace_ids(&self) -> Vec<TraceId> {
        let mut ids: Vec<TraceId> = Vec::new();
        for span in &self.spans {
            if !ids.contains(&span.trace_id) {
                ids.push(span.trace_id);
            }
        }
        ids.sort_unstable();
        ids
    }

    /// Assemble a single trace by id, or `None` if that trace has no spans.
    #[must_use]
    pub fn assemble(&self, trace_id: TraceId) -> Option<AssembledTrace> {
        // Collect this trace's spans, dropping duplicate span ids (keep first).
        let mut nodes: Vec<TraceNode> = Vec::new();
        let mut id_to_index: BTreeMap<SpanId, usize> = BTreeMap::new();
        for span in self.spans.iter().filter(|s| s.trace_id == trace_id) {
            if id_to_index.contains_key(&span.span_id) {
                continue;
            }
            id_to_index.insert(span.span_id, nodes.len());
            nodes.push(TraceNode {
                span: span.clone(),
                children: Vec::new(),
                orphan: false,
            });
        }
        if nodes.is_empty() {
            return None;
        }

        // Link children to parents; collect roots (true + orphan).
        let mut roots: Vec<usize> = Vec::new();
        // Precompute parent links to avoid aliasing the nodes borrow.
        let mut links: Vec<(usize, Option<usize>)> = Vec::with_capacity(nodes.len());
        for (index, node) in nodes.iter().enumerate() {
            let parent_index = match node.span.parent_span_id {
                None => None,
                Some(parent) => id_to_index.get(&parent).copied(),
            };
            links.push((index, parent_index));
        }
        for (index, parent_index) in links {
            match parent_index {
                Some(parent) => nodes[parent].children.push(index),
                None => {
                    // Orphan when a parent was declared but not found.
                    if nodes[index].span.parent_span_id.is_some() {
                        nodes[index].orphan = true;
                    }
                    roots.push(index);
                }
            }
        }

        // Deterministic ordering: children + roots by (start, span id).
        let order = |a: usize, b: usize, nodes: &[TraceNode]| {
            nodes[a]
                .span
                .start_nanos
                .cmp(&nodes[b].span.start_nanos)
                .then_with(|| nodes[a].span.span_id.cmp(&nodes[b].span.span_id))
        };
        // Sort each node's children.
        let snapshot: Vec<DistributedSpan> = nodes.iter().map(|n| n.span.clone()).collect();
        for node in &mut nodes {
            node.children.sort_by(|&a, &b| {
                snapshot[a]
                    .start_nanos
                    .cmp(&snapshot[b].start_nanos)
                    .then_with(|| snapshot[a].span_id.cmp(&snapshot[b].span_id))
            });
        }
        roots.sort_by(|&a, &b| order(a, b, &nodes));

        Some(AssembledTrace {
            trace_id,
            nodes,
            roots,
        })
    }

    /// Assemble every buffered trace, ordered by ascending [`TraceId`].
    #[must_use]
    pub fn assemble_all(&self) -> Vec<AssembledTrace> {
        self.trace_ids()
            .into_iter()
            .filter_map(|id| self.assemble(id))
            .collect()
    }
}
