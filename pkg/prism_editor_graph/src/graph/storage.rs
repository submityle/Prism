//! Full graph documents and bounded asynchronous single-file publication.
use super::{authoring::*, *};
use alloc::boxed::Box;
use alloc::sync::Arc;

use std::{
    fs::{self, File, OpenOptions, TryLockError},
    io::{self, Read, Write},
    path::{Component, Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc::{self, SyncSender, TrySendError},
        Mutex,
    },
    task::Poll,
    thread,
    time::{Duration, Instant},
};

/// Bounds for full documents, independent from command and history byte budgets.
#[derive(Clone, Copy, Debug)]
pub struct GraphDocumentLimits {
    /// Live graph, text, payload and per-node port policy.
    pub graph: GraphAuthoringLimits,
    /// Maximum groups, at most 16384. Zero forbids groups.
    pub groups: usize,
}
impl Default for GraphDocumentLimits {
    fn default() -> Self {
        Self {
            graph: GraphAuthoringLimits::default(),
            groups: 4096,
        }
    }
}
impl GraphDocumentLimits {
    fn validate(self) -> Result<(), GraphAuthoringError> {
        self.graph.validate()?;
        if self.groups > 16384 {
            return Err(GraphAuthoringError::Limit);
        }
        Ok(())
    }
}
/// A complete graph with opaque bytes and exactly one position per live node.
/// No history or process-local snapshot identity is serialized.
#[derive(Debug)]
pub struct GraphDocument {
    graph: Box<Graph<Vec<u8>>>,
    positions: BTreeMap<NodeId, GraphPoint>,
}
/// Process-local immutable state identity. Equality uses pointer identity, so
/// fresh editors and undo/redo states cannot accidentally confirm stale saves.
#[derive(Clone, Debug)]
pub struct GraphSnapshotToken(pub(super) Arc<()>);
impl PartialEq for GraphSnapshotToken {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for GraphSnapshotToken {}
/// Explicit serialized save capture. Creating one is a full-document operation,
/// independent of the editor's local inverse history.
#[derive(Debug)]
pub struct GraphSnapshot {
    /// Canonical complete document bytes.
    pub bytes: Vec<u8>,
    /// Compare with the live editor before acknowledging a save as current.
    pub token: GraphSnapshotToken,
}
impl GraphDocument {
    /// Consume a graph and exact node-position map, rejecting invalid DAGs,
    /// missing/extraneous positions or policy violations.
    pub fn new(
        graph: Graph<Vec<u8>>,
        positions: BTreeMap<NodeId, GraphPoint>,
        limits: GraphDocumentLimits,
    ) -> Result<Self, GraphAuthoringError> {
        let document = Self {
            graph: Box::new(graph),
            positions,
        };
        document.validate(limits)?;
        Ok(document)
    }
    /// Complete typed kernel, including groups and restored indexes.
    pub fn graph(&self) -> &Graph<Vec<u8>> {
        &self.graph
    }
    /// Stable document positions in ascending node ID order.
    pub fn positions(&self) -> impl Iterator<Item = (NodeId, GraphPoint)> + '_ {
        self.positions.iter().map(|(id, point)| (*id, *point))
    }
    /// All four allocation high-water marks: node, port, edge, group.
    pub fn next_ids(&self) -> (NodeId, PortId, EdgeId, GroupId) {
        (
            NodeId(self.graph.next_node),
            PortId(self.graph.next_port),
            EdgeId(self.graph.next_edge),
            GroupId(self.graph.next_group),
        )
    }
    /// Move into a host-owned model without cloning graph records.
    pub fn into_parts(self) -> (Graph<Vec<u8>>, BTreeMap<NodeId, GraphPoint>) {
        (*self.graph, self.positions)
    }
    pub(super) fn validate(&self, limits: GraphDocumentLimits) -> Result<(), GraphAuthoringError> {
        validate_document(&self.graph, &self.positions, limits)
    }
    /// Canonical complete v1 document, bounded on every append.
    pub fn encode(
        &self,
        limits: GraphDocumentLimits,
        max_bytes: usize,
    ) -> Result<Vec<u8>, GraphAuthoringError> {
        Self::encode_parts(&self.graph, &self.positions, limits, max_bytes)
    }
    pub(super) fn encode_parts(
        graph: &Graph<Vec<u8>>,
        positions: &BTreeMap<NodeId, GraphPoint>,
        limits: GraphDocumentLimits,
        max_bytes: usize,
    ) -> Result<Vec<u8>, GraphAuthoringError> {
        validate_document(graph, positions, limits)?;
        if max_bytes == 0 || max_bytes > MAX_GRAPH_STORAGE_BYTES {
            return Err(GraphAuthoringError::Limit);
        }
        let mut w = DocumentWriter {
            bytes: Vec::new(),
            max: max_bytes,
        };
        w.put(b"PGRD\x01\x00")?;
        for n in [
            graph.next_node,
            graph.next_port,
            graph.next_edge,
            graph.next_group,
        ] {
            w.u64(n)?;
        }
        w.u32(graph.nodes.len())?;
        w.u32(graph.groups.len())?;
        w.u32(graph.edges.len())?;
        for node in graph.nodes.values() {
            w.u64(node.id.0)?;
            w.blob(node.name.as_bytes())?;
            w.blob(&node.payload)?;
            let p = positions[&node.id];
            w.put(&p.x.to_le_bytes())?;
            w.put(&p.y.to_le_bytes())?;
            w.u64(node.group.map_or(0, |g| g.0))?;
            w.u32(node.ports.len())?;
            for id in &node.ports {
                let p = &graph.ports[id];
                w.u64(id.0)?;
                w.blob(p.name.as_bytes())?;
                w.blob(p.port_type.key().as_bytes())?;
                w.put(&[
                    match p.direction {
                        PortDirection::Input => 0,
                        PortDirection::Output => 1,
                    },
                    match p.connection_policy {
                        ConnectionPolicy::Single => 0,
                        ConnectionPolicy::Multiple => 1,
                    },
                ])?;
            }
        }
        for group in graph.groups.values() {
            w.u64(group.id.0)?;
            w.blob(group.name.as_bytes())?;
            w.u64(group.parent.map_or(0, |g| g.0))?;
        }
        for edge in graph.edges.values() {
            for n in [edge.id.0, edge.output.0, edge.input.0] {
                w.u64(n)?;
            }
        }
        Ok(w.bytes)
    }
    /// Parse one complete document. Reject duplicate/noncanonical IDs, unknown
    /// tags/versions, malformed references, group/DAG cycles and resource excess.
    /// Indexes and group membership are rebuilt through kernel validation paths.
    pub fn decode(
        bytes: &[u8],
        limits: GraphDocumentLimits,
        max_bytes: usize,
    ) -> Result<Self, GraphAuthoringError> {
        limits.validate()?;
        if max_bytes == 0 || max_bytes > MAX_GRAPH_STORAGE_BYTES || bytes.len() > max_bytes {
            return Err(GraphAuthoringError::Limit);
        }
        let mut r = DocumentReader { bytes, offset: 0 };
        if r.take(6)? != b"PGRD\x01\x00" {
            return Err(GraphAuthoringError::Codec);
        }
        let counters = [r.u64()?, r.u64()?, r.u64()?, r.u64()?];
        if counters.contains(&0) {
            return Err(GraphAuthoringError::Identity);
        }
        let nodes = r.count(limits.graph.nodes)?;
        let groups = r.count(limits.groups)?;
        let edges = r.count(limits.graph.edges)?;
        let mut graph = Graph::new();
        let mut positions = BTreeMap::new();
        let mut memberships = Vec::new();
        let mut last_node = 0;
        for _ in 0..nodes {
            let id = NodeId(r.identity(last_node)?);
            last_node = id.0;
            let name = r.text(limits.graph.text_bytes)?;
            let payload = r.blob(limits.graph.payload_bytes)?.to_vec();
            let position = GraphPoint {
                x: r.i32()?,
                y: r.i32()?,
            };
            point_valid(position)?;
            let group = r.u64()?;
            let count = r.count(limits.graph.ports_per_node)?;
            let mut ports = Vec::new();
            let mut specs = Vec::new();
            for _ in 0..count {
                let pid = PortId(r.identity(0)?);
                if graph.ports.contains_key(&pid) {
                    return Err(GraphAuthoringError::Identity);
                }
                let spec = PortSpec {
                    name: r.text(limits.graph.text_bytes)?,
                    port_type: PortType::named(r.text(limits.graph.text_bytes)?),
                    direction: match r.byte()? {
                        0 => PortDirection::Input,
                        1 => PortDirection::Output,
                        _ => return Err(GraphAuthoringError::Codec),
                    },
                    connection_policy: match r.byte()? {
                        0 => ConnectionPolicy::Single,
                        1 => ConnectionPolicy::Multiple,
                        _ => return Err(GraphAuthoringError::Codec),
                    },
                };
                graph.ports.insert(
                    pid,
                    Port {
                        id: pid,
                        node: id,
                        name: spec.name.clone(),
                        port_type: spec.port_type.clone(),
                        direction: spec.direction,
                        connection_policy: spec.connection_policy,
                    },
                );
                graph.port_names.insert((id, spec.name.clone()), pid);
                ports.push(pid);
                specs.push(spec);
            }
            graph.validate_node_spec(&NodeSpec {
                name: name.clone(),
                payload: Vec::new(),
                ports: specs,
            })?;
            graph.nodes.insert(
                id,
                Node {
                    id,
                    name,
                    payload,
                    group: None,
                    ports,
                },
            );
            positions.insert(id, position);
            if group != 0 {
                memberships.push((id, GroupId(group)));
            }
        }
        let mut parents = Vec::new();
        let mut last_group = 0;
        for _ in 0..groups {
            let id = GroupId(r.identity(last_group)?);
            last_group = id.0;
            let name = r.text(limits.graph.text_bytes)?;
            let parent = r.u64()?;
            graph.groups.insert(
                id,
                Group {
                    id,
                    name,
                    parent: None,
                    nodes: BTreeSet::new(),
                    children: BTreeSet::new(),
                },
            );
            if parent != 0 {
                parents.push((id, GroupId(parent)));
            }
        }
        for (id, parent) in parents {
            graph.set_group_parent(id, Some(parent))?;
        }
        for (id, group) in memberships {
            graph.set_node_group(id, Some(group))?;
        }
        let mut last_edge = 0;
        for _ in 0..edges {
            let id = EdgeId(r.identity(last_edge)?);
            last_edge = id.0;
            let output = PortId(r.identity(0)?);
            let input = PortId(r.identity(0)?);
            graph.next_edge = id.0;
            graph.connect(output, input)?;
        }
        if r.offset != bytes.len() {
            return Err(GraphAuthoringError::Codec);
        }
        (
            graph.next_node,
            graph.next_port,
            graph.next_edge,
            graph.next_group,
        ) = (counters[0], counters[1], counters[2], counters[3]);
        Self::new(graph, positions, limits)
    }
}
fn point_valid(p: GraphPoint) -> Result<(), GraphAuthoringError> {
    if p.x.unsigned_abs() > 1_000_000 || p.y.unsigned_abs() > 1_000_000 {
        Err(GraphAuthoringError::Limit)
    } else {
        Ok(())
    }
}
fn validate_document(
    g: &Graph<Vec<u8>>,
    positions: &BTreeMap<NodeId, GraphPoint>,
    l: GraphDocumentLimits,
) -> Result<(), GraphAuthoringError> {
    l.validate()?;
    if g.nodes.len() > l.graph.nodes
        || g.edges.len() > l.graph.edges
        || g.groups.len() > l.groups
        || positions.len() != g.nodes.len()
    {
        return Err(GraphAuthoringError::Limit);
    }
    for (next, max) in [
        (g.next_node, g.nodes.keys().next_back().map_or(0, |n| n.0)),
        (g.next_port, g.ports.keys().next_back().map_or(0, |n| n.0)),
        (g.next_edge, g.edges.keys().next_back().map_or(0, |n| n.0)),
        (g.next_group, g.groups.keys().next_back().map_or(0, |n| n.0)),
    ] {
        if next == 0 || next <= max {
            return Err(GraphAuthoringError::Identity);
        }
    }
    for node in g.nodes.values() {
        if node.ports.len() > l.graph.ports_per_node || node.payload.len() > l.graph.payload_bytes {
            return Err(GraphAuthoringError::Limit);
        }
        valid_text(&node.name, l.graph.text_bytes)?;
        point_valid(*positions.get(&node.id).ok_or(GraphAuthoringError::Codec)?)?;
        for id in &node.ports {
            let p = &g.ports[id];
            valid_text(&p.name, l.graph.text_bytes)?;
            valid_text(p.port_type.key(), l.graph.text_bytes)?;
        }
    }
    for group in g.groups.values() {
        valid_text(&group.name, l.graph.text_bytes)?;
    }
    g.validate()?;
    Ok(())
}
fn valid_text(s: &str, max: usize) -> Result<(), GraphAuthoringError> {
    if s.is_empty() {
        Err(GraphAuthoringError::Codec)
    } else if s.len() > max {
        Err(GraphAuthoringError::Limit)
    } else {
        Ok(())
    }
}
struct DocumentWriter {
    bytes: Vec<u8>,
    max: usize,
}
impl DocumentWriter {
    fn put(&mut self, bytes: &[u8]) -> Result<(), GraphAuthoringError> {
        if bytes.len() > self.max.saturating_sub(self.bytes.len()) {
            return Err(GraphAuthoringError::Limit);
        }
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }
    fn u64(&mut self, n: u64) -> Result<(), GraphAuthoringError> {
        self.put(&n.to_le_bytes())
    }
    fn u32(&mut self, n: usize) -> Result<(), GraphAuthoringError> {
        self.put(&(n as u32).to_le_bytes())
    }
    fn blob(&mut self, b: &[u8]) -> Result<(), GraphAuthoringError> {
        self.u32(b.len())?;
        self.put(b)
    }
}
struct DocumentReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}
impl<'a> DocumentReader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], GraphAuthoringError> {
        let end = self
            .offset
            .checked_add(n)
            .filter(|end| *end <= self.bytes.len())
            .ok_or(GraphAuthoringError::Codec)?;
        let bytes = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(bytes)
    }
    fn byte(&mut self) -> Result<u8, GraphAuthoringError> {
        Ok(self.take(1)?[0])
    }
    fn u64(&mut self) -> Result<u64, GraphAuthoringError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("fixed width"),
        ))
    }
    fn i32(&mut self) -> Result<i32, GraphAuthoringError> {
        Ok(i32::from_le_bytes(
            self.take(4)?.try_into().expect("fixed width"),
        ))
    }
    fn count(&mut self, max: usize) -> Result<usize, GraphAuthoringError> {
        let n = u32::from_le_bytes(self.take(4)?.try_into().expect("fixed width")) as usize;
        if n > max {
            return Err(GraphAuthoringError::Limit);
        }
        if n > self.bytes.len() - self.offset {
            return Err(GraphAuthoringError::Codec);
        }
        Ok(n)
    }
    fn blob(&mut self, max: usize) -> Result<&'a [u8], GraphAuthoringError> {
        let n = self.count(max)?;
        self.take(n)
    }
    fn text(&mut self, max: usize) -> Result<String, GraphAuthoringError> {
        let s = core::str::from_utf8(self.blob(max)?).map_err(|_| GraphAuthoringError::Codec)?;
        valid_text(s, max)?;
        Ok(s.into())
    }
    fn identity(&mut self, last: u64) -> Result<u64, GraphAuthoringError> {
        let n = self.u64()?;
        if n <= last || n == u64::MAX {
            Err(GraphAuthoringError::Identity)
        } else {
            Ok(n)
        }
    }
}

/// Hard bound on combined running, queued and unconsumed result slots.
pub const MAX_GRAPH_STORAGE_REQUESTS: usize = 8;
/// Hard bound on each document or expected-revision byte string.
pub const MAX_GRAPH_STORAGE_BYTES: usize = 64 * 1024 * 1024;

/// Resource policy for one serial worker and its retained results.
#[derive(Clone, Copy, Debug)]
pub struct GraphStorageLimits {
    /// Combined running, queued and unconsumed result slots.
    pub max_requests: usize,
    /// Serialized-byte cap checked on admission and during reads.
    pub max_bytes: usize,
    /// Limits passed to the real document codec on the worker.
    pub document: GraphDocumentLimits,
}
impl Default for GraphStorageLimits {
    fn default() -> Self {
        Self {
            max_requests: 2,
            max_bytes: 8 * 1024 * 1024,
            document: GraphDocumentLimits::default(),
        }
    }
}

/// Exact revision. Absence differs from an existing zero-byte file. Cloning
/// shares immutable bytes; equality compares all bytes, never mtime.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GraphFileRevision {
    /// Target is expected not to exist.
    Missing,
    /// Exact file contents. Submission enforces the service byte cap.
    Present(Arc<[u8]>),
}

/// Validated candidate; adoption remains an explicit host decision.
#[derive(Debug)]
pub struct GraphLoadCandidate {
    /// Real decoded document, independent from the live editor.
    pub document: GraphDocument,
    /// Exact observed bytes for a later conditional save.
    pub revision: GraphFileRevision,
}

/// Persistence confirmation after successful rename.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GraphSaveDurability {
    /// Staged file and parent directory were synced successfully.
    Synced,
    /// Rename succeeded, but directory sync failed. The file is published.
    Uncertain {
        /// Portable OS error category.
        kind: io::ErrorKind,
    },
}

/// Publication receipt. Never automatically marks a live editor clean.
#[derive(Clone, Debug)]
pub struct GraphSaveReceipt {
    /// Identity supplied with the bytes; compare with the current host token.
    pub snapshot_token: GraphSnapshotToken,
    /// Exact published bytes for the next conditional save.
    pub revision: GraphFileRevision,
    /// Explicit directory-sync outcome after publication.
    pub durability: GraphSaveDurability,
    /// Deadline passed after crossing the irrevocable publication fence.
    pub deadline_exceeded: bool,
}

/// Owned output, retained in a bounded request slot until polled/dropped.
#[derive(Debug)]
pub enum GraphStorageResult {
    /// Decoded candidate and exact file revision.
    Loaded(GraphLoadCandidate),
    /// Published file and original snapshot identity.
    Saved(GraphSaveReceipt),
}

/// Storage failures. Except `PublicationUncertain`, this operation did not replace
/// the target. External writers remain responsible for their own writes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GraphStorageError {
    /// Invalid bounds, absolute filename or unrepresentable deadline.
    InvalidConfiguration,
    /// Slots are full, including retained results, or the file lock is held.
    Busy,
    /// Input, expected revision or file exceeded the serialized-byte cap.
    ByteLimit,
    /// Target or lock is a symlink or not a regular file.
    UnsafePath,
    /// Exact current bytes differ from the submitted expected revision.
    Conflict,
    /// Loaded file did not exist.
    Missing,
    /// Cancelled before publication; late worker output is discarded.
    Cancelled,
    /// Deadline expired before publication; late worker output is discarded.
    TimedOut,
    /// Actual document codec rejected bytes; the target was not replaced.
    Codec(GraphAuthoringError),
    /// An OS operation failed before publication.
    Io {
        /// Fixed operation label.
        operation: &'static str,
        /// Portable OS category.
        kind: io::ErrorKind,
    },
    /// Rename returned an error after entering publication. Reload to reconcile
    /// disk before retrying; a remote filesystem may have acted.
    PublicationUncertain {
        /// Portable OS category from rename.
        kind: io::ErrorKind,
    },
    /// Worker is unavailable or request identities are exhausted.
    Unavailable,
    /// This handle's terminal output was already consumed.
    Consumed,
}
impl fmt::Display for GraphStorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { operation, kind } => write!(f, "graph storage {operation}: {kind}"),
            Self::PublicationUncertain { kind } => {
                write!(f, "graph publication uncertain: {kind}")
            }
            Self::Codec(error) => write!(f, "graph file: {error}"),
            other => write!(f, "graph storage {other:?}"),
        }
    }
}
impl std::error::Error for GraphStorageError {}

/// Whether cancellation can still prevent publication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphCancelOutcome {
    /// Fenced; this request cannot subsequently replace the file.
    Cancelled,
    /// Publication began or output was consumed. Inspect the eventual result.
    TooLate,
}

struct Shared {
    active: AtomicUsize,
    next: AtomicU64,
    stopping: AtomicBool,
}
#[derive(Clone, Copy, PartialEq)]
enum Phase {
    Working,
    Publishing,
    Complete,
}
struct State {
    phase: Phase,
    abort: Option<GraphStorageError>,
    output: Option<Result<GraphStorageResult, GraphStorageError>>,
}
struct Slot {
    id: u64,
    deadline: Instant,
    shared: Arc<Shared>,
    state: Mutex<State>,
    #[cfg(test)]
    hook: Option<Arc<TestHook>>,
}
impl Slot {
    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
    fn expire(&self, state: &mut State) {
        if state.phase == Phase::Working && state.abort.is_none() {
            let reason = if self.shared.stopping.load(Ordering::Acquire) {
                Some(GraphStorageError::Cancelled)
            } else if Instant::now() >= self.deadline {
                Some(GraphStorageError::TimedOut)
            } else {
                None
            };
            if let Some(reason) = reason {
                state.output = Some(Err(reason.clone()));
                state.abort = Some(reason);
            }
        }
    }
    fn check(&self) -> Result<(), GraphStorageError> {
        let mut state = self.state();
        self.expire(&mut state);
        state.abort.clone().map_or(Ok(()), Err)
    }
    fn publish(&self) -> Result<(), GraphStorageError> {
        let mut state = self.state();
        self.expire(&mut state);
        if let Some(error) = &state.abort {
            return Err(error.clone());
        }
        state.phase = Phase::Publishing;
        Ok(())
    }
    fn finish(&self, result: Result<GraphStorageResult, GraphStorageError>) {
        let mut state = self.state();
        self.expire(&mut state);
        if state.abort.is_none() {
            state.output = Some(result);
            state.phase = Phase::Complete;
        }
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        self.shared.active.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Nonblocking handle. Dropping fences unfinished work without joining a
/// potentially blocked OS call; its slot remains reserved until the worker exits.
pub struct GraphStorageRequest {
    slot: Arc<Slot>,
    consumed: bool,
}
impl GraphStorageRequest {
    /// Service-local monotonic identity for rejecting obsolete load results.
    pub fn id(&self) -> u64 {
        self.slot.id
    }
    /// Takes output once. A pre-publication timeout immediately fences later
    /// output. Calling this after consuming output returns Consumed.
    pub fn poll(&mut self) -> Poll<Result<GraphStorageResult, GraphStorageError>> {
        if self.consumed {
            return Poll::Ready(Err(GraphStorageError::Consumed));
        }
        let mut state = self.slot.state();
        self.slot.expire(&mut state);
        if let Some(output) = state.output.take() {
            self.consumed = true;
            Poll::Ready(output)
        } else {
            Poll::Pending
        }
    }
    /// Fences queued/running work or an unconsumed load candidate. Save
    /// publication is irrevocable: `TooLate` preserves its actual outcome.
    pub fn cancel(&self) -> GraphCancelOutcome {
        let mut state = self.slot.state();
        if self.consumed
            || state.phase == Phase::Publishing
            || matches!(
                state.output,
                Some(Ok(GraphStorageResult::Saved(_)))
                    | Some(Err(GraphStorageError::PublicationUncertain { .. }))
            )
        {
            return GraphCancelOutcome::TooLate;
        }
        state.abort = Some(GraphStorageError::Cancelled);
        state.output = Some(Err(GraphStorageError::Cancelled));
        GraphCancelOutcome::Cancelled
    }
}
impl Drop for GraphStorageRequest {
    fn drop(&mut self) {
        if !self.consumed {
            self.cancel();
        }
    }
}

enum Operation {
    Load,
    Save {
        bytes: Vec<u8>,
        snapshot_token: GraphSnapshotToken,
        expected: GraphFileRevision,
    },
}
struct Job {
    slot: Arc<Slot>,
    operation: Operation,
}

/// One bounded worker for one absolute filename in an existing trusted parent.
/// Construction/admission/poll do no filesystem I/O; all path checks, reads,
/// codec validation, stage writes and sync/rename happen on the worker.
pub struct GraphStorageService {
    sender: SyncSender<Job>,
    shared: Arc<Shared>,
    limits: GraphStorageLimits,
    #[cfg(test)]
    hook: Option<Arc<TestHook>>,
}
impl GraphStorageService {
    /// Starts the worker. Parent must exist when used and remain trusted/stable.
    /// Dropping the service fences pre-publication work without blocking to join.
    pub fn new(
        path: impl Into<PathBuf>,
        limits: GraphStorageLimits,
    ) -> Result<Self, GraphStorageError> {
        let path = path.into();
        if !path.is_absolute()
            || path.file_name().is_none()
            || path.components().any(|c| c == Component::ParentDir)
            || limits.max_requests == 0
            || limits.max_requests > MAX_GRAPH_STORAGE_REQUESTS
            || limits.max_bytes == 0
            || limits.max_bytes > MAX_GRAPH_STORAGE_BYTES
        {
            return Err(GraphStorageError::InvalidConfiguration);
        }
        limits
            .document
            .validate()
            .map_err(|_| GraphStorageError::InvalidConfiguration)?;
        let (sender, receiver) = mpsc::sync_channel::<Job>(limits.max_requests);
        let shared = Arc::new(Shared {
            active: AtomicUsize::new(0),
            next: AtomicU64::new(1),
            stopping: AtomicBool::new(false),
        });
        thread::Builder::new()
            .name("prism-graph-storage".into())
            .spawn(move || {
                for job in receiver {
                    let output = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        run(&path, limits, &job.slot, job.operation)
                    }))
                    .unwrap_or_else(|_| {
                        if job.slot.state().phase == Phase::Publishing {
                            Err(GraphStorageError::PublicationUncertain {
                                kind: io::ErrorKind::Other,
                            })
                        } else {
                            Err(GraphStorageError::Unavailable)
                        }
                    });
                    job.slot.finish(output);
                }
            })
            .map_err(|e| os_error("spawn worker", e))?;
        Ok(Self {
            sender,
            shared,
            limits,
            #[cfg(test)]
            hook: None,
        })
    }
    /// Queues actual bounded read/decode without changing any host editor.
    pub fn load(&self, timeout: Duration) -> Result<GraphStorageRequest, GraphStorageError> {
        self.submit(Operation::Load, timeout)
    }
    /// Takes serialized bytes and token ownership. Worker validates the document
    /// before saving. Compare the receipt token before marking a live editor clean.
    pub fn save(
        &self,
        bytes: Vec<u8>,
        snapshot_token: GraphSnapshotToken,
        expected: GraphFileRevision,
        timeout: Duration,
    ) -> Result<GraphStorageRequest, GraphStorageError> {
        if bytes.len() > self.limits.max_bytes
            || bytes.capacity() > self.limits.max_bytes
            || matches!(&expected, GraphFileRevision::Present(bytes) if bytes.len() > self.limits.max_bytes)
        {
            return Err(GraphStorageError::ByteLimit);
        }
        self.submit(
            Operation::Save {
                bytes,
                snapshot_token,
                expected,
            },
            timeout,
        )
    }
    fn submit(
        &self,
        operation: Operation,
        timeout: Duration,
    ) -> Result<GraphStorageRequest, GraphStorageError> {
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or(GraphStorageError::InvalidConfiguration)?;
        if self.shared.stopping.load(Ordering::Acquire) {
            return Err(GraphStorageError::Unavailable);
        }
        self.shared
            .active
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < self.limits.max_requests).then_some(n + 1)
            })
            .map_err(|_| GraphStorageError::Busy)?;
        let Ok(id) = self
            .shared
            .next
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1))
        else {
            self.shared.active.fetch_sub(1, Ordering::AcqRel);
            return Err(GraphStorageError::Unavailable);
        };
        let slot = Arc::new(Slot {
            id,
            deadline,
            shared: self.shared.clone(),
            state: Mutex::new(State {
                phase: Phase::Working,
                abort: None,
                output: None,
            }),
            #[cfg(test)]
            hook: self.hook.clone(),
        });
        let request = GraphStorageRequest {
            slot: slot.clone(),
            consumed: false,
        };
        match self.sender.try_send(Job { slot, operation }) {
            Ok(()) => Ok(request),
            Err(TrySendError::Full(_)) => Err(GraphStorageError::Busy),
            Err(TrySendError::Disconnected(_)) => Err(GraphStorageError::Unavailable),
        }
    }
}
impl Drop for GraphStorageService {
    fn drop(&mut self) {
        self.shared.stopping.store(true, Ordering::Release);
    }
}

fn os_error(operation: &'static str, error: io::Error) -> GraphStorageError {
    GraphStorageError::Io {
        operation,
        kind: error.kind(),
    }
}
fn regular_or_missing(path: &Path) -> Result<bool, GraphStorageError> {
    match fs::symlink_metadata(path) {
        Ok(meta) if meta.is_file() && !meta.file_type().is_symlink() => Ok(true),
        Ok(_) => Err(GraphStorageError::UnsafePath),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(os_error("inspect path", e)),
    }
}
fn read(path: &Path, max: usize, slot: &Slot) -> Result<GraphFileRevision, GraphStorageError> {
    slot.check()?;
    if !regular_or_missing(path)? {
        return Ok(GraphFileRevision::Missing);
    }
    let mut file = File::open(path).map_err(|e| os_error("open", e))?;
    if file
        .metadata()
        .map_err(|e| os_error("inspect file", e))?
        .len()
        > max as u64
    {
        return Err(GraphStorageError::ByteLimit);
    }
    let mut bytes = Vec::new();
    let mut block = [0; 16 * 1024];
    loop {
        slot.check()?;
        let limit = block.len().min(max.saturating_sub(bytes.len()) + 1);
        let count = file
            .read(&mut block[..limit])
            .map_err(|e| os_error("read", e))?;
        if count == 0 {
            break;
        }
        if count > max - bytes.len() {
            return Err(GraphStorageError::ByteLimit);
        }
        bytes.extend_from_slice(&block[..count]);
    }
    slot.check()?;
    Ok(GraphFileRevision::Present(bytes.into()))
}
fn lock(path: &Path) -> Result<File, GraphStorageError> {
    regular_or_missing(path)?;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(|e| os_error("open lock", e))?;
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(TryLockError::WouldBlock) => Err(GraphStorageError::Busy),
        Err(TryLockError::Error(e)) => Err(os_error("lock", e)),
    }
}
struct Stage(PathBuf);
impl Drop for Stage {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
fn stage(path: &Path) -> Result<(Stage, File), GraphStorageError> {
    let path = stage_path(path);
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|e| os_error("create stage", e))?;
    Ok((Stage(path), file))
}
fn stage_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().expect("validated filename").to_os_string();
    name.push(".prism-graph.stage");
    path.with_file_name(name)
}
fn run(
    requested: &Path,
    limits: GraphStorageLimits,
    slot: &Slot,
    operation: Operation,
) -> Result<GraphStorageResult, GraphStorageError> {
    slot.check()?;
    let parent = fs::canonicalize(requested.parent().expect("absolute filename"))
        .map_err(|e| os_error("resolve parent", e))?;
    let path = parent.join(requested.file_name().expect("validated filename"));
    let mut lock_name = path.file_name().expect("validated filename").to_os_string();
    lock_name.push(".prism-graph.lock");
    let _lock = lock(&path.with_file_name(lock_name))?;
    // One reserved stage per target: only the holder of this target lock can
    // create it. A stage left by a killed writer is uncommitted and discarded.
    // Exact-path cleanup is bounded and never scans unrelated directory files.
    let abandoned = stage_path(&path);
    if regular_or_missing(&abandoned)? {
        fs::remove_file(&abandoned).map_err(|e| os_error("recover stage", e))?;
    }
    let current = read(&path, limits.max_bytes, slot)?;
    match operation {
        Operation::Load => {
            let GraphFileRevision::Present(bytes) = &current else {
                return Err(GraphStorageError::Missing);
            };
            let document = GraphDocument::decode(bytes, limits.document, limits.max_bytes)
                .map_err(GraphStorageError::Codec)?;
            slot.check()?;
            Ok(GraphStorageResult::Loaded(GraphLoadCandidate {
                document,
                revision: current,
            }))
        }
        Operation::Save {
            bytes,
            snapshot_token,
            expected,
        } => {
            if current != expected {
                return Err(GraphStorageError::Conflict);
            }
            GraphDocument::decode(&bytes, limits.document, limits.max_bytes)
                .map_err(GraphStorageError::Codec)?;
            slot.check()?;
            let (stage, mut file) = stage(&path)?;
            for block in bytes.chunks(16 * 1024) {
                slot.check()?;
                file.write_all(block)
                    .map_err(|e| os_error("write stage", e))?;
            }
            file.sync_all().map_err(|e| os_error("sync stage", e))?;
            drop(file);
            #[cfg(test)]
            slot.checkpoint(Checkpoint::Staged);
            // Recheck after fsync to detect external changes during staging.
            if read(&path, limits.max_bytes, slot)? != expected {
                return Err(GraphStorageError::Conflict);
            }
            slot.publish()?;
            #[cfg(test)]
            slot.checkpoint(Checkpoint::Publishing);
            fs::rename(&stage.0, &path)
                .map_err(|e| GraphStorageError::PublicationUncertain { kind: e.kind() })?;
            #[cfg(test)]
            slot.checkpoint(Checkpoint::Renamed);
            let durability = match File::open(&parent).and_then(|dir| dir.sync_all()) {
                Ok(()) => GraphSaveDurability::Synced,
                Err(e) => GraphSaveDurability::Uncertain { kind: e.kind() },
            };
            Ok(GraphStorageResult::Saved(GraphSaveReceipt {
                snapshot_token,
                revision: GraphFileRevision::Present(bytes.into()),
                durability,
                deadline_exceeded: Instant::now() >= slot.deadline,
            }))
        }
    }
}

#[cfg(test)]
#[derive(Clone, Copy, PartialEq)]
enum Checkpoint {
    Staged,
    Publishing,
    Renamed,
}
#[cfg(test)]
struct TestHook {
    point: Checkpoint,
    entered: SyncSender<()>,
    resume: Mutex<mpsc::Receiver<()>>,
}
#[cfg(test)]
impl Slot {
    fn checkpoint(&self, point: Checkpoint) {
        if let Some(hook) = &self.hook
            && hook.point == point
        {
            hook.entered.send(()).unwrap();
            hook.resume.lock().unwrap().recv().unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static SERIAL: AtomicU64 = AtomicU64::new(1);
    const WAIT: Duration = Duration::from_secs(10);
    struct Project(PathBuf);
    impl Project {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "prism-graph-fence-{}-{}",
                std::process::id(),
                SERIAL.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&root).unwrap();
            Self(root)
        }
        fn path(&self) -> PathBuf {
            self.0.join("take.graph")
        }
    }
    impl Drop for Project {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }
    fn bytes(rate: u64) -> Vec<u8> {
        let mut graph = Graph::new();
        let id = graph
            .add_node(NodeSpec::new(format!("node-{rate}"), vec![rate as u8]))
            .unwrap();
        GraphDocument::new(
            graph,
            [(id, GraphPoint::default())].into(),
            GraphDocumentLimits::default(),
        )
        .unwrap()
        .encode(GraphDocumentLimits::default(), 1024)
        .unwrap()
    }
    fn token() -> GraphSnapshotToken {
        GraphEditor::new(GraphAuthoringLimits::default())
            .unwrap()
            .snapshot_token()
    }
    fn wait(request: &mut GraphStorageRequest) -> Result<GraphStorageResult, GraphStorageError> {
        let until = Instant::now() + WAIT;
        loop {
            if let Poll::Ready(result) = request.poll() {
                return result;
            }
            assert!(Instant::now() < until);
            thread::sleep(Duration::from_millis(1));
        }
    }
    struct Gate {
        entered: mpsc::Receiver<()>,
        resume: Option<SyncSender<()>>,
    }
    impl Gate {
        fn arrived(&self) {
            self.entered.recv_timeout(WAIT).unwrap();
        }
        fn release(&mut self) {
            self.resume.take().unwrap().send(()).unwrap();
        }
    }
    impl Drop for Gate {
        fn drop(&mut self) {
            if let Some(sender) = self.resume.take() {
                let _ = sender.send(());
            }
        }
    }
    fn service(project: &Project, point: Checkpoint) -> (GraphStorageService, Gate) {
        let (entered_tx, entered) = mpsc::sync_channel(1);
        let (resume, resume_rx) = mpsc::sync_channel(1);
        let mut service = GraphStorageService::new(
            project.path(),
            GraphStorageLimits {
                max_requests: 4,
                ..Default::default()
            },
        )
        .unwrap();
        service.hook = Some(Arc::new(TestHook {
            point,
            entered: entered_tx,
            resume: Mutex::new(resume_rx),
        }));
        (
            service,
            Gate {
                entered,
                resume: Some(resume),
            },
        )
    }
    fn submit(service: &GraphStorageService, timeout: Duration) -> GraphStorageRequest {
        service
            .save(
                bytes(240),
                token(),
                GraphFileRevision::Present(bytes(120).into()),
                timeout,
            )
            .unwrap()
    }
    fn flush(service: &GraphStorageService) {
        let mut load = service.load(WAIT).unwrap();
        assert!(matches!(wait(&mut load), Ok(GraphStorageResult::Loaded(_))));
    }

    #[test]
    fn real_staged_save_cancelled_before_rename_never_publishes_or_leaks_stage() {
        let project = Project::new();
        fs::write(project.path(), bytes(120)).unwrap();
        let (service, mut gate) = service(&project, Checkpoint::Staged);
        let mut request = submit(&service, WAIT);
        gate.arrived();
        assert_eq!(
            fs::read_dir(&project.0).unwrap().count(),
            3,
            "actual fsynced stage exists"
        );
        assert_eq!(request.cancel(), GraphCancelOutcome::Cancelled);
        assert!(matches!(
            wait(&mut request),
            Err(GraphStorageError::Cancelled)
        ));
        gate.release();
        flush(&service);
        assert_eq!(fs::read(project.path()).unwrap(), bytes(120));
        assert_eq!(fs::read_dir(&project.0).unwrap().count(), 2);
        assert!(matches!(
            request.poll(),
            Poll::Ready(Err(GraphStorageError::Consumed))
        ));
    }

    #[test]
    fn deadline_fences_a_real_fsynced_stage_and_discards_late_worker_result() {
        let project = Project::new();
        fs::write(project.path(), bytes(120)).unwrap();
        let (service, mut gate) = service(&project, Checkpoint::Staged);
        let mut request = submit(&service, Duration::from_secs(2));
        gate.arrived();
        thread::sleep(
            request
                .slot
                .deadline
                .saturating_duration_since(Instant::now())
                + Duration::from_millis(1),
        );
        assert!(matches!(
            wait(&mut request),
            Err(GraphStorageError::TimedOut)
        ));
        gate.release();
        flush(&service);
        assert_eq!(fs::read(project.path()).unwrap(), bytes(120));
        assert_eq!(fs::read_dir(&project.0).unwrap().count(), 2);
        assert!(matches!(
            request.poll(),
            Poll::Ready(Err(GraphStorageError::Consumed))
        ));
    }

    #[test]
    fn post_fence_cancel_timeout_and_service_drop_preserve_actual_publication_receipt() {
        let project = Project::new();
        fs::write(project.path(), bytes(120)).unwrap();
        let (service, mut gate) = service(&project, Checkpoint::Publishing);
        let mut request = submit(&service, Duration::from_secs(2));
        gate.arrived();
        assert_eq!(request.cancel(), GraphCancelOutcome::TooLate);
        drop(service);
        thread::sleep(
            request
                .slot
                .deadline
                .saturating_duration_since(Instant::now())
                + Duration::from_millis(1),
        );
        assert!(
            matches!(request.poll(), Poll::Pending),
            "deadline must not hide irrevocable publication"
        );
        gate.release();
        let GraphStorageResult::Saved(receipt) = wait(&mut request).unwrap() else {
            panic!()
        };
        assert!(receipt.deadline_exceeded);
        assert_eq!(receipt.durability, GraphSaveDurability::Synced);
        assert_eq!(fs::read(project.path()).unwrap(), bytes(240));
    }

    #[test]
    fn dropping_service_and_queued_handle_fences_real_prepublication_work() {
        let project = Project::new();
        fs::write(project.path(), bytes(120)).unwrap();
        let (service, mut gate) = service(&project, Checkpoint::Staged);
        let mut first = submit(&service, WAIT);
        gate.arrived();
        let queued = service
            .save(
                bytes(480),
                token(),
                GraphFileRevision::Present(bytes(120).into()),
                WAIT,
            )
            .unwrap();
        let shared = service.shared.clone();
        drop(queued);
        drop(service);
        assert!(matches!(
            wait(&mut first),
            Err(GraphStorageError::Cancelled)
        ));
        drop(first);
        gate.release();
        let until = Instant::now() + WAIT;
        while shared.active.load(Ordering::Acquire) != 0 {
            assert!(Instant::now() < until);
            thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(fs::read(project.path()).unwrap(), bytes(120));
        assert_eq!(fs::read_dir(&project.0).unwrap().count(), 2);
    }

    #[test]
    fn external_change_after_stage_fsync_is_detected_by_final_exact_byte_recheck() {
        let project = Project::new();
        fs::write(project.path(), bytes(120)).unwrap();
        let (service, mut gate) = service(&project, Checkpoint::Staged);
        let mut request = submit(&service, WAIT);
        gate.arrived();
        fs::write(project.path(), bytes(480)).unwrap();
        gate.release();
        assert!(matches!(
            wait(&mut request),
            Err(GraphStorageError::Conflict)
        ));
        assert_eq!(fs::read(project.path()).unwrap(), bytes(480));
        assert_eq!(fs::read_dir(&project.0).unwrap().count(), 2);
    }

    #[test]
    fn real_rename_failure_reports_publication_uncertainty_and_cleans_stage() {
        let project = Project::new();
        fs::write(project.path(), bytes(120)).unwrap();
        let (service, mut gate) = service(&project, Checkpoint::Publishing);
        let mut request = submit(&service, WAIT);
        gate.arrived();
        fs::remove_file(project.path()).unwrap();
        fs::create_dir(project.path()).unwrap();
        gate.release();
        assert!(matches!(
            wait(&mut request),
            Err(GraphStorageError::PublicationUncertain { .. })
        ));
        assert!(project.path().is_dir());
        assert_eq!(fs::read_dir(&project.0).unwrap().count(), 2);
    }

    #[test]
    fn real_directory_sync_open_failure_still_reports_published_bytes_and_uncertain_durability() {
        let mut project = Project::new();
        fs::write(project.path(), bytes(120)).unwrap();
        let (service, mut gate) = service(&project, Checkpoint::Renamed);
        let mut request = submit(&service, WAIT);
        gate.arrived();
        let moved = project.0.with_extension("moved");
        fs::rename(&project.0, &moved).unwrap();
        project.0 = moved;
        gate.release();
        let GraphStorageResult::Saved(receipt) = wait(&mut request).unwrap() else {
            panic!()
        };
        assert_eq!(
            receipt.durability,
            GraphSaveDurability::Uncertain {
                kind: io::ErrorKind::NotFound
            }
        );
        assert_eq!(
            receipt.revision,
            GraphFileRevision::Present(bytes(240).into())
        );
        assert_eq!(fs::read(project.path()).unwrap(), bytes(240));
    }

    #[test]
    fn killed_writer_releases_os_lock_and_recovers_before_and_after_rename() {
        const CHILD: &str = "PRISM_GRAPH_KILL_ROOT";
        const POINT: &str = "PRISM_GRAPH_KILL_POINT";
        if let Some(root) = std::env::var_os(CHILD) {
            let project = Project(root.into());
            let point = if std::env::var(POINT).unwrap() == "stage" {
                Checkpoint::Staged
            } else {
                Checkpoint::Renamed
            };
            let (service, gate) = service(&project, point);
            let _request = submit(&service, WAIT);
            gate.arrived();
            fs::write(project.0.join("ready"), b"ready").unwrap();
            loop {
                thread::park();
            }
        }
        struct Child(std::process::Child);
        impl Drop for Child {
            fn drop(&mut self) {
                let _ = self.0.kill();
                let _ = self.0.wait();
            }
        }
        for point in ["stage", "renamed"] {
            let project = Project::new();
            fs::write(project.path(), bytes(120)).unwrap();
            let mut child = Child(
                std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "graph::storage::tests::killed_writer_releases_os_lock_and_recovers_before_and_after_rename",
                    ])
                    .env(CHILD, &project.0)
                    .env(POINT, point)
                    .stdout(std::process::Stdio::null())
                    .spawn()
                    .unwrap(),
            );
            let until = Instant::now() + WAIT;
            while !project.0.join("ready").exists() {
                assert!(child.0.try_wait().unwrap().is_none(), "child exited early");
                assert!(Instant::now() < until, "child checkpoint timed out");
                thread::sleep(Duration::from_millis(1));
            }
            let recovery =
                GraphStorageService::new(project.path(), GraphStorageLimits::default()).unwrap();
            assert!(
                matches!(
                    wait(&mut recovery.load(WAIT).unwrap()),
                    Err(GraphStorageError::Busy)
                ),
                "live subprocess holds the actual OS lock"
            );
            child.0.kill().unwrap();
            assert!(!child.0.wait().unwrap().success());
            let expected = if point == "stage" {
                bytes(120)
            } else {
                bytes(240)
            };
            assert_eq!(fs::read(project.path()).unwrap(), expected);
            assert_eq!(stage_path(&project.path()).exists(), point == "stage");
            let GraphStorageResult::Loaded(candidate) =
                wait(&mut recovery.load(WAIT).unwrap()).unwrap()
            else {
                panic!("expected recovered load")
            };
            assert_eq!(
                candidate.revision,
                GraphFileRevision::Present(expected.into())
            );
            assert!(!stage_path(&project.path()).exists());
            assert!(matches!(
                wait(
                    &mut recovery
                        .save(bytes(480), token(), candidate.revision, WAIT)
                        .unwrap()
                ),
                Ok(GraphStorageResult::Saved(_))
            ));
            assert_eq!(fs::read(project.path()).unwrap(), bytes(480));
        }
    }
}
