//! Isolated manifest readers and durable redo transactions.
use super::*;
use crate::{ActorFileIdentity, ActorRevisionBase, StableGuid};
use std::{collections::BTreeSet, time::Duration};

mod disk;
mod format;
#[cfg(test)]
mod tests;

const STORE: &str = ".prism-world";
const GATE: &str = ".prism-world.lock";
const MAX_ENTRIES: usize = 4096;

/// A numeric, schema-neutral file identity in a transactional world.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WorldFileTarget {
    /// Opaque cell payload.
    Cell(CellFileIdentity),
    /// Opaque actor payload and its owning cell coordinate.
    Actor(ActorFileIdentity),
}
impl WorldFileTarget {
    fn key(&self) -> (u8, i32, i32, i32, u128) {
        let (kind, c, id) = match self {
            Self::Cell(f) => (0, f.coord(), f.stable_id()),
            Self::Actor(f) => (1, f.cell(), f.actor()),
        };
        (kind, c.x, c.y, c.z, id.to_u128())
    }
    pub(super) fn envelope(&self) -> EnvelopeIdentity {
        match self {
            Self::Cell(f) => EnvelopeIdentity::cell(f),
            Self::Actor(f) => EnvelopeIdentity {
                magic: *b"PEWACTR\0",
                coord: f.cell(),
                guid: f.actor(),
            },
        }
    }
    /// Coordinate of the cell payload or actor owner.
    pub fn cell(&self) -> crate::CellCoord {
        self.envelope().coord
    }
    fn validate(&self) -> Result<(), FileCellIoError> {
        if self.envelope().guid.is_nil() {
            return Err(FileCellIoError::InvalidRequest);
        }
        if let Self::Cell(f) = self {
            let c = f.coord();
            if f.relative_path() != format!("cells/{}_{}_{}.prismcell", c.x, c.y, c.z) {
                return Err(FileCellIoError::InvalidRequest);
            }
        }
        Ok(())
    }
}

/// One explicit CAS mutation. None payload deletes an existing target.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorldFileChange {
    target: WorldFileTarget,
    base: ActorRevisionBase,
    value: Option<(u64, Arc<[u8]>)>,
}
impl WorldFileChange {
    /// Replaces or creates an opaque envelope with a strictly advancing revision.
    pub fn put(
        target: WorldFileTarget,
        base: ActorRevisionBase,
        revision: u64,
        payload: Arc<[u8]>,
    ) -> Result<Self, FileCellIoError> {
        target.validate()?;
        if matches!(base, ActorRevisionBase::Revision(old) if revision <= old) {
            return Err(FileCellIoError::InvalidRequest);
        }
        Ok(Self {
            target,
            base,
            value: Some((revision, payload)),
        })
    }
    /// Deletes exactly the specified durable revision.
    pub fn delete(target: WorldFileTarget, base: u64) -> Result<Self, FileCellIoError> {
        target.validate()?;
        Ok(Self {
            target,
            base: ActorRevisionBase::Revision(base),
            value: None,
        })
    }
    /// Target being compared and changed.
    pub fn target(&self) -> &WorldFileTarget {
        &self.target
    }
    /// Required existing revision, or create-only.
    pub fn base(&self) -> ActorRevisionBase {
        self.base
    }
    /// New revision and opaque bytes, or deletion.
    pub fn value(&self) -> Option<(u64, &[u8])> {
        self.value.as_ref().map(|(r, b)| (*r, &**b))
    }
}

/// Canonically sorted, immutable transaction input. IDs must be unique per commit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorldTransactionRequest {
    id: StableGuid,
    base_generation: u64,
    changes: Vec<WorldFileChange>,
}
impl WorldTransactionRequest {
    /// Rejects nil IDs, duplicate targets and empty/unbounded mutation lists.
    pub fn new(
        id: StableGuid,
        base_generation: u64,
        mut changes: Vec<WorldFileChange>,
    ) -> Result<Self, FileCellIoError> {
        if id.is_nil()
            || changes.is_empty()
            || changes.len() > MAX_ENTRIES * 2
            || base_generation == u64::MAX
        {
            return Err(FileCellIoError::InvalidRequest);
        }
        changes.sort_by_key(|c| c.target.key());
        if changes
            .windows(2)
            .any(|w| w[0].target.key() == w[1].target.key())
        {
            return Err(FileCellIoError::InvalidRequest);
        }
        Ok(Self {
            id,
            base_generation,
            changes,
        })
    }
    /// Caller-chosen stable idempotency identity.
    pub fn id(&self) -> StableGuid {
        self.id
    }
    /// Exact root generation being replaced.
    pub fn base_generation(&self) -> u64 {
        self.base_generation
    }
    /// Ordered per-target changes.
    pub fn changes(&self) -> &[WorldFileChange] {
        &self.changes
    }
}

/// Bounded whole-world snapshot and request admission policy.
#[derive(Clone, Copy, Debug)]
pub struct WorldTransactionLimits {
    entries: usize,
    file_bytes: usize,
    snapshot_bytes: usize,
    requests: usize,
    inflight_bytes: usize,
}
impl WorldTransactionLimits {
    /// At most 4096 files and a 64 MiB snapshot. In-flight bytes must admit one
    /// old snapshot plus one maximum input. Two disk slots bound stored history.
    pub fn new(
        entries: usize,
        file_bytes: usize,
        snapshot_bytes: usize,
        requests: usize,
        inflight_bytes: usize,
    ) -> Result<Self, FileCellIoError> {
        FileCellIoLimits::new(file_bytes, requests)?;
        if entries == 0
            || entries > MAX_ENTRIES
            || snapshot_bytes < file_bytes
            || snapshot_bytes > prism_scene::partition::MAX_CELL_BYTES - HEADER_BYTES
            || inflight_bytes < snapshot_bytes.saturating_mul(2)
        {
            return Err(FileCellIoError::InvalidConfiguration);
        }
        Ok(Self {
            entries,
            file_bytes,
            snapshot_bytes,
            requests,
            inflight_bytes,
        })
    }
}

/// One validated file in an owned generation snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorldFileSnapshot {
    target: WorldFileTarget,
    revision: u64,
    bytes: Arc<[u8]>,
}
impl WorldFileSnapshot {
    /// Stable file identity.
    pub fn target(&self) -> &WorldFileTarget {
        &self.target
    }
    /// Durable revision.
    pub fn revision(&self) -> u64 {
        self.revision
    }
    /// Opaque bytes owned by this snapshot.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// All files selected from exactly one published version root.
#[derive(Clone, Debug)]
pub struct WorldTransactionSnapshot {
    generation: u64,
    id: StableGuid,
    files: Vec<WorldFileSnapshot>,
    root: PathBuf,
}
impl WorldTransactionSnapshot {
    /// Monotonic root generation; zero denotes an empty store.
    pub fn generation(&self) -> u64 {
        self.generation
    }
    /// Transaction that published this generation, or nil for empty.
    pub fn transaction_id(&self) -> StableGuid {
        self.id
    }
    /// Canonically ordered opaque cell and actor files.
    pub fn files(&self) -> &[WorldFileSnapshot] {
        &self.files
    }
    /// Canonical isolated world root.
    pub fn root(&self) -> &Path {
        &self.root
    }
}

/// A durable publication proof retaining the exact submitted changes.
#[derive(Clone, Debug)]
pub struct WorldTransactionReceipt {
    request: WorldTransactionRequest,
    root: PathBuf,
}
impl WorldTransactionReceipt {
    /// Exact input whose complete generation was durably published.
    pub fn request(&self) -> &WorldTransactionRequest {
        &self.request
    }
    /// Canonical isolated root.
    pub fn root(&self) -> &Path {
        &self.root
    }
    /// Resulting root generation.
    pub fn generation(&self) -> u64 {
        self.request.base_generation + 1
    }
}

/// Whole-snapshot reader and multi-file durable transaction executor.
/// Existing legacy cell roots are rejected: import into a separate root.
#[derive(Clone)]
pub struct FileWorldTransactionProvider {
    io: FileActorIoProvider,
    limits: WorldTransactionLimits,
}
impl FileWorldTransactionProvider {
    /// Initializes an isolated root under its exclusive OS gate. Reopening
    /// retains the isolation marker even after an interrupted initialization.
    pub fn new(
        root: impl AsRef<Path>,
        limits: WorldTransactionLimits,
    ) -> Result<Self, FileCellIoError> {
        let io = FileActorIoProvider::new(
            root,
            ActorIoLimits::new(limits.file_bytes, limits.requests, limits.inflight_bytes)?,
        )?;
        let gate = root_gate(io.root(), true)?;
        match fs::symlink_metadata(io.root().join("cells")) {
            Ok(_) => return Err(FileCellIoError::InvalidConfiguration),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(io_error("inspect legacy world", e)),
        }
        Directory::child(io.root(), STORE, true)?;
        drop(gate);
        Ok(Self { io, limits })
    }
    /// Canonical isolated root.
    pub fn root(&self) -> &Path {
        self.io.root()
    }
    /// Workers or futures still owning an operation slot.
    pub fn outstanding_operations(&self) -> usize {
        self.io.outstanding_operations()
    }
    /// Reserved payload memory; excludes returned caller-owned snapshots.
    pub fn reserved_bytes(&self) -> usize {
        self.io.reserved_bytes()
    }
    /// Recovers any decided transaction, then reads one entire version. Readers
    /// use the same nonblocking OS gate as writers and never mix generations.
    pub fn load_snapshot(
        &self,
        timeout: Duration,
    ) -> Result<ActorIoFuture<WorldTransactionSnapshot>, FileCellIoError> {
        let deadline = FileActorIoProvider::deadline(timeout)?;
        let reserved = self.io.reserve(self.limits.snapshot_bytes)?;
        let this = self.clone();
        worker::start_retained(
            self.io.cell.shared.clone(),
            Some(deadline),
            Some(reserved),
            move |cancel| this.read(cancel),
        )
    }
    /// Performs every CAS before staging. A durable journal decision is the
    /// commit point; cancellation cannot undo a decision already in the fence.
    pub fn commit(
        &self,
        request: WorldTransactionRequest,
        timeout: Duration,
    ) -> Result<ActorIoFuture<WorldTransactionReceipt>, FileCellIoError> {
        let deadline = FileActorIoProvider::deadline(timeout)?;
        let mut bytes = 0usize;
        if request.changes.len() > self.limits.entries * 2 {
            return Err(FileCellIoError::Capacity);
        }
        for change in &request.changes {
            if let Some((_, payload)) = &change.value {
                if payload.len() > self.limits.file_bytes {
                    return Err(FileCellIoError::TooLarge {
                        limit: self.limits.file_bytes,
                        actual: payload.len() as u64,
                    });
                }
                bytes = bytes
                    .checked_add(payload.len())
                    .ok_or(FileCellIoError::Capacity)?;
            }
        }
        if bytes > self.limits.snapshot_bytes {
            return Err(FileCellIoError::Capacity);
        }
        let reserved = self.io.reserve(self.limits.snapshot_bytes + bytes)?;
        let this = self.clone();
        worker::start_retained(
            self.io.cell.shared.clone(),
            Some(deadline),
            Some(reserved),
            move |cancel| this.execute(request, cancel),
        )
    }
}

fn root_gate(root: &Path, exclusive: bool) -> Result<File, FileCellIoError> {
    let dir = Directory {
        path: root.to_owned(),
        handle: File::open(root).map_err(|e| io_error("open world gate directory", e))?,
    };
    dir.acquire_path(&root.join(GATE), exclusive)
}
impl Shared {
    pub(super) fn legacy_guard(&self) -> Result<File, FileCellIoError> {
        let guard = root_gate(&self.root, false)?;
        match fs::symlink_metadata(self.root.join(STORE)) {
            Ok(_) => Err(FileCellIoError::TransactionalRoot),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(guard),
            Err(e) => Err(io_error("inspect world isolation", e)),
        }
    }
}
