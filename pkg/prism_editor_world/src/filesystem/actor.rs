//! Actor storage shares the cell envelope codec and durable replacement path.
use std::{
    future::Future,
    path::Path,
    pin::Pin,
    sync::Arc,
    time::{Duration, Instant},
};

use super::*;
use crate::ActorFileIdentity;

/// Actor storage result future. Dropping it fences further publication.
pub type ActorIoFuture<T> = Pin<Box<dyn Future<Output = Result<T, FileCellIoError>> + Send>>;

/// Per-file, request-count and aggregate in-flight payload bounds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ActorIoLimits {
    file: FileCellIoLimits,
    bytes: usize,
}
impl ActorIoLimits {
    /// All bounds must be nonzero; the aggregate must admit one maximum file.
    pub fn new(
        payload_bytes: usize,
        requests: usize,
        reserved_bytes: usize,
    ) -> Result<Self, FileCellIoError> {
        let file = FileCellIoLimits::new(payload_bytes, requests)?;
        if reserved_bytes < payload_bytes {
            return Err(FileCellIoError::InvalidConfiguration);
        }
        Ok(Self {
            file,
            bytes: reserved_bytes,
        })
    }
    /// Maximum opaque bytes in a file.
    pub const fn max_payload_bytes(self) -> usize {
        self.file.max_payload_bytes
    }
    /// Maximum outstanding workers/futures.
    pub const fn max_requests(self) -> usize {
        self.file.max_operations
    }
    /// Maximum reserved in-flight payload bytes.
    pub const fn max_reserved_bytes(self) -> usize {
        self.bytes
    }
}

/// Required durable base for compare-and-save.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActorRevisionBase {
    /// Create only: no actor file may already exist.
    Missing,
    /// Replace exactly this durable revision.
    Revision(u64),
}
impl ActorRevisionBase {
    fn revision(self) -> Option<u64> {
        match self {
            Self::Missing => None,
            Self::Revision(value) => Some(value),
        }
    }
}

/// Immutable actor write metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActorSaveRequest {
    identity: ActorFileIdentity,
    revision: u64,
    base: ActorRevisionBase,
}
impl ActorSaveRequest {
    /// Requires a revision no older than the observed base. Identical
    /// retries of a successful request remain accepted by the provider.
    pub fn new(
        identity: ActorFileIdentity,
        revision: u64,
        base: ActorRevisionBase,
    ) -> Result<Self, FileCellIoError> {
        if identity.actor().is_nil() || base.revision().is_some_and(|value| value > revision) {
            return Err(FileCellIoError::InvalidRequest);
        }
        Ok(Self {
            identity,
            revision,
            base,
        })
    }
    /// Stable actor file identity.
    pub fn identity(&self) -> &ActorFileIdentity {
        &self.identity
    }
    /// Incoming authoring revision.
    pub const fn revision(&self) -> u64 {
        self.revision
    }
    /// Required existing disk revision, or create-only.
    pub const fn base(&self) -> ActorRevisionBase {
        self.base
    }
}

/// Validated bytes read from an actor envelope.
#[derive(Debug)]
pub struct ActorSnapshot {
    identity: ActorFileIdentity,
    revision: u64,
    bytes: Vec<u8>,
    root: PathBuf,
}
impl ActorSnapshot {
    /// Canonical storage root from which this snapshot was loaded.
    pub fn root(&self) -> &Path {
        &self.root
    }
    /// Identity validated against the envelope and generated path.
    pub fn identity(&self) -> &ActorFileIdentity {
        &self.identity
    }
    /// Durable actor revision.
    pub const fn revision(&self) -> u64 {
        self.revision
    }
    /// Opaque application bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    /// Transfers ownership without copying.
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// Proof of a successful file sync, rename and directory sync. Constructed
/// only by the provider; retains the exact bytes, rather than a weak hash.
#[derive(Clone, Debug)]
pub struct ActorSaveReceipt {
    request: ActorSaveRequest,
    payload: Arc<[u8]>,
    root: PathBuf,
}
impl ActorSaveReceipt {
    /// Metadata of the durable snapshot.
    pub fn request(&self) -> &ActorSaveRequest {
        &self.request
    }
    /// Exact bytes durably written.
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }
    /// Canonical storage root of this receipt.
    pub fn root(&self) -> &Path {
        &self.root
    }
}

struct Admission {
    bytes: AtomicUsize,
    limit: usize,
}
pub(super) struct Reservation {
    admission: Arc<Admission>,
    bytes: usize,
}
impl Drop for Reservation {
    fn drop(&mut self) {
        self.admission.bytes.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

/// Bounded asynchronous OFPA storage with the cell provider's trusted-root
/// and local advisory-lock requirements. Each root represents one world.
/// No executor, entity schema or renderer is assumed.
#[derive(Clone)]
pub struct FileActorIoProvider {
    pub(super) cell: FileCellIoProvider,
    admission: Arc<Admission>,
}
impl FileActorIoProvider {
    /// Selects an existing private world directory. Clones share both bounds.
    pub fn new(root: impl AsRef<Path>, limits: ActorIoLimits) -> Result<Self, FileCellIoError> {
        Ok(Self {
            cell: FileCellIoProvider::new(root, limits.file)?,
            admission: Arc::new(Admission {
                bytes: AtomicUsize::new(0),
                limit: limits.bytes,
            }),
        })
    }
    /// Canonical world root.
    pub fn root(&self) -> &Path {
        self.cell.root()
    }
    /// Outstanding workers/futures, including cancelled workers still exiting.
    pub fn outstanding_operations(&self) -> usize {
        self.cell.outstanding_operations()
    }
    /// Bytes reserved by pending requests, including cancelled workers.
    pub fn reserved_bytes(&self) -> usize {
        self.admission.bytes.load(Ordering::Acquire)
    }
    pub(super) fn reserve(&self, bytes: usize) -> Result<Arc<Reservation>, FileCellIoError> {
        self.admission
            .bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current
                    .checked_add(bytes)
                    .filter(|total| *total <= self.admission.limit)
            })
            .map_err(|_| FileCellIoError::Capacity)?;
        Ok(Arc::new(Reservation {
            admission: self.admission.clone(),
            bytes,
        }))
    }
    pub(super) fn deadline(timeout: Duration) -> Result<Instant, FileCellIoError> {
        if timeout.is_zero() {
            return Err(FileCellIoError::InvalidRequest);
        }
        Instant::now()
            .checked_add(timeout)
            .ok_or(FileCellIoError::InvalidRequest)
    }
    /// Loads a validated envelope. The requested cap is reserved before worker
    /// admission; oversize files are rejected before allocating their payload.
    pub fn load_actor(
        &self,
        identity: ActorFileIdentity,
        max_bytes: usize,
        timeout: Duration,
    ) -> Result<ActorIoFuture<ActorSnapshot>, FileCellIoError> {
        let deadline = Self::deadline(timeout)?;
        if identity.actor().is_nil()
            || max_bytes == 0
            || max_bytes > self.cell.limits().max_payload_bytes
        {
            return Err(FileCellIoError::InvalidRequest);
        }
        let reservation = self.reserve(max_bytes)?;
        let shared = self.cell.shared.clone();
        let future = worker::start_retained(
            shared.clone(),
            Some(deadline),
            Some(reservation.clone()),
            move |cancel| {
                let _isolation = shared.legacy_guard()?;
                cancel.check()?;
                let directory = actor_directory(&shared, &identity, false)?;
                let path = directory.path.join(identity.actor().to_u128().to_string());
                let _guard = directory.acquire_path(&path.with_extension("lock"), false)?;
                let read = read_envelope(
                    &mut open_regular(&path)?,
                    &envelope(&identity),
                    max_bytes,
                    None,
                    cancel,
                )?;
                Ok(ActorSnapshot {
                    identity,
                    revision: read.revision,
                    bytes: read.bytes,
                    root: shared.root.clone(),
                })
            },
        )?;
        Ok(Box::pin(async move {
            let _reserved = reservation;
            future.await
        }))
    }
    /// Saves with an explicit revision base. Deadlines are cooperative before
    /// the commit fence; a rename already in progress finishes with its sync.
    /// Drop fences publication and cleanup may outlive the dropped future.
    pub fn save_actor(
        &self,
        request: ActorSaveRequest,
        payload: Arc<[u8]>,
        timeout: Duration,
    ) -> Result<ActorIoFuture<ActorSaveReceipt>, FileCellIoError> {
        let deadline = Self::deadline(timeout)?;
        if payload.len() > self.cell.limits().max_payload_bytes {
            return Err(FileCellIoError::TooLarge {
                limit: self.cell.limits().max_payload_bytes,
                actual: payload.len() as u64,
            });
        }
        let reservation = self.reserve(payload.len())?;
        let shared = self.cell.shared.clone();
        let future = worker::start_retained(
            shared.clone(),
            Some(deadline),
            Some(reservation.clone()),
            move |cancel| {
                let _isolation = shared.legacy_guard()?;
                cancel.check()?;
                let directory = actor_directory(&shared, &request.identity, true)?;
                let path = directory
                    .path
                    .join(request.identity.actor().to_u128().to_string());
                let _guard = directory.acquire_path(&path.with_extension("lock"), true)?;
                shared.replace(
                    &directory,
                    &path,
                    path.with_extension("pending"),
                    SaveEnvelope {
                        identity: &envelope(&request.identity),
                        revision: request.revision,
                        base: Some(request.base.revision()),
                        payload: &payload,
                    },
                    cancel,
                )?;
                Ok(ActorSaveReceipt {
                    request,
                    payload,
                    root: shared.root.clone(),
                })
            },
        )?;
        Ok(Box::pin(async move {
            let _reserved = reservation;
            future.await
        }))
    }
}
fn envelope(identity: &ActorFileIdentity) -> EnvelopeIdentity {
    EnvelopeIdentity {
        magic: *b"PEWACTR\0",
        coord: identity.cell(),
        guid: identity.actor(),
    }
}
fn actor_directory(
    shared: &Shared,
    identity: &ActorFileIdentity,
    create: bool,
) -> Result<Directory, FileCellIoError> {
    let cells = shared.directory(create)?;
    let c = identity.cell();
    let cell = Directory::child(&cells.path, &format!("{}_{}_{}", c.x, c.y, c.z), create)?;
    Directory::child(&cell.path, "actors", create)
}

#[cfg(test)]
mod tests;
