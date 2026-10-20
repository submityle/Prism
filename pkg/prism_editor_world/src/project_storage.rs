//! Real project files using the existing typed document storage worker.
use super::*;
use prism_editor_timeline::{
    DocumentLoadCandidate, DocumentSaveReceipt, DocumentStorageCodec, DocumentStorageRequest,
    DocumentStorageResult, DocumentStorageService, TimelineFileRevision, TimelineStorageError,
    TimelineStorageLimits,
};
use std::{path::PathBuf, sync::Arc, time::Duration};

/// Exact host snapshot fence. Cloning a project preserves this fence until
/// either clone is edited. Independently decoded identical bytes have a new
/// fence, so an old publication receipt cannot clean an adopted replacement.
#[derive(Clone, Debug)]
pub struct WorldProjectToken {
    identity: StableGuid,
    revision: u64,
    stamp: Arc<()>,
}
impl WorldProjectToken {
    /// Stable identity of the encoded world.
    pub const fn identity(&self) -> StableGuid {
        self.identity
    }
    /// Membership revision encoded with this token.
    pub const fn revision(&self) -> u64 {
        self.revision
    }
}
impl WorldProject {
    /// Captures the current exact host instance/revision without serializing.
    pub fn token(&self) -> WorldProjectToken {
        WorldProjectToken {
            identity: self.identity,
            revision: self.revision,
            stamp: self.stamp.clone(),
        }
    }
    /// Whether a successful save receipt still describes this exact host state.
    pub fn matches_token(&self, token: &WorldProjectToken) -> bool {
        self.identity == token.identity
            && self.revision == token.revision
            && Arc::ptr_eq(&self.stamp, &token.stamp)
    }
}

/// World codec supplied to the shared serial worker for both loads and saves.
#[derive(Clone, Copy, Debug)]
pub struct WorldProjectCodec {
    /// Expected non-nil world ID, checked after decoding.
    pub identity: StableGuid,
    /// Complete membership/encoded-byte admission policy.
    pub limits: ProjectLimits,
}
impl DocumentStorageCodec for WorldProjectCodec {
    type Document = WorldProject;
    type Token = WorldProjectToken;
    fn decode(&self, bytes: &[u8], max_bytes: usize) -> Result<WorldProject, TimelineStorageError> {
        let project = WorldProject::decode(
            bytes,
            ProjectLimits {
                max_bytes: self.limits.max_bytes.min(max_bytes),
                ..self.limits
            },
        )
        .map_err(codec_error)?;
        if project.identity != self.identity {
            return Err(codec_error(ProjectError::Invalid(
                "world identity mismatch",
            )));
        }
        Ok(project)
    }
}
fn codec_error(error: ProjectError) -> TimelineStorageError {
    TimelineStorageError::DocumentCodec {
        domain: "world-project",
        message: error.to_string(),
    }
}

/// Independently decoded real project and exact observed file bytes.
pub type WorldProjectLoadCandidate = DocumentLoadCandidate<WorldProjectCodec>;
/// Successful real publication, snapshot fence and exact published file bytes.
pub type WorldProjectSaveReceipt = DocumentSaveReceipt<WorldProjectCodec>;
/// Retained typed load/save output.
pub type WorldProjectStorageResult = DocumentStorageResult<WorldProjectCodec>;
/// Poll/cancel handle using the existing publication and cancellation fence.
pub type WorldProjectStorageRequest = DocumentStorageRequest<WorldProjectCodec>;

/// Admission or shared-worker failure. Live host state is never mutated.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WorldProjectStorageError {
    /// Actual membership codec rejected the project before submission.
    Project(ProjectError),
    /// Existing file worker rejected or failed an operation.
    Storage(TimelineStorageError),
}
impl fmt::Display for WorldProjectStorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Project(e) => e.fmt(f),
            Self::Storage(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for WorldProjectStorageError {}
impl From<ProjectError> for WorldProjectStorageError {
    fn from(e: ProjectError) -> Self {
        Self::Project(e)
    }
}
impl From<TimelineStorageError> for WorldProjectStorageError {
    fn from(e: TimelineStorageError) -> Self {
        Self::Storage(e)
    }
}

/// Thin project adapter. It creates no second IO implementation, transaction
/// protocol or worker loop. OFPA payload stores continue to own payload bytes.
pub struct WorldProjectStorageService {
    inner: DocumentStorageService<WorldProjectCodec>,
    codec: WorldProjectCodec,
}
impl WorldProjectStorageService {
    /// Starts the shared typed worker for a real absolute project filename in a
    /// trusted existing parent. Constructor performs no caller filesystem IO.
    pub fn new(
        path: impl Into<PathBuf>,
        identity: StableGuid,
        max_requests: usize,
        limits: ProjectLimits,
    ) -> Result<Self, WorldProjectStorageError> {
        limits.validate()?;
        WorldProject::new(identity)?;
        let codec = WorldProjectCodec { identity, limits };
        Ok(Self {
            inner: DocumentStorageService::with_codec(
                path,
                TimelineStorageLimits {
                    max_requests,
                    max_bytes: limits.max_bytes,
                    ..TimelineStorageLimits::default()
                },
                codec,
            )?,
            codec,
        })
    }
    /// Queues a bounded real file read and full codec validation. The caller
    /// explicitly adopts the candidate; this never replaces live host state.
    pub fn load(
        &self,
        timeout: Duration,
    ) -> Result<WorldProjectStorageRequest, TimelineStorageError> {
        self.inner.load(timeout)
    }
    /// Encodes actual project membership, captures its exact host token and
    /// queues a conditional file publication. The shared worker revalidates
    /// these bytes. No automatic dirty/clean or world installation side effects.
    pub fn save(
        &self,
        project: &WorldProject,
        expected: TimelineFileRevision,
        timeout: Duration,
    ) -> Result<WorldProjectStorageRequest, WorldProjectStorageError> {
        if project.identity != self.codec.identity {
            return Err(ProjectError::Invalid("world identity mismatch").into());
        }
        Ok(self.inner.save(
            project.encode(self.codec.limits)?,
            project.token(),
            expected,
            timeout,
        )?)
    }
}
