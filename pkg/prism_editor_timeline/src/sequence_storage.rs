//! Sequence project codec over the shared bounded atomic file worker.
use crate::storage::{
    DocumentLoadCandidate, DocumentSaveReceipt, DocumentStorageCodec, DocumentStorageRequest,
    DocumentStorageResult, DocumentStorageService, TimelineFileRevision, TimelineStorageError,
    TimelineStorageLimits,
};
use crate::{SequenceDocumentLimits, SequenceProjectDocument, SequenceProjectToken};
use alloc::vec::Vec;
use core::time::Duration;
use std::path::PathBuf;

/// File-worker codec for complete sequence projects, not individual timelines.
#[derive(Clone, Copy, Debug)]
pub struct SequenceProjectCodec {
    /// Authored data, topology and encoded project limits.
    pub limits: SequenceDocumentLimits,
}
impl DocumentStorageCodec for SequenceProjectCodec {
    type Document = SequenceProjectDocument;
    type Token = SequenceProjectToken;
    fn decode(
        &self,
        bytes: &[u8],
        max_bytes: usize,
    ) -> Result<Self::Document, TimelineStorageError> {
        SequenceProjectDocument::decode(
            bytes,
            SequenceDocumentLimits {
                max_bytes: self.limits.max_bytes.min(max_bytes),
                ..self.limits
            },
        )
        .map_err(TimelineStorageError::SequenceCodec)
    }
}

/// Actual isolated decoded project plus exact observed file revision.
pub type SequenceProjectLoadCandidate = DocumentLoadCandidate<SequenceProjectCodec>;
/// Host snapshot identity, exact published revision and durability outcome.
pub type SequenceProjectSaveReceipt = DocumentSaveReceipt<SequenceProjectCodec>;
/// Bounded retained output from the shared file worker.
pub type SequenceProjectStorageResult = DocumentStorageResult<SequenceProjectCodec>;
/// Poll/cancel handle sharing the same irrevocable publication fence.
pub type SequenceProjectStorageRequest = DocumentStorageRequest<SequenceProjectCodec>;

/// Thin project-specific admission adapter. No duplicate worker, file IO, sync,
/// lock, CAS, cancellation or rename implementation is hidden in this layer.
pub struct SequenceProjectStorageService {
    inner: DocumentStorageService<SequenceProjectCodec>,
}
impl SequenceProjectStorageService {
    /// Starts the existing serial worker with a complete-project codec. Constructor
    /// and requests do no caller file IO; the trusted existing parent is host-owned.
    pub fn new(
        path: impl Into<PathBuf>,
        max_requests: usize,
        limits: SequenceDocumentLimits,
    ) -> Result<Self, TimelineStorageError> {
        // Validate the complete policy before starting a worker; no filesystem IO.
        SequenceProjectDocument::default()
            .encode(limits)
            .map_err(TimelineStorageError::SequenceCodec)?;
        Ok(Self {
            inner: DocumentStorageService::with_codec(
                path,
                TimelineStorageLimits {
                    max_requests,
                    max_bytes: limits.max_bytes,
                    document: limits.sequence.document,
                },
                SequenceProjectCodec { limits },
            )?,
        })
    }
    /// Reads and decodes an isolated candidate. Host explicitly chooses adoption.
    pub fn load(
        &self,
        timeout: Duration,
    ) -> Result<SequenceProjectStorageRequest, TimelineStorageError> {
        self.inner.load(timeout)
    }
    /// Takes owned project bytes, exact host token and expected exact file bytes.
    /// Worker revalidates before publication; host compares returned token to live
    /// state before acknowledging save. No live editor is automatically marked clean.
    pub fn save(
        &self,
        bytes: Vec<u8>,
        token: SequenceProjectToken,
        expected: TimelineFileRevision,
        timeout: Duration,
    ) -> Result<SequenceProjectStorageRequest, TimelineStorageError> {
        self.inner.save(bytes, token, expected, timeout)
    }
}
