//! Exact local double-run checks. No CAS/DDC effects occur in this module.

use super::BuildExecutionError;
use crate::cas::CasWriter;
use crate::job::JobId;
use crate::types::{ArtifactKey, ContentHash};
use std::fmt;

/// Diagnostic evidence for two unequal fresh executions of one artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeterminismDifference {
    /// Graph job whose two outputs differed.
    pub job: JobId,
    /// Semantic artifact key shared by the two executions.
    pub key: ArtifactKey,
    /// Content digest of the first execution.
    pub first_hash: ContentHash,
    /// Content digest of the second execution.
    pub second_hash: ContentHash,
    /// Byte length of the first output.
    pub first_len: u64,
    /// Byte length of the second output.
    pub second_len: u64,
    /// First differing byte offset; the shorter length for equal prefixes.
    pub first_difference: u64,
}

impl fmt::Display for DeterminismDifference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "job {} is non-deterministic: key={} first={} ({} bytes), second={} ({} bytes), first_difference={}",
            self.job.0,
            self.key.to_hex(),
            self.first_hash.to_hex(),
            self.first_len,
            self.second_hash.to_hex(),
            self.second_len,
            self.first_difference,
        )
    }
}

pub(super) fn verify_streams(
    job: JobId,
    key: ArtifactKey,
    first: &mut CasWriter,
    second: &mut CasWriter,
) -> Result<(), BuildExecutionError> {
    let Some(first_difference) = first.first_difference(second)? else {
        return Ok(());
    };
    let first = first.pending_result()?;
    let second = second.pending_result()?;
    Err(BuildExecutionError::NonDeterministic(
        DeterminismDifference {
            job,
            key,
            first_hash: first.content_hash,
            second_hash: second.content_hash,
            first_len: first.byte_len,
            second_len: second.byte_len,
            first_difference,
        },
    ))
}
