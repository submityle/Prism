//! Exact local double-run checks. No CAS/DDC effects occur in this module.

use super::BuildExecutionError;
use crate::graph::BuildError;
use crate::job::{JobId, Processor};
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

pub(super) fn verify_second_run(
    job: JobId,
    key: ArtifactKey,
    processor: &dyn Processor,
    inputs: &[&[u8]],
    first: &[u8],
    max_output_bytes: u64,
) -> Result<(), BuildExecutionError> {
    let second = processor.process(inputs).map_err(|source| {
        BuildExecutionError::Graph(BuildError::ProcessorFailed { job, source })
    })?;
    super::output_limits::check_len(job, second.len() as u64, max_output_bytes)?;
    if first == second {
        return Ok(());
    }
    let first_difference = first
        .iter()
        .zip(&second)
        .position(|(a, b)| a != b)
        .unwrap_or(first.len().min(second.len())) as u64;
    Err(BuildExecutionError::NonDeterministic(
        DeterminismDifference {
            job,
            key,
            first_hash: ContentHash::digest(first),
            second_hash: ContentHash::digest(&second),
            first_len: first.len() as u64,
            second_len: second.len() as u64,
            first_difference,
        },
    ))
}
