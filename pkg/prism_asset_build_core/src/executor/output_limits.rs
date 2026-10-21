//! Per-hit admission and bounded reads, including DDC entries refreshed mid-build.

use super::BuildExecutionError;
use crate::cas::CasError;
use crate::{ArtifactDescriptor, ContentAddressedStore, JobId};
use std::io;

pub(super) fn check_len(
    job: JobId,
    actual: u64,
    max_output_bytes: u64,
) -> Result<(), BuildExecutionError> {
    if actual > max_output_bytes {
        return Err(BuildExecutionError::InvalidOutput(
            job,
            "output exceeds declared byte bound",
        ));
    }
    Ok(())
}

/// False denotes a missing/corrupt cache entry that may be rebuilt. Limits are
/// failures, not cache misses: silently rebuilding would hide a broken contract.
pub(super) fn verify_cached(
    job: JobId,
    cas: &ContentAddressedStore,
    descriptor: &ArtifactDescriptor,
    max_output_bytes: u64,
) -> Result<bool, BuildExecutionError> {
    check_len(job, descriptor.byte_len, max_output_bytes)?;
    match cas.open_verified(descriptor.content_hash, max_output_bytes) {
        Ok(reader) => Ok(reader.byte_len() == descriptor.byte_len),
        Err(CasError::Corrupt { .. }) => Ok(false),
        Err(CasError::Io(error))
            if matches!(
                error.kind(),
                io::ErrorKind::NotFound | io::ErrorKind::UnexpectedEof
            ) =>
        {
            Ok(false)
        }
        Err(CasError::Io(error)) if error.kind() == io::ErrorKind::InvalidInput => Err(
            BuildExecutionError::InvalidOutput(job, "output exceeds declared byte bound"),
        ),
        Err(error) => Err(error.into()),
    }
}
