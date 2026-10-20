//! Per-hit admission and bounded reads, including DDC entries refreshed mid-build.

use super::BuildExecutionError;
use crate::cas::CasError;
use crate::{ArtifactDescriptor, ContentAddressedStore, ContentHash, JobId};
use std::fs::{self, File};
use std::io::{self, Read};

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

/// None denotes a missing/corrupt cache entry that may be rebuilt. Limits are
/// failures, not cache misses: silently rebuilding would hide a broken contract.
pub(super) fn read_cached(
    job: JobId,
    cas: &ContentAddressedStore,
    descriptor: &ArtifactDescriptor,
    max_output_bytes: u64,
) -> Result<Option<Vec<u8>>, BuildExecutionError> {
    check_len(job, descriptor.byte_len, max_output_bytes)?;
    let path = cas.blob_path(descriptor.content_hash);
    let metadata = match fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(CasError::Io(error).into()),
    };
    if !metadata.is_file() {
        return Err(CasError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            "cached CAS blob is not a regular file",
        ))
        .into());
    }
    let file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(CasError::Io(error).into()),
    };
    let metadata = file.metadata().map_err(CasError::Io)?;
    if !metadata.is_file() {
        return Err(CasError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            "cached CAS blob is not a regular file",
        ))
        .into());
    }
    // Check the opened file, not only the untrusted descriptor. A short
    // descriptor must never authorize allocating a much larger CAS payload.
    check_len(job, metadata.len(), max_output_bytes)?;
    if metadata.len() != descriptor.byte_len {
        return Ok(None);
    }
    // The extra byte detects growth after metadata inspection while bounding
    // the read independently of concurrent file changes. No full payload copy.
    let mut bytes = Vec::new();
    file.take(descriptor.byte_len.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(CasError::Io)?;
    check_len(job, bytes.len() as u64, max_output_bytes)?;
    if bytes.len() as u64 != descriptor.byte_len
        || ContentHash::digest(&bytes) != descriptor.content_hash
    {
        return Ok(None);
    }
    Ok(Some(bytes))
}
