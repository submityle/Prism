use std::{
    fs::File,
    io::{Read, Write},
};

use super::{io_error, Cancellation, FileCellIoError};
use crate::{CellFileIdentity, CellSaveRequest};

pub(super) const HEADER_BYTES: usize = 64;
const MAGIC: &[u8; 8] = b"PEWCELL\0";
const VERSION: u32 = 1;
const CHUNK_BYTES: usize = 64 * 1024;
const FNV_OFFSET: u64 = 0xcbf29ce484222325;

fn fold(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash = (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3);
    }
    hash
}

pub(super) struct ReadCell {
    pub bytes: Vec<u8>,
    pub revision: u64,
    pub equal: bool,
}

pub(super) fn read_cell(
    file: &mut File,
    identity: &CellFileIdentity,
    limit: usize,
    compare: Option<&[u8]>,
    cancel: &Cancellation,
) -> Result<ReadCell, FileCellIoError> {
    let size = file
        .metadata()
        .map_err(|e| io_error("cell length", e))?
        .len();
    if size > (HEADER_BYTES + limit) as u64 {
        return Err(FileCellIoError::TooLarge {
            limit,
            actual: size.saturating_sub(HEADER_BYTES as u64),
        });
    }
    if size < HEADER_BYTES as u64 {
        return Err(FileCellIoError::Corrupt("truncated header"));
    }
    let mut header = [0; HEADER_BYTES];
    file.read_exact(&mut header)
        .map_err(|e| io_error("read cell header", e))?;
    if &header[..8] != MAGIC {
        return Err(FileCellIoError::Corrupt("magic"));
    }
    if header[8..12] != VERSION.to_le_bytes() {
        return Err(FileCellIoError::Corrupt("unsupported version"));
    }
    let coord = identity.coord();
    for (index, axis) in [coord.x, coord.y, coord.z].into_iter().enumerate() {
        if header[12 + index * 4..16 + index * 4] != axis.to_le_bytes() {
            return Err(FileCellIoError::Corrupt("coordinate mismatch"));
        }
    }
    if header[24..40] != identity.stable_id().to_u128().to_le_bytes() {
        return Err(FileCellIoError::Corrupt("stable identity mismatch"));
    }
    let revision = u64::from_le_bytes(header[40..48].try_into().expect("fixed header range"));
    let length = u64::from_le_bytes(header[48..56].try_into().expect("fixed header range"));
    if length > limit as u64 {
        return Err(FileCellIoError::TooLarge {
            limit,
            actual: length,
        });
    }
    if length.checked_add(HEADER_BYTES as u64) != Some(size) {
        return Err(FileCellIoError::Corrupt("file length mismatch"));
    }
    let expected = u64::from_le_bytes(header[56..64].try_into().expect("fixed header range"));
    let mut checksum = fold(FNV_OFFSET, &header[..56]);
    let length = length as usize; // length <= the validated usize limit
    let mut bytes = Vec::new();
    if compare.is_none() {
        bytes
            .try_reserve_exact(length)
            .map_err(|e| io_error("allocate cell payload", std::io::Error::other(e)))?;
    }
    let mut equal = compare.is_some_and(|payload| payload.len() == length);
    let mut buffer = [0; CHUNK_BYTES];
    let mut at = 0;
    while at < length {
        cancel.check()?;
        let end = (length - at).min(CHUNK_BYTES);
        let chunk = &mut buffer[..end];
        file.read_exact(chunk)
            .map_err(|e| io_error("read cell payload", e))?;
        checksum = fold(checksum, chunk);
        if let Some(payload) = compare {
            equal &= payload.get(at..at + end) == Some(&chunk[..]);
        } else {
            bytes.extend_from_slice(chunk);
        }
        at += end;
    }
    let mut extra = [0];
    if file
        .read(&mut extra)
        .map_err(|e| io_error("check cell boundary", e))?
        != 0
    {
        return Err(FileCellIoError::Corrupt("trailing bytes"));
    }
    if checksum != expected {
        return Err(FileCellIoError::Corrupt("checksum mismatch"));
    }
    cancel.check()?;
    Ok(ReadCell {
        bytes,
        revision,
        equal,
    })
}

pub(super) fn write_cell(
    file: &mut File,
    request: &CellSaveRequest,
    payload: &[u8],
    cancel: &Cancellation,
) -> Result<(), FileCellIoError> {
    let mut header = [0; HEADER_BYTES];
    header[..8].copy_from_slice(MAGIC);
    header[8..12].copy_from_slice(&VERSION.to_le_bytes());
    let coord = request.cell();
    for (index, axis) in [coord.x, coord.y, coord.z].into_iter().enumerate() {
        header[12 + index * 4..16 + index * 4].copy_from_slice(&axis.to_le_bytes());
    }
    header[24..40].copy_from_slice(&request.file().stable_id().to_u128().to_le_bytes());
    header[40..48].copy_from_slice(&request.revision().to_le_bytes());
    header[48..56].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    let mut checksum = fold(FNV_OFFSET, &header[..56]);
    for chunk in payload.chunks(CHUNK_BYTES) {
        cancel.check()?;
        checksum = fold(checksum, chunk);
    }
    header[56..64].copy_from_slice(&checksum.to_le_bytes());
    file.write_all(&header)
        .map_err(|e| io_error("write cell header", e))?;
    for chunk in payload.chunks(CHUNK_BYTES) {
        cancel.check()?;
        file.write_all(chunk)
            .map_err(|e| io_error("write cell payload", e))?;
    }
    Ok(())
}
