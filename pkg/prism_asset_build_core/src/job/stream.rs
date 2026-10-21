//! Declared, verified input streams for artifact processors.

use super::ProcessorError;
use crate::cas::VerifiedCasReader;
use crate::ContentHash;
use std::collections::BTreeMap;
use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};
use std::sync::Arc;

/// Default admission limit for the contiguous-input processor adapter.
pub const DEFAULT_MATERIALIZED_INPUT_BYTES: u64 = 512 * 1024 * 1024;

trait InputReader: Read + Seek + Send {}
impl<T: Read + Seek + Send> InputReader for T {}

/// One declared dependency, in the graph's input-slot order. The executor
/// verifies its complete immutable content before exposing the reader.
/// Seeking is supported; it never authorizes access to another input.
pub struct ProcessorInput<'a> {
    content_hash: ContentHash,
    byte_len: u64,
    reader: Box<dyn InputReader + 'a>,
}

impl<'a> ProcessorInput<'a> {
    pub fn content_hash(&self) -> ContentHash {
        self.content_hash
    }

    pub fn byte_len(&self) -> u64 {
        self.byte_len
    }

    pub(crate) fn verified(reader: VerifiedCasReader) -> Self {
        Self {
            content_hash: reader.content_hash(),
            byte_len: reader.byte_len(),
            reader: Box::new(reader),
        }
    }

    pub(crate) fn borrowed(bytes: &'a [u8]) -> Self {
        Self {
            content_hash: ContentHash::digest(bytes),
            byte_len: bytes.len() as u64,
            reader: Box::new(Cursor::new(bytes)),
        }
    }
}

impl Read for ProcessorInput<'_> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.reader.read(bytes)
    }
}

impl Seek for ProcessorInput<'_> {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        // Enforce the same logical bounds for both borrowed and CAS inputs.
        let current = self.reader.stream_position()?;
        let target = match from {
            SeekFrom::Start(offset) => i128::from(offset),
            SeekFrom::Current(offset) => i128::from(current) + i128::from(offset),
            SeekFrom::End(offset) => i128::from(self.byte_len) + i128::from(offset),
        };
        if target < 0 || target > i128::from(self.byte_len) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "processor input seek is out of bounds",
            ));
        }
        self.reader.seek(SeekFrom::Start(target as u64))
    }
}

/// A native streaming transformation. The executor supplies only declared
/// dependencies and a private, bounded output writer. Failed writes poison the
/// writer even if the processor ignores the error. Errors and panics discard
/// the unpublished output; verified builds compare both private outputs before
/// either becomes visible in CAS/DDC. This is a trusted in-process interface,
/// not an operating-system sandbox for ambient I/O or processor allocations.
pub trait StreamingProcessor: Send + Sync {
    fn execute(
        &self,
        inputs: &mut [ProcessorInput<'_>],
        output: &mut dyn Write,
    ) -> Result<(), ProcessorError>;
}

pub(crate) fn materialize_inputs(
    inputs: &mut [ProcessorInput<'_>],
    max_bytes: u64,
) -> Result<Vec<Arc<[u8]>>, ProcessorError> {
    admit_materialized_inputs(
        inputs
            .iter()
            .map(|input| (input.content_hash, input.byte_len)),
        max_bytes,
    )?;
    let mut payloads = BTreeMap::<ContentHash, Arc<[u8]>>::new();
    let mut slots = Vec::with_capacity(inputs.len());
    for input in inputs {
        let payload = match payloads.get(&input.content_hash) {
            Some(bytes) => Arc::clone(bytes),
            None => {
                let length = usize::try_from(input.byte_len)
                    .map_err(|_| ProcessorError::new("input length does not fit address space"))?;
                let mut bytes = Vec::new();
                bytes
                    .try_reserve_exact(length)
                    .map_err(|_| ProcessorError::new("input allocation failed"))?;
                bytes.resize(length, 0);
                input
                    .rewind()
                    .map_err(|error| ProcessorError::new(error.to_string()))?;
                input
                    .read_exact(&mut bytes)
                    .map_err(|error| ProcessorError::new(error.to_string()))?;
                let payload: Arc<[u8]> = bytes.into();
                payloads.insert(input.content_hash, Arc::clone(&payload));
                payload
            }
        };
        slots.push(payload);
    }
    Ok(slots)
}

/// Inspect descriptor metadata before opening or verifying any input blob.
pub(crate) fn admit_materialized_inputs(
    inputs: impl IntoIterator<Item = (ContentHash, u64)>,
    max_bytes: u64,
) -> Result<(), ProcessorError> {
    let mut unique = BTreeMap::new();
    let mut total = 0u64;
    for (hash, byte_len) in inputs {
        if let Some(length) = unique.insert(hash, byte_len) {
            if length != byte_len {
                return Err(ProcessorError::new("input hash has contradictory lengths"));
            }
        } else {
            total = total
                .checked_add(byte_len)
                .ok_or_else(|| ProcessorError::new("materialized input length overflow"))?;
        }
        if total > max_bytes {
            return Err(ProcessorError::new(
                "processor inputs exceed materialization byte bound",
            ));
        }
    }
    Ok(())
}

/// Used only by the graph API whose requested result is an in-memory map.
pub(crate) struct BufferOutput {
    bytes: Vec<u8>,
    max_bytes: u64,
    failed: bool,
}

impl BufferOutput {
    pub fn new(max_bytes: u64) -> Self {
        Self {
            bytes: Vec::new(),
            max_bytes,
            failed: false,
        }
    }

    pub fn finish(self) -> Result<Vec<u8>, ProcessorError> {
        if self.failed {
            Err(ProcessorError::new("processor output stream failed"))
        } else {
            Ok(self.bytes)
        }
    }
}

impl Write for BufferOutput {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.failed {
            return Err(io::Error::other("processor output stream failed"));
        }
        if (self.bytes.len() as u64)
            .checked_add(bytes.len() as u64)
            .is_none_or(|size| size > self.max_bytes)
        {
            self.failed = true;
            return Err(io::Error::other(
                "processor output exceeds declared byte bound",
            ));
        }
        if self.bytes.try_reserve(bytes.len()).is_err() {
            self.failed = true;
            return Err(io::Error::other("processor output allocation failed"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.failed {
            Err(io::Error::other("processor output stream failed"))
        } else {
            Ok(())
        }
    }
}
