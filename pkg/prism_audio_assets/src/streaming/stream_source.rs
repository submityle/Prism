//! Random-access byte sources feeding the streaming decode pipeline.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! The "disk / byte source" origin of design section 20's streaming path. A
//! [`ByteSource`] is a seekable byte provider that the stream producer reads
//! encoded media from. The in-memory [`MemoryByteSource`] works in `no_std`
//! and backs deterministic tests; the `std`-gated [`FileByteSource`] reads from
//! a real file for shipping builds.

use alloc::vec;
use alloc::vec::Vec;

/// Errors a byte source can report.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ByteSourceError {
    /// The requested offset lies beyond the end of the source.
    OutOfRange,
    /// An underlying I/O failure occurred (only possible for `std` sources).
    Io,
}

/// A seekable, random-access source of encoded bytes.
pub trait ByteSource {
    /// Returns the total length in bytes when known.
    fn len(&self) -> Option<u64>;

    /// Returns `true` when the source is known to be empty.
    ///
    /// Sources of unknown length conservatively report `false`.
    fn is_empty(&self) -> bool {
        matches!(self.len(), Some(0))
    }

    /// Reads bytes starting at `offset` into `out`, returning the number of
    /// bytes actually read (may be shorter than `out` near the end).
    fn read_at(&mut self, offset: u64, out: &mut [u8]) -> Result<usize, ByteSourceError>;

    /// Reads the entire source into a freshly allocated buffer.
    ///
    /// Used by the bank layer to materialise a memory-resident asset. Sources
    /// of unknown length are read in bounded chunks until exhausted.
    fn read_to_vec(&mut self) -> Result<Vec<u8>, ByteSourceError> {
        let mut out = Vec::new();
        if let Some(len) = self.len() {
            out.resize(len as usize, 0);
            let mut filled = 0usize;
            while filled < out.len() {
                let read = self.read_at(filled as u64, &mut out[filled..])?;
                if read == 0 {
                    break;
                }
                filled += read;
            }
            out.truncate(filled);
            return Ok(out);
        }
        let mut offset = 0u64;
        let mut chunk = vec![0u8; 4096];
        loop {
            let read = self.read_at(offset, &mut chunk)?;
            if read == 0 {
                break;
            }
            out.extend_from_slice(&chunk[..read]);
            offset += read as u64;
        }
        Ok(out)
    }
}

/// A [`ByteSource`] backed by an owned in-memory byte buffer.
#[derive(Debug, Clone)]
pub struct MemoryByteSource {
    bytes: Vec<u8>,
}

impl MemoryByteSource {
    /// Wraps an owned byte buffer.
    #[must_use]
    pub fn new(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    /// Returns the underlying bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl ByteSource for MemoryByteSource {
    fn len(&self) -> Option<u64> {
        Some(self.bytes.len() as u64)
    }

    fn read_at(&mut self, offset: u64, out: &mut [u8]) -> Result<usize, ByteSourceError> {
        if offset > self.bytes.len() as u64 {
            return Err(ByteSourceError::OutOfRange);
        }
        let start = offset as usize;
        let available = self.bytes.len() - start;
        let count = available.min(out.len());
        out[..count].copy_from_slice(&self.bytes[start..start + count]);
        Ok(count)
    }
}

/// A [`ByteSource`] backed by a file on disk (requires the `std` feature).
#[cfg(feature = "std")]
#[derive(Debug)]
pub struct FileByteSource {
    file: std::fs::File,
    len: u64,
}

#[cfg(feature = "std")]
impl FileByteSource {
    /// Opens `path` for streaming reads.
    ///
    /// # Errors
    ///
    /// Returns [`ByteSourceError::Io`] when the file cannot be opened or its
    /// length cannot be queried.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self, ByteSourceError> {
        let file = std::fs::File::open(path).map_err(|_| ByteSourceError::Io)?;
        let len = file.metadata().map_err(|_| ByteSourceError::Io)?.len();
        Ok(Self { file, len })
    }
}

#[cfg(feature = "std")]
impl ByteSource for FileByteSource {
    fn len(&self) -> Option<u64> {
        Some(self.len)
    }

    fn read_at(&mut self, offset: u64, out: &mut [u8]) -> Result<usize, ByteSourceError> {
        use std::io::{Read, Seek, SeekFrom};
        if offset > self.len {
            return Err(ByteSourceError::OutOfRange);
        }
        self.file
            .seek(SeekFrom::Start(offset))
            .map_err(|_| ByteSourceError::Io)?;
        let max = (self.len - offset) as usize;
        let want = max.min(out.len());
        let mut filled = 0usize;
        while filled < want {
            let read = self
                .file
                .read(&mut out[filled..want])
                .map_err(|_| ByteSourceError::Io)?;
            if read == 0 {
                break;
            }
            filled += read;
        }
        Ok(filled)
    }
}
