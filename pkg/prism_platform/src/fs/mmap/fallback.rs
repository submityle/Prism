//! Fallback memory-mapping backend for targets without `mmap`
//! (for example wasm).
//!
//! There is no OS mapping primitive here, so this is an honest *fallback by
//! read* (design doc §6: "平台无 mmap（Web）时回退普通读"): [`map`] reads the
//! requested range into an owned heap buffer and presents the same `[u8]`
//! surface as a real mapping. [`super::mmap_supported`] reports `false` so
//! callers that need true zero-copy can branch. A writable mapping writes the
//! buffer back to the file on [`Mapping::flush`] and on drop.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};

use super::{MmapError, Result};

/// This build does NOT map files zero-copy; it falls back to a read buffer.
pub(super) const SUPPORTED: bool = false;

/// An owned read buffer standing in for a file mapping.
pub(super) struct Mapping {
    buf: Vec<u8>,
    file: File,
    offset: u64,
    writable: bool,
}

/// "Map" `[offset, offset + len)` of `file` by reading it into a buffer.
pub(super) fn map(mut file: File, offset: u64, len: usize, writable: bool) -> Result<Mapping> {
    file.seek(SeekFrom::Start(offset))?;
    let mut buf = vec![0u8; len];
    file.read_exact(&mut buf)?;
    Ok(Mapping {
        buf,
        file,
        offset,
        writable,
    })
}

impl Mapping {
    pub(super) fn as_ptr(&self) -> *const u8 {
        self.buf.as_ptr()
    }

    pub(super) fn as_mut_ptr(&mut self) -> *mut u8 {
        self.buf.as_mut_ptr()
    }

    pub(super) fn len(&self) -> usize {
        self.buf.len()
    }

    pub(super) fn flush(&self) -> Result<()> {
        if !self.writable {
            return Ok(());
        }
        // The buffer may have been mutated through the exposed slice; write it
        // back to the originating range of the file.
        let mut file = &self.file;
        file.seek(SeekFrom::Start(self.offset))?;
        file.write_all(&self.buf)?;
        file.flush()?;
        Ok(())
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        if self.writable {
            // Best-effort write-back on drop; errors cannot be surfaced here, so
            // callers that must observe failures should call `flush` explicitly.
            let _ = (&self.file).seek(SeekFrom::Start(self.offset));
            let _ = (&self.file).write_all(&self.buf);
        }
    }
}
