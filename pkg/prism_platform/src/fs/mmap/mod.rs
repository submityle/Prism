//! Memory-mapped files: zero-copy mapping of large on-disk assets.
//!
//! This is the M4 `mmap` layer from the design doc (§6 内存映射 mmap, §22). It
//! gives `prism_asset`'s streaming path a way to map a large asset directly
//! into the address space so reads skip the extra user-space copy a normal
//! `read` would pay. Two views are offered:
//!
//! - [`Mmap`]: a read-only mapping, [`Deref`](core::ops::Deref)ing to `[u8]`.
//! - [`MmapMut`]: a read-write mapping, additionally
//!   [`DerefMut`](core::ops::DerefMut)ing to `[u8]`, with [`MmapMut::flush`]
//!   to push dirty pages back to the file.
//!
//! The whole module requires the `std` feature (it opens real OS files); it is
//! absent from `no_std` builds like the rest of [`crate::fs`].
//!
//! ## Backends
//! - **Unix (Linux / macOS / BSD)**: `mmap`/`munmap`/`msync` over the file
//!   descriptor, `MAP_SHARED` so writes reach the file. Real and exercised by
//!   this crate's test-suite on the host it runs on.
//! - **Windows**: `CreateFileMapping`/`MapViewOfFile`/`UnmapViewOfFile`/
//!   `FlushViewOfFile`. Compiled behind `cfg(windows)`; **written but not yet
//!   validated on a Windows host** — treat as provisional.
//! - **Other targets without mmap (e.g. wasm)**: an honest *fallback-by-read*
//!   backend. [`mmap_supported`] returns `false`, but [`Mmap::map`] still
//!   works: it reads the requested range into an owned heap buffer and presents
//!   the identical `[u8]` surface (design doc §6: "平台无 mmap（Web）时回退普通
//!   读"). A [`MmapMut`] fallback writes the buffer back to the file on
//!   [`MmapMut::flush`] / drop. This is **not** zero-copy; branch on
//!   [`mmap_supported`] when zero-copy actually matters.
//!
//! ## Lifetime / safety contract
//! A mapping owns its OS region for its whole lifetime and unmaps it exactly
//! once on [`Drop`]. The borrowed slice from `Deref` is valid only while the
//! mapping is alive. Mapping a file the OS or another process concurrently
//! truncates can make the tail of the region fault on access; the engine's VFS
//! layer owns that higher-level coordination (design doc §6 VFS 分层点).

use core::fmt;
use core::ops::{Deref, DerefMut};
use core::slice;
use std::fs::File;
use std::path::Path;

use crate::fs::file::OpenOptions;

#[cfg(unix)]
#[path = "unix.rs"]
mod backend;

#[cfg(windows)]
#[path = "windows.rs"]
mod backend;

#[cfg(not(any(unix, windows)))]
#[path = "fallback.rs"]
mod backend;

/// Result type for every memory-mapping operation.
pub type Result<T> = core::result::Result<T, MmapError>;

/// Why a memory-mapping operation did not succeed.
#[derive(Debug)]
pub enum MmapError {
    /// A zero-length mapping was requested (an empty file, or `len == 0`).
    /// `mmap`/`MapViewOfFile` reject empty regions, so this is surfaced
    /// explicitly instead of returning a dangling empty mapping.
    Empty,
    /// An argument was invalid: a range extending past the end of the file, or
    /// an offset/length that overflows.
    InvalidArgument,
    /// An underlying I/O error while opening or querying the file.
    Io(std::io::Error),
    /// The OS rejected the mapping call itself; carries the platform error code
    /// (`errno` on Unix, `GetLastError` on Windows).
    System(i32),
}

impl fmt::Display for MmapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MmapError::Empty => write!(f, "cannot memory-map a zero-length range"),
            MmapError::InvalidArgument => write!(f, "invalid memory-map range"),
            MmapError::Io(err) => write!(f, "memory-map I/O error: {err}"),
            MmapError::System(code) => write!(f, "OS rejected memory-map request (code {code})"),
        }
    }
}

impl std::error::Error for MmapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MmapError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for MmapError {
    fn from(err: std::io::Error) -> Self {
        MmapError::Io(err)
    }
}

impl From<crate::fs::FsError> for MmapError {
    fn from(err: crate::fs::FsError) -> Self {
        match err {
            crate::fs::FsError::Io(io) => MmapError::Io(io),
            other => MmapError::Io(std::io::Error::other(other.to_string())),
        }
    }
}

/// Returns `true` if this build maps files *zero-copy* through a real OS
/// mapping primitive.
///
/// When `false` (targets without `mmap`, such as wasm), [`Mmap`]/[`MmapMut`]
/// still function via a fallback that reads the range into an owned buffer, so
/// correctness is preserved — only the zero-copy benefit is lost. Branch on
/// this when the zero-copy property is load-bearing.
pub fn mmap_supported() -> bool {
    backend::SUPPORTED
}

/// Compute the byte length to map for "the whole file", validating it fits a
/// `usize` and is non-empty.
fn whole_file_len(file: &File) -> Result<usize> {
    let bytes = file.metadata()?.len();
    if bytes == 0 {
        return Err(MmapError::Empty);
    }
    usize::try_from(bytes).map_err(|_| MmapError::InvalidArgument)
}

/// A read-only memory mapping of a file (or file range).
///
/// Dereferences to the mapped bytes as `&[u8]`. The region is unmapped when the
/// `Mmap` is dropped.
pub struct Mmap {
    inner: backend::Mapping,
}

impl Mmap {
    /// Map an entire file read-only.
    ///
    /// The file is opened read-only. Fails with [`MmapError::Empty`] for a
    /// zero-length file (there is nothing to map).
    pub fn map<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = OpenOptions::new().read(true).open(path)?;
        let len = whole_file_len(&file)?;
        Self::from_file(file, 0, len)
    }

    /// Map the range `[offset, offset + len)` of `file` read-only.
    ///
    /// `offset` may be any byte offset; the backend internally aligns the
    /// underlying mapping and exposes exactly `len` bytes starting at
    /// `offset`. The range must lie within the file.
    pub fn from_file(file: File, offset: u64, len: usize) -> Result<Self> {
        if len == 0 {
            return Err(MmapError::Empty);
        }
        validate_range(&file, offset, len)?;
        Ok(Self {
            inner: backend::map(file, offset, len, false)?,
        })
    }

    /// Length of the mapped region in bytes.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the mapped region is empty. A live mapping is never
    /// empty; provided for the `len`/`is_empty` convention.
    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    /// Raw const pointer to the first mapped byte.
    pub fn as_ptr(&self) -> *const u8 {
        self.inner.as_ptr()
    }

    /// The mapped bytes as a slice.
    pub fn as_slice(&self) -> &[u8] {
        self
    }
}

impl Deref for Mmap {
    type Target = [u8];

    #[expect(
        unsafe_code,
        reason = "the backend guarantees ptr/len name a live, readable mapping for the lifetime of &self"
    )]
    fn deref(&self) -> &[u8] {
        // SAFETY: `as_ptr()`/`len()` describe one contiguous readable region
        // that the backend keeps mapped for as long as `self` lives, so the
        // borrow cannot outlive the mapping and the bytes are initialized by
        // the file contents.
        unsafe { slice::from_raw_parts(self.inner.as_ptr(), self.inner.len()) }
    }
}

impl fmt::Debug for Mmap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mmap")
            .field("len", &self.inner.len())
            .field("zero_copy", &backend::SUPPORTED)
            .finish()
    }
}

/// A read-write memory mapping of a file (or file range).
///
/// Dereferences to the mapped bytes as `&[u8]` / `&mut [u8]`. Changes become
/// durable when flushed via [`MmapMut::flush`] (and, on the real OS backends,
/// are written back lazily by the kernel regardless). The region is unmapped
/// when dropped.
pub struct MmapMut {
    inner: backend::Mapping,
}

impl MmapMut {
    /// Map an entire file read-write.
    ///
    /// The file is opened with read and write access. Fails with
    /// [`MmapError::Empty`] for a zero-length file — grow the file first (for
    /// example with [`File::set_len`]) so there is space to map.
    pub fn map<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let len = whole_file_len(&file)?;
        Self::from_file(file, 0, len)
    }

    /// Map the range `[offset, offset + len)` of a read-write `file`.
    ///
    /// `file` must have been opened with write access; the range must lie
    /// within its current length.
    pub fn from_file(file: File, offset: u64, len: usize) -> Result<Self> {
        if len == 0 {
            return Err(MmapError::Empty);
        }
        validate_range(&file, offset, len)?;
        Ok(Self {
            inner: backend::map(file, offset, len, true)?,
        })
    }

    /// Flush dirty pages back to the file, blocking until the write completes.
    ///
    /// On the real OS backends this is `msync(MS_SYNC)` / `FlushViewOfFile`;
    /// on the fallback-by-read backend it writes the owned buffer back to the
    /// file.
    pub fn flush(&self) -> Result<()> {
        self.inner.flush()
    }

    /// Length of the mapped region in bytes.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the mapped region is empty. A live mapping is never
    /// empty; provided for the `len`/`is_empty` convention.
    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    /// Raw const pointer to the first mapped byte.
    pub fn as_ptr(&self) -> *const u8 {
        self.inner.as_ptr()
    }

    /// Raw mutable pointer to the first mapped byte.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.inner.as_mut_ptr()
    }

    /// The mapped bytes as an immutable slice.
    pub fn as_slice(&self) -> &[u8] {
        self
    }

    /// The mapped bytes as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self
    }
}

impl Deref for MmapMut {
    type Target = [u8];

    #[expect(
        unsafe_code,
        reason = "the backend guarantees ptr/len name a live, readable mapping for the lifetime of &self"
    )]
    fn deref(&self) -> &[u8] {
        // SAFETY: as in `Mmap::deref`; the region stays mapped and readable for
        // the lifetime of `self`.
        unsafe { slice::from_raw_parts(self.inner.as_ptr(), self.inner.len()) }
    }
}

impl DerefMut for MmapMut {
    #[expect(
        unsafe_code,
        reason = "a read-write mapping grants exclusive &mut access to its ptr/len region while &mut self is held"
    )]
    fn deref_mut(&mut self) -> &mut [u8] {
        // SAFETY: the mapping was created writable; `&mut self` guarantees no
        // other reference aliases the region, and it stays mapped for the
        // lifetime of the borrow.
        unsafe { slice::from_raw_parts_mut(self.inner.as_mut_ptr(), self.inner.len()) }
    }
}

impl fmt::Debug for MmapMut {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MmapMut")
            .field("len", &self.inner.len())
            .field("zero_copy", &backend::SUPPORTED)
            .finish()
    }
}

/// Validate that `[offset, offset + len)` lies within `file`'s current length.
fn validate_range(file: &File, offset: u64, len: usize) -> Result<()> {
    let file_len = file.metadata()?.len();
    let len_u64 = len as u64;
    let end = offset
        .checked_add(len_u64)
        .ok_or(MmapError::InvalidArgument)?;
    if end > file_len {
        return Err(MmapError::InvalidArgument);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
