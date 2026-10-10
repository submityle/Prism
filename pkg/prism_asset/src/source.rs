//! Virtual file system: data-source abstraction (design §10).
//!
//! A loader never touches `std::fs` directly. Instead it asks an
//! [`AssetReader`] for bytes, and the engine wires a stack of named
//! [`AssetSource`]s into an ordered [`AssetSources`] mount table. This single
//! indirection is what makes the same loader work against a loose files tree in
//! the editor, an in-memory fixture in a test, a mounted `.pak`/container in a
//! shipped build, or a patch/DLC overlay — all selected by mount order and the
//! optional `source://name/...` scheme on an [`AssetPath`](crate::AssetPath).
//!
//! ## Why this lives behind `std`
//! The identity/storage/dependency/loader-policy core is `no_std + alloc`.
//! Reading bytes is inherently a platform/`std` concern, so the VFS is gated on
//! the `std` feature. The trait itself is deliberately small and synchronous:
//! an [`AssetReader::read`] returns owned bytes, and the async/batched/streaming
//! machinery (I/O reactor, GPU direct-storage) is layered *on top* by the
//! scheduler — it does not leak into every backend.
//!
//! ## Security (design §23.7)
//! Paths that originate from untrusted containers or the network must never
//! escape their mount root. [`FsSource`] rejects absolute paths and any `..`
//! traversal component before touching the filesystem, so a malicious
//! `../../etc/passwd` resolves to [`ReadError::InvalidPath`], not a file read.

#![cfg(feature = "std")]

use crate::path::AssetPath;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;
use std::path::{Component, Path, PathBuf};

/// Why a read failed. Deliberately small and `std`-free in representation so it
/// can cross the loader boundary cheaply and be matched on by callers that want
/// to distinguish "missing" (try the next mount) from "corrupt" (fail hard).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ReadError {
    /// No such asset at this path in this source.
    NotFound,
    /// The requested byte range lies outside the asset.
    OutOfRange {
        /// First byte requested.
        offset: u64,
        /// Number of bytes requested.
        len: u64,
        /// Actual asset size.
        size: u64,
    },
    /// The path was syntactically rejected before any I/O — absolute paths and
    /// `..` traversal are refused so untrusted paths cannot escape a mount root.
    InvalidPath,
    /// The backend does not support this operation (for example listing a
    /// directory on a flat in-memory source).
    Unsupported,
    /// An underlying I/O error; the string is a human-readable detail, never
    /// matched on programmatically.
    Io(String),
}

impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReadError::NotFound => f.write_str("asset not found"),
            ReadError::OutOfRange { offset, len, size } => write!(
                f,
                "range {offset}..{} out of bounds (size {size})",
                offset.saturating_add(*len)
            ),
            ReadError::InvalidPath => f.write_str("invalid or unsafe path"),
            ReadError::Unsupported => f.write_str("operation not supported by source"),
            ReadError::Io(detail) => write!(f, "io error: {detail}"),
        }
    }
}

impl std::error::Error for ReadError {}

/// Lightweight metadata about an asset, used for change detection (hot reload),
/// range validation, and budget accounting.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AssetMeta {
    /// Size in bytes.
    pub size: u64,
    /// Last-modified time in milliseconds since the Unix epoch, if the backend
    /// can report it. In-memory and container sources may have `None`.
    pub modified_ms: Option<u64>,
    /// Whether this path names a directory rather than a file.
    pub is_dir: bool,
}

/// A read-only byte source: loose files, an in-memory fixture, a mounted
/// container, or a remote cache. The scheduler calls these on an I/O worker, so
/// implementations may block; they must be `Send + Sync` to be shared across
/// workers.
///
/// `read_range` has a correct default (read-all then slice) so a backend only
/// needs to override it when it can seek cheaply (files, container chunks).
pub trait AssetReader: Send + Sync {
    /// Reads the full contents of `path`.
    ///
    /// # Errors
    /// [`ReadError::NotFound`] if the path does not exist, [`ReadError::InvalidPath`]
    /// if it is unsafe, or [`ReadError::Io`] on an underlying failure.
    fn read(&self, path: &str) -> Result<Vec<u8>, ReadError>;

    /// Reads `len` bytes starting at `offset`. The default implementation reads
    /// the whole asset and slices it; seekable backends should override this to
    /// avoid loading the entire payload (streaming, §11).
    ///
    /// # Errors
    /// As [`AssetReader::read`], plus [`ReadError::OutOfRange`] when the range
    /// exceeds the asset size.
    fn read_range(&self, path: &str, offset: u64, len: u64) -> Result<Vec<u8>, ReadError> {
        let bytes = self.read(path)?;
        let size = bytes.len() as u64;
        let end = offset
            .checked_add(len)
            .ok_or(ReadError::OutOfRange { offset, len, size })?;
        if end > size {
            return Err(ReadError::OutOfRange { offset, len, size });
        }
        Ok(bytes[offset as usize..end as usize].to_vec())
    }

    /// Returns metadata for `path`.
    ///
    /// # Errors
    /// As [`AssetReader::read`].
    fn metadata(&self, path: &str) -> Result<AssetMeta, ReadError>;

    /// Lists the immediate entries of directory `dir` (names only, not full
    /// paths), used by hot-reload discovery and tooling.
    ///
    /// # Errors
    /// [`ReadError::Unsupported`] for flat sources that cannot enumerate.
    fn list(&self, dir: &str) -> Result<Vec<String>, ReadError> {
        let _ = dir;
        Err(ReadError::Unsupported)
    }

    /// Whether `path` exists in this source. Default probes [`metadata`].
    ///
    /// [`metadata`]: AssetReader::metadata
    fn exists(&self, path: &str) -> bool {
        self.metadata(path).is_ok()
    }
}

/// Normalizes and validates a logical asset path into mount-relative segments.
///
/// Returns `None` (meaning [`ReadError::InvalidPath`]) when the path is absolute
/// or contains a `..` component, so untrusted paths can never escape a mount
/// root. `.` components and empty segments are dropped; the result is the
/// cleaned, forward-slash-joined relative path.
fn sanitize_relative(path: &str) -> Option<String> {
    if path.starts_with('/') || path.starts_with('\\') {
        return None;
    }
    let mut out: Vec<&str> = Vec::new();
    for seg in path.split(['/', '\\']) {
        match seg {
            "" | "." => {}
            ".." => return None,
            s => out.push(s),
        }
    }
    Some(out.join("/"))
}

/// An in-memory [`AssetReader`] backed by a path→bytes map. The workhorse for
/// tests and for synthesizing assets, and a reference implementation of the
/// trait's contract (range checking, listing, metadata).
#[derive(Default, Clone)]
pub struct MemSource {
    files: BTreeMap<String, Arc<[u8]>>,
}

impl MemSource {
    /// Creates an empty in-memory source.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts (or replaces) a file. The path is normalized the same way reads
    /// are, so `insert("a/./b.png", ..)` is retrievable as `a/b.png`.
    pub fn insert(&mut self, path: &str, bytes: impl Into<Arc<[u8]>>) {
        if let Some(clean) = sanitize_relative(path) {
            self.files.insert(clean, bytes.into());
        }
    }

    /// Removes a file, returning whether it was present (used by hot-reload
    /// delete simulation in tests).
    pub fn remove(&mut self, path: &str) -> bool {
        match sanitize_relative(path) {
            Some(clean) => self.files.remove(&clean).is_some(),
            None => false,
        }
    }

    /// The number of stored files.
    #[must_use]
    pub fn len(&self) -> usize {
        self.files.len()
    }

    /// Whether the source holds no files.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }
}

impl AssetReader for MemSource {
    fn read(&self, path: &str) -> Result<Vec<u8>, ReadError> {
        let clean = sanitize_relative(path).ok_or(ReadError::InvalidPath)?;
        self.files
            .get(&clean)
            .map(|b| b.to_vec())
            .ok_or(ReadError::NotFound)
    }

    fn read_range(&self, path: &str, offset: u64, len: u64) -> Result<Vec<u8>, ReadError> {
        let clean = sanitize_relative(path).ok_or(ReadError::InvalidPath)?;
        let bytes = self.files.get(&clean).ok_or(ReadError::NotFound)?;
        let size = bytes.len() as u64;
        let end = offset
            .checked_add(len)
            .ok_or(ReadError::OutOfRange { offset, len, size })?;
        if end > size {
            return Err(ReadError::OutOfRange { offset, len, size });
        }
        Ok(bytes[offset as usize..end as usize].to_vec())
    }

    fn metadata(&self, path: &str) -> Result<AssetMeta, ReadError> {
        let clean = sanitize_relative(path).ok_or(ReadError::InvalidPath)?;
        // A path that is a strict prefix (directory) of stored files is a dir.
        if let Some(bytes) = self.files.get(&clean) {
            return Ok(AssetMeta {
                size: bytes.len() as u64,
                modified_ms: None,
                is_dir: false,
            });
        }
        let dir_prefix = if clean.is_empty() {
            String::new()
        } else {
            let mut p = clean.clone();
            p.push('/');
            p
        };
        if clean.is_empty() || self.files.keys().any(|k| k.starts_with(&dir_prefix)) {
            return Ok(AssetMeta {
                size: 0,
                modified_ms: None,
                is_dir: true,
            });
        }
        Err(ReadError::NotFound)
    }

    fn list(&self, dir: &str) -> Result<Vec<String>, ReadError> {
        let clean = sanitize_relative(dir).ok_or(ReadError::InvalidPath)?;
        let prefix = if clean.is_empty() {
            String::new()
        } else {
            let mut p = clean.clone();
            p.push('/');
            p
        };
        // Collect immediate children (files and sub-directory names), unique.
        let mut names: BTreeMap<String, ()> = BTreeMap::new();
        for key in self.files.keys() {
            let Some(rest) = key.strip_prefix(&prefix) else {
                continue;
            };
            if rest.is_empty() {
                continue;
            }
            let child = match rest.find('/') {
                Some(slash) => &rest[..slash],
                None => rest,
            };
            names.insert(child.to_string(), ());
        }
        if names.is_empty()
            && !(clean.is_empty() || self.metadata(dir).map(|m| m.is_dir).unwrap_or(false))
        {
            return Err(ReadError::NotFound);
        }
        Ok(names.into_keys().collect())
    }
}

/// A filesystem-backed [`AssetReader`] rooted at a directory. All reads are
/// confined to the root: absolute paths and `..` traversal are rejected before
/// any syscall (design §23.7).
#[derive(Clone)]
pub struct FsSource {
    root: PathBuf,
}

impl FsSource {
    /// Roots a source at `root`. The directory need not exist yet; errors
    /// surface per-read.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The root directory of this source.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Joins a sanitized logical path onto the root, rejecting traversal.
    fn resolve(&self, path: &str) -> Result<PathBuf, ReadError> {
        let clean = sanitize_relative(path).ok_or(ReadError::InvalidPath)?;
        let mut full = self.root.clone();
        for seg in clean.split('/').filter(|s| !s.is_empty()) {
            full.push(seg);
        }
        // Defense in depth: ensure no symlink/`.` trickery re-introduced a
        // parent escape in the composed path.
        if full.components().any(|c| matches!(c, Component::ParentDir)) {
            return Err(ReadError::InvalidPath);
        }
        Ok(full)
    }

    fn map_io(err: &std::io::Error) -> ReadError {
        if err.kind() == std::io::ErrorKind::NotFound {
            ReadError::NotFound
        } else {
            ReadError::Io(err.to_string())
        }
    }
}

impl AssetReader for FsSource {
    fn read(&self, path: &str) -> Result<Vec<u8>, ReadError> {
        let full = self.resolve(path)?;
        std::fs::read(&full).map_err(|e| Self::map_io(&e))
    }

    fn read_range(&self, path: &str, offset: u64, len: u64) -> Result<Vec<u8>, ReadError> {
        use std::io::{Read, Seek, SeekFrom};
        let full = self.resolve(path)?;
        let mut file = std::fs::File::open(&full).map_err(|e| Self::map_io(&e))?;
        let size = file.metadata().map_err(|e| Self::map_io(&e))?.len();
        let end = offset
            .checked_add(len)
            .ok_or(ReadError::OutOfRange { offset, len, size })?;
        if end > size {
            return Err(ReadError::OutOfRange { offset, len, size });
        }
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| Self::map_io(&e))?;
        let mut buf = alloc::vec![0u8; len as usize];
        file.read_exact(&mut buf).map_err(|e| Self::map_io(&e))?;
        Ok(buf)
    }

    fn metadata(&self, path: &str) -> Result<AssetMeta, ReadError> {
        let full = self.resolve(path)?;
        let meta = std::fs::metadata(&full).map_err(|e| Self::map_io(&e))?;
        let modified_ms = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as u64);
        Ok(AssetMeta {
            size: meta.len(),
            modified_ms,
            is_dir: meta.is_dir(),
        })
    }

    fn list(&self, dir: &str) -> Result<Vec<String>, ReadError> {
        let full = self.resolve(dir)?;
        let mut names = Vec::new();
        for entry in std::fs::read_dir(&full).map_err(|e| Self::map_io(&e))? {
            let entry = entry.map_err(|e| Self::map_io(&e))?;
            if let Some(name) = entry.file_name().to_str() {
                names.push(name.to_string());
            }
        }
        names.sort();
        Ok(names)
    }
}

/// One named, prioritized mount in the [`AssetSources`] table.
struct Mount {
    name: String,
    reader: Arc<dyn AssetReader>,
    priority: i32,
    seq: u64,
}

/// An ordered table of named [`AssetSource`]s resolving an
/// [`AssetPath`](crate::AssetPath) to concrete bytes.
///
/// Resolution rules (design §10.4):
/// - A path with a `source://name/...` scheme reads **only** the mount named
///   `name` (explicit targeting; patch/DLC and tooling rely on this).
/// - A path without a scheme searches mounts in **overlay order** — highest
///   `priority` first, later registration breaking ties — and returns the first
///   mount that has the asset. This is how a patch/DLC overlay shadows a base
///   mount without the loader knowing.
#[derive(Default)]
pub struct AssetSources {
    mounts: Vec<Mount>,
    next_seq: u64,
}

impl AssetSources {
    /// Creates an empty mount table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Mounts `reader` under `name` at `priority` (higher shadows lower; ties
    /// break toward the later mount). Re-mounting the same name replaces the
    /// previous reader.
    pub fn mount(&mut self, name: &str, reader: Arc<dyn AssetReader>, priority: i32) {
        let seq = self.next_seq;
        self.next_seq += 1;
        if let Some(existing) = self.mounts.iter_mut().find(|m| m.name == name) {
            existing.reader = reader;
            existing.priority = priority;
            existing.seq = seq;
        } else {
            self.mounts.push(Mount {
                name: name.to_string(),
                reader,
                priority,
                seq,
            });
        }
    }

    /// Removes a mount by name, returning whether it was present.
    pub fn unmount(&mut self, name: &str) -> bool {
        let before = self.mounts.len();
        self.mounts.retain(|m| m.name != name);
        self.mounts.len() != before
    }

    /// Mount names in overlay order (highest priority first).
    #[must_use]
    pub fn overlay_order(&self) -> Vec<&str> {
        let mut refs: Vec<&Mount> = self.mounts.iter().collect();
        refs.sort_by_key(|m| (core::cmp::Reverse(m.priority), core::cmp::Reverse(m.seq)));
        refs.into_iter().map(|m| m.name.as_str()).collect()
    }

    /// The number of mounts.
    #[must_use]
    pub fn len(&self) -> usize {
        self.mounts.len()
    }

    /// Whether no mounts are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mounts.is_empty()
    }

    fn mounts_in_overlay_order(&self) -> Vec<&Mount> {
        let mut refs: Vec<&Mount> = self.mounts.iter().collect();
        refs.sort_by_key(|m| (core::cmp::Reverse(m.priority), core::cmp::Reverse(m.seq)));
        refs
    }

    /// Reads the asset named by `path`, honoring the scheme and overlay rules.
    ///
    /// # Errors
    /// [`ReadError::NotFound`] if no mount has the asset (or the named mount
    /// does not exist), propagating a hard [`ReadError::Io`]/[`ReadError::InvalidPath`]
    /// from the resolving mount.
    pub fn read(&self, path: &AssetPath) -> Result<Vec<u8>, ReadError> {
        self.dispatch(path, |reader, rel| reader.read(rel))
    }

    /// Reads a byte range of the asset named by `path` (streaming).
    ///
    /// # Errors
    /// As [`AssetSources::read`], plus [`ReadError::OutOfRange`].
    pub fn read_range(
        &self,
        path: &AssetPath,
        offset: u64,
        len: u64,
    ) -> Result<Vec<u8>, ReadError> {
        self.dispatch(path, |reader, rel| reader.read_range(rel, offset, len))
    }

    /// Returns metadata for the asset named by `path`.
    ///
    /// # Errors
    /// As [`AssetSources::read`].
    pub fn metadata(&self, path: &AssetPath) -> Result<AssetMeta, ReadError> {
        self.dispatch(path, |reader, rel| reader.metadata(rel))
    }

    /// Whether some mount can resolve `path`.
    #[must_use]
    pub fn exists(&self, path: &AssetPath) -> bool {
        self.metadata(path).is_ok()
    }

    /// Core resolution: pick the mount(s) and run `op`, falling back across the
    /// overlay for `NotFound` only (a hard error from the owning mount is
    /// returned immediately, so corruption is never silently masked by a lower
    /// mount).
    fn dispatch<T>(
        &self,
        path: &AssetPath,
        op: impl Fn(&dyn AssetReader, &str) -> Result<T, ReadError>,
    ) -> Result<T, ReadError> {
        let rel = path.path();
        match path.scheme() {
            Some(name) => {
                let mount = self
                    .mounts
                    .iter()
                    .find(|m| m.name == name)
                    .ok_or(ReadError::NotFound)?;
                op(mount.reader.as_ref(), rel)
            }
            None => {
                let mut last = ReadError::NotFound;
                for mount in self.mounts_in_overlay_order() {
                    match op(mount.reader.as_ref(), rel) {
                        Ok(v) => return Ok(v),
                        Err(ReadError::NotFound) => last = ReadError::NotFound,
                        Err(other) => return Err(other),
                    }
                }
                Err(last)
            }
        }
    }
}
