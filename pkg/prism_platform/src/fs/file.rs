//! Whole-file operations and file metadata.
//!
//! These functions mirror the shape of [`std::fs`] but return the crate's
//! [`Result`] and a compact [`Metadata`] snapshot. They never panic on I/O
//! errors.

use std::path::Path;
use std::time::SystemTime;

use crate::fs::Result;

/// Re-export of [`std::fs::File`]; opened via [`OpenOptions::open`].
pub use std::fs::File;

/// Read an entire file into a byte vector.
pub fn read<P: AsRef<Path>>(path: P) -> Result<Vec<u8>> {
    Ok(std::fs::read(path)?)
}

/// Read an entire file into a [`String`], failing if it is not valid UTF-8.
pub fn read_to_string<P: AsRef<Path>>(path: P) -> Result<String> {
    Ok(std::fs::read_to_string(path)?)
}

/// Write bytes to a file, creating it (and truncating any existing contents).
pub fn write<P: AsRef<Path>, C: AsRef<[u8]>>(path: P, contents: C) -> Result<()> {
    Ok(std::fs::write(path, contents)?)
}

/// Append bytes to a file, creating it if it does not exist.
pub fn append<P: AsRef<Path>, C: AsRef<[u8]>>(path: P, contents: C) -> Result<()> {
    use std::io::Write as _;
    let mut file = OpenOptions::new().append(true).create(true).open(path)?;
    file.write_all(contents.as_ref())?;
    Ok(())
}

/// Returns `true` if the path exists and is reachable.
///
/// Unlike [`std::path::Path::exists`], permission or other I/O errors while
/// probing are surfaced instead of being treated as "does not exist".
pub fn exists<P: AsRef<Path>>(path: P) -> Result<bool> {
    Ok(path.as_ref().try_exists()?)
}

/// Remove a file. Fails if the path is a directory or does not exist.
pub fn remove_file<P: AsRef<Path>>(path: P) -> Result<()> {
    Ok(std::fs::remove_file(path)?)
}

/// Copy the contents of `from` to `to`, returning the number of bytes copied.
pub fn copy<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> Result<u64> {
    Ok(std::fs::copy(from, to)?)
}

/// Rename (move) a file or directory, replacing `to` if it already exists.
pub fn rename<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> Result<()> {
    Ok(std::fs::rename(from, to)?)
}

/// Read a compact [`Metadata`] snapshot for a path.
pub fn metadata<P: AsRef<Path>>(path: P) -> Result<Metadata> {
    let meta = std::fs::metadata(path)?;
    Ok(Metadata::from_std(&meta))
}

/// A compact, copyable snapshot of filesystem metadata.
///
/// Captures only the fields the engine routinely needs; obtain it via
/// [`metadata`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Metadata {
    /// Size of the file in bytes (meaningless for directories on some OSes).
    pub len: u64,
    /// Whether the entry is a regular file.
    pub is_file: bool,
    /// Whether the entry is a directory.
    pub is_dir: bool,
    /// Last modification time, if the platform exposes one.
    pub modified: Option<SystemTime>,
}

impl Metadata {
    /// Build a snapshot from a [`std::fs::Metadata`] value.
    fn from_std(meta: &std::fs::Metadata) -> Self {
        Self {
            len: meta.len(),
            is_file: meta.is_file(),
            is_dir: meta.is_dir(),
            modified: meta.modified().ok(),
        }
    }
}

/// Builder for opening a [`File`] with explicit access and creation flags.
///
/// Mirrors [`std::fs::OpenOptions`] with a chainable, `Copy` surface.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenOptions {
    read: bool,
    write: bool,
    append: bool,
    truncate: bool,
    create: bool,
    create_new: bool,
}

impl OpenOptions {
    /// Create an options builder with every flag disabled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Open for reading.
    pub fn read(mut self, read: bool) -> Self {
        self.read = read;
        self
    }

    /// Open for writing.
    pub fn write(mut self, write: bool) -> Self {
        self.write = write;
        self
    }

    /// Open in append mode (writes go to the end of the file).
    pub fn append(mut self, append: bool) -> Self {
        self.append = append;
        self
    }

    /// Truncate the file to zero length when opening for writing.
    pub fn truncate(mut self, truncate: bool) -> Self {
        self.truncate = truncate;
        self
    }

    /// Create the file if it does not already exist.
    pub fn create(mut self, create: bool) -> Self {
        self.create = create;
        self
    }

    /// Create the file, failing if it already exists.
    pub fn create_new(mut self, create_new: bool) -> Self {
        self.create_new = create_new;
        self
    }

    /// Open the file at `path` with the configured flags.
    pub fn open<P: AsRef<Path>>(&self, path: P) -> Result<File> {
        let file = std::fs::OpenOptions::new()
            .read(self.read)
            .write(self.write)
            .append(self.append)
            .truncate(self.truncate)
            .create(self.create)
            .create_new(self.create_new)
            .open(path)?;
        Ok(file)
    }
}
