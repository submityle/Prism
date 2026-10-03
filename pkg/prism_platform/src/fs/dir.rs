//! Directory creation, removal, and listing.
//!
//! Listing returns owned [`DirEntry`] values so results can outlive the
//! underlying [`std::fs::ReadDir`] iterator. [`walk`] performs a depth-first
//! recursive listing.

use std::path::{Path, PathBuf};

use crate::fs::Result;

/// Create a single directory. Fails if the parent does not exist.
pub fn create_dir<P: AsRef<Path>>(path: P) -> Result<()> {
    Ok(std::fs::create_dir(path)?)
}

/// Create a directory and all missing parents. Succeeds if it already exists.
pub fn create_dir_all<P: AsRef<Path>>(path: P) -> Result<()> {
    Ok(std::fs::create_dir_all(path)?)
}

/// Remove an empty directory. Fails if it is not empty.
pub fn remove_dir<P: AsRef<Path>>(path: P) -> Result<()> {
    Ok(std::fs::remove_dir(path)?)
}

/// Remove a directory and all of its contents recursively.
pub fn remove_dir_all<P: AsRef<Path>>(path: P) -> Result<()> {
    Ok(std::fs::remove_dir_all(path)?)
}

/// One entry produced by [`read_dir`] or [`walk`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirEntry {
    /// Final path component (file or directory name) as a lossy string.
    pub name: String,
    /// Full path to the entry.
    pub path: PathBuf,
    /// Whether the entry is itself a directory.
    pub is_dir: bool,
}

/// List the immediate children of a directory (non-recursive).
///
/// Entries are returned in the order the OS yields them; callers that need a
/// stable order should sort the result.
pub fn read_dir<P: AsRef<Path>>(path: P) -> Result<Vec<DirEntry>> {
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let is_dir = entry.file_type()?.is_dir();
        entries.push(DirEntry {
            name: entry.file_name().to_string_lossy().into_owned(),
            path: entry.path(),
            is_dir,
        });
    }
    Ok(entries)
}

/// Recursively list every entry under `path`, depth-first.
///
/// Directories appear in the result before their contents. The root `path`
/// itself is not included.
pub fn walk<P: AsRef<Path>>(path: P) -> Result<Vec<DirEntry>> {
    let mut out = Vec::new();
    walk_into(path.as_ref(), &mut out)?;
    Ok(out)
}

/// Recursive worker for [`walk`].
fn walk_into(path: &Path, out: &mut Vec<DirEntry>) -> Result<()> {
    for entry in read_dir(path)? {
        let recurse = entry.is_dir;
        let child = entry.path.clone();
        out.push(entry);
        if recurse {
            walk_into(&child, out)?;
        }
    }
    Ok(())
}
