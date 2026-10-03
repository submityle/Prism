//! Path helpers that wrap [`std::path`] and add lexical normalization.
//!
//! All helpers are pure string/path manipulation and perform no I/O.

use std::path::{Component, Path, PathBuf};

/// Lexically normalize a path without touching the filesystem.
///
/// Collapses `.` components and resolves `..` against earlier normal
/// components, something [`std::path`] does not do on its own. Leading `..`
/// components are preserved for relative paths (there is nothing to pop), but
/// an absolute path can never ascend above its root.
///
/// This is purely lexical: it does not resolve symlinks or require the path to
/// exist, so it differs from [`std::fs::canonicalize`].
pub fn normalize<P: AsRef<Path>>(path: P) -> PathBuf {
    let path = path.as_ref();
    let mut out = PathBuf::new();
    // Tracks how many trailing components are plain names that `..` may pop.
    let mut popped_normals: usize = 0;
    let mut is_absolute = false;

    for component in path.components() {
        match component {
            Component::Prefix(prefix) => {
                out.push(prefix.as_os_str());
            }
            Component::RootDir => {
                out.push(Component::RootDir.as_os_str());
                is_absolute = true;
                popped_normals = 0;
            }
            Component::CurDir => {}
            Component::Normal(name) => {
                out.push(name);
                popped_normals += 1;
            }
            Component::ParentDir => {
                if popped_normals > 0 {
                    out.pop();
                    popped_normals -= 1;
                } else if !is_absolute {
                    out.push(Component::ParentDir.as_os_str());
                }
                // Absolute path at its root: `..` is a no-op.
            }
        }
    }

    if out.as_os_str().is_empty() {
        out.push(Component::CurDir.as_os_str());
    }
    out
}

/// Join `base` with `rhs`, returning a new owned path.
///
/// If `rhs` is absolute it replaces `base`, matching [`Path::join`].
pub fn join<P: AsRef<Path>, Q: AsRef<Path>>(base: P, rhs: Q) -> PathBuf {
    base.as_ref().join(rhs)
}

/// Return the file extension (without the leading dot), if any.
pub fn extension<P: AsRef<Path>>(path: P) -> Option<String> {
    path.as_ref()
        .extension()
        .map(|ext| ext.to_string_lossy().into_owned())
}

/// Return the file stem (file name without its extension), if any.
pub fn file_stem<P: AsRef<Path>>(path: P) -> Option<String> {
    path.as_ref()
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
}

/// Return the parent directory of a path, if it has one.
pub fn parent<P: AsRef<Path>>(path: P) -> Option<PathBuf> {
    path.as_ref().parent().map(Path::to_path_buf)
}

/// Return `true` if the path is absolute.
pub fn is_absolute<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().is_absolute()
}
