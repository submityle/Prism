//! File-system watching for hot-reload of assets, scripts, and shaders.
//!
//! This is the M4 `watch` layer from the design doc (§6 文件监视 watch, §22).
//! It gives `prism_asset` / `prism_script` a single facade to notice when a
//! watched file or directory changes on disk so they can hot-reload.
//!
//! Requires the `watch` feature (and therefore `std`).
//!
//! ## Backends
//! - **macOS / BSD — `kqueue` (native, real):** registers a `kqueue` filter
//!   per watched file/directory descriptor and reports `EVFILT_VNODE` events.
//!   Exercised by this crate's test-suite on macOS.
//! - **Everywhere else — polling (portable, real):** a `stat`-based mtime/size
//!   diff that re-scans watched paths. This is the honest baseline backend and
//!   works on every `std` platform (Linux, Windows, …). It is also selectable
//!   explicitly via [`Backend::Poll`].
//!
//! The design doc also lists `inotify` (Linux) and `ReadDirectoryChangesW`
//! (Windows) as the eventual native backends. Those are **not implemented
//! here**: on Linux/Windows this module uses the portable polling backend,
//! which the design explicitly accepts as the honest baseline. Wiring the
//! native Linux/Windows backends is future work and is called out honestly
//! rather than stubbed.
//!
//! ## Semantics
//! - Events are coalesced edge notifications, not a guaranteed per-change log.
//!   After any event, re-read the file; do not assume one event per write.
//! - The polling backend fully supports recursive directory watching. The
//!   `kqueue` backend registers the directory plus its *current* entries when
//!   `recursive` is requested; entries created *after* the watch starts are
//!   not auto-registered (a known `kqueue` limitation), though the parent
//!   directory still reports a change. Use [`Backend::Poll`] when you need
//!   fully-dynamic recursive trees.
//! - Rename detection is best-effort; a rename may surface as
//!   [`EventKind::Renamed`] (kqueue) or as a [`EventKind::Removed`] +
//!   [`EventKind::Created`] pair (polling).

use core::fmt;
use core::time::Duration;
use std::path::{Path, PathBuf};

mod poll;

#[cfg(any(
    target_vendor = "apple",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "dragonfly"
))]
mod kqueue;

/// Result type for every watch operation.
pub type Result<T> = core::result::Result<T, WatchError>;

/// What happened to a watched path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EventKind {
    /// A new file or directory appeared at the path.
    Created,
    /// Existing contents (or size/mtime) changed.
    Modified,
    /// The path was deleted.
    Removed,
    /// The path was renamed/moved.
    Renamed,
    /// A change the backend could not classify more precisely.
    Other,
}

/// A single change notification for a watched path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Event {
    /// The affected path. For directory watches this is the directory (native
    /// `kqueue`) or the specific entry (polling), depending on backend.
    pub path: PathBuf,
    /// The kind of change.
    pub kind: EventKind,
}

/// Which backend a [`Watcher`] is using.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// A native OS watcher (`kqueue` on macOS/BSD).
    Native,
    /// The portable `stat`-based polling watcher.
    Poll,
}

/// Why a watch operation failed.
#[derive(Debug)]
pub enum WatchError {
    /// The requested backend is not available in this build/platform (for
    /// example [`Backend::Native`] on Linux, where only polling is wired up).
    Unsupported,
    /// A path was asked to be unwatched that was never watched.
    NotWatched,
    /// An underlying I/O error (opening the path, reading a directory, …).
    Io(std::io::Error),
    /// The OS watcher syscall failed; carries the platform error code.
    System(i32),
}

impl fmt::Display for WatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WatchError::Unsupported => write!(f, "requested watch backend is unavailable here"),
            WatchError::NotWatched => write!(f, "path was not being watched"),
            WatchError::Io(err) => write!(f, "watch I/O error: {err}"),
            WatchError::System(code) => write!(f, "OS watcher failed (code {code})"),
        }
    }
}

impl std::error::Error for WatchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WatchError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for WatchError {
    fn from(err: std::io::Error) -> Self {
        WatchError::Io(err)
    }
}

/// Returns `true` if a native OS watcher backend is available in this build.
pub fn watch_native_supported() -> bool {
    #[cfg(any(
        target_vendor = "apple",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "dragonfly"
    ))]
    {
        true
    }
    #[cfg(not(any(
        target_vendor = "apple",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "dragonfly"
    )))]
    {
        false
    }
}

enum Inner {
    #[cfg(any(
        target_vendor = "apple",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "dragonfly"
    ))]
    Kqueue(kqueue::KqueueWatcher),
    Poll(poll::PollWatcher),
}

/// A file-system watcher over a chosen backend.
///
/// Register paths with [`Watcher::watch`], then drain changes with
/// [`Watcher::poll`] (blocking up to a timeout) or [`Watcher::try_poll`]
/// (non-blocking).
pub struct Watcher {
    inner: Inner,
    backend: Backend,
}

impl Watcher {
    /// Create a watcher using the best backend available: the native OS watcher
    /// where one exists, otherwise the portable polling watcher.
    pub fn new() -> Result<Self> {
        if watch_native_supported() {
            Self::with_backend(Backend::Native)
        } else {
            Self::with_backend(Backend::Poll)
        }
    }

    /// Create a watcher using a specific [`Backend`].
    ///
    /// Returns [`WatchError::Unsupported`] if [`Backend::Native`] is requested
    /// on a platform without a native backend.
    pub fn with_backend(backend: Backend) -> Result<Self> {
        match backend {
            Backend::Poll => Ok(Self {
                inner: Inner::Poll(poll::PollWatcher::new()),
                backend: Backend::Poll,
            }),
            Backend::Native => {
                #[cfg(any(
                    target_vendor = "apple",
                    target_os = "freebsd",
                    target_os = "netbsd",
                    target_os = "openbsd",
                    target_os = "dragonfly"
                ))]
                {
                    Ok(Self {
                        inner: Inner::Kqueue(kqueue::KqueueWatcher::new()?),
                        backend: Backend::Native,
                    })
                }
                #[cfg(not(any(
                    target_vendor = "apple",
                    target_os = "freebsd",
                    target_os = "netbsd",
                    target_os = "openbsd",
                    target_os = "dragonfly"
                )))]
                {
                    Err(WatchError::Unsupported)
                }
            }
        }
    }

    /// The backend this watcher is actually using.
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Begin watching `path`. If it is a directory and `recursive` is `true`,
    /// its subtree is watched too (see the module docs for backend-specific
    /// recursive caveats).
    pub fn watch<P: AsRef<Path>>(&mut self, path: P, recursive: bool) -> Result<()> {
        let path = path.as_ref();
        match &mut self.inner {
            #[cfg(any(
                target_vendor = "apple",
                target_os = "freebsd",
                target_os = "netbsd",
                target_os = "openbsd",
                target_os = "dragonfly"
            ))]
            Inner::Kqueue(w) => w.watch(path, recursive),
            Inner::Poll(w) => w.watch(path, recursive),
        }
    }

    /// Stop watching `path`.
    pub fn unwatch<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();
        match &mut self.inner {
            #[cfg(any(
                target_vendor = "apple",
                target_os = "freebsd",
                target_os = "netbsd",
                target_os = "openbsd",
                target_os = "dragonfly"
            ))]
            Inner::Kqueue(w) => w.unwatch(path),
            Inner::Poll(w) => w.unwatch(path),
        }
    }

    /// Block up to `timeout` for changes, returning all events observed.
    ///
    /// Returns an empty vector if nothing changed within `timeout`.
    pub fn poll(&mut self, timeout: Duration) -> Result<Vec<Event>> {
        match &mut self.inner {
            #[cfg(any(
                target_vendor = "apple",
                target_os = "freebsd",
                target_os = "netbsd",
                target_os = "openbsd",
                target_os = "dragonfly"
            ))]
            Inner::Kqueue(w) => w.poll(timeout),
            Inner::Poll(w) => w.poll(timeout),
        }
    }

    /// Return any changes observed since the last poll without blocking.
    pub fn try_poll(&mut self) -> Result<Vec<Event>> {
        self.poll(Duration::ZERO)
    }
}

#[cfg(test)]
mod tests;
