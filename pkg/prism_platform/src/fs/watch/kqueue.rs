//! Native macOS / BSD watcher backed by `kqueue` + `EVFILT_VNODE`.
//!
//! Each watched file or directory is opened to obtain a descriptor, which is
//! registered on a shared `kqueue` with the vnode filter. The kernel then
//! reports `NOTE_WRITE` / `NOTE_DELETE` / `NOTE_RENAME` / … edges on that
//! descriptor without any polling. This is the real native backend on macOS,
//! exercised by this crate's test-suite.
//!
//! Every FFI call is a dependency-free `extern` declaration resolved against
//! the C runtime that `std` already links (no `libc` crate), mirroring
//! [`crate::fs::mmap`] and [`crate::vm`].
//!
//! ## Recursive caveat
//! `EVFILT_VNODE` watches a single vnode, so a recursive directory watch
//! registers the directory plus each entry that exists at `watch` time. Entries
//! created afterwards are not auto-registered — but the parent directory
//! reports a `NOTE_WRITE`, so a caller can re-`watch` to pick up new children.
//! When fully-dynamic recursion matters, prefer [`super::Backend::Poll`]. This
//! limitation is documented, not stubbed.

use core::ffi::{c_int, c_void};
use core::time::Duration;
use std::collections::HashMap;
use std::os::fd::{AsRawFd, RawFd};
use std::path::{Path, PathBuf};

use super::{Event, EventKind, Result, WatchError};

#[expect(
    unsafe_code,
    reason = "kqueue/kevent/close are the native vnode-watch syscalls, declared as dependency-free C FFI"
)]
unsafe extern "C" {
    fn kqueue() -> c_int;
    fn kevent(
        kq: c_int,
        changelist: *const Kevent,
        nchanges: c_int,
        eventlist: *mut Kevent,
        nevents: c_int,
        timeout: *const Timespec,
    ) -> c_int;
    fn close(fd: c_int) -> c_int;
}

#[expect(unsafe_code, reason = "errno accessor FFI declaration")]
unsafe extern "C" {
    fn __error() -> *mut c_int;
}

/// Read this thread's `errno`.
#[expect(unsafe_code, reason = "reading this thread's errno through its C accessor")]
fn errno() -> c_int {
    // SAFETY: `__error` returns a valid pointer to this thread's `errno`, which
    // we only read.
    unsafe { *__error() }
}

/// `struct kevent` (64-bit macOS/BSD ABI).
#[repr(C)]
#[derive(Clone, Copy)]
struct Kevent {
    ident: usize,
    filter: i16,
    flags: u16,
    fflags: u32,
    data: isize,
    udata: *mut c_void,
}

/// `struct timespec`.
#[repr(C)]
struct Timespec {
    tv_sec: i64,
    tv_nsec: i64,
}

const EVFILT_VNODE: i16 = -4;
const EV_ADD: u16 = 0x0001;
const EV_DELETE: u16 = 0x0002;
const EV_CLEAR: u16 = 0x0020;

const NOTE_DELETE: u32 = 0x0001;
const NOTE_WRITE: u32 = 0x0002;
const NOTE_EXTEND: u32 = 0x0004;
const NOTE_ATTRIB: u32 = 0x0008;
const NOTE_LINK: u32 = 0x0010;
const NOTE_RENAME: u32 = 0x0020;
const NOTE_REVOKE: u32 = 0x0040;

/// The set of vnode changes we ask the kernel to report.
const WATCH_MASK: u32 =
    NOTE_DELETE | NOTE_WRITE | NOTE_EXTEND | NOTE_ATTRIB | NOTE_LINK | NOTE_RENAME | NOTE_REVOKE;

const EINTR: c_int = 4;

/// A single registered descriptor and the path it stands for.
struct Entry {
    path: PathBuf,
    // Owns the descriptor; dropping it closes the fd and the kernel drops the
    // registration automatically.
    _file: std::fs::File,
}

/// Native `kqueue` watcher.
pub(super) struct KqueueWatcher {
    kq: c_int,
    /// Registered descriptors keyed by their fd (the kevent `ident`).
    entries: HashMap<RawFd, Entry>,
    /// For each watch root, the fds registered on its behalf (root + entries),
    /// so [`unwatch`](Self::unwatch) can tear exactly them down.
    roots: HashMap<PathBuf, Vec<RawFd>>,
}

impl KqueueWatcher {
    #[expect(unsafe_code, reason = "kqueue() allocates the kernel event queue")]
    pub(super) fn new() -> Result<Self> {
        // SAFETY: `kqueue` takes no arguments and returns a new fd or -1.
        let kq = unsafe { kqueue() };
        if kq < 0 {
            return Err(WatchError::System(errno()));
        }
        Ok(Self {
            kq,
            entries: HashMap::new(),
            roots: HashMap::new(),
        })
    }

    pub(super) fn watch(&mut self, path: &Path, recursive: bool) -> Result<()> {
        // Re-watching replaces the previous registration for this root.
        if self.roots.contains_key(path) {
            self.unwatch(path)?;
        }

        let mut fds = Vec::new();
        self.register_path(path, &mut fds)?;

        // For a recursive directory watch, also register the entries present
        // right now (see the module-level recursive caveat).
        if recursive && path.is_dir() {
            self.register_tree(path, &mut fds);
        }

        self.roots.insert(path.to_path_buf(), fds);
        Ok(())
    }

    pub(super) fn unwatch(&mut self, path: &Path) -> Result<()> {
        let Some(fds) = self.roots.remove(path) else {
            return Err(WatchError::NotWatched);
        };
        for fd in fds {
            // Best-effort removal; dropping the owning file also drops the
            // registration, so a failure here is non-fatal.
            self.deregister(fd);
            self.entries.remove(&fd);
        }
        Ok(())
    }

    pub(super) fn poll(&mut self, timeout: Duration) -> Result<Vec<Event>> {
        const CAP: usize = 64;
        let mut buf: [Kevent; CAP] = [Kevent {
            ident: 0,
            filter: 0,
            flags: 0,
            fflags: 0,
            data: 0,
            udata: core::ptr::null_mut(),
        }; CAP];

        let ts = Timespec {
            tv_sec: timeout.as_secs() as i64,
            tv_nsec: timeout.subsec_nanos() as i64,
        };

        // SAFETY: `self.kq` is a live kqueue fd; the changelist is empty
        // (null/0), and `buf`/`CAP` describe a writable array of `CAP` events
        // the kernel fills in. `ts` is a valid, live timespec.
        #[expect(unsafe_code, reason = "kevent dequeues ready vnode events into our buffer")]
        let n = unsafe {
            kevent(
                self.kq,
                core::ptr::null(),
                0,
                buf.as_mut_ptr(),
                CAP as c_int,
                &ts,
            )
        };

        if n < 0 {
            let e = errno();
            if e == EINTR {
                return Ok(Vec::new());
            }
            return Err(WatchError::System(e));
        }

        let mut events = Vec::with_capacity(n as usize);
        let mut dropped = Vec::new();
        for ev in buf.iter().take(n as usize) {
            let fd = ev.ident as RawFd;
            let Some(entry) = self.entries.get(&fd) else {
                continue;
            };
            events.push(Event {
                path: entry.path.clone(),
                kind: classify(ev.fflags),
            });
            // Once a vnode is deleted/revoked its descriptor is useless; retire
            // it so a later re-create can be re-watched cleanly.
            if ev.fflags & (NOTE_DELETE | NOTE_REVOKE) != 0 {
                dropped.push(fd);
            }
        }
        for fd in dropped {
            self.entries.remove(&fd);
            for fds in self.roots.values_mut() {
                fds.retain(|&f| f != fd);
            }
        }
        Ok(events)
    }

    /// Open `path` and register its descriptor on the queue.
    fn register_path(&mut self, path: &Path, fds: &mut Vec<RawFd>) -> Result<()> {
        // `File::open` works on directories on unix, yielding a descriptor
        // suitable for `EVFILT_VNODE`.
        let file = std::fs::File::open(path).map_err(WatchError::Io)?;
        let fd = file.as_raw_fd();
        self.register_fd(fd)?;
        self.entries.insert(
            fd,
            Entry {
                path: path.to_path_buf(),
                _file: file,
            },
        );
        fds.push(fd);
        Ok(())
    }

    /// Register every immediate-and-deeper entry of directory `dir`.
    fn register_tree(&mut self, dir: &Path, fds: &mut Vec<RawFd>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let child = entry.path();
            let is_dir = child.is_dir();
            // Ignore individual failures (races, permissions) — a best-effort
            // recursive registration, as documented.
            let _ = self.register_path(&child, fds);
            if is_dir {
                self.register_tree(&child, fds);
            }
        }
    }

    /// Add `fd` to the queue with the vnode filter.
    fn register_fd(&self, fd: RawFd) -> Result<()> {
        let change = Kevent {
            ident: fd as usize,
            filter: EVFILT_VNODE,
            flags: EV_ADD | EV_CLEAR,
            fflags: WATCH_MASK,
            data: 0,
            udata: core::ptr::null_mut(),
        };
        // SAFETY: `self.kq` is a live kqueue fd; `&change` is one valid change
        // entry (nchanges = 1); the eventlist is null/0 so the kernel only
        // applies the change and returns no events. A null timeout returns
        // immediately for a pure registration.
        #[expect(unsafe_code, reason = "kevent registers one vnode filter on the queue")]
        let rc = unsafe {
            kevent(
                self.kq,
                &change,
                1,
                core::ptr::null_mut(),
                0,
                core::ptr::null(),
            )
        };
        if rc < 0 {
            Err(WatchError::System(errno()))
        } else {
            Ok(())
        }
    }

    /// Remove `fd`'s registration from the queue (best-effort).
    fn deregister(&self, fd: RawFd) {
        let change = Kevent {
            ident: fd as usize,
            filter: EVFILT_VNODE,
            flags: EV_DELETE,
            fflags: 0,
            data: 0,
            udata: core::ptr::null_mut(),
        };
        // SAFETY: identical contract to `register_fd`; a single valid change
        // entry removing one filter, no events requested.
        #[expect(unsafe_code, reason = "kevent removes one vnode filter from the queue")]
        unsafe {
            kevent(
                self.kq,
                &change,
                1,
                core::ptr::null_mut(),
                0,
                core::ptr::null(),
            );
        }
    }
}

impl Drop for KqueueWatcher {
    #[expect(unsafe_code, reason = "close releases the kqueue fd exactly once on drop")]
    fn drop(&mut self) {
        // SAFETY: `self.kq` was returned by `kqueue()` and is closed exactly
        // once here at end of life. The owned `File`s in `entries` drop
        // afterwards, closing their descriptors.
        unsafe {
            close(self.kq);
        }
    }
}

/// Map a vnode `fflags` bitmask to the closest [`EventKind`].
fn classify(fflags: u32) -> EventKind {
    const MODIFIED: u32 = NOTE_WRITE | NOTE_EXTEND | NOTE_ATTRIB | NOTE_LINK;
    if fflags & (NOTE_DELETE | NOTE_REVOKE) != 0 {
        EventKind::Removed
    } else if fflags & NOTE_RENAME != 0 {
        EventKind::Renamed
    } else if fflags & MODIFIED != 0 {
        // Content writes, truncation/extension, attribute, and link-count
        // changes all map to a single "the file changed, re-read it" edge.
        EventKind::Modified
    } else {
        EventKind::Other
    }
}
