//! Native Linux file watcher backed by `inotify`.
//!
//! A recursive root owns one descriptor per directory. New directories are
//! added when their parent reports `IN_CREATE` or `IN_MOVED_TO`; removed or
//! moved-out subtrees are retired from the descriptor map.

use core::ffi::{c_char, c_int, c_short, c_ulong};
use core::time::Duration;
use std::collections::HashMap;
use std::ffi::{CString, OsString};
use std::io::{ErrorKind, Read as _};
use std::os::fd::{AsRawFd, FromRawFd, RawFd};
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};

use super::{Event, EventKind, Result, WatchError};

#[expect(
    unsafe_code,
    reason = "inotify and poll are the Linux native filesystem-watch syscalls"
)]
unsafe extern "C" {
    fn inotify_init1(flags: c_int) -> c_int;
    fn inotify_add_watch(fd: c_int, pathname: *const c_char, mask: u32) -> c_int;
    fn inotify_rm_watch(fd: c_int, wd: c_int) -> c_int;
    fn poll(fds: *mut PollFd, nfds: c_ulong, timeout: c_int) -> c_int;
    fn __errno_location() -> *mut c_int;
}

#[repr(C)]
struct PollFd {
    fd: c_int,
    events: c_short,
    revents: c_short,
}

const IN_NONBLOCK: c_int = 0x800;
const POLLIN: c_short = 0x0001;
const EINTR: c_int = 4;
const EAGAIN: c_int = 11;
const EWOULDBLOCK: c_int = EAGAIN;
const EOVERFLOW: c_int = 75;

const IN_MODIFY: u32 = 0x0000_0002;
const IN_ATTRIB: u32 = 0x0000_0004;
const IN_CLOSE_WRITE: u32 = 0x0000_0008;
const IN_MOVED_FROM: u32 = 0x0000_0040;
const IN_MOVED_TO: u32 = 0x0000_0080;
const IN_CREATE: u32 = 0x0000_0100;
const IN_DELETE: u32 = 0x0000_0200;
const IN_DELETE_SELF: u32 = 0x0000_0400;
const IN_MOVE_SELF: u32 = 0x0000_0800;
const IN_Q_OVERFLOW: u32 = 0x0000_4000;
const IN_IGNORED: u32 = 0x0000_8000;
const IN_ISDIR: u32 = 0x4000_0000;

const WATCH_MASK: u32 = IN_MODIFY
    | IN_ATTRIB
    | IN_CLOSE_WRITE
    | IN_MOVED_FROM
    | IN_MOVED_TO
    | IN_CREATE
    | IN_DELETE
    | IN_DELETE_SELF
    | IN_MOVE_SELF;

const READ_BUFFER_SIZE: usize = 256 * 1024;

#[expect(unsafe_code, reason = "read the Linux thread-local errno value")]
fn errno() -> c_int {
    // SAFETY: `__errno_location` returns this thread's valid errno pointer;
    // only a read is performed.
    unsafe { *__errno_location() }
}

fn io_errno() -> std::io::Error {
    std::io::Error::last_os_error()
}

struct Root {
    recursive: bool,
    watches: Vec<RawFd>,
}

struct Entry {
    root: PathBuf,
    path: PathBuf,
    is_dir: bool,
}

/// Native Linux watcher.
pub(super) struct InotifyWatcher {
    file: std::fs::File,
    roots: HashMap<PathBuf, Root>,
    entries: HashMap<RawFd, Entry>,
    buffer: Vec<u8>,
}

impl InotifyWatcher {
    #[expect(unsafe_code, reason = "inotify_init1 allocates the native watch fd")]
    pub(super) fn new() -> Result<Self> {
        // SAFETY: no pointers are passed; the flags request a non-blocking fd.
        let fd = unsafe { inotify_init1(IN_NONBLOCK) };
        if fd < 0 {
            return Err(WatchError::Io(io_errno()));
        }
        // SAFETY: `fd` is a newly allocated inotify descriptor owned by this
        // value; `File` closes it exactly once when dropped.
        let file = unsafe { std::fs::File::from_raw_fd(fd) };
        Ok(Self {
            file,
            roots: HashMap::new(),
            entries: HashMap::new(),
            buffer: vec![0; READ_BUFFER_SIZE],
        })
    }

    pub(super) fn watch(&mut self, path: &Path, recursive: bool) -> Result<()> {
        if self.roots.contains_key(path) {
            self.unwatch(path)?;
        }

        let metadata = std::fs::symlink_metadata(path).map_err(WatchError::Io)?;
        let root = path.to_path_buf();
        let mut watches = Vec::new();
        if metadata.is_dir() {
            self.add_directory(&root, &root, &mut watches)?;
            if recursive {
                self.add_subdirectories(&root, &root, &mut watches);
            }
        } else {
            self.add_watch(&root, &root, false, &mut watches)?;
        }
        self.roots.insert(root, Root { recursive, watches });
        Ok(())
    }

    pub(super) fn unwatch(&mut self, path: &Path) -> Result<()> {
        let Some(root) = self.roots.remove(path) else {
            return Err(WatchError::NotWatched);
        };
        for wd in root.watches {
            self.remove_watch(wd, false);
        }
        Ok(())
    }

    #[expect(unsafe_code, reason = "wait on the native Linux inotify descriptor")]
    pub(super) fn poll(&mut self, timeout: Duration) -> Result<Vec<Event>> {
        let mut descriptor = PollFd {
            fd: self.file.as_raw_fd(),
            events: POLLIN,
            revents: 0,
        };
        let timeout_ms = duration_to_poll_timeout(timeout);
        loop {
            // SAFETY: `descriptor` is a valid pollfd for the live inotify fd;
            // nfds is one and the kernel only writes `revents`.
            let ready = unsafe { poll(&mut descriptor, 1, timeout_ms) };
            if ready < 0 {
                if errno() == EINTR {
                    continue;
                }
                return Err(WatchError::Io(io_errno()));
            }
            if ready == 0 {
                return Ok(Vec::new());
            }
            break;
        }

        let mut events = Vec::new();
        loop {
            let read_result = self.file.read(&mut self.buffer);
            match read_result {
                Ok(0) => break,
                Ok(size) => {
                    let bytes = self.buffer[..size].to_vec();
                    self.decode(&bytes, &mut events)?;
                }
                Err(error)
                    if error.kind() == ErrorKind::WouldBlock
                        || error.raw_os_error() == Some(EAGAIN)
                        || error.raw_os_error() == Some(EWOULDBLOCK) =>
                {
                    break;
                }
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                Err(error) => return Err(WatchError::Io(error)),
            }
        }
        Ok(events)
    }

    fn decode(&mut self, bytes: &[u8], events: &mut Vec<Event>) -> Result<()> {
        let mut offset = 0;
        while offset + 16 <= bytes.len() {
            let wd = i32::from_ne_bytes(bytes[offset..offset + 4].try_into().unwrap());
            let mask = u32::from_ne_bytes(bytes[offset + 4..offset + 8].try_into().unwrap());
            let name_len =
                u32::from_ne_bytes(bytes[offset + 12..offset + 16].try_into().unwrap()) as usize;
            let record_len = 16usize
                .checked_add(name_len)
                .ok_or(WatchError::System(EOVERFLOW))?;
            if offset + record_len > bytes.len() {
                return Err(WatchError::System(EOVERFLOW));
            }
            if mask & IN_Q_OVERFLOW != 0 {
                return Err(WatchError::System(EOVERFLOW));
            }

            let name = &bytes[offset + 16..offset + record_len];
            let name = name.split(|byte| *byte == 0).next().unwrap_or_default();
            let entry = self.entries.get(&wd).map(|entry| Entry {
                root: entry.root.clone(),
                path: entry.path.clone(),
                is_dir: entry.is_dir,
            });
            if let Some(entry) = entry {
                self.handle_event(entry, mask, name, events)?;
            }
            if mask & IN_IGNORED != 0 {
                self.remove_watch(wd, false);
            }
            offset += record_len;
        }
        if offset != bytes.len() {
            return Err(WatchError::System(EOVERFLOW));
        }
        Ok(())
    }

    fn handle_event(
        &mut self,
        entry: Entry,
        mask: u32,
        name: &[u8],
        events: &mut Vec<Event>,
    ) -> Result<()> {
        let path = if entry.is_dir && !name.is_empty() {
            let mut path = entry.path.clone();
            path.push(OsString::from_vec(name.to_vec()));
            path
        } else {
            entry.path.clone()
        };
        let is_directory = mask & IN_ISDIR != 0;
        let kind = if mask & (IN_DELETE | IN_DELETE_SELF) != 0 {
            EventKind::Removed
        } else if mask & (IN_MOVED_FROM | IN_MOVED_TO | IN_MOVE_SELF) != 0 {
            EventKind::Renamed
        } else if mask & IN_CREATE != 0 {
            EventKind::Created
        } else if mask & (IN_MODIFY | IN_ATTRIB | IN_CLOSE_WRITE) != 0 {
            EventKind::Modified
        } else {
            EventKind::Other
        };
        events.push(Event {
            path: path.clone(),
            kind,
        });

        if entry.is_dir && is_directory {
            if mask & (IN_DELETE | IN_MOVED_FROM) != 0 {
                self.remove_subtree(&entry.root, &path);
            } else if mask & (IN_CREATE | IN_MOVED_TO) != 0 {
                let recursive = self
                    .roots
                    .get(&entry.root)
                    .map(|root| root.recursive)
                    .unwrap_or(false);
                if recursive {
                    let mut watches = Vec::new();
                    self.add_directory(&entry.root, &path, &mut watches)?;
                    self.add_subdirectories(&entry.root, &path, &mut watches);
                }
            }
        }
        if entry.path == entry.root && !entry.is_dir && mask & (IN_DELETE_SELF | IN_MOVE_SELF) != 0
        {
            self.remove_subtree(&entry.root, &entry.path);
        } else if entry.path == entry.root
            && entry.is_dir
            && mask & (IN_DELETE_SELF | IN_MOVE_SELF) != 0
        {
            self.remove_subtree(&entry.root, &entry.path);
        }
        Ok(())
    }

    fn add_directory(&mut self, root: &Path, path: &Path, watches: &mut Vec<RawFd>) -> Result<()> {
        self.add_watch(root, path, true, watches)
    }

    fn add_subdirectories(&mut self, root: &Path, path: &Path, watches: &mut Vec<RawFd>) {
        let Ok(entries) = std::fs::read_dir(path) else {
            return;
        };
        for entry in entries.flatten() {
            let child = entry.path();
            if child.is_dir() && self.add_directory(root, &child, watches).is_ok() {
                self.add_subdirectories(root, &child, watches);
            }
        }
    }

    #[expect(
        unsafe_code,
        reason = "register a path with the native Linux inotify descriptor"
    )]
    fn add_watch(
        &mut self,
        root: &Path,
        path: &Path,
        is_dir: bool,
        watches: &mut Vec<RawFd>,
    ) -> Result<()> {
        let c_path = CString::new(path.as_os_str().as_bytes()).map_err(|_| {
            WatchError::Io(std::io::Error::new(
                ErrorKind::InvalidInput,
                "watch path contains an embedded NUL",
            ))
        })?;
        // SAFETY: `self.file` is a live inotify descriptor and `c_path` is a
        // NUL-terminated path valid for the duration of this call.
        let wd = unsafe { inotify_add_watch(self.file.as_raw_fd(), c_path.as_ptr(), WATCH_MASK) };
        if wd < 0 {
            return Err(WatchError::Io(io_errno()));
        }
        self.entries.insert(
            wd,
            Entry {
                root: root.to_path_buf(),
                path: path.to_path_buf(),
                is_dir,
            },
        );
        watches.push(wd);
        if let Some(root_state) = self.roots.get_mut(root) {
            root_state.watches.push(wd);
        }
        Ok(())
    }

    fn remove_subtree(&mut self, root: &Path, path: &Path) {
        let descriptors: Vec<_> = self
            .entries
            .iter()
            .filter_map(|(wd, entry)| {
                (entry.root == root && entry.path.starts_with(path)).then_some(*wd)
            })
            .collect();
        for wd in descriptors {
            self.remove_watch(wd, true);
        }
    }

    #[expect(
        unsafe_code,
        reason = "remove a path from the native Linux inotify descriptor"
    )]
    fn remove_watch(&mut self, wd: RawFd, remove_from_kernel: bool) {
        if remove_from_kernel {
            // SAFETY: `wd` came from this inotify instance. A failed removal
            // is harmless because the kernel will emit IN_IGNORED eventually.
            unsafe {
                let _ = inotify_rm_watch(self.file.as_raw_fd(), wd);
            }
        }
        self.entries.remove(&wd);
        for root in self.roots.values_mut() {
            root.watches.retain(|candidate| *candidate != wd);
        }
    }
}

fn duration_to_poll_timeout(timeout: Duration) -> c_int {
    let millis = timeout.as_millis();
    if millis == 0 && !timeout.is_zero() {
        1
    } else {
        millis.min(c_int::MAX as u128) as c_int
    }
}
