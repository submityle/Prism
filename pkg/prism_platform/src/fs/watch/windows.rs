//! Native Windows file watcher backed by `ReadDirectoryChangesW`.
//!
//! A recursive directory is represented by one overlapped directory handle
//! with `bWatchSubtree=TRUE`, so directories created after registration are
//! covered by the same kernel subscription. A file watch uses its parent
//! directory and filters notifications to the requested file name.
#![expect(
    unsafe_code,
    reason = "ReadDirectoryChangesW and its overlapped Win32 handle APIs are the native Windows filesystem watcher"
)]

use core::ffi::c_void;
use core::time::Duration;
use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::io::ErrorKind;
use std::os::windows::ffi::{OsStrExt as _, OsStringExt as _};
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::{Event, EventKind, Result, WatchError};

type Handle = *mut c_void;

#[repr(C)]
struct Overlapped {
    internal: usize,
    internal_high: usize,
    offset: u32,
    offset_high: u32,
    event: Handle,
}

#[expect(
    unsafe_code,
    reason = "Win32 directory notification and wait functions are declared without an external crate"
)]
unsafe extern "system" {
    fn CreateFileW(
        filename: *const u16,
        desired_access: u32,
        share_mode: u32,
        security_attributes: *mut c_void,
        creation_disposition: u32,
        flags_and_attributes: u32,
        template_file: Handle,
    ) -> Handle;
    fn CreateEventW(
        security_attributes: *mut c_void,
        manual_reset: i32,
        initial_state: i32,
        name: *const u16,
    ) -> Handle;
    fn ReadDirectoryChangesW(
        directory: Handle,
        buffer: *mut c_void,
        buffer_length: u32,
        watch_subtree: i32,
        notify_filter: u32,
        bytes_returned: *mut u32,
        overlapped: *mut Overlapped,
        completion_routine: *mut c_void,
    ) -> i32;
    fn GetOverlappedResult(
        file: Handle,
        overlapped: *mut Overlapped,
        bytes_transferred: *mut u32,
        wait: i32,
    ) -> i32;
    fn WaitForMultipleObjects(
        count: u32,
        handles: *const Handle,
        wait_all: i32,
        milliseconds: u32,
    ) -> u32;
    fn WaitForSingleObject(handle: Handle, milliseconds: u32) -> u32;
    fn ResetEvent(event: Handle) -> i32;
    fn CancelIoEx(file: Handle, overlapped: *mut Overlapped) -> i32;
    fn CloseHandle(object: Handle) -> i32;
    fn GetLastError() -> u32;
}

const INVALID_HANDLE_VALUE: Handle = (-1isize) as Handle;

const FILE_LIST_DIRECTORY: u32 = 0x0001;
const FILE_SHARE_READ: u32 = 0x0001;
const FILE_SHARE_WRITE: u32 = 0x0002;
const FILE_SHARE_DELETE: u32 = 0x0004;
const OPEN_EXISTING: u32 = 3;
const FILE_FLAG_OVERLAPPED: u32 = 0x4000_0000;
const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;

const FILE_NOTIFY_CHANGE_FILE_NAME: u32 = 0x0000_0001;
const FILE_NOTIFY_CHANGE_DIR_NAME: u32 = 0x0000_0002;
const FILE_NOTIFY_CHANGE_ATTRIBUTES: u32 = 0x0000_0004;
const FILE_NOTIFY_CHANGE_SIZE: u32 = 0x0000_0008;
const FILE_NOTIFY_CHANGE_LAST_WRITE: u32 = 0x0000_0010;
const FILE_NOTIFY_CHANGE_CREATION: u32 = 0x0000_0040;
const NOTIFY_FILTER: u32 = FILE_NOTIFY_CHANGE_FILE_NAME
    | FILE_NOTIFY_CHANGE_DIR_NAME
    | FILE_NOTIFY_CHANGE_ATTRIBUTES
    | FILE_NOTIFY_CHANGE_SIZE
    | FILE_NOTIFY_CHANGE_LAST_WRITE
    | FILE_NOTIFY_CHANGE_CREATION;

const FILE_ACTION_ADDED: u32 = 0x0000_0001;
const FILE_ACTION_REMOVED: u32 = 0x0000_0002;
const FILE_ACTION_MODIFIED: u32 = 0x0000_0003;
const FILE_ACTION_RENAMED_OLD_NAME: u32 = 0x0000_0004;
const FILE_ACTION_RENAMED_NEW_NAME: u32 = 0x0000_0005;

const ERROR_IO_PENDING: u32 = 997;
const ERROR_OPERATION_ABORTED: u32 = 995;
const ERROR_NOTIFY_ENUM_DIR: u32 = 1022;
const WAIT_OBJECT_0: u32 = 0;
const WAIT_TIMEOUT: u32 = 258;
const WAIT_FAILED: u32 = 0xffff_ffff;
const MAXIMUM_WAIT_OBJECTS: usize = 64;
const READ_BUFFER_SIZE: usize = 64 * 1024;

struct OwnedHandle(Handle);

impl OwnedHandle {
    fn new(handle: Handle) -> Result<Self> {
        if handle.is_null() || handle == INVALID_HANDLE_VALUE {
            Err(WatchError::System(last_error() as i32))
        } else {
            Ok(Self(handle))
        }
    }

    fn raw(&self) -> Handle {
        self.0
    }
}

impl Drop for OwnedHandle {
    fn drop(&mut self) {
        // SAFETY: this handle was returned by a successful Win32 creation call
        // and is owned by this wrapper until it is dropped exactly once.
        unsafe {
            CloseHandle(self.0);
        }
    }
}

struct WatchEntry {
    root: PathBuf,
    directory: PathBuf,
    file_name: Option<OsString>,
    recursive: bool,
    directory_handle: OwnedHandle,
    event_handle: OwnedHandle,
    overlapped: Box<Overlapped>,
    buffer: Vec<u8>,
    pending: bool,
}

impl WatchEntry {
    fn new(path: &Path, is_directory: bool, recursive: bool) -> Result<Self> {
        let (directory, file_name, recursive) = if is_directory {
            (path.to_path_buf(), None, recursive)
        } else {
            let directory = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf();
            let file_name = path.file_name().ok_or_else(|| {
                WatchError::Io(std::io::Error::new(
                    ErrorKind::InvalidInput,
                    "file watch path has no file name",
                ))
            })?;
            (directory, Some(file_name.to_os_string()), false)
        };

        let directory_handle = open_directory(&directory)?;
        let event_handle = create_event()?;
        let event = event_handle.raw();
        Ok(Self {
            root: path.to_path_buf(),
            directory,
            file_name,
            recursive,
            directory_handle,
            event_handle,
            overlapped: Box::new(Overlapped {
                internal: 0,
                internal_high: 0,
                offset: 0,
                offset_high: 0,
                event,
            }),
            buffer: vec![0; READ_BUFFER_SIZE],
            pending: false,
        })
    }

    fn begin_read(&mut self) -> Result<()> {
        // SAFETY: the event belongs to this entry and is not concurrently
        // waited on while this single-threaded watcher rearms the request.
        let reset = unsafe { ResetEvent(self.event_handle.raw()) };
        if reset == 0 {
            return Err(WatchError::System(last_error() as i32));
        }
        *self.overlapped = Overlapped {
            internal: 0,
            internal_high: 0,
            offset: 0,
            offset_high: 0,
            event: self.event_handle.raw(),
        };

        // SAFETY: the directory handle, buffer, and boxed OVERLAPPED remain
        // owned and immovable until the asynchronous request completes or is
        // cancelled by Drop. `bytes_returned` is null as required for an
        // overlapped ReadDirectoryChangesW call.
        let ok = unsafe {
            ReadDirectoryChangesW(
                self.directory_handle.raw(),
                self.buffer.as_mut_ptr().cast::<c_void>(),
                self.buffer.len() as u32,
                i32::from(self.recursive),
                NOTIFY_FILTER,
                core::ptr::null_mut(),
                core::ptr::from_mut(self.overlapped.as_mut()),
                core::ptr::null_mut(),
            )
        };
        if ok == 0 {
            let code = last_error() as u32;
            if code != ERROR_IO_PENDING {
                return Err(WatchError::System(code as i32));
            }
        }
        self.pending = true;
        Ok(())
    }

    fn complete(&mut self, events: &mut Vec<Event>) -> Result<()> {
        let mut transferred = 0;
        // SAFETY: the request was submitted with this live directory handle
        // and this stable OVERLAPPED allocation, and the event was signaled.
        let ok = unsafe {
            GetOverlappedResult(
                self.directory_handle.raw(),
                core::ptr::from_mut(self.overlapped.as_mut()),
                core::ptr::from_mut(&mut transferred),
                0,
            )
        };
        self.pending = false;
        if ok == 0 {
            let code = last_error();
            if code == ERROR_OPERATION_ABORTED {
                return Ok(());
            }
            if code == ERROR_NOTIFY_ENUM_DIR as u32 {
                let _ = self.begin_read();
                return Err(WatchError::System(ERROR_NOTIFY_ENUM_DIR as i32));
            }
            return Err(WatchError::System(code as i32));
        }

        if transferred == 0 {
            let _ = self.begin_read();
            return Err(WatchError::System(ERROR_NOTIFY_ENUM_DIR as i32));
        }

        let parsed = parse_notifications(&self.buffer[..transferred as usize]);
        let rearm = self.begin_read();
        let notifications = match parsed {
            Ok(notifications) => notifications,
            Err(error) => {
                let _ = rearm;
                return Err(error);
            }
        };
        rearm?;

        for notification in notifications {
            let name = OsString::from_wide(&notification.name);
            if let Some(file_name) = &self.file_name {
                if !same_file_name(file_name, &name) {
                    continue;
                }
            }
            let path = if self.file_name.is_some() {
                self.root.clone()
            } else {
                self.directory.join(name)
            };
            events.push(Event {
                path,
                kind: notification.kind,
            });
        }
        Ok(())
    }
}

impl Drop for WatchEntry {
    fn drop(&mut self) {
        if self.pending {
            // SAFETY: the handle and OVERLAPPED belong to this entry; dropping
            // the entry cancels the outstanding request before both are closed.
            unsafe {
                CancelIoEx(
                    self.directory_handle.raw(),
                    core::ptr::from_mut(self.overlapped.as_mut()),
                );
            }
        }
    }
}

/// Native Windows watcher using one overlapped subscription per watched root.
pub(super) struct WindowsWatcher {
    entries: HashMap<PathBuf, WatchEntry>,
}

impl WindowsWatcher {
    pub(super) fn new() -> Result<Self> {
        Ok(Self {
            entries: HashMap::new(),
        })
    }

    pub(super) fn watch(&mut self, path: &Path, recursive: bool) -> Result<()> {
        if self.entries.contains_key(path) {
            self.unwatch(path)?;
        }
        let metadata = std::fs::symlink_metadata(path).map_err(WatchError::Io)?;
        let mut entry = WatchEntry::new(path, metadata.is_dir(), recursive)?;
        entry.begin_read()?;
        self.entries.insert(path.to_path_buf(), entry);
        Ok(())
    }

    pub(super) fn unwatch(&mut self, path: &Path) -> Result<()> {
        if self.entries.remove(path).is_some() {
            Ok(())
        } else {
            Err(WatchError::NotWatched)
        }
    }

    pub(super) fn poll(&mut self, timeout: Duration) -> Result<Vec<Event>> {
        let handles: Vec<_> = self
            .entries
            .values()
            .filter(|entry| entry.pending)
            .map(|entry| entry.event_handle.raw())
            .collect();
        if handles.is_empty() || !wait_for_any(&handles, timeout)? {
            return Ok(Vec::new());
        }

        let keys: Vec<_> = self.entries.keys().cloned().collect();
        let mut events = Vec::new();
        for key in keys {
            let Some(entry) = self.entries.get_mut(&key) else {
                continue;
            };
            // A single wait wakes the poll; draining every already-signaled
            // entry coalesces a burst into one result just like the Unix backends.
            let signaled = unsafe { WaitForSingleObject(entry.event_handle.raw(), 0) };
            if signaled == WAIT_OBJECT_0 {
                entry.complete(&mut events)?;
            } else if signaled == WAIT_FAILED {
                return Err(WatchError::System(last_error() as i32));
            }
        }
        Ok(events)
    }
}

#[derive(Debug)]
struct Notification {
    kind: EventKind,
    name: Vec<u16>,
}

fn parse_notifications(bytes: &[u8]) -> Result<Vec<Notification>> {
    let mut notifications = Vec::new();
    let mut offset = 0usize;
    loop {
        let header_end = offset
            .checked_add(12)
            .ok_or(WatchError::System(ERROR_NOTIFY_ENUM_DIR as i32))?;
        if header_end > bytes.len() {
            return Err(WatchError::System(ERROR_NOTIFY_ENUM_DIR as i32));
        }
        let next = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        let action = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap());
        let name_len =
            u32::from_le_bytes(bytes[offset + 8..offset + 12].try_into().unwrap()) as usize;
        if name_len % 2 != 0 {
            return Err(WatchError::System(ERROR_NOTIFY_ENUM_DIR as i32));
        }
        let name_end = header_end
            .checked_add(name_len)
            .ok_or(WatchError::System(ERROR_NOTIFY_ENUM_DIR as i32))?;
        if name_end > bytes.len() {
            return Err(WatchError::System(ERROR_NOTIFY_ENUM_DIR as i32));
        }
        let mut name = Vec::with_capacity(name_len / 2);
        for pair in bytes[header_end..name_end].chunks_exact(2) {
            name.push(u16::from_le_bytes([pair[0], pair[1]]));
        }
        let kind = match action {
            FILE_ACTION_ADDED => EventKind::Created,
            FILE_ACTION_REMOVED => EventKind::Removed,
            FILE_ACTION_MODIFIED => EventKind::Modified,
            FILE_ACTION_RENAMED_OLD_NAME | FILE_ACTION_RENAMED_NEW_NAME => EventKind::Renamed,
            _ => EventKind::Other,
        };
        notifications.push(Notification { kind, name });
        if next == 0 {
            return Ok(notifications);
        }
        if next < 12 || !next.is_multiple_of(4) {
            return Err(WatchError::System(ERROR_NOTIFY_ENUM_DIR as i32));
        }
        let next_offset = offset
            .checked_add(next)
            .ok_or(WatchError::System(ERROR_NOTIFY_ENUM_DIR as i32))?;
        if next_offset > bytes.len() {
            return Err(WatchError::System(ERROR_NOTIFY_ENUM_DIR as i32));
        }
        offset = next_offset;
    }
}

fn same_file_name(left: &OsStr, right: &OsStr) -> bool {
    if left == right {
        return true;
    }
    left.to_string_lossy()
        .eq_ignore_ascii_case(&right.to_string_lossy())
}

fn open_directory(path: &Path) -> Result<OwnedHandle> {
    let wide = wide_path(path)?;
    // SAFETY: `wide` is a NUL-terminated UTF-16 path and all other pointers
    // are null/default values accepted by CreateFileW.
    let handle = unsafe {
        CreateFileW(
            wide.as_ptr(),
            FILE_LIST_DIRECTORY,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            core::ptr::null_mut(),
            OPEN_EXISTING,
            FILE_FLAG_OVERLAPPED | FILE_FLAG_BACKUP_SEMANTICS,
            core::ptr::null_mut(),
        )
    };
    OwnedHandle::new(handle)
}

fn create_event() -> Result<OwnedHandle> {
    // SAFETY: a manual-reset unnamed event needs no security descriptor/name.
    let handle = unsafe { CreateEventW(core::ptr::null_mut(), 1, 0, core::ptr::null()) };
    OwnedHandle::new(handle)
}

fn wide_path(path: &Path) -> Result<Vec<u16>> {
    let mut wide: Vec<_> = path.as_os_str().encode_wide().collect();
    if wide.contains(&0) {
        return Err(WatchError::Io(std::io::Error::new(
            ErrorKind::InvalidInput,
            "watch path contains an embedded NUL",
        )));
    }
    wide.push(0);
    Ok(wide)
}

fn wait_for_any(handles: &[Handle], timeout: Duration) -> Result<bool> {
    if handles.len() <= MAXIMUM_WAIT_OBJECTS {
        let result = unsafe {
            WaitForMultipleObjects(
                handles.len() as u32,
                handles.as_ptr(),
                0,
                duration_to_wait_timeout(timeout),
            )
        };
        return match result {
            WAIT_TIMEOUT => Ok(false),
            WAIT_FAILED => Err(WatchError::System(last_error() as i32)),
            _ => Ok(true),
        };
    }

    let deadline = Instant::now().checked_add(timeout);
    loop {
        for chunk in handles.chunks(MAXIMUM_WAIT_OBJECTS) {
            let result =
                unsafe { WaitForMultipleObjects(chunk.len() as u32, chunk.as_ptr(), 0, 0) };
            if result >= WAIT_OBJECT_0 && result < WAIT_OBJECT_0 + chunk.len() as u32 {
                return Ok(true);
            }
            match result {
                WAIT_TIMEOUT => {}
                WAIT_FAILED => return Err(WatchError::System(last_error() as i32)),
                _ => {}
            }
        }
        let Some(deadline) = deadline else {
            return Ok(false);
        };
        let now = Instant::now();
        if now >= deadline {
            return Ok(false);
        }
        std::thread::sleep(Duration::from_millis(1).min(deadline - now));
    }
}

fn duration_to_wait_timeout(timeout: Duration) -> u32 {
    let millis = timeout.as_millis();
    if millis == 0 && !timeout.is_zero() {
        1
    } else {
        millis.min((u32::MAX - 1) as u128) as u32
    }
}

fn last_error() -> u32 {
    // SAFETY: GetLastError reads the current thread's Win32 error code.
    unsafe { GetLastError() }
}

#[cfg(test)]
mod tests {
    use super::{parse_notifications, EventKind, ERROR_NOTIFY_ENUM_DIR, FILE_ACTION_ADDED};

    #[test]
    fn parses_file_notify_information_records() {
        let name: Vec<u16> = "nested\\asset.txt".encode_utf16().collect();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&FILE_ACTION_ADDED.to_le_bytes());
        bytes.extend_from_slice(&((name.len() * 2) as u32).to_le_bytes());
        for unit in name {
            bytes.extend_from_slice(&unit.to_le_bytes());
        }

        let parsed = parse_notifications(&bytes).expect("valid FILE_NOTIFY_INFORMATION");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].kind, EventKind::Created);
        assert_eq!(
            parsed[0].name,
            "nested\\asset.txt".encode_utf16().collect::<Vec<_>>()
        );
    }

    #[test]
    fn rejects_truncated_file_notify_information_as_overflow() {
        let error =
            parse_notifications(&[0, 0, 0, 0, 1, 0, 0, 0]).expect_err("truncated notification");
        assert!(
            matches!(error, super::WatchError::System(code) if code == ERROR_NOTIFY_ENUM_DIR as i32)
        );
    }
}
