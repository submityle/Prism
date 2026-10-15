//! Windows asynchronous-I/O backend backed by a native IOCP port.
//!
//! The public facade carries Windows CRT file descriptors, so this backend
//! resolves them to their underlying Win32 handles with `_get_osfhandle`.
//! Callers must open those handles with `FILE_FLAG_OVERLAPPED`; the backend
//! does not own or duplicate the caller's file handle.
#![expect(
    unsafe_code,
    reason = "IOCP requires dependency-free Win32 and CRT overlapped-I/O FFI"
)]

use core::ffi::c_void;
use core::ptr;
use core::time::Duration;

use super::{AioError, Completion, ReadOp, Result, SubmitOutcome, WriteOp};

type Handle = *mut c_void;

const INVALID_HANDLE_VALUE: Handle = (-1isize) as Handle;
const ERROR_IO_PENDING: u32 = 997;
const ERROR_INVALID_PARAMETER: u32 = 87;
const WAIT_TIMEOUT: u32 = 258;
const INFINITE: u32 = u32::MAX;
const QUEUE_CAPACITY: usize = 256;

#[repr(C)]
struct Overlapped {
    internal: usize,
    internal_high: usize,
    offset: u32,
    offset_high: u32,
    event: Handle,
}

#[repr(C)]
struct OverlappedEntry {
    completion_key: usize,
    overlapped: *mut Overlapped,
    internal: usize,
    bytes_transferred: u32,
}

#[expect(
    unsafe_code,
    reason = "Win32 IOCP, overlapped file, cancellation, and CRT descriptor APIs"
)]
unsafe extern "system" {
    fn CreateIoCompletionPort(
        file: Handle,
        existing_port: Handle,
        completion_key: usize,
        concurrent_threads: u32,
    ) -> Handle;
    fn GetQueuedCompletionStatusEx(
        port: Handle,
        entries: *mut OverlappedEntry,
        count: u32,
        removed: *mut u32,
        milliseconds: u32,
        alertable: i32,
    ) -> i32;
    fn ReadFile(
        file: Handle,
        buffer: *mut c_void,
        bytes_to_read: u32,
        bytes_read: *mut u32,
        overlapped: *mut Overlapped,
    ) -> i32;
    fn WriteFile(
        file: Handle,
        buffer: *const c_void,
        bytes_to_write: u32,
        bytes_written: *mut u32,
        overlapped: *mut Overlapped,
    ) -> i32;
    fn GetOverlappedResult(
        file: Handle,
        overlapped: *mut Overlapped,
        bytes_transferred: *mut u32,
        wait: i32,
    ) -> i32;
    fn PostQueuedCompletionStatus(
        port: Handle,
        bytes_transferred: u32,
        completion_key: usize,
        overlapped: *mut Overlapped,
    ) -> i32;
    fn CancelIoEx(file: Handle, overlapped: *mut Overlapped) -> i32;
    fn CloseHandle(object: Handle) -> i32;
    fn GetLastError() -> u32;
}

#[expect(
    unsafe_code,
    reason = "_get_osfhandle is the CRT bridge from the public RawFd API"
)]
unsafe extern "C" {
    fn _get_osfhandle(fd: i32) -> isize;
}

/// Windows has a real IOCP completion backend in this build.
pub(super) const SUPPORTED: bool = true;

struct InFlight {
    handle: Handle,
    overlapped: Box<Overlapped>,
    user_data: u64,
    immediate_error: Option<AioError>,
}

pub(super) struct Queue {
    port: Handle,
    pending: Vec<InFlight>,
}

fn last_error() -> AioError {
    // SAFETY: GetLastError only reads this thread's last-error slot.
    AioError::Io(unsafe { GetLastError() } as i32)
}

fn duration_to_millis(timeout: Duration) -> u32 {
    let millis = timeout.as_millis();
    let rounded = millis.saturating_add(u128::from(timeout.subsec_nanos() != 0));
    rounded.min(u128::from(INFINITE - 1)) as u32
}

fn raw_handle(fd: i32) -> Result<Handle> {
    if fd < 0 {
        return Err(AioError::InvalidArgument);
    }
    // SAFETY: `_get_osfhandle` only reads the CRT descriptor table.
    let handle = unsafe { _get_osfhandle(fd) } as Handle;
    if handle.is_null() || handle == INVALID_HANDLE_VALUE {
        Err(AioError::InvalidArgument)
    } else {
        Ok(handle)
    }
}

fn validate_len(len: usize) -> Result<u32> {
    u32::try_from(len).map_err(|_| AioError::InvalidArgument)
}

fn make_overlapped(offset: u64) -> Overlapped {
    Overlapped {
        internal: 0,
        internal_high: 0,
        offset: offset as u32,
        offset_high: (offset >> 32) as u32,
        event: ptr::null_mut(),
    }
}

impl Queue {
    pub(super) fn new() -> Result<Self> {
        // SAFETY: INVALID_HANDLE_VALUE requests a new private completion port;
        // no file handle is associated until a validated submit call.
        let port = unsafe { CreateIoCompletionPort(INVALID_HANDLE_VALUE, ptr::null_mut(), 0, 0) };
        if port.is_null() || port == INVALID_HANDLE_VALUE {
            Err(last_error())
        } else {
            Ok(Self {
                port,
                pending: Vec::new(),
            })
        }
    }

    pub(super) unsafe fn submit(&mut self, ops: &[ReadOp]) -> Result<SubmitOutcome> {
        // SAFETY: the facade's caller guarantees every fd and destination
        // buffer remains valid until this request's completion is reaped.
        unsafe {
            self.submit_descriptors(ops.len(), |i| {
                let op = &ops[i];
                Ok(Descriptor {
                    fd: op.fd,
                    buffer: op.buf.cast_const().cast::<c_void>(),
                    len: validate_len(op.len)?,
                    offset: op.offset,
                    user_data: op.user_data,
                    write: false,
                })
            })
        }
    }

    pub(super) unsafe fn submit_write(&mut self, ops: &[WriteOp]) -> Result<SubmitOutcome> {
        // SAFETY: the facade's caller guarantees every fd and source buffer
        // remains valid and unmodified until this request's completion.
        unsafe {
            self.submit_descriptors(ops.len(), |i| {
                let op = &ops[i];
                Ok(Descriptor {
                    fd: op.fd,
                    buffer: op.buf.cast::<c_void>(),
                    len: validate_len(op.len)?,
                    offset: op.offset,
                    user_data: op.user_data,
                    write: true,
                })
            })
        }
    }

    unsafe fn submit_descriptors(
        &mut self,
        count: usize,
        get: impl Fn(usize) -> Result<Descriptor>,
    ) -> Result<SubmitOutcome> {
        if count == 0 {
            return Ok(SubmitOutcome::default());
        }

        let mut descriptors = Vec::with_capacity(count);
        for i in 0..count {
            let descriptor = get(i)?;
            // Resolve all descriptors before touching the queue, so malformed
            // tails cannot produce a partial batch with hidden validation work.
            let _ = raw_handle(descriptor.fd)?;
            descriptors.push(descriptor);
        }

        let available = QUEUE_CAPACITY.saturating_sub(self.pending.len());
        let candidate = available.min(descriptors.len());
        if candidate == 0 {
            return Ok(SubmitOutcome {
                accepted: 0,
                rejected: descriptors.iter().map(|d| d.user_data).collect(),
            });
        }

        let mut accepted = 0;
        for descriptor in descriptors.iter().take(candidate) {
            match unsafe { self.start(*descriptor) } {
                Ok(request) => {
                    self.pending.push(request);
                    accepted += 1;
                }
                Err(error) => {
                    // Previously accepted requests remain owned by this queue,
                    // matching the existing POSIX backend's partial-submit
                    // behavior. The error is returned when no request was
                    // accepted; after a partial accept the caller can still
                    // reap the accepted prefix via wait().
                    if accepted == 0 {
                        return Err(error);
                    }
                    let rejected = descriptors
                        .iter()
                        .skip(accepted)
                        .map(|d| d.user_data)
                        .collect();
                    return Ok(SubmitOutcome { accepted, rejected });
                }
            }
        }

        Ok(SubmitOutcome {
            accepted,
            rejected: descriptors
                .iter()
                .skip(candidate)
                .map(|d| d.user_data)
                .collect(),
        })
    }

    #[expect(
        unsafe_code,
        reason = "associate a caller-owned overlapped handle with the private IOCP"
    )]
    unsafe fn start(&self, descriptor: Descriptor) -> Result<InFlight> {
        let handle = raw_handle(descriptor.fd)?;
        // SAFETY: `handle` is a live caller-owned handle and `self.port` is the
        // private completion port created by this queue.
        let associated = unsafe { CreateIoCompletionPort(handle, self.port, 0, 0) };
        if associated.is_null() || associated == INVALID_HANDLE_VALUE {
            let code = unsafe { GetLastError() };
            return unsafe { self.synthetic_error(handle, descriptor, code as i32) };
        }

        let mut overlapped = Box::new(make_overlapped(descriptor.offset));
        let overlapped_ptr = ptr::from_mut(overlapped.as_mut());
        // SAFETY: the boxed OVERLAPPED and caller buffer remain stable for the
        // complete operation; `FILE_FLAG_OVERLAPPED` is required by contract.
        let ok = if descriptor.write {
            unsafe {
                WriteFile(
                    handle,
                    descriptor.buffer,
                    descriptor.len,
                    ptr::null_mut(),
                    overlapped_ptr,
                )
            }
        } else {
            unsafe {
                ReadFile(
                    handle,
                    descriptor.buffer.cast_mut(),
                    descriptor.len,
                    ptr::null_mut(),
                    overlapped_ptr,
                )
            }
        };
        if ok == 0 {
            // SAFETY: GetLastError reads the failure generated by ReadFile or
            // WriteFile on this same thread.
            let code = unsafe { GetLastError() };
            if code != ERROR_IO_PENDING {
                return unsafe { self.synthetic_error(handle, descriptor, code as i32) };
            }
        }

        Ok(InFlight {
            handle,
            overlapped,
            user_data: descriptor.user_data,
            immediate_error: None,
        })
    }

    #[expect(
        unsafe_code,
        reason = "turn an immediate Win32 submit failure into an IOCP error completion"
    )]
    unsafe fn synthetic_error(
        &self,
        handle: Handle,
        descriptor: Descriptor,
        code: i32,
    ) -> Result<InFlight> {
        let mut overlapped = Box::new(make_overlapped(descriptor.offset));
        // SAFETY: the boxed OVERLAPPED is owned by the returned request and is
        // stable until its synthetic completion is reaped.
        let posted = unsafe {
            PostQueuedCompletionStatus(self.port, 0, 0, ptr::from_mut(overlapped.as_mut()))
        };
        if posted == 0 {
            return Err(last_error());
        }
        Ok(InFlight {
            handle,
            overlapped,
            user_data: descriptor.user_data,
            immediate_error: Some(AioError::Io(code)),
        })
    }

    pub(super) fn wait(
        &mut self,
        max: usize,
        timeout: Option<Duration>,
    ) -> Result<Vec<Completion>> {
        if max == 0 || self.pending.is_empty() {
            return Ok(Vec::new());
        }

        let count = max.min(QUEUE_CAPACITY);
        let mut entries: Vec<OverlappedEntry> = (0..count)
            .map(|_| OverlappedEntry {
                completion_key: 0,
                overlapped: ptr::null_mut(),
                internal: 0,
                bytes_transferred: 0,
            })
            .collect();
        let mut removed = 0_u32;
        let milliseconds = timeout.map_or(INFINITE, duration_to_millis);
        // SAFETY: the port and entry array are live and owned by this queue;
        // the call writes at most `count` entries.
        let ok = unsafe {
            GetQueuedCompletionStatusEx(
                self.port,
                entries.as_mut_ptr(),
                count as u32,
                &mut removed,
                milliseconds,
                0,
            )
        };
        if ok == 0 {
            let code = unsafe { GetLastError() };
            return if code == WAIT_TIMEOUT {
                Ok(Vec::new())
            } else {
                Err(AioError::Io(code as i32))
            };
        }

        let mut completions = Vec::with_capacity(removed as usize);
        for entry in entries.into_iter().take(removed as usize) {
            if entry.overlapped.is_null() {
                return Err(AioError::Io(ERROR_INVALID_PARAMETER as i32));
            }
            let Some(position) = self.pending.iter().position(|request| {
                ptr::eq(request.overlapped.as_ref(), unsafe { &*entry.overlapped })
            }) else {
                return Err(AioError::Io(ERROR_INVALID_PARAMETER as i32));
            };
            let request = self.pending.swap_remove(position);
            let mut bytes = entry.bytes_transferred;
            // SAFETY: this request's completion packet has been dequeued, so
            // its stable OVERLAPPED and caller buffer are still valid and may
            // now be inspected exactly once.
            let result = if let Some(error) = request.immediate_error {
                Err(error)
            } else if unsafe {
                GetOverlappedResult(
                    request.handle,
                    ptr::from_ref(request.overlapped.as_ref()).cast_mut(),
                    &mut bytes,
                    0,
                )
            } != 0
            {
                Ok(bytes as usize)
            } else {
                Err(last_error())
            };
            completions.push(Completion {
                user_data: request.user_data,
                result,
            });
        }
        Ok(completions)
    }

    pub(super) fn pending(&self) -> usize {
        self.pending.len()
    }
}

#[derive(Clone, Copy)]
struct Descriptor {
    fd: i32,
    buffer: *const c_void,
    len: u32,
    offset: u64,
    user_data: u64,
    write: bool,
}

impl Drop for Queue {
    fn drop(&mut self) {
        for request in &mut self.pending {
            // SAFETY: each handle/OVERLAPPED pair belongs to a still-live
            // request. ERROR_NOT_FOUND means it already completed or was
            // already canceled; the drain below still waits for its packet.
            let _ =
                unsafe { CancelIoEx(request.handle, ptr::from_mut(request.overlapped.as_mut())) };
        }

        while !self.pending.is_empty() {
            let _ = self.wait(self.pending.len(), None);
            if self.pending.is_empty() {
                break;
            }

            // A private IOCP should not fail an infinite wait. If a host API
            // reports an unexpected error, synchronously wait each request to
            // guarantee the OVERLAPPED allocation is no longer in use before
            // the queue releases it.
            while let Some(mut request) = self.pending.pop() {
                let mut bytes = 0;
                // SAFETY: wait=TRUE makes this request terminal before its
                // control block can be dropped.
                let _ = unsafe {
                    GetOverlappedResult(
                        request.handle,
                        ptr::from_mut(request.overlapped.as_mut()),
                        &mut bytes,
                        1,
                    )
                };
            }
        }

        // SAFETY: the port was created by this queue and is closed exactly once
        // after every request has been canceled and drained.
        unsafe {
            CloseHandle(self.port);
        }
    }
}

#[cfg(test)]
mod layout_tests {
    use super::{Overlapped, OverlappedEntry};
    use core::mem::{align_of, size_of};

    #[test]
    fn win32_overlapped_abi_layout_matches_pointer_width() {
        // OVERLAPPED is 20 bytes on Win32 and 32 bytes on Win64. The
        // OVERLAPPED_ENTRY returned by GetQueuedCompletionStatusEx is 16
        // bytes on Win32 and 32 bytes on Win64.
        let (overlapped_size, entry_size) = if size_of::<usize>() == 8 {
            (32, 32)
        } else {
            (20, 16)
        };
        assert_eq!(size_of::<Overlapped>(), overlapped_size);
        assert_eq!(size_of::<OverlappedEntry>(), entry_size);
        assert_eq!(align_of::<Overlapped>(), align_of::<usize>());
        assert_eq!(align_of::<OverlappedEntry>(), align_of::<usize>());
    }
}
