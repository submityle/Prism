//! Apple (macOS/iOS) & BSD async-I/O backend built on POSIX AIO.
//!
//! `lio_listio(LIO_NOWAIT, ...)` submits a whole batch of reads in one syscall;
//! `aio_suspend` blocks until at least one of the in-flight control blocks
//! completes; `aio_error`/`aio_return` reap each finished request (every
//! completed control block is `aio_return`-ed exactly once to release its
//! kernel resources); `aio_cancel` + a drain loop in [`Drop`] guarantee the
//! kernel is no longer touching a caller buffer before the control block is
//! freed.
//!
//! Each FFI call is a dependency-free `extern` declaration resolved against the
//! C runtime that `std` already links (no `libc` crate), mirroring
//! [`crate::vm`] and [`crate::fs::watch`].
#![expect(
    unsafe_code,
    reason = "batch async I/O requires lio_listio/aio_suspend/aio_error/aio_return/aio_cancel via dependency-free C FFI"
)]

use core::ffi::{c_int, c_long, c_void};
use core::time::Duration;

use super::{AioError, Completion, IoPriority, ReadOp, Result};

/// This build has a real POSIX AIO backend.
pub(super) const SUPPORTED: bool = true;

// POSIX AIO constants (stable across the Darwin / BSD ABIs we target).
const LIO_NOWAIT: c_int = 1;
const LIO_READ: c_int = 1;
const EINTR: c_int = 4;
const EAGAIN: c_int = 35;
const EINPROGRESS: c_int = 36;
/// Maximum control blocks a single `lio_listio` accepts; larger batches chunk.
const AIO_LISTIO_MAX: usize = 16;

/// `struct sigevent` (64-bit Darwin/BSD ABI). We never request notification —
/// the all-zero value means `SIGEV_NONE`, so completions are discovered by
/// polling `aio_error` instead of via signals or threads.
#[repr(C)]
#[derive(Clone, Copy)]
struct Sigevent {
    sigev_notify: c_int,
    sigev_signo: c_int,
    sigev_value: *mut c_void,
    sigev_notify_function: *mut c_void,
    sigev_notify_attributes: *mut c_void,
}

/// `struct aiocb` (64-bit Darwin/BSD ABI). `repr(C)` reproduces the C padding
/// (size 80, align 8).
#[repr(C)]
struct Aiocb {
    aio_fildes: c_int,
    aio_offset: i64,
    aio_buf: *mut c_void,
    aio_nbytes: usize,
    aio_reqprio: c_int,
    aio_sigevent: Sigevent,
    aio_lio_opcode: c_int,
}

/// `struct timespec` (64-bit ABI).
#[repr(C)]
struct Timespec {
    tv_sec: i64,
    tv_nsec: c_long,
}

unsafe extern "C" {
    fn lio_listio(mode: c_int, list: *const *mut Aiocb, nent: c_int, sig: *mut Sigevent) -> c_int;
    fn aio_error(cb: *const Aiocb) -> c_int;
    fn aio_return(cb: *mut Aiocb) -> isize;
    fn aio_suspend(list: *const *const Aiocb, nent: c_int, timeout: *const Timespec) -> c_int;
    fn aio_cancel(fd: c_int, cb: *mut Aiocb) -> c_int;
    fn __error() -> *mut c_int;
}

/// Read this thread's `errno`.
fn errno() -> c_int {
    // SAFETY: `__error` returns a valid pointer to this thread's `errno`, which
    // we only read.
    unsafe { *__error() }
}

/// Map an [`IoPriority`] to an `aio_reqprio` delta (0 = highest). Best-effort;
/// macOS largely ignores this field.
fn reqprio(p: IoPriority) -> c_int {
    match p {
        IoPriority::High => 0,
        IoPriority::Normal => 1,
        IoPriority::Low => 2,
    }
}

/// One in-flight control block plus the caller's correlation token. The
/// `Box` keeps the `Aiocb`'s address stable while the kernel holds a pointer to
/// it, even if the owning `Vec` reallocates.
struct InFlight {
    cb: Box<Aiocb>,
    user_data: u64,
}

/// POSIX-AIO-backed submission/completion queue.
pub(super) struct Queue {
    inflight: Vec<InFlight>,
}

impl Queue {
    pub(super) fn new() -> Result<Self> {
        Ok(Self {
            inflight: Vec::new(),
        })
    }

    /// Batch-submit reads via `lio_listio`.
    ///
    /// # Safety
    /// Each [`ReadOp`]'s `fd` and `buf`/`len` region must stay valid until the
    /// matching completion is reaped (see the facade's module-level contract).
    pub(super) unsafe fn submit(&mut self, ops: &[ReadOp]) -> Result<usize> {
        if ops.is_empty() {
            return Ok(0);
        }
        for op in ops {
            if op.fd < 0 {
                return Err(AioError::InvalidArgument);
            }
        }

        let mut accepted = 0usize;
        for chunk in ops.chunks(AIO_LISTIO_MAX) {
            let start = self.inflight.len();
            for op in chunk {
                // SAFETY: `Aiocb` is plain data (ints + raw pointers); an
                // all-zero bit pattern is a valid, inert control block (null
                // pointers, `SIGEV_NONE`).
                let mut cb: Box<Aiocb> = Box::new(unsafe { core::mem::zeroed() });
                cb.aio_fildes = op.fd;
                cb.aio_offset = op.offset as i64;
                cb.aio_buf = op.buf.cast::<c_void>();
                cb.aio_nbytes = op.len;
                cb.aio_reqprio = reqprio(op.priority);
                cb.aio_lio_opcode = LIO_READ;
                self.inflight.push(InFlight {
                    cb,
                    user_data: op.user_data,
                });
            }

            let list: Vec<*mut Aiocb> = self.inflight[start..]
                .iter_mut()
                .map(|f| core::ptr::from_mut::<Aiocb>(f.cb.as_mut()))
                .collect();

            // SAFETY: `list` points to `chunk.len()` live, boxed control blocks
            // owned by `self.inflight`; the heap allocations outlive the kernel's
            // use of them (reaped in `wait`, drained in `Drop`). A null `sig`
            // requests no notification (we poll), matching the zeroed sigevent.
            let rc = unsafe {
                lio_listio(
                    LIO_NOWAIT,
                    list.as_ptr(),
                    list.len() as c_int,
                    core::ptr::null_mut(),
                )
            };

            if rc == 0 {
                accepted += chunk.len();
                continue;
            }

            // Partial / failed enqueue: POSIX guarantees each control block's
            // per-request status is readable via `aio_error`. Keep the ones the
            // kernel actually accepted (in progress or already done); reap and
            // drop the rest so no stale control block lingers.
            let e = errno();
            let mut idx = start;
            while idx < self.inflight.len() {
                // SAFETY: `cb` is a live control block owned by `self.inflight`.
                let status = unsafe {
                    aio_error(core::ptr::from_ref::<Aiocb>(self.inflight[idx].cb.as_ref()))
                };
                if status == EINPROGRESS || status == 0 {
                    accepted += 1;
                    idx += 1;
                } else {
                    let mut f = self.inflight.remove(idx);
                    // SAFETY: a completed/failed control block must be returned
                    // exactly once to release kernel resources; its result is
                    // irrelevant here.
                    unsafe {
                        let _ = aio_return(core::ptr::from_mut::<Aiocb>(f.cb.as_mut()));
                    }
                }
            }

            // A hard, non-back-pressure error with nothing accepted is a real
            // failure worth surfacing; EAGAIN is honest back-pressure.
            if accepted == 0 && e != EAGAIN {
                return Err(AioError::Io(e));
            }
            // Back-pressure: stop submitting further chunks; caller retries tail.
            break;
        }

        Ok(accepted)
    }

    pub(super) fn wait(
        &mut self,
        max: usize,
        timeout: Option<Duration>,
    ) -> Result<Vec<Completion>> {
        if self.inflight.is_empty() || max == 0 {
            return Ok(Vec::new());
        }

        let list: Vec<*const Aiocb> = self
            .inflight
            .iter()
            .map(|f| core::ptr::from_ref::<Aiocb>(f.cb.as_ref()))
            .collect();
        let ts = timeout.map(|d| Timespec {
            tv_sec: d.as_secs() as i64,
            tv_nsec: c_long::from(d.subsec_nanos()),
        });
        let ts_ptr = ts
            .as_ref()
            .map_or(core::ptr::null(), core::ptr::from_ref::<Timespec>);

        // SAFETY: `list` points to live control blocks owned by `self.inflight`;
        // `ts_ptr` is either null or a valid local `Timespec`.
        let rc = unsafe { aio_suspend(list.as_ptr(), list.len() as c_int, ts_ptr) };
        if rc != 0 {
            let e = errno();
            // Timeout or interrupted: nothing reaped, but not a fatal error.
            if e == EAGAIN || e == EINTR {
                return Ok(Vec::new());
            }
            return Err(AioError::Io(e));
        }

        let mut out = Vec::new();
        let mut idx = 0;
        while idx < self.inflight.len() && out.len() < max {
            // SAFETY: `cb` is a live control block owned by `self.inflight`.
            let status =
                unsafe { aio_error(core::ptr::from_ref::<Aiocb>(self.inflight[idx].cb.as_ref())) };
            if status == EINPROGRESS {
                idx += 1;
                continue;
            }

            let mut f = self.inflight.remove(idx);
            // SAFETY: a completed control block is returned exactly once here to
            // collect its transferred byte count and release kernel resources.
            let ret = unsafe { aio_return(core::ptr::from_mut::<Aiocb>(f.cb.as_mut())) };
            let result = if status == 0 {
                Ok(ret.max(0) as usize)
            } else {
                Err(AioError::Io(status))
            };
            out.push(Completion {
                user_data: f.user_data,
                result,
            });
            // `remove` shifted the next element into `idx`; do not advance.
        }

        Ok(out)
    }

    pub(super) fn pending(&self) -> usize {
        self.inflight.len()
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        for f in &mut self.inflight {
            let fd = f.cb.aio_fildes;
            // SAFETY: `cb` is a live control block owned here; cancelling an
            // in-flight (or already-done) request is always safe.
            unsafe {
                let _ = aio_cancel(fd, core::ptr::from_mut::<Aiocb>(f.cb.as_mut()));
            }
            // Block until the request is no longer in progress so the kernel has
            // stopped touching the caller's buffer, then reap exactly once.
            loop {
                // SAFETY: `cb` is a live control block owned here.
                let status =
                    unsafe { aio_error(core::ptr::from_ref::<Aiocb>(f.cb.as_ref())) };
                if status != EINPROGRESS {
                    break;
                }
                let one = [core::ptr::from_ref::<Aiocb>(f.cb.as_ref())];
                // SAFETY: single-element list of a live control block; null
                // timeout blocks until it completes.
                unsafe {
                    let _ = aio_suspend(one.as_ptr(), 1, core::ptr::null());
                }
            }
            // SAFETY: the request has settled; return it once to free resources.
            unsafe {
                let _ = aio_return(core::ptr::from_mut::<Aiocb>(f.cb.as_mut()));
            }
        }
    }
}
