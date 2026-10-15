//! Linux async-I/O backend built directly on the `io_uring` syscalls.
//!
//! The queue owns the ring mappings and every submitted request's internal
//! token until its CQE has been reaped.  Drop submits `IORING_OP_ASYNC_CANCEL`
//! requests where possible, then drains the original request CQEs before the
//! mappings and queue fd are released; this is the point at which the facade's
//! caller-owned buffers are safe to release.
#![expect(
    unsafe_code,
    reason = "io_uring requires Linux syscall, mmap, and volatile ring FFI"
)]

use core::ffi::{c_int, c_long, c_void};
use core::mem::size_of;
use core::ptr::{self, null_mut};
use core::sync::atomic::{fence, Ordering};
use core::time::Duration;
use std::os::fd::RawFd;
use std::time::Instant;

use super::{AioError, Completion, IoPriority, ReadOp, Result, SubmitOutcome, WriteOp};

pub(super) const SUPPORTED: bool = true;

const SYS_IO_URING_SETUP: c_long = 425;
const SYS_IO_URING_ENTER: c_long = 426;
const IORING_OFF_SQ_RING: i64 = 0;
const IORING_OFF_CQ_RING: i64 = 0x8000_0000;
const IORING_OFF_SQES: i64 = 0x1_0000_0000;
const IORING_FEAT_SINGLE_MMAP: u32 = 1 << 0;
const IORING_ENTER_GETEVENTS: u32 = 1;
const IORING_OP_READ: u8 = 0;
const IORING_OP_WRITE: u8 = 1;
const IORING_OP_ASYNC_CANCEL: u8 = 14;
const MAP_SHARED: c_int = 0x01;
const PROT_READ: c_int = 0x01;
const PROT_WRITE: c_int = 0x02;
const MAP_FAILED: *mut c_void = (-1_isize) as *mut c_void;
const CANCEL_ID_BASE: u64 = 1_u64 << 63;
const RING_ENTRIES: u32 = 256;

const EINTR: i32 = 4;
const ENOSYS: i32 = 38;
const EINVAL: i32 = 22;
const EPERM: i32 = 1;

#[repr(C)]
#[derive(Clone, Copy)]
struct IoSqringOffsets {
    head: u32,
    tail: u32,
    ring_mask: u32,
    ring_entries: u32,
    flags: u32,
    dropped: u32,
    array: u32,
    resv1: u32,
    resv2: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct IoCqringOffsets {
    head: u32,
    tail: u32,
    ring_mask: u32,
    ring_entries: u32,
    overflow: u32,
    cqes: u32,
    resv1: u64,
    resv2: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct IoUringParams {
    sq_entries: u32,
    cq_entries: u32,
    flags: u32,
    sq_thread_cpu: u32,
    sq_thread_idle: u32,
    features: u32,
    wq_fd: u32,
    resv: [u32; 3],
    sq_off: IoSqringOffsets,
    cq_off: IoCqringOffsets,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct IoUringSqe {
    opcode: u8,
    flags: u8,
    ioprio: u16,
    fd: i32,
    off: u64,
    addr: u64,
    len: u32,
    rw_flags: u32,
    user_data: u64,
    buf_index: u16,
    personality: u16,
    splice_fd_in: i32,
    pad2: [u64; 2],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct IoUringCqe {
    user_data: u64,
    res: i32,
    flags: u32,
}

unsafe extern "C" {
    fn syscall(number: c_long, ...) -> c_long;
    fn __errno_location() -> *mut c_int;
    fn mmap(
        addr: *mut c_void,
        length: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, length: usize) -> c_int;
    fn close(fd: c_int) -> c_int;
}

fn errno() -> i32 {
    // SAFETY: libc returns a valid pointer to this thread's errno slot.
    unsafe { *__errno_location() }
}

fn syscall_error() -> i32 {
    let code = errno();
    if code == 0 {
        EINVAL
    } else {
        code
    }
}

fn map_setup_error(code: i32) -> AioError {
    match code {
        ENOSYS | EINVAL | EPERM => AioError::Unsupported,
        other => AioError::Io(other),
    }
}

fn req_ioprio(priority: IoPriority) -> u16 {
    match priority {
        IoPriority::High => 0,
        IoPriority::Normal => 1,
        IoPriority::Low => 2,
    }
}

fn to_u32_len(len: usize) -> Result<u32> {
    u32::try_from(len).map_err(|_| AioError::InvalidArgument)
}

fn to_i64_offset(offset: u64) -> Result<u64> {
    if offset > i64::MAX as u64 {
        Err(AioError::InvalidArgument)
    } else {
        Ok(offset)
    }
}

#[derive(Clone, Copy)]
struct RawDesc {
    fd: RawFd,
    offset: u64,
    addr: u64,
    len: u32,
    ioprio: u16,
    user_data: u64,
}

struct InFlight {
    id: u64,
    user_data: u64,
}

pub(super) struct Queue {
    fd: RawFd,
    params: IoUringParams,
    sq_ring: *mut u8,
    cq_ring: *mut u8,
    sq_ring_len: usize,
    cq_ring_len: usize,
    sqes: *mut IoUringSqe,
    sqes_len: usize,
    pending: Vec<InFlight>,
    cancel_ids: Vec<u64>,
    next_id: u64,
}

impl Queue {
    pub(super) fn new() -> Result<Self> {
        let mut params = IoUringParams {
            sq_entries: 0,
            cq_entries: 0,
            flags: 0,
            sq_thread_cpu: 0,
            sq_thread_idle: 0,
            features: 0,
            wq_fd: 0,
            resv: [0; 3],
            sq_off: IoSqringOffsets {
                head: 0,
                tail: 0,
                ring_mask: 0,
                ring_entries: 0,
                flags: 0,
                dropped: 0,
                array: 0,
                resv1: 0,
                resv2: 0,
            },
            cq_off: IoCqringOffsets {
                head: 0,
                tail: 0,
                ring_mask: 0,
                ring_entries: 0,
                overflow: 0,
                cqes: 0,
                resv1: 0,
                resv2: 0,
            },
        };
        // SAFETY: syscall number and pointer match the Linux io_uring ABI.
        let fd = unsafe { syscall(SYS_IO_URING_SETUP, RING_ENTRIES, &mut params) };
        if fd < 0 {
            return Err(map_setup_error(syscall_error()));
        }
        let fd = i32::try_from(fd).map_err(|_| AioError::Io(EINVAL))?;

        let sq_ring_len = ring_len(
            usize::try_from(params.sq_off.array).map_err(|_| AioError::Io(EINVAL))?,
            params.sq_entries,
            size_of::<u32>(),
        )?;
        let cq_ring_len = ring_len(
            usize::try_from(params.cq_off.cqes).map_err(|_| AioError::Io(EINVAL))?,
            params.cq_entries,
            size_of::<IoUringCqe>(),
        )?;
        let sqes_len = usize::try_from(params.sq_entries)
            .ok()
            .and_then(|n| n.checked_mul(size_of::<IoUringSqe>()))
            .ok_or(AioError::Io(EINVAL))?;

        let single = params.features & IORING_FEAT_SINGLE_MMAP != 0;
        let shared_len = sq_ring_len.max(cq_ring_len);
        let sq_ring = map_ring(
            fd,
            if single { shared_len } else { sq_ring_len },
            IORING_OFF_SQ_RING,
        );
        let sq_ring = match sq_ring {
            Ok(ptr) => ptr,
            Err(error) => {
                // SAFETY: fd was returned by io_uring_setup and is owned here.
                unsafe { close(fd) };
                return Err(error);
            }
        };
        let cq_ring = if single {
            sq_ring
        } else {
            match map_ring(fd, cq_ring_len, IORING_OFF_CQ_RING) {
                Ok(ptr) => ptr,
                Err(error) => {
                    // SAFETY: mapping and fd are owned by this constructor.
                    unsafe {
                        munmap(sq_ring.cast(), sq_ring_len);
                        close(fd);
                    }
                    return Err(error);
                }
            }
        };
        let sqes = match map_ring(fd, sqes_len, IORING_OFF_SQES) {
            Ok(ptr) => ptr.cast::<IoUringSqe>(),
            Err(error) => {
                // SAFETY: mappings and fd are owned by this constructor.
                unsafe {
                    if !single {
                        munmap(cq_ring.cast(), cq_ring_len);
                    }
                    munmap(
                        sq_ring.cast(),
                        if single { shared_len } else { sq_ring_len },
                    );
                    close(fd);
                }
                return Err(error);
            }
        };

        Ok(Self {
            fd,
            params,
            sq_ring,
            cq_ring,
            sq_ring_len: if single { shared_len } else { sq_ring_len },
            cq_ring_len: if single { 0 } else { cq_ring_len },
            sqes,
            sqes_len,
            pending: Vec::new(),
            cancel_ids: Vec::new(),
            next_id: 1,
        })
    }

    pub(super) unsafe fn submit(&mut self, ops: &[ReadOp]) -> Result<SubmitOutcome> {
        // SAFETY: the facade's caller guarantees every read buffer and fd live
        // until the corresponding CQE is reaped.
        unsafe {
            self.submit_descriptors(
                ops.len(),
                |i| {
                    let op = &ops[i];
                    Ok(RawDesc {
                        fd: op.fd,
                        offset: to_i64_offset(op.offset)?,
                        addr: op.buf as usize as u64,
                        len: to_u32_len(op.len)?,
                        ioprio: req_ioprio(op.priority),
                        user_data: op.user_data,
                    })
                },
                IORING_OP_READ,
            )
        }
    }

    pub(super) unsafe fn submit_write(&mut self, ops: &[WriteOp]) -> Result<SubmitOutcome> {
        // SAFETY: the facade's caller guarantees every write source and fd live
        // and remain unchanged until the corresponding CQE is reaped.
        unsafe {
            self.submit_descriptors(
                ops.len(),
                |i| {
                    let op = &ops[i];
                    Ok(RawDesc {
                        fd: op.fd,
                        offset: to_i64_offset(op.offset)?,
                        addr: op.buf as usize as u64,
                        len: to_u32_len(op.len)?,
                        ioprio: req_ioprio(op.priority),
                        user_data: op.user_data,
                    })
                },
                IORING_OP_WRITE,
            )
        }
    }

    unsafe fn submit_descriptors(
        &mut self,
        count: usize,
        get: impl Fn(usize) -> Result<RawDesc>,
        opcode: u8,
    ) -> Result<SubmitOutcome> {
        if count == 0 {
            return Ok(SubmitOutcome::default());
        }
        let mut descs = Vec::with_capacity(count);
        for i in 0..count {
            let desc = get(i)?;
            if desc.fd < 0 || desc.offset > i64::MAX as u64 {
                return Err(AioError::InvalidArgument);
            }
            descs.push(desc);
        }

        let old_tail = self.sq_tail();
        let available = self.sq_available(old_tail);
        let candidate = available.min(count as u32) as usize;
        if candidate == 0 {
            return Ok(SubmitOutcome {
                accepted: 0,
                rejected: descs.iter().map(|desc| desc.user_data).collect(),
            });
        }

        let mut ids = Vec::with_capacity(candidate);
        for desc in descs.iter().take(candidate).copied() {
            let id = self.alloc_id();
            ids.push(id);
            let tail = self.sq_tail();
            let index = tail & self.sq_ring_mask();
            let sqe = self.sqes.wrapping_add(index as usize);
            // SAFETY: index is bounded by the kernel-provided SQ ring mask and
            // the SQE mapping covers every ring entry.
            unsafe {
                ptr::write(
                    sqe,
                    IoUringSqe {
                        opcode,
                        flags: 0,
                        ioprio: desc.ioprio,
                        fd: desc.fd,
                        off: desc.offset,
                        addr: desc.addr,
                        len: desc.len,
                        rw_flags: 0,
                        user_data: id,
                        buf_index: 0,
                        personality: 0,
                        splice_fd_in: 0,
                        pad2: [0; 2],
                    },
                );
                ptr::write_volatile(self.sq_array().add(index as usize), index);
                fence(Ordering::Release);
                ptr::write_volatile(self.sq_tail_ptr(), tail.wrapping_add(1));
            }
        }

        // SAFETY: the SQ ring and SQEs are valid for this queue's lifetime.
        let submitted = match unsafe { self.enter(candidate as u32, 0, 0) } {
            Ok(submitted) => submitted as usize,
            Err(error) => {
                // No CQE can refer to these requests because the enter syscall
                // returned an error. Remove the unsubmitted SQ suffix before
                // exposing the error to the caller.
                unsafe { ptr::write_volatile(self.sq_tail_ptr(), old_tail) };
                return Err(error);
            }
        };
        let accepted = submitted.min(candidate);
        if accepted < candidate {
            // The kernel consumes SQEs in order. Roll the unsubmitted suffix
            // back so those slots can be reused on the next call.
            // SAFETY: no other thread accesses this queue; the suffix was not
            // submitted to the kernel.
            unsafe {
                ptr::write_volatile(self.sq_tail_ptr(), old_tail.wrapping_add(accepted as u32));
            }
        }

        self.pending.extend(
            ids.iter()
                .take(accepted)
                .zip(descs.iter())
                .map(|(&id, desc)| InFlight {
                    id,
                    user_data: desc.user_data,
                }),
        );

        let rejected: Vec<u64> = descs
            .iter()
            .skip(accepted)
            .map(|desc| desc.user_data)
            .collect();
        Ok(SubmitOutcome { accepted, rejected })
    }

    pub(super) fn wait(
        &mut self,
        max: usize,
        timeout: Option<Duration>,
    ) -> Result<Vec<Completion>> {
        if max == 0 || self.pending.is_empty() {
            return Ok(Vec::new());
        }
        let ready = self.reap(max);
        if !ready.is_empty() {
            return Ok(ready);
        }

        match timeout {
            None => loop {
                // SAFETY: waiting on this queue's own fd and CQ ring.
                match unsafe { self.enter(0, 1, IORING_ENTER_GETEVENTS) } {
                    Ok(_) => {
                        let ready = self.reap(max);
                        if !ready.is_empty() {
                            return Ok(ready);
                        }
                    }
                    Err(AioError::Io(EINTR)) => continue,
                    Err(error) => return Err(error),
                }
            },
            Some(timeout) => {
                // A regular io_uring fd is not a portable readiness
                // notification fd. Polling it can miss CQE arrival, so use
                // the ring's non-blocking GETEVENTS path and a bounded sleep
                // between probes. The no-timeout path remains the efficient
                // blocking completion path.
                let deadline = Instant::now().checked_add(timeout);
                loop {
                    match unsafe { self.enter(0, 0, IORING_ENTER_GETEVENTS) } {
                        Ok(_) => {
                            let ready = self.reap(max);
                            if !ready.is_empty() {
                                return Ok(ready);
                            }
                        }
                        Err(AioError::Io(EINTR)) => continue,
                        Err(error) => return Err(error),
                    }

                    let now = Instant::now();
                    let Some(deadline) = deadline else {
                        return Ok(Vec::new());
                    };
                    let Some(remaining) = deadline.checked_duration_since(now) else {
                        return Ok(Vec::new());
                    };
                    std::thread::sleep(remaining.min(Duration::from_millis(1)));
                }
            }
        }
    }

    pub(super) fn pending(&self) -> usize {
        self.pending.len()
    }

    fn sq_ring_mask(&self) -> u32 {
        // SAFETY: kernel initialized this field during setup.
        unsafe {
            ptr::read_volatile(
                self.sq_ring
                    .add(self.params.sq_off.ring_mask as usize)
                    .cast(),
            )
        }
    }

    fn sq_tail(&self) -> u32 {
        // SAFETY: kernel-owned ring field is read atomically enough for the
        // single queue owner; volatile prevents compiler caching.
        unsafe { ptr::read_volatile(self.sq_ring.add(self.params.sq_off.tail as usize).cast()) }
    }

    fn sq_head(&self) -> u32 {
        // SAFETY: see `sq_tail`.
        unsafe { ptr::read_volatile(self.sq_ring.add(self.params.sq_off.head as usize).cast()) }
    }

    fn sq_available(&self, tail: u32) -> u32 {
        self.params
            .sq_entries
            .saturating_sub(tail.wrapping_sub(self.sq_head()))
    }

    fn sq_tail_ptr(&self) -> *mut u32 {
        self.sq_ring
            .wrapping_add(self.params.sq_off.tail as usize)
            .cast()
    }

    fn sq_array(&self) -> *mut u32 {
        self.sq_ring
            .wrapping_add(self.params.sq_off.array as usize)
            .cast()
    }

    fn cq_head_ptr(&self) -> *mut u32 {
        self.cq_ring
            .wrapping_add(self.params.cq_off.head as usize)
            .cast()
    }

    fn cq_tail(&self) -> u32 {
        // SAFETY: kernel-owned ring field; see `sq_tail`.
        unsafe { ptr::read_volatile(self.cq_ring.add(self.params.cq_off.tail as usize).cast()) }
    }

    fn alloc_id(&mut self) -> u64 {
        let mut id = self.next_id;
        loop {
            self.next_id = self.next_id.wrapping_add(1).max(1);
            if id < CANCEL_ID_BASE
                && !self.pending.iter().any(|entry| entry.id == id)
                && !self.cancel_ids.contains(&id)
            {
                return id;
            }
            id = self.next_id;
        }
    }

    fn alloc_cancel_id(&mut self) -> u64 {
        CANCEL_ID_BASE | self.cancel_ids.len() as u64 + 1
    }

    fn reap(&mut self, max: usize) -> Vec<Completion> {
        let mut head = unsafe { ptr::read_volatile(self.cq_head_ptr()) };
        let tail = self.cq_tail();
        // `cq_off.ring_mask` is a byte offset in the mmap, not the mask value.
        let mask = unsafe {
            ptr::read_volatile(
                self.cq_ring
                    .add(self.params.cq_off.ring_mask as usize)
                    .cast(),
            )
        };
        let mut out = Vec::new();
        while head != tail && out.len() < max {
            let index = head & mask;
            let cqe_ptr = self
                .cq_ring
                .wrapping_add(self.params.cq_off.cqes as usize)
                .cast::<IoUringCqe>()
                .wrapping_add(index as usize);
            // SAFETY: CQE index is bounded by the kernel-provided mask.
            let cqe = unsafe { ptr::read_volatile(cqe_ptr) };
            if let Some(position) = self
                .pending
                .iter()
                .position(|entry| entry.id == cqe.user_data)
            {
                let entry = self.pending.swap_remove(position);
                let result = if cqe.res >= 0 {
                    Ok(cqe.res as usize)
                } else {
                    Err(AioError::Io(cqe.res.saturating_neg()))
                };
                out.push(Completion {
                    user_data: entry.user_data,
                    result,
                });
            } else if let Some(position) =
                self.cancel_ids.iter().position(|&id| id == cqe.user_data)
            {
                self.cancel_ids.swap_remove(position);
            }
            head = head.wrapping_add(1);
        }
        fence(Ordering::Release);
        // SAFETY: this queue is the sole CQ consumer.
        unsafe { ptr::write_volatile(self.cq_head_ptr(), head) };
        out
    }

    unsafe fn enter(&self, to_submit: u32, min_complete: u32, flags: u32) -> Result<i32> {
        // SAFETY: arguments match the Linux `io_uring_enter` syscall ABI.
        let rc = unsafe {
            syscall(
                SYS_IO_URING_ENTER,
                self.fd,
                to_submit,
                min_complete,
                flags,
                ptr::null::<c_void>(),
                0_u32,
            )
        };
        if rc < 0 {
            Err(AioError::Io(syscall_error()))
        } else {
            Ok(i32::try_from(rc).unwrap_or(i32::MAX))
        }
    }

    fn queue_cancel(&mut self, id: u64) -> Result<bool> {
        let tail = self.sq_tail();
        if self.sq_available(tail) == 0 {
            return Ok(false);
        }
        let cancel_id = self.alloc_cancel_id();
        let index = tail & self.sq_ring_mask();
        let sqe = self.sqes.wrapping_add(index as usize);
        // SAFETY: the SQE lies in the queue-owned mapping and the ring is not
        // concurrently accessed.
        unsafe {
            ptr::write(
                sqe,
                IoUringSqe {
                    opcode: IORING_OP_ASYNC_CANCEL,
                    flags: 0,
                    ioprio: 0,
                    fd: -1,
                    off: 0,
                    addr: id,
                    len: 0,
                    rw_flags: 0,
                    user_data: cancel_id,
                    buf_index: 0,
                    personality: 0,
                    splice_fd_in: 0,
                    pad2: [0; 2],
                },
            );
            ptr::write_volatile(self.sq_array().add(index as usize), index);
            fence(Ordering::Release);
            ptr::write_volatile(self.sq_tail_ptr(), tail.wrapping_add(1));
        }
        // SAFETY: one cancel SQE was just prepared in this queue.
        let submitted = unsafe { self.enter(1, 0, 0) }?;
        if submitted == 1 {
            self.cancel_ids.push(cancel_id);
            Ok(true)
        } else {
            // SAFETY: the kernel did not consume the zero/partial suffix.
            unsafe { ptr::write_volatile(self.sq_tail_ptr(), tail) };
            Ok(false)
        }
    }

    fn cancel_and_drain(&mut self) {
        let ids: Vec<u64> = self.pending.iter().map(|entry| entry.id).collect();
        for id in ids {
            let _ = self.queue_cancel(id);
        }
        while !self.pending.is_empty() {
            if self.reap(self.pending.len()).is_empty() {
                // SAFETY: drop owns the queue and deliberately waits for every
                // outstanding request to settle before freeing caller buffers'
                // kernel-visible mappings.
                let _ = unsafe { self.enter(0, 1, IORING_ENTER_GETEVENTS) };
            }
        }
        while !self.cancel_ids.is_empty() {
            let _ = unsafe { self.enter(0, 1, IORING_ENTER_GETEVENTS) };
            let _ = self.reap(usize::MAX);
        }
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        self.cancel_and_drain();
        // SAFETY: all requests have completed and these mappings are owned by
        // this queue. SINGLE_MMAP uses one shared mapping for both rings.
        unsafe {
            munmap(self.sqes.cast(), self.sqes_len);
            if self.cq_ring_len != 0 {
                munmap(self.cq_ring.cast(), self.cq_ring_len);
            }
            munmap(self.sq_ring.cast(), self.sq_ring_len);
            close(self.fd);
        }
    }
}

fn ring_len(offset: usize, entries: u32, element_size: usize) -> Result<usize> {
    offset
        .checked_add(
            usize::try_from(entries)
                .ok()
                .and_then(|count| count.checked_mul(element_size))
                .ok_or(AioError::InvalidArgument)?,
        )
        .ok_or(AioError::InvalidArgument)
}

fn map_ring(fd: RawFd, len: usize, offset: i64) -> Result<*mut u8> {
    if len == 0 {
        return Err(AioError::InvalidArgument);
    }
    // SAFETY: mmap arguments follow the Linux io_uring shared-ring ABI.
    let ptr = unsafe {
        mmap(
            null_mut(),
            len,
            PROT_READ | PROT_WRITE,
            MAP_SHARED,
            fd,
            offset,
        )
    };
    if ptr == MAP_FAILED {
        Err(AioError::Io(syscall_error()))
    } else {
        Ok(ptr.cast())
    }
}
