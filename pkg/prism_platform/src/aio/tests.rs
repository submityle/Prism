//! Tests for the §24.1 async-I/O facade.
//!
//! On Apple, Linux, and Windows the real native backend is exercised against a temp file:
//! the test writes a known byte pattern, batch-submits reads at distinct
//! offsets into distinct buffers, reaps completions, and asserts each buffer
//! matches the source bytes and each `user_data` round-trips. Other platforms
//! only assert honest `Unsupported` degradation.

#![expect(
    unsafe_code,
    reason = "exercising the unsafe batch-submit API against a real temp file"
)]

use super::{AioError, AioQueue, IoPriority, ReadOp, WriteOp};

#[cfg(any(target_vendor = "apple", target_os = "linux"))]
mod posix {
    use super::*;
    use core::time::Duration;
    use std::io::Write;
    use std::os::fd::AsRawFd;

    /// Deterministic source byte at a given file offset.
    fn pattern(off: usize) -> u8 {
        (off % 251) as u8
    }

    struct TempFile {
        path: std::path::PathBuf,
        file: std::fs::File,
    }

    impl TempFile {
        fn with_pattern(len: usize) -> Self {
            let mut path = std::env::temp_dir();
            let unique = format!(
                "prism_aio_{}_{}.bin",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            );
            path.push(unique);
            let bytes: Vec<u8> = (0..len).map(pattern).collect();
            {
                let mut f = std::fs::File::create(&path).expect("create temp file");
                f.write_all(&bytes).expect("write pattern");
                f.sync_all().expect("sync");
            }
            let file = std::fs::File::open(&path).expect("open temp file");
            Self { path, file }
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    /// A writable temp file (created empty, opened read+write) for write tests.
    struct WritableTempFile {
        path: std::path::PathBuf,
        file: std::fs::File,
    }

    impl WritableTempFile {
        fn new() -> Self {
            let mut path = std::env::temp_dir();
            let unique = format!(
                "prism_aio_w_{}_{}.bin",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            );
            path.push(unique);
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&path)
                .expect("create writable temp file");
            Self { path, file }
        }
    }

    impl Drop for WritableTempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    #[test]
    fn supported_on_native_backend() {
        assert!(AioQueue::supported());
        assert!(AioQueue::new().is_ok());
    }

    #[test]
    fn batch_reads_round_trip() {
        const PAGE: usize = 4096;
        const N: usize = 3;
        let tmp = TempFile::with_pattern(PAGE * N);
        let fd = tmp.file.as_raw_fd();

        let mut queue = AioQueue::new().expect("queue");
        let mut buffers: Vec<Vec<u8>> = (0..N).map(|_| vec![0u8; PAGE]).collect();

        let ops: Vec<ReadOp> = buffers
            .iter_mut()
            .enumerate()
            .map(|(i, buf)| ReadOp {
                fd,
                offset: (i * PAGE) as u64,
                buf: buf.as_mut_ptr(),
                len: PAGE,
                user_data: 0xA000 + i as u64,
                priority: IoPriority::Normal,
            })
            .collect();

        // SAFETY: `tmp.file` (hence `fd`) and `buffers` outlive every reaped
        // completion below; buffers are not touched until reaped.
        let outcome = unsafe { queue.submit(&ops) }.expect("submit");
        assert_eq!(outcome.accepted, N);
        assert!(
            outcome.all_accepted(),
            "full accept leaves nothing rejected"
        );
        assert_eq!(queue.pending(), N);

        let mut seen = alloc::collections::BTreeMap::new();
        while seen.len() < N {
            let batch = queue.wait(N, Some(Duration::from_secs(5))).expect("wait");
            assert!(
                !batch.is_empty(),
                "wait returned no completions before timeout"
            );
            for c in batch {
                let bytes = c.result.expect("read ok");
                assert_eq!(bytes, PAGE);
                seen.insert(c.user_data, bytes);
            }
        }

        assert_eq!(queue.pending(), 0);
        for (i, buf) in buffers.iter().enumerate() {
            assert!(seen.contains_key(&(0xA000 + i as u64)));
            for (j, &b) in buf.iter().enumerate() {
                assert_eq!(b, pattern(i * PAGE + j), "mismatch at page {i} byte {j}");
            }
        }
    }

    #[test]
    fn chunks_beyond_listio_max_with_backpressure() {
        // More ops than the kernel's AIO_LISTIO_MAX / AIO_MAX ceiling (16 on
        // macOS). `submit` chunks internally and honestly reports how many it
        // got in flight; the caller drains completions and resubmits the tail.
        const N: usize = 40;
        const LEN: usize = 64;
        let tmp = TempFile::with_pattern(LEN * N);
        let fd = tmp.file.as_raw_fd();

        let mut queue = AioQueue::new().expect("queue");
        let mut buffers: Vec<Vec<u8>> = (0..N).map(|_| vec![0u8; LEN]).collect();
        let ops: Vec<ReadOp> = buffers
            .iter_mut()
            .enumerate()
            .map(|(i, buf)| ReadOp {
                fd,
                offset: (i * LEN) as u64,
                buf: buf.as_mut_ptr(),
                len: LEN,
                user_data: i as u64,
                priority: IoPriority::Low,
            })
            .collect();

        let mut next = 0usize; // next op index not yet accepted
        let mut reaped = 0usize;
        while reaped < N {
            if next < N {
                // SAFETY: `tmp.file` and `buffers` outlive every completion
                // reaped in this loop; buffers are untouched until reaped.
                let outcome = unsafe { queue.submit(&ops[next..]) }.expect("submit");
                // macOS enqueues a prefix of the batch, so advancing by the
                // accepted count resubmits exactly the untaken tail.
                next += outcome.accepted;
            }
            if queue.pending() > 0 {
                let batch = queue.wait(N, Some(Duration::from_secs(5))).expect("wait");
                assert!(!batch.is_empty(), "no completions before timeout");
                for c in batch {
                    assert_eq!(c.result.expect("ok"), LEN);
                    reaped += 1;
                }
            }
        }
        assert_eq!(next, N);
        assert_eq!(queue.pending(), 0);

        for (i, buf) in buffers.iter().enumerate() {
            for (j, &b) in buf.iter().enumerate() {
                assert_eq!(b, pattern(i * LEN + j));
            }
        }
    }

    #[test]
    fn empty_submit_is_noop() {
        let mut queue = AioQueue::new().expect("queue");
        // SAFETY: no ops, no buffers involved.
        let outcome = unsafe { queue.submit(&[]) }.expect("submit");
        assert_eq!(outcome.accepted, 0);
        assert!(outcome.rejected.is_empty());
        assert_eq!(queue.pending(), 0);
    }

    #[test]
    fn negative_fd_is_rejected() {
        let mut queue = AioQueue::new().expect("queue");
        let op = ReadOp {
            fd: -1,
            offset: 0,
            buf: core::ptr::null_mut(),
            len: 0,
            user_data: 0,
            priority: IoPriority::Normal,
        };
        // SAFETY: validated and rejected before any buffer is dereferenced.
        let r = unsafe { queue.submit(core::slice::from_ref(&op)) };
        assert_eq!(r, Err(AioError::InvalidArgument));
    }

    #[test]
    fn wait_with_nothing_in_flight_is_empty() {
        let mut queue = AioQueue::new().expect("queue");
        let batch = queue
            .wait(8, Some(Duration::from_millis(10)))
            .expect("wait");
        assert!(batch.is_empty());
    }

    #[test]
    fn batch_writes_round_trip() {
        const PAGE: usize = 4096;
        const N: usize = 3;
        let tmp = WritableTempFile::new();
        let fd = tmp.file.as_raw_fd();

        // Distinct source buffers at distinct offsets.
        let sources: Vec<Vec<u8>> = (0..N)
            .map(|i| (0..PAGE).map(|j| pattern(i * PAGE + j)).collect())
            .collect();

        let mut queue = AioQueue::new().expect("queue");
        let ops: Vec<WriteOp> = sources
            .iter()
            .enumerate()
            .map(|(i, src)| WriteOp {
                fd,
                offset: (i * PAGE) as u64,
                buf: src.as_ptr(),
                len: PAGE,
                user_data: 0xB000 + i as u64,
                priority: IoPriority::Normal,
            })
            .collect();

        // SAFETY: `tmp.file` (hence `fd`) and `sources` outlive every reaped
        // completion below; the source buffers are not mutated until reaped.
        let outcome = unsafe { queue.submit_write(&ops) }.expect("submit_write");
        assert_eq!(outcome.accepted, N);
        assert!(
            outcome.all_accepted(),
            "full accept leaves nothing rejected"
        );

        let mut seen = alloc::collections::BTreeMap::new();
        while seen.len() < N {
            let batch = queue.wait(N, Some(Duration::from_secs(5))).expect("wait");
            assert!(
                !batch.is_empty(),
                "wait returned no completions before timeout"
            );
            for c in batch {
                let bytes = c.result.expect("write ok");
                assert_eq!(bytes, PAGE);
                seen.insert(c.user_data, bytes);
            }
        }
        assert_eq!(queue.pending(), 0);

        // Read the file back with ordinary std I/O and verify the bytes landed.
        tmp.file.sync_all().expect("sync");
        let readback = std::fs::read(&tmp.path).expect("read back");
        assert_eq!(readback.len(), PAGE * N);
        for (i, &b) in readback.iter().enumerate() {
            assert_eq!(b, pattern(i), "mismatch at byte {i}");
        }
        for i in 0..N {
            assert!(seen.contains_key(&(0xB000 + i as u64)));
        }
    }

    #[test]
    fn empty_submit_write_is_noop() {
        let mut queue = AioQueue::new().expect("queue");
        // SAFETY: no ops, no buffers involved.
        let outcome = unsafe { queue.submit_write(&[]) }.expect("submit_write");
        assert_eq!(outcome.accepted, 0);
        assert!(outcome.rejected.is_empty());
        assert_eq!(queue.pending(), 0);
    }

    #[test]
    fn negative_fd_write_is_rejected() {
        let mut queue = AioQueue::new().expect("queue");
        let op = WriteOp {
            fd: -1,
            offset: 0,
            buf: core::ptr::null(),
            len: 0,
            user_data: 0,
            priority: IoPriority::Normal,
        };
        // SAFETY: validated and rejected before any buffer is dereferenced.
        let r = unsafe { queue.submit_write(core::slice::from_ref(&op)) };
        assert_eq!(r, Err(AioError::InvalidArgument));
    }

    #[test]
    fn write_then_read_back_via_aio() {
        // Round-trip entirely through the AIO backend: write a pattern, then
        // read it back into a fresh buffer with the same queue.
        const LEN: usize = 2048;
        let tmp = WritableTempFile::new();
        let fd = tmp.file.as_raw_fd();
        let src: Vec<u8> = (0..LEN).map(pattern).collect();

        let mut queue = AioQueue::new().expect("queue");
        let wop = WriteOp {
            fd,
            offset: 0,
            buf: src.as_ptr(),
            len: LEN,
            user_data: 1,
            priority: IoPriority::High,
        };
        // SAFETY: `tmp.file` and `src` outlive the completion reaped below.
        let wrote = unsafe { queue.submit_write(core::slice::from_ref(&wop)) }.expect("w");
        assert_eq!(wrote.accepted, 1);
        loop {
            let batch = queue.wait(1, Some(Duration::from_secs(5))).expect("wait");
            if let Some(c) = batch.into_iter().next() {
                assert_eq!(c.result.expect("write ok"), LEN);
                break;
            }
        }
        tmp.file.sync_all().expect("sync");

        let mut dst = vec![0u8; LEN];
        let rop = ReadOp {
            fd,
            offset: 0,
            buf: dst.as_mut_ptr(),
            len: LEN,
            user_data: 2,
            priority: IoPriority::High,
        };
        // SAFETY: `tmp.file` and `dst` outlive the completion reaped below.
        let got = unsafe { queue.submit(core::slice::from_ref(&rop)) }.expect("r");
        assert_eq!(got.accepted, 1);
        loop {
            let batch = queue.wait(1, Some(Duration::from_secs(5))).expect("wait");
            if let Some(c) = batch.into_iter().next() {
                assert_eq!(c.result.expect("read ok"), LEN);
                break;
            }
        }
        for (i, &b) in dst.iter().enumerate() {
            assert_eq!(b, pattern(i), "mismatch at byte {i}");
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_sq_backpressure_and_requeue() {
        const LEN: usize = 64;
        const N: usize = 512;
        let tmp = TempFile::with_pattern(LEN * N);
        let fd = tmp.file.as_raw_fd();
        let mut queue = AioQueue::new().expect("io_uring queue");
        let mut buffers: Vec<Vec<u8>> = (0..N).map(|_| vec![0; LEN]).collect();
        let ops: Vec<ReadOp> = buffers
            .iter_mut()
            .enumerate()
            .map(|(i, buf)| ReadOp {
                fd,
                offset: (i * LEN) as u64,
                buf: buf.as_mut_ptr(),
                len: LEN,
                user_data: i as u64,
                priority: IoPriority::Low,
            })
            .collect();

        let mut next = 0;
        let mut reaped = 0;
        let mut observed_rejection = false;
        while reaped < N {
            if next < N {
                // SAFETY: the file and buffers outlive all completions; buffers
                // are inspected only after their completion is reaped.
                let outcome = unsafe { queue.submit(&ops[next..]) }.expect("submit");
                observed_rejection |= !outcome.rejected.is_empty();
                next += outcome.accepted;
            }
            if queue.pending() != 0 {
                let completions = queue.wait(N, Some(Duration::from_secs(5))).expect("wait");
                assert!(!completions.is_empty(), "io_uring wait timed out");
                reaped += completions.len();
            }
        }
        assert_eq!(next, N);
        assert_eq!(queue.pending(), 0);
        assert!(
            observed_rejection,
            "SQ capacity should exercise backpressure"
        );
        for (i, buf) in buffers.iter().enumerate() {
            assert!(buf
                .iter()
                .enumerate()
                .all(|(j, &byte)| byte == pattern(i * LEN + j)));
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_drop_cancels_or_drains_inflight_reads() {
        const LEN: usize = 4096;
        const N: usize = 256;
        let tmp = TempFile::with_pattern(LEN * N);
        let fd = tmp.file.as_raw_fd();
        let mut buffers: Vec<Vec<u8>> = (0..N).map(|_| vec![0; LEN]).collect();
        let ops: Vec<ReadOp> = buffers
            .iter_mut()
            .enumerate()
            .map(|(i, buf)| ReadOp {
                fd,
                offset: (i * LEN) as u64,
                buf: buf.as_mut_ptr(),
                len: LEN,
                user_data: i as u64,
                priority: IoPriority::Normal,
            })
            .collect();
        let mut queue = AioQueue::new().expect("io_uring queue");
        // SAFETY: the file and buffers remain alive through queue drop, which
        // must cancel or reap every request before returning.
        let outcome = unsafe { queue.submit(&ops) }.expect("submit");
        assert!(outcome.accepted > 0);
        drop(queue);
        drop(buffers);
    }
}

#[cfg(windows)]
mod windows {
    use super::*;
    use core::ffi::c_void;
    use core::time::Duration;
    use std::os::windows::ffi::OsStrExt as _;
    use std::path::{Path, PathBuf};

    type RawFd = i32;

    type Handle = *mut c_void;

    const INVALID_HANDLE_VALUE: Handle = (-1isize) as Handle;
    const GENERIC_READ: u32 = 0x8000_0000;
    const GENERIC_WRITE: u32 = 0x4000_0000;
    const FILE_SHARE_READ: u32 = 0x0000_0001;
    const FILE_SHARE_WRITE: u32 = 0x0000_0002;
    const FILE_SHARE_DELETE: u32 = 0x0000_0004;
    const OPEN_EXISTING: u32 = 3;
    const FILE_ATTRIBUTE_NORMAL: u32 = 0x0000_0080;
    const FILE_FLAG_OVERLAPPED: u32 = 0x4000_0000;
    const O_BINARY: i32 = 0x8000;

    #[expect(
        unsafe_code,
        reason = "Windows-only tests create real FILE_FLAG_OVERLAPPED CRT descriptors"
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
        fn CloseHandle(object: Handle) -> i32;
    }

    #[expect(
        unsafe_code,
        reason = "Windows-only tests bridge an overlapped Win32 handle into RawFd"
    )]
    unsafe extern "C" {
        fn _open_osfhandle(os_handle: isize, flags: i32) -> i32;
        fn _close(fd: i32) -> i32;
    }

    fn wide(path: &Path) -> Vec<u16> {
        path.as_os_str().encode_wide().chain([0]).collect()
    }

    fn overlapped_fd(path: &Path, access: u32) -> RawFd {
        let name = wide(path);
        // SAFETY: the UTF-16 path is NUL-terminated and all pointer arguments
        // are either null or point to owned values for this call.
        let handle = unsafe {
            CreateFileW(
                name.as_ptr(),
                access,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                core::ptr::null_mut(),
                OPEN_EXISTING,
                FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
                core::ptr::null_mut(),
            )
        };
        assert!(
            !handle.is_null() && handle != INVALID_HANDLE_VALUE,
            "CreateFileW failed"
        );
        // SAFETY: ownership of the handle is transferred to the CRT fd on
        // success; the test closes that fd in TempFd::drop.
        let fd = unsafe { _open_osfhandle(handle as isize, O_BINARY) };
        if fd < 0 {
            // SAFETY: _open_osfhandle did not take ownership on failure.
            unsafe {
                CloseHandle(handle);
            }
            panic!("_open_osfhandle failed");
        }
        fd
    }

    struct TempFd {
        path: PathBuf,
        fd: RawFd,
    }

    impl TempFd {
        fn with_bytes(bytes: &[u8], access: u32) -> Self {
            let mut path = std::env::temp_dir();
            path.push(format!(
                "prism_aio_win_{}_{}.bin",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::write(&path, bytes).expect("write Windows test fixture");
            let fd = overlapped_fd(&path, access);
            Self { path, fd }
        }
    }

    impl Drop for TempFd {
        fn drop(&mut self) {
            // SAFETY: this test owns the CRT fd and closes it exactly once.
            unsafe {
                _close(self.fd);
            }
            let _ = std::fs::remove_file(&self.path);
        }
    }

    fn pattern(offset: usize) -> u8 {
        (offset % 251) as u8
    }

    fn reap_one(queue: &mut AioQueue) -> super::super::Completion {
        loop {
            let completions = queue
                .wait(1, Some(Duration::from_secs(5)))
                .expect("IOCP wait");
            if let Some(completion) = completions.into_iter().next() {
                return completion;
            }
        }
    }

    #[test]
    fn supported_on_windows_iocp() {
        assert!(AioQueue::supported());
        assert!(AioQueue::new().is_ok());
    }

    #[test]
    fn batch_reads_round_trip() {
        const LEN: usize = 4096;
        const N: usize = 3;
        let bytes: Vec<u8> = (0..LEN * N).map(pattern).collect();
        let file = TempFd::with_bytes(&bytes, GENERIC_READ);
        let mut buffers: Vec<Vec<u8>> = (0..N).map(|_| vec![0; LEN]).collect();
        let ops: Vec<ReadOp> = buffers
            .iter_mut()
            .enumerate()
            .map(|(i, buffer)| ReadOp {
                fd: file.fd,
                offset: (i * LEN) as u64,
                buf: buffer.as_mut_ptr(),
                len: LEN,
                user_data: 0xA000 + i as u64,
                priority: IoPriority::Normal,
            })
            .collect();

        let mut queue = AioQueue::new().expect("IOCP queue");
        // SAFETY: the overlapped fd and buffers outlive all reaped completions.
        let outcome = unsafe { queue.submit(&ops) }.expect("IOCP submit");
        assert_eq!(outcome.accepted, N);
        assert!(outcome.all_accepted());
        for _ in 0..N {
            assert_eq!(reap_one(&mut queue).result.expect("read"), LEN);
        }
        assert_eq!(queue.pending(), 0);
        for (i, buffer) in buffers.iter().enumerate() {
            assert!(buffer
                .iter()
                .enumerate()
                .all(|(j, &byte)| byte == pattern(i * LEN + j)));
        }
    }

    #[test]
    fn batch_writes_round_trip() {
        const LEN: usize = 2048;
        const N: usize = 3;
        let file = TempFd::with_bytes(&[], GENERIC_READ | GENERIC_WRITE);
        let sources: Vec<Vec<u8>> = (0..N)
            .map(|i| (0..LEN).map(|j| pattern(i * LEN + j)).collect())
            .collect();
        let ops: Vec<WriteOp> = sources
            .iter()
            .enumerate()
            .map(|(i, source)| WriteOp {
                fd: file.fd,
                offset: (i * LEN) as u64,
                buf: source.as_ptr(),
                len: LEN,
                user_data: 0xB000 + i as u64,
                priority: IoPriority::Normal,
            })
            .collect();

        let mut queue = AioQueue::new().expect("IOCP queue");
        // SAFETY: the overlapped fd and source buffers remain alive and
        // unchanged until every completion is reaped.
        let outcome = unsafe { queue.submit_write(&ops) }.expect("IOCP submit");
        assert_eq!(outcome.accepted, N);
        for _ in 0..N {
            assert_eq!(reap_one(&mut queue).result.expect("write"), LEN);
        }
        let readback = std::fs::read(&file.path).expect("read Windows output");
        let expected: Vec<u8> = sources.into_iter().flatten().collect();
        assert_eq!(readback, expected);
    }

    #[test]
    fn bounded_submit_reports_backpressure_and_requeues() {
        const LEN: usize = 64;
        const N: usize = 300;
        let bytes: Vec<u8> = (0..LEN * N).map(pattern).collect();
        let file = TempFd::with_bytes(&bytes, GENERIC_READ);
        let mut buffers: Vec<Vec<u8>> = (0..N).map(|_| vec![0; LEN]).collect();
        let ops: Vec<ReadOp> = buffers
            .iter_mut()
            .enumerate()
            .map(|(i, buffer)| ReadOp {
                fd: file.fd,
                offset: (i * LEN) as u64,
                buf: buffer.as_mut_ptr(),
                len: LEN,
                user_data: i as u64,
                priority: IoPriority::Low,
            })
            .collect();
        let mut queue = AioQueue::new().expect("IOCP queue");
        // SAFETY: the fd and buffers outlive every completion below.
        let first = unsafe { queue.submit(&ops) }.expect("first submit");
        assert!(first.accepted > 0);
        assert!(!first.rejected.is_empty());
        while queue.pending() != 0 {
            let _ = queue.wait(N, Some(Duration::from_secs(5))).expect("drain");
        }
        let rejected: Vec<ReadOp> = ops
            .iter()
            .copied()
            .filter(|op| first.rejected.contains(&op.user_data))
            .collect();
        // SAFETY: the same fd and buffers remain valid for the rejected tail.
        let second = unsafe { queue.submit(&rejected) }.expect("requeue");
        assert_eq!(second.accepted, rejected.len());
        while queue.pending() != 0 {
            let _ = queue.wait(N, Some(Duration::from_secs(5))).expect("drain");
        }
        for (i, buffer) in buffers.iter().enumerate() {
            assert!(buffer
                .iter()
                .enumerate()
                .all(|(j, &byte)| byte == pattern(i * LEN + j)));
        }
    }

    #[test]
    fn invalid_fd_is_rejected_before_io() {
        let mut queue = AioQueue::new().expect("IOCP queue");
        let op = ReadOp {
            fd: -1,
            offset: 0,
            buf: core::ptr::null_mut(),
            len: 0,
            user_data: 0,
            priority: IoPriority::Normal,
        };
        // SAFETY: validation rejects the invalid descriptor before dereference.
        assert_eq!(
            unsafe { queue.submit(core::slice::from_ref(&op)) },
            Err(AioError::InvalidArgument)
        );
    }

    #[test]
    fn drop_cancels_and_drains_inflight_requests() {
        const LEN: usize = 4096;
        let bytes = vec![7_u8; LEN];
        let file = TempFd::with_bytes(&bytes, GENERIC_READ);
        let mut buffer = vec![0_u8; LEN];
        let op = ReadOp {
            fd: file.fd,
            offset: 0,
            buf: buffer.as_mut_ptr(),
            len: LEN,
            user_data: 1,
            priority: IoPriority::Normal,
        };
        let mut queue = AioQueue::new().expect("IOCP queue");
        // SAFETY: file and buffer remain alive through queue destruction.
        let outcome = unsafe { queue.submit(core::slice::from_ref(&op)) }.expect("submit");
        assert_eq!(outcome.accepted, 1);
        drop(queue);
        drop(buffer);
    }
}

#[cfg(not(any(target_vendor = "apple", target_os = "linux", target_os = "windows")))]
mod fallback {
    use super::*;

    #[test]
    fn honestly_unsupported() {
        assert!(!AioQueue::supported());
        assert!(matches!(AioQueue::new().err(), Some(AioError::Unsupported)));
    }
}
