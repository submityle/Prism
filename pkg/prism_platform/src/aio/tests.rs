//! Tests for the §24.1 async-I/O facade.
//!
//! On Apple/BSD the POSIX AIO backend is exercised for real against a temp
//! file: we write a known byte pattern, batch-submit reads at distinct offsets
//! into distinct buffers, reap the completions, and assert each buffer matches
//! the source bytes and each `user_data` round-trips. Elsewhere we only assert
//! honest `Unsupported` degradation.

#![expect(
    unsafe_code,
    reason = "exercising the unsafe batch-submit API against a real temp file"
)]

use super::{AioError, AioQueue, IoPriority, ReadOp};

#[cfg(any(
    target_vendor = "apple",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "dragonfly"
))]
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

    #[test]
    fn supported_on_apple_bsd() {
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
        let submitted = unsafe { queue.submit(&ops) }.expect("submit");
        assert_eq!(submitted, N);
        assert_eq!(queue.pending(), N);

        let mut seen = alloc::collections::BTreeMap::new();
        while seen.len() < N {
            let batch = queue
                .wait(N, Some(Duration::from_secs(5)))
                .expect("wait");
            assert!(!batch.is_empty(), "wait returned no completions before timeout");
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
                let accepted = unsafe { queue.submit(&ops[next..]) }.expect("submit");
                next += accepted;
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
        assert_eq!(unsafe { queue.submit(&[]) }.expect("submit"), 0);
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
}

#[cfg(not(any(
    target_vendor = "apple",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "dragonfly"
)))]
mod fallback {
    use super::*;

    #[test]
    fn honestly_unsupported() {
        assert!(!AioQueue::supported());
        assert!(matches!(AioQueue::new().err(), Some(AioError::Unsupported)));
    }
}
