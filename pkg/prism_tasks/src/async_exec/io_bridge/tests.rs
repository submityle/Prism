//! Real-backend tests for the §24.5 async-I/O bridge.
//!
//! These exercise the live POSIX AIO backend on Apple/BSD (the queue issues
//! real `lio_listio`/`aio_*` syscalls), so they must run outside a syscall
//! sandbox. On platforms without a backend they assert the honest
//! `Unsupported` contract instead.

use std::fs;
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{IoError, IoReactor};
use crate::TaskPool;
use prism_platform::aio::IoPriority;

/// Deterministic byte pattern so reads can be verified position-by-position.
fn pattern_byte(offset: usize) -> u8 {
    (offset % 251) as u8
}

/// Create a uniquely named temp file filled with `len` pattern bytes.
fn make_pattern_file(len: usize) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let mut path = std::env::temp_dir();
    path.push(format!("prism_io_bridge_{pid}_{id}.bin"));
    let data: Vec<u8> = (0..len).map(pattern_byte).collect();
    fs::write(&path, &data).expect("write temp pattern file");
    path
}

#[test]
fn unsupported_platforms_report_honestly() {
    if IoReactor::supported() {
        // Covered by the live-read tests below.
        return;
    }
    assert!(matches!(IoReactor::new(), Err(IoError::Unsupported)));
}

#[test]
fn reads_back_a_full_file() {
    if !IoReactor::supported() {
        return;
    }
    let len = 4096usize;
    let path = make_pattern_file(len);
    let file = fs::File::open(&path).expect("open temp file");
    let fd = file.as_raw_fd();

    let reactor = IoReactor::new().expect("start reactor");
    let pool = TaskPool::new();
    let fut = reactor.read(fd, 0, len, IoPriority::Normal);
    let buf = pool.block_on(fut).expect("read completes");

    assert_eq!(buf.bytes, len, "read the whole file");
    for (i, &byte) in buf.filled().iter().enumerate() {
        assert_eq!(byte, pattern_byte(i), "byte {i} mismatch");
    }

    drop(file);
    let _ = fs::remove_file(&path);
}

#[test]
fn reads_at_an_offset() {
    if !IoReactor::supported() {
        return;
    }
    let len = 8192usize;
    let path = make_pattern_file(len);
    let file = fs::File::open(&path).expect("open temp file");
    let fd = file.as_raw_fd();

    let reactor = IoReactor::new().expect("start reactor");
    let pool = TaskPool::new();
    let offset = 1000usize;
    let read_len = 2000usize;
    let fut = reactor.read(fd, offset as u64, read_len, IoPriority::High);
    let buf = pool.block_on(fut).expect("read completes");

    assert_eq!(buf.bytes, read_len);
    for (i, &byte) in buf.filled().iter().enumerate() {
        assert_eq!(byte, pattern_byte(offset + i), "byte {i} mismatch");
    }

    drop(file);
    let _ = fs::remove_file(&path);
}

#[test]
fn many_concurrent_reads_each_get_their_region() {
    if !IoReactor::supported() {
        return;
    }
    let chunk = 1024usize;
    let count = 32usize; // exceeds macOS AIO_MAX, exercising back-pressure
    let len = chunk * count;
    let path = make_pattern_file(len);
    let file = fs::File::open(&path).expect("open temp file");
    let fd = file.as_raw_fd();

    let reactor = IoReactor::new().expect("start reactor");
    let pool = TaskPool::new();

    // Submit every chunk up front, then block on each future in turn.
    let futures: Vec<_> = (0..count)
        .map(|i| reactor.read(fd, (i * chunk) as u64, chunk, IoPriority::Normal))
        .collect();

    for (i, fut) in futures.into_iter().enumerate() {
        let buf = pool.block_on(fut).expect("read completes");
        assert_eq!(buf.bytes, chunk, "chunk {i} length");
        let base = i * chunk;
        for (j, &byte) in buf.filled().iter().enumerate() {
            assert_eq!(byte, pattern_byte(base + j), "chunk {i} byte {j}");
        }
    }

    drop(file);
    let _ = fs::remove_file(&path);
}

#[test]
fn reading_past_eof_returns_short_count() {
    if !IoReactor::supported() {
        return;
    }
    let len = 500usize;
    let path = make_pattern_file(len);
    let file = fs::File::open(&path).expect("open temp file");
    let fd = file.as_raw_fd();

    let reactor = IoReactor::new().expect("start reactor");
    let pool = TaskPool::new();
    // Ask for more than the file holds, starting near the end.
    let fut = reactor.read(fd, 400, 1024, IoPriority::Low);
    let buf = pool.block_on(fut).expect("read completes");

    assert_eq!(buf.bytes, 100, "only 100 bytes remain past offset 400");
    for (i, &byte) in buf.filled().iter().enumerate() {
        assert_eq!(byte, pattern_byte(400 + i), "byte {i} mismatch");
    }

    drop(file);
    let _ = fs::remove_file(&path);
}

#[test]
fn read_into_reuses_the_caller_buffer() {
    if !IoReactor::supported() {
        return;
    }
    let len = 2048usize;
    let path = make_pattern_file(len);
    let file = fs::File::open(&path).expect("open temp file");
    let fd = file.as_raw_fd();

    let reactor = IoReactor::new().expect("start reactor");
    let pool = TaskPool::new();
    let scratch = vec![0u8; len].into_boxed_slice();
    let fut = reactor.read_into(fd, 0, IoPriority::Normal, scratch);
    let buf = pool.block_on(fut).expect("read completes");

    assert_eq!(buf.data.len(), len, "buffer kept its capacity");
    assert_eq!(buf.bytes, len);
    for (i, &byte) in buf.filled().iter().enumerate() {
        assert_eq!(byte, pattern_byte(i), "byte {i} mismatch");
    }

    drop(file);
    let _ = fs::remove_file(&path);
}

#[test]
fn dropping_a_future_before_completion_is_safe() {
    if !IoReactor::supported() {
        return;
    }
    let len = 16384usize;
    let path = make_pattern_file(len);
    let file = fs::File::open(&path).expect("open temp file");
    let fd = file.as_raw_fd();

    let reactor = IoReactor::new().expect("start reactor");
    // Submit a read and immediately drop the future without ever polling it.
    // The reactor still owns the destination buffer, so cancelling/draining it
    // on reactor drop must not touch freed memory.
    let fut = reactor.read(fd, 0, len, IoPriority::Normal);
    drop(fut);
    drop(reactor); // joins the thread; must not crash or leak unsafely.

    drop(file);
    let _ = fs::remove_file(&path);
}
