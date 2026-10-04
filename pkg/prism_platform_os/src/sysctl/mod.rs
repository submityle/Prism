//! Thin, safe wrappers over the macOS `sysctlbyname(3)` interface.
//!
//! This is the only place in the crate that touches raw FFI. Each public
//! function wraps a single `sysctlbyname` round-trip in a `// SAFETY:`-audited
//! `unsafe` block and returns a plain Rust value or an [`io::Error`]. The
//! length-probe-then-read pattern makes the readers agnostic to whether the
//! kernel reports a key as a 32-bit `int` or a 64-bit `int64`, so callers never
//! have to hard-code a width that differs across keys (`hw.logicalcpu` is an
//! `int`, `hw.memsize` is a `uint64`).
#![cfg(target_os = "macos")]

use alloc::ffi::CString;
use std::io;

/// Read the raw bytes a sysctl key resolves to.
///
/// Performs the standard two-call dance: a first call with a null output
/// pointer asks the kernel for the value's size, then a second call fills a
/// buffer of exactly that size.
///
/// # Errors
///
/// Returns the OS error if the key is unknown, unreadable in this process
/// (sandboxing can deny `sysctl`), or changes size between the two calls.
fn read_raw(name: &str) -> io::Result<Vec<u8>> {
    let cname = CString::new(name)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "sysctl name has NUL byte"))?;

    let mut size: libc::size_t = 0;
    #[expect(
        unsafe_code,
        reason = "sysctlbyname length probe is the only portable macOS topology source"
    )]
    // SAFETY: `cname` is a valid NUL-terminated C string that outlives the
    // call. Passing a null `oldp` with a valid `oldlenp` is the documented way
    // to query the value length; the kernel only writes `size` here.
    let rc = unsafe {
        libc::sysctlbyname(
            cname.as_ptr(),
            core::ptr::null_mut(),
            &mut size,
            core::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    if size == 0 {
        return Ok(Vec::new());
    }

    let mut buf = vec![0u8; size];
    #[expect(
        unsafe_code,
        reason = "sysctlbyname value read is the only portable macOS topology source"
    )]
    // SAFETY: `buf` has `size` bytes and `size` is the length the kernel just
    // reported for this key; `cname` is still valid. The kernel writes at most
    // `size` bytes and updates `size` to the amount actually written.
    let rc = unsafe {
        libc::sysctlbyname(
            cname.as_ptr(),
            buf.as_mut_ptr().cast::<libc::c_void>(),
            &mut size,
            core::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    buf.truncate(size);
    Ok(buf)
}

/// Read an unsigned integer sysctl key, transparently handling the 32-bit
/// (`int`) and 64-bit (`int64`/`uint64`) encodings the kernel uses.
///
/// # Errors
///
/// Returns an error if the key cannot be read or its byte width is neither 4
/// nor 8.
pub fn read_uint(name: &str) -> io::Result<u64> {
    let buf = read_raw(name)?;
    match buf.len() {
        4 => Ok(u64::from(u32::from_ne_bytes([buf[0], buf[1], buf[2], buf[3]]))),
        8 => Ok(u64::from_ne_bytes([
            buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
        ])),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("sysctl `{name}` returned {other} bytes, expected 4 or 8"),
        )),
    }
}

/// Read a NUL-terminated string sysctl key (for example
/// `machdep.cpu.brand_string`).
///
/// # Errors
///
/// Returns an error if the key cannot be read or is not valid UTF-8.
pub fn read_string(name: &str) -> io::Result<String> {
    let mut buf = read_raw(name)?;
    // Drop a single trailing NUL the kernel includes in the reported length.
    if buf.last() == Some(&0) {
        buf.pop();
    }
    String::from_utf8(buf)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("sysctl `{name}`: {e}")))
}
