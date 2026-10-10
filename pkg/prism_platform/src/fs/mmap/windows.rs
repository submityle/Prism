//! Windows memory-mapping backend.
//!
//! Maps the file with `CreateFileMappingW` + `MapViewOfFile`, unmaps with
//! `UnmapViewOfFile` + `CloseHandle`, and flushes with `FlushViewOfFile`.
//! Arbitrary byte offsets are supported by aligning the view down to the
//! allocation granularity and exposing the requested sub-slice.
//!
//! Every FFI call is a dependency-free `extern "system"` declaration resolved
//! against `kernel32` that `std` already links (no `windows`/`winapi` crate),
//! mirroring [`crate::vm`].
//!
//! **Status:** written to the documented Win32 contract but **not yet validated
//! on a Windows host** (this crate's M4 work was done on macOS). Treat as
//! provisional until exercised on Windows.

use core::ffi::c_void;
use std::fs::File;
use std::os::windows::io::AsRawHandle;

use super::{MmapError, Result};

type Handle = *mut c_void;

#[expect(
    unsafe_code,
    reason = "memory mapping requires CreateFileMapping/MapViewOfFile/UnmapViewOfFile/FlushViewOfFile/CloseHandle via dependency-free Win32 FFI"
)]
unsafe extern "system" {
    fn CreateFileMappingW(
        file: Handle,
        attributes: *mut c_void,
        protect: u32,
        max_size_high: u32,
        max_size_low: u32,
        name: *const u16,
    ) -> Handle;
    fn MapViewOfFile(
        mapping: Handle,
        desired_access: u32,
        offset_high: u32,
        offset_low: u32,
        bytes: usize,
    ) -> *mut c_void;
    fn UnmapViewOfFile(base: *const c_void) -> i32;
    fn FlushViewOfFile(base: *const c_void, bytes: usize) -> i32;
    fn CloseHandle(object: Handle) -> i32;
    fn GetLastError() -> u32;
    fn GetSystemInfo(info: *mut SystemInfo);
}

const PAGE_READONLY: u32 = 0x02;
const PAGE_READWRITE: u32 = 0x04;
const FILE_MAP_READ: u32 = 0x0004;
const FILE_MAP_WRITE: u32 = 0x0002;

/// Win32 `SYSTEM_INFO`; only `dw_allocation_granularity` is read.
#[repr(C)]
struct SystemInfo {
    w_processor_architecture: u16,
    w_reserved: u16,
    dw_page_size: u32,
    lp_minimum_application_address: *mut c_void,
    lp_maximum_application_address: *mut c_void,
    dw_active_processor_mask: usize,
    dw_number_of_processors: u32,
    dw_processor_type: u32,
    dw_allocation_granularity: u32,
    w_processor_level: u16,
    w_processor_revision: u16,
}

/// This build maps files zero-copy through the real `MapViewOfFile` primitive.
pub(super) const SUPPORTED: bool = true;

#[expect(unsafe_code, reason = "GetSystemInfo fills a caller-owned struct")]
fn allocation_granularity() -> u64 {
    let mut info = SystemInfo {
        w_processor_architecture: 0,
        w_reserved: 0,
        dw_page_size: 0,
        lp_minimum_application_address: core::ptr::null_mut(),
        lp_maximum_application_address: core::ptr::null_mut(),
        dw_active_processor_mask: 0,
        dw_number_of_processors: 0,
        dw_processor_type: 0,
        dw_allocation_granularity: 0,
        w_processor_level: 0,
        w_processor_revision: 0,
    };
    // SAFETY: `GetSystemInfo` fills the fully-owned, correctly-sized struct we
    // pass by writable pointer and reads nothing from it.
    unsafe {
        GetSystemInfo(core::ptr::from_mut(&mut info));
    }
    let g = info.dw_allocation_granularity as u64;
    if g > 0 {
        g
    } else {
        65536
    }
}

/// An owned Windows file-mapping view.
pub(super) struct Mapping {
    view: *mut c_void,
    mapping: Handle,
    data: *mut u8,
    len: usize,
    writable: bool,
    _file: File,
}

/// Map `[offset, offset + len)` of `file`, read-only or read-write.
#[expect(
    unsafe_code,
    reason = "CreateFileMapping + MapViewOfFile is the core zero-copy mapping primitive"
)]
pub(super) fn map(file: File, offset: u64, len: usize, writable: bool) -> Result<Mapping> {
    let granularity = allocation_granularity();
    let aligned = offset & !(granularity - 1);
    let delta = (offset - aligned) as usize;
    let view_len = delta.checked_add(len).ok_or(MmapError::InvalidArgument)?;
    let protect = if writable {
        PAGE_READWRITE
    } else {
        PAGE_READONLY
    };
    let access = FILE_MAP_READ | if writable { FILE_MAP_WRITE } else { 0 };
    let fhandle = file.as_raw_handle().cast::<c_void>();

    // SAFETY: `fhandle` is a live file handle owned by `file`; passing size 0/0
    // maps the whole file. A null attributes/name pointer requests the default
    // unnamed mapping. The result is checked for null before use.
    let mapping = unsafe {
        CreateFileMappingW(
            fhandle,
            core::ptr::null_mut(),
            protect,
            0,
            0,
            core::ptr::null(),
        )
    };
    if mapping.is_null() {
        // SAFETY: GetLastError only reads this thread's last-error code.
        return Err(MmapError::System(unsafe { GetLastError() } as i32));
    }
    let offset_high = (aligned >> 32) as u32;
    let offset_low = (aligned & 0xFFFF_FFFF) as u32;
    // SAFETY: `mapping` is the handle just created; `aligned` is a multiple of
    // the allocation granularity as `MapViewOfFile` requires; `view_len` is the
    // validated view length. Result checked for null.
    let view = unsafe { MapViewOfFile(mapping, access, offset_high, offset_low, view_len) };
    if view.is_null() {
        // SAFETY: closing the just-created mapping handle on the failure path.
        let err = unsafe { GetLastError() } as i32;
        // SAFETY: `mapping` is a valid handle we own and no longer use.
        unsafe {
            CloseHandle(mapping);
        }
        return Err(MmapError::System(err));
    }
    let data = view.cast::<u8>().wrapping_add(delta);
    Ok(Mapping {
        view,
        mapping,
        data,
        len,
        writable,
        _file: file,
    })
}

impl Mapping {
    pub(super) fn as_ptr(&self) -> *const u8 {
        self.data.cast_const()
    }

    pub(super) fn as_mut_ptr(&self) -> *mut u8 {
        self.data
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    #[expect(
        unsafe_code,
        reason = "FlushViewOfFile writes dirty mapped pages to disk"
    )]
    pub(super) fn flush(&self) -> Result<()> {
        if !self.writable {
            return Ok(());
        }
        // SAFETY: `view` is the live view base this value owns; a length of 0
        // flushes to the end of the view.
        let rc = unsafe { FlushViewOfFile(self.view.cast_const(), 0) };
        if rc != 0 {
            Ok(())
        } else {
            // SAFETY: GetLastError only reads this thread's last-error code.
            Err(MmapError::System(unsafe { GetLastError() } as i32))
        }
    }
}

impl Drop for Mapping {
    #[expect(
        unsafe_code,
        reason = "UnmapViewOfFile + CloseHandle release the view and mapping exactly once on drop"
    )]
    fn drop(&mut self) {
        // SAFETY: `view` and `mapping` are the exact resources this value owns;
        // each is released once, here, at end of life.
        unsafe {
            UnmapViewOfFile(self.view.cast_const());
            CloseHandle(self.mapping);
        }
    }
}
