//! Windows virtual-memory backend.
//!
//! Reservation uses `VirtualAlloc(MEM_RESERVE, PAGE_NOACCESS)`; commit raises a
//! sub-range with `VirtualAlloc(MEM_COMMIT, …)`; decommit uses
//! `VirtualFree(MEM_DECOMMIT)`; protection changes go through `VirtualProtect`;
//! release unmaps with `VirtualFree(MEM_RELEASE)`. Memory information comes from
//! `GlobalMemoryStatusEx` + `GetSystemInfo`, and large-page support from
//! `GetLargePageMinimum`.
//!
//! Every FFI call is a dependency-free `extern "system"` declaration resolved
//! against `kernel32` that `std` already links (no `windows`/`winapi` crate),
//! mirroring [`crate::thread::affinity`].
#![expect(
    unsafe_code,
    reason = "page-level virtual memory requires VirtualAlloc/VirtualFree/VirtualProtect/GlobalMemoryStatusEx/GetSystemInfo via dependency-free Win32 FFI"
)]

use core::ffi::c_void;

use super::{MemoryInfo, Protection, Region, Result, VmError};

unsafe extern "system" {
    fn VirtualAlloc(
        address: *mut c_void,
        size: usize,
        allocation_type: u32,
        protect: u32,
    ) -> *mut c_void;
    fn VirtualFree(address: *mut c_void, size: usize, free_type: u32) -> i32;
    fn VirtualProtect(address: *mut c_void, size: usize, new: u32, old: *mut u32) -> i32;
    fn GetLargePageMinimum() -> usize;
    fn GetLastError() -> u32;
    fn GetSystemInfo(info: *mut SystemInfo);
    fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
}

const MEM_COMMIT: u32 = 0x0000_1000;
const MEM_RESERVE: u32 = 0x0000_2000;
const MEM_DECOMMIT: u32 = 0x0000_4000;
const MEM_RELEASE: u32 = 0x0000_8000;
const MEM_LARGE_PAGES: u32 = 0x2000_0000;

const PAGE_NOACCESS: u32 = 0x01;
const PAGE_READONLY: u32 = 0x02;
const PAGE_READWRITE: u32 = 0x04;
const PAGE_EXECUTE_READ: u32 = 0x20;
const PAGE_EXECUTE_READWRITE: u32 = 0x40;

const ERROR_NOT_ENOUGH_MEMORY: u32 = 8;
const ERROR_OUTOFMEMORY: u32 = 14;
const ERROR_COMMITMENT_LIMIT: u32 = 1455;

/// Win32 `SYSTEM_INFO`; only `dw_page_size` is read.
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

/// Win32 `MEMORYSTATUSEX`; `dw_length` must be set before the call.
#[repr(C)]
struct MemoryStatusEx {
    dw_length: u32,
    dw_memory_load: u32,
    ull_total_phys: u64,
    ull_avail_phys: u64,
    ull_total_page_file: u64,
    ull_avail_page_file: u64,
    ull_total_virtual: u64,
    ull_avail_virtual: u64,
    ull_avail_extended_virtual: u64,
}

/// This build has a working page-level VM backend.
pub(super) const SUPPORTED: bool = true;

/// Translate a [`Protection`] into a Win32 page-protection constant.
fn prot_bits(prot: Protection) -> u32 {
    match prot {
        Protection::None => PAGE_NOACCESS,
        Protection::Read => PAGE_READONLY,
        Protection::ReadWrite => PAGE_READWRITE,
        Protection::ReadExec => PAGE_EXECUTE_READ,
        Protection::ReadWriteExec => PAGE_EXECUTE_READWRITE,
    }
}

/// Map a `GetLastError` code into a [`VmError`].
fn os_error() -> VmError {
    // SAFETY: `GetLastError` only reads this thread's last-error code.
    let code = unsafe { GetLastError() };
    if code == ERROR_NOT_ENOUGH_MEMORY
        || code == ERROR_OUTOFMEMORY
        || code == ERROR_COMMITMENT_LIMIT
    {
        VmError::OutOfMemory
    } else {
        VmError::SystemError(code as i32)
    }
}

fn system_info() -> SystemInfo {
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
    info
}

pub(super) fn page_size() -> usize {
    let ps = system_info().dw_page_size as usize;
    if ps > 0 { ps } else { 4096 }
}

pub(super) fn large_page_size() -> Option<usize> {
    // SAFETY: `GetLargePageMinimum` is a pure query taking no arguments.
    let min = unsafe { GetLargePageMinimum() };
    if min > 0 { Some(min) } else { None }
}

pub(super) fn huge_pages_supported() -> bool {
    large_page_size().is_some()
}

pub(super) fn memory_info() -> Result<MemoryInfo> {
    let mut status = MemoryStatusEx {
        dw_length: core::mem::size_of::<MemoryStatusEx>() as u32,
        dw_memory_load: 0,
        ull_total_phys: 0,
        ull_avail_phys: 0,
        ull_total_page_file: 0,
        ull_avail_page_file: 0,
        ull_total_virtual: 0,
        ull_avail_virtual: 0,
        ull_avail_extended_virtual: 0,
    };
    // SAFETY: `dw_length` is set to the struct size and we pass a writable
    // pointer to the fully-owned struct, as the API requires.
    let ok = unsafe { GlobalMemoryStatusEx(core::ptr::from_mut(&mut status)) };
    if ok == 0 {
        return Err(os_error());
    }
    Ok(MemoryInfo {
        total_physical: status.ull_total_phys,
        available_physical: status.ull_avail_phys.min(status.ull_total_phys),
        page_size: page_size(),
        large_page_size: large_page_size(),
    })
}

pub(super) fn reserve(size: usize) -> Result<Region> {
    // SAFETY: a null `address` lets the OS choose the base; `MEM_RESERVE` with
    // `PAGE_NOACCESS` touches no memory. The result is validated before use.
    let p = unsafe {
        VirtualAlloc(core::ptr::null_mut(), size, MEM_RESERVE, PAGE_NOACCESS)
    };
    if p.is_null() {
        return Err(os_error());
    }
    let ptr = p.cast::<u8>();
    Ok(Region {
        base: ptr,
        base_len: size,
        ptr,
        len: size,
    })
}

pub(super) fn reserve_aligned(size: usize, align: usize) -> Result<Region> {
    let total = size.checked_add(align).ok_or(VmError::InvalidArgument)?;
    // SAFETY: see `reserve`; this reserves `size + align` bytes so an aligned
    // sub-range is guaranteed to fit.
    let p = unsafe {
        VirtualAlloc(core::ptr::null_mut(), total, MEM_RESERVE, PAGE_NOACCESS)
    };
    if p.is_null() {
        return Err(os_error());
    }
    let base_addr = p.addr();
    let aligned_addr = (base_addr + (align - 1)) & !(align - 1);
    let front = aligned_addr - base_addr;
    let aligned = p.wrapping_byte_add(front);
    // Windows cannot free a partial reservation, so the whole `p` span is kept
    // as the release base; only the aligned sub-range is exposed.
    Ok(Region {
        base: p.cast::<u8>(),
        base_len: total,
        ptr: aligned.cast::<u8>(),
        len: size,
    })
}

pub(super) fn reserve_huge(size: usize) -> Result<Region> {
    // SAFETY: see `reserve`; `MEM_LARGE_PAGES` requests large pages, which are
    // reserved and committed together as `PAGE_READWRITE`. Lacking the
    // "Lock pages in memory" privilege makes this fail, reported via the error
    // code.
    let p = unsafe {
        VirtualAlloc(
            core::ptr::null_mut(),
            size,
            MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
            PAGE_READWRITE,
        )
    };
    if p.is_null() {
        return Err(os_error());
    }
    let ptr = p.cast::<u8>();
    Ok(Region {
        base: ptr,
        base_len: size,
        ptr,
        len: size,
    })
}

pub(super) fn commit(ptr: *mut u8, len: usize, prot: Protection) -> Result<()> {
    // SAFETY: `ptr`/`len` is a validated page-aligned sub-range of a live
    // reservation; committing it with the requested protection is sound.
    let p = unsafe { VirtualAlloc(ptr.cast::<c_void>(), len, MEM_COMMIT, prot_bits(prot)) };
    if p.is_null() { Err(os_error()) } else { Ok(()) }
}

pub(super) fn decommit(ptr: *mut u8, len: usize) -> Result<()> {
    // SAFETY: `ptr`/`len` is a validated page-aligned sub-range of a live
    // reservation; `MEM_DECOMMIT` returns its physical pages while keeping the
    // address space reserved.
    let rc = unsafe { VirtualFree(ptr.cast::<c_void>(), len, MEM_DECOMMIT) };
    if rc == 0 { Err(os_error()) } else { Ok(()) }
}

pub(super) fn protect(ptr: *mut u8, len: usize, prot: Protection) -> Result<()> {
    let mut old: u32 = 0;
    // SAFETY: `ptr`/`len` is a validated page-aligned sub-range of a live
    // reservation; `VirtualProtect` writes the previous protection into `old`.
    let rc = unsafe {
        VirtualProtect(
            ptr.cast::<c_void>(),
            len,
            prot_bits(prot),
            core::ptr::from_mut(&mut old),
        )
    };
    if rc == 0 { Err(os_error()) } else { Ok(()) }
}

pub(super) fn release(region: Region, _huge: bool) {
    // SAFETY: `region.base` is the original `VirtualAlloc` base; `MEM_RELEASE`
    // with a zero size releases the entire reservation exactly once, from
    // `Drop`.
    unsafe {
        VirtualFree(region.base.cast::<c_void>(), 0, MEM_RELEASE);
    }
}
