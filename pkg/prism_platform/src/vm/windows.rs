//! Windows virtual-memory backend.
//!
//! Reservation uses `VirtualAlloc(MEM_RESERVE, PAGE_NOACCESS)`; commit raises a
//! sub-range with `VirtualAlloc(MEM_COMMIT, …)`; decommit uses
//! `VirtualFree(MEM_DECOMMIT)`; protection changes go through `VirtualProtect`;
//! release unmaps with `VirtualFree(MEM_RELEASE)`. Memory information comes from
//! `GlobalMemoryStatusEx` + `GetSystemInfo`, and large-page support from
//! `GetLargePageMinimum`.
//!
//! The mirrored-ring path first resolves the Windows 10 placeholder APIs
//! (`VirtualAlloc2` + `MapViewOfFile3`) at runtime. Older Windows versions use
//! the real section-mapping fallback (`CreateFileMappingW` + two
//! `MapViewOfFileEx` calls), so an unavailable modern API never becomes a fake
//! ring. Every FFI call is a dependency-free `extern "system"` declaration
//! resolved against the system libraries that `std` already links (no
//! `windows`/`winapi` crate), mirroring [`crate::thread::affinity`].
#![expect(
    unsafe_code,
    reason = "page-level virtual memory requires VirtualAlloc/VirtualFree/VirtualProtect/GlobalMemoryStatusEx/GetSystemInfo via dependency-free Win32 FFI"
)]

use core::ffi::c_void;
use core::mem::size_of;

use super::{round_up, MemoryInfo, Protection, Region, Result, VmError};

unsafe extern "system" {
    fn VirtualAlloc(
        address: *mut c_void,
        size: usize,
        allocation_type: u32,
        protect: u32,
    ) -> *mut c_void;
    fn VirtualFree(address: *mut c_void, size: usize, free_type: u32) -> i32;
    fn VirtualProtect(address: *mut c_void, size: usize, new: u32, old: *mut u32) -> i32;
    fn CreateFileMappingW(
        file: *mut c_void,
        attributes: *mut c_void,
        protect: u32,
        maximum_size_high: u32,
        maximum_size_low: u32,
        name: *const u16,
    ) -> *mut c_void;
    fn MapViewOfFileEx(
        mapping: *mut c_void,
        desired_access: u32,
        file_offset_high: u32,
        file_offset_low: u32,
        number_of_bytes: usize,
        base_address: *mut c_void,
    ) -> *mut c_void;
    fn UnmapViewOfFile(base_address: *const c_void) -> i32;
    fn CloseHandle(handle: *mut c_void) -> i32;
    fn GetModuleHandleW(name: *const u16) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const u8) -> *const c_void;
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
const MEM_RESERVE_PLACEHOLDER: u32 = 0x0004_0000;
const MEM_REPLACE_PLACEHOLDER: u32 = 0x0000_4000;
const MEM_PRESERVE_PLACEHOLDER: u32 = 0x0000_0002;

const PAGE_NOACCESS: u32 = 0x01;
const PAGE_READONLY: u32 = 0x02;
const PAGE_READWRITE: u32 = 0x04;
const PAGE_EXECUTE_READ: u32 = 0x20;
const PAGE_EXECUTE_READWRITE: u32 = 0x40;

const FILE_MAP_READ: u32 = 0x0004;
const FILE_MAP_WRITE: u32 = 0x0002;
const ERROR_INVALID_ADDRESS: u32 = 487;

const MAP_RETRY_LIMIT: usize = 64;

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
    if ps > 0 {
        ps
    } else {
        4096
    }
}

pub(super) fn allocation_granularity() -> usize {
    let granularity = system_info().dw_allocation_granularity as usize;
    if granularity > 0 {
        granularity
    } else {
        64 * 1024
    }
}

pub(super) fn large_page_size() -> Option<usize> {
    // SAFETY: `GetLargePageMinimum` is a pure query taking no arguments.
    let min = unsafe { GetLargePageMinimum() };
    if min > 0 {
        Some(min)
    } else {
        None
    }
}

pub(super) fn huge_pages_supported() -> bool {
    large_page_size().is_some()
}

pub(super) fn memory_info() -> Result<MemoryInfo> {
    let mut status = MemoryStatusEx {
        dw_length: size_of::<MemoryStatusEx>() as u32,
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
    let p = unsafe { VirtualAlloc(core::ptr::null_mut(), size, MEM_RESERVE, PAGE_NOACCESS) };
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
    let p = unsafe { VirtualAlloc(core::ptr::null_mut(), total, MEM_RESERVE, PAGE_NOACCESS) };
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
    if p.is_null() {
        Err(os_error())
    } else {
        Ok(())
    }
}

pub(super) fn decommit(ptr: *mut u8, len: usize) -> Result<()> {
    // SAFETY: `ptr`/`len` is a validated page-aligned sub-range of a live
    // reservation; `MEM_DECOMMIT` returns its physical pages while keeping the
    // address space reserved.
    let rc = unsafe { VirtualFree(ptr.cast::<c_void>(), len, MEM_DECOMMIT) };
    if rc == 0 {
        Err(os_error())
    } else {
        Ok(())
    }
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
    if rc == 0 {
        Err(os_error())
    } else {
        Ok(())
    }
}

pub(super) fn release(region: Region, _huge: bool) {
    // SAFETY: `region.base` is the original `VirtualAlloc` base; `MEM_RELEASE`
    // with a zero size releases the entire reservation exactly once, from
    // `Drop`.
    unsafe {
        VirtualFree(region.base.cast::<c_void>(), 0, MEM_RELEASE);
    }
}

// ---------------------------------------------------------------------------
// Magic ring buffer (mirrored mapping) — design doc §24.2 "环形流送堆".
// ---------------------------------------------------------------------------

/// Windows has a real section-backed mirrored-ring implementation. The modern
/// placeholder APIs are optional at runtime; the section + `MapViewOfFileEx`
/// fallback works on older desktop Windows versions as well.
pub(super) const MIRROR_SUPPORTED: bool = true;

type VirtualAlloc2Fn = unsafe extern "system" fn(
    process: *mut c_void,
    base_address: *mut c_void,
    size: usize,
    allocation_type: u32,
    protect: u32,
    extended_parameters: *mut c_void,
    parameter_count: u32,
) -> *mut c_void;

type MapViewOfFile3Fn = unsafe extern "system" fn(
    mapping: *mut c_void,
    process: *mut c_void,
    base_address: *mut c_void,
    offset: u64,
    view_size: usize,
    allocation_type: u32,
    page_protection: u32,
    extended_parameters: *mut c_void,
    parameter_count: u32,
) -> *mut c_void;

const KERNEL32: [u16; 13] = [
    b'k' as u16,
    b'e' as u16,
    b'r' as u16,
    b'n' as u16,
    b'e' as u16,
    b'l' as u16,
    b'3' as u16,
    b'2' as u16,
    b'.' as u16,
    b'd' as u16,
    b'l' as u16,
    b'l' as u16,
    0,
];

const KERNELBASE: [u16; 14] = [
    b'k' as u16,
    b'e' as u16,
    b'r' as u16,
    b'n' as u16,
    b'e' as u16,
    b'l' as u16,
    b'b' as u16,
    b'a' as u16,
    b's' as u16,
    b'e' as u16,
    b'.' as u16,
    b'd' as u16,
    b'l' as u16,
    0,
];

const MEMORY_API_SET: [u16; 34] = [
    b'a' as u16,
    b'p' as u16,
    b'i' as u16,
    b'-' as u16,
    b'm' as u16,
    b's' as u16,
    b'-' as u16,
    b'w' as u16,
    b'i' as u16,
    b'n' as u16,
    b'-' as u16,
    b'c' as u16,
    b'o' as u16,
    b'r' as u16,
    b'e' as u16,
    b'-' as u16,
    b'm' as u16,
    b'e' as u16,
    b'm' as u16,
    b'o' as u16,
    b'r' as u16,
    b'y' as u16,
    b'-' as u16,
    b'l' as u16,
    b'1' as u16,
    b'-' as u16,
    b'1' as u16,
    b'-' as u16,
    b'6' as u16,
    b'.' as u16,
    b'd' as u16,
    b'l' as u16,
    b'l' as u16,
    0,
];

fn lookup_proc(name: &'static [u8]) -> *const c_void {
    let modules = [
        // SAFETY: the module names are valid, nul-terminated UTF-16 strings;
        // both system DLLs are loaded by every supported desktop Windows
        // process before this backend can be used.
        unsafe { GetModuleHandleW(KERNEL32.as_ptr()) },
        // SAFETY: see the first lookup; kernelbase owns the implementation on
        // Windows versions that forward the public kernel32 export.
        unsafe { GetModuleHandleW(KERNELBASE.as_ptr()) },
        // SAFETY: newer Windows exposes these entry points through the memory
        // API-set contract rather than directly from kernel32.
        unsafe { GetModuleHandleW(MEMORY_API_SET.as_ptr()) },
    ];
    for module in modules {
        if !module.is_null() {
            // SAFETY: `name` is a static nul-terminated ASCII symbol name and
            // `module` is a live system-module handle.
            let proc = unsafe { GetProcAddress(module, name.as_ptr()) };
            if !proc.is_null() {
                return proc;
            }
        }
    }
    core::ptr::null()
}

fn modern_apis() -> Option<(VirtualAlloc2Fn, MapViewOfFile3Fn)> {
    let virtual_alloc2 = lookup_proc(b"VirtualAlloc2\0");
    let map_view_of_file3 = lookup_proc(b"MapViewOfFile3\0");
    if virtual_alloc2.is_null() || map_view_of_file3.is_null() {
        return None;
    }

    // SAFETY: both addresses came from GetProcAddress for the exact exported
    // functions and are cast to their documented `extern "system"` ABIs.
    Some(unsafe {
        (
            core::mem::transmute::<*const c_void, VirtualAlloc2Fn>(virtual_alloc2),
            core::mem::transmute::<*const c_void, MapViewOfFile3Fn>(map_view_of_file3),
        )
    })
}

fn section(size: usize) -> Result<*mut c_void> {
    let size64 = size as u64;
    let high = (size64 >> 32) as u32;
    let low = size64 as u32;
    // SAFETY: INVALID_HANDLE_VALUE requests a pagefile-backed section; the
    // size is page- and allocation-granularity aligned and the remaining
    // arguments are null/default as required by CreateFileMappingW.
    let handle = unsafe {
        CreateFileMappingW(
            (-1isize) as *mut c_void,
            core::ptr::null_mut(),
            PAGE_READWRITE,
            high,
            low,
            core::ptr::null(),
        )
    };
    if handle.is_null() {
        Err(os_error())
    } else {
        Ok(handle)
    }
}

fn release_view(view: *mut u8) {
    // SAFETY: `view` is one of the two live views owned by a MirroredRing.
    unsafe {
        UnmapViewOfFile(view.cast::<c_void>());
    }
}

fn mirror_map_modern(
    size: usize,
    virtual_alloc2: VirtualAlloc2Fn,
    map_view_of_file3: MapViewOfFile3Fn,
) -> Result<Region> {
    let double = size.checked_mul(2).ok_or(VmError::InvalidArgument)?;
    // SAFETY: the size is page-aligned and the placeholder is only virtual
    // address space. NULL lets Windows choose a suitably aligned span.
    let placeholder1 = unsafe {
        virtual_alloc2(
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            double,
            MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
            PAGE_NOACCESS,
            core::ptr::null_mut(),
            0,
        )
    };
    if placeholder1.is_null() {
        return Err(os_error());
    }

    // Split the 2N placeholder into two exactly adjacent N placeholders.
    // SAFETY: `placeholder1`/`size` identify the first half of the placeholder
    // returned above; MEM_PRESERVE_PLACEHOLDER retains that half as a new
    // placeholder while splitting it from the second half.
    let split = unsafe { VirtualFree(placeholder1, size, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) };
    if split == 0 {
        // SAFETY: the full placeholder is still owned because the split failed.
        unsafe {
            VirtualFree(placeholder1, 0, MEM_RELEASE);
        }
        return Err(os_error());
    }
    let placeholder2 = placeholder1.wrapping_byte_add(size);
    let mapping = match section(size) {
        Ok(mapping) => mapping,
        Err(error) => {
            // SAFETY: both halves are still placeholders owned by this path.
            unsafe {
                VirtualFree(placeholder1, 0, MEM_RELEASE);
                VirtualFree(placeholder2, 0, MEM_RELEASE);
            }
            return Err(error);
        }
    };

    // SAFETY: each call replaces one exact N-byte placeholder with a view of
    // the same pagefile-backed section. The section offset is zero, so both
    // views alias the same physical pages.
    let first = unsafe {
        map_view_of_file3(
            mapping,
            core::ptr::null_mut(),
            placeholder1,
            0,
            size,
            MEM_REPLACE_PLACEHOLDER,
            PAGE_READWRITE,
            core::ptr::null_mut(),
            0,
        )
    };
    if first.is_null() || first != placeholder1 {
        if !first.is_null() {
            release_view(first.cast::<u8>());
        }
        // SAFETY: failed replacement leaves the corresponding placeholders
        // owned by this path; freeing the second is always required, and the
        // first is safe to release if the API returned a non-null mismatch.
        unsafe {
            VirtualFree(placeholder1, 0, MEM_RELEASE);
            VirtualFree(placeholder2, 0, MEM_RELEASE);
            CloseHandle(mapping);
        }
        return if first.is_null() {
            Err(os_error())
        } else {
            Err(VmError::SystemError(ERROR_INVALID_ADDRESS as i32))
        };
    }
    // SAFETY: the first view is already established and the second call uses
    // the exact adjacent placeholder; failure is handled without exposing a
    // partially mapped ring.
    let second = unsafe {
        map_view_of_file3(
            mapping,
            core::ptr::null_mut(),
            placeholder2,
            0,
            size,
            MEM_REPLACE_PLACEHOLDER,
            PAGE_READWRITE,
            core::ptr::null_mut(),
            0,
        )
    };
    if second.is_null() || second != placeholder2 {
        if !second.is_null() {
            release_view(second.cast::<u8>());
        }
        release_view(first.cast::<u8>());
        // SAFETY: the first view has been unmapped; a failed replacement leaves
        // the second placeholder owned by this path, including the defensive
        // non-null/mismatched-return case.
        unsafe {
            VirtualFree(placeholder2, 0, MEM_RELEASE);
            CloseHandle(mapping);
        }
        return if second.is_null() {
            Err(os_error())
        } else {
            Err(VmError::SystemError(ERROR_INVALID_ADDRESS as i32))
        };
    }
    // The section handle is no longer needed after both views are mapped.
    // SAFETY: `mapping` is the live section handle created above.
    unsafe {
        CloseHandle(mapping);
    }
    Ok(Region {
        base: first.cast::<u8>(),
        base_len: double,
        ptr: first.cast::<u8>(),
        len: size,
    })
}

fn mirror_map_legacy(size: usize) -> Result<Region> {
    let mapping = section(size)?;
    let access = FILE_MAP_READ | FILE_MAP_WRITE;
    let mut last_error = VmError::OutOfMemory;

    for _ in 0..MAP_RETRY_LIMIT {
        // SAFETY: mapping is a live pagefile-backed section; NULL lets Windows
        // choose the first view address, which is allocation-granularity aligned.
        let first = unsafe { MapViewOfFileEx(mapping, access, 0, 0, size, core::ptr::null_mut()) };
        if first.is_null() {
            last_error = os_error();
            break;
        }
        let second_address = first.wrapping_byte_add(size);
        // SAFETY: this asks Windows to map the same section at the exact
        // adjacent address. A failure is handled by unmapping the first view;
        // no guessed or unrelated address is ever returned.
        let second = unsafe { MapViewOfFileEx(mapping, access, 0, 0, size, second_address) };
        if second == second_address {
            // SAFETY: the section handle remains owned until both views have
            // been established and is then closed; views keep the section alive.
            unsafe {
                CloseHandle(mapping);
            }
            return Ok(Region {
                base: first.cast::<u8>(),
                base_len: size * 2,
                ptr: first.cast::<u8>(),
                len: size,
            });
        }
        if !second.is_null() {
            release_view(second.cast::<u8>());
            last_error = VmError::SystemError(ERROR_INVALID_ADDRESS as i32);
        } else {
            last_error = os_error();
        }
        release_view(first.cast::<u8>());
    }

    // SAFETY: no successful ring escaped this function; the section handle is
    // therefore still owned here.
    unsafe {
        CloseHandle(mapping);
    }
    Err(last_error)
}

/// Map a real mirrored ring. The capacity is rounded to Windows allocation
/// granularity because both exact views must start on that boundary.
pub(super) fn mirror_map(size: usize) -> Result<Region> {
    let size = round_up(size, allocation_granularity()).ok_or(VmError::InvalidArgument)?;
    if let Some((virtual_alloc2, map_view_of_file3)) = modern_apis() {
        mirror_map_modern(size, virtual_alloc2, map_view_of_file3)
    } else {
        mirror_map_legacy(size)
    }
}

/// Release both section views exactly once. The section handle was closed once
/// the views were established, so unmapping the two views is sufficient.
pub(super) fn mirror_release(region: Region) {
    release_view(region.ptr);
    release_view(region.ptr.wrapping_byte_add(region.len));
}
