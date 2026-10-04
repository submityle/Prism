//! Unit tests for the M3 virtual-memory layer.
//!
//! These exercise the real OS backend on the host they run on: reserve /
//! commit / write / read / protect / decommit / release round-trips, aligned
//! reservations, guard-page installation, honest huge-page capability
//! reporting, memory-info sanity, and argument validation.
#![expect(
    unsafe_code,
    reason = "tests write and read through committed VM pages via raw pointers to prove commit/protect actually work"
)]

use super::{
    huge_pages_supported, large_page_size, memory_info, page_size, virtual_memory_supported,
    Protection, Reservation, VmError,
};

#[test]
fn page_size_is_sane() {
    let ps = page_size();
    assert!(ps >= 4096, "page size should be at least 4 KiB, got {ps}");
    assert!(
        ps.is_power_of_two(),
        "page size {ps} must be a power of two"
    );
}

#[test]
fn memory_info_is_sane() {
    let info = memory_info().expect("desktop hosts expose memory info");
    assert!(
        info.total_physical > 0,
        "total physical memory must be positive"
    );
    assert!(
        info.available_physical > 0,
        "available physical memory should be positive on a live host"
    );
    assert!(
        info.available_physical <= info.total_physical,
        "available ({}) must not exceed total ({})",
        info.available_physical,
        info.total_physical
    );
    assert!(info.page_size.is_power_of_two());
    assert_eq!(info.page_size, page_size());
    if let Some(lps) = info.large_page_size {
        assert!(lps >= info.page_size, "large page must be >= base page");
        assert!(lps.is_power_of_two());
    }
}

#[test]
fn virtual_memory_is_supported_on_this_host() {
    // Linux / macOS / Windows all have a working backend; the fallback only
    // kicks in on exotic targets not exercised by this test binary.
    assert!(virtual_memory_supported());
}

#[test]
fn reserve_commit_write_read_protect_decommit() {
    let ps = page_size();
    let pages = 4;
    let len = ps * pages;
    let res = Reservation::reserve(len).expect("reserve should succeed");
    assert_eq!(res.len(), len);
    assert!(!res.is_empty());
    assert!(!res.as_ptr().is_null());

    // Commit the whole range read-write and prove we can touch every page.
    res.commit(0, len, Protection::ReadWrite)
        .expect("commit should succeed");
    let ptr = res.as_mut_ptr();
    // SAFETY: `[ptr, ptr + len)` was just committed read-write and is owned by
    // `res` for the duration of this borrow.
    let bytes = unsafe { core::slice::from_raw_parts_mut(ptr, len) };
    for (i, b) in bytes.iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    for (i, b) in bytes.iter().enumerate() {
        assert_eq!(*b, (i % 251) as u8, "written byte at {i} must read back");
    }

    // Re-protect the first page read-only; the call must succeed.
    res.protect(0, ps, Protection::Read)
        .expect("protect to read-only should succeed");
    // SAFETY: the first page is still readable; we only read from it.
    let first = unsafe { core::slice::from_raw_parts(res.as_ptr(), ps) };
    assert_eq!(first[0], 0);

    // Decommit the back half, returning its physical pages.
    res.decommit(ps * 2, ps * 2)
        .expect("decommit should succeed");

    // Re-committing the decommitted range works and reads back zero.
    res.commit(ps * 2, ps * 2, Protection::ReadWrite)
        .expect("re-commit should succeed");
    // SAFETY: `[ptr + 2*ps, ...)` was just re-committed read-write.
    let tail = unsafe { core::slice::from_raw_parts(res.as_ptr().add(ps * 2), ps * 2) };
    assert_eq!(tail[0], 0, "freshly re-committed memory reads back zero");

    // Dropping `res` releases the whole mapping.
    drop(res);
}

#[test]
fn aligned_reservation_is_aligned() {
    let ps = page_size();
    // Request a large alignment well beyond the page size.
    let align = ps * 16;
    let len = ps * 2;
    let res = Reservation::reserve_aligned(len, align).expect("aligned reserve should succeed");
    assert_eq!(res.len(), len);
    assert_eq!(
        res.as_ptr().addr() % align,
        0,
        "base {:p} is not aligned to {align}",
        res.as_ptr()
    );
    // The aligned region must be usable.
    res.commit(0, len, Protection::ReadWrite)
        .expect("commit of aligned region should succeed");
    // SAFETY: whole aligned region committed read-write.
    let bytes = unsafe { core::slice::from_raw_parts_mut(res.as_mut_ptr(), len) };
    bytes[0] = 0xAB;
    bytes[len - 1] = 0xCD;
    assert_eq!(bytes[0], 0xAB);
    assert_eq!(bytes[len - 1], 0xCD);
}

#[test]
fn small_alignment_falls_back_to_plain_reserve() {
    let ps = page_size();
    // Alignment <= page size is always satisfied by a plain mapping.
    let res = Reservation::reserve_aligned(ps, ps).expect("reserve should succeed");
    assert_eq!(res.as_ptr().addr() % ps, 0);
}

#[test]
fn guard_page_installs_without_error() {
    let ps = page_size();
    let len = ps * 3;
    let res = Reservation::reserve(len).expect("reserve should succeed");
    // Commit the middle data page; keep page 0 and page 2 as guard candidates.
    res.commit(ps, ps, Protection::ReadWrite)
        .expect("commit middle page");
    // A guard page is a no-access page bracketing the live region; installing
    // it must succeed. (Verifying the fault itself needs a signal handler and
    // is out of scope for a unit test.)
    res.guard_page(0).expect("front guard page should install");
    res.guard_page(ps * 2)
        .expect("back guard page should install");
}

#[test]
fn huge_pages_capability_is_honest() {
    let supported = huge_pages_supported();
    if supported {
        assert!(
            large_page_size().is_some(),
            "if huge pages are supported a large-page size must be reported"
        );
        let lps = large_page_size().unwrap();
        // Attempt a real huge reservation. Runtime policy (empty pool / missing
        // privilege) may still deny it; such denials are OutOfMemory or
        // SystemError, never a silent success or Unsupported.
        match Reservation::reserve_huge(lps) {
            Ok(res) => {
                assert!(res.is_huge());
                assert_eq!(res.len() % lps, 0);
                // SAFETY: huge reservations are committed read-write up front.
                let bytes = unsafe { core::slice::from_raw_parts_mut(res.as_mut_ptr(), lps) };
                bytes[0] = 1;
                bytes[lps - 1] = 2;
                assert_eq!(bytes[0], 1);
                assert_eq!(bytes[lps - 1], 2);
            }
            Err(e) => assert!(
                matches!(e, VmError::OutOfMemory | VmError::SystemError(_)),
                "a supported-but-denied huge reservation must be OOM/SystemError, got {e:?}"
            ),
        }
    } else {
        // Unsupported platforms (macOS, wasm) must report no large-page size
        // and reject huge reservations with Unsupported.
        assert!(large_page_size().is_none());
        assert!(matches!(
            Reservation::reserve_huge(1 << 20),
            Err(VmError::Unsupported)
        ));
    }
}

#[test]
fn invalid_arguments_are_rejected() {
    let ps = page_size();
    assert!(matches!(
        Reservation::reserve(0),
        Err(VmError::InvalidArgument)
    ));
    // Non-power-of-two alignment.
    assert!(matches!(
        Reservation::reserve_aligned(ps, 3),
        Err(VmError::InvalidArgument)
    ));

    let res = Reservation::reserve(ps * 2).expect("reserve should succeed");
    // Misaligned offset.
    assert_eq!(
        res.commit(1, ps, Protection::ReadWrite),
        Err(VmError::InvalidArgument)
    );
    // Misaligned length.
    assert_eq!(
        res.commit(0, 1, Protection::ReadWrite),
        Err(VmError::InvalidArgument)
    );
    // Zero length.
    assert_eq!(
        res.protect(0, 0, Protection::Read),
        Err(VmError::InvalidArgument)
    );
    // Out of range.
    assert_eq!(
        res.commit(0, ps * 3, Protection::ReadWrite),
        Err(VmError::InvalidArgument)
    );
    // Guard page past the end.
    assert_eq!(res.guard_page(ps * 2), Err(VmError::InvalidArgument));
}

#[test]
fn reservation_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Reservation>();

    // Move a reservation into another thread and use it there.
    let ps = page_size();
    let res = Reservation::reserve(ps).expect("reserve should succeed");
    let handle = std::thread::spawn(move || {
        res.commit(0, ps, Protection::ReadWrite)
            .expect("commit on worker thread");
        // SAFETY: committed read-write on this thread.
        let bytes = unsafe { core::slice::from_raw_parts_mut(res.as_mut_ptr(), ps) };
        bytes[0] = 7;
        bytes[0]
    });
    assert_eq!(handle.join().unwrap(), 7);
}

#[test]
fn mirrored_ring_capability_is_honest() {
    use super::MirroredRing;
    // On an unsupported platform the constructor must honestly refuse rather
    // than hand back a non-mirroring buffer.
    if !MirroredRing::is_supported() {
        assert!(matches!(
            MirroredRing::with_min_capacity(page_size()),
            Err(VmError::Unsupported)
        ));
    }
}

#[test]
fn mirrored_ring_aliases_both_halves() {
    use super::MirroredRing;
    if !MirroredRing::is_supported() {
        return; // Honest skip on platforms without a mirroring backend.
    }
    let ps = page_size();
    let ring = MirroredRing::with_min_capacity(1).expect("one-page ring should map");
    let cap = ring.capacity();
    assert_eq!(cap, ps, "capacity rounds up to a whole page");
    assert!(!ring.is_empty());
    assert!(!ring.as_ptr().is_null());

    let base = ring.as_mut_ptr();
    // SAFETY: the mapping is valid for reads and writes across the full `2*cap`
    // span and owned by `ring` for the duration of this borrow. The upper half
    // aliases the lower half.
    let full = unsafe { core::slice::from_raw_parts_mut(base, 2 * cap) };

    // A write into the lower half is visible in the mirror at `cap + i`.
    for (i, b) in full[..cap].iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    for i in 0..cap {
        assert_eq!(
            full[cap + i],
            (i % 251) as u8,
            "mirror byte at {} must alias lower byte {i}",
            cap + i
        );
    }

    // A write into the upper (mirror) half is visible in the lower half: this is
    // exactly the wrap a ring producer relies on when a record straddles the end.
    full[cap] = 0x5A;
    full[2 * cap - 1] = 0xA5;
    assert_eq!(full[0], 0x5A, "write at cap wraps to offset 0");
    assert_eq!(full[cap - 1], 0xA5, "write at 2*cap-1 wraps to cap-1");
}

#[test]
fn mirrored_ring_rejects_zero_capacity() {
    use super::MirroredRing;
    assert!(matches!(
        MirroredRing::with_min_capacity(0),
        Err(VmError::InvalidArgument)
    ));
}

#[test]
fn mirrored_ring_is_send_and_sync() {
    use super::MirroredRing;
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<MirroredRing>();

    if !MirroredRing::is_supported() {
        return;
    }
    let ps = page_size();
    let ring = MirroredRing::with_min_capacity(ps).expect("ring should map");
    let handle = std::thread::spawn(move || {
        let cap = ring.capacity();
        // SAFETY: valid for the full mirrored span; owned by this thread now.
        let full = unsafe { core::slice::from_raw_parts_mut(ring.as_mut_ptr(), 2 * cap) };
        full[0] = 0x33;
        full[cap] // reads the alias of offset 0
    });
    assert_eq!(handle.join().unwrap(), 0x33);
}
