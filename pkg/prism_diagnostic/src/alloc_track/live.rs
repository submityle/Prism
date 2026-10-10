//! Header-based [`GlobalAlloc`] wrapper delivering exact per-tag *live* bytes
//! (design §24.3).
//!
//! Unlike the header-free [`TrackingAllocator`](super::tracking::TrackingAllocator),
//! which can only attribute *cumulative* bytes to a tag (a `dealloc` is not told
//! which tag owned the block), [`LiveTrackingAllocator`] stamps every allocation
//! with a small header recording the owning tag. A later `dealloc` reads that
//! header and decrements the correct tag's live counter — even if the free runs
//! on a different thread or under a different (or no) tag scope.
//!
//! Memory layout of one allocation (`base` is the pointer returned by the inner
//! allocator; `user` is the pointer returned to the caller):
//!
//! ```text
//! base ──► [ Header ][ padding ][ user payload ... ]
//!          ^^^^^^^^^^^^^^^^^^^^^ prefix (multiple of layout.align())
//!                               ^ user = base + prefix
//! ```
//!
//! The header lives at `base`, which the inner allocator aligns to
//! `max(layout.align(), align_of::<Header>())`, so the header is always
//! correctly aligned. `prefix` is the smallest multiple of `layout.align()`
//! that is at least `size_of::<Header>()`, so `user` keeps the caller's
//! requested alignment and never overlaps the header. The deallocation path
//! recomputes `prefix` from the same `layout`, recovers `base = user - prefix`,
//! reads the header, and frees `base` with the matching block layout.
//!
//! `alloc_zeroed`, `realloc`, and friends intentionally use the default
//! [`GlobalAlloc`] implementations, which route through this type's `alloc` and
//! `dealloc`, so the header handling is applied uniformly without extra unsafe
//! code.

#![expect(
    unsafe_code,
    reason = "a GlobalAlloc that stamps a per-allocation header needs raw \
              pointer arithmetic and header read/write; every site is audited"
)]

use core::alloc::{GlobalAlloc, Layout};
use core::mem::{align_of, size_of};
use std::alloc::System;

use super::{
    current_tag_raw, record_alloc, record_free, record_tag_live_alloc, record_tag_live_free,
};

/// A per-allocation header stored immediately at the inner allocation's base.
#[repr(C)]
#[derive(Clone, Copy)]
struct Header {
    /// The tag index that owned this allocation (or `UNTAGGED`).
    tag: usize,
    /// Sentinel used to detect corruption / mismatched frees in debug builds.
    magic: usize,
}

/// Sentinel stamped into every [`Header`] (`"PRSMLV"` as a nibble pattern).
const MAGIC: usize = 0x5052_534D_4C56_3031;

/// Round `value` up to the next multiple of `align` (a power of two), returning
/// `None` on overflow.
#[inline]
const fn round_up(value: usize, align: usize) -> Option<usize> {
    match value.checked_add(align - 1) {
        Some(sum) => Some(sum & !(align - 1)),
        None => None,
    }
}

/// Compute the inner block [`Layout`] and the `base`→`user` prefix offset for a
/// caller `layout`. Returns `None` only on arithmetic overflow (treated as an
/// allocation failure). Deterministic in `layout`, so `alloc` and `dealloc`
/// agree.
#[inline]
fn block_layout(layout: Layout) -> Option<(Layout, usize)> {
    let header_size = size_of::<Header>();
    let header_align = align_of::<Header>();
    let align = layout.align();
    let block_align = if align > header_align {
        align
    } else {
        header_align
    };
    // Smallest multiple of `align` that still leaves room for the header.
    let prefix = round_up(header_size, align)?;
    let block_size = prefix.checked_add(layout.size())?;
    match Layout::from_size_align(block_size, block_align) {
        Ok(block) => Some((block, prefix)),
        Err(_) => None,
    }
}

/// A [`GlobalAlloc`] wrapper that tracks exact per-tag **live** residency by
/// stamping each allocation with the owning tag.
///
/// It feeds the same shared counters as
/// [`TrackingAllocator`](super::tracking::TrackingAllocator) (so
/// [`snapshot`](super::snapshot) is unaffected) and additionally populates the
/// `live_bytes` field of [`tag_report`](super::tag_report). Prefer the
/// header-free allocator when per-tag live bytes are not needed; this one adds
/// a small, aligned header per allocation and uses copy-based `realloc`.
///
/// Install it as the program allocator:
///
/// ```ignore
/// use prism_diagnostic::alloc_track::LiveTrackingAllocator;
/// use std::alloc::System;
///
/// #[global_allocator]
/// static GLOBAL: LiveTrackingAllocator<System> = LiveTrackingAllocator::new(System);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct LiveTrackingAllocator<A: GlobalAlloc = System> {
    inner: A,
}

impl<A: GlobalAlloc> LiveTrackingAllocator<A> {
    /// Wrap `inner`, accounting all traffic routed through it with per-tag live
    /// tracking.
    pub const fn new(inner: A) -> Self {
        Self { inner }
    }

    /// Borrow the wrapped allocator.
    pub fn inner(&self) -> &A {
        &self.inner
    }
}

// SAFETY: `alloc` returns `base + prefix` where `base` comes from a successful
// inner allocation of `block_layout(layout)` and `prefix` is a multiple of
// `layout.align()`; since `base` is aligned to `max(align, align_of::<Header>())`
// the returned pointer satisfies the caller's requested alignment and the
// `layout.size()` payload bytes `[user, user + size)` lie within the block.
// `dealloc` recomputes the identical `prefix`, recovers the same `base`, and
// frees it with the identical `block_layout`, so every pointer handed to the
// inner allocator is one it produced with its matching layout. The accounting
// only touches atomics; the header read/write touches memory owned by this
// allocation. The default `alloc_zeroed`/`realloc` route through these two
// methods, preserving the invariants.
unsafe impl<A: GlobalAlloc> GlobalAlloc for LiveTrackingAllocator<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let Some((block, prefix)) = block_layout(layout) else {
            return core::ptr::null_mut();
        };
        // SAFETY: `block` is a valid non-zero layout (size >= prefix >=
        // size_of::<Header>() > 0); forwarded to the inner allocator.
        let base = unsafe { self.inner.alloc(block) };
        if base.is_null() {
            return base;
        }
        let tag = current_tag_raw();
        // SAFETY: `base` is aligned to `block.align() >= align_of::<Header>()`
        // and the block reserves `prefix >= size_of::<Header>()` leading bytes,
        // so writing a `Header` at `base` stays in-bounds and well-aligned.
        unsafe { base.cast::<Header>().write(Header { tag, magic: MAGIC }) };
        record_alloc(layout.size());
        record_tag_live_alloc(tag, layout.size());
        // SAFETY: `prefix <= block.size()`, so `base + prefix` is within the
        // allocation (one-past-the-end at most) and is the user payload start.
        unsafe { base.add(prefix) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let Some((block, prefix)) = block_layout(layout) else {
            // Unreachable: `alloc` succeeded with this `layout`, so
            // `block_layout` was `Some` then and is deterministic in `layout`.
            debug_assert!(false, "alloc_track live: block_layout mismatch on dealloc");
            return;
        };
        // SAFETY: `ptr == base + prefix` from the matching `alloc`, so
        // `ptr - prefix` recovers the original inner-allocation base.
        let base = unsafe { ptr.sub(prefix) };
        // SAFETY: `base` holds a `Header` written by `alloc`, correctly aligned
        // and initialized; reading it back is valid.
        let header = unsafe { base.cast::<Header>().read() };
        debug_assert_eq!(header.magic, MAGIC, "alloc_track live: corrupted header");
        // SAFETY: `base`/`block` are exactly what the inner allocator returned
        // and the layout it was allocated with; forwarded to free it.
        unsafe { self.inner.dealloc(base, block) };
        record_free(layout.size());
        record_tag_live_free(header.tag, layout.size());
    }
}

#[cfg(test)]
mod tests {
    use super::super::{register_tag, tag_report, tag_scope};
    use super::*;

    fn live_bytes_for(name: &'static str) -> u64 {
        tag_report()
            .into_iter()
            .find(|s| s.name == name)
            .map(|s| s.live_bytes)
            .unwrap_or(0)
    }

    #[test]
    fn alloc_returns_aligned_in_bounds_pointer() {
        let alloc = LiveTrackingAllocator::new(System);
        for &align in &[1usize, 2, 4, 8, 16, 32, 64, 128, 256] {
            for &size in &[1usize, 7, 64, 4096] {
                let layout = Layout::from_size_align(size, align).unwrap();
                // SAFETY: non-zero layout; freed below with the same layout.
                let ptr = unsafe { alloc.alloc(layout) };
                assert!(!ptr.is_null(), "alloc failed for {layout:?}");
                assert_eq!(ptr as usize % align, 0, "misaligned for {layout:?}");
                // Touch every payload byte to prove the region is writable.
                // SAFETY: `[ptr, ptr+size)` is the live payload of this block.
                unsafe { core::ptr::write_bytes(ptr, 0xAB, size) };
                // SAFETY: same block, same layout as the allocation above.
                unsafe { alloc.dealloc(ptr, layout) };
            }
        }
    }

    #[test]
    fn per_tag_live_bytes_round_trip() {
        let name = "prism::test::live::round_trip";
        let tag = register_tag(name).expect("tag slot");
        let alloc = LiveTrackingAllocator::new(System);
        let start = live_bytes_for(name);

        let layout = Layout::from_size_align(4096, 16).unwrap();
        let ptr = {
            let _scope = tag_scope(tag);
            // SAFETY: non-zero layout; freed below.
            unsafe { alloc.alloc(layout) }
        };
        assert!(!ptr.is_null());
        assert_eq!(
            live_bytes_for(name),
            start + 4096,
            "tagged alloc must raise live bytes"
        );

        // Free happens with NO active scope — the header must still attribute
        // the free to the original tag.
        // SAFETY: same block/layout as the allocation above.
        unsafe { alloc.dealloc(ptr, layout) };
        assert_eq!(
            live_bytes_for(name),
            start,
            "free must return live bytes to the baseline via the header tag"
        );
    }

    #[test]
    fn untagged_alloc_does_not_touch_any_tag() {
        let name = "prism::test::live::untagged";
        let _tag = register_tag(name).expect("tag slot");
        let alloc = LiveTrackingAllocator::new(System);
        let before = live_bytes_for(name);
        let layout = Layout::from_size_align(2048, 8).unwrap();
        // No tag scope active here.
        // SAFETY: non-zero layout; freed below.
        let ptr = unsafe { alloc.alloc(layout) };
        assert!(!ptr.is_null());
        // SAFETY: same block/layout.
        unsafe { alloc.dealloc(ptr, layout) };
        assert_eq!(live_bytes_for(name), before);
    }

    #[test]
    fn interleaved_tags_track_independently() {
        let a_name = "prism::test::live::interleave_a";
        let b_name = "prism::test::live::interleave_b";
        let a = register_tag(a_name).expect("tag slot");
        let b = register_tag(b_name).expect("tag slot");
        let alloc = LiveTrackingAllocator::new(System);
        let (sa, sb) = (live_bytes_for(a_name), live_bytes_for(b_name));

        let la = Layout::from_size_align(1024, 16).unwrap();
        let lb = Layout::from_size_align(512, 32).unwrap();
        let pa = {
            let _s = tag_scope(a);
            // SAFETY: non-zero layout.
            unsafe { alloc.alloc(la) }
        };
        let pb = {
            let _s = tag_scope(b);
            // SAFETY: non-zero layout.
            unsafe { alloc.alloc(lb) }
        };
        assert_eq!(live_bytes_for(a_name), sa + 1024);
        assert_eq!(live_bytes_for(b_name), sb + 512);

        // SAFETY: matching block/layout; order independent of scope.
        unsafe { alloc.dealloc(pa, la) };
        assert_eq!(live_bytes_for(a_name), sa);
        assert_eq!(live_bytes_for(b_name), sb + 512);
        // SAFETY: matching block/layout.
        unsafe { alloc.dealloc(pb, lb) };
        assert_eq!(live_bytes_for(b_name), sb);
    }

    #[test]
    fn realloc_adjusts_live_bytes() {
        let name = "prism::test::live::realloc";
        let tag = register_tag(name).expect("tag slot");
        let alloc = LiveTrackingAllocator::new(System);
        let start = live_bytes_for(name);
        let layout = Layout::from_size_align(256, 8).unwrap();

        let (ptr, grown) = {
            let _scope = tag_scope(tag);
            // SAFETY: non-zero layout.
            let p = unsafe { alloc.alloc(layout) };
            assert_eq!(live_bytes_for(name), start + 256);
            // Default realloc routes through alloc+dealloc; the new block is
            // allocated under the same scope, the old one freed via its header.
            // SAFETY: `p`/`layout` are current; new size is valid.
            let g = unsafe { alloc.realloc(p, layout, 1024) };
            (g, 1024usize)
        };
        assert!(!ptr.is_null());
        assert_eq!(
            live_bytes_for(name),
            start + grown as u64,
            "after grow, only the new block is live"
        );
        let new_layout = Layout::from_size_align(grown, 8).unwrap();
        // SAFETY: `ptr` is the reallocated block with `new_layout`.
        unsafe { alloc.dealloc(ptr, new_layout) };
        assert_eq!(live_bytes_for(name), start);
    }
}
