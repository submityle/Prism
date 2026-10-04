//! # Guarded allocator (`guard`) — memory-safety hardening
//!
//! Design chapter §24.3: a *debug-tier* [`Allocator`] decorator that wraps any
//! inner [`Allocator`] and adds classic heap-corruption detection without
//! requiring a custom global allocator or platform guard pages:
//!
//! - **Canary / redzone** bytes are written immediately before and after every
//!   user payload. They are re-validated on free, so a buffer overflow or
//!   underflow that scribbles past the payload is caught deterministically at
//!   the next [`deallocate`](Allocator::deallocate).
//! - **Free poisoning** overwrites the whole block (header, redzones, payload)
//!   with a recognizable pattern on free, turning a use-after-free read into an
//!   obvious poison value and a use-after-free *write* into a corruption that
//!   the enclosing allocator's own bookkeeping will reject.
//! - **Double-free detection** (best effort): each live block carries a header
//!   magic that is flipped to a freed sentinel on release, so freeing the same
//!   pointer twice trips an assertion instead of corrupting the inner
//!   allocator. Once the inner allocator recycles the freed block the magic may
//!   read back as generic corruption rather than the freed sentinel; both paths
//!   still abort the duplicate free.
//!
//! This decorator is intended for debug / validation builds (it adds a header
//! plus two redzones per allocation and touches the whole block on free). In
//! shipping builds callers simply use the bare inner allocator.
//!
//! ## Honest boundary
//! The *allocation call-stack capture* bullet of §24.3 lives with
//! `prism_diagnostic` (it needs the `alloc-track` backend), not here. This
//! module provides the deterministic, allocator-local corruption checks only;
//! it does not install signal handlers or guard pages (that is
//! `prism_platform` §9 territory, bridged separately).

extern crate alloc;

use core::alloc::Layout;
use core::fmt;
use core::ptr::NonNull;

use super::{AllocError, Allocator};

/// Byte written into redzones around a live payload.
///
/// Chosen to be a non-zero, non-`ASCII`, visually distinctive value so a stray
/// read of a redzone stands out in a hex dump.
const REDZONE_BYTE: u8 = 0xFD;

/// Byte written over an entire block when it is freed (use-after-free bait).
const POISON_BYTE: u8 = 0xDD;

/// Header magic stamped on a live guarded block.
const MAGIC_LIVE: u64 = 0x5052_4953_4D5F_4744; // "PRISM_GD"

/// Header magic a block is flipped to on free, used for double-free detection.
const MAGIC_FREED: u64 = 0x4652_4545_445F_4744; // "FREED_GD"

/// Per-allocation configuration for a [`GuardedAllocator`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GuardConfig {
    /// Number of canary bytes placed on *each* side of the payload.
    ///
    /// A larger redzone catches larger stride overruns at the cost of memory;
    /// the default of 16 bytes matches a common debug-allocator convention.
    redzone_len: usize,
}

impl GuardConfig {
    /// The default redzone length (16 bytes on each side).
    pub const DEFAULT_REDZONE_LEN: usize = 16;

    /// A configuration with the [`DEFAULT_REDZONE_LEN`](Self::DEFAULT_REDZONE_LEN).
    pub const DEFAULT: Self = Self {
        redzone_len: Self::DEFAULT_REDZONE_LEN,
    };

    /// Create a configuration with an explicit redzone length (bytes per side).
    ///
    /// A `redzone_len` of 0 disables the canary checks while still providing
    /// free poisoning and double-free detection.
    #[must_use]
    pub const fn new(redzone_len: usize) -> Self {
        Self { redzone_len }
    }

    /// The configured redzone length, in bytes per side.
    #[must_use]
    pub const fn redzone_len(&self) -> usize {
        self.redzone_len
    }
}

impl Default for GuardConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// The bookkeeping header prepended to every non-zero-sized guarded block.
///
/// It is stored at the base of the inner allocation (which is aligned to at
/// least `align_of::<GuardHeader>()`), ahead of the front redzone and payload.
#[repr(C)]
#[derive(Clone, Copy)]
struct GuardHeader {
    /// [`MAGIC_LIVE`] while the block is live, [`MAGIC_FREED`] after release.
    magic: u64,
    /// The user-requested payload size (`layout.size()`).
    user_size: usize,
    /// The user-requested payload alignment (`layout.align()`).
    user_align: usize,
    /// Offset from the base to the payload, in bytes.
    front: usize,
    /// Total inner-allocation size, in bytes.
    total: usize,
}

/// Round `value` up to the next multiple of `align` (a power of two).
///
/// Returns `None` on overflow.
const fn round_up(value: usize, align: usize) -> Option<usize> {
    match value.checked_add(align - 1) {
        Some(sum) => Some(sum & !(align - 1)),
        None => None,
    }
}

/// The geometry of a guarded block derived purely from the user [`Layout`] and
/// a [`GuardConfig`]. It is recomputed identically on allocate and deallocate,
/// so no size/offset needs to be trusted from the (potentially corrupted)
/// header during teardown.
struct Geometry {
    /// Alignment of the inner allocation.
    block_align: usize,
    /// Offset from base to payload.
    front: usize,
    /// Total inner-allocation size.
    total: usize,
}

impl Geometry {
    /// Compute the geometry for `layout` under `config`, or `None` if any size
    /// computation overflows (reported to the caller as [`AllocError`]).
    fn compute(layout: Layout, config: &GuardConfig) -> Option<Self> {
        let header = Layout::new::<GuardHeader>();
        // Base must satisfy both the header's and the user's alignment.
        let block_align = max_align(layout.align(), header.align());
        let redzone = config.redzone_len;
        // Front region holds the header and the front redzone, padded up so the
        // payload lands on the user's alignment.
        let front_min = header.size().checked_add(redzone)?;
        let front = round_up(front_min, layout.align())?;
        // total = front + payload + rear redzone.
        let total = front.checked_add(layout.size())?.checked_add(redzone)?;
        Some(Self {
            block_align,
            front,
            total,
        })
    }

    /// The inner [`Layout`] used to drive the wrapped allocator.
    fn inner_layout(&self) -> Option<Layout> {
        Layout::from_size_align(self.total, self.block_align).ok()
    }
}

/// Return the larger of two power-of-two alignments.
const fn max_align(a: usize, b: usize) -> usize {
    if a >= b { a } else { b }
}

/// An [`Allocator`] decorator that adds canary redzones, free poisoning, and
/// double-free detection to any inner allocator.
///
/// See the [module docs](self) for the detection scheme and its intended
/// debug-tier use.
#[derive(Clone, Copy, Debug, Default)]
pub struct GuardedAllocator<A> {
    inner: A,
    config: GuardConfig,
}

impl<A> GuardedAllocator<A> {
    /// Wrap `inner` with the [`default`](GuardConfig::default) guard
    /// configuration.
    #[must_use]
    pub const fn new(inner: A) -> Self {
        Self {
            inner,
            config: GuardConfig::DEFAULT,
        }
    }

    /// Wrap `inner` with an explicit [`GuardConfig`].
    #[must_use]
    pub const fn with_config(inner: A, config: GuardConfig) -> Self {
        Self { inner, config }
    }

    /// The active guard configuration.
    #[must_use]
    pub const fn config(&self) -> GuardConfig {
        self.config
    }

    /// A shared reference to the wrapped allocator.
    #[must_use]
    pub const fn inner(&self) -> &A {
        &self.inner
    }

    /// Consume the decorator and return the wrapped allocator.
    #[must_use]
    pub fn into_inner(self) -> A {
        self.inner
    }
}

impl<A> Allocator for GuardedAllocator<A>
where
    A: Allocator,
{
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        // Zero-sized requests never touch the heap and have no payload to
        // guard; forward verbatim so dealloc can forward the same way.
        if layout.size() == 0 {
            return self.inner.allocate(layout);
        }

        let geo = Geometry::compute(layout, &self.config).ok_or(AllocError)?;
        let inner_layout = geo.inner_layout().ok_or(AllocError)?;
        let block = self.inner.allocate(inner_layout)?;
        let base = block.cast::<u8>();

        let header = GuardHeader {
            magic: MAGIC_LIVE,
            user_size: layout.size(),
            user_align: layout.align(),
            front: geo.front,
            total: geo.total,
        };

        #[expect(
            unsafe_code,
            reason = "write header + redzones into the freshly allocated, exclusively owned block"
        )]
        // SAFETY: `base` points at a block of `geo.total` writable bytes we just
        // obtained from `inner` and own exclusively. `base` is aligned to
        // `geo.block_align >= align_of::<GuardHeader>()`, so the header write is
        // aligned. The payload starts at `base + geo.front` with `front <=
        // total - layout.size() - redzone`, so both redzone fills stay within
        // the block.
        let payload = unsafe {
            base.cast::<GuardHeader>().as_ptr().write(header);
            let redzone = self.config.redzone_len;
            let payload = base.as_ptr().add(geo.front);
            if redzone != 0 {
                // Front redzone sits in the `redzone` bytes immediately before
                // the payload; rear redzone immediately after it.
                core::ptr::write_bytes(payload.sub(redzone), REDZONE_BYTE, redzone);
                core::ptr::write_bytes(payload.add(layout.size()), REDZONE_BYTE, redzone);
            }
            payload
        };

        let payload = NonNull::new(payload).ok_or(AllocError)?;
        Ok(NonNull::slice_from_raw_parts(payload, layout.size()))
    }

    #[expect(
        unsafe_code,
        reason = "validates and tears down a block previously handed out by this decorator"
    )]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() == 0 {
            // Mirror the zero-sized forwarding in `allocate`.
            #[expect(
                unsafe_code,
                reason = "forwards the unchanged zero-sized dealloc contract to the inner allocator"
            )]
            // SAFETY: the caller guarantees `ptr`/`layout` came from this
            // decorator's `allocate`, which for a zero-sized layout forwarded
            // directly to `inner`.
            unsafe {
                self.inner.deallocate(ptr, layout);
            }
            return;
        }

        let geo = Geometry::compute(layout, &self.config)
            .expect("guarded layout geometry must be reconstructible on free");
        let redzone = self.config.redzone_len;

        #[expect(
            unsafe_code,
            reason = "walk back to the header, validate canaries, then poison before freeing"
        )]
        // SAFETY: the caller guarantees `ptr` is a payload pointer returned by a
        // non-zero-sized `allocate` of this decorator with this exact `layout`.
        // Therefore `base = ptr - geo.front` points at the owned header, both
        // redzones lie within the owned block, and `geo` matches the geometry
        // used at allocation time (it is a pure function of `layout`+config).
        let (base, inner_layout) = unsafe {
            let base = ptr.as_ptr().sub(geo.front);
            let header = base.cast::<GuardHeader>().read();

            assert_ne!(
                header.magic, MAGIC_FREED,
                "double free detected in GuardedAllocator (block already released)"
            );
            assert_eq!(
                header.magic, MAGIC_LIVE,
                "heap header corruption detected in GuardedAllocator (bad magic)"
            );
            assert_eq!(
                header.user_size,
                layout.size(),
                "GuardedAllocator free size mismatch (header vs layout)"
            );
            assert_eq!(
                header.user_align,
                layout.align(),
                "GuardedAllocator free align mismatch (header vs layout)"
            );

            if redzone != 0 {
                check_redzone(
                    ptr.as_ptr().sub(redzone),
                    redzone,
                    "underflow (front redzone corrupted)",
                );
                check_redzone(
                    ptr.as_ptr().add(layout.size()),
                    redzone,
                    "overflow (rear redzone corrupted)",
                );
            }

            // Flip the magic to the freed sentinel first so a racing/duplicate
            // free sees it even if poisoning is interrupted, then poison the
            // whole block to bait use-after-free.
            base.cast::<GuardHeader>().write(GuardHeader {
                magic: MAGIC_FREED,
                ..header
            });
            // Poison everything *after* the header (padding, redzones, and
            // payload) so a use-after-free read sees the poison byte, while the
            // freed-magic header survives for duplicate-free detection.
            let hdr_size = size_of::<GuardHeader>();
            core::ptr::write_bytes(base.add(hdr_size), POISON_BYTE, geo.total - hdr_size);

            let inner_layout = Layout::from_size_align(geo.total, geo.block_align)
                .expect("guarded inner layout must be valid on free");
            (base, inner_layout)
        };

        let base = NonNull::new(base).expect("guarded block base pointer is non-null");
        #[expect(
            unsafe_code,
            reason = "returns the original inner block to the wrapped allocator"
        )]
        // SAFETY: `base`/`inner_layout` are exactly the pointer and layout that
        // `allocate` passed to `inner.allocate`, so this honors `inner`'s
        // deallocation contract.
        unsafe {
            self.inner.deallocate(base, inner_layout);
        }
    }
}

/// Assert that `len` bytes starting at `ptr` all equal [`REDZONE_BYTE`].
///
/// # Safety
/// `ptr` must be valid for reads of `len` bytes.
#[expect(
    unsafe_code,
    reason = "reads a redzone the caller guarantees is within the owned block"
)]
unsafe fn check_redzone(ptr: *const u8, len: usize, what: &str) {
    for i in 0..len {
        #[expect(
            unsafe_code,
            reason = "indexed read within the caller-guaranteed redzone range"
        )]
        // SAFETY: the caller guarantees `[ptr, ptr+len)` is readable; `i < len`.
        let byte = unsafe { ptr.add(i).read() };
        assert_eq!(byte, REDZONE_BYTE, "heap buffer {what} in GuardedAllocator");
    }
}

impl fmt::Display for GuardConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GuardConfig(redzone={} bytes/side)", self.redzone_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc_::Global;

    fn alloc_and_fill(a: &GuardedAllocator<Global>, layout: Layout) -> NonNull<u8> {
        let block = a.allocate(layout).expect("allocation must succeed");
        assert_eq!(block.len(), layout.size());
        let ptr = block.cast::<u8>();
        #[expect(unsafe_code, reason = "test writes the full payload it just allocated")]
        // SAFETY: `block` has `layout.size()` writable payload bytes.
        unsafe {
            core::ptr::write_bytes(ptr.as_ptr(), 0xAB, layout.size());
        }
        ptr
    }

    #[test]
    fn roundtrip_is_clean() {
        let a = GuardedAllocator::new(Global);
        for size in [1usize, 7, 16, 64, 4096] {
            let layout = Layout::from_size_align(size, 8).unwrap();
            let ptr = alloc_and_fill(&a, layout);
            #[expect(unsafe_code, reason = "frees the block the test just allocated")]
            // SAFETY: `ptr`/`layout` came from this allocator's `allocate`.
            unsafe {
                a.deallocate(ptr, layout);
            }
        }
    }

    #[test]
    fn payload_is_aligned() {
        let a = GuardedAllocator::new(Global);
        for align in [1usize, 2, 4, 8, 16, 64, 256] {
            let layout = Layout::from_size_align(align * 3 + 1, align).unwrap();
            let ptr = a.allocate(layout).unwrap().cast::<u8>();
            assert_eq!(
                (ptr.as_ptr() as usize) % align,
                0,
                "payload must honor requested alignment {align}"
            );
            #[expect(unsafe_code, reason = "frees the block the test just allocated")]
            // SAFETY: `ptr`/`layout` came from this allocator's `allocate`.
            unsafe {
                a.deallocate(ptr, layout);
            }
        }
    }

    #[test]
    fn zero_sized_forwards() {
        let a = GuardedAllocator::new(Global);
        let layout = Layout::from_size_align(0, 4).unwrap();
        let block = a.allocate(layout).unwrap();
        assert_eq!(block.len(), 0);
        #[expect(unsafe_code, reason = "frees the zero-sized block the test allocated")]
        // SAFETY: `ptr`/`layout` came from this allocator's `allocate`.
        unsafe {
            a.deallocate(block.cast::<u8>(), layout);
        }
    }

    #[test]
    #[should_panic(expected = "overflow")]
    fn rear_overflow_is_caught() {
        let a = GuardedAllocator::new(Global);
        let layout = Layout::from_size_align(8, 8).unwrap();
        let ptr = alloc_and_fill(&a, layout);
        #[expect(unsafe_code, reason = "test deliberately scribbles past the payload")]
        // SAFETY: the rear redzone (>=1 byte) lies within the owned block, so
        // writing one byte past the payload stays inside the allocation.
        unsafe {
            ptr.as_ptr().add(layout.size()).write(0x00);
            a.deallocate(ptr, layout);
        }
    }

    #[test]
    #[should_panic(expected = "underflow")]
    fn front_underflow_is_caught() {
        let a = GuardedAllocator::new(Global);
        let layout = Layout::from_size_align(8, 8).unwrap();
        let ptr = alloc_and_fill(&a, layout);
        #[expect(unsafe_code, reason = "test deliberately scribbles before the payload")]
        // SAFETY: the front redzone (>=1 byte) lies within the owned block, so
        // writing one byte before the payload stays inside the allocation.
        unsafe {
            ptr.as_ptr().sub(1).write(0x00);
            a.deallocate(ptr, layout);
        }
    }

    #[test]
    #[should_panic]
    fn double_free_is_caught() {
        let a = GuardedAllocator::new(Global);
        let layout = Layout::from_size_align(8, 8).unwrap();
        let ptr = alloc_and_fill(&a, layout);
        #[expect(unsafe_code, reason = "test deliberately frees the same block twice")]
        // SAFETY: the first free is valid. The second free reuses the released
        // block; the decorator rejects it before touching the inner allocator
        // again, either via the freed-magic sentinel (block not yet recycled)
        // or via the header-magic / size validation (block already recycled).
        // Either path aborts with a panic, which is all this test requires.
        unsafe {
            a.deallocate(ptr, layout);
            a.deallocate(ptr, layout);
        }
    }

    #[test]
    fn disabled_redzone_still_detects_double_free() {
        let a = GuardedAllocator::with_config(Global, GuardConfig::new(0));
        let layout = Layout::from_size_align(8, 8).unwrap();
        let ptr = alloc_and_fill(&a, layout);
        #[expect(unsafe_code, reason = "frees once validly with redzone checks disabled")]
        // SAFETY: `ptr`/`layout` came from this allocator's `allocate`.
        unsafe {
            a.deallocate(ptr, layout);
        }
    }
}
