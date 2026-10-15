//! Immutable, shareable, content-addressed byte blobs (design §16.4).
//!
//! Large, read-only data blocks — collision meshes, sampled curves, baked
//! animation clips — are frequently referenced by *many* entities at once.
//! Storing a private copy per entity would waste memory and defeat cache
//! locality, so this module provides a [`BlobStore`]: a reference-counted,
//! content-addressed arena of immutable byte blobs.
//!
//! # Model
//!
//! * **Content-addressed / deduplicated.** Inserting the same bytes twice
//!   returns the *same* [`BlobHandle`] and stores the payload only once. Dedup
//!   is driven by a deterministic 64-bit FNV-1a content hash; hash collisions
//!   are resolved by comparing the actual bytes, so distinct payloads never
//!   alias even if their hashes match.
//! * **Cheap handles.** A [`BlobHandle`] is a small `Copy` id (a slot index
//!   plus a generation guard) that entities store as component data.
//! * **Reference counted.** [`BlobStore::insert_bytes`] bumps an existing
//!   blob's refcount or creates a fresh one at refcount `1`;
//!   [`BlobStore::retain`] bumps it explicitly, and [`BlobStore::release`]
//!   decrements it, freeing the backing storage (and invalidating the handle)
//!   when the count reaches zero.
//!
//! # Typed convenience
//!
//! [`BlobStore::insert_pod`] / [`BlobStore::get_pod`] round-trip a plain
//! `Copy` value through its raw bytes without any external crate. See their
//! `# Safety` notes for the (documented) contract on padding and alignment.
//!
//! This module is `no_std + alloc`: it uses only `core::`/`alloc::` and the
//! crate-internal [`HashMap`](crate::collections::HashMap).

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::mem::{align_of, size_of};

use crate::collections::HashMap;

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Deterministic 64-bit FNV-1a content hash over `bytes`.
///
/// This is intentionally a tiny, fixed, dependency-free hash: it is used only
/// for content-addressed *bucketing*, and every hash match is confirmed by a
/// full byte comparison, so the hash never needs to be collision-free.
fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// A cheap, `Copy` reference to an immutable blob stored in a [`BlobStore`].
///
/// Handles are generational: a slot freed by [`BlobStore::release`] bumps its
/// generation, so a stale handle to a since-reused slot is detected and
/// rejected by [`BlobStore::get`] rather than silently aliasing new data.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BlobHandle {
    /// Dense index of the backing slot within the store.
    index: u32,
    /// Generation guard; must match the slot's current generation to be valid.
    generation: u32,
}

/// A stable content identity for a blob payload.
///
/// The hash is only a lookup key: callers that need collision-proof identity
/// must still use [`BlobStore::contains`] or [`BlobStore::find`], which compare
/// the full bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BlobId {
    hash: u64,
    len: usize,
}

impl BlobId {
    /// The deterministic content hash used for indexing.
    #[inline]
    pub const fn hash(self) -> u64 {
        self.hash
    }

    /// The payload length included in this identity.
    #[inline]
    pub const fn len(self) -> usize {
        self.len
    }

    /// Whether this identity describes an empty payload.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }
}

/// Errors returned by fallible blob operations.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlobError {
    /// The handle does not refer to a live blob in this store.
    InvalidHandle(BlobHandle),
    /// A reference count cannot be incremented without wrapping.
    RefcountOverflow(BlobHandle),
    /// A new slot cannot be represented by the compact handle index.
    SlotIndexOverflow,
    /// The generation guard cannot be incremented without reusing a stale
    /// handle value, so the last reference is retained instead of reclaiming.
    GenerationExhausted(BlobHandle),
}

impl fmt::Display for BlobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHandle(handle) => write!(f, "invalid blob handle {handle:?}"),
            Self::RefcountOverflow(handle) => {
                write!(f, "blob reference count overflow for {handle:?}")
            }
            Self::SlotIndexOverflow => f.write_str("blob slot index exceeds u32::MAX"),
            Self::GenerationExhausted(handle) => {
                write!(f, "blob generation exhausted for {handle:?}")
            }
        }
    }
}

impl BlobHandle {
    /// The raw dense slot index backing this handle.
    #[inline]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// The generation guard carried by this handle.
    #[inline]
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

/// One stored blob: its 8-byte-aligned backing buffer, logical byte length,
/// content hash, and live reference count.
struct BlobEntry {
    /// Backing storage, over-allocated as `u64` words to guarantee 8-byte
    /// alignment for [`BlobStore::get_pod`]. Only the first `len` bytes are
    /// logically part of the blob.
    data: Box<[u64]>,
    /// Logical length of the blob in bytes (`<= data.len() * 8`).
    len: usize,
    /// Cached FNV-1a content hash, used to drop the index out of the hash
    /// bucket on free without rehashing.
    hash: u64,
    /// Number of outstanding references; the entry is freed when this hits `0`.
    refcount: u32,
}

impl BlobEntry {
    /// View the logical blob payload as a byte slice.
    #[inline]
    fn bytes(&self) -> &[u8] {
        // SAFETY: `data` is a `Box<[u64]>` of `data.len()` initialized words,
        // so its first `data.len() * 8 >= len` bytes are valid and initialized
        // and live as long as `&self`. Reading them as `u8` is always sound
        // (every `u8` bit pattern is valid, and `u64` has stricter alignment).
        unsafe { core::slice::from_raw_parts(self.data.as_ptr().cast::<u8>(), self.len) }
    }
}

/// A backing slot: an optional live [`BlobEntry`] plus a generation counter
/// that is bumped each time the slot is freed.
struct Slot {
    /// The live entry, or `None` if this slot is free.
    entry: Option<BlobEntry>,
    /// Current generation; handles must match this to resolve.
    generation: u32,
}

/// Allocate an 8-byte-aligned `Box<[u64]>` holding a copy of `bytes`.
///
/// Trailing padding in the final word (when `bytes.len()` is not a multiple of
/// 8) is zero-initialized, which keeps the content hash and byte comparison
/// well defined and avoids ever reading uninitialized memory.
fn alloc_aligned(bytes: &[u8]) -> Box<[u64]> {
    let words = bytes.len().div_ceil(size_of::<u64>());
    let mut buf = vec![0u64; words];
    if !bytes.is_empty() {
        // SAFETY: `buf` holds `words * 8 >= bytes.len()` contiguous, writable,
        // initialized bytes starting at `buf.as_mut_ptr()`; `bytes` is a valid
        // readable region of `bytes.len()` bytes; the two allocations are
        // distinct so the ranges cannot overlap.
        unsafe {
            core::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buf.as_mut_ptr().cast::<u8>(),
                bytes.len(),
            );
        }
    }
    buf.into_boxed_slice()
}

/// A reference-counted, content-addressed arena of immutable byte blobs.
///
/// See the [module docs](self) for the full model. Construct one with
/// [`BlobStore::new`] (or [`Default`]), then [`insert_bytes`](Self::insert_bytes)
/// payloads and store the returned [`BlobHandle`]s on entities.
pub struct BlobStore {
    /// Dense slot table; freed slots are recycled via `free_list`.
    slots: Vec<Slot>,
    /// Indices of currently-free slots available for reuse.
    free_list: Vec<u32>,
    /// Content hash -> candidate slot indices sharing that hash. A `Vec` per
    /// bucket resolves collisions; membership is confirmed by byte comparison.
    by_hash: HashMap<u64, Vec<u32>>,
    /// Count of currently-live blobs (slots with a `Some` entry).
    live: usize,
}

impl Default for BlobStore {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl BlobStore {
    /// Create an empty [`BlobStore`].
    #[inline]
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_list: Vec::new(),
            by_hash: HashMap::new(),
            live: 0,
        }
    }

    /// Number of currently-live (allocated, non-freed) blobs.
    #[inline]
    pub fn blob_count(&self) -> usize {
        self.live
    }

    /// Whether the store currently holds no live blobs.
    #[inline]
    pub fn is_store_empty(&self) -> bool {
        self.live == 0
    }

    /// Return the collision-safe content identity for `bytes`.
    #[inline]
    pub fn content_id(bytes: &[u8]) -> BlobId {
        BlobId {
            hash: fnv1a_64(bytes),
            len: bytes.len(),
        }
    }

    /// Return the content identity of a live handle, or `None` when stale.
    #[inline]
    pub fn id(&self, handle: BlobHandle) -> Option<BlobId> {
        let slot = self.slots.get(handle.index as usize)?;
        (slot.generation == handle.generation)
            .then(|| {
                slot.entry.as_ref().map(|entry| BlobId {
                    hash: entry.hash,
                    len: entry.len,
                })
            })
            .flatten()
    }

    /// Find an existing payload without creating or retaining a reference.
    pub fn find(&self, bytes: &[u8]) -> Option<BlobHandle> {
        let id = Self::content_id(bytes);
        let candidates = self.by_hash.get(&id.hash)?;
        candidates.iter().copied().find_map(|index| {
            let slot = self.slots.get(index as usize)?;
            let entry = slot.entry.as_ref()?;
            (entry.len == id.len && entry.bytes() == bytes).then_some(BlobHandle {
                index,
                generation: slot.generation,
            })
        })
    }

    /// Whether `bytes` already has a live, collision-safe entry.
    #[inline]
    pub fn contains(&self, bytes: &[u8]) -> bool {
        self.find(bytes).is_some()
    }

    /// Insert `bytes`, deduplicating by content.
    ///
    /// If an identical payload is already stored, its refcount is incremented
    /// and the existing [`BlobHandle`] is returned (the bytes are *not* stored
    /// again). Otherwise a fresh blob is created at refcount `1`.
    pub fn try_insert_bytes(&mut self, bytes: &[u8]) -> Result<BlobHandle, BlobError> {
        let hash = fnv1a_64(bytes);

        if let Some(candidates) = self.by_hash.get(&hash) {
            for &index in candidates {
                let slot = &mut self.slots[index as usize];
                if let Some(entry) = slot.entry.as_mut()
                    && entry.bytes() == bytes
                {
                    let handle = BlobHandle {
                        index,
                        generation: slot.generation,
                    };
                    entry.refcount = entry
                        .refcount
                        .checked_add(1)
                        .ok_or(BlobError::RefcountOverflow(handle))?;
                    return Ok(handle);
                }
            }
        }

        let entry = BlobEntry {
            data: alloc_aligned(bytes),
            len: bytes.len(),
            hash,
            refcount: 1,
        };

        let index = if let Some(index) = self.free_list.pop() {
            let slot = &mut self.slots[index as usize];
            slot.entry = Some(entry);
            index
        } else {
            let index =
                u32::try_from(self.slots.len()).map_err(|_| BlobError::SlotIndexOverflow)?;
            self.slots.push(Slot {
                entry: Some(entry),
                generation: 0,
            });
            index
        };

        self.by_hash.entry(hash).or_default().push(index);
        self.live += 1;
        Ok(BlobHandle {
            index,
            generation: self.slots[index as usize].generation,
        })
    }

    /// Insert `bytes`, panicking only for allocation failure or an impossible
    /// compact-index exhaustion. Use [`try_insert_bytes`](Self::try_insert_bytes)
    /// when the caller needs explicit error handling.
    pub fn insert_bytes(&mut self, bytes: &[u8]) -> BlobHandle {
        self.try_insert_bytes(bytes)
            .unwrap_or_else(|error| panic!("blob insertion failed: {error}"))
    }

    /// Insert the raw bytes of a plain `Copy` value `T`, deduplicated by
    /// content, returning a handle that can later be read back with
    /// [`get_pod`](Self::get_pod).
    ///
    /// # Safety
    ///
    /// The caller must ensure that copying `size_of::<T>()` bytes out of
    /// `*value` is appropriate for the type. For the result to
    /// be meaningful and the read to be well defined, `T` must have **no
    /// uninitialized padding bytes** (e.g. `u32`, `[u8; N]`, or
    /// `#[repr(C)]`/`#[repr(packed)]` structs whose fields tile the type with
    /// no gaps). Passing a `T` with padding is not memory-unsafe here — the
    /// bytes are merely copied — but the padding contents are unspecified and
    /// will participate in dedup.
    pub unsafe fn insert_pod<T: Copy + 'static>(&mut self, value: &T) -> BlobHandle {
        // SAFETY: the caller guarantees that copying the complete object
        // representation, including any padding, is intended for this type.
        // `T: Copy` is `Sized`, and `value` is a valid reference to a
        // fully initialized `T`, so `size_of::<T>()` bytes starting at its
        // address are readable for the duration of this call. Reinterpreting
        // them as `u8` is sound (any byte pattern is a valid `u8`). The slice
        // does not outlive `value`.
        let bytes = unsafe {
            core::slice::from_raw_parts(core::ptr::from_ref(value).cast::<u8>(), size_of::<T>())
        };
        self.insert_bytes(bytes)
    }

    /// Explicitly add a reference to an existing blob, returning `true` on
    /// success or `false` if the handle is stale/invalid.
    pub fn retain(&mut self, handle: BlobHandle) -> bool {
        self.try_retain(handle).is_ok()
    }

    /// Explicitly add a reference, preserving the store on every error path.
    pub fn try_retain(&mut self, handle: BlobHandle) -> Result<(), BlobError> {
        let Some(slot) = self.slots.get_mut(handle.index as usize) else {
            return Err(BlobError::InvalidHandle(handle));
        };
        if slot.generation != handle.generation {
            return Err(BlobError::InvalidHandle(handle));
        }
        let Some(entry) = slot.entry.as_mut() else {
            return Err(BlobError::InvalidHandle(handle));
        };
        entry.refcount = entry
            .refcount
            .checked_add(1)
            .ok_or(BlobError::RefcountOverflow(handle))?;
        Ok(())
    }

    /// Alias for [`retain`](Self::retain) that returns a fresh copy of the
    /// handle, mirroring an `Rc::clone`-style call site. Returns `None` if the
    /// handle is stale/invalid.
    pub fn clone_handle(&mut self, handle: BlobHandle) -> Option<BlobHandle> {
        if self.retain(handle) {
            Some(handle)
        } else {
            None
        }
    }

    /// Release one reference, returning a precise error for stale handles.
    pub fn try_release(&mut self, handle: BlobHandle) -> Result<bool, BlobError> {
        let index = handle.index as usize;
        let Some(slot) = self.slots.get_mut(index) else {
            return Err(BlobError::InvalidHandle(handle));
        };
        if slot.generation != handle.generation {
            return Err(BlobError::InvalidHandle(handle));
        }
        let Some(entry) = slot.entry.as_mut() else {
            return Err(BlobError::InvalidHandle(handle));
        };
        if entry.refcount == 0 {
            return Err(BlobError::InvalidHandle(handle));
        }
        if entry.refcount == 1 && slot.generation == u32::MAX {
            return Err(BlobError::GenerationExhausted(handle));
        }
        debug_assert!(
            entry.refcount > 0,
            "live blob must own at least one reference"
        );
        entry.refcount -= 1;
        if entry.refcount != 0 {
            return Ok(false);
        }

        let hash = entry.hash;
        slot.entry = None;
        slot.generation = slot.generation.wrapping_add(1);
        self.live -= 1;
        self.free_list.push(handle.index);

        if let Some(bucket) = self.by_hash.get_mut(&hash) {
            if let Some(pos) = bucket.iter().position(|&i| i == handle.index) {
                bucket.swap_remove(pos);
            }
            if bucket.is_empty() {
                self.by_hash.remove(&hash);
            }
        }
        Ok(true)
    }

    /// Remove one reference to a blob.
    ///
    /// Returns `true` if the handle was valid (whether or not this call freed
    /// the blob), or `false` if it was stale/invalid. When the refcount
    /// reaches zero the backing storage is dropped, the slot is recycled, and
    /// its generation is bumped so the (now dangling) handle no longer
    /// resolves.
    pub fn release(&mut self, handle: BlobHandle) -> bool {
        self.try_release(handle).is_ok()
    }

    /// Resolve a handle to the blob's immutable bytes, or `None` if the handle
    /// is stale/invalid.
    pub fn get(&self, handle: BlobHandle) -> Option<&[u8]> {
        let slot = self.slots.get(handle.index as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        slot.entry.as_ref().map(BlobEntry::bytes)
    }

    /// Read a blob back as a reference to a plain `Copy` value `T`.
    ///
    /// Returns `None` if the handle is stale/invalid, if the blob's length is
    /// not exactly `size_of::<T>()`, or if the backing storage is not
    /// sufficiently aligned for `T` (the arena guarantees 8-byte alignment, so
    /// any `T` with `align_of::<T>() <= 8` always passes the alignment check).
    ///
    /// # Safety
    /// The referenced bytes must be a valid initialized representation of `T`.
    /// This must not be used to reinterpret arbitrary bytes as types with
    /// validity invariants such as `bool`, references, `NonZero*`, or enums.
    pub unsafe fn get_pod<T: Copy + 'static>(&self, handle: BlobHandle) -> Option<&T> {
        let bytes = self.get(handle)?;
        if bytes.len() != size_of::<T>() {
            return None;
        }
        let ptr = bytes.as_ptr();
        if !(ptr as usize).is_multiple_of(align_of::<T>()) {
            return None;
        }
        // SAFETY: we verified `bytes.len() == size_of::<T>()` and that `ptr` is
        // aligned for `T`; the bytes are an initialized copy of a `T`'s bytes
        // (written by `insert_pod`/`insert_bytes`) that lives as long as `&self`.
        // The caller's safety contract establishes that the bytes are a valid
        // initialized representation of `T`; the returned reference borrows
        // `self` so it cannot outlive the storage.
        Some(unsafe { &*ptr.cast::<T>() })
    }

    /// Length in bytes of the referenced blob, or `None` if the handle is
    /// stale/invalid.
    #[inline]
    pub fn len(&self, handle: BlobHandle) -> Option<usize> {
        self.get(handle).map(<[u8]>::len)
    }

    /// Whether the referenced blob is zero-length, or `None` if the handle is
    /// stale/invalid.
    #[inline]
    pub fn is_empty(&self, handle: BlobHandle) -> Option<bool> {
        self.get(handle).map(<[u8]>::is_empty)
    }

    /// Current reference count of the referenced blob, or `None` if the handle
    /// is stale/invalid. Primarily for tests and diagnostics.
    pub fn refcount(&self, handle: BlobHandle) -> Option<u32> {
        let slot = self.slots.get(handle.index as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        slot.entry.as_ref().map(|entry| entry.refcount)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_returns_same_handle_and_bumps_refcount() {
        let mut store = BlobStore::new();
        let a = store.insert_bytes(b"collision-mesh");
        let b = store.insert_bytes(b"collision-mesh");
        assert_eq!(a, b);
        assert_eq!(store.blob_count(), 1);
        assert_eq!(store.refcount(a), Some(2));
    }

    #[test]
    fn distinct_bytes_get_distinct_handles() {
        let mut store = BlobStore::new();
        let a = store.insert_bytes(b"curve-a");
        let b = store.insert_bytes(b"curve-b");
        assert_ne!(a, b);
        assert_eq!(store.blob_count(), 2);
        assert_eq!(store.get(a), Some(&b"curve-a"[..]));
        assert_eq!(store.get(b), Some(&b"curve-b"[..]));
    }

    #[test]
    fn retain_and_release_free_on_zero() {
        let mut store = BlobStore::new();
        let h = store.insert_bytes(b"anim-clip");
        assert_eq!(store.refcount(h), Some(1));
        assert!(store.retain(h));
        assert_eq!(store.refcount(h), Some(2));

        assert!(store.release(h));
        assert_eq!(store.refcount(h), Some(1));
        assert_eq!(store.blob_count(), 1);

        assert!(store.release(h));
        assert_eq!(store.blob_count(), 0);
        assert!(store.is_store_empty());
    }

    #[test]
    fn get_after_full_release_returns_none() {
        let mut store = BlobStore::new();
        let h = store.insert_bytes(b"payload");
        assert!(store.release(h));
        assert_eq!(store.get(h), None);
        assert_eq!(store.refcount(h), None);
        assert_eq!(store.len(h), None);
        assert_eq!(store.is_empty(h), None);
        // Double-release of a now-stale handle is rejected, not a panic.
        assert!(!store.release(h));
        assert!(!store.retain(h));
    }

    #[test]
    fn clone_handle_tracks_refcount_and_rejects_stale() {
        let mut store = BlobStore::new();
        let h = store.insert_bytes(b"shared");
        let h2 = store.clone_handle(h).expect("live handle clones");
        assert_eq!(h, h2);
        assert_eq!(store.refcount(h), Some(2));

        assert!(store.release(h));
        assert!(store.release(h2));
        assert_eq!(store.blob_count(), 0);
        assert_eq!(store.clone_handle(h), None);
    }

    #[test]
    fn freed_slot_is_recycled_with_new_generation() {
        let mut store = BlobStore::new();
        let first = store.insert_bytes(b"first");
        assert!(store.release(first));

        // Reuses the same slot index but with a bumped generation.
        let second = store.insert_bytes(b"second");
        assert_eq!(first.index(), second.index());
        assert_ne!(first.generation(), second.generation());

        // The stale handle must not resolve to the recycled slot's data.
        assert_eq!(store.get(first), None);
        assert_eq!(store.get(second), Some(&b"second"[..]));
    }

    #[test]
    fn empty_blob_round_trips() {
        let mut store = BlobStore::new();
        let h = store.insert_bytes(b"");
        assert_eq!(store.get(h), Some(&b""[..]));
        assert_eq!(store.len(h), Some(0));
        assert_eq!(store.is_empty(h), Some(true));
        // Empty payloads also dedup.
        let h2 = store.insert_bytes(b"");
        assert_eq!(h, h2);
        assert_eq!(store.refcount(h), Some(2));
    }

    #[test]
    fn hash_collision_path_compares_bytes() {
        // Simulate a genuine FNV-1a collision: force a payload with *different*
        // bytes into another payload's hash bucket, then confirm that a
        // re-insert still resolves to the correct slot by comparing bytes
        // rather than trusting the shared hash.
        let mut store = BlobStore::new();
        let a = store.insert_bytes(b"alpha");
        let b = store.insert_bytes(b"beta!");

        // `insert_bytes` buckets by the real content hash of its argument, so
        // the collision we inject must live in *alpha's* bucket for the
        // byte-comparison path to be exercised when re-inserting "alpha".
        let alpha_hash = fnv1a_64(b"alpha");
        store
            .by_hash
            .get_mut(&alpha_hash)
            .expect("alpha has a bucket")
            .push(b.index());
        let slot = store.slots.get_mut(b.index() as usize).expect("b slot");
        slot.entry.as_mut().expect("b entry").hash = alpha_hash;

        // Re-inserting "alpha" iterates the bucket [a, b]; the colliding `b`
        // candidate is skipped because its bytes differ, so dedup lands on `a`.
        let again = store.insert_bytes(b"alpha");
        assert_eq!(again, a);
        assert_eq!(store.refcount(a), Some(2));
        assert_eq!(store.blob_count(), 2);
        assert_eq!(store.get(a), Some(&b"alpha"[..]));
        assert_eq!(store.get(b), Some(&b"beta!"[..]));
    }

    #[test]
    fn insert_pod_get_pod_round_trip() {
        #[derive(Clone, Copy, PartialEq, Debug)]
        #[repr(C)]
        struct Pod {
            a: u32,
            b: u32,
            c: u64,
        }

        let mut store = BlobStore::new();
        let value = Pod {
            a: 7,
            b: 42,
            c: 0x0123_4567_89ab_cdef,
        };
        // SAFETY: `Pod` is repr(C), Copy, and has no padding.
        let h = unsafe { store.insert_pod(&value) };
        // SAFETY: the blob was produced from the same valid `Pod` type.
        assert_eq!(unsafe { store.get_pod::<Pod>(h) }, Some(&value));
        assert_eq!(store.len(h), Some(size_of::<Pod>()));

        // Scalar POD round-trip too.
        // SAFETY: u32 has no invalid bit patterns or padding.
        let hn = unsafe { store.insert_pod(&12345u32) };
        // SAFETY: the blob was produced from u32.
        assert_eq!(unsafe { store.get_pod::<u32>(hn) }, Some(&12345u32));
    }

    #[test]
    fn get_pod_rejects_length_mismatch() {
        let mut store = BlobStore::new();
        // SAFETY: u32 has no invalid bit patterns or padding.
        let h = unsafe { store.insert_pod(&0xAABB_CCDDu32) };
        // Correct type succeeds...
        // SAFETY: the blob was produced from u32.
        assert!(unsafe { store.get_pod::<u32>(h) }.is_some());
        // ...but a wrongly-sized type is rejected rather than mis-read.
        // SAFETY: size validation prevents a read for these types.
        assert_eq!(unsafe { store.get_pod::<u64>(h) }, None);
        // SAFETY: size validation prevents a read for these types.
        assert_eq!(unsafe { store.get_pod::<u16>(h) }, None);
    }

    #[test]
    fn get_pod_alignment_guarantee_holds_for_u64() {
        let mut store = BlobStore::new();
        // 8-byte-aligned backing means an 8-aligned type always resolves.
        // SAFETY: u64 has no invalid bit patterns or padding.
        let h = unsafe { store.insert_pod(&0x1122_3344_5566_7788u64) };
        // SAFETY: the blob was produced from u64.
        assert_eq!(
            // SAFETY: the handle was created from a u64 payload.
            unsafe { store.get_pod::<u64>(h) },
            Some(&0x1122_3344_5566_7788u64)
        );
    }

    #[test]
    fn content_id_find_and_contains_do_not_change_lifetime() {
        let mut store = BlobStore::new();
        let payload = b"shared immutable payload";
        let id = BlobStore::content_id(payload);
        assert_eq!(id.len(), payload.len());
        assert!(store.find(payload).is_none());
        assert!(!store.contains(payload));

        let handle = store.insert_bytes(payload);
        assert_eq!(store.id(handle), Some(id));
        assert_eq!(store.find(payload), Some(handle));
        assert!(store.contains(payload));
        assert_eq!(store.refcount(handle), Some(1));
    }

    #[test]
    fn fallible_lifecycle_rejects_stale_handle_without_mutation() {
        let mut store = BlobStore::new();
        let handle = store.insert_bytes(b"payload");
        assert_eq!(store.try_release(handle), Ok(true));
        assert_eq!(
            store.try_retain(handle),
            Err(BlobError::InvalidHandle(handle))
        );
        assert_eq!(
            store.try_release(handle),
            Err(BlobError::InvalidHandle(handle))
        );
        assert_eq!(store.blob_count(), 0);
    }

    #[test]
    fn release_reports_whether_storage_was_reclaimed() {
        let mut store = BlobStore::new();
        let handle = store.insert_bytes(b"payload");
        store.retain(handle);
        assert_eq!(store.try_release(handle), Ok(false));
        assert_eq!(store.try_release(handle), Ok(true));
        assert_eq!(store.get(handle), None);
    }

    #[test]
    fn generation_exhaustion_keeps_last_reference_live() {
        let mut store = BlobStore::new();
        let mut handle = store.insert_bytes(b"never alias");
        store.slots[handle.index() as usize].generation = u32::MAX;
        handle.generation = u32::MAX;

        assert_eq!(
            store.try_release(handle),
            Err(BlobError::GenerationExhausted(handle))
        );
        assert_eq!(store.refcount(handle), Some(1));
        assert_eq!(store.get(handle), Some(&b"never alias"[..]));
    }

    #[test]
    fn refcount_overflow_is_rejected_without_wrapping() {
        let mut store = BlobStore::new();
        let handle = store.insert_bytes(b"bounded refs");
        store.slots[handle.index() as usize]
            .entry
            .as_mut()
            .expect("live entry")
            .refcount = u32::MAX;

        assert_eq!(
            store.try_retain(handle),
            Err(BlobError::RefcountOverflow(handle))
        );
        assert_eq!(
            store.try_insert_bytes(b"bounded refs"),
            Err(BlobError::RefcountOverflow(handle))
        );
        assert_eq!(store.refcount(handle), Some(u32::MAX));
        assert_eq!(store.get(handle), Some(&b"bounded refs"[..]));
    }
}
