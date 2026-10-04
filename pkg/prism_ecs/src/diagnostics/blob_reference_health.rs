//! Blob reference-count and orphan census over a working set of handles
//! (design §16.4 / §16.6).
//!
//! The blob store (design §16.4) is a reference-counted, content-addressed
//! arena of immutable byte blobs ([`BlobStore`]). Inserting identical bytes
//! twice returns the *same* [`BlobHandle`] and bumps a refcount rather than
//! storing a second copy, and a handle is only dropped from storage when its
//! refcount falls to zero. Entities hold handles as plain component data, so
//! the store's internal refcounts are the single source of truth for whether a
//! blob is still shared, uniquely held, or already freed — but a stale handle
//! left in a component after [`release`](BlobStore::release) resolves to
//! nothing, and a blob whose last owning handle was dropped from the working
//! set but never released stays resident forever as a silent leak.
//!
//! The store exposes no iterator over its live blobs, so a full per-blob
//! census is not externally reachable. What *is* reachable is the caller's
//! working set of handles (for example, the blob handles gathered from a query
//! column) together with the store itself. This report resolves that working
//! set against the store once and makes the reference economy legible,
//! read-only, without inserting, retaining, or releasing anything:
//!
//! * **handle resolution** — per distinct handle: whether it still resolves
//!   (live) or dangles after a [`release`](BlobStore::release), the blob's
//!   current store-wide refcount, its logical byte length, and how many times
//!   the handle appears across the audited working set (its *reference
//!   sites*);
//! * **dedup visibility** — because the store is content-addressed, repeated
//!   handles in the working set are literally equal, so the gap between the
//!   raw handle count and the distinct-blob count is the amount of sharing the
//!   working set already enjoys;
//! * **orphan accounting** — comparing the number of distinct *live* blobs the
//!   working set reaches against the store's total live-blob count
//!   ([`blob_count`](BlobStore::blob_count)) surfaces blobs still resident in
//!   the store that nothing in the working set references — the leak signal a
//!   refcounted arena cannot otherwise raise.
//!
//! The census is `O(h)` over the `h` audited handles (each store lookup is
//! `O(1)`); distinct entries are reported in a deterministic order — ascending
//! handle slot index, then generation (design §14) — independent of the input
//! order or the store's internal hashing.

use alloc::vec::Vec;

use crate::blob::{BlobHandle, BlobStore};
use crate::collections::HashMap;

/// Integer permille (`parts per thousand`) of `num / den`, returning `0` when
/// `den` is zero.
#[inline]
fn permille(num: u64, den: u64) -> u64 {
    (num * 1000).checked_div(den).unwrap_or(0)
}

/// One distinct blob handle resolved against the audited [`BlobStore`]: its
/// liveness, store-wide refcount, byte length, and how many times it appears
/// across the working set (design §16.4).
#[derive(Clone, Copy, Debug)]
pub struct BlobReferenceEntry {
    /// The distinct handle this entry describes.
    pub handle: BlobHandle,
    /// The blob's current store-wide reference count, or `0` if the handle no
    /// longer resolves. This counts *all* outstanding references to the blob,
    /// not just those in the audited working set.
    pub ref_count: u32,
    /// Logical length of the blob in bytes, or `0` if the handle no longer
    /// resolves.
    pub byte_len: usize,
    /// How many times this handle appears across the audited working set. A
    /// content-addressed store returns equal handles for equal bytes, so a
    /// value above `1` is deduplicated sharing the working set already has.
    pub reference_sites: u32,
    /// Whether the handle still resolves to a live blob in the store. A `false`
    /// value is a dangling handle left behind by [`release`](BlobStore::release).
    pub is_live: bool,
}

/// Read-only reference-count and orphan census of a working set of
/// [`BlobHandle`]s resolved against a [`BlobStore`]: per-handle resolution,
/// deduplication visibility, and store-wide orphan accounting (design §16.4 /
/// §16.6).
#[derive(Clone, Debug)]
pub struct BlobReferenceHealth {
    referenced_handle_count: usize,
    live_count: usize,
    total_bytes: usize,
    store_blob_count: usize,
    max_ref_count: u32,
    max_reference_sites: u32,
    entries: Vec<BlobReferenceEntry>,
}

impl BlobReferenceHealth {
    /// Censuses a working set of blob `handles` against `store`.
    ///
    /// Handles are deduplicated (the store returns equal handles for equal
    /// content, so repeats are literally equal), each distinct handle is
    /// resolved to its liveness / refcount / byte length, and the number of
    /// distinct *live* blobs reached is compared against
    /// [`store.blob_count()`](BlobStore::blob_count) to surface orphaned
    /// (resident but unreferenced) blobs. Nothing is inserted, retained, or
    /// released. Distinct entries are returned sorted by ascending slot index
    /// then generation (design §14).
    pub fn from_handles(store: &BlobStore, handles: &[BlobHandle]) -> Self {
        // Count working-set occurrences per distinct handle. The store is
        // content-addressed, so equal bytes map to equal handles; distinct
        // handles therefore correspond to distinct blobs.
        let mut sites: HashMap<BlobHandle, u32> = HashMap::default();
        for &handle in handles {
            *sites.entry(handle).or_insert(0) += 1;
        }

        let mut entries = Vec::with_capacity(sites.len());
        let mut live_count = 0usize;
        let mut total_bytes = 0usize;
        let mut max_ref_count = 0u32;
        let mut max_reference_sites = 0u32;

        for (handle, reference_sites) in sites {
            let is_live = store.get(handle).is_some();
            let ref_count = store.refcount(handle).unwrap_or(0);
            let byte_len = store.len(handle).unwrap_or(0);
            if is_live {
                live_count += 1;
                total_bytes += byte_len;
            }
            if ref_count > max_ref_count {
                max_ref_count = ref_count;
            }
            if reference_sites > max_reference_sites {
                max_reference_sites = reference_sites;
            }
            entries.push(BlobReferenceEntry {
                handle,
                ref_count,
                byte_len,
                reference_sites,
                is_live,
            });
        }

        entries.sort_unstable_by_key(|entry| (entry.handle.index(), entry.handle.generation()));

        Self {
            referenced_handle_count: handles.len(),
            live_count,
            total_bytes,
            store_blob_count: store.blob_count(),
            max_ref_count,
            max_reference_sites,
            entries,
        }
    }

    /// Total number of handle references in the audited working set, counting
    /// repeats.
    #[inline]
    pub fn referenced_handle_count(&self) -> usize {
        self.referenced_handle_count
    }

    /// Number of distinct blobs the working set references (live or dangling).
    #[inline]
    pub fn unique_blob_count(&self) -> usize {
        self.entries.len()
    }

    /// Number of distinct handles that still resolve to a live blob.
    #[inline]
    pub fn live_blob_count(&self) -> usize {
        self.live_count
    }

    /// Number of distinct handles that no longer resolve — dangling handles
    /// left behind by [`release`](BlobStore::release).
    #[inline]
    pub fn dangling_handle_count(&self) -> usize {
        self.entries.len() - self.live_count
    }

    /// Number of handle references that were duplicates of an
    /// already-seen handle (`referenced_handle_count - unique_blob_count`): the
    /// amount of content-addressed sharing the working set already has.
    #[inline]
    pub fn duplicate_handle_count(&self) -> usize {
        self.referenced_handle_count - self.entries.len()
    }

    /// Sum of the logical byte lengths of every distinct *live* blob the
    /// working set references (each distinct blob counted once).
    #[inline]
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Total number of live blobs resident in the store, from
    /// [`BlobStore::blob_count`].
    #[inline]
    pub fn store_blob_count(&self) -> usize {
        self.store_blob_count
    }

    /// Number of live blobs resident in the store that the working set does
    /// *not* reference — resident-but-unreferenced orphans, the leak signal.
    #[inline]
    pub fn orphan_blob_count(&self) -> usize {
        self.store_blob_count.saturating_sub(self.live_count)
    }

    /// Highest store-wide refcount among the referenced blobs (`0` when none).
    #[inline]
    pub fn max_ref_count(&self) -> u32 {
        self.max_ref_count
    }

    /// Highest number of working-set reference sites pointing at a single
    /// distinct handle (`0` when none) — the most-shared blob in the working
    /// set.
    #[inline]
    pub fn max_reference_sites(&self) -> u32 {
        self.max_reference_sites
    }

    /// Permille (parts per thousand) of the store's live blobs that the working
    /// set reaches, `0` when the store is empty.
    #[inline]
    pub fn reached_permille(&self) -> u64 {
        permille(self.live_count as u64, self.store_blob_count as u64)
    }

    /// Permille (parts per thousand) of the store's live blobs that are orphans
    /// (resident but unreferenced), `0` when the store is empty.
    #[inline]
    pub fn orphan_permille(&self) -> u64 {
        permille(self.orphan_blob_count() as u64, self.store_blob_count as u64)
    }

    /// Whether the working set is empty *and* the store holds no live blobs —
    /// nothing to report.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.referenced_handle_count == 0 && self.store_blob_count == 0
    }

    /// Whether any referenced handle no longer resolves.
    #[inline]
    pub fn has_dangling(&self) -> bool {
        self.dangling_handle_count() > 0
    }

    /// Whether the store holds any live blob the working set does not
    /// reference.
    #[inline]
    pub fn has_orphans(&self) -> bool {
        self.orphan_blob_count() > 0
    }

    /// The distinct resolved handle entries, sorted by ascending slot index
    /// then generation (design §14).
    #[inline]
    pub fn entries(&self) -> &[BlobReferenceEntry] {
        &self.entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_store_and_no_handles() {
        let store = BlobStore::new();
        let health = BlobReferenceHealth::from_handles(&store, &[]);
        assert!(health.is_empty());
        assert_eq!(health.referenced_handle_count(), 0);
        assert_eq!(health.unique_blob_count(), 0);
        assert_eq!(health.live_blob_count(), 0);
        assert_eq!(health.dangling_handle_count(), 0);
        assert_eq!(health.duplicate_handle_count(), 0);
        assert_eq!(health.orphan_blob_count(), 0);
        assert_eq!(health.store_blob_count(), 0);
        assert_eq!(health.total_bytes(), 0);
        assert_eq!(health.max_ref_count(), 0);
        assert_eq!(health.max_reference_sites(), 0);
        assert_eq!(health.reached_permille(), 0);
        assert_eq!(health.orphan_permille(), 0);
        assert!(!health.has_dangling());
        assert!(!health.has_orphans());
        assert!(health.entries().is_empty());
    }

    #[test]
    fn single_blob_live() {
        let mut store = BlobStore::new();
        let handle = store.insert_bytes(b"collision-mesh");
        let health = BlobReferenceHealth::from_handles(&store, &[handle]);
        assert!(!health.is_empty());
        assert_eq!(health.referenced_handle_count(), 1);
        assert_eq!(health.unique_blob_count(), 1);
        assert_eq!(health.live_blob_count(), 1);
        assert_eq!(health.dangling_handle_count(), 0);
        assert_eq!(health.duplicate_handle_count(), 0);
        assert_eq!(health.orphan_blob_count(), 0);
        assert_eq!(health.store_blob_count(), 1);
        assert_eq!(health.total_bytes(), b"collision-mesh".len());
        assert_eq!(health.max_ref_count(), 1);
        assert_eq!(health.max_reference_sites(), 1);
        assert_eq!(health.reached_permille(), 1000);
        assert_eq!(health.orphan_permille(), 0);
        assert!(!health.has_dangling());
        assert!(!health.has_orphans());

        let entry = health.entries()[0];
        assert_eq!(entry.handle, handle);
        assert!(entry.is_live);
        assert_eq!(entry.ref_count, 1);
        assert_eq!(entry.byte_len, b"collision-mesh".len());
        assert_eq!(entry.reference_sites, 1);
    }

    #[test]
    fn released_handle_dangles() {
        let mut store = BlobStore::new();
        let handle = store.insert_bytes(b"curve");
        assert!(store.release(handle));
        // The store is now empty, but the working set still carries the stale
        // handle.
        let health = BlobReferenceHealth::from_handles(&store, &[handle]);
        assert_eq!(health.referenced_handle_count(), 1);
        assert_eq!(health.unique_blob_count(), 1);
        assert_eq!(health.live_blob_count(), 0);
        assert_eq!(health.dangling_handle_count(), 1);
        assert_eq!(health.store_blob_count(), 0);
        assert_eq!(health.orphan_blob_count(), 0);
        assert_eq!(health.total_bytes(), 0);
        assert!(health.has_dangling());
        assert!(!health.has_orphans());

        let entry = health.entries()[0];
        assert!(!entry.is_live);
        assert_eq!(entry.ref_count, 0);
        assert_eq!(entry.byte_len, 0);
        assert_eq!(entry.reference_sites, 1);
    }

    #[test]
    fn dedup_same_content_shares_handle() {
        let mut store = BlobStore::new();
        // Content-addressed: identical bytes return the same handle and bump
        // the refcount rather than storing a second copy.
        let a = store.insert_bytes(b"animation-clip");
        let b = store.insert_bytes(b"animation-clip");
        assert_eq!(a, b);
        assert_eq!(store.blob_count(), 1);

        let health = BlobReferenceHealth::from_handles(&store, &[a, b]);
        assert_eq!(health.referenced_handle_count(), 2);
        assert_eq!(health.unique_blob_count(), 1);
        assert_eq!(health.live_blob_count(), 1);
        assert_eq!(health.duplicate_handle_count(), 1);
        assert_eq!(health.max_reference_sites(), 2);
        assert_eq!(health.store_blob_count(), 1);
        assert_eq!(health.orphan_blob_count(), 0);

        let entry = health.entries()[0];
        assert_eq!(entry.reference_sites, 2);
        assert_eq!(entry.ref_count, 2);
    }

    #[test]
    fn orphan_blobs_detected() {
        let mut store = BlobStore::new();
        let referenced = store.insert_bytes(b"referenced");
        // A second, distinct blob stays resident but is never referenced by the
        // working set — an orphan / potential leak.
        let _orphan = store.insert_bytes(b"orphan");
        assert_eq!(store.blob_count(), 2);

        let health = BlobReferenceHealth::from_handles(&store, &[referenced]);
        assert_eq!(health.unique_blob_count(), 1);
        assert_eq!(health.live_blob_count(), 1);
        assert_eq!(health.store_blob_count(), 2);
        assert_eq!(health.orphan_blob_count(), 1);
        assert!(health.has_orphans());
        assert_eq!(health.reached_permille(), 500);
        assert_eq!(health.orphan_permille(), 500);
    }

    #[test]
    fn duplicate_handles_counted() {
        let mut store = BlobStore::new();
        let a = store.insert_bytes(b"a");
        let b = store.insert_bytes(b"bb");
        let health = BlobReferenceHealth::from_handles(&store, &[a, a, a, b]);
        assert_eq!(health.referenced_handle_count(), 4);
        assert_eq!(health.unique_blob_count(), 2);
        assert_eq!(health.duplicate_handle_count(), 2);
        assert_eq!(health.max_reference_sites(), 3);
    }

    #[test]
    fn entries_sorted_by_index() {
        let mut store = BlobStore::new();
        let a = store.insert_bytes(b"first");
        let b = store.insert_bytes(b"second");
        let c = store.insert_bytes(b"third");
        // Feed them out of order; entries must come back index-ascending.
        let health = BlobReferenceHealth::from_handles(&store, &[c, a, b]);
        let indices: Vec<u32> = health.entries().iter().map(|e| e.handle.index()).collect();
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        assert_eq!(indices, sorted);
        assert_eq!(health.unique_blob_count(), 3);
    }

    #[test]
    fn mixed_rollup() {
        let mut store = BlobStore::new();
        let shared = store.insert_bytes(b"shared-mesh");
        let _ = store.insert_bytes(b"shared-mesh"); // refcount -> 2, same handle
        let solo = store.insert_bytes(b"solo");
        let stale = store.insert_bytes(b"temp");
        assert!(store.release(stale)); // now dangling
        let _orphan = store.insert_bytes(b"orphan"); // resident, unreferenced

        // Working set: shared twice, solo once, plus the stale handle.
        let health = BlobReferenceHealth::from_handles(&store, &[shared, shared, solo, stale]);
        assert_eq!(health.referenced_handle_count(), 4);
        assert_eq!(health.unique_blob_count(), 3); // shared, solo, stale
        assert_eq!(health.live_blob_count(), 2); // shared, solo
        assert_eq!(health.dangling_handle_count(), 1); // stale
        assert_eq!(health.duplicate_handle_count(), 1); // the second `shared`
        assert_eq!(health.store_blob_count(), 3); // shared, solo, orphan
        assert_eq!(health.orphan_blob_count(), 1); // orphan
        assert_eq!(health.total_bytes(), b"shared-mesh".len() + b"solo".len());
        assert_eq!(health.max_ref_count(), 2); // shared
        assert_eq!(health.max_reference_sites(), 2); // shared
        assert!(health.has_dangling());
        assert!(health.has_orphans());
    }
}
