//! End-to-end streaming-integration parity for the virtual-geometry GPU twins.
//!
//! The upstream lane already proves each GPU twin in isolation
//! (`page_table_parity`, `page_storage_parity`). This test closes the loop: it
//! drives the twins from the *actual* CPU streaming orchestrator
//! [`PageStreamManager`] instead of a hand-built [`prism_render_architecture::paging::PagePool`],
//! proving the residency and slot map produced by [`PageStreamManager::reconcile`]
//! is reproduced bit-exactly on a real device.
//!
//! Two legs are asserted against the same manager state:
//!
//! * **Resolve parity** — the key-sorted [`PageStreamManager::entries`] table is
//!   fed to [`GpuPageTable`], and every query key must resolve to exactly the
//!   slot [`PageStreamManager::slot_of`] reports (misses to [`UNMAPPED_SLOT`]).
//!   The query set mixes resident keys, gap misses and never-seen keys so a
//!   degenerate all-hit or all-miss table cannot pass vacuously.
//! * **Storage parity** — each resident page's deterministic payload is uploaded
//!   to its manager-assigned slot through [`GpuPageStorage`], then the whole
//!   pool is fetched slot-major and compared against an independently computed
//!   golden buffer, proving the storage twin places page data at exactly the
//!   slots the orchestrator chose.
//!
//! The manager also exercises **displacement eviction** across two frames so the
//! parity legs run over a residency set that has actually churned (a page
//! streamed in, a lower-priority page evicted), not just an append-only fill.
//!
//! Both legs are pure integer lookups/copies, so parity is asserted bit-exact
//! with no floating-point tolerance. The test skips (with a printed notice) when
//! the host has no `wgpu` adapter, keeping the suite green everywhere while still
//! running the full dispatch-and-readback on any real device.
//!
//! Provenance: standard residency streaming and sorted-table resolve; no Unreal
//! Engine source or derived code.

use prism_render_architecture::paging::{
    PageSource, PageStreamManager, RequestBatch, UNMAPPED_SLOT,
};
use prism_render_architecture::virtual_geometry::GeometryPageKey;
use prism_virtual_geometry_gpu::{GpuContext, GpuPageStorage, GpuPageTable};

const PAGE_WORDS: u32 = 4;
const CAPACITY: u32 = 8;
const BUDGET: usize = 6;

fn key(asset: u32, page: u32) -> GeometryPageKey {
    GeometryPageKey::new(asset, page)
}

/// Deterministic page-payload source. Each `(key, word)` maps to a distinct
/// non-zero value so the parity checks cannot pass on all-zero degenerate data.
struct KeyedSource;

fn encode_word(k: GeometryPageKey, word: u32) -> u32 {
    // Mix asset/page/word into a non-zero, per-(key, word) distinct value.
    ((k.asset.wrapping_add(1)) << 20)
        ^ ((k.page.wrapping_add(1)) << 8)
        ^ word.wrapping_add(1)
}

impl PageSource<GeometryPageKey> for KeyedSource {
    fn load(&mut self, key: GeometryPageKey) -> Vec<u32> {
        (0..PAGE_WORDS).map(|w| encode_word(key, w)).collect()
    }
}

/// Asserts the GPU page-table twin resolves every query to exactly the slot the
/// manager assigned, with both hits and misses genuinely present.
fn assert_resolve_parity(
    ctx: &GpuContext,
    mgr: &PageStreamManager<GeometryPageKey>,
    queries: &[GeometryPageKey],
) {
    let entries = mgr.entries();
    let resolver = GpuPageTable::new(ctx);
    let gpu = resolver
        .resolve(ctx, &entries, queries)
        .expect("manager entries are key-sorted");
    assert_eq!(gpu.len(), queries.len(), "one slot per query");

    let mut hits = 0usize;
    let mut misses = 0usize;
    for (i, q) in queries.iter().enumerate() {
        let expected = mgr.slot_of(*q).unwrap_or(UNMAPPED_SLOT);
        assert_eq!(gpu[i], expected, "slot mismatch for query {q:?}");
        if expected == UNMAPPED_SLOT {
            misses += 1;
        } else {
            hits += 1;
        }
    }
    assert!(hits > 0, "queries must include resident hits");
    assert!(misses > 0, "queries must include misses");
}

/// Asserts the GPU physical-page storage twin places each resident page's
/// payload at exactly the manager-assigned slot and reads it back bit-exactly.
fn assert_storage_parity(ctx: &GpuContext, mgr: &PageStreamManager<GeometryPageKey>) {
    let entries = mgr.entries();
    let mut src = KeyedSource;

    // Golden slot-major buffer: zero-initialised pool with each resident page's
    // deterministic payload copied into its assigned slot span.
    let mut expected = vec![0u32; (CAPACITY * PAGE_WORDS) as usize];
    let payloads: Vec<(u32, Vec<u32>)> =
        entries.iter().map(|(k, slot)| (*slot, src.load(*k))).collect();
    for (slot, words) in &payloads {
        let base = (*slot * PAGE_WORDS) as usize;
        expected[base..base + PAGE_WORDS as usize].copy_from_slice(words);
    }

    let uploads: Vec<(u32, &[u32])> =
        payloads.iter().map(|(s, w)| (*s, w.as_slice())).collect();
    let mut fetches = Vec::new();
    for slot in 0..CAPACITY {
        for word in 0..PAGE_WORDS {
            fetches.push((slot, word));
        }
    }

    let storage = GpuPageStorage::new(ctx);
    let got = storage.round_trip(ctx, PAGE_WORDS, CAPACITY, &uploads, &fetches);
    assert_eq!(got, expected, "GPU storage round-trip must match golden buffer");
    assert!(
        got.iter().any(|&w| w != 0),
        "resident payloads must be non-zero (guards against degenerate data)"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn streamed_residency_resolves_and_stores_bit_exact() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping stream-integration parity: no wgpu adapter on this host");
        return;
    };

    let mut mgr = PageStreamManager::new(CAPACITY, PAGE_WORDS, BUDGET);
    let mut src = KeyedSource;

    let mut batch = RequestBatch::new();
    batch.record(key(0, 0), 5.0);
    batch.record(key(0, 4), 4.0);
    batch.record(key(1, 2), 9.0);
    batch.record(key(2, 0), 1.0);
    batch.record(key(3, 3), 7.0);

    let report = mgr.reconcile(&batch, 0, &mut src);
    assert_eq!(report.streamed_in.len(), 5, "all five pages fit the budget");
    assert!(report.evicted.is_empty(), "nothing to evict on a fresh fill");
    assert!(report.deferred.is_empty(), "budget covers the whole request");
    assert_eq!(mgr.resident_count(), 5);

    let queries = vec![
        key(0, 0), // resident
        key(0, 4), // resident
        key(1, 2), // resident
        key(2, 0), // resident
        key(3, 3), // resident
        key(0, 1), // gap between resident pages -> miss
        key(2, 5), // asset present, page absent -> miss
        key(9, 9), // never seen -> miss
    ];
    assert_resolve_parity(&ctx, &mgr, &queries);
    assert_storage_parity(&ctx, &mgr);
}

#[test]
fn displacement_eviction_stays_bit_exact() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };

    let mut mgr = PageStreamManager::new(CAPACITY, PAGE_WORDS, BUDGET);
    let mut src = KeyedSource;

    // Frame 0: fill the resident budget with six distinct-priority pages.
    let mut b0 = RequestBatch::new();
    b0.record(key(0, 0), 1.0);
    b0.record(key(0, 1), 2.0);
    b0.record(key(0, 2), 3.0);
    b0.record(key(0, 3), 4.0);
    b0.record(key(0, 4), 5.0);
    b0.record(key(0, 5), 6.0);
    let r0 = mgr.reconcile(&b0, 0, &mut src);
    assert_eq!(r0.streamed_in.len(), 6, "budget filled");
    assert_eq!(mgr.resident_count(), 6);

    // Frame 1: request only one new high-priority page. The budget is full and
    // none of the frame-0 pages are used this frame, so the lowest-priority
    // resident page (key(0, 0) at 1.0) is displaced.
    let mut b1 = RequestBatch::new();
    b1.record(key(7, 7), 100.0);
    let r1 = mgr.reconcile(&b1, 1, &mut src);
    assert_eq!(r1.streamed_in, vec![key(7, 7)], "the new page streams in");
    assert_eq!(r1.evicted, vec![key(0, 0)], "lowest-priority page is displaced");
    assert_eq!(mgr.resident_count(), 6, "residency stays at budget");
    assert!(mgr.slot_of(key(7, 7)).is_some(), "new page is resident");
    assert!(mgr.slot_of(key(0, 0)).is_none(), "displaced page is gone");

    let queries = vec![
        key(7, 7), // newly resident (reused the freed slot)
        key(0, 1), // still resident
        key(0, 5), // still resident
        key(0, 0), // displaced -> miss
        key(4, 4), // never seen -> miss
    ];
    assert_resolve_parity(&ctx, &mgr, &queries);
    assert_storage_parity(&ctx, &mgr);
}
