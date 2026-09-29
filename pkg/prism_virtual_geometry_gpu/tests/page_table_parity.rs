//! Real-device parity for the physical page-table resolve twin: [`GpuPageTable`]
//! must reproduce the CPU golden [`PagePool::slot_of`] for every query key,
//! resolving hits to the resident slot and misses to the unmapped sentinel.
//!
//! The test skips (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable integer
//! WGSL, so unlike the payload twin it needs no 64-bit atomic feature.
//!
//! # Parity criterion
//!
//! The resolve is pure integer lookup - lower-bound binary search then exact
//! match over a sorted table - so there is no floating-point tolerance: every
//! resolved slot is asserted bit-exact against the reference, hits and the
//! [`UNMAPPED_SLOT`] sentinel alike. The scene deliberately allocates, frees
//! and reallocates keys so slot reuse and gaps in the asset/page space are
//! covered, and queries mix present keys, evicted keys and never-seen keys.
//!
//! [`PagePool::slot_of`]: prism_render_architecture::paging::PagePool::slot_of
//! [`UNMAPPED_SLOT`]: prism_render_architecture::paging::UNMAPPED_SLOT
//!
//! Provenance: standard sorted-table binary search; no Unreal Engine source or
//! derived code.

use prism_render_architecture::paging::{PagePool, UNMAPPED_SLOT};
use prism_render_architecture::virtual_geometry::GeometryPageKey;
use prism_virtual_geometry_gpu::{GpuContext, GpuPageTable};

fn key(asset: u32, page: u32) -> GeometryPageKey {
    GeometryPageKey::new(asset, page)
}

/// Builds a pool whose slot map exercises dense allocation, a freed-slot reuse
/// and multiple assets, then returns the pool alongside the resident keys.
fn populate_pool() -> PagePool<GeometryPageKey> {
    let mut pool = PagePool::new(16);
    // Allocate across two assets and non-contiguous pages.
    for k in [
        key(0, 0),
        key(0, 1),
        key(0, 5),
        key(1, 0),
        key(1, 3),
        key(2, 9),
    ] {
        pool.allocate(k).expect("pool has room");
    }
    // Free two keys so their slots rejoin the free set, then reallocate new
    // keys that must reclaim the lowest freed slots deterministically.
    pool.free(key(0, 1));
    pool.free(key(1, 0));
    pool.allocate(key(3, 7)).expect("reuse a freed slot");
    pool.allocate(key(3, 8)).expect("reuse a freed slot");
    pool
}

/// Queries covering resident keys, freed (now-absent) keys and never-seen keys.
fn query_set() -> Vec<GeometryPageKey> {
    vec![
        key(0, 0), // resident
        key(0, 1), // freed -> miss
        key(0, 5), // resident
        key(1, 0), // freed -> miss
        key(1, 3), // resident
        key(2, 9), // resident
        key(3, 7), // resident (reused slot)
        key(3, 8), // resident (reused slot)
        key(9, 9), // never seen -> miss
        key(0, 4), // gap between resident pages -> miss
        key(2, 0), // asset present, page absent -> miss
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_page_table_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping page-table parity: no wgpu adapter on this host");
        return;
    };
    let resolver = GpuPageTable::new(&ctx);
    let pool = populate_pool();
    let entries = pool.entries();
    let queries = query_set();

    let gpu = resolver
        .resolve(&ctx, &entries, &queries)
        .expect("sorted entries resolve without error");

    assert_eq!(gpu.len(), queries.len(), "one slot per query");

    let mut hits = 0usize;
    let mut misses = 0usize;
    for (i, q) in queries.iter().enumerate() {
        let expected = pool.slot_of(*q).unwrap_or(UNMAPPED_SLOT);
        assert_eq!(
            gpu[i], expected,
            "slot mismatch for query {q:?}: gpu {}, cpu {expected}",
            gpu[i]
        );
        if expected == UNMAPPED_SLOT {
            misses += 1;
        } else {
            hits += 1;
        }
    }
    // The scene must genuinely cover both outcomes so a degenerate all-miss or
    // all-hit table cannot pass vacuously.
    assert!(hits > 0, "scene must resolve some hits");
    assert!(misses > 0, "scene must resolve some misses");
}

#[test]
fn empty_query_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let resolver = GpuPageTable::new(&ctx);
    let pool = populate_pool();
    let entries = pool.entries();
    let out = resolver
        .resolve(&ctx, &entries, &[])
        .expect("empty query set resolves");
    assert!(out.is_empty());
}

#[test]
fn empty_pool_resolves_all_misses() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let resolver = GpuPageTable::new(&ctx);
    let queries = vec![key(0, 0), key(4, 2)];
    let out = resolver
        .resolve(&ctx, &[], &queries)
        .expect("empty entry table resolves");
    assert_eq!(out, vec![UNMAPPED_SLOT, UNMAPPED_SLOT]);
}

#[test]
fn unsorted_entries_are_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let resolver = GpuPageTable::new(&ctx);
    // Deliberately out of order; the resolver must refuse rather than run an
    // unsound binary search.
    let entries = vec![(key(5, 0), 0u32), (key(1, 0), 1u32)];
    assert!(resolver.resolve(&ctx, &entries, &[key(1, 0)]).is_err());
}
