//! Real-device parity for the physical page-data placement twin:
//! [`GpuPageStorage`] must reproduce the CPU golden [`PageStorage`] word-for-word
//! after applying the same upload batch and reading back the same words.
//!
//! The test skips (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! scatter-gather-and-readback on any real device. Both kernels are portable
//! integer WGSL, so like the page-table twin they need no optional feature.
//!
//! # Parity criterion
//!
//! The pool moves raw `u32` words by index alone - there is no floating-point
//! arithmetic - so every gathered word is asserted bit-exact against
//! [`PageStorage::fetch`] with no tolerance. The scene covers multi-slot uploads
//! read back across slots, a never-written slot reading zero (the reference's
//! post-`clear_slot` and freshly-`new` state alike), and an out-of-range target
//! slot that is skipped on both sides.
//!
//! [`PageStorage`]: prism_render_architecture::paging::PageStorage
//! [`PageStorage::fetch`]: prism_render_architecture::paging::PageStorage::fetch
//!
//! Provenance: standard indexed scatter/gather; no Unreal Engine source or
//! derived code.

use prism_render_architecture::paging::PageStorage;
use prism_virtual_geometry_gpu::{GpuContext, GpuPageStorage};

/// Applies the same uploads to a fresh CPU golden pool and returns the words the
/// given fetches read, using `0` for the out-of-range reads the GPU defines to
/// zero so the two vectors line up index-for-index.
fn cpu_round_trip(
    page_words: u32,
    capacity: u32,
    uploads: &[(u32, &[u32])],
    fetches: &[(u32, u32)],
) -> Vec<u32> {
    let mut pool = PageStorage::new(capacity, page_words);
    for (slot, page) in uploads {
        // The reference rejects an out-of-range slot; the GPU skips it. Both
        // leave the pool unchanged for that upload, so ignore the error to keep
        // the two paths aligned.
        let _ = pool.upload(*slot, page);
    }
    fetches
        .iter()
        .map(|(slot, word)| pool.fetch(*slot, *word).unwrap_or(0))
        .collect()
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_page_storage_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping page-storage parity: no wgpu adapter on this host");
        return;
    };
    let storage = GpuPageStorage::new(&ctx);

    let page_words = 4u32;
    let capacity = 5u32;
    let uploads: &[(u32, &[u32])] = &[
        (1, &[10, 11, 12, 13]),
        (3, &[30, 31, 32, 33]),
        (0, &[1, 2, 3, 4]),
    ];
    let fetches: &[(u32, u32)] = &[
        (1, 0), // -> 10
        (1, 3), // -> 13
        (3, 2), // -> 32
        (0, 1), // -> 2
        (2, 0), // never written -> 0
        (4, 3), // never written -> 0
    ];

    let gpu = storage.round_trip(&ctx, page_words, capacity, uploads, fetches);
    let cpu = cpu_round_trip(page_words, capacity, uploads, fetches);

    assert_eq!(gpu.len(), fetches.len(), "one word per fetch");
    assert_eq!(gpu, cpu, "gpu gathered words must match cpu golden bit-exact");

    // The scene must genuinely read both written and unwritten slots so a
    // degenerate all-zero result cannot pass vacuously.
    assert!(gpu.iter().any(|&w| w != 0), "scene must read written data");
    assert!(gpu.contains(&0), "scene must read a zeroed slot");
}

#[test]
fn out_of_range_upload_target_is_skipped() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let storage = GpuPageStorage::new(&ctx);

    let page_words = 3u32;
    let capacity = 2u32;
    // Slot 2 is out of range for a 2-slot pool and must be skipped on both
    // sides, leaving every readable slot zero.
    let uploads: &[(u32, &[u32])] = &[(2, &[7, 8, 9])];
    let fetches: &[(u32, u32)] = &[(0, 0), (1, 2)];

    let gpu = storage.round_trip(&ctx, page_words, capacity, uploads, fetches);
    let cpu = cpu_round_trip(page_words, capacity, uploads, fetches);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, vec![0, 0], "out-of-range upload writes nothing");
}

#[test]
fn empty_fetch_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let storage = GpuPageStorage::new(&ctx);
    let out = storage.round_trip(&ctx, 4, 3, &[(0, &[1, 2, 3, 4])], &[]);
    assert!(out.is_empty(), "no fetches yield no words");
}

#[test]
fn empty_upload_reads_all_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let storage = GpuPageStorage::new(&ctx);
    let fetches: &[(u32, u32)] = &[(0, 0), (1, 1), (2, 3)];
    let gpu = storage.round_trip(&ctx, 4, 3, &[], fetches);
    let cpu = cpu_round_trip(4, 3, &[], fetches);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, vec![0, 0, 0], "an untouched pool reads all zero");
}
