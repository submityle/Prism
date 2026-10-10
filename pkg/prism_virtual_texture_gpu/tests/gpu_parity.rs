//! Real-device parity tests for the virtual-texture page-table lookup twin.
//!
//! Each test acquires a best-effort headless `GPU` via
//! [`GpuContext::try_headless`]. On a machine with a usable adapter (e.g. Apple
//! `M`-series Metal) the lookup kernel runs for real and is compared against the
//! device-free CPU golden
//! [`GpuPageTable::lookup`](prism_render_architecture::texture_streaming::GpuPageTable::lookup);
//! where no adapter exists the test prints a skip note and returns, so CI
//! without a `GPU` stays green.
//!
//! The lookup is integer-only, so the device result must equal the golden
//! result *exactly* — these tests assert bit-for-bit equality, including the
//! `MISS` sentinel for page coordinates with no resident entry.

use prism_render_architecture::texture_streaming::{
    GpuPageTable, PhysicalPagePool, TexturePageKey,
};
use prism_virtual_texture_gpu::{GpuContext, GpuPageLookup, MISS};

/// Acquires the device or prints a skip note and returns `None`.
#[expect(
    clippy::print_stderr,
    reason = "test diagnostics: explain a GPU-less skip so CI logs are legible"
)]
fn with_gpu() -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!(
                "[prism_virtual_texture_gpu] no headless GPU adapter available; \
                 skipping device parity test"
            );
            None
        }
    }
}

fn key(texture: u32, mip: u8, layer: u16, x: u16, y: u16) -> TexturePageKey {
    TexturePageKey {
        texture,
        mip,
        layer,
        x,
        y,
    }
}

/// Builds a resident pool with a deterministic spread of keys across textures,
/// mips, layers, and page grid coordinates, plus the matching page table.
fn resident_table() -> (GpuPageTable, Vec<TexturePageKey>) {
    let mut pool = PhysicalPagePool::new(4096);
    let mut resident = Vec::new();
    // A deterministic lattice: several textures, a mip pyramid, two layers, a
    // modest page grid. Order of admission is irrelevant — `GpuPageTable` is
    // built from the pool's key-ordered iterator.
    for texture in 0..6u32 {
        for mip in 0..4u8 {
            let extent = 8u16 >> mip; // 8,4,2,1 pages across at each mip
            for layer in 0..2u16 {
                for x in 0..extent {
                    for y in 0..extent {
                        let k = key(texture, mip, layer, x, y);
                        if pool.admit(k).is_some() {
                            resident.push(k);
                        }
                    }
                }
            }
        }
    }
    (GpuPageTable::from_pool(&pool), resident)
}

#[test]
fn lookup_resident_keys_match_golden_slots() {
    let Some(ctx) = with_gpu() else { return };
    let (table, resident) = resident_table();
    let gpu = GpuPageLookup::new(&ctx);

    let device = gpu.lookup(&ctx, &table, &resident);
    assert_eq!(device.len(), resident.len());
    for (i, &k) in resident.iter().enumerate() {
        let golden = table.lookup(k).expect("resident key must resolve");
        assert_eq!(
            device[i], golden,
            "device slot for resident key {k:?} diverged from golden"
        );
        assert_ne!(device[i], MISS, "resident key {k:?} must not report MISS");
    }
}

#[test]
fn lookup_mixed_hits_and_misses_match_golden() {
    let Some(ctx) = with_gpu() else { return };
    let (table, resident) = resident_table();
    let gpu = GpuPageLookup::new(&ctx);

    // Interleave resident keys with coordinates guaranteed absent (texture 99,
    // out-of-range page coords, unused mips) so both hit and miss paths run.
    let mut queries = Vec::new();
    for (i, &k) in resident.iter().enumerate() {
        queries.push(k);
        if i % 3 == 0 {
            queries.push(key(99, 0, 0, (i % 64) as u16, (i % 64) as u16));
            queries.push(key(k.texture, 7, k.layer, k.x, k.y));
            queries.push(key(k.texture, k.mip, k.layer, k.x + 1000, k.y + 1000));
        }
    }

    let device = gpu.lookup(&ctx, &table, &queries);
    assert_eq!(device.len(), queries.len());
    for (i, &k) in queries.iter().enumerate() {
        let golden = table.lookup(k).unwrap_or(MISS);
        assert_eq!(
            device[i], golden,
            "device slot for query {k:?} diverged from golden"
        );
    }
}

#[test]
fn lookup_against_empty_table_is_all_miss() {
    let Some(ctx) = with_gpu() else { return };
    let table = GpuPageTable::new();
    let gpu = GpuPageLookup::new(&ctx);
    let queries = vec![key(0, 0, 0, 0, 0), key(1, 2, 0, 3, 4), key(5, 1, 1, 6, 7)];

    let device = gpu.lookup(&ctx, &table, &queries);
    assert_eq!(device.len(), queries.len());
    for (i, &k) in queries.iter().enumerate() {
        assert_eq!(table.lookup(k), None, "golden must miss on empty table");
        assert_eq!(device[i], MISS, "device must report MISS for {k:?}");
    }
}
