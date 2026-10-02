//! Real-device parity for the snapshot byte-layout twin:
//! [`GpuSerialization`](prism_volumetric_gpu::serialization::GpuSerialization)
//! must reproduce the `CPU` golden
//! [`particle::serialization`](prism_render_architecture::particle::serialization)
//! across its three per-query answers — the per-channel size
//! [`attribute_channel_bytes`](prism_render_architecture::particle::serialization::attribute_channel_bytes),
//! the whole-snapshot size
//! [`snapshot_total_bytes`](prism_render_architecture::particle::serialization::snapshot_total_bytes)
//! and the version decision
//! [`guard`](prism_render_architecture::particle::serialization::guard) — over an
//! exhaustive cross-product of every attribute, a spread of capacities
//! (including the empty pool), a mix of attribute masks and every guard class,
//! plus a large random batch spanning many workgroups.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every value is a `u32` byte size, a classification code or a migration-range
//! word with no rounding anywhere on the path, so the outputs are bit-identical
//! and asserted with exact `==` and no tolerance. Fixtures keep `header +
//! sum(stride * capacity)` well below `2^31`, so the device `u32` arithmetic
//! never wraps where the golden `saturating_add` / `saturating_mul` would
//! otherwise clamp. Several scenarios additionally assert a non-trivial mix of
//! guard classes and a spread of byte sizes, so a degenerate constant kernel
//! could not pass.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::serialization`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::serialization::{
    attribute_channel_bytes, guard, snapshot_total_bytes, AttributeMask, SnapshotAttribute,
    SnapshotHeader, VersionGuard, SNAPSHOT_MAGIC, SNAPSHOT_VERSION,
};
use prism_volumetric_gpu::serialization::{
    GpuSerialization, GpuSerializationQuery, GpuSerializationResult, CODE_COMPATIBLE,
    CODE_INCOMPATIBLE, CODE_NEEDS_MIGRATION,
};
use prism_volumetric_gpu::GpuContext;

/// A magic word deliberately far from [`SNAPSHOT_MAGIC`], so a wrong-magic
/// query is unambiguously [`VersionGuard::Incompatible`].
const WRONG_MAGIC: u32 = 0xDEAD_BEEF;

/// Maps a golden [`VersionGuard`] to the twin's `(code, from, to)` triple.
fn guard_fields(g: VersionGuard) -> (u32, u32, u32) {
    match g {
        VersionGuard::Compatible => (CODE_COMPATIBLE, 0, 0),
        VersionGuard::NeedsMigration { from, to } => (CODE_NEEDS_MIGRATION, from, to),
        VersionGuard::Incompatible => (CODE_INCOMPATIBLE, 0, 0),
    }
}

/// Computes the golden answer for one query by calling the `CPU` reference
/// directly, matching exactly what the twin must reproduce.
fn golden_result(q: GpuSerializationQuery) -> GpuSerializationResult {
    let attr = SnapshotAttribute::ALL[q.attr_code as usize];
    let channel = attribute_channel_bytes(attr, q.capacity as usize) as u32;
    let header = SnapshotHeader::new(
        q.capacity,
        0,
        AttributeMask::from_bits(q.attribute_mask),
        0,
        0,
        0,
    );
    let total = snapshot_total_bytes(&header) as u32;
    let (code, from, to) = guard_fields(guard(q.header_magic, q.header_version));
    GpuSerializationResult {
        channel_bytes: channel,
        total_bytes: total,
        guard_code: code,
        guard_from: from,
        guard_to: to,
    }
}

/// Runs the twin over `queries` and asserts per-element parity against the
/// golden, returning the device results for extra assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuSerialization,
    queries: &[GpuSerializationQuery],
) -> Vec<GpuSerializationResult> {
    let results = gpu.evaluate(ctx, queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (&q, &r) in queries.iter().zip(results.iter()) {
        assert_eq!(r, golden_result(q), "mismatch for query {q:?}");
    }
    results
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns a raw `u64` state word.
fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// Draws a random in-contract query. The capacity stays below `2^20`, so the
/// full-mask total (`32 + 72 * capacity`) is well under `2^31` and neither the
/// device arithmetic nor the golden `saturating_*` clamps. Magic covers both
/// the real and a wrong word; version covers older / current / future, so all
/// three guard classes appear.
fn draw_query(state: &mut u64) -> GpuSerializationQuery {
    let attr_code = (lcg(state) >> 40) as u32 % SnapshotAttribute::ALL.len() as u32;
    // Capacity in [0, 2^20): with the full mask stride sum of 72 the total is
    // below 2^27.
    let capacity = ((lcg(state) >> 32) as u32) & ((1u32 << 20) - 1);
    // Low 8 bits select the known attribute channels; occasionally stray high
    // bits are set to confirm both sides ignore unknown bits.
    let mut attribute_mask = ((lcg(state) >> 24) as u32) & 0xFF;
    if (lcg(state) >> 63) != 0 {
        attribute_mask |= 1u32 << 20;
    }
    let header_magic = if (lcg(state) >> 63) != 0 {
        SNAPSHOT_MAGIC
    } else {
        WRONG_MAGIC
    };
    // Version in {0, 1, 2, 3} to span older / current / future.
    let header_version = ((lcg(state) >> 48) as u32) & 0x3;
    GpuSerializationQuery {
        attr_code,
        capacity,
        attribute_mask,
        header_magic,
        header_version,
    }
}

/// Builds the exhaustive cross-product of every attribute, a spread of
/// capacities (including the empty pool), a mix of attribute masks and every
/// guard class.
fn structured_queries() -> Vec<GpuSerializationQuery> {
    let capacities = [0u32, 1, 2, 7, 10, 64, 255, 256, 1024, 4096, 65_536];
    let masks = [
        AttributeMask::EMPTY.bits(),
        SnapshotAttribute::Position.bit(),
        SnapshotAttribute::Size.bit(),
        SnapshotAttribute::Position.bit() | SnapshotAttribute::Age.bit(),
        AttributeMask::all().bits(),
        // Full mask plus a stray high bit both sides must ignore.
        AttributeMask::all().bits() | (1u32 << 20),
    ];
    // (magic, version) pairs covering every guard class.
    let guards = [
        (SNAPSHOT_MAGIC, SNAPSHOT_VERSION),     // Compatible
        (SNAPSHOT_MAGIC, 0),                    // NeedsMigration
        (SNAPSHOT_MAGIC, SNAPSHOT_VERSION + 1), // Incompatible (future)
        (WRONG_MAGIC, SNAPSHOT_VERSION),        // Incompatible (wrong magic)
    ];
    let mut queries = Vec::new();
    for attr_code in 0u32..SnapshotAttribute::ALL.len() as u32 {
        for &capacity in &capacities {
            for &attribute_mask in &masks {
                for &(header_magic, header_version) in &guards {
                    queries.push(GpuSerializationQuery {
                        attr_code,
                        capacity,
                        attribute_mask,
                        header_magic,
                        header_version,
                    });
                }
            }
        }
    }
    queries
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_structured_queries() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping serialization parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSerialization::new(&ctx);
    let queries = structured_queries();
    let results = check(&ctx, &gpu, &queries);

    // Non-trivial: all three guard classes appear and the byte sizes span a
    // range, so a degenerate constant kernel could not pass.
    let compatible = results
        .iter()
        .filter(|r| r.guard_code == CODE_COMPATIBLE)
        .count();
    let migrating = results
        .iter()
        .filter(|r| r.guard_code == CODE_NEEDS_MIGRATION)
        .count();
    let incompatible = results
        .iter()
        .filter(|r| r.guard_code == CODE_INCOMPATIBLE)
        .count();
    assert!(compatible > 0, "fixture must include compatible snapshots");
    assert!(migrating > 0, "fixture must include migrating snapshots");
    assert!(
        incompatible > 0,
        "fixture must include incompatible snapshots"
    );
    let min_total = results.iter().map(|r| r.total_bytes).min().unwrap_or(0);
    let max_total = results.iter().map(|r| r.total_bytes).max().unwrap_or(0);
    assert!(
        min_total < max_total,
        "fixture should span a range of total sizes, got [{min_total}, {max_total}]"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_empty_pool_clamp() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping serialization parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSerialization::new(&ctx);

    // A zero-capacity pool still reserves one element per channel, so each
    // channel size equals its attribute stride and the total is the header plus
    // one element of every set channel.
    let queries: Vec<GpuSerializationQuery> = (0u32..SnapshotAttribute::ALL.len() as u32)
        .map(|attr_code| GpuSerializationQuery {
            attr_code,
            capacity: 0,
            attribute_mask: AttributeMask::all().bits(),
            header_magic: SNAPSHOT_MAGIC,
            header_version: SNAPSHOT_VERSION,
        })
        .collect();
    let results = check(&ctx, &gpu, &queries);
    for (q, r) in queries.iter().zip(results.iter()) {
        let stride = SnapshotAttribute::ALL[q.attr_code as usize].stride() as u32;
        assert_eq!(
            r.channel_bytes, stride,
            "empty pool reserves exactly one element"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_large_random_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping serialization parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSerialization::new(&ctx);

    // Several thousand in-contract queries spanning many workgroups.
    let mut state = 0x0BAD_F00D_1234_5678u64;
    let count = 8192usize;
    let mut queries = Vec::with_capacity(count);
    for _ in 0..count {
        queries.push(draw_query(&mut state));
    }
    let results = check(&ctx, &gpu, &queries);

    // The batch mixes all three guard classes, so a degenerate single-class
    // kernel could not pass.
    let any_compatible = results.iter().any(|r| r.guard_code == CODE_COMPATIBLE);
    let any_migrating = results.iter().any(|r| r.guard_code == CODE_NEEDS_MIGRATION);
    let any_incompatible = results.iter().any(|r| r.guard_code == CODE_INCOMPATIBLE);
    assert!(
        any_compatible,
        "random batch should include compatible snapshots"
    );
    assert!(
        any_migrating,
        "random batch should include migrating snapshots"
    );
    assert!(
        any_incompatible,
        "random batch should include incompatible snapshots"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_yields_empty_output() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping serialization parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSerialization::new(&ctx);
    let results = gpu.evaluate(&ctx, &[]);
    assert!(results.is_empty(), "empty input must yield empty output");
}
