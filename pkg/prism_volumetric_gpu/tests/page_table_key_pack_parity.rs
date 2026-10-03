//! Real-device parity for the texture-streaming page-key pack/unpack twin:
//! [`GpuPageTableKeyPack`](prism_volumetric_gpu::page_table_key_pack::GpuPageTableKeyPack)
//! must reproduce the stateless `u32` maps of the `CPU` golden
//! [`GpuPageTable::compare_words`](prism_render_architecture::texture_streaming::indirection::GpuPageTable::compare_words)
//! and
//! [`GpuPageTable::unpack_key`](prism_render_architecture::texture_streaming::indirection::GpuPageTable::unpack_key)
//! — the `(w0, w1, w2)` compare-word packing and its inverse — across the zero,
//! single-field, all-ones, boundary and round-trip cases plus a randomized
//! sweep compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! Both reference maps are public static methods, so they are called directly as
//! the oracle: a [`TexturePageKey`](prism_render_architecture::texture_streaming::TexturePageKey)
//! is built from each query's packed fields and
//! [`compare_words`](prism_render_architecture::texture_streaming::indirection::GpuPageTable::compare_words)
//! supplies the expected words, while
//! [`unpack_key`](prism_render_architecture::texture_streaming::indirection::GpuPageTable::unpack_key)
//! supplies the expected key from each query's word triple. A passing
//! `GPU == oracle` run is direct evidence the kernel computes the same packing.
//!
//! # Parity criterion
//!
//! Both maps are pure `u32` bit work built from shifts, masks and bitwise or, so
//! `CPU` and `GPU` agree bit-for-bit and every packed word and unpacked field is
//! asserted with an exact `==`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::texture_streaming::indirection::GpuPageTable`；无第三方引擎源码或衍生代码。

use prism_render_architecture::texture_streaming::indirection::GpuPageTable;
use prism_render_architecture::texture_streaming::TexturePageKey;
use prism_volumetric_gpu::page_table_key_pack::{
    GpuPageTableKeyPack, PageTableKeyPackQuery, PageTableKeyPackResult,
};
use prism_volumetric_gpu::GpuContext;

/// Computes the reference pack/unpack for one query by calling the golden
/// directly. The key fields are narrowed to their reference `u8` / `u16` widths
/// before packing, exactly as the host marshals a real page key.
fn oracle(q: &PageTableKeyPackQuery) -> PageTableKeyPackResult {
    let key = TexturePageKey {
        texture: q.texture,
        mip: q.mip as u8,
        layer: q.layer as u16,
        x: q.x as u16,
        y: q.y as u16,
    };
    let [w0, w1, w2] = GpuPageTable::compare_words(key);
    let unpacked = GpuPageTable::unpack_key(q.word0, q.word1, q.word2);
    PageTableKeyPackResult {
        word0: w0,
        word1: w1,
        word2: w2,
        texture: unpacked.texture,
        mip: u32::from(unpacked.mip),
        layer: u32::from(unpacked.layer),
        x: u32::from(unpacked.x),
        y: u32::from(unpacked.y),
    }
}

/// Pins one `GPU` result against the oracle: every word and field exactly.
fn check_result(idx: usize, got: &PageTableKeyPackResult, want: &PageTableKeyPackResult) {
    assert_eq!(
        got.word0, want.word0,
        "query {idx} word0: gpu {} vs cpu {}",
        got.word0, want.word0
    );
    assert_eq!(
        got.word1, want.word1,
        "query {idx} word1: gpu {} vs cpu {}",
        got.word1, want.word1
    );
    assert_eq!(
        got.word2, want.word2,
        "query {idx} word2: gpu {} vs cpu {}",
        got.word2, want.word2
    );
    assert_eq!(
        got.texture, want.texture,
        "query {idx} texture: gpu {} vs cpu {}",
        got.texture, want.texture
    );
    assert_eq!(
        got.mip, want.mip,
        "query {idx} mip: gpu {} vs cpu {}",
        got.mip, want.mip
    );
    assert_eq!(
        got.layer, want.layer,
        "query {idx} layer: gpu {} vs cpu {}",
        got.layer, want.layer
    );
    assert_eq!(
        got.x, want.x,
        "query {idx} x: gpu {} vs cpu {}",
        got.x, want.x
    );
    assert_eq!(
        got.y, want.y,
        "query {idx} y: gpu {} vs cpu {}",
        got.y, want.y
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuPageTableKeyPack, queries: &[PageTableKeyPackQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_result(idx, result, &want);
    }
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping page_table_key_pack parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuPageTableKeyPack::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn zero_key_and_words_pack_and_unpack_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPageTableKeyPack::new(&ctx);
    // The all-zero key packs to three zero words; the all-zero word triple
    // unpacks to the all-zero key.
    let queries = [PageTableKeyPackQuery::new(0, 0, 0, 0, 0, 0, 0, 0)];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got[0].word0, 0, "zero texture packs to zero w0");
    assert_eq!(got[0].word1, 0, "zero mip/layer packs to zero w1");
    assert_eq!(got[0].word2, 0, "zero x/y packs to zero w2");
    assert_eq!(got[0].texture, 0, "zero w0 unpacks to zero texture");
    assert_eq!(got[0].mip, 0, "zero w1 unpacks to zero mip");
    assert_eq!(got[0].y, 0, "zero w2 unpacks to zero y");
    check(&ctx, &gpu, &queries);
}

#[test]
fn single_fields_land_in_their_own_lanes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPageTableKeyPack::new(&ctx);
    // Each key field, set alone, must land in its documented word lane:
    // texture in w0, mip in w1's high byte, layer in w1's middle, x in w2's
    // high half, y in w2's low half.
    let queries = [
        PageTableKeyPackQuery::new(0xDEAD_BEEF, 0, 0, 0, 0, 0, 0, 0),
        PageTableKeyPackQuery::new(0, 0x7F, 0, 0, 0, 0, 0, 0),
        PageTableKeyPackQuery::new(0, 0, 0xABCD, 0, 0, 0, 0, 0),
        PageTableKeyPackQuery::new(0, 0, 0, 0x1234, 0, 0, 0, 0),
        PageTableKeyPackQuery::new(0, 0, 0, 0, 0x5678, 0, 0, 0),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got[0].word0, 0xDEAD_BEEF, "texture occupies all of w0");
    assert_eq!(got[1].word1, 0x7F << 24, "mip occupies w1's high byte");
    assert_eq!(got[2].word1, 0xABCD << 8, "layer occupies w1's middle");
    assert_eq!(got[3].word2, 0x1234 << 16, "x occupies w2's high half");
    assert_eq!(got[4].word2, 0x5678, "y occupies w2's low half");
    check(&ctx, &gpu, &queries);
}

#[test]
fn all_ones_fields_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPageTableKeyPack::new(&ctx);
    // Maximal field values: full u32 texture, u8 mip and u16 layer/x/y, plus an
    // all-ones word triple to unpack. The reserved low eight bits of w1 are
    // ignored on unpack, so an all-ones triple still yields in-range fields.
    let queries = [PageTableKeyPackQuery::new(
        0xFFFF_FFFF,
        0xFF,
        0xFFFF,
        0xFFFF,
        0xFFFF,
        0xFFFF_FFFF,
        0xFFFF_FFFF,
        0xFFFF_FFFF,
    )];
    check(&ctx, &gpu, &queries);
}

#[test]
fn pack_then_unpack_round_trips_a_representative_key() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPageTableKeyPack::new(&ctx);
    // Pack a representative key, then feed those same words back as the unpack
    // input to confirm the round trip reconstructs every field.
    let key = TexturePageKey {
        texture: 0x1234_5678,
        mip: 7,
        layer: 0xABCD,
        x: 0x4321,
        y: 0xFEDC,
    };
    let [w0, w1, w2] = GpuPageTable::compare_words(key);
    let queries = [PageTableKeyPackQuery::new(
        key.texture,
        u32::from(key.mip),
        u32::from(key.layer),
        u32::from(key.x),
        u32::from(key.y),
        w0,
        w1,
        w2,
    )];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!([got[0].word0, got[0].word1, got[0].word2], [w0, w1, w2]);
    assert_eq!(got[0].texture, key.texture, "round-trip texture");
    assert_eq!(got[0].mip, u32::from(key.mip), "round-trip mip");
    assert_eq!(got[0].layer, u32::from(key.layer), "round-trip layer");
    assert_eq!(got[0].x, u32::from(key.x), "round-trip x");
    assert_eq!(got[0].y, u32::from(key.y), "round-trip y");
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPageTableKeyPack::new(&ctx);
    let mut state = 0x5fb1_9c4e_a7d2_6038_u64;
    let mut queries = Vec::new();
    // Several workgroups' worth of queries. Key fields are masked to their
    // reference u8 / u16 widths (so host narrowing and GPU shifting agree),
    // while the unpack word triple spans the full u32 range since unpack is a
    // total map over any words.
    while queries.len() < 300 {
        let texture = lcg(&mut state);
        let mip = lcg(&mut state) & 0xff;
        let layer = lcg(&mut state) & 0xffff;
        let x = lcg(&mut state) & 0xffff;
        let y = lcg(&mut state) & 0xffff;
        let word0 = lcg(&mut state);
        let word1 = lcg(&mut state);
        let word2 = lcg(&mut state);
        queries.push(PageTableKeyPackQuery::new(
            texture, mip, layer, x, y, word0, word1, word2,
        ));
    }
    check(&ctx, &gpu, &queries);
}
