//! Real-device parity for the indirect-draw argument twin:
//! [`GpuIndirectDraw`](prism_volumetric_gpu::indirect_draw::GpuIndirectDraw)
//! must reproduce the `CPU` golden
//! [`particle::indirect_draw`](prism_render_architecture::particle::indirect_draw)
//! across every packed layout — the non-indexed sprite quad, the indexed mesh,
//! the ribbon and beam segment-to-index expansion, the explicit and linear
//! compute dispatches, and the verbatim indexed layout — over a structured
//! spread of boundary cases plus a large random batch spanning many workgroups.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every twinned value is a `u32` indirect-buffer word — a copied field, the
//! constant quad vertex count, a saturating multiply, or a guarded
//! `ceil`-division — with no rounding anywhere on the path, so the outputs are
//! bit-identical and asserted with exact `==` and no tolerance over each
//! layout's meaningful word prefix. Fixtures keep every count well inside `u32`
//! except the deliberate saturation probes (a `u32::MAX` segment count and a
//! clamped launch size), which pin the saturating paths. The flatten tests
//! cross-check that concatenating the twin's per-record words equals the golden
//! `pack_*` buffers.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::indirect_draw`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::indirect_draw::{
    pack_dispatch_indirect, pack_draw_indexed_indirect, pack_draw_indirect, DispatchIndirectArgs,
    DrawIndexedIndirectArgs, DrawIndirectArgs,
};
use prism_volumetric_gpu::indirect_draw::{
    GpuIndirectDraw, GpuIndirectDrawQuery, GpuIndirectDrawResult,
};
use prism_volumetric_gpu::GpuContext;

/// Pads a golden word slice to the fixed five-word result, zeroing the trailing
/// entries, and records the meaningful count.
fn expected(words: &[u32]) -> GpuIndirectDrawResult {
    let mut padded = [0u32; 5];
    padded[..words.len()].copy_from_slice(words);
    GpuIndirectDrawResult {
        words: padded,
        word_count: words.len() as u32,
    }
}

/// Computes the golden answer for one query by calling the `CPU` reference
/// packers directly, matching exactly what the twin must reproduce.
fn golden_result(q: GpuIndirectDrawQuery) -> GpuIndirectDrawResult {
    match q.kind {
        GpuIndirectDrawQuery::KIND_SPRITE => {
            expected(&DrawIndirectArgs::sprite(q.p0, q.p1).as_words())
        }
        GpuIndirectDrawQuery::KIND_MESH => {
            expected(&DrawIndexedIndirectArgs::mesh(q.p0, q.p1, q.p2).as_words())
        }
        GpuIndirectDrawQuery::KIND_RIBBON => {
            expected(&DrawIndexedIndirectArgs::ribbon(q.p0, q.p1, q.p2).as_words())
        }
        GpuIndirectDrawQuery::KIND_BEAM => {
            expected(&DrawIndexedIndirectArgs::beam(q.p0, q.p1, q.p2).as_words())
        }
        GpuIndirectDrawQuery::KIND_DISPATCH => {
            expected(&DispatchIndirectArgs::new(q.p0, q.p1, q.p2).as_words())
        }
        GpuIndirectDrawQuery::KIND_DISPATCH_LINEAR => {
            expected(&DispatchIndirectArgs::linear_1d(q.p0, q.p1, q.p2).as_words())
        }
        GpuIndirectDrawQuery::KIND_INDEXED_RAW => expected(
            &DrawIndexedIndirectArgs {
                index_count: q.p0,
                instance_count: q.p1,
                first_index: q.p2,
                base_vertex: i32::from_ne_bytes(q.p3.to_ne_bytes()),
                first_instance: q.p4,
            }
            .as_words(),
        ),
        other => panic!("unknown kind {other}"),
    }
}

/// Runs the twin over `queries` and asserts per-element parity against the
/// golden, returning the device results for extra assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuIndirectDraw,
    queries: &[GpuIndirectDrawQuery],
) -> Vec<GpuIndirectDrawResult> {
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

/// Draws a random in-contract query across all layouts, with every count well
/// inside `u32` so no saturating path is accidentally exercised: counts stay in
/// `[0, 2^16)`, dispatch workgroup sizes are non-zero, and the linear cap is
/// generous enough never to clamp.
fn random_query(state: &mut u64) -> GpuIndirectDrawQuery {
    let kind = ((lcg(state) >> 40) as u32) % 7;
    let a = ((lcg(state) >> 32) as u32) & 0xffff;
    let b = ((lcg(state) >> 32) as u32) & 0xffff;
    let c = ((lcg(state) >> 32) as u32) & 0xffff;
    match kind {
        GpuIndirectDrawQuery::KIND_SPRITE => GpuIndirectDrawQuery::sprite(a, b),
        GpuIndirectDrawQuery::KIND_MESH => GpuIndirectDrawQuery::mesh(a, b, c),
        GpuIndirectDrawQuery::KIND_RIBBON => GpuIndirectDrawQuery::ribbon(a, b, c),
        GpuIndirectDrawQuery::KIND_BEAM => GpuIndirectDrawQuery::beam(a, b, c),
        GpuIndirectDrawQuery::KIND_DISPATCH => GpuIndirectDrawQuery::dispatch(a, b, c),
        GpuIndirectDrawQuery::KIND_DISPATCH_LINEAR => {
            // Non-zero workgroup size and a cap large enough never to clamp.
            GpuIndirectDrawQuery::dispatch_linear(a, (b % 1024) + 1, 1_000_000)
        }
        _ => GpuIndirectDrawQuery::indexed_raw(a, b, c, lcg(state) as u32, a ^ b),
    }
}

/// Builds a structured spread covering every layout and its boundary cases: the
/// sprite quad, mesh, ribbon/beam segment expansion (including the `u32::MAX`
/// saturation probe), explicit and linear dispatch (exact multiple, one-over,
/// clamped-to-cap, zero workgroup-size guard, zero elements), and the verbatim
/// indexed layout with a negative `base_vertex`.
fn structured_queries() -> Vec<GpuIndirectDrawQuery> {
    vec![
        // Sprite: fixed quad instanced per alive particle.
        GpuIndirectDrawQuery::sprite(1_000, 0),
        GpuIndirectDrawQuery::sprite(0, 7),
        // Mesh: source index count carried through.
        GpuIndirectDrawQuery::mesh(250, 36, 7),
        GpuIndirectDrawQuery::mesh(1, 3, 0),
        // Ribbon / beam: segments * 6 indices.
        GpuIndirectDrawQuery::ribbon(4, 10, 0),
        GpuIndirectDrawQuery::beam(3, 5, 2),
        // Saturation probe: segments * 6 overflows u32 and saturates.
        GpuIndirectDrawQuery::ribbon(1, u32::MAX, 0),
        GpuIndirectDrawQuery::beam(2, u32::MAX, 1),
        // Explicit dispatch.
        GpuIndirectDrawQuery::dispatch(5, 1, 1),
        GpuIndirectDrawQuery::dispatch(7, 3, 2),
        // Linear dispatch boundaries.
        GpuIndirectDrawQuery::dispatch_linear(256, 64, 65_535), // exact multiple -> 4
        GpuIndirectDrawQuery::dispatch_linear(257, 64, 65_535), // one over -> 5
        GpuIndirectDrawQuery::dispatch_linear(255, 64, 65_535), // one under -> 4
        GpuIndirectDrawQuery::dispatch_linear(1_000_000, 64, 128), // clamped to cap 128
        GpuIndirectDrawQuery::dispatch_linear(1_000, 0, 128),   // zero size guard -> 0
        GpuIndirectDrawQuery::dispatch_linear(0, 64, 128),      // zero elements -> 0
        // Verbatim indexed with a negative base_vertex (raw bits = u32::MAX).
        GpuIndirectDrawQuery::indexed_raw(3, 1, 0, u32::MAX, 0),
        GpuIndirectDrawQuery::indexed_raw(36, 250, 12, 0x8000_0000, 4),
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_structured_queries() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping indirect-draw parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuIndirectDraw::new(&ctx);
    let queries = structured_queries();
    let results = check(&ctx, &gpu, &queries);

    // Non-trivial: the word counts span 3, 4 and 5, so a constant-width kernel
    // could not pass.
    let has3 = results.iter().any(|r| r.word_count == 3);
    let has4 = results.iter().any(|r| r.word_count == 4);
    let has5 = results.iter().any(|r| r.word_count == 5);
    assert!(
        has3 && has4 && has5,
        "fixture must span word counts 3, 4 and 5"
    );

    // The saturating ribbon/beam probe must both saturate and not saturate.
    let saturated = results
        .iter()
        .filter(|r| r.word_count == 5 && r.words[0] == u32::MAX)
        .count();
    assert!(
        saturated >= 2,
        "u32::MAX segment probes must saturate index count"
    );
    let small_ribbon = results
        .iter()
        .any(|r| r.word_count == 5 && r.words[0] == 60);
    assert!(
        small_ribbon,
        "a non-saturating ribbon (10 * 6 = 60) must appear"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_dispatch_linear_boundaries() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping indirect-draw parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuIndirectDraw::new(&ctx);

    // Deterministic assertions on both sides of the ceil-division tie and the
    // clamp / guard degenerates.
    let queries = [
        GpuIndirectDrawQuery::dispatch_linear(256, 64, 65_535),
        GpuIndirectDrawQuery::dispatch_linear(257, 64, 65_535),
        GpuIndirectDrawQuery::dispatch_linear(1_000_000, 64, 128),
        GpuIndirectDrawQuery::dispatch_linear(1_000, 0, 128),
    ];
    let results = check(&ctx, &gpu, &queries);
    assert_eq!(
        results[0].words,
        [4, 1, 1, 0, 0],
        "exact multiple divides cleanly"
    );
    assert_eq!(results[1].words, [5, 1, 1, 0, 0], "one over rounds up");
    assert_eq!(results[2].words, [128, 1, 1, 0, 0], "clamped to the cap");
    assert_eq!(
        results[3].words,
        [0, 1, 1, 0, 0],
        "zero workgroup size guards to 0"
    );
    for r in &results {
        assert_eq!(r.word_count, 3, "dispatch layouts pack three words");
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_large_random_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping indirect-draw parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuIndirectDraw::new(&ctx);

    // Several thousand in-contract queries across all layouts, spanning many
    // workgroups.
    let mut state = 0x0BAD_F00D_1234_5678u64;
    let count = 8192usize;
    let mut queries = Vec::with_capacity(count);
    for _ in 0..count {
        queries.push(random_query(&mut state));
    }
    let results = check(&ctx, &gpu, &queries);

    // The batch mixes every layout, so all three word counts appear.
    let has3 = results.iter().any(|r| r.word_count == 3);
    let has4 = results.iter().any(|r| r.word_count == 4);
    let has5 = results.iter().any(|r| r.word_count == 5);
    assert!(has3 && has4 && has5, "random batch must span all layouts");
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_per_record_words_match_golden_pack_buffers() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping indirect-draw parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuIndirectDraw::new(&ctx);

    // Non-indexed draws: concatenating the twin's four-word prefixes must equal
    // the golden pack_draw_indirect buffer.
    let draws = [
        DrawIndirectArgs::sprite(2, 0),
        DrawIndirectArgs::sprite(3, 2),
    ];
    let draw_queries = [
        GpuIndirectDrawQuery::sprite(2, 0),
        GpuIndirectDrawQuery::sprite(3, 2),
    ];
    let draw_results = check(&ctx, &gpu, &draw_queries);
    let flat: Vec<u32> = draw_results
        .iter()
        .flat_map(|r| r.words[..r.word_count as usize].to_vec())
        .collect();
    assert_eq!(
        flat,
        pack_draw_indirect(&draws),
        "sprite flatten matches golden"
    );

    // Indexed draws.
    let indexed = [
        DrawIndexedIndirectArgs::mesh(4, 36, 0),
        DrawIndexedIndirectArgs::ribbon(2, 7, 1),
    ];
    let indexed_queries = [
        GpuIndirectDrawQuery::mesh(4, 36, 0),
        GpuIndirectDrawQuery::ribbon(2, 7, 1),
    ];
    let indexed_results = check(&ctx, &gpu, &indexed_queries);
    let flat_indexed: Vec<u32> = indexed_results
        .iter()
        .flat_map(|r| r.words[..r.word_count as usize].to_vec())
        .collect();
    assert_eq!(
        flat_indexed,
        pack_draw_indexed_indirect(&indexed),
        "indexed flatten matches golden"
    );

    // Dispatches.
    let dispatches = [
        DispatchIndirectArgs::new(5, 1, 1),
        DispatchIndirectArgs::linear_1d(257, 64, 65_535),
    ];
    let dispatch_queries = [
        GpuIndirectDrawQuery::dispatch(5, 1, 1),
        GpuIndirectDrawQuery::dispatch_linear(257, 64, 65_535),
    ];
    let dispatch_results = check(&ctx, &gpu, &dispatch_queries);
    let flat_dispatch: Vec<u32> = dispatch_results
        .iter()
        .flat_map(|r| r.words[..r.word_count as usize].to_vec())
        .collect();
    assert_eq!(
        flat_dispatch,
        pack_dispatch_indirect(&dispatches),
        "dispatch flatten matches golden"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_yields_empty_output() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping indirect-draw parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuIndirectDraw::new(&ctx);
    let results = gpu.evaluate(&ctx, &[]);
    assert!(results.is_empty(), "empty input must yield empty output");
}
