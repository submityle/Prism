//! Real-device parity for the cloth "blend to skin" twin:
//! [`GpuClothBlendToSkin`](prism_volumetric_gpu::cloth_blend_to_skin::GpuClothBlendToSkin)
//! must reproduce, for one particle per thread, the single closed form of the
//! golden `prism_render_architecture::cloth::painted::blend_to_skin` pass
//! together with the `blend_weight` clamp of
//! `prism_render_architecture::cloth::asset::PaintedConstraint::clamped`.
//!
//! # Independent oracle
//!
//! This suite does not depend on the reference crate. The host [`oracle`] is an
//! independent `f32` reimplementation of the same closed form documented on the
//! twin, evaluated in the same arithmetic order as the kernel: a pinned
//! particle is returned unchanged, otherwise the painted `blend_weight` is
//! clamped to `[0, 1]` with `max(0.0).min(1.0)` (bit-identical to the kernel's
//! `clamp` for every finite input) and each component is mixed as
//! `anchor + (particle - anchor) * weight`. A passing run is therefore evidence
//! that the `WGSL` kernel and an independent `CPU` evaluation of the same map
//! agree, not merely that the shader compiles.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! Each component is a fixed, non-reorderable `anchor + (particle - anchor) *
//! weight` (one subtract and one multiply-add), so the two evaluations compute
//! the same closed form in the same order. They are not bit-exact: a `GPU` may
//! fuse the multiply-add the scalar host leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on the
//! continuous position components and an exact `==` on the discrete `valid`
//! flag.
//!
//! # Conditioning
//!
//! Because every finite `clamp(w, 0, 1)` equals `w.max(0).min(1)` exactly, the
//! continuous position is not sensitive to the clamp knees the way a discrete
//! classifier would be. The named fixtures still pin the clamp edges
//! (`blend = 2.0 -> 1`, `blend = -0.5 -> 0`) exactly, and the random sweep
//! keeps the sampled weight a safe margin clear of `0` and `1` so the mix is a
//! genuine interpolation rather than an endpoint.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::cloth::painted::blend_to_skin` 与 `prism_render_architecture::cloth::asset::PaintedConstraint::clamped`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::cloth_blend_to_skin::{
    ClothBlendToSkinQuery, ClothBlendToSkinResult, GpuClothBlendToSkin,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity floor on every continuous output.
const ABS_TOL: f32 = 1.0e-4;
/// Relative parity slope on every continuous output.
const REL_TOL: f32 = 1.0e-3;
/// Relative-tolerance floor so near-zero magnitudes stay meaningful.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the documented parity bound: an
/// absolute floor or a relative term keeping large-magnitude values meaningful.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    diff <= ABS_TOL || diff <= REL_TOL * a.abs().max(b.abs()).max(REL_FLOOR)
}

/// Independent `f32` reimplementation of the full result for one query, in the
/// same arithmetic order as the kernel: a pinned particle is returned
/// unchanged, otherwise the clamped blend mixes the particle toward the anchor
/// per component.
fn oracle(query: &ClothBlendToSkinQuery) -> ClothBlendToSkinResult {
    if query.pinned != 0 {
        return ClothBlendToSkinResult {
            pos_x: query.particle_x,
            pos_y: query.particle_y,
            pos_z: query.particle_z,
            valid: 1,
        };
    }
    // PaintedConstraint::clamped(): blend_weight.max(0.0).min(1.0).
    let weight = query.blend_weight.max(0.0).min(1.0);
    let mix = |particle: f32, anchor: f32| anchor + (particle - anchor) * weight;
    ClothBlendToSkinResult {
        pos_x: mix(query.particle_x, query.anchor_x),
        pos_y: mix(query.particle_y, query.anchor_y),
        pos_z: mix(query.particle_z, query.anchor_z),
        valid: 1,
    }
}

/// Pins one `GPU` result against the independent host oracle: all three
/// continuous position components within the parity bound and the discrete
/// `valid` flag exactly.
fn pin(idx: usize, query: &ClothBlendToSkinQuery, result: &ClothBlendToSkinResult) {
    let want = oracle(query);
    assert!(
        close(result.pos_x, want.pos_x),
        "query {idx}: pos_x gpu={} oracle={}",
        result.pos_x,
        want.pos_x
    );
    assert!(
        close(result.pos_y, want.pos_y),
        "query {idx}: pos_y gpu={} oracle={}",
        result.pos_y,
        want.pos_y
    );
    assert!(
        close(result.pos_z, want.pos_z),
        "query {idx}: pos_z gpu={} oracle={}",
        result.pos_z,
        want.pos_z
    );
    assert_eq!(
        result.valid, want.valid,
        "query {idx}: valid gpu={} oracle={}",
        result.valid, want.valid
    );
}

/// Evaluates `queries` on-device and pins every result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuClothBlendToSkin, queries: &[ClothBlendToSkinQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, query, result);
    }
}

/// 64-bit linear-congruential step (`Knuth`/`PCG` constants), returning the
/// high word so the stream has good spread without any transcendental math.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 32) as u32
}

/// A deterministic pseudo-random `f32` in `[0, 1]`.
fn unit01(state: &mut u64) -> f32 {
    lcg(state) as f32 / u32::MAX as f32
}

/// A deterministic pseudo-random `f32` in `[-range, range]`.
fn signed(state: &mut u64, range: f32) -> f32 {
    (unit01(state) * 2.0 - 1.0) * range
}

/// A deterministic, well-conditioned random query: finite particle and anchor
/// positions in `[-10, 10]`, a blend weight sampled a safe margin clear of the
/// clamp knees `0` and `1` (so the mix is a genuine interpolation), and a
/// pinned flag toggled from the stream so both branches are exercised.
fn rand_query(state: &mut u64) -> ClothBlendToSkinQuery {
    let particle_x = signed(state, 10.0);
    let particle_y = signed(state, 10.0);
    let particle_z = signed(state, 10.0);
    let anchor_x = signed(state, 10.0);
    let anchor_y = signed(state, 10.0);
    let anchor_z = signed(state, 10.0);
    // Weight in [0.02, 0.98]: in range (so clamp is a no-op) and clear of both
    // knees, so the interpolation is unambiguous on both evaluators.
    let blend_weight = 0.02 + unit01(state) * 0.96;
    // Pin roughly one query in four so the pinned short-circuit is exercised
    // alongside the free mix.
    let pinned = u32::from(lcg(state) & 0x3 == 0);
    ClothBlendToSkinQuery::new(
        particle_x,
        particle_y,
        particle_z,
        anchor_x,
        anchor_y,
        anchor_z,
        blend_weight,
        pinned,
    )
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothBlendToSkin::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn pinned_particle_is_unchanged() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothBlendToSkin::new(&ctx);
    // A pinned particle is never moved, regardless of blend weight or anchor.
    let queries = [
        ClothBlendToSkinQuery::new(2.0, -3.0, 5.0, -1.0, 1.0, 1.0, 0.5, 1),
        ClothBlendToSkinQuery::new(2.0, -3.0, 5.0, -1.0, 1.0, 1.0, 0.0, 7),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len());
    for (idx, (q, r)) in queries.iter().zip(got.iter()).enumerate() {
        assert!(
            close(r.pos_x, 2.0),
            "pinned pos_x unchanged, got {}",
            r.pos_x
        );
        assert!(
            close(r.pos_y, -3.0),
            "pinned pos_y unchanged, got {}",
            r.pos_y
        );
        assert!(
            close(r.pos_z, 5.0),
            "pinned pos_z unchanged, got {}",
            r.pos_z
        );
        pin(idx, q, r);
    }
}

#[test]
fn blend_zero_welds_to_anchor() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothBlendToSkin::new(&ctx);
    // blend_weight 0 (free particle) welds the output onto the anchor.
    let query = ClothBlendToSkinQuery::new(4.0, 4.0, 4.0, -2.0, 6.0, 0.5, 0.0, 0);
    let got = gpu.evaluate(&ctx, &[query]);
    assert_eq!(got.len(), 1, "one result for one query");
    assert!(
        close(got[0].pos_x, -2.0),
        "pos_x -> anchor, got {}",
        got[0].pos_x
    );
    assert!(
        close(got[0].pos_y, 6.0),
        "pos_y -> anchor, got {}",
        got[0].pos_y
    );
    assert!(
        close(got[0].pos_z, 0.5),
        "pos_z -> anchor, got {}",
        got[0].pos_z
    );
    pin(0, &query, &got[0]);
}

#[test]
fn blend_one_keeps_particle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothBlendToSkin::new(&ctx);
    // blend_weight 1 keeps the free simulated particle position.
    let query = ClothBlendToSkinQuery::new(4.0, 4.0, 4.0, -2.0, 6.0, 0.5, 1.0, 0);
    let got = gpu.evaluate(&ctx, &[query]);
    assert_eq!(got.len(), 1, "one result for one query");
    assert!(
        close(got[0].pos_x, 4.0),
        "pos_x -> particle, got {}",
        got[0].pos_x
    );
    assert!(
        close(got[0].pos_y, 4.0),
        "pos_y -> particle, got {}",
        got[0].pos_y
    );
    assert!(
        close(got[0].pos_z, 4.0),
        "pos_z -> particle, got {}",
        got[0].pos_z
    );
    pin(0, &query, &got[0]);
}

#[test]
fn blend_above_one_clamps_to_particle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothBlendToSkin::new(&ctx);
    // blend_weight 2.0 clamps to 1 -> output equals the free particle position.
    let query = ClothBlendToSkinQuery::new(1.0, -1.0, 3.0, 5.0, 5.0, 5.0, 2.0, 0);
    let got = gpu.evaluate(&ctx, &[query]);
    assert_eq!(got.len(), 1, "one result for one query");
    assert!(
        close(got[0].pos_x, 1.0),
        "clamp high -> particle x, got {}",
        got[0].pos_x
    );
    assert!(
        close(got[0].pos_y, -1.0),
        "clamp high -> particle y, got {}",
        got[0].pos_y
    );
    assert!(
        close(got[0].pos_z, 3.0),
        "clamp high -> particle z, got {}",
        got[0].pos_z
    );
    pin(0, &query, &got[0]);
}

#[test]
fn negative_blend_clamps_to_anchor() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothBlendToSkin::new(&ctx);
    // A negative blend_weight clamps to 0 -> output welds onto the anchor.
    let query = ClothBlendToSkinQuery::new(1.0, -1.0, 3.0, 5.0, 5.0, 5.0, -0.5, 0);
    let got = gpu.evaluate(&ctx, &[query]);
    assert_eq!(got.len(), 1, "one result for one query");
    assert!(
        close(got[0].pos_x, 5.0),
        "clamp low -> anchor x, got {}",
        got[0].pos_x
    );
    assert!(
        close(got[0].pos_y, 5.0),
        "clamp low -> anchor y, got {}",
        got[0].pos_y
    );
    assert!(
        close(got[0].pos_z, 5.0),
        "clamp low -> anchor z, got {}",
        got[0].pos_z
    );
    pin(0, &query, &got[0]);
}

#[test]
fn general_oblique_midblend() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothBlendToSkin::new(&ctx);
    // A general oblique mix at blend 0.25: output = anchor + (particle - anchor)
    // * 0.25, distinct per component.
    let query = ClothBlendToSkinQuery::new(8.0, -4.0, 2.0, 0.0, 4.0, -2.0, 0.25, 0);
    let got = gpu.evaluate(&ctx, &[query]);
    assert_eq!(got.len(), 1, "one result for one query");
    // anchor + 0.25 * (particle - anchor):
    // x: 0 + 0.25 * 8 = 2 ; y: 4 + 0.25 * -8 = 2 ; z: -2 + 0.25 * 4 = -1.
    assert!(
        close(got[0].pos_x, 2.0),
        "pos_x should be 2, got {}",
        got[0].pos_x
    );
    assert!(
        close(got[0].pos_y, 2.0),
        "pos_y should be 2, got {}",
        got[0].pos_y
    );
    assert!(
        close(got[0].pos_z, -1.0),
        "pos_z should be -1, got {}",
        got[0].pos_z
    );
    pin(0, &query, &got[0]);
}

#[test]
fn stride_regression_two_element_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothBlendToSkin::new(&ctx);
    // Two deliberately distinct queries in one batch: if the host `std430` query
    // or result stride disagreed with the `WGSL` struct, the second lane would
    // decode from the wrong bytes and the pin would fail. One pinned, one free
    // with distinct positions makes such a mis-stride observable across all four
    // output lanes (three continuous, one discrete), and the two differ in the
    // trailing `pinned` lane so a byte offset error cannot alias them.
    let queries = [
        ClothBlendToSkinQuery::new(1.0, 2.0, 3.0, -4.0, -5.0, -6.0, 0.5, 0),
        ClothBlendToSkinQuery::new(-7.0, 8.0, -9.0, 1.5, -2.5, 3.5, 0.75, 1),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothBlendToSkin::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing the deterministic fixtures with many well-conditioned
    // random queries, dispatched together so the per-thread indexing and the
    // contiguous storage layout are both exercised.
    let mut queries = vec![
        ClothBlendToSkinQuery::new(2.0, -3.0, 5.0, -1.0, 1.0, 1.0, 0.5, 1),
        ClothBlendToSkinQuery::new(4.0, 4.0, 4.0, -2.0, 6.0, 0.5, 0.0, 0),
        ClothBlendToSkinQuery::new(4.0, 4.0, 4.0, -2.0, 6.0, 0.5, 1.0, 0),
        ClothBlendToSkinQuery::new(8.0, -4.0, 2.0, 0.0, 4.0, -2.0, 0.25, 0),
    ];
    for _ in 0..48 {
        queries.push(rand_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothBlendToSkin::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) of well-conditioned random
    // queries pins every output across many invocations.
    let queries: Vec<ClothBlendToSkinQuery> = (0..512).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
