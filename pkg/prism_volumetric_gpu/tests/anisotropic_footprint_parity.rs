//! Real-device parity for the anisotropic-footprint twin:
//! [`GpuAnisotropicFootprint`](prism_volumetric_gpu::anisotropic_footprint::GpuAnisotropicFootprint)
//! must reproduce the `CPU` golden
//! ([`anisotropic_footprint`](prism_render_architecture::particle::anisotropic_footprint))
//! across an isotropic footprint (unit anisotropy), a stretched footprint
//! (anisotropy above one), an axis-aligned footprint whose minor axis sets a
//! sharp anisotropic `LOD`, a strongly elongated footprint clamped by a
//! non-integer `max_anisotropy`, and a randomized batch compared
//! element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds,
//! divides, one `sqrt` and integer bit work, so `CPU` and `GPU` evaluate the
//! same closed form in the same order. They are not bit-exact on the `f32`
//! fields: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The comparison therefore allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on
//! every continuous field (`major_len`, `minor_len`, `anisotropy`,
//! `lod_trilinear`, `lod_anisotropic`), while the integer `sample_count` is
//! compared for exact equality.
//!
//! # Conditioning
//!
//! Every random fixture is kept well away from the kernel's discrete cracks:
//! rejection sampling requires a long axis comfortably above the degeneracy
//! epsilon, a short axis well above the collapsed-minor branch, a discriminant
//! far from zero (so `major_len != minor_len`, avoiding a repeated eigenvalue),
//! an anisotropy strictly below the `max_anisotropy` clamp, and a scaled
//! anisotropy whose fractional part sits mid-step so the `floor` of the sample
//! count never lands on an integer tie. The deterministic fixtures use exact
//! integer geometry, so their `floor` boundaries are reached exactly on both
//! devices.
//!
//! Provenance: 孪生自本仓
//! `prism_render_architecture::particle::anisotropic_footprint`；no third-party
//! engine source or derived code.

use prism_volumetric_gpu::anisotropic_footprint::{
    golden, FootprintQuery, FootprintResult, GpuAnisotropicFootprint,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in
/// the last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// A pseudo-random value in `[-span, span)` drawn from `state`.
fn signed(state: &mut u64, span: f32) -> f32 {
    (lcg(state) * 2.0 - 1.0) * span
}

/// Builds a clearly-conditioned footprint query by rejection sampling: the two
/// gradients are drawn from well-spread components and accepted only when the
/// resulting ellipse stays far from every discrete crack — a non-degenerate
/// long axis, a short axis above the collapse branch, a well-separated axis
/// pair, an unclamped anisotropy and a mid-step sample fraction.
fn rand_query(state: &mut u64) -> FootprintQuery {
    loop {
        let ddx = [signed(state, 3.0), signed(state, 3.0)];
        let ddy = [signed(state, 3.0), signed(state, 3.0)];
        let max_anisotropy = 16.0;
        let query = FootprintQuery::new(ddx, ddy, max_anisotropy);
        let f = golden(&query);
        // Long axis clearly non-degenerate.
        if f.major_len < 0.5 {
            continue;
        }
        // Short axis well above the collapsed-minor branch.
        if f.minor_len < 0.25 {
            continue;
        }
        // Axes well separated (discriminant far from zero, no repeated root).
        if f.major_len - f.minor_len < 0.2 {
            continue;
        }
        // Anisotropy strictly below the clamp so neither device pins to the max.
        if f.anisotropy > max_anisotropy - 1.0 {
            continue;
        }
        // Scaled anisotropy fraction kept mid-step, away from a floor tie.
        let scaled = f.anisotropy * 256.0;
        let frac = scaled - scaled.floor();
        if !(0.2..=0.8).contains(&frac) {
            continue;
        }
        return query;
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: every continuous
/// field must agree within bound and the integer sample count exactly.
fn pin(idx: usize, query: &FootprintQuery, got: &FootprintResult) {
    let want = golden(query);
    assert!(
        close(got.major_len, want.major_len),
        "query {idx} major_len: gpu {} vs cpu {}",
        got.major_len,
        want.major_len
    );
    assert!(
        close(got.minor_len, want.minor_len),
        "query {idx} minor_len: gpu {} vs cpu {}",
        got.minor_len,
        want.minor_len
    );
    assert!(
        close(got.anisotropy, want.anisotropy),
        "query {idx} anisotropy: gpu {} vs cpu {}",
        got.anisotropy,
        want.anisotropy
    );
    assert!(
        close(got.lod_trilinear, want.lod_trilinear),
        "query {idx} lod_trilinear: gpu {} vs cpu {}",
        got.lod_trilinear,
        want.lod_trilinear
    );
    assert!(
        close(got.lod_anisotropic, want.lod_anisotropic),
        "query {idx} lod_anisotropic: gpu {} vs cpu {}",
        got.lod_anisotropic,
        want.lod_anisotropic
    );
    assert_eq!(
        got.sample_count, want.sample_count,
        "query {idx} sample_count: gpu {} vs cpu {}",
        got.sample_count, want.sample_count
    );
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuAnisotropicFootprint, queries: &[FootprintQuery]) {
    let got = gpu.eval(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, query, result);
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnisotropicFootprint::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn isotropic_footprint_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnisotropicFootprint::new(&ctx);
    // Orthonormal gradients: a round footprint with unit anisotropy, both axes
    // length one, both LOD levels log2(1) == 0 and a single sample. Integer
    // geometry is exact on both devices.
    let query = FootprintQuery::new([1.0, 0.0], [0.0, 1.0], 16.0);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn stretched_footprint_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnisotropicFootprint::new(&ctx);
    // A 4:1 axis-aligned stretch: major 4, minor 1, anisotropy 4, four samples,
    // trilinear LOD log2(4) == 2 and anisotropic LOD log2(1) == 0.
    let query = FootprintQuery::new([4.0, 0.0], [0.0, 1.0], 16.0);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn sharp_minor_axis_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnisotropicFootprint::new(&ctx);
    // An 8:1 stretch: the minor axis of length one sharpens the anisotropic LOD
    // to zero while the trilinear LOD follows the longest gradient, log2(8) == 3.
    let query = FootprintQuery::new([8.0, 0.0], [0.0, 1.0], 16.0);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn clamped_anisotropy_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnisotropicFootprint::new(&ctx);
    // A 16:1 stretch clamped by a non-integer max_anisotropy of 7.5, so the
    // anisotropy pins to 7.5 and the sample count is ceil(7.5) == 8. The
    // non-integer clamp keeps the scaled sample floor well away from a tie.
    let query = FootprintQuery::new([16.0, 0.0], [0.0, 1.0], 7.5);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn skew_footprint_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnisotropicFootprint::new(&ctx);
    // A non-axis-aligned gradient pair so the eigenvalue solve exercises a
    // non-zero off-diagonal metric term B.
    let query = FootprintQuery::new([3.0, 4.0], [-2.0, 1.0], 16.0);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnisotropicFootprint::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing deterministic fixtures with many random queries,
    // dispatched together so the per-thread indexing and the contiguous storage
    // layout are both exercised, then pinned element-for-element.
    let mut queries = vec![
        FootprintQuery::new([1.0, 0.0], [0.0, 1.0], 16.0),
        FootprintQuery::new([4.0, 0.0], [0.0, 1.0], 16.0),
        FootprintQuery::new([16.0, 0.0], [0.0, 1.0], 7.5),
    ];
    for _ in 0..48 {
        queries.push(rand_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_queries_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnisotropicFootprint::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) pins every field across many
    // random footprint geometries.
    let queries: Vec<FootprintQuery> = (0..200).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
