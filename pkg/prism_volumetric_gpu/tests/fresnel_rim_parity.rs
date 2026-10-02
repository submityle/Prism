//! Real-device parity for the `Fresnel` / edge-light rim twin:
//! [`GpuFresnelRim`](prism_volumetric_gpu::fresnel_rim::GpuFresnelRim) must
//! reproduce the `CPU` golden
//! [`fresnel_rim`](prism_render_architecture::particle::fresnel_rim) across a
//! head-on fixture (normal aligned with the view, so `n_dot_v = 1`, the band is
//! zero and the rim vanishes), a grazing fixture (normal perpendicular to the
//! view, so `n_dot_v = 0` and the rim factor saturates), a back-facing fixture
//! (a negative dot clamps to zero, also grazing), a sweep over several integer
//! `power` exponents at a fixed grazing cosine (exercising the `power_u32`
//! multiply loop at different iteration counts), and a randomized batch of
//! directions whose cosine is rejection-sampled clear of the `smoothstep`
//! transition knots, compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds, divides
//! and one `sqrt`, so `CPU` and `GPU` evaluate the same closed form in the same
//! order. They are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits by a few units
//! in the last place. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` on every continuous `f32` field, while the integer `power`
//! exponent is carried as a raw `u32` and so drives the exact same loop count.
//!
//! # Conditioning
//!
//! Every fixture keeps the derived `n_dot_v` well away from the `smoothstep`
//! band knots at `inner` and `outer`: random directions are accepted only when
//! their cosine lands at least a fixed margin below `inner` (grazing, band `1`)
//! or above `outer` (head-on interior, band `0`), and every direction clears a
//! safe non-degenerate length. This keeps `CPU` and `GPU` on the same side of
//! the `clamp01` knots regardless of a few units in the last place of slack.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::fresnel_rim`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use prism_render_architecture::particle::fresnel_rim::{fresnel_schlick, RimParams};
use prism_volumetric_gpu::fresnel_rim::{FresnelRimQuery, FresnelRimSample, GpuFresnelRim};
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

/// Minimum squared length below which the reference treats a direction as
/// degenerate; the host cross-check mirrors it to compute the expected cosine.
const MIN_LEN_SQ: f32 = 1.0e-12;

/// Inner `smoothstep` edge shared by every fixture.
const INNER: f32 = 0.2;

/// Outer `smoothstep` edge shared by every fixture.
const OUTER: f32 = 0.8;

/// Margin keeping the sampled cosine clear of the `inner` / `outer` knots.
const BAND_MARGIN: f32 = 0.08;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Clamps a scalar into `0..=1`, mirroring the reference `clamp01`.
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Dot product of two hand-rolled 3-component vectors.
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Normalizes a 3-component vector, mirroring the reference `normalize3` (a
/// degenerate input collapses to the zero vector). Uses `sqrt` only, which the
/// reference also uses, so no transcendental appears.
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len_sq = dot3(v, v);
    if len_sq < MIN_LEN_SQ {
        return [0.0, 0.0, 0.0];
    }
    let inv_len = 1.0 / len_sq.sqrt();
    [v[0] * inv_len, v[1] * inv_len, v[2] * inv_len]
}

/// The clamped cosine the reference derives inside `evaluate`.
fn host_n_dot_v(normal: [f32; 3], view_dir: [f32; 3]) -> f32 {
    clamp01(dot3(normalize3(normal), normalize3(view_dir)))
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

/// A pseudo-random 3-component vector with each component in `[-span, span)`.
fn rand_vec(state: &mut u64, span: f32) -> [f32; 3] {
    [
        signed(state, span),
        signed(state, span),
        signed(state, span),
    ]
}

/// Builds the reference parameters from the shared band and the given fields.
fn params(f0: f32, power: u32, rim_color: [f32; 3], intensity: f32) -> RimParams {
    RimParams::new(f0, power, rim_color, intensity, INNER, OUTER)
}

/// Builds a query carrying both the directions and the rim parameters.
fn query(
    normal: [f32; 3],
    view_dir: [f32; 3],
    f0: f32,
    power: u32,
    rim_color: [f32; 3],
    intensity: f32,
) -> FresnelRimQuery {
    FresnelRimQuery::new(
        normal, view_dir, f0, power, rim_color, intensity, INNER, OUTER,
    )
}

/// Draws a query whose derived cosine is clear of the `smoothstep` knots: its
/// `n_dot_v` must land at least `BAND_MARGIN` below `inner` (grazing) or above
/// `outer` (head-on), and both directions must clear a safe non-degenerate
/// length. This keeps `CPU` and `GPU` on the same side of every `clamp01` knot.
fn conditioned_query(state: &mut u64) -> FresnelRimQuery {
    loop {
        let normal = rand_vec(state, 2.0);
        let view_dir = rand_vec(state, 2.0);
        if dot3(normal, normal) < 1.0 || dot3(view_dir, view_dir) < 1.0 {
            continue;
        }
        let n_dot_v = host_n_dot_v(normal, view_dir);
        let clear = n_dot_v <= INNER - BAND_MARGIN || n_dot_v >= OUTER + BAND_MARGIN;
        if !clear {
            continue;
        }
        // Vary the artist controls so different power loop counts are exercised.
        let power = 1 + ((*state >> 7) as u32 & 0x7);
        let f0 = lcg(state) * 0.3;
        let rim_color = [lcg(state), lcg(state), lcg(state)];
        let intensity = 0.5 + lcg(state) * 2.5;
        return query(normal, view_dir, f0, power, rim_color, intensity);
    }
}

/// Pins one `GPU` sample against the `CPU` golden for `q`: the rim factor, the
/// `RGB` contribution, the derived cosine and the standalone `Schlick` value
/// must all agree within bound.
fn pin(idx: usize, q: &FresnelRimQuery, got: &FresnelRimSample) {
    let want = params(q.f0, q.power, q.rim_color, q.intensity).evaluate(q.normal, q.view_dir);
    let want_n_dot_v = host_n_dot_v(q.normal, q.view_dir);
    let want_fresnel = fresnel_schlick(want_n_dot_v, q.f0);

    assert!(
        close(got.factor, want.factor),
        "query {idx} factor: gpu {} vs cpu {}",
        got.factor,
        want.factor
    );
    assert!(
        close(got.rgb[0], want.rgb[0])
            && close(got.rgb[1], want.rgb[1])
            && close(got.rgb[2], want.rgb[2]),
        "query {idx} rgb: gpu ({}, {}, {}) vs cpu ({}, {}, {})",
        got.rgb[0],
        got.rgb[1],
        got.rgb[2],
        want.rgb[0],
        want.rgb[1],
        want.rgb[2]
    );
    assert!(
        close(got.n_dot_v, want_n_dot_v),
        "query {idx} n_dot_v: gpu {} vs cpu {}",
        got.n_dot_v,
        want_n_dot_v
    );
    assert!(
        close(got.fresnel, want_fresnel),
        "query {idx} fresnel: gpu {} vs cpu {}",
        got.fresnel,
        want_fresnel
    );
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuFresnelRim, queries: &[FresnelRimQuery]) {
    let got = gpu.eval(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, q, result);
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFresnelRim::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn head_on_view_has_zero_rim() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFresnelRim::new(&ctx);
    // Normal aligned with the view: n_dot_v = 1 >= outer, the band is zero and
    // the rim vanishes. Well clear of the outer knot.
    let q = query(
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        0.2,
        4,
        [1.0, 0.8, 0.5],
        3.0,
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn grazing_view_has_maximal_rim() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFresnelRim::new(&ctx);
    // Normal perpendicular to the view: n_dot_v = 0 <= inner, the band is one
    // and the rim factor saturates. Well clear of the inner knot.
    let q = query(
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        0.2,
        4,
        [1.0, 0.8, 0.5],
        3.0,
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn back_facing_normal_clamps_to_grazing() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFresnelRim::new(&ctx);
    // A negative dot clamps to zero, so this is grazing too: n_dot_v = 0.
    let q = query(
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
        0.05,
        5,
        [0.4, 0.7, 1.0],
        1.5,
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn power_sweep_at_fixed_grazing_cosine() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFresnelRim::new(&ctx);
    // A fixed grazing cosine (n_dot_v = 0) across a spread of integer exponents,
    // so the power_u32 multiply loop runs a different number of iterations each
    // time while staying clear of the inner knot.
    let queries: Vec<FresnelRimQuery> = [1_u32, 2, 3, 4, 5, 6, 8, 11]
        .into_iter()
        .map(|power| {
            query(
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                0.1,
                power,
                [0.6, 0.3, 0.9],
                2.0,
            )
        })
        .collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFresnelRim::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing the deterministic grazing / head-on fixtures with many
    // rejection-sampled directions, dispatched together so the per-thread
    // indexing and the contiguous storage layout are both exercised.
    let mut queries = vec![
        query(
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            0.2,
            4,
            [1.0, 0.8, 0.5],
            3.0,
        ),
        query(
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            0.2,
            4,
            [1.0, 0.8, 0.5],
            3.0,
        ),
    ];
    for _ in 0..48 {
        queries.push(conditioned_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_conditioned_directions_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFresnelRim::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) of clearly-conditioned
    // directions pins every reported field across many random geometries.
    let queries: Vec<FresnelRimQuery> = (0..200).map(|_| conditioned_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
