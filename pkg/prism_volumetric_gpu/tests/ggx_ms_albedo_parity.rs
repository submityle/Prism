//! Real-device parity for the `GGX` multiple-scattering directional-albedo
//! twin: [`GpuGgxMsAlbedo`](prism_volumetric_gpu::ggx_ms_albedo::GpuGgxMsAlbedo)
//! must reproduce the baked-table lookup of the reference path tracer's
//! energy-compensation core
//! [`ggx_energy`](prism_render_architecture::reference_pt::ggx_energy) — the
//! bilinear `directional_albedo`, the linear `average_albedo` and the scalar
//! Kulla-Conty `multiscatter_lobe` — across node centres, axis boundaries, a
//! roughness sweep and a randomized batch compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! Per the task's isolation rule this suite does not call the golden crate.
//! Instead it embeds an independent, verbatim copy of the baked `ALBEDO` and
//! `AVG_ALBEDO` tables and re-derives the three reference values in-host from
//! their exact closed form (`t = clamp(x * 16 - 0.5, 0, 15)`, `lo = floor(t)`,
//! `hi = min(lo + 1, 15)`, `frac = t - lo`, bilinear / linear blend, clamp to
//! `[0, 1]`, and `f_ms = (1 - E(mu_o)) * (1 - E(mu_i)) / (pi * (1 - E_avg))`).
//! Both the kernel and this oracle therefore read the same literals and perform
//! the same arithmetic independently, so a `GPU == oracle` pass is direct
//! evidence the ported kernel matches the reference lookup, not merely that the
//! shader compiles.
//!
//! # Parity criterion
//!
//! Every output threads through a clamp and a chain of multiplies, adds and one
//! divide, so a `GPU` evaluation may land a few units in the last place from the
//! scalar reference. Each of the three continuous outputs is asserted within
//! `abs_diff <= 1e-5` or `rel_diff <= 1e-4`, tight enough to catch a genuinely
//! wrong port (a transposed table index, a dropped `1 -`, a wrong axis) yet
//! loose enough to admit a legal last-place difference.
//!
//! # Conditioning
//!
//! The piecewise-linear interpolation is globally continuous: at a node
//! boundary one side's `frac` tends to zero while the other's tends to one, and
//! both evaluate to the same node value, so a unit-in-the-last-place `floor`
//! disagreement between `CPU` and `GPU` cannot move the result discontinuously.
//! No discrete-tie rejection sampling is therefore required. The degenerate
//! `denom <= 1e-4` guard in `multiscatter_lobe` is unreachable with the baked
//! table (the largest `E_avg` node is `0.99608`, so `denom >= 0.00392` always);
//! both the kernel and the oracle carry the identical guard, so parity holds
//! regardless, and no fixture can force the branch.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::reference_pt::ggx_energy`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::ggx_ms_albedo::{GgxMsAlbedoQuery, GgxMsAlbedoResult, GpuGgxMsAlbedo};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a continuous output. A `GPU` evaluation may land a
/// few units in the last place from the scalar reference; `1e-5` admits that
/// legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-5;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-4;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Lookup-table edge length: a `16x16` directional-albedo grid and a `16`-entry
/// average-albedo table.
const LUT_SIZE: usize = 16;

/// Independent, verbatim copy of the baked single-scatter directional albedo
/// `E(cos_theta, alpha)` under a white (`F = 1`) `Fresnel` term, row-major as
/// `ALBEDO[alpha_node][cos_theta_node]`, each node a cell centre at
/// `(i + 0.5) / 16`.
const ALBEDO: [[f32; LUT_SIZE]; LUT_SIZE] = [
    [
        0.89053, 0.94328, 0.97589, 0.98818, 0.99206, 0.99439, 0.99607, 0.99685, 0.99744, 0.99806,
        0.99817, 0.99817, 0.99858, 0.99886, 0.99885, 0.99904,
    ],
    [
        0.93451, 0.88498, 0.89344, 0.91887, 0.93761, 0.95195, 0.96259, 0.96975, 0.97565, 0.97898,
        0.98269, 0.98372, 0.98662, 0.98678, 0.98818, 0.98973,
    ],
    [
        0.94954, 0.88988, 0.87099, 0.87141, 0.88501, 0.89715, 0.91246, 0.92474, 0.93381, 0.94122,
        0.94994, 0.95394, 0.95931, 0.96293, 0.96593, 0.96903,
    ],
    [
        0.95385, 0.89693, 0.86369, 0.85375, 0.85025, 0.85593, 0.86688, 0.87682, 0.88543, 0.89518,
        0.90309, 0.91140, 0.91951, 0.92397, 0.93080, 0.93376,
    ],
    [
        0.95163, 0.89433, 0.86047, 0.83623, 0.82537, 0.82326, 0.82529, 0.83233, 0.83790, 0.84373,
        0.85562, 0.86116, 0.87050, 0.87715, 0.88401, 0.88939,
    ],
    [
        0.94932, 0.88946, 0.84891, 0.82225, 0.80592, 0.79468, 0.79105, 0.79034, 0.79559, 0.80038,
        0.80365, 0.80971, 0.81667, 0.82446, 0.82961, 0.83616,
    ],
    [
        0.94567, 0.88040, 0.83628, 0.80512, 0.78428, 0.76896, 0.76043, 0.75496, 0.75359, 0.75547,
        0.75319, 0.75845, 0.76217, 0.76699, 0.77263, 0.77446,
    ],
    [
        0.94142, 0.87175, 0.82330, 0.79018, 0.76237, 0.74273, 0.73166, 0.72252, 0.71253, 0.70897,
        0.70770, 0.70686, 0.70679, 0.70787, 0.71245, 0.71734,
    ],
    [
        0.93671, 0.86121, 0.80691, 0.76759, 0.73937, 0.72034, 0.69780, 0.68750, 0.67520, 0.66904,
        0.66345, 0.66195, 0.65407, 0.65477, 0.65289, 0.65404,
    ],
    [
        0.92915, 0.84826, 0.79412, 0.75430, 0.71678, 0.69452, 0.66872, 0.65459, 0.64058, 0.62912,
        0.62176, 0.61331, 0.60612, 0.60542, 0.60027, 0.59651,
    ],
    [
        0.92425, 0.83505, 0.77650, 0.73150, 0.69712, 0.66509, 0.64125, 0.61974, 0.60482, 0.59115,
        0.57870, 0.56560, 0.56079, 0.55360, 0.54713, 0.54434,
    ],
    [
        0.91756, 0.82414, 0.75863, 0.71164, 0.67160, 0.63993, 0.61433, 0.59389, 0.57071, 0.55353,
        0.53875, 0.52792, 0.51651, 0.50474, 0.50088, 0.49233,
    ],
    [
        0.91143, 0.81112, 0.74357, 0.68803, 0.65290, 0.61420, 0.58876, 0.55983, 0.53882, 0.51787,
        0.50314, 0.48608, 0.47485, 0.46401, 0.45315, 0.44662,
    ],
    [
        0.90468, 0.79955, 0.73071, 0.67325, 0.62805, 0.59150, 0.55480, 0.52925, 0.50862, 0.48782,
        0.46760, 0.45411, 0.43806, 0.42644, 0.41378, 0.40239,
    ],
    [
        0.89889, 0.78596, 0.71155, 0.65008, 0.60615, 0.56783, 0.53284, 0.50402, 0.47753, 0.45741,
        0.43591, 0.41926, 0.39868, 0.38978, 0.37722, 0.36418,
    ],
    [
        0.89346, 0.77409, 0.69544, 0.63218, 0.58563, 0.54142, 0.50747, 0.47595, 0.44834, 0.42990,
        0.40566, 0.38615, 0.36961, 0.35676, 0.34430, 0.32918,
    ],
];

/// Independent, verbatim copy of the baked cosine-weighted average albedo
/// `E_avg(alpha)` for each `alpha` node.
const AVG_ALBEDO: [f32; LUT_SIZE] = [
    0.99608, 0.97482, 0.94267, 0.90348, 0.86012, 0.81500, 0.76850, 0.72253, 0.67764, 0.63568,
    0.59390, 0.55526, 0.51834, 0.48476, 0.45266, 0.42318,
];

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Reconstructs the node split for a continuous axis coordinate `x`: the lower
/// node index, the upper node index and the fraction between them. Uses only
/// `floor` and never an `f32` `==`, matching the house rules.
fn axis_weights(x: f32) -> (usize, usize, f32) {
    let t = (x * LUT_SIZE as f32 - 0.5).clamp(0.0, (LUT_SIZE - 1) as f32);
    let lo = t as usize;
    let hi = (lo + 1).min(LUT_SIZE - 1);
    (lo, hi, t - lo as f32)
}

/// Bilinearly interpolated single-scatter directional albedo, clamped to
/// `[0, 1]`, mirroring the reference closed form.
fn directional_albedo(cos_theta: f32, alpha: f32) -> f32 {
    let (a_lo, a_hi, a_f) = axis_weights(alpha);
    let (c_lo, c_hi, c_f) = axis_weights(cos_theta);
    let row_lo = ALBEDO[a_lo][c_lo] + (ALBEDO[a_lo][c_hi] - ALBEDO[a_lo][c_lo]) * c_f;
    let row_hi = ALBEDO[a_hi][c_lo] + (ALBEDO[a_hi][c_hi] - ALBEDO[a_hi][c_lo]) * c_f;
    (row_lo + (row_hi - row_lo) * a_f).clamp(0.0, 1.0)
}

/// Linearly interpolated cosine-weighted average albedo, clamped to `[0, 1]`.
fn average_albedo(alpha: f32) -> f32 {
    let (lo, hi, f) = axis_weights(alpha);
    (AVG_ALBEDO[lo] + (AVG_ALBEDO[hi] - AVG_ALBEDO[lo]) * f).clamp(0.0, 1.0)
}

/// Scalar Kulla-Conty multiple-scattering lobe, zero once `E_avg` reaches one.
fn multiscatter_lobe(cos_o: f32, cos_i: f32, alpha: f32) -> f32 {
    let e_avg = average_albedo(alpha);
    let denom = 1.0 - e_avg;
    if denom <= 1.0e-4 {
        return 0.0;
    }
    let e_o = directional_albedo(cos_o, alpha);
    let e_i = directional_albedo(cos_i, alpha);
    (1.0 - e_o) * (1.0 - e_i) / (std::f32::consts::PI * denom)
}

/// Builds the full in-host oracle result for one query.
fn oracle(q: &GgxMsAlbedoQuery) -> GgxMsAlbedoResult {
    GgxMsAlbedoResult {
        dir_albedo_o: directional_albedo(q.cos_o, q.alpha),
        dir_albedo_i: directional_albedo(q.cos_i, q.alpha),
        ms_lobe: multiscatter_lobe(q.cos_o, q.cos_i, q.alpha),
    }
}

/// Pins one `GPU` result against the in-host oracle: all three continuous
/// outputs within tolerance.
fn check_query(idx: usize, got: &GgxMsAlbedoResult, want: &GgxMsAlbedoResult) {
    assert!(
        close(got.dir_albedo_o, want.dir_albedo_o),
        "query {idx} dir_albedo_o: gpu {} vs cpu {}",
        got.dir_albedo_o,
        want.dir_albedo_o
    );
    assert!(
        close(got.dir_albedo_i, want.dir_albedo_i),
        "query {idx} dir_albedo_i: gpu {} vs cpu {}",
        got.dir_albedo_i,
        want.dir_albedo_i
    );
    assert!(
        close(got.ms_lobe, want.ms_lobe),
        "query {idx} ms_lobe: gpu {} vs cpu {}",
        got.ms_lobe,
        want.ms_lobe
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuGgxMsAlbedo, queries: &[GgxMsAlbedoQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_query(idx, result, &want);
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

/// Draws a value in `[0, 1]` at milli resolution from `state`.
fn unit(state: &mut u64) -> f32 {
    (lcg(state) % 1001) as f32 / 1000.0
}

/// The deterministic query fixtures, spanning the roughness range, both axis
/// boundaries and a spread of asymmetric cosines.
fn fixture_queries() -> Vec<GgxMsAlbedoQuery> {
    vec![
        // Smooth surface, grazing-to-normal cosines.
        GgxMsAlbedoQuery::new(0.1, 0.9, 0.05),
        GgxMsAlbedoQuery::new(0.4, 0.4, 0.08),
        // Mid roughness, asymmetric cosines.
        GgxMsAlbedoQuery::new(0.2, 0.75, 0.35),
        GgxMsAlbedoQuery::new(0.6, 0.3, 0.5),
        // Rough surface.
        GgxMsAlbedoQuery::new(0.85, 0.15, 0.7),
        GgxMsAlbedoQuery::new(0.5, 0.5, 0.95),
        // Node centres: alpha and cos land exactly on cell centres (i + 0.5)/16.
        GgxMsAlbedoQuery::new(5.5 / 16.0, 11.5 / 16.0, 3.5 / 16.0),
        GgxMsAlbedoQuery::new(2.5 / 16.0, 14.5 / 16.0, 9.5 / 16.0),
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping ggx_ms_albedo parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuGgxMsAlbedo::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn alpha_min_boundary_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxMsAlbedo::new(&ctx);
    // alpha = 0 clamps the node coordinate to the first row (lo = hi = 0).
    check(
        &ctx,
        &gpu,
        &[
            GgxMsAlbedoQuery::new(0.0, 0.0, 0.0),
            GgxMsAlbedoQuery::new(0.3, 0.9, 0.0),
            GgxMsAlbedoQuery::new(0.5, 0.5, 0.0),
        ],
    );
}

#[test]
fn alpha_max_boundary_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxMsAlbedo::new(&ctx);
    // alpha = 1 clamps the node coordinate to the last row (lo = hi = 15).
    check(
        &ctx,
        &gpu,
        &[
            GgxMsAlbedoQuery::new(0.0, 1.0, 1.0),
            GgxMsAlbedoQuery::new(0.25, 0.75, 1.0),
            GgxMsAlbedoQuery::new(0.5, 0.5, 1.0),
        ],
    );
}

#[test]
fn cos_zero_boundary_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxMsAlbedo::new(&ctx);
    // cos = 0 clamps the cos_theta node coordinate to the first column.
    check(
        &ctx,
        &gpu,
        &[
            GgxMsAlbedoQuery::new(0.0, 0.0, 0.3),
            GgxMsAlbedoQuery::new(0.0, 0.6, 0.6),
        ],
    );
}

#[test]
fn cos_one_boundary_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxMsAlbedo::new(&ctx);
    // cos = 1 clamps the cos_theta node coordinate to the last column.
    check(
        &ctx,
        &gpu,
        &[
            GgxMsAlbedoQuery::new(1.0, 1.0, 0.2),
            GgxMsAlbedoQuery::new(1.0, 0.35, 0.8),
        ],
    );
}

#[test]
fn low_roughness_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxMsAlbedo::new(&ctx);
    check(
        &ctx,
        &gpu,
        &[
            GgxMsAlbedoQuery::new(0.15, 0.85, 0.03),
            GgxMsAlbedoQuery::new(0.45, 0.55, 0.07),
            GgxMsAlbedoQuery::new(0.9, 0.1, 0.12),
        ],
    );
}

#[test]
fn mid_roughness_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxMsAlbedo::new(&ctx);
    check(
        &ctx,
        &gpu,
        &[
            GgxMsAlbedoQuery::new(0.2, 0.7, 0.4),
            GgxMsAlbedoQuery::new(0.55, 0.35, 0.5),
            GgxMsAlbedoQuery::new(0.8, 0.25, 0.6),
        ],
    );
}

#[test]
fn high_roughness_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxMsAlbedo::new(&ctx);
    check(
        &ctx,
        &gpu,
        &[
            GgxMsAlbedoQuery::new(0.3, 0.65, 0.78),
            GgxMsAlbedoQuery::new(0.5, 0.5, 0.88),
            GgxMsAlbedoQuery::new(0.95, 0.2, 0.97),
        ],
    );
}

#[test]
fn asymmetric_cosines_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxMsAlbedo::new(&ctx);
    // cos_o and cos_i deliberately far apart so dir_albedo_o and dir_albedo_i
    // land in different table columns.
    check(
        &ctx,
        &gpu,
        &[
            GgxMsAlbedoQuery::new(0.05, 0.95, 0.45),
            GgxMsAlbedoQuery::new(0.95, 0.05, 0.45),
        ],
    );
}

#[test]
fn node_centre_cases_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxMsAlbedo::new(&ctx);
    // Sampling at exact cell centres (i + 0.5)/16 drives frac to zero on both
    // axes, pinning the pure table reads with no interpolation blend.
    let mut queries = Vec::new();
    for a in 0..LUT_SIZE {
        let alpha = (a as f32 + 0.5) / LUT_SIZE as f32;
        let co = ((a % LUT_SIZE) as f32 + 0.5) / LUT_SIZE as f32;
        let ci = (((a + 7) % LUT_SIZE) as f32 + 0.5) / LUT_SIZE as f32;
        queries.push(GgxMsAlbedoQuery::new(co, ci, alpha));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn degenerate_denom_guard_is_consistent() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxMsAlbedo::new(&ctx);
    // The denom <= 1e-4 guard is unreachable (max E_avg node is 0.99608, so
    // denom >= 0.00392), but both the kernel and the oracle carry it, so parity
    // holds at the smoothest alpha where denom is smallest.
    check(
        &ctx,
        &gpu,
        &[
            GgxMsAlbedoQuery::new(0.3, 0.7, 0.0),
            GgxMsAlbedoQuery::new(0.5, 0.5, 1.0 / 32.0),
        ],
    );
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxMsAlbedo::new(&ctx);
    // Every fixture dispatched together so the per-thread indexing and the
    // contiguous output slots are both exercised.
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxMsAlbedo::new(&ctx);
    let mut state = 0x0bad_c0de_dead_beef_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random queries across the full cos/alpha box
    // pin every output over a wide span of the table.
    for _ in 0..384 {
        let cos_o = unit(&mut state);
        let cos_i = unit(&mut state);
        let alpha = unit(&mut state);
        queries.push(GgxMsAlbedoQuery::new(cos_o, cos_i, alpha));
    }
    check(&ctx, &gpu, &queries);
}
