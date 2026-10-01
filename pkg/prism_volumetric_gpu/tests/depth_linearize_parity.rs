//! Real-device parity for the perspective depth-linearization twin:
//! [`GpuDepthLinearize`](prism_volumetric_gpu::depth_linearize::GpuDepthLinearize)
//! must reproduce the `CPU` golden
//! [`depth_linearize`](prism_render_architecture::particle::depth_linearize),
//! field for field, across the near and far planes, orthographic (`linear_to_01_normalized`)
//! versus perspective (`linearize_01` / `ndc_to_view_z`) remaps, the
//! `linearize` ↔ `delinearize` round trip, `perspective_interpolate` at
//! `t ∈ {0, 0.5, 1}`, a large random batch kept clear of every division crack,
//! the degenerate (`near == far`, non-positive linear, zero reciprocal-weight)
//! fallbacks, and the batch [`linearize_buffer`] entry against both the scalar
//! golden and the empty-input short circuit.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full dispatch-and-
//! readback on any real device such as an Apple `M`-series `GPU`. The kernels
//! are portable core-`WGSL`, so they need no optional device feature.
//!
//! # Parity criterion
//!
//! Every output is a fixed, non-reorderable sequence of multiplies, adds and
//! one divide, so `CPU` and `GPU` evaluate the same closed form in the same
//! order. They are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits by a few units
//! in the last place. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` (relative floor `1e-6`) -- loose enough to admit a legal
//! fused multiply-add contraction yet tight enough to fail a genuinely wrong
//! port (a swapped `near` / `far`, a dropped reverse-`Z` flip, a missing
//! degenerate guard). Every non-degenerate fixture keeps `far > near > 0`,
//! `depth ∈ [0, 1]`, `ndc ∈ [-1, 1]` and `linear >= near`, so every denominator
//! stays strictly positive and a `GPU`'s fused multiply-add cannot flip a
//! degenerate branch; the degenerate fixtures sit *exactly* on the branch
//! (`near == far` bit-identical, reciprocal weights exactly `0`) so both paths
//! agree on which fallback to take.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::depth_linearize`；
//! 无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::depth_linearize::{
    delinearize_01, linear_to_01_normalized, linearize_01, linearize_buffer, ndc_to_view_z,
    perspective_interpolate, DepthParams,
};
use prism_volumetric_gpu::depth_linearize::{
    DepthLinearizeQuery, DepthLinearizeResult, GpuDepthLinearize,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the continuous fields.
const EPS_ABS: f32 = 1.0e-4;

/// Relative parity bound on the continuous fields.
const EPS_REL: f32 = 1.0e-3;

/// Floor keeping the relative-error denominator away from zero.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS_ABS || rel <= EPS_REL
}

/// Builds a depth-linearization query from explicit scalars.
#[expect(
    clippy::too_many_arguments,
    reason = "a single query drives every portable function, so it carries all inputs"
)]
fn q(
    params: DepthParams,
    depth_01: f32,
    linear: f32,
    ndc_z: f32,
    lerp_a: f32,
    lerp_b: f32,
    inv_w_a: f32,
    inv_w_b: f32,
    t: f32,
) -> DepthLinearizeQuery {
    DepthLinearizeQuery {
        params,
        depth_01,
        linear,
        ndc_z,
        lerp_a,
        lerp_b,
        inv_w_a,
        inv_w_b,
        t,
    }
}

/// Asserts one `GPU` [`DepthLinearizeResult`] matches the `CPU` golden for its
/// originating query, field for field.
fn assert_parity(got: &DepthLinearizeResult, qq: &DepthLinearizeQuery, idx: usize) {
    let want_linearized = linearize_01(&qq.params, qq.depth_01);
    let want_delinearized = delinearize_01(&qq.params, qq.linear);
    let want_normalized = linear_to_01_normalized(&qq.params, qq.linear);
    let want_view_z = ndc_to_view_z(&qq.params, qq.ndc_z);
    let want_perspective =
        perspective_interpolate(qq.lerp_a, qq.lerp_b, qq.inv_w_a, qq.inv_w_b, qq.t);
    assert!(
        close(got.linearized, want_linearized),
        "linearized mismatch at query {idx}: gpu {}, cpu {want_linearized}",
        got.linearized,
    );
    assert!(
        close(got.delinearized, want_delinearized),
        "delinearized mismatch at query {idx}: gpu {}, cpu {want_delinearized}",
        got.delinearized,
    );
    assert!(
        close(got.normalized, want_normalized),
        "normalized mismatch at query {idx}: gpu {}, cpu {want_normalized}",
        got.normalized,
    );
    assert!(
        close(got.view_z, want_view_z),
        "view_z mismatch at query {idx}: gpu {}, cpu {want_view_z}",
        got.view_z,
    );
    assert!(
        close(got.perspective, want_perspective),
        "perspective mismatch at query {idx}: gpu {}, cpu {want_perspective}",
        got.perspective,
    );
}

/// Runs the `GPU` evaluation and asserts per-lane parity against the `CPU`
/// golden, returning the device results for any extra property checks.
fn check(
    ctx: &GpuContext,
    gpu: &GpuDepthLinearize,
    queries: &[DepthLinearizeQuery],
) -> Vec<DepthLinearizeResult> {
    let got = gpu.eval(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (idx, (g, qq)) in got.iter().zip(queries.iter()).enumerate() {
        assert_parity(g, qq, idx);
    }
    got
}

/// A deterministic `[0, 1)` pseudo-random stream (`SplitMix64`-style) so the
/// random batch is reproducible without a crate dependency.
fn lcg(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x_9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0x_bf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x_94d0_49bb_1331_11eb);
    z ^= z >> 31;
    // Keep the top 24 bits for a clean [0, 1) f32.
    ((z >> 40) as f32) / ((1u32 << 24) as f32)
}

/// The standard, non-degenerate depth range shared by most fixtures:
/// `far > near > 0`, so every perspective denominator stays strictly positive.
fn params() -> DepthParams {
    DepthParams::new(0.5, 100.0, false)
}

#[test]
fn near_and_far_planes_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDepthLinearize::new(&ctx);
    let p = params();
    // depth 0 -> near, depth 1 -> far; ndc -1 -> near, +1 -> far. `check`
    // asserts GPU/CPU parity; here we additionally pin the known endpoints.
    let queries = [
        q(p, 0.0, 0.5, -1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
        q(p, 1.0, 100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    ];
    let got = check(&ctx, &gpu, &queries);
    assert!(
        close(got[0].linearized, 0.5) && close(got[0].view_z, 0.5),
        "near plane resolves to near"
    );
    assert!(
        close(got[1].linearized, 100.0) && close(got[1].view_z, 100.0),
        "far plane resolves to far"
    );
}

#[test]
fn reverse_z_flips_endpoints() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDepthLinearize::new(&ctx);
    let rev = DepthParams::new(0.5, 100.0, true);
    // Reverse-Z writes 1 at the near plane and 0 at the far plane.
    let queries = [
        q(rev, 1.0, 0.5, 1.0, 2.0, 8.0, 1.5, 0.75, 0.3),
        q(rev, 0.0, 100.0, -1.0, 2.0, 8.0, 1.5, 0.75, 0.6),
    ];
    let got = check(&ctx, &gpu, &queries);
    assert!(
        close(got[0].linearized, 0.5) && close(got[0].view_z, 0.5),
        "reverse-Z near plane (1) resolves to near"
    );
    assert!(
        close(got[1].linearized, 100.0) && close(got[1].view_z, 100.0),
        "reverse-Z far plane (0) resolves to far"
    );
}

#[test]
fn orthographic_normalized_remap_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDepthLinearize::new(&ctx);
    let p = DepthParams::new(2.0, 40.0, false);
    // The plain linear remap: endpoints map to 0 and 1, the midpoint interior.
    let queries = [
        q(p, 0.3, 2.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0),
        q(p, 0.3, 21.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0),
        q(p, 0.3, 40.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0),
    ];
    let got = check(&ctx, &gpu, &queries);
    assert!(close(got[0].normalized, 0.0), "near maps to 0");
    assert!(close(got[2].normalized, 1.0), "far maps to 1");
    assert!(
        got[1].normalized > 0.0 && got[1].normalized < 1.0,
        "the interior sample stays strictly inside the unit range"
    );
}

#[test]
fn linearize_delinearize_roundtrips() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDepthLinearize::new(&ctx);
    for &reverse in &[false, true] {
        let p = DepthParams::new(0.5, 100.0, reverse);
        let depths = [0.0_f32, 0.1, 0.37, 0.5, 0.83, 1.0];
        // Feed each query's `linear` the CPU-linearized depth so the GPU
        // `delinearized` field should return to the original depth.
        let queries: Vec<DepthLinearizeQuery> = depths
            .iter()
            .map(|&d| q(p, d, linearize_01(&p, d), 0.0, 1.0, 1.0, 1.0, 1.0, 0.0))
            .collect();
        let got = check(&ctx, &gpu, &queries);
        for (res, &depth) in got.iter().zip(depths.iter()) {
            assert!(
                close(res.delinearized, depth),
                "linearize->delinearize round trip (reverse_z={reverse}) returns {depth}, got {}",
                res.delinearized,
            );
        }
    }
}

#[test]
fn perspective_interpolate_endpoints_and_midpoint() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDepthLinearize::new(&ctx);
    let p = params();
    // Distinct 1/w weights so the midpoint bends away from the screen-linear
    // average; `t ∈ {0, 0.5, 1}` is covered across the three queries.
    let queries = [
        q(p, 0.4, 10.0, 0.0, 3.0, 9.0, 0.25, 1.0, 0.0),
        q(p, 0.4, 10.0, 0.0, 3.0, 9.0, 0.25, 1.0, 0.5),
        q(p, 0.4, 10.0, 0.0, 3.0, 9.0, 0.25, 1.0, 1.0),
    ];
    let got = check(&ctx, &gpu, &queries);
    assert!(close(got[0].perspective, 3.0), "t=0 returns the first attribute");
    assert!(close(got[2].perspective, 9.0), "t=1 returns the second attribute");
    // Weighted midpoint: (3*0.25 + 9*1.0) / (0.25 + 1.0) = 7.8.
    assert!(close(got[1].perspective, 7.8), "t=0.5 blends perspective-correctly");
}

#[test]
fn degenerate_fallbacks_stay_finite_and_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDepthLinearize::new(&ctx);
    // near == far (bit-identical) so the range guard fires on both paths;
    // linear exactly 0 so delinearize takes the non-positive fallback; the
    // reciprocal weights are exactly 0 so perspective_interpolate falls back to
    // the screen-linear lerp. Each branch is hit on the branch, not near it.
    let degenerate = DepthParams::new(10.0, 10.0, false);
    let degenerate_rev = DepthParams::new(7.0, 7.0, true);
    let queries = [
        q(degenerate, 0.5, 10.0, 0.0, 4.0, 10.0, 0.0, 0.0, 0.5),
        q(degenerate_rev, 0.5, 0.0, 0.25, 4.0, 10.0, 0.0, 0.0, 0.25),
        // A non-degenerate range but linear exactly 0 -> delinearize fallback.
        q(params(), 0.3, 0.0, 0.0, 4.0, 10.0, 0.0, 0.0, 0.75),
    ];
    let got = check(&ctx, &gpu, &queries);
    for r in &got {
        assert!(
            r.linearized.is_finite()
                && r.delinearized.is_finite()
                && r.normalized.is_finite()
                && r.view_z.is_finite()
                && r.perspective.is_finite(),
            "degenerate fallbacks must stay finite"
        );
    }
    // The exactly-degenerate range returns the near-plane distance.
    assert!(close(got[0].linearized, 10.0), "degenerate range linearizes to near");
    assert!(close(got[0].view_z, 10.0), "degenerate range maps NDC to near");
    // Zero reciprocal weights collapse to the screen-linear lerp: 4 + (10-4)*0.5.
    assert!(close(got[0].perspective, 7.0), "zero weights fall back to the lerp");
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDepthLinearize::new(&ctx);
    assert!(gpu.eval(&ctx, &[]).is_empty(), "an empty query batch yields no results");
    assert!(
        gpu.linearize_buffer(&ctx, &params(), &[]).is_empty(),
        "an empty depth slice yields no linearized values"
    );
}

#[test]
fn linearize_buffer_matches_scalar_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDepthLinearize::new(&ctx);
    for &reverse in &[false, true] {
        let p = DepthParams::new(0.5, 100.0, reverse);
        let depths = [0.0_f32, 0.05, 0.2, 0.5, 0.73, 0.9, 1.0];
        let got = gpu.linearize_buffer(&ctx, &p, &depths);
        let want = linearize_buffer(&p, &depths);
        assert_eq!(got.len(), want.len(), "length preserved (reverse_z={reverse})");
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                close(g, w),
                "buffer lane {i} mismatch (reverse_z={reverse}): gpu {g}, cpu {w}"
            );
            // Each lane equals the scalar linearize_01 of the same depth.
            assert!(close(g, linearize_01(&p, depths[i])), "buffer lane {i} equals the scalar form");
        }
    }
}

#[test]
fn random_batch_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDepthLinearize::new(&ctx);
    let mut state = 0x_1234_5678_9abc_def0_u64;

    for _round in 0u32..8 {
        let mut queries: Vec<DepthLinearizeQuery> = Vec::with_capacity(128);
        while queries.len() < 128 {
            // Keep far > near > 0 with a clear gap so the range guard never
            // fires and every perspective denominator stays strictly positive.
            let near = lcg(&mut state) * 2.0 + 0.25;
            let far = near + lcg(&mut state) * 200.0 + 10.0;
            let reverse = lcg(&mut state) < 0.5;
            let p = DepthParams::new(near, far, reverse);
            // depth in [0, 1] and ndc in [-1, 1] keep the perspective forms safe.
            let depth_01 = lcg(&mut state);
            let ndc_z = lcg(&mut state) * 2.0 - 1.0;
            // linear >= near keeps delinearize clear of its non-positive crack.
            let linear = near + lcg(&mut state) * (far - near);
            let lerp_a = lcg(&mut state) * 10.0 - 5.0;
            let lerp_b = lcg(&mut state) * 10.0 - 5.0;
            // Reciprocal weights clearly positive so the weighted blend path is
            // taken, away from the zero-sum fallback.
            let inv_w_a = lcg(&mut state) * 2.0 + 0.25;
            let inv_w_b = lcg(&mut state) * 2.0 + 0.25;
            let t = lcg(&mut state);
            queries.push(q(p, depth_01, linear, ndc_z, lerp_a, lerp_b, inv_w_a, inv_w_b, t));
        }
        // check asserts per-lane parity against the CPU golden.
        let got = check(&ctx, &gpu, &queries);
        // Sanity: a mixed batch spans a wide range of linear distances, so the
        // test is not trivially comparing constants.
        let any_small = got.iter().any(|r| r.linearized < 5.0);
        let any_large = got.iter().any(|r| r.linearized > 20.0);
        assert!(
            any_small && any_large,
            "expected a spread of linearized distances across the batch"
        );
    }
}
