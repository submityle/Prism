//! Real-device parity for the `CIE` 1976 `L*u*v*` twin:
//! [`GpuCieLuv`](prism_volumetric_gpu::cie_luv::GpuCieLuv) must reproduce the
//! `CPU` golden
//! [`cie_luv`](prism_render_architecture::particle::cie_luv) across the forward
//! and inverse lightness nonlinearity (and the shared Newton cube root), the
//! `u'v'` chromaticity projection, the `XYZ` <-> `CIELUV` conversions, the
//! chroma, saturation, `ΔE*uv` color difference and the component-wise lerp.
//!
//! The fixtures cover the shapes the golden calls out: the cube-root leg and the
//! linear toe of both `luv_f` and `luv_f_inv` (sampled well clear of the `δ³`
//! and `δ` knots), the Newton cube root across perfect cubes, fractions, large
//! magnitudes, zero and negatives, a well-conditioned random batch, and the two
//! robustly-pinned degeneracies — a black `XYZ` whose `u'v'` denominator is
//! exactly zero and a zero-`L*` color whose `luv_to_xyz` collapses to the
//! origin. All deterministic fixtures are written as integers or simple
//! decimals, so they stay pure and need no external math library and no
//! transcendental call.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every twinned quantity is continuous — a sequence of multiplies, adds,
//! divides, one `sqrt` and the bounded Newton loop — so `CPU` and `GPU` are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The comparison therefore allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3`
//! (`REL_FLOOR = 1e-6`) on every `f32` field.
//!
//! # Conditioning
//!
//! Every random fixture is kept well away from the branch cracks: the `XYZ` fed
//! to the chromaticity projection has a tristimulus sum comfortably above zero,
//! the `CIELUV` colors have `L*` comfortably above zero (so neither the
//! lightness guard nor the reconstructed-`v'` guard is approached), and the
//! `luv_f` / `luv_f_inv` arguments sit clear of the `δ³` and `δ` knots. The two
//! guard branches are pinned by separate deterministic fixtures whose guarded
//! denominators are exactly zero.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::cie_luv`；
//! no third-party engine source or derived code.
#![forbid(unsafe_code)]

use prism_render_architecture::particle::cie_luv::{Luv, Xyz};
use prism_volumetric_gpu::cie_luv::{golden, CieLuvQuery, CieLuvResult, GpuCieLuv};
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

/// A pseudo-random value in `[lo, hi)` drawn from `state`.
fn range(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + lcg(state) * (hi - lo)
}

/// Asserts every twinned field of one `GPU` result matches the `CPU` golden.
fn assert_result(got: &CieLuvResult, want: &CieLuvResult) {
    assert!(
        close(got.luv_f, want.luv_f),
        "luv_f mismatch: gpu {} vs cpu {}",
        got.luv_f,
        want.luv_f
    );
    assert!(
        close(got.luv_f_inv, want.luv_f_inv),
        "luv_f_inv mismatch: gpu {} vs cpu {}",
        got.luv_f_inv,
        want.luv_f_inv
    );
    assert!(
        close(got.cbrt, want.cbrt),
        "cbrt mismatch: gpu {} vs cpu {}",
        got.cbrt,
        want.cbrt
    );
    assert!(
        close(got.uv_prime.u_prime, want.uv_prime.u_prime)
            && close(got.uv_prime.v_prime, want.uv_prime.v_prime),
        "uv_prime mismatch: gpu {:?} vs cpu {:?}",
        got.uv_prime,
        want.uv_prime
    );
    assert!(
        close(got.xyz_to_luv.l, want.xyz_to_luv.l)
            && close(got.xyz_to_luv.u, want.xyz_to_luv.u)
            && close(got.xyz_to_luv.v, want.xyz_to_luv.v),
        "xyz_to_luv mismatch: gpu {:?} vs cpu {:?}",
        got.xyz_to_luv,
        want.xyz_to_luv
    );
    assert!(
        close(got.luv_to_xyz.x, want.luv_to_xyz.x)
            && close(got.luv_to_xyz.y, want.luv_to_xyz.y)
            && close(got.luv_to_xyz.z, want.luv_to_xyz.z),
        "luv_to_xyz mismatch: gpu {:?} vs cpu {:?}",
        got.luv_to_xyz,
        want.luv_to_xyz
    );
    assert!(
        close(got.chroma, want.chroma),
        "chroma mismatch: gpu {} vs cpu {}",
        got.chroma,
        want.chroma
    );
    assert!(
        close(got.saturation, want.saturation),
        "saturation mismatch: gpu {} vs cpu {}",
        got.saturation,
        want.saturation
    );
    assert!(
        close(got.delta_e_uv, want.delta_e_uv),
        "delta_e_uv mismatch: gpu {} vs cpu {}",
        got.delta_e_uv,
        want.delta_e_uv
    );
    assert!(
        close(got.luv_lerp.l, want.luv_lerp.l)
            && close(got.luv_lerp.u, want.luv_lerp.u)
            && close(got.luv_lerp.v, want.luv_lerp.v),
        "luv_lerp mismatch: gpu {:?} vs cpu {:?}",
        got.luv_lerp,
        want.luv_lerp
    );
}

/// Dispatches one query and asserts its result matches the `CPU` golden.
fn assert_parity(gpu: &GpuCieLuv, ctx: &GpuContext, q: &CieLuvQuery) {
    let got = gpu.eval(ctx, std::slice::from_ref(q));
    assert_eq!(got.len(), 1, "one result per query");
    assert_result(&got[0], &golden(q));
}

/// Builds a clearly-conditioned random query, sampling every input well away
/// from the guards and the `δ³` / `δ` knots.
fn rand_query(state: &mut u64) -> CieLuvQuery {
    // Each XYZ component is at least 0.1, so the tristimulus sum is at least 0.3
    // and the chromaticity projection never approaches its denominator guard.
    let xyz = Xyz::new(
        range(state, 0.1, 0.9),
        range(state, 0.1, 0.9),
        range(state, 0.1, 0.9),
    );
    // L* in [10, 90) keeps both the lightness guard and the reconstructed-v'
    // guard far away; u*/v* stay moderate so v' stays well clear of zero.
    let luv_a = Luv::new(
        range(state, 10.0, 90.0),
        range(state, -40.0, 40.0),
        range(state, -40.0, 40.0),
    );
    let luv_b = Luv::new(
        range(state, 10.0, 90.0),
        range(state, -40.0, 40.0),
        range(state, -40.0, 40.0),
    );
    // f_t stays on the cube-root leg (knot 0.00886) and f_inv_t on the cube leg
    // (knot 0.2069); cbrt_x is a positive magnitude clear of zero.
    CieLuvQuery::new(
        xyz,
        luv_a,
        luv_b,
        range(state, 0.05, 1.0),
        range(state, 0.3, 1.2),
        range(state, 0.2, 50.0),
        range(state, 0.0, 1.0),
    )
}

#[test]
fn lightness_nonlinearity_both_legs() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCieLuv::new(&ctx);
    // f_t and f_inv_t on their cube-root / cube legs, well above the knots.
    let upper = CieLuvQuery::new(
        Xyz::new(0.4, 0.5, 0.6),
        Luv::new(55.0, 10.0, -20.0),
        Luv::new(40.0, -5.0, 15.0),
        0.5,
        0.8,
        27.0,
        0.5,
    );
    assert_parity(&gpu, &ctx, &upper);
    // f_t = 0.002 and f_inv_t = 0.1 sit on the linear toes, clear below the
    // knots (0.00886 and 0.2069).
    let toe = CieLuvQuery::new(
        Xyz::new(0.3, 0.4, 0.5),
        Luv::new(32.0, -12.0, 7.0),
        Luv::new(70.0, 20.0, -30.0),
        0.002,
        0.1,
        8.0,
        0.25,
    );
    assert_parity(&gpu, &ctx, &toe);
}

#[test]
fn cube_root_covers_sign_and_magnitude() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCieLuv::new(&ctx);
    // Perfect cube, fraction, large magnitude, exact zero and a negative input
    // all exercise the range reduction and the sign reflection.
    for &x in &[0.125_f32, 2.0, 64.0, 1_000_000.0, 0.0, -27.0] {
        let q = CieLuvQuery::new(
            Xyz::new(0.5, 0.5, 0.5),
            Luv::new(50.0, 8.0, -6.0),
            Luv::new(20.0, -4.0, 9.0),
            0.4,
            0.6,
            x,
            0.5,
        );
        assert_parity(&gpu, &ctx, &q);
    }
}

#[test]
fn lerp_returns_endpoints_and_midpoint() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCieLuv::new(&ctx);
    let a = Luv::new(10.0, -20.0, 30.0);
    let b = Luv::new(90.0, 40.0, -50.0);
    for &t in &[0.0_f32, 0.5, 1.0] {
        let q = CieLuvQuery::new(Xyz::new(0.4, 0.5, 0.3), a, b, 0.3, 0.5, 5.0, t);
        assert_parity(&gpu, &ctx, &q);
    }
}

#[test]
fn degenerate_uv_prime_denominator_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCieLuv::new(&ctx);
    // A black XYZ makes the chromaticity denominator X + 15Y + 3Z exactly zero,
    // so both coordinates collapse to zero (matching the reference guard).
    let q = CieLuvQuery::new(
        Xyz::new(0.0, 0.0, 0.0),
        Luv::new(45.0, 12.0, -8.0),
        Luv::new(25.0, -6.0, 10.0),
        0.3,
        0.5,
        9.0,
        0.5,
    );
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn degenerate_zero_lightness_is_black() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCieLuv::new(&ctx);
    // L* exactly zero drives luv_to_xyz through its lightness guard, returning
    // the XYZ origin (matching the reference), and the saturation guard too.
    let q = CieLuvQuery::new(
        Xyz::new(0.3, 0.4, 0.5),
        Luv::new(0.0, 5.0, 5.0),
        Luv::new(60.0, -10.0, 20.0),
        0.3,
        0.5,
        16.0,
        0.5,
    );
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn random_batch_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCieLuv::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    let batch: Vec<CieLuvQuery> = (0..256).map(|_| rand_query(&mut state)).collect();
    let got = gpu.eval(&ctx, &batch);
    assert_eq!(got.len(), batch.len());
    for (result, query) in got.iter().zip(batch.iter()) {
        assert_result(result, &golden(query));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCieLuv::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.eval(&ctx, &[]).is_empty());
}
