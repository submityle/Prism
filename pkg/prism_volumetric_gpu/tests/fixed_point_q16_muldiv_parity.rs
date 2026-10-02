//! Real-device parity for the `Q16.16` multiply/lerp twin:
//! [`GpuQ16MulDiv`](prism_volumetric_gpu::fixed_point_q16_muldiv::GpuQ16MulDiv)
//! must reproduce the `CPU` golden
//! [`fixed_point_q16`](prism_render_architecture::particle::fixed_point_q16)
//! query for query across the saturating multiply
//! ([`q_mul`](prism_render_architecture::particle::fixed_point_q16::q_mul)) and
//! the lerp
//! ([`q_lerp`](prism_render_architecture::particle::fixed_point_q16::q_lerp)).
//!
//! The fixtures exercise the whole `i32` input domain: the extremes
//! [`i32::MIN`] and [`i32::MAX`], `0`, `±ONE` (`65536`), the fractional
//! constants `0.5` (`32768`) and `0.25` (`16384`), small products that stay in
//! range, and large products that overflow and must saturate at both the
//! positive and the negative extreme. The sign matrix (`+ +`, `+ -`, `- +`,
//! `- -`) is covered explicitly so the `64`-bit two's-complement negation is
//! checked both ways. The lerp sweep covers `t = 0`, `t = ONE`, `t = ONE / 2`,
//! `t > ONE` (extrapolation) and negative `t`. A deterministic integer-`LCG`
//! sweep then draws full-range `i32` operand pairs across several workgroups.
//! The `LCG` lives on the host in `u64`, exactly as the golden fixtures allow;
//! the kernel itself stays in the `32`-bit lane set.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every output is exact integer algebra with no rounding error, so `CPU` and
//! `GPU` must agree bit for bit. The comparison is an exact `==` on every `i32`
//! output word, with no tolerance: any mismatch is a genuine port bug. `WGSL`
//! has no `i64`, so the twin rebuilds the `64`-bit path from a `u32` word pair;
//! this test is the evidence that reconstruction is faithful across the full
//! `i32` range.
//!
//! Provenance: twinned from this repository's
//! [`fixed_point_q16`](prism_render_architecture::particle::fixed_point_q16); no
//! third-party engine source or derived code.

use prism_render_architecture::particle::fixed_point_q16::{q_lerp, q_mul, Q16_16};
use prism_volumetric_gpu::fixed_point_q16_muldiv::{GpuQ16MulDiv, GpuQ16MulDivQuery, GpuQ16MulOp};
use prism_volumetric_gpu::GpuContext;

/// Raw `Q16.16` value of `1.0` (the fixed-point scale factor `2^16 = 65536`).
const ONE: i32 = 1 << 16;

/// Small linear-congruential generator for deterministic random samples,
/// mirroring the golden fixtures. The `u64` state lives on the host; the kernel
/// stays in the `32`-bit lane set.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        // Numerical Recipes constants.
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 32) as u32
    }

    /// A full-range `i32` draw (every bit pattern reachable).
    fn next_i32(&mut self) -> i32 {
        self.next_u32() as i32
    }
}

/// Reference answer for one query, built straight from the `CPU` golden so the
/// parity assertion compares against the authoritative contract.
fn cpu_result(q: &GpuQ16MulDivQuery) -> i32 {
    match q.op {
        GpuQ16MulOp::Mul => q_mul(Q16_16(q.a), Q16_16(q.b)).0,
        GpuQ16MulOp::Lerp => q_lerp(Q16_16(q.a), Q16_16(q.b), Q16_16(q.t)).0,
    }
}

/// Asserts the batch of queries resolves exactly element for element.
fn assert_batch(gpu: &GpuQ16MulDiv, ctx: &GpuContext, queries: &[GpuQ16MulDivQuery]) {
    let got = gpu.run(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (q, g) in queries.iter().zip(got.iter()) {
        let want = cpu_result(q);
        assert_eq!(*g, want, "Q16.16 muldiv parity mismatch for query {q:?}");
    }
}

/// Builds a `q_mul` query from two raw operands.
fn mul(a: i32, b: i32) -> GpuQ16MulDivQuery {
    GpuQ16MulDivQuery {
        a,
        b,
        t: 0,
        op: GpuQ16MulOp::Mul,
    }
}

/// Builds a `q_lerp` query from the start, end and interpolation parameter.
fn lerp(a: i32, b: i32, t: i32) -> GpuQ16MulDivQuery {
    GpuQ16MulDivQuery {
        a,
        b,
        t,
        op: GpuQ16MulOp::Lerp,
    }
}

#[test]
fn mul_fractional_and_integer_cases() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQ16MulDiv::new(&ctx);
    let half = ONE / 2;
    let quarter = ONE / 4;
    // Exact fractional and integer products that stay well inside range.
    let queries = [
        mul(0, 0),
        mul(0, ONE),
        mul(ONE, 0),
        mul(ONE, ONE),
        mul(ONE, -ONE),
        mul(-ONE, ONE),
        mul(-ONE, -ONE),
        mul(half, half),
        mul(half, quarter),
        mul(quarter, -quarter),
        mul(3 * ONE, 7 * ONE),
        mul(-5 * ONE, 9 * ONE),
        mul(1, 1),
        mul(1, -1),
        mul(32768, 32768),
        mul(123_456, -654_321),
    ];
    assert_batch(&gpu, &ctx, &queries);
}

#[test]
fn mul_saturation_both_ways() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQ16MulDiv::new(&ctx);
    // Products whose magnitude overflows i32 after the shift, hitting both the
    // positive (i32::MAX) and the negative (i32::MIN) saturation path, plus the
    // extreme operand combinations.
    let big = 1 << 30;
    let queries = [
        mul(i32::MAX, i32::MAX),
        mul(i32::MIN, i32::MIN),
        mul(i32::MAX, i32::MIN),
        mul(i32::MIN, i32::MAX),
        mul(i32::MAX, 2 * ONE),
        mul(i32::MIN, 2 * ONE),
        mul(i32::MAX, -2 * ONE),
        mul(i32::MIN, -2 * ONE),
        mul(big, big),
        mul(big, -big),
        mul(-big, big),
        mul(-big, -big),
        mul(i32::MIN, 1),
        mul(i32::MIN, -1),
        mul(i32::MAX, 1),
        mul(i32::MAX, -1),
    ];
    assert_batch(&gpu, &ctx, &queries);
}

#[test]
fn lerp_endpoints_midpoint_and_extrapolation() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQ16MulDiv::new(&ctx);
    let half = ONE / 2;
    let a = -3 * ONE;
    let b = 5 * ONE;
    // t = 0 -> a, t = ONE -> b, t = ONE/2 -> midpoint, t > ONE and t < 0 extrapolate.
    let queries = [
        lerp(a, b, 0),
        lerp(a, b, ONE),
        lerp(a, b, half),
        lerp(a, b, 2 * ONE),
        lerp(a, b, -ONE),
        lerp(0, ONE, half),
        lerp(ONE, -ONE, half),
        lerp(i32::MIN, i32::MAX, half),
        lerp(i32::MAX, i32::MIN, 2 * ONE),
        lerp(i32::MIN, i32::MAX, -ONE),
        lerp(-7 * ONE, 11 * ONE, 3 * ONE),
        lerp(42, -42, ONE / 4),
    ];
    assert_batch(&gpu, &ctx, &queries);
}

#[test]
fn mul_sign_matrix() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQ16MulDiv::new(&ctx);
    // All four sign pairings over the same magnitudes, so the 64-bit two's
    // complement negation is checked for same-sign (no negate) and opposite-sign.
    let mags = [ONE + 1, 3 * ONE + 7, 123_457, 7_777_777];
    let mut queries = Vec::new();
    for &m in &mags {
        for &n in &mags {
            queries.push(mul(m, n));
            queries.push(mul(m, -n));
            queries.push(mul(-m, n));
            queries.push(mul(-m, -n));
        }
    }
    assert_batch(&gpu, &ctx, &queries);
}

#[test]
fn random_full_range_sweep() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQ16MulDiv::new(&ctx);
    // 512 pseudo-random queries span several workgroups, drawing full-range i32
    // operands so saturation, every sign pairing and the whole shift path are
    // exercised against arbitrary bit patterns.
    let mut rng = Lcg::new(0x5151_2718_2818_2845);
    let mut queries = Vec::new();
    for i in 0..512u32 {
        let a = rng.next_i32();
        let b = rng.next_i32();
        let t = rng.next_i32();
        if i & 1 == 0 {
            queries.push(mul(a, b));
        } else {
            queries.push(lerp(a, b, t));
        }
    }
    assert_batch(&gpu, &ctx, &queries);
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQ16MulDiv::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.run(&ctx, &[]).is_empty());
}
