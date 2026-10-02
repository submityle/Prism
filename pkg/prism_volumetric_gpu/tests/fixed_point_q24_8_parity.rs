//! Real-device parity for the pure-`i32` `Q24.8` arithmetic twin:
//! [`GpuFixedPointQ24_8`](prism_volumetric_gpu::fixed_point_q24_8::GpuFixedPointQ24_8)
//! must reproduce the `CPU` golden
//! [`fixed_point_q24_8`](prism_render_architecture::particle::fixed_point_q24_8)
//! query for query across the ten `i32`-closed operations:
//! [`from_int`](prism_render_architecture::particle::fixed_point_q24_8::from_int),
//! [`to_int_trunc`](prism_render_architecture::particle::fixed_point_q24_8::to_int_trunc),
//! [`add`](prism_render_architecture::particle::fixed_point_q24_8::add),
//! [`saturating_add`](prism_render_architecture::particle::fixed_point_q24_8::saturating_add),
//! [`sub`](prism_render_architecture::particle::fixed_point_q24_8::sub),
//! [`saturating_sub`](prism_render_architecture::particle::fixed_point_q24_8::saturating_sub),
//! [`neg`](prism_render_architecture::particle::fixed_point_q24_8::neg),
//! [`floor`](prism_render_architecture::particle::fixed_point_q24_8::floor),
//! [`fract`](prism_render_architecture::particle::fixed_point_q24_8::fract) and
//! [`abs`](prism_render_architecture::particle::fixed_point_q24_8::abs).
//!
//! The fixtures cover positive, negative and zero operands, the near-[`i32::MIN`]
//! / near-[`i32::MAX`] extremes that exercise both the saturating and the
//! non-saturating paths of
//! [`saturating_add`](prism_render_architecture::particle::fixed_point_q24_8::saturating_add)
//! and
//! [`saturating_sub`](prism_render_architecture::particle::fixed_point_q24_8::saturating_sub),
//! the [`i32::MIN`] wrap of
//! [`abs`](prism_render_architecture::particle::fixed_point_q24_8::abs) and
//! [`neg`](prism_render_architecture::particle::fixed_point_q24_8::neg), the
//! floor-toward-negative-infinity behaviour of
//! [`to_int_trunc`](prism_render_architecture::particle::fixed_point_q24_8::to_int_trunc)
//! and
//! [`floor`](prism_render_architecture::particle::fixed_point_q24_8::floor), the
//! always-non-negative
//! [`fract`](prism_render_architecture::particle::fixed_point_q24_8::fract), and
//! a deterministic integer-`LCG` sweep across several workgroups. The `LCG`
//! lives on the host in `u64`, exactly as the golden fixtures allow; the kernel
//! itself stays pure `i32`. Only
//! [`from_int`](prism_render_architecture::particle::fixed_point_q24_8::from_int)
//! inputs are kept in the representable integer range `[-8388608, 8388607]` so
//! the reference `i * 256` does not overflow; every other operation accepts
//! arbitrary operands because both sides wrap or saturate identically.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every output is exact two's-complement `i32` algebra with no rounding
//! anywhere, so `CPU` and `GPU` must agree bit for bit. The comparison is an
//! exact `==` on every output word, with no tolerance: any mismatch is a genuine
//! port bug. The golden
//! [`from_ratio`](prism_render_architecture::particle::fixed_point_q24_8::from_ratio),
//! [`mul`](prism_render_architecture::particle::fixed_point_q24_8::mul) and
//! [`div`](prism_render_architecture::particle::fixed_point_q24_8::div) need an
//! `i64` intermediate that `WGSL` lacks and are out of scope here.
//!
//! Provenance: twinned from this repository's
//! [`fixed_point_q24_8`](prism_render_architecture::particle::fixed_point_q24_8);
//! no third-party engine source or derived code.

use prism_render_architecture::particle::fixed_point_q24_8::{
    abs, add, floor, fract, from_int, neg, saturating_add, saturating_sub, sub, to_int_trunc, Q24_8,
};
use prism_volumetric_gpu::fixed_point_q24_8::{GpuFixedPointQ24_8, GpuQ24Op, GpuQ24Query};
use prism_volumetric_gpu::GpuContext;

/// Smallest representable integer `i` such that `from_int(i)` does not overflow.
const INT_MIN: i32 = -8_388_608;
/// Largest representable integer `i` such that `from_int(i)` does not overflow.
const INT_MAX: i32 = 8_388_607;

/// Small linear-congruential generator for deterministic random samples,
/// mirroring the golden fixtures. The `u64` state lives on the host; the kernel
/// stays `i32`.
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

    fn next_i32(&mut self) -> i32 {
        self.next_u32() as i32
    }
}

/// Reference answer for one query, built straight from the `CPU` golden so the
/// parity assertion compares against the authoritative contract. The golden
/// functions are called directly as the oracle.
fn cpu_result(q: &GpuQ24Query) -> i32 {
    let a = Q24_8 { raw: q.a };
    let b = Q24_8 { raw: q.b };
    match q.op {
        GpuQ24Op::FromInt => from_int(q.a).raw,
        GpuQ24Op::ToIntTrunc => to_int_trunc(a),
        GpuQ24Op::Add => add(a, b).raw,
        GpuQ24Op::SaturatingAdd => saturating_add(a, b).raw,
        GpuQ24Op::Sub => sub(a, b).raw,
        GpuQ24Op::SaturatingSub => saturating_sub(a, b).raw,
        GpuQ24Op::Neg => neg(a).raw,
        GpuQ24Op::Floor => floor(a).raw,
        GpuQ24Op::Fract => fract(a).raw,
        GpuQ24Op::Abs => abs(a).raw,
    }
}

/// Asserts the batch of queries resolves exactly element for element.
fn assert_batch(gpu: &GpuFixedPointQ24_8, ctx: &GpuContext, queries: &[GpuQ24Query]) {
    let got = gpu.run(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (q, g) in queries.iter().zip(got.iter()) {
        let want = cpu_result(q);
        assert_eq!(*g, want, "Q24.8 parity mismatch for query {q:?}");
    }
}

/// Convenience constructor for a binary query.
fn bin(op: GpuQ24Op, a: i32, b: i32) -> GpuQ24Query {
    GpuQ24Query { a, b, op }
}

/// Convenience constructor for a unary query (second operand unused).
fn un(op: GpuQ24Op, a: i32) -> GpuQ24Query {
    GpuQ24Query { a, b: 0, op }
}

#[test]
fn from_int_scales_by_256() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFixedPointQ24_8::new(&ctx);
    // Small positive/negative/zero integers plus the representable boundaries,
    // where from_int must not overflow.
    let queries = [
        un(GpuQ24Op::FromInt, 0),
        un(GpuQ24Op::FromInt, 1),
        un(GpuQ24Op::FromInt, -1),
        un(GpuQ24Op::FromInt, 5),
        un(GpuQ24Op::FromInt, -123),
        un(GpuQ24Op::FromInt, INT_MAX),
        un(GpuQ24Op::FromInt, INT_MIN),
    ];
    assert_batch(&gpu, &ctx, &queries);
}

#[test]
fn to_int_trunc_floors_toward_negative_infinity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFixedPointQ24_8::new(&ctx);
    // -2.5 (raw -640) floors to -3; 2.5 (raw 640) floors to 2; integers and the
    // extremes round-trip through the arithmetic shift.
    let queries = [
        un(GpuQ24Op::ToIntTrunc, 640),
        un(GpuQ24Op::ToIntTrunc, -640),
        un(GpuQ24Op::ToIntTrunc, 256),
        un(GpuQ24Op::ToIntTrunc, -256),
        un(GpuQ24Op::ToIntTrunc, 0),
        un(GpuQ24Op::ToIntTrunc, 1),
        un(GpuQ24Op::ToIntTrunc, -1),
        un(GpuQ24Op::ToIntTrunc, i32::MAX),
        un(GpuQ24Op::ToIntTrunc, i32::MIN),
    ];
    assert_batch(&gpu, &ctx, &queries);
}

#[test]
fn add_and_sub_wrap_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFixedPointQ24_8::new(&ctx);
    // Ordinary sums and differences plus two cases that overflow and must wrap
    // identically on both sides.
    let queries = [
        bin(GpuQ24Op::Add, 256, 256),
        bin(GpuQ24Op::Add, 1280, -512),
        bin(GpuQ24Op::Add, i32::MAX, 1),
        bin(GpuQ24Op::Sub, 1280, 512),
        bin(GpuQ24Op::Sub, 512, 1280),
        bin(GpuQ24Op::Sub, i32::MIN, 1),
    ];
    assert_batch(&gpu, &ctx, &queries);
}

#[test]
fn saturating_add_covers_both_paths() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFixedPointQ24_8::new(&ctx);
    // Non-saturating sums, high saturation (positive overflow to MAX) and low
    // saturation (negative overflow to MIN).
    let queries = [
        bin(GpuQ24Op::SaturatingAdd, 256, 256),
        bin(GpuQ24Op::SaturatingAdd, -256, -256),
        bin(GpuQ24Op::SaturatingAdd, i32::MAX, 1),
        bin(GpuQ24Op::SaturatingAdd, i32::MAX, i32::MAX),
        bin(GpuQ24Op::SaturatingAdd, i32::MIN, -1),
        bin(GpuQ24Op::SaturatingAdd, i32::MIN, i32::MIN),
        bin(GpuQ24Op::SaturatingAdd, i32::MAX, i32::MIN),
    ];
    assert_batch(&gpu, &ctx, &queries);
}

#[test]
fn saturating_sub_covers_both_paths() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFixedPointQ24_8::new(&ctx);
    // Non-saturating differences, high saturation (positive minus negative) and
    // low saturation (negative minus positive).
    let queries = [
        bin(GpuQ24Op::SaturatingSub, 1280, 512),
        bin(GpuQ24Op::SaturatingSub, 512, 1280),
        bin(GpuQ24Op::SaturatingSub, i32::MAX, -1),
        bin(GpuQ24Op::SaturatingSub, i32::MAX, i32::MIN),
        bin(GpuQ24Op::SaturatingSub, i32::MIN, 1),
        bin(GpuQ24Op::SaturatingSub, i32::MIN, i32::MAX),
    ];
    assert_batch(&gpu, &ctx, &queries);
}

#[test]
fn neg_including_min_wrap() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFixedPointQ24_8::new(&ctx);
    // Ordinary negation plus the i32::MIN wrap, which stays i32::MIN.
    let queries = [
        un(GpuQ24Op::Neg, 0),
        un(GpuQ24Op::Neg, 768),
        un(GpuQ24Op::Neg, -768),
        un(GpuQ24Op::Neg, i32::MAX),
        un(GpuQ24Op::Neg, i32::MIN),
    ];
    assert_batch(&gpu, &ctx, &queries);
}

#[test]
fn floor_positive_and_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFixedPointQ24_8::new(&ctx);
    // 2.5 floors to 2.0 (raw 512); -2.5 floors to -3.0 (raw -768); integers are
    // unchanged.
    let queries = [
        un(GpuQ24Op::Floor, 640),
        un(GpuQ24Op::Floor, -640),
        un(GpuQ24Op::Floor, 1024),
        un(GpuQ24Op::Floor, -1024),
        un(GpuQ24Op::Floor, 0),
        un(GpuQ24Op::Floor, i32::MAX),
        un(GpuQ24Op::Floor, i32::MIN),
    ];
    assert_batch(&gpu, &ctx, &queries);
}

#[test]
fn fract_is_non_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFixedPointQ24_8::new(&ctx);
    // The fractional part is always in [0, 256), even for negative inputs.
    let queries = [
        un(GpuQ24Op::Fract, 640),
        un(GpuQ24Op::Fract, -640),
        un(GpuQ24Op::Fract, 0),
        un(GpuQ24Op::Fract, 255),
        un(GpuQ24Op::Fract, -1),
        un(GpuQ24Op::Fract, i32::MIN),
    ];
    assert_batch(&gpu, &ctx, &queries);
    // Spot-check the invariant directly: every fract output lands in [0, 256).
    let got = gpu.run(&ctx, &queries);
    for value in got {
        assert!((0..256).contains(&value), "fract out of range: {value}");
    }
}

#[test]
fn abs_including_min_wrap() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFixedPointQ24_8::new(&ctx);
    // abs is wrapping, not saturating: i32::MIN maps back to i32::MIN.
    let queries = [
        un(GpuQ24Op::Abs, 0),
        un(GpuQ24Op::Abs, 1792),
        un(GpuQ24Op::Abs, -1792),
        un(GpuQ24Op::Abs, i32::MAX),
        un(GpuQ24Op::Abs, i32::MIN),
    ];
    assert_batch(&gpu, &ctx, &queries);
}

#[test]
fn random_sweep_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFixedPointQ24_8::new(&ctx);
    // 256 pseudo-random queries span four workgroups, exercising all ten ops
    // against arbitrary i32 operands. FromInt operands are constrained to the
    // representable integer range so the reference i * 256 does not overflow.
    const OPS: [GpuQ24Op; 10] = [
        GpuQ24Op::FromInt,
        GpuQ24Op::ToIntTrunc,
        GpuQ24Op::Add,
        GpuQ24Op::SaturatingAdd,
        GpuQ24Op::Sub,
        GpuQ24Op::SaturatingSub,
        GpuQ24Op::Neg,
        GpuQ24Op::Floor,
        GpuQ24Op::Fract,
        GpuQ24Op::Abs,
    ];
    let mut rng = Lcg::new(0x2024_1002_5151_2718);
    let mut queries = Vec::new();
    for _ in 0..256 {
        let op = OPS[(rng.next_u32() % 10) as usize];
        let b = rng.next_i32();
        let a = if op == GpuQ24Op::FromInt {
            // Map into [INT_MIN, INT_MAX] so from_int cannot overflow.
            let span = (INT_MAX as i64 - INT_MIN as i64 + 1) as u64;
            INT_MIN + (rng.next_u32() as u64 % span) as i32
        } else {
            rng.next_i32()
        };
        queries.push(GpuQ24Query { a, b, op });
    }
    assert_batch(&gpu, &ctx, &queries);
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFixedPointQ24_8::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.run(&ctx, &[]).is_empty());
}
