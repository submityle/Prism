//! Real-device parity for the integer-`gcd` `u32` twin:
//! [`GpuIntegerGcd`](prism_volumetric_gpu::integer_gcd::GpuIntegerGcd) must
//! reproduce the `CPU` golden
//! [`integer_gcd_lcm`](prism_render_architecture::particle::integer_gcd_lcm)
//! query for query across the Euclidean `gcd`, `Stein`'s binary `gcd` and the
//! coprime predicate.
//!
//! The golden standard is a `u64` domain, so the oracle widens each `u32`
//! operand to `u64`, calls the reference and narrows the answer back to `u32`:
//! the true `gcd` of two `u32` values is itself `<= u32::MAX`, so no narrowing
//! ever truncates. The fixtures cover the degenerate `0` cases
//! (`gcd(0, n)`, `gcd(n, 0)`, `gcd(0, 0)`), coprime pairs, pairs sharing a
//! common factor, equal pairs, large values near `u32::MAX`, powers of two
//! (which exercise `Stein`'s `shift` path) and a `LCG`-generated batch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every operation is pure unsigned integer arithmetic with no rounding, so
//! `CPU` (restricted to the `u32` domain) and `GPU` must agree bit for bit. The
//! comparison is an exact `==` on every `u32` output with no tolerance: any
//! mismatch is a genuine port bug. `WGSL` has no `u64`, so the `u64`-only
//! reference variants (`lcm_u64`, `lcm_checked_u64`, `ext_gcd_i64`) are out of
//! scope here.
//!
//! Provenance: twinned from this repository's
//! [`integer_gcd_lcm`](prism_render_architecture::particle::integer_gcd_lcm);
//! no third-party engine source or derived code.

use prism_render_architecture::particle::integer_gcd_lcm::{binary_gcd_u64, coprime_u64, gcd_u64};
use prism_volumetric_gpu::integer_gcd::{GpuGcdOp, GpuGcdQuery, GpuIntegerGcd};
use prism_volumetric_gpu::GpuContext;

/// A deterministic linear-congruential generator so the randomized fixtures are
/// reproducible bit for bit across runs and platforms (the same constants the
/// `CPU` golden tests use). Pure integer math, no external math library.
fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// A broad fixture of `u32` operand pairs covering the degenerate, structured
/// and random cases the kernel must handle identically to the reference.
fn fixture_pairs() -> Vec<(u32, u32)> {
    let mut pairs = vec![
        // Degenerate zero cases.
        (0, 0),
        (0, 7),
        (7, 0),
        (0, u32::MAX),
        (u32::MAX, 0),
        // Coprime pairs.
        (17, 5),
        (9, 28),
        (1, 1),
        (1, 0),
        (1, u32::MAX),
        // Pairs sharing a common factor.
        (12, 18),
        (100, 75),
        (48, 36),
        (1071, 462),
        (1000, 250),
        // Equal pairs.
        (13, 13),
        (1, 1),
        (u32::MAX, u32::MAX),
        (0xDEAD_BEEF, 0xDEAD_BEEF),
        // Large values near u32::MAX.
        (u32::MAX, u32::MAX - 1),
        (u32::MAX - 1, u32::MAX),
        (u32::MAX, 2),
        (4_294_967_291, 4_294_967_279),
    ];
    // Powers of two, exercising Stein's shift path (common factor of two).
    for i in 0..32 {
        for j in 0..32 {
            pairs.push((1u32 << i, 1u32 << j));
        }
    }
    // A LCG-generated tail: full-range operands plus a few scaled-down pairs so
    // non-trivial common factors show up more often than at random.
    let mut state = 0x0123_4567_89AB_CDEF_u64;
    for _ in 0..2048 {
        let a = (lcg_next(&mut state) >> 32) as u32;
        let b = (lcg_next(&mut state) >> 32) as u32;
        pairs.push((a, b));
        // Mask to smaller ranges so shared factors are more likely.
        pairs.push((a & 0xFFFF, b & 0xFFFF));
        pairs.push((a & 0xFF, b & 0xFF));
    }
    pairs
}

/// The `u32`-domain oracle for one operation: widen to `u64`, call the golden
/// reference and narrow back to `u32`.
fn oracle(op: GpuGcdOp, a: u32, b: u32) -> u32 {
    match op {
        GpuGcdOp::Gcd => gcd_u64(u64::from(a), u64::from(b)) as u32,
        GpuGcdOp::BinaryGcd => binary_gcd_u64(u64::from(a), u64::from(b)) as u32,
        GpuGcdOp::Coprime => u32::from(coprime_u64(u64::from(a), u64::from(b))),
    }
}

/// Runs the full fixture through one operation on-device and asserts an exact
/// match against the oracle, query for query.
fn check_op(gpu: &GpuIntegerGcd, ctx: &GpuContext, op: GpuGcdOp) {
    let pairs = fixture_pairs();
    let queries: Vec<GpuGcdQuery> = pairs
        .iter()
        .map(|&(a, b)| GpuGcdQuery::new(a, b, op))
        .collect();
    let got = gpu.run(ctx, &queries);
    assert_eq!(got.len(), pairs.len());
    for (idx, &(a, b)) in pairs.iter().enumerate() {
        assert_eq!(
            got[idx],
            oracle(op, a, b),
            "{op:?} mismatch at {idx} for inputs a={a:#010x} b={b:#010x}"
        );
    }
}

#[test]
fn gcd_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntegerGcd::new(&ctx);
    check_op(&gpu, &ctx, GpuGcdOp::Gcd);
}

#[test]
fn binary_gcd_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntegerGcd::new(&ctx);
    check_op(&gpu, &ctx, GpuGcdOp::BinaryGcd);
}

#[test]
fn coprime_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntegerGcd::new(&ctx);
    check_op(&gpu, &ctx, GpuGcdOp::Coprime);
}

#[test]
fn euclidean_and_binary_gcd_agree_on_device() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntegerGcd::new(&ctx);
    let pairs = fixture_pairs();
    let euclid: Vec<GpuGcdQuery> = pairs
        .iter()
        .map(|&(a, b)| GpuGcdQuery::new(a, b, GpuGcdOp::Gcd))
        .collect();
    let stein: Vec<GpuGcdQuery> = pairs
        .iter()
        .map(|&(a, b)| GpuGcdQuery::new(a, b, GpuGcdOp::BinaryGcd))
        .collect();
    // Both gcd kernels must agree on every input, not just against the oracle.
    assert_eq!(gpu.run(&ctx, &euclid), gpu.run(&ctx, &stein));
}

#[test]
fn mixed_op_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntegerGcd::new(&ctx);
    // Interleave all three operations in one dispatch so the kernel's switch is
    // exercised across adjacent threads within a workgroup.
    let ops = [GpuGcdOp::Gcd, GpuGcdOp::BinaryGcd, GpuGcdOp::Coprime];
    let pairs = fixture_pairs();
    let queries: Vec<GpuGcdQuery> = pairs
        .iter()
        .enumerate()
        .map(|(idx, &(a, b))| GpuGcdQuery::new(a, b, ops[idx % ops.len()]))
        .collect();
    let got = gpu.run(&ctx, &queries);
    assert_eq!(got.len(), queries.len());
    for (idx, q) in queries.iter().enumerate() {
        assert_eq!(
            got[idx],
            oracle(q.op, q.a, q.b),
            "mixed-op mismatch at {idx} for {:?} a={:#010x} b={:#010x}",
            q.op,
            q.a,
            q.b
        );
    }
}

#[test]
fn empty_input_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntegerGcd::new(&ctx);
    assert!(gpu.run(&ctx, &[]).is_empty());
}
