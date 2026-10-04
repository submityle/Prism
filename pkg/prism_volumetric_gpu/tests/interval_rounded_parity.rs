//! Real-device parity for the outward-rounded interval-arithmetic twin:
//! [`GpuIntervalRounded`](prism_volumetric_gpu::interval_rounded::GpuIntervalRounded)
//! must reproduce the `CPU` golden `prism_math::interval` family. Each query
//! carries an operation id (`0..=13`), two intervals `a`/`b` and one scalar `v`;
//! any id `> 13` is rejected as invalid.
//!
//! The oracle here is an independent re-implementation of all fourteen closed
//! forms plus the `ULP` steppers `next_up`/`next_down` and `min4`/`max4`,
//! written out directly so the test never imports `prism_math`,
//! `prism_render_architecture`, `prism_physics_core` or `glam`. The `ULP`
//! steppers use integer bit arithmetic (`f32::to_bits`/`from_bits`) and are
//! therefore bit-exact; the continuous bound arithmetic may be contracted by
//! the device, so bounds are compared with a tolerance while the discrete
//! `flag` and `valid` words are compared exactly.
//!
//! The fixtures cover hand-verified spot values (ordering, width, midpoint,
//! `contains`, `overlaps`, `hull`, `intersect` empty/non-empty, `abs` across its
//! three branches, `sqrt` clamping a negative lower bound, `neg`, `add`, `sub`,
//! `mul`, and `div` with a divisor straddling zero producing the unbounded
//! interval), an out-of-range id, a mixed batch of two or more elements that
//! validates the `std430` stride, an empty batch the host short-circuits, and a
//! 512-step sweep over all fourteen ids kept away from the discrete knees.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! Each continuous bound is compared with `abs <= 1e-4 || rel <= 1e-3`
//! (`REL_FLOOR = 1e-6`); the discrete `flag` and `valid` words are compared
//! exactly. Infinite bounds (from `div` straddling zero) must match exactly.
//!
//! Provenance: 孪生自本仓 `prism_math::interval`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::interval_rounded::{
    GpuIntervalRounded, IntervalRoundedQuery, IntervalRoundedResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Smallest `f32` strictly greater than `x`, re-implementing the golden
/// `next_up` with the same integer bit arithmetic as the kernel. `NaN` and
/// `+inf` are returned unchanged; both signed zeros step to the smallest
/// positive subnormal.
fn next_up(x: f32) -> f32 {
    let bits = x.to_bits();
    let exp = bits & 0x7f80_0000;
    let mant = bits & 0x007f_ffff;
    let is_nan = exp == 0x7f80_0000 && mant != 0;
    let is_pinf = bits == 0x7f80_0000;
    if is_nan || is_pinf {
        return x;
    }
    if bits & 0x7fff_ffff == 0 {
        return f32::from_bits(1);
    }
    let stepped = if bits >> 31 == 0 {
        bits.wrapping_add(1)
    } else {
        bits.wrapping_sub(1)
    };
    f32::from_bits(stepped)
}

/// Smallest `f32` strictly less than `x`, re-implementing the golden
/// `next_down`. `NaN` and `-inf` are returned unchanged; both signed zeros step
/// to the smallest negative subnormal.
fn next_down(x: f32) -> f32 {
    let bits = x.to_bits();
    let exp = bits & 0x7f80_0000;
    let mant = bits & 0x007f_ffff;
    let is_nan = exp == 0x7f80_0000 && mant != 0;
    let is_ninf = bits == 0xff80_0000;
    if is_nan || is_ninf {
        return x;
    }
    if bits & 0x7fff_ffff == 0 {
        return f32::from_bits(0x8000_0001);
    }
    let stepped = if bits >> 31 == 0 {
        bits.wrapping_sub(1)
    } else {
        bits.wrapping_add(1)
    };
    f32::from_bits(stepped)
}

/// Minimum of four values, matching the kernel's `min4`.
fn min4(a: f32, b: f32, c: f32, d: f32) -> f32 {
    a.min(b).min(c).min(d)
}

/// Maximum of four values, matching the kernel's `max4`.
fn max4(a: f32, b: f32, c: f32, d: f32) -> f32 {
    a.max(b).max(c).max(d)
}

/// Independent host oracle: reproduces every golden interval operation in
/// operator order, returning `(lo, hi, flag, valid)`. An id `> 13`, or an empty
/// `intersect`, yields `valid = 0`.
fn oracle(q: &IntervalRoundedQuery) -> (f32, f32, u32, u32) {
    let alo = q.a_lo;
    let ahi = q.a_hi;
    let blo = q.b_lo;
    let bhi = q.b_hi;
    let sv = q.v;

    let mut lo = 0.0f32;
    let mut hi = 0.0f32;
    let mut flag = 0u32;
    let mut valid = 1u32;

    match q.op_id {
        0 => {
            lo = alo.min(ahi);
            hi = alo.max(ahi);
        }
        1 => {
            let w = next_up(ahi - alo);
            lo = w;
            hi = w;
        }
        2 => {
            let m = alo + (ahi - alo) * 0.5;
            lo = m;
            hi = m;
        }
        3 => {
            let c = alo <= sv && sv <= ahi;
            flag = u32::from(c);
        }
        4 => {
            let c = alo <= bhi && blo <= ahi;
            flag = u32::from(c);
        }
        5 => {
            lo = alo.min(blo);
            hi = ahi.max(bhi);
        }
        6 => {
            let il = alo.max(blo);
            let ih = ahi.min(bhi);
            if il <= ih {
                lo = il;
                hi = ih;
            } else {
                valid = 0;
            }
        }
        7 => {
            if alo >= 0.0 {
                lo = alo;
                hi = ahi;
            } else if ahi <= 0.0 {
                lo = -ahi;
                hi = -alo;
            } else {
                lo = 0.0;
                hi = alo.abs().max(ahi.abs());
            }
        }
        8 => {
            let sl = alo.max(0.0).sqrt();
            let sh = ahi.max(0.0).sqrt();
            lo = next_down(sl);
            hi = next_up(sh);
        }
        9 => {
            lo = -ahi;
            hi = -alo;
        }
        10 => {
            lo = next_down(alo + blo);
            hi = next_up(ahi + bhi);
        }
        11 => {
            lo = next_down(alo - bhi);
            hi = next_up(ahi - blo);
        }
        12 => {
            let p0 = alo * blo;
            let p1 = alo * bhi;
            let p2 = ahi * blo;
            let p3 = ahi * bhi;
            lo = next_down(min4(p0, p1, p2, p3));
            hi = next_up(max4(p0, p1, p2, p3));
        }
        13 => {
            if blo <= 0.0 && bhi >= 0.0 {
                lo = f32::NEG_INFINITY;
                hi = f32::INFINITY;
            } else {
                let r0 = alo / blo;
                let r1 = alo / bhi;
                let r2 = ahi / blo;
                let r3 = ahi / bhi;
                lo = next_down(min4(r0, r1, r2, r3));
                hi = next_up(max4(r0, r1, r2, r3));
            }
        }
        _ => {
            valid = 0;
        }
    }
    (lo, hi, flag, valid)
}

/// Mixed absolute-or-relative closeness for a continuous bound. Infinite bounds
/// must match exactly (sign included); any `NaN` fails.
fn close(a: f32, b: f32) -> bool {
    if a.is_infinite() || b.is_infinite() {
        return a == b;
    }
    if a.is_nan() || b.is_nan() {
        return false;
    }
    let diff = (a - b).abs();
    if diff <= 1.0e-4 {
        return true;
    }
    diff <= 1.0e-3 * a.abs().max(b.abs()).max(REL_FLOOR)
}

/// Asserts a single `GPU` result matches the independent oracle: the discrete
/// `flag` and `valid` words exactly, and the bounds to tolerance when valid.
fn assert_parity(gpu: &IntervalRoundedResult, q: &IntervalRoundedQuery, label: &str) {
    let (lo, hi, flag, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    assert_eq!(gpu.flag, flag, "{label}: flag mismatch");
    if valid == 1 {
        assert!(
            close(gpu.lo, lo),
            "{label}: lo mismatch gpu={} oracle={}",
            gpu.lo,
            lo
        );
        assert!(
            close(gpu.hi, hi),
            "{label}: hi mismatch gpu={} oracle={}",
            gpu.hi,
            hi
        );
    }
}

#[test]
fn new_orders_and_scalars_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalRounded::new(&ctx);
    let queries = vec![
        IntervalRoundedQuery::new(0, 3.0, -1.0, 0.0, 0.0, 0.0), // new -> (-1, 3)
        IntervalRoundedQuery::new(0, 2.0, 2.0, 0.0, 0.0, 0.0),  // point -> (2, 2)
        IntervalRoundedQuery::new(1, -1.0, 3.0, 0.0, 0.0, 0.0), // width ~ 4
        IntervalRoundedQuery::new(2, -1.0, 3.0, 0.0, 0.0, 0.0), // midpoint = 1
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert!(close(out[0].lo, -1.0) && close(out[0].hi, 3.0));
    assert!(close(out[1].lo, 2.0) && close(out[1].hi, 2.0));
    assert!(close(out[2].lo, 4.0) && close(out[2].hi, 4.0));
    assert!(close(out[3].lo, 1.0) && close(out[3].hi, 1.0));
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("new[{i}]"));
    }
}

#[test]
fn contains_and_overlaps_flags_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalRounded::new(&ctx);
    let queries = vec![
        IntervalRoundedQuery::new(3, -1.0, 3.0, 0.0, 0.0, 1.0), // contains(1) -> true
        IntervalRoundedQuery::new(3, -1.0, 3.0, 0.0, 0.0, 5.0), // contains(5) -> false
        IntervalRoundedQuery::new(4, -1.0, 3.0, 2.0, 6.0, 0.0), // overlaps -> true
        IntervalRoundedQuery::new(4, -1.0, 3.0, 7.0, 9.0, 0.0), // overlaps -> false
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[0].flag, 1);
    assert_eq!(out[1].flag, 0);
    assert_eq!(out[2].flag, 1);
    assert_eq!(out[3].flag, 0);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("flags[{i}]"));
    }
}

#[test]
fn hull_and_intersect_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalRounded::new(&ctx);
    let queries = vec![
        IntervalRoundedQuery::new(5, -1.0, 3.0, 2.0, 6.0, 0.0), // hull -> (-1, 6)
        IntervalRoundedQuery::new(6, -1.0, 3.0, 2.0, 6.0, 0.0), // intersect -> (2, 3)
        IntervalRoundedQuery::new(6, -1.0, 3.0, 7.0, 9.0, 0.0), // intersect empty -> invalid
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert!(close(out[0].lo, -1.0) && close(out[0].hi, 6.0));
    assert_eq!(out[1].valid, 1);
    assert!(close(out[1].lo, 2.0) && close(out[1].hi, 3.0));
    assert_eq!(out[2].valid, 0, "disjoint intersection should be invalid");
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("hull_isect[{i}]"));
    }
}

#[test]
fn abs_branches_and_sqrt_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalRounded::new(&ctx);
    let queries = vec![
        IntervalRoundedQuery::new(7, -3.0, 2.0, 0.0, 0.0, 0.0), // straddling -> (0, 3)
        IntervalRoundedQuery::new(7, -5.0, -2.0, 0.0, 0.0, 0.0), // negative -> (2, 5)
        IntervalRoundedQuery::new(7, 1.0, 4.0, 0.0, 0.0, 0.0),  // positive -> (1, 4)
        IntervalRoundedQuery::new(8, -4.0, 9.0, 0.0, 0.0, 0.0), // sqrt -> (~0, ~3)
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert!(close(out[0].lo, 0.0) && close(out[0].hi, 3.0));
    assert!(close(out[1].lo, 2.0) && close(out[1].hi, 5.0));
    assert!(close(out[2].lo, 1.0) && close(out[2].hi, 4.0));
    assert!(close(out[3].lo, 0.0) && close(out[3].hi, 3.0));
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("abs_sqrt[{i}]"));
    }
}

#[test]
fn arithmetic_ops_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalRounded::new(&ctx);
    let queries = vec![
        IntervalRoundedQuery::new(9, -1.0, 3.0, 0.0, 0.0, 0.0), // neg -> (-3, 1)
        IntervalRoundedQuery::new(10, -1.0, 3.0, 2.0, 6.0, 0.0), // add -> (1, 9)
        IntervalRoundedQuery::new(11, -1.0, 3.0, 2.0, 6.0, 0.0), // sub -> (-7, 1)
        IntervalRoundedQuery::new(12, -1.0, 3.0, 2.0, 6.0, 0.0), // mul -> (-6, 18)
        IntervalRoundedQuery::new(13, 2.0, 6.0, 1.0, 2.0, 0.0), // div positive -> (1, 6)
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert!(close(out[0].lo, -3.0) && close(out[0].hi, 1.0));
    assert!(close(out[1].lo, 1.0) && close(out[1].hi, 9.0));
    assert!(close(out[2].lo, -7.0) && close(out[2].hi, 1.0));
    assert!(close(out[3].lo, -6.0) && close(out[3].hi, 18.0));
    assert!(close(out[4].lo, 1.0) && close(out[4].hi, 6.0));
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("arith[{i}]"));
    }
}

#[test]
fn div_straddling_zero_is_unbounded() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalRounded::new(&ctx);
    let q = IntervalRoundedQuery::new(13, 1.0, 2.0, -1.0, 1.0, 0.0);
    let out = gpu.evaluate(&ctx, &[q]);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[0].lo, f32::NEG_INFINITY, "lower bound should be -inf");
    assert_eq!(out[0].hi, f32::INFINITY, "upper bound should be +inf");
    assert_parity(&out[0], &q, "div_straddle");
}

#[test]
fn out_of_range_op_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalRounded::new(&ctx);
    let a = IntervalRoundedQuery::new(14, 1.0, 2.0, 0.0, 0.0, 0.0);
    let b = IntervalRoundedQuery::new(99, 1.0, 2.0, 0.0, 0.0, 0.0);
    let out = gpu.evaluate(&ctx, &[a, b]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].valid, 0, "op 14 should be invalid");
    assert_eq!(out[1].valid, 0, "op 99 should be invalid");
    assert_parity(&out[0], &a, "oob14");
    assert_parity(&out[1], &b, "oob99");
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalRounded::new(&ctx);
    let queries = vec![
        IntervalRoundedQuery::new(10, -1.0, 3.0, 2.0, 6.0, 0.0),
        IntervalRoundedQuery::new(6, -1.0, 3.0, 7.0, 9.0, 0.0), // empty -> invalid
        IntervalRoundedQuery::new(7, -5.0, -2.0, 0.0, 0.0, 0.0),
        IntervalRoundedQuery::new(50, 0.0, 0.0, 0.0, 0.0, 0.0), // invalid id
        IntervalRoundedQuery::new(13, 2.0, 6.0, -1.0, 1.0, 0.0), // unbounded
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[1].valid, 0);
    assert_eq!(out[3].valid, 0);
    assert_eq!(out[4].lo, f32::NEG_INFINITY);
    assert_eq!(out[4].hi, f32::INFINITY);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalRounded::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalRounded::new(&ctx);
    let mut lcg = Lcg::new(0x197E_7A40);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Cycle through all fourteen ops so each gets at least ~36 samples.
        let op = (queries.len() % 14) as u32;

        // Interval `a`: a modest center with a non-trivial half-width, kept away
        // from the `abs` knee by nudging bounds off zero.
        let center = lcg.next_range(-40.0, 40.0);
        let half = lcg.next_range(1.0, 15.0);
        let mut alo = center - half;
        let mut ahi = center + half;
        if alo.abs() < 0.5 {
            alo -= 0.5;
        }
        if ahi.abs() < 0.5 {
            ahi += 0.5;
        }

        // Interval `b`: centered inside `a` so `overlaps`/`intersect` land
        // clearly non-empty, away from the emptiness knee.
        let bcenter = center + lcg.next_range(-half * 0.5, half * 0.5);
        let bhalf = lcg.next_range(1.0, 15.0);
        let (blo, bhi) = if op == 13 {
            // Keep the divisor strictly positive and clear of zero.
            let lo = lcg.next_range(1.0, 20.0);
            (lo, lo + lcg.next_range(1.0, 15.0))
        } else {
            (bcenter - bhalf, bcenter + bhalf)
        };

        // Scalar well inside `a` so `contains` is unambiguously true.
        let sv = center;

        queries.push(IntervalRoundedQuery::new(op, alo, ahi, blo, bhi, sv));
    }
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("sweep[{i}]"));
    }
}

/// A small deterministic linear-congruential generator; the fixture carries no
/// external randomness. Constants are the Numerical Recipes values.
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Lcg { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// A `[0, 1)` fraction built from the top bits, keeping the fixture pure
    /// integer host-side with no transcendental call.
    fn next_unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// A `[lo, hi)` fraction.
    fn next_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_unit()
    }
}
