//! Real-device parity for the dual-number twin:
//! [`GpuDual`](prism_volumetric_gpu::dual_number::GpuDual) must reproduce the
//! `CPU` golden `Dual` scalar operations of `prism_math::dual`. A dual number
//! `Dual { re, du }` carries a value `re = f(t)` together with its first
//! derivative `du = f'(t)`; evaluating any supported operation evaluates the
//! function and its exact analytic derivative at once. Each query selects one of
//! sixteen operations by an unsigned `op_id`, and one thread resolves one query.
//!
//! The oracle here is an independent re-implementation of the sixteen closed
//! forms, written out directly so the test never imports
//! `prism_render_architecture`, `prism_physics_core` or `glam`. The one subtle
//! point is the `abs` derivative at exactly zero: the sign-of-zero is taken as
//! zero (not the `f32::signum` convention of `1.0`), matching the golden and the
//! device built-in `sign`.
//!
//! The fixtures cover each operation at representative points, the `abs`
//! sign-of-zero rule, an out-of-range `op_id` (invalid, `re = du = 0`), a batch
//! of two or more elements that mixes operations and an invalid `op_id` to
//! validate the `std430` stride, and an empty batch the host short-circuits with
//! no dispatch. A sweep over random operands follows, keeping each operation in
//! its well-conditioned domain.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The continuous arithmetic threads through operators a `GPU` may contract, so
//! `CPU` and `GPU` evaluate the same closed form but need not be bit-exact. The
//! `re` and `du` scalars are compared with `abs <= 1e-4 || rel <= 1e-3`
//! (`REL_FLOOR = 1e-6`); the discrete `valid` flag is compared exactly. The
//! sweep keeps reciprocal/division denominators away from zero and `sqrt`/`ln`/
//! `powf` bases positive so the two sides agree.
//!
//! Provenance: 孪生自本仓 `prism_math::dual`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::dual_number::{DualQuery, DualResult, GpuDual};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent host oracle: reproduces the sixteen `Dual` operations in the
/// golden operator order, returning the value/derivative pair and the validity
/// flag. An `op_id` outside `0..=15` is invalid with `re = du = 0`.
fn oracle(q: &DualQuery) -> (f32, f32, u32) {
    let (a_re, a_du, b_re, b_du, p) = (q.a_re, q.a_du, q.b_re, q.b_du, q.p);
    let (re, du) = match q.op_id {
        0 => (-a_re, -a_du),
        1 => {
            let inv = 1.0 / a_re;
            (inv, -a_du * inv * inv)
        }
        2 => {
            let r = a_re.sqrt();
            (r, a_du / (2.0 * r))
        }
        3 => (a_re * a_re, 2.0 * a_re * a_du),
        4 => {
            let e = a_re.exp();
            (e, e * a_du)
        }
        5 => (a_re.ln(), a_du / a_re),
        6 => (a_re.sin(), a_re.cos() * a_du),
        7 => (a_re.cos(), -a_re.sin() * a_du),
        8 => {
            let t = a_re.tan();
            (t, (1.0 + t * t) * a_du)
        }
        9 => {
            // sign-of-zero is zero, matching the golden and the device built-in
            // sign (NOT f32::signum, which returns 1.0 at zero).
            let s = if a_re > 0.0 {
                1.0
            } else if a_re < 0.0 {
                -1.0
            } else {
                0.0
            };
            (a_re.abs(), s * a_du)
        }
        10 => (a_re.powf(p), p * a_re.powf(p - 1.0) * a_du),
        11 => (a_re * p, a_du * p),
        12 => (a_re + b_re, a_du + b_du),
        13 => (a_re - b_re, a_du - b_du),
        14 => (a_re * b_re, a_du * b_re + a_re * b_du),
        15 => {
            let inv = 1.0 / b_re;
            (a_re * inv, (a_du * b_re - a_re * b_du) * inv * inv)
        }
        _ => return (0.0, 0.0, 0),
    };
    (re, du, 1)
}

/// Mixed absolute-or-relative closeness for a continuous quantity.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= 1.0e-4 {
        return true;
    }
    diff <= 1.0e-3 * a.abs().max(b.abs()).max(REL_FLOOR)
}

/// Asserts a single `GPU` result matches the independent oracle: the discrete
/// `valid` flag exactly, and the `re`/`du` scalars to tolerance when valid.
fn assert_parity(gpu: &DualResult, q: &DualQuery, label: &str) {
    let (re, du, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    if valid == 1 {
        assert!(
            close(gpu.re, re),
            "{label}: re mismatch gpu={} oracle={}",
            gpu.re,
            re
        );
        assert!(
            close(gpu.du, du),
            "{label}: du mismatch gpu={} oracle={}",
            gpu.du,
            du
        );
    } else {
        assert_eq!(gpu.re, 0.0, "{label}: invalid re should be zero");
        assert_eq!(gpu.du, 0.0, "{label}: invalid du should be zero");
    }
}

#[test]
fn each_operation_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDual::new(&ctx);
    // Representative, well-conditioned point per operation.
    let queries = vec![
        DualQuery::new(0, 1.5, 2.0, 0.0, 0.0, 0.0),  // neg
        DualQuery::new(1, 2.0, 3.0, 0.0, 0.0, 0.0),  // recip
        DualQuery::new(2, 4.0, 1.0, 0.0, 0.0, 0.0),  // sqrt
        DualQuery::new(3, 1.5, 2.0, 0.0, 0.0, 0.0),  // squared
        DualQuery::new(4, 0.7, 1.0, 0.0, 0.0, 0.0),  // exp
        DualQuery::new(5, 3.0, 1.0, 0.0, 0.0, 0.0),  // ln
        DualQuery::new(6, 0.5, 1.0, 0.0, 0.0, 0.0),  // sin
        DualQuery::new(7, 0.5, 1.0, 0.0, 0.0, 0.0),  // cos
        DualQuery::new(8, 0.3, 1.0, 0.0, 0.0, 0.0),  // tan
        DualQuery::new(9, 2.5, 1.0, 0.0, 0.0, 0.0),  // abs (positive)
        DualQuery::new(10, 2.0, 1.0, 0.0, 0.0, 2.5), // powf
        DualQuery::new(11, 1.5, 2.0, 0.0, 0.0, 3.0), // mul_scalar
        DualQuery::new(12, 1.0, 2.0, 3.0, 4.0, 0.0), // add
        DualQuery::new(13, 1.0, 2.0, 3.0, 4.0, 0.0), // sub
        DualQuery::new(14, 1.5, 2.0, 2.5, 3.0, 0.0), // mul
        DualQuery::new(15, 1.5, 2.0, 2.5, 3.0, 0.0), // div
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1, "op[{i}] should be valid");
        assert_parity(res, q, &format!("op[{i}]"));
    }
}

#[test]
fn abs_sign_of_zero_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDual::new(&ctx);
    // At a_re == 0 the abs derivative is sign(0) * a_du == 0, not a_du.
    let at_zero = DualQuery::new(9, 0.0, 5.0, 0.0, 0.0, 0.0);
    let positive = DualQuery::new(9, 3.0, 5.0, 0.0, 0.0, 0.0);
    let negative = DualQuery::new(9, -3.0, 5.0, 0.0, 0.0, 0.0);
    let out = gpu.evaluate(&ctx, &[at_zero, positive, negative]);
    assert_eq!(out.len(), 3);
    assert_eq!(out[0].valid, 1);
    assert!(close(out[0].re, 0.0));
    assert!(close(out[0].du, 0.0), "sign-of-zero derivative should be 0");
    assert_parity(&out[0], &at_zero, "abs_zero");
    // Positive side: derivative is +a_du.
    assert!(close(out[1].re, 3.0));
    assert!(close(out[1].du, 5.0));
    assert_parity(&out[1], &positive, "abs_positive");
    // Negative side: derivative is -a_du.
    assert!(close(out[2].re, 3.0));
    assert!(close(out[2].du, -5.0));
    assert_parity(&out[2], &negative, "abs_negative");
}

#[test]
fn out_of_range_op_id_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDual::new(&ctx);
    let sixteen = DualQuery::new(16, 1.0, 2.0, 3.0, 4.0, 5.0);
    let big = DualQuery::new(99, 1.0, 2.0, 3.0, 4.0, 5.0);
    let out = gpu.evaluate(&ctx, &[sixteen, big]);
    assert_eq!(out.len(), 2);
    for (res, q) in out.iter().zip([sixteen, big].iter()) {
        assert_eq!(res.valid, 0, "out-of-range op_id should be invalid");
        assert_eq!(res.re, 0.0);
        assert_eq!(res.du, 0.0);
        assert_parity(res, q, "out_of_range");
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDual::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn batch_mixes_operations_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDual::new(&ctx);
    // Several distinct operations plus one out-of-range op_id, to confirm the
    // std430 stride is read back correctly for every slot.
    let queries = vec![
        DualQuery::new(14, 1.5, 2.0, 2.5, 3.0, 0.0), // mul
        DualQuery::new(2, 9.0, 1.0, 0.0, 0.0, 0.0),  // sqrt
        DualQuery::new(42, 1.0, 1.0, 1.0, 1.0, 1.0), // invalid
        DualQuery::new(15, 1.0, 0.0, 4.0, 1.0, 0.0), // div
        DualQuery::new(5, 2.0, 1.0, 0.0, 0.0, 0.0),  // ln
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[1].valid, 1);
    assert_eq!(out[2].valid, 0);
    assert_eq!(out[3].valid, 1);
    assert_eq!(out[4].valid, 1);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDual::new(&ctx);
    let mut lcg = Lcg::new(0x1234_ABCD);
    let powf_exponents = [0.5f32, 1.0, 2.0, -1.0, 2.5];
    let mut queries: Vec<DualQuery> = Vec::with_capacity(512);
    // Cycle through all sixteen operations, keeping each in its
    // well-conditioned domain so CPU and GPU agree.
    while queries.len() < 512 {
        let op = (queries.len() % 16) as u32;
        let a_du = lcg.next_range(-3.0, 3.0);
        let b_re = lcg.next_range(-3.0, 3.0);
        let b_du = lcg.next_range(-3.0, 3.0);
        let q = match op {
            // recip / div need denominators away from zero.
            1 => {
                let a_re = signed_away_from_zero(&mut lcg, 0.3, 3.0);
                DualQuery::new(1, a_re, a_du, 0.0, 0.0, 0.0)
            }
            15 => {
                let a_re = lcg.next_range(-3.0, 3.0);
                let denom = signed_away_from_zero(&mut lcg, 0.3, 3.0);
                DualQuery::new(15, a_re, a_du, denom, b_du, 0.0)
            }
            // sqrt / ln need a positive base.
            2 => {
                let a_re = lcg.next_range(0.1, 20.0);
                DualQuery::new(2, a_re, a_du, 0.0, 0.0, 0.0)
            }
            5 => {
                let a_re = lcg.next_range(0.1, 20.0);
                DualQuery::new(5, a_re, a_du, 0.0, 0.0, 0.0)
            }
            // powf needs a positive base to avoid fractional-power NaN.
            10 => {
                let a_re = lcg.next_range(0.1, 20.0);
                let p = powf_exponents[(lcg.next_u32() % 5) as usize];
                DualQuery::new(10, a_re, a_du, 0.0, 0.0, p)
            }
            11 => {
                let a_re = lcg.next_range(-3.0, 3.0);
                let p = lcg.next_range(-3.0, 3.0);
                DualQuery::new(11, a_re, a_du, 0.0, 0.0, p)
            }
            // abs: keep away from the exactly-zero knee for the sweep; the
            // sign-of-zero case has its own fixture.
            9 => {
                let a_re = signed_away_from_zero(&mut lcg, 0.1, 3.0);
                DualQuery::new(9, a_re, a_du, 0.0, 0.0, 0.0)
            }
            // tan: stay away from the pi/2 pole.
            8 => {
                let a_re = lcg.next_range(-1.2, 1.2);
                DualQuery::new(8, a_re, a_du, 0.0, 0.0, 0.0)
            }
            // The remaining ops accept any finite operands.
            _ => {
                let a_re = lcg.next_range(-3.0, 3.0);
                DualQuery::new(op, a_re, a_du, b_re, b_du, 0.0)
            }
        };
        queries.push(q);
    }
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1, "sweep[{i}] op {} should be valid", q.op_id);
        assert_parity(res, q, &format!("sweep[{i}]"));
    }
}

/// A value in `[-hi, -lo] ∪ [lo, hi]`, so it stays a comfortable margin away
/// from zero on either side.
fn signed_away_from_zero(lcg: &mut Lcg, lo: f32, hi: f32) -> f32 {
    let magnitude = lcg.next_range(lo, hi);
    if lcg.next_u32() & 1 == 0 {
        magnitude
    } else {
        -magnitude
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
