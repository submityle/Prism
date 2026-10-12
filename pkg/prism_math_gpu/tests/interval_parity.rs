//! Real-device parity for the §24.3 GPU interval-arithmetic shader mirror.
//!
//! Each test builds a batch of interval operand pairs, evaluates the enclosure
//! on the CPU with [`prism_math::interval::Interval`], runs the same op on a real
//! `GPU` from the single-sourced
//! [`WGSL_INTERVAL`](prism_math::shader_mirror::WGSL_INTERVAL) fragment, and
//! asserts the `[lo, hi]` bounds agree within a small tolerance.
//!
//! Each op is a few single-precision operations plus a deterministic one-ULP
//! `bitcast` step, so in the ideal the device is bit-exact to the CPU; Metal
//! compiles WGSL under fast-math, so the raw bounds can differ by a rounding
//! step and the outward step then lands one ULP away. The tolerance covers that
//! while still rejecting a genuine operand-order or rounding-direction drift.
//! The whole-line `[-inf, +inf]` enclosure (divisor straddling zero) is compared
//! exactly. It skips gracefully when no adapter is available so it still passes
//! on a device-less CI image while running the full dispatch on a real `GPU`.

use prism_math::interval::Interval;
use prism_math_gpu::{
    GpuContext, GpuInterval, IntervalBinaryOp, IntervalBounds, IntervalPair, IntervalUnaryOp,
};

/// Acquires a device, or prints a skip note and returns `None` on hosts without
/// a usable adapter.
#[expect(
    clippy::print_stderr,
    reason = "test-only skip note when no GPU adapter is present"
)]
fn with_gpu() -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping: no usable GPU adapter on this host");
            None
        }
    }
}

/// Absolute+relative closeness for finite bounds, matching the fast-math
/// tolerance contract.
fn close(a: f32, b: f32) -> bool {
    if a.is_infinite() || b.is_infinite() {
        return a == b;
    }
    let diff = (a - b).abs();
    // One ULP of an f32 is ~1.2e-7 relative; allow a few ULP of slack.
    diff <= 1.0e-6 + 1.0e-6 * a.abs().max(b.abs())
}

/// Compares a decoded GPU enclosure against the CPU `Interval`.
fn assert_bounds(gpu: IntervalBounds, cpu: Interval, tag: &str) {
    assert!(
        close(gpu.lo, cpu.lo()),
        "{tag}: lo mismatch gpu={} cpu={}",
        gpu.lo,
        cpu.lo()
    );
    assert!(
        close(gpu.hi, cpu.hi()),
        "{tag}: hi mismatch gpu={} cpu={}",
        gpu.hi,
        cpu.hi()
    );
}

#[test]
fn interval_binary_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuInterval::new(&ctx);

    // (a_lo, a_hi, b_lo, b_hi) pairs spanning positive, negative, zero-straddling,
    // and a divisor straddling zero (the whole-line case).
    let cases: &[([f32; 2], [f32; 2])] = &[
        ([1.0, 2.0], [3.0, 4.0]),
        ([-2.0, -1.0], [0.5, 1.5]),
        ([-1.0, 1.0], [2.0, 3.0]),
        ([0.0, 5.0], [-4.0, -1.0]),
        ([-3.0, 2.0], [-1.0, 4.0]),
        // Divisor straddles zero -> div must yield the whole line.
        ([1.0, 1.0], [-1.0, 1.0]),
    ];

    let ops = [
        IntervalBinaryOp::Add,
        IntervalBinaryOp::Sub,
        IntervalBinaryOp::Mul,
        IntervalBinaryOp::Div,
    ];
    for op in ops {
        let pairs: Vec<IntervalPair> = cases
            .iter()
            .map(|(a, b)| IntervalPair::new(*a, *b))
            .collect();
        let got = kernel.binary(&ctx, op, &pairs);
        assert_eq!(got.len(), cases.len());
        for (i, (a, b)) in cases.iter().enumerate() {
            let ia = Interval::new(a[0], a[1]);
            let ib = Interval::new(b[0], b[1]);
            let cpu = match op {
                IntervalBinaryOp::Add => ia + ib,
                IntervalBinaryOp::Sub => ia - ib,
                IntervalBinaryOp::Mul => ia * ib,
                IntervalBinaryOp::Div => ia / ib,
            };
            assert_bounds(got[i], cpu, &alloc_tag(op as usize, i));
        }
    }
}

#[test]
fn interval_unary_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuInterval::new(&ctx);

    // abs spans all-positive, all-negative, and zero-straddling; sqrt includes a
    // negative lower bound that must clamp to zero.
    let cases: &[[f32; 2]] = &[
        [1.0, 2.0],
        [-3.0, -1.0],
        [-2.0, 5.0],
        [0.0, 0.0],
        [0.25, 4.0],
        [-1.0, 9.0],
    ];

    let abs_got = kernel.unary(&ctx, IntervalUnaryOp::Abs, cases);
    let sqrt_got = kernel.unary(&ctx, IntervalUnaryOp::Sqrt, cases);
    assert_eq!(abs_got.len(), cases.len());
    assert_eq!(sqrt_got.len(), cases.len());

    for (i, c) in cases.iter().enumerate() {
        let cpu = Interval::new(c[0], c[1]);
        assert_bounds(abs_got[i], cpu.abs(), &alloc_tag(0, i));
        assert_bounds(sqrt_got[i], cpu.sqrt(), &alloc_tag(1, i));
    }
}

#[test]
fn empty_batch_returns_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuInterval::new(&ctx);
    assert!(kernel.binary(&ctx, IntervalBinaryOp::Add, &[]).is_empty());
    assert!(kernel.unary(&ctx, IntervalUnaryOp::Abs, &[]).is_empty());
}

/// Small heap tag so panic messages identify the failing op/element.
fn alloc_tag(op: usize, i: usize) -> String {
    format!("interval op#{op}[{i}]")
}
