//! Real-device parity for the §24.5 GPU forward-mode dual-number shader mirror.
//!
//! Each test seeds a batch of scalars as the independent variable, evaluates an
//! op and its analytic derivative on the CPU with
//! [`prism_math::dual::Dual`]'s matching method, runs the same op on a real
//! `GPU` from the single-sourced
//! [`WGSL_DUAL`](prism_math::shader_mirror::WGSL_DUAL) fragment, and asserts the
//! value and derivative agree within tolerance.
//!
//! The polynomial ops (`recip`/`squared`) and the `sqrt` builtin match the CPU
//! `libm` reference tightly; the transcendental ops (`exp`/`ln`/`sin`/`cos`/
//! `tan`/`powf`) evaluate device builtins and so agree within a looser
//! relative epsilon. Metal compiles WGSL under fast-math, so even the tight ops
//! carry a small absolute+relative slack. It skips gracefully when no adapter
//! is available so it still passes on a device-less CI image while running the
//! full dispatch on a real `GPU`.

use prism_math::dual::Dual;
use prism_math_gpu::{DualOp, DualSample, GpuContext, GpuDual, GpuDualResult};

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

/// Tight tolerance for pure polynomial ops and the `sqrt` builtin.
fn close_tight(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    diff <= 1.0e-5 + 1.0e-5 * a.abs().max(b.abs())
}

/// Looser tolerance for the transcendental builtins vs the CPU `libm`.
fn close_loose(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    diff <= 1.0e-4 + 1.0e-4 * a.abs().max(b.abs())
}

/// Compares one decoded GPU dual against a CPU dual, element-tagged.
fn assert_dual(gpu: GpuDualResult, cpu: Dual, tight: bool, tag: &str) {
    let ok = if tight { close_tight } else { close_loose };
    assert!(
        ok(gpu.re, cpu.re),
        "{tag}: value mismatch gpu={} cpu={}",
        gpu.re,
        cpu.re
    );
    assert!(
        ok(gpu.du, cpu.du),
        "{tag}: derivative mismatch gpu={} cpu={}",
        gpu.du,
        cpu.du
    );
}

#[test]
fn dual_ops_match_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuDual::new(&ctx);

    // Pure-arithmetic ops: tight comparison (sqrt is a hardware builtin).
    let tight_cases: &[(DualOp, &[f32])] = &[
        (DualOp::Recip, &[1.0, 2.0, -1.0, 0.5, 4.0, -2.5]),
        (DualOp::Sqrt, &[1.0, 4.0, 2.25, 9.0, 0.25, 16.0]),
        (DualOp::Squared, &[0.0, 1.0, -2.0, 3.5, -0.5, 10.0]),
        (DualOp::Abs, &[1.0, -2.0, 3.5, -0.5, 8.0, -0.25]),
    ];
    for (op, xs) in tight_cases {
        let samples: Vec<DualSample> = xs.iter().map(|&x| DualSample::new(x)).collect();
        let got = kernel.apply(&ctx, *op, &samples);
        assert_eq!(got.len(), xs.len());
        for (i, &x) in xs.iter().enumerate() {
            let cpu = match op {
                DualOp::Recip => Dual::variable(x).recip(),
                DualOp::Sqrt => Dual::variable(x).sqrt(),
                DualOp::Squared => Dual::variable(x).squared(),
                DualOp::Abs => Dual::variable(x).abs(),
                _ => unreachable!(),
            };
            assert_dual(got[i], cpu, true, &alloc_tag(*op as usize, i));
        }
    }

    // Transcendental ops: looser comparison against the CPU `libm`.
    let loose_cases: &[(DualOp, &[f32])] = &[
        (DualOp::Exp, &[0.0, 1.0, -1.0, 2.5, -0.5, 3.0]),
        (DualOp::Ln, &[1.0, 2.0, 0.5, 4.0, 10.0, 0.25]),
        (DualOp::Sin, &[0.0, 0.5, -1.0, 2.0, -2.5, 3.0]),
        (DualOp::Cos, &[0.0, 0.5, -1.0, 2.0, -2.5, 3.0]),
        (DualOp::Tan, &[0.0, 0.5, -0.5, 1.0, -1.2, 0.75]),
    ];
    for (op, xs) in loose_cases {
        let samples: Vec<DualSample> = xs.iter().map(|&x| DualSample::new(x)).collect();
        let got = kernel.apply(&ctx, *op, &samples);
        assert_eq!(got.len(), xs.len());
        for (i, &x) in xs.iter().enumerate() {
            let cpu = match op {
                DualOp::Exp => Dual::variable(x).exp(),
                DualOp::Ln => Dual::variable(x).ln(),
                DualOp::Sin => Dual::variable(x).sin(),
                DualOp::Cos => Dual::variable(x).cos(),
                DualOp::Tan => Dual::variable(x).tan(),
                _ => unreachable!(),
            };
            assert_dual(got[i], cpu, false, &alloc_tag(*op as usize, i));
        }
    }
}

#[test]
fn dual_powf_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuDual::new(&ctx);

    // (x, p) pairs with x > 0 so the constant-exponent power is real.
    let cases: &[(f32, f32)] = &[
        (2.0, 3.0),
        (4.0, 0.5),
        (1.5, 2.0),
        (9.0, -1.0),
        (0.5, 4.0),
        (3.0, 1.5),
    ];
    let samples: Vec<DualSample> = cases
        .iter()
        .map(|&(x, p)| DualSample::with_exponent(x, p))
        .collect();
    let got = kernel.apply(&ctx, DualOp::Powf, &samples);
    assert_eq!(got.len(), cases.len());

    for (i, &(x, p)) in cases.iter().enumerate() {
        let cpu = Dual::variable(x).powf(p);
        assert_dual(got[i], cpu, false, &alloc_tag(DualOp::Powf as usize, i));
    }
}

#[test]
fn empty_batch_returns_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuDual::new(&ctx);
    assert!(kernel.apply(&ctx, DualOp::Exp, &[]).is_empty());
}

/// Small heap tag so panic messages identify the failing op/element.
fn alloc_tag(op: usize, i: usize) -> String {
    format!("dual op#{op}[{i}]")
}
