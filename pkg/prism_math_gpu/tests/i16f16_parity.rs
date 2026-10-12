//! Real-device parity for the §24.7 GPU Q16.16 fixed-point shader mirror.
//!
//! Each test builds a batch of raw `i32` operand pairs, evaluates the saturating
//! op on the CPU with [`prism_math::fixed::i16f16::I16F16`], runs the same op on
//! a real `GPU` from the shared
//! [`WGSL_U64`](prism_math::shader_mirror::WGSL_U64) emulation plus the
//! single-sourced
//! [`WGSL_I16F16`](prism_math::shader_mirror::WGSL_I16F16) fragment, and asserts
//! the raw bits match.
//!
//! Every op is exact integer arithmetic (the multiply rounds a 64-bit product),
//! so agreement is **strict equality** — the determinism contract. It skips
//! gracefully when no adapter is available.

use prism_math::fixed::i16f16::I16F16;
use prism_math_gpu::{GpuContext, GpuI16F16, I16F16Op};

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

/// Raw Q16.16 values spanning zero, ±1, ±0.5, MAX, MIN, and near-overflow.
const RAWS: [i32; 10] = [
    0,
    1 << 16,
    -(1 << 16),
    1 << 15,
    1,
    -1,
    i32::MAX,
    i32::MIN,
    2_000_000_000,
    -2_000_000_000,
];

/// Builds pairs `(a, b)` covering the value grid (bounded count kept small).
fn pairs() -> Vec<[i32; 2]> {
    let mut v = Vec::new();
    for (i, &a) in RAWS.iter().enumerate() {
        let b = RAWS[(i + 3) % RAWS.len()];
        v.push([a, b]);
    }
    // Extra multiply cases that stress the 64-bit widening / saturation.
    v.push([i32::MAX, i32::MAX]);
    v.push([i32::MIN, i32::MIN]);
    v.push([i32::MAX, i32::MIN]);
    v.push([3 << 16, 5 << 16]);
    v.push([-(3 << 16), 5 << 16]);
    v
}

#[test]
fn i16f16_binary_ops_match_cpu_bits() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuI16F16::new(&ctx);
    let ps = pairs();

    let ops = [
        (
            I16F16Op::Add,
            I16F16::saturating_add as fn(I16F16, I16F16) -> I16F16,
        ),
        (
            I16F16Op::Sub,
            I16F16::saturating_sub as fn(I16F16, I16F16) -> I16F16,
        ),
        (
            I16F16Op::Mul,
            I16F16::saturating_mul as fn(I16F16, I16F16) -> I16F16,
        ),
        (I16F16Op::Min, I16F16::min as fn(I16F16, I16F16) -> I16F16),
        (I16F16Op::Max, I16F16::max as fn(I16F16, I16F16) -> I16F16),
    ];
    for (op, cpu_fn) in ops {
        let got = kernel.apply(&ctx, op, &ps);
        assert_eq!(got.len(), ps.len());
        for (i, p) in ps.iter().enumerate() {
            let expected = cpu_fn(I16F16::from_bits(p[0]), I16F16::from_bits(p[1])).to_bits();
            assert_eq!(
                got[i], expected,
                "op#{:?}[{i}] a={} b={} gpu={} cpu={expected}",
                op, p[0], p[1], got[i]
            );
        }
    }
}

#[test]
fn i16f16_unary_ops_match_cpu_bits() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuI16F16::new(&ctx);
    // Unary ops use only `a`; reuse the value grid.
    let ps: Vec<[i32; 2]> = RAWS.iter().map(|&a| [a, 0]).collect();

    let abs = kernel.apply(&ctx, I16F16Op::Abs, &ps);
    let neg = kernel.apply(&ctx, I16F16Op::Neg, &ps);
    for (i, &a) in RAWS.iter().enumerate() {
        assert_eq!(abs[i], I16F16::from_bits(a).abs().to_bits(), "abs[{i}]");
        assert_eq!(
            neg[i],
            I16F16::from_bits(a).saturating_neg().to_bits(),
            "neg[{i}]"
        );
    }
}

#[test]
fn empty_batch_returns_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuI16F16::new(&ctx);
    assert!(kernel.apply(&ctx, I16F16Op::Add, &[]).is_empty());
}
