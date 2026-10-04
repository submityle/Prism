//! Real-device parity for the §24.1 cubic-spline shader mirror.
//!
//! The kernel evaluates each of the six spline functions on a real `GPU` from
//! the single-sourced [`WGSL_SPLINE`](prism_math::shader_mirror::WGSL_SPLINE)
//! fragment (`prism_spline`) and diffs the result against the CPU reference
//! family [`prism_math::curve::spline`] evaluated with [`Vec3`].
//!
//! Every evaluator is pure multiply/add polynomial arithmetic with no
//! transcendental calls, so the contract is a tight `1e-5` absolute+relative
//! tolerance that rejects operand-order/layout drift while tolerating last-ULP
//! fast-math rounding. The suite skips gracefully when no adapter is available.

use prism_math::Vec3;
use prism_math::curve::spline;
use prism_math_gpu::{GpuContext, GpuSpline, Spline, SplineSample};

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

/// Absolute+relative closeness for the FMA-tolerant polynomial contract.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    diff <= 1e-5 + 1e-5 * a.abs().max(b.abs())
}

/// A small deterministic linear-congruential sequence.
fn lcg(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *seed
}

/// A pseudo-random component in `[-4, 4)`.
fn comp(seed: &mut u32) -> f32 {
    (lcg(seed) % 800_000) as f32 / 100_000.0 - 4.0
}

/// A Vec3 from a `[f32; 3]`.
fn v(a: [f32; 3]) -> Vec3 {
    Vec3::new(a[0], a[1], a[2])
}

/// Evaluate the CPU reference for `op` at parameter `t`.
fn cpu(op: Spline, a: Vec3, b: Vec3, c: Vec3, d: Vec3, t: f32) -> Vec3 {
    match op {
        Spline::Hermite => spline::hermite(a, b, c, d, t),
        Spline::HermiteTangent => spline::hermite_tangent(a, b, c, d, t),
        Spline::CatmullRom => spline::catmull_rom(a, b, c, d, t),
        Spline::CatmullRomTangent => spline::catmull_rom_tangent(a, b, c, d, t),
        Spline::BezierCubic => spline::bezier_cubic(a, b, c, d, t),
        Spline::BezierCubicTangent => spline::bezier_cubic_tangent(a, b, c, d, t),
    }
}

/// Builds a large batch of samples: random control tuples swept across a dense
/// grid of `t`, plus the hand-picked endpoints `t = 0`, `0.5`, `1`.
fn build_samples() -> Vec<([f32; 3], [f32; 3], [f32; 3], [f32; 3], f32)> {
    let mut out = Vec::new();
    let mut seed = 0x51u32.wrapping_mul(0x9e37_79b9);
    for _ in 0..340u32 {
        let a = [comp(&mut seed), comp(&mut seed), comp(&mut seed)];
        let b = [comp(&mut seed), comp(&mut seed), comp(&mut seed)];
        let c = [comp(&mut seed), comp(&mut seed), comp(&mut seed)];
        let d = [comp(&mut seed), comp(&mut seed), comp(&mut seed)];
        for k in 0..=12u32 {
            out.push((a, b, c, d, k as f32 / 12.0));
        }
        out.push((a, b, c, d, 0.5));
    }
    out
}

/// Runs one op over the whole batch on the GPU and diffs against the CPU
/// reference componentwise.
fn check_op(ctx: &GpuContext, gpu: &GpuSpline, op: Spline) {
    let raw = build_samples();
    let samples: Vec<SplineSample> = raw
        .iter()
        .map(|&(a, b, c, d, t)| SplineSample::new(a, b, c, d, t))
        .collect();
    let got = gpu.eval(ctx, op, &samples);
    assert_eq!(got.len(), raw.len());
    for (i, (&(a, b, c, d, t), out)) in raw.iter().zip(got.iter()).enumerate() {
        let want = cpu(op, v(a), v(b), v(c), v(d), t);
        assert!(
            close(out[0], want.x) && close(out[1], want.y) && close(out[2], want.z),
            "op {op:?} sample {i} t={t}: gpu={:?} cpu=({},{},{})",
            out,
            want.x,
            want.y,
            want.z,
        );
    }
}

#[test]
fn spline_parity_all_ops() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let gpu = GpuSpline::new(&ctx);
    for op in [
        Spline::Hermite,
        Spline::HermiteTangent,
        Spline::CatmullRom,
        Spline::CatmullRomTangent,
        Spline::BezierCubic,
        Spline::BezierCubicTangent,
    ] {
        check_op(&ctx, &gpu, op);
    }
}

/// Bézier interpolates its endpoints: `t = 0 -> p0`, `t = 1 -> p3`; Catmull-Rom
/// passes through `p1` at `t = 0` and `p2` at `t = 1`. Verify the GPU honors
/// these on real hardware.
#[test]
fn spline_parity_endpoints() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let gpu = GpuSpline::new(&ctx);
    let p0 = [1.0, 2.0, -3.0];
    let p1 = [-2.0, 0.5, 4.0];
    let p2 = [3.0, -1.0, 0.0];
    let p3 = [0.0, 5.0, 2.0];

    let bez = gpu.eval(
        &ctx,
        Spline::BezierCubic,
        &[
            SplineSample::new(p0, p1, p2, p3, 0.0),
            SplineSample::new(p0, p1, p2, p3, 1.0),
        ],
    );
    assert!(close(bez[0][0], p0[0]) && close(bez[0][1], p0[1]) && close(bez[0][2], p0[2]));
    assert!(close(bez[1][0], p3[0]) && close(bez[1][1], p3[1]) && close(bez[1][2], p3[2]));

    let cr = gpu.eval(
        &ctx,
        Spline::CatmullRom,
        &[
            SplineSample::new(p0, p1, p2, p3, 0.0),
            SplineSample::new(p0, p1, p2, p3, 1.0),
        ],
    );
    assert!(close(cr[0][0], p1[0]) && close(cr[0][1], p1[1]) && close(cr[0][2], p1[2]));
    assert!(close(cr[1][0], p2[0]) && close(cr[1][1], p2[1]) && close(cr[1][2], p2[2]));
}

/// An empty batch yields an empty result without dispatching.
#[test]
fn spline_parity_empty_batch() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let gpu = GpuSpline::new(&ctx);
    let got = gpu.eval(&ctx, Spline::Hermite, &[]);
    assert!(got.is_empty());
}
