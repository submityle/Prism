//! Real-device parity for the groom-global analysis reduction twin:
//! [`GpuHairAnalysisReduce`] must reproduce the `CPU` golden
//! [`reduce_lane`](prism_render_architecture::hair::analysis_readback::reduce_lane)
//! for a chosen [`HairMetricLane`] folded with a chosen [`HairReductionOp`],
//! covering the empty groom (identity, no dispatch), the `Max` selection of the
//! largest value, the `Sum` accumulation, per-lane selection (`X`/`Y`/`Z`/`W`),
//! the all-negative lane whose `Max` reduces to the identity `0`, the
//! single-element degenerate fold, and a large batch that crosses the `256`-wide
//! workgroup so the grid-stride load and the shared-memory tree fold are both
//! exercised.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel uses only comparisons,
//! addition and workgroup shared memory in the portable core-`WGSL` subset, so
//! it needs no optional device feature.
//!
//! # Parity criterion
//!
//! `Max` selects an existing element value with no arithmetic, so it is
//! bit-exact regardless of fold order and is asserted with `assert_eq!`. `Sum`
//! is floating-point addition, which is commutative but not associative, so the
//! tree's pairwise order differs from the golden's left-to-right walk by a few
//! low-mantissa `ULP`; it is asserted to within `abs_diff < 1e-4` or
//! `rel_diff < 1e-3` — tight enough to fail a genuinely wrong port (a wrong
//! lane, a missing element, a swapped operator), loose enough to admit the
//! reassociation. Every element is an explicit `[f32; 4]` literal (never
//! `sin`/`cos`), and the empty/all-negative reads are asserted exactly so a
//! no-op kernel could not pass.
//!
//! This is the crate's first **many-inputs-to-one-output** reduction twin — a
//! single-workgroup shared-memory tree fold — as opposed to the
//! one-thread-per-output map/gather kernels that precede it. It validates the
//! reduce-side of the `GPU` → host → `CPU` analysis bridge the golden
//! [`reduce_lane`] contract names.
//!
//! Provenance: standard single-workgroup shared-memory tree reduction plus
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.

use prism_hair_gpu::analysis_reduce::{reference_reduce, GpuHairAnalysisReduce};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::analysis_readback::{
    reduce_lane, HairMetricLane, HairReductionOp,
};

/// Asserts a `Sum` value matches the golden within the documented tolerance.
fn assert_close(got: f32, expected: f32, label: &str) {
    let abs_diff = (got - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-6);
    assert!(
        abs_diff < 1e-4 || rel_diff < 1e-3,
        "{label}: gpu {got} vs cpu {expected} (abs {abs_diff}, rel {rel_diff})",
    );
}

/// An empty groom folds to the operator identity (`0`) for both operators,
/// through the host early-return path that never touches the device.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_empty_yields_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping analysis reduce parity: no wgpu adapter on this host");
        return;
    };
    let reducer = GpuHairAnalysisReduce::new(&ctx);
    let elements: [[f32; 4]; 0] = [];
    for op in [HairReductionOp::Max, HairReductionOp::Sum] {
        let gpu = reducer.reduce(&ctx, &elements, HairMetricLane::X, op);
        let cpu = reference_reduce(&elements, HairMetricLane::X, op);
        assert_eq!(gpu, HairReductionOp::identity(), "empty groom is identity");
        assert_eq!(gpu, cpu, "empty groom matches golden");
    }
}

/// `Max` over the `X` lane selects the single largest value, bit-exactly.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_max_selects_largest() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping analysis reduce parity: no wgpu adapter on this host");
        return;
    };
    let reducer = GpuHairAnalysisReduce::new(&ctx);
    let elements = [
        [0.25, 9.0, -3.0, 1.0],
        [4.5, 2.0, 8.0, 0.0],
        [1.75, 0.5, 6.0, 2.0],
        [4.25, 7.0, -1.0, 5.0],
    ];
    let gpu = reducer.reduce(&ctx, &elements, HairMetricLane::X, HairReductionOp::Max);
    let cpu = reduce_lane(&elements, HairMetricLane::X, HairReductionOp::Max);
    assert_eq!(cpu, 4.5, "golden max over X lane");
    assert_eq!(gpu, cpu, "gpu max is bit-exact");
}

/// `Sum` over the `X` lane accumulates every element.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_sum_accumulates() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping analysis reduce parity: no wgpu adapter on this host");
        return;
    };
    let reducer = GpuHairAnalysisReduce::new(&ctx);
    let elements = [
        [0.5, 1.0, 2.0, 3.0],
        [1.5, 4.0, 5.0, 6.0],
        [2.25, 7.0, 8.0, 9.0],
        [3.75, 1.0, 1.0, 1.0],
    ];
    let gpu = reducer.reduce(&ctx, &elements, HairMetricLane::X, HairReductionOp::Sum);
    let cpu = reduce_lane(&elements, HairMetricLane::X, HairReductionOp::Sum);
    assert_close(gpu, cpu, "gpu sum over X lane");
}

/// Each lane (`X`/`Y`/`Z`/`W`) folds an independent component, so selecting a
/// different lane yields a different reduction.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_lane_selection() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping analysis reduce parity: no wgpu adapter on this host");
        return;
    };
    let reducer = GpuHairAnalysisReduce::new(&ctx);
    let elements = [
        [1.0, 10.0, 100.0, 1000.0],
        [2.0, 20.0, 200.0, 2000.0],
        [3.0, 30.0, 300.0, 3000.0],
    ];
    for lane in [
        HairMetricLane::X,
        HairMetricLane::Y,
        HairMetricLane::Z,
        HairMetricLane::W,
    ] {
        let gpu = reducer.reduce(&ctx, &elements, lane, HairReductionOp::Max);
        let cpu = reduce_lane(&elements, lane, HairReductionOp::Max);
        assert_eq!(gpu, cpu, "gpu max on selected lane matches golden");
    }
}

/// An all-negative lane reduced with `Max` returns the identity `0`, exactly as
/// the golden's `if x > acc` update starting from `0` does.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_max_all_negative_reads_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping analysis reduce parity: no wgpu adapter on this host");
        return;
    };
    let reducer = GpuHairAnalysisReduce::new(&ctx);
    let elements = [
        [-1.0, 0.0, 0.0, 0.0],
        [-9.5, 0.0, 0.0, 0.0],
        [-0.25, 0.0, 0.0, 0.0],
    ];
    let gpu = reducer.reduce(&ctx, &elements, HairMetricLane::X, HairReductionOp::Max);
    let cpu = reduce_lane(&elements, HairMetricLane::X, HairReductionOp::Max);
    assert_eq!(cpu, 0.0, "golden max of all-negative is the identity");
    assert_eq!(gpu, cpu, "gpu max of all-negative is the identity");
}

/// A single-element groom folds to that element's lane value under both
/// operators.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_single_element() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping analysis reduce parity: no wgpu adapter on this host");
        return;
    };
    let reducer = GpuHairAnalysisReduce::new(&ctx);
    let elements = [[2.5, 3.5, 4.5, 5.5]];
    let max = reducer.reduce(&ctx, &elements, HairMetricLane::Z, HairReductionOp::Max);
    let sum = reducer.reduce(&ctx, &elements, HairMetricLane::Z, HairReductionOp::Sum);
    assert_eq!(
        max,
        reduce_lane(&elements, HairMetricLane::Z, HairReductionOp::Max),
        "single-element max is the element",
    );
    assert_close(
        sum,
        reduce_lane(&elements, HairMetricLane::Z, HairReductionOp::Sum),
        "single-element sum is the element",
    );
}

/// A batch larger than the `256`-wide workgroup exercises the grid-stride load
/// and the full shared-memory tree fold for both operators.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_large_batch_crosses_workgroup() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping analysis reduce parity: no wgpu adapter on this host");
        return;
    };
    let reducer = GpuHairAnalysisReduce::new(&ctx);
    let mut elements = Vec::with_capacity(1000);
    for i in 0..1000u32 {
        let f = i as f32;
        elements.push([f * 0.5, 1000.0 - f, f * 0.25, f * 0.125]);
    }
    let sum_gpu = reducer.reduce(&ctx, &elements, HairMetricLane::X, HairReductionOp::Sum);
    let sum_cpu = reduce_lane(&elements, HairMetricLane::X, HairReductionOp::Sum);
    assert_close(sum_gpu, sum_cpu, "large-batch sum crosses workgroup");

    let max_gpu = reducer.reduce(&ctx, &elements, HairMetricLane::Y, HairReductionOp::Max);
    let max_cpu = reduce_lane(&elements, HairMetricLane::Y, HairReductionOp::Max);
    assert_eq!(max_gpu, max_cpu, "large-batch max crosses workgroup");
}

/// A mixed batch folds the `W` (padding) lane, confirming the lane selector
/// reaches the last component under both operators.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_w_lane_mixed_ops() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping analysis reduce parity: no wgpu adapter on this host");
        return;
    };
    let reducer = GpuHairAnalysisReduce::new(&ctx);
    let elements = [
        [0.0, 0.0, 0.0, 3.0],
        [0.0, 0.0, 0.0, 12.5],
        [0.0, 0.0, 0.0, 1.25],
        [0.0, 0.0, 0.0, 7.0],
    ];
    let max = reducer.reduce(&ctx, &elements, HairMetricLane::W, HairReductionOp::Max);
    let sum = reducer.reduce(&ctx, &elements, HairMetricLane::W, HairReductionOp::Sum);
    assert_eq!(
        max,
        reduce_lane(&elements, HairMetricLane::W, HairReductionOp::Max),
        "gpu max on W lane matches golden",
    );
    assert_close(
        sum,
        reduce_lane(&elements, HairMetricLane::W, HairReductionOp::Sum),
        "gpu sum on W lane matches golden",
    );
}
