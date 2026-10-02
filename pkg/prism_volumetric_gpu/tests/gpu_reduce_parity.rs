//! Real-device parity for the tiled-reduction twin:
//! [`GpuReduce`](prism_volumetric_gpu::gpu_reduce::GpuReduce) must reproduce the
//! `CPU` golden
//! [`gpu_reduce`](prism_render_architecture::particle::gpu_reduce) across every
//! fold op (`Min` / `Max` / `Sum`) on both `u32` and `f32` channels, an empty
//! input (host short-circuit), a single element, block-boundary aligned and
//! ragged lengths, degenerate and oversized `workgroup_size` clamps, a
//! cross-`workgroup_size` invariance sweep, and large pseudo-random arrays.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every `u32` reduction output is a `u32`: a `min`, a `max`, or a wrapping
//! `sum`, all exactly associative, so the `GPU` result equals the serial
//! [`reduce_all`](prism_render_architecture::particle::gpu_reduce::reduce_all)
//! bit for bit and is asserted with **exact `==`**. The `f32` reductions are
//! compared with the crate's relative-scaled epsilon
//! (`abs_diff <= 1e-4 || rel_diff <= 1e-3`), never for exact equality, because
//! floating-point addition is not associative and the device folds each block
//! in `f32` where the golden widens into `f64`. The random `f32` fixtures stay
//! strictly positive and bounded so their sums are comfortably far from zero
//! and from any overflow, keeping the comparison well inside tolerance.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_reduce`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use prism_render_architecture::particle::gpu_reduce::{reduce_all, ReduceConfig, ReduceOp};
use prism_volumetric_gpu::gpu_reduce::{GpuReduce, GpuReduceInput, GpuReduceOp, GpuReduceResult};
use prism_volumetric_gpu::GpuContext;

/// Relative floor so the `f32` comparison never divides by zero near the
/// origin.
const REL_FLOOR: f32 = 1e-6;
/// Absolute tolerance for the continuous `f32` comparison.
const ABS_EPS: f32 = 1e-4;
/// Relative tolerance for the continuous `f32` comparison.
const REL_EPS: f32 = 1e-3;

/// The crate's continuous-quantity comparison: either the absolute or the
/// relative difference must fall within tolerance.
fn approx_eq(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// A tiny integer linear-congruential generator; only integer arithmetic, so no
/// transcendental appears.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 32) as u32
    }
}

/// Maps the twin's op enum onto the golden op enum.
fn golden_op(op: GpuReduceOp) -> ReduceOp {
    match op {
        GpuReduceOp::Min => ReduceOp::Min,
        GpuReduceOp::Max => ReduceOp::Max,
        GpuReduceOp::Sum => ReduceOp::Sum,
    }
}

/// All three fold ops, for exhaustive sweeps.
const ALL_OPS: [GpuReduceOp; 3] = [GpuReduceOp::Min, GpuReduceOp::Max, GpuReduceOp::Sum];

/// Runs a `u32` reduction on the device and asserts exact parity: the final
/// scalar equals both the golden tiled reduce and the serial `reduce_all`, and
/// every per-block partial equals the golden first pass over that block.
fn check_u32(ctx: &GpuContext, gpu: &GpuReduce, op: GpuReduceOp, values: &[u32], ws: u32) {
    let query = prism_volumetric_gpu::gpu_reduce::GpuReduceQuery {
        op,
        workgroup_size: ws,
        input: GpuReduceInput::U32(values.to_vec()),
    };
    let result = gpu.evaluate(ctx, &query);
    let (partials, reduced) = match result {
        GpuReduceResult::U32 { partials, reduced } => (partials, reduced),
        GpuReduceResult::F32 { .. } => panic!("u32 query must return a u32 result"),
    };

    let gop = golden_op(op);
    let cfg = ReduceConfig::new(values.len() as u32, ws);
    assert_eq!(
        reduced,
        cfg.reduce(gop, values),
        "u32 {op:?} ws {ws}: gpu reduced vs golden tiled reduce"
    );
    assert_eq!(
        reduced,
        reduce_all(gop, values),
        "u32 {op:?} ws {ws}: gpu reduced vs serial reduce_all"
    );

    let width = ws.clamp(1, 1024) as usize;
    let want_blocks = values.len().div_ceil(width);
    assert_eq!(
        partials.len(),
        want_blocks,
        "u32 {op:?} ws {ws}: one partial per block"
    );
    for (block, chunk) in values.chunks(width).enumerate() {
        assert_eq!(
            partials[block],
            reduce_all(gop, chunk),
            "u32 {op:?} ws {ws}: block {block} partial vs golden first pass"
        );
    }
}

/// Runs an `f32` reduction on the device and asserts tolerance parity: the
/// final scalar is close to both the golden tiled reduce and the serial
/// `reduce_all`, and every per-block partial is close to the golden first pass.
fn check_f32(ctx: &GpuContext, gpu: &GpuReduce, op: GpuReduceOp, values: &[f32], ws: u32) {
    let query = prism_volumetric_gpu::gpu_reduce::GpuReduceQuery {
        op,
        workgroup_size: ws,
        input: GpuReduceInput::F32(values.to_vec()),
    };
    let result = gpu.evaluate(ctx, &query);
    let (partials, reduced) = match result {
        GpuReduceResult::F32 { partials, reduced } => (partials, reduced),
        GpuReduceResult::U32 { .. } => panic!("f32 query must return an f32 result"),
    };

    let gop = golden_op(op);
    let cfg = ReduceConfig::new(values.len() as u32, ws);
    assert!(
        approx_eq(reduced, cfg.reduce(gop, values)),
        "f32 {op:?} ws {ws}: gpu {reduced} vs golden tiled {}",
        cfg.reduce(gop, values)
    );
    assert!(
        approx_eq(reduced, reduce_all(gop, values)),
        "f32 {op:?} ws {ws}: gpu {reduced} vs serial {}",
        reduce_all(gop, values)
    );

    let width = ws.clamp(1, 1024) as usize;
    let want_blocks = values.len().div_ceil(width);
    assert_eq!(
        partials.len(),
        want_blocks,
        "f32 {op:?} ws {ws}: one partial per block"
    );
    for (block, chunk) in values.chunks(width).enumerate() {
        assert!(
            approx_eq(partials[block], reduce_all(gop, chunk)),
            "f32 {op:?} ws {ws}: block {block} partial {} vs golden {}",
            partials[block],
            reduce_all(gop, chunk)
        );
    }
}

#[test]
fn structured_u32_all_ops() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuReduce::new(&ctx);
    let values = [7u32, 3, 9, 1, 8, 4, 6, 2, 5, 10, 0, 11];
    for op in ALL_OPS {
        for ws in [1u32, 2, 3, 4, 6, 12, 64] {
            check_u32(&ctx, &gpu, op, &values, ws);
        }
    }
}

#[test]
fn structured_f32_all_ops() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuReduce::new(&ctx);
    let values = [7.5f32, -3.25, 9.0, 1.5, 8.0, 4.0, -1.0, 2.25, 6.5, 3.0];
    for op in ALL_OPS {
        for ws in [1u32, 2, 3, 5, 10, 64] {
            check_f32(&ctx, &gpu, op, &values, ws);
        }
    }
}

#[test]
fn single_element() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuReduce::new(&ctx);
    for op in ALL_OPS {
        check_u32(&ctx, &gpu, op, &[42u32], 64);
        check_f32(&ctx, &gpu, op, &[2.5f32], 64);
    }
}

#[test]
fn block_boundary_aligned_and_ragged() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuReduce::new(&ctx);
    let mut rng = Lcg::new(0x1234_5678_9abc_def0);
    // Aligned: length is a multiple of the width; ragged: one past a multiple.
    for &len in &[256usize, 257, 300, 512, 513] {
        let values: Vec<u32> = (0..len).map(|_| rng.next_u32() % 1_000_000).collect();
        for op in ALL_OPS {
            for ws in [64u32, 128, 256] {
                check_u32(&ctx, &gpu, op, &values, ws);
            }
        }
    }
}

#[test]
fn degenerate_and_oversized_workgroup_clamp() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuReduce::new(&ctx);
    let values = [11u32, 4, 9, 2, 7, 1, 8, 3, 6, 5];
    for op in ALL_OPS {
        // ws 0 clamps up to 1; ws 100_000 clamps down to 1024.
        check_u32(&ctx, &gpu, op, &values, 0);
        check_u32(&ctx, &gpu, op, &values, 100_000);
    }
}

#[test]
fn cross_workgroup_size_invariance_u32() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuReduce::new(&ctx);
    let mut rng = Lcg::new(0x0f0f_0f0f_dead_beef);
    let values: Vec<u32> = (0..300).map(|_| rng.next_u32() % 100_000).collect();
    for op in ALL_OPS {
        for ws in [1u32, 2, 7, 32, 64, 256, 1024] {
            check_u32(&ctx, &gpu, op, &values, ws);
        }
    }
}

#[test]
fn large_random_u32() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuReduce::new(&ctx);
    let mut rng = Lcg::new(0xdead_c0de_1234_5678);
    let values: Vec<u32> = (0..4096).map(|_| rng.next_u32() % 1_000_000).collect();
    for op in ALL_OPS {
        check_u32(&ctx, &gpu, op, &values, 256);
    }
}

#[test]
fn large_random_f32_positive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuReduce::new(&ctx);
    let mut rng = Lcg::new(0xcafe_babe_0bad_f00d);
    // Strictly positive, bounded in [1, 11], so sums stay far from zero and far
    // from any f32 overflow, keeping the f32-vs-f64 divergence inside tolerance.
    let values: Vec<f32> = (0..4096)
        .map(|_| f32::from((rng.next_u32() % 1000) as u16) / 100.0 + 1.0)
        .collect();
    for op in ALL_OPS {
        for ws in [64u32, 256] {
            check_f32(&ctx, &gpu, op, &values, ws);
        }
    }
}

#[test]
fn empty_input_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuReduce::new(&ctx);
    for op in ALL_OPS {
        let qu = prism_volumetric_gpu::gpu_reduce::GpuReduceQuery {
            op,
            workgroup_size: 64,
            input: GpuReduceInput::U32(Vec::new()),
        };
        match gpu.evaluate(&ctx, &qu) {
            GpuReduceResult::U32 { partials, reduced } => {
                assert!(partials.is_empty(), "empty u32 {op:?}: no partials");
                let want = match op {
                    GpuReduceOp::Min => u32::MAX,
                    GpuReduceOp::Max | GpuReduceOp::Sum => 0,
                };
                assert_eq!(reduced, want, "empty u32 {op:?}: identity reduced");
            }
            GpuReduceResult::F32 { .. } => panic!("u32 query must return u32"),
        }

        let qf = prism_volumetric_gpu::gpu_reduce::GpuReduceQuery {
            op,
            workgroup_size: 64,
            input: GpuReduceInput::F32(Vec::new()),
        };
        match gpu.evaluate(&ctx, &qf) {
            GpuReduceResult::F32 { partials, reduced } => {
                assert!(partials.is_empty(), "empty f32 {op:?}: no partials");
                match op {
                    GpuReduceOp::Min => {
                        assert!(
                            reduced.is_infinite() && reduced > 0.0,
                            "empty f32 Min: +inf"
                        );
                    }
                    GpuReduceOp::Max => {
                        assert!(
                            reduced.is_infinite() && reduced < 0.0,
                            "empty f32 Max: -inf"
                        );
                    }
                    GpuReduceOp::Sum => {
                        assert!(approx_eq(reduced, 0.0), "empty f32 Sum: zero");
                    }
                }
            }
            GpuReduceResult::U32 { .. } => panic!("f32 query must return f32"),
        }
    }
}
