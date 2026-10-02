//! Real-device parity for the fixed-function `stencil` state-machine twin:
//! [`GpuStencilOp`](prism_volumetric_gpu::stencil_op::GpuStencilOp) must
//! reproduce the `CPU` golden
//! [`stencil_op`](prism_render_architecture::particle::stencil_op) query for
//! query across the compare predicate
//! ([`CompareFunc::test`](prism_render_architecture::particle::stencil_op::CompareFunc::test)),
//! the selected operation code
//! ([`StencilFace::selected_op`](prism_render_architecture::particle::stencil_op::StencilFace::selected_op)),
//! the full per-face resolve
//! ([`StencilFace::resolve`](prism_render_architecture::particle::stencil_op::StencilFace::resolve))
//! and the standalone masked write-back
//! ([`write_masked`](prism_render_architecture::particle::stencil_op::write_masked)).
//!
//! The fixtures cover every enumerant: all eight
//! [`CompareFunc`](prism_render_architecture::particle::stencil_op::CompareFunc)
//! predicates against representative reference/buffer/mask combinations, all
//! eight
//! [`StencilOp`](prism_render_architecture::particle::stencil_op::StencilOp)
//! operations through resolve, the `IncrementWrap` / `DecrementWrap` `0` and
//! `max_val` wrap boundaries, the `Invert` bitwise complement, partial read and
//! write masks, the full `compare_passed` / `depth_passed` select matrix, and a
//! deterministic integer-`LCG` sweep across several workgroups. The `LCG` lives
//! on the host in `u64`, exactly as the golden fixtures allow; the kernel itself
//! stays pure `u32`. The whole contract is integer, so there is no branch
//! threshold to avoid.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every output is pure `u32` / `bool` integer algebra with no rounding
//! anywhere, so `CPU` and `GPU` must agree bit for bit. The comparison is an
//! exact `==` on every output word, with no tolerance: any mismatch is a genuine
//! port bug. `WGSL` has no `u64`, and the kernel is already `u32`-only, so
//! nothing is out of scope here.
//!
//! Provenance: twinned from this repository's
//! [`stencil_op`](prism_render_architecture::particle::stencil_op); no
//! third-party engine source or derived code.

use prism_render_architecture::particle::stencil_op::{
    write_masked, CompareFunc, StencilFace, StencilOp,
};
use prism_volumetric_gpu::stencil_op::{GpuStencilOp, StencilOpQuery, StencilOpResult};
use prism_volumetric_gpu::GpuContext;

/// Small linear-congruential generator for deterministic random samples,
/// mirroring the golden fixtures. The `u64` state lives on the host; the kernel
/// stays `u32`.
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
}

/// A neutral query (always-pass compare, keep ops, full masks) the per-fixture
/// helpers override field by field.
fn base() -> StencilOpQuery {
    StencilOpQuery {
        compare: CompareFunc::Always,
        fail_op: StencilOp::Keep,
        depth_fail_op: StencilOp::Keep,
        pass_op: StencilOp::Keep,
        ref_val: 0,
        stencil_val: 0,
        read_mask: 0xFF,
        write_mask: 0xFF,
        max_val: 255,
        compare_passed: true,
        depth_passed: true,
    }
}

/// Reference answer for one query, built straight from the `CPU` golden so the
/// parity assertion compares against the authoritative contract.
fn cpu_result(q: &StencilOpQuery) -> StencilOpResult {
    let face = StencilFace {
        compare: q.compare,
        reference: q.ref_val,
        read_mask: q.read_mask,
        write_mask: q.write_mask,
        fail_op: q.fail_op,
        depth_fail_op: q.depth_fail_op,
        pass_op: q.pass_op,
    };
    StencilOpResult {
        test: q.compare.test(q.ref_val, q.stencil_val, q.read_mask),
        selected_op: face.selected_op(q.compare_passed, q.depth_passed).to_u32(),
        resolved: face.resolve(q.stencil_val, q.depth_passed, q.max_val),
        write_masked: write_masked(q.stencil_val, q.ref_val, q.write_mask),
    }
}

/// Asserts the batch of queries resolves exactly element for element.
fn assert_batch(gpu: &GpuStencilOp, ctx: &GpuContext, queries: &[StencilOpQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (q, g) in queries.iter().zip(got.iter()) {
        let want = cpu_result(q);
        assert_eq!(*g, want, "stencil parity mismatch for query {q:?}");
    }
}

#[test]
fn all_compare_funcs_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStencilOp::new(&ctx);
    // Each predicate against a less-than, equal and greater-than operand pairing,
    // under both a full and a nibble read mask.
    let operands = [(2u32, 5u32), (5, 5), (9, 5), (0xF3, 0x03)];
    let masks = [0xFF_FF_FF_FFu32, 0xFF, 0x0F];
    let mut queries = Vec::new();
    for compare in CompareFunc::ALL {
        for &(ref_val, stencil_val) in &operands {
            for &read_mask in &masks {
                queries.push(StencilOpQuery {
                    compare,
                    ref_val,
                    stencil_val,
                    read_mask,
                    ..base()
                });
            }
        }
    }
    assert_batch(&gpu, &ctx, &queries);
}

#[test]
fn all_ops_resolve_through_pass() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStencilOp::new(&ctx);
    // Always-pass compare with depth passing routes every op through pass_op, so
    // apply() is exercised for all eight operations across several buffer values.
    let currents = [0u32, 1, 7, 128, 255, 0xFF_FF_FF_FF];
    let mut queries = Vec::new();
    for pass_op in StencilOp::ALL {
        for &stencil_val in &currents {
            queries.push(StencilOpQuery {
                compare: CompareFunc::Always,
                pass_op,
                ref_val: 42,
                stencil_val,
                max_val: 255,
                ..base()
            });
        }
    }
    assert_batch(&gpu, &ctx, &queries);
}

#[test]
fn incr_decr_wrap_boundaries() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStencilOp::new(&ctx);
    // IncrementWrap at max wraps to 0; DecrementWrap at 0 wraps to max; the clamp
    // variants saturate at the same boundaries instead.
    let max = 7u32;
    let queries = [
        StencilOpQuery {
            pass_op: StencilOp::IncrementWrap,
            stencil_val: max,
            max_val: max,
            ..base()
        },
        StencilOpQuery {
            pass_op: StencilOp::IncrementWrap,
            stencil_val: 0,
            max_val: max,
            ..base()
        },
        StencilOpQuery {
            pass_op: StencilOp::DecrementWrap,
            stencil_val: 0,
            max_val: max,
            ..base()
        },
        StencilOpQuery {
            pass_op: StencilOp::DecrementWrap,
            stencil_val: max,
            max_val: max,
            ..base()
        },
        StencilOpQuery {
            pass_op: StencilOp::IncrementClamp,
            stencil_val: max,
            max_val: max,
            ..base()
        },
        StencilOpQuery {
            pass_op: StencilOp::DecrementClamp,
            stencil_val: 0,
            max_val: max,
            ..base()
        },
    ];
    assert_batch(&gpu, &ctx, &queries);
}

#[test]
fn invert_is_bitwise_complement() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStencilOp::new(&ctx);
    // Invert takes the bitwise complement; with a full write mask the resolve
    // writes it through verbatim.
    let values = [0u32, 0x0F, 0xFF, 0x1234_5678, 0xFF_FF_FF_FF];
    let mut queries = Vec::new();
    for &stencil_val in &values {
        queries.push(StencilOpQuery {
            pass_op: StencilOp::Invert,
            stencil_val,
            write_mask: 0xFF_FF_FF_FF,
            ..base()
        });
    }
    assert_batch(&gpu, &ctx, &queries);
}

#[test]
fn partial_masks_preserve_bits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStencilOp::new(&ctx);
    // A nibble write mask over Replace keeps the old high bits; a nibble read
    // mask over Equal makes high-bit-differing operands compare equal.
    let queries = [
        StencilOpQuery {
            compare: CompareFunc::Always,
            pass_op: StencilOp::Replace,
            ref_val: 0xFF,
            stencil_val: 0x30,
            write_mask: 0x0F,
            ..base()
        },
        StencilOpQuery {
            compare: CompareFunc::Equal,
            ref_val: 0xF3,
            stencil_val: 0x03,
            read_mask: 0x0F,
            write_mask: 0xF0_F0_F0_F0,
            pass_op: StencilOp::Replace,
            ..base()
        },
        StencilOpQuery {
            compare: CompareFunc::Always,
            pass_op: StencilOp::Replace,
            ref_val: 0x1122_3344,
            stencil_val: 0xAABB_CCDD,
            write_mask: 0x00FF_00FF,
            ..base()
        },
    ];
    assert_batch(&gpu, &ctx, &queries);
}

#[test]
fn selected_op_covers_outcome_matrix() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStencilOp::new(&ctx);
    // The host-supplied compare/depth outcomes select among three distinct ops,
    // so each of the four (compare_passed, depth_passed) corners is checked.
    let mut queries = Vec::new();
    for compare_passed in [false, true] {
        for depth_passed in [false, true] {
            queries.push(StencilOpQuery {
                fail_op: StencilOp::Zero,
                depth_fail_op: StencilOp::Invert,
                pass_op: StencilOp::Replace,
                ref_val: 0x55,
                stencil_val: 0x30,
                compare_passed,
                depth_passed,
                ..base()
            });
        }
    }
    assert_batch(&gpu, &ctx, &queries);
}

#[test]
fn random_sweep_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStencilOp::new(&ctx);
    // 256 pseudo-random queries span four workgroups, exercising the full enum
    // code space against arbitrary u32 operands, masks and ceilings.
    let mut rng = Lcg::new(0x5151_2718_2818_2845);
    let mut queries = Vec::new();
    for _ in 0..256 {
        let compare = CompareFunc::ALL[(rng.next_u32() % 8) as usize];
        let fail_op = StencilOp::ALL[(rng.next_u32() % 8) as usize];
        let depth_fail_op = StencilOp::ALL[(rng.next_u32() % 8) as usize];
        let pass_op = StencilOp::ALL[(rng.next_u32() % 8) as usize];
        queries.push(StencilOpQuery {
            compare,
            fail_op,
            depth_fail_op,
            pass_op,
            ref_val: rng.next_u32(),
            stencil_val: rng.next_u32(),
            read_mask: rng.next_u32(),
            write_mask: rng.next_u32(),
            max_val: rng.next_u32(),
            compare_passed: rng.next_u32() & 1 == 1,
            depth_passed: rng.next_u32() & 1 == 1,
        });
    }
    assert_batch(&gpu, &ctx, &queries);
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStencilOp::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.evaluate(&ctx, &[]).is_empty());
}
