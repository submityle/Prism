//! Real-device parity for the premultiplied-alpha / `Porter-Duff` twin:
//! [`GpuPremultiplyAlpha`](prism_volumetric_gpu::premultiply_alpha::GpuPremultiplyAlpha)
//! must reproduce the `CPU` golden
//! [`premultiply_alpha`](prism_render_architecture::particle::premultiply_alpha)
//! functions across the straight/premultiplied round trip (including the
//! degenerate `alpha = 0` guard), every [`BlendOp`] variant composited both one
//! operator at a time and mixed within a single dispatch, and random batches of
//! the [`BlendOp::SrcOver`] workhorse — each compared colour-for-colour.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernels are portable core-`WGSL`, so they need no optional device
//! feature.
//!
//! # Parity criterion
//!
//! Each colour is a fixed, non-reorderable closed-form expression, so `CPU` and
//! `GPU` evaluate the same algebra in the same order. They are not bit-exact: a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate,
//! perturbing the low mantissa bits by a few units in the last place. The
//! comparison therefore allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` — loose
//! enough to admit a legal fused multiply-add contraction yet tight enough to
//! fail a genuinely wrong port (a swapped blend factor, a dropped clamp, a
//! missing degenerate guard). The degenerate `alpha = 0` branch is exact on
//! both sides and is asserted directly.
//!
//! Provenance: standard `Porter-Duff` compositing algebra (`Porter` and `Duff`,
//! "Compositing Digital Images", `SIGGRAPH` 1984); no third-party engine source
//! or derived code.

use prism_render_architecture::particle::premultiply_alpha::{
    composite, composite_over_batch, premul_to_straight, straight_to_premul, BlendOp, PremulRgba,
    Rgba,
};
use prism_volumetric_gpu::premultiply_alpha::GpuPremultiplyAlpha;
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in
/// the last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Every [`BlendOp`] variant, so a single list drives both the per-operator and
/// the mixed-dispatch coverage.
const ALL_OPS: [BlendOp; 14] = [
    BlendOp::Clear,
    BlendOp::Src,
    BlendOp::Dst,
    BlendOp::SrcOver,
    BlendOp::DstOver,
    BlendOp::SrcIn,
    BlendOp::DstIn,
    BlendOp::SrcOut,
    BlendOp::DstOut,
    BlendOp::SrcAtop,
    BlendOp::DstAtop,
    BlendOp::Xor,
    BlendOp::Plus,
    BlendOp::Add,
];

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Returns whether two premultiplied colours agree on all four lanes.
fn premul_close(a: PremulRgba, b: PremulRgba) -> bool {
    close(a.r, b.r) && close(a.g, b.g) && close(a.b, b.b) && close(a.a, b.a)
}

/// Returns whether two straight colours agree on all four lanes.
fn straight_close(a: Rgba, b: Rgba) -> bool {
    close(a.r, b.r) && close(a.g, b.g) && close(a.b, b.b) && close(a.a, b.a)
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Draws a plausible premultiplied colour: alpha in `[0, 1)` and each premulti-
/// plied channel in `[0, alpha]`, so the values stay in the physical range the
/// operators expect without ever hitting the exact floor.
fn random_premul(state: &mut u64) -> PremulRgba {
    let a = lcg(state);
    PremulRgba::new(lcg(state) * a, lcg(state) * a, lcg(state) * a, a)
}

/// Draws a plausible straight colour: alpha in `[0, 1)` and each straight
/// channel in `[0, 1)`.
fn random_straight(state: &mut u64) -> Rgba {
    Rgba::new(lcg(state), lcg(state), lcg(state), lcg(state))
}

#[test]
fn straight_to_premul_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPremultiplyAlpha::new(&ctx);
    let mut state = 0x1357_9bdf_0246_8ace_u64;
    let mut colors = vec![
        // Fully transparent: coverage zero must zero the premultiplied rgb.
        Rgba::new(1.0, 1.0, 1.0, 0.0),
        // Fully opaque: premultiplied rgb equals straight rgb.
        Rgba::new(0.4, 0.3, 0.2, 1.0),
        // Half coverage.
        Rgba::new(1.0, 0.5, 0.25, 0.5),
    ];
    for _ in 0..64 {
        colors.push(random_straight(&mut state));
    }
    let got = gpu.straight_to_premul(&ctx, &colors);
    assert_eq!(got.len(), colors.len(), "length must match the input");
    for (idx, (&c, &g)) in colors.iter().zip(got.iter()).enumerate() {
        let want = straight_to_premul(c);
        assert!(
            premul_close(g, want),
            "colour {idx}: gpu {g:?} vs cpu {want:?}"
        );
    }
}

#[test]
fn premul_to_straight_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPremultiplyAlpha::new(&ctx);
    let mut state = 0x2b7e_1516_28ae_d2a6_u64;
    let mut colors = vec![
        // Degenerate: alpha exactly zero must return a fully transparent colour.
        PremulRgba::new(0.0, 0.0, 0.0, 0.0),
        // Tiny-but-above-floor alpha exercises the reciprocal path.
        PremulRgba::new(0.0005, 0.0003, 0.0002, 0.001),
        // Opaque: dividing by one is the identity on rgb.
        PremulRgba::new(0.4, 0.3, 0.2, 1.0),
    ];
    for _ in 0..64 {
        colors.push(random_premul(&mut state));
    }
    let got = gpu.premul_to_straight(&ctx, &colors);
    assert_eq!(got.len(), colors.len(), "length must match the input");
    for (idx, (&c, &g)) in colors.iter().zip(got.iter()).enumerate() {
        let want = premul_to_straight(c);
        assert!(
            straight_close(g, want),
            "colour {idx}: gpu {g:?} vs cpu {want:?}"
        );
    }
}

#[test]
fn premul_to_straight_zero_alpha_is_transparent() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPremultiplyAlpha::new(&ctx);
    // The degenerate guard must zero every lane rather than divide by zero,
    // exactly as the reference does.
    let colors = [PremulRgba::new(0.9, 0.4, 0.7, 0.0)];
    let got = gpu.premul_to_straight(&ctx, &colors);
    assert!(
        straight_close(got[0], Rgba::new(0.0, 0.0, 0.0, 0.0)),
        "zero alpha must yield a fully transparent straight colour, got {:?}",
        got[0]
    );
}

#[test]
fn straight_premul_round_trip_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPremultiplyAlpha::new(&ctx);
    let mut state = 0x3243_f6a8_885a_308d_u64;
    let mut colors = Vec::new();
    for _ in 0..96 {
        colors.push(random_straight(&mut state));
    }
    // Fold coverage in on-device, then divide it back out on-device; for a
    // positive alpha the composition is the identity, exactly as it is on the
    // reference path, so compare against the reference round trip.
    let premul = gpu.straight_to_premul(&ctx, &colors);
    let back = gpu.premul_to_straight(&ctx, &premul);
    for (idx, (&c, &g)) in colors.iter().zip(back.iter()).enumerate() {
        let want = premul_to_straight(straight_to_premul(c));
        assert!(
            straight_close(g, want),
            "colour {idx}: gpu {g:?} vs cpu {want:?}"
        );
    }
}

#[test]
fn composite_covers_every_blend_op() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPremultiplyAlpha::new(&ctx);
    let mut state = 0xa409_3822_299f_31d0_u64;
    // Build a shared src/dst batch that includes transparent, opaque and
    // semi-transparent extremes plus random colours, then run every operator
    // over the whole batch with a uniform tag.
    let mut src = vec![
        PremulRgba::TRANSPARENT,
        PremulRgba::new(0.4, 0.3, 0.2, 1.0),
        PremulRgba::new(0.2, 0.2, 0.2, 0.5),
    ];
    let mut dst = vec![
        PremulRgba::new(0.1, 0.9, 0.5, 0.7),
        PremulRgba::TRANSPARENT,
        PremulRgba::new(0.6, 0.1, 0.3, 0.8),
    ];
    for _ in 0..48 {
        src.push(random_premul(&mut state));
        dst.push(random_premul(&mut state));
    }
    for op in ALL_OPS {
        let ops = vec![op; src.len()];
        let got = gpu.composite(&ctx, &ops, &src, &dst);
        assert_eq!(got.len(), src.len(), "{op:?}: length must match the input");
        for (idx, &g) in got.iter().enumerate() {
            let want = composite(op, src[idx], dst[idx]);
            assert!(
                premul_close(g, want),
                "{op:?} colour {idx}: gpu {g:?} vs cpu {want:?}"
            );
        }
    }
}

#[test]
fn composite_mixes_all_ops_in_one_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPremultiplyAlpha::new(&ctx);
    let mut state = 0x5eed_4a7d_0bad_c0de_u64;
    // One dispatch whose per-element tag cycles through every variant, so the
    // kernel's tag branch is exercised against neighbours of a different op.
    let count = ALL_OPS.len() * 7;
    let mut ops = Vec::with_capacity(count);
    let mut src = Vec::with_capacity(count);
    let mut dst = Vec::with_capacity(count);
    for i in 0..count {
        ops.push(ALL_OPS[i % ALL_OPS.len()]);
        src.push(random_premul(&mut state));
        dst.push(random_premul(&mut state));
    }
    let got = gpu.composite(&ctx, &ops, &src, &dst);
    assert_eq!(got.len(), count, "length must match the input");
    for (idx, &g) in got.iter().enumerate() {
        let want = composite(ops[idx], src[idx], dst[idx]);
        assert!(
            premul_close(g, want),
            "{:?} colour {idx}: gpu {g:?} vs cpu {want:?}",
            ops[idx]
        );
    }
}

#[test]
fn plus_and_add_clamp_each_channel_to_one() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPremultiplyAlpha::new(&ctx);
    // A deliberately over-unity pair pins the additive clamp on every lane and
    // confirms `Plus` and `Add` take the identical branch.
    let src = [PremulRgba::new(0.7, 0.2, 0.9, 0.6)];
    let dst = [PremulRgba::new(0.5, 0.3, 0.4, 0.7)];
    let via_plus = gpu.composite(&ctx, &[BlendOp::Plus], &src, &dst);
    let via_add = gpu.composite(&ctx, &[BlendOp::Add], &src, &dst);
    let want = PremulRgba::new(1.0, 0.5, 1.0, 1.0);
    assert!(premul_close(via_plus[0], want), "plus: {:?}", via_plus[0]);
    assert!(premul_close(via_add[0], want), "add: {:?}", via_add[0]);
}

#[test]
fn composite_over_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPremultiplyAlpha::new(&ctx);
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut src = vec![
        PremulRgba::new(0.4, 0.3, 0.2, 1.0),
        PremulRgba::TRANSPARENT,
        PremulRgba::new(0.2, 0.2, 0.2, 0.5),
    ];
    let mut dst = vec![
        PremulRgba::new(0.1, 0.9, 0.5, 0.7),
        PremulRgba::new(0.6, 0.1, 0.3, 0.8),
        PremulRgba::new(0.3, 0.3, 0.3, 0.4),
    ];
    for _ in 0..128 {
        src.push(random_premul(&mut state));
        dst.push(random_premul(&mut state));
    }
    let got = gpu.composite_over_batch(&ctx, &src, &dst);
    let want = composite_over_batch(&src, &dst);
    assert_eq!(got.len(), want.len(), "length must match the reference");
    for (idx, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(premul_close(g, w), "colour {idx}: gpu {g:?} vs cpu {w:?}");
    }
}

#[test]
fn composite_over_batch_length_is_shorter_input() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPremultiplyAlpha::new(&ctx);
    // The result length follows the shorter input, exactly as the reference does
    // so a length mismatch can never index out of bounds.
    let long = [
        PremulRgba::new(0.4, 0.3, 0.2, 1.0),
        PremulRgba::new(0.2, 0.2, 0.2, 0.5),
    ];
    let short = [PremulRgba::new(0.1, 0.9, 0.5, 0.7)];
    assert_eq!(gpu.composite_over_batch(&ctx, &long, &short).len(), 1);
    assert_eq!(gpu.composite_over_batch(&ctx, &short, &long).len(), 1);
}

#[test]
fn empty_inputs_return_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPremultiplyAlpha::new(&ctx);
    // Empty inputs issue no dispatch (a storage buffer cannot be zero-sized) and
    // return an empty vector on every entry point.
    assert!(gpu.straight_to_premul(&ctx, &[]).is_empty());
    assert!(gpu.premul_to_straight(&ctx, &[]).is_empty());
    assert!(gpu.composite(&ctx, &[], &[], &[]).is_empty());
    assert!(gpu.composite_over_batch(&ctx, &[], &[]).is_empty());
}
