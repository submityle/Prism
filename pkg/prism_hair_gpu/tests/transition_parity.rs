//! Real-device parity for the continuous-LOD screen-door dither twin:
//! [`GpuHairTransition`] must reproduce the `CPU` golden
//! [`strand_survives_dither`](prism_render_architecture::hair::transition::strand_survives_dither)
//! for a batch of strands at a given cross-fade blend, including the
//! everyone-survives (`blend == 0`), everyone-dissolves (`blend == 1`),
//! partial-dissolve, out-of-range-saturation and large-batch cases, plus the
//! empty (`strand_count == 0`) no-op.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The decision reproduces the golden's hand-written `splitmix64` integer hash
//! bit-for-bit and compares the resulting float against `blend`. The path has no
//! transcendental and no fused multiply-add, so the `CPU` and `GPU` produce the
//! identical survive flag for every strand. Parity is therefore asserted as
//! exact per-strand `bool` equality — a single mismatched flag fails the test.
//! Partial cases also assert the surviving count is neither everyone nor nobody,
//! so a degenerate all-true or all-false kernel could not pass. No `sin`/`cos`
//! appears anywhere.
//!
//! Provenance: standard stable-hash screen-door / stochastic-LOD dither; no
//! Unreal Engine source or derived code.

use prism_hair_gpu::transition::GpuHairTransition;
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::transition::strand_survives_dither;

/// Reference survive flags for `strand_count` strands via the `CPU` golden.
fn cpu_flags(seed: u32, blend: f32, strand_count: usize) -> Vec<bool> {
    (0..strand_count)
        .map(|i| strand_survives_dither(seed, i as u32, blend))
        .collect()
}

/// Asserts every `GPU` survive flag exactly equals its `CPU` golden.
fn assert_parity(seed: u32, blend: f32, strand_count: usize, gpu: &[bool]) {
    let cpu = cpu_flags(seed, blend, strand_count);
    assert_eq!(gpu.len(), cpu.len(), "one flag per strand");
    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert_eq!(
            g, c,
            "strand {i}: gpu {g}, cpu {c} (seed {seed}, blend {blend})"
        );
    }
}

/// Acquires a headless context, or `None` (with a skip notice) when the host has
/// no `wgpu` adapter so the suite stays green off-device.
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn context_or_skip(label: &str) -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping {label}: no wgpu adapter on this host");
            None
        }
    }
}

#[test]
fn blend_zero_keeps_every_strand() {
    let Some(ctx) = context_or_skip("transition blend-zero parity") else {
        return;
    };
    let dither = GpuHairTransition::new(&ctx);
    let n = 256usize;
    let flags = dither.eval(&ctx, 0x1234_5678, 0.0, n);
    assert_parity(0x1234_5678, 0.0, n, &flags);
    assert!(flags.iter().all(|&s| s), "blend 0 must keep every strand");
}

#[test]
fn blend_one_drops_every_strand() {
    let Some(ctx) = context_or_skip("transition blend-one parity") else {
        return;
    };
    let dither = GpuHairTransition::new(&ctx);
    let n = 256usize;
    let flags = dither.eval(&ctx, 0x1234_5678, 1.0, n);
    assert_parity(0x1234_5678, 1.0, n, &flags);
    assert!(flags.iter().all(|&s| !s), "blend 1 must drop every strand");
}

#[test]
fn partial_blend_dissolves_a_deterministic_subset() {
    let Some(ctx) = context_or_skip("transition partial parity") else {
        return;
    };
    let dither = GpuHairTransition::new(&ctx);
    let n = 256usize;
    let flags = dither.eval(&ctx, 0x1234_5678, 0.5, n);
    assert_parity(0x1234_5678, 0.5, n, &flags);
    let kept = flags.iter().filter(|&&s| s).count();
    assert!(
        kept > 0 && kept < n,
        "blend 0.5 must partially dissolve, kept {kept}"
    );
    let frac = kept as f32 / n as f32;
    assert!(
        (frac - 0.5).abs() < 0.15,
        "kept fraction {frac} should sit near (1 - blend) = 0.5"
    );
}

#[test]
fn a_second_seed_dissolves_independently() {
    let Some(ctx) = context_or_skip("transition second-seed parity") else {
        return;
    };
    let dither = GpuHairTransition::new(&ctx);
    let n = 256usize;
    let flags = dither.eval(&ctx, 0x0BAD_F00D, 0.5, n);
    assert_parity(0x0BAD_F00D, 0.5, n, &flags);
    let kept = flags.iter().filter(|&&s| s).count();
    assert!(
        kept > 0 && kept < n,
        "second seed must also partially dissolve, kept {kept}"
    );
}

#[test]
fn large_batch_matches_across_workgroups() {
    let Some(ctx) = context_or_skip("transition large-batch parity") else {
        return;
    };
    let dither = GpuHairTransition::new(&ctx);
    let n = 4096usize;
    let flags = dither.eval(&ctx, 0x00C0_FFEE, 0.25, n);
    assert_parity(0x00C0_FFEE, 0.25, n, &flags);
    let kept = flags.iter().filter(|&&s| s).count();
    let frac = kept as f32 / n as f32;
    assert!(
        (frac - 0.75).abs() < 0.1,
        "kept fraction {frac} should sit near (1 - blend) = 0.75"
    );
}

#[test]
fn out_of_range_blend_saturates() {
    let Some(ctx) = context_or_skip("transition saturation parity") else {
        return;
    };
    let dither = GpuHairTransition::new(&ctx);
    let n = 256usize;
    let low = dither.eval(&ctx, 0x0042_0042, -0.5, n);
    assert_parity(0x0042_0042, -0.5, n, &low);
    assert!(
        low.iter().all(|&s| s),
        "negative blend saturates to keep all"
    );
    let high = dither.eval(&ctx, 0x0042_0042, 2.0, n);
    assert_parity(0x0042_0042, 2.0, n, &high);
    assert!(
        high.iter().all(|&s| !s),
        "blend above one saturates to drop all"
    );
}

#[test]
fn empty_batch_is_a_noop() {
    let Some(ctx) = context_or_skip("transition empty no-op") else {
        return;
    };
    let dither = GpuHairTransition::new(&ctx);
    let flags = dither.eval(&ctx, 0x1234_5678, 0.5, 0);
    assert!(flags.is_empty(), "zero strands yields an empty result");
}
