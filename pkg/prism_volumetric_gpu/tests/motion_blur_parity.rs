//! Real-device parity for the screen-space per-particle motion-blur twin:
//! [`GpuMotionBlur`](prism_volumetric_gpu::motion_blur::GpuMotionBlur) must
//! reproduce the `CPU` golden
//! [`motion_blur`](prism_render_architecture::particle::motion_blur) —
//! [`MotionBlurParams::build_taps`](prism_render_architecture::particle::motion_blur::MotionBlurParams::build_taps)
//! plus
//! [`accumulate_taps`](prism_render_architecture::particle::motion_blur::accumulate_taps)
//! — particle for particle across empty, zero-motion, large-speed, random
//! multi-resolution and degenerate inputs.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The `SplitMix32` jitter hash is exact integer arithmetic plus an exact
//! power-of-two scale, so the per-particle jitter offset and the jittered `tap`
//! coordinates are bit-identical on both engines: the `tap` count and each
//! `tap`'s coordinate `t` match exactly. The offsets, weights and accumulated
//! colour then fold those coordinates through short closed forms (one `sqrt`, a
//! few divides and multiply-adds); they are not guaranteed bit-exact because a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate,
//! perturbing the low mantissa bits by a few units in the last place, which the
//! absolute/relative bound admits. The zero-motion and empty-colour cases are
//! additionally pinned bit for bit (`to_bits`), where the offsets collapse to an
//! exact zero vector and the accumulated colour to an exact transparent black.
//!
//! Provenance: standard reconstruction-filter per-particle motion blur; no
//! Unreal Engine source or derived code.

use prism_render_architecture::particle::motion_blur::{
    accumulate_taps, MotionBlurParams, Rgba, Vec2,
};
use prism_volumetric_gpu::motion_blur::{GpuMotionBlur, MotionBlurQuery, MotionBlurResult};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound for the folded offsets, weights and colours. A `GPU`
/// may fuse a multiply-add the scalar reference leaves separate (the direction
/// reciprocal, the weight denominator, the normalization and accumulation
/// divides), perturbing the low mantissa bits by a few units in the last place.
const ABS_EPS: f32 = 1.0e-4;

/// Relative parity bound, applied to the larger magnitude, so the larger
/// pixel-space offsets (tens of pixels) are judged on relative error while the
/// unit-scale weights and colours fall back to [`ABS_EPS`].
const REL_EPS: f32 = 1.0e-3;

/// Whether two scalars agree within the absolute or relative parity bound.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    diff <= ABS_EPS || diff <= REL_EPS * a.abs().max(b.abs())
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg_unit(state: &mut u64) -> f32 {
    // Knuth multiplier / increment; the shift takes the high bits where the
    // generator mixes best.
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    // 24 usable mantissa bits mapped onto [0, 1).
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Runs the motion blur on both the `CPU` reference and the `GPU` twin for the
/// same params and queries, asserts particle-for-particle parity within the
/// documented bound on offsets, weights and accumulated colour, and returns the
/// `GPU` results for any further shape assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuMotionBlur,
    params: &MotionBlurParams,
    queries: &[MotionBlurQuery<'_>],
) -> Vec<MotionBlurResult> {
    let got = gpu.eval(ctx, params, queries);
    assert_eq!(got.len(), queries.len(), "one result per particle");

    let taps = params.effective_taps();
    for (idx, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let kernel = params.build_taps(q.motion_px, q.particle_seed);
        let cpu_accum = accumulate_taps(q.tap_colors, &kernel.weights);

        assert_eq!(g.offsets.len(), taps, "particle {idx} offset count");
        assert_eq!(g.weights.len(), taps, "particle {idx} weight count");
        assert_eq!(kernel.offsets.len(), taps, "reference offset count");

        for (t, (go, co)) in g.offsets.iter().zip(kernel.offsets.iter()).enumerate() {
            assert!(
                go.x.is_finite() && go.y.is_finite(),
                "particle {idx} tap {t} offset non-finite"
            );
            assert!(
                close(go.x, co.x) && close(go.y, co.y),
                "particle {idx} tap {t} offset mismatch: gpu {go:?}, cpu {co:?}"
            );
        }
        for (t, (gw, cw)) in g.weights.iter().zip(kernel.weights.iter()).enumerate() {
            assert!(gw.is_finite(), "particle {idx} tap {t} weight non-finite");
            assert!(
                close(*gw, *cw),
                "particle {idx} tap {t} weight mismatch: gpu {gw}, cpu {cw}"
            );
        }
        for (c, (ga, ca)) in g.accumulated.iter().zip(cpu_accum.iter()).enumerate() {
            assert!(
                ga.is_finite(),
                "particle {idx} channel {c} accum non-finite"
            );
            assert!(
                close(*ga, *ca),
                "particle {idx} channel {c} accum mismatch: gpu {ga}, cpu {ca}"
            );
        }
    }
    got
}

/// Base params: a 50-pixel streak cap, no jitter, full shutter and five taps.
fn base_params(tap_count: u32) -> MotionBlurParams {
    MotionBlurParams::new(50.0, 0.0, 1.0, tap_count)
}

/// A simple opaque colour run of `n` distinct, deterministic RGBA samples.
fn color_run(n: usize) -> Vec<Rgba> {
    (0..n)
        .map(|i| {
            let f = i as f32;
            [0.1 * f, 1.0 - 0.05 * f, 0.5, 1.0]
        })
        .collect()
}

#[test]
fn empty_input_yields_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionBlur::new(&ctx);
    let out = gpu.eval(&ctx, &base_params(5), &[]);
    assert!(
        out.is_empty(),
        "an empty particle batch yields an empty result"
    );
}

#[test]
fn zero_motion_is_centered_passthrough() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionBlur::new(&ctx);
    let params = base_params(7);

    // A zero-length motion vector (and a sub-EPS_LEN vector) has no direction,
    // so every tap offset is the exact zero vector and the kernel degenerates to
    // a centred passthrough. The weights are still the normalized rational
    // profile, so the accumulated colour is the weighted average of the run.
    let colors = color_run(7);
    let queries = [
        MotionBlurQuery {
            motion_px: Vec2::ZERO,
            particle_seed: 11,
            tap_colors: &colors,
        },
        MotionBlurQuery {
            motion_px: Vec2::new(1e-13, -1e-13),
            particle_seed: 29,
            tap_colors: &colors,
        },
    ];

    let got = check(&ctx, &gpu, &params, &queries);
    for (idx, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        // Pin the degenerate offsets bit for bit against the `CPU` golden. With
        // no direction the golden evaluates `dir.scale(t * half)` from a zero
        // `dir`, so the taps whose signed coordinate is negative multiply
        // `+0.0` by a negative scalar and yield the IEEE signed zero `-0.0`; the
        // `GPU` kernel folds the identical arithmetic and reproduces the same
        // signed zero, so the two engines agree on every bit.
        let kernel = params.build_taps(q.motion_px, q.particle_seed);
        for (t, (off, cpu_off)) in g.offsets.iter().zip(kernel.offsets.iter()).enumerate() {
            assert_eq!(
                (off.x.to_bits(), off.y.to_bits()),
                (cpu_off.x.to_bits(), cpu_off.y.to_bits()),
                "particle {idx} tap {t} offset must bit-match the CPU zero vector"
            );
            // And it is genuinely the zero vector: `abs` folds either signed
            // zero onto `+0.0`, whose bit pattern is all-zero, so this pins the
            // magnitude without an `f32` equality test.
            assert_eq!(
                (off.x.abs().to_bits(), off.y.abs().to_bits()),
                (0u32, 0u32),
                "particle {idx} tap {t} offset must be the zero vector for zero motion"
            );
        }
    }
}

#[test]
fn large_speed_many_taps_clamps_and_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionBlur::new(&ctx);

    // A fast particle (hundreds of px/frame) whose streak is clamped to
    // max_blur_px, spread across many jittered taps. Parity must hold for the
    // clamped span, the jittered coordinates and the accumulation alike.
    let params = MotionBlurParams::new(40.0, 1.0, 1.0, 15);
    let colors = color_run(15);
    let queries = [
        MotionBlurQuery {
            motion_px: Vec2::new(400.0, 0.0),
            particle_seed: 7,
            tap_colors: &colors,
        },
        MotionBlurQuery {
            motion_px: Vec2::new(-250.0, 180.0),
            particle_seed: 4242,
            tap_colors: &colors,
        },
        MotionBlurQuery {
            motion_px: Vec2::new(0.0, -512.0),
            particle_seed: 99,
            tap_colors: &colors,
        },
    ];

    let got = check(&ctx, &gpu, &params, &queries);

    // The clamped half-span is max_blur_px / 2 = 20 px; the farthest tap offset
    // magnitude cannot exceed that (coordinates are clamped to [-1, 1]).
    for g in &got {
        for off in &g.offsets {
            let mag = (off.x * off.x + off.y * off.y).sqrt();
            assert!(
                mag <= 20.0 + ABS_EPS,
                "a clamped streak cannot smear past max_blur_px / 2"
            );
        }
    }
}

#[test]
fn random_motion_field_multi_resolution() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionBlur::new(&ctx);

    // A spread of (particle count, tap count) resolutions, each with its own
    // random motion-vector field, random per-tap colours and random params, so
    // the jittered taps, the direction, the weights and the accumulation are all
    // exercised per particle across batch sizes that cross workgroup boundaries.
    let cases: [(usize, u32, u64); 5] = [
        (1, 1, 0x1234_5678_9abc_def0),
        (16, 4, 0x0f0f_0f0f_1234_5678),
        (37, 9, 0xdead_beef_cafe_babe),
        (64, 12, 0x5555_aaaa_3333_cccc),
        (130, 16, 0x9e37_79b9_7f4a_7c15),
    ];

    for (particle_count, tap_count, seed) in cases {
        let mut state = seed;

        // Random, finite params over a useful range.
        let max_blur_px = 1.0 + lcg_unit(&mut state) * 60.0;
        let jitter_strength = lcg_unit(&mut state);
        let shutter = lcg_unit(&mut state);
        let params = MotionBlurParams::new(max_blur_px, jitter_strength, shutter, tap_count);

        // Per-particle: a random motion vector (occasionally near zero so the
        // degenerate direction branch is exercised), a random seed and a random
        // RGBA colour run of exactly tap_count samples.
        let mut color_store: Vec<Vec<Rgba>> = Vec::with_capacity(particle_count);
        let mut motions: Vec<Vec2> = Vec::with_capacity(particle_count);
        let mut seeds: Vec<u32> = Vec::with_capacity(particle_count);
        for _ in 0..particle_count {
            // Signed motion spanning roughly [-300, 300] px with the occasional
            // tiny vector.
            let mx = (lcg_unit(&mut state) - 0.5) * 600.0;
            let my = (lcg_unit(&mut state) - 0.5) * 600.0;
            let scale = if lcg_unit(&mut state) < 0.1 {
                1e-14
            } else {
                1.0
            };
            motions.push(Vec2::new(mx * scale, my * scale));
            seeds.push((state >> 16) as u32);

            let mut colors = Vec::with_capacity(tap_count as usize);
            for _ in 0..tap_count {
                colors.push([
                    lcg_unit(&mut state),
                    lcg_unit(&mut state),
                    lcg_unit(&mut state),
                    lcg_unit(&mut state),
                ]);
            }
            color_store.push(colors);
        }

        let queries: Vec<MotionBlurQuery<'_>> = (0..particle_count)
            .map(|i| MotionBlurQuery {
                motion_px: motions[i],
                particle_seed: seeds[i],
                tap_colors: &color_store[i],
            })
            .collect();

        check(&ctx, &gpu, &params, &queries);
    }
}

#[test]
fn degenerate_resolutions_do_not_panic() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionBlur::new(&ctx);

    // A zero tap count is clamped to a single centre tap by the reference's
    // constructor; a single-tap kernel is a centred, full-weight passthrough.
    let single = MotionBlurParams::new(30.0, 0.5, 1.0, 0);
    assert_eq!(single.effective_taps(), 1, "tap count floors at one");
    let one_color = color_run(1);
    let single_query = MotionBlurQuery {
        motion_px: Vec2::new(80.0, 20.0),
        particle_seed: 5,
        tap_colors: &one_color,
    };
    let got = check(&ctx, &gpu, &single, &[single_query]);
    assert_eq!(got[0].offsets.len(), 1, "single-tap kernel has one offset");
    assert_eq!(
        (got[0].offsets[0].x.to_bits(), got[0].offsets[0].y.to_bits()),
        (0.0f32.to_bits(), 0.0f32.to_bits()),
        "a single centre tap sits exactly on the particle"
    );

    // An empty colour run with a multi-tap kernel: the offsets and weights are
    // still built, but the accumulation has nothing to average, so it is exactly
    // transparent black.
    let params = base_params(6);
    let empty_colors: [Rgba; 0] = [];
    let empty_query = MotionBlurQuery {
        motion_px: Vec2::new(100.0, -40.0),
        particle_seed: 17,
        tap_colors: &empty_colors,
    };
    let got = check(&ctx, &gpu, &params, &[empty_query]);
    assert_eq!(
        got[0].weights.len(),
        6,
        "weights are built despite no colours"
    );
    for (c, channel) in got[0].accumulated.iter().enumerate() {
        assert_eq!(
            channel.to_bits(),
            0.0f32.to_bits(),
            "channel {c} of an empty colour run is exactly zero"
        );
    }

    // A colour run shorter than the tap count stops the accumulation early (the
    // reference zips its weights against the supplied colours), yet the full tap
    // kernel is still produced. Parity is checked by `check`.
    let short_colors = color_run(2);
    let short_query = MotionBlurQuery {
        motion_px: Vec2::new(60.0, 60.0),
        particle_seed: 23,
        tap_colors: &short_colors,
    };
    let got = check(&ctx, &gpu, &params, &[short_query]);
    assert_eq!(
        got[0].offsets.len(),
        6,
        "a short colour run still yields the full tap kernel"
    );
}
