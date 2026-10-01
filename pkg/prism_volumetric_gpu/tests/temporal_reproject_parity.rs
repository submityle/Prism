//! Real-device parity for the particle temporal-reprojection history-resolve
//! twin:
//! [`GpuTemporalReproject`](prism_volumetric_gpu::temporal_reproject::GpuTemporalReproject)
//! must reproduce the `CPU` golden
//! [`temporal_reprojection`](prism_render_architecture::particle::temporal_reprojection)
//! resolve chain pixel for pixel across empty, on-screen, off-screen,
//! disoccluded, fast-motion, threshold-boundary, random multi-resolution and
//! degenerate inputs.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The reproject is an exact `f32` subtraction, so the reprojected `UV` is
//! bit-identical on both engines. The `is_on_screen` test, the depth-reject
//! branch and the velocity saturation collapse to an exact `0.0` or `1.0` on
//! both engines, so the *hard* decisions (off-screen, disocclusion, fast
//! motion) match bit for bit and the resolved colour then collapses to exactly
//! the current sample (verified with [`f32::to_bits`]). Only the *soft* folded
//! values — the `smoothstep` cubic, the `YCoCg` linear combinations, the clip
//! crossing fraction and the final lerp — may differ by a legal fused
//! multiply-add in the low mantissa bits, which the `abs_diff <= EPS` bound
//! admits while still failing a genuinely wrong port. The random cases hold the
//! depths equal (full depth confidence, no branch to flip) and keep every soft
//! input a safe margin away from a hard threshold, so the only engine-to-engine
//! slack is fused-multiply-add rounding inside [`EPS`].
//!
//! Provenance: standard `TAA` / temporal-accumulation history resolve; no
//! Unreal Engine source or derived code.

use prism_render_architecture::particle::temporal_reprojection::{
    blend_history, clip_history_ycocg, history_valid, reproject_uv, ReprojectionParams, Rgba, Vec2,
};
use prism_volumetric_gpu::temporal_reproject::{
    GpuTemporalReproject, TemporalReprojectQuery, TemporalReprojectResult, NEIGHBORHOOD_TAPS,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound for the folded resolve. A `GPU` may fuse a
/// multiply-add the scalar reference leaves separate (the `smoothstep` cubic,
/// the `YCoCg` combinations, the clip crossing fraction, the blend lerp),
/// perturbing the low mantissa bits by a few units in the last place; `1e-5`
/// admits that legal slack while still failing a genuinely wrong port.
const EPS: f32 = 1.0e-5;

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

/// A flat `3x3` neighbourhood where every tap carries the same colour; its
/// `YCoCg` box has zero extent, so the clip pulls history exactly onto the
/// shared colour.
fn flat_neighborhood(color: Rgba) -> [Rgba; NEIGHBORHOOD_TAPS] {
    [color; NEIGHBORHOOD_TAPS]
}

/// Reproduces the reference resolve chain for one query under `params`.
///
/// Mirrors [`GpuTemporalReproject::eval`] stage for stage: reproject, validate,
/// `YCoCg`-clip, blend.
fn reference(
    params: &ReprojectionParams,
    query: &TemporalReprojectQuery,
) -> TemporalReprojectResult {
    let history_uv = reproject_uv(query.current_uv, query.motion_uv);
    let confidence = history_valid(
        history_uv,
        query.current_depth,
        query.history_depth,
        query.motion_uv,
        *params,
    );
    let clipped = clip_history_ycocg(query.current, query.history, &query.neighborhood, *params);
    let resolved = blend_history(query.current, clipped, confidence, *params);
    TemporalReprojectResult {
        resolved,
        confidence,
        history_uv,
    }
}

/// Runs the resolve on both the `CPU` reference and the `GPU` twin for the same
/// params and queries, asserts pixel-for-pixel parity within [`EPS`], and
/// returns the `GPU` results for any further shape assertions.
///
/// Every query here is built from finite inputs, so the reference and the twin
/// are both finite and the `abs_diff <= EPS` bound is meaningful.
fn check(
    ctx: &GpuContext,
    gpu: &GpuTemporalReproject,
    params: &ReprojectionParams,
    queries: &[TemporalReprojectQuery],
) -> Vec<TemporalReprojectResult> {
    let cpu: Vec<TemporalReprojectResult> = queries.iter().map(|q| reference(params, q)).collect();

    let got = gpu.eval(ctx, params, queries);
    assert_eq!(got.len(), cpu.len(), "one result per pixel");
    for (idx, (g, c)) in got.iter().zip(cpu.iter()).enumerate() {
        assert!(
            (g.history_uv.x - c.history_uv.x).abs() <= EPS
                && (g.history_uv.y - c.history_uv.y).abs() <= EPS,
            "pixel {idx} reprojected UV mismatch: gpu {:?}, cpu {:?}",
            g.history_uv,
            c.history_uv
        );
        assert!(
            (g.confidence - c.confidence).abs() <= EPS,
            "pixel {idx} confidence mismatch: gpu {}, cpu {}",
            g.confidence,
            c.confidence
        );
        assert!(
            g.confidence.is_finite() && (0.0..=1.0).contains(&g.confidence),
            "pixel {idx} confidence {} left the unit interval",
            g.confidence
        );
        for channel in 0..4 {
            assert!(
                (g.resolved[channel] - c.resolved[channel]).abs() <= EPS,
                "pixel {idx} channel {channel} mismatch: gpu {}, cpu {}",
                g.resolved[channel],
                c.resolved[channel]
            );
            assert!(
                g.resolved[channel].is_finite(),
                "pixel {idx} channel {channel} produced a non-finite colour"
            );
        }
    }
    got
}

/// Base params: slow steady-state accumulation, a `0.1` relative depth-reject
/// threshold, a `0.2` `UV` velocity reject and a tight (non-widened) box.
fn base_params() -> ReprojectionParams {
    ReprojectionParams::new(0.9, 0.1, 0.2, 0.0)
}

#[test]
fn empty_input_yields_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalReproject::new(&ctx);
    let out = gpu.eval(&ctx, &base_params(), &[]);
    assert!(
        out.is_empty(),
        "an empty query batch yields an empty result"
    );
}

#[test]
fn on_screen_history_resolves() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalReproject::new(&ctx);
    let params = base_params();

    // History reprojects well inside the frame, depths match (full depth
    // confidence) and the motion is a slow fraction of the reject speed, so the
    // blend keeps most of the (clipped) history. The neighbourhood brackets both
    // the current sample and the history, so the YCoCg clip leaves history
    // untouched and only the confidence-weighted lerp is exercised.
    let current: Rgba = [0.40, 0.55, 0.60, 1.0];
    let history: Rgba = [0.46, 0.52, 0.63, 1.0];
    let mut neighborhood = flat_neighborhood([0.50, 0.50, 0.50, 1.0]);
    neighborhood[0] = [0.30, 0.30, 0.30, 1.0];
    neighborhood[8] = [0.70, 0.70, 0.70, 1.0];

    let query = TemporalReprojectQuery {
        current_uv: Vec2::new(0.5, 0.5),
        motion_uv: Vec2::new(0.02, -0.01),
        current_depth: 10.0,
        history_depth: 10.0,
        current,
        history,
        neighborhood,
    };

    let got = check(&ctx, &gpu, &params, &[query]);
    assert!(
        got[0].confidence > 0.5,
        "a matching, on-screen, slow-motion sample keeps a high confidence, got {}",
        got[0].confidence
    );
}

#[test]
fn off_screen_history_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalReproject::new(&ctx);
    let params = base_params();

    // The motion drags the reprojection past the left screen edge, so the
    // history has no source pixel: confidence collapses to an exact 0.0 and the
    // blend weight is exactly 0.0, so the resolved colour is bit-for-bit the
    // current sample.
    let current: Rgba = [0.3, 0.6, 0.9, 1.0];
    let query = TemporalReprojectQuery {
        current_uv: Vec2::new(0.2, 0.5),
        motion_uv: Vec2::new(0.5, 0.0),
        current_depth: 10.0,
        history_depth: 10.0,
        current,
        history: [0.9, 0.1, 0.2, 1.0],
        neighborhood: flat_neighborhood([0.5, 0.5, 0.5, 1.0]),
    };

    let got = check(&ctx, &gpu, &params, &[query]);
    assert_eq!(
        got[0].confidence.to_bits(),
        0.0f32.to_bits(),
        "off-screen history has exactly zero confidence"
    );
    for (channel, (&resolved, &cur)) in got[0].resolved.iter().zip(current.iter()).enumerate() {
        assert_eq!(
            resolved.to_bits(),
            cur.to_bits(),
            "channel {channel} must collapse to the current sample bit for bit"
        );
    }
    assert!(
        got[0].history_uv.x < 0.0,
        "the reprojected UV left the frame to the left, got {}",
        got[0].history_uv.x
    );
}

#[test]
fn depth_disocclusion_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalReproject::new(&ctx);
    let params = base_params();

    // History reprojects on-screen with slow motion, but its depth is far from
    // the current surface (relative difference 0.5, well above the 0.1 reject
    // threshold with margin >> EPS so the branch cannot flip): a disocclusion,
    // confidence exactly 0.0, resolved colour bit-for-bit the current sample.
    let current: Rgba = [0.2, 0.4, 0.8, 1.0];
    let query = TemporalReprojectQuery {
        current_uv: Vec2::new(0.5, 0.5),
        motion_uv: Vec2::new(0.01, 0.0),
        current_depth: 10.0,
        history_depth: 5.0,
        current,
        history: [0.9, 0.9, 0.1, 1.0],
        neighborhood: flat_neighborhood([0.5, 0.5, 0.5, 1.0]),
    };

    let got = check(&ctx, &gpu, &params, &[query]);
    assert_eq!(
        got[0].confidence.to_bits(),
        0.0f32.to_bits(),
        "a disocclusion rejects history with exactly zero confidence"
    );
    for (channel, (&resolved, &cur)) in got[0].resolved.iter().zip(current.iter()).enumerate() {
        assert_eq!(
            resolved.to_bits(),
            cur.to_bits(),
            "channel {channel} must collapse to the current sample bit for bit"
        );
    }
}

#[test]
fn fast_motion_rejects_history() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalReproject::new(&ctx);
    let params = base_params();

    // On-screen reprojection with matching depth, but the motion magnitude
    // (0.4 UV) is twice the velocity reject (0.2), so the velocity confidence
    // saturates to an exact 0.0 and, as a factor, drives the product to exactly
    // 0.0 on both engines.
    let current: Rgba = [0.7, 0.2, 0.5, 1.0];
    let query = TemporalReprojectQuery {
        current_uv: Vec2::new(0.5, 0.5),
        motion_uv: Vec2::new(0.4, 0.0),
        current_depth: 12.0,
        history_depth: 12.0,
        current,
        history: [0.1, 0.8, 0.3, 1.0],
        neighborhood: flat_neighborhood([0.5, 0.5, 0.5, 1.0]),
    };

    let got = check(&ctx, &gpu, &params, &[query]);
    assert_eq!(
        got[0].confidence.to_bits(),
        0.0f32.to_bits(),
        "motion past the velocity reject fades confidence to exactly zero"
    );
    for (channel, (&resolved, &cur)) in got[0].resolved.iter().zip(current.iter()).enumerate() {
        assert_eq!(
            resolved.to_bits(),
            cur.to_bits(),
            "channel {channel} must collapse to the current sample bit for bit"
        );
    }
}

#[test]
fn depth_confidence_threshold_boundary() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalReproject::new(&ctx);
    let params = base_params();

    // Two queries straddle the 0.1 relative depth-reject threshold with a safe
    // margin so neither engine flips the hard branch: one just inside (relative
    // 0.05, soft confidence retained) and one just outside (relative 0.2,
    // rejected). Both are validated by the shared tolerance in `check`, and the
    // inside/outside shapes are asserted explicitly.
    let base = TemporalReprojectQuery {
        current_uv: Vec2::new(0.5, 0.5),
        motion_uv: Vec2::new(0.0, 0.0),
        current_depth: 10.0,
        history_depth: 10.0,
        current: [0.5, 0.5, 0.5, 1.0],
        history: [0.5, 0.5, 0.5, 1.0],
        neighborhood: flat_neighborhood([0.5, 0.5, 0.5, 1.0]),
    };

    let inside = TemporalReprojectQuery {
        history_depth: 9.5,
        ..base
    };
    let outside = TemporalReprojectQuery {
        history_depth: 8.0,
        ..base
    };

    let got = check(&ctx, &gpu, &params, &[inside, outside]);
    assert!(
        got[0].confidence > 0.0,
        "a sub-threshold depth gap keeps some confidence, got {}",
        got[0].confidence
    );
    assert_eq!(
        got[1].confidence.to_bits(),
        0.0f32.to_bits(),
        "a super-threshold depth gap rejects history outright"
    );
}

#[test]
fn velocity_confidence_threshold_boundary() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalReproject::new(&ctx);
    let params = base_params();

    // Two queries straddle the 0.2 UV velocity reject with a safe margin: one
    // just inside (speed 0.1, soft confidence retained) and one at/past the
    // reject (speed 0.2, saturated to zero). Depths match so the depth factor is
    // exactly one on both engines and the velocity factor is isolated.
    let base = TemporalReprojectQuery {
        current_uv: Vec2::new(0.5, 0.5),
        motion_uv: Vec2::new(0.0, 0.0),
        current_depth: 10.0,
        history_depth: 10.0,
        current: [0.5, 0.5, 0.5, 1.0],
        history: [0.5, 0.5, 0.5, 1.0],
        neighborhood: flat_neighborhood([0.5, 0.5, 0.5, 1.0]),
    };

    let inside = TemporalReprojectQuery {
        motion_uv: Vec2::new(0.1, 0.0),
        ..base
    };
    let outside = TemporalReprojectQuery {
        motion_uv: Vec2::new(0.2, 0.0),
        ..base
    };

    let got = check(&ctx, &gpu, &params, &[inside, outside]);
    assert!(
        got[0].confidence > 0.0,
        "a sub-reject speed keeps some confidence, got {}",
        got[0].confidence
    );
    assert_eq!(
        got[1].confidence.to_bits(),
        0.0f32.to_bits(),
        "a speed at the velocity reject fades confidence to exactly zero"
    );
}

#[test]
fn history_inside_box_blends_without_clipping() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalReproject::new(&ctx);
    let params = base_params();

    // The neighbourhood box fully contains both the current sample and the
    // history, so the YCoCg clip is the identity: the resolve reduces to the
    // confidence-weighted lerp, which the shared tolerance validates. A widened
    // box is also exercised in a second query.
    let neighborhood = {
        let mut n = flat_neighborhood([0.5, 0.5, 0.5, 1.0]);
        n[0] = [0.1, 0.1, 0.1, 1.0];
        n[8] = [0.9, 0.9, 0.9, 1.0];
        n
    };
    let query = TemporalReprojectQuery {
        current_uv: Vec2::new(0.5, 0.5),
        motion_uv: Vec2::new(0.03, 0.02),
        current_depth: 10.0,
        history_depth: 10.0,
        current: [0.45, 0.50, 0.55, 1.0],
        history: [0.55, 0.50, 0.45, 1.0],
        neighborhood,
    };

    check(&ctx, &gpu, &params, &[query]);

    let widen = ReprojectionParams::new(0.85, 0.1, 0.2, 1.5);
    check(&ctx, &gpu, &widen, &[query]);
}

#[test]
fn random_motion_field_multi_resolution() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalReproject::new(&ctx);

    // A spread of pixel counts, each with its own random motion field and random
    // params. Depths are held equal so the depth confidence is exactly one on
    // both engines (no hard branch to flip), every motion speed stays a safe
    // margin below the velocity reject, and the reprojected UV stays on-screen,
    // so the only engine-to-engine slack is fused-multiply-add rounding inside
    // the soft folded colour, which EPS admits.
    let cases: [(usize, u64); 5] = [
        (1, 0x1234_5678_9abc_def0),
        (16, 0x0f0f_0f0f_1234_5678),
        (37, 0xdead_beef_cafe_babe),
        (64, 0x5555_aaaa_3333_cccc),
        (130, 0x9e37_79b9_7f4a_7c15),
    ];

    for (pixel_count, seed) in cases {
        let mut state = seed;

        // Random, finite params. The velocity reject stays generous (>= 0.6) so
        // the capped motion speed is always a safe fraction of it; the depth
        // reject and widen span a useful range.
        let max_history_weight = lcg_unit(&mut state);
        let depth_reject = 0.05 + lcg_unit(&mut state) * 0.5;
        let velocity_reject = 0.6 + lcg_unit(&mut state) * 0.6;
        let clamp_widen = lcg_unit(&mut state) * 2.0;
        let params = ReprojectionParams::new(
            max_history_weight,
            depth_reject,
            velocity_reject,
            clamp_widen,
        );

        let mut queries: Vec<TemporalReprojectQuery> = Vec::with_capacity(pixel_count);
        for _ in 0..pixel_count {
            // Current UV in [0.3, 0.7] and motion capped at 0.2 UV so the
            // reprojection stays on-screen and the speed stays well under the
            // >= 0.6 velocity reject.
            let current_uv = Vec2::new(
                0.3 + lcg_unit(&mut state) * 0.4,
                0.3 + lcg_unit(&mut state) * 0.4,
            );
            let motion_uv = Vec2::new(
                (lcg_unit(&mut state) - 0.5) * 0.4,
                (lcg_unit(&mut state) - 0.5) * 0.4,
            );
            // Equal depths: full depth confidence on both engines.
            let depth = 5.0 + lcg_unit(&mut state) * 10.0;

            let current: Rgba = [
                lcg_unit(&mut state),
                lcg_unit(&mut state),
                lcg_unit(&mut state),
                1.0,
            ];
            let history: Rgba = [
                lcg_unit(&mut state),
                lcg_unit(&mut state),
                lcg_unit(&mut state),
                1.0,
            ];
            let mut neighborhood = [[0.0f32; 4]; NEIGHBORHOOD_TAPS];
            for tap in &mut neighborhood {
                *tap = [
                    lcg_unit(&mut state),
                    lcg_unit(&mut state),
                    lcg_unit(&mut state),
                    1.0,
                ];
            }

            queries.push(TemporalReprojectQuery {
                current_uv,
                motion_uv,
                current_depth: depth,
                history_depth: depth,
                current,
                history,
                neighborhood,
            });
        }

        check(&ctx, &gpu, &params, &queries);
    }
}

#[test]
fn degenerate_inputs_do_not_panic() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalReproject::new(&ctx);
    let params = base_params();

    // Degenerate parameters: a zero-width screen reject threshold and NaN-laden
    // inputs. The twin must not panic and must return exactly one result per
    // pixel; values through NaN channels are implementation-defined, so only the
    // count and the no-panic guarantee are asserted here.
    let nan = f32::NAN;
    let queries = [
        // NaN motion drives the reprojected UV to NaN (off-screen) — a guard
        // path, not a crash.
        TemporalReprojectQuery {
            current_uv: Vec2::new(0.5, 0.5),
            motion_uv: Vec2::new(nan, 0.0),
            current_depth: 10.0,
            history_depth: 10.0,
            current: [0.4, 0.4, 0.4, 1.0],
            history: [0.6, 0.6, 0.6, 1.0],
            neighborhood: flat_neighborhood([0.5, 0.5, 0.5, 1.0]),
        },
        // NaN depth exercises the depth-denominator guard.
        TemporalReprojectQuery {
            current_uv: Vec2::new(0.5, 0.5),
            motion_uv: Vec2::new(0.0, 0.0),
            current_depth: nan,
            history_depth: 10.0,
            current: [0.4, 0.4, 0.4, 1.0],
            history: [0.6, 0.6, 0.6, 1.0],
            neighborhood: flat_neighborhood([0.5, 0.5, 0.5, 1.0]),
        },
        // A zero depth both sides exercises the DEPTH_EPS denominator floor.
        TemporalReprojectQuery {
            current_uv: Vec2::new(0.0, 1.0),
            motion_uv: Vec2::new(0.0, 0.0),
            current_depth: 0.0,
            history_depth: 0.0,
            current: [0.2, 0.3, 0.4, 1.0],
            history: [0.4, 0.3, 0.2, 1.0],
            neighborhood: flat_neighborhood([0.5, 0.5, 0.5, 1.0]),
        },
    ];

    let got = gpu.eval(&ctx, &params, &queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "every query yields exactly one result, degenerate or not"
    );

    // The all-finite third pixel still matches the reference exactly.
    let reference_third = reference(&params, &queries[2]);
    for channel in 0..4 {
        assert!(
            (got[2].resolved[channel] - reference_third.resolved[channel]).abs() <= EPS,
            "finite degenerate pixel channel {channel} mismatch: gpu {}, cpu {}",
            got[2].resolved[channel],
            reference_third.resolved[channel]
        );
    }
}
