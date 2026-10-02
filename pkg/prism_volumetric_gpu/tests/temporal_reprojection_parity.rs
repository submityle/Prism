//! Real-device parity for the §21 temporal-reprojection primitive twin:
//! [`GpuTemporalReprojection`](prism_volumetric_gpu::temporal_reprojection::GpuTemporalReprojection)
//! must reproduce the `CPU` golden
//! [`temporal_reprojection`](prism_render_architecture::particle::temporal_reprojection)
//! lane for lane across every operation code the single `solve` kernel
//! dispatches — the [`Vec2`] algebra, the cubic
//! [`smoothstep`](prism_render_architecture::particle::temporal_reprojection::smoothstep),
//! [`reproject_uv`](prism_render_architecture::particle::temporal_reprojection::reproject_uv),
//! [`is_on_screen`](prism_render_architecture::particle::temporal_reprojection::is_on_screen),
//! [`depth_confidence`](prism_render_architecture::particle::temporal_reprojection::depth_confidence),
//! [`velocity_confidence`](prism_render_architecture::particle::temporal_reprojection::velocity_confidence),
//! [`history_valid`](prism_render_architecture::particle::temporal_reprojection::history_valid),
//! [`neighborhood_box`](prism_render_architecture::particle::temporal_reprojection::neighborhood_box),
//! [`AabbRgba::widened`](prism_render_architecture::particle::temporal_reprojection::AabbRgba::widened),
//! [`neighborhood_clamp`](prism_render_architecture::particle::temporal_reprojection::neighborhood_clamp),
//! [`rgb_to_ycocg`](prism_render_architecture::particle::temporal_reprojection::rgb_to_ycocg),
//! [`ycocg_to_rgb`](prism_render_architecture::particle::temporal_reprojection::ycocg_to_rgb),
//! [`clip_history_ycocg`](prism_render_architecture::particle::temporal_reprojection::clip_history_ycocg),
//! [`history_weight`](prism_render_architecture::particle::temporal_reprojection::history_weight)
//! and
//! [`blend_history`](prism_render_architecture::particle::temporal_reprojection::blend_history).
//!
//! The named fixtures cover an empty batch; every `Vec2` operation; both the
//! on-screen and off-screen verdicts, placed clear of the `[0, 1]` boundary; the
//! `smoothstep` endpoints and interior; both the near-match and the hard-reject
//! sides of the depth-disocclusion test, each clear of the threshold; the
//! velocity confidence and its disabled (`reject == 0`) degenerate; a combined
//! history-validity sample and its off-screen rejection; the neighbourhood box,
//! its symmetric widening, and the min/max clamp; the `YCoCg` round trip; the
//! line clip of an out-of-gamut "ghost" history far from the `p ≈ q`
//! degeneracy; the blend weight; the confidence blend; a mixed-variant batch
//! that pins input ordering; and a large pseudo-random batch over the
//! tie-free operations compared lane for lane.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every twinned function is fixed closed-form algebra (guarded divisions, one
//! `sqrt` for the vector length, and the hand-expanded cubic `smoothstep`), so
//! `CPU` and `GPU` evaluate the same expression in the same associativity. They
//! are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits by a few units in the last
//! place. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` on continuous values and asserts an *exact* match on the
//! discrete on-screen boolean. Every fixture is placed clear of the branch-
//! critical thresholds — the depth and velocity rejection cutoffs, the on-screen
//! rectangle edge, and the `clip_toward` `p ≈ q` degeneracy — so no legal `ULP`
//! perturbation can flip a verdict.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::temporal_reprojection`；
//! standard `TAA` / temporal-accumulation primitives (reproject, disocclusion /
//! velocity confidence, `YCoCg` neighbourhood clip, confidence blend); no
//! third-party engine source or derived code.

use prism_render_architecture::particle::temporal_reprojection::{
    AabbRgba, ReprojectionParams, Rgba, Vec2,
};
use prism_volumetric_gpu::temporal_reprojection::{
    cpu_reference, GpuTemporalReprojection, TemporalReprojectionQuery, TemporalReprojectionResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on continuous values. A `GPU` may fuse a multiply-add
/// the scalar reference leaves separate, perturbing the low mantissa bits by a
/// few units in the last place; `1e-4` admits that legal slack while still
/// failing a genuinely wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Shorthand for a two-component vector.
fn v2(x: f32, y: f32) -> Vec2 {
    Vec2::new(x, y)
}

/// A standard validity-clamped parameter block used by the confidence, clamp and
/// blend fixtures: a `0.9` steady-state weight, a `0.1` relative-depth cutoff, a
/// `0.5` velocity cutoff and a `0.25` box widening. Every fixture keeps its
/// operands clear of these thresholds.
fn params() -> ReprojectionParams {
    ReprojectionParams::new(0.9, 0.1, 0.5, 0.25)
}

/// Builds a flat `3x3` neighbourhood of one colour.
fn flat(colour: Rgba) -> [Rgba; 9] {
    [colour; 9]
}

/// Asserts strict lane-for-lane parity of one `GPU` result against the `CPU`
/// golden: the on-screen boolean matches exactly, and every continuous value
/// matches within tolerance.
fn assert_lane(lane: usize, got: &TemporalReprojectionResult, want: &TemporalReprojectionResult) {
    match (got, want) {
        (
            TemporalReprojectionResult::Vec2 { v: vg },
            TemporalReprojectionResult::Vec2 { v: vc },
        ) => {
            assert!(
                close(vg[0], vc[0]),
                "lane {lane}: vec2.x gpu {vg:?} vs cpu {vc:?}"
            );
            assert!(
                close(vg[1], vc[1]),
                "lane {lane}: vec2.y gpu {vg:?} vs cpu {vc:?}"
            );
        }
        (
            TemporalReprojectionResult::Scalar { value: sg },
            TemporalReprojectionResult::Scalar { value: sc },
        ) => {
            assert!(close(*sg, *sc), "lane {lane}: scalar gpu {sg} vs cpu {sc}");
        }
        (
            TemporalReprojectionResult::OnScreen { on: og },
            TemporalReprojectionResult::OnScreen { on: oc },
        ) => {
            assert_eq!(og, oc, "lane {lane}: on-screen gpu {og} vs cpu {oc}");
        }
        (
            TemporalReprojectionResult::Ycocg { ycocg: yg },
            TemporalReprojectionResult::Ycocg { ycocg: yc },
        ) => {
            for ch in 0..3 {
                assert!(
                    close(yg[ch], yc[ch]),
                    "lane {lane}: ycocg[{ch}] gpu {yg:?} vs cpu {yc:?}"
                );
            }
        }
        (
            TemporalReprojectionResult::Color { color: cg },
            TemporalReprojectionResult::Color { color: cc },
        ) => {
            for ch in 0..4 {
                assert!(
                    close(cg[ch], cc[ch]),
                    "lane {lane}: color[{ch}] gpu {cg:?} vs cpu {cc:?}"
                );
            }
        }
        (
            TemporalReprojectionResult::Aabb {
                min: ming,
                max: maxg,
            },
            TemporalReprojectionResult::Aabb {
                min: minc,
                max: maxc,
            },
        ) => {
            for ch in 0..4 {
                assert!(
                    close(ming[ch], minc[ch]),
                    "lane {lane}: aabb.min[{ch}] gpu {ming:?} vs cpu {minc:?}"
                );
                assert!(
                    close(maxg[ch], maxc[ch]),
                    "lane {lane}: aabb.max[{ch}] gpu {maxg:?} vs cpu {maxc:?}"
                );
            }
        }
        _ => panic!("lane {lane}: result variant mismatch gpu {got:?} vs cpu {want:?}"),
    }
}

/// Runs the `GPU` dispatch and asserts strict parity against the `CPU` golden
/// for every lane; returns the `GPU` verdicts for extra per-test assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuTemporalReprojection,
    queries: &[TemporalReprojectionQuery],
) -> Vec<TemporalReprojectionResult> {
    let got = gpu.eval(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        assert_lane(lane, g, &cpu_reference(q));
    }
    got
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

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalReprojection::new(&ctx);
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn vec2_ops_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalReprojection::new(&ctx);
    let queries = [
        TemporalReprojectionQuery::Vec2Minus {
            a: v2(0.7, 0.3),
            b: v2(0.2, 0.1),
        },
        TemporalReprojectionQuery::Vec2Scale {
            v: v2(0.3, 0.4),
            s: 2.0,
        },
        TemporalReprojectionQuery::Vec2LengthSquared { v: v2(0.3, 0.4) },
        TemporalReprojectionQuery::Vec2Length { v: v2(0.3, 0.4) },
        // Length clearly above the normalize epsilon, so the direction branch is
        // taken unambiguously.
        TemporalReprojectionQuery::Vec2Normalize { v: v2(3.0, 4.0) },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn reproject_subtracts_motion() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalReprojection::new(&ctx);
    let q = TemporalReprojectionQuery::Reproject {
        current_uv: v2(0.5, 0.5),
        motion_uv: v2(0.1, 0.05),
    };
    check(&ctx, &gpu, &[q]);
}

#[test]
fn on_screen_classifies_clear_of_edges() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalReprojection::new(&ctx);
    // Interior sample is on-screen; the others sit well outside the rectangle, so
    // no edge tie can flip the boolean.
    let queries = [
        TemporalReprojectionQuery::OnScreen { uv: v2(0.5, 0.5) },
        TemporalReprojectionQuery::OnScreen { uv: v2(1.3, 0.4) },
        TemporalReprojectionQuery::OnScreen { uv: v2(0.4, -0.3) },
    ];
    let got = check(&ctx, &gpu, &queries);
    assert!(matches!(
        got[0],
        TemporalReprojectionResult::OnScreen { on: true }
    ));
    assert!(matches!(
        got[1],
        TemporalReprojectionResult::OnScreen { on: false }
    ));
}

#[test]
fn smoothstep_endpoints_and_interior() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalReprojection::new(&ctx);
    let queries = [
        TemporalReprojectionQuery::Smoothstep { t: -1.0 },
        TemporalReprojectionQuery::Smoothstep { t: 0.3 },
        TemporalReprojectionQuery::Smoothstep { t: 0.75 },
        TemporalReprojectionQuery::Smoothstep { t: 2.0 },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn depth_confidence_near_match_and_hard_reject() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalReprojection::new(&ctx);
    // rel = 0.02, threshold 0.1: comfortably below the cutoff (soft falloff).
    // rel = 0.5, threshold 0.1: comfortably above the cutoff (hard zero).
    let queries = [
        TemporalReprojectionQuery::DepthConfidence {
            current_depth: 1.0,
            history_depth: 1.02,
            reject_relative: 0.1,
        },
        TemporalReprojectionQuery::DepthConfidence {
            current_depth: 1.0,
            history_depth: 1.5,
            reject_relative: 0.1,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn velocity_confidence_active_and_disabled() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalReprojection::new(&ctx);
    let queries = [
        // speed 0.1, limit 0.5: soft falloff, clear of the disable branch.
        TemporalReprojectionQuery::VelocityConfidence {
            motion_uv: v2(0.1, 0.0),
            reject_uv: 0.5,
        },
        // reject exactly zero disables the penalty: full confidence.
        TemporalReprojectionQuery::VelocityConfidence {
            motion_uv: v2(0.3, 0.4),
            reject_uv: 0.0,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn history_valid_sample_and_off_screen() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalReprojection::new(&ctx);
    let queries = [
        // On-screen, near-depth-match, slow motion: a healthy soft confidence.
        TemporalReprojectionQuery::HistoryValid {
            history_uv: v2(0.5, 0.5),
            current_depth: 1.0,
            history_depth: 1.02,
            motion_uv: v2(0.05, 0.0),
            params: params(),
        },
        // Off-screen reprojection: hard rejection to zero.
        TemporalReprojectionQuery::HistoryValid {
            history_uv: v2(1.4, 0.5),
            current_depth: 1.0,
            history_depth: 1.0,
            motion_uv: v2(0.0, 0.0),
            params: params(),
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn neighborhood_box_bounds_window() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalReprojection::new(&ctx);
    let mut samples = flat([0.5, 0.5, 0.5, 1.0]);
    samples[0] = [0.2, 0.6, 0.3, 1.0];
    samples[8] = [0.7, 0.1, 0.9, 1.0];
    let q = TemporalReprojectionQuery::NeighborhoodBox { samples };
    check(&ctx, &gpu, &[q]);
}

#[test]
fn widened_box_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalReprojection::new(&ctx);
    let q = TemporalReprojectionQuery::Widened {
        min: [0.2, 0.2, 0.2, 1.0],
        max: [0.6, 0.6, 0.6, 1.0],
        extra: 0.5,
    };
    check(&ctx, &gpu, &[q]);
}

#[test]
fn neighborhood_clamp_pulls_history_in() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalReprojection::new(&ctx);
    let mut samples = flat([0.5, 0.5, 0.5, 1.0]);
    samples[0] = [0.3, 0.3, 0.3, 1.0];
    samples[8] = [0.7, 0.7, 0.7, 1.0];
    // History well outside the box on the red channel so the clamp bites.
    let q = TemporalReprojectionQuery::NeighborhoodClamp {
        history: [1.5, 0.5, 0.5, 1.0],
        samples,
        params: params(),
    };
    check(&ctx, &gpu, &[q]);
}

#[test]
fn ycocg_round_trip_components() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalReprojection::new(&ctx);
    let queries = [
        TemporalReprojectionQuery::RgbToYcocg {
            color: [0.2, 0.7, 0.4, 0.8],
        },
        TemporalReprojectionQuery::YcocgToRgb {
            ycocg: [0.5, 0.1, -0.2],
            alpha: 0.8,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn clip_history_pulls_ghost_in() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalReprojection::new(&ctx);
    let mut samples = flat([0.5, 0.5, 0.5, 1.0]);
    samples[0] = [0.3, 0.3, 0.3, 1.0];
    samples[8] = [0.7, 0.7, 0.7, 1.0];
    // A far out-of-gamut "ghost": p is nowhere near q, so every clip axis has a
    // delta well beyond the degenerate band and the clip is unambiguous.
    let q = TemporalReprojectionQuery::ClipHistory {
        current: [0.5, 0.5, 0.5, 1.0],
        history: [2.0, 0.0, 0.0, 1.0],
        samples,
        params: params(),
    };
    check(&ctx, &gpu, &[q]);
}

#[test]
fn history_weight_scales_confidence() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalReprojection::new(&ctx);
    let q = TemporalReprojectionQuery::HistoryWeight {
        confidence: 0.5,
        params: params(),
    };
    check(&ctx, &gpu, &[q]);
}

#[test]
fn blend_mixes_by_confidence() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalReprojection::new(&ctx);
    let q = TemporalReprojectionQuery::Blend {
        current: [1.0, 0.0, 0.0, 1.0],
        clamped_history: [0.0, 0.0, 1.0, 1.0],
        confidence: 0.5,
        params: params(),
    };
    check(&ctx, &gpu, &[q]);
}

#[test]
fn mixed_variant_batch_pins_ordering() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalReprojection::new(&ctx);
    let mut samples = flat([0.5, 0.5, 0.5, 1.0]);
    samples[2] = [0.35, 0.45, 0.55, 1.0];
    samples[6] = [0.65, 0.55, 0.45, 1.0];
    let queries = [
        TemporalReprojectionQuery::Vec2Minus {
            a: v2(0.9, 0.2),
            b: v2(0.4, 0.1),
        },
        TemporalReprojectionQuery::Smoothstep { t: 0.6 },
        TemporalReprojectionQuery::OnScreen { uv: v2(0.3, 0.7) },
        TemporalReprojectionQuery::RgbToYcocg {
            color: [0.4, 0.6, 0.2, 1.0],
        },
        TemporalReprojectionQuery::NeighborhoodBox { samples },
        TemporalReprojectionQuery::Blend {
            current: [0.3, 0.6, 0.9, 1.0],
            clamped_history: [0.1, 0.2, 0.3, 1.0],
            confidence: 0.7,
            params: params(),
        },
        TemporalReprojectionQuery::HistoryWeight {
            confidence: 0.4,
            params: params(),
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_batch_matches_lane_for_lane() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalReprojection::new(&ctx);
    let mut state: u64 = 0x5eed_1234_abcd_9876;
    let mut queries: Vec<TemporalReprojectionQuery> = Vec::new();
    // Only tie-free operations appear here: the discrete and threshold-bearing
    // ops (on-screen, depth / velocity / history confidence, the line clip) are
    // pinned by their own clear-of-boundary fixtures instead.
    let kinds = 9u32;
    for _ in 0..256 {
        let pick = (lcg(&mut state) * kinds as f32) as u32;
        let signed = |s: &mut u64| lcg(s) * 2.0 - 1.0;
        let q = match pick.min(kinds - 1) {
            0 => TemporalReprojectionQuery::Vec2Minus {
                a: v2(signed(&mut state), signed(&mut state)),
                b: v2(signed(&mut state), signed(&mut state)),
            },
            1 => TemporalReprojectionQuery::Vec2Scale {
                v: v2(signed(&mut state), signed(&mut state)),
                s: signed(&mut state) * 3.0,
            },
            2 => TemporalReprojectionQuery::Vec2LengthSquared {
                v: v2(signed(&mut state), signed(&mut state)),
            },
            3 => {
                // Keep the length comfortably above the normalize epsilon.
                let x = 0.3 + lcg(&mut state);
                let y = signed(&mut state);
                TemporalReprojectionQuery::Vec2Length { v: v2(x, y) }
            }
            4 => {
                let x = 0.3 + lcg(&mut state);
                let y = signed(&mut state);
                TemporalReprojectionQuery::Vec2Normalize { v: v2(x, y) }
            }
            5 => TemporalReprojectionQuery::Reproject {
                current_uv: v2(lcg(&mut state), lcg(&mut state)),
                motion_uv: v2(signed(&mut state) * 0.2, signed(&mut state) * 0.2),
            },
            6 => {
                // Interior t keeps clear of the smoothstep clamp corners.
                let t = 0.05 + lcg(&mut state) * 0.9;
                TemporalReprojectionQuery::Smoothstep { t }
            }
            7 => TemporalReprojectionQuery::RgbToYcocg {
                color: [
                    lcg(&mut state),
                    lcg(&mut state),
                    lcg(&mut state),
                    lcg(&mut state),
                ],
            },
            _ => {
                // Interior confidence keeps clear of the weight clamp corners.
                let confidence = 0.05 + lcg(&mut state) * 0.9;
                TemporalReprojectionQuery::Blend {
                    current: [lcg(&mut state), lcg(&mut state), lcg(&mut state), 1.0],
                    clamped_history: [lcg(&mut state), lcg(&mut state), lcg(&mut state), 1.0],
                    confidence,
                    params: params(),
                }
            }
        };
        queries.push(q);
    }
    check(&ctx, &gpu, &queries);
}

/// Exercises [`AabbRgba`] widening on the host so the imported golden type is
/// used directly, mirroring the kernel's widening lane.
#[test]
fn widened_host_matches_golden_type() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalReprojection::new(&ctx);
    let aabb = AabbRgba {
        min: [0.1, 0.2, 0.3, 1.0],
        max: [0.5, 0.6, 0.7, 1.0],
    };
    let widened = aabb.widened(0.5);
    let q = TemporalReprojectionQuery::Widened {
        min: aabb.min,
        max: aabb.max,
        extra: 0.5,
    };
    let got = check(&ctx, &gpu, &[q]);
    if let TemporalReprojectionResult::Aabb { min, max } = got[0] {
        for ch in 0..4 {
            assert!(close(min[ch], widened.min[ch]), "widened.min[{ch}]");
            assert!(close(max[ch], widened.max[ch]), "widened.max[{ch}]");
        }
    } else {
        panic!("expected an aabb result");
    }
}
