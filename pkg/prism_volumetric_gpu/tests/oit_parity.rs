//! Real-device parity for the order-independent-transparency twin:
//! [`GpuOit`](prism_volumetric_gpu::oit::GpuOit) must reproduce the `CPU`
//! golden [`oit`](prism_render_architecture::particle::oit) across the
//! composite-method routing, the weighted-blended depth weight and single-pass
//! resolve / `over`, the multiply-only absorbance / transmittance series, the
//! four-power-moment reconstruction (occlusion, optical depth, transmittance)
//! and the soft-particle depth fade, plus a randomized batch compared
//! element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every routine threads through multiplies, adds and one guarded divide /
//! `sqrt`, so `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. Continuous channels are
//! compared with `abs_diff <= 1e-4` or `rel_diff <= 1e-3`. The composite-method
//! classification is an enum selector, so it is compared for exact equality.
//!
//! # Conditioning
//!
//! Fixtures are drawn clear of every branch knee: alpha is rejection-sampled
//! into `[0.05, 0.95]` away from `0` / `1`; `near < far` with a positive gap;
//! moment sequences are built from a spread of distinct weighted depths so the
//! Hankel system stays non-singular and `normalized` returns `Some`; the bias
//! stays strictly positive; and the soft-fade sample is kept in the smoothstep
//! interior so a fused multiply-add cannot tip the `clamp01` knee. All
//! randomness comes from a host-side integer generator, so no transcendental
//! appears in a fixture.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::oit`；no
//! third-party engine source or derived code.

use prism_render_architecture::particle::oit::{
    approx_absorbance, approx_transmittance, composite_method, moment_occlusion,
    moment_transmittance, reconstruct_optical_depth, soft_particle_fade, wboit_weight,
    MomentAccumulator, OitQuality, PowerMoments, ResolvedTransparency, WboitAccumulator, OIT_EPS,
};
use prism_render_architecture::particle::renderers::ParticleBlend;
use prism_render_architecture::particle::{SortStrategy, Vec3};
use prism_volumetric_gpu::oit::{GpuOit, OitQuery, OitResult};
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

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Clamps a scalar to the closed unit interval, mirroring the reference
/// `clamp01`.
fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
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

/// A pseudo-random value in `[lo, hi)` drawn from `state`.
fn range(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + lcg(state) * (hi - lo)
}

/// A straight alpha rejection-sampled into `[0.05, 0.95]`, away from the `0` /
/// `1` clamp knees.
fn rand_alpha(state: &mut u64) -> f32 {
    loop {
        let a = lcg(state);
        if (0.05..=0.95).contains(&a) {
            return a;
        }
    }
}

/// A pseudo-random `RGB` triple with each channel in `[0, span)`.
fn rand_rgb(state: &mut u64, span: f32) -> [f32; 3] {
    [lcg(state) * span, lcg(state) * span, lcg(state) * span]
}

/// Builds a [`Vec3`] from a component triple.
fn v3(a: [f32; 3]) -> Vec3 {
    Vec3::new(a[0], a[1], a[2])
}

/// Every sort strategy, in classification-code order.
const STRATEGIES: [SortStrategy; 4] = [
    SortStrategy::None,
    SortStrategy::SharedOit,
    SortStrategy::ViewDepthRadix,
    SortStrategy::ViewDepthBitonic,
];

/// Every particle blend mode, in classification-code order.
const BLENDS: [ParticleBlend; 5] = [
    ParticleBlend::Opaque,
    ParticleBlend::AlphaMask,
    ParticleBlend::Additive,
    ParticleBlend::Premultiplied,
    ParticleBlend::AlphaBlend,
];

/// Every quality tier, in classification-code order.
const TIERS: [OitQuality; 3] = [
    OitQuality::Fast,
    OitQuality::Balanced,
    OitQuality::Reference,
];

/// The reduced weighted-blended sums `(color_weight, alpha_weight, revealage)`
/// a host accumulation hands the twin, replayed with the exact arithmetic and
/// order of [`WboitAccumulator::accumulate`] so the inputs equal the golden
/// accumulator's private fields.
fn wboit_sums(frags: &[(Vec3, f32, f32)]) -> (Vec3, f32, f32) {
    let mut color_weight = Vec3::ZERO;
    let mut alpha_weight = 0.0_f32;
    let mut revealage = 1.0_f32;
    for &(color, alpha, weight) in frags {
        let a = clamp01(alpha);
        let w = weight.max(0.0);
        color_weight = color_weight.add(color.scale(a * w));
        alpha_weight += a * w;
        revealage *= 1.0 - a;
    }
    (color_weight, alpha_weight, revealage)
}

/// The golden [`ResolvedTransparency`] for the reduced weighted-blended sums,
/// built through the real [`WboitAccumulator`] so the comparison is against the
/// reference resolve, not a re-derivation.
fn golden_resolve(frags: &[(Vec3, f32, f32)]) -> ResolvedTransparency {
    let mut acc = WboitAccumulator::new();
    for &(color, alpha, weight) in frags {
        acc.accumulate(color, alpha, weight);
    }
    acc.resolve()
}

/// A random set of weighted-blended fragments with seam-safe alphas and
/// positive weights.
fn rand_wboit_frags(state: &mut u64) -> Vec<(Vec3, f32, f32)> {
    let count = 2 + (lcg(state) * 4.0) as usize;
    (0..count)
        .map(|_| {
            (
                v3(rand_rgb(state, 1.0)),
                rand_alpha(state),
                range(state, 0.1, 3.0),
            )
        })
        .collect()
}

/// Builds an [`OitQuery::WboitResolve`] from the reduced sums of random
/// fragments, returning the query and the golden resolve it must match.
fn make_wboit_resolve(state: &mut u64) -> (OitQuery, ResolvedTransparency) {
    let frags = rand_wboit_frags(state);
    let (cw, aw, rev) = wboit_sums(&frags);
    let query = OitQuery::WboitResolve {
        color_weight: [cw.x, cw.y, cw.z],
        alpha_weight: aw,
        revealage: rev,
    };
    (query, golden_resolve(&frags))
}

/// The reduced moment sums a host accumulation hands the twin, replayed with
/// the exact arithmetic and order of [`MomentAccumulator::accumulate`] (which
/// weights each depth power by [`approx_absorbance`]).
fn moment_sums(frags: &[(f32, f32)]) -> [f32; 5] {
    let mut total = 0.0_f32;
    let mut m1 = 0.0_f32;
    let mut m2 = 0.0_f32;
    let mut m3 = 0.0_f32;
    let mut m4 = 0.0_f32;
    for &(depth, alpha) in frags {
        let a = approx_absorbance(alpha);
        let d = clamp01(depth);
        let d2 = d * d;
        let d3 = d2 * d;
        let d4 = d3 * d;
        total += a;
        m1 += a * d;
        m2 += a * d2;
        m3 += a * d3;
        m4 += a * d4;
    }
    [total, m1, m2, m3, m4]
}

/// The golden normalized [`PowerMoments`] for a fragment set, built through the
/// real [`MomentAccumulator`].
fn golden_normalized(frags: &[(f32, f32)]) -> Option<PowerMoments> {
    let mut acc = MomentAccumulator::new();
    for &(depth, alpha) in frags {
        acc.accumulate(depth, alpha);
    }
    acc.normalized()
}

/// A random fragment set spread across distinct depths with seam-safe alphas,
/// so the accumulated absorbance is well above the `None` threshold.
fn rand_moment_frags(state: &mut u64) -> Vec<(f32, f32)> {
    let count = 4 + (lcg(state) * 3.0) as usize;
    (0..count)
        .map(|_| (range(state, 0.1, 0.9), rand_alpha(state)))
        .collect()
}

/// Builds an [`OitQuery::MomentsNormalized`] from the reduced moment sums of a
/// random fragment set, returning the query and the golden normalization.
fn make_moments_normalized(state: &mut u64) -> (OitQuery, Option<PowerMoments>) {
    let frags = rand_moment_frags(state);
    let s = moment_sums(&frags);
    let query = OitQuery::MomentsNormalized {
        total: s[0],
        m1: s[1],
        m2: s[2],
        m3: s[3],
        m4: s[4],
    };
    (query, golden_normalized(&frags))
}

/// A well-conditioned normalized moment sequence `[E[d], E[d^2], E[d^3],
/// E[d^4]]` built from a spread of distinct weighted depths, together with the
/// positive total weight used as the zeroth moment for the optical-depth
/// reconstruction.
fn rand_moments(state: &mut u64) -> ([f32; 4], f32) {
    let count = 5;
    let mut sw = 0.0_f32;
    let mut s1 = 0.0_f32;
    let mut s2 = 0.0_f32;
    let mut s3 = 0.0_f32;
    let mut s4 = 0.0_f32;
    for _ in 0..count {
        let d = range(state, 0.1, 0.9);
        let w = range(state, 0.3, 1.3);
        let d2 = d * d;
        sw += w;
        s1 += w * d;
        s2 += w * d2;
        s3 += w * d2 * d;
        s4 += w * d2 * d2;
    }
    let inv = 1.0 / sw;
    ([s1 * inv, s2 * inv, s3 * inv, s4 * inv], sw)
}

/// A conditioning bias kept strictly positive and small, away from the `0`
/// knee of the singular-fallback branch.
fn rand_bias(state: &mut u64) -> f32 {
    range(state, 1.0e-3, 1.0e-2)
}

/// Builds an [`OitQuery::SoftParticleFade`] whose depth delta lands in the
/// smoothstep interior, away from the `clamp01` knees.
fn make_soft_fade(state: &mut u64) -> OitQuery {
    let scene_depth = range(state, 5.0, 15.0);
    let contrast = range(state, 0.5, 3.0);
    let t = range(state, 0.1, 0.9);
    let particle_depth = scene_depth - t * contrast;
    OitQuery::SoftParticleFade {
        scene_depth,
        particle_depth,
        contrast,
    }
}

/// Pins a vector output against a reference [`Vec3`] within tolerance.
fn close_vec(idx: usize, got: &[f32; 3], want: Vec3) {
    let w = [want.x, want.y, want.z];
    for (lane, (g, c)) in got.iter().zip(w.iter()).enumerate() {
        assert!(close(*g, *c), "query {idx} lane {lane}: gpu {g} vs cpu {c}");
    }
}

/// Pins two optional [`PowerMoments`] for equal presence and, when present,
/// close total and lanes (its fields are `f32`, so no bare equality is used).
fn close_moments(idx: usize, got: &Option<PowerMoments>, want: &Option<PowerMoments>) {
    match (got, want) {
        (None, None) => {}
        (Some(g), Some(c)) => {
            assert!(
                close(g.total, c.total),
                "query {idx} total: gpu {} vs cpu {}",
                g.total,
                c.total
            );
            for (lane, (gb, cb)) in g.b.iter().zip(c.b.iter()).enumerate() {
                assert!(close(*gb, *cb), "query {idx} b{lane}: gpu {gb} vs cpu {cb}");
            }
        }
        _ => panic!("query {idx}: moment presence differs (gpu {got:?} vs cpu {want:?})"),
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`, matching the
/// result variant to the query variant and comparing channel-for-channel. The
/// two accumulation variants carry the host-reduced sums, so their golden is
/// the reference closed form replayed on those same sums.
#[expect(
    clippy::too_many_lines,
    reason = "one match arm per OIT routine keeps the full dispatch table in one readable place"
)]
fn pin(idx: usize, query: &OitQuery, got: &OitResult) {
    match (query, got) {
        (
            OitQuery::CompositeMethod {
                strategy,
                blend,
                tier,
            },
            OitResult::CompositeMethod(got_m),
        ) => {
            let want = composite_method(*strategy, *blend, *tier);
            assert_eq!(*got_m, want, "query {idx} composite method");
        }
        (
            OitQuery::WboitWeight {
                view_depth,
                alpha,
                near,
                far,
            },
            OitResult::WboitWeight(got_w),
        ) => {
            let want = wboit_weight(*view_depth, *alpha, *near, *far);
            assert!(
                close(*got_w, want),
                "query {idx} wboit weight: gpu {got_w} vs cpu {want}"
            );
        }
        (
            OitQuery::Over {
                color,
                coverage,
                background,
            },
            OitResult::Over(got_c),
        ) => {
            let resolved = ResolvedTransparency {
                color: v3(*color),
                coverage: *coverage,
            };
            close_vec(idx, got_c, resolved.over(v3(*background)));
        }
        (
            OitQuery::WboitResolve {
                color_weight,
                alpha_weight,
                revealage,
            },
            OitResult::WboitResolve { color, coverage },
        ) => {
            // Replay the reference resolve on the host-reduced sums.
            let average = v3(*color_weight).scale(1.0 / alpha_weight.max(OIT_EPS));
            let cov = 1.0 - clamp01(*revealage);
            close_vec(idx, color, average.scale(cov));
            assert!(
                close(*coverage, cov),
                "query {idx} coverage: gpu {coverage} vs cpu {cov}"
            );
        }
        (OitQuery::WboitRevealage { revealage }, OitResult::WboitRevealage(got_r)) => {
            let want = clamp01(*revealage);
            assert!(
                close(*got_r, want),
                "query {idx} revealage: gpu {got_r} vs cpu {want}"
            );
        }
        (OitQuery::ApproxAbsorbance { alpha }, OitResult::ApproxAbsorbance(got_a)) => {
            let want = approx_absorbance(*alpha);
            assert!(
                close(*got_a, want),
                "query {idx} absorbance: gpu {got_a} vs cpu {want}"
            );
        }
        (
            OitQuery::ApproxTransmittance { optical_depth },
            OitResult::ApproxTransmittance(got_t),
        ) => {
            let want = approx_transmittance(*optical_depth);
            assert!(
                close(*got_t, want),
                "query {idx} transmittance: gpu {got_t} vs cpu {want}"
            );
        }
        (
            OitQuery::MomentsNormalized {
                total,
                m1,
                m2,
                m3,
                m4,
            },
            OitResult::MomentsNormalized(got_pm),
        ) => {
            // Replay the reference normalization on the host-reduced sums.
            let want = if *total < OIT_EPS {
                None
            } else {
                let inv = 1.0 / total;
                Some(PowerMoments {
                    total: *total,
                    b: [m1 * inv, m2 * inv, m3 * inv, m4 * inv],
                })
            };
            close_moments(idx, got_pm, &want);
        }
        (
            OitQuery::MomentOcclusion {
                moments,
                depth,
                bias,
            },
            OitResult::MomentOcclusion(got_o),
        ) => {
            let want = moment_occlusion(*moments, *depth, *bias);
            assert!(
                close(*got_o, want),
                "query {idx} occlusion: gpu {got_o} vs cpu {want}"
            );
        }
        (
            OitQuery::ReconstructOpticalDepth {
                total,
                moments,
                depth,
                bias,
            },
            OitResult::ReconstructOpticalDepth(got_d),
        ) => {
            let pm = PowerMoments {
                total: *total,
                b: *moments,
            };
            let want = reconstruct_optical_depth(pm, *depth, *bias);
            assert!(
                close(*got_d, want),
                "query {idx} optical depth: gpu {got_d} vs cpu {want}"
            );
        }
        (
            OitQuery::MomentTransmittance {
                total,
                moments,
                depth,
                bias,
            },
            OitResult::MomentTransmittance(got_t),
        ) => {
            let pm = PowerMoments {
                total: *total,
                b: *moments,
            };
            let want = moment_transmittance(pm, *depth, *bias);
            assert!(
                close(*got_t, want),
                "query {idx} moment transmittance: gpu {got_t} vs cpu {want}"
            );
        }
        (
            OitQuery::SoftParticleFade {
                scene_depth,
                particle_depth,
                contrast,
            },
            OitResult::SoftParticleFade(got_f),
        ) => {
            let want = soft_particle_fade(*scene_depth, *particle_depth, *contrast);
            assert!(
                close(*got_f, want),
                "query {idx} soft fade: gpu {got_f} vs cpu {want}"
            );
        }
        _ => panic!("query {idx}: result variant does not match the query variant"),
    }
}

/// Draws a random query of a random routine with seam-safe fixtures.
fn rand_query(state: &mut u64) -> OitQuery {
    match (lcg(state) * 12.0) as u32 {
        0 => OitQuery::CompositeMethod {
            strategy: STRATEGIES[(lcg(state) * 4.0) as usize],
            blend: BLENDS[(lcg(state) * 5.0) as usize],
            tier: TIERS[(lcg(state) * 3.0) as usize],
        },
        1 => {
            let near = range(state, 0.1, 2.0);
            let far = near + range(state, 0.5, 8.0);
            OitQuery::WboitWeight {
                view_depth: range(state, 0.0, far * 1.5),
                alpha: rand_alpha(state),
                near,
                far,
            }
        }
        2 => OitQuery::Over {
            color: rand_rgb(state, 1.0),
            coverage: range(state, 0.05, 0.95),
            background: rand_rgb(state, 1.0),
        },
        3 => make_wboit_resolve(state).0,
        4 => OitQuery::WboitRevealage {
            revealage: range(state, 0.05, 0.95),
        },
        5 => OitQuery::ApproxAbsorbance {
            alpha: rand_alpha(state),
        },
        6 => OitQuery::ApproxTransmittance {
            optical_depth: range(state, 0.0, 6.0),
        },
        7 => make_moments_normalized(state).0,
        8 => {
            let (moments, _total) = rand_moments(state);
            OitQuery::MomentOcclusion {
                moments,
                depth: range(state, 0.1, 0.9),
                bias: rand_bias(state),
            }
        }
        9 => {
            let (moments, total) = rand_moments(state);
            OitQuery::ReconstructOpticalDepth {
                total,
                moments,
                depth: range(state, 0.1, 0.9),
                bias: rand_bias(state),
            }
        }
        10 => {
            let (moments, total) = rand_moments(state);
            OitQuery::MomentTransmittance {
                total,
                moments,
                depth: range(state, 0.1, 0.9),
                bias: rand_bias(state),
            }
        }
        _ => make_soft_fade(state),
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuOit, queries: &[OitQuery]) {
    let got = gpu.eval(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, query, result);
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOit::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn composite_method_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOit::new(&ctx);
    // Exhaustive 4 x 5 x 3 classification grid; every strategy / blend / tier
    // combination is pinned for exact method equality.
    let mut queries = Vec::new();
    for &strategy in &STRATEGIES {
        for &blend in &BLENDS {
            for &tier in &TIERS {
                queries.push(OitQuery::CompositeMethod {
                    strategy,
                    blend,
                    tier,
                });
            }
        }
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn wboit_weight_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOit::new(&ctx);
    let mut state = 0x5157_1a2b_3c4d_5e6f_u64;
    let mut queries = Vec::new();
    for _ in 0..64 {
        let near = range(&mut state, 0.1, 2.0);
        let far = near + range(&mut state, 0.5, 8.0);
        queries.push(OitQuery::WboitWeight {
            view_depth: range(&mut state, 0.0, far * 1.5),
            alpha: rand_alpha(&mut state),
            near,
            far,
        });
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn over_and_resolve_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOit::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut queries = Vec::new();
    for _ in 0..48 {
        queries.push(OitQuery::Over {
            color: rand_rgb(&mut state, 1.0),
            coverage: range(&mut state, 0.05, 0.95),
            background: rand_rgb(&mut state, 1.0),
        });
        queries.push(make_wboit_resolve(&mut state).0);
        queries.push(OitQuery::WboitRevealage {
            revealage: range(&mut state, 0.05, 0.95),
        });
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn wboit_resolve_matches_golden_accumulator() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOit::new(&ctx);
    // Cross-check the twin's resolve against the real `WboitAccumulator`: the
    // host feeds the GPU the reduced sums and the expected comes from the
    // accumulator's own `resolve`.
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    for _ in 0..48 {
        let (query, expected) = make_wboit_resolve(&mut state);
        let got = gpu.eval(&ctx, &[query]);
        assert_eq!(got.len(), 1);
        match got[0] {
            OitResult::WboitResolve { color, coverage } => {
                close_vec(0, &color, expected.color);
                assert!(
                    close(coverage, expected.coverage),
                    "coverage: gpu {coverage} vs cpu {}",
                    expected.coverage
                );
            }
            ref other => panic!("unexpected result variant: {other:?}"),
        }
    }
}

#[test]
fn physical_transforms_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOit::new(&ctx);
    let mut state = 0x00c0_ffee_dead_beef_u64;
    let mut queries = Vec::new();
    for _ in 0..64 {
        queries.push(OitQuery::ApproxAbsorbance {
            alpha: rand_alpha(&mut state),
        });
        queries.push(OitQuery::ApproxTransmittance {
            optical_depth: range(&mut state, 0.0, 6.0),
        });
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn moments_normalized_matches_golden_accumulator() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOit::new(&ctx);
    // Cross-check the twin's normalization against the real `MomentAccumulator`:
    // the host feeds the GPU the reduced sums, the expected is the accumulator's
    // `normalized`, and the well-conditioned fixtures guarantee a `Some`.
    let mut state = 0x3141_5926_5358_9793_u64;
    for _ in 0..48 {
        let (query, expected) = make_moments_normalized(&mut state);
        let got = gpu.eval(&ctx, &[query]);
        assert_eq!(got.len(), 1);
        match &got[0] {
            OitResult::MomentsNormalized(pm) => close_moments(0, pm, &expected),
            other => panic!("unexpected result variant: {other:?}"),
        }
    }
}

#[test]
fn moment_reconstruction_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOit::new(&ctx);
    let mut state = 0x2718_2818_2845_9045_u64;
    let mut queries = Vec::new();
    for _ in 0..48 {
        let (moments, total) = rand_moments(&mut state);
        let depth = range(&mut state, 0.1, 0.9);
        let bias = rand_bias(&mut state);
        queries.push(OitQuery::MomentOcclusion {
            moments,
            depth,
            bias,
        });
        queries.push(OitQuery::ReconstructOpticalDepth {
            total,
            moments,
            depth,
            bias,
        });
        queries.push(OitQuery::MomentTransmittance {
            total,
            moments,
            depth,
            bias,
        });
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn soft_particle_fade_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOit::new(&ctx);
    let mut state = 0x5a5a_a5a5_0f0f_f0f0_u64;
    let queries: Vec<OitQuery> = (0..64).map(|_| make_soft_fade(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOit::new(&ctx);
    let mut state = 0x2b2b_1a1a_3c3c_4d4d_u64;
    // One batch mixing deterministic fixtures with many random queries of every
    // routine, dispatched together so the per-thread indexing and the contiguous
    // storage layout are both exercised, then pinned element-for-element.
    let (moments, total) = rand_moments(&mut state);
    let mut queries = vec![
        OitQuery::CompositeMethod {
            strategy: SortStrategy::SharedOit,
            blend: ParticleBlend::AlphaBlend,
            tier: OitQuality::Balanced,
        },
        OitQuery::WboitWeight {
            view_depth: 2.0,
            alpha: 0.5,
            near: 0.5,
            far: 10.0,
        },
        OitQuery::Over {
            color: [0.2, 0.4, 0.6],
            coverage: 0.5,
            background: [0.1, 0.1, 0.1],
        },
        OitQuery::ApproxAbsorbance { alpha: 0.5 },
        OitQuery::ApproxTransmittance { optical_depth: 1.5 },
        OitQuery::MomentOcclusion {
            moments,
            depth: 0.5,
            bias: 3.0e-3,
        },
        OitQuery::ReconstructOpticalDepth {
            total,
            moments,
            depth: 0.5,
            bias: 3.0e-3,
        },
        OitQuery::MomentTransmittance {
            total,
            moments,
            depth: 0.5,
            bias: 3.0e-3,
        },
    ];
    for _ in 0..56 {
        queries.push(rand_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_queries_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOit::new(&ctx);
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    // A larger sweep (several workgroups' worth) pins every routine across many
    // random fixtures.
    let queries: Vec<OitQuery> = (0..256).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
