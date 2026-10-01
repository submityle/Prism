//! Real-device parity for the specular anti-aliasing (`Toksvig` / `Frostbite`)
//! twin:
//! [`GpuSpecularAa`](prism_volumetric_gpu::specular_aa::GpuSpecularAa) must
//! reproduce the `CPU` golden
//! [`specular_aa`](prism_render_architecture::particle::specular_aa) across both
//! of its entry points.
//!
//! `evaluate_main` reproduces
//! [`SpecularAaParams::evaluate`](prism_render_architecture::particle::specular_aa::SpecularAaParams::evaluate)
//! for a batch of average normal lengths under several parameter sets, and
//! `scalars_main` pins each standalone closed-form function
//! ([`toksvig_factor`](prism_render_architecture::particle::specular_aa::toksvig_factor),
//! [`toksvig_effective_gloss`](prism_render_architecture::particle::specular_aa::toksvig_effective_gloss),
//! [`normal_length_variance`](prism_render_architecture::particle::specular_aa::normal_length_variance),
//! [`perceptual_to_linear_roughness`](prism_render_architecture::particle::specular_aa::perceptual_to_linear_roughness),
//! [`linear_to_perceptual_roughness`](prism_render_architecture::particle::specular_aa::linear_to_perceptual_roughness)
//! and
//! [`frostbite_specular_aa`](prism_render_architecture::particle::specular_aa::frostbite_specular_aa))
//! on free per-sample inputs. The vector reductions
//! [`average_normal_length`](prism_render_architecture::particle::specular_aa::average_normal_length)
//! and
//! [`mip_average_normal_lengths`](prism_render_architecture::particle::specular_aa::mip_average_normal_lengths)
//! stay on the `CPU` and are not twinned here.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernels are portable core-`WGSL`, so they need no optional device
//! feature.
//!
//! # Parity criterion
//!
//! Each function is a fixed, non-reorderable sequence of clamps, at most one
//! divide and at most one `sqrt`, so `CPU` and `GPU` evaluate the same
//! closed-form algebra. They are not bit-exact: a `GPU` may fuse a multiply-add
//! the scalar reference leaves separate, and the `sqrt` plus the reciprocal in
//! the ratios carry a few units in the last place. The comparison therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` — loose enough to admit a
//! legal fused multiply-add contraction, yet tight enough to fail a genuinely
//! wrong port (a dropped clamp, a swapped `Toksvig` denominator, a missing
//! `kappa` cap).
//!
//! # Degenerate zones avoided
//!
//! Fixtures keep `avg_normal_length` inside roughly `[0.05, 1.0]` (never toward
//! `0` nor pinned at `MIN_LEN`), keep roughness strictly interior to `[0, 1]`
//! (never on a clamp tie), keep `variance` and `kappa` clearly positive, and
//! keep `base_sq + kernel` away from an exact `1.0` saturation, so no comparison
//! straddles a clamp boundary where `CPU` and `GPU` could legally round to
//! opposite sides.
//!
//! Provenance: standard `Toksvig` specular anti-aliasing and `Frostbite`
//! geometric specular anti-aliasing (design sections 16-17); no third-party
//! engine source or derived code.
#![forbid(unsafe_code)]

use prism_render_architecture::particle::specular_aa::{
    frostbite_specular_aa, linear_to_perceptual_roughness, normal_length_variance,
    perceptual_to_linear_roughness, toksvig_effective_gloss, toksvig_factor, SpecularAaParams,
};
use prism_volumetric_gpu::specular_aa::{
    GpuSpecularAa, SpecularAaBatchQuery, SpecularAaScalarQuery, SpecularAaScalarSample,
};
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

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    // Knuth multiplier / increment; the shift takes the high bits where the
    // generator mixes best.
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    // 24 usable mantissa bits mapped onto [0, 1).
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Draws a value in `[lo, hi)` from `state`.
fn lcg_range(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + lcg(state) * (hi - lo)
}

/// Runs the batch evaluator and asserts every result field matches
/// [`SpecularAaParams::evaluate`] for the same average normal length.
fn check_batch(ctx: &GpuContext, gpu: &GpuSpecularAa, params: SpecularAaParams, lengths: &[f32]) {
    let query = SpecularAaBatchQuery {
        params,
        avg_normal_lengths: lengths.to_vec(),
    };
    let got = gpu.eval(ctx, &query);
    assert_eq!(
        got.len(),
        lengths.len(),
        "one result per average normal length"
    );
    for (idx, (g, &l)) in got.iter().zip(lengths.iter()).enumerate() {
        let want = params.evaluate(l);
        assert!(
            close(g.linear_roughness, want.linear_roughness),
            "sample {idx} (l={l}): linear_roughness gpu {} vs cpu {}",
            g.linear_roughness,
            want.linear_roughness
        );
        assert!(
            close(g.perceptual_roughness, want.perceptual_roughness),
            "sample {idx} (l={l}): perceptual_roughness gpu {} vs cpu {}",
            g.perceptual_roughness,
            want.perceptual_roughness
        );
        assert!(
            close(g.toksvig_factor, want.toksvig_factor),
            "sample {idx} (l={l}): toksvig_factor gpu {} vs cpu {}",
            g.toksvig_factor,
            want.toksvig_factor
        );
        assert!(
            close(g.added_variance, want.added_variance),
            "sample {idx} (l={l}): added_variance gpu {} vs cpu {}",
            g.added_variance,
            want.added_variance
        );
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpecularAa::new(&ctx);
    let query = SpecularAaBatchQuery {
        params: SpecularAaParams::new(0.4, 0.1, 0.5, 16.0),
        avg_normal_lengths: Vec::new(),
    };
    assert!(
        gpu.eval(&ctx, &query).is_empty(),
        "an empty batch yields no results and issues no dispatch"
    );
}

#[test]
fn empty_scalars_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpecularAa::new(&ctx);
    let query = SpecularAaScalarQuery {
        samples: Vec::new(),
    };
    assert!(
        gpu.eval_scalars(&ctx, &query).is_empty(),
        "an empty scalar batch yields no results and issues no dispatch"
    );
}

#[test]
fn coherent_footprint_is_the_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpecularAa::new(&ctx);
    // A unit average length with zero screen variance is the documented identity
    // case: the perceptual roughness passes through and the Toksvig factor is 1.
    let params = SpecularAaParams::new(0.5, 0.0, 1.0, 32.0);
    check_batch(&ctx, &gpu, params, &[1.0]);

    let got = gpu
        .eval(
            &ctx,
            &SpecularAaBatchQuery {
                params,
                avg_normal_lengths: vec![1.0],
            },
        )
        .remove(0);
    assert!(
        close(got.perceptual_roughness, 0.5),
        "identity perceptual roughness, got {}",
        got.perceptual_roughness
    );
    assert!(
        close(got.toksvig_factor, 1.0),
        "identity Toksvig factor, got {}",
        got.toksvig_factor
    );
    assert!(
        close(got.added_variance, 0.0),
        "identity adds no variance, got {}",
        got.added_variance
    );
}

#[test]
fn batch_matches_reference_across_param_sets() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpecularAa::new(&ctx);
    // Interior average normal lengths: never toward 0, never pinned at MIN_LEN,
    // never exactly 1 except the coherent identity (handled above).
    let lengths = [0.95_f32, 0.8, 0.6, 0.45, 0.3, 0.15, 0.08];
    // Several parameter sets spanning low/high roughness, screen variance and a
    // tight versus loose kappa clamp. Values stay clearly interior so no clamp
    // tie is straddled.
    let param_sets = [
        SpecularAaParams::new(0.3, 0.05, 1.0, 16.0),
        SpecularAaParams::new(0.5, 0.1, 0.5, 32.0),
        SpecularAaParams::new(0.7, 0.02, 0.2, 8.0),
        SpecularAaParams::new(0.25, 0.15, 0.3, 48.0),
    ];
    for params in param_sets {
        check_batch(&ctx, &gpu, params, &lengths);
    }
}

#[test]
fn batch_random_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpecularAa::new(&ctx);
    let mut state = 0x5eed_a17a_c0de_f00d_u64;
    for _ in 0..8 {
        let params = SpecularAaParams::new(
            lcg_range(&mut state, 0.1, 0.9),
            lcg_range(&mut state, 0.0, 0.2),
            lcg_range(&mut state, 0.1, 0.8),
            lcg_range(&mut state, 4.0, 64.0),
        );
        let mut lengths = Vec::with_capacity(32);
        for _ in 0..32 {
            lengths.push(lcg_range(&mut state, 0.05, 1.0));
        }
        check_batch(&ctx, &gpu, params, &lengths);
    }
}

#[test]
fn scalars_match_each_reference_function() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpecularAa::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    let mut samples = Vec::with_capacity(64);
    for _ in 0..64 {
        samples.push(SpecularAaScalarSample {
            avg_normal_length: lcg_range(&mut state, 0.05, 1.0),
            gloss_power: lcg_range(&mut state, 1.0, 64.0),
            // Interior perceptual roughness so perceptual_to_linear avoids the
            // clamp ties at 0 and 1.
            perceptual: lcg_range(&mut state, 0.05, 0.95),
            // Interior linear roughness, kept >= ~0.01 so the sqrt input is well
            // away from zero.
            linear: lcg_range(&mut state, 0.02, 0.9),
            // Clearly-positive variance so the Frostbite kernel is non-trivial.
            variance: lcg_range(&mut state, 0.01, 0.3),
            // Positive kappa large enough not to always dominate, small enough
            // to sometimes clamp.
            kappa: lcg_range(&mut state, 0.05, 0.6),
        });
    }
    let got = gpu.eval_scalars(
        &ctx,
        &SpecularAaScalarQuery {
            samples: samples.clone(),
        },
    );
    assert_eq!(got.len(), samples.len(), "one tuple per sample");
    for (idx, (g, s)) in got.iter().zip(samples.iter()).enumerate() {
        assert!(
            close(
                g.toksvig_factor,
                toksvig_factor(s.avg_normal_length, s.gloss_power)
            ),
            "sample {idx}: toksvig_factor gpu {} vs cpu {}",
            g.toksvig_factor,
            toksvig_factor(s.avg_normal_length, s.gloss_power)
        );
        assert!(
            close(
                g.toksvig_effective_gloss,
                toksvig_effective_gloss(s.avg_normal_length, s.gloss_power)
            ),
            "sample {idx}: toksvig_effective_gloss gpu {} vs cpu {}",
            g.toksvig_effective_gloss,
            toksvig_effective_gloss(s.avg_normal_length, s.gloss_power)
        );
        assert!(
            close(
                g.normal_length_variance,
                normal_length_variance(s.avg_normal_length)
            ),
            "sample {idx}: normal_length_variance gpu {} vs cpu {}",
            g.normal_length_variance,
            normal_length_variance(s.avg_normal_length)
        );
        assert!(
            close(
                g.perceptual_to_linear,
                perceptual_to_linear_roughness(s.perceptual)
            ),
            "sample {idx}: perceptual_to_linear gpu {} vs cpu {}",
            g.perceptual_to_linear,
            perceptual_to_linear_roughness(s.perceptual)
        );
        assert!(
            close(
                g.linear_to_perceptual,
                linear_to_perceptual_roughness(s.linear)
            ),
            "sample {idx}: linear_to_perceptual gpu {} vs cpu {}",
            g.linear_to_perceptual,
            linear_to_perceptual_roughness(s.linear)
        );
        assert!(
            close(
                g.frostbite,
                frostbite_specular_aa(s.linear, s.variance, s.kappa)
            ),
            "sample {idx}: frostbite gpu {} vs cpu {}",
            g.frostbite,
            frostbite_specular_aa(s.linear, s.variance, s.kappa)
        );
    }
}

#[test]
fn scalars_round_trip_perceptual_linear() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpecularAa::new(&ctx);
    // Feed a perceptual roughness in as `perceptual` and its square in as
    // `linear`, so perceptual_to_linear and linear_to_perceptual invert on the
    // same sample. Values interior to (0, 1) avoid the clamp ties.
    let perceptuals = [0.1_f32, 0.25, 0.4, 0.55, 0.7, 0.85];
    let samples: Vec<SpecularAaScalarSample> = perceptuals
        .iter()
        .map(|&p| SpecularAaScalarSample {
            avg_normal_length: 0.5,
            gloss_power: 16.0,
            perceptual: p,
            linear: p * p,
            variance: 0.05,
            kappa: 0.5,
        })
        .collect();
    let got = gpu.eval_scalars(
        &ctx,
        &SpecularAaScalarQuery {
            samples: samples.clone(),
        },
    );
    for (g, &p) in got.iter().zip(perceptuals.iter()) {
        assert!(
            close(g.perceptual_to_linear, p * p),
            "perceptual_to_linear gpu {} vs cpu {}",
            g.perceptual_to_linear,
            p * p
        );
        // linear = p*p was uploaded, so linear_to_perceptual should recover p.
        assert!(
            close(g.linear_to_perceptual, p),
            "linear_to_perceptual gpu {} vs recovered {p}",
            g.linear_to_perceptual
        );
    }
}
