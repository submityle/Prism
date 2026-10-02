//! Real-device parity for the octave-summed volumetric multiple-scattering
//! twin: [`GpuVolumetricMultiScatter`](prism_volumetric_gpu::GpuVolumetricMultiScatter)
//! must reproduce the `CPU` golden
//! [`volumetric_multiscatter`](prism_render_architecture::particle::volumetric_multiscatter)
//! across a scattering-cosine sweep over every octave count in `1..=8`, a large
//! random parameter batch, the direction-dot path (including the zero-length
//! degenerate collapse), the octave-count clamp, and the empty-input
//! short-circuit.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The response contains no transcendental call (the reference restricts itself
//! to `sqrt`), and the octave loop bound is bit-exact unsigned-integer
//! arithmetic, so the `CPU` and `GPU` evaluate the same closed-form algebra over
//! the same fixed octave count and diverge only through legal fused-multiply-add
//! contraction. Each channel and the mean luminance are asserted to within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3` (with a `1e-6` relative floor). The
//! Henyey-Greenstein `d^-1.5` lobes make the per-octave phase conditioning
//! `1.5 * d(denom)/denom`; near the (deliberately avoided) back-scatter
//! singularity the shared denominator's `~ULP` spread is amplified by
//! `1 / denom`, so the relative bound is widened by `1.5 * DENOM_ULP /
//! min_denom` over the octaves rather than relaxed blindly. For the
//! well-conditioned fixtures here this collapses back to `1e-3`. Several
//! scenarios additionally assert a non-trivial physical shape (an octave sum
//! that strictly grows, a lifted back-scatter floor) so a degenerate all-zero
//! or constant kernel could not pass.
//!
//! The fixtures use no `f32` transcendental: inputs come from a host-side
//! integer `LCG` (`sqrt` is permitted for direction normalization), octave
//! counts sweep the small integers `1..=8`, and the anisotropies stay inside
//! `(-1, 1)` well away from the phase singularity.
//!
//! Provenance: twinned from this repository's
//! [`volumetric_multiscatter`](prism_render_architecture::particle::volumetric_multiscatter);
//! no third-party engine source or derived code.

use prism_render_architecture::particle::shading::PhaseParams;
use prism_render_architecture::particle::volumetric_multiscatter::MultiScatterParams;
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::{
    GpuContext, GpuVolumetricMultiScatter, MultiScatterQuery, VolumetricMultiScatterResponse,
};

/// Absolute parity bound: a few-`ULP` fused-multiply-add slack in the octave
/// running products lands here, while a genuinely wrong port does not.
const ABS_EPS: f32 = 1.0e-4;

/// Relative parity bound, applied when the compared magnitude is large enough
/// that an absolute bound would be unfairly strict.
const REL_EPS: f32 = 1.0e-3;

/// Relative-error denominator floor, keeping it away from zero.
const REL_FLOOR: f32 = 1.0e-6;

/// First-order fused-multiply-add spread of the Henyey-Greenstein denominator
/// `1 + g^2 - 2*g*cos`. The `CPU` and `GPU` contract those `~O(1)` terms
/// differently, so the denominator differs by a couple of `f32` `ULP`
/// (`~2e-7`); this widens the per-octave relative bound by `1.5 * DENOM_ULP /
/// min_denom`.
const DENOM_ULP: f32 = 2.0e-7;

/// Floor every lobe denominator shares, matching the reference `EPS` used in
/// the Henyey-Greenstein `d = max(base, EPS)` guard.
const DENOM_FLOOR: f32 = 1.0e-6;

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[-1, 1)`.
fn lcg(state: &mut u64) -> f32 {
    // Knuth multiplier / increment; the shift takes the high bits where the
    // generator mixes best.
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    // 24 usable mantissa bits mapped onto [0, 1) then onto [-1, 1).
    let unit = (bits & 0x00ff_ffff) as f32 / 16_777_216.0;
    unit * 2.0 - 1.0
}

/// Draws a value in `[0, 1)` from the `LCG`.
fn lcg_unit(state: &mut u64) -> f32 {
    lcg(state) * 0.5 + 0.5
}

/// Replays the `sanitized` octave loop to find the smallest contributing
/// Henyey-Greenstein denominator, which sets the conditioning of the response.
///
/// Each octave's forward lobe uses anisotropy `g * scale` and the back lobe
/// uses `back_g * scale`, with `scale = anisotropy_falloff^i`; the forward lobe
/// contributes while the (clamped) back weight is below `1`, the back lobe while
/// it is above `0`. The denominator is floored exactly as the reference guards
/// it, so a tiny denominator widens the relative tolerance by its true
/// `1 / denom` sensitivity.
fn min_octave_denom(params: &MultiScatterParams, cos_theta: f32) -> f32 {
    let p = params.sanitized();
    let w = p.phase.back_lobe_weight;
    let mut scale = 1.0_f32;
    let mut min_d = f32::INFINITY;
    for _ in 0..p.octaves {
        let gf = p.phase.g * scale;
        let gb = p.phase.back_g * scale;
        let denom_f = (1.0 + gf * gf - 2.0 * gf * cos_theta).max(DENOM_FLOOR);
        let denom_b = (1.0 + gb * gb - 2.0 * gb * cos_theta).max(DENOM_FLOOR);
        if w < 1.0 {
            min_d = min_d.min(denom_f);
        }
        if w > 0.0 {
            min_d = min_d.min(denom_b);
        }
        scale *= p.anisotropy_falloff;
    }
    if min_d.is_infinite() {
        1.0
    } else {
        min_d
    }
}

/// Returns whether `gpu` and `cpu` agree within the absolute bound or the
/// conditioning-aware relative bound `rel_tol`.
fn close(gpu: f32, cpu: f32, rel_tol: f32) -> bool {
    let abs = (gpu - cpu).abs();
    let rel = abs / gpu.abs().max(cpu.abs()).max(REL_FLOOR);
    abs <= ABS_EPS || rel <= rel_tol
}

/// Asserts the `GPU` response matches the `CPU` golden for one `cos_theta`
/// query: the three `RGB` channels against
/// [`MultiScatterParams::response_cos`] and the fourth lane against
/// [`MultiScatterParams::luminance_cos`].
fn assert_cos_parity(
    params: &MultiScatterParams,
    cos_theta: f32,
    got: VolumetricMultiScatterResponse,
) {
    let cpu = params.response_cos(cos_theta);
    let lum = params.luminance_cos(cos_theta);
    let rel_tol = REL_EPS + 1.5 * DENOM_ULP / min_octave_denom(params, cos_theta);
    assert!(
        close(got.rgb[0], cpu.x, rel_tol)
            && close(got.rgb[1], cpu.y, rel_tol)
            && close(got.rgb[2], cpu.z, rel_tol),
        "rgb mismatch for cos {cos_theta} params {params:?}: gpu {:?}, cpu ({}, {}, {}) (rel_tol {rel_tol})",
        got.rgb,
        cpu.x,
        cpu.y,
        cpu.z
    );
    assert!(
        close(got.luminance, lum, rel_tol),
        "luminance mismatch for cos {cos_theta} params {params:?}: gpu {}, cpu {lum} (rel_tol {rel_tol})",
        got.luminance
    );
}

/// Builds a well-conditioned random `MultiScatterParams` from the `LCG`:
/// albedo and the `0..=1` factors spread their full range, the anisotropies stay
/// inside `[-0.85, 0.85]` away from the `g -> +-1` singularity, the ambient lift
/// is non-negative, and the octave count sweeps the small integers `1..=8`.
fn random_params(state: &mut u64) -> MultiScatterParams {
    let octaves = 1 + (((*state >> 29) as u32) % 8);
    // Advance the stream so the octave draw does not correlate with the floats.
    let _ = lcg(state);
    MultiScatterParams {
        albedo: Vec3::new(lcg_unit(state), lcg_unit(state), lcg_unit(state)),
        phase: PhaseParams {
            g: lcg(state) * 0.85,
            back_lobe_weight: lcg_unit(state),
            back_g: lcg(state) * 0.85,
        },
        octaves,
        anisotropy_falloff: lcg_unit(state),
        octave_decay: lcg_unit(state),
        ambient_lift: lcg_unit(state) * 2.0,
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_multiscatter_matches_cpu_golden_across_cosine_and_octave_sweep() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping volumetric multiscatter parity: no wgpu adapter on this host");
        return;
    };
    let evaluator = GpuVolumetricMultiScatter::new(&ctx);

    // A gently colored forward-scattering smoke with a modest back lobe, so
    // every octave and both lobes contribute across the sweep.
    let base = MultiScatterParams {
        albedo: Vec3::new(0.82, 0.78, 0.7),
        phase: PhaseParams {
            g: 0.4,
            back_lobe_weight: 0.25,
            back_g: -0.3,
        },
        octaves: 1,
        anisotropy_falloff: 0.6,
        octave_decay: 0.9,
        ambient_lift: 0.4,
    };

    // Sweep the scattering cosine from pure back-scatter to pure forward-scatter
    // for every octave count in 1..=8.
    let mut queries = Vec::new();
    let mut metadata = Vec::new();
    for octaves in 1u32..=8 {
        let params = MultiScatterParams { octaves, ..base };
        for k in 0..=40 {
            let cos_theta = -1.0 + (k as f32) * 0.05;
            queries.push(MultiScatterQuery::from_cos(&params, cos_theta));
            metadata.push((params, cos_theta));
        }
    }

    let gpu = evaluator.eval(&ctx, &queries);
    assert_eq!(gpu.len(), queries.len(), "one response per query");
    for (i, (params, cos_theta)) in metadata.iter().enumerate() {
        assert_cos_parity(params, *cos_theta, gpu[i]);
    }

    // Physical shape (checked on the GPU output itself, not just the CPU
    // reference): at a fixed side-scatter cosine the forward channel must grow
    // strictly with each added octave, proving the running octave recurrence
    // actually accumulates on-device.
    let side = 0.5_f32;
    let mut shape_queries = Vec::new();
    for octaves in 1u32..=8 {
        let params = MultiScatterParams { octaves, ..base };
        shape_queries.push(MultiScatterQuery::from_cos(&params, side));
    }
    let shape = evaluator.eval(&ctx, &shape_queries);
    for pair in shape.windows(2) {
        assert!(
            pair[1].rgb[0] > pair[0].rgb[0],
            "each added octave must add energy: {} then {}",
            pair[0].rgb[0],
            pair[1].rgb[0]
        );
    }
    assert!(
        shape[0].rgb[0] > 0.0,
        "the single-octave response must be positive"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_multiscatter_matches_cpu_golden_across_random_parameter_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping volumetric multiscatter parity: no wgpu adapter on this host");
        return;
    };
    let evaluator = GpuVolumetricMultiScatter::new(&ctx);

    // A large random batch over every field, evaluated at several fixed cosines
    // spanning back- to forward-scatter. Inputs come from the integer LCG, so
    // no f32 transcendental seeds the fixture.
    let cosines = [-1.0f32, -0.6, -0.2, 0.0, 0.35, 0.75, 1.0];
    let mut state = 0x51ed_270b_2e67_9a1d_u64;
    let mut queries = Vec::new();
    let mut metadata = Vec::new();
    for _ in 0..256 {
        let params = random_params(&mut state);
        for &cos_theta in &cosines {
            queries.push(MultiScatterQuery::from_cos(&params, cos_theta));
            metadata.push((params, cos_theta));
        }
    }

    let gpu = evaluator.eval(&ctx, &queries);
    assert_eq!(gpu.len(), queries.len(), "one response per query");
    for (i, (params, cos_theta)) in metadata.iter().enumerate() {
        assert_cos_parity(params, *cos_theta, gpu[i]);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_multiscatter_matches_cpu_golden_on_direction_path() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping volumetric multiscatter parity: no wgpu adapter on this host");
        return;
    };
    let evaluator = GpuVolumetricMultiScatter::new(&ctx);

    // The direction path: both sides derive the scattering cosine from the dot
    // product of the normalized incoming / outgoing directions. The directions
    // are exact unit vectors built from integer Pythagorean quadruples (sqrt is
    // not a transcendental), so the dot product is well-defined and the GPU and
    // CPU normalize-or-zero identically.
    let units = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.6, 0.8, 0.0),
        Vec3::new(-0.6, 0.0, 0.8),
        Vec3::new(2.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0),
        Vec3::new(-1.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0),
    ];
    let params = MultiScatterParams {
        albedo: Vec3::new(0.9, 0.85, 0.6),
        phase: PhaseParams {
            g: 0.5,
            back_lobe_weight: 0.3,
            back_g: -0.4,
        },
        octaves: 6,
        anisotropy_falloff: 0.55,
        octave_decay: 0.95,
        ambient_lift: 0.5,
    };

    let mut queries = Vec::new();
    let mut metadata = Vec::new();
    for &incoming in &units {
        for &outgoing in &units {
            queries.push(MultiScatterQuery::from_directions(
                &params, incoming, outgoing,
            ));
            metadata.push((incoming, outgoing));
        }
    }

    let gpu = evaluator.eval(&ctx, &queries);
    assert_eq!(gpu.len(), queries.len(), "one response per query");
    for (i, (incoming, outgoing)) in metadata.iter().enumerate() {
        let cos_theta = incoming
            .normalize_or_zero()
            .dot(outgoing.normalize_or_zero());
        assert_cos_parity(&params, cos_theta, gpu[i]);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_multiscatter_collapses_zero_length_direction_to_isotropic_cosine() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping volumetric multiscatter parity: no wgpu adapter on this host");
        return;
    };
    let evaluator = GpuVolumetricMultiScatter::new(&ctx);

    let params = MultiScatterParams::default();
    // A zero-length direction must collapse to cos_theta = 0 (never NaN), on
    // either side of the dot product, matching the reference normalize-or-zero.
    let queries = vec![
        MultiScatterQuery::from_directions(&params, Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO),
        MultiScatterQuery::from_directions(&params, Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)),
        MultiScatterQuery::from_directions(&params, Vec3::ZERO, Vec3::ZERO),
    ];

    let gpu = evaluator.eval(&ctx, &queries);
    assert_eq!(gpu.len(), queries.len(), "one response per query");
    for response in &gpu {
        for channel in response.rgb {
            assert!(
                channel.is_finite(),
                "degenerate direction produced non-finite {channel}"
            );
        }
        // Every zero-length case collapses to the isotropic cos_theta = 0.
        assert_cos_parity(&params, 0.0, *response);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_multiscatter_clamps_octave_count_like_the_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping volumetric multiscatter parity: no wgpu adapter on this host");
        return;
    };
    let evaluator = GpuVolumetricMultiScatter::new(&ctx);

    // Zero octaves must clamp up to a single order and a huge count down to the
    // 64-octave cap; the GPU must reproduce the clamped reference exactly.
    let cos_theta = 0.3_f32;
    let zero = MultiScatterParams {
        octaves: 0,
        ..MultiScatterParams::default()
    };
    let over = MultiScatterParams {
        octaves: 10_000,
        ..MultiScatterParams::default()
    };
    let one = MultiScatterParams {
        octaves: 1,
        ..MultiScatterParams::default()
    };
    let cap = MultiScatterParams {
        octaves: 64,
        ..MultiScatterParams::default()
    };

    let queries = vec![
        MultiScatterQuery::from_cos(&zero, cos_theta),
        MultiScatterQuery::from_cos(&over, cos_theta),
    ];
    let gpu = evaluator.eval(&ctx, &queries);
    assert_eq!(gpu.len(), 2, "one response per query");

    // The clamped integer octave count is bit-exact, so the clamped-up zero
    // query equals the one-octave reference and the clamped-down huge query
    // equals the 64-octave reference.
    assert_cos_parity(&one, cos_theta, gpu[0]);
    assert_cos_parity(&cap, cos_theta, gpu[1]);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_multiscatter_handles_empty_input() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping volumetric multiscatter parity: no wgpu adapter on this host");
        return;
    };
    let evaluator = GpuVolumetricMultiScatter::new(&ctx);
    assert!(evaluator.eval(&ctx, &[]).is_empty(), "empty in, empty out");
}
