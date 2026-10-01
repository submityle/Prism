//! Real-device parity for the particle multiple-scattering twin:
//! [`GpuParticleMultiScatter`] must reproduce the `CPU` golden
//! [`MultiScatterParams::response_cos`](prism_render_architecture::particle::volumetric_multiscatter::MultiScatterParams::response_cos)
//! for every scattering cosine across the forward/back range, several authored
//! parameter sets, the one-octave degenerate case and deliberately
//! out-of-range inputs the reference sanitizes.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each output is an octave sum accumulated sequentially in the same order as
//! the reference, with no reorderable reduction, so `CPU` and `GPU` evaluate
//! the same algebra. They are not bit-exact (a `GPU` may fuse a multiply-add),
//! so values are asserted to within `abs_diff < 1e-4` or `rel_diff < 1e-3` —
//! far tighter than any physically meaningful response difference and enough to
//! fail a wrong port (a swapped factor, a dropped clamp, a missing ambient
//! term). The property scenes also assert octave-energy monotonicity, isotropic
//! cosine symmetry and a high-albedo non-black floor so a degenerate constant
//! kernel could not pass.
//!
//! Provenance: standard Wrenninge/Hillaire-style octave multiple-scattering
//! energy compensation; no Unreal Engine source or derived code.

use prism_render_architecture::particle::shading::PhaseParams;
use prism_render_architecture::particle::volumetric_multiscatter::MultiScatterParams;
use prism_render_architecture::particle::volumetrics::double_lobe_phase;
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::{GpuContext, GpuParticleMultiScatter, MultiScatterResponse};

/// Parity tolerance (shared with the module docs): a result passes when it is
/// within `1e-4` absolute or `1e-3` relative of the `CPU` golden, admitting
/// legal fused-multiply-add contraction while still failing a wrong port.
const ABS_TOL: f32 = 1e-4;
/// Relative tolerance companion to [`ABS_TOL`].
const REL_TOL: f32 = 1e-3;

/// Returns true when `got` matches `exp` within the documented tolerance.
fn close(got: f32, exp: f32) -> bool {
    let abs_diff = (got - exp).abs();
    let rel_diff = abs_diff / exp.abs().max(ABS_TOL);
    abs_diff < ABS_TOL || rel_diff < REL_TOL
}

/// Asserts every `gpu` response matches the `CPU` golden `response_cos` for the
/// paired cosine, channel by channel.
fn assert_parity(params: MultiScatterParams, cosines: &[f32], gpu: &[MultiScatterResponse]) {
    assert_eq!(gpu.len(), cosines.len(), "one response per cosine");
    for (i, &cos_theta) in cosines.iter().enumerate() {
        let exp = params.response_cos(cos_theta);
        let got = gpu[i];
        for (got_v, exp_v, name) in [
            (got.red, exp.x, "red"),
            (got.green, exp.y, "green"),
            (got.blue, exp.z, "blue"),
        ] {
            assert!(
                close(got_v, exp_v),
                "{name} mismatch at cos_theta {cos_theta}: gpu {got_v}, cpu {exp_v}"
            );
        }
    }
}

/// A sweep of scattering cosines spanning full back-scatter to full forward
/// scatter, including the isotropic midpoint.
fn cosine_sweep() -> Vec<f32> {
    vec![-1.0, -0.85, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.85, 1.0]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_golden_across_cosines_and_presets() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping particle multiscatter parity: no wgpu adapter on this host");
        return;
    };
    let twin = GpuParticleMultiScatter::new(&ctx);
    let cosines = cosine_sweep();

    // The authored default grey smoke.
    let default_params = MultiScatterParams::default();
    let gpu = twin.eval(&ctx, default_params, &cosines);
    assert_parity(default_params, &cosines, &gpu);

    // A coloured, strongly forward-scattering medium with a back lobe and a
    // steeper per-octave decay.
    let coloured = MultiScatterParams {
        albedo: Vec3::new(0.9, 0.5, 0.2),
        phase: PhaseParams {
            g: 0.7,
            back_lobe_weight: 0.3,
            back_g: -0.5,
        },
        octaves: 6,
        anisotropy_falloff: 0.6,
        octave_decay: 0.9,
        ambient_lift: 0.5,
    };
    let gpu = twin.eval(&ctx, coloured, &cosines);
    assert_parity(coloured, &cosines, &gpu);
}

#[test]
fn gpu_sanitizes_out_of_range_params_like_the_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuParticleMultiScatter::new(&ctx);
    let cosines = cosine_sweep();

    // Albedo beyond 1, falloff and decay beyond 1, a negative ambient lift, and
    // an octave count beyond MAX_OCTAVES: the shader must clamp each exactly as
    // `sanitized` does, so parity against `response_cos` still holds.
    let wild = MultiScatterParams {
        albedo: Vec3::new(1.5, -0.2, 0.8),
        phase: PhaseParams {
            g: 0.4,
            back_lobe_weight: 2.0,
            back_g: -0.3,
        },
        octaves: 10_000,
        anisotropy_falloff: 3.0,
        octave_decay: -1.0,
        ambient_lift: -5.0,
    };
    let gpu = twin.eval(&ctx, wild, &cosines);
    assert_parity(wild, &cosines, &gpu);

    // A zero octave count must clamp up to a single scattering order.
    let zero_octaves = MultiScatterParams {
        octaves: 0,
        ..MultiScatterParams::default()
    };
    let gpu = twin.eval(&ctx, zero_octaves, &cosines);
    assert_parity(zero_octaves, &cosines, &gpu);
}

#[test]
fn one_octave_degenerates_to_single_scatter_phase() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuParticleMultiScatter::new(&ctx);

    let phase = PhaseParams {
        g: 0.4,
        back_lobe_weight: 0.0,
        back_g: 0.0,
    };
    let params = MultiScatterParams {
        albedo: Vec3::splat(0.7),
        phase,
        octaves: 1,
        anisotropy_falloff: 0.5,
        octave_decay: 1.0,
        // No ambient lift, so the result is purely the single-scatter lobe.
        ambient_lift: 0.0,
    };
    let cos_theta = 0.3;
    let gpu = twin.eval(&ctx, params, &[cos_theta]);
    let single = double_lobe_phase(phase, cos_theta);
    // Order-0 throughput is unit on every channel, so all channels equal the
    // single-scatter lobe.
    assert!(close(gpu[0].red, single), "red: {} vs {single}", gpu[0].red);
    assert!(close(gpu[0].green, single));
    assert!(close(gpu[0].blue, single));
}

#[test]
fn energy_grows_with_octaves_on_device() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuParticleMultiScatter::new(&ctx);

    let base = MultiScatterParams {
        albedo: Vec3::splat(0.8),
        phase: PhaseParams {
            g: 0.4,
            back_lobe_weight: 0.0,
            back_g: 0.0,
        },
        octaves: 2,
        anisotropy_falloff: 0.5,
        octave_decay: 1.0,
        ambient_lift: 0.2,
    };
    // A side-scatter cosine where several octaves all contribute.
    let cos_theta = 0.5;
    let red_for = |octaves: u32| -> f32 {
        let params = MultiScatterParams { octaves, ..base };
        twin.eval(&ctx, params, &[cos_theta])[0].red
    };
    let r2 = red_for(2);
    let r4 = red_for(4);
    let r8 = red_for(8);
    // Each added octave contributes strictly positive energy on the device.
    assert!(r2 < r4, "energy must grow with octaves: {r2} !< {r4}");
    assert!(r4 < r8, "energy must grow with octaves: {r4} !< {r8}");
    // The geometric throughput converges, so a large octave count stays finite.
    let r64 = red_for(64);
    assert!(r64.is_finite(), "64-octave response must stay finite");
    assert!(r8 <= r64 + ABS_TOL);
}

#[test]
fn isotropic_albedo_is_symmetric_in_cosine() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuParticleMultiScatter::new(&ctx);

    let params = MultiScatterParams {
        albedo: Vec3::splat(0.6),
        // Fully isotropic base phase: g = 0, no back lobe.
        phase: PhaseParams::isotropic(),
        octaves: 5,
        anisotropy_falloff: 0.5,
        octave_decay: 1.0,
        ambient_lift: 0.3,
    };
    // An isotropic medium responds identically for mirrored cosines.
    let gpu = twin.eval(&ctx, params, &[0.7, -0.7]);
    assert!(
        close(gpu[0].red, gpu[1].red),
        "isotropic response must be symmetric: {} vs {}",
        gpu[0].red,
        gpu[1].red
    );
}

#[test]
fn high_albedo_white_furnace_is_not_dead_black() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuParticleMultiScatter::new(&ctx);

    let phase = PhaseParams {
        g: 0.8,
        back_lobe_weight: 0.0,
        back_g: 0.0,
    };
    let params = MultiScatterParams {
        // Near-white, strongly forward-scattering medium.
        albedo: Vec3::splat(0.95),
        phase,
        octaves: 8,
        anisotropy_falloff: 0.5,
        octave_decay: 1.0,
        ambient_lift: 1.0,
    };
    // Full back-scatter, where a single forward lobe is darkest.
    let back = twin.eval(&ctx, params, &[-1.0])[0].red;
    // Comfortably lifted away from black.
    assert!(back > 0.1, "back-scatter went dead black: {back}");
    // And strictly brighter than the bare single-scatter back lobe.
    let single_back = double_lobe_phase(phase, -1.0);
    assert!(
        back > single_back,
        "multi-scatter must exceed single-scatter: {back} !> {single_back}"
    );
}

#[test]
fn empty_cosines_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuParticleMultiScatter::new(&ctx);
    let out = twin.eval(&ctx, MultiScatterParams::default(), &[]);
    assert!(out.is_empty(), "an empty cosine slice yields no results");
}
