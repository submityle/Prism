//! Real-device parity for the ocean-surface-synthesis twin:
//! [`GpuWaterSurfaceSynth`](prism_volumetric_gpu::water_surface_synth::GpuWaterSurfaceSynth)
//! must reproduce the dependency-free `CPU` golden
//! [`synthesize_surface`](prism_render_architecture::water::synthesis::synthesize_surface)
//! — the full `initial_spectrum -> advance -> ifft2 -> surface` chain the
//! spectral ocean runs each frame — across wind speeds, resolutions, times and
//! choppiness, including the calm (flat) sea and the zero-choppiness case.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden [`build_initial_spectrum`] + [`synthesize_surface`] are public
//! and pure, so the expected surface is built in-host and the field's centred
//! amplitudes `h0` / `h0_neg` are fed verbatim to the `GPU`. A `GPU == oracle`
//! pass is therefore directly a `GPU == golden` pass.
//!
//! # Parity criterion
//!
//! Each field threads through `2*log2(n)` butterfly stages of range-reduced
//! Taylor `sin`/`cos` plus the two advance phasors. The `CPU` and `GPU` share
//! the polynomial and stage structure, so that approximation is common-mode and
//! cancels; the residual is only a `GPU` fused multiply-add's last-place slack,
//! accumulated over the stages. Each component is asserted within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::synthesis`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::initial_spectrum::{
    build_initial_spectrum, OceanSpectrumField,
};
use prism_render_architecture::water::spectrum::{SpectrumKind, SpectrumParams};
use prism_render_architecture::water::synthesis::{synthesize_surface, OceanSurface};
use prism_render_architecture::water::Vec2;
use prism_volumetric_gpu::water_surface_synth::{
    GpuWaterSurfaceSynth, WaterSurface, WaterSurfaceComplex,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a synthesised component. A `GPU` fused multiply-add
/// may land a few units in the last place from the scalar reference, summed
/// over the butterfly stages; `1e-4` admits that while still failing a wrong
/// port.
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

/// A `Phillips` sea blowing along `+x` at `wind_speed` m/s.
fn sea(wind_speed: f32) -> SpectrumParams {
    SpectrumParams {
        kind: SpectrumKind::Phillips,
        wind: Vec2::new(wind_speed, 0.0),
        amplitude: 0.5,
        peak_enhancement: 1.0,
        min_wavelength: 0.2,
        directional_exponent: 2,
    }
}

/// Converts a golden field's centred amplitudes to the `GPU` input element type.
fn to_gpu(field: &OceanSpectrumField) -> (Vec<WaterSurfaceComplex>, Vec<WaterSurfaceComplex>) {
    let h0 = field
        .h0
        .iter()
        .map(|c| WaterSurfaceComplex::new(c.re, c.im))
        .collect();
    let h0_neg = field
        .h0_neg
        .iter()
        .map(|c| WaterSurfaceComplex::new(c.re, c.im))
        .collect();
    (h0, h0_neg)
}

/// Pins one `GPU` synthesis against the `CPU` golden surface, field by field.
fn check_eq(got: &WaterSurface, want: &OceanSurface, n: u32, label: &str) {
    assert_eq!(got.resolution, n, "{label}: resolution");
    assert_eq!(got.height.len(), want.height.len(), "{label}: height len");
    assert_eq!(
        got.displacement_x.len(),
        want.displacement_x.len(),
        "{label}: disp_x len"
    );
    assert_eq!(
        got.displacement_z.len(),
        want.displacement_z.len(),
        "{label}: disp_z len"
    );
    for (i, (&g, &w)) in got.height.iter().zip(want.height.iter()).enumerate() {
        assert!(close(g, w), "{label}: height[{i}] gpu {g} vs cpu {w}");
    }
    for (i, (&g, &w)) in got
        .displacement_x
        .iter()
        .zip(want.displacement_x.iter())
        .enumerate()
    {
        assert!(close(g, w), "{label}: disp_x[{i}] gpu {g} vs cpu {w}");
    }
    for (i, (&g, &w)) in got
        .displacement_z
        .iter()
        .zip(want.displacement_z.iter())
        .enumerate()
    {
        assert!(close(g, w), "{label}: disp_z[{i}] gpu {g} vs cpu {w}");
    }
}

/// Builds the golden field, synthesises it on both paths and pins them.
fn check(
    ctx: &GpuContext,
    gpu: &GpuWaterSurfaceSynth,
    n: u32,
    patch_size: f32,
    wind: f32,
    seed: u32,
    time: f32,
    choppiness: f32,
    label: &str,
) {
    let field = build_initial_spectrum(n, patch_size, sea(wind), seed);
    let want = synthesize_surface(&field, time, choppiness);
    let (h0, h0_neg) = to_gpu(&field);
    let got = gpu.evaluate(ctx, &h0, &h0_neg, n, patch_size, time, choppiness);
    check_eq(&got, &want, n, label);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_surface() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_surface_synth parity: no wgpu adapter");
        return;
    };
    let gpu = GpuWaterSurfaceSynth::new(&ctx);
    let got = gpu.evaluate(&ctx, &[], &[], 16, 128.0, 1.0, 1.0);
    assert!(got.height.is_empty(), "an empty field produces no samples");
    assert!(got.displacement_x.is_empty());
    assert!(got.displacement_z.is_empty());
}

#[test]
fn matches_golden_over_resolutions() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSurfaceSynth::new(&ctx);
    for &n in &[1u32, 2, 4, 8, 16, 32] {
        check(&ctx, &gpu, n, 128.0, 12.0, 7, 0.0, 1.0, "resolution sweep");
    }
}

#[test]
fn matches_golden_over_time() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSurfaceSynth::new(&ctx);
    for &t in &[0.0f32, 1.5, 4.0, 12.5] {
        check(&ctx, &gpu, 32, 128.0, 11.0, 5, t, 1.0, "time sweep");
    }
}

#[test]
fn matches_golden_over_wind_speeds() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSurfaceSynth::new(&ctx);
    for &w in &[3.0f32, 6.0, 9.0, 16.0] {
        check(&ctx, &gpu, 16, 64.0, w, 42, 3.5, 0.8, "wind sweep");
    }
}

#[test]
fn calm_sea_is_flat() {
    // A zero-wind sea carries no spectral energy, so the surface is flat on both
    // paths: every component stays at zero to the transform's numerical floor.
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSurfaceSynth::new(&ctx);
    check(&ctx, &gpu, 32, 128.0, 0.0, 1, 2.0, 1.0, "calm sea");
}

#[test]
fn zero_choppiness_has_no_horizontal_displacement() {
    // With choppiness 0 both horizontal fields must be zero on both paths.
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSurfaceSynth::new(&ctx);
    let field = build_initial_spectrum(16, 64.0, sea(10.0), 8);
    let want = synthesize_surface(&field, 1.0, 0.0);
    let (h0, h0_neg) = to_gpu(&field);
    let got = gpu.evaluate(&ctx, &h0, &h0_neg, 16, 64.0, 1.0, 0.0);
    check_eq(&got, &want, 16, "zero choppiness");
    for (&dx, &dz) in got.displacement_x.iter().zip(got.displacement_z.iter()) {
        assert!(dx.abs() < EPS && dz.abs() < EPS, "horizontal must be zero");
    }
}

#[test]
fn batched_patches_synthesise_independently() {
    // Two distinct patches dispatched as one batch, each pinned to its own
    // golden surface, verifying the per-workgroup patch base offset.
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSurfaceSynth::new(&ctx);
    let n = 16u32;
    let patch_size = 64.0f32;
    let time = 2.5f32;
    let choppiness = 1.0f32;

    let field_a = build_initial_spectrum(n, patch_size, sea(8.0), 1);
    let field_b = build_initial_spectrum(n, patch_size, sea(14.0), 2);
    let want_a = synthesize_surface(&field_a, time, choppiness);
    let want_b = synthesize_surface(&field_b, time, choppiness);

    let (mut h0, mut h0_neg) = to_gpu(&field_a);
    let (h0_b, h0_neg_b) = to_gpu(&field_b);
    h0.extend_from_slice(&h0_b);
    h0_neg.extend_from_slice(&h0_neg_b);

    let got = gpu.evaluate(&ctx, &h0, &h0_neg, n, patch_size, time, choppiness);
    let per = (n * n) as usize;
    let got_a = WaterSurface {
        resolution: n,
        height: got.height[..per].to_vec(),
        displacement_x: got.displacement_x[..per].to_vec(),
        displacement_z: got.displacement_z[..per].to_vec(),
    };
    let got_b = WaterSurface {
        resolution: n,
        height: got.height[per..].to_vec(),
        displacement_x: got.displacement_x[per..].to_vec(),
        displacement_z: got.displacement_z[per..].to_vec(),
    };
    check_eq(&got_a, &want_a, n, "batch patch a");
    check_eq(&got_b, &want_b, n, "batch patch b");
}

#[test]
fn synthesis_is_deterministic() {
    // Two dispatches of the same field must give byte-identical results.
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSurfaceSynth::new(&ctx);
    let field = build_initial_spectrum(16, 64.0, sea(10.0), 9);
    let (h0, h0_neg) = to_gpu(&field);
    let a = gpu.evaluate(&ctx, &h0, &h0_neg, 16, 64.0, 3.5, 0.8);
    let b = gpu.evaluate(&ctx, &h0, &h0_neg, 16, 64.0, 3.5, 0.8);
    assert_eq!(a, b, "GPU synthesis must be deterministic");
}
