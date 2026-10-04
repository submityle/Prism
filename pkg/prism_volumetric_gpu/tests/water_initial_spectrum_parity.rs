//! Real-device parity for the ocean initial-spectrum build twin:
//! [`GpuWaterInitialSpectrum`](prism_volumetric_gpu::water_initial_spectrum::GpuWaterInitialSpectrum)
//! must reproduce the dependency-free `CPU` golden
//! [`build_initial_spectrum`](prism_render_architecture::water::initial_spectrum::build_initial_spectrum)
//! — the time-zero `h0(k)` / `h0(-k)` `Tessendorf` field a spectral ocean
//! builds once per sea state — across the three spectrum shapes, several wind
//! speeds, seeds and resolutions (including non-power-of-two), the calm sea and
//! the degenerate request.
//!
//! Closing this twin makes the whole `initial_spectrum -> advance -> ifft2 ->
//! surface` chain `GPU` self-consistent: the surface-synthesis twin already
//! runs the tail from a host-built field, and this twin builds that field on
//! device, so no spectral-ocean step is left `CPU`-only.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden [`build_initial_spectrum`] is public and pure, so the expected
//! field is built in-host and compared cell by cell against the `GPU` readback.
//! A `GPU == oracle` pass is therefore directly a `GPU == golden` pass.
//!
//! # Parity criterion
//!
//! The integer Gaussian field is bit-identical (`WGSL` `u32` arithmetic wraps
//! exactly like the golden's `wrapping_*` hash), so the only residual is the
//! last-place slack of the shared `exp_approx` polynomial under a `GPU` fused
//! multiply-add in the energy density. Each component is asserted within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::initial_spectrum`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::initial_spectrum::{
    build_initial_spectrum, OceanSpectrumField,
};
use prism_render_architecture::water::spectrum::{SpectrumKind, SpectrumParams};
use prism_render_architecture::water::Vec2;
use prism_volumetric_gpu::water_initial_spectrum::{
    GpuWaterInitialSpectrum, WaterInitialSpectrumField, WaterSpectrumKind,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on one amplitude component. A `GPU` fused multiply-add
/// may land a few units in the last place from the scalar `exp_approx`
/// reference; `1e-4` admits that while still failing a wrong port.
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

/// Maps the `GPU` spectrum kind to its `CPU`-golden twin.
fn golden_kind(kind: WaterSpectrumKind) -> SpectrumKind {
    match kind {
        WaterSpectrumKind::Phillips => SpectrumKind::Phillips,
        WaterSpectrumKind::Jonswap => SpectrumKind::Jonswap,
        WaterSpectrumKind::PiersonMoskowitz => SpectrumKind::PiersonMoskowitz,
    }
}

/// Builds the `CPU`-golden sea-state parameters for one wind speed and shape.
fn sea(kind: WaterSpectrumKind, wind: (f32, f32)) -> SpectrumParams {
    SpectrumParams {
        kind: golden_kind(kind),
        wind: Vec2::new(wind.0, wind.1),
        amplitude: 0.5,
        peak_enhancement: 3.3,
        min_wavelength: 0.2,
        directional_exponent: 2,
    }
}

/// Pins one `GPU` field against the `CPU` golden, cell by cell.
fn check_eq(got: &WaterInitialSpectrumField, want: &OceanSpectrumField, n: u32, label: &str) {
    assert_eq!(got.resolution, n, "{label}: resolution");
    assert_eq!(got.h0.len(), want.h0.len(), "{label}: h0 len");
    assert_eq!(got.h0_neg.len(), want.h0_neg.len(), "{label}: h0_neg len");
    for (i, (g, w)) in got.h0.iter().zip(want.h0.iter()).enumerate() {
        assert!(
            close(g.re, w.re),
            "{label}: h0[{i}].re gpu {} vs cpu {}",
            g.re,
            w.re
        );
        assert!(
            close(g.im, w.im),
            "{label}: h0[{i}].im gpu {} vs cpu {}",
            g.im,
            w.im
        );
    }
    for (i, (g, w)) in got.h0_neg.iter().zip(want.h0_neg.iter()).enumerate() {
        assert!(
            close(g.re, w.re),
            "{label}: h0_neg[{i}].re gpu {} vs cpu {}",
            g.re,
            w.re
        );
        assert!(
            close(g.im, w.im),
            "{label}: h0_neg[{i}].im gpu {} vs cpu {}",
            g.im,
            w.im
        );
    }
}

/// Builds the golden field and the `GPU` field for the same sea state and pins
/// them.
#[expect(
    clippy::too_many_arguments,
    reason = "one sea state is genuinely this many independent scalar inputs"
)]
fn check(
    ctx: &GpuContext,
    gpu: &GpuWaterInitialSpectrum,
    n: u32,
    patch_size: f32,
    kind: WaterSpectrumKind,
    wind: (f32, f32),
    seed: u32,
    label: &str,
) {
    let params = sea(kind, wind);
    let want = build_initial_spectrum(n, patch_size, params, seed);
    let got = gpu.evaluate(
        ctx,
        n,
        patch_size,
        kind,
        wind,
        params.amplitude,
        params.peak_enhancement,
        params.min_wavelength,
        params.directional_exponent,
        seed,
    );
    check_eq(&got, &want, n, label);
}

const KINDS: [WaterSpectrumKind; 3] = [
    WaterSpectrumKind::Phillips,
    WaterSpectrumKind::Jonswap,
    WaterSpectrumKind::PiersonMoskowitz,
];

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn degenerate_request_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_initial_spectrum parity: no wgpu adapter");
        return;
    };
    let gpu = GpuWaterInitialSpectrum::new(&ctx);
    // Zero resolution and non-positive patch are both honest no-ops.
    let zero_n = gpu.evaluate(
        &ctx,
        0,
        128.0,
        WaterSpectrumKind::Phillips,
        (12.0, 0.0),
        0.5,
        3.3,
        0.2,
        2,
        7,
    );
    assert_eq!(zero_n.resolution, 0, "zero resolution yields no field");
    assert!(zero_n.h0.is_empty() && zero_n.h0_neg.is_empty());
    let zero_patch = gpu.evaluate(
        &ctx,
        16,
        0.0,
        WaterSpectrumKind::Phillips,
        (12.0, 0.0),
        0.5,
        3.3,
        0.2,
        2,
        7,
    );
    assert_eq!(
        zero_patch.resolution, 0,
        "non-positive patch yields no field"
    );
    assert!(zero_patch.h0.is_empty() && zero_patch.h0_neg.is_empty());
}

#[test]
fn matches_golden_over_resolutions() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterInitialSpectrum::new(&ctx);
    // Powers of two and deliberate non-powers (the build is defined for any N).
    for &n in &[1u32, 2, 3, 4, 8, 16, 24, 32] {
        check(
            &ctx,
            &gpu,
            n,
            128.0,
            WaterSpectrumKind::Phillips,
            (12.0, 0.0),
            7,
            "resolution sweep",
        );
    }
}

#[test]
fn matches_golden_over_kinds() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterInitialSpectrum::new(&ctx);
    for &kind in &KINDS {
        check(&ctx, &gpu, 32, 128.0, kind, (11.0, 4.0), 5, "kind sweep");
    }
}

#[test]
fn matches_golden_over_wind_speeds() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterInitialSpectrum::new(&ctx);
    for &kind in &KINDS {
        for &w in &[3.0f32, 6.0, 9.0, 16.0] {
            check(&ctx, &gpu, 16, 64.0, kind, (w, 0.0), 42, "wind sweep");
        }
    }
}

#[test]
fn matches_golden_over_seeds() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterInitialSpectrum::new(&ctx);
    for &seed in &[0u32, 1, 7, 1234, 0xABCD_1234] {
        check(
            &ctx,
            &gpu,
            16,
            96.0,
            WaterSpectrumKind::Jonswap,
            (10.0, 2.0),
            seed,
            "seed sweep",
        );
    }
}

#[test]
fn calm_sea_is_all_zero_but_sized() {
    // A zero-wind sea carries no spectral energy: the golden draws the full
    // `n*n` field but every amplitude is zero (it does not special-case calm),
    // so the twin must return the same length of zeros rather than an empty
    // field.
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterInitialSpectrum::new(&ctx);
    let n = 16u32;
    check(
        &ctx,
        &gpu,
        n,
        128.0,
        WaterSpectrumKind::Phillips,
        (0.0, 0.0),
        3,
        "calm sea",
    );
    let got = gpu.evaluate(
        &ctx,
        n,
        128.0,
        WaterSpectrumKind::Phillips,
        (0.0, 0.0),
        0.5,
        3.3,
        0.2,
        2,
        3,
    );
    assert_eq!(
        got.h0.len(),
        (n as usize) * (n as usize),
        "calm field is sized"
    );
    for amp in got.h0.iter().chain(got.h0_neg.iter()) {
        assert!(
            amp.re.abs() <= EPS && amp.im.abs() <= EPS,
            "calm amplitude is zero"
        );
    }
}

#[test]
fn is_deterministic() {
    // The build is a pure function of its inputs, so two dispatches of the same
    // sea state must return bit-identical fields.
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterInitialSpectrum::new(&ctx);
    let run = || {
        gpu.evaluate(
            &ctx,
            24,
            128.0,
            WaterSpectrumKind::Jonswap,
            (13.0, 5.0),
            0.5,
            3.3,
            0.2,
            2,
            99,
        )
    };
    let a = run();
    let b = run();
    assert_eq!(a, b, "the same sea state rebuilds identically");
}
