//! Real-device parity for the fused water-optics planning twin:
//! [`GpuWaterOpticsPlan`](prism_volumetric_gpu::water_optics_plan::GpuWaterOpticsPlan)
//! must reproduce the `CPU` golden
//! [`plan_optics`](prism_render_architecture::water::optics::plan_optics) — the
//! complete dispersion-plus-transport plan — across hand-picked fixtures (clear,
//! turbid, grazing, deep, isotropic and strongly forward-scattering water) and a
//! randomized sweep compared sample-for-sample.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The reference [`plan_optics`](prism_render_architecture::water::optics::plan_optics)
//! is public and pure, so it is called directly as the oracle: the host builds an
//! [`OpticsProfile`](prism_render_architecture::water::optics::OpticsProfile) and
//! [`OpticsInputs`](prism_render_architecture::water::optics::OpticsInputs) from
//! each fixture, runs the golden, and the twin is pinned against the resulting
//! [`OpticsPlan`](prism_render_architecture::water::optics::OpticsPlan). A
//! `GPU == golden` pass is therefore direct evidence the kernel computes the same
//! fused plan.
//!
//! # Parity criterion
//!
//! The nine continuous outputs plus the phase, inscatter and scatter boost thread
//! through the replicated `exp_approx`, a `sqrt`, and chained multiplies, so a
//! device result may land a few units in the last place from the scalar
//! reference; they are asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//! The `visible` flag is a magnitude comparison and is asserted exactly with `==`.
//!
//! # Conditioning
//!
//! Every fixture keeps the view-depth transmittance a clear margin away from the
//! visibility threshold so the `visible` flag cannot flip between the `CPU` and
//! `GPU`, and keeps the asymmetry `g` and the scatter albedo away from the `0.999`
//! clamp edges so the phase denominator and the multiple-scatter boost stay well
//! conditioned.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::optics`（融合 `dispersion` 与 `underwater`）；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::dispersion::RgbIor;
use prism_render_architecture::water::optics::{
    plan_optics, OpticsInputs, OpticsPlan, OpticsProfile,
};
use prism_render_architecture::water::underwater::{RgbColor, RgbExtinction};
use prism_volumetric_gpu::water_optics_plan::{
    GpuWaterOpticsPlan, WaterOpticsPlanQuery, WaterOpticsPlanResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a continuous output. A device `sqrt` or the `12`-step
/// `exp_approx` squaring may land a few units in the last place from the scalar
/// reference; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does not
/// inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Builds a [`WaterOpticsPlanQuery`] from its two source structs.
fn query_of(profile: OpticsProfile, inputs: OpticsInputs) -> WaterOpticsPlanQuery {
    WaterOpticsPlanQuery {
        cauchy_a: profile.cauchy_a,
        cauchy_b: profile.cauchy_b,
        refraction_strength: profile.refraction_strength,
        extinction_r: profile.extinction.r,
        extinction_g: profile.extinction.g,
        extinction_b: profile.extinction.b,
        scatter_albedo: profile.scatter_albedo,
        asymmetry_g: profile.asymmetry_g,
        visibility_threshold: profile.visibility_threshold,
        sin_incidence: inputs.sin_incidence,
        view_depth: inputs.view_depth,
        surface_color_r: inputs.surface_color.r,
        surface_color_g: inputs.surface_color.g,
        surface_color_b: inputs.surface_color.b,
        surface_light: inputs.surface_light,
        cos_scatter: inputs.cos_scatter,
        shaft_length: inputs.shaft_length,
    }
}

/// Flattens a golden [`OpticsPlan`] into the comparable twin result shape.
fn plan_to_result(plan: &OpticsPlan) -> WaterOpticsPlanResult {
    let RgbIor { r, g, b } = plan.iors;
    WaterOpticsPlanResult {
        ior_r: r,
        ior_g: g,
        ior_b: b,
        dispersion_offsets: plan.dispersion_offsets,
        depth_color_r: plan.depth_color.r,
        depth_color_g: plan.depth_color.g,
        depth_color_b: plan.depth_color.b,
        phase: plan.phase,
        inscatter: plan.inscatter,
        scatter_boost: plan.scatter_boost,
        visible: plan.visible,
    }
}

/// Pins one `GPU` plan against the golden plan: every continuous output within
/// tolerance, the `visible` flag exactly.
fn check_sample(idx: usize, got: &WaterOpticsPlanResult, want: &WaterOpticsPlanResult) {
    let pairs = [
        ("ior_r", got.ior_r, want.ior_r),
        ("ior_g", got.ior_g, want.ior_g),
        ("ior_b", got.ior_b, want.ior_b),
        (
            "disp0",
            got.dispersion_offsets[0],
            want.dispersion_offsets[0],
        ),
        (
            "disp1",
            got.dispersion_offsets[1],
            want.dispersion_offsets[1],
        ),
        (
            "disp2",
            got.dispersion_offsets[2],
            want.dispersion_offsets[2],
        ),
        ("depth_r", got.depth_color_r, want.depth_color_r),
        ("depth_g", got.depth_color_g, want.depth_color_g),
        ("depth_b", got.depth_color_b, want.depth_color_b),
        ("phase", got.phase, want.phase),
        ("inscatter", got.inscatter, want.inscatter),
        ("scatter_boost", got.scatter_boost, want.scatter_boost),
    ];
    for (name, g, w) in pairs {
        assert!(close(g, w), "sample {idx} {name}: gpu {g} vs cpu {w}");
    }
    assert_eq!(
        got.visible, want.visible,
        "sample {idx} visible: gpu {} vs cpu {}",
        got.visible, want.visible
    );
}

/// Dispatches every sample and pins each result against the golden `plan_optics`.
fn check(ctx: &GpuContext, gpu: &GpuWaterOpticsPlan, fixtures: &[(OpticsProfile, OpticsInputs)]) {
    let queries: Vec<WaterOpticsPlanQuery> =
        fixtures.iter().map(|&(p, i)| query_of(p, i)).collect();
    let got = gpu.evaluate(ctx, &queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (&(profile, inputs), result)) in fixtures.iter().zip(got.iter()).enumerate() {
        let want = plan_to_result(&plan_optics(profile, inputs));
        check_sample(idx, result, &want);
    }
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws an `f32` in `[lo, hi]` at milli resolution from `state`.
fn draw(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (lcg(state) % 1000) as f32 / 1000.0 * (hi - lo)
}

/// The deterministic clear-water profile and shallow-view inputs.
fn clear_water() -> (OpticsProfile, OpticsInputs) {
    (
        OpticsProfile {
            cauchy_a: 1.324,
            cauchy_b: 0.0032,
            refraction_strength: 0.05,
            extinction: RgbExtinction {
                r: 0.45,
                g: 0.15,
                b: 0.08,
            },
            scatter_albedo: 0.7,
            asymmetry_g: 0.6,
            visibility_threshold: 0.02,
        },
        OpticsInputs {
            sin_incidence: 0.7,
            view_depth: 3.0,
            surface_color: RgbColor {
                r: 1.0,
                g: 0.9,
                b: 0.8,
            },
            surface_light: 1.0,
            cos_scatter: 0.5,
            shaft_length: 5.0,
        },
    )
}

/// A grazing-angle, moderately turbid profile.
fn grazing_turbid() -> (OpticsProfile, OpticsInputs) {
    (
        OpticsProfile {
            cauchy_a: 1.33,
            cauchy_b: 0.006,
            refraction_strength: 2.0,
            extinction: RgbExtinction {
                r: 0.6,
                g: 0.25,
                b: 0.12,
            },
            scatter_albedo: 0.4,
            asymmetry_g: -0.3,
            visibility_threshold: 0.1,
        },
        OpticsInputs {
            sin_incidence: 0.95,
            view_depth: 2.0,
            surface_color: RgbColor {
                r: 0.8,
                g: 0.85,
                b: 0.95,
            },
            surface_light: 1.5,
            cos_scatter: -0.2,
            shaft_length: 8.0,
        },
    )
}

/// An isotropic (`g = 0`) shallow fixture with modest extinction.
fn isotropic_shallow() -> (OpticsProfile, OpticsInputs) {
    (
        OpticsProfile {
            cauchy_a: 1.31,
            cauchy_b: 0.004,
            refraction_strength: 0.5,
            extinction: RgbExtinction {
                r: 0.3,
                g: 0.1,
                b: 0.05,
            },
            scatter_albedo: 0.2,
            asymmetry_g: 0.0,
            visibility_threshold: 0.05,
        },
        OpticsInputs {
            sin_incidence: 0.3,
            view_depth: 1.5,
            surface_color: RgbColor {
                r: 0.9,
                g: 0.9,
                b: 0.9,
            },
            surface_light: 0.8,
            cos_scatter: 0.1,
            shaft_length: 3.0,
        },
    )
}

/// A deep fixture whose transmittance falls far below the threshold, so geometry
/// is invisible with a comfortable margin.
fn deep_invisible() -> (OpticsProfile, OpticsInputs) {
    (
        OpticsProfile {
            cauchy_a: 1.34,
            cauchy_b: 0.005,
            refraction_strength: 1.0,
            extinction: RgbExtinction {
                r: 0.5,
                g: 0.3,
                b: 0.15,
            },
            scatter_albedo: 0.8,
            asymmetry_g: 0.85,
            visibility_threshold: 0.5,
        },
        OpticsInputs {
            sin_incidence: 0.6,
            view_depth: 40.0,
            surface_color: RgbColor {
                r: 1.0,
                g: 1.0,
                b: 1.0,
            },
            surface_light: 2.0,
            cos_scatter: 0.9,
            shaft_length: 12.0,
        },
    )
}

/// The deterministic hand-picked fixtures.
fn fixtures() -> Vec<(OpticsProfile, OpticsInputs)> {
    vec![
        clear_water(),
        grazing_turbid(),
        isotropic_shallow(),
        deep_invisible(),
    ]
}

/// Builds a randomized fixture well clear of the discrete visibility tie and the
/// clamp edges on `g` and the albedo. The visibility threshold is pinned near zero
/// and the view depth kept shallow so the transmittance stays comfortably above
/// it, keeping the `visible` flag stable across the `CPU`/`GPU` boundary.
fn random_fixture(state: &mut u64) -> (OpticsProfile, OpticsInputs) {
    let profile = OpticsProfile {
        cauchy_a: draw(state, 1.30, 1.36),
        cauchy_b: draw(state, 0.001, 0.008),
        refraction_strength: draw(state, 0.0, 2.5),
        extinction: RgbExtinction {
            r: draw(state, 0.2, 0.7),
            g: draw(state, 0.05, 0.3),
            b: draw(state, 0.02, 0.15),
        },
        scatter_albedo: draw(state, 0.1, 0.9),
        asymmetry_g: draw(state, -0.8, 0.8),
        visibility_threshold: draw(state, 0.001, 0.02),
    };
    let inputs = OpticsInputs {
        sin_incidence: draw(state, 0.0, 1.0),
        view_depth: draw(state, 0.5, 4.0),
        surface_color: RgbColor {
            r: draw(state, 0.3, 1.0),
            g: draw(state, 0.3, 1.0),
            b: draw(state, 0.3, 1.0),
        },
        surface_light: draw(state, 0.3, 2.0),
        cos_scatter: draw(state, -1.0, 1.0),
        shaft_length: draw(state, 1.0, 12.0),
    };
    (profile, inputs)
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_optics_plan parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWaterOpticsPlan::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn clear_water_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterOpticsPlan::new(&ctx);
    check(&ctx, &gpu, &[clear_water()]);
}

#[test]
fn grazing_turbid_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterOpticsPlan::new(&ctx);
    check(&ctx, &gpu, &[grazing_turbid()]);
}

#[test]
fn deep_fixture_is_invisible_and_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterOpticsPlan::new(&ctx);
    let (profile, inputs) = deep_invisible();
    // The fixture must genuinely be past the visibility threshold.
    assert!(
        !plan_optics(profile, inputs).visible,
        "deep fixture must be invisible"
    );
    check(&ctx, &gpu, &[deep_invisible()]);
}

#[test]
fn fixtures_match_golden_in_one_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterOpticsPlan::new(&ctx);
    // Every hand-picked fixture dispatched together exercises the per-thread
    // indexing and the contiguous output slots.
    check(&ctx, &gpu, &fixtures());
}

#[test]
fn random_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterOpticsPlan::new(&ctx);
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
    let mut cases = fixtures();
    // Several workgroups' worth of randomized fixtures pin the fused plan across a
    // wide span of water parameters.
    for _ in 0..200 {
        cases.push(random_fixture(&mut state));
    }
    check(&ctx, &gpu, &cases);
}
