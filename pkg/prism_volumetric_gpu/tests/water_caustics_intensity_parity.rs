//! Real-device parity for the water caustic-intensity twin:
//! [`GpuWaterCausticsIntensity`](prism_volumetric_gpu::water_caustics_intensity::GpuWaterCausticsIntensity)
//! must reproduce the `CPU` golden
//! [`jacobian_caustic_gain`](prism_render_architecture::water::caustics::jacobian_caustic_gain),
//! [`project_caustic_intensity`](prism_render_architecture::water::caustics::project_caustic_intensity),
//! and
//! [`photon_splat_density`](prism_render_architecture::water::caustics::photon_splat_density)
//! across the focus/defocus, clamp, projection, and splat-density regimes plus
//! a randomized batch compared value-for-value.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The three golden functions are `pub`, so each `GPU` result is pinned
//! directly against the golden run on the same input.
//!
//! # Parity criterion
//!
//! Every result is a continuous `f32` (a guarded reciprocal, a product, or a
//! splat-area divide), asserted within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`.
//!
//! # Conditioning
//!
//! Fixtures stay away from the two guard thresholds: the Jacobian magnitude and
//! the splat radius are kept well above `EPS = 1e-6` (except the exact-zero
//! degenerate cases, which both sides agree on deterministically), and the
//! reciprocal gain is kept clearly below or clearly above the `max_gain` clamp
//! so a last-bit difference cannot flip which operand the final `min` selects.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::caustics`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::caustics::{
    jacobian_caustic_gain, photon_splat_density, project_caustic_intensity,
};
use prism_volumetric_gpu::water_caustics_intensity::{
    GpuWaterCausticsIntensity, WaterCausticsIntensityQuery, WaterCausticsIntensityResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the intensity.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes.
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

/// Computes the golden result for one query, so the oracle lives beside the
/// device call and both read the same input.
fn expected(q: &WaterCausticsIntensityQuery) -> WaterCausticsIntensityResult {
    let value = match *q {
        WaterCausticsIntensityQuery::JacobianGain { jacobian, max_gain } => {
            jacobian_caustic_gain(jacobian, max_gain)
        }
        WaterCausticsIntensityQuery::ProjectIntensity {
            incident,
            jacobian,
            max_gain,
        } => project_caustic_intensity(incident, jacobian, max_gain),
        WaterCausticsIntensityQuery::PhotonDensity {
            photon_count,
            photon_power,
            radius,
        } => photon_splat_density(photon_count, photon_power, radius),
    };
    WaterCausticsIntensityResult { value }
}

/// Pins one `GPU` result against the golden oracle within tolerance.
fn assert_result(
    idx: usize,
    got: &WaterCausticsIntensityResult,
    want: &WaterCausticsIntensityResult,
) {
    assert!(
        close(got.value, want.value),
        "result {idx} value: gpu {} vs cpu {}",
        got.value,
        want.value
    );
}

/// Runs every query on the device and pins each result against the oracle.
fn run_and_check(ctx: &GpuContext, queries: &[WaterCausticsIntensityQuery]) {
    let gpu = GpuWaterCausticsIntensity::new(ctx);
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (idx, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        assert_result(idx, g, &expected(q));
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

/// Draws a float in `[0, 1)` from `state` using only integer work.
fn unit(state: &mut u64) -> f32 {
    (lcg(state) >> 8) as f32 / (1u32 << 24) as f32
}

/// Draws a float in `[lo, hi)` from `state`.
fn ranged(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * unit(state)
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_caustics_intensity parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWaterCausticsIntensity::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn jacobian_gain_fixtures() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let queries = vec![
        // Converging patch brightens: 1/0.25 = 4, far below the cap.
        WaterCausticsIntensityQuery::JacobianGain {
            jacobian: 0.25,
            max_gain: 100.0,
        },
        // Diverging patch dims: 1/4 = 0.25.
        WaterCausticsIntensityQuery::JacobianGain {
            jacobian: 4.0,
            max_gain: 100.0,
        },
        // Sign of the Jacobian is irrelevant: 1/0.5 = 2.
        WaterCausticsIntensityQuery::JacobianGain {
            jacobian: -0.5,
            max_gain: 100.0,
        },
        // Exact-zero Jacobian saturates to the cap (both sides deterministic).
        WaterCausticsIntensityQuery::JacobianGain {
            jacobian: 0.0,
            max_gain: 50.0,
        },
        // Clamp bites: 1/0.01 = 100 exceeds the cap of 10, so it saturates.
        WaterCausticsIntensityQuery::JacobianGain {
            jacobian: 0.01,
            max_gain: 10.0,
        },
    ];
    run_and_check(&ctx, &queries);
}

#[test]
fn project_intensity_fixtures() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let queries = vec![
        // 2 incident * gain(1/0.5 = 2) = 4.
        WaterCausticsIntensityQuery::ProjectIntensity {
            incident: 2.0,
            jacobian: 0.5,
            max_gain: 10.0,
        },
        // Zero incident light yields zero caustics regardless of gain.
        WaterCausticsIntensityQuery::ProjectIntensity {
            incident: 0.0,
            jacobian: 0.1,
            max_gain: 10.0,
        },
        // Negative incident sanitizes to zero.
        WaterCausticsIntensityQuery::ProjectIntensity {
            incident: -3.0,
            jacobian: 0.5,
            max_gain: 10.0,
        },
        // Clamp bounds the projection: 3 incident * cap(5) = 15.
        WaterCausticsIntensityQuery::ProjectIntensity {
            incident: 3.0,
            jacobian: 0.02,
            max_gain: 5.0,
        },
    ];
    run_and_check(&ctx, &queries);
}

#[test]
fn photon_density_fixtures() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let queries = vec![
        // Dense gather: 1000 photons of unit power over PI * 0.25.
        WaterCausticsIntensityQuery::PhotonDensity {
            photon_count: 1000,
            photon_power: 1.0,
            radius: 0.5,
        },
        // Wider gather spreads the same energy and dims the estimate.
        WaterCausticsIntensityQuery::PhotonDensity {
            photon_count: 1000,
            photon_power: 1.0,
            radius: 2.0,
        },
        // Zero photons yield zero irradiance.
        WaterCausticsIntensityQuery::PhotonDensity {
            photon_count: 0,
            photon_power: 1.0,
            radius: 0.5,
        },
        // Degenerate radius is inert rather than a division by zero.
        WaterCausticsIntensityQuery::PhotonDensity {
            photon_count: 1000,
            photon_power: 1.0,
            radius: 0.0,
        },
        // Negative power sanitizes to zero.
        WaterCausticsIntensityQuery::PhotonDensity {
            photon_count: 500,
            photon_power: -2.0,
            radius: 1.0,
        },
    ];
    run_and_check(&ctx, &queries);
}

#[test]
fn random_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let mut state = 0x2f6e_41b8_90c3_57a2_u64;

    let mut queries: Vec<WaterCausticsIntensityQuery> = Vec::new();
    while queries.len() < 256 {
        let op = lcg(&mut state) % 3;
        // Jacobian magnitude kept in [0.1, 5] (well above EPS); random sign.
        let mag = ranged(&mut state, 0.1, 5.0);
        let sign = if (lcg(&mut state) & 1) == 0 {
            1.0
        } else {
            -1.0
        };
        let jacobian = sign * mag;
        // Cap kept large so the reciprocal gain (in [0.2, 10]) never ties it.
        let max_gain = ranged(&mut state, 50.0, 200.0);
        let incident = ranged(&mut state, 0.0, 5.0);
        // Radius kept in [0.2, 3] (well above EPS), so the divide never ties.
        let radius = ranged(&mut state, 0.2, 3.0);
        let photon_power = ranged(&mut state, 0.0, 4.0);
        let photon_count = lcg(&mut state) % 5000;
        let query = if op == 0 {
            WaterCausticsIntensityQuery::JacobianGain { jacobian, max_gain }
        } else if op == 1 {
            WaterCausticsIntensityQuery::ProjectIntensity {
                incident,
                jacobian,
                max_gain,
            }
        } else {
            WaterCausticsIntensityQuery::PhotonDensity {
                photon_count,
                photon_power,
                radius,
            }
        };
        queries.push(query);
    }
    run_and_check(&ctx, &queries);
}
