//! Real-device parity for the octave-scatter twin: [`GpuOctaveScatter`] must
//! reproduce the `CPU` golden
//! [`octave_scatter`](prism_render_architecture::volumetric::scatter::octave_scatter)
//! for every query across the octave index range, several base coefficients and
//! more than one geometric schedule.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each output is a base coefficient times a geometric factor raised to the
//! octave index by the same repeated-multiply loop as the reference, with no
//! reorderable summation, so `CPU` and `GPU` evaluate the identical product
//! sequence. Values are asserted to within `abs_diff < 1e-6` or
//! `rel_diff < 1e-5` — far tighter than any physically meaningful decay
//! difference and enough to fail a wrong port (a swapped factor, a missing
//! index clamp, a dropped saturation). The scenes also assert the energy-decay
//! monotonicity (each octave is no larger than the previous) so a degenerate
//! constant kernel could not pass.
//!
//! Provenance: standard Wrenninge-style octave multiple-scattering decay; no
//! Unreal Engine source or derived code.

use prism_render_architecture::volumetric::scatter::{octave_scatter, OctaveParams};
use prism_volumetric_gpu::{GpuContext, GpuOctaveScatter, OctaveQuery, OctaveResult};

/// Asserts every `gpu` result matches the `CPU` golden to within the documented
/// tolerance.
fn assert_parity(params: OctaveParams, queries: &[OctaveQuery], gpu: &[OctaveResult]) {
    assert_eq!(gpu.len(), queries.len(), "one result per query");
    for (i, q) in queries.iter().enumerate() {
        let (sigma_s, sigma_t, g) = octave_scatter(
            q.base_sigma_s,
            q.base_sigma_t,
            q.base_g,
            q.octave_index,
            params,
        );
        let got = gpu[i];
        for (got_v, exp_v, name) in [
            (got.sigma_s, sigma_s, "sigma_s"),
            (got.sigma_t, sigma_t, "sigma_t"),
            (got.g, g, "g"),
        ] {
            let abs_diff = (got_v - exp_v).abs();
            let rel_diff = abs_diff / exp_v.abs().max(1e-6);
            assert!(
                abs_diff < 1e-6 || rel_diff < 1e-5,
                "{name} mismatch for query {q:?}: gpu {got_v}, cpu {exp_v} (abs {abs_diff}, rel {rel_diff})"
            );
        }
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_octave_matches_cpu_golden_across_octaves() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping octave parity: no wgpu adapter on this host");
        return;
    };
    let scatter = GpuOctaveScatter::new(&ctx);
    let params = OctaveParams::DEFAULT;

    // Every octave index of the default four-octave schedule, plus one beyond
    // the last (must clamp to octave 3), on one base coefficient set.
    let queries: Vec<OctaveQuery> = (0..=4)
        .map(|i| OctaveQuery {
            base_sigma_s: 0.9,
            base_sigma_t: 1.2,
            base_g: 0.8,
            octave_index: i,
        })
        .collect();

    let gpu = scatter.eval(&ctx, params, &queries);
    assert_parity(params, &queries, &gpu);

    // Energy-decay monotonicity across the first four octaves: sigma_s, sigma_t
    // and |g| must be non-increasing, proving a real per-octave decay ran.
    for w in gpu.windows(2).take(3) {
        assert!(
            w[1].sigma_s <= w[0].sigma_s + 1e-6,
            "sigma_s must not grow across octaves: {:?} -> {:?}",
            w[0],
            w[1]
        );
        assert!(
            w[1].sigma_t <= w[0].sigma_t + 1e-6,
            "sigma_t must not grow across octaves"
        );
        assert!(
            w[1].g.abs() <= w[0].g.abs() + 1e-6,
            "|g| must not grow across octaves"
        );
    }
    // The clamped out-of-range index must equal the last real octave.
    assert_eq!(
        gpu[4], gpu[3],
        "octave index beyond the schedule must clamp to the last octave"
    );
}

#[test]
fn gpu_octave_matches_cpu_golden_across_schedules_and_bases() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let scatter = GpuOctaveScatter::new(&ctx);

    // A steeper, longer schedule with asymmetric per-quantity decay.
    let params = OctaveParams {
        attenuation: 0.7,
        contribution: 0.4,
        eccentricity_attenuation: 0.3,
        octave_count: 6,
    };

    let queries: Vec<OctaveQuery> = vec![
        OctaveQuery {
            base_sigma_s: 2.0,
            base_sigma_t: 0.5,
            base_g: -0.4,
            octave_index: 0,
        },
        OctaveQuery {
            base_sigma_s: 0.1,
            base_sigma_t: 3.3,
            base_g: 0.95,
            octave_index: 2,
        },
        OctaveQuery {
            base_sigma_s: 1.0,
            base_sigma_t: 1.0,
            base_g: 0.0,
            octave_index: 5,
        },
        // Out-of-range base arguments the reference clamps: negative
        // coefficients and |g| beyond the anisotropy bound.
        OctaveQuery {
            base_sigma_s: -1.0,
            base_sigma_t: -2.0,
            base_g: 1.5,
            octave_index: 1,
        },
    ];

    let gpu = scatter.eval(&ctx, params, &queries);
    assert_parity(params, &queries, &gpu);
}

#[test]
fn gpu_octave_handles_zero_octave_count() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let scatter = GpuOctaveScatter::new(&ctx);
    // `octave_count == 0` must collapse to octave 0 (index clamps to 0), so the
    // result is the undecayed base, matching the reference.
    let params = OctaveParams {
        attenuation: 0.5,
        contribution: 0.5,
        eccentricity_attenuation: 0.5,
        octave_count: 0,
    };
    let queries = vec![OctaveQuery {
        base_sigma_s: 0.8,
        base_sigma_t: 1.1,
        base_g: 0.6,
        octave_index: 3,
    }];
    let gpu = scatter.eval(&ctx, params, &queries);
    assert_parity(params, &queries, &gpu);
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let scatter = GpuOctaveScatter::new(&ctx);
    let out = scatter.eval(&ctx, OctaveParams::DEFAULT, &[]);
    assert!(out.is_empty(), "an empty query slice yields no results");
}
