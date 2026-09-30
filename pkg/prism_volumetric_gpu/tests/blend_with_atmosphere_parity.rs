//! Real-device parity for the aerial-perspective blend twin:
//! [`GpuBlendWithAtmosphere`] must reproduce the `CPU` golden
//! [`blend_with_atmosphere`](prism_render_architecture::volumetric::atmosphere::blend_with_atmosphere)
//! across a deterministic spread of cloud colours, transmittances, in-scatter
//! colours and distance-fade weights, including out-of-range weights that must
//! be clamped.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The `lerp` is expanded to the same closed form the CPU `math` module uses and
//! `saturate` clamps both weights, so `CPU` and `GPU` evaluate identical
//! algebra. Values are asserted per channel to within `abs_diff < 1e-6` or
//! `rel_diff < 1e-5` — tight enough to fail a wrong port (a dropped saturate, a
//! swapped lerp order). The scenes also assert every output channel stays within
//! the closed interval spanned by `cloud_color` and `inscatter` (the
//! energy-conservation bound), so a degenerate kernel could not pass.
//!
//! Provenance: standard aerial-perspective composite; no Unreal Engine source or
//! derived code.

use prism_render_architecture::volumetric::atmosphere::blend_with_atmosphere;
use prism_render_architecture::volumetric::Vec3;
use prism_volumetric_gpu::{BlendQuery, BlendedColor, GpuBlendWithAtmosphere, GpuContext};

/// Asserts every `gpu` colour matches the `CPU` golden per channel to within the
/// documented tolerance and stays within the interval spanned by the cloud
/// colour and in-scatter (energy conservation).
fn assert_parity(queries: &[BlendQuery], gpu: &[BlendedColor]) {
    assert_eq!(gpu.len(), queries.len(), "one blended colour per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = blend_with_atmosphere(
            Vec3::new(q.cloud_color[0], q.cloud_color[1], q.cloud_color[2]),
            q.cloud_transmittance,
            Vec3::new(q.inscatter[0], q.inscatter[1], q.inscatter[2]),
            q.weight,
        );
        let got = gpu[i];
        for (channel, (g, e, lo, hi)) in [
            (got.r, exp.x, q.cloud_color[0], q.inscatter[0]),
            (got.g, exp.y, q.cloud_color[1], q.inscatter[1]),
            (got.b, exp.z, q.cloud_color[2], q.inscatter[2]),
        ]
        .into_iter()
        .enumerate()
        {
            let abs_diff = (g - e).abs();
            let rel_diff = abs_diff / e.abs().max(1e-6);
            assert!(
                abs_diff < 1e-6 || rel_diff < 1e-5,
                "aerial-perspective blend mismatch for query {i} channel {channel}: \
                 gpu {g}, cpu {e} (abs {abs_diff}, rel {rel_diff})"
            );
            let (min_c, max_c) = if lo <= hi { (lo, hi) } else { (hi, lo) };
            assert!(
                g >= min_c - 1e-6 && g <= max_c + 1e-6,
                "blended channel {channel} of query {i} must stay within the \
                 cloud/in-scatter interval [{min_c}, {max_c}]: {g}"
            );
        }
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_blend_with_atmosphere_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping aerial-perspective blend parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuBlendWithAtmosphere::new(&ctx);

    // A deterministic spread of colours, transmittances and weights, including
    // out-of-range weights that must be clamped into 0..=1.
    let mut queries: Vec<BlendQuery> = vec![
        BlendQuery {
            cloud_color: [0.8, 0.7, 0.6],
            cloud_transmittance: 0.0,
            inscatter: [0.2, 0.3, 0.5],
            weight: 0.0,
        },
        BlendQuery {
            cloud_color: [0.8, 0.7, 0.6],
            cloud_transmittance: 1.0,
            inscatter: [0.2, 0.3, 0.5],
            weight: 0.0,
        },
        BlendQuery {
            cloud_color: [0.9, 0.9, 0.9],
            cloud_transmittance: 0.5,
            inscatter: [0.1, 0.2, 0.4],
            weight: 1.0,
        },
        BlendQuery {
            cloud_color: [0.4, 0.5, 0.6],
            cloud_transmittance: 0.3,
            inscatter: [0.7, 0.6, 0.5],
            weight: 0.5,
        },
        BlendQuery {
            cloud_color: [1.0, 0.2, 0.0],
            cloud_transmittance: -0.5,
            inscatter: [0.0, 0.4, 0.9],
            weight: 2.0,
        },
    ];
    // A deterministic parameter sweep over transmittance and weight.
    for k in 0..=40 {
        let t = (k as f32) / 40.0;
        for j in 0..=10 {
            let w = (j as f32) / 10.0;
            queries.push(BlendQuery {
                cloud_color: [0.75, 0.55, 0.35],
                cloud_transmittance: t,
                inscatter: [0.15, 0.35, 0.65],
                weight: w,
            });
        }
    }

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // Zero transmittance and zero weight is the pure cloud colour.
    let pure = gpu[0];
    assert!(
        (pure.r - 0.8).abs() < 1e-6 && (pure.g - 0.7).abs() < 1e-6 && (pure.b - 0.6).abs() < 1e-6,
        "zero transmittance and weight is the pure cloud colour: {pure:?}"
    );
    // Full weight collapses to the in-scatter regardless of the cloud.
    let air = gpu[2];
    assert!(
        (air.r - 0.1).abs() < 1e-6 && (air.g - 0.2).abs() < 1e-6 && (air.b - 0.4).abs() < 1e-6,
        "full weight collapses to the in-scatter: {air:?}"
    );
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuBlendWithAtmosphere::new(&ctx);
    let out = gpu_kernel.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
