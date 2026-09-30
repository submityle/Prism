//! Real-device parity for the virga-fade twin: [`GpuVirgaFade`] must reproduce
//! the `CPU` golden
//! [`virga_fade`](prism_render_architecture::volumetric::storm::virga_fade)
//! across the full height range, including saturating out-of-range inputs.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The `smoothstep` is expanded to the same closed form the CPU
//! `math::smoothstep` uses, and the kernel contains no transcendental call —
//! saturate and a multiply-add — so `CPU` and `GPU` evaluate the same
//! closed-form algebra. Values are asserted to within `abs_diff < 1e-6` or
//! `rel_diff < 1e-5` — tight enough to fail a wrong port. The scenes also assert
//! the value stays in `0..=1`, is zero at the tip and one at the base, and rises
//! monotonically with height, so a degenerate kernel could not pass.
//!
//! Provenance: standard virga precipitation-veil falloff; no Unreal Engine
//! source or derived code.

use prism_render_architecture::volumetric::storm::virga_fade;
use prism_volumetric_gpu::{GpuContext, GpuVirgaFade, VirgaFadeQuery};

/// Asserts every `gpu` value matches the `CPU` golden to within the documented
/// tolerance and stays in `0..=1`.
fn assert_parity(queries: &[VirgaFadeQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one weight per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = virga_fade(q.height_fraction);
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-6 || rel_diff < 1e-5,
            "virga fade mismatch for query {i} ({q:?}): gpu {got}, cpu {exp} \
             (abs {abs_diff}, rel {rel_diff})"
        );
        assert!(
            (0.0..=1.0).contains(&got),
            "gpu virga fade must stay in 0..=1: {got}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_virga_fade_matches_cpu_golden_across_heights() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping virga fade parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuVirgaFade::new(&ctx);

    // A deterministic height sweep plus out-of-range inputs that must saturate.
    let mut queries: Vec<VirgaFadeQuery> = (0..=200)
        .map(|k| VirgaFadeQuery {
            height_fraction: k as f32 / 200.0,
        })
        .collect();
    queries.push(VirgaFadeQuery {
        height_fraction: -0.5,
    });
    queries.push(VirgaFadeQuery {
        height_fraction: 1.5,
    });

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // The tip is dry, the base is full veil, and the profile rises monotonically.
    assert!(gpu[0].abs() < 1e-6, "the trailing tip is dry: {}", gpu[0]);
    assert!(
        (gpu[200] - 1.0).abs() < 1e-6,
        "the cloud base is full veil: {}",
        gpu[200]
    );
    let mut prev = f32::NEG_INFINITY;
    for &v in gpu.iter().take(201) {
        assert!(
            v >= prev - 1e-6,
            "virga fade must rise with height: {prev} then {v}"
        );
        prev = v;
    }
    // Out-of-range inputs saturate to the endpoints.
    assert!(
        gpu[201].abs() < 1e-6,
        "a height below zero saturates to the dry tip: {}",
        gpu[201]
    );
    assert!(
        (gpu[202] - 1.0).abs() < 1e-6,
        "a height above one saturates to the full base: {}",
        gpu[202]
    );
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuVirgaFade::new(&ctx);
    let out = gpu_kernel.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
