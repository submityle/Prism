//! Real-device parity for the god-ray scattering-mask twin:
//! [`GpuScatteringMask`] must reproduce the `CPU` golden
//! [`scattering_mask`](prism_render_architecture::volumetric::shadow::scattering_mask)
//! across a spread of shadow transmittances and densities.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The mask has no transcendental call — it is two saturates and a single
//! multiply — so `CPU` and `GPU` evaluate the identical closed-form algebra
//! with no room for fma contraction (a lone multiply cannot be fused). Values
//! are asserted to within `abs_diff < 1e-6` or `rel_diff < 1e-5`, tight enough
//! to fail a wrong port (a dropped saturate, a swapped operand). The scenes
//! also assert the documented `[0, 1]` range and the two vanishing boundaries
//! (full shadow and clear air), so a degenerate kernel could not pass.
//!
//! Provenance: standard screen-space god-ray gating; no Unreal Engine source or
//! derived code.

use prism_render_architecture::volumetric::shadow::scattering_mask;
use prism_volumetric_gpu::{GpuContext, GpuScatteringMask, MaskQuery};

/// Asserts every `gpu` value matches the `CPU` golden to within the documented
/// tolerance and stays inside the unit range.
fn assert_parity(queries: &[MaskQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one mask value per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = scattering_mask(q.shadow_transmittance, q.density);
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-6 || rel_diff < 1e-5,
            "mask mismatch for query {i} ({q:?}): gpu {got}, cpu {exp} \
             (abs {abs_diff}, rel {rel_diff})"
        );
        assert!(
            (0.0..=1.0).contains(&got),
            "gpu mask must stay in the unit range: {got}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_mask_matches_cpu_golden_across_queries() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping mask parity: no wgpu adapter on this host");
        return;
    };
    let gpu_mask = GpuScatteringMask::new(&ctx);

    // A deterministic spread: full shadow, clear air, a lit dense fragment, a
    // partial mix, and out-of-range inputs that must clamp into `[0, 1]`.
    let mut queries: Vec<MaskQuery> = vec![
        MaskQuery {
            shadow_transmittance: 0.0,
            density: 1.0,
        },
        MaskQuery {
            shadow_transmittance: 1.0,
            density: 0.0,
        },
        MaskQuery {
            shadow_transmittance: 1.0,
            density: 1.0,
        },
        MaskQuery {
            shadow_transmittance: 0.6,
            density: 0.4,
        },
        MaskQuery {
            shadow_transmittance: 1.7,
            density: 0.5,
        },
        MaskQuery {
            shadow_transmittance: -0.3,
            density: 0.9,
        },
        MaskQuery {
            shadow_transmittance: 0.8,
            density: -2.0,
        },
    ];
    // A deterministic 2D sweep across the unit square.
    for a in 0..12 {
        for b in 0..12 {
            queries.push(MaskQuery {
                shadow_transmittance: (a as f32) / 11.0,
                density: (b as f32) / 11.0,
            });
        }
    }

    let gpu = gpu_mask.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // Full shadow and clear air both vanish; a lit dense fragment is fully lit.
    assert!(gpu[0].abs() < 1e-6, "full shadow vanishes");
    assert!(gpu[1].abs() < 1e-6, "clear air vanishes");
    assert!(
        (gpu[2] - 1.0).abs() < 1e-6,
        "a lit dense fragment is fully masked in"
    );
    // Out-of-range inputs clamp: 1.7 -> 1.0 so mask == density (0.5); negative
    // transmittance and negative density both clamp to a zero product.
    assert!(
        (gpu[4] - 0.5).abs() < 1e-6,
        "over-unit transmittance clamps to one"
    );
    assert!(gpu[5].abs() < 1e-6, "negative transmittance clamps to zero");
    assert!(gpu[6].abs() < 1e-6, "negative density clamps to zero");
}

#[test]
fn gpu_mask_is_monotone_non_decreasing_in_each_factor() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_mask = GpuScatteringMask::new(&ctx);

    // At a fixed density the mask rises with transmittance; at a fixed
    // transmittance it rises with density. Sweep each axis independently.
    let mut rising_t: Vec<MaskQuery> = Vec::new();
    for k in 0..=32 {
        rising_t.push(MaskQuery {
            shadow_transmittance: (k as f32) / 32.0,
            density: 0.7,
        });
    }
    let mut rising_d: Vec<MaskQuery> = Vec::new();
    for k in 0..=32 {
        rising_d.push(MaskQuery {
            shadow_transmittance: 0.7,
            density: (k as f32) / 32.0,
        });
    }

    let gpu_t = gpu_mask.eval(&ctx, &rising_t);
    let gpu_d = gpu_mask.eval(&ctx, &rising_d);
    assert_parity(&rising_t, &gpu_t);
    assert_parity(&rising_d, &gpu_d);

    for w in gpu_t.windows(2) {
        assert!(
            w[1] >= w[0] - 1e-6,
            "mask must rise with transmittance: {} then {}",
            w[0],
            w[1]
        );
    }
    for w in gpu_d.windows(2) {
        assert!(
            w[1] >= w[0] - 1e-6,
            "mask must rise with density: {} then {}",
            w[0],
            w[1]
        );
    }
    assert!(
        gpu_t[gpu_t.len() - 1] > gpu_t[0] + 1e-3,
        "full transmittance masks in strictly more than full shadow"
    );
    assert!(
        gpu_d[gpu_d.len() - 1] > gpu_d[0] + 1e-3,
        "full density masks in strictly more than clear air"
    );
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_mask = GpuScatteringMask::new(&ctx);
    let out = gpu_mask.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
