//! Real-device parity tests for the split-sum environment-`BRDF` (`DFG`)
//! analytic twin
//! ([`env_brdf_split_sum`](prism_volumetric_gpu::env_brdf_split_sum)).
//!
//! Each test uploads a batch of [`EnvBrdfSplitSumQuery`] values, runs the single
//! op-dispatch `solve` kernel on a real `wgpu` device, and compares every lane
//! against the `CPU` golden [`cpu_reference`] which delegates straight to the
//! published analytic routines of
//! [`env_brdf_split_sum`](prism_render_architecture::particle::env_brdf_split_sum).
//! The fixtures cover: the `2^(-9.28·NoV)` grazing falloff at head-on, grazing
//! and interior view angles; the Lazarov / Karis `(scale, bias)` fit for a
//! smooth mirror, a grazing silhouette and a mid-roughness surface; the scalar
//! and tinted-`RGB` specular reconstruction; the roughness-to-`mip`-`LOD` map
//! across a multi-level chain and the degenerate one- and zero-level chains; a
//! mixed-variant batch that pins input ordering; and a large pseudo-random batch
//! over all five operations compared lane for lane.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every twinned routine is fixed closed-form algebra (a short polynomial plus
//! a transcendental-free `2^(-9.28·NoV)` surrogate recovered by integer-power
//! squaring), so `CPU` and `GPU` evaluate the same expression. They are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The comparison therefore allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on
//! every continuous output. Every output of this twin is continuous, so no legal
//! `ULP` perturbation can flip a result variant; fixtures nonetheless stay clear
//! of the `clamp01` range edges and the `mip_count <= 1` level-count edge is
//! exercised only with exact integer level counts.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::env_brdf_split_sum`；
//! Lazarov / Karis split-sum environment-`BRDF` analytic fit (Dimitar Lazarov,
//! *Physically Based Lighting in Call of Duty: Black Ops 2*, 2013; Brian Karis
//! mobile `PBR` notes); no third-party engine source or derived code.

use prism_volumetric_gpu::env_brdf_split_sum::{
    cpu_reference, EnvBrdfSplitSumQuery, EnvBrdfSplitSumResult, GpuEnvBrdfSplitSum,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on continuous values. A `GPU` may fuse a multiply-add
/// the scalar reference leaves separate, perturbing the low mantissa bits by a
/// few units in the last place; `1e-4` admits that legal slack while still
/// failing a genuinely wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
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

/// Asserts strict lane-for-lane parity of one `GPU` result against the `CPU`
/// golden: the result variant must match and every continuous value must agree
/// within tolerance.
fn assert_lane(lane: usize, got: &EnvBrdfSplitSumResult, want: &EnvBrdfSplitSumResult) {
    match (got, want) {
        (
            EnvBrdfSplitSumResult::Scalar { value: sg },
            EnvBrdfSplitSumResult::Scalar { value: sc },
        ) => {
            assert!(close(*sg, *sc), "lane {lane}: scalar gpu {sg} vs cpu {sc}");
        }
        (
            EnvBrdfSplitSumResult::Terms {
                scale: scg,
                bias: bg,
            },
            EnvBrdfSplitSumResult::Terms {
                scale: scc,
                bias: bc,
            },
        ) => {
            assert!(
                close(*scg, *scc),
                "lane {lane}: scale gpu {scg} vs cpu {scc}"
            );
            assert!(close(*bg, *bc), "lane {lane}: bias gpu {bg} vs cpu {bc}");
        }
        (EnvBrdfSplitSumResult::Vector { v: vg }, EnvBrdfSplitSumResult::Vector { v: vc }) => {
            for ch in 0..3 {
                assert!(
                    close(vg[ch], vc[ch]),
                    "lane {lane}: vector[{ch}] gpu {vg:?} vs cpu {vc:?}"
                );
            }
        }
        _ => panic!("lane {lane}: result variant mismatch gpu {got:?} vs cpu {want:?}"),
    }
}

/// Runs the `GPU` dispatch and asserts strict parity against the `CPU` golden
/// for every lane; returns the `GPU` verdicts for extra per-test assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuEnvBrdfSplitSum,
    queries: &[EnvBrdfSplitSumQuery],
) -> Vec<EnvBrdfSplitSumResult> {
    let got = gpu.eval(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        assert_lane(lane, g, &cpu_reference(q));
    }
    got
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// A pseudo-random `f32` in `[lo, hi)`, derived from the integer `lcg`.
fn ranged(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * lcg(state)
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEnvBrdfSplitSum::new(&ctx);
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn exp2_grazing_falloff_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEnvBrdfSplitSum::new(&ctx);
    let queries = [
        // Head-on: the falloff is strongest (smallest value).
        EnvBrdfSplitSumQuery::Exp2GrazingFalloff { n_dot_v: 1.0 },
        // Grazing: the falloff is unity.
        EnvBrdfSplitSumQuery::Exp2GrazingFalloff { n_dot_v: 0.0 },
        // Interior view angles.
        EnvBrdfSplitSumQuery::Exp2GrazingFalloff { n_dot_v: 0.25 },
        EnvBrdfSplitSumQuery::Exp2GrazingFalloff { n_dot_v: 0.5 },
        EnvBrdfSplitSumQuery::Exp2GrazingFalloff { n_dot_v: 0.8 },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn env_brdf_approx_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEnvBrdfSplitSum::new(&ctx);
    let queries = [
        // Smooth mirror, head-on: scale ~ 1, bias ~ 0.
        EnvBrdfSplitSumQuery::EnvBrdfApprox {
            n_dot_v: 0.98,
            roughness: 0.02,
        },
        // Grazing silhouette on a smooth surface: the integrated Fresnel edge.
        EnvBrdfSplitSumQuery::EnvBrdfApprox {
            n_dot_v: 0.05,
            roughness: 0.1,
        },
        // Mid-roughness, mid-angle (clear of the min() crossover).
        EnvBrdfSplitSumQuery::EnvBrdfApprox {
            n_dot_v: 0.6,
            roughness: 0.5,
        },
        // Rough surface, interior angle.
        EnvBrdfSplitSumQuery::EnvBrdfApprox {
            n_dot_v: 0.35,
            roughness: 0.85,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn env_brdf_specular_scalar_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEnvBrdfSplitSum::new(&ctx);
    let queries = [
        // Common dielectric F0 ~ 0.04.
        EnvBrdfSplitSumQuery::EnvBrdfSpecular {
            n_dot_v: 0.7,
            roughness: 0.4,
            f0: 0.04,
        },
        // Higher reflectance, grazing.
        EnvBrdfSplitSumQuery::EnvBrdfSpecular {
            n_dot_v: 0.15,
            roughness: 0.25,
            f0: 0.5,
        },
        // Smooth, head-on.
        EnvBrdfSplitSumQuery::EnvBrdfSpecular {
            n_dot_v: 0.95,
            roughness: 0.08,
            f0: 0.9,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn env_brdf_specular_rgb_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEnvBrdfSplitSum::new(&ctx);
    let queries = [
        // Tinted metal F0 — each channel scaled independently, shared bias.
        EnvBrdfSplitSumQuery::EnvBrdfSpecularRgb {
            n_dot_v: 0.6,
            roughness: 0.5,
            f0: [0.9, 0.6, 0.3],
        },
        // Grayscale F0 — every channel equals the scalar reconstruction.
        EnvBrdfSplitSumQuery::EnvBrdfSpecularRgb {
            n_dot_v: 0.7,
            roughness: 0.4,
            f0: [0.08, 0.08, 0.08],
        },
        // Copper-like tint, interior angle.
        EnvBrdfSplitSumQuery::EnvBrdfSpecularRgb {
            n_dot_v: 0.45,
            roughness: 0.3,
            f0: [0.95, 0.64, 0.54],
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn prefilter_mip_lod_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEnvBrdfSplitSum::new(&ctx);
    let queries = [
        // A 5-level chain maps roughness linearly onto LOD 0..=4.
        EnvBrdfSplitSumQuery::PrefilterMipLod {
            roughness: 0.0,
            mip_count: 5,
        },
        EnvBrdfSplitSumQuery::PrefilterMipLod {
            roughness: 0.5,
            mip_count: 5,
        },
        EnvBrdfSplitSumQuery::PrefilterMipLod {
            roughness: 1.0,
            mip_count: 5,
        },
        EnvBrdfSplitSumQuery::PrefilterMipLod {
            roughness: 0.3,
            mip_count: 8,
        },
        // Degenerate chains always sample LOD 0.
        EnvBrdfSplitSumQuery::PrefilterMipLod {
            roughness: 0.7,
            mip_count: 1,
        },
        EnvBrdfSplitSumQuery::PrefilterMipLod {
            roughness: 0.7,
            mip_count: 0,
        },
    ];
    let got = check(&ctx, &gpu, &queries);
    // The degenerate one- and zero-level chains both pin LOD 0 exactly.
    assert!(matches!(
        got[4],
        EnvBrdfSplitSumResult::Scalar { value } if value.abs() <= EPS
    ));
    assert!(matches!(
        got[5],
        EnvBrdfSplitSumResult::Scalar { value } if value.abs() <= EPS
    ));
}

#[test]
fn mixed_variant_batch_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEnvBrdfSplitSum::new(&ctx);
    let queries = [
        EnvBrdfSplitSumQuery::Exp2GrazingFalloff { n_dot_v: 0.4 },
        EnvBrdfSplitSumQuery::PrefilterMipLod {
            roughness: 0.25,
            mip_count: 6,
        },
        EnvBrdfSplitSumQuery::EnvBrdfApprox {
            n_dot_v: 0.55,
            roughness: 0.45,
        },
        EnvBrdfSplitSumQuery::EnvBrdfSpecularRgb {
            n_dot_v: 0.3,
            roughness: 0.6,
            f0: [0.7, 0.5, 0.2],
        },
        EnvBrdfSplitSumQuery::EnvBrdfSpecular {
            n_dot_v: 0.65,
            roughness: 0.35,
            f0: 0.12,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_batch_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEnvBrdfSplitSum::new(&ctx);
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut queries = Vec::with_capacity(256);
    for _ in 0..256 {
        // Select one of the five operations by an integer draw.
        let op = (lcg(&mut state) * 5.0) as u32 % 5;
        let n_dot_v = ranged(&mut state, 0.02, 0.98);
        let roughness = ranged(&mut state, 0.02, 0.98);
        let query = match op {
            0 => EnvBrdfSplitSumQuery::Exp2GrazingFalloff { n_dot_v },
            1 => EnvBrdfSplitSumQuery::EnvBrdfApprox { n_dot_v, roughness },
            2 => EnvBrdfSplitSumQuery::EnvBrdfSpecular {
                n_dot_v,
                roughness,
                f0: ranged(&mut state, 0.02, 0.98),
            },
            3 => EnvBrdfSplitSumQuery::EnvBrdfSpecularRgb {
                n_dot_v,
                roughness,
                f0: [
                    ranged(&mut state, 0.02, 0.98),
                    ranged(&mut state, 0.02, 0.98),
                    ranged(&mut state, 0.02, 0.98),
                ],
            },
            // A multi-level chain (`mip_count >= 2`) stays in the linear branch.
            _ => EnvBrdfSplitSumQuery::PrefilterMipLod {
                roughness,
                mip_count: 2 + (lcg(&mut state) * 7.0) as u32,
            },
        };
        queries.push(query);
    }
    check(&ctx, &gpu, &queries);
}
