//! Real-device parity for the anisotropic-base-plus-clearcoat twin:
//! [`GpuAnisotropicClearcoat`](prism_volumetric_gpu::anisotropic_clearcoat::GpuAnisotropicClearcoat)
//! must reproduce the `CPU` golden
//! [`anisotropic_clearcoat`](prism_render_architecture::particle::anisotropic_clearcoat)
//! across every reflectance term it exposes: the anisotropy-to-aspect mapping
//! ([`aspect_ratio`](prism_render_architecture::particle::anisotropic_clearcoat::aspect_ratio),
//! [`anisotropic_alphas`](prism_render_architecture::particle::anisotropic_clearcoat::anisotropic_alphas)),
//! the anisotropic `GGX` base lobe
//! ([`ggx_aniso_ndf`](prism_render_architecture::particle::anisotropic_clearcoat::ggx_aniso_ndf),
//! [`ggx_aniso_visibility`](prism_render_architecture::particle::anisotropic_clearcoat::ggx_aniso_visibility)),
//! the isotropic clearcoat lobe
//! ([`clearcoat_ggx_ndf`](prism_render_architecture::particle::anisotropic_clearcoat::clearcoat_ggx_ndf),
//! [`clearcoat_visibility`](prism_render_architecture::particle::anisotropic_clearcoat::clearcoat_visibility)),
//! the `Schlick` clearcoat `Fresnel` and its base attenuation
//! ([`clearcoat_fresnel`](prism_render_architecture::particle::anisotropic_clearcoat::clearcoat_fresnel),
//! [`clearcoat_attenuation`](prism_render_architecture::particle::anisotropic_clearcoat::clearcoat_attenuation)),
//! and the composite
//! [`AnisotropicClearcoatParams::evaluate`](prism_render_architecture::particle::anisotropic_clearcoat::AnisotropicClearcoatParams::evaluate).
//!
//! The fixtures exercise one dedicated query per term plus a randomized mixed
//! batch compared element for element. Every direction is an exactly normalized
//! vector and every `roughness`, `anisotropy` and cosine is an interior value
//! held well away from the `0`/`1` boundaries, so no fixture samples a grazing
//! angle and none sits on a guard branch tie.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each term is a fixed, non-reorderable sequence of multiplies, adds, divides
//! and (for the aspect ratio and the `Smith` `Lambda`) one `sqrt`, so `CPU` and
//! `GPU` evaluate the same closed form in the same order. They are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The comparison therefore allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3`
//! (`REL_FLOOR = 1e-6`) on every `f32` lane.
//!
//! # Conditioning
//!
//! Every fixture is kept clear of the guard cracks: alphas are interior
//! (`~0.1..=0.5`) so no `GGX` denominator approaches `MIN_DENOM` and no alpha
//! floors at `MIN_ALPHA`; cosines are held comfortably positive so the `Smith`
//! sum never nears zero; and the composite `evaluate` directions all point into
//! the upper hemisphere so the half vector and clamped cosines agree on both
//! devices regardless of a few units in the last place of slack.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::anisotropic_clearcoat`；
//! 无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::anisotropic_clearcoat::{
    anisotropic_alphas, aspect_ratio, clearcoat_attenuation, clearcoat_fresnel, clearcoat_ggx_ndf,
    clearcoat_visibility, ggx_aniso_ndf, ggx_aniso_visibility, AnisotropicClearcoatParams,
};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::anisotropic_clearcoat::{
    AnisotropicClearcoatQuery, AnisotropicClearcoatResult, GpuAnisotropicClearcoat,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes (such as a sharp `NDF`
/// peak) where a few units in the last place exceed the absolute floor.
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

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// A pseudo-random value in `[-span, span)` drawn from `state`.
fn signed(state: &mut u64, span: f32) -> f32 {
    (lcg(state) * 2.0 - 1.0) * span
}

/// A pseudo-random value in `[lo, hi)` drawn from `state`.
fn ranged(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + lcg(state) * (hi - lo)
}

/// Builds an exactly normalized direction from raw components.
fn unit(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z).normalize_or_zero()
}

/// Converts a packed component triple into a [`Vec3`].
fn v3(c: [f32; 3]) -> Vec3 {
    Vec3::new(c[0], c[1], c[2])
}

/// A pseudo-random, exactly normalized upper-hemisphere direction whose `z`
/// stays above `0.4`, so against the `+z` normal both clamped cosines are
/// comfortably positive and no fixture grazes the horizon. Uses only
/// `normalize_or_zero` (one `sqrt`), so no transcendental method is called.
fn rand_upper(state: &mut u64) -> [f32; 3] {
    loop {
        let v = Vec3::new(
            signed(state, 1.0),
            signed(state, 1.0),
            ranged(state, 0.4, 1.0),
        );
        if v.length_squared() > 0.3 {
            let n = v.normalize_or_zero();
            if n.z > 0.4 {
                return [n.x, n.y, n.z];
            }
        }
    }
}

/// Recomputes the expected [`AnisotropicClearcoatResult`] by calling the `CPU`
/// golden directly for `query`.
fn golden_result(query: &AnisotropicClearcoatQuery) -> AnisotropicClearcoatResult {
    match *query {
        AnisotropicClearcoatQuery::AspectRatio { anisotropy } => {
            AnisotropicClearcoatResult::AspectRatio {
                aspect: aspect_ratio(anisotropy),
            }
        }
        AnisotropicClearcoatQuery::AnisotropicAlphas {
            roughness,
            anisotropy,
        } => {
            let alphas = anisotropic_alphas(roughness, anisotropy);
            AnisotropicClearcoatResult::AnisotropicAlphas {
                alpha_t: alphas.alpha_t,
                alpha_b: alphas.alpha_b,
            }
        }
        AnisotropicClearcoatQuery::GgxAnisoNdf {
            n_dot_h,
            t_dot_h,
            b_dot_h,
            alpha_t,
            alpha_b,
        } => AnisotropicClearcoatResult::GgxAnisoNdf {
            ndf: ggx_aniso_ndf(n_dot_h, t_dot_h, b_dot_h, alpha_t, alpha_b),
        },
        AnisotropicClearcoatQuery::GgxAnisoVisibility {
            alpha_t,
            alpha_b,
            t_dot_v,
            b_dot_v,
            n_dot_v,
            t_dot_l,
            b_dot_l,
            n_dot_l,
        } => AnisotropicClearcoatResult::GgxAnisoVisibility {
            visibility: ggx_aniso_visibility(
                alpha_t, alpha_b, t_dot_v, b_dot_v, n_dot_v, t_dot_l, b_dot_l, n_dot_l,
            ),
        },
        AnisotropicClearcoatQuery::ClearcoatGgxNdf { n_dot_h, alpha } => {
            AnisotropicClearcoatResult::ClearcoatGgxNdf {
                ndf: clearcoat_ggx_ndf(n_dot_h, alpha),
            }
        }
        AnisotropicClearcoatQuery::ClearcoatVisibility {
            n_dot_v,
            n_dot_l,
            alpha,
        } => AnisotropicClearcoatResult::ClearcoatVisibility {
            visibility: clearcoat_visibility(n_dot_v, n_dot_l, alpha),
        },
        AnisotropicClearcoatQuery::ClearcoatFresnel { cos_theta } => {
            AnisotropicClearcoatResult::ClearcoatFresnel {
                fresnel: clearcoat_fresnel(cos_theta),
            }
        }
        AnisotropicClearcoatQuery::ClearcoatAttenuation { fc } => {
            AnisotropicClearcoatResult::ClearcoatAttenuation {
                attenuation: clearcoat_attenuation(fc),
            }
        }
        AnisotropicClearcoatQuery::Evaluate {
            roughness,
            anisotropy,
            clearcoat_roughness,
            clearcoat_strength,
            tangent,
            bitangent,
            normal,
            view,
            light,
        } => {
            let params = AnisotropicClearcoatParams::new(
                roughness,
                anisotropy,
                clearcoat_roughness,
                clearcoat_strength,
            );
            let s = params.evaluate(v3(tangent), v3(bitangent), v3(normal), v3(view), v3(light));
            AnisotropicClearcoatResult::Evaluate {
                base_ndf: s.base_ndf,
                base_visibility: s.base_visibility,
                base_attenuation: s.base_attenuation,
                clearcoat_ndf: s.clearcoat_ndf,
                clearcoat_visibility: s.clearcoat_visibility,
                clearcoat_fresnel: s.clearcoat_fresnel,
            }
        }
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: each lane the
/// variant carries must agree within the parity bound.
fn pin(idx: usize, query: &AnisotropicClearcoatQuery, got: &AnisotropicClearcoatResult) {
    let want = golden_result(query);
    match (got, want) {
        (
            AnisotropicClearcoatResult::AspectRatio { aspect: g },
            AnisotropicClearcoatResult::AspectRatio { aspect: w },
        ) => {
            assert!(close(*g, w), "query {idx} aspect_ratio: gpu {g} vs cpu {w}");
        }
        (
            AnisotropicClearcoatResult::AnisotropicAlphas {
                alpha_t: gt,
                alpha_b: gb,
            },
            AnisotropicClearcoatResult::AnisotropicAlphas {
                alpha_t: wt,
                alpha_b: wb,
            },
        ) => {
            assert!(close(*gt, wt), "query {idx} alpha_t: gpu {gt} vs cpu {wt}");
            assert!(close(*gb, wb), "query {idx} alpha_b: gpu {gb} vs cpu {wb}");
        }
        (
            AnisotropicClearcoatResult::GgxAnisoNdf { ndf: g },
            AnisotropicClearcoatResult::GgxAnisoNdf { ndf: w },
        ) => {
            assert!(
                close(*g, w),
                "query {idx} ggx_aniso_ndf: gpu {g} vs cpu {w}"
            );
        }
        (
            AnisotropicClearcoatResult::GgxAnisoVisibility { visibility: g },
            AnisotropicClearcoatResult::GgxAnisoVisibility { visibility: w },
        ) => {
            assert!(
                close(*g, w),
                "query {idx} ggx_aniso_visibility: gpu {g} vs cpu {w}"
            );
        }
        (
            AnisotropicClearcoatResult::ClearcoatGgxNdf { ndf: g },
            AnisotropicClearcoatResult::ClearcoatGgxNdf { ndf: w },
        ) => {
            assert!(
                close(*g, w),
                "query {idx} clearcoat_ggx_ndf: gpu {g} vs cpu {w}"
            );
        }
        (
            AnisotropicClearcoatResult::ClearcoatVisibility { visibility: g },
            AnisotropicClearcoatResult::ClearcoatVisibility { visibility: w },
        ) => {
            assert!(
                close(*g, w),
                "query {idx} clearcoat_visibility: gpu {g} vs cpu {w}"
            );
        }
        (
            AnisotropicClearcoatResult::ClearcoatFresnel { fresnel: g },
            AnisotropicClearcoatResult::ClearcoatFresnel { fresnel: w },
        ) => {
            assert!(
                close(*g, w),
                "query {idx} clearcoat_fresnel: gpu {g} vs cpu {w}"
            );
        }
        (
            AnisotropicClearcoatResult::ClearcoatAttenuation { attenuation: g },
            AnisotropicClearcoatResult::ClearcoatAttenuation { attenuation: w },
        ) => {
            assert!(
                close(*g, w),
                "query {idx} clearcoat_attenuation: gpu {g} vs cpu {w}"
            );
        }
        (
            AnisotropicClearcoatResult::Evaluate {
                base_ndf: gn,
                base_visibility: gv,
                base_attenuation: ga,
                clearcoat_ndf: gcn,
                clearcoat_visibility: gcv,
                clearcoat_fresnel: gcf,
            },
            AnisotropicClearcoatResult::Evaluate {
                base_ndf: wn,
                base_visibility: wv,
                base_attenuation: wa,
                clearcoat_ndf: wcn,
                clearcoat_visibility: wcv,
                clearcoat_fresnel: wcf,
            },
        ) => {
            assert!(
                close(*gn, wn),
                "query {idx} evaluate base_ndf: gpu {gn} vs cpu {wn}"
            );
            assert!(
                close(*gv, wv),
                "query {idx} evaluate base_visibility: gpu {gv} vs cpu {wv}"
            );
            assert!(
                close(*ga, wa),
                "query {idx} evaluate base_attenuation: gpu {ga} vs cpu {wa}"
            );
            assert!(
                close(*gcn, wcn),
                "query {idx} evaluate clearcoat_ndf: gpu {gcn} vs cpu {wcn}"
            );
            assert!(
                close(*gcv, wcv),
                "query {idx} evaluate clearcoat_visibility: gpu {gcv} vs cpu {wcv}"
            );
            assert!(
                close(*gcf, wcf),
                "query {idx} evaluate clearcoat_fresnel: gpu {gcf} vs cpu {wcf}"
            );
        }
        (g, w) => panic!("query {idx} result variant mismatch: gpu {g:?} vs cpu {w:?}"),
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuAnisotropicClearcoat, queries: &[AnisotropicClearcoatQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, query, result);
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnisotropicClearcoat::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn aspect_ratio_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnisotropicClearcoat::new(&ctx);
    // Interior anisotropy magnitudes of both signs; none at 0 or 1 where the
    // mapping bottoms out, so the single `sqrt` argument stays comfortably away
    // from its endpoints.
    let queries: Vec<AnisotropicClearcoatQuery> = [-0.8_f32, -0.5, -0.25, 0.25, 0.5, 0.8]
        .iter()
        .map(|&anisotropy| AnisotropicClearcoatQuery::AspectRatio { anisotropy })
        .collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn anisotropic_alphas_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnisotropicClearcoat::new(&ctx);
    // Interior roughness and anisotropy of both signs; the squared roughness and
    // split alphas stay well above `MIN_ALPHA`, so neither floors.
    let queries = vec![
        AnisotropicClearcoatQuery::AnisotropicAlphas {
            roughness: 0.3,
            anisotropy: 0.5,
        },
        AnisotropicClearcoatQuery::AnisotropicAlphas {
            roughness: 0.6,
            anisotropy: -0.4,
        },
        AnisotropicClearcoatQuery::AnisotropicAlphas {
            roughness: 0.75,
            anisotropy: 0.25,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn ggx_aniso_ndf_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnisotropicClearcoat::new(&ctx);
    // Interior half-vector cosines and alphas, so the Burley denominator stays
    // well above `MIN_DENOM` and the GPU takes the finite-division branch.
    let queries = vec![
        AnisotropicClearcoatQuery::GgxAnisoNdf {
            n_dot_h: 0.95,
            t_dot_h: 0.2,
            b_dot_h: -0.15,
            alpha_t: 0.3,
            alpha_b: 0.2,
        },
        AnisotropicClearcoatQuery::GgxAnisoNdf {
            n_dot_h: 0.8,
            t_dot_h: -0.25,
            b_dot_h: 0.3,
            alpha_t: 0.45,
            alpha_b: 0.15,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn ggx_aniso_visibility_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnisotropicClearcoat::new(&ctx);
    // Comfortably positive cosines keep the Smith sum well above `MIN_DENOM`.
    let queries = vec![
        AnisotropicClearcoatQuery::GgxAnisoVisibility {
            alpha_t: 0.3,
            alpha_b: 0.2,
            t_dot_v: 0.2,
            b_dot_v: 0.1,
            n_dot_v: 0.8,
            t_dot_l: -0.15,
            b_dot_l: 0.25,
            n_dot_l: 0.7,
        },
        AnisotropicClearcoatQuery::GgxAnisoVisibility {
            alpha_t: 0.4,
            alpha_b: 0.35,
            t_dot_v: -0.3,
            b_dot_v: 0.2,
            n_dot_v: 0.6,
            t_dot_l: 0.1,
            b_dot_l: -0.2,
            n_dot_l: 0.55,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn clearcoat_ggx_ndf_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnisotropicClearcoat::new(&ctx);
    // Interior clearcoat alpha and half-vector cosine keep the Trowbridge-Reitz
    // kernel finite and off the `MIN_DENOM` guard.
    let queries = vec![
        AnisotropicClearcoatQuery::ClearcoatGgxNdf {
            n_dot_h: 0.9,
            alpha: 0.1,
        },
        AnisotropicClearcoatQuery::ClearcoatGgxNdf {
            n_dot_h: 0.7,
            alpha: 0.3,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn clearcoat_visibility_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnisotropicClearcoat::new(&ctx);
    // Positive cosines keep the isotropic Smith sum well above `MIN_DENOM`.
    let queries = vec![
        AnisotropicClearcoatQuery::ClearcoatVisibility {
            n_dot_v: 0.8,
            n_dot_l: 0.7,
            alpha: 0.1,
        },
        AnisotropicClearcoatQuery::ClearcoatVisibility {
            n_dot_v: 0.55,
            n_dot_l: 0.6,
            alpha: 0.25,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn clearcoat_fresnel_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnisotropicClearcoat::new(&ctx);
    // Interior cosines well inside `0..=1`, so the manually expanded fifth power
    // is exercised away from the clamp endpoints.
    let queries: Vec<AnisotropicClearcoatQuery> = [0.3_f32, 0.5, 0.7, 0.85, 0.95]
        .iter()
        .map(|&cos_theta| AnisotropicClearcoatQuery::ClearcoatFresnel { cos_theta })
        .collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn clearcoat_attenuation_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnisotropicClearcoat::new(&ctx);
    // Interior reflected fractions inside `0..=1`, away from the clamp endpoints.
    let queries: Vec<AnisotropicClearcoatQuery> = [0.04_f32, 0.2, 0.5, 0.8]
        .iter()
        .map(|&fc| AnisotropicClearcoatQuery::ClearcoatAttenuation { fc })
        .collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn evaluate_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnisotropicClearcoat::new(&ctx);
    // An orthonormal shading frame with the view and light in the upper
    // hemisphere, both cosines comfortably positive. Interior roughness,
    // anisotropy and clearcoat controls keep every lobe off its guard branch.
    let t = [1.0, 0.0, 0.0];
    let b = [0.0, 1.0, 0.0];
    let n = [0.0, 0.0, 1.0];
    let dir = |x: f32, y: f32, z: f32| {
        let u = unit(x, y, z);
        [u.x, u.y, u.z]
    };
    let queries = vec![
        AnisotropicClearcoatQuery::Evaluate {
            roughness: 0.4,
            anisotropy: 0.5,
            clearcoat_roughness: 0.15,
            clearcoat_strength: 0.8,
            tangent: t,
            bitangent: b,
            normal: n,
            view: dir(0.2, 0.1, 0.95),
            light: dir(-0.15, 0.25, 0.95),
        },
        AnisotropicClearcoatQuery::Evaluate {
            roughness: 0.65,
            anisotropy: -0.4,
            clearcoat_roughness: 0.3,
            clearcoat_strength: 0.5,
            tangent: t,
            bitangent: b,
            normal: n,
            view: dir(0.3, -0.2, 0.9),
            light: dir(0.1, 0.3, 0.95),
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn randomized_mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnisotropicClearcoat::new(&ctx);
    let mut state: u64 = 0x005e_eda1_15c0_ffee_u64 ^ 0x9e37_79b9_7f4a_7c15;
    let t = [1.0, 0.0, 0.0];
    let b = [0.0, 1.0, 0.0];
    let n = [0.0, 0.0, 1.0];
    let mut queries = Vec::new();
    for _ in 0..24 {
        // Interior anisotropy of either sign, magnitude in `0.2..=0.8`, so it
        // stays clear of both the isotropic `0` and the full-anisotropy `1`.
        let sign = if lcg(&mut state) < 0.5 { -1.0 } else { 1.0 };
        let anisotropy = sign * ranged(&mut state, 0.2, 0.8);
        let roughness = ranged(&mut state, 0.2, 0.8);
        queries.push(AnisotropicClearcoatQuery::AspectRatio { anisotropy });
        queries.push(AnisotropicClearcoatQuery::AnisotropicAlphas {
            roughness,
            anisotropy,
        });
        queries.push(AnisotropicClearcoatQuery::GgxAnisoNdf {
            n_dot_h: ranged(&mut state, 0.6, 0.98),
            t_dot_h: signed(&mut state, 0.3),
            b_dot_h: signed(&mut state, 0.3),
            alpha_t: ranged(&mut state, 0.15, 0.5),
            alpha_b: ranged(&mut state, 0.15, 0.5),
        });
        queries.push(AnisotropicClearcoatQuery::GgxAnisoVisibility {
            alpha_t: ranged(&mut state, 0.15, 0.5),
            alpha_b: ranged(&mut state, 0.15, 0.5),
            t_dot_v: signed(&mut state, 0.3),
            b_dot_v: signed(&mut state, 0.3),
            n_dot_v: ranged(&mut state, 0.4, 0.95),
            t_dot_l: signed(&mut state, 0.3),
            b_dot_l: signed(&mut state, 0.3),
            n_dot_l: ranged(&mut state, 0.4, 0.95),
        });
        queries.push(AnisotropicClearcoatQuery::ClearcoatGgxNdf {
            n_dot_h: ranged(&mut state, 0.6, 0.98),
            alpha: ranged(&mut state, 0.1, 0.4),
        });
        queries.push(AnisotropicClearcoatQuery::ClearcoatVisibility {
            n_dot_v: ranged(&mut state, 0.4, 0.95),
            n_dot_l: ranged(&mut state, 0.4, 0.95),
            alpha: ranged(&mut state, 0.1, 0.4),
        });
        queries.push(AnisotropicClearcoatQuery::ClearcoatFresnel {
            cos_theta: ranged(&mut state, 0.3, 0.95),
        });
        queries.push(AnisotropicClearcoatQuery::ClearcoatAttenuation {
            fc: ranged(&mut state, 0.05, 0.9),
        });
        queries.push(AnisotropicClearcoatQuery::Evaluate {
            roughness,
            anisotropy,
            clearcoat_roughness: ranged(&mut state, 0.1, 0.4),
            clearcoat_strength: ranged(&mut state, 0.3, 0.9),
            tangent: t,
            bitangent: b,
            normal: n,
            view: rand_upper(&mut state),
            light: rand_upper(&mut state),
        });
    }
    check(&ctx, &gpu, &queries);
}
