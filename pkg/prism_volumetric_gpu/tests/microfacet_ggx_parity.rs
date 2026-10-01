//! Real-device parity for the Cook-Torrance microfacet-`GGX` twin:
//! [`GpuMicrofacetGgx`](prism_volumetric_gpu::microfacet_ggx::GpuMicrofacetGgx)
//! must reproduce the `CPU` golden
//! [`microfacet_ggx`](prism_render_architecture::particle::microfacet_ggx)
//! across all eight twinned terms — the roughness remaps
//! ([`clamp_alpha`](prism_render_architecture::particle::microfacet_ggx::clamp_alpha),
//! [`alpha_from_perceptual_roughness`](prism_render_architecture::particle::microfacet_ggx::alpha_from_perceptual_roughness)),
//! the `GGX` distribution `D`
//! ([`ggx_distribution`](prism_render_architecture::particle::microfacet_ggx::ggx_distribution)),
//! the height-correlated `Smith` `G2`
//! ([`smith_g2_height_correlated`](prism_render_architecture::particle::microfacet_ggx::smith_g2_height_correlated))
//! and visibility `V`
//! ([`visibility_smith_ggx_correlated`](prism_render_architecture::particle::microfacet_ggx::visibility_smith_ggx_correlated)),
//! the three-channel `Fresnel`
//! ([`fresnel_schlick_f0_rgb`](prism_render_architecture::particle::microfacet_ggx::fresnel_schlick_f0_rgb))
//! and the assembled scalar and `RGB` lobes
//! ([`specular_ggx_scalar`](prism_render_architecture::particle::microfacet_ggx::specular_ggx_scalar),
//! [`specular_dvf_rgb`](prism_render_architecture::particle::microfacet_ggx::specular_dvf_rgb))
//! — each compared sample-for-sample across a batch of pseudo-random fixtures.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernels are portable core-`WGSL`, so they need no optional device
//! feature.
//!
//! # Parity criterion
//!
//! Each output is a fixed, non-reorderable sequence of multiplies, adds, one or
//! two divides and a `sqrt`, so `CPU` and `GPU` evaluate the same closed form.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, the `GGX` denominator and the `Smith` radicals shed a few
//! units in the last place, and the fifth-power `Fresnel` term is regrouped as
//! `x2 * x2 * x` rather than the reference's sequential multiply. The
//! comparison therefore allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` — loose
//! enough to admit that legal slack yet tight enough to fail a genuinely wrong
//! port (a swapped term, a dropped `sqrt`, a wrong normalizer, a missing
//! clamp). Fixtures deliberately avoid grazing angles (where the `NoL`/`NoV`
//! denominator guard engages) and the `alpha` clamp boundary tie, keeping every
//! sample in the smooth interior of each term.
//!
//! Provenance: standard Cook-Torrance / `GGX` microfacet specular with the
//! height-correlated `Smith` visibility of `Heitz` and the `Schlick` `Fresnel`;
//! no third-party engine source or derived code.

use prism_render_architecture::particle::microfacet_ggx::{
    alpha_from_perceptual_roughness, clamp_alpha, fresnel_schlick_f0_rgb, ggx_distribution,
    smith_g2_height_correlated, specular_dvf_rgb, specular_ggx_scalar,
    visibility_smith_ggx_correlated, MicrofacetDirs,
};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::microfacet_ggx::{GpuMicrofacetGgx, MicrofacetSample};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, and the fifth-power regrouping shifts the low mantissa
/// bits; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes (the `GGX` peak can be
/// well above one) where a few units in the last place exceed the absolute
/// floor.
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

/// Draws a value in `[lo, hi)` from `state`.
fn uniform_in(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + lcg(state) * (hi - lo)
}

/// Builds a batch of pseudo-random samples that stay in the smooth interior of
/// every term: cosines in `[0.2, 0.98]` (no grazing-angle denominator guard),
/// `alpha` in `[0.08, 0.9]` (clear of both the `MIN_ALPHA` floor and the upper
/// clamp tie) and `F0` in `[0.02, 0.6]`.
fn random_samples(count: usize, state: &mut u64) -> Vec<MicrofacetSample> {
    let mut samples = Vec::with_capacity(count);
    for _ in 0..count {
        samples.push(MicrofacetSample {
            n_dot_h: uniform_in(state, 0.2, 0.98),
            n_dot_l: uniform_in(state, 0.2, 0.98),
            n_dot_v: uniform_in(state, 0.2, 0.98),
            v_dot_h: uniform_in(state, 0.2, 0.98),
            alpha: uniform_in(state, 0.08, 0.9),
            f0: [
                uniform_in(state, 0.02, 0.6),
                uniform_in(state, 0.02, 0.6),
                uniform_in(state, 0.02, 0.6),
            ],
        });
    }
    samples
}

/// The reference clamped-cosine bundle for one sample (fields are public, so a
/// struct literal reuses the host-supplied cosines directly without re-deriving
/// a half vector).
fn dirs_of(s: &MicrofacetSample) -> MicrofacetDirs {
    MicrofacetDirs {
        n_dot_l: s.n_dot_l,
        n_dot_v: s.n_dot_v,
        n_dot_h: s.n_dot_h,
        v_dot_h: s.v_dot_h,
    }
}

#[test]
fn empty_inputs_do_not_panic_and_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMicrofacetGgx::new(&ctx);
    // Every entry point early-outs on an empty slice with no dispatch (a storage
    // buffer cannot be zero-sized) and returns an empty result.
    assert!(gpu.eval_clamp_alpha(&ctx, &[]).is_empty());
    assert!(gpu
        .eval_alpha_from_perceptual_roughness(&ctx, &[])
        .is_empty());
    assert!(gpu.eval_distribution(&ctx, &[]).is_empty());
    assert!(gpu.eval_smith_g2(&ctx, &[]).is_empty());
    assert!(gpu.eval_visibility(&ctx, &[]).is_empty());
    assert!(gpu.eval_fresnel_rgb(&ctx, &[]).is_empty());
    assert!(gpu.eval_specular_scalar(&ctx, &[]).is_empty());
    assert!(gpu.eval_specular_dvf_rgb(&ctx, &[]).is_empty());
}

#[test]
fn clamp_alpha_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMicrofacetGgx::new(&ctx);
    // Sweep below the floor, through the interior and above the ceiling so both
    // clamp arms are exercised (the exact boundary values are not tie-sensitive
    // for a clamp, only for ordering-dependent terms downstream).
    let alphas = [-0.5_f32, 0.0, 5.0e-5, 0.08, 0.3, 0.75, 1.0, 1.5];
    let got = gpu.eval_clamp_alpha(&ctx, &alphas);
    assert_eq!(got.len(), alphas.len());
    for (g, &a) in got.iter().zip(alphas.iter()) {
        assert!(close(*g, clamp_alpha(a)), "clamp_alpha({a}) gpu {g}");
    }
}

#[test]
fn alpha_from_perceptual_roughness_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMicrofacetGgx::new(&ctx);
    let perceptuals = [0.0_f32, 0.1, 0.3, 0.5, 0.8, 1.0];
    let got = gpu.eval_alpha_from_perceptual_roughness(&ctx, &perceptuals);
    assert_eq!(got.len(), perceptuals.len());
    for (g, &p) in got.iter().zip(perceptuals.iter()) {
        assert!(
            close(*g, alpha_from_perceptual_roughness(p)),
            "alpha_from_perceptual({p}) gpu {g}"
        );
    }
}

#[test]
fn distribution_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMicrofacetGgx::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    let samples = random_samples(256, &mut state);
    let got = gpu.eval_distribution(&ctx, &samples);
    assert_eq!(got.len(), samples.len());
    for (idx, (g, s)) in got.iter().zip(samples.iter()).enumerate() {
        let want = ggx_distribution(s.n_dot_h, s.alpha);
        assert!(close(*g, want), "D sample {idx}: gpu {g} vs cpu {want}");
    }
}

#[test]
fn smith_g2_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMicrofacetGgx::new(&ctx);
    let mut state = 0x0fed_cba9_8765_4321_u64;
    let samples = random_samples(256, &mut state);
    let got = gpu.eval_smith_g2(&ctx, &samples);
    assert_eq!(got.len(), samples.len());
    for (idx, (g, s)) in got.iter().zip(samples.iter()).enumerate() {
        let want = smith_g2_height_correlated(s.n_dot_l, s.n_dot_v, s.alpha);
        assert!(close(*g, want), "G2 sample {idx}: gpu {g} vs cpu {want}");
    }
}

#[test]
fn visibility_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMicrofacetGgx::new(&ctx);
    let mut state = 0x2b7e_1516_28ae_d2a6_u64;
    let samples = random_samples(256, &mut state);
    let got = gpu.eval_visibility(&ctx, &samples);
    assert_eq!(got.len(), samples.len());
    for (idx, (g, s)) in got.iter().zip(samples.iter()).enumerate() {
        let want = visibility_smith_ggx_correlated(s.n_dot_l, s.n_dot_v, s.alpha);
        assert!(close(*g, want), "V sample {idx}: gpu {g} vs cpu {want}");
    }
}

#[test]
fn fresnel_rgb_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMicrofacetGgx::new(&ctx);
    let mut state = 0x3243_f6a8_885a_308d_u64;
    let samples = random_samples(256, &mut state);
    let got = gpu.eval_fresnel_rgb(&ctx, &samples);
    assert_eq!(got.len(), samples.len());
    for (idx, (g, s)) in got.iter().zip(samples.iter()).enumerate() {
        let f0 = Vec3::new(s.f0[0], s.f0[1], s.f0[2]);
        let want = fresnel_schlick_f0_rgb(s.v_dot_h, f0);
        for (channel, (&gc, wc)) in g.iter().zip([want.x, want.y, want.z]).enumerate() {
            assert!(
                close(gc, wc),
                "F sample {idx} channel {channel}: gpu {gc} vs cpu {wc}"
            );
        }
    }
}

#[test]
fn specular_scalar_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMicrofacetGgx::new(&ctx);
    let mut state = 0xa409_3822_299f_31d0_u64;
    let samples = random_samples(256, &mut state);
    let got = gpu.eval_specular_scalar(&ctx, &samples);
    assert_eq!(got.len(), samples.len());
    for (idx, (g, s)) in got.iter().zip(samples.iter()).enumerate() {
        let want = specular_ggx_scalar(dirs_of(s), s.alpha, s.f0[0]);
        assert!(
            close(*g, want),
            "specular sample {idx}: gpu {g} vs cpu {want}"
        );
    }
}

#[test]
fn specular_dvf_rgb_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMicrofacetGgx::new(&ctx);
    let mut state = 0x5eed_4a7d_0bad_c0de_u64;
    let samples = random_samples(256, &mut state);
    let got = gpu.eval_specular_dvf_rgb(&ctx, &samples);
    assert_eq!(got.len(), samples.len());
    for (idx, (g, s)) in got.iter().zip(samples.iter()).enumerate() {
        let f0 = Vec3::new(s.f0[0], s.f0[1], s.f0[2]);
        let want = specular_dvf_rgb(dirs_of(s), s.alpha, f0);
        for (channel, (&gc, wc)) in g.iter().zip([want.x, want.y, want.z]).enumerate() {
            assert!(
                close(gc, wc),
                "specular rgb sample {idx} channel {channel}: gpu {gc} vs cpu {wc}"
            );
        }
    }
}

#[test]
fn specular_rgb_red_channel_tracks_scalar_lobe() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMicrofacetGgx::new(&ctx);
    let mut state = 0x1357_9bdf_0246_8ace_u64;
    let samples = random_samples(64, &mut state);
    // The scalar lobe uses `f0[0]`, so its value must equal the red channel of
    // the RGB lobe for the same sample; a cross-check that the two assemblies
    // agree on-device, not just against the reference.
    let scalar = gpu.eval_specular_scalar(&ctx, &samples);
    let rgb = gpu.eval_specular_dvf_rgb(&ctx, &samples);
    assert_eq!(scalar.len(), rgb.len());
    for (idx, (s, triple)) in scalar.iter().zip(rgb.iter()).enumerate() {
        assert!(
            close(*s, triple[0]),
            "sample {idx}: scalar {s} vs rgb red {}",
            triple[0]
        );
    }
}
