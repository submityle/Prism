//! Real-device parity for the soft-particle depth-fade twin:
//! [`GpuSoftParticle`](prism_volumetric_gpu::soft_particle::GpuSoftParticle)
//! must reproduce the `CPU` golden
//! [`soft_particle`](prism_render_architecture::particle::soft_particle) across
//! the whole seam/near fade regime — a particle far in front of the scene (no
//! seam fade), flush against it or behind it (full seam fade), a camera-near
//! dissolve and a camera-far full-opacity case, a near-zero and a very large
//! `fade_distance`, the degenerate near-band hard step and the degenerate
//! frustum guard — plus `linearize_depth` at the near/mid/far `NDC` endpoints
//! and a random batch, each compared element for element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernels are portable core-`WGSL`, so they need no optional device
//! feature.
//!
//! # Parity criterion
//!
//! Each fade is a fixed, non-reorderable subtract, one divide and a `clamp` (or
//! a guarded branch), so `CPU` and `GPU` evaluate the same closed-form algebra.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits by a few units in the last
//! place. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` (with a `1e-6` relative-error floor), loose enough to
//! admit a legal fused multiply-add contraction yet tight enough to fail a
//! genuinely wrong port (a dropped guard, a swapped seam sign, a missing
//! clamp).
//!
//! The combined-fade twin accepts a `view_depth` separate from
//! `particle_depth`; the expected value is the product of the golden
//! [`DepthFade::fade_factor`](prism_render_architecture::particle::soft_particle::DepthFade::fade_factor)
//! and
//! [`NearFade::camera_proximity_fade`](prism_render_architecture::particle::soft_particle::NearFade::camera_proximity_fade),
//! and the suite additionally pins that with `view_depth == particle_depth` the
//! twin equals the golden
//! [`combined_fade`](prism_render_architecture::particle::soft_particle::SoftParticleParams::combined_fade).
//!
//! Provenance: standard soft-particle depth fade (design section 14); no
//! third-party engine source or derived code.

use prism_render_architecture::particle::soft_particle::{linearize_depth, SoftParticleParams};
use prism_volumetric_gpu::soft_particle::{
    GpuSoftParticle, LinearizeQuery, SoftParticleQuery, SoftParticleSample,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes (such as a linearized
/// far-plane depth) where a few units in the last place exceed the absolute
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

/// The expected combined fade for one sample: the product of the golden seam
/// fade and near-plane fade, each evaluated through its public golden entry.
fn expected_fade(params: SoftParticleParams, s: &SoftParticleSample) -> f32 {
    let seam = params
        .depth_fade()
        .fade_factor(s.scene_depth, s.particle_depth);
    let near = params.near_fade().camera_proximity_fade(s.view_depth);
    seam * near
}

/// Runs the `GPU` combined-fade kernel and asserts element-for-element parity
/// against the golden building blocks, plus — for every sample whose
/// `view_depth` equals its `particle_depth` — against the golden
/// `combined_fade` itself. Also pins the result into `0..=1`.
fn check_fade(
    ctx: &GpuContext,
    gpu: &GpuSoftParticle,
    params: SoftParticleParams,
    samples: Vec<SoftParticleSample>,
) {
    let query = SoftParticleQuery {
        params,
        samples: samples.clone(),
    };
    let got = gpu.eval(ctx, &query);
    assert_eq!(got.len(), samples.len(), "one fade value per sample");

    for (idx, s) in samples.iter().enumerate() {
        let want = expected_fade(params, s);
        assert!(
            close(got[idx], want),
            "fade mismatch at sample {idx} (scene {}, particle {}, view {}; \
             fade_distance {}, near_start {}, near_end {}): gpu {}, cpu {}",
            s.scene_depth,
            s.particle_depth,
            s.view_depth,
            params.fade_distance,
            params.near_start,
            params.near_end,
            got[idx],
            want
        );
        assert!(
            (0.0..=1.0).contains(&got[idx]),
            "fade value at sample {idx} must stay in 0..=1: {}",
            got[idx]
        );
        // When the near fade reads the particle's own depth the twin must equal
        // the golden `combined_fade` exactly (to tolerance).
        if (s.view_depth - s.particle_depth).abs() < REL_FLOOR {
            let combined = params.combined_fade(s.scene_depth, s.particle_depth);
            assert!(
                close(got[idx], combined),
                "combined_fade mismatch at sample {idx}: gpu {}, cpu {combined}",
                got[idx]
            );
        }
    }
}

/// Runs the `GPU` linearize kernel and asserts element-for-element parity
/// against the golden `linearize_depth`.
fn check_linearize(
    ctx: &GpuContext,
    gpu: &GpuSoftParticle,
    near: f32,
    far: f32,
    ndc_depths: Vec<f32>,
) {
    let query = LinearizeQuery {
        near,
        far,
        ndc_depths: ndc_depths.clone(),
    };
    let got = gpu.eval_linearize(ctx, &query);
    assert_eq!(got.len(), ndc_depths.len(), "one depth per NDC entry");

    for (idx, &ndc) in ndc_depths.iter().enumerate() {
        let want = linearize_depth(ndc, near, far);
        assert!(
            close(got[idx], want),
            "linearize mismatch at entry {idx} (ndc {ndc}, near {near}, far {far}): \
             gpu {}, cpu {want}",
            got[idx]
        );
    }
}

#[test]
fn empty_batches_short_circuit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftParticle::new(&ctx);
    // Empty inputs must return empty with no dispatch (a storage buffer cannot
    // be zero-sized).
    let fade = gpu.eval(
        &ctx,
        &SoftParticleQuery {
            params: SoftParticleParams::new(2.0, 1.0, 3.0),
            samples: Vec::new(),
        },
    );
    assert!(fade.is_empty(), "an empty sample batch yields no values");
    let lin = gpu.eval_linearize(
        &ctx,
        &LinearizeQuery {
            near: 0.5,
            far: 100.0,
            ndc_depths: Vec::new(),
        },
    );
    assert!(lin.is_empty(), "an empty NDC batch yields no values");
}

#[test]
fn seam_fade_regime_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftParticle::new(&ctx);
    // A wide near window keeps the near fade saturated at 1 (view well past
    // near_end), isolating the seam fade across its whole regime.
    let params = SoftParticleParams::new(2.0, 1.0, 3.0);
    let view = 10.0; // past near_end => near fade is a clean 1.
    let samples = vec![
        // Particle a full band in front of the scene: seam saturates to 1.
        SoftParticleSample {
            scene_depth: 8.0,
            particle_depth: 5.5,
            view_depth: view,
        },
        // Mid-band contact fade (gap 1 over band 2 => 0.5), kept off the exact
        // clamp boundaries so no tie can flip.
        SoftParticleSample {
            scene_depth: 6.0,
            particle_depth: 5.0,
            view_depth: view,
        },
        // Flush against the surface: gap 0 => seam 0.
        SoftParticleSample {
            scene_depth: 5.0,
            particle_depth: 5.0,
            view_depth: view,
        },
        // Particle behind the geometry: negative gap => seam clamps to 0.
        SoftParticleSample {
            scene_depth: 4.0,
            particle_depth: 6.0,
            view_depth: view,
        },
    ];
    check_fade(&ctx, &gpu, params, samples);
}

#[test]
fn near_fade_regime_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftParticle::new(&ctx);
    // A wide seam band with the particle a full band in front keeps the seam
    // fade saturated at 1, isolating the near fade.
    let params = SoftParticleParams::new(1.0, 2.0, 6.0);
    let scene = 100.0;
    let particle = 50.0; // gap 50 over band 1 => seam saturates to 1.
    let samples = vec![
        // Camera extremely near (below near_start): near fade 0.
        SoftParticleSample {
            scene_depth: scene,
            particle_depth: particle,
            view_depth: 0.25,
        },
        // Just at near_start: near fade 0.
        SoftParticleSample {
            scene_depth: scene,
            particle_depth: particle,
            view_depth: 2.0,
        },
        // Mid near band (view 4 over [2, 6] => 0.5).
        SoftParticleSample {
            scene_depth: scene,
            particle_depth: particle,
            view_depth: 4.0,
        },
        // Camera extremely far (past near_end): near fade 1.
        SoftParticleSample {
            scene_depth: scene,
            particle_depth: particle,
            view_depth: 500.0,
        },
    ];
    check_fade(&ctx, &gpu, params, samples);
}

#[test]
fn fade_distance_extremes_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftParticle::new(&ctx);
    let view = 10.0; // wide near window below => near fade 1.

    // A non-positive fade_distance is a guard: the seam fade is disabled and
    // returns 1 regardless of the gap.
    check_fade(
        &ctx,
        &gpu,
        SoftParticleParams::new(0.0, 1.0, 3.0),
        vec![SoftParticleSample {
            scene_depth: 2.0,
            particle_depth: 9.0,
            view_depth: view,
        }],
    );
    check_fade(
        &ctx,
        &gpu,
        SoftParticleParams::new(-5.0, 1.0, 3.0),
        vec![SoftParticleSample {
            scene_depth: 2.0,
            particle_depth: 9.0,
            view_depth: view,
        }],
    );

    // A tiny-but-positive fade_distance makes any positive gap saturate to 1,
    // and any non-positive gap clamp to 0 — away from the single boundary tie.
    check_fade(
        &ctx,
        &gpu,
        SoftParticleParams::new(1.0e-3, 1.0, 3.0),
        vec![
            SoftParticleSample {
                scene_depth: 5.0,
                particle_depth: 4.0,
                view_depth: view,
            },
            SoftParticleSample {
                scene_depth: 4.0,
                particle_depth: 5.0,
                view_depth: view,
            },
        ],
    );

    // A very large fade_distance keeps a modest gap deep inside the ramp.
    check_fade(
        &ctx,
        &gpu,
        SoftParticleParams::new(1.0e6, 1.0, 3.0),
        vec![SoftParticleSample {
            scene_depth: 7.0,
            particle_depth: 2.0,
            view_depth: view,
        }],
    );
}

#[test]
fn degenerate_near_band_is_a_hard_step() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftParticle::new(&ctx);
    // near_start == near_end: the near fade degrades to a hard step (0 before
    // near_end, 1 at or beyond it) with no divide by a vanishing width. The
    // seam fade is kept saturated at 1. Views are held off the exact endpoint
    // so the branch result is unambiguous.
    let params = SoftParticleParams::new(1.0, 2.0, 2.0);
    let scene = 100.0;
    let particle = 50.0;
    let samples = vec![
        SoftParticleSample {
            scene_depth: scene,
            particle_depth: particle,
            view_depth: 1.5,
        },
        SoftParticleSample {
            scene_depth: scene,
            particle_depth: particle,
            view_depth: 2.5,
        },
        SoftParticleSample {
            scene_depth: scene,
            particle_depth: particle,
            view_depth: 9.0,
        },
    ];
    check_fade(&ctx, &gpu, params, samples);
}

#[test]
fn combined_fade_with_view_equal_particle_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftParticle::new(&ctx);
    // view_depth == particle_depth is the exact golden `combined_fade` contract;
    // `check_fade` additionally asserts equality with `combined_fade` here.
    let params = SoftParticleParams::new(2.0, 1.0, 3.0);
    let samples = vec![
        SoftParticleSample {
            scene_depth: 5.0,
            particle_depth: 2.0,
            view_depth: 2.0,
        },
        SoftParticleSample {
            scene_depth: 1.0,
            particle_depth: 4.0,
            view_depth: 4.0,
        },
        SoftParticleSample {
            scene_depth: 100.0,
            particle_depth: 50.0,
            view_depth: 50.0,
        },
        SoftParticleSample {
            scene_depth: 3.5,
            particle_depth: 2.0,
            view_depth: 2.0,
        },
    ];
    check_fade(&ctx, &gpu, params, samples);
}

#[test]
fn linearize_hits_endpoints_and_interior() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftParticle::new(&ctx);
    // Near (0), several interior and far (1) NDC depths over a standard frustum.
    check_linearize(
        &ctx,
        &gpu,
        0.5,
        100.0,
        vec![0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
    );
}

#[test]
fn linearize_degenerate_range_guard() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftParticle::new(&ctx);
    // near and far within EPS: the guard returns `near` without dividing by a
    // vanishing range.
    check_linearize(&ctx, &gpu, 4.0, 4.0, vec![0.0, 0.3, 0.7, 1.0]);
    check_linearize(&ctx, &gpu, 4.0, 4.0 + 1.0e-9, vec![0.0, 0.5, 1.0]);
}

#[test]
fn random_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftParticle::new(&ctx);
    let mut state = 0x50f7_9a13_2c4e_6d80_u64;

    // Several random parameter sets, each with a random batch of samples. The
    // bands are kept comfortably above the EPS guard so the non-degenerate
    // ramps are exercised (the guards have their own dedicated scenes).
    for _ in 0..8 {
        let fade_distance = 0.5 + lcg(&mut state) * 8.0;
        let near_start = lcg(&mut state) * 3.0;
        let near_end = near_start + 1.0 + lcg(&mut state) * 5.0;
        let params = SoftParticleParams::new(fade_distance, near_start, near_end);

        let mut samples = Vec::with_capacity(64);
        for _ in 0..64 {
            let scene_depth = lcg(&mut state) * 50.0;
            let particle_depth = lcg(&mut state) * 50.0;
            let view_depth = lcg(&mut state) * 20.0;
            samples.push(SoftParticleSample {
                scene_depth,
                particle_depth,
                view_depth,
            });
        }
        check_fade(&ctx, &gpu, params, samples);
    }

    // A random linearize batch over a standard frustum.
    let near = 0.1 + lcg(&mut state);
    let far = near + 50.0 + lcg(&mut state) * 450.0;
    let mut ndc = Vec::with_capacity(128);
    for _ in 0..128 {
        ndc.push(lcg(&mut state));
    }
    check_linearize(&ctx, &gpu, near, far, ndc);
}
