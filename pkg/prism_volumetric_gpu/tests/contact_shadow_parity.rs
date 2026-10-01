//! Real-device parity for the screen-space contact-shadow (`SSCS`) march twin:
//! [`GpuContactShadow`](prism_volumetric_gpu::contact_shadow::GpuContactShadow)
//! must reproduce the `CPU` golden
//! [`contact_shadow_occlusion`](prism_render_architecture::particle::contact_shadow::contact_shadow_occlusion)
//! pixel for pixel across empty, lit, occluded, jittered, random and degenerate
//! inputs.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The integer jitter hash is exact arithmetic plus an exact power-of-two
//! scale, so the per-pixel jitter offset and the jittered fractions are
//! bit-identical on both engines. For a `ray_depth_span` of `0` the per-step
//! depth gap `diff = start_depth - scene_z` is a bit-identical exact
//! subtraction too, so the *hit/miss decision* and *first-hit index* match
//! exactly on both engines regardless of the random depths; only the folded
//! shadow value (through the falloff reciprocal and the `smoothstep` cubic) may
//! differ by a legal fused multiply-add in the low mantissa bits, which the
//! `abs_diff <= EPS` bound admits. The hard-shadow test additionally pins the
//! saturated endpoints bit for bit (`to_bits`), where the result collapses to
//! an exact `0.0` or `1.0`, across a jitter-gated hit/miss boundary.
//!
//! Provenance: standard screen-space contact-shadow depth ray-march; no Unreal
//! Engine source or derived code.

use prism_render_architecture::particle::contact_shadow::{
    contact_shadow_occlusion, jitter_offset, ContactShadowParams,
};
use prism_volumetric_gpu::contact_shadow::{ContactShadowQuery, GpuContactShadow};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound for the folded shadow value. A `GPU` may fuse a
/// multiply-add the scalar reference leaves separate (the falloff denominator,
/// the `smoothstep` cubic), perturbing the low mantissa bits by a few units in
/// the last place; `1e-5` admits that legal slack while still failing a
/// genuinely wrong port.
const EPS: f32 = 1.0e-5;

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg_unit(state: &mut u64) -> f32 {
    // Knuth multiplier / increment; the shift takes the high bits where the
    // generator mixes best.
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    // 24 usable mantissa bits mapped onto [0, 1).
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Runs the march on both the `CPU` reference and the `GPU` twin for the same
/// params and queries, asserts pixel-for-pixel parity within [`EPS`], and
/// returns the `GPU` factors for any further shape assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuContactShadow,
    params: &ContactShadowParams,
    queries: &[ContactShadowQuery<'_>],
) -> Vec<f32> {
    let cpu: Vec<f32> = queries
        .iter()
        .map(|q| {
            contact_shadow_occlusion(
                params,
                q.start_depth,
                q.ray_depth_span,
                q.scene_depths,
                q.jitter_seed,
            )
        })
        .collect();

    let got = gpu.eval(ctx, params, queries);
    assert_eq!(got.len(), cpu.len(), "one factor per pixel");
    for (idx, (g, c)) in got.iter().zip(cpu.iter()).enumerate() {
        assert!(
            (g - c).abs() <= EPS,
            "pixel {idx} factor mismatch: gpu {g}, cpu {c}"
        );
        assert!(g.is_finite(), "pixel {idx} produced a non-finite factor");
        assert!(
            (0.0..=1.0).contains(g),
            "pixel {idx} factor {g} left the unit interval"
        );
    }
    got
}

/// Base parameters: no `bias`, a `0.5`-wide window, unit intensity and falloff.
fn base_params(step_count: u32) -> ContactShadowParams {
    ContactShadowParams::new(step_count, 1.0, 0.5, 0.0, 1.0, 1.0)
}

#[test]
fn empty_input_yields_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuContactShadow::new(&ctx);
    let out = gpu.eval(&ctx, &base_params(8), &[]);
    assert!(
        out.is_empty(),
        "an empty query batch yields an empty result"
    );
}

#[test]
fn no_occluder_stays_fully_lit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuContactShadow::new(&ctx);
    let params = base_params(8);
    // Every scene sample sits far behind the ray, so the gap is negative and
    // nothing lands in the window: the point is fully lit.
    let far = [20.0_f32; 8];
    let queries = [
        ContactShadowQuery {
            start_depth: 10.0,
            ray_depth_span: 0.0,
            jitter_seed: 1,
            scene_depths: &far,
        },
        ContactShadowQuery {
            start_depth: 10.0,
            ray_depth_span: 0.5,
            jitter_seed: 7,
            scene_depths: &far,
        },
    ];
    let got = check(&ctx, &gpu, &params, &queries);
    for g in &got {
        // Fully lit is an exact 1.0 on both engines (no hit, no arithmetic).
        assert_eq!(g.to_bits(), 1.0f32.to_bits(), "no occluder stays fully lit");
    }
}

#[test]
fn direct_contact_darkens_the_point() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuContactShadow::new(&ctx);
    let params = base_params(8);
    // A surface 0.2 nearer than the ray sits inside (0.0 .. 0.5): a hit.
    let near = [9.8_f32; 8];
    // A gap of 0.6 exceeds the 0.5 window: the ray is assumed to have slipped
    // behind a thin object into empty space, so no occlusion.
    let behind = [9.4_f32; 8];
    let queries = [
        ContactShadowQuery {
            start_depth: 10.0,
            ray_depth_span: 0.0,
            jitter_seed: 7,
            scene_depths: &near,
        },
        ContactShadowQuery {
            start_depth: 10.0,
            ray_depth_span: 0.0,
            jitter_seed: 3,
            scene_depths: &behind,
        },
    ];
    let got = check(&ctx, &gpu, &params, &queries);
    assert!(got[0] < 1.0, "a direct contact darkens the point");
    assert_eq!(
        got[1].to_bits(),
        1.0f32.to_bits(),
        "a gap wider than the window stays fully lit"
    );
}

#[test]
fn jitter_seed_takes_effect_and_matches_cpu_bit_for_bit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuContactShadow::new(&ctx);

    // A single-step march starting at depth 0 with a power-of-two depth span, so
    // the sole sample sits at `ray_z = 8 * jitter`. With `start_depth = 0`, a
    // span and step count that are powers of two, and a bit-exact jitter, the
    // gap `diff = 8 * jitter - scene_z` is computed identically on both engines,
    // so the hit/miss decision is bit-identical with no boundary flake. A huge
    // intensity saturates any hit to full shadow, collapsing the factor to an
    // exact 0.0, while a miss leaves the exact 1.0 initial value.
    //
    // Fields: 1 step, max_distance 1.0, thickness 100.0 (wide window), bias 0.0,
    // intensity 1e9 (saturating), falloff 0.0.
    let params = ContactShadowParams::new(1, 1.0, 100.0, 0.0, 1.0e9, 0.0);
    let span = 8.0_f32;
    // Occluder plane at depth 4: a hit needs `8 * jitter >= 4`, i.e. the jitter
    // fraction at or above 0.5.
    let scene = [4.0_f32];

    // Pick seeds whose jitter lands in safe hit / miss bands (well clear of the
    // 0.5 boundary and clear of jitter ~1 where a hit would not saturate), so
    // the saturated endpoints are exact on both engines.
    let mut hit_seeds: Vec<u32> = Vec::new();
    let mut miss_seeds: Vec<u32> = Vec::new();
    let mut seed = 1_u32;
    while (hit_seeds.len() < 8 || miss_seeds.len() < 8) && seed < 200_000 {
        let j = jitter_offset(seed);
        if (0.55..=0.90).contains(&j) && hit_seeds.len() < 8 {
            hit_seeds.push(seed);
        } else if (0.10..=0.40).contains(&j) && miss_seeds.len() < 8 {
            miss_seeds.push(seed);
        }
        seed += 1;
    }
    assert!(
        hit_seeds.len() == 8 && miss_seeds.len() == 8,
        "expected to find enough safe-band seeds"
    );

    let mut queries: Vec<ContactShadowQuery<'_>> = Vec::new();
    for &s in hit_seeds.iter().chain(miss_seeds.iter()) {
        queries.push(ContactShadowQuery {
            start_depth: 0.0,
            ray_depth_span: span,
            jitter_seed: s,
            scene_depths: &scene,
        });
    }

    let cpu: Vec<f32> = queries
        .iter()
        .map(|q| {
            contact_shadow_occlusion(
                &params,
                q.start_depth,
                q.ray_depth_span,
                q.scene_depths,
                q.jitter_seed,
            )
        })
        .collect();
    let got = gpu.eval(&ctx, &params, &queries);
    assert_eq!(got.len(), cpu.len());

    // Bit-for-bit equality across the whole jitter-gated batch.
    for (idx, (g, c)) in got.iter().zip(cpu.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            c.to_bits(),
            "pixel {idx}: gpu {g} vs cpu {c} differ at the bit level"
        );
    }
    // The jitter genuinely gates the outcome: the hit band is fully shadowed
    // (exact 0.0) and the miss band is fully lit (exact 1.0).
    for g in &got[..8] {
        assert_eq!(
            g.to_bits(),
            0.0f32.to_bits(),
            "hit-band seed stays shadowed"
        );
    }
    for g in &got[8..] {
        assert_eq!(g.to_bits(), 1.0f32.to_bits(), "miss-band seed stays lit");
    }
}

#[test]
fn random_depth_fields_multi_resolution() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuContactShadow::new(&ctx);

    // A spread of (pixel count, step count) resolutions, each with its own
    // random per-step depth field and random params. `ray_depth_span` is held
    // at 0 so the per-step gap `start_depth - scene_z` is a bit-identical exact
    // subtraction: the hit/miss decision and first-hit index match the
    // reference exactly, and only the folded shadow value may slip by a legal
    // fused multiply-add within EPS.
    let cases: [(usize, u32, u64); 5] = [
        (1, 1, 0x1234_5678_9abc_def0),
        (16, 4, 0x0f0f_0f0f_1234_5678),
        (37, 8, 0xdead_beef_cafe_babe),
        (64, 12, 0x5555_aaaa_3333_cccc),
        (130, 16, 0x9e37_79b9_7f4a_7c15),
    ];

    for (pixel_count, step_count, seed) in cases {
        let mut state = seed;

        // Random, finite params. Thickness and bias stay positive so the window
        // is a sensible contact band; intensity and falloff span a useful range.
        let max_distance = 0.5 + lcg_unit(&mut state) * 3.0;
        let thickness = 0.2 + lcg_unit(&mut state) * 1.5;
        let bias = lcg_unit(&mut state) * 0.1;
        let intensity = lcg_unit(&mut state) * 2.0;
        let falloff = lcg_unit(&mut state) * 2.0;
        let params = ContactShadowParams::new(
            step_count,
            max_distance,
            thickness,
            bias,
            intensity,
            falloff,
        );

        // Each pixel: a start depth near 10 and `step_count` scene depths spread
        // around it so some steps land in the window and some do not.
        let mut depth_store: Vec<Vec<f32>> = Vec::with_capacity(pixel_count);
        let mut starts: Vec<f32> = Vec::with_capacity(pixel_count);
        let mut seeds: Vec<u32> = Vec::with_capacity(pixel_count);
        for _ in 0..pixel_count {
            let start = 8.0 + lcg_unit(&mut state) * 4.0;
            starts.push(start);
            seeds.push((state >> 16) as u32);
            let mut depths = Vec::with_capacity(step_count as usize);
            for _ in 0..step_count {
                // Depths within roughly [start - 1.2, start + 0.8]: a mix of
                // nearer (potential occluder) and farther samples.
                depths.push(start - 1.2 + lcg_unit(&mut state) * 2.0);
            }
            depth_store.push(depths);
        }

        let queries: Vec<ContactShadowQuery<'_>> = (0..pixel_count)
            .map(|i| ContactShadowQuery {
                start_depth: starts[i],
                ray_depth_span: 0.0,
                jitter_seed: seeds[i],
                scene_depths: &depth_store[i],
            })
            .collect();

        check(&ctx, &gpu, &params, &queries);
    }
}

#[test]
fn degenerate_steps_and_params() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuContactShadow::new(&ctx);

    let near = [9.8_f32; 8];

    // A zero step count yields an empty schedule: fully lit despite an occluder.
    let zero_steps = base_params(0);
    // A non-positive thickness makes the window empty: no gap can land inside.
    let zero_thickness = ContactShadowParams::new(8, 1.0, 0.0, 0.0, 1.0, 1.0);
    let negative_thickness = ContactShadowParams::new(8, 1.0, -0.5, 0.0, 1.0, 1.0);

    let occluded = ContactShadowQuery {
        start_depth: 10.0,
        ray_depth_span: 0.0,
        jitter_seed: 5,
        scene_depths: &near,
    };

    for params in [zero_steps, zero_thickness, negative_thickness] {
        let got = check(&ctx, &gpu, &params, &[occluded]);
        assert_eq!(
            got[0].to_bits(),
            1.0f32.to_bits(),
            "a degenerate schedule or window stays fully lit"
        );
    }

    // A depth slice shorter than the step count stops the march early (the
    // reference zips its fixed schedule against the supplied depths). Here the
    // only in-window occluder would be read at step 5, but just two depths are
    // supplied, so the march never reaches it and the point stays lit.
    let params = base_params(8);
    let short = [10.9_f32, 10.9_f32];
    let short_query = ContactShadowQuery {
        start_depth: 10.0,
        ray_depth_span: 0.0,
        jitter_seed: 9,
        scene_depths: &short,
    };
    let got = check(&ctx, &gpu, &params, &[short_query]);
    assert_eq!(
        got[0].to_bits(),
        1.0f32.to_bits(),
        "a short depth slice stops the march before any hit"
    );

    // An empty depth slice with a non-zero step count also stays lit.
    let empty_depths: [f32; 0] = [];
    let empty_query = ContactShadowQuery {
        start_depth: 10.0,
        ray_depth_span: 0.0,
        jitter_seed: 2,
        scene_depths: &empty_depths,
    };
    let got = check(&ctx, &gpu, &params, &[empty_query]);
    assert_eq!(
        got[0].to_bits(),
        1.0f32.to_bits(),
        "an empty depth slice stays fully lit"
    );
}
