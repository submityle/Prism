//! Real-device parity for the `SDF` soft-shadow sphere-trace twin:
//! [`GpuDistanceFieldShadow`](prism_volumetric_gpu::distance_field_shadow::GpuDistanceFieldShadow)
//! must reproduce the `CPU` golden
//! [`sphere_trace_shadow`](prism_render_architecture::particle::distance_field_shadow::sphere_trace_shadow)
//! ray for ray across several baked fields, grid resolutions and march
//! parameters.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The march is a sequential sphere trace with no transcendental call and no
//! reorderable reduction, so `CPU` and `GPU` evaluate the same closed form in
//! the same order. The comparison allows `abs_diff <= 1e-4` — loose enough to
//! admit a legal fused multiply-add contraction (the `origin + unit * t` step,
//! the trilinear corner blends, the cone ratio) yet tight enough to fail a
//! wrong port (a swapped trilinear corner, a dropped border clamp, a missing
//! cone running-minimum, a wrong step floor). Several scenarios additionally
//! assert the physical shape of the result — a lit ray near `1.0`, a direct hit
//! at `0.0`, a soft penumbra value strictly between — so a degenerate kernel
//! that returned a constant could not pass.
//!
//! Provenance: standard Inigo-Quilez cone soft-shadow sphere trace; no Unreal
//! Engine source or derived code.

use prism_render_architecture::particle::distance_field_shadow::{
    sphere_trace_shadow, DistanceFieldShadowParams, SdfGrid,
};
use prism_volumetric_gpu::distance_field_shadow::{
    GpuDistanceFieldShadow, GpuSdfGrid, SdfShadowRay,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a genuinely wrong
/// port.
const EPS: f32 = 1.0e-4;

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[-1, 1)`.
fn lcg(state: &mut u64) -> f32 {
    // Knuth multiplier / increment; the shift takes the high bits where the
    // generator mixes best.
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    // 24 usable mantissa bits mapped onto [0, 1) then onto [-1, 1).
    let unit = (bits & 0x00ff_ffff) as f32 / 16_777_216.0;
    unit * 2.0 - 1.0
}

/// The analytic sphere `SDF`: distance to the sphere surface, negative inside.
/// Uses only `sqrt`, matching the ban on transcendentals.
fn sphere_sdf(p: [f32; 3], center: [f32; 3], radius: f32) -> f32 {
    let dx = p[0] - center[0];
    let dy = p[1] - center[1];
    let dz = p[2] - center[2];
    (dx * dx + dy * dy + dz * dz).sqrt() - radius
}

/// Bakes a sphere `SDF` into a `dims` grid over the `[0, extent]` cube.
fn bake_sphere_grid(
    dims: [usize; 3],
    extent: f32,
    center: [f32; 3],
    radius: f32,
) -> Vec<f32> {
    let [nx, ny, nz] = dims;
    let mut data = vec![0.0f32; nx * ny * nz];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let p = [
                    (i as f32) / (nx as f32 - 1.0) * extent,
                    (j as f32) / (ny as f32 - 1.0) * extent,
                    (k as f32) / (nz as f32 - 1.0) * extent,
                ];
                data[i + j * nx + k * nx * ny] = sphere_sdf(p, center, radius);
            }
        }
    }
    data
}

/// Runs the march on both the `CPU` reference and the `GPU` twin for the same
/// baked field, params and rays, asserts ray-for-ray parity, and returns the
/// `GPU` visibilities for any further shape assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuDistanceFieldShadow,
    dims: [usize; 3],
    min: [f32; 3],
    max: [f32; 3],
    data: &[f32],
    params: &DistanceFieldShadowParams,
    rays: &[SdfShadowRay],
) -> Vec<f32> {
    let grid = SdfGrid::new(dims, min, max, data.to_vec()).expect("valid grid");
    let cpu: Vec<f32> = rays
        .iter()
        .map(|r| sphere_trace_shadow(|p| grid.sample(p), r.origin, r.dir, params))
        .collect();

    let gpu_grid = GpuSdfGrid {
        dims,
        min,
        max,
        data,
    };
    let got = gpu.eval(ctx, &gpu_grid, params, rays);
    assert_eq!(got.len(), cpu.len(), "one visibility per ray");
    for (idx, (g, c)) in got.iter().zip(cpu.iter()).enumerate() {
        assert!(
            (g - c).abs() <= EPS,
            "ray {idx} visibility mismatch: gpu {g}, cpu {c}"
        );
        assert!(g.is_finite(), "ray {idx} produced a non-finite visibility");
    }
    got
}

/// Default march params covering the baked-sphere scenes: a generous step
/// budget, a moderate cone softness, a small lift-off and a reach that spans
/// the whole cube.
fn sphere_params() -> DistanceFieldShadowParams {
    // max_steps 128, softness_k 8, min_t 0.05, max_t 6.0, surface_eps 0.02.
    DistanceFieldShadowParams::new(128, 8.0, 0.05, 6.0, 2.0e-2)
}

#[test]
fn empty_rays_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDistanceFieldShadow::new(&ctx);
    let data = bake_sphere_grid([9, 9, 9], 4.0, [2.0, 2.0, 2.0], 0.8);
    let grid = GpuSdfGrid {
        dims: [9, 9, 9],
        min: [0.0; 3],
        max: [4.0; 3],
        data: &data,
    };
    let out = gpu.eval(&ctx, &grid, &sphere_params(), &[]);
    assert!(out.is_empty(), "an empty ray set yields no results");
}

#[test]
fn fully_lit_when_unoccluded() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDistanceFieldShadow::new(&ctx);
    // A field that is uniformly far from any surface: every sample is a large
    // positive distance, so the cone ratio never drops below one and the
    // receiver stays fully lit.
    let dims = [4, 4, 4];
    let data = vec![10.0f32; dims[0] * dims[1] * dims[2]];
    let rays = [
        SdfShadowRay {
            origin: [0.5, 0.5, 0.5],
            dir: [0.0, 0.0, 1.0],
        },
        SdfShadowRay {
            origin: [1.0, 2.0, 0.0],
            dir: [0.0, 1.0, 0.0],
        },
    ];
    let got = check(
        &ctx,
        &gpu,
        dims,
        [0.0; 3],
        [4.0; 3],
        &data,
        &sphere_params(),
        &rays,
    );
    for v in &got {
        assert!((v - 1.0).abs() <= EPS, "unoccluded ray should be fully lit");
    }
}

#[test]
fn hard_shadow_on_direct_hit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDistanceFieldShadow::new(&ctx);
    let dims = [17, 17, 17];
    let center = [2.0, 2.0, 2.0];
    let data = bake_sphere_grid(dims, 4.0, center, 0.8);
    // A ray fired straight through the baked sphere must reach geometry and
    // return fully shadowed.
    let rays = [SdfShadowRay {
        origin: [2.0, 2.0, 0.0],
        dir: [0.0, 0.0, 1.0],
    }];
    let got = check(
        &ctx,
        &gpu,
        dims,
        [0.0; 3],
        [4.0; 3],
        &data,
        &sphere_params(),
        &rays,
    );
    assert!(got[0] <= EPS, "a direct hit should be fully shadowed, got {}", got[0]);
}

#[test]
fn penumbra_soft_transition() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDistanceFieldShadow::new(&ctx);
    let dims = [33, 33, 33];
    let center = [2.0, 2.0, 2.0];
    let radius = 0.6f32;
    let data = bake_sphere_grid(dims, 4.0, center, radius);
    let params = sphere_params();

    // A fan of upward rays sweeping laterally past the sphere: near rays hit
    // (black), far rays clear (white), and the grazing rays in between return a
    // soft grey. Parity must hold for every one, and the fan must contain a
    // genuinely soft value so the penumbra path is exercised.
    let mut rays = Vec::new();
    for step in 0..24u32 {
        let offset = 0.4 + (step as f32) * 0.06;
        rays.push(SdfShadowRay {
            origin: [2.0 + offset, 0.0, 2.0],
            dir: [0.0, 1.0, 0.0],
        });
    }
    let got = check(
        &ctx,
        &gpu,
        dims,
        [0.0; 3],
        [4.0; 3],
        &data,
        &params,
        &rays,
    );
    let has_soft = got.iter().any(|&v| v > 0.05 && v < 0.95);
    assert!(
        has_soft,
        "the grazing fan should produce at least one soft penumbra value"
    );
}

#[test]
fn random_grids_multiple_resolutions() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDistanceFieldShadow::new(&ctx);

    // A spread of resolutions (including non-cubic and a thin slab), each with
    // its own random distance field and random ray bundle, plus a varied param
    // set. Parity alone is asserted — the random field has no analytic shape.
    let cases: [([usize; 3], f32, u64, DistanceFieldShadowParams); 4] = [
        (
            [8, 8, 8],
            4.0,
            0x1234_5678_9abc_def0,
            DistanceFieldShadowParams::new(96, 6.0, 0.05, 5.0, 1.5e-2),
        ),
        (
            [7, 5, 9],
            3.0,
            0x0f0f_0f0f_1234_5678,
            DistanceFieldShadowParams::new(64, 12.0, 0.1, 4.0, 3.0e-2),
        ),
        (
            [12, 6, 3],
            6.0,
            0xdead_beef_cafe_babe,
            DistanceFieldShadowParams::new(200, 3.5, 0.02, 7.0, 1.0e-2),
        ),
        (
            [5, 11, 7],
            2.5,
            0x5555_aaaa_3333_cccc,
            DistanceFieldShadowParams::new(128, 9.0, 0.08, 3.5, 2.5e-2),
        ),
    ];

    for (dims, extent, seed, params) in cases {
        let [nx, ny, nz] = dims;
        let mut state = seed;
        let mut data = vec![0.0f32; nx * ny * nz];
        for slot in &mut data {
            // Distances in roughly [-0.5, 1.5]: a mix of inside (negative) and
            // outside (positive) so rays both hit and graze.
            *slot = lcg(&mut state) + 0.5;
        }

        let mut rays = Vec::new();
        for _ in 0..40 {
            let origin = [
                (lcg(&mut state) * 0.5 + 0.5) * extent,
                (lcg(&mut state) * 0.5 + 0.5) * extent,
                (lcg(&mut state) * 0.5 + 0.5) * extent,
            ];
            let dir = [lcg(&mut state), lcg(&mut state), lcg(&mut state)];
            rays.push(SdfShadowRay { origin, dir });
        }

        check(
            &ctx,
            &gpu,
            dims,
            [0.0; 3],
            [extent; 3],
            &data,
            &params,
            &rays,
        );
    }
}

#[test]
fn degenerate_steps_and_direction() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDistanceFieldShadow::new(&ctx);
    let dims = [9, 9, 9];
    let data = bake_sphere_grid(dims, 4.0, [2.0, 2.0, 2.0], 0.8);

    // A zero step budget leaves the receiver fully lit regardless of the field.
    let zero_steps = DistanceFieldShadowParams::new(0, 8.0, 0.05, 6.0, 2.0e-2);
    // A start already past the reach also leaves it fully lit (the loop never
    // samples).
    let past_reach = DistanceFieldShadowParams::new(128, 8.0, 7.0, 6.0, 2.0e-2);

    // A degenerate (zero-length) direction must be guarded to fully lit, and a
    // normal ray is included so the batch is not trivially all the same.
    let rays = [
        SdfShadowRay {
            origin: [2.0, 2.0, 0.0],
            dir: [0.0, 0.0, 0.0],
        },
        SdfShadowRay {
            origin: [2.0, 2.0, 0.0],
            dir: [0.0, 0.0, 1.0],
        },
    ];

    let lit_zero = check(&ctx, &gpu, dims, [0.0; 3], [4.0; 3], &data, &zero_steps, &rays);
    for v in &lit_zero {
        assert!((v - 1.0).abs() <= EPS, "zero step budget stays fully lit");
    }
    let lit_past = check(&ctx, &gpu, dims, [0.0; 3], [4.0; 3], &data, &past_reach, &rays);
    for v in &lit_past {
        assert!((v - 1.0).abs() <= EPS, "start past reach stays fully lit");
    }

    // Under the normal params the degenerate direction stays lit while the real
    // ray hits the sphere.
    let normal = check(
        &ctx,
        &gpu,
        dims,
        [0.0; 3],
        [4.0; 3],
        &data,
        &sphere_params(),
        &rays,
    );
    assert!((normal[0] - 1.0).abs() <= EPS, "zero direction stays fully lit");
}

#[test]
fn degenerate_grid_axis_collapses() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDistanceFieldShadow::new(&ctx);
    // A grid whose x axis has zero extent: the sample must collapse that axis to
    // grid coordinate zero instead of dividing by zero, and the twin must agree
    // with the reference's identical collapse.
    let dims = [3, 4, 4];
    let [nx, ny, nz] = dims;
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut data = vec![0.0f32; nx * ny * nz];
    for slot in &mut data {
        *slot = lcg(&mut state) * 0.4 + 0.3;
    }
    let rays = [
        SdfShadowRay {
            origin: [0.3, 0.5, 0.5],
            dir: [1.0, 0.2, 0.1],
        },
        SdfShadowRay {
            origin: [0.1, 1.0, 1.5],
            dir: [0.0, 1.0, 0.0],
        },
        SdfShadowRay {
            origin: [0.5, 0.5, 0.0],
            dir: [0.2, 0.3, 1.0],
        },
    ];
    let params = DistanceFieldShadowParams::new(96, 7.0, 0.05, 3.0, 2.0e-2);
    // min.x == max.x == 0.0 gives a degenerate (zero-extent) x axis.
    check(
        &ctx,
        &gpu,
        dims,
        [0.0, 0.0, 0.0],
        [0.0, 2.0, 2.0],
        &data,
        &params,
        &rays,
    );
}
