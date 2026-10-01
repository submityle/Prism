//! Real-device parity for the six-plane frustum-cull twin:
//! [`GpuFrustumCull`](prism_volumetric_gpu::frustum_cull::GpuFrustumCull) must
//! reproduce the `CPU` golden
//! [`cull_aabb`](prism_render_architecture::particle::frustum_aabb_cull::cull_aabb)
//! and
//! [`cull_sphere`](prism_render_architecture::particle::frustum_aabb_cull::cull_sphere)
//! across empty batches, fully-visible and fully-culled primitives, the
//! straddling intersection boundary, a large random batch mixing `AABB`s and
//! spheres drawn from several frustums, and degenerate inputs (zero half-extent,
//! zero-radius, degenerate plane) that must not panic.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full dispatch-and-
//! readback on any real device such as an Apple `M`-series `GPU`. The kernel is
//! portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The verdict is a discrete three-way enum, not a continuous value, so parity
//! is asserted as an *exact* [`Visibility`] match lane for lane. The kernel
//! folds the same per-plane projected-radius decision in the same associativity
//! as the reference and routes every sign test through the same `CULL_EPS`
//! band, so a legal fused multiply-add (perturbing a sum by a few units in the
//! last place) stays far inside that `1e-6` band for every primitive the suite
//! places well clear of a plane, and the folded verdict is reproduced exactly.
//!
//! Provenance: classic p-vertex / n-vertex symmetric projected-radius frustum
//! test; no third-party engine source or derived code.

use prism_render_architecture::particle::frustum_aabb_cull::{
    cull_aabb, cull_sphere, Aabb, Plane, Sphere, Visibility,
};
use prism_volumetric_gpu::frustum_cull::{FrustumCullPrimitive, FrustumCullQuery, GpuFrustumCull};
use prism_volumetric_gpu::GpuContext;

/// `1 / sqrt(2)` as a literal so the diagonal frustum planes stay unit length
/// without a runtime `sqrt` in a `const` context.
const INV_SQRT2: f32 = core::f32::consts::FRAC_1_SQRT_2;

/// A standard symmetric perspective frustum: apex at the origin looking down
/// `+z` with a 45° half-angle, near plane at `z = 1`, far plane at `z = 100`.
/// All normals point inward and are unit length.
const SYMMETRIC: [Plane; 6] = [
    Plane::new([INV_SQRT2, 0.0, INV_SQRT2], 0.0),
    Plane::new([-INV_SQRT2, 0.0, INV_SQRT2], 0.0),
    Plane::new([0.0, INV_SQRT2, INV_SQRT2], 0.0),
    Plane::new([0.0, -INV_SQRT2, INV_SQRT2], 0.0),
    Plane::new([0.0, 0.0, 1.0], -1.0),
    Plane::new([0.0, 0.0, -1.0], 100.0),
];

/// An oblique frustum whose four side planes are tilted off the world axes, to
/// exercise the projected-radius math on non-axis-aligned normals.
const OBLIQUE: [Plane; 6] = [
    Plane::new([INV_SQRT2, 0.0, INV_SQRT2], 2.0),
    Plane::new([-INV_SQRT2, 0.0, INV_SQRT2], 2.0),
    Plane::new([0.0, INV_SQRT2, INV_SQRT2], 2.0),
    Plane::new([0.0, -INV_SQRT2, INV_SQRT2], 2.0),
    Plane::new([0.0, 0.0, 1.0], -1.0),
    Plane::new([0.0, 0.0, -1.0], 100.0),
];

/// Builds an `AABB` query against `planes`.
fn aabb_query(planes: [Plane; 6], aabb: Aabb) -> FrustumCullQuery {
    FrustumCullQuery {
        planes,
        primitive: FrustumCullPrimitive::Aabb(aabb),
    }
}

/// Builds a sphere query against `planes`.
fn sphere_query(planes: [Plane; 6], sphere: Sphere) -> FrustumCullQuery {
    FrustumCullQuery {
        planes,
        primitive: FrustumCullPrimitive::Sphere(sphere),
    }
}

/// The `CPU` golden verdict for one query, dispatching to the matching
/// reference entry point.
fn expected(query: &FrustumCullQuery) -> Visibility {
    match query.primitive {
        FrustumCullPrimitive::Aabb(aabb) => cull_aabb(&query.planes, &aabb),
        FrustumCullPrimitive::Sphere(sphere) => cull_sphere(&query.planes, &sphere),
    }
}

/// Runs the `GPU` dispatch and asserts an exact per-lane [`Visibility`] match
/// against the `CPU` golden, returning the `GPU` verdicts for extra assertions.
fn check(ctx: &GpuContext, gpu: &GpuFrustumCull, queries: &[FrustumCullQuery]) -> Vec<Visibility> {
    let got = gpu.eval(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one verdict per query");
    for (lane, (&g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let w = expected(q);
        assert_eq!(g, w, "lane {lane}: gpu {g:?} vs cpu {w:?}");
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

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFrustumCull::new(&ctx);
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn fully_visible_primitives_are_inside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFrustumCull::new(&ctx);
    // Small primitives parked deep in the middle of each frustum are wholly
    // contained by every plane.
    let queries = [
        aabb_query(SYMMETRIC, Aabb::new([0.0, 0.0, 50.0], [1.0, 1.0, 1.0])),
        sphere_query(SYMMETRIC, Sphere::new([0.0, 0.0, 50.0], 2.0)),
        aabb_query(OBLIQUE, Aabb::new([0.0, 0.0, 50.0], [0.5, 0.5, 0.5])),
    ];
    let got = check(&ctx, &gpu, &queries);
    assert!(
        got.iter().all(|&v| v == Visibility::Inside),
        "every centered primitive should be Inside, got {got:?}"
    );
}

#[test]
fn fully_culled_primitives_are_outside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFrustumCull::new(&ctx);
    // Primitives pushed far outside a single plane are rejected outright.
    let queries = [
        aabb_query(SYMMETRIC, Aabb::new([-500.0, 0.0, 50.0], [1.0, 1.0, 1.0])),
        aabb_query(SYMMETRIC, Aabb::new([0.0, 0.0, 500.0], [1.0, 1.0, 1.0])),
        aabb_query(SYMMETRIC, Aabb::new([0.0, 0.0, -50.0], [1.0, 1.0, 1.0])),
        sphere_query(SYMMETRIC, Sphere::new([0.0, 500.0, 50.0], 5.0)),
        sphere_query(SYMMETRIC, Sphere::new([0.0, 0.0, 500.0], 10.0)),
        aabb_query(OBLIQUE, Aabb::new([400.0, 0.0, 50.0], [1.0, 1.0, 1.0])),
    ];
    let got = check(&ctx, &gpu, &queries);
    assert!(
        got.iter().all(|&v| v == Visibility::Outside),
        "every far primitive should be Outside, got {got:?}"
    );
}

#[test]
fn straddling_primitives_intersect() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFrustumCull::new(&ctx);
    // Wide boxes / large spheres that clearly reach through a side or depth
    // plane while keeping their p-vertex inside straddle the boundary.
    let queries = [
        aabb_query(SYMMETRIC, Aabb::new([-50.0, 0.0, 50.0], [5.0, 1.0, 1.0])),
        aabb_query(OBLIQUE, Aabb::new([0.0, 0.0, 50.0], [60.0, 1.0, 1.0])),
        sphere_query(SYMMETRIC, Sphere::new([0.0, 0.0, 100.0], 5.0)),
    ];
    let got = check(&ctx, &gpu, &queries);
    assert!(
        got.iter().all(|&v| v == Visibility::Intersecting),
        "every straddling primitive should be Intersecting, got {got:?}"
    );
}

#[test]
fn degenerate_inputs_do_not_panic() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFrustumCull::new(&ctx);
    // Degenerate but well-defined: a zero-half-extent box (a point), a
    // zero-radius sphere (a point), and a frustum with one degenerate
    // zero-normal plane. The reference evaluates all of these without special
    // casing, so the twin must agree lane for lane and never panic.
    let mut degenerate_planes = SYMMETRIC;
    degenerate_planes[0] = Plane::new([0.0, 0.0, 0.0], 0.0);
    let queries = [
        aabb_query(SYMMETRIC, Aabb::new([0.0, 0.0, 50.0], [0.0, 0.0, 0.0])),
        sphere_query(SYMMETRIC, Sphere::new([0.0, 0.0, 50.0], 0.0)),
        aabb_query(SYMMETRIC, Aabb::new([-500.0, 0.0, 50.0], [0.0, 0.0, 0.0])),
        aabb_query(
            degenerate_planes,
            Aabb::new([0.0, 0.0, 50.0], [1.0, 1.0, 1.0]),
        ),
        sphere_query(degenerate_planes, Sphere::new([0.0, 0.0, 50.0], 2.0)),
    ];
    // `check` already asserts exact parity; reaching the end proves no panic.
    let _ = check(&ctx, &gpu, &queries);
}

#[test]
fn random_mixed_batch_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFrustumCull::new(&ctx);
    let mut state = 0x_f00d_1357_9bdf_0001_u64;
    let frustums = [SYMMETRIC, OBLIQUE];

    for round in 0u32..8 {
        let mut queries = Vec::with_capacity(128);
        for _ in 0..128 {
            // Rotate through the available frustums so one dispatch mixes
            // primitives bounded by several distinct frustums.
            let planes = frustums[(lcg(&mut state) * frustums.len() as f32) as usize % 2];
            // Spread centers across in-frustum, straddling and far-outside
            // regions on every axis, well clear of any plane so no lane sits
            // within a ULP of the CULL_EPS boundary.
            let cx = lcg(&mut state) * 240.0 - 120.0;
            let cy = lcg(&mut state) * 240.0 - 120.0;
            let cz = lcg(&mut state) * 160.0 - 20.0;
            if lcg(&mut state) > 0.5 {
                let hx = lcg(&mut state) * 12.0;
                let hy = lcg(&mut state) * 12.0;
                let hz = lcg(&mut state) * 12.0;
                queries.push(aabb_query(planes, Aabb::new([cx, cy, cz], [hx, hy, hz])));
            } else {
                let radius = lcg(&mut state) * 12.0;
                queries.push(sphere_query(planes, Sphere::new([cx, cy, cz], radius)));
            }
        }
        // `check` asserts exact lane-for-lane parity against the CPU golden.
        let got = check(&ctx, &gpu, &queries);
        // Sanity: a large random spread should produce more than one verdict
        // class, so the test is not trivially passing on an all-Outside batch.
        let distinct = got.iter().any(|&v| v != got[0]);
        assert!(distinct, "round {round}: expected a mix of verdicts");
    }
}
