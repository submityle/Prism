//! Real-device parity for the per-triangle mass-property contribution twin:
//! [`GpuMeshVolumeContribution`](prism_volumetric_gpu::mesh_volume_contribution::GpuMeshVolumeContribution)
//! must reproduce, for one triangle per query, the closed-form contribution the
//! golden `mass_properties` adds into its running sums for that triangle: the
//! surface area `0.5 * |(b - a) x (c - a)|`, the signed tetrahedron volume
//! `det(a, b, c) / 6`, the first moment `signed_volume * (a + b + c) / 4` and
//! the second-moment (covariance) matrix `det * A * Ccanon * A^T` (Blow &
//! Binstock).
//!
//! # Independent oracle
//!
//! This suite does not depend on the reference crate. The host [`oracle`] is an
//! independent `f32` reimplementation of the same per-triangle closed form
//! documented on the twin, evaluated in the same multiply-add order as the
//! kernel. The golden path accumulates in `f64` and performs a whole-mesh
//! reduction (summing across triangles, tallying directed edges for the
//! watertight test and deriving the centroid and inertia tensor); none of that
//! reduction is twinned, so the oracle here mirrors only the per-triangle
//! arithmetic in `f32` for a fair device-against-device check. A passing run is
//! therefore evidence that the `WGSL` kernel and an independent `CPU`
//! evaluation of the same contribution agree, not merely that the shader
//! compiles.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and a
//! single `sqrt` for the area, so the two evaluations compute the same closed
//! form in the same order. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar host leaves separate, perturbing the low mantissa
//! bits by a few units in the last place. The comparison therefore allows
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on every one
//! of the `14` continuous outputs. There are no discrete outputs.
//!
//! # Conditioning
//!
//! The random sweep is rejection-sampled well away from the only numerically
//! delicate boundary — a near-degenerate (collinear) triangle whose
//! cross-product magnitude approaches zero — so the area `sqrt` stays
//! well-conditioned and `CPU` and `GPU` agree to the documented tolerance. The
//! named degenerate fixture instead pins a deliberately collinear triangle,
//! verifying the area contribution collapses to zero on both evaluators.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_mass_properties`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::mesh_volume_contribution::{
    GpuMeshVolumeContribution, MeshVolumeContributionQuery, MeshVolumeContributionResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity floor on every continuous output.
const ABS_TOL: f32 = 1.0e-4;
/// Relative parity slope on every continuous output.
const REL_TOL: f32 = 1.0e-3;
/// Relative-tolerance floor so near-zero magnitudes stay meaningful.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the documented parity bound: an
/// absolute floor or a relative term keeping large-magnitude values meaningful.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    diff <= ABS_TOL || diff <= REL_TOL * a.abs().max(b.abs()).max(REL_FLOOR)
}

/// Component-wise difference `a - b` over a 3-vector.
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Cross product of two 3-vectors.
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Dot product of two 3-vectors.
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Independent `f32` host reimplementation of the per-triangle contribution,
/// evaluated in the same multiply-add order as the kernel.
fn oracle(q: &MeshVolumeContributionQuery) -> MeshVolumeContributionResult {
    let a = q.a;
    let b = q.b;
    let c = q.c;

    // Surface area via half the cross-product magnitude.
    let ab = sub3(b, a);
    let ac = sub3(c, a);
    let n = cross3(ab, ac);
    let area = 0.5 * dot3(n, n).sqrt();

    // Determinant of the matrix whose columns are a, b, c, same term order as
    // the golden path and the kernel.
    let det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0]);
    let tet_volume = det / 6.0;

    // Tetrahedron centroid (one vertex is the origin) is (a + b + c) / 4.
    let tc = [
        (a[0] + b[0] + c[0]) / 4.0,
        (a[1] + b[1] + c[1]) / 4.0,
        (a[2] + b[2] + c[2]) / 4.0,
    ];
    let moment1 = [tet_volume * tc[0], tet_volume * tc[1], tet_volume * tc[2]];

    // Canonical reference covariance scaled by 1/120 (Blow & Binstock).
    let c2 = 2.0 / 120.0;
    let c1 = 1.0 / 120.0;
    // cat[k] = sum_m Ccanon[k][m] * col[m], col[0]=a, col[1]=b, col[2]=c.
    let cat0 = [
        c2 * a[0] + c1 * b[0] + c1 * c[0],
        c2 * a[1] + c1 * b[1] + c1 * c[1],
        c2 * a[2] + c1 * b[2] + c1 * c[2],
    ];
    let cat1 = [
        c1 * a[0] + c2 * b[0] + c1 * c[0],
        c1 * a[1] + c2 * b[1] + c1 * c[1],
        c1 * a[2] + c2 * b[2] + c1 * c[2],
    ];
    let cat2 = [
        c1 * a[0] + c1 * b[0] + c2 * c[0],
        c1 * a[1] + c1 * b[1] + c2 * c[1],
        c1 * a[2] + c1 * b[2] + c2 * c[2],
    ];
    // row i = det * (a[i]*cat0 + b[i]*cat1 + c[i]*cat2), ranging over j.
    let row = |i: usize| {
        [
            det * (a[i] * cat0[0] + b[i] * cat1[0] + c[i] * cat2[0]),
            det * (a[i] * cat0[1] + b[i] * cat1[1] + c[i] * cat2[1]),
            det * (a[i] * cat0[2] + b[i] * cat1[2] + c[i] * cat2[2]),
        ]
    };
    let r0 = row(0);
    let r1 = row(1);
    let r2 = row(2);
    let moment2 = [
        r0[0], r0[1], r0[2], r1[0], r1[1], r1[2], r2[0], r2[1], r2[2],
    ];

    MeshVolumeContributionResult {
        area,
        signed_volume: tet_volume,
        moment1,
        moment2,
    }
}

/// Pins one `GPU` result against the independent host oracle, element for
/// element across all `14` continuous outputs.
fn pin(idx: usize, query: &MeshVolumeContributionQuery, result: &MeshVolumeContributionResult) {
    let want = oracle(query);
    assert!(
        close(result.area, want.area),
        "query {idx}: area gpu={} oracle={}",
        result.area,
        want.area
    );
    assert!(
        close(result.signed_volume, want.signed_volume),
        "query {idx}: signed_volume gpu={} oracle={}",
        result.signed_volume,
        want.signed_volume
    );
    for (lane, (&g, &w)) in result.moment1.iter().zip(want.moment1.iter()).enumerate() {
        assert!(
            close(g, w),
            "query {idx}: moment1[{lane}] gpu={g} oracle={w}"
        );
    }
    for (lane, (&g, &w)) in result.moment2.iter().zip(want.moment2.iter()).enumerate() {
        assert!(
            close(g, w),
            "query {idx}: moment2[{lane}] gpu={g} oracle={w}"
        );
    }
}

/// Evaluates `queries` on-device and pins every result against the oracle.
fn check(
    ctx: &GpuContext,
    gpu: &GpuMeshVolumeContribution,
    queries: &[MeshVolumeContributionQuery],
) {
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

/// 64-bit linear-congruential step (`Knuth`/`PCG` constants), returning the
/// high word so the stream has good spread without any transcendental math.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 32) as u32
}

/// A deterministic pseudo-random `f32` in `[0, 1]`.
fn unit01(state: &mut u64) -> f32 {
    lcg(state) as f32 / u32::MAX as f32
}

/// A deterministic pseudo-random `f32` in `[-scale, scale]`.
fn signed(state: &mut u64, scale: f32) -> f32 {
    (unit01(state) * 2.0 - 1.0) * scale
}

/// A deterministic pseudo-random vertex inside the `[-50, 50]` cube.
fn rand_vertex(state: &mut u64) -> [f32; 3] {
    [
        signed(state, 50.0),
        signed(state, 50.0),
        signed(state, 50.0),
    ]
}

/// A deterministic pseudo-random triangle, rejection-sampled away from the
/// collinear (near-zero area) boundary so the area `sqrt` stays well
/// conditioned and both evaluators agree to tolerance.
fn rand_triangle(state: &mut u64) -> MeshVolumeContributionQuery {
    loop {
        let a = rand_vertex(state);
        let b = rand_vertex(state);
        let c = rand_vertex(state);
        let n = cross3(sub3(b, a), sub3(c, a));
        // Keep the triangle comfortably non-degenerate: the cross-product
        // magnitude is twice the area, so a healthy floor keeps the area and
        // its sqrt far from the collinear singularity.
        if dot3(n, n) > 25.0 {
            return MeshVolumeContributionQuery::new(a, b, c);
        }
    }
}

/// A triangle taken from a face of the unit box.
fn unit_box_triangle() -> MeshVolumeContributionQuery {
    MeshVolumeContributionQuery::new([1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0])
}

/// A collinear triangle whose surface area contribution must collapse to zero.
fn degenerate_collinear() -> MeshVolumeContributionQuery {
    MeshVolumeContributionQuery::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [5.0, 5.0, 5.0])
}

/// A triangle whose winding yields a negative determinant (and so a negative
/// signed volume contribution).
fn inward_winding() -> MeshVolumeContributionQuery {
    MeshVolumeContributionQuery::new([3.0, 1.0, 2.0], [2.0, 0.0, 5.0], [1.0, 4.0, 0.0])
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshVolumeContribution::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn unit_box_triangle_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshVolumeContribution::new(&ctx);
    check(&ctx, &gpu, &[unit_box_triangle()]);
}

#[test]
fn degenerate_zero_area_triangle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshVolumeContribution::new(&ctx);
    let query = degenerate_collinear();
    let got = gpu.evaluate(&ctx, &[query]);
    assert_eq!(got.len(), 1, "one result for one query");
    // A collinear triangle has zero cross-product magnitude, so its area
    // contribution must vanish on-device as well.
    assert!(
        close(got[0].area, 0.0),
        "collinear triangle area should collapse to zero, got {}",
        got[0].area
    );
    pin(0, &query, &got[0]);
}

#[test]
fn inward_winding_negative_determinant() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshVolumeContribution::new(&ctx);
    let query = inward_winding();
    let want = oracle(&query);
    // Sanity-check the fixture actually exercises a negative signed volume so
    // the sign path is covered, not just the magnitude.
    assert!(
        want.signed_volume < 0.0,
        "fixture should yield a negative signed volume, got {}",
        want.signed_volume
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshVolumeContribution::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing the deterministic fixtures with many random triangles,
    // dispatched together so the per-thread indexing and the contiguous storage
    // layout are both exercised, then pinned element for element.
    let mut queries = vec![
        unit_box_triangle(),
        degenerate_collinear(),
        inward_winding(),
    ];
    for _ in 0..48 {
        queries.push(rand_triangle(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshVolumeContribution::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep of clearly-conditioned random triangles (several
    // workgroups' worth) pins all 14 contribution outputs across many frames.
    let queries: Vec<MeshVolumeContributionQuery> =
        (0..512).map(|_| rand_triangle(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
