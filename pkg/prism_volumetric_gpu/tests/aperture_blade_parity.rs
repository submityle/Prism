//! Real-device parity for the aperture-blade-`SDF` twin:
//! [`GpuApertureBlade`](prism_volumetric_gpu::aperture_blade::GpuApertureBlade)
//! must reproduce the `CPU` golden
//! [`aperture_blade`](prism_render_architecture::particle::aperture_blade)
//! across a point deep inside a square aperture (negative distance), a point
//! well outside it (positive distance), the circle-blend limit (`roundness`
//! `1.0`), an intermediate blend (`roundness` `0.3`), a richer eight-blade
//! aperture, an empty-blade aperture that degenerates to a circle, and a
//! randomized batch compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and one
//! `sqrt`, so `CPU` and `GPU` evaluate the same closed form in the same order.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits by a few units in the last
//! place. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` on the `f32` distance.
//!
//! # Conditioning
//!
//! Every deterministic fixture keeps its query point away from an edge
//! zero-line and away from a polygon vertex by a clear margin, so a few units
//! in the last place never flip which half-plane the `max` fold selects. The
//! random batch draws edge normals, inradius, roundness and point from
//! well-spread ranges and rejects any point whose two largest half-plane
//! distances are within a small margin, keeping both devices on the same fold
//! winner regardless of a little floating-point slack. All fixtures stay within
//! the valid `apothem >= 0`, `roundness` in `[0, 1]` domain so the host-side
//! clamp in the reference constructor is a no-op and the un-clamped kernel
//! matches it exactly.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::aperture_blade`；
//! no third-party engine source or derived code.

use prism_volumetric_gpu::aperture_blade::{
    golden, ApertureBladeQuery, ApertureBladeResult, GpuApertureBlade, MAX_BLADES,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in
/// the last place exceed the absolute floor.
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

/// Unit outward normals of an axis-aligned square aperture (four blades).
fn square_normals() -> Vec<[f32; 2]> {
    vec![[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
}

/// Unit outward normals of an eight-blade aperture built from exact 3-4-5 unit
/// vectors, so every normal is unit without calling trigonometry.
fn octagon_normals() -> Vec<[f32; 2]> {
    vec![
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [0.6, 0.8],
        [0.6, -0.8],
        [-0.6, 0.8],
        [-0.6, -0.8],
    ]
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

/// Builds a clearly-conditioned random query by rejection sampling: a handful
/// of random (not necessarily unit) edge normals, a positive inradius, a
/// roundness in `[0, 1]` and a point whose two largest half-plane distances are
/// comfortably apart, so the `max` fold is never on a tie. The kernel and the
/// reference both read the normals verbatim, so unit length is not required for
/// parity.
fn rand_query(state: &mut u64) -> ApertureBladeQuery {
    loop {
        let blades = 3 + ((*state >> 7) as usize % (MAX_BLADES - 2));
        let edge_normals: Vec<[f32; 2]> = (0..blades)
            .map(|_| [signed(state, 1.0), signed(state, 1.0)])
            .collect();
        let apothem = lcg(state) * 1.8 + 0.2;
        let roundness = lcg(state);
        let point = [signed(state, 3.0), signed(state, 3.0)];

        // Reject when the two largest half-plane distances are within a small
        // margin, so both devices select the same fold winner.
        let mut best = f32::NEG_INFINITY;
        let mut second = f32::NEG_INFINITY;
        for normal in &edge_normals {
            let d = point[0] * normal[0] + point[1] * normal[1];
            if d > best {
                second = best;
                best = d;
            } else if d > second {
                second = d;
            }
        }
        if best - second < 1.0e-2 {
            continue;
        }
        return ApertureBladeQuery::new(edge_normals, apothem, roundness, point);
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: the rounded
/// aperture signed distance must agree within bound.
fn pin(idx: usize, query: &ApertureBladeQuery, got: &ApertureBladeResult) {
    let want = golden(query);
    assert!(
        close(got.sdf, want.sdf),
        "query {idx} sdf: gpu {} vs cpu {}",
        got.sdf,
        want.sdf
    );
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuApertureBlade, queries: &[ApertureBladeQuery]) {
    let got = gpu.eval(ctx, queries);
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
    let gpu = GpuApertureBlade::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn inside_square_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuApertureBlade::new(&ctx);
    // Deep inside the unit-apothem square, clear of every edge zero-line: the
    // largest half-plane distance is 0.3, so the polygon SDF is 0.3 - 1 = -0.7.
    let query = ApertureBladeQuery::new(square_normals(), 1.0, 0.0, [0.3, 0.2]);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn outside_square_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuApertureBlade::new(&ctx);
    // Well outside the right edge, clear of the corner: the x half-plane wins at
    // distance 2.3, so the polygon SDF is 2.3 - 1 = 1.3.
    let query = ApertureBladeQuery::new(square_normals(), 1.0, 0.0, [2.3, 0.2]);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn roundness_one_is_circle_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuApertureBlade::new(&ctx);
    // At roundness 1.0 the blend returns the circle SDF, length(p) - apothem.
    // A 3-4-5 offset keeps the length exact: 5 * 0.5 = 2.5 off, apothem 1.5.
    let query = ApertureBladeQuery::new(square_normals(), 1.5, 1.0, [1.5, 2.0]);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn roundness_blend_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuApertureBlade::new(&ctx);
    // An intermediate 0.3 blend of the polygon and circle fields, with the point
    // clear of every edge zero-line and vertex.
    let query = ApertureBladeQuery::new(square_normals(), 1.0, 0.3, [1.4, 0.3]);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn octagon_aperture_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuApertureBlade::new(&ctx);
    // A richer eight-blade aperture at a mild 0.3 roundness, exercising a longer
    // max fold over the packed normal lanes.
    let query = ApertureBladeQuery::new(octagon_normals(), 1.2, 0.3, [0.7, 0.35]);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn empty_blades_fall_back_to_circle_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuApertureBlade::new(&ctx);
    // With no blade the polygon term collapses to the circle term, so the result
    // is the circle SDF regardless of roundness. A 3-4-5 offset keeps it exact.
    let query = ApertureBladeQuery::new(Vec::new(), 1.0, 0.0, [3.0, 4.0]);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuApertureBlade::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing deterministic fixtures with many random queries,
    // dispatched together so the per-thread indexing and the contiguous storage
    // layout are both exercised, then pinned element-for-element.
    let mut queries = vec![
        ApertureBladeQuery::new(square_normals(), 1.0, 0.0, [0.3, 0.2]),
        ApertureBladeQuery::new(octagon_normals(), 1.2, 0.3, [0.7, 0.35]),
    ];
    for _ in 0..48 {
        queries.push(rand_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_queries_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuApertureBlade::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) pins the signed distance across
    // many random aperture geometries.
    let queries: Vec<ApertureBladeQuery> = (0..200).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
