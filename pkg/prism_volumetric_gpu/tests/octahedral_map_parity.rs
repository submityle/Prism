//! Real-device parity for the octahedral direction/square map twin:
//! [`GpuOctahedralMap`](prism_volumetric_gpu::octahedral_map::GpuOctahedralMap)
//! must reproduce, for one query per thread, all three closed forms of the
//! golden `prism_render_architecture::reference_pt::octahedral` module: the
//! forward `direction_to_square` `L1` projection (lower hemisphere reflected
//! across the diagonals), the inverse `square_to_direction` reconstruction and
//! normalization, and the solid-angle Jacobian `|d|_1^3`.
//!
//! # Independent oracle
//!
//! This suite does not depend on the reference crate. The host [`oracle`] is an
//! independent `f32` reimplementation of the same closed form documented on the
//! twin, evaluated in the same multiply-add order as the kernel. The golden
//! `sign_unit` is `copysign(1.0, x)`; the oracle uses `f32::copysign` so a
//! negative zero on a fold seam takes the same branch as the kernel's sign-bit
//! test. A passing run is therefore evidence that the `WGSL` kernel and an
//! independent `CPU` evaluation of the same maps agree, not merely that the
//! shader compiles.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of adds, multiplies,
//! absolute values, sign copies and one `sqrt`, so the two evaluations compute
//! the same closed form in the same order. They are not bit-exact: a `GPU` may
//! fuse a multiply-add the scalar host leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The comparison therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on every
//! continuous output and an exact `==` on the discrete `valid` flag.
//!
//! # Conditioning
//!
//! Both maps branch on an ordered compare: the forward on `dir.z >= 0`, the
//! inverse on `z = 1 - |u| - |v| >= 0`. The forward predicate reads an input
//! component verbatim, so it never diverges; the inverse height is a derived
//! sum, so the random sweep is rejection-sampled with `|dir.z|` comfortably away
//! from zero. Setting each sweep query's `(u, v)` to the host forward square of
//! its direction makes the inverse height exactly `|dir.z| / |d|_1` (up to the
//! fold sign), so that margin keeps both evaluators on the same wedge and the
//! round trip well away from the diagonal seams. The named zero-direction
//! fixture instead pins the degenerate case, where the forward square and the
//! Jacobian collapse to an exact zero on both evaluators.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::reference_pt::octahedral`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::octahedral_map::{
    GpuOctahedralMap, OctahedralMapQuery, OctahedralMapResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity floor on every continuous output.
const ABS_TOL: f32 = 1.0e-4;
/// Relative parity slope on every continuous output.
const REL_TOL: f32 = 1.0e-3;
/// Relative-tolerance floor so near-zero magnitudes stay meaningful.
const REL_FLOOR: f32 = 1.0e-6;
/// Squared-length threshold below which the inverse map returns the zero
/// vector, matching the golden `EPS_LEN_SQ`.
const EPS_LEN_SQ: f32 = 1.0e-12;

/// Returns whether `a` and `b` agree within the documented parity bound: an
/// absolute floor or a relative term keeping large-magnitude values meaningful.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    diff <= ABS_TOL || diff <= REL_TOL * a.abs().max(b.abs()).max(REL_FLOOR)
}

/// `copysign(1.0, x)`: `+1` for a non-negative component (including `+0`), `-1`
/// for a negative component (including `-0`). Mirrors the golden `sign_unit`.
fn sign_unit(x: f32) -> f32 {
    1.0_f32.copysign(x)
}

/// Independent `f32` reimplementation of `direction_to_square`, in the same
/// arithmetic order as the kernel.
fn direction_to_square(dir: [f32; 3]) -> [f32; 2] {
    let l1 = dir[0].abs() + dir[1].abs() + dir[2].abs();
    let inv = if l1 > 0.0 { 1.0 / l1 } else { 0.0 };
    let px = dir[0] * inv;
    let py = dir[1] * inv;
    if dir[2] >= 0.0 {
        [px, py]
    } else {
        [
            (1.0 - py.abs()) * sign_unit(px),
            (1.0 - px.abs()) * sign_unit(py),
        ]
    }
}

/// Independent `f32` reimplementation of `square_to_direction` (with the
/// `normalize_or_zero` guard), in the same arithmetic order as the kernel.
fn square_to_direction(u: f32, v: f32) -> [f32; 3] {
    let z = 1.0 - u.abs() - v.abs();
    let (x, y) = if z >= 0.0 {
        (u, v)
    } else {
        (
            (1.0 - v.abs()) * sign_unit(u),
            (1.0 - u.abs()) * sign_unit(v),
        )
    };
    let len2 = x * x + y * y + z * z;
    if len2 > EPS_LEN_SQ {
        let inv_len = 1.0 / len2.sqrt();
        [x * inv_len, y * inv_len, z * inv_len]
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// Independent `f32` reimplementation of `solid_angle_jacobian`.
fn solid_angle_jacobian(dir: [f32; 3]) -> f32 {
    let l1 = dir[0].abs() + dir[1].abs() + dir[2].abs();
    l1 * l1 * l1
}

/// Builds the full host-side reference result for one query.
fn oracle(query: &OctahedralMapQuery) -> OctahedralMapResult {
    OctahedralMapResult {
        square: direction_to_square(query.dir),
        direction: square_to_direction(query.u, query.v),
        jacobian: solid_angle_jacobian(query.dir),
        valid: 1,
    }
}

/// Pins one `GPU` result against the independent host oracle, element for
/// element across every continuous output plus the discrete `valid` flag.
fn pin(idx: usize, query: &OctahedralMapQuery, result: &OctahedralMapResult) {
    let want = oracle(query);
    assert!(
        close(result.square[0], want.square[0]),
        "query {idx}: square u gpu={} oracle={}",
        result.square[0],
        want.square[0]
    );
    assert!(
        close(result.square[1], want.square[1]),
        "query {idx}: square v gpu={} oracle={}",
        result.square[1],
        want.square[1]
    );
    for (lane, (&g, &w)) in result
        .direction
        .iter()
        .zip(want.direction.iter())
        .enumerate()
    {
        assert!(
            close(g, w),
            "query {idx}: direction[{lane}] gpu={g} oracle={w}"
        );
    }
    assert!(
        close(result.jacobian, want.jacobian),
        "query {idx}: jacobian gpu={} oracle={}",
        result.jacobian,
        want.jacobian
    );
    assert_eq!(
        result.valid, want.valid,
        "query {idx}: valid gpu={} oracle={}",
        result.valid, want.valid
    );
}

/// Evaluates `queries` on-device and pins every result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuOctahedralMap, queries: &[OctahedralMapQuery]) {
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

/// Builds a round-trip query: the inverse-map square point is set to the host
/// forward square of `dir`, so `square_to_direction` reconstructs the
/// normalized direction.
fn round_trip_query(dir: [f32; 3]) -> OctahedralMapQuery {
    let sq = direction_to_square(dir);
    OctahedralMapQuery::new(dir, sq[0], sq[1])
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

/// A deterministic pseudo-random `f32` in `[-1, 1]`.
fn signed(state: &mut u64) -> f32 {
    unit01(state) * 2.0 - 1.0
}

/// A deterministic pseudo-random unit direction, rejection-sampled from the
/// cube and kept comfortably away from the `dir.z = 0` equator seam so both
/// evaluators take the same forward and inverse branch.
fn rand_unit_dir(state: &mut u64) -> [f32; 3] {
    loop {
        let x = signed(state);
        let y = signed(state);
        let z = signed(state);
        let r2 = x * x + y * y + z * z;
        if r2 > 1.0e-6 && r2 <= 1.0 {
            let inv = 1.0 / r2.sqrt();
            let d = [x * inv, y * inv, z * inv];
            // Keep |dir.z| away from the equator so the inverse fold height
            // |dir.z| / |d|_1 stays well clear of zero.
            if d[2].abs() > 0.05 {
                return d;
            }
        }
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOctahedralMap::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn plus_z_maps_to_square_center() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOctahedralMap::new(&ctx);
    // +Z sits at the square centre: both forward coordinates vanish.
    let query = OctahedralMapQuery::new([0.0, 0.0, 1.0], 0.0, 0.0);
    let got = gpu.evaluate(&ctx, &[query]);
    assert_eq!(got.len(), 1, "one result for one query");
    assert!(
        close(got[0].square[0], 0.0) && close(got[0].square[1], 0.0),
        "+Z should map to the square centre, got {:?}",
        got[0].square
    );
    pin(0, &query, &got[0]);
}

#[test]
fn equator_axes_map_to_edge_midpoints() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOctahedralMap::new(&ctx);
    // The equator axes land on the edge midpoints of the square.
    let queries = [
        round_trip_query([1.0, 0.0, 0.0]),
        round_trip_query([-1.0, 0.0, 0.0]),
        round_trip_query([0.0, 1.0, 0.0]),
        round_trip_query([0.0, -1.0, 0.0]),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len());
    // +X -> (1, 0).
    assert!(
        close(got[0].square[0], 1.0) && close(got[0].square[1], 0.0),
        "+X should map to (1, 0), got {:?}",
        got[0].square
    );
    for (idx, (q, r)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, q, r);
    }
}

#[test]
fn minus_z_spreads_to_corners() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOctahedralMap::new(&ctx);
    // -Z spreads to the square corners: both coordinates saturate to +/-1.
    let query = OctahedralMapQuery::new([0.0, 0.0, -1.0], 0.0, 0.0);
    let got = gpu.evaluate(&ctx, &[query]);
    assert_eq!(got.len(), 1, "one result for one query");
    assert!(
        close(got[0].square[0].abs(), 1.0) && close(got[0].square[1].abs(), 1.0),
        "-Z should map to a corner, got {:?}",
        got[0].square
    );
    pin(0, &query, &got[0]);
}

#[test]
fn zero_dir_degenerate() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOctahedralMap::new(&ctx);
    // A zero direction maps to the square origin with a zero Jacobian; the
    // independent inverse point (0.3, 0.2) still reconstructs a valid direction.
    let query = OctahedralMapQuery::new([0.0, 0.0, 0.0], 0.3, 0.2);
    let got = gpu.evaluate(&ctx, &[query]);
    assert_eq!(got.len(), 1, "one result for one query");
    // The forward square and the Jacobian are products of exact zeros, so they
    // are exactly zero on-device too.
    assert_eq!(
        got[0].square[0], 0.0,
        "zero direction forward u must be exactly zero"
    );
    assert_eq!(
        got[0].square[1], 0.0,
        "zero direction forward v must be exactly zero"
    );
    assert_eq!(
        got[0].jacobian, 0.0,
        "zero direction Jacobian must be exactly zero"
    );
    pin(0, &query, &got[0]);
}

#[test]
fn round_trip_dir_to_square_to_dir() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOctahedralMap::new(&ctx);
    // Forward then inverse recovers the normalized direction.
    let dir = [0.37, -0.52, 0.68];
    let query = round_trip_query(dir);
    let got = gpu.evaluate(&ctx, &[query]);
    assert_eq!(got.len(), 1, "one result for one query");
    let l = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    let unit = [dir[0] / l, dir[1] / l, dir[2] / l];
    for (lane, (&g, &w)) in got[0].direction.iter().zip(unit.iter()).enumerate() {
        assert!(
            close(g, w),
            "round trip direction[{lane}] gpu={g} expected={w}"
        );
    }
    pin(0, &query, &got[0]);
}

#[test]
fn seam_points_lower_hemisphere() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOctahedralMap::new(&ctx);
    // Several lower-hemisphere directions (forward reflection) plus some
    // independent lower-wedge inverse points (z < 0 branch).
    let queries = [
        OctahedralMapQuery::new([0.3, 0.4, -0.5], 0.6, -0.7),
        OctahedralMapQuery::new([-0.6, 0.2, -0.4], -0.55, 0.65),
        OctahedralMapQuery::new([0.1, -0.8, -0.3], 0.8, 0.5),
        round_trip_query([0.2, -0.3, -0.9]),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOctahedralMap::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing the deterministic fixtures with many round-trip random
    // directions, dispatched together so the per-thread indexing and the
    // contiguous storage layout are both exercised.
    let mut queries = vec![
        OctahedralMapQuery::new([0.0, 0.0, 1.0], 0.0, 0.0),
        OctahedralMapQuery::new([0.0, 0.0, -1.0], 0.0, 0.0),
        OctahedralMapQuery::new([0.0, 0.0, 0.0], 0.3, 0.2),
        round_trip_query([0.37, -0.52, 0.68]),
    ];
    for _ in 0..48 {
        queries.push(round_trip_query(rand_unit_dir(&mut state)));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOctahedralMap::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) of well-conditioned random unit
    // directions, each paired with its own forward square so the inverse map
    // round-trips, pins every output across many invocations.
    let queries: Vec<OctahedralMapQuery> = (0..512)
        .map(|_| round_trip_query(rand_unit_dir(&mut state)))
        .collect();
    check(&ctx, &gpu, &queries);
}
