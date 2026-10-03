//! Real-device parity for the arbitrary-orientation cone/rhombus
//! signed-distance twin:
//! [`GpuSdfConeSegment`](prism_volumetric_gpu::sdf_cone_segment::GpuSdfConeSegment)
//! must reproduce the `CPU` closed forms of
//! `prism_render_architecture::ray_scene::sdf_primitives` — the
//! [`capped_cone_segment`] (a truncated cone/frustum between two arbitrary
//! endpoints), the [`round_cone_segment`] (the convex hull of two spheres, a
//! tapered capsule) and the 3D [`rhombus`] (a rhombic cross-section extruded
//! along `y` with a rounding radius) — across interior, surface-adjacent and
//! exterior points for every shape plus a randomized sweep compared
//! query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! This wave forbids depending on the golden crate, so the host oracle is an
//! *independent* reimplementation of the same three closed forms (the axial
//! projection plus cap-rim/lateral-feature resolve for the capped cone, the
//! axial/radial split with the single slope comparison for the round cone, and
//! the first-octant fold plus clamped-edge projection for the rhombus). Because
//! the reference and this oracle are both scalar `f32`, a `GPU == oracle` pass
//! is direct evidence the ported kernel computes the same distances the
//! reference does.
//!
//! # Parity criterion
//!
//! Each distance threads through `sqrt`, products and quotients, so a `GPU`
//! result may land a few units in the last place from the scalar oracle; each
//! is asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, with a
//! `rel_diff` floor of `1e-6` so a near-zero expected value (a point on the
//! surface) does not inflate the relative error.
//!
//! # Conditioning
//!
//! The capped cone signs its distance by `cbx < 0 && cay < 0`, the round cone
//! selects its governing feature by two `signum`-weighted slope comparisons,
//! and the rhombus signs its planar edge distance by a side test; on each of
//! those loci a last-place difference between the `CPU` and `GPU` could pick a
//! different side. Every named fixture and every randomized query keeps all
//! three reported magnitudes a safe margin off their surfaces (and keeps the
//! endpoints distinct with a positive tangent discriminant) via rejection
//! sampling, so the reported distances stay well clear of each sign-flip locus.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_primitives`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sdf_cone_segment::{
    GpuSdfConeSegment, SdfConeSegmentQuery, SdfConeSegmentResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute bound on each signed distance. A `GPU` `sqrt`/divide may land a few
/// units in the last place from the scalar oracle; `1e-4` admits that legal
/// slack while still failing a wrong port.
const DIST_ABS: f32 = 1.0e-4;

/// Relative bound on each signed distance, applied for larger magnitudes where
/// a few units in the last place exceed the absolute floor.
const DIST_REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value (a
/// surface point) does not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Minimum magnitude kept on each of the three reported distances, so neither
/// the capped-cone sign test, the round-cone feature select, nor the rhombus
/// side test ever straddles the `CPU`/`GPU` boundary.
const SURF_GUARD: f32 = 0.05;

/// Minimum squared axis length kept between a shape's endpoints, so the axial
/// projection divide (`baba`, `l2`) is well away from zero.
const AXIS_GUARD: f32 = 0.4;

/// Minimum tangent discriminant (`l2 - (r1 - r2)^2`) kept for the round cone,
/// so the exact-flank square root argument stays comfortably positive.
const A2_GUARD: f32 = 0.1;

/// Returns whether `a` and `b` agree within the given absolute or relative
/// bound (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32, abs_eps: f32, rel_eps: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= abs_eps || rel <= rel_eps
}

/// Component-wise difference `a - b` of two 3-vectors, matching the golden
/// `sub3`.
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Dot product of two 3-vectors, matching the golden `dot` operation order.
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Euclidean length of a 2-vector, matching the golden `length2` operation
/// order `sqrt(x*x + y*y)` exactly.
fn length2(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

/// Rust `f32::signum` for the non-zero domain, matching the `WGSL` `signum_rs`
/// convention: `+1` for non-negative inputs, `-1` for negative inputs. Fixtures
/// stay off the exact zero so this never straddles the convention boundary.
fn signum_rs(x: f32) -> f32 {
    if x >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Independent reimplementation of the reference capped-cone-segment signed
/// distance: project onto the `a`->`b` axis, take the guarded radial distance,
/// resolve the nearer of the cap-rim and the slanted lateral feature, and flip
/// the interior sign when both residuals are negative.
fn capped_cone_segment_oracle(p: [f32; 3], a: [f32; 3], b: [f32; 3], ra: f32, rb: f32) -> f32 {
    let rba = rb - ra;
    let ba = sub3(b, a);
    let pa = sub3(p, a);
    let baba = dot3(ba, ba);
    let papa = dot3(pa, pa);
    let paba = dot3(pa, ba) / baba;
    let x = (papa - paba * paba * baba).max(0.0).sqrt();
    let cap_r = if paba < 0.5 { ra } else { rb };
    let cax = (x - cap_r).max(0.0);
    let cay = (paba - 0.5).abs() - 0.5;
    let k = rba * rba + baba;
    let f = ((rba * (x - ra) + paba * baba) / k).clamp(0.0, 1.0);
    let cbx = x - ra - f * rba;
    let cby = paba - f;
    let sign = if cbx < 0.0 && cay < 0.0 { -1.0 } else { 1.0 };
    sign * (cax * cax + cay * cay * baba)
        .min(cbx * cbx + cby * cby * baba)
        .sqrt()
}

/// Independent reimplementation of the reference round-cone-segment signed
/// distance: split into axial, beyond-far-cap and squared-radial components
/// scaled by the squared axis length, then one slope comparison selects the
/// near sphere, the far sphere, or the exact tangent flank.
fn round_cone_segment_oracle(p: [f32; 3], a: [f32; 3], b: [f32; 3], r1: f32, r2: f32) -> f32 {
    let ba = sub3(b, a);
    let l2 = dot3(ba, ba);
    let rr = r1 - r2;
    let a2 = l2 - rr * rr;
    let il2 = 1.0 / l2;
    let pa = sub3(p, a);
    let y = dot3(pa, ba);
    let z = y - l2;
    let perp = [
        pa[0] * l2 - ba[0] * y,
        pa[1] * l2 - ba[1] * y,
        pa[2] * l2 - ba[2] * y,
    ];
    let x2 = dot3(perp, perp);
    let y2 = y * y * l2;
    let z2 = z * z * l2;
    let k = signum_rs(rr) * rr * rr * x2;
    if signum_rs(z) * a2 * z2 > k {
        (x2 + z2).sqrt() * il2 - r2
    } else if signum_rs(y) * a2 * y2 < k {
        (x2 + y2).sqrt() * il2 - r1
    } else {
        (x2 * a2 * il2).sqrt() * il2 + y * rr * il2 - r1
    }
}

/// Independent reimplementation of the reference 3D rhombus signed distance:
/// fold into the first octant, project onto the clamped edge for the planar
/// edge distance signed by the side test, and combine with the vertical cap via
/// the rounded-box interior/exterior split.
fn rhombus_oracle(p: [f32; 3], bx: f32, bz: f32, half_height: f32, rounding: f32) -> f32 {
    let px = p[0].abs();
    let py = p[1].abs();
    let pz = p[2].abs();
    let ndot = bx * (bx - 2.0 * px) - bz * (bz - 2.0 * pz);
    let denom = bx * bx + bz * bz;
    let f = (ndot / denom).clamp(-1.0, 1.0);
    let foot_x = 0.5 * bx * (1.0 - f);
    let foot_z = 0.5 * bz * (1.0 + f);
    let edge = length2([px - foot_x, pz - foot_z]);
    let side = signum_rs(px * bz + pz * bx - bx * bz);
    let qx = edge * side - rounding;
    let qy = py - half_height;
    qx.max(qy).min(0.0) + length2([qx.max(0.0), qy.max(0.0)])
}

/// Computes the expected result from the independent host oracle, the faithful
/// reference the `GPU` is pinned against.
fn oracle(q: &SdfConeSegmentQuery) -> SdfConeSegmentResult {
    SdfConeSegmentResult {
        capped_cone_segment_value: capped_cone_segment_oracle(
            q.point,
            q.capped_cone_a,
            q.capped_cone_b,
            q.capped_cone_ra,
            q.capped_cone_rb,
        ),
        round_cone_segment_value: round_cone_segment_oracle(
            q.point,
            q.round_cone_a,
            q.round_cone_b,
            q.round_cone_r1,
            q.round_cone_r2,
        ),
        rhombus_value: rhombus_oracle(
            q.point,
            q.rhombus_half_diag_x,
            q.rhombus_half_diag_z,
            q.rhombus_half_height,
            q.rhombus_rounding,
        ),
    }
}

/// Returns whether a query keeps its endpoints distinct (positive axis length
/// and tangent discriminant) and all three reported distances a safe margin off
/// their surfaces, so no `CPU`/`GPU` sign or feature select can diverge.
fn well_conditioned(q: &SdfConeSegmentQuery) -> bool {
    let cc_ba = sub3(q.capped_cone_b, q.capped_cone_a);
    let cc_baba = dot3(cc_ba, cc_ba);
    let rc_ba = sub3(q.round_cone_b, q.round_cone_a);
    let rc_l2 = dot3(rc_ba, rc_ba);
    let rr = q.round_cone_r1 - q.round_cone_r2;
    let rc_a2 = rc_l2 - rr * rr;
    if cc_baba < AXIS_GUARD || rc_l2 < AXIS_GUARD || rc_a2 < A2_GUARD {
        return false;
    }
    let r = oracle(q);
    r.capped_cone_segment_value.abs() >= SURF_GUARD
        && r.round_cone_segment_value.abs() >= SURF_GUARD
        && r.rhombus_value.abs() >= SURF_GUARD
}

/// Pins one `GPU` result against the host oracle under the tolerance.
fn check_one(idx: usize, got: &SdfConeSegmentResult, want: &SdfConeSegmentResult) {
    assert!(
        close(
            got.capped_cone_segment_value,
            want.capped_cone_segment_value,
            DIST_ABS,
            DIST_REL
        ),
        "query {idx} capped_cone_segment: gpu {} vs oracle {}",
        got.capped_cone_segment_value,
        want.capped_cone_segment_value
    );
    assert!(
        close(
            got.round_cone_segment_value,
            want.round_cone_segment_value,
            DIST_ABS,
            DIST_REL
        ),
        "query {idx} round_cone_segment: gpu {} vs oracle {}",
        got.round_cone_segment_value,
        want.round_cone_segment_value
    );
    assert!(
        close(got.rhombus_value, want.rhombus_value, DIST_ABS, DIST_REL),
        "query {idx} rhombus: gpu {} vs oracle {}",
        got.rhombus_value,
        want.rhombus_value
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuSdfConeSegment, queries: &[SdfConeSegmentQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_one(idx, result, &want);
    }
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws a `f32` in `[lo, hi)` from the generator, using only integer-to-float
/// division (no transcendental).
fn uniform(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let u = lcg(state) as f32 * (1.0 / 4_294_967_296.0);
    lo + (hi - lo) * u
}

/// The baseline capped cone (an upright frustum), round cone (a horizontal
/// tapered capsule) and rhombus shared by most of the named fixtures.
fn base_query(point: [f32; 3]) -> SdfConeSegmentQuery {
    SdfConeSegmentQuery::new(
        point,
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        0.5,
        0.3,
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        0.6,
        0.4,
        1.0,
        0.6,
        0.5,
        0.1,
    )
}

/// A fixed battery of named cases spanning interior/exterior points for every
/// shape plus two degenerate variants (`ra == rb` capped cylinder, `r1 == r2`
/// capsule, zero rounding) and an off-axis diagonal cone. Each case clears the
/// surface guard on all three distances.
fn fixture_queries() -> Vec<SdfConeSegmentQuery> {
    vec![
        // Shared origin: deep interior of all three solids.
        base_query([0.0, 0.0, 0.0]),
        // Interior offset into one octant of every solid, well off each locus.
        base_query([0.2, 0.1, 0.15]),
        // Far exterior for all three solids.
        base_query([3.0, 3.0, 3.0]),
        // Beyond the capped cone's top cap along +y.
        base_query([0.0, 2.0, 0.0]),
        // Beyond the round cone's far sphere along +x.
        base_query([2.0, 0.0, 0.0]),
        // Outside the rhombus slab along +z.
        base_query([0.0, 0.0, 1.5]),
        // Near a lateral flank yet a safe margin outside every surface.
        base_query([0.9, 0.3, 0.4]),
        // Interior of the cone near its narrow cap.
        base_query([0.0, 1.4, 0.0]),
        // Exterior corner where several faces meet.
        base_query([1.5, 1.5, 0.0]),
        // Just outside the rhombus cap along +z, interiors elsewhere.
        base_query([0.0, 0.0, 0.9]),
        // Degenerate: capped cylinder (ra == rb), capsule (r1 == r2), sharp
        // rhombus (zero rounding).
        SdfConeSegmentQuery::new(
            [0.3, 0.2, 0.1],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            0.4,
            0.4,
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            0.5,
            0.5,
            1.0,
            0.6,
            0.5,
            0.0,
        ),
        // Off-axis diagonal cone, vertical capsule, wide rhombus.
        SdfConeSegmentQuery::new(
            [0.2, 0.2, 0.2],
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            0.5,
            0.3,
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            0.4,
            0.4,
            0.8,
            1.2,
            0.3,
            0.15,
        ),
    ]
}

/// Builds one well-conditioned random query via rejection sampling: a point in
/// `[-2.5, 2.5]^3`, endpoints in `[-1, 1]^3` with positive bounded radii,
/// retried until it clears the axis, discriminant and surface guards.
fn random_query(state: &mut u64) -> SdfConeSegmentQuery {
    loop {
        let candidate = SdfConeSegmentQuery::new(
            [
                uniform(state, -2.5, 2.5),
                uniform(state, -2.5, 2.5),
                uniform(state, -2.5, 2.5),
            ],
            [
                uniform(state, -1.0, 1.0),
                uniform(state, -1.0, 1.0),
                uniform(state, -1.0, 1.0),
            ],
            [
                uniform(state, -1.0, 1.0),
                uniform(state, -1.0, 1.0),
                uniform(state, -1.0, 1.0),
            ],
            uniform(state, 0.25, 0.6),
            uniform(state, 0.25, 0.6),
            [
                uniform(state, -1.0, 1.0),
                uniform(state, -1.0, 1.0),
                uniform(state, -1.0, 1.0),
            ],
            [
                uniform(state, -1.0, 1.0),
                uniform(state, -1.0, 1.0),
                uniform(state, -1.0, 1.0),
            ],
            uniform(state, 0.3, 0.6),
            uniform(state, 0.3, 0.6),
            uniform(state, 0.6, 1.4),
            uniform(state, 0.6, 1.4),
            uniform(state, 0.3, 0.8),
            uniform(state, 0.0, 0.2),
        );
        if well_conditioned(&candidate) {
            return candidate;
        }
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping sdf_cone_segment parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSdfConeSegment::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn capped_cone_interior_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfConeSegment::new(&ctx);
    // The centre sits inside the frustum, so its distance is negative.
    let q = base_query([0.0, 0.0, 0.0]);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].capped_cone_segment_value < 0.0,
        "centre capped-cone distance should be negative: {}",
        got[0].capped_cone_segment_value
    );
}

#[test]
fn round_cone_interior_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfConeSegment::new(&ctx);
    // The centre sits inside the tapered capsule, so its distance is negative.
    let q = base_query([0.0, 0.0, 0.0]);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].round_cone_segment_value < 0.0,
        "centre round-cone distance should be negative: {}",
        got[0].round_cone_segment_value
    );
}

#[test]
fn rhombus_interior_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfConeSegment::new(&ctx);
    // The centre sits inside the rhombic solid, so its distance is negative.
    let q = base_query([0.0, 0.0, 0.0]);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].rhombus_value < 0.0,
        "centre rhombus distance should be negative: {}",
        got[0].rhombus_value
    );
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfConeSegment::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfConeSegment::new(&ctx);
    let mut state = 0x0f0e_0d0c_0b0a_0908_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random, well-conditioned queries pin all
    // three reported distances across a wide span of points and parameters.
    for _ in 0..512 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
