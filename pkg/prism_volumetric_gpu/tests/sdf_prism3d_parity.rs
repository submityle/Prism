//! Real-device parity for the analytic prism/link signed-distance twin:
//! [`GpuSdfPrism3d`](prism_volumetric_gpu::sdf_prism3d::GpuSdfPrism3d) must
//! reproduce the `CPU` closed forms of
//! `prism_render_architecture::ray_scene::sdf_primitives` — the regular
//! [`hex_prism`], the regular [`octagon_prism`] and the [`link`] (a torus
//! stretched by a straight mid-section) — across interior, surface and
//! exterior points for every shape and a randomized sweep compared
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
//! *independent* reimplementation of the same three closed forms (the sextant
//! fold for the hexagon, the two-reflection fold for the octagon, and the
//! straight-section offset plus ring reduction for the link). Because the
//! reference and this oracle are both scalar `f32`, a `GPU == oracle` pass is
//! direct evidence the ported kernel computes the same distances the reference
//! does.
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
//! Both prism kernels sign the planar face distance by the ordered comparison
//! `folded_py < apothem` (`folded_py < radius` for the octagon). On that
//! knife-edge the two signs differ by twice the horizontal face offset, so a
//! last-place difference between the `CPU` and `GPU` could pick different sides.
//! The named fixtures and the randomized sweep both stay a safe margin away
//! from `folded_py == apothem` (and `folded_py == radius`) via rejection
//! sampling; the link is fully continuous and needs no such guard.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_primitives`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sdf_prism3d::{GpuSdfPrism3d, SdfPrism3dQuery, SdfPrism3dResult};
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

/// Minimum margin, in folded-`y` units, kept between a query's folded planar
/// coordinate and the prism inradius so the hard face-sign select never
/// straddles the `CPU`/`GPU` boundary.
const SIGN_GUARD: f32 = 0.08;

/// Baked hexagon fold constant `(-cos 30 degrees, sin 30 degrees, 1 / sqrt 3)`,
/// matching the reference exactly.
const HEX_K: [f32; 3] = [-0.866_025_4, 0.5, 0.577_35];

/// Baked octagon fold constant
/// `(-cos(pi/8), sin(pi/8), tan(pi/8) = sqrt 2 - 1)`, matching the reference
/// exactly.
const OCT_K: [f32; 3] = [-0.923_879_5, 0.382_683_4, 0.414_213_56];

/// Returns whether `a` and `b` agree within the given absolute or relative
/// bound (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32, abs_eps: f32, rel_eps: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= abs_eps || rel <= rel_eps
}

/// Euclidean length of a 2-vector, matching the golden `length2` operation
/// order `sqrt(x*x + y*y)` exactly.
fn length2(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

/// Folds the absolute-value point once across the hexagon sextant boundary and
/// returns the resulting planar `y` coordinate (the value the sign select
/// compares against the apothem).
fn hex_folded_py(point: [f32; 3]) -> f32 {
    let mut p = [point[0].abs(), point[1].abs()];
    let fold = 2.0 * (HEX_K[0] * p[0] + HEX_K[1] * p[1]).min(0.0);
    p[0] -= fold * HEX_K[0];
    p[1] -= fold * HEX_K[1];
    p[1]
}

/// Folds the absolute-value point through the two octagon reflections and
/// returns the resulting planar `y` coordinate (the value the sign select
/// compares against the radius).
fn octagon_folded_py(point: [f32; 3]) -> f32 {
    let mut p = [point[0].abs(), point[1].abs()];
    let fold0 = 2.0 * (OCT_K[0] * p[0] + OCT_K[1] * p[1]).min(0.0);
    p[0] -= fold0 * OCT_K[0];
    p[1] -= fold0 * OCT_K[1];
    let fold1 = 2.0 * (-OCT_K[0] * p[0] + OCT_K[1] * p[1]).min(0.0);
    p[0] -= fold1 * -OCT_K[0];
    p[1] -= fold1 * OCT_K[1];
    p[1]
}

/// Independent reimplementation of the reference hexagonal-prism signed
/// distance: fold into one sextant, clamp onto the top flat, interior/exterior
/// split against the folded face and the `z` slab.
fn hex_prism_oracle(point: [f32; 3], apothem: f32, half_depth: f32) -> f32 {
    let mut p = [point[0].abs(), point[1].abs(), point[2].abs()];
    let fold = 2.0 * (HEX_K[0] * p[0] + HEX_K[1] * p[1]).min(0.0);
    p[0] -= fold * HEX_K[0];
    p[1] -= fold * HEX_K[1];
    let clamped_x = p[0].clamp(-HEX_K[2] * apothem, HEX_K[2] * apothem);
    let face = [p[0] - clamped_x, p[1] - apothem];
    let sign = if p[1] - apothem < 0.0 { -1.0 } else { 1.0 };
    let d = [length2(face) * sign, p[2] - half_depth];
    let inside = d[0].max(d[1]).min(0.0);
    let outside = length2([d[0].max(0.0), d[1].max(0.0)]);
    inside + outside
}

/// Independent reimplementation of the reference octagonal-prism signed
/// distance: two reflections fold the quadrant to one wedge, then the same
/// flat-measure/slab combine as the hexagon.
fn octagon_prism_oracle(point: [f32; 3], radius: f32, half_depth: f32) -> f32 {
    let mut p = [point[0].abs(), point[1].abs(), point[2].abs()];
    let fold0 = 2.0 * (OCT_K[0] * p[0] + OCT_K[1] * p[1]).min(0.0);
    p[0] -= fold0 * OCT_K[0];
    p[1] -= fold0 * OCT_K[1];
    let fold1 = 2.0 * (-OCT_K[0] * p[0] + OCT_K[1] * p[1]).min(0.0);
    p[0] -= fold1 * -OCT_K[0];
    p[1] -= fold1 * OCT_K[1];
    let clamped_x = p[0].clamp(-OCT_K[2] * radius, OCT_K[2] * radius);
    let face = [p[0] - clamped_x, p[1] - radius];
    let sign = if p[1] - radius < 0.0 { -1.0 } else { 1.0 };
    let d = [length2(face) * sign, p[2] - half_depth];
    let inside = d[0].max(d[1]).min(0.0);
    let outside = length2([d[0].max(0.0), d[1].max(0.0)]);
    inside + outside
}

/// Independent reimplementation of the reference link signed distance: offset
/// the `y` coordinate by the straight half-length (clamped at zero), reduce to
/// the ring circle of radius `r1`, then subtract the tube radius `r2`.
fn link_oracle(point: [f32; 3], half_length: f32, r1: f32, r2: f32) -> f32 {
    let qy = (point[1].abs() - half_length).max(0.0);
    let planar = length2([point[0], qy]) - r1;
    length2([planar, point[2]]) - r2
}

/// Computes the expected result from the independent host oracle, the faithful
/// reference the `GPU` is pinned against.
fn oracle(q: &SdfPrism3dQuery) -> SdfPrism3dResult {
    SdfPrism3dResult {
        hex_prism_value: hex_prism_oracle(q.point, q.hex_apothem, q.hex_half_depth),
        octagon_prism_value: octagon_prism_oracle(q.point, q.octagon_radius, q.octagon_half_depth),
        link_value: link_oracle(q.point, q.link_half_length, q.link_r1, q.link_r2),
    }
}

/// Returns whether a query stays a safe margin away from both prism sign-select
/// boundaries, so the `CPU` and `GPU` cannot pick different face signs.
fn well_conditioned(q: &SdfPrism3dQuery) -> bool {
    let hex_gap = (hex_folded_py(q.point) - q.hex_apothem).abs();
    let oct_gap = (octagon_folded_py(q.point) - q.octagon_radius).abs();
    hex_gap >= SIGN_GUARD && oct_gap >= SIGN_GUARD
}

/// Pins one `GPU` result against the host oracle under the tolerance.
fn check_one(idx: usize, got: &SdfPrism3dResult, want: &SdfPrism3dResult) {
    assert!(
        close(
            got.hex_prism_value,
            want.hex_prism_value,
            DIST_ABS,
            DIST_REL
        ),
        "query {idx} hex_prism: gpu {} vs oracle {}",
        got.hex_prism_value,
        want.hex_prism_value
    );
    assert!(
        close(
            got.octagon_prism_value,
            want.octagon_prism_value,
            DIST_ABS,
            DIST_REL
        ),
        "query {idx} octagon_prism: gpu {} vs oracle {}",
        got.octagon_prism_value,
        want.octagon_prism_value
    );
    assert!(
        close(got.link_value, want.link_value, DIST_ABS, DIST_REL),
        "query {idx} link: gpu {} vs oracle {}",
        got.link_value,
        want.link_value
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuSdfPrism3d, queries: &[SdfPrism3dQuery]) {
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

/// A fixed battery of named cases spanning interior/surface/exterior points for
/// every shape, both sides of each prism fold, and the link's straight section
/// versus its end arcs. Every case clears the sign-select guard.
fn fixture_queries() -> Vec<SdfPrism3dQuery> {
    vec![
        // Shared origin: deep interior of both prisms, link hole centre.
        SdfPrism3dQuery::new([0.0, 0.0, 0.0], 1.0, 0.5, 1.0, 0.5, 0.3, 0.8, 0.25),
        // Prism interior offset into one wedge (well off the fold line).
        SdfPrism3dQuery::new([0.3, 0.2, 0.1], 1.0, 0.5, 1.0, 0.5, 0.3, 0.8, 0.25),
        // Far exterior along +x for both prisms.
        SdfPrism3dQuery::new([3.0, 0.1, 0.0], 1.0, 0.5, 1.0, 0.5, 0.3, 0.8, 0.25),
        // Outside the z slab (prisms positive by the slab term).
        SdfPrism3dQuery::new([0.2, 0.1, 2.0], 1.0, 0.5, 1.0, 0.5, 0.3, 0.8, 0.25),
        // Negative-x point so the fold brings a large folded coordinate.
        SdfPrism3dQuery::new([-1.6, 0.3, 0.2], 1.0, 0.5, 1.0, 0.5, 0.3, 0.8, 0.25),
        // Larger prisms with a shallow slab, point near a side wall.
        SdfPrism3dQuery::new([1.3, 0.2, 0.3], 1.6, 0.3, 1.8, 0.3, 0.4, 1.0, 0.3),
        // Link ring centre on the +x arc: on the tube axis, distance ~= -r2.
        SdfPrism3dQuery::new([0.8, 0.0, 0.0], 1.0, 0.5, 1.0, 0.5, 0.3, 0.8, 0.25),
        // Link straight-section flank (|y| beyond the half-length), kept a
        // safe margin off both prism sign boundaries.
        SdfPrism3dQuery::new([0.5, 1.3, 0.0], 1.0, 0.5, 1.0, 0.5, 0.3, 0.8, 0.25),
        // Link exterior out of plane (nonzero z pushes outside the tube).
        SdfPrism3dQuery::new([0.8, 0.0, 0.9], 1.0, 0.5, 1.0, 0.5, 0.3, 0.8, 0.25),
        // Link far exterior.
        SdfPrism3dQuery::new([3.0, 3.0, 3.0], 1.0, 0.5, 1.0, 0.5, 0.3, 0.8, 0.25),
        // Thin-tube link with a long straight section.
        SdfPrism3dQuery::new([1.0, 1.2, 0.1], 1.0, 0.5, 1.0, 0.5, 0.9, 1.0, 0.15),
        // Prism point near the top flat but a safe margin below the inradius.
        SdfPrism3dQuery::new([0.1, 0.6, 0.0], 1.0, 0.5, 1.0, 0.5, 0.3, 0.8, 0.25),
    ]
}

/// Builds one well-conditioned random query via rejection sampling: a point in
/// `[-3, 3]^3`, positive bounded shape parameters, retried until it clears the
/// prism sign-select guard.
fn random_query(state: &mut u64) -> SdfPrism3dQuery {
    loop {
        let point = [
            uniform(state, -3.0, 3.0),
            uniform(state, -3.0, 3.0),
            uniform(state, -3.0, 3.0),
        ];
        let candidate = SdfPrism3dQuery::new(
            point,
            uniform(state, 0.6, 1.8),
            uniform(state, 0.2, 0.8),
            uniform(state, 0.6, 1.8),
            uniform(state, 0.2, 0.8),
            uniform(state, 0.1, 0.9),
            uniform(state, 0.4, 1.1),
            uniform(state, 0.15, 0.5),
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
        eprintln!("skipping sdf_prism3d parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSdfPrism3d::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn hex_prism_interior_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfPrism3d::new(&ctx);
    // The centre is deep inside the hexagonal prism, so its distance is
    // negative.
    let q = SdfPrism3dQuery::new([0.0, 0.0, 0.0], 1.0, 0.5, 1.0, 0.5, 0.3, 0.8, 0.25);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].hex_prism_value < 0.0,
        "centre hexagonal-prism distance should be negative: {}",
        got[0].hex_prism_value
    );
}

#[test]
fn octagon_prism_interior_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfPrism3d::new(&ctx);
    // The centre is deep inside the octagonal prism, so its distance is
    // negative.
    let q = SdfPrism3dQuery::new([0.0, 0.0, 0.0], 1.0, 0.5, 1.0, 0.5, 0.3, 0.8, 0.25);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].octagon_prism_value < 0.0,
        "centre octagonal-prism distance should be negative: {}",
        got[0].octagon_prism_value
    );
}

#[test]
fn link_tube_axis_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfPrism3d::new(&ctx);
    // A point on the ring centre of the +x arc sits on the tube axis, so its
    // distance is about minus the tube radius.
    let q = SdfPrism3dQuery::new([0.8, 0.0, 0.0], 1.0, 0.5, 1.0, 0.5, 0.3, 0.8, 0.25);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].link_value < 0.0,
        "tube-axis link distance should be negative: {}",
        got[0].link_value
    );
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfPrism3d::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfPrism3d::new(&ctx);
    let mut state = 0x0f0e_0d0c_0b0a_0908_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random, well-conditioned queries pin all
    // three reported distances across a wide span of points and parameters.
    for _ in 0..512 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
