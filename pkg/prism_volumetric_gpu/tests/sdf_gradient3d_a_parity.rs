//! Real-device parity for the analytic signed-distance *gradient* twin:
//! `GpuSdfGradient3dA` must reproduce the `CPU` closed forms of the analytic
//! oracle `prism_render_architecture::ray_scene::sdf_primitives` — the
//! `sphere_gradient`, the `box_gradient`, the `torus_gradient` and the constant
//! `plane_gradient` — across interior, exterior, on-axis and off-axis cases
//! plus a randomized sweep compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! This wave forbids depending on the golden crate, so the host oracle is an
//! *independent* reimplementation of the same four closed forms in scalar
//! `f32`, operation-for-operation. Because the reference and this oracle are
//! both scalar `f32`, a `GPU == oracle` pass is direct evidence the ported
//! kernel computes the same analytic gradient the reference does.
//!
//! # Parity criterion
//!
//! Each gradient threads through `sqrt`, products and quotients, so a `GPU`
//! result may land a few units in the last place from the scalar oracle; each
//! component is asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, with a
//! `rel_diff` floor of `1e-6` so a near-zero expected component does not inflate
//! the relative error.
//!
//! # Conditioning
//!
//! Fixtures and the randomized sweep stay clear of the measure-zero creases
//! where a gradient is genuinely undefined or where a last-place wobble could
//! flip a branch: the sphere centre (`|p|` near zero), every axis-aligned sign
//! flip (any coordinate near zero, where `signum` is three-valued), the box
//! surface (`max(q_i)` near zero, the exterior/interior split), the interior
//! box face ties (the two largest `q_i` near equal), the torus central axis
//! (`rho` near zero) and the torus ring (`l` near zero). Everywhere else every
//! intermediate is a well-conditioned non-negative square root or a guarded
//! quotient, so a `GPU` evaluation lands a few units in the last place from the
//! scalar oracle and never straddles a branch cliff.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_primitives`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sdf_gradient3d_a::{
    GpuSdfGradient3dA, SdfGradient3dAQuery, SdfGradient3dAResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute bound on each gradient component. A `GPU` `sqrt`/divide may land a
/// few units in the last place from the scalar oracle; `1e-4` admits that legal
/// slack while still failing a wrong port.
const SD_ABS: f32 = 1.0e-4;

/// Relative bound on each gradient component, applied for larger magnitudes
/// where a few units in the last place exceed the absolute floor.
const SD_REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected component
/// does not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Floor on `|point|` so the sphere gradient's radial divide stays well
/// conditioned and clear of the degenerate centre.
const SPHERE_MIN: f32 = 0.25;

/// Margin keeping every coordinate clear of zero, where `signum` is three
/// valued and the `CPU`/`GPU` sign selections could differ.
const SIGN_MARGIN: f32 = 5.0e-2;

/// Margin keeping the point clear of the box surface, where the gradient's
/// exterior/interior branch split lives.
const BOX_SURFACE_MARGIN: f32 = 5.0e-2;

/// Margin keeping the two largest interior overshoot terms apart, so the
/// nearest-face selection does not sit on a tie.
const BOX_FACE_MARGIN: f32 = 5.0e-2;

/// Floor on `rho = |p.xz|` so the torus gradient's radial divide stays clear of
/// the central symmetry axis.
const TORUS_RHO_MIN: f32 = 0.3;

/// Floor on `l = |(rho - major, p.y)|` so the torus gradient's divide stays
/// clear of the ring circle itself.
const TORUS_L_MIN: f32 = 0.2;

/// Returns whether `a` and `b` agree within the given absolute or relative
/// bound (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32, abs_eps: f32, rel_eps: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= abs_eps || rel <= rel_eps
}

/// Independent reimplementation of the reference `sphere_gradient`: the outward
/// radial unit normal `point / |point|`, zero at the degenerate centre.
fn sphere_gradient(point: [f32; 3]) -> [f32; 3] {
    let l = (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt();
    if l == 0.0 {
        return [0.0, 0.0, 0.0];
    }
    [point[0] / l, point[1] / l, point[2] / l]
}

/// Independent reimplementation of the reference `box_gradient`: the normalised
/// signed overshoot outside, the nearest signed face axis inside.
fn box_gradient(point: [f32; 3], half_extent: [f32; 3]) -> [f32; 3] {
    let q = [
        point[0].abs() - half_extent[0],
        point[1].abs() - half_extent[1],
        point[2].abs() - half_extent[2],
    ];
    let m = [q[0].max(0.0), q[1].max(0.0), q[2].max(0.0)];
    let len = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt();
    if len > 0.0 {
        return [
            point[0].signum() * m[0] / len,
            point[1].signum() * m[1] / len,
            point[2].signum() * m[2] / len,
        ];
    }
    if q[0] >= q[1] && q[0] >= q[2] {
        [point[0].signum(), 0.0, 0.0]
    } else if q[1] >= q[2] {
        [0.0, point[1].signum(), 0.0]
    } else {
        [0.0, 0.0, point[2].signum()]
    }
}

/// Independent reimplementation of the reference `torus_gradient`: the radial
/// planar component paired with the tube cross-section, `+y` on the fallbacks.
fn torus_gradient(point: [f32; 3], major_radius: f32) -> [f32; 3] {
    let rho = (point[0] * point[0] + point[2] * point[2]).sqrt();
    let qx = rho - major_radius;
    let qy = point[1];
    let l = (qx * qx + qy * qy).sqrt();
    if l == 0.0 {
        return [0.0, 1.0, 0.0];
    }
    if rho == 0.0 {
        return [0.0, qy.signum(), 0.0];
    }
    let radial = (qx / l) / rho;
    [radial * point[0], qy / l, radial * point[2]]
}

/// Independent reimplementation of the reference `plane_gradient`: the constant
/// plane normal returned verbatim.
fn plane_gradient(normal: [f32; 3]) -> [f32; 3] {
    normal
}

/// Computes the expected result from the independent host oracle, the faithful
/// reference the `GPU` is pinned against.
fn oracle(q: &SdfGradient3dAQuery) -> SdfGradient3dAResult {
    SdfGradient3dAResult {
        sphere_gradient: sphere_gradient(q.point),
        box_gradient: box_gradient(q.point, q.box_half_extent),
        torus_gradient: torus_gradient(q.point, q.torus_major_radius),
        plane_gradient: plane_gradient(q.plane_normal),
    }
}

/// Pins one `GPU` gradient vector against the oracle, component by component.
fn check_vec3(idx: usize, name: &str, got: [f32; 3], want: [f32; 3]) {
    for (axis, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            close(*g, *w, SD_ABS, SD_REL),
            "query {idx} {name}[{axis}]: gpu {g} vs cpu {w}"
        );
    }
}

/// Pins one `GPU` result against the host oracle: all four gradient vectors
/// under the shared tolerance.
fn check_one(idx: usize, got: &SdfGradient3dAResult, want: &SdfGradient3dAResult) {
    check_vec3(
        idx,
        "sphere_gradient",
        got.sphere_gradient,
        want.sphere_gradient,
    );
    check_vec3(idx, "box_gradient", got.box_gradient, want.box_gradient);
    check_vec3(
        idx,
        "torus_gradient",
        got.torus_gradient,
        want.torus_gradient,
    );
    check_vec3(
        idx,
        "plane_gradient",
        got.plane_gradient,
        want.plane_gradient,
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuSdfGradient3dA, queries: &[SdfGradient3dAQuery]) {
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

/// Length of a 3-vector via a host-side `sqrt` (not transcendental).
fn length(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Returns a unit copy of `v`; the fixtures feed only non-zero normals.
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = length(v);
    [v[0] / l, v[1] / l, v[2] / l]
}

/// Builds a query at `point` with a fixed, well-conditioned set of shape
/// parameters shared by the named fixtures: a unit plane normal, a box half
/// extent well clear of the surface margin, and a torus of major radius `1.5`.
fn query_at(point: [f32; 3]) -> SdfGradient3dAQuery {
    SdfGradient3dAQuery::new(
        point,
        normalize([0.3, 0.9, 0.2]), // plane_normal: a fixed unit vector
        [0.8, 1.1, 0.6],            // box_half_extent
        1.5,                        // torus_major_radius
        0.4,                        // torus_minor_radius (unused by gradients)
    )
}

/// A fixed battery of named points spanning interior, exterior, on-axis and
/// off-axis cases, dispatched together. Every coordinate is kept clear of zero
/// and every point clear of the box surface and the torus ring/axis.
fn fixture_queries() -> Vec<SdfGradient3dAQuery> {
    vec![
        // Clearly outside everything along +x.
        query_at([3.0, 0.4, 0.3]),
        // Clearly outside along -z.
        query_at([0.4, 0.3, -3.0]),
        // Diagonal exterior corner.
        query_at([2.5, 2.0, 1.8]),
        query_at([-2.2, -1.7, -2.4]),
        // Deep inside the box (interior nearest-face branch).
        query_at([0.2, 0.3, 0.15]),
        query_at([-0.15, 0.25, -0.2]),
        // Off the torus ring but inside the tube band.
        query_at([1.6, 0.2, 0.3]),
        query_at([-1.4, 0.3, 0.5]),
        // Up the +y side (sphere/torus both well conditioned).
        query_at([0.5, 2.5, 0.4]),
        // Far away in every direction.
        query_at([3.5, 3.2, 3.1]),
        query_at([-3.3, -2.6, 1.4]),
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping sdf_gradient3d_a parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSdfGradient3dA::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn sphere_gradient_is_the_unit_radial_normal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfGradient3dA::new(&ctx);
    let q = query_at([1.3, -0.7, 2.1]);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    // Off the centre the sphere gradient is a unit vector.
    let g = got[0].sphere_gradient;
    assert!(
        close(length(g), 1.0, SD_ABS, SD_REL),
        "sphere gradient should be unit length: {g:?}"
    );
}

#[test]
fn plane_gradient_equals_the_constant_normal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfGradient3dA::new(&ctx);
    let q = query_at([0.9, -1.3, 0.6]);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    // The plane gradient is the supplied normal, independent of the point.
    check_vec3(0, "plane_gradient", got[0].plane_gradient, q.plane_normal);
}

#[test]
fn box_gradient_points_outward_outside_and_picks_a_face_inside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfGradient3dA::new(&ctx);
    // Clearly outside along +x: the gradient should be ~+x and unit length.
    let outside = query_at([3.0, 0.2, 0.1]);
    // Deep inside: the nearest face is picked, a signed unit axis.
    let inside = query_at([0.2, 0.1, 0.15]);
    let got = gpu.evaluate(&ctx, &[outside, inside]);
    assert_eq!(got.len(), 2);
    check_one(0, &got[0], &oracle(&outside));
    check_one(1, &got[1], &oracle(&inside));
    let out_g = got[0].box_gradient;
    assert!(
        out_g[0] > 0.5 && close(length(out_g), 1.0, SD_ABS, SD_REL),
        "exterior box gradient should point outward (+x) and be unit: {out_g:?}"
    );
    let in_g = got[1].box_gradient;
    assert!(
        close(length(in_g), 1.0, SD_ABS, SD_REL),
        "interior box gradient should be a unit face axis: {in_g:?}"
    );
}

#[test]
fn torus_gradient_is_unit_off_the_ring() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfGradient3dA::new(&ctx);
    // A point off the ring circle and off the central axis.
    let q = query_at([2.4, 0.6, 0.5]);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    let g = got[0].torus_gradient;
    assert!(
        close(length(g), 1.0, SD_ABS, SD_REL),
        "torus gradient should be unit length off the ring: {g:?}"
    );
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfGradient3dA::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

/// Builds one well-conditioned random query: the point drawn in `[-4, 4]^3`,
/// positive extents and a non-zero plane normal. Samples near the sphere
/// centre, any axis sign flip, the box surface, an interior face tie, the torus
/// central axis or the torus ring are rejected (see `# Conditioning`).
fn random_query(state: &mut u64) -> SdfGradient3dAQuery {
    loop {
        let point = [
            uniform(state, -4.0, 4.0),
            uniform(state, -4.0, 4.0),
            uniform(state, -4.0, 4.0),
        ];
        let normal = [
            uniform(state, -1.0, 1.0),
            uniform(state, -1.0, 1.0),
            uniform(state, -1.0, 1.0),
        ];
        let half_extent = [
            uniform(state, 0.3, 1.6),
            uniform(state, 0.3, 1.6),
            uniform(state, 0.3, 1.6),
        ];
        let major = uniform(state, 0.8, 2.4);
        let minor = uniform(state, 0.1, 0.6);

        // Reject near the sphere centre (radial divide).
        if length(point) < SPHERE_MIN {
            continue;
        }

        // Reject any coordinate near zero, where signum is three valued.
        if point[0].abs() < SIGN_MARGIN
            || point[1].abs() < SIGN_MARGIN
            || point[2].abs() < SIGN_MARGIN
        {
            continue;
        }

        // Reject a near-degenerate plane normal (the normalize divide).
        if length(normal) < 0.3 {
            continue;
        }

        // Reject near the box surface (exterior/interior split) and near an
        // interior nearest-face tie.
        let q = [
            point[0].abs() - half_extent[0],
            point[1].abs() - half_extent[1],
            point[2].abs() - half_extent[2],
        ];
        let max_q = q[0].max(q[1]).max(q[2]);
        if max_q.abs() < BOX_SURFACE_MARGIN {
            continue;
        }
        // Second-largest overshoot: reject when it ties the largest.
        let mid_q = q[0].min(q[1]).max(q[0].max(q[1]).min(q[2]));
        if (max_q - mid_q).abs() < BOX_FACE_MARGIN {
            continue;
        }

        // Reject near the torus central axis and the ring circle.
        let rho = (point[0] * point[0] + point[2] * point[2]).sqrt();
        if rho < TORUS_RHO_MIN {
            continue;
        }
        let qx = rho - major;
        let qy = point[1];
        let l = (qx * qx + qy * qy).sqrt();
        if l < TORUS_L_MIN {
            continue;
        }

        return SdfGradient3dAQuery::new(point, normalize(normal), half_extent, major, minor);
    }
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfGradient3dA::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random, well-conditioned queries pin every
    // reported gradient across a wide span of points and shape extents.
    for _ in 0..512 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
