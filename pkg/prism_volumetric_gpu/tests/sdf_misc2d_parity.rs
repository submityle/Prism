//! Real-device parity for the analytic two-dimensional cross/polygon
//! signed-distance twin:
//! [`GpuSdfMisc2d`](prism_volumetric_gpu::sdf_misc2d::GpuSdfMisc2d) must
//! reproduce the `CPU` closed forms of
//! `prism_render_architecture::ray_scene::sdf_primitives` — the Inigo Quilez
//! `rounded_x`, `rounded_cross_2d` and `polygon_2d` outlines — across interior,
//! exterior and surface points, every feature branch of each shape, several
//! convex and concave polygons, and a randomized sweep compared
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
//! *independent* reimplementation of the same closed forms: the rounded-X fold
//! and skeleton projection, the rounded cross's fillet-versus-tip split, and
//! the polygon's per-edge clamped projection with an even-odd crossing sign.
//! Because the reference and this oracle are both scalar `f32`, a
//! `GPU == oracle` pass is direct evidence the ported kernel computes the same
//! distances the reference does.
//!
//! # Parity criterion
//!
//! Every distance threads through products, quotients and a `sqrt`, so a `GPU`
//! result may land a few units in the last place from the scalar oracle; each
//! is asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, with a relative
//! floor of `1e-6` so a near-zero expected value does not inflate the relative
//! error.
//!
//! # Conditioning
//!
//! Where `polygon_2d` flips its running sign the governing locus is a polygon
//! edge where the distance passes through zero, and where `rounded_cross_2d`
//! switches between its fillet and tip branches the two branches meet
//! continuously, so the field stays continuous and no branch disagreement can
//! produce a distance cliff. The randomized sweep still rejects samples near
//! each polygon edge, branch line or fold to keep the comparison far from any
//! such locus, and draws only well-formed shape parameters (`rounded_x_width`,
//! `rounded_x_radius`, `rounded_cross_height` all positive).
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_primitives`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sdf_misc2d::{GpuSdfMisc2d, SdfMisc2dQuery, SdfMisc2dResult};
use prism_volumetric_gpu::GpuContext;

/// Absolute bound on any distance. A `GPU` `sqrt`/divide may land a few units
/// in the last place from the scalar oracle; `1e-4` admits that legal slack
/// while still failing a wrong port.
const ABS_EPS: f32 = 1.0e-4;

/// Relative bound on any distance, applied for larger magnitudes where a few
/// units in the last place exceed the absolute floor.
const REL_EPS: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Default rounded-X arm width and stroke radius for the point-focused
/// fixtures.
const RX_WIDTH: f32 = 1.0;
const RX_RADIUS: f32 = 0.1;

/// Default rounded-cross vertical reach `h` (positive).
const RC_HEIGHT: f32 = 1.0;

/// Default convex square polygon used by the point-focused fixtures.
const SQUARE: [[f32; 2]; 4] = [[-0.6, -0.6], [0.6, -0.6], [0.6, 0.6], [-0.6, 0.6]];

/// A convex triangle fixture.
const TRIANGLE: [[f32; 2]; 3] = [[0.0, 0.6], [-0.5, -0.4], [0.5, -0.4]];

/// A convex hexagon of circumradius `0.6`, vertices baked as literals so the
/// fixture needs no transcendental math.
const HEXAGON: [[f32; 2]; 6] = [
    [0.6, 0.0],
    [0.3, 0.519_615_2],
    [-0.3, 0.519_615_2],
    [-0.6, 0.0],
    [-0.3, -0.519_615_2],
    [0.3, -0.519_615_2],
];

/// A concave "arrowhead"/dart polygon: the lower middle vertex is reentrant.
const CONCAVE: [[f32; 2]; 4] = [[0.0, 0.6], [0.5, -0.5], [0.0, -0.2], [-0.5, -0.5]];

/// Returns whether `a` and `b` agree within the absolute or relative bound
/// (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= ABS_EPS || rel <= REL_EPS
}

/// Euclidean length of a 2-vector, matching the reference `length2`.
fn length2(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

/// Independent reimplementation of `ray_scene::sdf_primitives::rounded_x`: fold
/// into the first quadrant, project onto the `y = x` skeleton clamped to the
/// arm half-length, and inset by the stroke radius.
fn rounded_x_host(point: [f32; 2], w: f32, r: f32) -> f32 {
    let p = [point[0].abs(), point[1].abs()];
    let m = (p[0] + p[1]).min(w) * 0.5;
    length2([p[0] - m, p[1] - m]) - r
}

/// Independent reimplementation of
/// `ray_scene::sdf_primitives::rounded_cross_2d`: fold into the first quadrant,
/// then route to the fillet arc below the tip line or to the nearer convex tip.
fn rounded_cross_2d_host(point: [f32; 2], h: f32) -> f32 {
    let k = 0.5 * (h + 1.0 / h);
    let p = [point[0].abs(), point[1].abs()];
    if p[0] < 1.0 && p[1] < p[0] * (k - h) + h {
        k - length2([p[0] - 1.0, p[1] - k])
    } else {
        length2([p[0], p[1] - h]).min(length2([p[0] - 1.0, p[1]]))
    }
}

/// Independent reimplementation of `ray_scene::sdf_primitives::polygon_2d`:
/// per-edge clamped point-to-segment distance with an even-odd crossing test
/// flipping the running sign.
fn polygon_2d_host(point: [f32; 2], verts: &[[f32; 2]]) -> f32 {
    let n = verts.len();
    if n == 0 {
        return f32::INFINITY;
    }
    let mut d = {
        let w = [point[0] - verts[0][0], point[1] - verts[0][1]];
        w[0] * w[0] + w[1] * w[1]
    };
    let mut s = 1.0_f32;
    for i in 0..n {
        let j = (i + n - 1) % n;
        let e = [verts[j][0] - verts[i][0], verts[j][1] - verts[i][1]];
        let w = [point[0] - verts[i][0], point[1] - verts[i][1]];
        let dot_ee = e[0] * e[0] + e[1] * e[1];
        let t = if dot_ee > 0.0 {
            ((e[0] * w[0] + e[1] * w[1]) / dot_ee).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let b = [w[0] - e[0] * t, w[1] - e[1] * t];
        d = d.min(b[0] * b[0] + b[1] * b[1]);
        let c0 = point[1] >= verts[i][1];
        let c1 = point[1] < verts[j][1];
        let c2 = e[0] * w[1] > e[1] * w[0];
        if (c0 && c1 && c2) || (!c0 && !c1 && !c2) {
            s = -s;
        }
    }
    s * d.sqrt()
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws an `f32` in `[lo, hi]` at ten-thousandth resolution from `state`.
fn draw(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (lcg(state) % 10_001) as f32 / 10_000.0 * (hi - lo)
}

/// Pads a variable-length vertex list into the fixed eight-slot array the twin
/// query carries; unused slots are left at the origin and never read because
/// `polygon_vertex_count` bounds the kernel's loop.
fn pad_verts(verts: &[[f32; 2]]) -> [[f32; 2]; 8] {
    let mut out = [[0.0_f32; 2]; 8];
    for (slot, v) in verts.iter().enumerate() {
        out[slot] = *v;
    }
    out
}

/// Builds one query at `point` with the default per-shape scalars and the given
/// polygon so every shape is exercised on every dispatch.
fn q_poly(point: [f32; 2], verts: &[[f32; 2]]) -> SdfMisc2dQuery {
    SdfMisc2dQuery {
        point,
        rounded_x_width: RX_WIDTH,
        rounded_x_radius: RX_RADIUS,
        rounded_cross_height: RC_HEIGHT,
        polygon_vertex_count: verts.len() as u32,
        polygon_vertices: pad_verts(verts),
    }
}

/// Builds one query at `point` with the default square polygon.
fn q_at(point: [f32; 2]) -> SdfMisc2dQuery {
    q_poly(point, &SQUARE)
}

/// Dispatches `queries` and asserts every distance matches the host oracles.
fn check_batch(ctx: &GpuContext, gpu: &GpuSdfMisc2d, queries: &[SdfMisc2dQuery]) {
    let got: Vec<SdfMisc2dResult> = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(got.iter()) {
        let golden_rx = rounded_x_host(q.point, q.rounded_x_width, q.rounded_x_radius);
        let golden_rc = rounded_cross_2d_host(q.point, q.rounded_cross_height);
        let count = q.polygon_vertex_count as usize;
        let golden_poly = polygon_2d_host(q.point, &q.polygon_vertices[..count]);
        assert!(
            close(r.rounded_x, golden_rx),
            "rounded_x mismatch: gpu={} golden={} (point={:?} width={} radius={})",
            r.rounded_x,
            golden_rx,
            q.point,
            q.rounded_x_width,
            q.rounded_x_radius
        );
        assert!(
            close(r.rounded_cross, golden_rc),
            "rounded_cross mismatch: gpu={} golden={} (point={:?} height={})",
            r.rounded_cross,
            golden_rc,
            q.point,
            q.rounded_cross_height
        );
        assert!(
            close(r.polygon, golden_poly),
            "polygon mismatch: gpu={} golden={} (point={:?} count={})",
            r.polygon,
            golden_poly,
            q.point,
            count
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_batch_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping sdf_misc2d parity: no wgpu adapter available");
        return;
    };
    let gpu = GpuSdfMisc2d::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "empty batch yields no results");
}

#[test]
fn rounded_x_interior_and_exterior() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfMisc2d::new(&ctx);
    // On the diagonal arms (negative inside the stroke), off the arms, and far
    // away (positive), exercising both the clamped-skeleton and tip regions.
    let queries = [
        q_at([0.0, 0.0]),
        q_at([0.2, 0.2]),
        q_at([0.6, -0.6]),
        q_at([1.5, 0.0]),
        q_at([-1.4, 1.3]),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn rounded_cross_fillet_and_tip_branches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfMisc2d::new(&ctx);
    // Inside the fillet wedge (negative), near each convex tip, and exterior,
    // exercising both the arc branch and the min-of-tips branch.
    let queries = [
        q_at([0.0, 0.0]),
        q_at([0.3, 0.2]),
        q_at([1.1, 0.0]),
        q_at([0.0, 1.1]),
        q_at([1.3, 1.3]),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn polygon_triangle_points() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfMisc2d::new(&ctx);
    // Interior, near an edge, near a vertex, and exterior of a triangle.
    let queries = [
        q_poly([0.0, 0.0], &TRIANGLE),
        q_poly([0.0, 0.4], &TRIANGLE),
        q_poly([0.3, -0.3], &TRIANGLE),
        q_poly([1.0, 1.0], &TRIANGLE),
        q_poly([-0.8, -0.6], &TRIANGLE),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn polygon_square_and_hexagon_points() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfMisc2d::new(&ctx);
    // Square interior/exterior plus hexagon interior, edge-adjacent, and far.
    let queries = [
        q_poly([0.0, 0.0], &SQUARE),
        q_poly([0.3, -0.2], &SQUARE),
        q_poly([1.2, 0.0], &SQUARE),
        q_poly([0.0, 0.0], &HEXAGON),
        q_poly([0.4, 0.1], &HEXAGON),
        q_poly([-1.0, 0.9], &HEXAGON),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn polygon_concave_winding() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfMisc2d::new(&ctx);
    // Points chosen so the reentrant dart's even-odd sign test is exercised:
    // inside a lobe, inside the notch gap (outside the solid), and far away.
    let queries = [
        q_poly([0.0, 0.3], &CONCAVE),
        q_poly([0.3, -0.4], &CONCAVE),
        q_poly([0.0, -0.35], &CONCAVE),
        q_poly([-0.3, -0.4], &CONCAVE),
        q_poly([1.1, -1.1], &CONCAVE),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn mixed_single_dispatch_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfMisc2d::new(&ctx);
    // A heterogeneous batch with varied per-shape parameters and polygons in
    // one dispatch.
    let queries = [
        SdfMisc2dQuery {
            point: [0.3, 0.5],
            rounded_x_width: 1.2,
            rounded_x_radius: 0.15,
            rounded_cross_height: 1.3,
            polygon_vertex_count: 3,
            polygon_vertices: pad_verts(&TRIANGLE),
        },
        SdfMisc2dQuery {
            point: [-0.8, 0.9],
            rounded_x_width: 0.8,
            rounded_x_radius: 0.05,
            rounded_cross_height: 0.7,
            polygon_vertex_count: 6,
            polygon_vertices: pad_verts(&HEXAGON),
        },
        SdfMisc2dQuery {
            point: [1.4, -1.1],
            rounded_x_width: 1.5,
            rounded_x_radius: 0.25,
            rounded_cross_height: 1.1,
            polygon_vertex_count: 4,
            polygon_vertices: pad_verts(&CONCAVE),
        },
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn randomized_sweep() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfMisc2d::new(&ctx);
    let mut state: u64 = 0x2f6b_8d41_90ac_57e3;
    // Margin kept away from each shape's edge/branch/fold locus. Those loci are
    // continuous (the branches agree there), so this is extra safety only.
    let margin = 0.03_f32;
    let mut queries = Vec::with_capacity(384);
    while queries.len() < 384 {
        let point = [draw(&mut state, -1.8, 1.8), draw(&mut state, -1.8, 1.8)];

        // Rounded X: positive arm width and stroke radius.
        let rx_width = draw(&mut state, 0.5, 1.5);
        let rx_radius = draw(&mut state, 0.05, 0.4);
        // Rounded cross: positive vertical reach.
        let rc_height = draw(&mut state, 0.5, 1.5);
        // Polygon: a random-size square so the edge-projection and sign test
        // are exercised across scales.
        let ph = draw(&mut state, 0.3, 0.9);
        let square = [[-ph, -ph], [ph, -ph], [ph, ph], [-ph, ph]];

        // Reject near the polygon boundary where the running sign flips (the
        // field is continuous there, so this is safety margin only).
        if polygon_2d_host(point, &square).abs() < margin {
            continue;
        }
        // Reject near the rounded-cross branch line and tip columns.
        let p = [point[0].abs(), point[1].abs()];
        let k = 0.5 * (rc_height + 1.0 / rc_height);
        if (p[1] - (p[0] * (k - rc_height) + rc_height)).abs() < margin
            || (p[0] - 1.0).abs() < margin
        {
            continue;
        }

        queries.push(SdfMisc2dQuery {
            point,
            rounded_x_width: rx_width,
            rounded_x_radius: rx_radius,
            rounded_cross_height: rc_height,
            polygon_vertex_count: 4,
            polygon_vertices: pad_verts(&square),
        });
    }
    check_batch(&ctx, &gpu, &queries);
}
