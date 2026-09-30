//! Real-device parity for the isolated closest-point-on-triangle twin:
//! [`GpuHairClosestPointTriangle`] must reproduce the `CPU` golden
//! [`closest_point_on_triangle`](prism_render_architecture::hair::binding::closest_point_on_triangle)
//! for a batch of `(p, a, b, c)` queries. The suite drives every one of the
//! seven Voronoi regions in isolation — the three vertex regions (`A`/`B`/`C`),
//! the three edge regions (`AB`/`AC`/`BC`) and the interior face — plus a
//! zero-area degenerate fallback, the empty no-op, and a large multi-workgroup
//! batch that cycles through all regions across the 64-wide dispatch boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The vertex regions return a triangle corner and a canonical barycentric
//! basis (`[1,0,0]` etc.) with no arithmetic, so they are asserted *exactly*.
//! The edge and interior regions divide precomputed dot-product combinations,
//! which a `GPU` may evaluate with a fused multiply-add the scalar reference
//! leaves separate, so the closest point and weights can differ by a few
//! low-mantissa `ULP`; those are asserted with the closest point within `1e-4`
//! and each weight within `abs_diff < 1e-4` or `rel_diff < 1e-3`. Every query
//! sits well clear of the region boundaries so both sides pick the same region.
//! No `sin`/`cos` appears anywhere; all corners are explicit literals.
//!
//! Provenance: standard closest-point-on-triangle (Ericson Voronoi-region test)
//! plus `wgpu` compute dispatch; no Unreal Engine source or derived code.

use prism_hair_gpu::closest_point_triangle::{
    reference_closest_point, ClosestPointQuery, GpuHairClosestPointTriangle,
};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::interpolation::Vec3;

fn v(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}

/// The reference triangle: a unit right-triangle with the right angle at `A`,
/// legs along `+x` (to `B`) and `+y` (to `C`), lying in the `z = 0` plane.
fn tri() -> (Vec3, Vec3, Vec3) {
    (v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), v(0.0, 1.0, 0.0))
}

/// Acquires a headless context, or `None` (with a skip notice) when the host has
/// no `wgpu` adapter so the suite stays green off-device.
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn context_or_skip(label: &str) -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping {label}: no wgpu adapter on this host");
            None
        }
    }
}

/// Asserts one scalar component matches within the documented fma tolerance.
fn assert_close(got: f32, expected: f32, label: &str) {
    let abs_diff = (got - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-6);
    assert!(
        abs_diff < 1e-4 || rel_diff < 1e-3,
        "{label}: gpu {got}, cpu {expected} (abs {abs_diff}, rel {rel_diff})"
    );
}

/// Asserts one scalar component is bit-identical (zero difference), avoiding a
/// float `==` (the `float_cmp` lint) by bounding the absolute difference by
/// `0.0` — vertex-region hits must not drift at all.
fn assert_exact(got: f32, expected: f32, label: &str) {
    assert!(
        (got - expected).abs() <= 0.0,
        "{label}: {got} != {expected}"
    );
}

/// Squared distance between two points, used to bound the closest-point error
/// without an `==` on floats.
fn dist(got: Vec3, expected: Vec3) -> f32 {
    let d = got - expected;
    d.dot(d).sqrt()
}

/// Runs one query on the `GPU` and returns the single result, asserting the
/// batch shape.
fn eval_one(
    ctx: &GpuContext,
    kernel: &GpuHairClosestPointTriangle,
    q: ClosestPointQuery,
    label: &str,
) -> (Vec3, [f32; 3]) {
    let out = kernel.eval(ctx, &[q]);
    assert_eq!(out.len(), 1, "{label}: one result per query");
    (out[0].point, out[0].bary)
}

/// Asserts a `GPU` result matches the golden closest point exactly (used for the
/// arithmetic-free vertex and degenerate regions).
fn assert_exact_vs_golden(got_point: Vec3, got_bary: [f32; 3], q: ClosestPointQuery, label: &str) {
    let (cpu_point, cpu_bary) = reference_closest_point(q.p, q.a, q.b, q.c);
    assert_exact(dist(got_point, cpu_point), 0.0, &format!("{label} point"));
    for i in 0..3 {
        assert_exact(got_bary[i], cpu_bary[i], &format!("{label} bary{i}"));
    }
}

/// Asserts a `GPU` result matches the golden closest point within the fma
/// tolerance (used for the dividing edge and interior regions).
fn assert_close_vs_golden(got_point: Vec3, got_bary: [f32; 3], q: ClosestPointQuery, label: &str) {
    let (cpu_point, cpu_bary) = reference_closest_point(q.p, q.a, q.b, q.c);
    let point_err = dist(got_point, cpu_point);
    assert!(
        point_err < 1e-4,
        "{label} point: distance {point_err} exceeds 1e-4"
    );
    for i in 0..3 {
        assert_close(got_bary[i], cpu_bary[i], &format!("{label} bary{i}"));
    }
}

#[test]
fn gpu_vertex_region_a() {
    let Some(ctx) = context_or_skip("closest_point vertex A") else {
        return;
    };
    let kernel = GpuHairClosestPointTriangle::new(&ctx);
    let (a, b, c) = tri();
    // A point diagonally beyond corner A (negative on both legs) sits squarely
    // in vertex region A; the golden returns A with the canonical [1,0,0].
    let q = ClosestPointQuery {
        p: v(-1.0, -1.0, 0.5),
        a,
        b,
        c,
    };
    let (cpu_point, cpu_bary) = reference_closest_point(q.p, q.a, q.b, q.c);
    assert_exact(cpu_bary[0], 1.0, "cpu picks vertex A");
    assert_exact(dist(cpu_point, a), 0.0, "cpu point is corner A");
    let (point, bary) = eval_one(&ctx, &kernel, q, "vertex A");
    assert_exact_vs_golden(point, bary, q, "vertex A");
}

#[test]
fn gpu_vertex_region_b() {
    let Some(ctx) = context_or_skip("closest_point vertex B") else {
        return;
    };
    let kernel = GpuHairClosestPointTriangle::new(&ctx);
    let (a, b, c) = tri();
    // A point beyond corner B along +x sits in vertex region B.
    let q = ClosestPointQuery {
        p: v(2.0, -1.0, 0.5),
        a,
        b,
        c,
    };
    let (_, cpu_bary) = reference_closest_point(q.p, q.a, q.b, q.c);
    assert_exact(cpu_bary[1], 1.0, "cpu picks vertex B");
    let (point, bary) = eval_one(&ctx, &kernel, q, "vertex B");
    assert_exact_vs_golden(point, bary, q, "vertex B");
}

#[test]
fn gpu_vertex_region_c() {
    let Some(ctx) = context_or_skip("closest_point vertex C") else {
        return;
    };
    let kernel = GpuHairClosestPointTriangle::new(&ctx);
    let (a, b, c) = tri();
    // A point beyond corner C along +y sits in vertex region C.
    let q = ClosestPointQuery {
        p: v(-1.0, 2.0, 0.5),
        a,
        b,
        c,
    };
    let (_, cpu_bary) = reference_closest_point(q.p, q.a, q.b, q.c);
    assert_exact(cpu_bary[2], 1.0, "cpu picks vertex C");
    let (point, bary) = eval_one(&ctx, &kernel, q, "vertex C");
    assert_exact_vs_golden(point, bary, q, "vertex C");
}

#[test]
fn gpu_edge_region_ab() {
    let Some(ctx) = context_or_skip("closest_point edge AB") else {
        return;
    };
    let kernel = GpuHairClosestPointTriangle::new(&ctx);
    let (a, b, c) = tri();
    // Above the midpoint of edge AB, on the far side from C: clamps onto the
    // AB edge at the midpoint (weights [0.5, 0.5, 0]).
    let q = ClosestPointQuery {
        p: v(0.5, -1.0, 0.3),
        a,
        b,
        c,
    };
    let (_, cpu_bary) = reference_closest_point(q.p, q.a, q.b, q.c);
    assert_exact(cpu_bary[2], 0.0, "cpu picks edge AB (no C weight)");
    assert!(
        cpu_bary[0] > 0.0 && cpu_bary[1] > 0.0,
        "interior of edge AB"
    );
    let (point, bary) = eval_one(&ctx, &kernel, q, "edge AB");
    assert_close_vs_golden(point, bary, q, "edge AB");
}

#[test]
fn gpu_edge_region_ac() {
    let Some(ctx) = context_or_skip("closest_point edge AC") else {
        return;
    };
    let kernel = GpuHairClosestPointTriangle::new(&ctx);
    let (a, b, c) = tri();
    // Off the midpoint of edge AC, on the far side from B: clamps onto the AC
    // edge (weights [0.5, 0, 0.5]).
    let q = ClosestPointQuery {
        p: v(-1.0, 0.5, 0.3),
        a,
        b,
        c,
    };
    let (_, cpu_bary) = reference_closest_point(q.p, q.a, q.b, q.c);
    assert_exact(cpu_bary[1], 0.0, "cpu picks edge AC (no B weight)");
    assert!(
        cpu_bary[0] > 0.0 && cpu_bary[2] > 0.0,
        "interior of edge AC"
    );
    let (point, bary) = eval_one(&ctx, &kernel, q, "edge AC");
    assert_close_vs_golden(point, bary, q, "edge AC");
}

#[test]
fn gpu_edge_region_bc() {
    let Some(ctx) = context_or_skip("closest_point edge BC") else {
        return;
    };
    let kernel = GpuHairClosestPointTriangle::new(&ctx);
    let (a, b, c) = tri();
    // Beyond the midpoint of the hypotenuse BC, away from A: clamps onto the BC
    // edge (weights [0, 0.5, 0.5]).
    let q = ClosestPointQuery {
        p: v(1.0, 1.0, 0.3),
        a,
        b,
        c,
    };
    let (_, cpu_bary) = reference_closest_point(q.p, q.a, q.b, q.c);
    assert_exact(cpu_bary[0], 0.0, "cpu picks edge BC (no A weight)");
    assert!(
        cpu_bary[1] > 0.0 && cpu_bary[2] > 0.0,
        "interior of edge BC"
    );
    let (point, bary) = eval_one(&ctx, &kernel, q, "edge BC");
    assert_close_vs_golden(point, bary, q, "edge BC");
}

#[test]
fn gpu_interior_face() {
    let Some(ctx) = context_or_skip("closest_point interior") else {
        return;
    };
    let kernel = GpuHairClosestPointTriangle::new(&ctx);
    let (a, b, c) = tri();
    // A point floated above an interior spot projects into the face; all three
    // weights are strictly positive and sum to one.
    let q = ClosestPointQuery {
        p: v(0.25, 0.25, 0.5),
        a,
        b,
        c,
    };
    let (_, cpu_bary) = reference_closest_point(q.p, q.a, q.b, q.c);
    assert!(
        cpu_bary[0] > 0.0 && cpu_bary[1] > 0.0 && cpu_bary[2] > 0.0,
        "cpu picks the interior face"
    );
    let sum = cpu_bary[0] + cpu_bary[1] + cpu_bary[2];
    assert_close(sum, 1.0, "interior weights sum to one");
    let (point, bary) = eval_one(&ctx, &kernel, q, "interior");
    assert_close_vs_golden(point, bary, q, "interior");
    // The projected point drops the +z offset back onto the z = 0 plane.
    assert!(point.z.abs() < 1e-4, "interior projection lands in-plane");
}

#[test]
fn gpu_degenerate_zero_area() {
    let Some(ctx) = context_or_skip("closest_point degenerate") else {
        return;
    };
    let kernel = GpuHairClosestPointTriangle::new(&ctx);
    // A zero-area triangle (all three corners coincide) collapses onto vertex A
    // with the canonical [1,0,0], exactly on both sides.
    let corner = v(0.7, -0.2, 0.4);
    let q = ClosestPointQuery {
        p: v(1.5, 1.5, 1.5),
        a: corner,
        b: corner,
        c: corner,
    };
    let (cpu_point, cpu_bary) = reference_closest_point(q.p, q.a, q.b, q.c);
    assert_exact(cpu_bary[0], 1.0, "cpu falls back to vertex A");
    assert_exact(dist(cpu_point, corner), 0.0, "cpu point is corner A");
    let (point, bary) = eval_one(&ctx, &kernel, q, "degenerate");
    assert_exact_vs_golden(point, bary, q, "degenerate");
}

#[test]
fn gpu_empty_yields_empty() {
    let Some(ctx) = context_or_skip("closest_point empty") else {
        return;
    };
    let kernel = GpuHairClosestPointTriangle::new(&ctx);
    // An empty query slice returns an empty vector via the CPU early-out (no
    // dispatch — storage buffers cannot be zero-sized).
    let out = kernel.eval(&ctx, &[]);
    assert!(out.is_empty(), "empty queries yield empty results");
}

#[test]
fn gpu_batch_crosses_workgroup() {
    let Some(ctx) = context_or_skip("closest_point batch") else {
        return;
    };
    let kernel = GpuHairClosestPointTriangle::new(&ctx);
    let (a, b, c) = tri();
    // Cycle every region across a batch larger than the 64-wide workgroup so the
    // dispatch spans two groups; every entry must match its golden.
    let probes = [
        v(-1.0, -1.0, 0.5),
        v(2.0, -1.0, 0.5),
        v(-1.0, 2.0, 0.5),
        v(0.5, -1.0, 0.3),
        v(-1.0, 0.5, 0.3),
        v(1.0, 1.0, 0.3),
        v(0.25, 0.25, 0.5),
    ];
    let mut queries = Vec::with_capacity(140);
    for i in 0..140u32 {
        let p = probes[(i as usize) % probes.len()];
        queries.push(ClosestPointQuery { p, a, b, c });
    }
    let gpu = kernel.eval(&ctx, &queries);
    assert_eq!(gpu.len(), queries.len(), "one result per query");
    assert!(queries.len() > 64, "batch spans more than one workgroup");
    for (i, (g, q)) in gpu.iter().zip(queries.iter()).enumerate() {
        assert_close_vs_golden(g.point, g.bary, *q, &format!("batch {i}"));
    }
}
