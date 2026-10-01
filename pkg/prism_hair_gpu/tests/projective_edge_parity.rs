//! Real-device parity for the isolated Projective-Dynamics local edge-projection
//! twin: [`GpuProjectiveEdge`] must reproduce the `CPU` golden
//! [`reference_project_edge`](prism_hair_gpu::projective_edge::reference_project_edge)
//! (which forwards to
//! [`local_project_edge`](prism_render_architecture::hair::projective_global::local_project_edge))
//! for a batch of `(xi, xj, rest)` edge-length constraints, projecting each edge
//! independently onto the sphere of its rest length about the edge midpoint.
//!
//! # Parity criterion
//!
//! The midpoint, the `0.5` halving and the endpoint add/sub are exact, but the
//! projection scales the edge vector by `rest / length` — a divide the `GPU` may
//! round a few `ULP` differently from the scalar reference. Each returned
//! component is therefore compared against a tolerance (`abs_diff < 1e-4` or
//! `rel_diff < 1e-3`), not by raw bit pattern.
//!
//! The suite drives a regular edge, an edge already at rest length, a degenerate
//! coincident edge (`xi == xj`, split along `+x`), a zero rest length (both
//! endpoints collapse to the midpoint), sanitised non-finite coordinates and a
//! sanitised negative rest length, the empty no-op batch, and a large
//! multi-workgroup batch that crosses the 64-wide dispatch boundary. It also
//! asserts the structural invariants the projection must preserve: the midpoint
//! is unchanged and the projected separation equals the (sanitised) rest length.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`,
//! so it needs no optional device feature.
//!
//! Provenance: Projective Dynamics / position-based distance projection
//! (`Bouaziz` 2014) plus `wgpu` compute dispatch; no Unreal Engine source or
//! derived code.

use prism_hair_gpu::projective_edge::{reference_project_edge, GpuProjectiveEdge};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::projective_global::Vec3;

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

/// Dispatches one batch through the device twin.
fn run(ctx: &GpuContext, edges: &[(Vec3, Vec3, f32)]) -> Vec<(Vec3, Vec3)> {
    GpuProjectiveEdge::new(ctx).eval(ctx, edges)
}

/// True when `got` matches `want` within the fma tolerance.
fn close(got: f32, want: f32) -> bool {
    let diff = (got - want).abs();
    diff < 1.0e-4 || diff <= 1.0e-3 * want.abs()
}

/// Asserts a projected endpoint matches the golden within tolerance.
fn assert_vec_close(i: usize, tag: &str, got: Vec3, want: Vec3) {
    assert!(
        close(got.x, want.x) && close(got.y, want.y) && close(got.z, want.z),
        "edge {i} {tag}: device ({}, {}, {}) must match golden ({}, {}, {})",
        got.x,
        got.y,
        got.z,
        want.x,
        want.y,
        want.z
    );
}

/// Asserts a whole batch matches the `CPU` golden within tolerance.
fn assert_batch_close(got: &[(Vec3, Vec3)], edges: &[(Vec3, Vec3, f32)]) {
    assert_eq!(
        got.len(),
        edges.len(),
        "one endpoint pair per edge (got {}, want {})",
        got.len(),
        edges.len()
    );
    for (i, (&(gi, gj), &(xi, xj, rest))) in got.iter().zip(edges.iter()).enumerate() {
        let (wi, wj) = reference_project_edge(xi, xj, rest);
        assert_vec_close(i, "xi'", gi, wi);
        assert_vec_close(i, "xj'", gj, wj);
    }
}

#[test]
fn regular_edge_matches_golden() {
    let Some(ctx) = context_or_skip("regular_edge_matches_golden") else {
        return;
    };
    // An edge of length 10 projected onto rest length 4: the midpoint (3,0,0)
    // stays put, the endpoints land 2 either side along +x.
    let edges = [(Vec3::new(-2.0, 0.0, 0.0), Vec3::new(8.0, 0.0, 0.0), 4.0)];
    let got = run(&ctx, &edges);
    assert_batch_close(&got, &edges);

    let (xi, xj) = got[0];
    // Midpoint preserved.
    let mid = Vec3::new(
        (xi.x + xj.x) * 0.5,
        (xi.y + xj.y) * 0.5,
        (xi.z + xj.z) * 0.5,
    );
    assert!(
        close(mid.x, 3.0) && close(mid.y, 0.0) && close(mid.z, 0.0),
        "midpoint stays"
    );
    // Separation equals the rest length.
    let sep = xi.sub(xj).length();
    assert!(
        close(sep, 4.0),
        "projected separation equals rest length: {sep}"
    );
}

#[test]
fn diagonal_edge_separation_equals_rest() {
    let Some(ctx) = context_or_skip("diagonal_edge_separation_equals_rest") else {
        return;
    };
    let edges = [
        (Vec3::new(0.0, 0.0, 0.0), Vec3::new(3.0, 4.0, 0.0), 2.5),
        (Vec3::new(1.0, -2.0, 3.0), Vec3::new(-4.0, 5.0, -6.0), 7.0),
    ];
    let got = run(&ctx, &edges);
    assert_batch_close(&got, &edges);
    for (&(xi, xj), &(_, _, rest)) in got.iter().zip(edges.iter()) {
        let sep = xi.sub(xj).length();
        assert!(close(sep, rest), "separation {sep} equals rest {rest}");
    }
}

#[test]
fn coincident_edge_splits_along_x() {
    let Some(ctx) = context_or_skip("coincident_edge_splits_along_x") else {
        return;
    };
    // xi == xj: the edge has no direction, so the golden splits it along +x.
    let edges = [(Vec3::new(2.0, -3.0, 4.0), Vec3::new(2.0, -3.0, 4.0), 6.0)];
    let got = run(&ctx, &edges);
    assert_batch_close(&got, &edges);
    let (xi, xj) = got[0];
    let sep = xi.sub(xj).length();
    assert!(
        close(sep, 6.0),
        "degenerate edge still reaches rest length: {sep}"
    );
    // The split is along +x about the shared point (2,-3,4): endpoints at x=5/-1.
    assert!(close(xi.x, 5.0) && close(xj.x, -1.0), "split along +x");
}

#[test]
fn zero_rest_collapses_to_midpoint() {
    let Some(ctx) = context_or_skip("zero_rest_collapses_to_midpoint") else {
        return;
    };
    let edges = [(Vec3::new(-1.0, 2.0, -3.0), Vec3::new(5.0, -4.0, 1.0), 0.0)];
    let got = run(&ctx, &edges);
    assert_batch_close(&got, &edges);
    let (xi, xj) = got[0];
    // Rest length 0: both endpoints land on the midpoint (2,-1,-1).
    assert!(
        close(xi.x, 2.0) && close(xi.y, -1.0) && close(xi.z, -1.0),
        "xi at midpoint"
    );
    assert!(
        close(xj.x, 2.0) && close(xj.y, -1.0) && close(xj.z, -1.0),
        "xj at midpoint"
    );
}

#[test]
fn nonfinite_and_negative_sanitise() {
    let Some(ctx) = context_or_skip("nonfinite_and_negative_sanitise") else {
        return;
    };
    // Non-finite coordinates collapse to 0 and a negative rest length to 0, so
    // the device must track the golden's sanitised result exactly.
    let edges = [
        (
            Vec3::new(f32::NAN, 2.0, f32::INFINITY),
            Vec3::new(1.0, f32::NEG_INFINITY, 3.0),
            2.0,
        ),
        (Vec3::new(0.0, 0.0, 0.0), Vec3::new(4.0, 0.0, 0.0), -5.0),
    ];
    let got = run(&ctx, &edges);
    assert_batch_close(&got, &edges);
    // Edge 1: negative rest sanitises to 0, so both endpoints meet at (2,0,0).
    let (xi, xj) = got[1];
    assert!(
        close(xi.x, 2.0) && close(xj.x, 2.0),
        "negative rest collapses to midpoint"
    );
}

#[test]
fn empty_batch_is_a_no_op() {
    let Some(ctx) = context_or_skip("empty_batch_is_a_no_op") else {
        return;
    };
    let got = run(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields no endpoint pairs");
}

#[test]
fn large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("large_batch_crosses_workgroup_boundary") else {
        return;
    };
    // 200 edges > 3 full 64-wide workgroups: every edge index must map to its own
    // projected pair independent of the dispatch tiling.
    let mut edges = Vec::with_capacity(200);
    for i in 0..200u32 {
        let f = i as f32;
        edges.push((
            Vec3::new(f, -f, f * 0.5),
            Vec3::new(f + 3.0, f - 2.0, -f),
            0.5 + f * 0.05,
        ));
    }
    let got = run(&ctx, &edges);
    assert_batch_close(&got, &edges);
    for (&(xi, xj), &(_, _, rest)) in got.iter().zip(edges.iter()) {
        let sep = xi.sub(xj).length();
        assert!(close(sep, rest), "separation {sep} equals rest {rest}");
    }
}
