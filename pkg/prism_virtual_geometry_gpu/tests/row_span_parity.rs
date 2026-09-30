//! Real-device parity for the vis-buffer scanline fast-path twin:
//! [`GpuRowSpan`] must reproduce the CPU golden
//! [`TriangleGradients::row_span`](prism_render_architecture::virtual_geometry::TriangleGradients::row_span)
//! (the closed-form per-row coverage interval) and
//! [`TriangleGradients::depth_from_edges`](prism_render_architecture::virtual_geometry::TriangleGradients::depth_from_edges)
//! (the barycentric depth the walk carries) for every query, and must mark the
//! same uncovered rows as [`None`].
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Gradients are built on integer / dyadic pixel coordinates, so each `w_x` and
//! `w_row` value is exactly representable and a lone division `-w0 / g` is
//! correctly rounded identically on CPU and GPU; `ceil`/`floor` of the quotient
//! therefore agree and the interval bounds are asserted bit-for-bit. The
//! depth-parity scene additionally uses a power-of-two double area, so the
//! depth weights are exact and the depth is asserted bit-for-bit too — no
//! tolerance in either case.
//!
//! Provenance: standard affine edge-inequality span solve for software
//! rasterization; no Unreal Engine source or derived code.

use prism_render_architecture::virtual_geometry::{ScreenVertex, TriangleGradients};
use prism_virtual_geometry_gpu::{GpuContext, GpuRowSpan, ScanQuery, ScanResult};

/// Signed edge function, matching the reference `edge(a, b, p)`:
/// `(b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)`.
fn edge(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

fn sv(x: f32, y: f32, depth: f32) -> ScreenVertex {
    ScreenVertex::new([x, y], depth)
}

/// Edge-function triple at pixel center `p`, in the shipping `w_row` order
/// `(edge(v1, v2, p), edge(v2, v0, p), edge(v0, v1, p))`.
fn edge_triple(v0: ScreenVertex, v1: ScreenVertex, v2: ScreenVertex, p: [f32; 2]) -> [f32; 3] {
    [
        edge(v1.pos, v2.pos, p),
        edge(v2.pos, v0.pos, p),
        edge(v0.pos, v1.pos, p),
    ]
}

/// Asserts the twin's coverage interval equals the CPU golden `row_span`
/// field-for-field for a single query. Used on every scene: the interval bounds
/// are integers derived from correctly-rounded, fma-immune divisions, so they
/// are bit-exact regardless of the double area.
fn assert_span_parity(ctx: &GpuContext, twin: &GpuRowSpan, g: &TriangleGradients, q: ScanQuery) {
    let out = twin.scan(ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1, "one result per query");
    assert_eq!(
        out[0].span,
        g.row_span(q.w_row, q.steps),
        "span mismatch: gpu {:?}",
        out[0].span
    );
}

/// Asserts the twin's [`ScanResult`] equals the CPU golden
/// `row_span`/`depth_from_edges` field-for-field, span *and* depth. Only valid
/// on a power-of-two double area, where the depth weights are exact and the
/// `depth_from_edges` dot product is fma-immune (so no GPU multiply-add
/// contraction can drift the depth by a ULP).
fn assert_query_parity(ctx: &GpuContext, twin: &GpuRowSpan, g: &TriangleGradients, q: ScanQuery) {
    let out = twin.scan(ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1, "one result per query");
    let expected = ScanResult {
        span: g.row_span(q.w_row, q.steps),
        depth: g.depth_from_edges(q.depth_edges),
    };
    assert_eq!(
        out[0], expected,
        "scanline mismatch: gpu {:?}, cpu {expected:?}",
        out[0]
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_row_span_matches_cpu_golden_across_rows() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping row-span parity: no wgpu adapter on this host");
        return;
    };
    let twin = GpuRowSpan::new(&ctx);

    // Generic-position integer vertices: no pixel center lands exactly on an
    // edge, so the closed-form interval matches a per-pixel scan with no float
    // tie-breaking ambiguity. Double area is 317 (not a power of two), so only
    // the span fields are asserted here; depth edges are set to the row start.
    let v0 = sv(1.0, 1.0, 0.2);
    let v1 = sv(20.0, 3.0, 0.6);
    let v2 = sv(4.0, 18.0, 0.9);
    let g = TriangleGradients::new(v0, v1, v2).expect("front-facing");
    assert!(g.double_area > 0.0, "front-facing winding");

    const WIDTH: u32 = 24;
    let steps = WIDTH - 1;
    let mut queries: Vec<ScanQuery> = Vec::new();
    for y in 0..24u32 {
        let cy = f32::from(u16::try_from(y).expect("small range fits u16")) + 0.5;
        let p0 = [0.5, cy];
        let w_row = edge_triple(v0, v1, v2, p0);
        queries.push(ScanQuery {
            w_x: g.w_x,
            vertices_z: g.vertices_z,
            w_row,
            steps,
            depth_edges: w_row,
        });
    }

    // Batch-dispatch every row, then diff each against the CPU golden.
    let out = twin.scan(&ctx, &queries);
    assert_eq!(out.len(), queries.len(), "one result per row");
    for (y, (q, r)) in queries.iter().zip(out.iter()).enumerate() {
        // Double area 317 is not a power of two, so the depth dot product is not
        // fma-immune; the coverage interval is still bit-exact (integer bounds
        // from correctly-rounded divisions). Depth parity is covered by the
        // dedicated power-of-two scene below.
        assert_eq!(
            r.span,
            g.row_span(q.w_row, q.steps),
            "row y={y}: span gpu {:?}",
            r.span
        );
    }

    // Cross-check contiguity against a brute-force per-pixel scan for one row.
    let cy = 8.5;
    let p0 = [0.5, cy];
    let w_row = edge_triple(v0, v1, v2, p0);
    let mut covered: Vec<u32> = Vec::new();
    for k in 0..=steps {
        let p = [
            f32::from(u16::try_from(k).expect("small range fits u16")) + 0.5,
            cy,
        ];
        let w = edge_triple(v0, v1, v2, p);
        if w[0] >= 0.0 && w[1] >= 0.0 && w[2] >= 0.0 {
            covered.push(k);
        }
    }
    let gpu_span = twin
        .scan(
            &ctx,
            &[ScanQuery {
                w_x: g.w_x,
                vertices_z: g.vertices_z,
                w_row,
                steps,
                depth_edges: w_row,
            }],
        )
        .remove(0)
        .span;
    match gpu_span {
        None => assert!(covered.is_empty(), "twin empty but scan covered {covered:?}"),
        Some((k_lo, k_hi)) => {
            let expected: Vec<u32> = (k_lo..=k_hi).collect();
            assert_eq!(covered, expected, "twin span not the contiguous covered run");
        }
    }
}

#[test]
fn gpu_row_span_and_depth_are_bit_exact_on_power_of_two_area() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuRowSpan::new(&ctx);

    // Right triangle with double area edge(v0,v1,v2) = 16 * 16 = 256 = 2^8, so
    // `vertices_z` and every `depth_from_edges` product/sum are exact, and the
    // `-w0 / g` divisions (g in {+/-16}) are exact powers-of-two reciprocals.
    // Both the span and the depth are therefore asserted bit-for-bit.
    let v0 = sv(0.0, 0.0, 0.25);
    let v1 = sv(16.0, 0.0, 0.5);
    let v2 = sv(0.0, 16.0, 1.0);
    let g = TriangleGradients::new(v0, v1, v2).expect("front-facing");
    assert_eq!(g.double_area, 256.0, "double area must be 2^8 for exactness");

    let steps = 20u32;
    for y in 0..18u32 {
        let cy = f32::from(u16::try_from(y).expect("small range fits u16")) + 0.5;
        let p0 = [0.5, cy];
        let w_row = edge_triple(v0, v1, v2, p0);
        // Depth query at a distinct interior pixel to exercise a non-trivial
        // barycentric blend.
        let pd = [3.5, cy];
        let depth_edges = edge_triple(v0, v1, v2, pd);
        assert_query_parity(
            &ctx,
            &twin,
            &g,
            ScanQuery {
                w_x: g.w_x,
                vertices_z: g.vertices_z,
                w_row,
                steps,
                depth_edges,
            },
        );
    }
}

#[test]
fn gpu_row_span_clamps_to_the_step_range() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuRowSpan::new(&ctx);

    // A large triangle so an interior row is fully covered; the span must be
    // clamped to [0, steps] rather than running past the candidate range.
    let v0 = sv(-100.0, -100.0, 0.1);
    let v1 = sv(300.0, -100.0, 0.5);
    let v2 = sv(-100.0, 300.0, 0.9);
    let g = TriangleGradients::new(v0, v1, v2).expect("front-facing");
    let cy = 8.5;
    let p0 = [0.5, cy];
    let w_row = edge_triple(v0, v1, v2, p0);
    let steps = 15u32;
    let q = ScanQuery {
        w_x: g.w_x,
        vertices_z: g.vertices_z,
        w_row,
        steps,
        depth_edges: w_row,
    };
    assert_span_parity(&ctx, &twin, &g, q);

    // Positive control: the fully covered row clamps to exactly [0, steps].
    let span = twin.scan(&ctx, std::slice::from_ref(&q)).remove(0).span;
    assert_eq!(span, Some((0, steps)), "fully covered row clamps to [0, steps]");
}

#[test]
fn gpu_row_span_reports_uncovered_rows_as_none() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuRowSpan::new(&ctx);

    let v0 = sv(1.0, 1.0, 0.2);
    let v1 = sv(20.0, 3.0, 0.6);
    let v2 = sv(4.0, 18.0, 0.9);
    let g = TriangleGradients::new(v0, v1, v2).expect("front-facing");

    // A row far below the triangle (y beyond v2) is not covered anywhere.
    let cy = 40.5;
    let p0 = [0.5, cy];
    let w_row = edge_triple(v0, v1, v2, p0);
    let steps = 23u32;
    let q = ScanQuery {
        w_x: g.w_x,
        vertices_z: g.vertices_z,
        w_row,
        steps,
        depth_edges: w_row,
    };
    assert_span_parity(&ctx, &twin, &g, q);
    let span = twin.scan(&ctx, std::slice::from_ref(&q)).remove(0).span;
    assert_eq!(span, None, "a row outside the triangle must be uncovered");
}

#[test]
fn gpu_row_span_handles_many_queries() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuRowSpan::new(&ctx);

    let v0 = sv(1.0, 1.0, 0.2);
    let v1 = sv(20.0, 3.0, 0.6);
    let v2 = sv(4.0, 18.0, 0.9);
    let g = TriangleGradients::new(v0, v1, v2).expect("front-facing");

    // More than one workgroup (>64) of rows to exercise dispatch tiling and
    // per-thread independence.
    let steps = 23u32;
    let mut queries: Vec<ScanQuery> = Vec::new();
    for k in 0..200u32 {
        let cy = f32::from(i16::try_from(k % 30).expect("small range fits i16")) + 0.5;
        let p0 = [0.5, cy];
        let w_row = edge_triple(v0, v1, v2, p0);
        queries.push(ScanQuery {
            w_x: g.w_x,
            vertices_z: g.vertices_z,
            w_row,
            steps,
            depth_edges: w_row,
        });
    }

    let out = twin.scan(&ctx, &queries);
    assert_eq!(out.len(), queries.len(), "one result per query");
    for (k, (q, r)) in queries.iter().zip(out.iter()).enumerate() {
        assert_eq!(
            r.span,
            g.row_span(q.w_row, q.steps),
            "query {k}: span gpu {:?}",
            r.span
        );
    }
}

#[test]
fn empty_input_yields_empty_output() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let out = GpuRowSpan::new(&ctx).scan(&ctx, &[]);
    assert!(out.is_empty(), "no queries yields no results");
}
