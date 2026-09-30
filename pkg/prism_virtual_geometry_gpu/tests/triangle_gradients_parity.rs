//! Real-device parity for the triangle-gradient setup twin:
//! [`GpuTriangleGradients`] must reproduce the CPU golden
//! [`TriangleGradients::new`](prism_render_architecture::virtual_geometry::TriangleGradients::new)
//! for every triangle — the signed double area, the per-column/row edge
//! increments, the normalized depth weights and the depth-plane gradients — and
//! must mark the same degenerate/back-facing triangles invalid ([`None`]).
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The scenes use integer / dyadic pixel coordinates with a power-of-two double
//! area, so the reciprocal, each `depth * inv_area` product and the `z_x`/`z_y`
//! dot sums are all exactly representable and fma-immune. The twin's fields are
//! therefore asserted bit-for-bit (via `TriangleGradients`'s `PartialEq`) with
//! no tolerance, and the invalid triangles are asserted to come back as
//! [`None`], exactly as the reference culls them.
//!
//! Provenance: standard signed-edge affine gradient setup for software
//! rasterization; no Unreal Engine source or derived code.

use prism_render_architecture::virtual_geometry::{ScreenVertex, TriangleGradients};
use prism_virtual_geometry_gpu::{GpuContext, GpuTriangleGradients};

fn sv(x: f32, y: f32, depth: f32) -> ScreenVertex {
    ScreenVertex::new([x, y], depth)
}

/// Asserts the twin matches the golden field-for-field for every triangle in
/// `tris` (including the [`None`] verdicts for culled triangles).
fn assert_parity(ctx: &GpuContext, tris: &[(ScreenVertex, ScreenVertex, ScreenVertex)]) {
    let gpu = GpuTriangleGradients::new(ctx).setup(ctx, tris);
    assert_eq!(gpu.len(), tris.len(), "one gradient entry per triangle");
    for (i, &(v0, v1, v2)) in tris.iter().enumerate() {
        let expected = TriangleGradients::new(v0, v1, v2);
        assert_eq!(
            gpu[i], expected,
            "gradient mismatch for triangle {i}: gpu {:?}, cpu {expected:?}",
            gpu[i]
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_triangle_gradients_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping triangle-gradient parity: no wgpu adapter on this host");
        return;
    };
    // Two front-facing triangles, both with double_area = 256 (= 2^8) so every
    // derived field is dyadic-exact; a back-facing (clockwise) triangle and a
    // collinear degenerate one, both of which the reference culls to `None`.
    let tris = vec![
        // area = 16 * 16 = 256; dyadic depths.
        (
            sv(0.0, 0.0, 0.25),
            sv(16.0, 0.0, 0.5),
            sv(0.0, 16.0, 0.75),
        ),
        // area = 32 * 8 = 256; different shape, dyadic depths.
        (
            sv(0.0, 0.0, 0.5),
            sv(32.0, 0.0, 0.25),
            sv(0.0, 8.0, 0.125),
        ),
        // Clockwise winding in y-down space -> non-positive area -> None.
        (sv(0.0, 0.0, 0.1), sv(0.0, 4.0, 0.2), sv(4.0, 0.0, 0.3)),
        // Collinear vertices -> zero area -> None.
        (sv(0.0, 0.0, 0.0), sv(1.0, 1.0, 0.0), sv(2.0, 2.0, 0.0)),
    ];
    assert_parity(&ctx, &tris);

    // Positive control: the first two are valid, the last two are culled.
    let gpu = GpuTriangleGradients::new(&ctx).setup(&ctx, &tris);
    assert!(gpu[0].is_some(), "front-facing triangle 0 must be valid");
    assert!(gpu[1].is_some(), "front-facing triangle 1 must be valid");
    assert!(gpu[2].is_none(), "back-facing triangle must be culled");
    assert!(gpu[3].is_none(), "degenerate triangle must be culled");
    // The dyadic setup must reproduce the exact double area.
    assert_eq!(gpu[0].unwrap().double_area, 256.0);
    assert_eq!(gpu[1].unwrap().double_area, 256.0);
}

#[test]
fn gpu_triangle_gradients_depth_plane_is_dyadic_exact() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // Single dyadic triangle; the depth-plane gradients z_x/z_y must match the
    // reference bit-for-bit (no fma drift because every product is exact).
    let v0 = sv(0.0, 0.0, 0.25);
    let v1 = sv(16.0, 0.0, 0.5);
    let v2 = sv(0.0, 16.0, 0.75);
    let expected = TriangleGradients::new(v0, v1, v2).expect("front-facing");
    let gpu = GpuTriangleGradients::new(&ctx).setup(&ctx, &[(v0, v1, v2)]);
    let g = gpu[0].expect("front-facing must be valid");
    assert_eq!(g.z_x, expected.z_x, "z_x must match the reference exactly");
    assert_eq!(g.z_y, expected.z_y, "z_y must match the reference exactly");
    assert_eq!(g.w_x, expected.w_x, "per-column edge steps must match");
    assert_eq!(g.w_y, expected.w_y, "per-row edge steps must match");
    assert_eq!(g.vertices_z, expected.vertices_z, "depth weights must match");
}

#[test]
fn gpu_triangle_gradients_handles_many_triangles() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // More than one workgroup (>64) of alternating valid/culled triangles to
    // exercise the dispatch tiling and per-thread verdict independence.
    let mut tris: Vec<(ScreenVertex, ScreenVertex, ScreenVertex)> = Vec::new();
    for k in 0..200u32 {
        if k % 3 == 0 {
            // Front-facing, dyadic area 256.
            tris.push((
                sv(0.0, 0.0, 0.25),
                sv(16.0, 0.0, 0.5),
                sv(0.0, 16.0, 0.75),
            ));
        } else if k % 3 == 1 {
            // Back-facing -> culled.
            tris.push((sv(0.0, 0.0, 0.1), sv(0.0, 4.0, 0.2), sv(4.0, 0.0, 0.3)));
        } else {
            // Degenerate -> culled.
            tris.push((sv(0.0, 0.0, 0.0), sv(1.0, 1.0, 0.0), sv(2.0, 2.0, 0.0)));
        }
    }
    assert_parity(&ctx, &tris);
}

#[test]
fn empty_input_yields_empty_output() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let out = GpuTriangleGradients::new(&ctx).setup(&ctx, &[]);
    assert!(out.is_empty(), "no triangles yields no gradients");
}
