//! Real-device parity for the §24.1 tensor-product spline-surface shader
//! mirror.
//!
//! The kernel evaluates each of the eight patch queries (bicubic Bézier and
//! uniform cubic B-spline, each `sample`/`tangent_u`/`tangent_v`/`normal`) on a
//! real `GPU` from the single-sourced
//! [`WGSL_SPLINE`](prism_math::shader_mirror::WGSL_SPLINE) +
//! [`WGSL_SURFACE`](prism_math::shader_mirror::WGSL_SURFACE) fragments
//! (`prism_surface`) and diffs against the CPU reference family
//! [`prism_math::curve::surface`].
//!
//! `sample`/`tangent_*` are pure multiply/add polynomials (tight `1e-5`
//! tolerance); `normal` adds a fast-math normalize, verified on non-degenerate
//! patches at a slightly looser `5e-5`. The suite skips gracefully when no
//! adapter is available.

use prism_math::curve::surface::{BSplineSurface, BezierPatch};
use prism_math::Vec3;
use prism_math_gpu::{GpuContext, GpuSurface, Surface};

/// Acquires a device, or prints a skip note and returns `None` on hosts without
/// a usable adapter.
#[expect(
    clippy::print_stderr,
    reason = "test-only skip note when no GPU adapter is present"
)]
fn with_gpu() -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping: no usable GPU adapter on this host");
            None
        }
    }
}

/// Componentwise closeness at `tol` absolute+relative.
fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol + tol * a.abs().max(b.abs())
}

/// Build the row-major 4x4 control grid used by both sides: a tilted plane
/// with the four interior points lifted into a bump, so the surface curves and
/// its normals are well-conditioned (never degenerate).
fn curved_grid() -> [[f32; 3]; 16] {
    let mut g = [[0.0f32; 3]; 16];
    for i in 0..4usize {
        for j in 0..4usize {
            let x = i as f32 / 3.0;
            let y = j as f32 / 3.0;
            let mut z = 2.0 * x - 3.0 * y + 1.0;
            if (i == 1 || i == 2) && (j == 1 || j == 2) {
                z += 1.5;
            }
            g[i * 4 + j] = [x, y, z];
        }
    }
    g
}

/// Rebuild the CPU `[[Vec3; 4]; 4]` grid from the row-major GPU grid.
fn cpu_grid(g: &[[f32; 3]; 16]) -> [[Vec3; 4]; 4] {
    let mut out = [[Vec3::ZERO; 4]; 4];
    for i in 0..4usize {
        for j in 0..4usize {
            let c = g[i * 4 + j];
            out[i][j] = Vec3::new(c[0], c[1], c[2]);
        }
    }
    out
}

/// A deterministic sweep of `(u, v)` across the interior plus the corners.
fn uv_samples() -> Vec<[f32; 2]> {
    let mut v = Vec::new();
    for a in 0..=16u32 {
        for b in 0..=16u32 {
            v.push([a as f32 / 16.0, b as f32 / 16.0]);
        }
    }
    v
}

/// CPU reference for one op at `(u, v)`.
fn cpu_eval(op: Surface, bez: &BezierPatch, bsp: &BSplineSurface, u: f32, v: f32) -> Vec3 {
    match op {
        Surface::BezierSample => bez.sample(u, v),
        Surface::BezierTangentU => bez.tangent_u(u, v),
        Surface::BezierTangentV => bez.tangent_v(u, v),
        Surface::BezierNormal => bez.normal(u, v),
        Surface::BSplineSample => bsp.sample(u, v),
        Surface::BSplineTangentU => bsp.tangent_u(u, v),
        Surface::BSplineTangentV => bsp.tangent_v(u, v),
        Surface::BSplineNormal => bsp.normal(u, v),
    }
}

fn is_normal(op: Surface) -> bool {
    matches!(op, Surface::BezierNormal | Surface::BSplineNormal)
}

#[test]
fn surface_parity_all_ops() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let gpu = GpuSurface::new(&ctx);
    let grid = curved_grid();
    let cg = cpu_grid(&grid);
    let bez = BezierPatch::new(cg);
    let bsp = BSplineSurface::new(cg);
    let uvs = uv_samples();

    for op in [
        Surface::BezierSample,
        Surface::BezierTangentU,
        Surface::BezierTangentV,
        Surface::BezierNormal,
        Surface::BSplineSample,
        Surface::BSplineTangentU,
        Surface::BSplineTangentV,
        Surface::BSplineNormal,
    ] {
        let tol = if is_normal(op) { 5e-5 } else { 1e-5 };
        let got = gpu.eval(&ctx, op, &grid, &uvs);
        assert_eq!(got.len(), uvs.len());
        for (idx, (uv, out)) in uvs.iter().zip(got.iter()).enumerate() {
            let want = cpu_eval(op, &bez, &bsp, uv[0], uv[1]);
            assert!(
                close(out[0], want.x, tol)
                    && close(out[1], want.y, tol)
                    && close(out[2], want.z, tol),
                "op {op:?} sample {idx} uv={uv:?}: gpu={out:?} cpu=({},{},{})",
                want.x,
                want.y,
                want.z,
            );
        }
    }
}

/// A bicubic Bézier patch interpolates its four corner control points.
#[test]
fn surface_parity_bezier_corners() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let gpu = GpuSurface::new(&ctx);
    let grid = curved_grid();
    let corners = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let got = gpu.eval(&ctx, Surface::BezierSample, &grid, &corners);
    // Grid corners: p[0][0]=g[0], p[3][0]=g[12], p[0][3]=g[3], p[3][3]=g[15].
    let expect = [grid[0], grid[12], grid[3], grid[15]];
    for (out, e) in got.iter().zip(expect.iter()) {
        assert!(
            close(out[0], e[0], 1e-5) && close(out[1], e[1], 1e-5) && close(out[2], e[2], 1e-5),
            "corner gpu={out:?} expect={e:?}"
        );
    }
}

/// An empty `(u, v)` batch yields an empty result.
#[test]
fn surface_parity_empty_batch() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let gpu = GpuSurface::new(&ctx);
    let grid = curved_grid();
    let got = gpu.eval(&ctx, Surface::BezierSample, &grid, &[]);
    assert!(got.is_empty());
}
