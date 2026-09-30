//! Real-device parity for the world-to-screen vertex-projection twin:
//! [`GpuProjectVertex`] must reproduce the CPU golden
//! [`project_vertex`](prism_render_architecture::virtual_geometry::project_vertex)
//! for every vertex — the column-major clip product, the perspective divide,
//! the y-flipping `ndc_to_uv` and the viewport scale — and must mark the same
//! on/behind-plane vertices invalid ([`None`]).
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The scenes use integer / dyadic world and matrix operands with a power-of-two
//! `clip.w` and power-of-two viewports, so every product, partial sum, the
//! reciprocal, the perspective divide and the affine `ndc_to_uv`/viewport scale
//! are exactly representable and fma-immune. The twin's `pos`/`depth` are
//! therefore asserted bit-for-bit (via `ScreenVertex`'s `PartialEq`) with no
//! tolerance, and the on/behind-plane vertices are asserted to come back as
//! [`None`], exactly as the reference culls them.
//!
//! Provenance: standard world -> clip -> ndc -> viewport vertex projection for
//! software rasterization; no Unreal Engine source or derived code.

use prism_render_architecture::virtual_geometry::{ScreenVertex, project_vertex};
use prism_virtual_geometry_gpu::{GpuContext, GpuProjectVertex};

/// Column-major orthographic-style matrix: `clip.w` is a constant `2` (a power
/// of two), so `inv_w = 0.5` and every projected coordinate stays dyadic-exact.
/// `clip.x = 4x + 8`, `clip.y = 4y + 4`, `clip.z = 2z + 1`, `clip.w = 2`.
const ORTHO: [[f32; 4]; 4] = [
    [4.0, 0.0, 0.0, 0.0],
    [0.0, 4.0, 0.0, 0.0],
    [0.0, 0.0, 2.0, 0.0],
    [8.0, 4.0, 1.0, 2.0],
];

/// Column-major perspective-style matrix whose `clip.w = z`, so a vertex with
/// `z <= 0` lands on/behind the camera plane and must be culled to [`None`],
/// while `z = 2` gives `inv_w = 0.5` and a dyadic-exact projection.
/// `clip.x = 4x + 8`, `clip.y = 4y + 4`, `clip.z = 2z + 1`, `clip.w = z`.
const PERSP: [[f32; 4]; 4] = [
    [4.0, 0.0, 0.0, 0.0],
    [0.0, 4.0, 0.0, 0.0],
    [0.0, 0.0, 2.0, 1.0],
    [8.0, 4.0, 1.0, 0.0],
];

/// Asserts the twin matches the golden field-for-field for every vertex in
/// `positions` (including the [`None`] verdicts for culled vertices).
fn assert_parity(
    ctx: &GpuContext,
    clip_from_world: &[[f32; 4]; 4],
    positions: &[[f32; 3]],
    viewport: [f32; 2],
) {
    let gpu = GpuProjectVertex::new(ctx).project(ctx, clip_from_world, positions, viewport);
    assert_eq!(gpu.len(), positions.len(), "one screen vertex per input");
    for (i, &p) in positions.iter().enumerate() {
        let expected = project_vertex(clip_from_world, p, viewport);
        assert_eq!(
            gpu[i], expected,
            "projection mismatch for vertex {i}: gpu {:?}, cpu {expected:?}",
            gpu[i]
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_project_vertex_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping project-vertex parity: no wgpu adapter on this host");
        return;
    };
    // Dyadic world positions through the constant-w orthographic matrix; every
    // projected pos/depth is exactly representable, so parity is bit-for-bit.
    let vp = [8.0, 8.0];
    let positions = [
        [-2.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [2.0, 3.0, -2.0],
        [-4.0, 2.0, 4.0],
    ];
    assert_parity(&ctx, &ORTHO, &positions, vp);

    // Positive control: all four are in front of the camera (constant w = 2).
    let gpu = GpuProjectVertex::new(&ctx).project(&ctx, &ORTHO, &positions, vp);
    assert!(gpu.iter().all(Option::is_some), "constant w=2 is never culled");
    // Spot-check the first vertex: clip.x=0 -> ndc.x=0 -> u=0.5 -> pos.x=4.0;
    // clip.y=0 -> ndc.y=0 -> v=0.5 -> pos.y=4.0; ndc.z=(2*0+1)*0.5=0.5.
    assert_eq!(gpu[0], Some(ScreenVertex::new([4.0, 4.0], 0.5)));
}

#[test]
fn gpu_project_vertex_culls_on_and_behind_plane() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let vp = [16.0, 16.0];
    // clip.w = z: z <= 0 must cull to None; z = 2 stays valid and dyadic-exact.
    let positions = [
        [1.0, 1.0, 2.0],  // w = 2  -> valid
        [1.0, 1.0, -1.0], // w = -1 -> None (behind plane)
        [0.0, 0.0, 0.0],  // w = 0  -> None (on plane, divide undefined)
        [-1.0, 2.0, 2.0], // w = 2  -> valid
    ];
    assert_parity(&ctx, &PERSP, &positions, vp);

    let gpu = GpuProjectVertex::new(&ctx).project(&ctx, &PERSP, &positions, vp);
    assert!(gpu[0].is_some(), "z=2 vertex must be in front of the camera");
    assert!(gpu[1].is_none(), "z=-1 vertex must be culled (behind plane)");
    assert!(gpu[2].is_none(), "z=0 vertex must be culled (on plane)");
    assert!(gpu[3].is_some(), "second z=2 vertex must be in front");
}

#[test]
fn gpu_project_vertex_handles_many_vertices() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // More than one workgroup (>64) of alternating in-front / behind-plane
    // vertices to exercise the dispatch tiling and per-thread verdict
    // independence, projected through the perspective (clip.w = z) matrix.
    let vp = [32.0, 16.0];
    let mut positions: Vec<[f32; 3]> = Vec::new();
    for k in 0..200i32 {
        // z = 2 for even k (valid), z = -1 for odd k (culled to None).
        let z = if k % 2 == 0 { 2.0 } else { -1.0 };
        let x = f32::from(i16::try_from(k % 8 - 4).expect("small range fits i16"));
        let y = f32::from(i16::try_from(k % 4 - 2).expect("small range fits i16"));
        positions.push([x, y, z]);
    }
    assert_parity(&ctx, &PERSP, &positions, vp);
}

#[test]
fn empty_input_yields_empty_output() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let out = GpuProjectVertex::new(&ctx).project(&ctx, &ORTHO, &[], [8.0, 8.0]);
    assert!(out.is_empty(), "no vertices yields no screen vertices");
}
