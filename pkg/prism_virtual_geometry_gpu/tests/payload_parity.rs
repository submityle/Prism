//! Real-device parity for the 64-bit payload twin: the `GpuPayloadRaster` must
//! reproduce the CPU golden vis-buffer's packed `(depth << 32) | payload` word,
//! resolving both nearest depth and the winning payload through a single 64-bit
//! `atomicMax`.
//!
//! The test skips (with a printed notice) when the host has no `wgpu` adapter
//! or the adapter lacks the 64-bit atomic features, so the suite stays green
//! everywhere while still exercising the full dispatch-and-readback on any
//! device that exposes them (Apple silicon via Metal does).
//!
//! # Parity criterion
//!
//! Every vertex sits on an integer pixel coordinate, so the signed edge
//! function is bit-exact and the covered texel *set* matches exactly (a texel
//! is cleared on the `GPU` iff it is cleared on the `CPU`). The three scene
//! depths are far-separated powers of two, so at every overlap the nearest
//! surface is unambiguous regardless of the sub-`ULP` reassociation the
//! barycentric depth blend is subject to; the **winning payload is therefore
//! bit-exact**. The stored depth field is compared within one unit in the last
//! place (reversed-Z bits are monotonic), the same tolerance the depth-only
//! twin documents.
//!
//! Provenance: standard signed-edge / top-left-rule software rasterization; no
//! Unreal Engine source or derived code.

use prism_render_architecture::virtual_geometry::{
    rasterize_triangle, vis_depth, vis_payload, ScreenVertex, VisBuffer,
};
use prism_virtual_geometry_gpu::{GpuContext, GpuPayloadRaster};

const WIDTH: u32 = 32;
const HEIGHT: u32 = 32;

fn sv(x: f32, y: f32, depth: f32) -> ScreenVertex {
    ScreenVertex::new([x, y], depth)
}

/// Overlapping triangles on integer coordinates with far-separated power-of-two
/// depths, so nearest-depth winner selection is unambiguous and the composited
/// payload is bit-exact. The third triangle is wound clockwise to drive the
/// winding-swap path.
fn scene() -> Vec<[ScreenVertex; 3]> {
    vec![
        [
            sv(2.0, 2.0, 0.125),
            sv(2.0, 26.0, 0.125),
            sv(26.0, 14.0, 0.125),
        ],
        [
            sv(6.0, 6.0, 0.5),
            sv(6.0, 22.0, 0.5),
            sv(24.0, 14.0, 0.5),
        ],
        [
            sv(10.0, 4.0, 0.25),
            sv(28.0, 10.0, 0.25),
            sv(10.0, 20.0, 0.25),
        ],
    ]
}

/// CPU golden packed vis-buffer, payload = triangle index (matching the GPU
/// call's `payloads`).
fn cpu_vis(triangles: &[[ScreenVertex; 3]], cull_back: bool) -> Vec<u64> {
    let mut buffer = VisBuffer::new(WIDTH, HEIGHT);
    for (i, tri) in triangles.iter().enumerate() {
        rasterize_triangle(&mut buffer, *tri, i as u32, cull_back);
    }
    buffer.pixels().to_vec()
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without the 64-bit atomic feature"
)]
fn gpu_payload_raster_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping payload parity: no wgpu adapter on this host");
        return;
    };
    if !ctx.supports_u64_atomics() {
        eprintln!("skipping payload parity: adapter lacks 64-bit atomic features");
        return;
    }
    let Some(raster) = GpuPayloadRaster::new(&ctx) else {
        eprintln!("skipping payload parity: payload pipeline unavailable");
        return;
    };
    let triangles = scene();
    let payloads: Vec<u32> = (0..triangles.len() as u32).collect();

    for cull_back in [false, true] {
        let cpu = cpu_vis(&triangles, cull_back);
        let gpu = raster
            .rasterize(&ctx, WIDTH, HEIGHT, &triangles, &payloads, cull_back)
            .expect("payload raster on a valid framebuffer should succeed");

        assert_eq!(
            gpu.len(),
            cpu.len(),
            "vis-buffer length must match (cull_back = {cull_back})"
        );

        let mut covered = 0usize;
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let idx = (y as usize) * (WIDTH as usize) + (x as usize);
                // Coverage is bit-exact: cleared iff cleared.
                assert_eq!(
                    gpu[idx] == 0,
                    cpu[idx] == 0,
                    "coverage mismatch at ({x}, {y}) with cull_back = {cull_back}: \
                     gpu {}, cpu {}",
                    gpu[idx],
                    cpu[idx]
                );
                if cpu[idx] == 0 {
                    continue;
                }
                covered += 1;
                // Winner selection is unambiguous (depths far-separated), so
                // the composited payload is bit-exact.
                assert_eq!(
                    vis_payload(gpu[idx]),
                    vis_payload(cpu[idx]),
                    "payload mismatch at ({x}, {y}) with cull_back = {cull_back}"
                );
                // Depth field matches within one ULP.
                let diff = vis_depth(gpu[idx]).abs_diff(vis_depth(cpu[idx]));
                assert!(
                    diff <= 1,
                    "depth key at ({x}, {y}) differs by {diff} bits (> 1 ULP) \
                     with cull_back = {cull_back}"
                );
            }
        }
        assert!(
            covered > 0,
            "scene must cover some texels (cull_back = {cull_back})"
        );
    }
}

#[test]
fn payload_count_mismatch_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    if !ctx.supports_u64_atomics() {
        return;
    }
    let Some(raster) = GpuPayloadRaster::new(&ctx) else {
        return;
    };
    let triangles = scene();
    // One payload short of the triangle count.
    let short = vec![0u32; triangles.len() - 1];
    assert!(raster
        .rasterize(&ctx, WIDTH, HEIGHT, &triangles, &short, false)
        .is_err());
}

#[test]
fn empty_framebuffer_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    if !ctx.supports_u64_atomics() {
        return;
    }
    let Some(raster) = GpuPayloadRaster::new(&ctx) else {
        return;
    };
    assert!(raster.rasterize(&ctx, 0, 8, &[], &[], false).is_err());
    assert!(raster.rasterize(&ctx, 8, 0, &[], &[], false).is_err());
}
