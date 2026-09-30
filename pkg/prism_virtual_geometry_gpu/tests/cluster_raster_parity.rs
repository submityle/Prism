//! Real-device parity for the cluster-granularity 64-bit vis-buffer twin: the
//! [`GpuClusterRaster`] must reproduce the CPU golden
//! [`rasterize_cluster`] packed `(depth << 32) | payload` word for an *indexed*
//! cluster, resolving nearest depth and the winning
//! `(cluster_id, triangle_id)` payload through a single 64-bit `atomicMax`.
//!
//! The test skips (with a printed notice) when the host has no `wgpu` adapter
//! or the adapter lacks the 64-bit atomic features, so the suite stays green
//! everywhere while still exercising the full dispatch-and-readback on any
//! device that exposes them (Apple silicon via Metal does).
//!
//! # Parity criterion
//!
//! Every vertex sits on an integer pixel coordinate, so the signed edge
//! function is bit-exact and the covered texel *set* matches exactly. The scene
//! depths are far-separated powers of two, so at every overlap the nearest
//! surface is unambiguous regardless of the sub-`ULP` reassociation the
//! barycentric depth blend is subject to; the **winning payload is therefore
//! bit-exact**, including the high `cluster_id` bits synthesized on the device.
//! The stored depth field is compared within one unit in the last place, the
//! same tolerance the other vis-buffer twins document.
//!
//! The two failure modes unique to the cluster path are exercised directly: a
//! triangle whose index list points past the vertex array is dropped, and
//! triangles beyond
//! [`MAX_CLUSTER_TRIANGLES`](prism_render_architecture::virtual_geometry::MAX_CLUSTER_TRIANGLES)
//! are dropped so triangle ids never alias the 7-bit payload field.
//!
//! Provenance: standard signed-edge / top-left-rule software rasterization; no
//! Unreal Engine source or derived code.

use prism_render_architecture::virtual_geometry::{
    rasterize_cluster, vis_depth, vis_payload, ScreenVertex, VisBuffer, MAX_CLUSTER_TRIANGLES,
};
use prism_virtual_geometry_gpu::{GpuClusterRaster, GpuContext};

const WIDTH: u32 = 32;
const HEIGHT: u32 = 32;
/// Non-zero cluster id so the high payload bits are actually exercised.
const CLUSTER_ID: u32 = 0x0012_3456;

fn sv(x: f32, y: f32, depth: f32) -> ScreenVertex {
    ScreenVertex::new([x, y], depth)
}

/// A shared vertex array on integer pixel coordinates with far-separated
/// power-of-two depths. The last two vertices are only referenced by an
/// out-of-range index below, so their exact values do not matter.
fn shared_vertices() -> Vec<ScreenVertex> {
    vec![
        sv(2.0, 2.0, 0.125),   // 0
        sv(2.0, 26.0, 0.125),  // 1
        sv(26.0, 14.0, 0.125), // 2
        sv(6.0, 6.0, 0.5),     // 3
        sv(6.0, 22.0, 0.5),    // 4
        sv(24.0, 14.0, 0.5),   // 5
        sv(10.0, 4.0, 0.25),   // 6
        sv(28.0, 10.0, 0.25),  // 7
        sv(10.0, 20.0, 0.25),  // 8
    ]
}

/// Indexed triangles into [`shared_vertices`]. Triangles 0 and 1 are wound with
/// negative area to drive the winding-swap path (under `cull_back = false`),
/// while triangle 2 is front-facing so at least one triangle survives back-face
/// culling. Triangle 3 references a vertex index past the array end and must be
/// dropped by both the CPU golden and the GPU twin.
fn triangles() -> Vec<[u32; 3]> {
    vec![
        [0, 1, 2],  // negative winding -> exercises the swap under cull_back=false
        [3, 4, 5],  // negative winding -> also swapped
        [6, 7, 8],  // positive (front-facing) winding -> survives back-face cull
        [0, 1, 99], // out-of-range index -> dropped on both paths
    ]
}

/// CPU golden packed vis-buffer for an indexed cluster.
fn cpu_cluster_vis(
    vertices: &[ScreenVertex],
    tris: &[[u32; 3]],
    cluster_id: u32,
    cull_back: bool,
) -> Vec<u64> {
    let mut buffer = VisBuffer::new(WIDTH, HEIGHT);
    rasterize_cluster(&mut buffer, vertices, tris, cluster_id, cull_back);
    buffer.pixels().to_vec()
}

/// Asserts GPU/CPU vis-buffer parity with the documented tolerance: coverage is
/// bit-exact (cleared iff cleared), the winning payload is bit-exact, and the
/// reversed-Z depth key matches within one unit in the last place (the
/// barycentric depth blend is subject to sub-`ULP` reassociation between the
/// two backends). Returns the number of covered texels.
fn assert_vis_parity(gpu: &[u64], cpu: &[u64], label: &str) -> usize {
    assert_eq!(gpu.len(), cpu.len(), "vis-buffer length must match ({label})");
    let mut covered = 0usize;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let idx = (y as usize) * (WIDTH as usize) + (x as usize);
            assert_eq!(
                gpu[idx] == 0,
                cpu[idx] == 0,
                "coverage mismatch at ({x}, {y}) [{label}]: gpu {}, cpu {}",
                gpu[idx],
                cpu[idx]
            );
            if cpu[idx] == 0 {
                continue;
            }
            covered += 1;
            assert_eq!(
                vis_payload(gpu[idx]),
                vis_payload(cpu[idx]),
                "payload mismatch at ({x}, {y}) [{label}]"
            );
            let diff = vis_depth(gpu[idx]).abs_diff(vis_depth(cpu[idx]));
            assert!(
                diff <= 1,
                "depth key at ({x}, {y}) differs by {diff} bits (> 1 ULP) [{label}]"
            );
        }
    }
    covered
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without the 64-bit atomic feature"
)]
fn gpu_cluster_raster_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cluster parity: no wgpu adapter on this host");
        return;
    };
    if !ctx.supports_u64_atomics() {
        eprintln!("skipping cluster parity: adapter lacks 64-bit atomic features");
        return;
    }
    let Some(raster) = GpuClusterRaster::new(&ctx) else {
        eprintln!("skipping cluster parity: cluster pipeline unavailable");
        return;
    };
    let vertices = shared_vertices();
    let tris = triangles();

    for cull_back in [false, true] {
        let cpu = cpu_cluster_vis(&vertices, &tris, CLUSTER_ID, cull_back);
        let gpu = raster
            .rasterize(&ctx, WIDTH, HEIGHT, &vertices, &tris, CLUSTER_ID, cull_back)
            .expect("cluster raster on a valid framebuffer should succeed");

        assert_eq!(
            gpu.len(),
            cpu.len(),
            "vis-buffer length must match (cull_back = {cull_back})"
        );

        let mut covered = 0usize;
        let mut saw_cluster_high_bits = false;
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
                // the composited payload is bit-exact, including the high
                // cluster_id bits the device synthesizes.
                assert_eq!(
                    vis_payload(gpu[idx]),
                    vis_payload(cpu[idx]),
                    "payload mismatch at ({x}, {y}) with cull_back = {cull_back}"
                );
                // The 7-bit triangle field peels off; the remaining high bits
                // must be the cluster id folded in on the device.
                if (vis_payload(gpu[idx]) >> 7) == CLUSTER_ID {
                    saw_cluster_high_bits = true;
                }
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
        assert!(
            saw_cluster_high_bits,
            "at least one covered texel must carry the cluster_id in its high \
             payload bits (cull_back = {cull_back})"
        );
    }
}

#[test]
fn out_of_range_index_is_dropped() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    if !ctx.supports_u64_atomics() {
        return;
    }
    let Some(raster) = GpuClusterRaster::new(&ctx) else {
        return;
    };
    // A single triangle whose third index is past the (empty-ish) vertex array
    // must contribute nothing on both paths, leaving a fully cleared buffer.
    let vertices = vec![sv(2.0, 2.0, 0.5), sv(2.0, 20.0, 0.5)];
    let tris = vec![[0u32, 1, 7]]; // index 7 >= vert_count (2) -> dropped
    let cpu = cpu_cluster_vis(&vertices, &tris, CLUSTER_ID, false);
    let gpu = raster
        .rasterize(&ctx, WIDTH, HEIGHT, &vertices, &tris, CLUSTER_ID, false)
        .expect("cluster raster on a valid framebuffer should succeed");
    assert_eq!(gpu, cpu, "out-of-range index must drop identically");
    assert!(
        gpu.iter().all(|&w| w == 0),
        "a dropped triangle must leave the buffer cleared"
    );
}

#[test]
fn triangles_past_cap_are_dropped() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    if !ctx.supports_u64_atomics() {
        return;
    }
    let Some(raster) = GpuClusterRaster::new(&ctx) else {
        return;
    };
    // Region A (indices 0..3) is covered by the first `MAX_CLUSTER_TRIANGLES`
    // triangles. Region B (indices 3..6) sits at a distinct, nearer depth and
    // is only referenced by the (MAX + 1)-th triangle, which must be dropped by
    // both paths so region B stays cleared.
    let vertices = vec![
        sv(2.0, 2.0, 0.5),   // 0  region A
        sv(2.0, 14.0, 0.5),  // 1
        sv(14.0, 8.0, 0.5),  // 2
        sv(20.0, 20.0, 0.9), // 3  region B, nearer
        sv(20.0, 30.0, 0.9), // 4
        sv(30.0, 25.0, 0.9), // 5
    ];
    let mut tris: Vec<[u32; 3]> = vec![[0, 1, 2]; MAX_CLUSTER_TRIANGLES];
    tris.push([3, 4, 5]); // the (MAX + 1)-th triangle -> must be dropped
    assert_eq!(tris.len(), MAX_CLUSTER_TRIANGLES + 1);

    let cpu = cpu_cluster_vis(&vertices, &tris, CLUSTER_ID, false);
    let gpu = raster
        .rasterize(&ctx, WIDTH, HEIGHT, &vertices, &tris, CLUSTER_ID, false)
        .expect("cluster raster on a valid framebuffer should succeed");

    // 128 overlapping identical triangles blend depth through different
    // barycentric orderings on each backend, so compare with the documented
    // 1-ULP depth tolerance rather than bit-exact equality.
    assert_vis_parity(&gpu, &cpu, "triangles_past_cap");

    // Region B must be entirely cleared: the only triangle referencing it was
    // dropped by the cap.
    let mut region_b_covered = false;
    for y in 21..30 {
        for x in 21..30 {
            let idx = (y as usize) * (WIDTH as usize) + (x as usize);
            if gpu[idx] != 0 {
                region_b_covered = true;
            }
        }
    }
    assert!(
        !region_b_covered,
        "the triangle past the per-cluster cap must be dropped, leaving region B cleared"
    );
    // Region A must still be covered, proving the first MAX triangles ran.
    assert!(
        gpu.iter().any(|&w| w != 0),
        "the first MAX_CLUSTER_TRIANGLES triangles must still rasterize region A"
    );
}

#[test]
fn empty_framebuffer_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    if !ctx.supports_u64_atomics() {
        return;
    }
    let Some(raster) = GpuClusterRaster::new(&ctx) else {
        return;
    };
    assert!(raster
        .rasterize(&ctx, 0, 8, &[], &[], CLUSTER_ID, false)
        .is_err());
    assert!(raster
        .rasterize(&ctx, 8, 0, &[], &[], CLUSTER_ID, false)
        .is_err());
}
