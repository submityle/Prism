//! Real-device parity: the `GPU` software-raster twin must reproduce the `CPU`
//! golden vis-buffer's depth keys texel-for-texel.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full dispatch-and-readback on any machine with a real device.
//!
//! # Parity criterion
//!
//! Every vertex sits on an integer pixel coordinate, so the signed edge
//! function is evaluated on exactly representable operands: its value is
//! identical on `CPU` and `GPU` regardless of fused-multiply-add contraction.
//! The coverage decision (which texels a triangle owns under the top-left
//! rule) is therefore **bit-exact**, and the test asserts the covered texel
//! *set* matches exactly: a texel is cleared (`0`) on the `GPU` iff it is
//! cleared on the `CPU`.
//!
//! The depth *value* is a screen-space-linear barycentric blend
//! `b0*d0 + b1*d1 + b2*d2`. This three-term sum is subject to backend
//! floating-point reassociation (a `GPU` may evaluate it with fused
//! multiply-add or a dot-product instruction in a different grouping than the
//! `CPU`'s strict left-to-right adds), which perturbs the low bit. The depth
//! key is thus compared **within one unit in the last place** (`|gpu - cpu|`
//! at most `1` in monotonic reversed-Z bit space), the same tight floating
//! tolerance the physics `XPBD` twin uses. This is a faithful parity claim:
//! the coverage rule and reversed-Z encode are pinned exactly, and the blend
//! matches to the last representable bit.
//!
//! Provenance: standard signed-edge / top-left-rule software rasterization; no
//! Unreal Engine source or derived code.

use prism_render_architecture::virtual_geometry::{
    rasterize_triangle, vis_depth, ScreenVertex, VisBuffer,
};
use prism_virtual_geometry_gpu::{GpuContext, GpuSoftwareRaster};

const WIDTH: u32 = 32;
const HEIGHT: u32 = 32;

fn sv(x: f32, y: f32, depth: f32) -> ScreenVertex {
    ScreenVertex::new([x, y], depth)
}

/// Overlapping triangles on integer coordinates with power-of-two depths. The
/// second triangle overlaps the first at a nearer depth so the `atomicMax`
/// composite is exercised with genuine contention, and the third is wound
/// clockwise to drive the winding-swap path.
fn scene() -> Vec<[ScreenVertex; 3]> {
    vec![
        // Counter-clockwise (y-down positive area), far.
        [
            sv(2.0, 2.0, 0.125),
            sv(2.0, 26.0, 0.125),
            sv(26.0, 14.0, 0.125),
        ],
        // Overlaps the first, nearer -> wins the composite where they overlap.
        [
            sv(6.0, 6.0, 0.5),
            sv(6.0, 22.0, 0.5),
            sv(24.0, 14.0, 0.5),
        ],
        // Clockwise winding (negative raw area) -> exercises the vertex swap.
        [
            sv(10.0, 4.0, 0.25),
            sv(28.0, 10.0, 0.25),
            sv(10.0, 20.0, 0.25),
        ],
    ]
}

fn cpu_depth_keys(triangles: &[[ScreenVertex; 3]], cull_back: bool) -> Vec<u32> {
    let mut buffer = VisBuffer::new(WIDTH, HEIGHT);
    for (i, tri) in triangles.iter().enumerate() {
        rasterize_triangle(&mut buffer, *tri, i as u32, cull_back);
    }
    buffer
        .pixels()
        .iter()
        .map(|&packed| vis_depth(packed))
        .collect()
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_raster_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU software-raster parity: no wgpu adapter on this host");
        return;
    };
    let raster = GpuSoftwareRaster::new(&ctx);
    let triangles = scene();

    for cull_back in [false, true] {
        let cpu = cpu_depth_keys(&triangles, cull_back);
        let gpu = raster
            .rasterize(&ctx, WIDTH, HEIGHT, &triangles, cull_back)
            .expect("raster on a valid framebuffer should succeed");

        assert_eq!(
            gpu.len(),
            cpu.len(),
            "depth-key buffer length must match (cull_back = {cull_back})"
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
                     gpu key {}, cpu key {}",
                    gpu[idx],
                    cpu[idx]
                );
                // Depth key matches within one ULP (reversed-Z bits are
                // monotonic, so a 1-bit difference is exactly 1 ULP).
                let diff = gpu[idx].abs_diff(cpu[idx]);
                assert!(
                    diff <= 1,
                    "depth key at ({x}, {y}) differs by {diff} bits (> 1 ULP) \
                     with cull_back = {cull_back}: gpu {}, cpu {}",
                    gpu[idx],
                    cpu[idx]
                );
                if cpu[idx] != 0 {
                    covered += 1;
                }
            }
        }
        assert!(
            covered > 0,
            "scene must cover some texels so the comparison is meaningful (cull_back = {cull_back})"
        );
    }
}

#[test]
fn empty_framebuffer_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let raster = GpuSoftwareRaster::new(&ctx);
    assert!(raster.rasterize(&ctx, 0, 8, &[], false).is_err());
    assert!(raster.rasterize(&ctx, 8, 0, &[], false).is_err());
}
