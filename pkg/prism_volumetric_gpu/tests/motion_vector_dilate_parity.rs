//! Real-device parity for the closest-depth velocity dilation per-pixel twin:
//! [`GpuMotionVectorDilate`](prism_volumetric_gpu::motion_vector_dilate::GpuMotionVectorDilate)
//! must reproduce the `CPU` golden
//! [`dilate_closest_depth`](prism_render_architecture::motion::dilation::dilate_closest_depth)
//! exactly — the border-clamped window scan, the `ny`-outer / `nx`-inner order,
//! the center-pixel seed, and the strict
//! [`is_closer`](prism_render_architecture::motion::dilation::DepthOrder::is_closer)
//! tie-break — across hand-checked small grids, every radius from an identity
//! copy up to one that forces full border clamping, both
//! [`DepthOrder`](prism_render_architecture::motion::dilation::DepthOrder)
//! directions, and a randomized full-grid sweep compared pixel-for-pixel.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden
//! [`dilate_closest_depth`](prism_render_architecture::motion::dilation::dilate_closest_depth)
//! is public, so each test calls it directly as the oracle and asserts
//! `GPU == golden`. The kernel performs no floating-point arithmetic — it only
//! compares depth magnitudes and copies a velocity — so for the tie-free depth
//! fixtures used here the two agree *bit for bit*. The comparison is therefore
//! on the raw `f32` bit patterns (via [`f32::to_bits`]), which both honors the
//! exact-equality intent and avoids a bare `f32` `==`.
//!
//! # Conditioning
//!
//! Every fixture gives each pixel a distinct depth, so no two neighbors tie.
//! Because the port's tie-break is strict in both the golden and the kernel, a
//! tie would be resolved by scan order on each side; keeping depths distinct
//! removes any dependence on that and makes the required agreement exact rather
//! than merely tolerant.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::motion::dilation`；无第三方引擎源码或衍生代码。

use prism_render_architecture::motion::dilation::{
    dilate_closest_depth, DepthField, DepthOrder, VelocityField,
};
use prism_render_architecture::motion::Vec2;
use prism_volumetric_gpu::motion_vector_dilate::{
    GpuMotionVectorDilate, MotionVectorDilateInput, MotionVectorDilateOrder,
};
use prism_volumetric_gpu::GpuContext;

/// Maps the twin's ordering selector onto the golden
/// [`DepthOrder`](prism_render_architecture::motion::dilation::DepthOrder) so a
/// single fixture description drives both the device run and the oracle.
fn golden_order(order: MotionVectorDilateOrder) -> DepthOrder {
    match order {
        MotionVectorDilateOrder::SmallerIsCloser => DepthOrder::SmallerIsCloser,
        MotionVectorDilateOrder::LargerIsCloser => DepthOrder::LargerIsCloser,
    }
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental method and no external math dependency are involved. The
/// multiplier and increment are the well-known 64-bit `PCG` constants.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws a float in `[0, 1)` from the generator using the top 24 mantissa bits,
/// built purely from integer arithmetic.
fn lcg_f32(state: &mut u64) -> f32 {
    (lcg(state) & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Runs one fixture on the device and against the golden oracle, asserting the
/// `GPU` velocity equals the golden velocity bit for bit at every pixel.
fn run_case(
    ctx: &GpuContext,
    gpu: &GpuMotionVectorDilate,
    width: u32,
    height: u32,
    radius: u32,
    order: MotionVectorDilateOrder,
    velocities: &[[f32; 2]],
    depths: &[f32],
) {
    let count = (width as usize) * (height as usize);
    assert_eq!(velocities.len(), count, "velocity fixture length");
    assert_eq!(depths.len(), count, "depth fixture length");

    // Golden oracle: build the reference fields and run the public entry point.
    let vel_field = VelocityField::from_pixels(
        width as usize,
        height as usize,
        velocities.iter().map(|v| Vec2::new(v[0], v[1])).collect(),
    )
    .expect("velocity dimensions match");
    let depth_field = DepthField::from_depths(width as usize, height as usize, depths.to_vec())
        .expect("depth dimensions match");
    let golden = dilate_closest_depth(
        &vel_field,
        &depth_field,
        radius as usize,
        golden_order(order),
    )
    .expect("dimensions match");

    // Device twin.
    let input = MotionVectorDilateInput {
        width,
        height,
        radius,
        order,
        velocities,
        depths,
    };
    let got = gpu.evaluate(ctx, &input);
    assert_eq!(got.len(), count, "result count must equal width * height");

    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = y * (width as usize) + x;
            let want = golden.get(x, y).expect("golden pixel in bounds");
            assert_eq!(
                got[idx][0].to_bits(),
                want.x.to_bits(),
                "x at pixel ({x}, {y}) radius {radius}: gpu {} vs cpu {}",
                got[idx][0],
                want.x
            );
            assert_eq!(
                got[idx][1].to_bits(),
                want.y.to_bits(),
                "y at pixel ({x}, {y}) radius {radius}: gpu {} vs cpu {}",
                got[idx][1],
                want.y
            );
        }
    }
}

#[test]
fn empty_input_returns_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionVectorDilate::new(&ctx);
    // A zero-pixel grid is short-circuited on the host and never dispatched.
    let input = MotionVectorDilateInput {
        width: 0,
        height: 0,
        radius: 1,
        order: MotionVectorDilateOrder::SmallerIsCloser,
        velocities: &[],
        depths: &[],
    };
    let got = gpu.evaluate(&ctx, &input);
    assert!(got.is_empty(), "empty grid must return an empty vector");

    // A zero-width (but nonzero-height) grid is also empty.
    let input = MotionVectorDilateInput {
        width: 0,
        height: 4,
        radius: 2,
        order: MotionVectorDilateOrder::LargerIsCloser,
        velocities: &[],
        depths: &[],
    };
    assert!(gpu.evaluate(&ctx, &input).is_empty());
}

#[test]
fn small_grid_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionVectorDilate::new(&ctx);

    // 3x1 forward-Z: the near foreground (depth 0.1) at pixel 0 bleeds its
    // velocity one pixel right; pixel 2 keeps its own (nearest in its window).
    run_case(
        &ctx,
        &gpu,
        3,
        1,
        1,
        MotionVectorDilateOrder::SmallerIsCloser,
        &[[9.0, -1.0], [0.0, 0.0], [1.0, 2.0]],
        &[0.1, 0.9, 0.5],
    );

    // 3x3 with a unique near pixel in the center and all-distinct depths.
    let velocities = [
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0],
        [-5.0, 6.0],
        [7.0, 0.0],
        [8.0, 0.0],
        [9.0, 0.0],
        [10.0, 0.0],
    ];
    let depths = [0.90, 0.80, 0.70, 0.60, 0.11, 0.52, 0.43, 0.34, 0.25];
    run_case(
        &ctx,
        &gpu,
        3,
        3,
        1,
        MotionVectorDilateOrder::SmallerIsCloser,
        &velocities,
        &depths,
    );
}

#[test]
fn radius_variations_match_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionVectorDilate::new(&ctx);

    // One fixed 4x4 grid with all-distinct depths, swept across radii from an
    // identity copy (0) up to one that spans the whole grid (3).
    let velocities: Vec<[f32; 2]> = (0..16).map(|i| [i as f32, (16 - i) as f32]).collect();
    let depths: Vec<f32> = (0..16).map(|i| 0.03 * (i as f32) + 0.01).collect();
    for radius in [0u32, 1, 2, 3] {
        run_case(
            &ctx,
            &gpu,
            4,
            4,
            radius,
            MotionVectorDilateOrder::SmallerIsCloser,
            &velocities,
            &depths,
        );
    }
}

#[test]
fn both_orders_match_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionVectorDilate::new(&ctx);

    // The same grid under both orderings: smaller-is-closer picks the low-depth
    // winner, larger-is-closer picks the high-depth one, so the two runs
    // generally disagree with each other yet each match its own golden.
    let velocities = [
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0],
        [5.0, 0.0],
        [6.0, 0.0],
    ];
    let depths = [0.17, 0.42, 0.08, 0.91, 0.56, 0.33];
    for order in [
        MotionVectorDilateOrder::SmallerIsCloser,
        MotionVectorDilateOrder::LargerIsCloser,
    ] {
        run_case(&ctx, &gpu, 3, 2, 1, order, &velocities, &depths);
    }
}

#[test]
fn boundary_clamp_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionVectorDilate::new(&ctx);

    // A radius far larger than the grid forces every pixel's window to clamp to
    // the full grid, so every output becomes the single globally-nearest
    // velocity. Distinct depths keep that winner unique.
    let velocities: Vec<[f32; 2]> = (0..9).map(|i| [(i * 3) as f32, (i + 1) as f32]).collect();
    let depths = [0.61, 0.22, 0.77, 0.49, 0.05, 0.88, 0.34, 0.95, 0.13];
    for order in [
        MotionVectorDilateOrder::SmallerIsCloser,
        MotionVectorDilateOrder::LargerIsCloser,
    ] {
        run_case(&ctx, &gpu, 3, 3, 5, order, &velocities, &depths);
    }
}

#[test]
fn random_grid_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionVectorDilate::new(&ctx);

    let width = 7u32;
    let height = 5u32;
    let count = (width as usize) * (height as usize);
    let mut state = 0x1234_5678_9abc_def0u64;

    // Velocities are arbitrary; depths are the pixel index plus a jitter < 0.5,
    // so the integer parts keep every depth distinct (no ties) while the device
    // still sees an irregular ordering to scan.
    let mut velocities = Vec::with_capacity(count);
    let mut depths = Vec::with_capacity(count);
    for i in 0..count {
        let vx = lcg_f32(&mut state) * 20.0 - 10.0;
        let vy = lcg_f32(&mut state) * 20.0 - 10.0;
        velocities.push([vx, vy]);
        depths.push(i as f32 + lcg_f32(&mut state) * 0.5);
    }

    for radius in [1u32, 2, 3] {
        for order in [
            MotionVectorDilateOrder::SmallerIsCloser,
            MotionVectorDilateOrder::LargerIsCloser,
        ] {
            run_case(
                &ctx,
                &gpu,
                width,
                height,
                radius,
                order,
                &velocities,
                &depths,
            );
        }
    }
}
