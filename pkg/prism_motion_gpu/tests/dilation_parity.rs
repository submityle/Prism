//! Real-device parity for the closest-depth motion-vector dilation kernel.
//!
//! Each test builds the velocity + depth fields, runs the CPU golden
//! [`dilate_closest_depth`], runs the `GPU` kernel on a real adapter, and
//! asserts the two agree **bit-for-bit** (every component's `f32::to_bits`),
//! because the kernel only compares depths and copies whole velocity vectors.
//! The suite skips gracefully when no adapter is available so it still passes on
//! a device-less CI image, while running the full dispatch on a real `GPU`.

use prism_motion_gpu::context::GpuContext;
use prism_motion_gpu::dilation::GpuDilate;
use prism_render_architecture::motion::dilation::{
    dilate_closest_depth, DepthField, DepthOrder, VelocityField,
};
use prism_render_architecture::motion::Vec2;

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

/// Deterministic LCG byte-ish stream mapped to `f32`, no transcendentals.
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Lcg {
        Lcg { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// Uniform-ish `f32` in `[0, 1)` from the high bits.
    fn unit(&mut self) -> f32 {
        let v = self.next_u32() >> 8; // 24 bits of entropy
        (v as f32) / (16_777_216.0_f32)
    }

    /// `f32` in `[-range, range)`.
    fn signed(&mut self, range: f32) -> f32 {
        (self.unit() * 2.0 - 1.0) * range
    }
}

fn bits_eq(a: Vec2, b: Vec2) -> bool {
    a.x.to_bits() == b.x.to_bits() && a.y.to_bits() == b.y.to_bits()
}

/// Asserts GPU == golden bit-for-bit over the whole field.
fn assert_parity(gpu: &VelocityField, golden: &VelocityField) {
    assert_eq!(gpu.width(), golden.width(), "width mismatch");
    assert_eq!(gpu.height(), golden.height(), "height mismatch");
    for y in 0..golden.height() {
        for x in 0..golden.width() {
            let g = gpu.get(x, y).expect("gpu in bounds");
            let c = golden.get(x, y).expect("golden in bounds");
            assert!(bits_eq(g, c), "pixel ({x},{y}): gpu {g:?} != golden {c:?}");
        }
    }
}

fn field(width: usize, height: usize, values: &[(f32, f32)]) -> VelocityField {
    let data = values.iter().map(|&(x, y)| Vec2::new(x, y)).collect();
    VelocityField::from_pixels(width, height, data).expect("dimensions match")
}

#[test]
fn radius_zero_is_identity() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuDilate::new(&ctx);

    let vel = field(2, 1, &[(5.0, -3.0), (-1.0, 2.0)]);
    let depth = DepthField::from_depths(2, 1, vec![0.2, 0.8]).expect("matches");
    let golden = dilate_closest_depth(&vel, &depth, 0, DepthOrder::SmallerIsCloser).expect("dims");
    let gpu = kernel
        .dilate(&ctx, &vel, &depth, 0, DepthOrder::SmallerIsCloser)
        .expect("dims");
    assert_parity(&gpu, &golden);
    // Anti-vacuous: radius 0 must preserve the input exactly.
    assert!(bits_eq(gpu.get(0, 0).unwrap(), Vec2::new(5.0, -3.0)));
    assert!(bits_eq(gpu.get(1, 0).unwrap(), Vec2::new(-1.0, 2.0)));
}

#[test]
fn forward_z_silhouette_bleeds_outward() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuDilate::new(&ctx);

    // Near foreground (small depth, big velocity) on the left; far background on
    // the right. With radius 1 the background pixel adopts the foreground.
    let vel = field(2, 1, &[(9.0, 1.0), (0.0, 0.0)]);
    let depth = DepthField::from_depths(2, 1, vec![0.1, 0.9]).expect("matches");
    let golden = dilate_closest_depth(&vel, &depth, 1, DepthOrder::SmallerIsCloser).expect("dims");
    let gpu = kernel
        .dilate(&ctx, &vel, &depth, 1, DepthOrder::SmallerIsCloser)
        .expect("dims");
    assert_parity(&gpu, &golden);
    // Anti-vacuous: the background pixel genuinely changed (bled).
    assert!(bits_eq(gpu.get(1, 0).unwrap(), Vec2::new(9.0, 1.0)));
}

#[test]
fn reversed_z_picks_larger_depth() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuDilate::new(&ctx);

    // Reversed-Z: the larger depth (0.9) is the near foreground, so its velocity
    // spreads with radius 2 across a wider strip.
    let vel = field(3, 1, &[(0.0, 0.0), (0.0, 0.0), (7.0, -4.0)]);
    let depth = DepthField::from_depths(3, 1, vec![0.1, 0.5, 0.9]).expect("matches");
    let golden = dilate_closest_depth(&vel, &depth, 2, DepthOrder::LargerIsCloser).expect("dims");
    let gpu = kernel
        .dilate(&ctx, &vel, &depth, 2, DepthOrder::LargerIsCloser)
        .expect("dims");
    assert_parity(&gpu, &golden);
    // Anti-vacuous: all three pixels adopt the foreground velocity.
    for x in 0..3 {
        assert!(bits_eq(gpu.get(x, 0).unwrap(), Vec2::new(7.0, -4.0)));
    }
}

#[test]
fn non_square_dims_match_golden() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuDilate::new(&ctx);

    let width = 5;
    let height = 3;
    let mut rng = Lcg::new(0x1234_5678);
    let vel: Vec<(f32, f32)> = (0..width * height)
        .map(|_| (rng.signed(10.0), rng.signed(10.0)))
        .collect();
    let depths: Vec<f32> = (0..width * height).map(|_| rng.unit()).collect();
    let vel = field(width, height, &vel);
    let depth = DepthField::from_depths(width, height, depths).expect("matches");

    for order in [DepthOrder::SmallerIsCloser, DepthOrder::LargerIsCloser] {
        for radius in 0..=2 {
            let golden = dilate_closest_depth(&vel, &depth, radius, order).expect("dims");
            let gpu = kernel
                .dilate(&ctx, &vel, &depth, radius, order)
                .expect("dims");
            assert_parity(&gpu, &golden);
        }
    }
}

#[test]
fn large_multi_workgroup_grid_matches_golden() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuDilate::new(&ctx);

    // 40x24 spans several 8x8 workgroups in both dimensions.
    let width = 40;
    let height = 24;
    let mut rng = Lcg::new(0x9e37_79b9);
    let vel_vals: Vec<(f32, f32)> = (0..width * height)
        .map(|_| (rng.signed(12.0), rng.signed(12.0)))
        .collect();
    let depths: Vec<f32> = (0..width * height).map(|_| rng.unit()).collect();
    let vel = field(width, height, &vel_vals);
    let depth = DepthField::from_depths(width, height, depths).expect("matches");

    let mut any_changed = false;
    for order in [DepthOrder::SmallerIsCloser, DepthOrder::LargerIsCloser] {
        for radius in [1usize, 2, 3] {
            let golden = dilate_closest_depth(&vel, &depth, radius, order).expect("dims");
            let gpu = kernel
                .dilate(&ctx, &vel, &depth, radius, order)
                .expect("dims");
            assert_parity(&gpu, &golden);
            // Track whether dilation actually moved any velocity.
            for (g, v) in golden.as_slice().iter().zip(vel.as_slice().iter()) {
                if !bits_eq(*g, *v) {
                    any_changed = true;
                }
            }
        }
    }
    // Anti-vacuous: on random depths the dilation must change at least one pixel
    // somewhere, otherwise the kernel/golden could be trivially copying.
    assert!(any_changed, "dilation never changed any pixel");
}

#[test]
fn dimension_mismatch_returns_none() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuDilate::new(&ctx);

    let vel = VelocityField::zeroed(2, 2);
    let depth = DepthField::from_depths(2, 1, vec![0.0, 0.0]).expect("matches");
    assert!(kernel
        .dilate(&ctx, &vel, &depth, 1, DepthOrder::SmallerIsCloser)
        .is_none());
}
