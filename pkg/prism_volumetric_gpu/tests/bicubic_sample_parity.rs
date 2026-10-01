//! Real-device parity for the bicubic reconstruction twin:
//! [`GpuBicubicSample`](prism_volumetric_gpu::bicubic_sample::GpuBicubicSample)
//! must reproduce the `CPU` golden
//! [`bicubic_sample`](prism_render_architecture::particle::bicubic_sample)
//! `RGBA` sample for sample across the `Catmull-Rom` and generalized
//! `Mitchell-Netravali` reconstruction paths.
//!
//! The fixtures cover the degenerate and boundary shapes the golden unit tests
//! exercise: an empty image (transparent black), a solid image (a constant must
//! survive reconstruction), horizontal/vertical/separable linear ramps (an
//! interpolating `Catmull-Rom` reconstructs a ramp with no error away from the
//! border), integer coordinates that pass straight through a sample point,
//! coordinates well past every edge (`clamp-to-edge`), a step edge (ringing),
//! and a larger deterministic field swept at many fractional coordinates for
//! both the balanced `Mitchell` filter and `Catmull-Rom`.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each sample is a fixed sequence of polynomial-weighted adds, so `CPU` and
//! `GPU` evaluate the same closed form but need not be bit-exact (a `GPU` may
//! contract a multiply-add). The comparison is per channel with
//! `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`), tight enough to
//! catch a genuinely wrong port yet loose enough to admit legal fused
//! multiply-add contraction. Integer addressing is reproduced exactly, so the
//! clamp-to-edge fixtures agree to the same tolerance with no special case.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::bicubic_sample`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::bicubic_sample::SampleImage;
use prism_volumetric_gpu::bicubic_sample::GpuBicubicSample;
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_TOL: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_TOL: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;

/// Balanced `Mitchell-Netravali` parameters.
const MITCHELL_B: f32 = 1.0 / 3.0;
/// Balanced `Mitchell-Netravali` parameters.
const MITCHELL_C: f32 = 1.0 / 3.0;

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_TOL {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_TOL
}

/// Asserts the `GPU` samples match the `CPU` golden channel by channel.
fn assert_rgba_parity(gpu: &[[f32; 4]], cpu: &[[f32; 4]], label: &str) {
    assert_eq!(gpu.len(), cpu.len(), "{label}: sample count mismatch");
    for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        for k in 0..4 {
            assert!(
                close(g[k], c[k]),
                "{label}: sample {i} channel {k} GPU {} vs CPU {}",
                g[k],
                c[k]
            );
        }
    }
}

/// `CPU` golden `sample_bicubic` for a whole coordinate batch.
fn cpu_bicubic(image: &SampleImage, coords: &[[f32; 2]], b: f32, c: f32) -> Vec<[f32; 4]> {
    coords
        .iter()
        .map(|&[u, v]| image.sample_bicubic(u, v, b, c))
        .collect()
}

/// `CPU` golden `sample_catmull_rom` for a whole coordinate batch.
fn cpu_catmull_rom(image: &SampleImage, coords: &[[f32; 2]]) -> Vec<[f32; 4]> {
    coords
        .iter()
        .map(|&[u, v]| image.sample_catmull_rom(u, v))
        .collect()
}

/// Builds a horizontal `value = base + slope * x` ramp in the red channel.
fn ramp_image_x(width: u32, height: u32, base: f32, slope: f32) -> SampleImage {
    let mut data = Vec::new();
    for _y in 0..height {
        for x in 0..width {
            data.push([base + slope * x as f32, 0.0, 0.0, 1.0]);
        }
    }
    SampleImage::new(width, height, data)
}

/// Builds a vertical `value = base + slope * y` ramp in the green channel.
fn ramp_image_y(width: u32, height: u32, base: f32, slope: f32) -> SampleImage {
    let mut data = Vec::new();
    for y in 0..height {
        for _x in 0..width {
            data.push([0.0, base + slope * y as f32, 0.0, 1.0]);
        }
    }
    SampleImage::new(width, height, data)
}

/// A small deterministic linear-congruential generator so the large-field test
/// needs no external randomness and no floating point. Constants are the
/// Numerical Recipes values.
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Lcg { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// A `[0, 1)` fraction built from the top bits, keeping the fixture pure
    /// integer host-side with no transcendental call.
    fn next_unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
}

#[test]
fn empty_image_samples_transparent_black() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBicubicSample::new(&ctx);
    let image = SampleImage::new(0, 0, Vec::new());
    let coords = [[0.0, 0.0], [2.5, 3.5], [-1.0, 7.0]];
    let out = gpu.sample_catmull_rom(&ctx, &image, &coords);
    let want = cpu_catmull_rom(&image, &coords);
    assert_rgba_parity(&out, &want, "empty image");
    for s in &out {
        for &ch in s {
            assert!(close(ch, 0.0), "empty image channel {ch} should be 0");
        }
    }
}

#[test]
fn empty_coord_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBicubicSample::new(&ctx);
    let image = ramp_image_x(4, 4, 0.0, 1.0);
    assert!(gpu.sample_catmull_rom(&ctx, &image, &[]).is_empty());
    assert!(gpu
        .sample_bicubic(&ctx, &image, &[], MITCHELL_B, MITCHELL_C)
        .is_empty());
}

#[test]
fn solid_image_preserves_the_constant() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBicubicSample::new(&ctx);
    let image = SampleImage::solid(6, 5, [0.2, 0.4, 0.6, 0.8]);
    let coords = [[2.0, 2.0], [2.3, 1.7], [0.5, 4.2], [5.0, 0.0]];
    let out = gpu.sample_bicubic(&ctx, &image, &coords, MITCHELL_B, MITCHELL_C);
    let want = cpu_bicubic(&image, &coords, MITCHELL_B, MITCHELL_C);
    assert_rgba_parity(&out, &want, "solid Mitchell");
    let out_cat = gpu.sample_catmull_rom(&ctx, &image, &coords);
    let want_cat = cpu_catmull_rom(&image, &coords);
    assert_rgba_parity(&out_cat, &want_cat, "solid Catmull-Rom");
}

#[test]
fn catmull_rom_passes_through_sample_points() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBicubicSample::new(&ctx);
    let image = ramp_image_x(6, 4, 1.0, 2.0);
    let coords: Vec<[f32; 2]> = (1..5).map(|x| [x as f32, 2.0]).collect();
    let out = gpu.sample_catmull_rom(&ctx, &image, &coords);
    let want = cpu_catmull_rom(&image, &coords);
    assert_rgba_parity(&out, &want, "pass-through x");
}

#[test]
fn catmull_rom_reconstructs_linear_ramps() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBicubicSample::new(&ctx);
    let img_x = ramp_image_x(8, 4, 0.25, 0.5);
    let coords_x = [[2.3_f32, 2.0], [3.5, 2.0], [4.75, 2.0], [5.1, 2.0]];
    assert_rgba_parity(
        &gpu.sample_catmull_rom(&ctx, &img_x, &coords_x),
        &cpu_catmull_rom(&img_x, &coords_x),
        "ramp x",
    );
    let img_y = ramp_image_y(4, 8, -1.0, 0.75);
    let coords_y = [[2.0_f32, 2.2], [2.0, 3.6], [2.0, 4.9], [2.0, 5.4]];
    assert_rgba_parity(
        &gpu.sample_catmull_rom(&ctx, &img_y, &coords_y),
        &cpu_catmull_rom(&img_y, &coords_y),
        "ramp y",
    );
}

#[test]
fn clamps_at_the_border() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBicubicSample::new(&ctx);
    let image = ramp_image_x(4, 4, 0.0, 1.0);
    // Coordinates well past every edge must re-read the edge texels.
    let coords = [[10.0, 1.0], [-4.0, 1.0], [1.0, 20.0], [1.0, -7.0]];
    assert_rgba_parity(
        &gpu.sample_catmull_rom(&ctx, &image, &coords),
        &cpu_catmull_rom(&image, &coords),
        "clamp border",
    );
}

#[test]
fn step_edge_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBicubicSample::new(&ctx);
    // A horizontal step: 0 on the left half, 1 on the right half.
    let mut data = Vec::new();
    for _y in 0..4 {
        for x in 0..8 {
            let value = if x < 4 { 0.0 } else { 1.0 };
            data.push([value, 0.0, 0.0, 1.0]);
        }
    }
    let image = SampleImage::new(8, 4, data);
    let coords: Vec<[f32; 2]> = (0..=40).map(|s| [2.0 + s as f32 / 10.0, 2.0]).collect();
    assert_rgba_parity(
        &gpu.sample_catmull_rom(&ctx, &image, &coords),
        &cpu_catmull_rom(&image, &coords),
        "step Catmull-Rom",
    );
    assert_rgba_parity(
        &gpu.sample_bicubic(&ctx, &image, &coords, MITCHELL_B, MITCHELL_C),
        &cpu_bicubic(&image, &coords, MITCHELL_B, MITCHELL_C),
        "step Mitchell",
    );
}

#[test]
fn large_field_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBicubicSample::new(&ctx);
    // Build a deterministic RGBA field and sweep a large batch of fractional
    // coordinates, including out-of-range sentinels, for both filters.
    let (w, h) = (24u32, 20u32);
    let mut rng = Lcg::new(0x0bad_c0de);
    let mut data = Vec::with_capacity((w * h) as usize);
    for _ in 0..(w * h) {
        data.push([
            rng.next_unit(),
            rng.next_unit(),
            rng.next_unit(),
            rng.next_unit(),
        ]);
    }
    let image = SampleImage::new(w, h, data);
    let mut probe = Lcg::new(0x5eed_1234);
    let mut coords = Vec::with_capacity(600);
    for _ in 0..512 {
        let u = probe.next_unit() * (w as f32 + 4.0) - 2.0;
        let v = probe.next_unit() * (h as f32 + 4.0) - 2.0;
        coords.push([u, v]);
    }
    coords.push([-5.0, -5.0]);
    coords.push([w as f32 + 5.0, h as f32 + 5.0]);
    coords.push([0.0, 0.0]);
    assert_rgba_parity(
        &gpu.sample_bicubic(&ctx, &image, &coords, MITCHELL_B, MITCHELL_C),
        &cpu_bicubic(&image, &coords, MITCHELL_B, MITCHELL_C),
        "large Mitchell",
    );
    assert_rgba_parity(
        &gpu.sample_catmull_rom(&ctx, &image, &coords),
        &cpu_catmull_rom(&image, &coords),
        "large Catmull-Rom",
    );
}
