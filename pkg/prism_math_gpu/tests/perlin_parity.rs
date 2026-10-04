//! Real-device parity for the §24.1 Perlin gradient-noise shader mirror
//! (classic 2D/3D "improved" noise).
//!
//! Both forward kernels sample Perlin noise on a real `GPU` from the
//! single-sourced [`WGSL_PERLIN`](prism_math::shader_mirror::WGSL_PERLIN)
//! fragment, and each result is diffed against the CPU reference
//! [`prism_math::noise::Perlin::get2`] / [`get3`](prism_math::noise::Perlin::get3).
//!
//! The seeded 512-entry permutation table is uploaded verbatim, so the integer
//! hash path is bit-exact and both sides dot the identical gradient at every
//! lattice corner. Only the quintic fade, the gradient dot products, and the
//! lerps are floating point, so parity is a tight absolute+relative tolerance
//! that only absorbs Metal fast-math last-ULP rounding. The suite skips
//! gracefully when no adapter is available.

use prism_math::noise::Perlin;
use prism_math_gpu::GpuContext;
use prism_math_gpu::GpuPerlin;

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

/// A small deterministic linear-congruential sequence.
fn lcg(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *seed
}

/// Returns `true` if `a` and `b` agree within a combined absolute+relative
/// tolerance.
fn close(a: f32, b: f32, tol: f32) -> bool {
    let diff = (a - b).abs();
    diff <= tol + tol * a.abs().max(b.abs())
}

/// Maps a `u32` to a float in a moderate `[-64, 64)` range: the integer lattice
/// coordinate `floor(x)` therefore stays well within `i32`, so the bit-exact
/// hash path never overflows while the fractional part still exercises the full
/// fade/gradient curve.
fn coord(bits: u32) -> f32 {
    let unit = (bits % 1_000_000) as f32 / 1_000_000.0;
    (unit - 0.5) * 128.0
}

#[test]
fn get2_matches_cpu_within_tolerance() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuPerlin::new(&ctx);
    for &seed in &[0u64, 1, 0x5151_a2b3_c4d5_e6f7, u64::MAX] {
        let cpu = Perlin::new(seed);
        let mut points: Vec<[f32; 2]> = Vec::new();
        let mut rng = 0x1234_5678u32 ^ (seed as u32);
        // A large batch spanning multiple workgroups.
        for _ in 0..4099 {
            points.push([coord(lcg(&mut rng)), coord(lcg(&mut rng))]);
        }
        // Exact integer lattice points (fractional part zero) and tiny offsets.
        for &p in &[[0.0, 0.0], [1.0, -1.0], [12.0, 7.0], [-5.0, 3.0], [0.5, 0.5]] {
            points.push(p);
        }

        let gpu = kernel.get2(&ctx, seed, &points);
        assert_eq!(gpu.len(), points.len());
        for (&[x, y], &g) in points.iter().zip(gpu.iter()) {
            let c = cpu.get2(x, y);
            assert!(
                close(g, c, 1.0e-4),
                "perlin get2 drift at ({x}, {y}) seed {seed}: gpu {g} cpu {c}"
            );
        }
    }
}

#[test]
fn get3_matches_cpu_within_tolerance() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuPerlin::new(&ctx);
    for &seed in &[0u64, 7, 0x9e37_79b9_7f4a_7c15, u64::MAX] {
        let cpu = Perlin::new(seed);
        let mut points: Vec<[f32; 3]> = Vec::new();
        let mut rng = 0x90ab_cdefu32 ^ (seed as u32);
        for _ in 0..4099 {
            points.push([
                coord(lcg(&mut rng)),
                coord(lcg(&mut rng)),
                coord(lcg(&mut rng)),
            ]);
        }
        for &p in &[
            [0.0, 0.0, 0.0],
            [1.0, -1.0, 2.0],
            [12.0, 7.0, -3.0],
            [0.5, 0.5, 0.5],
        ] {
            points.push(p);
        }

        let gpu = kernel.get3(&ctx, seed, &points);
        assert_eq!(gpu.len(), points.len());
        for (&[x, y, z], &g) in points.iter().zip(gpu.iter()) {
            let c = cpu.get3(x, y, z);
            assert!(
                close(g, c, 1.0e-4),
                "perlin get3 drift at ({x}, {y}, {z}) seed {seed}: gpu {g} cpu {c}"
            );
        }
    }
}

#[test]
fn empty_batches_are_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuPerlin::new(&ctx);
    assert!(kernel.get2(&ctx, 0, &[]).is_empty());
    assert!(kernel.get3(&ctx, 0, &[]).is_empty());
}
