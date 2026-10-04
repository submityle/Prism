//! Real-device parity for the §24.1 fractal-noise shader mirror: multi-octave
//! fBm, turbulence, and ridged multifractal stacked over either Perlin or
//! Simplex gradient noise.
//!
//! Every forward kernel evaluates the fractal sum on a real `GPU` from the
//! single-sourced [`WGSL_FRACTAL`](prism_math::shader_mirror::WGSL_FRACTAL)
//! fragment (over the matching base fragment), and each result is diffed
//! against the CPU reference [`prism_math::noise::Fractal`].
//!
//! The seeded 512-entry permutation table is uploaded verbatim, so the base
//! sampler's integer hash path is bit-exact; only the octave-weighted float
//! sum plus the base `fade`/`grad`/corner-attenuation math is approximate, so
//! parity is an absolute+relative tolerance. Because each octave's base
//! sample carries its own fast-math rounding and the kernel sums several
//! octaves at `lacunarity`-scaled (hence large-magnitude, lower-fractional-
//! precision) frequencies, those per-octave roundings compound, so the
//! fractal tolerance (`5e-4`) is looser than the single-sample base twins'
//! `1e-4` — still tight enough to reject a genuine algorithm / operand-order
//! / layout drift.
//!
//! The simplex base selects corners with a magnitude branch, which a fast-math
//! wobble could flip on the measure-zero boundary where two skewed coordinates
//! are exactly equal — and octave accumulation would then amplify that jump.
//! The sampling grid below uses pseudo-random and asymmetric hand-picked
//! coordinates that never land on that boundary, so the branch is identical on
//! both sides. The suite skips gracefully when no adapter is available.

use prism_math::noise::Fractal;
use prism_math::noise::{Perlin, Simplex};
use prism_math_gpu::GpuContext;
use prism_math_gpu::{GpuFractal, NoiseSource};

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

/// Maps a `u32` to a float in a moderate `[-64, 64)` range so the skewed base
/// lattice index stays well within `i32` while the fractional part still
/// exercises the full octave stack.
fn coord(bits: u32) -> f32 {
    let unit = (bits % 1_000_003) as f32 / 1_000_003.0;
    (unit - 0.5) * 128.0
}

/// The `Fractal` parameter sets exercised on every source and kernel: the
/// default plus asymmetric octave / lacunarity / gain choices (including the
/// single-octave edge).
fn param_sets() -> [Fractal; 4] {
    [
        Fractal::default(),
        Fractal {
            octaves: 1,
            lacunarity: 2.0,
            gain: 0.5,
            frequency: 1.0,
        },
        Fractal {
            octaves: 6,
            lacunarity: 2.137,
            gain: 0.473,
            frequency: 0.37,
        },
        Fractal {
            octaves: 3,
            lacunarity: 1.871,
            gain: 0.61,
            frequency: 1.9,
        },
    ]
}

/// Builds the shared 2D sampling grid for a seed: a large multi-workgroup batch
/// of pseudo-random points plus asymmetric hand-picked points.
fn grid2(seed: u64) -> Vec<[f32; 2]> {
    let mut points: Vec<[f32; 2]> = Vec::new();
    let mut rng = 0x1234_5678u32 ^ (seed as u32);
    for _ in 0..4099 {
        points.push([coord(lcg(&mut rng)), coord(lcg(&mut rng))]);
    }
    for &p in &[[1.3, -2.7], [12.4, 7.1], [-5.2, 3.9], [0.3, 0.8]] {
        points.push(p);
    }
    points
}

/// Builds the shared 3D sampling grid for a seed.
fn grid3(seed: u64) -> Vec<[f32; 3]> {
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
        [1.3, -2.7, 0.4],
        [12.4, 7.1, -3.6],
        [-5.2, 3.9, 8.1],
        [0.3, 0.8, -0.2],
    ] {
        points.push(p);
    }
    points
}

/// Samples the CPU `Fractal` using the right base generator for `source`.
fn cpu_fbm2(source: NoiseSource, seed: u64, f: &Fractal, x: f32, y: f32) -> f32 {
    match source {
        NoiseSource::Perlin => f.fbm2(&Perlin::new(seed), x, y),
        NoiseSource::Simplex => f.fbm2(&Simplex::new(seed), x, y),
    }
}
fn cpu_turbulence2(source: NoiseSource, seed: u64, f: &Fractal, x: f32, y: f32) -> f32 {
    match source {
        NoiseSource::Perlin => f.turbulence2(&Perlin::new(seed), x, y),
        NoiseSource::Simplex => f.turbulence2(&Simplex::new(seed), x, y),
    }
}
fn cpu_ridged2(source: NoiseSource, seed: u64, f: &Fractal, x: f32, y: f32) -> f32 {
    match source {
        NoiseSource::Perlin => f.ridged2(&Perlin::new(seed), x, y),
        NoiseSource::Simplex => f.ridged2(&Simplex::new(seed), x, y),
    }
}
fn cpu_fbm3(source: NoiseSource, seed: u64, f: &Fractal, x: f32, y: f32, z: f32) -> f32 {
    match source {
        NoiseSource::Perlin => f.fbm3(&Perlin::new(seed), x, y, z),
        NoiseSource::Simplex => f.fbm3(&Simplex::new(seed), x, y, z),
    }
}

#[test]
fn fbm2_matches_cpu_within_tolerance() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    for source in [NoiseSource::Perlin, NoiseSource::Simplex] {
        let kernel = GpuFractal::new(&ctx, source);
        for &seed in &[0u64, 1, 0x5151_a2b3_c4d5_e6f7, u64::MAX] {
            let points = grid2(seed);
            for params in param_sets() {
                let gpu = kernel.fbm2(&ctx, seed, &params, &points);
                assert_eq!(gpu.len(), points.len());
                for (&[x, y], &g) in points.iter().zip(gpu.iter()) {
                    let c = cpu_fbm2(source, seed, &params, x, y);
                    assert!(
                        close(g, c, 5.0e-4),
                        "fbm2 drift at ({x}, {y}) seed {seed} src {source:?}: gpu {g} cpu {c}"
                    );
                }
            }
        }
    }
}

#[test]
fn turbulence2_matches_cpu_within_tolerance() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    for source in [NoiseSource::Perlin, NoiseSource::Simplex] {
        let kernel = GpuFractal::new(&ctx, source);
        for &seed in &[0u64, 3, 0x1357_9bdf_2468_ace0, u64::MAX] {
            let points = grid2(seed);
            for params in param_sets() {
                let gpu = kernel.turbulence2(&ctx, seed, &params, &points);
                assert_eq!(gpu.len(), points.len());
                for (&[x, y], &g) in points.iter().zip(gpu.iter()) {
                    let c = cpu_turbulence2(source, seed, &params, x, y);
                    assert!(
                        close(g, c, 5.0e-4),
                        "turbulence2 drift at ({x}, {y}) seed {seed} src {source:?}: gpu {g} cpu {c}"
                    );
                }
            }
        }
    }
}

#[test]
fn ridged2_matches_cpu_within_tolerance() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    for source in [NoiseSource::Perlin, NoiseSource::Simplex] {
        let kernel = GpuFractal::new(&ctx, source);
        for &seed in &[0u64, 5, 0x2bad_c0de_dead_beef, u64::MAX] {
            let points = grid2(seed);
            for params in param_sets() {
                let gpu = kernel.ridged2(&ctx, seed, &params, &points);
                assert_eq!(gpu.len(), points.len());
                for (&[x, y], &g) in points.iter().zip(gpu.iter()) {
                    let c = cpu_ridged2(source, seed, &params, x, y);
                    assert!(
                        close(g, c, 5.0e-4),
                        "ridged2 drift at ({x}, {y}) seed {seed} src {source:?}: gpu {g} cpu {c}"
                    );
                }
            }
        }
    }
}

#[test]
fn fbm3_matches_cpu_within_tolerance() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    for source in [NoiseSource::Perlin, NoiseSource::Simplex] {
        let kernel = GpuFractal::new(&ctx, source);
        for &seed in &[0u64, 7, 0x9e37_79b9_7f4a_7c15, u64::MAX] {
            let points = grid3(seed);
            for params in param_sets() {
                let gpu = kernel.fbm3(&ctx, seed, &params, &points);
                assert_eq!(gpu.len(), points.len());
                for (&[x, y, z], &g) in points.iter().zip(gpu.iter()) {
                    let c = cpu_fbm3(source, seed, &params, x, y, z);
                    assert!(
                        close(g, c, 5.0e-4),
                        "fbm3 drift at ({x}, {y}, {z}) seed {seed} src {source:?}: gpu {g} cpu {c}"
                    );
                }
            }
        }
    }
}

#[test]
fn empty_batches_are_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuFractal::new(&ctx, NoiseSource::Perlin);
    let params = Fractal::default();
    assert!(kernel.fbm2(&ctx, 0, &params, &[]).is_empty());
    assert!(kernel.fbm3(&ctx, 0, &params, &[]).is_empty());
    assert!(kernel.turbulence2(&ctx, 0, &params, &[]).is_empty());
    assert!(kernel.ridged2(&ctx, 0, &params, &[]).is_empty());
}
