//! Real-device parity for the curl-noise twin:
//! [`GpuCurlNoise`](prism_volumetric_gpu::curl_noise::GpuCurlNoise) must
//! reproduce the `CPU` golden
//! [`particle::curl_noise`](prism_render_architecture::particle::curl_noise)
//! across lattice points, in-cell interiors, several frequencies and
//! amplitudes, a large random batch, and the divergence-free property.
//!
//! The tests skip (with a printed notice on the first) when the host has no
//! `wgpu` adapter, so the suite stays green everywhere while still exercising
//! the full dispatch-and-readback on any real device such as an Apple
//! `M`-series `GPU`. The kernel is portable core-`WGSL`, so it needs no optional
//! device feature.
//!
//! # Parity criterion
//!
//! The lattice hash is bit-identical (pure unsigned-integer work), so the only
//! values that can diverge are the float fade / `lerp` / central-difference
//! blends, and only by a legal fused multiply-add contraction of a few units in
//! the last place amplified by the `1 / (2 * CURL_EPS)` difference quotient.
//! The comparison allows `abs_diff <= 1e-4` or `rel_diff <= 1e-5` — loose enough
//! to admit that slack yet tight enough to fail a wrong port (a swapped curl
//! component, a wrong fade polynomial, a mis-salted channel, a wrong amplitude
//! scale). Several scenarios additionally assert a non-trivial velocity so a
//! degenerate all-zero kernel could not pass.
//!
//! Provenance: standard analytic curl-noise advection; no Unreal Engine source
//! or derived code.

use prism_render_architecture::particle::curl_noise::{CurlNoiseField, Vec3};
use prism_volumetric_gpu::curl_noise::{CurlNoiseSample, GpuCurlNoise};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. The central-difference quotient scales the raw
/// potential blend by `1 / (2 * CURL_EPS) = 50`, so a few-`ULP` fused
/// multiply-add slack in the blend lands here; `1e-4` admits it while still
/// failing a genuinely wrong port.
const ABS_EPS: f32 = 1.0e-4;

/// Relative parity bound, applied when the compared magnitude is large enough
/// that an absolute bound would be unfairly strict.
const REL_EPS: f32 = 1.0e-5;

/// The relative-error denominator floor, keeping it away from zero.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= ABS_EPS || rel <= REL_EPS
}

/// Asserts two vectors agree component-wise within [`close`].
fn close_vec(gpu: Vec3, cpu: Vec3, idx: usize) {
    assert!(
        close(gpu.x, cpu.x) && close(gpu.y, cpu.y) && close(gpu.z, cpu.z),
        "velocity mismatch at point {idx}: gpu ({}, {}, {}), cpu ({}, {}, {})",
        gpu.x,
        gpu.y,
        gpu.z,
        cpu.x,
        cpu.y,
        cpu.z
    );
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[-1, 1)`.
fn lcg(state: &mut u64) -> f32 {
    // Knuth multiplier / increment; the shift takes the high bits where the
    // generator mixes best.
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    // 24 usable mantissa bits mapped onto [0, 1) then onto [-1, 1).
    let unit = (bits & 0x00ff_ffff) as f32 / 16_777_216.0;
    unit * 2.0 - 1.0
}

/// Runs the `GPU` twin over `points` for `field` and asserts point-by-point
/// velocity parity against the `CPU` reference. Returns the `GPU` samples for
/// extra assertions.
fn check_points(
    ctx: &GpuContext,
    gpu: &GpuCurlNoise,
    field: &CurlNoiseField,
    points: &[Vec3],
) -> Vec<CurlNoiseSample> {
    let samples = gpu.eval(ctx, field, points);
    assert_eq!(samples.len(), points.len(), "one sample per query point");
    for (idx, (p, sample)) in points.iter().zip(samples.iter()).enumerate() {
        let cpu_vel = field.sample_velocity(*p);
        close_vec(sample.velocity, cpu_vel, idx);
    }
    samples
}

/// Asserts that at least one point carries a non-trivial velocity, so a
/// degenerate all-zero kernel could not pass this scene.
fn assert_non_trivial(samples: &[CurlNoiseSample]) {
    let any = samples.iter().any(|s| s.velocity.length() > 1.0e-3);
    assert!(any, "scene should produce a non-zero velocity somewhere");
}

/// A handful of irregular sample points spread across the signed lattice,
/// including an exact lattice point and in-cell interiors.
const SAMPLES: [Vec3; 6] = [
    Vec3::new(0.37, -1.21, 2.05),
    Vec3::new(-3.05, 0.92, -0.48),
    Vec3::new(5.5, 5.5, 5.5),
    Vec3::new(-7.3, -2.1, 4.9),
    Vec3::new(0.0, 0.0, 0.0),
    Vec3::new(12.75, -8.4, 3.33),
];

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_yields_empty_output() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping curl-noise parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuCurlNoise::new(&ctx);
    let field = CurlNoiseField::new(0.7, 2.5, 0x1234_5678);
    let samples = gpu.eval(&ctx, &field, &[]);
    assert!(samples.is_empty(), "empty input must yield empty output");
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_lattice_and_interiors() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping curl-noise parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuCurlNoise::new(&ctx);
    let field = CurlNoiseField::new(0.9, 1.75, 42);

    // Mix exact lattice points (integer coordinates) with in-cell interiors
    // (fractional coordinates), the two regimes of the trilinear blend.
    let points = [
        Vec3::new(1.0, 2.0, -3.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.5, 2.5, -3.5),
        Vec3::new(0.25, 0.75, 0.1),
        Vec3::new(-4.0, 5.0, 6.0),
        Vec3::new(-4.33, 5.67, 6.01),
    ];
    let samples = check_points(&ctx, &gpu, &field, &points);
    assert_non_trivial(&samples);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_across_frequencies_and_amplitudes() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping curl-noise parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuCurlNoise::new(&ctx);
    let fields = [
        CurlNoiseField::new(1.3, 1.0, 1),
        CurlNoiseField::new(0.25, 9.0, 99),
        CurlNoiseField::new(3.0, 0.5, 0xDEAD_BEEF),
        CurlNoiseField::new(0.05, 4.0, 7),
    ];
    for field in fields {
        let samples = check_points(&ctx, &gpu, &field, &SAMPLES);
        assert_non_trivial(&samples);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_large_random_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping curl-noise parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuCurlNoise::new(&ctx);
    let field = CurlNoiseField::new(0.8, 2.0, 0xABCD_1234);

    // Several thousand points spread across many lattice cells, so the batch
    // crosses several workgroups and exercises many distinct hash cells.
    let mut state = 0x5151_2727_9999_3333u64;
    let count = 4096usize;
    let mut points = Vec::with_capacity(count);
    for _ in 0..count {
        // Scatter across roughly [-16, 16) on each axis.
        let x = lcg(&mut state) * 16.0;
        let y = lcg(&mut state) * 16.0;
        let z = lcg(&mut state) * 16.0;
        points.push(Vec3::new(x, y, z));
    }
    let samples = check_points(&ctx, &gpu, &field, &points);
    assert_non_trivial(&samples);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_field_is_divergence_free() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping curl-noise parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuCurlNoise::new(&ctx);
    let field = CurlNoiseField::new(0.7, 2.5, 0x1234_5678);
    let samples = gpu.eval(&ctx, &field, &SAMPLES);
    assert_eq!(samples.len(), SAMPLES.len());

    for (idx, sample) in samples.iter().enumerate() {
        // The discrete curl / divergence stencils commute, so the on-device
        // divergence cancels to f32 rounding, exactly like the CPU reference.
        assert!(
            sample.divergence.abs() < 1.0e-2,
            "gpu divergence at point {idx} should vanish, got {}",
            sample.divergence
        );
        // And the GPU divergence must track the CPU divergence closely.
        let cpu_div = field.divergence(SAMPLES[idx]);
        assert!(
            (sample.divergence - cpu_div).abs() < 1.0e-2,
            "gpu/cpu divergence mismatch at point {idx}: gpu {}, cpu {}",
            sample.divergence,
            cpu_div
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn zero_frequency_and_zero_amplitude_are_the_zero_field() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping curl-noise parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuCurlNoise::new(&ctx);

    // A zero frequency collapses the potential to a constant and a zero
    // amplitude scales the curl away; both yield the zero field on device.
    for field in [
        CurlNoiseField::new(0.0, 3.0, 7),
        CurlNoiseField::new(1.5, 0.0, 7),
    ] {
        let samples = gpu.eval(&ctx, &field, &SAMPLES);
        assert_eq!(samples.len(), SAMPLES.len());
        for (idx, sample) in samples.iter().enumerate() {
            let cpu_vel = field.sample_velocity(SAMPLES[idx]);
            close_vec(sample.velocity, cpu_vel, idx);
            assert!(
                sample.velocity.length() < ABS_EPS,
                "zero field should vanish at point {idx}, got length {}",
                sample.velocity.length()
            );
        }
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn distinct_seeds_produce_distinct_fields_on_device() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping curl-noise parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuCurlNoise::new(&ctx);
    let field_a = CurlNoiseField::new(0.9, 1.0, 1);
    let field_b = CurlNoiseField::new(0.9, 1.0, 2);

    let a = check_points(&ctx, &gpu, &field_a, &SAMPLES);
    let b = check_points(&ctx, &gpu, &field_b, &SAMPLES);

    // Both agree with their own CPU reference (checked above); here the two
    // seeds must disagree somewhere, so the seed is not inert on device.
    let differs = a.iter().zip(b.iter()).any(|(sa, sb)| {
        (sa.velocity.x - sb.velocity.x).abs() > 1.0e-3
            || (sa.velocity.y - sb.velocity.y).abs() > 1.0e-3
            || (sa.velocity.z - sb.velocity.z).abs() > 1.0e-3
    });
    assert!(differs, "distinct seeds should produce distinct fields");
}
