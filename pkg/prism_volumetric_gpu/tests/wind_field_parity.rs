//! Real-device parity for the wind-field twin:
//! [`GpuWindField`](prism_volumetric_gpu::wind_field::GpuWindField) must
//! reproduce the `CPU` golden
//! [`particle::wind_field`](prism_render_architecture::particle::wind_field)
//! across its four public outputs — the gust envelope, the rational height
//! attenuation, the sampled velocity and the relative-velocity drag — on
//! lattice points, in-cell interiors, several seeds and frequencies, and a
//! large random batch, with the raw gust-cell hash asserted bit-identical.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The lattice hash is pure unsigned-integer work, so the gust-cell hash is
//! bit-identical and asserted with exact `==`. The continuous outputs can
//! diverge only by a legal fused multiply-add contraction of a few units in the
//! last place, so they are compared with `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`. Several scenarios additionally assert a non-trivial
//! velocity so a degenerate all-zero kernel could not pass.
//!
//! Provenance: standard art-directable analytic wind model; no Unreal Engine
//! source or derived code.

use prism_render_architecture::particle::wind_field::{Vec3, WindField};
use prism_volumetric_gpu::wind_field::{GpuWindField, WindFieldQuery, WindFieldResult};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound for the continuous outputs.
const ABS_EPS: f32 = 1.0e-4;

/// Relative parity bound, applied when the compared magnitude is large enough
/// that an absolute bound would be unfairly strict.
const REL_EPS: f32 = 1.0e-3;

/// The relative-error denominator floor, keeping it away from zero.
const REL_FLOOR: f32 = 1.0e-6;

/// Odd-integer salt folded into the gust seed; mirrors the golden `GUST_SALT`.
const GUST_SALT: u32 = 0x9E37_79B1;

/// `FNV`-1a offset basis xored into the seed; mirrors the golden `hash_cell`.
const HASH_BASIS: u32 = 0x811C_9DC5;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= ABS_EPS || rel <= REL_EPS
}

/// Asserts two vectors agree component-wise within [`close`].
fn close_vec(gpu: Vec3, cpu: Vec3, idx: usize, what: &str) {
    assert!(
        close(gpu.x, cpu.x) && close(gpu.y, cpu.y) && close(gpu.z, cpu.z),
        "{what} mismatch at query {idx}: gpu ({}, {}, {}), cpu ({}, {}, {})",
        gpu.x,
        gpu.y,
        gpu.z,
        cpu.x,
        cpu.y,
        cpu.z
    );
}

/// One folding step of the integer hash, mirroring the golden `mix`. Pure
/// integer work, so no `f32` transcendental appears in the fixture.
fn mix(mut h: u32, v: u32) -> u32 {
    h ^= v.wrapping_mul(0x9E37_79B1);
    h = h.rotate_left(15).wrapping_mul(0x85EB_CA6B);
    h
}

/// Final avalanche, mirroring the golden `finalize`.
fn finalize(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^= h >> 16;
    h
}

/// Stateless integer hash of a 4D lattice cell and seed, mirroring the golden
/// `hash_cell`, replicated here so parity can assert the on-device integer path
/// bit for bit.
fn hash_cell(i: i32, j: i32, k: i32, l: i32, seed: u32) -> u32 {
    let mut h = seed ^ HASH_BASIS;
    h = mix(h, i as u32);
    h = mix(h, j as u32);
    h = mix(h, k as u32);
    h = mix(h, l as u32);
    finalize(h)
}

/// The reference gust-cell hash for a query against `field`: the hash of the
/// floor cell of the frequency-scaled position and time, seeded by
/// `seed ^ GUST_SALT`. Uses only `floor` (not a transcendental) on the fixture
/// coordinates, which are chosen in cell interiors so the floor is unambiguous.
fn expected_hash(field: &WindField, q: &WindFieldQuery) -> u32 {
    let gf = field.gust_frequency;
    let xi = (q.position.x * gf).floor() as i32;
    let yi = (q.position.y * gf).floor() as i32;
    let zi = (q.position.z * gf).floor() as i32;
    let ti = (q.time * gf).floor() as i32;
    hash_cell(xi, yi, zi, ti, field.seed ^ GUST_SALT)
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[-1, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    let unit = (bits & 0x00ff_ffff) as f32 / 16_777_216.0;
    unit * 2.0 - 1.0
}

/// Draws a strictly positive altitude in roughly `[0.5, 8.5)`, kept well away
/// from zero and from the `MIN_FALLOFF_DENOM` critical region so the rational
/// falloff is sampled on its smooth interior.
fn positive_height(state: &mut u64) -> f32 {
    (lcg(state) + 1.0) * 4.0 + 0.5
}

/// Runs the `GPU` twin over `queries` for `field` and asserts per-query parity
/// against the `CPU` reference on every output, returning the `GPU` results for
/// extra assertions.
fn check_queries(
    ctx: &GpuContext,
    gpu: &GpuWindField,
    field: &WindField,
    queries: &[WindFieldQuery],
) -> Vec<WindFieldResult> {
    let results = gpu.eval(ctx, field, queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (idx, (q, r)) in queries.iter().zip(results.iter()).enumerate() {
        let cpu_gust = field.gust_envelope(q.position, q.time);
        let cpu_atten = field.height_attenuation(q.height);
        let cpu_vel = field.sample_velocity(q.position, q.time);
        let cpu_drag = WindField::drag_acceleration(q.wind_vel, q.particle_vel, q.drag_coeff);

        assert!(
            close(r.gust_envelope, cpu_gust),
            "gust mismatch at query {idx}: gpu {}, cpu {}",
            r.gust_envelope,
            cpu_gust
        );
        assert!(
            close(r.height_attenuation, cpu_atten),
            "attenuation mismatch at query {idx}: gpu {}, cpu {}",
            r.height_attenuation,
            cpu_atten
        );
        close_vec(r.velocity, cpu_vel, idx, "velocity");
        close_vec(r.drag, cpu_drag, idx, "drag");
        assert_eq!(
            r.gust_cell_hash,
            expected_hash(field, q),
            "raw gust-cell hash must be bit-identical at query {idx}"
        );
    }
    results
}

/// Asserts at least one query carries a non-trivial velocity, so a degenerate
/// all-zero kernel could not pass this scene.
fn assert_non_trivial(results: &[WindFieldResult]) {
    let any = results.iter().any(|r| r.velocity.length() > 1.0e-3);
    assert!(any, "scene should produce a non-zero velocity somewhere");
}

/// Builds a batch of queries spread across the signed lattice, including an
/// exact lattice origin and in-cell interiors, with exactly-representable
/// drag inputs and strictly positive attenuation altitudes.
fn sample_queries() -> Vec<WindFieldQuery> {
    vec![
        WindFieldQuery {
            position: Vec3::new(0.0, 0.0, 0.0),
            time: 0.0,
            height: 1.5,
            wind_vel: Vec3::new(4.0, 0.0, -2.0),
            particle_vel: Vec3::new(1.0, 1.0, 0.0),
            drag_coeff: 0.5,
        },
        WindFieldQuery {
            position: Vec3::new(0.37, 1.25, -2.5),
            time: 0.75,
            height: 5.0,
            wind_vel: Vec3::new(-3.0, 2.0, 1.0),
            particle_vel: Vec3::new(0.5, -0.5, 0.25),
            drag_coeff: 0.75,
        },
        WindFieldQuery {
            position: Vec3::new(-3.05, 0.92, 4.9),
            time: 2.5,
            height: 0.5,
            wind_vel: Vec3::new(2.0, -3.0, 1.0),
            particle_vel: Vec3::new(2.0, -3.0, 1.0),
            drag_coeff: 0.9,
        },
        WindFieldQuery {
            position: Vec3::new(12.75, 3.3, -8.4),
            time: 4.7,
            height: 7.25,
            wind_vel: Vec3::new(1.0, 1.0, 1.0),
            particle_vel: Vec3::new(0.0, 0.0, 0.0),
            drag_coeff: 0.125,
        },
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_across_fields() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping wind-field parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWindField::new(&ctx);
    let queries = sample_queries();

    // A spread of directions (each exactly normalized by `WindField::new`),
    // speeds, gust amplitudes/frequencies, falloffs and seeds.
    let fields = [
        WindField::new(Vec3::new(1.0, 0.0, 0.0), 3.0, 5.0, 1.3, 0.5, 1),
        WindField::new(Vec3::new(0.0, 0.0, 1.0), 6.0, 2.0, 0.25, 0.2, 99),
        WindField::new(Vec3::new(3.0, 4.0, 0.0), 2.0, 4.0, 2.0, 1.0, 0xDEAD_BEEF),
        WindField::new(Vec3::new(-1.0, 2.0, -2.0), 10.0, 1.0, 0.8, 0.05, 7),
    ];
    for field in fields {
        let results = check_queries(&ctx, &gpu, &field, &queries);
        assert_non_trivial(&results);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_large_random_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping wind-field parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWindField::new(&ctx);
    let field = WindField::new(Vec3::new(0.0, 1.0, 0.0), 5.0, 3.0, 0.8, 0.3, 0xABCD_1234);

    // Several thousand queries across many lattice cells and workgroups.
    let mut state = 0x5151_2727_9999_3333u64;
    let count = 4096usize;
    let mut queries = Vec::with_capacity(count);
    for _ in 0..count {
        queries.push(WindFieldQuery {
            position: Vec3::new(
                lcg(&mut state) * 16.0,
                lcg(&mut state) * 16.0,
                lcg(&mut state) * 16.0,
            ),
            time: lcg(&mut state) * 8.0,
            height: positive_height(&mut state),
            wind_vel: Vec3::new(
                lcg(&mut state) * 5.0,
                lcg(&mut state) * 5.0,
                lcg(&mut state) * 5.0,
            ),
            particle_vel: Vec3::new(
                lcg(&mut state) * 5.0,
                lcg(&mut state) * 5.0,
                lcg(&mut state) * 5.0,
            ),
            drag_coeff: (lcg(&mut state) + 1.0) * 0.5,
        });
    }
    let results = check_queries(&ctx, &gpu, &field, &queries);
    assert_non_trivial(&results);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gust_cell_hash_is_bit_identical() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping wind-field parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWindField::new(&ctx);
    let field = WindField::new(Vec3::new(1.0, 0.0, 0.0), 1.0, 1.0, 1.0, 0.0, 123);
    let queries = sample_queries();
    let results = gpu.eval(&ctx, &field, &queries);
    assert_eq!(results.len(), queries.len());
    for (idx, (q, r)) in queries.iter().zip(results.iter()).enumerate() {
        assert_eq!(
            r.gust_cell_hash,
            expected_hash(&field, q),
            "raw gust-cell hash must match the reference lattice hash at query {idx}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn zero_direction_and_zero_speed_vanish_on_device() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping wind-field parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWindField::new(&ctx);
    let queries = sample_queries();

    // A zero direction is preserved by `WindField::new` as the zero vector, so
    // the wind vanishes; a zero base speed and gust amplitude also vanish.
    for field in [
        WindField::new(Vec3::ZERO, 10.0, 4.0, 1.0, 0.5, 3),
        WindField::new(Vec3::new(1.0, 0.0, 0.0), 0.0, 0.0, 1.0, 0.5, 9),
    ] {
        let results = check_queries(&ctx, &gpu, &field, &queries);
        for (idx, r) in results.iter().enumerate() {
            assert!(
                r.velocity.length() < ABS_EPS,
                "velocity should vanish at query {idx}, got length {}",
                r.velocity.length()
            );
        }
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn distinct_seeds_produce_distinct_gusts_on_device() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping wind-field parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWindField::new(&ctx);
    let field_a = WindField::new(Vec3::new(1.0, 0.0, 0.0), 1.0, 1.0, 1.0, 0.0, 1);
    let field_b = WindField::new(Vec3::new(1.0, 0.0, 0.0), 1.0, 1.0, 1.0, 0.0, 2);
    let queries = sample_queries();

    let a = check_queries(&ctx, &gpu, &field_a, &queries);
    let b = check_queries(&ctx, &gpu, &field_b, &queries);

    // Both agree with their own CPU reference (checked above); the two seeds
    // must disagree somewhere, so the seed is not inert on device.
    let differs = a
        .iter()
        .zip(b.iter())
        .any(|(ra, rb)| (ra.gust_envelope - rb.gust_envelope).abs() > 1.0e-3);
    assert!(differs, "distinct seeds should produce distinct gusts");
}
