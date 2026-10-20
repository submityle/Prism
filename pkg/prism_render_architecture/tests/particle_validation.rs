//! Executable acceptance gates for the particle contracts.
//!
//! The normal tests exercise the real CPU particle APIs. The GPU parity and
//! image gates are intentionally `ignored`: they require a real GPU producer
//! or a real rendered image pair and fail closed when the required artifact is
//! absent. The large-particle entries are also explicit opt-in benchmarks so
//! ordinary unit-test runs do not allocate million-particle fixtures.

use std::fs;
use std::path::Path;
use std::time::Instant;

use prism_render_architecture::particle::dual_backend::FrameDigest;
use prism_render_architecture::particle::emitter::{Emitter, EmitterShape, SpawnParams};
use prism_render_architecture::particle::events::{AppendOutcome, ChannelRecord, EventRouter};
use prism_render_architecture::particle::gpu_compact::CompactConfig;
use prism_render_architecture::particle::oit::{FragmentList, OitFragment, OverflowPolicy};
use prism_render_architecture::particle::pool::{CapacityPolicy, ParticlePool};
use prism_render_architecture::particle::runtime::{EmitterRuntime, EmitterRuntimeConfig};
use prism_render_architecture::particle::simulation::{integrate, Kinematics};
use prism_render_architecture::particle::validation::{
    check_capacity, check_compaction, check_event_backpressure, check_free_list, check_oit_layers,
    compare_vector_samples, evaluate_golden, ConsistencyOutcome, ErrorTolerance, FreeListOp,
    GoldenTolerance,
};
use prism_render_architecture::particle::{IntegratorKind, Vec3};

fn emitter_runtime(capacity: u32, rate_per_second: f32) -> EmitterRuntime {
    let spawn = SpawnParams {
        speed: 1.25,
        ..SpawnParams::default()
    };
    let config = EmitterRuntimeConfig {
        capacity,
        rate_per_second,
        lifetime: 10.0,
        seed: 0x51_27_2026,
        spawn,
        ..EmitterRuntimeConfig::default()
    };
    EmitterRuntime::new(
        Emitter::new(
            prism_render_architecture::particle::EmitterHandle(7),
            EmitterShape::Point,
            spawn,
        ),
        config,
        Vec3::ZERO,
        Vec3::ZERO,
    )
    .expect("acceptance fixture must be valid")
}

fn digest(runtime: &EmitterRuntime) -> FrameDigest {
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    for (_, particle) in runtime.live_particles() {
        positions.push(particle.position);
        velocities.push(particle.velocity);
    }
    FrameDigest::summarize(&positions, &velocities)
}

#[test]
fn cpu_runtime_replay_is_deterministic() {
    let mut cpu = emitter_runtime(64, 120.0);
    let mut gpu_contract_input = cpu.clone();
    for _ in 0..8 {
        cpu.step(1.0 / 60.0);
        gpu_contract_input.step(1.0 / 60.0);
    }
    assert_eq!(digest(&cpu), digest(&gpu_contract_input));
}

#[test]
fn cpu_numeric_reference_matches_closed_form_semi_implicit_euler() {
    let initial = Kinematics::new(Vec3::new(1.0, -2.0, 0.5), Vec3::new(2.0, 0.0, -1.0));
    let acceleration = Vec3::new(0.0, -9.81, 0.0);
    let cpu = integrate(
        IntegratorKind::SemiImplicitEuler,
        initial,
        1.0 / 120.0,
        |_, _| acceleration,
    );
    let dt = 1.0 / 120.0;
    let expected_velocity = initial.velocity.add(acceleration.scale(dt));
    let expected_position = initial.position.add(expected_velocity.scale(dt));
    match compare_vector_samples(
        &[cpu.position, cpu.velocity],
        &[expected_position, expected_velocity],
        ErrorTolerance::new(1.0e-6, 1.0e-6),
    ) {
        ConsistencyOutcome::Compared(report) => assert!(report.consistent, "{report:?}"),
        ConsistencyOutcome::LengthMismatch { .. } => panic!("fixture lengths must match"),
    }
}

#[test]
fn capacity_race_backpressure_compaction_and_oit_overflow_are_real_paths() {
    let mut pool = ParticlePool::with_capacity(4);
    let slots: Vec<_> = (0..4)
        .map(|_| {
            pool.spawn(CapacityPolicy::DiscardNewest)
                .expect("capacity fixture")
        })
        .collect();
    assert_eq!(pool.spawn(CapacityPolicy::DiscardNewest), None);
    pool.kill(slots[1]);
    pool.kill(slots[1]);
    assert_eq!(pool.spawn(CapacityPolicy::DiscardNewest), Some(slots[1]));
    assert_eq!(pool.alive_count(), 4);
    let compacted = pool.compact();
    assert_eq!(compacted.len(), 4);
    assert_eq!(check_capacity(4, 5).expect("overflow").overflow, 1);
    assert_eq!(
        check_free_list(2, &[FreeListOp::Allocate(0), FreeListOp::Allocate(0)],)
            .expect("race violation")
            .step,
        1
    );

    let mut router = EventRouter::new();
    assert!(router.register_channel(11, 2));
    assert_eq!(
        router.publish(11, ChannelRecord::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0)),
        AppendOutcome::Accepted
    );
    assert_eq!(
        router.publish(11, ChannelRecord::new(Vec3::ZERO, Vec3::ZERO, 2.0, 0)),
        AppendOutcome::Accepted
    );
    assert_eq!(
        router.publish(11, ChannelRecord::new(Vec3::ZERO, Vec3::ZERO, 3.0, 0)),
        AppendOutcome::Dropped
    );
    assert_eq!(router.dropped_total(11), Some(1));
    let pressure = check_event_backpressure(2, &[2, 1], &[0, 1]);
    assert_eq!(pressure.total_dropped, 1);

    assert_eq!(
        check_compaction(&[true, false, true, true], &[0, 2, 3]),
        None
    );
    let mut fragments = FragmentList::new(2);
    assert!(fragments.insert(
        OitFragment::new(Vec3::new(1.0, 0.0, 0.0), 0.5, 5.0),
        OverflowPolicy::DropFarthest
    ));
    assert!(fragments.insert(
        OitFragment::new(Vec3::new(0.0, 1.0, 0.0), 0.5, 3.0),
        OverflowPolicy::DropFarthest
    ));
    assert!(fragments.insert(
        OitFragment::new(Vec3::new(0.0, 0.0, 1.0), 0.5, 1.0),
        OverflowPolicy::DropFarthest
    ));
    assert_eq!(fragments.len(), 2);
    let oit = check_oit_layers(&[2, 3], 2);
    assert_eq!(oit.overflowed_pixels, 1);
}

fn run_particle_benchmark(count: u32) {
    let start = Instant::now();
    let mut pool = ParticlePool::with_capacity(count);
    for _ in 0..count {
        assert!(pool.spawn(CapacityPolicy::DiscardNewest).is_some());
    }
    let flags: Vec<u32> = (0..count).map(|index| u32::from(index % 3 != 0)).collect();
    let compacted = CompactConfig::new(256).compact(&flags);
    assert_eq!(compacted.len(), (count - count.div_ceil(3)) as usize);
    eprintln!(
        "particle benchmark count={count} elapsed={:?} compacted={}",
        start.elapsed(),
        compacted.len()
    );
}

#[test]
#[ignore = "opt-in benchmark: run with cargo test --release --test particle_validation -- --ignored --nocapture"]
fn benchmark_100k_particles() {
    run_particle_benchmark(100_000);
}

#[test]
#[ignore = "opt-in benchmark: run with cargo test --release --test particle_validation -- --ignored --nocapture"]
fn benchmark_1m_particles() {
    run_particle_benchmark(1_000_000);
}

#[test]
#[ignore = "requires a physical GPU readback artifact from the GPU particle path"]
fn gpu_readback_parity_gate() {
    let path = std::env::var("PRISM_PARTICLE_GPU_READBACK")
        .expect("set PRISM_PARTICLE_GPU_READBACK to a real GPU CSV readback");
    let text = fs::read_to_string(path).expect("GPU readback artifact must be readable");
    let mut gpu = Vec::new();
    for (line, row) in text.lines().enumerate() {
        let fields: Vec<_> = row.split(',').map(str::trim).collect();
        assert_eq!(
            fields.len(),
            15,
            "GPU row {} must contain initial position, initial velocity, acceleration, final position, and final velocity",
            line + 1
        );
        let values: Vec<f32> = fields
            .iter()
            .map(|field| {
                field
                    .parse()
                    .expect("GPU CSV values must be finite numbers")
            })
            .collect();
        assert!(
            values.iter().all(|value| value.is_finite()),
            "GPU row {} contains a non-finite value",
            line + 1
        );
        gpu.push((
            Kinematics::new(
                Vec3::new(values[0], values[1], values[2]),
                Vec3::new(values[3], values[4], values[5]),
            ),
            Vec3::new(values[6], values[7], values[8]),
            Kinematics::new(
                Vec3::new(values[9], values[10], values[11]),
                Vec3::new(values[12], values[13], values[14]),
            ),
        ));
    }
    assert!(
        !gpu.is_empty(),
        "GPU readback must contain at least one particle"
    );
    let dt: f32 = std::env::var("PRISM_PARTICLE_GPU_DT")
        .expect("set PRISM_PARTICLE_GPU_DT to the exact simulation timestep")
        .parse()
        .expect("PRISM_PARTICLE_GPU_DT must be numeric");
    assert!(
        dt.is_finite() && dt > 0.0,
        "GPU parity timestep must be finite and positive"
    );
    let cpu: Vec<_> = gpu
        .iter()
        .map(|(initial, acceleration, _)| {
            integrate(IntegratorKind::SemiImplicitEuler, *initial, dt, |_, _| {
                *acceleration
            })
        })
        .collect();
    let cpu_positions: Vec<_> = cpu.iter().map(|state| state.position).collect();
    let gpu_positions: Vec<_> = gpu.iter().map(|(_, _, state)| state.position).collect();
    let cpu_velocities: Vec<_> = cpu.iter().map(|state| state.velocity).collect();
    let gpu_velocities: Vec<_> = gpu.iter().map(|(_, _, state)| state.velocity).collect();
    for (label, expected, actual) in [
        ("position", &cpu_positions, &gpu_positions),
        ("velocity", &cpu_velocities, &gpu_velocities),
    ] {
        match compare_vector_samples(expected, actual, ErrorTolerance::new(1.0e-4, 1.0e-4)) {
            ConsistencyOutcome::Compared(report) => {
                assert!(
                    report.consistent,
                    "GPU/CPU {label} parity failed: {report:?}"
                )
            }
            ConsistencyOutcome::LengthMismatch { .. } => {
                panic!("GPU readback length changed during comparison")
            }
        }
    }
}

fn read_ppm(path: &Path) -> (usize, usize, Vec<[u8; 3]>) {
    let bytes = fs::read(path).expect("PPM image must be readable");
    let mut cursor = 0;
    let token = |bytes: &[u8], cursor: &mut usize| -> String {
        while *cursor < bytes.len() && bytes[*cursor].is_ascii_whitespace() {
            *cursor += 1;
        }
        let start = *cursor;
        while *cursor < bytes.len() && !bytes[*cursor].is_ascii_whitespace() {
            *cursor += 1;
        }
        String::from_utf8(bytes[start..*cursor].to_vec()).expect("PPM header is ASCII")
    };
    let magic = token(&bytes, &mut cursor);
    let width: usize = token(&bytes, &mut cursor).parse().expect("PPM width");
    let height: usize = token(&bytes, &mut cursor).parse().expect("PPM height");
    let max: usize = token(&bytes, &mut cursor).parse().expect("PPM max value");
    assert_eq!(max, 255, "only 8-bit PPM is accepted");
    let pixels: Vec<[u8; 3]> = match magic.as_str() {
        "P6" => {
            while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
                cursor += 1;
            }
            bytes[cursor..]
                .chunks_exact(3)
                .map(|p| [p[0], p[1], p[2]])
                .collect()
        }
        "P3" => {
            let values: Vec<u8> = bytes[cursor..]
                .split(|byte| byte.is_ascii_whitespace())
                .filter(|p| !p.is_empty())
                .map(|p| {
                    String::from_utf8(p.to_vec())
                        .expect("P3 sample is ASCII")
                        .parse()
                        .expect("P3 sample is numeric")
                })
                .collect();
            values.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect()
        }
        _ => panic!("only P3/P6 PPM is accepted"),
    };
    assert_eq!(
        pixels.len(),
        width * height,
        "PPM dimensions do not match pixel data"
    );
    (width, height, pixels)
}

#[test]
#[ignore = "requires a real rendered image and checked-in golden PPM"]
fn golden_image_gate() {
    let actual =
        std::env::var("PRISM_PARTICLE_IMAGE").expect("set PRISM_PARTICLE_IMAGE to rendered PPM");
    let golden =
        std::env::var("PRISM_PARTICLE_GOLDEN").expect("set PRISM_PARTICLE_GOLDEN to golden PPM");
    let (aw, ah, a) = read_ppm(Path::new(&actual));
    let (gw, gh, g) = read_ppm(Path::new(&golden));
    assert_eq!(
        (aw, ah),
        (gw, gh),
        "golden and rendered images must have equal dimensions"
    );
    let errors: Vec<f32> = a
        .iter()
        .zip(g.iter())
        .map(|(actual, golden)| {
            let dr = (actual[0] as f32 - golden[0] as f32) / 255.0;
            let dg = (actual[1] as f32 - golden[1] as f32) / 255.0;
            let db = (actual[2] as f32 - golden[2] as f32) / 255.0;
            (dr * dr + dg * dg + db * db).sqrt()
        })
        .collect();
    let report = evaluate_golden(
        &errors,
        GoldenTolerance {
            per_pixel_epsilon: 2.0 / 255.0,
            max_failed_pixels: 0,
            mean_error_limit: 0.5 / 255.0,
        },
    );
    assert!(
        report.passed,
        "particle golden image regression failed: {report:?}"
    );
}
