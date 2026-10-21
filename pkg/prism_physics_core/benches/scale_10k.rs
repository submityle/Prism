//! Reproducible ten-thousand-body latency benchmark.
//!
//! Run with `cargo bench -p prism_physics_core --bench scale_10k --features parallel`.
//! `PRISM_BENCH_SCENE=field|stacks|all`, `PRISM_BENCH_MODE=awake|sleep|all`,
//! `PRISM_BENCH_WARMUP`, `PRISM_BENCH_SAMPLES`, and `PRISM_BENCH_PARALLEL`
//! control the workload. Timing excludes setup, warmup, and validation.
#![expect(clippy::print_stdout, reason = "benchmark reports measurements")]

#[path = "support/latency.rs"]
mod latency;

use glam::Vec3;
use prism_physics_core::{
    BodyDesc, BodyKind, ColliderShape, PhysicsMaterial, PhysicsWorld, Solver, XpbdSolver,
};
use std::{process::Command, time::Instant};

const BODIES: usize = 10_000;
const DT: f32 = 1.0 / 60.0;
const SUBSTEPS: u32 = 4;
const BUDGET_MS: f64 = 1000.0 / 60.0;

fn build_world(stacks: bool, sleep: bool) -> PhysicsWorld {
    let mut world = PhysicsWorld::with_gravity(Vec3::new(0.0, -9.81, 0.0));
    world.config.sleep.enabled = sleep;
    let plane = world.shapes.insert(ColliderShape::Plane {
        normal: Vec3::Y,
        offset: 0.0,
    });
    world.spawn(BodyDesc::static_at(Vec3::ZERO).with_collider(plane));
    let shape = ColliderShape::Cuboid {
        half_extents: Vec3::splat(0.5),
    };
    let collider = world.shapes.insert(shape);
    let mass = shape.mass_properties(1.0);
    for index in 0..BODIES {
        let (x, z, layer) = if stacks {
            let column = index / 10;
            (column % 40, column / 40, index % 10)
        } else {
            (index % 100, index / 100, 0)
        };
        world.spawn(
            BodyDesc::dynamic_at(Vec3::new(
                x as f32 * 2.0,
                0.55 + layer as f32,
                z as f32 * 2.0,
            ))
            .with_collider(collider)
            .with_mass_properties(mass)
            .with_material(PhysicsMaterial::DEFAULT),
        );
    }
    world
}

fn env_count(name: &str, default: usize, minimum: usize) -> usize {
    match std::env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .ok()
            .filter(|v| (minimum..=100_000).contains(v))
            .unwrap_or_else(|| panic!("{name} must be an integer in {minimum}..=100000")),
        Err(std::env::VarError::NotPresent) => default,
        Err(error) => panic!("{name}: {error}"),
    }
}

fn selection(name: &str, default: &str, choices: &[&str]) -> String {
    let value = std::env::var(name).unwrap_or_else(|_| default.to_owned());
    assert!(
        choices.contains(&value.as_str()),
        "invalid {name}={value}; expected {choices:?}"
    );
    value
}

fn command_output(program: &str, args: &[&str]) -> String {
    Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|output| !output.is_empty())
        .unwrap_or_else(|| "unavailable".to_owned())
}

fn validate(world: &PhysicsWorld) -> usize {
    let mut awake = 0;
    let mut dynamic = 0;
    for slot in 0..world.bodies.slot_count() {
        let Some(body) = world.bodies.handle_at_slot(slot) else {
            continue;
        };
        if world.bodies.kind(body) != Some(BodyKind::Dynamic) {
            continue;
        }
        dynamic += 1;
        assert!(world.bodies.position(body).unwrap().is_finite());
        assert!(world.bodies.orientation(body).unwrap().is_finite());
        assert!(world.bodies.linear_velocity(body).unwrap().is_finite());
        assert!(world.bodies.angular_velocity(body).unwrap().is_finite());
        awake += usize::from(!world.bodies.is_sleeping(body).unwrap());
    }
    assert_eq!(dynamic, BODIES);
    awake
}

fn run(stacks: bool, sleep: bool, warmup: usize, samples: usize, parallel: bool) {
    let mut world = build_world(stacks, sleep);
    let mut solver = XpbdSolver::new();
    solver.config.parallel_islands = parallel;
    for _ in 0..warmup {
        solver.step(&mut world, DT, SUBSTEPS);
    }
    let awake_start = validate(&world);
    let mut timings = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        solver.step(&mut world, DT, SUBSTEPS);
        timings.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let awake_end = validate(&world);
    if !sleep {
        assert_eq!((awake_start, awake_end), (BODIES, BODIES));
    }
    let stats = latency::summarize(&timings, BUDGET_MS);
    println!(
        "scene={} sleep={} dynamic={} static=1 awake_start={} awake_end={} warmup={} samples={}",
        if stacks {
            "stacks_1000x10"
        } else {
            "independent_field_100x100"
        },
        sleep,
        BODIES,
        awake_start,
        awake_end,
        warmup,
        samples
    );
    println!(
        "  mean_ms={:.3} p50_ms={:.3} p95_ms={:.3} p99_ms={:.3} max_ms={:.3} over_16_667ms={}/{}",
        stats.mean, stats.p50, stats.p95, stats.p99, stats.max, stats.over_budget, samples
    );
}

fn main() {
    let scene = selection("PRISM_BENCH_SCENE", "field", &["field", "stacks", "all"]);
    let mode = selection("PRISM_BENCH_MODE", "all", &["awake", "sleep", "all"]);
    let parallel = selection("PRISM_BENCH_PARALLEL", "true", &["true", "false"]) == "true";
    let warmup = env_count("PRISM_BENCH_WARMUP", 120, 1);
    let samples = env_count("PRISM_BENCH_SAMPLES", 180, 100);
    println!("prism_physics_core scale_10k; nearest-rank latency; setup/validation excluded");
    println!(
        "  os={} arch={} cpu={} revision={}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        if cfg!(target_os = "macos") {
            command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
        } else {
            command_output("uname", &["-m"])
        },
        command_output("git", &["rev-parse", "HEAD"])
    );
    println!(
        "  dt={DT} substeps={SUBSTEPS} parallel_feature={} parallel_islands={} intra_island=false",
        cfg!(feature = "parallel"),
        parallel
    );
    #[cfg(feature = "parallel")]
    println!("  rayon_workers={}", rayon::current_num_threads());
    for stacks in [false, true] {
        if scene != "all" && stacks != (scene == "stacks") {
            continue;
        }
        for sleep in [false, true] {
            if mode != "all" && sleep != (mode == "sleep") {
                continue;
            }
            run(stacks, sleep, warmup, samples, parallel);
        }
    }
}
