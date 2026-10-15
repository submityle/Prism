//! Offline camera contract benchmark.
//!
//! Run with:
//! `cargo bench -p prism_camera_rig --bench camera_contracts`
//!
//! Results are printed and written to
//! `target/benchmarks/prism_camera_rig_camera_contracts.csv` unless
//! `PRISM_CAMERA_BENCH_OUTPUT` overrides the path.

#![expect(
    clippy::print_stdout,
    reason = "the benchmark reports measurements to stdout"
)]

use std::env;
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use prism_camera::{derive_view, CameraIntent, CameraPose, ViewHistory, ViewLod};
use prism_camera_rig::{CameraBrain, Impulse, ViewUpdateScheduler, VirtualCamera};
use prism_math::{DVec3, Vec3};

const SAMPLES: usize = 10_000;

fn repetitions() -> usize {
    let args: Vec<String> = env::args().collect();
    args.windows(2)
        .find(|pair| pair[0] == "--iterations")
        .and_then(|pair| pair[1].parse().ok())
        .filter(|value: &usize| *value > 0)
        .unwrap_or(5)
}

fn best_ns_per_op<F>(repetitions: usize, mut operation: F) -> f64
where
    F: FnMut(),
{
    let mut best = f64::INFINITY;
    for _ in 0..repetitions {
        let start = Instant::now();
        operation();
        best = best.min(start.elapsed().as_nanos() as f64 / SAMPLES as f64);
    }
    best
}

fn main() {
    let repetitions = repetitions();
    let mut history = ViewHistory::default();
    let intent = CameraIntent::default();
    let derive_ns = best_ns_per_op(repetitions, || {
        for frame in 0..SAMPLES as u64 {
            let view = derive_view(
                black_box(intent),
                CameraPose {
                    position: DVec3::new(0.01 * frame as f64, 0.0, 0.0),
                    ..Default::default()
                },
                frame,
                [1920, 1080],
                &mut history,
            )
            .expect("benchmark camera pose must derive");
            black_box(view);
        }
    });

    let candidates = [
        VirtualCamera {
            key: 1,
            priority: 2,
            pose: CameraPose {
                position: DVec3::new(1.0, 2.0, 3.0),
                ..Default::default()
            },
            ..Default::default()
        },
        VirtualCamera {
            key: 2,
            priority: 1,
            ..Default::default()
        },
    ];
    let brain_ns = best_ns_per_op(repetitions, || {
        let mut brain = CameraBrain::default();
        brain.blend_duration_s = 0.25;
        for _ in 0..SAMPLES {
            black_box(brain.update(black_box(&candidates), 1.0 / 60.0, CameraPose::default()));
        }
    });

    let impulse_ns = best_ns_per_op(repetitions, || {
        let mut impulse = Impulse {
            translation: Vec3::new(0.1, 0.2, 0.3),
            rotation: Vec3::new(0.01, 0.02, 0.03),
            remaining_s: 1000.0,
        };
        let base = CameraPose::default();
        for _ in 0..SAMPLES {
            let (pose, next) = impulse.apply(base, 1.0 / 60.0);
            black_box(pose);
            impulse = next;
        }
    });

    let scheduler_ns = best_ns_per_op(repetitions, || {
        let mut scheduler = ViewUpdateScheduler::default();
        let policy = ViewLod::default();
        for frame in 0..SAMPLES as u64 {
            black_box(scheduler.should_update(0, policy, frame, 60.0));
        }
    });

    let rows = [
        ("derive_view", derive_ns),
        ("camera_brain_update", brain_ns),
        ("impulse_apply_6dof", impulse_ns),
        ("view_lod_scheduler", scheduler_ns),
    ];
    println!("case,iterations,samples,best_ns_per_op");
    let mut csv = String::from("case,iterations,samples,best_ns_per_op\n");
    for (name, ns) in rows {
        println!("{name},{repetitions},{SAMPLES},{ns:.3}");
        csv.push_str(&format!("{name},{repetitions},{SAMPLES},{ns:.3}\n"));
    }

    let output = env::var("PRISM_CAMERA_BENCH_OUTPUT")
        .unwrap_or_else(|_| "target/benchmarks/prism_camera_rig_camera_contracts.csv".to_owned());
    if let Some(parent) = Path::new(&output).parent() {
        fs::create_dir_all(parent).expect("create benchmark output directory");
    }
    fs::write(&output, csv).expect("write benchmark output");
    eprintln!("benchmark_output={output}");
}
