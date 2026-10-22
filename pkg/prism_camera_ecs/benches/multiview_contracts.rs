//! Reproducible multi-view ECS publication benchmark.
//!
//! Run with:
//! `cargo bench -p prism_camera_ecs --bench multiview_contracts`
//!
//! The benchmark measures the steady-state camera publication path for 1, 8,
//! 16, and 64 active views. Warm-up frames are excluded from timing. Each
//! case reports p50/p95/p99 update latency, active-view counts, diagnostics,
//! registry size, and stable-handle/order guards. Results are also written to
//! `target/benchmarks/prism_camera_ecs_multiview_contracts.csv` unless
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

use prism_app::App;
use prism_camera::{
    derive_view, CameraIntent, CameraPose, RenderTarget, ViewHandle, ViewHistory, Viewport,
};
use prism_camera_ecs::{
    CameraBundle, CameraDiagnostics, CameraIntentComponent, CameraPlugin, CameraViewRegistry,
    PublishedViews,
};
use prism_tasks::TaskPool;

const VIEW_COUNTS: [usize; 4] = [1, 8, 16, 64];
const TARGET_SIZE: [u32; 2] = [4096, 2048];
const DEFAULT_WARMUP_FRAMES: usize = 64;
const DEFAULT_SAMPLES: usize = 128;
const DEFAULT_FRAMES_PER_SAMPLE: usize = 4;
const DERIVE_ITERATIONS_PER_SAMPLE: usize = 128;

#[derive(Clone, Copy, Debug)]
struct Config {
    warmup_frames: usize,
    samples: usize,
    frames_per_sample: usize,
}

#[derive(Clone, Copy, Debug)]
struct Measurement {
    view_count: usize,
    p50_ns_per_frame: u128,
    p95_ns_per_frame: u128,
    p99_ns_per_frame: u128,
    active_views: usize,
    diagnostics: usize,
    registry_views: usize,
    handles_stable: bool,
    order_stable: bool,
}

#[derive(Clone, Copy, Debug)]
struct DeriveMeasurement {
    view_count: usize,
    mode: &'static str,
    p50_ns_per_batch: u128,
    p95_ns_per_batch: u128,
    p99_ns_per_batch: u128,
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value: &usize| *value > 0)
        .unwrap_or(default)
}

fn config() -> Config {
    Config {
        warmup_frames: env_usize("PRISM_CAMERA_BENCH_WARMUP", DEFAULT_WARMUP_FRAMES),
        samples: env_usize("PRISM_CAMERA_BENCH_SAMPLES", DEFAULT_SAMPLES),
        frames_per_sample: env_usize(
            "PRISM_CAMERA_BENCH_FRAMES_PER_SAMPLE",
            DEFAULT_FRAMES_PER_SAMPLE,
        ),
    }
}

fn percentile(samples: &mut [u128], numerator: usize, denominator: usize) -> u128 {
    assert!(
        !samples.is_empty(),
        "percentile requires at least one sample"
    );
    samples.sort_unstable();
    let index = ((samples.len() - 1) * numerator).div_ceil(denominator);
    samples[index]
}

fn target_for(index: usize) -> RenderTarget {
    match index % 3 {
        0 => RenderTarget::PrimaryWindow,
        1 => RenderTarget::Window((index / 3 + 1) as u64),
        _ => RenderTarget::Texture((index / 3 + 1) as u64),
    }
}

fn build_app(view_count: usize) -> App {
    let mut app = App::new();
    app.add_plugins(CameraPlugin::new(TARGET_SIZE));

    let tile_width = TARGET_SIZE[0] / view_count as u32;
    for index in 0..view_count {
        let mut intent = CameraIntent {
            target: target_for(index),
            viewport: Viewport::from_pixels(
                [index as u32 * tile_width, 0],
                [tile_width, TARGET_SIZE[1]],
            ),
            // Deliberately use a deterministic non-entity order so the output
            // sort exercises the explicit order contract.
            order: ((index * 7) % view_count) as i32,
            ..CameraIntent::default()
        };
        intent.active = true;
        app.world_mut().spawn(CameraBundle {
            intent: CameraIntentComponent(intent),
            ..CameraBundle::default()
        });
    }
    app
}

fn ordered_and_stable(app: &App, handles: &mut Option<Vec<ViewHandle>>) -> (bool, bool) {
    let published = app.world().resource::<PublishedViews>();
    let current_handles: Vec<_> = published.iter().map(|view| view.data.handle).collect();
    let handles_stable = handles
        .as_ref()
        .is_none_or(|previous| previous == &current_handles);
    if handles.is_none() {
        *handles = Some(current_handles);
    }

    let order_stable = published
        .iter()
        .map(|view| (view.data.order, view.entity))
        .collect::<Vec<_>>()
        .windows(2)
        .all(|pair| pair[0] <= pair[1]);
    (handles_stable, order_stable)
}

fn measure(view_count: usize, config: Config) -> Measurement {
    let mut app = build_app(view_count);

    for _ in 0..config.warmup_frames {
        app.update();
    }

    let mut samples = Vec::with_capacity(config.samples);
    let mut handles = None;
    let mut handles_stable = true;
    let mut order_stable = true;
    let mut active_views = 0;
    let mut diagnostics = 0;
    let mut registry_views = 0;

    for _ in 0..config.samples {
        let start = Instant::now();
        for _ in 0..config.frames_per_sample {
            app.update();
        }
        samples.push(start.elapsed().as_nanos() / config.frames_per_sample as u128);

        let published = app.world().resource::<PublishedViews>();
        let current_diagnostics = app.world().resource::<CameraDiagnostics>();
        let registry = app.world().resource::<CameraViewRegistry>();
        active_views = published.len();
        diagnostics = current_diagnostics.errors().len();
        registry_views = registry.len();
        let (sample_handles_stable, sample_order_stable) = ordered_and_stable(&app, &mut handles);
        handles_stable &= sample_handles_stable;
        order_stable &= sample_order_stable;
        black_box((active_views, diagnostics, registry_views));
    }

    assert_eq!(
        active_views, view_count,
        "all configured views must publish"
    );
    assert_eq!(
        diagnostics, 0,
        "valid multi-view inputs must stay diagnostic-free"
    );
    assert_eq!(
        registry_views, view_count,
        "registry must retain every active view"
    );
    assert!(
        handles_stable,
        "view handles must remain stable after warm-up"
    );
    assert!(
        order_stable,
        "published views must stay sorted by order and entity"
    );

    Measurement {
        view_count,
        p50_ns_per_frame: percentile(&mut samples, 50, 100),
        p95_ns_per_frame: percentile(&mut samples, 95, 100),
        p99_ns_per_frame: percentile(&mut samples, 99, 100),
        active_views,
        diagnostics,
        registry_views,
        handles_stable,
        order_stable,
    }
}

fn measure_derive(view_count: usize, parallel: bool, config: Config) -> DeriveMeasurement {
    let mut inputs = Vec::with_capacity(view_count);
    let mut histories = vec![ViewHistory::default(); view_count];
    for index in 0..view_count {
        let mut intent = CameraIntent::default();
        intent.target = target_for(index);
        intent.viewport = Viewport::from_pixels(
            [index as u32 * (TARGET_SIZE[0] / view_count as u32), 0],
            [TARGET_SIZE[0] / view_count as u32, TARGET_SIZE[1]],
        );
        inputs.push((intent, CameraPose::default()));
    }
    let pool = (parallel && view_count >= 8).then(TaskPool::new);

    for iteration in 0..config.warmup_frames {
        derive_batch(&inputs, &mut histories, pool.as_ref(), iteration as u64);
    }

    let mut samples = Vec::with_capacity(config.samples);
    for sample in 0..config.samples {
        let start = Instant::now();
        for iteration in 0..DERIVE_ITERATIONS_PER_SAMPLE {
            derive_batch(
                &inputs,
                &mut histories,
                pool.as_ref(),
                (config.warmup_frames + sample * DERIVE_ITERATIONS_PER_SAMPLE + iteration) as u64,
            );
        }
        samples.push(start.elapsed().as_nanos() / DERIVE_ITERATIONS_PER_SAMPLE as u128);
    }

    DeriveMeasurement {
        view_count,
        mode: if pool.is_some() { "parallel" } else { "serial" },
        p50_ns_per_batch: percentile(&mut samples, 50, 100),
        p95_ns_per_batch: percentile(&mut samples, 95, 100),
        p99_ns_per_batch: percentile(&mut samples, 99, 100),
    }
}

fn derive_batch(
    inputs: &[(CameraIntent, CameraPose)],
    histories: &mut [ViewHistory],
    pool: Option<&TaskPool>,
    frame: u64,
) {
    if let Some(pool) = pool {
        pool.par_iter_mut(histories)
            .with_min_len(4)
            .enumerate_for_each(|index, history| {
                let (intent, pose) = inputs[index];
                black_box(derive_view(intent, pose, frame, TARGET_SIZE, history).unwrap());
            });
    } else {
        for ((intent, pose), history) in inputs.iter().zip(histories) {
            black_box(derive_view(*intent, *pose, frame, TARGET_SIZE, history).unwrap());
        }
    }
}

fn diagnostic_probe() -> usize {
    let mut app = App::new();
    app.add_plugins(CameraPlugin::new(TARGET_SIZE));
    app.world_mut().spawn(CameraBundle {
        intent: CameraIntentComponent(CameraIntent {
            viewport: Viewport::from_pixels([0, 0], [0, TARGET_SIZE[1]]),
            ..CameraIntent::default()
        }),
        ..CameraBundle::default()
    });
    app.update();
    app.world().resource::<CameraDiagnostics>().errors().len()
}

fn main() {
    let config = config();
    let derive_measurements = [
        measure_derive(1, false, config),
        measure_derive(16, false, config),
        measure_derive(16, true, config),
        measure_derive(64, false, config),
        measure_derive(64, true, config),
        measure_derive(256, false, config),
        measure_derive(256, true, config),
        measure_derive(1024, false, config),
        measure_derive(1024, true, config),
        measure_derive(4096, false, config),
        measure_derive(4096, true, config),
    ];
    let measurements: Vec<_> = VIEW_COUNTS
        .into_iter()
        .map(|view_count| measure(view_count, config))
        .collect();
    let diagnostic_errors = diagnostic_probe();

    println!("prism_camera_ecs multiview_contracts");
    println!("  target          : {}x{}", TARGET_SIZE[0], TARGET_SIZE[1]);
    println!("  warm-up frames  : {}", config.warmup_frames);
    println!("  samples         : {}", config.samples);
    println!("  frames/sample   : {}", config.frames_per_sample);
    println!("  diagnostic probe: {diagnostic_errors} invalid viewport error");
    println!("derive_mode,view_count,p50_ns/batch,p95_ns/batch,p99_ns/batch");
    for row in derive_measurements {
        println!(
            "{},{},{},{},{}",
            row.mode,
            row.view_count,
            row.p50_ns_per_batch,
            row.p95_ns_per_batch,
            row.p99_ns_per_batch
        );
    }
    println!("views,p50_ns/frame,p95_ns/frame,p99_ns/frame,active,diagnostics,registry,handles_stable,order_stable");
    for row in &measurements {
        println!(
            "{},{},{},{},{},{},{},{},{}",
            row.view_count,
            row.p50_ns_per_frame,
            row.p95_ns_per_frame,
            row.p99_ns_per_frame,
            row.active_views,
            row.diagnostics,
            row.registry_views,
            row.handles_stable,
            row.order_stable
        );
    }

    let output = env::var("PRISM_CAMERA_BENCH_OUTPUT").unwrap_or_else(|_| {
        "target/benchmarks/prism_camera_ecs_multiview_contracts.csv".to_owned()
    });
    if let Some(parent) = Path::new(&output).parent() {
        fs::create_dir_all(parent).expect("create benchmark output directory");
    }
    let mut csv = String::from(
        "view_count,target_width,target_height,warmup_frames,samples,frames_per_sample,p50_ns_per_frame,p95_ns_per_frame,p99_ns_per_frame,active_views,diagnostics,registry_views,handles_stable,order_stable\n",
    );
    for row in measurements {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            row.view_count,
            TARGET_SIZE[0],
            TARGET_SIZE[1],
            config.warmup_frames,
            config.samples,
            config.frames_per_sample,
            row.p50_ns_per_frame,
            row.p95_ns_per_frame,
            row.p99_ns_per_frame,
            row.active_views,
            row.diagnostics,
            row.registry_views,
            row.handles_stable,
            row.order_stable
        ));
    }
    fs::write(&output, csv).expect("write benchmark output");
    eprintln!("benchmark_output={output}");
}
