//! Opt-in CPU scale benchmark for the virtual-geometry per-view planning path.
//!
//! Run with `cargo bench -p prism_render_architecture --bench mesh_cluster_scale`
//! and optional `--clusters`, `--samples`, `--warmup`, and `--page-span` flags.
//! This measures CPU frustum/cut selection, page-request coalescing, and raster
//! binning through `plan_frame`; it does not submit work to, or measure, a GPU.
#![expect(
    clippy::print_stdout,
    reason = "the standalone benchmark reports its configuration and measurements"
)]

use std::error::Error;
use std::hint::black_box;
use std::time::{Duration, Instant};

use prism_render_architecture::gpu_scene::SceneBounds;
use prism_render_architecture::virtual_geometry::{
    plan_frame, ClusterHierarchy, ClusterNode, ClusterRasterStats, FrameView, Frustum,
    GeometryPageKey, LodProjection, Plane, RasterCapability, RasterConfig,
};

const DEFAULT_CLUSTERS: usize = 1_000_000;
const DEFAULT_SAMPLES: usize = 100;
const DEFAULT_WARMUP: usize = 5;
const DEFAULT_PAGE_SPAN: usize = 64;

#[derive(Clone, Copy, Debug)]
struct Config {
    clusters: usize,
    samples: usize,
    warmup: usize,
    page_span: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args()?;
    config
        .warmup
        .checked_add(config.samples)
        .ok_or("warmup and sample counts overflow")?;
    let frustum = benchmark_frustum();
    let (hierarchy, raster_stats) = fixtures(config)?;
    let view = FrameView {
        origin: [0.0, 0.0, 0.0],
        frustum: &frustum,
        projection: LodProjection::from_focal_length_pixels(1080.0),
        target_error_pixels: 1.0,
    };
    let raster = RasterConfig {
        stats: &raster_stats,
        capability: RasterCapability {
            mesh_shader: true,
            hardware_indirect: true,
        },
        software_pixel_threshold: 16.0,
    };

    let mut timings = Vec::with_capacity(config.samples);
    let mut observed = PlanCounts::default();
    for sample in 0..(config.warmup + config.samples) {
        let start = Instant::now();
        let plan = plan_frame(&hierarchy, view, raster);
        let elapsed = start.elapsed();
        let counts = PlanCounts::from_plan(&plan);
        black_box(&plan);
        drop(plan);

        if sample >= config.warmup {
            observed = counts;
            timings.push(elapsed);
        }
    }

    timings.sort_unstable();
    let culled = config.clusters.saturating_sub(observed.cut_clusters);
    println!("Mesh virtual-geometry CPU planning benchmark");
    println!("scope=CPU only; no GPU work submitted or measured");
    println!(
        "input_clusters={} hierarchy_nodes={} roots={} raster_stats={}",
        config.clusters,
        hierarchy.nodes().len(),
        hierarchy.roots().len(),
        raster_stats.len()
    );
    println!(
        "samples={} warmup={} page_span={} visible_cut={} frustum_rejected={} raster_binned={} page_requests={}",
        timings.len(), config.warmup, config.page_span, observed.cut_clusters,
        culled, observed.binned_clusters, observed.page_requests
    );
    println!(
        "plan_frame_cpu_ms p50={:.3} p95={:.3} p99={:.3} min={:.3} max={:.3}",
        percentile_ms(&timings, 50),
        percentile_ms(&timings, 95),
        percentile_ms(&timings, 99),
        duration_ms(timings[0]),
        duration_ms(*timings.last().expect("at least one measured sample"))
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, Default)]
struct PlanCounts {
    cut_clusters: usize,
    binned_clusters: usize,
    page_requests: usize,
}

impl PlanCounts {
    fn from_plan(plan: &prism_render_architecture::virtual_geometry::GeometryFramePlan) -> Self {
        Self {
            cut_clusters: plan.cut.len(),
            binned_clusters: plan.bins.total(),
            page_requests: plan.page_requests.len(),
        }
    }
}

fn parse_args() -> Result<Config, Box<dyn Error>> {
    let mut config = Config {
        clusters: DEFAULT_CLUSTERS,
        samples: DEFAULT_SAMPLES,
        warmup: DEFAULT_WARMUP,
        page_span: DEFAULT_PAGE_SPAN,
    };
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        // Cargo's custom-bench runner appends this marker after user arguments.
        if argument == "--bench" {
            continue;
        }
        if argument == "--help" || argument == "-h" {
            println!("Usage: mesh_cluster_scale [--clusters N] [--samples N] [--warmup N] [--page-span N]");
            println!("Defaults: --clusters 1000000 --samples 100 --warmup 5 --page-span 64");
            println!("All timings are CPU-only plan_frame measurements; no GPU work is submitted.");
            std::process::exit(0);
        }
        let (name, inline_value) = argument
            .split_once('=')
            .map_or((argument.as_str(), None), |(name, value)| {
                (name, Some(value))
            });
        let value = match inline_value {
            Some(value) => value.to_owned(),
            None => args
                .next()
                .ok_or_else(|| format!("missing value for {name}"))?,
        };
        let parsed = value
            .parse::<usize>()
            .map_err(|error| format!("invalid value for {name}: {error}"))?;
        match name {
            "--clusters" if parsed > 0 && parsed <= u32::MAX as usize => {
                config.clusters = parsed;
            }
            "--samples" if parsed > 0 => config.samples = parsed,
            "--warmup" => config.warmup = parsed,
            "--page-span" if parsed > 0 && parsed <= u32::MAX as usize => {
                config.page_span = parsed;
            }
            "--clusters" => return Err("--clusters must be in 1..=u32::MAX".into()),
            "--samples" => return Err("--samples must be greater than zero".into()),
            "--page-span" => return Err("--page-span must be in 1..=u32::MAX".into()),
            _ => return Err(format!("unknown option or invalid value: {name}").into()),
        }
    }
    Ok(config)
}

fn benchmark_frustum() -> Frustum {
    Frustum::from_planes([
        Plane::new([1.0, 0.0, 0.0], 50.0),
        Plane::new([-1.0, 0.0, 0.0], 50.0),
        Plane::new([0.0, 1.0, 0.0], 50.0),
        Plane::new([0.0, -1.0, 0.0], 50.0),
        Plane::new([0.0, 0.0, 1.0], 0.0),
        Plane::new([0.0, 0.0, -1.0], 500.0),
    ])
}

fn fixtures(config: Config) -> Result<(ClusterHierarchy, Vec<ClusterRasterStats>), Box<dyn Error>> {
    let mut nodes = Vec::with_capacity(config.clusters);
    let mut roots = Vec::with_capacity(config.clusters);
    let mut raster_stats = Vec::with_capacity(config.clusters);
    for index in 0..config.clusters {
        let index_u32 = u32::try_from(index)?;
        let x = centered_sample(index, 0x9e37_79b1, 2001);
        let y = centered_sample(index, 0x85eb_ca77, 2001);
        let z = 1.0
            + (((index as u64)
                .wrapping_mul(1_103_515_245)
                .wrapping_add(12_345)
                % 100_000) as f32
                / 100.0);
        let bounds = SceneBounds {
            center: [x, y, z],
            radius: 0.75,
            half_extents: [0.5, 0.5, 0.5],
            _padding: 0.0,
        };
        let page = u32::try_from(index / config.page_span)?;
        nodes.push(ClusterNode::leaf(
            bounds,
            0.0,
            GeometryPageKey::new(1, page),
        ));
        roots.push(index_u32);
        raster_stats.push(ClusterRasterStats {
            max_triangle_pixels: if index & 3 == 0 { 4.0 } else { 64.0 },
            triangle_count: 64,
        });
    }

    let hierarchy = ClusterHierarchy::new(nodes, roots);
    Ok((hierarchy, raster_stats))
}

fn centered_sample(index: usize, multiplier: usize, modulus: usize) -> f32 {
    let sample = (index as u64).wrapping_mul(multiplier as u64) % modulus as u64;
    (sample as f32 - (modulus / 2) as f32) * 0.1
}

fn percentile_ms(samples: &[Duration], percentile: usize) -> f64 {
    let rank = (samples.len() * percentile).div_ceil(100).max(1);
    duration_ms(samples[rank.min(samples.len()) - 1])
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}
