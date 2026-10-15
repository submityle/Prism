//! Physical-device benchmark for the complete heightfield placement pipeline.
//!
//! The timed section records command encoding and queue completion separately.
//! Uploads and readback are outside the measured window; a final indirect-buffer
//! readback still proves that the dispatch produced a bounded result.

#![cfg(feature = "terrain_scatter")]

use std::{convert::Infallible, sync::Arc, time::Instant};

use prism_world_gpu::{GpuContext, GpuHeightfieldPlacement, GpuHeightfieldPlacementWorkBudget};
use prism_world_residency::{ScatterDensityField, ScatterFieldBuildBudget, ScatterFieldConfig};
use prism_world_terrain::{
    estimate_heightfield_placement_work, BakeLimits, CubeFace, HeightSample, HeightSampleValue,
    HeightSampler, HeightfieldPlacementConfig, HeightfieldPlacementFilters,
    HeightfieldPlacementRegion, HeightfieldSpec, HeightfieldTile, PlanarHeightfieldEmbedding,
    PlanarHeightfieldSurface, TileKey,
};

const WARMUP: usize = 8;
const SAMPLES: usize = 64;
const SURFACE_SIDE: usize = 1_025;
const SPACING_MM: u32 = 1_000;

struct Flat;

impl HeightSampler for Flat {
    type Error = Infallible;

    fn sample_height(&self, _sample: HeightSample) -> Result<HeightSampleValue, Self::Error> {
        Ok(HeightSampleValue::from_millimetres(0))
    }
}

fn surface() -> PlanarHeightfieldSurface {
    let count = SURFACE_SIDE * SURFACE_SIDE;
    let tile = HeightfieldTile::bake(
        TileKey::root(CubeFace::PositiveY),
        HeightfieldSpec::new(SURFACE_SIDE as u32, 1).expect("valid surface dimensions"),
        &Flat,
        BakeLimits {
            max_samples: count,
            max_bytes: count * 2,
        },
    )
    .expect("flat heightfield bake");
    PlanarHeightfieldSurface::new(
        Arc::new(tile),
        PlanarHeightfieldEmbedding {
            sample_origin_mm: [-512_000, 0, -512_000],
            sample_spacing_mm: SPACING_MM,
        },
    )
    .expect("valid planar surface")
}

fn field() -> ScatterDensityField {
    ScatterDensityField::new(
        ScatterFieldConfig {
            seed: 0x1357_9bdf,
            version: 1,
            base_density: u16::MAX,
            base_species: 3,
            noise: None,
        },
        Vec::new(),
        ScatterFieldBuildBudget {
            max_brushes: 0,
            max_brush_bytes: 0,
        },
    )
    .expect("complete field recipe")
}

fn config(radius: u32) -> HeightfieldPlacementConfig {
    HeightfieldPlacementConfig {
        seed: 0x0123_4567_89ab_cdef,
        version: 1,
        salt: 17,
        grid_spacing_mm: SPACING_MM,
        min_distance_mm: radius,
        vertical_offset_mm: 0,
    }
}

fn filters() -> HeightfieldPlacementFilters {
    HeightfieldPlacementFilters {
        surface_height_mm: [i64::MIN, i64::MAX],
        max_slope_rise: u32::MAX,
        max_slope_run: 1,
    }
}

fn percentile(mut values: Vec<f64>) -> [f64; 3] {
    values.sort_by(f64::total_cmp);
    [50_usize, 95, 99].map(|p| values[(values.len() * p).div_ceil(100) - 1])
}

fn read_indirect(ctx: &GpuContext, gpu: &GpuHeightfieldPlacement) -> [u32; 4] {
    let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("heightfield placement benchmark verification"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(gpu.indirect_args(), 0, &staging, 0, 16);
    ctx.queue().submit([encoder.finish()]);
    let (send, recv) = std::sync::mpsc::channel();
    staging.map_async(wgpu::MapMode::Read, .., move |result| {
        send.send(result).expect("map callback receiver");
    });
    ctx.wait();
    recv.recv()
        .expect("map callback")
        .expect("indirect readback");
    let mapped = staging
        .get_mapped_range(..)
        .expect("mapped indirect buffer");
    let mut result = [0; 4];
    for (out, bytes) in result.iter_mut().zip(mapped.chunks_exact(4)) {
        *out = u32::from_le_bytes(bytes.try_into().expect("four-byte word"));
    }
    drop(mapped);
    staging.unmap();
    result
}

fn run(ctx: &GpuContext, primary: usize, radius: u32) {
    let config = config(radius);
    let (x_cells, z_cells) = match primary {
        10_000 => (100_i64, 100_i64),
        100_000 => (1_000_i64, 100_i64),
        _ => panic!("unsupported benchmark primary count {primary}"),
    };
    assert_eq!((x_cells * z_cells) as usize, primary);
    let region = HeightfieldPlacementRegion::new(
        [-512_000, -512_000],
        [
            -512_000 + x_cells * i64::from(SPACING_MM),
            -512_000 + z_cells * i64::from(SPACING_MM),
        ],
    )
    .expect("valid region");
    let field = field();
    let surface = surface();
    let filters = filters();
    let work = estimate_heightfield_placement_work(config, region, &field)
        .expect("bounded placement work");
    assert_eq!(work.primary_candidates as usize, primary);
    let budget = GpuHeightfieldPlacementWorkBudget {
        max_candidate_evaluations: work.max_candidate_evaluations,
        max_sample_reads: work.max_sample_reads,
        max_field_work: work.max_field_work,
    };
    let mut gpu =
        GpuHeightfieldPlacement::new(ctx, primary, 0, surface.tile().samples().len(), 3, u64::MAX)
            .expect("placement GPU allocation");
    gpu.write_request(config, region, &field, &surface, filters, budget)
        .expect("placement upload");

    let mut encode_us = Vec::with_capacity(SAMPLES);
    let mut submit_us = Vec::with_capacity(SAMPLES);
    for sample in 0..WARMUP + SAMPLES {
        let mut encoder = ctx.device().create_command_encoder(&Default::default());
        let start = Instant::now();
        gpu.encode(&mut encoder).expect("placement encode");
        let encode = start.elapsed().as_secs_f64() * 1e6;
        let start = Instant::now();
        ctx.queue().submit([encoder.finish()]);
        ctx.wait();
        let submit = start.elapsed().as_secs_f64() * 1e6;
        if sample >= WARMUP {
            encode_us.push(encode);
            submit_us.push(submit);
        }
    }
    let indirect = read_indirect(ctx, &gpu);
    assert_eq!(indirect[0], 3);
    assert!(indirect[1] <= primary as u32);
    let e = percentile(encode_us);
    let s = percentile(submit_us);
    println!(
        "{primary},{radius},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
        indirect[1],
        gpu.allocated_buffer_bytes(),
        gpu.candidate_count(),
        e[0],
        e[1],
        e[2],
        s[0],
        s[1],
        s[2]
    );
}

fn main() {
    let ctx = GpuContext::try_headless_with_limits(wgpu::Limits {
        max_storage_buffers_per_shader_stage: 8,
        ..wgpu::Limits::downlevel_defaults()
    })
    .expect("physical GPU acquisition")
    .expect("physical GPU required");
    let evidence = ctx.evidence();
    assert!(matches!(
        evidence.device_type.as_str(),
        "IntegratedGpu" | "DiscreteGpu"
    ));
    println!("adapter={evidence:?}; warmup={WARMUP}; samples={SAMPLES}; host microseconds; uploads/readback excluded from timed windows");
    println!("primary,radius_mm,accepted,logical_buffer_bytes,candidates,encode_p50_us,encode_p95_us,encode_p99_us,submit_p50_us,submit_p95_us,submit_p99_us");
    for primary in [10_000, 100_000] {
        for radius in [0, 150] {
            run(&ctx, primary, radius);
        }
    }
}
