//! Physical-device benchmark for the complete planetary placement pipeline.
//!
//! Request preparation and height upload occur before timing. Encode and
//! submit-plus-wait are measured independently; a final readback validates the
//! indirect result and remains outside the timed windows.

#![cfg(feature = "terrain_scatter")]

use std::{convert::Infallible, sync::Arc, time::Instant};

use prism_world_gpu::{GpuContext, GpuPlanetPlacement, GpuPlanetPlacementWorkBudget};
use prism_world_terrain::{
    BakeLimits, CubeFace, HeightSample, HeightSampleValue, HeightSampler, HeightfieldSpec,
    PlanetPlacementConfig, PlanetPlacementRegion, TileKey,
};

const WARMUP: usize = 8;
const SAMPLES: usize = 64;
const SIDE: u32 = 256;

struct Flat;

impl HeightSampler for Flat {
    type Error = Infallible;

    fn sample_height(&self, _: HeightSample) -> Result<HeightSampleValue, Self::Error> {
        Ok(HeightSampleValue::from_millimetres(0))
    }
}

fn percentile(mut values: Vec<f64>) -> [f64; 3] {
    values.sort_by(f64::total_cmp);
    [50_usize, 95, 99].map(|p| values[(values.len() * p).div_ceil(100) - 1])
}

fn read_indirect(ctx: &GpuContext, gpu: &GpuPlanetPlacement) -> [u32; 4] {
    let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("planet placement benchmark verification"),
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

fn run(ctx: &GpuContext, candidates: usize) {
    let cells_per_axis = (candidates as f64).sqrt() as u32;
    assert_eq!(
        cells_per_axis as usize * cells_per_axis as usize,
        candidates
    );
    let tile = Arc::new(
        prism_world_terrain::HeightfieldTile::bake(
            TileKey::root(CubeFace::PositiveZ),
            HeightfieldSpec::new(SIDE, 1).expect("valid heightfield dimensions"),
            &Flat,
            BakeLimits {
                max_samples: SIDE as usize * SIDE as usize,
                max_bytes: SIDE as usize * SIDE as usize * 2,
            },
        )
        .expect("flat heightfield bake"),
    );
    let mut gpu = GpuPlanetPlacement::new(ctx, candidates, tile.samples().len(), 3, u64::MAX)
        .expect("planet placement GPU allocation");
    gpu.write_request(
        PlanetPlacementConfig {
            seed: 0x1357_9bdf_2468_ace0,
            version: 1,
            salt: 17,
            radius_mm: 6_371_000_000,
            origin_mm: [0; 3],
            vertical_offset_mm: 0,
            acceptance_threshold: u16::MAX,
        },
        PlanetPlacementRegion::new(tile.tile(), [0, 0], [cells_per_axis, cells_per_axis])
            .expect("valid placement region"),
        &tile,
        GpuPlanetPlacementWorkBudget {
            max_candidate_evaluations: candidates as u64,
            max_sample_reads: candidates as u64 * 4,
        },
    )
    .expect("bounded placement upload");

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
    assert_eq!(indirect[1], candidates as u32);
    let encode = percentile(encode_us);
    let submit = percentile(submit_us);
    println!(
        "{candidates},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
        gpu.allocated_buffer_bytes(),
        gpu.candidate_count(),
        encode[0],
        encode[1],
        encode[2],
        submit[0],
        submit[1],
        submit[2]
    );
}

fn main() {
    let ctx = GpuContext::try_headless_with_limits(wgpu::Limits {
        max_storage_buffers_per_shader_stage: 6,
        max_bindings_per_bind_group: 6,
        ..wgpu::Limits::downlevel_defaults()
    })
    .expect("physical GPU acquisition")
    .expect("physical GPU required");
    let evidence = ctx.evidence();
    assert!(matches!(
        evidence.device_type.as_str(),
        "IntegratedGpu" | "DiscreteGpu"
    ));
    println!("adapter={evidence:?}; warmup={WARMUP}; samples={SAMPLES}; host microseconds; upload/readback excluded from timed windows");
    println!("candidates,logical_buffer_bytes,reported_candidates,encode_p50_us,encode_p95_us,encode_p99_us,submit_p50_us,submit_p95_us,submit_p99_us");
    for candidates in [256, 4_096, 62_500] {
        run(&ctx, candidates);
    }
}
