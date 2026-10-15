//! Bounded integer scatter generation, excluding uploads/readback/draw in samples.

use std::time::Instant;

use prism_world_gpu::{GpuContext, GpuScatter};
use prism_world_residency::{scatter, ScatterBudget, ScatterConfig, ScatterPoint, ScatterRegion};

const WARMUP: usize = 8;
const SAMPLES: usize = 64;

fn percentiles(mut values: Vec<f64>) -> [f64; 3] {
    values.sort_by(f64::total_cmp);
    [50_usize, 95, 99].map(|p| values[(values.len() * p).div_ceil(100) - 1])
}

fn verify(
    ctx: &GpuContext,
    gpu: &GpuScatter,
    config: ScatterConfig,
    region: ScatterRegion,
) -> usize {
    let count = gpu.candidate_count() as usize;
    let bytes = 16 + count as u64 * 68;
    let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("scatter benchmark verification only"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(gpu.indirect_args(), 0, &staging, 0, 16);
    encoder.copy_buffer_to_buffer(gpu.records(), 0, &staging, 16, count as u64 * 64);
    encoder.copy_buffer_to_buffer(
        gpu.compact_indices(),
        0,
        &staging,
        16 + count as u64 * 64,
        count as u64 * 4,
    );
    ctx.queue().submit([encoder.finish()]);
    let (send, recv) = std::sync::mpsc::channel();
    staging.map_async(wgpu::MapMode::Read, .., move |result| {
        send.send(result).unwrap();
    });
    ctx.wait();
    recv.recv().unwrap().unwrap();
    let mapped = staging.get_mapped_range(..).unwrap();
    let word = |i: usize| u32::from_le_bytes(mapped[i * 4..i * 4 + 4].try_into().unwrap());
    assert_eq!([word(0), word(2), word(3)], [3, 0, 0]);
    let accepted = word(1) as usize;
    assert!(accepted <= count);
    let mut seen = vec![false; count];
    for i in 0..accepted {
        let slot = word(4 + count * 16 + i) as usize;
        assert!(slot < count && !seen[slot], "compact slot uniqueness/range");
        seen[slot] = true;
    }
    let cpu = scatter(
        config,
        region,
        ScatterBudget {
            max_candidate_evaluations: u64::MAX,
            max_output_points: count,
            max_output_bytes: count * size_of::<ScatterPoint>(),
        },
    )
    .unwrap();
    assert_eq!(accepted, cpu.points.len());
    let mut expected = cpu.points.iter();
    for (slot, compacted) in seen.into_iter().enumerate() {
        let start = 4 + slot * 16;
        let valid = word(start + 12);
        assert!(valid <= 1);
        assert_eq!(compacted, valid == 1);
        if valid == 0 {
            continue;
        }
        let point = expected.next().unwrap();
        let pair = |index| {
            (u64::from(word(start + index)) | (u64::from(word(start + index + 1)) << 32)) as i64
        };
        assert_eq!(
            [pair(0), pair(2), pair(4)],
            [point.key.grid.x, point.key.grid.y, point.key.grid.z]
        );
        assert_eq!([pair(6), pair(8), pair(10)], point.position_mm);
    }
    assert!(expected.next().is_none());
    drop(mapped);
    staging.unmap();
    accepted
}

fn run(ctx: &GpuContext, count: usize, radius: u32) {
    let config = ScatterConfig {
        seed: 0x1234_5678_9abc_def0,
        version: 7,
        salt: 91,
        grid_spacing_mm: 17,
        min_distance_mm: radius,
    };
    // Exactly count primary voxels, around negative and positive X coordinates.
    let region = ScatterRegion::new(
        [-((count / 2) as i64) * 17, 0, 0],
        [(count - count / 2) as i64 * 17, 17, 17],
    )
    .unwrap();
    let mut gpu = GpuScatter::new(ctx, count, 3, u64::MAX).unwrap();
    gpu.write_request(config, region, u64::MAX).unwrap();
    assert_eq!(gpu.candidate_count() as usize, count);
    let mut record_times = Vec::with_capacity(SAMPLES);
    let mut submit_times = Vec::with_capacity(SAMPLES);
    for sample in 0..WARMUP + SAMPLES {
        let mut encoder = ctx.device().create_command_encoder(&Default::default());
        let start = Instant::now();
        gpu.encode(&mut encoder).unwrap();
        let commands = encoder.finish();
        let recorded = start.elapsed().as_secs_f64() * 1e6;
        let start = Instant::now();
        ctx.queue().submit([commands]);
        ctx.wait();
        let submitted = start.elapsed().as_secs_f64() * 1e6;
        if sample >= WARMUP {
            record_times.push(recorded);
            submit_times.push(submitted);
        }
    }
    let accepted = verify(ctx, &gpu, config, region);
    let record = percentiles(record_times);
    let submit = percentiles(submit_times);
    println!(
        "{count},{radius},{accepted},{},192,{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
        gpu.allocated_buffer_bytes(),
        record[0],
        record[1],
        record[2],
        submit[0],
        submit[1],
        submit[2]
    );
}

fn main() {
    let ctx = GpuContext::try_headless()
        .expect("GPU acquisition")
        .expect("GPU required");
    let evidence = ctx.evidence();
    assert!(matches!(
        evidence.device_type.as_str(),
        "IntegratedGpu" | "DiscreteGpu"
    ));
    if let Ok(backend) = std::env::var("WGPU_BACKEND") {
        assert_eq!(
            evidence.backend.to_ascii_lowercase(),
            backend.to_ascii_lowercase()
        );
    }
    println!("adapter={evidence:?}; warmup={WARMUP}; samples={SAMPLES}; nearest-rank P50/P95/P99; host microseconds, not GPU timestamps");
    println!("one initial 192B upload; no timed upload/readback/draw; shader hashes all candidates anew each dispatch; CPU exact set/positions verified after timing");
    println!("fixture=1D strip of 3D jittered voxels; radius17 tests 3D halo rejection including unrequested neighbors; not surface scatter or a product frame");
    println!("primary,radius_mm,accepted,logical_buffer_bytes,initial_uniform_bytes,record_finish_p50_us,record_finish_p95_us,record_finish_p99_us,submit_wait_p50_us,submit_wait_p95_us,submit_wait_p99_us");
    for count in [10_000, 100_000, 1_000_000] {
        run(&ctx, count, 0);
    }
    run(&ctx, 10_000, 17);
}
