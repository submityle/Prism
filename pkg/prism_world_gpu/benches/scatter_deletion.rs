//! Bounded full-chain GPU deletion timings, with complete validation per sample.

#[path = "../tests/support/scatter_deletion.rs"]
mod support;

use std::time::Instant;

use prism_world_gpu::{
    GpuContext, GpuScatter, GpuScatterDeletionCapacity, GpuScatterDeletionWorkBudget,
    GpuScatterDeletions, GpuScatterSourceRequest,
};
use support::{config, context, deletions, region, source, wait, Capture};

const WARMUP: usize = 8;
const SAMPLES: usize = 64;

fn percentiles(mut values: Vec<f64>) -> [f64; 3] {
    values.sort_by(f64::total_cmp);
    [50_usize, 95, 99].map(|p| values[(values.len() * p).div_ceil(100) - 1])
}

fn run(ctx: &GpuContext, n: usize, delete_percent: usize) {
    let cfg = config();
    let bounds = region(
        [-((n / 2) as i64) * 17, 0, 0],
        [(n - n / 2) as i64 * 17, 17, 17],
    );
    let points = source(cfg, bounds);
    assert_eq!(points.len(), n);
    let deleted = deletions(
        points
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 100 < delete_percent)
            .map(|(_, p)| p.key)
            .collect(),
    );
    let expected: Vec<_> = points
        .iter()
        .copied()
        .filter(|p| !deleted.is_deleted(p.key))
        .collect();
    let d = deleted.keys().len();
    let mut gpu = GpuScatterDeletions::new(
        ctx,
        GpuScatter::new(ctx, n, 3, u64::MAX).unwrap(),
        GpuScatterDeletionCapacity {
            max_keys: d,
            max_buffer_bytes: u64::MAX,
            max_scratch_bytes: d * 52,
        },
    )
    .unwrap();
    gpu.write_request(
        &GpuScatterSourceRequest {
            config: cfg,
            region: bounds,
            max_candidate_evaluations: n as u64,
        },
        &deleted,
        GpuScatterDeletionWorkBudget {
            max_input_slots: n as u64,
            max_keys: d,
            max_key_comparisons: n as u64 * u64::from(usize::BITS - d.leading_zeros()),
        },
    )
    .unwrap();
    let mut encode = Vec::with_capacity(SAMPLES);
    let mut complete = Vec::with_capacity(SAMPLES);
    for sample in 0..WARMUP + SAMPLES {
        let mut encoder = ctx.device().create_command_encoder(&Default::default());
        let start = Instant::now();
        gpu.encode(&mut encoder).unwrap();
        let commands = encoder.finish();
        let encoded = start.elapsed().as_secs_f64() * 1e6;
        ctx.queue().submit([commands]);
        wait(ctx);
        let completed = start.elapsed().as_secs_f64() * 1e6;
        if sample >= WARMUP {
            encode.push(encoded);
            complete.push(completed);
        }
        // Fresh readback and full CPU key/position/compact parity AFTER timing.
        // No readback, upload, clone, source generation or validation in samples.
        let mut verify = ctx.device().create_command_encoder(&Default::default());
        let snapshot = Capture::new(
            ctx,
            &mut verify,
            gpu.candidate_count(),
            gpu.records(),
            gpu.attributes(),
            gpu.compact_indices(),
            gpu.indirect_args(),
        );
        ctx.queue().submit([verify.finish()]);
        snapshot.verify(ctx, cfg, &expected, None, 3);
    }
    let a = percentiles(encode);
    let b = percentiles(complete);
    println!(
        "{n},{d},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
        expected.len(),
        gpu.allocated_buffer_bytes(),
        gpu.deletion_buffer_bytes(),
        gpu.key_scratch_bytes(),
        a[0],
        a[1],
        a[2],
        b[0],
        b[1],
        b[2]
    );
}

fn main() {
    let ctx = context();
    println!("# 8 warmup,64 samples; nearest-rank indices32/61/64; microseconds");
    println!("# encode=encode+finish; complete=encode+finish+submit+bounded wait; whole source+deletion chain");
    println!(
        "# not GPU timestamps,frame latency,physical VRAM or RSS; full parity outside every timing"
    );
    println!("n,d,remaining,total_logical_gpu_bytes,deletion_logical_gpu_bytes,key_scratch_bytes,encode_p50_us,encode_p95_us,encode_p99_us,complete_p50_us,complete_p95_us,complete_p99_us");
    for n in [10_000, 100_000] {
        for delete_percent in [0, 25, 100] {
            run(&ctx, n, delete_percent);
        }
    }
}
