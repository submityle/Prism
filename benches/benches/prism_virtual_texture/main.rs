use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use prism_render_architecture::texture_streaming::{
    decode_feedback, FeedbackTextureDesc, StreamerConfig, TextureSemantic, VirtualTextureStreamer,
    NOT_REQUESTED,
};
use prism_virtual_texture_gpu::GpuContext;
use std::hint::black_box;

fn desc(size: usize) -> FeedbackTextureDesc {
    let side = (size as f64).sqrt() as u16;
    FeedbackTextureDesc {
        texture: 7,
        layer: 0,
        semantic: TextureSemantic::Color,
        base_mip: 0,
        mip_count: 6,
        pages_x: side,
        pages_y: side,
        page_byte_cost: 65_536,
        screen_importance: 500,
    }
}

fn grid(size: usize) -> Vec<u8> {
    (0..size)
        .map(|index| {
            if index % 7 == 0 {
                NOT_REQUESTED
            } else {
                (index % 6) as u8
            }
        })
        .collect()
}

fn cpu_feedback_and_residency(c: &mut Criterion) {
    let mut group = c.benchmark_group("prism_virtual_texture/cpu");
    for size in [64usize, 1024, 16_384] {
        let desc = desc(size);
        let feedback = grid(desc.grid_len());
        group.bench_with_input(BenchmarkId::new("decode", size), &(), |b, &()| {
            b.iter(|| black_box(decode_feedback(&desc, &feedback, |_| None, 1)))
        });
        group.bench_with_input(BenchmarkId::new("stream", size), &(), |b, &()| {
            b.iter(|| {
                let mut streamer = VirtualTextureStreamer::new(
                    StreamerConfig::new(64 * 1024 * 1024).with_min_demand_frames(1),
                    256,
                );
                let demands = decode_feedback(&desc, &feedback, |_| None, 1);
                black_box(streamer.stream_demands(&demands))
            })
        });
    }
    group.finish();
}

fn gpu_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("prism_virtual_texture/gpu");
    let Some(ctx) = GpuContext::try_real_headless() else {
        eprintln!(
            "[prism_virtual_texture] no real integrated/discrete adapter; GPU benchmarks skipped"
        );
        group.finish();
        return;
    };
    group.bench_function("adapter_gate", |b| {
        b.iter(|| black_box(ctx.adapter_evidence()))
    });
    group.finish();
}

criterion_group!(benches, cpu_feedback_and_residency, gpu_gate);
criterion_main!(benches);
