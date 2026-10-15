use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use prism_render_architecture::texture_streaming::{
    decode_feedback, FeedbackTextureDesc, GpuPageTable, StreamerConfig, TexturePageKey,
    TextureSemantic, VirtualTextureStreamer, NOT_REQUESTED,
};
use prism_virtual_texture_gpu::{GpuContext, GpuFeedbackDecode, GpuPageLookup};
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

fn gpu_lookup_fixture() -> (GpuPageTable, Vec<TexturePageKey>) {
    let bindings = (0..1024u32)
        .map(|slot| {
            (
                TexturePageKey {
                    texture: slot / 256,
                    mip: ((slot / 64) % 4) as u8,
                    layer: ((slot / 32) % 2) as u16,
                    x: (slot % 32) as u16,
                    y: 0,
                },
                slot,
            )
        })
        .collect::<Vec<_>>();
    let table = GpuPageTable::from_bindings(&bindings);
    let queries = bindings
        .iter()
        .map(|&(key, _)| key)
        .chain((0..1024u32).map(|index| TexturePageKey {
            texture: 99,
            mip: (index % 6) as u8,
            layer: 0,
            x: (index % 32) as u16,
            y: ((index / 32) % 32) as u16,
        }))
        .collect();
    (table, queries)
}

fn cpu_feedback_and_residency(c: &mut Criterion) {
    let mut group = c.benchmark_group("prism_virtual_texture/cpu");
    for size in [64usize, 1024, 16_384] {
        let desc = desc(size);
        let feedback = grid(desc.grid_len());
        group.bench_with_input(BenchmarkId::new("decode", size), &(), |b, &()| {
            b.iter(|| black_box(decode_feedback(&desc, &feedback, |_| None, 1)));
        });
        group.bench_with_input(BenchmarkId::new("stream", size), &(), |b, &()| {
            b.iter(|| {
                let mut streamer = VirtualTextureStreamer::new(
                    StreamerConfig::new(64 * 1024 * 1024).with_min_demand_frames(1),
                    256,
                );
                let demands = decode_feedback(&desc, &feedback, |_| None, 1);
                black_box(streamer.stream_demands(&demands))
            });
        });
    }
    group.finish();
}

fn gpu_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("prism_virtual_texture/gpu");
    let Some(ctx) = GpuContext::try_real_headless() else {
        eprintln!("[prism_virtual_texture] SKIPPED: no real integrated/discrete adapter");
        group.finish();
        return;
    };
    eprintln!(
        "[prism_virtual_texture] GPU benchmark adapter: {:?}",
        ctx.adapter_evidence()
    );

    let (table, queries) = gpu_lookup_fixture();
    let lookup = GpuPageLookup::new(&ctx);
    group.bench_function("page_lookup_readback", |b| {
        b.iter(|| black_box(lookup.lookup(&ctx, &table, &queries).expect("GPU lookup")));
    });

    let feedback_desc = desc(16_384);
    let feedback = grid(feedback_desc.grid_len());
    let decode = GpuFeedbackDecode::new(&ctx);
    group.bench_function("feedback_decode_readback", |b| {
        b.iter(|| {
            black_box(
                decode
                    .decode(&ctx, &feedback_desc, &feedback, |_| None, 1)
                    .expect("GPU feedback decode"),
            )
        });
    });
    group.finish();
}

criterion_group!(benches, cpu_feedback_and_residency, gpu_kernels);
criterion_main!(benches);
