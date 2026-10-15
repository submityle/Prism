//! Resident compute benchmark; no rendering and no readback in timed samples.
//! Times are host wall clocks, not hardware timestamp queries or frame times.

use std::time::Instant;

use prism_world_gpu::{
    FrustumPlane, GpuContext, GpuResidentVisibility, ProxyInstance, ProxySoa, VisibilityInput,
};

const WARMUP: usize = 8;
const SAMPLES: usize = 64;

fn camera() -> VisibilityInput {
    VisibilityInput::from_projection_scale(
        [0.0; 3],
        [
            FrustumPlane::new([1.0, 0.0, 0.0], 64.0),
            FrustumPlane::new([-1.0, 0.0, 0.0], 64.0),
            FrustumPlane::new([0.0, 1.0, 0.0], 64.0),
            FrustumPlane::new([0.0, -1.0, 0.0], 64.0),
            FrustumPlane::new([0.0, 0.0, 1.0], 64.0),
            FrustumPlane::new([0.0, 0.0, -1.0], 64.0),
        ],
        1080.0,
        512.0,
    )
}

fn proxy(slot: usize) -> ProxyInstance {
    let x = if slot.is_multiple_of(4) {
        200.0
    } else {
        (slot % 61) as f32 - 30.0
    };
    ProxyInstance::new(
        slot as u32,
        (slot / 64) as u32,
        [x, ((slot / 61) % 61) as f32 - 30.0, 20.0],
        0.5,
        0.25 + (slot % 17) as f32,
        1024.0,
    )
}

fn percentiles(mut values: Vec<f64>) -> [f64; 3] {
    values.sort_by(f64::total_cmp);
    [50_usize, 95, 99].map(|p| values[(values.len() * p).div_ceil(100) - 1])
}

fn verify(ctx: &GpuContext, resident: &GpuResidentVisibility, input: VisibilityInput) -> usize {
    let count = resident.dense_count() as usize;
    let size = 16 + count as u64 * 4;
    let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("benchmark verification only"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(resident.indirect_args(), 0, &staging, 0, 16);
    encoder.copy_buffer_to_buffer(resident.compact_indices(), 0, &staging, 16, size - 16);
    ctx.queue().submit([encoder.finish()]);
    let (send, recv) = std::sync::mpsc::channel();
    staging.map_async(wgpu::MapMode::Read, .., move |result| {
        send.send(result).expect("verification receiver");
    });
    ctx.wait();
    recv.recv().expect("map callback").expect("map success");
    let mapped = staging.get_mapped_range(..).expect("mapped bytes");
    let word =
        |index: usize| u32::from_le_bytes(mapped[index * 4..index * 4 + 4].try_into().unwrap());
    assert_eq!([word(0), word(2), word(3)], [3, 0, 0]);
    let visible = word(1) as usize;
    assert!(visible <= count);
    let mut seen = vec![false; count];
    for i in 0..visible {
        let slot = word(4 + i) as usize;
        assert!(slot < count, "compact index out of range");
        assert!(!seen[slot], "duplicate compact index");
        seen[slot] = true;
    }
    for (slot, actual) in seen.into_iter().enumerate() {
        assert_eq!(actual, input.evaluate(&proxy(slot)).visible, "slot {slot}");
    }
    drop(mapped);
    staging.unmap();
    visible
}

fn run(ctx: &GpuContext, count: usize) {
    let input = camera();
    let proxies: Vec<_> = (0..count).map(proxy).collect();
    let soa = ProxySoa::from_proxies(&proxies).expect("valid benchmark proxies");
    let mut resident = GpuResidentVisibility::new(ctx, count, 3).expect("capacity supported");
    resident.write_range(0, &soa).expect("initial upload");
    resident.set_dense_count(count).expect("initialized prefix");
    resident.write_camera(input).expect("camera upload");
    let mut record_finish = Vec::with_capacity(SAMPLES);
    let mut submit_wait = Vec::with_capacity(SAMPLES);
    for sample in 0..WARMUP + SAMPLES {
        let mut encoder = ctx.device().create_command_encoder(&Default::default());
        let record_start = Instant::now();
        resident
            .encode(input, &mut encoder)
            .expect("resident encode");
        let commands = encoder.finish();
        let recorded = record_start.elapsed().as_secs_f64() * 1_000_000.0;
        let submit_start = Instant::now();
        ctx.queue().submit([commands]);
        ctx.wait();
        let completed = submit_start.elapsed().as_secs_f64() * 1_000_000.0;
        if sample >= WARMUP {
            record_finish.push(recorded);
            submit_wait.push(completed);
        }
    }
    let visible = verify(ctx, &resident, input);
    let record = percentiles(record_finish);
    let submit = percentiles(submit_wait);
    assert_eq!(resident.allocated_buffer_bytes(), 72 * count as u64 + 144);
    println!(
        "{count},{visible},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
        resident.allocated_buffer_bytes(),
        48 * count as u64 + 128,
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
        .expect("real GPU acquisition must succeed")
        .expect("GPU benchmark may not skip");
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
    println!("adapter={evidence:?}; warmup={WARMUP}; samples={SAMPLES}; nearest-rank percentiles");
    println!("host wall clocks in us; record_finish excludes encoder creation; submit_wait includes submission and device polling; no timestamp queries");
    println!("one initial SoA/camera upload; zero timed uploads/readbacks/draws; logical bytes exclude driver/staging/CPU memory");
    println!("count,visible,logical_buffer_bytes,initial_upload_bytes,record_finish_p50_us,record_finish_p95_us,record_finish_p99_us,submit_wait_p50_us,submit_wait_p95_us,submit_wait_p99_us");
    for count in [10_000, 100_000, 1_000_000] {
        run(&ctx, count);
    }
}
