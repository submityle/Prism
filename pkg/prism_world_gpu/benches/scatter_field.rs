//! Bounded physical-GPU source scatter + author-field benchmark.
//!
//! Both timed windows exclude uploads, readback, CPU verification and draw.
//! These are host-wall-clock measurements, not GPU timestamps or product frame
//! times. Every dispatch is checked against a complete, precomputed CPU oracle
//! outside the timed windows. Register with harness=false and required gpu.

#![cfg(feature = "gpu")]

use std::mem::size_of;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use prism_world_gpu::{GpuContext, GpuScatterField};
use prism_world_residency::{
    pcg_hash_u32, scatter, DensityBrush, DensityBrushOp, DensityNoise, ScatterBudget,
    ScatterCandidateKey, ScatterConfig, ScatterDensityField, ScatterFieldBuildBudget,
    ScatterFieldConfig, ScatterPoint, ScatterRegion, SCATTER_KERNEL_VERSION,
};

const WARMUP: usize = 8;
const SAMPLES: usize = 64;
const WAIT_TIMEOUT: Duration = Duration::from_secs(30);
const VERTEX_COUNT: u32 = 3;
const SPACING_MM: u32 = 50;
const SOURCE_UNIFORM_BYTES: u64 = 192;
const FIELD_UNIFORM_BYTES: u64 = 128;
const BRUSH_RECORD_BYTES: u64 = 64;
// Two constructor-initialized 16-byte indirect buffers, source and field.
const INITIAL_INDIRECT_BYTES: u64 = 32;
const DIGEST_INITIAL: u64 = 0xcbf2_9ce4_8422_2325;

#[derive(Clone, Copy)]
struct Fixture {
    count: usize,
    shape: [usize; 3],
}

impl Fixture {
    fn region(self) -> ScatterRegion {
        assert_eq!(self.shape.into_iter().product::<usize>(), self.count);
        assert!(self.shape.into_iter().all(|n| n > 1));
        assert!(self.count <= 100_000);
        let spacing = i64::from(SPACING_MM);
        let min = self.shape.map(|n| -(n as i64 / 2) * spacing);
        let max = std::array::from_fn(|axis| min[axis] + self.shape[axis] as i64 * spacing);
        ScatterRegion::new(min, max).expect("aligned, bounded three-dimensional fixture")
    }
}

#[derive(Clone, Copy)]
struct Workload {
    label: &'static str,
    noise: bool,
    warp: bool,
    brush_count: usize,
}

const WORKLOADS: [Workload; 4] = [
    Workload {
        label: "constant",
        noise: false,
        warp: false,
        brush_count: 0,
    },
    Workload {
        label: "noise",
        noise: true,
        warp: false,
        brush_count: 0,
    },
    Workload {
        label: "warp",
        noise: true,
        warp: true,
        brush_count: 0,
    },
    Workload {
        label: "warp+32-overlapping-brushes",
        noise: true,
        warp: true,
        brush_count: 32,
    },
];

fn scatter_config() -> ScatterConfig {
    ScatterConfig {
        seed: 0x0123_4567_89ab_cdef,
        version: 1,
        salt: 22,
        grid_spacing_mm: SPACING_MM,
        // This measures source generation + field filtering, not halo work.
        // Aligned full voxels make every one of the N source slots accepted.
        min_distance_mm: 0,
    }
}

fn field(workload: Workload) -> ScatterDensityField {
    assert!(workload.brush_count <= 32);
    assert!(!workload.warp || workload.noise);
    // All 32 spheres have the same centre and cover both complete 3D fixtures.
    // A 4000mm radius keeps every source point in every brush's outer sphere;
    // small inner radii exercise squared radial falloff, not full-weight only.
    // Reverse input IDs deliberately: the CPU recipe owns canonical sorting.
    let brushes: Vec<_> = (0..workload.brush_count)
        .rev()
        .map(|id| DensityBrush {
            id: id as u64,
            center_mm: [0; 3],
            inner_radius_mm: 50 + id as u32,
            outer_radius_mm: 4_000,
            opacity: 40_000,
            // Include replacement as well as both additive signs. All edits
            // stay in the validated UNORM16 / signed-delta ranges.
            density: if id % 8 == 0 {
                DensityBrushOp::Replace(12_000 + id as u16 * 1_000)
            } else {
                DensityBrushOp::Add(if id % 2 == 0 { 1_000 } else { -1_000 })
            },
            species: Some(id as u32 + 1),
        })
        .collect();
    let brush_capacity_bytes = brushes.capacity() * size_of::<DensityBrush>();
    ScatterDensityField::new(
        ScatterFieldConfig {
            seed: 0xfedc_ba98_7654_3210,
            version: 1,
            base_density: 32_768,
            base_species: 7,
            noise: workload.noise.then_some(DensityNoise {
                cell_size_mm: 311,
                amplitude: 24_000,
                warp_cell_size_mm: 127,
                warp_amplitude_mm: if workload.warp { 201 } else { 0 },
            }),
        },
        brushes,
        ScatterFieldBuildBudget {
            max_brushes: workload.brush_count,
            max_brush_bytes: brush_capacity_bytes,
        },
    )
    .expect("bounded validated author-field recipe")
}

fn percentiles(mut values: [f64; SAMPLES]) -> [f64; 3] {
    assert!(values
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0));
    values.sort_by(f64::total_cmp);
    [50_usize, 95, 99].map(|p| values[(SAMPLES * p).div_ceil(100) - 1])
}

fn wait_bounded(ctx: &GpuContext) {
    // Native backends wait for the latest submission and run its callbacks.
    // Fail on timeout instead of retrying or turning a bad sample into a skip.
    ctx.device()
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(WAIT_TIMEOUT),
        })
        .expect("GPU submission must complete within 30 seconds");
}

// The source record priority is not the field rank. Reproduce the CPU source
// key's ordered mix explicitly so all 16 source words, not just XYZ, are checked.
fn mix(value: u64) -> u64 {
    (u64::from(pcg_hash_u32(value)) << 32) | u64::from(pcg_hash_u32(value ^ 0x9e37_79b9_7f4a_7c15))
}

fn priority(key: ScatterCandidateKey) -> u64 {
    let mut hash = mix(0x7363_6174_7465_7200 ^ u64::from(key.kernel_version));
    for value in [
        key.seed,
        u64::from(key.version),
        key.salt,
        u64::from(key.grid_spacing_mm),
        key.grid.x as u64,
        key.grid.y as u64,
        key.grid.z as u64,
    ] {
        hash = mix(hash ^ value);
    }
    mix(hash ^ 0x082e_fa98_ec4e_6c89)
}

fn put_pair(words: &mut [u32], index: usize, value: u64) {
    words[index] = value as u32;
    words[index + 1] = (value >> 32) as u32;
}

// Fixed-order noncryptographic u32-word recurrence, NOT a byte/native-endian
// Hash implementation or cache identity. Compact reservation order is excluded.
fn digest_word(digest: &mut u64, word: u32) {
    *digest = (*digest ^ u64::from(word)).wrapping_mul(0x0000_0100_0000_01b3);
}

fn digest_header(accepted: usize) -> u64 {
    let mut digest = DIGEST_INITIAL;
    for word in [VERTEX_COUNT, u32::try_from(accepted).unwrap(), 0, 0] {
        digest_word(&mut digest, word);
    }
    digest
}

struct Oracle {
    // Exact source record (16 words) followed by field attributes (4 words).
    slots: Vec<[u32; 20]>,
    accepted: usize,
    digest: u64,
}

fn oracle(
    fixture: Fixture,
    config: ScatterConfig,
    region: ScatterRegion,
    field: &ScatterDensityField,
) -> Oracle {
    let source = scatter(
        config,
        region,
        ScatterBudget {
            max_candidate_evaluations: fixture.count as u64,
            max_output_points: fixture.count,
            max_output_bytes: fixture.count * size_of::<ScatterPoint>(),
        },
    )
    .expect("bounded CPU source oracle");
    assert_eq!(
        source.points.len(),
        fixture.count,
        "all aligned source voxels survive"
    );
    let min_grid = region.min_mm.map(|p| p.div_euclid(i64::from(SPACING_MM)));
    let mut slots = Vec::with_capacity(fixture.count);
    let mut accepted = 0;
    for (slot, point) in source.points.into_iter().enumerate() {
        let grid = [point.key.grid.x, point.key.grid.y, point.key.grid.z];
        assert_eq!(
            grid,
            [
                min_grid[0] + (slot / (fixture.shape[1] * fixture.shape[2])) as i64,
                min_grid[1] + (slot / fixture.shape[2] % fixture.shape[1]) as i64,
                min_grid[2] + (slot % fixture.shape[2]) as i64,
            ],
            "complete 3D source-slot enumeration, Z fastest"
        );
        assert_eq!(point.key.kernel_version, SCATTER_KERNEL_VERSION);
        for brush in field.brushes() {
            let distance: i128 = point
                .position_mm
                .into_iter()
                .zip(brush.center_mm)
                .map(|(p, center)| (i128::from(p) - i128::from(center)).pow(2))
                .sum();
            assert!(
                distance < i128::from(brush.outer_radius_mm).pow(2),
                "every benchmark source point must probe every overlapping brush"
            );
        }
        let sample = field.sample(point.position_mm);
        let rank = field.candidate_rank(point.key);
        let keep = field.accepts(point.key, sample.density);
        let mut words = [0; 20];
        for (axis, (coordinate, position)) in grid.into_iter().zip(point.position_mm).enumerate() {
            put_pair(&mut words, axis * 2, coordinate as u64);
            put_pair(&mut words, 6 + axis * 2, position as u64);
        }
        words[12] = 1;
        put_pair(&mut words, 13, priority(point.key));
        words[16..20].copy_from_slice(&[
            u32::from(sample.density),
            sample.species,
            u32::from(keep),
            rank,
        ]);
        accepted += usize::from(keep);
        slots.push(words);
    }
    assert!(
        accepted > 0 && accepted < fixture.count,
        "nontrivial field thinning"
    );
    let mut digest = digest_header(accepted);
    for (slot, words) in slots.iter().enumerate() {
        digest_word(&mut digest, slot as u32);
        for &word in words {
            digest_word(&mut digest, word);
        }
    }
    Oracle {
        slots,
        accepted,
        digest,
    }
}

fn read_word(bytes: &[u8], index: usize) -> u32 {
    u32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap())
}

// One persistent validation staging buffer and seen-set are reused outside
// the timed windows. The unused compact suffix is copied but never inspected.
fn verify(
    ctx: &GpuContext,
    gpu: &GpuScatterField,
    staging: &wgpu::Buffer,
    expected: &Oracle,
    seen: &mut [bool],
) -> u64 {
    let count = expected.slots.len();
    assert_eq!(gpu.candidate_count() as usize, count);
    assert_eq!(seen.len(), count);
    let n = count as u64;
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(gpu.indirect_args(), 0, staging, 0, 16);
    encoder.copy_buffer_to_buffer(gpu.records(), 0, staging, 16, n * 64);
    encoder.copy_buffer_to_buffer(gpu.attributes(), 0, staging, 16 + n * 64, n * 16);
    encoder.copy_buffer_to_buffer(gpu.compact_indices(), 0, staging, 16 + n * 80, n * 4);
    ctx.queue().submit([encoder.finish()]);
    let (send, recv) = mpsc::sync_channel(1);
    staging.map_async(wgpu::MapMode::Read, .., move |result| {
        // Never panic in the device callback if verification already unwound.
        let _ = send.send(result);
    });
    wait_bounded(ctx);
    recv.recv_timeout(WAIT_TIMEOUT)
        .expect("bounded verification map callback wait")
        .expect("verification buffer mapping");
    let mapped = staging
        .get_mapped_range(..)
        .expect("verification mapped range");
    assert_eq!(mapped.len(), 16 + 84 * count);
    let indirect: [u32; 4] = std::array::from_fn(|word| read_word(&mapped, word));
    assert_eq!(
        indirect,
        [VERTEX_COUNT, expected.accepted as u32, 0, 0],
        "complete field indirect arguments and CPU accepted count"
    );
    let accepted = indirect[1] as usize;
    assert!(accepted <= count);
    seen.fill(false);
    for compact_index in 0..accepted {
        let slot = read_word(&mapped, 4 + count * 20 + compact_index) as usize;
        assert!(slot < count, "compact source slot outside primary prefix");
        assert!(!seen[slot], "duplicate compact source slot {slot}");
        seen[slot] = true;
    }
    let mut digest = digest_header(accepted);
    let mut accepted_flags = 0;
    for (slot, words) in expected.slots.iter().enumerate() {
        let record: [u32; 16] =
            std::array::from_fn(|word| read_word(&mapped, 4 + slot * 16 + word));
        let attrs: [u32; 4] =
            std::array::from_fn(|word| read_word(&mapped, 4 + count * 16 + slot * 4 + word));
        assert_eq!(
            record.as_slice(),
            &words[..16],
            "full CPU source record at slot {slot}"
        );
        assert_eq!(
            attrs.as_slice(),
            &words[16..],
            "CPU density/species/accept/rank at slot {slot}"
        );
        assert_eq!(
            seen[slot],
            attrs[2] == 1,
            "compact membership at source slot {slot}"
        );
        accepted_flags += usize::from(attrs[2] == 1);
        digest_word(&mut digest, slot as u32);
        for word in record.into_iter().chain(attrs) {
            digest_word(&mut digest, word);
        }
    }
    assert_eq!(
        accepted_flags, accepted,
        "field flags, compact prefix and indirect agree"
    );
    assert_eq!(
        digest, expected.digest,
        "complete numeric digest equals the CPU oracle"
    );
    drop(mapped);
    staging.unmap();
    digest
}

fn run(ctx: &GpuContext, fixture: Fixture, workload: Workload) {
    let config = scatter_config();
    let region = fixture.region();
    let field = field(workload);
    let expected = oracle(fixture, config, region, &field);
    let brush_count = field.brushes().len();
    // Source 68*N+208, field 20*N+128+16, brush storage at least 64B.
    let gpu_bytes =
        88 * fixture.count as u64 + BRUSH_RECORD_BYTES * brush_count.max(1) as u64 + 352;
    let mut gpu = GpuScatterField::new(ctx, fixture.count, brush_count, VERTEX_COUNT, gpu_bytes)
        .expect("bounded GPU source + author-field resources");
    gpu.write_request(
        config,
        region,
        fixture.count as u64,
        &field,
        fixture.count as u64 * field.work_per_sample(),
    )
    .expect("one initial source/field/brush upload");
    assert_eq!(gpu.candidate_count() as usize, fixture.count);
    assert_eq!(gpu.allocated_buffer_bytes(), gpu_bytes);
    let scratch_bytes = gpu.brush_scratch_bytes();
    let request_upload_bytes =
        SOURCE_UNIFORM_BYTES + FIELD_UNIFORM_BYTES + BRUSH_RECORD_BYTES * brush_count as u64;
    let initial_upload_bytes = request_upload_bytes + INITIAL_INDIRECT_BYTES;
    let verification_bytes = 16 + 84 * fixture.count as u64;
    let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("scatter field benchmark untimed verification only"),
        size: verification_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut seen = vec![false; fixture.count];

    // Flush initial queue writes BEFORE even the first warmup measurement.
    // No request/recipe updates or queue.write_buffer calls occur in the loop.
    ctx.queue().submit([]);
    wait_bounded(ctx);
    println!("# primary={} shape={:?} region={region:?} scatter={config:?} field={:?} brushes={brush_count}",
        fixture.count, fixture.shape, field.config());
    let mut encode_times = [0.0; SAMPLES];
    let mut submit_times = [0.0; SAMPLES];
    let mut first_digest = None;
    // Fixed 72 dispatches. No rejection sampling, retry loop, or timed readback.
    for iteration in 0..WARMUP + SAMPLES {
        // Encoder creation is excluded; the first window is encode + finish.
        let mut encoder = ctx.device().create_command_encoder(&Default::default());
        let start = Instant::now();
        gpu.encode(&mut encoder).expect("source + field encode");
        let commands = encoder.finish();
        let encode_us = start.elapsed().as_secs_f64() * 1e6;
        let start = Instant::now();
        ctx.queue().submit([commands]);
        wait_bounded(ctx);
        let submit_us = start.elapsed().as_secs_f64() * 1e6;
        if iteration >= WARMUP {
            encode_times[iteration - WARMUP] = encode_us;
            submit_times[iteration - WARMUP] = submit_us;
        }
        // Validate EVERY warmup/measured result, always outside BOTH timers.
        // The CPU sample/rank/accept oracle was generated before this loop.
        let digest = verify(ctx, &gpu, &staging, &expected, &mut seen);
        if let Some(first) = first_digest {
            assert_eq!(
                digest, first,
                "numeric digest reproducibility at iteration {iteration}"
            );
        } else {
            first_digest = Some(digest);
        }
    }
    assert_eq!(gpu.brush_scratch_bytes(), scratch_bytes);
    let encode = percentiles(encode_times);
    let submit = percentiles(submit_times);
    println!(
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:016x},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
        fixture.count,
        fixture.shape[0],
        fixture.shape[1],
        fixture.shape[2],
        workload.label,
        brush_count,
        expected.accepted,
        field.work_per_sample(),
        gpu_bytes,
        scratch_bytes,
        request_upload_bytes,
        INITIAL_INDIRECT_BYTES,
        initial_upload_bytes,
        verification_bytes,
        WARMUP + SAMPLES,
        expected.digest,
        encode[0],
        encode[1],
        encode[2],
        submit[0],
        submit[1],
        submit[2],
    );
}

fn main() {
    let ctx = GpuContext::try_headless_with_limits(wgpu::Limits {
        max_storage_buffers_per_shader_stage: 5,
        ..wgpu::Limits::downlevel_defaults()
    })
    .expect("real GPU acquisition with five storage buffers")
    .expect("physical GPU required; no adapter skips");
    let evidence = ctx.evidence();
    assert!(
        matches!(
            evidence.device_type.as_str(),
            "IntegratedGpu" | "DiscreteGpu"
        ),
        "physical GPU required: {evidence:?}"
    );
    let backend = match evidence.backend.as_str() {
        "Metal" => wgpu::Backends::METAL,
        "Vulkan" => wgpu::Backends::VULKAN,
        "Dx12" => wgpu::Backends::DX12,
        other => panic!("unexpected native backend {other}"),
    };
    if let Some(requested) = wgpu::Backends::from_env() {
        assert!(
            requested.contains(backend),
            "requested WGPU_BACKEND must be honored"
        );
    }
    assert!(ctx.device().limits().max_storage_buffers_per_shader_stage >= 5);
    println!("# adapter={evidence:?}; warmup={WARMUP}; samples={SAMPLES}; nearest-rank P50/P95/P99; units=host microseconds");
    println!("# fixed 72 dispatches per row; each native device poll and map-callback receive has a 30-second timeout; timeout or parity failure aborts, never retries/skips");
    println!("# encode+finish and submit+wait include source generation, field filtering and both GPU count clears; encoder creation, pipeline/buffer creation, uploads, readback, CPU verification and draw are excluded");
    println!("# initial_upload_bytes=request_uniforms(192+128)+64*brush_count+32 indirect constructor bytes; logical host payload, excludes driver metadata/implicit zero initialization");
    println!("# logical_gpu_bytes=owned source+field buffers, not physical VRAM/RSS; owned_host_brush_scratch_bytes=actual reusable upload scratch capacity; validation staging reported separately");
    println!("# all 72 results fully verified: source records, density/species/rank/acceptance, unique unordered compact membership and indirect count; digest is source-slot ordered and reproducible, not a collision-free identity");
    println!("# fixtures=20x20x25 and 40x50x50 jittered full 3D voxels, not 1D strips; min_distance=0, all sources accepted; not surface placement, GPU timestamps, isolated GPU execution time or product frame times");
    println!("# brush profile=32 concentric spheres at [0,0,0], inner=50+id mm, outer=4000mm, opacity=40000, Replace(12000+id*1000) for id%8==0 otherwise Add(+1000 even/-1000 odd), species=id+1, ascending IDs");
    println!("primary,nx,ny,nz,workload,brushes,field_accepted,field_work_per_sample,logical_gpu_bytes,owned_host_brush_scratch_bytes,initial_request_upload_bytes,initial_indirect_init_bytes,initial_upload_bytes,untimed_verification_staging_bytes,verified_dispatches,digest,encode_finish_p50_us,encode_finish_p95_us,encode_finish_p99_us,submit_wait_p50_us,submit_wait_p95_us,submit_wait_p99_us");
    for fixture in [
        Fixture {
            count: 10_000,
            shape: [20, 20, 25],
        },
        Fixture {
            count: 100_000,
            shape: [40, 50, 50],
        },
    ] {
        for workload in WORKLOADS {
            run(&ctx, fixture, workload);
        }
    }
}
