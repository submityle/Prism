//! Strict, bounded physical-GPU source + field + real heightfield benchmark.
//!
//! Register with harness=false and required-features=["terrain_scatter"].
//! Both host-wall-clock windows exclude uploads, readback, CPU checks and draw.
//! Every warmup and measured dispatch is checked against a complete CPU oracle;
//! these are not GPU timestamps, isolated GPU execution times or product frames.

#![cfg(feature = "terrain_scatter")]

use std::mem::{size_of, size_of_val};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use prism_world_gpu::{GpuContext, GpuScatterHeightfield};
use prism_world_residency::{
    estimate_scatter_work, pcg_hash_u32, scatter, DensityNoise, FieldScatterBudget, ScatterBudget,
    ScatterCandidateKey, ScatterConfig, ScatterDensityField, ScatterFieldBuildBudget,
    ScatterFieldConfig, ScatterPoint, ScatterRegion, SCATTER_KERNEL_VERSION,
};
use prism_world_terrain::{
    scatter_with_heightfield, BakeLimits, CubeFace, HeightSample, HeightSampleValue, HeightSampler,
    HeightfieldScatterBudget, HeightfieldScatterFilters, HeightfieldSpec, HeightfieldTile,
    PlanarHeightfieldEmbedding, PlanarHeightfieldSurface, TileKey,
};

const WARMUP: usize = 8;
const SAMPLES: usize = 64;
const WAIT_TIMEOUT: Duration = Duration::from_secs(30);
const VERTEX_COUNT: u32 = 3;
const SPACING_MM: u32 = 100;
const SURFACE_SIDE: u32 = 65;
const SURFACE_SAMPLES: usize = SURFACE_SIDE as usize * SURFACE_SIDE as usize;
const MAX_BRUSHES: usize = 0;
const SOURCE_UNIFORM_BYTES: u64 = 192;
const FIELD_UNIFORM_BYTES: u64 = 128;
const SURFACE_UNIFORM_BYTES: u64 = 128;
// Source, field and final heightfield indirect buffers, each initialized once.
const INITIAL_INDIRECT_BYTES: u64 = 3 * 16;
const DIGEST_INITIAL: u64 = 0xcbf2_9ce4_8422_2325;

#[derive(Clone, Copy)]
struct Fixture {
    count: usize,
    shape: [usize; 3],
    min_mm: [i64; 3],
    max_mm: [i64; 3],
}

impl Fixture {
    fn region(self) -> ScatterRegion {
        assert_eq!(self.shape.into_iter().product::<usize>(), self.count);
        assert!(self.shape.into_iter().all(|n| n > 1));
        assert!(self.count <= 100_000);
        for ((min, max), cells) in self.min_mm.into_iter().zip(self.max_mm).zip(self.shape) {
            assert_eq!(min.rem_euclid(i64::from(SPACING_MM)), 0);
            assert_eq!(max.rem_euclid(i64::from(SPACING_MM)), 0);
            assert_eq!(max - min, cells as i64 * i64::from(SPACING_MM));
        }
        ScatterRegion::new(self.min_mm, self.max_mm).expect("complete bounded 3D source voxels")
    }
}

struct Ramp;

impl HeightSampler for Ramp {
    type Error = core::convert::Infallible;

    fn sample_height(&self, sample: HeightSample) -> Result<HeightSampleValue, Self::Error> {
        Ok(HeightSampleValue::from_millimetres(
            i64::from(sample.u_q16()) / 256 - i64::from(sample.v_q16()) / 256,
        ))
    }
}

fn surface() -> PlanarHeightfieldSurface {
    let tile = Arc::new(
        HeightfieldTile::bake(
            TileKey::root(CubeFace::PositiveY),
            HeightfieldSpec::new(SURFACE_SIDE, 1).expect("real 65x65 scale-one heightfield"),
            &Ramp,
            BakeLimits {
                max_samples: SURFACE_SAMPLES,
                max_bytes: SURFACE_SAMPLES * size_of::<i16>(),
            },
        )
        .expect("bounded real sampler bake, not a fabricated encoded payload"),
    );
    assert_eq!(tile.samples().len(), SURFACE_SAMPLES);
    let surface = PlanarHeightfieldSurface::new(
        tile,
        PlanarHeightfieldEmbedding {
            sample_origin_mm: [-32_000, 0, -32_000],
            sample_spacing_mm: 1_000,
        },
    )
    .expect("valid half-open planar surface");
    assert_eq!(surface.min_xz_mm(), [-32_000; 2]);
    assert_eq!(surface.max_xz_mm(), [32_000; 2]);
    surface
}

fn scatter_config() -> ScatterConfig {
    ScatterConfig {
        seed: 0x0123_4567_89ab_cdef,
        version: 1,
        salt: 22,
        grid_spacing_mm: SPACING_MM,
        // Full aligned voxels and radius zero accept every source slot.
        min_distance_mm: 0,
    }
}

fn field() -> ScatterDensityField {
    ScatterDensityField::new(
        ScatterFieldConfig {
            seed: 0xfedc_ba98_7654_3210,
            version: 1,
            base_density: 40_000,
            base_species: 7,
            noise: Some(DensityNoise {
                cell_size_mm: 311,
                amplitude: 10_000,
                warp_cell_size_mm: 127,
                warp_amplitude_mm: 201,
            }),
        },
        Vec::new(),
        ScatterFieldBuildBudget {
            max_brushes: MAX_BRUSHES,
            max_brush_bytes: 0,
        },
    )
    .expect("same bounded noise/warp recipe as the CPU heightfield benchmark")
}

fn filters() -> HeightfieldScatterFilters {
    HeightfieldScatterFilters {
        surface_height_mm: [-100, 100],
        clearance_mm: [-250, 250],
        max_slope_rise: 1,
        max_slope_run: 10,
    }
}

fn percentiles(mut values: [f64; SAMPLES]) -> [f64; 3] {
    assert!(values.iter().all(|v| v.is_finite() && *v >= 0.0));
    values.sort_by(f64::total_cmp);
    [50_usize, 95, 99].map(|p| values[(SAMPLES * p).div_ceil(100) - 1])
}

fn wait_bounded(ctx: &GpuContext) {
    ctx.device()
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(WAIT_TIMEOUT),
        })
        .expect("GPU submission must complete within 30 seconds");
}

// Source priority is distinct from field selection rank. Mirror the CPU's
// complete ordered key mix so all 16 source words, including padding, are tested.
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

// Numeric word recurrence: independent of native endian, Vec layout and the
// unordered compact reservation order. A diagnostic, not collision-free identity.
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
    // Source record (16 words), then FINAL density/species/accepted/rank.
    slots: Vec<[u32; 20]>,
    field_accepted: usize,
    accepted: usize,
    surface_reads: u64,
    digest: u64,
}

fn oracle(
    fixture: Fixture,
    config: ScatterConfig,
    region: ScatterRegion,
    field: &ScatterDensityField,
    surface: &PlanarHeightfieldSurface,
    filters: HeightfieldScatterFilters,
) -> Oracle {
    let source_budget = ScatterBudget {
        max_candidate_evaluations: fixture.count as u64,
        max_output_points: fixture.count,
        max_output_bytes: fixture.count * size_of::<ScatterPoint>(),
    };
    let source = scatter(config, region, source_budget).expect("complete CPU source oracle");
    assert_eq!(source.points.len(), fixture.count, "no source truncation");
    assert_eq!(source.candidate_evaluations, fixture.count as u64);
    let output = scatter_with_heightfield(
        config,
        region,
        source_budget,
        field,
        FieldScatterBudget {
            max_field_work: fixture.count as u64 * field.work_per_sample(),
            max_output_bytes: fixture.count
                * (size_of::<ScatterPoint>()
                    + size_of::<prism_world_residency::ScatterFieldSample>()),
        },
        surface,
        filters,
        HeightfieldScatterBudget {
            max_sample_reads: fixture.count as u64 * 4,
        },
    )
    .expect("complete CPU scatter_with_heightfield oracle, never a prefix");
    assert_eq!(output.points.len(), output.attributes.len());
    assert_eq!(output.candidate_evaluations, fixture.count as u64);
    assert_eq!(
        output.field_work,
        fixture.count as u64 * field.work_per_sample()
    );
    let min_grid = region.min_mm.map(|p| p.div_euclid(i64::from(SPACING_MM)));
    let mut slots = Vec::with_capacity(fixture.count);
    let mut field_accepted = 0;
    let mut next_output = 0;
    for (slot, point) in source.points.iter().enumerate() {
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
        let sample = field.sample(point.position_mm);
        let rank = field.candidate_rank(point.key);
        let field_keep = field.accepts(point.key, sample.density);
        field_accepted += usize::from(field_keep);
        // Match the COMPLETE stable CPU output, not a digest or a count. The
        // GPU may compact in any order; source-slot identity is canonical.
        let keep = output
            .points
            .get(next_output)
            .is_some_and(|retained| retained.key == point.key);
        if keep {
            assert!(field_keep, "terrain cannot resurrect a field rejection");
            assert_eq!(output.points[next_output], *point);
            assert_eq!(output.attributes[next_output], sample);
            next_output += 1;
        }
        let mut words = [0; 20];
        for (axis, (coordinate, position)) in grid.into_iter().zip(point.position_mm).enumerate() {
            put_pair(&mut words, axis * 2, coordinate as u64);
            put_pair(&mut words, 6 + axis * 2, position as u64);
        }
        words[12] = 1; // Immutable SOURCE acceptance, not final terrain acceptance.
        put_pair(&mut words, 13, priority(point.key));
        words[16..].copy_from_slice(&[
            u32::from(sample.density),
            sample.species,
            u32::from(keep),
            rank,
        ]);
        slots.push(words);
    }
    assert_eq!(
        next_output,
        output.points.len(),
        "all CPU survivors matched exactly"
    );
    assert!(field_accepted > 0 && field_accepted < fixture.count);
    assert!(!output.points.is_empty() && output.points.len() < field_accepted);
    assert_eq!(output.surface_sample_reads, field_accepted as u64 * 4);
    let mut digest = digest_header(output.points.len());
    for (slot, words) in slots.iter().enumerate() {
        digest_word(&mut digest, slot as u32);
        for &word in words {
            digest_word(&mut digest, word);
        }
    }
    Oracle {
        slots,
        field_accepted,
        accepted: output.points.len(),
        surface_reads: output.surface_sample_reads,
        digest,
    }
}

fn read_word(bytes: &[u8], index: usize) -> u32 {
    u32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap())
}

fn verify(
    ctx: &GpuContext,
    gpu: &GpuScatterHeightfield,
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
        let _ = send.send(result);
    });
    wait_bounded(ctx);
    recv.recv_timeout(WAIT_TIMEOUT)
        .expect("bounded readback callback wait")
        .expect("validation staging mapping");
    let mapped = staging
        .get_mapped_range(..)
        .expect("validation mapped range");
    assert_eq!(mapped.len(), 16 + 84 * count);
    let indirect: [u32; 4] = std::array::from_fn(|word| read_word(&mapped, word));
    assert_eq!(indirect, [VERTEX_COUNT, expected.accepted as u32, 0, 0]);
    let accepted = indirect[1] as usize;
    assert!(accepted <= count, "no out-of-capacity compact prefix");
    seen.fill(false);
    // Only the live prefix is meaningful; never compare the unused suffix or
    // assume workgroup reservation order is repeatable across dispatches.
    for compact_index in 0..accepted {
        let slot = read_word(&mapped, 4 + count * 20 + compact_index) as usize;
        assert!(
            slot < count,
            "compact slot must be in complete source prefix"
        );
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
        assert_eq!(record.as_slice(), &words[..16], "CPU source slot {slot}");
        assert_eq!(
            attrs.as_slice(),
            &words[16..],
            "CPU final density/species/accepted/rank at slot {slot}"
        );
        assert_eq!(
            seen[slot],
            attrs[2] == 1,
            "compact membership at slot {slot}"
        );
        accepted_flags += usize::from(attrs[2] == 1);
        digest_word(&mut digest, slot as u32);
        for word in record.into_iter().chain(attrs) {
            digest_word(&mut digest, word);
        }
    }
    assert_eq!(
        accepted_flags, accepted,
        "flags/compact/indirect exact agreement"
    );
    assert_eq!(digest, expected.digest, "complete CPU/GPU numeric digest");
    drop(mapped);
    staging.unmap();
    digest
}

fn run(ctx: &GpuContext, fixture: Fixture) {
    let config = scatter_config();
    let region = fixture.region();
    let field = field();
    let surface = surface();
    let filters = filters();
    let estimate = estimate_scatter_work(config, region).expect("bounded fixture estimate");
    assert_eq!(estimate.primary_candidates, fixture.count as u64);
    assert_eq!(estimate.max_candidate_evaluations, fixture.count as u64);
    let source_work = estimate.max_candidate_evaluations;
    let field_work = estimate.primary_candidates * field.work_per_sample();
    let surface_work = HeightfieldScatterBudget {
        max_sample_reads: estimate.primary_candidates * 4,
    };
    let expected = oracle(fixture, config, region, &field, &surface, filters);
    assert_eq!(field.brushes().len(), MAX_BRUSHES);
    // Owned logical buffers only: not physical VRAM, allocator metadata or RSS.
    let gpu_bytes = 108 * fixture.count.max(1) as u64
        + 64 * field.brushes().len().max(1) as u64
        + 4 * SURFACE_SAMPLES.max(1) as u64
        + 496;
    let mut gpu = GpuScatterHeightfield::new(
        ctx,
        fixture.count,
        MAX_BRUSHES,
        SURFACE_SAMPLES,
        VERTEX_COUNT,
        gpu_bytes,
    )
    .expect("complete bounded source/field/heightfield GPU allocation");
    gpu.write_request(
        config,
        region,
        source_work,
        &field,
        field_work,
        &surface,
        filters,
        surface_work,
    )
    .expect("one initial request upload, never CPU-generated output writes");
    assert_eq!(gpu.candidate_count() as usize, fixture.count);
    assert_eq!(gpu.allocated_buffer_bytes(), gpu_bytes);
    let brush_scratch_bytes = gpu.brush_scratch_bytes();
    let height_scratch_bytes = gpu.height_scratch_bytes();
    let host_scratch_bytes = brush_scratch_bytes + height_scratch_bytes;
    let surface_upload_bytes = 4 * SURFACE_SAMPLES as u64;
    let request_upload_bytes = SOURCE_UNIFORM_BYTES
        + FIELD_UNIFORM_BYTES
        + SURFACE_UNIFORM_BYTES
        + 64 * field.brushes().len() as u64
        + surface_upload_bytes;
    let initial_upload_bytes = request_upload_bytes + INITIAL_INDIRECT_BYTES;
    let verification_bytes = 16 + 84 * fixture.count as u64;
    let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("heightfield benchmark untimed full-output validation"),
        size: verification_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut seen = vec![false; fixture.count];
    let oracle_slots_bytes = expected.slots.capacity() * size_of::<[u32; 20]>();
    let validation_seen_bytes = seen.capacity() * size_of::<bool>();
    let cpu_height_bytes = size_of_val(surface.tile().samples());
    // Drain ALL initial queue writes before even the first warmup. No uploads,
    // request updates, CPU production writes or readbacks are in either timer.
    ctx.queue().submit([]);
    wait_bounded(ctx);
    println!("# primary={} shape={:?} region={region:?} scatter={config:?} field={:?} embedding={:?} filters={filters:?}", fixture.count, fixture.shape, field.config(), surface.embedding());
    let mut encode_times = [0.0; SAMPLES];
    let mut submit_times = [0.0; SAMPLES];
    let mut first_digest = None;
    for iteration in 0..WARMUP + SAMPLES {
        // Exclude encoder creation; measure record/encode + encoder.finish().
        let mut encoder = ctx.device().create_command_encoder(&Default::default());
        let start = Instant::now();
        gpu.encode(&mut encoder)
            .expect("source + field + terrain encode");
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
        // EVERY warmup and measured result: complete slot parity and unique
        // compact membership, not sampling, a digest-only check or a count-only check.
        let digest = verify(ctx, &gpu, &staging, &expected, &mut seen);
        if let Some(first) = first_digest {
            assert_eq!(
                digest, first,
                "reproducible result at iteration {iteration}"
            );
        } else {
            first_digest = Some(digest);
        }
    }
    assert_eq!(gpu.brush_scratch_bytes(), brush_scratch_bytes);
    assert_eq!(gpu.height_scratch_bytes(), height_scratch_bytes);
    assert_eq!(gpu.allocated_buffer_bytes(), gpu_bytes);
    let encode = percentiles(encode_times);
    let submit = percentiles(submit_times);
    println!(
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:016x},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
        fixture.count,
        fixture.shape[0],
        fixture.shape[1],
        fixture.shape[2],
        expected.field_accepted,
        expected.accepted,
        expected.surface_reads,
        source_work,
        field_work,
        surface_work.max_sample_reads,
        gpu_bytes,
        host_scratch_bytes,
        brush_scratch_bytes,
        height_scratch_bytes,
        request_upload_bytes,
        INITIAL_INDIRECT_BYTES,
        initial_upload_bytes,
        verification_bytes,
        oracle_slots_bytes,
        validation_seen_bytes,
        cpu_height_bytes,
        expected.digest,
        encode[0],
        encode[1],
        encode[2],
        submit[0],
        submit[1],
        submit[2],
    );
}

fn require_release() {
    #[cfg(debug_assertions)]
    panic!("run this benchmark in release mode");
}

fn main() {
    require_release();
    let ctx = GpuContext::try_headless_with_limits(wgpu::Limits {
        max_storage_buffers_per_shader_stage: 6,
        ..wgpu::Limits::downlevel_defaults()
    })
    .expect("native device acquisition with six storage bindings")
    .expect("physical GPU required; no adapter skips");
    let evidence = ctx.evidence();
    assert!(
        matches!(
            evidence.device_type.as_str(),
            "IntegratedGpu" | "DiscreteGpu"
        ),
        "physical GPU required, no software fallback: {evidence:?}"
    );
    let backend = match evidence.backend.as_str() {
        "Metal" => wgpu::Backends::METAL,
        "Vulkan" => wgpu::Backends::VULKAN,
        "Dx12" => wgpu::Backends::DX12,
        other => panic!("unexpected native backend {other}"),
    };
    if let Some(requested) = wgpu::Backends::from_env() {
        assert!(requested.contains(backend), "WGPU_BACKEND must be honored");
    }
    assert!(ctx.device().limits().max_storage_buffers_per_shader_stage >= 6);
    println!("# adapter={evidence:?}; release physical GPU; warmup={WARMUP}; samples={SAMPLES}; nearest-rank P50/P95/P99; units=host microseconds");
    println!("# fixed 72 dispatches per row, all fully verified; native poll/map-callback waits each bounded to 30 seconds; timeout/parity/capacity failures abort, never retry/skip/truncate");
    println!("# encode+finish and submit+wait include source, author-field, surface filtering/compaction and GPU count clears; exclude encoder/pipeline/buffer creation, uploads, readback, CPU oracle/checks and draw");
    println!("# host wall clock, NOT GPU timestamps, isolated GPU execution or product frame times; no CPU-generated production records/attrs/compact/indirect writes");
    println!("# real 65x65 bake: sampler=u_q16/256-v_q16/256 mm, scale=1, origin[-32000,0,-32000], spacing1000mm; shared CPU i16 payload8450B, GPU sample upload16900B");
    println!("# complete 3D jittered voxels at100mm, min_distance=0, all source slots accepted; density40000+noise311/amp10000+warp127/amp201, no brushes; surface height[-100,100] clearance[-250,250] slope<=1/10");
    println!("# logical_gpu_bytes=108*max(capacity,1)+64*max(brushes,1)+4*max(samples,1)+496, excludes validation staging and physical VRAM/RSS/driver metadata");
    println!("# initial upload=uniforms(192+128+128)+64*actual_brushes+4*actual_samples+48 indirect constructor bytes; logical host payload, excludes implicit zero initialization/driver metadata");
    println!("# owned_host_scratch_bytes sums reusable brush+surface upload scratch capacities; validation/oracle payloads reported separately, not process peak memory; numeric digest includes every slot and indirect, not unordered compact order or collision-free identity");
    println!("primary,nx,ny,nz,field_accepted,final_accepted,cpu_logical_surface_reads,source_work_budget,field_work_budget,surface_read_budget,logical_gpu_bytes,owned_host_scratch_bytes,brush_scratch_bytes,height_scratch_bytes,initial_request_upload_bytes,initial_indirect_init_bytes,initial_upload_bytes,untimed_validation_staging_bytes,cpu_oracle_slots_capacity_bytes,validation_seen_capacity_bytes,shared_cpu_height_payload_bytes,digest,encode_finish_p50_us,encode_finish_p95_us,encode_finish_p99_us,submit_wait_p50_us,submit_wait_p95_us,submit_wait_p99_us");
    for fixture in [
        Fixture {
            count: 10_000,
            shape: [20, 25, 20],
            min_mm: [-1_000, -1_200, -1_000],
            max_mm: [1_000, 1_300, 1_000],
        },
        Fixture {
            count: 100_000,
            shape: [40, 50, 50],
            min_mm: [-2_000, -2_500, -2_500],
            max_mm: [2_000, 2_500, 2_500],
        },
    ] {
        run(&ctx, fixture);
    }
}
