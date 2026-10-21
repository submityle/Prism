//! Physical-GPU benchmark of incremental 56-byte placement deletion uploads.
//!
//! Fixed resources, recipes, canonical host sets and CPU survivor sets are built
//! before timing. Each iteration replaces one middle slot, then repeats that
//! exact request. The replacement direction alternates; each scenario gets its
//! own 8 warmups and 64 samples. Write CPU is measured separately; encode includes
//! finish, and submit includes pending queue uploads, the full compute chain and
//! an exact bounded submission wait. Readback is outside every timed window.
//! All timings are host wall-clock measurements, not GPU timestamps.

#![cfg(feature = "terrain_scatter")]

use std::{
    collections::BTreeSet,
    convert::Infallible,
    mem::size_of,
    sync::{mpsc, Arc},
    time::{Duration, Instant},
};

use prism_world_gpu::{
    GpuContext, GpuHeightfieldPlacement, GpuHeightfieldPlacementDeletionCapacity,
    GpuHeightfieldPlacementDeletionRegistrationSnapshot,
    GpuHeightfieldPlacementDeletionUploadStats, GpuHeightfieldPlacementDeletionWorkBudget,
    GpuHeightfieldPlacementDeletions, GpuHeightfieldPlacementWorkBudget,
};
use prism_world_residency::{
    ScatterDensityField, ScatterFieldBuildBudget, ScatterFieldConfig, ScatterFieldSample,
};
use prism_world_terrain::{
    estimate_heightfield_placement_work, scatter_on_heightfield, BakeLimits, CubeFace,
    HeightSample, HeightSampleValue, HeightSampler, HeightfieldPlacementBudget,
    HeightfieldPlacementConfig, HeightfieldPlacementDeletionBudget,
    HeightfieldPlacementDeletionSet, HeightfieldPlacementFilters, HeightfieldPlacementKey,
    HeightfieldPlacementPoint, HeightfieldPlacementRegion, HeightfieldSpec, HeightfieldTile,
    PlanarHeightfieldEmbedding, PlanarHeightfieldSurface, TileKey, HEIGHTFIELD_PLACEMENT_VERSION,
};

const WARMUP: usize = 8;
const SAMPLES: usize = 64;
const CANDIDATES: usize = 10_000;
const SURFACE_SIDE: usize = 101;
const SPACING_MM: u32 = 1_000;
const KEY_BYTES: usize = 56;
const TIMEOUT: Duration = Duration::from_secs(30);
const READBACK_BYTES: u64 = (16 + CANDIDATES * (64 + 4)) as u64;

struct Flat;

impl HeightSampler for Flat {
    type Error = Infallible;

    fn sample_height(&self, _: HeightSample) -> Result<HeightSampleValue, Self::Error> {
        Ok(HeightSampleValue::from_millimetres(0))
    }
}

struct Fixture {
    config: HeightfieldPlacementConfig,
    region: HeightfieldPlacementRegion,
    field: ScatterDensityField,
    surface: PlanarHeightfieldSurface,
    filters: HeightfieldPlacementFilters,
    budget: GpuHeightfieldPlacementWorkBudget,
    source: Vec<HeightfieldPlacementKey>,
}

impl Fixture {
    fn new() -> Self {
        let tile = HeightfieldTile::bake(
            TileKey::root(CubeFace::PositiveY),
            HeightfieldSpec::new(SURFACE_SIDE as u32, 1).expect("valid heightfield"),
            &Flat,
            BakeLimits {
                max_samples: SURFACE_SIDE * SURFACE_SIDE,
                max_bytes: SURFACE_SIDE * SURFACE_SIDE * 2,
            },
        )
        .expect("flat heightfield bake");
        let surface = PlanarHeightfieldSurface::new(
            Arc::new(tile),
            PlanarHeightfieldEmbedding {
                sample_origin_mm: [-50_000, 0, -50_000],
                sample_spacing_mm: SPACING_MM,
            },
        )
        .expect("valid planar surface");
        let field = ScatterDensityField::new(
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
        .expect("complete field recipe");
        let config = HeightfieldPlacementConfig {
            seed: 0x0123_4567_89ab_cdef,
            version: 1,
            salt: 17,
            grid_spacing_mm: SPACING_MM,
            min_distance_mm: 0,
            vertical_offset_mm: 0,
        };
        let region = HeightfieldPlacementRegion::new([-50_000; 2], [50_000; 2])
            .expect("valid placement region");
        let filters = HeightfieldPlacementFilters {
            surface_height_mm: [i64::MIN, i64::MAX],
            max_slope_rise: u32::MAX,
            max_slope_run: 1,
        };
        let work = estimate_heightfield_placement_work(config, region, &field)
            .expect("bounded placement work");
        assert_eq!(work.primary_candidates, CANDIDATES as u64);
        let budget = GpuHeightfieldPlacementWorkBudget {
            max_candidate_evaluations: work.max_candidate_evaluations,
            max_sample_reads: work.max_sample_reads,
            max_field_work: work.max_field_work,
        };
        let cpu = scatter_on_heightfield(
            config,
            region,
            &field,
            &surface,
            filters,
            HeightfieldPlacementBudget {
                max_candidate_evaluations: work.max_candidate_evaluations,
                max_sample_reads: work.max_sample_reads,
                max_field_work: work.max_field_work,
                max_output_points: CANDIDATES,
                max_output_bytes: CANDIDATES
                    * (size_of::<HeightfieldPlacementPoint>() + size_of::<ScatterFieldSample>()),
            },
        )
        .expect("CPU placement reference");
        let mut source: Vec<_> = cpu.points.iter().map(|point| point.key).collect();
        assert_eq!(source.len(), CANDIDATES, "flat full-density source");
        source.sort_unstable();
        assert!(source.windows(2).all(|pair| pair[0] < pair[1]));
        Self {
            config,
            region,
            field,
            surface,
            filters,
            budget,
            source,
        }
    }

    fn write(
        &self,
        gpu: &mut GpuHeightfieldPlacementDeletions,
        deletions: &HeightfieldPlacementDeletionSet,
        budget: GpuHeightfieldPlacementDeletionWorkBudget,
    ) {
        gpu.write_request(
            self.config,
            self.region,
            &self.field,
            &self.surface,
            self.filters,
            self.budget,
            deletions,
            budget,
        )
        .expect("placement deletion write_request");
    }

    fn survivors(
        &self,
        deletions: &HeightfieldPlacementDeletionSet,
    ) -> BTreeSet<HeightfieldPlacementKey> {
        self.source
            .iter()
            .copied()
            .filter(|key| !deletions.is_deleted(*key))
            .collect()
    }
}

fn deletion_set(keys: Vec<HeightfieldPlacementKey>) -> HeightfieldPlacementDeletionSet {
    let budget = HeightfieldPlacementDeletionBudget {
        max_keys: keys.len(),
        max_key_bytes: keys.capacity() * size_of::<HeightfieldPlacementKey>(),
    };
    HeightfieldPlacementDeletionSet::new(keys, budget).expect("canonical deletion set")
}

fn wait(ctx: &GpuContext, index: wgpu::SubmissionIndex) {
    ctx.device()
        .poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: Some(TIMEOUT),
        })
        .expect("exact physical-GPU submission must complete");
}

fn execute(ctx: &GpuContext, gpu: &GpuHeightfieldPlacementDeletions) {
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    gpu.encode(&mut encoder).expect("placement deletion encode");
    wait(ctx, ctx.queue().submit([encoder.finish()]));
}

fn assert_upload(gpu: &GpuHeightfieldPlacementDeletions, uniform_bytes: u64, key_bytes: u64) {
    assert_eq!(
        gpu.upload_stats(),
        GpuHeightfieldPlacementDeletionUploadStats {
            uniform_bytes,
            key_bytes
        }
    );
}

fn assert_resources(
    gpu: &GpuHeightfieldPlacementDeletions,
    initial: &GpuHeightfieldPlacementDeletionRegistrationSnapshot,
    logical_bytes: u64,
    scratch_bytes: usize,
) {
    let now = gpu
        .registration_snapshot()
        .expect("uploaded resource snapshot");
    let mut bytes = 0;
    for (current, original) in [
        (&now.params, &initial.params),
        (&now.keys, &initial.keys),
        (&now.compact, &initial.compact),
        (&now.indirect, &initial.indirect),
        (&now.upstream.source_params, &initial.upstream.source_params),
        (
            &now.upstream.project_params,
            &initial.upstream.project_params,
        ),
        (&now.upstream.field_params, &initial.upstream.field_params),
        (&now.upstream.records, &initial.upstream.records),
        (&now.upstream.samples, &initial.upstream.samples),
        (&now.upstream.attributes, &initial.upstream.attributes),
        (&now.upstream.compact, &initial.upstream.compact),
        (&now.upstream.indirect, &initial.upstream.indirect),
        (&now.upstream.status, &initial.upstream.status),
        (&now.upstream.heights, &initial.upstream.heights),
        (&now.upstream.brushes, &initial.upstream.brushes),
    ] {
        assert_eq!(
            current.buffer(),
            original.buffer(),
            "native buffer must remain fixed"
        );
        assert_eq!(current.bytes(), original.bytes());
        assert_eq!(current.bytes(), current.buffer().size());
        bytes += current.bytes();
    }
    assert_eq!(bytes, logical_bytes);
    assert_eq!(gpu.allocated_buffer_bytes(), logical_bytes);
    assert_eq!(gpu.key_scratch_bytes(), scratch_bytes);
    assert_eq!(gpu.candidate_count(), CANDIDATES as u32);
}

// Copies only the completed output: never re-encodes the chain to verify a sample.
fn verify(
    ctx: &GpuContext,
    gpu: &GpuHeightfieldPlacementDeletions,
    fixture: &Fixture,
    staging: &wgpu::Buffer,
    expected: &BTreeSet<HeightfieldPlacementKey>,
) {
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(gpu.indirect_args(), 0, staging, 0, 16);
    encoder.copy_buffer_to_buffer(gpu.records(), 0, staging, 16, CANDIDATES as u64 * 64);
    encoder.copy_buffer_to_buffer(
        gpu.compact_indices(),
        0,
        staging,
        16 + CANDIDATES as u64 * 64,
        CANDIDATES as u64 * 4,
    );
    let index = ctx.queue().submit([encoder.finish()]);
    let (sender, receiver) = mpsc::channel();
    staging.map_async(wgpu::MapMode::Read, .., move |result| {
        sender.send(result).expect("readback receiver");
    });
    wait(ctx, index);
    receiver
        .recv_timeout(TIMEOUT)
        .expect("readback callback")
        .expect("GPU readback map");
    let mapped = staging.get_mapped_range(..).expect("mapped readback");
    let word = |index: usize| {
        u32::from_le_bytes(
            mapped[index * 4..index * 4 + 4]
                .try_into()
                .expect("u32 word"),
        )
    };
    assert_eq!(
        [word(0), word(1), word(2), word(3)],
        [3, expected.len() as u32, 0, 0]
    );
    let mut actual = BTreeSet::new();
    for compact_slot in 0..expected.len() {
        let slot = word(4 + CANDIDATES * 16 + compact_slot) as usize;
        assert!(slot < CANDIDATES, "compact source slot in range");
        let base = 4 + slot * 16;
        let signed_pair = |offset| {
            (u64::from(word(base + offset)) | (u64::from(word(base + offset + 1)) << 32)) as i64
        };
        // Full immutable recipe identity comes from the sealed cache request;
        // all three signed source-grid coordinates come from the GPU record.
        let mut key = fixture.source[0];
        assert_eq!(key.placement_version, HEIGHTFIELD_PLACEMENT_VERSION);
        key.source.grid.x = signed_pair(0);
        key.source.grid.y = signed_pair(2);
        key.source.grid.z = signed_pair(4);
        assert!(actual.insert(key), "no duplicate survivor keys");
    }
    assert_eq!(
        &actual, expected,
        "exact CPU source.filter deletion-key parity"
    );
    assert_eq!(
        gpu.cache_key().expect("sealed cache key").upstream.config,
        fixture.config
    );
    drop(mapped);
    staging.unmap();
}

struct Timings {
    write: Vec<f64>,
    encode: Vec<f64>,
    submit: Vec<f64>,
}

impl Timings {
    fn new() -> Self {
        Self {
            write: Vec::with_capacity(SAMPLES),
            encode: Vec::with_capacity(SAMPLES),
            submit: Vec::with_capacity(SAMPLES),
        }
    }
}

fn percentiles(mut values: Vec<f64>) -> [f64; 3] {
    assert_eq!(values.len(), SAMPLES);
    values.sort_by(f64::total_cmp);
    [50_usize, 95, 99].map(|p| values[(values.len() * p).div_ceil(100) - 1])
}

fn run(ctx: &GpuContext, fixture: &Fixture, count: usize) {
    let middle = count / 2;
    let mut keys = fixture.source[..count + 1].to_vec();
    let alternate_key = keys.remove(middle);
    let base = deletion_set(keys);
    let mut keys = base.keys().to_vec();
    keys[middle] = alternate_key;
    let alternate = deletion_set(keys);
    assert_eq!(base.keys().len(), count);
    assert_eq!(alternate.keys().len(), count);
    assert_eq!(
        base.keys()
            .iter()
            .zip(alternate.keys())
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, _)| i)
            .collect::<Vec<_>>(),
        vec![middle]
    );
    let mut keys = base.keys().to_vec();
    keys.push(fixture.source[count + 1]);
    let appended = deletion_set(keys);
    assert_eq!(
        &appended.keys()[..count],
        base.keys(),
        "append cannot shift slots"
    );
    let empty = HeightfieldPlacementDeletionSet::default();
    let expected_base = fixture.survivors(&base);
    let expected_alternate = fixture.survivors(&alternate);
    let expected_appended = fixture.survivors(&appended);
    let expected_empty = fixture.survivors(&empty);
    assert_eq!(expected_base.len(), CANDIDATES - count);
    assert_eq!(expected_alternate.len(), CANDIDATES - count);
    assert_ne!(
        expected_base, expected_alternate,
        "replacement changes actual membership"
    );

    let capacity = count + 1;
    let placement = GpuHeightfieldPlacement::new(
        ctx,
        CANDIDATES,
        0,
        fixture.surface.tile().samples().len(),
        3,
        u64::MAX,
    )
    .expect("fixed placement allocation");
    let upstream_bytes = placement.allocated_buffer_bytes();
    let deletion_bytes = (capacity * KEY_BYTES + CANDIDATES * 4 + 144) as u64;
    let logical_bytes = upstream_bytes + deletion_bytes;
    let mut gpu = GpuHeightfieldPlacementDeletions::new(
        ctx,
        placement,
        GpuHeightfieldPlacementDeletionCapacity {
            max_keys: capacity,
            max_buffer_bytes: logical_bytes,
            max_scratch_bytes: capacity * KEY_BYTES,
        },
    )
    .expect("fixed deletion allocation");
    let scratch_bytes = gpu.key_scratch_bytes();
    assert_eq!(scratch_bytes, capacity * KEY_BYTES);
    assert_eq!(gpu.deletion_buffer_bytes(), deletion_bytes);
    let budget = GpuHeightfieldPlacementDeletionWorkBudget {
        max_input_slots: CANDIDATES as u64,
        max_keys: capacity,
        max_key_comparisons: CANDIDATES as u64 * u64::from(usize::BITS - capacity.leading_zeros()),
    };
    let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("incremental placement deletion benchmark readback"),
        size: READBACK_BYTES,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    fixture.write(&mut gpu, &base, budget);
    assert_upload(&gpu, 128, (count * KEY_BYTES) as u64);
    let initial = gpu
        .registration_snapshot()
        .expect("initial resource snapshot");
    execute(ctx, &gpu);
    verify(ctx, &gpu, fixture, &staging, &expected_base);

    // Untimed deterministic admission/upload checks; submit before each rewrite.
    // Truncation/empty retain stale suffix bytes, but key_count fences them out.
    for (name, set, expected, uniform_bytes, key_bytes) in [
        (
            "append",
            &appended,
            &expected_appended,
            128,
            KEY_BYTES as u64,
        ),
        ("prefix_truncate", &base, &expected_base, 128, 0),
        ("empty", &empty, &expected_empty, 128, 0),
        ("empty_unchanged", &empty, &expected_empty, 0, 0),
        (
            "restore",
            &base,
            &expected_base,
            128,
            (count * KEY_BYTES) as u64,
        ),
    ] {
        fixture.write(&mut gpu, set, budget);
        assert_upload(&gpu, uniform_bytes, key_bytes);
        assert_resources(&gpu, &initial, logical_bytes, scratch_bytes);
        execute(ctx, &gpu);
        verify(ctx, &gpu, fixture, &staging, expected);
        println!("# check deletions={count},scenario={name},uniform_bytes={uniform_bytes},key_bytes={key_bytes},exact_survivors={}", expected.len());
    }

    let mut timings = [Timings::new(), Timings::new()];
    for sample in 0..WARMUP + SAMPLES {
        let (set, expected) = if sample % 2 == 0 {
            (&alternate, &expected_alternate)
        } else {
            (&base, &expected_base)
        };
        // A->B, B->B, B->A, A->A: no reset/upload is hidden in either scenario.
        for (scenario, timing) in timings.iter_mut().enumerate() {
            let start = Instant::now();
            fixture.write(&mut gpu, set, budget);
            let write_us = start.elapsed().as_secs_f64() * 1e6;
            let mut encoder = ctx.device().create_command_encoder(&Default::default());
            let start = Instant::now();
            gpu.encode(&mut encoder)
                .expect("timed placement deletion encode");
            let commands = encoder.finish();
            let encode_us = start.elapsed().as_secs_f64() * 1e6;
            let start = Instant::now();
            let index = ctx.queue().submit([commands]);
            wait(ctx, index);
            let submit_us = start.elapsed().as_secs_f64() * 1e6;
            assert_upload(&gpu, 0, if scenario == 0 { KEY_BYTES as u64 } else { 0 });
            assert_resources(&gpu, &initial, logical_bytes, scratch_bytes);
            if sample >= WARMUP {
                timing.write.push(write_us);
                timing.encode.push(encode_us);
                timing.submit.push(submit_us);
            }
            // Both final directions/scenarios are checked on their actual timed
            // output, after all three clocks stop, with no producer re-dispatch.
            if sample >= WARMUP + SAMPLES - 2 {
                verify(ctx, &gpu, fixture, &staging, expected);
            }
        }
    }
    for (scenario, key_bytes, timing) in ["middle_replace", "unchanged"]
        .into_iter()
        .zip([KEY_BYTES, 0])
        .zip(timings)
        .map(|((name, bytes), timing)| (name, bytes, timing))
    {
        let w = percentiles(timing.write);
        let e = percentiles(timing.encode);
        let s = percentiles(timing.submit);
        println!(
            "{CANDIDATES},{count},{scenario},{},{capacity},{logical_bytes},{deletion_bytes},{scratch_bytes},{READBACK_BYTES},0,{key_bytes},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
            expected_base.len(), w[0], w[1], w[2], e[0], e[1], e[2], s[0], s[1], s[2],
        );
    }
}

fn main() {
    let ctx = GpuContext::try_headless_with_limits(wgpu::Limits {
        max_storage_buffers_per_shader_stage: 8,
        ..wgpu::Limits::downlevel_defaults()
    })
    .expect("physical GPU acquisition failed")
    .expect("physical GPU required; no skip");
    let evidence = ctx.evidence();
    assert!(matches!(
        evidence.device_type.as_str(),
        "IntegratedGpu" | "DiscreteGpu"
    ));
    println!("# adapter={evidence:?}; profile_debug_assertions={}; Cargo profile environment inherited unchanged", cfg!(debug_assertions));
    println!("# warmup={WARMUP},samples_per_scenario={SAMPLES}; nearest-rank P50/P95/P99; host microseconds, NOT GPU timestamps");
    println!("# write=write_request CPU; encode=encode+finish CPU (encoder creation excluded); submit=submit+exact bounded wait, including queued transfers and full placement+deletion chain");
    println!("# logical buffer/scratch payloads only, NOT physical VRAM/RSS; readback allocation reported separately; setup/key construction/assertions/readback excluded");
    println!("primary,deletions,scenario,survivors,key_capacity,total_logical_gpu_bytes,deletion_logical_gpu_bytes,host_key_scratch_bytes,verification_logical_gpu_bytes,uniform_upload_bytes,key_upload_bytes,write_p50_us,write_p95_us,write_p99_us,encode_p50_us,encode_p95_us,encode_p99_us,submit_wait_p50_us,submit_wait_p95_us,submit_wait_p99_us");
    let fixture = Fixture::new();
    for count in [128, 8_192] {
        run(&ctx, &fixture, count);
    }
}
