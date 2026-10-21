//! Strict physical-GPU microbenchmark of deletion-only placement RHI reuse.
//!
//! One fixed producer/bridge/scene/draw is used sequentially in both modes.
//! Fixtures, canonical deletion sets and CPU references are built before timing.
//! Each mode starts from an untimed full rebuild, then alternates A->B and B->A.
//! Only the forced mode invalidates residency before each write, outside clocks.
//! Readback observes actual timed outputs without replaying placement or drawing.
//! These are host timings, not GPU timestamps or a product renderer benchmark.
//!
//! Low-disk development-profile invocation (run after other Cargo jobs finish):
//! CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 cargo bench -j1 --profile dev
//!     -p prism_world_gpu --features placement_rhi --bench placement_rhi_reuse

#![cfg(feature = "placement_rhi")]

use std::{
    collections::BTreeSet,
    convert::Infallible,
    mem::size_of,
    num::NonZeroUsize,
    sync::{mpsc, Arc},
    time::{Duration, Instant},
};

use prism_render_driver::{self as rhi, RenderDevice};
use prism_world_gpu::{
    placement_rhi::{
        PlacementRhiBridge, PlacementRhiDrawPipeline, PlacementRhiScene,
        RegisteredPlacementDeletions,
    },
    GpuContext, GpuHeightfieldPlacement, GpuHeightfieldPlacementDeletionCapacity,
    GpuHeightfieldPlacementDeletionRegistrationSnapshot,
    GpuHeightfieldPlacementDeletionUploadStats, GpuHeightfieldPlacementDeletionWorkBudget,
    GpuHeightfieldPlacementDeletions, GpuHeightfieldPlacementWorkBudget,
};
use prism_world_residency::{
    ScatterDensityField, ScatterFieldBuildBudget, ScatterFieldConfig, ScatterFieldSample,
};
use prism_world_stream::SectorId;
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
const KEYS: usize = 128;
const KEY_BYTES: usize = 56;
const SURFACE_SIDE: usize = 101;
const SIDE: u32 = 200;
const ROW_BYTES: u32 = 1_024;
const OUTPUT_BYTES: u64 = (16 + CANDIDATES * (64 + 4)) as u64;
const COPY_ALIGNMENT: u64 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u64;
const PIXEL_OFFSET: u64 = OUTPUT_BYTES.div_ceil(COPY_ALIGNMENT) * COPY_ALIGNMENT;
const PIXEL_BYTES: u64 = ROW_BYTES as u64 * SIDE as u64;
const READBACK_BYTES: u64 = PIXEL_OFFSET + PIXEL_BYTES;
const TARGET_BYTES: u64 = SIDE as u64 * SIDE as u64 * 4;
const SCRATCH_BYTES: usize = KEYS * KEY_BYTES;
const DELETION_BYTES: u64 = (KEYS * KEY_BYTES + CANDIDATES * 4 + 144) as u64;
const UPSTREAM_BYTES: u64 = (148 * CANDIDATES + 64 + 4 * SURFACE_SIDE * SURFACE_SIDE + 468) as u64;
const PRODUCER_BYTES: u64 = UPSTREAM_BYTES + DELETION_BYTES;
const TIMEOUT: Duration = Duration::from_secs(30);
const CELL: SectorId = SectorId::new(0, -1, 0, -1);

const _: () = {
    assert!(WARMUP.is_multiple_of(2) && SAMPLES >= 2 && SAMPLES.is_multiple_of(2));
    assert!(SIDE == 200 && CANDIDATES == 100 * 100);
    assert!(ROW_BYTES >= SIDE * 4);
    assert!(ROW_BYTES.is_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT));
    assert!(PIXEL_OFFSET >= OUTPUT_BYTES && PIXEL_OFFSET.is_multiple_of(COPY_ALIGNMENT));
};

// Test-only per-source raster: all four ABI buffers participate. Each source
// gets its own two-pixel cell. Its sampled center lies strictly inside the
// triangle, so both removed and surviving sources can be checked after drawing.
const DRAW_WGSL: &str = r#"
struct Record { words: array<u32, 16>, };
@group(0) @binding(0) var<storage, read> records: array<Record>;
@group(0) @binding(1) var<storage, read> samples: array<Record>;
@group(0) @binding(2) var<storage, read> attributes: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> compact: array<u32>;
struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) valid: u32,
};
@vertex fn vs(@builtin(vertex_index) vertex: u32,
              @builtin(instance_index) instance: u32) -> VertexOut {
    let slot = compact[instance];
    let x = bitcast<i32>(records[slot].words[0]) + 50;
    let z = bitcast<i32>(records[slot].words[4]) + 50;
    let center = vec2<f32>(-1.0 + (f32(x) + 0.75) * 0.02,
                           1.0 - (f32(z) + 0.75) * 0.02);
    let triangle = array<vec2<f32>, 3>(vec2<f32>(-0.009, -0.009),
                                      vec2<f32>(0.009, -0.009),
                                      vec2<f32>(0.0, 0.009));
    var out: VertexOut;
    out.position = vec4<f32>(center + triangle[vertex], 0.0, 1.0);
    out.valid = select(0u, 1u, records[slot].words[12] != 0u &&
                       samples[slot].words[8] != 0u &&
                       attributes[slot].y == 3u && attributes[slot].z != 0u);
    return out;
}
@fragment fn fs(in: VertexOut) -> @location(0) vec4<f32> {
    return select(vec4<f32>(1.0, 0.0, 0.0, 1.0),
                  vec4<f32>(0.0, 1.0, 0.0, 1.0), in.valid != 0u);
}
"#;

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
    sets: [HeightfieldPlacementDeletionSet; 2],
    expected: [BTreeSet<HeightfieldPlacementKey>; 2],
}

impl Fixture {
    fn new() -> Self {
        let tile = HeightfieldTile::bake(
            TileKey::root(CubeFace::PositiveY),
            HeightfieldSpec::new(SURFACE_SIDE as u32, 1).unwrap(),
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
                sample_spacing_mm: 1_000,
            },
        )
        .unwrap();
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
        .unwrap();
        let config = HeightfieldPlacementConfig {
            seed: 0x0123_4567_89ab_cdef,
            version: 1,
            salt: 17,
            grid_spacing_mm: 1_000,
            min_distance_mm: 0,
            vertical_offset_mm: 0,
        };
        let region = HeightfieldPlacementRegion::new([-50_000; 2], [50_000; 2]).unwrap();
        let filters = HeightfieldPlacementFilters {
            surface_height_mm: [i64::MIN, i64::MAX],
            max_slope_rise: u32::MAX,
            max_slope_run: 1,
        };
        let work = estimate_heightfield_placement_work(config, region, &field).unwrap();
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
        source.sort_unstable();
        assert_eq!(source.len(), CANDIDATES);
        assert!(source.windows(2).all(|pair| pair[0] < pair[1]));
        let middle = KEYS / 2;
        let mut keys = source[..KEYS + 1].to_vec();
        let alternate_key = keys.remove(middle);
        let base = deletion_set(keys);
        let mut keys = base.keys().to_vec();
        keys[middle] = alternate_key;
        let sets = [base, deletion_set(keys)];
        for set in &sets {
            assert_eq!(set.keys().len(), KEYS);
        }
        assert_eq!(
            sets[0]
                .keys()
                .iter()
                .zip(sets[1].keys())
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .map(|(index, _)| index)
                .collect::<Vec<_>>(),
            vec![middle],
            "exactly one canonical GPU key slot changes"
        );
        let expected = sets.each_ref().map(|set| {
            source
                .iter()
                .copied()
                .filter(|key| !set.is_deleted(*key))
                .collect::<BTreeSet<_>>()
        });
        for keys in &expected {
            assert_eq!(keys.len(), CANDIDATES - KEYS);
        }
        assert_ne!(expected[0], expected[1], "counts alone cannot prove parity");
        Self {
            config,
            region,
            field,
            surface,
            filters,
            budget,
            source,
            sets,
            expected,
        }
    }

    fn write(&self, gpu: &mut RegisteredPlacementDeletions, set: usize) {
        gpu.write_request(
            self.config,
            self.region,
            &self.field,
            &self.surface,
            self.filters,
            self.budget,
            &self.sets[set],
            GpuHeightfieldPlacementDeletionWorkBudget {
                max_input_slots: CANDIDATES as u64,
                max_keys: KEYS,
                max_key_comparisons: CANDIDATES as u64
                    * u64::from(usize::BITS - KEYS.leading_zeros()),
            },
        )
        .expect("registered deletion write_request");
    }
}

fn deletion_set(keys: Vec<HeightfieldPlacementKey>) -> HeightfieldPlacementDeletionSet {
    let budget = HeightfieldPlacementDeletionBudget {
        max_keys: keys.len(),
        max_key_bytes: keys.capacity() * size_of::<HeightfieldPlacementKey>(),
    };
    HeightfieldPlacementDeletionSet::new(keys, budget).expect("canonical deletion set")
}

fn pipeline(bridge: &PlacementRhiBridge) -> PlacementRhiDrawPipeline {
    let shader = bridge
        .device()
        .create_shader_module(&rhi::ShaderModuleDescriptor::wgsl(DRAW_WGSL));
    let draw = bridge
        .create_draw_pipeline(&rhi::RenderPipelineDescriptor {
            label: Some("placement reuse microbenchmark per-source draw".into()),
            layout: None,
            vertex: rhi::VertexState {
                module: shader,
                entry_point: "vs".into(),
                buffers: Vec::new(),
            },
            primitive: rhi::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(rhi::FragmentState {
                module: shader,
                entry_point: "fs".into(),
                targets: vec![Some(rhi::ColorTargetState::opaque(
                    rhi::TextureFormat::Rgba8Unorm,
                ))],
            }),
        })
        .expect("same-table draw pipeline");
    bridge.device().destroy_shader_module(shader);
    draw
}

fn assert_resources(
    gpu: &RegisteredPlacementDeletions,
    initial: &GpuHeightfieldPlacementDeletionRegistrationSnapshot,
) {
    let gpu = gpu.producer();
    let now = gpu.registration_snapshot().unwrap();
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
        assert_eq!(current.buffer(), original.buffer(), "fixed native buffer");
        assert_eq!(current.bytes(), original.bytes());
        assert_eq!(current.bytes(), current.buffer().size());
        bytes += current.bytes();
    }
    assert_eq!(bytes, PRODUCER_BYTES);
    assert_eq!(gpu.allocated_buffer_bytes(), PRODUCER_BYTES);
    assert_eq!(gpu.deletion_buffer_bytes(), DELETION_BYTES);
    assert_eq!(gpu.key_scratch_bytes(), SCRATCH_BYTES);
    assert_eq!(gpu.candidate_count(), CANDIDATES as u32);
}

// Observation copies only: no placement encode, RHI replay, or second draw.
fn verify(
    context: &GpuContext,
    gpu: &RegisteredPlacementDeletions,
    fixture: &Fixture,
    target: &PlacementRhiScene,
    staging: &wgpu::Buffer,
    set: usize,
) {
    let producer = gpu.producer();
    let texture = target.native_texture();
    let mut encoder = context.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(producer.indirect_args(), 0, staging, 0, 16);
    encoder.copy_buffer_to_buffer(producer.records(), 0, staging, 16, CANDIDATES as u64 * 64);
    encoder.copy_buffer_to_buffer(
        producer.compact_indices(),
        0,
        staging,
        16 + CANDIDATES as u64 * 64,
        CANDIDATES as u64 * 4,
    );
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: PIXEL_OFFSET,
                bytes_per_row: Some(ROW_BYTES),
                rows_per_image: Some(SIDE),
            },
        },
        wgpu::Extent3d {
            width: SIDE,
            height: SIDE,
            depth_or_array_layers: 1,
        },
    );
    let index = context.queue().submit([encoder.finish()]);
    let (sender, receiver) = mpsc::channel();
    staging.map_async(wgpu::MapMode::Read, .., move |result| {
        sender.send(result).expect("readback receiver");
    });
    context
        .device()
        .poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: Some(TIMEOUT),
        })
        .expect("exact observation submission wait");
    receiver.recv_timeout(TIMEOUT).unwrap().unwrap();
    let mapped = staging.get_mapped_range(..).unwrap();
    let word =
        |index: usize| u32::from_le_bytes(mapped[index * 4..index * 4 + 4].try_into().unwrap());
    let expected = &fixture.expected[set];
    assert_eq!(
        [word(0), word(1), word(2), word(3)],
        [3, expected.len() as u32, 0, 0],
        "actual timed indirect args"
    );
    let cache = producer.cache_key().expect("sealed producer cache key");
    assert_eq!(cache.upstream.config, fixture.config);
    assert_eq!(cache.deletions, fixture.sets[set]);
    let mut actual = BTreeSet::new();
    for compact_slot in 0..expected.len() {
        let slot = word(4 + CANDIDATES * 16 + compact_slot) as usize;
        assert!(slot < CANDIDATES, "compact source slot bounds");
        let base = 4 + slot * 16;
        let coordinate = |offset| {
            (u64::from(word(base + offset)) | (u64::from(word(base + offset + 1)) << 32)) as i64
        };
        // Recipe identity comes from the sealed cache, as in placement_rhi
        // tests; all three signed grid coordinates come from actual GPU data.
        let mut key = fixture.source[0];
        assert_eq!(key.placement_version, HEIGHTFIELD_PLACEMENT_VERSION);
        key.source.grid.x = coordinate(0);
        key.source.grid.y = coordinate(2);
        key.source.grid.z = coordinate(4);
        assert!(actual.insert(key), "no duplicate compact survivor keys");
    }
    assert_eq!(&actual, expected, "exact CPU source.filter full-key parity");
    for key in &fixture.source {
        let x = usize::try_from(key.source.grid.x + 50).unwrap();
        let z = usize::try_from(key.source.grid.z + 50).unwrap();
        assert!(x < 100 && z < 100);
        let offset = PIXEL_OFFSET as usize + (z * 2 + 1) * ROW_BYTES as usize + (x * 2 + 1) * 4;
        let rgb = &mapped[offset..offset + 3];
        if expected.contains(key) {
            assert_eq!(rgb, &[0, 255, 0], "draw survivor ({x}, {z})");
        } else {
            assert_eq!(rgb, &[0, 0, 0], "draw deleted source ({x}, {z})");
        }
    }
    drop(mapped);
    staging.unmap();
}

struct Timings {
    write: Vec<f64>,
    record: Vec<f64>,
    submit_wait: Vec<f64>,
}

impl Timings {
    fn new() -> Self {
        Self {
            write: Vec::with_capacity(SAMPLES),
            record: Vec::with_capacity(SAMPLES),
            submit_wait: Vec::with_capacity(SAMPLES),
        }
    }

    fn report(self, mode: &str, reused: usize) {
        let w = percentiles(self.write);
        let r = percentiles(self.record);
        let s = percentiles(self.submit_wait);
        // RHI exposes reuse count, not the internal execution plan. These
        // pass counts are derived from the current single-stream topology:
        // copy/reset/filter/draw, plus reset/source/project/field/finalize.
        let expected_passes = 4 + 5 * (1 - reused);
        println!(
            "{mode},{CANDIDATES},{KEYS},{},{reused},{expected_passes},{PRODUCER_BYTES},{DELETION_BYTES},{SCRATCH_BYTES},{READBACK_BYTES},{TARGET_BYTES},0,{KEY_BYTES},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
            CANDIDATES - KEYS, w[0], w[1], w[2], r[0], r[1], r[2], s[0], s[1], s[2]
        );
    }
}

fn percentiles(mut values: Vec<f64>) -> [f64; 3] {
    assert_eq!(values.len(), SAMPLES);
    values.sort_by(f64::total_cmp);
    [50_usize, 95, 99].map(|p| values[(values.len() * p).div_ceil(100) - 1])
}

fn main() {
    let context = GpuContext::try_headless_with_limits(wgpu::Limits {
        max_storage_buffers_per_shader_stage: 8,
        ..wgpu::Limits::downlevel_defaults()
    })
    .expect("physical GPU acquisition failed")
    .expect("physical GPU required; no silent skip");
    let evidence = context.evidence();
    assert!(matches!(
        evidence.device_type.as_str(),
        "IntegratedGpu" | "DiscreteGpu"
    ));
    println!("# adapter={evidence:?}; profile_debug_assertions={}; Cargo profile environment inherited unchanged", cfg!(debug_assertions));
    println!("# warmup_per_mode={WARMUP},samples_per_mode={SAMPLES}; nearest-rank P50/P95/P99 host microseconds; NOT GPU timestamps; NOT a product renderer benchmark");
    println!("# write=registered write_request CPU; record=begin_frame+add_deletions+record CPU; submit_wait=Recorded::submit+bridge.wait(exact receipt), INCLUDING native RHI encode/finish/error scopes, queued uploads, GPU work and retirement");
    println!("# same fixed producer/bridge/scene/draw, sequential modes; invalidation before write and outside clocks; setup/assertions/readback excluded; final two timed samples per mode verify both directions and draw pixels");
    println!("# logical payload bytes only, NOT physical VRAM/RSS: producer buffers, host key scratch, aligned/padded observation buffer, RGBA8 target; excludes upstream host scratch, retained recipe snapshots and opaque driver/RHI/pipeline/command allocations");
    println!("# expected_graph_passes is topology-derived, not introspected: full=9, cached=4; reused_upstream_count is asserted on every recorded frame");
    println!("mode,candidates,deletions,survivors,reused_upstreams,expected_graph_passes,producer_logical_buffer_bytes,deletion_logical_buffer_bytes,host_key_scratch_bytes,readback_logical_buffer_bytes,target_logical_texel_bytes,deletion_uniform_upload_bytes,key_upload_bytes,write_p50_us,write_p95_us,write_p99_us,record_p50_us,record_p95_us,record_p99_us,submit_wait_p50_us,submit_wait_p95_us,submit_wait_p99_us");

    let fixture = Fixture::new();
    let mut bridge = PlacementRhiBridge::new(&context, NonZeroUsize::new(1).unwrap());
    let placement = GpuHeightfieldPlacement::new(
        &context,
        CANDIDATES,
        0,
        fixture.surface.tile().samples().len(),
        3,
        UPSTREAM_BYTES,
    )
    .expect("fixed placement allocation");
    assert_eq!(placement.allocated_buffer_bytes(), UPSTREAM_BYTES);
    let producer = GpuHeightfieldPlacementDeletions::new(
        &context,
        placement,
        GpuHeightfieldPlacementDeletionCapacity {
            max_keys: KEYS,
            max_buffer_bytes: PRODUCER_BYTES,
            max_scratch_bytes: SCRATCH_BYTES,
        },
    )
    .expect("fixed deletion allocation");
    let mut gpu = bridge.register_deletions(producer).unwrap();
    let target = bridge
        .create_scene(&rhi::TextureDescriptor::new_2d(
            rhi::Extent3d::new_2d(SIDE, SIDE),
            rhi::TextureFormat::Rgba8Unorm,
            rhi::TextureUsages::RENDER_ATTACHMENT | rhi::TextureUsages::COPY_SRC,
        ))
        .unwrap();
    let draw = pipeline(&bridge);
    let staging = context.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("placement RHI reuse observation only"),
        size: READBACK_BYTES,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    assert_eq!(staging.size(), READBACK_BYTES);
    fixture.write(&mut gpu, 0);
    assert_eq!(
        gpu.producer().upload_stats(),
        GpuHeightfieldPlacementDeletionUploadStats {
            uniform_bytes: 128,
            key_bytes: (KEYS * KEY_BYTES) as u64,
        }
    );
    let initial = gpu.producer().registration_snapshot().unwrap();
    assert_resources(&gpu, &initial);

    for (mode, force) in [("rhi_cached", false), ("forced_full_rebuild", true)] {
        let mut timing = Timings::new();
        let mut verified_directions = [0_usize; 2];
        // Both modes receive the same untimed full rebuild of A, establishing
        // residency before the eight warmups; no first-frame miss is sampled.
        gpu.invalidate_resident_upstream();
        fixture.write(&mut gpu, 0);
        let mut frame = bridge.begin_frame(&target).unwrap();
        frame.add_deletions(&gpu, CELL, 0, &draw).unwrap();
        let recorded = frame.record().unwrap();
        assert_eq!(recorded.reused_upstream_count(), 0);
        let receipt = recorded.submit();
        bridge.wait(&receipt).expect("initial exact receipt wait");
        assert!(receipt.is_complete());
        assert_eq!(bridge.pending_count(), 0);
        assert_resources(&gpu, &initial);
        verify(&context, &gpu, &fixture, &target, &staging, 0);

        for sample in 0..WARMUP + SAMPLES {
            let set = usize::from(sample % 2 == 0);
            // Same pre-write position in both modes; only this branch differs.
            if force {
                gpu.invalidate_resident_upstream();
            }
            let start = Instant::now();
            fixture.write(&mut gpu, set);
            let write_us = start.elapsed().as_secs_f64() * 1e6;

            let start = Instant::now();
            let mut frame = bridge.begin_frame(&target).unwrap();
            frame
                .add_deletions(&gpu, CELL, sample as u64 + 1, &draw)
                .unwrap();
            let recorded = frame.record().unwrap();
            let record_us = start.elapsed().as_secs_f64() * 1e6;
            assert_eq!(recorded.reused_upstream_count(), usize::from(!force));

            let start = Instant::now();
            let receipt = recorded.submit();
            bridge.wait(&receipt).expect("timed exact RHI receipt wait");
            let submit_us = start.elapsed().as_secs_f64() * 1e6;

            assert!(receipt.is_complete());
            assert_eq!(bridge.pending_count(), 0);
            assert_eq!(
                gpu.producer().upload_stats(),
                GpuHeightfieldPlacementDeletionUploadStats {
                    uniform_bytes: 0,
                    key_bytes: KEY_BYTES as u64,
                },
                "both modes upload exactly one equal-size middle replacement"
            );
            assert_resources(&gpu, &initial);
            if sample >= WARMUP {
                timing.write.push(write_us);
                timing.record.push(record_us);
                timing.submit_wait.push(submit_us);
            }
            if sample >= WARMUP + SAMPLES - 2 {
                verify(&context, &gpu, &fixture, &target, &staging, set);
                verified_directions[set] += 1;
            }
        }
        assert_eq!(
            verified_directions,
            [1, 1],
            "both final timed directions verified"
        );
        println!("# verified mode={mode},samples={}:{},directions=A_to_B:B_to_A,survivors={},full_key_parity=true,draw_pixels=true", SAMPLES - 2, SAMPLES - 1, CANDIDATES - KEYS);
        timing.report(mode, usize::from(!force));
    }
    bridge.drain().expect("final RHI retirement");
    assert_eq!(bridge.pending_count(), 0);
}
