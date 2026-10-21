//! Strict physical-GPU coverage of the placement RHI lease/submit boundary.
//!
//! Readback uses native copies only. All placement, deletion and indirect raster
//! work under test must be recorded and submitted by `placement_rhi`.
#![cfg(feature = "placement_rhi")]

use std::{
    collections::BTreeSet,
    convert::Infallible,
    num::NonZeroUsize,
    sync::{mpsc, Arc},
    time::Duration,
};

use prism_render_driver::{self as rhi, RenderDevice};
use prism_world_gpu::{
    placement_rhi::{
        PlacementRhiBridge, PlacementRhiDrawPipeline, PlacementRhiScene,
        RegisteredHeightfieldPlacement, RegisteredPlacementDeletions,
    },
    GpuContext, GpuHeightfieldPlacement, GpuHeightfieldPlacementDeletionCapacity,
    GpuHeightfieldPlacementDeletionWorkBudget, GpuHeightfieldPlacementDeletions,
    GpuHeightfieldPlacementWorkBudget,
};
use prism_world_residency::{
    ScatterDensityField, ScatterFieldBuildBudget, ScatterFieldConfig, ScatterFieldSample,
    SectorCoord, SCATTER_KERNEL_VERSION,
};
use prism_world_stream::SectorId;
use prism_world_terrain::{
    estimate_heightfield_placement_work, scatter_on_heightfield, BakeLimits, CubeFace,
    HeightSample, HeightSampleValue, HeightSampler, HeightfieldPlacementBudget,
    HeightfieldPlacementConfig, HeightfieldPlacementDeletionBudget,
    HeightfieldPlacementDeletionSet, HeightfieldPlacementFilters, HeightfieldPlacementKey,
    HeightfieldPlacementRegion, HeightfieldSpec, HeightfieldTile, PlanarHeightfieldEmbedding,
    PlanarHeightfieldSurface, TileKey, HEIGHTFIELD_PLACEMENT_VERSION,
};

const SIDE: u32 = 48;
const ROW_BYTES: u32 = 256;
const CELL: SectorId = SectorId::new(0, -1, 0, -1);

// Each instance draws a separate triangle at its source-grid cell, rather than
// overlapping every instance at one pixel. Reading all four bindings catches a
// wrong compact buffer or a wrong records/samples/attributes binding order.
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
    let x = bitcast<i32>(records[slot].words[0]) + 3;
    let z = bitcast<i32>(records[slot].words[4]) + 3;
    let center = vec2<f32>(-1.0 + (f32(x) + 0.5) * (2.0 / 3.0),
                           1.0 - (f32(z) + 0.5) * (2.0 / 3.0));
    let triangle = array<vec2<f32>, 3>(vec2<f32>(-0.22, -0.22),
                                      vec2<f32>(0.22, -0.22),
                                      vec2<f32>(0.0, 0.22));
    var out: VertexOut;
    out.position = vec4<f32>(center + triangle[vertex], 0.0, 1.0);
    out.valid = select(0u, 1u, records[slot].words[12] != 0u &&
                       samples[slot].words[8] != 0u &&
                       attributes[slot].y == 7u && attributes[slot].z != 0u);
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

fn context() -> GpuContext {
    let context = GpuContext::try_headless_with_limits(wgpu::Limits {
        max_storage_buffers_per_shader_stage: 8,
        ..wgpu::Limits::downlevel_defaults()
    })
    .expect("physical placement RHI device acquisition failed")
    .expect("physical GPU required; no silent skip");
    let evidence = context.evidence();
    assert!(matches!(
        evidence.device_type.as_str(),
        "IntegratedGpu" | "DiscreteGpu"
    ));
    eprintln!("[placement_rhi] {evidence:?}");
    context
}

fn bridge(context: &GpuContext) -> PlacementRhiBridge {
    PlacementRhiBridge::new(context, NonZeroUsize::new(4).unwrap())
}

struct Fixture {
    config: HeightfieldPlacementConfig,
    region: HeightfieldPlacementRegion,
    field: ScatterDensityField,
    surface: PlanarHeightfieldSurface,
}

impl Fixture {
    fn new() -> Self {
        let tile = HeightfieldTile::bake(
            TileKey::root(CubeFace::PositiveY),
            HeightfieldSpec::new(4, 1).unwrap(),
            &Flat,
            BakeLimits {
                max_samples: 16,
                max_bytes: 32,
            },
        )
        .unwrap();
        Self {
            config: HeightfieldPlacementConfig {
                seed: 0x1234_5678_9abc_def0,
                version: 3,
                salt: 0x0fed_cba9_8765_4321,
                grid_spacing_mm: 1_000,
                min_distance_mm: 0,
                vertical_offset_mm: 0,
            },
            region: HeightfieldPlacementRegion::new([-3_000, -3_000], [0, 0]).unwrap(),
            field: ScatterDensityField::new(
                ScatterFieldConfig {
                    seed: 19,
                    version: 2,
                    base_density: u16::MAX,
                    base_species: 7,
                    noise: None,
                },
                Vec::new(),
                ScatterFieldBuildBudget {
                    max_brushes: 0,
                    max_brush_bytes: 0,
                },
            )
            .unwrap(),
            surface: PlanarHeightfieldSurface::new(
                Arc::new(tile),
                PlanarHeightfieldEmbedding {
                    sample_origin_mm: [-3_000, 0, -3_000],
                    sample_spacing_mm: 1_000,
                },
            )
            .unwrap(),
        }
    }

    fn filters(&self) -> HeightfieldPlacementFilters {
        HeightfieldPlacementFilters {
            surface_height_mm: [i64::MIN, i64::MAX],
            max_slope_rise: u32::MAX,
            max_slope_run: 1,
        }
    }

    fn budgets(
        &self,
        region: HeightfieldPlacementRegion,
    ) -> (
        GpuHeightfieldPlacementWorkBudget,
        HeightfieldPlacementBudget,
        usize,
    ) {
        let work = estimate_heightfield_placement_work(self.config, region, &self.field).unwrap();
        let n = work.primary_candidates as usize;
        (
            GpuHeightfieldPlacementWorkBudget {
                max_candidate_evaluations: work.max_candidate_evaluations,
                max_sample_reads: work.max_sample_reads,
                max_field_work: work.max_field_work,
            },
            HeightfieldPlacementBudget {
                max_candidate_evaluations: work.max_candidate_evaluations,
                max_sample_reads: work.max_sample_reads,
                max_field_work: work.max_field_work,
                max_output_points: n,
                max_output_bytes: n
                    * (std::mem::size_of::<prism_world_terrain::HeightfieldPlacementPoint>()
                        + std::mem::size_of::<ScatterFieldSample>()),
            },
            n,
        )
    }

    fn cpu_keys(&self) -> BTreeSet<HeightfieldPlacementKey> {
        scatter_on_heightfield(
            self.config,
            self.region,
            &self.field,
            &self.surface,
            self.filters(),
            self.budgets(self.region).1,
        )
        .unwrap()
        .points
        .iter()
        .map(|point| point.key)
        .collect()
    }

    fn placement(&self, context: &GpuContext) -> GpuHeightfieldPlacement {
        GpuHeightfieldPlacement::new(context, self.budgets(self.region).2, 0, 16, 3, u64::MAX)
            .unwrap()
    }

    fn deletions(&self, context: &GpuContext) -> GpuHeightfieldPlacementDeletions {
        GpuHeightfieldPlacementDeletions::new(
            context,
            self.placement(context),
            GpuHeightfieldPlacementDeletionCapacity {
                max_keys: 32,
                max_buffer_bytes: u64::MAX,
                max_scratch_bytes: 32 * 56,
            },
        )
        .unwrap()
    }

    fn write_deletions(
        &self,
        gpu: &mut RegisteredPlacementDeletions,
        region: HeightfieldPlacementRegion,
        deletions: &HeightfieldPlacementDeletionSet,
    ) {
        let (placement_budget, _, n) = self.budgets(region);
        gpu.write_request(
            self.config,
            region,
            &self.field,
            &self.surface,
            self.filters(),
            placement_budget,
            deletions,
            GpuHeightfieldPlacementDeletionWorkBudget {
                max_input_slots: n as u64,
                max_keys: 32,
                max_key_comparisons: u64::MAX,
            },
        )
        .unwrap();
    }

    fn write_heightfield(&self, gpu: &mut RegisteredHeightfieldPlacement) {
        gpu.write_request(
            self.config,
            self.region,
            &self.field,
            &self.surface,
            self.filters(),
            self.budgets(self.region).0,
        )
        .unwrap();
    }
}

fn deletion_set(keys: Vec<HeightfieldPlacementKey>) -> HeightfieldPlacementDeletionSet {
    HeightfieldPlacementDeletionSet::new(
        keys,
        HeightfieldPlacementDeletionBudget {
            max_keys: 32,
            max_key_bytes: 32 * std::mem::size_of::<HeightfieldPlacementKey>(),
        },
    )
    .unwrap()
}

fn scene(bridge: &PlacementRhiBridge) -> PlacementRhiScene {
    bridge
        .create_scene(&rhi::TextureDescriptor::new_2d(
            rhi::Extent3d::new_2d(SIDE, SIDE),
            rhi::TextureFormat::Rgba8Unorm,
            rhi::TextureUsages::RENDER_ATTACHMENT | rhi::TextureUsages::COPY_SRC,
        ))
        .unwrap()
}

fn pipeline(bridge: &PlacementRhiBridge) -> PlacementRhiDrawPipeline {
    let shader = bridge
        .device()
        .create_shader_module(&rhi::ShaderModuleDescriptor::wgsl(DRAW_WGSL));
    let result = bridge
        .create_draw_pipeline(&rhi::RenderPipelineDescriptor {
            label: Some("placement RHI per-source indirect raster".into()),
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
        .unwrap();
    bridge.device().destroy_shader_module(shader);
    result
}

fn staging(context: &GpuContext, size: u64) -> wgpu::Buffer {
    context.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("placement RHI observation only"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn read(context: &GpuContext, buffer: &wgpu::Buffer) -> Vec<u8> {
    let (sender, receiver) = mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        sender.send(result).unwrap();
    });
    context
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    receiver
        .recv_timeout(Duration::from_secs(30))
        .unwrap()
        .unwrap();
    let bytes = buffer.get_mapped_range(..).unwrap().to_vec();
    buffer.unmap();
    bytes
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

struct Observation {
    args: [u32; 4],
    keys: BTreeSet<HeightfieldPlacementKey>,
    pixels: Vec<u8>,
}

#[allow(clippy::too_many_arguments)]
fn observe(
    context: &GpuContext,
    config: HeightfieldPlacementConfig,
    capacity: usize,
    indirect: &wgpu::Buffer,
    records: &wgpu::Buffer,
    compact: &wgpu::Buffer,
    texture: &wgpu::Texture,
) -> Observation {
    let output = staging(context, (16 + 64 * capacity + 4 * capacity) as u64);
    let pixels = staging(context, u64::from(ROW_BYTES * SIDE));
    let mut encoder = context.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(indirect, 0, &output, 0, 16);
    encoder.copy_buffer_to_buffer(records, 0, &output, 16, 64 * capacity as u64);
    encoder.copy_buffer_to_buffer(
        compact,
        0,
        &output,
        16 + 64 * capacity as u64,
        4 * capacity as u64,
    );
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &pixels,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
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
    context.queue().submit([encoder.finish()]);
    let bytes = read(context, &output);
    let all = words(&bytes);
    let args: [u32; 4] = all[..4].try_into().unwrap();
    assert_eq!(args[0], 3);
    assert_eq!(&args[2..], &[0, 0]);
    assert!(args[1] as usize <= capacity);
    let records = &all[4..4 + capacity * 16];
    let compact = &all[4 + capacity * 16..];
    let keys: BTreeSet<_> = compact[..args[1] as usize]
        .iter()
        .map(|&slot| {
            assert!((slot as usize) < capacity);
            let base = slot as usize * 16;
            let coordinate = |offset| {
                (u64::from(records[base + offset]) | (u64::from(records[base + offset + 1]) << 32))
                    as i64
            };
            HeightfieldPlacementKey {
                placement_version: HEIGHTFIELD_PLACEMENT_VERSION,
                source: prism_world_residency::ScatterCandidateKey {
                    kernel_version: SCATTER_KERNEL_VERSION,
                    seed: config.seed,
                    version: config.version,
                    salt: config.salt,
                    grid_spacing_mm: config.grid_spacing_mm,
                    grid: SectorCoord::new(coordinate(0), coordinate(2), coordinate(4)),
                },
            }
        })
        .collect();
    assert_eq!(keys.len(), args[1] as usize, "no duplicate compact slots");
    Observation {
        args,
        keys,
        pixels: read(context, &pixels),
    }
}

fn assert_pixels(observed: &Observation, expected: &BTreeSet<HeightfieldPlacementKey>) {
    for z in -3..0 {
        for x in -3..0 {
            let present = expected
                .iter()
                .any(|key| key.source.grid == SectorCoord::new(x, 0, z));
            let px = ((x + 3) * 16 + 8) as usize;
            let py = ((z + 3) * 16 + 8) as usize;
            let offset = py * ROW_BYTES as usize + px * 4;
            let rgba = &observed.pixels[offset..offset + 4];
            if present {
                assert_eq!(rgba, &[0, 255, 0, 255], "survivor ({x}, {z})");
            } else {
                assert_eq!(&rgba[..3], &[0, 0, 0], "deleted/empty cell ({x}, {z})");
            }
        }
    }
}

#[test]
fn physical_rhi_full_key_deletion_cpu_parity_and_indirect_pixels() {
    let context = context();
    let mut bridge = bridge(&context);
    let fixture = Fixture::new();
    let cpu = fixture.cpu_keys();
    assert_eq!(cpu.len(), 9, "fixture must cover every source-grid cell");
    let mut iter = cpu.iter().copied();
    let removed = iter.next().unwrap();
    let survivor = iter.next().unwrap();
    let mut keys = vec![removed, removed];
    // Every admissible one-field near miss must preserve this survivor,
    // including high grid limbs. Placement/kernel versions and grid Y are
    // canonicalized by the public deletion-set validator, not GPU inputs.
    for component in 0..6 {
        let mut key = survivor;
        match component {
            0 => key.source.seed ^= 1u64 << 40,
            1 => key.source.version += 1,
            2 => key.source.salt ^= 1u64 << 40,
            3 => key.source.grid_spacing_mm += 1,
            4 => key.source.grid.x += 1i64 << 32,
            5 => key.source.grid.z += 1i64 << 32,
            _ => unreachable!(),
        }
        keys.push(key);
    }
    let deletions = deletion_set(keys);
    let expected: BTreeSet<_> = cpu
        .iter()
        .copied()
        .filter(|key| !deletions.is_deleted(*key))
        .collect();
    assert_eq!(expected.len(), 8);
    let mut gpu = bridge
        .register_deletions(fixture.deletions(&context))
        .unwrap();
    fixture.write_deletions(&mut gpu, fixture.region, &deletions);
    let target = scene(&bridge);
    let draw = pipeline(&bridge);
    assert_eq!(bridge.pending_count(), 0);
    let mut frame = bridge.begin_frame(&target).unwrap();
    frame.add_deletions(&gpu, CELL, 1, &draw).unwrap();
    let recorded = frame.record().unwrap();
    let token = recorded.submit();
    assert_eq!(bridge.pending_count(), 1);
    assert!(!token.is_complete());
    let snapshot = gpu.producer().registration_snapshot().unwrap();
    let actual = observe(
        &context,
        fixture.config,
        fixture.budgets(fixture.region).2,
        snapshot.indirect.buffer(),
        snapshot.upstream.records.buffer(),
        snapshot.compact.buffer(),
        &target.native_texture(),
    );
    assert_eq!(actual.keys, expected);
    assert_pixels(&actual, &expected);
    // Native polling/readback can finish GPU work, but only bridge retirement
    // may publish token completion and release registrations/frame allocations.
    assert!(!token.is_complete());
    assert!(bridge.wait(&token).unwrap() >= 1);
    assert!(token.is_complete());
    assert_eq!(bridge.pending_count(), 0);
}

#[test]
fn physical_rhi_nonempty_to_empty_deletion_resets_both_indirect_counts() {
    let context = context();
    let mut bridge = bridge(&context);
    let fixture = Fixture::new();
    let expected_nonempty = fixture.cpu_keys();
    let empty_region = HeightfieldPlacementRegion::new([0, 0], [0, 0]).unwrap();
    let deletions = deletion_set(Vec::new());
    let mut gpu = bridge
        .register_deletions(fixture.deletions(&context))
        .unwrap();
    let draw = pipeline(&bridge);
    for (revision, region) in [fixture.region, empty_region, fixture.region]
        .into_iter()
        .enumerate()
    {
        fixture.write_deletions(&mut gpu, region, &deletions);
        let target = scene(&bridge);
        let snapshot = gpu.producer().registration_snapshot().unwrap();
        let mut frame = bridge.begin_frame(&target).unwrap();
        frame
            .add_deletions(&gpu, CELL, revision as u64, &draw)
            .unwrap();
        let token = frame.record().unwrap().submit();
        let actual = observe(
            &context,
            fixture.config,
            fixture.budgets(fixture.region).2,
            snapshot.indirect.buffer(),
            snapshot.upstream.records.buffer(),
            snapshot.compact.buffer(),
            &target.native_texture(),
        );
        let expected = if revision == 1 {
            BTreeSet::new()
        } else {
            expected_nonempty.clone()
        };
        assert_eq!(actual.args[1] as usize, expected.len());
        assert_eq!(actual.keys, expected);
        assert_pixels(&actual, &expected);
        let upstream = staging(&context, 16);
        let mut encoder = context.device().create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(snapshot.upstream.indirect.buffer(), 0, &upstream, 0, 16);
        context.queue().submit([encoder.finish()]);
        assert_eq!(
            words(&read(&context, &upstream)),
            vec![3, expected.len() as u32, 0, 0]
        );
        bridge.wait(&token).unwrap();
        assert!(token.is_complete());
    }
    assert_eq!(bridge.pending_count(), 0);
}

#[test]
fn physical_rhi_rejects_foreign_context_and_same_context_foreign_bridge() {
    let context = context();
    let other_context = self::context();
    let mut a = bridge(&context);
    let b = bridge(&context);
    let fixture = Fixture::new();
    assert!(a
        .register_heightfield(fixture.placement(&other_context))
        .is_err());
    assert!(a
        .register_deletions(fixture.deletions(&other_context))
        .is_err());
    let mut gpu_a = a.register_deletions(fixture.deletions(&context)).unwrap();
    let mut gpu_b = b.register_deletions(fixture.deletions(&context)).unwrap();
    let deletions = deletion_set(Vec::new());
    fixture.write_deletions(&mut gpu_a, fixture.region, &deletions);
    fixture.write_deletions(&mut gpu_b, fixture.region, &deletions);
    let scene_a = scene(&a);
    let scene_b = scene(&b);
    let draw_a = pipeline(&a);
    let draw_b = pipeline(&b);
    assert!(a.begin_frame(&scene_b).is_err());
    let mut frame = a.begin_frame(&scene_a).unwrap();
    assert!(frame.add_deletions(&gpu_b, CELL, 1, &draw_a).is_err());
    assert!(frame.add_deletions(&gpu_a, CELL, 1, &draw_b).is_err());
    // Rejection must not poison a subsequent valid declaration on this frame.
    frame.add_deletions(&gpu_a, CELL, 1, &draw_a).unwrap();
    let token = frame.record().unwrap().submit();
    a.wait(&token).unwrap();
    assert!(token.is_complete());
    assert_eq!(a.pending_count(), 0);
    assert_eq!(b.pending_count(), 0);
}

#[test]
fn physical_rhi_cancellation_and_pending_frame_outlive_producer_pipeline_and_scene() {
    let context = context();
    let mut bridge = bridge(&context);
    let fixture = Fixture::new();
    let mut gpu = bridge
        .register_heightfield(fixture.placement(&context))
        .unwrap();
    fixture.write_heightfield(&mut gpu);
    let target = scene(&bridge);
    let draw = pipeline(&bridge);
    {
        let mut cancelled = bridge.begin_frame(&target).unwrap();
        cancelled.add_heightfield(&gpu, CELL, 1, &draw).unwrap();
    }
    assert_eq!(bridge.pending_count(), 0);
    fixture.write_heightfield(&mut gpu);
    {
        let mut cancelled = bridge.begin_frame(&target).unwrap();
        cancelled.add_heightfield(&gpu, CELL, 2, &draw).unwrap();
        drop(cancelled.record().unwrap());
    }
    assert_eq!(
        bridge.pending_count(),
        0,
        "recorded Drop must cancel, not submit"
    );
    fixture.write_heightfield(&mut gpu);
    let snapshot = gpu.producer().registration_snapshot().unwrap();
    let texture = target.native_texture();
    let mut frame = bridge.begin_frame(&target).unwrap();
    frame.add_heightfield(&gpu, CELL, 3, &draw).unwrap();
    let token = frame.record().unwrap().submit();
    // This also statically checks that submit releases producer/scene/pipeline
    // borrows. Native clones alone do not preserve RHI registration entries.
    drop(gpu);
    drop(draw);
    drop(target);
    assert_eq!(bridge.pending_count(), 1);
    assert!(!token.is_complete());
    let actual = observe(
        &context,
        fixture.config,
        fixture.budgets(fixture.region).2,
        snapshot.indirect.buffer(),
        snapshot.records.buffer(),
        snapshot.compact.buffer(),
        &texture,
    );
    let expected = fixture.cpu_keys();
    assert_eq!(actual.keys, expected);
    assert_pixels(&actual, &expected);
    assert!(!token.is_complete());
    assert!(bridge.poll().unwrap() >= 1);
    assert!(token.is_complete());
    assert_eq!(bridge.pending_count(), 0);
    assert_eq!(bridge.drain().unwrap(), 0);
    assert_eq!(bridge.poll().unwrap(), 0);
}

#[test]
fn physical_rhi_same_queue_rewrites_preserve_prior_scene_pixels_and_dropped_token_pending() {
    let context = context();
    let mut bridge = bridge(&context);
    let mut foreign = self::bridge(&context);
    let fixture = Fixture::new();
    let cpu = fixture.cpu_keys();
    assert_eq!(cpu.len(), 9);
    let requests = [
        deletion_set(Vec::new()),
        deletion_set(cpu.iter().copied().step_by(3).collect()),
        deletion_set(cpu.iter().copied().collect()),
    ];
    let expected: Vec<BTreeSet<_>> = requests
        .iter()
        .map(|deletions| {
            cpu.iter()
                .copied()
                .filter(|key| !deletions.is_deleted(*key))
                .collect()
        })
        .collect();
    assert_eq!(
        expected.iter().map(BTreeSet::len).collect::<Vec<_>>(),
        vec![9, 6, 0]
    );
    let targets: Vec<_> = (0..requests.len()).map(|_| scene(&bridge)).collect();
    let draw = pipeline(&bridge);
    let mut gpu = bridge
        .register_deletions(fixture.deletions(&context))
        .unwrap();
    let mut last_token = None;
    for (index, (deletions, target)) in requests.iter().zip(&targets).enumerate() {
        // All source requests remain nonempty, even when deletion removes every
        // candidate. Each write must be ordered after the previous native draw.
        fixture.write_deletions(&mut gpu, fixture.region, deletions);
        let mut frame = bridge.begin_frame(target).unwrap();
        frame
            .add_deletions(&gpu, CELL, index as u64 + 1, &draw)
            .unwrap();
        let token = frame.record().unwrap().submit();
        assert!(!token.is_complete());
        let pending = bridge.pending_count();
        assert!(pending >= 1);
        if index + 1 == requests.len() {
            last_token = Some(token);
        } else {
            drop(token);
            assert_eq!(
                bridge.pending_count(),
                pending,
                "dropping a token must not remove its pending submission"
            );
        }
        // begin_frame may nonblockingly retire earlier completed submissions;
        // do not assume all three remain pending on a fast physical device.
        // No explicit wait, readback or drain occurs between submissions.
    }
    let last_token = last_token.unwrap();
    let pending = bridge.pending_count();
    assert!(foreign.wait(&last_token).is_err());
    assert_eq!(foreign.pending_count(), 0);
    assert_eq!(bridge.pending_count(), pending);
    assert!(!last_token.is_complete());

    // Read only independent scene textures after every producer rewrite has
    // been submitted. The producer's current compact/indirect buffers describe
    // only the final all-deleted request and cannot verify earlier draws.
    let readbacks: Vec<_> = targets
        .iter()
        .map(|_| staging(&context, u64::from(ROW_BYTES * SIDE)))
        .collect();
    let mut encoder = context.device().create_command_encoder(&Default::default());
    for (target, readback) in targets.iter().zip(&readbacks) {
        let texture = target.native_texture();
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
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
    }
    context.queue().submit([encoder.finish()]);
    for (index, (readback, expected)) in readbacks.iter().zip(&expected).enumerate() {
        let pixels = read(&context, readback);
        for z in -3..0 {
            for x in -3..0 {
                let present = expected
                    .iter()
                    .any(|key| key.source.grid == SectorCoord::new(x, 0, z));
                let px = ((x + 3) * 16 + 8) as usize;
                let py = ((z + 3) * 16 + 8) as usize;
                let offset = py * ROW_BYTES as usize + px * 4;
                let rgba = &pixels[offset..offset + 4];
                if present {
                    assert_eq!(rgba, &[0, 255, 0, 255], "scene {index} survivor ({x}, {z})");
                } else {
                    assert_eq!(&rgba[..3], &[0, 0, 0], "scene {index} deleted ({x}, {z})");
                }
            }
        }
    }
    // Native readback completed all draws, but only the owning bridge retires
    // pending owners. This includes submissions whose tokens were dropped.
    assert_eq!(bridge.pending_count(), pending);
    assert!(!last_token.is_complete());
    assert_eq!(bridge.wait(&last_token).unwrap(), pending);
    assert!(last_token.is_complete());
    assert_eq!(bridge.pending_count(), 0);
    assert_eq!(bridge.drain().unwrap(), 0);
}

#[test]
fn physical_rhi_panicking_submit_callback_preserves_pending_until_drain() {
    use std::{
        panic::{catch_unwind, panic_any, AssertUnwindSafe},
        sync::atomic::{AtomicBool, Ordering},
    };

    const PANIC: &str = "placement RHI injected ready submit completion panic";
    let context = context();
    let mut bridge = bridge(&context);
    let fixture = Fixture::new();
    let mut gpu = bridge
        .register_deletions(fixture.deletions(&context))
        .unwrap();
    fixture.write_deletions(&mut gpu, fixture.region, &deletion_set(Vec::new()));
    let snapshot = gpu.producer().registration_snapshot().unwrap();
    let target = scene(&bridge);
    let texture = target.native_texture();
    let draw = pipeline(&bridge);
    let mut frame = bridge.begin_frame(&target).unwrap();
    frame.add_deletions(&gpu, CELL, 1, &draw).unwrap();
    let recorded = frame.record().unwrap();

    // wgpu-core 30.0.1: LifetimeTracker::add_work_done_closure stores an
    // idle-queue callback in work_done_closures. triage_submissions drains that
    // list unconditionally, and Queue::submit calls UserClosures::fire AFTER
    // enqueuing native work but BEFORE returning its index. Arm only after
    // record/begin_frame, explicitly flushing producer writes and waiting for
    // idle, so on_submitted_work_done cannot attach to a fresh upload submit
    // and no earlier poll can consume it.
    // This does not depend on the new GPU submission having already finished.
    context.queue().submit([]);
    context
        .device()
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        })
        .unwrap();
    let fired = Arc::new(AtomicBool::new(false));
    let callback_fired = fired.clone();
    let armed = Arc::new(AtomicBool::new(true));
    let callback_armed = armed.clone();
    context.queue().on_submitted_work_done(move || {
        if callback_armed.swap(false, Ordering::AcqRel) {
            callback_fired.store(true, Ordering::Release);
            panic_any(PANIC);
        }
    });
    let outcome = catch_unwind(AssertUnwindSafe(|| recorded.submit()));
    // If a backend changes callback scheduling, fail only AFTER disarming and
    // consuming the callback; never leave a pending panic for queue teardown.
    armed.store(false, Ordering::Release);
    if outcome.is_ok() {
        bridge.drain().unwrap();
    }
    let payload = outcome
        .err()
        .expect("ready native submit callback must panic");
    assert_eq!(payload.downcast_ref::<&str>(), Some(&PANIC));
    assert!(fired.load(Ordering::Acquire));
    assert_eq!(
        bridge.pending_count(),
        1,
        "submit unwind must retain the packet even without a returned index/token"
    );

    // The pending packet, not these public wrappers, must now own the RHI
    // registrations, draw bindings, pipeline and scene through recovery.
    drop(gpu);
    drop(draw);
    drop(target);
    // Native handle clones do not protect against SceneOwner::drop calling
    // Texture::destroy. Observe while the pending packet still pins its owner,
    // BEFORE drain releases the last scene owner and destroys the texture.
    let actual = observe(
        &context,
        fixture.config,
        fixture.budgets(fixture.region).2,
        snapshot.indirect.buffer(),
        snapshot.upstream.records.buffer(),
        snapshot.compact.buffer(),
        &texture,
    );
    let expected = fixture.cpu_keys();
    assert_eq!(actual.keys, expected);
    assert_pixels(&actual, &expected);
    assert_eq!(bridge.pending_count(), 1);
    assert_eq!(bridge.drain().unwrap(), 1);
    assert_eq!(bridge.pending_count(), 0);
    assert_eq!(bridge.drain().unwrap(), 0);
    assert_eq!(bridge.poll().unwrap(), 0);
}

#[test]
fn physical_rhi_exact_wait_recovers_after_host_completion_callback_panics() {
    use std::{
        panic::{catch_unwind, panic_any, AssertUnwindSafe},
        sync::atomic::{AtomicBool, Ordering},
    };

    const PANIC: &str = "placement RHI injected wait completion panic";
    let context = context();
    let mut bridge = bridge(&context);
    let fixture = Fixture::new();
    let mut gpu = bridge
        .register_heightfield(fixture.placement(&context))
        .unwrap();
    fixture.write_heightfield(&mut gpu);
    let target = scene(&bridge);
    let draw = pipeline(&bridge);
    let mut frame = bridge.begin_frame(&target).unwrap();
    frame.add_heightfield(&gpu, CELL, 1, &draw).unwrap();
    let token = frame.record().unwrap().submit();
    assert_eq!(bridge.pending_count(), 1);
    assert!(!token.is_complete());
    drop(gpu);
    drop(draw);
    drop(target);

    // Register the ONLY panicking callback AFTER submit returns, so it cannot
    // run during submit. Whether native work is already complete or still
    // active, the next exact wait processes this callback. There is no nested
    // registration, queue capture/self-cycle or assumed post-submit stage.
    let fired = Arc::new(AtomicBool::new(false));
    let callback_fired = fired.clone();
    let armed = Arc::new(AtomicBool::new(true));
    let callback_armed = armed.clone();
    context.queue().on_submitted_work_done(move || {
        if callback_armed.swap(false, Ordering::AcqRel) {
            callback_fired.store(true, Ordering::Release);
            panic_any(PANIC);
        }
    });
    let outcome = catch_unwind(AssertUnwindSafe(|| bridge.wait(&token)));
    armed.store(false, Ordering::Release);
    // Flush any deferred callback before assertions, including an unexpected
    // wait-error path. It is now disarmed, so teardown cannot panic a second
    // time. Native polling alone must not retire the bridge's pending packet.
    let cleanup = context.device().poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(Duration::from_secs(30)),
    });
    let payload = outcome
        .err()
        .expect("first native wait callback must panic");
    assert_eq!(payload.downcast_ref::<&str>(), Some(&PANIC));
    cleanup.unwrap();
    assert!(fired.load(Ordering::Acquire));
    assert_eq!(bridge.pending_count(), 1);
    assert!(!token.is_complete());

    // The first wait unwound before retirement, even if the bridge's own signal
    // callback already ran. A second successful exact wait retires the packet.
    assert_eq!(bridge.wait(&token).unwrap(), 1);
    assert!(token.is_complete());
    assert_eq!(bridge.pending_count(), 0);
    assert_eq!(bridge.wait(&token).unwrap(), 0);
    assert_eq!(bridge.drain().unwrap(), 0);
}
