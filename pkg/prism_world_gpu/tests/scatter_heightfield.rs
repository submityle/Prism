//! Physical-GPU parity for exact terrain thinning and a GPU-stream indirect draw.
//! All height payloads come from public bake/rebake APIs, never synthetic buffers.

#![cfg(feature = "terrain_scatter")]

use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::sync::Arc;

use prism_world_gpu::{
    GpuContext, GpuScatterError, GpuScatterFieldError, GpuScatterHeightfield,
    GpuScatterHeightfieldError,
};
use prism_world_residency::{
    estimate_scatter_work, pcg_hash_u32, scatter, DensityBrush, DensityBrushOp, DensityNoise,
    FieldScatterBudget, ScatterBudget, ScatterCandidateKey, ScatterConfig, ScatterDensityField,
    ScatterError, ScatterFieldBuildBudget, ScatterFieldConfig, ScatterFieldError,
    ScatterFieldSample, ScatterPoint, ScatterRegion, SectorCoord, SCATTER_KERNEL_VERSION,
};
use prism_world_terrain::{
    scatter_with_heightfield, BakeError, BakeLimits, CubeFace, DirtyRegion, HeightSample,
    HeightSampleValue, HeightSampler, HeightfieldQueryError, HeightfieldScatterBudget,
    HeightfieldScatterCacheKey, HeightfieldScatterError, HeightfieldScatterFilters,
    HeightfieldScatterOutput, HeightfieldSpec, HeightfieldTile, PlanarHeightfieldEmbedding,
    PlanarHeightfieldSurface, TileKey,
};

fn gpu_context() -> GpuContext {
    let ctx = GpuContext::try_headless_with_limits(wgpu::Limits {
        max_storage_buffers_per_shader_stage: 6,
        ..wgpu::Limits::downlevel_defaults()
    })
    .expect("terrain scatter requires real-device acquisition with six storage buffers")
    .expect("terrain scatter requires an adapter; skipping is forbidden");
    let evidence = ctx.evidence();
    let actual = match evidence.backend.as_str() {
        "Metal" => wgpu::Backends::METAL,
        "Vulkan" => wgpu::Backends::VULKAN,
        "Dx12" => wgpu::Backends::DX12,
        other => panic!("unexpected native terrain scatter backend {other}"),
    };
    if let Some(requested) = wgpu::Backends::from_env() {
        assert!(requested.contains(actual), "WGPU_BACKEND was not honored");
    }
    assert!(
        matches!(
            evidence.device_type.as_str(),
            "IntegratedGpu" | "DiscreteGpu"
        ),
        "physical GPU required: {evidence:?}"
    );
    assert!(ctx.device().limits().max_storage_buffers_per_shader_stage >= 6);
    eprintln!("[scatter_heightfield] adapter={evidence:?}");
    ctx
}

struct GridHeights<'a> {
    side: u32,
    heights: &'a [i64],
}

impl HeightSampler for GridHeights<'_> {
    type Error = Infallible;

    fn sample_height(&self, sample: HeightSample) -> Result<HeightSampleValue, Self::Error> {
        let index = |q: u16| ((u64::from(q) * u64::from(self.side - 1) + 32_767) / 65_535) as usize;
        Ok(HeightSampleValue::from_millimetres(
            self.heights[index(sample.v_q16()) * self.side as usize + index(sample.u_q16())],
        ))
    }
}

fn surface(
    side: u32,
    spacing: u32,
    origin: [i64; 3],
    scale: i64,
    heights: &[i64],
) -> PlanarHeightfieldSurface {
    assert_eq!(heights.len(), side as usize * side as usize);
    let tile = HeightfieldTile::bake(
        TileKey::root(CubeFace::PositiveY),
        HeightfieldSpec::new(side, scale).unwrap(),
        &GridHeights { side, heights },
        BakeLimits {
            max_samples: heights.len(),
            max_bytes: heights.len() * size_of::<i16>(),
        },
    )
    .unwrap();
    PlanarHeightfieldSurface::new(
        Arc::new(tile),
        PlanarHeightfieldEmbedding {
            sample_origin_mm: origin,
            sample_spacing_mm: spacing,
        },
    )
    .unwrap()
}

fn config(spacing: u32, distance: u32) -> ScatterConfig {
    ScatterConfig {
        seed: 0x3a65_c807_9742_1eab,
        version: 7,
        salt: 0xf09c_1172,
        grid_spacing_mm: spacing,
        min_distance_mm: distance,
    }
}

fn region(min: [i64; 3], max: [i64; 3]) -> ScatterRegion {
    ScatterRegion::new(min, max).unwrap()
}

fn field(
    density: u16,
    noise: Option<DensityNoise>,
    brushes: Vec<DensityBrush>,
) -> ScatterDensityField {
    let budget = ScatterFieldBuildBudget {
        max_brushes: brushes.len(),
        max_brush_bytes: brushes.capacity() * size_of::<DensityBrush>(),
    };
    ScatterDensityField::new(
        ScatterFieldConfig {
            seed: 91,
            version: 3,
            base_density: density,
            base_species: 11,
            noise,
        },
        brushes,
        budget,
    )
    .unwrap()
}

fn constant(density: u16) -> ScatterDensityField {
    field(density, None, vec![])
}

fn painted() -> ScatterDensityField {
    field(
        46_000,
        Some(DensityNoise {
            cell_size_mm: 7,
            amplitude: 12_000,
            warp_cell_size_mm: 11,
            warp_amplitude_mm: 3,
        }),
        vec![
            DensityBrush {
                id: 9,
                center_mm: [5, 0, 0],
                inner_radius_mm: 7,
                outer_radius_mm: 12,
                opacity: u16::MAX,
                density: DensityBrushOp::Add(-20_000),
                species: Some(97),
            },
            DensityBrush {
                id: 2,
                center_mm: [-5, 0, 0],
                inner_radius_mm: 7,
                outer_radius_mm: 12,
                opacity: u16::MAX,
                density: DensityBrushOp::Add(4_000),
                species: Some(23),
            },
        ],
    )
}

fn unrestricted() -> HeightfieldScatterFilters {
    HeightfieldScatterFilters {
        surface_height_mm: [i64::MIN, i64::MAX],
        clearance_mm: [i64::MIN, i64::MAX],
        max_slope_rise: u32::MAX,
        max_slope_run: 1,
    }
}

fn logical_bytes(capacity: usize, brushes: usize, samples: usize) -> u64 {
    108 * capacity.max(1) as u64 + 64 * brushes.max(1) as u64 + 4 * samples.max(1) as u64 + 496
}

fn new_gpu(
    ctx: &GpuContext,
    capacity: usize,
    brushes: usize,
    samples: usize,
) -> GpuScatterHeightfield {
    GpuScatterHeightfield::new(
        ctx,
        capacity,
        brushes,
        samples,
        3,
        logical_bytes(capacity, brushes, samples),
    )
    .unwrap()
}

fn source_budget(cfg: ScatterConfig, bounds: ScatterRegion) -> ScatterBudget {
    let work = estimate_scatter_work(cfg, bounds).unwrap();
    let n = usize::try_from(work.primary_candidates).unwrap();
    assert!(n <= 8_192, "successful fixtures must remain small");
    ScatterBudget {
        max_candidate_evaluations: work.max_candidate_evaluations,
        max_output_points: n,
        max_output_bytes: n * size_of::<ScatterPoint>(),
    }
}

fn cpu(
    cfg: ScatterConfig,
    bounds: ScatterRegion,
    f: &ScatterDensityField,
    terrain: &PlanarHeightfieldSurface,
    filters: HeightfieldScatterFilters,
) -> HeightfieldScatterOutput {
    let work = estimate_scatter_work(cfg, bounds).unwrap();
    let n = work.primary_candidates as usize;
    scatter_with_heightfield(
        cfg,
        bounds,
        source_budget(cfg, bounds),
        f,
        FieldScatterBudget {
            max_field_work: work.primary_candidates * f.work_per_sample(),
            max_output_bytes: n * (size_of::<ScatterPoint>() + size_of::<ScatterFieldSample>()),
        },
        terrain,
        filters,
        HeightfieldScatterBudget {
            max_sample_reads: 4 * work.primary_candidates,
        },
    )
    .unwrap()
}

fn upload(
    gpu: &mut GpuScatterHeightfield,
    cfg: ScatterConfig,
    bounds: ScatterRegion,
    f: &ScatterDensityField,
    terrain: &PlanarHeightfieldSurface,
    filters: HeightfieldScatterFilters,
) {
    let work = estimate_scatter_work(cfg, bounds).unwrap();
    assert!(work.primary_candidates <= 8_192);
    gpu.write_request(
        cfg,
        bounds,
        work.max_candidate_evaluations,
        f,
        work.primary_candidates * f.work_per_sample(),
        terrain,
        filters,
        HeightfieldScatterBudget {
            max_sample_reads: 4 * work.primary_candidates,
        },
    )
    .unwrap();
    assert_eq!(u64::from(gpu.candidate_count()), work.primary_candidates);
    assert_eq!(
        gpu.cache_key(),
        Some(&cpu(cfg, bounds, f, terrain, filters).cache_key)
    );
}

fn staging(ctx: &GpuContext, size: u64) -> wgpu::Buffer {
    ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("terrain scatter test readback only"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn read(ctx: &GpuContext, buffer: &wgpu::Buffer) -> Vec<u8> {
    let (send, recv) = std::sync::mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        send.send(result).unwrap();
    });
    ctx.wait();
    recv.recv().unwrap().expect("terrain GPU readback mapping");
    let bytes = buffer.get_mapped_range(..).unwrap().to_vec();
    buffer.unmap();
    bytes
}

fn capture(
    ctx: &GpuContext,
    gpu: &GpuScatterHeightfield,
    encoder: &mut wgpu::CommandEncoder,
) -> wgpu::Buffer {
    let n = u64::from(gpu.candidate_count());
    let output = staging(ctx, 16 + 84 * n);
    encoder.copy_buffer_to_buffer(gpu.indirect_args(), 0, &output, 0, 16);
    if n != 0 {
        encoder.copy_buffer_to_buffer(gpu.records(), 0, &output, 16, 64 * n);
        encoder.copy_buffer_to_buffer(gpu.attributes(), 0, &output, 16 + 64 * n, 16 * n);
        encoder.copy_buffer_to_buffer(gpu.compact_indices(), 0, &output, 16 + 80 * n, 4 * n);
    }
    output
}

fn submit_capture(ctx: &GpuContext, gpu: &GpuScatterHeightfield) -> wgpu::Buffer {
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    gpu.encode(&mut encoder).unwrap();
    let result = capture(ctx, gpu, &mut encoder);
    ctx.queue().submit([encoder.finish()]);
    result
}

fn pair(words: &[u32], index: usize) -> u64 {
    u64::from(words[index]) | (u64::from(words[index + 1]) << 32)
}

fn hash64(value: u64) -> u64 {
    (u64::from(pcg_hash_u32(value)) << 32) | u64::from(pcg_hash_u32(value ^ 0x9e37_79b9_7f4a_7c15))
}

fn priority(key: ScatterCandidateKey) -> u64 {
    let mut hash = hash64(0x7363_6174_7465_7200 ^ u64::from(key.kernel_version));
    for value in [
        key.seed,
        u64::from(key.version),
        key.salt,
        u64::from(key.grid_spacing_mm),
        key.grid.x as u64,
        key.grid.y as u64,
        key.grid.z as u64,
    ] {
        hash = hash64(hash ^ value);
    }
    hash64(hash ^ 0x082e_fa98_ec4e_6c89)
}

#[derive(Debug)]
struct Snapshot {
    accepted: Vec<(ScatterPoint, ScatterFieldSample)>,
    source_rejected: usize,
    field_rejected: usize,
    terrain_rejected: usize,
    invalid_positions: usize,
}

fn assert_snapshot(
    bytes: &[u8],
    cfg: ScatterConfig,
    bounds: ScatterRegion,
    f: &ScatterDensityField,
    terrain: &PlanarHeightfieldSurface,
    filters: HeightfieldScatterFilters,
) -> Snapshot {
    let words: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let n = estimate_scatter_work(cfg, bounds)
        .unwrap()
        .primary_candidates as usize;
    assert_eq!(bytes.len(), 16 + 84 * n);
    let source = scatter(cfg, bounds, source_budget(cfg, bounds)).unwrap();
    let source_keys: BTreeSet<_> = source.points.iter().map(|p| p.key).collect();
    let expected = cpu(cfg, bounds, f, terrain, filters);
    let final_keys: BTreeSet<_> = expected.points.iter().map(|p| p.key).collect();
    assert_eq!(
        &words[..4],
        &[3, expected.points.len() as u32, 0, 0],
        "terrain indirect args: {cfg:?}, {bounds:?}, {filters:?}"
    );
    let mut result = Snapshot {
        accepted: vec![],
        source_rejected: 0,
        field_rejected: 0,
        terrain_rejected: 0,
        invalid_positions: 0,
    };
    let mut accepted_slots = BTreeSet::new();
    if n != 0 {
        let spacing = i64::from(cfg.grid_spacing_mm);
        let min = bounds.min_mm.map(|v| v.div_euclid(spacing));
        let max = bounds.max_mm.map(|v| (v - 1).div_euclid(spacing));
        let mut slot = 0;
        for x in min[0]..=max[0] {
            for y in min[1]..=max[1] {
                for z in min[2]..=max[2] {
                    let record = &words[4 + 16 * slot..4 + 16 * (slot + 1)];
                    let attrs = &words[4 + 16 * n + 4 * slot..4 + 16 * n + 4 * (slot + 1)];
                    let key = ScatterCandidateKey {
                        kernel_version: SCATTER_KERNEL_VERSION,
                        seed: cfg.seed,
                        version: cfg.version,
                        salt: cfg.salt,
                        grid_spacing_mm: cfg.grid_spacing_mm,
                        grid: SectorCoord::new(x, y, z),
                    };
                    assert_eq!(
                        [
                            pair(record, 0) as i64,
                            pair(record, 2) as i64,
                            pair(record, 4) as i64
                        ],
                        [x, y, z],
                        "global XYZ source slot {slot}"
                    );
                    assert_eq!(record[15], 0, "source padding {slot}");
                    let source_accepted = source_keys.contains(&key);
                    assert_eq!(
                        record[12],
                        u32::from(source_accepted),
                        "source accepted {key:?}"
                    );
                    let point = match key.rebuild() {
                        Ok(point) => {
                            assert_eq!(
                                [
                                    pair(record, 6) as i64,
                                    pair(record, 8) as i64,
                                    pair(record, 10) as i64
                                ],
                                point.position_mm,
                                "source CPU position {key:?}"
                            );
                            assert_eq!(pair(record, 13), priority(key), "source priority {key:?}");
                            Some(point)
                        }
                        Err(ScatterError::CoordinateOverflow) => {
                            assert_eq!(&record[6..12], &[0; 6]);
                            assert!(!source_accepted);
                            result.invalid_positions += 1;
                            None
                        }
                        Err(error) => panic!("unexpected candidate rebuild error {error:?}"),
                    };
                    if source_accepted {
                        let point = point.unwrap();
                        let sample = f.sample(point.position_mm);
                        let field_accepted = f.accepts(key, sample.density);
                        let accepted = final_keys.contains(&key);
                        assert_eq!(
                            attrs,
                            &[
                                u32::from(sample.density),
                                sample.species,
                                u32::from(accepted),
                                f.candidate_rank(key)
                            ],
                            "sample/rank/final accepted {key:?}"
                        );
                        assert!(!accepted || field_accepted);
                        if accepted {
                            result.accepted.push((point, sample));
                            accepted_slots.insert(slot as u32);
                        } else if field_accepted {
                            result.terrain_rejected += 1;
                        } else {
                            result.field_rejected += 1;
                        }
                    } else {
                        assert_eq!(
                            attrs, &[0; 4],
                            "all source-rejected attributes must clear at {slot}"
                        );
                        result.source_rejected += 1;
                    }
                    slot += 1;
                }
            }
        }
        assert_eq!(slot, n);
    }
    assert_eq!(
        result.accepted,
        expected
            .points
            .into_iter()
            .zip(expected.attributes)
            .collect::<Vec<_>>()
    );
    let compact = &words[4 + 20 * n..];
    let mut compact_slots = BTreeSet::new();
    for &slot in &compact[..result.accepted.len()] {
        assert!((slot as usize) < n, "compact index outside primary prefix");
        assert!(compact_slots.insert(slot), "duplicate compact slot {slot}");
    }
    assert_eq!(
        compact_slots, accepted_slots,
        "exact unordered final compact set"
    );
    for (i, (point, _)) in result.accepted.iter().enumerate() {
        assert_eq!(point.key.rebuild().unwrap(), *point);
        for (other, _) in &result.accepted[..i] {
            let distance: i128 = point
                .position_mm
                .into_iter()
                .zip(other.position_mm)
                .map(|(a, b)| {
                    let d = i128::from(a) - i128::from(b);
                    d * d
                })
                .sum();
            assert!(distance >= i128::from(cfg.min_distance_mm).pow(2));
        }
    }
    result
}

fn run(
    ctx: &GpuContext,
    gpu: &mut GpuScatterHeightfield,
    cfg: ScatterConfig,
    bounds: ScatterRegion,
    f: &ScatterDensityField,
    terrain: &PlanarHeightfieldSurface,
    filters: HeightfieldScatterFilters,
) -> Snapshot {
    upload(gpu, cfg, bounds, f, terrain, filters);
    let output = submit_capture(ctx, gpu);
    assert_snapshot(&read(ctx, &output), cfg, bounds, f, terrain, filters)
}

#[test]
fn real_tiles_closed_height_clearance_fractional_comparisons_and_diagonal_ownership() {
    let ctx = gpu_context();
    let mut gpu = new_gpu(&ctx, 129, 0, 9);
    let f = constant(u16::MAX);
    let cfg = config(1, 0);
    let terrain = surface(2, 4, [0, -2, 0], 1, &[0, 4, 0, 4]);
    let result = run(
        &ctx,
        &mut gpu,
        cfg,
        region([0, -3, 0], [3, 4, 3]),
        &f,
        &terrain,
        HeightfieldScatterFilters {
            surface_height_mm: [-1, 0],
            clearance_mm: [-1, 1],
            max_slope_rise: 1,
            max_slope_run: 1,
        },
    );
    assert_eq!(result.accepted.len(), 18);
    assert_eq!(result.terrain_rejected, 45);
    for x in [1, 2] {
        for clearance in [-1, 1] {
            assert!(result
                .accepted
                .iter()
                .any(|(p, _)| p.position_mm == [x, x - 2 + clearance, 0]));
        }
    }

    // h=(-3+2*x)/3: neither -1/3 nor +1/3 may round into a closed zero band.
    let fractional = surface(2, 3, [0; 3], 1, &[-1, 1, -1, 1]);
    for (height, clearance, count) in [
        ([0, 0], [i64::MIN, i64::MAX], 0),
        ([-1, 0], [i64::MIN, i64::MAX], 30),
        ([i64::MIN, i64::MAX], [0, 0], 3),
        ([i64::MIN, i64::MAX], [-1, 0], 12),
        ([i64::MIN, i64::MAX], [0, 1], 12),
    ] {
        let result = run(
            &ctx,
            &mut gpu,
            cfg,
            region([0, -2, 0], [3, 3, 3]),
            &f,
            &fractional,
            HeightfieldScatterFilters {
                surface_height_mm: height,
                clearance_mm: clearance,
                ..unrestricted()
            },
        );
        assert_eq!(
            result.accepted.len(),
            count,
            "height={height:?}, clearance={clearance:?}"
        );
    }

    // Only b-c-d is sloped; the equality diagonal is owned by flat a-c-b.
    let noncoplanar = surface(2, 4, [-2, 0, -2], 1, &[0, 0, 0, 4]);
    let result = run(
        &ctx,
        &mut gpu,
        cfg,
        region([-2, 0, -2], [2, 1, 2]),
        &f,
        &noncoplanar,
        HeightfieldScatterFilters {
            max_slope_rise: 0,
            ..unrestricted()
        },
    );
    assert_eq!(result.accepted.len(), 13);
    for (point, _) in &result.accepted {
        assert!(point.position_mm[0] + point.position_mm[2] <= 0);
    }
    assert!(result
        .accepted
        .iter()
        .any(|(p, _)| p.position_mm == [0, 0, 0]));
    assert_eq!(result.terrain_rejected, 3);

    // A real multi-quad payload also probes positive-side internal grid-line ownership.
    let multiple = surface(3, 2, [0; 3], 1, &[0, 0, 0, 0, 0, 2, 0, 2, 0]);
    let result = run(
        &ctx,
        &mut gpu,
        cfg,
        region([0; 3], [4, 1, 4]),
        &f,
        &multiple,
        HeightfieldScatterFilters {
            max_slope_rise: 0,
            ..unrestricted()
        },
    );
    assert!(!result.accepted.is_empty() && result.terrain_rejected != 0);
}

#[test]
fn exact_slope_endpoints_signed_normals_and_wide_integer_overflow_guards() {
    let ctx = gpu_context();
    let mut gpu = new_gpu(&ctx, 100, 0, 4);
    let f = constant(u16::MAX);
    for (spacing, heights, rise, slope_run, accept) in [
        (5, [0, 3, 4, 7], 1, 1, true),
        (5, [0, 3, 4, 7], u32::MAX - 1, u32::MAX, false),
        (5, [0, 3, 4, 7], 4, 5, false),
        (10, [0, 3, 4, 7], 1, 2, true),
        (10, [0, 3, 4, 7], 2_147_483_647, u32::MAX, false),
        (3, [0, 1, 1, 2], 1, 2, true),
        (3, [0, 1, 1, 2], 1, 3, false),
        (3, [2, 1, 1, 0], 1, 2, true),
        (3, [2, 1, 1, 0], 0, u32::MAX, false),
        (3, [2, 2, 2, 2], 0, u32::MAX, true),
    ] {
        let terrain = surface(2, spacing, [0; 3], 1, &heights);
        let result = run(
            &ctx,
            &mut gpu,
            config(1, 0),
            region([0; 3], [i64::from(spacing), 1, i64::from(spacing)]),
            &f,
            &terrain,
            HeightfieldScatterFilters {
                max_slope_rise: rise,
                max_slope_run: slope_run,
                ..unrestricted()
            },
        );
        assert_eq!(
            result.accepted.len(),
            if accept {
                spacing as usize * spacing as usize
            } else {
                0
            },
            "spacing={spacing}, slope={rise}/{slope_run}, heights={heights:?}"
        );
    }

    let max = i64::from(u32::MAX);
    let exact_scale = max * (max / 3);
    let overflow_scale = i64::MAX / 2 + 2;
    for (scale, heights, slope_run, accept) in [
        // dx = spacing*rise = (2^32-1)^2: near-max u128 square equality.
        (
            exact_scale,
            [-exact_scale, i64::MAX, -exact_scale, i64::MAX],
            1,
            true,
        ),
        (exact_scale, [-exact_scale, i64::MAX, 0, i64::MAX], 1, false),
        // Both component squares fit u128; their sum does not. Must not wrap.
        (
            overflow_scale,
            [-overflow_scale, i64::MAX, i64::MAX, -overflow_scale],
            1,
            false,
        ),
        // Decoded normals exceed i64; component*run rejects before squaring.
        (
            i64::MAX,
            [-i64::MAX, i64::MAX, i64::MAX, -i64::MAX],
            u32::MAX,
            false,
        ),
    ] {
        let terrain = surface(2, u32::MAX, [0; 3], scale, &heights);
        let result = run(
            &ctx,
            &mut gpu,
            config(1, 0),
            region([0; 3], [1; 3]),
            &f,
            &terrain,
            HeightfieldScatterFilters {
                max_slope_rise: u32::MAX,
                max_slope_run: slope_run,
                ..unrestricted()
            },
        );
        assert_eq!(
            result.accepted.len(),
            usize::from(accept),
            "scale={scale}, run={slope_run}"
        );
    }
}

#[test]
fn world_i64_edges_u32_spacing_and_decoded_heights_beyond_i64_are_exact() {
    let ctx = gpu_context();
    let mut gpu = new_gpu(&ctx, 129, 1, 4);
    let full = constant(u16::MAX);
    for origin in [[i64::MIN; 3], [i64::MAX - 32; 3]] {
        let terrain = surface(2, 32, origin, 1, &[0; 4]);
        let bounds = region(origin, origin.map(|v| v + 32));
        for cfg in [config(1, 0), config(17, 0)] {
            // Spacing-one is restricted to a two-cube so no enormous request is made.
            let bounds = if cfg.grid_spacing_mm == 1 {
                region(origin, origin.map(|v| v + 2))
            } else {
                bounds
            };
            let noisy = field(
                48_000,
                Some(DensityNoise {
                    cell_size_mm: u32::MAX,
                    amplitude: 12_345,
                    warp_cell_size_mm: u32::MAX,
                    warp_amplitude_mm: u32::MAX,
                }),
                vec![],
            );
            run(
                &ctx,
                &mut gpu,
                cfg,
                bounds,
                &noisy,
                &terrain,
                unrestricted(),
            );
        }
    }
    let terrain = surface(2, u32::MAX, [0; 3], 1, &[0; 4]);
    let result = run(
        &ctx,
        &mut gpu,
        config(u32::MAX, 0),
        region([0; 3], [i64::from(u32::MAX); 3]),
        &full,
        &terrain,
        unrestricted(),
    );
    assert_eq!(result.accepted.len(), 1);

    // Valid bake inputs can decode above i64 due to quantization, without a Y embedding offset.
    let scale = i64::MAX / 2 + 1;
    let decoded = surface(2, u32::MAX, [0; 3], scale, &[i64::MAX; 4]);
    assert!(decoded.tile().height_mm([0, 0]).unwrap() > i128::from(i64::MAX));
    let result = run(
        &ctx,
        &mut gpu,
        config(1, 0),
        region([0; 3], [1; 3]),
        &full,
        &decoded,
        unrestricted(),
    );
    assert_eq!(result.terrain_rejected, 1);
    for (y, height) in [(i64::MAX, 1), (i64::MIN, -1)] {
        let terrain = surface(2, u32::MAX, [i64::MIN, y, i64::MIN], 1, &[height; 4]);
        let result = run(
            &ctx,
            &mut gpu,
            config(1, 0),
            region([i64::MIN; 3], [i64::MIN + 1; 3]),
            &full,
            &terrain,
            unrestricted(),
        );
        assert_eq!(
            result.terrain_rejected, 1,
            "world height outside closed i64 bounds"
        );
    }
    for (height, point_y) in [(i64::MAX, i64::MIN), (i64::MIN, i64::MAX - 1)] {
        let terrain = surface(2, 1, [0, height, 0], 1, &[0; 4]);
        let result = run(
            &ctx,
            &mut gpu,
            config(1, 0),
            region([0, point_y, 0], [1, point_y + 1, 1]),
            &full,
            &terrain,
            unrestricted(),
        );
        assert_eq!(
            result.terrain_rejected, 1,
            "clearance does not wrap into i64"
        );
    }
}

#[test]
fn structural_cache_keys_equal_revision_content_edits_and_public_rebake_snapshots() {
    let ctx = gpu_context();
    let mut gpu = new_gpu(&ctx, 8, 0, 4);
    let cfg = config(1, 0);
    let bounds = region([0; 3], [2; 3]);
    let f = constant(u16::MAX);
    let filters = HeightfieldScatterFilters {
        clearance_mm: [0, 0],
        ..unrestricted()
    };
    let old = surface(2, 2, [0; 3], 1, &[0; 4]);
    let equal = surface(2, 2, [0; 3], 1, &[0; 4]);
    let independent = surface(2, 2, [0; 3], 1, &[1; 4]);
    assert!(!Arc::ptr_eq(old.tile(), equal.tile()));
    assert_eq!(old, equal);
    assert_eq!(old.tile().tile(), independent.tile().tile());
    assert_eq!(old.tile().revision(), independent.tile().revision());
    let mut pending = Vec::new();
    upload(&mut gpu, cfg, bounds, &f, &old, filters);
    let retained: HeightfieldScatterCacheKey = gpu.cache_key().unwrap().clone();
    pending.push((submit_capture(&ctx, &gpu), old.clone(), filters));
    upload(&mut gpu, cfg, bounds, &f, &equal, filters);
    assert_eq!(gpu.cache_key(), Some(&retained));
    pending.push((submit_capture(&ctx, &gpu), equal, filters));
    upload(&mut gpu, cfg, bounds, &f, &independent, filters);
    assert_ne!(gpu.cache_key(), Some(&retained));
    assert_eq!(gpu.cache_key().unwrap().source, retained.source);
    pending.push((submit_capture(&ctx, &gpu), independent, filters));

    // Explicitly clone the underlying tile into a separate owner before editing.
    let mut owner = old.tile().as_ref().clone();
    assert_ne!(owner.samples().as_ptr(), old.tile().samples().as_ptr());
    let full = DirtyRegion::new(owner.tile(), 0, 0, u16::MAX, u16::MAX).unwrap();
    let limits = BakeLimits {
        max_samples: 4,
        max_bytes: 8,
    };
    let change = owner
        .rebake(
            full,
            &GridHeights {
                side: 2,
                heights: &[1; 4],
            },
            limits,
        )
        .unwrap();
    assert_eq!(change.revision(), old.tile().revision() + 1);
    let before_failure = owner.clone();
    assert_eq!(
        owner.rebake(
            full,
            &GridHeights {
                side: 2,
                heights: &[2, 2, 2, i64::MAX]
            },
            limits
        ),
        Err(BakeError::HeightOutOfRange)
    );
    assert_eq!(owner, before_failure);
    let published = PlanarHeightfieldSurface::new(Arc::new(owner), old.embedding()).unwrap();
    assert_eq!(published.embedding(), old.embedding());
    upload(&mut gpu, cfg, bounds, &f, &published, filters);
    let new_key = gpu.cache_key().unwrap().clone();
    assert_ne!(new_key, retained);
    assert_eq!(new_key.source, retained.source);
    pending.push((submit_capture(&ctx, &gpu), published.clone(), filters));
    drop(old);
    assert_eq!(retained.surface.tile().samples(), &[0; 4]);
    assert_eq!(new_key.surface.tile().samples(), &[1; 4]);

    // Restore the old immutable snapshot after both independent and rebaked edits.
    upload(&mut gpu, cfg, bounds, &f, &retained.surface, filters);
    assert_eq!(gpu.cache_key(), Some(&retained));
    pending.push((
        submit_capture(&ctx, &gpu),
        retained.surface.clone(),
        filters,
    ));

    // Inactive endpoints and the world embedding are still structural key inputs.
    for changed in [
        HeightfieldScatterFilters {
            surface_height_mm: [i64::MIN + 1, i64::MAX],
            ..filters
        },
        HeightfieldScatterFilters {
            surface_height_mm: [i64::MIN, i64::MAX - 1],
            ..filters
        },
        HeightfieldScatterFilters {
            max_slope_rise: u32::MAX - 1,
            ..filters
        },
        HeightfieldScatterFilters {
            max_slope_run: 2,
            ..filters
        },
    ] {
        upload(&mut gpu, cfg, bounds, &f, &retained.surface, changed);
        assert_ne!(gpu.cache_key(), Some(&retained));
        pending.push((
            submit_capture(&ctx, &gpu),
            retained.surface.clone(),
            changed,
        ));
    }
    let moved = PlanarHeightfieldSurface::new(
        retained.surface.tile().clone(),
        PlanarHeightfieldEmbedding {
            sample_origin_mm: [0, 1, 0],
            ..retained.surface.embedding()
        },
    )
    .unwrap();
    upload(&mut gpu, cfg, bounds, &f, &moved, filters);
    assert_ne!(gpu.cache_key(), Some(&retained));
    pending.push((submit_capture(&ctx, &gpu), moved, filters));

    // Every request above was submitted before any readback or host wait.
    for (index, (buffer, terrain, selection)) in pending.into_iter().enumerate() {
        let snapshot = assert_snapshot(&read(&ctx, &buffer), cfg, bounds, &f, &terrain, selection);
        assert_eq!(snapshot.accepted.len(), 4);
        let y = if [2, 3, 9].contains(&index) { 1 } else { 0 };
        assert!(snapshot
            .accepted
            .iter()
            .all(|(point, _)| point.position_mm[1] == y));
    }
}

#[test]
fn density_attributes_source_distance_and_arbitrary_xyz_partition_union() {
    let ctx = gpu_context();
    let mut gpu = new_gpu(&ctx, 1_521, 2, 9);
    let cfg = config(1, 2);
    let f = painted();
    let terrain = surface(3, 8, [-8, 0, -8], 1, &[0, 2, 4, -2, 0, 2, -4, -2, 0]);
    let whole = region([-6, -4, -6], [7, 5, 7]);
    let filters = HeightfieldScatterFilters {
        surface_height_mm: [-2, 2],
        clearance_mm: [-2, 2],
        max_slope_rise: 1,
        max_slope_run: 2,
    };
    let expected = run(&ctx, &mut gpu, cfg, whole, &f, &terrain, filters);
    assert!(!expected.accepted.is_empty());
    assert!(
        expected.source_rejected != 0
            && expected.field_rejected != 0
            && expected.terrain_rejected != 0
    );
    let as_map = |snapshot: Snapshot| {
        snapshot
            .accepted
            .into_iter()
            .map(|(p, a)| (p.key, (p, a)))
            .collect::<BTreeMap<_, _>>()
    };
    let expected = as_map(expected);
    let mut parts = Vec::new();
    let cuts = [[-6, -2, 3, 7], [-4, -1, 2, 5], [-6, -3, 1, 7]];
    for x in cuts[0].windows(2).rev() {
        for y in cuts[1].windows(2).rev() {
            for z in cuts[2].windows(2).rev() {
                let bounds = region([x[0], y[0], z[0]], [x[1], y[1], z[1]]);
                upload(&mut gpu, cfg, bounds, &f, &terrain, filters);
                parts.push((submit_capture(&ctx, &gpu), bounds));
            }
        }
    }
    let mut union = BTreeMap::new();
    for (buffer, bounds) in parts {
        let part = assert_snapshot(&read(&ctx, &buffer), cfg, bounds, &f, &terrain, filters);
        for (key, value) in as_map(part) {
            assert!(union.insert(key, value).is_none());
        }
    }
    assert_eq!(
        union, expected,
        "all structural identities, positions and attrs across 27 reverse XYZ partitions"
    );
    for (axis, axis_cuts) in cuts.iter().enumerate() {
        let mut union = BTreeMap::new();
        let mut pending = Vec::new();
        for range in axis_cuts.windows(2).rev() {
            let mut bounds = whole;
            bounds.min_mm[axis] = range[0];
            bounds.max_mm[axis] = range[1];
            upload(&mut gpu, cfg, bounds, &f, &terrain, filters);
            pending.push((submit_capture(&ctx, &gpu), bounds));
        }
        for (buffer, bounds) in pending {
            let part = assert_snapshot(&read(&ctx, &buffer), cfg, bounds, &f, &terrain, filters);
            for (key, value) in as_map(part) {
                assert!(union.insert(key, value).is_none());
            }
        }
        assert_eq!(union, expected, "partition axis {axis}");
    }
}

struct Allocations {
    buffers: [wgpu::Buffer; 4],
    bytes: u64,
    height_scratch: usize,
    brush_scratch: usize,
}

impl Allocations {
    fn save(gpu: &GpuScatterHeightfield, capacity: usize, brushes: usize, samples: usize) -> Self {
        assert_eq!(
            gpu.allocated_buffer_bytes(),
            logical_bytes(capacity, brushes, samples)
        );
        assert!(gpu.height_scratch_bytes() >= samples * 4);
        assert!(gpu.brush_scratch_bytes() >= brushes * 64);
        Self {
            buffers: [
                gpu.records().clone(),
                gpu.attributes().clone(),
                gpu.compact_indices().clone(),
                gpu.indirect_args().clone(),
            ],
            bytes: gpu.allocated_buffer_bytes(),
            height_scratch: gpu.height_scratch_bytes(),
            brush_scratch: gpu.brush_scratch_bytes(),
        }
    }

    fn assert_unchanged(&self, gpu: &GpuScatterHeightfield) {
        assert_eq!(gpu.records(), &self.buffers[0]);
        assert_eq!(gpu.attributes(), &self.buffers[1]);
        assert_eq!(gpu.compact_indices(), &self.buffers[2]);
        assert_eq!(gpu.indirect_args(), &self.buffers[3]);
        assert_eq!(gpu.allocated_buffer_bytes(), self.bytes);
        assert_eq!(gpu.height_scratch_bytes(), self.height_scratch);
        assert_eq!(gpu.brush_scratch_bytes(), self.brush_scratch);
    }
}

#[test]
fn tails_empty_axes_repeated_encode_reset_counts_and_sequential_submits_need_no_wait() {
    let ctx = gpu_context();
    let mut gpu = new_gpu(&ctx, 129, 2, 9);
    let allocations = Allocations::save(&gpu, 129, 2, 9);
    assert_eq!(gpu.cache_key(), None);
    assert_eq!(gpu.candidate_count(), 0);
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    assert_eq!(
        gpu.encode(&mut encoder),
        Err(GpuScatterHeightfieldError::Field(
            GpuScatterFieldError::Scatter(GpuScatterError::RequestNotUploaded)
        ))
    );
    let a = surface(2, 129, [0; 3], 1, &[0; 4]);
    let b = surface(3, 65, [0; 3], 1, &[1; 9]);
    let filters = HeightfieldScatterFilters {
        clearance_mm: [0, 0],
        ..unrestricted()
    };
    let mut pending = Vec::new();
    for (count, radius, density, terrain) in [
        (129, 0, 65_535, &a),
        (0, 0, 65_535, &b),
        (1, 0, 65_535, &a),
        (63, 0, 65_535, &a),
        (64, 0, 65_535, &a),
        (65, 0, 65_535, &a),
        (129, 0, 65_535, &b),
        (129, 2, 32_768, &a),
        (129, 0, 0, &a),
        (1, 0, 65_535, &a),
        (0, 0, 0, &b),
        (129, 0, 65_535, &a),
    ] {
        let cfg = config(1, radius);
        let bounds = region([0; 3], [count, 1, 1]);
        let f = constant(density);
        upload(&mut gpu, cfg, bounds, &f, terrain, filters);
        pending.push((submit_capture(&ctx, &gpu), cfg, bounds, f, terrain.clone()));
        allocations.assert_unchanged(&gpu);
    }
    let cfg = config(1, 0);
    let bounds = region([0; 3], [129, 1, 1]);
    let f = constant(32_768);
    upload(&mut gpu, cfg, bounds, &f, &a, filters);
    for _ in 0..2 {
        pending.push((
            submit_capture(&ctx, &gpu),
            cfg,
            bounds,
            f.clone(),
            a.clone(),
        ));
    }
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    for _ in 0..2 {
        gpu.encode(&mut encoder).unwrap();
        pending.push((
            capture(&ctx, &gpu, &mut encoder),
            cfg,
            bounds,
            f.clone(),
            a.clone(),
        ));
    }
    ctx.queue().submit([encoder.finish()]);
    let mut source_rejected = 0;
    for (buffer, cfg, bounds, f, terrain) in pending {
        let result = assert_snapshot(&read(&ctx, &buffer), cfg, bounds, &f, &terrain, filters);
        source_rejected += result.source_rejected;
    }
    assert!(
        source_rejected != 0,
        "previously live slots must be explicitly zeroed"
    );
    allocations.assert_unchanged(&gpu);

    let mut zero = new_gpu(&ctx, 0, 0, 4);
    let zero_allocations = Allocations::save(&zero, 0, 0, 4);
    // Entirely unrelated XZ coordinates need no surface coverage when ANY axis is empty.
    for axis in 0..3 {
        let min = [i64::MIN; 3];
        let mut max = [i64::MAX; 3];
        max[axis] = min[axis];
        let result = run(
            &ctx,
            &mut zero,
            config(1, u32::MAX),
            region(min, max),
            &constant(u16::MAX),
            &a,
            unrestricted(),
        );
        assert!(result.accepted.is_empty());
        zero_allocations.assert_unchanged(&zero);
    }
}

#[test]
fn all_preflight_failures_retain_previous_request_key_allocations_and_gpu_outputs() {
    let ctx = gpu_context();
    let mut gpu = new_gpu(&ctx, 65, 2, 9);
    let allocations = Allocations::save(&gpu, 65, 2, 9);
    let cfg = config(1, 2);
    let bounds = region([0; 3], [65, 1, 1]);
    let old_field = painted();
    let old_surface = surface(2, 65, [0; 3], 1, &[0; 4]);
    let filters = unrestricted();
    upload(&mut gpu, cfg, bounds, &old_field, &old_surface, filters);
    let old_key = gpu.cache_key().unwrap().clone();
    let mut pending = vec![submit_capture(&ctx, &gpu)];
    let changed = ScatterConfig {
        seed: !cfg.seed,
        salt: u64::MAX,
        min_distance_mm: 0,
        ..cfg
    };
    let changed_field = field(
        0,
        None,
        vec![DensityBrush {
            id: 1,
            center_mm: [0; 3],
            inner_radius_mm: 1,
            outer_radius_mm: u32::MAX,
            opacity: u16::MAX,
            density: DensityBrushOp::Add(65_535),
            species: Some(u32::MAX),
        }],
    );
    let changed_surface = surface(3, 33, [0, 10, 0], 1, &[2; 9]);
    let too_many_samples = surface(4, 22, [0; 3], 1, &[3; 16]);
    let too_many_brushes = field(
        65_535,
        None,
        (0..3)
            .map(|id| DensityBrush {
                id,
                center_mm: [0; 3],
                inner_radius_mm: 0,
                outer_radius_mm: 1,
                opacity: u16::MAX,
                density: DensityBrushOp::Add(-65_535),
                species: Some(0),
            })
            .collect(),
    );
    let field_work = 65 * changed_field.work_per_sample();
    let source_work = estimate_scatter_work(changed, bounds)
        .unwrap()
        .max_candidate_evaluations;
    let cases = [
        (
            changed,
            bounds,
            source_work,
            &changed_field,
            field_work,
            &changed_surface,
            HeightfieldScatterFilters {
                surface_height_mm: [1, 0],
                ..filters
            },
            260,
            GpuScatterHeightfieldError::Terrain(HeightfieldScatterError::InvalidFilters),
        ),
        (
            changed,
            bounds,
            source_work,
            &changed_field,
            field_work,
            &changed_surface,
            HeightfieldScatterFilters {
                clearance_mm: [1, 0],
                ..filters
            },
            260,
            GpuScatterHeightfieldError::Terrain(HeightfieldScatterError::InvalidFilters),
        ),
        (
            changed,
            bounds,
            source_work,
            &changed_field,
            field_work,
            &changed_surface,
            HeightfieldScatterFilters {
                max_slope_run: 0,
                ..filters
            },
            260,
            GpuScatterHeightfieldError::Terrain(HeightfieldScatterError::InvalidFilters),
        ),
        (
            changed,
            bounds,
            source_work,
            &changed_field,
            field_work,
            &changed_surface,
            filters,
            259,
            GpuScatterHeightfieldError::Terrain(
                HeightfieldScatterError::SurfaceWorkBudgetExceeded {
                    required: 260,
                    limit: 259,
                },
            ),
        ),
        (
            changed,
            region([-1, 0, 0], [64, 1, 1]),
            u64::MAX,
            &changed_field,
            u64::MAX,
            &changed_surface,
            filters,
            u64::MAX,
            GpuScatterHeightfieldError::Terrain(HeightfieldScatterError::Surface(
                HeightfieldQueryError::OutsideSurface,
            )),
        ),
        (
            changed,
            bounds,
            source_work,
            &changed_field,
            field_work,
            &too_many_samples,
            filters,
            260,
            GpuScatterHeightfieldError::SurfaceCapacityExceeded {
                required: 16,
                capacity: 9,
            },
        ),
        (
            changed,
            region([0; 3], [66, 1, 1]),
            u64::MAX,
            &changed_field,
            u64::MAX,
            &changed_surface,
            filters,
            u64::MAX,
            GpuScatterHeightfieldError::Field(GpuScatterFieldError::Scatter(
                GpuScatterError::CapacityExceeded {
                    required: 66,
                    capacity: 65,
                },
            )),
        ),
        (
            changed,
            bounds,
            source_work - 1,
            &changed_field,
            field_work,
            &changed_surface,
            filters,
            260,
            GpuScatterHeightfieldError::Field(GpuScatterFieldError::Scatter(
                GpuScatterError::Scatter(ScatterError::CandidateBudgetExceeded {
                    required: source_work,
                    limit: source_work - 1,
                }),
            )),
        ),
        (
            changed,
            bounds,
            source_work,
            &changed_field,
            field_work - 1,
            &changed_surface,
            filters,
            260,
            GpuScatterHeightfieldError::Field(GpuScatterFieldError::Field(
                ScatterFieldError::WorkBudgetExceeded {
                    required: field_work,
                    limit: field_work - 1,
                },
            )),
        ),
        (
            changed,
            bounds,
            source_work,
            &too_many_brushes,
            u64::MAX,
            &changed_surface,
            filters,
            260,
            GpuScatterHeightfieldError::Field(GpuScatterFieldError::BrushCapacityExceeded {
                required: 3,
                capacity: 2,
            }),
        ),
        (
            ScatterConfig {
                grid_spacing_mm: 0,
                ..changed
            },
            bounds,
            u64::MAX,
            &changed_field,
            u64::MAX,
            &changed_surface,
            filters,
            u64::MAX,
            GpuScatterHeightfieldError::Terrain(HeightfieldScatterError::Source(
                ScatterFieldError::Scatter(ScatterError::InvalidGridSpacing),
            )),
        ),
        (
            changed,
            ScatterRegion {
                min_mm: [1, 0, 0],
                max_mm: [0, 1, 1],
            },
            u64::MAX,
            &changed_field,
            u64::MAX,
            &changed_surface,
            filters,
            u64::MAX,
            GpuScatterHeightfieldError::Terrain(HeightfieldScatterError::Source(
                ScatterFieldError::Scatter(ScatterError::InvalidRegion),
            )),
        ),
    ];
    for (request, bounds, source_limit, f, field_limit, terrain, selection, reads, error) in cases {
        assert_eq!(
            gpu.write_request(
                request,
                bounds,
                source_limit,
                f,
                field_limit,
                terrain,
                selection,
                HeightfieldScatterBudget {
                    max_sample_reads: reads
                }
            ),
            Err(error)
        );
        assert_eq!(gpu.cache_key(), Some(&old_key));
        assert_eq!(gpu.candidate_count(), 65);
        allocations.assert_unchanged(&gpu);
        // First inspect existing outputs without encoding; then encode the old
        // request again to expose any partially uploaded uniforms/brushes/heights.
        let mut encoder = ctx.device().create_command_encoder(&Default::default());
        pending.push(capture(&ctx, &gpu, &mut encoder));
        gpu.encode(&mut encoder).unwrap();
        pending.push(capture(&ctx, &gpu, &mut encoder));
        ctx.queue().submit([encoder.finish()]);
    }
    // Zero density cannot hide either full-request coverage or primary*4 budget admission.
    let zero = constant(0);
    for (bounds, reads, error) in [
        (
            bounds,
            259,
            HeightfieldScatterError::SurfaceWorkBudgetExceeded {
                required: 260,
                limit: 259,
            },
        ),
        (
            region([0; 3], [65, 1, 67]),
            u64::MAX,
            HeightfieldScatterError::Surface(HeightfieldQueryError::OutsideSurface),
        ),
    ] {
        assert_eq!(
            gpu.write_request(
                changed,
                bounds,
                u64::MAX,
                &zero,
                u64::MAX,
                &changed_surface,
                filters,
                HeightfieldScatterBudget {
                    max_sample_reads: reads
                }
            ),
            Err(GpuScatterHeightfieldError::Terrain(error))
        );
        assert_eq!(gpu.cache_key(), Some(&old_key));
        pending.push(submit_capture(&ctx, &gpu));
    }
    for buffer in pending {
        let result = assert_snapshot(
            &read(&ctx, &buffer),
            cfg,
            bounds,
            &old_field,
            &old_surface,
            filters,
        );
        assert!(
            !result.accepted.is_empty(),
            "old transaction must have meaningful output to preserve"
        );
    }
    allocations.assert_unchanged(&gpu);

    // Overflow/budget constructor failures must be checked before allocation;
    // these are never enormous real GPU dispatches or successful large reserves.
    let bytes = logical_bytes(65, 2, 9);
    assert!(
        matches!(GpuScatterHeightfield::new(&ctx, 65, 2, 9, 3, bytes - 1),
        Err(GpuScatterHeightfieldError::Field(GpuScatterFieldError::Scatter(
            GpuScatterError::BufferBudgetExceeded { required, limit })))
        if required == bytes && limit == bytes - 1)
    );
    assert!(matches!(
        GpuScatterHeightfield::new(&ctx, usize::MAX, 0, 4, 3, u64::MAX),
        Err(GpuScatterHeightfieldError::Field(
            GpuScatterFieldError::Scatter(GpuScatterError::ArithmeticOverflow)
        ))
    ));
    assert!(matches!(
        GpuScatterHeightfield::new(&ctx, 1, 0, usize::MAX, 3, u64::MAX),
        Err(GpuScatterHeightfieldError::Field(
            GpuScatterFieldError::Scatter(GpuScatterError::ArithmeticOverflow)
        ))
    ));
    let mut no_samples = new_gpu(&ctx, 1, 0, 0);
    assert_eq!(
        no_samples.write_request(
            config(1, 0),
            region([0; 3], [1; 3]),
            1,
            &zero,
            zero.work_per_sample(),
            &old_surface,
            filters,
            HeightfieldScatterBudget {
                max_sample_reads: 4
            }
        ),
        Err(GpuScatterHeightfieldError::SurfaceCapacityExceeded {
            required: 4,
            capacity: 0
        })
    );
    assert_eq!(no_samples.cache_key(), None);
    assert_eq!(no_samples.candidate_count(), 0);
}

const DRAW_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> records: array<u32>;
@group(0) @binding(1) var<storage, read> attributes: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> compact: array<u32>;
struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) color: vec4<f32>,
};
@vertex fn vs(@builtin(vertex_index) vertex: u32,
              @builtin(instance_index) instance: u32) -> VertexOut {
    let slot = compact[instance];
    let base = slot * 16u;
    let attr = attributes[slot];
    // Only this raster consumer fixture is a small positive spacing-one XZ lattice.
    let x = records[base + 6u];
    let z = records[base + 10u];
    let offsets = array<vec2<f32>, 3>(vec2(-3.0, -3.0), vec2(3.0, -3.0), vec2(0.0, 3.0));
    let pixel = vec2<f32>(f32(x) * 8.0 + 4.5, f32(z) * 8.0 + 4.5) + offsets[vertex];
    var out: VertexOut;
    out.position = vec4(pixel.x / 32.0 - 1.0, 1.0 - pixel.y / 32.0, 0.0, 1.0);
    let blue = (1u + (attr.w & 127u)) * attr.z * records[base + 12u];
    out.color = vec4(f32(attr.y & 255u), f32(attr.x >> 8u), f32(blue), 255.0) / 255.0;
    return out;
}
@fragment fn fs(input: VertexOut) -> @location(0) vec4<f32> { return input.color; }
"#;

struct DrawHarness {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    target: wgpu::Texture,
    pixels: wgpu::Buffer,
}

impl DrawHarness {
    fn new(ctx: &GpuContext, gpu: &GpuScatterHeightfield) -> Self {
        let device = ctx.device();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain test GPU stream consumer"),
            source: wgpu::ShaderSource::Wgsl(DRAW_SHADER.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain generated records + attributes + compact indirect raster"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("unmodified production GPU terrain output buffers"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu.records().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gpu.attributes().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gpu.compact_indices().as_entire_binding(),
                },
            ],
        });
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain scatter 8x8 lattice raster"),
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        Self {
            pipeline,
            bind_group,
            target,
            pixels: staging(ctx, 64 * 256),
        }
    }

    fn draw(&self, ctx: &GpuContext, gpu: &GpuScatterHeightfield) -> (Vec<u8>, Vec<u8>) {
        let mut encoder = ctx.device().create_command_encoder(&Default::default());
        gpu.encode(&mut encoder).unwrap();
        let view = self.target.create_view(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain compute directly into GPU-count draw_indirect"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            // No CPU vertices, remapped instances, count readback or direct draw.
            pass.draw_indirect(gpu.indirect_args(), 0);
        }
        let stream = capture(ctx, gpu, &mut encoder);
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.pixels,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(64),
                },
            },
            wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
        );
        ctx.queue().submit([encoder.finish()]);
        (read(ctx, &self.pixels), read(ctx, &stream))
    }
}

#[test]
fn terrain_generated_streams_feed_real_draw_indirect_and_exact_pixels() {
    let ctx = gpu_context();
    let mut gpu = new_gpu(&ctx, 64, 2, 4);
    let draw = DrawHarness::new(&ctx, &gpu);
    let flat = surface(2, 8, [0; 3], 1, &[0; 4]);
    let noncoplanar = surface(2, 8, [0; 3], 1, &[0, 0, 0, 8]);
    let filters = HeightfieldScatterFilters {
        clearance_mm: [0, 0],
        max_slope_rise: 0,
        ..unrestricted()
    };
    for (terrain, f, radius, empty) in [
        (&flat, constant(65_535), 0, false),
        (&noncoplanar, constant(65_535), 0, false),
        (&noncoplanar, painted(), 0, false),
        (&noncoplanar, painted(), 2, false),
        (&flat, constant(0), 0, false),
        (&noncoplanar, constant(65_535), 0, true),
        (&noncoplanar, constant(65_535), 0, false),
    ] {
        let cfg = config(1, radius);
        let bounds = region([0; 3], [if empty { 0 } else { 8 }, 1, 8]);
        upload(&mut gpu, cfg, bounds, &f, terrain, filters);
        let (pixels, stream) = draw.draw(&ctx, &gpu);
        let result = assert_snapshot(&stream, cfg, bounds, &f, terrain, filters);
        if !empty && f.config().base_density == u16::MAX {
            assert_eq!(
                result.accepted.len(),
                if terrain == &flat { 64 } else { 43 }
            );
        }
        if radius == 2 {
            assert!(result.source_rejected != 0);
        }
        let expected: BTreeMap<_, _> = result
            .accepted
            .iter()
            .map(|(p, a)| {
                (
                    p.position_mm,
                    [
                        a.species as u8,
                        (a.density >> 8) as u8,
                        1 + (f.candidate_rank(p.key) & 127) as u8,
                        255,
                    ],
                )
            })
            .collect();
        for x in 0..8 {
            for z in 0..8 {
                let color = expected
                    .get(&[x as i64, 0, z as i64])
                    .copied()
                    .unwrap_or([0, 0, 0, 255]);
                let offset = ((z * 8 + 4) * 64 + x * 8 + 4) * 4;
                assert_eq!(
                    &pixels[offset..offset + 4],
                    &color,
                    "actual indirect pixel at XZ=({x},{z})"
                );
                // Gaps between candidate triangles must stay clear, including after reuse.
                let gap = ((z * 8) * 64 + x * 8) * 4;
                assert_eq!(&pixels[gap..gap + 4], &[0, 0, 0, 255]);
            }
        }
    }
}
