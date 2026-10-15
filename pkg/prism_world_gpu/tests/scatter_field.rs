//! Exact author-field parity on a physical GPU and a direct GPU-stream draw.
//! Test-only readback; no adapter skips, host-generated draw payloads or builds.

#![cfg(feature = "gpu")]

use std::collections::{BTreeMap, BTreeSet};

use prism_world_gpu::{GpuContext, GpuScatterError, GpuScatterField, GpuScatterFieldError};
use prism_world_residency::{
    estimate_scatter_work, pcg_hash_u32, scatter, DensityBrush, DensityBrushOp, DensityNoise,
    FieldScatterCacheKey, ScatterBudget, ScatterCacheKey, ScatterCandidateKey, ScatterConfig,
    ScatterDensityField, ScatterError, ScatterFieldBuildBudget, ScatterFieldConfig,
    ScatterFieldError, ScatterFieldSample, ScatterPoint, ScatterRegion, SectorCoord,
    SCATTER_FIELD_VERSION, SCATTER_KERNEL_VERSION,
};

fn gpu_context() -> GpuContext {
    let ctx = GpuContext::try_headless_with_limits(wgpu::Limits {
        max_storage_buffers_per_shader_stage: 5,
        ..wgpu::Limits::downlevel_defaults()
    })
    .expect("field scatter requires successful real-device acquisition with five storage buffers")
    .expect("field scatter requires an adapter; skipping is forbidden");
    let evidence = ctx.evidence();
    let actual = match evidence.backend.as_str() {
        "Metal" => wgpu::Backends::METAL,
        "Vulkan" => wgpu::Backends::VULKAN,
        "Dx12" => wgpu::Backends::DX12,
        other => panic!("unexpected native field scatter backend {other}"),
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
    assert!(ctx.device().limits().max_storage_buffers_per_shader_stage >= 5);
    eprintln!("[scatter_field] adapter={evidence:?}");
    ctx
}

fn config() -> ScatterConfig {
    ScatterConfig {
        seed: 0x1234_5678_9abc_def0,
        version: 7,
        salt: 91,
        grid_spacing_mm: 17,
        min_distance_mm: 0,
    }
}

fn field_config() -> ScatterFieldConfig {
    ScatterFieldConfig {
        seed: 0x1234_5678_9abc_def0,
        version: 7,
        base_density: 32_768,
        base_species: 91,
        noise: None,
    }
}

fn field(config: ScatterFieldConfig, brushes: Vec<DensityBrush>) -> ScatterDensityField {
    let budget = ScatterFieldBuildBudget {
        max_brushes: brushes.len(),
        max_brush_bytes: brushes.capacity() * size_of::<DensityBrush>(),
    };
    ScatterDensityField::new(config, brushes, budget).unwrap()
}

fn constant(density: u16) -> ScatterDensityField {
    field(
        ScatterFieldConfig {
            base_density: density,
            ..field_config()
        },
        vec![],
    )
}

fn brush(id: u64) -> DensityBrush {
    DensityBrush {
        id,
        center_mm: [0; 3],
        inner_radius_mm: 5,
        outer_radius_mm: 10,
        opacity: u16::MAX,
        density: DensityBrushOp::Replace(u16::MAX),
        species: Some(123),
    }
}

fn region(min: [i64; 3], max: [i64; 3]) -> ScatterRegion {
    ScatterRegion::new(min, max).unwrap()
}

fn logical_bytes(capacity: usize, brushes: usize) -> u64 {
    88 * capacity.max(1) as u64 + 64 * brushes.max(1) as u64 + 352
}

fn staging(ctx: &GpuContext, size: u64) -> wgpu::Buffer {
    ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("field scatter test readback only"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn read(ctx: &GpuContext, buffer: &wgpu::Buffer) -> Vec<u8> {
    let (send, recv) = std::sync::mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        send.send(result).expect("field scatter map receiver");
    });
    ctx.wait();
    recv.recv()
        .expect("field scatter map callback")
        .expect("field scatter readback mapping");
    let bytes = buffer.get_mapped_range(..).unwrap().to_vec();
    buffer.unmap();
    bytes
}

// Capture each request before a subsequent submission can overwrite it. Only
// candidate prefixes and the indirect-count compact prefix have defined data.
fn capture(
    ctx: &GpuContext,
    gpu: &GpuScatterField,
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

fn submit_capture(ctx: &GpuContext, gpu: &GpuScatterField) -> wgpu::Buffer {
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    gpu.encode(&mut encoder).unwrap();
    let output = capture(ctx, gpu, &mut encoder);
    ctx.queue().submit([encoder.finish()]);
    output
}

fn upload(
    gpu: &mut GpuScatterField,
    cfg: ScatterConfig,
    bounds: ScatterRegion,
    f: &ScatterDensityField,
) {
    let work = estimate_scatter_work(cfg, bounds).unwrap();
    gpu.write_request(
        cfg,
        bounds,
        work.max_candidate_evaluations,
        f,
        work.primary_candidates * f.work_per_sample(),
    )
    .unwrap();
    assert_eq!(u64::from(gpu.candidate_count()), work.primary_candidates);
    assert_eq!(
        gpu.cache_key(),
        Some(&FieldScatterCacheKey {
            scatter: ScatterCacheKey {
                kernel_version: SCATTER_KERNEL_VERSION,
                config: cfg,
                region: bounds
            },
            field_version: SCATTER_FIELD_VERSION,
            field: f.clone(),
        })
    );
}

fn cpu_source(cfg: ScatterConfig, bounds: ScatterRegion) -> Vec<ScatterPoint> {
    let work = estimate_scatter_work(cfg, bounds).unwrap();
    let n = usize::try_from(work.primary_candidates).unwrap();
    scatter(
        cfg,
        bounds,
        ScatterBudget {
            max_candidate_evaluations: work.max_candidate_evaluations,
            max_output_points: n,
            max_output_bytes: n * size_of::<ScatterPoint>(),
        },
    )
    .unwrap()
    .points
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

fn pair(words: &[u32], index: usize) -> u64 {
    u64::from(words[index]) | (u64::from(words[index + 1]) << 32)
}

#[derive(Debug)]
struct Snapshot {
    accepted: Vec<(ScatterPoint, ScatterFieldSample)>,
    source_rejected: usize,
    field_rejected: usize,
    invalid_positions: usize,
}

fn assert_snapshot(
    bytes: &[u8],
    cfg: ScatterConfig,
    bounds: ScatterRegion,
    f: &ScatterDensityField,
) -> Snapshot {
    let words: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let n = estimate_scatter_work(cfg, bounds)
        .unwrap()
        .primary_candidates as usize;
    assert_eq!(bytes.len(), 16 + 84 * n);
    let source = cpu_source(cfg, bounds);
    let source_keys: BTreeSet<_> = source.iter().map(|p| p.key).collect();
    let expected: Vec<_> = source
        .iter()
        .filter_map(|&point| {
            let sample = f.sample(point.position_mm);
            f.accepts(point.key, sample.density)
                .then_some((point, sample))
        })
        .collect();
    assert_eq!(
        &words[..4],
        &[3, expected.len() as u32, 0, 0],
        "field indirect args: config={cfg:?}, region={bounds:?}, field={f:?}"
    );
    let mut snapshot = Snapshot {
        accepted: Vec::new(),
        source_rejected: 0,
        field_rejected: 0,
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
                        "global XYZ, Z fastest at source slot {slot}"
                    );
                    assert_eq!(record[15], 0, "source padding {slot}");
                    let source_accepted = source_keys.contains(&key);
                    assert_eq!(
                        record[12],
                        u32::from(source_accepted),
                        "source CPU accept {key:?}"
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
                            assert_eq!(&record[6..12], &[0; 6], "unrepresentable position {key:?}");
                            assert!(!source_accepted);
                            snapshot.invalid_positions += 1;
                            None
                        }
                        Err(error) => panic!("unexpected candidate rebuild error {error:?}"),
                    };
                    if source_accepted {
                        let point = point.unwrap();
                        let sample = f.sample(point.position_mm);
                        let accepted = f.accepts(key, sample.density);
                        assert_eq!(attrs, &[u32::from(sample.density), sample.species,
                            u32::from(accepted), f.candidate_rank(key)],
                            "CPU sample/rank/accept at source slot {slot}, key={key:?}, field={f:?}");
                        if accepted {
                            snapshot.accepted.push((point, sample));
                            accepted_slots.insert(slot as u32);
                        } else {
                            snapshot.field_rejected += 1;
                        }
                    } else {
                        assert_eq!(
                            attrs, &[0; 4],
                            "ALL attributes of source-rejected slot {slot} must clear"
                        );
                        snapshot.source_rejected += 1;
                    }
                    slot += 1;
                }
            }
        }
        assert_eq!(slot, n);
    }
    assert_eq!(
        snapshot.accepted, expected,
        "full structural field survivor set and attributes"
    );
    let compact = &words[4 + 20 * n..];
    let mut compact_slots = BTreeSet::new();
    for &slot in &compact[..expected.len()] {
        assert!((slot as usize) < n, "compact slot outside primary prefix");
        assert!(
            compact_slots.insert(slot),
            "duplicate field compact slot {slot}"
        );
    }
    assert_eq!(
        compact_slots, accepted_slots,
        "exact unordered field compact slot set"
    );
    snapshot
}

fn run(
    ctx: &GpuContext,
    gpu: &mut GpuScatterField,
    cfg: ScatterConfig,
    bounds: ScatterRegion,
    f: &ScatterDensityField,
) -> Snapshot {
    upload(gpu, cfg, bounds, f);
    let output = submit_capture(ctx, gpu);
    assert_snapshot(&read(ctx, &output), cfg, bounds, f)
}

#[test]
fn density_endpoints_rank_thresholds_and_complete_candidate_namespace() {
    let ctx = gpu_context();
    let mut gpu = GpuScatterField::new(&ctx, 129, 0, 3, logical_bytes(129, 0)).unwrap();
    let cfg = ScatterConfig {
        grid_spacing_mm: 1,
        ..config()
    };
    let bounds = region([-64, -1, 0], [65, 0, 1]);
    for density in [0, 1, 32_767, 32_768, 65_534, 65_535] {
        let result = run(&ctx, &mut gpu, cfg, bounds, &constant(density));
        assert_eq!(result.source_rejected, 0);
        if density == 0 {
            assert_eq!(result.field_rejected, 129);
        } else if density == u16::MAX {
            assert_eq!(result.accepted.len(), 129);
        } else if density == 32_768 {
            assert!(
                !result.accepted.is_empty() && result.field_rejected != 0,
                "density threshold: accepted={}, field_rejected={}, source_rejected={}",
                result.accepted.len(),
                result.field_rejected,
                result.source_rejected
            );
        }
    }
    // Every structural key limb and the independent field seed reaches the
    // actual shader. One candidate per request keeps this deliberately tiny.
    for (seed, salt, version, spacing, origin, field_seed) in [
        (0, 0, 0, 1, -1, 0),
        (
            u64::MAX,
            u64::MAX,
            u32::MAX,
            17,
            -(1_i64 << 60) - 1,
            u64::MAX,
        ),
        (
            0x8000_0000_0000_0000,
            0x8000_0000_0000_0001,
            1,
            u32::MAX,
            1_i64 << 60,
            1,
        ),
    ] {
        let cfg = ScatterConfig {
            seed,
            salt,
            version,
            grid_spacing_mm: spacing,
            min_distance_mm: 0,
        };
        let key = ScatterCandidateKey {
            kernel_version: SCATTER_KERNEL_VERSION,
            seed,
            salt,
            version,
            grid_spacing_mm: spacing,
            grid: SectorCoord::new(origin.div_euclid(i64::from(spacing)), -1, 1),
        };
        let point = key.rebuild().unwrap();
        let f = field(
            ScatterFieldConfig {
                seed: field_seed,
                ..field_config()
            },
            vec![],
        );
        let bounds = region(point.position_mm, point.position_mm.map(|v| v + 1));
        let rank = u64::from(f.candidate_rank(key));
        let first = (rank / 65_537 + 1).min(u64::from(u16::MAX)) as u16;
        for density in [first - 1, first] {
            let f = field(
                ScatterFieldConfig {
                    base_density: density,
                    ..f.config()
                },
                vec![],
            );
            let result = run(&ctx, &mut gpu, cfg, bounds, &f);
            assert_eq!(gpu.candidate_count(), 1);
            assert_eq!(result.accepted.len(), usize::from(density == first));
        }
    }
    // Frozen exact equality distinguishes strict < from <= (random samples
    // almost never hit this), using the public CPU golden's full key.
    let cfg = config();
    let key = ScatterCandidateKey {
        kernel_version: SCATTER_KERNEL_VERSION,
        seed: cfg.seed,
        version: cfg.version,
        salt: cfg.salt,
        grid_spacing_mm: cfg.grid_spacing_mm,
        grid: SectorCoord::new(18_428, 4, 13),
    };
    assert_eq!(constant(0).candidate_rank(key), 2_411_630_526);
    assert_eq!(2_411_630_526_u64, 36_798 * 65_537);
    let position = key.rebuild().unwrap().position_mm;
    let bounds = region(position, position.map(|v| v + 1));
    assert!(run(&ctx, &mut gpu, cfg, bounds, &constant(36_798))
        .accepted
        .is_empty());
    assert_eq!(
        run(&ctx, &mut gpu, cfg, bounds, &constant(36_799))
            .accepted
            .len(),
        1
    );
}

#[test]
fn noise_warp_signed_coordinates_world_edges_and_u32_max_parameters() {
    let ctx = gpu_context();
    let mut gpu = GpuScatterField::new(&ctx, 64, 0, 3, logical_bytes(64, 0)).unwrap();
    let bands = [
        DensityNoise {
            cell_size_mm: 17,
            amplitude: 0,
            warp_cell_size_mm: 13,
            warp_amplitude_mm: u32::MAX,
        },
        DensityNoise {
            cell_size_mm: 17,
            amplitude: 30_000,
            warp_cell_size_mm: 13,
            warp_amplitude_mm: 0,
        },
        DensityNoise {
            cell_size_mm: 17,
            amplitude: 30_000,
            warp_cell_size_mm: 13,
            warp_amplitude_mm: 23,
        },
        DensityNoise {
            cell_size_mm: 1,
            amplitude: u16::MAX,
            warp_cell_size_mm: 1,
            warp_amplitude_mm: u32::MAX,
        },
        DensityNoise {
            cell_size_mm: u32::MAX,
            amplitude: u16::MAX,
            warp_cell_size_mm: u32::MAX,
            warp_amplitude_mm: u32::MAX,
        },
    ];
    let mut saw_field_accept = false;
    let mut saw_field_reject = false;
    let mut invalid_positions = 0;
    for (index, noise) in bands.into_iter().enumerate() {
        let f = field(
            ScatterFieldConfig {
                seed: if index % 2 == 0 {
                    field_config().seed
                } else {
                    u64::MAX
                },
                base_species: u32::MAX,
                noise: Some(noise),
                ..field_config()
            },
            vec![],
        );
        // Adjacent samples across zero, both u32 limbs, and beyond f64's exact
        // range. All requests have at most 4^3 candidates, not world-sized grids.
        for origin in [-18, -(1_i64 << 32) - 1, (1_i64 << 60) + 29] {
            let cfg = ScatterConfig {
                grid_spacing_mm: 1,
                ..config()
            };
            let result = run(
                &ctx,
                &mut gpu,
                cfg,
                region([origin, -2, -1], [origin + 4, 2, 3]),
                &f,
            );
            saw_field_accept |= !result.accepted.is_empty();
            saw_field_reject |= result.field_rejected != 0;
            if noise.amplitude == 0 {
                assert!(result
                    .accepted
                    .iter()
                    .all(|(_, sample)| sample.density == 32_768));
            }
        }
        // Widened virtual warp may leave the i64 domain. Check all eight
        // signed corner combinations, with non-unit jitter and clipped halos.
        for spacing in [17, u32::MAX] {
            let width = 2 * i64::from(spacing);
            let cfg = ScatterConfig {
                seed: u64::MAX,
                salt: u64::MAX - 1,
                grid_spacing_mm: spacing,
                min_distance_mm: if index % 2 == 0 { spacing } else { 0 },
                ..config()
            };
            for mask in 0..8 {
                let min = std::array::from_fn(|axis| {
                    if mask & (1 << axis) == 0 {
                        i64::MIN
                    } else {
                        i64::MAX - width
                    }
                });
                let max = std::array::from_fn(|axis| {
                    if mask & (1 << axis) == 0 {
                        i64::MIN + width
                    } else {
                        i64::MAX
                    }
                });
                let result = run(&ctx, &mut gpu, cfg, region(min, max), &f);
                invalid_positions += result.invalid_positions;
                saw_field_accept |= !result.accepted.is_empty();
                saw_field_reject |= result.field_rejected != 0;
            }
        }
    }
    assert!(
        saw_field_accept && saw_field_reject,
        "non-vacuous field filtering"
    );
    assert!(
        invalid_positions != 0,
        "fixture must reach source coordinate-overflow rejection"
    );
}

#[test]
fn brushes_squared_falloff_species_threshold_signed_rounding_and_stable_order() {
    let ctx = gpu_context();
    let mut gpu = GpuScatterField::new(&ctx, 32, 4, 3, logical_bytes(32, 4)).unwrap();
    let cfg = ScatterConfig {
        grid_spacing_mm: 1,
        ..config()
    };
    let line = region([-11, 0, 0], [12, 1, 1]);
    for opacity in [0, 32_767, 32_768, u16::MAX] {
        let f = field(
            ScatterFieldConfig {
                base_density: 0,
                ..field_config()
            },
            vec![DensityBrush {
                opacity,
                ..brush(1)
            }],
        );
        run(&ctx, &mut gpu, cfg, line, &f);
        if opacity == u16::MAX {
            for (x, expected) in [
                (0, 65_535),
                (5, 65_535),
                (6, 61_719),
                (7, 49_697),
                (8, 30_801),
                (9, 10_486),
                (10, 0),
                (11, 0),
            ] {
                assert_eq!(
                    f.sample([x, 0, 0]).density,
                    expected,
                    "squared-falloff golden"
                );
            }
        }
    }
    let singleton = region([0; 3], [1; 3]);
    for (base_density, density, opacity, expected) in [
        (10_000, DensityBrushOp::Add(-1), 32_768, 10_000),
        (10_000, DensityBrushOp::Add(1), 32_768, 10_000),
        (10_000, DensityBrushOp::Add(-3), 32_768, 9_999),
        (10_000, DensityBrushOp::Add(3), 32_768, 10_001),
        (10_000, DensityBrushOp::Add(-65_535), 65_535, 0),
        (10_000, DensityBrushOp::Add(65_535), 65_535, 65_535),
        (0, DensityBrushOp::Replace(65_535), 32_768, 32_768),
        (65_535, DensityBrushOp::Replace(0), 32_768, 32_767),
    ] {
        let f = field(
            ScatterFieldConfig {
                base_density,
                ..field_config()
            },
            vec![DensityBrush {
                density,
                opacity,
                ..brush(1)
            }],
        );
        assert_eq!(f.sample([0; 3]).density, expected);
        run(&ctx, &mut gpu, cfg, singleton, &f);
    }
    for opacity in [0, 32_767, 32_768, u16::MAX] {
        for species in [None, Some(0), Some(u32::MAX)] {
            let f = field(
                field_config(),
                vec![DensityBrush {
                    opacity,
                    density: DensityBrushOp::Add(0),
                    species,
                    ..brush(1)
                }],
            );
            assert_eq!(
                f.sample([0; 3]).species,
                if opacity >= 32_768 {
                    species.unwrap_or(91)
                } else {
                    91
                }
            );
            run(&ctx, &mut gpu, cfg, line, &f);
        }
    }
    // Unsorted stable IDs (including both 64-bit endpoints), per-stroke
    // saturation, and a final species-only edit. Reverse Vec order is identical.
    let strokes = vec![
        DensityBrush {
            density: DensityBrushOp::Add(7),
            species: Some(u32::MAX),
            ..brush(1_u64 << 32)
        },
        DensityBrush {
            density: DensityBrushOp::Add(0),
            species: Some(0),
            ..brush(u64::MAX)
        },
        DensityBrush {
            density: DensityBrushOp::Add(-65_535),
            species: None,
            ..brush(0)
        },
    ];
    let f = field(
        ScatterFieldConfig {
            base_density: 1,
            ..field_config()
        },
        strokes.clone(),
    );
    assert_eq!(
        f.sample([0; 3]),
        ScatterFieldSample {
            density: 7,
            species: 0
        }
    );
    let first = run(&ctx, &mut gpu, cfg, line, &f);
    let reversed = field(f.config(), strokes.into_iter().rev().collect());
    assert_eq!(f, reversed);
    assert_eq!(
        run(&ctx, &mut gpu, cfg, line, &reversed).accepted,
        first.accepted
    );
}

#[test]
fn brushes_u32_max_radius_world_corner_differences_and_hard_outer_boundary() {
    let ctx = gpu_context();
    let mut gpu = GpuScatterField::new(&ctx, 16, 2, 3, logical_bytes(16, 2)).unwrap();
    let cfg = ScatterConfig {
        grid_spacing_mm: 1,
        ..config()
    };
    for center in [[0; 3], [i64::MIN; 3], [i64::MAX; 3]] {
        for inner in [0, u32::MAX] {
            let f = field(
                ScatterFieldConfig {
                    base_density: 0,
                    ..field_config()
                },
                vec![DensityBrush {
                    center_mm: center,
                    inner_radius_mm: inner,
                    outer_radius_mm: u32::MAX,
                    species: Some(u32::MAX),
                    ..brush(1)
                }],
            );
            let near_center = center.map(|v| if v == i64::MAX { v - 1 } else { v });
            run(
                &ctx,
                &mut gpu,
                cfg,
                region(near_center, near_center.map(|v| v + 1)),
                &f,
            );
            // Radius-1 / radius / radius+1 along each axis, independently of
            // the huge world coordinates. Singleton requests avoid huge grids.
            for axis in 0..3 {
                for distance in [
                    0,
                    i64::from(u32::MAX) - 1,
                    i64::from(u32::MAX),
                    i64::from(u32::MAX) + 1,
                ] {
                    // At the upper corner choose an axis-aligned displacement
                    // from a representable sample centre, keeping other axes exact.
                    let mut sample_center = center;
                    if center[0] == i64::MAX {
                        sample_center = [i64::MAX - 1; 3];
                    }
                    let mut p = sample_center;
                    p[axis] += if center[axis] == i64::MAX {
                        -distance
                    } else {
                        distance
                    };
                    let adjusted = field(
                        f.config(),
                        vec![DensityBrush {
                            center_mm: sample_center,
                            ..f.brushes()[0]
                        }],
                    );
                    if inner == u32::MAX {
                        assert_eq!(
                            adjusted.sample(p).density,
                            if distance < i64::from(u32::MAX) {
                                u16::MAX
                            } else {
                                0
                            }
                        );
                    }
                    run(&ctx, &mut gpu, cfg, region(p, p.map(|v| v + 1)), &adjusted);
                }
            }
            // Opposite corners exercise 65-bit differences / enormous square
            // sums. They must reject early, not wrap into a nearby stroke.
            let opposite = if center[0] == i64::MIN {
                [i64::MAX - 1; 3]
            } else {
                [i64::MIN; 3]
            };
            assert_eq!(f.sample(opposite).density, 0);
            run(
                &ctx,
                &mut gpu,
                cfg,
                region(opposite, opposite.map(|v| v + 1)),
                &f,
            );
        }
    }
    for radius in [1, 5] {
        let f = field(
            ScatterFieldConfig {
                base_density: 0,
                ..field_config()
            },
            vec![DensityBrush {
                inner_radius_mm: radius,
                outer_radius_mm: radius,
                ..brush(1)
            }],
        );
        run(&ctx, &mut gpu, cfg, region([-6, 0, 0], [7, 1, 1]), &f);
    }
}

fn assert_buffers(
    gpu: &GpuScatterField,
    buffers: &[wgpu::Buffer; 4],
    capacity: usize,
    brushes: usize,
    scratch: usize,
) {
    assert_eq!(
        gpu.records(),
        &buffers[0],
        "source record buffer was replaced"
    );
    assert_eq!(
        gpu.attributes(),
        &buffers[1],
        "attribute buffer was replaced"
    );
    assert_eq!(
        gpu.compact_indices(),
        &buffers[2],
        "field compact buffer was replaced"
    );
    assert_eq!(
        gpu.indirect_args(),
        &buffers[3],
        "field indirect buffer was replaced"
    );
    assert_eq!(
        gpu.allocated_buffer_bytes(),
        logical_bytes(capacity, brushes)
    );
    assert_eq!(
        gpu.brush_scratch_bytes(),
        scratch,
        "brush scratch reallocated"
    );
}

#[test]
fn tails_repeated_encode_and_sequential_submits_without_host_wait() {
    let ctx = gpu_context();
    const CAPACITY: usize = 129;
    let mut gpu = GpuScatterField::new(&ctx, CAPACITY, 2, 3, logical_bytes(CAPACITY, 2)).unwrap();
    let buffers = [
        gpu.records().clone(),
        gpu.attributes().clone(),
        gpu.compact_indices().clone(),
        gpu.indirect_args().clone(),
    ];
    let scratch = gpu.brush_scratch_bytes();
    assert!(scratch >= 2 * 64);
    assert_eq!(gpu.candidate_count(), 0);
    assert_eq!(gpu.cache_key(), None);
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    assert_eq!(
        gpu.encode(&mut encoder),
        Err(GpuScatterFieldError::Scatter(
            GpuScatterError::RequestNotUploaded
        ))
    );
    let cfg = ScatterConfig {
        grid_spacing_mm: 1,
        ..config()
    };
    let mut captures = Vec::new();
    // Absolutely no wait, map, CPU scatter, or readback between submissions.
    // Revisit full prefixes after empty/short requests and reject previously
    // accepted prefixes through both source thinning and field editing.
    for (count, radius, density) in [
        (129, 0, 65_535),
        (0, 0, 65_535),
        (1, 0, 65_535),
        (63, 0, 65_535),
        (64, 0, 65_535),
        (65, 0, 65_535),
        (129, 0, 65_535),
        (129, 2, 32_768),
        (129, 0, 0),
        (1, 0, 65_535),
        (0, 0, 0),
        (129, 0, 65_535),
    ] {
        let cfg = ScatterConfig {
            min_distance_mm: radius,
            ..cfg
        };
        let bounds = region([0; 3], [count, 1, 1]);
        let f = constant(density);
        upload(&mut gpu, cfg, bounds, &f);
        captures.push((submit_capture(&ctx, &gpu), cfg, bounds, f));
        assert_buffers(&gpu, &buffers, CAPACITY, 2, scratch);
    }
    let bounds = region([0; 3], [129, 1, 1]);
    let f = constant(32_768);
    upload(&mut gpu, cfg, bounds, &f);
    // Repeat unchanged request in separate submissions AND twice in one encoder.
    captures.push((submit_capture(&ctx, &gpu), cfg, bounds, f.clone()));
    captures.push((submit_capture(&ctx, &gpu), cfg, bounds, f.clone()));
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    for _ in 0..2 {
        gpu.encode(&mut encoder).unwrap();
        captures.push((capture(&ctx, &gpu, &mut encoder), cfg, bounds, f.clone()));
    }
    ctx.queue().submit([encoder.finish()]);
    let mut saw_source_rejected = false;
    for (buffer, cfg, bounds, f) in captures {
        let result = assert_snapshot(&read(&ctx, &buffer), cfg, bounds, &f);
        saw_source_rejected |= result.source_rejected != 0;
    }
    assert!(
        saw_source_rejected,
        "must overwrite previously live attributes with zero"
    );
    assert_buffers(&gpu, &buffers, CAPACITY, 2, scratch);

    // Zero logical capacity still supports empty regions at arbitrary axes,
    // even when the nonempty-domain source halo would overflow its work bound.
    let mut zero = GpuScatterField::new(&ctx, 0, 0, 3, logical_bytes(0, 0)).unwrap();
    for axis in 0..3 {
        let mut max = [i64::MAX; 3];
        max[axis] = i64::MIN;
        let empty_cfg = ScatterConfig {
            min_distance_mm: u32::MAX,
            ..cfg
        };
        run(
            &ctx,
            &mut zero,
            empty_cfg,
            region([i64::MIN; 3], max),
            &constant(65_535),
        );
    }
    assert_eq!(
        zero.write_request(cfg, region([0; 3], [1; 3]), 1, &constant(0), 1),
        Err(GpuScatterFieldError::Scatter(
            GpuScatterError::CapacityExceeded {
                required: 1,
                capacity: 0
            }
        ))
    );
    assert_eq!(zero.candidate_count(), 0);
}

#[test]
fn preflight_errors_preserve_old_source_field_brushes_and_encoded_results() {
    let ctx = gpu_context();
    const CAPACITY: usize = 65;
    const BRUSHES: usize = 2;
    let bytes = logical_bytes(CAPACITY, BRUSHES);
    let mut gpu = GpuScatterField::new(&ctx, CAPACITY, BRUSHES, 3, bytes).unwrap();
    let buffers = [
        gpu.records().clone(),
        gpu.attributes().clone(),
        gpu.compact_indices().clone(),
        gpu.indirect_args().clone(),
    ];
    let scratch = gpu.brush_scratch_bytes();
    let cfg = ScatterConfig {
        grid_spacing_mm: 1,
        min_distance_mm: 2,
        ..config()
    };
    let bounds = region([0; 3], [65, 1, 1]);
    let old = field(
        ScatterFieldConfig {
            noise: Some(DensityNoise {
                cell_size_mm: 17,
                amplitude: 30_000,
                warp_cell_size_mm: 13,
                warp_amplitude_mm: 23,
            }),
            ..field_config()
        },
        vec![
            brush(9),
            DensityBrush {
                center_mm: [32, 0, 0],
                density: DensityBrushOp::Add(-12_345),
                ..brush(2)
            },
        ],
    );
    upload(&mut gpu, cfg, bounds, &old);
    let old_key = gpu.cache_key().unwrap().clone();
    let changed = ScatterConfig {
        seed: !cfg.seed,
        salt: u64::MAX,
        min_distance_mm: 0,
        ..cfg
    };
    let changed_field = field(
        ScatterFieldConfig {
            seed: u64::MAX,
            version: u32::MAX,
            base_density: 0,
            base_species: u32::MAX,
            noise: None,
        },
        vec![DensityBrush {
            density: DensityBrushOp::Add(65_535),
            ..brush(1)
        }],
    );
    let source_work = estimate_scatter_work(changed, bounds)
        .unwrap()
        .max_candidate_evaluations;
    let field_work = 65 * changed_field.work_per_sample();
    let mut captures = vec![submit_capture(&ctx, &gpu)];
    let cases = [
        (
            changed,
            region([0; 3], [66, 1, 1]),
            u64::MAX,
            u64::MAX,
            GpuScatterFieldError::Scatter(GpuScatterError::CapacityExceeded {
                required: 66,
                capacity: CAPACITY,
            }),
        ),
        (
            ScatterConfig {
                grid_spacing_mm: 0,
                ..changed
            },
            bounds,
            u64::MAX,
            u64::MAX,
            GpuScatterFieldError::Scatter(GpuScatterError::Scatter(
                ScatterError::InvalidGridSpacing,
            )),
        ),
        (
            changed,
            ScatterRegion {
                min_mm: [1, 0, 0],
                max_mm: [0, 1, 1],
            },
            u64::MAX,
            u64::MAX,
            GpuScatterFieldError::Scatter(GpuScatterError::Scatter(ScatterError::InvalidRegion)),
        ),
        (
            changed,
            bounds,
            source_work - 1,
            u64::MAX,
            GpuScatterFieldError::Scatter(GpuScatterError::Scatter(
                ScatterError::CandidateBudgetExceeded {
                    required: source_work,
                    limit: source_work - 1,
                },
            )),
        ),
        (
            changed,
            bounds,
            source_work,
            field_work - 1,
            GpuScatterFieldError::Field(ScatterFieldError::WorkBudgetExceeded {
                required: field_work,
                limit: field_work - 1,
            }),
        ),
        (
            changed,
            region([i64::MIN; 3], [i64::MAX; 3]),
            u64::MAX,
            u64::MAX,
            GpuScatterFieldError::Scatter(GpuScatterError::Scatter(
                ScatterError::ArithmeticOverflow,
            )),
        ),
    ];
    // No host waits: if an error enqueues even one new uniform/brush upload,
    // the next encode+capture must expose it despite unchanged host metadata.
    for (request, region, source_limit, field_limit, error) in cases {
        assert_eq!(
            gpu.write_request(request, region, source_limit, &changed_field, field_limit),
            Err(error)
        );
        assert_eq!(gpu.cache_key(), Some(&old_key));
        assert_eq!(gpu.candidate_count(), CAPACITY as u32);
        captures.push(submit_capture(&ctx, &gpu));
        assert_buffers(&gpu, &buffers, CAPACITY, BRUSHES, scratch);
    }
    let too_many = field(changed_field.config(), vec![brush(1), brush(2), brush(3)]);
    assert_eq!(
        gpu.write_request(changed, bounds, source_work, &too_many, u64::MAX),
        Err(GpuScatterFieldError::BrushCapacityExceeded {
            required: 3,
            capacity: BRUSHES
        })
    );
    assert_eq!(gpu.cache_key(), Some(&old_key));
    captures.push(submit_capture(&ctx, &gpu));
    for buffer in captures {
        assert_snapshot(&read(&ctx, &buffer), cfg, bounds, &old);
    }
    assert_buffers(&gpu, &buffers, CAPACITY, BRUSHES, scratch);
    // Composite constructor budget checks are pre-allocation failures, not
    // giant real requests. Do not allocate an oversized candidate/brush buffer.
    assert!(
        matches!(GpuScatterField::new(&ctx, CAPACITY, BRUSHES, 3, bytes - 1),
        Err(GpuScatterFieldError::Scatter(GpuScatterError::BufferBudgetExceeded { required, limit }))
        if required == bytes && limit == bytes - 1)
    );
    assert!(matches!(
        GpuScatterField::new(&ctx, usize::MAX, 0, 3, u64::MAX),
        Err(GpuScatterFieldError::Scatter(
            GpuScatterError::ArithmeticOverflow
        ))
    ));
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
    // This consumer fixture alone is a spacing-1 positive lattice. Production
    // records retain global signed i64 low/high pairs, never float positions.
    let x = records[base + 6u];
    let y = records[base + 8u];
    let offsets = array<vec2<f32>, 3>(vec2(-3.0, -3.0), vec2(3.0, -3.0), vec2(0.0, 3.0));
    let pixel = vec2<f32>(f32(x) * 8.0 + 4.5, f32(y) * 8.0 + 4.5) + offsets[vertex];
    var out: VertexOut;
    out.position = vec4(pixel.x / 32.0 - 1.0, 1.0 - pixel.y / 32.0, 0.0, 1.0);
    let blue = (1u + (attr.w & 127u)) * attr.z * records[base + 12u];
    out.color = vec4(f32(attr.y & 255u), f32(attr.x >> 8u), f32(blue), 255.0) / 255.0;
    return out;
}

@fragment fn fs(input: VertexOut) -> @location(0) vec4<f32> {
    return input.color;
}
"#;

struct DrawHarness {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    target: wgpu::Texture,
    pixels: wgpu::Buffer,
}

impl DrawHarness {
    fn new(ctx: &GpuContext, gpu: &GpuScatterField) -> Self {
        let device = ctx.device();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("test-only field GPU stream consumer"),
            source: wgpu::ShaderSource::Wgsl(DRAW_SHADER.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("field records + attributes + compact indirect raster"),
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
            label: Some("field-generated buffers directly consumed by vertex stage"),
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
            label: Some("field indirect 8x8 lattice offscreen target"),
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

    fn draw(&self, ctx: &GpuContext, gpu: &GpuScatterField) -> (Vec<u8>, Vec<u8>) {
        let mut encoder = ctx.device().create_command_encoder(&Default::default());
        gpu.encode(&mut encoder).unwrap();
        let view = self.target.create_view(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("field actual GPU-count draw_indirect"),
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
            // Compute -> generated source/attributes/compact -> indirect draw,
            // one submission. No count readback, CPU vertices or direct draw.
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

fn pixel(bytes: &[u8], x: usize, y: usize) -> &[u8] {
    let offset = (y * 64 + x) * 4;
    &bytes[offset..offset + 4]
}

#[test]
fn generated_records_attributes_and_compact_feed_actual_draw_indirect_pixels() {
    let ctx = gpu_context();
    let mut gpu = GpuScatterField::new(&ctx, 64, 2, 3, logical_bytes(64, 2)).unwrap();
    let draw = DrawHarness::new(&ctx, &gpu);
    let painted = field(
        ScatterFieldConfig {
            base_species: 17,
            noise: Some(DensityNoise {
                cell_size_mm: 3,
                amplitude: 30_000,
                warp_cell_size_mm: 5,
                warp_amplitude_mm: 7,
            }),
            ..field_config()
        },
        vec![
            DensityBrush {
                center_mm: [6, 6, 0],
                inner_radius_mm: 1,
                outer_radius_mm: 3,
                density: DensityBrushOp::Add(-40_000),
                species: Some(0),
                ..brush(9)
            },
            DensityBrush {
                center_mm: [1, 1, 0],
                inner_radius_mm: 1,
                outer_radius_mm: 4,
                species: Some(u32::MAX),
                ..brush(2)
            },
        ],
    );
    for (radius, empty, f) in [
        (0, false, constant(65_535)),
        (0, false, painted.clone()),
        (2, false, painted),
        (0, false, constant(0)),
        (0, true, constant(65_535)),
        (0, false, constant(65_535)),
    ] {
        let cfg = ScatterConfig {
            // This fixed namespace leaves both field-kept and field-rejected
            // source survivors even with the radius-2 halo suppression.
            seed: 9,
            grid_spacing_mm: 1,
            min_distance_mm: radius,
            ..config()
        };
        let bounds = region([0; 3], [if empty { 0 } else { 8 }, 8, 1]);
        upload(&mut gpu, cfg, bounds, &f);
        // CPU references below are only evaluated after the draw has consumed
        // GPU-produced records, field attributes, compact indices and args.
        let (pixels, stream) = draw.draw(&ctx, &gpu);
        let result = assert_snapshot(&stream, cfg, bounds, &f);
        if !empty && f.config().base_density == u16::MAX {
            assert_eq!(result.accepted.len(), 64);
        }
        if !empty && f.config().noise.is_some() {
            assert!(
                !result.accepted.is_empty() && result.field_rejected != 0,
                "painted radius={radius}: accepted={}, field_rejected={}, source_rejected={}",
                result.accepted.len(),
                result.field_rejected,
                result.source_rejected
            );
        }
        if radius == 2 {
            assert!(result.source_rejected != 0);
        }
        let expected: BTreeMap<_, _> = result
            .accepted
            .iter()
            .map(|(point, sample)| {
                (
                    point.position_mm,
                    [
                        sample.species as u8,
                        (sample.density >> 8) as u8,
                        1 + (f.candidate_rank(point.key) & 127) as u8,
                        255,
                    ],
                )
            })
            .collect();
        for x in 0..8 {
            for y in 0..8 {
                let color = expected
                    .get(&[x as i64, y as i64, 0])
                    .copied()
                    .unwrap_or([0, 0, 0, 255]);
                assert_eq!(
                    pixel(&pixels, x * 8 + 4, y * 8 + 4),
                    &color,
                    "field indirect raster cell ({x},{y}), radius={radius}, empty={empty}"
                );
                assert_eq!(
                    pixel(&pixels, x * 8, y * 8),
                    &[0, 0, 0, 255],
                    "tile gap must stay clear"
                );
            }
        }
        if empty || f.config().base_density == 0 {
            assert!(pixels.chunks_exact(4).all(|p| p == [0, 0, 0, 255]));
        }
    }
}
