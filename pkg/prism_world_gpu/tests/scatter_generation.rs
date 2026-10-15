//! Exact integer scatter parity on a physical GPU, plus GPU-to-GPU indirect draw.
//! Readback and the tiny consumer pipeline are test-only; no adapter skips.

#![cfg(feature = "gpu")]

use std::collections::BTreeSet;

use prism_world_gpu::{GpuContext, GpuScatter, GpuScatterError};
use prism_world_residency::{
    estimate_scatter_work, pcg_hash_u32, scatter, ScatterBudget, ScatterCacheKey,
    ScatterCandidateKey, ScatterConfig, ScatterError, ScatterPoint, ScatterRegion, SectorCoord,
    SCATTER_KERNEL_VERSION,
};

fn gpu_context() -> GpuContext {
    let ctx = GpuContext::try_headless()
        .expect("scatter requires successful real-device acquisition")
        .expect("scatter requires an adapter; skipping is forbidden");
    let evidence = ctx.evidence();
    let actual = match evidence.backend.as_str() {
        "Metal" => wgpu::Backends::METAL,
        "Vulkan" => wgpu::Backends::VULKAN,
        "Dx12" => wgpu::Backends::DX12,
        other => panic!("unexpected native scatter backend {other}"),
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
    eprintln!("[scatter_generation] adapter={evidence:?}");
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

fn region(min: [i64; 3], max: [i64; 3]) -> ScatterRegion {
    ScatterRegion::new(min, max).unwrap()
}

fn logical_bytes(capacity: usize) -> u64 {
    68 * capacity.max(1) as u64 + 208
}

fn staging(ctx: &GpuContext, size: u64) -> wgpu::Buffer {
    ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("scatter test readback only"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn read(ctx: &GpuContext, buffer: &wgpu::Buffer) -> Vec<u8> {
    let (send, recv) = std::sync::mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        send.send(result).expect("scatter map receiver");
    });
    ctx.wait();
    recv.recv()
        .expect("scatter map callback")
        .expect("scatter readback mapping");
    let bytes = buffer.get_mapped_range(..).unwrap().to_vec();
    buffer.unmap();
    bytes
}

// One capture per submitted request, including no-work requests. Only valid
// prefixes are copied: stale capacity tails have no defined contents.
fn capture(ctx: &GpuContext, gpu: &GpuScatter, encoder: &mut wgpu::CommandEncoder) -> wgpu::Buffer {
    let count = u64::from(gpu.candidate_count());
    let output = staging(ctx, 16 + 68 * count);
    encoder.copy_buffer_to_buffer(gpu.indirect_args(), 0, &output, 0, 16);
    if count != 0 {
        encoder.copy_buffer_to_buffer(gpu.records(), 0, &output, 16, 64 * count);
        encoder.copy_buffer_to_buffer(
            gpu.compact_indices(),
            0,
            &output,
            16 + 64 * count,
            4 * count,
        );
    }
    output
}

fn submit_capture(ctx: &GpuContext, gpu: &GpuScatter) -> wgpu::Buffer {
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    gpu.encode(&mut encoder).unwrap();
    let output = capture(ctx, gpu, &mut encoder);
    ctx.queue().submit([encoder.finish()]);
    output
}

fn cpu(config: ScatterConfig, region: ScatterRegion) -> Vec<ScatterPoint> {
    let estimate = estimate_scatter_work(config, region).unwrap();
    let count = usize::try_from(estimate.primary_candidates).unwrap();
    let output = scatter(
        config,
        region,
        ScatterBudget {
            max_candidate_evaluations: estimate.max_candidate_evaluations,
            max_output_points: count,
            max_output_bytes: count * size_of::<ScatterPoint>(),
        },
    )
    .unwrap();
    assert!(output.candidate_evaluations <= estimate.max_candidate_evaluations);
    output.points
}

fn hash64(value: u64) -> u64 {
    (u64::from(pcg_hash_u32(value)) << 32) | u64::from(pcg_hash_u32(value ^ 0x9e37_79b9_7f4a_7c15))
}

// Freeze the rank tuple separately from shader limb arithmetic. Positions use
// the public CPU key.rebuild oracle, which performs widened i128/u128 math.
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

fn assert_snapshot(
    bytes: &[u8],
    config: ScatterConfig,
    region: ScatterRegion,
    vertex_count: u32,
) -> Vec<ScatterPoint> {
    let words: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let count = estimate_scatter_work(config, region)
        .unwrap()
        .primary_candidates as usize;
    assert_eq!(bytes.len(), 16 + 68 * count);
    let expected = cpu(config, region);
    assert_eq!(
        &words[..4],
        &[vertex_count, expected.len() as u32, 0, 0],
        "indirect arguments: config={config:?}, region={region:?}"
    );
    let mut accepted = Vec::new();
    let mut accepted_slots = BTreeSet::new();
    if count != 0 {
        let spacing = i64::from(config.grid_spacing_mm);
        let min = region.min_mm.map(|v| v.div_euclid(spacing));
        let max = region.max_mm.map(|v| (v - 1).div_euclid(spacing));
        let mut slot = 0;
        for x in min[0]..=max[0] {
            for y in min[1]..=max[1] {
                for z in min[2]..=max[2] {
                    let record = &words[4 + slot * 16..4 + (slot + 1) * 16];
                    assert_eq!(
                        [
                            pair(record, 0) as i64,
                            pair(record, 2) as i64,
                            pair(record, 4) as i64
                        ],
                        [x, y, z],
                        "source slot {slot}: global XYZ, Z fastest"
                    );
                    assert!(record[12] <= 1, "non-boolean accepted flag at {slot}");
                    assert_eq!(record[15], 0, "padding at source slot {slot}");
                    let key = ScatterCandidateKey {
                        kernel_version: SCATTER_KERNEL_VERSION,
                        seed: config.seed,
                        version: config.version,
                        salt: config.salt,
                        grid_spacing_mm: config.grid_spacing_mm,
                        grid: SectorCoord::new(x, y, z),
                    };
                    match key.rebuild() {
                        Ok(point) => {
                            assert_eq!(
                                [
                                    pair(record, 6) as i64,
                                    pair(record, 8) as i64,
                                    pair(record, 10) as i64
                                ],
                                point.position_mm,
                                "exact position even for rejected source {key:?}"
                            );
                            assert_eq!(pair(record, 13), priority(key), "priority {key:?}");
                            if record[12] != 0 {
                                assert!(
                                    region.contains_mm(point.position_mm),
                                    "half-open ownership"
                                );
                                accepted.push(point);
                                accepted_slots.insert(slot as u32);
                            }
                        }
                        Err(ScatterError::CoordinateOverflow) => {
                            assert_eq!(&record[6..12], &[0; 6], "invalid world position {key:?}");
                            assert_eq!(record[12], 0, "invalid jitter must not be clamped");
                        }
                        Err(error) => panic!("unexpected rebuild error {error:?}"),
                    }
                    slot += 1;
                }
            }
        }
        assert_eq!(slot, count);
    }
    // Whole structural keys AND exact integer positions, not just hashes/counts.
    assert_eq!(
        accepted, expected,
        "CPU set: config={config:?}, region={region:?}"
    );
    let compact = &words[4 + 16 * count..];
    let mut compact_slots = BTreeSet::new();
    for &slot in &compact[..accepted.len()] {
        assert!((slot as usize) < count, "compact source slot out of range");
        assert!(
            compact_slots.insert(slot),
            "duplicate compact source slot {slot}"
        );
    }
    assert_eq!(
        compact_slots, accepted_slots,
        "exact compact source-slot set"
    );
    accepted
}

fn upload(gpu: &mut GpuScatter, config: ScatterConfig, region: ScatterRegion) {
    let estimate = estimate_scatter_work(config, region).unwrap();
    gpu.write_request(config, region, estimate.max_candidate_evaluations)
        .unwrap();
    assert_eq!(
        u64::from(gpu.candidate_count()),
        estimate.primary_candidates
    );
    assert_eq!(
        gpu.cache_key(),
        Some(ScatterCacheKey {
            kernel_version: SCATTER_KERNEL_VERSION,
            config,
            region
        })
    );
}

fn run(
    ctx: &GpuContext,
    gpu: &mut GpuScatter,
    config: ScatterConfig,
    region: ScatterRegion,
) -> Vec<ScatterPoint> {
    upload(gpu, config, region);
    let output = submit_capture(ctx, gpu);
    assert_snapshot(&read(ctx, &output), config, region, 3)
}

#[test]
fn exact_cpu_sets_integer_hash_jitter_radius_and_large_coordinates() {
    let ctx = gpu_context();
    let mut gpu = GpuScatter::new(&ctx, 1024, 3, logical_bytes(1024)).unwrap();
    for spacing in [1, 2, 17, 65_537, 0x8000_0001, u32::MAX - 1, u32::MAX] {
        let s = i64::from(spacing);
        let bounds = region(
            [-3 * s - 1, -2 * s + 1, -s / 2],
            [2 * s + 1, 2 * s + 1, 2 * s],
        );
        for (seed, salt, version) in [
            (0, 0, 0),
            (u64::MAX, 0x8000_0000_0000_0000, u32::MAX),
            (0x0123_4567_89ab_cdef, u64::MAX, 7),
        ] {
            for radius in [0, spacing, spacing.saturating_add(1)] {
                let cfg = ScatterConfig {
                    seed,
                    salt,
                    version,
                    grid_spacing_mm: spacing,
                    min_distance_mm: radius,
                };
                run(&ctx, &mut gpu, cfg, bounds);
            }
        }
    }
    // Beyond f64's exact integer range, including low-limb carries/sign changes.
    for origin in [-(1_i64 << 60) - 31, -(1_i64 << 32) - 1, (1_i64 << 60) + 29] {
        for radius in [0, 29] {
            run(
                &ctx,
                &mut gpu,
                ScatterConfig {
                    min_distance_mm: radius,
                    ..config()
                },
                region([origin, -origin - 11, -37], [origin + 71, -origin + 53, 49]),
            );
        }
    }
    // Equality at the minimum lattice distance is allowed, not excluded.
    let cfg = ScatterConfig {
        grid_spacing_mm: 1,
        min_distance_mm: 1,
        ..config()
    };
    assert_eq!(run(&ctx, &mut gpu, cfg, region([-2; 3], [3; 3])).len(), 125);
}

#[test]
fn exact_cpu_sets_at_signed_world_edges_and_clipped_halos() {
    let ctx = gpu_context();
    let mut gpu = GpuScatter::new(&ctx, 512, 3, logical_bytes(512)).unwrap();
    for spacing in [1, 3, 17, u32::MAX] {
        let width = i64::from(spacing) * 3;
        for radius in [0, spacing, spacing.saturating_add(1)] {
            let cfg = ScatterConfig {
                grid_spacing_mm: spacing,
                min_distance_mm: radius,
                seed: u64::MAX,
                salt: u64::MAX - 1,
                ..config()
            };
            // Every lower/upper corner combination exercises all axis edge paths.
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
                run(&ctx, &mut gpu, cfg, region(min, max));
            }
        }
    }
}

#[test]
fn exact_cpu_sets_survive_arbitrary_region_splits_and_external_blockers() {
    let ctx = gpu_context();
    let mut gpu = GpuScatter::new(&ctx, 1024, 3, logical_bytes(1024)).unwrap();
    let whole = region([-35, -23, -19], [40, 39, 31]);
    let xs = [-35, -3, 12, 40];
    let ys = [-23, 1, 39];
    let zs = [-19, 4, 31];
    for radius in [0, 11, 29] {
        let cfg = ScatterConfig {
            min_distance_mm: radius,
            ..config()
        };
        let expected = run(&ctx, &mut gpu, cfg, whole);
        let mut parts = Vec::new();
        for x in xs.windows(2) {
            for y in ys.windows(2) {
                for z in zs.windows(2) {
                    parts.push(region([x[0], y[0], z[0]], [x[1], y[1], z[1]]));
                }
            }
        }
        for reverse in [false, true] {
            if reverse {
                parts.reverse();
            }
            let mut union = Vec::new();
            for &part in &parts {
                union.extend(run(&ctx, &mut gpu, cfg, part));
            }
            union.sort_by_key(|point| point.key);
            let keys: BTreeSet<_> = union.iter().map(|point| point.key).collect();
            assert_eq!(keys.len(), union.len(), "partition ownership is unique");
            assert_eq!(union, expected, "split order/radius {radius}");
        }
    }
    let cfg = ScatterConfig {
        min_distance_mm: 29,
        ..config()
    };
    let survivors: BTreeSet<_> = cpu(cfg, whole).iter().map(|p| p.key).collect();
    let blocked = cpu(config(), whole)
        .into_iter()
        .find(|point| !survivors.contains(&point.key))
        .expect("fixture needs a globally blocked candidate");
    // Exactly this candidate is owned, but its outside-request original
    // neighbours must still reject it; no request-local recursive thinning.
    let singleton = region(blocked.position_mm, blocked.position_mm.map(|v| v + 1));
    assert_eq!(run(&ctx, &mut gpu, config(), singleton), vec![blocked]);
    assert!(run(&ctx, &mut gpu, cfg, singleton).is_empty());
}

fn assert_buffers(gpu: &GpuScatter, buffers: &[wgpu::Buffer; 3], capacity: usize) {
    assert_eq!(gpu.records(), &buffers[0], "record buffer was replaced");
    assert_eq!(
        gpu.compact_indices(),
        &buffers[1],
        "compact buffer was replaced"
    );
    assert_eq!(
        gpu.indirect_args(),
        &buffers[2],
        "indirect buffer was replaced"
    );
    assert_eq!(gpu.allocated_buffer_bytes(), logical_bytes(capacity));
}

#[test]
fn exact_counts_tail_reuse_and_transactional_preflight_errors() {
    let ctx = gpu_context();
    const CAPACITY: usize = 129;
    let bytes = logical_bytes(CAPACITY);
    let mut gpu = GpuScatter::new(&ctx, CAPACITY, 3, bytes).unwrap();
    let buffers = [
        gpu.records().clone(),
        gpu.compact_indices().clone(),
        gpu.indirect_args().clone(),
    ];
    assert_eq!(gpu.candidate_count(), 0);
    assert_eq!(gpu.cache_key(), None);
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    assert_eq!(
        gpu.encode(&mut encoder),
        Err(GpuScatterError::RequestNotUploaded)
    );
    let cfg = ScatterConfig {
        grid_spacing_mm: 1,
        ..config()
    };
    let mut captures = Vec::new();
    // Submit before the next upload, with NO wait/map between submissions.
    for count in [0, 1, 63, 64, 65, 128, 129, 0, 129] {
        let bounds = region([0; 3], [count, 1, 1]);
        upload(&mut gpu, cfg, bounds);
        captures.push((submit_capture(&ctx, &gpu), cfg, bounds));
        assert_buffers(&gpu, &buffers, CAPACITY);
    }
    // Overwrite a full prefix with rejected flags, then regenerate unchanged
    // uniforms twice: accepted count must reset, never accumulate.
    let bounds = region([0; 3], [129, 1, 1]);
    let thin = ScatterConfig {
        min_distance_mm: 2,
        ..cfg
    };
    upload(&mut gpu, thin, bounds);
    captures.push((submit_capture(&ctx, &gpu), thin, bounds));
    captures.push((submit_capture(&ctx, &gpu), thin, bounds));
    for (output, request, bounds) in captures {
        assert_snapshot(&read(&ctx, &output), request, bounds, 3);
    }
    let old_key = gpu.cache_key();
    let old_count = gpu.candidate_count();
    let changed = ScatterConfig {
        seed: !cfg.seed,
        ..cfg
    };
    let cases = [
        (
            changed,
            region([0; 3], [130, 1, 1]),
            u64::MAX,
            GpuScatterError::CapacityExceeded {
                required: 130,
                capacity: CAPACITY,
            },
        ),
        (
            ScatterConfig {
                grid_spacing_mm: 0,
                ..changed
            },
            bounds,
            u64::MAX,
            GpuScatterError::Scatter(ScatterError::InvalidGridSpacing),
        ),
        (
            changed,
            ScatterRegion {
                min_mm: [1, 0, 0],
                max_mm: [0, 1, 1],
            },
            u64::MAX,
            GpuScatterError::Scatter(ScatterError::InvalidRegion),
        ),
        (
            changed,
            bounds,
            128,
            GpuScatterError::Scatter(ScatterError::CandidateBudgetExceeded {
                required: 129,
                limit: 128,
            }),
        ),
        (
            changed,
            region([i64::MIN; 3], [i64::MAX; 3]),
            u64::MAX,
            GpuScatterError::Scatter(ScatterError::ArithmeticOverflow),
        ),
        (
            ScatterConfig {
                min_distance_mm: u32::MAX,
                ..changed
            },
            region([0; 3], [1; 3]),
            u64::MAX,
            GpuScatterError::Scatter(ScatterError::ArithmeticOverflow),
        ),
    ];
    for (request, bounds, budget, error) in cases {
        assert_eq!(gpu.write_request(request, bounds, budget), Err(error));
        assert_eq!(gpu.cache_key(), old_key);
        assert_eq!(gpu.candidate_count(), old_count);
        assert_buffers(&gpu, &buffers, CAPACITY);
        // Reading metadata alone would miss an erroneous uniform upload.
        let output = submit_capture(&ctx, &gpu);
        assert_snapshot(&read(&ctx, &output), thin, old_key.unwrap().region, 3);
    }
    assert!(matches!(GpuScatter::new(&ctx, CAPACITY, 3, 0),
        Err(GpuScatterError::BufferBudgetExceeded { required, limit: 0 }) if required == bytes));
    assert!(matches!(GpuScatter::new(&ctx, CAPACITY, 3, bytes - 1),
        Err(GpuScatterError::BufferBudgetExceeded { required, limit }) if required == bytes && limit == bytes - 1));
    assert!(matches!(
        GpuScatter::new(&ctx, usize::MAX, 3, u64::MAX),
        Err(GpuScatterError::ArithmeticOverflow)
    ));
    let limits = ctx.device().limits();
    let too_many = limits.max_storage_buffer_binding_size / 64 + 1;
    assert!(matches!(
        GpuScatter::new(&ctx, too_many as usize, 3, u64::MAX),
        Err(GpuScatterError::DeviceLimit {
            limit: "max_storage_buffer_binding_size",
            ..
        })
    ));

    let mut zero = GpuScatter::new(&ctx, 0, 3, logical_bytes(0)).unwrap();
    let zero_buffers = [
        zero.records().clone(),
        zero.compact_indices().clone(),
        zero.indirect_args().clone(),
    ];
    let empty_cfg = ScatterConfig {
        min_distance_mm: u32::MAX,
        ..cfg
    };
    // Empty axis can be anywhere and avoids even an overflowing halo estimate.
    for axis in 0..3 {
        let mut max = [i64::MAX; 3];
        max[axis] = i64::MIN;
        run(&ctx, &mut zero, empty_cfg, region([i64::MIN; 3], max));
        assert_buffers(&zero, &zero_buffers, 0);
    }
    assert_eq!(
        zero.write_request(cfg, region([0; 3], [1; 3]), 1),
        Err(GpuScatterError::CapacityExceeded {
            required: 1,
            capacity: 0
        })
    );
    assert_eq!(zero.candidate_count(), 0);

    // With the current downlevel logical-device limits, a 2D dispatch needs
    // >4M slots and >256MiB records, exceeding the 128MiB storage binding cap.
    // Do not claim that tail coverage exercises the unreachable second row.
    let second_row_slots = u64::from(limits.max_compute_workgroups_per_dimension) * 64 + 1;
    eprintln!("[scatter_generation] 2D threshold={second_row_slots} slots, record cap={} bytes; 2D dispatch arithmetic is covered by host unit tests", limits.max_storage_buffer_binding_size);
}

const DRAW_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> records: array<u32>;
@group(0) @binding(1) var<storage, read> compact: array<u32>;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) color: vec4<f32>,
};

@vertex fn vs(@builtin(vertex_index) vertex: u32,
              @builtin(instance_index) instance: u32) -> VertexOut {
    let base = compact[instance] * 16u;
    // This fixture is an exact nonnegative spacing-1 lattice. The general
    // production ABI remains i64 low/high pairs; this consumer needs only lows.
    let x = records[base + 6u];
    let y = records[base + 8u];
    let offsets = array<vec2<f32>, 3>(vec2(-3.0, -3.0), vec2(3.0, -3.0), vec2(0.0, 3.0));
    let pixel = vec2<f32>(f32(x) * 8.0 + 4.5, f32(y) * 8.0 + 4.5) + offsets[vertex];
    var out: VertexOut;
    out.position = vec4(pixel.x / 32.0 - 1.0, 1.0 - pixel.y / 32.0, 0.0, 1.0);
    out.color = vec4(f32(x & 1u), f32(y & 1u), f32(records[base + 12u]), 1.0);
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
    fn new(ctx: &GpuContext, gpu: &GpuScatter) -> Self {
        let device = ctx.device();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("test-only generated scatter stream consumer"),
            source: wgpu::ShaderSource::Wgsl(DRAW_SHADER.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scatter records + compact slots indirect raster"),
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
            label: Some("scatter generated buffers read directly by vertex stage"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu.records().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gpu.compact_indices().as_entire_binding(),
                },
            ],
        });
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scatter indirect 8x8 lattice offscreen target"),
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

    fn draw(&self, ctx: &GpuContext, gpu: &GpuScatter) -> (Vec<u8>, Vec<u8>) {
        let mut encoder = ctx.device().create_command_encoder(&Default::default());
        gpu.encode(&mut encoder).unwrap();
        let view = self.target.create_view(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scatter actual GPU-count draw_indirect"),
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
            // Generation -> record/compact fetch -> draw, one submission.
            // No mapped counts, CPU-generated vertices, or direct draw calls.
            pass.draw_indirect(gpu.indirect_args(), 0);
        }
        let output = capture(ctx, gpu, &mut encoder);
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
        (read(ctx, &self.pixels), read(ctx, &output))
    }
}

fn pixel(bytes: &[u8], x: usize, y: usize) -> &[u8] {
    let offset = (y * 64 + x) * 4;
    &bytes[offset..offset + 4]
}

#[test]
fn generated_record_and_compact_stream_feed_actual_draw_indirect() {
    let ctx = gpu_context();
    let mut gpu = GpuScatter::new(&ctx, 64, 3, logical_bytes(64)).unwrap();
    let draw = DrawHarness::new(&ctx, &gpu);
    for (radius, empty) in [(0, false), (1, false), (2, false), (0, true), (0, false)] {
        let cfg = ScatterConfig {
            grid_spacing_mm: 1,
            min_distance_mm: radius,
            ..config()
        };
        let bounds = region([0; 3], [if empty { 0 } else { 8 }, 8, 1]);
        upload(&mut gpu, cfg, bounds);
        // CPU references are inspected only after draw has consumed the stream.
        let (pixels, stream) = draw.draw(&ctx, &gpu);
        let expected = assert_snapshot(&stream, cfg, bounds, 3);
        if !empty && radius <= 1 {
            assert_eq!(expected.len(), 64);
        }
        if radius == 2 {
            assert!(!expected.is_empty() && expected.len() < 64);
        }
        let positions: BTreeSet<_> = expected.iter().map(|p| p.position_mm).collect();
        for x in 0..8 {
            for y in 0..8 {
                let color = if positions.contains(&[x as i64, y as i64, 0]) {
                    [(x as u8 & 1) * 255, (y as u8 & 1) * 255, 255, 255]
                } else {
                    [0, 0, 0, 255]
                };
                assert_eq!(
                    pixel(&pixels, x * 8 + 4, y * 8 + 4),
                    &color,
                    "indirect raster cell ({x},{y}), radius={radius}, empty={empty}"
                );
                assert_eq!(
                    pixel(&pixels, x * 8, y * 8),
                    &[0, 0, 0, 255],
                    "tile gap must remain clear"
                );
            }
        }
        if empty {
            assert!(pixels.chunks_exact(4).all(|p| p == [0, 0, 0, 255]));
        }
    }
}
