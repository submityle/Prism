//! Full-key GPU deletion parity, admission atomicity and actual indirect drawing.
#![cfg(feature = "gpu")]

#[path = "support/scatter_deletion.rs"]
mod support;

use prism_world_gpu::{
    GpuContext, GpuScatter, GpuScatterDeletionCapacity, GpuScatterDeletionError,
    GpuScatterDeletionUploadStats, GpuScatterDeletionWorkBudget, GpuScatterDeletions,
    GpuScatterSourceRequest,
};
use prism_world_residency::{
    estimate_scatter_work, ScatterConfig, ScatterDeletionSet, ScatterPoint, ScatterRegion,
};
use support::{config, context, deletions, read, region, source, staging, Capture};

fn request(config: ScatterConfig, region: ScatterRegion) -> GpuScatterSourceRequest {
    GpuScatterSourceRequest {
        config,
        region,
        max_candidate_evaluations: estimate_scatter_work(config, region)
            .unwrap()
            .max_candidate_evaluations,
    }
}

fn budget(n: u64, d: usize) -> GpuScatterDeletionWorkBudget {
    GpuScatterDeletionWorkBudget {
        max_input_slots: n,
        max_keys: d,
        max_key_comparisons: n * u64::from(usize::BITS - d.leading_zeros()),
    }
}

fn new(ctx: &GpuContext, n: usize, d: usize) -> GpuScatterDeletions<GpuScatter> {
    GpuScatterDeletions::new(
        ctx,
        GpuScatter::new(ctx, n, 3, u64::MAX).unwrap(),
        GpuScatterDeletionCapacity {
            max_keys: d,
            max_buffer_bytes: u64::MAX,
            max_scratch_bytes: d * 52,
        },
    )
    .unwrap()
}

fn capture(ctx: &GpuContext, gpu: &GpuScatterDeletions<GpuScatter>) -> Capture {
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    gpu.encode(&mut encoder).unwrap();
    let output = Capture::new(
        ctx,
        &mut encoder,
        gpu.candidate_count(),
        gpu.records(),
        gpu.attributes(),
        gpu.compact_indices(),
        gpu.indirect_args(),
    );
    ctx.queue().submit([encoder.finish()]);
    output
}

fn expected(points: &[ScatterPoint], deleted: &ScatterDeletionSet) -> Vec<ScatterPoint> {
    points
        .iter()
        .copied()
        .filter(|point| !deleted.is_deleted(point.key))
        .collect()
}

#[test]
fn empty_full_partial_tails_and_binary_search_boundaries() {
    let ctx = context();
    let mut gpu = new(&ctx, 129, 129);
    for n in [0, 1, 63, 64, 65, 129] {
        let bounds = region([-34, 0, 0], [-34 + n * 17, 17, 17]);
        let points = source(config(), bounds);
        assert_eq!(points.len(), n as usize);
        for d in [0, 1, 2, 3, 4, 7, 8, 9, 63, 64, 65, 127, 128, 129] {
            // Some deletion keys deliberately lie outside the current request.
            let all = source(config(), region([-34, 0, 0], [2159, 17, 17]));
            let deleted = deletions(all.iter().take(d).map(|p| p.key).collect());
            gpu.write_request(&request(config(), bounds), &deleted, budget(n as u64, d))
                .unwrap();
            capture(&ctx, &gpu).verify(&ctx, config(), &expected(&points, &deleted), None, 3);
        }
    }
}

#[test]
fn all_namespace_bits_and_signed_xyz_ordering_are_exact() {
    let ctx = context();
    let mut gpu = new(&ctx, 130, 1024);
    for axis in 0..3 {
        for origin in [-(1_i64 << 40) * 17, -17, (1_i64 << 40) * 17] {
            let mut lo = [-17; 3];
            let mut hi = [0; 3];
            lo[axis] = origin;
            hi[axis] = origin + 130 * 17;
            let bounds = region(lo, hi);
            let points = source(config(), bounds);
            let mut keys = Vec::new();
            for (i, point) in points.iter().enumerate() {
                let k = point.key;
                if i % 3 == 0 {
                    keys.push(k);
                }
                // Same grid, independent 64-bit halves and other namespace fields.
                keys.extend([
                    prism_world_residency::ScatterCandidateKey {
                        seed: k.seed ^ 1,
                        ..k
                    },
                    prism_world_residency::ScatterCandidateKey {
                        seed: k.seed ^ (1 << 63),
                        ..k
                    },
                    prism_world_residency::ScatterCandidateKey {
                        salt: k.salt ^ 1,
                        ..k
                    },
                    prism_world_residency::ScatterCandidateKey {
                        salt: k.salt ^ (1 << 63),
                        ..k
                    },
                    prism_world_residency::ScatterCandidateKey {
                        version: k.version ^ 1,
                        ..k
                    },
                    prism_world_residency::ScatterCandidateKey {
                        grid_spacing_mm: 18,
                        ..k
                    },
                ]);
            }
            let deleted = deletions(keys);
            gpu.write_request(
                &request(config(), bounds),
                &deleted,
                budget(130, deleted.keys().len()),
            )
            .unwrap();
            capture(&ctx, &gpu).verify(&ctx, config(), &expected(&points, &deleted), None, 3);
        }
    }
}

#[test]
fn minimum_distance_and_arbitrary_xyz_partition_union_are_preserved() {
    let ctx = context();
    let cfg = ScatterConfig {
        min_distance_mm: 23,
        ..config()
    };
    let whole = region([-51; 3], [51; 3]);
    let points = source(cfg, whole);
    assert!(!points.is_empty());
    let deleted = deletions(points.iter().step_by(3).map(|p| p.key).collect());
    let mut gpu = new(&ctx, 216, deleted.keys().len());
    gpu.write_request(
        &request(cfg, whole),
        &deleted,
        budget(216, deleted.keys().len()),
    )
    .unwrap();
    let mut full = capture(&ctx, &gpu).verify(&ctx, cfg, &expected(&points, &deleted), None, 3);
    let mut union = Vec::new();
    for x in [[-51, -7], [-7, 51]] {
        for y in [[-51, 8], [8, 51]] {
            for z in [[-51, 1], [1, 51]] {
                let bounds = region([x[0], y[0], z[0]], [x[1], y[1], z[1]]);
                let r = request(cfg, bounds);
                let n = estimate_scatter_work(cfg, bounds)
                    .unwrap()
                    .primary_candidates;
                gpu.write_request(&r, &deleted, budget(n, deleted.keys().len()))
                    .unwrap();
                union.extend(capture(&ctx, &gpu).verify(
                    &ctx,
                    cfg,
                    &expected(&source(cfg, bounds), &deleted),
                    None,
                    3,
                ));
            }
        }
    }
    full.sort_unstable_by_key(|p| p.key);
    union.sort_unstable_by_key(|p| p.key);
    assert_eq!(full, union);
}

#[test]
fn fixed_resources_changed_uploads_full_cache_and_sequential_submissions() {
    let ctx = context();
    let mut gpu = new(&ctx, 65, 66);
    let bounds = region([-34, 0, 0], [1071, 17, 17]);
    let points = source(config(), bounds);
    let r = request(config(), bounds);
    let a = deletions(points.iter().step_by(2).map(|p| p.key).collect());
    let empty = deletions(vec![]);
    let outside = source(config(), region([17000, 0, 0], [17017, 17, 17]))[0].key;
    let b = deletions(a.keys().iter().copied().chain([outside]).collect());
    let records = gpu.records().clone();
    let compact = gpu.compact_indices().clone();
    let indirect = gpu.indirect_args().clone();
    let bytes = gpu.allocated_buffer_bytes();
    assert_eq!(gpu.deletion_buffer_bytes(), 52 * 66 + 4 * 65 + 144);
    assert_eq!(bytes, 68 * 65 + 208 + gpu.deletion_buffer_bytes());
    assert_eq!(gpu.key_scratch_bytes(), 52 * 66);
    let mut captures = Vec::new();
    let mut first_key = None;
    for deleted in [&empty, &a, &a, &b, &empty, &a] {
        gpu.write_request(&r, deleted, budget(65, deleted.keys().len()))
            .unwrap();
        if first_key.is_none() && deleted == &a {
            first_key = gpu.cache_key().cloned();
        }
        if deleted == &b {
            assert_ne!(gpu.cache_key(), first_key.as_ref());
            assert_eq!(gpu.upload_stats().uniform_bytes, 128);
        }
        assert_eq!(gpu.records(), &records);
        assert_eq!(gpu.compact_indices(), &compact);
        assert_eq!(gpu.indirect_args(), &indirect);
        assert_eq!(gpu.allocated_buffer_bytes(), bytes);
        // Submit each request before the next upload, without CPU completion waits.
        captures.push((capture(&ctx, &gpu), expected(&points, deleted)));
    }
    let mut original_records = None;
    for (capture, expected) in captures {
        capture.verify(&ctx, config(), &expected, None, 3);
        let snapshot = read(&ctx, &capture.buffer);
        let records = snapshot[16..16 + 64 * 65].to_vec();
        if let Some(original) = &original_records {
            assert_eq!(original, &records);
        }
        original_records = Some(records);
    }
    gpu.write_request(&r, &a, budget(65, a.keys().len()))
        .unwrap();
    assert_eq!(gpu.upload_stats(), GpuScatterDeletionUploadStats::default());
    // Repeated encode clears rather than accumulates the GPU instance count.
    for _ in 0..3 {
        capture(&ctx, &gpu).verify(&ctx, config(), &expected(&points, &a), None, 3);
    }
}

#[test]
fn every_recoverable_preflight_error_preserves_request_and_gpu_result() {
    let ctx = context();
    let mut gpu = new(&ctx, 65, 2);
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    assert_eq!(
        gpu.encode(&mut encoder),
        Err(GpuScatterDeletionError::RequestNotUploaded)
    );
    let bounds = region([0; 3], [1105, 17, 17]);
    let points = source(config(), bounds);
    let r = request(config(), bounds);
    let deleted = deletions(vec![points[3].key]);
    gpu.write_request(&r, &deleted, budget(65, 1)).unwrap();
    let before = capture(&ctx, &gpu);
    let key = gpu.cache_key().cloned();
    let stats = gpu.upload_stats();
    let cases = [
        (
            r,
            deleted.clone(),
            GpuScatterDeletionWorkBudget {
                max_input_slots: 64,
                ..budget(65, 1)
            },
        ),
        (
            r,
            deleted.clone(),
            GpuScatterDeletionWorkBudget {
                max_keys: 0,
                ..budget(65, 1)
            },
        ),
        (
            r,
            deleted.clone(),
            GpuScatterDeletionWorkBudget {
                max_key_comparisons: 64,
                ..budget(65, 1)
            },
        ),
        (
            GpuScatterSourceRequest {
                max_candidate_evaluations: 64,
                ..r
            },
            deleted.clone(),
            budget(65, 1),
        ),
        (
            GpuScatterSourceRequest {
                config: ScatterConfig {
                    grid_spacing_mm: 0,
                    ..config()
                },
                ..r
            },
            deleted.clone(),
            budget(65, 1),
        ),
        (
            request(config(), region([0; 3], [1122, 17, 17])),
            deleted.clone(),
            budget(66, 1),
        ),
        (
            r,
            deletions(points.iter().take(3).map(|p| p.key).collect()),
            budget(65, 3),
        ),
    ];
    for (request, deleted, budget) in cases {
        assert!(gpu.validate_request(&request, &deleted, budget).is_err());
        assert!(gpu.write_request(&request, &deleted, budget).is_err());
        assert_eq!(gpu.cache_key(), key.as_ref());
        assert_eq!(gpu.upload_stats(), stats);
        capture(&ctx, &gpu).verify(
            &ctx,
            config(),
            &expected(&points, &key.as_ref().unwrap().deletions),
            None,
            3,
        );
    }
    before.verify(&ctx, config(), &expected(&points, &deleted), None, 3);
}

#[test]
fn constructor_checks_device_buffer_and_actual_scratch_budgets() {
    let ctx = context();
    let cap = GpuScatterDeletionCapacity {
        max_keys: 2,
        max_buffer_bytes: u64::MAX,
        max_scratch_bytes: 104,
    };
    assert!(matches!(
        GpuScatterDeletions::new(
            &ctx,
            GpuScatter::new(&ctx, 1, 3, 276).unwrap(),
            GpuScatterDeletionCapacity {
                max_scratch_bytes: 103,
                ..cap
            }
        ),
        Err(GpuScatterDeletionError::ScratchBudgetExceeded {
            required: 104,
            limit: 103
        })
    ));
    let bytes = 276 + 104 + 4 + 144;
    assert!(GpuScatterDeletions::new(
        &ctx,
        GpuScatter::new(&ctx, 1, 3, 276).unwrap(),
        GpuScatterDeletionCapacity {
            max_buffer_bytes: bytes - 1,
            ..cap
        }
    )
    .is_err());
    let other = context();
    assert!(matches!(
        GpuScatterDeletions::new(&ctx, GpuScatter::new(&other, 1, 3, 276).unwrap(), cap),
        Err(GpuScatterDeletionError::DeviceMismatch)
    ));
    // A cloned context shares the logical device, unlike an independent Instance.
    let shared = ctx.clone();
    let mut shared_gpu =
        GpuScatterDeletions::new(&ctx, GpuScatter::new(&shared, 1, 3, 276).unwrap(), cap).unwrap();
    let bounds = region([0; 3], [17; 3]);
    shared_gpu
        .write_request(&request(config(), bounds), &deletions(vec![]), budget(1, 0))
        .unwrap();
    capture(&shared, &shared_gpu).verify(&shared, config(), &source(config(), bounds), None, 3);
    let mut zero = new(&ctx, 0, 0);
    zero.write_request(
        &request(config(), region([0; 3], [0; 3])),
        &deletions(vec![]),
        budget(0, 0),
    )
    .unwrap();
    capture(&ctx, &zero).verify(&ctx, config(), &[], None, 3);
}

const DRAW_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> records: array<u32>;
@group(0) @binding(1) var<storage, read> compact: array<u32>;
@vertex fn vs(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32)
    -> @builtin(position) vec4<f32> {
    let base = compact[instance] * 16u;
    let offset = array<vec2<f32>, 3>(vec2(-3.0,-3.0), vec2(3.0,-3.0), vec2(0.0,3.0));
    let pixel = vec2(f32(records[base+6u]), f32(records[base+8u])) * 8.0 + 4.5 + offset[vertex];
    return vec4(pixel.x / 32.0 - 1.0, 1.0 - pixel.y / 32.0, 0.0, 1.0);
}
@fragment fn fs() -> @location(0) vec4<f32> { return vec4(1.0, 0.0, 1.0, 1.0); }
"#;

#[test]
fn final_compact_and_gpu_count_feed_actual_draw_indirect_without_cpu_vertices() {
    let ctx = context();
    let mut gpu = new(&ctx, 64, 64);
    let device = ctx.device();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("deletion actual indirect stream consumer"),
        source: wgpu::ShaderSource::Wgsl(DRAW_SHADER.into()),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("deletion GPU-count raster"),
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
    let binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("deletion GPU records and compact directly bound"),
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
        label: Some("deletion offscreen pixel oracle"),
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
    let pixels = staging(&ctx, 64 * 256);
    let cfg = ScatterConfig {
        grid_spacing_mm: 1,
        ..config()
    };
    let bounds = region([0; 3], [8, 8, 1]);
    let points = source(cfg, bounds);
    for step in [0, 2, 1, 0] {
        let deleted = deletions(if step == 0 {
            vec![]
        } else {
            points.iter().step_by(step).map(|p| p.key).collect()
        });
        gpu.write_request(
            &request(cfg, bounds),
            &deleted,
            budget(64, deleted.keys().len()),
        )
        .unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        gpu.encode(&mut encoder).unwrap();
        let view = target.create_view(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("deletion real indirect raster"),
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
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &binding, &[]);
            pass.draw_indirect(gpu.indirect_args(), 0);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &pixels,
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
        let pixels = read(&ctx, &pixels);
        // CPU decides expected pixels only after the GPU has drawn.
        for point in &points {
            let offset = ((point.position_mm[1] as usize * 8 + 4) * 64
                + point.position_mm[0] as usize * 8
                + 4)
                * 4;
            let expected = if deleted.is_deleted(point.key) {
                [0, 0, 0, 255]
            } else {
                [255, 0, 255, 255]
            };
            assert_eq!(&pixels[offset..offset + 4], &expected);
        }
    }
}
