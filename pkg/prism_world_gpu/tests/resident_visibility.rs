//! Strict physical-GPU regressions. Readback exists only in this test harness.

#![cfg(feature = "gpu")]

use std::collections::BTreeSet;

use prism_world_gpu::{
    FrustumPlane, GpuContext, GpuResidentError, GpuResidentVisibility, ProxyInstance, ProxySoa,
    VisibilityInput,
};

fn gpu_context() -> GpuContext {
    let ctx = GpuContext::try_headless()
        .expect("resident visibility requires successful real-device acquisition")
        .expect("resident visibility requires an adapter; skipping is forbidden");
    let evidence = ctx.evidence();
    if let Some(requested) = wgpu::Backends::from_env() {
        let actual = match evidence.backend.as_str() {
            "Metal" => wgpu::Backends::METAL,
            "Vulkan" => wgpu::Backends::VULKAN,
            "Dx12" => wgpu::Backends::DX12,
            other => panic!("unexpected native backend {other}"),
        };
        assert!(requested.contains(actual), "WGPU_BACKEND was not honored");
    }
    assert!(
        matches!(
            evidence.device_type.as_str(),
            "IntegratedGpu" | "DiscreteGpu"
        ),
        "physical GPU required: {evidence:?}"
    );
    eprintln!("[resident_visibility] adapter={evidence:?}");
    ctx
}

fn camera() -> VisibilityInput {
    VisibilityInput::from_projection_scale(
        [0.0; 3],
        [
            FrustumPlane::new([1.0, 0.0, 0.0], 10.0),
            FrustumPlane::new([-1.0, 0.0, 0.0], 10.0),
            FrustumPlane::new([0.0, 1.0, 0.0], 10.0),
            FrustumPlane::new([0.0, -1.0, 0.0], 10.0),
            FrustumPlane::new([0.0, 0.0, 1.0], 10.0),
            FrustumPlane::new([0.0, 0.0, -1.0], 10.0),
        ],
        1.0,
        32.0,
    )
}

fn boundary_proxies() -> Vec<ProxyInstance> {
    let errors = [0.0, 3.5, 4.0, 15.5, 16.0, 63.5, 64.0, 512.0, 1.0 / 512.0];
    (0..129)
        .map(|i| {
            // Deliberately repeat identities: compaction must return source slots.
            let mut proxy = ProxyInstance::new(
                900 + i % 3,
                7,
                [0.0, 0.0, 1.0],
                0.0,
                errors[i as usize % errors.len()],
                100.0,
            );
            match i % 20 {
                // Tangent spheres on all six planes survive; just outside fails.
                1..=6 => {
                    let plane = (i % 20 - 1) as usize;
                    proxy.position = [0.0; 3];
                    proxy.position[plane / 2] = if plane.is_multiple_of(2) { -10.5 } else { 10.5 };
                    proxy.radius = 0.5;
                }
                7 => {
                    proxy.position = [10.75, 0.0, 0.0];
                    proxy.radius = 0.5;
                }
                // Large sphere keeps range boundaries independent of frustum.
                8 => {
                    proxy.position = [0.0, 0.0, 32.0];
                    proxy.radius = 32.0;
                }
                9 => {
                    proxy.position = [0.0, 0.0, 32.25];
                    proxy.radius = 32.0;
                }
                10 => {
                    proxy.position = [0.0, 0.0, 2.0];
                    proxy.max_distance = 2.0;
                }
                11 => {
                    proxy.position = [0.0, 0.0, 2.0];
                    proxy.max_distance = 1.75;
                }
                12 => {
                    proxy.position = [0.0; 3];
                    proxy.geometric_error = 0.25;
                }
                _ => {}
            }
            proxy
        })
        .collect()
}

fn staging(ctx: &GpuContext, size: u64) -> wgpu::Buffer {
    ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("resident test readback only"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn read(ctx: &GpuContext, buffer: &wgpu::Buffer) -> Vec<u8> {
    let (send, recv) = std::sync::mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        send.send(result).expect("resident map callback receiver");
    });
    ctx.wait();
    recv.recv()
        .expect("resident map callback")
        .expect("resident map succeeded");
    let bytes = buffer
        .get_mapped_range(..)
        .expect("resident mapped bytes")
        .to_vec();
    buffer.unmap();
    bytes
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn capture(
    ctx: &GpuContext,
    resident: &GpuResidentVisibility,
    encoder: &mut wgpu::CommandEncoder,
) -> wgpu::Buffer {
    let slots = resident.capacity() as u64;
    let output = staging(ctx, 16 + slots * 24);
    encoder.copy_buffer_to_buffer(resident.indirect_args(), 0, &output, 0, 16);
    if slots != 0 {
        encoder.copy_buffer_to_buffer(resident.compact_indices(), 0, &output, 16, slots * 4);
        encoder.copy_buffer_to_buffer(
            resident.visibility_records(),
            0,
            &output,
            16 + slots * 4,
            slots * 20,
        );
    }
    output
}

fn submit_snapshot(
    ctx: &GpuContext,
    resident: &mut GpuResidentVisibility,
    input: VisibilityInput,
) -> wgpu::Buffer {
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    resident.write_camera(input).unwrap();
    resident.encode(input, &mut encoder).unwrap();
    let output = capture(ctx, resident, &mut encoder);
    ctx.queue().submit([encoder.finish()]);
    output
}

fn assert_snapshot(
    bytes: &[u8],
    capacity: usize,
    vertex_count: u32,
    input: VisibilityInput,
    proxies: &[ProxyInstance],
) {
    let data = words(bytes);
    let expected: BTreeSet<u32> = proxies
        .iter()
        .enumerate()
        .filter(|(_, p)| input.evaluate(p).visible)
        .map(|(i, _)| i as u32)
        .collect();
    assert_eq!(&data[..4], &[vertex_count, expected.len() as u32, 0, 0]);
    let compact = &data[4..4 + expected.len()];
    let actual: BTreeSet<_> = compact.iter().copied().collect();
    assert_eq!(actual.len(), compact.len(), "no duplicate source slots");
    assert_eq!(actual, expected, "complete visible source-slot set");
    for (i, proxy) in proxies.iter().enumerate() {
        let cpu = input.evaluate(proxy);
        let record = &data[4 + capacity + i * 5..4 + capacity + (i + 1) * 5];
        assert_eq!(
            &record[..4],
            &cpu.words(),
            "canonical record at source slot {i}"
        );
        let error = f32::from_bits(record[4]);
        assert!(
            (error - cpu.screen_error).abs() <= cpu.screen_error.abs().max(1.0) * 1.0e-5,
            "screen error at source slot {i}: {error} != {}",
            cpu.screen_error
        );
    }
}

#[test]
fn resident_tail_boundaries_incremental_upload_and_count_reset() {
    let ctx = gpu_context();
    let input = camera();
    let mut proxies = boundary_proxies();
    let mut resident = GpuResidentVisibility::new(&ctx, 129, 7).unwrap();
    assert_eq!(resident.dense_count(), 0);
    assert_eq!(resident.allocated_buffer_bytes(), 72 * 129 + 144);
    let original_indices = resident.compact_indices() as *const wgpu::Buffer;
    let original_args = resident.indirect_args() as *const wgpu::Buffer;
    let original_records = resident.visibility_records() as *const wgpu::Buffer;
    assert!(matches!(
        resident.set_dense_count(1),
        Err(GpuResidentError::UninitializedDenseRange { .. })
    ));

    // An uploaded tail cannot accidentally activate an unwritten prefix.
    resident
        .write_range(128, &ProxySoa::from_proxies(&proxies[128..]).unwrap())
        .unwrap();
    assert!(matches!(
        resident.set_dense_count(129),
        Err(GpuResidentError::UninitializedDenseRange { .. })
    ));
    resident
        .write_range(0, &ProxySoa::from_proxies(&proxies[..128]).unwrap())
        .unwrap();
    assert_eq!(
        resident.dense_count(),
        0,
        "uploads do not implicitly set count"
    );
    // Encoding performs no implicit upload, and stale count/camera uniforms
    // are rejected before recording any command.
    resident.set_dense_count(129).unwrap();
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    assert_eq!(
        resident.encode(input, &mut encoder),
        Err(GpuResidentError::CameraNotUploaded)
    );
    resident.write_camera(input).unwrap();
    resident.set_dense_count(128).unwrap();
    assert_eq!(
        resident.encode(input, &mut encoder),
        Err(GpuResidentError::CameraNotUploaded)
    );
    resident.set_dense_count(129).unwrap();
    let mut changed = input;
    changed.camera_position[0] = 1.0;
    assert_eq!(
        resident.encode(changed, &mut encoder),
        Err(GpuResidentError::CameraNotUploaded)
    );
    changed.max_distance = f32::NAN;
    assert_eq!(
        resident.write_camera(changed),
        Err(GpuResidentError::InvalidCamera)
    );
    assert_eq!(
        resident.encode(changed, &mut encoder),
        Err(GpuResidentError::InvalidCamera)
    );
    drop(encoder);
    for count in [0, 1, 63, 64, 65, 128, 129, 0, 129] {
        resident.set_dense_count(count).unwrap();
        let buffer = submit_snapshot(&ctx, &mut resident, input);
        assert_snapshot(&read(&ctx, &buffer), 129, 7, input, &proxies[..count]);
    }

    // A late invalid row must prevent even the earlier valid row's upload.
    let mut invalid = ProxySoa::from_proxies(&proxies[..2]).unwrap();
    invalid.positions[0][0] = 100.0;
    invalid.errors[1][0] = f32::NAN;
    assert_eq!(
        resident.write_range(0, &invalid),
        Err(GpuResidentError::InvalidProxy { index: 1 })
    );
    invalid.errors.pop();
    assert_eq!(
        resident.write_range(0, &invalid),
        Err(GpuResidentError::ColumnLengthMismatch)
    );
    let one = ProxySoa::from_proxies(&proxies[..1]).unwrap();
    assert_eq!(
        resident.write_range(usize::MAX, &one),
        Err(GpuResidentError::ArithmeticOverflow)
    );
    assert!(matches!(
        resident.write_range(129, &one),
        Err(GpuResidentError::CapacityExceeded { .. })
    ));
    assert!(matches!(
        resident.set_dense_count(130),
        Err(GpuResidentError::CapacityExceeded { .. })
    ));
    assert_eq!(resident.dense_count(), 129);
    let empty = ProxySoa::from_proxies(&[]).unwrap();
    resident.write_range(129, &empty).unwrap();
    let buffer = submit_snapshot(&ctx, &mut resident, input);
    assert_snapshot(&read(&ctx, &buffer), 129, 7, input, &proxies);

    // Update a middle range and the final lane of the third workgroup.
    for proxy in &mut proxies[63..66] {
        proxy.position = [100.0, 0.0, 0.0];
    }
    proxies[128].position = [100.0, 0.0, 0.0];
    resident
        .write_range(63, &ProxySoa::from_proxies(&proxies[63..66]).unwrap())
        .unwrap();
    resident
        .write_range(128, &ProxySoa::from_proxies(&proxies[128..]).unwrap())
        .unwrap();
    let buffer = submit_snapshot(&ctx, &mut resident, input);
    assert_snapshot(&read(&ctx, &buffer), 129, 7, input, &proxies);

    // Two sequential submissions without a host wait: neither count nor camera
    // can leak from a previous frame. Snapshot copies preserve each result.
    let full = submit_snapshot(&ctx, &mut resident, input);
    let mut rejecting = input;
    rejecting.max_distance = 0.0;
    resident.set_dense_count(1).unwrap();
    let culled = submit_snapshot(&ctx, &mut resident, rejecting);
    resident.set_dense_count(0).unwrap();
    let zero = submit_snapshot(&ctx, &mut resident, input);
    assert_snapshot(&read(&ctx, &full), 129, 7, input, &proxies);
    assert_snapshot(&read(&ctx, &culled), 129, 7, rejecting, &proxies[..1]);
    assert_snapshot(&read(&ctx, &zero), 129, 7, input, &[]);
    assert_eq!(
        resident.compact_indices() as *const wgpu::Buffer,
        original_indices
    );
    assert_eq!(
        resident.indirect_args() as *const wgpu::Buffer,
        original_args
    );
    assert_eq!(
        resident.visibility_records() as *const wgpu::Buffer,
        original_records
    );

    let mut zero_capacity = GpuResidentVisibility::new(&ctx, 0, 11).unwrap();
    assert_eq!(zero_capacity.allocated_buffer_bytes(), 72 + 144);
    zero_capacity.write_range(0, &empty).unwrap();
    zero_capacity.set_dense_count(0).unwrap();
    let buffer = submit_snapshot(&ctx, &mut zero_capacity, input);
    assert_snapshot(&read(&ctx, &buffer), 0, 11, input, &[]);
    assert!(matches!(
        GpuResidentVisibility::new(&ctx, usize::MAX, 3),
        Err(GpuResidentError::ArithmeticOverflow)
    ));
    let too_large = (ctx.device().limits().max_storage_buffer_binding_size as usize / 20) + 1;
    assert!(matches!(
        GpuResidentVisibility::new(&ctx, too_large, 3),
        Err(GpuResidentError::DeviceLimit { .. })
    ));
}

const DRAW_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> indices: array<u32>;
struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) slot: u32,
};
@vertex fn vs(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VertexOut {
    let triangle = array<vec2<f32>, 3>(vec2<f32>(-0.3, -0.5), vec2<f32>(0.3, -0.5), vec2<f32>(0.0, 0.5));
    let slot = indices[instance];
    let center = vec2<f32>(f32(slot) * 0.5 - 1.0, 0.0);
    var out: VertexOut;
    out.position = vec4<f32>(triangle[vertex] + center, 0.5, 1.0);
    out.slot = slot;
    return out;
}
@fragment fn fs(in: VertexOut) -> @location(0) vec4<f32> {
    if in.slot == 1u { return vec4<f32>(0.0, 1.0, 0.0, 1.0); }
    if in.slot == 3u { return vec4<f32>(1.0, 0.0, 0.0, 1.0); }
    return vec4<f32>(0.0, 0.0, 1.0, 1.0);
}
"#;

struct DrawHarness {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    target: wgpu::Texture,
    pixels: wgpu::Buffer,
}

impl DrawHarness {
    fn new(ctx: &GpuContext, resident: &GpuResidentVisibility) -> Self {
        let device = ctx.device();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("resident compact source-slot triangle"),
            source: wgpu::ShaderSource::Wgsl(DRAW_SHADER.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("resident compact indices indirect raster"),
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
            label: Some("resident indices consumed by vertex shader"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: resident.compact_indices().as_entire_binding(),
            }],
        });
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("resident indirect offscreen target"),
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

    fn draw(
        &self,
        ctx: &GpuContext,
        resident: &mut GpuResidentVisibility,
        proxies: &[ProxyInstance],
    ) -> Vec<u8> {
        let mut encoder = ctx.device().create_command_encoder(&Default::default());
        resident.write_camera(camera()).unwrap();
        resident.encode(camera(), &mut encoder).unwrap();
        let view = self.target.create_view(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("resident actual draw_indirect"),
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
            // No CPU count, readback, direct draw, or separate submission.
            pass.draw_indirect(resident.indirect_args(), 0);
        }
        let output = capture(ctx, resident, &mut encoder);
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
        let pixels = read(ctx, &self.pixels);
        assert_snapshot(
            &read(ctx, &output),
            resident.capacity(),
            3,
            camera(),
            proxies,
        );
        pixels
    }
}

fn pixel(bytes: &[u8], x: usize, y: usize) -> &[u8] {
    let offset = (y * 64 + x) * 4;
    &bytes[offset..offset + 4]
}

#[test]
fn resident_compact_indices_feed_a_real_indirect_instanced_draw() {
    let ctx = gpu_context();
    let mut resident = GpuResidentVisibility::new(&ctx, 4, 3).unwrap();
    let mut proxies = vec![
        ProxyInstance::new(70, 0, [100.0, 0.0, 0.0], 0.0, 1.0, 100.0),
        ProxyInstance::new(70, 0, [0.0, 0.0, 1.0], 0.0, 1.0, 100.0),
        ProxyInstance::new(70, 0, [100.0, 0.0, 0.0], 0.0, 1.0, 100.0),
        ProxyInstance::new(70, 0, [0.0, 0.0, 1.0], 0.0, 1.0, 100.0),
    ];
    resident
        .write_range(0, &ProxySoa::from_proxies(&proxies).unwrap())
        .unwrap();
    resident.set_dense_count(4).unwrap();
    let draw = DrawHarness::new(&ctx, &resident);
    let two = draw.draw(&ctx, &mut resident, &proxies);
    assert_eq!(pixel(&two, 16, 32), &[0, 255, 0, 255]);
    assert_eq!(pixel(&two, 48, 32), &[255, 0, 0, 255]);
    assert_eq!(pixel(&two, 32, 32), &[0, 0, 0, 255]);
    assert_eq!(pixel(&two, 0, 0), &[0, 0, 0, 255]);
    assert!(
        !two.chunks_exact(4).any(|rgba| rgba == [0, 0, 255, 255]),
        "vertex shader must consume source slots 1 and 3"
    );

    proxies[1].position = [100.0, 0.0, 0.0];
    resident
        .write_range(1, &ProxySoa::from_proxies(&proxies[1..2]).unwrap())
        .unwrap();
    let one = draw.draw(&ctx, &mut resident, &proxies);
    assert_eq!(pixel(&one, 16, 32), &[0, 0, 0, 255]);
    assert_eq!(pixel(&one, 48, 32), &[255, 0, 0, 255]);
    resident.set_dense_count(0).unwrap();
    let zero = draw.draw(&ctx, &mut resident, &[]);
    assert!(zero.chunks_exact(4).all(|rgba| rgba == [0, 0, 0, 255]));
    resident.set_dense_count(4).unwrap();
    assert_eq!(draw.draw(&ctx, &mut resident, &proxies), one);
}

#[test]
fn compact_and_test_draw_shaders_validate_without_a_device() {
    for source in [
        include_str!("../src/shaders/proxy_compact.wgsl"),
        DRAW_SHADER,
    ] {
        let module = wgpu::naga::front::wgsl::parse_str(source).expect("resident WGSL parses");
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("resident WGSL validates");
    }
}
