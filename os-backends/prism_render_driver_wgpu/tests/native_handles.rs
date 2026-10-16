//! Native exports of resources created by the adopted RHI resource factory.
//!
//! These are not raw imports or cross-Instance identity proofs. All native
//! objects originate from one host tuple and one shared RHI resource table.
//! PRISM_REQUIRE_GPU=1 makes unavailable/non-hardware adapters a failure.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

use prism_render_driver as rhi;
use prism_render_driver_wgpu::{
    require_hardware_gpu, AdapterEvidence, WgpuAdapter, WgpuDevice, WgpuInstance, WgpuQueue,
};
use rhi::{Adapter as _, RenderDevice as _};

struct Host {
    _instance: WgpuInstance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Host {
    fn adopt(&self) -> (WgpuDevice, WgpuQueue) {
        WgpuDevice::adopt(
            self.adapter.clone(),
            self.device.clone(),
            self.queue.clone(),
        )
    }
}

fn host_or_skip() -> Option<Host> {
    let instance = WgpuInstance::new();
    let Some(adapter) = instance.request_adapter(wgpu::PowerPreference::HighPerformance, false)
    else {
        assert!(
            !require_hardware_gpu(),
            "PRISM_REQUIRE_GPU=1 but no adapter"
        );
        eprintln!("SKIPPED native handle GPU test: no adapter");
        return None;
    };
    let description = WgpuAdapter::new(adapter.clone());
    let evidence = AdapterEvidence::from_info(description.info());
    if !evidence.is_hardware() {
        assert!(
            !require_hardware_gpu(),
            "PRISM_REQUIRE_GPU=1 requires physical hardware: {evidence}"
        );
        eprintln!("SKIPPED native handle physical-GPU evidence: {evidence}");
        return None;
    }
    eprintln!("native handle REAL HARDWARE GPU: {evidence}");
    let (device, queue) =
        futures_lite::future::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("native-handles-host"),
            required_features: wgpu::Features::empty(),
            ..Default::default()
        }))
        .expect("physical adapter must create the host device");
    Some(Host {
        _instance: instance,
        adapter,
        device,
        queue,
    })
}

fn buffer(device: &WgpuDevice) -> rhi::BufferId {
    device.create_buffer(&rhi::BufferDescriptor::new(
        16,
        rhi::BufferUsages::STORAGE | rhi::BufferUsages::COPY_SRC,
    ))
}

fn texture(device: &WgpuDevice) -> rhi::TextureId {
    device.create_texture(&rhi::TextureDescriptor::new_2d(
        rhi::Extent3d::new_2d(1, 1),
        rhi::TextureFormat::Rgba8Unorm,
        rhi::TextureUsages::RENDER_ATTACHMENT
            | rhi::TextureUsages::TEXTURE_BINDING
            | rhi::TextureUsages::COPY_SRC,
    ))
}

fn texture_view(device: &WgpuDevice) -> rhi::TextureViewId {
    device.create_texture_view(texture(device), &rhi::TextureViewDescriptor::default())
}

fn sampler(device: &WgpuDevice) -> rhi::SamplerId {
    device.create_sampler(&rhi::SamplerDescriptor::default())
}

// Shared by the ID-validity tests. No resources are required by its entry points.
const SIMPLE_SHADER: &str = r#"
@compute @workgroup_size(1) fn compute_main() {}

@vertex fn vertex_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(positions[index], 0.0, 1.0);
}

@fragment fn fragment_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
"#;

fn shader(device: &WgpuDevice) -> rhi::ShaderModuleId {
    device.create_shader_module(&rhi::ShaderModuleDescriptor::wgsl(SIMPLE_SHADER))
}

fn bind_group_layout(device: &WgpuDevice) -> rhi::BindGroupLayoutId {
    device.create_bind_group_layout(&rhi::BindGroupLayoutDescriptor::default())
}

fn bind_group(device: &WgpuDevice) -> rhi::BindGroupId {
    device.create_bind_group(&rhi::BindGroupDescriptor {
        layout: Some(bind_group_layout(device)),
        ..Default::default()
    })
}

fn pipeline_layout(device: &WgpuDevice) -> rhi::PipelineLayoutId {
    device.create_pipeline_layout(&rhi::PipelineLayoutDescriptor::default())
}

fn render_pipeline_for(device: &WgpuDevice, module: rhi::ShaderModuleId) -> rhi::RenderPipelineId {
    device.create_render_pipeline(&rhi::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: rhi::VertexState {
            module,
            entry_point: "vertex_main".into(),
            buffers: Vec::new(),
        },
        primitive: rhi::PrimitiveState {
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: rhi::MultisampleState::default(),
        fragment: Some(rhi::FragmentState {
            module,
            entry_point: "fragment_main".into(),
            targets: vec![Some(rhi::ColorTargetState::opaque(
                rhi::TextureFormat::Rgba8Unorm,
            ))],
        }),
    })
}

fn render_pipeline(device: &WgpuDevice) -> rhi::RenderPipelineId {
    render_pipeline_for(device, shader(device))
}

fn compute_pipeline(device: &WgpuDevice) -> rhi::ComputePipelineId {
    device.create_compute_pipeline(&rhi::ComputePipelineDescriptor {
        label: None,
        layout: Some(pipeline_layout(device)),
        module: shader(device),
        entry_point: "compute_main".into(),
    })
}

fn assert_invalid<T>(lookup: impl FnOnce() -> T) {
    assert!(
        catch_unwind(AssertUnwindSafe(lookup)).is_err(),
        "native getter must reject missing/stale IDs"
    );
}

// Exercise each public getter against its actual GPU-backed table. The IDs are
// not caller-origin tokens; equality here is only within the one host Global.
macro_rules! getter_lifecycle_test {
    ($name:ident, $create:ident, $get:ident, $destroy:ident) => {
        #[test]
        fn $name() {
            let Some(host) = host_or_skip() else {
                return;
            };
            let (device, queue) = host.adopt();
            let shared = device.clone();
            assert_invalid(|| device.$get(rhi::ResourceId::from_parts(u32::MAX, 0)));
            let original_id = $create(&device);
            let original = device.$get(original_id);
            assert_eq!(original, device.$get(original_id));
            assert_eq!(original, shared.$get(original_id));

            device.$destroy(original_id);
            assert_invalid(|| shared.$get(original_id));
            let replacement_id = $create(&device);
            assert_eq!(original_id.index(), replacement_id.index());
            assert_ne!(original_id.generation(), replacement_id.generation());
            assert_invalid(|| device.$get(original_id));
            assert_eq!(
                device.$get(replacement_id),
                shared.$get(replacement_id),
                "a failed stale lookup must not poison subsequent table access"
            );
            shared.$destroy(replacement_id);
            drop((shared, queue, device));
            // This handle has its own native reference. Actual use after RHI
            // destruction is tested by the draw/dispatch/readback below.
            assert_eq!(original, original.clone());
        }
    };
}

getter_lifecycle_test!(
    buffer_export_validates_live_generation,
    buffer,
    wgpu_buffer,
    destroy_buffer
);
getter_lifecycle_test!(
    texture_export_validates_live_generation,
    texture,
    wgpu_texture,
    destroy_texture
);
getter_lifecycle_test!(
    texture_view_export_validates_live_generation,
    texture_view,
    wgpu_texture_view,
    destroy_texture_view
);
getter_lifecycle_test!(
    sampler_export_validates_live_generation,
    sampler,
    wgpu_sampler,
    destroy_sampler
);
getter_lifecycle_test!(
    shader_module_export_validates_live_generation,
    shader,
    wgpu_shader_module,
    destroy_shader_module
);
getter_lifecycle_test!(
    bind_group_layout_export_validates_live_generation,
    bind_group_layout,
    wgpu_bind_group_layout,
    destroy_bind_group_layout
);
getter_lifecycle_test!(
    bind_group_export_validates_live_generation,
    bind_group,
    wgpu_bind_group,
    destroy_bind_group
);
getter_lifecycle_test!(
    pipeline_layout_export_validates_live_generation,
    pipeline_layout,
    wgpu_pipeline_layout,
    destroy_pipeline_layout
);
getter_lifecycle_test!(
    render_pipeline_export_validates_live_generation,
    render_pipeline,
    wgpu_render_pipeline,
    destroy_render_pipeline
);
getter_lifecycle_test!(
    compute_pipeline_export_validates_live_generation,
    compute_pipeline,
    wgpu_compute_pipeline,
    destroy_compute_pipeline
);

fn read_mapped(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u8> {
    let result = Arc::new(Mutex::new(None));
    let callback_result = Arc::clone(&result);
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |status| {
            *callback_result.lock().expect("map result lock") = Some(status);
        });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("host poll must complete mapping");
    assert_eq!(result.lock().expect("map result lock").take(), Some(Ok(())));
    let mapped = buffer
        .slice(..)
        .get_mapped_range()
        .expect("successful native mapping");
    let bytes = mapped.to_vec();
    drop(mapped);
    buffer.unmap();
    bytes
}

const SAMPLED_COMPUTE: &str = r#"
@group(0) @binding(0) var<storage, read_write> result: vec4<u32>;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var nearest_sampler: sampler;

@compute @workgroup_size(1) fn sample_main() {
    let color = textureSampleLevel(source, nearest_sampler, vec2<f32>(0.5), 0.0);
    result += vec4<u32>(round(color * 255.0));
}
"#;

#[test]
fn exported_objects_draw_sample_and_read_back_after_rhi_tables_are_released() {
    let Some(host) = host_or_skip() else {
        return;
    };
    let (device, queue) = host.adopt();
    let storage = buffer(&device);
    let image = texture(&device);
    let view = device.create_texture_view(image, &rhi::TextureViewDescriptor::default());
    let nearest = sampler(&device);
    // Mapping remains host-owned; RHI resource exports are verified using the
    // storage buffer and texture, not an unsupported RHI mapping usage.
    let native_readback = host.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("host-owned-native-export-staging"),
        size: 512,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let module = device.create_shader_module(&rhi::ShaderModuleDescriptor::wgsl(SAMPLED_COMPUTE));
    let layout = device.create_bind_group_layout(&rhi::BindGroupLayoutDescriptor {
        label: None,
        entries: vec![
            rhi::BindGroupLayoutEntry {
                binding: 0,
                visibility: rhi::ShaderStages::COMPUTE,
                ty: rhi::BindingType::Buffer {
                    ty: rhi::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(16),
                },
                count: None,
            },
            rhi::BindGroupLayoutEntry {
                binding: 1,
                visibility: rhi::ShaderStages::COMPUTE,
                ty: rhi::BindingType::Texture {
                    sample_type: rhi::TextureSampleType::Float,
                    view_dimension: rhi::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            rhi::BindGroupLayoutEntry {
                binding: 2,
                visibility: rhi::ShaderStages::COMPUTE,
                ty: rhi::BindingType::Sampler(rhi::SamplerBindingType::NonFiltering),
                count: None,
            },
        ],
    });
    let group = device.create_bind_group(&rhi::BindGroupDescriptor {
        label: None,
        layout: Some(layout),
        entries: vec![
            rhi::BindGroupEntry {
                binding: 0,
                resource: rhi::BindingResource::Buffer {
                    buffer: storage,
                    offset: 0,
                    size: Some(16),
                },
            },
            rhi::BindGroupEntry {
                binding: 1,
                resource: rhi::BindingResource::TextureView(view),
            },
            rhi::BindGroupEntry {
                binding: 2,
                resource: rhi::BindingResource::Sampler(nearest),
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&rhi::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: vec![layout],
        push_constant_ranges: Vec::new(),
    });
    let compute = device.create_compute_pipeline(&rhi::ComputePipelineDescriptor {
        label: None,
        layout: Some(pipeline_layout),
        module,
        entry_point: "sample_main".into(),
    });
    let raster_module = shader(&device);
    let raster = render_pipeline_for(&device, raster_module);

    // These are the exact objects the RHI created on the host's owning device.
    let native_buffer = device.wgpu_buffer(storage);
    let native_image = device.wgpu_texture(image);
    let native_view = device.wgpu_texture_view(view);
    let native_sampler = device.wgpu_sampler(nearest);
    let native_module = device.wgpu_shader_module(module);
    let native_layout = device.wgpu_bind_group_layout(layout);
    let native_group = device.wgpu_bind_group(group);
    let native_pipeline_layout = device.wgpu_pipeline_layout(pipeline_layout);
    let native_compute = device.wgpu_compute_pipeline(compute);
    let native_raster = device.wgpu_render_pipeline(raster);
    assert_eq!(native_view.texture(), &native_image);

    // Use the exported shader/layout/sampler as actual native creation inputs,
    // not merely values whose IDs compare equal. Both pipelines will dispatch.
    let derived_compute = host
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("host-pipeline-from-rhi-shader-and-layout"),
            layout: Some(&native_pipeline_layout),
            module: &native_module,
            entry_point: Some("sample_main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let derived_group = host.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("host-group-from-rhi-resources-and-layout"),
        layout: &native_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: native_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&native_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&native_sampler),
            },
        ],
    });

    // Remove every RHI slot and drop the adopted pair before recording. The
    // exported native references, not a leaked RHI table, must keep these alive.
    device.destroy_render_pipeline(raster);
    device.destroy_shader_module(raster_module);
    device.destroy_compute_pipeline(compute);
    device.destroy_pipeline_layout(pipeline_layout);
    device.destroy_bind_group(group);
    device.destroy_bind_group_layout(layout);
    device.destroy_shader_module(module);
    device.destroy_sampler(nearest);
    device.destroy_texture_view(view);
    device.destroy_texture(image);
    device.destroy_buffer(storage);
    drop((queue, device));

    let mut encoder = host
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("host-native-exports"),
        });
    {
        let attachments = [Some(wgpu::RenderPassColorAttachment {
            view: &native_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                // Distinct from the draw result: an omitted/no-op draw fails.
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })];
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("exported-raster"),
            color_attachments: &attachments,
            ..Default::default()
        });
        pass.set_pipeline(&native_raster);
        pass.draw(0..3, 0..1);
    }
    // Separate passes ensure storage read/write ordering between dispatches.
    for (pipeline, bind_group) in [
        (&native_compute, &native_group),
        (&derived_compute, &derived_group),
    ] {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&native_buffer, 0, &native_readback, 0, 16);
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &native_image,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &native_readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 256,
                bytes_per_row: Some(256),
                rows_per_image: Some(1),
            },
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let commands = encoder.finish();
    assert_eq!(read_mapped(&host.device, &native_readback), vec![0; 512]);
    // Only the host submits. The resource factory/getters must not submit.
    let submitted = host.queue.submit([commands]);
    host.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submitted),
            timeout: None,
        })
        .expect("host-owned native draw/dispatch must complete");
    let bytes = read_mapped(&host.device, &native_readback);
    let sums: Vec<_> = bytes[..16]
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().expect("one u32")))
        .collect();
    assert_eq!(sums, [510, 0, 0, 510], "both compute pipelines must run");
    assert_eq!(&bytes[256..260], &[255, 0, 0, 255], "actual raster pixel");
}
