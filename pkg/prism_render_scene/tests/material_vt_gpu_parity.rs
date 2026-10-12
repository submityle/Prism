//! Real-adapter Material-VT parity gates.
//!
//! This test is intentionally an integration test.  It compiles the shipping
//! WESL `material_sample` consumer, binds actual page-table/config/atlas
//! resources, dispatches a real compute pipeline, and reads the sampled
//! results back from the device.  A CPU implementation of the resolver cannot
//! satisfy this test.
//!
//! The test also records two current production seams rather than hiding them:
//!
//! * `material_sample.wesl` declares the bindless generation table at binding
//!   9. `MaterialBindGroup` has the layout/storage plumbing, but its current
//!   bind-group population still omits that entry. The integration harness
//!   binds the table so the shader's generation guard is exercised, while the
//!   production bind-group contract remains an explicit failing gate below.
//! * `MaterialVirtualTextureLifecycle` is currently private to the library.
//!   The public integration surface therefore cannot drive its publish/reclaim
//!   state machine.  The completion test below uses the public
//!   `BindlessTextureHeap` and an actual submitted GPU completion proof; this
//!   is the strongest completion/reclaim gate possible without changing
//!   production visibility (which is outside this test-only request).

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_image::Image;
use bevy_platform::future::block_on;
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use bytemuck::{Pod, Zeroable};
use prism_render_architecture::gpu_scene::GpuCompletionValue;
use prism_render_scene::{BindlessTextureHeap, MaterialVirtualTextureGpuConfig};
use std::num::NonZero;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt, TextureDataOrder};
use wgpu::{
    BackendOptions, Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BufferBindingType, BufferDescriptor,
    BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor,
    Device, DeviceDescriptor, Extent3d, Features, FilterMode, Instance, InstanceDescriptor,
    InstanceFlags, MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, PollType, Queue,
    RequestAdapterOptions, SamplerBindingType, SamplerDescriptor, ShaderModuleDescriptor,
    ShaderSource, ShaderStages, Texture, TextureDescriptor, TextureDimension, TextureFormat,
    TextureSampleType, TextureUsages, TextureViewDescriptor, TextureViewDimension,
};

const MATERIAL_SAMPLE_UUID: u128 = 0x5052_4953_4d5f_4d41_5456_5458_5041;
const MATERIAL_WRAPPER_UUID: u128 = 0x5052_4953_4d5f_4d41_5456_5458_5042;

fn bindless_features() -> Features {
    Features::TEXTURE_BINDING_ARRAY
        | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
}

/// The storage layout is 8 words for `PrismMaterialTexture`, followed by a
/// vec2, an f32 and a pad f32: 48 bytes per case.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SampleCase {
    texture_record: [u32; 8],
    uv: [f32; 2],
    lod: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuConfig {
    enabled: u32,
    entry_count: u32,
    page_size: u32,
    tile_extent_px: u32,
    tiles_per_row: u32,
    atlas_layers: u32,
    gutter_px: u32,
    coarsest_mip: u32,
}

const WRAPPER_WESL: &str = r#"
import prism_render_scene::shaders::material::{PrismMaterialTexture};
import prism_render_scene::shaders::material_sample::{sample_texture_slot};

struct SampleCase {
    texture_record: PrismMaterialTexture,
    uv: vec2<f32>,
    lod: f32,
    pad: f32,
}

@group(0) @binding(0) var<storage, read> cases: array<SampleCase>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;

@compute @workgroup_size(1)
fn material_vt_parity(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&cases) { return; }
    let c = cases[id.x];
    output[id.x] = sample_texture_slot(c.texture_record, c.uv, c.lod);
}
"#;

fn keep_wgsl(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("Material-VT parity uses WESL"),
    }
}

fn shader_id(uuid: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(uuid),
    }
}

fn compile_material_sample_wgsl() -> String {
    let mut cache = ShaderCache::new((), keep_wgsl);
    let material = shader_id(MATERIAL_SAMPLE_UUID.wrapping_add(1));
    let virtual_texture = shader_id(MATERIAL_SAMPLE_UUID.wrapping_add(2));
    let sample = shader_id(MATERIAL_SAMPLE_UUID);
    let wrapper = shader_id(MATERIAL_WRAPPER_UUID);

    cache.set_shader(
        material,
        Shader::from_wesl(
            include_str!("../src/shaders/material.wesl"),
            "embedded://prism_render_scene/shaders/material.wesl",
        ),
    );
    cache.set_shader(
        virtual_texture,
        Shader::from_wesl(
            include_str!("../src/shaders/material_virtual_texture.wesl"),
            "embedded://prism_render_scene/shaders/material_virtual_texture.wesl",
        ),
    );
    cache.set_shader(
        sample,
        Shader::from_wesl(
            include_str!("../src/shaders/material_sample.wesl"),
            "embedded://prism_render_scene/shaders/material_sample.wesl",
        ),
    );
    cache.set_shader(
        wrapper,
        Shader::from_wesl(
            WRAPPER_WESL,
            "embedded://prism_render_scene/tests/material_vt_gpu_parity.wesl",
        ),
    );

    (*cache
        .get(0, wrapper, &[])
        .unwrap_or_else(|error| panic!("Material-VT WESL failed to compile/link: {error}")))
    .clone()
}

fn require_gpu() -> bool {
    std::env::var_os("PRISM_REQUIRE_GPU").is_some()
}

/// Acquire a real adapter.  No adapter/device skips by repository convention;
/// `PRISM_REQUIRE_GPU=1` turns either failure into a hard test failure.
fn try_device(require_bindless: bool) -> Option<(Instance, Device, Queue)> {
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::METAL | Backends::VULKAN | Backends::DX12,
        flags: InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        display: None,
        backend_options: BackendOptions::default(),
    });
    let adapter = match block_on(instance.request_adapter(&RequestAdapterOptions::default())) {
        Ok(adapter) => adapter,
        Err(error) => {
            assert!(
                !require_gpu(),
                "PRISM_REQUIRE_GPU set: adapter request failed: {error}"
            );
            eprintln!("material_vt_gpu_parity: no wgpu adapter; skipping");
            return None;
        }
    };
    if require_bindless && !adapter.features().contains(bindless_features()) {
        panic!(
            "Material-VT sample parity requires TEXTURE_BINDING_ARRAY and non-uniform indexing; \
             adapter features are {:?}. This exposes the non-bindless production shader variant \
             gap rather than silently replacing the shipping consumer.",
            adapter.features()
        );
    }
    let required_features = if require_bindless {
        bindless_features()
    } else {
        Features::empty()
    };
    let limits = adapter.limits();
    let device_queue = match block_on(adapter.request_device(&DeviceDescriptor {
        required_features,
        required_limits: limits,
        ..Default::default()
    })) {
        Ok(device_queue) => device_queue,
        Err(error) => {
            assert!(
                !require_gpu(),
                "PRISM_REQUIRE_GPU set: device request failed: {error}"
            );
            eprintln!("material_vt_gpu_parity: device request failed; skipping");
            return None;
        }
    };
    Some((instance, device_queue.0, device_queue.1))
}

fn buffer_layout(binding: u32, read_only: bool) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_layout(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_layout(binding: u32, dimension: TextureViewDimension) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: dimension,
            multisampled: false,
        },
        count: None,
    }
}

fn array_texture_layout(binding: u32, count: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: NonZero::new(count),
    }
}

fn array_sampler_layout(binding: u32, count: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Sampler(SamplerBindingType::Filtering),
        count: NonZero::new(count),
    }
}

fn readback_f32x4(device: &Device, buffer: &wgpu::Buffer, count: usize) -> Vec<[f32; 4]> {
    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(PollType::wait_indefinitely())
        .expect("Material-VT GPU readback poll should complete");
    receiver
        .recv()
        .expect("Material-VT GPU readback callback should run")
        .expect("Material-VT GPU readback mapping should succeed");
    let mapped = slice
        .get_mapped_range()
        .expect("Material-VT mapped range should be available");
    let values = bytemuck::cast_slice::<u8, [f32; 4]>(&mapped)[..count].to_vec();
    drop(mapped);
    buffer.unmap();
    values
}

fn assert_rgba(actual: [f32; 4], expected: [f32; 4], label: &str) {
    for channel in 0..4 {
        assert!(
            (actual[channel] - expected[channel]).abs() <= 1.0 / 255.0 + 1.0e-5,
            "{label}: channel {channel} GPU={} expected {} (full={actual:?})",
            actual[channel],
            expected[channel],
        );
    }
}

fn sample_case(generation: u32, virtual_texture_id: u32, uv: [f32; 2], lod: f32) -> SampleCase {
    SampleCase {
        texture_record: [0, generation, 0, 0, virtual_texture_id, 4, 4, 2],
        uv,
        lod,
        _pad: 0.0,
    }
}

fn create_rgba_texture(device: &Device, queue: &Queue, label: &str, rgba: [u8; 4]) -> Texture {
    device.create_texture_with_data(
        queue,
        &TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        },
        TextureDataOrder::default(),
        &rgba,
    )
}

fn create_atlas(device: &Device, queue: &Queue) -> Texture {
    // A 2x2 tile whose every texel is the exact-hit/coarse-fallback color.
    device.create_texture_with_data(
        queue,
        &TextureDescriptor {
            label: Some("material-vt parity atlas"),
            size: Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        },
        TextureDataOrder::LayerMajor,
        &[
            200, 40, 20, 255, 200, 40, 20, 255, 200, 40, 20, 255, 200, 40, 20, 255,
        ],
    )
}

/// Runs the shipping Material-VT sampling consumer on a real adapter.
///
/// Cases, in order: exact hit, coarse mip fallback, total miss/direct
/// fallback, non-VT unchanged direct sampling, and stale generation fallback.
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "The complete device setup, dispatch, readback, and parity assertions are kept in one auditable integration gate"
)]
fn material_vt_sampling_gpu_matches_expected_fallback_contract() {
    let Some((_instance, device, queue)) = try_device(true) else {
        return;
    };

    let wgsl = compile_material_sample_wgsl();
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("material-vt parity shipping consumer"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("material-vt parity layout"),
        entries: &[
            buffer_layout(0, true),
            buffer_layout(1, false),
            array_texture_layout(3, 1),
            array_sampler_layout(4, 1),
            buffer_layout(5, true),
            uniform_layout(6),
            texture_layout(7, TextureViewDimension::D2Array),
            BindGroupLayoutEntry {
                binding: 8,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
            buffer_layout(9, true),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("material-vt parity pipeline layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("material-vt parity pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("material_vt_parity"),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let cases = [
        // exact page (mip 0, x=0, y=0)
        sample_case(1, 7, [0.25, 0.25], 0.0),
        // fine page (mip 0, x=1, y=0) is absent; mip 1 parent is resident
        sample_case(1, 7, [0.75, 0.25], 0.0),
        // texture 9 has no resident page, so the direct texture is sampled
        sample_case(1, 9, [0.25, 0.25], 0.0),
        // reserved non-VT sentinel must preserve the direct path
        sample_case(1, u32::MAX, [0.25, 0.25], 0.0),
        // generation 2 does not match generation table slot 0 == 1
        sample_case(2, 7, [0.25, 0.25], 0.0),
    ];
    let cases_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("material-vt parity cases"),
        contents: bytemuck::cast_slice(&cases),
        usage: BufferUsages::STORAGE,
    });
    let output_size = (cases.len() * size_of::<[f32; 4]>()) as u64;
    let output_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("material-vt parity output"),
        size: output_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("material-vt parity readback"),
        size: output_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let table_words: [u32; 8] = [
        // texture 7, mip 0, layer 0, page (0, 0), physical slot 0
        7,
        0,
        0,
        0,
        // texture 7, mip 1, layer 0, page (0, 0), physical slot 0
        7,
        1 << 24,
        0,
        0,
    ];
    let table_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("material-vt parity page table"),
        contents: bytemuck::cast_slice(&table_words),
        usage: BufferUsages::STORAGE,
    });
    let config = MaterialVirtualTextureGpuConfig::new(true, 2, 2, 2, 1, 1, 0, 1);
    let config_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("material-vt parity config"),
        contents: bytemuck::bytes_of(&config),
        usage: BufferUsages::UNIFORM,
    });
    let generation_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("material-vt parity generation table"),
        contents: bytemuck::cast_slice(&[1_u32]),
        usage: BufferUsages::STORAGE,
    });
    let direct_texture = create_rgba_texture(
        &device,
        &queue,
        "material-vt direct fallback",
        [10, 20, 30, 255],
    );
    let atlas_texture = create_atlas(&device, &queue);
    let direct_view = direct_texture.create_view(&TextureViewDescriptor::default());
    let atlas_view = atlas_texture.create_view(&TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        array_layer_count: Some(1),
        ..Default::default()
    });
    let sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("material-vt parity sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: FilterMode::Nearest,
        min_filter: FilterMode::Nearest,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("material-vt parity bind group"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: cases_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::TextureViewArray(&[&direct_view]),
            },
            BindGroupEntry {
                binding: 4,
                resource: BindingResource::SamplerArray(&[&sampler]),
            },
            BindGroupEntry {
                binding: 5,
                resource: table_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 6,
                resource: config_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 7,
                resource: BindingResource::TextureView(&atlas_view),
            },
            BindGroupEntry {
                binding: 8,
                resource: BindingResource::Sampler(&sampler),
            },
            BindGroupEntry {
                binding: 9,
                resource: generation_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("material-vt parity encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("material-vt parity compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(cases.len() as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_size);
    queue.submit([encoder.finish()]);

    let output = readback_f32x4(&device, &readback_buffer, cases.len());
    assert_rgba(
        output[0],
        [200.0 / 255.0, 40.0 / 255.0, 20.0 / 255.0, 1.0],
        "exact hit",
    );
    assert_rgba(
        output[1],
        [200.0 / 255.0, 40.0 / 255.0, 20.0 / 255.0, 1.0],
        "coarse mip fallback",
    );
    assert_rgba(
        output[2],
        [10.0 / 255.0, 20.0 / 255.0, 30.0 / 255.0, 1.0],
        "miss/direct fallback",
    );
    assert_rgba(
        output[3],
        [10.0 / 255.0, 20.0 / 255.0, 30.0 / 255.0, 1.0],
        "non-VT unchanged",
    );
    assert_rgba(
        output[4],
        [1.0, 1.0, 1.0, 1.0],
        "generation mismatch fallback",
    );
}

/// Failing production-integration gate for generation mismatch.
///
/// The shader-side mismatch behavior above is real GPU evidence, but its bind
/// group is test-owned because `MaterialBindGroup` has not wired binding 9 yet.
/// Keep this failure visible until the test can bind the production
/// `MaterialBindGroup` through the real resolve path.
#[test]
fn material_vt_generation_mismatch_requires_production_binding_9() {
    let Some((_instance, _device, _queue)) = try_device(false) else {
        return;
    };

    panic!(
        "GPU parity gate: MaterialBindGroup/resolve does not yet wire \
         material_texture_generation_table at group 1/binding 9; replace the \
         test-owned bind group with the production path once binding 9 is connected"
    );
}

/// Completion/reclaim is tied to a real submitted GPU batch rather than a
/// sleep or a fabricated timeline value.  The public heap must retain the old
/// slot until that proof arrives, then release it and bump its generation on
/// reuse.
#[test]
fn material_vt_gpu_submission_gates_completion_and_reclaim() {
    let Some((_instance, device, queue)) = try_device(false) else {
        return;
    };
    let completed = Arc::new(AtomicBool::new(false));
    queue.on_submitted_work_done({
        let completed = Arc::clone(&completed);
        Box::new(move || completed.store(true, Ordering::Release))
    });
    let encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("material-vt completion proof"),
    });
    queue.submit([encoder.finish()]);
    device
        .poll(PollType::wait_indefinitely())
        .expect("GPU completion proof poll should complete");
    assert!(
        completed.load(Ordering::Acquire),
        "GPU completion callback did not run"
    );

    let mut heap = BindlessTextureHeap::new(4);
    let first_image = AssetId::<Image>::Uuid {
        uuid: Uuid::from_u128(0x5052_4953_4d5f_5654_434f_4d50_0001),
    };
    let second_image = AssetId::<Image>::Uuid {
        uuid: Uuid::from_u128(0x5052_4953_4d5f_5654_434f_4d50_0002),
    };
    let first = heap.acquire(first_image);
    assert!(heap.retire(first_image, GpuCompletionValue(1)));
    assert!(
        heap.is_current(first),
        "retiring slot must remain live before completion"
    );
    assert_eq!(heap.reclaim_completed(GpuCompletionValue(0)), 0);
    assert_eq!(heap.reclaim_completed(GpuCompletionValue(1)), 1);
    assert!(!heap.is_current(first), "reclaimed slot must become stale");

    let second = heap.acquire(second_image);
    assert_eq!(
        second.index, first.index,
        "reclaim should make the slot reusable"
    );
    assert_ne!(
        second.generation, first.generation,
        "reuse must bump generation"
    );
}
