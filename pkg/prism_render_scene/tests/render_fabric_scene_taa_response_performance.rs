//! Opt-in performance acceptance for the largest public Render Fabric slice.
//!
//! The measured command chain is the production response path followed by the
//! production HDR bridge and the production TAA resolve shader:
//!
//! `response_composition -> response_to_hdr -> taa_resolve`
//!
//! This is intentionally a separate target from the parity tests.  It reports
//! real host submission/completion/readback timings and, when the adapter
//! exposes `TIMESTAMP_QUERY`, the device-side elapsed time for the whole chain.
//! It does not claim to be a 31-pass application frame: geometry/lighting
//! producers and the host's SceneFabric recorder are outside this public raw
//! wgpu harness.  The response and TAA stages are the actual production
//! shaders/pipelines, and the frame uses two persistent history slots just as
//! the production TAA path does.

use std::{
    hint::black_box,
    mem::size_of,
    sync::mpsc,
    time::{Duration, Instant},
};

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_platform::future::block_on;
use bevy_render::renderer::RenderDevice;
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use prism_render_architecture::{
    backend::capability::CapabilitySet,
    render_fabric::{builtin_catalog, builtin_roots, FeatureSelection, RenderQualityTier},
};
use prism_render_scene::{
    composition::CompositionFabric,
    response_gpu::{
        GpuAuxiliaryMerge, GpuCompositionControl, GpuResponseOutput, GpuWaterCoverage,
        ResponseCompositionPipelines,
    },
};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BackendOptions, Backends, Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device, DeviceDescriptor,
    Extent3d, Features, Instance, InstanceDescriptor, InstanceFlags, MapMode, PollType, QuerySet,
    QuerySetDescriptor, QueryType, Queue, RequestAdapterOptions, ShaderModuleDescriptor,
    ShaderSource, Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    TextureView, TextureViewDescriptor,
};

const DEFAULT_WIDTH: u32 = 640;
const DEFAULT_HEIGHT: u32 = 360;
const DEFAULT_WARMUP: usize = 3;
const DEFAULT_SAMPLES: usize = 31;
const MIB: u64 = 1024 * 1024;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TaaParams {
    width: u32,
    height: u32,
    history_blend: f32,
    variance_gamma: f32,
    valid_history: u32,
}

const _: () = assert!(size_of::<TaaParams>() == 20);

#[derive(Clone, Copy, Debug, Default)]
struct Sample {
    submit_ns: u128,
    completion_ns: u128,
    readback_ns: u128,
    total_ns: u128,
    gpu_ns: Option<u128>,
    checksum: u64,
}

struct PassTimer {
    query_set: QuerySet,
    resolve: Buffer,
    read: Buffer,
}

impl PassTimer {
    fn new(device: &Device) -> Self {
        let query_set = device.create_query_set(&QuerySetDescriptor {
            label: Some("render-frame-performance timestamps"),
            ty: QueryType::Timestamp,
            count: 2,
        });
        let resolve = device.create_buffer(&BufferDescriptor {
            label: Some("render-frame-performance timestamp resolve"),
            size: 16,
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.create_buffer(&BufferDescriptor {
            label: Some("render-frame-performance timestamp readback"),
            size: 16,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            query_set,
            resolve,
            read,
        }
    }

    fn writes_begin(&self) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: None,
        }
    }

    fn writes_end(&self) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: None,
            end_of_pass_write_index: Some(1),
        }
    }
}

struct FrameResources {
    left_seed: Buffer,
    left: Buffer,
    right: Buffer,
    controls: Buffer,
    coverage: Buffer,
    auxiliary_seed: Buffer,
    auxiliary: Buffer,
    hdr: Texture,
    hdr_view: TextureView,
    history: [Texture; 2],
    history_views: [TextureView; 2],
    motion: Texture,
    motion_view: TextureView,
    reactive: Texture,
    reactive_view: TextureView,
    visibility_ids: Texture,
    visibility_ids_view: TextureView,
    visibility_metadata: Texture,
    visibility_metadata_view: TextureView,
    hdr_stage: Buffer,
    auxiliary_stage: Buffer,
    timestamp: Option<PassTimer>,
    response_bind: wgpu::BindGroup,
    auxiliary_bind: wgpu::BindGroup,
    hdr_bind: wgpu::BindGroup,
    taa_bind: [wgpu::BindGroup; 2],
    taa_pipeline: ComputePipeline,
    response_to_hdr: ComputePipeline,
    pipelines: ResponseCompositionPipelines,
    logical_resource_bytes: u64,
    width: u32,
    height: u32,
    auxiliary_bytes: u64,
    padded_row_bytes: u32,
}

struct Harness {
    _instance: Instance,
    device: Device,
    queue: Queue,
    frame: FrameResources,
    timestamp_period_ns: Option<f64>,
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn required_gpu() -> bool {
    std::env::var_os("PRISM_REQUIRE_GPU").is_some()
}

fn percentile(values: &[u128], q: f64) -> u128 {
    assert!(!values.is_empty());
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let index = ((sorted.len() - 1) as f64 * q).ceil() as usize;
    sorted[index]
}

fn percentile_optional(values: &[u128]) -> Option<u128> {
    (!values.is_empty()).then(|| percentile(values, 0.50))
}

fn keep_wgsl(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("TAA benchmark uses WESL"),
    }
}

fn compile_taa_wgsl() -> String {
    let mut cache = ShaderCache::new((), keep_wgsl);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(0x5052_4953_4d5f_4245_4e43_4846_5254_0001),
    };
    cache.set_shader(
        id,
        Shader::from_wesl(
            include_str!("../src/shaders/taa_resolve.wesl"),
            "embedded://prism_render_scene/shaders/taa_resolve.wesl",
        ),
    );
    (*cache
        .get(0, id, &[])
        .unwrap_or_else(|error| panic!("TAA WESL failed to compile: {error}")))
    .clone()
}

fn adapter_and_device() -> Option<(Instance, wgpu::Adapter, Device, Queue, bool)> {
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::METAL | Backends::VULKAN | Backends::DX12,
        flags: InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        display: None,
        backend_options: BackendOptions::default(),
    });
    let adapter = match block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    })) {
        Ok(adapter) => adapter,
        Err(error) => {
            assert!(!required_gpu(), "PRISM_REQUIRE_GPU: no adapter: {error}");
            eprintln!("SKIP_UNVERIFIED render-frame performance: no adapter: {error}");
            return None;
        }
    };
    let info = adapter.get_info();
    if !matches!(
        info.device_type,
        wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::DiscreteGpu
    ) {
        assert!(
            !required_gpu(),
            "PRISM_REQUIRE_GPU: non-hardware adapter: {info:?}"
        );
        eprintln!("SKIP_UNVERIFIED render-frame performance: adapter={info:?}");
        return None;
    }
    let storage = TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    for (format, usages) in [
        (
            TextureFormat::Rgba16Float,
            storage | TextureUsages::COPY_SRC | TextureUsages::COPY_DST,
        ),
        (
            TextureFormat::Rg16Float,
            TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        ),
        (
            TextureFormat::R8Unorm,
            TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        ),
        (
            TextureFormat::Rgba32Uint,
            TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        ),
    ] {
        if !adapter
            .get_texture_format_features(format)
            .allowed_usages
            .contains(usages)
        {
            assert!(
                !required_gpu(),
                "PRISM_REQUIRE_GPU: {format:?} does not support {usages:?}"
            );
            eprintln!("SKIP_UNVERIFIED render-frame performance: {format:?} lacks {usages:?}");
            return None;
        }
    }
    let mut required_features = Features::IMMEDIATES;
    let timestamps = adapter.features().contains(Features::TIMESTAMP_QUERY);
    if timestamps {
        required_features |= Features::TIMESTAMP_QUERY;
    }
    if adapter.limits().max_immediate_size < 20 {
        assert!(
            !required_gpu(),
            "PRISM_REQUIRE_GPU: immediate block is unavailable"
        );
        eprintln!("SKIP_UNVERIFIED render-frame performance: max_immediate_size < 20");
        return None;
    }
    let limits = wgpu::Limits {
        max_immediate_size: 20,
        ..Default::default()
    };
    let (device, queue) = match block_on(adapter.request_device(&DeviceDescriptor {
        label: Some("render-fabric scene TAA response performance"),
        required_features,
        required_limits: limits,
        ..Default::default()
    })) {
        Ok(result) => result,
        Err(error) => {
            assert!(
                !required_gpu(),
                "PRISM_REQUIRE_GPU: device request failed: {error}"
            );
            eprintln!("SKIP_UNVERIFIED render-frame performance: device request failed: {error}");
            return None;
        }
    };
    eprintln!("render-frame performance adapter: {info:?} timestamps={timestamps}");
    Some((instance, adapter, device, queue, timestamps))
}

fn aligned_row_bytes(row_bytes: u32) -> u32 {
    row_bytes.div_ceil(256) * 256
}

fn logical_resource_bytes(width: u32, height: u32, timestamps: bool) -> u64 {
    let pixels = u64::from(width) * u64::from(height);
    let left_bytes = pixels * size_of::<GpuResponseOutput>() as u64;
    let control_bytes = pixels * size_of::<GpuCompositionControl>() as u64;
    let coverage_bytes = pixels * size_of::<GpuWaterCoverage>() as u64;
    let auxiliary_bytes = pixels * size_of::<GpuAuxiliaryMerge>() as u64;
    let padded_row_bytes = u64::from(aligned_row_bytes(width * 8));
    left_bytes * 3
        + control_bytes
        + coverage_bytes
        + auxiliary_bytes * 3
        + pixels * (8 + 16 + 4 + 1 + 16 + 16)
        + padded_row_bytes * u64::from(height)
        + if timestamps { 32 } else { 0 }
}

fn create_texture(
    device: &Device,
    queue: &Queue,
    label: &str,
    width: u32,
    height: u32,
    format: TextureFormat,
    usage: TextureUsages,
    bytes_per_pixel: u32,
) -> Texture {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: usage | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let data = vec![0_u8; (width * height * bytes_per_pixel) as usize];
    queue.write_texture(
        texture.as_image_copy(),
        &data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * bytes_per_pixel),
            rows_per_image: Some(height),
        },
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    texture
}

fn response_inputs(
    pixels: usize,
) -> (
    Vec<GpuResponseOutput>,
    Vec<GpuResponseOutput>,
    Vec<GpuCompositionControl>,
    Vec<GpuWaterCoverage>,
    Vec<GpuAuxiliaryMerge>,
) {
    let mut left = Vec::with_capacity(pixels);
    let mut right = Vec::with_capacity(pixels);
    let mut controls = Vec::with_capacity(pixels);
    let mut coverage = Vec::with_capacity(pixels);
    let mut auxiliary = Vec::with_capacity(pixels);
    for index in 0..pixels {
        let i = index as f32;
        let mut l = [0.0; 16];
        let mut r = [0.0; 16];
        for channel in 0..15 {
            l[channel] = 0.125 + (channel as f32) * 0.03125 + (i % 17.0) * 0.001;
            r[channel] = 0.25 + (channel as f32) * 0.015625 + (i % 11.0) * 0.002;
        }
        l[15] = 0.75;
        r[15] = 0.625;
        left.push(GpuResponseOutput { words: l });
        right.push(GpuResponseOutput { words: r });
        controls.push(GpuCompositionControl {
            mask: 0.35,
            channels: 31,
            shared_alpha: 0.8,
            alpha_rule: 0,
        });
        coverage.push(GpuWaterCoverage(1));
        auxiliary.push(GpuAuxiliaryMerge {
            left: [i.to_bits(), 0, 0, 0],
            right: [(i + 1.0).to_bits(), 0, 0, 0],
            shared: [0, 0, 0, 0],
            kind: 0,
            rule: 4,
            mask: 0.5,
            reserved: 0,
        });
    }
    (left, right, controls, coverage, auxiliary)
}

fn buffer_init<T: bytemuck::Pod>(
    device: &Device,
    label: &str,
    values: &[T],
    usage: BufferUsages,
) -> Buffer {
    device.create_buffer_init(&BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(values),
        usage,
    })
}

impl Harness {
    fn new(width: u32, height: u32) -> Option<Self> {
        let (instance, adapter, device, queue, timestamps) = adapter_and_device()?;
        let pixels = (width * height) as usize;
        let (left, right, controls, coverage, auxiliary) = response_inputs(pixels);
        let left_bytes = (pixels * size_of::<GpuResponseOutput>()) as u64;
        let control_bytes = (pixels * size_of::<GpuCompositionControl>()) as u64;
        let coverage_bytes = (pixels * size_of::<GpuWaterCoverage>()) as u64;
        let auxiliary_bytes = (pixels * size_of::<GpuAuxiliaryMerge>()) as u64;
        let left_seed = buffer_init(
            &device,
            "render-frame response left seed",
            &left,
            BufferUsages::COPY_SRC,
        );
        let left_buffer = buffer_init(
            &device,
            "render-frame response left",
            &left,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let right_buffer = buffer_init(
            &device,
            "render-frame response right",
            &right,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let controls_buffer = buffer_init(
            &device,
            "render-frame response controls",
            &controls,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let coverage_buffer = buffer_init(
            &device,
            "render-frame response coverage",
            &coverage,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let auxiliary_seed = buffer_init(
            &device,
            "render-frame auxiliary seed",
            &auxiliary,
            BufferUsages::COPY_SRC,
        );
        let auxiliary_buffer = buffer_init(
            &device,
            "render-frame auxiliary",
            &auxiliary,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );

        let hdr = create_texture(
            &device,
            &queue,
            "render-frame scene HDR",
            width,
            height,
            TextureFormat::Rgba16Float,
            TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            8,
        );
        let hdr_view = hdr.create_view(&TextureViewDescriptor::default());
        let history = [
            create_texture(
                &device,
                &queue,
                "render-frame TAA history A",
                width,
                height,
                TextureFormat::Rgba16Float,
                TextureUsages::STORAGE_BINDING
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_SRC,
                8,
            ),
            create_texture(
                &device,
                &queue,
                "render-frame TAA history B",
                width,
                height,
                TextureFormat::Rgba16Float,
                TextureUsages::STORAGE_BINDING
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_SRC,
                8,
            ),
        ];
        let history_views = [
            history[0].create_view(&TextureViewDescriptor::default()),
            history[1].create_view(&TextureViewDescriptor::default()),
        ];
        let motion = create_texture(
            &device,
            &queue,
            "render-frame TAA motion",
            width,
            height,
            TextureFormat::Rg16Float,
            TextureUsages::TEXTURE_BINDING,
            4,
        );
        let motion_view = motion.create_view(&TextureViewDescriptor::default());
        let reactive = create_texture(
            &device,
            &queue,
            "render-frame TAA reactive",
            width,
            height,
            TextureFormat::R8Unorm,
            TextureUsages::TEXTURE_BINDING,
            1,
        );
        let reactive_view = reactive.create_view(&TextureViewDescriptor::default());
        let visibility_ids = create_texture(
            &device,
            &queue,
            "render-frame visibility ids",
            width,
            height,
            TextureFormat::Rgba32Uint,
            TextureUsages::TEXTURE_BINDING,
            16,
        );
        let visibility_ids_view = visibility_ids.create_view(&TextureViewDescriptor::default());
        let visibility_metadata = create_texture(
            &device,
            &queue,
            "render-frame visibility metadata",
            width,
            height,
            TextureFormat::Rgba32Uint,
            TextureUsages::TEXTURE_BINDING,
            16,
        );
        let visibility_metadata_view =
            visibility_metadata.create_view(&TextureViewDescriptor::default());

        let padded_row_bytes = aligned_row_bytes(width * 8);
        let hdr_stage = device.create_buffer(&BufferDescriptor {
            label: Some("render-frame TAA HDR readback"),
            size: u64::from(padded_row_bytes) * u64::from(height),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let auxiliary_stage = device.create_buffer(&BufferDescriptor {
            label: Some("render-frame auxiliary readback"),
            size: auxiliary_bytes,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let timestamp = timestamps.then(|| PassTimer::new(&device));

        let render_device = RenderDevice::from(device.clone());
        let pipelines = ResponseCompositionPipelines::new(&render_device);
        let response_layout = pipelines.radiance.get_bind_group_layout(0);
        let response_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render-frame response radiance bindings"),
            layout: &response_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: left_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: right_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: controls_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: coverage_buffer.as_entire_binding(),
                },
            ],
        });
        let auxiliary_layout = pipelines.auxiliary.get_bind_group_layout(0);
        let auxiliary_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render-frame response auxiliary bindings"),
            layout: &auxiliary_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 4,
                resource: auxiliary_buffer.as_entire_binding(),
            }],
        });

        let response_to_hdr_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("render-frame response to HDR"),
            source: ShaderSource::Wgsl(include_str!("../src/shaders/response_to_hdr.wgsl").into()),
        });
        let response_to_hdr = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("render-frame response to HDR"),
            layout: None,
            module: &response_to_hdr_module,
            entry_point: Some("response_to_hdr"),
            compilation_options: Default::default(),
            cache: None,
        });
        let hdr_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render-frame response to HDR bindings"),
            layout: &response_to_hdr.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: left_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&visibility_ids_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&visibility_metadata_view),
                },
            ],
        });

        let taa_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("render-frame production TAA resolve"),
            source: ShaderSource::Wgsl(compile_taa_wgsl().into()),
        });
        let taa_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("render-frame production TAA resolve"),
            layout: None,
            module: &taa_module,
            entry_point: Some("resolve_taa"),
            compilation_options: Default::default(),
            cache: None,
        });
        let taa_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("render-frame TAA history sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let taa_bind = [0_usize, 1].map(|slot| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("render-frame TAA bindings"),
                layout: &taa_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&hdr_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&history_views[slot]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&taa_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&history_views[1 - slot]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&motion_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&reactive_view),
                    },
                ],
            })
        });

        let tracked_bytes = logical_resource_bytes(width, height, timestamps);
        eprintln!(
            "render-frame performance logical_resources_bytes={} ({:.2} MiB), response_bytes={} hdr_row_bytes={} auxiliary_bytes={}",
            tracked_bytes,
            tracked_bytes as f64 / MIB as f64,
            left_bytes * 2 + control_bytes + coverage_bytes + auxiliary_bytes,
            padded_row_bytes,
            auxiliary_bytes
        );

        let timestamp_period_ns = timestamps.then(|| f64::from(queue.get_timestamp_period()));
        black_box(tracked_bytes);
        Some(Self {
            _instance: instance,
            device,
            queue,
            frame: FrameResources {
                left_seed,
                left: left_buffer,
                right: right_buffer,
                controls: controls_buffer,
                coverage: coverage_buffer,
                auxiliary_seed,
                auxiliary: auxiliary_buffer,
                hdr,
                hdr_view,
                history,
                history_views,
                motion,
                motion_view,
                reactive,
                reactive_view,
                visibility_ids,
                visibility_ids_view,
                visibility_metadata,
                visibility_metadata_view,
                hdr_stage,
                auxiliary_stage,
                timestamp,
                response_bind,
                auxiliary_bind,
                hdr_bind,
                taa_bind,
                taa_pipeline,
                response_to_hdr,
                pipelines,
                logical_resource_bytes: tracked_bytes,
                width,
                height,
                auxiliary_bytes,
                padded_row_bytes,
            },
            timestamp_period_ns,
        })
    }

    fn run_frame(&self, frame_index: usize) -> Sample {
        let start = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("render-fabric scene TAA response performance frame"),
            });
        let timestamp = self.frame.timestamp.as_ref();
        encoder.copy_buffer_to_buffer(
            &self.frame.left_seed,
            0,
            &self.frame.left,
            0,
            self.frame.left.size(),
        );
        encoder.copy_buffer_to_buffer(
            &self.frame.auxiliary_seed,
            0,
            &self.frame.auxiliary,
            0,
            self.frame.auxiliary_bytes,
        );
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("production response composition radiance"),
                timestamp_writes: timestamp.map(|timer| timer.writes_begin()),
            });
            pass.set_pipeline(&self.frame.pipelines.radiance);
            pass.set_bind_group(0, &self.frame.response_bind, &[]);
            pass.dispatch_workgroups((self.frame.left.size() / 64).div_ceil(1) as u32, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("production response composition auxiliary"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.frame.pipelines.auxiliary);
            pass.set_bind_group(0, &self.frame.auxiliary_bind, &[]);
            pass.dispatch_workgroups((self.frame.auxiliary_bytes / 64).div_ceil(64) as u32, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("production response to HDR"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.frame.response_to_hdr);
            pass.set_bind_group(0, &self.frame.hdr_bind, &[]);
            pass.dispatch_workgroups(
                self.frame.width.div_ceil(8),
                self.frame.height.div_ceil(8),
                1,
            );
        }
        let slot = frame_index & 1;
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("production TAA resolve"),
                timestamp_writes: timestamp.map(|timer| timer.writes_end()),
            });
            pass.set_pipeline(&self.frame.taa_pipeline);
            pass.set_bind_group(0, &self.frame.taa_bind[slot], &[]);
            pass.set_immediates(
                0,
                bytemuck::bytes_of(&TaaParams {
                    width: self.frame.width,
                    height: self.frame.height,
                    history_blend: 0.9,
                    variance_gamma: 1.0,
                    valid_history: u32::from(frame_index > 0),
                }),
            );
            pass.dispatch_workgroups(
                self.frame.width.div_ceil(8),
                self.frame.height.div_ceil(8),
                1,
            );
        }
        let output = &self.frame.history[1 - slot];
        encoder.copy_texture_to_buffer(
            output.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &self.frame.hdr_stage,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.frame.padded_row_bytes),
                    rows_per_image: Some(self.frame.height),
                },
            },
            Extent3d {
                width: self.frame.width,
                height: self.frame.height,
                depth_or_array_layers: 1,
            },
        );
        encoder.copy_buffer_to_buffer(
            &self.frame.auxiliary,
            0,
            &self.frame.auxiliary_stage,
            0,
            self.frame.auxiliary_bytes,
        );
        if let Some(timer) = timestamp {
            encoder.resolve_query_set(&timer.query_set, 0..2, &timer.resolve, 0);
            encoder.copy_buffer_to_buffer(&timer.resolve, 0, &timer.read, 0, 16);
        }
        let submit_start = Instant::now();
        self.queue.submit([encoder.finish()]);
        let submit_ns = submit_start.elapsed().as_nanos();
        let completion_start = Instant::now();
        self.device
            .poll(PollType::wait_indefinitely())
            .expect("render-frame GPU completion");
        let completion_ns = completion_start.elapsed().as_nanos();
        let readback_start = Instant::now();
        let hdr_bytes = read_mapped(&self.device, &self.frame.hdr_stage);
        let auxiliary_bytes = read_mapped(&self.device, &self.frame.auxiliary_stage);
        let timestamp_ticks = timestamp.map(|timer| read_mapped(&self.device, &timer.read));
        let readback_ns = readback_start.elapsed().as_nanos();
        let mut checksum = 0_u64;
        for byte in hdr_bytes.iter().chain(auxiliary_bytes.iter()) {
            checksum = checksum
                .wrapping_mul(1_000_003)
                .wrapping_add(u64::from(*byte));
        }
        let gpu_ns = timestamp_ticks.and_then(|bytes| {
            let ticks = bytemuck::try_cast_slice::<u8, u64>(&bytes).ok()?;
            if ticks.len() != 2 {
                return None;
            }
            let delta = ticks[1].saturating_sub(ticks[0]);
            Some((delta as f64 * self.timestamp_period_ns?) as u128)
        });
        Sample {
            submit_ns,
            completion_ns,
            readback_ns,
            total_ns: start.elapsed().as_nanos(),
            gpu_ns,
            checksum,
        }
    }
}

fn read_mapped(device: &Device, buffer: &Buffer) -> Vec<u8> {
    let (sender, receiver) = mpsc::channel();
    buffer.slice(..).map_async(MapMode::Read, move |result| {
        sender.send(result).expect("readback callback receiver");
    });
    device
        .poll(PollType::wait_indefinitely())
        .expect("readback poll");
    receiver
        .recv()
        .expect("readback callback")
        .expect("readback map");
    let view = buffer.slice(..).get_mapped_range().expect("mapped range");
    let bytes = view.to_vec();
    drop(view);
    buffer.unmap();
    bytes
}

fn fabric_setup() -> (Duration, usize, usize) {
    let start = Instant::now();
    let full_selection = CompositionFabric::new();
    black_box(&full_selection);
    let catalog = builtin_catalog().expect("built-in Render Fabric catalog");
    let roots = builtin_roots(&catalog).expect("built-in Render Fabric roots");
    let selection = FeatureSelection::resolve(
        &catalog,
        &roots,
        CapabilitySet::WEBGPU_BASELINE,
        RenderQualityTier::High,
    )
    .expect("base Render Fabric selection");
    let package_count = selection
        .selected()
        .iter()
        .map(|feature| feature.bind_builtin_shader_package())
        .collect::<Result<Vec<_>, _>>()
        .expect("built-in shader packages");
    let pass_count = selection.passes().count();
    black_box(package_count);
    (start.elapsed(), pass_count, catalog.resources().count())
}

#[test]
fn render_fabric_scene_taa_response_gpu_performance() {
    let (fabric_setup_time, base_passes, base_resources) = fabric_setup();
    let width = env_u32("PRISM_RENDER_FRAME_BENCH_WIDTH", DEFAULT_WIDTH).max(8);
    let height = env_u32("PRISM_RENDER_FRAME_BENCH_HEIGHT", DEFAULT_HEIGHT).max(8);
    let warmup = env_usize("PRISM_RENDER_FRAME_BENCH_WARMUP", DEFAULT_WARMUP);
    let samples = env_usize("PRISM_RENDER_FRAME_BENCH_SAMPLES", DEFAULT_SAMPLES).max(3);
    eprintln!(
        "PRISM_RENDER_FRAME_BENCH plan workload=response_composition+response_to_hdr+taa_resolve extent={}x{} pixels={} warmup={} samples={} base_fabric_setup_ms={:.3} base_passes={} base_resources={} logical_resource_plan_mib_without_timestamps={:.3} logical_resource_plan_mib_with_timestamps={:.3}",
        width,
        height,
        width as u64 * height as u64,
        warmup,
        samples,
        fabric_setup_time.as_secs_f64() * 1_000.0,
        base_passes,
        base_resources,
        logical_resource_bytes(width, height, false) as f64 / MIB as f64,
        logical_resource_bytes(width, height, true) as f64 / MIB as f64,
    );
    let Some(harness) = Harness::new(width, height) else {
        eprintln!(
            "PRISM_RENDER_FRAME_BENCH status=blocked_gpu_metrics_unavailable p50_p95_p99=unavailable submit_completion_readback=unavailable gpu_chain=unavailable reason=physical_adapter_unavailable"
        );
        return;
    };
    for frame in 0..warmup {
        black_box(harness.run_frame(frame));
    }
    let timings: Vec<_> = (0..samples)
        .map(|frame| harness.run_frame(warmup + frame))
        .collect();
    let submits: Vec<_> = timings.iter().map(|sample| sample.submit_ns).collect();
    let completions: Vec<_> = timings.iter().map(|sample| sample.completion_ns).collect();
    let readbacks: Vec<_> = timings.iter().map(|sample| sample.readback_ns).collect();
    let totals: Vec<_> = timings.iter().map(|sample| sample.total_ns).collect();
    let gpu: Vec<_> = timings.iter().filter_map(|sample| sample.gpu_ns).collect();
    let checksums: Vec<_> = timings.iter().map(|sample| sample.checksum).collect();
    assert!(checksums.iter().all(|checksum| *checksum != 0));
    assert!(checksums.windows(2).all(|pair| pair[0] == pair[1]));
    eprintln!(
        "PRISM_RENDER_FRAME_BENCH workload=response_composition+response_to_hdr+taa_resolve extent={}x{} pixels={} warmup={} samples={} base_fabric_setup_ms={:.3} base_passes={} base_resources={}",
        width,
        height,
        width as u64 * height as u64,
        warmup,
        samples,
        fabric_setup_time.as_secs_f64() * 1_000.0,
        base_passes,
        base_resources
    );
    eprintln!(
        "PRISM_RENDER_FRAME_BENCH host_submit_ms P50/P95/P99={:.3}/{:.3}/{:.3} host_completion_ms P50/P95/P99={:.3}/{:.3}/{:.3} readback_ms P50/P95/P99={:.3}/{:.3}/{:.3} total_ms P50/P95/P99={:.3}/{:.3}/{:.3}",
        percentile(&submits, 0.50) as f64 / 1_000_000.0,
        percentile(&submits, 0.95) as f64 / 1_000_000.0,
        percentile(&submits, 0.99) as f64 / 1_000_000.0,
        percentile(&completions, 0.50) as f64 / 1_000_000.0,
        percentile(&completions, 0.95) as f64 / 1_000_000.0,
        percentile(&completions, 0.99) as f64 / 1_000_000.0,
        percentile(&readbacks, 0.50) as f64 / 1_000_000.0,
        percentile(&readbacks, 0.95) as f64 / 1_000_000.0,
        percentile(&readbacks, 0.99) as f64 / 1_000_000.0,
        percentile(&totals, 0.50) as f64 / 1_000_000.0,
        percentile(&totals, 0.95) as f64 / 1_000_000.0,
        percentile(&totals, 0.99) as f64 / 1_000_000.0,
    );
    if let Some(median_gpu) = percentile_optional(&gpu) {
        eprintln!(
            "PRISM_RENDER_FRAME_BENCH gpu_chain_ms P50/P95/P99={:.3}/{:.3}/{:.3} timestamp_samples={} timestamp_period_ns={:.3}",
            percentile(&gpu, 0.50) as f64 / 1_000_000.0,
            percentile(&gpu, 0.95) as f64 / 1_000_000.0,
            percentile(&gpu, 0.99) as f64 / 1_000_000.0,
            gpu.len(),
            harness.timestamp_period_ns.unwrap_or_default()
        );
        assert!(median_gpu > 0);
    } else {
        eprintln!(
            "PRISM_RENDER_FRAME_BENCH gpu_chain_ms=unavailable (adapter has no TIMESTAMP_QUERY); host timings remain measured"
        );
    }
    eprintln!(
        "PRISM_RENDER_FRAME_BENCH logical_resource_peak_mib={:.3} readback_bytes={} checksum={:016x} queue_depth=1",
        harness.frame.logical_resource_bytes as f64 / MIB as f64,
        harness.frame.hdr_stage.size() + harness.frame.auxiliary_stage.size(),
        checksums[0]
    );
}
