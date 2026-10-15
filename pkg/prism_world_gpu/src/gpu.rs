//! Optional real-device `wgpu` implementation of the world visibility twin.

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    Adapter, AdapterInfo, BackendOptions, Backends, BindGroupDescriptor, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, Buffer,
    BufferBindingType, BufferDescriptor, BufferUsages, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device, DeviceDescriptor,
    Features, Instance, InstanceDescriptor, InstanceFlags, Limits, MapMode,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PollType, Queue, RequestAdapterOptions,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::{ProxyInstance, ProxySoa, VisibilityInput, VisibilityOutput};

const WORKGROUP_SIZE: u32 = 64;

/// Adapter/device acquisition failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuAcquireError {
    /// The adapter request itself returned an error.
    Adapter(String),
    /// An adapter existed but could not create a logical device.
    Device(String),
}

impl std::fmt::Display for GpuAcquireError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Adapter(message) => write!(formatter, "adapter request failed: {message}"),
            Self::Device(message) => write!(formatter, "device request failed: {message}"),
        }
    }
}

impl std::error::Error for GpuAcquireError {}

/// Evidence captured from the physical adapter used by a parity run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterEvidence {
    /// Backend selected by wgpu.
    pub backend: String,
    /// Adapter class reported by wgpu.
    pub device_type: String,
    /// Vendor identifier.
    pub vendor: u32,
    /// Human-readable device name.
    pub device: String,
    /// Driver name/version string.
    pub driver: String,
    /// Backend-specific driver details.
    pub driver_info: String,
    /// Required features for this contract; empty means no optional feature is hidden.
    pub required_features: String,
}

/// Native `wgpu` instance, adapter, device, and queue kept alive together.
pub struct GpuContext {
    instance: Instance,
    adapter: Adapter,
    device: Device,
    queue: Queue,
}

impl GpuContext {
    /// Attempts to acquire a native headless adapter.
    ///
    /// An adapter/device acquisition error is the normal result on a GPU-less
    /// host. The caller decides whether that should skip or fail by checking
    /// `PRISM_REQUIRE_GPU`.
    pub fn try_headless() -> Result<Option<Self>, GpuAcquireError> {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::METAL | Backends::VULKAN | Backends::DX12,
            flags: InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
            backend_options: BackendOptions::default(),
        });
        let adapter = futures_lite::future::block_on(
            instance.request_adapter(&RequestAdapterOptions::default()),
        )
        .map_err(|error| GpuAcquireError::Adapter(error.to_string()))?;
        let descriptor = DeviceDescriptor {
            label: Some("prism_world_gpu_device"),
            required_features: Features::empty(),
            required_limits: Limits::downlevel_defaults(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        };
        let (device, queue) = futures_lite::future::block_on(adapter.request_device(&descriptor))
            .map_err(|error| GpuAcquireError::Device(error.to_string()))?;
        Ok(Some(Self {
            instance,
            adapter,
            device,
            queue,
        }))
    }

    /// Returns adapter metadata suitable for a test log or benchmark record.
    #[must_use]
    pub fn evidence(&self) -> AdapterEvidence {
        let info: AdapterInfo = self.adapter.get_info();
        AdapterEvidence {
            backend: format!("{:?}", info.backend),
            device_type: format!("{:?}", info.device_type),
            vendor: info.vendor,
            device: info.name,
            driver: info.driver,
            driver_info: info.driver_info,
            required_features: format!("{:?}", Features::empty()),
        }
    }

    /// Returns the native device.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns the native submission queue.
    #[must_use]
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Waits for all submitted work and surfaces device loss as a panic.
    pub fn wait(&self) {
        self.device
            .poll(PollType::wait_indefinitely())
            .expect("world GPU device poll must complete");
    }

    /// Keeps the `wgpu::Instance` visible to the type so its lifetime is part
    /// of the context contract.
    #[must_use]
    pub fn instance(&self) -> &Instance {
        &self.instance
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    camera: [f32; 4],
    projection_scale: f32,
    max_distance: f32,
    proxy_count: u32,
    reserved: u32,
    frustum: [[f32; 4]; 6],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuRecord {
    visible: u32,
    lod: u32,
    error_quantized: u32,
    proxy_id: u32,
}

/// Reusable compute pipeline for deterministic proxy visibility and LOD.
pub struct GpuVisibility {
    shader: wgpu::ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuVisibility {
    /// Compiles the checked-in WGSL shader and creates its compute pipeline.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_world_proxy_visibility_shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/proxy_visibility.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_world_proxy_visibility_layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_world_proxy_visibility_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_world_proxy_visibility_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            shader,
            layout,
            pipeline,
        }
    }

    /// Executes visibility/LOD and reads back the canonical four-word records.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        input: VisibilityInput,
        proxies: &[ProxyInstance],
    ) -> Vec<VisibilityOutput> {
        if proxies.is_empty() {
            return Vec::new();
        }
        let soa = ProxySoa::from_proxies(proxies).expect("validated proxy data is uploadable");
        let device = ctx.device();
        let positions = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_world_proxy_positions"),
            contents: bytemuck::cast_slice(&soa.positions),
            usage: BufferUsages::STORAGE,
        });
        let errors = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_world_proxy_errors"),
            contents: bytemuck::cast_slice(&soa.errors),
            usage: BufferUsages::STORAGE,
        });
        let metadata = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_world_proxy_metadata"),
            contents: bytemuck::cast_slice(&soa.metadata),
            usage: BufferUsages::STORAGE,
        });
        let params = GpuParams {
            camera: [
                input.camera_position[0],
                input.camera_position[1],
                input.camera_position[2],
                0.0,
            ],
            projection_scale: input.projection_scale,
            max_distance: input.max_distance,
            proxy_count: proxies.len() as u32,
            reserved: 0,
            frustum: input.frustum.map(|plane| {
                [
                    plane.normal[0],
                    plane.normal[1],
                    plane.normal[2],
                    plane.distance,
                ]
            }),
        };
        let params_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_world_visibility_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let output_size = (proxies.len() * std::mem::size_of::<GpuRecord>()) as u64;
        let output = device.create_buffer(&BufferDescriptor {
            label: Some("prism_world_visibility_output"),
            size: output_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&BufferDescriptor {
            label: Some("prism_world_visibility_readback"),
            size: output_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_world_visibility_bind_group"),
            layout: &self.layout,
            entries: &[
                buffer_binding(0, &params_buffer),
                buffer_binding(1, &positions),
                buffer_binding(2, &errors),
                buffer_binding(3, &metadata),
                buffer_binding(4, &output),
            ],
        });
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_world_visibility_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_world_visibility_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((proxies.len() as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, output_size);
        ctx.queue().submit([encoder.finish()]);
        staging.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let mapped = staging
            .slice(..)
            .get_mapped_range()
            .expect("world visibility readback must be mapped after device wait");
        let records: Vec<GpuRecord> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        records
            .into_iter()
            .map(|record| VisibilityOutput {
                proxy_id: record.proxy_id,
                visible: record.visible != 0,
                lod: record.lod,
                screen_error: record.error_quantized as f32 / crate::ERROR_QUANTUM,
                error_quantized: record.error_quantized,
            })
            .collect()
    }

    /// Ensures the shader module remains owned alongside the pipeline.
    #[must_use]
    pub fn shader_module(&self) -> &wgpu::ShaderModule {
        &self.shader
    }
}

fn uniform_entry(binding: u32) -> BindGroupLayoutEntry {
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

fn storage_entry(binding: u32, read_only: bool) -> BindGroupLayoutEntry {
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

fn buffer_binding(binding: u32, buffer: &Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: BindingResource::Buffer(buffer.as_entire_buffer_binding()),
    }
}
