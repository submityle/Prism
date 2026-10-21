//! Optional real-device `wgpu` implementation of the world visibility twin.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

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

pub(crate) const WORKGROUP_SIZE: u32 = 64;

/// Adapter/device acquisition failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuAcquireError {
    /// The adapter request itself returned an error.
    Adapter(String),
    /// An adapter existed, but it was not evidence of physical GPU hardware.
    NonHardware(AdapterEvidence),
    /// An adapter existed but could not create a logical device.
    Device(String),
}

impl std::fmt::Display for GpuAcquireError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Adapter(message) => write!(formatter, "adapter request failed: {message}"),
            Self::NonHardware(evidence) => {
                write!(
                    formatter,
                    "adapter is not physical-GPU evidence: {evidence:?}"
                )
            }
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

impl AdapterEvidence {
    /// Returns whether this evidence is sufficient for a physical-GPU claim.
    ///
    /// CPU, virtual and unknown adapter classes are intentionally rejected;
    /// the world GPU parity paths must not silently turn software execution
    /// into hardware evidence.
    #[must_use]
    pub fn is_hardware(&self) -> bool {
        matches!(self.device_type.as_str(), "IntegratedGpu" | "DiscreteGpu")
    }
}

#[cfg(test)]
mod adapter_evidence_tests {
    use super::*;

    fn evidence(device_type: &str) -> AdapterEvidence {
        AdapterEvidence {
            backend: "Metal".into(),
            device_type: device_type.into(),
            vendor: 1,
            device: "test adapter".into(),
            driver: "test driver".into(),
            driver_info: "test driver info".into(),
            required_features: "Features(0x0)".into(),
        }
    }

    #[test]
    fn only_integrated_and_discrete_adapters_are_hardware_evidence() {
        assert!(evidence("IntegratedGpu").is_hardware());
        assert!(evidence("DiscreteGpu").is_hardware());
        assert!(!evidence("Cpu").is_hardware());
        assert!(!evidence("VirtualGpu").is_hardware());
        assert!(!evidence("Other").is_hardware());
    }
}

/// Resource and host wall-clock measurements for one real-device evaluation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuTelemetry {
    /// Number of proxy records supplied to the dispatch.
    pub input_proxy_count: usize,
    /// Bytes uploaded for the camera parameters and proxy SoA columns.
    pub upload_bytes: u64,
    /// Bytes copied from the GPU output into the host-visible staging buffer.
    pub readback_bytes: u64,
    /// Number of one-dimensional workgroups submitted to the compute pass.
    pub dispatch_workgroups: u32,
    /// Adapter metadata captured for the dispatch.
    pub adapter_evidence: AdapterEvidence,
    /// Host wall-clock time from queue submission through GPU completion.
    pub dispatch_wall_time: Duration,
    /// Host wall-clock time for mapping and copying the readback records.
    pub readback_wall_time: Duration,
}

/// GPU outputs paired with the telemetry for the same contract evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuEvaluation {
    /// Canonical visibility records returned by the GPU.
    pub outputs: Vec<VisibilityOutput>,
    /// Resource and timing data for this evaluation.
    pub telemetry: GpuTelemetry,
}

/// Native `wgpu` instance, adapter, device, and queue kept alive together.
/// Clones share the same logical device identity; independent acquisitions do
/// not, even when backend-local resource IDs happen to compare equal.
#[derive(Clone)]
pub struct GpuContext {
    instance: Instance,
    adapter: Adapter,
    device: Device,
    queue: Queue,
    device_identity: Arc<()>,
}

impl GpuContext {
    /// Attempts to acquire a native headless adapter.
    ///
    /// An adapter/device acquisition error is returned to the caller so a
    /// strict real-device test can fail with the acquisition evidence.
    pub fn try_headless() -> Result<Option<Self>, GpuAcquireError> {
        Self::try_headless_with_limits(Limits::downlevel_defaults())
    }

    /// Acquires a device with explicit required limits and no optional features.
    /// Unsupported limits fail device acquisition; they are never silently
    /// reduced. The author-field scatter pass requires five storage bindings,
    /// whereas [`Self::try_headless`] retains four-binding downlevel defaults.
    pub fn try_headless_with_limits(
        required_limits: Limits,
    ) -> Result<Option<Self>, GpuAcquireError> {
        let instance = Instance::new(
            InstanceDescriptor {
                backends: Backends::METAL | Backends::VULKAN | Backends::DX12,
                flags: InstanceFlags::default(),
                memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
                display: None,
                backend_options: BackendOptions::default(),
            }
            .with_env(),
        );
        let adapter = futures_lite::future::block_on(
            instance.request_adapter(&RequestAdapterOptions::default()),
        )
        .map_err(|error| GpuAcquireError::Adapter(error.to_string()))?;
        let adapter_info = adapter.get_info();
        let evidence = AdapterEvidence {
            backend: format!("{:?}", adapter_info.backend),
            device_type: format!("{:?}", adapter_info.device_type),
            vendor: adapter_info.vendor,
            device: adapter_info.name.clone(),
            driver: adapter_info.driver.clone(),
            driver_info: adapter_info.driver_info.clone(),
            required_features: format!("{:?}", Features::empty()),
        };
        if !evidence.is_hardware() {
            return Err(GpuAcquireError::NonHardware(evidence));
        }
        let descriptor = DeviceDescriptor {
            label: Some("prism_world_gpu_device"),
            required_features: Features::empty(),
            required_limits,
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
            device_identity: Arc::new(()),
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

    // Private, unforgeable identity shared only by clones and resources built
    // from this context. Do not substitute wgpu handle Eq across Instances.
    pub(crate) fn device_identity(&self) -> &Arc<()> {
        &self.device_identity
    }

    #[cfg(feature = "placement_rhi")]
    pub(crate) fn adopt_placement_rhi(
        &self,
    ) -> (
        prism_render_driver_wgpu::WgpuDevice,
        prism_render_driver_wgpu::WgpuQueue,
    ) {
        prism_render_driver_wgpu::WgpuDevice::adopt(
            self.adapter.clone(),
            self.device.clone(),
            self.queue.clone(),
        )
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
pub(crate) struct GpuParams {
    camera: [f32; 4],
    projection_scale: f32,
    max_distance: f32,
    proxy_count: u32,
    reserved: u32,
    frustum: [[f32; 4]; 6],
}

impl GpuParams {
    pub(crate) fn new(input: VisibilityInput, proxy_count: u32) -> Self {
        Self {
            camera: [
                input.camera_position[0],
                input.camera_position[1],
                input.camera_position[2],
                0.0,
            ],
            projection_scale: input.projection_scale,
            max_distance: input.max_distance,
            proxy_count,
            reserved: 0,
            frustum: input.frustum.map(|plane| {
                [
                    plane.normal[0],
                    plane.normal[1],
                    plane.normal[2],
                    plane.distance,
                ]
            }),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuRecord {
    visible: u32,
    lod: u32,
    error_quantized: u32,
    proxy_id: u32,
    screen_error: f32,
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

    /// Reads back canonical visibility words and the unquantized screen error.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        input: VisibilityInput,
        proxies: &[ProxyInstance],
    ) -> Vec<VisibilityOutput> {
        self.evaluate_with_telemetry(ctx, input, proxies).outputs
    }

    /// Executes visibility/LOD and returns canonical records with resource and
    /// host wall-clock telemetry for the same dispatch.
    #[must_use]
    pub fn evaluate_with_telemetry(
        &self,
        ctx: &GpuContext,
        input: VisibilityInput,
        proxies: &[ProxyInstance],
    ) -> GpuEvaluation {
        let input_proxy_count = proxies.len();
        let adapter_evidence = ctx.evidence();
        if proxies.is_empty() {
            return GpuEvaluation {
                outputs: Vec::new(),
                telemetry: GpuTelemetry {
                    input_proxy_count,
                    upload_bytes: 0,
                    readback_bytes: 0,
                    dispatch_workgroups: 0,
                    adapter_evidence,
                    dispatch_wall_time: Duration::ZERO,
                    readback_wall_time: Duration::ZERO,
                },
            };
        }
        let soa = ProxySoa::from_proxies(proxies).expect("validated proxy data is uploadable");
        let device = ctx.device();
        let upload_bytes = (std::mem::size_of_val(soa.positions.as_slice())
            + std::mem::size_of_val(soa.errors.as_slice())
            + std::mem::size_of_val(soa.metadata.as_slice())
            + std::mem::size_of::<GpuParams>()) as u64;
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
        let params = GpuParams::new(input, proxies.len() as u32);
        let params_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_world_visibility_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let output_size = (proxies.len() * std::mem::size_of::<GpuRecord>()) as u64;
        let dispatch_workgroups = (proxies.len() as u32).div_ceil(WORKGROUP_SIZE);
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
        let bind_group = self.bind_group(
            device,
            &params_buffer,
            &positions,
            &errors,
            &metadata,
            &output,
        );
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_world_visibility_encoder"),
        });
        self.encode(&mut encoder, &bind_group, proxies.len() as u32);
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, output_size);
        let dispatch_started = Instant::now();
        ctx.queue().submit([encoder.finish()]);
        ctx.wait();
        let dispatch_wall_time = dispatch_started.elapsed();

        let readback_started = Instant::now();
        let readback = staging.slice(..);
        readback.map_async(MapMode::Read, |_| {});
        ctx.wait();
        let mapped = readback
            .get_mapped_range()
            .expect("world visibility readback must be mapped after device wait");
        let records: Vec<GpuRecord> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        let readback_wall_time = readback_started.elapsed();
        let outputs = records
            .into_iter()
            .map(|record| VisibilityOutput {
                proxy_id: record.proxy_id,
                visible: record.visible != 0,
                lod: record.lod,
                screen_error: record.screen_error,
                error_quantized: record.error_quantized,
            })
            .collect();

        GpuEvaluation {
            outputs,
            telemetry: GpuTelemetry {
                input_proxy_count,
                upload_bytes,
                readback_bytes: output_size,
                dispatch_workgroups,
                adapter_evidence,
                dispatch_wall_time,
                readback_wall_time,
            },
        }
    }

    pub(crate) fn bind_group(
        &self,
        device: &Device,
        params: &Buffer,
        positions: &Buffer,
        errors: &Buffer,
        metadata: &Buffer,
        output: &Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_world_visibility_bind_group"),
            layout: &self.layout,
            entries: &[
                buffer_binding(0, params),
                buffer_binding(1, positions),
                buffer_binding(2, errors),
                buffer_binding(3, metadata),
                buffer_binding(4, output),
            ],
        })
    }

    pub(crate) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        count: u32,
    ) {
        if count != 0 {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_world_visibility_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(count.div_ceil(WORKGROUP_SIZE), 1, 1);
        }
    }

    /// Ensures the shader module remains owned alongside the pipeline.
    #[must_use]
    pub fn shader_module(&self) -> &wgpu::ShaderModule {
        &self.shader
    }
}

pub(crate) fn uniform_entry(binding: u32) -> BindGroupLayoutEntry {
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

pub(crate) fn storage_entry(binding: u32, read_only: bool) -> BindGroupLayoutEntry {
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

pub(crate) fn buffer_binding(binding: u32, buffer: &Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: BindingResource::Buffer(buffer.as_entire_buffer_binding()),
    }
}
