//! `wgpu` compute twin of the precipitation-classification kernel
//! ([`classify_precip`](prism_render_architecture::volumetric::weather::classify_precip)).
//!
//! The weather map's precip (`B`) channel, the cloud kind and the air
//! temperature together decide whether a cell rains, snows or stays dry, and how
//! hard (design section 9). The classification is a pure threshold cascade: a
//! cell below [`PRECIP_TRIGGER`](prism_render_architecture::volumetric::weather::PRECIP_TRIGGER)
//! is dry; otherwise the phase is `Snow` at or below
//! [`FREEZING_POINT_C`](prism_render_architecture::volumetric::weather::FREEZING_POINT_C)
//! and `Rain` above it, and the intensity is the precip channel scaled by the
//! kind gain (the deep-convective `Cumulonimbus` is full strength, others
//! drizzle) and modulated by coverage, saturated to `0..=1`.
//!
//! The `CPU` golden
//! [`classify_precip`](prism_render_architecture::volumetric::weather::classify_precip)
//! owns that logic; [`GpuClassifyPrecip`] is the on-device twin that runs one
//! thread per query.
//!
//! # Correctness model
//!
//! The classification is comparisons and one saturated product, so the phase
//! decision is exact and the intensity matches the `CPU` golden to the last few
//! ULPs. The parity test asserts the phase bit for bit and the intensity within
//! a tight tolerance, including the dry/wet trigger and the freezing edge (which
//! is `Snow`, deterministically).
//!
//! # Portability
//!
//! The kernel is comparisons and `clamp` in the portable core-`WGSL` subset —
//! no `exp`, `pow` or optional device feature — so it runs unmodified on Metal,
//! Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard threshold-cascade precipitation classification plus
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.
use bytemuck::{Pod, Zeroable};
use prism_render_architecture::volumetric::weather::{Precip, PrecipKind};
use prism_render_architecture::volumetric::CloudKind;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// One classification query: the weather-map precip and coverage channels, the
/// air temperature (Celsius) and the cloud kind.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClassifyPrecipQuery {
    /// The `precip` (`B`) channel intensity `0..=1`.
    pub precipitation: f32,
    /// The `coverage` (`R`) channel `0..=1`.
    pub coverage: f32,
    /// The air temperature in degrees Celsius.
    pub temperature: f32,
    /// The cloud kind (only `Cumulonimbus` is full-strength precipitation).
    pub kind: CloudKind,
}

/// One query as uploaded. `16`-byte `repr(C)` matching `Query` in
/// `shaders/classify_precip.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    precipitation: f32,
    coverage: f32,
    temperature: f32,
    kind: u32,
}

/// One result as read back. `8`-byte `repr(C)` matching `Res` in
/// `shaders/classify_precip.wesl`: the phase ordinal plus the intensity.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    kind: u32,
    intensity: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/classify_precip.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// Maps a [`CloudKind`] to the ordinal the kernel expects.
fn kind_ordinal(kind: CloudKind) -> u32 {
    match kind {
        CloudKind::Cumulus => 0,
        CloudKind::Stratus => 1,
        CloudKind::Cirrus => 2,
        CloudKind::Cumulonimbus => 3,
    }
}

/// Maps a phase ordinal back to a [`PrecipKind`].
fn precip_kind_from_ordinal(ordinal: u32) -> PrecipKind {
    match ordinal {
        1 => PrecipKind::Rain,
        2 => PrecipKind::Snow,
        _ => PrecipKind::None,
    }
}

/// A compiled, reusable precipitation-classification pipeline.
pub struct GpuClassifyPrecip {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuClassifyPrecip {
    /// Compiles the precipitation-classification kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClassifyPrecip {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_classify_precip"),
            source: ShaderSource::Wgsl(include_str!("../shaders/classify_precip.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_classify_precip_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_classify_precip_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_classify_precip_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("classify_precip_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuClassifyPrecip {
            module,
            layout,
            pipeline,
        }
    }

    /// Classifies every query in `queries`, returning one [`Precip`] per query
    /// in input order.
    ///
    /// The returned value for query `q` equals
    /// [`classify_precip`](prism_render_architecture::volumetric::weather::classify_precip)
    /// applied to the same inputs: the phase matches exactly and the intensity
    /// within a tight floating-point tolerance. An empty `queries` slice yields
    /// an empty result — storage buffers cannot be zero-sized, so it is handled
    /// by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[ClassifyPrecipQuery]) -> Vec<Precip> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                precipitation: q.precipitation,
                coverage: q.coverage,
                temperature: q.temperature,
                kind: kind_ordinal(q.kind),
            })
            .collect();

        let gpu_params = Params {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (queries.len() as u64) * (size_of::<GpuResult>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_classify_precip_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_classify_precip_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_classify_precip_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_classify_precip_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_classify_precip_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_classify_precip_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_classify_precip_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (queries.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), queries.len());
        raw.into_iter()
            .map(|r| Precip {
                kind: precip_kind_from_ordinal(r.kind),
                intensity: r.intensity,
            })
            .collect()
    }
}

/// Builds a compute-visible buffer binding layout entry.
fn buffer_entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
