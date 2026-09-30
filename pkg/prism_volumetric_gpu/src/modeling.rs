//! `wgpu` compute twin of the procedural cloud-modelling composition
//! ([`compose_from_modeling`](prism_render_architecture::volumetric::modeling::compose_from_modeling)).
//!
//! The modelling stage owns the *shape* half of the density field (design
//! section 4, `Nubis`-style pipeline): it folds the cloud-type blend of the
//! stratus/cumulus base shapes, the weather-driven coverage `remap`, the
//! per-[`CloudKind`] vertical `height` gradient and the energy-preserving
//! `detail erosion` `remap` into one bounded density in `0..=1`. The `CPU`
//! golden
//! [`compose_from_modeling`](prism_render_architecture::volumetric::modeling::compose_from_modeling)
//! owns that math; [`GpuModeling`] is the on-device twin that runs one thread
//! per query and returns the same density.
//!
//! # Portability
//!
//! The composition is entirely `saturate` / `lerp` / `remap` / Hermite
//! `smoothstep` — no `exp`, `pow` or optional device feature — so the kernel
//! runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! Every stage is a `remap`, a Hermite polynomial or a product of unit-range
//! factors, all transcendental-free, so `CPU` and `GPU` evaluate the identical
//! arithmetic in the identical order. The only legal divergence is a backend
//! contracting a multiply-add, so the parity test asserts a tight tolerance
//! (`abs_diff < 1e-6` or `rel_diff < 1e-5`) rather than exact equality, far
//! tighter than any physically meaningful density difference. The test also
//! asserts the range and monotonicity contracts (`0..=1`, coverage grows the
//! cloud, erosion never brightens) so a degenerate kernel cannot pass.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `Nubis`-style procedural cloud modelling plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::volumetric::{CloudKind, CloudModeling};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// One cloud-modelling query: an authored [`CloudModeling`] contract, the
/// layer's [`CloudKind`] and the raw noise/geometry samples the composition
/// consumes.
///
/// This mirrors the argument list of
/// [`compose_from_modeling`](prism_render_architecture::volumetric::modeling::compose_from_modeling)
/// so a caller can twin any density evaluation without re-deriving inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelingQuery {
    /// Authored modelling contract (cloud type, coverage, erosion strength).
    pub modeling: CloudModeling,
    /// The layer's cloud kind, selecting the vertical `height` profile.
    pub kind: CloudKind,
    /// Low-frequency (stratus) base shape in `0..=1`.
    pub base_low: f32,
    /// High-frequency (cumulus) base shape in `0..=1`.
    pub base_high: f32,
    /// Normalised `height` fraction inside the layer's altitude band `0..=1`.
    pub height_fraction: f32,
    /// Sampled weather-map coverage channel in `0..=1`.
    pub weather_coverage: f32,
    /// High-frequency `Worley`/`fBm` detail value in `0..=1`.
    pub detail_noise: f32,
}

/// Maps a [`CloudKind`] to the shader's kind index, matching the `CPU` enum
/// declaration order (`0` Cumulus, `1` Stratus, `2` Cirrus, `3`
/// Cumulonimbus) and the `height_profile` selection in
/// `shaders/modeling.wesl`.
fn kind_index(kind: CloudKind) -> u32 {
    match kind {
        CloudKind::Cumulus => 0,
        CloudKind::Stratus => 1,
        CloudKind::Cirrus => 2,
        CloudKind::Cumulonimbus => 3,
    }
}

/// One modelling query as uploaded. `40`-byte `repr(C)` matching `Query` in
/// `shaders/modeling.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    cloud_type: f32,
    coverage: f32,
    detail_erosion: f32,
    base_low: f32,
    base_high: f32,
    height_fraction: f32,
    weather_coverage: f32,
    detail_noise: f32,
    kind: u32,
    pad: u32,
}

/// Uniform parameters for one modelling dispatch. Layout matches `Params` in
/// `shaders/modeling.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable cloud-modelling pipeline.
pub struct GpuModeling {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuModeling {
    /// Compiles the cloud-modelling kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuModeling {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_modeling"),
            source: ShaderSource::Wgsl(include_str!("../shaders/modeling.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_modeling_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_modeling_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_modeling_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("compose"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuModeling {
            module,
            layout,
            pipeline,
        }
    }

    /// Composes the final cloud density for every query in `queries`, returning
    /// one density per query in input order.
    ///
    /// The returned density for query `q` equals
    /// [`compose_from_modeling`](prism_render_architecture::volumetric::modeling::compose_from_modeling)`(q.modeling, q.kind, q.base_low, q.base_high, q.height_fraction, q.weather_coverage, q.detail_noise)`
    /// to within the tolerance documented on this module. An empty `queries`
    /// slice yields an empty result — storage buffers cannot be zero-sized, so
    /// it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[ModelingQuery]) -> Vec<f32> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                cloud_type: q.modeling.cloud_type,
                coverage: q.modeling.coverage,
                detail_erosion: q.modeling.detail_erosion,
                base_low: q.base_low,
                base_high: q.base_high,
                height_fraction: q.height_fraction,
                weather_coverage: q.weather_coverage,
                detail_noise: q.detail_noise,
                kind: kind_index(q.kind),
                pad: 0,
            })
            .collect();

        let gpu_params = Params {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (queries.len() as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_modeling_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_modeling_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_modeling_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_modeling_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_modeling_bind_group"),
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
            label: Some("prism_volumetric_modeling_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_modeling_pass"),
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
        let densities = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(densities.len(), queries.len());
        densities
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
