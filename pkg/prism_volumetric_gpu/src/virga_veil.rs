//! `wgpu` compute twin of the storm virga-veil seam
//! ([`StormState::virga_veil`](prism_render_architecture::volumetric::storm::StormState::virga_veil)).
//!
//! The cumulonimbus precipitation model (design section 9b) trails a `virga`
//! veil below the cloud base whose strength grows as the storm matures.
//! [`StormState::virga_veil`] scales the authored `virga_fade` falloff by the
//! state machine's current `virga` veil strength at a normalized
//! `veil_fraction`:
//!
//! ```text
//! result = saturate(virga * virga_fade(veil_fraction))
//! virga_fade(f) = smoothstep(0, 1, saturate(f))
//! ```
//!
//! so a young storm (no veil) yields zero and a mature storm trails a fading
//! precipitation curtain, densest at the cloud base (`veil_fraction == 1`) and
//! evaporating toward the trailing tip (`veil_fraction == 0`). The result is
//! bounded to `0..=1` and monotonically non-decreasing in both the veil
//! strength and the height fraction. The `CPU` golden
//! [`StormState::virga_veil`](prism_render_architecture::volumetric::storm::StormState::virga_veil)
//! owns that math; [`GpuVirgaVeil`] is the on-device twin that runs one thread
//! per query and reproduces the same value.
//!
//! # Correctness model
//!
//! The `smoothstep` is expanded to the *same* closed form the `CPU`
//! `math::smoothstep` uses (a degenerate-span guard, a saturated ramp, then the
//! cubic `t*t*(3-2t)`), and the kernel contains no transcendental call — only
//! `saturate` and multiply/add — so `CPU` and `GPU` evaluate the same algebra.
//! Values are asserted to within `abs_diff < 1e-6` or `rel_diff < 1e-5`. The
//! scenes also assert the veil stays in `0..=1` and is monotonically
//! non-decreasing in both the veil strength and the veil fraction, so a
//! degenerate kernel could not pass.
//!
//! # Portability
//!
//! The kernel is `clamp`/`saturate` and multiply/add in the portable core-`WGSL`
//! subset — no `exp`, `pow` or optional device feature — so it runs unmodified
//! on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard virga precipitation-veil falloff plus `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// One virga-veil query: the storm's current `virga` veil strength plus the
/// normalized `veil_fraction`, each a normalized `0..=1` signal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VirgaVeilQuery {
    /// `virga` veil strength `0..=1` (the maturing storm's precipitation gate).
    pub virga: f32,
    /// Normalized veil height (`1` at the cloud base, `0` at the trailing tip).
    pub veil_fraction: f32,
}

/// One query as uploaded. `16`-byte `repr(C)` matching `Query` in
/// `shaders/virga_veil.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    virga: f32,
    veil_fraction: f32,
    pad0: f32,
    pad1: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/virga_veil.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable virga-veil pipeline.
pub struct GpuVirgaVeil {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuVirgaVeil {
    /// Compiles the virga-veil shader on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuVirgaVeil {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_virga_veil"),
            source: ShaderSource::Wgsl(include_str!("../shaders/virga_veil.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_virga_veil_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_virga_veil_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_virga_veil_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("virga_veil_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuVirgaVeil {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the virga veil for every query in `queries`, returning one
    /// value per query in input order.
    ///
    /// The returned value for query `q` equals
    /// [`StormState::virga_veil`](prism_render_architecture::volumetric::storm::StormState::virga_veil)
    /// evaluated on a `StormState` whose `virga` field is `q.virga` (all other
    /// fields left at their default), called with `q.veil_fraction`, to within
    /// the tolerance documented on this module. An empty `queries` slice yields
    /// an empty result — storage buffers cannot be zero-sized, so it is handled
    /// by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[VirgaVeilQuery]) -> Vec<f32> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                virga: q.virga,
                veil_fraction: q.veil_fraction,
                pad0: 0.0,
                pad1: 0.0,
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
            label: Some("prism_volumetric_virga_veil_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_virga_veil_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_virga_veil_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_virga_veil_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_virga_veil_bind_group"),
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
            label: Some("prism_volumetric_virga_veil_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_virga_veil_pass"),
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
        let values = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(values.len(), queries.len());
        values
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
