//! `wgpu` compute twin of the pyrocumulus buoyant-lift ramp
//! ([`pyrocumulus_buoyancy`](prism_render_architecture::volumetric::storm::pyrocumulus_buoyancy)).
//!
//! The cumulonimbus vertical-development model (design section 9b) drives
//! fire-fed pyrocumulus growth from a heat proxy. `pyrocumulus_buoyancy` returns
//! a saturating exponential ramp of buoyant lift for a normalized `heat`:
//! `q = max(heat, 0)` then `saturate(1 - e^{-PYRO_BUOYANCY_GAIN * q})`, so more
//! heat yields more lift with diminishing returns, monotonically increasing and
//! bounded to `0..=1`; negative heat floors at zero. The `CPU` golden
//! [`pyrocumulus_buoyancy`](prism_render_architecture::volumetric::storm::pyrocumulus_buoyancy)
//! owns that math; [`GpuPyrocumulusBuoyancy`] is the on-device twin that runs
//! one thread per query and reproduces the same value.
//!
//! # Correctness model
//!
//! The exponential uses the *same* hand-rolled `exp_approx` the reference uses
//! (base-two range reduction: a fractional seven-term polynomial times an
//! integer power assembled from the float exponent field), *not* the
//! device-native `exp`, and the gain is the same `PYRO_BUOYANCY_GAIN`, so `CPU`
//! and `GPU` evaluate the same closed-form algebra. The only slack is a legal
//! multiply-add contraction of a few `ULP`, so the parity test asserts a tight
//! tolerance (`abs_diff < 1e-6` or `rel_diff < 1e-5`). The scenes also assert
//! the value stays in `0..=1`, rises monotonically with heat and floors negative
//! heat at zero, so a degenerate kernel could not pass.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `floor`, `bitcast`,
//! integer ops, clamp/saturate and multiply/add — so it runs unmodified on
//! Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard fire-fed pyrocumulus buoyancy ramp plus `wgpu` compute
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

/// One pyrocumulus-buoyancy query: the normalized heat proxy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PyrocumulusBuoyancyQuery {
    /// Normalized heat proxy driving buoyant lift; negative values floor at
    /// zero.
    pub heat: f32,
}

/// One query as uploaded. `16`-byte `repr(C)` matching `Query` in
/// `shaders/pyrocumulus_buoyancy.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    heat: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/pyrocumulus_buoyancy.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable pyrocumulus-buoyancy pipeline.
pub struct GpuPyrocumulusBuoyancy {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuPyrocumulusBuoyancy {
    /// Compiles the pyrocumulus-buoyancy kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPyrocumulusBuoyancy {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_pyrocumulus_buoyancy"),
            source: ShaderSource::Wgsl(include_str!("../shaders/pyrocumulus_buoyancy.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_pyrocumulus_buoyancy_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_pyrocumulus_buoyancy_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_pyrocumulus_buoyancy_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("pyrocumulus_buoyancy_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuPyrocumulusBuoyancy {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the pyrocumulus-buoyancy amount for every query in `queries`,
    /// returning one value per query in input order.
    ///
    /// The returned value for query `q` equals
    /// [`pyrocumulus_buoyancy`](prism_render_architecture::volumetric::storm::pyrocumulus_buoyancy)`(q.heat)`
    /// to within the tolerance documented on this module. An empty `queries`
    /// slice yields an empty result — storage buffers cannot be zero-sized, so
    /// it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[PyrocumulusBuoyancyQuery]) -> Vec<f32> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                heat: q.heat,
                pad0: 0.0,
                pad1: 0.0,
                pad2: 0.0,
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
            label: Some("prism_volumetric_pyrocumulus_buoyancy_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_pyrocumulus_buoyancy_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_pyrocumulus_buoyancy_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_pyrocumulus_buoyancy_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_pyrocumulus_buoyancy_bind_group"),
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
            label: Some("prism_volumetric_pyrocumulus_buoyancy_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_pyrocumulus_buoyancy_pass"),
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
