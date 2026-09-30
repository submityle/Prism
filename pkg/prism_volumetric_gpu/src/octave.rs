//! `wgpu` compute twin of the Wrenninge-style octave-scatter decay
//! ([`octave_scatter`](prism_render_architecture::volumetric::scatter::octave_scatter)).
//!
//! Octave scattering is a cheap multiple-scattering approximation (design
//! section 7): each successive scattering octave attenuates the scattering
//! coefficient `sigma_s`, the extinction coefficient `sigma_t` and the phase
//! eccentricity `g` by a fixed geometric factor, so the summed contribution
//! forms a convergent series whose energy only ever decreases. The `CPU` golden
//! [`octave_scatter`](prism_render_architecture::volumetric::scatter::octave_scatter)
//! owns that decay; [`GpuOctaveScatter`] is the on-device twin that runs one
//! thread per query and returns the same `(sigma_s, sigma_t, g)` triple.
//!
//! # Shared schedule
//!
//! Every query in one [`eval`](GpuOctaveScatter::eval) dispatch shares one
//! [`OctaveParams`](prism_render_architecture::volumetric::scatter::OctaveParams)
//! schedule (the common case: a whole march step decays on one preset) and
//! differs only by its base coefficients and octave index.
//!
//! # Portability
//!
//! The kernel is base clamps, factor saturation, an index clamp and a
//! repeated-multiply power in the portable core-`WGSL` subset — no `exp`,
//! `pow` or optional device feature — so it runs unmodified on Metal, Vulkan
//! and DX12.
//!
//! # Correctness model
//!
//! Each output is a base coefficient times a geometric factor raised to the
//! octave index by the same repeated-multiply loop the reference uses, with no
//! summation whose associativity a `GPU` could legally reorder. `CPU` and `GPU`
//! therefore evaluate the identical product sequence in the identical order;
//! the parity test still asserts a tight tolerance
//! (`abs_diff < 1e-6` or `rel_diff < 1e-5`) rather than exact equality to stay
//! robust to any backend rounding of the multiply, which is far tighter than
//! any physically meaningful decay difference.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard Wrenninge-style octave multiple-scattering decay plus
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::volumetric::scatter::OctaveParams;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// One octave-scatter query.
///
/// * `base_sigma_s` — base scattering coefficient (clamped non-negative);
/// * `base_sigma_t` — base extinction coefficient (clamped non-negative);
/// * `base_g` — base phase eccentricity (clamped to the valid anisotropy
///   range);
/// * `octave_index` — octave to evaluate (clamped to the last octave of the
///   shared schedule).
///
/// The struct is `16`-byte `repr(C)` matching `Query` in
/// `shaders/octave.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct OctaveQuery {
    /// Base scattering coefficient.
    pub base_sigma_s: f32,
    /// Base extinction coefficient.
    pub base_sigma_t: f32,
    /// Base phase eccentricity.
    pub base_g: f32,
    /// Octave index to evaluate.
    pub octave_index: u32,
}

/// One octave-scatter result: the decayed `(sigma_s, sigma_t, g)` triple.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OctaveResult {
    /// Decayed scattering coefficient for this octave.
    pub sigma_s: f32,
    /// Decayed extinction coefficient for this octave.
    pub sigma_t: f32,
    /// Decayed phase eccentricity for this octave.
    pub g: f32,
}

/// Uniform parameters for one octave dispatch. Layout matches `Params` in
/// `shaders/octave.wesl`: the three saturated geometric factors, the octave
/// count, the query count and two pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    attenuation: f32,
    contribution: f32,
    eccentricity_attenuation: f32,
    octave_count: u32,
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One octave result as uploaded/read back. `16`-byte stride matching `Octave`
/// in the shader (the triple plus one pad word).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuOctave {
    sigma_s: f32,
    sigma_t: f32,
    g: f32,
    pad: f32,
}

/// A compiled, reusable octave-scatter pipeline.
pub struct GpuOctaveScatter {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuOctaveScatter {
    /// Compiles the octave-scatter kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuOctaveScatter {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_octave"),
            source: ShaderSource::Wgsl(include_str!("../shaders/octave.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_octave_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_octave_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_octave_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("octave"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuOctaveScatter {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the octave-scatter decay for every query in `queries` against
    /// the shared `params`, returning one [`OctaveResult`] per query in input
    /// order.
    ///
    /// The returned result for query `q` equals
    /// [`octave_scatter`](prism_render_architecture::volumetric::scatter::octave_scatter)`(q.base_sigma_s, q.base_sigma_t, q.base_g, q.octave_index, params)`
    /// to within the tolerance documented on this module. An empty `queries`
    /// slice yields an empty result — storage buffers cannot be zero-sized, so
    /// it is handled by an early return.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        params: OctaveParams,
        queries: &[OctaveQuery],
    ) -> Vec<OctaveResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = Params {
            attenuation: params.attenuation,
            contribution: params.contribution,
            eccentricity_attenuation: params.eccentricity_attenuation,
            octave_count: params.octave_count,
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (queries.len() as u64) * (size_of::<GpuOctave>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_octave_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_octave_queries"),
            contents: bytemuck::cast_slice(queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_octave_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_octave_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_octave_bind_group"),
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
            label: Some("prism_volumetric_octave_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_octave_pass"),
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
        let gpu_results = bytemuck::cast_slice::<u8, GpuOctave>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_results.len(), queries.len());
        gpu_results
            .into_iter()
            .map(|o| OctaveResult {
                sigma_s: o.sigma_s,
                sigma_t: o.sigma_t,
                g: o.g,
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
