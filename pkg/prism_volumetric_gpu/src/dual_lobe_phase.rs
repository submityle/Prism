//! `wgpu` compute twin of the volumetric-cloud non-`Draine` dual-lobe
//! scattering phase
//! ([`dual_lobe_phase`](prism_render_architecture::volumetric::scatter::dual_lobe_phase)).
//!
//! When the sharper `Draine` shaping is disabled, cloud single scattering uses
//! a plain-`Henyey-Greenstein` dual-lobe phase: a forward lobe and a backward
//! lobe blended by a convex weight. The `CPU` golden for that math is
//! [`dual_lobe_phase`](prism_render_architecture::volumetric::scatter::dual_lobe_phase);
//! this crate is the on-device twin that evaluates the same phase, one thread
//! per query, so a passing real-device parity test is direct evidence the
//! ported kernel computes the same phase values as the reference — not merely
//! that its shader compiles.
//!
//! This is distinct from the `Draine`-shaped
//! [`dual_lobe_draine_phase`](prism_render_architecture::volumetric::scatter::dual_lobe_draine_phase)
//! mirrored by [`crate::phase`]: here both lobes are plain `HG`, so the query
//! carries only four scalars.
//!
//! # What the kernel evaluates
//!
//! [`GpuDualLobePhase::eval`] mirrors the reference's composition exactly:
//!
//! * `forward = hg_phase(cos_theta, g_forward)`;
//! * `backward = hg_phase(cos_theta, g_backward)`;
//! * `result = lerp(backward, forward, saturate(blend))`.
//!
//! Each lobe clamps `cos_theta` independently inside `hg_phase`, exactly as the
//! `CPU` golden does. Every clamp, the `4*PI` normalization constant, and the
//! `denom * sqrt(denom)` form of the `^1.5` denominator are reproduced
//! bit-for-bit in the shader source, so the only numeric divergence from the
//! reference is fused-multiply-add contraction.
//!
//! # Portability
//!
//! The kernel uses only `sqrt`, `min`, `max` and multiply/add in the portable
//! core-`WGSL` subset — no `exp`, `pow` or optional device feature — so it runs
//! unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The phase functions here contain no transcendental call (the reference
//! restricts itself to `sqrt` for exactly this reason), so `CPU` and `GPU`
//! evaluate the same closed-form algebra. They are **not** bit-exact, however:
//! a `GPU` may fuse a multiply-add that the scalar `CPU` reference leaves
//! separate, perturbing the low mantissa bits by a few `ULP`. The parity test
//! therefore asserts a tolerance (`abs_diff < 1e-4` or `rel_diff < 1e-3`)
//! rather than exact equality — tight enough that a genuinely wrong port (a
//! swapped lobe, a missing normalization factor, a sign error in the
//! anisotropy) fails, loose enough that legal fma contraction passes.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `Henyey-Greenstein` dual-lobe cloud phase plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

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

/// One non-`Draine` dual-lobe phase query.
///
/// The four scalars are exactly the arguments of the `CPU` golden
/// [`dual_lobe_phase`](prism_render_architecture::volumetric::scatter::dual_lobe_phase),
/// in the same order:
///
/// * `cos_theta` — cosine of the scattering angle (`+1` forward, `-1` back);
/// * `g_forward` — forward-lobe anisotropy (should be positive);
/// * `g_backward` — backward-lobe anisotropy (should be negative);
/// * `blend` — forward/backward mix in `[0, 1]` (`1` selects the forward lobe).
///
/// The struct is `16`-byte `repr(C)` matching `Query` in
/// `shaders/dual_lobe_phase.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct DualLobePhaseQuery {
    /// Cosine of the scattering angle.
    pub cos_theta: f32,
    /// Forward-lobe anisotropy.
    pub g_forward: f32,
    /// Backward-lobe anisotropy.
    pub g_backward: f32,
    /// Forward/backward lobe mix.
    pub blend: f32,
}

/// Uniform parameters for one phase dispatch. Layout matches `Params` in
/// `shaders/dual_lobe_phase.wesl`: the query count then three pad words for the
/// `16`-byte uniform alignment.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// On-device evaluator for the non-`Draine` dual-lobe cloud phase.
///
/// Construct once with [`GpuDualLobePhase::new`], then call
/// [`GpuDualLobePhase::eval`] for each batch of queries.
pub struct GpuDualLobePhase {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuDualLobePhase {
    /// Compiles the kernel and builds the compute pipeline on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDualLobePhase {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_dual_lobe_phase_module"),
            source: ShaderSource::Wgsl(include_str!("../shaders/dual_lobe_phase.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_dual_lobe_phase_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_dual_lobe_phase_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_dual_lobe_phase_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("dual_lobe_phase_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuDualLobePhase {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the non-`Draine` dual-lobe phase for every query in `queries`,
    /// returning one phase value per query in input order.
    ///
    /// The returned value for query `q` equals
    /// [`dual_lobe_phase`](prism_render_architecture::volumetric::scatter::dual_lobe_phase)`(q.cos_theta, q.g_forward, q.g_backward, q.blend)`
    /// to within the fused-multiply-add tolerance documented on this module. An
    /// empty `queries` slice yields an empty result — storage buffers cannot be
    /// zero-sized, so it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[DualLobePhaseQuery]) -> Vec<f32> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let params = Params {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (queries.len() as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_dual_lobe_phase_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_dual_lobe_phase_queries"),
            contents: bytemuck::cast_slice(queries),
            usage: BufferUsages::STORAGE,
        });
        let values_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_dual_lobe_phase_values"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let values_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_dual_lobe_phase_values_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_dual_lobe_phase_bind_group"),
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
                    resource: values_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_dual_lobe_phase_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_dual_lobe_phase_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (queries.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&values_buf, 0, &values_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        values_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = values_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let values = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        values_stage.unmap();
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
