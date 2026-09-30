//! `wgpu` compute twin of Prism's groom-global analysis reduction
//! ([`reduce_lane`](prism_render_architecture::hair::analysis_readback::reduce_lane)).
//!
//! The hair analysis passes emit a per-element `vec4<f32>` output buffer (arc
//! length / curvature / radius, or a scalar packed into `.x`). A `CPU` decision
//! then needs one groom-global scalar folded from a chosen lane: the `Max` that
//! normalizes the density/decimation `LOD`, or the `Sum` over per-particle
//! motion energy that gates the hysteretic sleep. This crate is the on-device
//! twin of that fold: a single `256`-wide workgroup cooperatively reduces one
//! [`HairMetricLane`] of every element with one [`HairReductionOp`] into a
//! single scalar, so a passing real-device parity test is direct evidence the
//! ported reduction reads the same groom-global value as the reference — not
//! merely that its shader compiles.
//!
//! # Reduction shape
//!
//! Unlike every other kernel in this crate (each an embarrassingly-parallel
//! one-thread-per-output map or gather), a reduction folds many inputs into one
//! output, so it is the crate's first shared-memory tree reduction. Each
//! invocation grid-strides across the element array accumulating a private
//! partial from the operator identity (`0`); the partials are staged into
//! workgroup memory and a logarithmic tree fold collapses them to lane `0`,
//! which writes the single output. There is exactly one output and one
//! workgroup, so there is no cross-workgroup race.
//!
//! # Correctness model
//!
//! The fold mirrors the golden operator branch-for-branch. `Max` selects an
//! existing element value with no arithmetic, so it is bit-exact regardless of
//! fold order (an all-negative lane reduces to the identity `0`, exactly as the
//! golden `if x > acc` update does). `Sum` is floating-point addition, which is
//! commutative but not associative, so the tree's pairwise order differs from
//! the golden's left-to-right walk by a few low-mantissa `ULP`; the parity test
//! asserts a per-value tolerance (`abs_diff < 1e-4` or `rel_diff < 1e-3`) for
//! `Sum` while `Max` is asserted exactly.
//!
//! # Portability
//!
//! The kernel uses only comparisons, addition and workgroup shared memory in
//! the portable core-`WGSL` subset — no `exp`, `pow`, atomics or optional
//! device feature — so the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard single-workgroup shared-memory tree reduction plus
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::analysis_readback::{
    reduce_lane, HairMetricLane, HairReductionOp,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Uniform reduction parameters. `16`-byte scalar-packed `repr(C)` matching
/// `HairReduceParams` in `shaders/analysis_reduce.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    element_count: u32,
    lane: u32,
    op: u32,
    pad0: u32,
}

/// Maps the golden [`HairReductionOp`] to the shader's numeric selector
/// (`0` = `Max`, `1` = `Sum`).
fn op_code(op: HairReductionOp) -> u32 {
    match op {
        HairReductionOp::Max => 0,
        HairReductionOp::Sum => 1,
    }
}

/// A compiled, reusable groom-global analysis reduction pipeline.
pub struct GpuHairAnalysisReduce {
    #[expect(
        dead_code,
        reason = "the shader module must outlive the pipeline that borrows it"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairAnalysisReduce {
    /// Compiles the analysis reduction kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset plus workgroup
    /// shared memory, so no optional device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairAnalysisReduce {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_analysis_reduce"),
            source: ShaderSource::Wgsl(include_str!("../shaders/analysis_reduce.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_analysis_reduce_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_analysis_reduce_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_analysis_reduce_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairAnalysisReduce {
            module,
            layout,
            pipeline,
        }
    }

    /// Folds `lane` of every element in `elements` with `op` into one
    /// groom-global scalar.
    ///
    /// The result equals
    /// [`reduce_lane`](prism_render_architecture::hair::analysis_readback::reduce_lane)
    /// to within the tolerance documented on this module (bit-exact for `Max`).
    /// An empty `elements` slice returns the operator identity
    /// ([`HairReductionOp::identity`], `0`) without a dispatch, mirroring the
    /// empty-groom behavior of the golden and avoiding a zero-sized storage
    /// buffer.
    #[must_use]
    pub fn reduce(
        &self,
        ctx: &GpuContext,
        elements: &[[f32; 4]],
        lane: HairMetricLane,
        op: HairReductionOp,
    ) -> f32 {
        // Empty groom: nothing to fold; return the identity without touching
        // the device (storage buffers cannot be zero-sized).
        if elements.is_empty() {
            return HairReductionOp::identity();
        }

        let uniform = Params {
            element_count: elements.len() as u32,
            lane: lane.index() as u32,
            op: op_code(op),
            pad0: 0,
        };

        let device = ctx.device();
        let out_bytes = size_of::<f32>() as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_analysis_reduce_params"),
            contents: bytemuck::bytes_of(&uniform),
            usage: BufferUsages::UNIFORM,
        });
        let elements_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_analysis_reduce_elements"),
            contents: bytemuck::cast_slice(elements),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_analysis_reduce_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_analysis_reduce_out_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_analysis_reduce_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: elements_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_analysis_reduce_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_analysis_reduce_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One workgroup cooperatively reduces the whole buffer.
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &out_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        out_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        read_f32_scalar(&out_stage)
    }
}

/// Runs the golden reduction directly; a thin re-export so the parity test can
/// name one reference path.
#[must_use]
pub fn reference_reduce(elements: &[[f32; 4]], lane: HairMetricLane, op: HairReductionOp) -> f32 {
    reduce_lane(elements, lane, op)
}

/// Reads the single mapped `f32` back from a staging buffer, then unmaps.
fn read_f32_scalar(stage: &wgpu::Buffer) -> f32 {
    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let value = bytemuck::cast_slice::<u8, f32>(&view)[0];
    drop(view);
    stage.unmap();
    value
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
