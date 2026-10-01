//! `wgpu` compute twin of Prism's stratified lobe allocation
//! ([`stratified_allocation`](prism_render_architecture::hair::scatter_lod::stratified_allocation)).
//!
//! Once a path tracer knows how many samples to spend on a fibre it splits that
//! budget across the three Marschner lobes in proportion to the sampling pdf
//! (see [`crate::scatter_lod`]). The split is deterministic largest-remainder
//! (Hamilton) apportionment: floor each lobe's real quota, then hand the
//! leftover samples to the largest fractional remainders, ties broken by
//! ascending lobe index, so a normalised pdf's counts always sum to the budget.
//! This twin evaluates that split batch-wide on the device, one thread per
//! `(pdf, total)` tuple — the array-in/array-out form the stratified sampler
//! consumes per fibre or per cluster.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairStratifiedAllocation::eval`] takes a batch of `(LobePdf, total)`
//! tuples and returns, per tuple, the three per-lobe sample counts (`R`, `TT`,
//! `TRT`) in input order. The tuple index is the invocation id
//! (`@compute @workgroup_size(64)`, a one-dimensional dispatch over
//! `global_invocation_id.x`); invocations past the element count early-return.
//!
//! # Correctness model
//!
//! Each pdf component is sanitised in-shader bit-faithfully to the golden
//! (`sanitize_nonneg`: negative or non-finite collapses to `0`). The map is
//! integer-exact — `prob * total` is a single multiply and `floor` is exact — so
//! `CPU` and `GPU` agree bit-for-bit and the parity test asserts the counts with
//! integer equality. For a normalised pdf the floored quotas never exceed the
//! budget, so only the leftover top-up runs (at most two picks for three lobes);
//! the overflow trim is kept to mirror the golden exactly on any input.
//!
//! # Portability
//!
//! The kernel uses only compares, `max`, multiply, `floor` and small bounded
//! loops in the portable core-`WGSL` subset — no `exp`, `pow` or optional device
//! feature — so the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: largest-remainder (Hamilton) apportionment of a `Marschner`
//! three-lobe pdf plus `wgpu` compute dispatch; no Unreal Engine source or
//! derived code.

use core::mem::size_of;

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use prism_render_architecture::hair::scatter_lod::{stratified_allocation, LobePdf, LOBE_COUNT};

use crate::context::GpuContext;

/// Number of `f32`s each input tuple occupies: `pdf.r`, `pdf.tt`, `pdf.trt`,
/// `total` (an exact integer stored as `f32`).
const INPUT_PER_ELEMENT: usize = 4;

/// Uniform parameters for one allocation dispatch. Layout matches `Params` in
/// `shaders/stratified_allocation.wesl`: the element count padded to one
/// `16`-byte uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    elem_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable per-tuple stratified-allocation pipeline.
pub struct GpuHairStratifiedAllocation {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairStratifiedAllocation {
    /// Compiles the per-tuple stratified-allocation kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairStratifiedAllocation {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_stratified_allocation"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/stratified_allocation.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_stratified_allocation_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_stratified_allocation_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_stratified_allocation_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairStratifiedAllocation {
            module,
            layout,
            pipeline,
        }
    }

    /// Apportions each `(pdf, total)` tuple into three per-lobe sample counts,
    /// returning one `[usize; LOBE_COUNT]` per input in order.
    ///
    /// The counts for tuple `i` match the `CPU` golden
    /// [`stratified_allocation`](prism_render_architecture::hair::scatter_lod::stratified_allocation)
    /// bit-for-bit (integer-exact), with each pdf component sanitised to the same
    /// finite, non-negative value the reference uses. An empty batch yields an
    /// empty vector without a dispatch — storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, tuples: &[(LobePdf, usize)]) -> Vec<[usize; LOBE_COUNT]> {
        let elem_count = tuples.len();
        if elem_count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        let uniforms = Params {
            elem_count: elem_count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // Flatten each tuple to 4 f32: pdf.r, pdf.tt, pdf.trt, total. The raw
        // authored pdf is uploaded unchanged; the shader sanitises it.
        let mut flat: Vec<f32> = Vec::with_capacity(elem_count * INPUT_PER_ELEMENT);
        for &(pdf, total) in tuples {
            flat.push(pdf.r);
            flat.push(pdf.tt);
            flat.push(pdf.trt);
            flat.push(total as f32);
        }

        let out_len = elem_count * LOBE_COUNT;
        let out_bytes = (out_len as u64) * (size_of::<u32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_stratified_allocation_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let inputs_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_stratified_allocation_inputs"),
            contents: bytemuck::cast_slice(&flat),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_stratified_allocation_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_stratified_allocation_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_stratified_allocation_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: inputs_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_stratified_allocation_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_stratified_allocation_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (elem_count as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &out_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        out_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = out_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        let mut out = Vec::with_capacity(elem_count);
        for chunk in raw.chunks_exact(LOBE_COUNT) {
            out.push([chunk[0] as usize, chunk[1] as usize, chunk[2] as usize]);
        }
        out
    }
}

/// The `CPU` golden stratified allocation for one `(pdf, total)` tuple,
/// re-exported so the parity test can assert the device twin against the
/// identical reference it mirrors.
#[must_use]
pub fn reference_stratified_allocation(pdf: LobePdf, total: usize) -> [usize; LOBE_COUNT] {
    stratified_allocation(pdf, total)
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
