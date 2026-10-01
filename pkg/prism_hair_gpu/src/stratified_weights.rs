//! `wgpu` compute twin of Prism's stratified per-lobe sample weights
//! ([`stratified_weights`](prism_render_architecture::hair::scatter_lod::stratified_weights)).
//!
//! Once the stratified sampler has split a budget across the three Marschner
//! lobes (see [`crate::stratified_allocation`]) each lobe needs the Monte-Carlo
//! weight that keeps the estimator unbiased: `pdf_i * total / count_i`, the
//! reciprocal of the per-sample probability. This twin evaluates that weight
//! vector batch-wide on the device, one thread per `(pdf, total)` tuple — the
//! array-in/array-out form the sampler consumes per fibre or per cluster.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairStratifiedWeights::eval`] takes a batch of `(LobePdf, total)` tuples
//! and returns, per tuple, the three per-lobe weights (`R`, `TT`, `TRT`) in
//! input order. The kernel first reproduces the integer-exact largest-remainder
//! allocation, then divides to form each weight; a lobe that received no samples
//! (`count_i == 0`) or a zero budget yields weight `0`. The tuple index is the
//! invocation id (`@compute @workgroup_size(64)`, a one-dimensional dispatch
//! over `global_invocation_id.x`); invocations past the element count
//! early-return.
//!
//! # Correctness model
//!
//! Each pdf component is sanitised in-shader bit-faithfully to the golden
//! (`sanitize_nonneg`: negative or non-finite collapses to `0`). The allocation
//! is integer-exact, so the only inexact step is the final
//! `prob * total / count` divide, which a `GPU` may round a few `ULP`
//! differently; the parity test therefore compares weights with a tolerance
//! (`abs_diff < 1e-4` or `rel_diff < 1e-3`). As with the allocation twin the
//! three-step leftover bound is exact only for a normalised pdf, so the parity
//! suite drives normalised pdfs.
//!
//! # Portability
//!
//! The kernel uses only compares, `max`, multiply, `floor`, a single divide and
//! small bounded loops in the portable core-`WGSL` subset — no `exp`, `pow` or
//! optional device feature — so the twin runs unmodified on Metal, Vulkan and
//! DX12.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: largest-remainder (Hamilton) apportionment of a `Marschner`
//! three-lobe pdf and its unbiased stratified weights plus `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.

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

use prism_render_architecture::hair::scatter_lod::{stratified_weights, LobePdf, LOBE_COUNT};

use crate::context::GpuContext;

/// Number of `f32`s each input tuple occupies: `pdf.r`, `pdf.tt`, `pdf.trt`,
/// `total` (an exact integer stored as `f32`).
const INPUT_PER_ELEMENT: usize = 4;

/// Uniform parameters for one weights dispatch. Layout matches `Params` in
/// `shaders/stratified_weights.wesl`: the element count padded to one `16`-byte
/// uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    elem_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable per-tuple stratified-weights pipeline.
pub struct GpuHairStratifiedWeights {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairStratifiedWeights {
    /// Compiles the per-tuple stratified-weights kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairStratifiedWeights {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_stratified_weights"),
            source: ShaderSource::Wgsl(include_str!("../shaders/stratified_weights.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_stratified_weights_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_stratified_weights_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_stratified_weights_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairStratifiedWeights {
            module,
            layout,
            pipeline,
        }
    }

    /// Computes each `(pdf, total)` tuple's three per-lobe sample weights,
    /// returning one `[f32; LOBE_COUNT]` per input in order.
    ///
    /// The weights for tuple `i` match the `CPU` golden
    /// [`stratified_weights`](prism_render_architecture::hair::scatter_lod::stratified_weights)
    /// within the fma tolerance, with each pdf component sanitised to the same
    /// finite, non-negative value the reference uses. An empty batch yields an
    /// empty vector without a dispatch — storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, tuples: &[(LobePdf, usize)]) -> Vec<[f32; LOBE_COUNT]> {
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
        let out_bytes = (out_len as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_stratified_weights_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let inputs_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_stratified_weights_inputs"),
            contents: bytemuck::cast_slice(&flat),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_stratified_weights_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_stratified_weights_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_stratified_weights_bind_group"),
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
            label: Some("prism_hair_stratified_weights_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_stratified_weights_pass"),
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
        let raw = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        let mut out = Vec::with_capacity(elem_count);
        for chunk in raw.chunks_exact(LOBE_COUNT) {
            out.push([chunk[0], chunk[1], chunk[2]]);
        }
        out
    }
}

/// The `CPU` golden stratified weight vector for one `(pdf, total)` tuple,
/// re-exported so the parity test can assert the device twin against the
/// identical reference it mirrors.
#[must_use]
pub fn reference_stratified_weights(pdf: LobePdf, total: usize) -> [f32; LOBE_COUNT] {
    let v = stratified_weights(pdf, total);
    let mut out = [0.0f32; LOBE_COUNT];
    let mut i = 0;
    while i < LOBE_COUNT {
        out[i] = v[i];
        i += 1;
    }
    out
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
