//! `wgpu` compute twin of Prism's follicle local-frame transfer
//! ([`transfer_frame`](prism_render_architecture::hair::follicle_bind::transfer_frame)),
//! the step that rebuilds each follicle's orthonormal basis on the deformed
//! scalp.
//!
//! After a scalp triangle deforms, each follicle needs a fresh orthonormal local
//! frame (normal, tangent, bitangent) at its root so the strand can be
//! re-oriented. The golden interpolates the per-vertex normal and tangent by the
//! binding's sanitised barycentric weights, then Gram-Schmidt re-orthonormalises
//! them into a right-handed basis. This kernel broadcasts one shared deformed
//! triangle (per-vertex normals and tangents) to a whole batch of bindings, one
//! thread per binding, and returns each binding's frame in input order.
//!
//! # A distinct sibling of the other follicle twins
//!
//! The [`follicle_bind`](crate::follicle_bind) root-transfer twin reconstructs
//! the root *position* (barycentric surface point plus a signed normal offset);
//! this kernel reconstructs the root's orthonormal *frame* (interpolation plus
//! Gram-Schmidt) — a different computation and a different output. Only the
//! binding's barycentric weights are consumed here; the normal offset plays no
//! part in the frame.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairTransferFrame::eval`] takes a batch of [`FollicleBinding`]s and one
//! shared deformed [`TriangleFrame`] (its per-vertex normals and tangents are
//! used), and returns each binding's [`FollicleFrame`] in input order. The
//! binding index is the invocation id (`@compute @workgroup_size(64)`,
//! one-dimensional dispatch over `global_invocation_id.x`); invocations past the
//! binding count early-return.
//!
//! # Degenerate inputs
//!
//! An all-non-positive (or non-finite) weight set sanitises to the centroid; a
//! zero-length interpolated normal falls back to `+Z`; a tangent (near) parallel
//! to the normal falls back to a canonical axis chosen by the normal's x
//! magnitude; a zero-length bitangent falls back to `+Y` — each exactly as the
//! golden does, so the result is always a finite orthonormal-ish basis. An empty
//! batch yields an empty vector without a dispatch — storage buffers cannot be
//! zero-sized.
//!
//! # Correctness model
//!
//! The interpolation folds, the Gram-Schmidt projection and three normalises are
//! multiply-adds and `sqrt`-reciprocals a `GPU` may fuse, perturbing the low
//! mantissa bits by a few `ULP`. The parity test therefore asserts a tolerance
//! (`abs_diff < 1e-4` or `rel_diff < 1e-3`) per component rather than a raw bit
//! compare.
//!
//! # Portability
//!
//! The kernel uses only `dot`, `cross`, compare, `abs`, `sqrt`, a reciprocal and
//! multiply/add in the portable core-`WGSL` subset — no `exp`, `pow` or optional
//! device feature — so the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard barycentric mesh-attachment frame transfer plus
//! Gram-Schmidt orthonormalisation; no Unreal Engine source or derived code.

use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::follicle_bind::{
    transfer_frame, FollicleBinding, FollicleFrame, TriangleFrame,
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

extern crate alloc;

/// Uniform parameters for one frame-transfer dispatch. Layout matches `Params`
/// in `shaders/transfer_frame.wesl`: the one shared deformed triangle (three
/// per-vertex normals then three per-vertex tangents as `16`-byte `vec4` slots,
/// xyz used) followed by the binding count in a final `16`-byte slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    nrm0: [f32; 4],
    nrm1: [f32; 4],
    nrm2: [f32; 4],
    tan0: [f32; 4],
    tan1: [f32; 4],
    tan2: [f32; 4],
    binding_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable per-binding frame-transfer pipeline.
pub struct GpuHairTransferFrame {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairTransferFrame {
    /// Compiles the per-binding frame-transfer kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset (`dot`, `cross`,
    /// compare, `abs`, `sqrt`, a reciprocal and multiply/add), so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairTransferFrame {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_transfer_frame"),
            source: ShaderSource::Wgsl(include_str!("../shaders/transfer_frame.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_transfer_frame_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_transfer_frame_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_transfer_frame_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairTransferFrame {
            module,
            layout,
            pipeline,
        }
    }

    /// Transfers a batch of bindings onto one shared deformed triangle, returning
    /// each binding's [`FollicleFrame`] in input order.
    ///
    /// Each frame matches the `CPU` golden
    /// [`transfer_frame`](prism_render_architecture::hair::follicle_bind::transfer_frame)
    /// of the same inputs within an fma tolerance (the only departures are
    /// possibly-fused multiply-adds and possibly-differently-rounded reciprocals
    /// / `sqrt`s). An empty batch yields an empty vector without a dispatch —
    /// storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        bindings: &[FollicleBinding],
        deformed: TriangleFrame,
    ) -> Vec<FollicleFrame> {
        let binding_count = bindings.len();
        if binding_count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        let n = deformed.normals;
        let t = deformed.tangents;
        let uniforms = Params {
            nrm0: [n[0][0], n[0][1], n[0][2], 0.0],
            nrm1: [n[1][0], n[1][1], n[1][2], 0.0],
            nrm2: [n[2][0], n[2][1], n[2][2], 0.0],
            tan0: [t[0][0], t[0][1], t[0][2], 0.0],
            tan1: [t[1][0], t[1][1], t[1][2], 0.0],
            tan2: [t[2][0], t[2][1], t[2][2], 0.0],
            binding_count: binding_count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // Each binding uploads one vec4 (16 bytes): (u, v, w) in xyz, offset in w.
        let gpu_bindings: Vec<[f32; 4]> = bindings
            .iter()
            .map(|b| [b.bary.u, b.bary.v, b.bary.w, b.normal_offset])
            .collect();

        // Output is three vec4 (48 bytes) per binding: normal, tangent, bitangent.
        let out_bytes = (binding_count as u64) * ((size_of::<f32>() * 4 * 3) as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_transfer_frame_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let bindings_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_transfer_frame_bindings"),
            contents: bytemuck::cast_slice(&gpu_bindings),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_transfer_frame_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_transfer_frame_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_transfer_frame_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: bindings_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_transfer_frame_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_transfer_frame_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (binding_count as u32).div_ceil(64);
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

        // 12 floats per binding: normal[0..3], tangent[4..7], bitangent[8..11]
        // (the w lane of each vec4 is padding).
        raw.chunks_exact(12)
            .map(|c| FollicleFrame {
                normal: [c[0], c[1], c[2]],
                tangent: [c[4], c[5], c[6]],
                bitangent: [c[8], c[9], c[10]],
            })
            .collect()
    }
}

/// The `CPU` golden frame transfer for one binding, re-exported so the parity
/// test can assert the device twin against the identical reference it mirrors.
#[must_use]
pub fn reference_transfer_frame(
    binding: FollicleBinding,
    deformed: TriangleFrame,
) -> FollicleFrame {
    transfer_frame(binding, deformed)
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
