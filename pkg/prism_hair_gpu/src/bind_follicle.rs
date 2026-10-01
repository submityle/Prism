//! `wgpu` compute twin of Prism's follicle-to-scalp binding capture
//! ([`bind_follicle`](prism_render_architecture::hair::follicle_bind::bind_follicle)),
//! the step that records how a hair root sits on its scalp triangle.
//!
//! Binding a follicle captures two things: the sanitised, on-surface barycentric
//! position of the root within the triangle, and the signed height of the root
//! above the surface along the interpolated normal — so a root floating slightly
//! off the skin keeps that height after the scalp deforms. This kernel
//! broadcasts one shared triangle (vertex positions and per-vertex normals) to a
//! whole batch of roots, one thread per root, and returns each root's binding in
//! input order.
//!
//! # A distinct sibling of the other follicle twins
//!
//! The [`compute_barycentric`](crate::compute_barycentric) twin returns the raw,
//! un-clamped planar-projection weights; this kernel composes that solve with the
//! `sanitized` clamp/renormalise (so the stored weights are non-negative and sum
//! to one) and then projects the signed normal offset — a different, larger
//! output. The [`follicle_bind`](crate::follicle_bind) root-transfer twin instead
//! consumes a finished binding to rebuild a world position; this is the earlier
//! capture step that *produces* the binding it would later transfer.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairBindFollicle::eval`] takes a batch of root positions and one shared
//! [`TriangleFrame`] (its vertex positions and per-vertex normals are used), and
//! returns each root's [`FollicleBinding`] in input order. The root index is the
//! invocation id (`@compute @workgroup_size(64)`, one-dimensional dispatch over
//! `global_invocation_id.x`); invocations past the root count early-return.
//!
//! # Degenerate inputs
//!
//! A zero-area (or non-finite determinant) triangle makes the raw solve fall
//! back to the centroid; an all-non-positive (or non-finite) sanitised weight set
//! likewise collapses to the centroid; a zero-length interpolated normal falls
//! back to `+Z`; a non-finite offset is clamped to zero — each exactly as the
//! golden does. An empty batch yields an empty vector without a dispatch —
//! storage buffers cannot be zero-sized.
//!
//! # Correctness model
//!
//! The barycentric solve, folds, normalise and offset projection are
//! dot/determinant chains plus a reciprocal, a `sqrt`-reciprocal normalise and
//! multiply-adds a `GPU` may fuse, perturbing the low mantissa bits by a few
//! `ULP`. The parity test therefore asserts a tolerance (`abs_diff < 1e-4` or
//! `rel_diff < 1e-3`) per component rather than a raw bit compare.
//!
//! # Portability
//!
//! The kernel uses only `dot`, compare, `abs`, `sqrt`, a reciprocal and
//! multiply/add in the portable core-`WGSL` subset — no `exp`, `pow` or optional
//! device feature — so the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: Ericson "Real-Time Collision Detection" barycentric solve plus
//! standard mesh-skinning attachment; no Unreal Engine source or derived code.

use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::follicle_bind::{
    bind_follicle, FollicleBinding, TriangleFrame,
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

/// Uniform parameters for one binding dispatch. Layout matches `Params` in
/// `shaders/bind_follicle.wesl`: the one shared triangle (three vertex positions
/// then three per-vertex normals as `16`-byte `vec4` slots, xyz used) followed by
/// the root count in a final `16`-byte slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    pos0: [f32; 4],
    pos1: [f32; 4],
    pos2: [f32; 4],
    nrm0: [f32; 4],
    nrm1: [f32; 4],
    nrm2: [f32; 4],
    root_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable per-root follicle-binding pipeline.
pub struct GpuHairBindFollicle {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairBindFollicle {
    /// Compiles the per-root follicle-binding kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset (`dot`, compare,
    /// `abs`, `sqrt`, a reciprocal and multiply/add), so no optional device
    /// feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairBindFollicle {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_bind_follicle"),
            source: ShaderSource::Wgsl(include_str!("../shaders/bind_follicle.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_bind_follicle_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_bind_follicle_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_bind_follicle_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairBindFollicle {
            module,
            layout,
            pipeline,
        }
    }

    /// Binds a batch of roots to one shared triangle, returning each root's
    /// [`FollicleBinding`] in input order.
    ///
    /// Each binding matches the `CPU` golden
    /// [`bind_follicle`](prism_render_architecture::hair::follicle_bind::bind_follicle)
    /// of the same inputs within an fma tolerance (the only departures are
    /// possibly-fused multiply-adds and a possibly-differently-rounded reciprocal
    /// / `sqrt`). An empty batch yields an empty vector without a dispatch —
    /// storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        roots: &[[f32; 3]],
        tri: TriangleFrame,
    ) -> Vec<FollicleBinding> {
        let root_count = roots.len();
        if root_count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        let p = tri.positions;
        let n = tri.normals;
        let uniforms = Params {
            pos0: [p[0][0], p[0][1], p[0][2], 0.0],
            pos1: [p[1][0], p[1][1], p[1][2], 0.0],
            pos2: [p[2][0], p[2][1], p[2][2], 0.0],
            nrm0: [n[0][0], n[0][1], n[0][2], 0.0],
            nrm1: [n[1][0], n[1][1], n[1][2], 0.0],
            nrm2: [n[2][0], n[2][1], n[2][2], 0.0],
            root_count: root_count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // Roots are uploaded one vec4 (16 bytes) each; xyz is the position.
        let gpu_roots: Vec<[f32; 4]> = roots.iter().map(|q| [q[0], q[1], q[2], 0.0]).collect();

        // Output is one vec4 (16 bytes) per root; xyz holds (u, v, w), w the offset.
        let out_bytes = (root_count as u64) * ((size_of::<f32>() * 4) as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_bind_follicle_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let roots_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_bind_follicle_roots"),
            contents: bytemuck::cast_slice(&gpu_roots),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_bind_follicle_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_bind_follicle_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_bind_follicle_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: roots_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_bind_follicle_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_bind_follicle_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (root_count as u32).div_ceil(64);
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

        raw.chunks_exact(4)
            .map(|c| FollicleBinding {
                bary: prism_render_architecture::hair::follicle_bind::Barycentric {
                    u: c[0],
                    v: c[1],
                    w: c[2],
                },
                normal_offset: c[3],
            })
            .collect()
    }
}

/// The `CPU` golden follicle binding for one root, re-exported so the parity
/// test can assert the device twin against the identical reference it mirrors.
#[must_use]
pub fn reference_bind_follicle(root: [f32; 3], tri: TriangleFrame) -> FollicleBinding {
    bind_follicle(root, tri)
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
