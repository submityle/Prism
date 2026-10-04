//! Host orchestration of the hierarchy-propagation compute kernel (§24.7 twin).
//!
//! [`GpuHierarchyPropagate`] walks a transform forest on a real device, level
//! by level, mirroring the CPU reference
//! [`propagate_by_levels`](prism_transform::compute_hierarchy::propagate_by_levels).
//! It consumes the device-free upload payload produced by
//! [`ComputeHierarchyInput::pack`](prism_transform::compute_hierarchy::ComputeHierarchyInput::pack):
//! the parent-index array, the row-major local matrices, and the flattened
//! per-level dispatch order. One compute pass is recorded per depth level
//! (shallow to deep); because every node in a level has its parent in a
//! shallower, already-dispatched level, each child reads a finalized parent
//! world matrix. `wgpu` inserts the storage-buffer barrier between the passes.
//!
//! # Layout
//!
//! The kernel decodes exactly the row-major 3x4 affine
//! [`MatrixLayout::RowMajor3x4`](prism_transform::gpu_upload::MatrixLayout) that
//! `pack` emits: three `vec4<f32>` rows per node where the first three lanes of
//! each row hold a basis-column component and the fourth lane holds that row's
//! translation component. The world output uses the same encoding, so a
//! renderer can keep the result resident for instancing / indirect draw with no
//! re-encode.
//!
//! # Parity, not bit-exactness
//!
//! The affine composition is floating-point multiply-accumulate (3x3 matrix
//! products plus a translated point). Metal compiles WGSL under fast-math, so
//! the compiler may contract and reassociate; the parity tests therefore assert
//! agreement within a small absolute+relative tolerance.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_math::{Affine3, Mat3, Vec3};
use prism_transform::GlobalTransform;
use prism_transform::compute_hierarchy::ComputeHierarchyInput;
use prism_transform::gpu_upload::MatrixLayout;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, PipelineCompilationOptions,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::buffer;
use crate::context::GpuContext;

/// Compute workgroup length (the kernel is `@workgroup_size(64, 1, 1)`).
pub const WORKGROUP: u32 = 64;

/// Floats per packed node matrix in [`MatrixLayout::RowMajor3x4`].
const MAT3X4_FLOATS: usize = 12;

/// The hierarchy-propagation compute kernel.
///
/// `locals` and `worlds` are `array<Mat3x4>` where `Mat3x4` is three
/// `vec4<f32>` rows (row-major 3x4 affine, 48 bytes). `affine_mul(a, b)`
/// composes `a ∘ b` with the identical column-major math as
/// [`prism_math::Affine3`]'s `Mul`: `world.basis = a.basis * b.basis` and
/// `world.t = a.basis * b.t + a.t`. One dispatch per level walks
/// `dispatch_order[params.start .. params.start + params.count]`.
const KERNEL_WGSL: &str = "\
struct Mat3x4 {\n\
    row0: vec4<f32>,\n\
    row1: vec4<f32>,\n\
    row2: vec4<f32>,\n\
};\n\
\n\
struct LevelParams {\n\
    start: u32,\n\
    count: u32,\n\
    pad0: u32,\n\
    pad1: u32,\n\
};\n\
\n\
@group(0) @binding(0) var<uniform> params: LevelParams;\n\
@group(0) @binding(1) var<storage, read> parents: array<i32>;\n\
@group(0) @binding(2) var<storage, read> dispatch_order: array<u32>;\n\
@group(0) @binding(3) var<storage, read> locals: array<Mat3x4>;\n\
@group(0) @binding(4) var<storage, read_write> worlds: array<Mat3x4>;\n\
\n\
fn affine_mul(a: Mat3x4, b: Mat3x4) -> Mat3x4 {\n\
    let a_c0 = vec3<f32>(a.row0.x, a.row1.x, a.row2.x);\n\
    let a_c1 = vec3<f32>(a.row0.y, a.row1.y, a.row2.y);\n\
    let a_c2 = vec3<f32>(a.row0.z, a.row1.z, a.row2.z);\n\
    let a_t  = vec3<f32>(a.row0.w, a.row1.w, a.row2.w);\n\
    let b_c0 = vec3<f32>(b.row0.x, b.row1.x, b.row2.x);\n\
    let b_c1 = vec3<f32>(b.row0.y, b.row1.y, b.row2.y);\n\
    let b_c2 = vec3<f32>(b.row0.z, b.row1.z, b.row2.z);\n\
    let b_t  = vec3<f32>(b.row0.w, b.row1.w, b.row2.w);\n\
    let w_c0 = a_c0 * b_c0.x + a_c1 * b_c0.y + a_c2 * b_c0.z;\n\
    let w_c1 = a_c0 * b_c1.x + a_c1 * b_c1.y + a_c2 * b_c1.z;\n\
    let w_c2 = a_c0 * b_c2.x + a_c1 * b_c2.y + a_c2 * b_c2.z;\n\
    let w_t  = a_c0 * b_t.x + a_c1 * b_t.y + a_c2 * b_t.z + a_t;\n\
    return Mat3x4(\n\
        vec4<f32>(w_c0.x, w_c1.x, w_c2.x, w_t.x),\n\
        vec4<f32>(w_c0.y, w_c1.y, w_c2.y, w_t.y),\n\
        vec4<f32>(w_c0.z, w_c1.z, w_c2.z, w_t.z),\n\
    );\n\
}\n\
\n\
@compute @workgroup_size(64, 1, 1)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let t = gid.x;\n\
    if (t >= params.count) {\n\
        return;\n\
    }\n\
    let node = dispatch_order[params.start + t];\n\
    let local = locals[node];\n\
    let p = parents[node];\n\
    var world: Mat3x4;\n\
    if (p < 0) {\n\
        world = local;\n\
    } else {\n\
        world = affine_mul(worlds[u32(p)], local);\n\
    }\n\
    worlds[node] = world;\n\
}\n";

/// Uniform block shared with `LevelParams` in the kernel. `start` and `count`
/// slice [`ComputeHierarchyInput::dispatch_order`] to one depth level; the two
/// pads bring the struct to the 16-byte uniform alignment WGSL assigns it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LevelParams {
    start: u32,
    count: u32,
    pad0: u32,
    pad1: u32,
}

/// `Pod` mirror of the WGSL `Mat3x4` storage element: three `vec4<f32>` rows
/// (row-major 3x4 affine, 48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuMat3x4 {
    rows: [f32; MAT3X4_FLOATS],
}

impl GpuMat3x4 {
    /// Decode the row-major 3x4 scalars back into an [`Affine3`]. Row `r` is
    /// `[basis.x[r], basis.y[r], basis.z[r], translation[r]]`, matching
    /// `prism_transform::gpu_upload`'s packing.
    fn to_affine(self) -> Affine3 {
        let f = self.rows;
        let x_axis = Vec3::new(f[0], f[4], f[8]);
        let y_axis = Vec3::new(f[1], f[5], f[9]);
        let z_axis = Vec3::new(f[2], f[6], f[10]);
        let translation = Vec3::new(f[3], f[7], f[11]);
        Affine3 {
            matrix3: Mat3::from_cols(x_axis, y_axis, z_axis),
            translation,
        }
    }
}

/// Compiled hierarchy-propagation pipeline and its bind-group layout.
pub struct GpuHierarchyPropagate {
    pipeline: ComputePipeline,
    layout: BindGroupLayout,
}

impl GpuHierarchyPropagate {
    /// Compiles the hierarchy-propagation kernel on `ctx`'s device.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHierarchyPropagate {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_transform_hierarchy_propagate"),
            source: ShaderSource::Wgsl(KERNEL_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_transform_hierarchy_propagate_layout"),
            entries: &[
                buffer_layout(
                    0,
                    BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                buffer_layout(
                    1,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                buffer_layout(
                    2,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                buffer_layout(
                    3,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                buffer_layout(
                    4,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_transform_hierarchy_propagate_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_transform_hierarchy_propagate_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHierarchyPropagate { pipeline, layout }
    }

    /// Propagates `input` on the device and returns one world
    /// [`GlobalTransform`] per node, node-indexed, mirroring
    /// [`propagate_by_levels`](prism_transform::compute_hierarchy::propagate_by_levels).
    ///
    /// An empty input returns an empty `Vec` without dispatching.
    ///
    /// # Panics
    /// Panics if `input`'s matrix layout is not
    /// [`MatrixLayout::RowMajor3x4`] (the only encoding this kernel decodes).
    #[must_use]
    pub fn propagate(&self, ctx: &GpuContext, input: &ComputeHierarchyInput) -> Vec<GlobalTransform> {
        assert_eq!(
            input.layout(),
            MatrixLayout::RowMajor3x4,
            "GpuHierarchyPropagate decodes only the RowMajor3x4 affine layout"
        );
        let node_count = input.node_count();
        if node_count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        let parents_buf =
            buffer::storage_read(device, "prism_transform_hierarchy_parents", input.parents());
        let order_buf = buffer::storage_read(
            device,
            "prism_transform_hierarchy_dispatch_order",
            input.dispatch_order(),
        );
        let locals_buf = buffer::storage_bytes(
            device,
            "prism_transform_hierarchy_locals",
            input.local_matrices(),
        );
        let worlds_bytes = (node_count * MAT3X4_FLOATS * size_of::<f32>()) as u64;
        let worlds_buf =
            buffer::storage_rw_zeroed(device, "prism_transform_hierarchy_worlds", worlds_bytes);

        // One uniform + bind group per level so each pass addresses its own
        // dispatch-order slice; the shared storage buffers are reused.
        let level_bind_groups: Vec<(LevelParams, wgpu::BindGroup)> = input
            .level_ranges()
            .iter()
            .map(|&(start, end)| {
                let params = LevelParams {
                    start,
                    count: end - start,
                    pad0: 0,
                    pad1: 0,
                };
                let params_buf =
                    buffer::uniform(device, "prism_transform_hierarchy_level_params", &params);
                let bind_group = device.create_bind_group(&BindGroupDescriptor {
                    label: Some("prism_transform_hierarchy_bind_group"),
                    layout: &self.layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: params_buf.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: parents_buf.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: order_buf.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: locals_buf.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 4,
                            resource: worlds_buf.as_entire_binding(),
                        },
                    ],
                });
                // Keep `params_buf` alive via the closure's bind group, which
                // retains a reference to it.
                let _ = params_buf;
                (params, bind_group)
            })
            .collect();

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_transform_hierarchy_encoder"),
        });
        for (params, bind_group) in &level_bind_groups {
            if params.count == 0 {
                continue;
            }
            let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_transform_hierarchy_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(params.count.div_ceil(WORKGROUP), 1, 1);
        }

        let stage = buffer::staging(device, "prism_transform_hierarchy_stage", worlds_bytes);
        buffer::copy(&mut enc, &worlds_buf, &stage, worlds_bytes);
        ctx.queue().submit([enc.finish()]);

        let out: Vec<GpuMat3x4> = buffer::read_back(ctx, &stage);
        out.iter()
            .take(node_count)
            .map(|m| GlobalTransform(m.to_affine()))
            .collect()
    }
}

/// One storage/uniform bind-group-layout entry visible to the compute stage.
fn buffer_layout(binding: u32, ty: BindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}
