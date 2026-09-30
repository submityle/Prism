//! `wgpu` compute twin of Prism's per-frame root-binding resolve
//! ([`resolve_root_frames`](prism_render_architecture::hair::binding::resolve_root_frames)).
//!
//! A groom is authored in a character's rest pose, but the scalp is a skinned
//! mesh that deforms every frame. The import-time bake
//! ([`bind_roots`](prism_render_architecture::hair::binding::bind_roots)) pins
//! each strand root to its closest scalp triangle as a barycentric attachment
//! plus a signed height along the face normal; every frame that attachment is
//! replayed against the *deformed* triangle to reconstruct the root world
//! transform so the hair rides the head. This kernel is the on-device twin of
//! that per-frame replay — the root-skinning step `UE5` Groom / `TressFX` run on
//! the GPU: one thread per binding reads its three deformed corners, rebuilds
//! the outward face normal, interpolates the barycentric surface point, floats
//! it off by the stored height, and completes a right-handed orthonormal basis.
//! A passing real-device parity test is direct evidence the ported kernel skins
//! the roots the same way the reference does, not merely that its shader
//! compiles.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairRootResolve::eval`] takes the per-binding
//! [`MeshBinding`] slice, the current deformed vertex positions and the
//! triangle list, and returns one [`RootFrame`] per binding. The binding index
//! is simply the invocation id. It resolves; it does not bake — the
//! nearest-triangle search stays the once-per-import `bind_roots` pass.
//!
//! # Degenerate inputs
//!
//! An unbound or out-of-range triangle, an out-of-range corner index, or a
//! zero-area deformed face all resolve to [`RootFrame::IDENTITY`] exactly as the
//! reference does — a bad binding never panics and never poisons its neighbours.
//! An empty binding list returns an empty vector without a dispatch; empty
//! vertex / triangle pools are padded with one dummy entry (the count guards
//! keep the kernel from ever reading it) because storage buffers cannot be
//! zero-sized.
//!
//! # Portability
//!
//! The resolve uses only `sqrt`, `dot`, `cross` and multiply/add in the portable
//! core-`WGSL` subset — no `exp`, `pow`, `sin` or optional device feature — so
//! the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The barycentric blend, the face normal and the Gram-Schmidt basis contain no
//! transcendental call, so `CPU` and `GPU` evaluate the same closed-form
//! geometry; they are not bit-exact only because a `GPU` may fuse a multiply-add
//! the scalar reference leaves separate, so parity is asserted to within the
//! documented tolerance (`abs_diff < 1e-4` or `rel_diff < 1e-3`) per component.
//! Identity frames (unbound / degenerate) match exactly.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard barycentric mesh-attachment resolve plus a Gram-Schmidt
//! orthonormal basis plus `wgpu` compute dispatch; no Unreal Engine source or
//! derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::binding::{MeshBinding, RootFrame};
use prism_render_architecture::hair::interpolation::Vec3;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Uniform parameters for one root-resolve dispatch. Layout matches `Params` in
/// `shaders/root_resolve.wesl`: the binding, triangle and vertex counts packed
/// into one `16`-byte uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    binding_count: u32,
    triangle_count: u32,
    vertex_count: u32,
    pad0: u32,
}

/// One strand root's attachment in the shader's upload layout. Matches
/// `MeshBinding` in `shaders/root_resolve.wesl`: the triangle index, three
/// barycentric weights, the signed height and three pad words so the stride is a
/// `16`-byte multiple with no `vec3` alignment hazard.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuMeshBinding {
    triangle: u32,
    bary0: f32,
    bary1: f32,
    bary2: f32,
    height: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable per-binding root-resolve pipeline.
pub struct GpuHairRootResolve {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairRootResolve {
    /// Compiles the per-binding root-resolve kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairRootResolve {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_root_resolve"),
            source: ShaderSource::Wgsl(include_str!("../shaders/root_resolve.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_root_resolve_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_root_resolve_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_root_resolve_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairRootResolve {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves each `binding` against the current deformed `vertices` and
    /// `triangles`, producing one world-space [`RootFrame`] per binding.
    ///
    /// The frame for binding `t` equals the `CPU` golden
    /// [`resolve_root_frames`](prism_render_architecture::hair::binding::resolve_root_frames)
    /// at index `t` to within the module's documented tolerance (identity frames
    /// match exactly). An empty binding list yields an empty vector without a
    /// dispatch — storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        bindings: &[MeshBinding],
        vertices: &[Vec3],
        triangles: &[[u32; 3]],
    ) -> Vec<RootFrame> {
        let binding_count = bindings.len();
        if binding_count == 0 {
            return Vec::new();
        }

        let gpu_bindings: Vec<GpuMeshBinding> = bindings
            .iter()
            .map(|b| GpuMeshBinding {
                triangle: b.triangle,
                bary0: b.bary[0],
                bary1: b.bary[1],
                bary2: b.bary[2],
                height: b.height,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            })
            .collect();

        // Flatten the triangle list to three indices per face and the vertex
        // pool to xyzw so both upload as core-aligned arrays. Empty pools are
        // padded with one dummy entry; the count guards keep the kernel from
        // ever reading it.
        let mut tri_indices: Vec<u32> = Vec::with_capacity(triangles.len() * 3);
        for tri in triangles {
            tri_indices.extend_from_slice(&[tri[0], tri[1], tri[2]]);
        }
        if tri_indices.is_empty() {
            tri_indices.extend_from_slice(&[0, 0, 0]);
        }
        let mut verts: Vec<[f32; 4]> = vertices.iter().map(|v| [v.x, v.y, v.z, 0.0]).collect();
        if verts.is_empty() {
            verts.push([0.0, 0.0, 0.0, 0.0]);
        }

        let device = ctx.device();

        let uniforms = Params {
            binding_count: binding_count as u32,
            triangle_count: triangles.len() as u32,
            vertex_count: vertices.len() as u32,
            pad0: 0,
        };

        let out_bytes = (binding_count as u64) * 4 * (size_of::<[f32; 4]>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_root_resolve_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let bindings_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_root_resolve_bindings"),
            contents: bytemuck::cast_slice(&gpu_bindings),
            usage: BufferUsages::STORAGE,
        });
        let tris_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_root_resolve_tris"),
            contents: bytemuck::cast_slice(&tri_indices),
            usage: BufferUsages::STORAGE,
        });
        let verts_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_root_resolve_verts"),
            contents: bytemuck::cast_slice(&verts),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_root_resolve_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_root_resolve_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_root_resolve_bind_group"),
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
                    resource: tris_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: verts_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_root_resolve_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_root_resolve_pass"),
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
        let flat = bytemuck::cast_slice::<u8, [f32; 4]>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        flat.chunks_exact(4)
            .map(|q| RootFrame {
                position: Vec3::new(q[0][0], q[0][1], q[0][2]),
                tangent: Vec3::new(q[1][0], q[1][1], q[1][2]),
                normal: Vec3::new(q[2][0], q[2][1], q[2][2]),
                bitangent: Vec3::new(q[3][0], q[3][1], q[3][2]),
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
