//! `wgpu` compute twin of Prism's import-time root-binding bake
//! ([`bind_roots`](prism_render_architecture::hair::binding::bind_roots)).
//!
//! A groom is authored in a character's rest pose. Before it can ride the
//! skinned scalp every frame (the
//! [`resolve_root_frames`](prism_render_architecture::hair::binding::resolve_root_frames)
//! replay), each strand root must first be pinned to the surface it grew from:
//! at import time the root is projected onto its closest scalp triangle and
//! stored as a barycentric attachment plus a signed height along the face
//! normal. That is the bake `UE5` Groom builds into its binding asset and the
//! root-skinning setup `TressFX` runs once per groom. This kernel is the
//! on-device twin of exactly that bake — one thread per root brute-force scans
//! every triangle for the nearest closest-surface-point, then records the
//! barycentric projection and signed height. It is the sibling of the
//! [`root_resolve`](crate::root_resolve) per-frame replay: this one bakes the
//! attachment, that one replays it against the deformed mesh. A passing
//! real-device parity test is direct evidence the ported kernel bakes the same
//! attachments the reference does, not merely that its shader compiles.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairRootBind::eval`] takes the authored root positions, the rest-pose
//! vertex pool and the triangle list, and returns one [`MeshBinding`] per root
//! (in input order). The root index is simply the invocation id, and each
//! thread writes one disjoint output slot, so the pass is race-free. It bakes;
//! it does not resolve — the per-frame replay stays the `root_resolve` pass.
//!
//! # Degenerate inputs
//!
//! A triangle whose vertex index is out of range is skipped exactly as the
//! reference's `triangle_corners` skips it; a root that finds no bindable
//! triangle (an empty triangle list, or every face out of range) resolves to
//! [`MeshBinding::UNBOUND`] with zeroed weights; a zero-area face contributes a
//! zero height. An empty root list returns an empty vector without a dispatch;
//! empty vertex / triangle pools are padded with one dummy entry (the count
//! guards keep the kernel from ever reading it) because storage buffers cannot
//! be zero-sized.
//!
//! # Portability
//!
//! The scan uses only `sqrt`, `dot`, `cross`, comparisons and multiply/add in
//! the portable core-`WGSL` subset — no `exp`, `pow`, `sin` or optional device
//! feature — so the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The closest-point Voronoi-region test and the barycentric weights contain no
//! transcendental call, so `CPU` and `GPU` walk the same branch and evaluate the
//! same closed-form geometry. The chosen triangle index is an integer selection,
//! so parity asserts it *exactly*; the barycentric weights and the signed height
//! are not bit-exact only because a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, so they are asserted to within the documented
//! tolerance (`abs_diff < 1e-4` or `rel_diff < 1e-3`) per component. Unbound
//! roots match exactly (sentinel index, zeroed weights and height). The test
//! geometry keeps each root's nearest triangle unambiguous so the integer
//! selection can never flip under that fma perturbation.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard closest-point-on-triangle (Ericson Voronoi-region test)
//! plus a brute-force nearest scan plus `wgpu` compute dispatch; no Unreal
//! Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::binding::MeshBinding;
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

/// Uniform parameters for one root-bind dispatch. Layout matches `Params` in
/// `shaders/root_bind.wesl`: the root, triangle and vertex counts packed into
/// one `16`-byte uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    root_count: u32,
    triangle_count: u32,
    vertex_count: u32,
    pad0: u32,
}

/// One strand root's baked attachment in the shader's upload layout. Matches
/// `MeshBinding` in `shaders/root_bind.wesl`: the triangle index, three
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

/// A compiled, reusable per-root root-bind pipeline.
pub struct GpuHairRootBind {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairRootBind {
    /// Compiles the per-root root-bind kernel once so a caller can bake many
    /// grooms without rebuilding the pipeline.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairRootBind {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_root_bind"),
            source: ShaderSource::Wgsl(include_str!("../shaders/root_bind.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_root_bind_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_root_bind_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_root_bind_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairRootBind {
            module,
            layout,
            pipeline,
        }
    }

    /// Bakes each `root` against the rest-pose `vertices` and `triangles`,
    /// producing one [`MeshBinding`] per root (in input order).
    ///
    /// The binding for root `t` equals the `CPU` golden
    /// [`bind_roots`](prism_render_architecture::hair::binding::bind_roots) at
    /// index `t`: the chosen triangle index is bit-identical (an integer
    /// selection) and the barycentric weights and signed height match to within
    /// the module's documented tolerance (unbound roots match exactly). An empty
    /// root list yields an empty vector without a dispatch — storage buffers
    /// cannot be zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        roots: &[Vec3],
        vertices: &[Vec3],
        triangles: &[[u32; 3]],
    ) -> Vec<MeshBinding> {
        let root_count = roots.len();
        if root_count == 0 {
            return Vec::new();
        }

        // Upload roots and vertices as xyzw so both align as core `vec4` arrays.
        let root_pool: Vec<[f32; 4]> = roots.iter().map(|r| [r.x, r.y, r.z, 0.0]).collect();

        // Flatten the triangle list to three indices per face. Empty pools are
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
            root_count: root_count as u32,
            triangle_count: triangles.len() as u32,
            vertex_count: vertices.len() as u32,
            pad0: 0,
        };

        let out_bytes = (root_count as u64) * (size_of::<GpuMeshBinding>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_root_bind_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let roots_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_root_bind_roots"),
            contents: bytemuck::cast_slice(&root_pool),
            usage: BufferUsages::STORAGE,
        });
        let tris_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_root_bind_tris"),
            contents: bytemuck::cast_slice(&tri_indices),
            usage: BufferUsages::STORAGE,
        });
        let verts_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_root_bind_verts"),
            contents: bytemuck::cast_slice(&verts),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_root_bind_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_root_bind_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_root_bind_bind_group"),
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
            label: Some("prism_hair_root_bind_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_root_bind_pass"),
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
        let baked = bytemuck::cast_slice::<u8, GpuMeshBinding>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        baked
            .into_iter()
            .map(|g| MeshBinding {
                triangle: g.triangle,
                bary: [g.bary0, g.bary1, g.bary2],
                height: g.height,
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
