//! `wgpu` compute twin of Prism's strand-to-shell meshing
//! ([`build_shell`](prism_render_architecture::hair::mesh_shell::build_shell)).
//!
//! `Mesh` is the coarsest rung of the hair LOD ladder. Past full sub-pixel
//! strands, reduced strands and the flat `Cards` ribbon ([`crate::ribbon`]), a
//! receding group of hairs is far enough that individual fibers are
//! indistinguishable, so `UE5` Groom and comparable engines collapse them onto a
//! baked low triangle-count shell tube swept along the strand centerline. Unlike
//! the flat ribbon, the shell has volume: each control point contributes a
//! four-corner rectangular cross-section oriented by the strand's coherent frame
//! (from the [`crate::frames`] twin) — the bitangent spans the width, the normal
//! spans the thickness. The `CPU` golden for that meshing is
//! [`build_shell`](prism_render_architecture::hair::mesh_shell::build_shell); this
//! crate is the on-device twin that walks the same meshing, one thread per
//! strand, so a passing real-device parity test is direct evidence the ported
//! kernel computes the same shell vertices as the reference — not merely that
//! its shader compiles.
//!
//! # What the kernel evaluates
//!
//! [`GpuMeshShell::eval`] takes a batch of strands (each its control points, the
//! per-point bitangent and normal from the frame transport, and a uniform
//! half-width / half-thickness) and returns one [`GpuShellMesh`] per strand.
//! Each mesh carries four ring vertices per control point: corner `j` sits at
//! `sign_w · bitangent · half_width + sign_h · normal · half_thickness` about the
//! centerline, its outward normal is the (normalized) `sign_w · bitangent +
//! sign_h · normal` diagonal, and its UV is `(j · 0.25, v)` where `v` runs `0` at
//! the root to `1` at the tip along arc length. The triangle-list indices are
//! reconstructed on the host from the deterministic winding the golden uses (a
//! pure function of the point count, carrying no `GPU` float error).
//!
//! Every guard is reproduced: a strand with fewer than two points (or with
//! mismatched attribute lengths) yields an empty mesh, and a zero-length strand
//! (coincident points) falls back to even `v` spacing rather than dividing by
//! zero.
//!
//! # Portability
//!
//! The kernel uses only `length` (`sqrt`), multiply and add in the portable
//! core-`WGSL` subset — no `exp`, `pow` or optional device feature — so it runs
//! unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The meshing contains no transcendental call (the reference restricts itself
//! to `sqrt` via vector length), so `CPU` and `GPU` evaluate the same closed-form
//! geometry. They are **not** bit-exact: a `GPU` may fuse a multiply-add or
//! reassociate the arc-length accumulation the scalar reference leaves separate,
//! perturbing the low mantissa bits by a few `ULP`. The parity test therefore
//! asserts a tolerance (`abs_diff < 1e-4` or `rel_diff < 1e-3`) per component.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard rectangular shell-tube meshing plus `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.

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

/// One strand's inputs to the shell builder, mirroring the `CPU` golden
/// arguments `points`, the bitangent/normal columns of `frames`, and the uniform
/// `half_width` / `half_thickness`.
///
/// The three slices describe the same strand and must share a length; a strand
/// with fewer than two points, or with mismatched slice lengths, yields an empty
/// mesh (matching
/// [`build_shell`](prism_render_architecture::hair::mesh_shell::build_shell)).
#[derive(Clone, Copy)]
pub struct ShellStrandInput<'a> {
    /// Control points along the strand centerline (root → tip).
    pub points: &'a [[f32; 3]],
    /// Per-point rotation-minimizing bitangent (spans the section width).
    pub bitangents: &'a [[f32; 3]],
    /// Per-point reference normal (spans the section thickness).
    pub normals: &'a [[f32; 3]],
    /// Half-extent along the bitangent (uniform along the strand).
    pub half_width: f32,
    /// Half-extent along the normal (uniform along the strand).
    pub half_thickness: f32,
}

/// An indexed triangle-list shell tube for one strand, mirroring the `CPU`
/// [`ShellMesh`](prism_render_architecture::hair::mesh_shell::ShellMesh).
///
/// The per-vertex arrays run parallel with length `4 * point_count` (four
/// rectangle corners per control point); `indices` is a triangle list with
/// `3 * (8 * (point_count - 1) + 4)` entries (eight side triangles per segment
/// plus two triangles for each of the root and tip caps).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GpuShellMesh {
    /// Shell surface positions; four per control point, one per corner.
    pub positions: Vec<[f32; 3]>,
    /// Outward per-vertex normals, parallel to `positions`.
    pub normals: Vec<[f32; 3]>,
    /// Per-vertex UVs: `u` walks the perimeter in quarter steps, `v` the arc.
    pub uvs: Vec<[f32; 2]>,
    /// Triangle-list indices into the vertex arrays.
    pub indices: Vec<u32>,
}

/// Uniform block; `16`-byte `repr(C)` matching `Params` in
/// `shaders/mesh_shell.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    strand_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One strand descriptor uploaded to the kernel. `32`-byte `repr(C)` matching
/// `Strand` in `shaders/mesh_shell.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShellStrand {
    point_offset: u32,
    point_count: u32,
    vertex_offset: u32,
    pad0: u32,
    half_width: f32,
    half_thickness: f32,
    pad1: f32,
    pad2: f32,
}

/// A compiled, reusable shell-meshing pipeline.
pub struct GpuMeshShell {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMeshShell {
    /// Compiles the shell-meshing kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMeshShell {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_mesh_shell"),
            source: ShaderSource::Wgsl(include_str!("../shaders/mesh_shell.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_mesh_shell_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_mesh_shell_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_mesh_shell_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMeshShell {
            module,
            layout,
            pipeline,
        }
    }

    /// Builds a shell mesh for every strand, returning one [`GpuShellMesh`] per
    /// input strand (same length as `strands`).
    ///
    /// The mesh for strand `s` equals
    /// [`build_shell`](prism_render_architecture::hair::mesh_shell::build_shell)
    /// applied to `strands[s]`, to within the fused-multiply-add tolerance
    /// documented on this module. A strand with fewer than two points or with
    /// mismatched attribute lengths yields an empty mesh. A batch that produces
    /// no vertices at all is handled without a dispatch — storage buffers cannot
    /// be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, strands: &[ShellStrandInput]) -> Vec<GpuShellMesh> {
        if strands.is_empty() {
            return Vec::new();
        }

        // Flatten valid strands' interleaved attributes and build descriptors.
        // An invalid strand (too short or mismatched lengths) gets a zero-count
        // descriptor so the kernel skips it and it reconstructs as an empty mesh.
        let mut attrs: Vec<f32> = Vec::new();
        let mut descriptors: Vec<ShellStrand> = Vec::with_capacity(strands.len());
        let mut valid = Vec::with_capacity(strands.len());
        let mut vertex_cursor: u32 = 0;
        for strand in strands {
            let n = strand.points.len();
            let ok = n >= 2 && strand.bitangents.len() == n && strand.normals.len() == n;
            valid.push(ok);
            if !ok {
                descriptors.push(ShellStrand {
                    point_offset: 0,
                    point_count: 0,
                    vertex_offset: 0,
                    pad0: 0,
                    half_width: 0.0,
                    half_thickness: 0.0,
                    pad1: 0.0,
                    pad2: 0.0,
                });
                continue;
            }
            let point_offset = (attrs.len() / 9) as u32;
            let vertex_offset = vertex_cursor;
            for i in 0..n {
                attrs.extend_from_slice(&strand.points[i]);
                attrs.extend_from_slice(&strand.bitangents[i]);
                attrs.extend_from_slice(&strand.normals[i]);
            }
            descriptors.push(ShellStrand {
                point_offset,
                point_count: n as u32,
                vertex_offset,
                pad0: 0,
                half_width: strand.half_width,
                half_thickness: strand.half_thickness,
                pad1: 0.0,
                pad2: 0.0,
            });
            vertex_cursor += (4 * n) as u32;
        }

        let total_vertices = vertex_cursor as usize;
        if total_vertices == 0 {
            // No strand produced geometry: one empty mesh per strand, no dispatch.
            return strands.iter().map(|_| GpuShellMesh::default()).collect();
        }

        let device = ctx.device();
        let params = Params {
            strand_count: strands.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // Eight f32 (position, normal, uv) per output vertex.
        let out_bytes = (total_vertices as u64) * 8 * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_mesh_shell_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let strands_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_mesh_shell_strands"),
            contents: bytemuck::cast_slice(&descriptors),
            usage: BufferUsages::STORAGE,
        });
        let attrs_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_mesh_shell_attrs"),
            contents: bytemuck::cast_slice(&attrs),
            usage: BufferUsages::STORAGE,
        });
        let verts_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_mesh_shell_verts"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let verts_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_mesh_shell_verts_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_mesh_shell_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: strands_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: attrs_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: verts_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_mesh_shell_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_mesh_shell_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (strands.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&verts_buf, 0, &verts_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        verts_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = verts_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let flat = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        verts_stage.unmap();
        debug_assert_eq!(flat.len(), total_vertices * 8);

        // Re-split the flat vertex stream back into per-strand meshes, and
        // rebuild the deterministic index winding the golden uses.
        strands
            .iter()
            .zip(descriptors.iter())
            .zip(valid.iter())
            .map(|((_, desc), &ok)| {
                if !ok {
                    return GpuShellMesh::default();
                }
                let n = desc.point_count as usize;
                let vbase = desc.vertex_offset as usize;
                let vcount = 4 * n;
                let mut positions = Vec::with_capacity(vcount);
                let mut normals = Vec::with_capacity(vcount);
                let mut uvs = Vec::with_capacity(vcount);
                for local in 0..vcount {
                    let b = (vbase + local) * 8;
                    positions.push([flat[b], flat[b + 1], flat[b + 2]]);
                    normals.push([flat[b + 3], flat[b + 4], flat[b + 5]]);
                    uvs.push([flat[b + 6], flat[b + 7]]);
                }
                GpuShellMesh {
                    positions,
                    normals,
                    uvs,
                    indices: shell_indices(n),
                }
            })
            .collect()
    }
}

/// Rebuilds the deterministic triangle-list winding the golden emits for an
/// `n`-ring shell: eight side triangles per segment plus two triangles for each
/// of the root and tip caps. Pure integer arithmetic, carrying no `GPU` error.
fn shell_indices(n: usize) -> Vec<u32> {
    let mut indices = Vec::with_capacity(3 * (8 * (n - 1) + 4));
    for r in 0..n - 1 {
        let base = 4 * r;
        let next = 4 * (r + 1);
        for s in 0..4usize {
            let sn = (s + 1) % 4;
            let a = (base + s) as u32;
            let b = (base + sn) as u32;
            let c = (next + sn) as u32;
            let d = (next + s) as u32;
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    // Root cap (ring 0), then tip cap (last ring) with reversed winding.
    indices.extend_from_slice(&[0, 2, 1, 0, 3, 2]);
    let tip = (4 * (n - 1)) as u32;
    indices.extend_from_slice(&[tip, tip + 1, tip + 2, tip, tip + 2, tip + 3]);
    indices
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
