//! `wgpu` compute twin of Prism's strand-to-ribbon meshing
//! ([`build_ribbon`](prism_render_architecture::hair::ribbon::build_ribbon)).
//!
//! When a render strand falls below a pixel of screen coverage, the LOD ladder
//! swaps its sub-pixel software raster for a flat camera-card / ribbon proxy: a
//! quad strip that follows the strand centerline and widens along the strand's
//! own rotation-minimizing bitangent (from the [`crate::frames`] twin), not
//! toward the camera, so the mesh is view-independent and can be built once and
//! cached. Production hair engines (`UE5` Groom, AMD `TressFX`) generate exactly
//! this proxy for distant grooms. The `CPU` golden for that meshing is
//! [`build_ribbon`](prism_render_architecture::hair::ribbon::build_ribbon); this
//! crate is the on-device twin that walks the same meshing, one thread per
//! strand, so a passing real-device parity test is direct evidence the ported
//! kernel computes the same ribbon vertices as the reference — not merely that
//! its shader compiles.
//!
//! # What the kernel evaluates
//!
//! [`GpuRibbon::eval`] takes a batch of strands (each its control points, the
//! per-point tangent and bitangent from the frame transport, and a per-point
//! radius) and returns one [`GpuRibbonMesh`] per strand. Each mesh carries two
//! edge vertices per control point (left then right), offset `±radius` along the
//! bitangent, with the strand tangent carried through for anisotropic shading
//! and a UV whose `u` is `0`/`1` across the ribbon and whose `v` runs `0` at the
//! root to `1` at the tip along arc length. The triangle-list indices are
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
//! Provenance: standard view-independent ribbon/card meshing plus `wgpu` compute
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

/// One strand's inputs to the ribbon builder, mirroring the `CPU` golden
/// arguments `points`, the tangent/bitangent columns of `frames`, and `radii`.
///
/// The four slices describe the same strand and must share a length; a strand
/// with fewer than two points, or with mismatched slice lengths, yields an empty
/// mesh (matching [`build_ribbon`](prism_render_architecture::hair::ribbon::build_ribbon)).
#[derive(Clone, Copy)]
pub struct RibbonStrandInput<'a> {
    /// Control points along the strand centerline (root → tip).
    pub points: &'a [[f32; 3]],
    /// Per-point strand tangent (from the frame transport).
    pub tangents: &'a [[f32; 3]],
    /// Per-point rotation-minimizing bitangent (the widening direction).
    pub bitangents: &'a [[f32; 3]],
    /// Per-point ribbon radius (half-width).
    pub radii: &'a [f32],
}

/// An indexed triangle-list ribbon for one strand, mirroring the `CPU`
/// [`RibbonMesh`](prism_render_architecture::hair::ribbon::RibbonMesh).
///
/// The per-vertex arrays run parallel with length `2 * point_count` (left then
/// right edge per control point); `indices` is a triangle list with
/// `6 * (point_count - 1)` entries.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GpuRibbonMesh {
    /// Ribbon edge positions; two per control point (left, then right).
    pub positions: Vec<[f32; 3]>,
    /// Strand tangent per vertex, parallel to `positions`.
    pub tangents: Vec<[f32; 3]>,
    /// Per-vertex UVs: `u` is `0` (left) or `1` (right); `v` runs root → tip.
    pub uvs: Vec<[f32; 2]>,
    /// Triangle-list indices into the per-vertex arrays.
    pub indices: Vec<u32>,
}

/// Uniform parameters for one ribbon dispatch. Layout matches `Params` in
/// `shaders/ribbon.wesl`: the strand count then three pad words for the
/// `16`-byte uniform alignment.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    strand_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One strand descriptor uploaded to the kernel. `16`-byte `repr(C)` matching
/// `Strand` in `shaders/ribbon.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RibbonStrand {
    point_offset: u32,
    point_count: u32,
    vertex_offset: u32,
    pad0: u32,
}

/// A compiled, reusable ribbon-meshing pipeline.
pub struct GpuRibbon {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRibbon {
    /// Compiles the ribbon-meshing kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRibbon {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_ribbon"),
            source: ShaderSource::Wgsl(include_str!("../shaders/ribbon.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_ribbon_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_ribbon_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_ribbon_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRibbon {
            module,
            layout,
            pipeline,
        }
    }

    /// Builds a ribbon mesh for every strand, returning one [`GpuRibbonMesh`]
    /// per input strand (same length as `strands`).
    ///
    /// The mesh for strand `s` equals
    /// [`build_ribbon`](prism_render_architecture::hair::ribbon::build_ribbon)
    /// applied to `strands[s]`, to within the fused-multiply-add tolerance
    /// documented on this module. A strand with fewer than two points or with
    /// mismatched attribute lengths yields an empty mesh. A batch that produces
    /// no vertices at all is handled without a dispatch — storage buffers cannot
    /// be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, strands: &[RibbonStrandInput]) -> Vec<GpuRibbonMesh> {
        if strands.is_empty() {
            return Vec::new();
        }

        // Flatten valid strands' interleaved attributes and build descriptors.
        // An invalid strand (too short or mismatched lengths) gets a zero-count
        // descriptor so the kernel skips it and it reconstructs as an empty mesh.
        let mut attrs: Vec<f32> = Vec::new();
        let mut descriptors: Vec<RibbonStrand> = Vec::with_capacity(strands.len());
        let mut valid = Vec::with_capacity(strands.len());
        let mut vertex_cursor: u32 = 0;
        for strand in strands {
            let n = strand.points.len();
            let ok = n >= 2
                && strand.tangents.len() == n
                && strand.bitangents.len() == n
                && strand.radii.len() == n;
            valid.push(ok);
            if !ok {
                descriptors.push(RibbonStrand {
                    point_offset: 0,
                    point_count: 0,
                    vertex_offset: 0,
                    pad0: 0,
                });
                continue;
            }
            let point_offset = (attrs.len() / 10) as u32;
            let vertex_offset = vertex_cursor;
            for i in 0..n {
                attrs.extend_from_slice(&strand.points[i]);
                attrs.extend_from_slice(&strand.tangents[i]);
                attrs.extend_from_slice(&strand.bitangents[i]);
                attrs.push(strand.radii[i]);
            }
            descriptors.push(RibbonStrand {
                point_offset,
                point_count: n as u32,
                vertex_offset,
                pad0: 0,
            });
            vertex_cursor += (2 * n) as u32;
        }

        let total_vertices = vertex_cursor as usize;
        if total_vertices == 0 {
            // No strand produced geometry: one empty mesh per strand, no dispatch.
            return strands.iter().map(|_| GpuRibbonMesh::default()).collect();
        }

        let device = ctx.device();
        let params = Params {
            strand_count: strands.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // Eight f32 (position, tangent, uv) per output vertex.
        let out_bytes = (total_vertices as u64) * 8 * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_ribbon_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let strands_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_ribbon_strands"),
            contents: bytemuck::cast_slice(&descriptors),
            usage: BufferUsages::STORAGE,
        });
        let attrs_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_ribbon_attrs"),
            contents: bytemuck::cast_slice(&attrs),
            usage: BufferUsages::STORAGE,
        });
        let verts_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_ribbon_verts"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let verts_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_ribbon_verts_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_ribbon_bind_group"),
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
            label: Some("prism_hair_ribbon_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_ribbon_pass"),
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
                    return GpuRibbonMesh::default();
                }
                let n = desc.point_count as usize;
                let vbase = desc.vertex_offset as usize;
                let vcount = 2 * n;
                let mut positions = Vec::with_capacity(vcount);
                let mut tangents = Vec::with_capacity(vcount);
                let mut uvs = Vec::with_capacity(vcount);
                for local in 0..vcount {
                    let b = (vbase + local) * 8;
                    positions.push([flat[b], flat[b + 1], flat[b + 2]]);
                    tangents.push([flat[b + 3], flat[b + 4], flat[b + 5]]);
                    uvs.push([flat[b + 6], flat[b + 7]]);
                }
                let mut indices = Vec::with_capacity(6 * (n - 1));
                for i in 0..n - 1 {
                    let l0 = (2 * i) as u32;
                    indices.extend_from_slice(&[l0, l0 + 1, l0 + 2, l0 + 1, l0 + 3, l0 + 2]);
                }
                GpuRibbonMesh {
                    positions,
                    tangents,
                    uvs,
                    indices,
                }
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
