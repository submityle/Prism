//! `wgpu` compute twin of the cloth pressure solver's closed-mesh volume
//! accumulator
//! ([`mesh_volume`](prism_render_architecture::cloth::pressure::mesh_volume)).
//!
//! A pressure (volume) constraint on a cloth shell needs the signed enclosed
//! volume of the triangulated surface. By the divergence theorem that volume is
//! a sum over triangles of the scalar triple product of the triangle's three
//! vertices, scaled by one sixth:
//!
//! ```text
//! V = (1/6) · Σ_tri  p0 · (p1 × p2)
//! ```
//!
//! The `CPU` golden
//! [`mesh_volume`](prism_render_architecture::cloth::pressure::mesh_volume)
//! delegates to the authoritative accumulator in `prism_physics_core`; a
//! triangle whose vertex indices fall outside the position array is skipped
//! rather than panicking, and an empty mesh yields `0`. This twin reproduces
//! that reduction on device.
//!
//! # What is twinned
//!
//! A single `GPU` invocation walks the triangle list in input order, and for
//! each triangle whose three indices are all in range accumulates
//! `p0 · (p1 × p2)` into a running sum, finally multiplying by `1/6`. The cross
//! and dot are spelled out component-wise in the same order as the `CPU`
//! reference so the running sum stays as close to bit-identical as the hardware
//! allows. Out-of-range triangles are skipped exactly as the golden's
//! `fetch_triangle_positions` returns `None`.
//!
//! # What stays on the host
//!
//! The host owns the flattening of the vertex and triangle arrays into `std430`
//! storage buffers, the triangle/vertex counts, and the empty-mesh
//! short-circuit (a storage buffer cannot be zero-sized, and an empty mesh is
//! defined to have volume `0`). Variable-length container work has no
//! fixed-width device analogue and is not twinned.
//!
//! # Correctness model
//!
//! The volume threads through many multiplies, subtracts and a long additive
//! reduction, so the `CPU` and `GPU` are not bit-exact across the sum; the
//! result is asserted within a tolerance (`abs_diff <= 2e-4` or
//! `rel_diff <= 2e-3`), slightly wider than the per-element floor to absorb the
//! reduction's accumulated rounding. The serial single-invocation accumulation
//! preserves the reference's triangle order so the two agree tightly in
//! practice.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — component-wise
//! `+ - *`, a single `/ 6` via a constant reciprocal, a bounded loop over at
//! most [`MAX_TRIS`] triangles, and unsigned index arithmetic — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `tan`, `sqrt`, no inverse trigonometry, no
//! `round`, and no `u64`/`u16`/`i64`/`f64`. No optional device feature is
//! required, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::cloth::pressure`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

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

/// Threads per workgroup. The volume accumulator is a serial additive reduction
/// that must preserve the reference's triangle order, so it runs as a single
/// invocation (`1`) rather than one thread per element.
const WORKGROUP_SIZE: u32 = 1;

/// Soft upper bound on the number of vertices a single mesh may submit. The
/// kernel indexes a flattened scalar position buffer and has no fixed-width
/// vertex array, so this is a batching recommendation rather than a structural
/// limit; callers with larger meshes should chunk their input.
pub const MAX_VERTS: usize = 4096;

/// Soft upper bound on the number of triangles a single mesh may submit. The
/// device loop is bounded by the submitted triangle count, which is clamped to
/// this ceiling on encode so the reduction terminates.
pub const MAX_TRIS: usize = 8192;

/// The portable core-`WGSL` mesh-volume kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`mesh_volume`](prism_render_architecture::cloth::pressure::mesh_volume); see
/// the module documentation for the algorithm.
const CLOTH_MESH_VOLUME_WGSL: &str = r#"
// Cloth mesh-volume twin: a single invocation accumulates the divergence-theorem
// signed volume (1/6) * sum of p0 . (p1 x p2) over the triangle list, mirroring
// the CPU golden `cloth::pressure` `mesh_volume` with only component-wise
// + - * and one constant reciprocal. Out-of-range triangles are skipped exactly
// as the reference's index fetch returns None. The variable-length flattening
// stays on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::cloth::pressure；无第三方引擎
// 源码或衍生代码。

const INV_SIX: f32 = 1.0 / 6.0;

struct Params {
    // Number of vertices in the flattened position buffer.
    vert_count: u32,
    // Number of triangles in the flattened index buffer.
    tri_count: u32,
    pad0: u32,
    pad1: u32,
}

struct Result {
    volume: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Flattened vertex positions: three f32 (x, y, z) per vertex.
@group(0) @binding(1) var<storage, read> positions: array<f32>;
// Flattened triangle indices: three u32 per triangle.
@group(0) @binding(2) var<storage, read> triangles: array<u32>;
@group(0) @binding(3) var<storage, read_write> results: array<Result>;

// Loads vertex `index` from the flattened position buffer.
fn load_position(index: u32) -> vec3<f32> {
    let base = index * 3u;
    return vec3<f32>(positions[base], positions[base + 1u], positions[base + 2u]);
}

// Scalar triple product p0 . (p1 x p2), spelled out component-wise in the same
// order as the CPU reference so the running sum stays close to bit-identical.
fn triple_product(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>) -> f32 {
    let cx = p1.y * p2.z - p1.z * p2.y;
    let cy = p1.z * p2.x - p1.x * p2.z;
    let cz = p1.x * p2.y - p1.y * p2.x;
    return p0.x * cx + p0.y * cy + p0.z * cz;
}

@compute @workgroup_size(1)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    // A single invocation performs the whole ordered reduction; extra threads
    // (if any workgroup padding exists) do nothing.
    if (gid.x > 0u) {
        return;
    }

    var sum: f32 = 0.0;
    for (var t: u32 = 0u; t < params.tri_count; t = t + 1u) {
        let base = t * 3u;
        let i0 = triangles[base];
        let i1 = triangles[base + 1u];
        let i2 = triangles[base + 2u];
        // Skip any triangle with an out-of-range vertex index, mirroring the
        // reference's `positions.get(i)?` short-circuit.
        if (i0 >= params.vert_count) {
            continue;
        }
        if (i1 >= params.vert_count) {
            continue;
        }
        if (i2 >= params.vert_count) {
            continue;
        }
        let p0 = load_position(i0);
        let p1 = load_position(i1);
        let p2 = load_position(i2);
        sum = sum + triple_product(p0, p1, p2);
    }

    var out: Result;
    out.volume = sum * INV_SIX;
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;
    results[0] = out;
}
"#;

/// Uniform parameters for one dispatch: the vertex and triangle counts plus two
/// pad words to fill a `16`-byte uniform struct matching `Params` in
/// [`CLOTH_MESH_VOLUME_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of vertices in the flattened position buffer.
    vert_count: u32,
    /// Number of triangles in the flattened index buffer.
    tri_count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// `repr(C)` `std430` layout of the single result slot, matching the `WGSL`
/// `Result` struct: the signed volume plus three pad words to a `16`-byte
/// stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Signed enclosed volume of the mesh.
    volume: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One closed-mesh volume query: a vertex array and a triangle index list, laid
/// out exactly like the slices the `CPU` golden
/// [`mesh_volume`](prism_render_architecture::cloth::pressure::mesh_volume)
/// consumes.
///
/// `positions[i]` is vertex `i` as `[x, y, z]`; `triangles[t]` indexes it with
/// outward winding. A triangle with any index at or beyond `positions.len()` is
/// skipped (mirroring the reference), and an empty mesh has volume `0`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ClothMeshVolumeQuery {
    /// Vertex positions as `[x, y, z]` triples.
    pub positions: Vec<[f32; 3]>,
    /// Triangle vertex indices with outward winding.
    pub triangles: Vec<[u32; 3]>,
}

/// The resolved signed enclosed volume of one mesh.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClothMeshVolumeResult {
    /// Signed enclosed volume: `(1/6) · Σ p0 · (p1 × p2)` over in-range
    /// triangles.
    pub volume: f32,
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

/// A compiled, reusable mesh-volume compute pipeline, twinning the closed-mesh
/// volume accumulator of the `CPU` golden
/// [`mesh_volume`](prism_render_architecture::cloth::pressure::mesh_volume).
pub struct GpuClothMeshVolume {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuClothMeshVolume {
    /// Compiles the mesh-volume kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothMeshVolume {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_cloth_mesh_volume"),
            source: ShaderSource::Wgsl(CLOTH_MESH_VOLUME_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_cloth_mesh_volume_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_cloth_mesh_volume_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_cloth_mesh_volume_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuClothMeshVolume {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves the signed enclosed volume of one `query`.
    ///
    /// The result matches the reference within the tolerance documented on this
    /// module. An empty mesh (no vertices or no triangles) short-circuits to a
    /// volume of `0` with no dispatch issued, since a storage buffer cannot be
    /// zero-sized and an empty mesh is defined to enclose no volume.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        query: &ClothMeshVolumeQuery,
    ) -> ClothMeshVolumeResult {
        if query.positions.is_empty() || query.triangles.is_empty() {
            return ClothMeshVolumeResult { volume: 0.0 };
        }
        let device = ctx.device();

        let vert_count = query.positions.len().min(MAX_VERTS);
        let tri_count = query.triangles.len().min(MAX_TRIS);

        // Flatten the vertex and triangle arrays into scalar std430 buffers,
        // avoiding the vec3 alignment trap entirely.
        let mut flat_positions: Vec<f32> = Vec::with_capacity(vert_count * 3);
        for p in &query.positions[..vert_count] {
            flat_positions.extend_from_slice(p);
        }
        let mut flat_triangles: Vec<u32> = Vec::with_capacity(tri_count * 3);
        for tri in &query.triangles[..tri_count] {
            flat_triangles.extend_from_slice(tri);
        }

        let params = GpuParams {
            vert_count: vert_count as u32,
            tri_count: tri_count as u32,
            pad0: 0,
            pad1: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cloth_mesh_volume_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let positions_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cloth_mesh_volume_positions"),
            contents: bytemuck::cast_slice(&flat_positions),
            usage: BufferUsages::STORAGE,
        });
        let triangles_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cloth_mesh_volume_triangles"),
            contents: bytemuck::cast_slice(&flat_triangles),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = size_of::<GpuResult>() as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cloth_mesh_volume_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_cloth_mesh_volume_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: positions_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: triangles_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cloth_mesh_volume_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_cloth_mesh_volume_encoder"),
        });
        {
            // The reduction is serial, so a single workgroup of a single thread
            // performs the whole ordered sum.
            let groups = 1u32.div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_cloth_mesh_volume_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        stage.unmap();

        ClothMeshVolumeResult {
            volume: raw[0].volume,
        }
    }
}
