//! `wgpu` compute twin of the area-weighted vertex-normal accumulation inside
//! the cloth multi-layer coupling contract
//! ([`accumulate_vertex_normals`](prism_render_architecture::cloth::layers::accumulate_vertex_normals),
//! cloth design §6.7).
//!
//! The `CPU` golden
//! [`accumulate_vertex_normals`](prism_render_architecture::cloth::layers::accumulate_vertex_normals)
//! turns a vertex array and a triangle index list into one outward normal per
//! vertex. Each face contributes its unnormalized cross product
//! `(p1 - p0) x (p2 - p0)` — whose magnitude is twice the triangle area — to
//! each of its three vertices, so larger faces weigh more; every per-vertex sum
//! is then normalized, and a vertex touched by no face (or only degenerate
//! faces) is left at zero. Out-of-range triangle indices are skipped and never
//! panic.
//!
//! [`GpuClothVertexNormals`] is the on-device twin of that numeric core. One
//! thread owns one output vertex: it replays, in the golden's exact
//! triangle-ascending order, the un-normalized face crosses of every triangle
//! corner incident to that vertex, sums them, and applies the same guarded
//! normalization. A passing real-device parity test is therefore direct
//! evidence the ported kernel reproduces the same area-weighted normal field,
//! not merely that the shader compiles.
//!
//! # What is twinned
//!
//! The per-vertex numeric core: the un-normalized face cross
//! `(p1 - p0) x (p2 - p0)` accumulated across every incident `(triangle,
//! corner)` in triangle-ascending order, then the guarded normalization
//! `g / sqrt(dot(g, g))` with a zero-length vector left at the zero vector. A
//! vertex that no in-range triangle touches sums nothing and stays zero,
//! matching the golden's `normalize_or_zero` of an untouched slot.
//!
//! # What stays on the host
//!
//! The variable-length container work: resizing the output to the vertex count,
//! walking the triangle list, and — crucially for a race-free device port —
//! building the per-vertex incidence (a compressed-sparse-row map from each
//! vertex to the triangles that reference it, in triangle-ascending order).
//! The host assembles that map in the golden's iteration order so the device
//! sum is a deterministic replay, never a parallel scatter. The host also owns
//! the out-of-range skip (an out-of-range triangle is simply never added to the
//! incidence map) and the empty-mesh short-circuit (a zero-vertex mesh returns
//! an empty result with no dispatch, since a storage buffer cannot be
//! zero-sized).
//!
//! # Correctness model
//!
//! Because the host replays the exact triangle-ascending summation order, the
//! device accumulates the identical sequence of face crosses the golden does,
//! so the pre-normalization sum differs only by whatever fused multiply-add a
//! device may apply inside a single cross product. The normalized components
//! therefore agree with the reference to within a few units in the last place,
//! which the parity test pins with `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//! Degenerate and isolated vertices are an exact zero on both sides.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `+ - * /`,
//! unsigned index arithmetic and magnitude comparisons — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `round`: the
//! zero-length test is a `len2 > 0.0` comparison, never an `f32` equality. No
//! optional device feature is required, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::cloth::layers`；无第三方引擎源码或衍生代码。
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

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` area-weighted vertex-normal kernel, embedded inline
/// so the twin ships as a single source file. The single entry point
/// `accumulate` mirrors the `CPU` golden
/// [`accumulate_vertex_normals`](prism_render_architecture::cloth::layers::accumulate_vertex_normals)
/// per-vertex closed form; see the module documentation for the algorithm.
const CLOTH_VERTEX_NORMALS_WGSL: &str = r#"
// Area-weighted vertex-normal twin: one thread owns one output vertex. It
// replays, in triangle-ascending order, the un-normalized face crosses of every
// triangle corner incident to the vertex (via a host-built compressed-sparse-row
// incidence map), sums them, and applies the guarded normalization
// g / sqrt(dot(g, g)) with a zero-length vector left at zero — mirroring the CPU
// golden `cloth::layers::accumulate_vertex_normals` with only sqrt and + - * /.
// The variable-length triangle walk, the out-of-range skip and the output resize
// stay on the host; the device only replays a fixed incidence range per vertex.
//
// Provenance: 孪生自本仓 prism_render_architecture::cloth::layers；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of output vertices; threads past this short-circuit.
    vertex_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// Flattened vertex positions, three f32 per vertex (x, y, z contiguous).
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
// Flattened triangle corner indices, three u32 per triangle.
@group(0) @binding(2) var<storage, read> tri_indices: array<u32>;
// Per-vertex incidence offsets, length vertex_count + 1 (prefix sums).
@group(0) @binding(3) var<storage, read> csr_offsets: array<u32>;
// Triangle index of each incidence, grouped by vertex in triangle-ascending
// order.
@group(0) @binding(4) var<storage, read> csr_tris: array<u32>;
// Flattened output normals, three f32 per vertex.
@group(0) @binding(5) var<storage, read_write> out_normals: array<f32>;

@compute @workgroup_size(64)
fn accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = gid.x;
    if (v >= params.vertex_count) {
        return;
    }

    let start = csr_offsets[v];
    let end = csr_offsets[v + 1u];

    // Accumulate every incident face's un-normalized cross in the golden's
    // triangle-ascending order; no atomics, since each vertex owns its own sum.
    var sx: f32 = 0.0;
    var sy: f32 = 0.0;
    var sz: f32 = 0.0;
    for (var k: u32 = start; k < end; k = k + 1u) {
        let t = csr_tris[k];
        let base = t * 3u;
        let i0 = tri_indices[base];
        let i1 = tri_indices[base + 1u];
        let i2 = tri_indices[base + 2u];

        let p0x = positions[i0 * 3u];
        let p0y = positions[i0 * 3u + 1u];
        let p0z = positions[i0 * 3u + 2u];
        let p1x = positions[i1 * 3u];
        let p1y = positions[i1 * 3u + 1u];
        let p1z = positions[i1 * 3u + 2u];
        let p2x = positions[i2 * 3u];
        let p2y = positions[i2 * 3u + 1u];
        let p2z = positions[i2 * 3u + 2u];

        // Edge vectors a = p1 - p0, b = p2 - p0.
        let ax = p1x - p0x;
        let ay = p1y - p0y;
        let az = p1z - p0z;
        let bx = p2x - p0x;
        let by = p2y - p0y;
        let bz = p2z - p0z;

        // Cross product a x b, component order matching the reference.
        let fx = ay * bz - az * by;
        let fy = az * bx - ax * bz;
        let fz = ax * by - ay * bx;

        sx = sx + fx;
        sy = sy + fy;
        sz = sz + fz;
    }

    // Guarded normalization: a zero-length sum (untouched or only-degenerate
    // vertex) stays at the zero vector, matching the reference's
    // normalize-or-zero. The test is a magnitude comparison, never an equality.
    let len2 = sx * sx + sy * sy + sz * sz;
    var ox: f32 = 0.0;
    var oy: f32 = 0.0;
    var oz: f32 = 0.0;
    if (len2 > 0.0) {
        let inv = 1.0 / sqrt(len2);
        ox = sx * inv;
        oy = sy * inv;
        oz = sz * inv;
    }

    out_normals[v * 3u] = ox;
    out_normals[v * 3u + 1u] = oy;
    out_normals[v * 3u + 2u] = oz;
}
"#;

/// Uniform parameters for one dispatch: the vertex count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`CLOTH_VERTEX_NORMALS_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of output vertices in the input and output buffers.
    vertex_count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One cloth mesh to accumulate vertex normals for: the vertex `positions` and
/// the triangle index list `triangles`, mirroring the arguments of the `CPU`
/// golden
/// [`accumulate_vertex_normals`](prism_render_architecture::cloth::layers::accumulate_vertex_normals).
///
/// Each entry of `positions` is one vertex `[x, y, z]`; each entry of
/// `triangles` is one face's three vertex indices `[i0, i1, i2]`. An index at
/// or past `positions.len()` makes its whole triangle a no-op, exactly as the
/// reference skips an out-of-range face.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::cloth::layers`；无第三方引擎源码或衍生代码。
#[derive(Clone, Debug, PartialEq)]
pub struct ClothVertexNormalsQuery {
    /// The vertex positions, one `[x, y, z]` per vertex.
    pub positions: Vec<[f32; 3]>,
    /// The triangle index list, one `[i0, i1, i2]` per face.
    pub triangles: Vec<[u32; 3]>,
}

/// The accumulated outward normals, one `[x, y, z]` per vertex and parallel to
/// the query's `positions`, mirroring the `out` written by the `CPU` golden
/// [`accumulate_vertex_normals`](prism_render_architecture::cloth::layers::accumulate_vertex_normals).
///
/// A vertex touched by no in-range face (or by only degenerate faces) is the
/// zero vector; every other vertex is a unit-length area-weighted normal.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::cloth::layers`；无第三方引擎源码或衍生代码。
#[derive(Clone, Debug, PartialEq)]
pub struct ClothVertexNormalsResult {
    /// The per-vertex normals, parallel to the query's `positions`.
    pub normals: Vec<[f32; 3]>,
}

/// The host-built, device-friendly incidence map: a per-vertex compressed
/// sparse row listing the triangles that reference each vertex, in the golden's
/// triangle-ascending order.
struct Incidence {
    /// Prefix-sum offsets, length `vertex_count + 1`.
    offsets: Vec<u32>,
    /// Triangle index of each incidence, grouped by vertex.
    tris: Vec<u32>,
}

/// Builds the compressed-sparse-row incidence map in the golden's iteration
/// order: walk triangles ascending, and for each in-range triangle append its
/// index once per corner to that corner vertex's list. An out-of-range triangle
/// is never appended, so it contributes nothing — exactly the reference skip.
fn build_incidence(query: &ClothVertexNormalsQuery) -> Incidence {
    let vertex_count = query.positions.len();
    let mut counts = vec![0u32; vertex_count];
    for tri in &query.triangles {
        let in_range = tri.iter().all(|&index| (index as usize) < vertex_count);
        if !in_range {
            continue;
        }
        for &index in tri {
            counts[index as usize] += 1;
        }
    }

    let mut offsets = vec![0u32; vertex_count + 1];
    let mut running: u32 = 0;
    for (slot, &count) in offsets.iter_mut().zip(counts.iter()).take(vertex_count) {
        *slot = running;
        running += count;
    }
    offsets[vertex_count] = running;

    let total = running as usize;
    let mut tris = vec![0u32; total];
    let mut cursor: Vec<u32> = offsets[..vertex_count].to_vec();
    for (triangle_index, tri) in query.triangles.iter().enumerate() {
        let in_range = tri.iter().all(|&index| (index as usize) < vertex_count);
        if !in_range {
            continue;
        }
        for &index in tri {
            let slot = cursor[index as usize] as usize;
            tris[slot] = triangle_index as u32;
            cursor[index as usize] += 1;
        }
    }

    Incidence { offsets, tris }
}

/// Pads a `u32` storage payload to at least one element so the device never
/// binds a zero-sized storage buffer.
fn pad_u32(mut data: Vec<u32>) -> Vec<u32> {
    if data.is_empty() {
        data.push(0);
    }
    data
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

/// A compiled, reusable area-weighted vertex-normal compute pipeline, twinning
/// the numeric core of the `CPU` golden
/// [`accumulate_vertex_normals`](prism_render_architecture::cloth::layers::accumulate_vertex_normals).
pub struct GpuClothVertexNormals {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuClothVertexNormals {
    /// Compiles the vertex-normal kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothVertexNormals {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_cloth_vertex_normals"),
            source: ShaderSource::Wgsl(CLOTH_VERTEX_NORMALS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_cloth_vertex_normals_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: true }),
                buffer_entry(5, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_cloth_vertex_normals_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_cloth_vertex_normals_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("accumulate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuClothVertexNormals {
            module,
            layout,
            pipeline,
        }
    }

    /// Accumulates one area-weighted outward normal per vertex of `query` and
    /// returns them parallel to the query's `positions`.
    ///
    /// Each vertex normal equals the reference's to within the tolerance
    /// documented on this module; a vertex touched by no in-range face is the
    /// zero vector on both sides. A zero-vertex mesh returns an empty result
    /// with no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        query: &ClothVertexNormalsQuery,
    ) -> ClothVertexNormalsResult {
        let vertex_count = query.positions.len();
        if vertex_count == 0 {
            return ClothVertexNormalsResult {
                normals: Vec::new(),
            };
        }
        let device = ctx.device();

        let params = GpuParams {
            vertex_count: vertex_count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cloth_vertex_normals_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let positions_flat: Vec<f32> = query
            .positions
            .iter()
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect();
        let positions_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cloth_vertex_normals_positions"),
            contents: bytemuck::cast_slice(&positions_flat),
            usage: BufferUsages::STORAGE,
        });

        let tri_flat: Vec<u32> = pad_u32(
            query
                .triangles
                .iter()
                .flat_map(|t| [t[0], t[1], t[2]])
                .collect(),
        );
        let tri_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cloth_vertex_normals_tris"),
            contents: bytemuck::cast_slice(&tri_flat),
            usage: BufferUsages::STORAGE,
        });

        let incidence = build_incidence(query);
        let offsets_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cloth_vertex_normals_offsets"),
            contents: bytemuck::cast_slice(&incidence.offsets),
            usage: BufferUsages::STORAGE,
        });
        let csr_tris = pad_u32(incidence.tris);
        let csr_tris_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cloth_vertex_normals_csr_tris"),
            contents: bytemuck::cast_slice(&csr_tris),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (vertex_count * 3 * size_of::<f32>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cloth_vertex_normals_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_cloth_vertex_normals_bind_group"),
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
                    resource: tri_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: offsets_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: csr_tris_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cloth_vertex_normals_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_cloth_vertex_normals_encoder"),
        });
        {
            let groups = (vertex_count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_cloth_vertex_normals_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per output vertex, flattened to a 1-D dispatch.
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
        let raw = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        stage.unmap();

        let normals = raw.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        ClothVertexNormalsResult { normals }
    }
}
