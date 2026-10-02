//! `wgpu` compute twin of the area-weighted mesh triangle selector — the
//! prefix-sum `CDF` picker extracted from the `CPU` golden mesh-emission
//! sampler
//! ([`mesh_emission`](prism_render_architecture::particle::mesh_emission),
//! design §8).
//!
//! The golden [`Mesh::select_triangle`](prism_render_architecture::particle::mesh_emission::Mesh::select_triangle)
//! turns a unit draw `u` into a triangle index drawn in proportion to triangle
//! area: it scales `u` by the total surface area, binary-searches the area
//! prefix-sum `CDF` for the containing interval, and walks back over any
//! trailing zero-area (degenerate) triangle. This twin reproduces that pick for
//! a whole batch of draws at once, one `GPU` thread per draw.
//!
//! # Host and device split
//!
//! The area `CDF` and the per-triangle areas are built once on the host from
//! the mesh's public surface (`triangle_area`, `total_area`), mirroring the
//! golden private `rebuild_cdf` prefix sum operation for operation, so the
//! uploaded `cdf` / `triangle_area` / `total_area` are the exact values the
//! golden selector reads. The kernel does no geometry: it only consumes those
//! scalars plus the batch of draws.
//!
//! # Algorithm
//!
//! One thread owns one draw `u`. It short-circuits an empty or fully degenerate
//! mesh to index `0`, scales `clamp(u, 0, 1)` by `total_area`, runs a
//! statically bounded binary search that returns the `partition_point` of
//! `cdf[i] <= needle` (the count of prefix entries at or below the scaled
//! draw), clamps that to the last triangle, then walks the index back over any
//! trailing degenerate triangle. The result is an integer index, compared bit
//! exactly against the golden selector.
//!
//! # Degenerate inputs
//!
//! An empty draw batch returns an empty result with no dispatch. A mesh with no
//! triangles (`count == 0`) or whose `total_area` is below `EPS` yields index
//! `0` for every draw, the exact guard the reference takes. A trailing
//! zero-area triangle is never selected; the back-walk lands on the last
//! positive-area triangle, matching the reference.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — integer index
//! arithmetic plus `clamp`, `min` and `<=` on `f32`, no transcendental call and
//! no optional device feature — so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. Both loops are statically bounded: the binary search runs at most
//! `32` iterations (enough for any `u32` count) and the degenerate back-walk at
//! most `count` iterations.
//!
//! # Correctness model
//!
//! The selector is pure integer control flow over host-supplied scalars; the
//! only `f32` work is the `clamp`, the scale and the `<=` comparisons, which the
//! host feeds with the identical `cdf` / `total_area` bits. The pick is
//! therefore bit-exact and the parity test asserts integer equality, not a
//! tolerance.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: `wgpu` compute transcription of the golden
//! `Mesh::select_triangle`; no Unreal Engine source or derived code.

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

/// One-dimensional dispatch width; one invocation per draw.
const WORKGROUP_SIZE: u32 = 64;

/// The area-weighted triangle selector kernel.
///
/// `Params` carries the triangle count, the draw-batch length and the total
/// surface area (`std430` layout, four `u32`-width words, one of them padding).
/// The single entry point `select_triangles` short-circuits a degenerate mesh,
/// binary-searches the `CDF`, then walks back over any trailing zero-area
/// triangle.
///
/// Provenance: `WGSL` transcription of the golden `Mesh::select_triangle`; no
/// Unreal Engine source or derived code.
const MESH_CDF_SAMPLE_WGSL: &str = r#"
struct Params {
    count: u32,
    draw_count: u32,
    total_area: f32,
    pad0: u32,
}

const EPS: f32 = 1e-6;

@group(0) @binding(0) var<uniform> params: Params;
// The area prefix-sum CDF, one f32 per triangle: cdf[i] = sum(area[0..=i]).
@group(0) @binding(1) var<storage, read> cdf: array<f32>;
// The per-triangle surface area, one f32 per triangle.
@group(0) @binding(2) var<storage, read> tri_area: array<f32>;
// The batch of unit draws, one f32 per query thread.
@group(0) @binding(3) var<storage, read> draws: array<f32>;
// The selected triangle indices, one u32 per draw.
@group(0) @binding(4) var<storage, read_write> out_idx: array<u32>;

@compute @workgroup_size(64)
fn select_triangles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.draw_count) {
        return;
    }
    let count = params.count;
    // Empty or fully degenerate mesh: the reference returns index 0.
    if (count == 0u || params.total_area < EPS) {
        out_idx[tid] = 0u;
        return;
    }

    // Scale the clamped draw into area space; 'needle' avoids the reserved
    // identifier 'target'.
    let needle = clamp(draws[tid], 0.0, 1.0) * params.total_area;

    // partition_point(|&c| c <= needle): the count of prefix entries at or
    // below the scaled draw, found with a statically bounded binary search.
    var lo = 0u;
    var hi = count;
    for (var iter = 0u; iter < 32u; iter = iter + 1u) {
        if (lo >= hi) {
            break;
        }
        let mid = lo + (hi - lo) / 2u;
        if (cdf[mid] <= needle) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    var idx = min(lo, count - 1u);

    // Walk back over any trailing zero-area triangle so the pick has positive
    // area whenever the mesh does; bounded by the triangle count.
    for (var step = 0u; step < count; step = step + 1u) {
        if (idx == 0u) {
            break;
        }
        if (tri_area[idx] >= EPS) {
            break;
        }
        idx = idx - 1u;
    }

    out_idx[tid] = idx;
}
"#;

/// `std430` parameter block: the triangle count, the draw-batch length, the
/// total surface area, then one pad word — `16` bytes with no interior padding.
///
/// Provenance: layout mirror of the `WGSL` `Params` block; no Unreal Engine
/// source or derived code.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of triangles (length of `cdf` and `tri_area`).
    count: u32,
    /// Number of draws in the batch (length of `draws` and `out_idx`).
    draw_count: u32,
    /// Total surface area, the golden `total_area`.
    total_area: f32,
    /// Padding to keep the struct a multiple of `16` bytes.
    pad0: u32,
}

/// One area-weighted triangle-selection query over a whole batch of draws.
///
/// The host builds `cdf` and `triangle_area` from the mesh's public surface:
/// `triangle_area[i]` is `Mesh::triangle_area(i)` and `cdf[i]` is the running
/// prefix sum `sum(triangle_area[0..=i])`, with `total_area` the final running
/// sum — the identical arithmetic the golden private `rebuild_cdf` performs, so
/// the uploaded scalars match the selector's inputs bit for bit. `draws` is the
/// batch of unit draws; each entry picks one triangle.
///
/// Provenance: input mirror of the golden `Mesh::select_triangle` inputs; no
/// Unreal Engine source or derived code.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuMeshCdfSampleQuery {
    /// The area prefix-sum `CDF`, one `f32` per triangle.
    pub cdf: Vec<f32>,
    /// The per-triangle surface area, one `f32` per triangle.
    pub triangle_area: Vec<f32>,
    /// The total surface area, the golden `total_area`.
    pub total_area: f32,
    /// The batch of unit draws to resolve into triangle indices.
    pub draws: Vec<f32>,
}

/// The outcome of a [`GpuMeshCdfSample::sample`] batch.
///
/// Provenance: output mirror of the golden `Mesh::select_triangle` return; no
/// Unreal Engine source or derived code.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuMeshCdfSampleResult {
    /// The selected triangle index for each draw, in draw order.
    pub indices: Vec<u32>,
}

/// A compiled, reusable area-weighted triangle-selection pipeline.
///
/// Provenance: `wgpu` pipeline wrapper around [`MESH_CDF_SAMPLE_WGSL`]; no
/// Unreal Engine source or derived code.
pub struct GpuMeshCdfSample {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMeshCdfSample {
    /// Compiles the triangle-selection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: standard `wgpu` compute-pipeline creation; no Unreal Engine
    /// source or derived code.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMeshCdfSample {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mesh_cdf_sample"),
            source: ShaderSource::Wgsl(MESH_CDF_SAMPLE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mesh_cdf_sample_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mesh_cdf_sample_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mesh_cdf_sample_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("select_triangles"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMeshCdfSample {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every draw in `query` into an area-weighted triangle index.
    ///
    /// The returned indices equal the golden `Mesh::select_triangle` bit for
    /// bit. An empty draw batch returns an empty result (no dispatch); an empty
    /// or fully degenerate mesh yields index `0` for every draw, matching the
    /// reference's guards.
    ///
    /// Provenance: dispatch-and-readback around the golden `Mesh::select_triangle`;
    /// no Unreal Engine source or derived code.
    #[must_use]
    pub fn sample(
        &self,
        ctx: &GpuContext,
        query: &GpuMeshCdfSampleQuery,
    ) -> GpuMeshCdfSampleResult {
        let draw_count = query.draws.len();
        if draw_count == 0 {
            return GpuMeshCdfSampleResult {
                indices: Vec::new(),
            };
        }

        let device = ctx.device();

        let count = query.cdf.len();
        let gpu_params = Params {
            count: count as u32,
            draw_count: draw_count as u32,
            total_area: query.total_area,
            pad0: 0,
        };

        // Pad the per-triangle uploads to at least one element so the storage
        // buffers are never zero-sized; the kernel never reads them when
        // `count == 0`.
        let mut cdf_data = query.cdf.clone();
        if cdf_data.is_empty() {
            cdf_data.push(0.0);
        }
        let mut area_data = query.triangle_area.clone();
        if area_data.is_empty() {
            area_data.push(0.0);
        }
        let out_bytes = (draw_count as u64) * (size_of::<u32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_cdf_sample_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let cdf_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_cdf_sample_cdf"),
            contents: bytemuck::cast_slice(&cdf_data),
            usage: BufferUsages::STORAGE,
        });
        let area_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_cdf_sample_area"),
            contents: bytemuck::cast_slice(&area_data),
            usage: BufferUsages::STORAGE,
        });
        let draws_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_cdf_sample_draws"),
            contents: bytemuck::cast_slice(&query.draws),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mesh_cdf_sample_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mesh_cdf_sample_out_stage"),
            size: out_bytes,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mesh_cdf_sample_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: cdf_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: area_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: draws_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let groups = (draw_count as u32).div_ceil(WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mesh_cdf_sample_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mesh_cdf_sample_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &out_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        out_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let out_view = out_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped index readback range should be available after poll");
        let indices = bytemuck::cast_slice::<u8, u32>(&out_view).to_vec();
        drop(out_view);
        out_stage.unmap();

        debug_assert_eq!(indices.len(), draw_count);

        GpuMeshCdfSampleResult { indices }
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
