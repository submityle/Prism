//! `wgpu` compute twin of the triangle-versus-axis-aligned-box boolean overlap
//! contract
//! ([`triangle_aabb_overlap`](prism_render_architecture::particle::triangle_aabb_overlap),
//! particle design §10, §13).
//!
//! The `CPU` golden
//! [`triangle_overlaps_aabb`](prism_render_architecture::particle::triangle_aabb_overlap::triangle_overlaps_aabb)
//! answers one yes/no question per query: does a triangle (three corners) touch
//! an axis-aligned box (center plus half-extents)? It is the thirteen-axis
//! Akenine-Möller separating-axis test — the three box face normals, the one
//! triangle face normal and the nine triangle-edge-cross-box-axis directions —
//! and it returns a single boolean. [`GpuTriangleAabbOverlap`] is the on-device
//! twin: one thread per `(triangle, box)` pair reproduces that boolean, so a
//! passing real-device parity test is direct evidence the ported kernel runs
//! the same `SAT` and classifies the same degenerate and touching cases the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Every per-query answer the reference computes is reproduced bit-for-decision:
//! the kernel mirrors the reference branch for branch. It first moves the
//! triangle into the box-centered frame, takes the box half-extents by
//! magnitude, then checks each candidate axis in the same order — the three box
//! face normals (`x`, `y`, `z`), the triangle face normal `cross(e0, e1)`, and
//! the nine `cross(edge, unit_axis)` directions. On every axis the box projects
//! to a symmetric interval of half-width `radius` and the triangle to its
//! `[min, max]` vertex-dot interval; a positive gap beyond [`SEP_EPS`] proves
//! disjointness. A near-zero candidate axis (squared length below
//! [`DEGENERATE_AXIS_EPS`]) cannot separate and is skipped, exactly as the
//! reference's `axis_separates` short-circuits.
//!
//! # Degenerate and touching cases
//!
//! A triangle collapsed to a segment or a point has vanishing edge-cross and/or
//! face-normal axes; those are skipped on both sides, so the query reduces to
//! the correct segment-versus-box or point-versus-box test. A zero-volume box
//! (a half-extent at zero) projects to a point on that axis and is handled by
//! the same interval algebra. An exact vertex/edge/face contact leaves a zero
//! gap, which the [`SEP_EPS`] slack keeps on the overlap side; the twin applies
//! the identical slack, so both report `true` there.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`, `abs`,
//! `dot`, `cross` and `+ - *` — with no `sqrt`, `sin`, `cos`, `exp`, `log`,
//! `pow`, `tan`, no rounding and no optional device feature, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! The output is a pure boolean, so parity is an exact `==` on every element
//! with no tolerance. The reference compares `f32` projection gaps against
//! [`SEP_EPS`] rather than `==`, so a `GPU` fusing a multiply-add perturbs a gap
//! by a few units in the last place but cannot flip a decision as long as every
//! fixture stays clearly on one side of each axis comparison — which the parity
//! fixtures ensure by construction. The boolean is therefore reproduced exactly.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`triangle_aabb_overlap`](prism_render_architecture::particle::triangle_aabb_overlap);
//! no third-party engine source or derived code.

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

/// The portable core-`WGSL` triangle-versus-box `SAT` kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`triangle_overlaps_aabb`](prism_render_architecture::particle::triangle_aabb_overlap::triangle_overlaps_aabb)
/// branch for branch; see the module documentation for the algorithm.
const TRIANGLE_AABB_OVERLAP_WGSL: &str = r#"
// Triangle-versus-axis-aligned-box boolean overlap twin: one thread per
// (triangle, box) pair runs the thirteen-axis Akenine-Möller separating-axis
// test and writes one u32 (1 = overlap, 0 = disjoint). It mirrors the CPU golden
// `particle::triangle_aabb_overlap` branch for branch, uses only the portable
// core-WGSL subset (min/max/abs/dot/cross and + - *), needs no sqrt and takes no
// optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's particle::triangle_aabb_overlap;
// no third-party engine source or derived code.

// Separation slack added to the summed projection radii: a configuration is
// disjoint only when the gap exceeds this, so an exact vertex/edge/face contact
// counts as an overlap. Matches the reference `SEP_EPS`; used in place of == /!=.
const SEP_EPS: f32 = 1.0e-6;

// Squared-length threshold below which a candidate axis is degenerate and
// skipped: a vanishing axis cannot separate anything. Matches the reference
// `DEGENERATE_AXIS_EPS`, the square of a ~1e-6 direction tolerance.
const DEGENERATE_AXIS_EPS: f32 = 1.0e-12;

struct Params {
    // Number of (triangle, box) pairs in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Triangle corner 0; a pad lane follows.
    v0: vec3<f32>,
    pad0: f32,
    // Triangle corner 1; a pad lane follows.
    v1: vec3<f32>,
    pad1: f32,
    // Triangle corner 2; a pad lane follows.
    v2: vec3<f32>,
    pad2: f32,
    // Box center; a pad lane follows.
    center: vec3<f32>,
    pad3: f32,
    // Box half-extents (taken by magnitude in-kernel); a pad lane follows.
    half: vec3<f32>,
    pad4: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<u32>;

// True when the box face axis separates the triangle (vertex coords c0/c1/c2
// along that axis) from the origin-centered box of half-extent h, mirroring the
// reference `face_separates`.
fn face_separates(c0: f32, c1: f32, c2: f32, h: f32) -> bool {
    let lo = min(min(c0, c1), c2);
    let hi = max(max(c0, c1), c2);
    return lo > h + SEP_EPS || hi < -h - SEP_EPS;
}

// True when `axis` separates the triangle a0/a1/a2 (box-centered) from the box
// with the given `half` extents, mirroring the reference `axis_separates`. A
// degenerate (near-zero) axis can never separate, so it returns false.
fn axis_separates(
    axis: vec3<f32>,
    a0: vec3<f32>,
    a1: vec3<f32>,
    a2: vec3<f32>,
    half: vec3<f32>,
) -> bool {
    if (dot(axis, axis) < DEGENERATE_AXIS_EPS) {
        return false;
    }
    let p0 = dot(axis, a0);
    let p1 = dot(axis, a1);
    let p2 = dot(axis, a2);
    let lo = min(min(p0, p1), p2);
    let hi = max(max(p0, p1), p2);
    let radius = half.x * abs(axis.x) + half.y * abs(axis.y) + half.z * abs(axis.z);
    return lo > radius + SEP_EPS || hi < -radius - SEP_EPS;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    // Half-extents taken by magnitude so a mis-signed input cannot invert the box.
    let half = abs(q.half);

    // Move the triangle into the box-centered frame.
    let v0 = q.v0 - q.center;
    let v1 = q.v1 - q.center;
    let v2 = q.v2 - q.center;

    var overlap: u32 = 1u;

    // Axes 1-3: the box face normals (triangle AABB vs box AABB).
    if (face_separates(v0.x, v1.x, v2.x, half.x)) {
        overlap = 0u;
    } else if (face_separates(v0.y, v1.y, v2.y, half.y)) {
        overlap = 0u;
    } else if (face_separates(v0.z, v1.z, v2.z, half.z)) {
        overlap = 0u;
    } else {
        // Triangle edges in the box-centered frame.
        let e0 = v1 - v0;
        let e1 = v2 - v1;
        let e2 = v0 - v2;

        // Axis 4: the triangle face normal (plane vs box).
        let normal = cross(e0, e1);
        if (axis_separates(normal, v0, v1, v2, half)) {
            overlap = 0u;
        } else {
            // Axes 5-13: each triangle edge crossed with each box axis.
            var edges = array<vec3<f32>, 3>(e0, e1, e2);
            var units = array<vec3<f32>, 3>(
                vec3<f32>(1.0, 0.0, 0.0),
                vec3<f32>(0.0, 1.0, 0.0),
                vec3<f32>(0.0, 0.0, 1.0),
            );
            for (var i: u32 = 0u; i < 3u; i = i + 1u) {
                for (var j: u32 = 0u; j < 3u; j = j + 1u) {
                    if (axis_separates(cross(edges[i], units[j]), v0, v1, v2, half)) {
                        overlap = 0u;
                    }
                }
            }
        }
    }

    results[idx] = overlap;
}
"#;

/// One triangle-versus-box overlap query: the triangle's three corners plus the
/// box as a center and half-extents, exactly the inputs the reference
/// [`triangle_overlaps_aabb`](prism_render_architecture::particle::triangle_aabb_overlap::triangle_overlaps_aabb)
/// consumes. Half-extents are taken by magnitude in-kernel, matching the
/// reference, so a mis-signed extent cannot invert the box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriangleAabbQuery {
    /// The triangle corners `[v0, v1, v2]`, each a 3D point.
    pub tri: [[f32; 3]; 3],
    /// The box center.
    pub center: [f32; 3],
    /// The box half-extents (taken by magnitude).
    pub half: [f32; 3],
}

impl TriangleAabbQuery {
    /// Builds a query from a triangle, a box center and box half-extents.
    #[must_use]
    pub const fn new(tri: [[f32; 3]; 3], center: [f32; 3], half: [f32; 3]) -> TriangleAabbQuery {
        TriangleAabbQuery { tri, center, half }
    }
}

/// `repr(C)` `std430` layout of one packed query: five `vec4` slots holding
/// `(v0.xyz, pad)`, `(v1.xyz, pad)`, `(v2.xyz, pad)`, `(center.xyz, pad)` and
/// `(half.xyz, pad)` — `80` bytes, each `vec3` on its `16`-byte-aligned slot
/// exactly as the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Triangle corner `0`.
    v0: [f32; 3],
    /// Padding lane after corner `0`.
    pad0: f32,
    /// Triangle corner `1`.
    v1: [f32; 3],
    /// Padding lane after corner `1`.
    pad1: f32,
    /// Triangle corner `2`.
    v2: [f32; 3],
    /// Padding lane after corner `2`.
    pad2: f32,
    /// Box center.
    center: [f32; 3],
    /// Padding lane after the center.
    pad3: f32,
    /// Box half-extents.
    half: [f32; 3],
    /// Padding lane after the half-extents.
    pad4: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &TriangleAabbQuery) -> GpuQuery {
        GpuQuery {
            v0: query.tri[0],
            pad0: 0.0,
            v1: query.tri[1],
            pad1: 0.0,
            v2: query.tri[2],
            pad2: 0.0,
            center: query.center,
            pad3: 0.0,
            half: query.half,
            pad4: 0.0,
        }
    }
}

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `16`-byte-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of queries in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A compiled, reusable triangle-versus-box overlap compute pipeline.
pub struct GpuTriangleAabbOverlap {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTriangleAabbOverlap {
    /// Compiles the triangle-versus-box overlap kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTriangleAabbOverlap {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_triangle_aabb_overlap"),
            source: ShaderSource::Wgsl(TRIANGLE_AABB_OVERLAP_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_triangle_aabb_overlap_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_triangle_aabb_overlap_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_triangle_aabb_overlap_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTriangleAabbOverlap {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one overlap boolean per input,
    /// in order.
    ///
    /// Each result equals the reference
    /// [`triangle_overlaps_aabb`](prism_render_architecture::particle::triangle_aabb_overlap::triangle_overlaps_aabb)
    /// exactly: `true` when the triangle intersects or merely touches the box,
    /// `false` when a separating axis proves them disjoint. An empty input
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[TriangleAabbQuery]) -> Vec<bool> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_triangle_aabb_overlap_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<u32>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_triangle_aabb_overlap_output"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_triangle_aabb_overlap_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_triangle_aabb_overlap_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_triangle_aabb_overlap_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_triangle_aabb_overlap_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_triangle_aabb_overlap_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        stage.unmap();

        raw.iter().map(|&flag| flag != 0).collect()
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
