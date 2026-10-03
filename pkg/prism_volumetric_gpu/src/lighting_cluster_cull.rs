//! `wgpu` compute twin of the clustered light-vs-cluster overlap predicate
//! inside the view-space light culling contract
//! ([`culling`](prism_render_architecture::lighting::culling)).
//!
//! The `CPU` golden
//! [`light_overlaps_cluster`](prism_render_architecture::lighting::culling::light_overlaps_cluster)
//! answers one geometric question: does a light's bounding sphere reach into a
//! cluster's axis-aligned box? It computes the squared distance from the sphere
//! center to the box (per axis clamped, via the private helper
//! `distance_sq_point_aabb`) and compares it against the squared radius, so no
//! square root is ever taken. The surrounding assignment
//! ([`assign_cluster_lights`](prism_render_architecture::lighting::culling::assign_cluster_lights))
//! is a *variable-length aggregate*: it tests every light against every cluster,
//! appends overlapping indices into a `CSR` layout, caps each cluster at
//! `max_per_cluster`, and records overflow. That container assembly stays on the
//! host; it has no fixed-width device analogue.
//!
//! [`GpuLightClusterCull`] is the on-device twin of the *numeric core* that
//! assembly is built from: the single light-cluster overlap test. One thread
//! solves one `(light, cluster)` pairing, reproducing the reference's exact
//! clamped squared-distance form, so a passing real-device parity test is direct
//! evidence the ported kernel computes the same overlap decision the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For a sphere center `c`, radius `r`, and box `[min, max]` the reference
//! accumulates, per axis, `(min - c)^2` when `c < min`, `(c - max)^2` when
//! `c > max`, and `0` otherwise, then returns `distance_sq <= r * r`. The twin
//! reproduces that accumulation and comparison per pairing, emitting the boolean
//! overlap decision as a `0`/`1` word.
//!
//! # What stays on the host
//!
//! The `CSR` assembly of
//! [`assign_cluster_lights`](prism_render_architecture::lighting::culling::assign_cluster_lights)
//! — the per-cluster index lists, the `max_per_cluster` cap, the overflow
//! counter, and the ascending offset table — is variable-length container work
//! that the host owns; the device never sees it. The host enqueues one query per
//! light-cluster pairing it wants tested, so a storage buffer is never
//! zero-sized.
//!
//! # Correctness model
//!
//! The overlap decision is built from `+ - *` and magnitude comparisons only
//! (no `sqrt`), so for pairings chosen clear of the exact tangent tie
//! (`distance_sq == r * r`) the `CPU` and `GPU` agree and the parity test
//! asserts an exact match on the boolean. Fixtures are kept away from the
//! grazing boundary so a legal last-place difference in the squared-distance sum
//! can never flip the decision.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+ - *`, magnitude
//! comparison and unsigned index arithmetic — with no `sqrt`, no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `round` and no
//! `smoothstep`, and only `i32`/`u32`/`f32`/`bool` (no `u64`). No optional
//! device feature is required, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. There is no loop over variable data: each thread performs a fixed,
//! bounded sequence of arithmetic, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::lighting::culling`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` light-vs-cluster overlap kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `cull`
/// mirrors the `CPU` golden
/// [`light_overlaps_cluster`](prism_render_architecture::lighting::culling::light_overlaps_cluster)
/// clamped squared-distance form; see the module documentation for the
/// algorithm.
const LIGHTING_CLUSTER_CULL_WGSL: &str = r#"
// Clustered light-vs-cluster overlap twin: one thread tests one (light,
// cluster) pairing, accumulating the per-axis clamped squared distance from the
// sphere center to the box and comparing it against the squared radius, exactly
// as the CPU golden `lighting::culling::light_overlaps_cluster` and its
// `distance_sq_point_aabb` helper do, with only + - * and magnitude compares.
// It owns no CSR assembly, no per-cluster cap and no overflow counter; those
// variable-length aggregates stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::lighting::culling；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of pairings in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Sphere center in view space.
    cx: f32,
    cy: f32,
    cz: f32,
    // Sphere radius (light influence range).
    radius: f32,
    // Cluster box minimum corner.
    min_x: f32,
    min_y: f32,
    min_z: f32,
    // Cluster box maximum corner.
    max_x: f32,
    max_y: f32,
    max_z: f32,
    pad0: f32,
    pad1: f32,
}

struct Result {
    // 1 when the sphere overlaps the box, else 0.
    overlaps: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Per-axis clamped squared distance contribution, a verbatim port of the
// reference `distance_sq_point_aabb` axis body: below min adds (min - p)^2,
// above max adds (p - max)^2, inside the slab adds nothing.
fn axis_sq(p: f32, lo: f32, hi: f32) -> f32 {
    if (p < lo) {
        let d = lo - p;
        return d * d;
    }
    if (p > hi) {
        let d = p - hi;
        return d * d;
    }
    return 0.0;
}

@compute @workgroup_size(64)
fn cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Squared distance from the sphere center to the cluster box, summed over
    // the three independent axes, matching the reference helper's while loop.
    let dx = axis_sq(q.cx, q.min_x, q.max_x);
    let dy = axis_sq(q.cy, q.min_y, q.max_y);
    let dz = axis_sq(q.cz, q.min_z, q.max_z);
    let distance_sq = dx + dy + dz;

    // Compare against the squared radius so no square root is taken, exactly as
    // the reference `distance_sq <= radius * radius`.
    var overlaps: u32 = 0u;
    if (distance_sq <= q.radius * q.radius) {
        overlaps = 1u;
    }

    var out: Result;
    out.overlaps = overlaps;
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the pairing count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`LIGHTING_CLUSTER_CULL_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid pairings in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one pairing query: the sphere center and radius
/// plus the box corners and two pad words to a `48`-byte stride, matching the
/// `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Sphere center `x`.
    cx: f32,
    /// Sphere center `y`.
    cy: f32,
    /// Sphere center `z`.
    cz: f32,
    /// Sphere radius.
    radius: f32,
    /// Box minimum `x`.
    min_x: f32,
    /// Box minimum `y`.
    min_y: f32,
    /// Box minimum `z`.
    min_z: f32,
    /// Box maximum `x`.
    max_x: f32,
    /// Box maximum `y`.
    max_y: f32,
    /// Box maximum `z`.
    max_z: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
}

/// `repr(C)` `std430` layout of one pairing result, matching the `WGSL` `Result`
/// struct: the overlap flag plus three pad words to a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `1` when the sphere overlaps the box, `0` otherwise.
    overlaps: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One light-vs-cluster overlap query: the light's bounding-sphere `center` and
/// `radius` in view space, and the cluster box corners `cluster_min` /
/// `cluster_max`.
///
/// The host owns the surrounding `CSR` assembly — the per-cluster index lists,
/// the `max_per_cluster` cap, and the overflow counter — and enqueues one
/// [`LightClusterCullQuery`] per `(light, cluster)` pairing it wants tested,
/// matching the reference
/// [`assign_cluster_lights`](prism_render_architecture::lighting::culling::assign_cluster_lights)
/// nested loop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightClusterCullQuery {
    /// Light bounding-sphere center in view space.
    pub center: [f32; 3],
    /// Light bounding-sphere radius (influence range) in view space.
    pub radius: f32,
    /// Cluster box minimum corner in view space.
    pub cluster_min: [f32; 3],
    /// Cluster box maximum corner in view space.
    pub cluster_max: [f32; 3],
}

impl LightClusterCullQuery {
    /// Builds a query for one light sphere against one cluster box.
    #[must_use]
    pub const fn new(
        center: [f32; 3],
        radius: f32,
        cluster_min: [f32; 3],
        cluster_max: [f32; 3],
    ) -> LightClusterCullQuery {
        LightClusterCullQuery {
            center,
            radius,
            cluster_min,
            cluster_max,
        }
    }
}

/// One resolved light-vs-cluster overlap, mirroring the boolean the reference
/// [`light_overlaps_cluster`](prism_render_architecture::lighting::culling::light_overlaps_cluster)
/// returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LightClusterCullResult {
    /// `true` when the light's sphere overlaps the cluster's box.
    pub overlaps: bool,
}

/// Encodes one [`LightClusterCullQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &LightClusterCullQuery) -> GpuQuery {
    GpuQuery {
        cx: q.center[0],
        cy: q.center[1],
        cz: q.center[2],
        radius: q.radius,
        min_x: q.cluster_min[0],
        min_y: q.cluster_min[1],
        min_z: q.cluster_min[2],
        max_x: q.cluster_max[0],
        max_y: q.cluster_max[1],
        max_z: q.cluster_max[2],
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`LightClusterCullResult`],
/// turning the `overlaps` word back into a [`bool`].
fn decode_result(raw: &GpuResult) -> LightClusterCullResult {
    LightClusterCullResult {
        overlaps: raw.overlaps != 0,
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

/// A compiled, reusable light-vs-cluster overlap compute pipeline, twinning the
/// `CPU` golden
/// [`light_overlaps_cluster`](prism_render_architecture::lighting::culling::light_overlaps_cluster).
pub struct GpuLightClusterCull {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuLightClusterCull {
    /// Compiles the light-vs-cluster overlap kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuLightClusterCull {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_lighting_cluster_cull"),
            source: ShaderSource::Wgsl(LIGHTING_CLUSTER_CULL_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_lighting_cluster_cull_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_lighting_cluster_cull_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_lighting_cluster_cull_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("cull"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuLightClusterCull {
            module,
            layout,
            pipeline,
        }
    }

    /// Tests every pairing in `queries` and returns one
    /// [`LightClusterCullResult`] per input, in order.
    ///
    /// Each result's `overlaps` flag equals the `CPU` golden
    /// [`light_overlaps_cluster`](prism_render_architecture::lighting::culling::light_overlaps_cluster)
    /// at the matching pairing; because the decision is built from `+ - *` and
    /// magnitude comparisons with no `sqrt`, the agreement is exact for pairings
    /// clear of the tangent tie. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[LightClusterCullQuery],
    ) -> Vec<LightClusterCullResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_lighting_cluster_cull_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_lighting_cluster_cull_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_lighting_cluster_cull_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_lighting_cluster_cull_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_lighting_cluster_cull_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_lighting_cluster_cull_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_lighting_cluster_cull_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per pairing, flattened to a 1-D dispatch.
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

        raw.iter().map(decode_result).collect()
    }
}
