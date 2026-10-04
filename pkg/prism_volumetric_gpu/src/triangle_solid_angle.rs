//! `wgpu` compute twin of the Van Oosterom-Strackee signed solid angle, from
//! the `CPU` golden `prism_physics_core::collider::winding_number`'s
//! `signed_solid_angle`.
//!
//! A triangle `(v0, v1, v2)` subtends a signed solid angle at a query `point`;
//! summed over a mesh and divided by `4*pi` this is the generalized winding
//! number used for robust point-in-mesh tests. This module ports that single
//! stateless closed form onto the device: one thread resolves one triangle, so
//! a passing real-device parity test is direct evidence the ported kernel
//! computes the same solid angle the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `signed_solid_angle` for one triangle,
//! with the three vertices lifted relative to the query point:
//!
//! * `a = v0 - point`, `b = v1 - point`, `c = v2 - point`.
//! * `la = |a|`, `lb = |b|`, `lc = |c|`.
//! * `denom = la*lb*lc + dot(a,b)*lc + dot(b,c)*la + dot(c,a)*lb`.
//! * `numer = dot(a, cross(b, c))`.
//! * `solid_angle = 2 * atan2(numer, denom)`, with the branch where both
//!   `numer` and `denom` vanish (a vertex coincident with the query) forced to
//!   `0` so an otherwise undefined `atan2(0, 0)` contributes nothing.
//! * `winding_contribution = solid_angle / (4 * pi)`.
//!
//! The golden accumulates in `f64`; the device kernel runs in `f32` with
//! `atan2` as the `WGSL` built-in, so the parity oracle also evaluates in `f64`
//! and the two sides are compared to tolerance rather than bit-exactly.
//!
//! # Correctness model
//!
//! The continuous arithmetic threads through operators a `GPU` may contract and
//! through the `f32`-vs-`f64` `atan2` gap, so `CPU` and `GPU` are not bit-exact;
//! the two outputs are compared with `abs <= 1e-4 || rel <= 1e-3`
//! (`REL_FLOOR = 1e-6`). The `atan2` branch cut lives at `numer -> 0` with
//! `denom < 0`, where the solid angle jumps between `+2*pi` and `-2*pi`; the
//! parity test keeps random fixtures clear of that knee (bounded magnitude and
//! non-coincident vertices), so round-off cannot land the two sides on opposite
//! branches.
//!
//! # Degenerate inputs
//!
//! When a vertex coincides with the query point, `numer` and `denom` both
//! vanish and the kernel emits `0`. The `atan2` divisor is additionally fed
//! through a `select` guard so the un-taken branch never evaluates
//! `atan2(0, 0)`. An empty query batch short-circuits on the host with no
//! dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `length`,
//! `dot`, `cross`, `atan2`, `+ - * /` and `select` with unsigned index
//! arithmetic — with no `sin`, `cos`, `tan`, `exp`, `log`, `pow`, no `round`,
//! no `f32` remainder, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//! The degeneracy test is the ordered compare `abs(x) < EPS0` fed to `select`
//! rather than a bare `x == x`; there is no `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::winding_number`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` solid-angle kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `signed_solid_angle`; see the module documentation for the
/// closed form.
const TRIANGLE_SOLID_ANGLE_WGSL: &str = r#"
// Signed-solid-angle twin: one thread per query reproduces signed_solid_angle.
// It uses only the portable core-WGSL subset (abs, length, dot, cross, atan2,
// + - * /, select plus unsigned index math), takes no optional feature, and has
// no loop and no branch, so it provably terminates. The vertex-coincident
// degeneracy is detected with ordered abs < EPS0 compares (never a bare
// equality) and the atan2 divisor is select-guarded so the un-taken branch
// never evaluates atan2(0, 0).

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Query point.
    point: vec3<f32>,
    pad0: f32,
    // First triangle vertex.
    v0: vec3<f32>,
    pad1: f32,
    // Second triangle vertex.
    v1: vec3<f32>,
    pad2: f32,
    // Third triangle vertex.
    v2: vec3<f32>,
    pad3: f32,
}

struct Result {
    // Signed solid angle in steradians.
    solid_angle: f32,
    // solid_angle / (4 * pi).
    winding_contribution: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

const EPS0: f32 = 1.0e-30;
const PI: f32 = 3.1415927;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    // Lift the vertices relative to the query point.
    let a = q.v0 - q.point;
    let b = q.v1 - q.point;
    let c = q.v2 - q.point;
    let la = length(a);
    let lb = length(b);
    let lc = length(c);

    // Golden operator order for the Van Oosterom-Strackee denominator.
    let denom = la * lb * lc + dot(a, b) * lc + dot(b, c) * la + dot(c, a) * lb;
    let numer = dot(a, cross(b, c));

    // A vertex coincident with the query makes both numer and denom vanish;
    // force a zero contribution rather than an undefined atan2(0, 0). Guard the
    // divisor so the un-taken branch never feeds atan2 a zero pair either.
    let degenerate = (abs(numer) < EPS0) && (abs(denom) < EPS0);
    let safe_denom = select(denom, 1.0, degenerate);
    let omega = select(2.0 * atan2(numer, safe_denom), 0.0, degenerate);

    var out: Result;
    out.solid_angle = omega;
    out.winding_contribution = omega / (4.0 * PI);
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    results[idx] = out;
}
"#;

/// `repr(C)` `std430` dispatch parameters: the query count plus padding to a
/// 16-byte uniform block.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
/// The query point and three vertices are each a `vec3<f32>` padded to a
/// 16-byte-aligned slot, so the whole query is `64` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    point: [f32; 3],
    pad0: f32,
    v0: [f32; 3],
    pad1: f32,
    v1: [f32; 3],
    pad2: f32,
    v2: [f32; 3],
    pad3: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the signed solid angle, the winding contribution and padding to a
/// `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    solid_angle: f32,
    winding_contribution: f32,
    pad0: f32,
    pad1: f32,
}

/// One solid-angle query: the query point and the triangle's three vertices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriangleSolidAngleQuery {
    /// The query point the solid angle is measured from.
    pub point: [f32; 3],
    /// First triangle vertex.
    pub v0: [f32; 3],
    /// Second triangle vertex.
    pub v1: [f32; 3],
    /// Third triangle vertex.
    pub v2: [f32; 3],
}

impl TriangleSolidAngleQuery {
    /// Builds a query from the query point and the triangle's three vertices.
    #[must_use]
    pub fn new(
        point: [f32; 3],
        v0: [f32; 3],
        v1: [f32; 3],
        v2: [f32; 3],
    ) -> TriangleSolidAngleQuery {
        TriangleSolidAngleQuery { point, v0, v1, v2 }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `signed_solid_angle` output for that triangle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriangleSolidAngleResult {
    /// Signed solid angle `2 * atan2(numer, denom)` in steradians; `0` when a
    /// vertex coincides with the query point.
    pub solid_angle: f32,
    /// The winding contribution `solid_angle / (4 * pi)`.
    pub winding_contribution: f32,
}

/// Encodes one [`TriangleSolidAngleQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &TriangleSolidAngleQuery) -> GpuQuery {
    GpuQuery {
        point: q.point,
        pad0: 0.0,
        v0: q.v0,
        pad1: 0.0,
        v1: q.v1,
        pad2: 0.0,
        v2: q.v2,
        pad3: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`TriangleSolidAngleResult`].
fn decode_result(raw: &GpuResult) -> TriangleSolidAngleResult {
    TriangleSolidAngleResult {
        solid_angle: raw.solid_angle,
        winding_contribution: raw.winding_contribution,
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

/// A compiled, reusable solid-angle compute pipeline, twinning the `CPU` golden
/// `signed_solid_angle`.
pub struct GpuTriangleSolidAngle {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTriangleSolidAngle {
    /// Compiles the solid-angle kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTriangleSolidAngle {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_triangle_solid_angle"),
            source: ShaderSource::Wgsl(TRIANGLE_SOLID_ANGLE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_triangle_solid_angle_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_triangle_solid_angle_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_triangle_solid_angle_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTriangleSolidAngle {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`TriangleSolidAngleResult`] per input, in order.
    ///
    /// The outputs match the reference to the module's tolerance. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[TriangleSolidAngleQuery],
    ) -> Vec<TriangleSolidAngleResult> {
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
            label: Some("prism_volumetric_triangle_solid_angle_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_triangle_solid_angle_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_triangle_solid_angle_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_triangle_solid_angle_bind_group"),
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
            label: Some("prism_volumetric_triangle_solid_angle_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_triangle_solid_angle_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_triangle_solid_angle_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
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
