//! `wgpu` compute twin of the plane-quadric helper from the `CPU` golden
//! `prism_physics_core::collider::quadric::Quadric::from_plane`.
//!
//! A quadric error metric packs the symmetric `4x4` matrix of a plane into the
//! ten upper-triangular coefficients `a2, ab, ac, ad, b2, bc, bd, c2, cd, d2`,
//! where `(a, b, c)` is the plane's unit normal and `d` its offset. Evaluating
//! that quadric against a point yields the squared distance from the point to
//! the plane. The golden builds those ten coefficients from the normal and
//! offset with a fixed sequence of scalar multiplications. This module ports
//! that stateless, no-`RNG`, branch-free closed form onto the device: one
//! compute thread resolves one plane, so a passing real-device parity test is
//! direct evidence the kernel reproduces the exact coefficient order, not
//! merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each query is one plane: a unit normal `n` and an offset `d`. The kernel
//! reproduces the reference closed form operation for operation:
//!
//! * `a2 = a*a`, `ab = a*b`, `ac = a*c`, `ad = a*d`;
//! * `b2 = b*b`, `bc = b*c`, `bd = b*d`;
//! * `c2 = c*c`, `cd = c*d`, `d2 = d*d`;
//!
//! with `(a, b, c) = (n.x, n.y, n.z)`.
//!
//! There is no division and no branch: the computation is pure multiplication,
//! so the kernel provably terminates and `valid` is always `1`.
//!
//! # Correctness model
//!
//! Each coefficient is a single multiply, so `CPU` and `GPU` are not required
//! to be bit-exact. The parity test asserts a tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on each of the ten coefficients; the
//! discrete `valid` flag is compared exactly. The kernel has no comparisons at
//! all, so no bare float equality and no fast-math `NaN` sentinel is involved.
//!
//! # Degenerate inputs
//!
//! There are no degenerate inputs: every normal and offset yields ten
//! well-defined coefficients, so `valid` is always `1`. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot
//! be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `*` on `f32` and
//! unsigned index arithmetic — with no `sin`, `cos`, `tan`, `exp`, `log`,
//! `pow`, no `round`, no float modulo and no optional device feature, so it
//! runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::quadric::Quadric::from_plane`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` plane-quadric kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `project` mirrors the
/// `CPU` golden `from_plane`; see the module documentation for the algorithm.
const QUADRIC_FROM_PLANE_WGSL: &str = r#"
// Plane-quadric twin: one thread per plane reproduces the ten upper-triangular
// quadric coefficients the golden Quadric::from_plane builds from a unit normal
// and an offset. It mirrors the CPU golden operation for operation, uses only
// the portable core-WGSL subset (f32 multiply plus unsigned index math), takes
// no optional feature, and has no loop, so the kernel provably terminates.
// There is no division and no branch, so no float equality or fast-math
// sentinel is involved.
//
// Provenance: 孪生自本仓
// prism_physics_core::collider::quadric::Quadric::from_plane；
// 无第三方引擎源码或衍生代码。

struct Params {
    // Number of planes in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Unit normal components.
    nx: f32,
    ny: f32,
    nz: f32,
    // Plane offset.
    d: f32,
}

struct Result {
    // Ten upper-triangular quadric coefficients in the golden's order.
    a2: f32,
    ab: f32,
    ac: f32,
    ad: f32,
    b2: f32,
    bc: f32,
    bd: f32,
    c2: f32,
    cd: f32,
    d2: f32,
    // Always 1: the closed form has no degenerate branch.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let a = q.nx;
    let b = q.ny;
    let c = q.nz;
    let d = q.d;

    var out: Result;
    out.a2 = a * a;
    out.ab = a * b;
    out.ac = a * c;
    out.ad = a * d;
    out.b2 = b * b;
    out.bc = b * c;
    out.bd = b * d;
    out.c2 = c * c;
    out.cd = c * d;
    out.d2 = d * d;
    out.valid = 1u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`QUADRIC_FROM_PLANE_WGSL`].
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
/// All components are scalar `f32`, so the four-word slot is naturally `16`-byte
/// aligned and the host and device agree on the array stride byte for byte.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    nx: f32,
    ny: f32,
    nz: f32,
    d: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct. The ten coefficients are stored as flat scalars in the golden's
/// order; the trailing `valid` word keeps the discrete flag beside them.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    coeffs: [f32; 10],
    valid: u32,
}

/// One query for the plane-quadric twin: a unit normal and a plane offset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadricFromPlaneQuery {
    /// Unit plane normal `(a, b, c)`.
    pub normal: [f32; 3],
    /// Plane offset `d` in `n . x + d = 0`.
    pub d: f32,
}

impl QuadricFromPlaneQuery {
    /// Builds a query from the unit normal and the plane offset.
    #[must_use]
    pub fn new(normal: [f32; 3], d: f32) -> QuadricFromPlaneQuery {
        QuadricFromPlaneQuery { normal, d }
    }
}

/// One resolved answer for a single plane: the ten upper-triangular quadric
/// coefficients in the golden's order, plus the `valid` flag (always `1`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadricFromPlaneResult {
    /// The ten coefficients, in the golden's order:
    /// `a2, ab, ac, ad, b2, bc, bd, c2, cd, d2`.
    pub coeffs: [f32; 10],
    /// Always `1`: the closed form has no degenerate branch.
    pub valid: u32,
}

/// Encodes one [`QuadricFromPlaneQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &QuadricFromPlaneQuery) -> GpuQuery {
    GpuQuery {
        nx: q.normal[0],
        ny: q.normal[1],
        nz: q.normal[2],
        d: q.d,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`QuadricFromPlaneResult`].
fn decode_result(raw: &GpuResult) -> QuadricFromPlaneResult {
    QuadricFromPlaneResult {
        coeffs: raw.coeffs,
        valid: raw.valid,
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

/// A compiled, reusable plane-quadric compute pipeline, twinning the `CPU`
/// golden `Quadric::from_plane`.
pub struct GpuQuadricFromPlane {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuQuadricFromPlane {
    /// Compiles the plane-quadric kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuQuadricFromPlane {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_quadric_from_plane"),
            source: ShaderSource::Wgsl(QUADRIC_FROM_PLANE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_quadric_from_plane_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_quadric_from_plane_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_quadric_from_plane_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("project"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuQuadricFromPlane {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every plane in `queries` and returns one
    /// [`QuadricFromPlaneResult`] per input, in order.
    ///
    /// Each coefficient matches the reference to within the tolerance
    /// documented on this module; the `valid` flag matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[QuadricFromPlaneQuery],
    ) -> Vec<QuadricFromPlaneResult> {
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
            label: Some("prism_volumetric_quadric_from_plane_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_quadric_from_plane_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_quadric_from_plane_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_quadric_from_plane_bind_group"),
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
            label: Some("prism_volumetric_quadric_from_plane_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_quadric_from_plane_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_quadric_from_plane_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per plane, flattened to a 1-D dispatch.
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
