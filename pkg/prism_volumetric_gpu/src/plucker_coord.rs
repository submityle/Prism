//! `wgpu` compute twin of the Plücker line-coordinate contract
//! ([`plucker_coord`](prism_render_architecture::particle::plucker_coord),
//! particle design §8.2 mesh emission, §11 collision, §13 culling).
//!
//! The `CPU` golden
//! [`plucker_coord`](prism_render_architecture::particle::plucker_coord)
//! owns the pure Plücker algebra over directed lines. From the two endpoints of
//! a segment it builds the six-tuple
//! [`Line6`](prism_render_architecture::particle::plucker_coord::Line6): the
//! direction `u = p1 - p0` and the moment `v = p0 × p1` about the origin
//! ([`Line6::from_points`](prism_render_architecture::particle::plucker_coord::Line6::from_points)).
//! The relative orientation of two such lines is then the single symmetric
//! permuted inner product
//! [`side`](prism_render_architecture::particle::plucker_coord::side)
//! `side(l1, l2) = u1·v2 + u2·v1`, whose sign tells whether one directed line
//! passes clockwise, counter-clockwise, or coplanarly about the other without
//! ever solving a linear system. [`GpuPluckerCoord`] is the on-device twin: one
//! thread per query rebuilds both lines from their endpoints and reproduces the
//! `side` product plus each line's Plücker residual `u·v`
//! ([`Line6::moment_orthogonality`](prism_render_architecture::particle::plucker_coord::Line6::moment_orthogonality)),
//! so a passing real-device parity test is direct evidence the ported kernel
//! builds the same coordinates and the same orientation form the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Per query the kernel reproduces both constructed
//! [`Line6`](prism_render_architecture::particle::plucker_coord::Line6) values
//! (direction and moment for each of the two input segments), the permuted
//! inner product
//! [`side`](prism_render_architecture::particle::plucker_coord::side), and the
//! two Plücker residuals
//! [`Line6::moment_orthogonality`](prism_render_architecture::particle::plucker_coord::Line6::moment_orthogonality)
//! (`u·v`, identically zero up to rounding for any line built from two points).
//!
//! # Portability
//!
//! The kernel is pure `-`, `*`, `+` and `dot`: the direction is a subtract, the
//! moment is a hand-expanded cross product of multiplies and subtracts, and
//! both `side` and the residuals are `dot` reductions. There is no `sin`,
//! `cos`, `exp`, `log`, `pow`, `tan`, no `sqrt` and no optional device feature,
//! so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and
//! subtracts, so `CPU` and `GPU` evaluate the same closed form in the same
//! order. They are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits by a few units
//! in the last place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32` fields, tight enough
//! to catch a genuinely wrong port (a swapped cross-product term, a dropped
//! `side` summand) yet loose enough to admit legal fused multiply-add
//! contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`plucker_coord`](prism_render_architecture::particle::plucker_coord); no
//! third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::plucker_coord::{Line6, Vec3};
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

/// The portable core-`WGSL` Plücker-coordinate kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `evaluate`
/// mirrors the `CPU` golden
/// [`plucker_coord`](prism_render_architecture::particle::plucker_coord): it
/// rebuilds each [`Line6`](prism_render_architecture::particle::plucker_coord::Line6)
/// from its two endpoints and reproduces `side` and the moment residuals; see
/// the module documentation for the algorithm.
const PLUCKER_COORD_WGSL: &str = r#"
// Plücker line-coordinate twin: one thread per query rebuilds two directed lines
// from their segment endpoints (direction u = p1 - p0, moment v = p0 × p1) and
// reports the permuted inner product side(l1, l2) = u1·v2 + u2·v1 together with
// each line's Plücker residual u·v. It is pure subtract/multiply/dot with no
// sqrt and no transcendental, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's particle::plucker_coord; no
// third-party engine source or derived code.

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // First line start endpoint; a pad lane follows.
    first_start: vec3<f32>,
    pad0: f32,
    // First line end endpoint; a pad lane follows.
    first_end: vec3<f32>,
    pad1: f32,
    // Second line start endpoint; a pad lane follows.
    second_start: vec3<f32>,
    pad2: f32,
    // Second line end endpoint; a pad lane follows.
    second_end: vec3<f32>,
    pad3: f32,
}

struct Result {
    // First line direction u1 = first_end - first_start; a pad lane follows.
    first_u: vec3<f32>,
    pad0: f32,
    // First line moment v1 = first_start × first_end; a pad lane follows.
    first_v: vec3<f32>,
    pad1: f32,
    // Second line direction u2 = second_end - second_start; a pad lane follows.
    second_u: vec3<f32>,
    pad2: f32,
    // Second line moment v2 = second_start × second_end; a pad lane follows.
    second_v: vec3<f32>,
    pad3: f32,
    // Permuted inner product, then the two Plücker residuals u1·v1 and u2·v2,
    // filling one vec4 slot with a trailing pad lane.
    side_value: f32,
    first_moment_residual: f32,
    second_moment_residual: f32,
    pad4: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// The origin moment p × q, hand-expanded to multiplies and subtracts in the
// exact component order of the reference `Vec3::cross` so the ported moment
// matches the golden term for term.
fn plucker_moment(p: vec3<f32>, q: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        p.y * q.z - p.z * q.y,
        p.z * q.x - p.x * q.z,
        p.x * q.y - p.y * q.x,
    );
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let a1 = q.first_start;
    let b1 = q.first_end;
    let a2 = q.second_start;
    let b2 = q.second_end;

    // Line6::from_points: direction is the endpoint difference, moment is the
    // origin cross product of the two endpoints.
    let u1 = b1 - a1;
    let v1 = plucker_moment(a1, b1);
    let u2 = b2 - a2;
    let v2 = plucker_moment(a2, b2);

    // side(l1, l2) = u1·v2 + u2·v1, the permuted inner product.
    let side_value = dot(u1, v2) + dot(u2, v1);

    // moment_orthogonality: the Plücker residual u·v for each line.
    let r1 = dot(u1, v1);
    let r2 = dot(u2, v2);

    var out: Result;
    out.first_u = u1;
    out.pad0 = 0.0;
    out.first_v = v1;
    out.pad1 = 0.0;
    out.second_u = u2;
    out.pad2 = 0.0;
    out.second_v = v2;
    out.pad3 = 0.0;
    out.side_value = side_value;
    out.first_moment_residual = r1;
    out.second_moment_residual = r2;
    out.pad4 = 0.0;
    results[idx] = out;
}
"#;

/// One Plücker query: the two endpoints of a first segment and of a second
/// segment, the same inputs the reference
/// [`Line6::from_points`](prism_render_architecture::particle::plucker_coord::Line6::from_points)
/// consumes to build each directed line before
/// [`side`](prism_render_architecture::particle::plucker_coord::side) compares
/// them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PluckerQuery {
    /// Start endpoint of the first segment (`p0` of the first line).
    pub first_start: Vec3,
    /// End endpoint of the first segment (`p1` of the first line).
    pub first_end: Vec3,
    /// Start endpoint of the second segment (`p0` of the second line).
    pub second_start: Vec3,
    /// End endpoint of the second segment (`p1` of the second line).
    pub second_end: Vec3,
}

impl PluckerQuery {
    /// Builds a query from the two endpoints of each of the two segments.
    #[must_use]
    pub const fn new(
        first_start: Vec3,
        first_end: Vec3,
        second_start: Vec3,
        second_end: Vec3,
    ) -> PluckerQuery {
        PluckerQuery {
            first_start,
            first_end,
            second_start,
            second_end,
        }
    }
}

/// The resolved answer for one query, mirroring the reference Plücker algebra:
/// both constructed [`Line6`](prism_render_architecture::particle::plucker_coord::Line6)
/// values, the permuted inner product and the two Plücker residuals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PluckerResult {
    /// The first directed line, matching
    /// [`Line6::from_points`](prism_render_architecture::particle::plucker_coord::Line6::from_points)
    /// of the first segment.
    pub first: Line6,
    /// The second directed line, matching
    /// [`Line6::from_points`](prism_render_architecture::particle::plucker_coord::Line6::from_points)
    /// of the second segment.
    pub second: Line6,
    /// The permuted inner product, matching
    /// [`side`](prism_render_architecture::particle::plucker_coord::side).
    pub side: f32,
    /// The Plücker residual `u·v` of the first line, matching
    /// [`Line6::moment_orthogonality`](prism_render_architecture::particle::plucker_coord::Line6::moment_orthogonality).
    pub first_moment_residual: f32,
    /// The Plücker residual `u·v` of the second line, matching
    /// [`Line6::moment_orthogonality`](prism_render_architecture::particle::plucker_coord::Line6::moment_orthogonality).
    pub second_moment_residual: f32,
}

/// `repr(C)` `std430` layout of one packed query: four `vec4` slots holding
/// `(first_start.xyz, pad)`, `(first_end.xyz, pad)`, `(second_start.xyz, pad)`
/// and `(second_end.xyz, pad)` — `64` bytes, each `vec3` on its `16`-byte
/// aligned slot exactly as the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// First segment start endpoint.
    first_start: [f32; 3],
    /// Padding lane after the first start.
    pad0: f32,
    /// First segment end endpoint.
    first_end: [f32; 3],
    /// Padding lane after the first end.
    pad1: f32,
    /// Second segment start endpoint.
    second_start: [f32; 3],
    /// Padding lane after the second start.
    pad2: f32,
    /// Second segment end endpoint.
    second_end: [f32; 3],
    /// Padding lane after the second end.
    pad3: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &PluckerQuery) -> GpuQuery {
        GpuQuery {
            first_start: [
                query.first_start.x,
                query.first_start.y,
                query.first_start.z,
            ],
            pad0: 0.0,
            first_end: [query.first_end.x, query.first_end.y, query.first_end.z],
            pad1: 0.0,
            second_start: [
                query.second_start.x,
                query.second_start.y,
                query.second_start.z,
            ],
            pad2: 0.0,
            second_end: [query.second_end.x, query.second_end.y, query.second_end.z],
            pad3: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: four `vec4` slots for the first
/// direction, first moment, second direction and second moment, then a slot
/// holding `(side, first_residual, second_residual, pad)` — `80` bytes matching
/// the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// First line direction `u1`.
    first_u: [f32; 3],
    /// Padding lane after the first direction.
    pad0: f32,
    /// First line moment `v1`.
    first_v: [f32; 3],
    /// Padding lane after the first moment.
    pad1: f32,
    /// Second line direction `u2`.
    second_u: [f32; 3],
    /// Padding lane after the second direction.
    pad2: f32,
    /// Second line moment `v2`.
    second_v: [f32; 3],
    /// Padding lane after the second moment.
    pad3: f32,
    /// Permuted inner product `side`.
    side_value: f32,
    /// First line Plücker residual `u1·v1`.
    first_moment_residual: f32,
    /// Second line Plücker residual `u2·v2`.
    second_moment_residual: f32,
    /// Padding lane filling the final `vec4` slot.
    pad4: f32,
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

/// A compiled, reusable Plücker-coordinate compute pipeline.
pub struct GpuPluckerCoord {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuPluckerCoord {
    /// Compiles the Plücker-coordinate kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPluckerCoord {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_plucker_coord"),
            source: ShaderSource::Wgsl(PLUCKER_COORD_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_plucker_coord_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_plucker_coord_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_plucker_coord_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuPluckerCoord {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query on-device and returns one [`PluckerResult`] per
    /// input, in order.
    ///
    /// Each result equals the reference answers
    /// ([`Line6::from_points`](prism_render_architecture::particle::plucker_coord::Line6::from_points),
    /// [`side`](prism_render_architecture::particle::plucker_coord::side) and
    /// [`Line6::moment_orthogonality`](prism_render_architecture::particle::plucker_coord::Line6::moment_orthogonality))
    /// to within the tolerance documented on this module. An empty input
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[PluckerQuery]) -> Vec<PluckerResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_plucker_coord_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_plucker_coord_output"),
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
            label: Some("prism_volumetric_plucker_coord_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_plucker_coord_bind_group"),
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
            label: Some("prism_volumetric_plucker_coord_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_plucker_coord_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_plucker_coord_pass"),
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
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        stage.unmap();

        raw.iter().map(decode_result).collect()
    }
}

/// Decodes one packed [`GpuResult`] into the public [`PluckerResult`], rebuilding
/// each [`Line6`](prism_render_architecture::particle::plucker_coord::Line6) from
/// its direction and moment lanes.
fn decode_result(raw: &GpuResult) -> PluckerResult {
    PluckerResult {
        first: Line6::from_raw(
            Vec3::new(raw.first_u[0], raw.first_u[1], raw.first_u[2]),
            Vec3::new(raw.first_v[0], raw.first_v[1], raw.first_v[2]),
        ),
        second: Line6::from_raw(
            Vec3::new(raw.second_u[0], raw.second_u[1], raw.second_u[2]),
            Vec3::new(raw.second_v[0], raw.second_v[1], raw.second_v[2]),
        ),
        side: raw.side_value,
        first_moment_residual: raw.first_moment_residual,
        second_moment_residual: raw.second_moment_residual,
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
