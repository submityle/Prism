//! `wgpu` compute twin of the sphere-vs-plane contact from the `CPU` golden
//! `prism_physics_core::collide::primitives::sphere_plane`.
//!
//! A rigid-body narrow phase needs the single contact a sphere makes against a
//! half-space. With the plane already reduced to a world-space unit normal and
//! a world offset, this is the stateless kernel that computes it:
//!
//! ```text
//! signed      = dot(normal_world, center) - offset_world
//! penetration = radius - signed
//! valid       = penetration >= -CONTACT_TOLERANCE
//! normal      = -normal_world
//! point_a     = center - normal_world * radius
//! point_b     = center - normal_world * signed
//! ```
//!
//! # What is twinned
//!
//! The single stateless, no-`RNG` body of `sphere_plane` operating in the world
//! frame (the pose reduction `plane_world` is assumed already applied by the
//! caller, so the query carries the world normal and offset directly). The
//! manifold assembly in the golden is dropped; this kernel returns the single
//! contact point pair, the contact normal and the penetration depth, plus a
//! `valid` flag mirroring the `penetration < -CONTACT_TOLERANCE` rejection.
//!
//! # Correctness model
//!
//! Every quantity threads through multiplies, adds and one dot product, so
//! `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate. The parity test asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on every
//! continuous channel; the discrete `valid` flag is compared exactly.
//!
//! # Degenerate inputs
//!
//! A sphere clearly on the far side of the plane
//! (`penetration < -CONTACT_TOLERANCE`) is rejected: `valid = 0` and every
//! output channel is zeroed. The rejection uses an ordered compare, never an
//! equality test, so a `Metal` fast-math reassociation cannot flip it. An
//! empty query batch short-circuits on the host with no dispatch, since a
//! storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+ - * /`, a hand
//! written dot — with no `sin`, `cos`, `exp`, `log`, `pow`, no `round`, no
//! `f32` remainder, no `u64`/`i64`, no `f64`, no `sqrt` and no optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collide::primitives::sphere_plane`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` sphere-plane contact kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `evaluate`
/// mirrors the `CPU` golden `sphere_plane`; see the module documentation for
/// the algorithm.
const SPHERE_PLANE_CONTACT_WGSL: &str = r#"
// Sphere-vs-plane contact twin: one thread per query computes the single
// contact a sphere makes against a world-space half-space. With the plane given
// as a world unit normal and a world offset,
//   signed      = dot(normal_world, center) - offset_world
//   penetration = radius - signed
// and the sphere is in contact when penetration >= -CONTACT_TOLERANCE. The
// contact normal is -normal_world, the sphere-surface point is
// center - normal_world*radius and the plane point is center - normal_world*signed.
// It uses only the portable core-WGSL subset (+ - * / with a hand-written dot)
// with no u64/i64/f64, no sqrt and no transcendental, so it runs unmodified on
// Metal, Vulkan and DX12. The rejection uses an ordered compare and select, so
// a fast-math reassociation cannot flip the valid flag.
//
// Provenance: 孪生自本仓 prism_physics_core::collide::primitives::sphere_plane；无第三方引擎源码或衍生代码。

// Contact tolerance mirroring CONTACT_TOLERANCE = 1e-4 in the golden crate.
const CONTACT_TOLERANCE: f32 = 1.0e-4;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query: the sphere center, its radius, the world-space unit plane normal
// and the world plane offset. Every vector is flattened to scalar lanes so the
// std430 layout never trips a 16-byte vector-alignment rule; the kernel
// rebuilds each vec3.
struct Query {
    cx: f32,
    cy: f32,
    cz: f32,
    radius: f32,
    nx: f32,
    ny: f32,
    nz: f32,
    offset: f32,
}

// One result: the contact normal, the sphere-surface point, the plane point,
// the penetration depth and a valid flag. All vectors flattened to scalars.
struct Res {
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
    point_a_x: f32,
    point_a_y: f32,
    point_a_z: f32,
    point_b_x: f32,
    point_b_y: f32,
    point_b_z: f32,
    penetration: f32,
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let center = vec3<f32>(q.cx, q.cy, q.cz);
    let normal_world = vec3<f32>(q.nx, q.ny, q.nz);

    let signed = dot(normal_world, center) - q.offset;
    let penetration = q.radius - signed;

    // Ordered rejection: a sphere clearly on the far side is a miss.
    let hit = penetration >= -CONTACT_TOLERANCE;

    let normal = -normal_world;
    let point_a = center - normal_world * q.radius;
    let point_b = center - normal_world * signed;

    var out: Res;
    out.normal_x = select(0.0, normal.x, hit);
    out.normal_y = select(0.0, normal.y, hit);
    out.normal_z = select(0.0, normal.z, hit);
    out.point_a_x = select(0.0, point_a.x, hit);
    out.point_a_y = select(0.0, point_a.y, hit);
    out.point_a_z = select(0.0, point_a.z, hit);
    out.point_b_x = select(0.0, point_b.x, hit);
    out.point_b_y = select(0.0, point_b.y, hit);
    out.point_b_z = select(0.0, point_b.z, hit);
    out.penetration = select(0.0, penetration, hit);
    out.valid = select(0u, 1u, hit);
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SPHERE_PLANE_CONTACT_WGSL`].
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
/// The vector inputs are flattened to scalar lanes so the layout never trips a
/// `16`-byte vector-alignment rule; the kernel rebuilds each `vec3<f32>`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    cx: f32,
    cy: f32,
    cz: f32,
    radius: f32,
    nx: f32,
    ny: f32,
    nz: f32,
    offset: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Res` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
    point_a_x: f32,
    point_a_y: f32,
    point_a_z: f32,
    point_b_x: f32,
    point_b_y: f32,
    point_b_z: f32,
    penetration: f32,
    valid: u32,
}

/// One query for the sphere-plane contact twin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpherePlaneContactQuery {
    /// Sphere center in world space.
    pub center: [f32; 3],
    /// Sphere radius.
    pub radius: f32,
    /// World-space unit plane normal.
    pub normal_world: [f32; 3],
    /// World plane offset along the normal.
    pub offset_world: f32,
}

impl SpherePlaneContactQuery {
    /// Builds a query from the sphere center and radius plus the world-space
    /// plane normal and offset.
    #[must_use]
    pub fn new(
        center: [f32; 3],
        radius: f32,
        normal_world: [f32; 3],
        offset_world: f32,
    ) -> SpherePlaneContactQuery {
        SpherePlaneContactQuery {
            center,
            radius,
            normal_world,
            offset_world,
        }
    }
}

/// One resolved sphere-plane contact, mirroring the single-point manifold the
/// reference `sphere_plane` would assemble.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpherePlaneContactResult {
    /// Contact normal `-normal_world`; zeroed when there is no contact.
    pub normal: [f32; 3],
    /// Point on the sphere surface `center - normal_world * radius`; zeroed when
    /// there is no contact.
    pub point_a: [f32; 3],
    /// Point on the plane `center - normal_world * signed`; zeroed when there is
    /// no contact.
    pub point_b: [f32; 3],
    /// Penetration depth `radius - signed`; zeroed when there is no contact.
    pub penetration: f32,
    /// `1` when the sphere is in contact (`penetration >= -CONTACT_TOLERANCE`),
    /// `0` otherwise.
    pub valid: u32,
}

/// Encodes one [`SpherePlaneContactQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SpherePlaneContactQuery) -> GpuQuery {
    GpuQuery {
        cx: q.center[0],
        cy: q.center[1],
        cz: q.center[2],
        radius: q.radius,
        nx: q.normal_world[0],
        ny: q.normal_world[1],
        nz: q.normal_world[2],
        offset: q.offset_world,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SpherePlaneContactResult`].
fn decode_result(raw: &GpuResult) -> SpherePlaneContactResult {
    SpherePlaneContactResult {
        normal: [raw.normal_x, raw.normal_y, raw.normal_z],
        point_a: [raw.point_a_x, raw.point_a_y, raw.point_a_z],
        point_b: [raw.point_b_x, raw.point_b_y, raw.point_b_z],
        penetration: raw.penetration,
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

/// A compiled, reusable sphere-plane contact compute pipeline, twinning the
/// `CPU` golden `sphere_plane`.
pub struct GpuSpherePlaneContact {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSpherePlaneContact {
    /// Compiles the sphere-plane contact kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSpherePlaneContact {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sphere_plane_contact"),
            source: ShaderSource::Wgsl(SPHERE_PLANE_CONTACT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sphere_plane_contact_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sphere_plane_contact_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sphere_plane_contact_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSpherePlaneContact {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`SpherePlaneContactResult`] per input, in order.
    ///
    /// The continuous channels match the reference to within the tolerance
    /// documented on this module; the `valid` flag matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SpherePlaneContactQuery],
    ) -> Vec<SpherePlaneContactResult> {
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
            label: Some("prism_volumetric_sphere_plane_contact_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sphere_plane_contact_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sphere_plane_contact_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sphere_plane_contact_bind_group"),
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
            label: Some("prism_volumetric_sphere_plane_contact_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sphere_plane_contact_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sphere_plane_contact_pass"),
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
