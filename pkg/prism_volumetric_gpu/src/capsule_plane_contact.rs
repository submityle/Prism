//! `wgpu` compute twin of the capsule-vs-plane contact generator from the `CPU`
//! golden `prism_physics_core::collide::primitives::capsule_plane`.
//!
//! A capsule is a line segment inflated by a radius; against a plane half-space
//! each of the segment's two endpoints can seat a contact point. For an
//! endpoint `e`, the signed distance to the plane is
//! `signed = dot(normal_world, e) - offset_world`, the penetration is
//! `pen = radius - signed`, and when `pen >= -CONTACT_TOLERANCE` the endpoint
//! contributes a manifold point whose witness pair is
//! `point_a = e - normal_world * radius` (on the capsule surface) and
//! `point_b = e - normal_world * signed` (on the plane). This module bakes the
//! poses into world space on the host and ports the per-endpoint test onto the
//! device: one thread resolves one capsule, so a passing real-device parity test
//! is direct evidence the kernel reproduces the same signed distances, the same
//! penetrations and the same witness points the reference does.
//!
//! # What is twinned
//!
//! The world-space body of `capsule_plane`: the shared manifold normal
//! `normal = -normal_world`, and for each of the two endpoints the signed
//! distance, penetration, admissibility test and witness pair. The golden
//! collects only admissible points and returns `None` when neither endpoint is
//! in contact; this twin reports the two endpoints independently with a `valid`
//! word each, so the caller reconstructs "no manifold" as
//! `valid0 == 0 && valid1 == 0`.
//!
//! # Correctness model
//!
//! The witness points (`point_a`, `point_b`) and penetrations are continuous
//! `f32`, compared with an absolute-or-relative tolerance; the `valid` words are
//! discrete and compared exactly. An endpoint that fails the admissibility test
//! (`pen < -CONTACT_TOLERANCE`) reports `valid = 0` and zeroes its witness
//! points and penetration, exactly as the host oracle does.
//!
//! The admissibility test is an ordered compare (`pen >= -CONTACT_TOLERANCE`)
//! and the outputs are chosen with `select`, so there is no bare `f32` equality
//! anywhere in the kernel and no `Metal` fast-math driver can fold a guard away.
//! The closed form has no division, so no divisor guard is needed.
//!
//! # Degenerate inputs
//!
//! A capsule fully separated from the plane reports `valid0 == 0` and
//! `valid1 == 0` (the golden's `None`). An empty query batch short-circuits on
//! the host with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — ordered compares,
//! `select`, and `+ - *` on scalar `f32` — with no `sin`, `cos`, `tan`, `exp`,
//! `log`, `pow`, no `round`, no `f32` remainder, no bare `f32` equality and no
//! `u64`/`i64`/`f64`, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//! Every vector is flattened to scalar `f32` lanes in the storage buffers and
//! recomposed by hand, so no vector alignment rule can perturb the `std430`
//! stride.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collide::primitives`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` capsule-plane contact kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the world-space body of the `CPU` golden `capsule_plane`; see the module
/// documentation for the algorithm.
const CAPSULE_PLANE_CONTACT_WGSL: &str = r#"
// Capsule-plane contact twin: one thread per query reproduces the world-space
// body of capsule_plane. For each of the two segment endpoints it computes the
// signed distance to the plane, the penetration radius - signed, and when the
// penetration clears -CONTACT_TOLERANCE emits a witness pair (capsule-surface
// point, plane point). It uses only the portable core-WGSL subset (ordered
// compares, select, + - * on scalar f32), has no division and no loop.

const CONTACT_TOLERANCE: f32 = 1.0e-4;

struct Params {
    // Number of valid queries in this dispatch.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Capsule segment endpoint 0 (world space).
    seg0x: f32,
    seg0y: f32,
    seg0z: f32,
    // Capsule segment endpoint 1 (world space).
    seg1x: f32,
    seg1y: f32,
    seg1z: f32,
    // Capsule radius.
    radius: f32,
    // Plane normal (world space, unit length).
    nx: f32,
    ny: f32,
    nz: f32,
    // Plane offset along the normal (world space).
    plane_offset: f32,
}

struct ContactResult {
    // Shared manifold normal = -normal_world.
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
    // Endpoint 0 witness pair and penetration.
    pa0x: f32,
    pa0y: f32,
    pa0z: f32,
    pb0x: f32,
    pb0y: f32,
    pb0z: f32,
    pen0: f32,
    valid0: u32,
    // Endpoint 1 witness pair and penetration.
    pa1x: f32,
    pa1y: f32,
    pa1z: f32,
    pb1x: f32,
    pb1y: f32,
    pb1z: f32,
    pen1: f32,
    valid1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<ContactResult>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let seg0 = vec3<f32>(q.seg0x, q.seg0y, q.seg0z);
    let seg1 = vec3<f32>(q.seg1x, q.seg1y, q.seg1z);
    let normal_world = vec3<f32>(q.nx, q.ny, q.nz);
    let plane_normal = -normal_world;

    // Endpoint 0.
    let signed0 = dot(normal_world, seg0) - q.plane_offset;
    let penetration0 = q.radius - signed0;
    let hit0 = penetration0 >= -CONTACT_TOLERANCE;
    let pa0 = seg0 - normal_world * q.radius;
    let pb0 = seg0 - normal_world * signed0;
    let zero = vec3<f32>(0.0, 0.0, 0.0);
    let out_pa0 = select(zero, pa0, hit0);
    let out_pb0 = select(zero, pb0, hit0);
    let out_pen0 = select(0.0, penetration0, hit0);

    // Endpoint 1.
    let signed1 = dot(normal_world, seg1) - q.plane_offset;
    let penetration1 = q.radius - signed1;
    let hit1 = penetration1 >= -CONTACT_TOLERANCE;
    let pa1 = seg1 - normal_world * q.radius;
    let pb1 = seg1 - normal_world * signed1;
    let out_pa1 = select(zero, pa1, hit1);
    let out_pb1 = select(zero, pb1, hit1);
    let out_pen1 = select(0.0, penetration1, hit1);

    var res: ContactResult;
    res.normal_x = plane_normal.x;
    res.normal_y = plane_normal.y;
    res.normal_z = plane_normal.z;
    res.pa0x = out_pa0.x;
    res.pa0y = out_pa0.y;
    res.pa0z = out_pa0.z;
    res.pb0x = out_pb0.x;
    res.pb0y = out_pb0.y;
    res.pb0z = out_pb0.z;
    res.pen0 = out_pen0;
    res.valid0 = select(0u, 1u, hit0);
    res.pa1x = out_pa1.x;
    res.pa1y = out_pa1.y;
    res.pa1z = out_pa1.z;
    res.pb1x = out_pb1.x;
    res.pb1y = out_pb1.y;
    res.pb1z = out_pb1.z;
    res.pen1 = out_pen1;
    res.valid1 = select(0u, 1u, hit1);
    results[idx] = res;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`CAPSULE_PLANE_CONTACT_WGSL`].
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
/// All eleven lanes are scalar `f32`, so the layout is a flat `44`-byte stride
/// with alignment `4` and no internal padding, and a batch of two or more packs
/// contiguously with no vector alignment rule to trip.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    seg0x: f32,
    seg0y: f32,
    seg0z: f32,
    seg1x: f32,
    seg1y: f32,
    seg1z: f32,
    radius: f32,
    nx: f32,
    ny: f32,
    nz: f32,
    plane_offset: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `ContactResult`
/// struct: a shared normal (three `f32`), then two endpoints each with a witness
/// pair (`point_a`, `point_b`), a penetration and a `valid` word. Nineteen
/// scalar lanes give a flat `76`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
    pa0x: f32,
    pa0y: f32,
    pa0z: f32,
    pb0x: f32,
    pb0y: f32,
    pb0z: f32,
    pen0: f32,
    valid0: u32,
    pa1x: f32,
    pa1y: f32,
    pa1z: f32,
    pb1x: f32,
    pb1y: f32,
    pb1z: f32,
    pen1: f32,
    valid1: u32,
}

/// One query for the capsule-plane contact twin: the capsule's world-space
/// segment endpoints (`seg0*`, `seg1*`), its `radius`, and the world-space plane
/// unit normal (`nx`, `ny`, `nz`) and `plane_offset`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CapsulePlaneContactQuery {
    /// Segment endpoint 0, x component.
    pub seg0x: f32,
    /// Segment endpoint 0, y component.
    pub seg0y: f32,
    /// Segment endpoint 0, z component.
    pub seg0z: f32,
    /// Segment endpoint 1, x component.
    pub seg1x: f32,
    /// Segment endpoint 1, y component.
    pub seg1y: f32,
    /// Segment endpoint 1, z component.
    pub seg1z: f32,
    /// Capsule radius.
    pub radius: f32,
    /// Plane unit normal, x component.
    pub nx: f32,
    /// Plane unit normal, y component.
    pub ny: f32,
    /// Plane unit normal, z component.
    pub nz: f32,
    /// Plane offset along the normal.
    pub plane_offset: f32,
}

impl CapsulePlaneContactQuery {
    /// Builds a query from the two segment endpoints, radius, plane normal and
    /// plane offset, in field order.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the query mirrors the flat scalar std430 layout one lane at a time"
    )]
    pub fn new(
        seg0x: f32,
        seg0y: f32,
        seg0z: f32,
        seg1x: f32,
        seg1y: f32,
        seg1z: f32,
        radius: f32,
        nx: f32,
        ny: f32,
        nz: f32,
        plane_offset: f32,
    ) -> CapsulePlaneContactQuery {
        CapsulePlaneContactQuery {
            seg0x,
            seg0y,
            seg0z,
            seg1x,
            seg1y,
            seg1z,
            radius,
            nx,
            ny,
            nz,
            plane_offset,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `capsule_plane` world-space manifold.
///
/// (`normal_x`, `normal_y`, `normal_z`) is the shared manifold normal
/// (`-normal_world`). Each endpoint reports a witness pair — `pa*` on the
/// capsule surface, `pb*` on the plane — a penetration `pen*` and a `valid*`
/// word (`1` when the endpoint is in contact, `0` otherwise, in which case its
/// witness points and penetration are zero). Both `valid` words zero means the
/// golden returned no manifold (`None`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CapsulePlaneContactResult {
    /// Shared manifold normal, x component.
    pub normal_x: f32,
    /// Shared manifold normal, y component.
    pub normal_y: f32,
    /// Shared manifold normal, z component.
    pub normal_z: f32,
    /// Endpoint 0 capsule-surface witness, x component.
    pub pa0x: f32,
    /// Endpoint 0 capsule-surface witness, y component.
    pub pa0y: f32,
    /// Endpoint 0 capsule-surface witness, z component.
    pub pa0z: f32,
    /// Endpoint 0 plane witness, x component.
    pub pb0x: f32,
    /// Endpoint 0 plane witness, y component.
    pub pb0y: f32,
    /// Endpoint 0 plane witness, z component.
    pub pb0z: f32,
    /// Endpoint 0 penetration.
    pub pen0: f32,
    /// `1` when endpoint 0 is in contact, `0` otherwise.
    pub valid0: u32,
    /// Endpoint 1 capsule-surface witness, x component.
    pub pa1x: f32,
    /// Endpoint 1 capsule-surface witness, y component.
    pub pa1y: f32,
    /// Endpoint 1 capsule-surface witness, z component.
    pub pa1z: f32,
    /// Endpoint 1 plane witness, x component.
    pub pb1x: f32,
    /// Endpoint 1 plane witness, y component.
    pub pb1y: f32,
    /// Endpoint 1 plane witness, z component.
    pub pb1z: f32,
    /// Endpoint 1 penetration.
    pub pen1: f32,
    /// `1` when endpoint 1 is in contact, `0` otherwise.
    pub valid1: u32,
}

/// Encodes one [`CapsulePlaneContactQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &CapsulePlaneContactQuery) -> GpuQuery {
    GpuQuery {
        seg0x: q.seg0x,
        seg0y: q.seg0y,
        seg0z: q.seg0z,
        seg1x: q.seg1x,
        seg1y: q.seg1y,
        seg1z: q.seg1z,
        radius: q.radius,
        nx: q.nx,
        ny: q.ny,
        nz: q.nz,
        plane_offset: q.plane_offset,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`CapsulePlaneContactResult`].
fn decode_result(raw: &GpuResult) -> CapsulePlaneContactResult {
    CapsulePlaneContactResult {
        normal_x: raw.normal_x,
        normal_y: raw.normal_y,
        normal_z: raw.normal_z,
        pa0x: raw.pa0x,
        pa0y: raw.pa0y,
        pa0z: raw.pa0z,
        pb0x: raw.pb0x,
        pb0y: raw.pb0y,
        pb0z: raw.pb0z,
        pen0: raw.pen0,
        valid0: raw.valid0,
        pa1x: raw.pa1x,
        pa1y: raw.pa1y,
        pa1z: raw.pa1z,
        pb1x: raw.pb1x,
        pb1y: raw.pb1y,
        pb1z: raw.pb1z,
        pen1: raw.pen1,
        valid1: raw.valid1,
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

/// A compiled, reusable capsule-plane contact compute pipeline, twinning the
/// `CPU` golden `capsule_plane`.
pub struct GpuCapsulePlaneContact {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCapsulePlaneContact {
    /// Compiles the capsule-plane contact kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCapsulePlaneContact {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_capsule_plane_contact"),
            source: ShaderSource::Wgsl(CAPSULE_PLANE_CONTACT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_capsule_plane_contact_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_capsule_plane_contact_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_capsule_plane_contact_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCapsulePlaneContact {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`CapsulePlaneContactResult`] per input, in order.
    ///
    /// An empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[CapsulePlaneContactQuery],
    ) -> Vec<CapsulePlaneContactResult> {
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
            label: Some("prism_volumetric_capsule_plane_contact_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_capsule_plane_contact_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_capsule_plane_contact_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_capsule_plane_contact_bind_group"),
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
            label: Some("prism_volumetric_capsule_plane_contact_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_capsule_plane_contact_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_capsule_plane_contact_pass"),
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
