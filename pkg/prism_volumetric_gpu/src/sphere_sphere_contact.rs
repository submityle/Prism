//! `wgpu` compute twin of the sphere-sphere narrow-phase contact helper from
//! the `CPU` golden
//! `prism_physics_core::collide::primitives::sphere_sphere`.
//!
//! Two spheres given by their world-space centers and radii either touch,
//! overlap, or are separated. The golden computes the center-to-center vector,
//! its length, the sum of radii and the penetration depth, rejects the pair
//! when the gap exceeds the contact tolerance, and otherwise emits a contact
//! normal (pointing from `a` toward `b`) with the two surface witness points.
//! This module ports that stateless, no-`RNG` closed form onto the device: one
//! compute thread resolves one pair, so a passing real-device parity test is
//! direct evidence the kernel reproduces the same branch structure and surface
//! arithmetic, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each query is one pair: `center_a`, `radius_a`, `center_b`, `radius_b`. For
//! each pair the kernel reproduces the reference closed form exactly:
//!
//! * `delta = center_b - center_a`, `distance = length(delta)`;
//! * `penetration = (radius_a + radius_b) - distance`;
//! * if `penetration < -CONTACT_TOLERANCE` the pair is rejected (`valid = 0`,
//!   all outputs zeroed);
//! * otherwise `normal = delta / distance` when `distance > GEOMETRIC_EPS`,
//!   else the fallback `(1, 0, 0)`;
//! * `point_a = center_a + normal * radius_a`,
//!   `point_b = center_b - normal * radius_b`.
//!
//! # Correctness model
//!
//! The witness points thread through a few multiply-adds, so `CPU` and `GPU`
//! are not required to be bit-exact. The parity test asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on each
//! continuous component; the discrete `valid` flag is compared exactly. Every
//! device-side branch uses an ordered comparison feeding `select`, never a bare
//! float equality or a fast-math `NaN` sentinel, and the normalizing division
//! is guarded so the unselected arm cannot produce an infinity or `NaN`.
//!
//! # Degenerate inputs
//!
//! A separated pair (`penetration < -CONTACT_TOLERANCE`) yields `valid = 0`
//! with the normal, both witness points and the penetration all zeroed. A
//! coincident pair (`distance <= GEOMETRIC_EPS`) is still a valid contact and
//! uses the fallback normal `(1, 0, 0)`. An empty query batch short-circuits on
//! the host with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+ - * /`, `sqrt`,
//! `length`, `select` and unsigned index arithmetic — with no `sin`, `cos`,
//! `tan`, `exp`, `log`, `pow`, no `round`, no float modulo and no optional
//! device feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collide::primitives::sphere_sphere`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` sphere-sphere contact kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `project`
/// mirrors the `CPU` golden `sphere_sphere`; see the module documentation for
/// the algorithm.
const SPHERE_SPHERE_CONTACT_WGSL: &str = r#"
// Sphere-sphere contact twin: one thread per pair reproduces the manifold the
// golden sphere_sphere builds from two world-space centers and radii. It
// mirrors the CPU golden operation for operation, uses only the portable
// core-WGSL subset (+ - * /, sqrt, length, select plus unsigned index math),
// takes no optional feature, and has no loop, so the kernel provably
// terminates. All degenerate branches use ordered comparisons feeding select;
// the normalizing division denominator is guarded so the unselected arm never
// produces an infinity or NaN.
//
// Provenance: 孪生自本仓
// prism_physics_core::collide::primitives::sphere_sphere；
// 无第三方引擎源码或衍生代码。

const CONTACT_TOLERANCE: f32 = 1.0e-4;
const GEOMETRIC_EPS: f32 = 1.0e-6;

struct Params {
    // Number of pairs in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // World-space center of sphere a.
    cax: f32,
    cay: f32,
    caz: f32,
    // Radius of sphere a.
    ra: f32,
    // World-space center of sphere b.
    cbx: f32,
    cby: f32,
    cbz: f32,
    // Radius of sphere b.
    rb: f32,
}

struct Result {
    // Contact normal, pointing from a toward b.
    nx: f32,
    ny: f32,
    nz: f32,
    // Surface witness point on sphere a.
    pax: f32,
    pay: f32,
    paz: f32,
    // Surface witness point on sphere b.
    pbx: f32,
    pby: f32,
    pbz: f32,
    // Penetration depth (positive when overlapping).
    penetration: f32,
    // 1 when the pair forms a contact, 0 when separated beyond tolerance.
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

    let center_a = vec3<f32>(q.cax, q.cay, q.caz);
    let center_b = vec3<f32>(q.cbx, q.cby, q.cbz);
    let delta = center_b - center_a;
    let dist = length(delta);
    let sum_r = q.ra + q.rb;
    let pen = sum_r - dist;

    // Contact exists unless the pair is separated beyond the tolerance.
    let is_contact = pen >= -CONTACT_TOLERANCE;

    // Normalize delta when the centers are meaningfully apart, else fall back to
    // (1, 0, 0). The denominator is guarded so the unselected arm is finite.
    let dist_ok = dist > GEOMETRIC_EPS;
    let safe_dist = select(1.0, dist, dist_ok);
    let normalized = delta / safe_dist;
    let fallback = vec3<f32>(1.0, 0.0, 0.0);
    let normal = select(fallback, normalized, dist_ok);

    let point_a = center_a + normal * q.ra;
    let point_b = center_b - normal * q.rb;

    let zero3 = vec3<f32>(0.0, 0.0, 0.0);
    let out_normal = select(zero3, normal, is_contact);
    let out_pa = select(zero3, point_a, is_contact);
    let out_pb = select(zero3, point_b, is_contact);
    let out_pen = select(0.0, pen, is_contact);

    var out: Result;
    out.nx = out_normal.x;
    out.ny = out_normal.y;
    out.nz = out_normal.z;
    out.pax = out_pa.x;
    out.pay = out_pa.y;
    out.paz = out_pa.z;
    out.pbx = out_pb.x;
    out.pby = out_pb.y;
    out.pbz = out_pb.z;
    out.penetration = out_pen;
    out.valid = select(0u, 1u, is_contact);
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SPHERE_SPHERE_CONTACT_WGSL`].
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
/// All components are scalar `f32` so the slot contains no `vec3` and the host
/// and device agree on the array stride byte for byte.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    cax: f32,
    cay: f32,
    caz: f32,
    ra: f32,
    cbx: f32,
    cby: f32,
    cbz: f32,
    rb: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct. The three vectors are stored as flat scalars; the trailing `valid`
/// word keeps the discrete flag beside the continuous quantities.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    nx: f32,
    ny: f32,
    nz: f32,
    pax: f32,
    pay: f32,
    paz: f32,
    pbx: f32,
    pby: f32,
    pbz: f32,
    penetration: f32,
    valid: u32,
}

/// One query for the sphere-sphere contact twin: two world-space sphere centers
/// with their radii.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereSphereContactQuery {
    /// World-space center of sphere `a`.
    pub center_a: [f32; 3],
    /// Radius of sphere `a`.
    pub radius_a: f32,
    /// World-space center of sphere `b`.
    pub center_b: [f32; 3],
    /// Radius of sphere `b`.
    pub radius_b: f32,
}

impl SphereSphereContactQuery {
    /// Builds a query from the two sphere centers and radii.
    #[must_use]
    pub fn new(
        center_a: [f32; 3],
        radius_a: f32,
        center_b: [f32; 3],
        radius_b: f32,
    ) -> SphereSphereContactQuery {
        SphereSphereContactQuery {
            center_a,
            radius_a,
            center_b,
            radius_b,
        }
    }
}

/// One resolved answer for a single pair: the contact normal (from `a` toward
/// `b`), the two surface witness points, the penetration depth and the `valid`
/// flag. When `valid` is `0` the pair is separated beyond tolerance and every
/// continuous field is zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereSphereContactResult {
    /// Contact normal, pointing from `a` toward `b`.
    pub normal: [f32; 3],
    /// Surface witness point on sphere `a`.
    pub point_a: [f32; 3],
    /// Surface witness point on sphere `b`.
    pub point_b: [f32; 3],
    /// Penetration depth (positive when overlapping).
    pub penetration: f32,
    /// `1` when the pair forms a contact, `0` when separated beyond tolerance.
    pub valid: u32,
}

/// Encodes one [`SphereSphereContactQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SphereSphereContactQuery) -> GpuQuery {
    GpuQuery {
        cax: q.center_a[0],
        cay: q.center_a[1],
        caz: q.center_a[2],
        ra: q.radius_a,
        cbx: q.center_b[0],
        cby: q.center_b[1],
        cbz: q.center_b[2],
        rb: q.radius_b,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`SphereSphereContactResult`].
fn decode_result(raw: &GpuResult) -> SphereSphereContactResult {
    SphereSphereContactResult {
        normal: [raw.nx, raw.ny, raw.nz],
        point_a: [raw.pax, raw.pay, raw.paz],
        point_b: [raw.pbx, raw.pby, raw.pbz],
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

/// A compiled, reusable sphere-sphere contact compute pipeline, twinning the
/// `CPU` golden `sphere_sphere`.
pub struct GpuSphereSphereContact {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSphereSphereContact {
    /// Compiles the sphere-sphere contact kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSphereSphereContact {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sphere_sphere_contact"),
            source: ShaderSource::Wgsl(SPHERE_SPHERE_CONTACT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sphere_sphere_contact_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sphere_sphere_contact_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sphere_sphere_contact_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("project"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSphereSphereContact {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every pair in `queries` and returns one
    /// [`SphereSphereContactResult`] per input, in order.
    ///
    /// Each continuous component matches the reference to within the tolerance
    /// documented on this module; the `valid` flag matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SphereSphereContactQuery],
    ) -> Vec<SphereSphereContactResult> {
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
            label: Some("prism_volumetric_sphere_sphere_contact_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sphere_sphere_contact_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sphere_sphere_contact_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sphere_sphere_contact_bind_group"),
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
            label: Some("prism_volumetric_sphere_sphere_contact_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sphere_sphere_contact_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sphere_sphere_contact_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per pair, flattened to a 1-D dispatch.
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
