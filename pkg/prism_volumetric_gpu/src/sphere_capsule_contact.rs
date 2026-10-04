//! `wgpu` compute twin of the sphere-versus-capsule narrow-phase contact kernel
//! from the `CPU` golden `prism_physics_core::collide::primitives::sphere_capsule`.
//!
//! A capsule is a swept sphere: the contact is found by projecting the sphere
//! centre onto the capsule's core segment, then treating the closest axis point
//! as the centre of a sphere of the capsule's radius. One thread resolves one
//! independent sphere/capsule pair, writing the contact normal, the two witness
//! points, the penetration depth and a validity flag.
//!
//! [`GpuSphereCapsuleContact`] is the on-device twin; a passing real-device
//! parity test is direct evidence the ported kernel reproduces the same segment
//! projection, the same penetration test, the same degenerate guards and the
//! same witness-point construction the reference computes, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces, independently of the golden crate and
//! of `glam`: the closest point on the capsule core segment
//! (`t = clamp(dot(center - seg0, ab) / dot(ab, ab), 0, 1)`, with a degenerate
//! zero-length segment collapsing to `seg0`); the signed penetration
//! `sum - distance`; the separation rejection (`penetration < -CONTACT_TOLERANCE`
//! yields `valid = 0` and all-zero output); the contact normal
//! (`delta / distance`, falling back to `(1, 0, 0)` when the sphere centre lies
//! on the axis); and the two witness points `center + normal * sphere_radius`
//! and `on_axis - normal * capsule_radius`. There is no loop: each thread runs a
//! fixed, bounded sequence of arithmetic, so the kernel provably terminates.
//!
//! # Correctness model
//!
//! The continuous channels thread through a dot product, two guarded divides, a
//! square root and several multiply-adds, so `CPU` and `GPU` evaluate the same
//! closed form but need not be bit-exact (a device may fuse a multiply-add the
//! scalar reference leaves separate). The parity test asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on every
//! continuous channel. The discrete `valid` flag is compared exactly: it is `1`
//! for a contact (touching or overlapping within tolerance) and `0` for a clean
//! separation, in which case every continuous channel is zero.
//!
//! # Degenerate inputs
//!
//! A zero-length core segment (`len_sq < GEOMETRIC_EPS`) collapses to `seg0`, so
//! the pair degenerates to sphere-versus-sphere; the segment-parameter divisor
//! is guarded with a unit fallback so the unselected arm cannot raise a `NaN` or
//! infinity. A sphere centre on the capsule axis (`distance <= GEOMETRIC_EPS`)
//! has no defined contact direction, so the normal falls back to `(1, 0, 0)`;
//! that divisor is guarded the same way. An empty query batch short-circuits on
//! the host with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `clamp`,
//! `select`, `+ - * /` — with no `sin`, `cos`, `exp`, `log`, `pow`, no `round`,
//! no float `%`, and no `u64` / `i64` / `f64`, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. Every branch is an ordered comparison feeding `select`,
//! which is robust under `Metal`'s fast-math (an `x == x` test would be folded
//! to `true`).
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collide::primitives`；无第三方
//! 引擎源码或衍生代码。
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

/// The portable core-`WGSL` sphere-versus-capsule contact kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden `sphere_capsule`; see the module docs for
/// the closed form it reproduces.
const SPHERE_CAPSULE_CONTACT_WGSL: &str = r#"
// Sphere-versus-capsule contact twin: one thread resolves one independent pair.
// It projects the sphere centre onto the capsule core segment, tests the
// penetration against the sum of radii, and emits the contact normal, two
// witness points, the penetration depth and a validity flag. It mirrors the CPU
// golden exactly and uses only the portable core subset.

struct Params {
    // Number of valid queries in the input and output buffers.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Sphere centre in world space.
    scx: f32,
    scy: f32,
    scz: f32,
    // Sphere radius.
    sphere_radius: f32,
    // Capsule core segment endpoint 0 in world space.
    s0x: f32,
    s0y: f32,
    s0z: f32,
    // Capsule core segment endpoint 1 in world space.
    s1x: f32,
    s1y: f32,
    s1z: f32,
    // Capsule radius.
    capsule_radius: f32,
}

struct Soln {
    // Contact normal.
    nx: f32,
    ny: f32,
    nz: f32,
    // Witness point on the sphere.
    pax: f32,
    pay: f32,
    paz: f32,
    // Witness point on the capsule.
    pbx: f32,
    pby: f32,
    pbz: f32,
    // Penetration depth (positive when overlapping).
    penetration: f32,
    // 1 when the pair is in contact, 0 when cleanly separated.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Soln>;

// Contact tolerance: pairs separated by more than this are rejected, matching
// the golden CONTACT_TOLERANCE.
const CONTACT_TOLERANCE: f32 = 1e-4;
// Geometric epsilon below which a length is treated as degenerate, matching the
// golden GEOMETRIC_EPS.
const GEOMETRIC_EPS: f32 = 1e-6;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let center = vec3<f32>(q.scx, q.scy, q.scz);
    let seg0 = vec3<f32>(q.s0x, q.s0y, q.s0z);
    let seg1 = vec3<f32>(q.s1x, q.s1y, q.s1z);

    // Closest point on the capsule core segment to the sphere centre.
    let ab = seg1 - seg0;
    let len_sq = dot(ab, ab);
    let seg_ok = len_sq >= GEOMETRIC_EPS;
    let safe_len_sq = select(1.0, len_sq, seg_ok);
    let raw_t = dot(center - seg0, ab) / safe_len_sq;
    let clamped_t = clamp(raw_t, 0.0, 1.0);
    // Degenerate zero-length segment collapses to seg0.
    let t_param = select(0.0, clamped_t, seg_ok);
    let on_axis = seg0 + ab * t_param;

    let delta = on_axis - center;
    let distance = sqrt(dot(delta, delta));
    let sum = q.sphere_radius + q.capsule_radius;
    let penetration = sum - distance;

    // Clean separation (beyond tolerance): no contact.
    let contact = penetration >= -CONTACT_TOLERANCE;

    // Contact normal: delta / distance, falling back to (1, 0, 0) when the
    // sphere centre lies on the axis.
    let dist_ok = distance > GEOMETRIC_EPS;
    let safe_distance = select(1.0, distance, dist_ok);
    let inv_distance = 1.0 / safe_distance;
    let fallback = vec3<f32>(1.0, 0.0, 0.0);
    let normal = select(fallback, delta * inv_distance, dist_ok);

    let point_a = center + normal * q.sphere_radius;
    let point_b = on_axis - normal * q.capsule_radius;

    var out: Soln;
    out.nx = select(0.0, normal.x, contact);
    out.ny = select(0.0, normal.y, contact);
    out.nz = select(0.0, normal.z, contact);
    out.pax = select(0.0, point_a.x, contact);
    out.pay = select(0.0, point_a.y, contact);
    out.paz = select(0.0, point_a.z, contact);
    out.pbx = select(0.0, point_b.x, contact);
    out.pby = select(0.0, point_b.y, contact);
    out.pbz = select(0.0, point_b.z, contact);
    out.penetration = select(0.0, penetration, contact);
    out.valid = select(0u, 1u, contact);
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in the kernel.
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
/// Eleven `f32` give a fixed `44`-byte stride with no trailing pad, since the
/// struct alignment is `4` and `44` is already a multiple of it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    scx: f32,
    scy: f32,
    scz: f32,
    sphere_radius: f32,
    s0x: f32,
    s0y: f32,
    s0z: f32,
    s1x: f32,
    s1y: f32,
    s1z: f32,
    capsule_radius: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Soln` struct.
/// Ten `f32` plus one `u32` give a fixed `44`-byte stride with no pad, since the
/// struct alignment is `4` and `44` is already a multiple of it.
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

/// One query for the sphere-versus-capsule contact twin: the sphere centre and
/// radius plus the capsule's core-segment endpoints and radius, all in world
/// space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereCapsuleContactQuery {
    /// `x` of the sphere centre.
    pub scx: f32,
    /// `y` of the sphere centre.
    pub scy: f32,
    /// `z` of the sphere centre.
    pub scz: f32,
    /// Sphere radius.
    pub sphere_radius: f32,
    /// `x` of capsule core-segment endpoint 0.
    pub s0x: f32,
    /// `y` of capsule core-segment endpoint 0.
    pub s0y: f32,
    /// `z` of capsule core-segment endpoint 0.
    pub s0z: f32,
    /// `x` of capsule core-segment endpoint 1.
    pub s1x: f32,
    /// `y` of capsule core-segment endpoint 1.
    pub s1y: f32,
    /// `z` of capsule core-segment endpoint 1.
    pub s1z: f32,
    /// Capsule radius.
    pub capsule_radius: f32,
}

impl SphereCapsuleContactQuery {
    /// Builds a query from the sphere (centre, radius) and the capsule (core
    /// segment endpoints, radius).
    ///
    /// The vectors are grouped into fixed-length arrays so the constructor stays
    /// within a small, readable argument count.
    #[must_use]
    pub fn new(
        sphere_center: [f32; 3],
        sphere_radius: f32,
        seg0: [f32; 3],
        seg1: [f32; 3],
        capsule_radius: f32,
    ) -> SphereCapsuleContactQuery {
        SphereCapsuleContactQuery {
            scx: sphere_center[0],
            scy: sphere_center[1],
            scz: sphere_center[2],
            sphere_radius,
            s0x: seg0[0],
            s0y: seg0[1],
            s0z: seg0[2],
            s1x: seg1[0],
            s1y: seg1[1],
            s1z: seg1[2],
            capsule_radius,
        }
    }
}

/// One resolved contact for a single query: the contact normal, the two witness
/// points, the penetration depth and the validity flag.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereCapsuleContactResult {
    /// `x` of the contact normal.
    pub nx: f32,
    /// `y` of the contact normal.
    pub ny: f32,
    /// `z` of the contact normal.
    pub nz: f32,
    /// `x` of the witness point on the sphere.
    pub pax: f32,
    /// `y` of the witness point on the sphere.
    pub pay: f32,
    /// `z` of the witness point on the sphere.
    pub paz: f32,
    /// `x` of the witness point on the capsule.
    pub pbx: f32,
    /// `y` of the witness point on the capsule.
    pub pby: f32,
    /// `z` of the witness point on the capsule.
    pub pbz: f32,
    /// Penetration depth (positive when overlapping).
    pub penetration: f32,
    /// `1` when the pair is in contact, `0` when cleanly separated.
    pub valid: u32,
}

/// Encodes one [`SphereCapsuleContactQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &SphereCapsuleContactQuery) -> GpuQuery {
    GpuQuery {
        scx: q.scx,
        scy: q.scy,
        scz: q.scz,
        sphere_radius: q.sphere_radius,
        s0x: q.s0x,
        s0y: q.s0y,
        s0z: q.s0z,
        s1x: q.s1x,
        s1y: q.s1y,
        s1z: q.s1z,
        capsule_radius: q.capsule_radius,
    }
}

/// Decodes one `std430` [`GpuResult`] slot into a public
/// [`SphereCapsuleContactResult`].
fn decode_result(r: &GpuResult) -> SphereCapsuleContactResult {
    SphereCapsuleContactResult {
        nx: r.nx,
        ny: r.ny,
        nz: r.nz,
        pax: r.pax,
        pay: r.pay,
        paz: r.paz,
        pbx: r.pbx,
        pby: r.pby,
        pbz: r.pbz,
        penetration: r.penetration,
        valid: r.valid,
    }
}

/// Builds a read-only or read-write storage-buffer bind-group-layout entry at
/// `binding`.
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

/// The on-device twin of the `CPU` golden `sphere_capsule`: a compiled compute
/// pipeline that resolves a batch of sphere-versus-capsule contacts.
pub struct GpuSphereCapsuleContact {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSphereCapsuleContact {
    /// Compiles the inline kernel and builds the compute pipeline on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSphereCapsuleContact {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sphere_capsule_contact_module"),
            source: ShaderSource::Wgsl(SPHERE_CAPSULE_CONTACT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sphere_capsule_contact_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sphere_capsule_contact_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sphere_capsule_contact_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSphereCapsuleContact {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`SphereCapsuleContactResult`] per input, in order.
    ///
    /// The continuous channels match the reference to within the tolerance
    /// documented on this module; the `valid` flag matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SphereCapsuleContactQuery],
    ) -> Vec<SphereCapsuleContactResult> {
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
            label: Some("prism_volumetric_sphere_capsule_contact_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sphere_capsule_contact_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sphere_capsule_contact_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sphere_capsule_contact_bind_group"),
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
            label: Some("prism_volumetric_sphere_capsule_contact_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sphere_capsule_contact_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sphere_capsule_contact_pass"),
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
