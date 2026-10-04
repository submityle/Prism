//! `wgpu` compute twin of the sphere-versus-oriented-box contact generator from
//! the `CPU` golden `prism_physics_core::collide::primitives`'s `point_vs_box`.
//!
//! A sphere is a point at `radius` offset, so the narrow-phase routine first
//! finds the closest point of the oriented box (`OBB`) to the sphere centre.
//! When the centre is outside the box the contact is built from that closest
//! point and the outward direction toward the sphere; when the centre is inside
//! the box the routine escapes along the nearest face. This module ports that
//! stateless, closed-form routine onto the device: one thread resolves one
//! sphere-`OBB` pair.
//!
//! # What is twinned
//!
//! * `point_vs_box` — the whole closed form. The sphere centre is projected
//!   into the box's local frame via the three box axes, clamped to the half
//!   extents, and classified inside or outside. The outside branch forms the
//!   closest point, its distance to the centre, the penetration `radius -
//!   distance`, and rejects (reports no contact) when the penetration drops
//!   below `-CONTACT_TOLERANCE`. The inside branch picks the nearest face by the
//!   smallest remaining half-extent slack, escapes outward along that face axis,
//!   and reports penetration `radius + slack`.
//!
//! # Result encoding
//!
//! The reference returns `Option<(normal, point_a, point_b, penetration)>`. The
//! twin reports the unit `normal` (from the sphere toward the box), the two
//! witness points `point_a` (on the sphere) and `point_b` (on the box), the
//! `penetration`, and a `valid` flag. `valid` is `1` when the reference returns
//! `Some` and `0` when it returns `None`; a rejected pair reports all-zero
//! payload with `valid = 0`.
//!
//! # Correctness model
//!
//! The normal, witness points and penetration are continuous quantities checked
//! with an absolute-or-relative tolerance; `valid` is discrete and compared
//! exactly. The conditioning knees are the closest-point distance crossing
//! `GEOMETRIC_EPS` (fallback normal), the penetration crossing
//! `-CONTACT_TOLERANCE` (accept or reject), and the inside-escape face selection
//! when two face slacks tie. The fixtures and sweep stay clear of those knees so
//! host and device make the same discrete choice.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — scalar `f32`
//! arithmetic, `vec3` reconstruction from flattened scalars, `dot`, `clamp`,
//! `min`, `max`, `abs`, `sqrt` and `select` — with no `sin`, `cos`, `tan`,
//! `exp`, `log`, `pow`, no `round`, no float modulo, no `u64`/`i64`/`f64` and no
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. Every `vec3` is flattened to scalar storage so no `std430` vector
//! alignment surprise can appear, and there is no bare float equality anywhere
//! in the kernel: every branch is an ordered comparison fed through `select`,
//! and the one division guards its divisor so the unselected arm never produces
//! an infinity or `NaN`.
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

/// The portable core-`WGSL` sphere-`OBB` contact kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `point_vs_box`; see the module documentation for the
/// algorithm.
const SPHERE_OBB_CONTACT_WGSL: &str = r#"
// Sphere-versus-oriented-box contact twin: one thread per pair projects the
// sphere centre into the box frame, classifies inside or outside, and builds the
// contact exactly as point_vs_box. All vectors are flattened to scalars.
// Provenance: 孪生自本仓 prism_physics_core::collide::primitives；无第三方引擎源码或衍生代码。

const CONTACT_TOLERANCE: f32 = 1.0e-4;
const GEOMETRIC_EPS: f32 = 1.0e-6;

struct Params {
    // Number of pairs in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Pair {
    // Sphere centre (x, y, z).
    scx: f32,
    scy: f32,
    scz: f32,
    // Sphere radius.
    radius: f32,
    // Box centre (x, y, z).
    bcx: f32,
    bcy: f32,
    bcz: f32,
    // Box axis 0 (unit) (x, y, z).
    a0x: f32,
    a0y: f32,
    a0z: f32,
    // Box axis 1 (unit) (x, y, z).
    a1x: f32,
    a1y: f32,
    a1z: f32,
    // Box axis 2 (unit) (x, y, z).
    a2x: f32,
    a2y: f32,
    a2z: f32,
    // Box half extents (x, y, z).
    hex: f32,
    hey: f32,
    hez: f32,
    pad0: f32,
}

struct Contact {
    // Unit contact normal, from the sphere toward the box (x, y, z).
    nx: f32,
    ny: f32,
    nz: f32,
    // Witness point on the sphere (x, y, z).
    pax: f32,
    pay: f32,
    paz: f32,
    // Witness point on the box (x, y, z).
    pbx: f32,
    pby: f32,
    pbz: f32,
    // Penetration depth.
    pen: f32,
    // 1 when the pair produces a contact, 0 when the reference rejects it.
    valid: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pairs: array<Pair>;
@group(0) @binding(2) var<storage, read_write> results: array<Contact>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = pairs[idx];

    let sc = vec3<f32>(q.scx, q.scy, q.scz);
    let bc = vec3<f32>(q.bcx, q.bcy, q.bcz);
    let a0 = vec3<f32>(q.a0x, q.a0y, q.a0z);
    let a1 = vec3<f32>(q.a1x, q.a1y, q.a1z);
    let a2 = vec3<f32>(q.a2x, q.a2y, q.a2z);
    let he = vec3<f32>(q.hex, q.hey, q.hez);
    let radius = q.radius;

    let delta = sc - bc;
    let local = vec3<f32>(dot(delta, a0), dot(delta, a1), dot(delta, a2));
    let clamped = clamp(local, -he, he);
    let inside = (abs(local.x) <= he.x) && (abs(local.y) <= he.y) && (abs(local.z) <= he.z);

    // Outside branch: build the contact from the closest point on the box.
    let closest = bc + a0 * clamped.x + a1 * clamped.y + a2 * clamped.z;
    let diff = closest - sc;
    let dist = sqrt(dot(diff, diff));
    let pen_out = radius - dist;
    let dist_ok = dist > GEOMETRIC_EPS;
    // Guard the divisor so the unselected arm never produces inf/nan.
    let safe_dist = select(1.0, dist, dist_ok);
    let normal_out = select(vec3<f32>(1.0, 0.0, 0.0), diff / safe_dist, dist_ok);
    let pa_out = sc + normal_out * radius;
    let pb_out = closest;
    let out_valid = pen_out >= -CONTACT_TOLERANCE;

    // Inside branch: escape along the nearest face.
    let fp = vec3<f32>(
        he.x - abs(local.x),
        he.y - abs(local.y),
        he.z - abs(local.z),
    );
    var axis_index = 0u;
    var min_fp = fp.x;
    if (fp.y < min_fp) {
        min_fp = fp.y;
        axis_index = 1u;
    }
    if (fp.z < min_fp) {
        min_fp = fp.z;
        axis_index = 2u;
    }
    let lc = select(select(local.x, local.y, axis_index == 1u), local.z, axis_index == 2u);
    let axis_sel = select(select(a0, a1, axis_index == 1u), a2, axis_index == 2u);
    let sign_val = select(-1.0, 1.0, lc >= 0.0);
    let outward = axis_sel * sign_val;
    let normal_in = -outward;
    let pen_in = radius + min_fp;
    let pb_in = sc + outward * min_fp;
    let pa_in = sc - outward * radius;

    // Combine the two branches, then gate on validity.
    let normal = select(normal_out, normal_in, inside);
    let pa = select(pa_out, pa_in, inside);
    let pb = select(pb_out, pb_in, inside);
    let pen = select(pen_out, pen_in, inside);
    let valid_b = select(out_valid, true, inside);

    var out: Contact;
    if (valid_b) {
        out.nx = normal.x;
        out.ny = normal.y;
        out.nz = normal.z;
        out.pax = pa.x;
        out.pay = pa.y;
        out.paz = pa.z;
        out.pbx = pb.x;
        out.pby = pb.y;
        out.pbz = pb.z;
        out.pen = pen;
        out.valid = 1u;
    } else {
        out.nx = 0.0;
        out.ny = 0.0;
        out.nz = 0.0;
        out.pax = 0.0;
        out.pay = 0.0;
        out.paz = 0.0;
        out.pbx = 0.0;
        out.pby = 0.0;
        out.pbz = 0.0;
        out.pen = 0.0;
        out.valid = 0u;
    }
    out.pad0 = 0u;

    results[idx] = out;
}
"#;

/// `repr(C)` `std430` layout of the dispatch parameters.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid pairs in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one pair, matching the `WGSL` `Pair` struct.
/// Nineteen payload words plus one padding word keep the stride a flat `80`
/// bytes, a multiple of `16`, with every `vec3` flattened to scalars so no
/// vector-alignment surprise can appear.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuPair {
    scx: f32,
    scy: f32,
    scz: f32,
    radius: f32,
    bcx: f32,
    bcy: f32,
    bcz: f32,
    a0x: f32,
    a0y: f32,
    a0z: f32,
    a1x: f32,
    a1y: f32,
    a1z: f32,
    a2x: f32,
    a2y: f32,
    a2z: f32,
    hex: f32,
    hey: f32,
    hez: f32,
    pad0: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Contact`
/// struct. Ten payload words plus the `valid` word and one padding word keep the
/// stride a flat `48` bytes, a multiple of `16`.
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
    pen: f32,
    valid: u32,
    pad0: u32,
}

/// One sphere-versus-`OBB` query: the sphere centre and radius, the box centre,
/// three orthonormal box axes and the box half extents. Every vector is
/// flattened to scalars so the `std430` stride stays an unambiguous flat layout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereObbContactQuery {
    /// Sphere centre, x component.
    pub scx: f32,
    /// Sphere centre, y component.
    pub scy: f32,
    /// Sphere centre, z component.
    pub scz: f32,
    /// Sphere radius.
    pub radius: f32,
    /// Box centre, x component.
    pub bcx: f32,
    /// Box centre, y component.
    pub bcy: f32,
    /// Box centre, z component.
    pub bcz: f32,
    /// Box axis `0`, x component.
    pub a0x: f32,
    /// Box axis `0`, y component.
    pub a0y: f32,
    /// Box axis `0`, z component.
    pub a0z: f32,
    /// Box axis `1`, x component.
    pub a1x: f32,
    /// Box axis `1`, y component.
    pub a1y: f32,
    /// Box axis `1`, z component.
    pub a1z: f32,
    /// Box axis `2`, x component.
    pub a2x: f32,
    /// Box axis `2`, y component.
    pub a2y: f32,
    /// Box axis `2`, z component.
    pub a2z: f32,
    /// Box half extent, x component.
    pub hex: f32,
    /// Box half extent, y component.
    pub hey: f32,
    /// Box half extent, z component.
    pub hez: f32,
}

impl SphereObbContactQuery {
    /// Builds a sphere-`OBB` query from the sphere centre and radius, the box
    /// centre, the three (orthonormal) box axes and the box half extents.
    #[must_use]
    pub fn new(
        sphere_center: [f32; 3],
        radius: f32,
        box_center: [f32; 3],
        axis0: [f32; 3],
        axis1: [f32; 3],
        axis2: [f32; 3],
        half_extents: [f32; 3],
    ) -> SphereObbContactQuery {
        SphereObbContactQuery {
            scx: sphere_center[0],
            scy: sphere_center[1],
            scz: sphere_center[2],
            radius,
            bcx: box_center[0],
            bcy: box_center[1],
            bcz: box_center[2],
            a0x: axis0[0],
            a0y: axis0[1],
            a0z: axis0[2],
            a1x: axis1[0],
            a1y: axis1[1],
            a1z: axis1[2],
            a2x: axis2[0],
            a2y: axis2[1],
            a2z: axis2[2],
            hex: half_extents[0],
            hey: half_extents[1],
            hez: half_extents[2],
        }
    }
}

/// One resolved sphere-`OBB` contact: the unit normal (from the sphere toward
/// the box), the two witness points, the penetration depth, and a `valid` flag
/// that is `1` when the reference returns a contact and `0` when it rejects the
/// pair (a rejected pair reports all-zero payload).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereObbContactResult {
    /// Contact normal, x component.
    pub nx: f32,
    /// Contact normal, y component.
    pub ny: f32,
    /// Contact normal, z component.
    pub nz: f32,
    /// Witness point on the sphere, x component.
    pub pax: f32,
    /// Witness point on the sphere, y component.
    pub pay: f32,
    /// Witness point on the sphere, z component.
    pub paz: f32,
    /// Witness point on the box, x component.
    pub pbx: f32,
    /// Witness point on the box, y component.
    pub pby: f32,
    /// Witness point on the box, z component.
    pub pbz: f32,
    /// Penetration depth.
    pub pen: f32,
    /// `1` when a contact is produced, `0` when the pair is rejected.
    pub valid: u32,
}

/// Encodes one [`SphereObbContactQuery`] into its `std430` [`GpuPair`].
fn encode_query(q: &SphereObbContactQuery) -> GpuPair {
    GpuPair {
        scx: q.scx,
        scy: q.scy,
        scz: q.scz,
        radius: q.radius,
        bcx: q.bcx,
        bcy: q.bcy,
        bcz: q.bcz,
        a0x: q.a0x,
        a0y: q.a0y,
        a0z: q.a0z,
        a1x: q.a1x,
        a1y: q.a1y,
        a1z: q.a1z,
        a2x: q.a2x,
        a2y: q.a2y,
        a2z: q.a2z,
        hex: q.hex,
        hey: q.hey,
        hez: q.hez,
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SphereObbContactResult`].
fn decode_result(raw: &GpuResult) -> SphereObbContactResult {
    SphereObbContactResult {
        nx: raw.nx,
        ny: raw.ny,
        nz: raw.nz,
        pax: raw.pax,
        pay: raw.pay,
        paz: raw.paz,
        pbx: raw.pbx,
        pby: raw.pby,
        pbz: raw.pbz,
        pen: raw.pen,
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

/// A compiled, reusable sphere-`OBB` contact compute pipeline, twinning the
/// `CPU` golden `point_vs_box`.
pub struct GpuSphereObbContact {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSphereObbContact {
    /// Compiles the sphere-`OBB` contact kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSphereObbContact {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sphere_obb_contact"),
            source: ShaderSource::Wgsl(SPHERE_OBB_CONTACT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sphere_obb_contact_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sphere_obb_contact_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sphere_obb_contact_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSphereObbContact {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`SphereObbContactResult`] per input, in order.
    ///
    /// An empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SphereObbContactQuery],
    ) -> Vec<SphereObbContactResult> {
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
            label: Some("prism_volumetric_sphere_obb_contact_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuPair> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sphere_obb_contact_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sphere_obb_contact_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sphere_obb_contact_bind_group"),
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
            label: Some("prism_volumetric_sphere_obb_contact_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sphere_obb_contact_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sphere_obb_contact_pass"),
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
