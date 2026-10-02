//! `wgpu` compute twin of the particle<->rigid-body coupling *resolve* step
//! ([`two_way_coupling`](prism_render_architecture::particle::two_way_coupling),
//! particle design §10).
//!
//! Where the sibling
//! [`coupling_impulse`](crate::coupling_impulse) twin reproduces the pure
//! per-contact *measurements* (the impulse the body applies to the particle and
//! the quantities feeding it), this twin reproduces the *state update*: the
//! golden
//! [`resolve_coupling`](prism_render_architecture::particle::two_way_coupling::resolve_coupling),
//! which is
//! [`coupling_impulse`](prism_render_architecture::particle::two_way_coupling::coupling_impulse)
//! followed by
//! [`apply_coupling`](prism_render_architecture::particle::two_way_coupling::apply_coupling).
//! Each contact gains the impulse `J = j * n` on the particle and the equal-and-
//! opposite reaction `-J` on the body at the same world point, so the closed
//! particle+body system conserves linear momentum.
//!
//! [`GpuCouplingApplyResolve`] is the on-device twin: one thread per contact
//! takes a full particle+body state, computes the contact impulse branch for
//! branch with the reference (its three zero-impulse identities included), then
//! applies the impulse and its reaction to a private copy of the two bodies,
//! returning the impulse together with the updated particle and body
//! velocities. A passing real-device parity test is direct evidence the ported
//! kernel resolves the same contact and mutates the same state the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For every contact the kernel reproduces the four values the reference's
//! resolve step produces: the contact `impulse`, the updated particle linear
//! velocity, and the updated body linear and angular velocities. Each contact
//! owns a private copy of its particle and body, so there is no cross-contact
//! accumulation and the pass is embarrassingly parallel — matching the
//! one-thread-per-contact contract. The system-momentum bookkeeping the
//! reference's tests perform by summing over many contacts into one shared body
//! is a host-side reduction outside this per-contact kernel and is deliberately
//! **not** twinned here.
//!
//! # The update
//!
//! With `r = particle.position - center_of_mass` and `reaction = -impulse`:
//!
//! - the particle velocity gains `impulse * particle.inv_mass`,
//! - the body linear velocity gains `reaction * body.inv_mass`,
//! - the body angular velocity gains
//!   `inv_inertia_world * (r × reaction)`.
//!
//! The world-space inverse inertia tensor arrives already assembled on the body
//! (as it does on the golden [`CouplingBody`](prism_render_architecture::particle::two_way_coupling::CouplingBody)),
//! so the kernel only has to apply it; it does not rebuild `R * diag * Rᵀ` here.
//!
//! # Degenerate regimes
//!
//! The impulse mirrors the reference's three zero-impulse identities branch for
//! branch: a degenerate (near-zero-length) contact normal
//! (`length² <= EPS_LEN_SQ`), a *separating* contact (relative normal velocity
//! `>= 0`), and a jointly immovable pair (effective inverse mass at or below
//! `MIN_EFFECTIVE_INV_MASS`). When the impulse is the zero vector the apply step
//! leaves every velocity unchanged (the reaction and its cross product are also
//! zero), exactly as the reference does. The parity fixtures stay clear of these
//! branch thresholds for the live-solve case by rejection sampling and hit each
//! one exactly for the degenerate cases.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `dot`, `cross`,
//! `clamp`, `sqrt` (reached only through the guarded normalize, exactly as the
//! reference's [`Vec3`](prism_render_architecture::particle::Vec3) normalization
//! does) and `+ - * /` — with no transcendental call, no `u64` and no optional
//! device feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. There
//! is no loop: each thread performs a fixed sequence of operations.
//!
//! # Correctness model
//!
//! Each contact is a fixed, non-reorderable sequence of multiplies, adds and
//! divides, so `CPU` and `GPU` evaluate the same closed form in the same order.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32` fields.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::two_way_coupling`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::Vec3;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;
use crate::coupling_impulse::GpuMat3;

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// Source of the single-entry coupling resolve kernel.
///
/// The twin inlines its `WGSL` rather than shipping an external `.wgsl` so it
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`resolve_coupling`](prism_render_architecture::particle::two_way_coupling::resolve_coupling);
/// see the module documentation for the algebra.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::two_way_coupling`。
const COUPLING_APPLY_RESOLVE_WGSL: &str = r#"
// coupling_apply_resolve twin: one thread per contact reproduces the CPU golden
// `particle::two_way_coupling::resolve_coupling` — compute the contact impulse
// (branch for branch with `coupling_impulse`, including its three zero-impulse
// identities) then apply it and its reaction to a private copy of the particle
// and body. It uses only the portable core-WGSL subset (dot/cross/clamp/sqrt
// and + - * /), takes no optional feature, and runs unmodified on Metal, Vulkan
// and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::two_way_coupling;
// 无第三方引擎源码或衍生代码。

// Squared-length floor below which a vector is treated as the zero vector so a
// normalize never yields NaN. Matches the reference `EPS_LEN_SQ`.
const EPS_LEN_SQ: f32 = 1.0e-12;
// Effective-inverse-mass floor below which a particle+body pair is jointly
// immovable and the impulse divide collapses to zero. Matches the reference
// `MIN_EFFECTIVE_INV_MASS`.
const MIN_EFFECTIVE_INV_MASS: f32 = 1.0e-12;

struct Params {
    // Number of contacts in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Particle contact point (world); a pad lane follows.
    particle_position: vec3<f32>,
    pad0: f32,
    // Particle linear velocity; a pad lane follows.
    particle_velocity: vec3<f32>,
    pad1: f32,
    // Body center of mass (lever-arm origin); a pad lane follows.
    body_center_of_mass: vec3<f32>,
    pad2: f32,
    // Body linear velocity; a pad lane follows.
    body_linear_velocity: vec3<f32>,
    pad3: f32,
    // Body angular velocity; a pad lane follows.
    body_angular_velocity: vec3<f32>,
    pad4: f32,
    // World inverse inertia tensor column x; a pad lane follows.
    inv_inertia_col_x: vec3<f32>,
    pad5: f32,
    // World inverse inertia tensor column y; a pad lane follows.
    inv_inertia_col_y: vec3<f32>,
    pad6: f32,
    // World inverse inertia tensor column z; a pad lane follows.
    inv_inertia_col_z: vec3<f32>,
    pad7: f32,
    // Contact normal (body -> particle, not necessarily unit); a pad follows.
    normal: vec3<f32>,
    pad8: f32,
    // Scalar block, one vec4 slot: particle inverse mass, body inverse mass,
    // restitution, and a trailing pad.
    particle_inv_mass: f32,
    body_inv_mass: f32,
    restitution: f32,
    pad9: f32,
}

struct Result {
    // Contact impulse applied to the particle; a pad lane follows.
    impulse: vec3<f32>,
    pad0: f32,
    // Updated particle linear velocity; a pad lane follows.
    particle_velocity: vec3<f32>,
    pad1: f32,
    // Updated body linear velocity; a pad lane follows.
    body_linear_velocity: vec3<f32>,
    pad2: f32,
    // Updated body angular velocity; a pad lane follows.
    body_angular_velocity: vec3<f32>,
    pad3: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Column-major matrix-vector product cols * v.
fn mat_mul_vec(cx: vec3<f32>, cy: vec3<f32>, cz: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    return cx * v.x + cy * v.y + cz * v.z;
}

// Unit vector along v, or the zero vector when v is (numerically) zero, exactly
// mirroring the reference `Vec3::normalize_or_zero`.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq > EPS_LEN_SQ) {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Lever arm from the body center of mass to the contact (particle) point.
    let contact = q.particle_position;
    let r = contact - q.body_center_of_mass;

    // body_point_velocity at the contact = v_linear + omega × r.
    let body_vel = q.body_linear_velocity + cross(q.body_angular_velocity, r);

    // coupling_impulse, mirroring the reference's three zero-impulse identities.
    let n = normalize_or_zero(q.normal);
    let rn = cross(r, n);
    let i_inv_rn = mat_mul_vec(q.inv_inertia_col_x, q.inv_inertia_col_y, q.inv_inertia_col_z, rn);
    let gen_inv_mass = q.body_inv_mass + dot(rn, i_inv_rn);

    var impulse = vec3<f32>(0.0, 0.0, 0.0);
    if (dot(q.normal, q.normal) > EPS_LEN_SQ) {
        let relative = q.particle_velocity - body_vel;
        let vn = dot(relative, n);
        if (vn < 0.0) {
            let effective = q.particle_inv_mass + gen_inv_mass;
            if (effective > MIN_EFFECTIVE_INV_MASS) {
                let e = clamp(q.restitution, 0.0, 1.0);
                let magnitude = -(1.0 + e) * vn / effective;
                impulse = n * magnitude;
            }
        }
    }

    // apply_coupling: particle gains the impulse, body gains the reaction.
    let reaction = impulse * -1.0;
    let out_particle_vel = q.particle_velocity + impulse * q.particle_inv_mass;
    let out_body_lin = q.body_linear_velocity + reaction * q.body_inv_mass;
    let ang_delta = mat_mul_vec(
        q.inv_inertia_col_x,
        q.inv_inertia_col_y,
        q.inv_inertia_col_z,
        cross(r, reaction),
    );
    let out_body_ang = q.body_angular_velocity + ang_delta;

    var out: Result;
    out.impulse = impulse;
    out.pad0 = 0.0;
    out.particle_velocity = out_particle_vel;
    out.pad1 = 0.0;
    out.body_linear_velocity = out_body_lin;
    out.pad2 = 0.0;
    out.body_angular_velocity = out_body_ang;
    out.pad3 = 0.0;
    results[idx] = out;
}
"#;

/// One particle<->body coupling contact to resolve: the full particle and body
/// state, the body's world inverse inertia tensor, the (not necessarily unit)
/// contact `normal` and the restitution.
///
/// `body_inv_inertia_world` arrives already assembled (as on the golden
/// [`CouplingBody`](prism_render_architecture::particle::two_way_coupling::CouplingBody)),
/// reusing the sibling twin's [`GpuMat3`](crate::coupling_impulse::GpuMat3) so no
/// second matrix type is exported.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::two_way_coupling`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuCouplingApplyResolveQuery {
    /// Particle inverse mass (`1 / mass`); `0` is an immovable particle.
    pub particle_inv_mass: f32,
    /// Particle world-space position, also the contact point.
    pub particle_position: Vec3,
    /// Particle world-space linear velocity.
    pub particle_velocity: Vec3,
    /// Body inverse mass (`1 / mass`); `0` is a static body.
    pub body_inv_mass: f32,
    /// Body world-space inverse inertia tensor (columns).
    pub body_inv_inertia_world: GpuMat3,
    /// Body world-space center of mass (the lever-arm origin).
    pub body_center_of_mass: Vec3,
    /// Body world-space linear velocity.
    pub body_linear_velocity: Vec3,
    /// Body world-space angular velocity.
    pub body_angular_velocity: Vec3,
    /// Contact normal (body toward particle); need not be unit length.
    pub normal: Vec3,
    /// Restitution coefficient, clamped to `0..=1` inside the solve.
    pub restitution: f32,
}

/// The resolved per-contact answer: the impulse the reference's
/// `resolve_coupling` returns together with the particle and body velocities it
/// leaves behind.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::two_way_coupling`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuCouplingApplyResolveResult {
    /// Contact impulse applied to the particle, matching `resolve_coupling`'s
    /// return value.
    pub impulse: Vec3,
    /// Particle linear velocity after the apply step.
    pub particle_velocity: Vec3,
    /// Body linear velocity after the apply step.
    pub body_linear_velocity: Vec3,
    /// Body angular velocity after the apply step.
    pub body_angular_velocity: Vec3,
}

/// `repr(C)` `std430` image of one packed contact: nine `vec4` slots carrying
/// the vector inputs each on its `16`-byte-aligned lane, then a trailing `vec4`
/// of the three scalar inputs — `160` bytes matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Particle contact point.
    particle_position: [f32; 3],
    /// Padding lane.
    pad0: f32,
    /// Particle linear velocity.
    particle_velocity: [f32; 3],
    /// Padding lane.
    pad1: f32,
    /// Body center of mass.
    body_center_of_mass: [f32; 3],
    /// Padding lane.
    pad2: f32,
    /// Body linear velocity.
    body_linear_velocity: [f32; 3],
    /// Padding lane.
    pad3: f32,
    /// Body angular velocity.
    body_angular_velocity: [f32; 3],
    /// Padding lane.
    pad4: f32,
    /// World inverse inertia column x.
    inv_inertia_col_x: [f32; 3],
    /// Padding lane.
    pad5: f32,
    /// World inverse inertia column y.
    inv_inertia_col_y: [f32; 3],
    /// Padding lane.
    pad6: f32,
    /// World inverse inertia column z.
    inv_inertia_col_z: [f32; 3],
    /// Padding lane.
    pad7: f32,
    /// Contact normal.
    normal: [f32; 3],
    /// Padding lane.
    pad8: f32,
    /// Particle inverse mass.
    particle_inv_mass: f32,
    /// Body inverse mass.
    body_inv_mass: f32,
    /// Restitution coefficient.
    restitution: f32,
    /// Trailing padding word.
    pad9: f32,
}

impl GpuQuery {
    /// Packs one public query into its `std430` image.
    fn new(query: &GpuCouplingApplyResolveQuery) -> GpuQuery {
        GpuQuery {
            particle_position: [
                query.particle_position.x,
                query.particle_position.y,
                query.particle_position.z,
            ],
            pad0: 0.0,
            particle_velocity: [
                query.particle_velocity.x,
                query.particle_velocity.y,
                query.particle_velocity.z,
            ],
            pad1: 0.0,
            body_center_of_mass: [
                query.body_center_of_mass.x,
                query.body_center_of_mass.y,
                query.body_center_of_mass.z,
            ],
            pad2: 0.0,
            body_linear_velocity: [
                query.body_linear_velocity.x,
                query.body_linear_velocity.y,
                query.body_linear_velocity.z,
            ],
            pad3: 0.0,
            body_angular_velocity: [
                query.body_angular_velocity.x,
                query.body_angular_velocity.y,
                query.body_angular_velocity.z,
            ],
            pad4: 0.0,
            inv_inertia_col_x: [
                query.body_inv_inertia_world.col_x.x,
                query.body_inv_inertia_world.col_x.y,
                query.body_inv_inertia_world.col_x.z,
            ],
            pad5: 0.0,
            inv_inertia_col_y: [
                query.body_inv_inertia_world.col_y.x,
                query.body_inv_inertia_world.col_y.y,
                query.body_inv_inertia_world.col_y.z,
            ],
            pad6: 0.0,
            inv_inertia_col_z: [
                query.body_inv_inertia_world.col_z.x,
                query.body_inv_inertia_world.col_z.y,
                query.body_inv_inertia_world.col_z.z,
            ],
            pad7: 0.0,
            normal: [query.normal.x, query.normal.y, query.normal.z],
            pad8: 0.0,
            particle_inv_mass: query.particle_inv_mass,
            body_inv_mass: query.body_inv_mass,
            restitution: query.restitution,
            pad9: 0.0,
        }
    }
}

/// `repr(C)` `std430` image of one result: four `vec4` slots carrying the
/// impulse and the three updated velocities — `64` bytes matching the `WGSL`
/// `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Contact impulse applied to the particle.
    impulse: [f32; 3],
    /// Padding lane.
    pad0: f32,
    /// Updated particle linear velocity.
    particle_velocity: [f32; 3],
    /// Padding lane.
    pad1: f32,
    /// Updated body linear velocity.
    body_linear_velocity: [f32; 3],
    /// Padding lane.
    pad2: f32,
    /// Updated body angular velocity.
    body_angular_velocity: [f32; 3],
    /// Padding lane.
    pad3: f32,
}

/// Uniform parameters for one dispatch: the contact count plus three pad words
/// to fill a `16`-byte, `16`-byte-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of contacts in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
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

/// A compiled, reusable coupling resolve compute pipeline, twinning the `CPU`
/// golden
/// [`resolve_coupling`](prism_render_architecture::particle::two_way_coupling::resolve_coupling).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::two_way_coupling`。
pub struct GpuCouplingApplyResolve {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCouplingApplyResolve {
    /// Compiles the coupling resolve kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset plus the guarded
    /// `sqrt` the reference normalize already uses, so no optional device
    /// feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::two_way_coupling`。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCouplingApplyResolve {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_coupling_apply_resolve"),
            source: ShaderSource::Wgsl(COUPLING_APPLY_RESOLVE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_coupling_apply_resolve_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_coupling_apply_resolve_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_coupling_apply_resolve_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCouplingApplyResolve {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every coupling contact on-device and returns one
    /// [`GpuCouplingApplyResolveResult`] per input, in order.
    ///
    /// Each result equals the reference's `resolve_coupling` (the returned
    /// impulse) together with the particle and body velocities it leaves behind,
    /// to within the tolerance documented on this module. An empty input returns
    /// an empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::two_way_coupling`。
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[GpuCouplingApplyResolveQuery],
    ) -> Vec<GpuCouplingApplyResolveResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_coupling_apply_resolve_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_coupling_apply_resolve_output"),
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
            label: Some("prism_volumetric_coupling_apply_resolve_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_coupling_apply_resolve_bind_group"),
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
            label: Some("prism_volumetric_coupling_apply_resolve_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_coupling_apply_resolve_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_coupling_apply_resolve_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per contact, flattened to a 1-D dispatch.
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

/// Decodes one packed [`GpuResult`] into the public
/// [`GpuCouplingApplyResolveResult`].
fn decode_result(raw: &GpuResult) -> GpuCouplingApplyResolveResult {
    GpuCouplingApplyResolveResult {
        impulse: Vec3::new(raw.impulse[0], raw.impulse[1], raw.impulse[2]),
        particle_velocity: Vec3::new(
            raw.particle_velocity[0],
            raw.particle_velocity[1],
            raw.particle_velocity[2],
        ),
        body_linear_velocity: Vec3::new(
            raw.body_linear_velocity[0],
            raw.body_linear_velocity[1],
            raw.body_linear_velocity[2],
        ),
        body_angular_velocity: Vec3::new(
            raw.body_angular_velocity[0],
            raw.body_angular_velocity[1],
            raw.body_angular_velocity[2],
        ),
    }
}
