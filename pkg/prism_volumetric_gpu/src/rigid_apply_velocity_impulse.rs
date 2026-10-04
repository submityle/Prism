//! `wgpu` compute twin of the rigid-body velocity-impulse application from the
//! `CPU` golden `prism_physics_core::solver::xpbd::rigid::apply_velocity_impulse`.
//!
//! An `XPBD` velocity solve resolves each contact or joint by applying an
//! impulse `p` at a world lever arm `r`: the linear velocity gains
//! `inv_mass * p` and the angular velocity gains the world inverse-inertia
//! matrix times the torque impulse `cross(r, p)`. This is the per-body velocity
//! update that follows the positional projection pass.
//!
//! This module ports that stateless, closed-form update onto the device: one
//! thread resolves one body. [`GpuRigidApplyVelocityImpulse`] is the on-device
//! twin; a passing real-device parity test is direct evidence the kernel takes
//! the same column-major matrix-vector product and the same cross-product sign
//! convention the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! * `apply_velocity_impulse` — the whole closed form: the linear update
//!   `new_lin = linear_velocity + inv_mass * p`, the torque impulse
//!   `rp = cross(r, p)`, and the angular update
//!   `new_ang = angular_velocity + I * rp`, where `I` is the world
//!   inverse-inertia stored column-major and the product is
//!   `I * rp = col0 * rp.x + col1 * rp.y + col2 * rp.z`.
//!
//! # Result encoding
//!
//! The reference updates the velocities in place. The twin reports the two
//! updated vectors and a `valid` flag that is always `1`: the update has no
//! degenerate branch, so every query produces a defined answer. The flag is
//! carried for symmetry with the other twins in this crate and to keep the
//! `std430` result stride a clean multiple of sixteen bytes.
//!
//! # Correctness model
//!
//! Every output is a continuous quantity checked with an absolute-or-relative
//! tolerance. There is no discrete branch and no division, so there is no
//! conditioning knee to avoid; the only subtlety is matching the column-major
//! matrix layout and the cross-product sign between host and device, which the
//! fixtures and sweep exercise directly.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — scalar `f32`
//! arithmetic and `vec3` reconstruction from flattened scalars — with no `sin`,
//! `cos`, `tan`, `exp`, `log`, `pow`, no `round`, no float modulo and no
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. Every `vec3` and the `3x3` matrix are flattened to scalar storage so
//! no `std430` vector- or matrix-alignment surprise can appear, and there is no
//! bare float equality anywhere in the kernel.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::solver::xpbd::rigid`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` velocity-impulse kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `apply_velocity_impulse`; see the module documentation for
/// the algorithm.
const RIGID_APPLY_VELOCITY_IMPULSE_WGSL: &str = r#"
// Rigid-body velocity-impulse twin: one thread per body applies the impulse p at
// lever arm r, updating linear and angular velocity, mirroring
// apply_velocity_impulse. The inverse-inertia matrix is column-major.
// Provenance: 孪生自本仓 prism_physics_core::solver::xpbd::rigid；无第三方引擎源码或衍生代码。

struct Params {
    // Number of bodies in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Body {
    // Linear velocity (x, y, z).
    linx: f32,
    liny: f32,
    linz: f32,
    // Angular velocity (x, y, z).
    angx: f32,
    angy: f32,
    angz: f32,
    // Inverse mass.
    inv_mass: f32,
    // World inverse-inertia, column-major: col0 = (m0, m1, m2),
    // col1 = (m3, m4, m5), col2 = (m6, m7, m8).
    m0: f32,
    m1: f32,
    m2: f32,
    m3: f32,
    m4: f32,
    m5: f32,
    m6: f32,
    m7: f32,
    m8: f32,
    // World lever arm r (x, y, z).
    rx: f32,
    ry: f32,
    rz: f32,
    // Impulse p (x, y, z).
    px: f32,
    py: f32,
    pz: f32,
    pad0: f32,
    pad1: f32,
}

struct Updated {
    // Updated linear velocity (x, y, z).
    linx: f32,
    liny: f32,
    linz: f32,
    // Updated angular velocity (x, y, z).
    angx: f32,
    angy: f32,
    angz: f32,
    // Always 1: the update has no degenerate branch.
    valid: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> bodies: array<Body>;
@group(0) @binding(2) var<storage, read_write> results: array<Updated>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let b = bodies[idx];

    let lin = vec3<f32>(b.linx, b.liny, b.linz);
    let ang = vec3<f32>(b.angx, b.angy, b.angz);
    let lever = vec3<f32>(b.rx, b.ry, b.rz);
    let imp = vec3<f32>(b.px, b.py, b.pz);

    // Column-major inverse-inertia reconstruction.
    let col0 = vec3<f32>(b.m0, b.m1, b.m2);
    let col1 = vec3<f32>(b.m3, b.m4, b.m5);
    let col2 = vec3<f32>(b.m6, b.m7, b.m8);

    // new_lin = lin + inv_mass * p.
    let new_lin = lin + b.inv_mass * imp;

    // Torque impulse rp = cross(r, p), same sign convention as the golden.
    let rp = cross(lever, imp);

    // new_ang = ang + I * rp, with I * rp = col0*rp.x + col1*rp.y + col2*rp.z.
    let new_ang = ang + col0 * rp.x + col1 * rp.y + col2 * rp.z;

    var out: Updated;
    out.linx = new_lin.x;
    out.liny = new_lin.y;
    out.linz = new_lin.z;
    out.angx = new_ang.x;
    out.angy = new_ang.y;
    out.angz = new_ang.z;
    out.valid = 1u;
    out.pad0 = 0u;

    results[idx] = out;
}
"#;

/// `repr(C)` `std430` layout of the dispatch parameters.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid bodies in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one body, matching the `WGSL` `Body` struct.
/// Twenty-two payload words plus two padding words keep the stride a flat `96`
/// bytes, a multiple of `16`, with every `vec3` and the `3x3` matrix flattened
/// to scalars so no vector- or matrix-alignment surprise can appear.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuBody {
    linx: f32,
    liny: f32,
    linz: f32,
    angx: f32,
    angy: f32,
    angz: f32,
    inv_mass: f32,
    m0: f32,
    m1: f32,
    m2: f32,
    m3: f32,
    m4: f32,
    m5: f32,
    m6: f32,
    m7: f32,
    m8: f32,
    rx: f32,
    ry: f32,
    rz: f32,
    px: f32,
    py: f32,
    pz: f32,
    pad0: f32,
    pad1: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Updated`
/// struct. Six velocity words plus the `valid` word and one padding word keep
/// the stride a flat `32` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    linx: f32,
    liny: f32,
    linz: f32,
    angx: f32,
    angy: f32,
    angz: f32,
    valid: u32,
    pad0: u32,
}

/// One velocity-impulse query: the body's linear and angular velocity, inverse
/// mass, world inverse-inertia (column-major, nine scalars), the world lever
/// arm `r` and the impulse `p`. Every vector and the matrix are flattened to
/// scalars so the `std430` stride stays an unambiguous flat layout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidApplyVelocityImpulseQuery {
    /// Linear velocity, x component.
    pub linx: f32,
    /// Linear velocity, y component.
    pub liny: f32,
    /// Linear velocity, z component.
    pub linz: f32,
    /// Angular velocity, x component.
    pub angx: f32,
    /// Angular velocity, y component.
    pub angy: f32,
    /// Angular velocity, z component.
    pub angz: f32,
    /// Inverse mass.
    pub inv_mass: f32,
    /// World inverse-inertia, column `0` x (col0.x).
    pub m0: f32,
    /// World inverse-inertia, column `0` y (col0.y).
    pub m1: f32,
    /// World inverse-inertia, column `0` z (col0.z).
    pub m2: f32,
    /// World inverse-inertia, column `1` x (col1.x).
    pub m3: f32,
    /// World inverse-inertia, column `1` y (col1.y).
    pub m4: f32,
    /// World inverse-inertia, column `1` z (col1.z).
    pub m5: f32,
    /// World inverse-inertia, column `2` x (col2.x).
    pub m6: f32,
    /// World inverse-inertia, column `2` y (col2.y).
    pub m7: f32,
    /// World inverse-inertia, column `2` z (col2.z).
    pub m8: f32,
    /// World lever arm `r`, x component.
    pub rx: f32,
    /// World lever arm `r`, y component.
    pub ry: f32,
    /// World lever arm `r`, z component.
    pub rz: f32,
    /// Impulse `p`, x component.
    pub px: f32,
    /// Impulse `p`, y component.
    pub py: f32,
    /// Impulse `p`, z component.
    pub pz: f32,
}

impl RigidApplyVelocityImpulseQuery {
    /// Builds a velocity-impulse query from the body's linear and angular
    /// velocity, inverse mass, world inverse-inertia matrix (column-major, as
    /// three columns `col0`, `col1`, `col2`), the world lever arm `r` and the
    /// impulse `p`.
    #[must_use]
    pub fn new(
        linear_velocity: [f32; 3],
        angular_velocity: [f32; 3],
        inv_mass: f32,
        inv_inertia_cols: [[f32; 3]; 3],
        r: [f32; 3],
        p: [f32; 3],
    ) -> RigidApplyVelocityImpulseQuery {
        RigidApplyVelocityImpulseQuery {
            linx: linear_velocity[0],
            liny: linear_velocity[1],
            linz: linear_velocity[2],
            angx: angular_velocity[0],
            angy: angular_velocity[1],
            angz: angular_velocity[2],
            inv_mass,
            m0: inv_inertia_cols[0][0],
            m1: inv_inertia_cols[0][1],
            m2: inv_inertia_cols[0][2],
            m3: inv_inertia_cols[1][0],
            m4: inv_inertia_cols[1][1],
            m5: inv_inertia_cols[1][2],
            m6: inv_inertia_cols[2][0],
            m7: inv_inertia_cols[2][1],
            m8: inv_inertia_cols[2][2],
            rx: r[0],
            ry: r[1],
            rz: r[2],
            px: p[0],
            py: p[1],
            pz: p[2],
        }
    }
}

/// One resolved answer for a single body: the updated linear and angular
/// velocity, plus a `valid` flag that is always `1` (the update has no
/// degenerate branch).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidApplyVelocityImpulseResult {
    /// Updated linear velocity, x component.
    pub linx: f32,
    /// Updated linear velocity, y component.
    pub liny: f32,
    /// Updated linear velocity, z component.
    pub linz: f32,
    /// Updated angular velocity, x component.
    pub angx: f32,
    /// Updated angular velocity, y component.
    pub angy: f32,
    /// Updated angular velocity, z component.
    pub angz: f32,
    /// Always `1`: the update is always defined.
    pub valid: u32,
}

/// Encodes one [`RigidApplyVelocityImpulseQuery`] into its `std430`
/// [`GpuBody`].
fn encode_query(q: &RigidApplyVelocityImpulseQuery) -> GpuBody {
    GpuBody {
        linx: q.linx,
        liny: q.liny,
        linz: q.linz,
        angx: q.angx,
        angy: q.angy,
        angz: q.angz,
        inv_mass: q.inv_mass,
        m0: q.m0,
        m1: q.m1,
        m2: q.m2,
        m3: q.m3,
        m4: q.m4,
        m5: q.m5,
        m6: q.m6,
        m7: q.m7,
        m8: q.m8,
        rx: q.rx,
        ry: q.ry,
        rz: q.rz,
        px: q.px,
        py: q.py,
        pz: q.pz,
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`RigidApplyVelocityImpulseResult`].
fn decode_result(raw: &GpuResult) -> RigidApplyVelocityImpulseResult {
    RigidApplyVelocityImpulseResult {
        linx: raw.linx,
        liny: raw.liny,
        linz: raw.linz,
        angx: raw.angx,
        angy: raw.angy,
        angz: raw.angz,
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

/// A compiled, reusable velocity-impulse compute pipeline, twinning the `CPU`
/// golden `apply_velocity_impulse`.
pub struct GpuRigidApplyVelocityImpulse {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRigidApplyVelocityImpulse {
    /// Compiles the velocity-impulse kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRigidApplyVelocityImpulse {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_rigid_apply_velocity_impulse"),
            source: ShaderSource::Wgsl(RIGID_APPLY_VELOCITY_IMPULSE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_rigid_apply_velocity_impulse_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_rigid_apply_velocity_impulse_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_rigid_apply_velocity_impulse_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRigidApplyVelocityImpulse {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`RigidApplyVelocityImpulseResult`] per input, in order.
    ///
    /// An empty `queries` batch returns an empty vector with no dispatch
    /// issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[RigidApplyVelocityImpulseQuery],
    ) -> Vec<RigidApplyVelocityImpulseResult> {
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
            label: Some("prism_volumetric_rigid_apply_velocity_impulse_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuBody> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_rigid_apply_velocity_impulse_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_rigid_apply_velocity_impulse_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_rigid_apply_velocity_impulse_bind_group"),
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
            label: Some("prism_volumetric_rigid_apply_velocity_impulse_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_rigid_apply_velocity_impulse_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_rigid_apply_velocity_impulse_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per body, flattened to a 1-D dispatch.
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
