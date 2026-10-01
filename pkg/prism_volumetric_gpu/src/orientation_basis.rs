//! `wgpu` compute twin of the per-particle `billboard` orientation golden
//! ([`orientation_basis`](prism_render_architecture::particle::orientation_basis),
//! particle design §16).
//!
//! The `CPU` golden
//! [`orientation_basis`](prism_render_architecture::particle::orientation_basis)
//! owns the contract that turns a [`FacingMode`] plus a particle's world state
//! (`pos`, `cam_pos`, `vel`, `world_up`, `fixed_axis`) into an orthonormal 2D
//! quad frame — a unit `right` axis and a unit `up` axis that are mutually
//! perpendicular. Every basis is built from `cross`/`normalize` products, so the
//! only floating-point primitive is `sqrt` (inside `normalize`); there is no
//! `sin`/`cos`/`atan`/quaternion path.
//!
//! [`GpuOrientationBasis`] is the on-device twin: one thread per particle
//! reproduces the same facing-mode branch for branch, including every degenerate
//! fallback the reference takes — a camera sitting on the particle, a zero
//! velocity, a `world_up` or `fixed_axis` parallel to the view direction, a zero
//! `world_up`. A passing real-device parity test is therefore direct evidence
//! the ported kernel builds the same frame and classifies the same degenerate
//! case the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Both public entry points of the reference are reproduced: the per-particle
//! [`compute_basis`](prism_render_architecture::particle::orientation_basis::compute_basis)
//! and the batch
//! [`compute_bases`](prism_render_architecture::particle::orientation_basis::compute_bases),
//! the latter being the per-element dispatch that fills an orientation buffer in
//! a single pass. Each of the five [`FacingMode`] variants is mirrored, as is
//! each degenerate fallback: a near-zero direction normalizes to a stable world
//! axis, a collapsed `cross` selects a well-conditioned perpendicular, and a
//! missing view direction falls back to the world `right`/`up` frame.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `+ - * /`, the `dot` and `cross` builtins and one `sqrt` for the genuine
//! Euclidean length — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan` and no
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`.
//!
//! # Correctness model
//!
//! Each basis is a fixed, non-reorderable sequence of multiplies, adds, divides
//! and one `sqrt`, so `CPU` and `GPU` evaluate the same closed form in the same
//! associativity. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on every `f32` axis component,
//! tight enough to catch a genuinely wrong port (a dropped branch, a swapped
//! axis, a wrong fallback) yet loose enough to admit legal fused multiply-add
//! contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::orientation_basis`；
//! standard `cross`/`normalize` `billboard` frame plus `wgpu` compute dispatch;
//! no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::orientation_basis::{
    compute_basis, FacingMode, OrientationBasis,
};
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

/// Mode code for [`FacingMode::Billboard`] in the packed `GPU` query.
const MODE_BILLBOARD: u32 = 0;
/// Mode code for [`FacingMode::HorizontalBillboard`] in the packed `GPU` query.
const MODE_HORIZONTAL_BILLBOARD: u32 = 1;
/// Mode code for [`FacingMode::VerticalBillboard`] in the packed `GPU` query.
const MODE_VERTICAL_BILLBOARD: u32 = 2;
/// Mode code for [`FacingMode::VelocityAligned`] in the packed `GPU` query.
const MODE_VELOCITY_ALIGNED: u32 = 3;
/// Mode code for [`FacingMode::FixedAxis`] in the packed `GPU` query.
const MODE_FIXED_AXIS: u32 = 4;

/// The portable core-`WGSL` orientation-basis kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`orientation_basis`](prism_render_architecture::particle::orientation_basis)
/// branch for branch; see the module documentation for the algorithm.
const ORIENTATION_BASIS_WGSL: &str = r#"
// Orientation-basis twin: one thread per particle builds the orthonormal
// right/up quad frame for a facing mode. It mirrors the CPU golden
// particle::orientation_basis branch for branch, uses only the portable
// core-WGSL subset (abs/min/max and + - * / plus the dot and cross builtins and
// one sqrt) and takes no optional feature, so it runs unmodified on Metal,
// Vulkan and DX12.
//
// Provenance: twinned from this repository's particle::orientation_basis; no
// third-party engine source or derived code.

// Length below which a vector is treated as degenerate and its normalization
// falls back instead of dividing by a near-zero magnitude. Matches the
// reference MIN_LENGTH.
const MIN_LENGTH: f32 = 1.0e-6;

// World axes, used as stable fallbacks exactly as the reference WORLD_X/Y/Z.
const WORLD_X: vec3<f32> = vec3<f32>(1.0, 0.0, 0.0);
const WORLD_Y: vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);
const WORLD_Z: vec3<f32> = vec3<f32>(0.0, 0.0, 1.0);

// Facing-mode codes, matching the host-side mode_code mapping.
const MODE_BILLBOARD: u32 = 0u;
const MODE_HORIZONTAL_BILLBOARD: u32 = 1u;
const MODE_VERTICAL_BILLBOARD: u32 = 2u;
const MODE_VELOCITY_ALIGNED: u32 = 3u;
const MODE_FIXED_AXIS: u32 = 4u;

struct Params {
    // Number of particles in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Particle position; the fourth lane carries the facing-mode code.
    pos: vec3<f32>,
    mode: u32,
    // Camera position; a pad lane follows.
    cam_pos: vec3<f32>,
    pad0: f32,
    // Particle velocity; a pad lane follows.
    vel: vec3<f32>,
    pad1: f32,
    // Scene world up axis; a pad lane follows.
    world_up: vec3<f32>,
    pad2: f32,
    // Constraint axis consulted by the fixed-axis mode; a pad lane follows.
    fixed_axis: vec3<f32>,
    pad3: f32,
}

struct Result {
    // Local right axis (unit length); a pad lane fills the vec4 slot.
    right: vec3<f32>,
    pad0: f32,
    // Local up axis (unit length, perpendicular to right); a pad lane follows.
    up: vec3<f32>,
    pad1: f32,
}

// A normalized vector together with an ok flag standing in for the reference's
// Option<[f32; 3]>: ok == 0u mirrors None (too degenerate to give a direction).
struct Norm {
    ok: u32,
    v: vec3<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

fn length3(v: vec3<f32>) -> f32 {
    return sqrt(dot(v, v));
}

// Normalizes v to unit length, or reports ok == 0u when v is shorter than
// MIN_LENGTH, mirroring the reference normalize3 returning None.
fn normalize3(v: vec3<f32>) -> Norm {
    let len = length3(v);
    var out: Norm;
    if (len < MIN_LENGTH) {
        out.ok = 0u;
        out.v = vec3<f32>(0.0, 0.0, 0.0);
    } else {
        let inv = 1.0 / len;
        out.ok = 1u;
        out.v = v * inv;
    }
    return out;
}

// Returns a stable unit vector perpendicular to the unit vector axis, crossing
// it with the world axis it is least aligned with (a well-conditioned cross),
// mirroring the reference any_perpendicular.
fn any_perpendicular(axis: vec3<f32>) -> vec3<f32> {
    let ax = abs(axis.x);
    let ay = abs(axis.y);
    let az = abs(axis.z);
    var reference: vec3<f32> = WORLD_Z;
    if (ax <= ay && ax <= az) {
        reference = WORLD_X;
    } else if (ay <= az) {
        reference = WORLD_Y;
    }
    let n = normalize3(cross(axis, reference));
    if (n.ok == 1u) {
        return n.v;
    }
    return WORLD_X;
}

// Builds a unit right vector perpendicular to up_unit, aiming it toward the
// camera when a valid view direction is available (has_cam == 1u), and falling
// back to a stable perpendicular otherwise. Mirrors right_perpendicular_to.
fn right_perpendicular_to(up_unit: vec3<f32>, to_cam: vec3<f32>, has_cam: u32) -> vec3<f32> {
    if (has_cam == 1u) {
        let r = normalize3(cross(up_unit, to_cam));
        if (r.ok == 1u) {
            return r.v;
        }
    }
    return any_perpendicular(up_unit);
}

// normalize3(v) or world up or WORLD_Y, mirroring the nested unwrap_or_else the
// velocity-aligned and fixed-axis modes use.
fn normalize_or_world_up(v: vec3<f32>, world_up: vec3<f32>) -> vec3<f32> {
    let a = normalize3(v);
    if (a.ok == 1u) {
        return a.v;
    }
    let b = normalize3(world_up);
    if (b.ok == 1u) {
        return b.v;
    }
    return WORLD_Y;
}

// normalize3(v) or WORLD_Y, mirroring normalize3(world_up).unwrap_or(WORLD_Y).
fn normalize_or_y(v: vec3<f32>) -> vec3<f32> {
    let a = normalize3(v);
    if (a.ok == 1u) {
        return a.v;
    }
    return WORLD_Y;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let to_cam = normalize3(q.cam_pos - q.pos);

    var right: vec3<f32> = WORLD_X;
    var up: vec3<f32> = WORLD_Y;

    if (q.mode == MODE_BILLBOARD) {
        if (to_cam.ok == 0u) {
            right = WORLD_X;
            up = WORLD_Y;
        } else {
            let view = to_cam.v;
            let rc = normalize3(cross(q.world_up, view));
            if (rc.ok == 1u) {
                right = rc.v;
            } else {
                right = any_perpendicular(view);
            }
            up = cross(view, right);
        }
    } else if (q.mode == MODE_HORIZONTAL_BILLBOARD) {
        up = normalize_or_y(q.world_up);
        right = right_perpendicular_to(up, to_cam.v, to_cam.ok);
    } else if (q.mode == MODE_VERTICAL_BILLBOARD) {
        let normal = normalize_or_y(q.world_up);
        right = right_perpendicular_to(normal, to_cam.v, to_cam.ok);
        // Ground-aligned: the plane spans the two horizontal axes while the
        // normal stays along world up.
        up = cross(normal, right);
    } else if (q.mode == MODE_VELOCITY_ALIGNED) {
        up = normalize_or_world_up(q.vel, q.world_up);
        right = right_perpendicular_to(up, to_cam.v, to_cam.ok);
    } else {
        // MODE_FIXED_AXIS.
        up = normalize_or_world_up(q.fixed_axis, q.world_up);
        right = right_perpendicular_to(up, to_cam.v, to_cam.ok);
    }

    var out: Result;
    out.right = right;
    out.pad0 = 0.0;
    out.up = up;
    out.pad1 = 0.0;
    results[idx] = out;
}
"#;

/// One orientation query: a [`FacingMode`] plus the particle world state the
/// reference
/// [`compute_basis`](prism_render_architecture::particle::orientation_basis::compute_basis)
/// consumes. Inputs a mode does not need are ignored exactly as in the golden.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::orientation_basis`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrientationQuery {
    /// How the quad orients itself toward the camera or a constraint axis.
    pub mode: FacingMode,
    /// Particle position.
    pub pos: [f32; 3],
    /// Camera position.
    pub cam_pos: [f32; 3],
    /// Particle velocity (consulted by [`FacingMode::VelocityAligned`]).
    pub vel: [f32; 3],
    /// Scene world `up` axis.
    pub world_up: [f32; 3],
    /// Constraint axis consulted by [`FacingMode::FixedAxis`].
    pub fixed_axis: [f32; 3],
}

impl OrientationQuery {
    /// Builds an orientation query from a facing mode and the particle world
    /// state.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::orientation_basis`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub const fn new(
        mode: FacingMode,
        pos: [f32; 3],
        cam_pos: [f32; 3],
        vel: [f32; 3],
        world_up: [f32; 3],
        fixed_axis: [f32; 3],
    ) -> OrientationQuery {
        OrientationQuery {
            mode,
            pos,
            cam_pos,
            vel,
            world_up,
            fixed_axis,
        }
    }
}

/// Maps a [`FacingMode`] to its packed `GPU` mode code.
#[must_use]
fn mode_code(mode: FacingMode) -> u32 {
    match mode {
        FacingMode::Billboard => MODE_BILLBOARD,
        FacingMode::HorizontalBillboard => MODE_HORIZONTAL_BILLBOARD,
        FacingMode::VerticalBillboard => MODE_VERTICAL_BILLBOARD,
        FacingMode::VelocityAligned => MODE_VELOCITY_ALIGNED,
        FacingMode::FixedAxis => MODE_FIXED_AXIS,
    }
}

/// Evaluates the `CPU` golden for one query, returning the reference
/// [`OrientationBasis`].
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::orientation_basis`；
/// no third-party engine source or derived code.
#[must_use]
pub fn golden(query: &OrientationQuery) -> OrientationBasis {
    compute_basis(
        query.mode,
        query.pos,
        query.cam_pos,
        query.vel,
        query.world_up,
        query.fixed_axis,
    )
}

/// `repr(C)` `std430` layout of one packed query: five `vec4` slots holding
/// `(pos.xyz, mode)`, `(cam_pos.xyz, pad)`, `(vel.xyz, pad)`,
/// `(world_up.xyz, pad)` and `(fixed_axis.xyz, pad)` — `80` bytes, each `vec3`
/// on its `16`-byte-aligned slot exactly as the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Particle position.
    pos: [f32; 3],
    /// Facing-mode code, packed in the fourth lane of the first slot.
    mode: u32,
    /// Camera position.
    cam_pos: [f32; 3],
    /// Padding lane after the camera position.
    pad0: f32,
    /// Particle velocity.
    vel: [f32; 3],
    /// Padding lane after the velocity.
    pad1: f32,
    /// Scene world `up` axis.
    world_up: [f32; 3],
    /// Padding lane after the world `up` axis.
    pad2: f32,
    /// Constraint axis.
    fixed_axis: [f32; 3],
    /// Padding lane after the constraint axis.
    pad3: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &OrientationQuery) -> GpuQuery {
        GpuQuery {
            pos: query.pos,
            mode: mode_code(query.mode),
            cam_pos: query.cam_pos,
            pad0: 0.0,
            vel: query.vel,
            pad1: 0.0,
            world_up: query.world_up,
            pad2: 0.0,
            fixed_axis: query.fixed_axis,
            pad3: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: two `vec4` slots holding
/// `(right.xyz, pad)` and `(up.xyz, pad)` — `32` bytes matching the `WGSL`
/// `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Local `right` axis.
    right: [f32; 3],
    /// Padding lane after the `right` axis.
    pad0: f32,
    /// Local `up` axis.
    up: [f32; 3],
    /// Padding lane after the `up` axis.
    pad1: f32,
}

/// Uniform parameters for one dispatch: the particle count plus three pad words
/// to fill a `16`-byte, `16`-byte-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of particles in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// Decodes one packed [`GpuResult`] into the public [`OrientationBasis`].
fn decode_result(raw: &GpuResult) -> OrientationBasis {
    OrientationBasis {
        right: raw.right,
        up: raw.up,
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

/// A compiled, reusable orientation-basis compute pipeline.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::orientation_basis`；
/// no third-party engine source or derived code.
pub struct GpuOrientationBasis {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuOrientationBasis {
    /// Compiles the orientation-basis kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::orientation_basis`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuOrientationBasis {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_orientation_basis"),
            source: ShaderSource::Wgsl(ORIENTATION_BASIS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_orientation_basis_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_orientation_basis_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_orientation_basis_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuOrientationBasis {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`OrientationBasis`] per
    /// input, in order — the batch twin of the reference
    /// [`compute_bases`](prism_render_architecture::particle::orientation_basis::compute_bases).
    ///
    /// Each basis equals the reference
    /// [`compute_basis`](prism_render_architecture::particle::orientation_basis::compute_basis)
    /// answer to within the tolerance documented on this module. An empty input
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::orientation_basis`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[OrientationQuery]) -> Vec<OrientationBasis> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_orientation_basis_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_orientation_basis_output"),
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
            label: Some("prism_volumetric_orientation_basis_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_orientation_basis_bind_group"),
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
            label: Some("prism_volumetric_orientation_basis_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_orientation_basis_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_orientation_basis_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per particle, flattened to a 1-D dispatch.
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
