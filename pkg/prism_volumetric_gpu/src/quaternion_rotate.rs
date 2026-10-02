//! `wgpu` compute twin of the unit-`quaternion` vector-rotation contract
//! ([`quaternion_rotate`](prism_render_architecture::particle::quaternion_rotate),
//! particle design §16).
//!
//! The `CPU` golden
//! [`quaternion_rotate`](prism_render_architecture::particle::quaternion_rotate)
//! owns the per-particle orientation algebra. This twin reproduces the two
//! steps a particle stage runs every frame to point a vector by a stored
//! orientation: first renormalize the orientation `quaternion`
//! ([`Quat::normalize`](prism_render_architecture::particle::quaternion_rotate::Quat::normalize)),
//! then rotate the vector by it
//! ([`Quat::rotate_vec3`](prism_render_architecture::particle::quaternion_rotate::Quat::rotate_vec3)).
//! The rotation uses the multiply/add identity `v + 2*w*(u x v) + 2*(u x (u x
//! v))`, with `u` the vector part, which equals `q * v * q^-1` for a unit
//! `quaternion` and preserves length.
//!
//! [`GpuQuaternionRotate`] is the on-device twin: one thread normalizes one
//! `quaternion` and rotates one vector, so a passing real-device parity test is
//! direct evidence the ported kernel renormalizes and rotates exactly as the
//! reference, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Both answers the reference produces for one `(quaternion, vector)` pair are
//! reproduced: the renormalized unit `quaternion` and the rotated vector. The
//! host side builds its expectation by calling the reference
//! [`Quat::normalize`](prism_render_architecture::particle::quaternion_rotate::Quat::normalize)
//! and
//! [`Quat::rotate_vec3`](prism_render_architecture::particle::quaternion_rotate::Quat::rotate_vec3)
//! directly, so the device twin and the host are checked against one source of
//! truth.
//!
//! # Correctness model
//!
//! Each pair is a fixed, non-reorderable sequence of multiplies, adds, divides
//! and one `sqrt` (the normalization length), so `CPU` and `GPU` evaluate the
//! same closed form in the same associativity. They are not bit-exact: a `GPU`
//! may fuse a multiply-add the scalar reference leaves separate, perturbing the
//! low mantissa bits by a few units in the last place. The parity test
//! therefore asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on
//! every `f32` field, tight enough to catch a genuinely wrong port (a dropped
//! cross term, a swapped sign, a wrong scalar lane) yet loose enough to admit
//! legal fused multiply-add contraction.
//!
//! # Degenerate inputs
//!
//! A near-zero `quaternion` has an undefined direction; the kernel checks its
//! length against [`MIN_LENGTH`] and falls back to the identity
//! `(0, 0, 0, 1)`, so normalization never divides by a near-zero magnitude and
//! the rotation returns the input vector unchanged — matching the reference
//! fallback. An empty batch short-circuits on the host with no dispatch, since
//! a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+ - * /`, the `dot`
//! builtin and one `sqrt` for the genuine Euclidean length, with a hand-written
//! cross product — and no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse
//! trigonometry and no optional device feature, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`. There is no loop: each thread performs a
//! fixed, bounded sequence of arithmetic, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::quaternion_rotate`；
//! no third-party engine source or derived code.
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::quaternion_rotate::Quat;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Length below which a `quaternion` is treated as degenerate and its
/// normalization falls back to the identity instead of dividing by a near-zero
/// magnitude. Mirrors the reference `MIN_LENGTH` constant exactly.
pub const MIN_LENGTH: f32 = 1.0e-6;

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` quaternion-rotation kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`quaternion_rotate`](prism_render_architecture::particle::quaternion_rotate)
/// step for step; see the module documentation for the algorithm.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::quaternion_rotate`；
/// no third-party engine source or derived code.
const QUATERNION_ROTATE_WGSL: &str = r#"
// Quaternion-rotate twin: one thread renormalizes one quaternion and rotates
// one vector by it. It mirrors the CPU golden particle::quaternion_rotate step
// for step, uses only the portable core-WGSL subset (+ - * / plus the dot
// builtin, one sqrt and a hand-written cross) and takes no optional feature, so
// it runs unmodified on Metal, Vulkan and DX12.

// Length below which a quaternion is degenerate and normalization returns the
// identity, matching the reference MIN_LENGTH.
const MIN_LENGTH: f32 = 1.0e-6;

struct Params {
    // Number of pairs in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Orientation quaternion stored as (x, y, z, w) with w the scalar part.
    quat: vec4<f32>,
    // Vector to rotate; a pad lane follows to fill the 16-byte slot.
    source: vec3<f32>,
    pad0: f32,
}

struct Result {
    // Renormalized unit quaternion (x, y, z, w).
    quat: vec4<f32>,
    // Rotated vector; a pad lane follows.
    rotated: vec3<f32>,
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// The right-handed cross product a x b, hand-written to mirror the reference
// v3_cross so no builtin outside the documented subset is relied on.
fn v3_cross(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    );
}

// Returns the unit quaternion in the same direction. A near-zero quaternion
// (whose direction is undefined) clamps to the identity so the result is never
// a NaN, mirroring the reference Quat::normalize.
fn q_normalize(q: vec4<f32>) -> vec4<f32> {
    let len = sqrt(dot(q, q));
    if (len < MIN_LENGTH) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let inv = 1.0 / len;
    return q * inv;
}

// Rotates v by q via the multiply/add identity v + 2*w*(u x v) + 2*(u x (u x
// v)) with u the vector part, mirroring the reference Quat::rotate_vec3.
fn q_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    let t = 2.0 * v3_cross(u, v);
    return v + q.w * t + v3_cross(u, t);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let item = queries[idx];
    let qn = q_normalize(item.quat);
    let rotated = q_rotate(qn, item.source);

    var out: Result;
    out.quat = qn;
    out.rotated = rotated;
    out.pad0 = 0.0;
    results[idx] = out;
}
"#;

/// One rotation query: the orientation `quaternion` to renormalize and the
/// vector to rotate by it — the same inputs the reference
/// [`Quat::normalize`](prism_render_architecture::particle::quaternion_rotate::Quat::normalize)
/// and
/// [`Quat::rotate_vec3`](prism_render_architecture::particle::quaternion_rotate::Quat::rotate_vec3)
/// consume.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::quaternion_rotate`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuatRotateQuery {
    /// Orientation `quaternion`, renormalized before the rotation.
    pub quat: Quat,
    /// The vector rotated by the renormalized `quaternion`.
    pub vector: [f32; 3],
}

impl QuatRotateQuery {
    /// Builds a query from the orientation `quaternion` and the vector to
    /// rotate.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::quaternion_rotate`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub const fn new(quat: Quat, vector: [f32; 3]) -> QuatRotateQuery {
        QuatRotateQuery { quat, vector }
    }
}

/// The resolved answer for one query: the renormalized unit `quaternion` and
/// the rotated vector.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::quaternion_rotate`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuatRotateResult {
    /// The renormalized unit `quaternion`, matching
    /// [`Quat::normalize`](prism_render_architecture::particle::quaternion_rotate::Quat::normalize).
    pub normalized: Quat,
    /// The rotated vector, matching
    /// [`Quat::rotate_vec3`](prism_render_architecture::particle::quaternion_rotate::Quat::rotate_vec3).
    pub rotated: [f32; 3],
}

/// Evaluates the `CPU` golden for one query, delegating to the reference
/// [`Quat::normalize`](prism_render_architecture::particle::quaternion_rotate::Quat::normalize)
/// and
/// [`Quat::rotate_vec3`](prism_render_architecture::particle::quaternion_rotate::Quat::rotate_vec3)
/// so the host side and the device twin are checked against the same source of
/// truth.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::quaternion_rotate`；
/// no third-party engine source or derived code.
#[must_use]
pub fn golden(query: &QuatRotateQuery) -> QuatRotateResult {
    let normalized = query.quat.normalize();
    let rotated = normalized.rotate_vec3(query.vector);
    QuatRotateResult {
        normalized,
        rotated,
    }
}

/// `repr(C)` `std430` layout of one packed query: a `vec4` slot holding the
/// `quaternion` `(x, y, z, w)` followed by a `vec4` slot holding `(vector.xyz,
/// pad)` — `32` bytes, each on its `16`-byte-aligned slot exactly as the `WGSL`
/// `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Orientation `quaternion` `(x, y, z, w)`.
    quat: [f32; 4],
    /// Vector to rotate.
    vector: [f32; 3],
    /// Padding lane after the vector.
    pad0: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &QuatRotateQuery) -> GpuQuery {
        GpuQuery {
            quat: [query.quat.x, query.quat.y, query.quat.z, query.quat.w],
            vector: query.vector,
            pad0: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: a `vec4` slot holding the
/// renormalized `quaternion` followed by a `vec4` slot holding `(rotated.xyz,
/// pad)` — `32` bytes matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Renormalized unit `quaternion` `(x, y, z, w)`.
    quat: [f32; 4],
    /// Rotated vector.
    rotated: [f32; 3],
    /// Padding lane after the rotated vector.
    pad0: f32,
}

/// Uniform parameters for one dispatch: the pair count plus three pad words to
/// fill a `16`-byte, `16`-byte-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of pairs in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A compiled, reusable quaternion-rotation compute pipeline.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::quaternion_rotate`；
/// no third-party engine source or derived code.
pub struct GpuQuaternionRotate {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuQuaternionRotate {
    /// Compiles the quaternion-rotation kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::quaternion_rotate`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuQuaternionRotate {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_quaternion_rotate"),
            source: ShaderSource::Wgsl(QUATERNION_ROTATE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_quaternion_rotate_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_quaternion_rotate_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_quaternion_rotate_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuQuaternionRotate {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`QuatRotateResult`] per
    /// input, in order.
    ///
    /// Each result equals the reference
    /// [`Quat::normalize`](prism_render_architecture::particle::quaternion_rotate::Quat::normalize)
    /// and
    /// [`Quat::rotate_vec3`](prism_render_architecture::particle::quaternion_rotate::Quat::rotate_vec3)
    /// answers to within the tolerance documented on this module. An empty input
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::quaternion_rotate`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[QuatRotateQuery]) -> Vec<QuatRotateResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_quaternion_rotate_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_quaternion_rotate_output"),
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
            label: Some("prism_volumetric_quaternion_rotate_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_quaternion_rotate_bind_group"),
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
            label: Some("prism_volumetric_quaternion_rotate_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_quaternion_rotate_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_quaternion_rotate_pass"),
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

/// Decodes one packed [`GpuResult`] into the public [`QuatRotateResult`].
fn decode_result(raw: &GpuResult) -> QuatRotateResult {
    QuatRotateResult {
        normalized: Quat::new(raw.quat[0], raw.quat[1], raw.quat[2], raw.quat[3]),
        rotated: raw.rotated,
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
