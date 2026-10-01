//! `wgpu` compute twin of the unit dual-quaternion rigid-transform golden
//! ([`dual_quaternion`](prism_render_architecture::particle::dual_quaternion),
//! particle design §16, §24).
//!
//! The `CPU` golden
//! [`dual_quaternion`](prism_render_architecture::particle::dual_quaternion) owns
//! scale-free rigid-transform algebra: a rotation plus translation stored as a
//! `DualQuat` (`real + dual*ε`), where `real` is a unit rotation quaternion and
//! `dual = 0.5 * (t_quat ⊗ real)` encodes the translation. The end-to-end
//! operation a skinning stage consumes is
//! [`DualQuat::transform_point`](prism_render_architecture::particle::dual_quaternion::DualQuat::transform_point):
//! build the transform from a rotation quaternion and a translation
//! ([`DualQuat::from_rotation_translation`](prism_render_architecture::particle::dual_quaternion::DualQuat::from_rotation_translation)),
//! then rotate a point and translate it.
//!
//! [`GpuDualQuaternion`] is the on-device twin: one thread per
//! `(rotation, translation, point)` query reproduces the same `Hamilton`-product
//! pipeline branch for branch, so a passing real-device parity test is direct
//! evidence the ported kernel evaluates the same rigid transform and classifies
//! the same degenerate case (a near-zero rotation quaternion, which falls back
//! to the identity rotation) the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! Each query reproduces the full transform chain the reference exposes: the
//! `Quat` building blocks (`norm` with its one `sqrt`, the guarded `normalized`,
//! `conjugate`, `scale` and the non-commutative `Hamilton` product `hamilton`),
//! the `DualQuat` construction `from_rotation_translation`, the renormalization
//! onto the unit-rigid manifold, the `to_rotation_translation` recovery and the
//! final rotate-then-translate of the point. The result is the transformed
//! point.
//!
//! # Degenerate inputs
//!
//! Both guarded divisions the reference performs are mirrored. A rotation
//! quaternion whose norm is at or below the compare epsilon has no stable
//! direction, so `Quat::normalized` falls back to the identity rotation (the
//! point is then only translated) instead of dividing by a near-zero magnitude;
//! and the dual-quaternion renormalization inside `transform_point` applies the
//! same guard on the `real` norm, falling back to the identity transform rather
//! than emitting a `NaN`. An empty query batch short-circuits on the host with
//! no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+ - * /`, the scalar
//! `*` on a `vec4` and one `sqrt` for the genuine quaternion norm — with no
//! `sin`, `cos`, `acos`, `exp`, `log`, `pow`, `tan` and no optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. Rotations
//! arrive as ready-made quaternions, so construction is pure `Hamilton`-product
//! arithmetic with no transcendental call. There is no loop: each thread
//! performs a fixed, bounded sequence of arithmetic, so the kernel provably
//! terminates.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds, divides
//! and one `sqrt`, so `CPU` and `GPU` evaluate the same closed form in the same
//! associativity. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on each `f32` lane of the
//! transformed point, tight enough to catch a genuinely wrong port (a dropped
//! branch, a swapped `Hamilton` coefficient, a wrong scale) yet loose enough to
//! admit legal fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::dual_quaternion`；
//! textbook unit dual-quaternion rigid-transform algebra plus `wgpu` compute
//! dispatch; no third-party engine source or derived code.
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::dual_quaternion::{DualQuat, Quat};
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

/// The portable core-`WGSL` dual-quaternion rigid-transform kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden
/// [`dual_quaternion`](prism_render_architecture::particle::dual_quaternion)
/// branch for branch; see the module documentation for the algorithm.
const DUAL_QUATERNION_WGSL: &str = r#"
// Dual-quaternion rigid-transform twin: one thread per (rotation, translation,
// point) query builds the DualQuat from a rotation quaternion and a translation,
// renormalizes it onto the unit-rigid manifold and transforms the point
// (rotate, then translate). It mirrors the CPU golden
// particle::dual_quaternion branch for branch, uses only the portable core-WGSL
// subset (+ - * / plus one sqrt for the quaternion norm), has no transcendental
// call and takes no optional feature, so it runs unmodified on Metal, Vulkan and
// DX12. There is no loop, so the kernel provably terminates.
//
// Provenance: twinned from this repository's particle::dual_quaternion; no
// third-party engine source or derived code.

// Magnitude below which a quaternion norm is treated as degenerate; the guarded
// divisions fall back to the identity instead of writing an exact == / != on an
// f32 or emitting a NaN. Matches the reference `CMP_EPS`.
const CMP_EPS: f32 = 1.0e-6;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Rotation quaternion (x, y, z, w), accepted pre-normalization.
    rot: vec4<f32>,
    // Translation applied after the rotation; a pad lane follows.
    translation: vec3<f32>,
    pad0: f32,
    // Point to transform; a pad lane follows.
    point: vec3<f32>,
    pad1: f32,
}

struct Result {
    // Transformed point and one pad lane: four scalars filling one vec4 slot.
    transformed: vec3<f32>,
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// The identity rotation (0, 0, 0, 1), returned when a quaternion has no stable
// direction, mirroring the reference `Quat::identity`.
fn q_identity() -> vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

// The Euclidean norm sqrt(x² + y² + z² + w²), mirroring `Quat::norm`.
fn q_norm(q: vec4<f32>) -> f32 {
    return sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
}

// The unit quaternion q / norm, mirroring `Quat::normalized`. A near-zero
// quaternion has no stable direction, so the identity is returned as a fallback.
fn q_normalized(q: vec4<f32>) -> vec4<f32> {
    let n = q_norm(q);
    if (n < CMP_EPS) {
        return q_identity();
    }
    let inv = 1.0 / n;
    return q * inv;
}

// The conjugate (-x, -y, -z, w), mirroring `Quat::conjugate`.
fn q_conjugate(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}

// The Hamilton product a * b composing two rotations (apply b first, then a),
// mirroring `Quat::hamilton`. Non-commutative, not a component-wise multiply.
fn q_hamilton(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    let x1 = a.x;
    let y1 = a.y;
    let z1 = a.z;
    let w1 = a.w;
    let x2 = b.x;
    let y2 = b.y;
    let z2 = b.z;
    let w2 = b.w;
    return vec4<f32>(
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    );
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // from_rotation_translation: renormalize the rotation, then
    // dual = 0.5 * (t_quat ⊗ real).
    let real = q_normalized(q.rot);
    let t_quat = vec4<f32>(q.translation.x, q.translation.y, q.translation.z, 0.0);
    let dual = q_hamilton(t_quat, real) * 0.5;

    // transform_point first renormalizes the DualQuat onto the unit-rigid
    // manifold. A near-zero `real` has no stable rotation, so the identity
    // transform is returned (real = identity, dual = 0), mirroring
    // `DualQuat::normalized`.
    let n = q_norm(real);
    var u_real: vec4<f32> = q_identity();
    var u_dual: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if (n >= CMP_EPS) {
        let inv = 1.0 / n;
        u_real = real * inv;
        u_dual = dual * inv;
    }

    // to_rotation_translation: t = 2 * (dual ⊗ real*), taking the vector part.
    let t_rec = q_hamilton(u_dual, q_conjugate(u_real)) * 2.0;

    // Rotate the point: rot ⊗ (p, 0) ⊗ rot*, then translate.
    let pv = vec4<f32>(q.point.x, q.point.y, q.point.z, 0.0);
    let rotated = q_hamilton(q_hamilton(u_real, pv), q_conjugate(u_real));

    var out: Result;
    out.transformed = vec3<f32>(
        rotated.x + t_rec.x,
        rotated.y + t_rec.y,
        rotated.z + t_rec.z,
    );
    out.pad0 = 0.0;
    results[idx] = out;
}
"#;

/// One dual-quaternion rigid-transform query: a rotation quaternion `rot`
/// (accepted pre-normalization), a `translation` and the `point` to transform —
/// the same inputs the reference
/// [`DualQuat::from_rotation_translation`](prism_render_architecture::particle::dual_quaternion::DualQuat::from_rotation_translation)
/// and
/// [`DualQuat::transform_point`](prism_render_architecture::particle::dual_quaternion::DualQuat::transform_point)
/// consume.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::dual_quaternion`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DualQuatTransformQuery {
    /// Rotation quaternion `(x, y, z, w)`, renormalized by the transform.
    pub rot: Quat,
    /// Translation applied after the rotation.
    pub translation: [f32; 3],
    /// The point to rigidly transform.
    pub point: [f32; 3],
}

impl DualQuatTransformQuery {
    /// Builds a query from a rotation quaternion, a translation and a point.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::dual_quaternion`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub const fn new(rot: Quat, translation: [f32; 3], point: [f32; 3]) -> DualQuatTransformQuery {
        DualQuatTransformQuery {
            rot,
            translation,
            point,
        }
    }
}

/// Evaluates the `CPU` golden for one query: builds the `DualQuat` from the
/// rotation and translation and transforms the point, returning the rigidly
/// transformed point.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::dual_quaternion`；
/// no third-party engine source or derived code.
#[must_use]
pub fn golden(query: &DualQuatTransformQuery) -> [f32; 3] {
    DualQuat::from_rotation_translation(&query.rot, query.translation).transform_point(query.point)
}

/// `repr(C)` `std430` layout of one packed query: three `vec4` slots holding
/// `(rot.xyzw)`, `(translation.xyz, pad)` and `(point.xyz, pad)` — `48` bytes,
/// each `vec3` on its `16`-byte-aligned slot exactly as the `WGSL` `Query`
/// struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Rotation quaternion `(x, y, z, w)`.
    rot: [f32; 4],
    /// Translation vector.
    translation: [f32; 3],
    /// Padding lane after the translation.
    pad0: f32,
    /// Point to transform.
    point: [f32; 3],
    /// Padding lane after the point.
    pad1: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &DualQuatTransformQuery) -> GpuQuery {
        GpuQuery {
            rot: [query.rot.x, query.rot.y, query.rot.z, query.rot.w],
            translation: query.translation,
            pad0: 0.0,
            point: query.point,
            pad1: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: a four-scalar slot
/// `(transformed.xyz, pad)` — `16` bytes matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Transformed point.
    transformed: [f32; 3],
    /// Padding lane.
    pad0: f32,
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

/// A compiled, reusable dual-quaternion rigid-transform compute pipeline.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::dual_quaternion`；
/// no third-party engine source or derived code.
pub struct GpuDualQuaternion {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuDualQuaternion {
    /// Compiles the dual-quaternion kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::dual_quaternion`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDualQuaternion {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_dual_quaternion"),
            source: ShaderSource::Wgsl(DUAL_QUATERNION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_dual_quaternion_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_dual_quaternion_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_dual_quaternion_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuDualQuaternion {
            module,
            layout,
            pipeline,
        }
    }

    /// Transforms every query's point on-device and returns one transformed
    /// point per input, in order.
    ///
    /// Each result equals the reference
    /// [`DualQuat::transform_point`](prism_render_architecture::particle::dual_quaternion::DualQuat::transform_point)
    /// answer (for the `DualQuat` built by
    /// [`DualQuat::from_rotation_translation`](prism_render_architecture::particle::dual_quaternion::DualQuat::from_rotation_translation))
    /// to within the tolerance documented on this module. An empty input returns
    /// an empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::dual_quaternion`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn transform(&self, ctx: &GpuContext, queries: &[DualQuatTransformQuery]) -> Vec<[f32; 3]> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_dual_quaternion_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_dual_quaternion_output"),
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
            label: Some("prism_volumetric_dual_quaternion_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_dual_quaternion_bind_group"),
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
            label: Some("prism_volumetric_dual_quaternion_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_dual_quaternion_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_dual_quaternion_pass"),
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

/// Decodes one packed [`GpuResult`] into the public transformed point.
fn decode_result(raw: &GpuResult) -> [f32; 3] {
    raw.transformed
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
