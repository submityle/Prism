//! `wgpu` compute twin of the rigid mass-property contract
//! ([`inertia_tensor`](prism_render_architecture::particle::inertia_tensor),
//! particle design §8, §10 rigid-body coupling).
//!
//! The `CPU` golden
//! [`inertia_tensor`](prism_render_architecture::particle::inertia_tensor) owns
//! the classic rigid-body mass properties of a finite point-mass cloud: the
//! total mass, the mass-weighted centre of mass (`COM`), and the symmetric
//! `3x3` inertia tensor taken about that `COM`, plus the per-body algebra that
//! shifts and combines those tensors (the parallel-axis translate, the
//! re-reference `inertia_about`, and the two-system `merge`). It draws on
//! nothing transcendental — only `+ - * /` — and never compares `f32` with `==`.
//! [`GpuInertiaTensor`] is the on-device twin, so a passing real-device parity
//! test is direct evidence the ported kernels compute the same mass properties
//! the reference does, not merely that the shaders compile.
//!
//! # What is twinned
//!
//! Two complementary kernels cover the whole contract:
//!
//! * The **point-cloud reduction** reproduces
//!   [`MassProperties::of`](prism_render_architecture::particle::inertia_tensor::MassProperties::of),
//!   [`MassProperties::total_mass`](prism_render_architecture::particle::inertia_tensor::MassProperties::total_mass)
//!   and
//!   [`MassProperties::center_of_mass`](prism_render_architecture::particle::inertia_tensor::MassProperties::center_of_mass)
//!   with **one thread per point**. A first pass `atomicAdd`s each point's mass
//!   and its mass-weighted position (the first moment); the host divides to
//!   recover the `COM`; a second pass `atomicAdd`s each point's inertia
//!   contribution relative to that `COM`. Both passes accumulate `f32` through a
//!   `bitcast` `atomicCompareExchangeWeak` compare-and-swap loop, since `WGSL`
//!   has no native float atomic.
//!
//! * The **per-body algebra** reproduces, one thread per body, the pure
//!   [`Vec3`](prism_render_architecture::particle::inertia_tensor::Vec3) /
//!   [`Mat3`](prism_render_architecture::particle::inertia_tensor::Mat3)
//!   operations: the parallel-axis
//!   [`Mat3::translate`](prism_render_architecture::particle::inertia_tensor::Mat3::translate),
//!   the re-reference
//!   [`MassProperties::inertia_about`](prism_render_architecture::particle::inertia_tensor::MassProperties::inertia_about),
//!   and the two-system
//!   [`MassProperties::merge`](prism_render_architecture::particle::inertia_tensor::MassProperties::merge)
//!   (which composes `Vec3` add/scale, `Mat3` add, and the parallel-axis
//!   translate). The zero-mass guard that collapses a merge to
//!   [`MassProperties::EMPTY`](prism_render_architecture::particle::inertia_tensor::MassProperties::EMPTY)
//!   is mirrored with the same [`MASS_EPS`](prism_render_architecture::particle::inertia_tensor::MASS_EPS)
//!   threshold.
//!
//! The host owns only the single `COM` divide between the two reduction passes
//! (exactly the reference's `inv = 1.0 / total_mass`); every multiply and add is
//! on-device.
//!
//! # Portability
//!
//! The kernels use only the portable core-`WGSL` subset — `+ - * /`,
//! `bitcast`, `atomicLoad` and `atomicCompareExchangeWeak` — with no `sqrt`,
//! `sin`, `cos`, `exp`, `log`, `pow` or optional device feature, so they run
//! unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! The per-body algebra is a fixed, non-reorderable sequence of multiplies and
//! adds, so `CPU` and `GPU` evaluate the same closed form; they are not
//! bit-exact only because a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate. The point-cloud reduction has one extra source of slack: the
//! float compare-and-swap accumulation sums the points in a hardware-defined,
//! run-to-run **non-deterministic order**, whereas the reference sums them left
//! to right, so the two totals differ by the rounding of a reordered sum. The
//! parity test therefore allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3`
//! (`REL_FLOOR` `1e-6`) on the per-body fields and a slightly wider absolute
//! floor on the reduced quantities, tight enough to catch a genuinely wrong port
//! yet loose enough to admit both legal fused multiply-add contraction and the
//! reordered-sum rounding.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`inertia_tensor`](prism_render_architecture::particle::inertia_tensor); no
//! third-party engine source or derived code.
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::inertia_tensor::{MassProperties, Mat3, Vec3, MASS_EPS};
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

/// First-moment accumulator slot count: total mass plus the three mass-weighted
/// position components, filling one `vec4` of atomics.
const MOMENT_SLOTS: usize = 4;

/// Inertia accumulator slot count: the six unique symmetric tensor entries
/// (`xx, yy, zz, xy, xz, yz`).
const INERTIA_SLOTS: usize = 6;

/// The portable core-`WGSL` inertia-tensor kernels, embedded inline so the twin
/// ships as a single source file. The entry points `accumulate_moments` and
/// `accumulate_inertia` mirror the two deterministic passes of the `CPU` golden
/// [`MassProperties::of`](prism_render_architecture::particle::inertia_tensor::MassProperties::of),
/// and `body_ops` mirrors the per-body
/// [`Mat3::translate`](prism_render_architecture::particle::inertia_tensor::Mat3::translate),
/// [`MassProperties::inertia_about`](prism_render_architecture::particle::inertia_tensor::MassProperties::inertia_about)
/// and
/// [`MassProperties::merge`](prism_render_architecture::particle::inertia_tensor::MassProperties::merge);
/// see the module documentation for the algorithm.
const INERTIA_TENSOR_WGSL: &str = r#"
// Inertia-tensor twin: a point-cloud reduction (one thread per point) that
// atomicAdds the mass, the mass-weighted first moment and the inertia tensor
// about the COM, plus a per-body kernel (one thread per body) that reproduces
// the parallel-axis translate, the inertia_about re-reference and the two-system
// merge. It mirrors the CPU golden `particle::inertia_tensor`, uses only the
// portable core-WGSL subset (+ - * /, bitcast, atomicLoad and
// atomicCompareExchangeWeak), needs no sqrt and takes no optional feature, so it
// runs unmodified on Metal, Vulkan and DX12.
//
// WGSL has no native f32 atomic, so float accumulation uses a bitcast
// compare-and-swap loop; the summation order across threads is hardware-defined
// and run-to-run non-deterministic, which perturbs the reduced sums by the
// rounding of a reordered addition relative to the reference's left-to-right
// sum (see the module docs for the parity tolerance that absorbs this).
//
// Provenance: twinned from this repository's particle::inertia_tensor; no
// third-party engine source or derived code.

// Masses at or below this magnitude are treated as zero, so the COM and inertia
// divisions stay guarded. Copied verbatim from the golden `MASS_EPS`.
const MASS_EPS: f32 = 1.0e-12;

// Reduction uniforms: the point count plus the COM the inertia pass references
// (the moments pass ignores `com`).
struct ReduceParams {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    com: vec3<f32>,
    pad3: f32,
}

// Per-body uniforms: the body count padded to a 16-byte uniform slot.
struct BodyParams {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// The six unique entries of a symmetric 3x3 matrix, matching the golden `Mat3`.
struct Mat3 {
    xx: f32,
    yy: f32,
    zz: f32,
    xy: f32,
    xz: f32,
    yz: f32,
}

// One per-body query: two systems `(total_mass, com, inertia)` plus the `about`
// reference point, each inertia tensor packed as two vec4 lanes.
struct BodyQuery {
    a_mass_com: vec4<f32>,
    a_in0: vec4<f32>,
    a_in1: vec4<f32>,
    b_mass_com: vec4<f32>,
    b_in0: vec4<f32>,
    b_in1: vec4<f32>,
    about: vec4<f32>,
}

// One per-body result: the direct translate, the inertia_about re-reference, and
// the merged system, each tensor packed as two vec4 lanes.
struct BodyResult {
    translated0: vec4<f32>,
    translated1: vec4<f32>,
    about0: vec4<f32>,
    about1: vec4<f32>,
    merged_mass_com: vec4<f32>,
    merged_in0: vec4<f32>,
    merged_in1: vec4<f32>,
}

// Reduction bindings (used by accumulate_moments / accumulate_inertia).
@group(0) @binding(0) var<uniform> reduce_params: ReduceParams;
@group(0) @binding(1) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> accum: array<atomic<u32>>;

// Per-body bindings (used by body_ops).
@group(0) @binding(3) var<uniform> body_params: BodyParams;
@group(0) @binding(4) var<storage, read> queries: array<BodyQuery>;
@group(0) @binding(5) var<storage, read_write> results: array<BodyResult>;

// Entry-wise sum of two symmetric matrices, mirroring the golden `Mat3::add`.
fn mat3_add(a: Mat3, b: Mat3) -> Mat3 {
    return Mat3(
        a.xx + b.xx,
        a.yy + b.yy,
        a.zz + b.zz,
        a.xy + b.xy,
        a.xz + b.xz,
        a.yz + b.yz,
    );
}

// The inertia contribution of a point mass `m` at `r`, the golden
// `point_inertia`: I_xx = m (y^2 + z^2), I_xy = -m x y, and symmetric partners.
fn point_inertia(m: f32, r: vec3<f32>) -> Mat3 {
    let x = r.x;
    let y = r.y;
    let z = r.z;
    return Mat3(
        m * (y * y + z * z),
        m * (x * x + z * z),
        m * (x * x + y * y),
        -m * x * y,
        -m * x * z,
        -m * y * z,
    );
}

// Parallel-axis (Huygens-Steiner) shift of a tensor by `offset`, the golden
// `Mat3::translate`.
fn mat3_translate(tensor: Mat3, total_mass: f32, offset: vec3<f32>) -> Mat3 {
    return mat3_add(tensor, point_inertia(total_mass, offset));
}

// Adds `value` to the f32 held (as a bit pattern) in accum[index], via a
// compare-and-swap loop since WGSL has no native f32 atomic. The cross-thread
// summation order is hardware-defined, hence non-deterministic.
fn accum_add(index: u32, value: f32) {
    var old_bits = atomicLoad(&accum[index]);
    loop {
        let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + value);
        let swap = atomicCompareExchangeWeak(&accum[index], old_bits, new_bits);
        if (swap.exchanged) {
            break;
        }
        old_bits = swap.old_value;
    }
}

// Pass 1: accumulate the total mass and the mass-weighted position (first
// moment), mirroring the first loop of the golden `MassProperties::of`.
@compute @workgroup_size(64)
fn accumulate_moments(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= reduce_params.count) {
        return;
    }
    let p = points[idx];
    let m = p.w;
    accum_add(0u, m);
    accum_add(1u, p.x * m);
    accum_add(2u, p.y * m);
    accum_add(3u, p.z * m);
}

// Pass 2: accumulate the inertia tensor of every point relative to the COM the
// host recovered from pass 1, mirroring the second loop of `MassProperties::of`.
@compute @workgroup_size(64)
fn accumulate_inertia(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= reduce_params.count) {
        return;
    }
    let p = points[idx];
    let m = p.w;
    let r = p.xyz - reduce_params.com;
    let pi = point_inertia(m, r);
    accum_add(0u, pi.xx);
    accum_add(1u, pi.yy);
    accum_add(2u, pi.zz);
    accum_add(3u, pi.xy);
    accum_add(4u, pi.xz);
    accum_add(5u, pi.yz);
}

// One thread per body reproduces the per-body algebra: the direct parallel-axis
// translate, the inertia_about re-reference and the two-system merge.
@compute @workgroup_size(64)
fn body_ops(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= body_params.count) {
        return;
    }
    let q = queries[idx];
    let a_mass = q.a_mass_com.x;
    let a_com = q.a_mass_com.yzw;
    let a_in = Mat3(q.a_in0.x, q.a_in0.y, q.a_in0.z, q.a_in0.w, q.a_in1.x, q.a_in1.y);
    let b_mass = q.b_mass_com.x;
    let b_com = q.b_mass_com.yzw;
    let b_in = Mat3(q.b_in0.x, q.b_in0.y, q.b_in0.z, q.b_in0.w, q.b_in1.x, q.b_in1.y);
    let about = q.about.xyz;

    // Mat3::translate(a_in, a_mass, about): the raw parallel-axis shift.
    let translated = mat3_translate(a_in, a_mass, about);
    // MassProperties::inertia_about(a, about) = a_in.translate(a_mass, com - about).
    let about_tensor = mat3_translate(a_in, a_mass, a_com - about);

    // MassProperties::merge(a, b): combine mass, COM and the tensor about the
    // combined COM, collapsing to EMPTY below the zero-mass guard.
    let total = a_mass + b_mass;
    var m_mass = 0.0;
    var m_com = vec3<f32>(0.0, 0.0, 0.0);
    var m_in = Mat3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    if (total > MASS_EPS) {
        let inv = 1.0 / total;
        m_mass = total;
        m_com = (a_com * a_mass + b_com * b_mass) * inv;
        let a_shift = mat3_translate(a_in, a_mass, a_com - m_com);
        let b_shift = mat3_translate(b_in, b_mass, b_com - m_com);
        m_in = mat3_add(a_shift, b_shift);
    }

    var out: BodyResult;
    out.translated0 = vec4<f32>(translated.xx, translated.yy, translated.zz, translated.xy);
    out.translated1 = vec4<f32>(translated.xz, translated.yz, 0.0, 0.0);
    out.about0 = vec4<f32>(about_tensor.xx, about_tensor.yy, about_tensor.zz, about_tensor.xy);
    out.about1 = vec4<f32>(about_tensor.xz, about_tensor.yz, 0.0, 0.0);
    out.merged_mass_com = vec4<f32>(m_mass, m_com.x, m_com.y, m_com.z);
    out.merged_in0 = vec4<f32>(m_in.xx, m_in.yy, m_in.zz, m_in.xy);
    out.merged_in1 = vec4<f32>(m_in.xz, m_in.yz, 0.0, 0.0);
    results[idx] = out;
}
"#;

/// One per-body query: two systems `a` and `b` and the reference point `about`,
/// the inputs the reference's
/// [`Mat3::translate`](prism_render_architecture::particle::inertia_tensor::Mat3::translate),
/// [`MassProperties::inertia_about`](prism_render_architecture::particle::inertia_tensor::MassProperties::inertia_about)
/// and
/// [`MassProperties::merge`](prism_render_architecture::particle::inertia_tensor::MassProperties::merge)
/// consume.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyOpQuery {
    /// The first system: the subject of the direct translate and the
    /// `inertia_about` re-reference, and the first `merge` argument.
    pub a: MassProperties,
    /// The second system: the second `merge` argument.
    pub b: MassProperties,
    /// The reference point used both as the raw translate offset and as the
    /// `inertia_about` target.
    pub about: Vec3,
}

impl BodyOpQuery {
    /// Builds a per-body query from two systems and a reference point.
    #[must_use]
    pub const fn new(a: MassProperties, b: MassProperties, about: Vec3) -> BodyOpQuery {
        BodyOpQuery { a, b, about }
    }
}

/// The resolved per-body answer, mirroring every value the reference reports
/// across its three twinned per-body operations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyOpResult {
    /// `a.inertia` shifted by `about` via the raw parallel-axis
    /// [`Mat3::translate`](prism_render_architecture::particle::inertia_tensor::Mat3::translate).
    pub translated: Mat3,
    /// `a`'s inertia re-expressed about `about`, matching
    /// [`MassProperties::inertia_about`](prism_render_architecture::particle::inertia_tensor::MassProperties::inertia_about).
    pub inertia_about: Mat3,
    /// `a` merged with `b`, matching
    /// [`MassProperties::merge`](prism_render_architecture::particle::inertia_tensor::MassProperties::merge).
    pub merged: MassProperties,
}

/// `repr(C)` `std430` image of one point mass: the position on the first three
/// lanes and the mass on the fourth, matching the `WGSL` `array<vec4<f32>>`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuPoint {
    /// Point position.
    pos: [f32; 3],
    /// Point mass, read as the `w` lane on the device.
    mass: f32,
}

/// `repr(C)` `std430` image of the reduction uniforms: the point count on the
/// first `16`-byte word and the `COM` on its own `16`-byte-aligned `vec3` slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuReduceParams {
    /// Number of points in the storage array.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
    /// Centre of mass the inertia pass references (ignored by the moments pass).
    com: [f32; 3],
    /// Padding lane after the `COM`.
    pad3: f32,
}

/// `repr(C)` `std430` image of the per-body uniforms: the body count padded to a
/// `16`-byte uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuBodyParams {
    /// Number of bodies in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` image of one per-body query: two `(mass, com)` lanes and
/// four inertia lanes plus the `about` lane — `112` bytes matching the `WGSL`
/// `BodyQuery` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuBodyQuery {
    /// `(a.total_mass, a.com.x, a.com.y, a.com.z)`.
    a_mass_com: [f32; 4],
    /// `(a.inertia.xx, a.inertia.yy, a.inertia.zz, a.inertia.xy)`.
    a_in0: [f32; 4],
    /// `(a.inertia.xz, a.inertia.yz, 0, 0)`.
    a_in1: [f32; 4],
    /// `(b.total_mass, b.com.x, b.com.y, b.com.z)`.
    b_mass_com: [f32; 4],
    /// `(b.inertia.xx, b.inertia.yy, b.inertia.zz, b.inertia.xy)`.
    b_in0: [f32; 4],
    /// `(b.inertia.xz, b.inertia.yz, 0, 0)`.
    b_in1: [f32; 4],
    /// `(about.x, about.y, about.z, 0)`.
    about: [f32; 4],
}

/// `repr(C)` `std430` image of one per-body result: the direct translate, the
/// `inertia_about` re-reference and the merged system — `112` bytes matching the
/// `WGSL` `BodyResult` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuBodyResult {
    /// Direct translate `(xx, yy, zz, xy)`.
    translated0: [f32; 4],
    /// Direct translate `(xz, yz, 0, 0)`.
    translated1: [f32; 4],
    /// `inertia_about` `(xx, yy, zz, xy)`.
    about0: [f32; 4],
    /// `inertia_about` `(xz, yz, 0, 0)`.
    about1: [f32; 4],
    /// Merged `(total_mass, com.x, com.y, com.z)`.
    merged_mass_com: [f32; 4],
    /// Merged inertia `(xx, yy, zz, xy)`.
    merged_in0: [f32; 4],
    /// Merged inertia `(xz, yz, 0, 0)`.
    merged_in1: [f32; 4],
}

/// A compiled, reusable inertia-tensor compute pipeline bundle: the two
/// point-cloud reduction passes and the per-body algebra kernel.
pub struct GpuInertiaTensor {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    reduce_layout: BindGroupLayout,
    moments_pipeline: ComputePipeline,
    inertia_pipeline: ComputePipeline,
    body_layout: BindGroupLayout,
    body_pipeline: ComputePipeline,
}

impl GpuInertiaTensor {
    /// Compiles the inertia-tensor kernels on `ctx`.
    ///
    /// The kernels use only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuInertiaTensor {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_inertia_tensor"),
            source: ShaderSource::Wgsl(INERTIA_TENSOR_WGSL.into()),
        });

        let reduce_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_inertia_tensor_reduce_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let reduce_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_inertia_tensor_reduce_pipeline_layout"),
            bind_group_layouts: &[Some(&reduce_layout)],
            immediate_size: 0,
        });
        let moments_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_inertia_tensor_moments_pipeline"),
            layout: Some(&reduce_pipeline_layout),
            module: &module,
            entry_point: Some("accumulate_moments"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let inertia_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_inertia_tensor_inertia_pipeline"),
            layout: Some(&reduce_pipeline_layout),
            module: &module,
            entry_point: Some("accumulate_inertia"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let body_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_inertia_tensor_body_layout"),
            entries: &[
                buffer_entry(3, BufferBindingType::Uniform),
                buffer_entry(4, BufferBindingType::Storage { read_only: true }),
                buffer_entry(5, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let body_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_inertia_tensor_body_pipeline_layout"),
            bind_group_layouts: &[Some(&body_layout)],
            immediate_size: 0,
        });
        let body_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_inertia_tensor_body_pipeline"),
            layout: Some(&body_pipeline_layout),
            module: &module,
            entry_point: Some("body_ops"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        GpuInertiaTensor {
            module,
            reduce_layout,
            moments_pipeline,
            inertia_pipeline,
            body_layout,
            body_pipeline,
        }
    }

    /// Reproduces
    /// [`MassProperties::of`](prism_render_architecture::particle::inertia_tensor::MassProperties::of):
    /// the total mass, centre of mass and inertia tensor about that `COM` of
    /// `points`, or [`None`] when the total mass is at or below
    /// [`MASS_EPS`](prism_render_architecture::particle::inertia_tensor::MASS_EPS).
    ///
    /// Runs the first-moment pass, divides on the host to recover the `COM`, then
    /// runs the inertia pass about that `COM`. An empty input issues **no
    /// dispatch** (a storage buffer cannot be zero-sized) and reports [`None`].
    #[must_use]
    pub fn mass_properties(
        &self,
        ctx: &GpuContext,
        points: &[(Vec3, f32)],
    ) -> Option<MassProperties> {
        if points.is_empty() {
            return None;
        }
        let packed = pack_points(points);
        let moments = self.run_reduce(
            ctx,
            &packed,
            Vec3::ZERO,
            MOMENT_SLOTS,
            &self.moments_pipeline,
        );
        let total_mass = moments[0];
        if total_mass <= MASS_EPS {
            return None;
        }
        let inv = 1.0 / total_mass;
        let com = Vec3::new(moments[1] * inv, moments[2] * inv, moments[3] * inv);
        let raw = self.run_reduce(ctx, &packed, com, INERTIA_SLOTS, &self.inertia_pipeline);
        let inertia = Mat3::new(raw[0], raw[1], raw[2], raw[3], raw[4], raw[5]);
        Some(MassProperties {
            total_mass,
            com,
            inertia,
        })
    }

    /// Reproduces
    /// [`MassProperties::total_mass`](prism_render_architecture::particle::inertia_tensor::MassProperties::total_mass):
    /// the summed mass of `points`. An empty input reports `0.0` with no
    /// dispatch.
    #[must_use]
    pub fn total_mass(&self, ctx: &GpuContext, points: &[(Vec3, f32)]) -> f32 {
        if points.is_empty() {
            return 0.0;
        }
        let packed = pack_points(points);
        let moments = self.run_reduce(
            ctx,
            &packed,
            Vec3::ZERO,
            MOMENT_SLOTS,
            &self.moments_pipeline,
        );
        moments[0]
    }

    /// Reproduces
    /// [`MassProperties::center_of_mass`](prism_render_architecture::particle::inertia_tensor::MassProperties::center_of_mass):
    /// the mass-weighted centroid of `points`, or [`None`] when the total mass
    /// is at or below
    /// [`MASS_EPS`](prism_render_architecture::particle::inertia_tensor::MASS_EPS).
    /// An empty input reports [`None`] with no dispatch.
    #[must_use]
    pub fn center_of_mass(&self, ctx: &GpuContext, points: &[(Vec3, f32)]) -> Option<Vec3> {
        if points.is_empty() {
            return None;
        }
        let packed = pack_points(points);
        let moments = self.run_reduce(
            ctx,
            &packed,
            Vec3::ZERO,
            MOMENT_SLOTS,
            &self.moments_pipeline,
        );
        let total = moments[0];
        if total <= MASS_EPS {
            return None;
        }
        let inv = 1.0 / total;
        Some(Vec3::new(
            moments[1] * inv,
            moments[2] * inv,
            moments[3] * inv,
        ))
    }

    /// Solves every per-body query on-device and returns one [`BodyOpResult`]
    /// per input, in order.
    ///
    /// Each result reproduces the reference's raw
    /// [`Mat3::translate`](prism_render_architecture::particle::inertia_tensor::Mat3::translate),
    /// [`MassProperties::inertia_about`](prism_render_architecture::particle::inertia_tensor::MassProperties::inertia_about)
    /// and
    /// [`MassProperties::merge`](prism_render_architecture::particle::inertia_tensor::MassProperties::merge).
    /// An empty input returns an empty vector with no dispatch issued.
    #[must_use]
    pub fn body_ops(&self, ctx: &GpuContext, queries: &[BodyOpQuery]) -> Vec<BodyOpResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuBodyQuery> = queries.iter().map(pack_body_query).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_inertia_tensor_body_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuBodyResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_inertia_tensor_body_output"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = GpuBodyParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_inertia_tensor_body_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_inertia_tensor_body_bind_group"),
            layout: &self.body_layout,
            entries: &[
                BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: input_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_inertia_tensor_body_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_inertia_tensor_body_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_inertia_tensor_body_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.body_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per body, flattened to a 1-D dispatch.
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
        let raw = bytemuck::cast_slice::<u8, GpuBodyResult>(&view).to_vec();
        drop(view);
        stage.unmap();

        raw.iter().map(decode_body_result).collect()
    }

    /// Runs one reduction pass (moments or inertia) and reads back its `slots`
    /// `f32` accumulators, decoded from their `u32` bit patterns.
    fn run_reduce(
        &self,
        ctx: &GpuContext,
        points: &[GpuPoint],
        com: Vec3,
        slots: usize,
        pipeline: &ComputePipeline,
    ) -> Vec<f32> {
        let device = ctx.device();
        let count = points.len();

        let points_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_inertia_tensor_points"),
            contents: bytemuck::cast_slice(points),
            usage: BufferUsages::STORAGE,
        });

        // Explicitly zero-initialized so every compare-and-swap accumulates from
        // +0.0 regardless of the backend's buffer-clearing policy.
        let zeros = vec![0u32; slots];
        let accum_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_inertia_tensor_accum"),
            contents: bytemuck::cast_slice(&zeros),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });

        let params = GpuReduceParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
            com: [com.x, com.y, com.z],
            pad3: 0.0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_inertia_tensor_reduce_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_inertia_tensor_reduce_bind_group"),
            layout: &self.reduce_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: points_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: accum_buf.as_entire_binding(),
                },
            ],
        });

        let out_bytes = (slots * size_of::<u32>()) as u64;
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_inertia_tensor_reduce_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_inertia_tensor_reduce_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_inertia_tensor_reduce_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per point, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&accum_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let bits = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        stage.unmap();

        bits.iter().map(|&b| f32::from_bits(b)).collect()
    }
}

/// Packs a `(position, mass)` slice into its `std430` images.
fn pack_points(points: &[(Vec3, f32)]) -> Vec<GpuPoint> {
    points
        .iter()
        .map(|&(p, m)| GpuPoint {
            pos: [p.x, p.y, p.z],
            mass: m,
        })
        .collect()
}

/// Packs one [`BodyOpQuery`] into its `std430` image.
fn pack_body_query(query: &BodyOpQuery) -> GpuBodyQuery {
    GpuBodyQuery {
        a_mass_com: [
            query.a.total_mass,
            query.a.com.x,
            query.a.com.y,
            query.a.com.z,
        ],
        a_in0: [
            query.a.inertia.xx,
            query.a.inertia.yy,
            query.a.inertia.zz,
            query.a.inertia.xy,
        ],
        a_in1: [query.a.inertia.xz, query.a.inertia.yz, 0.0, 0.0],
        b_mass_com: [
            query.b.total_mass,
            query.b.com.x,
            query.b.com.y,
            query.b.com.z,
        ],
        b_in0: [
            query.b.inertia.xx,
            query.b.inertia.yy,
            query.b.inertia.zz,
            query.b.inertia.xy,
        ],
        b_in1: [query.b.inertia.xz, query.b.inertia.yz, 0.0, 0.0],
        about: [query.about.x, query.about.y, query.about.z, 0.0],
    }
}

/// Decodes one packed [`GpuBodyResult`] into the public [`BodyOpResult`].
fn decode_body_result(raw: &GpuBodyResult) -> BodyOpResult {
    BodyOpResult {
        translated: Mat3::new(
            raw.translated0[0],
            raw.translated0[1],
            raw.translated0[2],
            raw.translated0[3],
            raw.translated1[0],
            raw.translated1[1],
        ),
        inertia_about: Mat3::new(
            raw.about0[0],
            raw.about0[1],
            raw.about0[2],
            raw.about0[3],
            raw.about1[0],
            raw.about1[1],
        ),
        merged: MassProperties {
            total_mass: raw.merged_mass_com[0],
            com: Vec3::new(
                raw.merged_mass_com[1],
                raw.merged_mass_com[2],
                raw.merged_mass_com[3],
            ),
            inertia: Mat3::new(
                raw.merged_in0[0],
                raw.merged_in0[1],
                raw.merged_in0[2],
                raw.merged_in0[3],
                raw.merged_in1[0],
                raw.merged_in1[1],
            ),
        },
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
