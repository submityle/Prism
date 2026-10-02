//! `wgpu` compute twin of the per-particle mesh-instance transform-assembly
//! golden
//! ([`mesh_renderer`](prism_render_architecture::particle::mesh_renderer),
//! design §15).
//!
//! The `CPU` golden turns a live particle plus a unit source mesh into a `TRS`
//! instance transform, its world-space bounds and a `LOD` tier. It owns its own
//! affine algebra
//! ([`Mat3`](prism_render_architecture::particle::mesh_renderer::Mat3),
//! [`Affine3`](prism_render_architecture::particle::mesh_renderer::Affine3)) and
//! four orientation modes
//! ([`MeshOrientation`](prism_render_architecture::particle::mesh_renderer::MeshOrientation)):
//! `Identity`, velocity-aligned
//! ([`velocity_aligned_basis`](prism_render_architecture::particle::mesh_renderer::velocity_aligned_basis)),
//! a numeric `(sin, cos)` axis rotation
//! ([`fixed_rotation_basis`](prism_render_architecture::particle::mesh_renderer::fixed_rotation_basis))
//! and a minimal align-onto-axis rotation
//! ([`align_to_axis_basis`](prism_render_architecture::particle::mesh_renderer::align_to_axis_basis)).
//! The scale folds a per-axis base, a uniform multiplier and an over-life size
//! sample
//! ([`scale_matrix`](prism_render_architecture::particle::mesh_renderer::scale_matrix)),
//! the whole thing assembles via
//! [`instance_affine`](prism_render_architecture::particle::mesh_renderer::instance_affine),
//! the world bounds come from
//! [`transform_aabb`](prism_render_architecture::particle::mesh_renderer::transform_aabb)
//! and the detail tier from
//! [`select_mesh_lod`](prism_render_architecture::particle::mesh_renderer::select_mesh_lod).
//!
//! [`GpuMeshRenderer`] is the on-device twin: one thread assembles one
//! instance, reproducing the resolved rotation matrix, the linear `R * S` part,
//! the translation, the world-space `AABB` and the `LOD` tier, so a passing
//! real-device parity test is direct evidence the ported kernel folds the same
//! hand-rolled matrix algebra, the same degenerate guards and the same discrete
//! orientation and tier classification the reference does.
//!
//! # Twinned angles
//!
//! Every rotation angle enters the kernel as a numeric `(sin, cos)` pair carried
//! on the query (`rot_sin` / `rot_cos` for the fixed rotation; a cross/dot pair
//! derived in-kernel for the align-onto-axis rotation). The kernel never calls
//! `sin` or `cos`, exactly like the golden, so the two evaluate the identical
//! closed-form Rodrigues matrix.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `abs`, `select`, `+ - * /` and one `sqrt` inside each robust
//! normalize — with no `sin`, `cos`, `exp`, `log`, `pow` or optional device
//! feature, so it runs unmodified on Metal, Vulkan and DX12. The golden is
//! likewise transcendental free (only `sqrt`), so the two evaluate the same
//! closed form.
//!
//! # Correctness model
//!
//! Each instance is a fixed, non-reorderable chain of guarded matrix operations,
//! so `CPU` and `GPU` evaluate the same algebra in the same associativity. They
//! are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits by a few units in the last
//! place. The world bounds additionally use an eight-corner `min`/`max`
//! reduction on-device versus the golden's algebraically identical
//! `abs(linear)` half-extent path, which differs only by a few units in the last
//! place. The parity test therefore asserts a tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`) on every continuous value yet an *exact* match on the
//! discrete `lod_tier`, which is driven by the same guard bands the reference
//! uses so an instance placed clear of a threshold tie folds the identical
//! verdict.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓
//! [`mesh_renderer`](prism_render_architecture::particle::mesh_renderer);
//! hand-rolled per-particle affine instance-transform algebra plus `wgpu`
//! compute dispatch; no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::mesh_renderer::{
    instance_affine, orientation_matrix, select_mesh_lod, transform_aabb, Affine3, LocalAxis, Mat3,
    MeshOrientation,
};
use prism_render_architecture::particle::sort_cull::Aabb;
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

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// Orientation-mode code for the identity (authored) orientation, mirroring
/// [`MeshOrientation::Identity`](prism_render_architecture::particle::mesh_renderer::MeshOrientation).
pub const ORIENTATION_IDENTITY: u32 = 0;
/// Orientation-mode code for the velocity-aligned basis, mirroring
/// [`MeshOrientation::VelocityAligned`](prism_render_architecture::particle::mesh_renderer::MeshOrientation).
pub const ORIENTATION_VELOCITY_ALIGNED: u32 = 1;
/// Orientation-mode code for the numeric `(sin, cos)` fixed rotation, mirroring
/// [`MeshOrientation::FixedRotation`](prism_render_architecture::particle::mesh_renderer::MeshOrientation).
pub const ORIENTATION_FIXED_ROTATION: u32 = 2;
/// Orientation-mode code for the minimal align-onto-axis rotation, mirroring
/// [`MeshOrientation::AlignToAxis`](prism_render_architecture::particle::mesh_renderer::MeshOrientation).
pub const ORIENTATION_ALIGN_TO_AXIS: u32 = 3;

/// Local-axis code for the mesh's `+X` axis, mirroring
/// [`LocalAxis::PlusX`](prism_render_architecture::particle::mesh_renderer::LocalAxis).
pub const LOCAL_AXIS_PLUS_X: u32 = 0;
/// Local-axis code for the mesh's `+Y` axis, mirroring
/// [`LocalAxis::PlusY`](prism_render_architecture::particle::mesh_renderer::LocalAxis).
pub const LOCAL_AXIS_PLUS_Y: u32 = 1;
/// Local-axis code for the mesh's `+Z` axis, mirroring
/// [`LocalAxis::PlusZ`](prism_render_architecture::particle::mesh_renderer::LocalAxis).
pub const LOCAL_AXIS_PLUS_Z: u32 = 2;

/// Maximum number of `LOD` thresholds a single query carries. The on-device
/// twin uses a fixed-size threshold array, so a query may describe at most this
/// many detail tiers.
pub const MAX_LOD_THRESHOLDS: usize = 4;

/// The portable core-`WGSL` mesh-instance assembly kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`mesh_renderer`](prism_render_architecture::particle::mesh_renderer)
/// functions; see the module documentation for the algorithm.
const MESH_RENDERER_WGSL: &str = r#"
// Per-particle mesh-instance assembly twin: one thread per query reproduces the
// resolved orientation matrix, the linear R*S part, the translation, the
// world-space AABB (eight-corner min/max reduction) and the LOD tier. It
// mirrors the CPU golden particle::mesh_renderer function for function.
//
// Portability: only the core subset (min, max, clamp, abs, select, + - * / and
// one sqrt) is used; no sin/cos/exp/log/pow/tan and no optional device feature,
// so it runs unmodified on Metal, Vulkan and DX12. Every rotation angle enters
// as a numeric (sin, cos) pair, so no trigonometry is ever called.
//
// Provenance: twinned from this repository's particle::mesh_renderer; no
// third-party engine source or derived code.

// Squared-length floor guarding every normalize divide and classifying a (near)
// zero vector or axis, matching the reference EPS_LEN_SQ. A direct f32 ==/!= is
// forbidden, so the degenerate tests compare the squared length against this
// floor instead of exact zero.
const EPS_LEN_SQ: f32 = 1.0e-12;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // World position (the translation of the TRS transform).
    position: vec3<f32>,
    // Sine of the fixed-rotation angle (packed in the position slot pad lane).
    rot_sin: f32,
    // Per-particle velocity for the velocity-aligned basis.
    velocity: vec3<f32>,
    // Cosine of the fixed-rotation angle.
    rot_cos: f32,
    // Up reference completing the velocity-aligned basis.
    up_reference: vec3<f32>,
    // Uniform scale multiplier.
    uniform_scale: f32,
    // Rotation axis for the fixed-rotation mode.
    axis: vec3<f32>,
    // Over-life size sample folded into the scale.
    size_over_life: f32,
    // World direction the local axis is locked onto (align-to-axis mode).
    align_target: vec3<f32>,
    // Screen-coverage fraction selecting the LOD tier.
    coverage: f32,
    // Per-axis base scale.
    per_axis_scale: vec3<f32>,
    pad0: f32,
    // Local-space AABB minimum corner.
    local_min: vec3<f32>,
    pad1: f32,
    // Local-space AABB maximum corner.
    local_max: vec3<f32>,
    pad2: f32,
    // Descending minimum-coverage thresholds for the LOD tiers.
    thresholds: array<f32, 4>,
    // Orientation mode: 0 Identity, 1 VelocityAligned, 2 FixedRotation,
    // 3 AlignToAxis.
    orientation_mode: u32,
    // Local axis for the align-to-axis mode: 0 +X, 1 +Y, 2 +Z.
    local_axis: u32,
    // Number of valid LOD thresholds (0..=4).
    threshold_count: u32,
    pad3: u32,
}

struct Result {
    // Resolved orientation matrix, row 0.
    rot0: vec3<f32>,
    pad0: f32,
    // Resolved orientation matrix, row 1.
    rot1: vec3<f32>,
    pad1: f32,
    // Resolved orientation matrix, row 2.
    rot2: vec3<f32>,
    pad2: f32,
    // Linear R*S instance part, row 0.
    lin0: vec3<f32>,
    pad3: f32,
    // Linear R*S instance part, row 1.
    lin1: vec3<f32>,
    pad4: f32,
    // Linear R*S instance part, row 2.
    lin2: vec3<f32>,
    pad5: f32,
    // World-space translation.
    translation: vec3<f32>,
    pad6: f32,
    // World-space AABB minimum corner.
    world_min: vec3<f32>,
    pad7: f32,
    // World-space AABB maximum corner.
    world_max: vec3<f32>,
    // Selected LOD tier (packed in the world_max slot pad lane).
    lod_tier: u32,
}

// A row-major 3x3 matrix held as three rows, used only inside the kernel.
struct Mat3 {
    r0: vec3<f32>,
    r1: vec3<f32>,
    r2: vec3<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// The identity matrix.
fn mat_identity() -> Mat3 {
    return Mat3(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
}

// Builds a matrix from its three rows.
fn mat_from_rows(r0: vec3<f32>, r1: vec3<f32>, r2: vec3<f32>) -> Mat3 {
    return Mat3(r0, r1, r2);
}

// Builds a matrix from its three columns, mirroring the reference
// Mat3::from_columns: the columns become the world-space images of the local
// axes.
fn mat_from_columns(c0: vec3<f32>, c1: vec3<f32>, c2: vec3<f32>) -> Mat3 {
    return Mat3(
        vec3<f32>(c0.x, c1.x, c2.x),
        vec3<f32>(c0.y, c1.y, c2.y),
        vec3<f32>(c0.z, c1.z, c2.z),
    );
}

// Builds a diagonal (per-axis scale) matrix.
fn mat_from_diagonal(d: vec3<f32>) -> Mat3 {
    return Mat3(
        vec3<f32>(d.x, 0.0, 0.0),
        vec3<f32>(0.0, d.y, 0.0),
        vec3<f32>(0.0, 0.0, d.z),
    );
}

// The transpose, which is also the inverse for a pure-rotation matrix.
fn mat_transpose(m: Mat3) -> Mat3 {
    return mat_from_columns(m.r0, m.r1, m.r2);
}

// Matrix-times-vector: three dot products of the rows with v.
fn vmul(m: Mat3, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(m.r0, v), dot(m.r1, v), dot(m.r2, v));
}

// Matrix product a*b, mirroring the reference Mat3::mul_mat: transpose b so its
// rows are b's columns, then each output entry is a dot product of a row of a
// with a column of b.
fn mat_mul(a: Mat3, b: Mat3) -> Mat3 {
    let t = mat_transpose(b);
    return Mat3(
        vec3<f32>(dot(a.r0, t.r0), dot(a.r0, t.r1), dot(a.r0, t.r2)),
        vec3<f32>(dot(a.r1, t.r0), dot(a.r1, t.r1), dot(a.r1, t.r2)),
        vec3<f32>(dot(a.r2, t.r0), dot(a.r2, t.r1), dot(a.r2, t.r2)),
    );
}

// Returns the unit vector along v, or the zero vector when v is (numerically)
// zero, mirroring the reference normalize_or_zero: a squared length at or below
// EPS_LEN_SQ yields zero so the divide never produces a NaN.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq > EPS_LEN_SQ) {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// A deterministic unit vector perpendicular to v, or +X when v is (numerically)
// zero, mirroring the reference any_perpendicular: cross with the world axis
// least aligned with v (squared components, no transcendentals), and fall back
// to +X if that cross collapses.
fn any_perpendicular(v: vec3<f32>) -> vec3<f32> {
    let ax = v.x * v.x;
    let ay = v.y * v.y;
    let az = v.z * v.z;
    var reference = vec3<f32>(0.0, 0.0, 1.0);
    if (ax <= ay && ax <= az) {
        reference = vec3<f32>(1.0, 0.0, 0.0);
    } else if (ay <= az) {
        reference = vec3<f32>(0.0, 1.0, 0.0);
    }
    let perp = normalize_or_zero(cross(v, reference));
    if (dot(perp, perp) > EPS_LEN_SQ) {
        return perp;
    }
    return vec3<f32>(1.0, 0.0, 0.0);
}

// The Rodrigues rotation matrix for a numeric (sin, cos) angle about axis,
// mirroring the reference fixed_rotation_basis: axis is normalized internally
// and a (near) zero axis yields the identity. The angle is never computed.
fn fixed_rotation_basis(axis: vec3<f32>, s: f32, c: f32) -> Mat3 {
    let n = normalize_or_zero(axis);
    if (dot(n, n) <= EPS_LEN_SQ) {
        return mat_identity();
    }
    let x = n.x;
    let y = n.y;
    let z = n.z;
    let omc = 1.0 - c;
    return mat_from_rows(
        vec3<f32>(c + x * x * omc, x * y * omc - z * s, x * z * omc + y * s),
        vec3<f32>(y * x * omc + z * s, c + y * y * omc, y * z * omc - x * s),
        vec3<f32>(z * x * omc - y * s, z * y * omc + x * s, c + z * z * omc),
    );
}

// An orthonormal right-handed rotation whose local +X column points along
// forward, completing +Z and +Y from up_ref, mirroring the reference
// basis_from_forward_x: a zero forward falls back to world +X and a parallel
// up_ref falls back to a deterministic perpendicular.
fn basis_from_forward_x(forward: vec3<f32>, up_ref: vec3<f32>) -> Mat3 {
    var x_axis = normalize_or_zero(forward);
    if (dot(x_axis, x_axis) <= EPS_LEN_SQ) {
        x_axis = vec3<f32>(1.0, 0.0, 0.0);
    }
    var z_axis = normalize_or_zero(cross(x_axis, up_ref));
    if (dot(z_axis, z_axis) <= EPS_LEN_SQ) {
        z_axis = any_perpendicular(x_axis);
    }
    let y_axis = cross(z_axis, x_axis);
    return mat_from_columns(x_axis, y_axis, z_axis);
}

// The minimal rotation taking unit direction src_dir onto unit direction
// dst_dir, mirroring the reference rotation_between: the sine and cosine come
// from the cross and dot products, equal directions give the identity, opposite
// directions give a half turn about a deterministic perpendicular, and a zero
// input gives the identity.
fn rotation_between(src_dir: vec3<f32>, dst_dir: vec3<f32>) -> Mat3 {
    let f = normalize_or_zero(src_dir);
    let t = normalize_or_zero(dst_dir);
    if (dot(f, f) <= EPS_LEN_SQ || dot(t, t) <= EPS_LEN_SQ) {
        return mat_identity();
    }
    let c = dot(f, t);
    let axis = cross(f, t);
    let axis_len_sq = dot(axis, axis);
    if (axis_len_sq <= EPS_LEN_SQ) {
        if (c >= 0.0) {
            return mat_identity();
        }
        let perp = any_perpendicular(f);
        return fixed_rotation_basis(perp, 0.0, -1.0);
    }
    let s = sqrt(axis_len_sq);
    let n = axis * (1.0 / s);
    return fixed_rotation_basis(n, s, c);
}

// The unit vector for a local-axis code, mirroring the reference
// LocalAxis::unit; any code other than +Y or +Z folds to +X.
fn local_axis_unit(code: u32) -> vec3<f32> {
    if (code == 1u) {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    if (code == 2u) {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    return vec3<f32>(1.0, 0.0, 0.0);
}

// Resolves the orientation matrix from the query's mode, mirroring the
// reference orientation_matrix; any mode other than the three explicit ones
// folds to the identity.
fn orientation_matrix(q: Query) -> Mat3 {
    if (q.orientation_mode == 1u) {
        return basis_from_forward_x(q.velocity, q.up_reference);
    }
    if (q.orientation_mode == 2u) {
        return fixed_rotation_basis(q.axis, q.rot_sin, q.rot_cos);
    }
    if (q.orientation_mode == 3u) {
        return rotation_between(local_axis_unit(q.local_axis), q.align_target);
    }
    return mat_identity();
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // orientation_matrix: the resolved per-particle rotation.
    let rotation = orientation_matrix(q);

    // scale_matrix: fold the per-axis base, the uniform multiplier and the
    // over-life sample into one diagonal matrix.
    let effective = q.per_axis_scale * (q.uniform_scale * q.size_over_life);
    let scale_mat = mat_from_diagonal(effective);

    // instance_affine: linear = rotation * scale, translation = position.
    let linear = mat_mul(rotation, scale_mat);
    let translation = q.position;

    // transform_aabb: transform all eight corners of the local box and reduce to
    // the world-space min/max. This is algebraically the reference abs(linear)
    // half-extent path, so the two agree within the parity tolerance.
    var world_min = vec3<f32>(
        bitcast<f32>(0x7f800000u),
        bitcast<f32>(0x7f800000u),
        bitcast<f32>(0x7f800000u),
    );
    var world_max = vec3<f32>(
        bitcast<f32>(0xff800000u),
        bitcast<f32>(0xff800000u),
        bitcast<f32>(0xff800000u),
    );
    for (var i = 0u; i < 2u; i = i + 1u) {
        for (var j = 0u; j < 2u; j = j + 1u) {
            for (var k = 0u; k < 2u; k = k + 1u) {
                let sx = select(q.local_min.x, q.local_max.x, i == 1u);
                let sy = select(q.local_min.y, q.local_max.y, j == 1u);
                let sz = select(q.local_min.z, q.local_max.z, k == 1u);
                let corner = vmul(linear, vec3<f32>(sx, sy, sz)) + translation;
                world_min = min(world_min, corner);
                world_max = max(world_max, corner);
            }
        }
    }

    // select_mesh_lod: the first tier whose threshold the clamped coverage meets
    // or exceeds; below every threshold the lowest tier (threshold_count).
    let c = clamp(q.coverage, 0.0, 1.0);
    let n = min(q.threshold_count, 4u);
    var tier: u32 = q.threshold_count;
    var found = false;
    for (var i = 0u; i < n; i = i + 1u) {
        if (!found && c >= q.thresholds[i]) {
            tier = i;
            found = true;
        }
    }

    var res: Result;
    res.rot0 = rotation.r0;
    res.pad0 = 0.0;
    res.rot1 = rotation.r1;
    res.pad1 = 0.0;
    res.rot2 = rotation.r2;
    res.pad2 = 0.0;
    res.lin0 = linear.r0;
    res.pad3 = 0.0;
    res.lin1 = linear.r1;
    res.pad4 = 0.0;
    res.lin2 = linear.r2;
    res.pad5 = 0.0;
    res.translation = translation;
    res.pad6 = 0.0;
    res.world_min = world_min;
    res.pad7 = 0.0;
    res.world_max = world_max;
    res.lod_tier = tier;
    results[idx] = res;
}
"#;

/// One per-particle mesh-instance query: the world placement, the orientation
/// selector and its inputs, the scale inputs, the local-space bounds and the
/// `LOD` thresholds the reference's twinned functions consume.
///
/// `orientation_mode` selects which orientation inputs apply
/// (`ORIENTATION_IDENTITY` ignores all of them, `ORIENTATION_VELOCITY_ALIGNED`
/// uses `velocity` / `up_reference`, `ORIENTATION_FIXED_ROTATION` uses `axis`
/// with the numeric `(rot_sin, rot_cos)` pair, and `ORIENTATION_ALIGN_TO_AXIS`
/// locks `local_axis` onto `target`). `per_axis_scale`, `uniform_scale` and
/// `size_over_life` fold into the scale, `local_aabb_min` / `local_aabb_max`
/// bound the unit mesh, and `coverage` picks a tier from the first
/// `threshold_count` entries of `thresholds` (descending).
///
/// Derives only [`PartialEq`] (no `Eq`/`Hash`) because it holds `f32` data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuMeshRendererQuery {
    /// World position (the translation of the `TRS` transform).
    pub position: [f32; 3],
    /// Per-particle velocity for `ORIENTATION_VELOCITY_ALIGNED`.
    pub velocity: [f32; 3],
    /// Up reference completing the velocity-aligned basis.
    pub up_reference: [f32; 3],
    /// Rotation axis for `ORIENTATION_FIXED_ROTATION`.
    pub axis: [f32; 3],
    /// World direction the local axis locks onto for
    /// `ORIENTATION_ALIGN_TO_AXIS`.
    pub target: [f32; 3],
    /// Per-axis base scale.
    pub per_axis_scale: [f32; 3],
    /// Local-space `AABB` minimum corner.
    pub local_aabb_min: [f32; 3],
    /// Local-space `AABB` maximum corner.
    pub local_aabb_max: [f32; 3],
    /// Descending minimum-coverage thresholds; only the first `threshold_count`
    /// entries are read.
    pub thresholds: [f32; MAX_LOD_THRESHOLDS],
    /// Sine of the fixed-rotation angle.
    pub rot_sin: f32,
    /// Cosine of the fixed-rotation angle.
    pub rot_cos: f32,
    /// Uniform scale multiplier.
    pub uniform_scale: f32,
    /// Over-life size sample folded into the scale.
    pub size_over_life: f32,
    /// Screen-coverage fraction selecting the `LOD` tier.
    pub coverage: f32,
    /// Orientation mode (`ORIENTATION_IDENTITY` / `ORIENTATION_VELOCITY_ALIGNED`
    /// / `ORIENTATION_FIXED_ROTATION` / `ORIENTATION_ALIGN_TO_AXIS`).
    pub orientation_mode: u32,
    /// Local axis for `ORIENTATION_ALIGN_TO_AXIS` (`LOCAL_AXIS_PLUS_X` /
    /// `LOCAL_AXIS_PLUS_Y` / `LOCAL_AXIS_PLUS_Z`).
    pub local_axis: u32,
    /// Number of valid entries in `thresholds` (`0..=MAX_LOD_THRESHOLDS`).
    pub threshold_count: u32,
}

/// The assembled answer for one query, mirroring every value the reference
/// reports across its twinned functions.
///
/// `rotation` is the resolved orientation matrix (row-major), `linear` is the
/// `R * S` linear part of the instance transform (row-major), `translation` is
/// the world placement, `world_aabb_min` / `world_aabb_max` are the world-space
/// bounds and `lod_tier` is the selected detail tier.
///
/// Derives only [`PartialEq`] (no `Eq`/`Hash`) because it holds `f32` data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuMeshRendererResult {
    /// Resolved orientation matrix, row-major (three rows of three).
    pub rotation: [[f32; 3]; 3],
    /// Linear `R * S` instance part, row-major (three rows of three).
    pub linear: [[f32; 3]; 3],
    /// World-space translation.
    pub translation: [f32; 3],
    /// World-space `AABB` minimum corner.
    pub world_aabb_min: [f32; 3],
    /// World-space `AABB` maximum corner.
    pub world_aabb_max: [f32; 3],
    /// Selected `LOD` tier.
    pub lod_tier: u32,
}

/// Builds a hand-rolled [`Vec3`] from a packed component triple.
fn to_vec3(v: [f32; 3]) -> Vec3 {
    Vec3::new(v[0], v[1], v[2])
}

/// Flattens a hand-rolled [`Vec3`] back to a packed component triple.
fn from_vec3(v: Vec3) -> [f32; 3] {
    [v.x, v.y, v.z]
}

/// Flattens a hand-rolled [`Mat3`] into a row-major triple of triples.
fn mat_rows(m: Mat3) -> [[f32; 3]; 3] {
    [from_vec3(m.row0), from_vec3(m.row1), from_vec3(m.row2)]
}

/// Maps a wire local-axis code to the golden
/// [`LocalAxis`](prism_render_architecture::particle::mesh_renderer::LocalAxis);
/// any code other than `LOCAL_AXIS_PLUS_Y` / `LOCAL_AXIS_PLUS_Z` folds to `+X`,
/// matching the kernel's `local_axis_unit` default.
fn local_axis_from(code: u32) -> LocalAxis {
    match code {
        LOCAL_AXIS_PLUS_Y => LocalAxis::PlusY,
        LOCAL_AXIS_PLUS_Z => LocalAxis::PlusZ,
        _ => LocalAxis::PlusX,
    }
}

/// Maps a query's orientation selector to the golden
/// [`MeshOrientation`](prism_render_architecture::particle::mesh_renderer::MeshOrientation);
/// any mode other than the three explicit ones folds to `Identity`, matching
/// the kernel's `else` arm.
fn orientation_from(query: &GpuMeshRendererQuery) -> MeshOrientation {
    match query.orientation_mode {
        ORIENTATION_VELOCITY_ALIGNED => MeshOrientation::VelocityAligned {
            velocity: to_vec3(query.velocity),
            up_reference: to_vec3(query.up_reference),
        },
        ORIENTATION_FIXED_ROTATION => MeshOrientation::FixedRotation {
            axis: to_vec3(query.axis),
            sin: query.rot_sin,
            cos: query.rot_cos,
        },
        ORIENTATION_ALIGN_TO_AXIS => MeshOrientation::AlignToAxis {
            local_axis: local_axis_from(query.local_axis),
            target: to_vec3(query.target),
        },
        _ => MeshOrientation::Identity,
    }
}

/// The `CPU` golden verdict for one query, composing the reference entry points
/// so callers (and the parity test) can pin the twin field for field.
///
/// Evaluates
/// [`instance_affine`](prism_render_architecture::particle::mesh_renderer::instance_affine),
/// [`orientation_matrix`](prism_render_architecture::particle::mesh_renderer::orientation_matrix),
/// [`transform_aabb`](prism_render_architecture::particle::mesh_renderer::transform_aabb)
/// and
/// [`select_mesh_lod`](prism_render_architecture::particle::mesh_renderer::select_mesh_lod)
/// on the query's inputs.
#[must_use]
pub fn cpu_reference(query: &GpuMeshRendererQuery) -> GpuMeshRendererResult {
    let orientation = orientation_from(query);
    let rotation = orientation_matrix(orientation);
    let xform: Affine3 = instance_affine(
        to_vec3(query.position),
        orientation,
        to_vec3(query.per_axis_scale),
        query.uniform_scale,
        query.size_over_life,
    );
    let local = Aabb {
        min: to_vec3(query.local_aabb_min),
        max: to_vec3(query.local_aabb_max),
    };
    let world = transform_aabb(xform, local);
    let count = (query.threshold_count as usize).min(MAX_LOD_THRESHOLDS);
    let tier = select_mesh_lod(query.coverage, &query.thresholds[..count]);

    GpuMeshRendererResult {
        rotation: mat_rows(rotation),
        linear: mat_rows(xform.linear),
        translation: from_vec3(xform.translation),
        world_aabb_min: from_vec3(world.min),
        world_aabb_max: from_vec3(world.max),
        lod_tier: tier as u32,
    }
}

/// `repr(C)` `std430` layout of one packed query: eight `vec3` slots (each on
/// its own `16`-byte-aligned slot, with the fixed-rotation `(sin, cos)`, the
/// scale scalars and the coverage tucked into the trailing pad lanes), a
/// four-`f32` threshold array and a four-`u32` slot holding the orientation
/// mode, local axis and threshold count — `160` bytes, exactly as the `WGSL`
/// `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// World position.
    position: [f32; 3],
    /// Fixed-rotation sine.
    rot_sin: f32,
    /// Per-particle velocity.
    velocity: [f32; 3],
    /// Fixed-rotation cosine.
    rot_cos: f32,
    /// Up reference.
    up_reference: [f32; 3],
    /// Uniform scale multiplier.
    uniform_scale: f32,
    /// Rotation axis.
    axis: [f32; 3],
    /// Over-life size sample.
    size_over_life: f32,
    /// Align-to-axis world target.
    align_target: [f32; 3],
    /// Screen-coverage fraction.
    coverage: f32,
    /// Per-axis base scale.
    per_axis_scale: [f32; 3],
    /// Padding lane after the per-axis scale.
    pad0: f32,
    /// Local-space `AABB` minimum corner.
    local_min: [f32; 3],
    /// Padding lane after the local minimum.
    pad1: f32,
    /// Local-space `AABB` maximum corner.
    local_max: [f32; 3],
    /// Padding lane after the local maximum.
    pad2: f32,
    /// Descending `LOD` thresholds.
    thresholds: [f32; 4],
    /// Orientation-mode code.
    orientation_mode: u32,
    /// Local-axis code.
    local_axis: u32,
    /// Valid-threshold count.
    threshold_count: u32,
    /// Padding word.
    pad3: u32,
}

impl GpuQuery {
    /// Packs one [`GpuMeshRendererQuery`] into its `std430` image.
    fn new(query: &GpuMeshRendererQuery) -> GpuQuery {
        GpuQuery {
            position: query.position,
            rot_sin: query.rot_sin,
            velocity: query.velocity,
            rot_cos: query.rot_cos,
            up_reference: query.up_reference,
            uniform_scale: query.uniform_scale,
            axis: query.axis,
            size_over_life: query.size_over_life,
            align_target: query.target,
            coverage: query.coverage,
            per_axis_scale: query.per_axis_scale,
            pad0: 0.0,
            local_min: query.local_aabb_min,
            pad1: 0.0,
            local_max: query.local_aabb_max,
            pad2: 0.0,
            thresholds: query.thresholds,
            orientation_mode: query.orientation_mode,
            local_axis: query.local_axis,
            threshold_count: query.threshold_count,
            pad3: 0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: nine `vec3` slots (each with a
/// trailing pad lane, the last holding the `LOD` tier) for the three rotation
/// rows, the three linear rows, the translation and the two world-bounds
/// corners — `144` bytes matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Rotation row 0.
    rot0: [f32; 3],
    /// Padding lane after rotation row 0.
    pad0: f32,
    /// Rotation row 1.
    rot1: [f32; 3],
    /// Padding lane after rotation row 1.
    pad1: f32,
    /// Rotation row 2.
    rot2: [f32; 3],
    /// Padding lane after rotation row 2.
    pad2: f32,
    /// Linear row 0.
    lin0: [f32; 3],
    /// Padding lane after linear row 0.
    pad3: f32,
    /// Linear row 1.
    lin1: [f32; 3],
    /// Padding lane after linear row 1.
    pad4: f32,
    /// Linear row 2.
    lin2: [f32; 3],
    /// Padding lane after linear row 2.
    pad5: f32,
    /// World-space translation.
    translation: [f32; 3],
    /// Padding lane after the translation.
    pad6: f32,
    /// World-space minimum corner.
    world_min: [f32; 3],
    /// Padding lane after the world minimum.
    pad7: f32,
    /// World-space maximum corner.
    world_max: [f32; 3],
    /// Selected `LOD` tier.
    lod_tier: u32,
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

/// A compiled, reusable mesh-instance-assembly compute pipeline.
pub struct GpuMeshRenderer {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMeshRenderer {
    /// Compiles the mesh-instance-assembly kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMeshRenderer {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mesh_renderer"),
            source: ShaderSource::Wgsl(MESH_RENDERER_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mesh_renderer_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mesh_renderer_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mesh_renderer_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMeshRenderer {
            module,
            layout,
            pipeline,
        }
    }

    /// Assembles every query on-device and returns one [`GpuMeshRendererResult`]
    /// per input, in order.
    ///
    /// Each result equals the reference answers (`instance_affine`,
    /// `orientation_matrix`, `transform_aabb` and `select_mesh_lod`) to within
    /// the tolerance documented on this module, with the `lod_tier`
    /// classification matching exactly. An empty input returns an empty vector
    /// with no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[GpuMeshRendererQuery],
    ) -> Vec<GpuMeshRendererResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_renderer_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mesh_renderer_output"),
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
            label: Some("prism_volumetric_mesh_renderer_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mesh_renderer_bind_group"),
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
            label: Some("prism_volumetric_mesh_renderer_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mesh_renderer_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mesh_renderer_pass"),
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
        debug_assert_eq!(raw.len(), count);

        raw.iter().map(decode_result).collect()
    }
}

/// Decodes one packed [`GpuResult`] into the public [`GpuMeshRendererResult`].
fn decode_result(raw: &GpuResult) -> GpuMeshRendererResult {
    GpuMeshRendererResult {
        rotation: [raw.rot0, raw.rot1, raw.rot2],
        linear: [raw.lin0, raw.lin1, raw.lin2],
        translation: raw.translation,
        world_aabb_min: raw.world_min,
        world_aabb_max: raw.world_max,
        lod_tier: raw.lod_tier,
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
