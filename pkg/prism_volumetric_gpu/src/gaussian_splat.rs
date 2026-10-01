//! `wgpu` compute twin of the `3DGS` / `EWA` Gaussian-splat projection
//! ([`gaussian_splat`](prism_render_architecture::particle::gaussian_splat),
//! design sections 16, 22).
//!
//! A `3DGS` splat is an anisotropic 3D Gaussian: a mean, a per-axis scale, an
//! orientation quaternion and an opacity. Before compositing, each splat is
//! projected to a 2D screen-space Gaussian: the 3D covariance
//! `Sigma = R S Sᵀ Rᵀ` is pushed through the perspective camera Jacobian to a
//! 2D covariance `Sigma'`, that covariance is inverted into a *conic* (its
//! `Option` is `None` for a degenerate, un-invertible footprint), and the
//! conic's 3σ bounding `AABB` and the inclusive `tile` range it overlaps are
//! derived for the binning stage.
//!
//! The `CPU` golden
//! [`particle::gaussian_splat`](prism_render_architecture::particle::gaussian_splat)
//! owns that math; [`GpuGaussianSplat`] is the on-device twin that runs one
//! thread per splat and reproduces the whole chain — the quaternion-to-rotation
//! expansion
//! ([`quat_to_rotation`](prism_render_architecture::particle::gaussian_splat::quat_to_rotation)),
//! the six-entry 3D covariance
//! ([`Gaussian3d::cov3_from_scale_quat`](prism_render_architecture::particle::gaussian_splat::Gaussian3d::cov3_from_scale_quat)),
//! the `EWA` projection
//! ([`project_to_2d`](prism_render_architecture::particle::gaussian_splat::project_to_2d)),
//! the conic inversion
//! ([`conic_from_cov2`](prism_render_architecture::particle::gaussian_splat::conic_from_cov2))
//! and, when the conic exists, its bounding `AABB`
//! ([`Conic2d::bounding_aabb`](prism_render_architecture::particle::gaussian_splat::Conic2d::bounding_aabb))
//! and `tile` range
//! ([`Conic2d::bounding_tiles`](prism_render_architecture::particle::gaussian_splat::Conic2d::bounding_tiles)).
//! A passing real-device parity test is therefore direct evidence the ported
//! kernel computes the same covariances, conic and binning bounds the reference
//! does, not merely that its shader compiles.
//!
//! # What is twinned
//!
//! The full per-splat projection chain is reproduced lane for lane: the
//! quaternion is normalized (identity fallback below [`MIN_LEN_SQ`]) and
//! expanded to a rotation with the same multiply-add polynomial; the 3D
//! covariance is formed as `Sigma_ij = Σ_k R_ik R_jk s_k²` in the reference's
//! summation order; the view depth is clamped away from zero by [`MIN_DEPTH`]
//! before the Jacobian divide; the 2D covariance is inverted into a conic iff
//! its determinant is at least [`DET_EPS`] (otherwise the footprint is reported
//! as absent, mirroring the reference's `None`); and the 3σ `AABB`
//! half-extents, floored and clamped to non-negative `tile` indices for a
//! `tile_size`-pixel square (a zero `tile_size` is treated as one), reproduce
//! the reference binning bounds. The caller supplies each splat's projected
//! pixel center, which is carried straight through onto the resulting conic and
//! into the `AABB`, exactly as the reference caller sets
//! [`Conic2d::center`](prism_render_architecture::particle::gaussian_splat::Conic2d).
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `floor`, `abs`, `+ - * /`, unsigned integer arithmetic and `sqrt` —
//! with no optional device feature, so it runs unmodified on `Metal`, `Vulkan`
//! and `DX12`. The only transcendental is `sqrt`, used exactly where the
//! reference uses it: once to normalize the quaternion (`1 / sqrt(len²)`) and
//! twice to turn the recovered 2D variances into 3σ `AABB` half-extents
//! (`SIGMA_RADIUS * sqrt(Sigma_xx)` and `SIGMA_RADIUS * sqrt(Sigma_yy)`). No
//! `sin`, `cos`, `exp`, `log` or `pow` appears, matching the reference, whose
//! `3DGS` projection likewise needs only `sqrt` and division. Every divide is
//! guarded: the quaternion divide is reached only when `len²` is at least
//! [`MIN_LEN_SQ`], the Jacobian divide only after the depth clamp, and the
//! conic / `to_cov2` divides only when the determinant magnitude is at least
//! [`DET_EPS`].
//!
//! # Correctness model
//!
//! Each lane is a fixed, non-reorderable sequence of multiplies, adds, guarded
//! divides and `sqrt`s, so `CPU` and `GPU` evaluate the same closed form in the
//! same order. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place. The parity test asserts an absolute-or-relative
//! tolerance tight enough to catch a genuinely wrong port (a transposed
//! rotation, a dropped covariance term, a missing depth clamp, a flipped conic
//! sign, a wrong `AABB` radius) yet loose enough to admit that legal fused
//! multiply-add contraction, and it compares the degenerate `None` branch and
//! the integer `tile` indices exactly.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `3DGS` / `EWA` splatting projection
//! (`Zwicker` et al. 2001; `Kerbl` et al. 2023) plus `wgpu` compute dispatch;
//! no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::gaussian_splat::{
    Conic2d, Gaussian3d, SplatAabb, TileRange, GAUSSIAN_STRIDE,
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

/// Threads per workgroup; one thread projects one Gaussian splat.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` Gaussian-splat projection kernel, embedded inline so
/// the twin ships as a single source file. Mirrors the `CPU` golden
/// chain step for step; see the module documentation for the algorithm.
///
/// The `Splat` struct lays its three `vec4` slots out exactly as
/// [`Gaussian3d::to_std430`](prism_render_architecture::particle::gaussian_splat::Gaussian3d::to_std430)
/// does ([`GAUSSIAN_STRIDE`] bytes): `vec4(mean, opacity)`, `vec4(scale, pad)`
/// and `vec4(quat)`.
const GAUSSIAN_SPLAT_WGSL: &str = r#"
// 3DGS / EWA Gaussian-splat projection twin: one thread per splat expands the
// quaternion to a rotation, forms the 3D covariance Sigma = R S Sᵀ Rᵀ, projects
// it through the perspective camera Jacobian to the 2D covariance Sigma',
// inverts Sigma' into a conic (reporting absence when the determinant is below
// DET_EPS), and derives the 3σ screen AABB and the inclusive tile range. It
// mirrors the CPU golden `particle::gaussian_splat`, uses only the portable
// core-WGSL subset (min/max/clamp/floor/abs, + - * /, unsigned integer math and
// sqrt), and takes no optional feature, so it runs unmodified on Metal, Vulkan
// and DX12.
//
// Provenance: standard 3DGS / EWA splatting projection; no third-party engine
// source or derived code.

struct Params {
    // Square tile edge in pixels for the tile-range binning. A zero is treated
    // as one, matching the reference `bounding_tiles`.
    tile_size: u32,
    // Number of valid splats in `splats` / `views`.
    count: u32,
    // Padding to a 16-byte std140 uniform struct.
    pad0: u32,
    // Padding to a 16-byte std140 uniform struct.
    pad1: u32,
}

// One input splat. 48-byte std430 stride (three vec4 slots) matching the
// reference `Gaussian3d::to_std430`: vec4(mean, opacity), vec4(scale, pad),
// vec4(quat).
struct Splat {
    mean_opacity: vec4<f32>,
    scale_pad: vec4<f32>,
    quat: vec4<f32>,
}

// One splat's view record. 16-byte std430 stride: the x/y focal lengths in
// pixels and the projected pixel-space center the conic is placed at.
struct View {
    focal_center: vec4<f32>,
}

// One splat's projection result. 96-byte std430 stride (six vec4 slots): the
// six unique 3D covariance entries, the three 2D covariance entries, the three
// conic coefficients, the AABB min/max corners, the four tile indices and a
// validity flag (1 when the conic exists, 0 for a degenerate footprint).
struct SplatOut {
    cov3_xx: f32,
    cov3_xy: f32,
    cov3_xz: f32,
    cov3_yy: f32,
    cov3_yz: f32,
    cov3_zz: f32,
    cov2_a: f32,
    cov2_b: f32,
    cov2_c: f32,
    conic_a: f32,
    conic_b: f32,
    conic_c: f32,
    aabb_min_x: f32,
    aabb_min_y: f32,
    aabb_max_x: f32,
    aabb_max_y: f32,
    tile_min_x: u32,
    tile_min_y: u32,
    tile_max_x: u32,
    tile_max_y: u32,
    valid: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> splats: array<Splat>;
@group(0) @binding(2) var<storage, read> views: array<View>;
@group(0) @binding(3) var<storage, read_write> results: array<SplatOut>;

// Minimum squared quaternion length below which the rotation falls back to the
// identity instead of dividing by zero. Matches the reference `MIN_LEN_SQ`.
const MIN_LEN_SQ: f32 = 1.0e-12;

// Minimum absolute view depth used in the Jacobian so a splat on or behind the
// camera plane cannot divide by zero. Matches the reference `MIN_DEPTH`.
const MIN_DEPTH: f32 = 1.0e-6;

// Determinant magnitude below which a 2x2 matrix is treated as singular.
// Matches the reference `DET_EPS`.
const DET_EPS: f32 = 1.0e-12;

// Gaussian footprint radius, in standard deviations, for the bounding AABB.
// Matches the reference `SIGMA_RADIUS`.
const SIGMA_RADIUS: f32 = 3.0;

// Three rows of the 3x3 rotation matrix, in the reference's row-major order.
struct Rot {
    row0: vec3<f32>,
    row1: vec3<f32>,
    row2: vec3<f32>,
}

// Expands a quaternion (in [x, y, z, w] order) to a rotation matrix with the
// standard multiply-add polynomial, normalizing first and falling back to the
// identity for a near-zero quaternion. Mirrors the reference `quat_to_rotation`.
fn quat_to_rotation(quat: vec4<f32>) -> Rot {
    let len_sq = quat.x * quat.x + quat.y * quat.y + quat.z * quat.z + quat.w * quat.w;
    var x = 0.0;
    var y = 0.0;
    var z = 0.0;
    var w = 1.0;
    if (len_sq >= MIN_LEN_SQ) {
        let inv_len = 1.0 / sqrt(len_sq);
        x = quat.x * inv_len;
        y = quat.y * inv_len;
        z = quat.z * inv_len;
        w = quat.w * inv_len;
    }
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;
    var r: Rot;
    r.row0 = vec3<f32>(1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy));
    r.row1 = vec3<f32>(2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx));
    r.row2 = vec3<f32>(2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy));
    return r;
}

// One symmetric 3D covariance entry: Sigma_ij = Σ_k R_ik R_jk s²_k, evaluated in
// the reference summation order. `s2` holds the squared per-axis scales.
fn sigma(ri: vec3<f32>, rj: vec3<f32>, s2: vec3<f32>) -> f32 {
    return ri.x * rj.x * s2.x + ri.y * rj.y * s2.y + ri.z * rj.z * s2.z;
}

// Projects the six-entry 3D covariance to the three unique entries [a, b, c] of
// the symmetric 2D covariance via the EWA perspective Jacobian, clamping depth
// away from zero. Mirrors the reference `project_to_2d`.
fn project_to_2d(c00: f32, c01: f32, c02: f32, c11: f32, c12: f32, c22: f32,
                 mean: vec3<f32>, focal: vec2<f32>) -> vec3<f32> {
    let fx = focal.x;
    let fy = focal.y;
    let depth = mean.z;
    var z = depth;
    if (abs(depth) < MIN_DEPTH) {
        if (depth < 0.0) {
            z = -MIN_DEPTH;
        } else {
            z = MIN_DEPTH;
        }
    }
    let inv_z = 1.0 / z;
    let inv_z2 = inv_z * inv_z;

    let j00 = fx * inv_z;
    let j02 = -fx * mean.x * inv_z2;
    let j11 = fy * inv_z;
    let j12 = -fy * mean.y * inv_z2;

    // A = J · Sigma (2x3).
    let a00 = j00 * c00 + j02 * c02;
    let a01 = j00 * c01 + j02 * c12;
    let a02 = j00 * c02 + j02 * c22;
    let a11 = j11 * c11 + j12 * c12;
    let a12 = j11 * c12 + j12 * c22;

    // Sigma' = A · Jᵀ (2x2), symmetric.
    let out_a = a00 * j00 + a02 * j02;
    let out_b = a01 * j11 + a02 * j12;
    let out_c = a11 * j11 + a12 * j12;
    return vec3<f32>(out_a, out_b, out_c);
}

// Recovers the 2D covariance [a, b, c] a conic was inverted from, or zeros for a
// degenerate conic. Mirrors the reference `Conic2d::to_cov2`.
fn conic_to_cov2(a: f32, b: f32, c: f32) -> vec3<f32> {
    let det = a * c - b * b;
    if (abs(det) < DET_EPS) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let inv_det = 1.0 / det;
    return vec3<f32>(c * inv_det, -b * inv_det, a * inv_det);
}

// Floors a scalar and clamps it to a non-negative u32 tile index. Mirrors the
// reference `floor_to_u32`.
fn floor_to_u32(x: f32) -> u32 {
    let f = max(floor(x), 0.0);
    return u32(f);
}

@compute @workgroup_size(64)
fn project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let splat = splats[idx];
    let view = views[idx];

    let scale = splat.scale_pad.xyz;
    let s2 = vec3<f32>(scale.x * scale.x, scale.y * scale.y, scale.z * scale.z);
    let rot = quat_to_rotation(splat.quat);

    // Six unique 3D covariance entries [xx, xy, xz, yy, yz, zz].
    let cov3_xx = sigma(rot.row0, rot.row0, s2);
    let cov3_xy = sigma(rot.row0, rot.row1, s2);
    let cov3_xz = sigma(rot.row0, rot.row2, s2);
    let cov3_yy = sigma(rot.row1, rot.row1, s2);
    let cov3_yz = sigma(rot.row1, rot.row2, s2);
    let cov3_zz = sigma(rot.row2, rot.row2, s2);

    let mean = splat.mean_opacity.xyz;
    let focal = vec2<f32>(view.focal_center.x, view.focal_center.y);
    let center = vec2<f32>(view.focal_center.z, view.focal_center.w);

    let cov2 = project_to_2d(cov3_xx, cov3_xy, cov3_xz, cov3_yy, cov3_yz, cov3_zz, mean, focal);

    var out: SplatOut;
    out.cov3_xx = cov3_xx;
    out.cov3_xy = cov3_xy;
    out.cov3_xz = cov3_xz;
    out.cov3_yy = cov3_yy;
    out.cov3_yz = cov3_yz;
    out.cov3_zz = cov3_zz;
    out.cov2_a = cov2.x;
    out.cov2_b = cov2.y;
    out.cov2_c = cov2.z;
    out.conic_a = 0.0;
    out.conic_b = 0.0;
    out.conic_c = 0.0;
    out.aabb_min_x = 0.0;
    out.aabb_min_y = 0.0;
    out.aabb_max_x = 0.0;
    out.aabb_max_y = 0.0;
    out.tile_min_x = 0u;
    out.tile_min_y = 0u;
    out.tile_max_x = 0u;
    out.tile_max_y = 0u;
    out.valid = 0u;
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;

    // Invert the 2D covariance into the conic; a determinant below DET_EPS is a
    // degenerate footprint reported with valid = 0, matching the reference
    // `conic_from_cov2` returning `None`.
    let det = cov2.x * cov2.z - cov2.y * cov2.y;
    if (det >= DET_EPS) {
        let inv_det = 1.0 / det;
        let conic_a = cov2.z * inv_det;
        let conic_b = -cov2.y * inv_det;
        let conic_c = cov2.x * inv_det;

        // 3σ AABB from the covariance recovered off the conic, centered on the
        // caller-provided projected pixel position.
        let rec = conic_to_cov2(conic_a, conic_b, conic_c);
        let rx = SIGMA_RADIUS * sqrt(max(rec.x, 0.0));
        let ry = SIGMA_RADIUS * sqrt(max(rec.z, 0.0));
        let min_x = center.x - rx;
        let min_y = center.y - ry;
        let max_x = center.x + rx;
        let max_y = center.y + ry;

        let ts_f = f32(max(params.tile_size, 1u));

        out.conic_a = conic_a;
        out.conic_b = conic_b;
        out.conic_c = conic_c;
        out.aabb_min_x = min_x;
        out.aabb_min_y = min_y;
        out.aabb_max_x = max_x;
        out.aabb_max_y = max_y;
        out.tile_min_x = floor_to_u32(min_x / ts_f);
        out.tile_min_y = floor_to_u32(min_y / ts_f);
        out.tile_max_x = floor_to_u32(max_x / ts_f);
        out.tile_max_y = floor_to_u32(max_y / ts_f);
        out.valid = 1u;
    }

    results[idx] = out;
}
"#;

/// One splat's projection query: the Gaussian primitive, the camera focal
/// lengths and the projected pixel-space center the resulting conic is placed
/// at.
///
/// The `splat` reuses the `CPU` golden
/// [`Gaussian3d`](prism_render_architecture::particle::gaussian_splat::Gaussian3d)
/// type so the host and device share one splat contract. `screen_center` plays
/// the role the reference caller fills by assigning
/// [`Conic2d::center`](prism_render_architecture::particle::gaussian_splat::Conic2d)
/// after inversion: it is carried straight onto the conic and into the `AABB`.
/// Derives only [`PartialEq`] (no `Eq`/`Hash`) because it holds `f32` fields.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuGaussianSplatQuery {
    /// The anisotropic 3D Gaussian primitive to project (its `mean` is the
    /// view-space center consumed by the `EWA` Jacobian).
    pub splat: Gaussian3d,
    /// The `x`/`y` focal lengths in pixels used to build the camera Jacobian.
    pub focal: [f32; 2],
    /// The projected pixel-space center placed on the conic and used as the
    /// `AABB` center.
    pub screen_center: [f32; 2],
}

/// The screen-space footprint of a projected splat: the inverted-covariance
/// conic, its 3σ bounding `AABB` and the inclusive `tile` range it overlaps.
///
/// Present only when the 2D covariance was invertible; a degenerate splat
/// yields [`None`] on [`GaussianSplatProjection::footprint`], mirroring the
/// reference
/// [`conic_from_cov2`](prism_render_architecture::particle::gaussian_splat::conic_from_cov2)
/// returning `None`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplatFootprint {
    /// The inverse-covariance conic, centered on the query's `screen_center`,
    /// matching the reference
    /// [`Conic2d`](prism_render_architecture::particle::gaussian_splat::Conic2d).
    pub conic: Conic2d,
    /// The 3σ axis-aligned bounding box in pixel space, matching
    /// [`Conic2d::bounding_aabb`](prism_render_architecture::particle::gaussian_splat::Conic2d::bounding_aabb).
    pub aabb: SplatAabb,
    /// The inclusive `tile` range the `AABB` overlaps, matching
    /// [`Conic2d::bounding_tiles`](prism_render_architecture::particle::gaussian_splat::Conic2d::bounding_tiles).
    pub tiles: TileRange,
}

/// One splat's full projection result: the six-entry 3D covariance, the
/// three-entry 2D covariance and the optional screen-space footprint.
///
/// The covariances are always produced; the footprint is [`Some`] exactly when
/// the 2D covariance inverted into a conic and [`None`] for a degenerate splat.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianSplatProjection {
    /// The six unique symmetric 3D covariance entries `[xx, xy, xz, yy, yz, zz]`,
    /// matching
    /// [`Gaussian3d::cov3_from_scale_quat`](prism_render_architecture::particle::gaussian_splat::Gaussian3d::cov3_from_scale_quat).
    pub cov3: [f32; 6],
    /// The three unique symmetric 2D covariance entries `[a, b, c]`, matching
    /// [`project_to_2d`](prism_render_architecture::particle::gaussian_splat::project_to_2d).
    pub cov2: [f32; 3],
    /// The screen-space footprint, or [`None`] for a degenerate splat.
    pub footprint: Option<SplatFootprint>,
}

/// Uniform parameters for one projection dispatch. `repr(C)` `std430` layout
/// matching `Params` in [`GAUSSIAN_SPLAT_WGSL`]: the tile edge, the splat count
/// and two pad words — `16` bytes with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Square tile edge in pixels (`0` is treated as `1`).
    tile_size: u32,
    /// Number of splats in this dispatch.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One input splat as uploaded. `48`-byte `std430` stride ([`GAUSSIAN_STRIDE`])
/// matching `Splat` in the shader and the reference
/// [`Gaussian3d::to_std430`](prism_render_architecture::particle::gaussian_splat::Gaussian3d::to_std430):
/// `vec4(mean, opacity)`, `vec4(scale, pad)`, `vec4(quat)`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSplat {
    /// Mean `x`.
    mean_x: f32,
    /// Mean `y`.
    mean_y: f32,
    /// Mean `z`.
    mean_z: f32,
    /// Scalar opacity.
    opacity: f32,
    /// Scale `x`.
    scale_x: f32,
    /// Scale `y`.
    scale_y: f32,
    /// Scale `z`.
    scale_z: f32,
    /// Padding word (the `scale` slot's fourth component).
    scale_pad: f32,
    /// Quaternion `x`.
    quat_x: f32,
    /// Quaternion `y`.
    quat_y: f32,
    /// Quaternion `z`.
    quat_z: f32,
    /// Quaternion `w`.
    quat_w: f32,
}

/// One splat's view record as uploaded. `16`-byte `std430` stride matching
/// `View` in the shader: the `x`/`y` focal lengths and the projected center.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuView {
    /// Focal length `x` in pixels.
    focal_x: f32,
    /// Focal length `y` in pixels.
    focal_y: f32,
    /// Projected center `x` in pixels.
    center_x: f32,
    /// Projected center `y` in pixels.
    center_y: f32,
}

/// One splat's projection result as read back. `96`-byte `std430` stride
/// matching `SplatOut` in the shader: the 3D covariance, the 2D covariance, the
/// conic, the `AABB`, the `tile` indices and the validity flag.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSplatOut {
    /// 3D covariance `xx`.
    cov3_xx: f32,
    /// 3D covariance `xy`.
    cov3_xy: f32,
    /// 3D covariance `xz`.
    cov3_xz: f32,
    /// 3D covariance `yy`.
    cov3_yy: f32,
    /// 3D covariance `yz`.
    cov3_yz: f32,
    /// 3D covariance `zz`.
    cov3_zz: f32,
    /// 2D covariance `a`.
    cov2_a: f32,
    /// 2D covariance `b`.
    cov2_b: f32,
    /// 2D covariance `c`.
    cov2_c: f32,
    /// Conic `a`.
    conic_a: f32,
    /// Conic `b`.
    conic_b: f32,
    /// Conic `c`.
    conic_c: f32,
    /// `AABB` minimum `x`.
    aabb_min_x: f32,
    /// `AABB` minimum `y`.
    aabb_min_y: f32,
    /// `AABB` maximum `x`.
    aabb_max_x: f32,
    /// `AABB` maximum `y`.
    aabb_max_y: f32,
    /// First overlapped column tile.
    tile_min_x: u32,
    /// First overlapped row tile.
    tile_min_y: u32,
    /// Last overlapped column tile (inclusive).
    tile_max_x: u32,
    /// Last overlapped row tile (inclusive).
    tile_max_y: u32,
    /// `1` when the conic exists, `0` for a degenerate footprint.
    valid: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A compiled, reusable Gaussian-splat projection pipeline.
pub struct GpuGaussianSplat {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuGaussianSplat {
    /// Compiles the Gaussian-splat projection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuGaussianSplat {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_gaussian_splat"),
            source: ShaderSource::Wgsl(GAUSSIAN_SPLAT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_gaussian_splat_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_gaussian_splat_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_gaussian_splat_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("project"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuGaussianSplat {
            module,
            layout,
            pipeline,
        }
    }

    /// Projects every splat in `queries` under the shared square `tile_size`,
    /// returning one [`GaussianSplatProjection`] per splat in input order.
    ///
    /// Each result reproduces the `CPU` golden chain
    /// [`Gaussian3d::cov3_from_scale_quat`](prism_render_architecture::particle::gaussian_splat::Gaussian3d::cov3_from_scale_quat)
    /// → [`project_to_2d`](prism_render_architecture::particle::gaussian_splat::project_to_2d)
    /// → [`conic_from_cov2`](prism_render_architecture::particle::gaussian_splat::conic_from_cov2)
    /// → [`Conic2d::bounding_aabb`](prism_render_architecture::particle::gaussian_splat::Conic2d::bounding_aabb)
    /// / [`Conic2d::bounding_tiles`](prism_render_architecture::particle::gaussian_splat::Conic2d::bounding_tiles)
    /// to within the tolerance documented on this module, with the footprint
    /// absent exactly when the reference conic is `None`. An empty `queries`
    /// slice yields an empty result — storage buffers cannot be zero-sized, so
    /// it is handled by an early return.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        tile_size: u32,
        queries: &[GpuGaussianSplatQuery],
    ) -> Vec<GaussianSplatProjection> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = Params {
            tile_size,
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
        };

        let splats: Vec<GpuSplat> = queries
            .iter()
            .map(|q| GpuSplat {
                mean_x: q.splat.mean[0],
                mean_y: q.splat.mean[1],
                mean_z: q.splat.mean[2],
                opacity: q.splat.opacity,
                scale_x: q.splat.scale[0],
                scale_y: q.splat.scale[1],
                scale_z: q.splat.scale[2],
                scale_pad: 0.0,
                quat_x: q.splat.quat[0],
                quat_y: q.splat.quat[1],
                quat_z: q.splat.quat[2],
                quat_w: q.splat.quat[3],
            })
            .collect();
        let views: Vec<GpuView> = queries
            .iter()
            .map(|q| GpuView {
                focal_x: q.focal[0],
                focal_y: q.focal[1],
                center_x: q.screen_center[0],
                center_y: q.screen_center[1],
            })
            .collect();

        // The uploaded splat stride must equal the reference std430 record size.
        debug_assert_eq!(size_of::<GpuSplat>(), GAUSSIAN_STRIDE);

        let out_bytes = (queries.len() as u64) * (size_of::<GpuSplatOut>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gaussian_splat_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let splats_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gaussian_splat_splats"),
            contents: bytemuck::cast_slice(&splats),
            usage: BufferUsages::STORAGE,
        });
        let views_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gaussian_splat_views"),
            contents: bytemuck::cast_slice(&views),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gaussian_splat_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gaussian_splat_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_gaussian_splat_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: splats_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: views_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_gaussian_splat_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_gaussian_splat_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per splat, in workgroups of `WORKGROUP_SIZE`.
            let groups = (queries.len() as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let gpu_results = bytemuck::cast_slice::<u8, GpuSplatOut>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_results.len(), queries.len());

        gpu_results
            .into_iter()
            .zip(queries.iter())
            .map(|(r, q)| {
                let footprint = if r.valid != 0 {
                    Some(SplatFootprint {
                        conic: Conic2d {
                            a: r.conic_a,
                            b: r.conic_b,
                            c: r.conic_c,
                            center: q.screen_center,
                        },
                        aabb: SplatAabb {
                            min: [r.aabb_min_x, r.aabb_min_y],
                            max: [r.aabb_max_x, r.aabb_max_y],
                        },
                        tiles: TileRange {
                            min_x: r.tile_min_x,
                            min_y: r.tile_min_y,
                            max_x: r.tile_max_x,
                            max_y: r.tile_max_y,
                        },
                    })
                } else {
                    None
                };
                GaussianSplatProjection {
                    cov3: [
                        r.cov3_xx, r.cov3_xy, r.cov3_xz, r.cov3_yy, r.cov3_yz, r.cov3_zz,
                    ],
                    cov2: [r.cov2_a, r.cov2_b, r.cov2_c],
                    footprint,
                }
            })
            .collect()
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
