//! `wgpu` compute twin of the particle camera-query `DataInterface` (`DI`)
//! ([`camera`](prism_render_architecture::particle::camera), particle design
//! §8.3, "Scene" category).
//!
//! The `CPU` golden
//! [`CameraProjection`](prism_render_architecture::particle::camera::CameraProjection)
//! owns the projection-relative queries a Set/Force module reads per particle:
//! the `world`->`view`->`clip`->`NDC`->`screen` transform chain
//! ([`world_to_view`](prism_render_architecture::particle::camera::CameraProjection::world_to_view),
//! [`view_space_position`](prism_render_architecture::particle::camera::CameraProjection::view_space_position),
//! [`view_to_clip`](prism_render_architecture::particle::camera::CameraProjection::view_to_clip),
//! [`clip_to_ndc`](prism_render_architecture::particle::camera::CameraProjection::clip_to_ndc),
//! [`world_to_ndc`](prism_render_architecture::particle::camera::CameraProjection::world_to_ndc),
//! [`ndc_to_screen`](prism_render_architecture::particle::camera::CameraProjection::ndc_to_screen)
//! and
//! [`world_to_screen`](prism_render_architecture::particle::camera::CameraProjection::world_to_screen)),
//! the coarse frustum bucket
//! ([`classify_visibility`](prism_render_architecture::particle::camera::CameraProjection::classify_visibility)),
//! the depth and camera-relative scalars
//! ([`linearize_depth`](prism_render_architecture::particle::camera::CameraProjection::linearize_depth),
//! [`distance_to_camera`](prism_render_architecture::particle::camera::CameraProjection::distance_to_camera),
//! [`distance_squared_to_camera`](prism_render_architecture::particle::camera::CameraProjection::distance_squared_to_camera),
//! [`direction_to_camera`](prism_render_architecture::particle::camera::CameraProjection::direction_to_camera))
//! and the screen-coverage inputs
//! ([`screen_coverage_ndc`](prism_render_architecture::particle::camera::CameraProjection::screen_coverage_ndc),
//! [`screen_coverage_pixels`](prism_render_architecture::particle::camera::CameraProjection::screen_coverage_pixels)).
//! [`GpuCamera`] is the on-device twin: the camera parameters ride a shared
//! uniform while one thread evaluates one world-point query, so a passing
//! real-device parity test is direct evidence the ported kernel folds the same
//! projective algebra and the same guard decisions the reference does, not
//! merely that its shader compiles.
//!
//! # What is twinned
//!
//! Each thread reads one world point (plus an independent `NDC` sample for
//! [`ndc_to_screen`](prism_render_architecture::particle::camera::CameraProjection::ndc_to_screen),
//! an independent `NDC` depth for
//! [`linearize_depth`](prism_render_architecture::particle::camera::CameraProjection::linearize_depth)
//! and a bounding-sphere radius for the coverage queries) and writes every
//! per-particle answer: the view-space position (which backs both
//! [`world_to_view`](prism_render_architecture::particle::camera::CameraProjection::world_to_view)
//! and its
//! [`view_space_position`](prism_render_architecture::particle::camera::CameraProjection::view_space_position)
//! alias), the homogeneous clip coordinate, the two `NDC` mappings (each with a
//! `u32` flag mirroring the reference [`Option`]), the two screen mappings, the
//! discrete visibility code, the linearized depth, the straight-line and
//! squared distances, the unit direction to the camera and the `NDC`/pixel
//! coverage. The host-side basis constructor
//! [`from_forward_up`](prism_render_architecture::particle::camera::CameraBasis::from_forward_up)
//! is deliberately not twinned: it is a one-off `CPU` setup whose orthonormal
//! basis is handed to the kernel through the uniform.
//!
//! # `Option` encoding
//!
//! The three fallible queries
//! ([`clip_to_ndc`](prism_render_architecture::particle::camera::CameraProjection::clip_to_ndc),
//! [`world_to_ndc`](prism_render_architecture::particle::camera::CameraProjection::world_to_ndc)
//! and
//! [`world_to_screen`](prism_render_architecture::particle::camera::CameraProjection::world_to_screen))
//! cannot return a native [`Option`] from a storage buffer, so the kernel
//! writes a `1`/`0` flag word and zeroes the coordinate payload on the [`None`]
//! branch. The host compares the flag exactly and reads the coordinate only
//! when the flag is set, exactly as the reference's [`Option`] forces.
//!
//! # Correctness model
//!
//! The flags and the visibility code are discrete classifications built from
//! `f32` magnitude comparisons against the reference
//! [`EPS`](prism_render_architecture::particle::camera::EPS), so for inputs
//! clear of the `w`-at-the-camera-plane, near/far and `NDC`-boundary thresholds
//! the `CPU` and `GPU` agree exactly and the parity test asserts `==` on them.
//! The continuous outputs thread through multiplies, adds, one guarded
//! reciprocal each and a single `sqrt` (for the distance and the normalized
//! direction), so `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on every
//! continuous quantity.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `dot`, `sqrt`, `+ - * /` and unsigned compares — with no `sin`, `cos`,
//! `tan`, inverse trigonometry, `exp`, `log`, `pow` or optional device feature,
//! so it runs unmodified on Metal, Vulkan and DX12. There is no loop: each
//! thread performs a fixed, bounded sequence of arithmetic, so the kernel
//! provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::camera`；
//! standard world/view/clip/`NDC`/screen projective algebra plus `wgpu` compute
//! dispatch; no third-party engine source or derived code.
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::camera::{
    CameraProjection, ClipVisibility, ProjectionKind,
};
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

/// Number of threads per workgroup. `64` is the portable, warp-friendly size
/// shared by every kernel in this crate.
const WORKGROUP_SIZE: u32 = 64;

/// Projection-kind code for a perspective camera, written into the camera
/// uniform's `kind` word. A direct `f32` tag would force a forbidden float
/// equality, so the discriminant travels as a `u32`.
const KIND_PERSPECTIVE: u32 = 0;

/// Projection-kind code for an orthographic camera.
const KIND_ORTHOGRAPHIC: u32 = 1;

/// Visibility code: inside the frustum (and, for perspective, in front of the
/// camera). Mirrors
/// [`ClipVisibility::Visible`](prism_render_architecture::particle::camera::ClipVisibility::Visible).
const VIS_VISIBLE: u32 = 0;

/// Visibility code: behind the camera plane (perspective only). Mirrors
/// [`ClipVisibility::BehindCamera`](prism_render_architecture::particle::camera::ClipVisibility::BehindCamera).
const VIS_BEHIND_CAMERA: u32 = 1;

/// Visibility code: in front of the camera but outside the clip cube. Mirrors
/// [`ClipVisibility::OutsideFrustum`](prism_render_architecture::particle::camera::ClipVisibility::OutsideFrustum).
const VIS_OUTSIDE_FRUSTUM: u32 = 2;

/// Visibility code: the perspective divide collapsed. Mirrors
/// [`ClipVisibility::Degenerate`](prism_render_architecture::particle::camera::ClipVisibility::Degenerate).
const VIS_DEGENERATE: u32 = 3;

/// Flag word written when a fallible query returned `Some`, matching the host
/// decode in [`decode_result`].
const FLAG_SOME: u32 = 1;

/// The portable core-`WGSL` camera-query kernel, embedded inline so the twin
/// ships as a single source file. Mirrors the `CPU` golden
/// [`CameraProjection`](prism_render_architecture::particle::camera::CameraProjection)
/// query by query; see the module documentation for the algorithm.
const CAMERA_WGSL: &str = r#"
// Camera-query twin: the camera parameters ride a shared uniform while one
// thread evaluates one world-point query, reproducing the whole
// world->view->clip->NDC->screen chain plus the camera-relative scalars and the
// coverage inputs. It mirrors the CPU golden `particle::camera` guard for
// guard, uses only the portable core-WGSL subset (abs/min/max/dot/sqrt and
// + - * / plus unsigned compares), and takes no optional feature, so it runs
// unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard projective camera algebra; no third-party engine source
// or derived code.

struct Params {
    // Number of valid queries in `queries`.
    count: u32,
    // Padding to a 16-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// The shared camera. 96-byte std430 layout matching the host `GpuCameraUniform`: the
// world position and the three orthonormal basis axes padded to vec4 lanes,
// then the near/far planes, the two projection parameters (tan(fov/2) for
// perspective, the view-plane half-extents for orthographic), the viewport
// dimensions and the projection-kind code.
struct Camera {
    world_pos: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    forward: vec4<f32>,
    near: f32,
    far: f32,
    p0: f32,
    p1: f32,
    viewport_w: f32,
    viewport_h: f32,
    kind: u32,
    pad: u32,
}

// One query. 48-byte std430 stride matching the host `GpuQuery`: the world
// point, an independent NDC sample for the standalone ndc_to_screen mapping,
// and the auxiliary scalars (x = coverage radius, y = NDC depth for the
// linearize query).
struct Query {
    world_pos: vec4<f32>,
    ndc_point: vec4<f32>,
    aux: vec4<f32>,
}

// One result. 144-byte std430 stride matching the host `GpuResult`.
struct Result {
    view: vec4<f32>,
    clip: vec4<f32>,
    ndc_from_clip: vec4<f32>,
    ndc_from_world: vec4<f32>,
    screen_from_ndc: vec4<f32>,
    screen_from_world: vec4<f32>,
    // linearize_depth, distance, distance_squared, coverage_ndc.
    scalars: vec4<f32>,
    // direction_to_camera.xyz, coverage_pixels.
    dir_cov: vec4<f32>,
    // clip_to_ndc flag, world_to_ndc flag, world_to_screen flag, visibility.
    flags: vec4<u32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<storage, read> queries: array<Query>;
@group(0) @binding(3) var<storage, read_write> results: array<Result>;

// Guarded reciprocal and tolerant comparison band, matching the reference
// `EPS` and `safe_recip`. A direct f32 `==`/`!=` is forbidden, so the camera
// plane, near/far and NDC boundary tests all compare magnitudes against this.
const EPS: f32 = 1.0e-6;

// Squared-length floor for the normalize-or-zero fallback, matching the
// reference `EPS_LEN_SQ` used by `Vec3::normalize_or_zero`.
const EPS_LEN_SQ: f32 = 1.0e-12;

// Visibility codes, matching the host `ClipVisibility` mapping.
const VIS_VISIBLE: u32 = 0u;
const VIS_BEHIND_CAMERA: u32 = 1u;
const VIS_OUTSIDE_FRUSTUM: u32 = 2u;
const VIS_DEGENERATE: u32 = 3u;

// A resolved perspective divide: whether `w` cleared the camera plane and the
// NDC coordinate when it did.
struct NdcOpt {
    ok: bool,
    ndc: vec3<f32>,
}

// Guarded scalar reciprocal: `1 / value` when `|value|` clears EPS, else 0, so
// a degenerate divisor collapses the mapped coordinate to the origin instead of
// yielding NaN or infinity. Mirrors the reference `safe_recip`.
fn safe_recip(value: f32) -> f32 {
    if (abs(value) > EPS) {
        return 1.0 / value;
    }
    return 0.0;
}

fn is_perspective() -> bool {
    return camera.kind == 0u;
}

// world->view: the camera-relative offset projected onto the three basis axes,
// matching the reference `world_to_view` (and its `view_space_position` alias).
fn world_to_view(world_pos: vec3<f32>) -> vec3<f32> {
    let rel = world_pos - camera.world_pos.xyz;
    return vec3<f32>(
        dot(rel, camera.right.xyz),
        dot(rel, camera.up.xyz),
        dot(rel, camera.forward.xyz),
    );
}

// view->clip: the homogeneous clip coordinate pre perspective divide, matching
// the reference `view_to_clip`. Perspective clip w is the view depth;
// orthographic w is 1. Every reciprocal is EPS-guarded.
fn view_to_clip(view: vec3<f32>) -> vec4<f32> {
    let inv_range = safe_recip(camera.far - camera.near);
    if (is_perspective()) {
        let inv_tx = safe_recip(camera.p0);
        let inv_ty = safe_recip(camera.p1);
        let cx = view.x * inv_tx;
        let cy = view.y * inv_ty;
        let cz = camera.far * (view.z - camera.near) * inv_range;
        return vec4<f32>(cx, cy, cz, view.z);
    }
    let inv_hw = safe_recip(camera.p0);
    let inv_hh = safe_recip(camera.p1);
    let cx = view.x * inv_hw;
    let cy = view.y * inv_hh;
    let cz = (view.z - camera.near) * inv_range;
    return vec4<f32>(cx, cy, cz, 1.0);
}

// clip->NDC: the perspective divide, flagged invalid when |w| is at or below
// EPS, matching the reference `clip_to_ndc` returning None.
fn clip_to_ndc(clip: vec4<f32>) -> NdcOpt {
    var out: NdcOpt;
    out.ok = false;
    out.ndc = vec3<f32>(0.0, 0.0, 0.0);
    if (abs(clip.w) <= EPS) {
        return out;
    }
    let inv_w = 1.0 / clip.w;
    out.ok = true;
    out.ndc = clip.xyz * inv_w;
    return out;
}

// world->NDC: invalid when the point is behind the camera (perspective) or the
// perspective divide is degenerate, matching the reference `world_to_ndc`.
fn world_to_ndc(world_pos: vec3<f32>) -> NdcOpt {
    let view = world_to_view(world_pos);
    if (is_perspective() && view.z <= EPS) {
        var out: NdcOpt;
        out.ok = false;
        out.ndc = vec3<f32>(0.0, 0.0, 0.0);
        return out;
    }
    return clip_to_ndc(view_to_clip(view));
}

// NDC->screen: the viewport mapping with y flipped to the usual top-left pixel
// convention, matching the reference `ndc_to_screen`.
fn ndc_to_screen(ndc: vec3<f32>) -> vec2<f32> {
    let sx = (ndc.x * 0.5 + 0.5) * camera.viewport_w;
    let sy = (0.5 - ndc.y * 0.5) * camera.viewport_h;
    return vec2<f32>(sx, sy);
}

// Whether an NDC point lies outside the clip cube, matching the reference
// culling band on x/y in [-1, 1] and z in [0, 1].
fn ndc_outside(ndc: vec3<f32>) -> bool {
    return ndc.x < -1.0 - EPS
        || ndc.x > 1.0 + EPS
        || ndc.y < -1.0 - EPS
        || ndc.y > 1.0 + EPS
        || ndc.z < -EPS
        || ndc.z > 1.0 + EPS;
}

// NDC depth -> positive linear view-space distance, matching the reference
// `linearize_depth`. The perspective denominator is EPS-guarded.
fn linearize_depth(ndc_z: f32) -> f32 {
    if (is_perspective()) {
        let range = camera.far - camera.near;
        let denom = camera.far - range * ndc_z;
        if (abs(denom) <= EPS) {
            return camera.far;
        }
        return camera.far * camera.near / denom;
    }
    return camera.near + (camera.far - camera.near) * ndc_z;
}

// Unit vector along `v`, or zero when `v` is numerically zero, matching the
// reference `Vec3::normalize_or_zero`.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq > EPS_LEN_SQ) {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Approximate NDC half-size a world bounding sphere subtends, matching the
// reference `screen_coverage_ndc`: depth-scaled for perspective, depth-
// invariant for orthographic, zero at or behind the camera.
fn coverage_ndc(world_center: vec3<f32>, world_radius: f32) -> f32 {
    let view = world_to_view(world_center);
    if (is_perspective()) {
        if (view.z <= EPS) {
            return 0.0;
        }
        let inv = safe_recip(view.z * camera.p1);
        return abs(world_radius * inv);
    }
    let inv = safe_recip(camera.p1);
    return abs(world_radius * inv);
}

// Coarse frustum bucket, matching the reference `classify_visibility`.
fn classify_visibility(world_pos: vec3<f32>) -> u32 {
    let view = world_to_view(world_pos);
    if (is_perspective() && view.z <= EPS) {
        return VIS_BEHIND_CAMERA;
    }
    let ndc = clip_to_ndc(view_to_clip(view));
    if (!ndc.ok) {
        return VIS_DEGENERATE;
    }
    if (ndc_outside(ndc.ndc)) {
        return VIS_OUTSIDE_FRUSTUM;
    }
    return VIS_VISIBLE;
}

@compute @workgroup_size(64)
fn run(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let world = queries[idx].world_pos.xyz;
    let ndc_point = queries[idx].ndc_point.xyz;
    let radius = queries[idx].aux.x;
    let ndc_z = queries[idx].aux.y;

    let view = world_to_view(world);
    let clip = view_to_clip(view);
    let cn = clip_to_ndc(clip);
    let wn = world_to_ndc(world);

    var res: Result;
    res.view = vec4<f32>(view, 0.0);
    res.clip = clip;

    var clip_flag = 0u;
    var clip_ndc = vec3<f32>(0.0, 0.0, 0.0);
    if (cn.ok) {
        clip_flag = 1u;
        clip_ndc = cn.ndc;
    }
    res.ndc_from_clip = vec4<f32>(clip_ndc, 0.0);

    var world_flag = 0u;
    var world_ndc = vec3<f32>(0.0, 0.0, 0.0);
    if (wn.ok) {
        world_flag = 1u;
        world_ndc = wn.ndc;
    }
    res.ndc_from_world = vec4<f32>(world_ndc, 0.0);

    let screen_ndc = ndc_to_screen(ndc_point);
    res.screen_from_ndc = vec4<f32>(screen_ndc, 0.0, 0.0);

    var screen_flag = 0u;
    var screen = vec2<f32>(0.0, 0.0);
    if (wn.ok && !ndc_outside(wn.ndc)) {
        screen_flag = 1u;
        screen = ndc_to_screen(wn.ndc);
    }
    res.screen_from_world = vec4<f32>(screen, 0.0, 0.0);

    let rel = world - camera.world_pos.xyz;
    let dist_sq = dot(rel, rel);
    let dist = sqrt(dist_sq);
    let cov_ndc = coverage_ndc(world, radius);
    let cov_px = cov_ndc * 0.5 * camera.viewport_h;
    let lin = linearize_depth(ndc_z);
    let dir = normalize_or_zero(camera.world_pos.xyz - world);

    res.scalars = vec4<f32>(lin, dist, dist_sq, cov_ndc);
    res.dir_cov = vec4<f32>(dir, cov_px);
    res.flags = vec4<u32>(clip_flag, world_flag, screen_flag, classify_visibility(world));

    results[idx] = res;
}
"#;

/// One camera query: the world point to project plus the independent samples
/// the standalone mappings consume.
///
/// `world_pos` drives the transform chain, the distance/direction scalars and
/// the coverage; `ndc_point` is the independent `NDC` fed to the standalone
/// [`ndc_to_screen`](prism_render_architecture::particle::camera::CameraProjection::ndc_to_screen)
/// mapping; `ndc_z` is the independent `NDC` depth fed to
/// [`linearize_depth`](prism_render_architecture::particle::camera::CameraProjection::linearize_depth);
/// `radius` is the bounding-sphere radius the coverage queries subtend. Derives
/// only [`PartialEq`] (no `Eq`/`Hash`) because it holds `f32` geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraQuery {
    /// The world point projected through the transform chain.
    pub world_pos: Vec3,
    /// The independent `NDC` point fed to the standalone screen mapping.
    pub ndc_point: Vec3,
    /// The independent `NDC` depth fed to the linearize query.
    pub ndc_z: f32,
    /// The bounding-sphere radius the coverage queries subtend.
    pub radius: f32,
}

/// The resolved verdict for one query, the host-side mirror of the kernel's
/// `Result` lane.
///
/// `ndc_from_clip`, `ndc_from_world` and `screen_from_world` are [`Option`]s
/// mirroring the reference's fallible returns (`None` on a camera-plane `w`, a
/// behind-camera point or a frustum-culled point). The remaining fields are
/// always meaningful. Derives only [`PartialEq`] (no `Eq`/`Hash`) because it
/// holds `f32` payloads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraResult {
    /// View-space position (backs `world_to_view` and `view_space_position`).
    pub view: Vec3,
    /// Homogeneous clip coordinate `[x, y, z, w]` pre perspective divide.
    pub clip: [f32; 4],
    /// `clip_to_ndc` of the clip coordinate, or [`None`] on a camera-plane `w`.
    pub ndc_from_clip: Option<Vec3>,
    /// `world_to_ndc`, or [`None`] behind the camera or on a degenerate divide.
    pub ndc_from_world: Option<Vec3>,
    /// `ndc_to_screen` of the independent `ndc_point` sample.
    pub screen_from_ndc: [f32; 2],
    /// `world_to_screen`, or [`None`] when the point is culled.
    pub screen_from_world: Option<[f32; 2]>,
    /// Linearized view-space depth from `ndc_z`.
    pub linearize_depth: f32,
    /// Straight-line distance to the camera.
    pub distance: f32,
    /// Squared distance to the camera.
    pub distance_squared: f32,
    /// `NDC` half-size the bounding sphere subtends.
    pub coverage_ndc: f32,
    /// Pixel radius the bounding sphere subtends.
    pub coverage_pixels: f32,
    /// Unit direction from the point toward the camera (zero at the camera).
    pub direction: Vec3,
    /// Coarse frustum bucket.
    pub visibility: ClipVisibility,
}

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`CAMERA_WGSL`]: the query count and three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// The shared camera as uploaded. `96`-byte `std430` layout matching `Camera`
/// in the shader: the world position and three basis axes padded to `vec4`
/// lanes, the near/far planes, the two projection parameters, the viewport
/// dimensions and the projection-kind code.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuCameraUniform {
    /// Camera world position in `xyz`; the `w` lane is unused padding.
    world_pos: [f32; 4],
    /// View `+X` basis axis in `xyz`; the `w` lane is unused padding.
    right: [f32; 4],
    /// View `+Y` basis axis in `xyz`; the `w` lane is unused padding.
    up: [f32; 4],
    /// View `+Z` (look) basis axis in `xyz`; the `w` lane is unused padding.
    forward: [f32; 4],
    /// Near clip distance.
    near: f32,
    /// Far clip distance.
    far: f32,
    /// First projection parameter (`tan_half_fov_x` or `half_width`).
    p0: f32,
    /// Second projection parameter (`tan_half_fov_y` or `half_height`).
    p1: f32,
    /// Viewport width in pixels.
    viewport_w: f32,
    /// Viewport height in pixels.
    viewport_h: f32,
    /// Projection-kind code (`0` perspective, `1` orthographic).
    kind: u32,
    /// Padding word to a `16`-byte-aligned, `96`-byte struct.
    pad: u32,
}

impl GpuCameraUniform {
    /// Packs a reference [`CameraProjection`] into the `std430` uniform layout.
    fn from_projection(camera: &CameraProjection) -> GpuCameraUniform {
        let w = camera.world_position;
        let r = camera.basis.right;
        let u = camera.basis.up;
        let f = camera.basis.forward;
        let (kind, p0, p1) = match camera.projection {
            ProjectionKind::Perspective {
                tan_half_fov_x,
                tan_half_fov_y,
            } => (KIND_PERSPECTIVE, tan_half_fov_x, tan_half_fov_y),
            ProjectionKind::Orthographic {
                half_width,
                half_height,
            } => (KIND_ORTHOGRAPHIC, half_width, half_height),
        };
        GpuCameraUniform {
            world_pos: [w.x, w.y, w.z, 0.0],
            right: [r.x, r.y, r.z, 0.0],
            up: [u.x, u.y, u.z, 0.0],
            forward: [f.x, f.y, f.z, 0.0],
            near: camera.near,
            far: camera.far,
            p0,
            p1,
            viewport_w: camera.viewport.width,
            viewport_h: camera.viewport.height,
            kind,
            pad: 0,
        }
    }
}

/// One query as uploaded. `48`-byte `std430` stride matching `Query` in the
/// shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// World point in `xyz`; the `w` lane is unused padding.
    world_pos: [f32; 4],
    /// Independent `NDC` sample in `xyz`; the `w` lane is unused padding.
    ndc_point: [f32; 4],
    /// `x` = coverage radius, `y` = `NDC` depth; `z`/`w` unused padding.
    aux: [f32; 4],
}

impl GpuQuery {
    /// Packs a [`CameraQuery`] into the `std430` upload layout.
    fn from_query(query: &CameraQuery) -> GpuQuery {
        let p = query.world_pos;
        let n = query.ndc_point;
        GpuQuery {
            world_pos: [p.x, p.y, p.z, 0.0],
            ndc_point: [n.x, n.y, n.z, 0.0],
            aux: [query.radius, query.ndc_z, 0.0, 0.0],
        }
    }
}

/// One result as read back. `144`-byte `std430` stride matching `Result` in the
/// shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// View-space position in `xyz`; the `w` lane is unused padding.
    view: [f32; 4],
    /// Homogeneous clip coordinate `[x, y, z, w]`.
    clip: [f32; 4],
    /// `clip_to_ndc` result in `xyz` (zeroed when the flag is clear).
    ndc_from_clip: [f32; 4],
    /// `world_to_ndc` result in `xyz` (zeroed when the flag is clear).
    ndc_from_world: [f32; 4],
    /// `ndc_to_screen` result in `xy`; `zw` unused padding.
    screen_from_ndc: [f32; 4],
    /// `world_to_screen` result in `xy` (zeroed when the flag is clear).
    screen_from_world: [f32; 4],
    /// `linearize_depth`, `distance`, `distance_squared`, `coverage_ndc`.
    scalars: [f32; 4],
    /// `direction_to_camera` in `xyz`, `coverage_pixels` in `w`.
    dir_cov: [f32; 4],
    /// `clip_to_ndc` flag, `world_to_ndc` flag, `world_to_screen` flag,
    /// visibility code.
    flags: [u32; 4],
}

/// Maps one kernel `Result` lane back to the host [`CameraResult`].
fn decode_result(raw: &GpuResult) -> CameraResult {
    let ndc_from_clip = if raw.flags[0] == FLAG_SOME {
        Some(Vec3::new(
            raw.ndc_from_clip[0],
            raw.ndc_from_clip[1],
            raw.ndc_from_clip[2],
        ))
    } else {
        None
    };
    let ndc_from_world = if raw.flags[1] == FLAG_SOME {
        Some(Vec3::new(
            raw.ndc_from_world[0],
            raw.ndc_from_world[1],
            raw.ndc_from_world[2],
        ))
    } else {
        None
    };
    let screen_from_world = if raw.flags[2] == FLAG_SOME {
        Some([raw.screen_from_world[0], raw.screen_from_world[1]])
    } else {
        None
    };
    let visibility = match raw.flags[3] {
        VIS_BEHIND_CAMERA => ClipVisibility::BehindCamera,
        VIS_OUTSIDE_FRUSTUM => ClipVisibility::OutsideFrustum,
        VIS_DEGENERATE => ClipVisibility::Degenerate,
        code => {
            debug_assert_eq!(
                code, VIS_VISIBLE,
                "kernel emitted out-of-range visibility code {code}; only                  VIS_VISIBLE (0) is left after the explicit arms"
            );
            ClipVisibility::Visible
        }
    };
    CameraResult {
        view: Vec3::new(raw.view[0], raw.view[1], raw.view[2]),
        clip: raw.clip,
        ndc_from_clip,
        ndc_from_world,
        screen_from_ndc: [raw.screen_from_ndc[0], raw.screen_from_ndc[1]],
        screen_from_world,
        linearize_depth: raw.scalars[0],
        distance: raw.scalars[1],
        distance_squared: raw.scalars[2],
        coverage_ndc: raw.scalars[3],
        coverage_pixels: raw.dir_cov[3],
        direction: Vec3::new(raw.dir_cov[0], raw.dir_cov[1], raw.dir_cov[2]),
        visibility,
    }
}

/// The `CPU` golden verdict for one query, dispatching to the reference
/// [`CameraProjection`](prism_render_architecture::particle::camera::CameraProjection)
/// methods so callers (and the parity test) can pin the twin lane for lane.
#[must_use]
pub fn cpu_reference(camera: &CameraProjection, query: &CameraQuery) -> CameraResult {
    let view = camera.world_to_view(query.world_pos);
    let clip = camera.view_to_clip(view);
    CameraResult {
        view,
        clip,
        ndc_from_clip: CameraProjection::clip_to_ndc(clip),
        ndc_from_world: camera.world_to_ndc(query.world_pos),
        screen_from_ndc: camera.ndc_to_screen(query.ndc_point),
        screen_from_world: camera.world_to_screen(query.world_pos),
        linearize_depth: camera.linearize_depth(query.ndc_z),
        distance: camera.distance_to_camera(query.world_pos),
        distance_squared: camera.distance_squared_to_camera(query.world_pos),
        coverage_ndc: camera.screen_coverage_ndc(query.world_pos, query.radius),
        coverage_pixels: camera.screen_coverage_pixels(query.world_pos, query.radius),
        direction: camera.direction_to_camera(query.world_pos),
        visibility: camera.classify_visibility(query.world_pos),
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

/// A compiled, reusable camera-query pipeline.
pub struct GpuCamera {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCamera {
    /// Compiles the camera-query kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCamera {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_camera"),
            source: ShaderSource::Wgsl(CAMERA_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_camera_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Uniform),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_camera_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_camera_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("run"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCamera {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries` against the shared `camera`, returning
    /// one [`CameraResult`] per query in input order.
    ///
    /// The returned result for query `q` mirrors the full reference query set on
    /// `camera` and `q`; see [`cpu_reference`]. An empty `queries` slice yields
    /// an empty result — storage buffers cannot be zero-sized, so it is handled
    /// by an early return before any dispatch.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        camera: &CameraProjection,
        queries: &[CameraQuery],
    ) -> Vec<CameraResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = GpuParams {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let gpu_camera = GpuCameraUniform::from_projection(camera);
        let gpu_queries: Vec<GpuQuery> = queries.iter().map(GpuQuery::from_query).collect();

        let out_bytes = (queries.len() * size_of::<GpuResult>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_camera_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let camera_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_camera_camera"),
            contents: bytemuck::bytes_of(&gpu_camera),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_camera_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_camera_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_camera_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_camera_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: camera_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_camera_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_camera_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
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
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), queries.len());

        raw.iter().map(decode_result).collect()
    }
}
