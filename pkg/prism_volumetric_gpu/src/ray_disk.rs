//! `wgpu` compute twin of the analytic `ray`-disk and `ray`-annulus
//! intersection contract
//! ([`ray_disk`](prism_render_architecture::particle::ray_disk), particle
//! design §10, §14).
//!
//! The `CPU` golden
//! [`ray_disk`](prism_render_architecture::particle::ray_disk) owns the
//! closed-form solution of the flat bounded circle problem. A [`Disk`] is a
//! center, a face normal (not required to be unit length) and a radius; an
//! [`Annulus`] additionally carries an inner radius that bores a circular hole.
//! The reference first solves the one-division `ray`-plane crossing
//! `t = (center - origin)·n / (dir·n)`, rejects a direction parallel to the
//! plane (its `dir·n` magnitude below the compare epsilon) and a crossing that
//! falls behind the origin (`t < -CMP_EPS`), then classifies the crossing by the
//! *squared* in-plane radial distance of the hit point from the center. A disk
//! accepts the hit when that distance is within `radius²` (plus the epsilon
//! slack); an annulus additionally requires it to clear `inner²`. The outward
//! unit normal is flipped to face against the incoming ray — the single `sqrt`
//! in the whole module — and the [`HitFace`] is read straight off the sign of
//! the denominator. [`GpuRayDisk`] is the on-device twin: it runs one thread per
//! query and reproduces the same answers the batch reference produces, so a
//! passing real-device parity test is direct evidence the ported kernel solves
//! the same geometry and classifies the same degenerate cases the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Every per-query answer the reference computes for a hit query is reproduced:
//! the degenerate guard (`Disk::is_degenerate` / `Annulus::is_degenerate`), the
//! private `solve_plane` crossing with its parallel-direction rejection, the
//! `t < -CMP_EPS` behind-origin rejection, the squared radial-distance band test
//! (`radius²` for a disk, `[inner², outer²]` for an annulus) and the final
//! `build_hit` that orients the unit normal against the ray and records the
//! `HitFace`. The boolean `intersects` predicate the reference exposes is the
//! same solve without the final `sqrt`; the twin emits the hit flag, the face
//! code, the ray parameter, the world-space point and the oriented unit normal
//! so a caller can reconstruct either the predicate or the full
//! [`RayDiskHit`](prism_render_architecture::particle::ray_disk::RayDiskHit).
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `+ - * /` and one `sqrt` for the emitted unit normal — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan` or optional device feature, so it runs unmodified
//! on `Metal`, `Vulkan` and `DX12`. The only non-rational operation is the
//! `sqrt` of the normal length, matching the reference's single `f32::sqrt`.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds, one
//! guarded division and at most one `sqrt`, so `CPU` and `GPU` evaluate the same
//! closed form in the same order. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32`
//! fields while pinning the hit flag and the face code exactly, tight enough to
//! catch a genuinely wrong port (a dropped band edge, a wrong normal flip, a
//! wrong parallel guard) yet loose enough to admit legal fused multiply-add
//! contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`ray_disk`](prism_render_architecture::particle::ray_disk) plus `wgpu`
//! compute dispatch; no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::ray_disk::{Annulus, Disk, Ray};
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

/// Primitive kind code for a solid [`Disk`] query, matching the `kind` lane the
/// kernel branches on. A direct `f32` equality is forbidden, so the kind travels
/// as an integer code.
const KIND_DISK: u32 = 0;
/// Primitive kind code for an [`Annulus`] query (a disk with a circular hole),
/// matching the `kind` lane the kernel branches on.
const KIND_ANNULUS: u32 = 1;

/// Face code written when the ray strikes the side the stored normal points
/// toward, mirroring the reference `HitFace::Front`
/// (`prism_render_architecture::particle::ray_disk::HitFace::Front`). The face
/// is a discrete classification, so it is compared with an exact `==`.
pub const FACE_FRONT: u32 = 0;
/// Face code written when the ray strikes the side opposite the stored normal,
/// mirroring the reference `HitFace::Back`
/// (`prism_render_architecture::particle::ray_disk::HitFace::Back`).
pub const FACE_BACK: u32 = 1;

/// The inlined `WGSL` kernel source. The crate ships as a single source string
/// per module (no external `.wgsl` / `.wesl`); the entry point `solve` mirrors
/// the `CPU` golden
/// [`ray_disk`](prism_render_architecture::particle::ray_disk) branch for
/// branch; see the module documentation for the algorithm.
const RAY_DISK_WGSL: &str = r#"
// Analytic ray-disk / ray-annulus twin: one thread per query solves the one-
// division ray-plane crossing, rejects a parallel direction and a behind-origin
// crossing, classifies the hit by the squared in-plane radial distance (within
// radius^2 for a disk, within [inner^2, outer^2] for an annulus) and emits the
// nearest forward hit with its outward unit normal flipped against the ray and
// the struck face. It mirrors the CPU golden `particle::ray_disk`, uses only the
// portable core-WGSL subset (abs/min/max and + - * / plus one sqrt for the unit
// normal) and takes no optional feature, so it runs unmodified on Metal, Vulkan
// and DX12.
//
// Provenance: twinned from this repository's particle::ray_disk; no third-party
// engine source or derived code.

// Epsilon guarding the plane-parameter division, accepting a hit exactly on a
// radius, and comparing the parameter against zero without ever writing an exact
// == / != on an f32. Matches the reference CMP_EPS.
const CMP_EPS: f32 = 1.0e-6;

// Primitive kind codes, mirroring the host KIND_* constants.
const KIND_DISK: u32 = 0u;
const KIND_ANNULUS: u32 = 1u;

// Face codes, mirroring the host FACE_* constants and the reference HitFace.
const FACE_FRONT: u32 = 0u;
const FACE_BACK: u32 = 1u;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Ray origin in the first three lanes; the primitive kind code in the
    // fourth (bitcast from u32 by the host packer).
    origin: vec3<f32>,
    kind: u32,
    // Ray direction (not required to be unit length); a pad lane follows.
    dir: vec3<f32>,
    pad0: f32,
    // Primitive center in the first three lanes; the (outer, for an annulus)
    // radius in the fourth.
    center: vec3<f32>,
    radius: f32,
    // Primitive face normal (may be un-normalized); a pad lane follows.
    normal: vec3<f32>,
    pad1: f32,
    // Annulus inner radius in the first lane; three pad lanes fill the slot.
    inner: f32,
    pad2: f32,
    pad3: f32,
    pad4: f32,
}

struct Result {
    // Hit flag (0 / 1), struck-face code, ray parameter and a pad lane.
    hit: u32,
    face: u32,
    t: f32,
    pad0: f32,
    // World-space hit point; a pad lane follows.
    point: vec3<f32>,
    pad1: f32,
    // Outward unit surface normal at the hit; a pad lane follows.
    normal: vec3<f32>,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Unit vector in the same direction, or the zero vector when the input is
// shorter than CMP_EPS, mirroring the reference `normalize_or_zero` so a
// degenerate normal can never yield a NaN.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len = sqrt(dot(v, v));
    if (len < CMP_EPS) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    return v * (1.0 / len);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let origin = q.origin;
    let dir = q.dir;
    let center = q.center;
    let normal = q.normal;
    let radius = q.radius;
    let inner = q.inner;
    let kind = q.kind;

    var hit: u32 = 0u;
    var face: u32 = FACE_FRONT;
    var t_out: f32 = 0.0;
    var point_out = vec3<f32>(0.0, 0.0, 0.0);
    var normal_out = vec3<f32>(0.0, 0.0, 0.0);

    // is_degenerate: a vanishing radius or a vanishing normal leaves no plane;
    // an annulus additionally needs its outer radius to exceed the inner one.
    let normal_len2 = dot(normal, normal);
    var degenerate = radius <= CMP_EPS || normal_len2 <= CMP_EPS * CMP_EPS;
    if (kind == KIND_ANNULUS) {
        if (radius <= inner + CMP_EPS) {
            degenerate = true;
        }
    }

    // solve_plane: reject a direction parallel to the plane, then the one-
    // division crossing; reject a crossing behind the origin.
    let denom = dot(dir, normal);
    if (!degenerate && abs(denom) >= CMP_EPS) {
        let plane_t = dot(center - origin, normal) / denom;
        if (plane_t >= -CMP_EPS) {
            let t = max(plane_t, 0.0);
            let point = origin + dir * t;
            let radial = point - center;
            let radial_dist_sq = dot(radial, radial);
            // Radial-band classification in squared space (no sqrt).
            var inside = radial_dist_sq <= radius * radius + CMP_EPS;
            if (kind == KIND_ANNULUS) {
                let inside_hole = radial_dist_sq < inner * inner - CMP_EPS;
                if (inside_hole) {
                    inside = false;
                }
            }
            if (inside) {
                // build_hit: orient the unit normal against the incoming ray and
                // read the struck face off the sign of the denominator.
                let unit = normalize_or_zero(normal);
                if (denom < 0.0) {
                    face = FACE_FRONT;
                    normal_out = unit;
                } else {
                    face = FACE_BACK;
                    normal_out = unit * (-1.0);
                }
                hit = 1u;
                t_out = t;
                point_out = point;
            }
        }
    }

    var out: Result;
    out.hit = hit;
    out.face = face;
    out.t = t_out;
    out.pad0 = 0.0;
    out.point = point_out;
    out.pad1 = 0.0;
    out.normal = normal_out;
    out.pad2 = 0.0;
    results[idx] = out;
}
"#;

/// The flat circular primitive a [`RayDiskQuery`] traces against: either a solid
/// [`Disk`] or an [`Annulus`] with a circular hole, mirroring the two reference
/// types the golden
/// [`ray_disk`](prism_render_architecture::particle::ray_disk) exposes.
///
/// Provenance: twinned from this repository's
/// [`ray_disk`](prism_render_architecture::particle::ray_disk).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RayDiskTarget {
    /// A solid flat disk (`Disk`).
    Disk(Disk),
    /// A flat annular band (`Annulus`).
    Annulus(Annulus),
}

/// One `ray`-disk / `ray`-annulus query: a [`Ray`] traced against a
/// [`RayDiskTarget`], the same inputs the reference `Disk::intersect` /
/// `Annulus::intersect` consume.
///
/// Provenance: twinned from this repository's
/// [`ray_disk`](prism_render_architecture::particle::ray_disk).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayDiskQuery {
    /// The ray being traced (direction need not be unit length).
    pub ray: Ray,
    /// The flat primitive being intersected and classified against.
    pub target: RayDiskTarget,
}

impl RayDiskQuery {
    /// Builds a query tracing `ray` against a solid [`Disk`].
    #[must_use]
    pub const fn disk(ray: Ray, disk: Disk) -> RayDiskQuery {
        RayDiskQuery {
            ray,
            target: RayDiskTarget::Disk(disk),
        }
    }

    /// Builds a query tracing `ray` against an [`Annulus`].
    #[must_use]
    pub const fn annulus(ray: Ray, annulus: Annulus) -> RayDiskQuery {
        RayDiskQuery {
            ray,
            target: RayDiskTarget::Annulus(annulus),
        }
    }
}

/// The resolved on-device answer for one query, mirroring the reference
/// `Option<RayDiskHit>`: a hit flag, the struck-face code, the ray parameter,
/// the world-space point and the outward unit normal.
///
/// The `hit` flag is `1` exactly when the reference returns `Some` and `0`
/// otherwise; the `face` code is [`FACE_FRONT`] or [`FACE_BACK`] and is only
/// meaningful when `hit` is `1`. The flag and the face code are discrete, so a
/// consumer compares them with an exact `==`; `t`, `point` and `normal` are
/// continuous and are compared within the module tolerance.
///
/// Provenance: twinned from this repository's
/// [`ray_disk`](prism_render_architecture::particle::ray_disk).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuRayDiskHit {
    /// `1` when the forward half-line strikes the primitive, `0` otherwise.
    pub hit: u32,
    /// The struck-face code ([`FACE_FRONT`] or [`FACE_BACK`]); meaningful only
    /// when `hit` is `1`.
    pub face: u32,
    /// The ray parameter at the hit (`>= 0`); equals the hit distance when the
    /// ray direction is unit length. Zero when there is no hit.
    pub t: f32,
    /// The world-space intersection point. Zero when there is no hit.
    pub point: [f32; 3],
    /// The unit surface normal oriented against the incoming ray. Zero when
    /// there is no hit.
    pub normal: [f32; 3],
}

/// `repr(C)` `std430` layout of one packed query: five `vec4` slots holding
/// `(origin.xyz, kind)`, `(dir.xyz, pad)`, `(center.xyz, radius)`,
/// `(normal.xyz, pad)` and `(inner, pad, pad, pad)` — `80` bytes, each `vec3` on
/// its `16`-byte-aligned slot exactly as the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Ray origin.
    origin: [f32; 3],
    /// Primitive kind code, packed into the origin slot's fourth lane.
    kind: u32,
    /// Ray direction.
    dir: [f32; 3],
    /// Padding lane after the direction.
    pad0: f32,
    /// Primitive center.
    center: [f32; 3],
    /// The disk radius, or the annulus outer radius.
    radius: f32,
    /// Primitive face normal (may be un-normalized).
    normal: [f32; 3],
    /// Padding lane after the normal.
    pad1: f32,
    /// Annulus inner radius (zero for a disk).
    inner: f32,
    /// Padding lane.
    pad2: f32,
    /// Padding lane.
    pad3: f32,
    /// Padding lane.
    pad4: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &RayDiskQuery) -> GpuQuery {
        let origin = [query.ray.origin.x, query.ray.origin.y, query.ray.origin.z];
        let dir = [query.ray.dir.x, query.ray.dir.y, query.ray.dir.z];
        let (kind, center, radius, normal, inner) = match query.target {
            RayDiskTarget::Disk(d) => (
                KIND_DISK,
                [d.center.x, d.center.y, d.center.z],
                d.radius,
                [d.normal.x, d.normal.y, d.normal.z],
                0.0,
            ),
            RayDiskTarget::Annulus(a) => (
                KIND_ANNULUS,
                [a.center.x, a.center.y, a.center.z],
                a.outer,
                [a.normal.x, a.normal.y, a.normal.z],
                a.inner,
            ),
        };
        GpuQuery {
            origin,
            kind,
            dir,
            pad0: 0.0,
            center,
            radius,
            normal,
            pad1: 0.0,
            inner,
            pad2: 0.0,
            pad3: 0.0,
            pad4: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: two scalars `(hit, face)`, the ray
/// parameter and a pad lane, then two `vec4` slots for the hit point and the
/// outward normal — `48` bytes matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Hit flag (`0` / `1`).
    hit: u32,
    /// Struck-face code ([`FACE_FRONT`] / [`FACE_BACK`]).
    face: u32,
    /// Ray parameter at the hit.
    t: f32,
    /// Padding lane.
    pad0: f32,
    /// World-space hit point.
    point: [f32; 3],
    /// Padding lane after the hit point.
    pad1: f32,
    /// Outward unit surface normal.
    normal: [f32; 3],
    /// Padding lane after the normal.
    pad2: f32,
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

/// Decodes one packed [`GpuResult`] into the public [`GpuRayDiskHit`].
fn decode_result(raw: &GpuResult) -> GpuRayDiskHit {
    GpuRayDiskHit {
        hit: raw.hit,
        face: raw.face,
        t: raw.t,
        point: raw.point,
        normal: raw.normal,
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

/// A compiled, reusable `ray`-disk / `ray`-annulus compute pipeline.
pub struct GpuRayDisk {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRayDisk {
    /// Compiles the `ray`-disk kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRayDisk {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_ray_disk"),
            source: ShaderSource::Wgsl(RAY_DISK_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_ray_disk_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_ray_disk_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_ray_disk_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRayDisk {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`GpuRayDiskHit`] per input,
    /// in order.
    ///
    /// Each result equals the reference answer (`Disk::intersect` /
    /// `Annulus::intersect`) to within the tolerance documented on this module:
    /// the hit flag and the face code match exactly, and `t`, `point` and
    /// `normal` match within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`. An empty
    /// input returns an empty vector with no dispatch issued, since a storage
    /// buffer cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[RayDiskQuery]) -> Vec<GpuRayDiskHit> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ray_disk_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ray_disk_output"),
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
            label: Some("prism_volumetric_ray_disk_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_ray_disk_bind_group"),
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
            label: Some("prism_volumetric_ray_disk_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_ray_disk_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_ray_disk_pass"),
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
