//! `wgpu` compute twin of the analytic ray-sphere intersection contract
//! ([`ray_sphere`](prism_render_architecture::particle::ray_sphere), particle
//! design §10, §14).
//!
//! The `CPU` golden
//! [`ray_sphere`](prism_render_architecture::particle::ray_sphere) owns the
//! closed-form solution of the ray-sphere problem: it forms the scalar
//! quadratic `a t^2 + b t + c = 0` whose roots are the ray parameters at the two
//! surface crossings, classifies the discriminant `b^2 - 4 a c` (negative →
//! miss, near-zero → tangent double root, positive → two distinct crossings),
//! and reports the boolean hit test, the ordered root pair and the nearest
//! forward hit with its outward unit normal. [`GpuRaySphere`] is the on-device
//! twin: it runs one thread per probe and reproduces the same answers the
//! batch reference produces, so a passing real-device parity test is direct
//! evidence the ported kernel solves the same quadratic and classifies the same
//! degenerate cases the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Every per-probe answer the reference computes is reproduced: [`Ray::at`]
//! (the `origin + t * dir` evaluation), [`Sphere::contains`] is folded into the
//! degenerate-radius guard, [`Sphere::solve`] (the ordered [`RootPair`] plus the
//! `is_tangent` test), [`Sphere::intersects`] (the unsigned boolean) and
//! [`Sphere::first_hit`] (the [`RaySphereHit`]: the first non-negative `t`, its
//! world-space point and the outward unit normal). The four degenerate cases
//! the reference handles explicitly are mirrored branch for branch: the origin
//! inside the sphere (only the far root is a forward hit), a tangent graze (the
//! discriminant collapses to one double root), a miss (negative discriminant)
//! and a hit entirely behind the origin (both roots negative, so `first_hit`
//! reports nothing while the unsigned `intersects` still sees the crossing). A
//! near-zero radius or a near-zero direction is likewise rejected rather than
//! producing a `NaN`, and the hand-rolled `normalize_or_zero` guards the normal
//! against a zero-length divide exactly as the reference does.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`, `abs`,
//! `sqrt`, `+ - * /`, unsigned bit arithmetic — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `tan` or optional device feature, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`. The only non-rational operation is the one
//! `sqrt` of the discriminant, matching the reference's single `f32::sqrt`.
//!
//! # Correctness model
//!
//! Each probe is a fixed, non-reorderable sequence of multiplies, adds and one
//! `sqrt`, so `CPU` and `GPU` evaluate the same closed form in the same order.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits by a few units in the last
//! place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32` fields while pinning
//! the integer classification flags exactly, tight enough to catch a genuinely
//! wrong port (a swapped root order, a dropped discriminant branch, a wrong
//! normal) yet loose enough to admit legal fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`ray_sphere`](prism_render_architecture::particle::ray_sphere) plus `wgpu`
//! compute dispatch; no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::ray_sphere::{Ray, RaySphereHit, RootPair, Sphere, Vec3};
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

/// Bit flag set when [`Sphere::solve`] returned a root pair (the line reaches
/// the sphere and neither the direction nor the radius is degenerate).
const FLAG_SOLVED: u32 = 1;
/// Bit flag set when the two roots coincide within the compare epsilon — a
/// tangent graze (`RootPair::is_tangent`).
const FLAG_TANGENT: u32 = 2;
/// Bit flag set when the forward half-line crosses the surface
/// ([`Sphere::intersects`]).
const FLAG_INTERSECTS: u32 = 4;
/// Bit flag set when a nearest forward hit exists ([`Sphere::first_hit`]).
const FLAG_HIT: u32 = 8;

/// The portable core-`WGSL` ray-sphere kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`ray_sphere`](prism_render_architecture::particle::ray_sphere) branch for
/// branch; see the module documentation for the algorithm.
const RAY_SPHERE_WGSL: &str = r#"
// Analytic ray-sphere twin: one thread per probe solves the quadratic
// `a t^2 + b t + c = 0`, classifies the discriminant, and emits the ordered
// roots, the tangent / intersects / hit flags, and the nearest forward hit with
// its outward unit normal. It mirrors the CPU golden `particle::ray_sphere`,
// uses only the portable core-WGSL subset (min/max/abs/sqrt and + - * / plus
// unsigned bit math) and takes no optional feature, so it runs unmodified on
// Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's particle::ray_sphere; no
// third-party engine source or derived code.

// Epsilon used to guard divisions, classify the discriminant, and compare
// parameters against zero without ever writing an exact == / != on an f32.
const CMP_EPS: f32 = 1.0e-6;

struct Params {
    // Number of probes in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Probe {
    // Ray origin in the first three lanes; sphere radius in the fourth.
    origin: vec3<f32>,
    radius: f32,
    // Ray direction (not required to be unit length); a pad lane follows.
    dir: vec3<f32>,
    pad0: f32,
    // Sphere center; a pad lane follows.
    center: vec3<f32>,
    pad1: f32,
}

struct Result {
    // Ordered near/far roots, the chosen forward-hit parameter and the flags.
    t_near: f32,
    t_far: f32,
    hit_t: f32,
    flags: u32,
    // World-space hit point; a pad lane follows.
    point: vec3<f32>,
    pad0: f32,
    // Outward unit surface normal; a pad lane follows.
    normal: vec3<f32>,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> probes: array<Probe>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Unit vector in the same direction, or the zero vector when the input is
// shorter than CMP_EPS, mirroring the reference `normalize_or_zero` so a
// degenerate input can never yield a NaN.
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
    let probe = probes[idx];
    let origin = probe.origin;
    let dir = probe.dir;
    let center = probe.center;
    let radius = probe.radius;

    var t_near: f32 = 0.0;
    var t_far: f32 = 0.0;
    var hit_t: f32 = 0.0;
    var point = vec3<f32>(0.0, 0.0, 0.0);
    var normal = vec3<f32>(0.0, 0.0, 0.0);
    var flags: u32 = 0u;

    // Sphere::solve: reject a degenerate radius or a near-zero direction, then
    // classify the discriminant exactly as the reference does.
    let degenerate = radius <= CMP_EPS;
    let a = dot(dir, dir);
    var solved = false;
    if (!degenerate && a > CMP_EPS) {
        let oc = origin - center;
        let b = 2.0 * dot(oc, dir);
        let c = dot(oc, oc) - radius * radius;
        let disc = b * b - 4.0 * a * c;
        if (disc >= -CMP_EPS) {
            let inv_2a = 1.0 / (2.0 * a);
            if (disc <= CMP_EPS) {
                // Discriminant clamped to zero: a single tangent double root.
                let t = -b * inv_2a;
                t_near = t;
                t_far = t;
            } else {
                let sqrt_disc = sqrt(disc);
                let r0 = (-b - sqrt_disc) * inv_2a;
                let r1 = (-b + sqrt_disc) * inv_2a;
                t_near = min(r0, r1);
                t_far = max(r0, r1);
            }
            solved = true;
        }
    }

    if (solved) {
        flags = flags | 1u;
        // RootPair::is_tangent: the two roots coincide within the epsilon.
        if (abs(t_far - t_near) <= CMP_EPS) {
            flags = flags | 2u;
        }
        // Sphere::intersects: at least the far root is at or ahead of origin.
        if (t_far >= -CMP_EPS) {
            flags = flags | 4u;
        }
        // Sphere::first_hit: near root if forward, else the far wall, else none.
        var t: f32 = 0.0;
        var have = true;
        if (t_near >= -CMP_EPS) {
            t = t_near;
        } else if (t_far >= -CMP_EPS) {
            t = t_far;
        } else {
            have = false;
        }
        if (have) {
            // Clamp a tiny negative epsilon root up to exactly zero.
            t = max(t, 0.0);
            hit_t = t;
            // Ray::at: origin + t * dir.
            point = origin + dir * t;
            normal = normalize_or_zero(point - center);
            flags = flags | 8u;
        }
    }

    var out: Result;
    out.t_near = t_near;
    out.t_far = t_far;
    out.hit_t = hit_t;
    out.flags = flags;
    out.point = point;
    out.pad0 = 0.0;
    out.normal = normal;
    out.pad1 = 0.0;
    results[idx] = out;
}
"#;

/// One ray-sphere probe: a [`Ray`] tested against a [`Sphere`], the same pair
/// the reference [`Sphere::solve`] / [`Sphere::first_hit`] consume.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RaySphereProbe {
    /// The ray being traced (direction need not be unit length).
    pub ray: Ray,
    /// The sphere being intersected.
    pub sphere: Sphere,
}

impl RaySphereProbe {
    /// Builds a probe from a ray and a sphere.
    #[must_use]
    pub const fn new(ray: Ray, sphere: Sphere) -> RaySphereProbe {
        RaySphereProbe { ray, sphere }
    }
}

/// The resolved answer for one probe, mirroring every value the reference
/// reports: the solve result, the ordered roots, the tangent and intersect
/// predicates and the nearest forward hit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RaySphereResult {
    /// Whether [`Sphere::solve`] returned a root pair for this probe. When
    /// `false` the root fields are unspecified and [`RaySphereResult::hit`] is
    /// [`None`].
    pub solved: bool,
    /// The ordered root pair when `solved`, matching the reference
    /// [`RootPair`].
    pub roots: RootPair,
    /// Whether the two roots coincide within the compare epsilon (a tangent
    /// graze), matching [`RootPair::is_tangent`]. `false` when not solved.
    pub is_tangent: bool,
    /// Whether the forward half-line crosses the surface, matching
    /// [`Sphere::intersects`].
    pub intersects: bool,
    /// The nearest forward hit, matching [`Sphere::first_hit`], or [`None`].
    pub hit: Option<RaySphereHit>,
}

/// `repr(C)` `std430` layout of one packed probe: three `vec4` slots holding
/// `(origin.xyz, radius)`, `(dir.xyz, pad)` and `(center.xyz, pad)` — `48`
/// bytes, each `vec3` on its `16`-byte-aligned slot exactly as the `WGSL`
/// `Probe` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuProbe {
    /// Ray origin.
    origin: [f32; 3],
    /// Sphere radius, packed into the origin slot's fourth lane.
    radius: f32,
    /// Ray direction.
    dir: [f32; 3],
    /// Padding lane after the direction.
    pad0: f32,
    /// Sphere center.
    center: [f32; 3],
    /// Padding lane after the center.
    pad1: f32,
}

impl GpuProbe {
    /// Packs one probe into its `std430` image.
    fn new(probe: &RaySphereProbe) -> GpuProbe {
        GpuProbe {
            origin: [probe.ray.origin.x, probe.ray.origin.y, probe.ray.origin.z],
            radius: probe.sphere.radius,
            dir: [probe.ray.dir.x, probe.ray.dir.y, probe.ray.dir.z],
            pad0: 0.0,
            center: [
                probe.sphere.center.x,
                probe.sphere.center.y,
                probe.sphere.center.z,
            ],
            pad1: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: four scalars `(t_near, t_far,`
/// `hit_t, flags)` then two `vec4` slots for the hit point and the outward
/// normal — `48` bytes matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Nearer root parameter.
    t_near: f32,
    /// Farther root parameter.
    t_far: f32,
    /// Chosen forward-hit parameter (valid only when the hit flag is set).
    hit_t: f32,
    /// Packed classification flags (`FLAG_SOLVED` and friends).
    flags: u32,
    /// World-space hit point.
    point: [f32; 3],
    /// Padding lane after the point.
    pad0: f32,
    /// Outward unit surface normal.
    normal: [f32; 3],
    /// Padding lane after the normal.
    pad1: f32,
}

/// Uniform parameters for one dispatch: the probe count plus three pad words to
/// fill a `16`-byte, `16`-byte-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of probes in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A compiled, reusable ray-sphere compute pipeline.
pub struct GpuRaySphere {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRaySphere {
    /// Compiles the ray-sphere kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRaySphere {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_ray_sphere"),
            source: ShaderSource::Wgsl(RAY_SPHERE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_ray_sphere_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_ray_sphere_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_ray_sphere_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRaySphere {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every probe on-device and returns one [`RaySphereResult`] per
    /// input, in order.
    ///
    /// Each result equals the reference answers
    /// ([`Sphere::solve`], [`RootPair::is_tangent`], [`Sphere::intersects`] and
    /// [`Sphere::first_hit`]) to within the tolerance documented on this module.
    /// An empty input returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, probes: &[RaySphereProbe]) -> Vec<RaySphereResult> {
        if probes.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = probes.len();

        let packed: Vec<GpuProbe> = probes.iter().map(GpuProbe::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ray_sphere_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ray_sphere_output"),
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
            label: Some("prism_volumetric_ray_sphere_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_ray_sphere_bind_group"),
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
            label: Some("prism_volumetric_ray_sphere_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_ray_sphere_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_ray_sphere_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per probe, flattened to a 1-D dispatch.
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

/// Decodes one packed [`GpuResult`] into the public [`RaySphereResult`],
/// unpacking the flag bits into the reference's boolean / `Option` shape.
fn decode_result(raw: &GpuResult) -> RaySphereResult {
    let solved = raw.flags & FLAG_SOLVED != 0;
    let is_tangent = raw.flags & FLAG_TANGENT != 0;
    let intersects = raw.flags & FLAG_INTERSECTS != 0;
    let has_hit = raw.flags & FLAG_HIT != 0;
    let roots = RootPair {
        t_near: raw.t_near,
        t_far: raw.t_far,
    };
    let hit = if has_hit {
        Some(RaySphereHit {
            t: raw.hit_t,
            point: Vec3::new(raw.point[0], raw.point[1], raw.point[2]),
            normal: Vec3::new(raw.normal[0], raw.normal[1], raw.normal[2]),
        })
    } else {
        None
    };
    RaySphereResult {
        solved,
        roots,
        is_tangent,
        intersects,
        hit,
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
