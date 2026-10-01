//! `wgpu` compute twin of the world-space signed-distance-field (`SDF`) soft
//! shadow sphere-trace
//! ([`sphere_trace_shadow`](prism_render_architecture::particle::distance_field_shadow::sphere_trace_shadow),
//! design sections 16-21, the `UE` / `Frostbite` distance-field soft-shadow
//! model).
//!
//! A distant emitter cannot afford a per-light shadow-map render for every
//! particle, so production engines bake the static scene into a signed-distance
//! field and *sphere trace* the shadow ray through it. Starting a short
//! distance off the receiver, the march repeatedly samples the nearest surface
//! distance `d` at the current point and advances by exactly `d` (the largest
//! step guaranteed not to tunnel through geometry). The penumbra falls out for
//! free: the ratio `softness_k * d / t` of the closest approach `d` to the
//! travelled distance `t` is an angular cone half-width, and the running minimum
//! of that ratio over the whole march is the soft-shadow visibility. This is the
//! classic Inigo-Quilez cone formulation that `UE`'s distance-field shadows and
//! `Frostbite`'s global distance field both build on.
//!
//! The `CPU` golden
//! [`particle::distance_field_shadow`](prism_render_architecture::particle::distance_field_shadow)
//! owns that march over an
//! [`SdfGrid`](prism_render_architecture::particle::distance_field_shadow::SdfGrid)
//! distance oracle; [`GpuDistanceFieldShadow`] is the on-device twin that runs
//! one thread per shadow ray and returns the same `0..=1` visibility (`1.0` =
//! fully lit, `0.0` = fully shadowed, soft grey in the penumbra). A passing
//! real-device parity test is therefore direct evidence the ported kernel
//! marches the same baked field in the same order the reference does, not merely
//! that its shader compiles.
//!
//! # What is twinned
//!
//! Only the undithered
//! [`sphere_trace_shadow`](prism_render_architecture::particle::distance_field_shadow::sphere_trace_shadow)
//! is reproduced. The reference's dithered wrapper offsets the start by an
//! integer-hash sub-step fraction; this twin deliberately omits that hash (the
//! crate ports only the real soft-shadow march, no hashing), so a caller that
//! wants dither can jitter `min_t` on the host and pass it through the shared
//! params exactly as the reference wrapper does.
//!
//! # Step-for-step parity
//!
//! The kernel mirrors the reference march exactly: the same
//! `normalize_or_zero` direction guard against [`CMP_EPS`], the same
//! `step_floor = max(surface_eps, CMP_EPS)`, the same start `t = max(min_t,
//! step_floor)`, the same `t >= max_t` and `max_steps` loop bounds, the same
//! `dist < surface_eps` hit test returning `0.0`, the same cone ratio
//! `softness_k * dist / t` folded into a running minimum, and the same
//! `t += max(dist, step_floor)` advance. The distance oracle mirrors
//! [`SdfGrid::sample`](prism_render_architecture::particle::distance_field_shadow::SdfGrid::sample):
//! the same world-to-grid map through the `[min, max]` box, the same degenerate
//! axis collapse against [`CMP_EPS`], the same border clamp, the same
//! row-major (`x`-fastest) texel index and the same eight-corner trilinear
//! blend.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `floor`, `sqrt`, `+ - * /` and unsigned integer compares — with no
//! `sin`, `cos`, `exp`, `log`, `pow` or optional device feature, so it runs
//! unmodified on Metal, Vulkan and DX12. The only non-integer primitive is
//! `sqrt` (the ray-direction length), exactly as in the reference, and every
//! reciprocal is guarded by a [`CMP_EPS`] clamp so no divide ever hits a
//! vanishing denominator.
//!
//! # Correctness model
//!
//! The march is a sequential sphere trace with no transcendental call and no
//! reorderable reduction, so `CPU` and `GPU` evaluate the same closed form in
//! the same order. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate (the `origin + unit * t` step, the trilinear
//! blends, the cone ratio), perturbing the low mantissa bits by a few units in
//! the last place. The parity test asserts an absolute tolerance tight enough to
//! catch a genuinely wrong port (a swapped trilinear corner, a dropped border
//! clamp, a missing cone running-minimum, a wrong step floor) yet loose enough
//! to admit that legal fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard Inigo-Quilez cone soft-shadow sphere trace over a baked
//! distance field plus `wgpu` compute dispatch; no Unreal Engine source or
//! derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::distance_field_shadow::DistanceFieldShadowParams;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Absolute tolerance mirroring the reference `CMP_EPS`.
///
/// The crate bans `==` / `!=` on floating point, so the degenerate-direction,
/// degenerate-axis and step-floor guards all compare a magnitude against this
/// epsilon instead, exactly as the `CPU` golden does. Shared with the kernel
/// `const CMP_EPS` so both sides collapse the same degenerate cases.
pub const CMP_EPS: f32 = 1.0e-6;

/// The portable core-`WGSL` sphere-trace soft-shadow kernel, embedded inline so
/// the twin ships as a single source file. Mirrors the `CPU` march and the
/// `SdfGrid` trilinear oracle exactly; see the module documentation for the
/// algorithm.
const DISTANCE_FIELD_SHADOW_WGSL: &str = r#"
// SDF soft-shadow twin: one thread per shadow ray sphere-traces the baked
// distance field and writes the Inigo-Quilez cone visibility in 0..=1 (1.0 =
// lit, 0.0 = shadowed). It mirrors the CPU golden
// `particle::distance_field_shadow::sphere_trace_shadow` and the `SdfGrid`
// trilinear reconstruction, uses only the portable core-WGSL subset
// (min/max/clamp/floor/sqrt and + - * /), and takes no optional feature, so it
// runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard Inigo-Quilez cone soft-shadow sphere trace; no Unreal
// Engine source or derived code.

struct Params {
    // Baked grid dimensions along x, y, z (x-fastest, row-major).
    nx: u32,
    ny: u32,
    nz: u32,
    // Shadow-ray count (one thread each).
    ray_count: u32,
    // World-space minimum corner of the grid's axis-aligned box.
    min_x: f32,
    min_y: f32,
    min_z: f32,
    // World-space maximum corner (x component here to keep scalars packed).
    max_x: f32,
    max_y: f32,
    max_z: f32,
    // Maximum sphere-trace iterations before the march gives up.
    max_steps: u32,
    // Cone-softness factor `k`.
    softness_k: f32,
    // March start distance (lifts the ray off the receiver).
    min_t: f32,
    // March stop distance (the light's reach).
    max_t: f32,
    // Surface hit threshold; also floors each step so the march progresses.
    surface_eps: f32,
    // Padding to a 64-byte, 16-byte-aligned uniform struct.
    pad0: u32,
}

// One shadow ray. 32-byte std430 stride: origin xyz plus a pad word, then
// direction xyz plus a pad word, matching the host `GpuRay`.
struct Ray {
    origin_x: f32,
    origin_y: f32,
    origin_z: f32,
    pad0: f32,
    dir_x: f32,
    dir_y: f32,
    dir_z: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> grid_data: array<f32>;
@group(0) @binding(2) var<storage, read> rays: array<Ray>;
@group(0) @binding(3) var<storage, read_write> results: array<f32>;

// Matches the reference `CMP_EPS`: degenerate-extent, degenerate-direction and
// step-floor guards all compare against this instead of an equality test.
const CMP_EPS: f32 = 1.0e-6;

// Linear interpolation `a + (b - a) * t`, matching the reference `lerp`.
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

// Euclidean length (the only sqrt in the kernel), matching `length3`.
fn length3(v: vec3<f32>) -> f32 {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

// Unit direction, or zero for a (near-)zero vector, matching
// `normalize_or_zero`.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len = length3(v);
    if (len < CMP_EPS) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    return v * (1.0 / len);
}

// Reads the raw baked distance at integer texel (x, y, z), row-major
// (x-fastest), matching `SdfGrid::at`.
fn grid_at(x: u32, y: u32, z: u32) -> f32 {
    let idx = x + y * params.nx + z * params.nx * params.ny;
    return grid_data[idx];
}

// The base texel index and fractional weight along one axis, matching the
// per-axis body of `SdfGrid::sample`: map into grid space through the box,
// collapse a degenerate (near-zero) extent to coordinate zero, border-clamp,
// floor to a cell, and clamp the index to the last texel. WGSL has no multiple
// return, so the pair is packed into a vec2 (x = index as f32, y = frac).
fn axis_cell(pa: f32, mn: f32, mx: f32, dim: u32) -> vec2<f32> {
    let last = dim - 1u;
    let last_f = f32(last);
    let extent = mx - mn;
    var coord = 0.0;
    if (extent < CMP_EPS) {
        coord = 0.0;
    } else {
        coord = ((pa - mn) / extent) * last_f;
    }
    let clamped = clamp(coord, 0.0, last_f);
    let floored = floor(clamped);
    var idx = u32(floored);
    if (idx > last) {
        idx = last;
    }
    return vec2<f32>(f32(idx), clamped - floored);
}

// Trilinearly samples the signed distance at world point `p`, matching
// `SdfGrid::sample` corner for corner.
fn sample_sdf(p: vec3<f32>) -> f32 {
    let cx = axis_cell(p.x, params.min_x, params.max_x, params.nx);
    let cy = axis_cell(p.y, params.min_y, params.max_y, params.ny);
    let cz = axis_cell(p.z, params.min_z, params.max_z, params.nz);

    let x0 = u32(cx.x);
    let y0 = u32(cy.x);
    let z0 = u32(cz.x);
    let x1 = min(x0 + 1u, params.nx - 1u);
    let y1 = min(y0 + 1u, params.ny - 1u);
    let z1 = min(z0 + 1u, params.nz - 1u);

    let fx = cx.y;
    let fy = cy.y;
    let fz = cz.y;

    let c00 = lerp(grid_at(x0, y0, z0), grid_at(x1, y0, z0), fx);
    let c10 = lerp(grid_at(x0, y1, z0), grid_at(x1, y1, z0), fx);
    let c01 = lerp(grid_at(x0, y0, z1), grid_at(x1, y0, z1), fx);
    let c11 = lerp(grid_at(x0, y1, z1), grid_at(x1, y1, z1), fx);

    let c0 = lerp(c00, c10, fy);
    let c1 = lerp(c01, c11, fy);
    return lerp(c0, c1, fz);
}

@compute @workgroup_size(64)
fn trace_shadows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ray_index = gid.x;
    if (ray_index >= params.ray_count) {
        return;
    }
    let ray = rays[ray_index];
    let origin = vec3<f32>(ray.origin_x, ray.origin_y, ray.origin_z);
    let dir = vec3<f32>(ray.dir_x, ray.dir_y, ray.dir_z);

    let unit = normalize_or_zero(dir);
    // A degenerate direction leaves the receiver fully lit.
    if (length3(unit) < CMP_EPS) {
        results[ray_index] = 1.0;
        return;
    }

    let step_floor = max(params.surface_eps, CMP_EPS);
    var visibility = 1.0;
    var t = max(params.min_t, step_floor);
    for (var step = 0u; step < params.max_steps; step = step + 1u) {
        if (t >= params.max_t) {
            break;
        }
        let point = origin + unit * t;
        let dist = sample_sdf(point);
        if (dist < params.surface_eps) {
            // Reached geometry: fully shadowed.
            visibility = 0.0;
            break;
        }
        let cone = params.softness_k * dist / t;
        visibility = min(visibility, cone);
        t = t + max(dist, step_floor);
    }
    results[ray_index] = clamp(visibility, 0.0, 1.0);
}
"#;

/// A baked scalar signed-distance field uploaded to the device, the twin of the
/// reference
/// [`SdfGrid`](prism_render_architecture::particle::distance_field_shadow::SdfGrid).
///
/// The grid is stored row-major (`x`-fastest) over the axis-aligned `[min, max]`
/// box: texel `(i, j, k)` lives at `data[i + j * nx + k * nx * ny]`. The twin
/// takes the raw pieces (rather than a borrowed `SdfGrid`, whose fields are
/// private) so a test can build one `SdfGrid` for the `CPU` reference and pass
/// the identical dimensions, bounds and samples here for the `GPU` run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuSdfGrid<'data> {
    /// Grid dimensions `[nx, ny, nz]`; every component must be non-zero.
    pub dims: [usize; 3],
    /// World-space minimum corner of the grid's axis-aligned box.
    pub min: [f32; 3],
    /// World-space maximum corner of the grid's axis-aligned box.
    pub max: [f32; 3],
    /// Row-major (`x`-fastest) distance samples; `data.len()` should equal
    /// `nx * ny * nz`. A short slice reads the missing tail as zero and a long
    /// slice ignores the extra tail, so the upload is always in bounds.
    pub data: &'data [f32],
}

/// One shadow ray: a world-space `origin` on the receiver and a `dir` toward the
/// emitter (normalized on device, so any non-zero length works).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfShadowRay {
    /// World-space ray origin (the receiver point, before the `min_t` lift-off).
    pub origin: [f32; 3],
    /// Ray direction toward the light; normalized on device.
    pub dir: [f32; 3],
}

/// Uniform parameters for one sphere-trace dispatch. `repr(C)` `std430` layout
/// matching `Params` in [`DISTANCE_FIELD_SHADOW_WGSL`]: four index words, six
/// box-bound `f32` words, the `max_steps` word, the four march `f32` words and
/// one pad word — `64` bytes with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    nx: u32,
    ny: u32,
    nz: u32,
    ray_count: u32,
    min_x: f32,
    min_y: f32,
    min_z: f32,
    max_x: f32,
    max_y: f32,
    max_z: f32,
    max_steps: u32,
    softness_k: f32,
    min_t: f32,
    max_t: f32,
    surface_eps: f32,
    pad0: u32,
}

/// One shadow ray as uploaded. `32`-byte `std430` stride matching `Ray` in the
/// shader: the origin triple plus a pad word, then the direction triple plus a
/// pad word.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuRay {
    origin_x: f32,
    origin_y: f32,
    origin_z: f32,
    pad0: f32,
    dir_x: f32,
    dir_y: f32,
    dir_z: f32,
    pad1: f32,
}

/// A compiled, reusable `SDF` sphere-trace soft-shadow pipeline.
pub struct GpuDistanceFieldShadow {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuDistanceFieldShadow {
    /// Compiles the sphere-trace soft-shadow kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDistanceFieldShadow {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_distance_field_shadow"),
            source: ShaderSource::Wgsl(DISTANCE_FIELD_SHADOW_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_distance_field_shadow_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_distance_field_shadow_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_distance_field_shadow_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("trace_shadows"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuDistanceFieldShadow {
            module,
            layout,
            pipeline,
        }
    }

    /// Sphere-traces every ray in `rays` through `grid` under the shared
    /// `params`, returning one `0..=1` visibility per ray in input order.
    ///
    /// The returned visibility for ray `r` equals
    /// [`sphere_trace_shadow`](prism_render_architecture::particle::distance_field_shadow::sphere_trace_shadow)`(grid.sample, r.origin, r.dir, params)`
    /// to within the tolerance documented on this module, where `grid.sample` is
    /// the trilinear reconstruction of the same baked field. An empty `rays`
    /// slice yields an empty result — storage buffers cannot be zero-sized, so
    /// it is handled by an early return.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        grid: &GpuSdfGrid<'_>,
        params: &DistanceFieldShadowParams,
        rays: &[SdfShadowRay],
    ) -> Vec<f32> {
        if rays.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let [nx, ny, nz] = grid.dims;

        // A dense, cell-count-length distance buffer. Missing samples stay zero
        // and extra samples are ignored, so the upload is always in bounds even
        // for a malformed slice; a correctly sized field copies verbatim.
        let cell_count = nx.saturating_mul(ny).saturating_mul(nz).max(1);
        let mut grid_samples = vec![0.0f32; cell_count];
        for (slot, &value) in grid_samples.iter_mut().zip(grid.data.iter()) {
            *slot = value;
        }

        let gpu_rays: Vec<GpuRay> = rays
            .iter()
            .map(|r| GpuRay {
                origin_x: r.origin[0],
                origin_y: r.origin[1],
                origin_z: r.origin[2],
                pad0: 0.0,
                dir_x: r.dir[0],
                dir_y: r.dir[1],
                dir_z: r.dir[2],
                pad1: 0.0,
            })
            .collect();

        let gpu_params = Params {
            nx: nx as u32,
            ny: ny as u32,
            nz: nz as u32,
            ray_count: rays.len() as u32,
            min_x: grid.min[0],
            min_y: grid.min[1],
            min_z: grid.min[2],
            max_x: grid.max[0],
            max_y: grid.max[1],
            max_z: grid.max[2],
            max_steps: params.max_steps,
            softness_k: params.softness_k,
            min_t: params.min_t,
            max_t: params.max_t,
            surface_eps: params.surface_eps,
            pad0: 0,
        };

        let out_bytes = (rays.len() as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_distance_field_shadow_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let grid_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_distance_field_shadow_grid"),
            contents: bytemuck::cast_slice(&grid_samples),
            usage: BufferUsages::STORAGE,
        });
        let rays_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_distance_field_shadow_rays"),
            contents: bytemuck::cast_slice(&gpu_rays),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_distance_field_shadow_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_distance_field_shadow_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_distance_field_shadow_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: grid_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: rays_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_distance_field_shadow_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_distance_field_shadow_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per ray, in workgroups of 64 (the kernel's size).
            let groups = (rays.len() as u32).div_ceil(64);
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
        let visibilities = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(visibilities.len(), rays.len());
        visibilities
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
