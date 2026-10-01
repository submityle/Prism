//! `wgpu` compute twin of the analytic divergence-free curl-noise velocity
//! field
//! ([`CurlNoiseField`](prism_render_architecture::particle::curl_noise::CurlNoiseField),
//! design sections 8 and 10).
//!
//! Particle advection in the production `VFX` stacks (Unreal `Niagara`, Unity
//! `VFX Graph`, `Houdini` flow noise) is driven by a storage-free, analytically
//! incompressible turbulence field: a smooth vector potential `Ψ` is built from
//! three decorrelated hash-seeded value-noise channels and the velocity is its
//! curl `∇ × Ψ`, which is divergence-free by construction. The `CPU` golden
//! [`particle::curl_noise`](prism_render_architecture::particle::curl_noise)
//! owns that math; [`GpuCurlNoise`] is the on-device twin that runs one thread
//! per query point and returns, for every point, both the advection velocity
//! and the re-estimated divergence. A passing real-device parity test is
//! therefore direct evidence the ported kernel hashes the same lattice and
//! differentiates the same potential the reference does, not merely that its
//! shader compiles.
//!
//! # Step-for-step parity
//!
//! The kernel mirrors the reference exactly: the same integer lattice hash
//! (seed xor `FNV` basis, three `rotate`/multiply mixing folds, a final
//! `xorshift`-multiply avalanche), the same `((hash >> 8) * 2^-24) * 2 - 1`
//! cell value in `[-1, 1)`, the same multiply-only smoothstep fade
//! `t * t * (3 - 2 t)`, the same trilinear lattice blend, the same
//! frequency-scaled three-channel potential, and the same central-difference
//! curl of half-step [`CURL_EPS`](prism_render_architecture::particle::curl_noise::CURL_EPS)
//! scaled by the field amplitude. The divergence re-estimate reuses the same
//! stencil on the velocity, so it cancels to `f32` rounding exactly as on the
//! `CPU`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — unsigned integer
//! multiply / xor / shift, `floor`, `+ - * /` and comparisons — with no `sin`,
//! `cos`, `exp`, `log`, `pow` or optional device feature, so it runs unmodified
//! on Metal, Vulkan and DX12. There is no `sqrt` at all here: the curl is pure
//! finite-difference arithmetic.
//!
//! # Correctness model
//!
//! The lattice hash is pure unsigned-integer work and `WGSL` unsigned integers
//! wrap on overflow exactly like Rust's `wrapping_mul` / `^` / `>>`, so the
//! `GPU` selects bit-identical cell values to the reference. The only values
//! that can diverge are the float fade / `lerp` / central-difference blends,
//! and only by a legal multiply-add contraction of a few units in the last
//! place. The parity test therefore asserts a tight absolute tolerance rather
//! than exact equality.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard analytic curl-noise advection (Bridson-style
//! divergence-free noise) plus `wgpu` compute dispatch; no Unreal Engine source
//! or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::curl_noise::{CurlNoiseField, Vec3};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// The portable core-`WGSL` curl-noise kernel, embedded inline so the twin
/// ships as a single source file. Mirrors the `CPU` golden exactly; see the
/// module documentation for the algorithm.
const CURL_NOISE_WGSL: &str = r#"
// Curl-noise twin: one thread per query point hashes the same integer lattice
// as the CPU golden `particle::curl_noise`, builds the three-channel scalar
// potential by trilinear value noise, and returns the central-difference curl
// (the divergence-free advection velocity) plus the re-estimated divergence.
// It uses only the portable core-WGSL subset (unsigned integer mix, floor and
// + - * /), takes no optional feature, so it runs unmodified on Metal, Vulkan
// and DX12.
//
// Provenance: standard analytic curl-noise advection; no Unreal Engine source
// or derived code.

// Scale turning a 24-bit hash mantissa into [0, 1); matches `INV_2POW24`.
const INV_2POW24: f32 = 1.0 / 16777216.0;
// Central-difference half-step; matches the reference `CURL_EPS`.
const CURL_EPS: f32 = 1.0e-2;
// FNV-1a offset basis xored into the seed; matches the reference `hash_cell`.
const HASH_BASIS: u32 = 0x811c9dc5u;
// Multiplier xored into each folded input word; matches `mix`.
const MIX_MUL_A: u32 = 0x9e3779b1u;
// Post-rotate multiplier of the mixing fold; matches `mix`.
const MIX_MUL_B: u32 = 0x85ebca6bu;
// First avalanche multiplier; matches `finalize`.
const FIN_MUL_A: u32 = 0x7feb352du;
// Second avalanche multiplier; matches `finalize`.
const FIN_MUL_B: u32 = 0x846ca68bu;
// Odd-integer salts selecting the three decorrelated potential channels;
// match `POT_SEED_X` / `POT_SEED_Y` / `POT_SEED_Z`.
const POT_SEED_X: u32 = 0x68e31da4u;
const POT_SEED_Y: u32 = 0xb5439c13u;
const POT_SEED_Z: u32 = 0x2545f491u;

struct Params {
    // Spatial frequency scaling the query point before the potential is sampled.
    frequency: f32,
    // Peak velocity scale multiplying the raw curl.
    amplitude: f32,
    // Seed selecting the pseudo-random realization.
    seed: u32,
    // Number of valid query points in `points` / outputs in `results`.
    count: u32,
}

// One query point. 16-byte std430 stride: the xyz components plus one pad word,
// matching the host `GpuPoint`.
struct Point {
    x: f32,
    y: f32,
    z: f32,
    pad: f32,
}

// One query result. 16-byte std430 stride: the advection velocity triple plus
// the re-estimated divergence, matching the host `GpuSample`.
struct CurlResult {
    vx: f32,
    vy: f32,
    vz: f32,
    divergence: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> points: array<Point>;
@group(0) @binding(2) var<storage, read_write> results: array<CurlResult>;

// One folding step of the hash: xor-in a multiplied input word, then rotate
// left 15 and multiply, mirroring the reference `mix`. WGSL unsigned shift and
// multiply wrap exactly like Rust `wrapping_mul` / rotate_left.
fn hash_mix(h_in: u32, word: u32) -> u32 {
    var h = h_in ^ (word * MIX_MUL_A);
    let rotated = (h << 15u) | (h >> 17u);
    return rotated * MIX_MUL_B;
}

// Final avalanche applied once after all words are folded, mirroring the
// reference `finalize`.
fn hash_finalize(h_in: u32) -> u32 {
    var h = h_in;
    h = h ^ (h >> 16u);
    h = h * FIN_MUL_A;
    h = h ^ (h >> 15u);
    h = h * FIN_MUL_B;
    h = h ^ (h >> 16u);
    return h;
}

// Stateless integer hash of a lattice cell and seed, mirroring `hash_cell`.
// Negative coordinates address the whole signed lattice through the two's
// complement reinterpret of the i32 cell index.
fn hash_cell(i: i32, j: i32, k: i32, seed: u32) -> u32 {
    var h = seed ^ HASH_BASIS;
    h = hash_mix(h, bitcast<u32>(i));
    h = hash_mix(h, bitcast<u32>(j));
    h = hash_mix(h, bitcast<u32>(k));
    return hash_finalize(h);
}

// The reproducible scalar value of a lattice cell in [-1, 1), mirroring
// `cell_value`. `(hash >> 8)` is a 24-bit integer, exactly representable in
// f32, so the conversion matches the reference `as f32`.
fn cell_value(i: i32, j: i32, k: i32, seed: u32) -> f32 {
    let h = hash_cell(i, j, k, seed);
    let unit = f32(h >> 8u) * INV_2POW24;
    return unit * 2.0 - 1.0;
}

// The multiply-only smoothstep fade `t * t * (3 - 2 t)`, mirroring `fade`.
fn fade(t: f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
}

// Linear interpolation `a + (b - a) * t`, mirroring `lerp`.
fn lerp_v(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

// Trilinearly interpolated, smoothstep-faded value noise in [-1, 1], mirroring
// `value_noise`. The floor split uses `floor` then an exact integer cast, like
// the reference `floor_split`.
fn value_noise(p: vec3<f32>, seed: u32) -> f32 {
    let fx = floor(p.x);
    let fy = floor(p.y);
    let fz = floor(p.z);
    let xi = i32(fx);
    let yi = i32(fy);
    let zi = i32(fz);
    let xf = p.x - fx;
    let yf = p.y - fy;
    let zf = p.z - fz;

    let u = fade(xf);
    let v = fade(yf);
    let w = fade(zf);

    let c000 = cell_value(xi, yi, zi, seed);
    let c100 = cell_value(xi + 1, yi, zi, seed);
    let c010 = cell_value(xi, yi + 1, zi, seed);
    let c110 = cell_value(xi + 1, yi + 1, zi, seed);
    let c001 = cell_value(xi, yi, zi + 1, seed);
    let c101 = cell_value(xi + 1, yi, zi + 1, seed);
    let c011 = cell_value(xi, yi + 1, zi + 1, seed);
    let c111 = cell_value(xi + 1, yi + 1, zi + 1, seed);

    let x00 = lerp_v(c000, c100, u);
    let x10 = lerp_v(c010, c110, u);
    let x01 = lerp_v(c001, c101, u);
    let x11 = lerp_v(c011, c111, u);

    let y0 = lerp_v(x00, x10, v);
    let y1 = lerp_v(x01, x11, v);

    return lerp_v(y0, y1, w);
}

// The vector potential `Ψ(p)` whose curl is the velocity, mirroring
// `potential`: three decorrelated value-noise channels at the frequency-scaled
// position.
fn potential(p: vec3<f32>) -> vec3<f32> {
    let q = p * params.frequency;
    return vec3<f32>(
        value_noise(q, params.seed ^ POT_SEED_X),
        value_noise(q, params.seed ^ POT_SEED_Y),
        value_noise(q, params.seed ^ POT_SEED_Z)
    );
}

// The divergence-free advection velocity: the central-difference curl of the
// potential scaled by the amplitude, mirroring `sample_velocity`.
fn curl_velocity(p: vec3<f32>) -> vec3<f32> {
    let inv = 1.0 / (2.0 * CURL_EPS);

    let px_p = potential(p + vec3<f32>(CURL_EPS, 0.0, 0.0));
    let px_m = potential(p - vec3<f32>(CURL_EPS, 0.0, 0.0));
    let py_p = potential(p + vec3<f32>(0.0, CURL_EPS, 0.0));
    let py_m = potential(p - vec3<f32>(0.0, CURL_EPS, 0.0));
    let pz_p = potential(p + vec3<f32>(0.0, 0.0, CURL_EPS));
    let pz_m = potential(p - vec3<f32>(0.0, 0.0, CURL_EPS));

    let dpz_dy = (py_p.z - py_m.z) * inv;
    let dpy_dz = (pz_p.y - pz_m.y) * inv;
    let dpx_dz = (pz_p.x - pz_m.x) * inv;
    let dpz_dx = (px_p.z - px_m.z) * inv;
    let dpy_dx = (px_p.y - px_m.y) * inv;
    let dpx_dy = (py_p.x - py_m.x) * inv;

    let vx = dpz_dy - dpy_dz;
    let vy = dpx_dz - dpz_dx;
    let vz = dpy_dx - dpx_dy;
    return vec3<f32>(vx, vy, vz) * params.amplitude;
}

// Re-estimates the divergence of the velocity with the same central-difference
// stencil, mirroring `divergence`. Discrete curl / divergence stencils commute,
// so for a smooth potential this cancels to f32 rounding.
fn divergence_at(p: vec3<f32>) -> f32 {
    let inv = 1.0 / (2.0 * CURL_EPS);

    let dux_dx = (curl_velocity(p + vec3<f32>(CURL_EPS, 0.0, 0.0)).x
        - curl_velocity(p - vec3<f32>(CURL_EPS, 0.0, 0.0)).x) * inv;
    let duy_dy = (curl_velocity(p + vec3<f32>(0.0, CURL_EPS, 0.0)).y
        - curl_velocity(p - vec3<f32>(0.0, CURL_EPS, 0.0)).y) * inv;
    let duz_dz = (curl_velocity(p + vec3<f32>(0.0, 0.0, CURL_EPS)).z
        - curl_velocity(p - vec3<f32>(0.0, 0.0, CURL_EPS)).z) * inv;

    return dux_dx + duy_dy + duz_dz;
}

@compute @workgroup_size(64)
fn curl_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let pt = points[idx];
    let p = vec3<f32>(pt.x, pt.y, pt.z);
    let vel = curl_velocity(p);

    var out: CurlResult;
    out.vx = vel.x;
    out.vy = vel.y;
    out.vz = vel.z;
    out.divergence = divergence_at(p);
    results[idx] = out;
}
"#;

/// One query point's curl-noise outputs: the divergence-free advection velocity
/// and the central-difference re-estimate of the field divergence at the point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurlNoiseSample {
    /// Advection velocity `∇ × Ψ` scaled by the field amplitude.
    pub velocity: Vec3,
    /// Divergence `∇ · u` re-estimated with the same central-difference
    /// stencil; cancels to `f32` rounding for the incompressible field.
    pub divergence: f32,
}

/// Uniform parameters for one curl-noise dispatch. `repr(C)` `std430` layout
/// matching `Params` in [`CURL_NOISE_WGSL`]: two `f32` words then two `u32`
/// words — `16` bytes with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Spatial frequency.
    frequency: f32,
    /// Peak velocity scale.
    amplitude: f32,
    /// Seed selecting the pseudo-random realization.
    seed: u32,
    /// Number of query points.
    count: u32,
}

/// One query point as uploaded. `16`-byte `std430` stride: the `xyz` components
/// plus one pad word, matching `Point` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuPoint {
    /// The `x` component.
    x: f32,
    /// The `y` component.
    y: f32,
    /// The `z` component.
    z: f32,
    /// Padding word.
    pad: f32,
}

/// One query result as read back. `16`-byte `std430` stride matching
/// `CurlResult` in the shader: the velocity triple plus the divergence word.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSample {
    /// Velocity `x`.
    vx: f32,
    /// Velocity `y`.
    vy: f32,
    /// Velocity `z`.
    vz: f32,
    /// Re-estimated divergence.
    divergence: f32,
}

/// A compiled, reusable curl-noise pipeline.
pub struct GpuCurlNoise {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCurlNoise {
    /// Compiles the curl-noise kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCurlNoise {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_curl_noise"),
            source: ShaderSource::Wgsl(CURL_NOISE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_curl_noise_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_curl_noise_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_curl_noise_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("curl_eval"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCurlNoise {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the curl-noise field `field` at every point in `points`,
    /// returning one [`CurlNoiseSample`] per point in input order.
    ///
    /// The returned velocity for point `p` equals
    /// [`CurlNoiseField::sample_velocity`](prism_render_architecture::particle::curl_noise::CurlNoiseField::sample_velocity)`(p)`
    /// and the divergence equals
    /// [`CurlNoiseField::divergence`](prism_render_architecture::particle::curl_noise::CurlNoiseField::divergence)`(p)`
    /// to within the tolerance documented on this module. An empty `points`
    /// slice yields an empty result — storage buffers cannot be zero-sized, so
    /// it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, field: &CurlNoiseField, points: &[Vec3]) -> Vec<CurlNoiseSample> {
        if points.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_points: Vec<GpuPoint> = points
            .iter()
            .map(|p| GpuPoint {
                x: p.x,
                y: p.y,
                z: p.z,
                pad: 0.0,
            })
            .collect();

        let gpu_params = Params {
            frequency: field.frequency,
            amplitude: field.amplitude,
            seed: field.seed,
            count: points.len() as u32,
        };

        let out_bytes = (points.len() as u64) * (size_of::<GpuSample>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_curl_noise_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let points_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_curl_noise_points"),
            contents: bytemuck::cast_slice(&gpu_points),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_curl_noise_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_curl_noise_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_curl_noise_bind_group"),
            layout: &self.layout,
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
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_curl_noise_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_curl_noise_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per point, in workgroups of 64 (the kernel's size).
            let groups = (points.len() as u32).div_ceil(64);
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
        let gpu_results = bytemuck::cast_slice::<u8, GpuSample>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_results.len(), points.len());

        gpu_results
            .into_iter()
            .map(|r| CurlNoiseSample {
                velocity: Vec3::new(r.vx, r.vy, r.vz),
                divergence: r.divergence,
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
