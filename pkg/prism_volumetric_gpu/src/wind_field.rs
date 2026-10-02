//! `wgpu` compute twin of the art-directable deterministic wind model
//! ([`WindField`](prism_render_architecture::particle::wind_field::WindField),
//! particle design §8.2).
//!
//! The `CPU` golden
//! [`wind_field`](prism_render_architecture::particle::wind_field) owns a
//! storage-free parametric wind: a constant base wind along a unit `direction`,
//! a deterministic `gust_envelope` that swells and lulls over space and time, a
//! rational `height_attenuation` that weakens the wind with altitude, a
//! `sample_velocity` combining the two, and a relative-velocity
//! `drag_acceleration`. [`GpuWindField`] is the on-device twin: one thread per
//! query reproduces all four outputs, so a passing real-device parity test is
//! direct evidence the ported kernel hashes the same integer lattice and
//! evaluates the same closed-form algebra the reference does, not merely that
//! the shader compiles.
//!
//! # What is twinned
//!
//! Each [`WindFieldQuery`] packs a sample `position` and `time`, a standalone
//! `height`, the full [`WindField`](prism_render_architecture::particle::wind_field::WindField)
//! parameter set (already-normalized `direction`, `base_speed`,
//! `gust_amplitude`, `gust_frequency`, `height_falloff`, `seed`) and a drag
//! triple (`wind_vel`, `particle_vel`, `drag_coeff`). The kernel returns a
//! [`WindFieldResult`] holding the four public outputs
//! ([`gust_envelope`](prism_render_architecture::particle::wind_field::WindField::gust_envelope),
//! [`height_attenuation`](prism_render_architecture::particle::wind_field::WindField::height_attenuation),
//! [`sample_velocity`](prism_render_architecture::particle::wind_field::WindField::sample_velocity),
//! [`drag_acceleration`](prism_render_architecture::particle::wind_field::WindField::drag_acceleration))
//! plus the raw lattice hash of the gust floor cell, exposed so the parity test
//! can assert the integer path is bit-identical on its own.
//!
//! # Step-for-step parity
//!
//! The gust noise mirrors the reference exactly: the same integer lattice hash
//! (seed xor `FNV` basis, four `rotate`/multiply mixing folds over the three
//! spatial and one temporal cell index, a final `xorshift`-multiply
//! avalanche), the same `((hash >> 8) * 2^-24) * 2 - 1` cell value in
//! `[-1, 1)`, the same multiply-only smoothstep fade `t * t * (3 - 2 t)`, the
//! same four-dimensional (trilinear-then-temporal) blend, and the same
//! `GUST_SALT`-decorrelated seed. The gust envelope remaps the noise from
//! `[-1, 1]` to `[0, 1]` and clamps; the height attenuation is the rational
//! `1 / max(1 + h * falloff, MIN_FALLOFF_DENOM)` on `max(height, 0)`; the
//! sampled velocity is `direction * (base_speed + gust_amplitude * gust) *
//! atten`; and the drag is `(wind_vel - particle_vel) * drag_coeff`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — unsigned integer
//! multiply / xor / shift, `floor`, `clamp`, `max`, `+ - * /` and comparisons —
//! with no `sin`, `cos`, `exp`, `log`, `pow` or optional device feature, so it
//! runs unmodified on Metal, Vulkan and DX12. There is no `sqrt` here: the
//! sampled velocity scales an already-normalized host `direction`, so no length
//! is taken on device.
//!
//! # Correctness model
//!
//! The lattice hash is pure unsigned-integer work and `WGSL` unsigned integers
//! wrap on overflow exactly like Rust's `wrapping_mul` / `^` / `>>`, so the
//! `GPU` selects bit-identical cell values and the raw hash output matches the
//! reference to the bit. The only values that can diverge are the float fade /
//! `lerp` / rational blends, and only by a legal fused multiply-add contraction
//! of a few units in the last place. The parity test therefore asserts a tight
//! tolerance on the continuous outputs and exact equality on the hash word.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard art-directable analytic wind model (directional base
//! wind, value-noise gust envelope, rational height falloff, linear drag) plus
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::wind_field::{Vec3, WindField};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Threads per workgroup; one thread evaluates one wind-field query.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` wind-field kernel, embedded inline so the twin
/// ships as a single source file. Mirrors the `CPU` golden exactly; see the
/// module documentation for the algorithm.
const WIND_FIELD_WGSL: &str = r#"
// Wind-field twin: one thread per query reproduces the CPU golden
// `particle::wind_field`. It hashes the same 4D integer lattice as the gust
// noise, remaps it to the gust envelope, applies the rational height falloff,
// composes the sampled velocity along the already-normalized direction, and
// evaluates the relative-velocity drag. It uses only the portable core-WGSL
// subset (unsigned integer mix, floor, clamp, max and + - * /), takes no
// optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard art-directable analytic wind model; no Unreal Engine
// source or derived code.

// Scale turning a 24-bit hash mantissa into [0, 1); matches `INV_2POW24`.
const INV_2POW24: f32 = 1.0 / 16777216.0;
// Smallest allowed falloff denominator; matches `MIN_FALLOFF_DENOM`.
const MIN_FALLOFF_DENOM: f32 = 1.0e-3;
// Odd-integer salt folded into the gust seed; matches `GUST_SALT`.
const GUST_SALT: u32 = 0x9e3779b1u;
// FNV-1a offset basis xored into the seed; matches `hash_cell`.
const HASH_BASIS: u32 = 0x811c9dc5u;
// Multiplier xored into each folded input word; matches `mix`.
const MIX_MUL_A: u32 = 0x9e3779b1u;
// Post-rotate multiplier of the mixing fold; matches `mix`.
const MIX_MUL_B: u32 = 0x85ebca6bu;
// First avalanche multiplier; matches `finalize`.
const FIN_MUL_A: u32 = 0x7feb352du;
// Second avalanche multiplier; matches `finalize`.
const FIN_MUL_B: u32 = 0x846ca68bu;

// One wind-field query. 80-byte std430 stride of 20 scalar words, matching the
// host `GpuQuery`: position, time, standalone height, the WindField parameters
// and the drag triple.
struct Query {
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    time: f32,
    height: f32,
    dir_x: f32,
    dir_y: f32,
    dir_z: f32,
    base_speed: f32,
    gust_amplitude: f32,
    gust_frequency: f32,
    height_falloff: f32,
    seed: u32,
    wind_x: f32,
    wind_y: f32,
    wind_z: f32,
    part_x: f32,
    part_y: f32,
    part_z: f32,
    drag_coeff: f32,
}

// One wind-field result. 36-byte std430 stride of 9 scalar words, matching the
// host `GpuResult`: the gust envelope, the height attenuation, the sampled
// velocity triple, the drag acceleration triple and the raw gust-cell hash.
struct WindResult {
    gust: f32,
    atten: f32,
    vel_x: f32,
    vel_y: f32,
    vel_z: f32,
    drag_x: f32,
    drag_y: f32,
    drag_z: f32,
    hash: u32,
}

// Dispatch parameters. 16-byte std430/uniform block: the valid query count
// plus three pad words, matching the host `Params`.
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<WindResult>;

// One folding step of the hash: xor-in a multiplied input word, then rotate
// left 15 and multiply, mirroring the reference `mix`. WGSL unsigned shift and
// multiply wrap exactly like Rust `wrapping_mul` / `rotate_left`.
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

// Stateless integer hash of a 4D lattice cell and seed, mirroring `hash_cell`.
// Negative coordinates address the whole signed lattice through the two's
// complement reinterpret of the i32 cell index.
fn hash_cell(i: i32, j: i32, k: i32, l: i32, seed: u32) -> u32 {
    var h = seed ^ HASH_BASIS;
    h = hash_mix(h, bitcast<u32>(i));
    h = hash_mix(h, bitcast<u32>(j));
    h = hash_mix(h, bitcast<u32>(k));
    h = hash_mix(h, bitcast<u32>(l));
    return hash_finalize(h);
}

// The reproducible scalar value of a lattice cell in [-1, 1), mirroring
// `cell_value`. `(hash >> 8)` is a 24-bit integer, exactly representable in
// f32, so the conversion matches the reference `as f32`.
fn cell_value(i: i32, j: i32, k: i32, l: i32, seed: u32) -> f32 {
    let h = hash_cell(i, j, k, l, seed);
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

// Trilinearly interpolated value of one temporal slice `l`, mirroring
// `spatial_slice`.
fn spatial_slice(
    xi: i32,
    yi: i32,
    zi: i32,
    l: i32,
    seed: u32,
    u: f32,
    v: f32,
    w: f32,
) -> f32 {
    let c000 = cell_value(xi, yi, zi, l, seed);
    let c100 = cell_value(xi + 1, yi, zi, l, seed);
    let c010 = cell_value(xi, yi + 1, zi, l, seed);
    let c110 = cell_value(xi + 1, yi + 1, zi, l, seed);
    let c001 = cell_value(xi, yi, zi + 1, l, seed);
    let c101 = cell_value(xi + 1, yi, zi + 1, l, seed);
    let c011 = cell_value(xi, yi + 1, zi + 1, l, seed);
    let c111 = cell_value(xi + 1, yi + 1, zi + 1, l, seed);

    let x00 = lerp_v(c000, c100, u);
    let x10 = lerp_v(c010, c110, u);
    let x01 = lerp_v(c001, c101, u);
    let x11 = lerp_v(c011, c111, u);

    let y0 = lerp_v(x00, x10, v);
    let y1 = lerp_v(x01, x11, v);

    return lerp_v(y0, y1, w);
}

// Smoothstep-faded value noise over space and time in [-1, 1], mirroring
// `value_noise_4d`. The floor split uses `floor` then an exact integer cast,
// like the reference `floor_split`.
fn value_noise_4d(px: f32, py: f32, pz: f32, t: f32, seed: u32) -> f32 {
    let fx = floor(px);
    let fy = floor(py);
    let fz = floor(pz);
    let ft = floor(t);
    let xi = i32(fx);
    let yi = i32(fy);
    let zi = i32(fz);
    let ti = i32(ft);
    let xf = px - fx;
    let yf = py - fy;
    let zf = pz - fz;
    let tf = t - ft;

    let u = fade(xf);
    let v = fade(yf);
    let w = fade(zf);
    let s = fade(tf);

    let slice0 = spatial_slice(xi, yi, zi, ti, seed, u, v, w);
    let slice1 = spatial_slice(xi, yi, zi, ti + 1, seed, u, v, w);
    return lerp_v(slice0, slice1, s);
}

@compute @workgroup_size(64)
fn wind_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Gust envelope: value noise on the frequency-scaled position and time,
    // seeded by `seed ^ GUST_SALT`, remapped to [0, 1] and clamped.
    let gf = q.gust_frequency;
    let qx = q.pos_x * gf;
    let qy = q.pos_y * gf;
    let qz = q.pos_z * gf;
    let qt = q.time * gf;
    let gseed = q.seed ^ GUST_SALT;
    let noise = value_noise_4d(qx, qy, qz, qt, gseed);
    let unit = (noise + 1.0) * 0.5;
    let gust = clamp(unit, 0.0, 1.0);

    // Standalone height attenuation on `max(height, 0)`.
    let h_std = max(q.height, 0.0);
    let denom_std = max(1.0 + h_std * q.height_falloff, MIN_FALLOFF_DENOM);
    let atten = 1.0 / denom_std;

    // Sampled velocity: the attenuation uses the position's own altitude.
    let h_pos = max(q.pos_y, 0.0);
    let denom_pos = max(1.0 + h_pos * q.height_falloff, MIN_FALLOFF_DENOM);
    let atten_pos = 1.0 / denom_pos;
    let speed = q.base_speed + q.gust_amplitude * gust;
    let scale = speed * atten_pos;

    // Relative-velocity drag `(wind - particle) * drag`.
    let drag_x = (q.wind_x - q.part_x) * q.drag_coeff;
    let drag_y = (q.wind_y - q.part_y) * q.drag_coeff;
    let drag_z = (q.wind_z - q.part_z) * q.drag_coeff;

    // Raw hash of the gust floor cell, exposed for the integer-path parity.
    let hxi = i32(floor(qx));
    let hyi = i32(floor(qy));
    let hzi = i32(floor(qz));
    let hti = i32(floor(qt));

    var out: WindResult;
    out.gust = gust;
    out.atten = atten;
    out.vel_x = q.dir_x * scale;
    out.vel_y = q.dir_y * scale;
    out.vel_z = q.dir_z * scale;
    out.drag_x = drag_x;
    out.drag_y = drag_y;
    out.drag_z = drag_z;
    out.hash = hash_cell(hxi, hyi, hzi, hti, gseed);
    results[idx] = out;
}
"#;

/// One wind-field query: a sample `position` and `time`, a standalone `height`
/// for the attenuation probe, the wind parameters, and the drag inputs.
///
/// The `height` is kept separate from `position` so the twin can probe
/// [`height_attenuation`](prism_render_architecture::particle::wind_field::WindField::height_attenuation)
/// at an arbitrary altitude while
/// [`sample_velocity`](prism_render_architecture::particle::wind_field::WindField::sample_velocity)
/// still attenuates by the position's own `position.y`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindFieldQuery {
    /// Sample position fed to the gust envelope and sampled velocity.
    pub position: Vec3,
    /// Sample time fed to the gust envelope.
    pub time: f32,
    /// Standalone altitude for the height-attenuation probe.
    pub height: f32,
    /// Wind velocity for the drag probe.
    pub wind_vel: Vec3,
    /// Particle velocity for the drag probe.
    pub particle_vel: Vec3,
    /// Linear drag coefficient for the drag probe.
    pub drag_coeff: f32,
}

/// One wind-field result: the four public outputs plus the raw gust-cell hash.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindFieldResult {
    /// The gust envelope in `[0, 1]`, matching
    /// [`gust_envelope`](prism_render_architecture::particle::wind_field::WindField::gust_envelope).
    pub gust_envelope: f32,
    /// The rational height attenuation in `(0, 1]`, matching
    /// [`height_attenuation`](prism_render_architecture::particle::wind_field::WindField::height_attenuation).
    pub height_attenuation: f32,
    /// The sampled wind velocity, matching
    /// [`sample_velocity`](prism_render_architecture::particle::wind_field::WindField::sample_velocity).
    pub velocity: Vec3,
    /// The relative-velocity drag acceleration, matching
    /// [`drag_acceleration`](prism_render_architecture::particle::wind_field::WindField::drag_acceleration).
    pub drag: Vec3,
    /// The raw lattice hash of the gust floor cell, exposed so parity can
    /// assert the integer path is bit-identical to the reference `hash_cell`.
    pub gust_cell_hash: u32,
}

/// Uniform parameters for one wind-field dispatch. `repr(C)` `std430` layout
/// matching `Params` in [`WIND_FIELD_WGSL`]: the valid query count plus three
/// pad words — `16` bytes with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of valid queries.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One query as uploaded. `80`-byte `std430` stride of `20` scalar words,
/// matching `Query` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Position `x`.
    pos_x: f32,
    /// Position `y`.
    pos_y: f32,
    /// Position `z`.
    pos_z: f32,
    /// Sample time.
    time: f32,
    /// Standalone attenuation altitude.
    height: f32,
    /// Direction `x` (already normalized by the host).
    dir_x: f32,
    /// Direction `y`.
    dir_y: f32,
    /// Direction `z`.
    dir_z: f32,
    /// Calm-air wind speed.
    base_speed: f32,
    /// Peak gust speed.
    gust_amplitude: f32,
    /// Gust spatial/temporal frequency.
    gust_frequency: f32,
    /// Height falloff rate.
    height_falloff: f32,
    /// Gust realization seed.
    seed: u32,
    /// Drag wind velocity `x`.
    wind_x: f32,
    /// Drag wind velocity `y`.
    wind_y: f32,
    /// Drag wind velocity `z`.
    wind_z: f32,
    /// Drag particle velocity `x`.
    part_x: f32,
    /// Drag particle velocity `y`.
    part_y: f32,
    /// Drag particle velocity `z`.
    part_z: f32,
    /// Linear drag coefficient.
    drag_coeff: f32,
}

/// One result as read back. `36`-byte `std430` stride of `9` scalar words,
/// matching `WindResult` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Gust envelope.
    gust: f32,
    /// Height attenuation.
    atten: f32,
    /// Velocity `x`.
    vel_x: f32,
    /// Velocity `y`.
    vel_y: f32,
    /// Velocity `z`.
    vel_z: f32,
    /// Drag `x`.
    drag_x: f32,
    /// Drag `y`.
    drag_y: f32,
    /// Drag `z`.
    drag_z: f32,
    /// Raw gust-cell hash.
    hash: u32,
}

/// A compiled, reusable wind-field pipeline.
pub struct GpuWindField {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWindField {
    /// Compiles the wind-field kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWindField {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_wind_field"),
            source: ShaderSource::Wgsl(WIND_FIELD_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_wind_field_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_wind_field_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_wind_field_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("wind_eval"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWindField {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates `field` at every query in `queries`, returning one
    /// [`WindFieldResult`] per query in input order.
    ///
    /// For query `i`, `velocity` equals
    /// [`field.sample_velocity(position, time)`](prism_render_architecture::particle::wind_field::WindField::sample_velocity),
    /// `gust_envelope` equals
    /// [`field.gust_envelope(position, time)`](prism_render_architecture::particle::wind_field::WindField::gust_envelope),
    /// `height_attenuation` equals
    /// [`field.height_attenuation(height)`](prism_render_architecture::particle::wind_field::WindField::height_attenuation),
    /// and `drag` equals
    /// [`WindField::drag_acceleration(wind_vel, particle_vel, drag_coeff)`](prism_render_architecture::particle::wind_field::WindField::drag_acceleration),
    /// each to within the tolerance documented on this module; `gust_cell_hash`
    /// matches the reference lattice hash exactly. An empty `queries` slice
    /// yields an empty result — storage buffers cannot be zero-sized, so it is
    /// handled by an early return.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        field: &WindField,
        queries: &[WindFieldQuery],
    ) -> Vec<WindFieldResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                pos_x: q.position.x,
                pos_y: q.position.y,
                pos_z: q.position.z,
                time: q.time,
                height: q.height,
                dir_x: field.direction.x,
                dir_y: field.direction.y,
                dir_z: field.direction.z,
                base_speed: field.base_speed,
                gust_amplitude: field.gust_amplitude,
                gust_frequency: field.gust_frequency,
                height_falloff: field.height_falloff,
                seed: field.seed,
                wind_x: q.wind_vel.x,
                wind_y: q.wind_vel.y,
                wind_z: q.wind_vel.z,
                part_x: q.particle_vel.x,
                part_y: q.particle_vel.y,
                part_z: q.particle_vel.z,
                drag_coeff: q.drag_coeff,
            })
            .collect();

        let gpu_params = Params {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (queries.len() as u64) * (size_of::<GpuResult>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_wind_field_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_wind_field_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_wind_field_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_wind_field_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_wind_field_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_wind_field_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_wind_field_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, in workgroups of `WORKGROUP_SIZE`.
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
        let gpu_results = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_results.len(), queries.len());

        gpu_results
            .into_iter()
            .map(|r| WindFieldResult {
                gust_envelope: r.gust,
                height_attenuation: r.atten,
                velocity: Vec3::new(r.vel_x, r.vel_y, r.vel_z),
                drag: Vec3::new(r.drag_x, r.drag_y, r.drag_z),
                gust_cell_hash: r.hash,
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
