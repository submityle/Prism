//! `wgpu` compute twin of the stateless Worley / cellular (`Voronoi`) noise
//! fields of
//! [`worley`](prism_render_architecture::particle::worley).
//!
//! Worley noise scatters exactly one jittered feature point per unit integer
//! cell and, for a sample point, measures the distance to the nearest (`F1`)
//! and next-nearest (`F2`) feature point over the sample's own cell plus its
//! `26` neighbours (the `3x3x3`, `27`-candidate block). The derived fields are
//! the classic cellular-noise toolkit:
//!
//! - `F1`: the nearest-feature distance (pitting, bubbles, packed spheres).
//! - `F2`: the next-nearest distance (`F2 >= F1` always).
//! - `edges = F2 - F1`: the `Voronoi` edge / crack map.
//! - `inverted = 1 - F1`: the bright-at-feature "spot / blister" companion.
//!
//! [`GpuWorleyNoise`] evaluates all four for one query per thread, reproducing
//! the reference's exact closed form — the same integer avalanche hash, the
//! same `[0, 1)` jitter, the same `3x3x3` search and the same `Euclidean` /
//! `Manhattan` / `Chebyshev` metric — so a passing real-device parity test is
//! direct evidence the ported kernel computes the same field the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each thread reads one [`WorleyNoiseQuery`] — the sample position, the seed
//! and the metric selector — and writes one [`WorleyNoiseResult`] holding `F1`,
//! `F2`, `edges` and `inverted`. The kernel floors the position to a base cell,
//! visits the `27` surrounding cells, hashes each cell to a jittered feature
//! point through a pure `32`-bit avalanche, measures the chosen metric, and
//! keeps the smallest two distances.
//!
//! # What stays on the host
//!
//! The fractal `worley_fbm` octave stack and the per-cell colour id
//! `worley_cells` stay on the host; the device sees only the stateless,
//! fixed-width four-field evaluation, one query at a time, so a storage buffer
//! is never zero-sized.
//!
//! # Correctness model
//!
//! The hash and the jitter are pure integer / exact-multiply arithmetic, so
//! those agree bit for bit. Only the metric distance threads through `sqrt`
//! (`Euclidean`) or `abs` / `max` (`Manhattan` / `Chebyshev`), so a `GPU`
//! result may land a few units in the last place from the scalar reference. The
//! parity test asserts each field within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` with a relative floor of `1e-6`, and keeps its random
//! sweep clear of the knife-edge where two candidate distances tie (which could
//! otherwise flip the `F1` / `F2` selection between host and device).
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `floor`,
//! `abs`, `min`, `max`, `+ - * /`, unsigned bit arithmetic and `bitcast` — with
//! no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no
//! `round` and no `ceil`, and no `f64` / `u64` / `u16` / `i64` / `i16`. It runs
//! unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::worley`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
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

/// Metric selector for `Euclidean` (`L2`, round cells) distance.
pub const METRIC_EUCLIDEAN: u32 = 0;

/// Metric selector for `Manhattan` (`L1`, diamond cells) distance.
pub const METRIC_MANHATTAN: u32 = 1;

/// Metric selector for `Chebyshev` (`L∞`, box cells) distance.
pub const METRIC_CHEBYSHEV: u32 = 2;

/// The portable core-`WGSL` Worley-noise kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`worley_f1`](prism_render_architecture::particle::worley::worley_f1),
/// [`worley_f2`](prism_render_architecture::particle::worley::worley_f2),
/// [`worley_edges`](prism_render_architecture::particle::worley::worley_edges)
/// and
/// [`worley_inverted`](prism_render_architecture::particle::worley::worley_inverted)
/// closed forms; see the module documentation for the algorithm.
const WORLEY_NOISE_WGSL: &str = r#"
// Worley / cellular noise twin: one thread floors its query position to a base
// cell, searches the 3x3x3 block of 27 neighbouring cells, hashes each cell to
// a jittered feature point with a pure 32-bit avalanche, measures the selected
// metric and keeps the nearest two distances F1 <= F2, then writes F1, F2,
// edges = F2 - F1 and inverted = 1 - F1. Only sqrt / floor / abs / min / max,
// exact-wrapping integer arithmetic and bitcast appear; the fractal octave
// stack and the per-cell colour id stay on the host.
//
// Provenance: 孪生自本仓 `prism_render_architecture::particle::worley`；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Sample position in noise space.
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    // Deterministic hash seed.
    seed: u32,
    // Distance metric selector: 0 = Euclidean, 1 = Manhattan, 2 = Chebyshev.
    metric: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct NoiseResult {
    // Nearest feature-point distance.
    f1: f32,
    // Next-nearest feature-point distance (>= f1).
    f2: f32,
    // Voronoi edge / crack value, f2 - f1.
    edges: f32,
    // Inverted F1, 1 - f1.
    inverted: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<NoiseResult>;

// Odd-integer salts mixing the seed per jitter axis so the three jitter
// coordinates are statistically independent.
const SALT_X: u32 = 0x68E31DA4u;
const SALT_Y: u32 = 0xB5297A4Du;
const SALT_Z: u32 = 0x1B56C4E9u;

// Scale turning a 16-bit hash segment into [0, 1): 1 / 65536.
const INV_2POW16: f32 = 0.0000152587890625;

// One folding step of the lattice hash: xor-in a multiplied input word, then a
// 15-bit left rotate and a multiply spread the bits (rotate_left(15) is
// (h << 15) | (h >> 17)). Integer multiply wraps mod 2^32, matching the CPU
// wrapping_mul.
fn mix_hash(h: u32, v: u32) -> u32 {
    var hh: u32 = h ^ (v * 0x9E3779B1u);
    hh = ((hh << 15u) | (hh >> 17u)) * 0x85EBCA6Bu;
    return hh;
}

// Final avalanche applied once after all inputs are folded.
fn finalize_hash(h: u32) -> u32 {
    var hh: u32 = h;
    hh = hh ^ (hh >> 16u);
    hh = hh * 0x7FEB352Du;
    hh = hh ^ (hh >> 15u);
    hh = hh * 0x846CA68Bu;
    hh = hh ^ (hh >> 16u);
    return hh;
}

// Stateless integer hash of a lattice cell and seed. The signed cell indices
// are reinterpreted through their two's-complement bits (bitcast), matching the
// CPU `i as u32`.
fn hash_lattice(i: i32, j: i32, k: i32, seed: u32) -> u32 {
    var h: u32 = seed ^ 0x811C9DC5u;
    h = mix_hash(h, bitcast<u32>(i));
    h = mix_hash(h, bitcast<u32>(j));
    h = mix_hash(h, bitcast<u32>(k));
    return finalize_hash(h);
}

// Maps a 32-bit hash to [0, 1) using its low 16-bit segment.
fn unit01(h: u32) -> f32 {
    return f32(h & 0xFFFFu) * INV_2POW16;
}

// The jittered feature point of integer cell (i, j, k): the cell's corner
// offset by an independent per-axis jitter in [0, 1).
fn feature_point(i: i32, j: i32, k: i32, seed: u32) -> vec3<f32> {
    let jx = unit01(hash_lattice(i, j, k, seed ^ SALT_X));
    let jy = unit01(hash_lattice(i, j, k, seed ^ SALT_Y));
    let jz = unit01(hash_lattice(i, j, k, seed ^ SALT_Z));
    return vec3<f32>(f32(i) + jx, f32(j) + jy, f32(k) + jz);
}

// Distance from a to b under the selected metric. Euclidean sums the squared
// components in the same order as the CPU dot product before the single sqrt.
fn metric_distance(metric: u32, a: vec3<f32>, b: vec3<f32>) -> f32 {
    let d = a - b;
    if (metric == 1u) {
        return abs(d.x) + abs(d.y) + abs(d.z);
    }
    if (metric == 2u) {
        return max(abs(d.x), max(abs(d.y), abs(d.z)));
    }
    return sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let pos = vec3<f32>(q.pos_x, q.pos_y, q.pos_z);
    let seed = q.seed;
    let metric = q.metric;

    let bi = i32(floor(pos.x));
    let bj = i32(floor(pos.y));
    let bk = i32(floor(pos.z));

    // Positive infinity via bitcast of the IEEE-754 bit pattern (no literal
    // infinity in the core subset).
    let inf = bitcast<f32>(0x7F800000u);
    var f1: f32 = inf;
    var f2: f32 = inf;

    for (var di: i32 = -1; di <= 1; di = di + 1) {
        for (var dj: i32 = -1; dj <= 1; dj = dj + 1) {
            for (var dk: i32 = -1; dk <= 1; dk = dk + 1) {
                let fp = feature_point(bi + di, bj + dj, bk + dk, seed);
                let d = metric_distance(metric, pos, fp);
                if (d < f1) {
                    f2 = f1;
                    f1 = d;
                } else if (d < f2) {
                    f2 = d;
                }
            }
        }
    }

    var out: NoiseResult;
    out.f1 = f1;
    out.f2 = f2;
    out.edges = f2 - f1;
    out.inverted = 1.0 - f1;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in [`WORLEY_NOISE_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct:
/// the sample position, seed and metric selector, padded to a `32`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Sample position x.
    pos_x: f32,
    /// Sample position y.
    pos_y: f32,
    /// Sample position z.
    pos_z: f32,
    /// Deterministic hash seed.
    seed: u32,
    /// Distance metric selector.
    metric: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `NoiseResult`
/// struct: the four cellular fields in a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Nearest feature-point distance `F1`.
    f1: f32,
    /// Next-nearest feature-point distance `F2`.
    f2: f32,
    /// Edge / crack value `F2 - F1`.
    edges: f32,
    /// Inverted `F1`, `1 - F1`.
    inverted: f32,
}

/// One query for the Worley-noise twin: the sample position, the hash seed and
/// the distance-metric selector.
///
/// `metric` selects the distance metric: [`METRIC_EUCLIDEAN`] (`0`),
/// [`METRIC_MANHATTAN`] (`1`) or [`METRIC_CHEBYSHEV`] (`2`); any other value is
/// treated as `Euclidean` by the kernel, matching the reference's catch-all
/// `match` arm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorleyNoiseQuery {
    /// Sample position x.
    pub pos_x: f32,
    /// Sample position y.
    pub pos_y: f32,
    /// Sample position z.
    pub pos_z: f32,
    /// Deterministic hash seed.
    pub seed: u32,
    /// Distance metric selector.
    pub metric: u32,
}

impl WorleyNoiseQuery {
    /// Builds a query from the sample position, seed and metric selector.
    #[must_use]
    pub const fn new(
        pos_x: f32,
        pos_y: f32,
        pos_z: f32,
        seed: u32,
        metric: u32,
    ) -> WorleyNoiseQuery {
        WorleyNoiseQuery {
            pos_x,
            pos_y,
            pos_z,
            seed,
            metric,
        }
    }
}

/// One resolved query of the Worley-noise twin: the four cellular fields.
///
/// `f1` is
/// [`worley_f1`](prism_render_architecture::particle::worley::worley_f1); the
/// pair `(f1, f2)` is
/// [`worley_f2`](prism_render_architecture::particle::worley::worley_f2);
/// `edges` is
/// [`worley_edges`](prism_render_architecture::particle::worley::worley_edges)
/// (`f2 - f1`); and `inverted` is
/// [`worley_inverted`](prism_render_architecture::particle::worley::worley_inverted)
/// (`1 - f1`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorleyNoiseResult {
    /// Nearest feature-point distance `F1`.
    pub f1: f32,
    /// Next-nearest feature-point distance `F2` (`>= f1`).
    pub f2: f32,
    /// Edge / crack value `F2 - F1` (non-negative).
    pub edges: f32,
    /// Inverted `F1`, `1 - f1`.
    pub inverted: f32,
}

/// Encodes one [`WorleyNoiseQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &WorleyNoiseQuery) -> GpuQuery {
    GpuQuery {
        pos_x: q.pos_x,
        pos_y: q.pos_y,
        pos_z: q.pos_z,
        seed: q.seed,
        metric: q.metric,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`WorleyNoiseResult`].
fn decode_result(raw: &GpuResult) -> WorleyNoiseResult {
    WorleyNoiseResult {
        f1: raw.f1,
        f2: raw.f2,
        edges: raw.edges,
        inverted: raw.inverted,
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

/// A compiled, reusable Worley-noise compute pipeline, twinning the `CPU`
/// golden
/// [`worley_f1`](prism_render_architecture::particle::worley::worley_f1),
/// [`worley_f2`](prism_render_architecture::particle::worley::worley_f2),
/// [`worley_edges`](prism_render_architecture::particle::worley::worley_edges)
/// and
/// [`worley_inverted`](prism_render_architecture::particle::worley::worley_inverted).
pub struct GpuWorleyNoise {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWorleyNoise {
    /// Compiles the Worley-noise kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWorleyNoise {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_worley_noise"),
            source: ShaderSource::Wgsl(WORLEY_NOISE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_worley_noise_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_worley_noise_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_worley_noise_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWorleyNoise {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`WorleyNoiseResult`]
    /// per input, in order.
    ///
    /// The fields match the reference to within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WorleyNoiseQuery],
    ) -> Vec<WorleyNoiseResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_worley_noise_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_worley_noise_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_worley_noise_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_worley_noise_bind_group"),
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
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_worley_noise_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_worley_noise_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_worley_noise_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
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
