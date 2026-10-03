//! `wgpu` compute twin of the film-grain luma and hashed-grain primitives
//! ([`luma709`](prism_render_architecture::particle::film_grain::luma709) and
//! [`grain_hash01`](prism_render_architecture::particle::film_grain::grain_hash01)).
//!
//! The animated film-grain post pass needs two device-side building blocks per
//! pixel: the `Rec. 709` relative luminance of the base color (which modulates
//! how visible the grain is) and a per-pixel, per-frame hashed value in
//! `[0, 1)` that seeds the speckle. Both are pure integer/rational kernels with
//! no transcendental math, so they port to the device unchanged, and a passing
//! real-device parity run is direct evidence the ported kernel folds the same
//! arithmetic the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! One thread serves one query. It reproduces both golden routines exactly:
//! `luma709` as the clamped `Rec. 709` dot product `0.2126*r + 0.7152*g +
//! 0.0722*b`, and `grain_hash01` as the odd-multiplier decorrelation of the
//! pixel coordinates and frame index, the `xor-shift` avalanche finalizer, and
//! the top-`24`-bit normalization into `[0, 1)`. The `WGSL` `u32` multiply is
//! itself wrapping, so each `*` corresponds to the reference `wrapping_mul`.
//!
//! # What stays on the host
//!
//! The surrounding grain pipeline — sized value noise, the luminance response
//! curve, and the blend compositing — are upstream responsibilities; this twin
//! only reconstructs the luma and the base hashed grain from the supplied color
//! and pixel/frame indices.
//!
//! # Correctness model
//!
//! The luma is a continuous `f32` built from multiply-adds and one clamp,
//! compared with `abs <= 1e-6 || rel <= 1e-5`. The grain is derived from an
//! integer hash and an exact power-of-two divisor, so it should be essentially
//! exact; it is compared with `abs <= 1e-5 || rel <= 1e-4` to absorb any
//! last-bit widening difference.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — multiply-add, a
//! clamp, `u32` wrapping multiplies, logical shifts, and `xor` — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `smoothstep`,
//! no `round`, and no `cbrt`, and no `u64`/`i64`/`u16`/`i16`/`f64`. Each thread
//! performs a bounded, branch-free sequence, so the kernel provably terminates.
//! No optional device feature is required, so it runs unmodified across
//! backends.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::film_grain`；无第三方引擎源码或衍生代码。
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

/// Number of threads per workgroup. The reconstruction is a
/// single-thread-per-element kernel, so a one-dimensional dispatch of this
/// width keeps every lane busy on real hardware.
const WORKGROUP_SIZE: u32 = 64;

/// The inlined `WGSL` twin of the film-grain luma and hashed-grain kernels: one
/// thread per query, computing the clamped `Rec. 709` luma and the normalized
/// avalanche-hash grain.
const FILM_GRAIN_LUMA_WGSL: &str = r#"
// Twin of particle::film_grain::{luma709, grain_hash01}.
// One thread per query:
//   luma  = clamp(0.2126*r + 0.7152*g + 0.0722*b, 0, 1)
//   a     = px    * 0x9E3779B1
//   b     = py    * 0x85EBCA77
//   c     = frame * 0xC2B2AE3D
//   h     = hash_u32(a ^ b ^ c)
//   grain = f32(h >> 8) / 16777216.0
// All grain work is u32 (wrapping multiply + shift + xor); luma is f32.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::film_grain；无第三方引擎源码或衍生代码。

// Rec. 709 luma weights.
const LUMA_R: f32 = 0.2126;
const LUMA_G: f32 = 0.7152;
const LUMA_B: f32 = 0.0722;

// Odd decorrelation multipliers for pixel x/y and frame index.
const ODD_X: u32 = 0x9E3779B1u;
const ODD_Y: u32 = 0x85EBCA77u;
const ODD_FRAME: u32 = 0xC2B2AE3Du;

// Avalanche-finalizer odd multiply constants.
const MIX_A: u32 = 0x7FEB352Du;
const MIX_B: u32 = 0x846CA68Bu;

// 2^24, the exact f32 normalization divisor for the top 24 hash bits.
const NORM_24: f32 = 16777216.0;

struct Params {
    // Number of queries in the storage arrays; threads past this stop.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Base color red channel.
    color_r: f32,
    // Base color green channel.
    color_g: f32,
    // Base color blue channel.
    color_b: f32,
    // Pixel x coordinate.
    px: u32,
    // Pixel y coordinate.
    py: u32,
    // Frame index.
    frame: u32,
}

struct Shaded {
    // Clamped Rec. 709 luma.
    luma: f32,
    // Hashed grain value in [0, 1).
    grain: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Shaded>;

// Pure u32 avalanche hash: xor-shift and odd-constant-multiply finalizer.
fn hash_u32(h_in: u32) -> u32 {
    var h = h_in;
    h = h ^ (h >> 16u);
    h = h * MIX_A;
    h = h ^ (h >> 15u);
    h = h * MIX_B;
    h = h ^ (h >> 16u);
    return h;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let q = queries[idx];

    let luma = clamp(
        LUMA_R * q.color_r + LUMA_G * q.color_g + LUMA_B * q.color_b,
        0.0,
        1.0,
    );

    let a = q.px * ODD_X;
    let b = q.py * ODD_Y;
    let c = q.frame * ODD_FRAME;
    let h = hash_u32(a ^ b ^ c);
    let grain = f32(h >> 8u) / NORM_24;

    results[idx].luma = luma;
    results[idx].grain = grain;
}
"#;

/// Uniform parameters for one dispatch: the count plus three pad words to fill
/// a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`FILM_GRAIN_LUMA_WGSL`].
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
/// the three color channels and the pixel/frame indices (alignment `4`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Base color red channel.
    color_r: f32,
    /// Base color green channel.
    color_g: f32,
    /// Base color blue channel.
    color_b: f32,
    /// Pixel `x` coordinate.
    px: u32,
    /// Pixel `y` coordinate.
    py: u32,
    /// Frame index.
    frame: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Shaded`
/// struct: the clamped luma and the hashed grain (alignment `4`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Clamped `Rec. 709` luma.
    luma: f32,
    /// Hashed grain value in `[0, 1)`.
    grain: f32,
}

/// One film-grain query to run on the device: the base color and the pixel and
/// frame indices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilmGrainLumaQuery {
    /// Base color red channel.
    pub color_r: f32,
    /// Base color green channel.
    pub color_g: f32,
    /// Base color blue channel.
    pub color_b: f32,
    /// Pixel `x` coordinate.
    pub px: u32,
    /// Pixel `y` coordinate.
    pub py: u32,
    /// Frame index.
    pub frame: u32,
}

/// One film-grain result: the clamped `Rec. 709` luma and the hashed grain in
/// `[0, 1)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilmGrainLumaResult {
    /// Clamped `Rec. 709` luma.
    pub luma: f32,
    /// Hashed grain value in `[0, 1)`.
    pub grain: f32,
}

/// Encodes one [`FilmGrainLumaQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &FilmGrainLumaQuery) -> GpuQuery {
    GpuQuery {
        color_r: q.color_r,
        color_g: q.color_g,
        color_b: q.color_b,
        px: q.px,
        py: q.py,
        frame: q.frame,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`FilmGrainLumaResult`].
fn decode_result(raw: &GpuResult) -> FilmGrainLumaResult {
    FilmGrainLumaResult {
        luma: raw.luma,
        grain: raw.grain,
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

/// A compiled, reusable film-grain compute pipeline, twinning the `CPU` golden
/// [`luma709`](prism_render_architecture::particle::film_grain::luma709) and
/// [`grain_hash01`](prism_render_architecture::particle::film_grain::grain_hash01).
pub struct GpuFilmGrainLuma {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuFilmGrainLuma {
    /// Compiles the film-grain kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuFilmGrainLuma {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_film_grain_luma"),
            source: ShaderSource::Wgsl(FILM_GRAIN_LUMA_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_film_grain_luma_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_film_grain_luma_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_film_grain_luma_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuFilmGrainLuma {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs every query in `queries` and returns one [`FilmGrainLumaResult`]
    /// per input, in order.
    ///
    /// An empty `queries` batch returns an empty vector with no dispatch
    /// issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[FilmGrainLumaQuery],
    ) -> Vec<FilmGrainLumaResult> {
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
            label: Some("prism_volumetric_film_grain_luma_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_film_grain_luma_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_film_grain_luma_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_film_grain_luma_bind_group"),
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
            label: Some("prism_volumetric_film_grain_luma_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_film_grain_luma_encoder"),
        });
        {
            // One thread per query.
            let threads = count as u32;
            let groups = threads.div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_film_grain_luma_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
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
