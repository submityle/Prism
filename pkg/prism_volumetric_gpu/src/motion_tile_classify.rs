//! `wgpu` compute twin of the stateless per-tile motion classification, `TAA`
//! tile flags, and motion-blur half-length budget inside the screen-tile motion
//! contract ([`tiles`](prism_render_architecture::motion::tiles)).
//!
//! The `CPU` golden derives, from one coarse per-tile velocity, three per-tile
//! decisions the full-resolution passes read back:
//!
//! - [`classify_tile`](prism_render_architecture::motion::tiles::classify_tile):
//!   buckets the tile into static / slow / fast by comparing the velocity
//!   magnitude against two pixel thresholds.
//! - [`taa_flags`](prism_render_architecture::motion::tiles::taa_flags) and
//!   [`TaaTileFlags::bits`](prism_render_architecture::motion::tiles::TaaTileFlags::bits):
//!   maps the bucket to the `TAA` resolve flag bitmask.
//! - [`motion_blur_half_length`](prism_render_architecture::motion::tiles::motion_blur_half_length):
//!   scales the per-frame displacement by the shutter fraction, halves it, and
//!   clamps it to a hard pixel radius.
//!
//! [`GpuMotionTileClassify`] is the on-device twin of those three maps. One
//! thread solves one query, computing the class code, the flag bitmask and the
//! half-length at once, reproducing the reference's exact thresholds and clamps,
//! so a passing real-device parity test is direct evidence the ported kernel
//! computes the same tile decisions the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! For the classifier the kernel reproduces the reference's downward-inclusive
//! bucketing — magnitude at or below `static_max_pixels` is static, else at or
//! below `slow_max_pixels` is slow, else fast — but compares *squared*
//! magnitudes (`mag_sq <= t * t`) against the squared thresholds instead of
//! taking a square root, so the discrete bucket never flips on a square-root
//! unit-in-the-last-place. Because the host passes sanitized thresholds (each
//! `>= 0`, with the slow threshold at least the static one), the squared
//! comparison is equivalent to the reference's magnitude comparison. The flag
//! bitmask is reproduced by the same class-to-flags map
//! (`STATIC = 1`, `NEEDS_NEIGHBOR_CLAMP = 4`, fast = `FAST_MOTION |
//! NEEDS_DILATION | NEEDS_NEIGHBOR_CLAMP = 14`). The half-length reproduces
//! `length * shutter * 0.5` clamped to the max radius, with the reference's
//! `NaN`/negative sanitization of the shutter fraction and the max radius; this
//! one output needs the true magnitude, so it uses the single `sqrt` intrinsic
//! the reference crate's determinism policy already allows.
//!
//! # What stays on the host
//!
//! The whole-field aggregation —
//! [`classify_field`](prism_render_architecture::motion::tiles) and the
//! per-bucket tallies over a variable-length
//! [`TileVelocityField`](prism_render_architecture::motion::dilation::TileVelocityField)
//! — is a variable-length reduction with no fixed-width per-query device
//! analogue, so it stays on the host and is never dispatched. Only the
//! stateless per-tile closed forms are twinned.
//!
//! # Correctness model
//!
//! The discrete answers — the class code and the flag bitmask — are built from
//! ordered squared-magnitude comparisons, so for fixtures chosen clear of a
//! threshold they agree exactly and the parity test asserts an exact `==` on
//! each. The continuous half-length threads through a `sqrt`, a multiply and a
//! clamp, so `CPU` and `GPU` are not bit-exact; it is asserted within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`, `+ - *
//! /`, ordered comparison and the single `sqrt` intrinsic — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry and no `u64`. No
//! optional device feature is required, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. There is no loop: each thread performs a fixed, bounded
//! sequence of arithmetic, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::motion::tiles`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` motion-tile-classification kernel, embedded inline
/// so the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`tiles`](prism_render_architecture::motion::tiles) closed forms; see the
/// module documentation for the algorithm.
const MOTION_TILE_CLASSIFY_WGSL: &str = r#"
// Motion-tile-classification twin: one thread computes all three stateless
// per-tile decisions for one query — the motion bucket (static / slow / fast,
// via squared-magnitude comparison), the TAA resolve flag bitmask, and the
// motion-blur half-length — mirroring the CPU golden `motion::tiles` with only
// + - * /, min/max, ordered comparison and the single sqrt intrinsic. It owns no
// whole-field aggregation; that variable-length reduction stays on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::motion::tiles；无第三方引擎
// 源码或衍生代码。

// Class codes, matching the host enum discriminant order.
const CLASS_STATIC: u32 = 0u;
const CLASS_SLOW: u32 = 1u;
const CLASS_FAST: u32 = 2u;

// TaaTileFlags bits, mirroring the golden associated constants.
const FLAG_STATIC: u32 = 1u;               // 1 << 0
const FLAG_NEEDS_DILATION: u32 = 2u;       // 1 << 1
const FLAG_NEEDS_NEIGHBOR_CLAMP: u32 = 4u; // 1 << 2
const FLAG_FAST_MOTION: u32 = 8u;          // 1 << 3

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Tile velocity in pixels; z and w lanes are unused padding.
    velocity: vec4<f32>,
    // static_max_pixels, slow_max_pixels, shutter_fraction, max_radius_pixels.
    thresholds: vec4<f32>,
}

struct Result {
    // Motion-blur half-length in pixels; y, z and w lanes are unused padding.
    half_length: vec4<f32>,
    // Motion class code (0 static, 1 slow, 2 fast).
    class_code: u32,
    // TAA resolve flag bitmask.
    flags: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Clamps NaN and negatives to zero, leaving other finite values untouched.
// `!(x >= 0.0)` captures both `x < 0` and `x` NaN (NaN >= 0.0 is false),
// mirroring the golden `x.is_nan() || x < 0.0` branch.
fn sanitize_nn(x: f32) -> f32 {
    if (!(x >= 0.0)) {
        return 0.0;
    }
    return x;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let v = q.velocity.xy;
    let mag_sq = v.x * v.x + v.y * v.y;
    let static_max = q.thresholds.x;
    let slow_max = q.thresholds.y;

    // --- classify (squared-magnitude comparison) -------------------------
    // The host passes sanitized thresholds (each >= 0, slow >= static), so
    // `mag_sq <= t * t` is equivalent to the golden `magnitude <= t` without a
    // square root, keeping the discrete bucket free of sqrt rounding.
    var class_code: u32 = CLASS_FAST;
    if (mag_sq <= static_max * static_max) {
        class_code = CLASS_STATIC;
    } else if (mag_sq <= slow_max * slow_max) {
        class_code = CLASS_SLOW;
    }

    // --- flags (class -> TAA bitmask) ------------------------------------
    var flags: u32 = FLAG_FAST_MOTION | FLAG_NEEDS_DILATION | FLAG_NEEDS_NEIGHBOR_CLAMP;
    if (class_code == CLASS_STATIC) {
        flags = FLAG_STATIC;
    } else if (class_code == CLASS_SLOW) {
        flags = FLAG_NEEDS_NEIGHBOR_CLAMP;
    }

    // --- motion-blur half-length -----------------------------------------
    // Scale the per-frame displacement by the sanitized shutter fraction
    // (clamped to 1), halve it, and clamp to the sanitized max radius. The true
    // magnitude is needed here, so the single sqrt intrinsic is used.
    let shutter = min(sanitize_nn(q.thresholds.z), 1.0);
    let max_radius = sanitize_nn(q.thresholds.w);
    let mag = sqrt(mag_sq);
    let hl = mag * shutter * 0.5;
    let half_length = min(hl, max_radius);

    var out: Result;
    out.half_length = vec4<f32>(half_length, 0.0, 0.0, 0.0);
    out.class_code = class_code;
    out.flags = flags;
    out.pad0 = 0u;
    out.pad1 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`MOTION_TILE_CLASSIFY_WGSL`].
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

/// `repr(C)` `std430` layout of one classification query, matching the `WGSL`
/// `Query` struct: a `16`-byte velocity lane followed by a `16`-byte lane of the
/// two magnitude thresholds, the shutter fraction and the max radius.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Tile velocity in pixels, padded to a `16`-byte lane.
    velocity: [f32; 4],
    /// `static_max_pixels`, `slow_max_pixels`, `shutter_fraction`,
    /// `max_radius_pixels`.
    thresholds: [f32; 4],
}

/// `repr(C)` `std430` layout of one classification result, matching the `WGSL`
/// `Result` struct: a `16`-byte half-length lane, the class code, the flag
/// bitmask, and two pad words to a `32`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Motion-blur half-length in pixels, padded to a `16`-byte lane.
    half_length: [f32; 4],
    /// Motion class code (`0` static, `1` slow, `2` fast).
    class_code: u32,
    /// `TAA` resolve flag bitmask.
    flags: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One motion-tile-classification query: the tile velocity plus the thresholds
/// and shutter parameters the three reference maps read.
///
/// The host owns the surrounding whole-field aggregation and enqueues one
/// [`MotionTileClassifyQuery`] per tile, mirroring the inputs the reference
/// [`tiles`](prism_render_architecture::motion::tiles) functions take.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionTileClassifyQuery {
    /// Tile velocity `(x, y)` in pixels (the golden `velocity`).
    pub velocity: [f32; 2],
    /// Static-bucket magnitude threshold in pixels (the golden
    /// `static_max_pixels`; host-sanitized `>= 0`).
    pub static_max_pixels: f32,
    /// Slow-bucket magnitude threshold in pixels (the golden `slow_max_pixels`;
    /// host-sanitized `>= static_max_pixels`).
    pub slow_max_pixels: f32,
    /// Shutter fraction in `[0, 1]` for the half-length (the golden
    /// `shutter_fraction`).
    pub shutter_fraction: f32,
    /// Hard half-length clamp in pixels (the golden `max_radius_pixels`).
    pub max_radius_pixels: f32,
}

impl MotionTileClassifyQuery {
    /// Builds a query from the tile velocity, the two magnitude thresholds, and
    /// the half-length shutter parameters.
    #[must_use]
    pub const fn new(
        velocity: [f32; 2],
        static_max_pixels: f32,
        slow_max_pixels: f32,
        shutter_fraction: f32,
        max_radius_pixels: f32,
    ) -> MotionTileClassifyQuery {
        MotionTileClassifyQuery {
            velocity,
            static_max_pixels,
            slow_max_pixels,
            shutter_fraction,
            max_radius_pixels,
        }
    }
}

/// One resolved classification query, mirroring the three reference maps.
///
/// `class_code` is the motion bucket (`0` static, `1` slow, `2` fast); `flags`
/// is the `TAA` resolve flag bitmask; `half_length` is the motion-blur
/// half-length in pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionTileClassifyResult {
    /// Motion class code (`0` static, `1` slow, `2` fast).
    pub class_code: u32,
    /// `TAA` resolve flag bitmask (the golden
    /// [`TaaTileFlags::bits`](prism_render_architecture::motion::tiles::TaaTileFlags::bits)).
    pub flags: u32,
    /// Motion-blur half-length in pixels.
    pub half_length: f32,
}

/// Encodes one [`MotionTileClassifyQuery`] into its `std430` [`GpuQuery`] slot,
/// packing the velocity into a `16`-byte lane and the four scalars into the
/// next.
fn encode_query(q: &MotionTileClassifyQuery) -> GpuQuery {
    GpuQuery {
        velocity: [q.velocity[0], q.velocity[1], 0.0, 0.0],
        thresholds: [
            q.static_max_pixels,
            q.slow_max_pixels,
            q.shutter_fraction,
            q.max_radius_pixels,
        ],
    }
}

/// Decodes one packed [`GpuResult`] into the public [`MotionTileClassifyResult`].
fn decode_result(raw: &GpuResult) -> MotionTileClassifyResult {
    MotionTileClassifyResult {
        class_code: raw.class_code,
        flags: raw.flags,
        half_length: raw.half_length[0],
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

/// A compiled, reusable motion-tile-classification compute pipeline, twinning
/// the stateless per-tile closed forms of the `CPU` golden
/// [`tiles`](prism_render_architecture::motion::tiles).
pub struct GpuMotionTileClassify {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMotionTileClassify {
    /// Compiles the motion-tile-classification kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMotionTileClassify {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_motion_tile_classify"),
            source: ShaderSource::Wgsl(MOTION_TILE_CLASSIFY_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_motion_tile_classify_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_motion_tile_classify_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_motion_tile_classify_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMotionTileClassify {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`MotionTileClassifyResult`] per input, in order.
    ///
    /// The class code and flag bitmask equal the reference exactly for queries
    /// clear of a threshold; the half-length matches to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[MotionTileClassifyQuery],
    ) -> Vec<MotionTileClassifyResult> {
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
            label: Some("prism_volumetric_motion_tile_classify_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_motion_tile_classify_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_motion_tile_classify_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_motion_tile_classify_bind_group"),
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
            label: Some("prism_volumetric_motion_tile_classify_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_motion_tile_classify_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_motion_tile_classify_pass"),
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
