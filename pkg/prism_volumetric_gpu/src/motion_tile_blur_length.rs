//! `wgpu` compute twin of the per-tile motion-blur half-length budget
//! [`motion_blur_half_length`](prism_render_architecture::motion::tiles::motion_blur_half_length)
//! that the screen-tile classifier derives from a tile's coarse velocity.
//!
//! The reconstruction blur reads back a per-tile half-length (in pixels) so the
//! sample budget stays bounded even when a tile carries a runaway velocity.
//! That half-length is a stateless, closed-form reduction of the tile velocity,
//! the shutter fraction, and a hard clamp, so it ports cleanly to the device: a
//! passing real-device parity run is direct evidence the ported kernel honors
//! the same shutter scaling and clamp the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! One thread computes one half-length. For a tile `velocity`, a
//! `shutter_fraction`, and a `max_radius_pixels` clamp, the result mirrors the
//! `CPU` golden
//! [`motion_blur_half_length`](prism_render_architecture::motion::tiles::motion_blur_half_length):
//! the shutter is sanitized to `[0, 1]`, the clamp is sanitized to be
//! non-negative, the per-frame displacement `sqrt(vx * vx + vy * vy)` is scaled
//! by the shutter and halved to a symmetric half-length, and the result is
//! clamped to `max_radius_pixels`. Only `+`, `*`, `sqrt`, `min`, and `max`
//! appear, so the kernel maps directly onto the portable core-`WGSL` subset.
//!
//! # What stays on the host
//!
//! The `NaN` leg of the reference's `sanitize_non_negative` (which maps `NaN` to
//! `0`) stays a host concern: the device twin assumes finite inputs, so the
//! parity fixtures feed only finite, in-range values and never probe `NaN` or
//! infinite sanitization. The surrounding tile-field sweep
//! ([`classify_field`](prism_render_architecture::motion::tiles::classify_field)),
//! the [`TileClassification`](prism_render_architecture::motion::tiles::TileClassification)
//! bookkeeping, and the velocity-field dilation that produces each coarse tile
//! velocity are stateful, variable-length host passes and are not twinned here.
//!
//! # Correctness model
//!
//! The half-length is a single `sqrt` followed by three multiplies and two
//! clamps — no transcendental — so the `CPU` and `GPU` agree to within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+`, `*`, `sqrt`,
//! `min`, and `max` on `f32` — with no `sin`, `cos`, `exp`, `log`, `pow`,
//! `tan`, no inverse trigonometry, no `smoothstep`, and no `round`. Each thread
//! performs a fixed, bounded sequence of arithmetic, so the kernel provably
//! terminates. No optional device feature is required, so it runs unmodified
//! across backends.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::motion::tiles::motion_blur_half_length`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` motion-blur half-length kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`motion_blur_half_length`](prism_render_architecture::motion::tiles::motion_blur_half_length);
/// see the module documentation for the algorithm.
const MOTION_TILE_BLUR_LENGTH_WGSL: &str = r#"
// Per-tile motion-blur half-length twin: one thread derives one half-length
// from a tile velocity, the shutter fraction, and a clamp, mirroring the CPU
// golden `motion_blur_half_length` with only + * sqrt min max on f32.
//
// Provenance: 孪生自本仓 prism_render_architecture::motion::tiles::motion_blur_half_length；无第三方引擎源码或衍生代码。

struct Params {
    // Number of tiles in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Tile velocity in pixels per frame.
    velocity_x: f32,
    velocity_y: f32,
    // Shutter fraction (portion of the frame interval the shutter is open).
    shutter_fraction: f32,
    // Hard clamp on the returned half-length, in pixels.
    max_radius_pixels: f32,
}

struct Result {
    // Clamped motion-blur half-length in pixels.
    half_length: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Finite-input equivalent of the golden `sanitize_non_negative`: negatives
// clamp to zero. The NaN leg of the golden stays a host concern.
fn sanitize_non_negative(x: f32) -> f32 {
    return max(x, 0.0);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let shutter = min(sanitize_non_negative(q.shutter_fraction), 1.0);
    let max_radius = sanitize_non_negative(q.max_radius_pixels);
    let length = sqrt(q.velocity_x * q.velocity_x + q.velocity_y * q.velocity_y);
    let half_length = length * shutter * 0.5;
    var out: Result;
    out.half_length = min(half_length, max_radius);
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    out.pad2 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the operation count plus three pad words
/// to fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`MOTION_TILE_BLUR_LENGTH_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid tiles in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one half-length query, matching the `WGSL`
/// `Query` struct: the tile velocity components followed by the shutter
/// fraction and the clamp.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Tile velocity `x` in pixels per frame.
    velocity_x: f32,
    /// Tile velocity `y` in pixels per frame.
    velocity_y: f32,
    /// Shutter fraction (portion of the frame the shutter is open).
    shutter_fraction: f32,
    /// Hard clamp on the returned half-length, in pixels.
    max_radius_pixels: f32,
}

/// `repr(C)` `std430` layout of one half-length result, matching the `WGSL`
/// `Result` struct: the clamped half-length plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Clamped motion-blur half-length in pixels.
    half_length: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
    /// Padding word.
    pad2: f32,
}

/// One per-tile motion-blur half-length query to run on the device, mirroring
/// the golden
/// [`motion_blur_half_length`](prism_render_architecture::motion::tiles::motion_blur_half_length).
///
/// `velocity` is the tile velocity `(x, y)` in pixels per frame,
/// `shutter_fraction` is the portion of the frame interval the shutter is open,
/// and `max_radius_pixels` is the hard clamp on the returned half-length.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionTileBlurLengthQuery {
    /// Tile velocity `(x, y)` in pixels per frame.
    pub velocity: [f32; 2],
    /// Shutter fraction (portion of the frame the shutter is open).
    pub shutter_fraction: f32,
    /// Hard clamp on the returned half-length, in pixels.
    pub max_radius_pixels: f32,
}

impl MotionTileBlurLengthQuery {
    /// Builds a half-length query from the tile `velocity`, the
    /// `shutter_fraction`, and the `max_radius_pixels` clamp.
    #[must_use]
    pub const fn new(
        velocity: [f32; 2],
        shutter_fraction: f32,
        max_radius_pixels: f32,
    ) -> MotionTileBlurLengthQuery {
        MotionTileBlurLengthQuery {
            velocity,
            shutter_fraction,
            max_radius_pixels,
        }
    }
}

/// One resolved half-length result, mirroring the golden
/// [`motion_blur_half_length`](prism_render_architecture::motion::tiles::motion_blur_half_length)
/// output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionTileBlurLengthResult {
    /// Clamped motion-blur half-length in pixels.
    pub half_length: f32,
}

/// Encodes one [`MotionTileBlurLengthQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &MotionTileBlurLengthQuery) -> GpuQuery {
    GpuQuery {
        velocity_x: q.velocity[0],
        velocity_y: q.velocity[1],
        shutter_fraction: q.shutter_fraction,
        max_radius_pixels: q.max_radius_pixels,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`MotionTileBlurLengthResult`].
fn decode_result(raw: &GpuResult) -> MotionTileBlurLengthResult {
    MotionTileBlurLengthResult {
        half_length: raw.half_length,
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

/// A compiled, reusable per-tile motion-blur half-length compute pipeline,
/// twinning the numeric core of the `CPU` golden
/// [`motion_blur_half_length`](prism_render_architecture::motion::tiles::motion_blur_half_length).
pub struct GpuMotionTileBlurLength {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMotionTileBlurLength {
    /// Compiles the half-length kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMotionTileBlurLength {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_motion_tile_blur_length"),
            source: ShaderSource::Wgsl(MOTION_TILE_BLUR_LENGTH_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_motion_tile_blur_length_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_motion_tile_blur_length_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_motion_tile_blur_length_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMotionTileBlurLength {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs every half-length query in `queries` and returns one
    /// [`MotionTileBlurLengthResult`] per input, in order.
    ///
    /// The outputs match the reference to within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[MotionTileBlurLengthQuery],
    ) -> Vec<MotionTileBlurLengthResult> {
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
            label: Some("prism_volumetric_motion_tile_blur_length_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_motion_tile_blur_length_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_motion_tile_blur_length_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_motion_tile_blur_length_bind_group"),
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
            label: Some("prism_volumetric_motion_tile_blur_length_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_motion_tile_blur_length_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_motion_tile_blur_length_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per tile, flattened to a 1-D dispatch.
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
