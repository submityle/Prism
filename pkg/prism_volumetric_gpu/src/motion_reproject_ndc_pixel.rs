//! `wgpu` compute twin of the `NDC` <-> pixel coordinate affine transforms in
//! the motion reprojection reference
//! ([`reproject`](prism_render_architecture::motion::reproject)).
//!
//! The reprojection kernel maps a current-frame pixel back to where its surface
//! was last frame; the first and last steps of every such map are the two
//! stateless affine conversions between the `[-1, 1]` `NDC` cube (`y` up) and the
//! top-left-origin pixel grid (`y` down). This twin reproduces that pure numeric
//! pair — [`ndc_to_pixel`](prism_render_architecture::motion::reproject::ndc_to_pixel)
//! and [`pixel_to_ndc`](prism_render_architecture::motion::reproject::pixel_to_ndc)
//! — on the device, so a passing real-device parity test is direct evidence the
//! ported kernel produces the same screen-space coordinates the reference does,
//! not merely that the shader compiles.
//!
//! # What is twinned
//!
//! One thread performs one conversion, selected by an op code:
//!
//! - [`ndc_to_pixel`](prism_render_architecture::motion::reproject::ndc_to_pixel):
//!   `px = (x * 0.5 + 0.5) * width`, `py = (0.5 - y * 0.5) * height` (the `y`
//!   flip puts `NDC` `+1` at pixel row `0`).
//! - [`pixel_to_ndc`](prism_render_architecture::motion::reproject::pixel_to_ndc):
//!   `nx = (px / width) * 2 - 1`, `ny = 1 - (py / height) * 2` (the inverse map).
//!
//! Both are affine in the input coordinate with the render-target extents as
//! coefficients, using only `+ - * /`, so each maps directly onto the portable
//! core-`WGSL` subset.
//!
//! # What stays on the host
//!
//! The integer screen dimensions carried by
//! [`ScreenDims`](prism_render_architecture::motion::ScreenDims) are clamped to a
//! minimum of one pixel at construction so the divide in `pixel_to_ndc` never
//! hits zero; the host mirrors that clamp when packing a query, so the kernel
//! consumes an already-sanitized extent and performs no zero guard. The matrix
//! reconstruction, perspective divide, and confidence logic that frame these two
//! conversions in [`ReprojectionContext`](prism_render_architecture::motion::reproject::ReprojectionContext)
//! are not twinned here: they consume a cached `4x4` inverse and are reprojection
//! policy, not the stateless coordinate affine this module targets.
//!
//! # Correctness model
//!
//! Each output threads through a multiply-add (or a divide and a multiply-add)
//! only — no `sqrt`, no transcendental — so the `CPU` and `GPU` agree to within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+ - * /`, an unsigned
//! `f32` conversion, and integer equality — with no `sin`, `cos`, `exp`, `log`,
//! `pow`, `tan`, no inverse trigonometry, no `smoothstep`, and no `sqrt`. Each
//! thread performs a fixed, bounded sequence of arithmetic, so the kernel
//! provably terminates. No optional device feature is required, so it runs
//! unmodified across backends.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::motion::reproject`；无第三方引擎源码或衍生代码。
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

/// Op code: convert an `NDC` coordinate to a pixel coordinate.
const OP_NDC_TO_PIXEL: u32 = 0;
/// Op code: convert a pixel coordinate to an `NDC` coordinate.
const OP_PIXEL_TO_NDC: u32 = 1;

/// The portable core-`WGSL` `NDC` <-> pixel conversion kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`reproject`](prism_render_architecture::motion::reproject) affine pair; see
/// the module documentation for the algorithm.
const MOTION_REPROJECT_NDC_PIXEL_WGSL: &str = r#"
// Per-operation NDC <-> pixel coordinate twin: one thread performs one affine
// conversion, selected by an op code, mirroring the CPU golden
// `motion::reproject` closed forms with only + - * / and an unsigned-to-float
// conversion. The screen extents are sanitized (clamped to at least one pixel)
// by the host, so the divide never hits zero and the kernel needs no guard.
//
// Provenance: 孪生自本仓 prism_render_architecture::motion::reproject；无第三方
// 引擎源码或衍生代码。

// Op codes, matching the host-side constants.
const OP_NDC_TO_PIXEL: u32 = 0u;
const OP_PIXEL_TO_NDC: u32 = 1u;

struct Params {
    // Number of operations in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Selected operation (see op constants).
    op: u32,
    // Render-target width and height in pixels (already clamped to >= 1 by host).
    width: u32,
    height: u32,
    pad0: u32,
    // Input coordinate: NDC (x, y) or pixel (x, y) depending on the op.
    in_x: f32,
    in_y: f32,
    pad1: f32,
    pad2: f32,
}

struct Result {
    // Output coordinate: pixel (x, y) or NDC (x, y) depending on the op.
    out_x: f32,
    out_y: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// NDC -> pixel: scale the [-1, 1] range into [0, extent] with a Y flip so NDC
// +1 (top) lands on pixel row 0.
fn ndc_to_pixel(nx: f32, ny: f32, w: f32, h: f32) -> vec2<f32> {
    let px = (nx * 0.5 + 0.5) * w;
    let py = (0.5 - ny * 0.5) * h;
    return vec2<f32>(px, py);
}

// pixel -> NDC: the inverse map, normalizing the pixel into [0, 1], scaling to
// [-1, 1], and flipping Y back to point up.
fn pixel_to_ndc(px: f32, py: f32, w: f32, h: f32) -> vec2<f32> {
    let nx = (px / w) * 2.0 - 1.0;
    let ny = 1.0 - (py / h) * 2.0;
    return vec2<f32>(nx, ny);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let w = f32(q.width);
    let h = f32(q.height);

    var out: Result;
    out.out_x = 0.0;
    out.out_y = 0.0;
    out.pad0 = 0.0;
    out.pad1 = 0.0;

    if (q.op == OP_NDC_TO_PIXEL) {
        let p = ndc_to_pixel(q.in_x, q.in_y, w, h);
        out.out_x = p.x;
        out.out_y = p.y;
    } else if (q.op == OP_PIXEL_TO_NDC) {
        let n = pixel_to_ndc(q.in_x, q.in_y, w, h);
        out.out_x = n.x;
        out.out_y = n.y;
    }

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the operation count plus three pad words
/// to fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`MOTION_REPROJECT_NDC_PIXEL_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid operations in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one conversion operation, matching the `WGSL`
/// `Query` struct: the op code, the sanitized render-target extents, and the
/// input coordinate, padded to a `32`-byte block.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Selected operation.
    op: u32,
    /// Render-target width in pixels (already clamped to at least `1`).
    width: u32,
    /// Render-target height in pixels (already clamped to at least `1`).
    height: u32,
    /// Padding word.
    pad0: u32,
    /// Input coordinate `x` (`NDC` or pixel depending on the op).
    in_x: f32,
    /// Input coordinate `y` (`NDC` or pixel depending on the op).
    in_y: f32,
    /// Padding word.
    pad1: f32,
    /// Padding word.
    pad2: f32,
}

/// `repr(C)` `std430` layout of one conversion result, matching the `WGSL`
/// `Result` struct: the output coordinate pair padded to a `16`-byte block.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Output coordinate `x` (pixel or `NDC` depending on the op).
    out_x: f32,
    /// Output coordinate `y` (pixel or `NDC` depending on the op).
    out_y: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
}

/// One `NDC` <-> pixel conversion to run on the device, mirroring the golden
/// [`reproject`](prism_render_architecture::motion::reproject) affine pair.
///
/// The `dims` carried by each variant are the render-target extents; the host
/// clamps each axis up to `1` (matching
/// [`ScreenDims::new`](prism_render_architecture::motion::ScreenDims::new)) when
/// packing the query, so the kernel's divide never sees a zero extent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MotionReprojectNdcPixelQuery {
    /// Convert an `NDC` coordinate (`x, y` in `[-1, 1]`, `y` up) to a pixel.
    NdcToPixel {
        /// `NDC` coordinate `(x, y)`.
        ndc: [f32; 2],
        /// Render-target `(width, height)` in pixels.
        dims: [u32; 2],
    },
    /// Convert a pixel coordinate (origin top-left, `y` down) to `NDC`.
    PixelToNdc {
        /// Pixel coordinate `(x, y)`.
        pixel: [f32; 2],
        /// Render-target `(width, height)` in pixels.
        dims: [u32; 2],
    },
}

/// One resolved conversion result, mirroring the golden
/// [`reproject`](prism_render_architecture::motion::reproject) affine outputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MotionReprojectNdcPixelResult {
    /// Pixel coordinate from a [`MotionReprojectNdcPixelQuery::NdcToPixel`].
    NdcToPixel {
        /// Pixel coordinate `(x, y)`.
        pixel: [f32; 2],
    },
    /// `NDC` coordinate from a [`MotionReprojectNdcPixelQuery::PixelToNdc`].
    PixelToNdc {
        /// `NDC` coordinate `(x, y)`.
        ndc: [f32; 2],
    },
}

/// Clamps a render-target extent up to at least `1`, mirroring
/// [`ScreenDims::new`](prism_render_architecture::motion::ScreenDims::new) so the
/// device divide never sees a zero extent.
fn sanitize_extent(extent: u32) -> u32 {
    if extent == 0 {
        1
    } else {
        extent
    }
}

/// Encodes one [`MotionReprojectNdcPixelQuery`] into its `std430` [`GpuQuery`]
/// slot, clamping the extents to the sanitized minimum.
fn encode_query(q: &MotionReprojectNdcPixelQuery) -> GpuQuery {
    let mut g = GpuQuery {
        op: OP_NDC_TO_PIXEL,
        width: 1,
        height: 1,
        pad0: 0,
        in_x: 0.0,
        in_y: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    };
    match *q {
        MotionReprojectNdcPixelQuery::NdcToPixel { ndc, dims } => {
            g.op = OP_NDC_TO_PIXEL;
            g.in_x = ndc[0];
            g.in_y = ndc[1];
            g.width = sanitize_extent(dims[0]);
            g.height = sanitize_extent(dims[1]);
        }
        MotionReprojectNdcPixelQuery::PixelToNdc { pixel, dims } => {
            g.op = OP_PIXEL_TO_NDC;
            g.in_x = pixel[0];
            g.in_y = pixel[1];
            g.width = sanitize_extent(dims[0]);
            g.height = sanitize_extent(dims[1]);
        }
    }
    g
}

/// Decodes one packed [`GpuResult`] into the public
/// [`MotionReprojectNdcPixelResult`], selecting the variant from the original
/// query.
fn decode_result(
    q: &MotionReprojectNdcPixelQuery,
    raw: &GpuResult,
) -> MotionReprojectNdcPixelResult {
    match *q {
        MotionReprojectNdcPixelQuery::NdcToPixel { .. } => {
            MotionReprojectNdcPixelResult::NdcToPixel {
                pixel: [raw.out_x, raw.out_y],
            }
        }
        MotionReprojectNdcPixelQuery::PixelToNdc { .. } => {
            MotionReprojectNdcPixelResult::PixelToNdc {
                ndc: [raw.out_x, raw.out_y],
            }
        }
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

/// A compiled, reusable `NDC` <-> pixel conversion compute pipeline, twinning the
/// stateless coordinate affine of the `CPU` golden
/// [`reproject`](prism_render_architecture::motion::reproject) module.
pub struct GpuMotionReprojectNdcPixel {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMotionReprojectNdcPixel {
    /// Compiles the `NDC` <-> pixel conversion kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMotionReprojectNdcPixel {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_motion_reproject_ndc_pixel"),
            source: ShaderSource::Wgsl(MOTION_REPROJECT_NDC_PIXEL_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_motion_reproject_ndc_pixel_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_motion_reproject_ndc_pixel_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_motion_reproject_ndc_pixel_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMotionReprojectNdcPixel {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs every conversion in `queries` and returns one
    /// [`MotionReprojectNdcPixelResult`] per input, in order.
    ///
    /// The outputs match the reference to within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[MotionReprojectNdcPixelQuery],
    ) -> Vec<MotionReprojectNdcPixelResult> {
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
            label: Some("prism_volumetric_motion_reproject_ndc_pixel_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_motion_reproject_ndc_pixel_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_motion_reproject_ndc_pixel_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_motion_reproject_ndc_pixel_bind_group"),
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
            label: Some("prism_volumetric_motion_reproject_ndc_pixel_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_motion_reproject_ndc_pixel_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_motion_reproject_ndc_pixel_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per operation, flattened to a 1-D dispatch.
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

        queries
            .iter()
            .zip(raw.iter())
            .map(|(q, r)| decode_result(q, r))
            .collect()
    }
}
