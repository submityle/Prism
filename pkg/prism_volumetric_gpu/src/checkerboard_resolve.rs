//! `wgpu` compute twin of the checkerboard-rendering *resolve* pass
//! ([`checkerboard_resolve`](prism_render_architecture::particle::checkerboard_resolve),
//! design §21).
//!
//! Checkerboard rendering shades only half the pixels each frame — the cells of
//! one colour of a checkerboard — and reconstructs the complementary cells from
//! the previous frame plus the current frame's spatial neighbourhood. A pixel
//! is *shaded this frame* when its parity `(x + y + frame) & 1` is `0`; the
//! other cells are *missing* and are rebuilt. The `CPU` golden
//! [`resolve`](prism_render_architecture::particle::checkerboard_resolve::resolve)
//! owns that math; [`GpuCheckerboardResolve`] is the on-device twin that runs
//! one thread per full-resolution output pixel and reproduces the same image.
//! A passing real-device parity test is therefore direct evidence the ported
//! kernel weaves the same shaded cells, picks the same edge-aware fill axis,
//! clamps history into the same neighbour box and blends by the same weight the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! One kernel covers the whole resolve. For a shaded cell it copies the
//! `current` sample through verbatim. For a missing cell it reproduces, bit for
//! bit, the four pieces the reference combines:
//!
//! * **Spatial fill** — the in-bounds orthogonal neighbours `[left, right, up,
//!   down]` form a horizontal and a vertical opposed pair; when both pairs
//!   exist the fill leans toward the axis of smaller `Rec.601` luma gradient
//!   (interpolating along an edge, not across it) via a hand-rolled `Hermite`
//!   `smoothstep`; a lone pair is used directly; a border/corner cell averages
//!   whatever single neighbours exist and falls back to its own stored value
//!   when fully isolated.
//! * **Neighbour box clamp** — the supplied `history` colour is clamped
//!   component-wise into the min/max box of the missing cell's known
//!   neighbours (the anti-ghosting guard), falling back to the spatial fill
//!   when the cell is isolated.
//! * **Blend** — the edge-aware fill is mixed against the box-clamped history
//!   by the clamped `history_weight`.
//!
//! The neighbour scan, the accumulation order (`[left, right, up, down]`), the
//! `a * (1 - t) + b * t` mix order and the luma weights all match the reference
//! so the low mantissa bits agree.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `abs`, the four arithmetic operators `+ - * /` and unsigned integer
//! and bitwise index math — with no `sin`, `cos`, `exp`, `log`, `pow`, `sqrt`,
//! no built-in `smoothstep` and no optional device feature, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. The `smoothstep` easing is
//! written out as the `Hermite` polynomial `t * t * (3 - 2t)` after a clamp.
//!
//! # Correctness model
//!
//! Each output pixel is a fixed, non-reorderable sequence of multiplies, adds,
//! `min` / `max` / `clamp` and one easing polynomial, so `CPU` and `GPU`
//! evaluate the same closed form in the same order. They are not bit-exact: a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate,
//! perturbing the low mantissa bits by a few units in the last place. The
//! parity test therefore asserts a tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`), tight enough to catch a genuinely wrong port (a swapped
//! fill axis, a dropped neighbour, a missing clamp, a wrong blend weight) yet
//! loose enough to admit legal fused multiply-add contraction. The parity
//! branch itself is exact on both sides: `(x + y + frame) & 1` is integer and
//! bitwise, so the two never disagree on which cells are shaded.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard spatial checkerboard reconstruction with a local
//! neighbour-box history clamp plus `wgpu` compute dispatch; no third-party
//! engine source or derived code.

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

/// Number of threads per workgroup. `64` is the portable, warp-friendly size
/// used across this crate; one thread handles one output pixel and the dispatch
/// is flattened to a single linear index so it stays one-dimensional.
const WORKGROUP_SIZE: u32 = 64;

/// Scalar components per pixel: an `RGBA` quad packed as one `std430`
/// `vec4<f32>` slot (`[r, g, b, a]`).
const CHANNELS: usize = 4;

/// The portable core-`WGSL` resolve kernel, embedded inline so the twin ships
/// as a single source file. The entry point `resolve_main` mirrors the `CPU`
/// golden
/// [`resolve`](prism_render_architecture::particle::checkerboard_resolve::resolve)
/// pixel for pixel; see the module documentation for the algorithm.
const CHECKERBOARD_RESOLVE_WGSL: &str = r#"
// Checkerboard-resolve twin: one thread per full-resolution output pixel. A
// shaded cell (parity (x + y + frame) & 1 == 0) is copied through verbatim; a
// missing cell is rebuilt from an edge-aware spatial fill of its orthogonal
// neighbours, blended against the neighbour-box-clamped history by the clamped
// history weight. Mirrors the CPU golden `particle::checkerboard_resolve`, uses
// only the portable core-WGSL subset (min/max/clamp/abs, + - * /, unsigned and
// bitwise index math, no built-in smoothstep), and takes no optional feature,
// so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard spatial checkerboard reconstruction with a local
// neighbour-box history clamp; no third-party engine source or derived code.

struct Params {
    // Full-resolution extents in pixels (one thread per pixel).
    full_width: u32,
    full_height: u32,
    // Checkerboard frame index selecting the shaded parity.
    frame: u32,
    // Trust placed in the box-clamped history versus the spatial fill, already
    // clamped into [0, 1] by the host to mirror the reference.
    weight: f32,
    // Padding to a 32-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

// The min/max box of a missing cell's known neighbours.
struct Box {
    lo: vec4<f32>,
    hi: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> current: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> history: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out_buf: array<vec4<f32>>;

// Gradient magnitude below which the two fill axes are treated as equally
// smooth, matching the reference `GRAD_EPS`.
const GRAD_EPS: f32 = 1.0e-6;
// Rec.601 luma weights, used only for the edge-aware axis decision.
const LUMA_R: f32 = 0.299;
const LUMA_G: f32 = 0.587;
const LUMA_B: f32 = 0.114;

// Flat row-major index of pixel (x, y).
fn flat_index(x: u32, y: u32) -> u32 {
    return y * params.full_width + x;
}

// Reads the colour at (x, y) from the full-resolution `current` buffer.
fn sample_current(x: u32, y: u32) -> vec4<f32> {
    return current[flat_index(x, y)];
}

// Rec.601 luma, accumulated R then G then B to match the reference order.
fn luma(c: vec4<f32>) -> f32 {
    return LUMA_R * c.x + LUMA_G * c.y + LUMA_B * c.z;
}

// Linear interpolation `a * (1 - t) + b * t`, component-wise, matching the
// reference `Rgba::mix` order exactly so the low mantissa bits agree.
fn mix_rgba(a: vec4<f32>, b: vec4<f32>, t: f32) -> vec4<f32> {
    let inv = 1.0 - t;
    return vec4<f32>(
        a.x * inv + b.x * t,
        a.y * inv + b.y * t,
        a.z * inv + b.z * t,
        a.w * inv + b.w * t,
    );
}

// The reference `smoothstep`: `t * t * (3 - 2t)` after clamping t into [0, 1].
// Written out as a Hermite polynomial so no built-in is used.
fn smoothstep_hermite(t_in: f32) -> f32 {
    let t = clamp(t_in, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Chooses between the horizontal and vertical fill by leaning toward the axis
// of smaller luma gradient, mirroring the reference `edge_aware`.
fn edge_aware(h_mean: vec4<f32>, h_grad: f32, v_mean: vec4<f32>, v_grad: f32) -> vec4<f32> {
    let denom = h_grad + v_grad;
    if (denom < GRAD_EPS) {
        return mix_rgba(h_mean, v_mean, 0.5);
    }
    let lean_vertical = smoothstep_hermite(h_grad / denom);
    return mix_rgba(h_mean, v_mean, lean_vertical);
}

// The edge-aware spatial fill for a missing cell at (x, y), mirroring the
// reference `spatial_fill` with the same `[left, right, up, down]` ordering.
fn spatial_fill(x: u32, y: u32) -> vec4<f32> {
    let has_left = x > 0u;
    let has_right = x + 1u < params.full_width;
    let has_up = y > 0u;
    let has_down = y + 1u < params.full_height;

    var left = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var right = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var up = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var down = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if (has_left) { left = sample_current(x - 1u, y); }
    if (has_right) { right = sample_current(x + 1u, y); }
    if (has_up) { up = sample_current(x, y - 1u); }
    if (has_down) { down = sample_current(x, y + 1u); }

    let horizontal = has_left && has_right;
    let vertical = has_up && has_down;

    if (horizontal && vertical) {
        let h_mean = mix_rgba(left, right, 0.5);
        let h_grad = abs(luma(left) - luma(right));
        let v_mean = mix_rgba(up, down, 0.5);
        let v_grad = abs(luma(up) - luma(down));
        return edge_aware(h_mean, h_grad, v_mean, v_grad);
    }
    if (horizontal) {
        return mix_rgba(left, right, 0.5);
    }
    if (vertical) {
        return mix_rgba(up, down, 0.5);
    }

    // Border/corner fallback: average whatever single neighbours exist,
    // accumulated in [left, right, up, down] order, or the cell's own value
    // when fully isolated.
    var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var count = 0u;
    if (has_left) { acc = acc + left; count = count + 1u; }
    if (has_right) { acc = acc + right; count = count + 1u; }
    if (has_up) { acc = acc + up; count = count + 1u; }
    if (has_down) { acc = acc + down; count = count + 1u; }
    if (count == 0u) {
        return sample_current(x, y);
    }
    if (count == 1u) {
        return acc;
    }
    if (count == 2u) {
        return acc * 0.5;
    }
    return acc * (1.0 / 3.0);
}

// The min/max box of a missing cell's known neighbours, scanned in the same
// `[left, right, up, down]` order; falls back to (fallback, fallback) when the
// cell is fully isolated, mirroring the reference `neighbor_box`.
fn neighbor_box(x: u32, y: u32, fallback: vec4<f32>) -> Box {
    let has_left = x > 0u;
    let has_right = x + 1u < params.full_width;
    let has_up = y > 0u;
    let has_down = y + 1u < params.full_height;

    var has_any = false;
    var lo = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var hi = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    if (has_left) {
        let c = sample_current(x - 1u, y);
        if (has_any) { lo = min(lo, c); hi = max(hi, c); } else { lo = c; hi = c; has_any = true; }
    }
    if (has_right) {
        let c = sample_current(x + 1u, y);
        if (has_any) { lo = min(lo, c); hi = max(hi, c); } else { lo = c; hi = c; has_any = true; }
    }
    if (has_up) {
        let c = sample_current(x, y - 1u);
        if (has_any) { lo = min(lo, c); hi = max(hi, c); } else { lo = c; hi = c; has_any = true; }
    }
    if (has_down) {
        let c = sample_current(x, y + 1u);
        if (has_any) { lo = min(lo, c); hi = max(hi, c); } else { lo = c; hi = c; has_any = true; }
    }

    if (has_any) {
        return Box(lo, hi);
    }
    return Box(fallback, fallback);
}

@compute @workgroup_size(64)
fn resolve_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.full_width * params.full_height;
    if (idx >= total) {
        return;
    }
    let x = idx % params.full_width;
    let y = idx / params.full_width;

    // Shaded cells (parity 0) are woven through verbatim.
    let parity = (x + y + params.frame) & 1u;
    if (parity == 0u) {
        out_buf[idx] = current[idx];
        return;
    }

    // Missing cells: edge-aware spatial fill, box-clamped history, blend.
    let fill = spatial_fill(x, y);
    let bounds = neighbor_box(x, y, fill);
    let clamped = clamp(history[idx], bounds.lo, bounds.hi);
    out_buf[idx] = mix_rgba(fill, clamped, params.weight);
}
"#;

/// One checkerboard-resolve request in the row-major, `4`-per-pixel `RGBA`
/// layout the device storage buffers expect.
///
/// Mirrors the `(res, current, history, frame, config)` arguments the reference
/// [`resolve`](prism_render_architecture::particle::checkerboard_resolve::resolve)
/// consumes. `current` and `history` must each hold exactly
/// `full_width * full_height` pixels in row-major order. `history_weight` is the
/// trust placed in the box-clamped history versus the spatial fill; the host
/// clamps it into `[0, 1]` exactly as the reference does. Derives only
/// [`PartialEq`] (no [`Eq`] / [`Hash`]) because the colours are `f32`.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckerboardResolveQuery {
    /// Full-resolution width in pixels.
    pub full_width: u32,
    /// Full-resolution height in pixels.
    pub full_height: u32,
    /// The current frame's shaded samples (only shaded cells are read).
    pub current: Vec<[f32; 4]>,
    /// The previous resolved frame, already reprojected upstream if at all.
    pub history: Vec<[f32; 4]>,
    /// The checkerboard frame index selecting the shaded parity.
    pub frame: u32,
    /// Trust placed in the box-clamped history versus the spatial fill.
    pub history_weight: f32,
}

impl CheckerboardResolveQuery {
    /// The number of pixels in the full-resolution image (`width * height`).
    #[must_use]
    pub fn full_pixel_count(&self) -> usize {
        self.full_width as usize * self.full_height as usize
    }
}

/// Uniform parameters for one resolve dispatch. `repr(C)` `std430` layout
/// matching `Params` in [`CHECKERBOARD_RESOLVE_WGSL`]: the full-resolution
/// extents, the frame index, the clamped history weight and four pad words —
/// `32` bytes, each field at the uniform offset the shader expects.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Full-resolution width in pixels.
    full_width: u32,
    /// Full-resolution height in pixels.
    full_height: u32,
    /// Checkerboard frame index selecting the shaded parity.
    frame: u32,
    /// History blend weight, pre-clamped into `[0, 1]` by the host.
    weight: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
    /// Padding word.
    pad3: u32,
}

/// A compiled, reusable checkerboard-resolve compute pipeline.
pub struct GpuCheckerboardResolve {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCheckerboardResolve {
    /// Compiles the checkerboard-resolve kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCheckerboardResolve {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_checkerboard_resolve"),
            source: ShaderSource::Wgsl(CHECKERBOARD_RESOLVE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_checkerboard_resolve_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_checkerboard_resolve_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_checkerboard_resolve_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("resolve_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCheckerboardResolve {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs the resolve on `query`, returning the reconstructed full-resolution
    /// image as row-major `RGBA` quads.
    ///
    /// The result equals the reference
    /// [`resolve`](prism_render_architecture::particle::checkerboard_resolve::resolve)
    /// to within the tolerance documented on this module. An empty image
    /// (`full_pixel_count == 0`) short-circuits to an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    ///
    /// # Panics
    ///
    /// Panics if the mapped readback range is unavailable after the device
    /// poll, which indicates a broken adapter and cannot be recovered from
    /// within a single dispatch.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, query: &CheckerboardResolveQuery) -> Vec<[f32; 4]> {
        let pixel_count = query.full_pixel_count();
        if pixel_count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            full_width: query.full_width,
            full_height: query.full_height,
            frame: query.frame,
            // Mirror the reference `resolve`, which clamps `history_weight`.
            weight: query.history_weight.clamp(0.0, 1.0),
            pad0: 0,
            pad1: 0,
            pad2: 0,
            pad3: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_checkerboard_resolve_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let current_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_checkerboard_resolve_current"),
            contents: bytemuck::cast_slice(&query.current),
            usage: BufferUsages::STORAGE,
        });
        let history_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_checkerboard_resolve_history"),
            contents: bytemuck::cast_slice(&query.history),
            usage: BufferUsages::STORAGE,
        });
        let out_bytes = (pixel_count * CHANNELS * size_of::<f32>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_checkerboard_resolve_output"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_checkerboard_resolve_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: current_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: history_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_checkerboard_resolve_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_checkerboard_resolve_encoder"),
        });
        {
            let groups = (pixel_count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_checkerboard_resolve_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per output pixel, flattened to a 1-D dispatch.
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
        let flat = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        stage.unmap();

        let mut pixels: Vec<[f32; 4]> = Vec::with_capacity(pixel_count);
        for chunk in flat.chunks_exact(CHANNELS) {
            pixels.push([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        debug_assert_eq!(pixels.len(), pixel_count);
        pixels
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
