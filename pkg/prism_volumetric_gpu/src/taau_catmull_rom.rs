//! `wgpu` compute twin of the per-pixel history-reprojection numeric core
//! ([`reproject`](prism_render_architecture::temporal_upscale::reproject)).
//!
//! A temporal upscaler must follow each pixel's motion back to its
//! previous-frame location, test whether that location still lies on screen,
//! build the separable Catmull-Rom resampling weights for the fractional
//! offset, and reject history whose stored depth disagrees with the current
//! surface. The `CPU` golden exposes those jobs as four pure, transcendental-free
//! functions:
//! [`reproject_pixel`](prism_render_architecture::temporal_upscale::reproject::reproject_pixel),
//! [`on_screen`](prism_render_architecture::temporal_upscale::reproject::on_screen),
//! [`catmull_rom_weights`](prism_render_architecture::temporal_upscale::reproject::catmull_rom_weights)
//! and
//! [`depth_disoccluded`](prism_render_architecture::temporal_upscale::reproject::depth_disoccluded).
//!
//! [`GpuTaauCatmullRom`] is the on-device twin that runs one thread per pixel
//! sample and reproduces each of those four quantities in a single pass, so a
//! passing real-device parity test is direct evidence the ported kernel
//! computes the same reprojection the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! For one sample `(t, x, y, motion, width, height, current_depth,
//! history_depth, tolerance)` the kernel emits:
//!
//! * `weights` — the four separable Catmull-Rom taps
//!   `catmull_rom_weights(t)`, the compact `a = -0.5` cubic polynomials that
//!   sum to exactly `1`;
//! * `reprojected` — the current-to-previous location
//!   `reproject_pixel(x, y, motion) = [x + motion.0, y + motion.1]`;
//! * `on_screen` — whether the reprojected location lies in
//!   `[0, width) x [0, height)`, with zero-size frames never on screen;
//! * `depth_disoccluded` — the relative-depth rejection
//!   `|current - history| / max(|current|, |history|, 1e-6) > tolerance`, with
//!   a non-positive or `NaN` tolerance disabling the test.
//!
//! The `on_screen` flag is evaluated on the *reprojected* location the same
//! sample produced, mirroring the reference call chain
//! `on_screen(reproject_pixel(x, y, motion), width, height)`.
//!
//! # What is not twinned (host-only)
//!
//! The reference `sample_catmull_rom` fetch is not twinned: it takes a caller
//! closure `F: Fn(i32, i32) -> [f32; 3]` reading an arbitrary history buffer,
//! which has no fixed-width, portable device analogue. The host owns that
//! `4 x 4` gather and feeds this twin only the scalar reprojection quantities.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+ - * /`, `abs`,
//! `max` and the ordered comparisons — with no `sin`, `cos`, `exp`, `log`,
//! `pow`, inverse trigonometry, `smoothstep`, `round`, `cbrt` or even `sqrt`,
//! and no optional device feature, so it runs unmodified on `Metal`, `Vulkan`
//! and `DX12`. There is no loop: each thread performs a fixed, bounded sequence
//! of arithmetic, so the kernel provably terminates.
//!
//! The `on_screen` zero-size guard uses integer equality on the `u32` extents
//! (`width == 0u || height == 0u`), which is exact and allowed; the fractional
//! bounds test uses only ordered `>=` / `<` comparisons. The
//! `depth_disoccluded` disable branch is expressed without an `f32` equality:
//! the test runs only when `tolerance > 0.0`, which is `false` for both a
//! non-positive tolerance and a `NaN` (every ordered comparison with `NaN` is
//! `false`), matching the golden `tolerance.is_nan() || tolerance <= 0.0`
//! fallback exactly.
//!
//! # Correctness model
//!
//! The discrete answers — `on_screen` and `depth_disoccluded` — are built from
//! magnitude comparisons, so for fixtures chosen clear of a boundary tie (a
//! reprojected coordinate near an integer extent, or a relative depth near the
//! exact tolerance) the `CPU` and `GPU` agree exactly and the parity test
//! asserts an exact `==` on each. The continuous `weights` and `reprojected`
//! thread through shared `+ - * /` sequences, so `CPU` and `GPU` are not
//! necessarily bit-exact: a `GPU` may land a few units in the last place from
//! the scalar reference. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on those continuous outputs.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::temporal_upscale::reproject`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` reprojection kernel, embedded inline so the twin
/// ships no external shader asset. It reproduces the
/// [`reproject`](prism_render_architecture::temporal_upscale::reproject) closed
/// forms; see the module documentation for the algorithm.
const TAAU_CATMULL_ROM_WGSL: &str = r#"
// History-reprojection twin: one thread maps one pixel sample through
// catmull_rom_weights + reproject_pixel + on_screen + depth_disoccluded,
// mirroring the CPU golden `temporal_upscale::reproject` with only + - * /,
// abs, max and ordered comparisons. Bools are returned as u32 0/1.
//
// Provenance: 孪生自本仓 prism_render_architecture::temporal_upscale::reproject；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of samples in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Fractional offset in [0, 1] for the Catmull-Rom taps.
    t: f32,
    // Current display-space pixel coordinates.
    x: f32,
    y: f32,
    // Current-to-previous motion vector.
    mx: f32,
    my: f32,
    // Frame extents (zero-size frames are never on screen).
    width: u32,
    height: u32,
    // Linear depths and the relative-depth rejection tolerance.
    current_depth: f32,
    history_depth: f32,
    tolerance: f32,
}

struct Res {
    // Catmull-Rom taps at offsets -1, 0, +1, +2.
    w0: f32,
    w1: f32,
    w2: f32,
    w3: f32,
    // Reprojected previous-frame location.
    rx: f32,
    ry: f32,
    // Discrete flags encoded as 0/1.
    on_screen: u32,
    disoccluded: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

@compute @workgroup_size(64)
fn reproject(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // catmull_rom_weights(t): the compact a = -0.5 cubic polynomials.
    let t = q.t;
    let t2 = t * t;
    let t3 = t2 * t;

    var out: Res;
    out.w0 = -0.5 * t3 + t2 - 0.5 * t;
    out.w1 = 1.5 * t3 - 2.5 * t2 + 1.0;
    out.w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
    out.w3 = 0.5 * t3 - 0.5 * t2;

    // reproject_pixel: current + motion.
    out.rx = q.x + q.mx;
    out.ry = q.y + q.my;

    // on_screen on the reprojected location; zero-size frames are never on
    // screen (integer equality on the u32 extents is exact).
    var on: u32 = 0u;
    if (q.width == 0u || q.height == 0u) {
        on = 0u;
    } else if (out.rx >= 0.0 && out.ry >= 0.0 && out.rx < f32(q.width) && out.ry < f32(q.height)) {
        on = 1u;
    }
    out.on_screen = on;

    // depth_disoccluded: a non-positive or NaN tolerance disables the test.
    // `tolerance > 0.0` is false for both cases, matching the golden
    // `tolerance.is_nan() || tolerance <= 0.0` disable branch without an f32
    // equality.
    var dis: u32 = 0u;
    if (q.tolerance > 0.0) {
        let denom = max(max(abs(q.current_depth), abs(q.history_depth)), 1e-6);
        let relative = abs(q.current_depth - q.history_depth) / denom;
        if (relative > q.tolerance) {
            dis = 1u;
        }
    }
    out.disoccluded = dis;

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the sample count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`TAAU_CATMULL_ROM_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid samples in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one reprojection query: five `f32` reprojection
/// inputs, the two `u32` frame extents and the three `f32` depth inputs, a
/// `40`-byte scalar-packed stride matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Fractional offset for the Catmull-Rom taps.
    t: f32,
    /// Current pixel `x`.
    x: f32,
    /// Current pixel `y`.
    y: f32,
    /// Motion `x` component.
    mx: f32,
    /// Motion `y` component.
    my: f32,
    /// Frame width.
    width: u32,
    /// Frame height.
    height: u32,
    /// Current linear depth.
    current_depth: f32,
    /// History linear depth.
    history_depth: f32,
    /// Relative-depth rejection tolerance.
    tolerance: f32,
}

/// `repr(C)` `std430` layout of one reprojection result, matching the `WGSL`
/// `Res` struct: the four Catmull-Rom taps, the reprojected location and the two
/// discrete flags encoded as `u32` `0`/`1`, a `32`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Catmull-Rom tap at offset `-1`.
    w0: f32,
    /// Catmull-Rom tap at offset `0`.
    w1: f32,
    /// Catmull-Rom tap at offset `+1`.
    w2: f32,
    /// Catmull-Rom tap at offset `+2`.
    w3: f32,
    /// Reprojected location `x`.
    rx: f32,
    /// Reprojected location `y`.
    ry: f32,
    /// `1` when the reprojected location is on screen, else `0`.
    on_screen: u32,
    /// `1` when the sample is depth-disoccluded, else `0`.
    disoccluded: u32,
}

/// One reprojection query: the fractional offset `t`, the current pixel
/// `(x, y)`, its current-to-previous `motion` vector, the frame extents and the
/// current / history linear depths with a relative-depth `tolerance`.
///
/// `on_screen` is evaluated on the reprojected location this query produces, and
/// a non-positive or `NaN` `tolerance` disables the depth test, exactly as the
/// golden
/// [`reproject`](prism_render_architecture::temporal_upscale::reproject)
/// functions resolve them.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::temporal_upscale::reproject`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TaauCatmullRomQuery {
    /// Fractional offset in `[0, 1]` for the Catmull-Rom taps.
    pub t: f32,
    /// Current display-space pixel `x`.
    pub x: f32,
    /// Current display-space pixel `y`.
    pub y: f32,
    /// Current-to-previous motion vector.
    pub motion: [f32; 2],
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Current surface linear depth.
    pub current_depth: f32,
    /// Sampled history linear depth.
    pub history_depth: f32,
    /// Relative-depth rejection tolerance.
    pub tolerance: f32,
}

impl TaauCatmullRomQuery {
    /// Builds a query from the fractional offset, pixel, motion, extents and
    /// depth inputs.
    #[must_use]
    pub const fn new(
        t: f32,
        x: f32,
        y: f32,
        motion: [f32; 2],
        width: u32,
        height: u32,
        current_depth: f32,
        history_depth: f32,
        tolerance: f32,
    ) -> TaauCatmullRomQuery {
        TaauCatmullRomQuery {
            t,
            x,
            y,
            motion,
            width,
            height,
            current_depth,
            history_depth,
            tolerance,
        }
    }
}

/// One sample's resolved reprojection quantities, mirroring the `CPU` golden
/// [`reproject`](prism_render_architecture::temporal_upscale::reproject)
/// functions.
///
/// `weights` is
/// [`catmull_rom_weights`](prism_render_architecture::temporal_upscale::reproject::catmull_rom_weights)`(t)`,
/// `reprojected` is
/// [`reproject_pixel`](prism_render_architecture::temporal_upscale::reproject::reproject_pixel),
/// `on_screen` is
/// [`on_screen`](prism_render_architecture::temporal_upscale::reproject::on_screen)
/// of the reprojected location, and `depth_disoccluded` is
/// [`depth_disoccluded`](prism_render_architecture::temporal_upscale::reproject::depth_disoccluded).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::temporal_upscale::reproject`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TaauCatmullRomResult {
    /// The four separable Catmull-Rom taps at offsets `-1, 0, +1, +2`.
    pub weights: [f32; 4],
    /// The reprojected previous-frame location.
    pub reprojected: [f32; 2],
    /// Whether the reprojected location lies on screen.
    pub on_screen: bool,
    /// Whether the sample is rejected as depth-disoccluded.
    pub depth_disoccluded: bool,
}

/// Encodes one [`TaauCatmullRomQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &TaauCatmullRomQuery) -> GpuQuery {
    GpuQuery {
        t: q.t,
        x: q.x,
        y: q.y,
        mx: q.motion[0],
        my: q.motion[1],
        width: q.width,
        height: q.height,
        current_depth: q.current_depth,
        history_depth: q.history_depth,
        tolerance: q.tolerance,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`TaauCatmullRomResult`],
/// mapping the `u32` `0`/`1` flags back to `bool` with an exact integer compare.
fn decode_result(raw: &GpuResult) -> TaauCatmullRomResult {
    TaauCatmullRomResult {
        weights: [raw.w0, raw.w1, raw.w2, raw.w3],
        reprojected: [raw.rx, raw.ry],
        on_screen: raw.on_screen != 0,
        depth_disoccluded: raw.disoccluded != 0,
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

/// A compiled, reusable reprojection compute pipeline, twinning the `CPU` golden
/// [`reproject`](prism_render_architecture::temporal_upscale::reproject) numeric
/// core.
pub struct GpuTaauCatmullRom {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTaauCatmullRom {
    /// Compiles the reprojection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTaauCatmullRom {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_taau_catmull_rom"),
            source: ShaderSource::Wgsl(TAAU_CATMULL_ROM_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_taau_catmull_rom_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_taau_catmull_rom_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_taau_catmull_rom_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("reproject"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTaauCatmullRom {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every sample in `queries` and returns one
    /// [`TaauCatmullRomResult`] per input, in order.
    ///
    /// The continuous `weights` and `reprojected` fields equal the matching
    /// `CPU` golden quantities to within a last-place rounding slack; the
    /// `on_screen` and `depth_disoccluded` flags match exactly for samples clear
    /// of a boundary tie. An empty `queries` batch returns an empty vector with
    /// no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[TaauCatmullRomQuery],
    ) -> Vec<TaauCatmullRomResult> {
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
            label: Some("prism_volumetric_taau_catmull_rom_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_taau_catmull_rom_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_taau_catmull_rom_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_taau_catmull_rom_bind_group"),
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
            label: Some("prism_volumetric_taau_catmull_rom_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_taau_catmull_rom_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_taau_catmull_rom_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per sample, flattened to a 1-D dispatch.
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
