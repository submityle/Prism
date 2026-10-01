//! `wgpu` compute twin of the screen-space edge-detection contract
//! ([`edge_detect`](prism_render_architecture::particle::edge_detect), design
//! sections 16-21, "边缘检测 / 轮廓").
//!
//! A stylized particle outline pass derives "where the silhouettes are" from
//! three screen-space signals — a `luminance`/color break, a depth
//! discontinuity, or a normal-direction break — and folds each into an edge
//! strength that can drive an outline stroke or an edge-directed anti-aliasing
//! (`AA`) blend. The `CPU` golden
//! [`edge_detect`](prism_render_architecture::particle::edge_detect) owns that
//! math: the two canonical `Sobel` 3x3 gradient kernels
//! ([`SOBEL_GX`](prism_render_architecture::particle::edge_detect::SOBEL_GX),
//! [`SOBEL_GY`](prism_render_architecture::particle::edge_detect::SOBEL_GY)),
//! the cheap `Roberts` cross, the `Rec. 709`
//! [`luminance`](prism_render_architecture::particle::edge_detect::luminance),
//! the `Sobel`
//! [`sobel_magnitude`](prism_render_architecture::particle::edge_detect::sobel_magnitude)
//! and `Roberts`
//! [`roberts_magnitude`](prism_render_architecture::particle::edge_detect::roberts_magnitude)
//! responses, the depth
//! [`depth_edge`](prism_render_architecture::particle::edge_detect::depth_edge)
//! and normal
//! [`normal_edge`](prism_render_architecture::particle::edge_detect::normal_edge)
//! responses, and the
//! [`EdgeParams`](prism_render_architecture::particle::edge_detect::EdgeParams)
//! `smoothstep` masking policy.
//!
//! [`GpuEdgeDetect`] is the on-device twin that runs **one thread per output
//! pixel** over a `width * height` screen-space frame and reproduces every one
//! of those responses in a single dispatch. A passing real-device parity test
//! is therefore direct evidence the ported kernel gathers the same `3x3`
//! neighborhood, clamps the same edges, convolves the same kernels and
//! thresholds through the same `smoothstep` knee the reference does, not merely
//! that the shader compiles.
//!
//! # What is twinned
//!
//! Each output pixel packs five scalar responses computed from the clamped
//! neighborhood of the color, depth and normal frames:
//!
//! 1. `luma_sobel` — the `Sobel` magnitude of the `Rec. 709` `luminance` of the
//!    `3x3` color window (the canonical `luminance` edge response).
//! 2. `luma_roberts` — the `Roberts` cross magnitude of the `2x2` `luminance`
//!    footprint anchored at the pixel.
//! 3. `depth_edge` — the `Sobel` magnitude of the `3x3` depth window.
//! 4. `normal_edge` — `max(1 - dot(n_center, n_neighbor))` over the eight `3x3`
//!    normal neighbors.
//! 5. `mask` — the [`EdgeParams`](prism_render_architecture::particle::edge_detect::EdgeParams)
//!    `smoothstep` edge mask applied to `luma_sobel`.
//!
//! The hand-rolled gather mirrors the reference exactly: the `3x3` window is
//! laid out row-major (column maps to the `x` offset, row to the `y` offset) so
//! [`SOBEL_GX`](prism_render_architecture::particle::edge_detect::SOBEL_GX)
//! differences the left/right columns and
//! [`SOBEL_GY`](prism_render_architecture::particle::edge_detect::SOBEL_GY) the
//! top/bottom rows; out-of-bounds taps clamp to the nearest edge pixel exactly
//! as the host-side gather does; the `luminance` is the same explicit
//! `dot(rgb, [0.2126, 0.7152, 0.0722])`; and the mask reproduces the reference
//! `smoothstep` (`t * t * (3 - 2 t)`) with the same degenerate-span hard-step
//! guard.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `abs`, `floor`, `sqrt`, `+ - * /` and unsigned index arithmetic —
//! with no `sin`, `cos`, `exp`, `log`, `pow`, `tan` or optional device feature,
//! so it runs unmodified on `Metal`, `Vulkan` and `DX12`. In particular
//! `normal_edge` is `1 - dot`, a plain dot-product difference, so it stays in
//! the portable subset: there is **no** `acos` angle recovery on this path.
//! The single transcendental-adjacent call is the `sqrt` the gradient
//! magnitude needs, which is in the portable subset.
//!
//! # Correctness model
//!
//! Each response is a fixed, non-reorderable sequence of multiplies, adds and
//! one `sqrt` (plus the cubic `smoothstep` for the mask), so `CPU` and `GPU`
//! evaluate the same closed form. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`), tight enough
//! to catch a genuinely wrong port (a swapped kernel sign, a dropped tap, a
//! missing edge clamp, a wrong `luminance` weight) yet loose enough to admit a
//! legal fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `Sobel`/`Roberts` edge detection plus `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::edge_detect::EdgeParams;
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
/// the sibling twins use.
const WORKGROUP_SIZE: u32 = 64;

/// Color and normal frames store three `f32` per pixel in a flat row-major
/// buffer, so a storage buffer needs no `vec3` alignment padding.
const RGB_CHANNELS: usize = 3;

/// Each output pixel packs five scalar responses: `luma_sobel`,
/// `luma_roberts`, `depth_edge`, `normal_edge` and the `mask`.
const OUT_CHANNELS: usize = 5;

/// The portable core-`WGSL` edge-detection kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `edge_detect` mirrors
/// the `CPU` golden
/// [`edge_detect`](prism_render_architecture::particle::edge_detect) response
/// by response; see the module documentation for the layout and portability
/// contract.
const EDGE_DETECT_WGSL: &str = r#"
// Screen-space edge-detection twin: one thread per output pixel gathers the
// clamped 3x3 color/depth/normal neighborhood and packs five responses --
// Sobel luminance magnitude, Roberts luminance magnitude, Sobel depth
// magnitude, the max 1 - dot(normal) crease, and the EdgeParams smoothstep mask
// applied to the luminance Sobel. It mirrors the CPU golden
// `particle::edge_detect`, uses only the portable core-WGSL subset (min/max/
// clamp/abs/floor/sqrt and + - * / plus unsigned index math), has no acos and
// no optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard Sobel/Roberts edge detection; no third-party engine
// source or derived code.

struct Params {
    // Frame extents in pixels (one thread per pixel).
    width: u32,
    height: u32,
    // EdgeParams policy: band center, half-width knee, and pre-threshold gain.
    threshold: f32,
    knee: f32,
    scale: f32,
    // Padding to a 32-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> color: array<f32>;
@group(0) @binding(2) var<storage, read> depth: array<f32>;
@group(0) @binding(3) var<storage, read> normals: array<f32>;
@group(0) @binding(4) var<storage, read_write> edges: array<f32>;

// Rec. 709 luminance weights, matching the reference `luminance` dot product.
const LUMA_R: f32 = 0.2126;
const LUMA_G: f32 = 0.7152;
const LUMA_B: f32 = 0.0722;

// smoothstep intervals narrower than this collapse to a hard step so the
// division stays defined, matching the reference `MIN_SPAN` guard.
const MIN_SPAN: f32 = 1e-6;

// Clamps a (possibly negative) integer coordinate into `[0, extent)`, the
// clamp-to-edge addressing the reference host gather uses. The host never
// dispatches an empty frame, so `extent` is always positive, but the zero guard
// is kept for exactness.
fn clamp_coord(coord: i32, extent: u32) -> u32 {
    if (extent == 0u) {
        return 0u;
    }
    if (coord < 0) {
        return 0u;
    }
    let c = u32(coord);
    let hi = extent - 1u;
    return min(c, hi);
}

fn pixel_index(x: u32, y: u32) -> u32 {
    return y * params.width + x;
}

// Reads the Rec. 709 luminance of the clamped color pixel at `(x, y)`.
fn load_luma(x: i32, y: i32) -> f32 {
    let cx = clamp_coord(x, params.width);
    let cy = clamp_coord(y, params.height);
    let base = pixel_index(cx, cy) * 3u;
    let r = color[base];
    let g = color[base + 1u];
    let b = color[base + 2u];
    return r * LUMA_R + g * LUMA_G + b * LUMA_B;
}

// Reads the clamped depth sample at `(x, y)`.
fn load_depth(x: i32, y: i32) -> f32 {
    let cx = clamp_coord(x, params.width);
    let cy = clamp_coord(y, params.height);
    return depth[pixel_index(cx, cy)];
}

// Reads the clamped normal at `(x, y)` as an RGB-packed 3-vector.
fn load_normal(x: i32, y: i32) -> vec3<f32> {
    let cx = clamp_coord(x, params.width);
    let cy = clamp_coord(y, params.height);
    let base = pixel_index(cx, cy) * 3u;
    return vec3<f32>(normals[base], normals[base + 1u], normals[base + 2u]);
}

// Hand-rolled dot product written as an explicit sum, matching the reference
// `dot3`'s evaluation order.
fn dot3(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Sobel gradient magnitude of a row-major 3x3 scalar window: convolve against
// SOBEL_GX and SOBEL_GY, then sqrt(gx*gx + gy*gy). The nine terms are summed in
// the reference row-major order (including the zero-weight center column/row)
// so the accumulation matches `convolve3x3`.
fn sobel_mag(w: array<f32, 9>) -> f32 {
    // SOBEL_GX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] row-major.
    let gx = -1.0 * w[0] + 0.0 * w[1] + 1.0 * w[2]
           + -2.0 * w[3] + 0.0 * w[4] + 2.0 * w[5]
           + -1.0 * w[6] + 0.0 * w[7] + 1.0 * w[8];
    // SOBEL_GY = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]] row-major.
    let gy = -1.0 * w[0] + -2.0 * w[1] + -1.0 * w[2]
           + 0.0 * w[3] + 0.0 * w[4] + 0.0 * w[5]
           + 1.0 * w[6] + 2.0 * w[7] + 1.0 * w[8];
    return sqrt(gx * gx + gy * gy);
}

// Gathers the clamped 3x3 luminance window around `(x, y)`, row-major: index
// row*3 + col, with column mapping to the x offset and row to the y offset.
fn gather_luma(x: i32, y: i32) -> array<f32, 9> {
    var w: array<f32, 9>;
    w[0] = load_luma(x - 1, y - 1);
    w[1] = load_luma(x,     y - 1);
    w[2] = load_luma(x + 1, y - 1);
    w[3] = load_luma(x - 1, y);
    w[4] = load_luma(x,     y);
    w[5] = load_luma(x + 1, y);
    w[6] = load_luma(x - 1, y + 1);
    w[7] = load_luma(x,     y + 1);
    w[8] = load_luma(x + 1, y + 1);
    return w;
}

// Gathers the clamped 3x3 depth window around `(x, y)` in the same layout.
fn gather_depth(x: i32, y: i32) -> array<f32, 9> {
    var w: array<f32, 9>;
    w[0] = load_depth(x - 1, y - 1);
    w[1] = load_depth(x,     y - 1);
    w[2] = load_depth(x + 1, y - 1);
    w[3] = load_depth(x - 1, y);
    w[4] = load_depth(x,     y);
    w[5] = load_depth(x + 1, y);
    w[6] = load_depth(x - 1, y + 1);
    w[7] = load_depth(x,     y + 1);
    w[8] = load_depth(x + 1, y + 1);
    return w;
}

// Roberts cross magnitude of the 2x2 luminance footprint anchored at `(x, y)`:
//   a b     a = (x, y)       b = (x+1, y)
//   c d     c = (x, y+1)     d = (x+1, y+1)
// with gx = a - d, gy = b - c, response sqrt(gx*gx + gy*gy).
fn luma_roberts(x: i32, y: i32) -> f32 {
    let a = load_luma(x, y);
    let b = load_luma(x + 1, y);
    let c = load_luma(x, y + 1);
    let d = load_luma(x + 1, y + 1);
    let gx = a - d;
    let gy = b - c;
    return sqrt(gx * gx + gy * gy);
}

// One per-neighbor normal crease term `(1 - dot(center, n)).max(0)`.
fn crease(center: vec3<f32>, n: vec3<f32>) -> f32 {
    return max(1.0 - dot3(center, n), 0.0);
}

// Normal edge response: max of `1 - dot(center, neighbor)` over the eight 3x3
// normal neighbors (center excluded), seeded with 0 like the reference fold.
fn normal_edge_at(x: i32, y: i32) -> f32 {
    let center = load_normal(x, y);
    var m: f32 = 0.0;
    m = max(m, crease(center, load_normal(x - 1, y - 1)));
    m = max(m, crease(center, load_normal(x,     y - 1)));
    m = max(m, crease(center, load_normal(x + 1, y - 1)));
    m = max(m, crease(center, load_normal(x - 1, y)));
    m = max(m, crease(center, load_normal(x + 1, y)));
    m = max(m, crease(center, load_normal(x - 1, y + 1)));
    m = max(m, crease(center, load_normal(x,     y + 1)));
    m = max(m, crease(center, load_normal(x + 1, y + 1)));
    return m;
}

fn clamp01(v: f32) -> f32 {
    return clamp(v, 0.0, 1.0);
}

// Reference `smoothstep`: 0 below edge0, 1 above edge1, cubic t*t*(3-2t)
// between, with a degenerate span collapsing to a hard step at edge0.
fn smoothstep_ref(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    if (span <= MIN_SPAN) {
        if (x < edge0) {
            return 0.0;
        }
        return 1.0;
    }
    let t = clamp01((x - edge0) / span);
    return t * t * (3.0 - 2.0 * t);
}

// EdgeParams edge mask: scale the raw magnitude, then smoothstep over the band
// [threshold - knee, threshold + knee], matching `EdgeParams::edge_mask`.
fn edge_mask(mag: f32) -> f32 {
    let scaled = mag * params.scale;
    let lo = params.threshold - params.knee;
    let hi = params.threshold + params.knee;
    return smoothstep_ref(lo, hi, scaled);
}

@compute @workgroup_size(64)
fn edge_detect(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.width * params.height;
    if (idx >= total) {
        return;
    }
    let x = i32(idx % params.width);
    let y = i32(idx / params.width);

    let luma_sobel = sobel_mag(gather_luma(x, y));
    let luma_rob = luma_roberts(x, y);
    let dep_edge = sobel_mag(gather_depth(x, y));
    let nrm_edge = normal_edge_at(x, y);
    let mask = edge_mask(luma_sobel);

    let base = idx * 5u;
    edges[base] = luma_sobel;
    edges[base + 1u] = luma_rob;
    edges[base + 2u] = dep_edge;
    edges[base + 3u] = nrm_edge;
    edges[base + 4u] = mask;
}
"#;

/// A screen-space frame of co-located color, depth and normal samples the edge
/// detector consumes.
///
/// `color` and `normal` hold one `[f32; 3]` per pixel and `depth` one `f32` per
/// pixel, all row-major and all `width * height` long. Derives only
/// [`PartialEq`] (no [`Eq`]/[`Hash`]) because the samples are `f32`.
#[derive(Clone, Debug, PartialEq)]
pub struct EdgeFrame {
    /// Frame width in pixels.
    pub width: usize,
    /// Frame height in pixels.
    pub height: usize,
    /// Row-major linear `RGB` color samples, `width * height` long.
    pub color: Vec<[f32; 3]>,
    /// Row-major depth samples, `width * height` long.
    pub depth: Vec<f32>,
    /// Row-major unit-ish normal samples, `width * height` long.
    pub normal: Vec<[f32; 3]>,
}

impl EdgeFrame {
    /// Builds a frame from its color, depth and normal planes.
    ///
    /// # Panics
    ///
    /// Panics if any plane length differs from `width * height`.
    #[must_use]
    pub fn new(
        width: usize,
        height: usize,
        color: Vec<[f32; 3]>,
        depth: Vec<f32>,
        normal: Vec<[f32; 3]>,
    ) -> EdgeFrame {
        let texels = width.saturating_mul(height);
        assert_eq!(color.len(), texels, "color plane must be width * height");
        assert_eq!(depth.len(), texels, "depth plane must be width * height");
        assert_eq!(normal.len(), texels, "normal plane must be width * height");
        EdgeFrame {
            width,
            height,
            color,
            depth,
            normal,
        }
    }

    /// Whether the frame has no pixels.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }
}

/// One edge-detection request: the source frame plus the
/// [`EdgeParams`](prism_render_architecture::particle::edge_detect::EdgeParams)
/// masking policy.
#[derive(Clone, Debug, PartialEq)]
pub struct EdgeDetectQuery {
    /// The source color/depth/normal frame.
    pub frame: EdgeFrame,
    /// The masking policy applied to the `luminance` `Sobel` magnitude.
    pub params: EdgeParams,
}

/// The five scalar edge responses computed for one pixel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeResponse {
    /// `Sobel` magnitude of the `3x3` `luminance` window.
    pub luma_sobel: f32,
    /// `Roberts` cross magnitude of the `2x2` `luminance` footprint.
    pub luma_roberts: f32,
    /// `Sobel` magnitude of the `3x3` depth window.
    pub depth_edge: f32,
    /// Max `1 - dot(normal)` crease over the eight `3x3` normal neighbors.
    pub normal_edge: f32,
    /// [`EdgeParams`](prism_render_architecture::particle::edge_detect::EdgeParams)
    /// `smoothstep` mask applied to [`EdgeResponse::luma_sobel`].
    pub mask: f32,
}

/// A full frame of per-pixel [`EdgeResponse`] values, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct EdgeDetectOutput {
    /// Frame width in pixels.
    pub width: usize,
    /// Frame height in pixels.
    pub height: usize,
    /// Row-major per-pixel responses, `width * height` long.
    pub pixels: Vec<EdgeResponse>,
}

/// A compiled, reusable edge-detection pipeline.
pub struct GpuEdgeDetect {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuEdgeDetect {
    /// Compiles the edge-detection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuEdgeDetect {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_edge_detect"),
            source: ShaderSource::Wgsl(EDGE_DETECT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_edge_detect_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_edge_detect_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_edge_detect_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("edge_detect"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuEdgeDetect {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs the edge detector on `query.frame`, returning one [`EdgeResponse`]
    /// per pixel.
    ///
    /// Each response matches the `CPU` golden
    /// [`edge_detect`](prism_render_architecture::particle::edge_detect)
    /// primitives applied to the identically clamped neighborhood, to within
    /// the tolerance documented on this module. An empty frame returns an empty
    /// output with no dispatch issued (a storage buffer cannot be zero-sized).
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, query: &EdgeDetectQuery) -> EdgeDetectOutput {
        let frame = &query.frame;
        if frame.is_empty() {
            return EdgeDetectOutput {
                width: frame.width,
                height: frame.height,
                pixels: Vec::new(),
            };
        }
        let device = ctx.device();
        let texels = frame.width * frame.height;

        let color_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_edge_detect_color"),
            contents: bytemuck::cast_slice(&flatten_rgb(&frame.color)),
            usage: BufferUsages::STORAGE,
        });
        let depth_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_edge_detect_depth"),
            contents: bytemuck::cast_slice(&frame.depth),
            usage: BufferUsages::STORAGE,
        });
        let normal_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_edge_detect_normal"),
            contents: bytemuck::cast_slice(&flatten_rgb(&frame.normal)),
            usage: BufferUsages::STORAGE,
        });
        let out_bytes = (texels * OUT_CHANNELS * size_of::<f32>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_edge_detect_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = GpuParams::new(frame.width, frame.height, query.params);
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_edge_detect_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_edge_detect_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: color_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: depth_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: normal_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_edge_detect_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_edge_detect_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_edge_detect_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per output pixel, flattened to a 1-D dispatch.
            let groups = (texels as u32).div_ceil(WORKGROUP_SIZE);
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

        let mut pixels: Vec<EdgeResponse> = Vec::with_capacity(texels);
        for chunk in flat.chunks_exact(OUT_CHANNELS) {
            pixels.push(EdgeResponse {
                luma_sobel: chunk[0],
                luma_roberts: chunk[1],
                depth_edge: chunk[2],
                normal_edge: chunk[3],
                mask: chunk[4],
            });
        }
        debug_assert_eq!(pixels.len(), texels);

        EdgeDetectOutput {
            width: frame.width,
            height: frame.height,
            pixels,
        }
    }
}

/// Flattens a per-pixel `[f32; 3]` plane into the row-major, 3-per-pixel `f32`
/// layout the device storage buffers expect.
fn flatten_rgb(plane: &[[f32; 3]]) -> Vec<f32> {
    let mut data = Vec::with_capacity(plane.len() * RGB_CHANNELS);
    for triple in plane {
        data.push(triple[0]);
        data.push(triple[1]);
        data.push(triple[2]);
    }
    data
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

/// Uniform parameters for the dispatch. `repr(C)` `std140` layout matching
/// `Params` in [`EDGE_DETECT_WGSL`]: the frame extents, the three
/// [`EdgeParams`](prism_render_architecture::particle::edge_detect::EdgeParams)
/// scalars and three pad words — `32` bytes, each field at the uniform offset
/// the shader expects.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Frame width in pixels.
    width: u32,
    /// Frame height in pixels.
    height: u32,
    /// Magnitude at the center of the `smoothstep` band.
    threshold: f32,
    /// Half-width of the soft `knee` band.
    knee: f32,
    /// Gain applied to the raw magnitude before thresholding.
    scale: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

impl GpuParams {
    /// Packs the frame extents and the `EdgeParams` policy into the uniform
    /// block. The policy's stored (already non-negative-clamped) `knee` and
    /// `scale` are forwarded verbatim.
    fn new(width: usize, height: usize, params: EdgeParams) -> GpuParams {
        GpuParams {
            width: width as u32,
            height: height as u32,
            threshold: params.threshold,
            knee: params.knee,
            scale: params.scale,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        }
    }
}
