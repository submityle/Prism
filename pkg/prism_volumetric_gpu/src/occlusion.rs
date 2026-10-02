//! `wgpu` compute twin of the hierarchical `Z`-buffer (`HZB`) occlusion-culling
//! golden
//! ([`occlusion`](prism_render_architecture::particle::occlusion), design §13).
//!
//! The particle sort-and-cull layer rejects a candidate particle system when a
//! max-depth `HZB` proves its whole screen footprint sits strictly behind
//! previously rasterized geometry. The `CPU` golden
//! [`is_occluded`](prism_render_architecture::particle::occlusion::is_occluded)
//! owns that verdict, and the surrounding
//! [`HzbPyramid`](prism_render_architecture::particle::occlusion::HzbPyramid)
//! and
//! [`ScreenRect`](prism_render_architecture::particle::occlusion::ScreenRect)
//! accessors own the `mip`-selection and footprint arithmetic that feed it.
//! [`GpuOcclusion`] is the on-device twin that runs one thread per
//! [`GpuOcclusionQuery`] and reproduces every lane. A passing real-device
//! parity test is therefore direct evidence the ported kernel folds the same
//! integer `mip` chain, the same tolerant depth comparison and the same
//! viewport clamp the reference does, not merely that its shader compiles.
//!
//! # What is twinned
//!
//! The kernel reproduces, per query and guard-for-guard, the per-thread half of
//! the golden contract:
//! [`HzbPyramid::mip_count`](prism_render_architecture::particle::occlusion::HzbPyramid::mip_count),
//! [`HzbPyramid::mip_size`](prism_render_architecture::particle::occlusion::HzbPyramid::mip_size),
//! [`HzbPyramid::mip_texel_count`](prism_render_architecture::particle::occlusion::HzbPyramid::mip_texel_count),
//! [`HzbPyramid::select_mip`](prism_render_architecture::particle::occlusion::HzbPyramid::select_mip),
//! [`ScreenRect::width`](prism_render_architecture::particle::occlusion::ScreenRect::width),
//! [`ScreenRect::height`](prism_render_architecture::particle::occlusion::ScreenRect::height),
//! [`ScreenRect::is_valid`](prism_render_architecture::particle::occlusion::ScreenRect::is_valid),
//! [`ScreenRect::clamp_to`](prism_render_architecture::particle::occlusion::ScreenRect::clamp_to),
//! [`is_occluded`](prism_render_architecture::particle::occlusion::is_occluded)
//! and
//! [`conservative_false_negative_bound`](prism_render_architecture::particle::occlusion::conservative_false_negative_bound).
//!
//! The host-side `u64` aggregation
//! [`HzbPyramid::total_texel_count`](prism_render_architecture::particle::occlusion::HzbPyramid::total_texel_count)
//! is deliberately *not* twinned: it sums every `mip` level into a `u64`, and
//! `WGSL` has no `u64` type and no per-thread reduction over the pyramid, so it
//! stays a `CPU` aggregation rather than a one-thread-one-element kernel lane.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `+ - * /`, integer `>>`, `<<` and `u32` comparison — with no
//! transcendental, no `u64` and no optional device feature. The `mip` chain is
//! counted with integer shifts exactly as the reference does, so it runs
//! unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! Every integer output (`mip` count, `mip` size, texel count, selected `mip`,
//! the false-negative bound) and every discrete flag (rectangle validity and
//! the occlusion verdict) is bit-exact against the reference: the kernel runs
//! the same integer shifts and the same `>` comparisons the `CPU` does. Only
//! the continuous rectangle extents and clamped edges are compared with a
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`), since a `GPU` may round
//! a subtraction a few units in the last place differently from the scalar
//! reference. The named fixtures place depths and rectangle edges clear of the
//! [`CMP_EPS`](prism_render_architecture::particle::occlusion::CMP_EPS) band so
//! their flags are asserted exactly.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::occlusion`；
//! standard `Hi-Z` max-depth occlusion culling plus `wgpu` compute dispatch; no
//! third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::occlusion::{
    conservative_false_negative_bound, is_occluded, HzbPyramid, OcclusionQuery, ScreenRect,
};
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
/// shared by every kernel in this crate.
const WORKGROUP_SIZE: u32 = 64;

/// Discrete true code written by the kernel for a boolean lane: matches the
/// host `== 1` decode in [`decode_result`]. A direct `f32` equality is
/// forbidden, so the kernel emits an integer flag rather than a sentinel float.
const CODE_TRUE: u32 = 1;

/// The portable core-`WGSL` occlusion kernel, embedded inline so the twin ships
/// as a single source file. Mirrors the `CPU` golden
/// [`occlusion`](prism_render_architecture::particle::occlusion) per-query
/// accessors guard-for-guard; see the module documentation for the algorithm.
const OCCLUSION_WGSL: &str = r#"
// HZB occlusion twin: one thread per query reproduces the per-thread half of
// the CPU golden `particle::occlusion` contract. It counts the pyramid mip
// chain with integer shifts, selects the covering mip, measures and clamps the
// screen rectangle, decides the tolerant occlusion verdict and the per-mip
// false-negative bound. It uses only the portable core-WGSL subset (min/max/
// clamp, + - * /, integer >> and <<, u32 comparison), has no u64 and takes no
// optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard Hi-Z max-depth occlusion culling; no third-party engine
// source or derived code.

struct Params {
    // Number of valid queries in `queries`.
    count: u32,
    // Padding to a 16-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. 64-byte std430 stride matching the host `GpuQuery`: the screen
// rectangle edges, the clamp bounds packed with the two depths, and two u32
// lanes carrying the pyramid base extent and the integer query selectors.
struct Query {
    // min_x, min_y, max_x, max_y.
    rect: vec4<f32>,
    // clamp_w, clamp_h, nearest_depth, hzb_sampled_depth.
    clamp_depth: vec4<f32>,
    // base_width, base_height, mip_query_level, select_rect_w.
    dims: vec4<u32>,
    // select_rect_h, fnb_mip_level, pad, pad.
    extra: vec4<u32>,
}

// One result. 64-byte std430 stride matching the host `GpuResult`: the integer
// mip and bound outputs and the two discrete flags as 0u/1u, followed by the
// continuous rectangle extents and clamped edges.
struct Result {
    mip_count: u32,
    mip_w: u32,
    mip_h: u32,
    mip_texel_count: u32,
    selected_mip: u32,
    rect_valid: u32,
    clamped_valid: u32,
    is_occluded: u32,
    false_negative_bound: u32,
    rect_width: f32,
    rect_height: f32,
    clamp_min_x: f32,
    clamp_min_y: f32,
    clamp_max_x: f32,
    clamp_max_y: f32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Tolerant-comparison epsilon mirroring the reference `CMP_EPS`. A direct f32
// `==`/`!=` is forbidden, so the occlusion test compares against this floor.
const CMP_EPS: f32 = 1.0e-6;

// Counts the pyramid levels including the base, mirroring `HzbPyramid::mip_count`:
// the larger base dimension is repeatedly halved (floored at 1) until it reaches
// 1, counting one level per halving plus the final 1x1 level.
fn mip_count_of(bw: u32, bh: u32) -> u32 {
    var dim = max(bw, bh);
    var levels = 1u;
    loop {
        if (dim <= 1u) {
            break;
        }
        dim = max(dim / 2u, 1u);
        levels = levels + 1u;
    }
    return levels;
}

// Saturating u32 multiply mirroring `HzbPyramid::mip_texel_count`'s
// `saturating_mul`: a product that would wrap is clamped to u32::MAX instead.
fn sat_mul(a: u32, b: u32) -> u32 {
    if (a == 0u || b == 0u) {
        return 0u;
    }
    let limit = 0xffffffffu;
    if (a > limit / b) {
        return limit;
    }
    return a * b;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let q = queries[idx];
    // `HzbPyramid::new` clamps each base dimension up to at least 1; replay it.
    let bw = max(q.dims.x, 1u);
    let bh = max(q.dims.y, 1u);
    let mip_query_level = q.dims.z;
    let select_rect_w = q.dims.w;
    let select_rect_h = q.extra.x;
    let fnb_mip_level = q.extra.y;

    let min_x = q.rect.x;
    let min_y = q.rect.y;
    let max_x = q.rect.z;
    let max_y = q.rect.w;

    let clamp_w_in = q.clamp_depth.x;
    let clamp_h_in = q.clamp_depth.y;
    let nearest_depth = q.clamp_depth.z;
    let hzb_sampled_depth = q.clamp_depth.w;

    var res: Result;
    res.pad0 = 0u;

    // mip_count.
    let levels = mip_count_of(bw, bh);
    res.mip_count = levels;

    // mip_size(mip_query_level): each dimension is max(1, base >> level), with
    // the shift clamped to 31 so a huge level saturates to the 1x1 tail.
    let shift = min(mip_query_level, 31u);
    let mw = max(bw >> shift, 1u);
    let mh = max(bh >> shift, 1u);
    res.mip_w = mw;
    res.mip_h = mh;

    // mip_texel_count(mip_query_level).
    res.mip_texel_count = sat_mul(mw, mh);

    // select_mip(select_rect_w, select_rect_h): ceil-log2 of the longer side by
    // counting halvings to a single texel, clamped into 0..=mip_count-1.
    var span = max(max(select_rect_w, select_rect_h), 1u);
    var level = 0u;
    loop {
        if (span <= 1u) {
            break;
        }
        span = max(span / 2u, 1u);
        level = level + 1u;
    }
    let max_level = levels - 1u;
    res.selected_mip = min(level, max_level);

    // ScreenRect::width / height, clamped non-negative.
    let width = max(max_x - min_x, 0.0);
    let height = max(max_y - min_y, 0.0);
    res.rect_width = width;
    res.rect_height = height;

    // ScreenRect::is_valid: strict `>` on both axes, no f32 equality.
    var valid = 0u;
    if (max_x > min_x && max_y > min_y) {
        valid = 1u;
    }
    res.rect_valid = valid;

    // ScreenRect::clamp_to([0, w] x [0, h]).
    let cw = max(clamp_w_in, 0.0);
    let ch = max(clamp_h_in, 0.0);
    let cmin_x = clamp(min_x, 0.0, cw);
    let cmin_y = clamp(min_y, 0.0, ch);
    let cmax_x = clamp(max_x, 0.0, cw);
    let cmax_y = clamp(max_y, 0.0, ch);
    res.clamp_min_x = cmin_x;
    res.clamp_min_y = cmin_y;
    res.clamp_max_x = cmax_x;
    res.clamp_max_y = cmax_y;

    var clamped_valid = 0u;
    if (cmax_x > cmin_x && cmax_y > cmin_y) {
        clamped_valid = 1u;
    }
    res.clamped_valid = clamped_valid;

    // is_occluded: provably occluded only when strictly behind by more than
    // CMP_EPS; boundary equality stays visible, preserving conservatism.
    var occ = 0u;
    if (nearest_depth > hzb_sampled_depth + CMP_EPS) {
        occ = 1u;
    }
    res.is_occluded = occ;

    // conservative_false_negative_bound: 1 << mip_level base texels, saturating
    // at u32::MAX for an extreme mip.
    if (fnb_mip_level >= 31u) {
        res.false_negative_bound = 0xffffffffu;
    } else {
        res.false_negative_bound = 1u << fnb_mip_level;
    }

    results[idx] = res;
}
"#;

/// One occlusion query bundling every input the per-thread kernel reads.
///
/// A single lane evaluates the whole per-query contract: it derives the `mip`
/// chain of `pyramid`, selects the `mip` covering a `select_rect_w` x
/// `select_rect_h` footprint, measures and clamps `rect` into
/// `clamp_width` x `clamp_height`, decides whether `nearest_depth` is occluded
/// behind `hzb_sampled_depth`, and returns the false-negative bound of
/// `fnb_mip_level`. Derives only [`PartialEq`] (no `Eq`/`Hash`) because it holds
/// `f32` geometry.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::occlusion`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuOcclusionQuery {
    /// The `HZB` pyramid whose `mip` chain and selection are evaluated.
    pub pyramid: HzbPyramid,
    /// The projected screen footprint whose extents and validity are measured.
    pub rect: ScreenRect,
    /// Viewport width the rectangle is clamped into by `clamp_to`.
    pub clamp_width: f32,
    /// Viewport height the rectangle is clamped into by `clamp_to`.
    pub clamp_height: f32,
    /// Nearest (closest-to-camera) depth of the candidate footprint.
    pub nearest_depth: f32,
    /// Sampled max-depth `HZB` texel the candidate is tested against.
    pub hzb_sampled_depth: f32,
    /// `mip` level whose `mip_size` and `mip_texel_count` are reported.
    pub mip_query_level: u32,
    /// Footprint width fed to `select_mip`.
    pub select_rect_w: u32,
    /// Footprint height fed to `select_mip`.
    pub select_rect_h: u32,
    /// `mip` level whose `conservative_false_negative_bound` is reported.
    pub fnb_mip_level: u32,
}

/// The resolved per-query verdict, the host-side mirror of the kernel's
/// `Result` lane.
///
/// Every integer field and both booleans are bit-exact against the reference;
/// `rect_width`, `rect_height` and the `clamped_rect` edges are continuous and
/// compared with a tolerance by the parity test. Derives only [`PartialEq`] (no
/// `Eq`/`Hash`) because it holds `f32` extents.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::occlusion`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuOcclusionResult {
    /// Pyramid level count including the base, from `mip_count`.
    pub mip_count: u32,
    /// `(width, height)` of `mip_query_level`, from `mip_size`.
    pub mip_size: (u32, u32),
    /// Texel count of `mip_query_level`, from `mip_texel_count`.
    pub mip_texel_count: u32,
    /// Covering `mip` for the `select_rect_w` x `select_rect_h` footprint.
    pub selected_mip: u32,
    /// Non-negative rectangle width, from `ScreenRect::width`.
    pub rect_width: f32,
    /// Non-negative rectangle height, from `ScreenRect::height`.
    pub rect_height: f32,
    /// Whether the input rectangle has strictly positive area.
    pub rect_valid: bool,
    /// The rectangle clamped into `[0, clamp_width] x [0, clamp_height]`.
    pub clamped_rect: ScreenRect,
    /// Whether the clamped rectangle still has strictly positive area.
    pub clamped_valid: bool,
    /// Whether the candidate is provably occluded, from `is_occluded`.
    pub occluded: bool,
    /// Worst-case per-edge false-negative bound in base texels.
    pub false_negative_bound: u32,
}

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`OCCLUSION_WGSL`]: the query count and three pad words — `16`
/// bytes, each field at the uniform offset the shader expects.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One query as uploaded. `64`-byte `std430` stride matching `Query` in the
/// shader: the rectangle edges, the clamp bounds packed with the two depths and
/// two `u32` lanes carrying the pyramid base extent and integer selectors.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// `min_x`, `min_y`, `max_x`, `max_y`.
    rect: [f32; 4],
    /// `clamp_width`, `clamp_height`, `nearest_depth`, `hzb_sampled_depth`.
    clamp_depth: [f32; 4],
    /// `base_width`, `base_height`, `mip_query_level`, `select_rect_w`.
    dims: [u32; 4],
    /// `select_rect_h`, `fnb_mip_level`, pad, pad.
    extra: [u32; 4],
}

impl GpuQuery {
    /// Packs a [`GpuOcclusionQuery`] into the `std430` upload layout.
    fn from_query(query: &GpuOcclusionQuery) -> GpuQuery {
        GpuQuery {
            rect: [
                query.rect.min_x,
                query.rect.min_y,
                query.rect.max_x,
                query.rect.max_y,
            ],
            clamp_depth: [
                query.clamp_width,
                query.clamp_height,
                query.nearest_depth,
                query.hzb_sampled_depth,
            ],
            dims: [
                query.pyramid.base_width(),
                query.pyramid.base_height(),
                query.mip_query_level,
                query.select_rect_w,
            ],
            extra: [query.select_rect_h, query.fnb_mip_level, 0, 0],
        }
    }
}

/// One result as read back. `64`-byte `std430` stride matching `Result` in the
/// shader: the integer `mip` and bound outputs, the two discrete flags as
/// `0`/`1`, then the continuous rectangle extents and clamped edges.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Pyramid level count.
    mip_count: u32,
    /// `mip_query_level` width.
    mip_w: u32,
    /// `mip_query_level` height.
    mip_h: u32,
    /// `mip_query_level` texel count.
    mip_texel_count: u32,
    /// Selected covering `mip`.
    selected_mip: u32,
    /// Input-rectangle validity flag (`1` = valid).
    rect_valid: u32,
    /// Clamped-rectangle validity flag (`1` = valid).
    clamped_valid: u32,
    /// Occlusion verdict flag (`1` = occluded).
    is_occluded: u32,
    /// Per-`mip` false-negative bound in base texels.
    false_negative_bound: u32,
    /// Non-negative rectangle width.
    rect_width: f32,
    /// Non-negative rectangle height.
    rect_height: f32,
    /// Clamped `min_x`.
    clamp_min_x: f32,
    /// Clamped `min_y`.
    clamp_min_y: f32,
    /// Clamped `max_x`.
    clamp_max_x: f32,
    /// Clamped `max_y`.
    clamp_max_y: f32,
    /// Padding word.
    pad0: u32,
}

/// Maps one kernel `Result` lane back to the host [`GpuOcclusionResult`].
fn decode_result(raw: &GpuResult) -> GpuOcclusionResult {
    GpuOcclusionResult {
        mip_count: raw.mip_count,
        mip_size: (raw.mip_w, raw.mip_h),
        mip_texel_count: raw.mip_texel_count,
        selected_mip: raw.selected_mip,
        rect_width: raw.rect_width,
        rect_height: raw.rect_height,
        rect_valid: raw.rect_valid == CODE_TRUE,
        clamped_rect: ScreenRect::new(
            raw.clamp_min_x,
            raw.clamp_min_y,
            raw.clamp_max_x,
            raw.clamp_max_y,
        ),
        clamped_valid: raw.clamped_valid == CODE_TRUE,
        occluded: raw.is_occluded == CODE_TRUE,
        false_negative_bound: raw.false_negative_bound,
    }
}

/// The `CPU` golden verdict for one query, dispatching to the reference entry
/// points so callers (and the parity test) can pin the twin lane for lane.
///
/// Evaluates
/// [`HzbPyramid::mip_count`](prism_render_architecture::particle::occlusion::HzbPyramid::mip_count),
/// [`HzbPyramid::mip_size`](prism_render_architecture::particle::occlusion::HzbPyramid::mip_size),
/// [`HzbPyramid::mip_texel_count`](prism_render_architecture::particle::occlusion::HzbPyramid::mip_texel_count),
/// [`HzbPyramid::select_mip`](prism_render_architecture::particle::occlusion::HzbPyramid::select_mip),
/// the [`ScreenRect`] accessors,
/// [`is_occluded`](prism_render_architecture::particle::occlusion::is_occluded)
/// and
/// [`conservative_false_negative_bound`](prism_render_architecture::particle::occlusion::conservative_false_negative_bound)
/// on `query`.
#[must_use]
pub fn cpu_reference(query: &GpuOcclusionQuery) -> GpuOcclusionResult {
    let pyramid = query.pyramid;
    let clamped = query.rect.clamp_to(query.clamp_width, query.clamp_height);
    let occ_query = OcclusionQuery {
        rect: query.rect,
        nearest_depth: query.nearest_depth,
    };
    GpuOcclusionResult {
        mip_count: pyramid.mip_count(),
        mip_size: pyramid.mip_size(query.mip_query_level),
        mip_texel_count: pyramid.mip_texel_count(query.mip_query_level),
        selected_mip: pyramid.select_mip(query.select_rect_w, query.select_rect_h),
        rect_width: query.rect.width(),
        rect_height: query.rect.height(),
        rect_valid: query.rect.is_valid(),
        clamped_rect: clamped,
        clamped_valid: clamped.is_valid(),
        occluded: is_occluded(&occ_query, query.hzb_sampled_depth),
        false_negative_bound: conservative_false_negative_bound(query.fnb_mip_level),
    }
}

/// A compiled, reusable `HZB` occlusion pipeline.
pub struct GpuOcclusion {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuOcclusion {
    /// Compiles the occlusion kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuOcclusion {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_occlusion"),
            source: ShaderSource::Wgsl(OCCLUSION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_occlusion_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_occlusion_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_occlusion_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuOcclusion {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries`, returning one [`GpuOcclusionResult`] per
    /// query in input order.
    ///
    /// The returned result for query `q` mirrors [`cpu_reference`] evaluated on
    /// `q`. An empty `queries` slice yields an empty result — storage buffers
    /// cannot be zero-sized, so it is handled by an early return before any
    /// dispatch.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[GpuOcclusionQuery]) -> Vec<GpuOcclusionResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = GpuParams {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let gpu_queries: Vec<GpuQuery> = queries.iter().map(GpuQuery::from_query).collect();

        let out_bytes = (queries.len() * size_of::<GpuResult>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_occlusion_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_occlusion_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_occlusion_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_occlusion_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_occlusion_bind_group"),
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
            label: Some("prism_volumetric_occlusion_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_occlusion_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
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
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), queries.len());

        raw.iter().map(decode_result).collect()
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
