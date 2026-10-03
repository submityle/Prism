//! `wgpu` compute twin of the `ReSTIR` DI temporal-reprojection admissibility
//! gate (`reproject_history` in
//! [`prism_render_architecture::lighting::restir_temporal`]).
//!
//! The `CPU` golden `reproject_history` validates and `M`-caps a reprojected
//! history reservoir for the current pixel. Given last frame's reservoir (plus
//! the surface it was produced on) and this pixel's surface, it accepts the
//! history only when it is non-empty, both surfaces are valid (finite, strictly
//! positive view depth), their view depths agree to within
//! `depth_rel_tolerance * current.view_depth`, and their normals agree to
//! within `normal_cos_tolerance`. On acceptance the held reservoir is returned
//! with its folded count `m` clamped to `max_history_m` (`cap_history`), leaving
//! `sample`, `w_sum`, `w` and `target_pdf` untouched; on rejection (an empty
//! history or a geometry mismatch / disocclusion) it returns `None`.
//!
//! [`GpuRestirTemporalReproject`] is the on-device twin: one thread evaluates
//! one history-versus-current pair, reproducing the reference decision exactly,
//! so a passing real-device parity test is direct evidence the ported kernel
//! gates and caps history the same way the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! For one pair the twin reads the history reservoir (`sample`, `w_sum`, `m`,
//! `w`, `target_pdf`), the history and current surfaces (`view_depth` plus unit
//! `normal`), and the tolerances (`max_history_m`, `depth_rel_tolerance`,
//! `normal_cos_tolerance`). It emits a `hit` flag plus the reservoir fields the
//! reference `reproject_history` would return (the `m`-capped reservoir on
//! acceptance, all-zero on rejection), mirroring the reference branch order:
//! empty check, validity, relative depth test, normal agreement, then the
//! `cap_history` clamp.
//!
//! # What stays on the host
//!
//! The surrounding temporal pipeline: the motion-vector texture fetch that maps
//! this pixel to a previous-frame texel (a `GPU` texture op the caller performs,
//! handing us the already-reprojected history), the initial `RIS` streaming,
//! the unbiased combine that folds the admissible history into the current
//! reservoir, and the empty-batch short-circuit (a storage buffer cannot be
//! zero-sized). The twin owns only the fixed-width per-pixel gate and clamp.
//!
//! # Correctness model
//!
//! The decision is a chain of ordered comparisons over `abs`, a multiply, and a
//! dot product; the accept / reject outcome and the integer count are therefore
//! exact and asserted with `==`. The returned `w_sum`, `w` and `target_pdf` are
//! verbatim copies of the input (`cap_history` touches only the count), so they
//! are bit-identical passthroughs; the parity test still admits the documented
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3` tolerance on them to stay robust.
//! Fixtures are held a clear margin away from the depth and cosine thresholds
//! (rejection sampling) so a last-place `GPU` multiply cannot flip the gate.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, one
//! multiply, a three-term dot product, and ordered comparisons — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `sqrt`, no inverse trigonometry, no `round`, and
//! no bare f32 equality. Finiteness is tested with an ordered compare against
//! the largest finite f32, so no infinity is constructed. No optional device
//! feature is required, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//! There is no loop: each thread performs a fixed, bounded sequence, so the
//! kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::lighting::restir_temporal`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` temporal-reprojection admissibility kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden `reproject_history`; see the module
/// documentation for the algorithm.
const RESTIR_TEMPORAL_REPROJECT_WGSL: &str = r#"
// ReSTIR temporal-reprojection twin: one thread decides whether one reprojected
// history reservoir is admissible for the current pixel (non-empty, both
// surfaces valid, close depth, agreeing normal) and, on acceptance, returns the
// held reservoir with its count M clamped to max_history_m, mirroring the CPU
// golden `lighting::restir_temporal::reproject_history` with only abs, min,
// + - * and ordered comparisons. It owns no motion-vector fetch, no RIS
// streaming and no combine; those stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::lighting::restir_temporal；无第三方
// 引擎源码或衍生代码。

// Largest finite f32. A positive depth is finite iff it is not greater than
// this, so `d > 0.0 && !(d > F32_MAX)` reproduces `d > 0 && d.is_finite()`
// without constructing an infinity: NaN fails `d > 0.0`, +inf fails
// `!(d > F32_MAX)`.
const F32_MAX: f32 = 3.40282347e38;

struct Params {
    // Number of pairs in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // History reservoir: held light index, running weight sum, folded count,
    // finalized contribution weight, and held-sample target density.
    h_sample: u32,
    h_w_sum: f32,
    h_m: u32,
    h_w: f32,
    h_target_pdf: f32,
    // History surface: view depth and unit normal.
    h_depth: f32,
    h_nx: f32,
    h_ny: f32,
    h_nz: f32,
    // Current surface: view depth and unit normal.
    c_depth: f32,
    c_nx: f32,
    c_ny: f32,
    c_nz: f32,
    // Upper bound on the reprojected history count M.
    max_history_m: u32,
    // Relative view-depth tolerance.
    depth_rel_tol: f32,
    // Minimum normal agreement cosine.
    normal_cos_tol: f32,
}

struct Result {
    // 1 when the history is admissible (Some), 0 otherwise (None).
    hit: u32,
    // Returned reservoir fields (M-capped on a hit, all-zero on a miss).
    out_sample: u32,
    out_m: u32,
    pad0: u32,
    out_w_sum: f32,
    out_w: f32,
    out_target_pdf: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Whether a view depth is a real, in-front-of-camera hit: finite and strictly
// positive, matching `SurfaceGeometry::is_valid`.
fn is_valid(d: f32) -> bool {
    return (d > 0.0) && !(d > F32_MAX);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var hit: u32 = 0u;
    var out_sample: u32 = 0u;
    var out_m: u32 = 0u;
    var out_w_sum: f32 = 0.0;
    var out_w: f32 = 0.0;
    var out_target_pdf: f32 = 0.0;

    // Reference branch order: empty history, surface validity, relative depth,
    // normal agreement, then the cap_history clamp.
    if (q.h_m != 0u) {
        if (is_valid(q.c_depth) && is_valid(q.h_depth)) {
            let depth_diff = abs(q.h_depth - q.c_depth);
            let depth_bound = q.depth_rel_tol * q.c_depth;
            if (!(depth_diff > depth_bound)) {
                let dotn = q.h_nx * q.c_nx + q.h_ny * q.c_ny + q.h_nz * q.c_nz;
                if (dotn >= q.normal_cos_tol) {
                    hit = 1u;
                    out_sample = q.h_sample;
                    // cap_history: clamp the folded count M to max_history_m,
                    // leaving the weights and target density untouched.
                    out_m = min(q.h_m, q.max_history_m);
                    out_w_sum = q.h_w_sum;
                    out_w = q.h_w;
                    out_target_pdf = q.h_target_pdf;
                }
            }
        }
    }

    var out: Result;
    out.hit = hit;
    out.out_sample = out_sample;
    out.out_m = out_m;
    out.pad0 = 0u;
    out.out_w_sum = out_w_sum;
    out.out_w = out_w;
    out.out_target_pdf = out_target_pdf;
    out.pad1 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the pair count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`RESTIR_TEMPORAL_REPROJECT_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid pairs in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one reprojection query: the history reservoir,
/// the history and current surfaces, and the tolerances, matching the `WGSL`
/// `Query` struct. The `16` scalar words already fill a `16`-byte-aligned
/// `64`-byte stride, so no trailing pad word is needed.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// History held light index `sample`.
    h_sample: u32,
    /// History running weight sum `w_sum`.
    h_w_sum: f32,
    /// History folded candidate count `m`.
    h_m: u32,
    /// History finalized contribution weight `w`.
    h_w: f32,
    /// History held-sample target density `target_pdf`.
    h_target_pdf: f32,
    /// History surface view depth.
    h_depth: f32,
    /// History surface normal `x`.
    h_nx: f32,
    /// History surface normal `y`.
    h_ny: f32,
    /// History surface normal `z`.
    h_nz: f32,
    /// Current surface view depth.
    c_depth: f32,
    /// Current surface normal `x`.
    c_nx: f32,
    /// Current surface normal `y`.
    c_ny: f32,
    /// Current surface normal `z`.
    c_nz: f32,
    /// Upper bound on the reprojected history count `max_history_m`.
    max_history_m: u32,
    /// Relative view-depth tolerance `depth_rel_tolerance`.
    depth_rel_tol: f32,
    /// Minimum normal agreement cosine `normal_cos_tolerance`.
    normal_cos_tol: f32,
}

/// `repr(C)` `std430` layout of one reprojection result, matching the `WGSL`
/// `Result` struct: the `hit` flag, the `M`-capped reservoir fields, and two
/// pad words to a `16`-byte-aligned `32`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `1` when the history is admissible, `0` otherwise.
    hit: u32,
    /// Returned held light index `sample`.
    out_sample: u32,
    /// Returned `M`-capped folded count `m`.
    out_m: u32,
    /// Padding word.
    pad0: u32,
    /// Returned running weight sum `w_sum`.
    out_w_sum: f32,
    /// Returned finalized contribution weight `w`.
    out_w: f32,
    /// Returned held-sample target density `target_pdf`.
    out_target_pdf: f32,
    /// Padding word.
    pad1: f32,
}

/// One reprojection query: a history reservoir paired with its surface, the
/// current pixel's surface, and the admissibility tolerances.
///
/// Mirrors the state the `CPU` golden `reproject_history` reads to decide
/// admissibility and to `M`-cap the accepted history.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RestirTemporalReprojectQuery {
    /// History held light index `sample`.
    pub h_sample: u32,
    /// History running weight sum `w_sum`.
    pub h_w_sum: f32,
    /// History folded candidate count `m`.
    pub h_m: u32,
    /// History finalized contribution weight `w`.
    pub h_w: f32,
    /// History held-sample target density `target_pdf`.
    pub h_target_pdf: f32,
    /// History surface view depth.
    pub h_depth: f32,
    /// History surface unit normal.
    pub h_normal: [f32; 3],
    /// Current surface view depth.
    pub c_depth: f32,
    /// Current surface unit normal.
    pub c_normal: [f32; 3],
    /// Upper bound on the reprojected history count `max_history_m`.
    pub max_history_m: u32,
    /// Relative view-depth tolerance `depth_rel_tolerance`.
    pub depth_rel_tolerance: f32,
    /// Minimum normal agreement cosine `normal_cos_tolerance`.
    pub normal_cos_tolerance: f32,
}

impl RestirTemporalReprojectQuery {
    /// Builds a reprojection query from the history reservoir fields
    /// (`h_sample`, `h_w_sum`, `h_m`, `h_w`, `h_target_pdf`), the history
    /// surface (`h_depth`, `h_normal`), the current surface (`c_depth`,
    /// `c_normal`), and the tolerances (`max_history_m`, `depth_rel_tolerance`,
    /// `normal_cos_tolerance`).
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the full reproject_history input state in one flat query"
    )]
    pub const fn new(
        h_sample: u32,
        h_w_sum: f32,
        h_m: u32,
        h_w: f32,
        h_target_pdf: f32,
        h_depth: f32,
        h_normal: [f32; 3],
        c_depth: f32,
        c_normal: [f32; 3],
        max_history_m: u32,
        depth_rel_tolerance: f32,
        normal_cos_tolerance: f32,
    ) -> RestirTemporalReprojectQuery {
        RestirTemporalReprojectQuery {
            h_sample,
            h_w_sum,
            h_m,
            h_w,
            h_target_pdf,
            h_depth,
            h_normal,
            c_depth,
            c_normal,
            max_history_m,
            depth_rel_tolerance,
            normal_cos_tolerance,
        }
    }
}

/// One reprojection result, mirroring the value the `CPU` golden
/// `reproject_history` returns.
///
/// `hit` is `true` when the golden returned `Some`; the reservoir fields then
/// hold the `M`-capped history. On a miss (`None`) `hit` is `false` and the
/// reservoir fields are all zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RestirTemporalReprojectResult {
    /// Whether the history was admissible (golden returned `Some`).
    pub hit: bool,
    /// Returned held light index `sample`.
    pub sample: u32,
    /// Returned `M`-capped folded count `m`.
    pub m: u32,
    /// Returned running weight sum `w_sum`.
    pub w_sum: f32,
    /// Returned finalized contribution weight `w`.
    pub w: f32,
    /// Returned held-sample target density `target_pdf`.
    pub target_pdf: f32,
}

/// Encodes one [`RestirTemporalReprojectQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &RestirTemporalReprojectQuery) -> GpuQuery {
    GpuQuery {
        h_sample: q.h_sample,
        h_w_sum: q.h_w_sum,
        h_m: q.h_m,
        h_w: q.h_w,
        h_target_pdf: q.h_target_pdf,
        h_depth: q.h_depth,
        h_nx: q.h_normal[0],
        h_ny: q.h_normal[1],
        h_nz: q.h_normal[2],
        c_depth: q.c_depth,
        c_nx: q.c_normal[0],
        c_ny: q.c_normal[1],
        c_nz: q.c_normal[2],
        max_history_m: q.max_history_m,
        depth_rel_tol: q.depth_rel_tolerance,
        normal_cos_tol: q.normal_cos_tolerance,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`RestirTemporalReprojectResult`].
fn decode_result(raw: &GpuResult) -> RestirTemporalReprojectResult {
    RestirTemporalReprojectResult {
        hit: raw.hit != 0,
        sample: raw.out_sample,
        m: raw.out_m,
        w_sum: raw.out_w_sum,
        w: raw.out_w,
        target_pdf: raw.out_target_pdf,
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

/// A compiled, reusable `ReSTIR` temporal-reprojection compute pipeline,
/// twinning the `CPU` golden `reproject_history` in
/// [`prism_render_architecture::lighting::restir_temporal`].
pub struct GpuRestirTemporalReproject {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRestirTemporalReproject {
    /// Compiles the kernel and builds the reusable bind-group layout and compute
    /// pipeline on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_restir_temporal_reproject_module"),
            source: ShaderSource::Wgsl(RESTIR_TEMPORAL_REPROJECT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_restir_temporal_reproject_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_restir_temporal_reproject_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_restir_temporal_reproject_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRestirTemporalReproject {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every pair in `queries` and returns one
    /// [`RestirTemporalReprojectResult`] per input, in order.
    ///
    /// Each result equals the reference `reproject_history` outcome for the same
    /// inputs: the `hit` flag and `m` match exactly, while the passthrough
    /// weights match within the tolerance documented on this module. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[RestirTemporalReprojectQuery],
    ) -> Vec<RestirTemporalReprojectResult> {
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
            label: Some("prism_volumetric_restir_temporal_reproject_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_restir_temporal_reproject_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_restir_temporal_reproject_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_restir_temporal_reproject_bind_group"),
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
            label: Some("prism_volumetric_restir_temporal_reproject_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_restir_temporal_reproject_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_restir_temporal_reproject_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per history-versus-current pair, flattened to a 1-D
            // dispatch.
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
