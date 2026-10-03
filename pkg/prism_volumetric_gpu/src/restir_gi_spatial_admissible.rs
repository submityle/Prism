//! `wgpu` compute twin of the `ReSTIR` GI spatial-admissibility predicate
//! ([`gi_spatial_admissible`](prism_render_architecture::lighting::restir_gi_resolve::gi_spatial_admissible)).
//!
//! Before a neighbor pixel's GI reservoir may be folded into the center pixel's
//! during the spatial-reuse pass, the two surfaces must be geometrically
//! compatible: both must be real hits in front of the camera, their linear
//! view depths must agree within a relative tolerance, and their shading
//! normals must agree within a cosine tolerance. That disocclusion-style screen
//! is a stateless, closed-form boolean test over surface geometry only — it
//! never inspects the held sample — so it is a clean, portable twin target and
//! is distinct from the already-twinned reconnection Jacobian.
//!
//! [`GpuRestirGiSpatialAdmissible`] runs one thread per `(center, neighbor,
//! tolerances)` query and reproduces that predicate exactly, so a passing
//! real-device parity test is direct evidence the ported kernel accepts and
//! rejects the same neighbors the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! For one query the kernel reproduces the full
//! [`gi_spatial_admissible`](prism_render_architecture::lighting::restir_gi_resolve::gi_spatial_admissible)
//! decision: the two surface-validity checks (`view_depth > 0` and finite), the
//! relative view-depth agreement `|z_n − z_c| <= tol · z_c`, and the normal
//! cosine agreement `n_c · n_n >= cos_tol`. The shading-point *positions* play
//! no part in the predicate and are not uploaded; only the normals and depths
//! are.
//!
//! # What is not twinned (host-only)
//!
//! The surrounding resolve orchestration — the per-frame neighbor gather, the
//! variable-length admissible-neighbor collection
//! ([`gather_admissible_gi_neighbors`](prism_render_architecture::lighting::restir_gi_resolve::gather_admissible_gi_neighbors)),
//! the reservoir folds and the target-function closures — is variable-length or
//! closure-parameterized and stays on the host. Only the pure, per-pair
//! predicate is twinned here.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+ - * /`, `abs`, the
//! ordered comparisons and a `bitcast` to form `+∞` for the finiteness guard —
//! with no `sin`, `cos`, `exp`, `log`, `pow`, inverse trigonometry,
//! `smoothstep`, `round`, `sqrt` or `cbrt`, and no optional device feature, so
//! it runs unmodified on `Metal`, `Vulkan` and `DX12`. There is no loop: each
//! thread performs a fixed, bounded sequence of arithmetic and comparisons, so
//! the kernel provably terminates.
//!
//! The finiteness half of the golden `is_valid` (`view_depth > 0 &&
//! view_depth.is_finite()`) is reproduced without an `f32` equality as
//! `d > 0.0 && d < +∞`: a `NaN` fails `d > 0.0`, `+∞` fails `d < +∞`, and every
//! finite positive depth passes both, matching the reference exactly. The two
//! agreement guards are expressed as `<=` and `>=`, the exact complements of the
//! golden `depth_diff > tol · z_c` and `cos < cos_tol` rejections.
//!
//! # Correctness model
//!
//! The predicate is discrete, so the parity test asserts an exact boolean match
//! and keeps its random fixtures clear of both comparison boundaries (the depth
//! agreement margin and the cosine threshold) so a last-place difference in the
//! shared `+ - * /` sequence can never flip the decision and desync the two
//! sides. The deliberate fixtures land on unambiguous accepts and rejects.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::lighting::restir_gi_resolve::gi_spatial_admissible`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` spatial-admissibility kernel, embedded inline so the
/// twin ships no external shader asset. It reproduces the
/// [`gi_spatial_admissible`](prism_render_architecture::lighting::restir_gi_resolve::gi_spatial_admissible)
/// predicate; see the module documentation for the algorithm.
const RESTIR_GI_SPATIAL_ADMISSIBLE_WGSL: &str = r#"
// ReSTIR GI spatial-admissibility twin: one thread screens one (center,
// neighbor, tolerances) pair, mirroring the CPU golden
// `lighting::restir_gi_resolve::gi_spatial_admissible` with only + - * /, abs,
// a bitcast to +inf and ordered comparisons.
//
// Provenance: 孪生自本仓 prism_render_architecture::lighting::restir_gi_resolve::gi_spatial_admissible；
// 无第三方引擎源码或衍生代码。

struct Params {
    // Number of pairs in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Center surface linear view depth.
    center_depth: f32,
    // Neighbor surface linear view depth.
    neighbor_depth: f32,
    // Center shading-point unit normal.
    cnx: f32,
    cny: f32,
    cnz: f32,
    // Neighbor shading-point unit normal.
    nnx: f32,
    nny: f32,
    nnz: f32,
    // Relative view-depth agreement tolerance.
    depth_rel_tolerance: f32,
    // Minimum normal agreement (cosine) tolerance.
    normal_cos_tolerance: f32,
}

struct Res {
    // 1 when the neighbor is admissible, 0 otherwise.
    admissible: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

fn is_valid(d: f32) -> bool {
    // Golden `view_depth > 0 && view_depth.is_finite()` without an f32 ==:
    // NaN fails `d > 0.0`; +inf fails `d < inf`; finite positive passes both.
    let inf = bitcast<f32>(0x7f800000u);
    return d > 0.0 && d < inf;
}

@compute @workgroup_size(64)
fn admissible(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var ok: u32 = 0u;
    if (is_valid(q.center_depth) && is_valid(q.neighbor_depth)) {
        // Complement of the golden `depth_diff > tol * center_depth` rejection.
        let depth_diff = abs(q.neighbor_depth - q.center_depth);
        if (depth_diff <= q.depth_rel_tolerance * q.center_depth) {
            // Golden `dot3(center.normal, neighbor.normal) >= cos_tol`.
            let d = q.cnx * q.nnx + q.cny * q.nny + q.cnz * q.nnz;
            if (d >= q.normal_cos_tolerance) {
                ok = 1u;
            }
        }
    }

    var out: Res;
    out.admissible = ok;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the pair count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`RESTIR_GI_SPATIAL_ADMISSIBLE_WGSL`].
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

/// `repr(C)` `std430` layout of one admissibility query: the two view depths,
/// the two unit normals and the two tolerances, a `40`-byte scalar-packed stride
/// matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Center surface view depth.
    center_depth: f32,
    /// Neighbor surface view depth.
    neighbor_depth: f32,
    /// Center normal `x`.
    cnx: f32,
    /// Center normal `y`.
    cny: f32,
    /// Center normal `z`.
    cnz: f32,
    /// Neighbor normal `x`.
    nnx: f32,
    /// Neighbor normal `y`.
    nny: f32,
    /// Neighbor normal `z`.
    nnz: f32,
    /// Relative view-depth agreement tolerance.
    depth_rel_tolerance: f32,
    /// Minimum normal agreement (cosine) tolerance.
    normal_cos_tolerance: f32,
}

/// `repr(C)` `std430` layout of one admissibility result, matching the `WGSL`
/// `Res` struct: the single `u32` flag, a `4`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `1` when the neighbor is admissible, `0` otherwise.
    admissible: u32,
}

/// One spatial-admissibility query: the center and neighbor surfaces (each a
/// linear view depth plus a unit shading normal) and the two agreement
/// tolerances.
///
/// The shading-point positions play no part in the predicate and are omitted; a
/// non-positive or non-finite view depth marks an invalid surface and is always
/// rejected, exactly as the golden
/// [`gi_spatial_admissible`](prism_render_architecture::lighting::restir_gi_resolve::gi_spatial_admissible)
/// rejects it.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::lighting::restir_gi_resolve::gi_spatial_admissible`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RestirGiSpatialAdmissibleQuery {
    /// Center surface linear view depth.
    pub center_view_depth: f32,
    /// Neighbor surface linear view depth.
    pub neighbor_view_depth: f32,
    /// Center shading-point unit normal.
    pub center_normal: [f32; 3],
    /// Neighbor shading-point unit normal.
    pub neighbor_normal: [f32; 3],
    /// Relative view-depth agreement tolerance.
    pub depth_rel_tolerance: f32,
    /// Minimum normal agreement (cosine) tolerance.
    pub normal_cos_tolerance: f32,
}

impl RestirGiSpatialAdmissibleQuery {
    /// Builds a query from the center / neighbor view depths, their unit
    /// normals and the two agreement tolerances.
    #[must_use]
    pub const fn new(
        center_view_depth: f32,
        neighbor_view_depth: f32,
        center_normal: [f32; 3],
        neighbor_normal: [f32; 3],
        depth_rel_tolerance: f32,
        normal_cos_tolerance: f32,
    ) -> RestirGiSpatialAdmissibleQuery {
        RestirGiSpatialAdmissibleQuery {
            center_view_depth,
            neighbor_view_depth,
            center_normal,
            neighbor_normal,
            depth_rel_tolerance,
            normal_cos_tolerance,
        }
    }
}

/// One query's resolved admissibility, mirroring the `CPU` golden
/// [`gi_spatial_admissible`](prism_render_architecture::lighting::restir_gi_resolve::gi_spatial_admissible).
///
/// `admissible` is `true` when the neighbor surface is geometrically compatible
/// with the center and may be folded into its reservoir.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::lighting::restir_gi_resolve::gi_spatial_admissible`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RestirGiSpatialAdmissibleResult {
    /// Whether the neighbor is admissible for spatial reuse.
    pub admissible: bool,
}

/// Encodes one [`RestirGiSpatialAdmissibleQuery`] into its `std430`
/// [`GpuQuery`] slot.
fn encode_query(q: &RestirGiSpatialAdmissibleQuery) -> GpuQuery {
    GpuQuery {
        center_depth: q.center_view_depth,
        neighbor_depth: q.neighbor_view_depth,
        cnx: q.center_normal[0],
        cny: q.center_normal[1],
        cnz: q.center_normal[2],
        nnx: q.neighbor_normal[0],
        nny: q.neighbor_normal[1],
        nnz: q.neighbor_normal[2],
        depth_rel_tolerance: q.depth_rel_tolerance,
        normal_cos_tolerance: q.normal_cos_tolerance,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`RestirGiSpatialAdmissibleResult`].
fn decode_result(raw: &GpuResult) -> RestirGiSpatialAdmissibleResult {
    RestirGiSpatialAdmissibleResult {
        admissible: raw.admissible != 0,
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

/// A compiled, reusable spatial-admissibility compute pipeline, twinning the
/// `CPU` golden
/// [`gi_spatial_admissible`](prism_render_architecture::lighting::restir_gi_resolve::gi_spatial_admissible).
pub struct GpuRestirGiSpatialAdmissible {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRestirGiSpatialAdmissible {
    /// Compiles the spatial-admissibility kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRestirGiSpatialAdmissible {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_restir_gi_spatial_admissible"),
            source: ShaderSource::Wgsl(RESTIR_GI_SPATIAL_ADMISSIBLE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_restir_gi_spatial_admissible_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_restir_gi_spatial_admissible_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_restir_gi_spatial_admissible_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("admissible"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRestirGiSpatialAdmissible {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every pair in `queries` and returns one
    /// [`RestirGiSpatialAdmissibleResult`] per input, in order.
    ///
    /// Each flag equals the matching `CPU` golden
    /// [`gi_spatial_admissible`](prism_render_architecture::lighting::restir_gi_resolve::gi_spatial_admissible)
    /// decision. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[RestirGiSpatialAdmissibleQuery],
    ) -> Vec<RestirGiSpatialAdmissibleResult> {
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
            label: Some("prism_volumetric_restir_gi_spatial_admissible_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_restir_gi_spatial_admissible_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_restir_gi_spatial_admissible_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_restir_gi_spatial_admissible_bind_group"),
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
            label: Some("prism_volumetric_restir_gi_spatial_admissible_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_restir_gi_spatial_admissible_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_restir_gi_spatial_admissible_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per pair, flattened to a 1-D dispatch.
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
