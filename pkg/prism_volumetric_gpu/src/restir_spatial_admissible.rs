//! `wgpu` compute twin of the `ReSTIR` DI spatial-neighbor admissibility gate
//! ([`restir_spatial`](prism_render_architecture::lighting::restir_spatial)).
//!
//! The `CPU` golden
//! [`spatial_admissible`](prism_render_architecture::lighting::restir_spatial::spatial_admissible)
//! decides whether a screen-space neighbor pixel is geometrically compatible
//! with the center pixel and may be folded into its `ReSTIR` reservoir: both
//! surfaces must be valid (finite, in front of the camera), their view depths
//! must agree to within a relative tolerance, and their normals must agree to
//! within a cosine tolerance. The decision is a deterministic function of
//! surface geometry alone — `abs`, a subtract, a multiply, a dot product and
//! ordered comparisons — so a `GPU` kernel reproduces the `bool` exactly.
//!
//! [`GpuRestirSpatialAdmissible`] is the on-device twin of that per-sample
//! predicate: one thread evaluates one `(center, neighbor, params)` sample and
//! writes back the admissibility bit, reproducing the reference's exact branch
//! structure so a passing real-device parity test is direct evidence the ported
//! kernel computes the same accept/reject the reference does, not merely that
//! the shader compiles.
//!
//! # What is twinned
//!
//! For one `(center, neighbor, params)` sample the twin reproduces the whole of
//! [`spatial_admissible`](prism_render_architecture::lighting::restir_spatial::spatial_admissible):
//!
//! - the `is_valid` gate on each surface — a finite, strictly positive view
//!   depth (`view_depth > 0 && view_depth.is_finite()`);
//! - the relative depth test
//!   `|neighbor.view_depth - center.view_depth| <= depth_rel_tolerance * center.view_depth`;
//! - the normal agreement test
//!   `dot(center.normal, neighbor.normal) >= normal_cos_tolerance`.
//!
//! The surface validity's finiteness check is reproduced with ordered
//! comparisons only: a positive finite depth satisfies
//! `view_depth > 0 && !(view_depth > F32_MAX)`, which rejects `NaN`
//! (`NaN > 0` is false) and `+inf` (`+inf > F32_MAX` is true) exactly as the
//! golden's `is_finite` does, without constructing an infinity or using any
//! transcendental. The host may therefore send raw depths (including
//! non-finite ones) and the device still agrees bit-for-bit.
//!
//! # What stays on the host
//!
//! The variable-length neighbor collection
//! ([`gather_admissible_neighbors`](prism_render_architecture::lighting::restir_spatial::gather_admissible_neighbors))
//! — the screen-space disk/spiral addressing, the per-neighbor reservoir
//! non-empty filter, the `max`-bounded gather, and the preserved neighbor order
//! — is variable-length container work that the host owns; the device only
//! evaluates the geometric predicate. The host enqueues one
//! [`RestirSpatialAdmissibleQuery`] per candidate pair, so a storage buffer is
//! never zero-sized; an empty batch short-circuits on the host with no dispatch.
//!
//! # Correctness model
//!
//! Every quantity is an `abs`, a subtract, a multiply, a multiply-add dot
//! product and an ordered comparison, so the discrete `bool` output agrees
//! exactly for fixtures kept clear of a decision tie. A `GPU` multiply-add may
//! land a few units in the last place from the scalar reference, so a fixture
//! whose `depth_diff` sits right on `depth_rel_tolerance * center.view_depth`,
//! or whose `dot` sits right on `normal_cos_tolerance`, could flip; the parity
//! test keeps every random fixture a minimum margin clear of both thresholds
//! (rejection sampling) so `CPU` and `GPU` stay on the same side of every
//! comparison and the `bool` is asserted with an exact `==`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `+ - *` and
//! ordered comparisons — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no
//! inverse trigonometry, no `sqrt`, no `round`. No optional device feature is
//! required, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. There is no
//! loop: each thread performs a fixed, bounded sequence of arithmetic, so the
//! kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::lighting::restir_spatial`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` `ReSTIR` spatial-admissibility kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden
/// [`spatial_admissible`](prism_render_architecture::lighting::restir_spatial::spatial_admissible)
/// per-sample closed form; see the module documentation for the algorithm.
const RESTIR_SPATIAL_ADMISSIBLE_WGSL: &str = r#"
// ReSTIR spatial-admissibility twin: one thread decides whether one neighbor
// surface is geometrically compatible with its center (valid, close depth,
// agreeing normal), mirroring the CPU golden
// `lighting::restir_spatial::spatial_admissible` with only abs, + - * and
// ordered comparisons. It owns no screen-space neighbor addressing and no
// variable-length gather; those stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::lighting::restir_spatial；无第三方
// 引擎源码或衍生代码。

// Largest finite f32. A positive depth is finite iff it is not greater than
// this, so `d > 0.0 && !(d > F32_MAX)` reproduces `d > 0 && d.is_finite()`
// without constructing an infinity: NaN fails `d > 0.0`, +inf fails
// `!(d > F32_MAX)`.
const F32_MAX: f32 = 3.40282347e38;

struct Params {
    // Number of samples in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // center view_depth
    cd: f32,
    // center normal.xyz
    cnx: f32, cny: f32, cnz: f32,
    // neighbor view_depth
    nd: f32,
    // neighbor normal.xyz
    nnx: f32, nny: f32, nnz: f32,
    // relative view-depth tolerance
    depth_rel_tol: f32,
    // minimum normal agreement cosine
    normal_cos_tol: f32,
    pad0: f32,
    pad1: f32,
}

struct Result {
    // 1 when the neighbor is admissible, 0 otherwise.
    admissible: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
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

    var admissible: u32 = 0u;
    if (is_valid(q.cd) && is_valid(q.nd)) {
        // Relative depth test: reject when the depths differ by more than the
        // tolerance times the center depth.
        let depth_diff = abs(q.nd - q.cd);
        let depth_bound = q.depth_rel_tol * q.cd;
        if (!(depth_diff > depth_bound)) {
            // Normal agreement test: accept when the dot product meets the
            // cosine tolerance.
            let d = q.cnx * q.nnx + q.cny * q.nny + q.cnz * q.nnz;
            if (d >= q.normal_cos_tol) {
                admissible = 1u;
            }
        }
    }

    var out: Result;
    out.admissible = admissible;
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the sample count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`RESTIR_SPATIAL_ADMISSIBLE_WGSL`].
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

/// `repr(C)` `std430` layout of one admissibility query: the two surfaces
/// (depth plus normal, as scalar channels) and the two tolerances, plus two pad
/// words to a `48`-byte (`16`-byte-multiple) stride matching the `WGSL` `Query`
/// struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Center view depth.
    cd: f32,
    /// Center normal `x`.
    cnx: f32,
    /// Center normal `y`.
    cny: f32,
    /// Center normal `z`.
    cnz: f32,
    /// Neighbor view depth.
    nd: f32,
    /// Neighbor normal `x`.
    nnx: f32,
    /// Neighbor normal `y`.
    nny: f32,
    /// Neighbor normal `z`.
    nnz: f32,
    /// Relative view-depth tolerance.
    depth_rel_tol: f32,
    /// Minimum normal agreement cosine.
    normal_cos_tol: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
}

/// `repr(C)` `std430` layout of one admissibility result: the decision bit plus
/// three pad words to a `16`-byte stride matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `1` when the neighbor is admissible, `0` otherwise.
    admissible: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One per-sample query for the `ReSTIR` spatial-admissibility twin: the center
/// and neighbor surface geometry and the admissibility tolerances.
///
/// The host owns the surrounding gather — the screen-space neighbor addressing,
/// the reservoir non-empty filter and the `max`-bounded collection — and
/// enqueues one [`RestirSpatialAdmissibleQuery`] per candidate pair, matching
/// the reference
/// [`gather_admissible_neighbors`](prism_render_architecture::lighting::restir_spatial::gather_admissible_neighbors).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RestirSpatialAdmissibleQuery {
    /// Center linear view-space depth (must be `> 0` and finite to be valid).
    pub center_depth: f32,
    /// Center unit surface normal.
    pub center_normal: [f32; 3],
    /// Neighbor linear view-space depth (must be `> 0` and finite to be valid).
    pub neighbor_depth: f32,
    /// Neighbor unit surface normal.
    pub neighbor_normal: [f32; 3],
    /// Relative view-depth tolerance.
    pub depth_rel_tolerance: f32,
    /// Minimum normal agreement cosine.
    pub normal_cos_tolerance: f32,
}

impl RestirSpatialAdmissibleQuery {
    /// Builds a query from the two surfaces and the two tolerances.
    #[must_use]
    pub const fn new(
        center_depth: f32,
        center_normal: [f32; 3],
        neighbor_depth: f32,
        neighbor_normal: [f32; 3],
        depth_rel_tolerance: f32,
        normal_cos_tolerance: f32,
    ) -> RestirSpatialAdmissibleQuery {
        RestirSpatialAdmissibleQuery {
            center_depth,
            center_normal,
            neighbor_depth,
            neighbor_normal,
            depth_rel_tolerance,
            normal_cos_tolerance,
        }
    }
}

/// One resolved admissibility decision, mirroring the [`bool`] the golden
/// [`spatial_admissible`](prism_render_architecture::lighting::restir_spatial::spatial_admissible)
/// returns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RestirSpatialAdmissibleResult {
    /// `true` when the neighbor is geometrically compatible with the center.
    pub admissible: bool,
}

/// Encodes one [`RestirSpatialAdmissibleQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &RestirSpatialAdmissibleQuery) -> GpuQuery {
    GpuQuery {
        cd: q.center_depth,
        cnx: q.center_normal[0],
        cny: q.center_normal[1],
        cnz: q.center_normal[2],
        nd: q.neighbor_depth,
        nnx: q.neighbor_normal[0],
        nny: q.neighbor_normal[1],
        nnz: q.neighbor_normal[2],
        depth_rel_tol: q.depth_rel_tolerance,
        normal_cos_tol: q.normal_cos_tolerance,
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`RestirSpatialAdmissibleResult`], turning the `admissible` word back into a
/// [`bool`].
fn decode_result(raw: &GpuResult) -> RestirSpatialAdmissibleResult {
    RestirSpatialAdmissibleResult {
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

/// A compiled, reusable `ReSTIR` spatial-admissibility compute pipeline,
/// twinning the `CPU` golden
/// [`spatial_admissible`](prism_render_architecture::lighting::restir_spatial::spatial_admissible).
pub struct GpuRestirSpatialAdmissible {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRestirSpatialAdmissible {
    /// Compiles the `ReSTIR` spatial-admissibility kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRestirSpatialAdmissible {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_restir_spatial_admissible"),
            source: ShaderSource::Wgsl(RESTIR_SPATIAL_ADMISSIBLE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_restir_spatial_admissible_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_restir_spatial_admissible_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_restir_spatial_admissible_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRestirSpatialAdmissible {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every sample in `queries` and returns one
    /// [`RestirSpatialAdmissibleResult`] per input, in order.
    ///
    /// Each decision equals the reference exactly for samples kept clear of a
    /// threshold tie, as documented on this module. An empty `queries` batch
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[RestirSpatialAdmissibleQuery],
    ) -> Vec<RestirSpatialAdmissibleResult> {
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
            label: Some("prism_volumetric_restir_spatial_admissible_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_restir_spatial_admissible_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_restir_spatial_admissible_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_restir_spatial_admissible_bind_group"),
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
            label: Some("prism_volumetric_restir_spatial_admissible_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_restir_spatial_admissible_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_restir_spatial_admissible_pass"),
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
