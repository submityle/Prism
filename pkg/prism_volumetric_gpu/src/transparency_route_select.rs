//! `wgpu` compute twin of the per-surface transparency path selector from the
//! transparency routing module
//! ([`routing`](prism_render_architecture::transparency::routing)).
//!
//! A transparent surface cannot be composited by a single universal rule:
//! sortable glass wants back-to-front alpha blending, overlapping foliage needs
//! an order-independent (`OIT`) resolve, water and hair have bespoke passes, and
//! participating media are marched as volumes. The golden
//! [`select_transparency_path`](prism_render_architecture::transparency::routing::select_transparency_path)
//! turns a surface's content class plus the backend capabilities into one
//! concrete [`TransparencyPath`](prism_render_architecture::transparency::TransparencyPath),
//! and [`outputs_for`](prism_render_architecture::transparency::routing::outputs_for)
//! reports which shared frame targets that path writes.
//!
//! Both routines are pure branch logic over small discrete enums, so
//! [`GpuTransparencyRouteSelect`] ports them exactly: one thread owns one
//! surface, reads the content class and the three boolean flags plus the layer
//! count, reproduces the `match` ladder to pick a path discriminant, and then
//! reproduces the output `match` to pack the three target-write booleans. A
//! passing real-device parity test is direct evidence the ported kernel
//! reproduces the reference classification, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! For one query carrying the content-class discriminant `kind`, the
//! `order_independent`, `high_fidelity` and `moment_oit` flags, and the glass
//! `layer_count`, the kernel reproduces:
//! - `select_transparency_path`: `Water` to `SingleLayerWater`, `Hair` to
//!   `HairVisibility`, `Volume` to `Volumetric`, `Glass` to `LayeredGlass` when
//!   `layer_count > 1` else `Sorted`, and `General` to `MomentOit` when it is
//!   order-independent, high-fidelity and the backend supports moment `OIT`,
//!   to `WeightedOit` when order-independent otherwise, and to `Sorted` when
//!   depth-sortable; and
//! - `outputs_for`: the three booleans `writes_reactive_mask`, `writes_motion`
//!   and `contributes_to_ray_scene` for the selected path.
//!
//! # Discriminant encoding
//!
//! The twin encodes [`TransparencyPath`](prism_render_architecture::transparency::TransparencyPath)
//! exactly as the golden enum's `as u32`, which follows the declaration order:
//! `Sorted = 0`, `WeightedOit = 1`, `MomentOit = 2`, `LayeredGlass = 3`,
//! `SingleLayerWater = 4`, `HairVisibility = 5`, `Volumetric = 6`. The content
//! class `TransparentKind` is encoded likewise: `General = 0`, `Water = 1`,
//! `Hair = 2`, `Glass = 3`, `Volume = 4`. The parity test pins the kernel's
//! codes against the golden enums' `as u32`, so the two share one source of
//! truth.
//!
//! # Correctness model
//!
//! Every step is integer comparison and selection with no floating-point
//! arithmetic, so the `GPU` and `CPU` produce bit-identical discriminants and
//! booleans; parity is asserted with exact equality.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `u32` comparisons and
//! branches — with no transcendental call and no `64`-bit integers or floats.
//! No optional device feature is required, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::transparency::routing`；无第三方引擎源码或衍生代码。
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

/// Upper bound on the number of surfaces one dispatch resolves. The twin is
/// batch-agnostic, but callers can size staging against this cap.
pub const MAX_SURFACES: usize = 65_536;

/// The portable core-`WGSL` transparency-route selector kernel, embedded inline
/// so the twin ships as a single source file. The single entry point `route`
/// resolves one surface per thread.
const TRANSPARENCY_ROUTE_SELECT_WGSL: &str = r#"
// Transparency route selector twin: one thread owns one surface. It reproduces
// the golden select_transparency_path match ladder to pick a path discriminant,
// then the outputs_for match to pack the three target-write booleans. All
// integer comparison and selection, no floating-point.
//
// Path codes match the golden enum `as u32`: Sorted=0, WeightedOit=1,
// MomentOit=2, LayeredGlass=3, SingleLayerWater=4, HairVisibility=5,
// Volumetric=6. Kind codes: General=0, Water=1, Hair=2, Glass=3, Volume=4.
//
// Provenance: 孪生自本仓 prism_render_architecture::transparency::routing；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of surfaces in the batch; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Content-class discriminant (General=0, Water=1, Hair=2, Glass=3,
    // Volume=4).
    kind: u32,
    // 1 when fragments overlap unpredictably and need an OIT resolve.
    order_independent: u32,
    // 1 when the surface wants moment OIT over weighted-blended OIT.
    high_fidelity: u32,
    // 1 when the backend can run the moment-based OIT resolve.
    moment_oit: u32,
    // Number of stacked refractive layers for glass.
    layer_count: u32,
}

struct Outcome {
    // Selected path discriminant (0..6).
    path_code: u32,
    // Writes into the TAA reactive mask (0 or 1).
    writes_reactive_mask: u32,
    // Emits clean motion vectors (0 or 1).
    writes_motion: u32,
    // Contributes to the ray scene for refraction/reflection (0 or 1).
    contributes_to_ray_scene: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Outcome>;

@compute @workgroup_size(64)
fn route(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.count) {
        return;
    }
    let q = queries[gid.x];

    // select_transparency_path: dedicated kinds take their paths; glass splits
    // on layer_count; general sorts unless order-independent, then moment OIT
    // only with both the request and the capability.
    var path: u32 = 0u;
    if (q.kind == 1u) {
        path = 4u;
    } else if (q.kind == 2u) {
        path = 5u;
    } else if (q.kind == 4u) {
        path = 6u;
    } else if (q.kind == 3u) {
        if (q.layer_count > 1u) {
            path = 3u;
        } else {
            path = 0u;
        }
    } else {
        if (q.order_independent == 1u) {
            if (q.high_fidelity == 1u && q.moment_oit == 1u) {
                path = 2u;
            } else {
                path = 1u;
            }
        } else {
            path = 0u;
        }
    }

    // outputs_for: every path writes the reactive mask; the trio splits on the
    // path code.
    var mask: u32 = 1u;
    var motion: u32 = 0u;
    var ray: u32 = 0u;
    if (path == 0u || path == 5u) {
        // Sorted | HairVisibility: order-dependent, motion yes, ray no.
        motion = 1u;
        ray = 0u;
    } else if (path == 1u || path == 2u) {
        // WeightedOit | MomentOit: OIT, motion no, ray no.
        motion = 0u;
        ray = 0u;
    } else if (path == 3u || path == 4u) {
        // LayeredGlass | SingleLayerWater: motion yes, ray yes.
        motion = 1u;
        ray = 1u;
    } else {
        // Volumetric (6): motion no, ray yes.
        motion = 0u;
        ray = 1u;
    }

    results[gid.x] = Outcome(path, mask, motion, ray);
}
"#;

/// Uniform parameters for the dispatch: the surface count and three pad words,
/// filling a `16`-byte uniform struct matching `Params` in
/// [`TRANSPARENCY_ROUTE_SELECT_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid surfaces in the batch.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one surface query, matching the `WGSL` `Query`
/// struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Content-class discriminant.
    kind: u32,
    /// `1` when the surface needs an order-independent resolve.
    order_independent: u32,
    /// `1` when the surface wants moment `OIT`.
    high_fidelity: u32,
    /// `1` when the backend supports moment `OIT`.
    moment_oit: u32,
    /// Number of stacked refractive layers.
    layer_count: u32,
}

/// `repr(C)` `std430` layout of one resolved outcome, matching the `WGSL`
/// `Outcome` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuOutcome {
    /// Selected path discriminant (`0..6`).
    path_code: u32,
    /// Writes into the `TAA` reactive mask (`0` or `1`).
    writes_reactive_mask: u32,
    /// Emits clean motion vectors (`0` or `1`).
    writes_motion: u32,
    /// Contributes to the ray scene (`0` or `1`).
    contributes_to_ray_scene: u32,
}

/// One transparency-route query for the twin: the content class and the three
/// boolean flags plus the glass layer count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransparencyRouteSelectQuery {
    /// Content-class discriminant (`General = 0`, `Water = 1`, `Hair = 2`,
    /// `Glass = 3`, `Volume = 4`), matching the golden `TransparentKind as u32`.
    pub kind: u32,
    /// `true` when the surface needs an order-independent resolve.
    pub order_independent: bool,
    /// `true` when the surface wants moment `OIT` over weighted-blended `OIT`.
    pub high_fidelity: bool,
    /// `true` when the backend supports the moment-based `OIT` resolve.
    pub moment_oit: bool,
    /// Number of stacked refractive layers for glass.
    pub layer_count: u32,
}

impl TransparencyRouteSelectQuery {
    /// Builds a query from the content-class discriminant, the three flags, and
    /// the layer count.
    #[must_use]
    pub fn new(
        kind: u32,
        order_independent: bool,
        high_fidelity: bool,
        moment_oit: bool,
        layer_count: u32,
    ) -> TransparencyRouteSelectQuery {
        TransparencyRouteSelectQuery {
            kind,
            order_independent,
            high_fidelity,
            moment_oit,
            layer_count,
        }
    }
}

/// One resolved transparency route: the selected path discriminant and the
/// three shared-target-write booleans, mirroring the reference
/// [`select_transparency_path`](prism_render_architecture::transparency::routing::select_transparency_path)
/// and [`outputs_for`](prism_render_architecture::transparency::routing::outputs_for).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransparencyRouteSelectResult {
    /// Selected path discriminant (`0..6`), matching the golden
    /// `TransparencyPath as u32`.
    pub path_code: u32,
    /// `true` when the path writes into the `TAA` reactive mask.
    pub writes_reactive_mask: bool,
    /// `true` when the path emits clean motion vectors.
    pub writes_motion: bool,
    /// `true` when the path contributes to the ray scene.
    pub contributes_to_ray_scene: bool,
}

/// Encodes one [`TransparencyRouteSelectQuery`] into its `std430` [`GpuQuery`]
/// slot, mapping each boolean flag to a `0`/`1` word.
fn encode_query(q: &TransparencyRouteSelectQuery) -> GpuQuery {
    GpuQuery {
        kind: q.kind,
        order_independent: u32::from(q.order_independent),
        high_fidelity: u32::from(q.high_fidelity),
        moment_oit: u32::from(q.moment_oit),
        layer_count: q.layer_count,
    }
}

/// Decodes one `std430` [`GpuOutcome`] into a [`TransparencyRouteSelectResult`],
/// reading each boolean back from its `0`/`1` word.
fn decode_outcome(o: &GpuOutcome) -> TransparencyRouteSelectResult {
    TransparencyRouteSelectResult {
        path_code: o.path_code,
        writes_reactive_mask: o.writes_reactive_mask != 0,
        writes_motion: o.writes_motion != 0,
        contributes_to_ray_scene: o.contributes_to_ray_scene != 0,
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

/// A compiled, reusable transparency-route selector compute pipeline, twinning
/// the `CPU` golden `select_transparency_path` and `outputs_for` from
/// [`routing`](prism_render_architecture::transparency::routing).
pub struct GpuTransparencyRouteSelect {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTransparencyRouteSelect {
    /// Compiles the route-selector kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTransparencyRouteSelect {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_transparency_route_select"),
            source: ShaderSource::Wgsl(TRANSPARENCY_ROUTE_SELECT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_transparency_route_select_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_transparency_route_select_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_transparency_route_select_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("route"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTransparencyRouteSelect {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every surface in `queries` and returns one
    /// [`TransparencyRouteSelectResult`] per input, in order.
    ///
    /// An empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[TransparencyRouteSelectQuery],
    ) -> Vec<TransparencyRouteSelectResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_transparency_route_select_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let result_bytes = (count * size_of::<GpuOutcome>()) as u64;
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_transparency_route_select_results"),
            size: result_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_transparency_route_select_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_transparency_route_select_bind_group"),
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

        let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_transparency_route_select_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_transparency_route_select_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per surface, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_transparency_route_select_stage"),
            size: result_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&results_buf, 0, &stage, 0, result_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let outcomes = bytemuck::cast_slice::<u8, GpuOutcome>(&view).to_vec();
        drop(view);
        stage.unmap();

        outcomes.iter().map(decode_outcome).collect()
    }
}
