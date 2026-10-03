//! `wgpu` compute twin of five signed-distance domain/scalar operators of the
//! `CPU` golden path
//! ([`round_distance`](prism_render_architecture::ray_scene::sdf_domain::round_distance),
//! [`onion`](prism_render_architecture::ray_scene::sdf_domain::onion),
//! [`scale_point`](prism_render_architecture::ray_scene::sdf_domain::scale_point),
//! [`scale_distance`](prism_render_architecture::ray_scene::sdf_domain::scale_distance)
//! and [`extrude_round`](prism_render_architecture::ray_scene::sdf_domain::extrude_round)).
//!
//! Constructive implicit modelling reshapes a single field either by
//! remapping its returned distance or by transforming the query point before
//! the base primitive is sampled. The reference exposes a vocabulary of such
//! operators: `round_distance` chamfers an edge by inflating the surface
//! outward, `onion` hollows a solid into a shell, `scale_point` paired with
//! `scale_distance` uniformly rescales a shape while keeping the value a valid
//! distance, and `extrude_round` lifts a 2D profile distance into a 3D slab
//! with a rounded rim. [`GpuSdfDomainShell`] is the on-device twin: each
//! thread reads one query bundling every operator's inputs and writes all five
//! outputs, reproducing the reference closed forms with only `abs`, `min`,
//! `max`, `sqrt`, products and quotients.
//!
//! # What is twinned
//!
//! Each thread reads one [`SdfDomainShellQuery`] — the base distance `d` with
//! chamfer `radius` and shell `thickness`, a `point` plus uniform `factor`,
//! the scalar `distance` to rescale, and the extrusion inputs (`d2d`, `z`,
//! `half_height`, `rounding`) — and writes one [`SdfDomainShellResult`]
//! holding `round_distance_value` (`d - radius`), `onion_value`
//! (`|d| - thickness`), `scale_distance_value` (`distance * factor`),
//! `extrude_round_value` (the rounded `opExtrusion` combine) and
//! `scale_point_value` (`point / factor`).
//!
//! # What stays on the host
//!
//! The point operators that fold space onto a lattice ([`repeat`], the mirror
//! and finite-lattice variants), the `CSG` operators that compose fields, and
//! the ray-marcher that evaluates them all stay on the host; the device sees
//! only the five stateless, fixed-width evaluations, one query at a time, so a
//! storage buffer is never zero-sized.
//!
//! [`repeat`]: prism_render_architecture::ray_scene::sdf_domain::repeat
//!
//! # Correctness model
//!
//! `round_distance`, `onion`, `scale_distance` and `scale_point` are affine or
//! fold-and-subtract scalar maps; `extrude_round` threads through one `sqrt`,
//! so the `CPU` and `GPU` are not bit-exact: a `GPU` `sqrt` or divide may land
//! a few units in the last place from the scalar reference. The parity test
//! asserts each value within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, loose
//! enough to admit a legal last-place difference yet tight enough to catch a
//! wrong port.
//!
//! # Conditioning
//!
//! `scale_point` divides by `factor`, so fixtures keep `|factor|` a safe
//! margin off zero; `extrude_round` picks the governing feature by
//! `max(wx, wy)`, so fixtures keep the two components a safe margin apart via
//! rejection sampling. `round_distance` and `onion` are continuous through
//! `d == 0`, so no branch flips there.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`,
//! `max`, `sqrt`, `+ - * /` and unsigned index arithmetic — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `round` and
//! no `ceil`, and no `f64`/`u64`/`u16`/`i64`/`i16`. It runs unmodified on
//! `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_domain`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` domain/scalar-operator kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`round_distance`](prism_render_architecture::ray_scene::sdf_domain::round_distance),
/// [`onion`](prism_render_architecture::ray_scene::sdf_domain::onion),
/// [`scale_point`](prism_render_architecture::ray_scene::sdf_domain::scale_point),
/// [`scale_distance`](prism_render_architecture::ray_scene::sdf_domain::scale_distance)
/// and [`extrude_round`](prism_render_architecture::ray_scene::sdf_domain::extrude_round)
/// closed forms; see the module documentation for the algorithm.
const SDF_DOMAIN_SHELL_WGSL: &str = r#"
// Domain/scalar-operator twin: one thread computes one query's round_distance,
// onion, scale_distance, extrude_round and scale_point outputs, mirroring the
// CPU golden
// `ray_scene::sdf_domain::{round_distance, onion, scale_point, scale_distance,
// extrude_round}` with only abs, min, max, sqrt, products and quotients. The
// lattice-folding point operators and the ray-marcher stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::sdf_domain；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Base distance fed to round_distance and onion.
    d: f32,
    // Chamfer radius for round_distance.
    radius: f32,
    // Shell half-width for onion.
    thickness: f32,
    // Point x/y/z for scale_point.
    px: f32,
    py: f32,
    pz: f32,
    // Uniform scale factor shared by scale_point and scale_distance.
    factor: f32,
    // Scalar distance for scale_distance.
    dist: f32,
    // 2D profile distance for extrude_round.
    d2d: f32,
    // Axial coordinate for extrude_round.
    z: f32,
    // Slab half-height for extrude_round.
    half_height: f32,
    // Rim fillet radius for extrude_round.
    rounding: f32,
}

struct Shell {
    // round_distance(d, radius).
    round_distance_value: f32,
    // onion(d, thickness).
    onion_value: f32,
    // scale_distance(distance, factor).
    scale_distance_value: f32,
    // extrude_round(d2d, z, half_height, rounding).
    extrude_round_value: f32,
    // scale_point(point, factor) components.
    spx: f32,
    spy: f32,
    spz: f32,
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Shell>;

// Chamfer: inflate the surface outward by subtracting the radius.
fn round_distance_op(d: f32, radius: f32) -> f32 {
    return d - radius;
}

// Hollow shell: fold about zero (|d|) and subtract the half-width.
fn onion_op(d: f32, thickness: f32) -> f32 {
    return abs(d) - thickness;
}

// Distance rescale: multiply the base distance by the uniform factor.
fn scale_distance_op(dist: f32, factor: f32) -> f32 {
    return dist * factor;
}

// Rounded 2D->3D extrusion (Inigo Quilez opExtrusion with a rounded rim): inset
// by rounding on both axes, do the standard interior/exterior combine, offset
// back out by rounding. Matches the golden operation order exactly.
fn extrude_round_op(d2d: f32, z: f32, half_height: f32, rounding: f32) -> f32 {
    let wx = d2d + rounding;
    let wy = abs(z) - (half_height - rounding);
    let inside = min(max(wx, wy), 0.0);
    let ox = max(wx, 0.0);
    let oy = max(wy, 0.0);
    let outside = sqrt(ox * ox + oy * oy);
    return inside + outside - rounding;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Shell;
    out.round_distance_value = round_distance_op(q.d, q.radius);
    out.onion_value = onion_op(q.d, q.thickness);
    out.scale_distance_value = scale_distance_op(q.dist, q.factor);
    out.extrude_round_value = extrude_round_op(q.d2d, q.z, q.half_height, q.rounding);
    out.spx = q.px / q.factor;
    out.spy = q.py / q.factor;
    out.spz = q.pz / q.factor;
    out.pad0 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte uniform struct matching the `WGSL` `Params`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct:
/// every operator's inputs packed into a `48`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Base distance for `round_distance` and `onion`.
    d: f32,
    /// Chamfer radius for `round_distance`.
    radius: f32,
    /// Shell half-width for `onion`.
    thickness: f32,
    /// Point component `x` for `scale_point`.
    px: f32,
    /// Point component `y` for `scale_point`.
    py: f32,
    /// Point component `z` for `scale_point`.
    pz: f32,
    /// Uniform scale factor shared by `scale_point` and `scale_distance`.
    factor: f32,
    /// Scalar distance for `scale_distance`.
    dist: f32,
    /// 2D profile distance for `extrude_round`.
    d2d: f32,
    /// Axial coordinate for `extrude_round`.
    z: f32,
    /// Slab half-height for `extrude_round`.
    half_height: f32,
    /// Rim fillet radius for `extrude_round`.
    rounding: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Shell` struct:
/// four scalar outputs plus the scaled point and one pad word to a `32`-byte
/// stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `round_distance` signed distance.
    round_distance_value: f32,
    /// `onion` signed distance.
    onion_value: f32,
    /// `scale_distance` rescaled distance.
    scale_distance_value: f32,
    /// `extrude_round` signed distance.
    extrude_round_value: f32,
    /// Scaled point component `x`.
    spx: f32,
    /// Scaled point component `y`.
    spy: f32,
    /// Scaled point component `z`.
    spz: f32,
    /// Padding word.
    pad0: f32,
}

/// One query for the domain/scalar-operator twin: the inputs for all five
/// operators.
///
/// `d` is the base distance fed to
/// [`round_distance`](prism_render_architecture::ray_scene::sdf_domain::round_distance)
/// (with chamfer `radius`) and
/// [`onion`](prism_render_architecture::ray_scene::sdf_domain::onion) (with
/// shell `thickness`); `point` and `factor` drive
/// [`scale_point`](prism_render_architecture::ray_scene::sdf_domain::scale_point),
/// `distance` and `factor` drive
/// [`scale_distance`](prism_render_architecture::ray_scene::sdf_domain::scale_distance);
/// `d2d`, `z`, `half_height` and `rounding` drive
/// [`extrude_round`](prism_render_architecture::ray_scene::sdf_domain::extrude_round).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfDomainShellQuery {
    /// Base distance for `round_distance` and `onion`.
    pub d: f32,
    /// Chamfer radius for `round_distance`.
    pub radius: f32,
    /// Shell half-width for `onion`.
    pub thickness: f32,
    /// Point `[x, y, z]` for `scale_point`.
    pub point: [f32; 3],
    /// Uniform scale factor shared by `scale_point` and `scale_distance`.
    pub factor: f32,
    /// Scalar distance for `scale_distance`.
    pub distance: f32,
    /// 2D profile distance for `extrude_round`.
    pub d2d: f32,
    /// Axial coordinate for `extrude_round`.
    pub z: f32,
    /// Slab half-height for `extrude_round`.
    pub half_height: f32,
    /// Rim fillet radius for `extrude_round`.
    pub rounding: f32,
}

impl SdfDomainShellQuery {
    /// Builds a query from every operator's inputs.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the five golden signatures packed into one query slot"
    )]
    pub const fn new(
        d: f32,
        radius: f32,
        thickness: f32,
        point: [f32; 3],
        factor: f32,
        distance: f32,
        d2d: f32,
        z: f32,
        half_height: f32,
        rounding: f32,
    ) -> SdfDomainShellQuery {
        SdfDomainShellQuery {
            d,
            radius,
            thickness,
            point,
            factor,
            distance,
            d2d,
            z,
            half_height,
            rounding,
        }
    }
}

/// One resolved query of the domain/scalar-operator twin: the four scalar
/// outputs plus the scaled point.
///
/// `round_distance_value` is
/// [`round_distance`](prism_render_architecture::ray_scene::sdf_domain::round_distance);
/// `onion_value` is
/// [`onion`](prism_render_architecture::ray_scene::sdf_domain::onion);
/// `scale_distance_value` is
/// [`scale_distance`](prism_render_architecture::ray_scene::sdf_domain::scale_distance);
/// `extrude_round_value` is
/// [`extrude_round`](prism_render_architecture::ray_scene::sdf_domain::extrude_round);
/// `scale_point_value` is
/// [`scale_point`](prism_render_architecture::ray_scene::sdf_domain::scale_point).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfDomainShellResult {
    /// `round_distance` signed distance.
    pub round_distance_value: f32,
    /// `onion` signed distance.
    pub onion_value: f32,
    /// `scale_distance` rescaled distance.
    pub scale_distance_value: f32,
    /// `extrude_round` signed distance.
    pub extrude_round_value: f32,
    /// `scale_point` scaled point `[x, y, z]`.
    pub scale_point_value: [f32; 3],
}

/// Encodes one [`SdfDomainShellQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SdfDomainShellQuery) -> GpuQuery {
    GpuQuery {
        d: q.d,
        radius: q.radius,
        thickness: q.thickness,
        px: q.point[0],
        py: q.point[1],
        pz: q.point[2],
        factor: q.factor,
        dist: q.distance,
        d2d: q.d2d,
        z: q.z,
        half_height: q.half_height,
        rounding: q.rounding,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SdfDomainShellResult`].
fn decode_result(raw: &GpuResult) -> SdfDomainShellResult {
    SdfDomainShellResult {
        round_distance_value: raw.round_distance_value,
        onion_value: raw.onion_value,
        scale_distance_value: raw.scale_distance_value,
        extrude_round_value: raw.extrude_round_value,
        scale_point_value: [raw.spx, raw.spy, raw.spz],
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

/// A compiled, reusable domain/scalar-operator compute pipeline, twinning the
/// `CPU` golden
/// [`round_distance`](prism_render_architecture::ray_scene::sdf_domain::round_distance),
/// [`onion`](prism_render_architecture::ray_scene::sdf_domain::onion),
/// [`scale_point`](prism_render_architecture::ray_scene::sdf_domain::scale_point),
/// [`scale_distance`](prism_render_architecture::ray_scene::sdf_domain::scale_distance)
/// and [`extrude_round`](prism_render_architecture::ray_scene::sdf_domain::extrude_round).
pub struct GpuSdfDomainShell {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfDomainShell {
    /// Compiles the domain/scalar-operator kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdfDomainShell {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sdf_domain_shell"),
            source: ShaderSource::Wgsl(SDF_DOMAIN_SHELL_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sdf_domain_shell_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sdf_domain_shell_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sdf_domain_shell_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfDomainShell {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`SdfDomainShellResult`]
    /// per input, in order.
    ///
    /// The outputs match the reference to within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SdfDomainShellQuery],
    ) -> Vec<SdfDomainShellResult> {
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
            label: Some("prism_volumetric_sdf_domain_shell_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_domain_shell_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sdf_domain_shell_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sdf_domain_shell_bind_group"),
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
            label: Some("prism_volumetric_sdf_domain_shell_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sdf_domain_shell_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sdf_domain_shell_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
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
