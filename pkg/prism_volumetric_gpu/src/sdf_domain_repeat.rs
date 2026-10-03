//! `wgpu` compute twin of four signed-distance *domain* operators of the `CPU`
//! golden path
//! ([`limited_repeat`](prism_render_architecture::ray_scene::sdf_domain::limited_repeat),
//! [`mirror_repeat`](prism_render_architecture::ray_scene::sdf_domain::mirror_repeat),
//! [`elongate`](prism_render_architecture::ray_scene::sdf_domain::elongate)
//! and
//! [`elongate_correction`](prism_render_architecture::ray_scene::sdf_domain::elongate_correction)).
//!
//! Domain operators reshape a single signed-distance field by transforming the
//! query point before the primitive is sampled (or by adding an interior
//! distance correction). The reference exposes four such closed forms:
//! [`limited_repeat`] folds the point into a *finite* lattice of
//! `2 * limit + 1` instances per axis, [`mirror_repeat`] folds into an
//! *infinite* lattice whose odd cells are reflected so neighbours meet without
//! a seam, [`elongate`] carves a `[-h, h]` core out of each axis to stretch the
//! primitive, and [`elongate_correction`] restores the exact interior gradient
//! that bare [`elongate`] loses. [`GpuSdfDomainRepeat`] is the on-device twin:
//! each thread reads one point plus the lattice `period`, the finite `limit`
//! and the elongation `half_extent`, and writes the three transformed points
//! plus the scalar correction, reproducing the reference operation for
//! operation.
//!
//! # What is twinned
//!
//! Each thread reads one [`SdfDomainRepeatQuery`] — a query `point` plus the
//! per-axis `period`, `limit` and `half_extent` — and writes one
//! [`SdfDomainRepeatResult`] holding the [`limited_repeat`] point, the
//! [`mirror_repeat`] point, the [`elongate`] point and the
//! [`elongate_correction`] scalar. The lattice folds subtract an integral cell
//! index times the period; the mirror fold negates odd cells; the elongation
//! subtracts the clamped core and the correction returns the non-positive
//! signed distance into the inserted elongation box.
//!
//! # What stays on the host
//!
//! The signed-distance *values*, the primitives these operators reshape, the
//! `CSG` operators that compose fields, and the ray-marcher that walks a ray
//! all stay on the host; the device sees only the four stateless, fixed-width
//! per-axis transforms, one query at a time, so a storage buffer is never
//! zero-sized.
//!
//! # Correctness model
//!
//! The lattice folds thread through a divide (`point / period`) and a
//! round-to-nearest-cell, so the `CPU` and `GPU` are not bit-exact: a `GPU`
//! divide may land a few units in the last place from the scalar reference. The
//! parity test asserts each component within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`. The cell index is discontinuous at the half-period
//! boundaries `|p / period| = n + 0.5`, where a last-place difference could
//! pick a neighbouring cell; fixtures stay a safe margin from those boundaries
//! so both paths fold into the same cell.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `clamp`, `floor`, `select`, `+ - * /` and unsigned index arithmetic — with
//! no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, and no
//! `f64`/`u64`/`u16`/`i64`/`i16`. The reference uses `f32::round` (round half
//! away from zero) for lattice folding; because the built-in `round` is not in
//! the permitted subset it is reproduced as `sign(x) * floor(abs(x) + 0.5)`,
//! which matches `f32::round` exactly including at the half-integer ties the
//! fixtures reject. It runs unmodified on `Metal`, `Vulkan` and `DX12`.
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

/// The portable core-`WGSL` domain-operator kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`limited_repeat`](prism_render_architecture::ray_scene::sdf_domain::limited_repeat),
/// [`mirror_repeat`](prism_render_architecture::ray_scene::sdf_domain::mirror_repeat),
/// [`elongate`](prism_render_architecture::ray_scene::sdf_domain::elongate)
/// and
/// [`elongate_correction`](prism_render_architecture::ray_scene::sdf_domain::elongate_correction).
const SDF_DOMAIN_REPEAT_WGSL: &str = r#"
// Signed-distance domain-operator twin: one thread folds one query point into
// the finite lattice, the mirrored lattice, the elongation displacement and the
// elongation interior correction, mirroring the CPU golden
// `ray_scene::sdf_domain::{limited_repeat, mirror_repeat, elongate,
// elongate_correction}` with only abs, min, max, clamp, floor, select and the
// arithmetic operators. The distance values, primitives and the ray-marcher
// stay on the host.
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
    // Query point components.
    px: f32,
    py: f32,
    pz: f32,
    // Lattice period per axis.
    perx: f32,
    pery: f32,
    perz: f32,
    // Finite-lattice instance limit per axis.
    limx: f32,
    limy: f32,
    limz: f32,
    // Elongation half-extent per axis.
    hex: f32,
    hey: f32,
    hez: f32,
}

struct DomainResult {
    // limited_repeat folded point.
    lr_x: f32,
    lr_y: f32,
    lr_z: f32,
    // mirror_repeat folded point.
    mr_x: f32,
    mr_y: f32,
    mr_z: f32,
    // elongate displacement.
    el_x: f32,
    el_y: f32,
    el_z: f32,
    // elongate_correction scalar.
    corr: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<DomainResult>;

// Smallest positive normal f32, matching the reference `f32::MIN_POSITIVE`
// guard that leaves a zero- (or sub-normal-) period axis unchanged.
const MIN_POSITIVE: f32 = 1.17549435e-38;

// Reproduces Rust `f32::round` (round half away from zero) without the banned
// `round` built-in: `sign(x) * floor(abs(x) + 0.5)`. `select(-1, 1, x >= 0)`
// gives +1 at zero, matching `floor(0.5) == 0` so zero maps to zero.
fn round_away(x: f32) -> f32 {
    let s = select(-1.0, 1.0, x >= 0.0);
    return s * floor(abs(x) + 0.5);
}

// Per-axis finite lattice fold:
// `p - period * clamp(round(p / period), -limit, limit)`. A zero- (or
// sub-normal-) period axis passes through unchanged.
fn limited_repeat_axis(p: f32, period: f32, limit: f32) -> f32 {
    if (abs(period) > MIN_POSITIVE) {
        let cell = clamp(round_away(p / period), -limit, limit);
        return p - period * cell;
    }
    return p;
}

// Per-axis mirrored lattice fold: fold to the cell-local offset and negate the
// odd cells so neighbours meet as reflections. A zero- (or sub-normal-) period
// axis passes through unchanged.
fn mirror_repeat_axis(p: f32, period: f32) -> f32 {
    if (abs(period) > MIN_POSITIVE) {
        let cell = round_away(p / period);
        var local = p - period * cell;
        // `cell` is integral, so the floored remainder modulo two is 0.0 for
        // even indices and 1.0 for odd indices; an odd cell reflects.
        let parity = cell - 2.0 * floor(cell * 0.5);
        if (parity > 0.5) {
            local = -local;
        }
        return local;
    }
    return p;
}

// Per-axis elongation displacement: `p - clamp(p, -h, h)`.
fn elongate_axis(p: f32, h: f32) -> f32 {
    return p - clamp(p, -h, h);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: DomainResult;

    out.lr_x = limited_repeat_axis(q.px, q.perx, q.limx);
    out.lr_y = limited_repeat_axis(q.py, q.pery, q.limy);
    out.lr_z = limited_repeat_axis(q.pz, q.perz, q.limz);

    out.mr_x = mirror_repeat_axis(q.px, q.perx);
    out.mr_y = mirror_repeat_axis(q.py, q.pery);
    out.mr_z = mirror_repeat_axis(q.pz, q.perz);

    out.el_x = elongate_axis(q.px, q.hex);
    out.el_y = elongate_axis(q.py, q.hey);
    out.el_z = elongate_axis(q.pz, q.hez);

    // elongate_correction: min(max(|p.x| - h.x, |p.y| - h.y, |p.z| - h.z), 0).
    let cx = abs(q.px) - q.hex;
    let cy = abs(q.py) - q.hey;
    let cz = abs(q.pz) - q.hez;
    out.corr = min(max(max(cx, cy), cz), 0.0);

    out.pad0 = 0.0;
    out.pad1 = 0.0;

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SDF_DOMAIN_REPEAT_WGSL`].
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
/// the point components plus the per-axis `period`, `limit` and `half_extent`,
/// twelve scalars to a `48`-byte stride (already a `16`-byte multiple, so no
/// pad word is needed).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Query point `x`.
    px: f32,
    /// Query point `y`.
    py: f32,
    /// Query point `z`.
    pz: f32,
    /// Lattice period `x`.
    perx: f32,
    /// Lattice period `y`.
    pery: f32,
    /// Lattice period `z`.
    perz: f32,
    /// Finite-lattice limit `x`.
    limx: f32,
    /// Finite-lattice limit `y`.
    limy: f32,
    /// Finite-lattice limit `z`.
    limz: f32,
    /// Elongation half-extent `x`.
    hex: f32,
    /// Elongation half-extent `y`.
    hey: f32,
    /// Elongation half-extent `z`.
    hez: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `DomainResult`
/// struct: the three transformed points and the correction scalar packed as ten
/// scalars with two pad words to a `48`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `limited_repeat` point `x`.
    lr_x: f32,
    /// `limited_repeat` point `y`.
    lr_y: f32,
    /// `limited_repeat` point `z`.
    lr_z: f32,
    /// `mirror_repeat` point `x`.
    mr_x: f32,
    /// `mirror_repeat` point `y`.
    mr_y: f32,
    /// `mirror_repeat` point `z`.
    mr_z: f32,
    /// `elongate` displacement `x`.
    el_x: f32,
    /// `elongate` displacement `y`.
    el_y: f32,
    /// `elongate` displacement `z`.
    el_z: f32,
    /// `elongate_correction` scalar.
    corr: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
}

/// One query for the domain-operator twin: the query `point` plus the lattice
/// `period`, the finite-lattice `limit` and the elongation `half_extent`.
///
/// `point` is the evaluation position; `period` is the per-axis lattice period
/// shared by
/// [`limited_repeat`](prism_render_architecture::ray_scene::sdf_domain::limited_repeat)
/// and
/// [`mirror_repeat`](prism_render_architecture::ray_scene::sdf_domain::mirror_repeat);
/// `limit` is the per-axis instance cap of
/// [`limited_repeat`](prism_render_architecture::ray_scene::sdf_domain::limited_repeat);
/// `half_extent` is the per-axis core carved by
/// [`elongate`](prism_render_architecture::ray_scene::sdf_domain::elongate) and
/// [`elongate_correction`](prism_render_architecture::ray_scene::sdf_domain::elongate_correction).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfDomainRepeatQuery {
    /// Query point `[x, y, z]`.
    pub point: [f32; 3],
    /// Lattice period per axis.
    pub period: [f32; 3],
    /// Finite-lattice instance limit per axis.
    pub limit: [f32; 3],
    /// Elongation half-extent per axis.
    pub half_extent: [f32; 3],
}

impl SdfDomainRepeatQuery {
    /// Builds a query from the point, the lattice period, the finite-lattice
    /// limit and the elongation half-extent.
    #[must_use]
    pub const fn new(
        point: [f32; 3],
        period: [f32; 3],
        limit: [f32; 3],
        half_extent: [f32; 3],
    ) -> SdfDomainRepeatQuery {
        SdfDomainRepeatQuery {
            point,
            period,
            limit,
            half_extent,
        }
    }
}

/// One resolved query of the domain-operator twin: the three transformed points
/// and the elongation interior correction at the query point.
///
/// `limited_repeat` is
/// [`limited_repeat`](prism_render_architecture::ray_scene::sdf_domain::limited_repeat);
/// `mirror_repeat` is
/// [`mirror_repeat`](prism_render_architecture::ray_scene::sdf_domain::mirror_repeat);
/// `elongate` is
/// [`elongate`](prism_render_architecture::ray_scene::sdf_domain::elongate);
/// `elongate_correction` is
/// [`elongate_correction`](prism_render_architecture::ray_scene::sdf_domain::elongate_correction).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfDomainRepeatResult {
    /// Finite-lattice folded point `[x, y, z]`.
    pub limited_repeat: [f32; 3],
    /// Mirrored-lattice folded point `[x, y, z]`.
    pub mirror_repeat: [f32; 3],
    /// Elongation displacement `[x, y, z]`.
    pub elongate: [f32; 3],
    /// Elongation interior correction scalar (non-positive inside the inserted
    /// box, zero outside it).
    pub elongate_correction: f32,
}

/// Encodes one [`SdfDomainRepeatQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SdfDomainRepeatQuery) -> GpuQuery {
    GpuQuery {
        px: q.point[0],
        py: q.point[1],
        pz: q.point[2],
        perx: q.period[0],
        pery: q.period[1],
        perz: q.period[2],
        limx: q.limit[0],
        limy: q.limit[1],
        limz: q.limit[2],
        hex: q.half_extent[0],
        hey: q.half_extent[1],
        hez: q.half_extent[2],
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SdfDomainRepeatResult`].
fn decode_result(raw: &GpuResult) -> SdfDomainRepeatResult {
    SdfDomainRepeatResult {
        limited_repeat: [raw.lr_x, raw.lr_y, raw.lr_z],
        mirror_repeat: [raw.mr_x, raw.mr_y, raw.mr_z],
        elongate: [raw.el_x, raw.el_y, raw.el_z],
        elongate_correction: raw.corr,
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

/// A compiled, reusable domain-operator compute pipeline, twinning the `CPU`
/// golden
/// [`limited_repeat`](prism_render_architecture::ray_scene::sdf_domain::limited_repeat),
/// [`mirror_repeat`](prism_render_architecture::ray_scene::sdf_domain::mirror_repeat),
/// [`elongate`](prism_render_architecture::ray_scene::sdf_domain::elongate)
/// and
/// [`elongate_correction`](prism_render_architecture::ray_scene::sdf_domain::elongate_correction).
pub struct GpuSdfDomainRepeat {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfDomainRepeat {
    /// Compiles the kernel and builds the reusable pipeline on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdfDomainRepeat {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sdf_domain_repeat_module"),
            source: ShaderSource::Wgsl(SDF_DOMAIN_REPEAT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sdf_domain_repeat_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sdf_domain_repeat_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sdf_domain_repeat_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfDomainRepeat {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`SdfDomainRepeatResult`] per input, in order.
    ///
    /// The transformed points match the reference to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SdfDomainRepeatQuery],
    ) -> Vec<SdfDomainRepeatResult> {
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
            label: Some("prism_volumetric_sdf_domain_repeat_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_domain_repeat_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sdf_domain_repeat_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sdf_domain_repeat_bind_group"),
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
            label: Some("prism_volumetric_sdf_domain_repeat_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sdf_domain_repeat_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sdf_domain_repeat_pass"),
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
