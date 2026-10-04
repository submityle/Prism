//! `wgpu` compute twin of the bounding-capsule axis/height accessors from the
//! `CPU` golden `prism_physics_core::collider::bounding_capsule::BoundingCapsule`
//! (`::axis` and `::height`).
//!
//! A bounding capsule is given by its two end-cap centres `center_a`,
//! `center_b` and a radius. Two derived scalars/vectors describe its central
//! segment: the unit axis direction `axis = (center_b - center_a)` normalized,
//! degenerating to the zero vector when the two centres coincide (the capsule
//! has collapsed to a sphere), and the cylindrical `height`, the plain distance
//! between the two centres. This module ports both accessors onto the device:
//! one thread resolves one capsule, so a passing real-device parity test is
//! direct evidence the kernel reproduces the same axis direction and the same
//! height the reference does, including the degenerate zero-length guard.
//!
//! # What is twinned
//!
//! `BoundingCapsule::axis` — `(center_b - center_a).normalize_or_zero()` — and
//! `BoundingCapsule::height` — `center_a.distance(center_b)`. The golden's
//! `normalize_or_zero` returns the zero vector when the reciprocal length is not
//! a finite positive number, i.e. when the segment has zero length; this twin
//! reproduces that with an ordered guard `len > 0` and chooses the normalized
//! direction or the zero vector with `select`.
//!
//! # Correctness model
//!
//! The axis components and the height are continuous `f32`, compared with an
//! absolute-or-relative tolerance. The `valid` word is always `1`: neither
//! accessor has a rejection path (a zero-length capsule is a legal sphere whose
//! axis is simply the zero vector), so `valid` is carried only for layout
//! parity with the crate's other twins and documented as a constant here.
//!
//! The degenerate guard is an ordered compare (`len > 0`) and the outputs are
//! chosen with `select`, so there is no bare `f32` equality anywhere in the
//! kernel. The divisor is guarded with `select(1.0, len, ok)` so the
//! unselected arm never forms `inf` or `nan`, matching the semantics of glam's
//! `normalize_or_zero` (which keys off `length_recip().is_finite() && > 0`).
//!
//! # Degenerate inputs
//!
//! A capsule whose centres coincide (`center_a == center_b`) reports
//! `axis = (0, 0, 0)` and `height = 0`, exactly the golden's sphere case. An
//! empty query batch short-circuits on the host with no dispatch, since a
//! storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — ordered compares,
//! `select`, `sqrt`, and `+ - * /` on scalar `f32` — with no `sin`, `cos`,
//! `tan`, `exp`, `log`, `pow`, no `round`, no `f32` remainder, no bare `f32`
//! equality and no `u64`/`i64`/`f64`, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. Every vector is flattened to scalar `f32` lanes in the
//! storage buffers and recomposed by hand, so no vector alignment rule can
//! perturb the `std430` stride.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::bounding_capsule`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` bounding-capsule axis/height kernel, embedded inline
/// so the twin ships as a single source file. The single entry point `solve`
/// mirrors `BoundingCapsule::axis` and `::height`; see the module documentation
/// for the algorithm.
const BOUNDING_CAPSULE_AXIS_WGSL: &str = r#"
// Bounding-capsule axis/height twin: one thread per query reproduces
// BoundingCapsule::axis and ::height. It forms d = center_b - center_a, takes
// len = length(d) as the height, and emits axis = d/len when len > 0 or the
// zero vector otherwise, matching normalize_or_zero. It uses only the portable
// core-WGSL subset (ordered compares, select, sqrt, + - * / on scalar f32).

struct Params {
    // Number of valid queries in this dispatch.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // First cap centre (world space).
    ax: f32,
    ay: f32,
    az: f32,
    // Second cap centre (world space).
    bx: f32,
    by: f32,
    bz: f32,
}

struct AxisResult {
    // Unit axis direction, zero vector when degenerate.
    axis_x: f32,
    axis_y: f32,
    axis_z: f32,
    // Segment length (cylindrical height).
    height: f32,
    // Always 1: neither accessor rejects.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<AxisResult>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let center_a = vec3<f32>(q.ax, q.ay, q.az);
    let center_b = vec3<f32>(q.bx, q.by, q.bz);
    let delta = center_b - center_a;
    let len = sqrt(dot(delta, delta));

    // Ordered guard matching normalize_or_zero: normalize only when the length
    // is strictly positive, otherwise emit the zero vector. The divisor is
    // guarded so the unselected arm never forms inf/nan.
    let ok = len > 0.0;
    let inv = 1.0 / select(1.0, len, ok);
    let zero = vec3<f32>(0.0, 0.0, 0.0);
    let axis = select(zero, delta * inv, ok);

    var res: AxisResult;
    res.axis_x = axis.x;
    res.axis_y = axis.y;
    res.axis_z = axis.z;
    res.height = len;
    res.valid = 1u;
    results[idx] = res;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`BOUNDING_CAPSULE_AXIS_WGSL`].
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
/// All six lanes are scalar `f32`, so the layout is a flat `24`-byte stride with
/// alignment `4` and no internal padding, and a batch of two or more packs
/// contiguously with no vector alignment rule to trip.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    ax: f32,
    ay: f32,
    az: f32,
    bx: f32,
    by: f32,
    bz: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `AxisResult`
/// struct: a unit axis (three `f32`), the height and a `valid` word. Five scalar
/// lanes give a flat `20`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    axis_x: f32,
    axis_y: f32,
    axis_z: f32,
    height: f32,
    valid: u32,
}

/// One query for the bounding-capsule axis/height twin: the two end-cap centres
/// `center_a` (`ax`, `ay`, `az`) and `center_b` (`bx`, `by`, `bz`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundingCapsuleAxisQuery {
    /// First cap centre, x component.
    pub ax: f32,
    /// First cap centre, y component.
    pub ay: f32,
    /// First cap centre, z component.
    pub az: f32,
    /// Second cap centre, x component.
    pub bx: f32,
    /// Second cap centre, y component.
    pub by: f32,
    /// Second cap centre, z component.
    pub bz: f32,
}

impl BoundingCapsuleAxisQuery {
    /// Builds a query from the two cap centres, in field order.
    #[must_use]
    pub fn new(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32) -> BoundingCapsuleAxisQuery {
        BoundingCapsuleAxisQuery {
            ax,
            ay,
            az,
            bx,
            by,
            bz,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `BoundingCapsule::axis` and `::height`.
///
/// (`axis_x`, `axis_y`, `axis_z`) is the unit axis direction from `center_a` to
/// `center_b`, or the zero vector when the capsule has collapsed to a sphere.
/// `height` is the distance between the two centres. `valid` is always `1`:
/// neither accessor has a rejection path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundingCapsuleAxisResult {
    /// Unit axis direction, x component.
    pub axis_x: f32,
    /// Unit axis direction, y component.
    pub axis_y: f32,
    /// Unit axis direction, z component.
    pub axis_z: f32,
    /// Segment length (cylindrical height).
    pub height: f32,
    /// Always `1`: the accessors never reject.
    pub valid: u32,
}

/// Encodes one [`BoundingCapsuleAxisQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &BoundingCapsuleAxisQuery) -> GpuQuery {
    GpuQuery {
        ax: q.ax,
        ay: q.ay,
        az: q.az,
        bx: q.bx,
        by: q.by,
        bz: q.bz,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`BoundingCapsuleAxisResult`].
fn decode_result(raw: &GpuResult) -> BoundingCapsuleAxisResult {
    BoundingCapsuleAxisResult {
        axis_x: raw.axis_x,
        axis_y: raw.axis_y,
        axis_z: raw.axis_z,
        height: raw.height,
        valid: raw.valid,
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

/// A compiled, reusable bounding-capsule axis/height compute pipeline, twinning
/// the `CPU` golden `BoundingCapsule::axis` and `::height`.
pub struct GpuBoundingCapsuleAxis {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuBoundingCapsuleAxis {
    /// Compiles the bounding-capsule axis/height kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBoundingCapsuleAxis {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_bounding_capsule_axis"),
            source: ShaderSource::Wgsl(BOUNDING_CAPSULE_AXIS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_bounding_capsule_axis_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_bounding_capsule_axis_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_bounding_capsule_axis_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBoundingCapsuleAxis {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`BoundingCapsuleAxisResult`] per input, in order.
    ///
    /// An empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[BoundingCapsuleAxisQuery],
    ) -> Vec<BoundingCapsuleAxisResult> {
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
            label: Some("prism_volumetric_bounding_capsule_axis_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_bounding_capsule_axis_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_bounding_capsule_axis_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_bounding_capsule_axis_bind_group"),
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
            label: Some("prism_volumetric_bounding_capsule_axis_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_bounding_capsule_axis_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_bounding_capsule_axis_pass"),
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
