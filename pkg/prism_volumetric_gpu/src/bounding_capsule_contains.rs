//! `wgpu` compute twin of the capsule membership test from the `CPU` golden
//! `prism_physics_core::collider::bounding_capsule`'s `BoundingCapsule::contains`.
//!
//! A point lies inside the capsule when its squared distance to the central
//! segment `[center_a, center_b]` is within an inflated radius `r`. The golden
//! derives `r` from `radius + radius * CONTAIN_EPS + eps` (with
//! `CONTAIN_EPS = 1e-5`) and tests `dsq <= r * r`. One capsule-plus-point query
//! is evaluated per thread; a zero-length segment collapses to a sphere via an
//! ordered guard, so there is no degenerate rejection and `valid` is always `1`.
//!
//! # Precision note
//!
//! The golden computes `distance_sq_to_segment` in `f64` for cross-platform
//! determinism, whereas this twin evaluates it in `f32` on the device. The
//! continuous `dsq` channel is therefore compared with an absolute-or-relative
//! tolerance, and the discrete `contains` flag can only disagree with the golden
//! within an `f32`-vs-`f64` sliver around the `dsq == r * r` boundary. The parity
//! fixtures stay far from that boundary (`|dsq - r*r|` large relative to the
//! tolerance) so the discrete flag never flips.
//!
//! The host oracle in the parity test independently reimplements the same
//! closed form in plain `f32`; it does not pull in `prism_physics_core`,
//! `prism_render_architecture` or `glam`.
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

/// The portable core-`WGSL` capsule-contains kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `BoundingCapsule::contains`; see the module documentation for
/// the algorithm and the `f32`-vs-`f64` precision note.
///
/// `CONTAIN_EPS` is written as `1e-5`, matching the golden's `f32` constant.
const BOUNDING_CAPSULE_CONTAINS_WGSL: &str = r#"
// Capsule-contains twin: one thread per query projects the point onto the
// central segment, measures the squared distance, and tests it against the
// inflated radius. No transcendental calls; the one division is guarded so the
// degenerate (zero-length) arm never produces inf/nan. All vectors are
// flattened to scalars.
// Provenance: 孪生自本仓 prism_physics_core::collider::bounding_capsule；无第三方引擎源码或衍生代码。

const CONTAIN_EPS: f32 = 1e-5;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // First cap centre (x, y, z).
    cax: f32,
    cay: f32,
    caz: f32,
    // Second cap centre (x, y, z).
    cbx: f32,
    cby: f32,
    cbz: f32,
    // Capsule radius.
    radius: f32,
    // Query point (x, y, z).
    px: f32,
    py: f32,
    pz: f32,
    // Caller-supplied absolute slack.
    eps: f32,
    pad0: f32,
}

struct Hit {
    // Squared distance from the point to the central segment.
    dsq: f32,
    // 1 when the point lies inside the inflated capsule, else 0.
    contains: u32,
    // Always 1; the closed form has no degenerate rejection.
    valid: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Hit>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let ca = vec3<f32>(q.cax, q.cay, q.caz);
    let cb = vec3<f32>(q.cbx, q.cby, q.cbz);
    let point = vec3<f32>(q.px, q.py, q.pz);

    let axis = cb - ca;
    let seg_len_sq = dot(axis, axis);

    // Guard the divisor: the degenerate arm (seg_len_sq <= 0) uses t = 0 and the
    // guarded denominator keeps the unselected arm free of inf/nan.
    let positive = seg_len_sq > 0.0;
    let denom = select(1.0, seg_len_sq, positive);
    let raw_t = dot(point - ca, axis) / denom;
    let t = select(0.0, clamp(raw_t, 0.0, 1.0), positive);

    let closest = ca + axis * t;
    let diff = point - closest;
    let dsq = dot(diff, diff);

    // r = radius + radius * CONTAIN_EPS + eps, then compare dsq <= r * r.
    let slack = q.radius * CONTAIN_EPS + q.eps;
    let r = q.radius + slack;

    var out: Hit;
    out.dsq = dsq;
    out.contains = select(0u, 1u, dsq <= r * r);
    out.valid = 1u;
    out.pad0 = 0u;

    results[idx] = out;
}
"#;

/// `repr(C)` `std430` layout of the dispatch parameters.
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
/// Eleven payload words plus one padding word keep the stride a flat `48` bytes,
/// a multiple of `16`, with every `vec3` flattened to scalars so no
/// vector-alignment surprise can appear.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    cax: f32,
    cay: f32,
    caz: f32,
    cbx: f32,
    cby: f32,
    cbz: f32,
    radius: f32,
    px: f32,
    py: f32,
    pz: f32,
    eps: f32,
    pad0: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Hit` struct.
/// Two payload words plus two padding words keep the stride a flat `16` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    dsq: f32,
    contains: u32,
    valid: u32,
    pad0: u32,
}

/// One capsule-contains query: the two cap centres, the capsule radius, the
/// query point, and the caller's absolute slack `eps`. Every vector is flattened
/// to scalars so the `std430` stride stays an unambiguous flat layout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundingCapsuleContainsQuery {
    /// First cap centre, x component.
    pub cax: f32,
    /// First cap centre, y component.
    pub cay: f32,
    /// First cap centre, z component.
    pub caz: f32,
    /// Second cap centre, x component.
    pub cbx: f32,
    /// Second cap centre, y component.
    pub cby: f32,
    /// Second cap centre, z component.
    pub cbz: f32,
    /// Capsule radius.
    pub radius: f32,
    /// Query point, x component.
    pub px: f32,
    /// Query point, y component.
    pub py: f32,
    /// Query point, z component.
    pub pz: f32,
    /// Caller-supplied absolute slack added to the inflated radius.
    pub eps: f32,
}

impl BoundingCapsuleContainsQuery {
    /// Builds a capsule-contains query from the two cap centres, the radius, the
    /// query point, and the absolute slack `eps`.
    #[must_use]
    pub fn new(
        center_a: [f32; 3],
        center_b: [f32; 3],
        radius: f32,
        point: [f32; 3],
        eps: f32,
    ) -> BoundingCapsuleContainsQuery {
        BoundingCapsuleContainsQuery {
            cax: center_a[0],
            cay: center_a[1],
            caz: center_a[2],
            cbx: center_b[0],
            cby: center_b[1],
            cbz: center_b[2],
            radius,
            px: point[0],
            py: point[1],
            pz: point[2],
            eps,
        }
    }
}

/// One resolved membership test: the squared distance to the segment, the
/// discrete `contains` flag, and a `valid` flag that is always `1`, since the
/// closed form never rejects an input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundingCapsuleContainsResult {
    /// Squared distance from the point to the central segment.
    pub dsq: f32,
    /// `1` when the point lies inside the inflated capsule, else `0`.
    pub contains: u32,
    /// Always `1`; the closed form has no degenerate rejection.
    pub valid: u32,
}

/// Encodes one [`BoundingCapsuleContainsQuery`] into its `std430` [`GpuQuery`].
fn encode_query(q: &BoundingCapsuleContainsQuery) -> GpuQuery {
    GpuQuery {
        cax: q.cax,
        cay: q.cay,
        caz: q.caz,
        cbx: q.cbx,
        cby: q.cby,
        cbz: q.cbz,
        radius: q.radius,
        px: q.px,
        py: q.py,
        pz: q.pz,
        eps: q.eps,
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`BoundingCapsuleContainsResult`].
fn decode_result(raw: &GpuResult) -> BoundingCapsuleContainsResult {
    BoundingCapsuleContainsResult {
        dsq: raw.dsq,
        contains: raw.contains,
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

/// A compiled, reusable capsule-contains compute pipeline, twinning the `CPU`
/// golden `BoundingCapsule::contains`.
pub struct GpuBoundingCapsuleContains {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuBoundingCapsuleContains {
    /// Compiles the capsule-contains kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBoundingCapsuleContains {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_bounding_capsule_contains"),
            source: ShaderSource::Wgsl(BOUNDING_CAPSULE_CONTAINS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_bounding_capsule_contains_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_bounding_capsule_contains_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_bounding_capsule_contains_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBoundingCapsuleContains {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`BoundingCapsuleContainsResult`] per input, in order.
    ///
    /// An empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[BoundingCapsuleContainsQuery],
    ) -> Vec<BoundingCapsuleContainsResult> {
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
            label: Some("prism_volumetric_bounding_capsule_contains_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_bounding_capsule_contains_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_bounding_capsule_contains_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_bounding_capsule_contains_bind_group"),
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
            label: Some("prism_volumetric_bounding_capsule_contains_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_bounding_capsule_contains_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_bounding_capsule_contains_pass"),
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
