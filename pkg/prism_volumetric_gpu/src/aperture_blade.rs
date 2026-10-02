//! `wgpu` compute twin of the analytic aperture-blade signed-distance-field
//! (`SDF`) golden
//! ([`aperture_blade`](prism_render_architecture::particle::aperture_blade),
//! particle design §16, §21).
//!
//! The `CPU` golden
//! [`Aperture`](prism_render_architecture::particle::aperture_blade::Aperture)
//! owns the closed-form distance to a camera-diaphragm aperture: the convex
//! intersection of the blade half-planes, optionally rounded toward a circle.
//! Three closed forms compose the field. The polygon `SDF`
//! ([`polygon_sdf`](prism_render_architecture::particle::aperture_blade::Aperture::polygon_sdf))
//! folds `max(dot(p, n_k) - apothem)` over every unit edge normal `n_k`, so it
//! is negative inside every half-plane, zero on an edge and positive outside.
//! The circle `SDF`
//! ([`circle_sdf`](prism_render_architecture::particle::aperture_blade::Aperture::circle_sdf))
//! is `length(p) - apothem`, the one genuine `sqrt`. The rounded field
//! ([`sdf`](prism_render_architecture::particle::aperture_blade::Aperture::sdf))
//! linearly blends the two by `roundness`:
//! `poly + (circle - poly) * roundness` (`0` = polygon, `1` = circle).
//!
//! [`GpuApertureBlade`] is the on-device twin: one thread per query point `p`
//! reproduces the same three closed forms branch for branch, folding the same
//! `max` over the same packed edge-normal array, so a passing real-device
//! parity test is direct evidence the ported kernel evaluates the same convex
//! half-plane intersection and the same polygon-to-circle blend the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! The single rounded field
//! [`sdf`](prism_render_architecture::particle::aperture_blade::Aperture::sdf)
//! is reproduced per query point `p`. The reference's empty-blade branch is
//! mirrored implicitly: with a zero `blade_count` the polygon term collapses to
//! the circle term, so the blend returns the circle `SDF` exactly as the golden
//! [`polygon_sdf`](prism_render_architecture::particle::aperture_blade::Aperture::polygon_sdf)
//! fallback does, with no divide-by-zero and no `NaN`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+ - * /`, a `max`
//! fold expressed as a guarded compare, `bitcast` for the fold seed and one
//! `sqrt` for the genuine Euclidean length — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `tan` and no optional device feature, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and one
//! `sqrt`, so `CPU` and `GPU` evaluate the same closed form in the same
//! associativity. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32` distance, tight
//! enough to catch a genuinely wrong port (a dropped edge, a swapped normal, a
//! wrong blend) yet loose enough to admit legal fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::aperture_blade`；
//! standard convex-polygon half-plane signed-distance field plus `wgpu` compute
//! dispatch; no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::aperture_blade::Aperture;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Maximum number of blade edge normals a single packed query carries.
///
/// A query's `edge_normals` are copied into a fixed-length `std430` lane array
/// of this size and the live prefix is bounded by the per-query `blade_count`,
/// so a diaphragm of up to `MAX_BLADES` blades evaluates with no host
/// allocation inside the dispatch. Physical camera apertures never exceed this
/// blade count. The matching `WGSL` `Query` struct declares the same length.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::aperture_blade`；
/// no third-party engine source or derived code.
pub const MAX_BLADES: usize = 16;

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` aperture-blade-`SDF` kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`Aperture`](prism_render_architecture::particle::aperture_blade::Aperture)
/// branch for branch; see the module documentation for the algorithm.
const APERTURE_BLADE_WGSL: &str = r#"
// Aperture-blade-SDF twin: one thread per query point reproduces the rounded
// aperture signed distance. It folds max(dot(p, n_k) - apothem) over the packed
// edge normals for the polygon SDF, takes length(p) - apothem for the circle
// SDF, and blends them by roundness, mirroring the CPU golden
// particle::aperture_blade branch for branch. It uses only the portable
// core-WGSL subset (+ - * / plus a guarded max compare, bitcast and one sqrt)
// and takes no optional feature, so it runs unmodified on Metal, Vulkan and
// DX12.
//
// Provenance: twinned from this repository's particle::aperture_blade; no
// third-party engine source or derived code.

// Fixed packed edge-normal lane count; must equal the host MAX_BLADES.
const MAX_BLADES: u32 = 16u;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Query point p.xy with the apothem and roundness filling the first slot.
    point: vec2<f32>,
    apothem: f32,
    roundness: f32,
    // Live blade count (0..=MAX_BLADES) with a padding tail filling the slot.
    blade_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    // Fixed-length edge-normal lanes; only the first blade_count are live.
    normals: array<vec2<f32>, 16>,
}

struct Result {
    // Rounded aperture signed distance plus three pad lanes filling one vec4.
    sdf: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Signed distance to the convex blade polygon: max over the live edge normals
// of dot(p, n_k) - apothem, mirroring the reference polygon_sdf. With no live
// blade the fold seed stays at negative infinity and the caller falls back to
// the circle SDF, so the empty-blade branch degenerates to a circle exactly as
// the golden does. The negative-infinity seed is built with bitcast, never a
// transcendental.
fn polygon_max(q: Query) -> f32 {
    var m: f32 = bitcast<f32>(0xff800000u);
    for (var k: u32 = 0u; k < q.blade_count; k = k + 1u) {
        let n = q.normals[k];
        let d = q.point.x * n.x + q.point.y * n.y;
        if (d > m) {
            m = d;
        }
    }
    return m;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Circle SDF: length(p) - apothem, the one genuine sqrt.
    let circle = sqrt(q.point.x * q.point.x + q.point.y * q.point.y) - q.apothem;

    // Polygon SDF: max half-plane distance minus apothem, or the circle SDF
    // when no blade is live (the fold seed stays at negative infinity).
    var poly: f32 = circle;
    if (q.blade_count > 0u) {
        poly = polygon_max(q) - q.apothem;
    }

    // Rounded blend: poly + (circle - poly) * roundness.
    var out: Result;
    out.sdf = poly + (circle - poly) * q.roundness;
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    out.pad2 = 0.0;
    results[idx] = out;
}
"#;

/// One aperture-blade `SDF` query: the point `p` evaluated against the aperture
/// whose convex half-planes are given by `edge_normals`, inradius `apothem` and
/// polygon-to-circle blend `roundness` — the same inputs the reference
/// [`Aperture`](prism_render_architecture::particle::aperture_blade::Aperture)
/// consumes.
///
/// Only the first [`MAX_BLADES`] edge normals are packed for the device; a
/// caller supplying more is truncated to that bound, matching the fixed `WGSL`
/// lane array.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::aperture_blade`；
/// no third-party engine source or derived code.
#[derive(Clone, Debug, PartialEq)]
pub struct ApertureBladeQuery {
    /// Unit outward normal of each blade edge, pre-rotated by the caller.
    pub edge_normals: Vec<[f32; 2]>,
    /// Inradius: perpendicular centre-to-edge distance.
    pub apothem: f32,
    /// Polygon-to-circle blend in `[0, 1]`: `0` = polygon, `1` = circle.
    pub roundness: f32,
    /// The query point whose signed distance to the aperture is evaluated.
    pub point: [f32; 2],
}

impl ApertureBladeQuery {
    /// Builds a query from the edge normals, inradius, roundness blend and the
    /// evaluation point.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::aperture_blade`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn new(
        edge_normals: Vec<[f32; 2]>,
        apothem: f32,
        roundness: f32,
        point: [f32; 2],
    ) -> ApertureBladeQuery {
        ApertureBladeQuery {
            edge_normals,
            apothem,
            roundness,
            point,
        }
    }
}

/// The resolved answer for one query: the rounded aperture signed distance,
/// mirroring
/// [`sdf`](prism_render_architecture::particle::aperture_blade::Aperture::sdf).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::aperture_blade`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ApertureBladeResult {
    /// Signed distance to the rounded aperture (negative inside, zero on the
    /// boundary, positive outside).
    pub sdf: f32,
}

/// Evaluates the `CPU` golden for one query, delegating to the reference
/// [`Aperture`](prism_render_architecture::particle::aperture_blade::Aperture)
/// so the host side and the device twin are checked against the same source of
/// truth.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::aperture_blade`；
/// no third-party engine source or derived code.
#[must_use]
pub fn golden(query: &ApertureBladeQuery) -> ApertureBladeResult {
    let aperture = Aperture::new(query.edge_normals.clone(), query.apothem, query.roundness);
    ApertureBladeResult {
        sdf: aperture.sdf(query.point),
    }
}

/// `repr(C)` `std430` layout of one packed query: a header `vec4` holding
/// `(point.xy, apothem, roundness)`, a second `vec4` holding
/// `(blade_count, pad, pad, pad)`, then the fixed `MAX_BLADES` edge-normal lanes
/// (`vec2` each), matching the `WGSL` `Query` struct lane for lane.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Query point `p.xy`.
    point: [f32; 2],
    /// Inradius.
    apothem: f32,
    /// Polygon-to-circle blend.
    roundness: f32,
    /// Live blade count, bounded by `MAX_BLADES`.
    blade_count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
    /// Fixed-length edge-normal lanes; only the first `blade_count` are live.
    normals: [[f32; 2]; MAX_BLADES],
}

impl GpuQuery {
    /// Packs one query into its `std430` image, copying up to [`MAX_BLADES`]
    /// edge normals and zero-filling the unused lanes.
    fn new(query: &ApertureBladeQuery) -> GpuQuery {
        let live = query.edge_normals.len().min(MAX_BLADES);
        let mut normals = [[0.0f32; 2]; MAX_BLADES];
        for (lane, normal) in normals.iter_mut().zip(query.edge_normals.iter()).take(live) {
            *lane = *normal;
        }
        GpuQuery {
            point: query.point,
            apothem: query.apothem,
            roundness: query.roundness,
            blade_count: live as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
            normals,
        }
    }
}

/// `repr(C)` `std430` layout of one result: a four-scalar slot
/// `(sdf, pad, pad, pad)` — `16` bytes matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Rounded aperture signed distance.
    sdf: f32,
    /// Padding lane.
    pad0: f32,
    /// Padding lane.
    pad1: f32,
    /// Padding lane.
    pad2: f32,
}

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `16`-byte-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of queries in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A compiled, reusable aperture-blade-`SDF` compute pipeline.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::aperture_blade`；
/// no third-party engine source or derived code.
pub struct GpuApertureBlade {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuApertureBlade {
    /// Compiles the aperture-blade-`SDF` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::aperture_blade`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuApertureBlade {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_aperture_blade"),
            source: ShaderSource::Wgsl(APERTURE_BLADE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_aperture_blade_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_aperture_blade_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_aperture_blade_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuApertureBlade {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`ApertureBladeResult`] per
    /// input, in order.
    ///
    /// Each result equals the reference
    /// [`sdf`](prism_render_architecture::particle::aperture_blade::Aperture::sdf)
    /// answer to within the tolerance documented on this module. An empty input
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::aperture_blade`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[ApertureBladeQuery],
    ) -> Vec<ApertureBladeResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_aperture_blade_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_aperture_blade_output"),
            size: out_bytes,
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
            label: Some("prism_volumetric_aperture_blade_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_aperture_blade_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_aperture_blade_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_aperture_blade_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_aperture_blade_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &stage, 0, out_bytes);
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

/// Decodes one packed [`GpuResult`] into the public [`ApertureBladeResult`].
fn decode_result(raw: &GpuResult) -> ApertureBladeResult {
    ApertureBladeResult { sdf: raw.sdf }
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
