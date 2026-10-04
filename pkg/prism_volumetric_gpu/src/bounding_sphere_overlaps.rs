//! `wgpu` compute twin of the bounding-sphere overlap test from the `CPU`
//! golden `prism_physics_core::collider::bounding_sphere`'s
//! `BoundingSphere::overlaps`.
//!
//! Two bounding spheres overlap when the squared distance between their centres
//! is at most the squared sum of their radii — the branch-free, square-root-free
//! broad-phase test used everywhere in collision culling. This module ports that
//! single stateless closed form onto the device: one thread resolves one query,
//! so a passing real-device parity test is direct evidence the ported kernel
//! computes the same overlap decision the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `overlaps` for one sphere pair with
//! explicit centres and radii:
//!
//! * If any of the eight input scalars is non-finite, the pair is invalid
//!   (`valid = 0`, `overlaps = 0`, `dist_sq = 0`).
//! * Otherwise `dist_sq = |center_a - center_b|^2`, `r = radius_a + radius_b`,
//!   and `overlaps = dist_sq <= r * r`, evaluated in exactly the golden operator
//!   order.
//!
//! # Correctness model
//!
//! The continuous `dist_sq` scalar threads through multiplies and adds that a
//! `GPU` may contract, so it is compared with an `abs <= 1e-4 || rel <= 1e-3`
//! tolerance (`REL_FLOOR = 1e-6`). The discrete `overlaps` and `valid` words are
//! compared exactly; because the overlap decision can flip under round-off when
//! `dist_sq` is within rounding of `r * r`, the parity fixtures hold every
//! sample clear of that knee (ordered margin), so the discrete decision is never
//! sensitive to `f32`/`f64` divergence.
//!
//! # Degenerate inputs
//!
//! A non-finite centre component or radius yields `valid = 0` with `overlaps = 0`
//! and `dist_sq = 0`. The test is pure multiply-add with no division, so there
//! is no divisor to guard. An empty query batch short-circuits on the host with
//! no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `+ - *`,
//! `select` and unsigned index arithmetic — with no `sin`, `cos`, `tan`, `exp`,
//! `log`, `pow`, no `round`, no `f32` remainder, no `sqrt` and no division, so it
//! runs unmodified on `Metal`, `Vulkan` and `DX12`. Finiteness is tested with
//! the ordered compare `abs(x) < 3.0e38` (which rejects both infinities and
//! `NaN`) rather than a bare `x == x`, so there is no `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::bounding_sphere`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` bounding-sphere overlap kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden `BoundingSphere::overlaps`; see the module
/// documentation for the closed form.
const BOUNDING_SPHERE_OVERLAPS_WGSL: &str = r#"
// Bounding-sphere overlap twin: one thread per query reproduces
// BoundingSphere::overlaps. It uses only the portable core-WGSL subset (abs,
// + - *, select plus unsigned index math), takes no optional feature, and has
// no loop and no branch, so it provably terminates. Finiteness is an ordered
// abs < 3.0e38 compare (rejecting infinities and NaN) fed to select; the
// overlap decision is dist_sq <= r*r via an ordered compare. No division, so no
// divisor guard is needed.

struct Params {
    // Number of valid queries in this dispatch.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Centre of the first sphere (ax, ay, az) and its radius ra.
    ax: f32,
    ay: f32,
    az: f32,
    ra: f32,
    // Centre of the second sphere (bx, by, bz) and its radius rb.
    bx: f32,
    by: f32,
    bz: f32,
    rb: f32,
}

struct OverlapResult {
    // 1 when the spheres overlap, 0 when disjoint or invalid.
    overlaps: u32,
    // Squared centre distance (continuous parity scalar).
    dist_sq: f32,
    // 1 when every input is finite, 0 otherwise.
    valid: u32,
    // Padding word to a 16-byte stride.
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<OverlapResult>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Every input scalar must be finite (ordered magnitude test rejecting both
    // infinities and NaN); no bare x == x anywhere.
    let fin =
        abs(q.ax) < 3.0e38 && abs(q.ay) < 3.0e38 && abs(q.az) < 3.0e38 && abs(q.ra) < 3.0e38 &&
        abs(q.bx) < 3.0e38 && abs(q.by) < 3.0e38 && abs(q.bz) < 3.0e38 && abs(q.rb) < 3.0e38;

    // Squared centre distance, in the golden's operator order.
    let dx = q.ax - q.bx;
    let dy = q.ay - q.by;
    let dz = q.az - q.bz;
    let dist_sq = dx * dx + dy * dy + dz * dz;

    // Radius sum squared and the overlap decision.
    let r = q.ra + q.rb;
    let rr = r * r;
    let hit = fin && (dist_sq <= rr);

    var res: OverlapResult;
    res.overlaps = select(0u, 1u, hit);
    res.dist_sq = select(0.0, dist_sq, fin);
    res.valid = select(0u, 1u, fin);
    res.pad0 = 0u;
    results[idx] = res;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`BOUNDING_SPHERE_OVERLAPS_WGSL`].
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
/// All eight lanes are scalar `f32`, so the layout is a flat `32`-byte stride
/// with alignment `4` and no internal padding, and a batch of two or more packs
/// contiguously with no vector alignment rule to trip.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    ax: f32,
    ay: f32,
    az: f32,
    ra: f32,
    bx: f32,
    by: f32,
    bz: f32,
    rb: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `OverlapResult`
/// struct: the discrete overlap word, the squared distance, the validity word
/// and one padding word. Four scalar lanes give a flat, hole-free `16`-byte
/// stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    overlaps: u32,
    dist_sq: f32,
    valid: u32,
    pad0: u32,
}

/// One query for the bounding-sphere overlap twin: two sphere centres and radii.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundingSphereOverlapsQuery {
    /// First sphere centre, x component.
    pub ax: f32,
    /// First sphere centre, y component.
    pub ay: f32,
    /// First sphere centre, z component.
    pub az: f32,
    /// First sphere radius.
    pub ra: f32,
    /// Second sphere centre, x component.
    pub bx: f32,
    /// Second sphere centre, y component.
    pub by: f32,
    /// Second sphere centre, z component.
    pub bz: f32,
    /// Second sphere radius.
    pub rb: f32,
}

impl BoundingSphereOverlapsQuery {
    /// Builds a query from the two sphere centres and radii.
    #[must_use]
    pub fn new(
        ax: f32,
        ay: f32,
        az: f32,
        ra: f32,
        bx: f32,
        by: f32,
        bz: f32,
        rb: f32,
    ) -> BoundingSphereOverlapsQuery {
        BoundingSphereOverlapsQuery {
            ax,
            ay,
            az,
            ra,
            bx,
            by,
            bz,
            rb,
        }
    }
}

/// One resolved answer for a single query, mirroring `BoundingSphere::overlaps`.
///
/// `overlaps` is `1` when the two spheres intersect and `0` when they are
/// disjoint or any input is non-finite. `dist_sq` is the squared centre distance
/// (zero when invalid), carried as a continuous parity scalar. `valid` is `1`
/// when every input is finite, `0` otherwise.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundingSphereOverlapsResult {
    /// `1` when the spheres overlap, `0` when disjoint or invalid.
    pub overlaps: u32,
    /// Squared centre distance, zero when invalid.
    pub dist_sq: f32,
    /// `1` when every input is finite, `0` otherwise.
    pub valid: u32,
}

/// Encodes one [`BoundingSphereOverlapsQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &BoundingSphereOverlapsQuery) -> GpuQuery {
    GpuQuery {
        ax: q.ax,
        ay: q.ay,
        az: q.az,
        ra: q.ra,
        bx: q.bx,
        by: q.by,
        bz: q.bz,
        rb: q.rb,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`BoundingSphereOverlapsResult`].
fn decode_result(raw: &GpuResult) -> BoundingSphereOverlapsResult {
    BoundingSphereOverlapsResult {
        overlaps: raw.overlaps,
        dist_sq: raw.dist_sq,
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

/// A compiled, reusable bounding-sphere overlap compute pipeline, twinning the
/// `CPU` golden `BoundingSphere::overlaps`.
pub struct GpuBoundingSphereOverlaps {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuBoundingSphereOverlaps {
    /// Compiles the bounding-sphere overlap kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBoundingSphereOverlaps {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_bounding_sphere_overlaps"),
            source: ShaderSource::Wgsl(BOUNDING_SPHERE_OVERLAPS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_bounding_sphere_overlaps_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_bounding_sphere_overlaps_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_bounding_sphere_overlaps_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBoundingSphereOverlaps {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`BoundingSphereOverlapsResult`] per input, in order.
    ///
    /// An empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[BoundingSphereOverlapsQuery],
    ) -> Vec<BoundingSphereOverlapsResult> {
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
            label: Some("prism_volumetric_bounding_sphere_overlaps_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_bounding_sphere_overlaps_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_bounding_sphere_overlaps_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_bounding_sphere_overlaps_bind_group"),
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
            label: Some("prism_volumetric_bounding_sphere_overlaps_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_bounding_sphere_overlaps_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_bounding_sphere_overlaps_pass"),
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
