//! `wgpu` compute twin of the per-body ocean patch resolution that feeds the
//! `clipmap` binning in the ocean level-of-detail contract
//! ([`ocean_lod`](prism_render_architecture::water::ocean_lod)).
//!
//! The `CPU` golden
//! [`bin_ocean_patches`](prism_render_architecture::water::ocean_lod::bin_ocean_patches)
//! walks a batch of water bodies, resolves each one's `clipmap` ring and
//! geomorph weight at its camera distance via
//! [`resolve_ocean_patch`](prism_render_architecture::water::ocean_lod::resolve_ocean_patch),
//! and routes the resolved patch into a per-ring bucket (preserving input
//! order). This twin reproduces the data-parallel core of that walk on device:
//! one thread resolves one body's patch. The variable-length per-ring bucket
//! assembly — a container operation with no fixed-width device analogue — stays
//! on the host.
//!
//! # What is twinned
//!
//! For a batch of bodies, each paired with a camera `distance` and an
//! [`OceanClipmapConfig`](prism_render_architecture::water::ocean_lod::OceanClipmapConfig),
//! the kernel reproduces, per body:
//! - `ring_outer_radius(ring) = inner_radius * radius_growth^ring` via repeated
//!   multiplication (no `powf`), with `ring` clamped to the last ring.
//! - `ring_inner_radius(ring)`: `0` for ring `0`, else the previous ring's
//!   outer radius.
//! - `select_clipmap_ring(distance)`: the first ring whose outer radius is at
//!   least `distance`, clamping distances past the outermost ring to the last
//!   ring.
//! - `clipmap_morph_weight(distance)`: `0` inside the ring, ramping linearly to
//!   `1` across the outermost `morph_fraction` of the ring's band; a degenerate
//!   band or disabled morphing returns `1` past the ring and `0` otherwise;
//!   `EPS = 1e-6`.
//!
//! Each body's resolved patch carries its originating body identifier so the
//! host can reconstruct the ring buckets in input order.
//!
//! # What stays on the host
//!
//! The pairing of `bodies[i]` with `distances[i]` (a body with no matching
//! distance entry is skipped, mirroring the golden's `distances.get(i)`) is a
//! host-side slice walk, exposed through [`WaterOceanBin::pair_queries`]. The
//! variable-length per-ring bucket assembly of
//! [`OceanClipmapPlan`](prism_render_architecture::water::ocean_lod::OceanClipmapPlan)
//! stays host-side; it is container work with no fixed-width device analogue.
//! The host also owns the empty-batch short-circuit (a storage buffer cannot be
//! zero-sized) and the packing of the public [`WaterOceanBinQuery`] into its
//! `std430` slot.
//!
//! # Correctness model
//!
//! The selected `ring` and the carried body identifier are pure integer
//! classification and are asserted exactly. The `morph` weight threads through
//! a subtract, a divide and a `clamp`, so the `CPU` and `GPU` are not bit-exact
//! across the divide; it is asserted within a tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`). Because `ring` is a discrete tier, fixtures and the
//! random sweep keep `distance` well clear of every ring's outer radius and of
//! the morph band start, so the two agree on every branch.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `clamp`,
//! `+ - * /`, bounded loops over at most [`MAX_RINGS`] rings, and unsigned index
//! arithmetic — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, `sqrt`, no
//! inverse trigonometry, no `round` or `ceil`, and no `u64`/`u16`/`i64`/`f64`.
//! No optional device feature is required, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::ocean_lod`；无第三方引擎源码或衍生代码。
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

/// Maximum number of `clipmap` rings the device kernel bounds its growth loop
/// to. The reference
/// [`OceanClipmapConfig`](prism_render_architecture::water::ocean_lod::OceanClipmapConfig)
/// has no hard cap, so the twin fixes a fixed-width ceiling of `16`; the packed
/// `ring_count` is clamped to it so every bounded loop terminates.
pub const MAX_RINGS: usize = 16;

/// Recommended upper bound on the number of bodies submitted in a single
/// dispatch. The one-thread-per-body kernel has no fixed-width body array, so
/// this is a soft batching ceiling rather than a structural limit; callers with
/// larger batches should chunk their input.
pub const MAX_BODIES: usize = 1024;

/// The portable core-`WGSL` ocean-bin kernel, embedded inline so the twin ships
/// as a single source file. The single entry point `solve` mirrors the per-body
/// core of the `CPU` golden
/// [`bin_ocean_patches`](prism_render_architecture::water::ocean_lod::bin_ocean_patches),
/// namely
/// [`resolve_ocean_patch`](prism_render_architecture::water::ocean_lod::resolve_ocean_patch);
/// see the module documentation for the algorithm.
const WATER_OCEAN_BIN_WGSL: &str = r#"
// Ocean clipmap bin twin: one thread resolves one body's ring and geomorph
// weight, mirroring the per-body core of the CPU golden `water::ocean_lod`
// `bin_ocean_patches` with only min/clamp and + - * / over bounded ring loops.
// The variable-length per-ring bucket assembly stays on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::ocean_lod；无第三方引擎
// 源码或衍生代码。

const EPS: f32 = 1.0e-6;
const MAX_RINGS: u32 = 16u;

struct Params {
    // Number of bodies in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    distance: f32,
    inner_radius: f32,
    radius_growth: f32,
    morph_fraction: f32,
    ring_count: u32,
    body: u32,
    pad0: u32,
    pad1: u32,
}

struct Result {
    body: u32,
    ring: u32,
    morph: f32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Highest valid ring index: ring_count - 1, saturating at zero.
fn last_ring(q: Query) -> u32 {
    if (q.ring_count > 0u) {
        return q.ring_count - 1u;
    }
    return 0u;
}

// Outer radius of a ring: inner_radius * radius_growth^ring via repeated
// multiplication, with the ring index clamped to the last ring.
fn ring_outer_radius(ring: u32, q: Query) -> f32 {
    let clamped = min(ring, last_ring(q));
    var radius = q.inner_radius;
    for (var level: u32 = 0u; level < MAX_RINGS; level = level + 1u) {
        if (level >= clamped) {
            break;
        }
        radius = radius * q.radius_growth;
    }
    return radius;
}

// Inner radius of a ring: zero for ring 0, else the previous ring's outer edge.
fn ring_inner_radius(ring: u32, q: Query) -> f32 {
    if (ring == 0u) {
        return 0.0;
    }
    return ring_outer_radius(ring - 1u, q);
}

// Selects the ring that owns a distance: the first ring whose outer radius is
// at least the distance, clamping beyond the outermost ring to the last ring.
fn select_clipmap_ring(distance: f32, q: Query) -> u32 {
    let last = last_ring(q);
    for (var ring: u32 = 0u; ring < MAX_RINGS; ring = ring + 1u) {
        if (ring >= last) {
            break;
        }
        if (distance <= ring_outer_radius(ring, q)) {
            return ring;
        }
    }
    return last;
}

// Continuous geomorph weight in 0..=1 for a distance.
fn clipmap_morph_weight(distance: f32, ring: u32, q: Query) -> f32 {
    let inner = ring_inner_radius(ring, q);
    let outer = ring_outer_radius(ring, q);
    let band = outer - inner;
    if (band <= EPS) {
        if (distance > outer) {
            return 1.0;
        }
        return 0.0;
    }
    if (q.morph_fraction <= EPS) {
        if (distance > outer) {
            return 1.0;
        }
        return 0.0;
    }
    let morph_start = outer - band * min(q.morph_fraction, 1.0);
    if (distance <= morph_start) {
        return 0.0;
    }
    let denom = outer - morph_start;
    if (denom <= EPS) {
        return 1.0;
    }
    return clamp((distance - morph_start) / denom, 0.0, 1.0);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let ring = select_clipmap_ring(q.distance, q);
    let morph = clipmap_morph_weight(q.distance, ring, q);

    var out: Result;
    out.body = q.body;
    out.ring = ring;
    out.morph = morph;
    out.pad0 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the body count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in
/// [`WATER_OCEAN_BIN_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid bodies in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one body query: four scalars, a ring count and
/// a body identifier to a `32`-byte stride, matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Camera distance in meters.
    distance: f32,
    /// Outer radius of ring `0`.
    inner_radius: f32,
    /// Geometric growth factor of the ring outer radius per level.
    radius_growth: f32,
    /// Fraction of each ring's radial band used for geomorph blending.
    morph_fraction: f32,
    /// Number of concentric rings, clamped to `MAX_RINGS`.
    ring_count: u32,
    /// Originating body identifier, carried through for host bucket assembly.
    body: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// `repr(C)` `std430` layout of one resolved body, matching the `WGSL` `Result`
/// struct: the body identifier, the selected ring, the geomorph weight, and a
/// pad word to a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Originating body identifier.
    body: u32,
    /// Selected `clipmap` ring.
    ring: u32,
    /// Geomorph weight toward the next coarser ring, in `0..=1`.
    morph: f32,
    /// Padding word.
    pad0: u32,
}

/// One body query: a camera distance plus the fields of the reference
/// [`OceanClipmapConfig`](prism_render_architecture::water::ocean_lod::OceanClipmapConfig),
/// flattened, plus the originating body identifier.
///
/// `distance`, `inner_radius`, `radius_growth`, `morph_fraction` and
/// `ring_count` feed
/// [`resolve_ocean_patch`](prism_render_architecture::water::ocean_lod::resolve_ocean_patch);
/// `body` is carried through unchanged so the host can reassemble the ring
/// buckets of
/// [`bin_ocean_patches`](prism_render_architecture::water::ocean_lod::bin_ocean_patches)
/// in input order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterOceanBinQuery {
    /// Camera distance in meters.
    pub distance: f32,
    /// Outer radius of ring `0`, in meters.
    pub inner_radius: f32,
    /// Geometric growth factor of the ring outer radius per level.
    pub radius_growth: f32,
    /// Fraction of each ring's radial band used for geomorph blending, in
    /// `0..=1`.
    pub morph_fraction: f32,
    /// Number of concentric rings; clamped to [`MAX_RINGS`] on encode.
    pub ring_count: u32,
    /// Originating body identifier, carried through to the result.
    pub body: u32,
}

/// One resolved body result: the originating body identifier, the `clipmap`
/// ring and geomorph weight the reference
/// [`resolve_ocean_patch`](prism_render_architecture::water::ocean_lod::resolve_ocean_patch)
/// produces for it inside
/// [`bin_ocean_patches`](prism_render_architecture::water::ocean_lod::bin_ocean_patches).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterOceanBinResult {
    /// Originating body identifier.
    pub body: u32,
    /// Selected `clipmap` ring.
    pub ring: u32,
    /// Geomorph weight toward the next coarser ring, in `0..=1`.
    pub morph: f32,
}

/// Encodes one [`WaterOceanBinQuery`] into its `std430` [`GpuQuery`] slot,
/// clamping `ring_count` to [`MAX_RINGS`] so the device growth loop stays
/// bounded.
fn encode_query(q: &WaterOceanBinQuery) -> GpuQuery {
    let cap = MAX_RINGS as u32;
    GpuQuery {
        distance: q.distance,
        inner_radius: q.inner_radius,
        radius_growth: q.radius_growth,
        morph_fraction: q.morph_fraction,
        ring_count: q.ring_count.min(cap),
        body: q.body,
        pad0: 0,
        pad1: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`WaterOceanBinResult`].
fn decode_result(raw: &GpuResult) -> WaterOceanBinResult {
    WaterOceanBinResult {
        body: raw.body,
        ring: raw.ring,
        morph: raw.morph,
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

/// A compiled, reusable ocean-bin compute pipeline, twinning the per-body patch
/// resolution of the `CPU` golden
/// [`bin_ocean_patches`](prism_render_architecture::water::ocean_lod::bin_ocean_patches).
pub struct GpuWaterOceanBin {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterOceanBin {
    /// Compiles the ocean-bin kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterOceanBin {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_ocean_bin"),
            source: ShaderSource::Wgsl(WATER_OCEAN_BIN_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_ocean_bin_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_ocean_bin_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_ocean_bin_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterOceanBin {
            module,
            layout,
            pipeline,
        }
    }

    /// Pairs `bodies` with `distances` into one query per valid body, mirroring
    /// the host-side slice walk of the golden
    /// [`bin_ocean_patches`](prism_render_architecture::water::ocean_lod::bin_ocean_patches).
    ///
    /// A body with no matching distance entry is skipped (as the golden's
    /// `distances.get(i)` returns `None`), so the paired length is
    /// `min(bodies.len(), distances.len())`. Each query carries the shared
    /// `clipmap` configuration scalars and the body identifier.
    #[must_use]
    pub fn pair_queries(
        bodies: &[u32],
        distances: &[f32],
        ring_count: u32,
        inner_radius: f32,
        radius_growth: f32,
        morph_fraction: f32,
    ) -> Vec<WaterOceanBinQuery> {
        bodies
            .iter()
            .zip(distances.iter())
            .map(|(&body, &distance)| WaterOceanBinQuery {
                distance,
                inner_radius,
                radius_growth,
                morph_fraction,
                ring_count,
                body,
            })
            .collect()
    }

    /// Solves every body in `queries` and returns one [`WaterOceanBinResult`]
    /// per input, in order.
    ///
    /// The selected `ring` and carried `body` match the reference exactly and
    /// the `morph` weight matches within the tolerance documented on this
    /// module. An empty `queries` batch returns an empty vector with no dispatch
    /// issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterOceanBinQuery],
    ) -> Vec<WaterOceanBinResult> {
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
            label: Some("prism_volumetric_water_ocean_bin_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_ocean_bin_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_ocean_bin_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_ocean_bin_bind_group"),
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
            label: Some("prism_volumetric_water_ocean_bin_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_ocean_bin_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_ocean_bin_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per body, flattened to a 1-D dispatch.
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
