//! `wgpu` compute twin of the ocean `clipmap` ring selection and geomorph
//! weighting inside the surface level-of-detail stage
//! ([`ocean_lod`](prism_render_architecture::water::ocean_lod)).
//!
//! The `CPU` golden [`ocean_lod`](prism_render_architecture::water::ocean_lod)
//! lays the open ocean out as camera-centred concentric rings (a `clipmap`):
//! ring `0` is the finest patch out to `inner_radius`, and each outer ring
//! grows its radius by `radius_growth`. Two pure, deterministic classifications
//! drive the geometry stage from a caller-supplied camera distance: which ring
//! owns the distance
//! ([`select_clipmap_ring`](prism_render_architecture::water::ocean_lod::select_clipmap_ring)),
//! and the continuous geomorph blend weight within that ring
//! ([`clipmap_morph_weight`](prism_render_architecture::water::ocean_lod::clipmap_morph_weight)).
//! Both use only comparisons, division and integer powers of the growth factor
//! (repeated multiplication, never `powf`), with no transcendental and no `f32`
//! equality test.
//!
//! [`GpuWaterOceanClipmap`] is the on-device twin of exactly those two cores.
//! One thread resolves one camera-distance sample — its owning ring and its
//! morph weight — reproducing the reference's exact closed form, so a passing
//! real-device parity test is direct evidence the ported kernel classifies the
//! same way the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For one sample the kernel reproduces, in closed form:
//!
//! - `last_ring` as `ring_count.saturating_sub(1)` (`0` for a degenerate
//!   zero-ring config).
//! - `ring_outer_radius(ring)` as `inner_radius * radius_growth^min(ring, last)`
//!   via repeated multiplication, with a bounded loop capped at
//!   [`MAX_RINGS`](self) iterations.
//! - `ring_inner_radius(ring)` as `0` for ring `0`, else the previous ring's
//!   outer radius.
//! - [`select_clipmap_ring`](prism_render_architecture::water::ocean_lod::select_clipmap_ring):
//!   the first ring whose outer radius is at least `distance`, clamping beyond
//!   the outermost ring to `last_ring`.
//! - [`clipmap_morph_weight`](prism_render_architecture::water::ocean_lod::clipmap_morph_weight):
//!   `0` in the inner part of the selected ring, a linear ramp to `1` across the
//!   outermost `morph_fraction` of the ring's band, saturating at `1` past the
//!   outermost ring; a degenerate band (`<= EPS`) or disabled morphing
//!   (`morph_fraction <= EPS`) returns `1` past the ring's outer radius and `0`
//!   otherwise.
//!
//! # What stays on the host
//!
//! The spectral cascade fade layout, the per-body binning plan
//! (`bin_ocean_patches`) and every variable-length aggregate stay host-side; the
//! device sees one independent distance sample per thread. An empty batch
//! short-circuits with no dispatch, since a storage buffer cannot be zero-sized.
//! The caller must keep `ring_count <= MAX_RINGS`.
//!
//! # Correctness model
//!
//! The owning ring is a discrete classification, asserted bit-exact (`==`) in
//! the parity test; the morph weight threads through subtracts and a guarded
//! divide, so a `GPU` divide may land a few units in the last place from the
//! scalar reference and is asserted within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` (relative floor `1e-6`). The ring boundaries
//! (`distance ~ ring_outer_radius`) and the morph-band start are
//! discontinuities; fixtures and the randomized sweep keep every sample well
//! clear of both so `CPU` and `GPU` cannot straddle a boundary and flip a ring
//! or a morph branch.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `+ - * /`, unsigned index arithmetic and a bounded loop — with no
//! `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no
//! `smoothstep`, no `round` and no `sqrt`. No optional device feature is
//! required, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. Every loop
//! is capped at [`MAX_RINGS`](self) iterations, so the kernel provably
//! terminates.
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

/// Hard cap on the ring count the twin supports, matching the bounded `WGSL`
/// loop. Callers must keep [`OceanClipmapConfig::ring_count`](prism_render_architecture::water::ocean_lod::OceanClipmapConfig)
/// at or below this value.
pub const MAX_RINGS: u32 = 16;

/// The portable core-`WGSL` ocean `clipmap` twin, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`select_clipmap_ring`](prism_render_architecture::water::ocean_lod::select_clipmap_ring)
/// and
/// [`clipmap_morph_weight`](prism_render_architecture::water::ocean_lod::clipmap_morph_weight)
/// closed forms; see the module documentation for the algorithm.
const WATER_OCEAN_CLIPMAP_WGSL: &str = r#"
// Ocean clipmap twin: one thread resolves one camera-distance sample's owning
// clipmap ring and its continuous geomorph blend weight, mirroring the CPU
// golden `water::ocean_lod` closed forms with only min, max, clamp and
// + - * / plus a ring-count-bounded loop. It owns no cascade fade, no binning
// plan and no variable-length aggregate; those stay host-side.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::ocean_lod；无第三方引擎
// 源码或衍生代码。

// Shared band/degeneracy guard; mirrors the water module `EPS`.
const EPS: f32 = 1.0e-6;
// Bounded loop cap; mirrors the host-side MAX_RINGS.
const MAX_RINGS: u32 = 16u;

struct Params {
    // Number of samples in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Camera distance in meters feeding both classifications.
    distance: f32,
    // Outer radius of ring 0.
    inner_radius: f32,
    // Geometric growth factor of the ring outer radius per level.
    radius_growth: f32,
    // Fraction of each ring's radial band used for geomorph blending.
    morph_fraction: f32,
    // Number of concentric rings; last_ring = saturating_sub(ring_count, 1).
    ring_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Result {
    // Owning clipmap ring index.
    ring: u32,
    // Continuous geomorph blend weight in 0..=1.
    morph_weight: f32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// last_ring = ring_count.saturating_sub(1): 0 when no rings exist.
fn last_ring(ring_count: u32) -> u32 {
    return select(ring_count - 1u, 0u, ring_count == 0u);
}

// ring_outer_radius: inner_radius * radius_growth^min(ring, last) via repeated
// multiplication, matching the golden (no pow). The loop is capped at
// MAX_RINGS iterations so it provably terminates.
fn ring_outer_radius(inner_radius: f32, radius_growth: f32, last: u32, ring: u32) -> f32 {
    let clamped = min(ring, last);
    var radius = inner_radius;
    var level = 0u;
    loop {
        if (level >= clamped) { break; }
        if (level >= MAX_RINGS) { break; }
        radius = radius * radius_growth;
        level = level + 1u;
    }
    return radius;
}

// ring_inner_radius: 0 for ring 0, else the previous ring's outer radius.
fn ring_inner_radius(inner_radius: f32, radius_growth: f32, last: u32, ring: u32) -> f32 {
    if (ring == 0u) {
        return 0.0;
    }
    return ring_outer_radius(inner_radius, radius_growth, last, ring - 1u);
}

// select_clipmap_ring: first ring whose outer radius is at least distance,
// clamping beyond the outermost ring to last.
fn select_clipmap_ring(distance: f32, inner_radius: f32, radius_growth: f32, last: u32) -> u32 {
    var ring = 0u;
    loop {
        if (ring >= last) { break; }
        if (ring >= MAX_RINGS) { break; }
        if (distance <= ring_outer_radius(inner_radius, radius_growth, last, ring)) {
            return ring;
        }
        ring = ring + 1u;
    }
    return last;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let last = last_ring(q.ring_count);
    let ring = select_clipmap_ring(q.distance, q.inner_radius, q.radius_growth, last);

    let inner = ring_inner_radius(q.inner_radius, q.radius_growth, last, ring);
    let outer = ring_outer_radius(q.inner_radius, q.radius_growth, last, ring);
    let band = outer - inner;

    var weight = 0.0;
    if (band <= EPS || q.morph_fraction <= EPS) {
        // No usable band or morphing disabled: saturate past the ring, else 0.
        weight = select(0.0, 1.0, q.distance > outer);
    } else {
        let morph_start = outer - band * min(q.morph_fraction, 1.0);
        if (q.distance <= morph_start) {
            weight = 0.0;
        } else {
            let denom = outer - morph_start;
            if (denom <= EPS) {
                weight = 1.0;
            } else {
                weight = clamp((q.distance - morph_start) / denom, 0.0, 1.0);
            }
        }
    }

    var out: Result;
    out.ring = ring;
    out.morph_weight = weight;
    out.pad0 = 0u;
    out.pad1 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the sample count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WATER_OCEAN_CLIPMAP_WGSL`].
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

/// `repr(C)` `std430` layout of one sample query, matching the `WGSL` `Query`
/// struct: the camera distance and the three `clipmap` layout scalars, the ring
/// count, and three pad words to a `32`-byte stride of eight words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Camera distance in meters.
    distance: f32,
    /// Outer radius of ring `0`.
    inner_radius: f32,
    /// Geometric growth factor per ring.
    radius_growth: f32,
    /// Geomorph band fraction.
    morph_fraction: f32,
    /// Number of concentric rings.
    ring_count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one sample result, matching the `WGSL` `Result`
/// struct: the owning ring index, the morph weight, and two pad words to a
/// `16`-byte stride of four words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Owning `clipmap` ring index.
    ring: u32,
    /// Continuous geomorph blend weight.
    morph_weight: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One per-sample query for the ocean `clipmap` twin: the camera distance plus
/// the `clipmap` layout it is classified against.
///
/// `select_clipmap_ring` and `clipmap_morph_weight` both read
/// [`distance`](Self::distance) against the layout formed by
/// [`inner_radius`](Self::inner_radius), [`radius_growth`](Self::radius_growth),
/// [`morph_fraction`](Self::morph_fraction) and [`ring_count`](Self::ring_count).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterOceanClipmapQuery {
    /// Camera distance in meters.
    pub distance: f32,
    /// Outer radius of ring `0` (`> 0`).
    pub inner_radius: f32,
    /// Geometric growth factor of the ring outer radius per level (`> 1`).
    pub radius_growth: f32,
    /// Fraction of each ring's radial band used for geomorph blending, in
    /// `0..=1`.
    pub morph_fraction: f32,
    /// Number of concentric rings; must be at most [`MAX_RINGS`].
    pub ring_count: u32,
}

impl WaterOceanClipmapQuery {
    /// Builds a query from the camera distance and the `clipmap` layout.
    #[must_use]
    pub const fn new(
        distance: f32,
        inner_radius: f32,
        radius_growth: f32,
        morph_fraction: f32,
        ring_count: u32,
    ) -> WaterOceanClipmapQuery {
        WaterOceanClipmapQuery {
            distance,
            inner_radius,
            radius_growth,
            morph_fraction,
            ring_count,
        }
    }
}

/// One resolved sample of the ocean `clipmap` twin, mirroring the golden
/// [`select_clipmap_ring`](prism_render_architecture::water::ocean_lod::select_clipmap_ring)
/// ring index and the
/// [`clipmap_morph_weight`](prism_render_architecture::water::ocean_lod::clipmap_morph_weight)
/// blend weight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterOceanClipmapResult {
    /// Owning `clipmap` ring index.
    pub ring: u32,
    /// Continuous geomorph blend weight in `0..=1`.
    pub morph_weight: f32,
}

/// Encodes one [`WaterOceanClipmapQuery`] into its `std430` [`GpuQuery`] slot.
/// The real fields are a direct copy; the pad words are zeroed.
fn encode_query(q: &WaterOceanClipmapQuery) -> GpuQuery {
    GpuQuery {
        distance: q.distance,
        inner_radius: q.inner_radius,
        radius_growth: q.radius_growth,
        morph_fraction: q.morph_fraction,
        ring_count: q.ring_count,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`WaterOceanClipmapResult`].
fn decode_result(raw: &GpuResult) -> WaterOceanClipmapResult {
    WaterOceanClipmapResult {
        ring: raw.ring,
        morph_weight: raw.morph_weight,
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

/// A compiled, reusable ocean `clipmap` compute pipeline, twinning the numeric
/// core of the `CPU` golden
/// [`ocean_lod`](prism_render_architecture::water::ocean_lod).
pub struct GpuWaterOceanClipmap {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterOceanClipmap {
    /// Compiles the ocean `clipmap` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterOceanClipmap {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_ocean_clipmap"),
            source: ShaderSource::Wgsl(WATER_OCEAN_CLIPMAP_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_ocean_clipmap_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_ocean_clipmap_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_ocean_clipmap_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterOceanClipmap {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every sample in `queries` and returns one
    /// [`WaterOceanClipmapResult`] per input, in order.
    ///
    /// The owning ring matches the reference exactly; the morph weight matches
    /// within the tolerance documented on this module. An empty `queries` batch
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterOceanClipmapQuery],
    ) -> Vec<WaterOceanClipmapResult> {
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
            label: Some("prism_volumetric_water_ocean_clipmap_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_ocean_clipmap_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_ocean_clipmap_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_ocean_clipmap_bind_group"),
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
            label: Some("prism_volumetric_water_ocean_clipmap_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_ocean_clipmap_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_ocean_clipmap_pass"),
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
