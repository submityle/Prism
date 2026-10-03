//! `wgpu` compute twin of the ocean `clipmap` ring selection and geomorph
//! weighting inside the ocean level-of-detail contract
//! ([`ocean_lod`](prism_render_architecture::water::ocean_lod)).
//!
//! The `CPU` golden
//! [`ocean_lod`](prism_render_architecture::water::ocean_lod) owns the
//! deterministic ocean-surface LOD classification. This twin reproduces one
//! patch's LOD resolution on device: for a camera distance and an
//! [`OceanClipmapConfig`](prism_render_architecture::water::ocean_lod::OceanClipmapConfig)
//! it returns the selected `clipmap` ring and the continuous geomorph weight,
//! mirroring
//! [`resolve_ocean_patch`](prism_render_architecture::water::ocean_lod::resolve_ocean_patch)
//! and its two kernels
//! [`select_clipmap_ring`](prism_render_architecture::water::ocean_lod::select_clipmap_ring)
//! and
//! [`clipmap_morph_weight`](prism_render_architecture::water::ocean_lod::clipmap_morph_weight).
//! One thread resolves one patch.
//!
//! # What is twinned
//!
//! For one patch at `distance` and an
//! [`OceanClipmapConfig`](prism_render_architecture::water::ocean_lod::OceanClipmapConfig)
//! the kernel reproduces:
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
//! # What stays on the host
//!
//! The variable-length patch-binning plan of
//! [`ocean_lod`](prism_render_architecture::water::ocean_lod)
//! ([`bin_ocean_patches`](prism_render_architecture::water::ocean_lod::bin_ocean_patches))
//! stays host-side; it is container work with no fixed-width device analogue.
//! The patch's [`WaterBodyHandle`](prism_render_architecture::water::WaterBodyHandle)
//! is a pure pass-through in
//! [`resolve_ocean_patch`](prism_render_architecture::water::ocean_lod::resolve_ocean_patch),
//! so it never reaches the device. The host also owns the empty-batch
//! short-circuit (a storage buffer cannot be zero-sized) and the packing of the
//! public [`WaterOceanPatchQuery`] into its `std430` slot.
//!
//! # Correctness model
//!
//! The selected `ring` is pure integer classification and is asserted exactly.
//! The `morph` weight threads through a subtract, a divide and a `clamp`, so the
//! `CPU` and `GPU` are not bit-exact across the divide; it is asserted within a
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`). Because `ring` is a
//! discrete tier, fixtures and the random sweep keep `distance` well clear of
//! every ring's outer radius and of the morph band start, so the two agree on
//! every branch.
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

/// The portable core-`WGSL` ocean-patch kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`resolve_ocean_patch`](prism_render_architecture::water::ocean_lod::resolve_ocean_patch);
/// see the module documentation for the algorithm.
const WATER_OCEAN_PATCH_WGSL: &str = r#"
// Ocean clipmap patch twin: one thread resolves one patch's ring and geomorph
// weight, mirroring the CPU golden `water::ocean_lod` closed form with only
// min/clamp and + - * / over bounded ring loops. It owns no variable-length
// patch binning and no body-handle routing.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::ocean_lod；无第三方引擎
// 源码或衍生代码。

const EPS: f32 = 1.0e-6;
const MAX_RINGS: u32 = 16u;

struct Params {
    // Number of patches in the storage arrays; threads past this short-circuit.
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
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Result {
    ring: u32,
    morph: f32,
    pad0: u32,
    pad1: u32,
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
    out.ring = ring;
    out.morph = morph;
    out.pad0 = 0u;
    out.pad1 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the patch count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in
/// [`WATER_OCEAN_PATCH_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid patches in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one patch query: four scalars and a ring count
/// to a `32`-byte stride, matching the `WGSL` `Query` struct.
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
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one patch result, matching the `WGSL` `Result`
/// struct: the selected ring, the geomorph weight, and two pad words to a
/// `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Selected `clipmap` ring.
    ring: u32,
    /// Geomorph weight toward the next coarser ring, in `0..=1`.
    morph: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One patch query: a camera distance plus the fields of the reference
/// [`OceanClipmapConfig`](prism_render_architecture::water::ocean_lod::OceanClipmapConfig),
/// flattened.
///
/// `distance`, `inner_radius`, `radius_growth`, `morph_fraction` and
/// `ring_count` feed
/// [`resolve_ocean_patch`](prism_render_architecture::water::ocean_lod::resolve_ocean_patch).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterOceanPatchQuery {
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
}

/// One resolved patch result: the `clipmap` ring and geomorph weight the
/// reference
/// [`resolve_ocean_patch`](prism_render_architecture::water::ocean_lod::resolve_ocean_patch)
/// produces (the body handle is a host-side pass-through and is not carried).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterOceanPatchResult {
    /// Selected `clipmap` ring.
    pub ring: u32,
    /// Geomorph weight toward the next coarser ring, in `0..=1`.
    pub morph: f32,
}

/// Encodes one [`WaterOceanPatchQuery`] into its `std430` [`GpuQuery`] slot,
/// clamping `ring_count` to [`MAX_RINGS`] so the device growth loop stays
/// bounded.
fn encode_query(q: &WaterOceanPatchQuery) -> GpuQuery {
    let cap = MAX_RINGS as u32;
    GpuQuery {
        distance: q.distance,
        inner_radius: q.inner_radius,
        radius_growth: q.radius_growth,
        morph_fraction: q.morph_fraction,
        ring_count: q.ring_count.min(cap),
        pad0: 0,
        pad1: 0,
        pad2: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`WaterOceanPatchResult`].
fn decode_result(raw: &GpuResult) -> WaterOceanPatchResult {
    WaterOceanPatchResult {
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

/// A compiled, reusable ocean-patch compute pipeline, twinning the `clipmap`
/// ring selection and geomorph weighting of the `CPU` golden
/// [`ocean_lod`](prism_render_architecture::water::ocean_lod) module.
pub struct GpuWaterOceanPatch {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterOceanPatch {
    /// Compiles the ocean-patch kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterOceanPatch {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_ocean_patch"),
            source: ShaderSource::Wgsl(WATER_OCEAN_PATCH_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_ocean_patch_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_ocean_patch_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_ocean_patch_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterOceanPatch {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every patch in `queries` and returns one [`WaterOceanPatchResult`]
    /// per input, in order.
    ///
    /// The selected `ring` matches the reference exactly and the `morph` weight
    /// matches within the tolerance documented on this module. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterOceanPatchQuery],
    ) -> Vec<WaterOceanPatchResult> {
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
            label: Some("prism_volumetric_water_ocean_patch_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_ocean_patch_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_ocean_patch_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_ocean_patch_bind_group"),
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
            label: Some("prism_volumetric_water_ocean_patch_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_ocean_patch_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_ocean_patch_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per patch, flattened to a 1-D dispatch.
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
