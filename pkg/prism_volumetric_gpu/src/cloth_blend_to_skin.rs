//! `wgpu` compute twin of the cloth "blend to skin" painted pass, mirroring
//! this repository's `prism_render_architecture::cloth::painted::blend_to_skin`
//! together with `prism_render_architecture::cloth::asset::PaintedConstraint::clamped`.
//!
//! An authored cloth asset lets artists paint, per vertex, how strongly the
//! simulated particle should be pulled back toward its skinned anchor (the
//! garment following the animated body). The golden pass walks every particle
//! and, for the ones that are not pinned, mixes the simulated position toward
//! the anchor by the painted `blend_weight`: `0` keeps the free simulation, `1`
//! welds the particle onto the anchor. This twin reproduces, for one particle
//! per thread, that single closed form:
//!
//! - A pinned particle (`pinned != 0`) is never moved; its output position is
//!   its input position unchanged.
//! - Otherwise the painted `blend_weight` is clamped to `[0, 1]` (the relevant
//!   half of `PaintedConstraint::clamped`) and the output is
//!   `anchor + (particle - anchor) * blend_weight'`, evaluated per component.
//!
//! The array-batched `&mut [ClothParticle]` / `zip` driver of the golden
//! function and the other painted passes (backstop, `clamp_max_distance`,
//! anim-drive) are intentionally not twinned; they mutate shared slices and
//! walk zipped iterators, whereas this kernel is a stateless per-particle map,
//! one thread per query.
//!
//! # Clamp model
//!
//! The golden `clamped()` enforces `blend_weight.max(0.0).min(1.0)`. For every
//! finite input this is bit-identical to `clamp(blend_weight, 0.0, 1.0)`, so
//! the kernel uses `clamp` and the host oracle uses `max(0.0).min(1.0)`; they
//! agree exactly across the whole finite range. The pinned branch is selected
//! with an ordered `pinned != 0u` test feeding `select`, never a bare `f32`
//! equality.
//!
//! # Precision model
//!
//! The golden path evaluates in `f32`; this twin and its host oracle both
//! evaluate the same closed form in `f32`, each component a fixed,
//! non-reorderable `anchor + (particle - anchor) * weight` (one subtract and
//! one multiply-add). They are not bit-exact: a `GPU` may fuse the
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! asserts `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on the
//! continuous position components and an exact `==` on the discrete `valid`
//! flag.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `select`
//! and `+ - * /` — with no `sin`, `cos`, `tan`, `exp`, `log`, `pow`, `round`
//! or optional device feature, and no `u64`, `i64` or `f64`, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::cloth::painted::blend_to_skin` 与 `prism_render_architecture::cloth::asset::PaintedConstraint::clamped`；无第三方引擎源码或衍生代码。

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

/// Number of threads per workgroup. `64` is the portable, warp-friendly
/// default shared by every one-thread-per-element kernel in this crate.
const WORKGROUP_SIZE: u32 = 64;

/// Inlined `WGSL` compute shader source. Keeping it in the Rust binary avoids
/// shipping a sidecar asset and keeps the twin and its kernel versioned as a
/// single source file. The single entry point `blend_to_skin` mirrors the
/// pinned short-circuit, the `blend_weight` clamp and the per-component
/// anchor-to-particle mix of the `CPU` golden pass; see the module
/// documentation for the algorithm.
const CLOTH_BLEND_TO_SKIN_WGSL: &str = r#"
// Cloth "blend to skin" twin: one thread per particle evaluates the painted
// blend of a simulated position toward its skinned anchor. It uses only the
// portable core-WGSL subset (clamp/select and + - * /) with no u64/i64/f64 and
// no transcendental, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::cloth::painted::blend_to_skin 与 prism_render_architecture::cloth::asset::PaintedConstraint::clamped；无第三方引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query: the simulated particle position, the skinned anchor position, the
// painted blend weight and the pinned flag. Seven f32 lanes plus one u32 pack
// to exactly 32 bytes.
struct Query {
    particle_x: f32,
    particle_y: f32,
    particle_z: f32,
    anchor_x: f32,
    anchor_y: f32,
    anchor_z: f32,
    blend_weight: f32,
    pinned: u32,
}

// One result: the blended position and the always-1 valid flag.
struct Res {
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

// Mixes one component from the anchor toward the particle by the clamped weight:
// anchor + (particle - anchor) * weight. At weight 0 the result is the anchor
// (welded to skin), at weight 1 the free particle position.
fn mix_component(particle: f32, anchor: f32, weight: f32) -> f32 {
    return anchor + (particle - anchor) * weight;
}

@compute @workgroup_size(64)
fn blend_to_skin(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // PaintedConstraint::clamped(): blend_weight.max(0.0).min(1.0). For every
    // finite input this equals clamp(blend_weight, 0.0, 1.0).
    let weight = clamp(q.blend_weight, 0.0, 1.0);

    // Blended (not pinned) position, per component.
    let bx = mix_component(q.particle_x, q.anchor_x, weight);
    let by = mix_component(q.particle_y, q.anchor_y, weight);
    let bz = mix_component(q.particle_z, q.anchor_z, weight);

    // A pinned particle is never moved: its output is its input position. The
    // branch is chosen with an ordered pinned != 0u test feeding select.
    let is_pinned = q.pinned != 0u;
    let out_x = select(bx, q.particle_x, is_pinned);
    let out_y = select(by, q.particle_y, is_pinned);
    let out_z = select(bz, q.particle_z, is_pinned);

    var out: Res;
    out.pos_x = out_x;
    out.pos_y = out_y;
    out.pos_z = out_z;
    out.valid = 1u;
    results[idx] = out;
}
"#;

/// One cloth "blend to skin" query: the simulated particle position, the
/// skinned anchor position, the painted blend weight and the pinned flag.
///
/// The three position pairs are flattened to scalar lanes. `blend_weight` need
/// not be in range; the kernel clamps it to `[0, 1]` before use. `pinned` is a
/// `u32` boolean (`0` free, non-zero pinned); a pinned particle is returned
/// unchanged. Derives only [`PartialEq`] (no [`Eq`] / [`Hash`]) because it
/// holds `f32` parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClothBlendToSkinQuery {
    /// Simulated particle position, `x`.
    pub particle_x: f32,
    /// Simulated particle position, `y`.
    pub particle_y: f32,
    /// Simulated particle position, `z`.
    pub particle_z: f32,
    /// Skinned anchor position, `x`.
    pub anchor_x: f32,
    /// Skinned anchor position, `y`.
    pub anchor_y: f32,
    /// Skinned anchor position, `z`.
    pub anchor_z: f32,
    /// Painted blend weight; clamped to `[0, 1]` before use (`0` welds to the
    /// anchor, `1` keeps the free particle).
    pub blend_weight: f32,
    /// Pinned flag: `0` free, non-zero pinned (returned unchanged).
    pub pinned: u32,
}

impl ClothBlendToSkinQuery {
    /// Builds a query from the particle and anchor positions, the blend weight
    /// and the pinned flag.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the kernel query is a flat scalar record of two vec3s plus two scalars"
    )]
    pub const fn new(
        particle_x: f32,
        particle_y: f32,
        particle_z: f32,
        anchor_x: f32,
        anchor_y: f32,
        anchor_z: f32,
        blend_weight: f32,
        pinned: u32,
    ) -> ClothBlendToSkinQuery {
        ClothBlendToSkinQuery {
            particle_x,
            particle_y,
            particle_z,
            anchor_x,
            anchor_y,
            anchor_z,
            blend_weight,
            pinned,
        }
    }
}

/// The cloth "blend to skin" output for one query, the host-side mirror of the
/// kernel's `Res` lane.
///
/// The three components are the blended (or, when pinned, unchanged) particle
/// position; `valid` is the always-`1` flag (the pass always produces a
/// defined position). Derives only [`PartialEq`] (no [`Eq`] / [`Hash`])
/// because it holds `f32` parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClothBlendToSkinResult {
    /// Blended particle position, `x`.
    pub pos_x: f32,
    /// Blended particle position, `y`.
    pub pos_y: f32,
    /// Blended particle position, `z`.
    pub pos_z: f32,
    /// Always `1`: the pass always produces a defined position.
    pub valid: u32,
}

/// `repr(C)` `std430` layout of one packed query: six position lanes, the blend
/// weight and the pinned flag — `32` bytes, exactly as the `WGSL` `Query`
/// struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Particle position `x`.
    particle_x: f32,
    /// Particle position `y`.
    particle_y: f32,
    /// Particle position `z`.
    particle_z: f32,
    /// Anchor position `x`.
    anchor_x: f32,
    /// Anchor position `y`.
    anchor_y: f32,
    /// Anchor position `z`.
    anchor_z: f32,
    /// Painted blend weight.
    blend_weight: f32,
    /// Pinned flag.
    pinned: u32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &ClothBlendToSkinQuery) -> GpuQuery {
        GpuQuery {
            particle_x: query.particle_x,
            particle_y: query.particle_y,
            particle_z: query.particle_z,
            anchor_x: query.anchor_x,
            anchor_y: query.anchor_y,
            anchor_z: query.anchor_z,
            blend_weight: query.blend_weight,
            pinned: query.pinned,
        }
    }
}

/// `repr(C)` `std430` layout of one result: the three blended position
/// components and the `valid` flag in the same order as the `WGSL` `Res`
/// struct — `16` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Blended position `x`.
    pos_x: f32,
    /// Blended position `y`.
    pos_y: f32,
    /// Blended position `z`.
    pos_z: f32,
    /// Always-`1` valid flag.
    valid: u32,
}

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// round the uniform block out to `16` bytes.
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

/// Decodes one packed `GpuResult` into the public [`ClothBlendToSkinResult`].
fn decode_result(raw: &GpuResult) -> ClothBlendToSkinResult {
    ClothBlendToSkinResult {
        pos_x: raw.pos_x,
        pos_y: raw.pos_y,
        pos_z: raw.pos_z,
        valid: raw.valid,
    }
}

/// Builds one storage/uniform buffer bind-group-layout entry.
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

/// On-device twin of the cloth "blend to skin" kernel.
///
/// Owns the compiled [`ComputePipeline`] and its [`BindGroupLayout`]; build it
/// once with [`GpuClothBlendToSkin::new`] and reuse it across
/// [`GpuClothBlendToSkin::evaluate`] calls.
pub struct GpuClothBlendToSkin {
    /// The compiled shader module (retained so the pipeline stays valid).
    #[expect(
        dead_code,
        reason = "retained so the compiled module outlives the pipeline"
    )]
    module: ShaderModule,
    /// The bind group layout shared by every dispatch.
    layout: BindGroupLayout,
    /// The compute pipeline running the `blend_to_skin` entry point.
    pipeline: ComputePipeline,
}

impl GpuClothBlendToSkin {
    /// Compiles the kernel and builds the reusable pipeline on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothBlendToSkin {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_cloth_blend_to_skin_shader"),
            source: ShaderSource::Wgsl(CLOTH_BLEND_TO_SKIN_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_cloth_blend_to_skin_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_cloth_blend_to_skin_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_cloth_blend_to_skin_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("blend_to_skin"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuClothBlendToSkin {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every cloth "blend to skin" query on-device and returns one
    /// [`ClothBlendToSkinResult`] per input, in order.
    ///
    /// Each result equals the reference closed form to within the tolerance
    /// documented on this module. An empty input returns an empty vector with
    /// no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[ClothBlendToSkinQuery],
    ) -> Vec<ClothBlendToSkinResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cloth_blend_to_skin_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cloth_blend_to_skin_output"),
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
            label: Some("prism_volumetric_cloth_blend_to_skin_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_cloth_blend_to_skin_bind_group"),
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
            label: Some("prism_volumetric_cloth_blend_to_skin_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_cloth_blend_to_skin_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_cloth_blend_to_skin_pass"),
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
        debug_assert_eq!(raw.len(), count);

        raw.iter().map(decode_result).collect()
    }
}
