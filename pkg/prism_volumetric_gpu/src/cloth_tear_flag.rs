//! `wgpu` compute twin of the cloth constraint-tearing decision, mirroring this
//! repository's `prism_physics_core::soft::damage::tearing::tear_flag`.
//!
//! A distance edge in a cloth solve is torn (permanently removed) once its
//! tensile strain exceeds a painted break threshold, so an over-stretched
//! garment rips instead of stretching without bound. This twin reproduces, for
//! one edge per thread, that single scalar decision:
//!
//! - The authored `break_strain` is sanitized: a `NaN` or negative threshold
//!   maps to `+inf`, so a mis-authored value tears nothing.
//! - A degenerate rest length (`<= EPS_REST`, with `EPS_REST = 1e-9`) never
//!   tears; `tear` is `0` while `valid` stays `1`.
//! - Otherwise the tensile strain `(length - rest_length) / rest_length` is
//!   compared to the sanitized threshold with a strict `>`; a strain exactly
//!   equal to the threshold does not tear.
//!
//! The array-batched `tear_flags` / `apply_tearing` drivers of the golden
//! module (which walk a `&[DistanceConstraint]`, resolve endpoint positions and
//! `retain` surviving edges) are intentionally not twinned; they are stateful
//! list reductions, whereas this kernel is a stateless per-edge map, one thread
//! per query.
//!
//! # Infinity model
//!
//! The scalar reference sanitizes `break_strain` to `f32::INFINITY`. `WGSL` has
//! no `f32` infinity literal, so the kernel substitutes the sentinel `3.0e38`
//! (just below `f32::MAX`, `3.4e38`). For every realistic finite strain this
//! sentinel yields the same `strain > threshold` decision as `+inf` (a strain
//! never reaches `3.0e38`), so the discrete `tear` flag agrees exactly. The
//! host oracle keeps the golden `f32::INFINITY`; both branches resolve to the
//! same boolean on the sampled domain.
//!
//! # Degenerate model
//!
//! There is no rejected / invalid state: the golden always returns a defined
//! boolean, so `valid` is unconditionally `1`. A degenerate rest length is a
//! normal, defined `tear = 0` outcome, not a rejection. The division by a near
//! zero rest length that would occur in the degenerate case is masked by an
//! ordered `rest_length <= EPS_REST` guard feeding `select`, so no `NaN`/`inf`
//! intermediate can leak into the result.
//!
//! # Precision model
//!
//! The golden path evaluates in `f32`; this twin and its host oracle both
//! evaluate the same closed form in `f32`. The only continuous intermediate is
//! the strain quotient; the published outputs `tear` and `valid` are discrete
//! `u32` flags compared with an exact `==`. Fixtures and the random sweep keep
//! the strain clear of the exact `strain == break_strain` knee so a few units
//! in the last place cannot flip the boolean.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `select`, ordered
//! comparisons and `+ - * /` — with no `sin`, `cos`, `tan`, `exp`, `log`,
//! `pow`, `round` or optional device feature, and no `u64`, `i64` or `f64`, so
//! it runs unmodified on `Metal`, `Vulkan` and `DX12`. `NaN` is detected with
//! the self-compare idiom `!(x == x)`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::damage::tearing::tear_flag`；无第三方引擎源码或衍生代码。

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
/// single source file. The single entry point `tear_flag` mirrors the
/// `break_strain` sanitize, the degenerate rest-length guard and the strict
/// strain comparison of the `CPU` golden; see the module documentation for the
/// algorithm.
const CLOTH_TEAR_FLAG_WGSL: &str = r#"
// Cloth constraint-tearing twin: one thread per edge decides whether the edge
// tears from its rest length, current length and break-strain threshold. It
// uses only the portable core-WGSL subset (select, ordered compares and
// + - * /) with no u64/i64/f64 and no transcendental, so it runs unmodified on
// Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 prism_physics_core::soft::damage::tearing::tear_flag；无第三方引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query: the edge rest length, its current length and the painted break
// strain threshold. Three f32 lanes pack to exactly 12 bytes.
struct Query {
    rest_length: f32,
    length: f32,
    break_strain: f32,
}

// One result: the tear flag and the always-1 valid flag.
struct Res {
    tear: u32,
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

// Smallest rest length considered valid; a degenerate edge at or below this
// never tears (EPS_REST in the golden module).
const EPS_REST: f32 = 1.0e-9;

// Sentinel standing in for f32::INFINITY (just below f32::MAX = 3.4e38); a real
// tensile strain never reaches it, so strain > sentinel matches strain > +inf.
const INF_SENTINEL: f32 = 3.0e38;

// NaN self-compare: a NaN is the only value not equal to itself. This is the
// single permitted use of f32 == in the kernel.
fn is_nan(x: f32) -> bool {
    return !(x == x);
}

@compute @workgroup_size(64)
fn tear_flag(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Sanitize break_strain: a NaN or negative threshold maps to +inf (nothing
    // tears). Ordered compare + self-compare, never a bare equality.
    let bad = is_nan(q.break_strain) || (q.break_strain < 0.0);
    let bs = select(q.break_strain, INF_SENTINEL, bad);

    // A degenerate rest length never tears; the guard also masks the division.
    let degenerate = q.rest_length <= EPS_REST;
    let strain = (q.length - q.rest_length) / q.rest_length;

    // Strict comparison: a strain exactly equal to the threshold does not tear.
    let tears = (strain > bs) && !degenerate;

    var out: Res;
    out.tear = select(0u, 1u, tears);
    // No rejected state: the golden always returns a defined boolean.
    out.valid = 1u;
    results[idx] = out;
}
"#;

/// One cloth tearing query: the edge rest length, its current length and the
/// painted break-strain threshold.
///
/// `break_strain` need not be in range; the kernel sanitizes a `NaN` or
/// negative threshold to `+inf` (nothing tears). A degenerate `rest_length`
/// (`<= 1e-9`) never tears. Derives only [`PartialEq`] (no [`Eq`] / [`Hash`])
/// because it holds `f32` parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClothTearFlagQuery {
    /// Edge rest length. A value `<= 1e-9` is degenerate and never tears.
    pub rest_length: f32,
    /// Edge current length (the separation of its two endpoints).
    pub length: f32,
    /// Painted break-strain threshold; a `NaN` or negative value is sanitized
    /// to `+inf` so nothing tears.
    pub break_strain: f32,
}

impl ClothTearFlagQuery {
    /// Builds a query from the rest length, current length and break strain.
    #[must_use]
    pub const fn new(rest_length: f32, length: f32, break_strain: f32) -> ClothTearFlagQuery {
        ClothTearFlagQuery {
            rest_length,
            length,
            break_strain,
        }
    }
}

/// The cloth tearing decision for one query, the host-side mirror of the
/// kernel's `Res` lane.
///
/// `tear` is `1` exactly when the edge has a valid rest length and a tensile
/// strain strictly exceeding the sanitized threshold; `valid` is the always-`1`
/// flag (the golden always produces a defined decision). Derives [`Eq`] because
/// both fields are integral.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClothTearFlagResult {
    /// `1` when the edge tears, `0` otherwise.
    pub tear: u32,
    /// Always `1`: the decision is always defined.
    pub valid: u32,
}

/// `repr(C)` `std430` layout of one packed query: three `f32` lanes, exactly
/// `12` bytes, as the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Edge rest length.
    rest_length: f32,
    /// Edge current length.
    length: f32,
    /// Break-strain threshold.
    break_strain: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &ClothTearFlagQuery) -> GpuQuery {
        GpuQuery {
            rest_length: query.rest_length,
            length: query.length,
            break_strain: query.break_strain,
        }
    }
}

/// `repr(C)` `std430` layout of one result: the tear flag and the `valid` flag
/// in the same order as the `WGSL` `Res` struct — `8` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Tear flag (`0` / `1`).
    tear: u32,
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

/// Decodes one packed `GpuResult` into the public [`ClothTearFlagResult`].
fn decode_result(raw: &GpuResult) -> ClothTearFlagResult {
    ClothTearFlagResult {
        tear: raw.tear,
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

/// On-device twin of the cloth constraint-tearing kernel.
///
/// Owns the compiled [`ComputePipeline`] and its [`BindGroupLayout`]; build it
/// once with [`GpuClothTearFlag::new`] and reuse it across
/// [`GpuClothTearFlag::evaluate`] calls.
pub struct GpuClothTearFlag {
    /// The compiled shader module (retained so the pipeline stays valid).
    #[expect(
        dead_code,
        reason = "retained so the compiled module outlives the pipeline"
    )]
    module: ShaderModule,
    /// The bind group layout shared by every dispatch.
    layout: BindGroupLayout,
    /// The compute pipeline running the `tear_flag` entry point.
    pipeline: ComputePipeline,
}

impl GpuClothTearFlag {
    /// Compiles the kernel and builds the reusable pipeline on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothTearFlag {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_cloth_tear_flag_shader"),
            source: ShaderSource::Wgsl(CLOTH_TEAR_FLAG_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_cloth_tear_flag_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_cloth_tear_flag_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_cloth_tear_flag_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("tear_flag"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuClothTearFlag {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every cloth tearing query on-device and returns one
    /// [`ClothTearFlagResult`] per input, in order.
    ///
    /// Each result equals the reference decision exactly (the published outputs
    /// are discrete flags). An empty input returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[ClothTearFlagQuery],
    ) -> Vec<ClothTearFlagResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cloth_tear_flag_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cloth_tear_flag_output"),
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
            label: Some("prism_volumetric_cloth_tear_flag_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_cloth_tear_flag_bind_group"),
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
            label: Some("prism_volumetric_cloth_tear_flag_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_cloth_tear_flag_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_cloth_tear_flag_pass"),
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
