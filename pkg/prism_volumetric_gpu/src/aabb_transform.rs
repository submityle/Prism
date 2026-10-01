//! `wgpu` compute twin of the affine `AABB` matrix-transform contract
//! ([`aabb_transform`](prism_render_architecture::particle::aabb_transform),
//! particle design §12, §13).
//!
//! The `CPU` golden
//! [`Aabb::transform_by_mat4`](prism_render_architecture::particle::aabb_transform::Aabb::transform_by_mat4)
//! maps one axis-aligned box through a column-major affine `mat4` using Arvo's
//! method: a box with center `c` and half-extent `h` maps to the box centered at
//! `M * c` whose half-extent along output axis `i` is `sum_j |m[j][i]| * h[j]`.
//! For a centered box this is the exact tight transformed `AABB`.
//! [`GpuAabbTransform`] is the on-device twin: one thread transforms one box, so
//! a passing real-device parity test is direct evidence the ported kernel runs
//! the same matrix algebra and classifies the same degenerate box the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each per-query answer the reference computes is reproduced: the transformed
//! box `min` and `max` corners. The kernel mirrors the reference branch for
//! branch — it short-circuits an empty input box (any axis with `min > max`) to
//! the empty-box identity (`min` at `f32::MAX`, `max` at `f32::MIN`), and
//! otherwise forms the center / half-extent, applies the full column-major
//! affine to the center and accumulates the absolute-value-weighted half-extent,
//! exactly as [`Aabb::transform_by_mat4`](prism_render_architecture::particle::aabb_transform::Aabb::transform_by_mat4)
//! does.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`
//! and `+ - *` — with no `sqrt`, `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no
//! rounding and no optional device feature, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. The matrix is passed as four `vec4` columns and read as
//! a column-major basis, matching the reference's `m[c]`-is-column convention.
//!
//! # Correctness model
//!
//! Each box is a fixed, non-reorderable sequence of multiplies and adds, so
//! `CPU` and `GPU` evaluate the same closed form. They are not bit-exact: a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate, perturbing
//! the low mantissa bits by a few units in the last place. The parity test
//! therefore asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`),
//! tight enough to catch a genuinely wrong port (a dropped `abs`, a swapped
//! column, a wrong translation lane) yet loose enough to admit legal fused
//! multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`aabb_transform`](prism_render_architecture::particle::aabb_transform);
//! no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::aabb_transform::Aabb;
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

/// The portable core-`WGSL` affine `AABB`-transform kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`Aabb::transform_by_mat4`](prism_render_architecture::particle::aabb_transform::Aabb::transform_by_mat4)
/// branch for branch; see the module documentation for the algorithm.
const AABB_TRANSFORM_WGSL: &str = r#"
// Affine AABB-transform twin: one thread transforms one axis-aligned box through
// a column-major affine mat4 by Arvo's method and writes the new box min/max. It
// mirrors the CPU golden `particle::aabb_transform::Aabb::transform_by_mat4`
// branch for branch, uses only the portable core-WGSL subset (abs/min/max and
// + - *), needs no sqrt and takes no optional feature, so it runs unmodified on
// Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's particle::aabb_transform;
// no third-party engine source or derived code.

// The empty-box sentinels: min seeded at +INF semantics (f32::MAX) and max at
// -INF semantics (f32::MIN), matching the reference `Aabb::empty`. These exact
// decimals round to the largest finite f32 of each sign, so they reproduce the
// reference sentinel bit-for-bit; used in place of writing an == / != on an f32.
const F32_MAX: f32 = 3.4028235e38;
const F32_MIN: f32 = -3.4028235e38;

struct Params {
    // Number of boxes to transform in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Column 0 of the column-major affine mat4 (basis x, w lane ignored).
    col0: vec4<f32>,
    // Column 1 of the affine mat4 (basis y).
    col1: vec4<f32>,
    // Column 2 of the affine mat4 (basis z).
    col2: vec4<f32>,
    // Column 3 of the affine mat4 (translation in xyz).
    col3: vec4<f32>,
    // Input box min corner; a pad lane follows.
    bmin: vec3<f32>,
    pad0: f32,
    // Input box max corner; a pad lane follows.
    bmax: vec3<f32>,
    pad1: f32,
}

struct Result {
    // Transformed box min corner; a pad lane follows.
    rmin: vec3<f32>,
    pad0: f32,
    // Transformed box max corner; a pad lane follows.
    rmax: vec3<f32>,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let lo = q.bmin;
    let hi = q.bmax;

    var out: Result;
    out.pad0 = 0.0;
    out.pad1 = 0.0;

    // An empty input box (min > max on any axis) maps to the empty-box identity,
    // exactly as the reference short-circuits before touching the matrix.
    if (lo.x > hi.x || lo.y > hi.y || lo.z > hi.z) {
        out.rmin = vec3<f32>(F32_MAX, F32_MAX, F32_MAX);
        out.rmax = vec3<f32>(F32_MIN, F32_MIN, F32_MIN);
        results[idx] = out;
        return;
    }

    // Center and half-extent of the input box.
    let center = (lo + hi) * 0.5;
    let half = (hi - lo) * 0.5;

    // Column-major affine basis columns and translation.
    let c0 = q.col0.xyz;
    let c1 = q.col1.xyz;
    let c2 = q.col2.xyz;
    let translation = q.col3.xyz;

    // New center = M * center (translation plus the basis-weighted center) and
    // new half-extent = sum_j |column_j| * half[j] (Arvo's method).
    let new_center = translation + c0 * center.x + c1 * center.y + c2 * center.z;
    let new_half = abs(c0) * half.x + abs(c1) * half.y + abs(c2) * half.z;

    out.rmin = new_center - new_half;
    out.rmax = new_center + new_half;
    results[idx] = out;
}
"#;

/// One affine `AABB`-transform query: a column-major affine `mat4` and the input
/// box, exactly the inputs the reference
/// [`Aabb::transform_by_mat4`](prism_render_architecture::particle::aabb_transform::Aabb::transform_by_mat4)
/// consumes. The matrix is column-major: `matrix[c]` is column `c`, so element
/// `(row i, col j)` is `matrix[j][i]` and the translation is `matrix[3][0..3]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AabbTransformQuery {
    /// The column-major affine `mat4` to apply.
    pub matrix: [[f32; 4]; 4],
    /// The input box to transform.
    pub aabb: Aabb,
}

impl AabbTransformQuery {
    /// Builds a query from a column-major affine `matrix` and an input `aabb`.
    #[must_use]
    pub const fn new(matrix: [[f32; 4]; 4], aabb: Aabb) -> AabbTransformQuery {
        AabbTransformQuery { matrix, aabb }
    }
}

/// The resolved answer for one query: the transformed box, matching the
/// reference `Aabb::transform_by_mat4` return value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AabbTransformResult {
    /// The transformed axis-aligned box.
    pub transformed: Aabb,
}

/// `repr(C)` `std430` layout of one packed query: four `vec4` matrix columns
/// (`64` bytes) followed by the input box as `(min.xyz, pad)` and
/// `(max.xyz, pad)` — `96` bytes, each `vec3` on its `16`-byte-aligned slot
/// exactly as the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Column `0` of the affine matrix.
    col0: [f32; 4],
    /// Column `1` of the affine matrix.
    col1: [f32; 4],
    /// Column `2` of the affine matrix.
    col2: [f32; 4],
    /// Column `3` of the affine matrix (translation in `xyz`).
    col3: [f32; 4],
    /// Input box `min` corner.
    bmin: [f32; 3],
    /// Padding lane after the `min` corner.
    pad0: f32,
    /// Input box `max` corner.
    bmax: [f32; 3],
    /// Padding lane after the `max` corner.
    pad1: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &AabbTransformQuery) -> GpuQuery {
        GpuQuery {
            col0: query.matrix[0],
            col1: query.matrix[1],
            col2: query.matrix[2],
            col3: query.matrix[3],
            bmin: query.aabb.min,
            pad0: 0.0,
            bmax: query.aabb.max,
            pad1: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: two `vec4` slots holding the
/// transformed `(min.xyz, pad)` and `(max.xyz, pad)` — `32` bytes matching the
/// `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Transformed box `min` corner.
    rmin: [f32; 3],
    /// Padding lane after the `min` corner.
    pad0: f32,
    /// Transformed box `max` corner.
    rmax: [f32; 3],
    /// Padding lane after the `max` corner.
    pad1: f32,
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

/// A compiled, reusable affine `AABB`-transform compute pipeline.
pub struct GpuAabbTransform {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuAabbTransform {
    /// Compiles the affine `AABB`-transform kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuAabbTransform {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_aabb_transform"),
            source: ShaderSource::Wgsl(AABB_TRANSFORM_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_aabb_transform_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_aabb_transform_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_aabb_transform_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuAabbTransform {
            module,
            layout,
            pipeline,
        }
    }

    /// Transforms every box on-device and returns one [`AabbTransformResult`]
    /// per input, in order.
    ///
    /// Each result equals the reference
    /// [`Aabb::transform_by_mat4`](prism_render_architecture::particle::aabb_transform::Aabb::transform_by_mat4)
    /// to within the tolerance documented on this module, reproducing the
    /// empty-box short-circuit as well. An empty input returns an empty vector
    /// with no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[AabbTransformQuery],
    ) -> Vec<AabbTransformResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_aabb_transform_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_aabb_transform_output"),
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
            label: Some("prism_volumetric_aabb_transform_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_aabb_transform_bind_group"),
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
            label: Some("prism_volumetric_aabb_transform_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_aabb_transform_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_aabb_transform_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per box, flattened to a 1-D dispatch.
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

/// Decodes one packed [`GpuResult`] into the public [`AabbTransformResult`].
fn decode_result(raw: &GpuResult) -> AabbTransformResult {
    AabbTransformResult {
        transformed: Aabb::new(raw.rmin, raw.rmax),
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
