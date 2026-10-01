//! `wgpu` compute twin of the ribbon/trail *geometry expansion* golden
//! [`ribbon_geometry`](prism_render_architecture::particle::ribbon_geometry)
//! (design §15, "飘带/轨迹几何").
//!
//! The `CPU` golden turns an ordered centerline plus a matching per-point
//! `width` into the two camera-facing (or fixed-normal) corner points that flank
//! each centerline point, tagged with a normalized arc-length `UV`.v. This crate
//! is the on-device twin: it runs one thread per *output* centerline vertex and
//! reproduces exactly what
//! [`build_strip`](prism_render_architecture::particle::ribbon_geometry::build_strip),
//! [`build_strip_flat`](prism_render_architecture::particle::ribbon_geometry::build_strip_flat)
//! and
//! [`strip_indices`](prism_render_architecture::particle::ribbon_geometry::strip_indices)
//! produce. A passing real-device parity test is therefore direct evidence the
//! ported kernels estimate the same tangents, pick the same side (binormal)
//! axis, apply the same deterministic fallback, resolve the same widths and
//! accumulate the same arc length the reference does, not merely that the
//! shaders compile.
//!
//! # What is twinned
//!
//! Three portable pure functions are reproduced:
//!
//! 1. The centerline → camera-facing strip expansion
//!    [`build_strip`](prism_render_architecture::particle::ribbon_geometry::build_strip):
//!    central-difference tangents (one-sided at the endpoints), a side axis of
//!    `tangent x view` with `view = camera_position - point`, a deterministic
//!    perpendicular fallback when that cross product degenerates, half-`width`
//!    corner offsets and a length-parameterized `UV`.v.
//! 2. The fixed-normal variant
//!    [`build_strip_flat`](prism_render_architecture::particle::ribbon_geometry::build_strip_flat),
//!    identical except the side axis is `tangent x fixed_normal`; the host
//!    selects it with the [`RibbonStripQuery::facing`] flag so both share one
//!    kernel.
//! 3. The triangle-list index builder
//!    [`strip_indices`](prism_render_architecture::particle::ribbon_geometry::strip_indices),
//!    a pure `u32` kernel that emits `6 * (vertex_count - 1)` indices with the
//!    same alternating winding the reference pushes.
//!
//! The tangent normalizer, the side-axis fallback and the arc-length prefix sum
//! are reproduced step for step. In particular the `UV`.v prefix sum is summed
//! left-to-right in the same ascending order the reference walks its
//! `cumulative` vector, so the two accumulate the identical closed form.
//!
//! # Portability
//!
//! The kernels use only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `abs`, `floor`-class arithmetic, `+ - * /`, unsigned index math and
//! `sqrt` (through the hand-rolled vector length) — with no `sin`, `cos`,
//! `exp`, `log`, `pow` or optional device feature, so they run unmodified on
//! `Metal`, `Vulkan` and `DX12`. The only non-linear operation is `sqrt`, which
//! both the length and the normalizer need, exactly as the reference does.
//!
//! # Correctness model
//!
//! Each output vertex is a fixed, non-reorderable sequence of subtractions,
//! cross products, one reciprocal-`sqrt` normalization and a prefix-sum divide,
//! so `CPU` and `GPU` evaluate the same closed form in the same order. They are
//! not bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The parity test therefore asserts a tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`) on the continuous corner positions and `UV`.v, tight
//! enough to catch a genuinely wrong port (a swapped tangent difference, a wrong
//! fallback axis, a dropped half-`width`) yet loose enough to admit legal fused
//! multiply-add contraction. The integer index list is compared exactly.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::ribbon_geometry`；
//! 无第三方引擎源码或衍生代码。

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::ribbon_geometry::RibbonStripVertex;
use prism_render_architecture::particle::Vec3;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly size
/// used across this crate's kernels; the output vertices (or index spans) are
/// flattened to a single linear index so the dispatch stays one-dimensional.
const WORKGROUP_SIZE: u32 = 64;

/// Scalar floats emitted per strip vertex in the flat output buffer: the three
/// `left` components, the three `right` components and the `uv_v` scalar.
const FLOATS_PER_VERTEX: usize = 7;

/// Indices emitted per centerline span: two triangles of three indices each.
const INDICES_PER_SPAN: usize = 6;

/// The portable core-`WGSL` ribbon-geometry kernels, embedded inline so the twin
/// ships as a single source file. The entry points `expand_strip` and
/// `build_indices` mirror the `CPU` golden
/// [`build_strip`](prism_render_architecture::particle::ribbon_geometry::build_strip)
/// / [`build_strip_flat`](prism_render_architecture::particle::ribbon_geometry::build_strip_flat)
/// and
/// [`strip_indices`](prism_render_architecture::particle::ribbon_geometry::strip_indices);
/// see the module documentation for the algorithm.
const RIBBON_GEOMETRY_WGSL: &str = r#"
// Ribbon/trail geometry expansion twin: one thread per output centerline vertex
// resolves a central-difference tangent, a camera-facing or fixed-normal side
// (binormal) axis with a deterministic degenerate fallback, a non-negative
// half-width corner offset and a length-parameterized UV.v. A second entry
// point emits the triangle-list indices. Both mirror the CPU golden
// `particle::ribbon_geometry`, use only the portable core-WGSL subset
// (min/max/clamp/abs, + - * /, unsigned index math and sqrt) and take no
// optional feature, so they run unmodified on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::ribbon_geometry;
// no third-party engine source or derived code.

struct Params {
    // Number of centerline points (one output vertex each).
    count: u32,
    // Number of supplied widths (may be shorter than, equal to, or longer than
    // `count`, or zero).
    widths_len: u32,
    // 0 = camera-facing (`axis` is the camera position); 1 = fixed normal
    // (`axis` is the world-space normal).
    mode: u32,
    // Padding to a 16-byte boundary.
    pad0: u32,
    // Camera position (mode 0) or fixed normal (mode 1).
    axis: vec3<f32>,
    // Padding to a 16-byte boundary.
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> centerline: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read> widths: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_strip: array<f32>;

// Absolute tolerance for the degeneracy and total-length guards, matching the
// reference `EPS = 1e-6`. Squared comparisons use `EPS * EPS = 1e-12`, which is
// also the reference `EPS_LEN_SQ` the vector normalizer gates on.
const EPS: f32 = 1.0e-6;
const EPS_LEN_SQ: f32 = 1.0e-12;

// Squared Euclidean length `dot(v, v)`.
fn len_sq(v: vec3<f32>) -> f32 {
    return dot(v, v);
}

// Unit vector along `v`, or the zero vector when `v` is (numerically) zero, so
// normalization never yields NaN. Mirrors `Vec3::normalize_or_zero`: gate on
// `len_sq > EPS_LEN_SQ`, then scale by `1 / sqrt(len_sq)`.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let ls = len_sq(v);
    if (ls > EPS_LEN_SQ) {
        return v * (1.0 / sqrt(ls));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Unit tangent at centerline index `i`: central difference at interior points,
// one-sided at the endpoints, zero for a lone point.
fn tangent_at(i: u32) -> vec3<f32> {
    let count = params.count;
    if (count < 2u) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    var raw: vec3<f32>;
    if (i == 0u) {
        raw = centerline[1] - centerline[0];
    } else if (i + 1u == count) {
        raw = centerline[count - 1u] - centerline[count - 2u];
    } else {
        raw = centerline[i + 1u] - centerline[i - 1u];
    }
    return normalize_or_zero(raw);
}

// A deterministic unit vector perpendicular to `tangent`, used when the primary
// side axis degenerates. Crosses with whichever cardinal axis is least parallel
// to the tangent, falling back to world +X when the tangent itself is zero.
fn fallback_side(tangent: vec3<f32>) -> vec3<f32> {
    var reference: vec3<f32>;
    if (abs(tangent.x) <= 0.9) {
        reference = vec3<f32>(1.0, 0.0, 0.0);
    } else {
        reference = vec3<f32>(0.0, 1.0, 0.0);
    }
    let side = normalize_or_zero(cross(tangent, reference));
    if (len_sq(side) > EPS_LEN_SQ) {
        return side;
    }
    return vec3<f32>(1.0, 0.0, 0.0);
}

// Side (binormal) axis `tangent x other`, with the deterministic fallback when
// the two are (near) parallel or the tangent is zero. `other` is the view ray
// `camera - point` in camera mode or the fixed normal in flat mode.
fn side_axis(tangent: vec3<f32>, other: vec3<f32>) -> vec3<f32> {
    let raw = normalize_or_zero(cross(tangent, other));
    if (len_sq(raw) > EPS_LEN_SQ) {
        return raw;
    }
    return fallback_side(tangent);
}

// Resolves the strip width at index `i`: `widths[i]` when present, else the last
// supplied width, else `1.0`; the result is clamped to be non-negative.
fn resolve_width(i: u32) -> f32 {
    var raw: f32;
    if (i < params.widths_len) {
        raw = widths[i];
    } else if (params.widths_len > 0u) {
        raw = widths[params.widths_len - 1u];
    } else {
        raw = 1.0;
    }
    return max(raw, 0.0);
}

@compute @workgroup_size(64)
fn expand_strip(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let count = params.count;
    if (i >= count) {
        return;
    }
    let point = centerline[i];
    let tangent = tangent_at(i);

    // Camera mode crosses the tangent with the view ray `camera - point`; flat
    // mode crosses it with the supplied fixed normal.
    var other: vec3<f32>;
    if (params.mode == 0u) {
        other = params.axis - point;
    } else {
        other = params.axis;
    }
    let side = side_axis(tangent, other);

    // Cumulative arc length to `i` and the total, summed left-to-right in the
    // same ascending order the reference walks its `cumulative` vector so the
    // floating-point sums agree. The head is always 0.
    var run = 0.0;
    var cumulative_i = 0.0;
    for (var j = 1u; j < count; j = j + 1u) {
        let d = centerline[j] - centerline[j - 1u];
        run = run + sqrt(len_sq(d));
        if (j == i) {
            cumulative_i = run;
        }
    }
    let total = run;

    var uv_v = 0.0;
    if (total > EPS) {
        uv_v = clamp(cumulative_i / total, 0.0, 1.0);
    }

    let half = 0.5 * resolve_width(i);
    let offset = side * half;
    let left = point + offset;
    let right = point - offset;

    let base = i * 7u;
    out_strip[base] = left.x;
    out_strip[base + 1u] = left.y;
    out_strip[base + 2u] = left.z;
    out_strip[base + 3u] = right.x;
    out_strip[base + 4u] = right.y;
    out_strip[base + 5u] = right.z;
    out_strip[base + 6u] = uv_v;
}

struct IndexParams {
    // Number of centerline vertices; the host guarantees `>= 2` before any
    // dispatch, so `vertex_count - 1u` never underflows here.
    vertex_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> idx_params: IndexParams;
@group(0) @binding(1) var<storage, read_write> out_indices: array<u32>;

@compute @workgroup_size(64)
fn build_indices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let spans = idx_params.vertex_count - 1u;
    if (i >= spans) {
        return;
    }
    // Centerline point `k` occupies `left` at index `2*k` and `right` at
    // `2*k + 1`. Each span emits two triangles with alternating winding:
    // (left0, right0, left1) then (right0, right1, left1).
    let left0 = 2u * i;
    let right0 = 2u * i + 1u;
    let left1 = 2u * (i + 1u);
    let right1 = 2u * (i + 1u) + 1u;
    let base = i * 6u;
    out_indices[base] = left0;
    out_indices[base + 1u] = right0;
    out_indices[base + 2u] = left1;
    out_indices[base + 3u] = right0;
    out_indices[base + 4u] = right1;
    out_indices[base + 5u] = left1;
}
"#;

/// One ribbon-strip expansion request: the centerline, the per-point widths and
/// the side-axis selector (design §15).
///
/// Mirrors the `(centerline, widths, camera_position | fixed_normal)` triple the
/// reference
/// [`build_strip`](prism_render_architecture::particle::ribbon_geometry::build_strip)
/// / [`build_strip_flat`](prism_render_architecture::particle::ribbon_geometry::build_strip_flat)
/// consume. `axis` is the camera position when [`RibbonStripQuery::facing`] is
/// `true` (the camera-facing variant) or the fixed world-space normal when it is
/// `false` (the beam-like world-locked variant). Derives only [`PartialEq`] (no
/// `Eq`/`Hash`) because the payload holds `f32` data.
#[derive(Clone, Debug, PartialEq)]
pub struct RibbonStripQuery {
    /// The ordered centerline points to expand into a strip.
    pub centerline: Vec<Vec3>,
    /// Per-point widths; index `i` uses `widths[i]` when present, else the last
    /// supplied width, else `1.0`, with negatives clamped to `0.0`.
    pub widths: Vec<f32>,
    /// Camera position (when [`RibbonStripQuery::facing`]) or fixed normal.
    pub axis: Vec3,
    /// `true` selects the camera-facing variant (`tangent x view`); `false`
    /// selects the fixed-normal variant (`tangent x axis`).
    pub facing: bool,
}

/// Uniform parameters for the strip-expansion dispatch. `repr(C)` layout
/// matching `Params` in [`RIBBON_GEOMETRY_WGSL`]: the centerline and width
/// counts, the mode selector, one pad word, the `axis` triple and one trailing
/// pad — `32` bytes, each field at the `std140` uniform offset the shader
/// expects (the `vec3` field is 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct StripParams {
    /// Number of centerline points.
    count: u32,
    /// Number of supplied widths.
    widths_len: u32,
    /// `0` camera-facing, `1` fixed normal.
    mode: u32,
    /// Padding word.
    pad0: u32,
    /// Camera position (mode `0`) or fixed normal (mode `1`), x component.
    axis_x: f32,
    /// `axis` y component.
    axis_y: f32,
    /// `axis` z component.
    axis_z: f32,
    /// Padding word completing the 16-byte `vec3` slot.
    pad1: f32,
}

impl StripParams {
    /// Packs one strip request's counts, mode and axis.
    fn new(count: usize, widths_len: usize, facing: bool, axis: Vec3) -> StripParams {
        StripParams {
            count: count as u32,
            widths_len: widths_len as u32,
            mode: if facing { 0 } else { 1 },
            pad0: 0,
            axis_x: axis.x,
            axis_y: axis.y,
            axis_z: axis.z,
            pad1: 0.0,
        }
    }
}

/// Uniform parameters for the index-building dispatch. `repr(C)` layout matching
/// `IndexParams` in [`RIBBON_GEOMETRY_WGSL`]: the vertex count and three pad
/// words — `16` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct IndexParams {
    /// Number of centerline vertices.
    vertex_count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A device-visible centerline point: the `xyz` triple padded to the 16-byte
/// stride a `WGSL` `array<vec3<f32>>` storage binding uses.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuVec3 {
    /// X component.
    x: f32,
    /// Y component.
    y: f32,
    /// Z component.
    z: f32,
    /// Padding to the 16-byte `vec3` stride.
    pad: f32,
}

/// A compiled, reusable ribbon-geometry pipeline pair: the strip expansion and
/// the triangle-index builder.
pub struct GpuRibbonGeometry {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    strip_layout: BindGroupLayout,
    index_layout: BindGroupLayout,
    pipeline_strip: ComputePipeline,
    pipeline_index: ComputePipeline,
}

impl GpuRibbonGeometry {
    /// Compiles the ribbon-geometry strip and index kernels on `ctx`.
    ///
    /// The kernels use only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRibbonGeometry {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_ribbon_geometry"),
            source: ShaderSource::Wgsl(RIBBON_GEOMETRY_WGSL.into()),
        });
        let strip_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_ribbon_geometry_strip_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let index_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_ribbon_geometry_index_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let strip_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_ribbon_geometry_strip_pipeline_layout"),
            bind_group_layouts: &[Some(&strip_layout)],
            immediate_size: 0,
        });
        let index_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_ribbon_geometry_index_pipeline_layout"),
            bind_group_layouts: &[Some(&index_layout)],
            immediate_size: 0,
        });
        let pipeline_strip = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_ribbon_geometry_strip_pipeline"),
            layout: Some(&strip_pipeline_layout),
            module: &module,
            entry_point: Some("expand_strip"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_index = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_ribbon_geometry_index_pipeline"),
            layout: Some(&index_pipeline_layout),
            module: &module,
            entry_point: Some("build_indices"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRibbonGeometry {
            module,
            strip_layout,
            index_layout,
            pipeline_strip,
            pipeline_index,
        }
    }

    /// Expands `query.centerline` into a ribbon strip, returning one
    /// [`RibbonStripVertex`] per centerline point in input order.
    ///
    /// The result equals
    /// [`build_strip`](prism_render_architecture::particle::ribbon_geometry::build_strip)`(centerline, widths, axis)`
    /// when [`RibbonStripQuery::facing`] is `true`, or
    /// [`build_strip_flat`](prism_render_architecture::particle::ribbon_geometry::build_strip_flat)`(centerline, widths, axis)`
    /// when it is `false`, to within the tolerance documented on this module. An
    /// empty centerline yields an empty strip and issues no dispatch (a storage
    /// buffer cannot be zero-sized).
    #[must_use]
    pub fn build_strip(
        &self,
        ctx: &GpuContext,
        query: &RibbonStripQuery,
    ) -> Vec<RibbonStripVertex> {
        let count = query.centerline.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_points: Vec<GpuVec3> = query
            .centerline
            .iter()
            .map(|p| GpuVec3 {
                x: p.x,
                y: p.y,
                z: p.z,
                pad: 0.0,
            })
            .collect();

        // A storage buffer cannot be zero-sized; when no widths are supplied the
        // kernel ignores the contents (gated on `widths_len == 0`), so a single
        // placeholder float keeps the binding valid.
        let widths_data: Vec<f32> = if query.widths.is_empty() {
            vec![0.0]
        } else {
            query.widths.clone()
        };

        let params = StripParams::new(count, query.widths.len(), query.facing, query.axis);
        let out_floats = count * FLOATS_PER_VERTEX;
        let out_bytes = (out_floats * size_of::<f32>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ribbon_geometry_strip_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let points_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ribbon_geometry_centerline"),
            contents: bytemuck::cast_slice(&gpu_points),
            usage: BufferUsages::STORAGE,
        });
        let widths_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ribbon_geometry_widths"),
            contents: bytemuck::cast_slice(&widths_data),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ribbon_geometry_strip_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ribbon_geometry_strip_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_ribbon_geometry_strip_bind_group"),
            layout: &self.strip_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: points_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: widths_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_ribbon_geometry_strip_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_ribbon_geometry_strip_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_strip);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per output centerline vertex, flattened to a 1-D
            // dispatch.
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
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
        let flat = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        stage.unmap();

        let mut strip: Vec<RibbonStripVertex> = Vec::with_capacity(count);
        for chunk in flat.chunks_exact(FLOATS_PER_VERTEX) {
            strip.push(RibbonStripVertex {
                left: Vec3::new(chunk[0], chunk[1], chunk[2]),
                right: Vec3::new(chunk[3], chunk[4], chunk[5]),
                uv_v: chunk[6],
            });
        }
        debug_assert_eq!(strip.len(), count);
        strip
    }

    /// Builds the triangle-list indices for a strip of `vertex_count` centerline
    /// vertices, returning `6 * (vertex_count - 1)` indices in span order.
    ///
    /// The result equals
    /// [`strip_indices`](prism_render_architecture::particle::ribbon_geometry::strip_indices)`(vertex_count)`
    /// exactly (an integer kernel, compared bit-for-bit). A `vertex_count` below
    /// `2` yields an empty list and issues no dispatch.
    #[must_use]
    pub fn strip_indices(&self, ctx: &GpuContext, vertex_count: u32) -> Vec<u32> {
        if vertex_count < 2 {
            return Vec::new();
        }
        let device = ctx.device();

        let spans = (vertex_count - 1) as usize;
        let out_len = spans * INDICES_PER_SPAN;
        let out_bytes = (out_len * size_of::<u32>()) as u64;

        let params = IndexParams {
            vertex_count,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ribbon_geometry_index_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ribbon_geometry_index_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ribbon_geometry_index_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_ribbon_geometry_index_bind_group"),
            layout: &self.index_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_ribbon_geometry_index_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_ribbon_geometry_index_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_index);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per centerline span, flattened to a 1-D dispatch.
            let groups = (spans as u32).div_ceil(WORKGROUP_SIZE);
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
        let indices = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        stage.unmap();
        debug_assert_eq!(indices.len(), out_len);
        indices
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
