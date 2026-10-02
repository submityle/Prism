//! `wgpu` compute twin of the particle bounding-volume-hierarchy (`BVH`)
//! box-algebra, `Morton` encoding and tree-sizing arithmetic
//! ([`bvh`](prism_render_architecture::particle::bvh), particle design §12,
//! §13, §22).
//!
//! The `CPU` golden [`bvh`](prism_render_architecture::particle::bvh) owns the
//! small, verifiable numeric core a `GPU` broadphase shares: the hand-rolled
//! three-component vector algebra
//! ([`Vec3::add`](prism_render_architecture::particle::bvh::Vec3::add),
//! [`Vec3::sub`](prism_render_architecture::particle::bvh::Vec3::sub),
//! [`Vec3::scale`](prism_render_architecture::particle::bvh::Vec3::scale),
//! [`Vec3::min`](prism_render_architecture::particle::bvh::Vec3::min),
//! [`Vec3::max`](prism_render_architecture::particle::bvh::Vec3::max),
//! [`Vec3::length_squared`](prism_render_architecture::particle::bvh::Vec3::length_squared),
//! [`Vec3::length`](prism_render_architecture::particle::bvh::Vec3::length) and
//! [`Vec3::component`](prism_render_architecture::particle::bvh::Vec3::component)),
//! the axis-aligned box operators
//! ([`Aabb::union`](prism_render_architecture::particle::bvh::Aabb::union),
//! [`Aabb::center`](prism_render_architecture::particle::bvh::Aabb::center),
//! [`Aabb::half_extent`](prism_render_architecture::particle::bvh::Aabb::half_extent),
//! [`Aabb::surface_area`](prism_render_architecture::particle::bvh::Aabb::surface_area),
//! [`Aabb::longest_axis`](prism_render_architecture::particle::bvh::Aabb::longest_axis),
//! [`Aabb::contains`](prism_render_architecture::particle::bvh::Aabb::contains),
//! [`Aabb::is_empty`](prism_render_architecture::particle::bvh::Aabb::is_empty)
//! and [`Aabb::expand_point`](prism_render_architecture::particle::bvh::Aabb::expand_point)),
//! the grid quantization and `Morton` interleave
//! ([`quantize_to_grid`](prism_render_architecture::particle::bvh::quantize_to_grid),
//! [`expand_bits`](prism_render_architecture::particle::bvh::expand_bits) and
//! [`morton_code_3d`](prism_render_architecture::particle::bvh::morton_code_3d)),
//! the integer tree-sizing counts
//! ([`BvhTopology::internal_node_count`](prism_render_architecture::particle::bvh::BvhTopology::internal_node_count),
//! [`BvhTopology::total_node_count`](prism_render_architecture::particle::bvh::BvhTopology::total_node_count)
//! and
//! [`BvhTopology::max_stack_depth`](prism_render_architecture::particle::bvh::BvhTopology::max_stack_depth)),
//! and the surface-area-heuristic split cost
//! ([`sah_cost`](prism_render_architecture::particle::bvh::sah_cost)).
//! [`GpuBvh`] is the on-device twin: one thread answers one [`BvhQuery`], so a
//! passing real-device parity test is direct evidence the ported kernel
//! computes the same box algebra, `Morton` codes, node counts and split costs
//! the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! A single batched kernel evaluates a tagged union of independent queries, one
//! per thread. The vector and box operators are component-wise multiply-add;
//! [`Vec3::length`](prism_render_architecture::particle::bvh::Vec3::length) is
//! the only radical and `sqrt` is explicitly permitted. The `Morton` encoding
//! is pure integer bit arithmetic: the reference
//! [`expand_bits`](prism_render_architecture::particle::bvh::expand_bits) and
//! [`morton_code_3d`](prism_render_architecture::particle::bvh::morton_code_3d)
//! hold values in the low `30` bits of a `u64`, so the twin reproduces the exact
//! same bit-spread as a `u32` (the whole `30`-bit interleave fits), and the
//! parity test compares the `u32` answer against the reference `u64` by value.
//! The tree-sizing counts and the grid quantization are integer arithmetic with
//! a bounded bit-shift `log2` loop.
//!
//! # What stays on the host (not twinned)
//!
//! `WGSL` has no `u64`, so the wide byte-sizing helper
//! [`BvhTopology::node_buffer_bytes`](prism_render_architecture::particle::bvh::BvhTopology::node_buffer_bytes)
//! (a `u64` stride-times-count product) must stay on the host, even though its
//! `u32` node count and `std430` stride are themselves twinnable. The
//! median-split tree builder
//! [`build_median_split`](prism_render_architecture::particle::bvh::build_median_split)
//! is recursive host control flow over a sort-ordered, variable-length node
//! array, so it also remains on the host; the twin covers the per-node box
//! algebra and cost terms that builder calls, not the recursion itself.
//!
//! # Correctness model
//!
//! The continuous answers (vector algebra, box center / half-extent / surface
//! area, split cost) thread through multiplies, adds and one guarded divide, so
//! `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits. The parity test
//! therefore asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`,
//! `REL_FLOOR = 1e-6`) on every continuous quantity. The discrete answers (the
//! longest-axis index, the containment and emptiness verdicts, the grid cells,
//! the `Morton` codes and the node counts) are integer or boolean
//! classifications and are compared exactly.
//!
//! # Degenerate inputs
//!
//! A degenerate parent box (`total_area <= 0`) collapses
//! [`sah_cost`](prism_render_architecture::particle::bvh::sah_cost) to the bare
//! traversal cost instead of dividing by zero, matching the reference. A
//! degenerate grid domain (`hi <= lo`) or zero resolution collapses the
//! quantized cell to `0`, so the kernel never divides by zero and never indexes
//! out of range. The longest-axis tie resolves toward the lower index using
//! `>=`, never an `f32` `==`. An empty query batch short-circuits on the host
//! with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `floor`, `abs`, `sqrt`, `dot`, `+ - * /`, unsigned bit and index
//! arithmetic — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse
//! trigonometry and no optional device feature, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`. The only loop is the bounded (`<= 32`
//! iteration) bit-shift `log2`, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::bvh`；无第三方引擎源码或衍生代码。
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

/// Operation tag selecting [`BvhQuery::Vec3Add`] on device.
const OP_VEC3_ADD: u32 = 0;
/// Operation tag selecting [`BvhQuery::Vec3Sub`] on device.
const OP_VEC3_SUB: u32 = 1;
/// Operation tag selecting [`BvhQuery::Vec3Scale`] on device.
const OP_VEC3_SCALE: u32 = 2;
/// Operation tag selecting [`BvhQuery::Vec3Min`] on device.
const OP_VEC3_MIN: u32 = 3;
/// Operation tag selecting [`BvhQuery::Vec3Max`] on device.
const OP_VEC3_MAX: u32 = 4;
/// Operation tag selecting [`BvhQuery::Vec3LengthSquared`] on device.
const OP_VEC3_LENGTH_SQUARED: u32 = 5;
/// Operation tag selecting [`BvhQuery::Vec3Length`] on device.
const OP_VEC3_LENGTH: u32 = 6;
/// Operation tag selecting [`BvhQuery::Vec3Component`] on device.
const OP_VEC3_COMPONENT: u32 = 7;
/// Operation tag selecting [`BvhQuery::AabbUnion`] on device.
const OP_AABB_UNION: u32 = 8;
/// Operation tag selecting [`BvhQuery::AabbCenter`] on device.
const OP_AABB_CENTER: u32 = 9;
/// Operation tag selecting [`BvhQuery::AabbHalfExtent`] on device.
const OP_AABB_HALF_EXTENT: u32 = 10;
/// Operation tag selecting [`BvhQuery::AabbSurfaceArea`] on device.
const OP_AABB_SURFACE_AREA: u32 = 11;
/// Operation tag selecting [`BvhQuery::AabbLongestAxis`] on device.
const OP_AABB_LONGEST_AXIS: u32 = 12;
/// Operation tag selecting [`BvhQuery::AabbContains`] on device.
const OP_AABB_CONTAINS: u32 = 13;
/// Operation tag selecting [`BvhQuery::AabbIsEmpty`] on device.
const OP_AABB_IS_EMPTY: u32 = 14;
/// Operation tag selecting [`BvhQuery::AabbExpandPoint`] on device.
const OP_AABB_EXPAND_POINT: u32 = 15;
/// Operation tag selecting [`BvhQuery::QuantizeToGrid`] on device.
const OP_QUANTIZE_TO_GRID: u32 = 16;
/// Operation tag selecting [`BvhQuery::ExpandBits`] on device.
const OP_EXPAND_BITS: u32 = 17;
/// Operation tag selecting [`BvhQuery::MortonCode3d`] on device.
const OP_MORTON_CODE_3D: u32 = 18;
/// Operation tag selecting [`BvhQuery::InternalNodeCount`] on device.
const OP_INTERNAL_NODE_COUNT: u32 = 19;
/// Operation tag selecting [`BvhQuery::TotalNodeCount`] on device.
const OP_TOTAL_NODE_COUNT: u32 = 20;
/// Operation tag selecting [`BvhQuery::MaxStackDepth`] on device.
const OP_MAX_STACK_DEPTH: u32 = 21;
/// Operation tag selecting [`BvhQuery::SahCost`] on device.
const OP_SAH_COST: u32 = 22;

/// The portable core-`WGSL` `BVH` kernel, embedded inline so the twin ships as a
/// single source file. The single entry point `solve` mirrors the `CPU` golden
/// [`bvh`](prism_render_architecture::particle::bvh) function for function; see
/// the module documentation for the algorithm.
const BVH_WGSL: &str = r#"
// Particle BVH twin: one thread per query reproduces the hand-rolled Vec3
// algebra, the AABB operators (union, center, half-extent, surface area,
// longest axis, contains, is-empty, expand-point), the grid quantization and
// 30-bit Morton interleave, the integer tree-sizing counts and the SAH split
// cost. It mirrors the CPU golden `particle::bvh` function for function, uses
// only the portable core-WGSL subset (min/max/clamp/floor/abs/sqrt/dot and
// + - * / plus unsigned bit and index math), needs no transcendental call and
// takes no optional feature, so it runs unmodified on Metal, Vulkan and DX12.
// The only loop is the bounded (<= 32 iteration) bit-shift log2, so the kernel
// provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::bvh；无第三方
// 引擎源码或衍生代码。

// Fixed traversal cost charged per internal node, matching SAH_TRAVERSAL_COST.
const SAH_TRAVERSAL_COST: f32 = 1.0;
// Cost charged per primitive intersection test, matching SAH_INTERSECT_COST.
const SAH_INTERSECT_COST: f32 = 2.0;
// Traversal-stack safety margin, matching SAFETY_MARGIN.
const SAFETY_MARGIN: u32 = 2u;

// Operation tags, matching the host op codes.
const OP_VEC3_ADD: u32 = 0u;
const OP_VEC3_SUB: u32 = 1u;
const OP_VEC3_SCALE: u32 = 2u;
const OP_VEC3_MIN: u32 = 3u;
const OP_VEC3_MAX: u32 = 4u;
const OP_VEC3_LENGTH_SQUARED: u32 = 5u;
const OP_VEC3_LENGTH: u32 = 6u;
const OP_VEC3_COMPONENT: u32 = 7u;
const OP_AABB_UNION: u32 = 8u;
const OP_AABB_CENTER: u32 = 9u;
const OP_AABB_HALF_EXTENT: u32 = 10u;
const OP_AABB_SURFACE_AREA: u32 = 11u;
const OP_AABB_LONGEST_AXIS: u32 = 12u;
const OP_AABB_CONTAINS: u32 = 13u;
const OP_AABB_IS_EMPTY: u32 = 14u;
const OP_AABB_EXPAND_POINT: u32 = 15u;
const OP_QUANTIZE_TO_GRID: u32 = 16u;
const OP_EXPAND_BITS: u32 = 17u;
const OP_MORTON_CODE_3D: u32 = 18u;
const OP_INTERNAL_NODE_COUNT: u32 = 19u;
const OP_TOTAL_NODE_COUNT: u32 = 20u;
const OP_MAX_STACK_DEPTH: u32 = 21u;
const OP_SAH_COST: u32 = 22u;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Operation tag selecting which twinned function this thread evaluates.
    op: u32,
    // Unsigned operand 0: leaf_count / Morton x / expand-bits v / left_n /
    // component axis / grid resolution.
    u0: u32,
    // Unsigned operand 1: Morton y / right_n.
    u1: u32,
    // Unsigned operand 2: Morton z.
    u2: u32,
    // Scalar operand 0: scale factor / left_area.
    scalar0: f32,
    // Scalar operand 1: right_area.
    scalar1: f32,
    // Scalar operand 2: total_area.
    scalar2: f32,
    pad0: f32,
    // Vector operand a: Vec3 left / box min / quantize point. A pad lane keeps
    // the vec3 16-byte aligned.
    vec_a: vec3<f32>,
    pad_a: f32,
    // Vector operand b: Vec3 right / box max / quantize domain min.
    vec_b: vec3<f32>,
    pad_b: f32,
    // Vector operand c: union second-box min / expand/contains point / quantize
    // domain max.
    vec_c: vec3<f32>,
    pad_c: f32,
    // Vector operand d: union second-box max.
    vec_d: vec3<f32>,
    pad_d: f32,
}

struct Result {
    // Operation tag echoed back so the host decodes the right union variant.
    op: u32,
    // Boolean verdict (0 / 1) for `contains` and `is_empty`.
    bool_flag: u32,
    // Unsigned result 0: longest axis / node counts / Morton code / expand-bits
    // / quantized cell x.
    uint0: u32,
    // Unsigned result 1: quantized cell y.
    uint1: u32,
    // Unsigned result 2: quantized cell z.
    uint2: u32,
    pad_u0: u32,
    pad_u1: u32,
    // Scalar result: length / length-squared / component / surface area / SAH
    // cost.
    scalar: f32,
    // Vector result 0: scaled / add / sub / min / max / center / half-extent /
    // union min / expand-point min. A pad lane follows.
    vec0: vec3<f32>,
    pad_v0: f32,
    // Vector result 1: union max / expand-point max.
    vec1: vec3<f32>,
    pad_v1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Surface area 2 * (dx*dy + dy*dz + dz*dx) of the box spanned by min/max,
// mirroring the reference `Aabb::surface_area`.
fn surface_area(lo: vec3<f32>, hi: vec3<f32>) -> f32 {
    let d = hi - lo;
    return 2.0 * (d.x * d.y + d.y * d.z + d.z * d.x);
}

// Longest-axis index (0/1/2) with ties toward the lower index, mirroring the
// reference `Aabb::longest_axis`; comparisons use `>=`, never `==`.
fn longest_axis(lo: vec3<f32>, hi: vec3<f32>) -> u32 {
    let d = hi - lo;
    if (d.x >= d.y && d.x >= d.z) {
        return 0u;
    } else if (d.y >= d.z) {
        return 1u;
    }
    return 2u;
}

// Whether `p` lies within the closed box on every axis, mirroring the reference
// `Aabb::contains`.
fn contains(lo: vec3<f32>, hi: vec3<f32>, p: vec3<f32>) -> bool {
    return p.x >= lo.x && p.x <= hi.x
        && p.y >= lo.y && p.y <= hi.y
        && p.z >= lo.z && p.z <= hi.z;
}

// Spreads the low 10 bits of `v` so two zero bits sit between each, the 30-bit
// Morton lane step, mirroring the reference `expand_bits` as a u32 (the whole
// interleave fits in the low 30 bits).
fn expand_bits(v: u32) -> u32 {
    var x = v & 0x3FFu;
    x = (x | (x << 16u)) & 0x030000FFu;
    x = (x | (x << 8u)) & 0x0300F00Fu;
    x = (x | (x << 4u)) & 0x030C30C3u;
    x = (x | (x << 2u)) & 0x09249249u;
    return x;
}

// Interleaves three 10-bit grid coordinates into a 30-bit Morton code,
// mirroring the reference `morton_code_3d`.
fn morton_code_3d(x: u32, y: u32, z: u32) -> u32 {
    return expand_bits(x) | (expand_bits(y) << 1u) | (expand_bits(z) << 2u);
}

// Quantizes a single scalar coordinate to a grid cell in [0, resolution),
// mirroring the reference `quantize_axis`: normalize, clamp, scale, floor, clamp
// into the last cell. A degenerate domain or zero resolution collapses to 0.
fn quantize_axis(v: f32, lo: f32, hi: f32, resolution: u32) -> u32 {
    var res = resolution;
    if (res < 1u) {
        res = 1u;
    }
    let extent = hi - lo;
    if (extent <= 0.0) {
        return 0u;
    }
    let res_f = f32(res);
    let t = clamp((v - lo) / extent, 0.0, 1.0);
    let cell_f = floor(t * res_f);
    let cell = u32(cell_f);
    return min(cell, res - 1u);
}

// ceil(log2(n)) via a bounded integer bit-shift loop, mirroring the reference
// `ceil_log2`. `n <= 1` returns 0; the loop runs at most 32 iterations.
fn ceil_log2(n: u32) -> u32 {
    if (n <= 1u) {
        return 0u;
    }
    var v = n - 1u;
    var bits = 0u;
    loop {
        if (v == 0u) {
            break;
        }
        v = v >> 1u;
        bits = bits + 1u;
    }
    return bits;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let op = q.op;

    var out: Result;
    out.op = op;
    out.bool_flag = 0u;
    out.uint0 = 0u;
    out.uint1 = 0u;
    out.uint2 = 0u;
    out.pad_u0 = 0u;
    out.pad_u1 = 0u;
    out.scalar = 0.0;
    out.vec0 = vec3<f32>(0.0, 0.0, 0.0);
    out.pad_v0 = 0.0;
    out.vec1 = vec3<f32>(0.0, 0.0, 0.0);
    out.pad_v1 = 0.0;

    if (op == OP_VEC3_ADD) {
        out.vec0 = q.vec_a + q.vec_b;
    } else if (op == OP_VEC3_SUB) {
        out.vec0 = q.vec_a - q.vec_b;
    } else if (op == OP_VEC3_SCALE) {
        out.vec0 = q.vec_a * q.scalar0;
    } else if (op == OP_VEC3_MIN) {
        out.vec0 = min(q.vec_a, q.vec_b);
    } else if (op == OP_VEC3_MAX) {
        out.vec0 = max(q.vec_a, q.vec_b);
    } else if (op == OP_VEC3_LENGTH_SQUARED) {
        out.scalar = dot(q.vec_a, q.vec_a);
    } else if (op == OP_VEC3_LENGTH) {
        out.scalar = sqrt(dot(q.vec_a, q.vec_a));
    } else if (op == OP_VEC3_COMPONENT) {
        if (q.u0 == 0u) {
            out.scalar = q.vec_a.x;
        } else if (q.u0 == 1u) {
            out.scalar = q.vec_a.y;
        } else {
            out.scalar = q.vec_a.z;
        }
    } else if (op == OP_AABB_UNION) {
        out.vec0 = min(q.vec_a, q.vec_c);
        out.vec1 = max(q.vec_b, q.vec_d);
    } else if (op == OP_AABB_CENTER) {
        out.vec0 = (q.vec_a + q.vec_b) * 0.5;
    } else if (op == OP_AABB_HALF_EXTENT) {
        out.vec0 = (q.vec_b - q.vec_a) * 0.5;
    } else if (op == OP_AABB_SURFACE_AREA) {
        out.scalar = surface_area(q.vec_a, q.vec_b);
    } else if (op == OP_AABB_LONGEST_AXIS) {
        out.uint0 = longest_axis(q.vec_a, q.vec_b);
    } else if (op == OP_AABB_CONTAINS) {
        if (contains(q.vec_a, q.vec_b, q.vec_c)) {
            out.bool_flag = 1u;
        }
    } else if (op == OP_AABB_IS_EMPTY) {
        if (q.vec_a.x > q.vec_b.x || q.vec_a.y > q.vec_b.y || q.vec_a.z > q.vec_b.z) {
            out.bool_flag = 1u;
        }
    } else if (op == OP_AABB_EXPAND_POINT) {
        out.vec0 = min(q.vec_a, q.vec_c);
        out.vec1 = max(q.vec_b, q.vec_c);
    } else if (op == OP_QUANTIZE_TO_GRID) {
        out.uint0 = quantize_axis(q.vec_a.x, q.vec_b.x, q.vec_c.x, q.u0);
        out.uint1 = quantize_axis(q.vec_a.y, q.vec_b.y, q.vec_c.y, q.u0);
        out.uint2 = quantize_axis(q.vec_a.z, q.vec_b.z, q.vec_c.z, q.u0);
    } else if (op == OP_EXPAND_BITS) {
        out.uint0 = expand_bits(q.u0);
    } else if (op == OP_MORTON_CODE_3D) {
        out.uint0 = morton_code_3d(q.u0, q.u1, q.u2);
    } else if (op == OP_INTERNAL_NODE_COUNT) {
        if (q.u0 == 0u) {
            out.uint0 = 0u;
        } else {
            out.uint0 = q.u0 - 1u;
        }
    } else if (op == OP_TOTAL_NODE_COUNT) {
        if (q.u0 == 0u) {
            out.uint0 = 0u;
        } else {
            out.uint0 = 2u * q.u0 - 1u;
        }
    } else if (op == OP_MAX_STACK_DEPTH) {
        out.uint0 = ceil_log2(q.u0) + SAFETY_MARGIN;
    } else {
        // OP_SAH_COST: C_trav + C_isect * (A_left/A_total * n_left +
        // A_right/A_total * n_right). A non-positive A_total collapses to the
        // bare traversal cost so the kernel never divides by zero.
        if (q.scalar2 <= 0.0) {
            out.scalar = SAH_TRAVERSAL_COST;
        } else {
            let inv_total = 1.0 / q.scalar2;
            let left_f = f32(q.u0);
            let right_f = f32(q.u1);
            out.scalar = SAH_TRAVERSAL_COST
                + SAH_INTERSECT_COST
                    * (q.scalar0 * inv_total * left_f + q.scalar1 * inv_total * right_f);
        }
    }

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`BVH_WGSL`].
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
/// Each `vec3` lane carries a trailing pad word so every member stays `16`-byte
/// aligned on device.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Operation tag selecting the twinned function.
    op: u32,
    /// Unsigned operand 0.
    u0: u32,
    /// Unsigned operand 1.
    u1: u32,
    /// Unsigned operand 2.
    u2: u32,
    /// Scalar operand 0 (`scale` factor or `left_area`).
    scalar0: f32,
    /// Scalar operand 1 (`right_area`).
    scalar1: f32,
    /// Scalar operand 2 (`total_area`).
    scalar2: f32,
    /// Padding word.
    pad0: f32,
    /// Vector operand a.
    vec_a: [f32; 3],
    /// Pad lane after `vec_a`.
    pad_a: f32,
    /// Vector operand b.
    vec_b: [f32; 3],
    /// Pad lane after `vec_b`.
    pad_b: f32,
    /// Vector operand c.
    vec_c: [f32; 3],
    /// Pad lane after `vec_c`.
    pad_c: f32,
    /// Vector operand d.
    vec_d: [f32; 3],
    /// Pad lane after `vec_d`.
    pad_d: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Operation tag echoed back for decoding.
    op: u32,
    /// Boolean verdict (`0` / `1`) for `contains` and `is_empty`.
    bool_flag: u32,
    /// Unsigned result 0.
    uint0: u32,
    /// Unsigned result 1.
    uint1: u32,
    /// Unsigned result 2.
    uint2: u32,
    /// Padding word.
    pad_u0: u32,
    /// Padding word.
    pad_u1: u32,
    /// Scalar result.
    scalar: f32,
    /// Vector result 0.
    vec0: [f32; 3],
    /// Pad lane after `vec0`.
    pad_v0: f32,
    /// Vector result 1.
    vec1: [f32; 3],
    /// Pad lane after `vec1`.
    pad_v1: f32,
}

/// One query for the `BVH` twin: a tagged union selecting which of the twinned
/// reference functions this element evaluates.
///
/// The vector variants carry raw `[f32; 3]` lane triples (the reference
/// [`Vec3`](prism_render_architecture::particle::bvh::Vec3) flattened); the box
/// variants carry the box corners (and a point where needed); the `Morton` and
/// tree-sizing variants carry the raw `u32` inputs; the split-cost variant
/// carries the areas and primitive counts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BvhQuery {
    /// Component-wise sum, twinning
    /// [`Vec3::add`](prism_render_architecture::particle::bvh::Vec3::add).
    Vec3Add {
        /// Left operand.
        lhs: [f32; 3],
        /// Right operand.
        rhs: [f32; 3],
    },
    /// Component-wise difference, twinning
    /// [`Vec3::sub`](prism_render_architecture::particle::bvh::Vec3::sub).
    Vec3Sub {
        /// Left operand.
        lhs: [f32; 3],
        /// Right operand.
        rhs: [f32; 3],
    },
    /// Uniform scale, twinning
    /// [`Vec3::scale`](prism_render_architecture::particle::bvh::Vec3::scale).
    Vec3Scale {
        /// Vector to scale.
        vector: [f32; 3],
        /// Uniform scale factor.
        scale: f32,
    },
    /// Component-wise minimum, twinning
    /// [`Vec3::min`](prism_render_architecture::particle::bvh::Vec3::min).
    Vec3Min {
        /// Left operand.
        lhs: [f32; 3],
        /// Right operand.
        rhs: [f32; 3],
    },
    /// Component-wise maximum, twinning
    /// [`Vec3::max`](prism_render_architecture::particle::bvh::Vec3::max).
    Vec3Max {
        /// Left operand.
        lhs: [f32; 3],
        /// Right operand.
        rhs: [f32; 3],
    },
    /// Squared length, twinning
    /// [`Vec3::length_squared`](prism_render_architecture::particle::bvh::Vec3::length_squared).
    Vec3LengthSquared {
        /// Vector whose squared length is taken.
        vector: [f32; 3],
    },
    /// Euclidean length, twinning
    /// [`Vec3::length`](prism_render_architecture::particle::bvh::Vec3::length).
    Vec3Length {
        /// Vector whose length is taken.
        vector: [f32; 3],
    },
    /// Axis component selection, twinning
    /// [`Vec3::component`](prism_render_architecture::particle::bvh::Vec3::component).
    Vec3Component {
        /// Vector to index.
        vector: [f32; 3],
        /// Axis selector (`0` = x, `1` = y, anything else = z).
        axis: u32,
    },
    /// Box union, twinning
    /// [`Aabb::union`](prism_render_architecture::particle::bvh::Aabb::union).
    AabbUnion {
        /// First box minimum corner.
        a_min: [f32; 3],
        /// First box maximum corner.
        a_max: [f32; 3],
        /// Second box minimum corner.
        b_min: [f32; 3],
        /// Second box maximum corner.
        b_max: [f32; 3],
    },
    /// Box center, twinning
    /// [`Aabb::center`](prism_render_architecture::particle::bvh::Aabb::center).
    AabbCenter {
        /// Box minimum corner.
        min: [f32; 3],
        /// Box maximum corner.
        max: [f32; 3],
    },
    /// Box half-extent, twinning
    /// [`Aabb::half_extent`](prism_render_architecture::particle::bvh::Aabb::half_extent).
    AabbHalfExtent {
        /// Box minimum corner.
        min: [f32; 3],
        /// Box maximum corner.
        max: [f32; 3],
    },
    /// Box surface area, twinning
    /// [`Aabb::surface_area`](prism_render_architecture::particle::bvh::Aabb::surface_area).
    AabbSurfaceArea {
        /// Box minimum corner.
        min: [f32; 3],
        /// Box maximum corner.
        max: [f32; 3],
    },
    /// Longest-axis index, twinning
    /// [`Aabb::longest_axis`](prism_render_architecture::particle::bvh::Aabb::longest_axis).
    AabbLongestAxis {
        /// Box minimum corner.
        min: [f32; 3],
        /// Box maximum corner.
        max: [f32; 3],
    },
    /// Point containment test, twinning
    /// [`Aabb::contains`](prism_render_architecture::particle::bvh::Aabb::contains).
    AabbContains {
        /// Box minimum corner.
        min: [f32; 3],
        /// Box maximum corner.
        max: [f32; 3],
        /// Point to test.
        point: [f32; 3],
    },
    /// Emptiness test, twinning
    /// [`Aabb::is_empty`](prism_render_architecture::particle::bvh::Aabb::is_empty).
    AabbIsEmpty {
        /// Box minimum corner.
        min: [f32; 3],
        /// Box maximum corner.
        max: [f32; 3],
    },
    /// Grow-to-point, twinning
    /// [`Aabb::expand_point`](prism_render_architecture::particle::bvh::Aabb::expand_point).
    AabbExpandPoint {
        /// Box minimum corner.
        min: [f32; 3],
        /// Box maximum corner.
        max: [f32; 3],
        /// Point to grow the box toward.
        point: [f32; 3],
    },
    /// Grid quantization, twinning
    /// [`quantize_to_grid`](prism_render_architecture::particle::bvh::quantize_to_grid).
    QuantizeToGrid {
        /// World-space point to quantize.
        point: [f32; 3],
        /// Grid domain minimum corner.
        domain_min: [f32; 3],
        /// Grid domain maximum corner.
        domain_max: [f32; 3],
        /// Grid resolution (cells per axis).
        resolution: u32,
    },
    /// `Morton` lane bit-spread, twinning
    /// [`expand_bits`](prism_render_architecture::particle::bvh::expand_bits).
    ExpandBits {
        /// Value whose low `10` bits are spread.
        value: u32,
    },
    /// `Morton` code interleave, twinning
    /// [`morton_code_3d`](prism_render_architecture::particle::bvh::morton_code_3d).
    MortonCode3d {
        /// X grid coordinate.
        x: u32,
        /// Y grid coordinate.
        y: u32,
        /// Z grid coordinate.
        z: u32,
    },
    /// Internal-node count, twinning
    /// [`BvhTopology::internal_node_count`](prism_render_architecture::particle::bvh::BvhTopology::internal_node_count).
    InternalNodeCount {
        /// Number of leaf nodes.
        leaf_count: u32,
    },
    /// Total-node count, twinning
    /// [`BvhTopology::total_node_count`](prism_render_architecture::particle::bvh::BvhTopology::total_node_count).
    TotalNodeCount {
        /// Number of leaf nodes.
        leaf_count: u32,
    },
    /// Traversal-stack depth, twinning
    /// [`BvhTopology::max_stack_depth`](prism_render_architecture::particle::bvh::BvhTopology::max_stack_depth).
    MaxStackDepth {
        /// Number of leaf nodes.
        leaf_count: u32,
    },
    /// Surface-area-heuristic split cost, twinning
    /// [`sah_cost`](prism_render_architecture::particle::bvh::sah_cost).
    SahCost {
        /// Left-child box surface area.
        left_area: f32,
        /// Right-child box surface area.
        right_area: f32,
        /// Parent box surface area.
        total_area: f32,
        /// Left-child primitive count.
        left_n: u32,
        /// Right-child primitive count.
        right_n: u32,
    },
}

/// One resolved answer for a single [`BvhQuery`], mirroring the value the
/// reference reports for the corresponding function.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BvhResult {
    /// Summed vector, matching
    /// [`Vec3::add`](prism_render_architecture::particle::bvh::Vec3::add).
    Vec3Add {
        /// Summed vector.
        vector: [f32; 3],
    },
    /// Difference vector, matching
    /// [`Vec3::sub`](prism_render_architecture::particle::bvh::Vec3::sub).
    Vec3Sub {
        /// Difference vector.
        vector: [f32; 3],
    },
    /// Scaled vector, matching
    /// [`Vec3::scale`](prism_render_architecture::particle::bvh::Vec3::scale).
    Vec3Scale {
        /// Scaled vector.
        vector: [f32; 3],
    },
    /// Component-wise minimum, matching
    /// [`Vec3::min`](prism_render_architecture::particle::bvh::Vec3::min).
    Vec3Min {
        /// Minimum vector.
        vector: [f32; 3],
    },
    /// Component-wise maximum, matching
    /// [`Vec3::max`](prism_render_architecture::particle::bvh::Vec3::max).
    Vec3Max {
        /// Maximum vector.
        vector: [f32; 3],
    },
    /// Squared length, matching
    /// [`Vec3::length_squared`](prism_render_architecture::particle::bvh::Vec3::length_squared).
    Vec3LengthSquared {
        /// Squared length.
        value: f32,
    },
    /// Euclidean length, matching
    /// [`Vec3::length`](prism_render_architecture::particle::bvh::Vec3::length).
    Vec3Length {
        /// Vector length.
        length: f32,
    },
    /// Selected axis component, matching
    /// [`Vec3::component`](prism_render_architecture::particle::bvh::Vec3::component).
    Vec3Component {
        /// Component value.
        value: f32,
    },
    /// Union box, matching
    /// [`Aabb::union`](prism_render_architecture::particle::bvh::Aabb::union).
    AabbUnion {
        /// Union minimum corner.
        min: [f32; 3],
        /// Union maximum corner.
        max: [f32; 3],
    },
    /// Box center, matching
    /// [`Aabb::center`](prism_render_architecture::particle::bvh::Aabb::center).
    AabbCenter {
        /// Center point.
        center: [f32; 3],
    },
    /// Box half-extent, matching
    /// [`Aabb::half_extent`](prism_render_architecture::particle::bvh::Aabb::half_extent).
    AabbHalfExtent {
        /// Half-extent vector.
        half_extent: [f32; 3],
    },
    /// Box surface area, matching
    /// [`Aabb::surface_area`](prism_render_architecture::particle::bvh::Aabb::surface_area).
    AabbSurfaceArea {
        /// Surface area.
        area: f32,
    },
    /// Longest-axis index, matching
    /// [`Aabb::longest_axis`](prism_render_architecture::particle::bvh::Aabb::longest_axis).
    AabbLongestAxis {
        /// Longest-axis index (`0` = x, `1` = y, `2` = z).
        axis: u32,
    },
    /// Containment verdict, matching
    /// [`Aabb::contains`](prism_render_architecture::particle::bvh::Aabb::contains).
    AabbContains {
        /// Whether the point lies within the closed box.
        contains: bool,
    },
    /// Emptiness verdict, matching
    /// [`Aabb::is_empty`](prism_render_architecture::particle::bvh::Aabb::is_empty).
    AabbIsEmpty {
        /// Whether the box holds no points.
        is_empty: bool,
    },
    /// Grown box, matching
    /// [`Aabb::expand_point`](prism_render_architecture::particle::bvh::Aabb::expand_point).
    AabbExpandPoint {
        /// Grown minimum corner.
        min: [f32; 3],
        /// Grown maximum corner.
        max: [f32; 3],
    },
    /// Quantized grid cell, matching
    /// [`quantize_to_grid`](prism_render_architecture::particle::bvh::quantize_to_grid).
    QuantizeToGrid {
        /// Per-axis grid cell indices.
        cell: [u32; 3],
    },
    /// Spread lane, matching
    /// [`expand_bits`](prism_render_architecture::particle::bvh::expand_bits).
    ExpandBits {
        /// Spread `30`-bit lane value.
        bits: u32,
    },
    /// `Morton` code, matching
    /// [`morton_code_3d`](prism_render_architecture::particle::bvh::morton_code_3d).
    MortonCode3d {
        /// `30`-bit `Morton` code.
        code: u32,
    },
    /// Internal-node count, matching
    /// [`BvhTopology::internal_node_count`](prism_render_architecture::particle::bvh::BvhTopology::internal_node_count).
    InternalNodeCount {
        /// Internal-node count.
        count: u32,
    },
    /// Total-node count, matching
    /// [`BvhTopology::total_node_count`](prism_render_architecture::particle::bvh::BvhTopology::total_node_count).
    TotalNodeCount {
        /// Total-node count.
        count: u32,
    },
    /// Traversal-stack depth, matching
    /// [`BvhTopology::max_stack_depth`](prism_render_architecture::particle::bvh::BvhTopology::max_stack_depth).
    MaxStackDepth {
        /// Traversal-stack depth upper bound.
        depth: u32,
    },
    /// Split cost, matching
    /// [`sah_cost`](prism_render_architecture::particle::bvh::sah_cost).
    SahCost {
        /// Surface-area-heuristic split cost.
        cost: f32,
    },
}

/// Encodes one [`BvhQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &BvhQuery) -> GpuQuery {
    let mut slot = GpuQuery::zeroed();
    match q {
        BvhQuery::Vec3Add { lhs, rhs } => {
            slot.op = OP_VEC3_ADD;
            slot.vec_a = *lhs;
            slot.vec_b = *rhs;
        }
        BvhQuery::Vec3Sub { lhs, rhs } => {
            slot.op = OP_VEC3_SUB;
            slot.vec_a = *lhs;
            slot.vec_b = *rhs;
        }
        BvhQuery::Vec3Scale { vector, scale } => {
            slot.op = OP_VEC3_SCALE;
            slot.vec_a = *vector;
            slot.scalar0 = *scale;
        }
        BvhQuery::Vec3Min { lhs, rhs } => {
            slot.op = OP_VEC3_MIN;
            slot.vec_a = *lhs;
            slot.vec_b = *rhs;
        }
        BvhQuery::Vec3Max { lhs, rhs } => {
            slot.op = OP_VEC3_MAX;
            slot.vec_a = *lhs;
            slot.vec_b = *rhs;
        }
        BvhQuery::Vec3LengthSquared { vector } => {
            slot.op = OP_VEC3_LENGTH_SQUARED;
            slot.vec_a = *vector;
        }
        BvhQuery::Vec3Length { vector } => {
            slot.op = OP_VEC3_LENGTH;
            slot.vec_a = *vector;
        }
        BvhQuery::Vec3Component { vector, axis } => {
            slot.op = OP_VEC3_COMPONENT;
            slot.vec_a = *vector;
            slot.u0 = *axis;
        }
        BvhQuery::AabbUnion {
            a_min,
            a_max,
            b_min,
            b_max,
        } => {
            slot.op = OP_AABB_UNION;
            slot.vec_a = *a_min;
            slot.vec_b = *a_max;
            slot.vec_c = *b_min;
            slot.vec_d = *b_max;
        }
        BvhQuery::AabbCenter { min, max } => {
            slot.op = OP_AABB_CENTER;
            slot.vec_a = *min;
            slot.vec_b = *max;
        }
        BvhQuery::AabbHalfExtent { min, max } => {
            slot.op = OP_AABB_HALF_EXTENT;
            slot.vec_a = *min;
            slot.vec_b = *max;
        }
        BvhQuery::AabbSurfaceArea { min, max } => {
            slot.op = OP_AABB_SURFACE_AREA;
            slot.vec_a = *min;
            slot.vec_b = *max;
        }
        BvhQuery::AabbLongestAxis { min, max } => {
            slot.op = OP_AABB_LONGEST_AXIS;
            slot.vec_a = *min;
            slot.vec_b = *max;
        }
        BvhQuery::AabbContains { min, max, point } => {
            slot.op = OP_AABB_CONTAINS;
            slot.vec_a = *min;
            slot.vec_b = *max;
            slot.vec_c = *point;
        }
        BvhQuery::AabbIsEmpty { min, max } => {
            slot.op = OP_AABB_IS_EMPTY;
            slot.vec_a = *min;
            slot.vec_b = *max;
        }
        BvhQuery::AabbExpandPoint { min, max, point } => {
            slot.op = OP_AABB_EXPAND_POINT;
            slot.vec_a = *min;
            slot.vec_b = *max;
            slot.vec_c = *point;
        }
        BvhQuery::QuantizeToGrid {
            point,
            domain_min,
            domain_max,
            resolution,
        } => {
            slot.op = OP_QUANTIZE_TO_GRID;
            slot.vec_a = *point;
            slot.vec_b = *domain_min;
            slot.vec_c = *domain_max;
            slot.u0 = *resolution;
        }
        BvhQuery::ExpandBits { value } => {
            slot.op = OP_EXPAND_BITS;
            slot.u0 = *value;
        }
        BvhQuery::MortonCode3d { x, y, z } => {
            slot.op = OP_MORTON_CODE_3D;
            slot.u0 = *x;
            slot.u1 = *y;
            slot.u2 = *z;
        }
        BvhQuery::InternalNodeCount { leaf_count } => {
            slot.op = OP_INTERNAL_NODE_COUNT;
            slot.u0 = *leaf_count;
        }
        BvhQuery::TotalNodeCount { leaf_count } => {
            slot.op = OP_TOTAL_NODE_COUNT;
            slot.u0 = *leaf_count;
        }
        BvhQuery::MaxStackDepth { leaf_count } => {
            slot.op = OP_MAX_STACK_DEPTH;
            slot.u0 = *leaf_count;
        }
        BvhQuery::SahCost {
            left_area,
            right_area,
            total_area,
            left_n,
            right_n,
        } => {
            slot.op = OP_SAH_COST;
            slot.scalar0 = *left_area;
            slot.scalar1 = *right_area;
            slot.scalar2 = *total_area;
            slot.u0 = *left_n;
            slot.u1 = *right_n;
        }
    }
    slot
}

/// Decodes one packed [`GpuResult`] into the public [`BvhResult`], selecting the
/// union variant from the echoed operation tag.
fn decode_result(raw: &GpuResult) -> BvhResult {
    match raw.op {
        OP_VEC3_ADD => BvhResult::Vec3Add { vector: raw.vec0 },
        OP_VEC3_SUB => BvhResult::Vec3Sub { vector: raw.vec0 },
        OP_VEC3_SCALE => BvhResult::Vec3Scale { vector: raw.vec0 },
        OP_VEC3_MIN => BvhResult::Vec3Min { vector: raw.vec0 },
        OP_VEC3_MAX => BvhResult::Vec3Max { vector: raw.vec0 },
        OP_VEC3_LENGTH_SQUARED => BvhResult::Vec3LengthSquared { value: raw.scalar },
        OP_VEC3_LENGTH => BvhResult::Vec3Length { length: raw.scalar },
        OP_VEC3_COMPONENT => BvhResult::Vec3Component { value: raw.scalar },
        OP_AABB_UNION => BvhResult::AabbUnion {
            min: raw.vec0,
            max: raw.vec1,
        },
        OP_AABB_CENTER => BvhResult::AabbCenter { center: raw.vec0 },
        OP_AABB_HALF_EXTENT => BvhResult::AabbHalfExtent {
            half_extent: raw.vec0,
        },
        OP_AABB_SURFACE_AREA => BvhResult::AabbSurfaceArea { area: raw.scalar },
        OP_AABB_LONGEST_AXIS => BvhResult::AabbLongestAxis { axis: raw.uint0 },
        OP_AABB_CONTAINS => BvhResult::AabbContains {
            contains: raw.bool_flag != 0,
        },
        OP_AABB_IS_EMPTY => BvhResult::AabbIsEmpty {
            is_empty: raw.bool_flag != 0,
        },
        OP_AABB_EXPAND_POINT => BvhResult::AabbExpandPoint {
            min: raw.vec0,
            max: raw.vec1,
        },
        OP_QUANTIZE_TO_GRID => BvhResult::QuantizeToGrid {
            cell: [raw.uint0, raw.uint1, raw.uint2],
        },
        OP_EXPAND_BITS => BvhResult::ExpandBits { bits: raw.uint0 },
        OP_MORTON_CODE_3D => BvhResult::MortonCode3d { code: raw.uint0 },
        OP_INTERNAL_NODE_COUNT => BvhResult::InternalNodeCount { count: raw.uint0 },
        OP_TOTAL_NODE_COUNT => BvhResult::TotalNodeCount { count: raw.uint0 },
        OP_MAX_STACK_DEPTH => BvhResult::MaxStackDepth { depth: raw.uint0 },
        _ => BvhResult::SahCost { cost: raw.scalar },
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

/// A compiled, reusable `BVH` compute pipeline, twinning the `CPU` golden
/// [`bvh`](prism_render_architecture::particle::bvh).
pub struct GpuBvh {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuBvh {
    /// Compiles the `BVH` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBvh {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_bvh"),
            source: ShaderSource::Wgsl(BVH_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_bvh_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_bvh_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_bvh_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBvh {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries` and returns one [`BvhResult`] per
    /// input, in order.
    ///
    /// The continuous answers match the reference to within the tolerance
    /// documented on this module; the discrete answers match exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[BvhQuery]) -> Vec<BvhResult> {
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
            label: Some("prism_volumetric_bvh_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_bvh_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_bvh_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_bvh_bind_group"),
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
            label: Some("prism_volumetric_bvh_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_bvh_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_bvh_pass"),
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
