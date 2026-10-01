//! `wgpu` compute twin of the 2D `Hilbert` space-filling-curve transcoder
//! ([`hilbert_curve`](prism_render_architecture::particle::hilbert_curve),
//! design §12 sort-key quantization, §7 `PerNeighborCell`).
//!
//! A `Hilbert` index linearizes a `2^order × 2^order` grid so spatially close
//! cells land close together along a one-dimensional ordering, with the extra
//! property that two consecutive indices are always axis neighbors one cell
//! apart. The `CPU` golden
//! [`hilbert_curve`](prism_render_architecture::particle::hilbert_curve) owns
//! the closed form; [`GpuHilbertCurve`] is the on-device twin that runs one
//! thread per element and reproduces the same `u32`-domain rotate/reflect
//! iteration element for element. A passing real-device parity test is
//! therefore direct evidence the ported kernels fold the quadrant bits exactly
//! as the reference does, not merely that the shaders compile.
//!
//! # What is twinned
//!
//! Only the 2D `u32` variants are ported, since `WGSL` has no `u64`. The 3D
//! `Skilling` transpose variants of the reference
//! ([`hilbert_distance_3d`](prism_render_architecture::particle::hilbert_curve::hilbert_distance_3d),
//! [`hilbert_to_xyz_3d`](prism_render_architecture::particle::hilbert_curve::hilbert_to_xyz_3d))
//! and the `u64` interleave helpers they lean on are deliberately skipped. The
//! three twinned transforms are:
//!
//! * **Forward** ([`xy_to_hilbert`](GpuHilbertCurve::xy_to_hilbert)): the
//!   classic Wikipedia `rot` iteration that walks the quadrant bits from most
//!   to least significant, accumulating `d += s*s * ((3*rx) ^ ry)` and
//!   reflecting the frame at each level, matching
//!   [`xy_to_hilbert`](prism_render_architecture::particle::hilbert_curve::xy_to_hilbert).
//! * **Inverse** ([`hilbert_to_xy`](GpuHilbertCurve::hilbert_to_xy)): the exact
//!   inverse walk that rebuilds `(x, y)` from the index two bits at a time,
//!   matching
//!   [`hilbert_to_xy`](prism_render_architecture::particle::hilbert_curve::hilbert_to_xy).
//! * **Key** ([`hilbert_key`](GpuHilbertCurve::hilbert_key)): the sort-pipeline
//!   alias of the forward map, matching
//!   [`hilbert_key`](prism_render_architecture::particle::hilbert_curve::hilbert_key).
//!
//! Both maps clamp `order` to [`MAX_ORDER`] and fold out-of-range coordinates
//! and indices back into their valid window, exactly as the reference does.
//!
//! # Portability
//!
//! The kernels use only the portable core-`WGSL` subset — the bit operators
//! `<< >> & | ^`, the arithmetic `+ - * /`, `min`/`max` and `select` — with no
//! transcendental call, no `smoothstep`, no optional device feature and no
//! `u64`. The two quadrant loops run under a constant `MAX_ORDER` iteration
//! bound so they unroll statically on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Every transform here is pure integer bit arithmetic, so `CPU` and `GPU`
//! compute identical bit patterns with no rounding anywhere on the path. The
//! parity test therefore asserts an exact `==` on every element (the forward
//! `d`, the inverse `(x, y)` and the key), with no tolerance: any mismatch is a
//! genuine port bug (a wrong shift count, a dropped reflect, a miscounted
//! quadrant).
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::hilbert_curve`；无第三方引擎源码或衍生代码。仅移植 2D u32 变体；3D u64 变体不移植。

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
/// that divides evenly on every target backend.
const WORKGROUP_SIZE: u32 = 64;

/// Largest curve order the kernels accept, mirroring the reference
/// [`MAX_ORDER`](prism_render_architecture::particle::hilbert_curve::MAX_ORDER):
/// at `order = 16` a 2D index spans the full `2^32` range of a [`u32`]. Both
/// `WGSL` quadrant loops run under this constant iteration bound so they unroll
/// statically.
pub const MAX_ORDER: u32 = 16;

/// The 2D `u32`-domain `Hilbert` kernels, mirroring the `CPU` golden
/// [`hilbert_curve`](prism_render_architecture::particle::hilbert_curve) step
/// for step. A single source file hosts the forward and inverse entry points
/// sharing one bind-group layout.
const HILBERT_CURVE_WGSL: &str = r#"
// Hilbert-curve transcoder twin: one thread per element. Two entry points
// mirror the CPU golden `particle::hilbert_curve` 2D u32 domain: the forward
// rotate/reflect map (xy -> d) and its exact inverse (d -> xy). Pure integer
// bit arithmetic only (<< >> & | ^, + - * /, min/max/select), no transcendental,
// no smoothstep and no u64, so the kernels run unmodified on Metal, Vulkan and
// DX12. WGSL unsigned arithmetic wraps modulo 2^32, which matches the reference
// u32 arithmetic (a full-range order-16 index reaches 2^32 - 1 without overflow).
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::hilbert_curve；无第三方
// 引擎源码或衍生代码。仅移植 2D u32 变体；3D u64 变体不移植。

struct Params {
    // Number of valid elements; threads past this short-circuit.
    count: u32,
    // Curve order, clamped to MAX_ORDER (16) inside the kernels.
    order: u32,
    // Padding to a 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
}

// Largest order the maps accept; both loops iterate at most this many times.
const MAX_ORDER: u32 = 16u;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> in_a: array<u32>;
@group(0) @binding(2) var<storage, read> in_b: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_a: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_b: array<u32>;

// Forward map: walk the quadrant bits from most to least significant, folding
// each (rx, ry) pair into the running index and reflecting the frame. Mirrors
// `xy_to_hilbert`: order is clamped, x/y are folded into [0, 2^order).
fn xy_to_hilbert(order_in: u32, x_in: u32, y_in: u32) -> u32 {
    let order = min(order_in, MAX_ORDER);
    let n = 1u << order;
    let mask = n - 1u;
    var x = x_in & mask;
    var y = y_in & mask;
    var d = 0u;
    var s = n >> 1u;
    for (var i = 0u; i < MAX_ORDER; i = i + 1u) {
        if (s == 0u) {
            break;
        }
        let rx = select(0u, 1u, (x & s) > 0u);
        let ry = select(0u, 1u, (y & s) > 0u);
        d = d + s * s * ((3u * rx) ^ ry);
        // rotate_quadrant with frame side n.
        if (ry == 0u) {
            if (rx == 1u) {
                x = n - 1u - x;
                y = n - 1u - y;
            }
            let t = x;
            x = y;
            y = t;
        }
        s = s >> 1u;
    }
    return d;
}

// Inverse map: rebuild (x, y) two index bits at a time, reflecting the current
// sub-square frame. Mirrors `hilbert_to_xy`: order is clamped and d is folded
// into [0, 4^order). total_bits == 32 is handled without a 1u << 32u shift.
fn hilbert_to_xy(order_in: u32, d_in: u32) -> vec2<u32> {
    let order = min(order_in, MAX_ORDER);
    let n = 1u << order;
    let total_bits = order * 2u;
    var t = d_in;
    if (total_bits < 32u) {
        t = d_in & ((1u << total_bits) - 1u);
    }
    var x = 0u;
    var y = 0u;
    var s = 1u;
    for (var i = 0u; i < MAX_ORDER; i = i + 1u) {
        if (s >= n) {
            break;
        }
        let rx = 1u & (t >> 1u);
        let ry = 1u & (t ^ rx);
        // rotate_quadrant with current sub-square side s.
        if (ry == 0u) {
            if (rx == 1u) {
                x = s - 1u - x;
                y = s - 1u - y;
            }
            let tmp = x;
            x = y;
            y = tmp;
        }
        x = x + s * rx;
        y = y + s * ry;
        t = t >> 2u;
        s = s << 1u;
    }
    return vec2<u32>(x, y);
}

@compute @workgroup_size(64)
fn forward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    // in_a holds x, in_b holds y; out_a receives the index. out_b is unused.
    out_a[idx] = xy_to_hilbert(params.order, in_a[idx], in_b[idx]);
}

@compute @workgroup_size(64)
fn inverse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    // in_a holds the index; out_a receives x and out_b receives y.
    let xy = hilbert_to_xy(params.order, in_a[idx]);
    out_a[idx] = xy.x;
    out_b[idx] = xy.y;
}
"#;

/// Uniform parameters for one dispatch: the element `count` and the curve
/// `order`, plus padding to a `16`-byte, `std140`-aligned uniform struct
/// matching `Params` in [`HILBERT_CURVE_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid elements in the input and output buffers.
    count: u32,
    /// Curve order, clamped to [`MAX_ORDER`] inside the kernels.
    order: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// A compiled, reusable pair of 2D `Hilbert` `u32` kernels (forward and
/// inverse).
pub struct GpuHilbertCurve {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline_forward: ComputePipeline,
    pipeline_inverse: ComputePipeline,
}

impl GpuHilbertCurve {
    /// Compiles the forward and inverse `Hilbert` kernels on `ctx`.
    ///
    /// The kernels use only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHilbertCurve {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_hilbert_curve"),
            source: ShaderSource::Wgsl(HILBERT_CURVE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_hilbert_curve_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_hilbert_curve_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let make = |entry: &str, label: &str| {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let pipeline_forward = make("forward", "prism_volumetric_hilbert_curve_forward_pipeline");
        let pipeline_inverse = make("inverse", "prism_volumetric_hilbert_curve_inverse_pipeline");
        GpuHilbertCurve {
            module,
            layout,
            pipeline_forward,
            pipeline_inverse,
        }
    }

    /// Maps each `(x, y)` grid coordinate to its 2D `Hilbert` index for the
    /// given `order`, mirroring
    /// [`xy_to_hilbert`](prism_render_architecture::particle::hilbert_curve::xy_to_hilbert).
    ///
    /// `order` is clamped to [`MAX_ORDER`] and coordinates are folded into
    /// `[0, 2^order)` inside the kernel. Returns one index per input pair, in
    /// order. An empty input returns an empty vector with no dispatch issued (a
    /// storage buffer cannot be zero-sized).
    ///
    /// # Panics
    ///
    /// Panics if `xs` and `ys` differ in length.
    #[must_use]
    pub fn xy_to_hilbert(&self, ctx: &GpuContext, order: u32, xs: &[u32], ys: &[u32]) -> Vec<u32> {
        assert_eq!(xs.len(), ys.len(), "x and y inputs must be the same length");
        let (out_a, _out_b) = self.dispatch(ctx, &self.pipeline_forward, order, xs, ys);
        out_a
    }

    /// Maps each 2D `Hilbert` index back to its `(x, y)` grid coordinate, the
    /// exact inverse of [`xy_to_hilbert`](GpuHilbertCurve::xy_to_hilbert) and a
    /// mirror of
    /// [`hilbert_to_xy`](prism_render_architecture::particle::hilbert_curve::hilbert_to_xy).
    ///
    /// `order` is clamped to [`MAX_ORDER`] and each index is folded into
    /// `[0, 4^order)` inside the kernel. Returns one `(x, y)` pair per input, in
    /// order. An empty input returns an empty vector with no dispatch issued.
    #[must_use]
    pub fn hilbert_to_xy(&self, ctx: &GpuContext, order: u32, ds: &[u32]) -> Vec<(u32, u32)> {
        let (xs, ys) = self.dispatch(ctx, &self.pipeline_inverse, order, ds, ds);
        xs.into_iter().zip(ys).collect()
    }

    /// Returns the `Hilbert` sort key for each 2D cell, the sort-pipeline alias
    /// of [`xy_to_hilbert`](GpuHilbertCurve::xy_to_hilbert) mirroring
    /// [`hilbert_key`](prism_render_architecture::particle::hilbert_curve::hilbert_key).
    ///
    /// Returns one key per input pair, in order. An empty input returns an empty
    /// vector with no dispatch issued.
    ///
    /// # Panics
    ///
    /// Panics if `xs` and `ys` differ in length.
    #[must_use]
    pub fn hilbert_key(&self, ctx: &GpuContext, order: u32, xs: &[u32], ys: &[u32]) -> Vec<u32> {
        self.xy_to_hilbert(ctx, order, xs, ys)
    }

    /// Issues one `1-D` dispatch of `pipeline` over `a` (and `b`; the inverse
    /// kernel binds the index into both slots), reading both `u32` output
    /// buffers back. Empty inputs short-circuit without a dispatch because a
    /// storage buffer cannot be zero-sized.
    fn dispatch(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        order: u32,
        a: &[u32],
        b: &[u32],
    ) -> (Vec<u32>, Vec<u32>) {
        let count = a.len();
        if count == 0 {
            return (Vec::new(), Vec::new());
        }
        let device = ctx.device();

        let params = GpuParams {
            count: count as u32,
            order,
            pad0: 0,
            pad1: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_hilbert_curve_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let in_a = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_hilbert_curve_in_a"),
            contents: bytemuck::cast_slice(a),
            usage: BufferUsages::STORAGE,
        });
        let in_b = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_hilbert_curve_in_b"),
            contents: bytemuck::cast_slice(b),
            usage: BufferUsages::STORAGE,
        });
        let out_bytes = size_of_val(a) as u64;
        let make_out = |label: &str| {
            device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: out_bytes,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let out_a = make_out("prism_volumetric_hilbert_curve_out_a");
        let out_b = make_out("prism_volumetric_hilbert_curve_out_b");
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_hilbert_curve_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: in_a.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: in_b.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_a.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: out_b.as_entire_binding(),
                },
            ],
        });
        let make_stage = |label: &str| {
            device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: out_bytes,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let stage_a = make_stage("prism_volumetric_hilbert_curve_stage_a");
        let stage_b = make_stage("prism_volumetric_hilbert_curve_stage_b");

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_hilbert_curve_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_hilbert_curve_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per element, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_a, 0, &stage_a, 0, out_bytes);
        encoder.copy_buffer_to_buffer(&out_b, 0, &stage_b, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage_a.slice(..).map_async(MapMode::Read, |_| {});
        stage_b.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let result_a = read_mapped(&stage_a);
        let result_b = read_mapped(&stage_b);
        (result_a, result_b)
    }
}

/// Reads a mapped staging buffer back into a `u32` vector and unmaps it.
fn read_mapped(stage: &wgpu::Buffer) -> Vec<u32> {
    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let result = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
    drop(view);
    stage.unmap();
    result
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
