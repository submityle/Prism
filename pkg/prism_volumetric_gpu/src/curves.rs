//! `wgpu` compute twin of the authored keyframe-`curve` per-`t` scalar sample
//! ([`curves`](prism_render_architecture::particle::curves), particle design
//! §5, §8, §30).
//!
//! The `CPU` golden
//! [`Curve`](prism_render_architecture::particle::curves::Curve) reconstructs a
//! scalar from an ordered list of
//! [`Keyframe`](prism_render_architecture::particle::curves::Keyframe)s under
//! one of five
//! [`InterpolationMode`](prism_render_architecture::particle::curves::InterpolationMode)
//! families (`Step`, `Linear`, `Hermite`, `CatmullRom`, `Bezier`), evaluated by
//! [`Curve::sample`](prism_render_architecture::particle::curves::Curve::sample):
//! outside the key domain it holds the nearest endpoint value, inside it locates
//! the bounding segment and reconstructs with the segment polynomial. [`GpuCurve`]
//! is the on-device twin for that stateless, per-`t` query: one thread resolves
//! one `(keyframe sub-slice, mode, t)` query, so a passing real-device parity
//! test is direct evidence the ported kernel reproduces the same values the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Only
//! [`Curve::sample`](prism_render_architecture::particle::curves::Curve::sample)
//! is twinned: the empty and single-key short-circuits, the two endpoint-hold
//! clamps, the segment `locate` (a `partition_point` over the keys counting
//! `time <= t`), the local parameter `u`, and the five per-mode reconstructions
//! — `Step`, `Linear`, the cubic `Hermite` basis, the `Catmull-Rom` inferred
//! tangents feeding that same basis, and the cubic `Bezier` Bernstein form. The
//! baking tooling
//! ([`Curve::bake`](prism_render_architecture::particle::curves::Curve::bake),
//! `CurveLut`) and the colour-ramp types are deliberately **not** twinned here.
//!
//! # Mode codes
//!
//! The reference
//! [`InterpolationMode`](prism_render_architecture::particle::curves::InterpolationMode)
//! is carried into the kernel as a `u32`: [`CURVE_STEP`] is `0`, [`CURVE_LINEAR`]
//! is `1`, [`CURVE_HERMITE`] is `2`, [`CURVE_CATMULL_ROM`] is `3` and
//! [`CURVE_BEZIER`] is `4`. These classification codes are integer and compared
//! with `==`.
//!
//! # Correctness model
//!
//! The segment polynomials thread through multiplies, adds and one guarded
//! division, so they are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits. The
//! parity test therefore asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <=
//! 1e-3`, `REL_FLOOR = 1e-6`) on the sampled value, tight enough to catch a
//! dropped term, a swapped key or a wrong basis coefficient yet loose enough to
//! admit legal fused multiply-add contraction.
//!
//! # Degenerate inputs
//!
//! A query with zero keys samples to `0`; a single-key query returns that key's
//! value. A zero-width segment (its two key times within [`EPS`] of each other)
//! collapses the local parameter `u` to `0` instead of dividing by zero, and the
//! `Catmull-Rom` tangent secants fall back to `0` when their spans are within
//! [`EPS`] of zero, matching the reference guards. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `abs`,
//! `+ - * /`, unsigned/boundary index arithmetic and one bounded counting loop
//! over the keys — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse
//! trigonometry, no `sqrt` and no optional device feature, so it runs unmodified
//! on `Metal`, `Vulkan` and `DX12`. The only loop is bounded by the host
//! key-count budget, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::curves`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, PollType, Queue,
    ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::curves`。
const WORKGROUP_SIZE: u32 = 64;

/// Mode code for a `Step` hold, matching
/// [`InterpolationMode::Step`](prism_render_architecture::particle::curves::InterpolationMode::Step).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::curves`。
pub const CURVE_STEP: u32 = 0;

/// Mode code for a `Linear` blend, matching
/// [`InterpolationMode::Linear`](prism_render_architecture::particle::curves::InterpolationMode::Linear).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::curves`。
pub const CURVE_LINEAR: u32 = 1;

/// Mode code for a cubic `Hermite` segment, matching
/// [`InterpolationMode::Hermite`](prism_render_architecture::particle::curves::InterpolationMode::Hermite).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::curves`。
pub const CURVE_HERMITE: u32 = 2;

/// Mode code for a `Catmull-Rom` segment, matching
/// [`InterpolationMode::CatmullRom`](prism_render_architecture::particle::curves::InterpolationMode::CatmullRom).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::curves`。
pub const CURVE_CATMULL_ROM: u32 = 3;

/// Mode code for a cubic `Bezier` segment, matching
/// [`InterpolationMode::Bezier`](prism_render_architecture::particle::curves::InterpolationMode::Bezier).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::curves`。
pub const CURVE_BEZIER: u32 = 4;

/// The portable core-`WGSL` curve-sample kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`Curve::sample`](prism_render_architecture::particle::curves::Curve::sample)
/// branch for branch; see the module documentation for the algorithm.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::curves`。
const CURVES_WGSL: &str = r#"
// Curve-sample twin: one shared read-only keyframe buffer, one thread per query.
// Each thread reproduces the per-t scalar sample. It mirrors the CPU golden
// `particle::curves` Curve::sample branch for branch, uses only the portable
// core-WGSL subset (min/abs and + - * / plus index arithmetic and one bounded
// counting loop), needs no transcendental call and takes no optional feature,
// so it runs unmodified on Metal, Vulkan and DX12. The only loop is bounded by
// the host key-count budget, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::curves；无第三方
// 引擎源码或衍生代码。

// Segment-width / tangent-span guard below which a denominator is treated as
// zero so evaluation never divides by (near) zero. Matches the reference `EPS`.
const EPS: f32 = 1.0e-9;

// Mode codes mirroring the reference `InterpolationMode`.
const MODE_STEP: u32 = 0u;
const MODE_LINEAR: u32 = 1u;
const MODE_HERMITE: u32 = 2u;
const MODE_CATMULL: u32 = 3u;
const MODE_BEZIER: u32 = 4u;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Keyframe {
    // Position along the curve's domain.
    time: f32,
    // Value exactly at `time`.
    value: f32,
    // Incoming tangent (slope for Hermite, control value for Bezier).
    in_tangent: f32,
    // Outgoing tangent (slope for Hermite, control value for Bezier).
    out_tangent: f32,
}

struct Query {
    // First keyframe lane index of this query's sub-slice.
    keys_offset: u32,
    // Number of keyframes in this query's sub-slice.
    keys_len: u32,
    // Reconstruction mode code (MODE_STEP .. MODE_BEZIER).
    mode: u32,
    // Sample parameter.
    t: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> keys: array<Keyframe>;
@group(0) @binding(2) var<storage, read> queries: array<Query>;
@group(0) @binding(3) var<storage, read_write> results: array<f32>;

// Saturating u32 subtraction: a - b, floored at zero (u32 wraps otherwise).
fn sat_sub(a: u32, b: u32) -> u32 {
    if (a > b) {
        return a - b;
    }
    return 0u;
}

// Evaluates a cubic Hermite segment [i, i+1] at local parameter u using the
// explicit endpoint slopes m0 (left) and m1 (right); the standard Hermite basis
// with the tangents scaled by the segment width, mirroring `eval_hermite`.
fn eval_hermite(base: u32, i: u32, u: f32, m0: f32, m1: f32) -> f32 {
    let k0 = keys[base + i];
    let k1 = keys[base + i + 1u];
    let dt = k1.time - k0.time;
    let u2 = u * u;
    let u3 = u2 * u;
    let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
    let h10 = u3 - 2.0 * u2 + u;
    let h01 = -2.0 * u3 + 3.0 * u2;
    let h11 = u3 - u2;
    return h00 * k0.value + h10 * dt * m0 + h01 * k1.value + h11 * dt * m1;
}

// Infers the Catmull-Rom endpoint slopes for segment [i, i+1] from the
// neighbouring keys, clamping at the curve ends (bounded one-sided secant),
// mirroring `catmull_tangents`. Returns (m0, m1) packed in a vec2.
fn catmull_tangents(base: u32, i: u32, n: u32) -> vec2<f32> {
    let k_prev = keys[base + sat_sub(i, 1u)];
    let k0 = keys[base + i];
    let k1 = keys[base + i + 1u];
    let k_next = keys[base + min(i + 2u, n - 1u)];
    var m0: f32 = 0.0;
    let span0 = k1.time - k_prev.time;
    if (abs(span0) > EPS) {
        m0 = (k1.value - k_prev.value) / span0;
    }
    var m1: f32 = 0.0;
    let span1 = k_next.time - k0.time;
    if (abs(span1) > EPS) {
        m1 = (k_next.value - k0.value) / span1;
    }
    return vec2<f32>(m0, m1);
}

// Evaluates a cubic Bezier segment [i, i+1] at local parameter u, the two
// intermediate control values taken from the bounding keys' tangent fields
// (Bezier control-point form), mirroring `eval_bezier`.
fn eval_bezier(base: u32, i: u32, u: f32) -> f32 {
    let k0 = keys[base + i];
    let k1 = keys[base + i + 1u];
    let p0 = k0.value;
    let p1 = k0.out_tangent;
    let p2 = k1.in_tangent;
    let p3 = k1.value;
    let inv = 1.0 - u;
    let inv2 = inv * inv;
    let inv3 = inv2 * inv;
    let u2 = u * u;
    let u3 = u2 * u;
    return inv3 * p0 + 3.0 * inv2 * u * p1 + 3.0 * inv * u2 * p2 + u3 * p3;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let base = q.keys_offset;
    let n = q.keys_len;
    let mode = q.mode;
    let t = q.t;

    var value: f32 = 0.0;
    if (n == 0u) {
        value = 0.0;
    } else if (n == 1u) {
        value = keys[base].value;
    } else {
        let first = keys[base];
        let last = keys[base + n - 1u];
        if (t <= first.time) {
            value = first.value;
        } else if (t >= last.time) {
            value = last.value;
        } else {
            // locate: upper = partition_point(time <= t) over the sub-slice.
            var upper: u32 = 0u;
            for (var j: u32 = 0u; j < n; j = j + 1u) {
                if (keys[base + j].time <= t) {
                    upper = upper + 1u;
                }
            }
            let i = min(sat_sub(upper, 1u), n - 2u);
            let k0 = keys[base + i];
            let k1 = keys[base + i + 1u];
            let dt = k1.time - k0.time;
            var u: f32 = 0.0;
            if (abs(dt) > EPS) {
                u = (t - k0.time) / dt;
            }
            if (mode == MODE_STEP) {
                value = k0.value;
            } else if (mode == MODE_LINEAR) {
                value = k0.value + (k1.value - k0.value) * u;
            } else if (mode == MODE_HERMITE) {
                value = eval_hermite(base, i, u, k0.out_tangent, k1.in_tangent);
            } else if (mode == MODE_CATMULL) {
                let m = catmull_tangents(base, i, n);
                value = eval_hermite(base, i, u, m.x, m.y);
            } else {
                value = eval_bezier(base, i, u);
            }
        }
    }
    results[idx] = value;
}
"#;

/// Uniform parameters for one dispatch: the query count padded to the `std140`
/// `16`-byte alignment matching `Params` in [`CURVES_WGSL`].
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

/// A single authored keyframe uploaded to the shared buffer: `time`, `value`
/// and the two tangent fields, matching the reference
/// [`Keyframe`](prism_render_architecture::particle::curves::Keyframe) and the
/// `std430` `WGSL` `Keyframe` struct (four `f32`, `16` bytes).
///
/// The `in_tangent` / `out_tangent` fields are interpreted per mode: slopes for
/// `Hermite`, intermediate control values for `Bezier`, ignored otherwise.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::curves`。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuKeyframe {
    /// Position along the curve's domain.
    pub time: f32,
    /// Value the curve takes exactly at `time`.
    pub value: f32,
    /// Incoming tangent, used by the segment ending at this key.
    pub in_tangent: f32,
    /// Outgoing tangent, used by the segment starting at this key.
    pub out_tangent: f32,
}

impl GpuKeyframe {
    /// Builds a keyframe from its `time`, `value` and the two tangent fields.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::curves`。
    #[must_use]
    pub const fn new(time: f32, value: f32, in_tangent: f32, out_tangent: f32) -> GpuKeyframe {
        GpuKeyframe {
            time,
            value,
            in_tangent,
            out_tangent,
        }
    }
}

/// One query for the curve twin against the shared uploaded keyframes: a
/// sub-slice of keyframes (by lane offset and length), the reconstruction mode
/// and the sample parameter `t`. The `repr(C)` layout matches the `std430`
/// `WGSL` `Query` struct (four words, `16` bytes), so it uploads directly.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::curves`。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct CurveSampleQuery {
    /// First keyframe lane index of this query's sub-slice in the shared buffer.
    pub keys_offset: u32,
    /// Number of keyframes in this query's sub-slice.
    pub keys_len: u32,
    /// Mode code ([`CURVE_STEP`], [`CURVE_LINEAR`], [`CURVE_HERMITE`],
    /// [`CURVE_CATMULL_ROM`] or [`CURVE_BEZIER`]), matching the reference
    /// [`InterpolationMode`](prism_render_architecture::particle::curves::InterpolationMode).
    pub mode: u32,
    /// Sample parameter fed to
    /// [`Curve::sample`](prism_render_architecture::particle::curves::Curve::sample).
    pub t: f32,
}

impl CurveSampleQuery {
    /// Builds a query from the sub-slice offset and length, the mode code and
    /// the sample parameter `t`.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::curves`。
    #[must_use]
    pub const fn new(keys_offset: u32, keys_len: u32, mode: u32, t: f32) -> CurveSampleQuery {
        CurveSampleQuery {
            keys_offset,
            keys_len,
            mode,
            t,
        }
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

/// A compiled, reusable curve-sample compute pipeline, twinning the per-`t`
/// scalar sample of the `CPU` golden
/// [`Curve`](prism_render_architecture::particle::curves::Curve).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::curves`。
pub struct GpuCurve {
    /// Logical device, cloned from the acquiring [`GpuContext`] so a dispatch
    /// needs no borrowed context.
    device: Device,
    /// Submission queue, cloned from the acquiring [`GpuContext`].
    queue: Queue,
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCurve {
    /// Compiles the curve-sample kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::curves`。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCurve {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_curves"),
            source: ShaderSource::Wgsl(CURVES_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_curves_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_curves_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_curves_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCurve {
            device: device.clone(),
            queue: ctx.queue().clone(),
            module,
            layout,
            pipeline,
        }
    }

    /// Samples every query in `queries` against one shared keyframe buffer and
    /// returns one `f32` value per input, in order.
    ///
    /// `keys` is the shared, flattened list of [`GpuKeyframe`]s every query
    /// indexes through its `keys_offset` / `keys_len`. Each returned value
    /// matches the reference
    /// [`Curve::sample`](prism_render_architecture::particle::curves::Curve::sample)
    /// to within the tolerance documented on this module. An empty `queries`
    /// batch returns an empty vector with no dispatch issued, since a storage
    /// buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::curves`。
    #[must_use]
    pub fn sample(&self, queries: &[CurveSampleQuery], keys: &[GpuKeyframe]) -> Vec<f32> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = &self.device;

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_curves_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        // A storage buffer cannot be zero-sized, so an empty keyframe list
        // uploads a single unused zero key; no query with keys_len > 0 reaches
        // it, and keys_len == 0 queries never index the buffer.
        let mut key_lanes: Vec<GpuKeyframe> = keys.to_vec();
        if key_lanes.is_empty() {
            key_lanes.push(GpuKeyframe::new(0.0, 0.0, 0.0, 0.0));
        }
        let keys_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_curves_keys"),
            contents: bytemuck::cast_slice(&key_lanes),
            usage: BufferUsages::STORAGE,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_curves_queries"),
            contents: bytemuck::cast_slice(queries),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<f32>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_curves_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_curves_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: keys_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_curves_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_curves_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_curves_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
        self.queue.submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        self.device
            .poll(PollType::wait_indefinitely())
            .expect("device poll should complete the submitted work");
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        stage.unmap();

        out
    }
}
