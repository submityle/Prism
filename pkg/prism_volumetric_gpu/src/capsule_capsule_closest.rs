//! `wgpu` compute twin of the capsule-vs-capsule closest-distance geometry
//! contract
//! ([`capsule_capsule_closest`](prism_render_architecture::particle::capsule_capsule_closest),
//! particle design §10, §14).
//!
//! The `CPU` golden
//! [`capsule_capsule_closest`](prism_render_architecture::particle::capsule_capsule_closest)
//! answers the pairwise collision-proximity question a broad phase and the
//! response solver need: given two swept-sphere capsules (a core-axis segment
//! inflated by a radius), it reports the signed surface gap, the non-negative
//! penetration depth, the unit contact normal from capsule A toward capsule B,
//! and the two surface closest points. Its core is Christer Ericson's
//! segment-vs-segment closed form
//! ([`segment_segment_closest`](prism_render_architecture::particle::capsule_capsule_closest::segment_segment_closest),
//! *Real-Time Collision Detection* §5.1.9): the clamped parameters `(s, t)` of
//! the mutually closest axis points, lifted into capsule space by subtracting
//! the radius sum and projecting onto each surface.
//!
//! [`GpuCapsuleCapsuleClosest`] is the on-device twin: one thread per capsule
//! pair reproduces the same closed form branch for branch, so a passing
//! real-device parity test is direct evidence the ported kernel solves the same
//! geometry and classifies the same degenerate cases (parallel axes, collapsed
//! segments, endpoint caps, coincident axes) the reference does, not merely that
//! the shader compiles.
//!
//! # What is twinned
//!
//! Every per-pair answer the reference computes is reproduced: the signed
//! `distance`, the `penetration` depth, the discrete `intersecting` flag, the
//! unit `normal` (with the reference's stable `[0, 1, 0]` fallback when the axes
//! are effectively coincident), and the two surface points `point_a` /
//! `point_b`. The segment-segment core's four regimes are mirrored branch for
//! branch: both segments degenerate (two points), the first degenerate (a point
//! projected onto the second), the second degenerate (a point projected onto
//! the first), and the general / parallel solve where a determinant at or below
//! the compare epsilon pins `s` before the `t` recovery and its re-clamp place
//! both parameters inside `[0, 1]²`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `abs`, `select`, `+ - * /`, the `dot`/`length` builtins and `sqrt`
//! (the capsule query reports a real length, not a squared distance) — with no
//! `sin`, `cos`, `exp`, `log`, `pow`, `tan` and no optional device feature, so
//! it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Each pair is a fixed, non-reorderable sequence of multiplies, adds, divides
//! and one `sqrt`, so `CPU` and `GPU` evaluate the same closed form in the same
//! order. They are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits by a few units in
//! the last place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32` fields and an exact
//! match on the discrete `intersecting` flag, tight enough to catch a genuinely
//! wrong port (a dropped branch, a swapped coefficient, a wrong clamp) yet loose
//! enough to admit legal fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`capsule_capsule_closest`](prism_render_architecture::particle::capsule_capsule_closest);
//! no third-party engine source or derived code.

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

/// The portable core-`WGSL` capsule-vs-capsule closest kernel, embedded inline
/// so the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`capsule_capsule_closest`](prism_render_architecture::particle::capsule_capsule_closest)
/// branch for branch; see the module documentation for the algorithm.
const CAPSULE_CAPSULE_CLOSEST_WGSL: &str = r#"
// Capsule-vs-capsule closest twin: one thread per capsule pair reproduces the
// signed surface gap, the penetration depth, the discrete intersecting flag,
// the unit contact normal and the two surface closest points. It mirrors the
// CPU golden particle::capsule_capsule_closest branch for branch, uses only the
// portable core-WGSL subset (min/max/clamp/abs/select and + - * / plus the
// dot/length builtins and sqrt) and takes no optional feature, so it runs
// unmodified on Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's
// particle::capsule_capsule_closest; no third-party engine source or derived
// code.

// Epsilon that guards every division and every degeneracy test so the kernel
// never writes an exact == / != on an f32 and never emits a NaN. Matches the
// reference `EPS`.
const EPS: f32 = 1.0e-6;

struct Params {
    // Number of capsule pairs in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Capsule A core-axis start endpoint; the fourth lane carries radius ra.
    a0: vec3<f32>,
    ra: f32,
    // Capsule A core-axis end endpoint; a pad lane follows.
    a1: vec3<f32>,
    pad0: f32,
    // Capsule B core-axis start endpoint; the fourth lane carries radius rb.
    b0: vec3<f32>,
    rb: f32,
    // Capsule B core-axis end endpoint; a pad lane follows.
    b1: vec3<f32>,
    pad1: f32,
}

struct Result {
    // Signed surface gap, non-negative penetration depth, the discrete
    // intersecting flag (0 or 1) and one pad lane: four scalars in one slot.
    gap: f32,
    penetration: f32,
    intersecting: u32,
    pad0: u32,
    // Unit contact normal from A to B; a pad lane follows.
    normal: vec3<f32>,
    pad1: f32,
    // Closest point on capsule A's surface; a pad lane follows.
    point_a: vec3<f32>,
    pad2: f32,
    // Closest point on capsule B's surface; a pad lane follows.
    point_b: vec3<f32>,
    pad3: f32,
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
    let a0 = q.a0;
    let a1 = q.a1;
    let b0 = q.b0;
    let b1 = q.b1;
    let ra = q.ra;
    let rb = q.rb;

    // segment_segment_closest(a0, a1, b0, b1), Ericson 5.1.9.
    let d1 = a1 - a0;
    let d2 = b1 - b0;
    let r = a0 - b0;
    let a = dot(d1, d1);
    let e = dot(d2, d2);
    let f = dot(d2, r);

    let first_degenerate = a <= EPS;
    let second_degenerate = e <= EPS;

    var s: f32 = 0.0;
    var t: f32 = 0.0;
    if (first_degenerate && second_degenerate) {
        // Both segments collapse to points: nothing to project.
        s = 0.0;
        t = 0.0;
    } else if (first_degenerate) {
        // Segment A is a point; clamp its projection onto segment B.
        s = 0.0;
        t = clamp(f / e, 0.0, 1.0);
    } else {
        let c = dot(d1, r);
        if (second_degenerate) {
            // Segment B is a point; clamp its projection onto segment A.
            t = 0.0;
            s = clamp(-c / a, 0.0, 1.0);
        } else {
            // The fully general non-degenerate case.
            let b = dot(d1, d2);
            let denom = a * e - b * b;

            // Not near zero => axes are not parallel: solve for the line-line
            // optimum along segment A; otherwise pin s to the start.
            var s_line: f32 = 0.0;
            if (denom > EPS) {
                s_line = clamp((b * f - c * e) / denom, 0.0, 1.0);
            } else {
                s_line = 0.0;
            }

            // Recover t for this s: t = (b*s + f) / e.
            let t_line = (b * s_line + f) / e;

            // If t fell outside [0, 1], clamp it and re-derive s for the clamped
            // t, clamping back to [0, 1].
            if (t_line < 0.0) {
                t = 0.0;
                s = clamp(-c / a, 0.0, 1.0);
            } else if (t_line > 1.0) {
                t = 1.0;
                s = clamp((b - c) / a, 0.0, 1.0);
            } else {
                s = s_line;
                t = t_line;
            }
        }
    }

    let pa = a0 + d1 * s;
    let pb = b0 + d2 * t;

    // Lift the axis closest points into capsule space.
    let delta = pb - pa; // from A's axis point toward B's axis point
    let axis_dist = length(delta);
    let radius_sum = ra + rb;

    let gap = axis_dist - radius_sum;
    let penetration = max(radius_sum - axis_dist, 0.0);
    let intersecting = select(0u, 1u, axis_dist <= radius_sum + EPS);

    // Normalize the connector; fall back to a stable default when the axes are
    // effectively coincident (axis_dist at or below EPS) to avoid a NaN.
    var normal = vec3<f32>(0.0, 1.0, 0.0);
    if (axis_dist > EPS) {
        normal = delta * (1.0 / axis_dist);
    }

    let point_a = pa + normal * ra;
    let point_b = pb - normal * rb;

    var out: Result;
    out.gap = gap;
    out.penetration = penetration;
    out.intersecting = intersecting;
    out.pad0 = 0u;
    out.normal = normal;
    out.pad1 = 0.0;
    out.point_a = point_a;
    out.pad2 = 0.0;
    out.point_b = point_b;
    out.pad3 = 0.0;
    results[idx] = out;
}
"#;

/// One capsule-vs-capsule closest query: capsule A is the core-axis segment
/// `(a0, a1)` inflated by radius `ra`, capsule B is `(b0, b1)` inflated by
/// radius `rb` — the same inputs the reference
/// [`capsule_capsule_closest`](prism_render_architecture::particle::capsule_capsule_closest)
/// consumes.
///
/// Provenance: twinned from this repository's
/// [`capsule_capsule_closest`](prism_render_architecture::particle::capsule_capsule_closest);
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CapsuleClosestQuery {
    /// Capsule A core-axis start endpoint.
    pub a0: [f32; 3],
    /// Capsule A core-axis end endpoint.
    pub a1: [f32; 3],
    /// Capsule A radius.
    pub ra: f32,
    /// Capsule B core-axis start endpoint.
    pub b0: [f32; 3],
    /// Capsule B core-axis end endpoint.
    pub b1: [f32; 3],
    /// Capsule B radius.
    pub rb: f32,
}

impl CapsuleClosestQuery {
    /// Builds a query from the two capsule core axes and their radii.
    ///
    /// Provenance: twinned from this repository's
    /// [`capsule_capsule_closest`](prism_render_architecture::particle::capsule_capsule_closest);
    /// no third-party engine source or derived code.
    #[must_use]
    pub const fn new(
        a0: [f32; 3],
        a1: [f32; 3],
        ra: f32,
        b0: [f32; 3],
        b1: [f32; 3],
        rb: f32,
    ) -> CapsuleClosestQuery {
        CapsuleClosestQuery {
            a0,
            a1,
            ra,
            b0,
            b1,
            rb,
        }
    }
}

/// The resolved answer for one capsule pair, mirroring every field the
/// reference [`CapsuleHit`](prism_render_architecture::particle::capsule_capsule_closest::CapsuleHit)
/// reports.
///
/// Provenance: twinned from this repository's
/// [`capsule_capsule_closest`](prism_render_architecture::particle::capsule_capsule_closest);
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CapsuleClosestResult {
    /// Signed surface gap `axis_dist - (ra + rb)`, matching `CapsuleHit::distance`.
    pub distance: f32,
    /// Non-negative overlap depth, matching `CapsuleHit::penetration`.
    pub penetration: f32,
    /// Whether the two capsule surfaces touch or overlap, matching
    /// `CapsuleHit::intersecting`.
    pub intersecting: bool,
    /// Unit contact normal from A to B, matching `CapsuleHit::normal`.
    pub normal: [f32; 3],
    /// Closest point on capsule A's surface, matching `CapsuleHit::point_a`.
    pub point_a: [f32; 3],
    /// Closest point on capsule B's surface, matching `CapsuleHit::point_b`.
    pub point_b: [f32; 3],
}

/// `repr(C)` `std430` layout of one packed query: four `vec4` slots holding
/// `(a0.xyz, ra)`, `(a1.xyz, pad)`, `(b0.xyz, rb)` and `(b1.xyz, pad)` — `64`
/// bytes, each `vec3` on its `16`-byte-aligned slot exactly as the `WGSL`
/// `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Capsule A core-axis start endpoint.
    a0: [f32; 3],
    /// Capsule A radius, packed in the fourth lane of the first slot.
    ra: f32,
    /// Capsule A core-axis end endpoint.
    a1: [f32; 3],
    /// Padding lane after capsule A's end.
    pad0: f32,
    /// Capsule B core-axis start endpoint.
    b0: [f32; 3],
    /// Capsule B radius, packed in the fourth lane of the third slot.
    rb: f32,
    /// Capsule B core-axis end endpoint.
    b1: [f32; 3],
    /// Padding lane after capsule B's end.
    pad1: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &CapsuleClosestQuery) -> GpuQuery {
        GpuQuery {
            a0: query.a0,
            ra: query.ra,
            a1: query.a1,
            pad0: 0.0,
            b0: query.b0,
            rb: query.rb,
            b1: query.b1,
            pad1: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: a four-scalar slot
/// `(distance, penetration, intersecting, pad)`, then three `vec4` slots for the
/// unit normal, the surface point on A and the surface point on B — `64` bytes
/// matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Signed surface gap.
    gap: f32,
    /// Non-negative penetration depth.
    penetration: f32,
    /// Discrete intersecting flag (`0` or `1`).
    intersecting: u32,
    /// Padding word.
    pad0: u32,
    /// Unit contact normal from A to B.
    normal: [f32; 3],
    /// Padding lane after the normal.
    pad1: f32,
    /// Closest point on capsule A's surface.
    point_a: [f32; 3],
    /// Padding lane after the point on A.
    pad2: f32,
    /// Closest point on capsule B's surface.
    point_b: [f32; 3],
    /// Padding lane after the point on B.
    pad3: f32,
}

/// Uniform parameters for one dispatch: the pair count plus three pad words to
/// fill a `16`-byte, `16`-byte-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of capsule pairs in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A compiled, reusable capsule-vs-capsule closest compute pipeline.
///
/// Provenance: twinned from this repository's
/// [`capsule_capsule_closest`](prism_render_architecture::particle::capsule_capsule_closest);
/// no third-party engine source or derived code.
pub struct GpuCapsuleCapsuleClosest {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCapsuleCapsuleClosest {
    /// Compiles the capsule-vs-capsule closest kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: twinned from this repository's
    /// [`capsule_capsule_closest`](prism_render_architecture::particle::capsule_capsule_closest);
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCapsuleCapsuleClosest {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_capsule_capsule_closest"),
            source: ShaderSource::Wgsl(CAPSULE_CAPSULE_CLOSEST_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_capsule_capsule_closest_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_capsule_capsule_closest_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_capsule_capsule_closest_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCapsuleCapsuleClosest {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every capsule pair on-device and returns one
    /// [`CapsuleClosestResult`] per input, in order.
    ///
    /// Each result equals the reference
    /// [`capsule_capsule_closest`](prism_render_architecture::particle::capsule_capsule_closest)
    /// answer to within the tolerance documented on this module, with the
    /// discrete `intersecting` flag matching exactly. An empty input returns an
    /// empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    ///
    /// Provenance: twinned from this repository's
    /// [`capsule_capsule_closest`](prism_render_architecture::particle::capsule_capsule_closest);
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[CapsuleClosestQuery],
    ) -> Vec<CapsuleClosestResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_capsule_capsule_closest_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_capsule_capsule_closest_output"),
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
            label: Some("prism_volumetric_capsule_capsule_closest_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_capsule_capsule_closest_bind_group"),
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
            label: Some("prism_volumetric_capsule_capsule_closest_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_capsule_capsule_closest_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_capsule_capsule_closest_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per capsule pair, flattened to a 1-D dispatch.
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

/// Decodes one packed [`GpuResult`] into the public [`CapsuleClosestResult`].
fn decode_result(raw: &GpuResult) -> CapsuleClosestResult {
    CapsuleClosestResult {
        distance: raw.gap,
        penetration: raw.penetration,
        intersecting: raw.intersecting != 0,
        normal: raw.normal,
        point_a: raw.point_a,
        point_b: raw.point_b,
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
