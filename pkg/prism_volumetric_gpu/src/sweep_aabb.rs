//! `wgpu` compute twin of the analytic swept *axis-aligned bounding box*
//! (`AABB`) *time-of-impact* (`TOI`) golden
//! ([`sweep_aabb`](prism_render_architecture::particle::sweep_aabb), design §10,
//! §13).
//!
//! The particle broad phase needs, per pair of uniformly moving boxes, the
//! continuous first-contact answer: *does* the pair touch within the unit
//! timestep `t ∈ [0, 1]`, *when* (the fraction `toi`), *along which axis*, and
//! *with which contact normal*. The `CPU` golden
//! [`swept_toi`](prism_render_architecture::particle::sweep_aabb::swept_toi)
//! owns that separating-axis "slab" math; [`GpuSweepAabb`] is the on-device
//! twin that runs one thread per pair and reproduces every lane. A passing
//! real-device parity test is therefore direct evidence the ported kernel folds
//! the same relative-velocity reduction, the same three per-axis Minkowski-slab
//! entry/exit intervals, the same near-zero-velocity classification guard and
//! the same running interval intersection the reference does, not merely that
//! its shader compiles.
//!
//! # What is twinned
//!
//! The kernel reproduces
//! [`swept_toi`](prism_render_architecture::particle::sweep_aabb::swept_toi)
//! guard-for-guard. The pair is reduced to `a`'s box moving with the relative
//! velocity `a.vel - b.vel` against a stationary `b`. Each axis forms an
//! entry/exit interval from the Minkowski-expanded slab: an axis whose relative
//! velocity magnitude is below
//! [`EPS`](prism_render_architecture::particle::sweep_aabb::EPS) has no relative
//! motion and either rejects the whole pair (already separated on that axis, a
//! guaranteed miss) or contributes the sentinel interval `(-inf, +inf)`;
//! every other axis forms the reciprocal `1 / v` and its ordered `(entry, exit)`
//! from the two slab faces. The running `t_entry = max` of the per-axis entries
//! and `t_exit = min` of the per-axis exits give the contact window; the pair
//! touches when `t_entry <= t_exit`, `t_entry <= 1` and `t_exit >= 0` (compared
//! with `<=` / `>=`, never with `==`). The last axis to enter carries the
//! separating normal, `toi` is `t_entry` clamped to `0`, and a pair already
//! overlapping at `t = 0` reports `toi == 0`, `initially_overlapping == true`
//! and a zero normal, exactly as the reference.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `+ - * /` and one `bitcast` to materialize the `IEEE`-754 infinity sentinel
//! that mirrors the reference's [`f32::INFINITY`] slab bounds — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `sqrt` or optional device feature (the box faces
//! are flat, so there is no quadratic and no square root). It therefore runs
//! unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! Each pair is a fixed, non-reorderable reduction to a relative velocity, three
//! guarded divisions and a running interval intersection, so `CPU` and `GPU`
//! evaluate the same closed form in the same associativity. They are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The parity test therefore asserts a tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`) on `toi` and the normal lanes yet an *exact* match on the
//! discrete hit flag, the axis index and the `initially_overlapping` flag, which
//! are routed through the same `EPS` magnitude band the reference uses so a pair
//! placed clear of a classification boundary folds the identical verdict.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::sweep_aabb`；
//! standard separating-axis swept-`AABB` continuous-collision time-of-impact
//! plus `wgpu` compute dispatch; no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::sweep_aabb::{MovingAabb, SweepResult, Vec3};
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
/// shared by every kernel in this crate.
const WORKGROUP_SIZE: u32 = 64;

/// Discrete hit code written by the kernel for a lane whose boxes touch within
/// the step: matches the host `== 1` decode in [`decode_result`]. A direct `f32`
/// equality is forbidden, so the kernel emits an integer flag rather than a
/// sentinel float.
const CODE_HIT: u32 = 1;

/// The portable core-`WGSL` swept-`AABB` time-of-impact kernel, embedded inline
/// so the twin ships as a single source file. Mirrors the `CPU` golden
/// [`swept_toi`](prism_render_architecture::particle::sweep_aabb::swept_toi)
/// division guard for guard and uses only the portable core subset plus one
/// `bitcast` for the infinity sentinel, so it needs no optional device feature.
const SWEEP_AABB_WGSL: &str = r#"
// Swept AABB time-of-impact twin: one thread per (MovingAabb, MovingAabb) pair
// reduces the pair to the relative velocity a.vel - b.vel, folds the three
// per-axis Minkowski-slab entry/exit intervals into the running contact window
// [t_entry, t_exit], then writes the hit flag, the clamped toi, the last-entering
// axis index, the contact normal and the initially-overlapping flag. It mirrors
// the CPU golden `particle::sweep_aabb` guard for guard, uses only the portable
// core-WGSL subset (abs/min/max, + - * / and one bitcast for the infinity
// sentinel), and takes no optional feature, so it runs unmodified on Metal,
// Vulkan and DX12.
//
// Provenance: standard separating-axis swept-AABB continuous-collision test; no
// third-party engine source or derived code.

struct Params {
    // Number of valid pairs in `queries`.
    count: u32,
    // Padding to a 16-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. 96-byte std430 stride matching the host `GpuQuery`: box a's min and
// max corners and velocity, then box b's min and max corners and velocity, each
// padded to a vec4 so the storage array needs no manual vec3 alignment
// arithmetic.
struct Query {
    a_min: vec4<f32>,
    a_max: vec4<f32>,
    a_vel: vec4<f32>,
    b_min: vec4<f32>,
    b_max: vec4<f32>,
    b_vel: vec4<f32>,
}

// One result. 32-byte std430 stride matching the host `GpuResult`: the hit flag
// as 0u/1u, the last-entering axis index, the initially-overlapping flag, one
// pad word, the clamped time-of-impact and the three contact-normal lanes.
struct Result {
    hit: u32,
    axis: u32,
    overlap: u32,
    pad0: u32,
    toi: f32,
    nx: f32,
    ny: f32,
    nz: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Magnitude floor guarding the slab `1 / v` division and classifying a relative
// velocity component as "no relative motion", matching the reference `EPS`. A
// direct f32 `==`/`!=` is forbidden, so the near-zero test compares magnitude
// against this floor instead of exact zero.
const EPS: f32 = 1.0e-6;

// Per-axis entry/exit interval for the moving-point-vs-slab test. `miss` is set
// when the axis has no relative motion and the boxes are already separated on
// it, meaning they can never overlap; otherwise `(entry, exit)` is the ordered
// contact interval, which is the sentinel `(-inf, +inf)` for a no-motion axis
// that is currently overlapping.
struct Interval {
    entry: f32,
    exit: f32,
    miss: bool,
}

// The shared per-axis slab solve, mirroring the reference `axis_interval` guard
// for guard: a near-zero relative velocity either rejects the pair (already
// separated) or yields the unbounded sentinel interval; a positive component
// forms (entry, exit) from the lower/upper faces, and a negative component swaps
// them so the interval stays ordered.
fn axis_interval(a_min: f32, a_max: f32, b_min: f32, b_max: f32, v: f32) -> Interval {
    var out: Interval;
    out.entry = 0.0;
    out.exit = 0.0;
    out.miss = false;

    // IEEE-754 +/-infinity sentinels, mirroring the reference's `f32::INFINITY`
    // unbounded slab exactly; built by bitcast because WGSL has no inf literal.
    let pos_inf = bitcast<f32>(0x7f800000u);
    let neg_inf = bitcast<f32>(0xff800000u);

    if (abs(v) < EPS) {
        if (a_max < b_min || a_min > b_max) {
            out.miss = true;
        } else {
            out.entry = neg_inf;
            out.exit = pos_inf;
        }
    } else if (v > 0.0) {
        let inv = 1.0 / v;
        out.entry = (b_min - a_max) * inv;
        out.exit = (b_max - a_min) * inv;
    } else {
        let inv = 1.0 / v;
        out.entry = (b_max - a_min) * inv;
        out.exit = (b_min - a_max) * inv;
    }
    return out;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let a_min = queries[idx].a_min.xyz;
    let a_max = queries[idx].a_max.xyz;
    let b_min = queries[idx].b_min.xyz;
    let b_max = queries[idx].b_max.xyz;
    // Reduce the pair to box a moving with the relative velocity against b.
    let rv = queries[idx].a_vel.xyz - queries[idx].b_vel.xyz;

    var res: Result;
    res.hit = 0u;
    res.axis = 0u;
    res.overlap = 0u;
    res.pad0 = 0u;
    res.toi = 0.0;
    res.nx = 0.0;
    res.ny = 0.0;
    res.nz = 0.0;

    let ix = axis_interval(a_min.x, a_max.x, b_min.x, b_max.x, rv.x);
    let iy = axis_interval(a_min.y, a_max.y, b_min.y, b_max.y, rv.y);
    let iz = axis_interval(a_min.z, a_max.z, b_min.z, b_max.z, rv.z);

    // Any axis with no relative motion and already-separated boxes is a hard
    // miss, mirroring the reference `?` early return on `axis_interval == None`.
    if (ix.miss || iy.miss || iz.miss) {
        results[idx] = res;
        return;
    }

    let t_entry = max(max(ix.entry, iy.entry), iz.entry);
    let t_exit = min(min(ix.exit, iy.exit), iz.exit);

    // No common contact window inside the unit step is a miss.
    if (t_entry > t_exit || t_entry > 1.0 || t_exit < 0.0) {
        results[idx] = res;
        return;
    }

    // The last axis to enter contact carries the separating normal; strict `>`
    // keeps the lower axis index on a tie, matching the reference.
    var axis = 0u;
    var best = ix.entry;
    if (iy.entry > best) {
        best = iy.entry;
        axis = 1u;
    }
    if (iz.entry > best) {
        axis = 2u;
    }

    let overlapping = t_entry <= 0.0;
    let toi = max(t_entry, 0.0);

    var rel_axis = rv.x;
    if (axis == 1u) {
        rel_axis = rv.y;
    } else if (axis == 2u) {
        rel_axis = rv.z;
    }

    // The contact normal is zero on an initial overlap or a no-motion contact
    // axis; otherwise it is the unit axis opposing the relative velocity sign.
    var nx = 0.0;
    var ny = 0.0;
    var nz = 0.0;
    if (!overlapping && abs(rel_axis) >= EPS) {
        var s = 1.0;
        if (rel_axis > 0.0) {
            s = -1.0;
        }
        if (axis == 0u) {
            nx = s;
        } else if (axis == 1u) {
            ny = s;
        } else {
            nz = s;
        }
    }

    res.hit = 1u;
    res.axis = axis;
    if (overlapping) {
        res.overlap = 1u;
    }
    res.toi = toi;
    res.nx = nx;
    res.ny = ny;
    res.nz = nz;
    results[idx] = res;
}
"#;

/// One swept-`AABB` query: the two uniformly moving boxes to test against each
/// other over the unit timestep.
///
/// Mirrors a single reference
/// [`swept_toi`](prism_render_architecture::particle::sweep_aabb::swept_toi)
/// call. Carrying both moving boxes per query lets one dispatch resolve many
/// independent pairs. Derives only [`PartialEq`] (no `Eq`/`Hash`) because it
/// holds `f32` geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SweepAabbQuery {
    /// The first moving box (`a`); the contact normal points toward it.
    pub a: MovingAabb,
    /// The second moving box (`b`); the reported axis and normal lie on its
    /// surface.
    pub b: MovingAabb,
}

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`SWEEP_AABB_WGSL`]: the pair count and three pad words — `16`
/// bytes, each field at the uniform offset the shader expects.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid pairs.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One query as uploaded. `96`-byte `std430` stride matching `Query` in the
/// shader: box `a`'s min and max corners and velocity, then box `b`'s min and
/// max corners and velocity, each padded to a `vec4` lane.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Box `a` `min` corner in `xyz`; the `w` lane is unused padding.
    a_min: [f32; 4],
    /// Box `a` `max` corner in `xyz`; the `w` lane is unused padding.
    a_max: [f32; 4],
    /// Box `a` velocity in `xyz`; the `w` lane is unused padding.
    a_vel: [f32; 4],
    /// Box `b` `min` corner in `xyz`; the `w` lane is unused padding.
    b_min: [f32; 4],
    /// Box `b` `max` corner in `xyz`; the `w` lane is unused padding.
    b_max: [f32; 4],
    /// Box `b` velocity in `xyz`; the `w` lane is unused padding.
    b_vel: [f32; 4],
}

impl GpuQuery {
    /// Packs a [`SweepAabbQuery`] into the `std430` upload layout.
    fn from_query(query: &SweepAabbQuery) -> GpuQuery {
        let a_lo = query.a.aabb.min;
        let a_hi = query.a.aabb.max;
        let a_v = query.a.vel;
        let b_lo = query.b.aabb.min;
        let b_hi = query.b.aabb.max;
        let b_v = query.b.vel;
        GpuQuery {
            a_min: [a_lo.x, a_lo.y, a_lo.z, 0.0],
            a_max: [a_hi.x, a_hi.y, a_hi.z, 0.0],
            a_vel: [a_v.x, a_v.y, a_v.z, 0.0],
            b_min: [b_lo.x, b_lo.y, b_lo.z, 0.0],
            b_max: [b_hi.x, b_hi.y, b_hi.z, 0.0],
            b_vel: [b_v.x, b_v.y, b_v.z, 0.0],
        }
    }
}

/// One result as read back. `32`-byte `std430` stride matching `Result` in the
/// shader: the hit flag as `0`/`1`, the last-entering axis index, the
/// initially-overlapping flag, one pad word, the clamped time-of-impact and the
/// three contact-normal lanes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Hit flag (`1` = the boxes touch within the step).
    hit: u32,
    /// Index of the last axis to enter contact (`0 = x`, `1 = y`, `2 = z`).
    axis: u32,
    /// Initial-overlap flag (`1` = already overlapping at `t = 0`).
    overlap: u32,
    /// Padding word.
    pad0: u32,
    /// Clamped first-contact fraction of the step.
    toi: f32,
    /// Contact-normal `x` lane.
    nx: f32,
    /// Contact-normal `y` lane.
    ny: f32,
    /// Contact-normal `z` lane.
    nz: f32,
}

/// Maps one kernel `Result` lane back to the host [`Option<SweepResult>`],
/// reusing the golden
/// [`SweepResult`](prism_render_architecture::particle::sweep_aabb::SweepResult)
/// type so the twin reports exactly what the reference does: [`None`] on a miss,
/// otherwise the clamped `toi`, the contact normal, the last-entering axis and
/// the initial-overlap flag.
fn decode_result(raw: &GpuResult) -> Option<SweepResult> {
    if raw.hit == CODE_HIT {
        Some(SweepResult {
            toi: raw.toi,
            normal: Vec3::new(raw.nx, raw.ny, raw.nz),
            axis: raw.axis as usize,
            initially_overlapping: raw.overlap == CODE_HIT,
        })
    } else {
        None
    }
}

/// A compiled, reusable swept-`AABB` time-of-impact pipeline.
pub struct GpuSweepAabb {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSweepAabb {
    /// Compiles the swept-`AABB` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSweepAabb {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sweep_aabb"),
            source: ShaderSource::Wgsl(SWEEP_AABB_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sweep_aabb_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sweep_aabb_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sweep_aabb_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSweepAabb {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every pair in `queries`, returning one [`Option<SweepResult>`] per
    /// query in input order.
    ///
    /// The returned result for query `q` mirrors
    /// [`swept_toi`](prism_render_architecture::particle::sweep_aabb::swept_toi)
    /// evaluated on `q.a` and `q.b`. An empty `queries` slice yields an empty
    /// result — storage buffers cannot be zero-sized, so it is handled by an
    /// early return before any dispatch.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[SweepAabbQuery]) -> Vec<Option<SweepResult>> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = GpuParams {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let gpu_queries: Vec<GpuQuery> = queries.iter().map(GpuQuery::from_query).collect();

        let out_bytes = (queries.len() * size_of::<GpuResult>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sweep_aabb_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sweep_aabb_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sweep_aabb_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sweep_aabb_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sweep_aabb_bind_group"),
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
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sweep_aabb_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sweep_aabb_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per pair, flattened to a 1-D dispatch.
            let groups = (queries.len() as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), queries.len());

        raw.iter().map(decode_result).collect()
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
