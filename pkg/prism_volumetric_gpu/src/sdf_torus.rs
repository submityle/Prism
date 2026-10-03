//! `wgpu` compute twin of the two analytic torus signed-distance primitives of
//! the `CPU` golden path
//! ([`torus`](prism_render_architecture::ray_scene::sdf_primitives::torus) and
//! [`capped_torus`](prism_render_architecture::ray_scene::sdf_primitives::capped_torus)).
//!
//! Implicit modelling needs *analytic* primitives whose exact signed distance
//! is known in closed form rather than sampled on a grid. The reference derives
//! two swept-ring distances: a full [`torus`] in the `xz` plane and the
//! Inigo-Quilez [`capped_torus`], a torus arc bounded by an aperture whose
//! trigonometry is folded by the caller into a `sin`/`cos` pair so the routine
//! itself stays transcendental-free. [`GpuSdfTorus`] is the on-device twin:
//! each thread reads one point plus both shapes' parameters and writes both
//! signed distances, reproducing the reference closed forms with only `sqrt`,
//! `abs`, `max`, products and quotients.
//!
//! # What is twinned
//!
//! Each thread reads one [`SdfTorusQuery`] — a query `point` plus the full
//! torus (`major_radius`, `minor_radius`) and capped-torus (`sin_aperture`,
//! `cos_aperture`, `capped_major_radius`, `tube_radius`) parameters — and writes
//! one [`SdfTorusResult`] holding the two signed distances `torus_sd` and
//! `capped_torus_sd`. The torus kernel reduces the point to its distance from
//! the ring circle in the `xz` plane paired with its `y` offset, then subtracts
//! the tube radius. The capped-torus kernel folds the point across the `x`
//! axis, selects by a single ordered comparison whether the nearest feature is
//! the flat cap (`k = px * sin_a + py * cos_a`) or the full ring
//! (`k = sqrt(px^2 + py^2)`), forms the ring distance through a guarded
//! `max(_, 0)` square root, and subtracts the tube radius.
//!
//! # What stays on the host
//!
//! The domain and `CSG` operators that compose these atoms into complex shapes,
//! the ray-marcher that evaluates them along a ray, and the surface-normal
//! estimation all stay on the host; the device sees only the two stateless,
//! fixed-width signed-distance evaluations, one query at a time, so a storage
//! buffer is never zero-sized.
//!
//! # Correctness model
//!
//! Both distances thread through `sqrt`, products and quotients, so the `CPU`
//! and `GPU` are not bit-exact: a `GPU` `sqrt` or divide may land a few units in
//! the last place from the scalar reference. The parity test asserts each
//! distance within `abs_diff <= 1e-5` or `rel_diff <= 1e-5`, tight enough to
//! catch a genuinely wrong port yet loose enough to admit a legal last-place
//! difference. Fixtures stay clear of the capped-torus branch boundary
//! `cos_a * px == sin_a * py`, where the `CPU` and `GPU` could pick different
//! sides of the select.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `abs`, `max`,
//! `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `tan`, no inverse trigonometry, no `round` and no `ceil`, and
//! no `f64`/`u64`/`u16`/`i64`/`i16`. It runs unmodified on `Metal`, `Vulkan`
//! and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_primitives`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` torus signed-distance kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`torus`](prism_render_architecture::ray_scene::sdf_primitives::torus) and
/// [`capped_torus`](prism_render_architecture::ray_scene::sdf_primitives::capped_torus)
/// closed forms; see the module documentation for the algorithm.
const SDF_TORUS_WGSL: &str = r#"
// Torus signed-distance twin: one thread computes one query point's full-torus
// signed distance and capped-torus signed distance, mirroring the CPU golden
// `ray_scene::sdf_primitives::{torus, capped_torus}` with only sqrt, abs, max,
// products and quotients. The domain/CSG operators and the ray-marcher stay on
// the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::sdf_primitives；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Query point components.
    px: f32,
    py: f32,
    pz: f32,
    // Full torus: ring centre-to-tube-centre radius and tube radius.
    major_radius: f32,
    minor_radius: f32,
    // Capped torus: aperture sine/cosine, ring radius and tube radius.
    sin_aperture: f32,
    cos_aperture: f32,
    capped_major_radius: f32,
    tube_radius: f32,
    pad0: f32,
}

struct Distances {
    // Full-torus signed distance.
    torus_sd: f32,
    // Capped-torus signed distance.
    capped_torus_sd: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Distances>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Distances;

    // Full torus: reduce the point to its distance from the ring circle in the
    // xz plane paired with its y offset, then subtract the tube radius.
    let ring = sqrt(q.px * q.px + q.pz * q.pz) - q.major_radius;
    out.torus_sd = sqrt(ring * ring + q.py * q.py) - q.minor_radius;

    // Capped torus: fold across the x axis (px only), then a single ordered
    // comparison selects whether the nearest feature is the flat cap or the
    // full ring. The ring distance is a guarded max(_, 0) square root.
    let apx = abs(q.px);
    let apy = q.py;
    let apz = q.pz;
    var k: f32;
    if (q.cos_aperture * apx > q.sin_aperture * apy) {
        k = apx * q.sin_aperture + apy * q.cos_aperture;
    } else {
        k = sqrt(apx * apx + apy * apy);
    }
    let p_dot_p = apx * apx + apy * apy + apz * apz;
    let under = p_dot_p + q.capped_major_radius * q.capped_major_radius
        - 2.0 * q.capped_major_radius * k;
    out.capped_torus_sd = sqrt(max(under, 0.0)) - q.tube_radius;

    out.pad0 = 0.0;
    out.pad1 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SDF_TORUS_WGSL`].
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct:
/// the point components plus both shapes' parameters and one pad word to a
/// `40`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Query point `x`.
    px: f32,
    /// Query point `y`.
    py: f32,
    /// Query point `z`.
    pz: f32,
    /// Full torus ring radius.
    major_radius: f32,
    /// Full torus tube radius.
    minor_radius: f32,
    /// Capped-torus aperture sine.
    sin_aperture: f32,
    /// Capped-torus aperture cosine.
    cos_aperture: f32,
    /// Capped-torus ring radius.
    capped_major_radius: f32,
    /// Capped-torus tube radius.
    tube_radius: f32,
    /// Padding word.
    pad0: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Distances`
/// struct: the two signed distances plus two pad words to a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Full-torus signed distance.
    torus_sd: f32,
    /// Capped-torus signed distance.
    capped_torus_sd: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
}

/// One query for the torus signed-distance twin: the query `point` plus the
/// full-torus and capped-torus shape parameters.
///
/// `point` is the evaluation position; `major_radius`/`minor_radius` are the
/// full [`torus`](prism_render_architecture::ray_scene::sdf_primitives::torus)
/// ring and tube radii; `sin_aperture`/`cos_aperture` are the folded aperture
/// trigonometry, `capped_major_radius` the ring radius and `tube_radius` the
/// tube radius of the
/// [`capped_torus`](prism_render_architecture::ray_scene::sdf_primitives::capped_torus).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfTorusQuery {
    /// Query point `[x, y, z]`.
    pub point: [f32; 3],
    /// Full-torus ring centre-to-tube-centre radius.
    pub major_radius: f32,
    /// Full-torus tube radius.
    pub minor_radius: f32,
    /// Capped-torus aperture sine.
    pub sin_aperture: f32,
    /// Capped-torus aperture cosine.
    pub cos_aperture: f32,
    /// Capped-torus ring radius.
    pub capped_major_radius: f32,
    /// Capped-torus tube radius.
    pub tube_radius: f32,
}

impl SdfTorusQuery {
    /// Builds a query from the point and both shapes' parameters.
    #[must_use]
    pub const fn new(
        point: [f32; 3],
        major_radius: f32,
        minor_radius: f32,
        sin_aperture: f32,
        cos_aperture: f32,
        capped_major_radius: f32,
        tube_radius: f32,
    ) -> SdfTorusQuery {
        SdfTorusQuery {
            point,
            major_radius,
            minor_radius,
            sin_aperture,
            cos_aperture,
            capped_major_radius,
            tube_radius,
        }
    }
}

/// One resolved query of the torus signed-distance twin: the full-torus and
/// capped-torus signed distances at the query point.
///
/// `torus_sd` is
/// [`torus`](prism_render_architecture::ray_scene::sdf_primitives::torus);
/// `capped_torus_sd` is
/// [`capped_torus`](prism_render_architecture::ray_scene::sdf_primitives::capped_torus).
/// Both are negative inside the solid, positive outside, zero on the surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfTorusResult {
    /// Full-torus signed distance.
    pub torus_sd: f32,
    /// Capped-torus signed distance.
    pub capped_torus_sd: f32,
}

/// Encodes one [`SdfTorusQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SdfTorusQuery) -> GpuQuery {
    GpuQuery {
        px: q.point[0],
        py: q.point[1],
        pz: q.point[2],
        major_radius: q.major_radius,
        minor_radius: q.minor_radius,
        sin_aperture: q.sin_aperture,
        cos_aperture: q.cos_aperture,
        capped_major_radius: q.capped_major_radius,
        tube_radius: q.tube_radius,
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SdfTorusResult`].
fn decode_result(raw: &GpuResult) -> SdfTorusResult {
    SdfTorusResult {
        torus_sd: raw.torus_sd,
        capped_torus_sd: raw.capped_torus_sd,
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

/// A compiled, reusable torus signed-distance compute pipeline, twinning the
/// `CPU` golden
/// [`torus`](prism_render_architecture::ray_scene::sdf_primitives::torus) and
/// [`capped_torus`](prism_render_architecture::ray_scene::sdf_primitives::capped_torus).
pub struct GpuSdfTorus {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfTorus {
    /// Compiles the torus signed-distance kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdfTorus {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sdf_torus"),
            source: ShaderSource::Wgsl(SDF_TORUS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sdf_torus_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sdf_torus_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sdf_torus_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfTorus {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`SdfTorusResult`] per
    /// input, in order.
    ///
    /// The signed distances match the reference to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[SdfTorusQuery]) -> Vec<SdfTorusResult> {
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
            label: Some("prism_volumetric_sdf_torus_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_torus_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sdf_torus_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sdf_torus_bind_group"),
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
            label: Some("prism_volumetric_sdf_torus_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sdf_torus_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sdf_torus_pass"),
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
