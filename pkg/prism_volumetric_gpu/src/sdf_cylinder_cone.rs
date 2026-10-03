//! `wgpu` compute twin of the three analytic cylinder/cone signed-distance
//! primitives of the `CPU` golden path `capped_cylinder`, `capped_cone` and
//! `rounded_cylinder` in
//! `prism_render_architecture::ray_scene::sdf_primitives`.
//!
//! Implicit modelling needs *analytic* primitives whose exact signed distance
//! is known in closed form rather than sampled on a grid. The reference derives
//! three `y`-axis solids: a `capped_cylinder` (a rectangular meridian profile),
//! the Inigo-Quilez `capped_cone` (two cap radii joined by a slanted side), and
//! a `rounded_cylinder` (a cylinder with filleted vertical edges).
//! [`GpuSdfCylinderCone`] is the on-device twin: each thread reads one point
//! plus all three shapes' parameters and writes all three signed distances,
//! reproducing the reference closed forms with only `sqrt`, `abs`, `min`,
//! `max`, `clamp`, `select`, products and quotients.
//!
//! # What is twinned
//!
//! Each thread reads one [`SdfCylinderConeQuery`] — a query `point` plus the
//! `capped_cylinder` (`cyl_half_height`, `cyl_radius`), the `capped_cone`
//! (`cone_half_height`, `cone_bottom_radius`, `cone_top_radius`) and the
//! `rounded_cylinder` (`round_outer_radius`, `round_rounding`,
//! `round_half_height`) parameters — and writes one [`SdfCylinderConeResult`]
//! holding the three signed distances `capped_cylinder_sd`, `capped_cone_sd`
//! and `rounded_cylinder_sd`.
//!
//! The `capped_cylinder` kernel reduces the point to its radial distance in the
//! `xz` plane paired with its `y` offset, then measures that pair against the
//! rectangular cross-section with the exact interior/exterior split. The
//! `capped_cone` kernel compares the squared distance to the nearer cap rim
//! against the squared distance to the clamped projection onto the slanted
//! side segment, takes the smaller, and signs it negative when the point lies
//! inside both the lateral and the cap slabs. The `rounded_cylinder` kernel
//! insets the rectangular profile by the fillet radius, applies the same
//! box split in the meridian plane, then offsets outward by the fillet radius.
//!
//! # What stays on the host
//!
//! The domain and `CSG` operators that compose these atoms into complex shapes,
//! the ray-marcher that evaluates them along a ray, and the surface-normal
//! estimation all stay on the host; the device sees only the three stateless,
//! fixed-width signed-distance evaluations, one query at a time, so a storage
//! buffer is never zero-sized.
//!
//! # Correctness model
//!
//! All three distances thread through `sqrt`, products and quotients, so the
//! `CPU` and `GPU` are not bit-exact: a `GPU` `sqrt` or divide may land a few
//! units in the last place from the scalar reference. The parity test asserts
//! each distance within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, tight enough
//! to catch a genuinely wrong port yet loose enough to admit a legal last-place
//! difference. Fixtures stay clear of the `capped_cone` sign-flip boundary and
//! of the `min` tie between the rim and the side distances, where the `CPU` and
//! `GPU` could pick different sides of a `select`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `abs`,
//! `min`, `max`, `clamp`, `select`, `+ - * /` and unsigned index arithmetic —
//! with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry,
//! no `round` and no `ceil`, and no `f64`/`u64`/`u16`/`i64`/`i16`. It runs
//! unmodified on `Metal`, `Vulkan` and `DX12`.
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

/// The portable core-`WGSL` cylinder/cone signed-distance kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden `capped_cylinder`, `capped_cone` and
/// `rounded_cylinder` closed forms; see the module documentation for the
/// algorithm.
const SDF_CYLINDER_CONE_WGSL: &str = r#"
// Cylinder/cone signed-distance twin: one thread computes one query point's
// capped-cylinder, capped-cone and rounded-cylinder signed distances, mirroring
// the CPU golden `ray_scene::sdf_primitives::{capped_cylinder, capped_cone,
// rounded_cylinder}` with only sqrt, abs, min, max, clamp, select, products and
// quotients. The domain/CSG operators and the ray-marcher stay on the host.
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
    // Capped cylinder: half-height along y and lateral radius.
    cyl_half_height: f32,
    cyl_radius: f32,
    // Capped cone: half-height, bottom cap radius and top cap radius.
    cone_half_height: f32,
    cone_bottom_radius: f32,
    cone_top_radius: f32,
    // Rounded cylinder: outer radius, fillet radius and half-height.
    round_outer_radius: f32,
    round_rounding: f32,
    round_half_height: f32,
    pad0: f32,
}

struct Distances {
    // Capped-cylinder signed distance.
    capped_cylinder_sd: f32,
    // Capped-cone signed distance.
    capped_cone_sd: f32,
    // Rounded-cylinder signed distance.
    rounded_cylinder_sd: f32,
    pad0: f32,
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

    // --- Capped cylinder: box SDF in the (radial, |y|) plane. ---
    let radial = sqrt(q.px * q.px + q.pz * q.pz);
    let cyl_dx = radial - q.cyl_radius;
    let cyl_dy = abs(q.py) - q.cyl_half_height;
    let cyl_inside = min(max(cyl_dx, cyl_dy), 0.0);
    let cyl_mx = max(cyl_dx, 0.0);
    let cyl_my = max(cyl_dy, 0.0);
    let cyl_outside = sqrt(cyl_mx * cyl_mx + cyl_my * cyl_my);
    out.capped_cylinder_sd = cyl_inside + cyl_outside;

    // --- Capped cone: nearer cap rim vs clamped projection onto the side. ---
    let qx = radial;
    let qy = q.py;
    let k1x = q.cone_top_radius;
    let k1y = q.cone_half_height;
    let k2x = q.cone_top_radius - q.cone_bottom_radius;
    let k2y = 2.0 * q.cone_half_height;
    // Snap the radius to whichever cap the point faces for the rim distance.
    let cap_radius = select(q.cone_top_radius, q.cone_bottom_radius, qy < 0.0);
    let ca_x = qx - min(qx, cap_radius);
    let ca_y = abs(qy) - q.cone_half_height;
    // Project onto the slanted side segment, parameter clamped to the caps.
    let km_x = k1x - qx;
    let km_y = k1y - qy;
    let dot_k2 = k2x * k2x + k2y * k2y;
    let proj = clamp((km_x * k2x + km_y * k2y) / dot_k2, 0.0, 1.0);
    let cb_x = qx - k1x + k2x * proj;
    let cb_y = qy - k1y + k2y * proj;
    let cone_sign = select(1.0, -1.0, cb_x < 0.0 && ca_y < 0.0);
    let dca = ca_x * ca_x + ca_y * ca_y;
    let dcb = cb_x * cb_x + cb_y * cb_y;
    out.capped_cone_sd = cone_sign * sqrt(min(dca, dcb));

    // --- Rounded cylinder: inset box profile offset outward by the fillet. ---
    let rnd_dx = radial - (q.round_outer_radius - q.round_rounding);
    let rnd_dy = abs(q.py) - (q.round_half_height - q.round_rounding);
    let rnd_inside = min(max(rnd_dx, rnd_dy), 0.0);
    let rnd_mx = max(rnd_dx, 0.0);
    let rnd_my = max(rnd_dy, 0.0);
    let rnd_outside = sqrt(rnd_mx * rnd_mx + rnd_my * rnd_my);
    out.rounded_cylinder_sd = rnd_inside + rnd_outside - q.round_rounding;

    out.pad0 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SDF_CYLINDER_CONE_WGSL`].
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
/// the point components plus all three shapes' parameters and one pad word to a
/// `48`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Query point `x`.
    px: f32,
    /// Query point `y`.
    py: f32,
    /// Query point `z`.
    pz: f32,
    /// Capped-cylinder half-height.
    cyl_half_height: f32,
    /// Capped-cylinder lateral radius.
    cyl_radius: f32,
    /// Capped-cone half-height.
    cone_half_height: f32,
    /// Capped-cone bottom cap radius.
    cone_bottom_radius: f32,
    /// Capped-cone top cap radius.
    cone_top_radius: f32,
    /// Rounded-cylinder outer radius.
    round_outer_radius: f32,
    /// Rounded-cylinder fillet radius.
    round_rounding: f32,
    /// Rounded-cylinder half-height.
    round_half_height: f32,
    /// Padding word.
    pad0: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Distances`
/// struct: the three signed distances plus one pad word to a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Capped-cylinder signed distance.
    capped_cylinder_sd: f32,
    /// Capped-cone signed distance.
    capped_cone_sd: f32,
    /// Rounded-cylinder signed distance.
    rounded_cylinder_sd: f32,
    /// Padding word.
    pad0: f32,
}

/// One query for the cylinder/cone signed-distance twin: the query `point` plus
/// the `capped_cylinder`, `capped_cone` and `rounded_cylinder` shape
/// parameters.
///
/// `point` is the evaluation position; `cyl_half_height`/`cyl_radius` are the
/// `capped_cylinder` extents; `cone_half_height`/`cone_bottom_radius`/
/// `cone_top_radius` are the `capped_cone` extents; `round_outer_radius`/
/// `round_rounding`/`round_half_height` are the `rounded_cylinder` extents.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfCylinderConeQuery {
    /// Query point `[x, y, z]`.
    pub point: [f32; 3],
    /// Capped-cylinder half-height along `y`.
    pub cyl_half_height: f32,
    /// Capped-cylinder lateral radius.
    pub cyl_radius: f32,
    /// Capped-cone half-height along `y`.
    pub cone_half_height: f32,
    /// Capped-cone bottom cap radius.
    pub cone_bottom_radius: f32,
    /// Capped-cone top cap radius.
    pub cone_top_radius: f32,
    /// Rounded-cylinder outer radius.
    pub round_outer_radius: f32,
    /// Rounded-cylinder fillet radius.
    pub round_rounding: f32,
    /// Rounded-cylinder half-height along `y`.
    pub round_half_height: f32,
}

impl SdfCylinderConeQuery {
    /// Builds a query from the point and all three shapes' parameters.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the query flattens three shapes' parameters into one record"
    )]
    pub const fn new(
        point: [f32; 3],
        cyl_half_height: f32,
        cyl_radius: f32,
        cone_half_height: f32,
        cone_bottom_radius: f32,
        cone_top_radius: f32,
        round_outer_radius: f32,
        round_rounding: f32,
        round_half_height: f32,
    ) -> SdfCylinderConeQuery {
        SdfCylinderConeQuery {
            point,
            cyl_half_height,
            cyl_radius,
            cone_half_height,
            cone_bottom_radius,
            cone_top_radius,
            round_outer_radius,
            round_rounding,
            round_half_height,
        }
    }
}

/// One resolved query of the cylinder/cone signed-distance twin: the
/// `capped_cylinder`, `capped_cone` and `rounded_cylinder` signed distances at
/// the query point.
///
/// Every field is negative inside the solid, positive outside, zero on the
/// surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfCylinderConeResult {
    /// Capped-cylinder signed distance.
    pub capped_cylinder_sd: f32,
    /// Capped-cone signed distance.
    pub capped_cone_sd: f32,
    /// Rounded-cylinder signed distance.
    pub rounded_cylinder_sd: f32,
}

/// Encodes one [`SdfCylinderConeQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SdfCylinderConeQuery) -> GpuQuery {
    GpuQuery {
        px: q.point[0],
        py: q.point[1],
        pz: q.point[2],
        cyl_half_height: q.cyl_half_height,
        cyl_radius: q.cyl_radius,
        cone_half_height: q.cone_half_height,
        cone_bottom_radius: q.cone_bottom_radius,
        cone_top_radius: q.cone_top_radius,
        round_outer_radius: q.round_outer_radius,
        round_rounding: q.round_rounding,
        round_half_height: q.round_half_height,
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SdfCylinderConeResult`].
fn decode_result(raw: &GpuResult) -> SdfCylinderConeResult {
    SdfCylinderConeResult {
        capped_cylinder_sd: raw.capped_cylinder_sd,
        capped_cone_sd: raw.capped_cone_sd,
        rounded_cylinder_sd: raw.rounded_cylinder_sd,
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

/// A compiled, reusable cylinder/cone signed-distance compute pipeline,
/// twinning the `CPU` golden `capped_cylinder`, `capped_cone` and
/// `rounded_cylinder` of `prism_render_architecture::ray_scene::sdf_primitives`.
pub struct GpuSdfCylinderCone {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfCylinderCone {
    /// Compiles the cylinder/cone signed-distance kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdfCylinderCone {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sdf_cylinder_cone"),
            source: ShaderSource::Wgsl(SDF_CYLINDER_CONE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sdf_cylinder_cone_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sdf_cylinder_cone_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sdf_cylinder_cone_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfCylinderCone {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`SdfCylinderConeResult`] per input, in order.
    ///
    /// The signed distances match the reference to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SdfCylinderConeQuery],
    ) -> Vec<SdfCylinderConeResult> {
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
            label: Some("prism_volumetric_sdf_cylinder_cone_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_cylinder_cone_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sdf_cylinder_cone_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sdf_cylinder_cone_bind_group"),
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
            label: Some("prism_volumetric_sdf_cylinder_cone_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sdf_cylinder_cone_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sdf_cylinder_cone_pass"),
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
