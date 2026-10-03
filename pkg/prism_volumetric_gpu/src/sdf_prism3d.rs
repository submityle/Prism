//! `wgpu` compute twin of three analytic prism/link signed-distance primitives
//! of the `CPU` golden path
//! ([`hex_prism`](prism_render_architecture::ray_scene::sdf_primitives::hex_prism),
//! [`octagon_prism`](prism_render_architecture::ray_scene::sdf_primitives::octagon_prism)
//! and [`link`](prism_render_architecture::ray_scene::sdf_primitives::link)).
//!
//! Implicit modelling needs *analytic* primitives whose exact signed distance
//! is known in closed form rather than sampled on a grid. The reference derives
//! three solids: a regular hexagonal prism and a regular octagonal prism, both
//! extruded along `z` with their cross-section folded through the polygon
//! mirror planes so a single wedge represents the whole shape, and a `link` (a
//! torus stretched by a straight mid-section, the exact distance to a chain
//! link). [`GpuSdfPrism3d`] is the on-device twin: each thread reads one point
//! plus every shape's parameters and writes all three signed distances,
//! reproducing the reference closed forms with only `sqrt`, `abs`, `min`,
//! `max`, `clamp`, products and quotients.
//!
//! # What is twinned
//!
//! Each thread reads one [`SdfPrism3dQuery`] — a query `point` plus the hexagon
//! (`hex_apothem`, `hex_half_depth`), octagon (`octagon_radius`,
//! `octagon_half_depth`) and link (`link_half_length`, `link_r1`, `link_r2`)
//! parameters — and writes one [`SdfPrism3dResult`] holding the three signed
//! distances `hex_prism_value`, `octagon_prism_value` and `link_value`.
//!
//! The hexagonal-prism kernel folds the absolute-value point once across the
//! sextant boundary with the baked constant
//! `k = (-cos 30 degrees, sin 30 degrees, 1 / sqrt 3)`, clamps onto the top
//! flat, measures the planar face distance signed by the interior test
//! `p.y - apothem < 0`, and combines it with the `z` slab via the exact
//! interior/exterior split. The octagonal-prism kernel is its eight-sided
//! companion: two reflections with the baked constant
//! `k = (-cos(pi/8), sin(pi/8), tan(pi/8))` fold the quadrant down to one wedge,
//! then the same flat-measure/slab combine applies. The link kernel offsets the
//! `y` coordinate by the straight half-length (clamped at zero), reduces to the
//! ring circle of radius `link_r1` in the folded `xy` plane, then subtracts the
//! tube radius `link_r2`.
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
//! Every distance threads through `sqrt`, products and quotients, so the `CPU`
//! and `GPU` are not bit-exact: a `GPU` `sqrt` or divide may land a few units
//! in the last place from the scalar reference. The parity test asserts each
//! distance within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, loose enough to
//! admit a legal last-place difference yet tight enough to catch a wrong port.
//! Fixtures stay a safe margin away from the prism's interior-test boundary
//! `p.y == apothem` (likewise `p.y == radius`), where the hard `sign` select
//! could pick different sides on the `CPU` and `GPU`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `abs`, `min`,
//! `max`, `clamp`, `+ - * /` and unsigned index arithmetic — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `round` and
//! no `ceil`, and no `f64`/`u64`/`u16`/`i64`/`i16`. It runs unmodified on
//! `Metal`, `Vulkan` and `DX12`.
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

/// The portable core-`WGSL` prism/link signed-distance kernel, embedded inline
/// so the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`hex_prism`](prism_render_architecture::ray_scene::sdf_primitives::hex_prism),
/// [`octagon_prism`](prism_render_architecture::ray_scene::sdf_primitives::octagon_prism)
/// and [`link`](prism_render_architecture::ray_scene::sdf_primitives::link)
/// closed forms; see the module documentation for the algorithm.
const SDF_PRISM3D_WGSL: &str = r#"
// Prism/link signed-distance twin: one thread computes one query point's
// hexagonal-prism, octagonal-prism and link signed distances, mirroring the CPU
// golden `ray_scene::sdf_primitives::{hex_prism, octagon_prism, link}` with only
// sqrt, abs, min, max, clamp, products and quotients. The domain/CSG operators
// and the ray-marcher stay on the host.
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
    // Hexagonal prism: inradius (apothem) and half-depth along z.
    hex_apothem: f32,
    hex_half_depth: f32,
    // Octagonal prism: inradius (apothem) and half-depth along z.
    octagon_radius: f32,
    octagon_half_depth: f32,
    // Link: straight half-length, ring radius and tube radius.
    link_half_length: f32,
    link_r1: f32,
    link_r2: f32,
    pad0: f32,
    pad1: f32,
}

struct Distances {
    // Hexagonal-prism signed distance.
    hex_prism_value: f32,
    // Octagonal-prism signed distance.
    octagon_prism_value: f32,
    // Link signed distance.
    link_value: f32,
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Distances>;

// Euclidean length of a 2-vector, matching the golden `length2` operation order
// `sqrt(x*x + y*y)` exactly.
fn len2(v: vec2<f32>) -> f32 {
    return sqrt(v.x * v.x + v.y * v.y);
}

// Regular hexagonal prism extruded along z: fold into one sextant, clamp onto
// the top flat, interior/exterior split against the folded face and the z slab.
fn hex_prism_sd(p_in: vec3<f32>, apothem: f32, half_depth: f32) -> f32 {
    // k = (-cos(30 degrees), sin(30 degrees), 1 / sqrt(3)); baked constants.
    let k = vec3<f32>(-0.8660254, 0.5, 0.57735);
    var p = vec3<f32>(abs(p_in.x), abs(p_in.y), abs(p_in.z));
    let fold = 2.0 * min(k.x * p.x + k.y * p.y, 0.0);
    p.x = p.x - fold * k.x;
    p.y = p.y - fold * k.y;
    let clamped_x = clamp(p.x, -k.z * apothem, k.z * apothem);
    let face = vec2<f32>(p.x - clamped_x, p.y - apothem);
    var sign_val = 1.0;
    if (p.y - apothem < 0.0) {
        sign_val = -1.0;
    }
    let d = vec2<f32>(len2(face) * sign_val, p.z - half_depth);
    let inside = min(max(d.x, d.y), 0.0);
    let outside = len2(vec2<f32>(max(d.x, 0.0), max(d.y, 0.0)));
    return inside + outside;
}

// Regular octagonal prism extruded along z: two reflections fold the quadrant
// to one wedge, then the same flat-measure/slab combine as the hexagon.
fn octagon_prism_sd(p_in: vec3<f32>, radius: f32, half_depth: f32) -> f32 {
    // (-cos(22.5 degrees), sin(22.5 degrees), tan(22.5 degrees) = sqrt(2) - 1).
    let k = vec3<f32>(-0.9238795, 0.3826834, 0.41421356);
    var p = vec3<f32>(abs(p_in.x), abs(p_in.y), abs(p_in.z));
    let fold0 = 2.0 * min(k.x * p.x + k.y * p.y, 0.0);
    p.x = p.x - fold0 * k.x;
    p.y = p.y - fold0 * k.y;
    let fold1 = 2.0 * min(-k.x * p.x + k.y * p.y, 0.0);
    p.x = p.x - fold1 * (-k.x);
    p.y = p.y - fold1 * k.y;
    let clamped_x = clamp(p.x, -k.z * radius, k.z * radius);
    let face = vec2<f32>(p.x - clamped_x, p.y - radius);
    var sign_val = 1.0;
    if (p.y - radius < 0.0) {
        sign_val = -1.0;
    }
    let d = vec2<f32>(len2(face) * sign_val, p.z - half_depth);
    let inside = min(max(d.x, d.y), 0.0);
    let outside = len2(vec2<f32>(max(d.x, 0.0), max(d.y, 0.0)));
    return inside + outside;
}

// Link (stretched torus): offset y by the straight half-length clamped at zero,
// reduce to the ring circle of radius r1, then subtract the tube radius r2.
fn link_sd(p: vec3<f32>, half_length: f32, r1: f32, r2: f32) -> f32 {
    let qy = max(abs(p.y) - half_length, 0.0);
    let planar = len2(vec2<f32>(p.x, qy)) - r1;
    return len2(vec2<f32>(planar, p.z)) - r2;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let point = vec3<f32>(q.px, q.py, q.pz);

    var out: Distances;
    out.hex_prism_value = hex_prism_sd(point, q.hex_apothem, q.hex_half_depth);
    out.octagon_prism_value = octagon_prism_sd(point, q.octagon_radius, q.octagon_half_depth);
    out.link_value = link_sd(point, q.link_half_length, q.link_r1, q.link_r2);
    out.pad0 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SDF_PRISM3D_WGSL`].
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
/// the point components plus every shape's parameters and two pad words to a
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
    /// Hexagonal-prism apothem.
    hex_apothem: f32,
    /// Hexagonal-prism half-depth along `z`.
    hex_half_depth: f32,
    /// Octagonal-prism apothem.
    octagon_radius: f32,
    /// Octagonal-prism half-depth along `z`.
    octagon_half_depth: f32,
    /// Link straight half-length.
    link_half_length: f32,
    /// Link ring radius.
    link_r1: f32,
    /// Link tube radius.
    link_r2: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Distances`
/// struct: the three signed distances plus one pad word to a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Hexagonal-prism signed distance.
    hex_prism_value: f32,
    /// Octagonal-prism signed distance.
    octagon_prism_value: f32,
    /// Link signed distance.
    link_value: f32,
    /// Padding word.
    pad0: f32,
}

/// One query for the prism/link signed-distance twin: the query `point` plus
/// the hexagonal-prism, octagonal-prism and link shape parameters.
///
/// `point` is the evaluation position; `hex_apothem`/`hex_half_depth` are the
/// [`hex_prism`](prism_render_architecture::ray_scene::sdf_primitives::hex_prism)
/// inradius and `z` half-depth; `octagon_radius`/`octagon_half_depth` are the
/// [`octagon_prism`](prism_render_architecture::ray_scene::sdf_primitives::octagon_prism)
/// inradius and `z` half-depth; `link_half_length`/`link_r1`/`link_r2` are the
/// [`link`](prism_render_architecture::ray_scene::sdf_primitives::link) straight
/// half-length, ring radius and tube radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfPrism3dQuery {
    /// Query point `[x, y, z]`.
    pub point: [f32; 3],
    /// Hexagonal-prism apothem (centre-to-flat-face distance).
    pub hex_apothem: f32,
    /// Hexagonal-prism half-depth along `z`.
    pub hex_half_depth: f32,
    /// Octagonal-prism apothem (centre-to-flat-face distance).
    pub octagon_radius: f32,
    /// Octagonal-prism half-depth along `z`.
    pub octagon_half_depth: f32,
    /// Link straight half-length.
    pub link_half_length: f32,
    /// Link ring radius.
    pub link_r1: f32,
    /// Link tube radius.
    pub link_r2: f32,
}

impl SdfPrism3dQuery {
    /// Builds a query from the point and every shape's parameters.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the three golden signatures packed into one query slot"
    )]
    pub const fn new(
        point: [f32; 3],
        hex_apothem: f32,
        hex_half_depth: f32,
        octagon_radius: f32,
        octagon_half_depth: f32,
        link_half_length: f32,
        link_r1: f32,
        link_r2: f32,
    ) -> SdfPrism3dQuery {
        SdfPrism3dQuery {
            point,
            hex_apothem,
            hex_half_depth,
            octagon_radius,
            octagon_half_depth,
            link_half_length,
            link_r1,
            link_r2,
        }
    }
}

/// One resolved query of the prism/link signed-distance twin: the hexagonal-
/// prism, octagonal-prism and link signed distances at the query point.
///
/// `hex_prism_value` is
/// [`hex_prism`](prism_render_architecture::ray_scene::sdf_primitives::hex_prism);
/// `octagon_prism_value` is
/// [`octagon_prism`](prism_render_architecture::ray_scene::sdf_primitives::octagon_prism);
/// `link_value` is
/// [`link`](prism_render_architecture::ray_scene::sdf_primitives::link). Each is
/// negative inside the solid, positive outside, zero on the surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfPrism3dResult {
    /// Hexagonal-prism signed distance.
    pub hex_prism_value: f32,
    /// Octagonal-prism signed distance.
    pub octagon_prism_value: f32,
    /// Link signed distance.
    pub link_value: f32,
}

/// Encodes one [`SdfPrism3dQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SdfPrism3dQuery) -> GpuQuery {
    GpuQuery {
        px: q.point[0],
        py: q.point[1],
        pz: q.point[2],
        hex_apothem: q.hex_apothem,
        hex_half_depth: q.hex_half_depth,
        octagon_radius: q.octagon_radius,
        octagon_half_depth: q.octagon_half_depth,
        link_half_length: q.link_half_length,
        link_r1: q.link_r1,
        link_r2: q.link_r2,
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SdfPrism3dResult`].
fn decode_result(raw: &GpuResult) -> SdfPrism3dResult {
    SdfPrism3dResult {
        hex_prism_value: raw.hex_prism_value,
        octagon_prism_value: raw.octagon_prism_value,
        link_value: raw.link_value,
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

/// A compiled, reusable prism/link signed-distance compute pipeline, twinning
/// the `CPU` golden
/// [`hex_prism`](prism_render_architecture::ray_scene::sdf_primitives::hex_prism),
/// [`octagon_prism`](prism_render_architecture::ray_scene::sdf_primitives::octagon_prism)
/// and [`link`](prism_render_architecture::ray_scene::sdf_primitives::link).
pub struct GpuSdfPrism3d {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfPrism3d {
    /// Compiles the prism/link signed-distance kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdfPrism3d {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sdf_prism3d"),
            source: ShaderSource::Wgsl(SDF_PRISM3D_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sdf_prism3d_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sdf_prism3d_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sdf_prism3d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfPrism3d {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`SdfPrism3dResult`] per
    /// input, in order.
    ///
    /// The signed distances match the reference to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[SdfPrism3dQuery]) -> Vec<SdfPrism3dResult> {
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
            label: Some("prism_volumetric_sdf_prism3d_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_prism3d_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sdf_prism3d_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sdf_prism3d_bind_group"),
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
            label: Some("prism_volumetric_sdf_prism3d_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sdf_prism3d_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sdf_prism3d_pass"),
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
