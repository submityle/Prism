//! `wgpu` compute twin of the per-triangle cloth aerodynamic force from the
//! cloth wind module
//! ([`wind`](prism_render_architecture::cloth::wind)).
//!
//! Wind acts on a cloth mesh *per triangle*, not per particle, because lift and
//! drag depend on how each face is oriented relative to the airflow: a face
//! broadside to the wind catches far more force than one edge-on to it, and
//! only a triangle carries the surface normal that distinction needs. The
//! golden
//! [`triangle_wind_force`](prism_render_architecture::cloth::wind::triangle_wind_force)
//! decomposes the relative wind (ambient wind minus the face's own velocity)
//! into a component along the unit face normal (scaled by `drag`) and the
//! remaining in-plane component (scaled by `lift`), then multiplies the sum by
//! a pressure scale that is either the triangle area (linear model) or the
//! area times the dynamic pressure `0.5 * air_density * |relative|` (quadratic
//! model). A degenerate triangle contributes no force.
//!
//! The force math is pure arithmetic plus `sqrt`, so [`GpuClothWindForce`]
//! ports it exactly: one thread owns one triangle, reconstructs the three
//! positions, three velocities and the wind from flat scalar fields, reproduces
//! the drag/lift decomposition and the pressure scale, and writes the resulting
//! force vector. A passing real-device parity test is direct evidence the
//! ported kernel reproduces the reference force, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! For one query carrying the triangle vertices `p0`/`p1`/`p2`, their
//! velocities `v0`/`v1`/`v2`, the ambient `wind`, and the `drag`/`lift`/
//! `air_density` coefficients, the kernel reproduces `triangle_wind_force`
//! (which delegates to
//! [`triangle_aero_force`](prism_physics_core::soft::aero::triangle_aero_force)):
//! - the edge cross product `(p1 - p0) × (p2 - p0)` and its squared length,
//!   with an early zero-force exit when that squared length is at most the
//!   degeneracy threshold `1e-12`;
//! - the triangle `area = 0.5 * sqrt(cross_len_sq)` and the unit face `normal`;
//! - the relative wind `wind - (v0 + v1 + v2) / 3`, split into a normal
//!   component (times `drag`) and the in-plane remainder (times `lift`); and
//! - the pressure scale: the bare `area` when `air_density <= 0`, otherwise
//!   `area * (0.5 * air_density * sqrt(dot(relative, relative)))`.
//!
//! # Sanitize
//!
//! The golden sanitizes `drag`, `lift` and `air_density` to be non-negative
//! before the math. For the finite inputs this twin is fixtured against, that
//! clamp is reproduced in the kernel with `max(coefficient, 0.0)`, matching the
//! reference for every finite coefficient.
//!
//! # Correctness model
//!
//! The force is a continuous `f32` quantity, so parity is asserted with the
//! crate's tolerance (`abs <= 1e-4` or `rel <= 1e-3`, floor `1e-6`). The only
//! branch is the degeneracy early-exit and the linear/quadratic pressure
//! split, both driven by well-conditioned inputs the fixtures keep away from
//! the thresholds.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+ - * /`, `dot`,
//! `cross`, `sqrt`, `max` — with no transcendental call and no `64`-bit
//! integers or floats. No optional device feature is required, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::cloth::wind`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` cloth wind-force kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `wind_force`
/// resolves one triangle per thread.
const CLOTH_WIND_FORCE_WGSL: &str = r#"
// Cloth wind-force twin: one thread owns one triangle. It reconstructs the
// three vertex positions, three vertex velocities and the ambient wind from
// flat f32 fields, then reproduces the golden triangle_wind_force /
// triangle_aero_force: the drag/lift decomposition of the relative wind scaled
// by the triangle area and (optionally) the dynamic pressure. Pure arithmetic
// plus sqrt; no transcendental.
//
// Provenance: 孪生自本仓 prism_render_architecture::cloth::wind；无第三方引擎源码
// 或衍生代码。

struct Params {
    // Number of triangles in the batch; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Vertex positions p0, p1, p2 as flat scalars.
    p0x: f32, p0y: f32, p0z: f32,
    p1x: f32, p1y: f32, p1z: f32,
    p2x: f32, p2y: f32, p2z: f32,
    // Vertex velocities v0, v1, v2 as flat scalars.
    v0x: f32, v0y: f32, v0z: f32,
    v1x: f32, v1y: f32, v1z: f32,
    v2x: f32, v2y: f32, v2z: f32,
    // Ambient wind velocity.
    windx: f32, windy: f32, windz: f32,
    // Aerodynamic coefficients; clamped non-negative in-kernel.
    drag: f32,
    lift: f32,
    air_density: f32,
}

struct Outcome {
    // Resulting aerodynamic force on the triangle.
    fx: f32,
    fy: f32,
    fz: f32,
    pad: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Outcome>;

@compute @workgroup_size(64)
fn wind_force(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.count) {
        return;
    }
    let q = queries[gid.x];

    let p0 = vec3<f32>(q.p0x, q.p0y, q.p0z);
    let p1 = vec3<f32>(q.p1x, q.p1y, q.p1z);
    let p2 = vec3<f32>(q.p2x, q.p2y, q.p2z);
    let v0 = vec3<f32>(q.v0x, q.v0y, q.v0z);
    let v1 = vec3<f32>(q.v1x, q.v1y, q.v1z);
    let v2 = vec3<f32>(q.v2x, q.v2y, q.v2z);
    let wind = vec3<f32>(q.windx, q.windy, q.windz);

    // Sanitize: the golden clamps the coefficients non-negative before use.
    let drag = max(q.drag, 0.0);
    let lift = max(q.lift, 0.0);
    let air_density = max(q.air_density, 0.0);

    let cross_v = cross(p1 - p0, p2 - p0);
    let cross_len_sq = dot(cross_v, cross_v);

    // Degenerate triangle: near-zero normal contributes no force.
    if (cross_len_sq <= 1e-12) {
        results[gid.x] = Outcome(0.0, 0.0, 0.0, 0.0);
        return;
    }

    let root = sqrt(cross_len_sq);
    let area = 0.5 * root;
    // Unit face normal: cross / |cross|. Safe since cross_len_sq > 1e-12.
    let normal = cross_v * (1.0 / root);

    let face_velocity = (v0 + v1 + v2) * (1.0 / 3.0);
    let relative = wind - face_velocity;

    let normal_component = normal * dot(relative, normal);
    let tangent_component = relative - normal_component;

    // Directional force per unit pressure: drag along the normal, lift in-plane.
    let directional = normal_component * drag + tangent_component * lift;

    // Pressure scale. Linear model (density <= 0) uses the area directly; the
    // quadratic model (density > 0) adds the dynamic pressure factor
    // 0.5 * air_density * |relative|.
    var pressure: f32 = area;
    if (air_density > 0.0) {
        pressure = area * (0.5 * air_density * sqrt(dot(relative, relative)));
    }

    let force = directional * pressure;
    results[gid.x] = Outcome(force.x, force.y, force.z, 0.0);
}
"#;

/// Uniform parameters for the dispatch: the triangle count and three pad words,
/// filling a `16`-byte uniform struct matching `Params` in
/// [`CLOTH_WIND_FORCE_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid triangles in the batch.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one triangle query, matching the `WGSL` `Query`
/// struct. All fields are flat `f32` scalars so the struct needs no interior
/// padding (`24` words, a `16`-byte multiple).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// `p0` x component.
    p0x: f32,
    /// `p0` y component.
    p0y: f32,
    /// `p0` z component.
    p0z: f32,
    /// `p1` x component.
    p1x: f32,
    /// `p1` y component.
    p1y: f32,
    /// `p1` z component.
    p1z: f32,
    /// `p2` x component.
    p2x: f32,
    /// `p2` y component.
    p2y: f32,
    /// `p2` z component.
    p2z: f32,
    /// `v0` x component.
    v0x: f32,
    /// `v0` y component.
    v0y: f32,
    /// `v0` z component.
    v0z: f32,
    /// `v1` x component.
    v1x: f32,
    /// `v1` y component.
    v1y: f32,
    /// `v1` z component.
    v1z: f32,
    /// `v2` x component.
    v2x: f32,
    /// `v2` y component.
    v2y: f32,
    /// `v2` z component.
    v2z: f32,
    /// `wind` x component.
    windx: f32,
    /// `wind` y component.
    windy: f32,
    /// `wind` z component.
    windz: f32,
    /// Normal-direction drag coefficient.
    drag: f32,
    /// In-plane lift coefficient.
    lift: f32,
    /// Fluid (air) density selecting the linear/quadratic model.
    air_density: f32,
}

/// `repr(C)` `std430` layout of one resolved force, matching the `WGSL`
/// `Outcome` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuOutcome {
    /// Force x component.
    fx: f32,
    /// Force y component.
    fy: f32,
    /// Force z component.
    fz: f32,
    /// Padding word rounding the struct to a `16`-byte multiple.
    pad: f32,
}

/// One cloth wind-force query for the twin: the triangle vertices, their
/// velocities, the ambient wind, and the aerodynamic coefficients.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClothWindForceQuery {
    /// Vertex position `p0` as `[x, y, z]`.
    pub p0: [f32; 3],
    /// Vertex position `p1` as `[x, y, z]`.
    pub p1: [f32; 3],
    /// Vertex position `p2` as `[x, y, z]`.
    pub p2: [f32; 3],
    /// Vertex velocity `v0` as `[x, y, z]`.
    pub v0: [f32; 3],
    /// Vertex velocity `v1` as `[x, y, z]`.
    pub v1: [f32; 3],
    /// Vertex velocity `v2` as `[x, y, z]`.
    pub v2: [f32; 3],
    /// Ambient wind velocity as `[x, y, z]`.
    pub wind: [f32; 3],
    /// Normal-direction drag coefficient (clamped non-negative in-kernel).
    pub drag: f32,
    /// In-plane lift coefficient (clamped non-negative in-kernel).
    pub lift: f32,
    /// Fluid (air) density: `<= 0` selects the linear model, `> 0` the
    /// quadratic model.
    pub air_density: f32,
}

impl ClothWindForceQuery {
    /// Builds a query from the triangle geometry, velocities, wind, and
    /// coefficients.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "a triangle force is intrinsically defined by its three positions, \
                  three velocities, the wind, and the three coefficients; grouping \
                  them into structs here would obscure the plain mapping to the golden"
    )]
    pub fn new(
        p0: [f32; 3],
        p1: [f32; 3],
        p2: [f32; 3],
        v0: [f32; 3],
        v1: [f32; 3],
        v2: [f32; 3],
        wind: [f32; 3],
        drag: f32,
        lift: f32,
        air_density: f32,
    ) -> ClothWindForceQuery {
        ClothWindForceQuery {
            p0,
            p1,
            p2,
            v0,
            v1,
            v2,
            wind,
            drag,
            lift,
            air_density,
        }
    }
}

/// One resolved cloth wind force: the aerodynamic force vector on the triangle,
/// mirroring the reference
/// [`triangle_wind_force`](prism_render_architecture::cloth::wind::triangle_wind_force).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClothWindForceResult {
    /// Resulting aerodynamic force on the triangle as `[x, y, z]`.
    pub force: [f32; 3],
}

/// Encodes one [`ClothWindForceQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &ClothWindForceQuery) -> GpuQuery {
    GpuQuery {
        p0x: q.p0[0],
        p0y: q.p0[1],
        p0z: q.p0[2],
        p1x: q.p1[0],
        p1y: q.p1[1],
        p1z: q.p1[2],
        p2x: q.p2[0],
        p2y: q.p2[1],
        p2z: q.p2[2],
        v0x: q.v0[0],
        v0y: q.v0[1],
        v0z: q.v0[2],
        v1x: q.v1[0],
        v1y: q.v1[1],
        v1z: q.v1[2],
        v2x: q.v2[0],
        v2y: q.v2[1],
        v2z: q.v2[2],
        windx: q.wind[0],
        windy: q.wind[1],
        windz: q.wind[2],
        drag: q.drag,
        lift: q.lift,
        air_density: q.air_density,
    }
}

/// Decodes one `std430` [`GpuOutcome`] into a [`ClothWindForceResult`].
fn decode_outcome(o: &GpuOutcome) -> ClothWindForceResult {
    ClothWindForceResult {
        force: [o.fx, o.fy, o.fz],
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

/// A compiled, reusable cloth wind-force compute pipeline, twinning the `CPU`
/// golden
/// [`triangle_wind_force`](prism_render_architecture::cloth::wind::triangle_wind_force)
/// from [`wind`](prism_render_architecture::cloth::wind).
pub struct GpuClothWindForce {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuClothWindForce {
    /// Compiles the wind-force kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothWindForce {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_cloth_wind_force"),
            source: ShaderSource::Wgsl(CLOTH_WIND_FORCE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_cloth_wind_force_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_cloth_wind_force_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_cloth_wind_force_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("wind_force"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuClothWindForce {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every triangle in `queries` and returns one
    /// [`ClothWindForceResult`] per input, in order.
    ///
    /// An empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[ClothWindForceQuery],
    ) -> Vec<ClothWindForceResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cloth_wind_force_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let result_bytes = (count * size_of::<GpuOutcome>()) as u64;
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cloth_wind_force_results"),
            size: result_bytes,
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
            label: Some("prism_volumetric_cloth_wind_force_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_cloth_wind_force_bind_group"),
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

        let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_cloth_wind_force_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_cloth_wind_force_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per triangle, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cloth_wind_force_stage"),
            size: result_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&results_buf, 0, &stage, 0, result_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let outcomes = bytemuck::cast_slice::<u8, GpuOutcome>(&view).to_vec();
        drop(view);
        stage.unmap();

        outcomes.iter().map(decode_outcome).collect()
    }
}
