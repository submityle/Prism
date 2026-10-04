//! `wgpu` compute twin of the mesh-diameter radius closed form, from the `CPU`
//! golden `prism_physics_core::collider::diameter`'s `MeshDiameter::radius`.
//!
//! The diameter of a shape is the greatest distance between any two of its
//! points; half the diameter, `radius = diameter * 0.5`, is a lower bound on
//! any enclosing-sphere radius. This module ports that single stateless,
//! no-`RNG`, branch-free closed form onto the device: one compute thread
//! resolves one diameter, so a passing real-device parity test is direct
//! evidence the kernel reproduces the exact half-scale multiply, not merely
//! that the shader compiles.
//!
//! # What is twinned
//!
//! Each query is one diameter scalar. The kernel reproduces the reference
//! closed form `radius = diameter * 0.5`, evaluated in exactly the golden
//! operator form (a single multiply by one half).
//!
//! There is no division and no branch: the computation is pure multiplication,
//! so the kernel provably terminates and every input is valid.
//!
//! # Correctness model
//!
//! The scalar is a single multiply, so `CPU` and `GPU` are not required to be
//! bit-exact (though a half-scale is in practice exact). The parity test
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`,
//! `REL_FLOOR = 1e-6`) on the `radius`. The kernel has no comparisons at all,
//! so no bare float equality and no fast-math `NaN` sentinel is involved.
//!
//! # Degenerate inputs
//!
//! There are no degenerate inputs: every diameter yields a well-defined
//! radius, including `0` (radius `0`) and negative values (scaled as-is by one
//! half). An empty query batch short-circuits on the host with no dispatch,
//! since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `*` on `f32` and
//! unsigned index arithmetic — with no `sin`, `cos`, `tan`, `exp`, `log`,
//! `pow`, no `round`, no float modulo and no optional device feature, so it
//! runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::diameter::MeshDiameter::radius`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` diameter-radius kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `MeshDiameter::radius`; see the module documentation for the
/// closed form.
const DIAMETER_RADIUS_WGSL: &str = r#"
// Diameter-radius twin: one thread per query halves the diameter the golden
// MeshDiameter::radius scales by one half. It mirrors the CPU golden exactly,
// uses only the portable core-WGSL subset (f32 multiply plus unsigned index
// math), takes no optional feature, and has no loop and no branch, so the
// kernel provably terminates. There is no division and no comparison, so no
// float equality or fast-math sentinel is involved.
//
// Provenance: 孪生自本仓
// prism_physics_core::collider::diameter::MeshDiameter::radius；
// 无第三方引擎源码或衍生代码。

struct Params {
    // Number of diameters in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // The diameter (farthest-pair distance) to halve.
    diameter: f32,
    // Padding words to a 16-byte-friendly stride.
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct Result {
    // Half the diameter.
    radius: f32,
    // Padding word to an 8-byte stride.
    pad0: u32,
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

    var out: Result;
    // Golden closed form: radius = diameter * 0.5.
    out.radius = q.diameter * 0.5;
    out.pad0 = 0u;
    results[idx] = out;
}
"#;

/// `repr(C)` `std430` dispatch parameters: the query count plus padding to a
/// 16-byte uniform block.
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
/// The diameter is padded to `4` `f32` words (`16` bytes), aligned to `4`, so
/// the host and device agree on the array stride byte for byte.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    diameter: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the radius and one trailing pad word — `2` words (`8` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    radius: f32,
    pad0: u32,
}

/// One diameter-radius query: the diameter scalar to halve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiameterRadiusQuery {
    /// The diameter (farthest-pair distance).
    pub diameter: f32,
}

impl DiameterRadiusQuery {
    /// Builds a query from the diameter scalar.
    #[must_use]
    pub fn new(diameter: f32) -> DiameterRadiusQuery {
        DiameterRadiusQuery { diameter }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `MeshDiameter::radius` output for that diameter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiameterRadiusResult {
    /// Half the diameter: `diameter * 0.5`.
    pub radius: f32,
}

/// Encodes one [`DiameterRadiusQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &DiameterRadiusQuery) -> GpuQuery {
    GpuQuery {
        diameter: q.diameter,
        pad0: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`DiameterRadiusResult`].
fn decode_result(raw: &GpuResult) -> DiameterRadiusResult {
    DiameterRadiusResult { radius: raw.radius }
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

/// A compiled, reusable diameter-radius compute pipeline, twinning the `CPU`
/// golden `MeshDiameter::radius`.
pub struct GpuDiameterRadius {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuDiameterRadius {
    /// Compiles the diameter-radius kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDiameterRadius {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_diameter_radius"),
            source: ShaderSource::Wgsl(DIAMETER_RADIUS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_diameter_radius_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_diameter_radius_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_diameter_radius_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuDiameterRadius {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`DiameterRadiusResult`]
    /// per input, in order.
    ///
    /// Each `radius` matches the reference to the module's tolerance. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[DiameterRadiusQuery],
    ) -> Vec<DiameterRadiusResult> {
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
            label: Some("prism_volumetric_diameter_radius_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_diameter_radius_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_diameter_radius_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_diameter_radius_bind_group"),
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
            label: Some("prism_volumetric_diameter_radius_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_diameter_radius_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_diameter_radius_pass"),
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
