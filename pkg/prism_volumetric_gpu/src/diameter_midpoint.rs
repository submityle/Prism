//! `wgpu` compute twin of the diameter-midpoint closed form, from the `CPU`
//! golden `prism_physics_core::collider::diameter`'s `MeshDiameter::midpoint`.
//!
//! Given the two endpoints of a mesh's farthest pair, the midpoint is the
//! component-wise average `(endpoint_a + endpoint_b) * 0.5`. The golden
//! `midpoint` computes exactly that; the convex-hull diameter search that
//! produces those endpoints is out of scope here. This module ports the
//! stateless, no-`RNG`, branch-free average onto the device: one compute thread
//! resolves one endpoint pair, so a passing real-device parity test is direct
//! evidence the kernel reproduces the exact average, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! Each query is one endpoint pair `a` and `b`. The kernel reproduces the
//! reference closed form component for component:
//!
//! * `mid.x = (a.x + b.x) * 0.5`;
//! * `mid.y = (a.y + b.y) * 0.5`;
//! * `mid.z = (a.z + b.z) * 0.5`.
//!
//! There is no division by a variable and no branch: the computation is a pure
//! add-then-halve, so the kernel provably terminates and every pair is valid.
//!
//! # Correctness model
//!
//! Each component is one add and one multiply, so `CPU` and `GPU` are not
//! required to be bit-exact. The parity test asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on each of the
//! three components. The kernel has no comparisons at all, so no bare float
//! equality and no fast-math `NaN` sentinel is involved.
//!
//! # Degenerate inputs
//!
//! There are no degenerate inputs: every endpoint pair yields a well-defined
//! midpoint, so no validity flag is carried. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+` and `*` on
//! `vec3<f32>` plus unsigned index arithmetic — with no `sin`, `cos`, `tan`,
//! `exp`, `log`, `pow`, no `round`, no float modulo and no optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::diameter::MeshDiameter::midpoint`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` diameter-midpoint kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `MeshDiameter::midpoint`; see the module documentation for
/// the closed form.
const DIAMETER_MIDPOINT_WGSL: &str = r#"
// Diameter-midpoint twin: one thread per endpoint pair reproduces the midpoint
// the golden MeshDiameter::midpoint forms as (endpoint_a + endpoint_b) * 0.5.
// It mirrors the CPU golden operation for operation, uses only the portable
// core-WGSL subset (vec3 add and scalar multiply plus unsigned index math),
// takes no optional feature, and has no loop and no branch, so the kernel
// provably terminates. There is no variable division, so no float equality or
// fast-math sentinel is involved.
//
// Provenance: 孪生自本仓
// prism_physics_core::collider::diameter::MeshDiameter::midpoint；
// 无第三方引擎源码或衍生代码。

struct Params {
    // Number of endpoint pairs in the storage arrays; threads past this
    // short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // First endpoint of the farthest pair.
    a: vec3<f32>,
    // Padding word to a 16-byte-aligned slot.
    pad_a: f32,
    // Second endpoint of the farthest pair.
    b: vec3<f32>,
    // Padding word to a 16-byte-aligned slot.
    pad_b: f32,
}

struct Result {
    // The component-wise midpoint of the pair.
    mid: vec3<f32>,
    // Padding word to a 16-byte-aligned slot.
    pad: f32,
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
    out.mid = (q.a + q.b) * 0.5;
    out.pad = 0.0;
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
/// Each endpoint is a flat `vec3<f32>` padded to a `16`-byte slot, so the host
/// and device agree on the `32`-byte stride byte for byte.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    ax: f32,
    ay: f32,
    az: f32,
    pad_a: f32,
    bx: f32,
    by: f32,
    bz: f32,
    pad_b: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the midpoint padded to a `16`-byte slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    mid_x: f32,
    mid_y: f32,
    mid_z: f32,
    pad: f32,
}

/// One diameter-midpoint query: the two endpoints of the farthest pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiameterMidpointQuery {
    /// First endpoint of the farthest pair.
    pub endpoint_a: [f32; 3],
    /// Second endpoint of the farthest pair.
    pub endpoint_b: [f32; 3],
}

impl DiameterMidpointQuery {
    /// Builds a query from the two endpoints.
    #[must_use]
    pub fn new(endpoint_a: [f32; 3], endpoint_b: [f32; 3]) -> DiameterMidpointQuery {
        DiameterMidpointQuery {
            endpoint_a,
            endpoint_b,
        }
    }
}

/// One resolved answer for a single pair: the component-wise midpoint, mirroring
/// the reference `MeshDiameter::midpoint`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiameterMidpointResult {
    /// The midpoint `(endpoint_a + endpoint_b) * 0.5`.
    pub midpoint: [f32; 3],
}

/// Encodes one [`DiameterMidpointQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &DiameterMidpointQuery) -> GpuQuery {
    GpuQuery {
        ax: q.endpoint_a[0],
        ay: q.endpoint_a[1],
        az: q.endpoint_a[2],
        pad_a: 0.0,
        bx: q.endpoint_b[0],
        by: q.endpoint_b[1],
        bz: q.endpoint_b[2],
        pad_b: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`DiameterMidpointResult`].
fn decode_result(raw: &GpuResult) -> DiameterMidpointResult {
    DiameterMidpointResult {
        midpoint: [raw.mid_x, raw.mid_y, raw.mid_z],
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

/// A compiled, reusable diameter-midpoint compute pipeline, twinning the `CPU`
/// golden `MeshDiameter::midpoint`.
pub struct GpuDiameterMidpoint {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuDiameterMidpoint {
    /// Compiles the diameter-midpoint kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDiameterMidpoint {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_diameter_midpoint"),
            source: ShaderSource::Wgsl(DIAMETER_MIDPOINT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_diameter_midpoint_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_diameter_midpoint_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_diameter_midpoint_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuDiameterMidpoint {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every endpoint pair in `queries` and returns one
    /// [`DiameterMidpointResult`] per input, in order.
    ///
    /// Each component matches the reference to within the tolerance documented
    /// on this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[DiameterMidpointQuery],
    ) -> Vec<DiameterMidpointResult> {
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
            label: Some("prism_volumetric_diameter_midpoint_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_diameter_midpoint_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_diameter_midpoint_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_diameter_midpoint_bind_group"),
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
            label: Some("prism_volumetric_diameter_midpoint_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_diameter_midpoint_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_diameter_midpoint_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per endpoint pair, flattened to a 1-D dispatch.
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
