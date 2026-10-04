//! `wgpu` compute twin of the oriented-bounding-box volume from the `CPU`
//! golden `prism_physics_core::collider::obb::Obb::volume`.
//!
//! An `OBB` is a box with three half-extents, so its enclosed volume is the
//! product of its three full side lengths. With the box given as a half-extent
//! triple this is the stateless kernel that computes it:
//!
//! ```text
//! volume = 8.0 * he.x * he.y * he.z
//! ```
//!
//! # What is twinned
//!
//! The single stateless, no-`RNG`, no-branch body of `Obb::volume`. There is no
//! degenerate case: the formula is a pure product, so `valid` is always `1`.
//!
//! # Correctness model
//!
//! The volume threads through two multiplies, so `CPU` and `GPU` are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate. The parity test asserts a tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on the continuous `volume` channel;
//! the discrete `valid` flag is compared exactly. The relative term carries a
//! large volume while the floor keeps a tiny volume honest.
//!
//! # Degenerate inputs
//!
//! None. Every half-extent triple yields a well-defined product and `valid =
//! 1`. An empty query batch short-circuits on the host with no dispatch, since
//! a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — a single chain of
//! multiplies — with no `pow`, no `sin`, `cos`, `exp`, `log`, no `round`, no
//! `f32` remainder, no `u64`/`i64`, no `f64` and no optional device feature, so
//! it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::obb::Obb::volume`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` `OBB` volume kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `evaluate` mirrors the
/// `CPU` golden `Obb::volume`; see the module documentation for the algorithm.
const OBB_VOLUME_WGSL: &str = r#"
// OBB volume twin: one thread per query multiplies the three full side lengths
// of a box given by its half-extents,
//   volume = 8.0 * he.x * he.y * he.z
// It uses only the portable core-WGSL subset (a chain of multiplies) with no
// u64/i64/f64, no pow and no transcendental, so it runs unmodified on Metal,
// Vulkan and DX12. The formula has no degenerate branch, so valid is always 1.
//
// Provenance: 孪生自本仓 prism_physics_core::collider::obb::Obb::volume；无第三方引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query: the three half-extents of the box, flattened to scalar lanes so
// the std430 layout never trips a 16-byte vector-alignment rule.
struct Query {
    hx: f32,
    hy: f32,
    hz: f32,
    pad: f32,
}

// One result: the enclosed volume and a valid flag that is always 1.
struct Res {
    volume: f32,
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Res;
    out.volume = 8.0 * q.hx * q.hy * q.hz;
    out.valid = 1u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`OBB_VOLUME_WGSL`].
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
/// The half-extents are flattened to scalar lanes plus one pad word so the
/// layout never trips a `16`-byte vector-alignment rule.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    hx: f32,
    hy: f32,
    hz: f32,
    pad: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Res` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    volume: f32,
    valid: u32,
}

/// One query for the `OBB` volume twin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObbVolumeQuery {
    /// The three half-extents of the box.
    pub half_extents: [f32; 3],
}

impl ObbVolumeQuery {
    /// Builds a query from the box half-extents.
    #[must_use]
    pub fn new(half_extents: [f32; 3]) -> ObbVolumeQuery {
        ObbVolumeQuery { half_extents }
    }
}

/// One resolved `OBB` volume.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObbVolumeResult {
    /// Enclosed volume `8.0 * he.x * he.y * he.z`.
    pub volume: f32,
    /// Always `1`; the formula has no degenerate branch.
    pub valid: u32,
}

/// Encodes one [`ObbVolumeQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &ObbVolumeQuery) -> GpuQuery {
    GpuQuery {
        hx: q.half_extents[0],
        hy: q.half_extents[1],
        hz: q.half_extents[2],
        pad: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`ObbVolumeResult`].
fn decode_result(raw: &GpuResult) -> ObbVolumeResult {
    ObbVolumeResult {
        volume: raw.volume,
        valid: raw.valid,
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

/// A compiled, reusable `OBB` volume compute pipeline, twinning the `CPU`
/// golden `Obb::volume`.
pub struct GpuObbVolume {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuObbVolume {
    /// Compiles the `OBB` volume kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuObbVolume {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_obb_volume_module"),
            source: ShaderSource::Wgsl(OBB_VOLUME_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_obb_volume_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_obb_volume_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_obb_volume_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuObbVolume {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`ObbVolumeResult`] per
    /// input, in order.
    ///
    /// The continuous `volume` channel matches the reference to within the
    /// tolerance documented on this module; the `valid` flag matches exactly. An
    /// empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[ObbVolumeQuery]) -> Vec<ObbVolumeResult> {
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
            label: Some("prism_volumetric_obb_volume_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_obb_volume_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_obb_volume_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_obb_volume_bind_group"),
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
            label: Some("prism_volumetric_obb_volume_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_obb_volume_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_obb_volume_pass"),
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
