//! `wgpu` compute twin of the tensile-strain primitive from the `CPU` golden
//! `prism_physics_core::soft::damage::strain::tensile_strain`.
//!
//! Both the tearing and the plasticity damage models key off one scalar: the
//! tensile strain of a distance edge, `(length - rest_length) / rest_length`.
//! The golden centralises that ratio so a torn edge and a plastically creeping
//! edge always agree on the same divide guard and the same degeneracy floor.
//! This module ports that stateless, no-`RNG` scalar onto the device: one
//! compute thread resolves one query, so a passing real-device parity test is
//! direct evidence the kernel takes the same `rest_length <= EPS_REST`
//! short-circuit and computes the same signed ratio, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces the reference closed form exactly: when
//! the rest length is at or below the degeneracy floor `EPS_REST = 1e-9` the
//! edge is skipped (`valid = 0`, strain cleared to zero), otherwise it returns
//! the signed ratio `(length - rest_length) / rest_length` (`valid = 1`). A
//! positive value is tension, a negative value is compression. There is no
//! loop: each thread performs one guarded subtraction and division, so the
//! kernel provably terminates.
//!
//! # Correctness model
//!
//! The ratio threads through a subtraction and a division, so `CPU` and `GPU`
//! are not required to be bit-exact. The parity test asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on the
//! continuous `strain`; the discrete `valid` flag is compared exactly. The
//! degeneracy guard uses an ordered `<=` compare against the floor, matching the
//! reference, so no bare float equality is involved.
//!
//! # Degenerate inputs
//!
//! A rest length at or below `EPS_REST` reports `valid = 0` with a cleared
//! strain, exactly as the golden returns `None`. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot
//! be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — an ordered compare,
//! `select`, `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`,
//! `tan`, `exp`, `log`, `pow`, no `round`, no float modulo and no optional
//! device feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::damage::strain::tensile_strain`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` tensile-strain kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `tensile_strain` branch for branch; see the module
/// documentation for the algorithm.
const CLOTH_TENSILE_STRAIN_WGSL: &str = r#"
// Tensile-strain twin: one thread per query reproduces the signed edge strain
// (length - rest_length) / rest_length that tensile_strain derives from a rest
// length and a current endpoint separation. It mirrors the CPU golden branch
// for branch, uses only the portable core-WGSL subset (an ordered compare,
// select, + - * / plus unsigned index math), takes no optional feature, and
// has no loop, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_physics_core::soft::damage::strain::tensile_strain；
// 无第三方引擎源码或衍生代码。

// Degeneracy floor on the rest length, matching the golden EPS_REST.
const EPS_REST: f32 = 1.0e-9;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Rest (undeformed) length of the edge.
    rest_length: f32,
    // Current separation of the two endpoints.
    length: f32,
    pad0: u32,
    pad1: u32,
}

struct Result {
    // Signed tensile strain; cleared to zero when invalid.
    strain: f32,
    // 1 when the rest length is above the floor, 0 for a degenerate edge.
    valid: u32,
    pad0: u32,
    pad1: u32,
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
    out.strain = 0.0;
    out.valid = 0u;
    out.pad0 = 0u;
    out.pad1 = 0u;

    // Degenerate rest length: skip the edge exactly as the golden returns None.
    if (q.rest_length <= EPS_REST) {
        results[idx] = out;
        return;
    }

    out.strain = (q.length - q.rest_length) / q.rest_length;
    out.valid = 1u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`CLOTH_TENSILE_STRAIN_WGSL`].
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
/// Two trailing pad words round the slot to `16` bytes so the host and device
/// agree on the array stride byte for byte.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    rest_length: f32,
    length: f32,
    pad0: u32,
    pad1: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct. Two trailing pad words round the slot to `16` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    strain: f32,
    valid: u32,
    pad0: u32,
    pad1: u32,
}

/// One query for the tensile-strain twin: an edge's rest length and the current
/// separation of its two endpoints.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClothTensileStrainQuery {
    /// Rest (undeformed) length of the edge.
    pub rest_length: f32,
    /// Current separation of the two endpoints.
    pub length: f32,
}

impl ClothTensileStrainQuery {
    /// Builds a query from the rest length and the current endpoint separation.
    #[must_use]
    pub fn new(rest_length: f32, length: f32) -> ClothTensileStrainQuery {
        ClothTensileStrainQuery {
            rest_length,
            length,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `tensile_strain` output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClothTensileStrainResult {
    /// The signed tensile strain `(length - rest_length) / rest_length`; zero
    /// when invalid.
    pub strain: f32,
    /// `1` when the rest length is above the floor, `0` for a degenerate edge
    /// (the golden's `None`).
    pub valid: u32,
}

/// Encodes one [`ClothTensileStrainQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &ClothTensileStrainQuery) -> GpuQuery {
    GpuQuery {
        rest_length: q.rest_length,
        length: q.length,
        pad0: 0,
        pad1: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`ClothTensileStrainResult`].
fn decode_result(raw: &GpuResult) -> ClothTensileStrainResult {
    ClothTensileStrainResult {
        strain: raw.strain,
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

/// A compiled, reusable tensile-strain compute pipeline, twinning the `CPU`
/// golden `tensile_strain`.
pub struct GpuClothTensileStrain {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuClothTensileStrain {
    /// Compiles the tensile-strain kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothTensileStrain {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_cloth_tensile_strain"),
            source: ShaderSource::Wgsl(CLOTH_TENSILE_STRAIN_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_cloth_tensile_strain_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_cloth_tensile_strain_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_cloth_tensile_strain_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuClothTensileStrain {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`ClothTensileStrainResult`] per input, in order.
    ///
    /// The continuous `strain` matches the reference to within the tolerance
    /// documented on this module; the `valid` flag matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[ClothTensileStrainQuery],
    ) -> Vec<ClothTensileStrainResult> {
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
            label: Some("prism_volumetric_cloth_tensile_strain_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cloth_tensile_strain_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cloth_tensile_strain_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_cloth_tensile_strain_bind_group"),
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
            label: Some("prism_volumetric_cloth_tensile_strain_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_cloth_tensile_strain_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_cloth_tensile_strain_pass"),
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
