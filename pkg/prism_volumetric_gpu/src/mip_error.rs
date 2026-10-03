//! `wgpu` compute twin of the stateless texture-streaming mip-shortfall charge
//! [`PageDemand::mip_error`](prism_render_architecture::texture_streaming::feedback::PageDemand::mip_error).
//!
//! The streaming-feedback priority scorer charges each visible page by how many
//! mip levels coarser the finest resident data is than what the view wants: a
//! page with resident data at least as fine as desired is charged `0`, a page
//! whose resident level is `k` levels coarser is charged `k`, and a page with no
//! resident data at all is charged a fixed
//! [`MISSING_PAGE_MIP_PENALTY`](prism_render_architecture::texture_streaming::feedback::MISSING_PAGE_MIP_PENALTY)
//! so a blank surface always outranks a merely-blurry one. The charge is a pure
//! integer map with no floating-point and no transcendental, so the port is
//! bit-exact.
//!
//! [`GpuMipError`] is the on-device twin of that one map. One thread solves one
//! query, reproducing the reference's saturating level subtraction and its
//! missing-page sentinel, so a passing real-device parity test is direct
//! evidence the ported kernel computes the same charge the reference does, not
//! merely that the shader compiles.
//!
//! # What is twinned
//!
//! The kernel reproduces exactly the reference branch: when no level is resident
//! the charge is the `MISSING_PAGE_MIP_PENALTY` constant; otherwise it is the
//! resident level minus the desired level, saturating at `0` when the resident
//! level is already at least as fine as desired (a smaller mip index is finer).
//! The reference performs the subtraction on `u8` levels with
//! [`u8::saturating_sub`]; the kernel widens both levels to `u32` on the host
//! and reproduces the saturation with an ordered `resident > desired` guard,
//! which is identical for levels in the `u8` range the host supplies.
//!
//! # What stays on the host
//!
//! The surrounding priority score
//! [`PageDemand::priority`](prism_render_architecture::texture_streaming::feedback::PageDemand)
//! and the clamped importance are `u64` / `u16` fixed-point values with no
//! portable device integer width, so they stay on the host; only the `u32`
//! mip-error charge is twinned. The residency-table requests and the scheduler
//! are stateful host infrastructure and are never dispatched.
//!
//! # Correctness model
//!
//! The charge is a pure `u32` map built from an integer subtraction and ordered
//! comparisons, so `CPU` and `GPU` agree bit-for-bit and the parity test asserts
//! an exact `==` on every query.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — unsigned subtraction
//! and ordered comparison — with no `sin`, `cos`, `exp`, `log`, `pow`, no
//! inverse trigonometry, no `sqrt` and no `u64`. No optional device feature is
//! required, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. There is no
//! loop: each thread performs a fixed, bounded sequence of arithmetic, so the
//! kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::texture_streaming::feedback::PageDemand::mip_error`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` mip-shortfall kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`mip_error`](prism_render_architecture::texture_streaming::feedback::PageDemand::mip_error);
/// see the module documentation for the algorithm.
const MIP_ERROR_WGSL: &str = r#"
// Texture-streaming mip-shortfall twin: one thread computes the integer mip
// charge for one page demand — the missing-page sentinel when no level is
// resident, otherwise the resident-minus-desired level gap saturating at zero —
// mirroring the CPU golden `feedback::PageDemand::mip_error` with only unsigned
// subtraction and ordered comparison. It owns no priority score; those u64/u16
// fixed-point values stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::texture_streaming::feedback
// ::PageDemand::mip_error；无第三方引擎源码或衍生代码。

// mip-error charge for a page with no resident data of any level, mirroring the
// golden MISSING_PAGE_MIP_PENALTY constant.
const MISSING_PAGE_MIP_PENALTY: u32 = 24u;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Finest mip level the view wants (0 = finest).
    desired_mip: u32,
    // Finest resident mip level; meaningful only when has_resident != 0.
    resident_mip: u32,
    // 1 when some level is resident, 0 when nothing is resident.
    has_resident: u32,
    pad0: u32,
}

struct Result {
    // Integer mip-shortfall charge.
    mip_error: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
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

    var charge: u32 = MISSING_PAGE_MIP_PENALTY;
    if (q.has_resident != 0u) {
        // Resident data exists: charge the level gap, saturating at zero when
        // the resident level is already at least as fine as desired (a smaller
        // mip index is finer). This mirrors u8 saturating subtraction for the
        // u8-range levels the host supplies.
        if (q.resident_mip > q.desired_mip) {
            charge = q.resident_mip - q.desired_mip;
        } else {
            charge = 0u;
        }
    }

    var out: Result;
    out.mip_error = charge;
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`MIP_ERROR_WGSL`].
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

/// `repr(C)` `std430` layout of one mip-error query, matching the `WGSL` `Query`
/// struct: the desired and resident levels, the resident flag, and a pad word to
/// a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Finest mip level the view wants (`0` = finest).
    desired_mip: u32,
    /// Finest resident mip level; meaningful only when `has_resident` is `1`.
    resident_mip: u32,
    /// `1` when some level is resident, `0` when nothing is resident.
    has_resident: u32,
    /// Padding word.
    pad0: u32,
}

/// `repr(C)` `std430` layout of one mip-error result, matching the `WGSL`
/// `Result` struct: the integer charge and three pad words to a `16`-byte
/// stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Integer mip-shortfall charge.
    mip_error: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One mip-error query: the desired mip level and the resident mip level (or its
/// absence), mirroring the inputs the reference `mip_error` reads from a
/// [`PageDemand`](prism_render_architecture::texture_streaming::feedback::PageDemand).
///
/// The host owns the surrounding priority scoring and enqueues one
/// [`MipErrorQuery`] per visible page.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MipErrorQuery {
    /// Finest mip level the view wants (`0` = finest; the golden `desired_mip`).
    pub desired_mip: u32,
    /// Finest resident mip level; meaningful only when `has_resident` is `1`
    /// (the golden `resident_mip` payload).
    pub resident_mip: u32,
    /// `1` when some level is resident, `0` when nothing is resident (the golden
    /// `resident_mip` `Option` discriminant).
    pub has_resident: u32,
}

impl MipErrorQuery {
    /// Builds a query from the desired mip level and the optional resident mip
    /// level, mirroring the reference `resident_mip: Option<u8>` field.
    #[must_use]
    pub const fn new(desired_mip: u32, resident_mip: Option<u32>) -> MipErrorQuery {
        match resident_mip {
            Some(resident) => MipErrorQuery {
                desired_mip,
                resident_mip: resident,
                has_resident: 1,
            },
            None => MipErrorQuery {
                desired_mip,
                resident_mip: 0,
                has_resident: 0,
            },
        }
    }
}

/// One resolved mip-error query, mirroring the reference charge.
///
/// `mip_error` is the integer mip-shortfall charge: the missing-page sentinel
/// when nothing is resident, otherwise the saturating resident-minus-desired
/// level gap.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MipErrorResult {
    /// Integer mip-shortfall charge (the golden
    /// [`mip_error`](prism_render_architecture::texture_streaming::feedback::PageDemand::mip_error)).
    pub mip_error: u32,
}

/// Encodes one [`MipErrorQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &MipErrorQuery) -> GpuQuery {
    GpuQuery {
        desired_mip: q.desired_mip,
        resident_mip: q.resident_mip,
        has_resident: q.has_resident,
        pad0: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`MipErrorResult`].
fn decode_result(raw: &GpuResult) -> MipErrorResult {
    MipErrorResult {
        mip_error: raw.mip_error,
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

/// A compiled, reusable mip-error compute pipeline, twinning the stateless
/// `u32` charge of the `CPU` golden
/// [`mip_error`](prism_render_architecture::texture_streaming::feedback::PageDemand::mip_error).
pub struct GpuMipError {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMipError {
    /// Compiles the mip-error kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMipError {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mip_error"),
            source: ShaderSource::Wgsl(MIP_ERROR_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mip_error_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mip_error_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mip_error_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMipError {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`MipErrorResult`] per
    /// input, in order.
    ///
    /// Each charge equals the reference exactly. An empty `queries` batch
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[MipErrorQuery]) -> Vec<MipErrorResult> {
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
            label: Some("prism_volumetric_mip_error_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mip_error_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mip_error_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mip_error_bind_group"),
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
            label: Some("prism_volumetric_mip_error_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mip_error_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mip_error_pass"),
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
