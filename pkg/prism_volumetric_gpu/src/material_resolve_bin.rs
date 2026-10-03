//! `wgpu` compute twin of the deferred material-resolve per-element routing
//! ([`resolve`](prism_render_architecture::material::resolve)).
//!
//! Prism shades in a screen-space resolve pass, so every frame the materials
//! referenced by the visibility buffer are partitioned into one bucket per
//! [`MaterialExecutionPath`](prism_render_architecture::material::MaterialExecutionPath)
//! by the `CPU` golden
//! [`bin_visible_materials`](prism_render_architecture::material::resolve::bin_visible_materials).
//! That routine is a single deterministic pass whose only per-element decision
//! is: look a visible index up in the record table, skip it when it is out of
//! range (`records.get(index)` yields `None`), otherwise route it to the bucket
//! for its record's execution path. [`GpuMaterialResolveBin`] is the on-device
//! twin of exactly that per-element decision, so a passing real-device parity
//! test is direct evidence the ported routing reproduces the reference's bucket
//! assignment, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! One thread handles one visible entry. Given the record table encoded as one
//! execution-path code per record and the visible index for the thread, it
//! reproduces the golden's per-element routing:
//!
//! - an index `>= num_records` is reported as skipped (`valid == false`), the
//!   device mirror of the golden's `records.get(index)` returning `None`;
//! - an in-range index reports `valid == true` and the record's execution-path
//!   code, the device mirror of routing to `record.execution`'s bucket.
//!
//! The execution-path code is the discriminant order of
//! [`MaterialExecutionPath`](prism_render_architecture::material::MaterialExecutionPath):
//! `0` = `FixedPbr`, `1` = `FixedNpr`, `2` = `ClosureTable`,
//! `3` = `DiagnosticFallback`. The decision is a single unsigned comparison and
//! an indexed load, so it is exact: there is no floating-point arithmetic in the
//! kernel at all.
//!
//! # What stays on the host
//!
//! The bucket aggregation — the four growable
//! [`Vec`](alloc::vec::Vec)s of
//! [`MaterialResolveBins`](prism_render_architecture::material::resolve::MaterialResolveBins)
//! and the order-preserving `push` into them — stays on the host: it is a
//! variable-length append per path, not a fixed per-element numeric transform,
//! and it allocates. The host also owns the record-table encoding (mapping each
//! [`MaterialRecord`](prism_render_architecture::material::MaterialRecord)'s
//! execution path to its code), which is one-time marshalling. The device
//! returns the per-element `(valid, path_code)` routing; the host replays the
//! trivial ordered push to materialize identical buckets. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Correctness model
//!
//! Every output is an integer comparison and an indexed load — no divide, no
//! transcendental, no floating point — so the `CPU` and `GPU` agree exactly and
//! the parity test asserts the per-element flag and code with `==`, then
//! reconstructs the full [`MaterialResolveBins`] from the device output and
//! asserts it equals the golden's bins verbatim.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — an unsigned compare
//! and an array load — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no
//! inverse trigonometry, no `sqrt`, no `round`, and no `u64`. No optional device
//! feature is required, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//! There is no loop: each thread performs a fixed, bounded sequence of work, so
//! the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::material::resolve::bin_visible_materials`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` material-resolve per-element routing kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden
/// [`bin_visible_materials`](prism_render_architecture::material::resolve::bin_visible_materials)
/// per-element decision; see the module documentation for the algorithm.
const MATERIAL_RESOLVE_BIN_WGSL: &str = r#"
// Material-resolve per-element routing twin: one thread routes one visible index
// to its material execution-path bucket, mirroring the CPU golden
// `material::resolve::bin_visible_materials`. An out-of-range index is reported
// as skipped (valid == 0), otherwise the record's execution-path code is
// returned. Only an unsigned compare and an array load; no floating point, no
// u64, no transcendental. The variable-length bucket Vec push stays on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::material::resolve::bin_visible_materials；
// 无第三方引擎源码或衍生代码。

struct Params {
    // Number of records in the execution-path-code table.
    num_records: u32,
    // Number of visible queries in the storage array; threads past this return.
    count: u32,
    pad0: u32,
    pad1: u32,
}

struct Query {
    // Visible entry: an index into the record table.
    record_index: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Result {
    // 1 when the index is in range (routed), 0 when skipped (out of range).
    valid: u32,
    // Execution-path code of the routed record (meaningful only when valid).
    path_code: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> records: array<u32>;
@group(0) @binding(2) var<storage, read> queries: array<Query>;
@group(0) @binding(3) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.count) {
        return;
    }
    let index = queries[tid].record_index;

    var out: Result;
    out.pad0 = 0u;
    out.pad1 = 0u;
    if (index >= params.num_records) {
        // Mirrors `records.get(index)` returning `None`: the entry is skipped.
        out.valid = 0u;
        out.path_code = 0u;
    } else {
        // Mirrors routing to `record.execution`'s bucket.
        out.valid = 1u;
        out.path_code = records[index];
    }
    results[tid] = out;
}
"#;

/// Uniform parameters for one dispatch: the record-table length, the query count
/// and two pad words to fill a `16`-byte, `std140`-aligned uniform struct
/// matching `Params` in [`MATERIAL_RESOLVE_BIN_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of records in the execution-path-code table.
    num_records: u32,
    /// Number of valid queries in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// `repr(C)` `std430` layout of one visible query: the record index plus three
/// pad words to a `16`-byte stride matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Index into the record table.
    record_index: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one routing result: the validity flag and the
/// execution-path code plus two pad words to a `16`-byte stride matching the
/// `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `1` when routed, `0` when skipped (index out of range).
    valid: u32,
    /// Execution-path code of the routed record.
    path_code: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One per-visible query for the material-resolve routing twin: a single index
/// into the record table, exactly one entry of the golden's `visible` slice.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaterialResolveBinQuery {
    /// Index into the record table (the golden's `visible` entry).
    pub record_index: u32,
}

impl MaterialResolveBinQuery {
    /// Builds a query from one visible record index.
    #[must_use]
    pub const fn new(record_index: u32) -> MaterialResolveBinQuery {
        MaterialResolveBinQuery { record_index }
    }
}

/// One routed result, mirroring the per-element decision the golden
/// [`bin_visible_materials`](prism_render_architecture::material::resolve::bin_visible_materials)
/// makes before pushing into a bucket.
///
/// `valid` is `false` when the index fell outside the record table (the golden
/// skips it); `path_code` is the execution-path code of the routed record,
/// meaningful only when `valid` is `true`. The code is the discriminant order of
/// [`MaterialExecutionPath`](prism_render_architecture::material::MaterialExecutionPath):
/// `0` = `FixedPbr`, `1` = `FixedNpr`, `2` = `ClosureTable`,
/// `3` = `DiagnosticFallback`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaterialResolveBinResult {
    /// Whether the index was in range and therefore routed.
    pub valid: bool,
    /// Execution-path code of the routed record (only when `valid`).
    pub path_code: u32,
}

/// Encodes one [`MaterialResolveBinQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &MaterialResolveBinQuery) -> GpuQuery {
    GpuQuery {
        record_index: q.record_index,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`MaterialResolveBinResult`].
fn decode_result(raw: &GpuResult) -> MaterialResolveBinResult {
    MaterialResolveBinResult {
        valid: raw.valid != 0,
        path_code: raw.path_code,
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

/// A compiled, reusable material-resolve routing compute pipeline, twinning the
/// `CPU` golden
/// [`bin_visible_materials`](prism_render_architecture::material::resolve::bin_visible_materials).
pub struct GpuMaterialResolveBin {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMaterialResolveBin {
    /// Compiles the routing kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMaterialResolveBin {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_material_resolve_bin"),
            source: ShaderSource::Wgsl(MATERIAL_RESOLVE_BIN_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_material_resolve_bin_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_material_resolve_bin_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_material_resolve_bin_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMaterialResolveBin {
            module,
            layout,
            pipeline,
        }
    }

    /// Routes every query against the record table and returns one
    /// [`MaterialResolveBinResult`] per input, in order.
    ///
    /// `path_codes` is the record table: `path_codes[i]` is the execution-path
    /// code of record `i` (`0` = `FixedPbr`, `1` = `FixedNpr`,
    /// `2` = `ClosureTable`, `3` = `DiagnosticFallback`). Each result equals the
    /// reference's per-element routing exactly. An empty `queries` batch returns
    /// an empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        path_codes: &[u32],
        queries: &[MaterialResolveBinQuery],
    ) -> Vec<MaterialResolveBinResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            num_records: path_codes.len() as u32,
            count: count as u32,
            pad0: 0,
            pad1: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_material_resolve_bin_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        // A storage buffer cannot be zero-sized; when the record table is empty
        // bind a single sentinel word. `num_records` stays `0`, so every index
        // is routed as out of range and the sentinel is never read.
        let records_storage: Vec<u32> = if path_codes.is_empty() {
            vec![0]
        } else {
            path_codes.to_vec()
        };
        let records_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_material_resolve_bin_records"),
            contents: bytemuck::cast_slice(&records_storage),
            usage: BufferUsages::STORAGE,
        });

        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_material_resolve_bin_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_material_resolve_bin_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_material_resolve_bin_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: records_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_material_resolve_bin_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_material_resolve_bin_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_material_resolve_bin_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per visible query, flattened to a 1-D dispatch.
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
