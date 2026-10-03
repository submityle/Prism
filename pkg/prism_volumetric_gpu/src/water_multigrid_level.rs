//! `wgpu` compute twin of the geometric `multigrid` level helpers
//! ([`pressure_multigrid`](prism_render_architecture::water::pressure_multigrid)).
//!
//! The `FLIP`/`APIC` pressure projection removes divergence by solving a Poisson
//! system on a vertex-centred grid hierarchy. Setting up that hierarchy needs a
//! few small, stateless numeric helpers: a test that a per-axis node count is a
//! valid `2^L + 1` level size, the coarse node count one level down, and the
//! `L2` norm of a residual field used as the V-cycle stop criterion. All three
//! are built from integer bit tests, an integer divide, and a single `sqrt`, so
//! they port to the device directly.
//!
//! [`GpuWaterMultigridLevel`] is the on-device twin: one thread resolves one
//! query, reproducing
//! [`is_valid_level_size`](prism_render_architecture::water::pressure_multigrid::is_valid_level_size),
//! [`coarse_size`](prism_render_architecture::water::pressure_multigrid::coarse_size)
//! and
//! [`l2_norm`](prism_render_architecture::water::pressure_multigrid::l2_norm). A
//! passing real-device parity test is direct evidence the ported kernel
//! reproduces the reference arithmetic, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For one query carrying a per-axis node count `n`, a `field_count`, and a
//! fixed-capacity `field` buffer, the kernel reproduces:
//! - `valid`: `is_valid_level_size`, true when `n >= 3` and `n - 1` is a power
//!   of two, computed as `m != 0 && (m & (m - 1)) == 0` with `m = n - 1`;
//! - `coarse`: `coarse_size`, the integer `(n - 1) / 2 + 1`, guarded so a count
//!   below `3` yields `0` instead of wrapping (the golden requires `n >= 3`);
//!   and
//! - `l2`: `l2_norm`, the square root of the sum of the first `field_count`
//!   squared `field` entries.
//!
//! # What stays on the host
//!
//! Nothing of the twinned trio stays on the host; each query is self-contained.
//! The variable-length residual field is bounded to [`MAX_FIELD`] entries and
//! padded on the host so the storage layout is fixed. An empty batch
//! short-circuits on the host, since a storage buffer cannot be zero-sized.
//!
//! # Correctness model
//!
//! `valid` and `coarse` are exact integer results, so the parity test pins them
//! with equality. `l2` goes through one `sqrt`, so it is asserted within the
//! shared continuous tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`).
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — integer `&`, `-`,
//! `==`, an integer divide, a loop, and one `sqrt` — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan`, no inverse trigonometry, and no `64`-bit integers
//! or floats. No optional device feature is required, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::pressure_multigrid`；无第三方引擎源码或衍生代码。
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

/// Fixed upper bound on the residual `field` length carried by one query. The
/// variable-length golden slice is bounded and host-padded to this capacity so
/// the `std430` layout stays fixed.
pub const MAX_FIELD: usize = 64;

/// The portable core-`WGSL` `multigrid`-level kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `is_valid_level_size`, `coarse_size` and `l2_norm`; see the
/// module documentation for the algorithm.
const WATER_MULTIGRID_LEVEL_WGSL: &str = r#"
// Multigrid-level twin: one thread resolves one query's level validity, coarse
// node count and residual L2 norm, mirroring the CPU golden
// `water::pressure_multigrid::{is_valid_level_size, coarse_size, l2_norm}` with
// only integer bit tests, an integer divide and a single sqrt.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::pressure_multigrid；无第三方
// 引擎源码或衍生代码。

const MAX_FIELD: u32 = 64u;

// is_valid_level_size: n >= 3 and n - 1 is a power of two.
fn is_valid_level(n: u32) -> bool {
    if (n < 3u) {
        return false;
    }
    let m = n - 1u;
    return (m != 0u) && ((m & (m - 1u)) == 0u);
}

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Per-axis node count under test.
    n: u32,
    // Number of valid leading entries in `field`.
    field_count: u32,
    pad0: u32,
    pad1: u32,
    // Residual field, host-padded to MAX_FIELD entries.
    field: array<f32, 64>,
}

struct Result {
    // 1 when `n` is a valid 2^L + 1 level size, else 0.
    valid: u32,
    // Coarse node count (n - 1) / 2 + 1, guarded to 0 when n < 3.
    coarse: u32,
    // L2 norm of the first `field_count` field entries.
    l2: f32,
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

    var valid: u32 = 0u;
    if (is_valid_level(q.n)) {
        valid = 1u;
    }

    // coarse_size requires n >= 3; guard the subtraction so a smaller count
    // yields 0 rather than wrapping in unsigned arithmetic.
    var coarse: u32 = 0u;
    if (q.n >= 3u) {
        coarse = (q.n - 1u) / 2u + 1u;
    }

    // L2 norm over the first field_count entries.
    var acc: f32 = 0.0;
    var i: u32 = 0u;
    loop {
        if (i >= q.field_count) {
            break;
        }
        let v = q.field[i];
        acc = acc + v * v;
        i = i + 1u;
    }

    var out: Result;
    out.valid = valid;
    out.coarse = coarse;
    out.l2 = sqrt(acc);
    out.pad0 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words,
/// filling a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WATER_MULTIGRID_LEVEL_WGSL`].
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

/// `repr(C)` `std430` layout of one query: the node count, the field length,
/// two pad words, and the fixed-capacity residual field, matching the `WGSL`
/// `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Per-axis node count under test.
    n: u32,
    /// Number of valid leading entries in `field`.
    field_count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Residual field, host-padded to `MAX_FIELD` entries.
    field: [f32; MAX_FIELD],
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct:
/// the validity flag, the coarse node count, the `L2` norm and a pad word, a
/// `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `1` when the level size is valid, else `0`.
    valid: u32,
    /// Coarse node count.
    coarse: u32,
    /// `L2` norm of the leading field entries.
    l2: f32,
    /// Padding word.
    pad0: u32,
}

/// One `multigrid`-level query for the twin: a per-axis node count `n`, the
/// residual field length `field_count`, and the residual `field` itself.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterMultigridLevelQuery {
    /// Per-axis node count under test.
    pub n: u32,
    /// Number of valid leading entries in `field`.
    pub field_count: u32,
    /// Residual field, host-padded to `MAX_FIELD` entries.
    pub field: [f32; MAX_FIELD],
}

impl WaterMultigridLevelQuery {
    /// Builds a query from a node count and a residual field slice.
    ///
    /// The slice is copied into a fixed `MAX_FIELD`-entry buffer, with
    /// `field_count` set to the slice length and the remaining entries left at
    /// `0`. Slices longer than `MAX_FIELD` are truncated to the capacity.
    #[must_use]
    pub fn new(n: u32, field: &[f32]) -> WaterMultigridLevelQuery {
        let field_count = field.len().min(MAX_FIELD);
        let mut buf = [0.0f32; MAX_FIELD];
        buf[..field_count].copy_from_slice(&field[..field_count]);
        WaterMultigridLevelQuery {
            n,
            field_count: field_count as u32,
            field: buf,
        }
    }
}

/// One resolved `multigrid`-level query, mirroring the reference
/// [`is_valid_level_size`](prism_render_architecture::water::pressure_multigrid::is_valid_level_size),
/// [`coarse_size`](prism_render_architecture::water::pressure_multigrid::coarse_size)
/// and
/// [`l2_norm`](prism_render_architecture::water::pressure_multigrid::l2_norm).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterMultigridLevelResult {
    /// Whether `n` is a valid `2^L + 1` level size.
    pub valid: bool,
    /// Coarse node count `(n - 1) / 2 + 1`, or `0` when `n < 3`.
    pub coarse: u32,
    /// `L2` norm of the first `field_count` field entries.
    pub l2: f32,
}

/// Encodes one [`WaterMultigridLevelQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &WaterMultigridLevelQuery) -> GpuQuery {
    GpuQuery {
        n: q.n,
        field_count: q.field_count,
        pad0: 0,
        pad1: 0,
        field: q.field,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`WaterMultigridLevelResult`], mapping the `u32` flag back to `bool`.
fn decode_result(raw: &GpuResult) -> WaterMultigridLevelResult {
    WaterMultigridLevelResult {
        valid: raw.valid != 0,
        coarse: raw.coarse,
        l2: raw.l2,
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

/// A compiled, reusable `multigrid`-level compute pipeline, twinning the `CPU`
/// golden level helpers from
/// [`pressure_multigrid`](prism_render_architecture::water::pressure_multigrid).
pub struct GpuWaterMultigridLevel {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterMultigridLevel {
    /// Compiles the `multigrid`-level kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterMultigridLevel {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_multigrid_level"),
            source: ShaderSource::Wgsl(WATER_MULTIGRID_LEVEL_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_multigrid_level_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_multigrid_level_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_multigrid_level_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterMultigridLevel {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one
    /// [`WaterMultigridLevelResult`] per input, in order.
    ///
    /// Each output matches the reference: the level validity flag, the coarse
    /// node count and the residual `L2` norm. An empty `queries` batch returns
    /// an empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterMultigridLevelQuery],
    ) -> Vec<WaterMultigridLevelResult> {
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
            label: Some("prism_volumetric_water_multigrid_level_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_multigrid_level_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_multigrid_level_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_multigrid_level_bind_group"),
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
            label: Some("prism_volumetric_water_multigrid_level_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_multigrid_level_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_multigrid_level_pass"),
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
