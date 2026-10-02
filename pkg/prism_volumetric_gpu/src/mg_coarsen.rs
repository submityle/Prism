//! `wgpu` compute twin of the `CPU` golden multigrid resolution-coarsening
//! integer helpers
//! ([`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure),
//! design §10).
//!
//! Building a pressure multigrid hierarchy repeatedly halves a grid's
//! resolution. The `CPU` golden
//! [`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure)
//! module owns the (private) integer rules this twin reproduces, and
//! [`GpuMgCoarsen`] is the on-device twin validated against a line-for-line
//! mirror of those rules so a passing real-device parity test is direct evidence
//! the ported kernel halves and counts axes identically, not merely that its
//! shader compiles.
//!
//! # What is twinned
//!
//! Each thread owns one grid and produces, in a single [`GpuMgCoarsenResult`],
//! the next coarser resolution and the number of axes that genuinely shrank:
//!
//! - [`cx`](GpuMgCoarsenResult::cx), [`cy`](GpuMgCoarsenResult::cy) and
//!   [`cz`](GpuMgCoarsenResult::cz) are the per-axis coarsened extents, twinning
//!   the golden `coarse_axis` applied to each axis (the golden `coarsen`): an
//!   axis of one cell is left untouched, otherwise it is halved rounding up
//!   (`n.div_ceil(2)`, which equals `(n + 1) / 2` for `n >= 1`).
//! - [`k`](GpuMgCoarsenResult::k) is the number of axes strictly reduced by the
//!   coarsening, twinning the golden `coarsened_axis_count`; it lies in
//!   `0..=3`.
//!
//! # Correctness model
//!
//! Every output is pure unsigned integer arithmetic — a compare, an add and a
//! divide-by-two — with no floating point anywhere, so the `CPU` and `GPU`
//! results are bit-identical and the parity test asserts exact equality
//! (`==`), not a tolerance.
//!
//! # Degenerate inputs
//!
//! An empty query batch short-circuits on the host with no dispatch (a storage
//! buffer cannot be zero-sized). A single-cell axis (`n <= 1`) is left unchanged
//! and does not contribute to [`k`](GpuMgCoarsenResult::k); a `1x1x1` grid
//! coarsens to itself with [`k`](GpuMgCoarsenResult::k) zero, exactly as the
//! golden rule returns.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `select` plus unsigned
//! `+` and `/` — with no `sin`, `cos`, `exp`, `pow`, optional device feature,
//! `f32` or `u64`, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::multigrid_pressure`
//! 的分辨率粗化整数规则；无第三方引擎源码或衍生代码。
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

/// The number of threads per workgroup. `64` is a portable, warp-friendly size
/// used across this crate's kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` coarsening kernel, embedded inline so the twin ships
/// as a single source file. One thread owns one grid; see the module
/// documentation for the algorithm.
const MG_COARSEN_WGSL: &str = r#"
// Multigrid resolution-coarsening twin: one thread owns one (nx, ny, nz) grid
// and reproduces the CPU golden coarsen rule plus the coarsened-axis count in a
// single result record. It uses only the portable core-WGSL subset (unsigned
// select, + and /), takes no optional feature, uses no floating point, and runs
// unmodified on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 particle::multigrid_pressure 的分辨率粗化整数规则；无第三方引擎源码或衍生代码。

struct Params {
    // Number of valid grids; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Fine-grid extents in cells along each axis.
    nx: u32,
    ny: u32,
    nz: u32,
    pad: u32,
}

struct CoarsenResult {
    // Coarsened extents along each axis.
    cx: u32,
    cy: u32,
    cz: u32,
    // Number of axes genuinely reduced (0..=3).
    k: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// The per-thread grid batch.
@group(0) @binding(1) var<storage, read> queries: array<Query>;
// The per-thread result batch.
@group(0) @binding(2) var<storage, read_write> results: array<CoarsenResult>;

// Coarsens a single axis: halve it (rounding up) unless it is already a single
// cell. div_ceil(2) equals (n + 1) / 2 for n >= 1, so the n <= 1 branch simply
// returns n unchanged, matching coarse_axis.
fn coarse_axis(n: u32) -> u32 {
    return select(n, (n + 1u) / 2u, n > 1u);
}

@compute @workgroup_size(64)
fn coarsen(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let q = queries[idx];
    let cx = coarse_axis(q.nx);
    let cy = coarse_axis(q.ny);
    let cz = coarse_axis(q.nz);

    // Count axes strictly reduced, matching coarsened_axis_count.
    var k = 0u;
    if (cx < q.nx) {
        k = k + 1u;
    }
    if (cy < q.ny) {
        k = k + 1u;
    }
    if (cz < q.nz) {
        k = k + 1u;
    }

    results[idx].cx = cx;
    results[idx].cy = cy;
    results[idx].cz = cz;
    results[idx].k = k;
}
"#;

/// Uniform parameters for one coarsening dispatch. `repr(C)` `std140` layout
/// matching `Params` in [`MG_COARSEN_WGSL`]: the grid count plus three pad words
/// — `16` bytes total.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of valid grids in the batch.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One grid as uploaded to the device. `16`-byte `std430` stride matching
/// `Query` in [`MG_COARSEN_WGSL`]: the three fine-grid extents plus one pad word.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQueryRaw {
    /// Fine-grid extent in cells along `x`.
    nx: u32,
    /// Fine-grid extent in cells along `y`.
    ny: u32,
    /// Fine-grid extent in cells along `z`.
    nz: u32,
    /// Padding word, held at zero.
    pad: u32,
}

/// One result as read back from the device. `16`-byte `std430` stride matching
/// `CoarsenResult` in [`MG_COARSEN_WGSL`]: the three coarsened extents plus the
/// coarsened-axis count.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResultRaw {
    /// Coarsened extent along `x`.
    cx: u32,
    /// Coarsened extent along `y`.
    cy: u32,
    /// Coarsened extent along `z`.
    cz: u32,
    /// Number of axes genuinely reduced.
    k: u32,
}

/// One grid-coarsening query: the three fine-grid extents in cells.
///
/// Provenance: twin input record for the golden
/// [`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure)
/// coarsening rules; no third party engine source or derived code.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuMgCoarsenQuery {
    /// Fine-grid extent in cells along `x`.
    pub nx: u32,
    /// Fine-grid extent in cells along `y`.
    pub ny: u32,
    /// Fine-grid extent in cells along `z`.
    pub nz: u32,
}

/// The coarsening outputs evaluated at one [`GpuMgCoarsenQuery`]: the next
/// coarser resolution and the number of axes that genuinely shrank.
///
/// Provenance: twin output record for the golden
/// [`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure)
/// coarsening rules; no third party engine source or derived code.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuMgCoarsenResult {
    /// Coarsened extent along `x`.
    pub cx: u32,
    /// Coarsened extent along `y`.
    pub cy: u32,
    /// Coarsened extent along `z`.
    pub cz: u32,
    /// Number of axes strictly reduced by the coarsening, in `0..=3`.
    pub k: u32,
}

/// A compiled, reusable multigrid coarsening kernel, twinning the `CPU` golden
/// [`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure)
/// resolution-coarsening rules.
///
/// Provenance: on-device twin of the golden
/// [`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure)
/// coarsening rules; no third party engine source or derived code.
pub struct GpuMgCoarsen {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMgCoarsen {
    /// Compiles the coarsening kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is requested and compilation succeeds on any `Metal`,
    /// `Vulkan` or `DX12` backend.
    ///
    /// Provenance: pipeline construction for the golden
    /// [`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure)
    /// coarsening twin; no third party engine source or derived code.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMgCoarsen {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mg_coarsen_module"),
            source: ShaderSource::Wgsl(MG_COARSEN_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mg_coarsen_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mg_coarsen_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mg_coarsen_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("coarsen"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMgCoarsen {
            module,
            layout,
            pipeline,
        }
    }

    /// Coarsens every grid in `queries`, returning one [`GpuMgCoarsenResult`] per
    /// query in input order.
    ///
    /// Each result matches the golden rules exactly (bit-identical unsigned
    /// integers). An empty query batch returns an empty vector with no dispatch
    /// issued (a storage buffer cannot be zero-sized). A single-cell axis is left
    /// unchanged and a `1x1x1` grid coarsens to itself with
    /// [`k`](GpuMgCoarsenResult::k) zero, matching the golden rule.
    ///
    /// Provenance: dispatch and read-back for the golden
    /// [`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure)
    /// coarsening twin; no third party engine source or derived code.
    #[must_use]
    pub fn coarsen(
        &self,
        ctx: &GpuContext,
        queries: &[GpuMgCoarsenQuery],
    ) -> Vec<GpuMgCoarsenResult> {
        if queries.is_empty() {
            return Vec::new();
        }

        let device = ctx.device();

        let gpu_params = Params {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mg_coarsen_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });

        let raw_queries: Vec<GpuQueryRaw> = queries
            .iter()
            .map(|q| GpuQueryRaw {
                nx: q.nx,
                ny: q.ny,
                nz: q.nz,
                pad: 0,
            })
            .collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mg_coarsen_queries"),
            contents: bytemuck::cast_slice(&raw_queries),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (queries.len() * size_of::<GpuResultRaw>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mg_coarsen_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mg_coarsen_bind_group"),
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
            label: Some("prism_volumetric_mg_coarsen_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mg_coarsen_encoder"),
        });
        {
            let groups = (queries.len() as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mg_coarsen_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per grid, flattened to a 1-D dispatch.
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
        let raw = bytemuck::cast_slice::<u8, GpuResultRaw>(&view).to_vec();
        drop(view);
        stage.unmap();
        debug_assert_eq!(raw.len(), queries.len());

        raw.into_iter()
            .map(|r| GpuMgCoarsenResult {
                cx: r.cx,
                cy: r.cy,
                cz: r.cz,
                k: r.k,
            })
            .collect()
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
