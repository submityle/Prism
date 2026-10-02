//! `wgpu` compute twin of the `CPU` golden multigrid coarsest-level predicate
//! ([`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure),
//! design §10).
//!
//! Building a pressure multigrid hierarchy recurses coarser and coarser until a
//! grid is small enough to solve directly. The `CPU` golden
//! [`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure)
//! module owns the (private) integer rule `is_coarsest` that decides when to
//! stop, and [`GpuMgIsCoarsest`] is the on-device twin validated against a
//! line-for-line mirror of that rule so a passing real-device parity test is
//! direct evidence the ported kernel halts the recursion identically, not merely
//! that its shader compiles.
//!
//! # What is twinned
//!
//! Each thread owns one `(resolution, coarsest_axis)` pair and produces, in a
//! single [`GpuMgIsCoarsestResult`], the stopping verdict as a `0u`/`1u` flag:
//!
//! - [`flag`](GpuMgIsCoarsestResult::flag) twins the golden `is_coarsest`: it is
//!   one when the grid is *within* budget (every axis at or below
//!   `coarsest_axis`) **or** *stuck* (coarsening would not shrink it — the
//!   coarsened resolution equals the input), and zero otherwise.
//!
//! The coarsening the predicate consults twins the golden `coarse_axis` and
//! `coarsen`: an axis of one cell is left untouched, otherwise it is halved
//! rounding up (`(n + 1) / 2` for `n >= 1`).
//!
//! # Correctness model
//!
//! Every input and intermediate is an unsigned integer — compares, an add and a
//! divide-by-two — with no floating point anywhere, and the verdict is folded to
//! an exact `0u`/`1u` flag. The `CPU` and `GPU` results are therefore
//! bit-identical and the parity test asserts exact equality (`==`), not a
//! tolerance.
//!
//! # Degenerate inputs
//!
//! An empty query batch short-circuits on the host with no dispatch (a storage
//! buffer cannot be zero-sized). A `1x1x1` grid is both within any non-zero
//! budget and stuck (it coarsens to itself), so the flag is one; a grid already
//! at or below `coarsest_axis` on every axis is within budget regardless of
//! whether it could still shrink; a grid whose only coarsenable axis still
//! exceeds the budget is neither within nor stuck, so the flag is zero.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `select` plus unsigned
//! compares, `+` and `/` — with no `sin`, `cos`, `exp`, `pow`, optional device
//! feature, `f32` or `u64`, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::multigrid_pressure`
//! 的最粗层停止判定整数规则；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` coarsest-level predicate kernel, embedded inline so
/// the twin ships as a single source file. One thread owns one
/// `(resolution, coarsest_axis)` pair; see the module documentation for the
/// algorithm.
const MG_IS_COARSEST_WGSL: &str = r#"
// Multigrid coarsest-level predicate twin: one thread owns one (nx, ny, nz)
// grid plus its coarsest_axis budget and reproduces the CPU golden is_coarsest
// rule as a 0u/1u flag. It uses only the portable core-WGSL subset (unsigned
// select, compares, + and /), takes no optional feature, uses no floating
// point, and runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 particle::multigrid_pressure 的最粗层停止判定整数规则；无第三方引擎源码或衍生代码。

struct Params {
    // Number of valid queries; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Grid extents in cells along each axis.
    nx: u32,
    ny: u32,
    nz: u32,
    // The largest axis extent still considered "coarsest enough" to stop.
    coarsest_axis: u32,
}

struct IsCoarsestResult {
    // The stopping verdict as a 0u/1u flag.
    flag: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// The per-thread query batch.
@group(0) @binding(1) var<storage, read> queries: array<Query>;
// The per-thread result batch.
@group(0) @binding(2) var<storage, read_write> results: array<IsCoarsestResult>;

// Coarsens a single axis: halve it (rounding up) unless it is already a single
// cell. (n + 1u) / 2u equals div_ceil(2) for n >= 1, so the n <= 1 branch simply
// returns n unchanged, matching coarse_axis.
fn coarse_axis(n: u32) -> u32 {
    return select(n, (n + 1u) / 2u, n > 1u);
}

@compute @workgroup_size(64)
fn is_coarsest(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let q = queries[idx];

    // within: every axis at or below the coarsest_axis budget, matching the
    // golden `within`.
    let within = (q.nx <= q.coarsest_axis) && (q.ny <= q.coarsest_axis) && (q.nz <= q.coarsest_axis);

    // stuck: coarsening would not shrink any axis (coarsen(res) == res),
    // matching the golden `stuck`.
    let cx = coarse_axis(q.nx);
    let cy = coarse_axis(q.ny);
    let cz = coarse_axis(q.nz);
    let stuck = (cx == q.nx) && (cy == q.ny) && (cz == q.nz);

    // within || stuck, folded to an exact 0u/1u flag.
    let flag = select(0u, 1u, within || stuck);
    results[idx].flag = flag;
}
"#;

/// Uniform parameters for one predicate dispatch. `repr(C)` `std140` layout
/// matching `Params` in [`MG_IS_COARSEST_WGSL`]: the query count plus three pad
/// words — `16` bytes total.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of valid queries in the batch.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One query as uploaded to the device. `16`-byte `std430` stride matching
/// `Query` in [`MG_IS_COARSEST_WGSL`]: the three grid extents plus the
/// `coarsest_axis` budget.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQueryRaw {
    /// Grid extent in cells along `x`.
    nx: u32,
    /// Grid extent in cells along `y`.
    ny: u32,
    /// Grid extent in cells along `z`.
    nz: u32,
    /// The largest axis extent still considered coarse enough to stop.
    coarsest_axis: u32,
}

/// One result as read back from the device. `16`-byte `std430` stride matching
/// `IsCoarsestResult` in [`MG_IS_COARSEST_WGSL`]: the `0u`/`1u` verdict flag plus
/// three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResultRaw {
    /// The stopping verdict as a `0`/`1` flag.
    flag: u32,
    /// Padding word, held at zero.
    pad0: u32,
    /// Padding word, held at zero.
    pad1: u32,
    /// Padding word, held at zero.
    pad2: u32,
}

/// One coarsest-level predicate query: the three grid extents in cells plus the
/// `coarsest_axis` budget that decides when the hierarchy stops recursing.
///
/// Provenance: twin input record for the golden
/// [`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure)
/// `is_coarsest` rule; no third party engine source or derived code.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuMgIsCoarsestQuery {
    /// Grid extent in cells along `x`.
    pub nx: u32,
    /// Grid extent in cells along `y`.
    pub ny: u32,
    /// Grid extent in cells along `z`.
    pub nz: u32,
    /// The largest axis extent still considered coarse enough to stop.
    pub coarsest_axis: u32,
}

/// The predicate output evaluated at one [`GpuMgIsCoarsestQuery`]: the stopping
/// verdict as a `0`/`1` flag.
///
/// Provenance: twin output record for the golden
/// [`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure)
/// `is_coarsest` rule; no third party engine source or derived code.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuMgIsCoarsestResult {
    /// The stopping verdict: `1` when the grid is coarse enough to solve
    /// directly (within budget or stuck), `0` otherwise.
    pub flag: u32,
}

/// A compiled, reusable multigrid coarsest-level predicate kernel, twinning the
/// `CPU` golden
/// [`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure)
/// `is_coarsest` rule.
///
/// Provenance: on-device twin of the golden
/// [`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure)
/// `is_coarsest` rule; no third party engine source or derived code.
pub struct GpuMgIsCoarsest {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMgIsCoarsest {
    /// Compiles the predicate kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is requested and compilation succeeds on any `Metal`,
    /// `Vulkan` or `DX12` backend.
    ///
    /// Provenance: pipeline construction for the golden
    /// [`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure)
    /// `is_coarsest` twin; no third party engine source or derived code.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMgIsCoarsest {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mg_is_coarsest_module"),
            source: ShaderSource::Wgsl(MG_IS_COARSEST_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mg_is_coarsest_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mg_is_coarsest_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mg_is_coarsest_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("is_coarsest"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMgIsCoarsest {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the coarsest-level predicate for every query in `queries`,
    /// returning one [`GpuMgIsCoarsestResult`] per query in input order.
    ///
    /// Each result matches the golden rule exactly (bit-identical `0`/`1`
    /// flags). An empty query batch returns an empty vector with no dispatch
    /// issued (a storage buffer cannot be zero-sized). A `1x1x1` grid is both
    /// within any non-zero budget and stuck, so its flag is one, matching the
    /// golden rule.
    ///
    /// Provenance: dispatch and read-back for the golden
    /// [`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure)
    /// `is_coarsest` twin; no third party engine source or derived code.
    #[must_use]
    pub fn is_coarsest(
        &self,
        ctx: &GpuContext,
        queries: &[GpuMgIsCoarsestQuery],
    ) -> Vec<GpuMgIsCoarsestResult> {
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
            label: Some("prism_volumetric_mg_is_coarsest_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });

        let raw_queries: Vec<GpuQueryRaw> = queries
            .iter()
            .map(|q| GpuQueryRaw {
                nx: q.nx,
                ny: q.ny,
                nz: q.nz,
                coarsest_axis: q.coarsest_axis,
            })
            .collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mg_is_coarsest_queries"),
            contents: bytemuck::cast_slice(&raw_queries),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (queries.len() * size_of::<GpuResultRaw>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mg_is_coarsest_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mg_is_coarsest_bind_group"),
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
            label: Some("prism_volumetric_mg_is_coarsest_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mg_is_coarsest_encoder"),
        });
        {
            let groups = (queries.len() as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mg_is_coarsest_pass"),
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
        let raw = bytemuck::cast_slice::<u8, GpuResultRaw>(&view).to_vec();
        drop(view);
        stage.unmap();
        debug_assert_eq!(raw.len(), queries.len());

        raw.into_iter()
            .map(|r| GpuMgIsCoarsestResult { flag: r.flag })
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
