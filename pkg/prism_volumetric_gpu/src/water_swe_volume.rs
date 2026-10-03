//! `wgpu` compute twin of the shallow-water conserved-volume reduction
//! [`SweState::total_volume`](prism_render_architecture::water::swe::SweState::total_volume)
//! from the regular-grid shallow-water surface solver.
//!
//! A shallow-water body stores a depth value `h` per grid cell. Its conserved
//! quantity is the total water volume, `sum(h) * dx * dx`, where `dx` is the
//! (square) cell size: each cell contributes its depth times its base area
//! `dx * dx`. The reference folds the depth array left-to-right into a running
//! sum and multiplies once by the cell area. It is a pure reduction plus a
//! single `f32` multiply, so it ports to the device with no floating-point
//! transcendental.
//!
//! [`GpuWaterSweVolume`] is the on-device twin of
//! [`total_volume`](prism_render_architecture::water::swe::SweState::total_volume).
//! One thread owns one query, folding that query's active depth prefix in the
//! same left-to-right order and scaling by `dx * dx`, so a passing real-device
//! parity test is direct evidence the ported kernel computes the same volume
//! the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces the conserved volume:
//!
//! * The running sum `sum = sum + h[i]` for `i` in `0 .. len`, in the same
//!   left-to-right order as the reference `for &depth in &self.h` loop.
//! * The final scale `sum * dx * dx`, matching the reference `cell_area` factor.
//! * Depth cells past the active `len`, up to the fixed cap `MAX_H`, stay unused
//!   (host zero-padded) and never enter the sum.
//!
//! # What stays on the host
//!
//! The variable-length depth, `x`-velocity and `z`-velocity vectors, their
//! grid-layout bookkeeping in
//! [`SweConfig`](prism_render_architecture::water::swe::SweConfig), the solver's
//! advection, flux and damping time step, and any cross-cell reduction tree
//! stay on the host. This twin models only the stateless volume reduction, with
//! each query carrying one depth array zero-padded to the fixed cap `MAX_H` and
//! the host dispatching by the active `len`.
//!
//! # Correctness model
//!
//! The host and the device evaluate the identical left-to-right fold and the
//! identical area scale, so the volume matches to within floating-point
//! tolerance. There is no branch on an `f32`, so there is no floating-point
//! crossing to flip; a large sum is compared with the shared relative
//! tolerance.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — an unsigned integer
//! loop bound and `+` and `*` on `f32` — with no `sin`, `cos`, `exp`, `log`,
//! `pow`, no inverse trigonometry, no `sqrt` and no `u64`. No optional device
//! feature is required, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::swe`；无第三方引擎源码或衍生代码。
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

/// Fixed per-query depth-cell cap. The host zero-pads each query's depth array
/// to this length and dispatches by the active `len`; a cap of `4096` covers a
/// `64x64` tile of the regular shallow-water grid.
pub const MAX_H: usize = 4096;

/// The portable core-`WGSL` conserved-volume reduction kernel, embedded inline
/// so the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`total_volume`](prism_render_architecture::water::swe::SweState::total_volume);
/// see the module documentation for the algorithm.
const WATER_SWE_VOLUME_WGSL: &str = r#"
// Shallow-water conserved-volume twin: one thread owns one query. It folds that
// query's active depth prefix h[0..len] left-to-right into a running sum, then
// scales by the square cell area dx*dx, mirroring the CPU golden
// `water::swe::SweState::total_volume`. Depth cells past len (up to the padded
// cap) never enter the sum. It owns no advection, flux, damping step or
// variable-length layout; those stay on the host. Only an unsigned integer loop
// bound and + and * on f32 are used — no transcendental, no u64.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::swe；
// 无第三方引擎源码或衍生代码。

// The fixed padded depth-cell count MAX_H of each query's h array.
const CELLS: u32 = 4096u;

struct Params {
    // Number of queries in the storage arrays; threads past count exit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Zero-padded depth array; only the first len entries are summed.
    h: array<f32, 4096>,
    // Active depth-cell count (host guarantees len <= MAX_H).
    len: u32,
    // Square cell size; the volume scale is dx*dx.
    dx: f32,
}

struct Result {
    // Conserved volume sum(h[0..len]) * dx * dx.
    volume: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qi = gid.x;
    if (qi >= params.count) {
        return;
    }
    let len = queries[qi].len;
    let dx = queries[qi].dx;

    // Left-to-right fold matching the reference `for &depth in &self.h` loop.
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < len; i = i + 1u) {
        sum = sum + queries[qi].h[i];
    }
    results[qi].volume = sum * dx * dx;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WATER_SWE_VOLUME_WGSL`].
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct:
/// the padded depth array, the active depth-cell count and the square cell size.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Zero-padded depth array (the golden `h`).
    h: [f32; MAX_H],
    /// Active depth-cell count (the length of the golden `h`).
    len: u32,
    /// Square cell size (the golden `cfg.dx`).
    dx: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct:
/// the conserved volume.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Conserved volume `sum(h) * dx * dx` (the golden return value).
    volume: f32,
}

/// One volume query: the padded depth array, the active depth-cell count and the
/// square cell size, mirroring the state the golden
/// [`total_volume`](prism_render_architecture::water::swe::SweState::total_volume)
/// reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterSweVolumeQuery {
    /// Zero-padded depth array; only the first `len` entries are summed (the
    /// golden `h`).
    pub h: [f32; MAX_H],
    /// Active depth-cell count; the host guarantees `len <= MAX_H` (the length
    /// of the golden `h`).
    pub len: u32,
    /// Square cell size; the volume scale is `dx * dx` (the golden `cfg.dx`).
    pub dx: f32,
}

impl WaterSweVolumeQuery {
    /// Builds a query from the padded depth array, the active depth-cell count
    /// and the square cell size.
    #[must_use]
    pub const fn new(h: [f32; MAX_H], len: u32, dx: f32) -> WaterSweVolumeQuery {
        WaterSweVolumeQuery { h, len, dx }
    }
}

/// One resolved volume response: the conserved water volume, mirroring the
/// golden
/// [`total_volume`](prism_render_architecture::water::swe::SweState::total_volume)
/// return value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterSweVolumeResult {
    /// Conserved volume `sum(h) * dx * dx` (the golden return value).
    pub volume: f32,
}

/// Encodes one [`WaterSweVolumeQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &WaterSweVolumeQuery) -> GpuQuery {
    GpuQuery {
        h: q.h,
        len: q.len,
        dx: q.dx,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`WaterSweVolumeResult`].
fn decode_result(raw: &GpuResult) -> WaterSweVolumeResult {
    WaterSweVolumeResult { volume: raw.volume }
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

/// A compiled, reusable conserved-volume compute pipeline, twinning the
/// stateless reduction of the `CPU` golden
/// [`total_volume`](prism_render_architecture::water::swe::SweState::total_volume).
pub struct GpuWaterSweVolume {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterSweVolume {
    /// Compiles the conserved-volume kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterSweVolume {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_swe_volume"),
            source: ShaderSource::Wgsl(WATER_SWE_VOLUME_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_swe_volume_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_swe_volume_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_swe_volume_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterSweVolume {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves the conserved volume of every query in `queries` and returns one
    /// [`WaterSweVolumeResult`] per input, in order.
    ///
    /// The volume matches the reference to within floating-point tolerance. An
    /// empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterSweVolumeQuery],
    ) -> Vec<WaterSweVolumeResult> {
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
            label: Some("prism_volumetric_water_swe_volume_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_swe_volume_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_swe_volume_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_swe_volume_bind_group"),
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
            label: Some("prism_volumetric_water_swe_volume_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_swe_volume_encoder"),
        });
        {
            // One thread per query, flattened to a 1-D dispatch of count
            // threads.
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_swe_volume_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
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
