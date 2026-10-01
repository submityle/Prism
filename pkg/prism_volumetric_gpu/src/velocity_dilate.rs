//! `wgpu` compute twin of the `McGuire` 2012 motion-blur velocity pre-pass
//! ([`particle::velocity_dilate`](prism_render_architecture::particle::velocity_dilate),
//! design section 21, "速度膨胀 `TileMax`/`NeighborMax`").
//!
//! Reconstruction-filter motion blur (`McGuire` et al., *A Reconstruction
//! Filter for Plausible Motion Blur*, 2012) does not read every screen pixel's
//! own velocity; it first dilates the per-pixel velocity field in two stages so
//! a fast silhouette can still smear across the slower pixels it sweeps past:
//!
//! 1. **`TileMax`** down-samples the per-pixel velocity grid into `tile`s
//!    (typically `20px` square), each keeping the single velocity of *largest
//!    magnitude* found inside it.
//! 2. **`NeighborMax`** then replaces every `tile` with the largest-magnitude
//!    velocity among its `3x3` `tile` neighbourhood, clamped at the grid
//!    border, so the blur can reach one `tile` beyond the moving object.
//!
//! The `CPU` golden
//! [`particle::velocity_dilate`](prism_render_architecture::particle::velocity_dilate)
//! owns that math over its [`Vel2`] field and [`VelocityDilateConfig`] tiling;
//! [`GpuVelocityDilate`] is the on-device twin that runs one thread per `tile`
//! for each stage and returns the same row-major dominant-velocity grid. A
//! passing real-device parity test is therefore direct evidence the ported
//! kernels reduce the same field the reference does, not merely that the shader
//! compiles.
//!
//! # Step-for-step parity
//!
//! Each kernel mirrors the reference exactly: the same `div_ceil` `tile`
//! partition (`tile_cols` / `tile_rows`), the same partial-`tile` border clamp
//! (`x1 = min(x0 + tile_size, width)`), the same row-major (`x`-fastest)
//! pixel and `tile` indexing, and the same strict-greater "keep incumbent on a
//! tie" magnitude rule evaluated in the identical scan order. Magnitude is
//! compared through squared length (`x * x + y * y`) so no square root is ever
//! taken; the stored winner is a verbatim copy of an input velocity, so the two
//! implementations agree to the bit in the common case. The `3x3`
//! `NeighborMax` window reuses the reference's clamped neighbour indices
//! (`lo = saturating_sub(1)`, `hi = min(c + 1, last)`) and visits them in the
//! same nine-step order, so even a tie among equal-magnitude neighbours resolves
//! to the same `tile`.
//!
//! # Portability
//!
//! The kernels use only the portable core-`WGSL` subset — `min`, `max`, integer
//! arithmetic and `+ - *` on `f32` — with no `sin`, `cos`, `exp`, `log`, `pow`
//! or optional device feature, so they run unmodified on Metal, Vulkan and
//! DX12. There is no square root at all: magnitude *comparisons* use squared
//! length, matching the reference, so the dilation never evaluates a
//! transcendental and never divides.
//!
//! # Correctness model
//!
//! Both stages are a pure maximum-selection reduction that stores a copied
//! input velocity, with no arithmetic applied to the surviving value and no
//! reorderable accumulation (each `tile` is reduced by one thread in the
//! reference's scan order). `CPU` and `GPU` therefore select the identical
//! winner and copy identical bits. The parity test still asserts a tolerance
//! (`abs_diff <= 1e-5` or `rel_diff <= 1e-5`) to stay robust against a legal
//! fused multiply-add in the squared-length comparison that could, in a
//! pathological near-tie, flip which of two almost-equal-magnitude velocities
//! wins.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `McGuire` 2012 reconstruction-filter velocity dilation
//! plus `wgpu` compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::velocity_dilate::{Vel2, VelocityDilateConfig};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// The portable core-`WGSL` velocity-dilation kernels, embedded inline so the
/// twin ships as a single source file. Both entrypoints share one bind-group
/// signature (`uniform` params, read-only source grid, read-write destination
/// grid); `TileMax` binds the per-pixel field as the source and the `tile`
/// grid as the destination, while `NeighborMax` binds the `tile` grid as the
/// source and the dilated grid as the destination. See the module
/// documentation for the algorithm.
///
/// Provenance: standard `McGuire` 2012 velocity dilation; no Unreal Engine
/// source or derived code.
const VELOCITY_DILATE_WGSL: &str = r#"
// Velocity-dilation twin: TileMax runs one thread per tile and keeps the
// largest-magnitude per-pixel velocity in that tile; NeighborMax runs one
// thread per tile and keeps the largest-magnitude velocity across its clamped
// 3x3 tile neighbourhood. Both mirror the CPU golden
// `particle::velocity_dilate`, compare magnitude via squared length (no sqrt),
// use only the portable core-WGSL subset (min/max and + - *), and take no
// optional feature, so they run unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard McGuire 2012 reconstruction-filter velocity dilation; no
// Unreal Engine source or derived code.

struct Params {
    // Per-pixel velocity field width in pixels (TileMax only).
    width: u32,
    // Per-pixel velocity field height in pixels (TileMax only).
    height: u32,
    // Square tile edge in pixels (TileMax only).
    tile_size: u32,
    // Number of tile columns in the dilated grid.
    tile_cols: u32,
    // Number of tile rows in the dilated grid.
    tile_rows: u32,
    // Total tile count `tile_cols * tile_rows`.
    num_tiles: u32,
    // Padding to a 32-byte std430 uniform block.
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec2<f32>>;

// Squared Euclidean magnitude; comparisons use this so no square root is ever
// taken, matching the reference `Vel2::length_squared`.
fn length_squared(v: vec2<f32>) -> f32 {
    return v.x * v.x + v.y * v.y;
}

// Returns the larger-magnitude of two velocities, preferring `challenger` only
// when it is strictly larger so ties keep the incumbent, like the reference
// `keep_larger`.
fn keep_larger(incumbent: vec2<f32>, challenger: vec2<f32>) -> vec2<f32> {
    if (length_squared(challenger) > length_squared(incumbent)) {
        return challenger;
    }
    return incumbent;
}

// Saturating `index - 1` on unsigned coordinates, like the reference
// `usize::saturating_sub(1)` used for the low neighbour.
fn sat_sub(v: u32) -> u32 {
    if (v == 0u) {
        return 0u;
    }
    return v - 1u;
}

// Stage 1 — TileMax: each thread owns one tile and scans its clamped pixel
// window in row-major order, keeping the largest-magnitude velocity.
@compute @workgroup_size(64)
fn tile_max_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.num_tiles) {
        return;
    }
    let tx = idx % params.tile_cols;
    let ty = idx / params.tile_cols;

    let x0 = tx * params.tile_size;
    let y0 = ty * params.tile_size;
    let x1 = min(x0 + params.tile_size, params.width);
    let y1 = min(y0 + params.tile_size, params.height);

    var best = vec2<f32>(0.0, 0.0);
    var y = y0;
    loop {
        if (y >= y1) {
            break;
        }
        let row = y * params.width;
        var x = x0;
        loop {
            if (x >= x1) {
                break;
            }
            best = keep_larger(best, src[row + x]);
            x = x + 1u;
        }
        y = y + 1u;
    }
    dst[idx] = best;
}

// Stage 2 — NeighborMax: each thread owns one tile and reduces the largest-
// magnitude velocity across its clamped 3x3 tile neighbourhood, visiting the
// nine tiles in the same (row-outer, column-inner) order as the reference.
@compute @workgroup_size(64)
fn neighbor_max_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.num_tiles) {
        return;
    }
    let tx = idx % params.tile_cols;
    let ty = idx / params.tile_cols;

    var last_col = 0u;
    if (params.tile_cols != 0u) {
        last_col = params.tile_cols - 1u;
    }
    var last_row = 0u;
    if (params.tile_rows != 0u) {
        last_row = params.tile_rows - 1u;
    }

    // Clamped neighbour rows/columns; duplicates on the border are harmless
    // because the reduction is idempotent under repeated maxima.
    let ny0 = sat_sub(ty);
    let ny1 = ty;
    let ny2 = min(ty + 1u, last_row);
    let nx0 = sat_sub(tx);
    let nx1 = tx;
    let nx2 = min(tx + 1u, last_col);

    let base0 = ny0 * params.tile_cols;
    let base1 = ny1 * params.tile_cols;
    let base2 = ny2 * params.tile_cols;

    var best = vec2<f32>(0.0, 0.0);
    best = keep_larger(best, src[base0 + nx0]);
    best = keep_larger(best, src[base0 + nx1]);
    best = keep_larger(best, src[base0 + nx2]);
    best = keep_larger(best, src[base1 + nx0]);
    best = keep_larger(best, src[base1 + nx1]);
    best = keep_larger(best, src[base1 + nx2]);
    best = keep_larger(best, src[base2 + nx0]);
    best = keep_larger(best, src[base2 + nx1]);
    best = keep_larger(best, src[base2 + nx2]);
    dst[idx] = best;
}
"#;

/// Threads per workgroup for both dilation passes. One thread handles one
/// `tile`, so the dispatch rounds the `tile` count up to this group size.
const WORKGROUP_SIZE: u32 = 64;

/// Uniform parameters for one dilation dispatch. `repr(C)` `std430` layout
/// matching `Params` in [`VELOCITY_DILATE_WGSL`]: six index words then two pad
/// words — `32` bytes with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Per-pixel field width in pixels.
    width: u32,
    /// Per-pixel field height in pixels.
    height: u32,
    /// Square `tile` edge in pixels.
    tile_size: u32,
    /// Number of `tile` columns.
    tile_cols: u32,
    /// Number of `tile` rows.
    tile_rows: u32,
    /// Total `tile` count.
    num_tiles: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One velocity as uploaded or read back. `8`-byte `std430` stride matching a
/// `vec2<f32>` storage element, mirroring the reference [`Vel2`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuVel {
    /// Horizontal component (pixels per frame).
    x: f32,
    /// Vertical component (pixels per frame).
    y: f32,
}

/// A compiled, reusable velocity-dilation pipeline pair sharing one bind-group
/// layout: the `TileMax` and `NeighborMax` compute kernels.
pub struct GpuVelocityDilate {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    tile_max_pipeline: ComputePipeline,
    neighbor_max_pipeline: ComputePipeline,
}

impl GpuVelocityDilate {
    /// Compiles the two dilation kernels on `ctx`.
    ///
    /// The kernels use only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuVelocityDilate {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_velocity_dilate"),
            source: ShaderSource::Wgsl(VELOCITY_DILATE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_velocity_dilate_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_velocity_dilate_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let tile_max_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_velocity_dilate_tile_max_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("tile_max_pass"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let neighbor_max_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_velocity_dilate_neighbor_max_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("neighbor_max_pass"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuVelocityDilate {
            module,
            layout,
            tile_max_pipeline,
            neighbor_max_pipeline,
        }
    }

    /// Stage 1 — `TileMax`: down-samples the row-major per-pixel `velocity`
    /// field into the `tile_cols x tile_rows` grid of largest-magnitude
    /// velocities, mirroring
    /// [`velocity_dilate::tile_max`](prism_render_architecture::particle::velocity_dilate::tile_max).
    ///
    /// Pixels beyond `velocity.len()` are treated as zero and extra samples are
    /// ignored, matching the reference's `get`-with-default reads. An empty
    /// grid (`width` or `height` zero) returns an empty [`Vec`].
    #[must_use]
    pub fn tile_max(
        &self,
        ctx: &GpuContext,
        velocity: &[Vel2],
        config: VelocityDilateConfig,
    ) -> Vec<Vel2> {
        let num_tiles = config.num_tiles();
        if num_tiles == 0 {
            return Vec::new();
        }
        // Dense, width*height-length field: missing samples stay zero and extra
        // samples are ignored, matching the clamped reference reads.
        let pixel_count = config.width * config.height;
        let mut dense = vec![GpuVel { x: 0.0, y: 0.0 }; pixel_count];
        for (slot, v) in dense.iter_mut().zip(velocity.iter()) {
            slot.x = v.x;
            slot.y = v.y;
        }
        let params = Params {
            width: as_u32(config.width),
            height: as_u32(config.height),
            tile_size: as_u32(config.tile_size),
            tile_cols: as_u32(config.tile_cols()),
            tile_rows: as_u32(config.tile_rows()),
            num_tiles: as_u32(num_tiles),
            pad0: 0,
            pad1: 0,
        };
        self.run_pass(ctx, &self.tile_max_pipeline, params, &dense, num_tiles)
    }

    /// Stage 2 — `NeighborMax`: replaces each `tile` in the row-major
    /// `cols x rows` `grid` with the largest-magnitude velocity among its
    /// clamped `3x3` neighbourhood, mirroring
    /// [`velocity_dilate::neighbor_max`](prism_render_architecture::particle::velocity_dilate::neighbor_max).
    ///
    /// A `grid` whose length disagrees with `cols * rows` yields an empty
    /// [`Vec`], exactly like the reference guard.
    #[must_use]
    pub fn neighbor_max(
        &self,
        ctx: &GpuContext,
        grid: &[Vel2],
        cols: usize,
        rows: usize,
    ) -> Vec<Vel2> {
        let num_tiles = cols.saturating_mul(rows);
        if num_tiles == 0 || grid.len() != num_tiles {
            return Vec::new();
        }
        let dense: Vec<GpuVel> = grid.iter().map(|v| GpuVel { x: v.x, y: v.y }).collect();
        let params = Params {
            width: 0,
            height: 0,
            tile_size: 0,
            tile_cols: as_u32(cols),
            tile_rows: as_u32(rows),
            num_tiles: as_u32(num_tiles),
            pad0: 0,
            pad1: 0,
        };
        self.run_pass(ctx, &self.neighbor_max_pipeline, params, &dense, num_tiles)
    }

    /// Main entry point — the dilated dominant-velocity grid.
    ///
    /// Runs [`GpuVelocityDilate::tile_max`] then
    /// [`GpuVelocityDilate::neighbor_max`] and returns the
    /// [`VelocityDilateConfig::num_tiles`]-length row-major grid a downstream
    /// motion-blur pass reads, mirroring
    /// [`velocity_dilate::dominant_velocity`](prism_render_architecture::particle::velocity_dilate::dominant_velocity).
    #[must_use]
    pub fn dominant_velocity(
        &self,
        ctx: &GpuContext,
        velocity: &[Vel2],
        config: VelocityDilateConfig,
    ) -> Vec<Vel2> {
        let tiles = self.tile_max(ctx, velocity, config);
        self.neighbor_max(ctx, &tiles, config.tile_cols(), config.tile_rows())
    }

    /// Runs one dilation kernel: uploads `src`, dispatches one thread per
    /// output `tile`, and reads back `num_out` velocities.
    fn run_pass(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        params: Params,
        src: &[GpuVel],
        num_out: usize,
    ) -> Vec<Vel2> {
        let device = ctx.device();
        let out_bytes = (num_out as u64) * (size_of::<GpuVel>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_velocity_dilate_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let src_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_velocity_dilate_src"),
            contents: bytemuck::cast_slice(src),
            usage: BufferUsages::STORAGE,
        });
        let dst_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_velocity_dilate_dst"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dst_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_velocity_dilate_dst_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_velocity_dilate_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: src_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: dst_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_velocity_dilate_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_velocity_dilate_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per output tile, in workgroups of `WORKGROUP_SIZE`.
            let groups = (num_out as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&dst_buf, 0, &dst_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        dst_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = dst_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let gpu_out = bytemuck::cast_slice::<u8, GpuVel>(&view).to_vec();
        drop(view);
        dst_stage.unmap();
        debug_assert_eq!(gpu_out.len(), num_out);

        gpu_out
            .into_iter()
            .map(|v| Vel2::new(v.x, v.y))
            .collect()
    }
}

/// Narrows a host `usize` to a `u32`, saturating to `u32::MAX` rather than
/// panicking on a pathological value, matching the reference `to_std430`.
fn as_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
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
