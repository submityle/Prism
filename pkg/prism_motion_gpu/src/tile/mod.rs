//! Host orchestration of the `McGuire` `TileMax` / `NeighborMax` velocity
//! reductions.
//!
//! Motion-blur reconstruction bounds its per-pixel gather radius by the largest
//! velocity that can reach a pixel. Computing that bound in two coarse passes is
//! the classic `McGuire` scheme: [`GpuTileMax`] reduces the full-resolution
//! velocity field to one maximum-magnitude vector per screen tile, then
//! [`GpuNeighborMax`] expands each tile to the maximum over its 3x3 tile
//! neighborhood (a fast mover one tile away can still streak into this tile).
//! The complete device-free reference — the tiling, the clamping, and the
//! "keep the larger magnitude, ties keep the incumbent" fold — lives in
//! [`tile_max`](prism_render_architecture::motion::dilation::tile_max) and
//! [`neighbor_max`](prism_render_architecture::motion::dilation::neighbor_max).
//!
//! Each kernel runs one invocation per *output* tile, scanning its source
//! region in the identical row-major order as the golden and folding with the
//! same strict `>` comparison seeded at the zero vector. Every written value is
//! a whole-vector copy of some input velocity; the only derived quantity is the
//! squared length `x*x + y*y`, computed in the golden's operation order (two
//! muls + one add, no fused multiply-add). The device output therefore equals
//! the golden bit-for-bit, so the parity tests assert exact equality
//! (`f32::to_bits` per component) rather than a tolerance.
//!
//! The golden [`TileVelocityField`](prism_render_architecture::motion::dilation::TileVelocityField)
//! exposes no public constructor from raw parts, so this module carries its own
//! [`TileField`] value that the twin can both *produce* (from `TileMax`) and
//! *consume* (as `NeighborMax` input) without reaching into the golden's
//! private storage.
//!
//! Provenance: Morgan `McGuire`, Padraic Hennessy, Michael Mara, Derek Nowrouzezahrai,
//! "A Reconstruction Filter for Plausible Motion Blur" (I3D 2012). Classical
//! integer-indexed reductions with float comparisons; no neural, learned, or
//! data-driven components. No Unreal Engine source or derived code.

use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::motion::Vec2;
use prism_render_architecture::motion::dilation::VelocityField;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, PipelineCompilationOptions,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::buffer;
use crate::context::GpuContext;
use crate::dilation::TILE;

/// `Pod` mirror of [`Vec2`] matching the `vec2<f32>` storage layout in the
/// shaders (two tightly-packed `f32`s, 8-byte stride).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuVec2 {
    x: f32,
    y: f32,
}

impl GpuVec2 {
    fn from_vec2(v: Vec2) -> GpuVec2 {
        GpuVec2 { x: v.x, y: v.y }
    }

    fn to_vec2(self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }
}

/// A coarse, per-tile velocity field — the twin's own equivalent of the golden
/// [`TileVelocityField`](prism_render_architecture::motion::dilation::TileVelocityField).
///
/// It exists because the golden type has no public constructor from parts, and
/// the twin must both emit a tile field (from `TileMax`) and accept one (as
/// `NeighborMax` input). The buffer is row-major with stride `tiles_x`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TileField {
    tile_size: usize,
    tiles_x: usize,
    tiles_y: usize,
    data: Vec<Vec2>,
}

impl TileField {
    /// Builds a tile field from raw parts, or [`None`] when `data.len()` does
    /// not equal `tiles_x * tiles_y`.
    #[must_use]
    pub fn from_parts(
        tile_size: usize,
        tiles_x: usize,
        tiles_y: usize,
        data: Vec<Vec2>,
    ) -> Option<TileField> {
        if data.len() == tiles_x * tiles_y {
            Some(TileField {
                tile_size,
                tiles_x,
                tiles_y,
                data,
            })
        } else {
            None
        }
    }

    /// Tile edge length in pixels.
    #[must_use]
    pub fn tile_size(&self) -> usize {
        self.tile_size
    }

    /// Number of tile columns.
    #[must_use]
    pub fn tiles_x(&self) -> usize {
        self.tiles_x
    }

    /// Number of tile rows.
    #[must_use]
    pub fn tiles_y(&self) -> usize {
        self.tiles_y
    }

    /// Total tile count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` when there are no tiles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Read-only view of the row-major tile buffer.
    #[must_use]
    pub fn as_slice(&self) -> &[Vec2] {
        &self.data
    }

    /// Reads tile `(tx, ty)`, or [`None`] when out of bounds.
    #[must_use]
    pub fn get(&self, tx: usize, ty: usize) -> Option<Vec2> {
        if tx < self.tiles_x && ty < self.tiles_y {
            Some(self.data[ty * self.tiles_x + tx])
        } else {
            None
        }
    }
}

/// Uniform block shared with `Params` in `tile_max.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TileMaxParams {
    width: u32,
    height: u32,
    tile_size: u32,
    tiles_x: u32,
    tiles_y: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// Uniform block shared with `Params` in `neighbor_max.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NeighborMaxParams {
    tiles_x: u32,
    tiles_y: u32,
    pad0: u32,
    pad1: u32,
}

/// Compiled `TileMax` pipeline and its bind-group layout.
pub struct GpuTileMax {
    pipeline: ComputePipeline,
    layout: BindGroupLayout,
}

impl GpuTileMax {
    /// Compiles the `TileMax` reduction kernel on `ctx`'s device.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTileMax {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_motion_tile_max"),
            source: ShaderSource::Wgsl(include_str!("../shaders/tile_max.wgsl").into()),
        });
        let layout = rw_reduction_layout(device, "prism_motion_tile_max_layout");
        let pipeline = reduction_pipeline(device, "prism_motion_tile_max", &module, &layout);
        GpuTileMax { pipeline, layout }
    }

    /// Reduces `velocity` to a per-tile maximum-magnitude field on the device.
    ///
    /// Returns [`None`] when `tile_size` is `0` or the field is empty, matching
    /// the golden [`tile_max`](prism_render_architecture::motion::dilation::tile_max)
    /// contract.
    #[must_use]
    pub fn reduce(
        &self,
        ctx: &GpuContext,
        velocity: &VelocityField,
        tile_size: usize,
    ) -> Option<TileField> {
        if tile_size == 0 || velocity.is_empty() {
            return None;
        }

        let width = velocity.width();
        let height = velocity.height();
        let tiles_x = width.div_ceil(tile_size);
        let tiles_y = height.div_ceil(tile_size);
        let tile_count = tiles_x * tiles_y;

        let device = ctx.device();
        let vel_in: Vec<GpuVec2> = velocity
            .as_slice()
            .iter()
            .map(|&v| GpuVec2::from_vec2(v))
            .collect();

        let params = TileMaxParams {
            width: u32::try_from(width).expect("field width fits in u32"),
            height: u32::try_from(height).expect("field height fits in u32"),
            tile_size: u32::try_from(tile_size).expect("tile_size fits in u32"),
            tiles_x: u32::try_from(tiles_x).expect("tiles_x fits in u32"),
            tiles_y: u32::try_from(tiles_y).expect("tiles_y fits in u32"),
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let params_buf = buffer::uniform(device, "prism_motion_tile_max_params", &params);
        let in_buf = buffer::storage_read(device, "prism_motion_tile_max_in", &vel_in);
        let out = dispatch_reduction(
            ctx,
            &self.pipeline,
            &self.layout,
            &params_buf,
            &in_buf,
            tile_count,
            params.tiles_x,
            params.tiles_y,
        );

        TileField::from_parts(tile_size, tiles_x, tiles_y, out)
    }
}

/// Compiled `NeighborMax` pipeline and its bind-group layout.
pub struct GpuNeighborMax {
    pipeline: ComputePipeline,
    layout: BindGroupLayout,
}

impl GpuNeighborMax {
    /// Compiles the `NeighborMax` expansion kernel on `ctx`'s device.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuNeighborMax {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_motion_neighbor_max"),
            source: ShaderSource::Wgsl(include_str!("../shaders/neighbor_max.wgsl").into()),
        });
        let layout = rw_reduction_layout(device, "prism_motion_neighbor_max_layout");
        let pipeline = reduction_pipeline(device, "prism_motion_neighbor_max", &module, &layout);
        GpuNeighborMax { pipeline, layout }
    }

    /// Expands `tiles` to its 3x3 neighborhood maximum on the device, mirroring
    /// the golden [`neighbor_max`](prism_render_architecture::motion::dilation::neighbor_max).
    ///
    /// An empty input yields an empty field of the same (zero) dimensions.
    #[must_use]
    pub fn expand(&self, ctx: &GpuContext, tiles: &TileField) -> TileField {
        if tiles.is_empty() {
            return TileField {
                tile_size: tiles.tile_size(),
                tiles_x: tiles.tiles_x(),
                tiles_y: tiles.tiles_y(),
                data: Vec::new(),
            };
        }

        let tiles_x = tiles.tiles_x();
        let tiles_y = tiles.tiles_y();
        let tile_count = tiles_x * tiles_y;

        let device = ctx.device();
        let src_in: Vec<GpuVec2> = tiles
            .as_slice()
            .iter()
            .map(|&v| GpuVec2::from_vec2(v))
            .collect();

        let params = NeighborMaxParams {
            tiles_x: u32::try_from(tiles_x).expect("tiles_x fits in u32"),
            tiles_y: u32::try_from(tiles_y).expect("tiles_y fits in u32"),
            pad0: 0,
            pad1: 0,
        };

        let params_buf = buffer::uniform(device, "prism_motion_neighbor_max_params", &params);
        let in_buf = buffer::storage_read(device, "prism_motion_neighbor_max_in", &src_in);
        let out = dispatch_reduction(
            ctx,
            &self.pipeline,
            &self.layout,
            &params_buf,
            &in_buf,
            tile_count,
            params.tiles_x,
            params.tiles_y,
        );

        TileField {
            tile_size: tiles.tile_size(),
            tiles_x,
            tiles_y,
            data: out,
        }
    }
}

/// Builds the bind-group layout shared by both reductions: a uniform `Params`
/// at binding 0, a read-only storage input at 1, and a read-write storage
/// output at 2.
fn rw_reduction_layout(device: &wgpu::Device, label: &str) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            buffer_layout(0, BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            }),
            buffer_layout(1, BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            }),
            buffer_layout(2, BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            }),
        ],
    })
}

/// Builds a `main`-entry compute pipeline for `module` against `layout`.
fn reduction_pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    layout: &BindGroupLayout,
) -> ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module,
        entry_point: Some("main"),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    })
}

/// Uploads the output buffer, binds all three buffers, dispatches one
/// invocation per output tile, and reads the tile field back as `Vec<Vec2>`.
#[expect(
    clippy::too_many_arguments,
    reason = "a single reduction dispatch legitimately needs the pipeline, layout, both buffers, and the output geometry"
)]
fn dispatch_reduction(
    ctx: &GpuContext,
    pipeline: &ComputePipeline,
    layout: &BindGroupLayout,
    params_buf: &Buffer,
    in_buf: &Buffer,
    tile_count: usize,
    tiles_x: u32,
    tiles_y: u32,
) -> Vec<Vec2> {
    let device = ctx.device();
    let out_bytes = (tile_count * size_of::<GpuVec2>()) as u64;
    let out_buf = buffer::storage_rw_zeroed(device, "prism_motion_reduction_out", out_bytes);
    let bind_group = reduction_bind_group(device, layout, params_buf, in_buf, &out_buf);

    let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("prism_motion_reduction_encoder"),
    });
    {
        let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism_motion_reduction_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(tiles_x.div_ceil(TILE), tiles_y.div_ceil(TILE), 1);
    }

    let stage = buffer::staging(device, "prism_motion_reduction_stage", out_bytes);
    buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
    ctx.queue().submit([enc.finish()]);

    let out_words = buffer::read_back::<GpuVec2>(ctx, &stage);
    out_words
        .iter()
        .take(tile_count)
        .map(|&w| w.to_vec2())
        .collect()
}

/// Binds `params`/`input`/`output` to the shared reduction layout.
fn reduction_bind_group(
    device: &wgpu::Device,
    layout: &BindGroupLayout,
    params_buf: &Buffer,
    in_buf: &Buffer,
    out_buf: &Buffer,
) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        label: Some("prism_motion_reduction_bind_group"),
        layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: in_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
        ],
    })
}

/// One storage/uniform bind-group-layout entry visible to the compute stage.
fn buffer_layout(binding: u32, ty: BindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}
