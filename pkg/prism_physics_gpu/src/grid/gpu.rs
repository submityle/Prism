//! Real-device `wgpu` compute implementation of the bounded uniform grid.
//!
//! [`GpuUniformGrid`] compiles `shaders/grid_hash.wgsl` and
//! `shaders/grid_cell_ranges.wgsl` once and exposes [`GpuUniformGrid::build`],
//! which constructs the whole neighbourhood grid on the device in a single
//! submission: a hash pass writes each particle's dense linear cell index (and
//! an identity payload), the sibling [`GpuRadixSort`] stably sorts the particle
//! indices by cell entirely on device, and a "find cell start" pass scans the
//! sorted keys into per-cell `[start, end)` ranges. Only the three result
//! buffers are read back.
//!
//! The build is a pure integer permutation plus a range scan, so it matches the
//! [`cpu_grid_sort`](super::cpu::cpu_grid_sort) golden twin bit-for-bit.
//!
//! # On-device composition
//!
//! The hash pass produces the sort keys in device buffers that are handed
//! straight to [`GpuRadixSort::record_sort`], which records its four passes into
//! the same encoder without a host round-trip; the cell-ranges pass then reads
//! the sorted keys the radix sort left in place. The hash bind group holds
//! strong references to the key and payload buffers, so they stay valid after
//! their handles move into the sort.
//!
//! # Provenance
//!
//! The bounded uniform grid built by hashing to a linear cell index, sorting by
//! cell, and finding cell starts is the classical technique of Green, "Particle
//! Simulation using CUDA" (NVIDIA 2008). No Unreal Engine source or derived
//! code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BufferBindingType,
    CommandEncoder, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource,
};

use crate::buffer;
use crate::context::GpuContext;
use crate::radix::GpuRadixSort;

use super::config::{GridConfig, GridError};
use super::cpu::{GridBuild, EMPTY};
use super::layout::{buffer_entry, entry};

/// Lanes per workgroup for both grid kernels; must match `@workgroup_size`.
const WORKGROUP: u32 = 64;

/// Uniform parameters shared with `Params` in `shaders/grid_hash.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HashParams {
    /// Grid minimum corner, `x`.
    origin_x: f32,
    /// Grid minimum corner, `y`.
    origin_y: f32,
    /// Grid minimum corner, `z`.
    origin_z: f32,
    /// Uniform cell edge length.
    cell_size: f32,
    /// Cell count along `x`.
    nx: u32,
    /// Cell count along `y`.
    ny: u32,
    /// Cell count along `z`.
    nz: u32,
    /// Number of particles.
    n: u32,
}

/// Uniform parameters shared with `Params` in `shaders/grid_cell_ranges.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RangeParams {
    /// Number of sorted keys (particles).
    n: u32,
    /// Padding to a 16-byte boundary.
    pad0: u32,
    /// Padding to a 16-byte boundary.
    pad1: u32,
    /// Padding to a 16-byte boundary.
    pad2: u32,
}

/// A compiled, reusable `GPU` bounded uniform-grid builder.
pub struct GpuUniformGrid {
    /// Kept alive so the hash pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    hash_module: ShaderModule,
    /// Kept alive so the cell-ranges pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    ranges_module: ShaderModule,
    /// Layout wiring params, positions, and the key/payload outputs.
    hash_layout: BindGroupLayout,
    /// Layout wiring params, the sorted keys, and the cell range outputs.
    ranges_layout: BindGroupLayout,
    /// Writes each particle's cell index and identity payload.
    hash: ComputePipeline,
    /// Scans the sorted keys into per-cell `[start, end)` ranges.
    ranges: ComputePipeline,
    /// The sibling radix sort that orders the particle indices by cell.
    radix: GpuRadixSort,
}

impl GpuUniformGrid {
    /// Compiles the hash and cell-ranges kernels on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuUniformGrid {
        let device = ctx.device();

        let hash_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_grid_hash"),
            source: ShaderSource::Wgsl(include_str!("../shaders/grid_hash.wgsl").into()),
        });
        let ranges_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_grid_cell_ranges"),
            source: ShaderSource::Wgsl(include_str!("../shaders/grid_cell_ranges.wgsl").into()),
        });

        let hash_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_grid_hash_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let ranges_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_grid_cell_ranges_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });

        let hash_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_grid_hash_pipeline_layout"),
            bind_group_layouts: &[Some(&hash_layout)],
            immediate_size: 0,
        });
        let ranges_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_grid_cell_ranges_pipeline_layout"),
            bind_group_layouts: &[Some(&ranges_layout)],
            immediate_size: 0,
        });

        let hash = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_grid_hash_pipeline"),
            layout: Some(&hash_pipeline_layout),
            module: &hash_module,
            entry_point: Some("hash"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let ranges = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_grid_cell_ranges_pipeline"),
            layout: Some(&ranges_pipeline_layout),
            module: &ranges_module,
            entry_point: Some("cell_ranges"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        GpuUniformGrid {
            hash_module,
            ranges_module,
            hash_layout,
            ranges_layout,
            hash,
            ranges,
            radix: GpuRadixSort::new(ctx),
        }
    }

    /// Builds the uniform grid for `positions` under `config` on the `GPU`.
    ///
    /// Returns the stably sorted particle indices and the per-cell ranges into
    /// them, matching [`cpu_grid_sort`](super::cpu::cpu_grid_sort) bit-for-bit.
    /// An empty input skips the device and returns all-sentinel ranges.
    ///
    /// # Errors
    ///
    /// Returns [`GridError`] when `config` fails [`GridConfig::validate`].
    pub fn build(
        &self,
        ctx: &GpuContext,
        positions: &[Vec3],
        config: &GridConfig,
    ) -> Result<GridBuild, GridError> {
        config.validate()?;
        let n = positions.len();
        let num_cells = config.num_cells() as usize;
        if n == 0 {
            return Ok(GridBuild {
                sorted_indices: Vec::new(),
                cell_start: vec![EMPTY; num_cells],
                cell_end: vec![EMPTY; num_cells],
            });
        }

        let device = ctx.device();

        let packed: Vec<[f32; 4]> = positions.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();
        let pos_buf = buffer::storage_read(device, "prism_grid_positions", &packed);

        let key_bytes = (n * size_of::<u32>()) as u64;
        let keys_buf = buffer::storage_rw_zeroed(device, "prism_grid_keys", key_bytes);
        let idx_buf = buffer::storage_rw_zeroed(device, "prism_grid_indices", key_bytes);

        let empty_cells = vec![EMPTY; num_cells];
        let cell_bytes = size_of_val(empty_cells.as_slice()) as u64;
        let cell_start_buf = buffer::storage_rw_init(device, "prism_grid_cell_start", &empty_cells);
        let cell_end_buf = buffer::storage_rw_init(device, "prism_grid_cell_end", &empty_cells);

        let n_u32 = u32::try_from(n).unwrap_or(u32::MAX);
        let [nx, ny, nz] = config.dims;
        let hash_params = buffer::uniform(
            device,
            "prism_grid_hash_params",
            &HashParams {
                origin_x: config.origin.x,
                origin_y: config.origin.y,
                origin_z: config.origin.z,
                cell_size: config.cell_size,
                nx,
                ny,
                nz,
                n: n_u32,
            },
        );
        let range_params = buffer::uniform(
            device,
            "prism_grid_range_params",
            &RangeParams {
                n: n_u32,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            },
        );

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_grid_encoder"),
        });

        // Hash pass: write per-particle cell keys and identity payloads. The
        // bind group holds strong references to `keys_buf`/`idx_buf`, so they
        // stay valid after their handles move into the sort below.
        let groups = n_u32.div_ceil(WORKGROUP);
        let hash_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_grid_hash_bind"),
            layout: &self.hash_layout,
            entries: &[
                entry(0, &hash_params),
                entry(1, &pos_buf),
                entry(2, &keys_buf),
                entry(3, &idx_buf),
            ],
        });
        dispatch(
            &mut encoder,
            "prism_grid_hash_pass",
            &self.hash,
            &hash_bind,
            groups,
        );

        // Sort particle indices by cell on device, consuming the key/payload
        // buffers and leaving the sorted keys in `sorted.keys()`.
        let sorted = self
            .radix
            .record_sort(device, &mut encoder, keys_buf, idx_buf, n);

        // Find cell start/end from the sorted keys.
        let range_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_grid_cell_ranges_bind"),
            layout: &self.ranges_layout,
            entries: &[
                entry(0, &range_params),
                entry(1, sorted.keys()),
                entry(2, &cell_start_buf),
                entry(3, &cell_end_buf),
            ],
        });
        dispatch(
            &mut encoder,
            "prism_grid_cell_ranges_pass",
            &self.ranges,
            &range_bind,
            groups,
        );

        let idx_stage = buffer::staging(device, "prism_grid_indices_stage", key_bytes);
        let start_stage = buffer::staging(device, "prism_grid_cell_start_stage", cell_bytes);
        let end_stage = buffer::staging(device, "prism_grid_cell_end_stage", cell_bytes);
        buffer::copy(&mut encoder, sorted.values(), &idx_stage, key_bytes);
        buffer::copy(&mut encoder, &cell_start_buf, &start_stage, cell_bytes);
        buffer::copy(&mut encoder, &cell_end_buf, &end_stage, cell_bytes);
        ctx.queue().submit([encoder.finish()]);

        let sorted_indices = buffer::read_back::<u32>(ctx, &idx_stage);
        let cell_start = buffer::read_back::<u32>(ctx, &start_stage);
        let cell_end = buffer::read_back::<u32>(ctx, &end_stage);
        drop(sorted);
        drop(hash_bind);
        drop(range_bind);

        Ok(GridBuild {
            sorted_indices,
            cell_start,
            cell_end,
        })
    }
}

/// Records one block-granular dispatch of `pipeline` bound to `bind`.
fn dispatch(
    encoder: &mut CommandEncoder,
    label: &str,
    pipeline: &ComputePipeline,
    bind: &BindGroup,
    groups: u32,
) {
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind, &[]);
    pass.dispatch_workgroups(groups, 1, 1);
}
