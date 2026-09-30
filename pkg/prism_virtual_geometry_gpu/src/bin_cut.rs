//! `wgpu` compute twin of the virtual-geometry raster-bin partition
//! ([`bin_cut`](prism_render_architecture::virtual_geometry::bin_cut)).
//!
//! Once the hierarchy has resolved which cluster nodes to draw this frame, the
//! draw prep must fan the selected cut out into the four GPU raster-path
//! buckets so the backend can emit one indirect batch per path. The CPU golden
//! [`bin_cut`](prism_render_architecture::virtual_geometry::bin_cut) owns that
//! partition; [`GpuCutBinner`] is the on-device twin that produces a
//! bit-identical, order-preserving partition of the same cut.
//!
//! # Partition model
//!
//! The kernel returns a *permutation* of the input cut indices plus the four
//! bucket counts, which [`GpuCutBinner::bin`] slices to rebuild the same
//! [`RasterBins`](prism_render_architecture::virtual_geometry::RasterBins) the
//! reference produces. The partition is a stable multi-bucket split computed in
//! three passes over shared storage: a per-element `classify`, a single serial
//! `scan` that mirrors the reference's sequential push to assign within-bucket
//! ranks and bucket counts, and a per-element `scatter` that places each
//! element at `base[bucket] + rank`. The `CutCluster` payload never enters the
//! shader - only node indices and statistics do - so the host reattaches each
//! cluster from the returned permutation.
//!
//! # Portability
//!
//! The kernel is integer branching plus a single `<=` comparison of the
//! per-cluster `max_triangle_pixels` against the clamped threshold, in the
//! portable core-WGSL subset, so it needs no optional device feature and runs
//! unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The bucket decision performs no floating-point arithmetic beyond a
//! `max(threshold, 0.0)` clamp and a single `<=` compare on identical operands,
//! so every bucket assignment is bit-exact against the reference regardless of
//! fused-multiply-add contraction, and the emitted permutation is a discrete
//! index array with no tolerance. The parity test asserts bucket-for-bucket,
//! element-for-element equality against
//! [`bin_cut`](prism_render_architecture::virtual_geometry::bin_cut).
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard stable multi-bucket partition and cluster raster-path
//! selection heuristic and `wgpu` compute dispatch; no Unreal Engine source or
//! derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::virtual_geometry::{
    ClusterRasterStats, CutCluster, RasterBins, RasterCapability,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Capability bit for an available mesh/amplification shader pipeline. Matches
/// `CAP_MESH_SHADER` in `shaders/bin_cut.wesl`.
const CAP_MESH_SHADER: u32 = 1;
/// Capability bit for available hardware indirect (multi-)draw. Matches
/// `CAP_HARDWARE_INDIRECT` in `shaders/bin_cut.wesl`.
const CAP_HARDWARE_INDIRECT: u32 = 2;

/// Uniform parameters for one bin dispatch. Layout matches `Params` in
/// `shaders/bin_cut.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    stats_count: u32,
    capability_bits: u32,
    threshold: f32,
}

/// One cluster's raster statistics. `8`-byte stride, matching `ClusterStats` in
/// the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuClusterStats {
    max_triangle_pixels: f32,
    triangle_count: u32,
}

/// A compiled, reusable raster-bin partition pipeline.
///
/// The three passes (`classify`, `scan`, `scatter`) share one bind-group layout
/// over the shared storage buffers.
pub struct GpuCutBinner {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    classify: ComputePipeline,
    scan: ComputePipeline,
    scatter: ComputePipeline,
}

impl GpuCutBinner {
    /// Compiles the three-pass bin kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-WGSL subset, so unlike
    /// [`GpuPayloadRaster::new`](crate::GpuPayloadRaster::new) this never
    /// returns [`None`]: it compiles on any backend `ctx` acquired.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCutBinner {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_bin_cut"),
            source: ShaderSource::Wgsl(include_str!("../shaders/bin_cut.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_bin_cut_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
                buffer_entry(5, BufferBindingType::Storage { read_only: false }),
                buffer_entry(6, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_bin_cut_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let make = |entry: &str, label: &str| {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let classify = make("classify", "prism_bin_cut_classify");
        let scan = make("scan", "prism_bin_cut_scan");
        let scatter = make("scatter", "prism_bin_cut_scatter");
        GpuCutBinner {
            module,
            layout,
            classify,
            scan,
            scatter,
        }
    }

    /// Partitions `cut` into per-path raster buckets under `capability` and
    /// `software_pixel_threshold`, returning the same
    /// [`RasterBins`](prism_render_architecture::virtual_geometry::RasterBins)
    /// as [`bin_cut`](prism_render_architecture::virtual_geometry::bin_cut).
    ///
    /// Each [`CutCluster::node`] indexes `raster_stats`; a cluster whose node
    /// index is out of range is skipped rather than routed, and per-bucket
    /// order matches the input cut order.
    #[must_use]
    pub fn bin(
        &self,
        ctx: &GpuContext,
        cut: &[CutCluster],
        raster_stats: &[ClusterRasterStats],
        capability: RasterCapability,
        software_pixel_threshold: f32,
    ) -> RasterBins {
        // Empty inputs cannot allocate zero-sized storage; both cases route
        // every element to nothing, so the reference returns the default.
        if cut.is_empty() || raster_stats.is_empty() {
            return RasterBins::default();
        }
        let device = ctx.device();

        let mut capability_bits = 0u32;
        if capability.mesh_shader {
            capability_bits |= CAP_MESH_SHADER;
        }
        if capability.hardware_indirect {
            capability_bits |= CAP_HARDWARE_INDIRECT;
        }

        let params = Params {
            count: cut.len() as u32,
            stats_count: raster_stats.len() as u32,
            capability_bits,
            threshold: software_pixel_threshold,
        };

        let node_index: Vec<u32> = cut.iter().map(|c| c.node).collect();
        let gpu_stats: Vec<GpuClusterStats> = raster_stats
            .iter()
            .map(|s| GpuClusterStats {
                max_triangle_pixels: s.max_triangle_pixels,
                triangle_count: s.triangle_count,
            })
            .collect();

        let count = cut.len() as u64;
        let index_bytes = count * 4;
        let counts_bytes = 4u64 * 4;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_bin_cut_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let stats_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_bin_cut_stats"),
            contents: bytemuck::cast_slice(&gpu_stats),
            usage: BufferUsages::STORAGE,
        });
        let node_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_bin_cut_node_index"),
            contents: bytemuck::cast_slice(&node_index),
            usage: BufferUsages::STORAGE,
        });
        let bucket_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_bin_cut_bucket"),
            size: index_bytes,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let rank_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_bin_cut_rank"),
            size: index_bytes,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let counts_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_bin_cut_counts"),
            size: counts_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_order_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_bin_cut_out_order"),
            size: index_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let counts_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_bin_cut_counts_stage"),
            size: counts_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_order_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_bin_cut_out_order_stage"),
            size: index_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_bin_cut_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: stats_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: node_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: bucket_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: rank_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: counts_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: out_order_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_bin_cut_encoder"),
        });
        let groups = (cut.len() as u32).div_ceil(64);
        // Separate compute passes so wgpu inserts the storage read/write
        // barriers that make each pass's writes visible to the next.
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_bin_cut_classify_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.classify);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_bin_cut_scan_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scan);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_bin_cut_scatter_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scatter);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&counts_buf, 0, &counts_stage, 0, counts_bytes);
        encoder.copy_buffer_to_buffer(&out_order_buf, 0, &out_order_stage, 0, index_bytes);
        ctx.queue().submit([encoder.finish()]);

        counts_stage.slice(..).map_async(MapMode::Read, |_| {});
        out_order_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let counts_view = counts_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let counts = bytemuck::cast_slice::<u8, u32>(&counts_view).to_vec();
        drop(counts_view);
        counts_stage.unmap();

        let order_view = out_order_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let out_order = bytemuck::cast_slice::<u8, u32>(&order_view).to_vec();
        drop(order_view);
        out_order_stage.unmap();

        rebuild_bins(cut, &out_order, &counts)
    }
}

/// Rebuilds [`RasterBins`] from the emitted permutation and per-bucket counts.
///
/// `out_order` is a permutation of the in-range cut indices, grouped in the
/// fixed bucket order (mesh shader, compute software, indirect hardware,
/// fallback mesh); `counts[k]` is the length of bucket `k`. Each bucket's slice
/// of `out_order` is mapped back through `cut` to recover the original
/// [`CutCluster`] payloads in cut order.
fn rebuild_bins(cut: &[CutCluster], out_order: &[u32], counts: &[u32]) -> RasterBins {
    debug_assert_eq!(counts.len(), 4);
    let mut bins = RasterBins::default();
    let mut base = 0usize;
    let buckets: [&mut Vec<CutCluster>; 4] = [
        &mut bins.mesh_shader,
        &mut bins.compute_software,
        &mut bins.indirect_hardware,
        &mut bins.fallback_mesh,
    ];
    for (k, dst) in buckets.into_iter().enumerate() {
        let len = counts[k] as usize;
        for &idx in &out_order[base..base + len] {
            dst.push(cut[idx as usize]);
        }
        base += len;
    }
    bins
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
