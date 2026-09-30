//! `wgpu` compute twin of the virtual-geometry LOD selector
//! ([`select_lod`](prism_render_architecture::virtual_geometry::select_lod)).
//!
//! Once a cluster survives culling, a GPU-driven virtual-geometry pipeline must
//! decide, per cluster, which discrete LOD level to display and which (equal or
//! finer) level to prefetch for imminent motion. The CPU golden
//! [`select_lod`](prism_render_architecture::virtual_geometry::select_lod) owns
//! that screen-space-error decision with hysteresis and velocity-scaled
//! prefetch; [`GpuLodSelector`] is the on-device twin that runs one thread per
//! cluster and returns the same
//! [`LodSelection`](prism_render_architecture::virtual_geometry::LodSelection)
//! the reference does.
//!
//! # Shared chain
//!
//! Every cluster in one [`select`](GpuLodSelector::select) dispatch shares a
//! single LOD chain: the common case is that all clusters of a mesh share its
//! level chain and differ only by view distance, closing speed and previously
//! displayed level. The chain need not be sorted; a level is identified by its
//! stored `level` index, not its array slot.
//!
//! # Portability
//!
//! The kernel is a fixed loop over the chain of multiply/divide-then-compare
//! operations in the portable core-`WGSL` subset, so it needs no optional
//! device feature and runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The emitted result is a pair of discrete level indices, not a continuous
//! value. The kernel mirrors the reference's projected-error term order
//! (`geometric_error.max(0) * focal / max(distance, EPSILON)`) and every branch
//! is an integer level compare or a float budget compare on identical operands,
//! so away from a razor-thin budget boundary the emitted level is identical to
//! the reference regardless of fused-multiply-add contraction. The parity test
//! picks distances and budgets that clear each boundary by a wide margin, so
//! the integer levels are stable under any legal float reassociation and can be
//! asserted index-for-index rather than with a tolerance.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard screen-space-error LOD selection with hysteresis and
//! velocity-scaled prefetch plus `wgpu` compute dispatch; no Unreal Engine
//! source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::virtual_geometry::{
    GeometryLodPolicy, LodLevel, LodProjection, LodSelection,
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

/// Uniform parameters for one LOD-select dispatch. Layout matches `Params` in
/// `shaders/lod_select.wesl`: four `f32` policy/projection scalars then the two
/// counts and two pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    focal_length_pixels: f32,
    target_error_pixels: f32,
    hysteresis_pixels: f32,
    prefetch_velocity_scale: f32,
    level_count: u32,
    cluster_count: u32,
    pad0: u32,
    pad1: u32,
}

/// One LOD chain entry. `8`-byte stride, matching `ChainLevel` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuChainLevel {
    level: u32,
    geometric_error: f32,
}

/// One cluster's LOD query inputs. `16`-byte stride, matching `Cluster` in the
/// shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuCluster {
    view_distance: f32,
    closing_speed: f32,
    has_previous: u32,
    previous_level: u32,
}

/// One cluster's selection result. `8`-byte stride, matching `Selection` in the
/// shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSelection {
    level: u32,
    prefetch_level: u32,
}

/// One cluster's per-cluster LOD query.
///
/// * `view_distance` — view-space distance to the cluster.
/// * `closing_speed` — component of relative velocity reducing that distance
///   (positive when approaching), in world units per selection step.
/// * `previous` — display level chosen on the prior step, used for hysteresis;
///   [`None`] on the first frame a cluster appears.
pub type LodQuery = (f32, f32, Option<u32>);

/// A compiled, reusable LOD-selection pipeline.
pub struct GpuLodSelector {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuLodSelector {
    /// Compiles the LOD-select kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so this never
    /// returns [`None`]: it compiles on any backend `ctx` acquired.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuLodSelector {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_lod_select"),
            source: ShaderSource::Wgsl(include_str!("../shaders/lod_select.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_lod_select_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_lod_select_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_lod_select_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("select"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuLodSelector {
            module,
            layout,
            pipeline,
        }
    }

    /// Selects the display and prefetch LODs for each cluster in `clusters`
    /// against the shared `chain`, returning one [`LodSelection`] per cluster in
    /// input order.
    ///
    /// The returned selection for query `q` equals
    /// [`select_lod`](prism_render_architecture::virtual_geometry::select_lod)`(chain, projection, q.0, q.1, policy, q.2).unwrap()`.
    /// An empty `chain` (the reference returns [`None`] for every query) or an
    /// empty `clusters` slice yields an empty result — storage buffers cannot be
    /// zero-sized, so both are handled by an early return.
    #[must_use]
    pub fn select(
        &self,
        ctx: &GpuContext,
        projection: LodProjection,
        policy: GeometryLodPolicy,
        chain: &[LodLevel],
        clusters: &[LodQuery],
    ) -> Vec<LodSelection> {
        if chain.is_empty() || clusters.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let params = Params {
            focal_length_pixels: projection.focal_length_pixels,
            target_error_pixels: policy.target_error_pixels,
            hysteresis_pixels: policy.hysteresis_pixels,
            prefetch_velocity_scale: policy.prefetch_velocity_scale,
            level_count: chain.len() as u32,
            cluster_count: clusters.len() as u32,
            pad0: 0,
            pad1: 0,
        };

        let gpu_chain: Vec<GpuChainLevel> = chain
            .iter()
            .map(|l| GpuChainLevel {
                level: l.level,
                geometric_error: l.geometric_error,
            })
            .collect();

        let gpu_clusters: Vec<GpuCluster> = clusters
            .iter()
            .map(|&(view_distance, closing_speed, previous)| {
                let (has_previous, previous_level) = match previous {
                    Some(level) => (1u32, level),
                    None => (0u32, 0u32),
                };
                GpuCluster {
                    view_distance,
                    closing_speed,
                    has_previous,
                    previous_level,
                }
            })
            .collect();

        let out_bytes = (clusters.len() as u64) * (size_of::<GpuSelection>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_lod_select_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let chain_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_lod_select_chain"),
            contents: bytemuck::cast_slice(&gpu_chain),
            usage: BufferUsages::STORAGE,
        });
        let clusters_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_lod_select_clusters"),
            contents: bytemuck::cast_slice(&gpu_clusters),
            usage: BufferUsages::STORAGE,
        });
        let selections_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_lod_select_selections"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let selections_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_lod_select_selections_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_lod_select_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: chain_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: clusters_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: selections_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_lod_select_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_lod_select_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (clusters.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&selections_buf, 0, &selections_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        selections_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = selections_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let gpu_selections = bytemuck::cast_slice::<u8, GpuSelection>(&view).to_vec();
        drop(view);
        selections_stage.unmap();
        debug_assert_eq!(gpu_selections.len(), clusters.len());
        gpu_selections
            .into_iter()
            .map(|s| LodSelection {
                level: s.level,
                prefetch_level: s.prefetch_level,
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
