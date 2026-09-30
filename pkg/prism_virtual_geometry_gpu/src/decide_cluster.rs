//! `wgpu` compute twin of the fused per-cluster decision
//! ([`ViewCullContext::decide`](prism_render_architecture::virtual_geometry::ViewCullContext::decide)).
//!
//! A GPU-driven virtual-geometry pipeline decides every candidate cluster once
//! per frame: cull first, and only a surviving cluster derives a streaming
//! priority, selects a LOD (display + prefetch) and picks a raster path. The CPU
//! golden
//! [`ViewCullContext::decide`](prism_render_architecture::virtual_geometry::ViewCullContext::decide)
//! owns that composition; [`GpuClusterDecider`] is the on-device twin that runs
//! one thread per cluster and returns the same
//! [`ClusterDecision`](prism_render_architecture::virtual_geometry::ClusterDecision)
//! the reference does, minus the host-side page-table residency bookkeeping the
//! reference performs (mutable state that stays on the host).
//!
//! # Shared chain
//!
//! Every cluster in one [`decide`](GpuClusterDecider::decide) dispatch shares a
//! single LOD chain, exactly like
//! [`GpuLodSelector`](crate::GpuLodSelector): the common case is that all
//! clusters of a mesh share its level chain and differ only by view distance,
//! closing speed and previously displayed level. The reference's per-request
//! `lods` slice is therefore supplied once here as the shared `chain`.
//!
//! # Portability
//!
//! The kernel is a fixed loop of integer compares and multiply/divide-then-
//! compare operations in the portable core-`WGSL` subset, so it needs no
//! optional device feature and runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The emitted decision is discrete (a cull verdict, an optional pair of level
//! indices, an optional raster-path index) plus a single streaming priority.
//! The kernel mirrors the reference's frustum term order, projected-error term
//! order (`geometric_error.max(0) * focal / max(distance, EPSILON)`), hysteresis
//! level pick and raster-path cascade, so away from a razor-thin budget boundary
//! every discrete decision is identical to the reference regardless of
//! fused-multiply-add contraction. Priority is a single multiply-then-divide of
//! correctly-rounded IEEE operations; the parity test asserts it within 1 ULP
//! and asserts the discrete fields index-for-index with distances/budgets that
//! clear each boundary by a wide margin.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard frustum/Hi-Z culling, screen-space-error LOD selection
//! and software/hardware raster classification plus `wgpu` compute dispatch; no
//! Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::gpu_scene::SceneBounds;
use prism_render_architecture::virtual_geometry::{
    ClusterDecision, ClusterRasterStats, CullVerdict, GeometryRasterPath, LodLevel, LodSelection,
    OcclusionProbe, ViewCullContext,
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

/// Uniform parameters for one decide dispatch. Layout matches `Params` in
/// `shaders/decide_cluster.wesl`: six `vec4` frustum planes, five `f32`
/// policy/projection/threshold scalars, four `u32` capability/count words then
/// three pad words (`144` bytes total).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    planes: [[f32; 4]; 6],
    focal_length_pixels: f32,
    target_error_pixels: f32,
    hysteresis_pixels: f32,
    prefetch_velocity_scale: f32,
    software_pixel_threshold: f32,
    mesh_shader: u32,
    hardware_indirect: u32,
    level_count: u32,
    cluster_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One LOD chain entry. `8`-byte stride, matching `ChainLevel` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuChainLevel {
    level: u32,
    geometric_error: f32,
}

/// One cluster's decision inputs. `64`-byte stride, matching `Cluster` in the
/// shader (`center`/`half_extents` are `vec3<f32>` whose trailing padding word
/// carries the following scalar).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuCluster {
    center: [f32; 3],
    radius: f32,
    half_extents: [f32; 3],
    closest_depth: f32,
    occluder_depth: f32,
    has_occlusion: u32,
    view_distance: f32,
    closing_speed: f32,
    max_triangle_pixels: f32,
    triangle_count: u32,
    has_previous: u32,
    previous_level: u32,
}

/// One cluster's fused decision result. `32`-byte stride, matching `Decision`
/// in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuDecision {
    verdict: u32,
    has_lod: u32,
    level: u32,
    prefetch_level: u32,
    raster_path: u32,
    priority: f32,
    pad0: u32,
    pad1: u32,
}

/// One cluster's decision inputs for [`GpuClusterDecider::decide`].
///
/// This mirrors the per-cluster fields of
/// [`ClusterRequest`](prism_render_architecture::virtual_geometry::ClusterRequest)
/// that the fused decision reads, minus the host-only page identity (residency
/// bookkeeping stays on the host and is not part of the returned decision). The
/// LOD chain is supplied once per dispatch as the shared `chain`, since every
/// cluster of a mesh shares its level chain.
#[derive(Clone, Copy, Debug)]
pub struct ClusterDecisionInput {
    /// World-space bounds for culling and screen-size priority.
    pub bounds: SceneBounds,
    /// Cluster raster statistics for path classification.
    pub raster_stats: ClusterRasterStats,
    /// View-space distance to the cluster.
    pub view_distance: f32,
    /// Closing speed (positive when approaching) for LOD prefetch.
    pub closing_speed: f32,
    /// Optional occlusion probe over the cluster footprint.
    pub occlusion: Option<OcclusionProbe>,
    /// LOD chosen for this cluster last frame, for hysteresis.
    pub previous_lod: Option<u32>,
}

/// A compiled, reusable fused-decision pipeline.
pub struct GpuClusterDecider {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuClusterDecider {
    /// Compiles the decide kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-WGSL subset, so it compiles on any
    /// backend `ctx` acquired and never returns [`None`].
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClusterDecider {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_decide_cluster"),
            source: ShaderSource::Wgsl(include_str!("../shaders/decide_cluster.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_decide_cluster_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_decide_cluster_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_decide_cluster_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("decide"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuClusterDecider {
            module,
            layout,
            pipeline,
        }
    }

    /// Decides each cluster in `clusters` against the shared `chain` and the
    /// per-view state in `view_ctx`, returning one [`ClusterDecision`] per
    /// cluster in input order.
    ///
    /// The returned decision for cluster `c` equals
    /// [`ViewCullContext::decide`](prism_render_architecture::virtual_geometry::ViewCullContext::decide)
    /// for a
    /// [`ClusterRequest`](prism_render_architecture::virtual_geometry::ClusterRequest)
    /// built from `c` with `lods = chain`, discarding the host-side page-table
    /// mutation the reference performs. An empty `chain` yields decisions whose
    /// `lod` is [`None`] (matching the reference's `select_lod` on an empty
    /// chain) while still carrying the verdict, raster path and priority. An
    /// empty `clusters` slice yields an empty result — storage buffers cannot be
    /// zero-sized, so it is handled by an early return.
    #[must_use]
    pub fn decide(
        &self,
        ctx: &GpuContext,
        view_ctx: &ViewCullContext,
        chain: &[LodLevel],
        clusters: &[ClusterDecisionInput],
    ) -> Vec<ClusterDecision> {
        if clusters.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let mut planes = [[0.0f32; 4]; 6];
        for (dst, plane) in planes.iter_mut().zip(view_ctx.frustum.planes.iter()) {
            *dst = [
                plane.normal[0],
                plane.normal[1],
                plane.normal[2],
                plane.distance,
            ];
        }

        let params = Params {
            planes,
            focal_length_pixels: view_ctx.projection.focal_length_pixels,
            target_error_pixels: view_ctx.lod_policy.target_error_pixels,
            hysteresis_pixels: view_ctx.lod_policy.hysteresis_pixels,
            prefetch_velocity_scale: view_ctx.lod_policy.prefetch_velocity_scale,
            software_pixel_threshold: view_ctx.software_pixel_threshold,
            mesh_shader: u32::from(view_ctx.raster_capability.mesh_shader),
            hardware_indirect: u32::from(view_ctx.raster_capability.hardware_indirect),
            level_count: chain.len() as u32,
            cluster_count: clusters.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // Storage buffers cannot be zero-sized. An empty chain uploads one inert
        // entry the kernel never reads: `level_count == 0` short-circuits every
        // chain loop and the `level_count > 0` LOD guard, so `has_lod` stays `0`
        // and the returned `lod` is `None`, matching the reference.
        let gpu_chain: Vec<GpuChainLevel> = if chain.is_empty() {
            vec![GpuChainLevel {
                level: 0,
                geometric_error: 0.0,
            }]
        } else {
            chain
                .iter()
                .map(|l| GpuChainLevel {
                    level: l.level,
                    geometric_error: l.geometric_error,
                })
                .collect()
        };

        let gpu_clusters: Vec<GpuCluster> = clusters
            .iter()
            .map(|c| {
                let (has_occlusion, closest_depth, occluder_depth) = match c.occlusion {
                    Some(p) => (1u32, p.closest_depth, p.occluder_depth),
                    None => (0u32, 0.0, 0.0),
                };
                let (has_previous, previous_level) = match c.previous_lod {
                    Some(level) => (1u32, level),
                    None => (0u32, 0u32),
                };
                GpuCluster {
                    center: c.bounds.center,
                    radius: c.bounds.radius,
                    half_extents: c.bounds.half_extents,
                    closest_depth,
                    occluder_depth,
                    has_occlusion,
                    view_distance: c.view_distance,
                    closing_speed: c.closing_speed,
                    max_triangle_pixels: c.raster_stats.max_triangle_pixels,
                    triangle_count: c.raster_stats.triangle_count,
                    has_previous,
                    previous_level,
                }
            })
            .collect();

        let out_bytes = (clusters.len() as u64) * (size_of::<GpuDecision>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_decide_cluster_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let chain_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_decide_cluster_chain"),
            contents: bytemuck::cast_slice(&gpu_chain),
            usage: BufferUsages::STORAGE,
        });
        let clusters_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_decide_cluster_clusters"),
            contents: bytemuck::cast_slice(&gpu_clusters),
            usage: BufferUsages::STORAGE,
        });
        let decisions_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_decide_cluster_decisions"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let decisions_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_decide_cluster_decisions_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_decide_cluster_bind_group"),
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
                    resource: decisions_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_decide_cluster_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_decide_cluster_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (clusters.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&decisions_buf, 0, &decisions_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        decisions_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = decisions_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let gpu_decisions = bytemuck::cast_slice::<u8, GpuDecision>(&view).to_vec();
        drop(view);
        decisions_stage.unmap();
        debug_assert_eq!(gpu_decisions.len(), clusters.len());
        gpu_decisions
            .into_iter()
            .map(rebuild_decision)
            .collect()
    }
}

/// Rebuilds the reference [`ClusterDecision`] from one GPU decision record.
fn rebuild_decision(d: GpuDecision) -> ClusterDecision {
    let verdict = match d.verdict {
        0 => CullVerdict::Visible,
        1 => CullVerdict::FrustumCulled,
        2 => CullVerdict::OcclusionCulled,
        other => unreachable!("kernel emits verdict in 0..=2, got {other}"),
    };
    let lod = if d.has_lod == 1 {
        Some(LodSelection {
            level: d.level,
            prefetch_level: d.prefetch_level,
        })
    } else {
        None
    };
    let raster_path = if verdict == CullVerdict::Visible {
        Some(match d.raster_path {
            0 => GeometryRasterPath::MeshShader,
            1 => GeometryRasterPath::ComputeSoftware,
            2 => GeometryRasterPath::IndirectHardware,
            3 => GeometryRasterPath::FallbackMesh,
            other => unreachable!("kernel emits raster path in 0..=3, got {other}"),
        })
    } else {
        None
    };
    ClusterDecision {
        verdict,
        lod,
        raster_path,
        priority: d.priority,
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
