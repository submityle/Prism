//! `wgpu` compute twin of Prism's per-cluster strand cull verdict
//! ([`cluster_cull_verdict`](prism_render_architecture::hair::cluster::cluster_cull_verdict)).
//!
//! A groom is grouped into spatially-local *strand clusters* (the Nanite-style
//! coarse culling unit). Before the groom is rasterized, every cluster is tested
//! against the view in a fixed order — frustum, then back-facing, then occlusion
//! — and the first failing test decides its verdict. The `CPU` golden for that
//! per-cluster decision is
//! [`cluster_cull_verdict`](prism_render_architecture::hair::cluster::cluster_cull_verdict);
//! this module is the on-device twin that runs one thread per cluster over the
//! same closed-form tests, so a passing real-device parity test is direct
//! evidence the ported kernel reaches the same visible flag and deciding reason
//! as the reference — not merely that its shader compiles.
//!
//! Only the per-cluster *verdict* is ported. The clustering itself (grid
//! assignment, bounds accumulation, mean-tangent reduction) is a reduction over
//! strands and stays on the `CPU`
//! ([`cluster_strands`](prism_render_architecture::hair::cluster::cluster_strands));
//! this kernel consumes the already-built [`StrandCluster`] bounds and mean
//! tangent, which is the GPU-driven culling hot path.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairClusterCull::eval`] takes the same inputs as a batch of
//! `cluster_cull_verdict` calls — the clusters, the shared [`Frustum`], the
//! [`ClusterView`], a per-cluster optional [`OcclusionProbe`], and the
//! back-facing cosine bias — and returns one [`ClusterCullVerdict`] per cluster.
//! The three tests mirror the reference exactly: an invalid (un-grown) bounds is
//! treated as outside the frustum; the projected-radius plane test passes a box
//! only when its centre is within the projected half-extent of all six inward
//! planes; the back-facing test normalises both the mean tangent and the to-eye
//! direction defensively (a degenerate vector yields a dot of `0`, which never
//! trips a positive bias); the occlusion phase runs only when a probe is present
//! and culls when `closest_depth > occluder_depth`.
//!
//! # Device-side faithfulness
//!
//! The bounds corners are uploaded **raw** (including an empty box's `+inf` /
//! `-inf`), and the device recomputes `is_valid` / `center` / `half_extents`
//! exactly as [`Aabb`](prism_render_architecture::hair::cluster::Aabb) does, so
//! the empty-box and degenerate paths are exercised on the `GPU` too rather than
//! being short-circuited on the host. The reason encoding (`0` = Visible,
//! `1` = Frustum, `2` = Backface, `3` = Occlusion) is shared by both sides.
//!
//! A no-op call (no clusters) returns an empty vector without a dispatch —
//! storage buffers cannot be zero-sized.
//!
//! # Portability
//!
//! The verdict uses only `dot`, `sqrt`, `abs`, `clamp`, `min`, `max` and
//! multiply/add in the portable core-`WGSL` subset — no `exp`, `pow` or optional
//! device feature — so the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The verdict is an integer (a visible flag plus a reason code), so the parity
//! test compares it **bit-exact**. The only floating-point work is the plane and
//! facing dot products, which a `GPU` may fuse into a multiply-add the scalar
//! reference leaves separate; the parity cases therefore keep every decision
//! well clear of its threshold so that legal contraction cannot flip a verdict.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard GPU-driven cluster culling (frustum projected-radius +
//! tangent back-face + Hi-Z occlusion) plus `wgpu` compute dispatch; no Unreal
//! Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::cluster::{
    ClusterCullVerdict, ClusterView, CullReason, Frustum, OcclusionProbe, StrandCluster,
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

/// Shared view parameters uploaded to the kernel. `128`-byte scalar-packed
/// `repr(C)` matching `Params` in `shaders/cluster_cull.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Six inward-facing frustum planes: `[nx, ny, nz, distance]`.
    planes: [[f32; 4]; 6],
    /// Eye position `xyz` packed with the raw back-facing bias in `w`.
    eye_bias: [f32; 4],
    /// Number of cluster descriptors in the clusters buffer.
    cluster_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One cluster descriptor uploaded to the kernel. `64`-byte `repr(C)` matching
/// `Cluster` in `shaders/cluster_cull.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Cluster {
    bounds_min: [f32; 3],
    pad_a: f32,
    bounds_max: [f32; 3],
    pad_b: f32,
    mean_tangent: [f32; 3],
    has_occlusion: u32,
    closest_depth: f32,
    occluder_depth: f32,
    pad_c: f32,
    pad_d: f32,
}

/// One verdict read back from the kernel. `8`-byte `repr(C)` matching `Verdict`
/// in `shaders/cluster_cull.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Verdict {
    visible: u32,
    reason: u32,
}

/// A compiled, reusable per-cluster cull pipeline.
pub struct GpuHairClusterCull {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairClusterCull {
    /// Compiles the per-cluster cull kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairClusterCull {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_cluster_cull"),
            source: ShaderSource::Wgsl(include_str!("../shaders/cluster_cull.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_cluster_cull_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_cluster_cull_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_cluster_cull_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairClusterCull {
            module,
            layout,
            pipeline,
        }
    }

    /// Culls a batch of clusters against the view, returning one
    /// [`ClusterCullVerdict`] per cluster.
    ///
    /// The result equals
    /// [`cluster_cull_verdict`](prism_render_architecture::hair::cluster::cluster_cull_verdict)
    /// applied to each cluster with the same `frustum`, `view`, per-cluster
    /// occlusion probe, and `backface_bias`. The `occlusion` slice is indexed by
    /// cluster: a `None` entry (or an index past its end) skips the occlusion
    /// phase for that cluster, exactly as passing `None` to the reference does.
    /// Because the verdict is integer, it is reproduced bit-exact.
    ///
    /// A no-op call (no clusters) returns an empty vector without a dispatch —
    /// storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        clusters: &[StrandCluster],
        frustum: &Frustum,
        view: ClusterView,
        occlusion: &[Option<OcclusionProbe>],
        backface_bias: f32,
    ) -> Vec<ClusterCullVerdict> {
        if clusters.is_empty() {
            return Vec::new();
        }

        let descriptors: Vec<Cluster> = clusters
            .iter()
            .enumerate()
            .map(|(i, cluster)| {
                let probe = occlusion.get(i).copied().flatten();
                let (has_occlusion, closest_depth, occluder_depth) = match probe {
                    Some(p) => (1u32, p.closest_depth, p.occluder_depth),
                    None => (0u32, 0.0, 0.0),
                };
                Cluster {
                    bounds_min: cluster.bounds.min,
                    pad_a: 0.0,
                    bounds_max: cluster.bounds.max,
                    pad_b: 0.0,
                    mean_tangent: cluster.mean_tangent,
                    has_occlusion,
                    closest_depth,
                    occluder_depth,
                    pad_c: 0.0,
                    pad_d: 0.0,
                }
            })
            .collect();

        let mut planes = [[0.0f32; 4]; 6];
        for (slot, plane) in planes.iter_mut().zip(frustum.planes.iter()) {
            *slot = [
                plane.normal[0],
                plane.normal[1],
                plane.normal[2],
                plane.distance,
            ];
        }
        let uniform = Params {
            planes,
            eye_bias: [view.eye[0], view.eye[1], view.eye[2], backface_bias],
            cluster_count: descriptors.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let device = ctx.device();
        let verdict_bytes = (descriptors.len() * size_of::<Verdict>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_cluster_cull_params"),
            contents: bytemuck::bytes_of(&uniform),
            usage: BufferUsages::UNIFORM,
        });
        let clusters_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_cluster_cull_clusters"),
            contents: bytemuck::cast_slice(&descriptors),
            usage: BufferUsages::STORAGE,
        });
        let verdicts_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_cluster_cull_verdicts"),
            size: verdict_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let verdicts_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_cluster_cull_verdicts_stage"),
            size: verdict_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_cluster_cull_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: clusters_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: verdicts_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_cluster_cull_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_cluster_cull_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (descriptors.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&verdicts_buf, 0, &verdicts_stage, 0, verdict_bytes);
        ctx.queue().submit([encoder.finish()]);

        verdicts_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let readback = {
            let view = verdicts_stage
                .slice(..)
                .get_mapped_range()
                .expect("mapped readback range should be available after poll");
            let verdicts = bytemuck::cast_slice::<u8, Verdict>(&view).to_vec();
            drop(view);
            verdicts
        };
        verdicts_stage.unmap();
        debug_assert_eq!(readback.len(), descriptors.len());

        readback
            .iter()
            .map(|v| {
                if v.visible != 0 {
                    ClusterCullVerdict::visible()
                } else {
                    ClusterCullVerdict::culled(decode_reason(v.reason))
                }
            })
            .collect()
    }
}

/// Decodes the kernel's `reason` code into a [`CullReason`], matching the
/// shader's encoding (`1` = Frustum, `2` = Backface, `3` = Occlusion; any other
/// value, including `0`, is [`CullReason::Visible`]).
fn decode_reason(code: u32) -> CullReason {
    match code {
        1 => CullReason::Frustum,
        2 => CullReason::Backface,
        3 => CullReason::Occlusion,
        _ => CullReason::Visible,
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
