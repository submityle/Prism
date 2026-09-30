//! `wgpu` compute twin of the cloud level-of-detail classifier
//! ([`select_lod`](prism_render_architecture::volumetric::cloud_lod::select_lod)).
//!
//! Volumetric clouds reduce `raymarch` cost with distance: near clouds get the
//! full step count and buffer resolution, distant clouds get coarser marching,
//! and horizon clouds fall back to camera-facing imposters (design section 11).
//! This kernel maps a per-query view distance against three ascending distance
//! thresholds to one of four
//! [`CloudLod`](prism_render_architecture::volumetric::cloud_lod::CloudLod)
//! buckets:
//!
//! ```text
//! distance < mid_beyond      -> Near     (rank 0)
//! distance < far_beyond      -> Mid      (rank 1)
//! distance < imposter_beyond -> Far      (rank 2)
//! otherwise                  -> Imposter (rank 3)
//! ```
//!
//! The `CPU` golden
//! [`select_lod`](prism_render_architecture::volumetric::cloud_lod::select_lod)
//! owns that logic; [`GpuSelectLod`] is the on-device twin that runs one thread
//! per query and reproduces the same bucket.
//!
//! # Correctness model
//!
//! The classification is float comparisons against ascending thresholds, so the
//! decision is exact away from ties and the boundaries (`distance ==
//! threshold`, which selects the coarser bucket because the tests are strict
//! `<`) reproduce bit for bit. The parity test asserts every bucket matches the
//! `CPU` golden, so a degenerate kernel (dropped branch, `<=` instead of `<`)
//! could not pass.
//!
//! # Portability
//!
//! The kernel is float comparisons in the portable core-`WGSL` subset — no
//! `exp`, `pow` or optional device feature — so it runs unmodified on Metal,
//! Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard distance-bucketed cloud LOD selection plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.
use bytemuck::{Pod, Zeroable};
use prism_render_architecture::volumetric::cloud_lod::{CloudLod, CloudLodThresholds};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// One LOD query: the view distance and the thresholds to classify it against.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelectLodQuery {
    /// The view distance to classify. Any finite value is valid; negatives map
    /// to [`CloudLod::Near`] and very large values to [`CloudLod::Imposter`].
    pub distance: f32,
    /// The ascending distance thresholds separating the four buckets.
    pub thresholds: CloudLodThresholds,
}

/// Maps a shader bucket ordinal back to its [`CloudLod`].
fn lod_from_ordinal(ordinal: u32) -> CloudLod {
    match ordinal {
        0 => CloudLod::Near,
        1 => CloudLod::Mid,
        2 => CloudLod::Far,
        _ => CloudLod::Imposter,
    }
}

/// One query as uploaded. `16`-byte `repr(C)` matching `Query` in
/// `shaders/select_lod.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    distance: f32,
    mid_beyond: f32,
    far_beyond: f32,
    imposter_beyond: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/select_lod.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable cloud-LOD selection pipeline.
pub struct GpuSelectLod {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSelectLod {
    /// Compiles the cloud-LOD kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSelectLod {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_select_lod"),
            source: ShaderSource::Wgsl(include_str!("../shaders/select_lod.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_select_lod_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_select_lod_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_select_lod_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("select_lod_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSelectLod {
            module,
            layout,
            pipeline,
        }
    }

    /// Classifies every query in `queries`, returning one [`CloudLod`] per query
    /// in input order.
    ///
    /// The returned bucket for query `q` equals
    /// [`select_lod`](prism_render_architecture::volumetric::cloud_lod::select_lod)`(q.distance, q.thresholds)`
    /// exactly. An empty `queries` slice yields an empty result — storage
    /// buffers cannot be zero-sized, so it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[SelectLodQuery]) -> Vec<CloudLod> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                distance: q.distance,
                mid_beyond: q.thresholds.mid_beyond,
                far_beyond: q.thresholds.far_beyond,
                imposter_beyond: q.thresholds.imposter_beyond,
            })
            .collect();

        let gpu_params = Params {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (queries.len() as u64) * (size_of::<u32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_select_lod_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_select_lod_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_select_lod_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_select_lod_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_select_lod_bind_group"),
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
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_select_lod_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_select_lod_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (queries.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let ordinals = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(ordinals.len(), queries.len());
        ordinals.into_iter().map(lod_from_ordinal).collect()
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
