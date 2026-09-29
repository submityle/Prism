//! Real-device `wgpu` compute implementation of the broad phase.
//!
//! [`GpuBroadphase`] compiles `shaders/broadphase.wgsl` once and exposes
//! [`GpuBroadphase::run`], which uploads the particle spheres, dispatches the
//! two kernel stages (`populate` then `find_pairs`) in separate compute passes
//! so the grid writes are visible to the neighbour scan, and reads the emitted
//! pairs back. The result is the same candidate set the [`cpu_broadphase`](
//! super::cpu_broadphase) twin produces, only reordered by the device's atomic
//! append; callers compare by sorting both.
//!
//! Provenance: Teschner et al. 2003 spatial hash; standard `wgpu` compute
//! dispatch. No Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, PipelineCompilationOptions,
    PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::buffer;
use crate::context::GpuContext;

use super::config::{BroadphaseConfig, BroadphaseError};
use super::pair::CandidatePair;
use super::particle::Particle;

/// Uniform parameters shared by both kernel stages. Layout matches `Params` in
/// `shaders/broadphase.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    table_size: u32,
    max_per_bucket: u32,
    pair_capacity: u32,
    cell_size: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

/// A compiled, reusable `GPU` broad-phase pipeline pair.
pub struct GpuBroadphase {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    populate: ComputePipeline,
    find_pairs: ComputePipeline,
}

impl GpuBroadphase {
    /// Compiles the broad-phase kernels on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBroadphase {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_broadphase"),
            source: ShaderSource::Wgsl(include_str!("../shaders/broadphase.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_broadphase_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
                buffer_entry(5, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_broadphase_pipeline_layout"),
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
        let populate = make("populate", "prism_broadphase_populate");
        let find_pairs = make("find_pairs", "prism_broadphase_find_pairs");
        GpuBroadphase {
            module,
            layout,
            populate,
            find_pairs,
        }
    }

    /// Runs the broad phase on device and returns the candidate pairs.
    ///
    /// # Errors
    ///
    /// Returns [`BroadphaseError`] when the config is invalid or the device
    /// reported more pairs than `pair_capacity` (which would have silently
    /// dropped output), mirroring the `CPU` twin's overflow contract.
    pub fn run(
        &self,
        ctx: &GpuContext,
        particles: &[Particle],
        config: &BroadphaseConfig,
    ) -> Result<Vec<CandidatePair>, BroadphaseError> {
        config.validate()?;
        let device = ctx.device();

        let count = particles.len() as u32;
        let params = Params {
            count,
            table_size: config.table_size,
            max_per_bucket: config.max_per_bucket,
            pair_capacity: config.pair_capacity,
            cell_size: config.cell_size,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };

        let gpu_particles: Vec<[f32; 4]> = particles.iter().map(|p| p.to_gpu().data).collect();
        // A storage buffer must never be zero-sized even when there are no
        // particles, so fall back to a single padded element.
        let particle_upload: &[[f32; 4]] = if gpu_particles.is_empty() {
            &[[0.0; 4]]
        } else {
            &gpu_particles
        };

        let params_buf = buffer::uniform(device, "broadphase_params", &params);
        let particle_buf = buffer::storage_read(device, "broadphase_particles", particle_upload);
        let counts_buf = buffer::storage_rw_zeroed(
            device,
            "broadphase_counts",
            u64::from(config.table_size) * 4,
        );
        let entries_buf =
            buffer::storage_rw_zeroed(device, "broadphase_entries", config.entry_slots() * 4);
        let pair_count_buf = buffer::storage_rw_zeroed(device, "broadphase_pair_count", 4);
        let pairs_buf = buffer::storage_rw_zeroed(
            device,
            "broadphase_pairs",
            u64::from(config.pair_capacity) * 8,
        );

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_broadphase_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: particle_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: counts_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: entries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: pair_count_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: pairs_buf.as_entire_binding(),
                },
            ],
        });

        let pair_count_stage = buffer::staging(device, "broadphase_pair_count_stage", 4);
        let pairs_bytes = u64::from(config.pair_capacity) * 8;
        let pairs_stage = buffer::staging(device, "broadphase_pairs_stage", pairs_bytes);

        let workgroups = count.div_ceil(64).max(1);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_broadphase_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_broadphase_populate_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.populate);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_broadphase_find_pairs_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.find_pairs);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        buffer::copy(&mut encoder, &pair_count_buf, &pair_count_stage, 4);
        buffer::copy(&mut encoder, &pairs_buf, &pairs_stage, pairs_bytes);
        ctx.queue().submit([encoder.finish()]);

        let total = buffer::read_back::<u32>(ctx, &pair_count_stage)[0];
        if total > config.pair_capacity {
            return Err(BroadphaseError::PairCapacityExceeded {
                capacity: config.pair_capacity,
            });
        }
        let raw = buffer::read_back::<[u32; 2]>(ctx, &pairs_stage);
        let pairs = raw
            .into_iter()
            .take(total as usize)
            .map(|[i, j]| CandidatePair::new(i, j))
            .collect();
        Ok(pairs)
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
