//! Optional GPU coverage resources. No per-frame allocation or CPU proxy walk.

use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::gpu::{buffer_binding, storage_entry, uniform_entry, WORKGROUP_SIZE};
use crate::{DensityPolicy, GpuResidentError};

pub(crate) struct ResidentDensity {
    pub(crate) coverage: wgpu::Buffer,
    params: wgpu::Buffer,
    policy: DensityPolicy,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    compact_pipeline: wgpu::ComputePipeline,
    compact_bind_group: wgpu::BindGroup,
}

fn policy_words(policy: DensityPolicy) -> [u32; 4] {
    [
        policy.density().to_bits(),
        policy.transition().to_bits(),
        policy.seed(),
        policy.distance_fade().to_bits(),
    ]
}

impl ResidentDensity {
    // All referenced buffers have the same fixed-capacity owner and device.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        device: &wgpu::Device,
        capacity: usize,
        policy: DensityPolicy,
        camera: &wgpu::Buffer,
        positions: &wgpu::Buffer,
        errors: &wgpu::Buffer,
        metadata: &wgpu::Buffer,
        records: &wgpu::Buffer,
        indices: &wgpu::Buffer,
        indirect: &wgpu::Buffer,
    ) -> Self {
        let coverage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resident density coverage"),
            size: capacity.max(1) as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("resident density policy"),
            contents: bytemuck::cast_slice(&policy_words(policy)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("resident density layout"),
            entries: &[
                uniform_entry(0),
                uniform_entry(1),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
                storage_entry(5, true),
            ],
        });
        let pipeline = create_pipeline(
            device,
            "resident density pipeline",
            include_str!("shaders/proxy_density.wgsl"),
            &layout,
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resident density bind group"),
            layout: &layout,
            entries: &[
                buffer_binding(0, camera),
                buffer_binding(1, &params),
                buffer_binding(2, positions),
                buffer_binding(3, metadata),
                buffer_binding(4, &coverage),
                buffer_binding(5, errors),
            ],
        });
        let compact_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("resident density compact layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, false),
                storage_entry(4, false),
            ],
        });
        let compact_pipeline = create_pipeline(
            device,
            "resident density compact pipeline",
            include_str!("shaders/proxy_compact_density.wgsl"),
            &compact_layout,
        );
        let compact_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resident density compact bind group"),
            layout: &compact_layout,
            entries: &[
                buffer_binding(0, camera),
                buffer_binding(1, records),
                buffer_binding(2, indices),
                buffer_binding(3, indirect),
                buffer_binding(4, &coverage),
            ],
        });
        Self {
            coverage,
            params,
            policy,
            pipeline,
            bind_group,
            compact_pipeline,
            compact_bind_group,
        }
    }

    pub(crate) fn policy(&self) -> DensityPolicy {
        self.policy
    }

    pub(crate) fn write(&mut self, queue: &wgpu::Queue, policy: DensityPolicy) {
        if policy_words(self.policy) != policy_words(policy) {
            queue.write_buffer(&self.params, 0, bytemuck::cast_slice(&policy_words(policy)));
            self.policy = policy;
        }
    }

    pub(crate) fn allocated_buffer_bytes(&self) -> u64 {
        self.coverage.size() + self.params.size()
    }

    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder, count: u32) {
        if count == 0 {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("resident density and compact pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(count.div_ceil(WORKGROUP_SIZE), 1, 1);
        pass.set_pipeline(&self.compact_pipeline);
        pass.set_bind_group(0, &self.compact_bind_group, &[]);
        pass.dispatch_workgroups(count.div_ceil(WORKGROUP_SIZE), 1, 1);
    }
}

fn create_pipeline(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(bind_group_layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

pub(crate) fn validate_limits(limits: &wgpu::Limits) -> Result<(), GpuResidentError> {
    for (limit, required, available) in [
        (
            "max_uniform_buffers_per_shader_stage",
            2,
            limits.max_uniform_buffers_per_shader_stage,
        ),
        (
            "max_bindings_per_bind_group",
            6,
            limits.max_bindings_per_bind_group,
        ),
    ] {
        if required > available {
            return Err(GpuResidentError::DeviceLimit {
                limit,
                required: u64::from(required),
                available: u64::from(available),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn density_limits_are_validated_without_allocating() {
        let mut limits = wgpu::Limits::downlevel_defaults();
        assert_eq!(validate_limits(&limits), Ok(()));
        limits.max_uniform_buffers_per_shader_stage = 1;
        assert!(matches!(
            validate_limits(&limits),
            Err(GpuResidentError::DeviceLimit {
                limit: "max_uniform_buffers_per_shader_stage",
                ..
            })
        ));
        limits.max_uniform_buffers_per_shader_stage = 2;
        limits.max_bindings_per_bind_group = 5;
        assert!(matches!(
            validate_limits(&limits),
            Err(GpuResidentError::DeviceLimit {
                limit: "max_bindings_per_bind_group",
                ..
            })
        ));
    }
}
