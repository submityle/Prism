//! Device-local, immutable deformation inputs and shared compute pipeline.

use super::*;

/// Immutable authored streams and pipeline shared by independently posed mesh
/// instances on one device. Share an `Arc<DeformationSource>` explicitly;
/// there is no process-global cache or implicit content-hash collision policy.
/// Dropping the last instance/source owner releases these device resources.
pub struct DeformationSource {
    pub(super) device: Arc<dyn RenderDevice + Send + Sync>,
    pub(super) mesh: DeformationMesh,
    pub(super) buffers: [rhi::BufferId; 3],
    pub(super) layout: rhi::BindGroupLayoutId,
    pipeline_layout: rhi::PipelineLayoutId,
    shader: rhi::ShaderModuleId,
    pub(super) pipeline: rhi::ComputePipelineId,
    allocated_bytes: u64,
}

impl DeformationSource {
    /// Validates all source/instance limits before creating resources, uploads
    /// the three static streams once, and compiles one shared compute pipeline.
    /// The queue must belong to this device, as for every native RHI upload.
    pub fn new(
        device: Arc<dyn RenderDevice + Send + Sync>,
        queue: &dyn RenderQueue,
        mesh: DeformationMesh,
    ) -> Result<Self, DeformationCacheError> {
        let sizes = DeformationCache::buffer_sizes(&mesh, device.capabilities().limits)?;
        let buffers = core::array::from_fn(|i| {
            device.create_buffer(&rhi::BufferDescriptor::new(
                sizes[i],
                rhi::BufferUsages::STORAGE | rhi::BufferUsages::COPY_DST,
            ))
        });
        for (i, bytes) in [
            bytemuck::cast_slice(mesh.vertices()),
            bytemuck::cast_slice(mesh.morphs()),
            bytemuck::cast_slice(mesh.influences()),
        ]
        .into_iter()
        .enumerate()
        {
            if !bytes.is_empty() {
                queue.write_buffer(buffers[i], 0, bytes);
            }
        }
        let layout = device.create_bind_group_layout(&rhi::BindGroupLayoutDescriptor {
            label: Some("mesh-deformation-layout".into()),
            entries: (0..8)
                .map(|binding| rhi::BindGroupLayoutEntry {
                    binding,
                    visibility: rhi::ShaderStages::COMPUTE,
                    ty: rhi::BindingType::Buffer {
                        ty: if binding == 7 {
                            rhi::BufferBindingType::Uniform
                        } else {
                            rhi::BufferBindingType::Storage {
                                read_only: binding != 6,
                            }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect(),
        });
        let pipeline_layout = device.create_pipeline_layout(&rhi::PipelineLayoutDescriptor {
            label: Some("mesh-deformation-pipeline-layout".into()),
            bind_group_layouts: vec![layout],
            push_constant_ranges: vec![],
        });
        let shader = device.create_shader_module(&rhi::ShaderModuleDescriptor::wgsl(include_str!(
            "../shaders/deformation.wgsl"
        )));
        let pipeline = device.create_compute_pipeline(&rhi::ComputePipelineDescriptor {
            label: Some("mesh-deformation".into()),
            layout: Some(pipeline_layout),
            module: shader,
            entry_point: "main".into(),
        });
        Ok(Self {
            device,
            mesh,
            buffers,
            layout,
            pipeline_layout,
            shader,
            pipeline,
            allocated_bytes: sizes[..3].iter().sum(),
        })
    }

    /// Device-independent source used by pose validation and CPU reference.
    pub fn mesh(&self) -> &DeformationMesh {
        &self.mesh
    }

    /// Logical bytes of the three shared static GPU buffers, including padding.
    /// Does not include per-instance buffers or opaque driver pipeline memory.
    pub fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes
    }
}

impl Drop for DeformationSource {
    fn drop(&mut self) {
        self.device.destroy_compute_pipeline(self.pipeline);
        self.device.destroy_shader_module(self.shader);
        self.device.destroy_pipeline_layout(self.pipeline_layout);
        self.device.destroy_bind_group_layout(self.layout);
        for buffer in self.buffers {
            self.device.destroy_buffer(buffer);
        }
    }
}
