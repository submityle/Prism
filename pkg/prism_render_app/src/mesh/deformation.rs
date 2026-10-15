//! Persistent native-RHI batch deformation, with explicit temporal history.
//!
//! A cache uploads immutable inputs once and submits one compute dispatch per
//! pose, with no production readback. Consumers bind the returned storage/vertex
//! buffers in depth, shadow, velocity and refit passes. Output is mesh-local;
//! velocity consumers must also apply current/previous instance transforms.
//! Submit consumers before the next update on the same queue: two outputs are
//! retained, not an unbounded number of in-flight frames. Dropping the cache
//! releases its resources through its owning device.

use std::sync::Arc;

use prism_mesh::deformation::{
    DeformationCorrection, DeformationError, DeformationMesh, DeformationPose,
};
use prism_render_driver as rhi;
use rhi::{RenderDevice, RenderQueue};

mod source;
pub use source::DeformationSource;

/// Validation failure before any GPU write, dispatch, or history mutation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeformationCacheError {
    /// Malformed mesh/pose.
    Input(DeformationError),
    /// A static or dynamic buffer exceeds the device binding limit.
    BufferLimit {
        /// Requested bytes.
        requested: u64,
        /// Device-supported bytes.
        limit: u64,
    },
    /// The device cannot dispatch this mesh with the portable compute layout.
    ComputeLimit,
    /// A frame must strictly increase; repeated submissions would lose history.
    FrameNotIncreasing,
}

impl core::fmt::Display for DeformationCacheError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "mesh deformation cache: {self:?}")
    }
}
impl std::error::Error for DeformationCacheError {}

/// Borrowed output identity; buffer IDs remain owned by the cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeformationOutput {
    /// Current pose (VERTEX | STORAGE | COPY_SRC).
    pub current: rhi::BufferId,
    /// Previous contiguous pose, or current when history is invalid.
    pub previous: rhi::BufferId,
    /// False on first use, a frame gap, or explicit invalidation/teleport.
    pub history_valid: bool,
    /// Frame that produced this view.
    pub frame: u64,
    /// Number of valid 32-byte output records (can be zero).
    pub vertex_count: u32,
}

impl DeformationOutput {
    /// Interleaved position/normal vertex layout; offsets 0/16, stride 32.
    /// This is not the 12-byte split position stream of a static RenderMesh.
    pub fn vertex_layout(position_location: u32, normal_location: u32) -> rhi::VertexBufferLayout {
        rhi::VertexBufferLayout {
            array_stride: 32,
            step_mode: rhi::VertexStepMode::Vertex,
            attributes: vec![
                rhi::VertexAttribute {
                    format: rhi::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: position_location,
                },
                rhi::VertexAttribute {
                    format: rhi::VertexFormat::Float32x3,
                    offset: 16,
                    shader_location: normal_location,
                },
            ],
        }
    }
}

/// One mesh instance's persistent compute resources and double-buffered output.
/// Different instances may use the same authored mesh with independent poses.
pub struct DeformationCache {
    source: Arc<DeformationSource>,
    // First three IDs are borrowed from source; the remaining six are owned.
    buffers: [rhi::BufferId; 9],
    groups: [rhi::BindGroupId; 2],
    current: usize,
    frame: Option<u64>,
    history_valid: bool,
    reset_history: bool,
    zero_corrections: Vec<DeformationCorrection>,
    had_corrections: bool,
    allocated_bytes: u64,
}

impl DeformationCache {
    /// Creates/upload static streams after checking all reported device limits.
    /// `queue` must belong to `device`, as for all native RHI resources.
    pub fn new(
        device: Arc<dyn RenderDevice + Send + Sync>,
        queue: &dyn RenderQueue,
        mesh: DeformationMesh,
    ) -> Result<Self, DeformationCacheError> {
        let source = Arc::new(DeformationSource::new(device, queue, mesh)?);
        Self::from_source(source, queue)
    }

    /// Allocates only this instance's pose/history resources. Immutable inputs,
    /// CPU validation data and pipeline remain shared with every source owner.
    /// The queue must belong to the source's device.
    pub fn from_source(
        source: Arc<DeformationSource>,
        queue: &dyn RenderQueue,
    ) -> Result<Self, DeformationCacheError> {
        let device = &source.device;
        let mesh = &source.mesh;
        let sizes = Self::buffer_sizes(mesh, device.capabilities().limits)?;
        let buffers = core::array::from_fn(|i| {
            if i < 3 {
                return source.buffers[i];
            }
            let usage = if i == 8 {
                rhi::BufferUsages::UNIFORM | rhi::BufferUsages::COPY_DST
            } else if i == 6 || i == 7 {
                rhi::BufferUsages::STORAGE | rhi::BufferUsages::VERTEX | rhi::BufferUsages::COPY_SRC
            } else {
                rhi::BufferUsages::STORAGE | rhi::BufferUsages::COPY_DST
            };
            device.create_buffer(&rhi::BufferDescriptor::new(sizes[i], usage))
        });
        queue.write_buffer(
            buffers[8],
            0,
            bytemuck::cast_slice(&[mesh.vertex_count(), 0, 0, 0]),
        );
        let groups = core::array::from_fn(|output| {
            device.create_bind_group(&rhi::BindGroupDescriptor {
                label: Some("mesh-deformation-bindings".into()),
                layout: Some(source.layout),
                entries: (0..8)
                    .map(|binding| rhi::BindGroupEntry {
                        binding,
                        resource: rhi::BindingResource::Buffer {
                            buffer: buffers[match binding {
                                6 => 6 + output,
                                7 => 8,
                                _ => binding as usize,
                            }],
                            offset: 0,
                            size: None,
                        },
                    })
                    .collect(),
            })
        });
        let zero_corrections = vec![DeformationCorrection::default(); mesh.vertex_count() as usize];
        // Zero initialization is explicit rather than relying on backend rules.
        if !zero_corrections.is_empty() {
            queue.write_buffer(buffers[5], 0, bytemuck::cast_slice(&zero_corrections));
        }
        Ok(Self {
            source,
            buffers,
            groups,
            current: 1,
            frame: None,
            history_valid: false,
            reset_history: false,
            zero_corrections,
            had_corrections: false,
            allocated_bytes: sizes[3..].iter().sum(),
        })
    }

    fn buffer_sizes(
        mesh: &DeformationMesh,
        limits: rhi::Limits,
    ) -> Result<[u64; 9], DeformationCacheError> {
        if limits.max_compute_invocations_per_workgroup < 64
            || limits.max_compute_workgroups_per_dimension < mesh.vertex_count().div_ceil(64)
            || limits.max_bindings_per_bind_group < 8
            || limits.max_bind_groups < 1
            || limits.max_uniform_buffer_binding_size < 16
        {
            return Err(DeformationCacheError::ComputeLimit);
        }
        let sizes = [
            (mesh.vertices().len() as u64 * 48).max(48),
            (mesh.morphs().len() as u64 * 16).max(16),
            (mesh.influences().len() as u64 * 8).max(8),
            (u64::from(mesh.target_count()) * 4).max(4),
            (u64::from(mesh.joint_count()) * 96).max(96),
            (u64::from(mesh.vertex_count()) * 32).max(32),
            (u64::from(mesh.vertex_count()) * 32).max(32),
            (u64::from(mesh.vertex_count()) * 32).max(32),
            16,
        ];
        for &size in &sizes[..8] {
            if size > limits.max_storage_buffer_binding_size {
                return Err(DeformationCacheError::BufferLimit {
                    requested: size,
                    limit: limits.max_storage_buffer_binding_size,
                });
            }
        }
        Ok(sizes)
    }

    /// This instance's exclusive GPU buffer bytes, including both output frames
    /// and padding. Count `source().allocated_bytes()` once per shared source;
    /// opaque driver pipeline memory is not included.
    pub fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes
    }

    /// Shared immutable resource identity. Clone this Arc to create independent
    /// poses without re-uploading streams or recompiling the compute pipeline.
    pub fn source(&self) -> &Arc<DeformationSource> {
        &self.source
    }

    /// Marks a teleport/rebind. Previous output aliases current until a fresh
    /// contiguous pair of submissions has been produced after this call.
    pub fn invalidate_history(&mut self) {
        self.history_valid = false;
        self.reset_history = true;
    }

    /// Most recently submitted pose; `None` before the first successful update.
    pub fn output(&self) -> Option<DeformationOutput> {
        Some(DeformationOutput {
            current: self.buffers[6 + self.current],
            previous: self.buffers[6 + if self.history_valid {
                1 - self.current
            } else {
                self.current
            }],
            history_valid: self.history_valid,
            frame: self.frame?,
            vertex_count: self.source.mesh.vertex_count(),
        })
    }

    /// Validates, uploads dynamic data, and submits exactly one batch dispatch.
    /// Errors preserve buffers and temporal state. No per-vertex submissions or
    /// readbacks occur. Queue must be the owning device's same ordered queue.
    pub fn submit_pose(
        &mut self,
        queue: &dyn RenderQueue,
        frame: u64,
        pose: DeformationPose<'_>,
    ) -> Result<DeformationOutput, DeformationCacheError> {
        if self.frame.is_some_and(|previous| frame <= previous) {
            return Err(DeformationCacheError::FrameNotIncreasing);
        }
        self.source
            .mesh
            .validate_pose(pose)
            .map_err(DeformationCacheError::Input)?;
        if !pose.morph_weights.is_empty() {
            queue.write_buffer(self.buffers[3], 0, bytemuck::cast_slice(pose.morph_weights));
        }
        if !pose.joints.is_empty() {
            queue.write_buffer(self.buffers[4], 0, bytemuck::cast_slice(pose.joints));
        }
        if let Some(corrections) = pose.corrections {
            if !corrections.is_empty() {
                queue.write_buffer(self.buffers[5], 0, bytemuck::cast_slice(corrections));
            }
        } else if self.had_corrections && !self.zero_corrections.is_empty() {
            queue.write_buffer(
                self.buffers[5],
                0,
                bytemuck::cast_slice(&self.zero_corrections),
            );
        }
        let next = 1 - self.current;
        if self.source.mesh.vertex_count() != 0 {
            let mut encoder = rhi::CommandEncoder::labeled("mesh-deformation");
            encoder.push_compute_pass(
                Some("mesh-deformation".into()),
                vec![
                    rhi::ComputeCommand::SetPipeline(self.source.pipeline),
                    rhi::ComputeCommand::SetBindGroup {
                        index: 0,
                        bind_group: self.groups[next],
                        dynamic_offsets: vec![],
                    },
                    rhi::ComputeCommand::Dispatch {
                        x: self.source.mesh.vertex_count().div_ceil(64),
                        y: 1,
                        z: 1,
                    },
                ],
            );
            queue.submit(&[encoder.finish()]);
        }
        self.history_valid =
            !self.reset_history && self.frame.and_then(|f| f.checked_add(1)) == Some(frame);
        self.reset_history = false;
        self.had_corrections = pose.corrections.is_some();
        self.frame = Some(frame);
        self.current = next;
        Ok(self.output().expect("frame was recorded"))
    }
}

impl Drop for DeformationCache {
    fn drop(&mut self) {
        for group in self.groups {
            self.source.device.destroy_bind_group(group);
        }
        for &buffer in &self.buffers[3..] {
            self.source.device.destroy_buffer(buffer);
        }
    }
}

#[cfg(test)]
mod tests;
