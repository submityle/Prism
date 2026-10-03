//! Device capabilities: optional features and numeric limits.

use crate::flags::bitflags;

bitflags! {
    /// Optional GPU features a device may advertise. A pipeline requesting a
    /// feature the device lacks is rejected at creation.
    pub struct Features {
        /// Indirect draw/dispatch from a GPU buffer.
        const INDIRECT_DRAW = 1 << 0;
        /// Multi-draw indirect in a single call.
        const MULTI_DRAW_INDIRECT = 1 << 1;
        /// Hardware-accelerated ray tracing acceleration structures.
        const RAY_TRACING = 1 << 2;
        /// Bindless / large descriptor arrays.
        const BINDLESS = 1 << 3;
        /// 64-bit shader integers.
        const SHADER_INT64 = 1 << 4;
        /// 16-bit shader floats.
        const SHADER_FLOAT16 = 1 << 5;
        /// Mesh and task shader stages.
        const MESH_SHADER = 1 << 6;
        /// Hardware timestamp queries.
        const TIMESTAMP_QUERY = 1 << 7;
        /// Pipeline statistics queries.
        const PIPELINE_STATISTICS_QUERY = 1 << 8;
        /// Depth clamping (unclipped depth).
        const DEPTH_CLAMP = 1 << 9;
        /// BC (DXT) texture compression.
        const TEXTURE_COMPRESSION_BC = 1 << 10;
        /// Dual-source blending.
        const DUAL_SOURCE_BLENDING = 1 << 11;
    }
}

/// Numeric limits a device guarantees. Resource creation that exceeds a limit
/// is rejected.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Limits {
    /// The maximum size of any 1D/2D texture dimension.
    pub max_texture_dimension_2d: u32,
    /// The maximum size of a 3D texture dimension.
    pub max_texture_dimension_3d: u32,
    /// The maximum number of array layers.
    pub max_texture_array_layers: u32,
    /// The maximum number of bind groups bound at once.
    pub max_bind_groups: u32,
    /// The maximum bindings per bind group.
    pub max_bindings_per_bind_group: u32,
    /// The maximum uniform buffer binding size in bytes.
    pub max_uniform_buffer_binding_size: u64,
    /// The maximum storage buffer binding size in bytes.
    pub max_storage_buffer_binding_size: u64,
    /// The maximum number of vertex buffers.
    pub max_vertex_buffers: u32,
    /// The maximum number of vertex attributes across all buffers.
    pub max_vertex_attributes: u32,
    /// The maximum number of color attachments in a render pass.
    pub max_color_attachments: u32,
    /// The maximum push constant size in bytes.
    pub max_push_constant_size: u32,
    /// The maximum total invocations per compute workgroup.
    pub max_compute_invocations_per_workgroup: u32,
    /// The maximum workgroups per dispatch dimension.
    pub max_compute_workgroups_per_dimension: u32,
}

impl Limits {
    /// A conservative baseline every modern desktop/mobile GPU is expected to
    /// meet, suitable as a portability floor.
    #[must_use]
    pub const fn baseline() -> Self {
        Self {
            max_texture_dimension_2d: 8192,
            max_texture_dimension_3d: 2048,
            max_texture_array_layers: 256,
            max_bind_groups: 4,
            max_bindings_per_bind_group: 1000,
            max_uniform_buffer_binding_size: 64 * 1024,
            max_storage_buffer_binding_size: 128 * 1024 * 1024,
            max_vertex_buffers: 8,
            max_vertex_attributes: 16,
            max_color_attachments: 8,
            max_push_constant_size: 128,
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroups_per_dimension: 65535,
        }
    }

    /// A higher tier representative of modern discrete desktop GPUs.
    #[must_use]
    pub const fn desktop() -> Self {
        Self {
            max_texture_dimension_2d: 16384,
            max_texture_dimension_3d: 2048,
            max_texture_array_layers: 2048,
            max_bind_groups: 8,
            max_bindings_per_bind_group: 1_000_000,
            max_uniform_buffer_binding_size: 64 * 1024,
            max_storage_buffer_binding_size: 2 * 1024 * 1024 * 1024,
            max_vertex_buffers: 16,
            max_vertex_attributes: 32,
            max_color_attachments: 8,
            max_push_constant_size: 256,
            max_compute_invocations_per_workgroup: 1024,
            max_compute_workgroups_per_dimension: 65535,
        }
    }

    /// Whether every limit in `self` is at least as permissive as the matching
    /// limit in `required`. Used to check a device meets an app's needs.
    #[must_use]
    pub const fn satisfies(&self, required: &Self) -> bool {
        self.max_texture_dimension_2d >= required.max_texture_dimension_2d
            && self.max_texture_dimension_3d >= required.max_texture_dimension_3d
            && self.max_texture_array_layers >= required.max_texture_array_layers
            && self.max_bind_groups >= required.max_bind_groups
            && self.max_bindings_per_bind_group >= required.max_bindings_per_bind_group
            && self.max_uniform_buffer_binding_size >= required.max_uniform_buffer_binding_size
            && self.max_storage_buffer_binding_size >= required.max_storage_buffer_binding_size
            && self.max_vertex_buffers >= required.max_vertex_buffers
            && self.max_vertex_attributes >= required.max_vertex_attributes
            && self.max_color_attachments >= required.max_color_attachments
            && self.max_push_constant_size >= required.max_push_constant_size
            && self.max_compute_invocations_per_workgroup
                >= required.max_compute_invocations_per_workgroup
            && self.max_compute_workgroups_per_dimension
                >= required.max_compute_workgroups_per_dimension
    }
}

impl Default for Limits {
    fn default() -> Self {
        Self::baseline()
    }
}

/// The full capability set a device reports: its backend, features, and limits.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DeviceCapabilities {
    /// The graphics backend the device runs on.
    pub backend: Backend,
    /// The optional features the device supports.
    pub features: Features,
    /// The numeric limits the device guarantees.
    pub limits: Limits,
}

impl DeviceCapabilities {
    /// Whether the device supports all `features` and meets all `limits`.
    #[must_use]
    pub const fn supports(&self, features: Features, limits: &Limits) -> bool {
        self.features.contains(features) && self.limits.satisfies(limits)
    }
}

/// The concrete graphics API a backend targets.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub enum Backend {
    /// A software/reference backend (e.g. for tests).
    Noop,
    /// Vulkan.
    Vulkan,
    /// Apple Metal.
    Metal,
    /// Direct3D 12.
    Dx12,
    /// WebGPU / wgpu.
    WebGpu,
}
