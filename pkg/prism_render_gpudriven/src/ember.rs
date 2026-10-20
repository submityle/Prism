//! GPU resource lifetime for a compiled Ember graph.
//!
//! The architecture crate deliberately stops at validated WGSL and a typed
//! ABI. This module is the next M2 boundary: it turns that contract into
//! device buffers, bind-group layouts/groups, a pipeline layout, and one
//! compute pipeline per generated entry point. Module-owned storage buffers
//! and textures remain caller-owned; parameter blocks and dispatch counts are
//! owned here because their layout and initial contents come from the compiled
//! graph itself.

use alloc::borrow::Cow;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use prism_render_architecture::particle::compiler::{CompiledGraph, PreparedParameterBlock};
use prism_render_architecture::particle::modules::{ResourceAccess, ResourceDecl, ResourceKind};
use prism_render_driver as rhi;
use rhi::{
    BindGroupDescriptor, BindGroupEntry, BindGroupId, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindGroupLayoutId, BindingResource, BindingType, BufferBindingType,
    BufferDescriptor, BufferId, BufferUsages, ComputePipelineDescriptor, ComputePipelineId,
    PipelineLayoutDescriptor, PipelineLayoutId, RenderDevice, RenderQueue, ShaderModuleDescriptor,
    ShaderModuleId, ShaderSource, ShaderStages, TextureSampleType, TextureViewDimension,
};

/// A caller-owned resource used to satisfy one compiled non-uniform binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompiledGraphGpuBinding {
    /// A whole storage buffer.
    Buffer {
        /// Compiled group index.
        group: u32,
        /// Compiled binding index.
        binding: u32,
        /// Existing caller-owned buffer.
        buffer: BufferId,
        /// Optional byte range exposed to the shader.
        size: Option<u64>,
    },
    /// An existing sampled texture view.
    TextureView {
        /// Compiled group index.
        group: u32,
        /// Compiled binding index.
        binding: u32,
        /// Existing caller-owned view.
        view: rhi::TextureViewId,
    },
}

impl CompiledGraphGpuBinding {
    const fn slot(self) -> (u32, u32) {
        match self {
            Self::Buffer { group, binding, .. } | Self::TextureView { group, binding, .. } => {
                (group, binding)
            }
        }
    }
}

/// A failure found while resolving a compiled graph to device resources.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompiledGraphGpuError {
    /// Two caller-owned entries claimed one compiled slot.
    DuplicateExternalBinding {
        /// Compiled group index.
        group: u32,
        /// Compiled binding index.
        binding: u32,
    },
    /// A compiled non-uniform resource has no caller-owned resource.
    MissingExternalBinding {
        /// Compiled group index.
        group: u32,
        /// Compiled binding index.
        binding: u32,
        /// Compiled resource name.
        name: String,
    },
    /// An external entry does not correspond to a compiled non-uniform resource.
    UnexpectedExternalBinding {
        /// Supplied group index.
        group: u32,
        /// Supplied binding index.
        binding: u32,
    },
    /// A resource kind needs a buffer but the caller supplied a texture view.
    ExpectedBuffer {
        /// Compiled group index.
        group: u32,
        /// Compiled binding index.
        binding: u32,
        /// Compiled resource name.
        name: String,
    },
    /// A resource kind needs a texture view but the caller supplied a buffer.
    ExpectedTexture {
        /// Compiled group index.
        group: u32,
        /// Compiled binding index.
        binding: u32,
        /// Compiled resource name.
        name: String,
    },
    /// A resource kind is not representable by this bridge.
    UnsupportedResource {
        /// Compiled group index.
        group: u32,
        /// Compiled binding index.
        binding: u32,
        /// Compiled resource name.
        name: String,
    },
    /// The generated graph has a uniform resource without a typed block.
    MissingParameterBlock {
        /// Compiled group index.
        group: u32,
        /// Compiled binding index.
        binding: u32,
        /// Compiled resource name.
        name: String,
    },
}

/// A failure while updating compiled parameter buffers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ParameterUploadError {
    /// One parameter resource was supplied more than once.
    DuplicateResource {
        /// Parameter resource name.
        resource: String,
    },
    /// The update refers to no compiled parameter block.
    UnknownResource {
        /// Parameter resource name.
        resource: String,
    },
    /// The supplied bytes do not match the compiled block size.
    SizeMismatch {
        /// Parameter resource name.
        resource: String,
        /// Compiled block size.
        expected: usize,
        /// Supplied byte count.
        actual: usize,
    },
    /// The update's binding metadata disagrees with the compiled block.
    BindingMismatch {
        /// Parameter resource name.
        resource: String,
        /// Compiled group index.
        expected_group: u32,
        /// Compiled binding index.
        expected_binding: u32,
        /// Supplied group index.
        actual_group: u32,
        /// Supplied binding index.
        actual_binding: u32,
    },
    /// A complete frame update omitted a compiled parameter block.
    MissingResource {
        /// Parameter resource name.
        resource: String,
    },
}

struct ParameterBuffer {
    resource: String,
    group: u32,
    binding: u32,
    size: u64,
    buffer: BufferId,
}

struct ResolvedEntry {
    binding: u32,
    layout: BindGroupLayoutEntry,
    resource: BindingResource,
}

/// GPU resources and pipelines for one compiled Ember graph.
pub struct CompiledGraphGpuState {
    shader: ShaderModuleId,
    pipeline_layout: PipelineLayoutId,
    bind_group_layouts: Vec<BindGroupLayoutId>,
    bind_groups: Vec<BindGroupId>,
    compute_pipelines: Vec<ComputePipelineId>,
    parameter_buffers: Vec<ParameterBuffer>,
    dispatch_counts: Option<BufferId>,
}

impl CompiledGraphGpuState {
    /// Creates all device-owned resources and uploads compiled defaults.
    pub fn new(
        device: &impl RenderDevice,
        queue: &impl RenderQueue,
        graph: &CompiledGraph,
        external: &[CompiledGraphGpuBinding],
    ) -> Result<Self, CompiledGraphGpuError> {
        validate_external_bindings(graph, external)?;
        validate_parameter_blocks(graph)?;

        let group_count = max_group(graph).map_or(0, |group| group as usize + 1);
        let mut parameter_buffers = Vec::with_capacity(graph.parameter_blocks.len());
        for block in &graph.parameter_blocks {
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some(format!("ember-parameter-{}", block.resource)),
                size: block.size as u64,
                usage: BufferUsages::UNIFORM.union(BufferUsages::COPY_DST),
                mapped_at_creation: false,
            });
            queue.write_buffer(buffer, 0, &block.default_bytes);
            parameter_buffers.push(ParameterBuffer {
                resource: block.resource.clone(),
                group: block.group,
                binding: block.binding,
                size: block.size as u64,
                buffer,
            });
        }

        let dispatch_counts = if graph.dispatches.is_empty() {
            None
        } else {
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some(String::from("ember-dispatch-counts")),
                size: graph.dispatches.len().max(1) as u64 * 4,
                usage: BufferUsages::STORAGE.union(BufferUsages::COPY_DST),
                mapped_at_creation: false,
            });
            let mut bytes = Vec::with_capacity(graph.dispatches.len() * 4);
            for dispatch in &graph.dispatches {
                bytes.extend_from_slice(&dispatch.default_count.to_le_bytes());
            }
            queue.write_buffer(buffer, 0, &bytes);
            Some(buffer)
        };

        let mut bind_group_layouts = Vec::with_capacity(group_count);
        let mut bind_groups = Vec::with_capacity(group_count);
        for group in 0..group_count as u32 {
            let entries =
                group_entries(graph, external, &parameter_buffers, dispatch_counts, group)?;
            let mut layout_entries: Vec<_> = entries.iter().map(|entry| entry.layout).collect();
            layout_entries.sort_by_key(|entry| entry.binding);
            let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some(format!("ember-graph-group-{group}")),
                entries: layout_entries,
            });
            let mut bind_entries: Vec<_> = entries
                .into_iter()
                .map(|entry| BindGroupEntry {
                    binding: entry.binding,
                    resource: entry.resource,
                })
                .collect();
            bind_entries.sort_by_key(|entry| entry.binding);
            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some(format!("ember-graph-group-{group}-bindings")),
                layout: Some(layout),
                entries: bind_entries,
            });
            bind_group_layouts.push(layout);
            bind_groups.push(bind_group);
        }

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(String::from("ember-graph-pipeline-layout")),
            bind_group_layouts: bind_group_layouts.clone(),
            push_constant_ranges: Vec::new(),
        });
        let shader = device.create_shader_module(&ShaderModuleDescriptor {
            label: Some(String::from("ember-compiled-graph")),
            source: ShaderSource::Wgsl(Cow::Owned(graph.source.clone())),
        });
        let mut compute_pipelines = Vec::with_capacity(graph.entry_points.len());
        for entry_point in &graph.entry_points {
            compute_pipelines.push(device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(format!("ember-graph-{entry_point}")),
                layout: Some(pipeline_layout),
                module: shader,
                entry_point: Cow::Owned(entry_point.clone()),
            }));
        }

        Ok(Self {
            shader,
            pipeline_layout,
            bind_group_layouts,
            bind_groups,
            compute_pipelines,
            parameter_buffers,
            dispatch_counts,
        })
    }

    /// Uploads a complete prepared parameter set without rebuilding bind groups.
    pub fn update_parameters(
        &self,
        queue: &impl RenderQueue,
        updates: &[PreparedParameterBlock],
    ) -> Result<(), ParameterUploadError> {
        for (index, update) in updates.iter().enumerate() {
            if updates[..index]
                .iter()
                .any(|previous| previous.resource == update.resource)
            {
                return Err(ParameterUploadError::DuplicateResource {
                    resource: update.resource.clone(),
                });
            }
            let Some(buffer) = self
                .parameter_buffers
                .iter()
                .find(|buffer| buffer.resource == update.resource)
            else {
                return Err(ParameterUploadError::UnknownResource {
                    resource: update.resource.clone(),
                });
            };
            if (update.group, update.binding) != (buffer.group, buffer.binding) {
                return Err(ParameterUploadError::BindingMismatch {
                    resource: update.resource.clone(),
                    expected_group: buffer.group,
                    expected_binding: buffer.binding,
                    actual_group: update.group,
                    actual_binding: update.binding,
                });
            }
            if update.bytes.len() != buffer.size as usize {
                return Err(ParameterUploadError::SizeMismatch {
                    resource: update.resource.clone(),
                    expected: buffer.size as usize,
                    actual: update.bytes.len(),
                });
            }
            queue.write_buffer(buffer.buffer, 0, &update.bytes);
        }
        for buffer in &self.parameter_buffers {
            if !updates
                .iter()
                .any(|update| update.resource == buffer.resource)
            {
                return Err(ParameterUploadError::MissingResource {
                    resource: buffer.resource.clone(),
                });
            }
        }
        Ok(())
    }

    /// Uploads the runtime stage counts used by generated dispatch guards.
    pub fn update_dispatch_counts(
        &self,
        queue: &impl RenderQueue,
        counts: &[u32],
    ) -> Result<(), &'static str> {
        let Some(buffer) = self.dispatch_counts else {
            return if counts.is_empty() {
                Ok(())
            } else {
                Err("compiled graph has no dispatch-count buffer")
            };
        };
        if counts.len() != self.compute_pipelines.len() {
            return Err("dispatch-count length does not match compiled stages");
        }
        let mut bytes = Vec::with_capacity(counts.len() * 4);
        for count in counts {
            bytes.extend_from_slice(&count.to_le_bytes());
        }
        queue.write_buffer(buffer, 0, &bytes);
        Ok(())
    }

    /// Returns the generated shader module.
    #[must_use]
    pub const fn shader(&self) -> ShaderModuleId {
        self.shader
    }

    /// Returns the pipeline layout shared by generated stage pipelines.
    #[must_use]
    pub const fn pipeline_layout(&self) -> PipelineLayoutId {
        self.pipeline_layout
    }

    /// Returns a group layout by compiled group index.
    #[must_use]
    pub fn bind_group_layout(&self, group: u32) -> Option<BindGroupLayoutId> {
        self.bind_group_layouts.get(group as usize).copied()
    }

    /// Returns a bind group by compiled group index.
    #[must_use]
    pub fn bind_group(&self, group: u32) -> Option<BindGroupId> {
        self.bind_groups.get(group as usize).copied()
    }

    /// Returns a generated compute pipeline by entry-point order.
    #[must_use]
    pub fn compute_pipeline(&self, index: usize) -> Option<ComputePipelineId> {
        self.compute_pipelines.get(index).copied()
    }

    /// Releases every resource created by [`Self::new`].
    pub fn destroy(self, device: &impl RenderDevice) {
        for pipeline in self.compute_pipelines {
            device.destroy_compute_pipeline(pipeline);
        }
        device.destroy_shader_module(self.shader);
        device.destroy_pipeline_layout(self.pipeline_layout);
        for bind_group in self.bind_groups {
            device.destroy_bind_group(bind_group);
        }
        for layout in self.bind_group_layouts {
            device.destroy_bind_group_layout(layout);
        }
        for parameter in self.parameter_buffers {
            device.destroy_buffer(parameter.buffer);
        }
        if let Some(dispatch_counts) = self.dispatch_counts {
            device.destroy_buffer(dispatch_counts);
        }
    }
}

fn max_group(graph: &CompiledGraph) -> Option<u32> {
    graph
        .attribute_bindings
        .iter()
        .map(|binding| binding.group)
        .chain(graph.bindings.iter().map(|binding| binding.group))
        .max()
}

fn validate_external_bindings(
    graph: &CompiledGraph,
    external: &[CompiledGraphGpuBinding],
) -> Result<(), CompiledGraphGpuError> {
    for (index, entry) in external.iter().enumerate() {
        let slot = entry.slot();
        if external[..index]
            .iter()
            .any(|previous| previous.slot() == slot)
        {
            return Err(CompiledGraphGpuError::DuplicateExternalBinding {
                group: slot.0,
                binding: slot.1,
            });
        }
        let expected = graph
            .attribute_bindings
            .iter()
            .find(|binding| (binding.group, binding.binding) == slot)
            .map(|binding| (binding.name.as_str(), ResourceKind::StorageBuffer))
            .or_else(|| {
                graph
                    .bindings
                    .iter()
                    .find(|binding| (binding.group, binding.binding) == slot)
                    .map(|binding| (binding.name, binding.kind))
            });
        let Some((name, kind)) = expected else {
            return Err(CompiledGraphGpuError::UnexpectedExternalBinding {
                group: slot.0,
                binding: slot.1,
            });
        };
        if kind == ResourceKind::Uniform {
            return Err(CompiledGraphGpuError::UnexpectedExternalBinding {
                group: slot.0,
                binding: slot.1,
            });
        }
        match (kind, entry) {
            (
                ResourceKind::Curve | ResourceKind::StorageBuffer,
                CompiledGraphGpuBinding::Buffer { .. },
            ) => {}
            (
                ResourceKind::Texture2d | ResourceKind::Texture3d,
                CompiledGraphGpuBinding::TextureView { .. },
            ) => {}
            (ResourceKind::Curve | ResourceKind::StorageBuffer, _) => {
                return Err(CompiledGraphGpuError::ExpectedBuffer {
                    group: slot.0,
                    binding: slot.1,
                    name: String::from(name),
                });
            }
            (ResourceKind::Texture2d | ResourceKind::Texture3d, _) => {
                return Err(CompiledGraphGpuError::ExpectedTexture {
                    group: slot.0,
                    binding: slot.1,
                    name: String::from(name),
                });
            }
            _ => {
                return Err(CompiledGraphGpuError::UnsupportedResource {
                    group: slot.0,
                    binding: slot.1,
                    name: String::from(name),
                });
            }
        }
    }
    for (group, binding, name) in graph
        .attribute_bindings
        .iter()
        .map(|binding| (binding.group, binding.binding, binding.name.as_str()))
        .chain(graph.bindings.iter().filter_map(|binding| {
            (binding.kind != ResourceKind::Uniform && binding.name != "ember_dispatch_counts")
                .then_some((binding.group, binding.binding, binding.name))
        }))
    {
        if !external
            .iter()
            .any(|entry| entry.slot() == (group, binding))
        {
            return Err(CompiledGraphGpuError::MissingExternalBinding {
                group,
                binding,
                name: String::from(name),
            });
        }
    }
    Ok(())
}

fn validate_parameter_blocks(graph: &CompiledGraph) -> Result<(), CompiledGraphGpuError> {
    for binding in graph
        .bindings
        .iter()
        .filter(|binding| binding.kind == ResourceKind::Uniform)
    {
        let Some(block) = graph.parameter_blocks.iter().find(|block| {
            block.resource == binding.name
                && block.group == binding.group
                && block.binding == binding.binding
        }) else {
            return Err(CompiledGraphGpuError::MissingParameterBlock {
                group: binding.group,
                binding: binding.binding,
                name: String::from(binding.name),
            });
        };
        if block.size == 0 {
            return Err(CompiledGraphGpuError::MissingParameterBlock {
                group: binding.group,
                binding: binding.binding,
                name: String::from(binding.name),
            });
        }
    }
    Ok(())
}

fn group_entries(
    graph: &CompiledGraph,
    external: &[CompiledGraphGpuBinding],
    parameter_buffers: &[ParameterBuffer],
    dispatch_counts: Option<BufferId>,
    group: u32,
) -> Result<Vec<ResolvedEntry>, CompiledGraphGpuError> {
    let mut entries = Vec::new();
    for attribute in graph
        .attribute_bindings
        .iter()
        .filter(|binding| binding.group == group)
    {
        let external = external
            .iter()
            .find(|entry| entry.slot() == (group, attribute.binding))
            .expect("validated attribute external binding");
        let (buffer, size) = match external {
            CompiledGraphGpuBinding::Buffer { buffer, size, .. } => (*buffer, *size),
            CompiledGraphGpuBinding::TextureView { .. } => unreachable!("validated buffer binding"),
        };
        entries.push(ResolvedEntry {
            binding: attribute.binding,
            layout: storage_layout_entry(attribute.binding, attribute.access),
            resource: BindingResource::Buffer {
                buffer,
                offset: 0,
                size,
            },
        });
    }
    for binding in graph
        .bindings
        .iter()
        .filter(|binding| binding.group == group)
    {
        if binding.name == "ember_dispatch_counts" {
            let Some(buffer) = dispatch_counts else {
                continue;
            };
            entries.push(ResolvedEntry {
                binding: binding.binding,
                layout: storage_layout_entry(binding.binding, binding.access),
                resource: BindingResource::Buffer {
                    buffer,
                    offset: 0,
                    size: None,
                },
            });
            continue;
        }
        match binding.kind {
            ResourceKind::Uniform => {
                let Some(parameter) = parameter_buffers.iter().find(|parameter| {
                    parameter.resource == binding.name
                        && parameter.group == binding.group
                        && parameter.binding == binding.binding
                }) else {
                    return Err(CompiledGraphGpuError::MissingParameterBlock {
                        group,
                        binding: binding.binding,
                        name: String::from(binding.name),
                    });
                };
                entries.push(ResolvedEntry {
                    binding: binding.binding,
                    layout: BindGroupLayoutEntry::uniform(binding.binding, ShaderStages::COMPUTE),
                    resource: BindingResource::Buffer {
                        buffer: parameter.buffer,
                        offset: 0,
                        size: Some(parameter.size),
                    },
                });
            }
            ResourceKind::Curve | ResourceKind::StorageBuffer => {
                let external = external
                    .iter()
                    .find(|entry| entry.slot() == (group, binding.binding))
                    .expect("validated storage external binding");
                let (buffer, size) = match external {
                    CompiledGraphGpuBinding::Buffer { buffer, size, .. } => (*buffer, *size),
                    CompiledGraphGpuBinding::TextureView { .. } => {
                        unreachable!("validated buffer binding")
                    }
                };
                entries.push(ResolvedEntry {
                    binding: binding.binding,
                    layout: storage_layout_entry(binding.binding, binding.access),
                    resource: BindingResource::Buffer {
                        buffer,
                        offset: 0,
                        size,
                    },
                });
            }
            ResourceKind::Texture2d | ResourceKind::Texture3d => {
                let external = external
                    .iter()
                    .find(|entry| entry.slot() == (group, binding.binding))
                    .expect("validated texture external binding");
                let view = match external {
                    CompiledGraphGpuBinding::TextureView { view, .. } => *view,
                    CompiledGraphGpuBinding::Buffer { .. } => {
                        unreachable!("validated texture binding")
                    }
                };
                entries.push(ResolvedEntry {
                    binding: binding.binding,
                    layout: texture_layout_entry(binding),
                    resource: BindingResource::TextureView(view),
                });
            }
        }
    }
    Ok(entries)
}

fn storage_layout_entry(binding: u32, access: ResourceAccess) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage {
                read_only: access == ResourceAccess::ReadOnly,
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_layout_entry(binding: &ResourceDecl) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding: binding.binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float,
            view_dimension: match binding.kind {
                ResourceKind::Texture2d => TextureViewDimension::D2,
                ResourceKind::Texture3d => TextureViewDimension::D3,
                _ => unreachable!("texture layout requested for non-texture"),
            },
            multisampled: false,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::particle::attributes::AttributeLayoutPlan;
    use prism_render_architecture::particle::compiler::{
        CompiledParameterBlock, CompiledParameterField,
    };
    use prism_render_architecture::particle::modules::{ParameterType, ResourceDecl};
    use prism_render_driver::Instance;
    use prism_render_driver_null::NullInstance;

    fn graph_with_one_parameter() -> CompiledGraph {
        CompiledGraph {
            source: String::new(),
            stage_order: Vec::new(),
            entry_points: Vec::new(),
            dispatches: Vec::new(),
            bindings: vec![ResourceDecl::uniform("params", 2, 0)],
            parameter_blocks: vec![CompiledParameterBlock {
                resource: String::from("params"),
                group: 2,
                binding: 0,
                fields: vec![CompiledParameterField {
                    name: String::from("gain"),
                    ty: ParameterType::F32,
                    offset: 0,
                    default_bytes: 1.0f32.to_bits().to_le_bytes().to_vec(),
                }],
                size: 16,
                default_bytes: {
                    let mut bytes = vec![0u8; 16];
                    bytes[..4].copy_from_slice(&1.0f32.to_bits().to_le_bytes());
                    bytes
                },
            }],
            attribute_layout: AttributeLayoutPlan::build(0, &[]),
            attribute_bindings: Vec::new(),
        }
    }

    #[test]
    fn compiled_graph_state_owns_parameter_buffer_and_bindings() {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&Default::default())
            .expect("null adapter");
        let (device, queue) = instance
            .create_device(&adapter, &Default::default())
            .expect("null device");
        let graph = graph_with_one_parameter();
        let state = CompiledGraphGpuState::new(&device, &queue, &graph, &[])
            .expect("uniform resources are self-contained");
        assert!(state.bind_group_layout(2).is_some());
        assert!(state.bind_group(2).is_some());
        assert!(state.pipeline_layout().index() != u32::MAX);
        state
            .update_parameters(
                &queue,
                &[PreparedParameterBlock {
                    resource: String::from("params"),
                    group: 2,
                    binding: 0,
                    bytes: {
                        let mut bytes = vec![0u8; 16];
                        bytes[..4].copy_from_slice(&2.0f32.to_bits().to_le_bytes());
                        bytes
                    },
                }],
            )
            .expect("compiled parameter ABI update");
        assert!(device.live_resource_count() >= 5);
        state.destroy(&device);
        assert_eq!(device.live_resource_count(), 0);
    }

    #[test]
    fn parameter_update_rejects_incomplete_and_wrong_sized_payloads() {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&Default::default())
            .expect("null adapter");
        let (device, queue) = instance
            .create_device(&adapter, &Default::default())
            .expect("null device");
        let graph = graph_with_one_parameter();
        let state = CompiledGraphGpuState::new(&device, &queue, &graph, &[])
            .expect("uniform resources are self-contained");
        let error = state
            .update_parameters(&queue, &[])
            .expect_err("missing parameter must be observable");
        assert_eq!(
            error,
            ParameterUploadError::MissingResource {
                resource: String::from("params")
            }
        );
        let error = state
            .update_parameters(
                &queue,
                &[PreparedParameterBlock {
                    resource: String::from("params"),
                    group: 2,
                    binding: 0,
                    bytes: vec![0; 4],
                }],
            )
            .expect_err("wrong parameter size must be observable");
        assert_eq!(
            error,
            ParameterUploadError::SizeMismatch {
                resource: String::from("params"),
                expected: 16,
                actual: 4,
            }
        );
        state.destroy(&device);
    }
}
