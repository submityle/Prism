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
use prism_render_graph::{BufferHandle, PassFlags, RenderGraph, TextureHandle};
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

/// A graph-owned handle corresponding to one compiled Ember binding.
///
/// The real GPU object is still owned by the caller or by
/// [`CompiledGraphGpuState`]. The handle only lets the frame graph declare the
/// same resource access that the compiled bind group will use at execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompiledGraphGpuGraphBinding {
    /// A buffer imported or created in the caller's frame graph.
    Buffer {
        /// Compiled group index.
        group: u32,
        /// Compiled binding index.
        binding: u32,
        /// Frame-graph buffer handle for the same real buffer bound to Ember.
        handle: BufferHandle,
    },
    /// A texture imported or created in the caller's frame graph.
    Texture {
        /// Compiled group index.
        group: u32,
        /// Compiled binding index.
        binding: u32,
        /// Frame-graph texture handle for the same real view bound to Ember.
        handle: TextureHandle,
    },
}

impl CompiledGraphGpuGraphBinding {
    const fn slot(self) -> (u32, u32) {
        match self {
            Self::Buffer { group, binding, .. } | Self::Texture { group, binding, .. } => {
                (group, binding)
            }
        }
    }
}

/// A failure while declaring a compiled graph in a frame graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompiledGraphGraphError {
    /// A graph resource was supplied more than once for one binding slot.
    DuplicateBinding { group: u32, binding: u32 },
    /// A compiled non-uniform binding has no graph handle.
    MissingBinding {
        /// Compiled group index.
        group: u32,
        /// Compiled binding index.
        binding: u32,
        /// Compiled resource name.
        name: String,
    },
    /// A graph resource does not correspond to a compiled binding.
    UnexpectedBinding { group: u32, binding: u32 },
    /// The caller supplied a texture handle where a buffer was required.
    ExpectedBuffer {
        group: u32,
        binding: u32,
        name: String,
    },
    /// The caller supplied a buffer handle where a texture was required.
    ExpectedTexture {
        group: u32,
        binding: u32,
        name: String,
    },
    /// The runtime count array does not address every compiled stage index.
    DispatchCountLength { expected: usize, actual: usize },
    /// The state and compiled graph disagree about generated stage pipelines.
    PipelineCountMismatch { expected: usize, actual: usize },
}

/// A failure while updating the compiled graph's persistent attribute pool.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AttributeUploadError {
    /// The write range exceeds the compiled SoA pool.
    OutOfBounds {
        offset: u64,
        size: usize,
        capacity: u64,
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
    bind_groups: [Vec<BindGroupId>; 2],
    compute_pipelines: Vec<ComputePipelineId>,
    parameter_buffers: Vec<ParameterBuffer>,
    dispatch_counts: Option<BufferId>,
    dispatch_count_slots: usize,
    attribute_pool: Option<BufferId>,
    attribute_pool_size: u64,
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
        let attribute_pool_size = graph.attribute_layout.total_bytes;
        let attribute_pool = (attribute_pool_size != 0).then(|| {
            device.create_buffer(&BufferDescriptor {
                label: Some(String::from("ember-attribute-pool")),
                size: attribute_pool_size,
                usage: BufferUsages::STORAGE.union(BufferUsages::COPY_DST),
                mapped_at_creation: false,
            })
        });
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

        let dispatch_count_slots = graph
            .dispatches
            .iter()
            .map(|dispatch| dispatch.stage_index + 1)
            .max()
            .unwrap_or(0);
        let dispatch_counts = if dispatch_count_slots == 0 {
            None
        } else {
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some(String::from("ember-dispatch-counts")),
                size: dispatch_count_slots as u64 * 4,
                usage: BufferUsages::STORAGE.union(BufferUsages::COPY_DST),
                mapped_at_creation: false,
            });
            let mut bytes = vec![0; dispatch_count_slots * 4];
            for dispatch in &graph.dispatches {
                let offset = dispatch.stage_index * 4;
                bytes[offset..offset + 4].copy_from_slice(&dispatch.default_count.to_le_bytes());
            }
            queue.write_buffer(buffer, 0, &bytes);
            Some(buffer)
        };

        let mut bind_group_layouts = Vec::with_capacity(group_count);
        for group in 0..group_count as u32 {
            let entries = group_entries(
                graph,
                external,
                &parameter_buffers,
                dispatch_counts,
                attribute_pool,
                0,
                group,
            )?;
            let mut layout_entries: Vec<_> = entries.iter().map(|entry| entry.layout).collect();
            layout_entries.sort_by_key(|entry| entry.binding);
            bind_group_layouts.push(device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some(format!("ember-graph-group-{group}")),
                entries: layout_entries,
            }));
        }
        let mut bind_groups = [
            Vec::with_capacity(group_count),
            Vec::with_capacity(group_count),
        ];
        for frame_index in 0..2u64 {
            for group in 0..group_count as u32 {
                let entries = group_entries(
                    graph,
                    external,
                    &parameter_buffers,
                    dispatch_counts,
                    attribute_pool,
                    frame_index,
                    group,
                )?;
                let mut bind_entries: Vec<_> = entries
                    .into_iter()
                    .map(|entry| BindGroupEntry {
                        binding: entry.binding,
                        resource: entry.resource,
                    })
                    .collect();
                bind_entries.sort_by_key(|entry| entry.binding);
                bind_groups[frame_index as usize].push(device.create_bind_group(
                    &BindGroupDescriptor {
                        label: Some(format!("ember-graph-group-{group}-frame-{frame_index}")),
                        layout: Some(bind_group_layouts[group as usize]),
                        entries: bind_entries,
                    },
                ));
            }
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
            dispatch_count_slots,
            attribute_pool,
            attribute_pool_size,
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
        if counts.len() != self.dispatch_count_slots {
            return Err("dispatch-count length does not match compiled stages");
        }
        let mut bytes = Vec::with_capacity(counts.len() * 4);
        for count in counts {
            bytes.extend_from_slice(&count.to_le_bytes());
        }
        queue.write_buffer(buffer, 0, &bytes);
        Ok(())
    }

    /// Returns a state-owned parameter buffer and its compiled binding slot.
    ///
    /// Callers use this to import the buffer into the frame graph before
    /// [`Self::append_to_graph`] declares the uniform read.
    #[must_use]
    pub fn parameter_buffer(&self, resource: &str) -> Option<(u32, u32, BufferId, u64)> {
        self.parameter_buffers
            .iter()
            .find(|parameter| parameter.resource == resource)
            .map(|parameter| {
                (
                    parameter.group,
                    parameter.binding,
                    parameter.buffer,
                    parameter.size,
                )
            })
    }

    /// Returns the state-owned dispatch-count storage buffer.
    #[must_use]
    pub const fn dispatch_counts_buffer(&self) -> Option<BufferId> {
        self.dispatch_counts
    }

    /// Returns the number of stage-index slots in the dispatch-count ABI.
    #[must_use]
    pub const fn dispatch_count_slots(&self) -> usize {
        self.dispatch_count_slots
    }

    /// Returns the persistent SoA attribute pool, when the graph has live
    /// attributes.
    #[must_use]
    pub const fn attribute_pool_buffer(&self) -> Option<BufferId> {
        self.attribute_pool
    }

    /// Returns the total byte size of the persistent SoA attribute pool.
    #[must_use]
    pub const fn attribute_pool_size(&self) -> u64 {
        self.attribute_pool_size
    }

    /// Uploads a byte range into the persistent SoA attribute pool.
    pub fn upload_attribute_bytes(
        &self,
        queue: &impl RenderQueue,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), AttributeUploadError> {
        let end = offset.saturating_add(bytes.len() as u64);
        if end > self.attribute_pool_size {
            return Err(AttributeUploadError::OutOfBounds {
                offset,
                size: bytes.len(),
                capacity: self.attribute_pool_size,
            });
        }
        if let Some(buffer) = self.attribute_pool {
            queue.write_buffer(buffer, offset, bytes);
        } else if !bytes.is_empty() {
            return Err(AttributeUploadError::OutOfBounds {
                offset,
                size: bytes.len(),
                capacity: 0,
            });
        }
        Ok(())
    }

    /// Declares and records all compiled stage dispatches in a frame graph.
    ///
    /// `bindings` must contain one graph handle for every compiled resource
    /// binding, including the state-owned parameter and dispatch-count buffers.
    /// The handles must refer to the same real resources used when this state
    /// was created. This method records commands only; callers still upload
    /// parameters/counts and execute the returned graph through the runtime.
    pub fn append_to_graph(
        &self,
        compiled: &CompiledGraph,
        graph: &mut RenderGraph,
        bindings: &[CompiledGraphGpuGraphBinding],
        stage_counts: &[u32],
    ) -> Result<(), CompiledGraphGraphError> {
        self.append_to_graph_for_frame(compiled, graph, bindings, stage_counts, 0)
    }

    /// Variant of [`Self::append_to_graph`] selecting the ping-pong side for a
    /// specific frame parity.
    pub fn append_to_graph_for_frame(
        &self,
        compiled: &CompiledGraph,
        graph: &mut RenderGraph,
        bindings: &[CompiledGraphGpuGraphBinding],
        stage_counts: &[u32],
        frame_index: u64,
    ) -> Result<(), CompiledGraphGraphError> {
        if self.compute_pipelines.len() != compiled.dispatches.len() {
            return Err(CompiledGraphGraphError::PipelineCountMismatch {
                expected: compiled.dispatches.len(),
                actual: self.compute_pipelines.len(),
            });
        }
        if stage_counts.len() != self.dispatch_count_slots {
            return Err(CompiledGraphGraphError::DispatchCountLength {
                expected: self.dispatch_count_slots,
                actual: stage_counts.len(),
            });
        }

        let accesses = graph_accesses(compiled, bindings)?;
        for (pipeline_index, dispatch) in compiled.dispatches.iter().enumerate() {
            let workgroups = stage_counts
                .get(dispatch.stage_index)
                .copied()
                .unwrap_or(0)
                .saturating_add(63)
                / 64;
            let pipeline = self.compute_pipelines[pipeline_index];
            let bind_groups = self.bind_groups[(frame_index & 1) as usize].clone();
            let entry_name = dispatch.entry_point.clone();
            let iterations = dispatch.iterations;
            let accesses_for_setup = accesses.clone();
            graph.add_compute_pass(
                format!("ember-{entry_name}"),
                PassFlags::SIDE_EFFECT,
                move |builder| {
                    for access in &accesses_for_setup {
                        access.declare(builder);
                    }
                    move |context| {
                        context.set_compute_pipeline(pipeline);
                        for (group, bind_group) in bind_groups.iter().copied().enumerate() {
                            context.set_compute_bind_group(group as u32, bind_group, Vec::new());
                        }
                        for _ in 0..iterations {
                            if workgroups != 0 {
                                context.dispatch(workgroups, 1, 1);
                            }
                        }
                    }
                },
            );
        }
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
        self.bind_groups[0].get(group as usize).copied()
    }

    /// Returns a bind group for the selected ping-pong frame parity.
    #[must_use]
    pub fn bind_group_for_frame(&self, group: u32, frame_index: u64) -> Option<BindGroupId> {
        self.bind_groups[(frame_index & 1) as usize]
            .get(group as usize)
            .copied()
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
        for bind_groups in self.bind_groups {
            for bind_group in bind_groups {
                device.destroy_bind_group(bind_group);
            }
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
        if let Some(attribute_pool) = self.attribute_pool {
            device.destroy_buffer(attribute_pool);
        }
    }
}

#[derive(Clone, Copy)]
enum GraphAccess {
    Uniform(BufferHandle),
    StorageRead(BufferHandle),
    StorageWrite(BufferHandle),
    Sample(TextureHandle),
}

impl GraphAccess {
    fn declare(self, builder: &mut prism_render_graph::PassBuilder<'_>) {
        match self {
            Self::Uniform(handle) => {
                let _ = builder.read_uniform(handle);
            }
            Self::StorageRead(handle) => {
                let _ = builder.read_storage_buffer(handle);
            }
            Self::StorageWrite(handle) => {
                let _ = builder.write_storage_buffer(handle);
            }
            Self::Sample(handle) => {
                let _ = builder.sample(handle);
            }
        }
    }
}

fn graph_accesses(
    compiled: &CompiledGraph,
    bindings: &[CompiledGraphGpuGraphBinding],
) -> Result<Vec<GraphAccess>, CompiledGraphGraphError> {
    for (index, entry) in bindings.iter().enumerate() {
        if bindings[..index]
            .iter()
            .any(|previous| previous.slot() == entry.slot())
        {
            let (group, binding) = entry.slot();
            return Err(CompiledGraphGraphError::DuplicateBinding { group, binding });
        }
    }

    let mut accesses = Vec::new();
    let mut expected = Vec::new();
    for attribute in &compiled.attribute_bindings {
        expected.push((
            attribute.group,
            attribute.binding,
            attribute.name.as_str(),
            ResourceKind::StorageBuffer,
            attribute.access,
        ));
        let entry = bindings
            .iter()
            .find(|entry| entry.slot() == (attribute.group, attribute.binding))
            .ok_or_else(|| CompiledGraphGraphError::MissingBinding {
                group: attribute.group,
                binding: attribute.binding,
                name: attribute.name.clone(),
            })?;
        let CompiledGraphGpuGraphBinding::Buffer { handle, .. } = entry else {
            return Err(CompiledGraphGraphError::ExpectedBuffer {
                group: attribute.group,
                binding: attribute.binding,
                name: attribute.name.clone(),
            });
        };
        accesses.push(match attribute.access {
            ResourceAccess::ReadOnly => GraphAccess::StorageRead(*handle),
            ResourceAccess::ReadWrite => GraphAccess::StorageWrite(*handle),
        });
    }
    for binding in &compiled.bindings {
        if binding.name == "ember_dispatch_counts" {
            expected.push((
                binding.group,
                binding.binding,
                binding.name,
                binding.kind,
                binding.access,
            ));
            let entry = bindings
                .iter()
                .find(|entry| entry.slot() == (binding.group, binding.binding))
                .ok_or_else(|| CompiledGraphGraphError::MissingBinding {
                    group: binding.group,
                    binding: binding.binding,
                    name: String::from(binding.name),
                })?;
            let CompiledGraphGpuGraphBinding::Buffer { handle, .. } = entry else {
                return Err(CompiledGraphGraphError::ExpectedBuffer {
                    group: binding.group,
                    binding: binding.binding,
                    name: String::from(binding.name),
                });
            };
            accesses.push(GraphAccess::StorageRead(*handle));
            continue;
        }
        expected.push((
            binding.group,
            binding.binding,
            binding.name,
            binding.kind,
            binding.access,
        ));
        let entry = bindings
            .iter()
            .find(|entry| entry.slot() == (binding.group, binding.binding))
            .ok_or_else(|| CompiledGraphGraphError::MissingBinding {
                group: binding.group,
                binding: binding.binding,
                name: String::from(binding.name),
            })?;
        match (binding.kind, entry) {
            (ResourceKind::Uniform, CompiledGraphGpuGraphBinding::Buffer { handle, .. }) => {
                accesses.push(GraphAccess::Uniform(*handle));
            }
            (
                ResourceKind::Curve | ResourceKind::StorageBuffer,
                CompiledGraphGpuGraphBinding::Buffer { handle, .. },
            ) => {
                accesses.push(match binding.access {
                    ResourceAccess::ReadOnly => GraphAccess::StorageRead(*handle),
                    ResourceAccess::ReadWrite => GraphAccess::StorageWrite(*handle),
                });
            }
            (
                ResourceKind::Texture2d | ResourceKind::Texture3d,
                CompiledGraphGpuGraphBinding::Texture { handle, .. },
            ) => accesses.push(GraphAccess::Sample(*handle)),
            (ResourceKind::Uniform, _) => {
                return Err(CompiledGraphGraphError::ExpectedBuffer {
                    group: binding.group,
                    binding: binding.binding,
                    name: String::from(binding.name),
                });
            }
            (_, _) => {
                return Err(CompiledGraphGraphError::ExpectedTexture {
                    group: binding.group,
                    binding: binding.binding,
                    name: String::from(binding.name),
                });
            }
        }
    }
    for entry in bindings {
        if !expected
            .iter()
            .any(|(group, binding, _, _, _)| (*group, *binding) == entry.slot())
        {
            let (group, binding) = entry.slot();
            return Err(CompiledGraphGraphError::UnexpectedBinding { group, binding });
        }
    }
    Ok(accesses)
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
            .bindings
            .iter()
            .find(|binding| {
                (binding.group, binding.binding) == slot
                    && binding.name != "ember_dispatch_counts"
                    && binding.kind != ResourceKind::Uniform
            })
            .map(|binding| (binding.name, binding.kind));
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
    for (group, binding, name) in graph.bindings.iter().filter_map(|binding| {
        (binding.kind != ResourceKind::Uniform && binding.name != "ember_dispatch_counts")
            .then_some((binding.group, binding.binding, binding.name))
    }) {
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
    attribute_pool: Option<BufferId>,
    frame_index: u64,
    group: u32,
) -> Result<Vec<ResolvedEntry>, CompiledGraphGpuError> {
    let mut entries = Vec::new();
    for attribute in graph
        .attribute_bindings
        .iter()
        .filter(|binding| binding.group == group)
    {
        let pool = attribute_pool.expect("compiled attribute layout owns the SoA pool");
        let planned = graph
            .attribute_layout
            .get(attribute.semantic)
            .expect("compiled attribute binding has a planned layout");
        let per_copy = u64::from(planned.stride) * u64::from(graph.attribute_layout.capacity);
        let physical_copy = attribute.physical_copy(frame_index) as u64;
        let offset = planned.byte_offset + physical_copy * per_copy;
        entries.push(ResolvedEntry {
            binding: attribute.binding,
            layout: storage_layout_entry(attribute.binding, attribute.access),
            resource: BindingResource::Buffer {
                buffer: pool,
                offset,
                size: Some(per_copy),
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
    use prism_render_architecture::particle::attributes::{
        AttributeAccess, AttributeFormat, AttributeLayoutPlan, AttributeSemantic, AttributeUsage,
    };
    use prism_render_architecture::particle::compiler::{
        AttributeBufferRole, CompiledAttributeBinding, CompiledDispatch, CompiledParameterBlock,
        CompiledParameterField,
    };
    use prism_render_architecture::particle::modules::{ParameterType, ResourceDecl};
    use prism_render_architecture::particle::IterationDomain;
    use prism_render_driver::Instance;
    use prism_render_driver_null::NullInstance;
    use prism_render_graph::{BufferDesc, ImportedBuffer};

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

    fn graph_with_ping_pong_attribute() -> CompiledGraph {
        let layout = AttributeLayoutPlan::build(
            4,
            &[AttributeUsage::new(
                AttributeSemantic::Position,
                AttributeFormat::Vec3,
                AttributeAccess::READ_PREV.union(AttributeAccess::WRITE),
            )],
        );
        CompiledGraph {
            source: String::new(),
            stage_order: Vec::new(),
            entry_points: Vec::new(),
            dispatches: Vec::new(),
            bindings: Vec::new(),
            parameter_blocks: Vec::new(),
            attribute_layout: layout,
            attribute_bindings: vec![
                CompiledAttributeBinding {
                    semantic: AttributeSemantic::Position,
                    format: AttributeFormat::Vec3,
                    role: AttributeBufferRole::Previous,
                    name: String::from("position_previous"),
                    group: 0,
                    binding: 0,
                    access: ResourceAccess::ReadOnly,
                },
                CompiledAttributeBinding {
                    semantic: AttributeSemantic::Position,
                    format: AttributeFormat::Vec3,
                    role: AttributeBufferRole::Current,
                    name: String::from("position_current"),
                    group: 0,
                    binding: 1,
                    access: ResourceAccess::ReadWrite,
                },
            ],
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

    #[test]
    fn compiled_graph_records_stage_dispatches_into_render_graph() {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&Default::default())
            .expect("null adapter");
        let (device, queue) = instance
            .create_device(&adapter, &Default::default())
            .expect("null device");
        let graph = CompiledGraph {
            source: String::new(),
            stage_order: vec![2],
            entry_points: vec![String::from("ember_stage_2")],
            dispatches: vec![CompiledDispatch {
                stage_index: 2,
                domain: IterationDomain::PerParticle,
                default_count: 1,
                iterations: 2,
                entry_point: String::from("ember_stage_2"),
            }],
            bindings: vec![ResourceDecl::buffer("ember_dispatch_counts", 1, 0)],
            parameter_blocks: Vec::new(),
            attribute_layout: AttributeLayoutPlan::build(0, &[]),
            attribute_bindings: Vec::new(),
        };
        let state = CompiledGraphGpuState::new(&device, &queue, &graph, &[])
            .expect("dispatch count binding is state-owned");
        assert_eq!(state.dispatch_count_slots(), 3);
        state
            .update_dispatch_counts(&queue, &[0, 0, 65])
            .expect("counts are indexed by compiled stage index");

        let dispatch_buffer = state
            .dispatch_counts_buffer()
            .expect("compiled stage owns dispatch counts");
        let mut render_graph = RenderGraph::new(rhi::Extent3d::new_2d(1, 1));
        let dispatch_handle = render_graph.import_buffer(
            "ember-dispatch-counts",
            BufferDesc::new(12, BufferUsages::STORAGE),
            ImportedBuffer {
                buffer: dispatch_buffer,
                entry_state: rhi::BufferState::initial(),
            },
        );
        state
            .append_to_graph(
                &graph,
                &mut render_graph,
                &[CompiledGraphGpuGraphBinding::Buffer {
                    group: 1,
                    binding: 0,
                    handle: dispatch_handle,
                }],
                &[0, 0, 65],
            )
            .expect("compiled stages are declared in the frame graph");
        let recorded = render_graph
            .execute(&device, &queue)
            .expect("null RHI records the compute frame");
        assert_eq!(recorded.commands().passes.len(), 1);
        let prism_render_driver::Pass::Compute { commands, .. } = &recorded.commands().passes[0]
        else {
            panic!("compiled Ember stage must record a compute pass");
        };
        assert_eq!(
            commands
                .iter()
                .filter(|command| matches!(
                    command,
                    prism_render_driver::ComputeCommand::Dispatch { x: 2, y: 1, z: 1 }
                ))
                .count(),
            2
        );
        recorded.release(&device);
        state.destroy(&device);
        assert_eq!(device.live_resource_count(), 0);
    }

    #[test]
    fn attribute_pool_uses_layout_ranges_and_ping_pong_bindings() {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&Default::default())
            .expect("null adapter");
        let (device, queue) = instance
            .create_device(&adapter, &Default::default())
            .expect("null device");
        let graph = graph_with_ping_pong_attribute();
        assert_eq!(graph.attribute_layout.total_bytes, 128);

        let state = CompiledGraphGpuState::new(&device, &queue, &graph, &[])
            .expect("attribute pool is state-owned");
        let pool = state
            .attribute_pool_buffer()
            .expect("live attributes own a persistent pool");
        assert_eq!(state.attribute_pool_size(), 128);
        state
            .upload_attribute_bytes(&queue, 64, &[7; 64])
            .expect("an in-range copy upload succeeds");
        assert_eq!(
            state
                .upload_attribute_bytes(&queue, 120, &[7; 9])
                .expect_err("an upload past the pool must be rejected"),
            AttributeUploadError::OutOfBounds {
                offset: 120,
                size: 9,
                capacity: 128,
            }
        );

        let frame_zero = group_entries(&graph, &[], &[], None, Some(pool), 0, 0)
            .expect("frame-zero attribute bindings");
        let frame_one = group_entries(&graph, &[], &[], None, Some(pool), 1, 0)
            .expect("frame-one attribute bindings");
        assert_eq!(
            frame_zero[0].resource,
            BindingResource::Buffer {
                buffer: pool,
                offset: 64,
                size: Some(64),
            }
        );
        assert_eq!(
            frame_zero[1].resource,
            BindingResource::Buffer {
                buffer: pool,
                offset: 0,
                size: Some(64),
            }
        );
        assert_eq!(
            frame_one[0].resource,
            BindingResource::Buffer {
                buffer: pool,
                offset: 0,
                size: Some(64),
            }
        );
        assert_eq!(
            frame_one[1].resource,
            BindingResource::Buffer {
                buffer: pool,
                offset: 64,
                size: Some(64),
            }
        );
        assert_ne!(
            state.bind_group_for_frame(0, 0),
            state.bind_group_for_frame(0, 1)
        );

        state.destroy(&device);
        assert_eq!(device.live_resource_count(), 0);
    }
}
