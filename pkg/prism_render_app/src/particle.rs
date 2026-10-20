//! Render-world bridge for the Ember CPU snapshot and the portable GPU sprite pass.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use alloc::vec::Vec;
use std::sync::Arc;

use prism_ecs::prelude::{Res, ResMut, Resource};
use prism_ecs::world::World;
use prism_particle_app::ExtractedParticleSnapshot;
use prism_render_architecture::particle::asset_registry::{
    EmberAssetRevision, EmberCompiledGraphRegistry,
};
use prism_render_architecture::particle::authoring::{
    EmberAsset, EmberAssetHandle, GraphAssetRef, ParamHandle, ParamValue,
};
use prism_render_architecture::particle::compiler::{
    AssetParameterError, CompiledGraph, PreparedParameterBlock,
};
use prism_render_driver::{
    BufferId, BufferState, BufferUsages, FrameSync, Instance, PipelineStages, RenderDevice,
    RenderQueue,
};
use prism_render_gpudriven::{
    CompiledGraphGpuError, CompiledGraphGpuExternalResource, CompiledGraphGpuGraphBinding,
    CompiledGraphGpuResourceRegistry, CompiledGraphGpuRevision, CompiledGraphGpuState,
    CompiledGraphGraphError, ParameterUploadError, Particle, ParticlePipelineCache, ParticleStage,
    ParticleStageError,
};
use prism_render_graph::{BufferDesc, BufferHandle, ImportedBuffer, RenderGraph};

use crate::config::RenderAppConfig;
use crate::runtime_resource::RenderRuntimeResource;

/// GPU resources owned by the render world for the built-in Ember bridge.
///
/// The CPU runtime remains authoritative. This resource only owns persistent
/// GPU storage and a reusable conversion scratch vector; it never reads the
/// GPU particle pool back to the CPU.
pub struct ParticleGpuState {
    stage: ParticleStage,
    pipeline_cache: ParticlePipelineCache,
    scratch: Vec<Particle>,
    active_indices_scratch: Vec<u8>,
}

/// Result of the prepare upload, kept in the render world so the frame system
/// can turn malformed snapshots into an observable failed frame.
#[derive(Default)]
pub struct ParticleUploadStatus {
    /// The most recent upload error, if any.
    pub error: Option<ParticleStageError>,
    /// Why the optional particle renderer is currently disabled, if it could
    /// not be created for the active device or configuration.
    pub disabled_reason: Option<ParticleStageError>,
}

impl Resource for ParticleUploadStatus {}

impl Resource for ParticleGpuState {}

/// Main-world registry of compiled Ember graphs.
///
/// Authored `.ember` assets refer to graphs through [`GraphAssetRef`]. The
/// graph registry is the sole CPU-side publication point for compiled graphs;
/// RenderApp extracts a revisioned snapshot from it instead of accepting a
/// compiled graph embedded in every render asset entry.
#[derive(Default)]
pub struct EmberCompiledGraphAssets {
    registry: EmberCompiledGraphRegistry,
}

/// Main-world publication point for caller-owned Ember GPU resources.
///
/// Resource ids are copyable Prism RHI handles. Their owning render-world
/// systems publish them here together with a stable slot revision; extraction
/// preserves that revision so a render frame cannot mistake a copied snapshot
/// for a newly-created GPU resource.
#[derive(Clone, Default)]
pub struct EmberExternalGpuResources {
    registry: CompiledGraphGpuResourceRegistry,
}

impl Resource for EmberExternalGpuResources {}

impl EmberExternalGpuResources {
    /// Creates an empty external resource registry.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            registry: CompiledGraphGpuResourceRegistry::new(),
        }
    }

    /// Returns the published registry.
    #[must_use]
    pub const fn registry(&self) -> &CompiledGraphGpuResourceRegistry {
        &self.registry
    }

    /// Returns the mutable registry used by resource owners.
    #[must_use]
    pub const fn registry_mut(&mut self) -> &mut CompiledGraphGpuResourceRegistry {
        &mut self.registry
    }
}

impl Resource for EmberCompiledGraphAssets {}

impl EmberCompiledGraphAssets {
    /// Creates an empty compiled-graph registry.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            registry: EmberCompiledGraphRegistry::new(),
        }
    }

    /// Returns the underlying publication registry.
    #[must_use]
    pub const fn registry(&self) -> &EmberCompiledGraphRegistry {
        &self.registry
    }

    /// Returns the mutable publication registry used by the compiler host.
    #[must_use]
    pub const fn registry_mut(&mut self) -> &mut EmberCompiledGraphRegistry {
        &mut self.registry
    }
}

/// Main-world Ember asset input consumed by the render bridge.
struct EmberRenderAssetInput {
    revision: EmberAssetRevision,
    asset: Arc<EmberAsset>,
    runtime_values: Arc<Vec<(ParamHandle, ParamValue)>>,
    stage_counts: Arc<Vec<u32>>,
    attribute_bytes: Arc<Vec<u8>>,
}

/// Main-world collection of authored Ember asset instances to render.
#[derive(Default)]
pub struct EmberRenderAssets {
    entries: BTreeMap<EmberAssetHandle, EmberRenderAssetInput>,
}

impl Resource for EmberRenderAssets {}

impl EmberRenderAssets {
    /// Replaces one authored asset input atomically from the host's point of
    /// view. The compiled graph is resolved by [`EmberCompiledGraphAssets`]
    /// through the asset's stable graph reference during extraction.
    pub fn upsert(
        &mut self,
        handle: EmberAssetHandle,
        revision: u64,
        asset: EmberAsset,
        runtime_values: Vec<(ParamHandle, ParamValue)>,
        stage_counts: Vec<u32>,
        attribute_bytes: Vec<u8>,
    ) {
        self.entries.insert(
            handle,
            EmberRenderAssetInput {
                revision: EmberAssetRevision::from_value(revision),
                asset: Arc::new(asset),
                runtime_values: Arc::new(runtime_values),
                stage_counts: Arc::new(stage_counts),
                attribute_bytes: Arc::new(attribute_bytes),
            },
        );
    }

    /// Removes one authored asset input.
    pub fn remove(&mut self, handle: EmberAssetHandle) -> bool {
        self.entries.remove(&handle).is_some()
    }

    /// Returns the deterministic input set.
    pub fn iter(&self) -> impl Iterator<Item = (EmberAssetHandle, &EmberRenderAssetInput)> {
        self.entries.iter().map(|(handle, asset)| (*handle, asset))
    }

    /// Removes every authored asset input.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

/// One extracted Ember asset plus the graph publication it resolved to.
#[derive(Clone)]
pub struct EmberRenderAsset {
    /// Monotonic authored asset revision.
    pub asset_revision: EmberAssetRevision,
    /// Stable graph reference stored in the authored asset.
    pub graph: GraphAssetRef,
    /// Revision of the compiled graph publication, if available.
    pub graph_revision: Option<EmberAssetRevision>,
    /// Authored parameter targets and renderer metadata.
    pub asset: Arc<EmberAsset>,
    /// Naga-validated compiled graph resolved from the formal graph registry.
    pub compiled: Option<Arc<CompiledGraph>>,
    /// Runtime values for externally driven authored parameters.
    pub runtime_values: Arc<Vec<(ParamHandle, ParamValue)>>,
    /// Invocation count for each compiled stage index.
    pub stage_counts: Arc<Vec<u32>>,
    /// Bytes uploaded to the graph's persistent SoA attribute pool.
    pub attribute_bytes: Arc<Vec<u8>>,
}

/// Render-world projection of [`EmberRenderAssets`].
#[derive(Clone, Default)]
pub struct ExtractedEmberRenderAssets {
    entries: BTreeMap<EmberAssetHandle, EmberRenderAsset>,
}

impl Resource for ExtractedEmberRenderAssets {}

/// Render-world snapshot of [`EmberExternalGpuResources`].
#[derive(Clone, Default)]
pub struct ExtractedEmberExternalGpuResources {
    registry: CompiledGraphGpuResourceRegistry,
}

impl Resource for ExtractedEmberExternalGpuResources {}

impl ExtractedEmberExternalGpuResources {
    /// Returns the exact revisioned snapshot consumed by prepare.
    #[must_use]
    pub const fn registry(&self) -> &CompiledGraphGpuResourceRegistry {
        &self.registry
    }
}

/// Incrementally extracts changed Ember graph inputs without copying compiled
/// shader source or unchanged runtime payloads.
pub fn extract_ember_render_assets(main: &mut World, render: &mut World) {
    render.init_resource::<ExtractedEmberRenderAssets>();
    let Some(source) = main.get_resource::<EmberRenderAssets>() else {
        render
            .resource_mut::<ExtractedEmberRenderAssets>()
            .entries
            .clear();
        return;
    };
    let source_entries: Vec<_> = source.iter().collect();
    let graphs = main.get_resource::<EmberCompiledGraphAssets>();
    let extracted = render.resource_mut::<ExtractedEmberRenderAssets>();
    let live_handles: BTreeSet<_> = source_entries.iter().map(|(handle, _)| *handle).collect();
    extracted
        .entries
        .retain(|handle, _| live_handles.contains(handle));
    for (handle, entry) in source_entries {
        let graph = entry.asset.graph;
        let graph_entry = graphs.and_then(|graphs| graphs.registry().get(graph));
        let graph_revision = graph_entry.map(|entry| entry.revision());
        let unchanged = extracted.entries.get(&handle).is_some_and(|current| {
            current.asset_revision == entry.revision
                && current.graph_revision == graph_revision
                && current.graph == graph
        });
        if !unchanged {
            extracted.entries.insert(
                handle,
                EmberRenderAsset {
                    asset_revision: entry.revision,
                    graph,
                    graph_revision,
                    asset: entry.asset.clone(),
                    compiled: graph_entry.map(|entry| Arc::new(entry.graph().clone())),
                    runtime_values: entry.runtime_values.clone(),
                    stage_counts: entry.stage_counts.clone(),
                    attribute_bytes: entry.attribute_bytes.clone(),
                },
            );
        }
    }
}

/// Extracts caller-owned Ember GPU resources without minting new revisions.
pub fn extract_ember_external_gpu_resources(main: &mut World, render: &mut World) {
    render.init_resource::<ExtractedEmberExternalGpuResources>();
    let snapshot = main
        .get_resource::<EmberExternalGpuResources>()
        .map(|resources| resources.registry.clone())
        .unwrap_or_default();
    render
        .resource_mut::<ExtractedEmberExternalGpuResources>()
        .registry = snapshot;
}

/// A compiled Ember graph and its device-owned resources in the render world.
struct EmberGpuAsset {
    graph: GraphAssetRef,
    graph_revision: EmberAssetRevision,
    compiled: CompiledGraph,
    state: CompiledGraphGpuState,
    stage_counts: Vec<u32>,
}

/// Render-world registry for compiled Ember graph resources.
///
/// The CPU [`EmberAssetRegistry`](prism_render_architecture::particle::asset_registry::EmberAssetRegistry)
/// owns authored assets and their revisions. This registry owns only the GPU
/// realization of a compiled graph. A lower revision is rejected before any
/// old GPU state is replaced, so an asynchronous compile cannot roll the
/// render world backward.
#[derive(Default)]
pub struct EmberGpuAssetRegistry {
    entries: BTreeMap<EmberAssetHandle, EmberGpuAsset>,
}

impl Resource for EmberGpuAssetRegistry {}

/// Failure while publishing a compiled graph into the render-world registry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EmberGpuAssetRegistryError {
    /// A refresh targeted an asset that is not resident on the device.
    MissingAsset {
        /// Asset handle that was not found.
        handle: EmberAssetHandle,
    },
    /// A stale compiled result attempted to replace a newer GPU graph.
    StaleRevision {
        /// Asset whose result was stale.
        handle: EmberAssetHandle,
        /// Current live revision.
        current: EmberAssetRevision,
        /// Revision supplied by the caller.
        incoming: EmberAssetRevision,
    },
    /// The graph could not be materialized on the active device.
    Gpu(CompiledGraphGpuError),
}

/// Failure while preparing one extracted Ember graph for a frame.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EmberGpuPrepareError {
    /// The authored asset references a graph that is not currently published.
    GraphUnavailable,
    /// The authored parameters could not be resolved into the compiled ABI.
    Parameters(AssetParameterError),
    /// The prepared parameter bytes did not match the GPU state.
    ParameterUpload(ParameterUploadError),
    /// The stage-count array did not match the compiled dispatch ABI.
    DispatchCounts,
    /// The persistent attribute pool rejected the upload range.
    Attributes(prism_render_gpudriven::AttributeUploadError),
    /// The graph has caller-owned resources that the built-in host bridge has
    /// not supplied yet.
    ExternalResourcesRequired,
}

/// Stable error category for Ember frame diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmberPrepareErrorKind {
    /// The authored asset references a graph that is not currently published.
    GraphUnavailable,
    /// GPU graph construction failed.
    GpuCreation,
    /// Authored parameters could not be resolved.
    Parameters,
    /// Prepared uniform bytes were rejected by the GPU state.
    ParameterUpload,
    /// Runtime stage counts did not match the compiled dispatch ABI.
    DispatchCounts,
    /// The persistent attribute pool rejected the upload.
    Attributes,
    /// A caller-owned graph resource was not supplied.
    ExternalResources,
}

/// The result of preparing the current Ember inputs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmberPrepareOutcome {
    /// No Ember input was prepared this frame.
    Idle,
    /// At least one graph was successfully published or updated.
    Updated,
    /// A failed update left an older valid graph resident.
    PreservedPrevious,
    /// A graph failed before any valid GPU graph existed.
    InitialFailure,
}

impl Default for EmberPrepareOutcome {
    fn default() -> Self {
        Self::Idle
    }
}

/// Render-world diagnostics for Ember preparation and graph submission.
#[derive(Default)]
pub struct EmberPrepareStatus {
    /// The latest preparation or graph-recording error.
    pub error: Option<String>,
    /// Whether the current frame must be rejected because no valid graph
    /// state exists for an input.
    pub fatal: bool,
    /// Stable category for frame statistics and diagnostics consumers.
    pub error_kind: Option<EmberPrepareErrorKind>,
    /// Number of graph inputs successfully uploaded for this frame.
    pub updated_graphs: u32,
    /// Number of failed updates for which an older valid graph remains live.
    pub preserved_graphs: u32,
    /// Number of graph inputs that failed before any valid GPU graph existed.
    pub initial_failures: u32,
    /// Aggregate outcome for this preparation pass.
    pub outcome: EmberPrepareOutcome,
}

impl Resource for EmberPrepareStatus {}

impl EmberGpuAssetRegistry {
    /// Number of compiled graph resources currently resident on the device.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns the revision of a resident compiled graph.
    #[must_use]
    pub fn graph_revision(&self, handle: EmberAssetHandle) -> Option<EmberAssetRevision> {
        self.entries.get(&handle).map(|entry| entry.graph_revision)
    }

    /// Returns the stable graph reference of a resident compiled graph.
    #[must_use]
    pub fn graph(&self, handle: EmberAssetHandle) -> Option<GraphAssetRef> {
        self.entries.get(&handle).map(|entry| entry.graph)
    }

    /// Returns the compiled graph for frame preparation.
    #[must_use]
    pub fn compiled(&self, handle: EmberAssetHandle) -> Option<&CompiledGraph> {
        self.entries.get(&handle).map(|entry| &entry.compiled)
    }

    /// Returns the GPU state for frame-graph submission.
    #[must_use]
    pub fn state(&self, handle: EmberAssetHandle) -> Option<&CompiledGraphGpuState> {
        self.entries.get(&handle).map(|entry| &entry.state)
    }

    /// Materializes and publishes one compiled graph.
    ///
    /// GPU construction completes before an existing entry is replaced. A
    /// failed construction leaves the currently live state untouched.
    pub fn upsert(
        &mut self,
        device: &impl RenderDevice,
        queue: &impl RenderQueue,
        handle: EmberAssetHandle,
        graph: GraphAssetRef,
        graph_revision: EmberAssetRevision,
        compiled: CompiledGraph,
        external: &CompiledGraphGpuResourceRegistry,
    ) -> Result<(), EmberGpuAssetRegistryError> {
        if let Some(current) = self.entries.get(&handle) {
            if graph == current.graph && graph_revision <= current.graph_revision {
                return Err(EmberGpuAssetRegistryError::StaleRevision {
                    handle,
                    current: current.graph_revision,
                    incoming: graph_revision,
                });
            }
        }
        let state = CompiledGraphGpuState::new(device, queue, &compiled, external)
            .map_err(EmberGpuAssetRegistryError::Gpu)?;
        if let Some(previous) = self.entries.insert(
            handle,
            EmberGpuAsset {
                graph,
                graph_revision,
                compiled,
                state,
                stage_counts: Vec::new(),
            },
        ) {
            previous.state.destroy(device);
        }
        Ok(())
    }

    /// Removes one graph and releases all resources it owns.
    pub fn remove(&mut self, device: &impl RenderDevice, handle: EmberAssetHandle) -> bool {
        let Some(entry) = self.entries.remove(&handle) else {
            return false;
        };
        entry.state.destroy(device);
        true
    }

    /// Releases every resident graph while the device is still usable.
    pub fn destroy(self, device: &impl RenderDevice) {
        for entry in self.entries.into_values() {
            entry.state.destroy(device);
        }
    }

    /// Updates one resident graph's parameters, counts, and persistent SoA
    /// bytes for the current frame.
    pub fn prepare(
        &mut self,
        queue: &impl RenderQueue,
        handle: EmberAssetHandle,
        asset: &EmberAsset,
        runtime_values: &[(ParamHandle, ParamValue)],
        stage_counts: &[u32],
        attribute_bytes: &[u8],
    ) -> Result<(), EmberGpuPrepareError> {
        let Some(entry) = self.entries.get_mut(&handle) else {
            return Err(EmberGpuPrepareError::GraphUnavailable);
        };
        let prepared: Vec<PreparedParameterBlock> = entry
            .compiled
            .prepare_asset_parameters(asset, runtime_values)
            .map_err(EmberGpuPrepareError::Parameters)?;
        entry
            .state
            .update_parameters(queue, &prepared)
            .map_err(EmberGpuPrepareError::ParameterUpload)?;
        let mut resolved_stage_counts = stage_counts.to_vec();
        if resolved_stage_counts.is_empty() && !entry.compiled.dispatches.is_empty() {
            let slots = entry
                .compiled
                .dispatches
                .iter()
                .map(|dispatch| dispatch.stage_index + 1)
                .max()
                .unwrap_or(0);
            resolved_stage_counts.resize(slots, 0);
            for dispatch in &entry.compiled.dispatches {
                resolved_stage_counts[dispatch.stage_index] = dispatch.default_count;
            }
        }
        entry
            .state
            .update_dispatch_counts(queue, &resolved_stage_counts)
            .map_err(|_| EmberGpuPrepareError::DispatchCounts)?;
        entry
            .state
            .upload_attribute_bytes(queue, 0, attribute_bytes)
            .map_err(EmberGpuPrepareError::Attributes)?;
        entry.stage_counts.clear();
        entry.stage_counts.extend_from_slice(&resolved_stage_counts);
        Ok(())
    }

    /// Refreshes caller-owned bind groups for a newer external-resource
    /// publication while retaining the compiled graph and its pipelines.
    pub fn refresh_external_resources(
        &mut self,
        device: &impl RenderDevice,
        handle: EmberAssetHandle,
        registry: &CompiledGraphGpuResourceRegistry,
    ) -> Result<(), EmberGpuAssetRegistryError> {
        let Some(entry) = self.entries.get_mut(&handle) else {
            return Err(EmberGpuAssetRegistryError::MissingAsset { handle });
        };
        entry
            .state
            .refresh_external_resources(device, &entry.compiled, registry)
            .map(|_| ())
            .map_err(EmberGpuAssetRegistryError::Gpu)
    }

    /// Removes GPU entries not present in the latest extracted input set.
    pub fn retain_handles<I>(&mut self, device: &impl RenderDevice, handles: I)
    where
        I: IntoIterator<Item = EmberAssetHandle>,
    {
        let live: BTreeSet<_> = handles.into_iter().collect();
        let stale: Vec<_> = self
            .entries
            .keys()
            .copied()
            .filter(|handle| !live.contains(handle))
            .collect();
        for handle in stale {
            self.remove(device, handle);
        }
    }

    /// Number of compute passes the current graph set will append.
    #[must_use]
    pub fn compute_pass_count(&self) -> usize {
        self.entries
            .values()
            .map(|entry| entry.compiled.dispatches.len())
            .sum()
    }

    /// Appends every resident graph's compute stages to a frame graph.
    pub fn append_to_graph(
        &self,
        graph: &mut RenderGraph,
        frame_index: u64,
    ) -> Result<(), CompiledGraphGraphError> {
        for entry in self.entries.values() {
            let mut imported = BTreeMap::<BufferId, BufferHandle>::new();
            let mut bindings = Vec::new();
            let mut import = |id: BufferId, size: u64, usage: BufferUsages, state: BufferState| {
                if let Some(handle) = imported.get(&id) {
                    return *handle;
                }
                let handle = graph.import_buffer(
                    "ember-graph-buffer",
                    BufferDesc::new(size, usage),
                    ImportedBuffer {
                        buffer: id,
                        entry_state: state,
                    },
                );
                imported.insert(id, handle);
                handle
            };

            let attribute_handle = entry.state.attribute_pool_buffer().map(|buffer| {
                import(
                    buffer,
                    entry.state.attribute_pool_size(),
                    BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    BufferState::storage_read(PipelineStages::COMPUTE_SHADER),
                )
            });
            for attribute in &entry.compiled.attribute_bindings {
                let Some(handle) = attribute_handle else {
                    return Err(CompiledGraphGraphError::MissingBinding {
                        group: attribute.group,
                        binding: attribute.binding,
                        name: attribute.name.clone(),
                    });
                };
                bindings.push(CompiledGraphGpuGraphBinding::Buffer {
                    group: attribute.group,
                    binding: attribute.binding,
                    revision: CompiledGraphGpuRevision::from_value(0),
                    handle,
                });
            }
            for binding in &entry.compiled.bindings {
                let (buffer, size, usage, state) = if binding.name == "ember_dispatch_counts" {
                    let Some(buffer) = entry.state.dispatch_counts_buffer() else {
                        return Err(CompiledGraphGraphError::MissingBinding {
                            group: binding.group,
                            binding: binding.binding,
                            name: String::from(binding.name),
                        });
                    };
                    (
                        buffer,
                        (entry.state.dispatch_count_slots() * 4) as u64,
                        BufferUsages::STORAGE | BufferUsages::COPY_DST,
                        BufferState::storage_read(PipelineStages::COMPUTE_SHADER),
                    )
                } else if let Some((group, slot, buffer, size)) =
                    entry.state.parameter_buffer(binding.name)
                {
                    if (group, slot) != (binding.group, binding.binding) {
                        return Err(CompiledGraphGraphError::MissingBinding {
                            group: binding.group,
                            binding: binding.binding,
                            name: String::from(binding.name),
                        });
                    }
                    (
                        buffer,
                        size,
                        BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                        BufferState::uniform(PipelineStages::COMPUTE_SHADER),
                    )
                } else {
                    return Err(CompiledGraphGraphError::UnexpectedBinding {
                        group: binding.group,
                        binding: binding.binding,
                    });
                };
                let handle = import(buffer, size, usage, state);
                bindings.push(CompiledGraphGpuGraphBinding::Buffer {
                    group: binding.group,
                    binding: binding.binding,
                    revision: CompiledGraphGpuRevision::from_value(0),
                    handle,
                });
            }
            for external in entry.state.external_bindings() {
                let (group, binding) = (external.slot.group, external.slot.binding);
                let handle = match external.resource {
                    CompiledGraphGpuExternalResource::StorageBuffer { buffer, size }
                    | CompiledGraphGpuExternalResource::CurveBuffer { buffer, size } => import(
                        buffer,
                        size.unwrap_or(1).max(1),
                        BufferUsages::STORAGE,
                        BufferState::initial(),
                    ),
                    CompiledGraphGpuExternalResource::Texture { .. }
                    | CompiledGraphGpuExternalResource::TextureView { .. } => {
                        return Err(CompiledGraphGraphError::ExternalTextureImportRequired {
                            group,
                            binding,
                        });
                    }
                };
                if let Some(graph_binding) = entry
                    .state
                    .external_graph_buffer_binding(external.slot, handle)
                {
                    bindings.push(graph_binding);
                } else {
                    return Err(CompiledGraphGraphError::ExpectedBuffer {
                        group,
                        binding,
                        name: String::from("external Ember resource"),
                    });
                }
            }
            entry.state.append_to_graph_for_frame(
                &entry.compiled,
                graph,
                &bindings,
                &entry.stage_counts,
                frame_index,
            )?;
        }
        Ok(())
    }
}

impl ParticleGpuState {
    /// Creates the GPU stage for the current render device.
    pub fn new(
        device: &impl RenderDevice,
        capacity: u32,
        resolution: u32,
    ) -> Result<Self, ParticleStageError> {
        let mut pipeline_cache = ParticlePipelineCache::new();
        let stage = ParticleStage::new_with_pipeline_cache(
            device,
            capacity,
            resolution,
            &mut pipeline_cache,
        )?;
        Ok(Self {
            stage,
            pipeline_cache,
            scratch: Vec::with_capacity(capacity as usize),
            active_indices_scratch: Vec::with_capacity(capacity as usize * 4),
        })
    }

    /// The persistent GPU stage used by frame-graph composition.
    #[must_use]
    pub const fn stage(&self) -> &ParticleStage {
        &self.stage
    }

    /// Converts and uploads one extracted CPU snapshot without GPU simulation.
    pub fn upload_snapshot<I>(
        &mut self,
        queue: &I::Queue,
        snapshot: &ExtractedParticleSnapshot,
        config: &RenderAppConfig,
    ) -> Result<(), ParticleStageError>
    where
        I: Instance,
        I::Queue: RenderQueue,
    {
        self.scratch.clear();
        self.scratch
            .extend(snapshot.particles().iter().map(|entry| {
                let record = entry.particle;
                let remaining_life = (record.lifetime - record.age).max(0.0);
                Particle {
                    position: [record.position.x, record.position.y, record.position.z],
                    velocity: [record.velocity.x, record.velocity.y, record.velocity.z],
                    life: remaining_life,
                    padding: if record.size > 0.0 {
                        record.size
                    } else {
                        config.particle_size
                    },
                }
            }));
        self.stage.upload_render_pool_with_scratch(
            queue,
            &self.scratch,
            &mut self.active_indices_scratch,
            config.particle_world_min,
            config.particle_world_size,
            config.particle_size,
        )
    }

    /// Releases all resources while the owning device is still usable.
    pub fn destroy(mut self, device: &impl RenderDevice) {
        self.stage
            .destroy_with_pipeline_cache(device, &mut self.pipeline_cache);
    }
}

/// Render-startup system that constructs the particle stage against the current device.
pub fn init_particle_gpu_state<I>(world: &mut World)
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    if world.get_resource::<ParticleGpuState>().is_some() {
        return;
    }
    let config = *world.resource::<RenderAppConfig>();
    let runtime = world.resource::<RenderRuntimeResource<I>>();
    let resolution = config.extent.width.max(config.extent.height).max(1);
    match ParticleGpuState::new(
        runtime.runtime().device(),
        config.particle_capacity,
        resolution,
    ) {
        Ok(state) => {
            let status = world.resource_mut::<ParticleUploadStatus>();
            status.error = None;
            status.disabled_reason = None;
            world.insert_resource(state);
        }
        Err(error) => {
            let status = world.resource_mut::<ParticleUploadStatus>();
            status.error = None;
            status.disabled_reason = Some(error);
        }
    }
}

/// Uploads the extracted snapshot during the render prepare phase.
pub fn prepare_particle_snapshot<I>(
    state: Option<ResMut<ParticleGpuState>>,
    mut status: ResMut<ParticleUploadStatus>,
    snapshot: Res<ExtractedParticleSnapshot>,
    config: Res<RenderAppConfig>,
    runtime: Res<RenderRuntimeResource<I>>,
) where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    let Some(mut state) = state else {
        status.error = None;
        return;
    };
    status.error = state
        .upload_snapshot::<I>(runtime.runtime().queue(), &snapshot, &config)
        .err();
}

/// Prepares every changed Ember graph and uploads its current runtime inputs.
pub fn prepare_ember_graphs<I>(
    mut registry: ResMut<EmberGpuAssetRegistry>,
    extracted: Res<ExtractedEmberRenderAssets>,
    external: Res<ExtractedEmberExternalGpuResources>,
    mut status: ResMut<EmberPrepareStatus>,
    runtime: Res<RenderRuntimeResource<I>>,
) where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    status.error = None;
    status.fatal = false;
    status.error_kind = None;
    status.updated_graphs = 0;
    status.preserved_graphs = 0;
    status.initial_failures = 0;
    status.outcome = EmberPrepareOutcome::Idle;
    registry.retain_handles(
        runtime.runtime().device(),
        extracted.entries.keys().copied(),
    );

    for (handle, input) in &extracted.entries {
        let Some(compiled) = input.compiled.as_ref() else {
            status.error = Some(format!(
                "Ember graph {handle:?} references unavailable graph {:?}",
                input.graph
            ));
            status.error_kind = Some(EmberPrepareErrorKind::GraphUnavailable);
            if registry.graph(*handle).is_some() {
                status.preserved_graphs = status.preserved_graphs.saturating_add(1);
                status.outcome = EmberPrepareOutcome::PreservedPrevious;
            } else {
                status.initial_failures = status.initial_failures.saturating_add(1);
                status.fatal = true;
                status.outcome = EmberPrepareOutcome::InitialFailure;
            }
            continue;
        };
        let Some(graph_revision) = input.graph_revision else {
            status.error = Some(format!(
                "Ember graph {handle:?} has compiled bytes without a publication revision"
            ));
            status.error_kind = Some(EmberPrepareErrorKind::GraphUnavailable);
            if registry.graph(*handle).is_some() {
                status.preserved_graphs = status.preserved_graphs.saturating_add(1);
                status.outcome = EmberPrepareOutcome::PreservedPrevious;
            } else {
                status.initial_failures = status.initial_failures.saturating_add(1);
                status.fatal = true;
                status.outcome = EmberPrepareOutcome::InitialFailure;
            }
            continue;
        };
        let needs_upsert = registry.graph(*handle) != Some(input.graph)
            || registry.graph_revision(*handle) != Some(graph_revision);
        if needs_upsert
            && let Err(error) = registry.upsert(
                runtime.runtime().device(),
                runtime.runtime().queue(),
                *handle,
                input.graph,
                graph_revision,
                (**compiled).clone(),
                external.registry(),
            )
        {
            status.error = Some(format!(
                "Ember graph {handle:?} GPU creation failed: {error:?}"
            ));
            status.error_kind = Some(EmberPrepareErrorKind::GpuCreation);
            if registry.graph(*handle).is_some() {
                status.preserved_graphs = status.preserved_graphs.saturating_add(1);
                status.outcome = EmberPrepareOutcome::PreservedPrevious;
            } else {
                status.initial_failures = status.initial_failures.saturating_add(1);
                status.fatal = true;
                status.outcome = EmberPrepareOutcome::InitialFailure;
            }
            continue;
        }
        if !needs_upsert
            && let Err(error) = registry.refresh_external_resources(
                runtime.runtime().device(),
                *handle,
                external.registry(),
            )
        {
            status.error = Some(format!(
                "Ember graph {handle:?} external resource refresh failed: {error:?}"
            ));
            status.error_kind = Some(EmberPrepareErrorKind::ExternalResources);
            if registry.graph(*handle).is_some() {
                status.preserved_graphs = status.preserved_graphs.saturating_add(1);
                status.outcome = EmberPrepareOutcome::PreservedPrevious;
            } else {
                status.initial_failures = status.initial_failures.saturating_add(1);
                status.fatal = true;
                status.outcome = EmberPrepareOutcome::InitialFailure;
            }
            continue;
        }

        if registry.graph(*handle) != Some(input.graph)
            || registry.graph_revision(*handle) != Some(graph_revision)
        {
            continue;
        }
        if let Err(error) = registry.prepare(
            runtime.runtime().queue(),
            *handle,
            &input.asset,
            &input.runtime_values,
            &input.stage_counts,
            &input.attribute_bytes,
        ) {
            status.error = Some(format!(
                "Ember graph {handle:?} frame preparation failed: {error:?}"
            ));
            status.error_kind = Some(match error {
                EmberGpuPrepareError::GraphUnavailable => EmberPrepareErrorKind::GraphUnavailable,
                EmberGpuPrepareError::Parameters(_) => EmberPrepareErrorKind::Parameters,
                EmberGpuPrepareError::ParameterUpload(_) => EmberPrepareErrorKind::ParameterUpload,
                EmberGpuPrepareError::DispatchCounts => EmberPrepareErrorKind::DispatchCounts,
                EmberGpuPrepareError::Attributes(_) => EmberPrepareErrorKind::Attributes,
                EmberGpuPrepareError::ExternalResourcesRequired => {
                    EmberPrepareErrorKind::ExternalResources
                }
            });
            status.preserved_graphs = status.preserved_graphs.saturating_add(1);
            status.outcome = EmberPrepareOutcome::PreservedPrevious;
            // A failed update leaves the previous GPU graph live. The frame
            // may continue using that last known-good state.
        } else {
            status.updated_graphs = status.updated_graphs.saturating_add(1);
            status.outcome = EmberPrepareOutcome::Updated;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        extract_ember_external_gpu_resources, extract_ember_render_assets,
        EmberCompiledGraphAssets, EmberExternalGpuResources, EmberGpuAssetRegistry,
        EmberRenderAssets, ExtractedEmberExternalGpuResources, ExtractedEmberRenderAssets,
        ParticleGpuState, ParticleStageError, ParticleUploadStatus,
    };
    use crate::{PrismRenderPlugin, RenderApp, RenderAppConfig};
    use prism_app::App;
    use prism_ecs::world::World;
    use prism_render_architecture::particle::attributes::AttributeLayoutPlan;
    use prism_render_architecture::particle::authoring::{
        EmberAsset, GraphAssetRef, RendererConfig, RendererConfigId, RendererKind,
        SerializationFormat, ShadingSettings,
    };
    use prism_render_architecture::particle::compiler::CompiledGraph;
    use prism_render_architecture::particle::EmberShadingModel;
    use prism_render_driver_null::NullInstance;
    use prism_render_gpudriven::{
        CompiledGraphGpuExternalKind, CompiledGraphGpuExternalResource,
        CompiledGraphGpuResourceSlot,
    };

    fn empty_asset() -> EmberAsset {
        EmberAsset {
            format_version: 1,
            graph: GraphAssetRef(1),
            params: Vec::new(),
            renderer_config: RendererConfig {
                id: RendererConfigId(0),
                kind: RendererKind::Sprite,
                writes_motion_vectors: false,
            },
            shading: ShadingSettings {
                model: EmberShadingModel::Unlit,
                receive_shadows: false,
                cast_shadows: false,
            },
            format: SerializationFormat::Binary,
        }
    }

    fn empty_compiled_graph() -> CompiledGraph {
        CompiledGraph {
            source: String::new(),
            stage_order: Vec::new(),
            entry_points: Vec::new(),
            dispatches: Vec::new(),
            bindings: Vec::new(),
            parameter_blocks: Vec::new(),
            attribute_layout: AttributeLayoutPlan::build(0, &[]),
            attribute_bindings: Vec::new(),
        }
    }

    #[test]
    fn unsupported_particle_device_disables_only_the_optional_particle_path() {
        let mut app = App::new();
        app.add_plugins(PrismRenderPlugin::with_config(
            NullInstance::new(),
            RenderAppConfig::default(),
        ));
        app.update();

        let render = app
            .get_sub_app(RenderApp)
            .expect("the render sub-app is installed");
        assert!(render
            .world
            .get_resource::<EmberGpuAssetRegistry>()
            .is_some());
        assert!(render.world.get_resource::<ParticleGpuState>().is_none());
        let status = render.world.resource::<ParticleUploadStatus>();
        assert_eq!(status.error, None);
        assert_eq!(
            status.disabled_reason,
            Some(ParticleStageError::MissingIndirectDraw)
        );
    }

    #[test]
    fn ember_extract_reuses_unchanged_compiled_assets() {
        let mut main = World::new();
        main.insert_resource(EmberRenderAssets::default());
        main.insert_resource(EmberCompiledGraphAssets::default());
        main.resource_mut::<EmberCompiledGraphAssets>()
            .registry_mut()
            .insert(
                GraphAssetRef(1),
                prism_render_architecture::particle::asset_registry::EmberAssetRevision::from_value(
                    1,
                ),
                empty_compiled_graph(),
            )
            .expect("graph publication succeeds");
        main.resource_mut::<EmberRenderAssets>().upsert(
            prism_render_architecture::particle::authoring::EmberAssetHandle(7),
            1,
            empty_asset(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        let mut render = World::new();
        render.insert_resource(ExtractedEmberRenderAssets::default());

        extract_ember_render_assets(&mut main, &mut render);
        let handle = prism_render_architecture::particle::authoring::EmberAssetHandle(7);
        let first = render
            .resource::<ExtractedEmberRenderAssets>()
            .entries
            .get(&handle)
            .expect("first asset is extracted")
            .compiled
            .as_ref()
            .expect("published graph is extracted")
            .clone();
        extract_ember_render_assets(&mut main, &mut render);
        let second = render
            .resource::<ExtractedEmberRenderAssets>()
            .entries
            .get(&handle)
            .expect("unchanged asset remains extracted")
            .compiled
            .as_ref()
            .expect("published graph remains extracted")
            .clone();
        assert!(std::sync::Arc::ptr_eq(&first, &second));

        main.resource_mut::<EmberRenderAssets>().upsert(
            handle,
            2,
            empty_asset(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        extract_ember_render_assets(&mut main, &mut render);
        let third = render
            .resource::<ExtractedEmberRenderAssets>()
            .entries
            .get(&handle)
            .expect("new revision is extracted")
            .compiled
            .as_ref()
            .expect("published graph remains available")
            .clone();
        assert!(!std::sync::Arc::ptr_eq(&first, &third));
    }

    #[test]
    fn ember_extract_publishes_missing_graph_as_an_observable_unresolved_input() {
        let mut main = World::new();
        main.insert_resource(EmberRenderAssets::default());
        main.insert_resource(EmberCompiledGraphAssets::default());
        main.resource_mut::<EmberRenderAssets>().upsert(
            prism_render_architecture::particle::authoring::EmberAssetHandle(8),
            1,
            EmberAsset {
                graph: GraphAssetRef(404),
                ..empty_asset()
            },
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        let mut render = World::new();
        render.insert_resource(ExtractedEmberRenderAssets::default());

        extract_ember_render_assets(&mut main, &mut render);

        let extracted = render
            .resource::<ExtractedEmberRenderAssets>()
            .entries
            .get(&prism_render_architecture::particle::authoring::EmberAssetHandle(8))
            .expect("asset remains visible to render diagnostics");
        assert_eq!(extracted.graph, GraphAssetRef(404));
        assert_eq!(extracted.graph_revision, None);
        assert!(extracted.compiled.is_none());
    }

    #[test]
    fn ember_external_extract_preserves_published_slot_revision() {
        let mut main = World::new();
        main.insert_resource(EmberExternalGpuResources::default());
        let slot = CompiledGraphGpuResourceSlot::new(2, 7);
        let revision = main
            .resource_mut::<EmberExternalGpuResources>()
            .registry_mut()
            .set(
                slot,
                CompiledGraphGpuExternalResource::StorageBuffer {
                    buffer: prism_render_driver::BufferId::from_parts(4, 9),
                    size: Some(256),
                },
            )
            .expect("external slot publication succeeds");

        let mut render = World::new();
        render.insert_resource(ExtractedEmberExternalGpuResources::default());
        extract_ember_external_gpu_resources(&mut main, &mut render);

        let extracted = render
            .resource::<ExtractedEmberExternalGpuResources>()
            .registry()
            .get(slot)
            .expect("published external slot is extracted");
        assert_eq!(extracted.revision, revision);
        assert_eq!(extracted.slot, slot);
        assert_eq!(
            extracted.resource.kind(),
            CompiledGraphGpuExternalKind::StorageBuffer
        );
    }
}
