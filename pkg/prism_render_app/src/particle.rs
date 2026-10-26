//! Render-world bridge for extracted particle bootstrap and GPU-authoritative rendering.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex, Weak};

pub use gameplay_feedback::extract_particle_simulation_inputs;
use prism_ecs::prelude::{Res, ResMut, Resource};
use prism_ecs::world::World;
pub use prism_particle_app::ExtractedParticleSimulationInputs as ParticleGpuSimulationInputs;
use prism_particle_app::{
    ExtractedParticleSnapshot, ParticleRendererComponent, ParticleSimulationInputError,
    ParticleSimulationLifecycle, ParticleSimulationStep, ParticleSurfaceRenderer,
};
use prism_particle_render::{
    particle_resource_payload, ParticleFabricError, ParticleFabricFrame,
    ParticleFabricGpuBuffer as SceneParticleFabricGpuBuffer,
    ParticleFabricGpuDraw as SceneParticleFabricGpuDraw, ParticleFabricScenePublication,
    PARTICLE_ALIVE_RESOURCE_PAYLOAD_ROLE, PARTICLE_EMITTER_COUNT_PAYLOAD_ROLE,
    PARTICLE_EMITTER_POSITION_BOUNDS_PAYLOAD_ROLE, PARTICLE_STREAM_RESOURCE_PAYLOAD_ROLE,
};
use prism_render_architecture::particle::asset_registry::{
    EmberAssetRevision, EmberCompiledGraphRegistry,
};
use prism_render_architecture::particle::authoring::{
    EmberAsset, EmberAssetHandle, GraphAssetRef, ParamHandle, ParamValue,
};
use prism_render_architecture::particle::compiler::{
    AssetParameterError, CompiledGraph, PreparedParameterBlock,
};
use prism_render_architecture::particle::renderers::RendererKind;
use prism_render_architecture::{
    motion::MotionSource,
    particle::{indirect_draw::RendererGeometry, SortStrategy},
    render_fabric::{
        DomainObjectHandle, DomainPayload, FrameRenderItem, PayloadRole, PayloadValue,
        RenderDomain, RenderFabricFrame, RenderFabricFrameError, RenderObjectId, ResourcePayload,
        ValuePayload, PARTICLE_ALIVE_RESOURCE, PARTICLE_STREAM_RESOURCE,
    },
};
use prism_render_driver::{
    BufferId, BufferState, BufferUsages, FrameSync, Instance, PipelineStages, RenderDevice,
    RenderQueue, TextureViewId,
};
use prism_render_gpudriven::particle::renderer_matrix::{
    ParticleCullView, ParticleEmitterGpuSource, ParticleEmitterRendererConfig,
    ParticleEmitterRendererGpu, ParticleMeshVertex, ParticleSurfaceMaterial,
};
use prism_render_gpudriven::particle::simulation_bridge::{
    PARTICLE_EMITTER_ROUTE_HEADER_BYTES, PARTICLE_EMITTER_ROUTE_STRIDE,
};
use prism_render_gpudriven::{
    CompiledGraphGpuError, CompiledGraphGpuExternalResource, CompiledGraphGpuGraphBinding,
    CompiledGraphGpuGraphImport, CompiledGraphGpuResourceRegistry, CompiledGraphGpuResourceSlot,
    CompiledGraphGpuRevision, CompiledGraphGpuState, CompiledGraphGraphError, GpuParticle,
    GpuSimulationGraphOutputs, GpuSimulationParams, GpuSimulationStage, ParameterUploadError,
    ParticleEmitterRenderRoute, ParticleGpuError, ParticleRendererGpu, ParticleRendererRoute,
    ParticleSimulationRendererBridge, PARTICLE_RENDER_INPUT_STRIDE,
};
use prism_render_graph::{BufferDesc, BufferHandle, ImportedBuffer, RenderGraph, TextureHandle};

use crate::config::RenderAppConfig;
use crate::runtime_resource::RenderRuntimeResource;

#[cfg(test)]
#[path = "particle/checkpoint_tests.rs"]
mod checkpoint_tests;
mod ember_build;
pub mod gameplay_feedback;
mod graph_compile;
#[cfg(test)]
#[path = "particle/multiview.rs"]
pub(crate) mod multiview;
#[cfg(test)]
#[path = "particle/multiview_gpu.rs"]
mod multiview_gpu;
mod native_views;
mod pipeline_build;
mod recovery;
pub use graph_compile::{
    EmberGraphCompileBudget, EmberGraphCompileError, EmberGraphCompilePhase,
    EmberGraphCompileRequestError, EmberGraphCompileStatus, EmberGraphCompileTicket,
    EmberGraphSource, EmberGraphStageModule,
};
pub use native_views::{
    extract_particle_views, ExtractedParticleViews, ParticleViewBudget, ParticleViewError,
};
use native_views::{
    NativeParticleViews, ParticleCoreLease, ParticleRendererOwner, RendererBlueprint,
    RetiredParticleCore,
};

/// GPU resources owned by the render world for the built-in particle bridge.
///
/// After the initial bootstrap, simulation, renderer input, culling, sorting,
/// indirect arguments, and rasterization remain GPU-authoritative. The
/// extracted runtime snapshot is not uploaded again on subsequent frames.
pub struct ParticleGpuState {
    gpu_simulation: GpuSimulationStage,
    gpu_bridge: ParticleSimulationRendererBridge,
    // Owns the shared input/count buffers only; never emits an aggregate draw.
    gpu_renderer: ParticleRendererGpu,
    gpu_emitters: Option<Arc<ParticleRendererOwner>>,
    renderer_blueprint: Option<RendererBlueprint>,
    native_views: Option<NativeParticleViews>,
    core_lease: Arc<ParticleCoreLease>,
    bridge_publication: Mutex<Option<ParticleBridgePublication>>,
    fabric: Option<ParticleFabricGpuFrame>,
    fabric_semantics: Option<RenderFabricFrame>,
    fabric_error: Option<ParticleFabricGpuBindingError>,
    gpu_initialized: bool,
    lifecycle: Option<Arc<ParticleSimulationLifecycle>>,
    applied_steps: u64,
    pending_checkpoint: Option<recovery::PendingParticleCheckpoint>,
    checkpoint_unsupported: bool,
    gameplay_readbacks: gameplay_feedback::GameplayReadbacks,
    device: Arc<dyn RenderDevice + Send + Sync>,
    device_generation: u64,
}

/// Completion channel for particle pipeline/resource specialization.
/// The worker captures the device generation so late results from a lost
/// device cannot be published into a new frame. Only one worker may be in flight.
#[derive(Default)]
pub struct ParticleGpuBuildQueue {
    pending: Option<PendingParticleGpuBuild>,
    failed: Option<(u64, u32, u64)>,
}

struct PendingParticleGpuBuild {
    generation: u64,
    capacity: u32,
    lifecycle_serial: u64,
    allowed: Arc<AtomicBool>,
    receiver: Mutex<mpsc::Receiver<Result<pipeline_build::ParticleBuildResult, ParticleGpuError>>>,
}

impl Resource for ParticleGpuBuildQueue {}

impl ParticleGpuBuildQueue {
    /// Stop allocation and explicit destruction on the invalidated device,
    /// including a worker or completed result that has not been polled yet.
    pub(crate) fn abandon_device(&self) {
        if let Some(pending) = &self.pending {
            pending.allowed.store(false, Ordering::Release);
        }
    }
}

/// Result of the prepare upload, kept in the render world so the frame system
/// can turn malformed snapshots into an observable failed frame.
#[derive(Default)]
pub struct ParticleUploadStatus {
    /// The most recent upload error, if any.
    pub error: Option<ParticleGpuError>,
    /// Why the optional particle renderer is currently disabled, if it could
    /// not be created for the active device or configuration.
    pub disabled_reason: Option<ParticleGpuError>,
    /// Invalid main-world inputs are rejected before reaching fixed GPU ABI limits.
    pub input_error: Option<ParticleSimulationInputError>,
    /// Nonfatal readback, validation, or publication failure. The durable
    /// checkpoint and complete journal tail remain available for recovery.
    pub checkpoint_error: Option<String>,
    /// Input capture is temporarily frozen until a checkpoint frees tail budget.
    pub checkpoint_backpressure: bool,
    /// Latest durable queue-position snapshot, independent of device lifetime.
    pub checkpoint_step: Option<u64>,
    /// Recovery replays a bounded number of ticks per Prepare, with rendering hidden.
    pub recovering: bool,
    pub applied_steps: u64,
    pub total_steps: u64,
}

impl Resource for ParticleUploadStatus {}

/// CPU Render Fabric projection prepared from the same extracted runtime
/// snapshot as the optional GPU particle path.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PreparedParticleFabricFrame {
    frame: ParticleFabricFrame,
    error: Option<ParticleFabricError>,
    consumption_error: Option<ParticleFabricConsumptionError>,
    geometry: ParticleFabricGeometry,
}

impl Resource for PreparedParticleFabricFrame {}

impl PreparedParticleFabricFrame {
    /// Most recent valid frame, or an empty frame carrying the rejected
    /// render frame number when validation failed.
    #[must_use]
    pub const fn frame(&self) -> &ParticleFabricFrame {
        &self.frame
    }

    /// Validation failure for the latest extracted snapshot.
    #[must_use]
    pub const fn error(&self) -> Option<ParticleFabricError> {
        self.error
    }

    /// Strict geometry projection for the latest accepted Fabric frame.
    #[must_use]
    pub const fn geometry(&self) -> &ParticleFabricGeometry {
        &self.geometry
    }

    /// Failure while consuming otherwise valid Fabric payloads into geometry.
    #[must_use]
    pub const fn consumption_error(&self) -> Option<ParticleFabricConsumptionError> {
        self.consumption_error
    }

    fn replace_from_lowering(
        &mut self,
        render_frame: u64,
        lowered: Result<ParticleFabricFrame, ParticleFabricError>,
    ) {
        match lowered {
            Ok(frame) => match ParticleFabricGeometry::from_frame(&frame) {
                Ok(geometry) => {
                    self.frame = frame;
                    self.error = None;
                    self.consumption_error = None;
                    self.geometry = geometry;
                }
                Err(error) => {
                    self.frame = ParticleFabricFrame::empty(render_frame);
                    self.error = None;
                    self.consumption_error = Some(error);
                    self.geometry = ParticleFabricGeometry::empty(render_frame);
                }
            },
            Err(error) => {
                self.frame = ParticleFabricFrame::empty(render_frame);
                self.error = Some(error);
                self.consumption_error = None;
                self.geometry = ParticleFabricGeometry::empty(render_frame);
            }
        }
    }
}

/// One strict Fabric-to-geometry draw projection. `stream` and `alive` remain
/// semantic Render Fabric descriptors; physical GPU binding happens later in
/// the Fabric runtime and is intentionally not fabricated by this bridge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleFabricDraw {
    /// Stable particle emitter identity.
    pub object_id: RenderObjectId,
    /// Stable authored consumer ordinal within this emitter.
    pub renderer_index: u32,
    /// Live-particle bounds used by a visibility/culling consumer.
    pub bounds_min: [f32; 3],
    /// Live-particle bounds used by a visibility/culling consumer.
    pub bounds_max: [f32; 3],
    /// Existing renderer geometry contract derived from the count payload.
    pub geometry: RendererGeometry,
    /// Canonical particle state stream descriptor.
    pub stream: ResourcePayload,
    /// Canonical particle alive/visibility stream descriptor.
    pub alive: ResourcePayload,
}

/// Geometry/visibility-ready projection for one Fabric frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParticleFabricGeometry {
    frame: u64,
    draws: Vec<ParticleFabricDraw>,
}

impl ParticleFabricGeometry {
    #[must_use]
    const fn empty(frame: u64) -> Self {
        Self {
            frame,
            draws: Vec::new(),
        }
    }

    /// Source frame represented by this draw list.
    #[must_use]
    pub const fn frame(&self) -> u64 {
        self.frame
    }

    /// Geometry work accepted by the strict Fabric payload consumer.
    #[must_use]
    pub fn draws(&self) -> &[ParticleFabricDraw] {
        &self.draws
    }

    fn from_frame(frame: &ParticleFabricFrame) -> Result<Self, ParticleFabricConsumptionError> {
        let mut emitters = BTreeMap::<RenderObjectId, ParticleFabricEmitter>::new();
        for item in frame.items() {
            if item.domain != RenderDomain::Particle
                || item.object_id.domain() != RenderDomain::Particle
            {
                return Err(ParticleFabricConsumptionError::InvalidParticleIdentity {
                    object: item.object_id,
                });
            }
            if item.history_key != 0 {
                return Err(ParticleFabricConsumptionError::NonCanonicalHistory {
                    object: item.object_id,
                });
            }
            if item.motion_source != MotionSource::Particle {
                return Err(ParticleFabricConsumptionError::InvalidMotionSource {
                    object: item.object_id,
                });
            }

            let emitter = emitters.entry(item.object_id).or_default();
            match item.payload {
                DomainPayload::Value(payload)
                    if payload.role == PARTICLE_EMITTER_COUNT_PAYLOAD_ROLE =>
                {
                    if emitter
                        .count
                        .replace(decode_particle_count(payload.value).ok_or(
                            ParticleFabricConsumptionError::InvalidCountPayload {
                                object: item.object_id,
                            },
                        )?)
                        .is_some()
                    {
                        return Err(ParticleFabricConsumptionError::DuplicatePayloadRole {
                            object: item.object_id,
                            role: payload.role,
                        });
                    }
                }
                DomainPayload::Value(payload)
                    if payload.role == PARTICLE_EMITTER_POSITION_BOUNDS_PAYLOAD_ROLE =>
                {
                    let bounds = decode_particle_bounds(payload.value).ok_or(
                        ParticleFabricConsumptionError::InvalidBoundsPayload {
                            object: item.object_id,
                        },
                    )?;
                    if emitter.bounds.replace(bounds).is_some() {
                        return Err(ParticleFabricConsumptionError::DuplicatePayloadRole {
                            object: item.object_id,
                            role: payload.role,
                        });
                    }
                }
                DomainPayload::Resource(payload)
                    if payload.role == PARTICLE_STREAM_RESOURCE_PAYLOAD_ROLE =>
                {
                    if payload
                        != particle_resource_payload(
                            PARTICLE_STREAM_RESOURCE,
                            PARTICLE_STREAM_RESOURCE_PAYLOAD_ROLE,
                        )
                        .expect("built-in particle stream template is declared")
                    {
                        return Err(ParticleFabricConsumptionError::InvalidResourcePayload {
                            object: item.object_id,
                            role: payload.role,
                        });
                    }
                    if emitter.stream.replace(payload).is_some() {
                        return Err(ParticleFabricConsumptionError::DuplicatePayloadRole {
                            object: item.object_id,
                            role: payload.role,
                        });
                    }
                }
                DomainPayload::Resource(payload)
                    if payload.role == PARTICLE_ALIVE_RESOURCE_PAYLOAD_ROLE =>
                {
                    if payload
                        != particle_resource_payload(
                            PARTICLE_ALIVE_RESOURCE,
                            PARTICLE_ALIVE_RESOURCE_PAYLOAD_ROLE,
                        )
                        .expect("built-in particle alive template is declared")
                    {
                        return Err(ParticleFabricConsumptionError::InvalidResourcePayload {
                            object: item.object_id,
                            role: payload.role,
                        });
                    }
                    if emitter.alive.replace(payload).is_some() {
                        return Err(ParticleFabricConsumptionError::DuplicatePayloadRole {
                            object: item.object_id,
                            role: payload.role,
                        });
                    }
                }
                DomainPayload::Value(payload) => {
                    return Err(ParticleFabricConsumptionError::UnexpectedPayloadRole {
                        object: item.object_id,
                        role: payload.role,
                    });
                }
                DomainPayload::Resource(payload) => {
                    return Err(ParticleFabricConsumptionError::UnexpectedPayloadRole {
                        object: item.object_id,
                        role: payload.role,
                    });
                }
            }
        }

        let mut draws = Vec::with_capacity(emitters.len());
        for (object_id, emitter) in emitters {
            let count =
                emitter
                    .count
                    .ok_or(ParticleFabricConsumptionError::MissingPayloadRole {
                        object: object_id,
                        role: PARTICLE_EMITTER_COUNT_PAYLOAD_ROLE,
                    })?;
            let stream =
                emitter
                    .stream
                    .ok_or(ParticleFabricConsumptionError::MissingPayloadRole {
                        object: object_id,
                        role: PARTICLE_STREAM_RESOURCE_PAYLOAD_ROLE,
                    })?;
            let alive =
                emitter
                    .alive
                    .ok_or(ParticleFabricConsumptionError::MissingPayloadRole {
                        object: object_id,
                        role: PARTICLE_ALIVE_RESOURCE_PAYLOAD_ROLE,
                    })?;
            match (count, emitter.bounds) {
                (0, None) => {}
                (0, Some(_)) => {
                    return Err(ParticleFabricConsumptionError::BoundsForEmptyEmitter {
                        object: object_id,
                    });
                }
                (alive_count, Some(bounds)) => draws.push(ParticleFabricDraw {
                    object_id,
                    renderer_index: 0,
                    bounds_min: bounds.min,
                    bounds_max: bounds.max,
                    geometry: RendererGeometry::Sprite {
                        alive_count,
                        first_instance: 0,
                    },
                    stream,
                    alive,
                }),
                (_, None) => {
                    return Err(ParticleFabricConsumptionError::MissingPayloadRole {
                        object: object_id,
                        role: PARTICLE_EMITTER_POSITION_BOUNDS_PAYLOAD_ROLE,
                    });
                }
            }
        }
        Ok(Self {
            frame: frame.frame(),
            draws,
        })
    }
}

/// Render-world owner of the physical buffers behind a particle Fabric role.
///
/// Both built-in roles resolve to buffers allocated by [`ParticleRendererGpu`].
/// The simulation-to-renderer bridge is a writer of the stream buffer, but it
/// does not own or mint that buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParticleFabricGpuResourceOwner {
    /// The persistent buffer was allocated and is destroyed by the GPU
    /// particle renderer.
    Renderer,
}

/// One canonical Fabric resource resolved to a concrete, device-owned buffer.
///
/// `semantic` retains the validated Fabric declaration. `buffer`, `size`,
/// `usage`, and `entry_state` are the exact RHI/import contract used by the
/// existing GPU simulation bridge and renderer recorder.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParticleFabricGpuResource {
    /// The canonical Render Fabric resource declaration.
    pub semantic: ResourcePayload,
    /// Physical RHI buffer allocated by the stated owner.
    pub buffer: BufferId,
    /// Physical buffer size in bytes.
    pub size: u64,
    /// Physical RHI usage flags declared to the frame graph.
    pub usage: BufferUsages,
    /// State on entry to the recorder that first imports this buffer.
    pub entry_state: BufferState,
    /// Persistent owner responsible for destroying this buffer.
    pub owner: ParticleFabricGpuResourceOwner,
}

/// One Fabric draw together with its real GPU resource bindings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleFabricGpuDraw {
    /// Stable emitter identity.
    pub object_id: RenderObjectId,
    /// Stable authored consumer ordinal, used to validate recorder ownership.
    pub renderer_index: u32,
    /// Temporal identity, namespaced by native camera incarnation.
    pub history_key: u32,
    /// Live-particle bounds consumed by visibility work.
    pub bounds_min: [f32; 3],
    /// Live-particle bounds consumed by visibility work.
    pub bounds_max: [f32; 3],
    /// Geometry contract for this emitter.
    pub geometry: RendererGeometry,
    /// Bridge-written renderer-input stream.
    pub stream: ParticleFabricGpuResource,
    /// Renderer-written compact visible-index stream.
    pub alive: ParticleFabricGpuResource,
}

/// Physical Fabric bindings published for one particle frame.
#[derive(Clone, Debug, PartialEq)]
pub struct ParticleFabricGpuFrame {
    frame: u64,
    draws: Vec<ParticleFabricGpuDraw>,
}

/// Projects physical particle bindings back into the strict per-frame Fabric
/// metadata required by the native frame assembler.  The resource descriptors
/// remain semantic here; their device ownership is carried by the associated
/// [`ParticleFabricGpuFrame`].
fn fabric_semantic_items(
    frame: &ParticleFabricGpuFrame,
) -> impl Iterator<Item = FrameRenderItem> + '_ {
    frame.draws().iter().flat_map(|draw| {
        [
            DomainPayload::Value(ValuePayload {
                value: PayloadValue::Bounds {
                    min: draw.bounds_min,
                    max: draw.bounds_max,
                },
                role: PARTICLE_EMITTER_POSITION_BOUNDS_PAYLOAD_ROLE,
            }),
            DomainPayload::Resource(draw.stream.semantic),
            DomainPayload::Resource(draw.alive.semantic),
        ]
        .into_iter()
        .map(move |payload| FrameRenderItem {
            object_id: draw.object_id,
            domain: RenderDomain::Particle,
            history_key: draw.history_key,
            motion_source: MotionSource::Particle,
            payload,
        })
    })
}

impl ParticleFabricGpuFrame {
    #[must_use]
    pub const fn frame(&self) -> u64 {
        self.frame
    }

    #[must_use]
    pub fn draws(&self) -> &[ParticleFabricGpuDraw] {
        &self.draws
    }

    /// Exports the exact native Particle publication for a Bevy SceneFabric
    /// host.  The returned packet preserves the semantic items and the
    /// renderer-owned `BufferId` generations; the host must resolve those IDs
    /// to live Bevy buffers on the same native device before importing them.
    #[must_use]
    pub fn scene_publication(&self) -> ParticleFabricScenePublication {
        let items = fabric_semantic_items(self).collect();
        let draws = self
            .draws
            .iter()
            .map(|draw| SceneParticleFabricGpuDraw {
                object_id: draw.object_id,
                renderer_index: draw.renderer_index,
                history_key: draw.history_key,
                bounds_min: draw.bounds_min,
                bounds_max: draw.bounds_max,
                geometry: draw.geometry,
                stream: scene_particle_buffer(draw.stream),
                alive: scene_particle_buffer(draw.alive),
            })
            .collect();
        ParticleFabricScenePublication::new(self.frame, items, draws)
    }

    fn from_prepared(
        prepared: &PreparedParticleFabricFrame,
        simulation: &GpuSimulationStage,
        renderer: &ParticleRendererGpu,
    ) -> Result<Option<Self>, ParticleFabricGpuBindingError> {
        if let Some(error) = prepared.error() {
            return Err(ParticleFabricGpuBindingError::FabricLowering(error));
        }
        if let Some(error) = prepared.consumption_error() {
            return Err(ParticleFabricGpuBindingError::FabricConsumption(error));
        }
        let geometry = prepared.geometry();
        if prepared.frame().frame() != geometry.frame() {
            return Err(ParticleFabricGpuBindingError::FrameMismatch {
                fabric: prepared.frame().frame(),
                geometry: geometry.frame(),
            });
        }
        if geometry.draws().is_empty() {
            return Ok(None);
        }
        if simulation.capacity() != renderer.capacity() {
            return Err(ParticleFabricGpuBindingError::CapacityMismatch {
                simulation: simulation.capacity(),
                renderer: renderer.capacity(),
            });
        }

        let capacity = u64::from(renderer.capacity());
        let stream_size = capacity
            .checked_mul(PARTICLE_RENDER_INPUT_STRIDE)
            .ok_or(ParticleFabricGpuBindingError::CapacityOverflow)?;
        let alive_size = capacity
            .checked_mul(u64::from(core::mem::size_of::<u32>() as u32))
            .ok_or(ParticleFabricGpuBindingError::CapacityOverflow)?;
        let stream_semantic = particle_resource_payload(
            PARTICLE_STREAM_RESOURCE,
            PARTICLE_STREAM_RESOURCE_PAYLOAD_ROLE,
        )
        .expect("built-in particle stream template is declared");
        let alive_semantic = particle_resource_payload(
            PARTICLE_ALIVE_RESOURCE,
            PARTICLE_ALIVE_RESOURCE_PAYLOAD_ROLE,
        )
        .expect("built-in particle alive template is declared");
        let stream = ParticleFabricGpuResource {
            semantic: stream_semantic,
            buffer: renderer.input_buffer(),
            size: stream_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            entry_state: BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
            owner: ParticleFabricGpuResourceOwner::Renderer,
        };
        let alive = ParticleFabricGpuResource {
            semantic: alive_semantic,
            buffer: renderer.visible_indices(),
            size: alive_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            entry_state: BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
            owner: ParticleFabricGpuResourceOwner::Renderer,
        };

        let mut draws = Vec::with_capacity(geometry.draws().len());
        for draw in geometry.draws() {
            if draw.stream != stream.semantic {
                return Err(ParticleFabricGpuBindingError::InvalidSemanticResource {
                    object: draw.object_id,
                    role: draw.stream.role,
                });
            }
            if draw.alive != alive.semantic {
                return Err(ParticleFabricGpuBindingError::InvalidSemanticResource {
                    object: draw.object_id,
                    role: draw.alive.role,
                });
            }
            draws.push(ParticleFabricGpuDraw {
                object_id: draw.object_id,
                renderer_index: draw.renderer_index,
                history_key: draw.renderer_index,
                bounds_min: draw.bounds_min,
                bounds_max: draw.bounds_max,
                geometry: draw.geometry,
                stream,
                alive,
            });
        }
        Ok(Some(Self {
            frame: geometry.frame(),
            draws,
        }))
    }

    fn validate_owners(
        &self,
        simulation: &GpuSimulationStage,
        renderers: &ParticleEmitterRendererGpu,
    ) -> Result<(), ParticleFabricGpuBindingError> {
        if self.draws.len() != renderers.emitters().len() {
            return Err(ParticleFabricGpuBindingError::RendererCountMismatch {
                draws: self.draws.len(),
                renderers: renderers.emitters().len(),
            });
        }
        for (draw, renderer) in self.draws.iter().zip(renderers.emitters()) {
            if simulation.capacity() != renderer.capacity() {
                return Err(ParticleFabricGpuBindingError::CapacityMismatch {
                    simulation: simulation.capacity(),
                    renderer: renderer.capacity(),
                });
            }
            if draw.stream.buffer != renderer.input_buffer()
                || draw.stream.owner != ParticleFabricGpuResourceOwner::Renderer
                || draw.alive.buffer != renderer.visible_indices()
                || draw.alive.owner != ParticleFabricGpuResourceOwner::Renderer
            {
                return Err(ParticleFabricGpuBindingError::OwnerMismatch {
                    object: draw.object_id,
                });
            }
        }
        Ok(())
    }

    fn from_emitters(
        prepared: &PreparedParticleFabricFrame,
        simulation: &GpuSimulationStage,
        input_owner: &ParticleRendererGpu,
        renderers: &ParticleEmitterRendererGpu,
    ) -> Result<Option<Self>, ParticleFabricGpuBindingError> {
        if prepared.geometry().draws().len() != renderers.emitters().len() {
            return Err(ParticleFabricGpuBindingError::RendererCountMismatch {
                draws: prepared.geometry().draws().len(),
                renderers: renderers.emitters().len(),
            });
        }
        let Some(mut frame) = Self::from_prepared(prepared, simulation, input_owner)? else {
            return Ok(None);
        };
        for (draw, renderer) in frame.draws.iter_mut().zip(renderers.emitters()) {
            draw.alive.buffer = renderer.visible_indices();
            draw.alive.size = u64::from(renderer.work_capacity()) * 4;
        }
        frame.validate_owners(simulation, renderers)?;
        Ok(Some(frame))
    }

    fn consume_recorder_outputs(
        &self,
        bridge_stream: BufferHandle,
        renderer_stream: BufferHandle,
        renderer_alive: BufferHandle,
    ) -> Result<(), ParticleFabricGpuBindingError> {
        if bridge_stream != renderer_stream {
            return Err(ParticleFabricGpuBindingError::RecorderStreamDiscontinuity);
        }
        if self.draws.is_empty() {
            return Err(ParticleFabricGpuBindingError::MissingRecorderDraw);
        }
        if renderer_alive.version() == 0 {
            return Err(ParticleFabricGpuBindingError::RecorderAliveNotWritten);
        }
        Ok(())
    }
}

fn scene_particle_buffer(resource: ParticleFabricGpuResource) -> SceneParticleFabricGpuBuffer {
    SceneParticleFabricGpuBuffer {
        semantic: resource.semantic,
        buffer: resource.buffer,
        version: resource.buffer.generation(),
        size: resource.size,
        usage: resource.usage,
        entry_state: resource.entry_state,
    }
}

/// Failure while resolving semantic Fabric particle roles to device-owned RHI
/// buffers or consuming those bindings in the frame recorder.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParticleFabricGpuBindingError {
    /// Semantic Fabric lowering rejected the extracted runtime snapshot.
    FabricLowering(ParticleFabricError),
    /// Fabric items did not meet the strict geometry consumer contract.
    FabricConsumption(ParticleFabricConsumptionError),
    /// GPU route metadata failed shared frame identity or payload validation.
    InvalidSemanticFrame(RenderFabricFrameError),
    /// The semantic and geometry projections came from different frames.
    FrameMismatch { fabric: u64, geometry: u64 },
    /// The extracted runtime snapshot and semantic Fabric projection diverged.
    SnapshotFrameMismatch { snapshot: u64, fabric: u64 },
    /// The simulation pool and renderer buffers cannot describe one ABI.
    CapacityMismatch { simulation: u32, renderer: u32 },
    /// Every registered emitter must have its own physical Fabric consumer.
    RendererCountMismatch { draws: usize, renderers: usize },
    /// Buffer-footprint arithmetic overflowed before an import was recorded.
    CapacityOverflow,
    /// A draw no longer carries one of the canonical semantic resources.
    InvalidSemanticResource {
        object: RenderObjectId,
        role: PayloadRole,
    },
    /// A bound buffer does not belong to the live particle renderer.
    OwnerMismatch { object: RenderObjectId },
    /// The bridge output was not handed to the renderer as the same SSA stream.
    RecorderStreamDiscontinuity,
    /// A physical Fabric frame unexpectedly contained no recorder draw.
    MissingRecorderDraw,
    /// The renderer did not produce the compact alive-index stream this frame.
    RecorderAliveNotWritten,
}

impl fmt::Display for ParticleFabricGpuBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "particle Fabric GPU binding failed: {self:?}")
    }
}

impl std::error::Error for ParticleFabricGpuBindingError {}

/// Failure while admitting the particle Fabric recorder chain to a frame graph.
#[derive(Debug)]
pub enum ParticleGpuGraphError {
    /// The first graph producing this frame's bridge has not been submitted.
    PendingBridgeSubmission,
    /// A lost device cannot publish or reuse resident bridge outputs.
    DeviceUnavailable,
    /// Every native viewport must fit both attachments before graph mutation.
    InvalidViewport { camera: prism_ecs::prelude::Entity },
    /// The semantic-to-physical Fabric binding no longer matches the live GPU owners.
    Fabric(ParticleFabricGpuBindingError),
    /// The frame graph rejected one of the existing GPU recorder passes.
    Graph(prism_render_graph::CompileError),
}

impl fmt::Display for ParticleGpuGraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "particle GPU graph admission failed: {self:?}")
    }
}

impl std::error::Error for ParticleGpuGraphError {}

impl From<ParticleFabricGpuBindingError> for ParticleGpuGraphError {
    fn from(error: ParticleFabricGpuBindingError) -> Self {
        Self::Fabric(error)
    }
}

impl From<prism_render_graph::CompileError> for ParticleGpuGraphError {
    fn from(error: prism_render_graph::CompileError) -> Self {
        Self::Graph(error)
    }
}

/// Camera-local handles for shared, GPU-resident bridge outputs.
#[derive(Clone)]
struct ParticleBridgeOutputs {
    renderer_input: BufferHandle,
    renderer_input_count: BufferHandle,
    emitter_route_table: BufferHandle,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct ParticleBridgeKey {
    frame: u64,
    device_generation: u64,
    lifecycle: usize,
}

struct ParticleBridgePublication {
    key: ParticleBridgeKey,
    producer: Weak<prism_render_graph::GraphSubmissionToken>,
    submitted: Arc<AtomicBool>,
    complete: bool,
}

/// Submission witness for one frame's shared bridge output.
///
/// Keep this across queue submission and call [`Self::confirm_submitted`] after
/// it succeeds, before presenting. Recording alone never publishes the cache.
/// The producer graph also retains a witness: explicit allocation release saves
/// the submitted state even if a fast queue retires before the next output.
/// Cancellation drops the token naturally; the state cache holds only a Weak.
pub struct ParticleGraphOutputLease {
    token: Weak<prism_render_graph::GraphSubmissionToken>,
    submitted: Arc<AtomicBool>,
}

impl ParticleGraphOutputLease {
    /// Returns false until the graph's allocations were marked after actual
    /// submission. This acknowledges queue ordering, not GPU completion.
    pub fn confirm_submitted(&self) -> bool {
        // Upgrade before reading the durable flag: retirement may save the
        // submitted state and expire the token between these two operations.
        let token = self.token.upgrade();
        if !self.submitted.load(Ordering::Acquire)
            && !token.as_ref().is_some_and(|token| token.is_submitted())
        {
            return false;
        }
        self.submitted.store(true, Ordering::Release);
        true
    }
}

// Only the graph holds a strong witness. A caller keeping a lease must not
// accidentally keep a cancelled graph's pending mutation gate alive.
struct ParticleBridgeSubmissionWitness {
    token: Arc<prism_render_graph::GraphSubmissionToken>,
    submitted: Arc<AtomicBool>,
}

impl Drop for ParticleBridgeSubmissionWitness {
    fn drop(&mut self) {
        if self.token.is_submitted() {
            self.submitted.store(true, Ordering::Release);
        }
    }
}

/// One import/SSA chain per shared simulation per graph, irrespective of fan-out.
/// The core is pinned by the graph, so its address cannot be reused for another
/// simulation while this cache is live.
#[derive(Default)]
struct ParticleGraphBridges(BTreeMap<usize, ParticleBridgeOutputs>);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Failure while turning particle Fabric items into the geometry contract.
pub enum ParticleFabricConsumptionError {
    InvalidParticleIdentity {
        object: RenderObjectId,
    },
    NonCanonicalHistory {
        object: RenderObjectId,
    },
    InvalidMotionSource {
        object: RenderObjectId,
    },
    UnexpectedPayloadRole {
        object: RenderObjectId,
        role: PayloadRole,
    },
    DuplicatePayloadRole {
        object: RenderObjectId,
        role: PayloadRole,
    },
    MissingPayloadRole {
        object: RenderObjectId,
        role: PayloadRole,
    },
    InvalidCountPayload {
        object: RenderObjectId,
    },
    InvalidBoundsPayload {
        object: RenderObjectId,
    },
    InvalidResourcePayload {
        object: RenderObjectId,
        role: PayloadRole,
    },
    BoundsForEmptyEmitter {
        object: RenderObjectId,
    },
}

impl fmt::Display for ParticleFabricConsumptionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "particle Fabric payload consumption failed: {self:?}"
        )
    }
}

impl std::error::Error for ParticleFabricConsumptionError {}

#[derive(Default)]
struct ParticleFabricEmitter {
    count: Option<u32>,
    bounds: Option<ParticleFabricBounds>,
    stream: Option<ResourcePayload>,
    alive: Option<ResourcePayload>,
}

#[derive(Clone, Copy)]
struct ParticleFabricBounds {
    min: [f32; 3],
    max: [f32; 3],
}

fn decode_particle_count(value: PayloadValue) -> Option<u32> {
    let PayloadValue::Vector([low, high, 0.0, 0.0]) = value else {
        return None;
    };
    let low = decode_u16_lane(low)?;
    let high = decode_u16_lane(high)?;
    Some(low | (high << 16))
}

fn decode_u16_lane(value: f32) -> Option<u32> {
    (value.is_finite() && (0.0..=65535.0).contains(&value) && value.fract() == 0.0)
        .then_some(value as u32)
}

fn decode_particle_bounds(value: PayloadValue) -> Option<ParticleFabricBounds> {
    let PayloadValue::Bounds { min, max } = value else {
        return None;
    };
    min.into_iter()
        .zip(max)
        .all(|(min, max)| min.is_finite() && max.is_finite() && min <= max)
        .then_some(ParticleFabricBounds { min, max })
}

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
    compiler: graph_compile::GraphCompiler,
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
    owner_leases: BTreeMap<CompiledGraphGpuResourceSlot, ember_build::ExternalOwnerLease>,
    graph_imports: BTreeMap<
        CompiledGraphGpuResourceSlot,
        (CompiledGraphGpuRevision, CompiledGraphGpuGraphImport),
    >,
}

impl Resource for EmberExternalGpuResources {}

impl EmberExternalGpuResources {
    /// Creates an empty external resource registry.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            registry: CompiledGraphGpuResourceRegistry::new(),
            owner_leases: BTreeMap::new(),
            graph_imports: BTreeMap::new(),
        }
    }

    /// Pins the actual owner of the current exact binding for asynchronous
    /// materialization and all submissions using it. The owner must keep the
    /// buffer/texture/view alive until its last Arc is dropped; an unrelated
    /// marker Arc is not sufficient. Owners must handle device loss themselves.
    /// `device_generation` must be the owning runtime's current generation;
    /// recovery requires republishing ids and leases from the replacement device.
    /// Republishing a slot invalidates its lease even if the raw ids are reused.
    /// Unleased external bindings are never sent to the GPU build worker.
    pub fn set_owner_lease(
        &mut self,
        slot: CompiledGraphGpuResourceSlot,
        revision: CompiledGraphGpuRevision,
        device_generation: u64,
        owner: Arc<dyn core::any::Any + Send + Sync>,
    ) -> bool {
        let Some(binding) = self.registry.get(slot).copied() else {
            return false;
        };
        if binding.revision != revision {
            return false;
        }
        self.owner_leases.insert(
            slot,
            ember_build::ExternalOwnerLease {
                binding,
                device_generation,
                _owner: owner,
            },
        );
        true
    }

    /// Stops publication of a lease. Extracted, building, resident and retiring
    /// versions retain their own references until they are safe to release.
    pub fn remove_owner_lease(&mut self, slot: CompiledGraphGpuResourceSlot) {
        self.owner_leases.remove(&slot);
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

    /// Publishes the symbolic frame-graph import contract for one external
    /// resource slot and exact resource revision. A later resource revision
    /// invalidates this descriptor until its owner publishes replacement
    /// metadata. The concrete GPU resource remains owned by the registry; this
    /// descriptor supplies metadata a frame graph cannot infer from a raw RHI
    /// handle (format, extent, mip/layer counts and synchronization states).
    pub fn set_graph_import(
        &mut self,
        slot: CompiledGraphGpuResourceSlot,
        revision: CompiledGraphGpuRevision,
        import: CompiledGraphGpuGraphImport,
    ) {
        self.graph_imports.insert(slot, (revision, import));
    }

    /// Removes the symbolic frame-graph import contract for one slot.
    pub fn remove_graph_import(
        &mut self,
        slot: CompiledGraphGpuResourceSlot,
    ) -> Option<CompiledGraphGpuGraphImport> {
        self.graph_imports.remove(&slot).map(|(_, import)| import)
    }

    /// Returns the published frame-graph import contract for one slot.
    #[must_use]
    pub fn graph_import(
        &self,
        slot: CompiledGraphGpuResourceSlot,
    ) -> Option<&CompiledGraphGpuGraphImport> {
        let (revision, import) = self.graph_imports.get(&slot)?;
        (self.registry.get(slot)?.revision == *revision).then_some(import)
    }
}

impl Resource for EmberCompiledGraphAssets {}

impl EmberCompiledGraphAssets {
    /// Creates an empty compiled-graph registry.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            registry: EmberCompiledGraphRegistry::new(),
            compiler: graph_compile::GraphCompiler::new(EmberGraphCompileBudget::DEFAULT),
        }
    }

    /// Returns the underlying publication registry.
    #[must_use]
    pub const fn registry(&self) -> &EmberCompiledGraphRegistry {
        &self.registry
    }

    /// Returns the mutable registry, cancelling all authoring requests first.
    /// This external-publication boundary prevents late workers from overwriting
    /// manual replacement or removal, even if a graph handle is reused.
    #[must_use]
    pub fn registry_mut(&mut self) -> &mut EmberCompiledGraphRegistry {
        self.compiler.cancel_all();
        &mut self.registry
    }

    /// Creates a compiler with explicit nonzero admission limits.
    pub fn with_compile_budget(
        budget: EmberGraphCompileBudget,
    ) -> Result<Self, EmberGraphCompileRequestError> {
        Ok(Self {
            registry: EmberCompiledGraphRegistry::new(),
            compiler: graph_compile::GraphCompiler::with_budget(budget)?,
        })
    }

    /// Queues the latest immutable source without running plugins on this thread.
    pub fn request_compile(
        &mut self,
        graph: GraphAssetRef,
        source_revision: u64,
        source: Arc<EmberGraphSource>,
    ) -> Result<EmberGraphCompileTicket, EmberGraphCompileRequestError> {
        self.compiler
            .request(&self.registry, graph, source_revision, source)
    }

    /// Polls without waiting; returns the number of atomic publications (0 or 1).
    /// RenderApp extraction invokes this even without cameras or live instances.
    pub fn poll_compilations(&mut self) -> usize {
        self.compiler.poll(&mut self.registry)
    }

    /// Latest desired source and its observable state.
    #[must_use]
    pub fn compile_status(&self, graph: GraphAssetRef) -> Option<&EmberGraphCompileStatus> {
        self.compiler.status(graph)
    }

    /// Explicitly retries a current failed ticket, never an invalidated publication.
    pub fn retry_compile(&mut self, ticket: EmberGraphCompileTicket) -> bool {
        self.compiler.retry(ticket)
    }

    /// Cancels pending publication and releases a queue slot, preserving the live graph.
    pub fn cancel_compile(&mut self, graph: GraphAssetRef) -> bool {
        self.compiler.cancel(graph)
    }

    /// Removes a graph and cancels its pending authoring publication.
    pub fn remove_graph(
        &mut self,
        graph: GraphAssetRef,
    ) -> Result<
        Arc<CompiledGraph>,
        prism_render_architecture::particle::asset_registry::EmberCompiledGraphRegistryError,
    > {
        self.compiler.cancel(graph);
        self.registry.remove(graph)
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

impl ExtractedEmberRenderAssets {
    /// Looks up the exact revisioned input consumed by render preparation.
    #[must_use]
    pub fn get(&self, handle: EmberAssetHandle) -> Option<&EmberRenderAsset> {
        self.entries.get(&handle)
    }

    /// Iterates extracted instances in stable asset-handle order.
    pub fn iter(&self) -> impl Iterator<Item = (EmberAssetHandle, &EmberRenderAsset)> {
        self.entries.iter().map(|(handle, entry)| (*handle, entry))
    }
}

/// Render-world snapshot of [`EmberExternalGpuResources`].
#[derive(Clone, Default)]
pub struct ExtractedEmberExternalGpuResources {
    registry: CompiledGraphGpuResourceRegistry,
    owner_leases: BTreeMap<CompiledGraphGpuResourceSlot, ember_build::ExternalOwnerLease>,
    graph_imports: BTreeMap<CompiledGraphGpuResourceSlot, CompiledGraphGpuGraphImport>,
}

impl Resource for ExtractedEmberExternalGpuResources {}

impl ExtractedEmberExternalGpuResources {
    /// Returns the exact revisioned snapshot consumed by prepare.
    #[must_use]
    pub const fn registry(&self) -> &CompiledGraphGpuResourceRegistry {
        &self.registry
    }

    /// Returns the exact extracted frame-graph import metadata.
    #[must_use]
    pub fn graph_import(
        &self,
        slot: CompiledGraphGpuResourceSlot,
    ) -> Option<&CompiledGraphGpuGraphImport> {
        self.graph_imports.get(&slot)
    }

    /// Returns all extracted frame-graph import metadata in stable slot order.
    #[must_use]
    pub const fn graph_imports(
        &self,
    ) -> &BTreeMap<CompiledGraphGpuResourceSlot, CompiledGraphGpuGraphImport> {
        &self.graph_imports
    }
}

/// Incrementally extracts changed Ember graph inputs without copying compiled
/// shader source or unchanged runtime payloads.
pub fn extract_ember_render_assets(main: &mut World, render: &mut World) {
    if let Some(mut graphs) = main.get_resource_mut::<EmberCompiledGraphAssets>() {
        graphs.poll_compilations();
    }
    render.init_resource::<ExtractedEmberRenderAssets>();
    let Some(source) = main.get_resource::<EmberRenderAssets>() else {
        render
            .resource_mut::<ExtractedEmberRenderAssets>()
            .entries
            .clear();
        return;
    };
    let graphs = main.get_resource::<EmberCompiledGraphAssets>();
    let extracted = render.resource_mut::<ExtractedEmberRenderAssets>();
    extracted
        .entries
        .retain(|handle, _| source.entries.contains_key(handle));
    for (handle, entry) in source.iter() {
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
                    compiled: graph_entry.map(|entry| Arc::clone(entry.shared_graph())),
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
    let (snapshot, graph_imports, owner_leases) = main
        .get_resource::<EmberExternalGpuResources>()
        .map(|resources| {
            let graph_imports = resources
                .graph_imports
                .iter()
                .filter_map(|(slot, (revision, import))| {
                    (resources.registry.get(*slot)?.revision == *revision)
                        .then_some((*slot, import.clone()))
                })
                .collect();
            let owner_leases = resources
                .owner_leases
                .iter()
                .filter_map(|(slot, lease)| {
                    (resources.registry.get(*slot) == Some(&lease.binding))
                        .then_some((*slot, lease.clone()))
                })
                .collect();
            (resources.registry.clone(), graph_imports, owner_leases)
        })
        .unwrap_or_default();
    let extracted = render.resource_mut::<ExtractedEmberExternalGpuResources>();
    extracted.registry = snapshot;
    extracted.graph_imports = graph_imports;
    extracted.owner_leases = owner_leases;
}

/// A compiled Ember graph and its device-owned resources in the render world.
struct EmberGpuAsset {
    graph: GraphAssetRef,
    graph_revision: EmberAssetRevision,
    compiled: CompiledGraph,
    state: ember_build::OwnedGraphState,
    stage_counts: Vec<u32>,
    active: bool,
    // Async residents import the metadata of their own exact external revision,
    // not a newer publication while an old graph is being preserved.
    external_imports: Option<BTreeMap<CompiledGraphGpuResourceSlot, CompiledGraphGpuGraphImport>>,
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
    builds: ember_build::EmberBuildScheduler,
}

impl Drop for EmberGpuAssetRegistry {
    fn drop(&mut self) {
        // Recovery removes this resource without an idle guarantee. Never call
        // destroy through that lost device; its backend tears down the ids.
        // Healthy shutdown must use destroy() after queue idle instead.
        self.builds.abandon_device();
    }
}

impl Resource for EmberGpuAssetRegistry {}

fn import_graph_buffer(
    graph: &mut RenderGraph,
    imported: &mut BTreeMap<BufferId, BufferHandle>,
    id: BufferId,
    size: u64,
    usage: BufferUsages,
    state: BufferState,
) -> BufferHandle {
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
}

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
    /// Initial or replacement variants awaiting bounded asynchronous admission
    /// or materialization. Pending initial graphs do not reject the frame.
    pub pending_graphs: u32,
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
        self.entries.get(&handle).map(|entry| &*entry.state)
    }

    /// Synchronously materializes and publishes one compiled graph.
    ///
    /// GPU construction completes before an existing entry is replaced. A
    /// failed construction leaves the currently live state untouched.
    /// This explicit low-level API requires the caller to establish queue idle
    /// before replacing an entry and to pin external ids. Normal RenderApp
    /// preparation uses the bounded async scheduler and completion retirement.
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
        let mut stage_counts = Vec::new();
        if !compiled.dispatches.is_empty() {
            let slots = compiled
                .dispatches
                .iter()
                .map(|dispatch| dispatch.stage_index + 1)
                .max()
                .unwrap_or(0);
            stage_counts.resize(slots, 0);
            for dispatch in &compiled.dispatches {
                stage_counts[dispatch.stage_index] = dispatch.default_count;
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
                state: ember_build::OwnedGraphState::unmanaged(state),
                stage_counts,
                active: true,
                external_imports: None,
            },
        ) {
            previous.state.destroy(device);
        }
        Ok(())
    }

    /// Removes one graph and releases all resources it owns. The caller must
    /// first establish queue idle. RenderApp removal uses deferred retirement.
    pub fn remove(&mut self, device: &impl RenderDevice, handle: EmberAssetHandle) -> bool {
        let Some(entry) = self.entries.remove(&handle) else {
            return false;
        };
        entry.state.destroy(device);
        true
    }

    /// Releases every resident/pending graph after the caller confirms queue
    /// idle, while the device is still usable. May join one materialization job.
    pub fn destroy(mut self, device: &impl RenderDevice) {
        // Caller has confirmed queue idle (the plugin shutdown path does so).
        // Join only here, never in a frame's PrepareResources schedule.
        self.builds.shutdown();
        for entry in core::mem::take(&mut self.entries).into_values() {
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
    /// This low-level synchronous API requires queue idle and caller-pinned
    /// external ids. RenderApp instead schedules an exact-revision new variant.
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
    /// Like [`Self::remove`], this synchronous API requires queue idle.
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
            .filter(|entry| entry.active)
            .map(|entry| entry.compiled.dispatches.len())
            .sum()
    }

    /// Appends every resident graph's compute stages to a frame graph.
    pub fn append_to_graph(
        &self,
        graph: &mut RenderGraph,
        frame_index: u64,
        external_imports: &BTreeMap<CompiledGraphGpuResourceSlot, CompiledGraphGpuGraphImport>,
        external_texture_handles: &BTreeMap<TextureViewId, TextureHandle>,
    ) -> Result<(), CompiledGraphGraphError> {
        for entry in self.entries.values() {
            if !entry.active {
                continue;
            }
            let external_imports = entry.external_imports.as_ref().unwrap_or(external_imports);
            let mut imported = BTreeMap::<BufferId, BufferHandle>::new();
            let mut bindings = Vec::new();

            let attribute_handle = entry.state.attribute_pool_buffer().map(|buffer| {
                import_graph_buffer(
                    graph,
                    &mut imported,
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
                } else if entry
                    .state
                    .external_binding(CompiledGraphGpuResourceSlot::new(
                        binding.group,
                        binding.binding,
                    ))
                    .is_some()
                {
                    continue;
                } else {
                    return Err(CompiledGraphGraphError::UnexpectedBinding {
                        group: binding.group,
                        binding: binding.binding,
                    });
                };
                let handle = import_graph_buffer(graph, &mut imported, buffer, size, usage, state);
                bindings.push(CompiledGraphGpuGraphBinding::Buffer {
                    group: binding.group,
                    binding: binding.binding,
                    revision: CompiledGraphGpuRevision::from_value(0),
                    handle,
                });
            }
            for external in entry.state.external_bindings() {
                let (group, binding) = (external.slot.group, external.slot.binding);
                if let CompiledGraphGpuExternalResource::Texture { view, .. }
                | CompiledGraphGpuExternalResource::TextureView { view } = external.resource
                    && let Some(handle) = external_texture_handles.get(&view)
                {
                    let Some(graph_binding) = entry
                        .state
                        .external_graph_texture_binding(external.slot, *handle)
                    else {
                        return Err(CompiledGraphGraphError::ExpectedTexture {
                            group,
                            binding,
                            name: String::from("external Ember resource"),
                        });
                    };
                    bindings.push(graph_binding);
                    continue;
                }
                if let Some(import) = external_imports.get(&external.slot) {
                    let graph_binding = entry
                        .state
                        .import_external_to_graph(
                            graph,
                            external.slot,
                            format!("ember-external-{}-{}", group, binding),
                            import.clone(),
                        )
                        .map_err(CompiledGraphGraphError::ExternalImport)?;
                    bindings.push(graph_binding);
                    continue;
                }
                let handle = match external.resource {
                    CompiledGraphGpuExternalResource::StorageBuffer { buffer, size }
                    | CompiledGraphGpuExternalResource::CurveBuffer { buffer, size } => {
                        import_graph_buffer(
                            graph,
                            &mut imported,
                            buffer,
                            size.unwrap_or(1).max(1),
                            BufferUsages::STORAGE,
                            BufferState::initial(),
                        )
                    }
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

fn gpu_simulation_params(step: &ParticleSimulationStep) -> GpuSimulationParams {
    let p = step.params;
    GpuSimulationParams {
        dt: step.dt,
        restitution: p.restitution,
        friction: p.friction,
        spawn_count: p.spawn_count,
        spawn_position: p.spawn_position,
        spawn_velocity: p.spawn_velocity,
        spawn_lifetime: p.spawn_lifetime,
        condition_x: p.condition_x,
        condition_enabled: p.condition_enabled,
        integrator: p.integrator,
    }
}

/// GPU live counts/ranges belong to the bridge. CPU metadata is only a stable
/// capacity envelope, never a later CPU runtime's live-count/position projection.
fn simulation_fabric_frame(
    lifecycle: &ParticleSimulationLifecycle,
    frame: u64,
) -> PreparedParticleFabricFrame {
    let stream = particle_resource_payload(
        PARTICLE_STREAM_RESOURCE,
        PARTICLE_STREAM_RESOURCE_PAYLOAD_ROLE,
    )
    .expect("built-in particle stream template is declared");
    let alive = particle_resource_payload(
        PARTICLE_ALIVE_RESOURCE,
        PARTICLE_ALIVE_RESOURCE_PAYLOAD_ROLE,
    )
    .expect("built-in particle alive template is declared");
    let mut draws = Vec::new();
    let mut push = |entity: prism_ecs::prelude::Entity,
                    first_instance,
                    alive_count,
                    renderers: &ParticleRendererComponent| {
        if alive_count == 0 {
            return;
        }
        let bits = entity.to_bits();
        for (renderer_index, renderer) in renderers.renderers.iter().enumerate() {
            draws.push(ParticleFabricDraw {
                object_id: RenderObjectId::Particle(DomainObjectHandle::new(
                    bits as u32,
                    (bits >> 32) as u32,
                )),
                renderer_index: renderer_index as u32,
                // Without GPU bounds readback, no finite CPU snapshot can safely cull
                // an evolving pool. Keep the envelope conservative; renderer culling
                // uses the actual GPU-produced stream.
                bounds_min: [-f32::MAX; 3],
                bounds_max: [f32::MAX; 3],
                geometry: renderer.geometry(alive_count, first_instance),
                stream,
                alive,
            });
        }
    };
    for (emitter, renderer) in lifecycle
        .bootstrap
        .emitters()
        .iter()
        .zip(&lifecycle.renderers)
    {
        push(
            emitter.emitter_entity,
            emitter.first_particle,
            emitter.particle_count,
            renderer,
        );
    }
    let used = lifecycle.bootstrap.len() as u32;
    push(
        lifecycle.spawn_entity,
        used,
        lifecycle.capacity.saturating_sub(used),
        &lifecycle.spawn_renderer,
    );
    PreparedParticleFabricFrame {
        frame: ParticleFabricFrame::empty(frame),
        error: None,
        consumption_error: None,
        geometry: ParticleFabricGeometry { frame, draws },
    }
}

fn import_simulation_outputs(
    graph: &mut RenderGraph,
    simulation: &GpuSimulationStage,
) -> GpuSimulationGraphOutputs {
    use prism_render_gpudriven::gpu_simulation::{
        CONTROL_WORDS, EVENT_CHANNEL_COUNT, GPU_EVENT_STRIDE, GPU_PARTICLE_STRIDE,
    };
    let mut imported = BTreeMap::new();
    let mut import = |id, size, usage| {
        import_graph_buffer(
            graph,
            &mut imported,
            id,
            size,
            usage,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        )
    };
    GpuSimulationGraphOutputs {
        particles: import(
            simulation.particles(),
            u64::from(simulation.capacity()) * GPU_PARTICLE_STRIDE,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        ),
        active_indices: import(
            simulation.active_indices(),
            u64::from(simulation.capacity()) * 4,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        ),
        compact_events: import(
            simulation.compact_events(),
            u64::from(simulation.event_capacity())
                * u64::from(EVENT_CHANNEL_COUNT)
                * GPU_EVENT_STRIDE,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        ),
        control: import(
            simulation.control(),
            u64::from(CONTROL_WORDS) * 4,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        ),
    }
}

impl ParticleGpuState {
    /// Marks GPU-owned particle resources as unusable after device loss.
    ///
    /// Recovery removes this state without waiting for the old device. The
    /// shared lease makes delayed view renderers skip explicit destruction,
    /// because the backend has already invalidated their native handles.
    pub fn abandon(&self) {
        self.core_lease.abandon();
        self.gameplay_readbacks.abandon();
    }

    /// Creates the GPU stage for the current render device.
    pub fn new(
        device: &impl RenderDevice,
        device_handle: Arc<dyn RenderDevice + Send + Sync>,
        device_generation: u64,
        capacity: u32,
    ) -> Result<Self, ParticleGpuError> {
        if !device
            .capabilities()
            .features
            .contains(prism_render_driver::Features::INDIRECT_DRAW)
        {
            return Err(ParticleGpuError::MissingIndirectDraw);
        }
        let gpu_route = ParticleRendererRoute::new(
            RendererKind::Sprite,
            SortStrategy::ViewDepthBitonic,
            RendererGeometry::Sprite {
                alive_count: capacity,
                first_instance: 0,
            },
        )
        .map_err(|error| ParticleGpuError::RendererMatrix(error.into()))?;
        let gpu_simulation = GpuSimulationStage::new(device, capacity, capacity.max(1))
            .map_err(ParticleGpuError::Simulation)?;
        let gpu_renderer =
            ParticleRendererGpu::new(device, capacity, gpu_route).map_err(|error| {
                gpu_simulation.destroy(device);
                ParticleGpuError::RendererMatrix(error)
            })?;
        let gpu_bridge = ParticleSimulationRendererBridge::new(
            device,
            &gpu_simulation,
            &gpu_renderer,
            &[ParticleEmitterRenderRoute {
                emitter_id: 0,
                active_start: 0,
                particle_count: capacity,
                renderer: gpu_route,
            }],
        )
        .map_err(|error| {
            gpu_renderer.destroy(device);
            gpu_simulation.destroy(device);
            ParticleGpuError::SimulationBridge(error)
        })?;
        Ok(Self {
            gpu_simulation,
            gpu_bridge,
            gpu_renderer,
            gpu_emitters: None,
            renderer_blueprint: None,
            native_views: None,
            core_lease: Arc::new(ParticleCoreLease::new(device_handle.clone())),
            bridge_publication: Mutex::new(None),
            fabric: None,
            fabric_semantics: None,
            fabric_error: None,
            gpu_initialized: false,
            lifecycle: None,
            applied_steps: 0,
            pending_checkpoint: None,
            checkpoint_unsupported: false,
            gameplay_readbacks: gameplay_feedback::GameplayReadbacks::default(),
            device: device_handle,
            device_generation,
        })
    }

    /// Seeds once per lifecycle/device and consumes each fixed input exactly once.
    /// Device reconstruction replays the bounded journal, never the latest CPU pool.
    /// `render_frame` identifies publication, independently of the fixed-step clock.
    pub fn prepare_gpu_snapshot<Q>(
        &mut self,
        queue: &Q,
        config: &RenderAppConfig,
        inputs: &mut ParticleGpuSimulationInputs,
        render_frame: u64,
    ) -> Result<(), ParticleGpuError>
    where
        Q: RenderQueue + FrameSync,
    {
        self.fabric = None;
        // Retain the validated container's allocation across healthy frames, but
        // never publish an old frame on any early return or preparation failure.
        let mut semantics = self.fabric_semantics.take().unwrap_or_default();
        self.fabric_error = None;
        if let Some(views) = self.native_views.as_mut() {
            views.revoke();
        }
        self.poll_checkpoint(inputs);
        self.gameplay_readbacks.poll(
            inputs.lifecycle.as_ref(),
            self.device_generation,
            render_frame,
        );
        // Arc pins guarantee lifetime, not the contents seen by an older graph.
        // Queue ordering permits updates after submission, without waiting idle.
        if self.core_lease.has_unsubmitted_graph() {
            return Err(ParticleGpuError::PendingGraphSubmission);
        }
        // Every admitted Prepare revokes reuse, including same-frame retries,
        // disabled inputs and lifecycle resets. Never revoke a live producer.
        *self
            .bridge_publication
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
        if inputs.overflow.is_some() || inputs.error.is_some() || !inputs.enabled {
            return Ok(());
        }
        let Some(lifecycle) = inputs.lifecycle.clone() else {
            return Ok(());
        };
        if lifecycle.capacity != self.gpu_simulation.capacity() {
            return Err(ParticleGpuError::InvalidSnapshotRange);
        }
        let lifecycle_changed = self
            .lifecycle
            .as_ref()
            .is_none_or(|current| !Arc::ptr_eq(current, &lifecycle));
        let renderer_authoring_changed = lifecycle_changed
            && self.lifecycle.as_ref().is_none_or(|current| {
                current.renderers != lifecycle.renderers
                    || current.spawn_renderer != lifecycle.spawn_renderer
            });
        if lifecycle_changed {
            self.gpu_initialized = false;
            self.applied_steps = 0;
            self.pending_checkpoint = None;
        }
        if (self.gpu_initialized && self.applied_steps < inputs.history_start_step)
            || (!self.gpu_initialized
                && inputs.history_start_step != inputs.checkpoint.as_ref().map_or(0, |c| c.step()))
        {
            inputs.error = Some(ParticleSimulationInputError::RecoveryUnavailable);
            inputs.recovering = false;
            return Err(ParticleGpuError::InvalidSnapshotRange);
        }
        let snapshot = &lifecycle.bootstrap;
        if !self.gpu_initialized {
            if lifecycle.renderers.len() != snapshot.emitters().len() {
                return Err(ParticleGpuError::InvalidSnapshotRange);
            }
            for (entity, renderer) in snapshot
                .emitters()
                .iter()
                .zip(&lifecycle.renderers)
                .map(|(emitter, renderer)| (emitter.emitter_entity, renderer))
                .chain(core::iter::once((
                    lifecycle.spawn_entity,
                    &lifecycle.spawn_renderer,
                )))
            {
                if let Err(error) = renderer.validate() {
                    inputs.error = Some(ParticleSimulationInputError::InvalidRendererAuthoring {
                        entity,
                        error,
                    });
                    return Err(ParticleGpuError::InvalidSnapshotRange);
                }
            }
            let capacity = self.gpu_simulation.capacity() as usize;
            if snapshot.particles().len() > capacity {
                return Err(ParticleGpuError::PoolExceedsCapacity {
                    requested: snapshot.particles().len(),
                    capacity: self.gpu_simulation.capacity(),
                });
            }
            let mut expected_particle = 0_u32;
            let mut seen_emitters = BTreeSet::new();
            for emitter in snapshot.emitters() {
                let end = emitter
                    .first_particle
                    .checked_add(emitter.particle_count)
                    .ok_or(ParticleGpuError::InvalidSnapshotRange)?;
                if emitter.first_particle != expected_particle
                    || end as usize > snapshot.particles().len()
                    || !seen_emitters.insert(emitter.emitter_entity.to_bits())
                {
                    return Err(ParticleGpuError::InvalidSnapshotRange);
                }
                expected_particle = end;
            }
            if expected_particle as usize != snapshot.particles().len() {
                return Err(ParticleGpuError::InvalidSnapshotRange);
            }
            let mut authors = Vec::new();
            let mut routes = snapshot
                .emitters()
                .iter()
                .enumerate()
                .filter(|(_, emitter)| emitter.particle_count != 0)
                .map(|(route_index, emitter)| {
                    let container = &lifecycle.renderers[route_index];
                    authors.push(container);
                    // A bridge route describes shared simulation identity and
                    // range; its first renderer is metadata, not a consumer cap.
                    let author = &container.renderers[0];
                    Ok(ParticleEmitterRenderRoute {
                        emitter_id: u32::try_from(route_index)
                            .map_err(|_| ParticleGpuError::InvalidSnapshotRange)?,
                        active_start: emitter.first_particle,
                        particle_count: emitter.particle_count,
                        renderer: ParticleRendererRoute::new(
                            author.kind(),
                            author.sort,
                            author.geometry(emitter.particle_count, emitter.first_particle),
                        )
                        .map_err(|error| ParticleGpuError::RendererMatrix(error.into()))?,
                    })
                })
                .collect::<Result<Vec<_>, ParticleGpuError>>()?;
            let spawn_id = u32::try_from(snapshot.emitters().len())
                .map_err(|_| ParticleGpuError::InvalidSnapshotRange)?;
            let free = lifecycle.capacity.saturating_sub(snapshot.len() as u32);
            if free != 0 {
                let container = &lifecycle.spawn_renderer;
                authors.push(container);
                let author = &container.renderers[0];
                routes.push(ParticleEmitterRenderRoute {
                    emitter_id: spawn_id,
                    active_start: snapshot.len() as u32,
                    particle_count: free,
                    renderer: ParticleRendererRoute::new(
                        author.kind(),
                        author.sort,
                        author.geometry(free, snapshot.len() as u32),
                    )
                    .map_err(|error| ParticleGpuError::RendererMatrix(error.into()))?,
                });
            }
            // The bridge has exactly one route per emitter. Consumers fan out
            // explicitly and never upload CPU counts or duplicate particle pools.
            if self.renderer_blueprint.is_none()
                || self.gpu_bridge.emitter_routes() != routes.as_slice()
                || renderer_authoring_changed
            {
                let registry = routes
                    .iter()
                    .zip(&authors)
                    .enumerate()
                    .flat_map(|(route_index, (route, container))| {
                        container
                            .renderers
                            .iter()
                            .map(move |author| (route_index, route, author))
                    })
                    .map(|(route_index, route, author)| {
                        Ok(ParticleEmitterRendererConfig {
                            emitter_id: route.emitter_id,
                            route_index: route_index as u32,
                            renderer: ParticleRendererRoute::new(
                                author.kind(),
                                author.sort,
                                author.geometry(route.particle_count, route.active_start),
                            )
                            .map_err(|error| ParticleGpuError::RendererMatrix(error.into()))?,
                            material: ParticleSurfaceMaterial {
                                scale: author.material.scale,
                                softness: author.material.softness,
                                depth_fade_distance: author.material.depth_fade_distance,
                                tint: author.material.tint,
                                ribbon_width: author.material.ribbon_width,
                                strip: prism_render_gpudriven::particle::renderer_matrix::ParticleStripSettings {
                                    uv_units_per_world_unit: author.material.strip_uv_units_per_world_unit,
                                    max_segment_length: author.material.strip_max_segment_length,
                                    max_birth_gap: author.material.strip_max_birth_gap,
                                },
                            },
                        })
                    })
                    .collect::<Result<Vec<_>, ParticleGpuError>>()?;
                // Spawn reuses a slot without changing its emitter identity.
                // Bootstrap ranges therefore bound each emitter for the entire
                // lifecycle, including recycled particles; the spawn route owns
                // only the initially free slots. Every renderer reserves that
                // emitter's bound independently; fan-out work is not free memory.
                let work_capacities = registry
                    .iter()
                    .map(|consumer| routes[consumer.route_index as usize].particle_count)
                    .collect::<Vec<_>>();
                self.renderer_blueprint = Some(RendererBlueprint {
                    registry,
                    work_capacities,
                    authors: authors
                        .iter()
                        .flat_map(|container| container.renderers.iter().cloned())
                        .collect(),
                });
                self.gpu_emitters = None;
                if let Some(views) = self.native_views.as_mut() {
                    views.retire_all();
                }
            }
            if self.native_views.is_none() && self.gpu_emitters.is_none() {
                self.gpu_emitters =
                    Some(self.build_renderers(queue, ParticleCullView::unrestricted())?);
            }
            self.gpu_bridge
                .set_emitter_routes(queue, &routes)
                .map_err(ParticleGpuError::SimulationBridge)?;
            if let Some(checkpoint) = inputs.checkpoint.as_ref() {
                if checkpoint.step() > inputs.total_steps() {
                    return Err(ParticleGpuError::InvalidSnapshotRange);
                }
                self.gpu_simulation
                    .restore_checkpoint(queue, checkpoint)
                    .map_err(|error| {
                        inputs.checkpoint_error =
                            Some(format!("checkpoint restore failed: {error:?}"));
                        ParticleGpuError::InvalidSnapshotRange
                    })?;
                self.applied_steps = checkpoint.step();
            } else {
                let mut initial = (0..capacity)
                    .map(|index| {
                        let mut particle = GpuParticle::dead_for_emitter(
                            index as u32,
                            spawn_id,
                            lifecycle.spawn_renderer.ribbon_id().unwrap_or(0),
                        );
                        particle.radius = config.particle_size;
                        particle
                    })
                    .collect::<Vec<_>>();
                let mut emitter_index = 0_usize;
                for (index, entry) in snapshot.particles().iter().enumerate() {
                    while let Some(emitter) = snapshot.emitters().get(emitter_index) {
                        let emitter_end = emitter
                            .first_particle
                            .checked_add(emitter.particle_count)
                            .ok_or(ParticleGpuError::InvalidSnapshotRange)?;
                        if (index as u32) < emitter_end {
                            break;
                        }
                        emitter_index += 1;
                    }
                    let record = entry.particle;
                    let finite = record.position.x.is_finite()
                        && record.position.y.is_finite()
                        && record.position.z.is_finite()
                        && record.velocity.x.is_finite()
                        && record.velocity.y.is_finite()
                        && record.velocity.z.is_finite()
                        && record.age.is_finite()
                        && record.lifetime.is_finite()
                        && record.size.is_finite();
                    if !finite
                        || record.lifetime < 0.0
                        || record.age < 0.0
                        || (record.lifetime > 0.0 && record.age >= record.lifetime)
                    {
                        return Err(ParticleGpuError::InvalidParticleRecord { index });
                    }
                    if record.lifetime == 0.0 || record.age < record.lifetime {
                        let emitter = snapshot
                            .emitters()
                            .get(emitter_index)
                            .filter(|emitter| {
                                (index as u32) >= emitter.first_particle
                                    && (index as u32)
                                        < emitter.first_particle + emitter.particle_count
                            })
                            .ok_or(ParticleGpuError::InvalidSnapshotRange)?;
                        let emitter_id = u32::try_from(emitter_index)
                            .map_err(|_| ParticleGpuError::InvalidSnapshotRange)?;
                        if emitter.emitter_entity != entry.emitter_entity {
                            return Err(ParticleGpuError::InvalidSnapshotRange);
                        }
                        let mut particle = GpuParticle::live(
                            [record.position.x, record.position.y, record.position.z],
                            [record.velocity.x, record.velocity.y, record.velocity.z],
                            record.lifetime,
                            record.size.max(config.particle_size),
                            record.particle_id,
                            emitter_id,
                        )
                        .with_ribbon_topology(
                            lifecycle.renderers[emitter_index].ribbon_id().unwrap_or(0),
                            record.birth_sequence,
                        );
                        particle.age = record.age;
                        initial[index] = particle;
                    }
                }
                self.gpu_simulation.initialize_particles(queue, &initial);
                // Populate compact/control outputs even when this render frame has no
                // fixed tick. This neither spawns nor advances age, forces or collisions.
                self.gpu_simulation.upload_inputs(
                    queue,
                    &[],
                    &[],
                    GpuSimulationParams {
                        dt: 0.0,
                        ..GpuSimulationParams::default()
                    },
                );
                queue.submit(&[self.gpu_simulation.encode()]);
            }
            // Commit only after bootstrap succeeds. In particular, a failed
            // retirement wait must not suppress authoring comparison on retry.
            self.lifecycle = Some(lifecycle.clone());
            self.gpu_initialized = true;
        }
        let total_steps = inputs.total_steps();
        let replay_end = self
            .applied_steps
            .saturating_add(inputs.max_replay_steps_per_frame as u64)
            .min(total_steps);
        let first = inputs
            .segments
            .partition_point(|s| s.end_step <= self.applied_steps);
        for segment in &inputs.segments[first..] {
            if self.applied_steps >= replay_end {
                break;
            }
            if segment.end_step <= self.applied_steps {
                continue;
            }
            let step = &segment.input;
            if let Err(error) = step.validate() {
                inputs.error = Some(error);
                return Err(ParticleGpuError::InvalidSnapshotRange);
            }
            while self.applied_steps < segment.end_step.min(replay_end) {
                self.gpu_simulation.upload_inputs(
                    queue,
                    &step.forces,
                    &step.colliders,
                    gpu_simulation_params(step),
                );
                // A submission per tick preserves input-buffer versions on the queue.
                queue.submit(&[self.gpu_simulation.encode()]);
                self.applied_steps += 1;
                // Snapshot this tick before the next reset overwrites events.
                // The committed floor suppresses gameplay delivery during replay.
                self.capture_gameplay_feedback(queue, inputs, render_frame);
                inputs.committed_steps = inputs.committed_steps.max(self.applied_steps);
            }
        }
        self.request_checkpoint(queue, inputs);
        inputs.recovering = self.applied_steps < total_steps;
        if inputs.recovering {
            return Ok(());
        }
        let fabric = simulation_fabric_frame(&lifecycle, render_frame);
        // Validate authored identities before assigning camera history ranges.
        // View remapping must not hide duplicate emitter/renderer identities.
        let authored_items = fabric
            .geometry()
            .draws()
            .iter()
            .map(|draw| FrameRenderItem {
                object_id: draw.object_id,
                domain: RenderDomain::Particle,
                history_key: draw.renderer_index,
                motion_source: MotionSource::Particle,
                payload: DomainPayload::Resource(draw.stream),
            });
        if let Err(error) = semantics.replace(render_frame, authored_items) {
            self.fabric_error = Some(ParticleFabricGpuBindingError::InvalidSemanticFrame(error));
            return Err(ParticleGpuError::InvalidSnapshotRange);
        }
        if self.native_views.is_some() {
            self.prepare_native_views(queue, &fabric)?;
            let publication = semantics.replace(
                render_frame,
                self.fabric_gpu_views()
                    .flat_map(|(_, frame)| fabric_semantic_items(frame)),
            );
            if let Err(error) = publication {
                self.fabric_error =
                    Some(ParticleFabricGpuBindingError::InvalidSemanticFrame(error));
                if let Some(views) = self.native_views.as_mut() {
                    views.revoke();
                }
                return Err(ParticleGpuError::InvalidSnapshotRange);
            }
            self.fabric_semantics = Some(semantics);
            return Ok(());
        }
        let renderers = self
            .gpu_emitters
            .as_ref()
            .ok_or(ParticleGpuError::InvalidSnapshotRange)?;
        match ParticleFabricGpuFrame::from_emitters(
            &fabric,
            &self.gpu_simulation,
            &self.gpu_renderer,
            renderers,
        ) {
            Ok(frame) => {
                // Publish only metadata backed by this frame's validated GPU
                // bindings. Capacity ranges are not exact live counts: those
                // and actual positions remain resident on the GPU.
                let items = frame.iter().flat_map(fabric_semantic_items);
                if let Err(error) = semantics.replace(render_frame, items) {
                    self.fabric_error =
                        Some(ParticleFabricGpuBindingError::InvalidSemanticFrame(error));
                    return Err(ParticleGpuError::InvalidSnapshotRange);
                }
                self.fabric_semantics = Some(semantics);
                self.fabric = frame;
            }
            Err(error) => {
                self.fabric_error = Some(error);
                return Err(ParticleGpuError::InvalidSnapshotRange);
            }
        }
        Ok(())
    }

    /// The latest valid semantic-to-physical Fabric binding, if this frame
    /// contains GPU particle routes (capacity envelopes, not proven live counts).
    /// An empty or rejected snapshot returns no binding and therefore cannot
    /// reuse an older GPU resource mapping.
    #[must_use]
    pub fn fabric_gpu_frame(&self) -> Option<&ParticleFabricGpuFrame> {
        self.fabric.as_ref()
    }

    /// Validated metadata for the latest successful GPU Prepare's render frame.
    ///
    /// A native aggregator must consume this after Prepare and check `frame()`
    /// against its render frame. Entries identify GPU routes and canonical
    /// resource roles with conservative finite bounds, not exact liveness or
    /// positions from a later CPU snapshot. The exact-count role is absent.
    /// `None` revokes admission on disabled, invalid, or recovering input; a
    /// successful frame with no routes is `Some` with an empty item slice.
    #[must_use]
    pub fn fabric_semantic_frame(&self) -> Option<&RenderFabricFrame> {
        self.fabric_semantics.as_ref()
    }

    /// Detailed reason the current Fabric frame could not be bound to live RHI
    /// resources. GPU upload failures remain in [`ParticleUploadStatus`].
    #[must_use]
    pub const fn fabric_binding_error(&self) -> Option<ParticleFabricGpuBindingError> {
        self.fabric_error
    }

    /// Appends camera work, producing the shared bridge once per prepared frame.
    /// Later graphs import its resident outputs only after actual submission.
    pub fn append_gpu_to_graph(
        &self,
        graph: &mut RenderGraph,
        color: TextureHandle,
        depth: TextureHandle,
    ) -> Result<(TextureHandle, TextureHandle), ParticleGpuGraphError> {
        let Some(fabric) = self.fabric.as_ref() else {
            return Ok((color, depth));
        };
        let renderers = self
            .gpu_emitters
            .as_ref()
            .ok_or(ParticleFabricGpuBindingError::MissingRecorderDraw)?;
        self.append_fabric_to_graph(graph, color, depth, fabric, renderers)
    }

    /// Draws only the accepted native camera, using that camera's local HDR and
    /// scene depth. Missing cameras never fall back to the headless fixture.
    pub fn append_view_gpu_to_graph(
        &self,
        camera: prism_ecs::prelude::Entity,
        graph: &mut RenderGraph,
        color: TextureHandle,
        depth: TextureHandle,
    ) -> Result<(TextureHandle, TextureHandle), ParticleGpuGraphError> {
        let Some(view) = self
            .native_views
            .as_ref()
            .and_then(|views| views.active.get(&camera))
        else {
            return Ok((color, depth));
        };
        let Some(fabric) = view.fabric.as_ref() else {
            return Ok((color, depth));
        };
        let [x, y, width, height] = view.camera.viewport();
        for attachment in [color, depth] {
            if graph.texture_extent(attachment).is_none_or(|extent| {
                x.checked_add(width).is_none_or(|end| end > extent.width)
                    || y.checked_add(height).is_none_or(|end| end > extent.height)
                    || extent.depth_or_array_layers != 1
            }) {
                return Err(ParticleGpuGraphError::InvalidViewport { camera });
            }
        }
        self.append_fabric_to_graph(graph, color, depth, fabric, &view.renderer)
    }

    fn append_fabric_to_graph(
        &self,
        graph: &mut RenderGraph,
        color: TextureHandle,
        depth: TextureHandle,
        fabric: &ParticleFabricGpuFrame,
        renderers: &Arc<ParticleRendererOwner>,
    ) -> Result<(TextureHandle, TextureHandle), ParticleGpuGraphError> {
        if !self.core_lease.usable.load(Ordering::Acquire) {
            return Err(ParticleGpuGraphError::DeviceUnavailable);
        }
        fabric.validate_owners(&self.gpu_simulation, renderers)?;
        // Fixed steps were already submitted in queue order during preparation.
        // Rendering only imports their resident outputs; zero-step frames cannot
        // accidentally simulate again.
        let key = Arc::as_ptr(&self.core_lease) as usize;
        let cached = graph
            .blackboard()
            .get::<ParticleGraphBridges>()
            .and_then(|bridges| bridges.0.get(&key))
            .cloned();
        let bridge = if let Some(cached) = cached {
            self.core_lease.retain_for_graph(graph);
            graph.retain_external_owner(renderers.clone());
            cached
        } else {
            let publication_key = ParticleBridgeKey {
                frame: fabric.frame(),
                device_generation: self.device_generation,
                lifecycle: self
                    .lifecycle
                    .as_ref()
                    .map_or(0, |lifecycle| Arc::as_ptr(lifecycle) as usize),
            };
            let mut publication = self
                .bridge_publication
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let reuse = if let Some(previous) = publication.as_ref() {
                let producer = previous.producer.upgrade();
                let submitted = previous.submitted.load(Ordering::Acquire)
                    || producer.as_ref().is_some_and(|token| token.is_submitted());
                if !submitted && producer.is_some() {
                    // Reject before imports/pins: no second producer can race a
                    // graph which might still be submitted with these outputs.
                    return Err(ParticleGpuGraphError::PendingBridgeSubmission);
                }
                submitted && previous.complete && previous.key == publication_key
            } else {
                false
            };
            // Pin before registering the producer, including recorder errors.
            self.core_lease.retain_for_graph(graph);
            graph.retain_external_owner(renderers.clone());
            let bridge = if reuse {
                self.import_bridge_outputs(graph)
            } else {
                let submitted = Arc::new(AtomicBool::new(false));
                let producer = graph.submission_token();
                graph.retain_external_owner(Arc::new(ParticleBridgeSubmissionWitness {
                    token: producer.upgrade().expect("live graph submission token"),
                    submitted: submitted.clone(),
                }));
                // Register before recorder admission. A partial/error graph may
                // contain bridge writes and must block a replacement until drop.
                *publication = Some(ParticleBridgePublication {
                    key: publication_key,
                    producer,
                    submitted,
                    complete: false,
                });
                let simulation = import_simulation_outputs(graph, &self.gpu_simulation);
                let output = self.gpu_bridge.append_to_graph(graph, &simulation)?;
                // A submitted partial/error graph cannot publish incomplete
                // bridge output, even if its caller ignores recorder failure.
                publication
                    .as_mut()
                    .expect("registered bridge producer")
                    .complete = true;
                ParticleBridgeOutputs {
                    renderer_input: output.renderer_input,
                    renderer_input_count: output.renderer_input_count,
                    emitter_route_table: output.emitter_route_table,
                }
            };
            if graph.blackboard().get::<ParticleGraphBridges>().is_none() {
                graph.blackboard_mut().set(ParticleGraphBridges::default());
            }
            graph
                .blackboard_mut()
                .get_mut::<ParticleGraphBridges>()
                .expect("initialized bridge cache")
                .0
                .insert(key, bridge.clone());
            bridge
        };
        let output = renderers.append_to_graph(
            graph,
            bridge.renderer_input,
            bridge.renderer_input_count,
            bridge.emitter_route_table,
            color,
            depth,
        )?;
        for renderer in &output.emitters {
            fabric.consume_recorder_outputs(
                bridge.renderer_input,
                renderer.input,
                renderer.visible_indices,
            )?;
        }
        Ok((output.color, output.depth))
    }

    /// Acquire a producer submission witness before consuming the graph.
    /// Consumer-only graphs need no acknowledgment and return None. Every graph
    /// still owns its core/renderer pins and participates in the Prepare gate.
    pub fn graph_output_lease(&self, graph: &RenderGraph) -> Option<ParticleGraphOutputLease> {
        let publication = self
            .bridge_publication
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let publication = publication.as_ref()?;
        if !Weak::ptr_eq(&publication.producer, &graph.submission_token()) {
            return None;
        }
        Some(ParticleGraphOutputLease {
            token: publication.producer.clone(),
            submitted: publication.submitted.clone(),
        })
    }

    fn import_bridge_outputs(&self, graph: &mut RenderGraph) -> ParticleBridgeOutputs {
        // Match the producer's physical descriptors exactly, not new storage.
        // Conservative compute-write entry states synchronize the prior queue
        // producer with this graph's storage reads without a CPU/GPU wait.
        let mut import = |name, buffer, size, usage| {
            graph.import_buffer(
                name,
                BufferDesc::new(size, usage),
                ImportedBuffer {
                    buffer,
                    entry_state: BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
                },
            )
        };
        ParticleBridgeOutputs {
            renderer_input: import(
                "particle-resident-bridge-input",
                self.gpu_renderer.input_buffer(),
                u64::from(self.gpu_simulation.capacity()) * PARTICLE_RENDER_INPUT_STRIDE,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            ),
            renderer_input_count: import(
                "particle-resident-bridge-count",
                self.gpu_renderer.input_count_buffer(),
                4,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            emitter_route_table: import(
                "particle-resident-bridge-routes",
                self.gpu_bridge.emitter_route_table_buffer(),
                PARTICLE_EMITTER_ROUTE_HEADER_BYTES
                    + u64::from(self.gpu_bridge.emitter_route_capacity())
                        * PARTICLE_EMITTER_ROUTE_STRIDE,
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            ),
        }
    }

    /// Releases all resources while the owning device is still usable.
    pub fn destroy(self) {
        let Self {
            gpu_simulation,
            gpu_bridge,
            gpu_renderer,
            gpu_emitters,
            native_views,
            core_lease,
            ..
        } = self;
        core_lease.retire(RetiredParticleCore {
            simulation: gpu_simulation,
            bridge: gpu_bridge,
            input: gpu_renderer,
        });
        // A recorded graph retains both leases until explicit GPU retirement.
        // Renderer bind groups must die before the shared inputs they reference.
        drop(gpu_emitters);
        drop(native_views);
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
    let generation = world
        .resource::<RenderRuntimeResource<I>>()
        .runtime()
        .device_generation();
    if world
        .resource::<RenderRuntimeResource<I>>()
        .runtime()
        .device_lost()
        .is_some()
    {
        world.resource::<ParticleGpuBuildQueue>().abandon_device();
        if let Some(state) = world.get_resource::<ParticleGpuState>() {
            state.abandon();
        }
        return;
    }
    if let Some(pending) = world.resource::<ParticleGpuBuildQueue>().pending.as_ref() {
        if pending.generation != generation {
            pending.allowed.store(false, Ordering::Release);
        }
    }
    let desired = world
        .resource::<ParticleGpuSimulationInputs>()
        .lifecycle
        .as_ref()
        .map(|lifecycle| (lifecycle.capacity, lifecycle.serial));
    if world
        .get_resource::<ParticleGpuState>()
        .is_some_and(|state| {
            state.device_generation != generation
                || desired.is_none_or(|(capacity, _)| capacity != state.gpu_simulation.capacity())
        })
    {
        let state = world.remove_resource::<ParticleGpuState>().unwrap();
        // Never call a lost device. Usable-device transitions must retire queued work.
        if state.device_generation == generation
            && world
                .resource::<RenderRuntimeResource<I>>()
                .runtime()
                .queue()
                .wait_idle()
        {
            state.destroy();
        } else {
            state.abandon();
            state.destroy();
        }
    }
    let Some((capacity, serial)) = desired.filter(|(capacity, _)| *capacity != 0) else {
        return;
    };
    if world.get_resource::<ParticleGpuState>().is_some() {
        return;
    }
    let build_queue = world.resource_mut::<ParticleGpuBuildQueue>();
    // Wait for a stale worker before starting another one: rapid capacity/device
    // changes cannot create an unbounded collection of background GPU builds.
    if build_queue.pending.is_some() || build_queue.failed == Some((generation, capacity, serial)) {
        return;
    }
    let runtime = world.resource::<RenderRuntimeResource<I>>();
    let device = runtime.runtime().device_handle();
    let Some(permit) = pipeline_build::WorkerPermit::acquire() else {
        return;
    };
    let allowed = Arc::new(AtomicBool::new(true));
    let worker_allowed = allowed.clone();
    let (sender, receiver) = mpsc::channel();
    let worker = std::thread::Builder::new()
        .name(String::from("prism-particle-pipeline"))
        .spawn(move || {
            let _permit = permit;
            let result = pipeline_build::build(device, generation, capacity, worker_allowed);
            // Result owns rollback even if shutdown dropped the receiver after
            // construction completed but before prepare could consume it.
            let _ = sender.send(result);
        });
    if worker.is_err() {
        world.resource_mut::<ParticleGpuBuildQueue>().failed = Some((generation, capacity, serial));
        world.resource_mut::<ParticleUploadStatus>().disabled_reason =
            Some(ParticleGpuError::PipelineWorkerUnavailable);
        return;
    }
    world.resource_mut::<ParticleGpuBuildQueue>().pending = Some(PendingParticleGpuBuild {
        generation,
        capacity,
        lifecycle_serial: serial,
        allowed,
        receiver: Mutex::new(receiver),
    });
}

/// Builds the device-independent Fabric projection before GPU preparation.
/// Must run after snapshot and render-record extraction. The snapshot retains
/// its independent source simulation clock; Fabric uses the render identity.
pub fn extract_particle_fabric_frame(_main: &mut World, render: &mut World) {
    let render_frame = render
        .get_resource::<crate::RenderRecord>()
        // Direct-call tests may omit the render clock; production installs it.
        .map_or(0, |record| record.frame_index);
    let lowered = {
        let snapshot = render.resource::<ExtractedParticleSnapshot>();
        ParticleFabricFrame::from_runtime_snapshot(
            render_frame,
            snapshot.emitters(),
            snapshot.particles(),
        )
    };
    render.init_resource::<PreparedParticleFabricFrame>();
    render
        .resource_mut::<PreparedParticleFabricFrame>()
        .replace_from_lowering(render_frame, lowered);
}

/// Uploads the extracted snapshot during the render prepare phase.
/// The CPU Fabric projection is already published by extraction.
pub fn prepare_particle_snapshot<I>(world: &mut World)
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    init_particle_gpu_state::<I>(world);
    let generation = world
        .resource::<RenderRuntimeResource<I>>()
        .runtime()
        .device_generation();
    let mut build_queue = world
        .remove_resource::<ParticleGpuBuildQueue>()
        .unwrap_or_default();
    if let Some(pending) = build_queue.pending.as_ref() {
        let desired = world
            .resource::<ParticleGpuSimulationInputs>()
            .lifecycle
            .as_ref()
            .map(|lifecycle| (lifecycle.capacity, lifecycle.serial));
        let key = (
            pending.generation,
            pending.capacity,
            pending.lifecycle_serial,
        );
        // The worker creates only device/capacity resources. A reset/seed change
        // during construction uses the latest bootstrap at first preparation.
        let current = pending.generation == generation
            && desired.is_some_and(|(capacity, _)| pending.capacity == capacity);
        let completed = pending
            .receiver
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .try_recv();
        match completed {
            Ok(Ok(result)) => {
                let state = result.into_state();
                build_queue.pending = None;
                if current {
                    world.insert_resource(state);
                    let status = world.resource_mut::<ParticleUploadStatus>();
                    status.disabled_reason = None;
                } else if state.device_generation == generation {
                    state.destroy();
                } else {
                    state.abandon();
                    state.destroy();
                }
            }
            Ok(Err(error)) => {
                build_queue.pending = None;
                if current {
                    build_queue.failed = Some(key);
                    world.resource_mut::<ParticleUploadStatus>().disabled_reason = Some(error);
                }
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                build_queue.pending = None;
                if current {
                    build_queue.failed = Some(key);
                    world.resource_mut::<ParticleUploadStatus>().disabled_reason =
                        Some(ParticleGpuError::PipelineWorkerUnavailable);
                }
            }
            Err(mpsc::TryRecvError::Empty) => {}
        }
    }
    world.insert_resource(build_queue);
    let mut inputs = world
        .remove_resource::<ParticleGpuSimulationInputs>()
        .unwrap_or_default();
    let status = world.resource_mut::<ParticleUploadStatus>();
    status.error = None;
    status.input_error = inputs.overflow.or(inputs.error);
    status.checkpoint_error = inputs.checkpoint_error.clone();
    status.checkpoint_backpressure = inputs.checkpoint_backpressure;
    status.checkpoint_step = inputs
        .checkpoint
        .as_ref()
        .map(|checkpoint| checkpoint.step());
    status.recovering = inputs.recovering;
    status.total_steps = inputs.total_steps();
    status.applied_steps = 0;
    let Some(mut state) = world.remove_resource::<ParticleGpuState>() else {
        inputs.recovering = inputs.enabled
            && inputs.lifecycle.is_some()
            && inputs.error.is_none()
            && inputs.overflow.is_none()
            && inputs.total_steps() != 0;
        world.resource_mut::<ParticleUploadStatus>().recovering = inputs.recovering;
        world.insert_resource(inputs);
        return;
    };
    state.set_gameplay_feedback(
        world
            .get_resource::<gameplay_feedback::ParticleGameplayFeedback>()
            .cloned(),
    );
    let render_frame = world
        .get_resource::<crate::RenderRecord>()
        .map_or(0, |record| record.frame_index);
    let requested = world
        .get_resource::<ExtractedParticleViews>()
        .cloned()
        .unwrap_or_default();
    let budget = world
        .get_resource::<ParticleViewBudget>()
        .copied()
        .unwrap_or_default();
    // Select native mode even for zero cameras; production must never invent a
    // fixture projection. Mode selection does not touch any queued GPU data.
    let views = state
        .native_views
        .get_or_insert_with(NativeParticleViews::default);
    views.requested = requested;
    views.budget = budget;
    state.gpu_emitters = None;
    let result = {
        let runtime = world.resource::<RenderRuntimeResource<I>>();
        let config = world.resource::<RenderAppConfig>();
        if state.device_generation == runtime.runtime().device_generation() {
            state.prepare_gpu_snapshot(runtime.runtime().queue(), config, &mut inputs, render_frame)
        } else {
            state.fabric = None;
            state.fabric_semantics = None;
            state.fabric_error = None;
            if let Some(views) = state.native_views.as_mut() {
                views.revoke();
            }
            Err(ParticleGpuError::InvalidSnapshotRange)
        }
    };
    let status = world.resource_mut::<ParticleUploadStatus>();
    status.error = result.err();
    status.input_error = inputs.overflow.or(inputs.error);
    status.checkpoint_error = inputs.checkpoint_error.clone();
    status.checkpoint_backpressure = inputs.checkpoint_backpressure;
    status.checkpoint_step = inputs
        .checkpoint
        .as_ref()
        .map(|checkpoint| checkpoint.step());
    status.recovering = inputs.recovering;
    status.applied_steps = state.applied_steps;
    world.insert_resource(inputs);
    world.insert_resource(state);
}

/// Admits changed graphs to bounded asynchronous GPU materialization and
/// uploads current inputs of resident variants. Pending initial graphs are
/// nonfatal; pending or failed replacements preserve the last valid variant.
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
    ember_build::prepare(
        &mut registry,
        &extracted,
        &external,
        &mut status,
        runtime.runtime().device_handle(),
        runtime.runtime().queue(),
        runtime.runtime().device_generation(),
    );
}

#[cfg(test)]
mod simulation_input_tests {
    use super::*;
    use prism_app::{App, FixedPreUpdate};
    use prism_particle_app::{
        set_particle_fixed_timestep, ParticleSimulationConfig, ParticleSimulationParams,
        ParticleSimulationPlugin,
    };
    use prism_render_architecture::particle::forces::{Force, ForceEntry};
    use prism_render_architecture::particle::{IntegratorKind, Vec3};
    use prism_render_driver::{Backend, DeviceCapabilities, Features, Limits};
    use prism_render_driver_null::{NullDevice, NullInstance, NullQueue};

    pub(super) fn particle_device(limits: Limits) -> (NullDevice, NullQueue) {
        (
            NullDevice::new_with_capabilities(DeviceCapabilities {
                backend: Backend::Noop,
                features: Features::INDIRECT_DRAW,
                limits,
            }),
            NullQueue::new(),
        )
    }

    #[test]
    fn particle_constructor_rejects_missing_indirect_before_allocation() {
        let device = NullDevice::new_with_capabilities(DeviceCapabilities {
            backend: Backend::Noop,
            features: Features::from_bits_truncate(0),
            limits: Limits::desktop(),
        });
        let baseline = device.live_resource_count();
        assert!(matches!(
            ParticleGpuState::new(&device, Arc::new(device.clone()), 1, 4),
            Err(ParticleGpuError::MissingIndirectDraw)
        ));
        assert_eq!(device.live_resource_count(), baseline);
    }

    #[test]
    fn particle_constructor_failure_releases_each_completed_stage() {
        use prism_render_gpudriven::{GpuSimulationError, ParticleSimulationBridgeError};

        // Simulation preflight must allocate nothing on the portable floor.
        let (device, _) = particle_device(Limits::baseline());
        let baseline = device.live_resource_count();
        assert!(matches!(
            ParticleGpuState::new(&device, Arc::new(device.clone()), 1, 4),
            Err(ParticleGpuError::Simulation(
                GpuSimulationError::LimitExceeded {
                    resource: "storage_buffers_per_stage",
                    ..
                }
            ))
        ));
        assert_eq!(device.live_resource_count(), baseline);

        // Simulation needs nine workgroups, but the renderer rounds its sort
        // capacity up to 1024 and needs sixteen. Destroy the completed pool.
        let (device, _) = particle_device(Limits {
            max_compute_workgroups_per_dimension: 10,
            ..Limits::desktop()
        });
        let baseline = device.live_resource_count();
        assert!(matches!(
            ParticleGpuState::new(&device, Arc::new(device.clone()), 1, 513),
            Err(ParticleGpuError::RendererMatrix(_))
        ));
        assert_eq!(device.live_resource_count(), baseline);

        // The bridge needs eleven storage bindings. Both already-created
        // simulation and renderer stages must retire if only ten are enabled.
        let (device, _) = particle_device(Limits {
            max_storage_buffers_per_shader_stage: 10,
            ..Limits::desktop()
        });
        let baseline = device.live_resource_count();
        assert!(matches!(
            ParticleGpuState::new(&device, Arc::new(device.clone()), 1, 4),
            Err(ParticleGpuError::SimulationBridge(
                ParticleSimulationBridgeError::StorageBufferStageLimitTooSmall {
                    requested: 11,
                    limit: 10,
                }
            ))
        ));
        assert_eq!(device.live_resource_count(), baseline);

        let (device, _) = particle_device(Limits::desktop());
        let baseline = device.live_resource_count();
        let state = ParticleGpuState::new(&device, Arc::new(device.clone()), 1, 4)
            .expect("desktop Null device supports the complete particle ABI");
        assert!(device.live_resource_count() > baseline);
        state.destroy();
        assert_eq!(device.live_resource_count(), baseline);
    }

    pub(super) fn routed_simulation_app() -> (App, World) {
        use prism_particle_app::EmitterRuntimeComponent;
        use prism_render_architecture::particle::emitter::{
            BurstSpawn, Emitter, EmitterShape, SpawnParams,
        };
        use prism_render_architecture::particle::runtime::{EmitterRuntime, EmitterRuntimeConfig};
        use prism_render_architecture::particle::EmitterHandle;

        let mut app = App::new();
        app.add_plugins(ParticleSimulationPlugin);
        app.world_mut()
            .resource_mut::<ParticleSimulationConfig>()
            .capacity = 4;
        // Include an empty emitter before live ones: route_index is a compact
        // registry ordinal, whereas emitter_id retains the bootstrap ordinal.
        for index in 0..3 {
            let mut runtime = EmitterRuntime::new(
                Emitter::new(
                    EmitterHandle(index),
                    EmitterShape::Point,
                    SpawnParams::default(),
                ),
                EmitterRuntimeConfig {
                    capacity: 1,
                    bursts: vec![BurstSpawn {
                        time: 0.1,
                        count: 1,
                    }],
                    lifetime: 10.0,
                    ..EmitterRuntimeConfig::default()
                },
                Vec3::ZERO,
                Vec3::ZERO,
            )
            .unwrap();
            if index != 0 {
                runtime.step(0.1);
            }
            app.world_mut().spawn(EmitterRuntimeComponent::new(runtime));
        }
        let mut render = World::new();
        extract_particle_simulation_inputs(app.world_mut(), &mut render);
        (app, render)
    }

    #[derive(Default)]
    struct ParticleRecordingQueue {
        writes: std::sync::Mutex<Vec<(BufferId, u64, Vec<u8>)>>,
        fail_retirement: std::sync::atomic::AtomicBool,
        retirement_waits: std::sync::atomic::AtomicUsize,
    }

    impl RenderQueue for ParticleRecordingQueue {
        fn write_buffer(&self, buffer: BufferId, offset: u64, data: &[u8]) {
            self.writes
                .lock()
                .unwrap()
                .push((buffer, offset, data.to_vec()));
        }

        fn write_texture(
            &self,
            _: prism_render_driver::TextureWrite,
            _: &[u8],
            _: prism_render_driver::ImageDataLayout,
        ) {
        }

        fn submit(&self, _: &[prism_render_driver::CommandBuffer]) {}
    }

    impl FrameSync for ParticleRecordingQueue {
        fn submit_tracked(
            &self,
            _: &[prism_render_driver::CommandBuffer],
        ) -> prism_render_driver::CompletionToken {
            Box::new(())
        }

        fn wait_for(&self, _: &(dyn core::any::Any + Send + Sync)) -> bool {
            true
        }

        fn wait_idle(&self) -> bool {
            use std::sync::atomic::Ordering;
            self.retirement_waits.fetch_add(1, Ordering::Relaxed);
            !self.fail_retirement.swap(false, Ordering::Relaxed)
        }

        fn on_submitted_work_done(&self, callback: Box<dyn FnOnce() + Send + 'static>) {
            callback();
        }
    }

    #[test]
    fn authored_renderers_reach_bootstrap_and_retry_failed_same_route_reset() {
        use prism_particle_app::{
            ParticleMaterialStyle, ParticleMeshVertex as AuthoredVertex, ParticleRendererConfig,
        };
        use std::sync::atomic::Ordering;

        let (device, _) = particle_device(Limits::desktop());
        let baseline = device.live_resource_count();
        let queue = ParticleRecordingQueue::default();
        let (mut app, mut render) = routed_simulation_app();
        let old = render
            .resource::<ParticleGpuSimulationInputs>()
            .lifecycle
            .as_ref()
            .unwrap()
            .clone();
        let mesh_entity = old.bootstrap.emitters()[1].emitter_entity;
        let mesh: ParticleRendererComponent = ParticleRendererConfig {
            renderer: ParticleSurfaceRenderer::Mesh {
                vertices: Arc::from([
                    AuthoredVertex {
                        position: [-0.4, -0.3, 0.0],
                        uv: [0.0, 0.0],
                    },
                    AuthoredVertex {
                        position: [0.4, -0.3, 0.0],
                        uv: [1.0, 0.0],
                    },
                    AuthoredVertex {
                        position: [0.0, 0.4, 0.0],
                        uv: [0.5, 1.0],
                    },
                ]),
                indices: Arc::from([0, 1, 2]),
            },
            material: ParticleMaterialStyle {
                tint: [2.0, 0.25, 0.5, 0.75],
                ..Default::default()
            },
            ..Default::default()
        }
        .into();
        assert!(app.world_mut().insert(mesh_entity, mesh.clone()));
        assert!(app.world_mut().insert(
            old.bootstrap.emitters()[2].emitter_entity,
            ParticleRendererComponent::from(ParticleRendererConfig {
                renderer: ParticleSurfaceRenderer::Ribbon { ribbon_id: 31 },
                ..Default::default()
            })
        ));
        assert!(app.world_mut().insert(
            old.spawn_entity,
            ParticleRendererComponent::from(ParticleRendererConfig {
                renderer: ParticleSurfaceRenderer::Beam { ribbon_id: 97 },
                ..Default::default()
            })
        ));
        app.world_mut()
            .resource_mut::<ParticleSimulationConfig>()
            .reset();
        extract_particle_simulation_inputs(app.world_mut(), &mut render);
        let mut inputs = render
            .remove_resource::<ParticleGpuSimulationInputs>()
            .unwrap();
        let first = inputs.lifecycle.as_ref().unwrap().clone();
        let mut state = ParticleGpuState::new(&device, Arc::new(device.clone()), 1, 4).unwrap();
        let config = RenderAppConfig::default();
        state
            .prepare_gpu_snapshot(&queue, &config, &mut inputs, 10)
            .unwrap();
        let stages = state.gpu_emitters.as_ref().unwrap().emitters();
        assert_eq!(
            stages
                .iter()
                .map(|s| s.route().geometry.kind())
                .collect::<Vec<_>>(),
            [RendererKind::Mesh, RendererKind::Ribbon, RendererKind::Beam]
        );
        assert_eq!(
            state
                .gpu_bridge
                .emitter_routes()
                .iter()
                .map(|r| r.emitter_id)
                .collect::<Vec<_>>(),
            [1, 2, 3]
        );
        let prepared = simulation_fabric_frame(&first, 10);
        assert_eq!(
            prepared
                .geometry()
                .draws()
                .iter()
                .map(|d| d.geometry.kind())
                .collect::<Vec<_>>(),
            [RendererKind::Mesh, RendererKind::Ribbon, RendererKind::Beam]
        );
        let handles = stages
            .iter()
            .map(ParticleRendererGpu::visible_indices)
            .collect::<Vec<_>>();
        let float_at = |bytes: &[u8], offset: usize| {
            f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
        };
        let word_at = |bytes: &[u8], offset: usize| {
            u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
        };
        {
            let writes = queue.writes.lock().unwrap();
            assert!(writes.iter().any(|(_, _, bytes)| bytes.len() == 64
                && float_at(bytes, 16) == 2.0
                && float_at(bytes, 28) == 0.75));
            // WGSL packs each vertex as position.xyz/pad, uv.xy/pad/pad.
            let expected_mesh = [
                [-0.4_f32, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.4, -0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.4, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0],
            ]
            .into_iter()
            .flatten()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
            assert_eq!(expected_mesh.len(), 3 * 32);
            let mesh_uploads = writes
                .iter()
                .filter(|(_, offset, bytes)| *offset == 0 && *bytes == expected_mesh)
                .collect::<Vec<_>>();
            assert_eq!(mesh_uploads.len(), 1, "one exact authored 3D mesh upload");
            let mesh_buffer = mesh_uploads[0].0;
            for stage in stages {
                assert_ne!(mesh_buffer, stage.input_buffer());
                assert_ne!(mesh_buffer, stage.visible_indices());
                assert_ne!(mesh_buffer, stage.control());
            }
            let bootstrap = &writes
                .iter()
                .find(|(buffer, _, bytes)| {
                    *buffer == state.gpu_simulation.particles() && bytes.len() == 4 * 64
                })
                .unwrap()
                .2;
            // The first authored emitter was empty, so neither emitter identity
            // nor authoring can be indexed by the compact renderer ordinal.
            assert_eq!(word_at(bootstrap, 44), 1);
            assert_eq!(word_at(bootstrap, 52), 1);
            assert_eq!(word_at(bootstrap, 64 + 44), 2);
            assert_eq!(word_at(bootstrap, 64 + 48), 31);
            assert_eq!(word_at(bootstrap, 64 + 52), 1);
            for slot in 2..4 {
                assert_eq!(word_at(bootstrap, slot * 64 + 44), 3);
                assert_eq!(word_at(bootstrap, slot * 64 + 48), 97);
                assert_eq!(word_at(bootstrap, slot * 64 + 36), 0);
            }
        }
        render.insert_resource(inputs);
        let mut edited = mesh;
        edited.renderers[0].material.tint = [0.125, 3.0, 0.25, 0.5];
        assert!(app.world_mut().insert(mesh_entity, edited));
        extract_particle_simulation_inputs(app.world_mut(), &mut render);
        inputs = render
            .remove_resource::<ParticleGpuSimulationInputs>()
            .unwrap();
        let uploads = queue.writes.lock().unwrap().len();
        state
            .prepare_gpu_snapshot(&queue, &config, &mut inputs, 11)
            .unwrap();
        assert!(Arc::ptr_eq(state.lifecycle.as_ref().unwrap(), &first));
        assert_eq!(
            queue.writes.lock().unwrap().len(),
            uploads,
            "editing does not reset a resident pool"
        );
        render.insert_resource(inputs);
        app.world_mut()
            .resource_mut::<ParticleSimulationConfig>()
            .reset();
        extract_particle_simulation_inputs(app.world_mut(), &mut render);
        inputs = render
            .remove_resource::<ParticleGpuSimulationInputs>()
            .unwrap();
        let next = inputs.lifecycle.as_ref().unwrap().clone();
        let live = device.live_resource_count();
        queue.fail_retirement.store(true, Ordering::Relaxed);
        assert!(state
            .prepare_gpu_snapshot(&queue, &config, &mut inputs, 12)
            .is_err());
        assert!(state.fabric_gpu_frame().is_none());
        assert!(
            Arc::ptr_eq(state.lifecycle.as_ref().unwrap(), &first),
            "failed prepare cannot acknowledge new authoring"
        );
        assert_eq!(device.live_resource_count(), live);
        assert_eq!(queue.writes.lock().unwrap().len(), uploads);
        state
            .prepare_gpu_snapshot(&queue, &config, &mut inputs, 13)
            .unwrap();
        assert!(Arc::ptr_eq(state.lifecycle.as_ref().unwrap(), &next));
        assert_eq!(queue.retirement_waits.load(Ordering::Relaxed), 2);
        assert_ne!(
            state
                .gpu_emitters
                .as_ref()
                .unwrap()
                .emitters()
                .iter()
                .map(ParticleRendererGpu::visible_indices)
                .collect::<Vec<_>>(),
            handles
        );
        assert!(queue.writes.lock().unwrap()[uploads..]
            .iter()
            .any(|(_, _, bytes)| bytes.len() == 64
                && float_at(bytes, 16) == 0.125
                && float_at(bytes, 20) == 3.0));
        assert_eq!(
            device.live_resource_count(),
            live,
            "same-size reset retires every previous resource"
        );
        state.destroy();
        assert_eq!(device.live_resource_count(), baseline);
    }

    #[test]
    fn shared_emitter_consumers_have_distinct_fabric_identity_and_local_work() {
        use prism_particle_app::{ParticleMaterialStyle, ParticleRendererConfig};

        let (device, _) = particle_device(Limits::desktop());
        let baseline = device.live_resource_count();
        let queue = ParticleRecordingQueue::default();
        let (mut app, mut render) = routed_simulation_app();
        let emitter = render
            .resource::<ParticleGpuSimulationInputs>()
            .lifecycle
            .as_ref()
            .unwrap()
            .bootstrap
            .emitters()[1]
            .emitter_entity;
        assert!(app.world_mut().insert(
            emitter,
            ParticleRendererComponent {
                renderers: vec![
                    ParticleRendererConfig::default(),
                    ParticleRendererConfig {
                        material: ParticleMaterialStyle {
                            tint: [0.25, 2.0, 0.5, 1.0],
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                ],
            }
        ));
        app.world_mut()
            .resource_mut::<ParticleSimulationConfig>()
            .reset();
        extract_particle_simulation_inputs(app.world_mut(), &mut render);
        let mut inputs = render
            .remove_resource::<ParticleGpuSimulationInputs>()
            .unwrap();
        let mut state = ParticleGpuState::new(&device, Arc::new(device.clone()), 1, 4).unwrap();
        state
            .prepare_gpu_snapshot(&queue, &RenderAppConfig::default(), &mut inputs, 19)
            .unwrap();

        let routes = state.gpu_bridge.emitter_routes();
        assert_eq!(routes.len(), 3, "fan-out must not duplicate bridge routes");
        let stages = state.gpu_emitters.as_ref().unwrap().emitters();
        assert_eq!(stages.len(), 4);
        assert_eq!(stages[0].input_buffer(), stages[1].input_buffer());
        assert_eq!(
            stages[0].input_count_buffer(),
            stages[1].input_count_buffer()
        );
        assert_ne!(stages[0].sprite_args(), stages[1].sprite_args());
        assert_ne!(stages[0].visible_indices(), stages[1].visible_indices());
        assert_ne!(stages[0].control(), stages[1].control());
        let draws = state.fabric_gpu_frame().unwrap().draws();
        assert_eq!(draws.len(), 4);
        assert_eq!(draws[0].object_id, draws[1].object_id);
        assert_eq!((draws[0].renderer_index, draws[1].renderer_index), (0, 1));
        for (draw, stage) in draws.iter().zip(stages) {
            assert_eq!(draw.stream.size, 4 * PARTICLE_RENDER_INPUT_STRIDE);
            assert_eq!(draw.alive.size, u64::from(stage.work_capacity()) * 4);
        }
        assert_eq!(
            draws[0].alive.size, 4,
            "local budget, not whole-pool extent"
        );
        let items = state.fabric_semantic_frame().unwrap().items();
        assert_eq!(items.len(), 12);
        for (item, draw) in items.iter().zip(draws.iter().flat_map(|draw| [draw; 3])) {
            assert_eq!(item.object_id, draw.object_id);
            assert_eq!(item.history_key, draw.renderer_index);
        }
        state.destroy();
        assert_eq!(device.live_resource_count(), baseline);
    }

    #[test]
    fn invalid_renderer_lifecycle_is_rejected_before_upload_or_allocation() {
        let (device, _) = particle_device(Limits::desktop());
        let baseline = device.live_resource_count();
        let queue = ParticleRecordingQueue::default();
        let (_, mut render) = routed_simulation_app();
        let mut inputs = render
            .remove_resource::<ParticleGpuSimulationInputs>()
            .unwrap();
        let mut lifecycle = inputs.lifecycle.as_ref().unwrap().as_ref().clone();
        lifecycle.spawn_renderer.renderers[0].material.ribbon_width = f32::NAN;
        inputs.lifecycle = Some(Arc::new(lifecycle));
        let mut state = ParticleGpuState::new(&device, Arc::new(device.clone()), 1, 4).unwrap();
        let live = device.live_resource_count();
        assert!(state
            .prepare_gpu_snapshot(&queue, &RenderAppConfig::default(), &mut inputs, 1)
            .is_err());
        assert!(matches!(
            inputs.error,
            Some(ParticleSimulationInputError::InvalidRendererAuthoring { .. })
        ));
        assert!(state.lifecycle.is_none());
        assert!(state.fabric_gpu_frame().is_none());
        assert!(queue.writes.lock().unwrap().is_empty());
        assert_eq!(device.live_resource_count(), live);
        state.destroy();
        assert_eq!(device.live_resource_count(), baseline);
    }

    #[test]
    fn gpu_prepare_uses_render_record_while_simulation_is_paused() {
        // Null exercises publication and ownership, not physical GPU execution.
        let runtime = prism_render_runtime::RenderRuntime::headless(
            NullInstance::with_capabilities(DeviceCapabilities {
                backend: Backend::Noop,
                features: Features::INDIRECT_DRAW,
                limits: Limits::desktop(),
            }),
            prism_render_driver::Extent3d::new_2d(8, 8),
        )
        .unwrap();
        let device = runtime.device_handle();
        let state = ParticleGpuState::new(
            device.as_ref(),
            device.clone(),
            runtime.device_generation(),
            4,
        )
        .unwrap();
        let (mut app, mut render) = routed_simulation_app();
        let camera = app.world_mut().spawn((
            crate::visibility::PrismCamera::new(prism_math::Mat4::IDENTITY, [0, 0, 8, 8]),
            crate::visibility::PrismFrustum(prism_math::Frustum::from_planes(
                [prism_math::Plane::new(prism_math::Vec3::ZERO, 1.0); 6],
            )),
            prism_transform::GlobalTransform::default(),
        ));
        crate::surface_views::extract_surface_views(app.world_mut(), &mut render);
        extract_particle_views(app.world_mut(), &mut render);
        app.world_mut()
            .resource_mut::<ParticleSimulationConfig>()
            .paused = true;
        app.world_mut().insert_resource(crate::RenderRecord {
            frame_index: 999,
            ..Default::default()
        });
        render.insert_resource(RenderRuntimeResource::new(runtime));
        render.insert_resource(RenderAppConfig::default());
        render.insert_resource(ParticleUploadStatus::default());
        render.insert_resource(ParticleGpuBuildQueue::default());
        render.insert_resource(state);

        // No render record must use zero, never the main-world/simulation clock.
        prepare_particle_snapshot::<NullInstance>(&mut render);
        let state = render.resource::<ParticleGpuState>();
        let (published_camera, fabric) = state.fabric_gpu_views().next().unwrap();
        assert_eq!(published_camera, camera);
        assert_eq!(state.fabric_gpu_views().count(), 1);
        assert!(
            state.fabric_gpu_frame().is_none(),
            "production must not publish a headless camera"
        );
        let baseline = fabric.draws().to_vec();
        let semantics = state.fabric_semantic_frame().unwrap();
        assert_eq!(semantics.frame(), 0);
        assert_eq!(fabric.frame(), 0);
        assert_eq!(semantics.len(), baseline.len() * 3);
        assert!(!semantics
            .items()
            .iter()
            .any(|item| item.payload_role() == PARTICLE_EMITTER_COUNT_PAYLOAD_ROLE));
        for draw in &baseline {
            assert!(semantics.contains_object(draw.object_id));
            for payload in [
                DomainPayload::Value(ValuePayload {
                    value: PayloadValue::Bounds {
                        min: [-f32::MAX; 3],
                        max: [f32::MAX; 3],
                    },
                    role: PARTICLE_EMITTER_POSITION_BOUNDS_PAYLOAD_ROLE,
                }),
                DomainPayload::Resource(draw.stream.semantic),
                DomainPayload::Resource(draw.alive.semantic),
            ] {
                assert!(semantics.items().contains(&FrameRenderItem {
                    object_id: draw.object_id,
                    domain: RenderDomain::Particle,
                    history_key: draw.history_key,
                    motion_source: MotionSource::Particle,
                    payload,
                }));
            }
        }
        let metadata = semantics.items().to_vec();
        for render_frame in [41, 42] {
            render.insert_resource(crate::RenderRecord {
                frame_index: render_frame,
                ..Default::default()
            });
            app.world_mut().run_schedule(FixedPreUpdate);
            extract_particle_simulation_inputs(app.world_mut(), &mut render);
            crate::surface_views::extract_surface_views(app.world_mut(), &mut render);
            extract_particle_views(app.world_mut(), &mut render);
            prepare_particle_snapshot::<NullInstance>(&mut render);
            let state = render.resource::<ParticleGpuState>();
            assert_eq!(state.applied_steps, 0);
            let (published_camera, fabric) = state.fabric_gpu_views().next().unwrap();
            assert_eq!(published_camera, camera);
            assert_eq!(fabric.frame(), render_frame);
            assert_eq!(fabric.draws(), baseline);
            assert_eq!(state.fabric_semantic_frame().unwrap().frame(), render_frame);
            assert_eq!(state.fabric_semantic_frame().unwrap().items(), metadata);
            assert!(render.resource::<ParticleUploadStatus>().error.is_none());
        }

        render.resource_mut::<ParticleGpuSimulationInputs>().enabled = false;
        render.resource_mut::<crate::RenderRecord>().frame_index = 43;
        prepare_particle_snapshot::<NullInstance>(&mut render);
        assert!(render
            .resource::<ParticleGpuState>()
            .fabric_gpu_frame()
            .is_none());
        assert_eq!(
            render
                .resource::<ParticleGpuState>()
                .fabric_gpu_views()
                .count(),
            0
        );
        assert!(render
            .resource::<ParticleGpuState>()
            .fabric_semantic_frame()
            .is_none());

        render.resource_mut::<ParticleGpuSimulationInputs>().enabled = true;
        render.resource_mut::<crate::RenderRecord>().frame_index = 44;
        prepare_particle_snapshot::<NullInstance>(&mut render);
        assert_eq!(
            render
                .resource::<ParticleGpuState>()
                .fabric_semantic_frame()
                .unwrap()
                .frame(),
            44
        );
        let inputs = render.resource_mut::<ParticleGpuSimulationInputs>();
        // A malformed lifecycle cannot publish two identities for one role.
        let mut lifecycle = inputs.lifecycle.as_ref().unwrap().as_ref().clone();
        lifecycle.spawn_entity = lifecycle.bootstrap.emitters()[1].emitter_entity;
        inputs.lifecycle = Some(Arc::new(lifecycle));
        render.resource_mut::<crate::RenderRecord>().frame_index = 45;
        prepare_particle_snapshot::<NullInstance>(&mut render);
        let state = render.resource::<ParticleGpuState>();
        assert!(state.fabric_gpu_frame().is_none());
        assert_eq!(state.fabric_gpu_views().count(), 0);
        assert!(state.fabric_semantic_frame().is_none());
        assert!(matches!(
            state.fabric_binding_error(),
            Some(ParticleFabricGpuBindingError::InvalidSemanticFrame(
                RenderFabricFrameError::DuplicateItem { .. }
            ))
        ));
        assert!(render.resource::<ParticleUploadStatus>().error.is_some());
        render
            .remove_resource::<ParticleGpuState>()
            .unwrap()
            .destroy();
    }

    #[test]
    fn render_app_emitter_graph_shares_bridge_ssa_but_owns_materials_and_draws() {
        use prism_render_driver::{
            Barrier, BufferState, Extent3d, Pass, PipelineStages, RenderCommand, TextureFormat,
        };
        use prism_render_graph::{LogicalResource, PassFlags, SizeClass, TextureDesc};

        let (device, queue) = particle_device(Limits::desktop());
        let baseline = device.live_resource_count();
        let (mut app, mut render) = routed_simulation_app();
        let mut inputs = render
            .remove_resource::<ParticleGpuSimulationInputs>()
            .unwrap();
        let lifecycle = inputs.lifecycle.as_ref().unwrap().clone();
        let mut state = ParticleGpuState::new(&device, Arc::new(device.clone()), 1, 4).unwrap();
        let config = RenderAppConfig::default();
        state
            .prepare_gpu_snapshot(&queue, &config, &mut inputs, 40)
            .unwrap();
        assert_eq!(
            state.applied_steps, 0,
            "bootstrap does not invent a fixed tick"
        );
        let initial_metadata = state.fabric_semantic_frame().unwrap().items().to_vec();
        assert_eq!(
            state
                .gpu_bridge
                .emitter_routes()
                .iter()
                .map(|r| r.emitter_id)
                .collect::<Vec<_>>(),
            [1, 2, 3]
        );
        let renderers = Arc::get_mut(state.gpu_emitters.as_mut().unwrap())
            .expect("test fixture does not share emitter renderers");
        assert_eq!(
            renderers.emitters().len(),
            3,
            "two live emitters and one GPU spawn route"
        );
        assert_eq!(
            renderers
                .emitters()
                .iter()
                .map(ParticleRendererGpu::work_capacity)
                .collect::<Vec<_>>(),
            [1, 1, 2],
            "emitter work budgets partition the pool instead of duplicating it"
        );
        for (index, renderer) in renderers.emitters_mut().iter_mut().enumerate() {
            let mut material = ParticleSurfaceMaterial::for_route(renderer.route());
            material.tint = [index as f32 / 3.0, 0.5, 1.0, 1.0];
            renderer.set_material(material).unwrap();
        }
        renderers
            .set_view(&queue, ParticleCullView::unrestricted())
            .unwrap();
        let renderers = state.gpu_emitters.as_ref().unwrap();
        let expected_args = renderers
            .emitters()
            .iter()
            .map(ParticleRendererGpu::sprite_args)
            .collect::<Vec<_>>();
        let expected_alive = renderers
            .emitters()
            .iter()
            .map(ParticleRendererGpu::visible_indices)
            .collect::<Vec<_>>();
        let frame = state.fabric_gpu_frame().unwrap();
        for (index, draw) in frame.draws().iter().enumerate() {
            assert_eq!(draw.stream.buffer, state.gpu_renderer.input_buffer());
            assert_eq!(draw.alive.buffer, expected_alive[index]);
            for previous in 0..index {
                assert_ne!(expected_alive[index], expected_alive[previous]);
                assert_ne!(expected_args[index], expected_args[previous]);
                assert_ne!(
                    renderers.emitters()[index].control(),
                    renderers.emitters()[previous].control()
                );
            }
        }
        // A swapped per-emitter visible stream cannot silently bind another
        // emitter's draw, even though both streams share the canonical role.
        let mut invalid = frame.clone();
        invalid.draws[0].alive.buffer = expected_alive[1];
        assert!(matches!(
            invalid.validate_owners(&state.gpu_simulation, renderers),
            Err(ParticleFabricGpuBindingError::OwnerMismatch { .. })
        ));
        invalid.draws.pop();
        assert_eq!(
            invalid.validate_owners(&state.gpu_simulation, renderers),
            Err(ParticleFabricGpuBindingError::RendererCountMismatch {
                draws: 2,
                renderers: 3
            })
        );

        let mut graph = RenderGraph::new(Extent3d::new_2d(8, 8));
        let mut targets = None;
        graph.add_raster_pass(
            "particle test scene clear",
            PassFlags::SIDE_EFFECT,
            |builder| {
                let color = builder.create_texture(
                    "HDR",
                    TextureDesc::color(TextureFormat::Rgba16Float)
                        .with_size(SizeClass::absolute_2d(8, 8)),
                );
                let depth = builder.create_texture(
                    "depth",
                    TextureDesc::depth(TextureFormat::Depth32Float)
                        .with_size(SizeClass::absolute_2d(8, 8)),
                );
                targets = Some((
                    builder.color_attachment(color),
                    builder.depth_attachment(depth),
                ));
                |_| {}
            },
        );
        let (color, depth) = targets.unwrap();
        let (output_color, output_depth) =
            state.append_gpu_to_graph(&mut graph, color, depth).unwrap();
        assert_eq!(output_color.version(), color.version() + 3);
        assert_eq!(output_depth, depth);
        let recorded = graph
            .record(&device)
            .expect("formal RenderApp particle graph records");
        let plan = recorded.plan();
        let passes = &recorded.commands().passes;
        assert_eq!(passes.len(), plan.order.len());
        let mut reset_reads = Vec::new();
        let mut draws = Vec::new();
        let mut materials = Vec::new();
        let mut cameras = Vec::new();
        let mut material_groups = Vec::new();
        for (&pass_index, pass) in plan.order.iter().zip(passes) {
            let accesses = &plan.accesses[pass_index];
            match pass {
                Pass::Compute { label, .. }
                    if label.as_deref() == Some("particle-renderer-reset") =>
                {
                    let reads = accesses
                        .iter()
                        .filter(|access| {
                            matches!(access.resource, LogicalResource::Buffer(_))
                                && access.output_version.is_none()
                                && access.input_version > 0
                        })
                        .map(|access| (access.resource, access.input_version))
                        .collect::<Vec<_>>();
                    assert_eq!(
                        reads.len(),
                        3,
                        "input/count/route table must all consume bridge writes, not imports"
                    );
                    for &(resource, version) in &reads {
                        let writer = plan
                            .order
                            .iter()
                            .copied()
                            .find(|&candidate| {
                                plan.accesses[candidate].iter().any(|access| {
                                    access.resource == resource
                                        && access.output_version == Some(version)
                                })
                            })
                            .expect("bridge SSA producer exists");
                        assert!(
                            plan.position(writer).unwrap() < plan.position(pass_index).unwrap()
                        );
                    }
                    reset_reads.push(reads);
                }
                Pass::Render { commands, .. } if !commands.is_empty() => {
                    for command in commands {
                        match command {
                            RenderCommand::DrawIndirect { buffer, .. } => draws.push(*buffer),
                            RenderCommand::SetBindGroup {
                                index: 0,
                                bind_group,
                                ..
                            } => material_groups.push(*bind_group),
                            RenderCommand::Draw { .. }
                            | RenderCommand::DrawIndexed { .. }
                            | RenderCommand::DrawIndexedIndirect { .. } => {
                                panic!("unexpected aggregate or non-sprite draw")
                            }
                            _ => {}
                        }
                    }
                    // Declaration order is mesh storage, material uniform,
                    // camera uniform, then index buffer. Group 1 binds depth,
                    // not another emitter material.
                    let imports = accesses
                        .iter()
                        .filter(|access| {
                            matches!(access.resource, LogicalResource::Buffer(_))
                                && access.input_version == 0
                                && access.output_version.is_none()
                        })
                        .map(|access| access.resource)
                        .collect::<Vec<_>>();
                    let [mesh, material, camera, index] = imports.as_slice() else {
                        panic!("expected mesh/material/camera/index imports: {imports:?}");
                    };
                    for (ordinal, resource) in imports.iter().enumerate() {
                        assert!(!imports[..ordinal].contains(resource));
                    }
                    let uniform_imports = plan.barriers[plan.position(pass_index).unwrap()]
                        .iter()
                        .filter_map(|barrier| match barrier {
                            Barrier::Buffer(barrier)
                                if barrier.after
                                    == BufferState::uniform(
                                        PipelineStages::VERTEX_SHADER
                                            .union(PipelineStages::FRAGMENT_SHADER),
                                    ) =>
                            {
                                Some(LogicalResource::Buffer(barrier.buffer))
                            }
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(
                        uniform_imports,
                        [*material, *camera],
                        "material and camera identities must be the two raster uniforms"
                    );
                    materials.push(vec![*mesh, *index, *material]);
                    cameras.push(*camera);
                }
                _ => {}
            }
        }
        assert_eq!(
            draws, expected_args,
            "one independent indirect draw per registered emitter; no aggregate draw"
        );
        assert!(!draws.contains(&state.gpu_renderer.sprite_args()));
        assert_eq!(reset_reads.len(), 3);
        assert!(
            reset_reads.windows(2).all(|pair| pair[0] == pair[1]),
            "all emitters share identical bridge SSA versions"
        );
        assert_eq!(material_groups.len(), 3);
        assert_eq!(materials.len(), 3);
        assert_eq!(cameras.len(), 3);
        for index in 0..3 {
            assert!(materials
                .iter()
                .all(|imports| !imports.contains(&cameras[index])));
            for previous in 0..index {
                assert_ne!(material_groups[index], material_groups[previous]);
                assert_ne!(cameras[index], cameras[previous]);
                assert!(
                    materials[index]
                        .iter()
                        .all(|resource| !materials[previous].contains(resource)),
                    "mesh/index/material imports are independent per emitter"
                );
            }
        }
        recorded.release(&device);

        // Paused repeated extracts neither replay nor replace resident stages.
        render.insert_resource(inputs);
        app.world_mut()
            .resource_mut::<ParticleSimulationConfig>()
            .paused = true;
        for render_frame in 41..44 {
            app.world_mut().run_schedule(FixedPreUpdate);
            extract_particle_simulation_inputs(app.world_mut(), &mut render);
            let inputs = render.resource_mut::<ParticleGpuSimulationInputs>();
            state
                .prepare_gpu_snapshot(&queue, &config, inputs, render_frame)
                .unwrap();
            assert_eq!(state.fabric_gpu_frame().unwrap().frame(), render_frame);
            assert_eq!(state.fabric_semantic_frame().unwrap().frame(), render_frame);
            assert!(Arc::ptr_eq(inputs.lifecycle.as_ref().unwrap(), &lifecycle));
            assert_eq!(state.applied_steps, 0);
            assert_eq!(
                state
                    .gpu_emitters
                    .as_ref()
                    .unwrap()
                    .emitters()
                    .iter()
                    .map(ParticleRendererGpu::sprite_args)
                    .collect::<Vec<_>>(),
                expected_args
            );
        }
        // Mutating the CPU emitters after initialization cannot refill or
        // reroute the resident GPU pool, even on a paused zero-step frame.
        let emitters = app
            .world_mut()
            .query::<&mut prism_particle_app::EmitterRuntimeComponent>();
        for mut emitter in emitters.iter_mut(app.world_mut()) {
            // Trigger the initially empty emitter's burst before expiring it:
            // newborns are deliberately not aged on their spawn step.
            emitter.runtime_mut().step(0.1);
            emitter.runtime_mut().step(20.0);
        }
        extract_particle_simulation_inputs(app.world_mut(), &mut render);
        let inputs = render.resource_mut::<ParticleGpuSimulationInputs>();
        state
            .prepare_gpu_snapshot(&queue, &config, inputs, 44)
            .unwrap();
        assert_eq!(
            state.fabric_semantic_frame().unwrap().items(),
            initial_metadata
        );
        assert_eq!(state.fabric_semantic_frame().unwrap().frame(), 44);
        assert!(Arc::ptr_eq(inputs.lifecycle.as_ref().unwrap(), &lifecycle));
        assert_eq!(
            state
                .gpu_emitters
                .as_ref()
                .unwrap()
                .emitters()
                .iter()
                .map(ParticleRendererGpu::sprite_args)
                .collect::<Vec<_>>(),
            expected_args
        );

        // An explicit reset, unlike normal extraction, uses current CPU state
        // and replaces the old per-emitter stages with the free-slot route.
        app.world_mut()
            .resource_mut::<ParticleSimulationConfig>()
            .reset();
        extract_particle_simulation_inputs(app.world_mut(), &mut render);
        let inputs = render.resource_mut::<ParticleGpuSimulationInputs>();
        assert!(inputs.lifecycle.as_ref().unwrap().bootstrap.is_empty());
        assert!(!Arc::ptr_eq(inputs.lifecycle.as_ref().unwrap(), &lifecycle));
        state
            .prepare_gpu_snapshot(&queue, &config, inputs, 45)
            .unwrap();
        assert_eq!(state.gpu_emitters.as_ref().unwrap().emitters().len(), 1);
        assert_eq!(state.fabric_gpu_frame().unwrap().draws().len(), 1);
        assert_eq!(state.applied_steps, 0);
        state.destroy();
        assert_eq!(
            device.live_resource_count(),
            baseline,
            "borrowed bridge input and per-emitter resources retire exactly once"
        );
    }

    #[test]
    fn render_app_unsupported_checkpoint_preserves_replay_with_bounded_backpressure() {
        let (device, queue) = particle_device(Limits::desktop());
        let baseline = device.live_resource_count();
        let (mut app, mut render) = routed_simulation_app();
        let config = app.world_mut().resource_mut::<ParticleSimulationConfig>();
        config.max_history_segments = 2;
        config.max_recovery_steps = 3;
        let mut state = ParticleGpuState::new(&device, Arc::new(device.clone()), 1, 4).unwrap();
        let config = RenderAppConfig::default();
        for tick in 0..12 {
            app.world_mut()
                .resource_mut::<ParticleSimulationConfig>()
                .params
                .condition_x = tick as f32;
            app.world_mut().run_schedule(FixedPreUpdate);
            extract_particle_simulation_inputs(app.world_mut(), &mut render);
            let inputs = render.resource_mut::<ParticleGpuSimulationInputs>();
            state
                .prepare_gpu_snapshot(&queue, &config, inputs, 100 + tick)
                .unwrap();
            assert_eq!(state.applied_steps, (tick + 1).min(2));
            assert_eq!(inputs.overflow, None);
            assert_eq!(inputs.error, None);
            assert!(inputs.segments.len() <= 2);
            assert_eq!(inputs.history_start_step, 0);
            assert!(inputs.checkpoint.is_none());
            if tick >= 2 {
                assert!(inputs.checkpoint_backpressure);
            }
            assert!(state.fabric_gpu_frame().is_some());
        }
        let mut replacement =
            ParticleGpuState::new(&device, Arc::new(device.clone()), 2, 4).unwrap();
        let inputs = render.resource_mut::<ParticleGpuSimulationInputs>();
        assert!(inputs
            .checkpoint_error
            .as_deref()
            .unwrap()
            .contains("Unsupported"));
        assert_eq!(inputs.history_start_step, 0);
        assert_eq!(inputs.total_steps(), 2);
        replacement
            .prepare_gpu_snapshot(&queue, &config, inputs, 112)
            .unwrap();
        assert_eq!(inputs.error, None);
        assert_eq!(replacement.applied_steps, 2);
        assert!(replacement.gpu_initialized);
        assert!(replacement.fabric_gpu_frame().is_some());
        assert!(replacement.fabric_semantic_frame().is_some());

        // Reset is data loss/rebootstrap, not a checkpoint recovery claim.
        // The replacement must consume the latest lifecycle even when paused.
        let authored = app.world_mut().resource_mut::<ParticleSimulationConfig>();
        authored.seed = 91;
        authored.paused = true;
        authored.reset();
        extract_particle_simulation_inputs(app.world_mut(), &mut render);
        let inputs = render.resource_mut::<ParticleGpuSimulationInputs>();
        replacement
            .prepare_gpu_snapshot(&queue, &config, inputs, 113)
            .unwrap();
        assert_eq!(inputs.error, None);
        assert!(inputs.checkpoint.is_none());
        assert_eq!(inputs.checkpoint_error, None);
        assert!(!inputs.checkpoint_backpressure);
        assert_eq!(inputs.total_steps(), 0);
        assert_eq!(replacement.applied_steps, 0);
        assert_eq!(replacement.lifecycle.as_ref().unwrap().seed, 91);
        assert!(Arc::ptr_eq(
            replacement.lifecycle.as_ref().unwrap(),
            inputs.lifecycle.as_ref().unwrap()
        ));
        assert!(replacement.fabric_gpu_frame().is_some());
        assert_eq!(replacement.fabric_semantic_frame().unwrap().frame(), 113);
        replacement.destroy();
        state.destroy();
        assert_eq!(device.live_resource_count(), baseline);
    }

    #[test]
    fn simulation_params_keep_fixed_clock_and_all_authored_scalars() {
        let params = ParticleSimulationParams {
            restitution: 0.3,
            friction: 0.4,
            spawn_count: 7,
            spawn_position: [1.0, 2.0, 3.0],
            spawn_velocity: [4.0, 5.0, 6.0],
            spawn_lifetime: 8.0,
            condition_x: 9.0,
            condition_enabled: true,
            integrator: IntegratorKind::Rk2,
        };
        let step = ParticleSimulationStep {
            dt: 0.125,
            forces: Vec::new(),
            colliders: Vec::new(),
            params,
        };
        let gpu = gpu_simulation_params(&step);
        assert!((gpu.dt - 0.125).abs() < 1.0e-6);
        assert!((gpu.restitution - params.restitution).abs() < 1.0e-6);
        assert!((gpu.friction - params.friction).abs() < 1.0e-6);
        assert_eq!(gpu.spawn_count, params.spawn_count);
        assert_eq!(gpu.spawn_position, params.spawn_position);
        assert_eq!(gpu.spawn_velocity, params.spawn_velocity);
        assert!((gpu.spawn_lifetime - params.spawn_lifetime).abs() < 1.0e-6);
        assert!((gpu.condition_x - params.condition_x).abs() < 1.0e-6);
        assert_eq!(gpu.condition_enabled, params.condition_enabled);
        assert_eq!(gpu.integrator, params.integrator);
    }

    #[test]
    fn render_plugin_preserves_main_world_simulation_configuration() {
        use prism_render_driver_null::NullInstance;
        let mut app = App::new();
        let mut config = ParticleSimulationConfig::default();
        config.capacity = 17;
        config.seed = 73;
        app.insert_resource(config);
        app.add_plugins(crate::PrismRenderPlugin::with_config(
            NullInstance::new(),
            RenderAppConfig::default(),
        ));
        assert_eq!(
            app.world().resource::<ParticleSimulationConfig>().capacity,
            17
        );
        assert_eq!(app.world().resource::<ParticleSimulationConfig>().seed, 73);
        let mut render = World::new();
        app.world_mut().run_schedule(FixedPreUpdate);
        extract_particle_simulation_inputs(app.world_mut(), &mut render);
        let inputs = render.resource::<ParticleGpuSimulationInputs>();
        assert_eq!(inputs.total_steps(), 1);
        let prepared = simulation_fabric_frame(inputs.lifecycle.as_ref().unwrap(), 1);
        assert_eq!(
            prepared.geometry().draws().len(),
            1,
            "GPU-only free slots have a stable source route"
        );
    }

    #[test]
    fn physical_gpu_simulation_inputs_consume_once_and_recover_latest_lifecycle() {
        use prism_render_driver_wgpu::{require_hardware_gpu, HardwareDeviceError, WgpuInstance};
        let (device, queue) = match WgpuInstance::new().request_default_hardware_device() {
            Ok(pair) => pair,
            Err(HardwareDeviceError::NoAdapter) => {
                assert!(
                    !require_hardware_gpu(),
                    "PRISM_REQUIRE_GPU=1 but no hardware adapter was found"
                );
                eprintln!("SKIPPED particle input lifecycle test: no hardware adapter");
                return;
            }
            Err(HardwareDeviceError::NonHardware(evidence)) => {
                assert!(!require_hardware_gpu(), "physical GPU required: {evidence}");
                eprintln!("SKIPPED particle input lifecycle test: {evidence}");
                return;
            }
            Err(HardwareDeviceError::Device(error)) => {
                panic!("hardware device creation failed: {error}")
            }
        };
        let mut app = App::new();
        app.add_plugins(ParticleSimulationPlugin);
        set_particle_fixed_timestep(&mut app, core::time::Duration::from_millis(20));
        let config = app.world_mut().resource_mut::<ParticleSimulationConfig>();
        config.capacity = 4;
        config.max_replay_steps_per_frame = 2;
        config.params.spawn_count = 1;
        config.params.spawn_lifetime = 10.0;
        config.forces.push(ForceEntry::new(Force::Gravity {
            acceleration: Vec3::new(0.0, -1.0, 0.0),
        }));
        let mut render = World::new();
        for _ in 0..3 {
            app.world_mut().run_schedule(FixedPreUpdate);
        }
        extract_particle_simulation_inputs(app.world_mut(), &mut render);
        let mut inputs = render
            .remove_resource::<ParticleGpuSimulationInputs>()
            .unwrap();
        let mut state = ParticleGpuState::new(&device, Arc::new(device.clone()), 1, 4).unwrap();
        let render_config = RenderAppConfig::default();
        state
            .prepare_gpu_snapshot(&queue, &render_config, &mut inputs, 500)
            .unwrap();
        assert_eq!(state.applied_steps, 2);
        assert!(inputs.recovering);
        assert!(
            state.fabric_gpu_frame().is_none(),
            "partial replay is never published"
        );
        assert!(state.fabric_semantic_frame().is_none());
        state
            .prepare_gpu_snapshot(&queue, &render_config, &mut inputs, 501)
            .unwrap();
        assert_eq!(state.applied_steps, 3);
        assert!(!inputs.recovering);
        assert_eq!(state.fabric_gpu_frame().unwrap().frame(), 501);
        assert_eq!(state.fabric_semantic_frame().unwrap().frame(), 501);
        let before = queue
            .read_buffer(
                state.gpu_simulation.particles(),
                4 * prism_render_gpudriven::GPU_PARTICLE_STRIDE,
            )
            .expect("particle pause baseline readback must complete");
        let metadata = state.fabric_semantic_frame().unwrap().items().to_vec();
        let bindings = state.fabric_gpu_frame().unwrap().draws().to_vec();
        render.insert_resource(inputs);
        app.world_mut()
            .resource_mut::<ParticleSimulationConfig>()
            .paused = true;
        for render_frame in 502..505 {
            app.world_mut().run_schedule(FixedPreUpdate);
            extract_particle_simulation_inputs(app.world_mut(), &mut render);
            inputs = render
                .remove_resource::<ParticleGpuSimulationInputs>()
                .unwrap();
            assert_eq!(inputs.total_steps(), 3);
            state
                .prepare_gpu_snapshot(&queue, &render_config, &mut inputs, render_frame)
                .unwrap();
            assert_eq!(state.applied_steps, 3);
            assert_eq!(state.fabric_gpu_frame().unwrap().frame(), render_frame);
            assert_eq!(state.fabric_gpu_frame().unwrap().draws(), bindings);
            assert_eq!(state.fabric_semantic_frame().unwrap().frame(), render_frame);
            assert_eq!(state.fabric_semantic_frame().unwrap().items(), metadata);
            assert_eq!(
                queue
                    .read_buffer(
                        state.gpu_simulation.particles(),
                        4 * prism_render_gpudriven::GPU_PARTICLE_STRIDE,
                    )
                    .expect("paused particle frame readback must complete"),
                before,
                "paused render frames must not spawn, integrate, or refill the GPU pool"
            );
            render.insert_resource(inputs);
        }

        // Extending an already consumed run consumes only its new tail.
        app.world_mut()
            .resource_mut::<ParticleSimulationConfig>()
            .paused = false;
        app.world_mut().run_schedule(FixedPreUpdate);
        extract_particle_simulation_inputs(app.world_mut(), &mut render);
        inputs = render
            .remove_resource::<ParticleGpuSimulationInputs>()
            .unwrap();
        assert_eq!(inputs.segments.len(), 1);
        state
            .prepare_gpu_snapshot(&queue, &render_config, &mut inputs, 505)
            .unwrap();
        assert_eq!(state.applied_steps, 4);

        // A later reset supplies a different bootstrap and scalar input. The
        // replacement device must use this lifecycle, not the original one.
        render.insert_resource(inputs);
        let config = app.world_mut().resource_mut::<ParticleSimulationConfig>();
        config.seed = 91;
        config.params.spawn_position = [10.0, 0.0, 0.0];
        config.reset();
        app.world_mut().run_schedule(FixedPreUpdate);
        extract_particle_simulation_inputs(app.world_mut(), &mut render);
        inputs = render
            .remove_resource::<ParticleGpuSimulationInputs>()
            .unwrap();
        assert_eq!(inputs.total_steps(), 1);
        assert_eq!(inputs.committed_steps, 0);
        state
            .prepare_gpu_snapshot(&queue, &render_config, &mut inputs, 506)
            .unwrap();
        let expected = queue
            .read_buffer(
                state.gpu_simulation.particles(),
                4 * prism_render_gpudriven::GPU_PARTICLE_STRIDE,
            )
            .expect("particle replacement baseline readback must complete");
        let mut replacement =
            ParticleGpuState::new(&device, Arc::new(device.clone()), 2, 4).unwrap();
        replacement
            .prepare_gpu_snapshot(&queue, &render_config, &mut inputs, 506)
            .unwrap();
        assert_eq!(replacement.lifecycle.as_ref().unwrap().seed, 91);
        let actual = queue
            .read_buffer(
                replacement.gpu_simulation.particles(),
                4 * prism_render_gpudriven::GPU_PARTICLE_STRIDE,
            )
            .expect("particle replacement readback must complete");
        // Slot claiming is atomic; compare live simulation values independently
        // of which free slot won the spawn (not a bitwise replay guarantee).
        let live_values = |bytes: &[u8]| {
            let mut values = bytes
                .chunks_exact(prism_render_gpudriven::GPU_PARTICLE_STRIDE as usize)
                .filter(|record| u32::from_le_bytes(record[36..40].try_into().unwrap()) != 0)
                .map(|record| record[..36].to_vec())
                .collect::<Vec<_>>();
            values.sort();
            values
        };
        let live = live_values(&actual);
        assert_eq!(live.len(), 1);
        assert_eq!(live, live_values(&expected));
        assert!((f32::from_le_bytes(live[0][..4].try_into().unwrap()) - 10.0).abs() < 1.0e-6);
        assert!(queue.wait_idle());
        replacement.destroy();
        state.destroy();
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use super::{
        extract_ember_external_gpu_resources, extract_ember_render_assets,
        extract_particle_fabric_frame, EmberAssetHandle, EmberCompiledGraphAssets,
        EmberExternalGpuResources, EmberGpuAssetRegistry, EmberRenderAssets,
        ExtractedEmberExternalGpuResources, ExtractedEmberRenderAssets, ParticleFabricGeometry,
        ParticleFabricGpuBindingError, ParticleFabricGpuFrame, ParticleFabricGpuResourceOwner,
        ParticleGpuError, ParticleGpuState, ParticleUploadStatus, PreparedParticleFabricFrame,
        PARTICLE_RENDER_INPUT_STRIDE,
    };
    use crate::{PrismRenderPlugin, RenderApp, RenderAppConfig, RenderRecord};
    use prism_app::{App, Last};
    use prism_ecs::prelude::Entity;
    use prism_ecs::world::World;
    use prism_particle_app::{
        extract_particle_snapshot, ExtractedParticleSnapshot, ParticleEmitterSnapshot,
        ParticleSnapshot, ParticleSnapshotEntry,
    };
    use prism_particle_render::ParticleFabricFrame;
    use prism_render_architecture::particle::asset_registry::EmberAssetRevision;
    use prism_render_architecture::particle::attributes::AttributeLayoutPlan;
    use prism_render_architecture::particle::authoring::{
        EmberAsset, GraphAssetRef, RendererConfig, RendererConfigId, RendererKind,
        SerializationFormat, ShadingSettings,
    };
    use prism_render_architecture::particle::compiler::CompiledGraph;
    use prism_render_architecture::particle::modules::ResourceDecl;
    use prism_render_architecture::particle::runtime::ParticleRecord;
    use prism_render_architecture::particle::EmberShadingModel;
    use prism_render_architecture::particle::{indirect_draw::RendererGeometry, Vec3};
    use prism_render_architecture::render_fabric::{
        PARTICLE_ALIVE_RESOURCE, PARTICLE_STREAM_RESOURCE,
    };
    use prism_render_driver::BufferUsages;
    use prism_render_driver_null::NullInstance;
    use prism_render_gpudriven::{
        CompiledGraphGpuExternalKind, CompiledGraphGpuGraphImport,
        CompiledGraphGpuResourceRegistry, CompiledGraphGpuResourceSlot, CompiledGraphGraphError,
    };
    use prism_render_graph::RenderGraph;

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

    fn compiled_graph_with_texture() -> CompiledGraph {
        CompiledGraph {
            source: String::from("@compute @workgroup_size(64) fn main() {}"),
            stage_order: vec![0],
            entry_points: vec![String::from("main")],
            dispatches: vec![
                prism_render_architecture::particle::compiler::CompiledDispatch {
                    stage_index: 0,
                    domain: prism_render_architecture::particle::IterationDomain::PerParticle,
                    default_count: 1,
                    iterations: 1,
                    entry_point: String::from("main"),
                },
            ],
            bindings: vec![
                ResourceDecl::texture2d("field", 0, 0),
                ResourceDecl::buffer("ember_dispatch_counts", 1, 0),
            ],
            ..empty_compiled_graph()
        }
    }

    fn entity(index: u32, generation: u32) -> Entity {
        Entity::from_bits(((generation as u64) << 32) | index as u64).unwrap()
    }

    fn runtime_particle(emitter: Entity, slot: u32, position: Vec3) -> ParticleSnapshotEntry {
        ParticleSnapshotEntry {
            emitter_entity: emitter,
            slot,
            particle: ParticleRecord {
                position,
                velocity: Vec3::ZERO,
                age: 0.0,
                lifetime: 1.0,
                particle_id: slot,
                birth_sequence: 1,
                size: 1.0,
            },
        }
    }

    fn particle_fabric_extract_worlds() -> (App, World) {
        use prism_particle_app::{EmitterRuntimeComponent, ParticleRuntimePlugin};
        use prism_render_architecture::particle::emitter::{
            BurstSpawn, Emitter, EmitterShape, SpawnParams,
        };
        use prism_render_architecture::particle::runtime::{EmitterRuntime, EmitterRuntimeConfig};
        use prism_render_architecture::particle::EmitterHandle;

        let mut app = App::new();
        app.add_plugins(ParticleRuntimePlugin);
        let mut runtime = EmitterRuntime::new(
            Emitter::new(
                EmitterHandle(0),
                EmitterShape::Point,
                SpawnParams::default(),
            ),
            EmitterRuntimeConfig {
                capacity: 1,
                bursts: vec![BurstSpawn {
                    time: 0.1,
                    count: 1,
                }],
                lifetime: 10.0,
                ..EmitterRuntimeConfig::default()
            },
            Vec3::ZERO,
            Vec3::ZERO,
        )
        .expect("test emitter configuration is valid");
        runtime.step(0.1);
        app.world_mut().spawn(EmitterRuntimeComponent::new(runtime));
        app.world_mut().run_schedule(Last);
        let mut render = World::new();
        extract_particle_snapshot(app.world(), &mut render);
        assert_eq!(render.resource::<ExtractedParticleSnapshot>().len(), 1);
        (app, render)
    }

    #[test]
    fn particle_fabric_extract_advances_render_frame_with_unchanged_simulation_snapshot() {
        let (mut app, mut render) = particle_fabric_extract_worlds();
        let snapshot = render.resource::<ExtractedParticleSnapshot>().clone();
        assert_eq!(snapshot.frame(), 1);
        app.world_mut().insert_resource(RenderRecord {
            frame_index: 999,
            ..RenderRecord::default()
        });

        for frame_index in [41, 42] {
            render.insert_resource(RenderRecord {
                frame_index,
                ..RenderRecord::default()
            });
            extract_particle_snapshot(app.world(), &mut render);
            extract_particle_fabric_frame(app.world_mut(), &mut render);

            let prepared = render.resource::<PreparedParticleFabricFrame>();
            assert_eq!(prepared.frame().frame(), frame_index);
            assert_eq!(prepared.geometry().frame(), frame_index);
            assert_eq!(prepared.geometry().draws().len(), 1);
            assert_eq!(prepared.error(), None);
            assert_eq!(prepared.consumption_error(), None);
            assert_eq!(render.resource::<ExtractedParticleSnapshot>(), &snapshot);
        }
    }

    #[test]
    fn particle_fabric_extract_without_render_clock_uses_zero() {
        let (mut app, mut render) = particle_fabric_extract_worlds();
        extract_particle_fabric_frame(app.world_mut(), &mut render);

        let prepared = render.resource::<PreparedParticleFabricFrame>();
        assert_eq!(prepared.frame().frame(), 0);
        assert_eq!(prepared.geometry().frame(), 0);
        assert_eq!(prepared.geometry().draws().len(), 1);
        assert_eq!(prepared.error(), None);
        assert_eq!(prepared.consumption_error(), None);
        assert_eq!(render.resource::<ExtractedParticleSnapshot>().frame(), 1);
    }

    #[test]
    fn particle_fabric_consumer_requires_resources_and_emits_sprite_geometry() {
        let emitter = entity(7, 2);
        let frame = ParticleFabricFrame::from_runtime_snapshot(
            31,
            &[ParticleEmitterSnapshot {
                emitter_entity: emitter,
                first_particle: 0,
                particle_count: 1,
            }],
            &[runtime_particle(emitter, 4, Vec3::new(1.0, 2.0, 3.0))],
        )
        .expect("valid runtime snapshot lowers to Fabric");

        let geometry = ParticleFabricGeometry::from_frame(&frame)
            .expect("canonical particle Fabric payloads are consumable");
        assert_eq!(geometry.frame(), 31);
        assert_eq!(geometry.draws().len(), 1);
        let draw = geometry.draws()[0];
        assert_eq!(
            draw.geometry,
            RendererGeometry::Sprite {
                alive_count: 1,
                first_instance: 0,
            }
        );
        assert_eq!(draw.bounds_min, [1.0, 2.0, 3.0]);
        assert_eq!(draw.bounds_max, [1.0, 2.0, 3.0]);
        assert_eq!(draw.stream.resource, PARTICLE_STREAM_RESOURCE);
        assert_eq!(draw.alive.resource, PARTICLE_ALIVE_RESOURCE);
    }

    #[test]
    fn particle_fabric_consumer_clears_stale_geometry_on_empty_frame() {
        let (mut app, mut render) = particle_fabric_extract_worlds();
        render.insert_resource(RenderRecord {
            frame_index: 41,
            ..RenderRecord::default()
        });
        extract_particle_fabric_frame(app.world_mut(), &mut render);
        assert_eq!(
            render
                .resource::<PreparedParticleFabricFrame>()
                .geometry()
                .draws()
                .len(),
            1
        );

        app.world_mut().remove_resource::<ParticleSnapshot>();
        extract_particle_snapshot(app.world(), &mut render);
        render.resource_mut::<RenderRecord>().frame_index = 42;
        extract_particle_fabric_frame(app.world_mut(), &mut render);
        let prepared = render.resource::<PreparedParticleFabricFrame>();
        assert_eq!(prepared.frame().frame(), 42);
        assert!(prepared.frame().is_empty());
        assert!(prepared.geometry().draws().is_empty());
        assert_eq!(prepared.geometry().frame(), 42);
        assert_eq!(prepared.error(), None);
        assert_eq!(prepared.consumption_error(), None);
    }

    #[test]
    fn particle_fabric_gpu_frame_resolves_roles_to_renderer_owned_rhi_buffers() {
        use prism_render_driver_wgpu::{require_hardware_gpu, HardwareDeviceError, WgpuInstance};

        let (device, _queue) = match WgpuInstance::new().request_default_hardware_device() {
            Ok(pair) => pair,
            Err(HardwareDeviceError::NoAdapter) => {
                assert!(
                    !require_hardware_gpu(),
                    "PRISM_REQUIRE_GPU=1 but no hardware adapter was found"
                );
                eprintln!("SKIPPED particle Fabric physical binding test: no adapter");
                return;
            }
            Err(HardwareDeviceError::NonHardware(evidence)) => {
                assert!(
                    !require_hardware_gpu(),
                    "PRISM_REQUIRE_GPU=1 requires physical GPU evidence: {evidence}"
                );
                eprintln!("SKIPPED particle Fabric physical binding test: {evidence}");
                return;
            }
            Err(HardwareDeviceError::Device(error)) => {
                panic!("physical GPU was found but device creation failed: {error}");
            }
        };
        let state = ParticleGpuState::new(&device, Arc::new(device.clone()), 1, 4)
            .expect("hardware RHI supports the particle GPU resource set");
        let emitter = entity(12, 4);
        let lowered = ParticleFabricFrame::from_runtime_snapshot(
            51,
            &[ParticleEmitterSnapshot {
                emitter_entity: emitter,
                first_particle: 0,
                particle_count: 1,
            }],
            &[runtime_particle(emitter, 0, Vec3::ZERO)],
        )
        .expect("valid runtime snapshot lowers to Fabric");
        let mut prepared = PreparedParticleFabricFrame::default();
        prepared.replace_from_lowering(51, Ok(lowered));

        let bound = ParticleFabricGpuFrame::from_prepared(
            &prepared,
            &state.gpu_simulation,
            &state.gpu_renderer,
        )
        .expect("canonical semantic resources resolve")
        .expect("one live emitter publishes one GPU draw");
        assert_eq!(bound.frame(), 51);
        assert_eq!(bound.draws().len(), 1);
        let draw = bound.draws()[0];
        assert_eq!(draw.stream.buffer, state.gpu_renderer.input_buffer());
        assert_eq!(draw.alive.buffer, state.gpu_renderer.visible_indices());
        assert_eq!(draw.stream.owner, ParticleFabricGpuResourceOwner::Renderer);
        assert_eq!(draw.alive.owner, ParticleFabricGpuResourceOwner::Renderer);
        assert_eq!(draw.stream.size, 4 * PARTICLE_RENDER_INPUT_STRIDE);
        assert_eq!(draw.alive.size, 4 * 4);
        assert_eq!(
            draw.stream.usage,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST
        );
        assert_eq!(
            draw.alive.usage,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC
        );

        let mut empty = PreparedParticleFabricFrame::default();
        empty.replace_from_lowering(52, Ok(ParticleFabricFrame::empty(52)));
        assert_eq!(
            ParticleFabricGpuFrame::from_prepared(
                &empty,
                &state.gpu_simulation,
                &state.gpu_renderer,
            )
            .expect("empty Fabric frame is valid"),
            None,
            "an empty frame cannot publish or reuse a previous physical binding"
        );

        state.destroy();
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
            Some(ParticleGpuError::MissingIndirectDraw)
        );
        let fabric = render.world.resource::<PreparedParticleFabricFrame>();
        assert_eq!(fabric.frame().frame(), 1);
        assert!(fabric.frame().is_empty());
        assert_eq!(fabric.error(), None);
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
        assert!(std::sync::Arc::ptr_eq(&first, &third));
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
            .set_texture(
                slot,
                prism_render_driver::TextureId::from_parts(4, 9),
                prism_render_driver::TextureViewId::from_parts(5, 9),
            )
            .expect("external slot publication succeeds");
        let import = CompiledGraphGpuGraphImport::Texture {
            desc: prism_render_graph::TextureDesc::color(
                prism_render_driver::TextureFormat::Rgba8Unorm,
            ),
            entry_state: prism_render_driver::TextureState::initial(),
            exit_state: prism_render_driver::TextureState::initial(),
            mip_levels: 1,
            array_layers: 1,
        };
        main.resource_mut::<EmberExternalGpuResources>()
            .set_graph_import(slot, revision, import.clone());

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
            CompiledGraphGpuExternalKind::Texture
        );
        assert_eq!(
            render
                .resource::<ExtractedEmberExternalGpuResources>()
                .graph_import(slot),
            Some(&import)
        );

        main.resource_mut::<EmberExternalGpuResources>()
            .registry_mut()
            .set_texture(
                slot,
                prism_render_driver::TextureId::from_parts(6, 9),
                prism_render_driver::TextureViewId::from_parts(7, 9),
            )
            .expect("replacement texture publication succeeds");
        extract_ember_external_gpu_resources(&mut main, &mut render);
        assert_eq!(
            render
                .resource::<ExtractedEmberExternalGpuResources>()
                .graph_import(slot),
            None,
            "metadata from an earlier resource revision must not be reused"
        );
    }

    #[test]
    fn ember_render_app_imports_external_texture_with_exact_frame_graph_description() {
        use prism_render_driver::{Instance as _, RenderDevice as _};

        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&Default::default())
            .expect("null adapter");
        let (device, queue) = instance
            .create_device(&adapter, &Default::default())
            .expect("null device");
        let texture = device.create_texture(&prism_render_driver::TextureDescriptor::new_2d(
            prism_render_driver::Extent3d::new_2d(1, 1),
            prism_render_driver::TextureFormat::Rgba8Unorm,
            prism_render_driver::TextureUsages::TEXTURE_BINDING,
        ));
        let view = device.create_texture_view(
            texture,
            &prism_render_driver::TextureViewDescriptor::default(),
        );
        let slot = CompiledGraphGpuResourceSlot::new(0, 0);
        let mut external = CompiledGraphGpuResourceRegistry::new();
        external
            .set_texture(slot, texture, view)
            .expect("texture publication succeeds");
        let mut registry = EmberGpuAssetRegistry::default();
        let handle = EmberAssetHandle(9);
        registry
            .upsert(
                &device,
                &queue,
                handle,
                GraphAssetRef(1),
                EmberAssetRevision::from_value(1),
                compiled_graph_with_texture(),
                &external,
            )
            .expect("compiled graph is materialized");
        registry
            .prepare(&queue, handle, &empty_asset(), &[], &[1], &[])
            .expect("compiled graph inputs are prepared");
        let import = CompiledGraphGpuGraphImport::Texture {
            desc: prism_render_graph::TextureDesc::color(
                prism_render_driver::TextureFormat::Rgba8Unorm,
            )
            .with_size(prism_render_graph::SizeClass::absolute_2d(1, 1))
            .with_usage(prism_render_driver::TextureUsages::TEXTURE_BINDING),
            entry_state: prism_render_driver::TextureState::initial(),
            exit_state: prism_render_driver::TextureState::initial(),
            mip_levels: 1,
            array_layers: 1,
        };
        let mut graph = RenderGraph::new(prism_render_driver::Extent3d::new_2d(1, 1));
        assert_eq!(
            registry.append_to_graph(
                &mut graph,
                0,
                &BTreeMap::from([(slot, import)]),
                &BTreeMap::new()
            ),
            Ok(())
        );
        assert!(graph.compile().is_ok());

        registry.destroy(&device);
        device.destroy_texture_view(view);
        device.destroy_texture(texture);
    }

    #[test]
    fn ember_render_app_requires_metadata_or_a_matching_existing_texture_view() {
        use prism_render_driver::{Instance as _, RenderDevice as _};

        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&Default::default())
            .expect("null adapter");
        let (device, queue) = instance
            .create_device(&adapter, &Default::default())
            .expect("null device");
        let texture = device.create_texture(&prism_render_driver::TextureDescriptor::new_2d(
            prism_render_driver::Extent3d::new_2d(1, 1),
            prism_render_driver::TextureFormat::Rgba8Unorm,
            prism_render_driver::TextureUsages::TEXTURE_BINDING,
        ));
        let view = device.create_texture_view(
            texture,
            &prism_render_driver::TextureViewDescriptor::default(),
        );
        let slot = CompiledGraphGpuResourceSlot::new(0, 0);
        let mut external = CompiledGraphGpuResourceRegistry::new();
        external
            .set_texture_view(slot, view)
            .expect("view publication succeeds");
        let mut registry = EmberGpuAssetRegistry::default();
        let handle = EmberAssetHandle(10);
        registry
            .upsert(
                &device,
                &queue,
                handle,
                GraphAssetRef(1),
                EmberAssetRevision::from_value(1),
                compiled_graph_with_texture(),
                &external,
            )
            .expect("compiled graph is materialized");
        registry
            .prepare(&queue, handle, &empty_asset(), &[], &[1], &[])
            .expect("compiled graph inputs are prepared");

        let mut missing_graph = RenderGraph::new(prism_render_driver::Extent3d::new_2d(1, 1));
        assert_eq!(
            registry.append_to_graph(&mut missing_graph, 0, &BTreeMap::new(), &BTreeMap::new()),
            Err(CompiledGraphGraphError::ExternalTextureImportRequired {
                group: slot.group,
                binding: slot.binding,
            })
        );

        let mut graph = RenderGraph::new(prism_render_driver::Extent3d::new_2d(1, 1));
        let imported = graph.import_texture(
            "caller-imported-view",
            prism_render_graph::TextureDesc::color(prism_render_driver::TextureFormat::Rgba8Unorm)
                .with_size(prism_render_graph::SizeClass::absolute_2d(1, 1))
                .with_usage(prism_render_driver::TextureUsages::TEXTURE_BINDING),
            prism_render_graph::ImportedTexture {
                texture,
                default_view: view,
                entry_state: prism_render_driver::TextureState::initial(),
                exit_state: prism_render_driver::TextureState::initial(),
                mip_levels: 1,
                array_layers: 1,
            },
        );
        let texture_handles = BTreeMap::from([(view, imported)]);
        assert_eq!(
            registry.append_to_graph(&mut graph, 0, &BTreeMap::new(), &texture_handles),
            Ok(())
        );
        assert!(graph.compile().is_ok());

        registry.destroy(&device);
        device.destroy_texture_view(view);
        device.destroy_texture(texture);
    }
}

#[cfg(test)]
#[path = "particle/authoring_gpu_tests.rs"]
mod authoring_gpu_tests;
