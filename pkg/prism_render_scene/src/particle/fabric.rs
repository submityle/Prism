//! Bevy host adapter for the native RenderApp Particle publication.
//!
//! This adapter deliberately does not manufacture a buffer from a semantic
//! key. The RenderApp publication carries renderer-owned `BufferId`s; the
//! Bevy host resolves those IDs to the exact live `Buffer`s on the adopted
//! `SceneRhi` device, and this module imports those buffers into the existing
//! SceneFabric runtime together with the caller's real pipelines and groups.

use bevy_ecs::{
    prelude::{Res, ResMut, Resource},
    schedule::IntoScheduleConfigs,
};
use bevy_render::{
    render_resource::{BindGroup, Buffer, ComputePipeline},
    renderer::RenderContext,
};
use prism_particle_render::{
    particle_resource_payload, ParticleFabricScenePublication,
    PARTICLE_ALIVE_RESOURCE_PAYLOAD_ROLE, PARTICLE_STREAM_RESOURCE_PAYLOAD_ROLE,
};
use prism_render_architecture::render_fabric::{
    DomainPayload, FrameRenderItem, RenderDomain, ResourceKey, PARTICLE_ALIVE_RESOURCE,
    PARTICLE_BRIDGE_PASS, PARTICLE_RENDER_PASS, PARTICLE_STREAM_RESOURCE,
};
use prism_render_driver::{BufferId, BufferUsages, Extent3d};
use prism_render_graph::RecordedGraph;
use thiserror::Error;
use wgpu_types::BufferUsages as NativeBufferUsages;

use crate::{
    fabric_runtime::{
        prepare_with_templates, SceneFabricDispatch, SceneFabricPass, SceneFabricRequest,
        SceneFabricResource, SceneFabricRuntime, SceneFabricRuntimeError,
    },
    gpu::rhi::SceneRhi,
};

/// A renderer-owned `BufferId` paired with the exact Bevy buffer resolved by
/// the host on the same native device.
#[derive(Clone, Copy, Debug)]
pub struct ParticleSceneFabricBuffer<'a> {
    pub id: BufferId,
    pub buffer: &'a Buffer,
}

/// One real compute recorder input. Bind groups are created by the host for
/// the resolved Particle buffers; this adapter only imports and records them.
#[derive(Clone, Copy, Debug)]
pub struct ParticleSceneFabricDispatch<'a> {
    pub pipeline: &'a ComputePipeline,
    pub bind_group: &'a BindGroup,
    pub bind_group_index: u32,
    pub workgroups: [u32; 3],
}

/// Inputs for one same-frame Particle bridge/render slice.
pub struct ParticleSceneFabricInput<'a> {
    pub frame: u64,
    pub extent: Extent3d,
    pub publication: &'a ParticleFabricScenePublication,
    pub stream: Option<ParticleSceneFabricBuffer<'a>>,
    pub alive: Option<ParticleSceneFabricBuffer<'a>>,
    pub bridge: ParticleSceneFabricDispatch<'a>,
    pub renderer: ParticleSceneFabricDispatch<'a>,
}

/// A host-owned Particle buffer handle for one publication.
///
/// The producer must pass the real `BufferId` from the same host table as the
/// publication and the exact live Bevy buffer behind it. This type never
/// creates an ID and has no fallback allocation path.
pub struct ParticleSceneFabricBufferOwned {
    pub id: BufferId,
    pub buffer: Buffer,
}

impl ParticleSceneFabricBufferOwned {
    /// Couples a real host-table ID to its exact live Bevy buffer.
    #[must_use]
    pub const fn new(id: BufferId, buffer: Buffer) -> Self {
        Self { id, buffer }
    }
}

/// A host-owned compute dispatch for one Particle Fabric pass.
pub struct ParticleSceneFabricDispatchOwned {
    pub pipeline: ComputePipeline,
    pub bind_group: BindGroup,
    pub bind_group_index: u32,
    pub workgroups: [u32; 3],
}

impl ParticleSceneFabricDispatchOwned {
    /// Couples a real pipeline, bind group and dispatch dimensions.
    #[must_use]
    pub const fn new(
        pipeline: ComputePipeline,
        bind_group: BindGroup,
        bind_group_index: u32,
        workgroups: [u32; 3],
    ) -> Self {
        Self {
            pipeline,
            bind_group,
            bind_group_index,
            workgroups,
        }
    }
}

/// One real host publication handed to the production recorder.
///
/// Empty publications are valid no-work frames and are consumed without
/// requesting a pipeline or manufacturing a resource. Non-empty publications
/// must carry all four physical inputs; missing inputs are an observable
/// production error.
pub struct ParticleSceneFabricHostInput {
    pub extent: Extent3d,
    pub publication: ParticleFabricScenePublication,
    pub stream: Option<ParticleSceneFabricBufferOwned>,
    pub alive: Option<ParticleSceneFabricBufferOwned>,
    pub bridge: Option<ParticleSceneFabricDispatchOwned>,
    pub renderer: Option<ParticleSceneFabricDispatchOwned>,
}

impl ParticleSceneFabricHostInput {
    /// Creates a host publication. Physical inputs remain optional only so an
    /// empty frame can be represented without fake handles; the recorder
    /// rejects any missing input for a non-empty frame.
    #[must_use]
    pub const fn new(
        extent: Extent3d,
        publication: ParticleFabricScenePublication,
        stream: Option<ParticleSceneFabricBufferOwned>,
        alive: Option<ParticleSceneFabricBufferOwned>,
        bridge: Option<ParticleSceneFabricDispatchOwned>,
        renderer: Option<ParticleSceneFabricDispatchOwned>,
    ) -> Self {
        Self {
            extent,
            publication,
            stream,
            alive,
            bridge,
            renderer,
        }
    }

    fn borrowed(&self) -> Result<ParticleSceneFabricInput<'_>, ParticleSceneFabricError> {
        Ok(ParticleSceneFabricInput {
            frame: self.publication.frame(),
            extent: self.extent,
            publication: &self.publication,
            stream: self
                .stream
                .as_ref()
                .map(|buffer| ParticleSceneFabricBuffer {
                    id: buffer.id,
                    buffer: &buffer.buffer,
                }),
            alive: self.alive.as_ref().map(|buffer| ParticleSceneFabricBuffer {
                id: buffer.id,
                buffer: &buffer.buffer,
            }),
            bridge: self
                .bridge
                .as_ref()
                .ok_or(ParticleSceneFabricError::MissingBridgeDispatch)
                .map(|dispatch| ParticleSceneFabricDispatch {
                    pipeline: &dispatch.pipeline,
                    bind_group: &dispatch.bind_group,
                    bind_group_index: dispatch.bind_group_index,
                    workgroups: dispatch.workgroups,
                })?,
            renderer: self
                .renderer
                .as_ref()
                .ok_or(ParticleSceneFabricError::MissingRendererDispatch)
                .map(|dispatch| ParticleSceneFabricDispatch {
                    pipeline: &dispatch.pipeline,
                    bind_group: &dispatch.bind_group,
                    bind_group_index: dispatch.bind_group_index,
                    workgroups: dispatch.workgroups,
                })?,
        })
    }
}

/// Render-world mailbox consumed by the production Core3d recorder.
///
/// The mailbox is one-shot and frame-keyed. A stale publication is discarded
/// with an error, so a successful frame can never replay a previous stream or
/// alive buffer after a producer has stopped publishing.
#[derive(Resource, Default)]
pub struct ParticleSceneFabricHost {
    pending: Option<ParticleSceneFabricHostInput>,
    last_error: Option<ParticleSceneFabricError>,
    last_recorded_frame: Option<u64>,
}

impl ParticleSceneFabricHost {
    /// Publishes one physical publication for the next host frame.
    pub fn publish(
        &mut self,
        input: ParticleSceneFabricHostInput,
    ) -> Result<(), ParticleSceneFabricError> {
        if let Some(pending) = self.pending.as_ref() {
            return Err(ParticleSceneFabricError::PendingPublication {
                frame: pending.publication.frame(),
            });
        }
        self.last_error = None;
        self.pending = Some(input);
        Ok(())
    }

    /// Takes only the publication belonging to the shared Render Fabric frame.
    pub fn take_for_frame(
        &mut self,
        frame: u64,
    ) -> Result<Option<ParticleSceneFabricHostInput>, ParticleSceneFabricError> {
        let Some(input) = self.pending.take() else {
            return Ok(None);
        };
        if input.publication.frame() != frame {
            let error = ParticleSceneFabricError::FrameMismatch {
                expected: frame,
                actual: input.publication.frame(),
            };
            self.last_error = Some(ParticleSceneFabricError::FrameMismatch {
                expected: frame,
                actual: input.publication.frame(),
            });
            return Err(error);
        }
        Ok(Some(input))
    }

    /// Records the frame whose command buffer was handed to the host context.
    fn record_success(&mut self, frame: u64) {
        self.last_error = None;
        self.last_recorded_frame = Some(frame);
    }

    /// Records a rejected production frame.
    fn record_error(&mut self, error: ParticleSceneFabricError) {
        self.last_error = Some(error);
        self.last_recorded_frame = None;
    }

    /// Last production error, if the most recent publication was rejected.
    #[must_use]
    pub fn last_error(&self) -> Option<&ParticleSceneFabricError> {
        self.last_error.as_ref()
    }

    /// Last frame handed to the shared recorder and host context.
    #[must_use]
    pub const fn last_recorded_frame(&self) -> Option<u64> {
        self.last_recorded_frame
    }
}

/// Host-owned publication slot with explicit error clearing. A failed frame
/// can never leave the previous frame's Particle publication available for
/// accidental reuse.
#[derive(Default)]
pub struct ParticleSceneFabricSlot {
    publication: Option<ParticleFabricScenePublication>,
}

impl ParticleSceneFabricSlot {
    pub fn publish(
        &mut self,
        result: Result<ParticleFabricScenePublication, ParticleSceneFabricError>,
    ) -> Result<(), ParticleSceneFabricError> {
        match result {
            Ok(publication) => {
                self.publication = Some(publication);
                Ok(())
            }
            Err(error) => {
                self.publication = None;
                Err(error)
            }
        }
    }

    pub fn clear(&mut self) {
        self.publication = None;
    }

    pub fn take_for_frame(&mut self, frame: u64) -> Option<ParticleFabricScenePublication> {
        let publication = self.publication.take()?;
        (publication.frame() == frame).then_some(publication)
    }

    pub fn is_empty(&self) -> bool {
        self.publication.is_none()
    }
}

/// Errors raised before any SceneRhi import is allowed, or by the authoritative
/// SceneFabric compiler/runtime after real imports have been created.
#[derive(Debug, Error)]
pub enum ParticleSceneFabricError {
    #[error("Particle host frame {actual} does not match publication frame {expected}")]
    FrameMismatch { expected: u64, actual: u64 },
    #[error("Particle publication has no stream buffer")]
    MissingStream,
    #[error("Particle publication has no alive buffer")]
    MissingAlive,
    #[error("Particle publication has no bridge dispatch")]
    MissingBridgeDispatch,
    #[error("Particle publication has no renderer dispatch")]
    MissingRendererDispatch,
    #[error("Particle host already has an unconsumed publication for frame {frame}")]
    PendingPublication { frame: u64 },
    #[error("Particle stream BufferId {actual:?} does not match publication {expected:?}")]
    StreamIdMismatch {
        expected: BufferId,
        actual: BufferId,
    },
    #[error("Particle alive BufferId {actual:?} does not match publication {expected:?}")]
    AliveIdMismatch {
        expected: BufferId,
        actual: BufferId,
    },
    #[error(
        "Particle {resource:?} host buffer is {actual} bytes, publication requires {required}"
    )]
    BufferTooSmall {
        resource: ResourceKey,
        required: u64,
        actual: u64,
    },
    #[error("Particle {resource:?} host buffer usage {actual:?} misses required {required:?}")]
    BufferUsageMismatch {
        resource: ResourceKey,
        required: NativeBufferUsages,
        actual: NativeBufferUsages,
    },
    #[error("Particle semantic publication is invalid: {0}")]
    SemanticMismatch(&'static str),
    #[error(transparent)]
    Runtime(#[from] SceneFabricRuntimeError),
}

/// Imports the publication's real stream/alive buffers and records the two
/// frozen Particle compute passes in bridge -> render order.
pub(crate) fn prepare_particle_fabric<'a>(
    scene_fabric: &crate::gpu::fabric::SceneFabric,
    rhi: &SceneRhi,
    input: ParticleSceneFabricInput<'a>,
) -> Result<Option<SceneFabricRuntime<'a>>, ParticleSceneFabricError> {
    let publication = input.publication;
    if publication.is_empty() {
        return Ok(None);
    }
    if input.frame != publication.frame() {
        return Err(ParticleSceneFabricError::FrameMismatch {
            expected: publication.frame(),
            actual: input.frame,
        });
    }

    validate_items(publication.items())?;
    validate_draw_bindings(publication)?;

    let stream = input
        .stream
        .ok_or(ParticleSceneFabricError::MissingStream)?;
    let alive = input.alive.ok_or(ParticleSceneFabricError::MissingAlive)?;
    let expected_stream = publication.draws()[0].stream;
    let expected_alive = publication.draws()[0].alive;
    validate_host_buffer(PARTICLE_STREAM_RESOURCE, expected_stream, stream)?;
    validate_host_buffer(PARTICLE_ALIVE_RESOURCE, expected_alive, alive)?;

    let resources = vec![
        SceneFabricResource::ImportedBuffer {
            key: PARTICLE_STREAM_RESOURCE,
            frame: publication.frame(),
            version: expected_stream.version,
            buffer: stream.buffer,
            offset: 0,
            entry_state: expected_stream.entry_state,
        },
        SceneFabricResource::ImportedBuffer {
            key: PARTICLE_ALIVE_RESOURCE,
            frame: publication.frame(),
            version: expected_alive.version,
            buffer: alive.buffer,
            offset: 0,
            entry_state: expected_alive.entry_state,
        },
    ];
    let recorders = vec![
        SceneFabricPass::Compute {
            pass: PARTICLE_BRIDGE_PASS,
            dispatches: vec![SceneFabricDispatch {
                pipeline: input.bridge.pipeline,
                bind_group: input.bridge.bind_group,
                bind_group_index: input.bridge.bind_group_index,
                workgroups: input.bridge.workgroups,
                push_constants: None,
            }],
        },
        SceneFabricPass::Compute {
            pass: PARTICLE_RENDER_PASS,
            dispatches: vec![SceneFabricDispatch {
                pipeline: input.renderer.pipeline,
                bind_group: input.renderer.bind_group,
                bind_group_index: input.renderer.bind_group_index,
                workgroups: input.renderer.workgroups,
                push_constants: None,
            }],
        },
    ];
    let request = SceneFabricRequest {
        frame: publication.frame(),
        extent: input.extent,
        items: publication.items(),
        active_passes: &[PARTICLE_BRIDGE_PASS, PARTICLE_RENDER_PASS],
        fulfilled_dependencies: &[],
        resources,
        recorders,
    };
    Ok(Some(prepare_with_templates(
        scene_fabric,
        rhi,
        request,
        &[],
    )?))
}

/// Production Bevy host entry point: import the already-authenticated native
/// publication, then hand the prepared graph to the host render context. The
/// context owns submission; the runtime's completion callback retains imported
/// leases until the native command buffer completes.
pub(crate) fn record_particle_fabric<'a>(
    scene_fabric: &crate::gpu::fabric::SceneFabric,
    rhi: &SceneRhi,
    input: ParticleSceneFabricInput<'a>,
    context: &mut RenderContext,
) -> Result<Option<RecordedGraph>, ParticleSceneFabricError> {
    let Some(runtime) = prepare_particle_fabric(scene_fabric, rhi, input)? else {
        return Ok(None);
    };
    Ok(Some(runtime.record_into(rhi, context)))
}

/// Production Core3d call site for Particle Fabric.
///
/// The system consumes the same shared frame number used by Surface and the
/// other Fabric domains, adopts the already-initialized host `SceneRhi`, and
/// delegates recording to [`record_particle_fabric`]. `record_into` attaches
/// the real completion/retirement handoff to the command buffer submitted by
/// the host `RenderContext`; this system does not submit a second queue or
/// release imported resources early.
pub(crate) fn record_particle_fabric_system(
    mut host: ResMut<ParticleSceneFabricHost>,
    frame: Res<crate::fabric_frame::RenderFabricFrame>,
    scene_fabric: Res<crate::gpu::fabric::SceneFabric>,
    rhi: Res<SceneRhi>,
    mut context: RenderContext,
) {
    let Some(input) = (match host.take_for_frame(frame.frame()) {
        Ok(input) => input,
        Err(error) => {
            host.record_error(error);
            return;
        }
    }) else {
        return;
    };

    if input.publication.is_empty() {
        host.record_success(input.publication.frame());
        return;
    }

    let frame_number = input.publication.frame();
    let borrowed = match input.borrowed() {
        Ok(input) => input,
        Err(error) => {
            host.record_error(error);
            return;
        }
    };
    match record_particle_fabric(scene_fabric.as_ref(), rhi.as_ref(), borrowed, &mut context) {
        Ok(Some(_recorded)) => host.record_success(frame_number),
        Ok(None) => host.record_error(ParticleSceneFabricError::SemanticMismatch(
            "non-empty Particle publication produced no recorded graph",
        )),
        Err(error) => host.record_error(error),
    }
}

/// Installs the Particle recorder into the host's real `RenderApp`.
///
/// The caller must publish physical inputs through [`ParticleSceneFabricHost`]
/// before the matching frame reaches `Core3d`. Device/table creation remains
/// owned by `init_scene_rhi`; this helper only installs the consumer and never
/// allocates a replacement device or resource handle.
pub fn install_particle_fabric(render_app: &mut bevy_app::SubApp) {
    render_app.init_resource::<ParticleSceneFabricHost>();
    render_app.add_systems(
        bevy_core_pipeline::Core3d,
        record_particle_fabric_system.before(bevy_core_pipeline::Core3dSystems::MainPass),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_input(frame: u64) -> ParticleSceneFabricHostInput {
        ParticleSceneFabricHostInput::new(
            Extent3d::new_2d(1, 1),
            ParticleFabricScenePublication::new(frame, Vec::new(), Vec::new()),
            None,
            None,
            None,
            None,
        )
    }

    #[test]
    fn host_consumes_only_the_shared_frame_and_clears_stale_publication() {
        let mut host = ParticleSceneFabricHost::default();
        host.publish(empty_input(7))
            .expect("first publication accepted");

        let error = host
            .take_for_frame(8)
            .expect_err("stale publication must be rejected");
        assert!(matches!(
            error,
            ParticleSceneFabricError::FrameMismatch {
                expected: 8,
                actual: 7
            }
        ));
        assert_eq!(host.last_error(), Some(&error));
        assert!(host.take_for_frame(7).unwrap().is_none());
    }

    #[test]
    fn host_rejects_a_second_unconsumed_publication_without_replacing_the_first() {
        let mut host = ParticleSceneFabricHost::default();
        host.publish(empty_input(3))
            .expect("first publication accepted");

        let error = host
            .publish(empty_input(4))
            .expect_err("mailbox must be one-shot");
        assert!(matches!(
            error,
            ParticleSceneFabricError::PendingPublication { frame: 3 }
        ));
        assert!(host.take_for_frame(3).unwrap().is_some());
    }

    #[test]
    fn non_empty_host_input_requires_every_real_physical_binding() {
        let publication = ParticleFabricScenePublication::new(
            11,
            vec![FrameRenderItem {
                object_id: RenderObjectId::Particle(DomainObjectHandle::new(1, 1)),
                domain: RenderDomain::Particle,
                history_key: 0,
                motion_source: prism_render_architecture::motion::MotionSource::Particle,
                payload: DomainPayload::Resource(
                    particle_resource_payload(
                        PARTICLE_STREAM_RESOURCE,
                        PARTICLE_STREAM_RESOURCE_PAYLOAD_ROLE,
                    )
                    .expect("stream template"),
                ),
            }],
            vec![ParticleFabricGpuDraw {
                object_id: RenderObjectId::Particle(DomainObjectHandle::new(1, 1)),
                renderer_index: 0,
                history_key: 0,
                bounds_min: [-1.0; 3],
                bounds_max: [1.0; 3],
                geometry:
                    prism_render_architecture::particle::indirect_draw::RendererGeometry::Sprite {
                        alive_count: 1,
                        first_instance: 0,
                    },
                stream: ParticleFabricGpuBuffer {
                    semantic: particle_resource_payload(
                        PARTICLE_STREAM_RESOURCE,
                        PARTICLE_STREAM_RESOURCE_PAYLOAD_ROLE,
                    )
                    .expect("stream template"),
                    buffer: BufferId::from_parts(1, 1),
                    version: 1,
                    size: 256,
                    usage: BufferUsages::STORAGE,
                    entry_state: BufferState::initial(),
                },
                alive: ParticleFabricGpuBuffer {
                    semantic: particle_resource_payload(
                        PARTICLE_ALIVE_RESOURCE,
                        PARTICLE_ALIVE_RESOURCE_PAYLOAD_ROLE,
                    )
                    .expect("alive template"),
                    buffer: BufferId::from_parts(2, 1),
                    version: 1,
                    size: 256,
                    usage: BufferUsages::STORAGE,
                    entry_state: BufferState::initial(),
                },
            }],
        );
        let host = ParticleSceneFabricHostInput::new(
            Extent3d::new_2d(1, 1),
            publication,
            None,
            None,
            None,
            None,
        );
        assert!(matches!(
            host.borrowed(),
            Err(ParticleSceneFabricError::MissingBridgeDispatch)
        ));
    }
}

fn validate_items(items: &[FrameRenderItem]) -> Result<(), ParticleSceneFabricError> {
    if items.is_empty() {
        return Err(ParticleSceneFabricError::SemanticMismatch(
            "non-empty publication has no semantic items",
        ));
    }
    for item in items {
        if item.domain != RenderDomain::Particle {
            return Err(ParticleSceneFabricError::SemanticMismatch(
                "publication contains a non-Particle item",
            ));
        }
        match item.payload {
            DomainPayload::Value(_) => {}
            DomainPayload::Resource(payload) => {
                let expected_role = match payload.resource {
                    PARTICLE_STREAM_RESOURCE => PARTICLE_STREAM_RESOURCE_PAYLOAD_ROLE,
                    PARTICLE_ALIVE_RESOURCE => PARTICLE_ALIVE_RESOURCE_PAYLOAD_ROLE,
                    _ => {
                        return Err(ParticleSceneFabricError::SemanticMismatch(
                            "publication contains an unexpected resource key",
                        ));
                    }
                };
                let expected = particle_resource_payload(payload.resource, expected_role).ok_or(
                    ParticleSceneFabricError::SemanticMismatch(
                        "Particle catalog resource template is unavailable",
                    ),
                )?;
                if payload != expected {
                    return Err(ParticleSceneFabricError::SemanticMismatch(
                        "publication resource payload differs from the frozen Particle template",
                    ));
                }
            }
        }
    }
    Ok(())
}

fn validate_draw_bindings(
    publication: &ParticleFabricScenePublication,
) -> Result<(), ParticleSceneFabricError> {
    let first = publication
        .draws()
        .first()
        .ok_or(ParticleSceneFabricError::SemanticMismatch(
            "publication has no draw bindings",
        ))?;
    for draw in publication.draws() {
        if draw.stream.semantic
            != particle_resource_payload(
                PARTICLE_STREAM_RESOURCE,
                PARTICLE_STREAM_RESOURCE_PAYLOAD_ROLE,
            )
            .expect("built-in Particle stream template")
            || draw.alive.semantic
                != particle_resource_payload(
                    PARTICLE_ALIVE_RESOURCE,
                    PARTICLE_ALIVE_RESOURCE_PAYLOAD_ROLE,
                )
                .expect("built-in Particle alive template")
        {
            return Err(ParticleSceneFabricError::SemanticMismatch(
                "draw binding uses the wrong Particle semantic role",
            ));
        }
        if draw.stream.buffer != first.stream.buffer || draw.stream.version != first.stream.version
        {
            return Err(ParticleSceneFabricError::StreamIdMismatch {
                expected: first.stream.buffer,
                actual: draw.stream.buffer,
            });
        }
        if draw.alive.buffer != first.alive.buffer || draw.alive.version != first.alive.version {
            return Err(ParticleSceneFabricError::AliveIdMismatch {
                expected: first.alive.buffer,
                actual: draw.alive.buffer,
            });
        }
    }
    Ok(())
}

fn validate_host_buffer(
    resource: ResourceKey,
    expected: prism_particle_render::ParticleFabricGpuBuffer,
    actual: ParticleSceneFabricBuffer<'_>,
) -> Result<(), ParticleSceneFabricError> {
    if actual.id != expected.buffer || actual.id.generation() != expected.version {
        return Err(if resource == PARTICLE_STREAM_RESOURCE {
            ParticleSceneFabricError::StreamIdMismatch {
                expected: expected.buffer,
                actual: actual.id,
            }
        } else {
            ParticleSceneFabricError::AliveIdMismatch {
                expected: expected.buffer,
                actual: actual.id,
            }
        });
    }
    if actual.buffer.size() < expected.size {
        return Err(ParticleSceneFabricError::BufferTooSmall {
            resource,
            required: expected.size,
            actual: actual.buffer.size(),
        });
    }
    let required = native_usage(expected.usage);
    let actual_usage = actual.buffer.usage();
    if !actual_usage.contains(required) {
        return Err(ParticleSceneFabricError::BufferUsageMismatch {
            resource,
            required,
            actual: actual_usage,
        });
    }
    Ok(())
}

fn native_usage(usage: BufferUsages) -> NativeBufferUsages {
    let mut native = NativeBufferUsages::empty();
    if usage.contains(BufferUsages::COPY_SRC) {
        native |= NativeBufferUsages::COPY_SRC;
    }
    if usage.contains(BufferUsages::COPY_DST) {
        native |= NativeBufferUsages::COPY_DST;
    }
    if usage.contains(BufferUsages::STORAGE) {
        native |= NativeBufferUsages::STORAGE;
    }
    if usage.contains(BufferUsages::INDIRECT) {
        native |= NativeBufferUsages::INDIRECT;
    }
    native
}
