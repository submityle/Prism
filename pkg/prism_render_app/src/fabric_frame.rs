//! Native render-world aggregation and Render Fabric lowering.

use alloc::{collections::BTreeSet, string::String, sync::Arc, vec::Vec};
use std::sync::Mutex;

use prism_ecs::prelude::{Resource, World};
use prism_particle_render::ParticleFabricError;
use prism_render_architecture::{
    gpu_scene::CpuRenderScene,
    motion::MotionSource,
    render_fabric::{
        CompiledFeatureGraph, DeviceFabric, DomainPayload, FrameRenderItem, FrameResourceBinding,
        GraphLimits, ObjectResourceBinding, ObjectRuntimeImport, PassKey, PayloadRole,
        PayloadValue, QueueClass, RenderDomain, RenderFabricFrame, RenderFabricFrameError,
        RenderObjectId, ResourceTemplate, RuntimeGraphLowering, RuntimeGraphResourceMapping,
        RuntimeImport, RuntimePassBinding, RuntimePassContext, ValuePayload, SURFACE_IDENTITY_PASS,
    },
};
use prism_render_driver::{FrameSync, Instance};
use prism_render_graph::{BufferHandle, RenderGraph, TextureHandle};

use crate::{
    particle::ParticleFabricConsumptionError, runtime_resource::RenderRuntimeResource,
    visibility::PrismRenderScene, PreparedParticleFabricFrame, RenderRecord,
};

/// One exact host-graph resource used by the native Surface identity pass.
/// The physical binding and graph-local handle are captured together so the
/// Fabric compiler cannot be pointed at a different generation accidentally.
#[derive(Clone, Copy)]
pub(crate) struct NativeSurfaceIdentityResource {
    pub key: prism_render_architecture::render_fabric::ResourceKey,
    pub layout: prism_render_architecture::render_fabric::ResourceLayout,
    pub alignment: u64,
    pub physical: prism_render_architecture::render_fabric::PhysicalResource,
    pub handle: NativeSurfaceIdentityHandle,
}

#[derive(Clone, Copy)]
pub(crate) enum NativeSurfaceIdentityHandle {
    Buffer(BufferHandle),
    Texture(TextureHandle),
}

pub(crate) struct NativeSurfaceIdentityInput {
    pub resources: Vec<NativeSurfaceIdentityResource>,
    pub pipeline: prism_render_driver::ComputePipelineId,
    pub bind_group: prism_render_driver::BindGroupId,
    pub workgroups: [u32; 3],
}

/// A single physical phase recorder destined for one compiled Fabric pass.
///
/// The recorder is deliberately limited to [`RuntimePassContext`]. It cannot
/// create a graph, import a resource, or replace the compiled access list. A
/// physical phase must therefore have a real catalog pass identity; lowering
/// rejects attempts to fan several physical phases into one semantic pass.
pub(crate) struct NativeFabricPhase {
    pass: PassKey,
    queue: QueueClass,
    record: NativeFabricPhaseRecorder,
}

type NativeFabricPhaseRecorder = Box<
    dyn for<'a, 'graph> FnOnce(
            &mut RuntimePassContext<'a, 'graph>,
        ) -> Result<(), NativeFabricPhaseRecordError>
        + Send,
>;

impl NativeFabricPhase {
    pub(crate) fn new<F>(pass: PassKey, queue: QueueClass, record: F) -> Self
    where
        F: for<'a, 'graph> FnOnce(
                &mut RuntimePassContext<'a, 'graph>,
            ) -> Result<(), NativeFabricPhaseRecordError>
            + Send
            + 'static,
    {
        Self {
            pass,
            queue,
            record: Box::new(record),
        }
    }

    #[cfg(test)]
    fn identity(pass: PassKey, queue: QueueClass) -> Self {
        Self::new(pass, queue, move |context| {
            if context.pass().key() != pass {
                return Err(NativeFabricPhaseRecordError::Rejected {
                    pass,
                    phase: 0,
                    reason: String::from("runtime pass identity changed during recording"),
                });
            }
            context.commands().dispatch(1, 1, 1);
            Ok(())
        })
    }
}

/// A recorder rejected by a runtime context after graph admission.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum NativeFabricPhaseRecordError {
    Rejected {
        pass: PassKey,
        phase: usize,
        reason: String,
    },
}

/// Structural errors detected before any phase recorder is attached to a host
/// graph. These errors are intentionally strict: a missing or duplicated
/// physical phase would otherwise turn into an incomplete GPU frame.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NativeFabricPhaseLoweringError {
    EmptyPlan,
    MissingPass(PassKey),
    UnexpectedPass(PassKey),
    DuplicatePass(PassKey),
    PassOrderMismatch {
        index: usize,
        expected: PassKey,
        actual: PassKey,
    },
    QueueMismatch {
        pass: PassKey,
        expected: QueueClass,
        actual: QueueClass,
    },
}

/// Runtime diagnostics for phase recorders. The existing graph runtime uses a
/// move-only `FnOnce` callback without a `Result` return, so recorder failures
/// are retained here and must be checked before host submission.
#[derive(Clone, Default)]
pub(crate) struct NativeFabricPhaseMonitor {
    errors: Arc<Mutex<Vec<NativeFabricPhaseRecordError>>>,
}

impl NativeFabricPhaseMonitor {
    fn push(&self, error: NativeFabricPhaseRecordError) {
        self.errors.lock().expect("phase monitor lock").push(error);
    }

    pub(crate) fn errors(&self) -> Vec<NativeFabricPhaseRecordError> {
        self.errors.lock().expect("phase monitor lock").clone()
    }

    pub(crate) fn is_clean(&self) -> bool {
        self.errors.lock().expect("phase monitor lock").is_empty()
    }
}

/// Validated phase-to-pass lowering for an existing RenderGraph.
pub(crate) struct NativeFabricPhaseLowering {
    recorders: Vec<RuntimePassBinding>,
    monitor: NativeFabricPhaseMonitor,
}

impl NativeFabricPhaseLowering {
    pub(crate) fn new(
        graph: &CompiledFeatureGraph,
        phases: Vec<NativeFabricPhase>,
    ) -> Result<Self, NativeFabricPhaseLoweringError> {
        if phases.is_empty() {
            return Err(NativeFabricPhaseLoweringError::EmptyPlan);
        }

        let expected: Vec<_> = graph.passes().iter().map(|pass| pass.key()).collect();
        let actual: Vec<_> = phases.iter().map(|phase| phase.pass).collect();
        let mut seen = BTreeSet::new();
        for &pass in &actual {
            if !seen.insert(pass) {
                return Err(NativeFabricPhaseLoweringError::DuplicatePass(pass));
            }
            if !expected.contains(&pass) {
                return Err(NativeFabricPhaseLoweringError::UnexpectedPass(pass));
            }
        }
        for &pass in &expected {
            if !seen.contains(&pass) {
                return Err(NativeFabricPhaseLoweringError::MissingPass(pass));
            }
        }
        if expected.len() != actual.len() {
            // The missing/unexpected checks above provide the stable diagnostic;
            // this guard keeps the zip below total even if the graph API grows.
            return Err(NativeFabricPhaseLoweringError::EmptyPlan);
        }
        for (index, (pass, phase)) in graph.passes().iter().zip(&phases).enumerate() {
            if pass.key() != phase.pass {
                return Err(NativeFabricPhaseLoweringError::PassOrderMismatch {
                    index,
                    expected: pass.key(),
                    actual: phase.pass,
                });
            }
            if pass.template().queue != phase.queue {
                return Err(NativeFabricPhaseLoweringError::QueueMismatch {
                    pass: pass.key(),
                    expected: pass.template().queue,
                    actual: phase.queue,
                });
            }
        }

        let monitor = NativeFabricPhaseMonitor::default();
        let mut recorders = Vec::with_capacity(phases.len());
        for (phase_index, phase) in phases.into_iter().enumerate() {
            let pass = phase.pass;
            let monitor = monitor.clone();
            recorders.push(RuntimePassBinding::new(pass, move |context| {
                if let Err(error) = (phase.record)(context) {
                    monitor.push(error);
                }
            }));
            // Keep the index in the lowering loop so adding a phase-specific
            // diagnostic cannot silently lose the physical order contract.
            debug_assert_eq!(phase_index, recorders.len() - 1);
        }
        Ok(Self { recorders, monitor })
    }

    pub(crate) fn into_parts(self) -> (Vec<RuntimePassBinding>, NativeFabricPhaseMonitor) {
        (self.recorders, self.monitor)
    }
}

/// Device-generation-frozen Render Fabric selection used by the native frame
/// path. The catalog is built once per runtime generation, not once per frame.
pub(crate) struct NativeRenderFabricDevice {
    generation: u64,
    device: DeviceFabric,
}

impl Resource for NativeRenderFabricDevice {}

impl NativeRenderFabricDevice {
    pub(crate) fn new(generation: u64) -> Self {
        let catalog = prism_render_architecture::render_fabric::builtin_catalog()
            .expect("built-in Render Fabric catalog is valid");
        let roots = prism_render_architecture::render_fabric::builtin_roots(&catalog)
            .expect("built-in Render Fabric roots are valid");
        let device = DeviceFabric::new(
            catalog,
            roots,
            prism_render_architecture::backend::capability::CapabilitySet::WEBGPU_BASELINE,
            prism_render_architecture::render_fabric::RenderQualityTier::Low,
        )
        .expect("built-in Render Fabric selection is valid");
        Self { generation, device }
    }

    #[cfg(test)]
    fn generation(&self) -> u64 {
        self.generation
    }
}

pub(crate) fn ensure_native_render_fabric<I>(world: &mut World)
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: FrameSync + Send + Sync + 'static,
    I::Surface: Send + Sync + 'static,
{
    let generation = world
        .resource::<RenderRuntimeResource<I>>()
        .runtime()
        .device_generation();
    let replace = world
        .get_resource::<NativeRenderFabricDevice>()
        .is_none_or(|fabric| fabric.generation != generation);
    if replace {
        world.insert_resource(NativeRenderFabricDevice::new(generation));
    }
}

/// Shared semantic frame: extracted Surface and GPU-prepared Particle metadata.
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct ExtractedRenderFabricFrame {
    frame: RenderFabricFrame,
    error: Option<NativeFabricFrameError>,
}

/// A domain could not contribute a valid frame; stale payloads are never kept.
#[derive(Clone, Debug, PartialEq)]
pub enum NativeFabricFrameError {
    /// Particle snapshot validation failed.
    Particle(ParticleFabricError),
    /// Particle semantic payloads could not produce valid geometry.
    ParticleConsumption(ParticleFabricConsumptionError),
    /// Particle extraction or GPU preparation published another render frame.
    ParticleFrameMismatch { expected: u64, actual: u64 },
    /// A Surface or domain item failed shared semantic validation.
    InvalidFrame(RenderFabricFrameError),
}

/// Why the shared CPU frame cannot enter render graph recording.
#[derive(Clone, Debug, PartialEq)]
pub enum NativeFabricFrameAdmissionError {
    /// Extraction did not publish a shared frame.
    MissingFrame,
    /// A domain rejected the current extraction.
    InvalidFrame(NativeFabricFrameError),
    /// The frame belongs to another render attempt.
    FrameMismatch { expected: u64, actual: u64 },
    /// The shared frame failed the authoritative Render Fabric compiler.
    Compile(prism_render_architecture::render_fabric::FabricCompileError),
    /// The compiled frame could not be imported or recorder-admitted.
    Runtime(prism_render_architecture::render_fabric::FabricRuntimeError),
    /// Physical phase coverage could not be matched one-to-one with the
    /// compiled Fabric pass table.
    PhaseLowering(NativeFabricPhaseLoweringError),
}

pub(crate) fn admit_render_fabric_frame(
    extracted: Option<&ExtractedRenderFabricFrame>,
    expected: u64,
) -> Result<(), NativeFabricFrameAdmissionError> {
    let extracted = extracted.ok_or(NativeFabricFrameAdmissionError::MissingFrame)?;
    if let Some(error) = extracted.error() {
        return Err(NativeFabricFrameAdmissionError::InvalidFrame(error.clone()));
    }
    let actual = extracted.frame().frame();
    if actual != expected {
        return Err(NativeFabricFrameAdmissionError::FrameMismatch { expected, actual });
    }
    Ok(())
}

impl ExtractedRenderFabricFrame {
    /// Validated frame, empty on extraction failure.
    #[must_use]
    pub const fn frame(&self) -> &RenderFabricFrame {
        &self.frame
    }

    /// Explicit extraction failure for this frame.
    #[must_use]
    pub const fn error(&self) -> Option<&NativeFabricFrameError> {
        self.error.as_ref()
    }

    /// Compiles this exact native frame through the device-frozen Fabric
    /// selection. The compiled resource IDs are then used to authenticate
    /// physical imports in [`Self::prepare_compiled_native_fabric`].
    pub fn compile_native_fabric(
        &self,
        expected: u64,
        device: &DeviceFabric,
        templates: &[ResourceTemplate],
        bindings: &[FrameResourceBinding],
        active_passes: &[PassKey],
        fulfilled_dependencies: &[PassKey],
        limits: GraphLimits,
    ) -> Result<
        prism_render_architecture::render_fabric::CompiledFeatureGraph,
        NativeFabricFrameAdmissionError,
    > {
        admit_render_fabric_frame(Some(self), expected)?;
        device
            .compile_active(
                templates,
                expected,
                bindings,
                self.frame.items(),
                active_passes,
                fulfilled_dependencies,
                limits,
            )
            .map_err(NativeFabricFrameAdmissionError::Compile)
    }

    /// Compiles an active native slice with explicit object-scoped bindings.
    /// Shared and object resources stay separate so an object import can never
    /// be satisfied by a shared buffer or another object's range.
    pub fn compile_native_fabric_instanced(
        &self,
        expected: u64,
        device: &DeviceFabric,
        templates: &[ResourceTemplate],
        shared_bindings: &[FrameResourceBinding],
        object_bindings: &[ObjectResourceBinding],
        active_passes: &[PassKey],
        fulfilled_dependencies: &[PassKey],
        limits: GraphLimits,
    ) -> Result<CompiledFeatureGraph, NativeFabricFrameAdmissionError> {
        admit_render_fabric_frame(Some(self), expected)?;
        device
            .compile_active_instanced(
                templates,
                prism_render_architecture::render_fabric::builtin_object_resource_accesses(),
                expected,
                shared_bindings,
                object_bindings,
                self.frame.items(),
                active_passes,
                fulfilled_dependencies,
                limits,
            )
            .map_err(NativeFabricFrameAdmissionError::Compile)
    }

    /// Lower a compiled Fabric declaration into the caller's existing graph.
    ///
    /// This is the only Particle phase adapter entrypoint: it never creates a
    /// `RenderGraph`, never imports a physical resource twice, and requires one
    /// real recorder for every compiled pass. The host must prepare the graph
    /// after all frame domains have appended their lowerings and must inspect
    /// the returned monitor before submission.
    pub(crate) fn append_native_fabric_phases(
        graph: CompiledFeatureGraph,
        host_graph: &mut RenderGraph,
        shared_imports: &[RuntimeImport],
        object_imports: &[ObjectRuntimeImport],
        resource_mappings: &[RuntimeGraphResourceMapping],
        phases: Vec<NativeFabricPhase>,
    ) -> Result<(RuntimeGraphLowering, NativeFabricPhaseMonitor), NativeFabricFrameAdmissionError>
    {
        let phase_lowering = NativeFabricPhaseLowering::new(&graph, phases)
            .map_err(NativeFabricFrameAdmissionError::PhaseLowering)?;
        let (recorders, monitor) = phase_lowering.into_parts();
        let lowering = graph
            .append_to_graph_with_resource_mappings(
                host_graph,
                shared_imports,
                object_imports,
                resource_mappings,
                recorders,
            )
            .map_err(NativeFabricFrameAdmissionError::Runtime)?;
        Ok((lowering, monitor))
    }

    /// Append the real Surface identity pass to a host-owned graph. Every
    /// resource is mapped to the host's existing graph-local import; no second
    /// graph, physical re-import, or synthetic recorder is allowed here.
    pub(crate) fn append_native_surface_identity(
        &self,
        expected: u64,
        fabric: &mut NativeRenderFabricDevice,
        graph: &mut RenderGraph,
        input: NativeSurfaceIdentityInput,
    ) -> Result<
        prism_render_architecture::render_fabric::RuntimeGraphOutput,
        NativeFabricFrameAdmissionError,
    > {
        if input.resources.is_empty() {
            return Err(NativeFabricFrameAdmissionError::Runtime(
                prism_render_architecture::render_fabric::FabricRuntimeError::MissingImport(
                    prism_render_graph::frame_graph::ResourceId(u32::MAX),
                ),
            ));
        }
        let bindings: Vec<_> = input
            .resources
            .iter()
            .map(|resource| FrameResourceBinding {
                key: resource.key,
                frame: expected,
                version: 0,
                physical: resource.physical,
                layout: resource.layout,
                alignment: resource.alignment,
            })
            .collect();
        let overrides: Vec<_> = input
            .resources
            .iter()
            .map(|resource| ResourceTemplate {
                key: resource.key,
                lifetime: prism_render_graph::frame_graph::ResourceLifetime::Persistent,
                layout: resource.layout,
                alignment: resource.alignment,
            })
            .collect();
        let compiled = self.compile_native_fabric_with_overrides(
            expected,
            &fabric.device,
            &bindings,
            &overrides,
            &[SURFACE_IDENTITY_PASS],
            &[],
            GraphLimits::default(),
        )?;
        let mappings = input
            .resources
            .iter()
            .map(|resource| {
                let id = compiled
                    .resource_id(resource.key, expected)
                    .expect("Surface identity resource must be compiled");
                match resource.handle {
                    NativeSurfaceIdentityHandle::Buffer(handle) => {
                        RuntimeGraphResourceMapping::Buffer {
                            resource: id,
                            handle,
                        }
                    }
                    NativeSurfaceIdentityHandle::Texture(handle) => {
                        RuntimeGraphResourceMapping::Texture {
                            resource: id,
                            handle,
                        }
                    }
                }
            })
            .collect::<Vec<_>>();
        let identity_resource = compiled
            .resource_id(
                prism_render_architecture::render_fabric::SURFACE_IDENTITY_RESOURCE,
                expected,
            )
            .expect("Surface identity output resource must be compiled");
        let pipeline = input.pipeline;
        let bind_group = input.bind_group;
        let workgroups = input.workgroups;
        let lowering = compiled
            .append_to_graph_with_resource_mappings(
                graph,
                &[],
                &[],
                &mappings,
                vec![RuntimePassBinding::new(
                    SURFACE_IDENTITY_PASS,
                    move |context| {
                        context
                            .commands()
                            .set_compute_pipeline(pipeline)
                            .set_compute_bind_group(0, bind_group, Vec::new())
                            .dispatch(workgroups[0], workgroups[1], workgroups[2]);
                    },
                )],
            )
            .map_err(NativeFabricFrameAdmissionError::Runtime)?;
        if lowering.pass_count() != 1 {
            return Err(NativeFabricFrameAdmissionError::Runtime(
                prism_render_architecture::render_fabric::FabricRuntimeError::PlanMismatch,
            ));
        }
        lowering.output(identity_resource).ok_or_else(|| {
            NativeFabricFrameAdmissionError::Runtime(
                prism_render_architecture::render_fabric::FabricRuntimeError::PlanMismatch,
            )
        })
    }

    fn compile_native_fabric_with_overrides(
        &self,
        expected: u64,
        device: &DeviceFabric,
        bindings: &[FrameResourceBinding],
        overrides: &[ResourceTemplate],
        active_passes: &[PassKey],
        fulfilled_dependencies: &[PassKey],
        limits: GraphLimits,
    ) -> Result<
        prism_render_architecture::render_fabric::CompiledFeatureGraph,
        NativeFabricFrameAdmissionError,
    > {
        admit_render_fabric_frame(Some(self), expected)?;
        device
            .compile_active_with_template_overrides(
                prism_render_architecture::render_fabric::builtin_resource_templates(),
                overrides,
                expected,
                bindings,
                self.frame.items(),
                active_passes,
                fulfilled_dependencies,
                limits,
            )
            .map_err(NativeFabricFrameAdmissionError::Compile)
    }

    fn replace(
        &mut self,
        frame: u64,
        scene: Option<&CpuRenderScene>,
        particle: Option<&PreparedParticleFabricFrame>,
    ) {
        self.error = None;
        if let Some(particle) = particle {
            let error = if let Some(error) = particle.error() {
                Some(NativeFabricFrameError::Particle(error))
            } else if let Some(error) = particle.consumption_error() {
                Some(NativeFabricFrameError::ParticleConsumption(error))
            } else if particle.frame().frame() != frame {
                Some(NativeFabricFrameError::ParticleFrameMismatch {
                    expected: frame,
                    actual: particle.frame().frame(),
                })
            } else {
                None
            };
            if let Some(error) = error {
                self.frame.clear(frame);
                self.error = Some(error);
                return;
            }
        }
        let surfaces = scene
            .into_iter()
            .flat_map(CpuRenderScene::live_records)
            .map(|(handle, record)| FrameRenderItem {
                object_id: RenderObjectId::Surface(handle),
                domain: RenderDomain::Surface,
                history_key: 0,
                // Native InstanceRecord owns rigid transforms, not deformation.
                motion_source: MotionSource::Rigid,
                payload: DomainPayload::Value(ValuePayload {
                    value: PayloadValue::Bounds {
                        min: core::array::from_fn(|axis| {
                            record.bounds.center[axis] - record.bounds.half_extents[axis]
                        }),
                        max: core::array::from_fn(|axis| {
                            record.bounds.center[axis] + record.bounds.half_extents[axis]
                        }),
                    },
                    role: PayloadRole(0),
                }),
            });
        let particles = particle
            .into_iter()
            .flat_map(|particle| particle.frame().items().iter().copied());
        if let Err(error) = self.frame.replace(frame, surfaces.chain(particles)) {
            self.error = Some(NativeFabricFrameError::InvalidFrame(error));
        }
    }

    fn replace_prepared_particles(&mut self, frame: u64, particles: Option<&RenderFabricFrame>) {
        // Do not repair an invalid/missing extraction with a partial GPU frame.
        // Admission reports stale extraction without hiding its original clock.
        if self.error.is_some() || self.frame.frame() != frame {
            return;
        }
        if let Some(particles) = particles {
            if particles.frame() != frame {
                self.frame.clear(frame);
                self.error = Some(NativeFabricFrameError::ParticleFrameMismatch {
                    expected: frame,
                    actual: particles.frame(),
                });
                return;
            }
        }
        let items = particles
            .into_iter()
            .flat_map(|frame| frame.items().iter().copied());
        if let Err(error) = self.frame.replace_domain(RenderDomain::Particle, items) {
            self.error = Some(NativeFabricFrameError::InvalidFrame(error));
        }
    }
}

/// Runs after particle extraction and before visibility or queue consumers.
pub fn extract_render_fabric_frame(main: &mut World, render: &mut World) {
    let frame = render
        .get_resource::<RenderRecord>()
        .map_or(0, |record| record.frame_index);
    // Move the container, not its items: preserve allocation across frames while
    // borrowing other resources without cloning the particle snapshot.
    let mut extracted = render
        .remove_resource::<ExtractedRenderFabricFrame>()
        .unwrap_or_default();
    extracted.replace(
        frame,
        main.get_resource::<PrismRenderScene>()
            .map(PrismRenderScene::scene),
        render.get_resource::<PreparedParticleFabricFrame>(),
    );
    render.insert_resource(extracted);
}

/// Refreshes Particle metadata after GPU Prepare, before frame recording.
///
/// GPU simulation owns liveness and positions. Its validated resource roles and
/// conservative bounds replace the CPU projection, including its exact-count
/// role. Disabled, pending or recovering GPU state removes the Particle domain;
/// it must never leave CPU or previous-frame payloads claiming live GPU routes.
pub(crate) fn finalize_render_fabric_frame(render: &mut World) {
    let frame = render.resource::<RenderRecord>().frame_index;
    let Some(mut extracted) = render.remove_resource::<ExtractedRenderFabricFrame>() else {
        return;
    };
    extracted.replace_prepared_particles(
        frame,
        render
            .get_resource::<crate::particle::ParticleGpuState>()
            .and_then(crate::particle::ParticleGpuState::fabric_semantic_frame),
    );
    render.insert_resource(extracted);
}

#[cfg(test)]
#[path = "fabric_frame/tests.rs"]
mod integration_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::gpu_scene::{
        InstanceRecord, SceneBounds, SceneHandle, SceneOperation, SceneTransaction,
    };

    fn apply(scene: &mut CpuRenderScene, frame: u64, operations: Vec<SceneOperation>) {
        let report = scene.apply(&SceneTransaction {
            frame_epoch: frame,
            operations,
            ..Default::default()
        });
        assert!(report.errors.is_empty());
    }

    #[test]
    fn authoritative_live_rows_survive_no_delta_and_drop_retired_generations() {
        let mut scene = CpuRenderScene::default();
        let old = SceneHandle::new(1, 1);
        let new = SceneHandle::new(1, 2);
        let record = InstanceRecord {
            bounds: SceneBounds {
                center: [3.0; 3],
                half_extents: [1.0; 3],
                ..Default::default()
            },
            ..Default::default()
        };
        apply(
            &mut scene,
            1,
            vec![SceneOperation::Create {
                handle: old,
                record,
            }],
        );
        let mut output = ExtractedRenderFabricFrame::default();
        output.replace(1, Some(&scene), None);
        assert_eq!(output.frame.len(), 1);
        apply(&mut scene, 2, vec![]);
        output.replace(2, Some(&scene), None);
        assert!(output.frame.contains_object(RenderObjectId::Surface(old)));
        apply(&mut scene, 3, vec![SceneOperation::Destroy { handle: old }]);
        apply(
            &mut scene,
            4,
            vec![SceneOperation::Create {
                handle: new,
                record,
            }],
        );
        output.replace(4, Some(&scene), None);
        assert!(!output.frame.contains_object(RenderObjectId::Surface(old)));
        assert!(output.frame.contains_object(RenderObjectId::Surface(new)));
        assert_eq!(
            output.frame.items()[0].payload,
            DomainPayload::Value(ValuePayload {
                value: PayloadValue::Bounds {
                    min: [2.0; 3],
                    max: [4.0; 3]
                },
                role: PayloadRole(0)
            })
        );
    }

    #[test]
    fn stale_domain_fails_closed_and_next_valid_frame_recovers() {
        let mut output = ExtractedRenderFabricFrame::default();
        output.replace(9, None, Some(&PreparedParticleFabricFrame::default()));
        assert_eq!(
            output.error(),
            Some(&NativeFabricFrameError::ParticleFrameMismatch {
                expected: 9,
                actual: 0
            })
        );
        assert_eq!(output.frame().frame(), 9);
        assert!(output.frame().is_empty());
        output.replace(10, None, None);
        assert!(output.error().is_none());
        assert_eq!(output.frame().frame(), 10);
    }

    #[test]
    fn invalid_surface_bounds_clear_previous_frame() {
        let mut scene = CpuRenderScene::default();
        let handle = SceneHandle::new(1, 1);
        apply(
            &mut scene,
            1,
            vec![SceneOperation::Create {
                handle,
                record: InstanceRecord::default(),
            }],
        );
        let mut output = ExtractedRenderFabricFrame::default();
        output.replace(1, Some(&scene), None);
        assert_eq!(output.frame.len(), 1);
        apply(
            &mut scene,
            2,
            vec![SceneOperation::SetBounds {
                handle,
                bounds: SceneBounds {
                    center: [f32::NAN; 3],
                    ..Default::default()
                },
            }],
        );
        output.replace(2, Some(&scene), None);
        assert!(matches!(
            output.error(),
            Some(NativeFabricFrameError::InvalidFrame(_))
        ));
        assert!(output.frame.is_empty());
    }

    #[test]
    fn prepared_particle_refresh_preserves_errors_and_rejects_stale_metadata() {
        let mut output = ExtractedRenderFabricFrame::default();
        output.replace(41, None, None);
        let stale = RenderFabricFrame::from_items(40, Vec::new()).unwrap();
        output.replace_prepared_particles(41, Some(&stale));
        let expected = NativeFabricFrameError::ParticleFrameMismatch {
            expected: 41,
            actual: 40,
        };
        assert_eq!(output.error(), Some(&expected));
        assert!(output.frame().is_empty());
        assert_eq!(output.frame().frame(), 41);
        // A later Prepare cannot mask the extraction/preparation failure.
        let fresh = RenderFabricFrame::from_items(41, Vec::new()).unwrap();
        output.replace_prepared_particles(41, Some(&fresh));
        assert_eq!(output.error(), Some(&expected));
        // Stale extraction also remains stale for the admission diagnostic.
        output.replace(40, None, None);
        output.replace_prepared_particles(41, Some(&fresh));
        assert_eq!(output.frame().frame(), 40);
        assert_eq!(output.error(), None);
        assert_eq!(
            admit_render_fabric_frame(Some(&output), 41),
            Err(NativeFabricFrameAdmissionError::FrameMismatch {
                expected: 41,
                actual: 40,
            })
        );
        output.replace(42, None, None);
        output.replace_prepared_particles(42, None);
        assert_eq!(admit_render_fabric_frame(Some(&output), 42), Ok(()));
    }
}
