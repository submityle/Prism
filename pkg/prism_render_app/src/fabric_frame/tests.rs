use prism_app::{App, Last};
use prism_ecs::{entity::Entity, world::World};
use prism_math::{Aabb3, Vec3};
use prism_particle_app::{
    extract_particle_snapshot, EmitterRuntimeComponent, ExtractedParticleSnapshot,
    ParticleRuntimePlugin, ParticleSnapshot,
};
use prism_particle_render::ParticleFabricError;
use prism_particle_render::{
    PARTICLE_ALIVE_RESOURCE_PAYLOAD_ROLE, PARTICLE_STREAM_RESOURCE_PAYLOAD_ROLE,
};
use prism_render_architecture::{
    abi::GenerationalHandle,
    gpu_scene::SceneHandle,
    particle::{
        emitter::{BurstSpawn, Emitter, EmitterShape, SpawnParams},
        indirect_draw::RendererGeometry,
        runtime::{EmitterRuntime, EmitterRuntimeConfig},
        EmitterHandle, Vec3 as ParticleVec3,
    },
    render_fabric::{
        builtin_catalog, builtin_resource_templates, builtin_roots, CompiledFeatureGraph,
        DeviceFabric, DomainPayload, FrameRenderItem, FrameResourceBinding, GraphLimits,
        ObjectResourceBinding, ObjectResourceIdentity, ObjectRuntimeImport, PayloadRole,
        PayloadValue, PhysicalResource, QueueClass, RenderDomain, RenderFabricFrame,
        RenderObjectId, ResourceKey, ResourceLayout, ResourceTemplate, RuntimeGraphResourceMapping,
        RuntimeImport, ValuePayload, PARTICLE_ALIVE_RESOURCE,
        PARTICLE_BRIDGE_ACTIVE_ROUTE_INDICES_RESOURCE, PARTICLE_BRIDGE_EVENT_FLAGS_RESOURCE,
        PARTICLE_BRIDGE_PASS, PARTICLE_BRIDGE_ROUTE_SCRATCH_RESOURCE,
        PARTICLE_BRIDGE_VALIDATION_RESOURCE, PARTICLE_RENDER_INPUT_COUNT_RESOURCE,
        PARTICLE_RENDER_PASS, PARTICLE_RENDER_ROUTE_TABLE_RESOURCE,
        PARTICLE_SIMULATION_ACTIVE_INDICES_RESOURCE, PARTICLE_SIMULATION_COMPACT_EVENTS_RESOURCE,
        PARTICLE_SIMULATION_CONTROL_RESOURCE, PARTICLE_SIMULATION_PARTICLES_RESOURCE,
        PARTICLE_STREAM_RESOURCE,
    },
};
use prism_render_driver::{BufferId, BufferState};
use prism_render_graph::{BufferDesc, ImportedBuffer, RenderGraph};
use prism_transform::{ecs::LocalBounds, GlobalTransform, Transform};

use crate::{
    fabric_frame::{
        extract_render_fabric_frame, ExtractedRenderFabricFrame, NativeFabricFrameError,
        NativeFabricPhase, NativeFabricPhaseLowering, NativeFabricPhaseLoweringError,
    },
    particle::extract_particle_fabric_frame,
    visibility::{sync_scene, PrismRenderScene, PrismRenderable},
    PreparedParticleFabricFrame, RenderRecord,
};

fn live_runtime(origin: ParticleVec3) -> EmitterRuntime {
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
            ..Default::default()
        },
        origin,
        ParticleVec3::ZERO,
    )
    .expect("valid emitter configuration");
    runtime.step(0.1);
    assert_eq!(runtime.alive_count(), 1);
    runtime
}

fn worlds() -> (App, World, Entity, SceneHandle) {
    let mut app = App::new();
    app.add_plugins(ParticleRuntimePlugin);
    app.world_mut().insert_resource(PrismRenderScene::default());
    let surface = app.world_mut().spawn((
        PrismRenderable::new(GenerationalHandle::new(7, 1), GenerationalHandle::new(1, 1)),
        LocalBounds(Aabb3::from_center_half_extents(
            Vec3::ZERO,
            Vec3::splat(1.0),
        )),
        GlobalTransform::from_transform(&Transform::from_xyz(3.0, 3.0, 3.0)),
    ));
    sync_scene(app.world_mut());
    let handle = app
        .world()
        .resource::<PrismRenderScene>()
        .handle_for(surface)
        .expect("actual ECS surface has a scene identity");
    let emitter = app
        .world_mut()
        .spawn(EmitterRuntimeComponent::new(live_runtime(
            ParticleVec3::new(1.0, 2.0, 3.0),
        )));
    app.world_mut().run_schedule(Last);
    assert_eq!(app.world().resource::<ParticleSnapshot>().len(), 1);
    (app, World::new(), emitter, handle)
}

fn extract(app: &mut App, render: &mut World, frame_index: u64) {
    render.insert_resource(RenderRecord {
        frame_index,
        ..Default::default()
    });
    extract_particle_snapshot(app.world(), render);
    extract_particle_fabric_frame(app.world_mut(), render);
    extract_render_fabric_frame(app.world_mut(), render);
}

fn assert_shared_frame(render: &World, surface: SceneHandle, frame: u64, position: [f32; 3]) {
    let prepared = render.resource::<PreparedParticleFabricFrame>();
    assert_eq!(prepared.error(), None);
    assert_eq!(prepared.consumption_error(), None);
    assert_eq!(prepared.frame().frame(), frame);
    assert_eq!(prepared.frame().items().len(), 4);
    assert_eq!(prepared.geometry().frame(), frame);
    assert_eq!(prepared.geometry().draws().len(), 1);
    let draw = &prepared.geometry().draws()[0];
    assert_eq!(draw.bounds_min, position);
    assert_eq!(draw.bounds_max, position);
    assert_eq!(
        draw.geometry,
        RendererGeometry::Sprite {
            alive_count: 1,
            first_instance: 0
        }
    );
    assert!(matches!(draw.object_id, RenderObjectId::Particle(_)));

    let extracted = render.resource::<ExtractedRenderFabricFrame>();
    assert_eq!(extracted.error(), None);
    assert_eq!(extracted.frame().frame(), frame);
    assert_eq!(extracted.frame().len(), 5);
    assert!(extracted.frame().contains_object(draw.object_id));
    let surfaces: Vec<_> = extracted
        .frame()
        .items()
        .iter()
        .filter(|item| item.domain == RenderDomain::Surface)
        .collect();
    assert_eq!(surfaces.len(), 1);
    assert_eq!(surfaces[0].object_id, RenderObjectId::Surface(surface));
    assert_eq!(
        surfaces[0].payload,
        DomainPayload::Value(ValuePayload {
            value: PayloadValue::Bounds {
                min: [2.0; 3],
                max: [4.0; 3]
            },
            role: PayloadRole(0),
        })
    );
    let particles: Vec<_> = extracted
        .frame()
        .items()
        .iter()
        .filter(|item| item.domain == RenderDomain::Particle)
        .copied()
        .collect();
    assert_eq!(particles.as_slice(), prepared.frame().items());
}

fn particle_template(resource: ResourceKey) -> ResourceTemplate {
    builtin_resource_templates()
        .iter()
        .copied()
        .find(|template| template.key == resource)
        .expect("particle resource template is declared")
}

fn particle_compile_bindings(
    frame: u64,
    stream_buffer: Option<BufferId>,
    include_alive: bool,
) -> Vec<FrameResourceBinding> {
    let resources = [
        PARTICLE_SIMULATION_PARTICLES_RESOURCE,
        PARTICLE_SIMULATION_ACTIVE_INDICES_RESOURCE,
        PARTICLE_SIMULATION_COMPACT_EVENTS_RESOURCE,
        PARTICLE_SIMULATION_CONTROL_RESOURCE,
        PARTICLE_BRIDGE_EVENT_FLAGS_RESOURCE,
        PARTICLE_STREAM_RESOURCE,
        PARTICLE_RENDER_INPUT_COUNT_RESOURCE,
        PARTICLE_BRIDGE_VALIDATION_RESOURCE,
        PARTICLE_RENDER_ROUTE_TABLE_RESOURCE,
        PARTICLE_BRIDGE_ROUTE_SCRATCH_RESOURCE,
        PARTICLE_BRIDGE_ACTIVE_ROUTE_INDICES_RESOURCE,
        PARTICLE_ALIVE_RESOURCE,
    ];
    resources
        .into_iter()
        .enumerate()
        .filter(|(_, resource)| include_alive || *resource != PARTICLE_ALIVE_RESOURCE)
        .map(|(index, resource)| {
            let template = particle_template(resource);
            FrameResourceBinding {
                key: resource,
                frame,
                version: 0,
                physical: PhysicalResource::Buffer {
                    id: if resource == PARTICLE_STREAM_RESOURCE {
                        stream_buffer.unwrap_or_else(|| BufferId::from_parts(0x7200, 1))
                    } else {
                        BufferId::from_parts(0x7100 + index as u32, 1)
                    },
                    offset: 0,
                },
                layout: template.layout,
                alignment: template.alignment,
            }
        })
        .collect()
}

fn host_imports_for_graph(
    graph: &CompiledFeatureGraph,
    host: &mut RenderGraph,
) -> (
    Vec<RuntimeImport>,
    Vec<ObjectRuntimeImport>,
    Vec<RuntimeGraphResourceMapping>,
) {
    let mut shared = Vec::new();
    let mut objects = Vec::new();
    let mut mappings = Vec::new();
    for (index, resource) in graph.resources().iter().enumerate() {
        let binding = resource
            .external
            .expect("test graph resource must retain its external binding");
        let PhysicalResource::Buffer { id, offset } = binding.physical else {
            panic!("Particle test resources are buffers");
        };
        assert_eq!(offset, 0, "test bindings use zero-based physical ranges");
        let ResourceLayout::Buffer { bytes, usage } = resource.template.layout else {
            panic!("Particle test resources are buffer layouts");
        };
        let imported = ImportedBuffer {
            buffer: id,
            entry_state: BufferState::initial(),
        };
        if let Some(identity) = resource.identity {
            let runtime_import = RuntimeImport::Buffer {
                resource: resource.id,
                imported,
            };
            objects.push(ObjectRuntimeImport {
                address: resource.address(),
                binding,
                imported: runtime_import,
            });
        } else {
            let handle = host.import_buffer(
                format!("particle-fabric-test-{index}"),
                BufferDesc::new(bytes, usage),
                imported,
            );
            mappings.push(RuntimeGraphResourceMapping::Buffer {
                resource: resource.id,
                handle,
            });
        }
    }
    (shared, objects, mappings)
}

#[test]
fn surface_and_runtime_particle_share_one_validated_render_frame() {
    let (mut app, mut render, emitter, surface) = worlds();
    extract(&mut app, &mut render, 41);

    let snapshot = render.resource::<ExtractedParticleSnapshot>();
    assert_eq!(snapshot.frame(), 1);
    assert_eq!(snapshot.emitters().len(), 1);
    assert_eq!(snapshot.emitters()[0].emitter_entity, emitter);
    assert_eq!(snapshot.particles()[0].emitter_entity, emitter);
    assert_shared_frame(&render, surface, 41, [1.0, 2.0, 3.0]);
}

#[test]
fn frozen_simulation_clock_allows_advancing_render_frames() {
    let (mut app, mut render, _, surface) = worlds();
    // The render world's clock wins even if the main world contains another one.
    app.world_mut().insert_resource(RenderRecord {
        frame_index: 999,
        ..Default::default()
    });
    extract(&mut app, &mut render, 41);
    let snapshot = render.resource::<ExtractedParticleSnapshot>().clone();
    let items = render
        .resource::<ExtractedRenderFabricFrame>()
        .frame()
        .items()
        .to_vec();
    assert_eq!(snapshot.frame(), 1);
    assert_shared_frame(&render, surface, 41, [1.0, 2.0, 3.0]);

    // No simulation step or Last schedule: both snapshot clock and data freeze.
    for frame in [42, 43] {
        extract(&mut app, &mut render, frame);
        assert_eq!(app.world().resource::<ParticleSnapshot>().frame(), 1);
        assert_eq!(render.resource::<ExtractedParticleSnapshot>(), &snapshot);
        assert_shared_frame(&render, surface, frame, [1.0, 2.0, 3.0]);
        assert_eq!(
            render
                .resource::<ExtractedRenderFabricFrame>()
                .frame()
                .items(),
            items.as_slice()
        );
    }
}

#[test]
fn invalid_runtime_particle_clears_whole_frame_and_next_valid_frame_recovers() {
    let (mut app, mut render, emitter, surface) = worlds();
    extract(&mut app, &mut render, 41);
    assert_shared_frame(&render, surface, 41, [1.0, 2.0, 3.0]);

    // A real runtime can emit a non-finite position; extraction must reject it.
    *app.world_mut()
        .get_mut::<EmitterRuntimeComponent>(emitter)
        .expect("emitter still exists")
        .runtime_mut() = live_runtime(ParticleVec3::new(f32::NAN, 0.0, 0.0));
    app.world_mut().run_schedule(Last);
    extract(&mut app, &mut render, 42);
    let snapshot = render.resource::<ExtractedParticleSnapshot>();
    assert_eq!(snapshot.frame(), 2);
    assert_eq!(snapshot.len(), 1);
    assert!(snapshot.particles()[0].particle.position.x.is_nan());
    let error = ParticleFabricError::NonFiniteRuntimeParticle {
        instance: emitter.to_bits(),
        slot: snapshot.particles()[0].slot,
    };
    let prepared = render.resource::<PreparedParticleFabricFrame>();
    assert_eq!(prepared.error(), Some(error));
    assert_eq!(prepared.consumption_error(), None);
    assert_eq!(prepared.frame().frame(), 42);
    assert!(prepared.frame().is_empty());
    assert_eq!(prepared.geometry().frame(), 42);
    assert!(prepared.geometry().draws().is_empty());

    let extracted = render.resource::<ExtractedRenderFabricFrame>();
    assert_eq!(
        extracted.error(),
        Some(&NativeFabricFrameError::Particle(error))
    );
    assert_eq!(extracted.frame().frame(), 42);
    assert!(extracted.frame().is_empty());
    assert!(!extracted
        .frame()
        .contains_object(RenderObjectId::Surface(surface)));
    assert!(app
        .world()
        .resource::<PrismRenderScene>()
        .scene()
        .get(surface)
        .is_some());

    *app.world_mut()
        .get_mut::<EmitterRuntimeComponent>(emitter)
        .expect("same emitter recovers")
        .runtime_mut() = live_runtime(ParticleVec3::new(4.0, 5.0, 6.0));
    app.world_mut().run_schedule(Last);
    extract(&mut app, &mut render, 43);
    assert_eq!(render.resource::<ExtractedParticleSnapshot>().frame(), 3);
    assert_shared_frame(&render, surface, 43, [4.0, 5.0, 6.0]);
}

#[test]
fn frame_admission_rejects_missing_stale_and_invalid_frames_without_recording() {
    use crate::{
        systems::FrameExecutionOutcome, NativeFabricFrameAdmissionError, PrismRenderPlugin, Render,
        RenderApp, RenderAppConfig, RenderFrameStats,
    };
    use prism_render_driver::Extent3d;
    use prism_render_driver_null::NullInstance;

    let mut app = App::new();
    app.add_plugins(PrismRenderPlugin::with_config(
        NullInstance::new(),
        RenderAppConfig::new(Extent3d::new_2d(16, 12)),
    ));
    app.update();
    let baseline = app
        .get_sub_app(RenderApp)
        .unwrap()
        .world
        .resource::<RenderFrameStats>()
        .clone();
    assert_eq!(baseline.frames_submitted, 1);
    assert_eq!(baseline.fabric_frame_error, None);

    for rejection in 0..4 {
        let render = app.sub_app_mut(RenderApp).unwrap();
        let frame = render.world.resource::<RenderRecord>().frame_index;
        let expected = match rejection {
            0 => {
                render.world.remove_resource::<ExtractedRenderFabricFrame>();
                NativeFabricFrameAdmissionError::MissingFrame
            }
            1 | 2 => {
                let mut stale = ExtractedRenderFabricFrame::default();
                let actual = if rejection == 1 { frame - 1 } else { frame + 1 };
                stale.replace(actual, None, None);
                render.world.insert_resource(stale);
                NativeFabricFrameAdmissionError::FrameMismatch {
                    expected: frame,
                    actual,
                }
            }
            _ => {
                let mut invalid = ExtractedRenderFabricFrame::default();
                invalid.replace(frame, None, None);
                let error = NativeFabricFrameError::ParticleFrameMismatch {
                    expected: frame,
                    actual: frame - 1,
                };
                invalid.error = Some(error.clone());
                render.world.insert_resource(invalid);
                NativeFabricFrameAdmissionError::InvalidFrame(error)
            }
        };
        render.world.run_schedule(Render);
        let stats = render.world.resource::<RenderFrameStats>();
        assert_eq!(stats.fabric_frame_error, Some(expected));
        assert_eq!(stats.frames_failed, baseline.frames_failed + rejection + 1);
        assert_eq!(stats.frames_submitted, baseline.frames_submitted);
        assert_eq!(stats.passes_declared, baseline.passes_declared);
        assert_eq!(stats.draws_recorded, baseline.draws_recorded);
        assert_eq!(stats.execution_trace.frame_index, frame);
        assert_eq!(stats.execution_trace.graph, None);
        assert_eq!(stats.execution_trace.outcome, FrameExecutionOutcome::Failed);
    }

    // A real extraction replaces the rejected frame without resetting the app.
    app.update();
    let stats = app
        .get_sub_app(RenderApp)
        .unwrap()
        .world
        .resource::<RenderFrameStats>();
    assert_eq!(stats.fabric_frame_error, None);
    assert_eq!(stats.frames_failed, baseline.frames_failed + 4);
    assert_eq!(stats.frames_submitted, baseline.frames_submitted + 1);
    assert!(stats.execution_trace.graph.is_some());
    assert_eq!(
        stats.execution_trace.outcome,
        FrameExecutionOutcome::Submitted
    );
}

#[test]
fn native_particle_fabric_authenticates_imports_before_recording() {
    use prism_render_architecture::{
        backend::capability::CapabilitySet, render_fabric::RenderQualityTier,
    };
    use prism_render_driver::{
        BufferDescriptor, BufferUsages, DeviceDescriptor, Instance, RenderDevice,
    };
    use prism_render_driver_null::NullInstance;

    let catalog = builtin_catalog().expect("built-in catalog is valid");
    let roots = builtin_roots(&catalog).expect("built-in roots are frozen");
    let device_fabric = DeviceFabric::new(
        catalog,
        roots,
        CapabilitySet::WEBGPU_BASELINE,
        RenderQualityTier::Low,
    )
    .expect("built-in device selection resolves");
    let templates = builtin_resource_templates();
    let stream = particle_template(PARTICLE_STREAM_RESOURCE);
    let alive = particle_template(PARTICLE_ALIVE_RESOURCE);
    let object = RenderObjectId::Particle(
        prism_render_architecture::render_fabric::DomainObjectHandle::new(41, 2),
    );
    let item = |template: ResourceTemplate, role| FrameRenderItem {
        object_id: object,
        domain: RenderDomain::Particle,
        history_key: 0,
        motion_source: prism_render_architecture::motion::MotionSource::Particle,
        payload: DomainPayload::Resource(
            prism_render_architecture::render_fabric::ResourcePayload {
                resource: template.key,
                lifetime: template.lifetime,
                size: template.layout.size().unwrap(),
                alignment: template.alignment,
                layout: template.layout,
                role: PayloadRole(role),
            },
        ),
    };
    let mut extracted = ExtractedRenderFabricFrame::default();
    extracted
        .frame
        .replace(
            55,
            [
                item(stream, PARTICLE_STREAM_RESOURCE_PAYLOAD_ROLE.0),
                item(alive, PARTICLE_ALIVE_RESOURCE_PAYLOAD_ROLE.0),
            ],
        )
        .unwrap();

    let instance = NullInstance::new();
    let adapter = instance.enumerate_adapters().pop().unwrap();
    let device = instance
        .create_device(&adapter, &DeviceDescriptor::default())
        .unwrap()
        .0;
    let stream_buffer = device.create_buffer(&BufferDescriptor::new(256, BufferUsages::STORAGE));
    let alive_buffer = device.create_buffer(&BufferDescriptor::new(256, BufferUsages::STORAGE));

    let bindings = particle_compile_bindings(55, Some(stream_buffer), false);
    let alive_identity =
        ObjectResourceIdentity::new(object, 0, PARTICLE_ALIVE_RESOURCE_PAYLOAD_ROLE);
    let object_bindings = [ObjectResourceBinding {
        identity: alive_identity,
        binding: FrameResourceBinding {
            key: alive.key,
            frame: 55,
            version: 0,
            physical: PhysicalResource::Buffer {
                id: alive_buffer,
                offset: 0,
            },
            layout: alive.layout,
            alignment: alive.alignment,
        },
    }];
    let active = [PARTICLE_BRIDGE_PASS, PARTICLE_RENDER_PASS];
    let graph = extracted
        .compile_native_fabric_instanced(
            55,
            &device_fabric,
            templates,
            &bindings,
            &object_bindings,
            &active,
            &[],
            GraphLimits::default(),
        )
        .expect("particle items compile through the native Fabric");
    assert_eq!(graph.frame(), 55);
    assert!(graph
        .items()
        .iter()
        .all(|item| item.domain == RenderDomain::Particle));

    let bridge = NativeFabricPhase::identity(PARTICLE_BRIDGE_PASS, QueueClass::Compute);
    let renderer = NativeFabricPhase::identity(PARTICLE_RENDER_PASS, QueueClass::Compute);
    assert!(NativeFabricPhaseLowering::new(&graph, vec![bridge]).is_err());

    let mut host = RenderGraph::new(prism_render_driver::Extent3d::new_2d(1, 1));
    let (shared_imports, object_imports, resource_mappings) =
        host_imports_for_graph(&graph, &mut host);
    let (lowering, monitor) = ExtractedRenderFabricFrame::append_native_fabric_phases(
        graph,
        &mut host,
        &shared_imports,
        &object_imports,
        &resource_mappings,
        vec![
            NativeFabricPhase::identity(PARTICLE_BRIDGE_PASS, QueueClass::Compute),
            renderer,
        ],
    )
    .expect("exact Particle phase coverage appends to the host graph");
    assert_eq!(lowering.pass_offset(), 0);
    assert_eq!(lowering.pass_count(), 2);
    assert_eq!(host.pass_count(), 2);
    assert!(monitor.is_clean());
    host.prepare()
        .expect("host graph remains the sole graph owner");
    device.destroy_buffer(stream_buffer);
    device.destroy_buffer(alive_buffer);
}

fn empty_particle_compiled_graph() -> prism_render_architecture::render_fabric::CompiledFeatureGraph
{
    use prism_render_architecture::backend::capability::CapabilitySet;
    use prism_render_architecture::render_fabric::RenderQualityTier;

    let catalog = builtin_catalog().expect("built-in catalog is valid");
    let roots = builtin_roots(&catalog).expect("built-in roots are frozen");
    let device = DeviceFabric::new(
        catalog,
        roots,
        CapabilitySet::WEBGPU_BASELINE,
        RenderQualityTier::Low,
    )
    .expect("built-in device selection resolves");
    let mut extracted = ExtractedRenderFabricFrame::default();
    extracted.frame.clear(55);
    extracted
        .compile_native_fabric(
            55,
            &device,
            builtin_resource_templates(),
            &particle_compile_bindings(55, None, false),
            &[PARTICLE_BRIDGE_PASS, PARTICLE_RENDER_PASS],
            &[],
            GraphLimits::default(),
        )
        .expect("empty Particle frame compiles through both semantic passes")
}

#[test]
fn native_particle_phase_lowering_requires_exact_pass_coverage() {
    let graph = empty_particle_compiled_graph();

    assert!(matches!(
        NativeFabricPhaseLowering::new(
            &graph,
            vec![NativeFabricPhase::identity(
                PARTICLE_BRIDGE_PASS,
                QueueClass::Compute
            )],
        ),
        Err(NativeFabricPhaseLoweringError::MissingPass(
            PARTICLE_RENDER_PASS
        ))
    ));
    assert!(matches!(
        NativeFabricPhaseLowering::new(
            &graph,
            vec![
                NativeFabricPhase::identity(PARTICLE_BRIDGE_PASS, QueueClass::Compute),
                NativeFabricPhase::identity(PARTICLE_BRIDGE_PASS, QueueClass::Compute),
            ],
        ),
        Err(NativeFabricPhaseLoweringError::DuplicatePass(
            PARTICLE_BRIDGE_PASS
        ))
    ));
    let unknown = prism_render_architecture::render_fabric::PassKey(0xdead);
    assert!(matches!(
        NativeFabricPhaseLowering::new(
            &graph,
            vec![
                NativeFabricPhase::identity(PARTICLE_BRIDGE_PASS, QueueClass::Compute),
                NativeFabricPhase::identity(unknown, QueueClass::Compute),
            ],
        ),
        Err(NativeFabricPhaseLoweringError::UnexpectedPass(pass)) if pass == unknown
    ));
}

#[test]
fn native_particle_phase_lowering_rejects_order_and_queue_mismatch() {
    let graph = empty_particle_compiled_graph();

    assert!(matches!(
        NativeFabricPhaseLowering::new(
            &graph,
            vec![
                NativeFabricPhase::identity(PARTICLE_RENDER_PASS, QueueClass::Compute),
                NativeFabricPhase::identity(PARTICLE_BRIDGE_PASS, QueueClass::Compute),
            ],
        ),
        Err(NativeFabricPhaseLoweringError::PassOrderMismatch { index: 0, .. })
    ));
    assert!(matches!(
        NativeFabricPhaseLowering::new(
            &graph,
            vec![
                NativeFabricPhase::identity(PARTICLE_BRIDGE_PASS, QueueClass::Graphics),
                NativeFabricPhase::identity(PARTICLE_RENDER_PASS, QueueClass::Compute),
            ],
        ),
        Err(NativeFabricPhaseLoweringError::QueueMismatch {
            pass: PARTICLE_BRIDGE_PASS,
            expected: QueueClass::Compute,
            actual: QueueClass::Graphics,
        })
    ));
}

#[test]
fn native_particle_phase_lowering_appends_to_the_host_graph_once() {
    use prism_render_driver::Extent3d;
    use prism_render_graph::RenderGraph;

    let graph = empty_particle_compiled_graph();
    let mut host = RenderGraph::new(Extent3d::new_2d(1, 1));
    let (shared_imports, object_imports, resource_mappings) =
        host_imports_for_graph(&graph, &mut host);
    let (lowering, monitor) = ExtractedRenderFabricFrame::append_native_fabric_phases(
        graph,
        &mut host,
        &shared_imports,
        &object_imports,
        &resource_mappings,
        vec![
            NativeFabricPhase::identity(PARTICLE_BRIDGE_PASS, QueueClass::Compute),
            NativeFabricPhase::identity(PARTICLE_RENDER_PASS, QueueClass::Compute),
        ],
    )
    .expect("exact Particle phase coverage appends to the host graph");

    assert_eq!(lowering.pass_offset(), 0);
    assert_eq!(lowering.pass_count(), 2);
    assert_eq!(host.pass_count(), 2);
    assert!(monitor.is_clean());
    host.prepare()
        .expect("host graph remains the sole graph preparation owner");
}

#[test]
fn render_schedule_publishes_gpu_particle_roles_and_revokes_disabled_routes() {
    use crate::{
        particle::{ExtractedParticleViews, ParticleGpuState, ParticleUploadStatus},
        visibility::{PrismCamera, PrismFrustum},
        PrismRenderPlugin, RenderApp, RenderAppConfig, RenderFrameStats, RenderRuntimeResource,
    };
    use prism_math::{Frustum, Mat4, Vec4};
    use prism_particle_app::ParticleSimulationConfig;
    use prism_particle_render::{
        PARTICLE_EMITTER_COUNT_PAYLOAD_ROLE, PARTICLE_EMITTER_POSITION_BOUNDS_PAYLOAD_ROLE,
    };
    use prism_render_architecture::{motion::MotionSource, render_fabric::FrameRenderItem};
    use prism_render_driver::{Backend, DeviceCapabilities, Extent3d, Features, Limits};
    use prism_render_driver_null::NullInstance;

    let (mut app, _, _, surface) = worlds();
    // A real native orthographic camera sees both the emitter and surface.
    // World z in [-10, 10] maps to reversed depth [0, 1]; its pose faces -Z.
    let clip_from_world = Mat4::from_cols(
        Vec4::new(0.1, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 0.1, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 0.05, 0.0),
        Vec4::new(0.0, 0.0, 0.5, 1.0),
    );
    let camera = app.world_mut().spawn((
        PrismCamera::new(clip_from_world, [0, 0, 16, 12]),
        PrismFrustum(Frustum::from_view_proj(clip_from_world)),
        GlobalTransform::from_transform(&Transform::from_xyz(0.0, 0.0, 10.0)),
    ));
    let mut simulation = ParticleSimulationConfig::default();
    simulation.capacity = 4;
    simulation.paused = true;
    app.insert_resource(simulation);
    app.add_plugins(PrismRenderPlugin::with_config(
        NullInstance::with_capabilities(DeviceCapabilities {
            backend: Backend::Noop,
            features: Features::INDIRECT_DRAW,
            limits: Limits::desktop(),
        }),
        RenderAppConfig::new(Extent3d::new_2d(16, 12)).with_particle_capacity(4),
    ));
    // Use the real GPU-owner path with Null, avoiding timing assertions about
    // the asynchronous pipeline worker. No physical GPU execution is claimed.
    let render = app.sub_app_mut(RenderApp).unwrap();
    render
        .world
        .resource_mut::<crate::material::RenderMaterials>()
        .registry_mut()
        .install_execution_tables(&prism_render_material::MaterialExecutionTables::new())
        .expect("empty execution tables are valid for the fallback material");
    let runtime = render
        .world
        .resource::<RenderRuntimeResource<NullInstance>>()
        .runtime();
    let device = runtime.device_handle();
    let state = ParticleGpuState::new(
        device.as_ref(),
        device.clone(),
        runtime.device_generation(),
        4,
    )
    .unwrap();
    render.world.insert_resource(state);

    for enabled in [true, true, false, true] {
        app.world_mut()
            .resource_mut::<ParticleSimulationConfig>()
            .enabled = enabled;
        app.update();
        let render = &app.get_sub_app(RenderApp).unwrap().world;
        let frame = render.resource::<RenderRecord>().frame_index;
        let shared = render.resource::<ExtractedRenderFabricFrame>();
        assert_eq!(shared.error(), None);
        assert_eq!(shared.frame().frame(), frame);
        assert!(shared
            .frame()
            .contains_object(RenderObjectId::Surface(surface)));
        let particles: Vec<_> = shared
            .frame()
            .items()
            .iter()
            .filter(|item| item.domain == RenderDomain::Particle)
            .copied()
            .collect();
        let state = render.resource::<ParticleGpuState>();
        let views = render.resource::<ExtractedParticleViews>();
        assert_eq!(views.len(), 1);
        assert_eq!(views.iter().next().unwrap().0, camera);
        assert!(views.rejections().is_empty());
        assert_eq!(render.resource::<ParticleUploadStatus>().error, None);
        assert_eq!(state.view_error(), None);
        assert!(
            state.fabric_gpu_frame().is_none(),
            "native camera rendering must not publish a headless fixture"
        );
        if enabled {
            let semantics = state.fabric_semantic_frame().unwrap();
            assert_eq!(semantics.frame(), frame);
            assert!(!semantics.is_empty());
            assert_eq!(particles, semantics.items());
            assert!(!particles
                .iter()
                .any(|item| item.payload_role() == PARTICLE_EMITTER_COUNT_PAYLOAD_ROLE));
            assert_eq!(state.fabric_gpu_views().count(), 1);
            let (published_camera, fabric) = state.fabric_gpu_views().next().unwrap();
            assert_eq!(published_camera, camera);
            assert_eq!(fabric.frame(), frame);
            assert!(!fabric.draws().is_empty());
            assert_eq!(semantics.len(), fabric.draws().len() * 3);
            for draw in fabric.draws() {
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
        } else {
            assert!(particles.is_empty());
            assert!(state.fabric_semantic_frame().is_none());
            assert_eq!(state.fabric_gpu_views().count(), 0);
        }
        let stats = render.resource::<RenderFrameStats>();
        assert_eq!(stats.fabric_frame_error, None);
        assert_eq!(stats.frames_failed, 0);
        assert!(stats.frames_submitted > 0);
    }
}
