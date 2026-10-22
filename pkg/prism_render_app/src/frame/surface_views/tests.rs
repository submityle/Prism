use super::*;
use crate::draw::DrawFunctions;
use crate::queue::RenderQueues;
use crate::target::RenderTarget;
use prism_ecs::prelude::Entity;
use prism_render_driver::{
    CopyCommand, DepthLoadOp, DeviceDescriptor, Extent3d, Instance, LoadOp, Pass, RenderCommand,
    RenderDevice, RequestAdapterOptions, ScissorRect, Viewport,
};
use prism_render_driver_null::NullInstance;

fn device() -> prism_render_driver_null::NullDevice {
    let instance = NullInstance::new();
    let adapter = instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("the null backend always has an adapter");
    instance
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null backend always creates a device")
        .0
}

fn camera(index: u32) -> Entity {
    Entity::from_bits((1_u64 << 32) | u64::from(index)).expect("valid test entity")
}

#[test]
fn surface_graph_rejects_duplicate_camera_before_declaring_any_target_graph() {
    let device = device();
    let output = RenderTarget::new(&device, Extent3d::new_2d(16, 8));
    let first = RenderTarget::new(&device, Extent3d::new_2d(8, 8));
    let second = RenderTarget::new(&device, Extent3d::new_2d(8, 8));
    let queues = RenderQueues::new();
    let functions = DrawFunctions::default();
    let views = [
        SurfaceViewGraphInput {
            camera: camera(7),
            order: 0,
            viewport: [0, 0, 8, 8],
            target: &first,
            queues: &queues,
            lighting: None,
        },
        SurfaceViewGraphInput {
            camera: camera(7),
            order: 0,
            viewport: [8, 0, 8, 8],
            target: &second,
            queues: &queues,
            lighting: None,
        },
    ];

    assert!(matches!(
        prepare_surface_view_graph(
            &output,
            &queues,
            &functions,
            None,
            None,
            &views,
            &FrameGraphTextureImports::default(),
        ),
        Err(SurfaceViewGraphError::DuplicateCamera(entity)) if entity == camera(7)
    ));

    second.destroy(&device);
    first.destroy(&device);
    output.destroy(&device);
}

#[test]
fn surface_graph_rejects_a_camera_rect_outside_the_output() {
    let device = device();
    let output = RenderTarget::new(&device, Extent3d::new_2d(16, 8));
    let local = RenderTarget::new(&device, Extent3d::new_2d(8, 8));
    let queues = RenderQueues::new();
    let functions = DrawFunctions::default();
    let views = [SurfaceViewGraphInput {
        camera: camera(9),
        order: 0,
        viewport: [9, 0, 8, 8],
        target: &local,
        queues: &queues,
        lighting: None,
    }];

    assert!(matches!(
        prepare_surface_view_graph(
            &output,
            &queues,
            &functions,
            None,
            None,
            &views,
            &FrameGraphTextureImports::default(),
        ),
        Err(SurfaceViewGraphError::InvalidViewport(entity)) if entity == camera(9)
    ));

    local.destroy(&device);
    output.destroy(&device);
}

fn camera_copies(passes: &[Pass]) -> Vec<(usize, &CopyCommand)> {
    passes
        .iter()
        .enumerate()
        .flat_map(|(index, pass)| match pass {
            Pass::Transfer { label, commands }
                if label.as_deref() == Some("prism-surface-view-copy") =>
            {
                commands
                    .iter()
                    .map(move |command| (index, command))
                    .collect::<Vec<_>>()
            }
            _ => Vec::new(),
        })
        .collect()
}

#[test]
fn surface_graph_defers_sorted_camera_copies_and_keeps_targets_local() {
    let device = device();
    let output = RenderTarget::new(&device, Extent3d::new_2d(32, 16));
    let first = RenderTarget::new(&device, Extent3d::new_2d(8, 6));
    let second = RenderTarget::new(&device, Extent3d::new_2d(6, 4));
    let queues = RenderQueues::new();
    let functions = DrawFunctions::default();
    // Deliberately reversed input order and overlapping output rectangles.
    let views = [
        SurfaceViewGraphInput {
            camera: camera(9),
            order: -2,
            viewport: [7, 4, 6, 4],
            target: &second,
            queues: &queues,
            lighting: None,
        },
        SurfaceViewGraphInput {
            camera: camera(7),
            order: 3,
            viewport: [3, 2, 8, 6],
            target: &first,
            queues: &queues,
            lighting: None,
        },
    ];
    let prepared = prepare_surface_view_graph(
        &output,
        &queues,
        &functions,
        None,
        None,
        &views,
        &FrameGraphTextureImports::default(),
    )
    .unwrap();
    assert_eq!(prepared.stages().copy_passes, 2);
    assert_eq!(
        prepared
            .stages()
            .cameras
            .iter()
            .map(|(entity, _)| *entity)
            .collect::<Vec<_>>(),
        vec![camera(9), camera(7)]
    );
    let scene_passes = prepared.stages().scene_passes as usize;
    assert_eq!(
        prepared.graph.pass_count(),
        scene_passes,
        "preparation must leave camera copies deferred until after particle append"
    );
    for (local, extent) in prepared
        .camera_outputs
        .iter()
        .zip([Extent3d::new_2d(6, 4), Extent3d::new_2d(8, 6)])
    {
        assert_eq!(prepared.graph.texture_extent(local.color), Some(extent));
        assert_eq!(prepared.graph.texture_extent(local.depth), Some(extent));
    }
    assert_ne!(
        prepared.camera_outputs[0].depth.resource(),
        prepared.camera_outputs[1].depth.resource()
    );
    let (graph, source) =
        finalize_surface_view_graph(prepared, None, None, &BTreeMap::new(), 0).unwrap();
    assert_eq!(graph.pass_count(), scene_passes + 2);
    let stages = graph.blackboard().get::<SurfaceViewGraphStages>().unwrap();
    stages.assert_contract();
    assert_eq!(stages.present_source, source);
    assert_eq!(stages.particle_passes, 0);
    let recorded = graph
        .record(&device)
        .expect("camera graph records on the null backend");
    let copies = camera_copies(&recorded.commands().passes);
    assert_eq!(copies.len(), 2);
    for ((_, command), (target, rect)) in copies
        .iter()
        .zip([(&second, [7, 4, 6, 4]), (&first, [3, 2, 8, 6])])
    {
        assert_eq!(
            **command,
            CopyCommand::CopyTextureToTexture {
                src: target.texture(),
                src_mip_level: 0,
                src_origin: [0, 0, 0],
                dst: output.texture(),
                dst_mip_level: 0,
                dst_origin: [rect[0], rect[1], 0],
                size: Extent3d::new_2d(rect[2], rect[3]),
            }
        );
        let clears = recorded
            .commands()
            .passes
            .iter()
            .filter(|pass| {
                matches!(pass,
                    Pass::Render { descriptor, .. } if descriptor.depth_stencil_attachment
                        .is_some_and(|depth| depth.view == target.depth_view()
                            && depth.depth.is_some_and(|ops| ops.load == DepthLoadOp::Clear(0.0)))
                )
            })
            .count();
        assert_eq!(
            clears, 1,
            "each local scene initializes its own reversed-Z depth"
        );
    }
    assert!(
        copies[0].0 < copies[1].0,
        "overlapping cameras compose in entity order"
    );
    recorded.release(&device);
    second.destroy(&device);
    first.destroy(&device);
    output.destroy(&device);
}

#[test]
fn surface_graph_native_particles_use_local_depth_and_precede_each_camera_copy() {
    use crate::particle::{
        extract_particle_views, prepare_particle_snapshot, ExtractedParticleViews,
        ParticleGpuBuildQueue, ParticleUploadStatus, ParticleViewBudget,
    };
    use crate::runtime_resource::RenderRuntimeResource;
    use crate::visibility::{PrismCamera, PrismFrustum};
    use prism_app::App;
    use prism_ecs::prelude::World;
    use prism_math::{Frustum, Mat4, Plane, Vec3};
    use prism_particle_app::{
        extract_particle_simulation_inputs, EmitterRuntimeComponent, ParticleSimulationConfig,
        ParticleSimulationPlugin,
    };
    use prism_render_architecture::particle::{
        emitter::{BurstSpawn, Emitter, EmitterShape, SpawnParams},
        runtime::{EmitterRuntime, EmitterRuntimeConfig},
        EmitterHandle,
    };
    use prism_render_driver::{Backend, DeviceCapabilities, Features, Limits};
    use prism_transform::GlobalTransform;

    // This checks production extraction, graph dependencies and recorded RHI
    // commands only. The null backend does not execute particle shader math.
    let runtime = prism_render_runtime::RenderRuntime::headless(
        NullInstance::with_capabilities(DeviceCapabilities {
            backend: Backend::Noop,
            features: Features::INDIRECT_DRAW,
            limits: Limits::desktop(),
        }),
        Extent3d::new_2d(32, 16),
    )
    .unwrap();
    let device = runtime.device_handle();
    let state = ParticleGpuState::new(
        device.as_ref(),
        device.clone(),
        runtime.device_generation(),
        1,
    )
    .unwrap();
    let mut app = App::new();
    app.add_plugins(ParticleSimulationPlugin);
    app.world_mut()
        .resource_mut::<ParticleSimulationConfig>()
        .capacity = 1;
    let mut emitter = EmitterRuntime::new(
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
        prism_render_architecture::particle::Vec3::ZERO,
        prism_render_architecture::particle::Vec3::ZERO,
    )
    .unwrap();
    emitter.step(0.1);
    app.world_mut().spawn(EmitterRuntimeComponent::new(emitter));
    let mut cameras = Vec::new();
    for rect in [[3, 2, 8, 6], [7, 4, 6, 4]] {
        let entity = app.world_mut().spawn((
            PrismCamera::new(Mat4::IDENTITY, rect),
            PrismFrustum(Frustum::from_planes([
                Plane::new(Vec3::X, 10.0),
                Plane::new(-Vec3::X, 10.0),
                Plane::new(Vec3::Y, 10.0),
                Plane::new(-Vec3::Y, 10.0),
                Plane::new(Vec3::Z, 10.0),
                Plane::new(Vec3::ZERO, 1.0),
            ])),
            GlobalTransform::default(),
        ));
        cameras.push((entity, rect));
    }
    cameras.sort_by_key(|(entity, _)| (entity.index(), entity.generation()));
    let mut render = World::new();
    extract_particle_simulation_inputs(app.world_mut(), &mut render);
    crate::surface_views::extract_surface_views(app.world_mut(), &mut render);
    extract_particle_views(app.world_mut(), &mut render);
    assert_eq!(render.resource::<ExtractedParticleViews>().len(), 2);
    assert!(render
        .resource::<ExtractedParticleViews>()
        .rejections()
        .is_empty());
    render.insert_resource(RenderRuntimeResource::new(runtime));
    render.insert_resource(crate::config::RenderAppConfig::default());
    render.insert_resource(ParticleUploadStatus::default());
    render.insert_resource(ParticleViewBudget::default());
    render.insert_resource(ParticleGpuBuildQueue::default());
    render.insert_resource(state);
    prepare_particle_snapshot::<NullInstance>(&mut render);
    assert_eq!(render.resource::<ParticleUploadStatus>().error, None);
    let state = render.resource::<ParticleGpuState>();
    assert_eq!(state.view_error(), None);
    assert!(
        state.fabric_gpu_frame().is_none(),
        "native cameras must not publish a headless fixture"
    );
    let draws: Vec<_> = state
        .fabric_gpu_views()
        .map(|(entity, fabric)| (entity, fabric.draws().len()))
        .collect();
    assert_eq!(draws.len(), 2);
    assert!(draws.iter().all(|(_, count)| *count > 0));

    let output = RenderTarget::new(device.as_ref(), Extent3d::new_2d(32, 16));
    let targets: Vec<_> = cameras
        .iter()
        .map(|(_, rect)| RenderTarget::new(device.as_ref(), Extent3d::new_2d(rect[2], rect[3])))
        .collect();
    let queues = RenderQueues::new();
    let functions = DrawFunctions::default();
    let views: Vec<_> = cameras
        .iter()
        .zip(&targets)
        .rev()
        .map(|((camera, viewport), target)| SurfaceViewGraphInput {
            camera: *camera,
            order: 0,
            viewport: *viewport,
            target,
            queues: &queues,
            lighting: None,
        })
        .collect();
    let prepared = prepare_surface_view_graph(
        &output,
        &queues,
        &functions,
        None,
        None,
        &views,
        &FrameGraphTextureImports::default(),
    )
    .unwrap();
    let local_depths: Vec<_> = prepared
        .camera_outputs
        .iter()
        .map(|local| local.depth)
        .collect();
    let (graph, _) =
        finalize_surface_view_graph(prepared, Some(state), None, &BTreeMap::new(), 0).unwrap();
    let stages = graph.blackboard().get::<SurfaceViewGraphStages>().unwrap();
    stages.assert_contract();
    assert_eq!(stages.manual.particle_passes, 0);
    assert_eq!(stages.particle_passes as usize, state.fabric_draw_count());
    for ((camera, receipt), (entity, count)) in stages.cameras.iter().zip(&draws) {
        assert_eq!(camera, entity);
        assert_eq!(receipt.particle_passes as usize, *count);
    }
    let recorded = graph
        .record(device.as_ref())
        .expect("native multiview graph records");
    let passes = &recorded.commands().passes;
    let copies = camera_copies(passes);
    assert_eq!(copies.len(), 2);
    let mut total_draws = 0;
    for ((((entity, rect), target), depth), (copy_position, copy)) in
        cameras.iter().zip(&targets).zip(&local_depths).zip(&copies)
    {
        assert_eq!(
            **copy,
            CopyCommand::CopyTextureToTexture {
                src: target.texture(),
                src_mip_level: 0,
                src_origin: [0, 0, 0],
                dst: output.texture(),
                dst_mip_level: 0,
                dst_origin: [rect[0], rect[1], 0],
                size: Extent3d::new_2d(rect[2], rect[3]),
            }
        );
        let mut local_draws = 0;
        for (position, pass) in passes.iter().enumerate() {
            let Pass::Render {
                descriptor,
                commands,
            } = pass
            else {
                continue;
            };
            if !commands.iter().any(|command| {
                matches!(
                    command,
                    RenderCommand::DrawIndirect { .. } | RenderCommand::DrawIndexedIndirect { .. }
                )
            }) {
                continue;
            }
            assert_ne!(
                descriptor.color_attachments[0].unwrap().view,
                output.view(),
                "native particles must never rasterize onto the frame-wide base target"
            );
            if descriptor.color_attachments[0].unwrap().view != target.view() {
                continue;
            }
            local_draws += 1;
            assert!(
                position < *copy_position,
                "camera particles must reach their local HDR before copy"
            );
            assert_eq!(descriptor.color_attachments[0].unwrap().load, LoadOp::Load);
            let attachment = descriptor.depth_stencil_attachment.unwrap();
            assert_eq!(attachment.view, target.depth_view());
            assert_eq!(
                attachment.depth, None,
                "particle depth testing must use the scene depth read-only"
            );
            assert!(commands.contains(&RenderCommand::SetViewport(Viewport {
                x: 0.0,
                y: 0.0,
                width: rect[2] as f32,
                height: rect[3] as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            })));
            assert!(commands.contains(&RenderCommand::SetScissor(ScissorRect {
                x: 0,
                y: 0,
                width: rect[2],
                height: rect[3],
            })));
        }
        let expected = draws.iter().find(|(camera, _)| camera == entity).unwrap().1;
        assert_eq!(local_draws, expected);
        total_draws += local_draws;
        let read_only: Vec<_> = recorded
            .plan()
            .attachments
            .iter()
            .flatten()
            .filter_map(|attachments| attachments.depth)
            .filter(|attachment| attachment.resource == depth.resource() && !attachment.writes)
            .collect();
        assert_eq!(read_only.len(), expected);
        assert!(
            read_only
                .iter()
                .all(|attachment| attachment.output_version == depth.version()
                    && attachment.depth_load == DepthLoadOp::Load),
            "particle depth sampling/testing must retain the local scene's SSA version"
        );
    }
    assert_eq!(total_draws, state.fabric_draw_count());
    assert!(copies[0].0 < copies[1].0);
    recorded.release(device.as_ref());
    render
        .remove_resource::<ParticleGpuState>()
        .unwrap()
        .destroy();
    for target in targets {
        target.destroy(device.as_ref());
    }
    output.destroy(device.as_ref());
}
