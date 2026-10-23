use super::*;
use crate::draw::{DrawFunctions, RenderBindGroupBinding};
use crate::queue::{PhaseItem, RenderQueues};
use crate::target::RenderTarget;
use prism_ecs::prelude::Entity;
use prism_render_architecture::gpu_scene::{GeometryHandle, SceneHandle, SceneMaterialHandle};
use prism_render_driver::{
    BindGroupId, DepthLoadOp, DeviceDescriptor, Extent3d, Instance, LoadOp, Pass, RenderCommand,
    RenderDevice, RenderPipelineId, RequestAdapterOptions, ScissorRect, TextureDescriptor,
    TextureFormat, TextureUsages, TextureViewDescriptor, Viewport,
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

fn upscale_ids(
    device: &impl RenderDevice,
) -> (RenderPipelineId, prism_render_driver::BindGroupLayoutId) {
    use prism_render_driver::*;
    let shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(include_str!(
        "../../surface_view_upscale.wgsl"
    )));
    let bindings = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("test camera upscale bindings".into()),
        entries: vec![BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float,
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }],
    });
    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("test camera upscale layout".into()),
        bind_group_layouts: vec![bindings],
        push_constant_ranges: vec![PushConstantRange {
            stages: ShaderStages::FRAGMENT,
            start: 0,
            end: 16,
        }],
    });
    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("test camera upscale".into()),
        layout: Some(layout),
        vertex: VertexState {
            module: shader,
            entry_point: "vs_main".into(),
            buffers: vec![],
        },
        primitive: PrimitiveState {
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            module: shader,
            entry_point: "fs_main".into(),
            targets: vec![Some(ColorTargetState::opaque(crate::RENDER_TARGET_FORMAT))],
        }),
    });
    (pipeline, bindings)
}

fn camera(index: u32) -> Entity {
    Entity::from_bits((1_u64 << 32) | u64::from(index)).expect("valid test entity")
}

fn sampled_queue(view: TextureViewId) -> (RenderQueues, DrawFunctions) {
    let mut functions = DrawFunctions::default();
    let draw = functions.register_with_bind_groups(
        |_, _| {},
        vec![RenderBindGroupBinding {
            identity: crate::draw::BindGroupBindingId(1),
            index: 0,
            bind_group: BindGroupId::from_parts(2, 1),
            dynamic_offsets: Vec::new(),
            buffers: Vec::new(),
            sampled_textures: vec![view],
            external_owners: Vec::new(),
        }],
    );
    let mut queues = RenderQueues::new();
    queues.push(
        RenderPhase::Opaque,
        PhaseItem {
            pipeline: RenderPipelineId::from_parts(3, 1),
            draw,
            geometry: GeometryHandle::new(0, 1),
            vertices: Vec::new(),
            vertex_sizes: Vec::new(),
            indices: None,
            index_size: None,
            first: 0,
            count: 3,
            base_vertex: 0,
            indirect: None,
            lod_level: 0,
            material: SceneMaterialHandle::new(0, 1),
            scene: SceneHandle::new(0, 1),
            order: 0,
        },
    );
    (queues, functions)
}

fn surface_view_input<'a>(
    target: &'a RenderTarget,
    queues: &'a RenderQueues,
) -> SurfaceViewGraphInput<'a> {
    SurfaceViewGraphInput {
        camera: camera(1),
        order: 0,
        clear: ClearMode::ColorAndDepth,
        viewport: [0, 0, 8, 8],
        exposure_stops: 0.0,
        target,
        queues,
        lighting: None,
        upscale_pipeline: RenderPipelineId::from_parts(30, 1),
        upscale_bindings: prism_render_driver::BindGroupLayoutId::from_parts(31, 1),
    }
}

#[test]
fn surface_graph_rejects_missing_sampled_texture_import_without_panicking() {
    let device = device();
    let output = RenderTarget::new(&device, Extent3d::new_2d(16, 8));
    let local = RenderTarget::new(&device, Extent3d::new_2d(8, 8));
    let missing_view = TextureViewId::from_parts(900, 1);
    let (queues, functions) = sampled_queue(missing_view);
    let views = [surface_view_input(&local, &queues)];

    let result = prepare_surface_view_graph(
        &output,
        &queues,
        &functions,
        None,
        None,
        &views,
        &FrameGraphTextureImports::default(),
    );
    match result {
        Err(SurfaceViewGraphError::MissingExternalTextureImport(actual)) => {
            assert_eq!(actual, missing_view);
        }
        _ => panic!("expected missing-import error"),
    }

    local.destroy(&device);
    output.destroy(&device);
}

#[test]
fn surface_graph_resolves_sampled_texture_import_before_target_append() {
    let device = device();
    let output = RenderTarget::new(&device, Extent3d::new_2d(16, 8));
    let local = RenderTarget::new(&device, Extent3d::new_2d(8, 8));
    let texture = device.create_texture(&TextureDescriptor::new_2d(
        Extent3d::new_2d(4, 4),
        TextureFormat::Rgba8Unorm,
        TextureUsages::TEXTURE_BINDING,
    ));
    let view = device.create_texture_view(texture, &TextureViewDescriptor::default());
    let mut imports = FrameGraphTextureImports::default();
    imports.set(FrameGraphTextureImport {
        texture,
        view,
        desc: TextureDesc::color(TextureFormat::Rgba8Unorm)
            .with_size(prism_render_graph::SizeClass::absolute_2d(4, 4))
            .with_usage(TextureUsages::TEXTURE_BINDING),
        entry_state: TextureState::initial(),
        exit_state: TextureState::initial(),
        mip_levels: 1,
        array_layers: 1,
    });
    let (queues, functions) = sampled_queue(view);
    let views = [surface_view_input(&local, &queues)];

    let prepared =
        prepare_surface_view_graph(&output, &queues, &functions, None, None, &views, &imports)
            .expect("published sampled texture import is resolved during target construction");
    assert!(prepared.textures.contains_key(&view));
    assert!(prepared.graph.compile().is_ok());

    device.destroy_texture_view(view);
    device.destroy_texture(texture);
    local.destroy(&device);
    output.destroy(&device);
}

#[test]
fn surface_graph_resolves_per_camera_clear_loads_in_stable_composition_order() {
    let device = device();
    let output = RenderTarget::new(&device, Extent3d::new_2d(16, 8));
    let first = RenderTarget::new(&device, Extent3d::new_2d(8, 8));
    let second = RenderTarget::new(&device, Extent3d::new_2d(8, 8));
    let third = RenderTarget::new(&device, Extent3d::new_2d(8, 8));
    let (upscale_pipeline, upscale_bindings) = upscale_ids(&device);
    let queues = RenderQueues::new();
    let functions = DrawFunctions::default();
    let views = [
        SurfaceViewGraphInput {
            camera: camera(3),
            order: 2,
            clear: ClearMode::None,
            viewport: [0, 0, 8, 8],
            target: &third,
            queues: &queues,
            lighting: None,
            exposure_stops: 0.0,
            upscale_pipeline,
            upscale_bindings,
        },
        SurfaceViewGraphInput {
            camera: camera(1),
            order: -1,
            clear: ClearMode::Inherit,
            viewport: [0, 0, 8, 8],
            target: &first,
            queues: &queues,
            lighting: None,
            exposure_stops: 0.0,
            upscale_pipeline,
            upscale_bindings,
        },
        SurfaceViewGraphInput {
            camera: camera(2),
            order: 0,
            clear: ClearMode::DepthOnly,
            viewport: [0, 0, 8, 8],
            target: &second,
            queues: &queues,
            lighting: None,
            exposure_stops: 0.0,
            upscale_pipeline,
            upscale_bindings,
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
    .expect("all clear modes form a valid graph");
    let plan = prepared.graph.compile().expect("clear graph compiles");
    let loads = |resource| {
        plan.attachments
            .iter()
            .flatten()
            .find_map(|attachments| {
                let color = attachments
                    .colors
                    .iter()
                    .find(|attachment| attachment.resource == resource)
                    .map(|attachment| attachment.load);
                let depth = attachments
                    .depth
                    .filter(|attachment| attachment.resource == resource)
                    .map(|attachment| attachment.depth_load);
                color
                    .map(|load| (Some(load), None))
                    .or_else(|| depth.map(|load| (None, Some(load))))
            })
            .expect("target attachment is used by a pass")
    };
    let first_color = prepared
        .textures
        .get(&first.view())
        .expect("first camera target was imported");
    let first_depth = prepared
        .textures
        .get(&first.depth_view())
        .expect("first camera depth was imported");
    let second_color = prepared.textures.get(&second.view()).unwrap();
    let second_depth = prepared.textures.get(&second.depth_view()).unwrap();
    let third_color = prepared.textures.get(&third.view()).unwrap();
    let third_depth = prepared.textures.get(&third.depth_view()).unwrap();

    assert_eq!(
        loads(first_color.resource()),
        (Some(LoadOp::Clear(Color::TRANSPARENT)), None)
    );
    assert_eq!(
        loads(first_depth.resource()),
        (None, Some(DepthLoadOp::Clear(0.0)))
    );
    assert_eq!(loads(second_color.resource()), (Some(LoadOp::Load), None));
    assert_eq!(
        loads(second_depth.resource()),
        (None, Some(DepthLoadOp::Clear(0.0)))
    );
    assert_eq!(loads(third_color.resource()), (Some(LoadOp::Load), None));
    assert_eq!(
        loads(third_depth.resource()),
        (None, Some(DepthLoadOp::Load))
    );

    third.destroy(&device);
    second.destroy(&device);
    first.destroy(&device);
    output.destroy(&device);
}

#[test]
fn physical_gpu_camera_clear_modes_preserve_or_clear_private_hdr_attachments() {
    use prism_render_architecture::particle::half_float_f16::f32_to_f16_bits;
    use prism_render_driver::{ImageDataLayout, RenderQueue, TextureWrite};
    use prism_render_driver_wgpu::{require_hardware_gpu, HardwareDeviceError, WgpuInstance};

    let (device, queue) = match WgpuInstance::new().request_default_hardware_device() {
        Ok(pair) => pair,
        Err(HardwareDeviceError::NoAdapter) => {
            assert!(
                !require_hardware_gpu(),
                "PRISM_REQUIRE_GPU=1 but no physical GPU adapter was found"
            );
            eprintln!("SKIPPED camera clear readback: no adapter");
            return;
        }
        Err(HardwareDeviceError::NonHardware(evidence)) => {
            assert!(
                !require_hardware_gpu(),
                "PRISM_REQUIRE_GPU=1 requires a physical GPU; observed {evidence}"
            );
            eprintln!("SKIPPED camera clear readback: non-hardware adapter: {evidence}");
            return;
        }
        Err(HardwareDeviceError::Device(error)) => {
            panic!("physical GPU device creation failed: {error}")
        }
    };
    let extent = Extent3d::new_2d(2, 2);
    let output = RenderTarget::new(&device, extent);
    let local = RenderTarget::new(&device, extent);
    let input_pixel = [0.5_f32, 0.25, 0.75, 1.0].map(f32_to_f16_bits);
    let input_texel = input_pixel
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    let input: Vec<_> = input_texel.repeat((extent.width * extent.height) as usize);

    for (mode, expected) in [
        (ClearMode::None, [0.5_f32, 0.25, 0.75, 1.0]),
        (ClearMode::DepthOnly, [0.5_f32, 0.25, 0.75, 1.0]),
        (ClearMode::ColorAndDepth, [0.0_f32; 4]),
    ] {
        queue.write_texture(
            TextureWrite {
                texture: local.texture(),
                mip_level: 0,
                origin: [0, 0, 0],
                size: extent,
            },
            &input,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(extent.width * 8),
                rows_per_image: Some(extent.height),
            },
        );
        let queues = RenderQueues::new();
        let functions = DrawFunctions::default();
        let views = [SurfaceViewGraphInput {
            camera: camera(1),
            order: 0,
            clear: mode,
            viewport: [0, 0, extent.width, extent.height],
            target: &local,
            queues: &queues,
            lighting: None,
            exposure_stops: 0.0,
            upscale_pipeline: upscale_ids(&device).0,
            upscale_bindings: upscale_ids(&device).1,
        }];
        let prepared = prepare_surface_view_graph(
            &output,
            &queues,
            &functions,
            None,
            None,
            &views,
            &FrameGraphTextureImports::default(),
        )
        .expect("physical camera clear graph is valid");
        let (graph, _) = finalize_surface_view_graph(prepared, None, None, &BTreeMap::new(), 0)
            .expect("camera output composition graph is valid");
        let recorded = graph
            .execute(&device, &queue)
            .expect("camera clear graph executes");
        let pixels = queue.read_texture_rgba16f(output.texture(), extent.width, extent.height);
        assert!(
            pixels.iter().all(|pixel| *pixel == expected),
            "mode={mode:?}: {pixels:?}"
        );
        recorded.release(&device);
    }

    local.destroy(&device);
    output.destroy(&device);
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
            clear: ClearMode::Inherit,
            viewport: [0, 0, 8, 8],
            target: &first,
            queues: &queues,
            lighting: None,
            exposure_stops: 0.0,
            upscale_pipeline: RenderPipelineId::from_parts(30, 1),
            upscale_bindings: prism_render_driver::BindGroupLayoutId::from_parts(31, 1),
        },
        SurfaceViewGraphInput {
            camera: camera(7),
            order: 0,
            clear: ClearMode::None,
            viewport: [8, 0, 8, 8],
            target: &second,
            queues: &queues,
            lighting: None,
            exposure_stops: 0.0,
            upscale_pipeline: RenderPipelineId::from_parts(30, 1),
            upscale_bindings: prism_render_driver::BindGroupLayoutId::from_parts(31, 1),
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
        clear: ClearMode::ColorAndDepth,
        viewport: [9, 0, 8, 8],
        target: &local,
        queues: &queues,
        lighting: None,
        exposure_stops: 0.0,
        upscale_pipeline: RenderPipelineId::from_parts(30, 1),
        upscale_bindings: prism_render_driver::BindGroupLayoutId::from_parts(31, 1),
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

fn camera_upscales(passes: &[Pass]) -> Vec<(usize, &[RenderCommand])> {
    passes
        .iter()
        .enumerate()
        .flat_map(|(index, pass)| match pass {
            Pass::Render {
                descriptor,
                commands,
            } if descriptor.label.as_deref() == Some("prism-surface-view-upscale") => {
                vec![(index, commands.as_slice())]
            }
            _ => Vec::new(),
        })
        .collect()
}

#[test]
fn surface_graph_defers_sorted_camera_upscales_and_keeps_targets_local() {
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
            clear: ClearMode::Inherit,
            viewport: [7, 4, 6, 4],
            exposure_stops: 0.0,
            target: &second,
            queues: &queues,
            lighting: None,
            upscale_pipeline: RenderPipelineId::from_parts(30, 1),
            upscale_bindings: prism_render_driver::BindGroupLayoutId::from_parts(31, 1),
        },
        SurfaceViewGraphInput {
            camera: camera(7),
            order: 3,
            clear: ClearMode::DepthOnly,
            viewport: [3, 2, 8, 6],
            exposure_stops: 0.0,
            target: &first,
            queues: &queues,
            lighting: None,
            upscale_pipeline: RenderPipelineId::from_parts(30, 1),
            upscale_bindings: prism_render_driver::BindGroupLayoutId::from_parts(31, 1),
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
    assert_eq!(prepared.stages().upscale_passes, 2);
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
        "preparation must leave camera composition deferred until after particle append"
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
    let upscales = camera_upscales(&recorded.commands().passes);
    assert_eq!(upscales.len(), 2);
    for ((_, commands), (target, rect)) in upscales
        .iter()
        .zip([(&second, [7, 4, 6, 4]), (&first, [3, 2, 8, 6])])
    {
        assert!(commands.iter().any(
            |command| matches!(command, RenderCommand::SetViewport(viewport)
            if viewport.x == rect[0] as f32 && viewport.y == rect[1] as f32
                && viewport.width == rect[2] as f32 && viewport.height == rect[3] as f32)
        ));
        assert!(commands.iter().any(
            |command| matches!(command, RenderCommand::SetScissor(scissor)
            if scissor.x == rect[0] && scissor.y == rect[1]
                && scissor.width == rect[2] && scissor.height == rect[3])
        ));
        assert!(commands
            .iter()
            .any(|command| matches!(command, RenderCommand::Draw { .. })));
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
        upscales[0].0 < upscales[1].0,
        "overlapping cameras compose in entity order"
    );
    recorded.release(&device);
    second.destroy(&device);
    first.destroy(&device);
    output.destroy(&device);
}

#[test]
fn camera_local_exposure_isolated_in_null_graph_before_global_display_stage() {
    use prism_render_driver::{RenderCommand, ShaderStages};

    let device = device();
    let output = RenderTarget::new(&device, Extent3d::new_2d(24, 8));
    let first = RenderTarget::new(&device, Extent3d::new_2d(8, 8));
    let second = RenderTarget::new(&device, Extent3d::new_2d(8, 8));
    let (upscale_pipeline, upscale_bindings) = upscale_ids(&device);
    let queues = RenderQueues::new();
    let functions = DrawFunctions::default();
    let views = [
        SurfaceViewGraphInput {
            camera: camera(2),
            order: 0,
            clear: ClearMode::ColorAndDepth,
            viewport: [0, 0, 8, 8],
            exposure_stops: -2.0,
            target: &first,
            queues: &queues,
            lighting: None,
            upscale_pipeline,
            upscale_bindings,
        },
        SurfaceViewGraphInput {
            camera: camera(3),
            order: 1,
            clear: ClearMode::ColorAndDepth,
            viewport: [8, 0, 8, 8],
            exposure_stops: 1.5,
            target: &second,
            queues: &queues,
            lighting: None,
            upscale_pipeline,
            upscale_bindings,
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
    let (graph, _) =
        finalize_surface_view_graph(prepared, None, None, &BTreeMap::new(), 0).unwrap();
    let stages = graph.blackboard().get::<SurfaceViewGraphStages>().unwrap();
    assert_eq!(stages.upscale_passes, 2);
    let recorded = graph.record(&device).expect("Null graph records");
    let upscale_commands = camera_upscales(&recorded.commands().passes);
    assert_eq!(upscale_commands.len(), 2);
    let read_exposure = |commands: &[RenderCommand]| {
        let bytes = commands
            .iter()
            .find_map(|command| match command {
                RenderCommand::SetPushConstants {
                    stages: actual_stages,
                    offset: 0,
                    data,
                } if *actual_stages == ShaderStages::FRAGMENT && data.len() == 16 => {
                    Some(data.as_slice())
                }
                _ => None,
            })
            .expect("each camera upscale uploads its local exposure");
        f32::from_ne_bytes(bytes[0..4].try_into().unwrap())
    };
    assert_eq!(read_exposure(upscale_commands[0].1), -2.0);
    assert_eq!(read_exposure(upscale_commands[1].1), 1.5);
    assert!(upscale_commands[0].0 < upscale_commands[1].0);
    // This graph ends in frame-wide linear HDR. The host appends the single
    // global display transform after this graph, never per camera.
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
            clear: ClearMode::Inherit,
            viewport: *viewport,
            target,
            queues: &queues,
            lighting: None,
            exposure_stops: 0.0,
            upscale_pipeline: RenderPipelineId::from_parts(30, 1),
            upscale_bindings: prism_render_driver::BindGroupLayoutId::from_parts(31, 1),
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
    let upscales = camera_upscales(passes);
    assert_eq!(upscales.len(), 2);
    let mut total_draws = 0;
    for ((((entity, rect), target), depth), (upscale_position, upscale_commands)) in cameras
        .iter()
        .zip(&targets)
        .zip(&local_depths)
        .zip(&upscales)
    {
        assert!(upscale_commands.iter().any(
            |command| matches!(command, RenderCommand::SetViewport(viewport)
            if viewport.x == rect[0] as f32 && viewport.y == rect[1] as f32
                && viewport.width == rect[2] as f32 && viewport.height == rect[3] as f32)
        ));
        assert!(upscale_commands.iter().any(
            |command| matches!(command, RenderCommand::SetScissor(scissor)
            if scissor.x == rect[0] && scissor.y == rect[1]
                && scissor.width == rect[2] && scissor.height == rect[3])
        ));
        assert!(upscale_commands
            .iter()
            .any(|command| matches!(command, RenderCommand::Draw { .. })));
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
                position < *upscale_position,
                "camera particles must reach their local HDR before composition"
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
    assert!(upscales[0].0 < upscales[1].0);
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
