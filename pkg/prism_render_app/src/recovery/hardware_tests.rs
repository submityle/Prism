use crate::{
    DeviceRecoveryStatus, PrismRenderPlugin, RenderApp, RenderAppConfig, RenderAssets,
    RenderFrameStats, RenderMesh, RenderRuntimeResource, RenderSystems, RenderTarget,
    RenderTransformBinding, RenderTransforms, VisibilityWorkList,
};
use prism_app::lifecycle::{AppLifecycle, DeviceLost, DeviceRestored};
use prism_app::App;
use prism_ecs::event::{EventCursor, Events};
use prism_ecs::schedule::{IntoSystemConfigs, Schedules};
use prism_ecs::system::ResMut;
use prism_ecs::world::World;
use prism_render_driver::{BufferUsages, RenderDevice, RenderQueue};
use prism_render_driver_wgpu::{
    require_hardware_gpu, HardwareDeviceError, WgpuDevice, WgpuInstance,
};
use prism_render_material::{GpuMaterialHeader, GpuMaterialTexture};
use prism_render_runtime::RenderRuntime;
use prism_render_visibility::GpuRenderWorkItem;
use prism_transform::Transform;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

const SIZE: u32 = 16;

fn pixel(pixels: &[u8], x: u32, y: u32) -> &[u8] {
    let start = ((y * SIZE + x) * 4) as usize;
    &pixels[start..start + 4]
}

fn read_target(world: &World) -> Vec<u8> {
    let runtime = world
        .resource::<RenderRuntimeResource<WgpuInstance>>()
        .runtime();
    let target = world.resource::<RenderTarget>();
    runtime
        .queue()
        .read_texture_rgba8(target.texture(), SIZE, SIZE)
        .expect("recovery target readback must complete")
}

fn read_material_buffers(world: &World) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let runtime = world
        .resource::<RenderRuntimeResource<WgpuInstance>>()
        .runtime();
    let buffers = world.resource::<crate::MaterialParameterBuffers>();
    let queue = runtime.queue();
    let headers = queue
        .read_buffer(
            buffers.headers().expect("header buffer"),
            u64::from(buffers.header_capacity()) * core::mem::size_of::<GpuMaterialHeader>() as u64,
        )
        .expect("recovery material headers readback must complete");
    let parameters = queue
        .read_buffer(
            buffers.parameters().expect("parameter buffer"),
            u64::from(buffers.parameter_capacity_words()) * 4,
        )
        .expect("recovery material parameters readback must complete");
    let textures = queue
        .read_buffer(
            buffers.textures().expect("texture buffer"),
            u64::from(buffers.texture_capacity_rows())
                * core::mem::size_of::<GpuMaterialTexture>() as u64,
        )
        .expect("recovery material textures readback must complete");
    (headers, parameters, textures)
}

fn wait_for_callback(world: &World) {
    for _ in 0..200 {
        let runtime = world
            .resource::<RenderRuntimeResource<WgpuInstance>>()
            .runtime();
        if runtime.device_lost().is_some() {
            return;
        }
        let _ = runtime.wait_idle();
        thread::sleep(Duration::from_millis(5));
    }
}

fn gpu_available() -> bool {
    match WgpuInstance::new().request_default_hardware_device() {
        Ok((device, _queue)) => {
            eprintln!(
                "GPU recovery REAL HARDWARE GPU: {:?}",
                device.wgpu_adapter().get_info()
            );
            true
        }
        Err(HardwareDeviceError::NoAdapter) => {
            assert!(
                !require_hardware_gpu(),
                "PRISM_REQUIRE_GPU=1 but no wgpu adapter was found"
            );
            false
        }
        Err(HardwareDeviceError::NonHardware(evidence)) => {
            assert!(
                !require_hardware_gpu(),
                "PRISM_REQUIRE_GPU=1 requires a physical GPU; observed {evidence}"
            );
            eprintln!(
                "SOFTWARE/VIRTUAL adapter; skipped physical-GPU recovery evidence: {evidence}"
            );
            false
        }
        Err(HardwareDeviceError::Device(error)) => {
            panic!("physical GPU adapter was found but device creation failed: {error}");
        }
    }
}

#[test]
fn gpu_recovery_rebuilds_resources_and_restores_pixels() {
    if !gpu_available() {
        eprintln!("SKIPPED GPU recovery: no adapter");
        return;
    }

    let extent = prism_render_driver::Extent3d::new_2d(SIZE, SIZE);
    let mut app = App::new();
    app.add_plugins(PrismRenderPlugin::with_config(
        WgpuInstance::new(),
        RenderAppConfig::new(extent),
    ));
    let geometry = crate::tests::queue_scene_fixture(&mut app);
    let transform_entity = app.world_mut().spawn((
        Transform::from_xyz(0.75, 0.0, 0.0),
        RenderTransformBinding(prism_render_architecture::gpu_scene::SceneHandle::new(3, 1)),
    ));

    let work_item: GpuRenderWorkItem = {
        let render = app.get_sub_app(RenderApp).expect("render sub-app");
        render.world.resource::<VisibilityWorkList>().items()[0]
    };
    {
        let render = app.sub_app_mut(RenderApp).expect("render sub-app");
        let schedule = render
            .world
            .resource_mut::<Schedules>()
            .get_mut(crate::Render)
            .expect("render schedule");
        schedule.add_systems(
            (move |mut work: ResMut<VisibilityWorkList>| {
                work.set(vec![work_item]);
            })
            .in_set(RenderSystems::PrepareViews),
        );
    }

    // Make the material ABI buffers readable so recovery can be checked by
    // bytes, while keeping the production prepare path unchanged.
    {
        let render = app.sub_app_mut(RenderApp).expect("render sub-app");
        let runtime = render
            .world
            .resource::<RenderRuntimeResource<WgpuInstance>>();
        let device = runtime.runtime().device_handle();
        let old = render
            .world
            .remove_resource::<crate::MaterialParameterBuffers>()
            .expect("material buffers");
        let header_capacity = old.header_capacity();
        let parameter_words = old.parameter_capacity_words();
        let texture_rows = old.texture_capacity_rows();
        let mut old = old;
        old.release(device.as_ref());
        render
            .world
            .insert_resource(crate::MaterialParameterBuffers::new(
                device.as_ref(),
                header_capacity,
                parameter_words,
                texture_rows,
                BufferUsages::COPY_SRC,
            ));
        render
            .world
            .resource_mut::<crate::RenderMaterials>()
            .registry_mut()
            .mark_all_dirty();
    }

    let replay = {
        let render = app.sub_app_mut(RenderApp).expect("render sub-app");
        render
            .world
            .resource_mut::<RenderRuntimeResource<WgpuInstance>>()
            .runtime_mut()
            .register_replay("recovery readback", |device, queue| {
                let buffer = device.create_buffer(&prism_render_driver::BufferDescriptor {
                    label: Some("recovery readback".into()),
                    size: 16,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                queue.write_buffer(buffer, 0, &[9, 8, 7, 6]);
                Ok(buffer)
            })
            .expect("replay registration")
    };

    app.update();
    {
        let render = app.get_sub_app(RenderApp).expect("render sub-app");
        let runtime = render
            .world
            .resource::<RenderRuntimeResource<WgpuInstance>>();
        assert!(runtime.runtime().wait_idle());
    }
    let (right_pixels, right_materials, right_replay) = {
        let render = app.get_sub_app(RenderApp).expect("render sub-app");
        let runtime = render
            .world
            .resource::<RenderRuntimeResource<WgpuInstance>>();
        let replay_id = *replay.current();
        let replay_bytes = runtime
            .runtime()
            .queue()
            .read_buffer(replay_id, 16)
            .expect("recovery replay readback must complete");
        (
            read_target(&render.world),
            read_material_buffers(&render.world),
            replay_bytes,
        )
    };
    assert_eq!(pixel(&right_pixels, 13, 8), &[255, 255, 255, 255]);
    assert_eq!(pixel(&right_pixels, 8, 8), &[0, 0, 0, 255]);

    // Record the expected post-recovery image before destroying the device.
    app.world_mut().increment_change_tick();
    *app.world_mut()
        .get_mut::<Transform>(transform_entity)
        .expect("transform") = Transform::from_xyz(-0.75, 0.0, 0.0);
    app.update();
    let left_pixels = {
        let render = app.get_sub_app(RenderApp).expect("render sub-app");
        let runtime = render
            .world
            .resource::<RenderRuntimeResource<WgpuInstance>>();
        assert!(runtime.runtime().wait_idle());
        read_target(&render.world)
    };
    assert_eq!(pixel(&left_pixels, 2, 8), &[255, 255, 255, 255]);
    assert_eq!(pixel(&left_pixels, 8, 8), &[0, 0, 0, 255]);

    // Restore the right-hand baseline, then destroy the actual wgpu device.
    app.world_mut().increment_change_tick();
    *app.world_mut()
        .get_mut::<Transform>(transform_entity)
        .expect("transform") = Transform::from_xyz(0.75, 0.0, 0.0);
    app.update();
    let frames_before = app
        .get_sub_app(RenderApp)
        .expect("render sub-app")
        .world
        .resource::<RenderFrameStats>()
        .frames_submitted;
    let old_device: Arc<WgpuDevice> = app
        .get_sub_app(RenderApp)
        .expect("render sub-app")
        .world
        .resource::<RenderRuntimeResource<WgpuInstance>>()
        .runtime()
        .device_handle();
    let old_replay = replay.current();
    old_device.wgpu_device().destroy();
    wait_for_callback(&app.get_sub_app(RenderApp).expect("render sub-app").world);
    assert_eq!(
        app.get_sub_app(RenderApp)
            .expect("render sub-app")
            .world
            .resource::<RenderRuntimeResource<WgpuInstance>>()
            .runtime()
            .device_lost(),
        Some(prism_render_driver::DeviceLostReason::Destroyed)
    );

    let mut lost = EventCursor::<DeviceLost>::new();
    let mut restored = EventCursor::<DeviceRestored>::new();
    app.update();
    assert_eq!(
        app.get_sub_app(RenderApp)
            .expect("render sub-app")
            .world
            .resource::<RenderFrameStats>()
            .frames_submitted,
        frames_before,
        "the loss frame does not submit against the destroyed device"
    );

    // The latest main-world publication must win even though it was changed
    // after the callback and before recovery's resource rebuild.
    app.world_mut().increment_change_tick();
    *app.world_mut()
        .get_mut::<Transform>(transform_entity)
        .expect("transform") = Transform::from_xyz(-0.75, 0.0, 0.0);

    let mut saw_recovering = false;
    let mut rebuilt = false;
    for _ in 0..12 {
        app.update();
        if app.world().resource::<DeviceRecoveryStatus>().recovering {
            saw_recovering = true;
            assert_eq!(
                *app.world().resource::<AppLifecycle>(),
                AppLifecycle::Recovering
            );
        }
        let render = app.get_sub_app(RenderApp).expect("render sub-app");
        let ready = render
            .world
            .resource::<RenderAssets<RenderMesh>>()
            .iter()
            .next()
            .is_some();
        let transforms = render.world.resource::<RenderTransforms>();
        if render
            .world
            .resource::<RenderRuntimeResource<WgpuInstance>>()
            .runtime()
            .device_generation()
            == 1
            && ready
            && transforms.stats().uploaded_bytes == 48
            && app.world().resource::<DeviceRecoveryStatus>().recovering
        {
            rebuilt = true;
            break;
        }
    }
    assert!(
        rebuilt,
        "recovery rebuilt resources before normal rendering resumed"
    );
    assert!(saw_recovering, "the real callback entered App Recovering");
    assert_eq!(
        lost.read(app.world().resource::<Events<DeviceLost>>())
            .count(),
        1
    );
    assert_eq!(
        restored
            .read(app.world().resource::<Events<DeviceRestored>>())
            .count(),
        0
    );
    assert_eq!(
        app.get_sub_app(RenderApp)
            .expect("render sub-app")
            .world
            .resource::<RenderFrameStats>()
            .frames_submitted,
        frames_before,
        "recovery uploads do not run a normal frame"
    );

    let render = app.get_sub_app(RenderApp).expect("render sub-app");
    let runtime = render
        .world
        .resource::<RenderRuntimeResource<WgpuInstance>>();
    assert!(runtime.runtime().wait_idle());
    let (_, recovered_materials, recovered_replay, _) = {
        let replay_id = *replay.current();
        let replay_bytes = runtime
            .runtime()
            .queue()
            .read_buffer(replay_id, 16)
            .expect("recovered replay readback must complete");
        (
            read_target(&render.world),
            read_material_buffers(&render.world),
            replay_bytes,
            render.world.resource::<RenderFrameStats>().frames_submitted,
        )
    };
    assert_eq!(right_materials, recovered_materials);
    assert_eq!(right_replay, recovered_replay);
    assert!(!Arc::ptr_eq(
        &old_device,
        &runtime.runtime().device_handle()
    ));
    assert!(!Arc::ptr_eq(&old_replay, &replay.current()));
    assert_eq!(runtime.runtime().device_generation(), 1);

    app.update();
    assert!(!app.world().resource::<DeviceRecoveryStatus>().recovering);
    assert_eq!(
        *app.world().resource::<AppLifecycle>(),
        AppLifecycle::Running
    );
    assert_eq!(
        restored
            .read(app.world().resource::<Events<DeviceRestored>>())
            .count(),
        1
    );
    let render = app.get_sub_app(RenderApp).expect("render sub-app");
    let runtime = render
        .world
        .resource::<RenderRuntimeResource<WgpuInstance>>();
    assert!(runtime.runtime().wait_idle());
    let pixels = read_target(&render.world);
    assert_eq!(pixels, left_pixels);
    assert_eq!(pixel(&pixels, 2, 8), &[255, 255, 255, 255]);
    assert_eq!(pixel(&pixels, 8, 8), &[0, 0, 0, 255]);
    assert!(
        render.world.resource::<RenderFrameStats>().frames_submitted > frames_before,
        "normal rendering resumes only after DeviceRestored"
    );

    // Keep the geometry handle live in the test's source-of-truth assertion.
    assert!(render
        .world
        .resource::<crate::SceneRegistry>()
        .geometry_mesh(geometry)
        .is_some());
}
