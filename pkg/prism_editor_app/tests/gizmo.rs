//! Real host schema, offscreen axes/geometry, transactional gestures and persistence.
#![cfg(feature = "desktop")]

#[path = "../examples/native_gizmo/support.rs"]
mod support;

use prism_editor_app::{
    ControllerFieldEdit, ControllerLimits, DesktopError, DesktopGpu, DesktopRenderer, DesktopScene,
    DesktopViewportBounds, DocumentCodec, DocumentError, DocumentLimits, DocumentSchemaCatalog,
    EditorController, EditorDesktopUi, SceneTransformAdapter,
};
use prism_editor_core::SelectionItem;
use prism_editor_ui::{EditorAction, SelectionMode};
use prism_editor_viewport::{
    EntityId, GizmoAxis, GizmoMode, MeshId, SceneIdMatrix, SceneObjectId, SnapSettings,
    SubobjectId, TransformDelta,
};
use prism_reflect::Reflect;
use prism_ui_layout::{Point, Rect, Size};
use std::time::{Duration, Instant};

fn id(n: u128) -> SceneObjectId {
    SceneObjectId::new(
        EntityId::from_raw(n),
        MeshId::from_raw(n),
        SubobjectId::from_raw(0),
    )
}

fn models(controller: &EditorController) -> Vec<SceneIdMatrix> {
    (1..=2)
        .map(|n| {
            support::read(controller.object_root(&SelectionItem::entity(n)).unwrap())
                .unwrap()
                .matrix
        })
        .collect()
}

fn catalog() -> DocumentSchemaCatalog {
    let mut catalog = DocumentSchemaCatalog::new();
    catalog
        .register(
            std::any::type_name::<support::ExampleTransform>(),
            1,
            support::registry,
        )
        .unwrap();
    catalog
}

fn bounds() -> DesktopViewportBounds {
    // Exercise logical input against a real 2x physical target.
    DesktopViewportBounds {
        logical: Rect::new(Point::new(0.0, 0.0), Size::new(256.0, 256.0)),
        physical: Rect::new(Point::new(0.0, 0.0), Size::new(512.0, 512.0)),
        surface_extent: [512, 512],
    }
}

#[expect(
    clippy::print_stderr,
    reason = "required hardware and explicit skip evidence"
)]
fn gpu() -> Option<DesktopGpu> {
    let required = std::env::var("PRISM_REQUIRE_GPU").as_deref() == Ok("1");
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter =
        match futures_lite::future::block_on(instance.request_adapter(&Default::default())) {
            Ok(adapter) => adapter,
            Err(error) => {
                assert!(!required, "required Gizmo GPU unavailable: {error}");
                eprintln!("SKIP Gizmo GPU: {error}");
                return None;
            }
        };
    eprintln!("Gizmo GPU: {:?}", adapter.get_info());
    let (device, queue) =
        futures_lite::future::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .expect("real GPU device");
    Some(DesktopGpu {
        adapter,
        device,
        queue,
        format: wgpu::TextureFormat::Rgba8Unorm,
    })
}

fn output(gpu: &DesktopGpu) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Gizmo integration"),
        size: wgpu::Extent3d {
            width: 512,
            height: 512,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: gpu.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn render(
    scene: &mut DesktopScene,
    gpu: &DesktopGpu,
    controller: &EditorController,
    texture: &wgpu::Texture,
) {
    let view = texture.create_view(&Default::default());
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
    }
    scene
        .encode_preview(gpu, controller, &mut encoder, &view, bounds())
        .unwrap();
    gpu.queue.submit([encoder.finish()]);
}

fn pixels(gpu: &DesktopGpu, texture: &wgpu::Texture) -> Vec<u8> {
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Gizmo assertion pixels"),
        size: 512 * 512 * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(2048),
                rows_per_image: Some(512),
            },
        },
        wgpu::Extent3d {
            width: 512,
            height: 512,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    receiver
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    let mapped = buffer.slice(..).get_mapped_range().unwrap();
    let bytes = mapped.to_vec();
    drop(mapped);
    buffer.unmap();
    bytes
}

fn axis_point(scene: &DesktopScene, controller: &EditorController, axis: GizmoAxis) -> [f32; 2] {
    let layout = scene.gizmo_frame(controller).unwrap().unwrap().layout;
    let a = layout.origin();
    let b = layout.endpoint(axis);
    [(a.x() + b.x()) * 0.5, (a.y() + b.y()) * 0.5]
}

fn along(
    scene: &DesktopScene,
    controller: &EditorController,
    axis: GizmoAxis,
    start: [f32; 2],
    distance: f32,
) -> [f32; 2] {
    let layout = scene.gizmo_frame(controller).unwrap().unwrap().layout;
    let a = layout.origin();
    let b = layout.endpoint(axis);
    let length = prism_math::ops::hypot(b.x() - a.x(), b.y() - a.y());
    [
        start[0] + (b.x() - a.x()) / length * distance,
        start[1] + (b.y() - a.y()) / length * distance,
    ]
}

fn assert_scene(scene: &DesktopScene, expected: &[SceneIdMatrix]) {
    for (n, matrix) in (1..=2).zip(expected) {
        assert_eq!(scene.object_model(id(n)), Some(*matrix));
    }
}

fn pick(
    scene: &mut DesktopScene,
    gpu: &DesktopGpu,
    controller: &mut EditorController,
    position: [f32; 2],
) {
    assert!(scene
        .request_pick(gpu, controller, position, SelectionMode::Replace)
        .unwrap());
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if !scene.poll_controller(controller).unwrap().pending {
            break;
        }
        assert!(Instant::now() < deadline, "GPU selection deadline");
        std::thread::yield_now();
    }
}

#[test]
fn gpu_axes_multiselect_real_command_undo_redo_picking_and_codec_reload() {
    let Some(gpu) = gpu() else { return };
    let (mut controller, mut scene) = support::scene().unwrap();
    scene.initialize(&gpu).unwrap();
    let texture = output(&gpu);
    render(&mut scene, &gpu, &controller, &texture);
    let original = models(&controller);
    let before = pixels(&gpu, &texture);
    for (axis, channel) in [(GizmoAxis::X, 0), (GizmoAxis::Y, 1), (GizmoAxis::Z, 2)] {
        let point = axis_point(&scene, &controller, axis);
        let offset = ((point[1] * 2.0) as usize * 512 + (point[0] * 2.0) as usize) * 4;
        let pixel = &before[offset..offset + 4];
        assert_eq!(pixel[channel], 255, "visible {axis:?} axis: {pixel:?}");
        assert!(pixel[(channel + 1) % 3] < 200);
    }
    let start = axis_point(&scene, &controller, GizmoAxis::X);
    assert!(scene.gizmo_press(&mut controller, start).unwrap());
    assert!(controller.has_active_edit());
    assert!(matches!(
        DocumentCodec::encode(&controller, &catalog(), DocumentLimits::default()),
        Err(DocumentError::Busy)
    ));
    assert!(controller.execute(EditorAction::Undo).is_err());
    assert!(controller.select(SelectionItem::entity(1)).is_err());
    assert!(scene.update_object_model(id(1), original[0]).is_err());
    scene
        .gizmo_move(&mut controller, [start[0] + 10.0, start[1]])
        .unwrap();
    scene
        .gizmo_move(&mut controller, [start[0] + 26.0, start[1]])
        .unwrap();
    let snapped_generation = scene.status().scene_generation;
    assert!(!scene
        .gizmo_move(&mut controller, [start[0] + 27.0, start[1]])
        .unwrap());
    assert_eq!(
        scene.status().scene_generation,
        snapped_generation,
        "unchanged snap does not upload transforms"
    );
    assert_eq!(
        models(&controller),
        original,
        "preview cannot publish model edits"
    );
    assert_eq!(controller.document_history_lengths(), (0, 0));
    let preview: Vec<_> = (1..=2)
        .map(|n| scene.object_model(id(n)).unwrap())
        .collect();
    for i in 0..2 {
        assert!((preview[i][3][0] - original[i][3][0] - 0.2).abs() < 1e-5);
    }
    render(&mut scene, &gpu, &controller, &texture);
    assert_ne!(pixels(&gpu, &texture), before);
    scene.gizmo_release(&mut controller).unwrap();
    assert!(!controller.has_active_edit());
    assert_eq!(controller.document_history_lengths(), (1, 0));
    assert_eq!(models(&controller), preview);
    render(&mut scene, &gpu, &controller, &texture);
    assert_scene(&scene, &preview);
    // Clear selection to remove the overlay, then read actual indexed geometry.
    controller.clear_selection().unwrap();
    render(&mut scene, &gpu, &controller, &texture);
    let moved_pixels = pixels(&gpu, &texture);
    pick(&mut scene, &gpu, &mut controller, [170.0, 116.0]);
    assert_eq!(
        controller.selection().primary(),
        Some(&SelectionItem::entity(2))
    );
    controller.clear_selection().unwrap();
    controller.execute(EditorAction::Undo).unwrap();
    render(&mut scene, &gpu, &controller, &texture);
    assert_scene(&scene, &original);
    let undo_pixels = pixels(&gpu, &texture);
    assert_ne!(undo_pixels, moved_pixels, "Undo changes real GPU geometry");
    controller.execute(EditorAction::Redo).unwrap();
    render(&mut scene, &gpu, &controller, &texture);
    assert_scene(&scene, &preview);
    assert_eq!(pixels(&gpu, &texture), moved_pixels);
    let generation = scene.status().scene_generation;
    render(&mut scene, &gpu, &controller, &texture);
    assert_eq!(
        scene.status().scene_generation,
        generation,
        "idle frame has no transform uploads"
    );
    let bytes = DocumentCodec::encode(&controller, &catalog(), DocumentLimits::default()).unwrap();
    let restored = DocumentCodec::decode(
        &bytes,
        &catalog(),
        DocumentLimits::default(),
        ControllerLimits::default(),
    )
    .unwrap()
    .into_controller();
    assert_eq!(models(&restored), preview);
    controller.execute(EditorAction::Undo).unwrap();
    render(&mut scene, &gpu, &controller, &texture);
    scene.document_reloaded();
    render(&mut scene, &gpu, &restored, &texture);
    assert_scene(&scene, &preview);
    assert_eq!(pixels(&gpu, &texture), moved_pixels);
}

#[test]
fn gpu_rotation_scale_snap_zero_drag_and_renderer_cancellation_hooks() {
    let Some(gpu) = gpu() else { return };
    let (controller, mut scene) = support::scene().unwrap();
    let mut ui = EditorDesktopUi::new(controller).unwrap();
    scene.initialize(&gpu).unwrap();
    let texture = output(&gpu);
    render(&mut scene, &gpu, ui.controller(), &texture);
    let original = models(ui.controller());
    for (mode, axis, distance) in [
        (GizmoMode::Rotate, GizmoAxis::Z, 30.0),
        (GizmoMode::Scale, GizmoAxis::Y, 20.0),
    ] {
        scene.set_gizmo_tool(mode, SnapSettings::default()).unwrap();
        let start = axis_point(&scene, ui.controller(), axis);
        let end = along(&scene, ui.controller(), axis, start, distance);
        assert!(scene.gesture_press(&mut ui, start).unwrap());
        scene.gesture_move(&mut ui, end).unwrap();
        assert_eq!(models(ui.controller()), original);
        scene.gesture_release(&mut ui).unwrap();
        let changed = models(ui.controller());
        for i in 0..2 {
            assert_eq!(changed[i][3], original[i][3], "each object's own pivot");
            match mode {
                GizmoMode::Rotate => assert!((changed[i][0][1] - 0.5).abs() < 1e-5),
                GizmoMode::Scale => assert!((changed[i][1][1] - 1.2).abs() < 1e-5),
                GizmoMode::Translate => unreachable!(),
            }
        }
        ui.controller_mut().execute(EditorAction::Undo).unwrap();
        render(&mut scene, &gpu, ui.controller(), &texture);
        assert_scene(&scene, &original);
    }
    let history = ui.controller().document_history_lengths();
    let revision = ui.controller().model_revision();
    scene
        .set_gizmo_tool(GizmoMode::Translate, SnapSettings::default())
        .unwrap();
    let start = axis_point(&scene, ui.controller(), GizmoAxis::X);
    scene.gesture_press(&mut ui, start).unwrap();
    scene
        .gesture_move(&mut ui, [start[0] + 26.0, start[1]])
        .unwrap();
    scene.gesture_move(&mut ui, start).unwrap();
    scene.gesture_release(&mut ui).unwrap();
    assert_eq!(ui.controller().document_history_lengths(), history);
    assert_eq!(ui.controller().model_revision(), revision);
    // A snap step that does not divide identity must leave the other axes
    // unchanged, and returning to origin must preserve the existing redo branch.
    scene
        .set_gizmo_tool(
            GizmoMode::Scale,
            SnapSettings {
                scale_milliunits: 300,
                ..SnapSettings::default()
            },
        )
        .unwrap();
    let start = axis_point(&scene, ui.controller(), GizmoAxis::Y);
    let end = along(&scene, ui.controller(), GizmoAxis::Y, start, 20.0);
    scene.gesture_press(&mut ui, start).unwrap();
    scene.gesture_move(&mut ui, end).unwrap();
    for n in 1..=2 {
        let preview = scene.object_model(id(n)).unwrap();
        assert_eq!(preview[0], original[n as usize - 1][0]);
        assert_eq!(preview[2], original[n as usize - 1][2]);
        assert!((preview[1][1] - 1.2).abs() < 1e-5);
    }
    scene.gesture_move(&mut ui, start).unwrap();
    scene.gesture_release(&mut ui).unwrap();
    assert_scene(&scene, &original);
    assert_eq!(ui.controller().document_history_lengths(), history);
    assert_eq!(ui.controller().model_revision(), revision);
    scene
        .set_gizmo_tool(GizmoMode::Translate, SnapSettings::default())
        .unwrap();
    let start = axis_point(&scene, ui.controller(), GizmoAxis::X);
    // Escape/focus-loss/resize/suspend native routes all call this hook.
    scene.gesture_press(&mut ui, start).unwrap();
    scene
        .gesture_move(&mut ui, [start[0] + 26.0, start[1]])
        .unwrap();
    assert!(ui.has_uncommitted_edit());
    scene.gesture_cancel(&mut ui).unwrap();
    assert!(!ui.has_uncommitted_edit());
    assert_scene(&scene, &original);
    assert_eq!(models(ui.controller()), original);
    assert_eq!(ui.controller().document_history_lengths(), history);
    assert_eq!(ui.controller().model_revision(), revision);
    scene.suspend();
    scene.initialize(&gpu).unwrap();
    render(&mut scene, &gpu, ui.controller(), &texture);
    assert_scene(&scene, &original);
}

struct RejectSecond;
impl SceneTransformAdapter for RejectSecond {
    fn model(
        &self,
        item: &SelectionItem,
        mesh: SceneObjectId,
        root: &dyn Reflect,
    ) -> Result<SceneIdMatrix, DesktopError> {
        let model = support::ExampleBinding.model(item, mesh, root)?;
        // Candidate model validation fails after the first object's validation.
        if item == &SelectionItem::entity(2) && model[3][0] > 0.35 {
            return Err(DesktopError::Renderer(
                "host rejected second candidate".into(),
            ));
        }
        Ok(model)
    }
    fn edits(
        &self,
        item: &SelectionItem,
        root: &dyn Reflect,
        delta: TransformDelta,
    ) -> Result<Vec<ControllerFieldEdit>, DesktopError> {
        support::ExampleBinding.edits(item, root, delta)
    }
}

#[test]
fn gpu_failed_second_candidate_restores_previous_preview_atomically() {
    let Some(gpu) = gpu() else { return };
    let (mut controller, mut scene) = support::scene().unwrap();
    scene.set_transform_adapter(RejectSecond).unwrap();
    scene.initialize(&gpu).unwrap();
    let texture = output(&gpu);
    render(&mut scene, &gpu, &controller, &texture);
    let original = models(&controller);
    let before = pixels(&gpu, &texture);
    let start = axis_point(&scene, &controller, GizmoAxis::X);
    scene.gizmo_press(&mut controller, start).unwrap();
    // Accepted negative preview, then failure at the second positive candidate.
    scene
        .gizmo_move(&mut controller, [start[0] - 26.0, start[1]])
        .unwrap();
    assert_ne!(scene.object_model(id(1)), Some(original[0]));
    assert!(scene
        .gizmo_move(&mut controller, [start[0] + 26.0, start[1]])
        .is_err());
    assert!(!controller.has_active_edit());
    assert_eq!(models(&controller), original);
    assert_eq!(controller.document_history_lengths(), (0, 0));
    assert!(!controller.is_dirty());
    assert_scene(&scene, &original);
    render(&mut scene, &gpu, &controller, &texture);
    assert_eq!(pixels(&gpu, &texture), before);
}
