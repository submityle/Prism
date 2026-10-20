//! Required-mode GPU acceptance for the public indexed geometry adapter.

use crate::*;

const QUAD_INDICES: &[u32] = &[0, 1, 2, 2, 3, 0];

fn quad(left: f32, right: f32, z: f32) -> [[f32; 3]; 4] {
    [
        [left, -0.6, z],
        [right, -0.6, z],
        [right, 0.6, z],
        [left, 0.6, z],
    ]
}

fn object(subobject: u64) -> SceneObjectId {
    SceneObjectId::new(
        EntityId::from_raw(u128::MAX),
        MeshId::from_raw(u128::MAX - 1),
        SubobjectId::from_raw(subobject),
    )
}

struct Harness {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    viewports: ViewportRegistry,
    viewport: Viewport,
    targets: RenderTargetRegistry,
    scene: SceneIdPass,
    renderer: SceneIdGeometryRenderer,
    picks: PickCoordinator,
    matrix: SceneIdMatrix,
    depth: SceneIdDepth,
}

impl Harness {
    fn new() -> Option<Self> {
        let (adapter, device, queue) = test_support::device()?;
        let mut viewports = ViewportRegistry::new();
        let viewport = viewports.open("indexed geometry").expect("viewport");
        let backend =
            GpuPickingBackend::new(&device, &adapter, &queue, GpuPickingConfig::new(32, 32))
                .expect("backend");
        let mut targets = RenderTargetRegistry::new();
        targets
            .open(viewport.id(), viewport.generation(), backend)
            .expect("target");
        let mut renderer = SceneIdGeometryRenderer::new(&device);
        let mut scene = SceneIdPass::new();
        renderer
            .upload_geometry(&mut scene, 10, &quad(-0.6, 0.2, 0.2), QUAD_INDICES)
            .expect("near mesh");
        renderer
            .upload_geometry(&mut scene, 20, &quad(-0.2, 0.6, 0.7), QUAD_INDICES)
            .expect("far mesh");
        renderer
            .set_transform(&mut scene, 1, SCENE_ID_IDENTITY_MATRIX)
            .expect("model transform");
        let commit = scene
            .submit([
                SceneIdPassItem::new(object(u64::MAX), 10, 1, true, 0),
                SceneIdPassItem::new(object(u64::MAX - 1), 20, 1, true, 1),
            ])
            .expect("scene");
        scene
            .record_commit(&commit, &mut renderer)
            .expect("sink additions");
        Some(Self {
            adapter,
            device,
            queue,
            viewports,
            viewport,
            targets,
            scene,
            renderer,
            picks: PickCoordinator::new(),
            matrix: SCENE_ID_IDENTITY_MATRIX,
            depth: SceneIdDepth::Forward,
        })
    }

    fn state(&self) -> RenderTargetState {
        self.targets
            .state(self.viewport.id())
            .expect("target state")
    }

    fn camera(&self) -> SceneIdCamera {
        SceneIdCamera {
            target: self.state(),
            view_projection: self.matrix,
            depth: self.depth,
        }
    }

    fn encoder(&self) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default())
    }

    fn start_pick(&mut self, pixel: PixelPoint) -> InFlightPickReadback {
        let request = self
            .picks
            .submit(&self.viewport, self.state(), &self.scene, pixel)
            .expect("request");
        let target = self.targets.get(self.viewport.id()).expect("target");
        let frame = target.begin_frame(&self.scene);
        let mut encoder = self.encoder();
        self.renderer
            .render(&mut encoder, &frame, &self.scene, &self.camera())
            .expect("indexed render");
        let pending = frame
            .encode_readback(&mut encoder, request)
            .expect("copy pixel");
        target.backend().submit_readback(encoder, pending)
    }

    fn complete(&self, pending: &mut InFlightPickReadback) -> PickResult {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("wait GPU");
        let result = pending.poll().expect("map").expect("result");
        assert_eq!(pending.poll().expect("single completion"), None);
        result
    }

    fn resolve(&mut self, result: PickResult) -> PickOutcome {
        self.picks.resolve(
            &self.viewport,
            self.targets.state(self.viewport.id()),
            &self.scene,
            result,
        )
    }

    fn check_pixel(&mut self, x: u32, y: u32, expected: Option<SceneObjectId>) {
        let mut pending = self.start_pick(PixelPoint { x, y });
        let request = pending.request();
        let result = self.complete(&mut pending);
        assert_eq!(result.id, request.id);
        assert_eq!(result.viewport, request.viewport);
        assert_eq!(result.generation, request.generation);
        assert_eq!(result.scene_generation, request.scene_generation);
        assert_eq!(result.target_handle, request.target_handle);
        assert_eq!(result.target_generation, request.target_generation);
        assert_eq!(
            self.resolve(result),
            PickOutcome::Accepted { object: expected }
        );
    }

    fn change_camera(&mut self, matrix: SceneIdMatrix, depth: SceneIdDepth) {
        let mut state = self.viewport.camera();
        state.position_milliunits[0] += 1000;
        self.viewports
            .set_camera(self.viewport.id(), state)
            .expect("camera generation");
        self.viewport = self
            .viewports
            .get(self.viewport.id())
            .expect("current viewport")
            .clone();
        self.targets
            .set_viewport_generation(self.viewport.id(), self.viewport.generation())
            .expect("target generation");
        self.matrix = matrix;
        self.depth = depth;
    }

    fn start_box(
        &mut self,
        boxes: &mut BoxSelectionCoordinator,
        rectangle: PixelRect,
        limits: GpuBoxSelectionLimits,
    ) -> InFlightBoxSelectionReadback {
        let request = boxes
            .submit(&self.viewport, self.state(), &self.scene, rectangle)
            .expect("box request");
        let target = self.targets.get(self.viewport.id()).expect("target");
        let frame = target.begin_frame(&self.scene);
        let mut encoder = self.encoder();
        self.renderer
            .render(&mut encoder, &frame, &self.scene, &self.camera())
            .expect("render");
        let pending = frame
            .encode_box_readback(&mut encoder, request, limits)
            .expect("rectangle reduction");
        target.backend().submit_box_readback(encoder, pending)
    }
}

#[test]
fn indexed_gpu_occlusion_both_draw_orders_and_depth_conventions() {
    let Some(mut h) = Harness::new() else { return };
    for reverse_order in [false, true] {
        let commit = h
            .scene
            .submit([
                SceneIdPassItem::new(object(u64::MAX), 10, 1, true, u64::from(reverse_order)),
                SceneIdPassItem::new(object(u64::MAX - 1), 20, 1, true, u64::from(!reverse_order)),
            ])
            .expect("change order");
        h.scene
            .record_commit(&commit, &mut h.renderer)
            .expect("sink updates");
        for reverse_depth in [false, true] {
            let mut matrix = SCENE_ID_IDENTITY_MATRIX;
            if reverse_depth {
                matrix[2][2] = -1.0;
                matrix[3][2] = 1.0;
            }
            h.change_camera(
                matrix,
                if reverse_depth {
                    SceneIdDepth::Reverse
                } else {
                    SceneIdDepth::Forward
                },
            );
            h.check_pixel(16, 16, Some(object(u64::MAX))); // overlapping, near wins
            h.check_pixel(8, 16, Some(object(u64::MAX))); // near only
            h.check_pixel(24, 16, Some(object(u64::MAX - 1))); // far only
            h.check_pixel(0, 0, None); // uncovered, not a fullscreen fixture
        }
    }
    test_support::diagnostic(format_args!("EXERCISED indexed GPU geometry: partial overlap/order-independent depth/forward+reverse/full identities/miss"));
}

#[test]
fn indexed_gpu_camera_resize_resources_sink_and_stale_readbacks() {
    let Some(mut h) = Harness::new() else { return };
    let mut old = h.start_pick(PixelPoint { x: 16, y: 16 });
    let old_camera = h.camera();
    let mut matrix = SCENE_ID_IDENTITY_MATRIX;
    matrix[3][0] = 2.0;
    h.change_camera(matrix, SceneIdDepth::Forward);
    let old_result = h.complete(&mut old);
    assert_eq!(
        h.resolve(old_result),
        PickOutcome::Rejected(PickRejectReason::StaleGeneration)
    );
    let frame = h
        .targets
        .get(h.viewport.id())
        .expect("target")
        .begin_frame(&h.scene);
    assert_eq!(
        h.renderer
            .render(&mut h.encoder(), &frame, &h.scene, &old_camera),
        Err(SceneIdGeometryError::CameraMismatch)
    );
    h.check_pixel(16, 16, None);

    h.change_camera(SCENE_ID_IDENTITY_MATRIX, SceneIdDepth::Forward);
    let mut old = h.start_pick(PixelPoint { x: 16, y: 16 });
    let old_camera = h.camera();
    let state = h.state();
    h.targets
        .resize(
            h.viewport.id(),
            h.viewport.generation(),
            GpuPickingConfig::new(64, 16),
        )
        .expect("resize both attachments");
    assert_eq!(h.state().handle, state.handle);
    assert_eq!(h.state().target_generation, state.target_generation + 1);
    let result = h.complete(&mut old);
    assert_eq!(
        h.resolve(result),
        PickOutcome::Rejected(PickRejectReason::StaleRenderTarget)
    );
    let frame = h
        .targets
        .get(h.viewport.id())
        .expect("target")
        .begin_frame(&h.scene);
    assert_eq!(
        h.renderer
            .render(&mut h.encoder(), &frame, &h.scene, &old_camera),
        Err(SceneIdGeometryError::CameraMismatch)
    );
    h.check_pixel(32, 8, Some(object(u64::MAX)));
    h.check_pixel(0, 0, None);

    let mut old = h.start_pick(PixelPoint { x: 32, y: 8 });
    let frame = h
        .targets
        .get(h.viewport.id())
        .expect("target")
        .begin_frame(&h.scene);
    let mut model = SCENE_ID_IDENTITY_MATRIX;
    model[3][0] = 2.0;
    h.renderer
        .set_transform(&mut h.scene, 1, model)
        .expect("model update invalidates content");
    assert_eq!(
        h.renderer
            .render(&mut h.encoder(), &frame, &h.scene, &h.camera()),
        Err(SceneIdGeometryError::SceneGenerationMismatch)
    );
    let result = h.complete(&mut old);
    assert_eq!(
        h.resolve(result),
        PickOutcome::Rejected(PickRejectReason::StaleScene)
    );
    h.check_pixel(32, 8, None);
    h.renderer
        .set_transform(&mut h.scene, 1, SCENE_ID_IDENTITY_MATRIX)
        .expect("restore model");

    let mut old = h.start_pick(PixelPoint { x: 32, y: 8 });
    h.renderer
        .upload_geometry(&mut h.scene, 10, &quad(-0.6, 0.2, 0.9), QUAD_INDICES)
        .expect("near mesh becomes far");
    let result = h.complete(&mut old);
    assert_eq!(
        h.resolve(result),
        PickOutcome::Rejected(PickRejectReason::StaleScene)
    );
    h.check_pixel(32, 8, Some(object(u64::MAX - 1)));

    let generation = h.scene.generation();
    assert!(h
        .renderer
        .upload_geometry(&mut h.scene, 10, &quad(-0.6, 0.2, 0.1), &[0, 1, 9])
        .is_err());
    assert_eq!(h.scene.generation(), generation);
    h.check_pixel(32, 8, Some(object(u64::MAX - 1)));

    let prior_token = h
        .scene
        .entries()
        .find(|entry| entry.item().id == object(u64::MAX - 1))
        .expect("entry")
        .encoded_id();
    let commit = h
        .scene
        .apply_patch([SceneIdPassOperation::Remove(object(u64::MAX - 1))])
        .expect("remove");
    let frame = h
        .targets
        .get(h.viewport.id())
        .expect("target")
        .begin_frame(&h.scene);
    assert_eq!(
        h.renderer
            .render(&mut h.encoder(), &frame, &h.scene, &h.camera()),
        Err(SceneIdGeometryError::SceneOutOfSync)
    );
    h.scene
        .record_commit(&commit, &mut h.renderer)
        .expect("sink removal");
    h.check_pixel(32, 8, Some(object(u64::MAX)));
    let commit = h
        .scene
        .apply_patch([SceneIdPassOperation::Upsert(SceneIdPassItem::new(
            object(u64::MAX - 1),
            20,
            1,
            true,
            1,
        ))])
        .expect("readd");
    h.scene
        .record_commit(&commit, &mut h.renderer)
        .expect("sink readd");
    assert_eq!(h.scene.resolve(prior_token), Some(object(u64::MAX - 1)));
    h.check_pixel(32, 8, Some(object(u64::MAX - 1)));

    let mut old = h.start_pick(PixelPoint { x: 32, y: 8 });
    let state = h.targets.close(h.viewport.id()).expect("close");
    let backend =
        GpuPickingBackend::new(&h.device, &h.adapter, &h.queue, state.config).expect("replacement");
    h.targets
        .open(h.viewport.id(), h.viewport.generation(), backend)
        .expect("reopen");
    assert_ne!(state.handle, h.state().handle);
    let result = h.complete(&mut old);
    assert_eq!(
        h.resolve(result),
        PickOutcome::Rejected(PickRejectReason::StaleRenderTarget)
    );
    h.check_pixel(32, 8, Some(object(u64::MAX - 1)));
    test_support::diagnostic(format_args!("EXERCISED indexed GPU camera/model/mesh mutation/resize/reopen/sink synchronization/generation rejection"));
}

#[test]
fn indexed_gpu_bounded_rectangle_gpu_reduction_and_shared_submission() {
    let Some(mut h) = Harness::new() else { return };
    let limits = GpuBoxSelectionLimits {
        max_pixels: 512,
        max_bytes: 8192,
        max_ids: 4,
    };
    let mut boxes = BoxSelectionCoordinator::new(4);
    let mut behind = SCENE_ID_IDENTITY_MATRIX;
    behind[3][2] = 0.3;
    h.renderer
        .set_transform(&mut h.scene, 2, behind)
        .expect("occluded instance");
    let commit = h
        .scene
        .apply_patch([SceneIdPassOperation::Upsert(SceneIdPassItem::new(
            object(1),
            10,
            2,
            true,
            99,
        ))])
        .expect("occluded object");
    h.scene
        .record_commit(&commit, &mut h.renderer)
        .expect("sink add");

    // Exact nonzero origin and a rectangle smaller than the target; the result
    // comes from the GPU hash reduction, not a full-target readback.
    for (rectangle, expected) in [
        (
            PixelRect::new(18, 12, 27, 20).expect("both"),
            vec![object(u64::MAX - 1), object(u64::MAX)],
        ),
        (
            PixelRect::new(21, 12, 27, 20).expect("far only"),
            vec![object(u64::MAX - 1)],
        ),
        (PixelRect::new(0, 0, 3, 3).expect("miss"), vec![]),
    ] {
        let mut pending = h.start_box(&mut boxes, rectangle, limits);
        let request = pending.request();
        h.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU wait");
        let result = pending.poll().expect("map").expect("box result");
        assert_eq!(pending.poll().expect("completed once"), None);
        assert_eq!(result.id, request.id);
        assert_eq!(result.viewport, request.viewport);
        assert_eq!(result.generation, request.generation);
        assert_eq!(result.scene_generation, request.scene_generation);
        assert_eq!(result.target_generation, request.target_generation);
        assert_eq!(result.target_handle, request.target_handle);
        assert_eq!(result.ids.len(), expected.len());
        assert_eq!(
            boxes.resolve(&h.viewport, Some(h.state()), &h.scene, result),
            SelectionOutcome::Accepted(expected)
        );
    }

    let rectangle = PixelRect::new(18, 12, 27, 20).expect("both");
    let mut pending = h.start_box(
        &mut boxes,
        rectangle,
        GpuBoxSelectionLimits {
            max_ids: 1,
            ..limits
        },
    );
    h.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("wait");
    assert_eq!(
        pending.poll(),
        Err(GpuBoxSelectionError::IdLimitExceeded { limit: 1 })
    );
    assert_eq!(pending.poll().expect("error delivered once"), None);
    boxes.cancel_viewport(h.viewport.id());

    let request = boxes
        .submit(&h.viewport, h.state(), &h.scene, rectangle)
        .expect("box");
    let point_request = h
        .picks
        .submit(
            &h.viewport,
            h.state(),
            &h.scene,
            PixelPoint { x: 16, y: 16 },
        )
        .expect("point");
    let (mut box_flight, mut point_flight) = {
        let target = h.targets.get(h.viewport.id()).expect("target");
        let frame = target.begin_frame(&h.scene);
        let mut encoder = h.encoder();
        h.renderer
            .render(&mut encoder, &frame, &h.scene, &h.camera())
            .expect("render");
        assert!(matches!(
            frame.encode_box_readback(
                &mut encoder,
                request,
                GpuBoxSelectionLimits {
                    max_pixels: 1,
                    ..limits
                }
            ),
            Err(GpuBoxSelectionError::PixelLimitExceeded { .. })
        ));
        assert!(matches!(
            frame.encode_box_readback(
                &mut encoder,
                request,
                GpuBoxSelectionLimits {
                    max_bytes: 1,
                    ..limits
                }
            ),
            Err(GpuBoxSelectionError::ByteLimitExceeded { .. })
        ));
        assert!(matches!(
            frame.encode_box_readback(
                &mut encoder,
                BoxSelectionRequest {
                    scene_generation: request.scene_generation + 1,
                    ..request
                },
                limits
            ),
            Err(GpuBoxSelectionError::Picking(
                GpuPickingError::SceneGenerationMismatch { .. }
            ))
        ));
        assert!(matches!(
            frame.encode_box_readback(
                &mut encoder,
                BoxSelectionRequest {
                    rectangle: PixelRect::new(0, 0, 33, 1).expect("out of bounds"),
                    ..request
                },
                limits
            ),
            Err(GpuBoxSelectionError::RectangleOutOfBounds)
        ));
        let box_pending = frame
            .encode_box_readback(&mut encoder, request, limits)
            .expect("box reduction");
        let point_pending = frame
            .encode_readback(&mut encoder, point_request)
            .expect("point copy");
        h.queue.submit([encoder.finish()]); // One host-owned shared submission.
        (
            target.backend().start_box_readback(box_pending),
            target.backend().start_readback(point_pending),
        )
    };
    let point = h.complete(&mut point_flight);
    assert_eq!(
        h.resolve(point),
        PickOutcome::Accepted {
            object: Some(object(u64::MAX))
        }
    );
    let result = box_flight.poll().expect("box map").expect("box result");
    assert_eq!(
        boxes.resolve(&h.viewport, Some(h.state()), &h.scene, result),
        SelectionOutcome::Accepted(vec![object(u64::MAX - 1), object(u64::MAX)])
    );
    test_support::diagnostic(format_args!("EXERCISED indexed GPU box reduction/nonzero origin/occluded dedup/miss/capacity failure/shared point+box submission"));
}

#[test]
fn indexed_gpu_rectangle_results_reject_camera_scene_resize_and_reopen() {
    let Some(mut h) = Harness::new() else { return };
    let mut boxes = BoxSelectionCoordinator::new(4);
    let rectangle = PixelRect::new(18, 12, 27, 20).expect("rectangle");
    for mutation in 0..4 {
        let mut pending = h.start_box(&mut boxes, rectangle, GpuBoxSelectionLimits::default());
        let expected = match mutation {
            0 => {
                h.change_camera(SCENE_ID_IDENTITY_MATRIX, SceneIdDepth::Forward);
                SelectionRejectReason::StaleGeneration
            }
            1 => {
                h.renderer
                    .set_transform(&mut h.scene, 1, SCENE_ID_IDENTITY_MATRIX)
                    .expect("content invalidation");
                SelectionRejectReason::StaleScene
            }
            2 => {
                h.targets
                    .resize(
                        h.viewport.id(),
                        h.viewport.generation(),
                        GpuPickingConfig::new(48, 24),
                    )
                    .expect("resize");
                SelectionRejectReason::StaleRenderTarget
            }
            _ => {
                let state = h.targets.close(h.viewport.id()).expect("close before map");
                let backend = GpuPickingBackend::new(&h.device, &h.adapter, &h.queue, state.config)
                    .expect("replacement");
                h.targets
                    .open(h.viewport.id(), h.viewport.generation(), backend)
                    .expect("reopen");
                SelectionRejectReason::StaleRenderTarget
            }
        };
        h.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("wait");
        let result = pending
            .poll()
            .expect("map after lifecycle mutation")
            .expect("result");
        assert_eq!(
            boxes.resolve(&h.viewport, Some(h.state()), &h.scene, result),
            SelectionOutcome::Rejected(expected)
        );
    }
    test_support::diagnostic(format_args!(
        "EXERCISED indexed GPU box camera/scene/resize/close+reopen stale result rejection"
    ));
}

#[test]
fn indexed_gpu_id_texture_is_bindable_for_gpu_only_debug_preview() {
    let Some(h) = Harness::new() else { return };
    let output = h.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ID debug preview color target"),
        size: wgpu::Extent3d {
            width: 32,
            height: 32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = output.create_view(&wgpu::TextureViewDescriptor::default());
    let shader = h.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("GPU-only debug ID colorization"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
            @group(0) @binding(0) var ids: texture_2d<u32>;
            @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
                let p = array(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
                return vec4(p[i], 0.0, 1.0);
            }
            @fragment fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
                let id = textureLoad(ids, vec2<i32>(p.xy), 0);
                return vec4(f32(id.x & 1u), f32((id.x >> 1u) & 1u), 0.0, 1.0);
            }
        "#
            .into(),
        ),
    });
    let pipeline = h
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("GPU-only preview pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
    let target = h.targets.get(h.viewport.id()).expect("target");
    let binding = h.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("host ID texture binding"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(target.backend().target_view()),
        }],
    });
    let mut encoder = h.encoder();
    h.renderer
        .render(
            &mut encoder,
            &target.begin_frame(&h.scene),
            &h.scene,
            &h.camera(),
        )
        .expect("geometry ID pass");
    {
        // The fullscreen triangle here colorizes already rasterized real geometry.
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &binding, &[]);
        pass.draw(0..3, 0..1);
    }
    let buffer = h.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test-only color assertion readback"),
        size: 256 * 32,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &output,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(32),
            },
        },
        wgpu::Extent3d {
            width: 32,
            height: 32,
            depth_or_array_layers: 1,
        },
    );
    h.queue.submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).expect("callback");
        });
    h.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("wait");
    receiver.recv().expect("callback").expect("map");
    let bytes = buffer.slice(..).get_mapped_range().expect("mapped");
    let pixel = |x: usize, y: usize| &bytes[y * 256 + x * 4..y * 256 + x * 4 + 4];
    assert_eq!(pixel(0, 0), &[0, 0, 0, 255]);
    for (x, id) in [(16, object(u64::MAX)), (24, object(u64::MAX - 1))] {
        let token = h
            .scene
            .entries()
            .find(|entry| entry.item().id == id)
            .expect("entry")
            .encoded_id()
            .channels()[0];
        assert_eq!(
            pixel(x, 16),
            &[
                (token & 1) as u8 * 255,
                ((token >> 1) & 1) as u8 * 255,
                0,
                255
            ]
        );
    }
    drop(bytes);
    buffer.unmap();
    test_support::diagnostic(format_args!("EXERCISED indexed GPU ID texture binding -> GPU-only debug color pass (test-only output assertion readback)"));
}
