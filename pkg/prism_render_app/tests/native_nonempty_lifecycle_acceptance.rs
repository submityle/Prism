//! Native non-empty scene lifecycle acceptance.
//!
//! The test uses a real wgpu device and queue.  The queue wrapper delays only
//! Prism's completion proof, so the first frame can be read back from hardware
//! while a resize still has to fail closed until that proof is released.
#![forbid(unsafe_code)]

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

use prism_app::App;
use prism_asset::{AssetId, Assets};
use prism_ecs::prelude::Entity;
use prism_math::{Aabb3, Frustum, Mat4, Plane, Vec3};
use prism_mesh::{Indices, Mesh, MeshVertexAttribute, VertexAttributeValues};
use prism_render_app::surface_view_targets::{SurfaceViewTargetKey, SurfaceViewTargets};
use prism_render_app::visibility::{
    sync_scene, GeometryHandle, LocalBounds, PrismCamera, PrismFrustum, PrismRenderScene,
    PrismRenderable, PrismVisibilityPlugin,
};
use prism_render_app::{
    LightingLightScene, NativeFrameExecution, NativeFramePhase, PrismRenderPlugin, RenderApp,
    RenderAppConfig, RenderFrameStats, RenderMaterials, RenderRuntimeResource, SceneRegistry,
    VisibilityWorkList,
};
use prism_render_architecture::geometry::{
    GeometryLodRecord, GeometryPrimitiveKind, GeometryRecord,
};
use prism_render_architecture::gpu_scene::{SceneHandle, SceneMaterialHandle};
use prism_render_architecture::lighting::{
    LightClass, LightClassification, LightEmission, LightGeometry, LightMobility, LightRecord,
    LightUnit,
};
use prism_render_driver as rhi;
use prism_render_driver::{Adapter, FrameSync, Instance, PrimitiveTopology, RenderQueue};
use prism_render_driver_wgpu::{
    require_hardware_gpu, WgpuAdapter, WgpuDevice, WgpuInstance, WgpuQueue, WgpuSurface,
};
use prism_render_material::{
    fallback_material_record, BlendDomain, BuiltinProgram, DynamicChannels,
    MaterialExecutionTables, ProgramRef, ResponseMaskSource, ShadingPlan,
};
use prism_render_visibility::{
    GeometryLod, GeometryLodChain, GpuRenderWorkItem, RenderPassMask, VisibilityStageMask,
    WorkSortKey,
};
use prism_transform::{GlobalTransform, Transform};

const INITIAL_WIDTH: u32 = 8;
const INITIAL_HEIGHT: u32 = 8;
const RESIZED_WIDTH: u32 = 4;
const RESIZED_HEIGHT: u32 = 6;

#[derive(Default)]
struct CompletionGate {
    released: AtomicBool,
    callbacks: Mutex<Vec<Box<dyn FnOnce() + Send + 'static>>>,
}

impl CompletionGate {
    fn is_released(&self) -> bool {
        self.released.load(Ordering::Acquire)
    }

    fn block(&self) {
        self.released.store(false, Ordering::Release);
    }

    fn release(&self, queue: &WgpuQueue) {
        self.released.store(true, Ordering::Release);
        let callbacks = self
            .callbacks
            .lock()
            .expect("completion callback lock")
            .drain(..)
            .collect::<Vec<_>>();
        for callback in callbacks {
            queue.on_submitted_work_done(callback);
        }
    }

    fn register(&self, queue: &WgpuQueue, callback: Box<dyn FnOnce() + Send + 'static>) {
        if self.is_released() {
            queue.on_submitted_work_done(callback);
            return;
        }
        let mut callbacks = self.callbacks.lock().expect("completion callback lock");
        if self.is_released() {
            drop(callbacks);
            queue.on_submitted_work_done(callback);
        } else {
            callbacks.push(callback);
        }
    }
}

struct DelayedQueue {
    inner: WgpuQueue,
    gate: Arc<CompletionGate>,
}

impl DelayedQueue {
    fn read_texture_rgba16f(
        &self,
        texture: rhi::TextureId,
        width: u32,
        height: u32,
    ) -> Result<Vec<[f32; 4]>, rhi::ReadbackError> {
        self.inner.read_texture_rgba16f(texture, width, height)
    }

    fn read_texture_rgba8(
        &self,
        texture: rhi::TextureId,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, rhi::ReadbackError> {
        self.inner.read_texture_rgba8(texture, width, height)
    }
}

impl RenderQueue for DelayedQueue {
    fn write_buffer(&self, buffer: rhi::BufferId, offset: u64, data: &[u8]) {
        self.inner.write_buffer(buffer, offset, data);
    }

    fn write_texture(
        &self,
        destination: rhi::TextureWrite,
        data: &[u8],
        layout: rhi::ImageDataLayout,
    ) {
        self.inner.write_texture(destination, data, layout);
    }

    fn submit(&self, command_buffers: &[rhi::CommandBuffer]) {
        self.inner.submit(command_buffers);
    }

    fn request_buffer_readback(
        &self,
        ranges: &[rhi::BufferReadbackRange],
        max_bytes: u64,
    ) -> Result<Box<dyn rhi::BufferReadback>, rhi::BufferReadbackError> {
        self.inner.request_buffer_readback(ranges, max_bytes)
    }
}

impl FrameSync for DelayedQueue {
    fn submit_tracked(&self, command_buffers: &[rhi::CommandBuffer]) -> rhi::CompletionToken {
        Box::new(self.inner.submit_tracked(command_buffers))
    }

    fn wait_for(&self, token: &(dyn core::any::Any + Send + Sync)) -> bool {
        self.gate.is_released() && self.inner.wait_for(token)
    }

    fn wait_idle(&self) -> bool {
        self.gate.is_released() && self.inner.wait_idle()
    }

    fn on_submitted_work_done(&self, callback: Box<dyn FnOnce() + Send + 'static>) {
        self.gate.register(&self.inner, callback);
    }
}

struct DelayedInstance {
    inner: WgpuInstance,
    gate: Arc<CompletionGate>,
    hardware_only: bool,
}

impl DelayedInstance {
    fn new(gate: Arc<CompletionGate>, hardware_only: bool) -> Self {
        Self {
            inner: WgpuInstance::new(),
            gate,
            hardware_only,
        }
    }

    fn adapters(&self) -> Vec<WgpuAdapter> {
        <WgpuInstance as Instance>::enumerate_adapters(&self.inner)
    }
}

impl Instance for DelayedInstance {
    type Adapter = WgpuAdapter;
    type Device = WgpuDevice;
    type Queue = DelayedQueue;
    type Surface = WgpuSurface;

    fn enumerate_adapters(&self) -> Vec<Self::Adapter> {
        self.adapters()
    }

    fn request_adapter(&self, options: &rhi::RequestAdapterOptions) -> Option<Self::Adapter> {
        if self.hardware_only {
            self.adapters().into_iter().find(|adapter| {
                matches!(
                    adapter.info().device_type,
                    rhi::DeviceType::IntegratedGpu | rhi::DeviceType::DiscreteGpu
                )
            })
        } else {
            <WgpuInstance as Instance>::request_adapter(&self.inner, options)
        }
    }

    fn request_adapter_for_surface(
        &self,
        options: &rhi::RequestAdapterOptions,
        _surface: &Self::Surface,
    ) -> Option<Self::Adapter> {
        self.request_adapter(options)
    }

    fn create_device(
        &self,
        adapter: &Self::Adapter,
        descriptor: &rhi::DeviceDescriptor,
    ) -> Result<(Self::Device, Self::Queue), rhi::DeviceError> {
        let (device, queue) =
            <WgpuInstance as Instance>::create_device(&self.inner, adapter, descriptor)?;
        Ok((
            device,
            DelayedQueue {
                inner: queue,
                gate: Arc::clone(&self.gate),
            },
        ))
    }

    fn create_surface<W>(
        &self,
        _window: Arc<W>,
        _device: &Self::Device,
        _queue: &Self::Queue,
    ) -> Result<Self::Surface, rhi::SurfaceError>
    where
        W: raw_window_handle::HasWindowHandle
            + raw_window_handle::HasDisplayHandle
            + Send
            + Sync
            + 'static,
    {
        Err(rhi::SurfaceError::Unsupported)
    }
}

struct SceneFixture {
    camera: Entity,
    material: SceneMaterialHandle,
}

fn broad_frustum() -> PrismFrustum {
    PrismFrustum(Frustum::from_planes([
        Plane::new(Vec3::X, 10.0),
        Plane::new(-Vec3::X, 10.0),
        Plane::new(Vec3::Y, 10.0),
        Plane::new(-Vec3::Y, 10.0),
        Plane::new(Vec3::Z, 10.0),
        Plane::new(-Vec3::Z, 10.0),
    ]))
}

fn scene_fixture(app: &mut App) -> SceneFixture {
    scene_fixture_with_material(
        app,
        ShadingPlan::single_builtin(BuiltinProgram::Unlit),
        MaterialExecutionTables::builtin(),
        DynamicChannels {
            wetness: 0.5,
            ..DynamicChannels::NONE
        },
    )
}

fn scene_fixture_with_material(
    app: &mut App,
    response_plan: ShadingPlan,
    execution_tables: MaterialExecutionTables,
    dynamic_channels: DynamicChannels,
) -> SceneFixture {
    let has_mask_blend = matches!(response_plan, ShadingPlan::MaskBlend { .. });
    let mesh = Mesh::new(PrimitiveTopology::TriangleList)
        .with_attribute(
            MeshVertexAttribute::new("Vertex_Position", rhi::VertexFormat::Float32x3, 0),
            VertexAttributeValues::Float32x3(vec![
                [-0.5, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [0.5, 0.5, 0.0],
                [-0.5, 0.5, 0.0],
            ]),
        )
        .expect("positions fit")
        .with_attribute(
            MeshVertexAttribute::new("Vertex_Normal", rhi::VertexFormat::Float32x3, 1),
            VertexAttributeValues::Float32x3(vec![[0.0, 0.0, 1.0]; 4]),
        )
        .expect("normals fit")
        .with_attribute(
            MeshVertexAttribute::new("Vertex_Uv", rhi::VertexFormat::Float32x2, 2),
            VertexAttributeValues::Float32x2(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        )
        .expect("UVs fit")
        .indexed(Indices::Uint16(vec![0, 1, 2, 0, 2, 3]));
    let mut meshes = Assets::new();
    let mesh_id: AssetId<Mesh> = meshes.insert(mesh).id();
    app.world_mut().insert_resource(meshes);

    let geometry = GeometryHandle::new(1, 1);
    let material = app
        .sub_app_mut(RenderApp)
        .expect("the native render sub-app")
        .world
        .resource_mut::<RenderMaterials>()
        .registry_mut()
        .allocate()
        .expect("the native material fixture allocates a stable handle");
    {
        let scene = app.world_mut().resource_mut::<PrismRenderScene>();
        scene.insert_geometry(GeometryLodChain {
            geometry,
            lods: vec![GeometryLod {
                level: 0,
                screen_error: 0.0,
                resident: true,
                fallback: true,
            }],
        });
        let mut record = fallback_material_record(material, 1);
        record.response_plan = response_plan;
        record.surface.base_color = [1.0; 4];
        scene.insert_material(record);
        assert!(scene.set_material_dynamic_channels(material, dynamic_channels));
        scene.set_execution_tables(execution_tables);
    }

    // The Lit/NPR composition slice needs a nonzero direct-light sample; keep
    // the light in the main-world canonical lighting scene so extraction owns
    // the same path as production content.
    if has_mask_blend {
        app.insert_resource(LightingLightScene::default());
        app.world_mut()
            .resource_mut::<LightingLightScene>()
            .scene
            .insert(
                LightRecord::new(
                    LightClassification {
                        class: LightClass::Directional,
                        mobility: LightMobility::Static,
                    },
                    LightEmission::new([1.0; 3], 1.0, LightUnit::Lux),
                    LightGeometry::Directional {
                        direction: [0.0, 0.0, -1.0],
                    },
                ),
                1,
            )
            .expect("MaskBlend fixture publishes a directional light");
    }

    let camera = app.world_mut().spawn((
        PrismCamera::new(Mat4::IDENTITY, [0, 0, INITIAL_WIDTH, INITIAL_HEIGHT]),
        broad_frustum(),
        GlobalTransform::IDENTITY,
    ));
    let instance = app.world_mut().spawn((
        PrismRenderable::new(geometry, material),
        LocalBounds(Aabb3::from_center_half_extents(
            Vec3::ZERO,
            Vec3::splat(0.5),
        )),
        GlobalTransform::from_transform(&Transform::from_xyz(0.0, 0.0, 0.0)),
        prism_render_app::visibility::PrismViewVisibility::default(),
    ));
    sync_scene(app.world_mut());
    let scene_handle = app
        .world()
        .resource::<PrismRenderScene>()
        .handle_for(instance)
        .expect("the renderable receives a stable native scene identity");
    assert_eq!(scene_handle, SceneHandle::new(1, 1));

    let render = app
        .sub_app_mut(RenderApp)
        .expect("the native render sub-app");
    render
        .world
        .resource_mut::<SceneRegistry>()
        .insert_geometry(GeometryRecord {
            handle: geometry,
            revision: 1,
            vertex_buffer_class: 5,
            index_buffer_class: 0,
            primitive_start: 0,
            primitive_count: 1,
            lods: vec![GeometryLodRecord {
                level: 0,
                primitive_kind: GeometryPrimitiveKind::Indexed,
                element_count: 6,
                first_element: 0,
                base_vertex: 0,
                vertex_count: 4,
                screen_error: 0.0,
                resident: true,
                fallback: false,
            }],
        });
    render
        .world
        .resource_mut::<SceneRegistry>()
        .bind_geometry_mesh(geometry, mesh_id);
    render
        .world
        .resource_mut::<VisibilityWorkList>()
        .set(vec![GpuRenderWorkItem {
            scene: scene_handle,
            geometry,
            material,
            lod_or_cluster: 0,
            pass_mask: RenderPassMask::OPAQUE,
            visibility_stages: VisibilityStageMask::EARLY,
            sort_key: WorkSortKey::new(0, 0, 0, 0, 3),
        }]);

    SceneFixture { camera, material }
}

fn target_snapshot(app: &App) -> (SurfaceViewTargetKey, prism_render_app::RenderTarget) {
    let render = app.get_sub_app(RenderApp).expect("native render sub-app");
    let runtime = render
        .world
        .resource::<RenderRuntimeResource<DelayedInstance>>();
    let targets = render
        .world
        .resource::<SurfaceViewTargets<WgpuDevice>>()
        .iter(runtime.runtime())
        .next()
        .map(|(key, target)| (key, *target));
    targets.expect("camera-local target exists")
}

fn assert_mesh_readback(
    queue: &DelayedQueue,
    target: prism_render_app::RenderTarget,
    width: u32,
    height: u32,
) {
    let normals = queue
        .read_texture_rgba16f(target.normal_texture(), width, height)
        .expect("native resized normal readback must complete");
    let albedo = queue
        .read_texture_rgba8(target.albedo_texture(), width, height)
        .expect("native resized albedo readback must complete");
    let hdr = queue
        .read_texture_rgba16f(target.texture(), width, height)
        .expect("native composite HDR readback must complete");
    let center = ((height / 2) * width + width / 2) as usize;
    assert_eq!(normals[center], [0.0, 0.0, 1.0, 1.0]);
    assert_eq!(&albedo[center * 4..center * 4 + 4], &[255, 255, 255, 255]);
    for (channel, value) in hdr[center].into_iter().enumerate() {
        assert!(
            (value - 1.0).abs() <= 0.002,
            "native composite HDR channel {channel}: expected 1.0, got {value}"
        );
    }
}

fn center_response_readback(
    queue: &DelayedQueue,
    target: prism_render_app::RenderTarget,
    width: u32,
    height: u32,
) -> [f32; 16] {
    let bytes = queue
        .inner
        .read_buffer(
            target.surface_identity_buffer(),
            target.surface_identity_size_bytes(),
        )
        .expect("Native ResponseOutput readback must complete");
    let center = ((height / 2) * width + width / 2) as usize;
    let base = center * 16 * core::mem::size_of::<f32>();
    core::array::from_fn(|index| {
        let offset = base + index * core::mem::size_of::<f32>();
        f32::from_le_bytes(
            bytes[offset..offset + core::mem::size_of::<f32>()]
                .try_into()
                .expect("ResponseOutput word has four bytes"),
        )
    })
}

fn require_adapter() -> Option<(Arc<CompletionGate>, DelayedInstance)> {
    let gate = Arc::new(CompletionGate::default());
    let instance = DelayedInstance::new(Arc::clone(&gate), require_hardware_gpu());
    let adapters = instance.adapters();
    if adapters.is_empty() {
        assert!(
            !require_hardware_gpu(),
            "PRISM_REQUIRE_GPU=1: no physical GPU adapter was found"
        );
        eprintln!("SKIPPED native lifecycle acceptance: no wgpu adapter");
        return None;
    }
    if require_hardware_gpu() {
        assert!(
            adapters.iter().any(|adapter| {
                matches!(
                    adapter.info().device_type,
                    rhi::DeviceType::IntegratedGpu | rhi::DeviceType::DiscreteGpu
                )
            }),
            "PRISM_REQUIRE_GPU=1: no physical GPU adapter was found"
        );
    }
    Some((gate, instance))
}

#[test]
fn native_nonempty_resize_preserves_old_target_until_completion_then_reclaims_it() {
    let Some((gate, instance)) = require_adapter() else {
        return;
    };
    let mut app = App::new();
    app.add_plugins(PrismRenderPlugin::with_config(
        instance,
        RenderAppConfig::new(rhi::Extent3d::new_2d(INITIAL_WIDTH, INITIAL_HEIGHT)),
    ));
    app.add_plugins(PrismVisibilityPlugin);
    let fixture = scene_fixture(&mut app);

    // The execution-table upload is a real queue publication. Warm that cold
    // material generation once with completion open, then close the gate so
    // the frame under test still exercises deferred submission retirement.
    {
        let render = app.get_sub_app(RenderApp).expect("native render sub-app");
        let queue = render
            .world
            .resource::<RenderRuntimeResource<DelayedInstance>>()
            .runtime()
            .queue();
        gate.release(&queue.inner);
    }
    app.update();
    {
        let render = app.get_sub_app(RenderApp).expect("native render sub-app");
        let materials = render.world.resource::<RenderMaterials>();
        assert_eq!(
            materials
                .registry()
                .record_at(fixture.material.index)
                .expect("main-world material extraction must publish the native row")
                .handle,
            fixture.material
        );
        assert_eq!(
            materials
                .registry()
                .dynamic_channels_at(fixture.material.index),
            DynamicChannels {
                wetness: 0.5,
                ..DynamicChannels::NONE
            },
            "dynamic channels must cross the main/render world boundary"
        );
    }
    {
        let render = app.sub_app_mut(RenderApp).expect("native render sub-app");
        assert!(
            render
                .world
                .resource::<RenderRuntimeResource<DelayedInstance>>()
                .runtime()
                .wait_idle(),
            "the cold execution-table upload must complete before the deferred frame"
        );
    }
    gate.block();
    {
        let render = app.sub_app_mut(RenderApp).expect("native render sub-app");
        render.world.insert_resource(RenderFrameStats::default());
        render
            .world
            .insert_resource(NativeFrameExecution::default());
    }

    app.update();
    let (old_key, old_target) = target_snapshot(&app);
    let first_stats = app
        .get_sub_app(RenderApp)
        .expect("native render sub-app")
        .world
        .resource::<RenderFrameStats>();
    assert_eq!(first_stats.draws_queued, 1, "the native mesh was queued");
    assert_eq!(
        first_stats.draws_recorded, 1,
        "the native mesh was recorded"
    );
    let first_lifecycle = app
        .get_sub_app(RenderApp)
        .expect("native render sub-app")
        .world
        .resource::<NativeFrameExecution>();
    assert!(matches!(
        first_lifecycle.phase(),
        NativeFramePhase::Submitted | NativeFramePhase::Completed
    ));
    assert_eq!(first_lifecycle.graph_count(), 1);
    assert!(first_lifecycle.compiled_passes() > 0);
    assert!(first_lifecycle.prepared_passes() > 0);
    assert!(first_lifecycle.recorded_passes() > 0);
    assert_eq!(first_lifecycle.submissions(), 1);
    assert_eq!(old_key.width, INITIAL_WIDTH);
    assert_eq!(old_key.height, INITIAL_HEIGHT);
    assert_eq!(old_target.viewport().extent().width, INITIAL_WIDTH);
    assert_eq!(old_target.viewport().extent().height, INITIAL_HEIGHT);
    let first_queue = app
        .get_sub_app(RenderApp)
        .expect("native render sub-app")
        .world
        .resource::<RenderRuntimeResource<DelayedInstance>>()
        .runtime()
        .queue();
    assert_mesh_readback(&first_queue, old_target, INITIAL_WIDTH, INITIAL_HEIGHT);
    assert!(
        !gate.is_released(),
        "the first submission is still logically in flight"
    );

    app.world_mut()
        .get_mut::<PrismCamera>(fixture.camera)
        .expect("fixture camera")
        .viewport = [0, 0, RESIZED_WIDTH, RESIZED_HEIGHT];
    app.update();

    let render = app.get_sub_app(RenderApp).expect("native render sub-app");
    let blocked_stats = render.world.resource::<RenderFrameStats>();
    assert_eq!(blocked_stats.frames_submitted, 1);
    assert_eq!(blocked_stats.frames_failed, 1);
    assert_eq!(blocked_stats.frames_recorded(), 2);
    let blocked_lifecycle = render.world.resource::<NativeFrameExecution>();
    assert_eq!(blocked_lifecycle.phase(), NativeFramePhase::AdmissionFailed);
    assert_eq!(blocked_lifecycle.graph_count(), 0);
    assert_eq!(blocked_lifecycle.submissions(), 0);
    assert!(blocked_lifecycle.failure().is_some());
    let (blocked_key, blocked_target) = target_snapshot(&app);
    assert_eq!(
        blocked_key, old_key,
        "resize did not publish an unconfirmed target"
    );
    assert_eq!(blocked_target.texture(), old_target.texture());
    assert_eq!(
        render
            .world
            .resource::<SurfaceViewTargets<WgpuDevice>>()
            .quarantined_len(),
        0,
        "the old active target is not moved or reclaimed before completion"
    );
    let blocked_queue = render
        .world
        .resource::<RenderRuntimeResource<DelayedInstance>>()
        .runtime()
        .queue();
    assert_mesh_readback(&blocked_queue, old_target, INITIAL_WIDTH, INITIAL_HEIGHT);

    gate.release(&blocked_queue.inner);
    app.update();

    let (new_key, new_target) = target_snapshot(&app);
    let final_render = app.get_sub_app(RenderApp).expect("native render sub-app");
    let final_stats = final_render.world.resource::<RenderFrameStats>();
    assert_eq!(final_stats.frames_submitted, 2);
    assert_eq!(final_stats.frames_failed, 1);
    // The rejected resize attempt still reaches queue admission, so it is
    // counted here even though graph recording remains fail-closed.
    assert_eq!(final_stats.draws_queued, 3);
    assert_eq!(final_stats.draws_recorded, 2);
    assert!(matches!(
        final_render
            .world
            .resource::<NativeFrameExecution>()
            .phase(),
        NativeFramePhase::Submitted | NativeFramePhase::Completed
    ));
    assert_eq!(new_key.width, RESIZED_WIDTH);
    assert_eq!(new_key.height, RESIZED_HEIGHT);
    assert_eq!(new_target.viewport().extent().width, RESIZED_WIDTH);
    assert_eq!(new_target.viewport().extent().height, RESIZED_HEIGHT);
    assert_ne!(new_target.texture(), old_target.texture());
    let final_queue = final_render
        .world
        .resource::<RenderRuntimeResource<DelayedInstance>>()
        .runtime()
        .queue();
    assert_mesh_readback(&final_queue, new_target, RESIZED_WIDTH, RESIZED_HEIGHT);
    assert!(
        final_queue
            .read_texture_rgba16f(old_target.normal_texture(), INITIAL_WIDTH, INITIAL_HEIGHT)
            .is_err(),
        "the old target texture was reclaimed only after completion"
    );
}

#[test]
fn native_main_world_mask_blend_consumes_dynamic_channel_and_matches_endpoints() {
    let Some((gate, instance)) = require_adapter() else {
        return;
    };
    let mut app = App::new();
    app.add_plugins(PrismRenderPlugin::with_config(
        instance,
        RenderAppConfig::new(rhi::Extent3d::new_2d(INITIAL_WIDTH, INITIAL_HEIGHT)),
    ));
    app.add_plugins(PrismVisibilityPlugin);
    let fixture = scene_fixture_with_material(
        &mut app,
        ShadingPlan::MaskBlend {
            left: ProgramRef::Builtin(BuiltinProgram::Lit),
            right: ProgramRef::Builtin(BuiltinProgram::Npr),
            mask: ResponseMaskSource::Channel("wetness".into()),
            domain: BlendDomain::DirectOnly,
        },
        MaterialExecutionTables::builtin_with_mask_blends(),
        DynamicChannels {
            wetness: 0.5,
            ..DynamicChannels::NONE
        },
    );

    {
        let render = app.get_sub_app(RenderApp).expect("native render sub-app");
        let queue = render
            .world
            .resource::<RenderRuntimeResource<DelayedInstance>>()
            .runtime()
            .queue();
        gate.release(&queue.inner);
    }
    app.update();
    {
        let render = app.get_sub_app(RenderApp).expect("native render sub-app");
        assert!(
            render
                .world
                .resource::<RenderRuntimeResource<DelayedInstance>>()
                .runtime()
                .wait_idle(),
            "initial material/execution publication must complete"
        );
    }
    app.update();

    let sample = |app: &mut App, wetness: f32| {
        assert!(
            app.world_mut()
                .resource_mut::<PrismRenderScene>()
                .set_material_dynamic_channels(
                    fixture.material,
                    DynamicChannels {
                        wetness,
                        ..DynamicChannels::NONE
                    },
                ),
            "the main-world material must accept dynamic-channel updates"
        );
        app.update();
        let (key, target) = target_snapshot(app);
        let render = app.get_sub_app(RenderApp).expect("native render sub-app");
        let queue = render
            .world
            .resource::<RenderRuntimeResource<DelayedInstance>>()
            .runtime()
            .queue();
        let hdr = queue
            .read_texture_rgba16f(target.texture(), key.width, key.height)
            .expect("MaskBlend HDR readback must complete");
        let center = ((key.height / 2) * key.width + key.width / 2) as usize;
        (
            hdr[center],
            center_response_readback(queue, target, key.width, key.height),
        )
    };

    let (left_hdr, left_response) = sample(&mut app, 0.0);
    let (right_hdr, right_response) = sample(&mut app, 1.0);
    let (mid_hdr, mid_response) = sample(&mut app, 0.5);

    {
        let render = app.get_sub_app(RenderApp).expect("native render sub-app");
        let materials = render.world.resource::<RenderMaterials>();
        let record = materials
            .registry()
            .record_at(fixture.material.index)
            .expect("main-world MaskBlend record must be extracted");
        assert!(matches!(
            record.response_plan,
            ShadingPlan::MaskBlend { .. }
        ));
        assert_eq!(
            materials
                .registry()
                .dynamic_channels_at(fixture.material.index),
            DynamicChannels {
                wetness: 0.5,
                ..DynamicChannels::NONE
            }
        );
    }

    for channel in 0..3 {
        let expected_response = (left_response[channel] + right_response[channel]) * 0.5;
        assert!(
            (mid_response[channel] - expected_response).abs() <= 0.003,
            "main-world MaskBlend response lane {channel}: GPU {} != endpoint average {expected_response}",
            mid_response[channel]
        );
        let expected_hdr = (left_hdr[channel] + right_hdr[channel]) * 0.5;
        assert!(
            (mid_hdr[channel] - expected_hdr).abs() <= 0.003,
            "main-world MaskBlend HDR channel {channel}: GPU {} != endpoint average {expected_hdr}",
            mid_hdr[channel]
        );
    }
    for channel in 3..16 {
        assert!(
            (mid_response[channel] - left_response[channel]).abs() <= 0.003,
            "DirectOnly MaskBlend changed unrelated response lane {channel}"
        );
    }
    assert!(
        (left_response[0] - right_response[0]).abs() > 0.005
            || (left_response[1] - right_response[1]).abs() > 0.005
            || (left_response[2] - right_response[2]).abs() > 0.005,
        "Lit and NPR endpoint responses must differ so the dynamic mask is observable"
    );
    assert_eq!(mid_response[15], 1.0);
}
