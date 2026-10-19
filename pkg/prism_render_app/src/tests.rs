//! Tests for the render sub-app seam.
//!
//! The headless tests run on the null backend, so they need no window and no
//! GPU: they drive the full plugin → extract → render-schedule → frame-graph
//! path and assert on what the render world recorded. The wgpu test is gated on
//! a real adapter and reads the rendered frame back, so the same seam is proven
//! on hardware rather than only in the ring buffer.

use prism_app::App;
use prism_ecs::schedule::Schedules;
use prism_render_driver::{
    Adapter, CommandBuffer, Extent3d, ImageDataLayout, RenderQueue, TextureWrite,
};
use prism_render_driver_null::NullInstance;
use prism_render_runtime::RenderRuntime;
use std::collections::BTreeSet;

use crate::config::RenderAppConfig;
use crate::intent::{RenderIntent, RenderViewport};
use crate::label::{Render, RenderApp};
use crate::lighting::{
    evaluate_deterministic_direct, evaluate_deterministic_gi, ExtractedLightingInputs,
    LightingDebugViewState, LightingDirectionalInput, LightingGiCandidateInput, LightingPlanState,
    LightingPunctualInput, LightingSceneInputs, LightingSurfaceInput,
};
use crate::lighting_gpu::LightingGpuState;
use crate::plugin::{save_persisted_pso_cache, PrismRenderPlugin};
use crate::record::RenderRecord;
use crate::runtime_resource::RenderRuntimeResource;
use crate::stats::RenderFrameStats;
use crate::target::RenderTarget;

use crate::bin_queue::bin_indirect_args;
use crate::draw::{DrawFunctionId, DrawFunctions};
use crate::mesh_pipeline::{full_lobe_fallback, MeshPipelineKey, MeshPipelineSpecializer};
use crate::pipeline_cache::{
    hash_vertex_layouts, pso_device_fingerprint, PipelineCache, PipelineCachePersistenceStatus,
    PipelineCompiler, PipelineKey, PipelineResolution,
};
use crate::queue::DrawRequest;
use crate::queue::{phase_for_material, phase_for_render_class};
use crate::queue::{PhaseItem, RenderQueues};
use crate::sets::{RenderPhase, RenderSystems};
use crate::texture_cache::{TextureCache, TextureKey};
use crate::views::RenderViews;
use prism_ecs::prelude::{ResMut, Resource};
use prism_ecs::schedule::IntoSystemConfigs;
use prism_render_architecture::abi::GenerationalHandle;
use prism_render_architecture::gpu_scene::{CpuRenderScene, GeometryHandle, SceneHandle};
use prism_render_architecture::lighting::contract::{
    ChangeFields, FrameResourceBudget, LightingPlan, PlanCaps, RecipeId, Rejection, SceneStats,
    Tier,
};
use prism_render_architecture::lighting::DebugView;
use prism_render_architecture::shader_package::{
    ArtifactDigest, ExecutionFamily, ModuleId, ModuleIdentity, PermutationIndex, PermutationSpace,
    PersistedPsoCache, PipelineStateHash, PsoCacheKey, ShaderPackageId,
};
use prism_render_architecture::view_family::ViewKind;
use prism_render_material::{
    material_pipeline_state, MaterialBlendMode, MaterialDepthTest, MaterialRenderClass,
    MaterialRenderPhase, MaterialRenderTarget, RenderDomain,
};

fn dense(index: u64) -> PermutationIndex {
    PermutationSpace::try_new([64])
        .unwrap()
        .check_index(index)
        .unwrap()
}

#[test]
fn pso_persistence_is_wired_through_startup_explicit_save_and_shutdown() {
    let path = std::env::temp_dir().join(format!("prism-plugin-pso-{}", std::process::id()));
    let adapter = NullInstance::new()
        .request_adapter(&Default::default())
        .expect("the null backend has an adapter");
    let fingerprint = pso_device_fingerprint(adapter.info());
    let mut persisted = PersistedPsoCache::new(fingerprint.clone());
    persisted.insert(
        PsoCacheKey::new(ShaderPackageId::new("plugin-pso"), 4, PipelineStateHash(8)),
        vec![9, 8, 7],
    );
    std::fs::write(&path, persisted.encode()).expect("write startup image");

    let mut app = App::new();
    app.add_plugins(PrismRenderPlugin::new(NullInstance::new()).with_pso_cache_path(path.clone()));
    assert!(app.try_finish(), "the render plugin finishes synchronously");
    let render = app.get_sub_app(RenderApp).expect("render sub-app");
    let status = render.world.resource::<PipelineCachePersistenceStatus>();
    assert_eq!(status.load, Some(Ok(1)));
    assert_eq!(
        render
            .world
            .resource::<MeshPipelineSpecializer>()
            .pending_persisted_pipeline_count(),
        1
    );

    save_persisted_pso_cache::<NullInstance>(&mut app.sub_app_mut(RenderApp).unwrap().world);
    assert_eq!(
        app.get_sub_app(RenderApp)
            .unwrap()
            .world
            .resource::<PipelineCachePersistenceStatus>()
            .save,
        Some(Ok(0))
    );

    app.run_shutdown();
    let bytes = std::fs::read(&path).expect("shutdown wrote destination");
    assert_eq!(
        PersistedPsoCache::load_for(&bytes, &fingerprint)
            .expect("shutdown image remains valid")
            .len(),
        0
    );
    let _ = std::fs::remove_file(path);
}

use prism_render_visibility::GpuRenderWorkItem;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use crate::pipeline_pass::PassRole;
use crate::scene::{SceneRegistry, VisibilityWorkList};
use prism_render_driver::{
    BufferDescriptor, BufferId, BufferUsages, ColorTargetState, DeviceDescriptor, FragmentState,
    Instance, MultisampleState, PrimitiveState, PrimitiveTopology, RenderDevice,
    RenderPipelineDescriptor, RenderPipelineId, RequestAdapterOptions, ShaderModuleDescriptor,
    TextureDescriptor, TextureFormat, TextureUsages, VertexState,
};

/// A deterministic compiler gate for exercising the asynchronous cache without
/// racing the real worker.
struct GateCompiler {
    open: Mutex<bool>,
    wake: Condvar,
    result: RenderPipelineId,
    calls: AtomicUsize,
}

impl GateCompiler {
    fn new(result: RenderPipelineId) -> Arc<Self> {
        Arc::new(Self {
            open: Mutex::new(false),
            wake: Condvar::new(),
            result,
            calls: AtomicUsize::new(0),
        })
    }

    fn release(&self) {
        *self.open.lock().expect("gate compiler lock") = true;
        self.wake.notify_all();
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl PipelineCompiler for GateCompiler {
    fn compile(
        &self,
        _key: PipelineKey,
        _descriptor: &RenderPipelineDescriptor,
    ) -> RenderPipelineId {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let mut open = self.open.lock().expect("gate compiler lock");
        while !*open {
            open = self.wake.wait(open).expect("gate compiler wait");
        }
        self.result
    }
}

/// A prepared, position-only mesh bound to `geometry`, plus the table holding it.
///
/// Buffers belong to the geometry's own mesh (a vertex class is a vertex stage),
/// so a test that wants a drawable geometry needs a prepared mesh for it — which
/// is exactly what the frame's asset pipeline produces.
fn bind_prepared_mesh(
    registry: &mut SceneRegistry,
    device: &prism_render_driver_null::NullDevice,
    queue: &impl RenderQueue,
    geometry: GenerationalHandle,
    mesh_id: prism_asset::AssetId<prism_mesh::Mesh>,
    meshes: &mut crate::RenderAssets<crate::RenderMesh>,
) {
    let mesh = prism_mesh::Mesh::new(PrimitiveTopology::TriangleList)
        .with_attribute(
            prism_mesh::MeshVertexAttribute::new(
                "Vertex_Position",
                prism_render_driver::VertexFormat::Float32x3,
                0,
            ),
            prism_mesh::VertexAttributeValues::Float32x3(vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]),
        )
        .expect("positions fit")
        .indexed(prism_mesh::Indices::Uint16(vec![0, 1, 2, 0, 2, 3]));
    use crate::render_asset::RenderAsset;
    let prepared = crate::RenderMesh::prepare(&mesh, mesh_id, device, queue, None)
        .expect("the test mesh prepares");
    meshes.insert(mesh_id, prepared);
    registry.bind_geometry_mesh(geometry, mesh_id);
}

/// A dummy vertex buffer id from the null device, for items that only need an id.
fn null_vertex_buffer(device: &prism_render_driver_null::NullDevice) -> BufferId {
    device.create_buffer(&BufferDescriptor {
        label: Some("test-vertices".into()),
        size: 64,
        usage: BufferUsages::VERTEX,
        mapped_at_creation: false,
    })
}

/// A mesh pipeline provider that hands every variant the same trivial
/// descriptor, so a test observes the *keys* the queue specializes rather than
/// the shader contents.
struct StubProvider {
    descriptor: RenderPipelineDescriptor,
}

impl StubProvider {
    fn new(device: &prism_render_driver_null::NullDevice) -> Self {
        let module = device.create_shader_module(&ShaderModuleDescriptor::wgsl(
            "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }",
        ));
        Self {
            descriptor: RenderPipelineDescriptor {
                label: Some("stub".into()),
                layout: None,
                vertex: VertexState {
                    module,
                    entry_point: "vs".into(),
                    buffers: vec![],
                },
                primitive: PrimitiveState::default(),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                fragment: None,
            },
        }
    }
}

impl crate::mesh_pipeline::MeshPipelineProvider for StubProvider {
    fn descriptor(&self, _: &MeshPipelineKey) -> RenderPipelineDescriptor {
        self.descriptor.clone()
    }
}

/// The pipeline cache builds a PSO once per content key, shares it, and destroys
/// it on the last release (or on clear).
#[test]
fn pipeline_cache_dedups_and_evicts_by_key() {
    let options = RequestAdapterOptions::default();
    let adapter = NullInstance::new()
        .request_adapter(&options)
        .expect("the null backend always has an adapter");
    let (device, _queue) = NullInstance::new()
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null device always comes up");

    let module = device.create_shader_module(&ShaderModuleDescriptor::wgsl(
        "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }",
    ));
    let mut descriptor = RenderPipelineDescriptor {
        label: Some("cache-test".into()),
        layout: None,
        vertex: VertexState {
            module,
            entry_point: "vs".into(),
            buffers: vec![],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: None,
    };

    let layout = prism_render_driver::VertexBufferLayout::packed(
        prism_render_driver::VertexStepMode::Vertex,
        0,
        &[
            prism_render_driver::VertexFormat::Float32x3,
            prism_render_driver::VertexFormat::Float32x2,
        ],
    );
    let layout_hash = hash_vertex_layouts(core::slice::from_ref(&layout));
    descriptor.vertex.buffers = vec![layout.clone()];

    let mut cache = PipelineCache::new();
    let key = PipelineKey::from_descriptor(
        ExecutionFamily::FusedBuiltin,
        ModuleId(0xABCD),
        dense(7),
        &descriptor,
    );
    let first = cache.get_or_create(&device, key.clone(), &descriptor);
    let second = cache.get_or_create(&device, key.clone(), &descriptor);
    assert_eq!(first, second, "the same key returns the same pipeline");
    assert_eq!(cache.len(), 1, "the pipeline is built once");

    let other = PipelineKey::from_descriptor(
        ExecutionFamily::FusedBuiltin,
        ModuleId(0xABCD),
        dense(8),
        &descriptor,
    );
    let distinct = cache.get_or_create(&device, other, &descriptor);
    assert_ne!(distinct, first, "a different key compiles a new pipeline");
    assert_eq!(cache.len(), 2);

    // The key must separate pipelines that differ only in topology or shader:
    // keying on neither aliased two different PSOs onto one cache entry.
    let mut strip_descriptor = descriptor.clone();
    strip_descriptor.primitive.topology = PrimitiveTopology::TriangleStrip;
    strip_descriptor.primitive.strip_index_format = Some(prism_render_driver::IndexFormat::Uint16);
    let other_topology = PipelineKey::from_descriptor(
        ExecutionFamily::FusedBuiltin,
        ModuleId(0xABCD),
        dense(7),
        &strip_descriptor,
    );
    assert_ne!(other_topology, key, "topology is part of the identity");
    let other_shader = PipelineKey::from_descriptor(
        ExecutionFamily::FusedBuiltin,
        ModuleId(0x1234),
        dense(7),
        &descriptor,
    );
    assert_ne!(other_shader, key, "the shader identity is part of the key");

    // A layout with the same total stride but different attributes must not
    // hash equal: a single stride would alias these two vertex stages.
    let same_stride_other_attrs = prism_render_driver::VertexBufferLayout::packed(
        prism_render_driver::VertexStepMode::Vertex,
        0,
        &[
            prism_render_driver::VertexFormat::Float32x2,
            prism_render_driver::VertexFormat::Float32x3,
        ],
    );
    assert_eq!(
        same_stride_other_attrs.array_stride, layout.array_stride,
        "the layouts share a stride"
    );
    assert_ne!(
        hash_vertex_layouts(core::slice::from_ref(&same_stride_other_attrs)),
        layout_hash,
        "different attributes must hash differently"
    );
    assert_eq!(
        hash_vertex_layouts(core::slice::from_ref(&layout)),
        layout_hash,
        "the vertex layout hash is deterministic"
    );

    assert!(
        !cache.release(&device, &key),
        "one of the two references still holds the pipeline"
    );
    assert!(cache.release(&device, &key), "the last release evicts");
    assert_eq!(cache.len(), 1);
    assert!(!cache.release(&device, &key), "an absent key is a no-op");

    cache.clear(&device);
    assert!(cache.is_empty(), "clear destroys every cached pipeline");
}

/// The texture cache uploads a texture once per key, shares it, and destroys it
/// on the last release (or on clear).
#[test]
fn texture_cache_dedups_and_evicts_by_key() {
    let options = RequestAdapterOptions::default();
    let adapter = NullInstance::new()
        .request_adapter(&options)
        .expect("the null backend always has an adapter");
    let (device, queue) = NullInstance::new()
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null device always comes up");

    let descriptor = TextureDescriptor::new_2d(
        Extent3d::new_2d(2, 2),
        TextureFormat::Rgba8Unorm,
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
    );
    let pixels = [0u8; 2 * 2 * 4];
    let layout = ImageDataLayout {
        offset: 0,
        bytes_per_row: Some(8),
        rows_per_image: Some(2),
    };

    let mut cache = TextureCache::new();
    let a = cache.get_or_create(&device, &queue, TextureKey(1), &descriptor, &pixels, layout);
    let b = cache.get_or_create(&device, &queue, TextureKey(1), &descriptor, &pixels, layout);
    assert_eq!(a, b, "the same key returns the same texture");
    assert_eq!(cache.len(), 1, "the texture is uploaded once");

    cache.get_or_create(&device, &queue, TextureKey(2), &descriptor, &[], layout);
    assert_eq!(cache.len(), 2, "a different key creates a new texture");

    assert!(
        !cache.release(&device, &TextureKey(1)),
        "one of the two references still holds the texture"
    );
    assert!(
        cache.release(&device, &TextureKey(1)),
        "the last release evicts"
    );
    assert_eq!(cache.len(), 1);

    cache.clear(&device);
    assert!(cache.is_empty(), "clear destroys every cached texture");
}

/// The mesh queue separates phases and sorts each one deterministically.
#[test]
fn render_queues_are_phased_and_sort_deterministically() {
    let options = RequestAdapterOptions::default();
    let adapter = NullInstance::new()
        .request_adapter(&options)
        .expect("the null backend always has an adapter");
    let (device, _queue) = NullInstance::new()
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null device always comes up");
    let module = device.create_shader_module(&ShaderModuleDescriptor::wgsl(
        "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }",
    ));
    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("queue-test".into()),
        layout: None,
        vertex: VertexState {
            module,
            entry_point: "vs".into(),
            buffers: vec![],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: None,
    });

    let geometry = GenerationalHandle::new(2, 1);
    let material = GenerationalHandle::new(3, 1);
    let scene = GenerationalHandle::new(4, 1);
    let vertices = null_vertex_buffer(&device);
    let item = |order: u64| PhaseItem {
        pipeline,
        draw: DrawFunctionId(0),
        geometry,
        vertices: vec![vertices],
        indices: None,
        first: 0,
        count: 3,
        base_vertex: 0,
        indirect: None,
        lod_level: 0,
        material,
        scene,
        order,
    };

    let mut queues = RenderQueues::new();
    // Pushed out of order on purpose: `sort` must impose the order.
    queues.push(RenderPhase::Opaque, item(5));
    queues.push(RenderPhase::Opaque, item(1));
    queues.push(RenderPhase::Transparent, item(0));

    assert_eq!(queues.phase(RenderPhase::Opaque).len(), 2);
    assert!(
        queues.phase(RenderPhase::Stylized).is_empty(),
        "phases are isolated"
    );
    assert_eq!(queues.len(), 3);

    queues.sort();
    let orders: Vec<u64> = queues
        .phase(RenderPhase::Opaque)
        .iter()
        .map(|item| item.order)
        .collect();
    assert_eq!(orders, [1, 5], "the phase sorts by its order key");

    queues.clear();
    assert!(queues.is_empty(), "clear drops every queued item");
}

/// Recording a phase invokes each item's registered draw function, in queue
/// order.
#[test]
fn recording_a_phase_invokes_each_draw_function() {
    let options = RequestAdapterOptions::default();
    let adapter = NullInstance::new()
        .request_adapter(&options)
        .expect("the null backend always has an adapter");
    let (device, _queue) = NullInstance::new()
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null device always comes up");
    let module = device.create_shader_module(&ShaderModuleDescriptor::wgsl(
        "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }",
    ));
    let make_pipeline = |label: &str| {
        device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some(label.into()),
            layout: None,
            vertex: VertexState {
                module,
                entry_point: "vs".into(),
                buffers: vec![],
            },
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: None,
        })
    };
    let pipeline_a = make_pipeline("record-a");
    let pipeline_b = make_pipeline("record-b");

    // The recording context logs the pipeline each item binds, so the recorded
    // order is observable without a GPU.
    #[derive(Default)]
    struct PipelineLog(Vec<prism_render_driver::RenderPipelineId>);
    impl crate::MeshRecorder for PipelineLog {
        fn set_pipeline(&mut self, pipeline: prism_render_driver::RenderPipelineId) {
            self.0.push(pipeline);
        }
        fn set_vertex_buffer(&mut self, _: u32, _: BufferId, _: u64) {}
        fn set_index_buffer(&mut self, _: BufferId, _: prism_render_driver::IndexFormat, _: u64) {}
        fn draw(&mut self, _: core::ops::Range<u32>, _: core::ops::Range<u32>) {}
        fn draw_indexed(&mut self, _: core::ops::Range<u32>, _: i32, _: core::ops::Range<u32>) {}
        fn draw_indexed_indirect(&mut self, _: prism_render_graph::BufferHandle, _: u64) {}
        fn multi_draw_indirect(
            &mut self,
            _: prism_render_graph::BufferHandle,
            _: u64,
            _: u32,
            _: u32,
        ) {
        }
    }

    let mut functions = DrawFunctions::new();
    let draw = functions.register(|item, ctx| ctx.set_pipeline(item.pipeline));

    let geometry = GenerationalHandle::new(2, 1);
    let material = GenerationalHandle::new(3, 1);
    let scene = GenerationalHandle::new(4, 1);
    let vertices = null_vertex_buffer(&device);
    let mut queues = RenderQueues::new();
    for (pipeline, order) in [(pipeline_a, 2u64), (pipeline_b, 1)] {
        queues.push(
            RenderPhase::Opaque,
            PhaseItem {
                pipeline,
                draw,
                geometry,
                vertices: vec![vertices],
                indices: None,
                first: 0,
                count: 3,
                base_vertex: 0,
                indirect: None,
                lod_level: 0,
                material,
                scene,
                order,
            },
        );
    }
    queues.sort();

    let mut recorded = PipelineLog::default();
    queues.record(RenderPhase::Opaque, &mut recorded, &functions);
    assert_eq!(
        recorded.0,
        [pipeline_b, pipeline_a],
        "items record in sorted order (order 1 before order 2)"
    );

    // An unknown id is a no-op, never a panic.
    functions.call(
        DrawFunctionId(99),
        &queues.phase(RenderPhase::Opaque)[0],
        &mut recorded,
    );
    assert_eq!(recorded.0.len(), 2, "an unknown id records nothing");
}

/// The queue stage end to end: visible work items resolve, specialize, and
/// land in their phase; a work item with no resident material is skipped.
#[test]
fn fill_from_visibility_resolves_and_queues() {
    use crate::mesh_pipeline::MeshPipelineProvider;
    use crate::scene::{MeshClass, SceneRegistry};
    use prism_render_architecture::geometry::{
        GeometryLodRecord, GeometryPrimitiveKind, GeometryRecord,
    };
    use prism_render_visibility::{GpuRenderWorkItem, WorkSortKey};

    let options = RequestAdapterOptions::default();
    let adapter = NullInstance::new()
        .request_adapter(&options)
        .expect("the null backend always has an adapter");
    let (device, queue) = NullInstance::new()
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null device always comes up");
    let module = device.create_shader_module(&ShaderModuleDescriptor::wgsl(
        "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }",
    ));

    struct Stub {
        descriptor: RenderPipelineDescriptor,
    }
    impl MeshPipelineProvider for Stub {
        fn descriptor(&self, _: &MeshPipelineKey) -> RenderPipelineDescriptor {
            self.descriptor.clone()
        }
    }
    let provider = Stub {
        descriptor: RenderPipelineDescriptor {
            label: Some("stub".into()),
            layout: None,
            vertex: VertexState {
                module,
                entry_point: "vs".into(),
                buffers: vec![],
            },
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: None,
        },
    };

    let geometry = GenerationalHandle::new(1, 1);
    let material = GenerationalHandle::new(2, 1);
    let scene = GenerationalHandle::new(3, 1);
    let layout = prism_render_driver::VertexBufferLayout::packed(
        prism_render_driver::VertexStepMode::Vertex,
        0,
        &[prism_render_driver::VertexFormat::Float32x3],
    );

    let mut registry = SceneRegistry::new();
    registry.insert_geometry(GeometryRecord {
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
    registry.insert_material(prism_render_material::fallback_material_record(material, 1));
    registry
        .set_material_passes(
            material,
            vec![crate::PassDescriptor {
                role: PassRole::Surface,
                execution_family: ExecutionFamily::FusedBuiltin,
                response_module_id: ModuleId(1),
                permutation_index: dense(1),
                fallback_permutation: Some(dense(63)),
                render_phase: RenderPhase::Opaque,
                draw: None,
                sort_class: crate::PassDescriptorClass::SURFACE,
            }],
        )
        .unwrap();
    registry.insert_mesh_class(
        5,
        MeshClass::new(vec![layout], PrimitiveTopology::TriangleList),
    );
    let mut meshes = crate::RenderAssets::<crate::RenderMesh>::new();
    bind_prepared_mesh(
        &mut registry,
        &device,
        &queue,
        geometry,
        prism_asset::AssetId::new(prism_asset::AssetIndex::from_parts(0, 1)),
        &mut meshes,
    );

    let item = |material: GenerationalHandle| GpuRenderWorkItem {
        scene,
        geometry,
        material,
        lod_or_cluster: 0,
        pass_mask: Default::default(),
        visibility_stages: Default::default(),
        sort_key: WorkSortKey::new(0, 0, 0, 0, 3),
    };
    let work = [item(material), item(GenerationalHandle::new(9, 1))];

    let mut queues = RenderQueues::new();
    let mut specializer = MeshPipelineSpecializer::new();
    let queued = queues.fill_from_visibility(
        &device,
        &mut specializer,
        &registry,
        &meshes,
        &work,
        None,
        1,
        DrawFunctionId(0),
        &provider,
    );

    assert_eq!(queued, 1, "the non-resident material is skipped");
    assert_eq!(
        queues.phase(RenderPhase::Opaque).len(),
        1,
        "the draw lands in the opaque phase"
    );
    assert_eq!(
        specializer.pipeline_count(),
        2,
        "one specialization plus its full-lobe fallback"
    );
}

/// The scene registry resolves a visibility work item into a queueable draw,
/// and skips one whose material is not resident.
#[test]
fn scene_registry_resolves_a_work_item() {
    use crate::scene::{MeshClass, SceneRegistry};
    use prism_render_architecture::geometry::{
        GeometryLodRecord, GeometryPrimitiveKind, GeometryRecord,
    };
    use prism_render_visibility::{GpuRenderWorkItem, WorkSortKey};

    let options = RequestAdapterOptions::default();
    let adapter = NullInstance::new()
        .request_adapter(&options)
        .expect("the null backend always has an adapter");
    let (device, queue) = NullInstance::new()
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null device always comes up");

    let geometry = GenerationalHandle::new(1, 1);
    let material = GenerationalHandle::new(2, 1);
    let scene = GenerationalHandle::new(3, 1);
    let layout = prism_render_driver::VertexBufferLayout::packed(
        prism_render_driver::VertexStepMode::Vertex,
        0,
        &[prism_render_driver::VertexFormat::Float32x3],
    );

    let mut registry = SceneRegistry::new();
    registry.insert_geometry(GeometryRecord {
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
    registry.insert_material(prism_render_material::fallback_material_record(material, 1));
    registry.insert_mesh_class(
        5,
        MeshClass::new(vec![layout.clone()], PrimitiveTopology::TriangleList),
    );
    let mut meshes = crate::RenderAssets::<crate::RenderMesh>::new();
    bind_prepared_mesh(
        &mut registry,
        &device,
        &queue,
        geometry,
        prism_asset::AssetId::new(prism_asset::AssetIndex::from_parts(0, 1)),
        &mut meshes,
    );

    let item = GpuRenderWorkItem {
        scene,
        geometry,
        material,
        lod_or_cluster: 0,
        pass_mask: Default::default(),
        visibility_stages: Default::default(),
        sort_key: WorkSortKey::new(0, 0, 0, 0, 7),
    };
    let draw = registry
        .resolve(&item, &meshes, None, 1, DrawFunctionId(0))
        .expect("a resident geometry + material resolves");
    assert_eq!(
        draw.phase,
        RenderPhase::Opaque,
        "the opaque family phases here"
    );
    assert_eq!(draw.order, 7, "the work item's sort key carries through");
    assert_eq!(draw.lod_level, 0, "the resolved LOD level is carried");
    assert_eq!(draw.variant.topology, PrimitiveTopology::TriangleList);
    assert_eq!(draw.layout, vec![layout], "the class's layout is used");

    let missing_material = GpuRenderWorkItem {
        material: GenerationalHandle::new(9, 1),
        ..item
    };
    assert!(
        registry
            .resolve(&missing_material, &meshes, None, 1, DrawFunctionId(0))
            .is_none(),
        "a non-resident material is skipped, not an error"
    );
}

/// Filling the queue specializes each resolved draw and appends it to its phase.
#[test]
fn filling_the_queue_specializes_and_appends() {
    let options = RequestAdapterOptions::default();
    let adapter = NullInstance::new()
        .request_adapter(&options)
        .expect("the null backend always has an adapter");
    let (device, _queue) = NullInstance::new()
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null device always comes up");
    let module = device.create_shader_module(&ShaderModuleDescriptor::wgsl(
        "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }",
    ));
    let descriptor = RenderPipelineDescriptor {
        label: Some("fill-test".into()),
        layout: None,
        vertex: VertexState {
            module,
            entry_point: "vs".into(),
            buffers: vec![],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: None,
    };

    let variant = MeshPipelineKey {
        execution_family: ExecutionFamily::FusedBuiltin,
        response_module_id: ModuleId(9),
        permutation_index: dense(1),
        fallback_permutation: Some(dense(63)),
        color_format: Some(TextureFormat::Rgba8Unorm),
        depth_format: None,
        sample_count: 1,
        topology: PrimitiveTopology::TriangleList,
    };
    let layout_a = prism_render_driver::VertexBufferLayout::packed(
        prism_render_driver::VertexStepMode::Vertex,
        0,
        &[prism_render_driver::VertexFormat::Float32x3],
    );
    let layout_b = prism_render_driver::VertexBufferLayout::packed(
        prism_render_driver::VertexStepMode::Vertex,
        0,
        &[prism_render_driver::VertexFormat::Float32x2],
    );
    let geometry = GenerationalHandle::new(2, 1);
    let material = GenerationalHandle::new(3, 1);
    let scene = GenerationalHandle::new(4, 1);
    let vertices = null_vertex_buffer(&device);
    let request = |layout: prism_render_driver::VertexBufferLayout, order: u64| DrawRequest {
        phase: RenderPhase::Opaque,
        variant,
        layout: vec![layout],
        vertices: vec![vertices],
        indices: None,
        first: 0,
        count: 3,
        base_vertex: 0,
        lod_level: 0,
        geometry,
        material,
        scene,
        order,
        draw: DrawFunctionId(0),
    };

    let mut specializer = MeshPipelineSpecializer::new();
    let mut queues = RenderQueues::new();
    queues.fill(
        &device,
        &mut specializer,
        [
            request(layout_a.clone(), 0),
            request(layout_a, 1),
            request(layout_b, 2),
        ],
        |_| descriptor.clone(),
    );

    assert_eq!(
        queues.phase(RenderPhase::Opaque).len(),
        3,
        "every draw queues"
    );
    assert_eq!(
        specializer.pipeline_count(),
        4,
        "two distinct layouts specialize two pipelines plus their fallbacks"
    );
}

/// A bin class indexes the same explicit pass table as direct resolution.
#[test]
fn phase_for_bin_class_round_trips_the_render_class() {
    let mut registry = SceneRegistry::new();
    for (index, (domain, class, phase)) in [
        (
            RenderDomain::Surface,
            MaterialRenderClass::Opaque,
            RenderPhase::Opaque,
        ),
        (
            RenderDomain::Hair,
            MaterialRenderClass::Opaque,
            RenderPhase::Opaque,
        ),
        (
            RenderDomain::Surface,
            MaterialRenderClass::Transparent,
            RenderPhase::Transparent,
        ),
        (
            RenderDomain::Volume,
            MaterialRenderClass::Opaque,
            RenderPhase::Custom,
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let handle = GenerationalHandle::new(index as u32, 1);
        let mut record = prism_render_material::fallback_material_record(handle, 1);
        record.render_domain = domain;
        record.render_class = class;
        registry.insert_material(record);
        let table = registry.material_pass_index(handle).unwrap();
        assert_eq!(crate::phase_for_bin_class(&registry, table), Some(phase));
    }
    assert_eq!(crate::phase_for_bin_class(&registry, u32::MAX), None);
}

/// Surface draws are phased by blend family only — the illumination axis is a
/// response, not a phase.
#[test]
fn material_render_class_selects_the_phase() {
    use MaterialRenderClass::{
        Additive, Masked, MaskedTwoSided, Opaque, OpaqueTwoSided, Transmissive, Transparent,
    };
    for class in [Opaque, OpaqueTwoSided, Masked, MaskedTwoSided] {
        assert_eq!(
            phase_for_render_class(class),
            RenderPhase::Opaque,
            "{class:?} is opaque-family"
        );
    }
    for class in [Transmissive, Transparent, Additive] {
        assert_eq!(
            phase_for_render_class(class),
            RenderPhase::Transparent,
            "{class:?} is transparent-family"
        );
    }
    let mut water =
        prism_render_material::fallback_material_record(GenerationalHandle::new(90, 1), 1);
    water.features |= prism_render_material::MaterialFeatureFlags::WATER;
    assert_eq!(phase_for_material(&water), RenderPhase::Transparent);
    let water_state = material_pipeline_state(&water);
    assert_eq!(water_state.phase, MaterialRenderPhase::Transparent);
    assert_eq!(water_state.target, MaterialRenderTarget::HdrTransparent);
    assert_eq!(water_state.blend, MaterialBlendMode::PremultipliedAlpha);
    assert_eq!(water_state.depth_test, MaterialDepthTest::LessEqual);
    assert!(!water_state.depth_write);
    water.render_domain = RenderDomain::Hair;
    water.features = prism_render_material::MaterialFeatureFlags::default();
    assert_eq!(phase_for_material(&water), RenderPhase::Opaque);
    water.render_domain = RenderDomain::Volume;
    assert_eq!(phase_for_material(&water), RenderPhase::Custom);
}

/// Mesh specialization builds one pipeline per `(variant, layout)` pair and
/// reuses it.
#[test]
fn mesh_specialization_dedups_by_variant_and_layout() {
    let options = RequestAdapterOptions::default();
    let adapter = NullInstance::new()
        .request_adapter(&options)
        .expect("the null backend always has an adapter");
    let (device, _queue) = NullInstance::new()
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null device always comes up");
    let module = device.create_shader_module(&ShaderModuleDescriptor::wgsl(
        "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }",
    ));
    let descriptor = RenderPipelineDescriptor {
        label: Some("mesh-specialize".into()),
        layout: None,
        vertex: VertexState {
            module,
            entry_point: "vs".into(),
            buffers: vec![],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: None,
    };

    let variant = MeshPipelineKey {
        execution_family: ExecutionFamily::FusedBuiltin,
        response_module_id: ModuleId(0x55),
        permutation_index: dense(3),
        fallback_permutation: None,
        color_format: Some(TextureFormat::Rgba8Unorm),
        depth_format: None,
        sample_count: 1,
        topology: PrimitiveTopology::TriangleList,
    };
    let layout_a = prism_render_driver::VertexBufferLayout::packed(
        prism_render_driver::VertexStepMode::Vertex,
        0,
        &[prism_render_driver::VertexFormat::Float32x3],
    );
    let layout_b = prism_render_driver::VertexBufferLayout::packed(
        prism_render_driver::VertexStepMode::Vertex,
        0,
        &[prism_render_driver::VertexFormat::Float32x2],
    );

    let mut specializer = MeshPipelineSpecializer::new();
    let a1 = specializer.specialize(
        &device,
        variant,
        core::slice::from_ref(&layout_a),
        descriptor.clone(),
    );
    let a2 = specializer.specialize(
        &device,
        variant,
        core::slice::from_ref(&layout_a),
        descriptor.clone(),
    );
    assert_eq!(a1, a2, "the same variant + layout reuses the pipeline");
    assert_eq!(specializer.pipeline_count(), 1);

    let b = specializer.specialize(
        &device,
        variant,
        core::slice::from_ref(&layout_b),
        descriptor.clone(),
    );
    assert_ne!(a1, b, "a different layout specializes a new pipeline");
    assert_eq!(specializer.pipeline_count(), 2);

    assert!(specializer.release(&device, variant, core::slice::from_ref(&layout_b)));
    assert_eq!(specializer.pipeline_count(), 1);
    specializer.clear(&device);
    assert_eq!(specializer.pipeline_count(), 0);
}

/// Fallback is declared by the owning package, never inferred from mask bits.
#[test]
fn full_lobe_fallback_uses_the_declared_dense_index() {
    let variant = MeshPipelineKey {
        execution_family: ExecutionFamily::SplitResponse,
        response_module_id: ModuleId(17),
        permutation_index: dense(4),
        fallback_permutation: Some(dense(63)),
        color_format: Some(TextureFormat::Rgba8Unorm),
        depth_format: None,
        sample_count: 1,
        topology: PrimitiveTopology::TriangleList,
    };
    let fallback = full_lobe_fallback(variant).unwrap();
    assert_eq!(fallback.permutation_index, dense(63));
    assert_eq!(fallback.response_module_id, variant.response_module_id);
    assert_eq!(fallback.execution_family, variant.execution_family);
    assert_eq!(fallback.color_format, variant.color_format);
    assert!(full_lobe_fallback(MeshPipelineKey {
        fallback_permutation: None,
        ..variant
    })
    .is_none());
}

#[test]
fn response_modules_are_interned_dense_checked_and_composition_resident() {
    use crate::mesh_pipelines::{
        ResponseModuleCache, ResponseModuleCacheError, ResponseModuleSpec,
    };

    let adapter = NullInstance::new()
        .request_adapter(&RequestAdapterOptions::default())
        .expect("the null backend always has an adapter");
    let (device, _queue) = NullInstance::new()
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null device always comes up");
    let space = PermutationSpace::try_new([2]).unwrap();
    let descriptor = ShaderModuleDescriptor::wgsl(
        "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }",
    );
    let spec = ResponseModuleSpec {
        execution_family: ExecutionFamily::SplitResponse,
        identity: ModuleIdentity {
            package: ShaderPackageId::new("app.response.test"),
            digest: ArtifactDigest([9; 32]),
            entry_point: "vs".into(),
            binding_layout_digest: [7; 32],
        },
        permutations: space.clone(),
        full_lobe_fallback: Some(space.check_index(1).unwrap()),
        descriptor: descriptor.clone(),
        builtin: false,
    };
    let mut modules = ResponseModuleCache::new();
    let first = modules.intern(&device, spec.clone()).unwrap();
    assert_eq!(modules.intern(&device, spec.clone()), Ok(first));
    assert_eq!(
        modules.len(),
        1,
        "same package/module identity shares one RHI module"
    );
    let composition = modules.ensure_composition_kernel(&device).unwrap();
    assert_ne!(composition, first);
    assert!(modules.composition_kernel().is_some());
    assert_eq!(
        modules.len(),
        2,
        "composition is a resident module, not a per-material package"
    );

    let mut fused_custom = spec.clone();
    fused_custom.execution_family = ExecutionFamily::FusedBuiltin;
    assert_eq!(
        modules.intern(&device, fused_custom),
        Err(ResponseModuleCacheError::FusedCustom)
    );
    let mut changed_execution = spec.clone();
    changed_execution.execution_family = ExecutionFamily::Composition;
    changed_execution.builtin = true;
    assert_eq!(
        modules.intern(&device, changed_execution),
        Err(ResponseModuleCacheError::ExecutionDeclarationChanged(first))
    );

    let mut specializer = MeshPipelineSpecializer::new();
    specializer.sync_response_modules(&modules);
    let provider = StubProvider::new(&device);
    let variant = MeshPipelineKey {
        execution_family: ExecutionFamily::SplitResponse,
        response_module_id: first,
        permutation_index: space.check_index(1).unwrap(),
        fallback_permutation: Some(space.check_index(1).unwrap()),
        color_format: None,
        depth_format: None,
        sample_count: 1,
        topology: PrimitiveTopology::TriangleList,
    };
    assert!(specializer
        .specialize_with_provider(&device, variant, &[], &provider)
        .is_ready());
    let invalid = MeshPipelineKey {
        permutation_index: dense(2),
        ..variant
    };
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        specializer.specialize_with_provider(&device, invalid, &[], &provider);
    }))
    .is_err());
}

/// A cold specialization resolves to the über fallback immediately; the worker
/// result then replaces it on the next lookup.
#[test]
fn async_pipeline_cache_serves_fallback_then_switches() {
    let options = RequestAdapterOptions::default();
    let adapter = NullInstance::new()
        .request_adapter(&options)
        .expect("the null backend always has an adapter");
    let (device, _queue) = NullInstance::new()
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null device always comes up");
    let module = device.create_shader_module(&ShaderModuleDescriptor::wgsl(
        "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }",
    ));
    let mut descriptor = RenderPipelineDescriptor {
        label: Some("async-cache-test".into()),
        layout: None,
        vertex: VertexState {
            module,
            entry_point: "vs".into(),
            buffers: vec![],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: None,
    };
    let layout = prism_render_driver::VertexBufferLayout::packed(
        prism_render_driver::VertexStepMode::Vertex,
        0,
        &[prism_render_driver::VertexFormat::Float32x3],
    );
    let layout = core::slice::from_ref(&layout);

    let variant = MeshPipelineKey {
        execution_family: ExecutionFamily::FusedBuiltin,
        response_module_id: ModuleId(1),
        permutation_index: dense(1),
        fallback_permutation: Some(dense(63)),
        color_format: Some(TextureFormat::Rgba8Unorm),
        depth_format: None,
        sample_count: 1,
        topology: PrimitiveTopology::TriangleList,
    };
    let fallback = full_lobe_fallback(variant).expect("fallback variant");
    descriptor.vertex.buffers = layout.to_vec();
    let target_key = PipelineKey::from_descriptor(
        variant.execution_family,
        variant.response_module_id,
        variant.permutation_index,
        &descriptor,
    );
    let fallback_key = PipelineKey::from_descriptor(
        fallback.execution_family,
        fallback.response_module_id,
        fallback.permutation_index,
        &descriptor,
    );
    assert_ne!(target_key, fallback_key, "the test uses a partial lobe set");

    let compiler = GateCompiler::new(RenderPipelineId::from_parts(101, 1));
    let mut cache = PipelineCache::with_compiler(compiler.clone());
    let first = cache.resolve_or_request(
        &device,
        target_key.clone(),
        descriptor.clone(),
        fallback_key.clone(),
        descriptor.clone(),
    );
    assert!(
        matches!(first, PipelineResolution::Fallback(_)),
        "a cold permutation must use the fallback immediately"
    );
    assert_eq!(cache.pending_len(), 1);
    compiler.release();
    for _ in 0..10_000 {
        cache.pump(&device);
        if cache.pending_len() == 0 {
            break;
        }
        std::thread::yield_now();
    }
    assert_eq!(cache.pending_len(), 0);
    assert_eq!(compiler.calls(), 1, "the worker compiles the key once");
    let second = cache.resolve_or_request(
        &device,
        target_key,
        descriptor.clone(),
        fallback_key,
        descriptor,
    );
    assert_eq!(
        second,
        PipelineResolution::Ready(compiler.result),
        "the completed specialization replaces the fallback"
    );
}

/// A frame that discovers more cold permutations than the in-flight bound
/// queues the excess instead of growing the worker channel without limit.
#[test]
fn async_pipeline_cache_bounds_the_in_flight_backlog() {
    let options = RequestAdapterOptions::default();
    let adapter = NullInstance::new()
        .request_adapter(&options)
        .expect("the null backend always has an adapter");
    let (device, _queue) = NullInstance::new()
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null device always comes up");
    let module = device.create_shader_module(&ShaderModuleDescriptor::wgsl(
        "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }",
    ));
    let mut descriptor = RenderPipelineDescriptor {
        label: Some("async-backpressure-test".into()),
        layout: None,
        vertex: VertexState {
            module,
            entry_point: "vs".into(),
            buffers: vec![],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: None,
    };
    let layout = prism_render_driver::VertexBufferLayout::packed(
        prism_render_driver::VertexStepMode::Vertex,
        0,
        &[prism_render_driver::VertexFormat::Float32x3],
    );
    let layout = core::slice::from_ref(&layout);
    descriptor.vertex.buffers = layout.to_vec();
    let compiler = GateCompiler::new(RenderPipelineId::from_parts(303, 1));
    let mut cache = PipelineCache::with_compiler(compiler.clone());

    let mut keys = Vec::new();
    for bit in 0..5 {
        let variant = MeshPipelineKey {
            execution_family: ExecutionFamily::FusedBuiltin,
            response_module_id: ModuleId(1),
            permutation_index: dense(u64::from(bit as u32) + 1),
            fallback_permutation: Some(dense(63)),
            color_format: Some(TextureFormat::Rgba8Unorm),
            depth_format: None,
            sample_count: 1,
            topology: PrimitiveTopology::TriangleList,
        };
        let fallback = full_lobe_fallback(variant).expect("fallback variant");
        let key = PipelineKey::from_descriptor(
            variant.execution_family,
            variant.response_module_id,
            variant.permutation_index,
            &descriptor,
        );
        let fallback_key = PipelineKey::from_descriptor(
            fallback.execution_family,
            fallback.response_module_id,
            fallback.permutation_index,
            &descriptor,
        );
        assert!(matches!(
            cache.resolve_or_request(
                &device,
                key.clone(),
                descriptor.clone(),
                fallback_key,
                descriptor.clone(),
            ),
            PipelineResolution::Fallback(_)
        ));
        keys.push(key);
    }
    assert_eq!(cache.pending_len(), 5);
    assert_eq!(
        cache.waiting_len(),
        1,
        "five cold jobs with a four-job in-flight bound leave one queued"
    );

    compiler.release();
    for _ in 0..20_000 {
        cache.pump(&device);
        if cache.pending_len() == 0 {
            break;
        }
        std::thread::yield_now();
    }
    assert_eq!(cache.pending_len(), 0);
    assert_eq!(cache.waiting_len(), 0);
    assert_eq!(compiler.calls(), 5);
}

/// Archetype/CI warmup queues hot keys through the same worker path and leaves
/// them ready for the first real lookup.
#[test]
fn pipeline_prewarm_queues_hot_variants() {
    let options = RequestAdapterOptions::default();
    let adapter = NullInstance::new()
        .request_adapter(&options)
        .expect("the null backend always has an adapter");
    let (device, _queue) = NullInstance::new()
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null device always comes up");
    let module = device.create_shader_module(&ShaderModuleDescriptor::wgsl(
        "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }",
    ));
    let mut descriptor = RenderPipelineDescriptor {
        label: Some("prewarm-test".into()),
        layout: None,
        vertex: VertexState {
            module,
            entry_point: "vs".into(),
            buffers: vec![],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: None,
    };
    let layout = prism_render_driver::VertexBufferLayout::packed(
        prism_render_driver::VertexStepMode::Vertex,
        0,
        &[prism_render_driver::VertexFormat::Float32x3],
    );
    let layout = core::slice::from_ref(&layout);
    descriptor.vertex.buffers = layout.to_vec();
    let compiler = GateCompiler::new(RenderPipelineId::from_parts(404, 1));
    let mut cache = PipelineCache::with_compiler(compiler.clone());

    let mut requests = Vec::new();
    let mut keys = Vec::new();
    for bit in 0..3 {
        let variant = MeshPipelineKey {
            execution_family: ExecutionFamily::FusedBuiltin,
            response_module_id: ModuleId(1),
            permutation_index: dense(u64::from(bit as u32) + 1),
            fallback_permutation: Some(dense(63)),
            color_format: Some(TextureFormat::Rgba8Unorm),
            depth_format: None,
            sample_count: 1,
            topology: PrimitiveTopology::TriangleList,
        };
        let fallback = full_lobe_fallback(variant).expect("fallback variant");
        let key = PipelineKey::from_descriptor(
            variant.execution_family,
            variant.response_module_id,
            variant.permutation_index,
            &descriptor,
        );
        keys.push(key.clone());
        requests.push((
            key,
            descriptor.clone(),
            PipelineKey::from_descriptor(
                fallback.execution_family,
                fallback.response_module_id,
                fallback.permutation_index,
                &descriptor,
            ),
            descriptor.clone(),
        ));
    }
    cache.prewarm(&device, requests);
    assert_eq!(cache.pending_len(), 3);
    assert_eq!(cache.waiting_len(), 0);

    compiler.release();
    for _ in 0..20_000 {
        cache.pump(&device);
        if cache.pending_len() == 0 {
            break;
        }
        std::thread::yield_now();
    }
    assert_eq!(cache.pending_len(), 0);
    assert_eq!(compiler.calls(), 3);
    for key in keys {
        assert_eq!(
            cache.resolve_or_request(
                &device,
                key.clone(),
                descriptor.clone(),
                key,
                descriptor.clone(),
            ),
            PipelineResolution::Ready(compiler.result)
        );
    }
}

/// The mesh specializer wires the same fallback/switch behavior around the
/// provider's descriptors.
#[test]
fn mesh_specialization_uses_the_full_lobe_fallback_while_compiling() {
    let options = RequestAdapterOptions::default();
    let adapter = NullInstance::new()
        .request_adapter(&options)
        .expect("the null backend always has an adapter");
    let (device, _queue) = NullInstance::new()
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null device always comes up");
    let module = device.create_shader_module(&ShaderModuleDescriptor::wgsl(
        "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }",
    ));
    let descriptor = RenderPipelineDescriptor {
        label: Some("mesh-async-test".into()),
        layout: None,
        vertex: VertexState {
            module,
            entry_point: "vs".into(),
            buffers: vec![],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: None,
    };
    let layout = prism_render_driver::VertexBufferLayout::packed(
        prism_render_driver::VertexStepMode::Vertex,
        0,
        &[prism_render_driver::VertexFormat::Float32x3],
    );
    let layout = core::slice::from_ref(&layout);

    let variant = MeshPipelineKey {
        execution_family: ExecutionFamily::FusedBuiltin,
        response_module_id: ModuleId(1),
        permutation_index: dense(1),
        fallback_permutation: Some(dense(63)),
        color_format: Some(TextureFormat::Rgba8Unorm),
        depth_format: None,
        sample_count: 1,
        topology: PrimitiveTopology::TriangleList,
    };
    let compiler = GateCompiler::new(RenderPipelineId::from_parts(202, 1));
    let mut specializer = MeshPipelineSpecializer::with_compiler(compiler.clone());

    let first = specializer.specialize_with(&device, variant, layout, |_| descriptor.clone());
    assert!(matches!(first, PipelineResolution::Fallback(_)));
    assert_eq!(specializer.pending_pipeline_count(), 1);
    compiler.release();
    for _ in 0..10_000 {
        specializer.pump(&device);
        if specializer.pending_pipeline_count() == 0 {
            break;
        }
        std::thread::yield_now();
    }
    assert_eq!(specializer.pending_pipeline_count(), 0);
    let second = specializer.specialize_with(&device, variant, layout, |_| descriptor.clone());
    assert_eq!(second, PipelineResolution::Ready(compiler.result));
}

/// The real backend proof for P1-d: a cold specialization draws with the
/// full-lobe fallback while the worker compiles the exact PSO on a live device.
#[test]
fn wgpu_pipeline_worker_compiles_a_cold_specialization_in_the_background() {
    let extent = Extent3d::new_2d(4, 4);
    if RenderRuntime::<prism_render_driver_wgpu::WgpuInstance>::wgpu(extent).is_err() {
        assert!(
            !require_gpu(),
            "PRISM_REQUIRE_GPU is set but no adapter was found"
        );
        return;
    }
    let runtime = RenderRuntime::<prism_render_driver_wgpu::WgpuInstance>::wgpu(extent)
        .expect("the probe above succeeded");
    let device = runtime.device();
    let module = device.create_shader_module(&ShaderModuleDescriptor::wgsl(
        "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }\n\
         @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }",
    ));
    let descriptor = RenderPipelineDescriptor {
        label: Some("wgpu-async-pipeline-test".into()),
        layout: None,
        vertex: VertexState {
            module,
            entry_point: "vs".into(),
            buffers: vec![],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            module,
            entry_point: "fs".into(),
            targets: vec![Some(ColorTargetState::opaque(TextureFormat::Rgba8Unorm))],
        }),
    };
    let layout = prism_render_driver::VertexBufferLayout::packed(
        prism_render_driver::VertexStepMode::Vertex,
        0,
        &[prism_render_driver::VertexFormat::Float32x3],
    );
    let layout = core::slice::from_ref(&layout);

    let variant = MeshPipelineKey {
        execution_family: ExecutionFamily::FusedBuiltin,
        response_module_id: ModuleId(1),
        permutation_index: dense(1),
        fallback_permutation: Some(dense(63)),
        color_format: Some(TextureFormat::Rgba8Unorm),
        depth_format: None,
        sample_count: 1,
        topology: PrimitiveTopology::TriangleList,
    };
    let mut specializer = MeshPipelineSpecializer::with_worker(runtime.device_handle());
    let first = specializer.specialize_with(device, variant, layout, |_| descriptor.clone());
    assert!(
        matches!(first, PipelineResolution::Fallback(_)),
        "the first cold request must bind the fallback"
    );
    for _ in 0..100_000 {
        specializer.pump(device);
        if specializer.pending_pipeline_count() == 0 {
            break;
        }
        std::thread::yield_now();
    }
    assert_eq!(
        specializer.pending_pipeline_count(),
        0,
        "the background worker must finish the exact PSO"
    );
    let second = specializer.specialize_with(device, variant, layout, |_| descriptor.clone());
    assert!(
        matches!(second, PipelineResolution::Ready(_)),
        "the exact PSO replaces the fallback once compiled"
    );
    specializer.clear(device);
}

/// The render world materializes the frame's view family from the extracted
/// viewport, and a minimized frame disables (rather than removes) the view.
#[test]
fn render_view_family_tracks_the_frame() {
    let mut app = app_on_null_backend(RenderAppConfig::new(EXTENT));
    app.update();

    let render = app
        .get_sub_app(RenderApp)
        .expect("the plugin registers the render sub-app");
    let views = render
        .world
        .get_resource::<RenderViews>()
        .expect("the plugin installs the view family");
    assert_eq!(views.registry().len(), 1, "one primary view");
    let handle = views.primary_view().expect("the primary view was built");
    let entry = views
        .registry()
        .get(handle)
        .expect("the view is registered");
    assert_eq!(
        entry.kind,
        ViewKind::Main,
        "the primary view is the Main kind"
    );
    assert!(entry.enabled, "a non-empty frame enables the view");

    // A zero-sized frame disables the view but keeps its handle.
    app.world_mut().resource_mut::<RenderIntent>().viewport = RenderViewport::new(0, 0);
    app.update();

    let views = app
        .get_sub_app(RenderApp)
        .expect("the render sub-app")
        .world
        .get_resource::<RenderViews>()
        .expect("the view family");
    assert_eq!(views.registry().len(), 1, "the view is kept, not removed");
    let entry = views
        .registry()
        .get(views.primary_view().expect("the handle survives"))
        .expect("the view is still registered");
    assert!(!entry.enabled, "a minimized frame disables the view");
}

/// Records the order render systems ran in, for the set-ordering test.
#[derive(Resource, Default)]
struct OrderLog(Vec<&'static str>);

fn mark_prepare(mut log: ResMut<OrderLog>) {
    log.0.push("prepare");
}

fn mark_queue(mut log: ResMut<OrderLog>) {
    log.0.push("queue");
}

fn mark_opaque(mut log: ResMut<OrderLog>) {
    log.0.push("opaque");
}

fn mark_queue_meshes(mut log: ResMut<OrderLog>) {
    log.0.push("queue-meshes");
}

fn mark_prepare_bind_groups(mut log: ResMut<OrderLog>) {
    log.0.push("prepare-bind-groups");
}

/// The render stages and phases order systems whether or not the registering
/// plugin knows about the others: queueing precedes GPU-resource preparation
/// (the renderer's real order — preparation walks the queued order), mesh
/// queueing nests inside queueing, and the `Opaque` phase records after both.
#[test]
fn render_sets_order_stages_and_phases() {
    let mut app = app_on_null_backend(RenderAppConfig::new(EXTENT));
    {
        let render = app
            .sub_app_mut(RenderApp)
            .expect("the plugin registers the render sub-app");
        render.world.insert_resource(OrderLog::default());
        let schedule = render
            .world
            .resource_mut::<Schedules>()
            .get_mut(Render)
            .expect("the plugin registers the Render schedule");
        // Registered out of order on purpose: the sets must impose the order.
        schedule.add_systems(mark_queue.in_set(RenderSystems::Queue));
        schedule.add_systems(mark_queue_meshes.in_set(RenderSystems::QueueMeshes));
        schedule.add_systems(mark_opaque.in_set(RenderPhase::Opaque));
        schedule.add_systems(mark_prepare.in_set(RenderSystems::Prepare));
        schedule.add_systems(mark_prepare_bind_groups.in_set(RenderSystems::PrepareBindGroups));
    }

    app.update();

    let log = app
        .get_sub_app(RenderApp)
        .expect("the render sub-app")
        .world
        .get_resource::<OrderLog>()
        .expect("the order log");
    assert_eq!(
        log.0,
        [
            "queue",
            "queue-meshes",
            "prepare",
            "prepare-bind-groups",
            "opaque"
        ],
        "queueing precedes preparation, mesh queueing nests inside queueing, \
         bind groups nest inside preparation, and the Opaque phase records last"
    );
}

/// The frame count the headless tests drive.
const FRAMES: u64 = 5;

fn mark_graph_begin(mut log: ResMut<OrderLog>) {
    log.0.push("graph-begin");
}

fn mark_graph_render(mut log: ResMut<OrderLog>) {
    log.0.push("graph-render");
}

fn mark_graph_submit(mut log: ResMut<OrderLog>) {
    log.0.push("graph-submit");
}

fn mark_graph_finish(mut log: ResMut<OrderLog>) {
    log.0.push("graph-finish");
}

/// The render graph's own sets order node systems: the graph crate carries the
/// policy as data (it cannot name `prism_ecs`'s `SystemSet`), and the renderer
/// makes it schedulable — so a node registers `in_set(RenderGraphSet(..))` and
/// gets the order without wiring edges by hand.
#[test]
fn render_graph_sets_order_node_systems() {
    use crate::sets::{configure_render_graph_sets, RenderGraphSet};
    use prism_ecs::schedule::IntoSystemConfigs;

    let mut world = prism_ecs::world::World::new();
    world.insert_resource(OrderLog::default());
    let mut schedule = prism_ecs::schedule::Schedule::new();
    configure_render_graph_sets(&mut schedule);
    // Registered in reverse, so only the set policy can produce the order.
    schedule.add_systems(mark_graph_finish.in_set(RenderGraphSet(
        prism_render_graph::RenderGraphSystems::Finish,
    )));
    schedule.add_systems(mark_graph_submit.in_set(RenderGraphSet(
        prism_render_graph::RenderGraphSystems::Submit,
    )));
    schedule.add_systems(mark_graph_render.in_set(RenderGraphSet(
        prism_render_graph::RenderGraphSystems::Render,
    )));
    schedule.add_systems(mark_graph_begin.in_set(RenderGraphSet(
        prism_render_graph::RenderGraphSystems::Begin,
    )));

    schedule.run(&mut world);

    let log = world.resource::<OrderLog>();
    assert_eq!(
        log.0,
        [
            "graph-begin",
            "graph-render",
            "graph-submit",
            "graph-finish"
        ],
        "the graph's begin → render → submit → finish order is imposed by the sets"
    );
}

/// The handles of the views a per-view system saw, in the order it saw them.
#[derive(Default, Resource)]
struct ViewLog(Vec<prism_render_architecture::view_family::ViewHandle>);

/// A render system in the ported style: it reads the view it is recording for
/// through the view query instead of looking the view up itself.
fn record_view(mut log: ResMut<ViewLog>, view: Option<crate::ViewQuery<&crate::RenderView>>) {
    let Some(view) = view else {
        return;
    };
    log.0.push(view.into_inner().handle);
}

/// Each live view is backed by one render entity, and the entity outlives a
/// frame that merely stops rendering the view.
#[test]
fn the_view_family_materializes_a_render_entity_per_view() {
    let mut app = app_on_null_backend(RenderAppConfig::new(EXTENT));
    app.update();

    let render = app.get_sub_app(RenderApp).expect("the render sub-app");
    let primary = render
        .world
        .resource::<RenderViews>()
        .primary_view()
        .expect("the frame built a primary view");
    let entity = render
        .world
        .resource::<RenderViews>()
        .entity(primary)
        .expect("the primary view has a render entity");
    let view = render
        .world
        .resource::<RenderViews>()
        .entities()
        .find(|(candidate, _)| *candidate == entity)
        .map(|(_, view)| view)
        .expect("the entity is in the family");
    assert_eq!(view.handle, primary);
    assert_eq!(view.kind, ViewKind::Main);
    assert!(view.enabled, "a non-empty frame enables the view");

    // A zero-sized frame disables the view but keeps its entity.
    app.world_mut().resource_mut::<RenderIntent>().viewport = RenderViewport::new(0, 0);
    app.update();

    let render = app.get_sub_app(RenderApp).expect("the render sub-app");
    let views = render.world.resource::<RenderViews>();
    assert_eq!(
        views.entity(primary),
        Some(entity),
        "the entity survives a skipped frame"
    );
    assert!(
        !views
            .entities()
            .find(|(candidate, _)| *candidate == entity)
            .expect("the entity is still in the family")
            .1
            .enabled,
        "the view's entity records that it is not rendering this frame"
    );
}

/// Per-view work runs once per view with that view as the current one: a system
/// taking the view query sees its own view, and no view is left current after the
/// walk (a later system must not read a stale one).
#[test]
fn per_view_work_sees_its_own_view_and_clears_it_afterwards() {
    use crate::view_driver::for_each_view;
    use crate::view_query::CurrentView;
    use prism_render_architecture::view_family::ViewKind;

    let mut app = app_on_null_backend(RenderAppConfig::new(EXTENT));
    {
        // A second view of another kind, on top of the primary one.
        let render = app
            .sub_app_mut(RenderApp)
            .expect("the plugin registers the render sub-app");
        render
            .world
            .resource_mut::<RenderViews>()
            .registry_mut()
            .register(ViewKind::Shadow);
    }
    app.update();

    let world = &mut app
        .sub_app_mut(RenderApp)
        .expect("the render sub-app")
        .world;
    world.insert_resource(ViewLog::default());
    let mut schedule = prism_ecs::schedule::Schedule::new();
    schedule.add_systems(record_view);

    let seen = std::cell::Cell::new(0_usize);
    for_each_view(world, |world, _entity, _view| {
        schedule.run(world);
        seen.set(seen.get() + 1);
    });

    assert_eq!(seen.get(), 2, "the walk visits every materialized view");
    let log = world.resource::<ViewLog>();
    assert_eq!(log.0.len(), 2, "the system ran once per view");
    log.0.iter().for_each(|handle| {
        assert!(
            world.resource::<RenderViews>().registry().contains(*handle),
            "each recorded handle is a live view"
        );
    });
    assert_ne!(log.0[0], log.0[1], "the two views are distinct");
    assert!(
        world.get_resource::<CurrentView>().is_none(),
        "no view is left current after the walk"
    );
}

/// A small, non-square frame so a swapped axis cannot pass by accident.
const EXTENT: Extent3d = Extent3d::new_2d(64, 48);

/// An app with the plugin built on the headless null backend.
fn app_on_null_backend(config: RenderAppConfig) -> App {
    let mut app = App::new();
    app.add_plugins(PrismRenderPlugin::with_config(NullInstance::new(), config));
    app
}

/// Whether a missing adapter must fail a GPU-gated test instead of skipping it.
pub(crate) fn require_gpu() -> bool {
    std::env::var_os("PRISM_REQUIRE_GPU").is_some()
}

/// The render schedule runs every frame, the extract feeds it, and the runtime
/// submits one frame per `App::update`.
#[test]
fn render_schedule_runs_each_frame_and_submits_a_frame() {
    let mut app = app_on_null_backend(RenderAppConfig::new(EXTENT));

    for _ in 0..FRAMES {
        app.update();
    }

    let render = app
        .get_sub_app(RenderApp)
        .expect("the plugin registers the render sub-app");
    let stats = *render
        .world
        .get_resource::<RenderFrameStats>()
        .expect("the plugin installs the stats resource");
    assert_eq!(
        stats.frames_submitted, FRAMES,
        "one frame is submitted per App::update"
    );
    assert_eq!(stats.frames_failed, 0, "no frame failed to record");
    assert_eq!(
        stats.frames_skipped, 0,
        "every frame had a non-empty viewport"
    );
    assert_eq!(
        stats.passes_declared,
        FRAMES * 5,
        "each recorded frame declares three lighting passes, one clear, and one OCIO pass"
    );
    assert_eq!(stats.last_viewport, RenderViewport::from_extent(EXTENT));

    let record = *render
        .world
        .get_resource::<RenderRecord>()
        .expect("the plugin installs the render record");
    assert_eq!(
        record.frame_index, FRAMES,
        "the extract ran once per frame and advanced the record"
    );
    assert_eq!(record.viewport, RenderViewport::from_extent(EXTENT));
    let plan = render
        .world
        .get_resource::<LightingPlanState>()
        .expect("the render world owns the unified lighting plan");
    assert_eq!(plan.compile_count, FRAMES);
    assert_eq!(stats.lighting_plan_compiles, FRAMES);
    assert_eq!(
        stats.lighting_recipes_selected,
        plan.plan.recipes.len() as u64
    );
    assert_eq!(
        stats.lighting_recipes_rejected,
        plan.plan.rejected.len() as u64
    );
}

#[test]
fn lighting_plan_uses_host_caps_scene_stats_and_budget() {
    let config = RenderAppConfig::new(EXTENT)
        .with_lighting_scene_stats(SceneStats {
            lights: 256,
            punctual_max: 8,
            npr_materials: 3,
            proxy_page_ratio: 0.5,
            triangle_density: 4.0,
        })
        .with_lighting_budget(FrameResourceBudget {
            gpu_ms: 1.0,
            bandwidth: 8 * 1024 * 1024,
            resident_bytes: 16 * 1024 * 1024,
            io_ops: 16,
            shading_rate_budget: 0.5,
        })
        .with_lighting_caps(PlanCaps {
            ray_tracing: false,
            virtual_shadow: false,
            offline: false,
            deterministic_npr: false,
        })
        .with_lighting_tier(Tier::Preview);
    let mut app = app_on_null_backend(config);
    app.update();
    let render = app.get_sub_app(RenderApp).expect("render sub-app");
    let state = render
        .world
        .get_resource::<LightingPlanState>()
        .expect("lighting plan state");
    assert_eq!(state.scene_stats.lights, 256);
    assert_eq!(state.plan.tier, Tier::Preview);
    assert!(state
        .plan
        .rejected
        .iter()
        .any(|rejection| matches!(rejection, Rejection::Budget(_))));
    assert!(state
        .plan
        .rejected
        .iter()
        .any(|rejection| matches!(rejection, Rejection::Unsupported(_))));
    assert!(state
        .plan
        .rejected
        .iter()
        .any(|rejection| matches!(rejection, Rejection::Unavailable(_))));
    assert!(state.plan.explain().contains("tier=Preview"));
}

#[test]
fn deterministic_gi_oracle_selects_valid_reservoir_sample() {
    let surface = LightingSurfaceInput {
        position: [0.0, 0.0, 0.0],
        normal: [0.0, 1.0, 0.0],
        albedo: [0.5, 0.25, 1.0],
        ..LightingSurfaceInput::default()
    };
    let candidate = LightingGiCandidateInput {
        position: [0.0, 2.0, 0.0],
        source_pdf: 1.0,
        normal: [0.0, -1.0, 0.0],
        target_pdf: 1.0,
        radiance: [2.0, 4.0, 8.0],
        selection_u01: 0.25,
        visibility: 0.75,
        ..LightingGiCandidateInput::default()
    };
    assert_eq!(
        evaluate_deterministic_gi(surface, &[candidate]),
        [0.1875, 0.1875, 1.5]
    );
}

#[test]
fn deterministic_gi_oracle_ignores_invalid_and_backfacing_candidates() {
    let surface = LightingSurfaceInput::default();
    let invalid = LightingGiCandidateInput {
        source_pdf: 0.0,
        target_pdf: 1.0,
        ..LightingGiCandidateInput::default()
    };
    let backfacing = LightingGiCandidateInput {
        position: [0.0, 2.0, 0.0],
        normal: [0.0, 1.0, 0.0],
        source_pdf: 1.0,
        target_pdf: 1.0,
        radiance: [10.0; 3],
        ..LightingGiCandidateInput::default()
    };
    assert_eq!(
        evaluate_deterministic_gi(surface, &[invalid, backfacing]),
        [0.0; 3]
    );
}

#[test]
fn lighting_inputs_extract_stats_and_field_epochs_into_the_render_world() {
    let mut app = app_on_null_backend(RenderAppConfig::new(EXTENT));
    let mut inputs = LightingSceneInputs::with_stats(SceneStats {
        lights: 12,
        punctual_max: 4,
        ..SceneStats::default()
    });
    inputs.directionals = vec![LightingDirectionalInput::default(); 2];
    inputs.punctuals = vec![LightingPunctualInput::default(); 10];
    app.world_mut().insert_resource(inputs);
    app.world_mut()
        .resource_mut::<LightingSceneInputs>()
        .mark_changed(1, ChangeFields::LIGHTING);

    app.update();

    let render = app.get_sub_app(RenderApp).expect("render sub-app");
    let extracted = render.world.resource::<ExtractedLightingInputs>();
    assert_eq!(extracted.stats.lights, 12);
    assert_eq!(extracted.directionals.len(), 2);
    assert_eq!(extracted.punctuals.len(), 10);
    assert_eq!(extracted.journal_snapshot.sequence, 1);
    assert_eq!(extracted.journal_snapshot.epoch.lighting, 1);

    let state = render.world.resource::<LightingPlanState>();
    assert_eq!(state.scene_stats.lights, 12);
    assert_eq!(state.journal_snapshot, extracted.journal_snapshot);
    assert_eq!(state.executions.len(), state.plan.recipes.len());
    assert_eq!(state.explain, state.plan.explain());
    let stats = render.world.resource::<RenderFrameStats>();
    assert_eq!(stats.lighting_cache_invalidations, 1);
    assert_eq!(
        stats.lighting_executions_planned,
        state.executions.len() as u64
    );
}

#[test]
fn lighting_debug_view_snapshot_consumes_selected_metadata_and_rejections() {
    let mut app = app_on_null_backend(RenderAppConfig::new(EXTENT));
    app.update();

    let render = app.get_sub_app(RenderApp).expect("render sub-app");
    let plan = render.world.resource::<LightingPlanState>();
    let debug = render.world.resource::<LightingDebugViewState>();
    assert_eq!(debug.compiled_frame, plan.compiled_frame);
    assert_eq!(debug.plan_compile_count, plan.compile_count);
    assert_eq!(debug.explain, plan.explain);
    assert_eq!(
        debug.rejection_explanations,
        plan.plan.rejection_explanations()
    );
    assert_eq!(debug.selected_recipes.len(), plan.plan.recipes.len());
    let direct = debug
        .selected_recipes
        .iter()
        .find(|entry| entry.recipe == RecipeId::SpecializedDeterministic)
        .expect("the default plan selects the deterministic direct recipe");
    assert_eq!(direct.meta, direct.recipe.debug_meta());
    assert!(direct.views.contains(&DebugView::Visibility));
    assert!(direct.views.contains(&DebugView::BudgetUsage));

    let stats = render.world.resource::<RenderFrameStats>();
    assert_eq!(stats.lighting_debug_updates, 1);
    assert_eq!(
        stats.lighting_debug_selected_recipes,
        debug.selected_recipes.len() as u64
    );
    assert_eq!(
        stats.lighting_debug_rejections,
        debug.rejection_explanations.len() as u64
    );
}

#[test]
fn lighting_debug_view_resynchronizes_as_the_plan_changes() {
    let config = RenderAppConfig::new(EXTENT).with_lighting_caps(PlanCaps {
        ray_tracing: true,
        virtual_shadow: true,
        offline: true,
        deterministic_npr: false,
    });
    let mut app = app_on_null_backend(config);
    app.update();

    app.world_mut()
        .insert_resource(LightingSceneInputs::with_stats(SceneStats {
            lights: 4096,
            punctual_max: 1,
            ..SceneStats::default()
        }));
    app.world_mut()
        .resource_mut::<LightingSceneInputs>()
        .mark_changed(2, ChangeFields::LIGHTING);
    app.update();

    let render = app.get_sub_app(RenderApp).expect("render sub-app");
    let plan = render.world.resource::<LightingPlanState>();
    let debug = render.world.resource::<LightingDebugViewState>();
    assert_eq!(debug.compiled_frame, plan.compiled_frame);
    assert_eq!(debug.plan_compile_count, plan.compile_count);
    assert_eq!(debug.selected_recipes.len(), plan.plan.recipes.len());
    assert_eq!(
        debug.rejection_explanations,
        plan.plan.rejection_explanations()
    );
    assert!(debug
        .rejection_explanations
        .iter()
        .any(|rejection| rejection.recipe == RecipeId::DirectDeterministicCsm));
    let stats = render.world.resource::<RenderFrameStats>();
    assert_eq!(stats.lighting_debug_updates, 2);
}

#[test]
fn deterministic_direct_cpu_oracle_accumulates_all_light_terms() {
    let surface = LightingSurfaceInput {
        position: [0.0, 0.0, 0.0],
        normal: [0.0, 1.0, 0.0],
        albedo: [0.5, 0.25, 1.0],
        ambient: [1.0, 2.0, 3.0],
    };
    let directionals = [LightingDirectionalInput {
        direction_to_light: [0.0, 1.0, 0.0],
        visibility: 0.5,
        illuminance: [2.0, 4.0, 8.0],
        _padding: 0.0,
    }];
    let punctuals = [LightingPunctualInput {
        position: [0.0, 2.0, 0.0],
        intensity: [4.0, 8.0, 16.0],
        ..LightingPunctualInput::default()
    }];
    assert_eq!(
        evaluate_deterministic_direct(surface, &directionals, &punctuals),
        [1.5, 1.5, 11.0]
    );
}

/// The render sub-app owns the GPU state; the main world never sees it.
#[test]
fn gpu_state_lives_only_in_the_render_world() {
    let app = app_on_null_backend(RenderAppConfig::new(EXTENT));

    assert!(
        app.world()
            .get_resource::<RenderRuntimeResource<NullInstance>>()
            .is_none(),
        "the main world must never hold the render runtime"
    );
    let render = app
        .get_sub_app(RenderApp)
        .expect("render sub-app registered");
    assert!(
        render
            .world
            .get_resource::<RenderRuntimeResource<NullInstance>>()
            .is_some(),
        "the render world owns the render runtime"
    );
    assert!(
        render.world.get_resource::<RenderTarget>().is_some(),
        "the render world owns the frame target"
    );
    assert!(
        app.world().get_resource::<RenderTarget>().is_none(),
        "the frame target must not leak into the main world"
    );
    // The render schedule really is a schedule, not a bare core phase.
    let schedules = render
        .world
        .get_resource::<Schedules>()
        .expect("the render world owns its schedules");
    assert!(
        schedules.contains(Render),
        "the Render schedule is registered"
    );
}

/// Changing the main world's camera resizes the render world's frame target:
/// the extract genuinely drives the renderer rather than merely running.
#[test]
fn extract_resizes_the_frame_target_from_the_main_world() {
    let mut app = app_on_null_backend(RenderAppConfig::new(EXTENT));
    app.update();
    {
        let render = app.get_sub_app(RenderApp).expect("render sub-app");
        let target = render
            .world
            .get_resource::<RenderTarget>()
            .expect("frame target");
        assert_eq!(target.viewport(), RenderViewport::from_extent(EXTENT));
    }

    let resized = RenderViewport::new(32, 96);
    *app.world_mut().resource_mut::<RenderIntent>() = RenderIntent::new(resized);
    app.update();

    let render = app.get_sub_app(RenderApp).expect("render sub-app");
    let target = render
        .world
        .get_resource::<RenderTarget>()
        .expect("frame target");
    assert_eq!(
        target.viewport(),
        resized,
        "the extracted camera resized the render target"
    );
    let stats = render
        .world
        .get_resource::<RenderFrameStats>()
        .expect("stats");
    assert_eq!(stats.frames_submitted, 2, "both frames were submitted");
}

/// A zero-sized frame is a successful no-op, not a failed or submitted frame.
#[test]
fn empty_viewport_skips_the_frame() {
    let mut app = app_on_null_backend(RenderAppConfig::new(EXTENT));
    *app.world_mut().resource_mut::<RenderIntent>() = RenderIntent::new(RenderViewport::new(0, 0));
    app.update();

    let render = app.get_sub_app(RenderApp).expect("render sub-app");
    let stats = render
        .world
        .get_resource::<RenderFrameStats>()
        .expect("stats");
    assert_eq!(stats.frames_skipped, 1);
    assert_eq!(stats.frames_submitted, 0);
    assert_eq!(stats.frames_recorded(), 1);
}

/// The seam verified on hardware: the extracted viewport sizes a real GPU
/// frame, and reading it back shows the frame's pass actually wrote the target.
///
/// The target is pre-filled opaque white first (and the pre-fill is itself read
/// back to prove it landed), so a black result can only come from the frame's
/// clear pass executing — a freshly created wgpu texture is already zeroed, so
/// an unfilled target would pass vacuously.
#[test]
fn wgpu_backend_records_and_clears_a_real_frame() {
    const SIZE: u32 = 32;
    let extent = Extent3d::new_2d(SIZE, SIZE);

    // The plugin panics when no adapter exists, so probe first and skip cleanly.
    if RenderRuntime::<prism_render_driver_wgpu::WgpuInstance>::wgpu(extent).is_err() {
        assert!(
            !require_gpu(),
            "PRISM_REQUIRE_GPU is set but no adapter was found"
        );
        return;
    }

    let mut app = App::new();
    app.add_plugins(PrismRenderPlugin::with_config(
        prism_render_driver_wgpu::WgpuInstance::new(),
        RenderAppConfig::new(extent),
    ));
    let surface = LightingSurfaceInput {
        position: [0.0, 0.0, 0.0],
        normal: [0.0, 1.0, 0.0],
        albedo: [0.5, 0.25, 1.0],
        ambient: [1.0, 2.0, 3.0],
    };
    let directionals = vec![LightingDirectionalInput {
        direction_to_light: [0.0, 1.0, 0.0],
        visibility: 0.5,
        illuminance: [2.0, 4.0, 8.0],
        _padding: 0.0,
    }];
    let punctuals = vec![LightingPunctualInput {
        position: [0.0, 2.0, 0.0],
        intensity: [4.0, 8.0, 16.0],
        ..LightingPunctualInput::default()
    }];
    let gi_candidates = vec![LightingGiCandidateInput {
        position: [0.0, 2.0, 0.0],
        normal: [0.0, -1.0, 0.0],
        source_pdf: 1.0,
        target_pdf: 1.0,
        radiance: [2.0, 4.0, 8.0],
        selection_u01: 0.25,
        visibility: 0.75,
        ..LightingGiCandidateInput::default()
    }];
    let surface2 = LightingSurfaceInput {
        position: [1.0, 0.0, 0.0],
        normal: [0.0, 1.0, 0.0],
        albedo: [0.25, 0.5, 0.75],
        ambient: [2.0, 1.0, 0.5],
    };
    let mut surfaces = vec![surface, surface2];
    surfaces.extend((0..63).map(|index| LightingSurfaceInput {
        position: [index as f32 * 0.02, 0.0, 0.0],
        normal: [0.0, 1.0, 0.0],
        albedo: [0.1 + (index % 9) as f32 * 0.1, 0.6, 0.3],
        ambient: [0.0; 3],
    }));
    let gi_candidates_per_surface = surfaces
        .iter()
        .enumerate()
        .map(|(index, _)| {
            let mut candidate = gi_candidates[0];
            let scale = 1.0 + index as f32 * 0.01;
            candidate.radiance = [2.0 * scale, 4.0 * scale, 8.0 * scale];
            vec![candidate]
        })
        .collect::<Vec<_>>();
    let expected_direct = surfaces
        .iter()
        .copied()
        .map(|sample| evaluate_deterministic_direct(sample, &directionals, &punctuals))
        .collect::<Vec<_>>();
    let expected_gi = surfaces
        .iter()
        .copied()
        .zip(&gi_candidates_per_surface)
        .map(|(sample, candidates)| evaluate_deterministic_gi(sample, candidates))
        .collect::<Vec<_>>();
    let expected_composite = expected_direct
        .iter()
        .zip(&expected_gi)
        .map(|(direct, gi)| [direct[0] + gi[0], direct[1] + gi[1], direct[2] + gi[2]])
        .collect::<Vec<_>>();
    {
        let inputs = app.world_mut().resource_mut::<LightingSceneInputs>();
        inputs.surfaces = surfaces.clone();
        inputs.directionals = directionals;
        inputs.punctuals = punctuals;
        inputs.gi_candidates = gi_candidates_per_surface;
    }

    // Pre-fill the offscreen target with opaque white, then prove the upload
    // landed before any frame is rendered.
    let white = vec![0xFF_u8; (SIZE * SIZE * 4) as usize];
    {
        let render = app.get_sub_app(RenderApp).expect("render sub-app");
        let runtime = render
            .world
            .get_resource::<RenderRuntimeResource<prism_render_driver_wgpu::WgpuInstance>>()
            .expect("render runtime");
        let target = *render
            .world
            .get_resource::<RenderTarget>()
            .expect("frame target");
        let queue = runtime.runtime().queue();
        queue.write_texture(
            TextureWrite {
                texture: target.texture(),
                mip_level: 0,
                origin: [0, 0, 0],
                size: extent,
            },
            &white,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(SIZE * 4),
                rows_per_image: Some(SIZE),
            },
        );
        assert!(runtime.runtime().wait_idle());
        let prefill = queue.read_texture_rgba8(target.texture(), SIZE, SIZE);
        assert!(
            prefill
                .chunks_exact(4)
                .all(|pixel| pixel == [0xFF, 0xFF, 0xFF, 0xFF]),
            "the white pre-fill landed, so the black assertion below is meaningful"
        );
    }

    app.update();

    let render = app.get_sub_app(RenderApp).expect("render sub-app");
    let runtime = render
        .world
        .get_resource::<RenderRuntimeResource<prism_render_driver_wgpu::WgpuInstance>>()
        .expect("render runtime");
    assert!(
        runtime.runtime().wait_idle(),
        "the submitted frame completes on the GPU"
    );

    let target = *render
        .world
        .get_resource::<RenderTarget>()
        .expect("frame target");
    assert_eq!(target.viewport(), RenderViewport::from_extent(extent));
    let pixels = runtime
        .runtime()
        .queue()
        .read_texture_rgba8(target.texture(), SIZE, SIZE);
    assert_eq!(pixels.len(), (SIZE * SIZE * 4) as usize);
    assert!(
        pixels.chunks_exact(4).all(|pixel| pixel == [0, 0, 0, 255]),
        "the frame's clear pass overwrote the white target with opaque black"
    );

    let lighting = render
        .world
        .resource::<LightingGpuState<prism_render_driver_wgpu::WgpuInstance>>();
    let read_output = |buffer| {
        runtime
            .runtime()
            .queue()
            .read_buffer(buffer, crate::lighting_gpu::OUTPUT_BYTES)
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("f32-sized output")))
            .collect::<Vec<_>>()
    };
    let direct_values = read_output(lighting.direct_output_buffer());
    let gi_values = read_output(lighting.gi_output_buffer());
    let composite_values = read_output(lighting.output_buffer());
    assert_eq!(
        direct_values.len(),
        crate::lighting_gpu::MAX_LIGHTING_SURFACES * 4
    );
    assert_eq!(gi_values.len(), direct_values.len());
    assert_eq!(composite_values.len(), direct_values.len());
    for surface_index in 0..surfaces.len() {
        let base = surface_index * 4;
        for channel in 0..3 {
            assert!(
                (direct_values[base + channel] - expected_direct[surface_index][channel]).abs()
                    < 1.0e-4,
                "GPU direct surface {surface_index} channel {channel}: {} vs {}",
                direct_values[base + channel],
                expected_direct[surface_index][channel]
            );
            assert!(
                (composite_values[base + channel] - expected_composite[surface_index][channel])
                    .abs()
                    < 1.0e-4,
                "GPU composite surface {surface_index} channel {channel}: {} vs {}",
                composite_values[base + channel],
                expected_composite[surface_index][channel]
            );
        }
        assert_eq!(direct_values[base + 3], 1.0);
        assert_eq!(gi_values[base + 3], 1.0);
        assert_eq!(composite_values[base + 3], 1.0);
        for channel in 0..3 {
            assert!(
                (gi_values[base + channel] - expected_gi[surface_index][channel]).abs() < 1.0e-4,
                "GPU GI surface {surface_index} channel {channel}: {} vs {}",
                gi_values[base + channel],
                expected_gi[surface_index][channel]
            );
        }
    }
    for tail in (surfaces.len() * 4)..direct_values.len() {
        assert_eq!(
            direct_values[tail], 0.0,
            "direct tail is deterministically clear"
        );
        assert_eq!(gi_values[tail], 0.0, "GI tail is deterministically clear");
        assert_eq!(
            composite_values[tail], 0.0,
            "composite tail is deterministically clear"
        );
    }
    assert_eq!(
        render
            .world
            .resource::<RenderFrameStats>()
            .lighting_gpu_passes_recorded,
        3
    );
}

/// A `RenderQueue` that captures submitted command buffers instead of issuing
/// them, so a graph recording can be asserted without a GPU.
#[derive(Default)]
struct CapturingQueue {
    buffers: std::cell::RefCell<Vec<CommandBuffer>>,
}

impl CapturingQueue {
    fn take(&self) -> Vec<CommandBuffer> {
        core::mem::take(&mut *self.buffers.borrow_mut())
    }
}

impl RenderQueue for CapturingQueue {
    fn write_buffer(&self, _: BufferId, _: u64, _: &[u8]) {}

    fn write_texture(&self, _: TextureWrite, _: &[u8], _: ImageDataLayout) {}

    fn submit(&self, command_buffers: &[CommandBuffer]) {
        self.buffers.borrow_mut().extend_from_slice(command_buffers);
    }
}

/// `mesh_draw` records a full draw into a graph pass: bind the pipeline, bind the
/// vertex and index buffers, then one indexed draw.
#[test]
fn mesh_draw_records_into_a_graph_pass() {
    use prism_render_driver::{CommandBuffer, IndexFormat, Pass, RenderCommand};
    use prism_render_graph::{ImportedTexture, PassFlags, RenderGraph, TextureDesc};

    let options = RequestAdapterOptions::default();
    let adapter = NullInstance::new()
        .request_adapter(&options)
        .expect("the null backend always has an adapter");
    let (device, _queue) = NullInstance::new()
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null device always comes up");

    let vertices = null_vertex_buffer(&device);
    let indices = device.create_buffer(&BufferDescriptor {
        label: Some("test-indices".into()),
        size: 64,
        usage: BufferUsages::INDEX,
        mapped_at_creation: false,
    });
    let target = device.create_texture(&TextureDescriptor::new_2d(
        Extent3d::new_2d(4, 4),
        TextureFormat::Rgba8Unorm,
        TextureUsages::RENDER_ATTACHMENT,
    ));
    let view = device.create_texture_view(target, &Default::default());
    let module = device.create_shader_module(&ShaderModuleDescriptor::wgsl(
        "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }",
    ));
    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("record-mesh".into()),
        layout: None,
        vertex: VertexState {
            module,
            entry_point: "vs".into(),
            buffers: vec![],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: None,
    });

    let mut functions = DrawFunctions::new();
    let draw = functions.register(crate::mesh_draw);
    assert_eq!(draw, DrawFunctionId(0), "the first registration is id 0");

    let mut queues = RenderQueues::new();
    queues.push(
        RenderPhase::Opaque,
        PhaseItem {
            pipeline,
            draw,
            geometry: GenerationalHandle::new(1, 1),
            vertices: vec![vertices],
            indices: Some((indices, IndexFormat::Uint16)),
            first: 0,
            count: 3,
            base_vertex: 0,
            indirect: None,
            lod_level: 0,
            material: GenerationalHandle::new(2, 1),
            scene: GenerationalHandle::new(3, 1),
            order: 0,
        },
    );

    let mut graph = RenderGraph::new(Extent3d::new_2d(4, 4));
    let handle = graph.import_texture(
        "target",
        TextureDesc::color(TextureFormat::Rgba8Unorm),
        ImportedTexture {
            texture: target,
            default_view: view,
            entry_state: prism_render_driver::TextureState::initial(),
            exit_state: prism_render_driver::TextureState::copy_src(),
            mip_levels: 1,
            array_layers: 1,
        },
    );
    graph.add_raster_pass("queue-draw", PassFlags::SIDE_EFFECT, move |builder| {
        let _ = builder.color_attachment(handle);
        move |ctx| queues.record(RenderPhase::Opaque, ctx, &functions)
    });

    let capture = CapturingQueue::default();
    graph
        .execute(&device, &capture)
        .expect("the graph executes")
        // CapturingQueue only clones commands; it never starts GPU work.
        .release(&device);

    let buffers: Vec<CommandBuffer> = capture.take();
    let commands = buffers
        .iter()
        .find_map(|buffer| match buffer.passes.first() {
            Some(Pass::Render { commands, .. }) => Some(commands),
            _ => None,
        })
        .expect("the raster pass was recorded");
    assert!(
        matches!(commands.first(), Some(RenderCommand::SetPipeline(_))),
        "the item's pipeline is bound first"
    );
    assert!(
        matches!(
            commands.get(1),
            Some(RenderCommand::SetVertexBuffer { slot: 0, .. })
        ),
        "the vertex buffer is bound at slot 0"
    );
    assert!(
        matches!(commands.get(2), Some(RenderCommand::SetIndexBuffer(_))),
        "the index buffer is bound"
    );
    assert!(
        matches!(commands.get(3), Some(RenderCommand::DrawIndexed { .. })),
        "one indexed draw is issued"
    );
}

/// The lighting plan is recorded as a real compute pass rather than only
/// incrementing a frame statistic: the pass binds its pipeline and resources
/// before dispatching one workgroup.
#[test]
fn lighting_plan_records_compute_pipeline_bind_group_and_dispatch() {
    use prism_render_architecture::lighting::contract::{
        BudgetCompiler, FrameResourceBudget, PlanCaps, Tier,
    };
    use prism_render_driver::{ComputeCommand, Pass};

    let extent = Extent3d::new_2d(4, 4);
    let mut runtime = RenderRuntime::headless(NullInstance::new(), extent)
        .expect("the null runtime always comes up");
    let lighting = LightingGpuState::new(&mut runtime).expect("lighting GPU bundle");
    let plan = BudgetCompiler::compile(
        SceneStats::default(),
        FrameResourceBudget {
            gpu_ms: 16.666,
            bandwidth: 512 * 1024 * 1024,
            resident_bytes: 512 * 1024 * 1024,
            io_ops: 4096,
            shading_rate_budget: 1.0,
        },
        PlanCaps {
            ray_tracing: false,
            virtual_shadow: false,
            offline: false,
            deterministic_npr: false,
        },
        Tier::Balanced,
    );
    let mut extracted = ExtractedLightingInputs::default();
    extracted.surfaces = vec![LightingSurfaceInput::default()];
    lighting.prepare(&plan, &extracted, runtime.queue());

    let target = RenderTarget::new(runtime.device(), extent);
    let queues = RenderQueues::new();
    let functions = DrawFunctions::new();
    let graph = crate::frame::target_graph(&target, &queues, &functions, lighting.pass(&plan));
    let capture = CapturingQueue::default();
    graph
        .execute(runtime.device(), &capture)
        .expect("the lighting graph records on the null backend")
        .release(runtime.device());

    let compute_labels = capture
        .take()
        .into_iter()
        .flat_map(|buffer| buffer.passes)
        .filter_map(|pass| match pass {
            Pass::Compute { label, commands } => Some((label, commands)),
            _ => None,
        })
        .map(|(label, commands)| {
            assert!(matches!(
                commands.first(),
                Some(ComputeCommand::SetPipeline(_))
            ));
            assert!(matches!(
                commands.get(1),
                Some(ComputeCommand::SetBindGroup { index: 0, .. })
            ));
            assert!(matches!(
                commands.get(2),
                Some(ComputeCommand::Dispatch { x: 1, y: 1, z: 1 })
            ));
            label.expect("lighting compute passes are labeled")
        })
        .collect::<Vec<_>>();
    assert_eq!(
        compute_labels,
        [
            "prism-lighting-direct-dispatch",
            "prism-lighting-gi-dispatch",
            "prism-lighting-composite",
        ]
    );
}

#[test]
fn frame_graph_records_optional_trace_between_scene_and_gi() {
    use crate::lighting_trace_gpu::{LightingTraceGpu, PackedTraceScene, TraceRay};
    use prism_render_architecture::lighting::contract::RecipeId;
    use prism_render_architecture::ray_scene::{Bvh, Triangle};
    use prism_render_driver::{BufferDescriptor, BufferUsages, Pass};

    let extent = Extent3d::new_2d(4, 4);
    let mut runtime = RenderRuntime::headless(NullInstance::new(), extent)
        .expect("the null runtime always comes up");
    let lighting = LightingGpuState::new(&mut runtime).expect("lighting GPU bundle");
    let plan = LightingPlan {
        tier: Tier::Balanced,
        recipes: vec![RecipeId::WorldRestir],
        partition: Default::default(),
        budget: FrameResourceBudget::default(),
        rejected: Vec::new(),
    };
    let mut extracted = ExtractedLightingInputs::default();
    extracted.surfaces = vec![LightingSurfaceInput::default()];
    lighting.prepare(&plan, &extracted, runtime.queue());

    let triangle = Triangle::new([-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [0.0, 1.0, -1.0], 0);
    let bvh = Bvh::build(&[triangle]);
    let scene = PackedTraceScene::from(
        &prism_render_architecture::ray_scene::GpuBvhBuffers::from_bvh(&bvh),
    );
    let rays = [TraceRay {
        origin: [0.0, 0.0, 0.0],
        t_min: 0.0,
        direction: [0.0, 0.0, -1.0],
        t_max: 10.0,
    }];
    let output = runtime.device().create_buffer(&BufferDescriptor::new(
        32,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    ));
    let mut trace = LightingTraceGpu::new(runtime.device_handle());
    let borrowed_scene = crate::lighting_trace_gpu::BorrowedPackedTraceScene::from(&scene);
    let recorder = trace
        .prepare_graph(runtime.queue(), &borrowed_scene, &rays, output, 32)
        .expect("trace inputs prepare without submitting a command buffer");
    let target = RenderTarget::new(runtime.device(), extent);
    let queues = RenderQueues::new();
    let functions = DrawFunctions::new();

    let absent = crate::frame::target_graph(&target, &queues, &functions, lighting.pass(&plan));
    assert_eq!(
        absent
            .blackboard()
            .get::<crate::frame::FrameGraphStages>()
            .unwrap()
            .trace_dispatches,
        0
    );

    let graph = crate::frame::target_graph_with_trace(
        &target,
        &queues,
        &functions,
        lighting.pass(&plan),
        Some(recorder),
    );
    let stages = *graph
        .blackboard()
        .get::<crate::frame::FrameGraphStages>()
        .expect("trace-aware graph publishes stage receipt");
    stages.assert_contract();
    assert_eq!(stages.trace_dispatches, 1);
    assert_eq!(stages.gi_recipes, 1);

    let capture = CapturingQueue::default();
    graph
        .execute(runtime.device(), &capture)
        .expect("trace and GI compose on the null graph")
        .release(runtime.device());
    let labels = capture
        .take()
        .into_iter()
        .flat_map(|buffer| buffer.passes)
        .filter_map(|pass| match pass {
            Pass::Render { descriptor, .. } => descriptor.label,
            Pass::Compute { label, .. } => label,
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        labels,
        [
            "prism-render-app-clear",
            "prism-lighting-trace-dispatch",
            "prism-lighting-gi-dispatch",
            "prism-lighting-composite",
        ]
    );
}

#[test]
fn offscreen_frame_graph_runs_shared_display_transform_before_terminal_output() {
    use prism_render_architecture::lighting::ocio::{
        OcioDisplayGamut, OcioToneMapping, OcioTransferFunction,
    };
    use prism_render_driver::Pass;

    let extent = Extent3d::new_2d(8, 4);
    let mut runtime = RenderRuntime::headless(NullInstance::new(), extent)
        .expect("the null runtime always comes up");
    let target = RenderTarget::new(runtime.device(), extent);
    let config = prism_render_architecture::lighting::ocio::OcioConfig {
        tone_mapping: OcioToneMapping::Aces,
        display_gamut: OcioDisplayGamut::DisplayP3,
        transfer_function: OcioTransferFunction::Gamma22,
        ..Default::default()
    };
    let mut transform = crate::display_transform_gpu::OffscreenDisplayTransform::new(config);
    transform
        .ensure(
            runtime.device_handle(),
            target.view(),
            extent,
            runtime.device_generation(),
        )
        .expect("the display transform resources are prepared");
    let graph = crate::frame::target_graph_with_display_transform(
        &target,
        &RenderQueues::new(),
        &DrawFunctions::new(),
        None,
        &transform,
        runtime.queue(),
    )
    .expect("the configured transform appends to the host graph");
    let stages = *graph
        .blackboard()
        .get::<crate::frame::FrameGraphStages>()
        .expect("the host frame graph publishes its stages");
    stages.assert_contract();
    assert!(transform.output_texture().is_some());
    let capture = CapturingQueue::default();
    graph
        .execute(runtime.device(), &capture)
        .expect("the offscreen transform graph records on the null backend")
        .release(runtime.device());
    let labels = capture
        .take()
        .into_iter()
        .flat_map(|buffer| buffer.passes)
        .filter_map(|pass| match pass {
            Pass::Compute { label, .. } => label,
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(labels, ["ocio-display-transform"]);
}

#[test]
fn lighting_graph_skips_direct_when_only_world_restir_is_admitted() {
    use prism_render_architecture::lighting::contract::RecipeId;
    use prism_render_driver::Pass;

    let extent = Extent3d::new_2d(4, 4);
    let mut runtime = RenderRuntime::headless(NullInstance::new(), extent)
        .expect("the null runtime always comes up");
    let lighting = LightingGpuState::new(&mut runtime).expect("lighting GPU bundle");
    let plan = LightingPlan {
        tier: Tier::Balanced,
        recipes: vec![RecipeId::WorldRestir],
        partition: Default::default(),
        budget: FrameResourceBudget::default(),
        rejected: Vec::new(),
    };
    let mut extracted = ExtractedLightingInputs::default();
    extracted.surfaces = vec![LightingSurfaceInput::default()];
    lighting.prepare(&plan, &extracted, runtime.queue());

    let target = RenderTarget::new(runtime.device(), extent);
    let graph = crate::frame::target_graph(
        &target,
        &RenderQueues::new(),
        &DrawFunctions::new(),
        lighting.pass(&plan),
    );
    let capture = CapturingQueue::default();
    graph
        .execute(runtime.device(), &capture)
        .expect("the GI-only lighting graph records on the null backend")
        .release(runtime.device());
    let labels = capture
        .take()
        .into_iter()
        .flat_map(|buffer| buffer.passes)
        .filter_map(|pass| match pass {
            Pass::Compute { label, .. } => label,
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        labels,
        ["prism-lighting-gi-dispatch", "prism-lighting-composite",]
    );
}

#[test]
fn lighting_graph_has_no_compute_pass_without_an_admitted_backend() {
    use prism_render_driver::Pass;

    let extent = Extent3d::new_2d(4, 4);
    let mut runtime = RenderRuntime::headless(NullInstance::new(), extent)
        .expect("the null runtime always comes up");
    let lighting = LightingGpuState::new(&mut runtime).expect("lighting GPU bundle");
    let plan = LightingPlan {
        tier: Tier::Balanced,
        recipes: Vec::new(),
        partition: Default::default(),
        budget: FrameResourceBudget::default(),
        rejected: Vec::new(),
    };
    lighting.prepare(&plan, &ExtractedLightingInputs::default(), runtime.queue());

    let target = RenderTarget::new(runtime.device(), extent);
    let graph = crate::frame::target_graph(
        &target,
        &RenderQueues::new(),
        &DrawFunctions::new(),
        lighting.pass(&plan),
    );
    let capture = CapturingQueue::default();
    graph
        .execute(runtime.device(), &capture)
        .expect("the graph without lighting records on the null backend")
        .release(runtime.device());
    assert!(!capture
        .take()
        .into_iter()
        .flat_map(|buffer| buffer.passes)
        .any(|pass| matches!(pass, Pass::Compute { .. })));
}

#[test]
fn lighting_empty_surface_batch_clears_outputs_without_dispatching() {
    use prism_render_architecture::lighting::contract::RecipeId;
    use prism_render_driver::Pass;

    let extent = Extent3d::new_2d(4, 4);
    let mut runtime = RenderRuntime::headless(NullInstance::new(), extent)
        .expect("the null runtime always comes up");
    let lighting = LightingGpuState::new(&mut runtime).expect("lighting GPU bundle");
    let plan = LightingPlan {
        tier: Tier::Balanced,
        recipes: vec![RecipeId::SpecializedDeterministic, RecipeId::WorldRestir],
        partition: Default::default(),
        budget: FrameResourceBudget::default(),
        rejected: Vec::new(),
    };
    let extracted = ExtractedLightingInputs::default();
    lighting.prepare(&plan, &extracted, runtime.queue());
    assert!(
        lighting.pass(&plan).is_none(),
        "empty batches have no lighting pass"
    );

    let target = RenderTarget::new(runtime.device(), extent);
    let graph = crate::frame::target_graph(
        &target,
        &RenderQueues::new(),
        &DrawFunctions::new(),
        lighting.pass(&plan),
    );
    let capture = CapturingQueue::default();
    graph
        .execute(runtime.device(), &capture)
        .expect("the empty-batch graph records")
        .release(runtime.device());
    assert!(!capture
        .take()
        .into_iter()
        .flat_map(|buffer| buffer.passes)
        .any(|pass| matches!(pass, Pass::Compute { .. })));
}

/// `mesh_bin_draw` records a GPU-driven bin draw: bind the pipeline and
/// geometry, then issue one multi-draw over the item's indirect argument range.
#[test]
fn mesh_bin_draw_records_a_multi_draw() {
    use prism_render_driver::{
        BufferState, BufferUsages, CommandBuffer, IndexFormat, Pass, RenderCommand,
    };
    use prism_render_graph::{
        BufferDesc, ImportedBuffer, ImportedTexture, PassFlags, RenderGraph, TextureDesc,
    };

    let options = RequestAdapterOptions::default();
    let adapter = NullInstance::new()
        .request_adapter(&options)
        .expect("the null backend always has an adapter");
    let (device, _queue) = NullInstance::new()
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null device always comes up");

    let vertices = null_vertex_buffer(&device);
    let args_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("bin-args".into()),
        size: 64,
        usage: BufferUsages::INDIRECT | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let indices = device.create_buffer(&BufferDescriptor {
        label: Some("bin-indices".into()),
        size: 64,
        usage: BufferUsages::INDEX,
        mapped_at_creation: false,
    });
    let target = device.create_texture(&TextureDescriptor::new_2d(
        Extent3d::new_2d(4, 4),
        TextureFormat::Rgba8Unorm,
        TextureUsages::RENDER_ATTACHMENT,
    ));
    let view = device.create_texture_view(target, &Default::default());
    let module = device.create_shader_module(&ShaderModuleDescriptor::wgsl(
        "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }",
    ));
    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("bin-mesh".into()),
        layout: None,
        vertex: VertexState {
            module,
            entry_point: "vs".into(),
            buffers: vec![],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: None,
    });

    let mut functions = DrawFunctions::new();
    let draw = functions.register(crate::mesh_bin_draw);

    let mut graph = RenderGraph::new(Extent3d::new_2d(4, 4));
    let target = graph.import_texture(
        "target",
        TextureDesc::color(TextureFormat::Rgba8Unorm),
        ImportedTexture {
            texture: target,
            default_view: view,
            entry_state: prism_render_driver::TextureState::initial(),
            exit_state: prism_render_driver::TextureState::copy_src(),
            mip_levels: 1,
            array_layers: 1,
        },
    );
    let args = graph.import_buffer(
        "bin-args",
        BufferDesc::new(64, BufferUsages::INDIRECT),
        ImportedBuffer {
            buffer: args_buffer,
            entry_state: BufferState::initial(),
        },
    );

    let mut queues = RenderQueues::new();
    queues.push(
        RenderPhase::Opaque,
        PhaseItem {
            pipeline,
            draw,
            geometry: GenerationalHandle::new(1, 1),
            vertices: vec![vertices],
            indices: Some((indices, IndexFormat::Uint16)),
            first: 0,
            count: 0,
            base_vertex: 0,
            indirect: Some(crate::IndirectArgs {
                args,
                offset: 0,
                count: 1,
                stride: prism_render_driver::DRAW_INDEXED_INDIRECT_ARGS_SIZE,
            }),
            lod_level: 0,
            material: GenerationalHandle::new(2, 1),
            scene: GenerationalHandle::new(3, 1),
            order: 0,
        },
    );

    graph.add_raster_pass("bin-draw", PassFlags::SIDE_EFFECT, move |builder| {
        let _ = builder.color_attachment(target);
        let _ = builder.read_indirect(args);
        move |ctx| queues.record(RenderPhase::Opaque, ctx, &functions)
    });

    let capture = CapturingQueue::default();
    graph
        .execute(&device, &capture)
        .expect("the graph executes")
        // CapturingQueue only clones commands; it never starts GPU work.
        .release(&device);

    let buffers: Vec<CommandBuffer> = capture.take();
    let commands = buffers
        .iter()
        .find_map(|buffer| match buffer.passes.first() {
            Some(Pass::Render { commands, .. }) => Some(commands),
            _ => None,
        })
        .expect("the raster pass was recorded");
    assert!(
        matches!(commands.first(), Some(RenderCommand::SetPipeline(_))),
        "the bin's pipeline is bound first"
    );
    assert!(
        commands
            .iter()
            .any(|command| matches!(command, RenderCommand::MultiDrawIndirect { count: 1, .. })),
        "one multi-draw is issued from the indirect argument range"
    );
}

/// An argument buffer declared in a throwaway graph, for tests that need a
/// graph-owned [`BufferHandle`] without recording a frame.
fn declared_args_buffer(
    device: &prism_render_driver_null::NullDevice,
) -> prism_render_graph::BufferHandle {
    use prism_render_graph::{BufferDesc, ImportedBuffer, RenderGraph};
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("declared-args".into()),
        size: 128,
        usage: BufferUsages::INDIRECT | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut graph = RenderGraph::new(Extent3d::new_2d(4, 4));
    graph.import_buffer(
        "declared-args",
        BufferDesc::new(128, BufferUsages::INDIRECT),
        ImportedBuffer {
            buffer,
            entry_state: prism_render_driver::BufferState::initial(),
        },
    )
}

/// The bin-path fixture: one geometry and mesh class, two materials whose
/// permutation classes differ (lit opaque vs lit transparent), one instance per
/// material, and the work list naming both.
struct BinFixture {
    registry: SceneRegistry,
    meshes: crate::RenderAssets<crate::RenderMesh>,
    instances: CpuRenderScene,
    geometry: GeometryHandle,
    opaque: SceneHandle,
    transparent: SceneHandle,
    work: Vec<GpuRenderWorkItem>,
}

fn bin_fixture(
    device: &prism_render_driver_null::NullDevice,
    queue: &impl RenderQueue,
) -> BinFixture {
    use crate::scene::MeshClass;
    use prism_render_architecture::geometry::{
        GeometryLodRecord, GeometryPrimitiveKind, GeometryRecord,
    };
    use prism_render_architecture::gpu_scene::{
        InstanceRecord, SceneOperation, SceneTransactionBuilder,
    };
    use prism_render_material::{fallback_material_record, MaterialRenderClass};
    use prism_render_visibility::{
        GpuRenderWorkItem, RenderPassMask, VisibilityStageMask, WorkSortKey,
    };

    let geometry: GeometryHandle = GenerationalHandle::new(1, 1);
    let opaque_material = GenerationalHandle::new(2, 1);
    let transparent_material = GenerationalHandle::new(2, 2);
    let opaque: SceneHandle = GenerationalHandle::new(3, 1);
    let transparent: SceneHandle = GenerationalHandle::new(4, 1);
    let layout = prism_render_driver::VertexBufferLayout::packed(
        prism_render_driver::VertexStepMode::Vertex,
        0,
        &[prism_render_driver::VertexFormat::Float32x3],
    );

    let mut registry = SceneRegistry::new();
    registry.insert_geometry(GeometryRecord {
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
    registry.insert_mesh_class(
        5,
        MeshClass::new(vec![layout], PrimitiveTopology::TriangleList),
    );
    // The geometry's own mesh owns the buffers the draws bind.
    let mesh_source = prism_mesh::Mesh::new(PrimitiveTopology::TriangleList)
        .with_attribute(
            prism_mesh::MeshVertexAttribute::new(
                "Vertex_Position",
                prism_render_driver::VertexFormat::Float32x3,
                0,
            ),
            prism_mesh::VertexAttributeValues::Float32x3(vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]),
        )
        .expect("positions fit")
        .indexed(prism_mesh::Indices::Uint16(vec![0, 1, 2, 0, 2, 3]));
    let mesh_id = prism_asset::AssetId::new(prism_asset::AssetIndex::from_parts(0, 1));
    let mut meshes = crate::RenderAssets::<crate::RenderMesh>::new();
    {
        use crate::render_asset::RenderAsset;
        let prepared = crate::RenderMesh::prepare(&mesh_source, mesh_id, device, queue, None)
            .expect("the fixture mesh prepares");
        meshes.insert(mesh_id, prepared);
    }
    registry.bind_geometry_mesh(geometry, mesh_id);
    registry.insert_material(fallback_material_record(opaque_material, 1));
    let mut clear = fallback_material_record(transparent_material, 1);
    clear.render_class = MaterialRenderClass::Transparent;
    registry.insert_material(clear);

    // Compiled-table handoff is explicit: material records do not smuggle a
    // packed specialization or an implicit fallback into the pipeline cache.
    for (material, module, phase) in [
        (opaque_material, ModuleId(1), RenderPhase::Opaque),
        (transparent_material, ModuleId(2), RenderPhase::Transparent),
    ] {
        registry
            .set_material_passes(
                material,
                vec![crate::PassDescriptor {
                    role: PassRole::Surface,
                    execution_family: ExecutionFamily::FusedBuiltin,
                    response_module_id: module,
                    permutation_index: dense(1),
                    fallback_permutation: Some(dense(63)),
                    render_phase: phase,
                    draw: None,
                    sort_class: crate::PassDescriptorClass::SURFACE,
                }],
            )
            .unwrap();
    }

    let mut builder = SceneTransactionBuilder::new(0, 1);
    for (handle, material) in [
        (opaque, opaque_material),
        (transparent, transparent_material),
    ] {
        builder.push(SceneOperation::Create {
            handle,
            record: InstanceRecord {
                geometry,
                material,
                ..Default::default()
            },
        });
    }
    let mut instances = CpuRenderScene::default();
    let report = instances.apply(&builder.finish());
    assert_eq!(report.created, 2, "the fixture scene holds both instances");
    assert_eq!(instances.live_count(), 2);

    let item = |scene, material| GpuRenderWorkItem {
        scene,
        geometry,
        material,
        lod_or_cluster: 0,
        pass_mask: RenderPassMask::OPAQUE,
        visibility_stages: VisibilityStageMask::EARLY,
        sort_key: WorkSortKey::new(0, 0, 0, 0, 3),
    };
    let work = vec![
        item(opaque, opaque_material),
        item(transparent, transparent_material),
    ];

    BinFixture {
        registry,
        meshes,
        instances,
        geometry,
        opaque,
        transparent,
        work,
    }
}

/// The bin path end to end: the visible set bins by permutation class, each
/// bin's packed slot range becomes one indirect item in its phase, and the frame
/// records one multi-draw per bin.
#[test]
fn bin_stream_fills_the_indirect_queue_and_records_a_multi_draw() {
    use prism_render_driver::{BufferState, IndexFormat, Pass, RenderCommand};
    use prism_render_graph::{
        BufferDesc, ImportedBuffer, ImportedTexture, PassFlags, RenderGraph, TextureDesc,
    };
    use prism_render_visibility::build_view_draw_bins;

    let options = RequestAdapterOptions::default();
    let adapter = NullInstance::new()
        .request_adapter(&options)
        .expect("the null backend always has an adapter");
    let (device, queue) = NullInstance::new()
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null device always comes up");
    let fixture = bin_fixture(&device, &queue);
    let provider = StubProvider::new(&device);

    let candidates = fixture
        .registry
        .bin_candidates(&fixture.meshes, &fixture.work);
    assert_eq!(candidates.len(), 2, "both instances are drawable");
    let bins = build_view_draw_bins(GenerationalHandle::new(0, 1), 8, candidates);
    assert_eq!(bins.bins.len(), 2, "two permutation classes bin separately");
    assert_eq!(bins.command_count, 2, "one command slot per instance");

    let target = device.create_texture(&TextureDescriptor::new_2d(
        Extent3d::new_2d(4, 4),
        TextureFormat::Rgba8Unorm,
        TextureUsages::RENDER_ATTACHMENT,
    ));
    let target_view = device.create_texture_view(target, &Default::default());
    let args_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("bin-args".into()),
        size: 128,
        usage: BufferUsages::INDIRECT | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut graph = RenderGraph::new(Extent3d::new_2d(4, 4));
    let target = graph.import_texture(
        "target",
        TextureDesc::color(TextureFormat::Rgba8Unorm),
        ImportedTexture {
            texture: target,
            default_view: target_view,
            entry_state: prism_render_driver::TextureState::initial(),
            exit_state: prism_render_driver::TextureState::copy_src(),
            mip_levels: 1,
            array_layers: 1,
        },
    );
    let args = graph.import_buffer(
        "bin-args",
        BufferDesc::new(128, BufferUsages::INDIRECT),
        ImportedBuffer {
            buffer: args_buffer,
            entry_state: BufferState::initial(),
        },
    );

    let mut functions = DrawFunctions::new();
    let draw = functions.register(crate::mesh_bin_draw);
    let mut queues = RenderQueues::new();
    let mut specializer = MeshPipelineSpecializer::new();
    let outcome = queues.fill_bins(
        &device,
        &mut specializer,
        &fixture.registry,
        &fixture.meshes,
        &fixture.instances,
        args,
        &bins,
        None,
        1,
        draw,
        &provider,
    );
    assert_eq!(
        outcome,
        crate::BinQueueOutcome {
            queued: 2,
            work_items: 2,
            unknown_class: 0,
            unresolved: 0
        }
    );
    assert_eq!(
        specializer.pipeline_count(),
        4,
        "one pipeline plus fallback per class"
    );

    // Each bin lands in the phase its packed class names, and its argument
    // range is its own slot range in the packed stream.
    let opaque = queues.phase(RenderPhase::Opaque);
    let clear = queues.phase(RenderPhase::Transparent);
    assert_eq!(opaque.len(), 1, "the opaque bin queues in the opaque phase");
    assert_eq!(clear.len(), 1, "the transparent bin is blend-family phased");
    assert_eq!(
        (opaque[0].scene, opaque[0].geometry),
        (fixture.opaque, fixture.geometry),
        "the opaque bin represents the opaque instance's geometry"
    );
    assert_eq!(
        clear[0].scene, fixture.transparent,
        "the transparent bin represents its own instance"
    );
    let opaque_args = opaque[0].indirect.expect("a bin item carries args");
    assert_eq!(opaque_args.offset, 0, "the first bin owns slot 0");
    assert_eq!(opaque_args.count, 1);
    let clear_args = clear[0].indirect.expect("a bin item carries args");
    assert_eq!(clear_args.offset, 20, "the second bin owns slot 1");
    assert_eq!(clear_args.count, 1);
    assert_eq!(
        clear_args.stride,
        prism_render_driver::DRAW_INDEXED_INDIRECT_ARGS_SIZE
    );
    assert_eq!(
        opaque[0].indices.map(|(_, format)| format),
        Some(IndexFormat::Uint16)
    );

    graph.add_raster_pass("bin-draw", PassFlags::SIDE_EFFECT, move |builder| {
        let _ = builder.color_attachment(target);
        let _ = builder.read_indirect(args);
        move |ctx| {
            queues.record(RenderPhase::Opaque, ctx, &functions);
            queues.record(RenderPhase::Transparent, ctx, &functions);
        }
    });

    let capture = CapturingQueue::default();
    graph
        .execute(&device, &capture)
        .expect("the graph executes")
        // CapturingQueue only clones commands; it never starts GPU work.
        .release(&device);
    let buffers: Vec<CommandBuffer> = capture.take();
    let commands = buffers
        .iter()
        .find_map(|buffer| match buffer.passes.first() {
            Some(Pass::Render { commands, .. }) => Some(commands),
            _ => None,
        })
        .expect("the raster pass was recorded");
    let multi: Vec<(u32, u64)> = commands
        .iter()
        .filter_map(|command| match command {
            RenderCommand::MultiDrawIndirect { offset, count, .. } => Some((*count, *offset)),
            _ => None,
        })
        .collect();
    assert_eq!(
        multi,
        vec![(1, 0), (1, 20)],
        "one multi-draw per bin, in packed slot order"
    );
}

/// A split material owns three pass descriptors: both response arms and the
/// resident composition kernel.  The bin path must preserve that pass set,
/// reuse the same indirect range for every item, and never collapse the split
/// material back to one surface pipeline.
#[test]
fn split_bin_expands_to_three_work_items_with_one_indirect_range() {
    use prism_render_visibility::build_view_draw_bins;

    let adapter = NullInstance::new()
        .request_adapter(&RequestAdapterOptions::default())
        .expect("the null backend always has an adapter");
    let (device, queue) = NullInstance::new()
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null device always comes up");
    let mut fixture = bin_fixture(&device, &queue);
    let material = fixture.work[0].material;
    fixture
        .registry
        .set_material_passes(
            material,
            vec![
                crate::PassDescriptor {
                    role: PassRole::ResponseLeft,
                    execution_family: ExecutionFamily::SplitResponse,
                    response_module_id: ModuleId(10),
                    permutation_index: dense(1),
                    fallback_permutation: Some(dense(63)),
                    render_phase: RenderPhase::Opaque,
                    draw: Some(DrawFunctionId(0)),
                    sort_class: crate::PassDescriptorClass::SURFACE,
                },
                crate::PassDescriptor {
                    role: PassRole::ResponseRight,
                    execution_family: ExecutionFamily::SplitResponse,
                    response_module_id: ModuleId(11),
                    permutation_index: dense(1),
                    fallback_permutation: Some(dense(63)),
                    render_phase: RenderPhase::Opaque,
                    draw: Some(DrawFunctionId(0)),
                    sort_class: crate::PassDescriptorClass::SURFACE,
                },
                crate::PassDescriptor {
                    role: PassRole::Composition,
                    execution_family: ExecutionFamily::Composition,
                    response_module_id: ModuleId(12),
                    permutation_index: PermutationIndex::ZERO,
                    fallback_permutation: None,
                    render_phase: RenderPhase::Opaque,
                    draw: Some(DrawFunctionId(1)),
                    sort_class: crate::PassDescriptorClass::SURFACE,
                },
            ],
        )
        .expect("the split pass set is a valid resident descriptor shape");

    let candidates = fixture
        .registry
        .bin_candidates(&fixture.meshes, &fixture.work);
    let bins = build_view_draw_bins(GenerationalHandle::new(0, 1), 8, candidates);
    assert_eq!(bins.bins.len(), 2, "the split material remains one bin");
    let split_bin = bins
        .bins
        .iter()
        .find(|range| range.representative_scene == fixture.opaque)
        .expect("the split material bin is present");

    let mut queues = RenderQueues::new();
    let mut specializer = MeshPipelineSpecializer::new();
    let provider = StubProvider::new(&device);
    let outcome = queues.fill_bins(
        &device,
        &mut specializer,
        &fixture.registry,
        &fixture.meshes,
        &fixture.instances,
        declared_args_buffer(&device),
        &bins,
        None,
        1,
        DrawFunctionId(0),
        &provider,
    );

    assert_eq!(outcome.queued, 2);
    assert_eq!(
        outcome.work_items, 4,
        "one fused bin plus three split items"
    );
    assert_eq!(outcome.unknown_class, 0);
    assert_eq!(outcome.unresolved, 0);

    let split_items: Vec<_> = queues
        .phase(RenderPhase::Opaque)
        .iter()
        .filter(|item| item.scene == fixture.opaque)
        .collect();
    assert_eq!(split_items.len(), 3);
    let expected = bin_indirect_args(
        split_items[0]
            .indirect
            .expect("split item has indirect args")
            .args,
        split_bin,
        bins.command_buffer_start,
    );
    for item in &split_items {
        assert_eq!(item.indirect, Some(expected));
    }
    assert_eq!(
        split_items
            .iter()
            .map(|item| item.draw)
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([DrawFunctionId(0), DrawFunctionId(1)]),
        "the composition item keeps its distinct draw function"
    );
}

/// The invariant that item ④/⑤ exist to preserve: the direct path and the bin
/// path must reach the *same* pipeline for the same material. The bin key only
/// carries the compact class, so a decode that produced a different shader
/// identity would compile a second PSO for identical draws.
#[test]
fn direct_and_bin_paths_specialize_one_pipeline_per_material() {
    use prism_render_visibility::build_view_draw_bins;

    let options = RequestAdapterOptions::default();
    let adapter = NullInstance::new()
        .request_adapter(&options)
        .expect("the null backend always has an adapter");
    let (device, queue) = NullInstance::new()
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null device always comes up");
    let fixture = bin_fixture(&device, &queue);
    let provider = StubProvider::new(&device);

    let mut queues = RenderQueues::new();
    let mut specializer = MeshPipelineSpecializer::new();
    let queued = queues.fill_from_visibility(
        &device,
        &mut specializer,
        &fixture.registry,
        &fixture.meshes,
        &fixture.work,
        None,
        1,
        DrawFunctionId(0),
        &provider,
    );
    assert_eq!(queued, 2);
    assert_eq!(specializer.pipeline_count(), 4);

    let bins = build_view_draw_bins(
        GenerationalHandle::new(0, 1),
        8,
        fixture
            .registry
            .bin_candidates(&fixture.meshes, &fixture.work),
    );
    let outcome = queues.fill_bins(
        &device,
        &mut specializer,
        &fixture.registry,
        &fixture.meshes,
        &fixture.instances,
        declared_args_buffer(&device),
        &bins,
        None,
        1,
        DrawFunctionId(0),
        &provider,
    );
    assert_eq!(outcome.queued, 2, "the bin path queues both instances");
    assert_eq!(
        specializer.pipeline_count(),
        4,
        "the bin path reuses the direct path's pipelines and fallbacks instead of compiling more"
    );
}

/// Packing several bin streams into one argument buffer moves each stream's
/// slots past the previous stream's, and the queued range follows the packed
/// base (the same base `GpuDrawBinHeader::from_range_at` publishes).
#[test]
fn packed_bin_streams_offset_the_indirect_arguments() {
    use prism_render_visibility::{build_view_draw_bins, pack_draw_bin_streams};

    let options = RequestAdapterOptions::default();
    let adapter = NullInstance::new()
        .request_adapter(&options)
        .expect("the null backend always has an adapter");
    let (device, queue) = NullInstance::new()
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null device always comes up");
    let fixture = bin_fixture(&device, &queue);
    let provider = StubProvider::new(&device);

    // Two views seeing the same two instances each: four commands in total.
    let mut first = build_view_draw_bins(
        GenerationalHandle::new(0, 1),
        8,
        fixture
            .registry
            .bin_candidates(&fixture.meshes, &fixture.work),
    );
    let mut second = build_view_draw_bins(
        GenerationalHandle::new(1, 1),
        8,
        fixture
            .registry
            .bin_candidates(&fixture.meshes, &fixture.work),
    );
    pack_draw_bin_streams([&mut first, &mut second]);
    assert_eq!(second.command_buffer_start, first.command_count);

    let mut queues = RenderQueues::new();
    let mut specializer = MeshPipelineSpecializer::new();
    let outcome = queues.fill_bins(
        &device,
        &mut specializer,
        &fixture.registry,
        &fixture.meshes,
        &fixture.instances,
        declared_args_buffer(&device),
        &second,
        None,
        1,
        DrawFunctionId(0),
        &provider,
    );
    assert_eq!(outcome.queued, 2);

    // The second stream's bins sit past the first stream's two slots.
    let base = u64::from(first.command_count)
        * u64::from(prism_render_driver::DRAW_INDEXED_INDIRECT_ARGS_SIZE);
    let offsets: Vec<u64> = [RenderPhase::Opaque, RenderPhase::Transparent]
        .into_iter()
        .flat_map(|phase| queues.phase(phase))
        .map(|item| item.indirect.expect("a bin item carries args").offset)
        .collect();
    assert_eq!(offsets, vec![base, base + 20]);
}

/// Unknown descriptor IDs remain drawable through the complete fallback;
/// temporarily absent geometry is still counted as unresolved.
#[test]
fn unknown_class_is_zero_or_loud() {
    use prism_render_visibility::build_view_draw_bins;

    let options = RequestAdapterOptions::default();
    let adapter = NullInstance::new()
        .request_adapter(&options)
        .expect("the null backend always has an adapter");
    let (device, queue) = NullInstance::new()
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null device always comes up");
    let fixture = bin_fixture(&device, &queue);
    let provider = StubProvider::new(&device);

    let mut candidates = fixture
        .registry
        .bin_candidates(&fixture.meshes, &fixture.work);
    // A foreign descriptor-table ID must never silently drop geometry.
    let mut foreign = candidates[0];
    foreign.key.pipeline_class = u32::MAX;
    // A bin whose vertex-buffer class has no resident pipeline inputs.
    let mut orphan = candidates[1];
    orphan.key.vertex_buffer_class = 99;
    candidates.push(orphan);

    let foreign_bins = build_view_draw_bins(GenerationalHandle::new(0, 1), 8, [foreign]);
    let mut foreign_queues = RenderQueues::new();
    let foreign_outcome = foreign_queues.fill_bins(
        &device,
        &mut MeshPipelineSpecializer::new(),
        &fixture.registry,
        &fixture.meshes,
        &fixture.instances,
        declared_args_buffer(&device),
        &foreign_bins,
        None,
        1,
        DrawFunctionId(0),
        &provider,
    );
    assert_eq!(
        foreign_outcome.queued, 1,
        "unknown bins use the complete fallback"
    );
    assert_eq!(foreign_outcome.unknown_class, 1);
    assert_eq!(foreign_queues.phase(RenderPhase::Opaque).len(), 1);

    let bins = build_view_draw_bins(GenerationalHandle::new(0, 1), 8, candidates);
    let mut queues = RenderQueues::new();
    let mut specializer = MeshPipelineSpecializer::new();
    let outcome = queues.fill_bins(
        &device,
        &mut specializer,
        &fixture.registry,
        &fixture.meshes,
        &fixture.instances,
        declared_args_buffer(&device),
        &bins,
        None,
        1,
        DrawFunctionId(0),
        &provider,
    );
    assert_eq!(
        outcome.queued, 2,
        "the foreign bin queues through fallback and the resident bin remains drawable"
    );
    assert_eq!(
        outcome.unknown_class, 0,
        "the foreign bin was tested separately"
    );
    assert_eq!(outcome.unresolved, 1, "the orphan class is not drawn");
    assert_eq!(outcome.total(), bins.bins.len());
}

/// What the startup probes recorded: how many times a GPU resource was built,
/// and whether it could see the render runtime while being built.
#[derive(Default, Resource)]
struct StartupLog {
    builds: u32,
    saw_runtime: bool,
}

/// A GPU resource: it must read the render world (the runtime) to exist, which
/// is exactly why it cannot be a `Default` value.
#[derive(Resource)]
struct StartedResource;

impl prism_ecs::resource::FromWorld for StartedResource {
    fn from_world(world: &mut prism_ecs::world::World) -> Self {
        let saw_runtime = world
            .get_resource::<RenderRuntimeResource<NullInstance>>()
            .is_some();
        let log = world.resource_mut::<StartupLog>();
        log.builds += 1;
        log.saw_runtime |= saw_runtime;
        Self
    }
}

/// The startup seam: a registered GPU resource is built once, before the first
/// frame reads it, against the render world that owns the device — and a
/// device-recovery invalidation rebuilds it.
#[test]
fn render_startup_builds_registered_gpu_resources_once_per_device() {
    use crate::startup::{GpuResourceAppExt, RenderStartupState};

    let mut app = app_on_null_backend(RenderAppConfig::new(EXTENT));
    {
        let render = app
            .sub_app_mut(RenderApp)
            .expect("the plugin registers the render sub-app");
        render.world.insert_resource(StartupLog::default());
        // Registered *after* the plugin built, as a later renderer plugin would.
        render.init_gpu_resource::<StartedResource>();
    }

    app.update();

    let render = app.get_sub_app(RenderApp).expect("the render sub-app");
    assert!(
        render.world.get_resource::<StartedResource>().is_some(),
        "the resource exists before the frame that needs it"
    );
    let log = render.world.resource::<StartupLog>();
    assert_eq!(log.builds, 1, "startup built it exactly once");
    assert!(
        log.saw_runtime,
        "a GPU resource is built against the render world's device"
    );
    assert!(
        render.world.resource::<RenderStartupState>().initialized(),
        "the startup guard is set"
    );

    // Frames after the first must not rebuild anything.
    app.update();
    app.update();
    assert_eq!(
        app.get_sub_app(RenderApp)
            .expect("the render sub-app")
            .world
            .resource::<StartupLog>()
            .builds,
        1,
        "startup is idempotent across frames"
    );

    // Device recovery: invalidating makes the next frame rebuild every resource.
    app.sub_app_mut(RenderApp)
        .expect("the render sub-app")
        .world
        .resource_mut::<RenderStartupState>()
        .invalidate();
    app.update();
    assert_eq!(
        app.get_sub_app(RenderApp)
            .expect("the render sub-app")
            .world
            .resource::<StartupLog>()
            .builds,
        2,
        "an invalidated startup rebuilds against the new device"
    );
}

/// Startup runs before the frame's `Render` schedule, so a render system can
/// read a resource a startup system inserted — regardless of registration order.
#[test]
fn render_startup_runs_before_the_render_schedule() {
    let mut app = app_on_null_backend(RenderAppConfig::new(EXTENT));
    {
        let render = app
            .sub_app_mut(RenderApp)
            .expect("the plugin registers the render sub-app");
        let schedules = render.world.resource_mut::<Schedules>();
        schedules
            .get_mut(crate::label::RenderStartup)
            .expect("the plugin registers the RenderStartup schedule")
            .add_systems(insert_startup_value);
        schedules
            .get_mut(Render)
            .expect("the plugin registers the Render schedule")
            .add_systems(assert_startup_value_present);
    }

    // The render schedule asserts the value is present when it runs.
    app.update();
}

/// A resource only a startup system creates, so a render system reading it
/// proves the startup schedule ran first.
#[derive(Resource)]
struct StartupValue(u32);

fn insert_startup_value(world: &mut prism_ecs::world::World) {
    world.insert_resource(StartupValue(7));
}

fn assert_startup_value_present(world: &mut prism_ecs::world::World) {
    assert!(
        world.resource::<StartupValue>().0 == 7,
        "RenderStartup ran before the Render schedule, with its value"
    );
}
/// The frame records what the queue holds: one pass per non-empty phase, every
/// queued draw inside it, and the queue is per-frame state that resets.
#[test]
fn the_frame_records_the_queued_draws() {
    let mut app = app_on_null_backend(RenderAppConfig::new(EXTENT));

    // A pipeline and geometry from the app's own device, so the ids the queue
    // carries are the ones the frame's runtime knows.
    let (pipeline, vertices, indices) = {
        let render = app.get_sub_app(RenderApp).expect("the render sub-app");
        let runtime = render
            .world
            .resource::<RenderRuntimeResource<NullInstance>>();
        let device = runtime.runtime().device();
        let module = device.create_shader_module(&ShaderModuleDescriptor::wgsl(
            "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }",
        ));
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("queued-draw".into()),
            layout: None,
            vertex: VertexState {
                module,
                entry_point: "vs".into(),
                buffers: vec![],
            },
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: None,
        });
        (
            pipeline,
            null_vertex_buffer(device),
            device.create_buffer(&BufferDescriptor {
                label: Some("queued-indices".into()),
                size: 64,
                usage: BufferUsages::INDEX,
                mapped_at_creation: false,
            }),
        )
    };
    let item = |pipeline, index| PhaseItem {
        pipeline,
        draw: crate::draw::canonical::MESH,
        geometry: GenerationalHandle::new(1, 1),
        vertices: vec![vertices],
        indices: Some((indices, prism_render_driver::IndexFormat::Uint16)),
        first: 0,
        count: 3,
        base_vertex: 0,
        indirect: None,
        lod_level: 0,
        material: GenerationalHandle::new(2, 1),
        scene: GenerationalHandle::new(3, 1),
        order: index,
    };
    {
        let render = app.sub_app_mut(RenderApp).expect("the render sub-app");
        let queues = render.world.resource_mut::<RenderQueues>();
        queues.push(RenderPhase::Opaque, item(pipeline, 0));
        queues.push(RenderPhase::Transparent, item(pipeline, 1));
    }

    app.update();

    let render = app.get_sub_app(RenderApp).expect("the render sub-app");
    let stats = render.world.resource::<RenderFrameStats>();
    assert_eq!(stats.draws_recorded, 2, "both queued draws were recorded");
    let stages = stats
        .execution_trace
        .graph
        .expect("the submitted frame publishes its graph receipt");
    assert_eq!(
        stats.passes_declared,
        u64::from(stages.scene_passes + stages.trace_dispatches + stages.lighting_dispatches + 1),
        "the frame graph includes trace and display-transform passes in addition to scene and lighting"
    );
    assert!(
        render.world.resource::<RenderQueues>().is_empty(),
        "the queue is per-frame state and resets after recording"
    );
}

/// The render plugin bridges a lost device into the app: reporting a loss on
/// the shared signal surfaces as a `DeviceLost` lifecycle event on the next
/// frame, which `drive_app_lifecycle` folds into `AppLifecycle::Recovering`
/// (design §12/§24.3). The signal is the only channel the render world has back
/// to the main world — the extract seam stays one-way.
#[test]
fn plugin_bridges_device_loss_to_the_app_event() {
    use prism_app::lifecycle::DeviceLost;
    use prism_ecs::event::Events;
    use prism_render_driver::DeviceLostReason;

    let mut app = app_on_null_backend(RenderAppConfig::new(EXTENT));

    let signal = app
        .world()
        .get_resource::<crate::DeviceLossSignal>()
        .expect("the plugin installs the loss signal")
        .clone();
    assert!(signal.take().is_none(), "no loss before it happens");

    // Simulate the render world recording a loss, then run a frame.
    signal.report(DeviceLostReason::Destroyed);
    app.update();

    let events = app
        .world()
        .get_resource::<Events<DeviceLost>>()
        .expect("the plugin registers the DeviceLost event");
    assert!(
        !events.is_empty(),
        "the loss is bridged to a DeviceLost event"
    );

    // The signal was drained, so a second frame does not re-emit.
    app.update();
    assert!(
        signal.take().is_none(),
        "the bridge drains the signal (no loss reported since)"
    );
}
/// The spine draws on hardware: a mesh prepared into GPU buffers is queued as a
/// phase item, the frame's opaque pass records it with the layout the mesh
/// derived, and the pixels land where the mesh's vertices say.
///
/// A cleared target is uniformly black, so a red center can only come from the
/// queued draw actually rasterizing the quad.
#[test]
fn the_frame_draws_a_queued_mesh_on_a_gpu() {
    use crate::render_asset::RenderAsset;
    use prism_mesh::{Indices, Mesh, MeshVertexAttribute, VertexAttributeValues};

    const SIZE: u32 = 8;
    let extent = Extent3d::new_2d(SIZE, SIZE);
    if RenderRuntime::<prism_render_driver_wgpu::WgpuInstance>::wgpu(extent).is_err() {
        assert!(
            !require_gpu(),
            "PRISM_REQUIRE_GPU is set but no adapter was found"
        );
        return;
    }

    let mut app = App::new();
    app.add_plugins(PrismRenderPlugin::with_config(
        prism_render_driver_wgpu::WgpuInstance::new(),
        RenderAppConfig::new(extent),
    ));

    let (pipeline, vertices, indices, index_count) = {
        let render = app.get_sub_app(RenderApp).expect("the render sub-app");
        let runtime = render
            .world
            .resource::<RenderRuntimeResource<prism_render_driver_wgpu::WgpuInstance>>();
        let device = runtime.runtime().device();
        let queue = runtime.runtime().queue();

        let mesh = Mesh::new(PrimitiveTopology::TriangleList)
            .with_attribute(
                MeshVertexAttribute::new(
                    "Vertex_Position",
                    prism_render_driver::VertexFormat::Float32x3,
                    0,
                ),
                VertexAttributeValues::Float32x3(vec![
                    [-0.5, -0.5, 0.0],
                    [0.5, -0.5, 0.0],
                    [0.5, 0.5, 0.0],
                    [-0.5, 0.5, 0.0],
                ]),
            )
            .expect("positions fit")
            .indexed(Indices::Uint16(vec![0, 1, 2, 0, 2, 3]));
        let prepared = crate::RenderMesh::prepare(
            &mesh,
            prism_asset::AssetId::new(prism_asset::AssetIndex::from_parts(0, 1)),
            device,
            queue,
            None,
        )
        .expect("the quad prepares");

        let module = device.create_shader_module(&ShaderModuleDescriptor::wgsl(
            "@vertex fn vs(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {\n\
                 return vec4<f32>(pos, 1.0);\n\
             }\n\
             @fragment fn fs() -> @location(0) vec4<f32> {\n\
                 return vec4<f32>(1.0, 0.0, 0.0, 1.0);\n\
             }",
        ));
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("spine-draw".into()),
            layout: None,
            vertex: VertexState {
                module,
                entry_point: "vs".into(),
                buffers: prepared.layout.clone(),
            },
            primitive: PrimitiveState::default(),
            // The frame has a depth attachment, so a pipeline recorded into it
            // must declare the matching depth state.
            depth_stencil: Some(prism_render_driver::DepthStencilState::reverse_z(
                crate::RENDER_DEPTH_FORMAT,
            )),
            multisample: MultisampleState::default(),
            fragment: Some(prism_render_driver::FragmentState {
                module,
                entry_point: "fs".into(),
                targets: vec![Some(prism_render_driver::ColorTargetState::opaque(
                    crate::RENDER_TARGET_FORMAT,
                ))],
            }),
        });
        let (vertices, indices) = (
            prepared.buffers.vertices.clone(),
            prepared.buffers.indices.expect("the quad is indexed").0,
        );
        (pipeline, vertices, indices, prepared.element_count())
    };

    {
        let render = app.sub_app_mut(RenderApp).expect("the render sub-app");
        render.world.resource_mut::<RenderQueues>().push(
            RenderPhase::Opaque,
            PhaseItem {
                pipeline,
                draw: crate::draw::canonical::MESH,
                geometry: GenerationalHandle::new(1, 1),
                vertices,
                indices: Some((indices, prism_render_driver::IndexFormat::Uint16)),
                first: 0,
                count: index_count,
                base_vertex: 0,
                indirect: None,
                lod_level: 0,
                material: GenerationalHandle::new(2, 1),
                scene: GenerationalHandle::new(3, 1),
                order: 0,
            },
        );
    }

    app.update();

    let render = app.get_sub_app(RenderApp).expect("the render sub-app");
    let runtime = render
        .world
        .resource::<RenderRuntimeResource<prism_render_driver_wgpu::WgpuInstance>>();
    assert_eq!(
        render.world.resource::<RenderFrameStats>().draws_recorded,
        1,
        "the frame recorded the queued mesh"
    );
    assert!(
        runtime.runtime().wait_idle(),
        "the draw completes on the GPU"
    );

    let target = *render.world.resource::<RenderTarget>();
    let pixels = runtime
        .runtime()
        .queue()
        .read_texture_rgba8(target.texture(), SIZE, SIZE);
    let pixel = |x: u32, y: u32| {
        let index = ((y * SIZE + x) * 4) as usize;
        [
            pixels[index],
            pixels[index + 1],
            pixels[index + 2],
            pixels[index + 3],
        ]
    };
    assert_eq!(
        pixel(SIZE / 2, SIZE / 2),
        [255, 0, 0, 255],
        "the queued mesh rasterized in the frame's target"
    );
    assert_eq!(
        pixel(0, 0),
        [0, 0, 0, 255],
        "outside the quad the frame's clear is what remains"
    );
}
/// The scene fixture the queue-stage tests draw: a mesh in the simulation's
/// arena, a geometry + material + binding in the render world's registry, and a
/// visible work item naming the instance.
pub(super) fn queue_scene_fixture(app: &mut App) -> GenerationalHandle {
    use prism_mesh::{Indices, Mesh, MeshVertexAttribute, VertexAttributeValues};
    use prism_render_architecture::geometry::{
        GeometryLodRecord, GeometryPrimitiveKind, GeometryRecord,
    };
    use prism_render_visibility::{
        GpuRenderWorkItem, RenderPassMask, VisibilityStageMask, WorkSortKey,
    };

    let mesh = Mesh::new(PrimitiveTopology::TriangleList)
        .with_attribute(
            MeshVertexAttribute::new(
                "Vertex_Position",
                prism_render_driver::VertexFormat::Float32x3,
                0,
            ),
            VertexAttributeValues::Float32x3(vec![
                [-0.5, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [0.5, 0.5, 0.0],
                [-0.5, 0.5, 0.0],
            ]),
        )
        .expect("positions fit")
        .indexed(Indices::Uint16(vec![0, 1, 2, 0, 2, 3]));
    let mut meshes = prism_asset::Assets::new();
    let mesh_id = meshes.insert(mesh).id();
    app.world_mut().insert_resource(meshes);

    let geometry = GenerationalHandle::new(1, 1);
    let material = GenerationalHandle::new(2, 1);
    let scene = GenerationalHandle::new(3, 1);
    let render = app
        .sub_app_mut(RenderApp)
        .expect("the plugin registers the render sub-app");
    {
        let registry = render.world.resource_mut::<SceneRegistry>();
        registry.insert_geometry(GeometryRecord {
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
        registry.insert_material(prism_render_material::fallback_material_record(material, 1));
        registry.bind_geometry_mesh(geometry, mesh_id);
    }
    let work = render.world.resource_mut::<VisibilityWorkList>();
    work.push(GpuRenderWorkItem {
        scene,
        geometry,
        material,
        lod_or_cluster: 0,
        pass_mask: RenderPassMask::OPAQUE,
        visibility_stages: VisibilityStageMask::EARLY,
        sort_key: WorkSortKey::new(0, 0, 0, 0, 3),
    });
    geometry
}

#[test]
fn the_queue_stage_turns_the_visible_set_into_draws() {
    let extent = Extent3d::new_2d(8, 8);
    let mut app = app_on_null_backend(RenderAppConfig::new(extent));
    let geometry = queue_scene_fixture(&mut app);

    app.update();

    let render = app.get_sub_app(RenderApp).expect("the render sub-app");
    let registry = render.world.resource::<SceneRegistry>();
    assert!(
        registry.mesh_class(5).is_some(),
        "the mesh class was derived from the prepared mesh"
    );
    assert!(
        registry.geometry_mesh(geometry).is_some(),
        "the geometry keeps its mesh binding"
    );
    let stats = render.world.resource::<RenderFrameStats>();
    assert_eq!(stats.draws_queued, 1, "the visible item became a draw");
    assert_eq!(stats.draws_recorded, 1, "and the frame recorded it");
    assert!(stats.passes_declared >= 2, "clear plus the opaque pass");
}
/// The whole spine on hardware: a visible work item in the scene becomes a
/// queued draw, the frame records it with the pipeline the mesh's own layout
/// specialized, and the pixels land in the frame target — no test pushing items
/// into the queue and no test registering a mesh class by hand.
#[test]
fn the_spine_draws_the_visible_set_on_a_gpu() {
    const SIZE: u32 = 8;
    let extent = Extent3d::new_2d(SIZE, SIZE);
    if RenderRuntime::<prism_render_driver_wgpu::WgpuInstance>::wgpu(extent).is_err() {
        assert!(
            !require_gpu(),
            "PRISM_REQUIRE_GPU is set but no adapter was found"
        );
        return;
    }

    let mut app = App::new();
    app.add_plugins(PrismRenderPlugin::with_config(
        prism_render_driver_wgpu::WgpuInstance::new(),
        RenderAppConfig::new(extent),
    ));
    queue_scene_fixture(&mut app);

    app.update();

    let render = app.get_sub_app(RenderApp).expect("the render sub-app");
    let stats = render.world.resource::<RenderFrameStats>();
    assert_eq!(stats.draws_queued, 1, "the visible item became a draw");
    assert_eq!(stats.draws_recorded, 1, "the frame recorded it");
    let runtime = render
        .world
        .resource::<RenderRuntimeResource<prism_render_driver_wgpu::WgpuInstance>>();
    assert!(
        runtime.runtime().wait_idle(),
        "the draw completes on the GPU"
    );

    let target = *render.world.resource::<RenderTarget>();
    let pixels = runtime
        .runtime()
        .queue()
        .read_texture_rgba8(target.texture(), SIZE, SIZE);
    let pixel = |x: u32, y: u32| {
        let index = ((y * SIZE + x) * 4) as usize;
        [
            pixels[index],
            pixels[index + 1],
            pixels[index + 2],
            pixels[index + 3],
        ]
    };
    assert_eq!(
        pixel(SIZE / 2, SIZE / 2),
        [255, 255, 255, 255],
        "the visible mesh was drawn (the spine's flat pipeline)"
    );
    assert_eq!(
        pixel(0, 0),
        [0, 0, 0, 255],
        "outside the mesh the frame's clear remains"
    );
}
fn mark_prepare_assets(mut log: ResMut<OrderLog>) {
    log.0.push("prepare-assets");
}

/// A stage with no systems in it must not break the chain around it.
///
/// Set-ordering edges expand against concrete members, so an adjacent-only chain
/// would silently disconnect the stages on either side of an empty one — which is
/// not hypothetical: `PrepareAssets` and `PrepareViews` are usually empty, and the
/// asset stage must still run before the queue stage.
#[test]
fn an_empty_stage_does_not_break_the_stage_chain() {
    let mut app = app_on_null_backend(RenderAppConfig::new(EXTENT));
    {
        let render = app
            .sub_app_mut(RenderApp)
            .expect("the plugin registers the render sub-app");
        render.world.insert_resource(OrderLog::default());
        let schedule = render
            .world
            .resource_mut::<Schedules>()
            .get_mut(Render)
            .expect("the plugin registers the Render schedule");
        // Nothing is registered in CreateViews or PrepareViews here.
        schedule.add_systems(mark_queue.in_set(RenderSystems::Queue));
        schedule.add_systems(mark_prepare_assets.in_set(RenderSystems::PrepareAssets));
    }

    app.update();

    let log = app
        .get_sub_app(RenderApp)
        .expect("the render sub-app")
        .world
        .resource::<OrderLog>();
    let prepare = log
        .0
        .iter()
        .position(|entry| *entry == "prepare-assets")
        .expect("the asset stage ran");
    let queue = log
        .0
        .iter()
        .position(|entry| *entry == "queue")
        .expect("the queue stage ran");
    assert!(
        prepare < queue,
        "the asset stage still precedes the queue stage across the empty stages"
    );
}
/// A provider that gives one colour per material permutation class, so a test
/// can tell two draws apart by their pixels.
struct ModuleColourPipeline {
    lit: RenderPipelineDescriptor,
    stylized: RenderPipelineDescriptor,
}

impl crate::mesh_pipeline::MeshPipelineProvider for ModuleColourPipeline {
    fn descriptor(&self, variant: &MeshPipelineKey) -> RenderPipelineDescriptor {
        if variant.response_module_id == ModuleId(2) {
            self.stylized.clone()
        } else {
            self.lit.clone()
        }
    }
}

/// Depth resolves occlusion, and it does so *by depth*, not by draw order: two
/// overlapping quads are drawn far-last (which would overwrite the near one if
/// nothing tested depth), and the near quad's colour must survive.
///
/// This is also what pins the engine's depth *convention*: reversed-Z with the
/// clear at the far plane (0). Clearing to 1 would reject every fragment, and a
/// pipeline without depth state would let the far quad win — either way the
/// centre pixel would not be the near quad's colour.
#[test]
fn depth_resolves_occlusion_on_a_gpu() {
    use crate::mesh_pipelines::MeshPipelines;
    use prism_render_architecture::geometry::{
        GeometryLodRecord, GeometryPrimitiveKind, GeometryRecord,
    };
    use prism_render_material::MaterialRenderClass;
    use prism_render_visibility::{
        GpuRenderWorkItem, RenderPassMask, VisibilityStageMask, WorkSortKey,
    };

    const SIZE: u32 = 8;
    let extent = Extent3d::new_2d(SIZE, SIZE);
    if RenderRuntime::<prism_render_driver_wgpu::WgpuInstance>::wgpu(extent).is_err() {
        assert!(
            !require_gpu(),
            "PRISM_REQUIRE_GPU is set but no adapter was found"
        );
        return;
    }

    let mut app = App::new();
    app.add_plugins(PrismRenderPlugin::with_config(
        prism_render_driver_wgpu::WgpuInstance::new(),
        RenderAppConfig::new(extent),
    ));

    // Two quads over the same screen area, one nearer (reversed-Z: bigger depth).
    let quad = |depth: f32| {
        prism_mesh::Mesh::new(PrimitiveTopology::TriangleList)
            .with_attribute(
                prism_mesh::MeshVertexAttribute::new(
                    "Vertex_Position",
                    prism_render_driver::VertexFormat::Float32x3,
                    0,
                ),
                prism_mesh::VertexAttributeValues::Float32x3(vec![
                    [-0.5, -0.5, depth],
                    [0.5, -0.5, depth],
                    [0.5, 0.5, depth],
                    [-0.5, 0.5, depth],
                ]),
            )
            .expect("positions fit")
            .indexed(prism_mesh::Indices::Uint16(vec![0, 1, 2, 0, 2, 3]))
    };
    let mut meshes = prism_asset::Assets::new();
    let near_mesh = meshes.insert(quad(0.9)).id();
    let far_mesh = meshes.insert(quad(0.1)).id();
    app.world_mut().insert_resource(meshes);

    let near_geometry = GenerationalHandle::new(1, 1);
    let far_geometry = GenerationalHandle::new(2, 1);
    let near_material = GenerationalHandle::new(3, 1);
    let far_material = GenerationalHandle::new(4, 1);
    // Two permutations, so the two draws specialize to different pipelines.
    let near_record = prism_render_material::fallback_material_record(near_material, 1);
    let mut far_record = prism_render_material::fallback_material_record(far_material, 1);
    far_record.render_class = MaterialRenderClass::Opaque;

    let (device, colour_pipelines) = {
        let render = app.get_sub_app(RenderApp).expect("the render sub-app");
        let runtime = render
            .world
            .resource::<RenderRuntimeResource<prism_render_driver_wgpu::WgpuInstance>>();
        let device = runtime.runtime().device();
        let descriptor = |rgb: &str| {
            let source = format!(
                "@vertex fn vs(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {{\n\
                     return vec4<f32>(position, 1.0);\n\
                 }}\n\
                 @fragment fn fs() -> @location(0) vec4<f32> {{\n\
                     return vec4<f32>({rgb});\n\
                 }}"
            );
            let module = device.create_shader_module(&ShaderModuleDescriptor {
                label: Some("depth-test".into()),
                source: prism_render_driver::ShaderSource::Wgsl(alloc::borrow::Cow::Owned(source)),
            });
            RenderPipelineDescriptor {
                label: Some("depth-test".into()),
                layout: None,
                vertex: VertexState {
                    module,
                    entry_point: "vs".into(),
                    buffers: vec![],
                },
                primitive: PrimitiveState::default(),
                depth_stencil: Some(prism_render_driver::DepthStencilState::reverse_z(
                    crate::RENDER_DEPTH_FORMAT,
                )),
                multisample: MultisampleState::default(),
                fragment: Some(prism_render_driver::FragmentState {
                    module,
                    entry_point: "fs".into(),
                    targets: vec![Some(prism_render_driver::ColorTargetState::opaque(
                        crate::RENDER_TARGET_FORMAT,
                    ))],
                }),
            }
        };
        (
            device,
            ModuleColourPipeline {
                lit: descriptor("1.0, 0.0, 0.0, 1.0"),
                stylized: descriptor("0.0, 0.0, 1.0, 1.0"),
            },
        )
    };
    let _ = device;

    let render = app.sub_app_mut(RenderApp).expect("the render sub-app");
    render
        .world
        .insert_resource(MeshPipelines::new(colour_pipelines));
    {
        let registry = render.world.resource_mut::<SceneRegistry>();
        // Both quads share class 5: they share a vertex stage while each binds
        // its own mesh's buffers.
        for (geometry, mesh) in [(near_geometry, near_mesh), (far_geometry, far_mesh)] {
            registry.insert_geometry(GeometryRecord {
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
            registry.bind_geometry_mesh(geometry, mesh);
        }
        registry.insert_material(near_record);
        registry.insert_material(far_record);
        for (handle, module) in [(near_material, ModuleId(1)), (far_material, ModuleId(2))] {
            registry
                .set_material_passes(
                    handle,
                    vec![crate::PassDescriptor {
                        role: PassRole::Surface,
                        execution_family: ExecutionFamily::FusedBuiltin,
                        response_module_id: module,
                        permutation_index: PermutationIndex::ZERO,
                        fallback_permutation: None,
                        render_phase: RenderPhase::Opaque,
                        draw: None,
                        sort_class: crate::PassDescriptorClass::SURFACE,
                    }],
                )
                .unwrap();
        }
    }
    {
        let work = render.world.resource_mut::<VisibilityWorkList>();
        // The *far* quad sorts last, so it is drawn last: without a working depth
        // test it would overwrite the near quad and the centre would be blue.
        work.push(GpuRenderWorkItem {
            scene: GenerationalHandle::new(10, 1),
            geometry: near_geometry,
            material: near_material,
            lod_or_cluster: 0,
            pass_mask: RenderPassMask::OPAQUE,
            visibility_stages: VisibilityStageMask::EARLY,
            sort_key: WorkSortKey::new(0, 0, 0, 0, 1),
        });
        work.push(GpuRenderWorkItem {
            scene: GenerationalHandle::new(11, 1),
            geometry: far_geometry,
            material: far_material,
            lod_or_cluster: 0,
            pass_mask: RenderPassMask::OPAQUE,
            visibility_stages: VisibilityStageMask::EARLY,
            sort_key: WorkSortKey::new(0, 0, 0, 0, 9),
        });
    }

    app.update();

    let render = app.get_sub_app(RenderApp).expect("the render sub-app");
    assert_eq!(
        render.world.resource::<RenderFrameStats>().draws_recorded,
        2,
        "both quads were drawn"
    );
    let runtime = render
        .world
        .resource::<RenderRuntimeResource<prism_render_driver_wgpu::WgpuInstance>>();
    assert!(runtime.runtime().wait_idle(), "the draws complete");
    let target = *render.world.resource::<RenderTarget>();
    let pixels = runtime
        .runtime()
        .queue()
        .read_texture_rgba8(target.texture(), SIZE, SIZE);
    let centre = ((SIZE / 2 * SIZE + SIZE / 2) * 4) as usize;
    assert_eq!(
        &pixels[centre..centre + 4],
        &[255, 0, 0, 255],
        "the nearer quad survives the later, farther draw"
    );
}
