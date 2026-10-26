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
    LightingAreaInput, LightingDebugViewState, LightingDirectionalInput, LightingGiCandidateInput,
    LightingPlanState, LightingPunctualInput, LightingSceneInputs, LightingSurfaceInput,
};
use crate::lighting_gpu::LightingGpuState;
use crate::plugin::{save_persisted_pso_cache, PrismRenderPlugin};
use crate::record::RenderRecord;
use crate::runtime_resource::RenderRuntimeResource;
use crate::stats::RenderFrameStats;
use crate::target::RenderTarget;
use crate::visibility_bridge::ExtractedVisibilityViews;

use crate::bin_queue::bin_indirect_args;
use crate::draw::{DrawFunctionId, DrawFunctions};
use crate::mesh_pipeline::{full_lobe_fallback, MeshPipelineKey, MeshPipelineSpecializer};
use crate::pipeline_cache::{
    hash_vertex_layouts, pso_device_fingerprint, PipelineCache, PipelineCachePersistenceStatus,
    PipelineCompiler, PipelineKey, PipelineResolution,
};
use crate::queue::{phase_for_material, phase_for_render_class};
use crate::queue::{CanonicalMaterialBindings, DrawRequest};
use crate::queue::{PhaseItem, RenderQueues};
use crate::sets::{RenderPhase, RenderSystems};
use crate::texture_cache::{TextureCache, TextureKey};
use crate::views::RenderViews;
use crate::{MaterialParameterBuffers, RenderMaterials};
use prism_ecs::prelude::{ResMut, Resource};
use prism_ecs::schedule::IntoSystemConfigs;
use prism_render_architecture::abi::GenerationalHandle;
use prism_render_architecture::gpu_scene::{CpuRenderScene, GeometryHandle, SceneHandle};
use prism_render_architecture::lighting::contract::{
    ChangeFields, FrameResourceBudget, GiCandidateAvailability, GiCandidateRejection,
    GiCandidateSource, LightingPlan, PlanCaps, RecipeId, Rejection, SceneStats, Tier,
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
        vertex_sizes: vec![64],
        indices: None,
        index_size: None,
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
        fn set_vertex_buffer(
            &mut self,
            _: u32,
            _: BufferId,
            _: u64,
            _: Option<core::num::NonZeroU64>,
        ) {
        }
        fn set_index_buffer(
            &mut self,
            _: BufferId,
            _: prism_render_driver::IndexFormat,
            _: u64,
            _: Option<core::num::NonZeroU64>,
        ) {
        }
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
                vertex_sizes: vec![64],
                indices: None,
                index_size: None,
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
                shader_package: None,
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
        vertex_sizes: vec![64],
        indices: None,
        index_size: None,
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

#[test]
fn lighting_extraction_does_not_create_a_missing_canonical_scene() {
    let mut app = app_on_null_backend(RenderAppConfig::new(EXTENT));
    assert!(
        app.world_mut()
            .get_resource::<crate::LightingLightScene>()
            .is_none(),
        "the application must explicitly publish the canonical light scene"
    );

    app.update();

    assert!(
        app.world_mut()
            .get_resource::<crate::LightingLightScene>()
            .is_none(),
        "lighting extraction must not install an implicit authority"
    );
    let render = app.get_sub_app(RenderApp).expect("render sub-app");
    let extracted = render.world.resource::<ExtractedLightingInputs>();
    assert_eq!(extracted.stats.lights, 0);
    assert!(extracted.directionals.is_empty());
    assert!(extracted.punctuals.is_empty());
    assert!(extracted.areas.is_empty());
    assert_eq!(extracted.light_scene_identity, 0);
    assert_eq!(extracted.light_scene_sequence, 0);
    assert_eq!(extracted.light_scene_epoch, 0);
    assert_eq!(extracted.light_scene_error, None);
}

/// Whether a missing adapter must fail a GPU-gated test instead of skipping it.
pub(crate) fn require_gpu() -> bool {
    std::env::var_os("PRISM_REQUIRE_GPU").is_some()
}

/// Uploads one covered material sample to the real frame target's GBuffer.
/// Albedo is quantized through the target's sRGB8 format, and the returned
/// surface carries the decoded value used by the per-pixel CPU oracle.
pub(super) fn write_lighting_test_gbuffer_pixel(
    queue: &impl RenderQueue,
    target: &RenderTarget,
    x: u32,
    y: u32,
    mut surface: LightingSurfaceInput,
    surface_id: u32,
) -> LightingSurfaceInput {
    fn encode_srgb8(linear: f32) -> u8 {
        let linear = linear.clamp(0.0, 1.0);
        let encoded = if linear <= 0.003_130_8 {
            linear * 12.92
        } else {
            1.055 * linear.powf(1.0 / 2.4) - 0.055
        };
        (encoded * 255.0).round() as u8
    }

    fn decode_srgb8(encoded: u8) -> f32 {
        let encoded = f32::from(encoded) / 255.0;
        if encoded <= 0.040_45 {
            encoded / 12.92
        } else {
            ((encoded + 0.055) / 1.055).powf(2.4)
        }
    }

    fn exact_axis_half(value: f32) -> u16 {
        if value == 0.0 {
            0x0000
        } else if value == 1.0 {
            0x3c00
        } else if value == -1.0 {
            0xbc00
        } else {
            panic!("GBuffer fixture normal must use exact axis values, got {value}");
        }
    }

    let mut upload = |texture, bytes: &[u8]| {
        queue.write_texture(
            TextureWrite {
                texture,
                mip_level: 0,
                origin: [x, y, 0],
                size: Extent3d::new_2d(1, 1),
            },
            bytes,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bytes.len() as u32),
                rows_per_image: Some(1),
            },
        );
    };

    for (texture, value) in [
        (target.position_x_texture(), surface.position[0]),
        (target.position_y_texture(), surface.position[1]),
        (target.position_z_texture(), surface.position[2]),
    ] {
        upload(texture, &value.to_le_bytes());
    }
    let normal = [
        exact_axis_half(surface.normal[0]),
        exact_axis_half(surface.normal[1]),
        exact_axis_half(surface.normal[2]),
        0x3c00,
    ];
    let normal_bytes = normal
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    upload(target.normal_texture(), &normal_bytes);

    let encoded_albedo = surface.albedo.map(encode_srgb8);
    let mut albedo_bytes = encoded_albedo.to_vec();
    albedo_bytes.push(255);
    upload(target.albedo_texture(), &albedo_bytes);
    surface.albedo = encoded_albedo.map(decode_srgb8);
    upload(target.surface_id_texture(), &surface_id.to_le_bytes());
    surface
}

/// Executes the production composite pipeline against one target's actual
/// GBuffer bindings, without invoking the frame graph's GBuffer clear first.
pub(super) fn dispatch_lighting_test_composite<I>(
    lighting: &LightingGpuState<I>,
    plan: &LightingPlan,
    target: &RenderTarget,
    runtime: &RenderRuntime<I>,
) where
    I: prism_render_driver::Instance + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: prism_render_driver::FrameSync,
{
    use prism_render_driver::{BufferUsages, CommandEncoder, ComputeCommand};

    // Production composite bind-group layout includes canonical material
    // snapshots even when this focused test only exercises direct/GI math.
    // Build and retain real snapshots for the entire submission instead of
    // dispatching an incomplete test-only pipeline binding set.
    let mut registry = prism_render_material::MaterialRegistry::new(1);
    registry
        .install_execution_tables(&prism_render_material::MaterialExecutionTables::new())
        .expect("empty canonical execution tables are valid for fallback material");
    let device = runtime.device_handle();
    let mut material_buffers =
        MaterialParameterBuffers::new_owned(device.clone(), 1, 1, 1, BufferUsages::NONE);
    material_buffers.sync_registry(runtime.device(), runtime.queue(), &registry, [0]);
    let mut execution_buffers = crate::material::NativeExecutionTableBuffers::new();
    execution_buffers
        .sync_registry(device, runtime.queue(), &registry)
        .expect("canonical execution-table upload is available");
    assert!(runtime.wait_idle(), "material snapshots publish on GPU");
    let material_snapshot = material_buffers
        .snapshot()
        .expect("material allocation snapshot");
    let execution_snapshot = execution_buffers.snapshot().expect("execution snapshot");
    let material_bindings = crate::lighting_gpu::LightingCompositeMaterialBuffers {
        headers: Some(material_snapshot.headers()),
        headers_size_bytes: u64::from(material_snapshot.header_capacity())
            * core::mem::size_of::<prism_render_material::GpuMaterialHeader>() as u64,
        parameters: Some(material_snapshot.parameters()),
        parameters_size_bytes: u64::from(material_snapshot.parameter_capacity_words()) * 4,
        execution_plans: Some(execution_snapshot.plan_buffer()),
        execution_plans_size_bytes: execution_snapshot.plan_rows().len() as u64
            * (prism_render_material::EXECUTION_PLAN_ROW_WORDS * 4) as u64,
        execution_programs: Some(execution_snapshot.program_buffer()),
        execution_programs_size_bytes: execution_snapshot.program_rows().len() as u64 * 20,
    };
    let pass = lighting
        .pass_for_target_with_materials(plan, target, runtime, material_bindings)
        .expect("prepared lighting plan binds to the real GBuffer target and material tables");
    let target_bind_group = pass
        .composite_target_bind_group
        .expect("target-aware lighting pass owns its GBuffer bindings");
    let extent = target.viewport().extent();
    let mut encoder = CommandEncoder::new();
    encoder.push_compute_pass(
        Some("prism-test-lighting-composite".into()),
        vec![
            ComputeCommand::SetPipeline(pass.composite_pipeline),
            ComputeCommand::SetBindGroup {
                index: 0,
                bind_group: pass.composite_bind_group,
                dynamic_offsets: Vec::new(),
            },
            ComputeCommand::SetBindGroup {
                index: 1,
                bind_group: target_bind_group,
                dynamic_offsets: Vec::new(),
            },
            ComputeCommand::SetBindGroup {
                index: crate::lighting_gpu::LightingGpuPass::composite_material_group(),
                bind_group: pass
                    .composite_material_bind_group
                    .expect("composite requires canonical material bindings"),
                dynamic_offsets: Vec::new(),
            },
            ComputeCommand::Dispatch {
                x: extent
                    .width
                    .div_ceil(crate::lighting_gpu::COMPOSITE_WORKGROUP_X),
                y: extent
                    .height
                    .div_ceil(crate::lighting_gpu::COMPOSITE_WORKGROUP_Y),
                z: 1,
            },
        ],
    );
    // Bind groups refer to external canonical buffers, so keep their owners
    // alive through queue completion just as the host graph does.
    runtime.queue().submit(&[encoder.finish()]);
    assert!(
        runtime.wait_idle(),
        "lighting composite completes on the GPU"
    );
    drop((
        material_snapshot,
        execution_snapshot,
        material_buffers,
        execution_buffers,
        registry,
    ));
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
    let stats = render
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
        FRAMES * 3,
        "without GI candidates each frame declares composite, clear, and OCIO passes"
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
            gpu_ms: 0.5,
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
            prefiltered_environment: false,
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

fn lighting_candidate_test_capabilities() -> prism_render_driver::DeviceCapabilities {
    prism_render_driver::DeviceCapabilities {
        backend: prism_render_driver::Backend::Noop,
        features: prism_render_driver::Features::from_bits_truncate(0),
        limits: prism_render_driver::Limits::baseline(),
    }
}

fn lighting_candidate_test_trace_scene(surface_count: usize) -> crate::LightingTraceSceneInput {
    use prism_render_architecture::ray_scene::{
        Bvh, GpuBvhBuffers, RayHitAttributeRecord, RayHitIdentity, Triangle,
    };

    let triangle = Triangle::new([-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [0.0, 1.0, -1.0], 73);
    crate::LightingTraceSceneInput::new(
        Arc::new(GpuBvhBuffers::from_bvh(&Bvh::build(&[triangle]))),
        (0..surface_count).map(|index| index as u64 + 100).collect(),
        1,
        0.001,
        10.0,
        Arc::from([crate::LightingTriangleRadiance {
            primitive_id: 73,
            rgb: [4.0, 2.0, 1.0],
        }]),
        1,
    )
}

#[test]
fn lighting_candidate_availability_is_missing_without_a_producer() {
    let caps = lighting_candidate_test_capabilities();
    let mut inputs = ExtractedLightingInputs::default();
    assert_eq!(
        inputs.candidate_availability(&caps),
        GiCandidateAvailability::Missing
    );
    inputs.surfaces = vec![LightingSurfaceInput::default(); 2];
    assert_eq!(
        inputs.candidate_availability(&caps),
        GiCandidateAvailability::Missing
    );
}

#[test]
fn lighting_candidate_availability_accepts_host_batches_aligned_with_surfaces() {
    let caps = lighting_candidate_test_capabilities();
    let mut inputs = ExtractedLightingInputs::default();
    inputs.surfaces = vec![LightingSurfaceInput::default(); 2];
    let candidate = LightingGiCandidateInput {
        target_pdf: 1.0,
        radiance: [1.0; 3],
        ..LightingGiCandidateInput::default()
    };
    // The empty batch is an explicit, valid zero-proposal surface.
    inputs.gi_candidates = vec![vec![candidate], Vec::new()];
    assert_eq!(
        inputs.candidate_availability(&caps),
        GiCandidateAvailability::HostProvided
    );
    inputs.gi_candidates = vec![Vec::new(); inputs.surfaces.len()];
    assert_eq!(
        inputs.candidate_availability(&caps),
        GiCandidateAvailability::HostProvided
    );
}

#[test]
fn lighting_candidate_availability_rejects_misaligned_or_invalid_host_batches() {
    let caps = lighting_candidate_test_capabilities();
    let mut inputs = ExtractedLightingInputs::default();
    inputs.surfaces = vec![LightingSurfaceInput::default(); 2];
    inputs.gi_candidates = vec![Vec::new()];
    assert_eq!(
        inputs.candidate_availability(&caps),
        GiCandidateAvailability::Rejected(GiCandidateRejection::InvalidInputs)
    );
    inputs.gi_candidates = vec![Vec::new(); inputs.surfaces.len()];
    inputs.gi_candidates[1].push(LightingGiCandidateInput {
        source_pdf: f32::NAN,
        ..LightingGiCandidateInput::default()
    });
    assert_eq!(
        inputs.candidate_availability(&caps),
        GiCandidateAvailability::Rejected(GiCandidateRejection::InvalidInputs)
    );
}

#[test]
fn lighting_candidate_availability_accepts_software_bvh_without_hardware_ray_tracing() {
    let mut caps = lighting_candidate_test_capabilities();
    let mut inputs = ExtractedLightingInputs::default();
    inputs.surfaces = vec![LightingSurfaceInput::default(); 2];
    inputs.trace_scene = Some(lighting_candidate_test_trace_scene(inputs.surfaces.len()));
    // A valid trace request takes precedence over a valid host producer.
    inputs.gi_candidates = vec![Vec::new(); inputs.surfaces.len()];
    assert_eq!(
        inputs.candidate_availability(&caps),
        GiCandidateAvailability::SoftwareBvhGpu
    );
    caps.features |= prism_render_driver::Features::RAY_TRACING;
    assert_eq!(
        inputs.candidate_availability(&caps),
        GiCandidateAvailability::SoftwareBvhGpu
    );
    inputs.gi_candidates.clear();
    assert_eq!(
        inputs.candidate_availability(&caps),
        GiCandidateAvailability::SoftwareBvhGpu
    );
}

#[test]
fn lighting_candidate_availability_rejects_invalid_trace_without_host_fallback() {
    let caps = lighting_candidate_test_capabilities();
    let mut inputs = ExtractedLightingInputs::default();
    inputs.surfaces = vec![LightingSurfaceInput::default(); 2];
    inputs.gi_candidates = vec![Vec::new(); inputs.surfaces.len()];
    assert_eq!(
        inputs.candidate_availability(&caps),
        GiCandidateAvailability::HostProvided
    );

    let valid = lighting_candidate_test_trace_scene(inputs.surfaces.len());
    let mut invalid = Vec::new();
    let mut scene = valid.clone();
    scene.surface_ids.pop();
    invalid.push(("surface ID count", scene));
    let mut scene = valid.clone();
    scene.surface_ids[1] = scene.surface_ids[0];
    invalid.push(("duplicate surface ID", scene));
    let mut scene = valid.clone();
    scene.rays_per_surface = 0;
    invalid.push(("zero rays", scene));
    let mut scene = valid.clone();
    scene.rays_per_surface = crate::lighting_gpu::GI_CANDIDATES_PER_SURFACE as u32 + 1;
    invalid.push(("ray capacity", scene));
    let mut scene = valid.clone();
    scene.ray_epsilon = f32::NAN;
    invalid.push(("non-finite ray epsilon", scene));
    let mut scene = valid.clone();
    scene.ray_tmax = scene.ray_epsilon;
    invalid.push(("invalid ray interval", scene));
    let mut scene = valid.clone();
    scene.outgoing_radiance = Arc::from([]);
    invalid.push(("radiance count", scene));
    let mut scene = valid.clone();
    scene.outgoing_radiance = Arc::from([crate::LightingTriangleRadiance {
        primitive_id: 74,
        rgb: [1.0; 3],
    }]);
    invalid.push(("radiance primitive ID", scene));
    let mut scene = valid;
    scene.outgoing_radiance = Arc::from([crate::LightingTriangleRadiance {
        primitive_id: 73,
        rgb: [f32::NAN, 1.0, 1.0],
    }]);
    invalid.push(("non-finite radiance", scene));

    for (case, scene) in invalid {
        inputs.trace_scene = Some(scene);
        assert_eq!(
            inputs.candidate_availability(&caps),
            GiCandidateAvailability::Rejected(GiCandidateRejection::InvalidInputs),
            "{case}: a malformed trace request must not fall back to valid host candidates"
        );
    }
}

#[test]
fn lighting_candidate_availability_checks_device_limits_for_host_and_trace() {
    let caps = lighting_candidate_test_capabilities();
    let mut inputs = ExtractedLightingInputs::default();
    inputs.surfaces = vec![LightingSurfaceInput::default(); 2];
    inputs.gi_candidates = vec![Vec::new(); inputs.surfaces.len()];
    let valid_trace = lighting_candidate_test_trace_scene(inputs.surfaces.len());

    let mut cases = Vec::new();
    let mut limited = caps;
    limited.limits.max_storage_buffer_binding_size = 0;
    cases.push((
        "storage binding",
        limited,
        GiCandidateRejection::DeviceLimits,
    ));
    let mut limited = caps;
    limited.limits.max_uniform_buffer_binding_size = 0;
    cases.push((
        "uniform binding",
        limited,
        GiCandidateRejection::DeviceLimits,
    ));
    let mut limited = caps;
    limited.limits.max_compute_workgroups_per_dimension = 0;
    cases.push((
        "dispatch workgroups",
        limited,
        GiCandidateRejection::DeviceLimits,
    ));
    let mut limited = caps;
    limited.limits.max_compute_invocations_per_workgroup = 63;
    cases.push((
        "workgroup invocations",
        limited,
        GiCandidateRejection::TraceBackendUnavailable,
    ));
    let mut limited = caps;
    limited.limits.max_bind_groups = 0;
    cases.push((
        "bind groups",
        limited,
        GiCandidateRejection::TraceBackendUnavailable,
    ));
    let mut limited = caps;
    limited.limits.max_bindings_per_bind_group = 6;
    cases.push((
        "bindings per group",
        limited,
        GiCandidateRejection::TraceBackendUnavailable,
    ));

    for trace_enabled in [false, true] {
        inputs.trace_scene = trace_enabled.then(|| valid_trace.clone());
        assert_eq!(
            inputs.candidate_availability(&caps),
            if trace_enabled {
                GiCandidateAvailability::SoftwareBvhGpu
            } else {
                GiCandidateAvailability::HostProvided
            }
        );
        for (case, limited, reason) in &cases {
            assert_eq!(
                inputs.candidate_availability(limited),
                GiCandidateAvailability::Rejected(*reason),
                "{case}, trace_enabled={trace_enabled}"
            );
        }
    }
}

#[test]
fn lighting_candidate_availability_checks_trace_workgroups_and_radiance_storage() {
    let mut caps = lighting_candidate_test_capabilities();
    let mut inputs = ExtractedLightingInputs::default();
    inputs.surfaces = vec![LightingSurfaceInput::default(); 2];
    inputs.gi_candidates = vec![Vec::new(); inputs.surfaces.len()];
    caps.limits.max_compute_workgroups_per_dimension = 1;
    assert_eq!(
        inputs.candidate_availability(&caps),
        GiCandidateAvailability::HostProvided
    );
    let mut trace = lighting_candidate_test_trace_scene(inputs.surfaces.len());
    trace.rays_per_surface = crate::lighting_gpu::GI_CANDIDATES_PER_SURFACE as u32;
    inputs.trace_scene = Some(trace);
    assert_eq!(
        inputs.candidate_availability(&caps),
        GiCandidateAvailability::Rejected(GiCandidateRejection::DeviceLimits),
        "two surfaces fit lighting dispatch but 128 trace rays need two workgroups"
    );

    caps = lighting_candidate_test_capabilities();
    caps.limits.max_storage_buffer_binding_size = crate::lighting_gpu::TRACE_RADIANCE_BYTES - 1;
    inputs.trace_scene = None;
    assert_eq!(
        inputs.candidate_availability(&caps),
        GiCandidateAvailability::HostProvided
    );
    inputs.trace_scene = Some(lighting_candidate_test_trace_scene(inputs.surfaces.len()));
    assert_eq!(
        inputs.candidate_availability(&caps),
        GiCandidateAvailability::Rejected(GiCandidateRejection::DeviceLimits),
        "the trace radiance binding must fit the actual device limit"
    );
}

#[test]
fn lighting_candidate_plugin_source_switches_keep_plan_debug_and_graph_in_the_same_frame() {
    use crate::lighting_trace_gpu::{LightingTraceState, TracePreparationStatus};

    let mut app = app_on_null_backend(RenderAppConfig::new(EXTENT));
    app.world_mut()
        .resource_mut::<LightingSceneInputs>()
        .gi_candidates = vec![Vec::new()];
    let mut trace_dispatches = 0;

    for (index, source) in [
        GiCandidateSource::HostProvided,
        GiCandidateSource::SoftwareBvhGpu,
        GiCandidateSource::Disabled,
        GiCandidateSource::SoftwareBvhGpu,
        GiCandidateSource::Disabled,
        GiCandidateSource::HostProvided,
    ]
    .into_iter()
    .enumerate()
    {
        let frame = index as u64 + 1;
        match index {
            0 => {}
            1 | 3 => {
                app.world_mut()
                    .insert_resource(lighting_candidate_test_trace_scene(1));
            }
            2 => {
                app.world_mut()
                    .remove_resource::<crate::LightingTraceSceneInput>();
                app.world_mut()
                    .resource_mut::<LightingSceneInputs>()
                    .gi_candidates
                    .clear();
            }
            4 => {
                // Reject an explicit invalid trace even though host batches are valid again.
                app.world_mut()
                    .resource_mut::<LightingSceneInputs>()
                    .gi_candidates = vec![Vec::new()];
                app.world_mut()
                    .resource_mut::<crate::LightingTraceSceneInput>()
                    .surface_ids
                    .clear();
            }
            5 => {
                app.world_mut()
                    .remove_resource::<crate::LightingTraceSceneInput>();
            }
            _ => unreachable!(),
        }
        app.world_mut()
            .resource_mut::<LightingSceneInputs>()
            .mark_changed(frame, ChangeFields::LIGHTING);
        app.update();

        let render = app.get_sub_app(RenderApp).expect("render sub-app");
        let plan = render.world.resource::<LightingPlanState>();
        let debug = render.world.resource::<LightingDebugViewState>();
        let extracted = render.world.resource::<ExtractedLightingInputs>();
        let record = render.world.resource::<RenderRecord>();
        assert_eq!(record.frame_index, frame);
        assert_eq!(plan.compiled_frame, frame);
        assert_eq!(plan.compile_count, frame);
        assert_eq!(plan.plan.candidate_source, source);
        let gi_admitted = source != GiCandidateSource::Disabled;
        assert_eq!(
            plan.plan.recipes.contains(&RecipeId::WorldRestir),
            gi_admitted
        );
        assert_eq!(plan.journal_snapshot, extracted.journal_snapshot);
        assert_eq!(debug.compiled_frame, frame);
        assert_eq!(debug.plan_compile_count, frame);
        assert_eq!(debug.explain, plan.explain);
        assert_eq!(
            debug.rejection_explanations,
            plan.plan.rejection_explanations()
        );
        assert_eq!(
            debug
                .selected_recipes
                .iter()
                .map(|entry| entry.recipe)
                .collect::<Vec<_>>(),
            plan.plan.recipes
        );
        if !gi_admitted {
            let reason = if index == 2 {
                GiCandidateRejection::MissingInputs
            } else {
                GiCandidateRejection::InvalidInputs
            };
            assert!(plan.plan.rejected.contains(&Rejection::CandidateSource {
                recipe: RecipeId::WorldRestir,
                reason
            }));
        }

        let trace_status = render.world.resource::<LightingTraceState>().status();
        let expected_trace = u32::from(source == GiCandidateSource::SoftwareBvhGpu);
        if expected_trace == 1 {
            assert_eq!(
                trace_status,
                TracePreparationStatus::Prepared {
                    rays: 1,
                    frame_index: frame
                }
            );
        } else {
            assert_eq!(trace_status, TracePreparationStatus::Disabled);
        }
        let stats = render.world.resource::<RenderFrameStats>();
        assert_eq!(stats.frames_submitted, frame);
        assert_eq!(stats.frames_failed, 0);
        assert_eq!(stats.lighting_plan_compiles, frame);
        assert_eq!(stats.lighting_debug_updates, frame);
        assert_eq!(stats.execution_trace.frame_index, frame);
        assert_eq!(
            stats.execution_trace.journal_snapshot,
            extracted.journal_snapshot
        );
        let stages = stats
            .execution_trace
            .graph
            .expect("submitted host graph publishes stages");
        stages.assert_contract();
        assert_eq!(stages.gi_recipes, u32::from(gi_admitted));
        assert_eq!(stages.trace_dispatches, expected_trace);
        assert_eq!(stages.candidate_build_dispatches, expected_trace);
        trace_dispatches += u64::from(expected_trace);
        assert_eq!(stats.trace_gpu_dispatches_recorded, trace_dispatches);
    }
}

#[test]
fn lighting_candidate_plugin_disables_valid_trace_when_gi_exceeds_budget() {
    use crate::lighting_trace_gpu::{LightingTraceState, TracePreparationStatus};

    let config = RenderAppConfig::new(EXTENT).with_lighting_budget(FrameResourceBudget {
        gpu_ms: 1.0,
        ..RenderAppConfig::DEFAULT.lighting_budget
    });
    let mut app = app_on_null_backend(config);
    app.world_mut()
        .insert_resource(lighting_candidate_test_trace_scene(1));
    app.update();

    let render = app.get_sub_app(RenderApp).expect("render sub-app");
    let inputs = render.world.resource::<ExtractedLightingInputs>();
    let runtime = render
        .world
        .resource::<RenderRuntimeResource<NullInstance>>()
        .runtime();
    assert_eq!(
        inputs.candidate_availability(runtime.device().capabilities()),
        GiCandidateAvailability::SoftwareBvhGpu
    );
    let plan = render.world.resource::<LightingPlanState>();
    assert_eq!(plan.plan.candidate_source, GiCandidateSource::Disabled);
    assert!(!plan.plan.recipes.contains(&RecipeId::WorldRestir));
    assert!(plan
        .plan
        .rejected
        .contains(&Rejection::Budget(RecipeId::WorldRestir)));
    let debug = render.world.resource::<LightingDebugViewState>();
    assert_eq!(debug.compiled_frame, plan.compiled_frame);
    assert_eq!(
        debug.rejection_explanations,
        plan.plan.rejection_explanations()
    );
    assert_eq!(
        render.world.resource::<LightingTraceState>().status(),
        TracePreparationStatus::Disabled
    );
    let stats = render.world.resource::<RenderFrameStats>();
    assert_eq!(stats.frames_submitted, 1);
    assert_eq!(stats.frames_failed, 0);
    assert_eq!(stats.trace_gpu_dispatches_recorded, 0);
    let stages = stats
        .execution_trace
        .graph
        .expect("submitted host graph publishes stages");
    stages.assert_contract();
    assert_eq!(stages.trace_dispatches, 0);
    assert_eq!(stages.candidate_build_dispatches, 0);
    assert_eq!(stages.gi_recipes, 0);
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
    let result = evaluate_deterministic_gi(surface, &[candidate]);
    let expected = [0.75, 0.75, 6.0].map(|value| value / core::f32::consts::PI);
    for (actual, expected) in result.into_iter().zip(expected) {
        assert!((actual - expected).abs() < 1.0e-6);
    }
}

#[test]
fn deterministic_gi_ris_counts_zero_contribution_proposals_in_normalization() {
    let surface = LightingSurfaceInput {
        position: [0.0, 0.0, 0.0],
        normal: [0.0, 1.0, 0.0],
        ..LightingSurfaceInput::default()
    };
    let contributing = LightingGiCandidateInput {
        position: [0.0, 2.0, 0.0],
        source_pdf: 1.0,
        normal: [0.0, -1.0, 0.0],
        target_pdf: 2.0,
        radiance: [1.0; 3],
        selection_u01: 0.0,
        visibility: 1.0,
        ..LightingGiCandidateInput::default()
    };
    let miss = LightingGiCandidateInput {
        source_pdf: 0.25,
        target_pdf: 0.0,
        ..LightingGiCandidateInput::default()
    };
    let one_proposal = evaluate_deterministic_gi(surface, &[contributing]);
    let two_proposals = evaluate_deterministic_gi(surface, &[contributing, miss]);
    for (one, two) in one_proposal.into_iter().zip(two_proposals) {
        assert!((two * 2.0 - one).abs() < 1.0e-6);
    }
}

#[test]
fn deterministic_gi_oracle_ignores_invalid_and_primary_backfacing_candidates() {
    let surface = LightingSurfaceInput::default();
    let invalid = LightingGiCandidateInput {
        source_pdf: 0.0,
        target_pdf: 1.0,
        ..LightingGiCandidateInput::default()
    };
    // With the solid-angle measure the candidate normal is not part of the
    // Lambert directional integrand. A sample is rejected here only because
    // its direction is behind the owning surface normal.
    let primary_backfacing = LightingGiCandidateInput {
        position: [0.0, -2.0, 0.0],
        normal: [0.0, -1.0, 0.0],
        source_pdf: 1.0,
        target_pdf: 1.0,
        radiance: [10.0; 3],
        ..LightingGiCandidateInput::default()
    };
    assert_eq!(
        evaluate_deterministic_gi(surface, &[invalid, primary_backfacing]),
        [0.0; 3]
    );
}

#[test]
fn lighting_inputs_extract_stats_and_field_epochs_into_the_render_world() {
    let mut app = app_on_null_backend(RenderAppConfig::new(EXTENT));
    app.world_mut()
        .insert_resource(crate::LightingLightScene::default());
    let mut inputs = LightingSceneInputs::with_stats(SceneStats {
        lights: 12,
        punctual_max: 4,
        ..SceneStats::default()
    });
    app.world_mut().insert_resource(inputs);
    {
        use prism_render_architecture::lighting::{
            LightClass, LightClassification, LightEmission, LightGeometry, LightMobility,
            LightRecord, LightUnit,
        };
        let mut scene = app.world_mut().resource_mut::<crate::LightingLightScene>();
        for _ in 0..2 {
            scene
                .scene
                .insert(
                    LightRecord::new(
                        LightClassification {
                            class: LightClass::Directional,
                            mobility: LightMobility::Static,
                        },
                        LightEmission::new([1.0; 3], 1.0, LightUnit::Lux),
                        LightGeometry::Directional {
                            direction: [0.0, -1.0, 0.0],
                        },
                    ),
                    1,
                )
                .expect("insert directional");
        }
        for index in 0..10 {
            scene
                .scene
                .insert(
                    LightRecord::new(
                        LightClassification {
                            class: LightClass::Point,
                            mobility: LightMobility::Dynamic,
                        },
                        LightEmission::new([1.0; 3], 1.0, LightUnit::Candela),
                        LightGeometry::Point {
                            position: [index as f32, 1.0, 0.0],
                            radius: 0.1,
                            range: 10.0,
                        },
                    ),
                    1,
                )
                .expect("insert point");
        }
    }
    app.world_mut()
        .resource_mut::<LightingSceneInputs>()
        .mark_changed(1, ChangeFields::LIGHTING);

    app.update();

    let render = app.get_sub_app(RenderApp).expect("render sub-app");
    let extracted = render.world.resource::<ExtractedLightingInputs>();
    assert_eq!(extracted.stats.lights, 12);
    assert_eq!(extracted.directionals.len(), 2);
    assert_eq!(extracted.punctuals.len(), 10);
    assert_eq!(extracted.light_scene_sequence, 12);
    assert_eq!(extracted.light_scene_epoch, 12);
    assert_eq!(extracted.light_scene_error, None);
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
fn canonical_light_scene_insert_update_remove_are_extracted_with_freshness() {
    use prism_render_architecture::lighting::{
        LightClass, LightClassification, LightEmission, LightGeometry, LightMobility, LightPatch,
        LightRecord, LightUnit,
    };
    let mut app = app_on_null_backend(RenderAppConfig::new(EXTENT));
    app.world_mut()
        .insert_resource(crate::LightingLightScene::default());
    let (handle, _) = app
        .world_mut()
        .resource_mut::<crate::LightingLightScene>()
        .scene
        .insert(
            LightRecord::new(
                LightClassification {
                    class: LightClass::Point,
                    mobility: LightMobility::Dynamic,
                },
                LightEmission::new([1.0; 3], 2.0, LightUnit::Candela),
                LightGeometry::Point {
                    position: [0.0, 2.0, 0.0],
                    radius: 0.1,
                    range: 8.0,
                },
            ),
            1,
        )
        .expect("insert");
    app.update();
    {
        let render = app.get_sub_app(RenderApp).expect("render sub-app");
        let extracted = render.world.resource::<ExtractedLightingInputs>();
        assert_eq!(extracted.light_scene_sequence, 1);
        assert_eq!(extracted.punctuals.len(), 1);
        assert_eq!(extracted.punctuals[0].intensity, [2.0; 3]);
        assert_eq!(extracted.light_scene_error, None);
        let state = render.world.resource::<LightingPlanState>();
        assert_eq!(state.light_scene_sequence, extracted.light_scene_sequence);
        assert_eq!(state.light_scene_epoch, extracted.light_scene_epoch);
    }

    app.world_mut()
        .resource_mut::<crate::LightingLightScene>()
        .scene
        .update(
            handle,
            LightPatch::new().with_emission(LightEmission::new([0.5; 3], 4.0, LightUnit::Candela)),
            2,
        )
        .expect("update");
    app.update();
    {
        let render = app.get_sub_app(RenderApp).expect("render sub-app");
        let extracted = render.world.resource::<ExtractedLightingInputs>();
        assert_eq!(extracted.light_scene_sequence, 2);
        assert_eq!(extracted.punctuals[0].intensity, [2.0; 3]);
        assert_eq!(
            render
                .world
                .resource::<RenderFrameStats>()
                .lighting_cache_invalidations,
            1
        );
    }

    app.world_mut()
        .resource_mut::<crate::LightingLightScene>()
        .scene
        .remove(handle, 3)
        .expect("remove");
    app.update();
    let render = app.get_sub_app(RenderApp).expect("render sub-app");
    let extracted = render.world.resource::<ExtractedLightingInputs>();
    assert_eq!(extracted.light_scene_sequence, 3);
    assert_eq!(extracted.light_scene_epoch, 3);
    assert!(extracted.punctuals.is_empty());
    assert_eq!(
        render
            .world
            .resource::<RenderFrameStats>()
            .lighting_cache_invalidations,
        1
    );
}

#[test]
fn replacing_light_scene_lineage_with_matching_epochs_refreshes_derived_lights() {
    use prism_render_architecture::lighting::{
        LightClass, LightClassification, LightEmission, LightGeometry, LightMobility, LightRecord,
        LightScene, LightUnit,
    };
    let mut app = app_on_null_backend(RenderAppConfig::new(EXTENT));
    app.world_mut()
        .insert_resource(crate::LightingLightScene::default());
    let make_scene = |intensity| {
        let mut scene = LightScene::new();
        scene
            .insert(
                LightRecord::new(
                    LightClassification {
                        class: LightClass::Point,
                        mobility: LightMobility::Dynamic,
                    },
                    LightEmission::new([1.0; 3], intensity, LightUnit::Candela),
                    LightGeometry::Point {
                        position: [0.0, 2.0, 0.0],
                        radius: 0.1,
                        range: 8.0,
                    },
                ),
                1,
            )
            .expect("insert light");
        scene
    };
    let first_scene = make_scene(2.0);
    let first_identity = first_scene.identity();
    app.world_mut()
        .resource_mut::<crate::LightingLightScene>()
        .scene = first_scene;
    app.update();

    let replacement = make_scene(7.0);
    assert_ne!(replacement.identity(), first_identity);
    assert_eq!(replacement.sequence(), 1);
    assert_eq!(replacement.scene_epoch(), 1);
    app.world_mut()
        .resource_mut::<crate::LightingLightScene>()
        .scene = replacement;
    app.update();

    let render = app.get_sub_app(RenderApp).expect("render sub-app");
    let extracted = render.world.resource::<ExtractedLightingInputs>();
    assert_eq!(extracted.light_scene_sequence, 1);
    assert_eq!(extracted.light_scene_epoch, 1);
    assert_ne!(extracted.light_scene_identity, first_identity);
    assert_eq!(extracted.punctuals[0].intensity, [7.0; 3]);
    let state = render.world.resource::<LightingPlanState>();
    assert_eq!(state.light_scene_identity, extracted.light_scene_identity);
    assert_eq!(
        render
            .world
            .resource::<RenderFrameStats>()
            .lighting_cache_invalidations,
        1
    );
}

#[test]
fn mixed_canonical_scene_preserves_supported_lights_and_submits_with_emitter_rejections() {
    use prism_render_architecture::lighting::{
        LightClass, LightClassification, LightEmission, LightFlags, LightGeometry, LightMobility,
        LightRecord, LightUnit, LightVisibility,
    };
    let mut app = app_on_null_backend(RenderAppConfig::new(EXTENT));
    app.world_mut()
        .insert_resource(crate::LightingLightScene::default());
    let (
        directional_handle,
        point_handle,
        spot_handle,
        rect_handle,
        disk_handle,
        capsule_handle,
        sky_handle,
        emissive_handle,
        unit_handle,
        layer_handle,
        flags_handle,
    ) = {
        let scene = app.world_mut().resource_mut::<crate::LightingLightScene>();
        let mut frame = 1_u64;
        let mut insert = |record, description| {
            let handle = scene
                .scene
                .insert(record, frame)
                .unwrap_or_else(|error| panic!("{description}: {error:?}"))
                .0;
            frame += 1;
            handle
        };
        let directional_handle = insert(
            LightRecord::new(
                LightClassification {
                    class: LightClass::Directional,
                    mobility: LightMobility::Static,
                },
                LightEmission::new([1.0; 3], 1.0, LightUnit::Lux),
                LightGeometry::Directional {
                    direction: [0.0, -1.0, 0.0],
                },
            ),
            "supported directional",
        );
        let point_handle = insert(
            LightRecord::new(
                LightClassification {
                    class: LightClass::Point,
                    mobility: LightMobility::Dynamic,
                },
                LightEmission::new([1.0; 3], 2.0, LightUnit::Candela),
                LightGeometry::Point {
                    position: [1.0, 2.0, 3.0],
                    radius: 0.1,
                    range: 10.0,
                },
            ),
            "supported point",
        );
        let spot_handle = insert(
            LightRecord::new(
                LightClassification {
                    class: LightClass::Spot,
                    mobility: LightMobility::Dynamic,
                },
                LightEmission::new([1.0; 3], 3.0, LightUnit::Candela),
                LightGeometry::Spot {
                    position: [0.0, 3.0, 0.0],
                    direction: [0.0, -1.0, 0.0],
                    inner_angle: 0.2,
                    outer_angle: 0.6,
                    radius: 0.1,
                    range: 12.0,
                },
            ),
            "supported spot",
        );
        let rect_handle = insert(
            LightRecord::new(
                LightClassification {
                    class: LightClass::Rect,
                    mobility: LightMobility::Static,
                },
                LightEmission::new([1.0; 3], 1.0, LightUnit::Nits),
                LightGeometry::Rect {
                    position: [0.0; 3],
                    direction: [0.0, -1.0, 0.0],
                    right: [1.0, 0.0, 0.0],
                    up: [0.0, 0.0, 1.0],
                    width: 1.0,
                    height: 1.0,
                },
            ),
            "unsupported rect",
        );
        let disk_handle = insert(
            LightRecord::new(
                LightClassification {
                    class: LightClass::Disk,
                    mobility: LightMobility::Static,
                },
                LightEmission::new([1.0; 3], 1.0, LightUnit::Nits),
                LightGeometry::Disk {
                    position: [0.0; 3],
                    direction: [0.0, -1.0, 0.0],
                    radius: 0.5,
                },
            ),
            "unsupported disk",
        );
        let capsule_handle = insert(
            LightRecord::new(
                LightClassification {
                    class: LightClass::Capsule,
                    mobility: LightMobility::Static,
                },
                LightEmission::new([1.0; 3], 1.0, LightUnit::Nits),
                LightGeometry::Capsule {
                    a: [-0.5, 0.0, 0.0],
                    b: [0.5, 0.0, 0.0],
                    radius: 0.1,
                },
            ),
            "unsupported capsule",
        );
        let sky_handle = insert(
            LightRecord::new(
                LightClassification {
                    class: LightClass::Sky,
                    mobility: LightMobility::Static,
                },
                LightEmission::new([1.0; 3], 1.0, LightUnit::Radiance),
                LightGeometry::Sky {
                    rotation: [0.0, 0.0, 0.0, 1.0],
                },
            ),
            "unsupported sky",
        );
        let emissive_handle = insert(
            LightRecord::new(
                LightClassification {
                    class: LightClass::Emissive,
                    mobility: LightMobility::Static,
                },
                LightEmission::new([1.0; 3], 1.0, LightUnit::Radiance),
                LightGeometry::Emissive { emitter_id: 7 },
            ),
            "unsupported emissive",
        );
        let unit_handle = insert(
            LightRecord::new(
                LightClassification {
                    class: LightClass::Point,
                    mobility: LightMobility::Dynamic,
                },
                LightEmission::new([1.0; 3], 1.0, LightUnit::Unitless),
                LightGeometry::Point {
                    position: [0.0; 3],
                    radius: 0.0,
                    range: 4.0,
                },
            ),
            "unsupported direct-light unit",
        );
        let mut layered = LightRecord::new(
            LightClassification {
                class: LightClass::Point,
                mobility: LightMobility::Dynamic,
            },
            LightEmission::new([1.0; 3], 1.0, LightUnit::Candela),
            LightGeometry::Point {
                position: [0.0; 3],
                radius: 0.0,
                range: 4.0,
            },
        );
        layered.visibility = LightVisibility {
            enabled: true,
            layer_mask: 1,
        };
        let layer_handle = insert(layered, "unsupported layer mask");
        let mut flagged = LightRecord::new(
            LightClassification {
                class: LightClass::Point,
                mobility: LightMobility::Dynamic,
            },
            LightEmission::new([1.0; 3], 1.0, LightUnit::Candela),
            LightGeometry::Point {
                position: [0.0; 3],
                radius: 0.0,
                range: 4.0,
            },
        );
        flagged.flags = LightFlags::CAST_SHADOW;
        let flags_handle = insert(flagged, "unsupported renderer flags");
        (
            directional_handle,
            point_handle,
            spot_handle,
            rect_handle,
            disk_handle,
            capsule_handle,
            sky_handle,
            emissive_handle,
            unit_handle,
            layer_handle,
            flags_handle,
        )
    };
    app.update();
    let render = app.get_sub_app(RenderApp).expect("render sub-app");
    let extracted = render.world.resource::<ExtractedLightingInputs>();
    assert_eq!(extracted.light_scene_error, None);
    assert_eq!(extracted.directional_handles, vec![directional_handle]);
    assert_eq!(extracted.punctual_handles, vec![point_handle, spot_handle]);
    assert_eq!(
        extracted.area_handles,
        vec![rect_handle, disk_handle, capsule_handle]
    );
    assert_eq!(extracted.directionals.len(), 1);
    assert_eq!(extracted.punctuals.len(), 2);
    assert_eq!(extracted.areas.len(), 3);
    assert_eq!(
        extracted.light_scene_rejections,
        vec![
            crate::LightingSceneExtractError::UnsupportedClass {
                handle: sky_handle,
                class: LightClass::Sky,
            },
            crate::LightingSceneExtractError::UnsupportedClass {
                handle: emissive_handle,
                class: LightClass::Emissive,
            },
            crate::LightingSceneExtractError::UnsupportedUnit {
                handle: unit_handle,
                unit: LightUnit::Unitless,
            },
            crate::LightingSceneExtractError::UnsupportedLayerMask {
                handle: layer_handle,
                mask: 1,
            },
            crate::LightingSceneExtractError::UnsupportedFlags {
                handle: flags_handle,
                bits: LightFlags::CAST_SHADOW.bits(),
            },
        ]
    );
    let stats = render.world.resource::<RenderFrameStats>();
    assert_eq!(stats.frames_submitted, 1);
    assert_eq!(stats.frames_failed, 0);
    assert_eq!(stats.lighting_input_error, None);
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
        prefiltered_environment: false,
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
    let actual = evaluate_deterministic_direct(surface, &directionals, &punctuals, &[]);
    for (actual, irradiance_times_albedo) in actual.into_iter().zip([1.5, 1.5, 11.0]) {
        let expected = irradiance_times_albedo / core::f32::consts::PI;
        assert!((actual - expected).abs() <= 1.0e-6);
    }
}

#[test]
fn deterministic_direct_lambert_normalizes_ambient_directional_and_point_independently() {
    let pi = core::f32::consts::PI;
    let surface = LightingSurfaceInput {
        albedo: [1.0, 0.5, 0.25],
        ..LightingSurfaceInput::default()
    };
    // Each fixture independently supplies PI irradiance at normal incidence.
    // Expectations do not call the CPU oracle or duplicate the shader formula.
    let ambient = evaluate_deterministic_direct(
        LightingSurfaceInput {
            ambient: [pi; 3],
            ..surface
        },
        &[],
        &[],
        &[],
    );
    let directional = evaluate_deterministic_direct(
        surface,
        &[LightingDirectionalInput {
            illuminance: [pi; 3],
            ..Default::default()
        }],
        &[],
        &[],
    );
    let point = LightingPunctualInput {
        position: [0.0, 2.0, 0.0],
        intensity: [4.0 * pi; 3],
        ..Default::default()
    };
    let punctual = evaluate_deterministic_direct(surface, &[], &[point], &[]);
    for result in [ambient, directional, punctual] {
        for (actual, expected) in result.into_iter().zip(surface.albedo) {
            assert!((actual - expected).abs() <= 1.0e-6);
        }
    }
    let twice_distance = evaluate_deterministic_direct(
        surface,
        &[],
        &[LightingPunctualInput {
            position: [0.0, 4.0, 0.0],
            ..point
        }],
        &[],
    );
    for (actual, albedo) in twice_distance.into_iter().zip(surface.albedo) {
        assert!((actual - albedo * 0.25).abs() <= 1.0e-6);
    }
}

#[test]
fn deterministic_direct_rect_lambert_matches_analytic_quadrature() {
    // All four rectangle quadrature points have r²=4+1/6. The receiver and
    // emitter cosines are each 2/r. With area=1 and emitted radiance=PI,
    // outgoing radiance is albedo * 4/r⁴ after the Lambert normalization.
    let surface = LightingSurfaceInput {
        albedo: [1.0, 0.5, 0.25],
        ..Default::default()
    };
    let light = LightingAreaInput {
        origin: [0.0, 2.0, 0.0, 0.0],
        radiance: [core::f32::consts::PI; 4],
        shape: [0.5, 0.5, 0.0, 0.0],
        ..Default::default()
    };
    let actual = evaluate_deterministic_direct(surface, &[], &[], &[light]);
    let geometry = 4.0 / (4.0_f32 + 1.0 / 6.0).powi(2);
    for (actual, albedo) in actual.into_iter().zip(surface.albedo) {
        assert!((actual - albedo * geometry).abs() <= 1.0e-6);
    }
}

#[test]
fn deterministic_direct_cpu_oracle_matches_area_estimators_for_rect_disk_and_capsule() {
    let surface = LightingSurfaceInput {
        position: [0.0, 0.0, 0.0],
        normal: [0.0, 1.0, 0.0],
        albedo: [0.8, 0.6, 0.4],
        ambient: [0.0; 3],
    };
    let areas = [
        LightingAreaInput {
            origin: [0.0, 3.0, 0.0, 0.0],
            vector_a: [0.0, -1.0, 0.0, 0.0],
            vector_b: [1.0, 0.0, 0.0, 0.0],
            vector_c: [0.0, 0.0, 1.0, 0.0],
            radiance: [2.0, 1.0, 0.5, 0.0],
            shape: [0.75, 0.5, 0.0, 0.0],
            shape_kind: [0, 1, 0, 0],
        },
        LightingAreaInput {
            origin: [2.0, 3.0, 0.0, 0.0],
            vector_a: [0.0, -1.0, 0.0, 0.0],
            vector_b: [0.0; 4],
            vector_c: [0.0; 4],
            radiance: [1.0, 2.0, 3.0, 0.0],
            shape: [0.5, 0.0, 0.0, 0.0],
            shape_kind: [1, 1, 0, 0],
        },
        LightingAreaInput {
            origin: [-2.0, 1.5, 0.0, 0.0],
            vector_a: [-2.0, 3.5, 0.0, 0.0],
            vector_b: [0.0; 4],
            vector_c: [0.0; 4],
            radiance: [0.75, 1.5, 2.25, 0.0],
            shape: [0.4, 0.0, 0.0, 0.0],
            shape_kind: [2, 1, 0, 0],
        },
    ];
    let all = evaluate_deterministic_direct(surface, &[], &[], &areas);
    let mut sum = [0.0; 3];
    for (index, area) in areas.iter().enumerate() {
        let contribution =
            evaluate_deterministic_direct(surface, &[], &[], core::slice::from_ref(area));
        assert!(
            contribution.iter().any(|value| *value > 0.0),
            "area {index} must contribute a positive direct-light sample"
        );
        for channel in 0..3 {
            sum[channel] += contribution[channel];
        }
    }
    for channel in 0..3 {
        assert!((all[channel] - sum[channel]).abs() < 1.0e-6);
    }
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
/// back to prove it landed), so the zero HDR result proves the cleared GBuffer
/// was rejected by the composite — a freshly created wgpu texture is already
/// zeroed, so an unfilled target would pass vacuously.
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
    app.world_mut()
        .insert_resource(crate::LightingLightScene::default());
    let surface = LightingSurfaceInput {
        position: [0.0, 0.0, 0.0],
        normal: [0.0, 1.0, 0.0],
        albedo: [0.5, 0.25, 1.0],
        ambient: [1.0, 2.0, 3.0],
    };
    let directionals = vec![LightingDirectionalInput {
        direction_to_light: [0.0, 1.0, 0.0],
        visibility: 1.0,
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
    let expected_gi = surfaces
        .iter()
        .copied()
        .zip(&gi_candidates_per_surface)
        .map(|(sample, candidates)| evaluate_deterministic_gi(sample, candidates))
        .collect::<Vec<_>>();
    {
        use prism_render_architecture::lighting::{
            LightClass, LightClassification, LightEmission, LightGeometry, LightMobility,
            LightRecord, LightUnit,
        };
        {
            let inputs = app.world_mut().resource_mut::<LightingSceneInputs>();
            inputs.surfaces = surfaces.clone();
            inputs.gi_candidates = gi_candidates_per_surface;
        }
        let mut lights = app.world_mut().resource_mut::<crate::LightingLightScene>();
        lights
            .scene
            .insert(
                LightRecord::new(
                    LightClassification {
                        class: LightClass::Directional,
                        mobility: LightMobility::Static,
                    },
                    LightEmission::new([2.0, 4.0, 8.0], 1.0, LightUnit::Lux),
                    LightGeometry::Directional {
                        direction: [0.0, -1.0, 0.0],
                    },
                ),
                1,
            )
            .expect("insert directional");
        lights
            .scene
            .insert(
                LightRecord::new(
                    LightClassification {
                        class: LightClass::Point,
                        mobility: LightMobility::Dynamic,
                    },
                    LightEmission::new([4.0, 8.0, 16.0], 1.0, LightUnit::Candela),
                    LightGeometry::Point {
                        position: [0.0, 2.0, 0.0],
                        radius: 0.0,
                        range: 0.0,
                    },
                ),
                1,
            )
            .expect("insert point");
    }

    // Pre-fill the offscreen target with opaque white, then prove the upload
    // landed before any frame is rendered.
    // Each RGBA16F channel is little-endian half-float 1.0 (0x3c00).
    let white = [0x00_u8, 0x3c].repeat((SIZE * SIZE * 4) as usize);
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
                bytes_per_row: Some(SIZE * 8),
                rows_per_image: Some(SIZE),
            },
        );
        assert!(runtime.runtime().wait_idle());
        let prefill = queue
            .read_texture_rgba16f(target.texture(), SIZE, SIZE)
            .expect("pre-fill texture readback must complete");
        assert_eq!(prefill.len(), (SIZE * SIZE) as usize);
        assert!(
            prefill.iter().all(|pixel| *pixel == [1.0; 4]),
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
        .read_texture_rgba16f(target.texture(), SIZE, SIZE)
        .expect("frame target texture readback must complete");
    assert_eq!(pixels.len(), (SIZE * SIZE) as usize);
    let first_nonzero = pixels
        .iter()
        .enumerate()
        .find(|(_, pixel)| **pixel != [0.0; 4]);
    assert!(
        pixels.iter().all(|pixel| *pixel == [0.0; 4]),
        "the composite overwrote the white target with transparent black for an uncovered GBuffer; first nonzero={first_nonzero:?}, lighting_passes={}",
        render.world.resource::<RenderFrameStats>().lighting_gpu_passes_recorded
    );

    let lighting = render
        .world
        .resource::<LightingGpuState<prism_render_driver_wgpu::WgpuInstance>>();
    let extracted = render.world.resource::<ExtractedLightingInputs>();
    assert_eq!(
        extracted.candidate_availability(runtime.runtime().device().capabilities()),
        GiCandidateAvailability::HostProvided
    );
    let plan = render.world.resource::<LightingPlanState>();
    assert_eq!(
        plan.compiled_frame,
        render.world.resource::<RenderRecord>().frame_index
    );
    assert_eq!(plan.plan.candidate_source, GiCandidateSource::HostProvided);
    assert!(plan.plan.recipes.contains(&RecipeId::WorldRestir));
    let gi_values = runtime
        .runtime()
        .queue()
        .read_buffer(
            lighting.gi_output_buffer(),
            crate::lighting_gpu::OUTPUT_BYTES,
        )
        .expect("GI output buffer readback must complete")
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("f32-sized GI output")))
        .collect::<Vec<_>>();
    assert_eq!(
        gi_values.len(),
        crate::lighting_gpu::MAX_LIGHTING_SURFACES * 4
    );
    for surface_index in 0..surfaces.len() {
        let base = surface_index * 4;
        assert_eq!(gi_values[base + 3], 1.0);
        for channel in 0..3 {
            assert!(
                (gi_values[base + channel] - expected_gi[surface_index][channel]).abs() < 1.0e-4,
                "GPU GI surface {surface_index} channel {channel}: {} vs {}",
                gi_values[base + channel],
                expected_gi[surface_index][channel]
            );
        }
    }
    for tail in (surfaces.len() * 4)..gi_values.len() {
        assert_eq!(gi_values[tail], 0.0, "GI tail is deterministically clear");
    }
    // Direct light is evaluated from per-pixel GBuffer data by the composite.
    // Use id zero here only to select GI row zero; direct lighting is
    // independent of that identity and uses position, normal, and albedo.
    let gbuffer_surface =
        write_lighting_test_gbuffer_pixel(runtime.runtime().queue(), &target, 0, 0, surfaces[0], 0);
    let mut direct_surface = gbuffer_surface;
    direct_surface.ambient = [0.0; 3];
    let expected_direct =
        evaluate_deterministic_direct(direct_surface, &directionals, &punctuals, &[]);
    dispatch_lighting_test_composite(lighting, &plan.plan, &target, runtime.runtime());
    let hdr_pixels = runtime
        .runtime()
        .queue()
        .read_texture_rgba16f(target.texture(), SIZE, SIZE)
        .expect("direct-light HDR readback must complete");
    assert_eq!(hdr_pixels.len(), (SIZE * SIZE) as usize);
    for channel in 0..3 {
        let expected = expected_direct[channel] + expected_gi[0][channel];
        assert!(
            (hdr_pixels[0][channel] - expected).abs() <= 2.0e-3,
            "per-pixel HDR channel {channel}: GPU {} != CPU {}",
            hdr_pixels[0][channel],
            expected
        );
    }
    assert_eq!(hdr_pixels[0][3], 1.0);
    assert!(
        hdr_pixels[1..].iter().all(|pixel| *pixel == [0.0; 4]),
        "uncovered GBuffer pixels remain transparent black in HDR"
    );
    assert_eq!(
        render
            .world
            .resource::<RenderFrameStats>()
            .lighting_gpu_passes_recorded,
        2,
        "GI and composite are the only lighting compute passes"
    );
}

#[test]
fn physical_gpu_area_direct_light_matches_cpu_oracle_for_rect_disk_and_capsule() {
    use prism_render_architecture::lighting::LightSceneHandle;
    use prism_render_driver::DeviceType;
    use prism_render_driver_wgpu::WgpuInstance;

    let extent = Extent3d::new_2d(1, 1);
    let mut runtime = match RenderRuntime::headless(WgpuInstance::new(), extent) {
        Ok(runtime) => runtime,
        Err(error) => {
            assert!(
                !require_gpu(),
                "PRISM_REQUIRE_GPU=1 but physical GPU runtime creation failed: {error}"
            );
            eprintln!("SKIPPED area direct parity: {error}");
            return;
        }
    };
    if !matches!(
        runtime.adapter_info().device_type,
        DeviceType::IntegratedGpu | DeviceType::DiscreteGpu
    ) {
        assert!(
            !require_gpu(),
            "PRISM_REQUIRE_GPU=1 requires a physical GPU, found {:?} ({})",
            runtime.adapter_info().device_type,
            runtime.adapter_info().name
        );
        eprintln!(
            "SKIPPED area direct parity: adapter is {:?} ({})",
            runtime.adapter_info().device_type,
            runtime.adapter_info().name
        );
        return;
    }

    let surface = LightingSurfaceInput {
        position: [0.0, 0.0, 0.0],
        normal: [0.0, 1.0, 0.0],
        albedo: [0.8, 0.6, 0.4],
        ambient: [0.0; 3],
    };
    let areas = vec![
        LightingAreaInput {
            origin: [0.0, 3.0, 0.0, 0.0],
            vector_a: [0.0, -1.0, 0.0, 0.0],
            vector_b: [1.0, 0.0, 0.0, 0.0],
            vector_c: [0.0, 0.0, 1.0, 0.0],
            radiance: [2.0, 1.0, 0.5, 0.0],
            shape: [0.75, 0.5, 0.0, 0.0],
            shape_kind: [0, 1, 0, 0],
        },
        LightingAreaInput {
            origin: [2.0, 3.0, 0.0, 0.0],
            vector_a: [0.0, -1.0, 0.0, 0.0],
            vector_b: [0.0; 4],
            vector_c: [0.0; 4],
            radiance: [1.0, 2.0, 3.0, 0.0],
            shape: [0.5, 0.0, 0.0, 0.0],
            shape_kind: [1, 1, 0, 0],
        },
        LightingAreaInput {
            origin: [-2.0, 1.5, 0.0, 0.0],
            vector_a: [-2.0, 3.5, 0.0, 0.0],
            vector_b: [0.0; 4],
            vector_c: [0.0; 4],
            radiance: [0.75, 1.5, 2.25, 0.0],
            shape: [0.4, 0.0, 0.0, 0.0],
            shape_kind: [2, 1, 0, 0],
        },
    ];
    let area_handles = [
        LightSceneHandle::new(10, 1),
        LightSceneHandle::new(11, 1),
        LightSceneHandle::new(12, 1),
    ];
    for (index, area) in areas.iter().enumerate() {
        let contribution =
            evaluate_deterministic_direct(surface, &[], &[], std::slice::from_ref(area));
        assert!(
            contribution.iter().any(|value| *value > 0.0),
            "area {index} must contribute a positive direct-light sample"
        );
    }

    let mut inputs = ExtractedLightingInputs::default();
    // Direct is evaluated from the bound per-pixel GBuffer. The host GI
    // surface list is deliberately empty so this guards direct-only admission
    // against an accidental dependency on GI/oracle staging data.
    inputs.areas = areas.clone();
    inputs.area_handles = area_handles.to_vec();
    let plan = LightingPlan {
        tier: Tier::Balanced,
        candidate_source: GiCandidateSource::Disabled,
        recipes: vec![RecipeId::SpecializedDeterministic],
        partition: Default::default(),
        budget: FrameResourceBudget::default(),
        rejected: Vec::new(),
    };
    let lighting = LightingGpuState::new(&mut runtime).expect("lighting GPU bundle");
    lighting
        .prepare(&plan, &inputs, 0, &runtime)
        .expect("area light tables prepare");
    let pass = lighting
        .pass(&plan)
        .expect("direct area pass must not require host GI surfaces");
    assert!(pass.run_direct);
    assert!(!pass.run_gi);
    assert_eq!(pass.candidate_surface_count, 0);
    let target = RenderTarget::new(runtime.device(), extent);
    let gbuffer_surface =
        write_lighting_test_gbuffer_pixel(runtime.queue(), &target, 0, 0, surface, 0xf123_4567);
    let expected = evaluate_deterministic_direct(gbuffer_surface, &[], &[], &areas);
    dispatch_lighting_test_composite(&lighting, &plan, &target, &runtime);
    let hdr = runtime
        .queue()
        .read_texture_rgba16f(target.texture(), 1, 1)
        .expect("area-light HDR readback must complete");
    assert_eq!(hdr.len(), 1);
    for channel in 0..3 {
        assert!(
            (hdr[0][channel] - expected[channel]).abs() <= 2.0e-3,
            "area composite HDR channel {channel}: GPU {} != CPU {}",
            hdr[0][channel],
            expected[channel]
        );
    }
    assert_eq!(hdr[0][3], 1.0);
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

/// `mesh_draw` records a full draw into the real graph recorder: bind the
/// pipeline and group (including every dynamic offset), bind geometry, then
/// issue one indexed draw.
#[test]
fn mesh_draw_records_into_a_graph_pass() {
    use crate::draw::{BindGroupBindingId, RenderBindGroupBinding};
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

    let bind_group = prism_render_driver::BindGroupId::from_parts(6, 1);
    let mut functions = DrawFunctions::new();
    let draw = functions.register_mesh_with_bind_groups(vec![RenderBindGroupBinding {
        identity: BindGroupBindingId(5),
        index: 2,
        bind_group,
        dynamic_offsets: vec![128, 640],
        buffers: Vec::new(),
        sampled_textures: Vec::new(),
        external_owners: Vec::new(),
    }]);
    assert_eq!(draw, DrawFunctionId(0), "the first registration is id 0");

    let mut queues = RenderQueues::new();
    queues.push(
        RenderPhase::Opaque,
        PhaseItem {
            pipeline,
            draw,
            geometry: GenerationalHandle::new(1, 1),
            vertices: vec![vertices],
            vertex_sizes: vec![64],
            indices: Some((indices, IndexFormat::Uint16)),
            index_size: Some(64),
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
            Some(RenderCommand::SetBindGroup {
                index: 2,
                bind_group: recorded_bind_group,
                dynamic_offsets,
            }) if *recorded_bind_group == bind_group && dynamic_offsets == &[128, 640]
        ),
        "the render recorder preserves every bind-group dynamic offset"
    );
    assert!(
        matches!(
            commands.get(2),
            Some(RenderCommand::SetVertexBuffer { slot: 0, size: Some(size), .. }) if size.get() == 64
        ),
        "the vertex buffer is bound at slot 0"
    );
    assert!(
        matches!(commands.get(3), Some(RenderCommand::SetIndexBuffer(binding)) if binding.size.is_some_and(|size| size.get() == 64)),
        "the index buffer is bound"
    );
    assert!(
        matches!(commands.get(4), Some(RenderCommand::DrawIndexed { .. })),
        "one indexed draw is issued"
    );
}

/// A forward-lit mesh draw carries one explicit binding contract: the pipeline
/// is recorded first, followed by the declared render bind group, then geometry
/// and the draw. The same contract is visible from the queued item.
#[test]
fn forward_mesh_draw_records_bind_group_and_exposes_identity() {
    use crate::draw::{
        BindGroupBindingId, MeshRecorder, RenderBindGroupBinding, RenderBindGroupBuffer,
        RenderBindGroupBufferAccess,
    };
    use prism_render_driver::{BindGroupId, IndexFormat, RenderPipelineId};

    #[derive(Debug, PartialEq, Eq)]
    enum RecordedDrawCommand {
        Pipeline,
        BindGroup {
            index: u32,
            bind_group: BindGroupId,
            dynamic_offsets: Vec<u32>,
        },
        Vertex,
        Index,
        Draw,
        DrawIndexed,
        DrawIndirect,
        MultiDrawIndirect,
    }

    #[derive(Default)]
    struct Log(Vec<RecordedDrawCommand>);

    impl MeshRecorder for Log {
        fn set_pipeline(&mut self, _: RenderPipelineId) {
            self.0.push(RecordedDrawCommand::Pipeline);
        }

        fn set_render_bind_group(
            &mut self,
            index: u32,
            bind_group: BindGroupId,
            dynamic_offsets: Vec<u32>,
        ) {
            self.0.push(RecordedDrawCommand::BindGroup {
                index,
                bind_group,
                dynamic_offsets,
            });
        }

        fn set_vertex_buffer(
            &mut self,
            _: u32,
            _: BufferId,
            _: u64,
            _: Option<core::num::NonZeroU64>,
        ) {
            self.0.push(RecordedDrawCommand::Vertex);
        }

        fn set_index_buffer(
            &mut self,
            _: BufferId,
            _: IndexFormat,
            _: u64,
            _: Option<core::num::NonZeroU64>,
        ) {
            self.0.push(RecordedDrawCommand::Index);
        }

        fn draw(&mut self, _: core::ops::Range<u32>, _: core::ops::Range<u32>) {
            self.0.push(RecordedDrawCommand::Draw);
        }

        fn draw_indexed(&mut self, _: core::ops::Range<u32>, _: i32, _: core::ops::Range<u32>) {
            self.0.push(RecordedDrawCommand::DrawIndexed);
        }

        fn draw_indexed_indirect(&mut self, _: prism_render_graph::BufferHandle, _: u64) {
            self.0.push(RecordedDrawCommand::DrawIndirect);
        }

        fn multi_draw_indirect(
            &mut self,
            _: prism_render_graph::BufferHandle,
            _: u64,
            _: u32,
            _: u32,
        ) {
            self.0.push(RecordedDrawCommand::MultiDrawIndirect);
        }
    }

    let pipeline = RenderPipelineId::from_parts(7, 1);
    let vertices = BufferId::from_parts(8, 1);
    let material_buffer = BufferId::from_parts(9, 1);
    let bind_group = BindGroupId::from_parts(10, 1);
    let binding = RenderBindGroupBinding {
        identity: BindGroupBindingId(42),
        index: 3,
        bind_group,
        dynamic_offsets: vec![256, 768],
        buffers: vec![RenderBindGroupBuffer {
            buffer: material_buffer,
            access: RenderBindGroupBufferAccess::Storage,
        }],
        sampled_textures: Vec::new(),
        external_owners: Vec::new(),
    };

    let mut functions = DrawFunctions::new();
    let draw = functions.register_mesh_with_bind_groups(vec![binding.clone()]);
    let item = PhaseItem {
        pipeline,
        draw,
        geometry: GenerationalHandle::new(1, 1),
        vertices: vec![vertices],
        vertex_sizes: vec![64],
        indices: None,
        index_size: None,
        first: 0,
        count: 3,
        base_vertex: 0,
        indirect: None,
        lod_level: 0,
        material: GenerationalHandle::new(2, 1),
        scene: GenerationalHandle::new(3, 1),
        order: 0,
    };

    assert_eq!(
        item.bind_group_identity(&functions),
        Some(BindGroupBindingId(42))
    );
    assert_eq!(item.bind_groups(&functions), &[binding]);

    let mut log = Log::default();
    functions.call(draw, &item, &mut log);
    assert_eq!(
        log.0,
        [
            RecordedDrawCommand::Pipeline,
            RecordedDrawCommand::BindGroup {
                index: 3,
                bind_group,
                dynamic_offsets: vec![256, 768],
            },
            RecordedDrawCommand::Vertex,
            RecordedDrawCommand::Draw,
        ]
    );
}

/// Per-draw groups are resolved after the static contract, in contract order,
/// and remain paired with their phase item when queue sorting reorders draws.
#[test]
fn dynamic_bind_groups_follow_items_and_record_after_static_groups() {
    use crate::draw::{
        BindGroupBindingId, MeshRecorder, RenderBindGroupBinding, RenderBindGroupBindingContract,
        RenderBindGroupBuffer, RenderBindGroupBufferAccess,
    };
    use prism_render_driver::{BindGroupId, IndexFormat, RenderPipelineId};

    #[derive(Default)]
    struct Log(Vec<(u32, BindGroupId, Vec<u32>)>);

    impl MeshRecorder for Log {
        fn set_pipeline(&mut self, _: RenderPipelineId) {}

        fn set_render_bind_group(
            &mut self,
            index: u32,
            bind_group: BindGroupId,
            dynamic_offsets: Vec<u32>,
        ) {
            self.0.push((index, bind_group, dynamic_offsets));
        }

        fn set_vertex_buffer(
            &mut self,
            _: u32,
            _: BufferId,
            _: u64,
            _: Option<core::num::NonZeroU64>,
        ) {
        }
        fn set_index_buffer(
            &mut self,
            _: BufferId,
            _: IndexFormat,
            _: u64,
            _: Option<core::num::NonZeroU64>,
        ) {
        }
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

    let static_group = RenderBindGroupBinding {
        identity: BindGroupBindingId(1),
        index: 0,
        bind_group: BindGroupId::from_parts(10, 1),
        dynamic_offsets: vec![64, 128],
        buffers: vec![],
        sampled_textures: Vec::new(),
        external_owners: Vec::new(),
    };
    let material_contract = RenderBindGroupBindingContract {
        identity: BindGroupBindingId(2),
        index: 2,
    };
    let view_contract = RenderBindGroupBindingContract {
        identity: BindGroupBindingId(3),
        index: 1,
    };
    let material_a = RenderBindGroupBinding {
        identity: material_contract.identity,
        index: material_contract.index,
        bind_group: BindGroupId::from_parts(20, 1),
        dynamic_offsets: vec![256, 512],
        buffers: vec![RenderBindGroupBuffer {
            buffer: BufferId::from_parts(30, 1),
            access: RenderBindGroupBufferAccess::Storage,
        }],
        sampled_textures: Vec::new(),
        external_owners: Vec::new(),
    };
    let view_a = RenderBindGroupBinding {
        identity: view_contract.identity,
        index: view_contract.index,
        bind_group: BindGroupId::from_parts(21, 1),
        dynamic_offsets: vec![1024],
        buffers: vec![RenderBindGroupBuffer {
            buffer: BufferId::from_parts(31, 1),
            access: RenderBindGroupBufferAccess::Uniform,
        }],
        sampled_textures: Vec::new(),
        external_owners: Vec::new(),
    };
    let mut functions = DrawFunctions::new();
    let draw = functions.register_with_bind_groups_and_dynamic_contract(
        |item, ctx| {
            ctx.set_pipeline(item.pipeline);
            ctx.set_vertex_buffer(0, item.vertices[0], 0, None);
            ctx.draw(0..item.count, 0..1);
        },
        vec![static_group.clone()],
        vec![material_contract, view_contract],
    );

    let item = |order| PhaseItem {
        pipeline: RenderPipelineId::from_parts(40, 1),
        draw,
        geometry: GenerationalHandle::new(1, 1),
        vertices: vec![BufferId::from_parts(41, 1)],
        vertex_sizes: vec![64],
        indices: None,
        index_size: None,
        first: 0,
        count: 3,
        base_vertex: 0,
        indirect: None,
        lod_level: 0,
        material: GenerationalHandle::new(2, 1),
        scene: GenerationalHandle::new(3, 1),
        order,
    };

    let mut queues = RenderQueues::new();
    // Supply the dynamic groups in the opposite order. The contract order
    // must still be slot 2 followed by slot 1 after the static slot 0.
    queues.push_with_bind_groups(
        RenderPhase::Opaque,
        item(2),
        vec![view_a.clone(), material_a.clone()],
    );
    queues.push_with_bind_groups(
        RenderPhase::Opaque,
        item(1),
        vec![material_a.clone(), view_a.clone()],
    );
    queues.sort();

    let mut log = Log::default();
    queues.record(RenderPhase::Opaque, &mut log, &functions);
    assert_eq!(
        log.0,
        vec![
            (0, static_group.bind_group, vec![64, 128]),
            (2, material_a.bind_group, vec![256, 512]),
            (1, view_a.bind_group, vec![1024]),
            (0, static_group.bind_group, vec![64, 128]),
            (2, material_a.bind_group, vec![256, 512]),
            (1, view_a.bind_group, vec![1024]),
        ]
    );
    assert_eq!(
        queues.dynamic_bind_groups(RenderPhase::Opaque, 0),
        &[material_a, view_a]
    );
}

#[test]
#[should_panic(expected = "missing required dynamic render bind groups")]
fn dynamic_bind_group_contract_rejects_missing_binding() {
    use crate::draw::{BindGroupBindingId, MeshRecorder, RenderBindGroupBindingContract};
    use prism_render_driver::{BindGroupId, IndexFormat, RenderPipelineId};
    use std::ops::Range;

    #[derive(Default)]
    struct Noop;
    impl MeshRecorder for Noop {
        fn set_pipeline(&mut self, _: RenderPipelineId) {}
        fn set_render_bind_group(&mut self, _: u32, _: BindGroupId, _: Vec<u32>) {}
        fn set_vertex_buffer(
            &mut self,
            _: u32,
            _: BufferId,
            _: u64,
            _: Option<core::num::NonZeroU64>,
        ) {
        }
        fn set_index_buffer(
            &mut self,
            _: BufferId,
            _: IndexFormat,
            _: u64,
            _: Option<core::num::NonZeroU64>,
        ) {
        }
        fn draw(&mut self, _: Range<u32>, _: Range<u32>) {}
        fn draw_indexed(&mut self, _: Range<u32>, _: i32, _: Range<u32>) {}
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
    let draw = functions.register_with_bind_groups_and_dynamic_contract(
        |item, ctx| ctx.set_pipeline(item.pipeline),
        Vec::new(),
        vec![RenderBindGroupBindingContract {
            identity: BindGroupBindingId(7),
            index: 1,
        }],
    );
    let item = PhaseItem {
        pipeline: RenderPipelineId::from_parts(50, 1),
        draw,
        geometry: GenerationalHandle::new(1, 1),
        vertices: vec![BufferId::from_parts(51, 1)],
        vertex_sizes: vec![64],
        indices: None,
        index_size: None,
        first: 0,
        count: 3,
        base_vertex: 0,
        indirect: None,
        lod_level: 0,
        material: GenerationalHandle::new(2, 1),
        scene: GenerationalHandle::new(3, 1),
        order: 0,
    };
    let mut log = Noop;
    functions.call_with_bindings(draw, &item, &[], &mut log);
}

#[test]
#[should_panic(expected = "unknown dynamic render bind group")]
fn dynamic_bind_group_contract_rejects_unknown_binding_with_offsets() {
    use crate::draw::{
        BindGroupBindingId, MeshRecorder, RenderBindGroupBinding, RenderBindGroupBindingContract,
    };
    use prism_render_driver::{BindGroupId, IndexFormat, RenderPipelineId};
    use std::ops::Range;

    #[derive(Default)]
    struct Noop;
    impl MeshRecorder for Noop {
        fn set_pipeline(&mut self, _: RenderPipelineId) {}
        fn set_render_bind_group(&mut self, _: u32, _: BindGroupId, _: Vec<u32>) {}
        fn set_vertex_buffer(
            &mut self,
            _: u32,
            _: BufferId,
            _: u64,
            _: Option<core::num::NonZeroU64>,
        ) {
        }
        fn set_index_buffer(
            &mut self,
            _: BufferId,
            _: IndexFormat,
            _: u64,
            _: Option<core::num::NonZeroU64>,
        ) {
        }
        fn draw(&mut self, _: Range<u32>, _: Range<u32>) {}
        fn draw_indexed(&mut self, _: Range<u32>, _: i32, _: Range<u32>) {}
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
    let draw = functions.register_with_bind_groups_and_dynamic_contract(
        |item, ctx| ctx.set_pipeline(item.pipeline),
        Vec::new(),
        vec![RenderBindGroupBindingContract {
            identity: BindGroupBindingId(7),
            index: 1,
        }],
    );
    let item = PhaseItem {
        pipeline: RenderPipelineId::from_parts(52, 1),
        draw,
        geometry: GenerationalHandle::new(1, 1),
        vertices: vec![BufferId::from_parts(53, 1)],
        vertex_sizes: vec![64],
        indices: None,
        index_size: None,
        first: 0,
        count: 3,
        base_vertex: 0,
        indirect: None,
        lod_level: 0,
        material: GenerationalHandle::new(2, 1),
        scene: GenerationalHandle::new(3, 1),
        order: 0,
    };
    let unknown = RenderBindGroupBinding {
        identity: BindGroupBindingId(8),
        index: 1,
        bind_group: BindGroupId::from_parts(54, 1),
        dynamic_offsets: vec![256, 512],
        buffers: Vec::new(),
        sampled_textures: Vec::new(),
        external_owners: Vec::new(),
    };
    let mut log = Noop;
    functions.call_with_bindings(draw, &item, &[unknown], &mut log);
}

#[test]
fn frame_graph_declares_dynamic_binding_buffers() {
    use crate::draw::{
        BindGroupBindingId, RenderBindGroupBinding, RenderBindGroupBuffer,
        RenderBindGroupBufferAccess,
    };
    use prism_render_driver::{BindGroupId, BufferUsages, RenderPipelineId};

    let extent = Extent3d::new_2d(4, 4);
    let runtime = RenderRuntime::headless(NullInstance::new(), extent)
        .expect("the null runtime always comes up");
    let target = RenderTarget::new(runtime.device(), extent);
    let dynamic_buffer = runtime.device().create_buffer(&BufferDescriptor {
        label: Some("dynamic-bind-group-buffer".into()),
        size: 64,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let binding = RenderBindGroupBinding {
        identity: BindGroupBindingId(91),
        index: 1,
        bind_group: BindGroupId::from_parts(92, 1),
        dynamic_offsets: Vec::new(),
        buffers: vec![RenderBindGroupBuffer {
            buffer: dynamic_buffer,
            access: RenderBindGroupBufferAccess::Storage,
        }],
        sampled_textures: Vec::new(),
        external_owners: Vec::new(),
    };
    let mut functions = DrawFunctions::new();
    let draw = functions.register(|item, context| {
        context.set_pipeline(item.pipeline);
        context.set_vertex_buffer(0, item.vertices[0], 0, None);
        context.draw(0..item.count, 0..1);
    });
    let item = PhaseItem {
        pipeline: RenderPipelineId::from_parts(93, 1),
        draw,
        geometry: GenerationalHandle::new(1, 1),
        vertices: vec![],
        vertex_sizes: vec![],
        indices: None,
        index_size: None,
        first: 0,
        count: 3,
        base_vertex: 0,
        indirect: None,
        lod_level: 0,
        material: GenerationalHandle::new(2, 1),
        scene: GenerationalHandle::new(3, 1),
        order: 0,
    };
    let mut queues = RenderQueues::new();
    queues.push_with_bind_groups(RenderPhase::Opaque, item, vec![binding]);

    let graph = crate::frame::target_graph(&target, &queues, &functions, None);
    let plan = graph.compile().expect("dynamic binding graph compiles");
    assert!(
        plan.used_buffers.iter().any(|used| *used),
        "the frame graph must realize the dynamic binding buffer"
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
            prefiltered_environment: false,
        },
        Tier::Balanced,
        GiCandidateAvailability::HostProvided,
    );
    let mut extracted = ExtractedLightingInputs::default();
    extracted.surfaces = vec![LightingSurfaceInput::default()];
    extracted.gi_candidates = vec![Vec::new()];
    lighting
        .prepare(&plan, &extracted, 0, &runtime)
        .expect("lighting tables prepare");

    let target = RenderTarget::new(runtime.device(), extent);
    let queues = RenderQueues::new();
    let functions = DrawFunctions::new();
    let graph = crate::frame::target_graph(
        &target,
        &queues,
        &functions,
        lighting.pass_for_target(&plan, &target, &runtime),
    );
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
            let label = label.expect("lighting compute passes are labeled");
            assert!(matches!(
                commands.first(),
                Some(ComputeCommand::SetPipeline(_))
            ));
            assert!(matches!(
                commands.get(1),
                Some(ComputeCommand::SetBindGroup { index: 0, .. })
            ));
            match label.as_str() {
                "prism-lighting-gi-dispatch" => {
                    assert_eq!(commands.len(), 3);
                    assert!(matches!(
                        commands.get(2),
                        Some(ComputeCommand::Dispatch { x: 1, y: 1, z: 1 })
                    ));
                }
                "prism-lighting-composite-dispatch" => {
                    assert_eq!(commands.len(), 4);
                    assert!(matches!(
                        commands.get(2),
                        Some(ComputeCommand::SetBindGroup { index: 1, .. })
                    ));
                    assert!(matches!(
                        commands.get(3),
                        Some(ComputeCommand::Dispatch { x: 1, y: 1, z: 1 })
                    ));
                }
                other => panic!("unexpected lighting compute pass {other}"),
            }
            label
        })
        .collect::<Vec<_>>();
    assert_eq!(
        compute_labels,
        [
            "prism-lighting-gi-dispatch",
            "prism-lighting-composite-dispatch",
        ]
    );
}

#[test]
fn frame_graph_records_optional_trace_between_scene_and_gi() {
    use crate::lighting::LightingTraceSceneInput;
    use crate::lighting_trace_gpu::{LightingTraceGpu, PackedTraceScene, TraceRay};
    use prism_render_architecture::lighting::contract::RecipeId;
    use prism_render_architecture::ray_scene::{
        Bvh, RayHitAttributeRecord, RayHitIdentity, Triangle,
    };
    use prism_render_driver::{BufferDescriptor, BufferUsages, Pass};

    let extent = Extent3d::new_2d(4, 4);
    let mut runtime = RenderRuntime::headless(NullInstance::new(), extent)
        .expect("the null runtime always comes up");
    let lighting = LightingGpuState::new(&mut runtime).expect("lighting GPU bundle");
    let plan = LightingPlan {
        tier: Tier::Balanced,
        candidate_source: GiCandidateSource::SoftwareBvhGpu,
        recipes: vec![RecipeId::WorldRestir],
        partition: Default::default(),
        budget: FrameResourceBudget::default(),
        rejected: Vec::new(),
    };
    let mut extracted = ExtractedLightingInputs::default();
    extracted.surfaces = vec![LightingSurfaceInput::default()];

    let triangle = Triangle::new([-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [0.0, 1.0, -1.0], 0);
    let bvh = Bvh::build(&[triangle]);
    let scene = PackedTraceScene::from(
        &prism_render_architecture::ray_scene::GpuBvhBuffers::from_bvh(&bvh),
    );
    extracted.trace_scene = Some(LightingTraceSceneInput::new(
        std::sync::Arc::new(prism_render_architecture::ray_scene::GpuBvhBuffers::from_bvh(&bvh)),
        vec![1],
        1,
        0.001,
        10.0,
        std::sync::Arc::from([crate::LightingTriangleRadiance {
            primitive_id: 0,
            rgb: [1.0, 1.0, 1.0],
        }]),
        1,
    ));
    lighting
        .prepare(&plan, &extracted, 0, &runtime)
        .expect("lighting tables prepare");
    let rays = [TraceRay {
        origin: [0.0, 0.0, 0.0],
        t_min: 0.0,
        direction: [0.0, 0.0, -1.0],
        t_max: 10.0,
        direction_pdf: 1.0 / core::f32::consts::PI,
        selection_u01: 0.5,
    }];
    let output = runtime.device().create_buffer(&BufferDescriptor::new(
        32,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    ));
    let mut trace = LightingTraceGpu::new(runtime.device_handle());
    let borrowed_scene = crate::lighting_trace_gpu::BorrowedPackedTraceScene::from(&scene);
    let lighting_pass = lighting.pass(&plan).expect("the GI recipe has a surface");
    let recorder = trace
        .prepare_graph(
            runtime.queue(),
            &borrowed_scene,
            &rays,
            output,
            32,
            Some(lighting_pass.gi_candidate_target()),
        )
        .expect("trace inputs prepare without submitting a command buffer");
    let target = RenderTarget::new(runtime.device(), extent);
    let queues = RenderQueues::new();
    let functions = DrawFunctions::new();

    // The no-trace path must use an admitted host producer; a software-BVH
    // plan cannot be recorded without its trace/candidate-build passes.
    let mut host_plan = plan.clone();
    host_plan.candidate_source = GiCandidateSource::HostProvided;
    extracted.gi_candidates = vec![Vec::new(); extracted.surfaces.len()];
    lighting
        .prepare(&host_plan, &extracted, 0, &runtime)
        .expect("host lighting tables prepare");
    let absent = crate::frame::target_graph(
        &target,
        &queues,
        &functions,
        lighting.pass_for_target(&host_plan, &target, &runtime),
    );
    assert_eq!(
        absent
            .blackboard()
            .get::<crate::frame::FrameGraphStages>()
            .unwrap()
            .trace_dispatches,
        0
    );
    absent
        .blackboard()
        .get::<crate::frame::FrameGraphStages>()
        .unwrap()
        .assert_contract();
    lighting
        .prepare(&plan, &extracted, 0, &runtime)
        .expect("trace lighting tables prepare");

    let graph = crate::frame::target_graph_with_trace(
        &target,
        &queues,
        &functions,
        Some(lighting_pass),
        Some(recorder),
    );
    let stages = *graph
        .blackboard()
        .get::<crate::frame::FrameGraphStages>()
        .expect("trace-aware graph publishes stage receipt");
    stages.assert_contract();
    assert_eq!(stages.trace_dispatches, 1);
    assert_eq!(stages.candidate_build_dispatches, 1);
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
            "prism-render-app-clear-gbuffer",
            "prism-lighting-trace-dispatch",
            "prism-lighting-gi-candidate-build",
            "prism-lighting-gi-dispatch",
            "prism-lighting-composite-dispatch",
        ]
    );
}

#[test]
fn physical_gpu_offscreen_host_graph_traces_builds_gi_and_composites() {
    use crate::lighting::{LightingTraceSceneInput, LightingTriangleRadiance};
    use crate::lighting_trace_gpu::{
        build_gi_candidates_reference, trace_reference, BorrowedPackedTraceScene, LightingTraceGpu,
        PackedTraceScene, TraceRay,
    };
    use prism_render_architecture::lighting::contract::BudgetCompiler;
    use prism_render_architecture::ray_scene::{
        Bvh, GpuBvhBuffers, RayHitAttributeRecord, RayHitIdentity, Triangle,
    };
    use prism_render_driver::{BufferDescriptor, BufferUsages, DeviceType};
    use prism_render_driver_wgpu::WgpuInstance;

    const FRAME_INDEX: u64 = 19;
    const SURFACE_ID: u64 = 0xA11CE;
    let extent = Extent3d::new_2d(4, 4);
    let mut runtime = match RenderRuntime::headless(WgpuInstance::new(), extent) {
        Ok(runtime) => runtime,
        Err(error) => {
            assert!(
                !require_gpu(),
                "PRISM_REQUIRE_GPU=1 but physical GPU runtime creation failed: {error}"
            );
            eprintln!("SKIPPED physical host GI graph: {error}");
            return;
        }
    };
    if !matches!(
        runtime.adapter_info().device_type,
        DeviceType::IntegratedGpu | DeviceType::DiscreteGpu
    ) {
        assert!(
            !require_gpu(),
            "PRISM_REQUIRE_GPU=1 requires a physical GPU, found {:?} ({})",
            runtime.adapter_info().device_type,
            runtime.adapter_info().name
        );
        eprintln!(
            "SKIPPED physical host GI graph: adapter is {:?} ({})",
            runtime.adapter_info().device_type,
            runtime.adapter_info().name
        );
        return;
    }

    let triangle = Triangle::new([-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [0.0, 1.0, -1.0], 73);
    let bvh = Arc::new(
        GpuBvhBuffers::from_bvh(&Bvh::build(&[triangle]))
            .with_hit_attributes(vec![RayHitAttributeRecord::new(
                RayHitIdentity::new(11, 73, 1),
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                [[0.0, 0.0, 1.0]; 3],
            )])
            .expect("valid secondary shading attributes"),
    );
    let packed_scene = PackedTraceScene::from(&*bvh);
    let surface = LightingSurfaceInput {
        position: [0.0, 0.0, 0.0],
        normal: [0.0, 0.0, -1.0],
        albedo: [0.8, 0.5, 0.25],
        ambient: [0.0; 3],
    };
    // The primitive id is deliberately explicit: the per-triangle radiance
    // contract must resolve the packed BVH primitive rather than rely on an
    // accidental vector-index == primitive-id assumption.
    let radiance = vec![LightingTriangleRadiance {
        primitive_id: 73,
        rgb: [4.0, 2.0, 1.0],
    }];
    let trace_scene = LightingTraceSceneInput::new(
        bvh.clone(),
        vec![SURFACE_ID],
        1,
        0.001,
        10.0,
        Arc::from(radiance.clone()),
        1,
    );
    let rays = [TraceRay {
        origin: surface.position,
        t_min: trace_scene.ray_epsilon,
        direction: [0.0, 0.0, -1.0],
        t_max: trace_scene.ray_tmax,
        direction_pdf: 1.0 / core::f32::consts::PI,
        selection_u01: 0.5,
    }];
    let mut extracted = ExtractedLightingInputs::default();
    extracted.surfaces = vec![surface];
    extracted.trace_scene = Some(trace_scene);
    let availability = extracted.candidate_availability(runtime.device().capabilities());
    assert_eq!(availability, GiCandidateAvailability::SoftwareBvhGpu);
    let config = RenderAppConfig::new(extent);
    let plan = BudgetCompiler::compile(
        extracted.stats,
        config.lighting_budget,
        config.lighting_caps,
        config.lighting_tier,
        availability,
    );
    assert_eq!(plan.candidate_source, GiCandidateSource::SoftwareBvhGpu);
    assert!(plan.recipes.contains(&RecipeId::WorldRestir));
    assert!(plan.recipes.contains(&RecipeId::SpecializedDeterministic));

    let lighting = LightingGpuState::new(&mut runtime).expect("lighting GPU bundle");
    lighting
        .prepare(&plan, &extracted, FRAME_INDEX, &runtime)
        .expect("lighting tables prepare");
    let target = RenderTarget::new(runtime.device(), extent);
    let mut material_registry = prism_render_material::MaterialRegistry::new(1);
    material_registry
        .install_execution_tables(&prism_render_material::MaterialExecutionTables::new())
        .expect("empty canonical execution tables are valid for fallback material");
    let device = runtime.device_handle();
    let mut material_buffers =
        MaterialParameterBuffers::new_owned(device.clone(), 1, 1, 1, BufferUsages::NONE);
    material_buffers.sync_registry(runtime.device(), runtime.queue(), &material_registry, [0]);
    let mut execution_buffers = crate::material::NativeExecutionTableBuffers::new();
    execution_buffers
        .sync_registry(device, runtime.queue(), &material_registry)
        .expect("canonical execution-table upload is available");
    assert!(runtime.wait_idle(), "material snapshots publish on GPU");
    let material_snapshot = material_buffers
        .snapshot()
        .expect("material allocation snapshot");
    let execution_snapshot = execution_buffers.snapshot().expect("execution snapshot");
    let material_bindings = crate::lighting_gpu::LightingCompositeMaterialBuffers {
        headers: Some(material_snapshot.headers()),
        headers_size_bytes: u64::from(material_snapshot.header_capacity())
            * core::mem::size_of::<prism_render_material::GpuMaterialHeader>() as u64,
        parameters: Some(material_snapshot.parameters()),
        parameters_size_bytes: u64::from(material_snapshot.parameter_capacity_words()) * 4,
        execution_plans: Some(execution_snapshot.plan_buffer()),
        execution_plans_size_bytes: execution_snapshot.plan_rows().len() as u64
            * (prism_render_material::EXECUTION_PLAN_ROW_WORDS * 4) as u64,
        execution_programs: Some(execution_snapshot.program_buffer()),
        execution_programs_size_bytes: execution_snapshot.program_rows().len() as u64 * 20,
    };
    let lighting_pass = lighting
        .pass_for_target_with_materials(&plan, &target, &runtime, material_bindings)
        .expect("GI surface admits its target-aware recipe");
    let hit_output = runtime.device().create_buffer(&BufferDescriptor::new(
        (core::mem::size_of::<u32>() * 16) as u64,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    ));
    let mut trace = LightingTraceGpu::new(runtime.device_handle());
    let recorder = trace
        .prepare_graph(
            runtime.queue(),
            &BorrowedPackedTraceScene::from(&packed_scene),
            &rays,
            hit_output,
            (core::mem::size_of::<u32>() * 16) as u64,
            Some(lighting_pass.gi_candidate_target()),
        )
        .expect("valid scene, rays, radiance table, and GI targets prepare");

    // This is the same target_graph_with_trace constructor used by the
    // offscreen/window host composition path, not an independently assembled
    // trace/candidate/GI graph.
    let queues = RenderQueues::new();
    let functions = DrawFunctions::new();
    let mut graph = crate::frame::target_graph_with_trace(
        &target,
        &queues,
        &functions,
        Some(lighting_pass),
        Some(recorder),
    );
    graph.retain_external_owner(material_snapshot.external_owner());
    graph.retain_external_owner(execution_snapshot.external_owner());
    let stages = *graph
        .blackboard()
        .get::<crate::frame::FrameGraphStages>()
        .expect("host frame graph publishes its stage receipt");
    stages.assert_contract();
    assert_eq!(stages.trace_dispatches, 1);
    assert_eq!(stages.candidate_build_dispatches, 1);
    assert_eq!(stages.gi_recipes, 1);
    assert_eq!(stages.direct_recipes, 1);
    assert_eq!(stages.composite_passes, 1);
    assert_eq!(
        graph.pass_count(),
        5,
        "clear, trace, candidate, GI, composite"
    );
    let recorded = graph
        .execute(runtime.device(), runtime.queue())
        .expect("offscreen host graph submits on the physical GPU");
    assert!(runtime.wait_idle(), "host graph GPU work completes");
    drop(recorded);

    let cpu_hits = trace_reference(&packed_scene, &rays).expect("valid CPU trace oracle");
    assert_eq!(cpu_hits.len(), 1);
    assert!(cpu_hits[0].hit, "fixture ray must hit the packed triangle");
    let outgoing_rgb = radiance.iter().map(|entry| entry.rgb).collect::<Vec<_>>();
    let cpu_candidates = build_gi_candidates_reference(
        &packed_scene,
        &rays,
        &cpu_hits,
        &[surface],
        &[SURFACE_ID],
        FRAME_INDEX,
        &outgoing_rgb,
    )
    .expect("CPU candidate oracle accepts the same authored inputs");
    let expected_gi = evaluate_deterministic_gi(surface, &cpu_candidates[0]);

    let gi = runtime
        .queue()
        .read_buffer(
            lighting.gi_output_buffer(),
            crate::lighting_gpu::OUTPUT_BYTES,
        )
        .expect("host graph GI output readback must complete")
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("f32-sized output")))
        .collect::<Vec<_>>();
    for channel in 0..3 {
        assert!(
            (gi[channel] - expected_gi[channel]).abs() <= 2.0e-4,
            "host graph GI channel {channel}: GPU {} != CPU {}",
            gi[channel],
            expected_gi[channel]
        );
    }

    // This fixture has no raster draw queue, so the same-frame graph composite
    // must preserve the cleared-coverage result. Do not inject a synthetic
    // GBuffer pixel and dispatch a second composite: that would not prove
    // scene -> trace -> GI -> composite parity.
    let hdr = runtime
        .queue()
        .read_texture_rgba16f(target.texture(), extent.width, extent.height)
        .expect("host graph HDR readback must complete");
    assert_eq!(hdr.len(), (extent.width * extent.height) as usize);
    assert!(expected_gi.iter().any(|channel| *channel > 0.0));
    assert!(
        hdr.iter().all(|pixel| *pixel == [0.0; 4]),
        "a host graph with no raster coverage composites cleared pixels"
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
        candidate_source: GiCandidateSource::HostProvided,
        recipes: vec![RecipeId::WorldRestir],
        partition: Default::default(),
        budget: FrameResourceBudget::default(),
        rejected: Vec::new(),
    };
    let mut extracted = ExtractedLightingInputs::default();
    extracted.surfaces = vec![LightingSurfaceInput::default()];
    extracted.gi_candidates = vec![Vec::new()];
    lighting
        .prepare(&plan, &extracted, 0, &runtime)
        .expect("lighting tables prepare");

    let target = RenderTarget::new(runtime.device(), extent);
    let graph = crate::frame::target_graph(
        &target,
        &RenderQueues::new(),
        &DrawFunctions::new(),
        lighting.pass_for_target(&plan, &target, &runtime),
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
        [
            "prism-lighting-gi-dispatch",
            "prism-lighting-composite-dispatch",
        ]
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
        candidate_source: GiCandidateSource::Disabled,
        recipes: Vec::new(),
        partition: Default::default(),
        budget: FrameResourceBudget::default(),
        rejected: Vec::new(),
    };
    lighting
        .prepare(&plan, &ExtractedLightingInputs::default(), 0, &runtime)
        .expect("lighting tables prepare");

    let target = RenderTarget::new(runtime.device(), extent);
    let graph = crate::frame::target_graph(
        &target,
        &RenderQueues::new(),
        &DrawFunctions::new(),
        lighting.pass_for_target(&plan, &target, &runtime),
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
        candidate_source: GiCandidateSource::HostProvided,
        recipes: vec![RecipeId::SpecializedDeterministic, RecipeId::WorldRestir],
        partition: Default::default(),
        budget: FrameResourceBudget::default(),
        rejected: Vec::new(),
    };
    let extracted = ExtractedLightingInputs::default();
    lighting
        .prepare(&plan, &extracted, 0, &runtime)
        .expect("lighting tables prepare");
    assert!(
        lighting.pass(&plan).is_none(),
        "empty batches have no lighting pass"
    );

    let target = RenderTarget::new(runtime.device(), extent);
    let graph = crate::frame::target_graph(
        &target,
        &RenderQueues::new(),
        &DrawFunctions::new(),
        lighting.pass_for_target(&plan, &target, &runtime),
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
            vertex_sizes: vec![64],
            indices: Some((indices, IndexFormat::Uint16)),
            index_size: Some(64),
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
                    shader_package: None,
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
    let mut fixture = bin_fixture(&device, &queue);
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
                    shader_package: None,
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
                    shader_package: None,
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
                    shader_package: None,
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
    let mut fixture = bin_fixture(&device, &queue);
    let provider = StubProvider::new(&device);

    let mut candidates = fixture
        .registry
        .bin_candidates(&fixture.meshes, &fixture.work);
    // A foreign descriptor-table ID must never silently drop geometry.
    let mut foreign = candidates[0];
    foreign.key.pipeline_class = u32::MAX;
    let frozen_material = fixture
        .instances
        .get(foreign.scene)
        .expect("bin representative is resident")
        .material;
    fixture.registry.set_material_frozen(frozen_material, true);
    let frozen_bins = build_view_draw_bins(GenerationalHandle::new(0, 1), 8, [foreign]);
    let frozen_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        RenderQueues::new().fill_bins(
            &device,
            &mut MeshPipelineSpecializer::new(),
            &fixture.registry,
            &fixture.meshes,
            &fixture.instances,
            declared_args_buffer(&device),
            &frozen_bins,
            None,
            1,
            DrawFunctionId(0),
            &provider,
        )
    }));
    assert!(
        frozen_result.is_err(),
        "a frozen material cannot fall through the unknown-class spine fallback"
    );
    fixture.registry.set_material_frozen(frozen_material, false);
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
        draw: crate::draw::canonical::MESH_UNBOUND,
        geometry: GenerationalHandle::new(1, 1),
        vertices: vec![vertices],
        vertex_sizes: vec![64],
        indices: Some((indices, prism_render_driver::IndexFormat::Uint16)),
        index_size: Some(64),
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
                draw: crate::draw::canonical::MESH_UNBOUND,
                geometry: GenerationalHandle::new(1, 1),
                vertices,
                vertex_sizes: vec![index_count as u64 * 4],
                indices: Some((indices, prism_render_driver::IndexFormat::Uint16)),
                index_size: Some(index_count as u64 * 2),
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
        .read_texture_rgba16f(target.texture(), SIZE, SIZE)
        .expect("mesh HDR target readback must complete");
    assert_eq!(pixels.len(), (SIZE * SIZE) as usize);
    let pixel = |x: u32, y: u32| pixels[(y * SIZE + x) as usize];
    assert_eq!(
        pixel(SIZE / 2, SIZE / 2),
        [1.0, 0.0, 0.0, 1.0],
        "the queued mesh rasterized in the frame's target"
    );
    assert_eq!(
        pixel(0, 0),
        [0.0, 0.0, 0.0, 1.0],
        "outside the quad the frame's clear is what remains"
    );
}
/// Stable identifiers for the real camera/renderable queue fixture.
#[derive(prism_ecs::prelude::Resource)]
pub(super) struct QueueSceneFixtureIdentity {
    pub scene: prism_render_architecture::gpu_scene::SceneHandle,
    pub instance: prism_ecs::prelude::Entity,
}

pub(super) fn queue_scene_handle(app: &App) -> prism_render_architecture::gpu_scene::SceneHandle {
    app.world().resource::<QueueSceneFixtureIdentity>().scene
}

/// The scene fixture the queue-stage tests draw: a real Prism camera and
/// renderable feed native per-view culling; matching mesh/material records are
/// registered in the render world's queue resolver.
pub(super) fn queue_scene_fixture(app: &mut App) -> GenerationalHandle {
    use prism_math::{Frustum, Mat4, Plane, Vec3};
    use prism_mesh::{Indices, Mesh, MeshVertexAttribute, VertexAttributeValues};
    use prism_render_architecture::geometry::{
        GeometryLodRecord, GeometryPrimitiveKind, GeometryRecord,
    };
    use prism_render_architecture::gpu_scene::SceneHandle;
    use prism_render_visibility::{
        GeometryLod, GeometryLodChain, GpuRenderWorkItem, RenderPassMask, VisibilityStageMask,
        WorkSortKey,
    };
    use prism_transform::ecs::LocalBounds;
    use prism_transform::{GlobalTransform, Transform};

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
        .with_attribute(
            MeshVertexAttribute::new(
                "Vertex_Normal",
                prism_render_driver::VertexFormat::Float32x3,
                1,
            ),
            VertexAttributeValues::Float32x3(vec![
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ]),
        )
        .expect("normals fit")
        .with_attribute(
            MeshVertexAttribute::new("Vertex_Uv", prism_render_driver::VertexFormat::Float32x2, 2),
            VertexAttributeValues::Float32x2(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        )
        .expect("UVs fit")
        .indexed(Indices::Uint16(vec![0, 1, 2, 0, 2, 3]));
    let mut meshes = prism_asset::Assets::new();
    let mesh_id = meshes.insert(mesh).id();
    app.world_mut().insert_resource(meshes);

    let geometry = GenerationalHandle::new(1, 1);
    let material = GenerationalHandle::new(2, 1);
    if app
        .world()
        .get_resource::<crate::visibility::PrismRenderScene>()
        .is_none()
    {
        app.add_plugins(crate::visibility::PrismVisibilityPlugin);
    }
    {
        let scene = app
            .world_mut()
            .resource_mut::<crate::visibility::PrismRenderScene>();
        scene.insert_geometry(GeometryLodChain {
            geometry,
            lods: vec![GeometryLod {
                level: 0,
                screen_error: 0.0,
                resident: true,
                fallback: true,
            }],
        });
        scene.insert_material(prism_render_material::fallback_material_record(material, 1));
    }
    let mut camera = crate::visibility::PrismCamera::new(Mat4::IDENTITY, [0, 0, 8, 8]);
    camera.layer_mask = 1;
    app.world_mut().spawn((
        camera,
        crate::visibility::PrismFrustum(Frustum::from_planes([
            Plane::new(Vec3::new(1.0, 0.0, 0.0), 10.0),
            Plane::new(Vec3::new(-1.0, 0.0, 0.0), 10.0),
            Plane::new(Vec3::new(0.0, -1.0, 0.0), 10.0),
            Plane::new(Vec3::new(0.0, 1.0, 0.0), 10.0),
            Plane::new(Vec3::new(0.0, 0.0, -1.0), 1.0),
            Plane::new(Vec3::new(0.0, 0.0, 1.0), 100.0),
        ])),
        GlobalTransform::from_transform(&Transform::from_xyz(0.0, 0.0, 0.0)),
    ));
    let mut renderable = crate::visibility::PrismRenderable::new(geometry, material);
    renderable.render_layers = 1;
    let instance = app.world_mut().spawn((
        renderable,
        LocalBounds(prism_math::Aabb3::from_center_half_extents(
            Vec3::ZERO,
            Vec3::splat(0.5),
        )),
        GlobalTransform::IDENTITY,
        crate::visibility::PrismViewVisibility::default(),
    ));
    crate::visibility::sync_scene(app.world_mut());
    let scene_handle = app
        .world()
        .resource::<crate::visibility::PrismRenderScene>()
        .handle_for(instance)
        .expect("the real renderable receives a stable scene identity");
    debug_assert_eq!(scene_handle, SceneHandle::new(1, 1));
    app.world_mut().insert_resource(QueueSceneFixtureIdentity {
        scene: scene_handle,
        instance,
    });
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
    {
        let materials = render.world.resource_mut::<RenderMaterials>();
        let first = materials
            .registry_mut()
            .allocate()
            .expect("the material fixture has room for the first slot");
        assert_eq!(first.index, 1);
        let published = materials
            .registry_mut()
            .allocate()
            .expect("the material fixture has room for the scene slot");
        assert_eq!(published, material);
        materials
            .publish(prism_render_material::fallback_material_record(material, 1))
            .expect("the scene material publishes into the canonical registry");
    }
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
    geometry
}

#[test]
fn canonical_default_mesh_binding_uses_material_generation_and_graph_buffers() {
    use crate::mesh_pipelines::MeshPipelines;
    use prism_render_driver::{BufferUsages, DeviceDescriptor, Instance, RequestAdapterOptions};

    let adapter = NullInstance::new()
        .request_adapter(&RequestAdapterOptions::default())
        .expect("the null backend has an adapter");
    let (device, queue) = NullInstance::new()
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null device comes up");
    let pipelines = MeshPipelines::surface_encode(&device);
    let mut bindings = CanonicalMaterialBindings::new(4);
    assert!(bindings.ensure_layout(&device, 0, pipelines.surface_bind_group_layout()));
    assert!(bindings.prepare_capacity(&device, 4));
    assert!(!bindings.prepare_frame(&queue, None));
    let view = crate::surface_views::SurfaceView {
        camera: prism_ecs::prelude::Entity::from_bits(1_u64 << 32).unwrap(),
        target: prism_camera::RenderTarget::PrimaryWindow,
        clear: prism_camera::ClearMode::ColorAndDepth,
        order: 0,
        clip_from_world: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        previous_clip_from_world: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        viewport: [0, 0, 8, 8],
        history_epoch: 0,
        history_valid: true,
        resolution_scale: 1.0,
        history_frames: 0,
        exposure_stops: 0.0,
    };
    assert!(bindings.prepare_frame(&queue, Some(&view)));

    let mut buffers = MaterialParameterBuffers::new(&device, 4, 64, 4, BufferUsages::NONE);
    let mut registry = prism_render_material::MaterialRegistry::new(1);
    let handle = registry.allocate().unwrap();
    let record = prism_render_material::fallback_material_record(handle, 3);
    registry.publish(record.clone()).unwrap();
    buffers.sync_registry(&device, &queue, &registry, [handle.index]);
    assert!(bindings.prepare_material_buffers(&device, &buffers));

    let binding = bindings
        .binding_for(&queue, handle, &record, &buffers)
        .expect("the live generation/revision resolves to a binding");
    assert_eq!(binding.identity, crate::draw::canonical::SURFACE_MATERIAL);
    assert_eq!(binding.index, crate::draw::canonical::SURFACE_GROUP);
    assert_eq!(binding.dynamic_offsets, vec![0, 256]);
    assert_eq!(binding.buffers.len(), 5);
    assert!(binding.buffers[..2].iter().all(|buffer| matches!(
        buffer.access,
        crate::draw::RenderBindGroupBufferAccess::Uniform
    )));
    assert_ne!(binding.buffers[0].buffer, binding.buffers[1].buffer);
    assert_eq!(
        binding.buffers[2..],
        [
            crate::draw::RenderBindGroupBuffer {
                buffer: buffers.headers().unwrap(),
                access: crate::draw::RenderBindGroupBufferAccess::Storage,
            },
            crate::draw::RenderBindGroupBuffer {
                buffer: buffers.parameters().unwrap(),
                access: crate::draw::RenderBindGroupBufferAccess::Storage,
            },
        ]
    );

    let stale = GenerationalHandle::new(handle.index, handle.generation + 1);
    assert!(bindings
        .binding_for(&queue, stale, &record, &buffers)
        .is_none());
}

#[test]
fn the_queue_stage_turns_the_visible_set_into_draws() {
    let extent = Extent3d::new_2d(8, 8);
    let mut app = app_on_null_backend(RenderAppConfig::new(extent));
    let geometry = queue_scene_fixture(&mut app);
    capture_camera_queues(&mut app);

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
    assert_eq!(stats.view_queue_rejection, None);
    let snapshot = render.world.resource::<CameraQueueSnapshot>();
    assert!(snapshot.cameras.is_ready());
    assert_eq!(snapshot.cameras.len(), 1);
    assert!(
        snapshot.global.is_empty(),
        "automatic work is not global work"
    );
    let camera = snapshot.cameras.iter().next().unwrap();
    assert_eq!(camera.frame_slot, 0);
    assert_eq!(camera.queues.phase(RenderPhase::Opaque).len(), 1);
    assert_eq!(
        camera.queues.phase(RenderPhase::Opaque)[0].geometry,
        geometry
    );
    assert_eq!(
        camera.queues.dynamic_bind_groups(RenderPhase::Opaque, 0)[0].dynamic_offsets[0],
        0
    );
}

/// Observe Queue before the frame consumer and Cleanup mutate frame-local data.
/// These tests assert queue production, independently of output routing/recording.
#[derive(Default, Resource)]
struct CameraQueueSnapshot {
    cameras: crate::SurfaceViewQueues,
    global: RenderQueues,
    windows: std::collections::BTreeMap<prism_window::WindowId, RenderQueues>,
}

fn snapshot_camera_queues(
    cameras: prism_ecs::prelude::Res<crate::SurfaceViewQueues>,
    global: prism_ecs::prelude::Res<RenderQueues>,
    windows: prism_ecs::prelude::Res<crate::WindowRenderQueues>,
    mut snapshot: ResMut<CameraQueueSnapshot>,
) {
    snapshot.cameras = (*cameras).clone();
    snapshot.global = (*global).clone();
    snapshot.windows = windows.0.clone();
}

fn capture_camera_queues(app: &mut App) {
    let render = app.sub_app_mut(RenderApp).expect("the render sub-app");
    render.world.init_resource::<CameraQueueSnapshot>();
    render
        .world
        .resource_mut::<Schedules>()
        .get_mut(Render)
        .expect("the Render schedule")
        .add_systems(snapshot_camera_queues.in_set(RenderSystems::Prepare));
}

fn spawn_queue_camera(
    app: &mut App,
    layer_mask: u32,
    translation: f32,
) -> prism_ecs::prelude::Entity {
    use prism_math::{Frustum, Mat4, Plane, Vec3, Vec4};

    let mut camera = crate::visibility::PrismCamera::new(
        Mat4::from_cols(
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(translation, 0.0, 0.0, 1.0),
        ),
        [0, 0, 4, 8],
    );
    camera.layer_mask = layer_mask;
    app.world_mut().spawn((
        camera,
        crate::visibility::PrismFrustum(Frustum::from_planes([
            Plane::new(Vec3::X, 10.0),
            Plane::new(-Vec3::X, 10.0),
            Plane::new(Vec3::Y, 10.0),
            Plane::new(-Vec3::Y, 10.0),
            Plane::new(-Vec3::Z, 1.0),
            Plane::new(Vec3::Z, 100.0),
        ])),
        prism_transform::GlobalTransform::IDENTITY,
    ))
}

#[test]
fn the_queue_stage_admits_multiple_cameras_including_empty_work() {
    let mut app = app_on_null_backend(RenderAppConfig::new(Extent3d::new_2d(8, 8)));
    queue_scene_fixture(&mut app);
    capture_camera_queues(&mut app);
    // This camera's layer excludes the object. A valid empty camera must still
    // reserve a frame slot, rather than inheriting another camera's draws.
    let second = spawn_queue_camera(&mut app, 2, 0.25);

    app.update();

    let render = app.get_sub_app(RenderApp).expect("the render sub-app");
    let views = render.world.resource::<ExtractedVisibilityViews>();
    assert_eq!(views.len(), 2, "both validated cameras survive extraction");
    assert_eq!(views.get(second).unwrap().work.len(), 0);
    let stats = render.world.resource::<RenderFrameStats>();
    assert_eq!(stats.draws_queued, 1);
    assert_eq!(stats.view_queue_rejection, None);
    let snapshot = render.world.resource::<CameraQueueSnapshot>();
    assert!(snapshot.cameras.is_ready());
    assert!(snapshot.cameras.rejections().is_empty());
    assert_eq!(snapshot.cameras.len(), 2);
    let cameras: Vec<_> = snapshot.cameras.iter().collect();
    assert!(
        (
            cameras[0].camera().index(),
            cameras[0].camera().generation()
        ) < (
            cameras[1].camera().index(),
            cameras[1].camera().generation()
        )
    );
    assert_eq!([cameras[0].frame_slot, cameras[1].frame_slot], [0, 1]);
    assert_eq!(cameras.iter().map(|v| v.queues.len()).sum::<usize>(), 1);
    let empty = snapshot.cameras.get(second).unwrap();
    assert!(empty.queues.is_empty());
    assert_eq!(empty.surface.viewport, [0, 0, 4, 8]);
    assert_eq!(empty.surface.clip_from_world[3][0], 0.25);
    assert!(snapshot.global.is_empty());
}

#[test]
fn the_queue_stage_shares_materials_but_not_camera_frame_slots() {
    let mut app = app_on_null_backend(RenderAppConfig::new(Extent3d::new_2d(8, 8)));
    queue_scene_fixture(&mut app);
    capture_camera_queues(&mut app);
    let second = spawn_queue_camera(&mut app, 1, 0.25);
    app.update();

    let first_camera = {
        let render = app.get_sub_app(RenderApp).unwrap();
        let snapshot = render.world.resource::<CameraQueueSnapshot>();
        assert!(snapshot.cameras.is_ready());
        assert_eq!(snapshot.cameras.len(), 2);
        let cameras: Vec<_> = snapshot.cameras.iter().collect();
        assert_eq!([cameras[0].frame_slot, cameras[1].frame_slot], [0, 1]);
        for camera in &cameras {
            assert_eq!(camera.queues.phase(RenderPhase::Opaque).len(), 1);
        }
        let a = &cameras[0]
            .queues
            .dynamic_bind_groups(RenderPhase::Opaque, 0)[0];
        let b = &cameras[1]
            .queues
            .dynamic_bind_groups(RenderPhase::Opaque, 0)[0];
        assert_eq!(a.bind_group, b.bind_group);
        assert_eq!(a.buffers, b.buffers, "geometry/material storage is shared");
        assert_eq!(a.dynamic_offsets.len(), 2);
        assert_eq!(b.dynamic_offsets.len(), 2);
        assert_eq!(a.dynamic_offsets[0], 0);
        assert!(b.dynamic_offsets[0] > a.dynamic_offsets[0]);
        assert_eq!(a.dynamic_offsets[1], b.dynamic_offsets[1]);
        assert_ne!(
            cameras[0].surface.clip_from_world,
            cameras[1].surface.clip_from_world
        );
        cameras
            .iter()
            .find(|v| v.camera() != second)
            .unwrap()
            .camera()
    };

    // Removing one camera discards its queue, and a later camera is admitted
    // with current matrices rather than reusing the removed camera's snapshot.
    assert!(app.world_mut().despawn(second));
    app.update();
    {
        let render = app.get_sub_app(RenderApp).unwrap();
        let snapshot = render.world.resource::<CameraQueueSnapshot>();
        assert!(snapshot.cameras.is_ready());
        assert_eq!(snapshot.cameras.len(), 1);
        assert!(snapshot.cameras.get(second).is_none());
        assert_eq!(snapshot.cameras.get(first_camera).unwrap().frame_slot, 0);
    }
    let replacement = spawn_queue_camera(&mut app, 1, -0.25);
    app.update();
    let render = app.get_sub_app(RenderApp).unwrap();
    let snapshot = render.world.resource::<CameraQueueSnapshot>();
    assert!(snapshot.cameras.is_ready());
    assert_eq!(snapshot.cameras.len(), 2);
    assert!(snapshot.cameras.get(second).is_none());
    assert_eq!(
        snapshot
            .cameras
            .get(replacement)
            .unwrap()
            .surface
            .clip_from_world[3][0],
        -0.25
    );
    assert_eq!(
        snapshot
            .cameras
            .iter()
            .map(|v| v.frame_slot)
            .collect::<Vec<_>>(),
        [0, 1]
    );
}

#[test]
fn the_queue_stage_preserves_explicit_global_and_window_producers() {
    let mut app = app_on_null_backend(RenderAppConfig::new(Extent3d::new_2d(8, 8)));
    queue_scene_fixture(&mut app);
    capture_camera_queues(&mut app);
    app.update();
    // Reuse a real prepared draw/binding, so the frame consumer sees valid
    // resources too. Queue must neither replace nor append to this producer.
    let mut explicit = {
        let render = app.get_sub_app(RenderApp).unwrap();
        render
            .world
            .resource::<CameraQueueSnapshot>()
            .cameras
            .iter()
            .next()
            .unwrap()
            .queues
            .clone()
    };
    let original = explicit.phase(RenderPhase::Opaque)[0].clone();
    let bindings = explicit
        .dynamic_bind_groups(RenderPhase::Opaque, 0)
        .to_vec();
    explicit.clear();
    let mut item = original;
    item.order = 777;
    explicit.push_with_bind_groups(RenderPhase::Opaque, item.clone(), bindings.clone());
    {
        let render = app.sub_app_mut(RenderApp).unwrap();
        render.world.insert_resource(explicit.clone());
        let windows = render.world.resource_mut::<crate::WindowRenderQueues>();
        windows.0.insert(prism_window::WindowId(17), explicit);
        windows
            .0
            .insert(prism_window::WindowId(18), RenderQueues::new());
    }
    app.update();
    let render = app.get_sub_app(RenderApp).unwrap();
    let snapshot = render.world.resource::<CameraQueueSnapshot>();
    assert!(snapshot.cameras.is_ready());
    assert_eq!(snapshot.cameras.len(), 1);
    assert_eq!(snapshot.cameras.iter().next().unwrap().queues.len(), 1);
    assert_eq!(snapshot.global.phase(RenderPhase::Opaque), &[item.clone()]);
    assert_eq!(
        snapshot.global.dynamic_bind_groups(RenderPhase::Opaque, 0),
        bindings
    );
    assert_eq!(snapshot.windows.len(), 2);
    let window = &snapshot.windows[&prism_window::WindowId(17)];
    assert_eq!(window.phase(RenderPhase::Opaque), &[item]);
    assert_eq!(window.dynamic_bind_groups(RenderPhase::Opaque, 0), bindings);
    assert!(snapshot.windows[&prism_window::WindowId(18)].is_empty());
}

#[test]
fn the_queue_stage_keeps_valid_camera_when_another_camera_is_invalid() {
    let mut app = app_on_null_backend(RenderAppConfig::new(Extent3d::new_2d(8, 8)));
    queue_scene_fixture(&mut app);
    capture_camera_queues(&mut app);
    let invalid = spawn_queue_camera(&mut app, 1, f32::NAN);
    app.update();
    let render = app.get_sub_app(RenderApp).unwrap();
    let snapshot = render.world.resource::<CameraQueueSnapshot>();
    assert!(snapshot.cameras.is_ready());
    assert_eq!(snapshot.cameras.len(), 1);
    assert!(snapshot.cameras.get(invalid).is_none());
    assert_eq!(snapshot.cameras.iter().next().unwrap().queues.len(), 1);
    assert_eq!(
        render
            .world
            .resource::<RenderFrameStats>()
            .view_queue_rejection,
        None
    );
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
        .read_texture_rgba16f(target.texture(), SIZE, SIZE)
        .expect("visible mesh HDR target readback must complete");
    assert_eq!(pixels.len(), (SIZE * SIZE) as usize);
    let pixel = |x: u32, y: u32| pixels[(y * SIZE + x) as usize];
    assert_eq!(
        pixel(SIZE / 2, SIZE / 2),
        [1.0; 4],
        "the visible mesh was drawn (the spine's flat pipeline)"
    );
    assert_eq!(
        pixel(0, 0),
        [0.0, 0.0, 0.0, 1.0],
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
    use prism_render_visibility::{GeometryLod, GeometryLodChain};

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
    app.add_plugins(crate::visibility::PrismVisibilityPlugin);

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
    {
        let scene = app
            .world_mut()
            .resource_mut::<crate::visibility::PrismRenderScene>();
        for (geometry, material) in [
            (near_geometry, near_record.clone()),
            (far_geometry, far_record.clone()),
        ] {
            scene.insert_geometry(GeometryLodChain {
                geometry,
                lods: vec![GeometryLod {
                    level: 0,
                    screen_error: 0.0,
                    resident: true,
                    fallback: true,
                }],
            });
            scene.insert_material(material);
        }
    }
    app.world_mut().spawn((
        crate::visibility::PrismCamera::new(prism_math::Mat4::IDENTITY, [0, 0, SIZE, SIZE]),
        crate::visibility::PrismFrustum(prism_math::Frustum::from_planes([
            prism_math::Plane::new(prism_math::Vec3::X, 10.0),
            prism_math::Plane::new(-prism_math::Vec3::X, 10.0),
            prism_math::Plane::new(prism_math::Vec3::Y, 10.0),
            prism_math::Plane::new(-prism_math::Vec3::Y, 10.0),
            prism_math::Plane::new(-prism_math::Vec3::Z, 1.0),
            prism_math::Plane::new(prism_math::Vec3::Z, 100.0),
        ])),
        prism_transform::GlobalTransform::from_transform(&prism_transform::Transform::from_xyz(
            0.0, 0.0, 1.5,
        )),
    ));
    for (geometry, material, depth) in [
        (near_geometry, near_material, 0.9),
        (far_geometry, far_material, 0.1),
    ] {
        app.world_mut().spawn((
            crate::visibility::PrismRenderable::new(geometry, material),
            prism_transform::ecs::LocalBounds(prism_math::Aabb3::from_center_half_extents(
                prism_math::Vec3::new(0.0, 0.0, depth),
                prism_math::Vec3::new(0.5, 0.5, 0.01),
            )),
            prism_transform::GlobalTransform::IDENTITY,
            crate::visibility::PrismViewVisibility::default(),
        ));
    }

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
                        shader_package: None,
                    }],
                )
                .unwrap();
        }
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
        .read_texture_rgba16f(target.texture(), SIZE, SIZE)
        .expect("depth-order HDR target readback must complete");
    assert_eq!(pixels.len(), (SIZE * SIZE) as usize);
    let centre = (SIZE / 2 * SIZE + SIZE / 2) as usize;
    assert_eq!(
        pixels[centre],
        [1.0, 0.0, 0.0, 1.0],
        "the nearer quad survives the later, farther draw"
    );
}
