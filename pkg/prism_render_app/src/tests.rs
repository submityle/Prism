//! Tests for the render sub-app seam.
//!
//! The headless tests run on the null backend, so they need no window and no
//! GPU: they drive the full plugin → extract → render-schedule → frame-graph
//! path and assert on what the render world recorded. The wgpu test is gated on
//! a real adapter and reads the rendered frame back, so the same seam is proven
//! on hardware rather than only in the ring buffer.

use prism_app::App;
use prism_ecs::schedule::Schedules;
use prism_render_driver::{Extent3d, ImageDataLayout, RenderQueue, TextureWrite};
use prism_render_driver_null::NullInstance;
use prism_render_runtime::RenderRuntime;

use crate::camera::{RenderCamera, RenderViewport};
use crate::config::RenderAppConfig;
use crate::label::{Render, RenderApp};
use crate::plugin::PrismRenderPlugin;
use crate::record::RenderRecord;
use crate::runtime_resource::RenderRuntimeResource;
use crate::stats::RenderFrameStats;
use crate::target::RenderTarget;

use crate::sets::{RenderPhase, RenderSystems};
use crate::views::RenderViews;
use crate::pipeline_cache::{PipelineCache, PipelineKey, hash_vertex_layouts};
use crate::queue::{PhaseItem, RenderQueues};
use crate::queue::DrawRequest;
use crate::queue::phase_for_render_class;
use prism_render_material::MaterialRenderClass;
use crate::draw::{DrawFunctionId, DrawFunctions};
use crate::mesh_pipeline::{MeshPipelineKey, MeshPipelineSpecializer};
use crate::texture_cache::{TextureCache, TextureKey};
use prism_ecs::prelude::{ResMut, Resource};
use prism_ecs::schedule::IntoSystemConfigs;
use prism_render_architecture::view_family::ViewKind;
use prism_render_architecture::abi::GenerationalHandle;
use prism_render_driver::{
    DeviceDescriptor, Instance, MultisampleState, PrimitiveState, RenderDevice,
    RenderPipelineDescriptor, RequestAdapterOptions, PrimitiveTopology, ShaderModuleDescriptor,
    TextureDescriptor, TextureFormat, TextureUsages, VertexState,
};

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
    let descriptor = RenderPipelineDescriptor {
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
        &[prism_render_driver::VertexFormat::Float32x3, prism_render_driver::VertexFormat::Float32x2],
    );
    let layout_hash = hash_vertex_layouts(core::slice::from_ref(&layout));

    let mut cache = PipelineCache::new();
    let key = PipelineKey::new(
        7,
        Some(TextureFormat::Rgba8Unorm),
        None,
        1,
        layout_hash,
        PrimitiveTopology::TriangleList,
        0xABCD,
    );
    let first = cache.get_or_create(&device, key, &descriptor);
    let second = cache.get_or_create(&device, key, &descriptor);
    assert_eq!(first, second, "the same key returns the same pipeline");
    assert_eq!(cache.len(), 1, "the pipeline is built once");

    let other = PipelineKey::new(
        8,
        Some(TextureFormat::Rgba8Unorm),
        None,
        1,
        layout_hash,
        PrimitiveTopology::TriangleList,
        0xABCD,
    );
    let distinct = cache.get_or_create(&device, other, &descriptor);
    assert_ne!(distinct, first, "a different key compiles a new pipeline");
    assert_eq!(cache.len(), 2);

    // The key must separate pipelines that differ only in topology or shader:
    // keying on neither aliased two different PSOs onto one cache entry.
    let other_topology = PipelineKey::new(
        7,
        Some(TextureFormat::Rgba8Unorm),
        None,
        1,
        layout_hash,
        PrimitiveTopology::TriangleStrip,
        0xABCD,
    );
    assert_ne!(other_topology, key, "topology is part of the identity");
    let other_shader = PipelineKey::new(
        7,
        Some(TextureFormat::Rgba8Unorm),
        None,
        1,
        layout_hash,
        PrimitiveTopology::TriangleList,
        0x1234,
    );
    assert_ne!(other_shader, key, "the shader identity is part of the key");

    // A layout with the same total stride but different attributes must not
    // hash equal: a single stride would alias these two vertex stages.
    let same_stride_other_attrs = prism_render_driver::VertexBufferLayout::packed(
        prism_render_driver::VertexStepMode::Vertex,
        0,
        &[prism_render_driver::VertexFormat::Float32x2, prism_render_driver::VertexFormat::Float32x3],
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
    assert!(cache.release(&device, &TextureKey(1)), "the last release evicts");
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
    let item = |order: u64| PhaseItem {
        pipeline,
        draw: DrawFunctionId(0),
        geometry,
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
    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("record-test".into()),
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

    // The recording context is a plain buffer here; production passes the
    // graph's pass recorder instead.
    let mut functions = DrawFunctions::<String>::new();
    let draw = functions.register(|item, out: &mut String| {
        out.push_str(&format!("{};", item.order));
    });

    let geometry = GenerationalHandle::new(2, 1);
    let material = GenerationalHandle::new(3, 1);
    let scene = GenerationalHandle::new(4, 1);
    let mut queues = RenderQueues::new();
    for order in [2u64, 1] {
        queues.push(
            RenderPhase::Opaque,
            PhaseItem {
                pipeline,
                draw,
                geometry,
                lod_level: 0,
                material,
                scene,
                order,
            },
        );
    }
    queues.sort();

    let mut recorded = String::new();
    queues.record(RenderPhase::Opaque, &mut recorded, &functions);
    assert_eq!(recorded, "1;2;", "items record in sorted order");

    // An unknown id is a no-op, never a panic.
    functions.call(DrawFunctionId(99), &queues.phase(RenderPhase::Opaque)[0], &mut recorded);
    assert_eq!(recorded, "1;2;");
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
    let (device, _queue) = NullInstance::new()
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
    registry.insert_mesh_class(
        5,
        MeshClass::new(vec![layout], PrimitiveTopology::TriangleList),
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
    assert_eq!(specializer.pipeline_count(), 1, "one pipeline specialized");
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
        .resolve(&item, None, 1, DrawFunctionId(0))
        .expect("a resident geometry + material resolves");
    assert_eq!(draw.phase, RenderPhase::Opaque, "the opaque family phases here");
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
            .resolve(&missing_material, None, 1, DrawFunctionId(0))
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
        class: 1,
        color_format: Some(TextureFormat::Rgba8Unorm),
        depth_format: None,
        sample_count: 1,
        topology: PrimitiveTopology::TriangleList,
        shader_id: 9,
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
    let request = |layout: prism_render_driver::VertexBufferLayout, order: u64| DrawRequest {
        phase: RenderPhase::Opaque,
        variant,
        layout: vec![layout],
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
        [request(layout_a.clone(), 0), request(layout_a, 1), request(layout_b, 2)],
        |_| descriptor.clone(),
    );

    assert_eq!(queues.phase(RenderPhase::Opaque).len(), 3, "every draw queues");
    assert_eq!(
        specializer.pipeline_count(),
        2,
        "two distinct layouts specialize two pipelines"
    );
}

/// Surface draws are phased by blend family only — the illumination axis is a
/// response, not a phase.
#[test]
fn material_render_class_selects_the_phase() {
    use MaterialRenderClass::{
        Additive, Decal, Hair, Masked, MaskedTwoSided, Opaque, OpaqueTwoSided, Transmissive,
        Transparent, Volume, Water,
    };
    for class in [Opaque, OpaqueTwoSided, Masked, MaskedTwoSided, Hair, Water] {
        assert_eq!(
            phase_for_render_class(class),
            RenderPhase::Opaque,
            "{class:?} is opaque-family"
        );
    }
    for class in [Transmissive, Transparent, Additive, Decal] {
        assert_eq!(
            phase_for_render_class(class),
            RenderPhase::Transparent,
            "{class:?} is transparent-family"
        );
    }
    assert_eq!(
        phase_for_render_class(Volume),
        RenderPhase::Custom,
        "a volume is a project-injected pass, not a surface draw"
    );
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
        class: 3,
        color_format: Some(TextureFormat::Rgba8Unorm),
        depth_format: None,
        sample_count: 1,
        topology: PrimitiveTopology::TriangleList,
        shader_id: 0x55,
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
    let a1 = specializer.specialize(&device, variant, core::slice::from_ref(&layout_a), &descriptor);
    let a2 = specializer.specialize(&device, variant, core::slice::from_ref(&layout_a), &descriptor);
    assert_eq!(a1, a2, "the same variant + layout reuses the pipeline");
    assert_eq!(specializer.pipeline_count(), 1);

    let b = specializer.specialize(&device, variant, core::slice::from_ref(&layout_b), &descriptor);
    assert_ne!(a1, b, "a different layout specializes a new pipeline");
    assert_eq!(specializer.pipeline_count(), 2);

    assert!(specializer.release(&device, variant, core::slice::from_ref(&layout_b)));
    assert_eq!(specializer.pipeline_count(), 1);
    specializer.clear(&device);
    assert_eq!(specializer.pipeline_count(), 0);
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
    let entry = views.registry().get(handle).expect("the view is registered");
    assert_eq!(entry.kind, ViewKind::Main, "the primary view is the Main kind");
    assert!(entry.enabled, "a non-empty frame enables the view");

    // A zero-sized frame disables the view but keeps its handle.
    app.world_mut()
        .resource_mut::<RenderCamera>()
        .viewport = RenderViewport::new(0, 0);
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

/// The render stages and phases order systems whether or not the registering
/// plugin knows about the others: `Prepare` precedes `Queue`, and the `Opaque`
/// phase records after `Queue`.
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
        schedule.add_systems(mark_opaque.in_set(RenderPhase::Opaque));
        schedule.add_systems(mark_prepare.in_set(RenderSystems::Prepare));
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
        ["prepare", "queue", "opaque"],
        "Prepare runs before Queue, and the Opaque phase records after Queue"
    );
}

/// The frame count the headless tests drive.
const FRAMES: u64 = 5;

/// A small, non-square frame so a swapped axis cannot pass by accident.
const EXTENT: Extent3d = Extent3d::new_2d(64, 48);

/// An app with the plugin built on the headless null backend.
fn app_on_null_backend(config: RenderAppConfig) -> App {
    let mut app = App::new();
    app.add_plugins(PrismRenderPlugin::with_config(NullInstance::new(), config));
    app
}

/// Whether a missing adapter must fail a GPU-gated test instead of skipping it.
fn require_gpu() -> bool {
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
        stats.passes_declared, FRAMES,
        "each recorded frame declares exactly one pass"
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
    *app.world_mut().resource_mut::<RenderCamera>() = RenderCamera::new(resized);
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
    *app.world_mut().resource_mut::<RenderCamera>() = RenderCamera::new(RenderViewport::new(0, 0));
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
}
