//! Behavioural tests for the prepared mesh, plus a real-device byte check.

use alloc::vec;
use alloc::vec::Vec;

use prism_mesh::{Indices, Mesh, MeshVertexAttribute, VertexAttributeValues};
use prism_render_driver::{
    DeviceDescriptor, IndexFormat, Instance, PrimitiveTopology, RequestAdapterOptions, VertexFormat,
};
use prism_render_driver_null::{NullDevice, NullInstance};
use prism_asset::AssetId;

use crate::render_asset::{PrepareAssetError, RenderAsset};

use super::{RenderMesh, RenderMeshBufferInfo};

const POSITION: MeshVertexAttribute =
    MeshVertexAttribute::new("Vertex_Position", VertexFormat::Float32x3, 0);

/// A quad of `scale` units: positions only, so a test can change the byte size
/// by changing `scale`'s vertex count.
fn quad(vertex_count: usize, indexed: bool) -> Mesh {
    let positions: Vec<[f32; 3]> = (0..vertex_count)
        .map(|index| [index as f32, 0.0, 0.5])
        .collect();
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList)
        .with_attribute(POSITION, VertexAttributeValues::Float32x3(positions))
        .expect("positions fit");
    if indexed {
        mesh.with_indices(Indices::Uint16(vec![0, 1, 2, 2, 3, 0]));
    }
    mesh
}

fn null_device() -> (NullDevice, impl prism_render_driver::RenderQueue) {
    let options = RequestAdapterOptions::default();
    let adapter = NullInstance::new()
        .request_adapter(&options)
        .expect("the null backend always has an adapter");
    NullInstance::new()
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null device always comes up")
}

/// Preparing a mesh on the null device creates one buffer per kind, with the
/// counts and layout a draw needs.
#[test]
fn preparing_a_mesh_creates_its_buffers() {
    let (device, queue) = null_device();
    let mesh = quad(4, true);
    let prepared = RenderMesh::prepare(&mesh, AssetId::new(prism_asset::AssetIndex::from_parts(0, 1)), &device, &queue, None)
        .expect("a valid mesh prepares");

    assert_eq!(prepared.vertex_count, 4);
    assert!(prepared.is_indexed());
    assert_eq!(
        prepared.buffer_info,
        RenderMeshBufferInfo::Indexed {
            index_format: IndexFormat::Uint16,
            index_count: 6,
        }
    );
    assert_eq!(prepared.element_count(), 6, "the draw covers the indices");
    assert_eq!(prepared.primitive_topology(), PrimitiveTopology::TriangleList);
    assert_eq!(
        prepared.layout.array_stride, 12,
        "the layout is the mesh's"
    );
    assert!(prepared.bounds.is_some(), "the quad's positions gave bounds");
    assert_eq!(
        device.live_resource_count(),
        2,
        "one vertex buffer and one index buffer"
    );
}

/// A non-indexed mesh gets no index buffer, and the draw covers its vertices.
#[test]
fn a_non_indexed_mesh_has_no_index_buffer() {
    let (device, queue) = null_device();
    let mesh = quad(3, false);
    let prepared =
        RenderMesh::prepare(&mesh, AssetId::new(prism_asset::AssetIndex::from_parts(0, 1)), &device, &queue, None)
            .expect("a valid mesh prepares");

    assert!(!prepared.is_indexed());
    assert_eq!(prepared.buffer_info, RenderMeshBufferInfo::NonIndexed);
    assert_eq!(prepared.element_count(), 3);
    assert_eq!(prepared.buffers.indices, None);
    assert_eq!(device.live_resource_count(), 1, "vertices only");
}

/// A mesh that changed without outgrowing its buffers keeps them: re-preparing
/// something as common as a morph or a rebuild must not churn GPU objects.
#[test]
fn a_re_prepare_reuses_buffers_that_still_fit() {
    let (device, queue) = null_device();
    let id = AssetId::new(prism_asset::AssetIndex::from_parts(0, 1));
    let first = RenderMesh::prepare(&quad(4, true), id, &device, &queue, None).expect("prepares");
    let first_vertices = first.buffers.vertices;
    let first_indices = first.buffers.indices;

    // Same vertex count, different positions: the buffers still fit.
    let moved = quad(4, true);
    let second =
        RenderMesh::prepare(&moved, id, &device, &queue, Some(&first)).expect("re-prepares");

    assert_eq!(second.buffers.vertices, first_vertices, "the vertex buffer is reused");
    assert_eq!(second.buffers.indices, first_indices, "the index buffer is reused");
    assert_eq!(
        device.live_resource_count(),
        2,
        "re-preparing allocated nothing"
    );
}

/// A mesh that outgrows its buffer, or stops being indexed, releases what it no
/// longer uses — the ownership rule the trait documents.
#[test]
fn a_re_prepare_releases_what_it_no_longer_uses() {
    let (device, queue) = null_device();
    let id = AssetId::new(prism_asset::AssetIndex::from_parts(0, 1));
    let first = RenderMesh::prepare(&quad(4, true), id, &device, &queue, None).expect("prepares");
    let first_indices = first.buffers.indices.expect("indexed").0;

    // Grow: the vertex buffer is replaced (and the old one destroyed), while the
    // index buffer still fits and is kept — only what outgrew is reallocated.
    let grown = quad(8, true);
    let second =
        RenderMesh::prepare(&grown, id, &device, &queue, Some(&first)).expect("re-prepares");
    assert_ne!(second.buffers.vertices, first.buffers.vertices);
    assert_eq!(second.vertex_count, 8);
    assert_eq!(
        second.buffers.indices.map(|(id, _)| id),
        Some(first_indices),
        "the index buffer still fits"
    );
    assert_eq!(
        device.live_resource_count(),
        2,
        "the old vertex buffer was released, not leaked"
    );

    // Drop the indices: the index buffer goes with them.
    let unindexed = quad(8, false);
    let third =
        RenderMesh::prepare(&unindexed, id, &device, &queue, Some(&second)).expect("re-prepares");
    assert_eq!(third.buffers.indices, None);
    assert_eq!(
        device.live_resource_count(),
        1,
        "the index buffer was released when the mesh stopped being indexed"
    );
    assert_eq!(
        third.buffers.vertices, second.buffers.vertices,
        "the vertex buffer was not disturbed by the indices going away"
    );
}

/// An unusable mesh is reported as unpreparable rather than retried forever, and
/// the previous representation survives.
#[test]
fn an_invalid_mesh_is_unpreparable_not_retried() {
    let (device, queue) = null_device();
    let id = AssetId::new(prism_asset::AssetIndex::from_parts(0, 1));
    let good = quad(4, true);
    let prepared = RenderMesh::prepare(&good, id, &device, &queue, None).expect("prepares");

    let mut bad = quad(4, true);
    bad.with_indices(Indices::Uint16(vec![0, 1, 9]));
    let refused = RenderMesh::prepare(&bad, id, &device, &queue, Some(&prepared));
    assert_eq!(refused.err(), Some(PrepareAssetError::Unpreparable { id }));
    assert_eq!(
        device.live_resource_count(),
        2,
        "the refused prepare touched nothing"
    );
}

/// Unloading releases every buffer the representation owns.
#[test]
fn unload_releases_every_buffer() {
    let (device, queue) = null_device();
    let id = AssetId::new(prism_asset::AssetIndex::from_parts(0, 1));
    let prepared = RenderMesh::prepare(&quad(4, true), id, &device, &queue, None).expect("prepares");
    assert_eq!(device.live_resource_count(), 2);

    RenderMesh::unload(&prepared, &device);
    assert_eq!(device.live_resource_count(), 0);
}

/// The family works through the frame, like any other render asset: the mesh is
/// held in the simulation's arena, carried across by the extract, and prepared
/// into the render world's table.
#[test]
fn a_mesh_family_prepares_through_the_frame() {
    use crate::render_asset::{RenderAssets, register_render_assets};

    let mut app = prism_app::App::new();
    app.add_plugins(crate::PrismRenderPlugin::new(NullInstance::new()));
    register_render_assets::<NullInstance, RenderMesh>(&mut app);

    let mut meshes = prism_asset::Assets::new();
    let id = meshes.insert(quad(4, true)).id();
    app.world_mut().insert_resource(meshes);

    app.update();

    let render = app
        .get_sub_app(crate::RenderApp)
        .expect("the render sub-app");
    let table = render.world.resource::<RenderAssets<RenderMesh>>();
    let prepared = table.get(id).expect("the frame prepared the mesh");
    assert_eq!(prepared.vertex_count, 4);
    assert!(prepared.is_indexed());
}

/// On a real device the prepared mesh *draws*: a quad whose positions live in the
/// uploaded vertex buffer, bound with the layout the mesh derived, rasterizes
/// where the positions say. A wrong interleaving or a wrong offset would put the
/// triangles somewhere else (or nowhere), so the pixels are the proof that the
/// bytes and the layout agree.
#[test]
fn a_prepared_mesh_draws_with_its_layout_on_a_gpu() {
    use prism_render_driver::{
        BufferState, ColorTargetState, Extent3d, FragmentState, MultisampleState, PrimitiveState,
        RenderDevice, RenderPipelineDescriptor, ShaderModuleDescriptor, TextureDescriptor,
        TextureFormat, TextureState, TextureUsages, VertexState,
    };
    use prism_render_driver_wgpu::WgpuInstance;
    use prism_render_graph::{
        BufferDesc, ImportedBuffer, ImportedTexture, PassFlags, RenderGraph, TextureDesc,
    };

    const SIZE: u32 = 8;
    let extent = Extent3d::new_2d(SIZE, SIZE);
    let Ok(runtime) = prism_render_runtime::RenderRuntime::<WgpuInstance>::wgpu(extent) else {
        assert!(
            !crate::tests::require_gpu(),
            "PRISM_REQUIRE_GPU is set but no adapter was found"
        );
        return;
    };
    let device = runtime.device();
    let queue = runtime.queue();

    // A clip-space quad: positions are the vertex data under test, so the draw
    // needs no transform and every pixel difference is the buffer's doing.
    let mesh = Mesh::new(PrimitiveTopology::TriangleList)
        .with_attribute(
            POSITION,
            VertexAttributeValues::Float32x3(vec![
                [-0.5, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [0.5, 0.5, 0.0],
                [-0.5, 0.5, 0.0],
            ]),
        )
        .expect("positions fit")
        .indexed(Indices::Uint16(vec![0, 1, 2, 0, 2, 3]));
    let prepared = RenderMesh::prepare(
        &mesh,
        AssetId::new(prism_asset::AssetIndex::from_parts(0, 1)),
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
        label: Some("mesh-gpu-test".into()),
        layout: None,
        vertex: VertexState {
            module,
            entry_point: "vs".into(),
            // The layout the *mesh* derived, not one the test wrote: this is what
            // makes the draw depend on the mesh's interleaving being right.
            buffers: vec![prepared.layout.clone()],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            module,
            entry_point: "fs".into(),
            targets: vec![Some(ColorTargetState::opaque(TextureFormat::Rgba8Unorm))],
        }),
    });

    let target_texture = device.create_texture(&TextureDescriptor::new_2d(
        extent,
        TextureFormat::Rgba8Unorm,
        TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
    ));
    let target_view = device.create_texture_view(target_texture, &Default::default());
    let (vertices, indices) = (
        prepared.buffers.vertices,
        prepared.buffers.indices.expect("the quad is indexed").0,
    );
    let vertex_bytes = mesh.vertex_buffer_data().len() as u64;
    let index_bytes = mesh.indices().expect("indexed").to_bytes().len() as u64;

    let mut graph = RenderGraph::new(extent);
    let target = graph.import_texture(
        "gpu-test-target",
        TextureDesc::color(TextureFormat::Rgba8Unorm).with_usage(TextureUsages::COPY_SRC),
        ImportedTexture {
            texture: target_texture,
            default_view: target_view,
            entry_state: TextureState::initial(),
            exit_state: TextureState::copy_src(),
            mip_levels: 1,
            array_layers: 1,
        },
    );
    let vertex_handle = graph.import_buffer(
        "mesh-vertices",
        BufferDesc::new(vertex_bytes, prism_render_driver::BufferUsages::VERTEX),
        ImportedBuffer {
            buffer: vertices,
            entry_state: BufferState::initial(),
        },
    );
    let index_handle = graph.import_buffer(
        "mesh-indices",
        BufferDesc::new(index_bytes, prism_render_driver::BufferUsages::INDEX),
        ImportedBuffer {
            buffer: indices,
            entry_state: BufferState::initial(),
        },
    );
    graph.add_raster_pass("mesh-draw", PassFlags::SIDE_EFFECT, move |builder| {
        let _ = builder.color_attachment(target);
        let _ = builder.read_vertex(vertex_handle);
        let _ = builder.read_index(index_handle);
        move |ctx| {
            ctx.set_render_pipeline(pipeline);
            let vertices = ctx.buffer(vertex_handle);
            ctx.set_vertex_buffer(0, vertices, 0);
            let indices = ctx.buffer(index_handle);
            ctx.set_index_buffer(indices, IndexFormat::Uint16, 0);
            ctx.draw_indexed(0..6, 0, 0..1);
        }
    });

    graph.execute(device, queue).expect("the pass records");
    assert!(runtime.wait_idle(), "the submitted draw completes");

    let pixels = queue.read_texture_rgba8(target_texture, SIZE, SIZE);
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
        "the quad rasterized where its uploaded vertices and derived layout say"
    );
    assert_eq!(
        pixel(0, 0),
        [0, 0, 0, 255],
        "a pixel outside the quad keeps the cleared target"
    );
    assert_eq!(pixel(SIZE - 1, SIZE - 1), [0, 0, 0, 255]);
}
