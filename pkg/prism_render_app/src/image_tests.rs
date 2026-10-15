//! Behavioural tests for the prepared image, plus a real-device byte check.

use alloc::vec;
use alloc::vec::Vec;

use prism_image::{Image, ImageArtifact};
use prism_render_driver::{
    DeviceDescriptor, Extent3d, Instance, RenderQueue, RequestAdapterOptions, SamplerDescriptor,
    TextureFormat,
};
use prism_render_driver_null::NullInstance;

use crate::image::{prepare_image, GpuImage};

fn null_device() -> (prism_render_driver_null::NullDevice, impl RenderQueue) {
    let options = RequestAdapterOptions::default();
    let adapter = NullInstance::new()
        .request_adapter(&options)
        .expect("the null backend always has an adapter");
    NullInstance::new()
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null device always comes up")
}

/// A `width`×`height` image whose texels encode their own coordinates, so a
/// readback can be checked texel by texel.
fn coordinated_image(width: u32, height: u32) -> Image {
    coordinated_image_with_sampler(width, height, SamplerDescriptor::default())
}

fn coordinated_image_with_sampler(width: u32, height: u32, sampler: SamplerDescriptor) -> Image {
    let mut data = Vec::new();
    for y in 0..height {
        for x in 0..width {
            data.extend_from_slice(&[x as u8, y as u8, 0x40, 0xFF]);
        }
    }
    Image::with_sampler(
        Extent3d::new_2d(width, height),
        TextureFormat::Rgba8Unorm,
        data,
        sampler,
    )
    .expect("the fixture image is valid")
}

#[test]
fn preparing_an_image_creates_its_texture_view_and_sampler() {
    let (device, queue) = null_device();
    let image = coordinated_image(4, 3);

    let prepared = prepare_image(&device, &queue, &image, None);

    assert_eq!(prepared.extent, Extent3d::new_2d(4, 3));
    assert_eq!(prepared.format, TextureFormat::Rgba8Unorm);
    assert_eq!(prepared.bytes_per_row(), 16);
    assert_eq!(
        device.live_resource_count(),
        3,
        "a texture, its view, and the sampler"
    );
}

/// A re-prepare always uploads into a private texture, then publishes it and
/// releases the previous texture/view. An unchanged sampler can be shared.
#[test]
fn a_re_prepare_publishes_a_private_texture_before_releasing_the_previous_one() {
    let (device, queue) = null_device();
    let first = prepare_image(&device, &queue, &coordinated_image(4, 3), None);
    let first_texture = first.texture;
    let first_view = first.view;
    let first_sampler = first.sampler;

    // Same extents and format: the upload still targets a private texture, so
    // a failed upload cannot mutate the last-known-good representation.
    let second = prepare_image(&device, &queue, &coordinated_image(4, 3), Some(&first));
    assert_ne!(second.texture, first_texture, "the texture is replaced");
    assert_ne!(second.view, first_view, "the view follows the replacement");
    assert_eq!(second.sampler, first_sampler, "the sampler is reused");
    assert_eq!(device.live_resource_count(), 3, "nothing new was allocated");

    // Different extents follow the same publish-then-release ordering.
    let third = prepare_image(&device, &queue, &coordinated_image(8, 1), Some(&second));
    assert_ne!(third.texture, second.texture);
    assert_ne!(third.view, second.view);
    assert_eq!(
        device.live_resource_count(),
        3,
        "the old texture and view were released, not leaked"
    );

    let changed_sampler = coordinated_image_with_sampler(8, 1, SamplerDescriptor::linear_repeat());
    let fourth = prepare_image(&device, &queue, &changed_sampler, Some(&third));
    assert_ne!(fourth.sampler, third.sampler, "the sampler is replaced");
    assert_eq!(
        device.live_resource_count(),
        3,
        "the old sampler was released"
    );
}

#[test]
fn unloading_an_image_releases_every_object() {
    let (device, queue) = null_device();
    let image = coordinated_image(2, 2);
    let prepared = prepare_image(&device, &queue, &image, None);
    assert_eq!(device.live_resource_count(), 3);

    <GpuImage as crate::render_asset::RenderAsset>::unload(&prepared, &device);
    assert_eq!(device.live_resource_count(), 0);
}

/// The prepared image is *sampleable* on hardware: the frame uploads it, a
/// draw samples it through its own sampler, and the rendered pixels are the
/// image's texels — the property a material actually needs (an upload that only
/// round-tripped through a copy would prove less).
#[test]
fn an_uploaded_image_is_sampleable_on_a_gpu() {
    use crate::render_asset::{register_render_assets, RenderAssets};
    use crate::RenderAppConfig;
    use prism_render_driver::{
        BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
        BindingResource, BindingType, ColorTargetState, FragmentState, MultisampleState,
        PipelineLayoutDescriptor, PrimitiveState, RenderDevice, RenderPipelineDescriptor,
        SamplerBindingType, ShaderModuleDescriptor, ShaderStages, TextureFormat, TextureSampleType,
        TextureUsages, TextureViewDimension, VertexState,
    };
    use prism_render_driver_wgpu::WgpuInstance;
    use prism_render_graph::{ImportedTexture, PassFlags, RenderGraph, TextureDesc};

    const WIDTH: u32 = 5;
    const HEIGHT: u32 = 3;
    let extent = Extent3d::new_2d(WIDTH, HEIGHT);
    if prism_render_runtime::RenderRuntime::<WgpuInstance>::wgpu(extent).is_err() {
        assert!(
            !crate::tests::require_gpu(),
            "PRISM_REQUIRE_GPU is set but no adapter was found"
        );
        return;
    }

    let mut app = prism_app::App::new();
    app.add_plugins(crate::PrismRenderPlugin::with_config(
        WgpuInstance::new(),
        RenderAppConfig::new(extent),
    ));
    register_render_assets::<WgpuInstance, GpuImage>(&mut app);

    let coordinated = coordinated_image(WIDTH, HEIGHT);
    let image = ImageArtifact {
        width: WIDTH,
        height: HEIGHT,
        rgba8: coordinated.data().to_vec(),
    }
    .to_image()
    .expect("canonical image artifact must decode to the runtime image");
    let expected = image.data().to_vec();
    let mut images = prism_asset::Assets::new();
    let id = images.insert(image).id();
    app.world_mut().insert_resource(images);

    // The frame uploads the image.
    app.update();

    let render = app
        .get_sub_app(crate::RenderApp)
        .expect("the render sub-app");
    let prepared = render
        .world
        .resource::<RenderAssets<GpuImage>>()
        .get(id)
        .expect("the frame prepared the image");
    let runtime = render
        .world
        .get_resource::<crate::RenderRuntimeResource<WgpuInstance>>()
        .expect("the render runtime");
    let device = runtime.runtime().device();

    // A layout for the sampled texture and its sampler.
    let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("image-sample-layout".into()),
        entries: vec![
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float,
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("image-sample-group".into()),
        layout: Some(layout),
        entries: vec![
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(prepared.view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(prepared.sampler),
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("image-sample-pipeline-layout".into()),
        bind_group_layouts: vec![layout],
        push_constant_ranges: vec![],
    });
    let module = device.create_shader_module(&ShaderModuleDescriptor::wgsl(
        "@group(0) @binding(0) var tex: texture_2d<f32>;\n\
         @group(0) @binding(1) var samp: sampler;\n\
         @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {\n\
             // Two front-facing (counter-clockwise) triangles covering the target.\n\
             var corners = array<vec2<f32>, 6>(\n\
                 vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),\n\
                 vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),\n\
             );\n\
             let corner = corners[index];\n\
             return vec4<f32>(corner, 0.0, 1.0);\n\
         }\n\
         @fragment fn fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {\n\
             let uv = position.xy / vec2<f32>(5.0, 3.0);\n\
             return textureSample(tex, samp, uv);\n\
         }",
    ));
    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("image-sample".into()),
        layout: Some(pipeline_layout),
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
    });

    let target_texture = device.create_texture(&prism_render_driver::TextureDescriptor::new_2d(
        extent,
        TextureFormat::Rgba8Unorm,
        TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
    ));
    let target_view = device.create_texture_view(target_texture, &Default::default());
    let mut graph = RenderGraph::new(extent);
    let color = graph.import_texture(
        "sample-target",
        TextureDesc::color(TextureFormat::Rgba8Unorm).with_usage(TextureUsages::COPY_SRC),
        ImportedTexture {
            texture: target_texture,
            default_view: target_view,
            entry_state: prism_render_driver::TextureState::initial(),
            exit_state: prism_render_driver::TextureState::copy_src(),
            mip_levels: 1,
            array_layers: 1,
        },
    );
    graph.add_raster_pass("sample", PassFlags::SIDE_EFFECT, move |builder| {
        let _ = builder.color_attachment(color);
        move |ctx| {
            ctx.set_render_pipeline(pipeline);
            ctx.set_render_bind_group(0, bind_group, Vec::new());
            ctx.draw(0..6, 0..1);
        }
    });
    let queue = runtime.runtime().queue();
    graph.execute(device, queue).expect("the pass records");
    assert!(runtime.runtime().wait_idle(), "the draw completes");

    assert_eq!(
        queue.read_texture_rgba8(target_texture, WIDTH, HEIGHT),
        expected,
        "the sampled draw is the image's texels, texel for texel"
    );
}
